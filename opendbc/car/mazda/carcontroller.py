from enum import StrEnum

import numpy as np

from opendbc.can import CANPacker
from opendbc.car import Bus, DT_CTRL, make_tester_present_msg, structs
from opendbc.car.lateral import apply_driver_steer_torque_limits
from opendbc.car.interfaces import CarControllerBase
from opendbc.car.mazda import mazdacan
from opendbc.car.mazda.longitudinal import (RADAR_ADDR, SESSION_DEFAULT, SESSION_PROGRAMMING,
                                            RadarSessionManager, RadarSessionState, create_radar_session_msg)
from opendbc.car.mazda.values import CarControllerParams, Buttons

from opendbc.sunnypilot.car.mazda.icbm import IntelligentCruiseButtonManagementInterface

VisualAlert = structs.CarControl.HUDControl.VisualAlert
LongCtrlState = structs.CarControl.Actuators.LongControlState

# Synthetic radar frames go to the car and to the camera; the panda only forwards
# received frames between those buses, not our own transmissions.
LONG_BUSES = (0, 2)

HOLD_CTRL_LATCH_FRAMES = int(CarControllerParams.HOLD_CTRL_LATCH_T / DT_CTRL)
HOLD_LATCH_FRAMES = int(CarControllerParams.HOLD_LATCH_T / DT_CTRL)
HOLD_PASSIVE_FRAMES = int(CarControllerParams.HOLD_PASSIVE_T / DT_CTRL)
RESUME_RELEASE_FRAMES = int(CarControllerParams.RESUME_RELEASE_T / DT_CTRL)
RESUME_REACTIVATE_FRAMES = int(CarControllerParams.RESUME_REACTIVATE_T / DT_CTRL)
RESUME_UNLATCH_FRAMES = int(CarControllerParams.RESUME_UNLATCH_T / DT_CTRL)


class StopGoState(StrEnum):
  CRUISING = "cruising"
  STOPPING = "stopping"
  HOLD = "hold"
  HOLD_LATCHED = "hold latched"
  HOLD_PASSIVE = "hold passive"
  RESUMING = "resuming"


class StopAndGoStateMachine:
  """Replays stock MRCC's stop-and-go sequence: ramp to a stop, hold at -1.024 m/s2,
  relax to the latched hold, drop to the passive hold on long stops, and release
  through a short brake-release window on resume."""

  def __init__(self):
    self.state = StopGoState.CRUISING
    self.hold_frames = 0
    self.release_frames = 0
    self.reactivate_frames = 0
    self.unlatch_frames = 0

  def update(self, long_active: bool, stopping: bool, standstill: bool,
             resume_pressed: bool = False, virtual_resume: bool = False,
             gas_override: bool = False) -> StopGoState:
    if not long_active:
      self.state = StopGoState.CRUISING
      self.hold_frames = 0
      self.release_frames = 0
      self.reactivate_frames = 0
      self.unlatch_frames = 0
      return self.state

    if self.state == StopGoState.CRUISING:
      if stopping:
        self.state = StopGoState.STOPPING

    elif self.state == StopGoState.STOPPING:
      if standstill:
        self.state = StopGoState.HOLD
        self.hold_frames = 0
      elif not stopping:
        self.state = StopGoState.CRUISING

    elif self.state in (StopGoState.HOLD, StopGoState.HOLD_LATCHED, StopGoState.HOLD_PASSIVE):
      self.hold_frames += 1
      # Stock's earliest observed hold release comes after the 2 s CRZ_CTRL latch, so a
      # physical RES waits for it. A virtual resume additionally waits for the relaxed
      # hold command so a transient plan flicker cannot release the hold early.
      ctrl_latched = self.hold_frames >= HOLD_CTRL_LATCH_FRAMES
      resume = (resume_pressed and ctrl_latched) or (virtual_resume and self.state != StopGoState.HOLD)
      if gas_override or resume:
        # stock briefly re-raises the latched profile when resuming out of a passive hold
        self.reactivate_frames = RESUME_REACTIVATE_FRAMES if self.state == StopGoState.HOLD_PASSIVE else 0
        self.release_frames = RESUME_RELEASE_FRAMES
        self.unlatch_frames = RESUME_UNLATCH_FRAMES
        self.state = StopGoState.RESUMING
      elif self.hold_frames >= HOLD_PASSIVE_FRAMES:
        self.state = StopGoState.HOLD_PASSIVE
      elif self.hold_frames >= HOLD_LATCH_FRAMES:
        self.state = StopGoState.HOLD_LATCHED

    elif self.state == StopGoState.RESUMING:
      if self.reactivate_frames > 0:
        self.reactivate_frames -= 1
      elif self.unlatch_frames > 0:
        self.unlatch_frames -= 1
      if resume_pressed or virtual_resume or gas_override:
        # keep the brake released while the resume request holds
        self.release_frames = RESUME_RELEASE_FRAMES
      else:
        self.release_frames -= 1
      if self.release_frames <= 0:
        # re-hold if the car never moved, otherwise hand control back to the plan
        self.state = StopGoState.HOLD if standstill else StopGoState.CRUISING
        self.hold_frames = 0

    return self.state

  @property
  def stop_bits(self) -> bool:
    # CRZ_INFO stop flags are held through the approach and the strong hold, and clear
    # once the hold command relaxes
    return self.state in (StopGoState.STOPPING, StopGoState.HOLD)

  @property
  def resume_unlatching(self) -> bool:
    return self.state == StopGoState.RESUMING and self.reactivate_frames == 0 and self.unlatch_frames > 0

  @property
  def acc_active_2(self) -> bool:
    # stock drops ACC_ACTIVE_2 together with the command relax at the hold latch
    return self.state not in (StopGoState.HOLD_LATCHED, StopGoState.HOLD_PASSIVE)

  def radar_has_lead(self, lead_visible: bool) -> bool:
    return lead_visible or self.state in (StopGoState.STOPPING, StopGoState.HOLD,
                                          StopGoState.HOLD_LATCHED, StopGoState.HOLD_PASSIVE)

  def ctrl_phase(self, lead_visible: bool) -> int:
    # RADAR_LEAD_RELATIVE_DISTANCE: 1 cruise, 2 follow, 3 stop/hold, 4 hold-far. Stock holds a
    # constant stop phase through the whole hold and drops to follow on resume; without a real
    # lead distance we advertise the near stop phase (3) throughout the hold.
    if self.state == StopGoState.CRUISING:
      return 2 if lead_visible else 1
    return 3


class CarController(CarControllerBase, IntelligentCruiseButtonManagementInterface):
  def __init__(self, dbc_names, CP, CP_SP):
    CarControllerBase.__init__(self, dbc_names, CP, CP_SP)
    IntelligentCruiseButtonManagementInterface.__init__(self, CP, CP_SP)
    self.params = CarControllerParams(CP)
    self.apply_torque_last = 0
    self.packer = CANPacker(dbc_names[Bus.pt])
    self.brake_counter = 0
    self.stop_and_go = StopAndGoStateMachine()
    self.virtual_resume_latched = False
    self.long_counter = 0
    self.radar_counter = 0
    self.radar_session = RadarSessionManager()

  def update(self, CC, CC_SP, CS, now_nanos):
    can_sends = []

    apply_torque = 0

    # Speed-dependent STEER_MAX (CX-5 2022: 1200 below 32 mph, 800 above)
    if hasattr(self.params, 'STEER_MAX_LOOKUP'):
      steer_max = round(float(np.interp(CS.out.vEgoRaw, self.params.STEER_MAX_LOOKUP[0],
                                         self.params.STEER_MAX_LOOKUP[1])))
    else:
      steer_max = self.params.STEER_MAX

    if CC.latActive:
      # calculate steer and also set limits due to driver torque
      new_torque = int(round(CC.actuators.torque * steer_max))
      apply_torque = apply_driver_steer_torque_limits(new_torque, self.apply_torque_last,
                                                      CS.out.steeringTorque, self.params, steer_max)

    virtual_resume_sent = False
    if CC.cruiseControl.cancel:
      # If brake is pressed, let us wait >70ms before trying to disable crz to avoid
      # a race condition with the stock system, where the second cancel from openpilot
      # will disable the crz 'main on'. crz ctrl msg runs at 50hz. 70ms allows us to
      # read 3 messages and most likely sync state before we attempt cancel.
      self.brake_counter = self.brake_counter + 1
      if self.frame % 10 == 0 and not (CS.out.brakePressed and self.brake_counter < 7):
        # Cancel Stock ACC if it's enabled while OP is disengaged
        # Send at a rate of 10hz until we sync with stock ACC state
        can_sends.append(mazdacan.create_button_cmd(self.packer, self.CP, CS.crz_btns_counter, Buttons.CANCEL))
    else:
      self.brake_counter = 0
      if CC.cruiseControl.resume and self.frame % 5 == 0:
        # Mazda Stop and Go requires a RES button (or gas) press if the car stops more than 3 seconds
        # Send Resume button when planner wants car to move
        if not self.CP.openpilotLongitudinalControl:
          can_sends.append(mazdacan.create_button_cmd(self.packer, self.CP, CS.crz_btns_counter, Buttons.RESUME))
        elif CS.out.standstill:
          # with openpilot longitudinal the RES press asks the body ECU to leave its
          # standstill hold; only meaningful from a stop
          can_sends.append(mazdacan.create_button_cmd(self.packer, self.CP, CS.crz_btns_counter, Buttons.RESUME))
          virtual_resume_sent = True

    self.apply_torque_last = apply_torque

    if self.CP.openpilotLongitudinalControl:
      can_sends.extend(self.update_longitudinal(CC, CC_SP, CS, virtual_resume_sent))

    # send HUD alerts
    if self.frame % 50 == 0:
      ldw = CC.hudControl.visualAlert == VisualAlert.ldw
      steer_required = CC.hudControl.visualAlert == VisualAlert.steerRequired
      # TODO: find a way to silence audible warnings so we can add more hud alerts
      steer_required = steer_required and CS.lkas_allowed_speed
      can_sends.append(mazdacan.create_alert_command(self.packer, CS.cam_laneinfo, ldw, steer_required))

    # send steering command
    can_sends.append(mazdacan.create_steering_control(self.packer, self.CP,
                                                      self.frame, apply_torque, CS.cam_lkas))

    # Intelligent Cruise Button Management
    # Suppress ICBM CRZ_BTNS spam while cancel/resume are in flight or while the driver is
    # holding the wheel cancel button. Without this guard ICBM's interleaved cancel=0 frames
    # race the driver's cancel=1 frames on the bus and the body ECU drops the cancel intent.
    icbm_suppress = CC.cruiseControl.cancel or CC.cruiseControl.resume or CS.cancel_button == 1
    if not icbm_suppress:
      can_sends.extend(IntelligentCruiseButtonManagementInterface.update(self, CC_SP, CS, self.packer, self.frame, self.last_button_frame))

    new_actuators = CC.actuators.as_builder()
    new_actuators.torque = apply_torque / steer_max
    new_actuators.torqueOutputCan = apply_torque

    self.frame += 1
    return new_actuators, can_sends

  def update_longitudinal(self, CC, CC_SP, CS, virtual_resume_sent):
    can_sends = []

    # Radar session sequencing: hold off the teardown until the FSC's cold-boot
    # radar-presence check has cleared (carstate's settle gate), keep the radar in its
    # programming session while we own the bus, and on an onroad toggle-off return it
    # to the default session before card requests the process restart.
    stock_radar_alive = CS.stock_radar_silent_frames < CarControllerParams.STOCK_RADAR_ALIVE_FRAMES
    # never yank the radar out from under an active stock MRCC engagement (driver SET
    # before the gate passed on a warm boot): wait for the driver to disengage first
    stock_cruise_engaged = stock_radar_alive and CS.out.cruiseState.enabled
    teardown_ok = CS.radar_teardown_gate and not stock_cruise_engaged
    session_state = self.radar_session.update(teardown_ok, stock_radar_alive, CC_SP.radarHandBack)

    if self.frame % CarControllerParams.SESSION_STEP == 0:
      if session_state == RadarSessionState.SILENCING:
        can_sends.append(create_radar_session_msg(SESSION_PROGRAMMING))
      elif session_state == RadarSessionState.HANDBACK:
        can_sends.append(create_radar_session_msg(SESSION_DEFAULT))

    if session_state == RadarSessionState.SILENCED and self.frame % CarControllerParams.TESTER_PRESENT_STEP == 0:
      # keeps the radar in its diagnostic session, and with it the stock frames silenced
      can_sends.append(make_tester_present_msg(RADAR_ADDR, 0, suppress_response=True))

    # only trust a virtual resume once a RES frame has actually gone out on the bus
    if not CC.cruiseControl.resume or not CS.out.standstill:
      self.virtual_resume_latched = False
    elif virtual_resume_sent:
      self.virtual_resume_latched = True

    stopping = CC.actuators.longControlState == LongCtrlState.stopping
    gas_override = CC.cruiseControl.override or CS.out.gasPressed
    sm = self.stop_and_go
    state = sm.update(CC.longActive, stopping, CS.out.standstill,
                      resume_pressed=bool(CS.resume_button),
                      virtual_resume=self.virtual_resume_latched,
                      gas_override=gas_override)

    accel = 0.
    if CC.longActive:
      accel = float(np.clip(CC.actuators.accel, CarControllerParams.ACCEL_MIN, CarControllerParams.ACCEL_MAX))
      if state == StopGoState.HOLD:
        accel = CarControllerParams.ACCEL_HOLD
      elif state in (StopGoState.HOLD_LATCHED, StopGoState.HOLD_PASSIVE):
        accel = CarControllerParams.ACCEL_HOLD_LATCHED
      elif state == StopGoState.RESUMING:
        # brake-release window: let the car creep off the hold, never brake into it
        accel = max(accel, 0.)

    lead_visible = CC.hudControl.leadVisible
    if self.radar_session.radar_master and self.frame % CarControllerParams.RADAR_STEP == 0:
      synthetic_lead = CC.longActive and (lead_visible or state != StopGoState.CRUISING)
      for bus in LONG_BUSES:
        can_sends.extend(mazdacan.create_radar_frames(bus, self.radar_counter, synthetic_lead))
      self.radar_counter += 1

    if self.radar_session.radar_master and self.frame % CarControllerParams.LONG_STEP == 0:
      acc_available = CS.out.cruiseState.available
      # mirror the driver's distance setting on the dash; stock shows gap 2 by default
      gap = (int(CC.hudControl.leadDistanceBars) or 2) if (CC.longActive or acc_available) else 0
      if CC.longActive:
        has_lead = sm.radar_has_lead(lead_visible)
        phase = sm.ctrl_phase(lead_visible)
        acc_active_2 = sm.acc_active_2
      else:
        has_lead = False
        phase = 0
        acc_active_2 = False
      for bus in LONG_BUSES:
        can_sends.append(mazdacan.create_acc_command(self.packer, bus, self.long_counter, accel,
                                                     CC.longActive, acc_available,
                                                     stopping=sm.stop_bits, resume_unlatching=sm.resume_unlatching))
        can_sends.append(mazdacan.create_crz_ctrl(self.packer, bus, CC.longActive, acc_available, gap,
                                                  has_lead, phase, acc_active_2))
      self.long_counter += 1

    return can_sends
