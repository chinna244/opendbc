import numpy as np

from opendbc.can import CANPacker
from opendbc.car import Bus, make_tester_present_msg, structs, uds
from opendbc.car.lateral import apply_driver_steer_torque_limits
from opendbc.car.interfaces import CarControllerBase
from opendbc.car.mazda import mazdacan
from opendbc.car.mazda.longitudinal import (RADAR_ADDR, RadarSessionManager, RadarSessionState, StopAndGoStateMachine,
                                            StopGoState, create_radar_session_msg)
from opendbc.car.mazda.values import CarControllerParams, Buttons

from opendbc.sunnypilot.car.mazda.icbm import IntelligentCruiseButtonManagementInterface

VisualAlert = structs.CarControl.HUDControl.VisualAlert
LongCtrlState = structs.CarControl.Actuators.LongControlState

# Synthetic radar frames go to the car and to the camera; the panda only forwards
# received frames between those buses, not our own transmissions.
LONG_BUSES = (0, 2)


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
        # Send Resume button when planner wants car to move. With openpilot longitudinal the
        # RES press asks the body ECU to leave its standstill hold; only meaningful from a stop.
        if not self.CP.openpilotLongitudinalControl or CS.out.standstill:
          can_sends.append(mazdacan.create_button_cmd(self.packer, self.CP, CS.crz_btns_counter, Buttons.RESUME))
          virtual_resume_sent = self.CP.openpilotLongitudinalControl

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
    # radar-presence check has cleared (carstate's settle timer), keep the radar in its
    # programming session while we own the bus, and on an onroad toggle-off return it
    # to the default session before card requests the process restart. Never yank the
    # radar out from under an active stock MRCC engagement (driver SET before the gate
    # passed on a warm boot): wait for the driver to disengage first.
    stock_radar_alive = CS.stock_radar_alive
    teardown_ok = CS.fsc_settled and not (stock_radar_alive and CS.out.cruiseState.enabled)
    session_state = self.radar_session.update(teardown_ok, stock_radar_alive, CC_SP.stockEcuHandBack)
    # synthetic radar frames flow while we own the bus, and keep flowing through the
    # hand-back so the camera never sees a radar gap
    radar_master = session_state in (RadarSessionState.SILENCED, RadarSessionState.HANDBACK)

    if self.frame % CarControllerParams.RADAR_UDS_STEP == 0:
      if session_state == RadarSessionState.SILENCING:
        can_sends.append(create_radar_session_msg(uds.SESSION_TYPE.PROGRAMMING))
      elif session_state == RadarSessionState.HANDBACK:
        can_sends.append(create_radar_session_msg(uds.SESSION_TYPE.DEFAULT))
      elif session_state == RadarSessionState.SILENCED:
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
    if radar_master and self.frame % CarControllerParams.RADAR_STEP == 0:
      synthetic_lead = CC.longActive and (lead_visible or state != StopGoState.CRUISING)
      for bus in LONG_BUSES:
        can_sends.extend(mazdacan.create_radar_frames(bus, self.radar_counter, synthetic_lead))
      self.radar_counter += 1

    if radar_master and self.frame % CarControllerParams.LONG_STEP == 0:
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
