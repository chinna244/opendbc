from collections import deque

import numpy as np

from opendbc.can import CANPacker
from opendbc.car import Bus, DT_CTRL, make_tester_present_msg, rate_limit, structs, uds
from opendbc.car.lateral import apply_driver_steer_torque_limits
from opendbc.car.interfaces import CarControllerBase
from opendbc.car.mazda import mazdacan
from opendbc.car.mazda.longitudinal import (BREAKAWAY_FRAMES, RADAR_ADDR, AdvertisedLead, RadarSessionManager,
                                            RadarSessionState, StandstillHold, create_radar_session_msg)
from opendbc.car.mazda.values import CarControllerParams, Buttons, MazdaFlags

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
    if not CP.flags & MazdaFlags.GEN1:
      # every message builder in mazdacan assumes the GEN1 frame layouts
      raise NotImplementedError(f"unsupported platform: {CP.carFingerprint}")
    self.params = CarControllerParams(CP)
    # the whole 2022 EPS lateral block in values.py keys on this flag
    self.eps_2022 = bool(CP.flags & MazdaFlags.STEER_TO_ZERO_EPS)
    self.apply_torque_last = 0
    self.driver_torque_samples: deque[float] = deque(maxlen=self.params.STEER_DRIVER_SAMPLES if self.eps_2022 else 1)
    self.packer = CANPacker(dbc_names[Bus.pt])
    self.brake_counter = 0
    self.stop_and_go = StandstillHold()
    self.lead_adv = AdvertisedLead()
    self.long_counter = 0
    self.radar_counter = 0
    self.radar_session = RadarSessionManager()
    self.accel_last = 0.
    self.release_ramp = None
    self.breakaway_frames = 0

  def update(self, CC, CC_SP, CS, now_nanos):
    can_sends = []

    apply_torque = 0

    # speed-dependent STEER_MAX on the 2022 EPS: 1200 below 32 mph, 800 above
    if self.eps_2022:
      steer_max = round(float(np.interp(CS.out.vEgoRaw, self.params.STEER_MAX_LOOKUP[0],
                                         self.params.STEER_MAX_LOOKUP[1])))
    else:
      steer_max = self.params.STEER_MAX

    self.driver_torque_samples.append(CS.out.steeringTorque)

    if CC.latActive:
      # calculate steer and also set limits due to driver torque
      new_torque = int(round(CC.actuators.torque * steer_max))

      # Clamp to what the EPS will actually apply at this speed, so the reported torque shows
      # the saturation and controlsd's steer_limited_by_safety freezes the integrator. Kept
      # separate from steer_max, which the latAccelFactor seeds depend on.
      if self.eps_2022:
        eps_ceiling = round(float(np.interp(CS.out.vEgoRaw, self.params.EPS_CEILING_LOOKUP[0],
                                            self.params.EPS_CEILING_LOOKUP[1])))
        new_torque = int(np.clip(new_torque, -eps_ceiling, eps_ceiling))

      # Bound the driver-torque ceiling with the most adverse sample in the window plus a
      # margin, not the newest sample, so the command stays inside the envelope the panda
      # enforces from its own fresher samples. Only the commanded side can bind.
      margin = self.params.STEER_DRIVER_MARGIN if self.eps_2022 else 0
      if new_torque >= 0:
        driver_torque = min(self.driver_torque_samples) - margin
      else:
        driver_torque = max(self.driver_torque_samples) + margin

      apply_torque = apply_driver_steer_torque_limits(new_torque, self.apply_torque_last,
                                                      driver_torque, self.params, steer_max)

    # non-delivery latch: the EPS is applying nothing to a real request, so stop sending one
    # before the camera latches ERR_BIT_1. apply_torque_last follows, so delivery coming back
    # ramps from zero.
    if self.eps_2022 and CS.steer_undelivered:
      apply_torque = 0

    # While the stock radar still owns the bus under op-long, an engagement is the driver's own
    # stock MRCC and controlsd's cancel (pcmCruise state desync) would turn its main off. Leave
    # it alone; the teardown gate waits out a stock engagement anyway.
    stock_mrcc_owns_cruise = self.CP.openpilotLongitudinalControl and not CS.radar_was_silenced
    if CC.cruiseControl.cancel and not stock_mrcc_owns_cruise:
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
      if self.resume_requested(CC) and self.frame % 5 == 0:
        can_sends.append(mazdacan.create_button_cmd(self.packer, self.CP, CS.crz_btns_counter, Buttons.RESUME))

    self.apply_torque_last = apply_torque

    if self.CP.openpilotLongitudinalControl:
      can_sends.extend(self.update_longitudinal(CC, CC_SP, CS))

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

    # Intelligent Cruise Button Management: suppressed while cancel/resume are in flight or the
    # driver holds the wheel cancel, or its cancel=0 frames race the driver's cancel=1
    icbm_suppress = CC.cruiseControl.cancel or CC.cruiseControl.resume or CS.cancel_button == 1
    if not icbm_suppress:
      can_sends.extend(IntelligentCruiseButtonManagementInterface.update(self, CC_SP, CS, self.packer, self.frame, self.last_button_frame))

    new_actuators = CC.actuators.as_builder()
    new_actuators.torque = apply_torque / steer_max
    new_actuators.torqueOutputCan = apply_torque
    # report what went on the wire (clip, hold values, slew, the override zero), not the plan
    new_actuators.accel = self.accel_last

    self.frame += 1
    return new_actuators, can_sends

  def resume_requested(self, CC) -> bool:
    """The resume button belongs to the stock-longitudinal path alone. Under openpilot longitudinal
    the hold is released in-protocol (stop bits drop, RESUME_UNLATCHING pulses, the command ramps),
    which is what stock MRCC does, and ICBM owns CRZ_BTNS. Toyota, Honda and Hyundai gate their
    resume button the same way.
    """
    return not self.CP.openpilotLongitudinalControl and CC.cruiseControl.resume

  def update_longitudinal(self, CC, CC_SP, CS):
    can_sends = []

    # Radar session sequencing: hold off the takeover until the FSC's cold-boot radar-presence
    # check has cleared, and never pull the radar out from under an active stock engagement
    stock_radar_alive = CS.stock_radar_alive
    setup_ok = CS.fsc_settled and not (stock_radar_alive and CS.out.cruiseState.enabled)
    session_state = self.radar_session.update(setup_ok, stock_radar_alive, CC_SP.stockEcuHandBack,
                                              standstill=CS.out.standstill,
                                              session_refused=CS.radar_session_refused,
                                              stock_radar_gone=CS.stock_radar_gone)
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

    stopping = CC.actuators.longControlState == LongCtrlState.stopping
    # The engaged bits follow CC.enabled the way Honda drives CONTROL_ON: a gas press is an
    # override, not a disengagement, and clearing the bits mid-decel makes the PCM lurch as the
    # driver adds throttle. MADS lateral-only sits outside CC.enabled.
    long_engaged = CC.enabled
    sm = self.stop_and_go
    sm.update(long_engaged, stopping, CS.out.standstill, CC.actuators.accel, CS.brake_hold,
              gas_pressed=CS.out.gasPressed)
    # runs engaged or not: the advertisement is perception (see AdvertisedLead)
    self.lead_adv.update(CC.hudControl.leadVisible, CC_SP.leadOne.dRel,
                         CC_SP.leadOne.vRel, sm.holding)

    if sm.just_released:
      # stock's release shape: a never-latched stop relax-jumps into the release band in one
      # frame, a latched hold ramps off the relaxed -0.001
      self.release_ramp = CarControllerParams.ACCEL_HOLD_LATCHED if sm.latched_release else \
                          CarControllerParams.ACCEL_RELEASE_BAND
    elif sm.holding or not CC.longActive:
      # a re-hold or a driver override takes the command back; the ramp is release-only
      self.release_ramp = None

    accel = 0.
    if CC.longActive:
      accel = float(np.clip(CC.actuators.accel, CarControllerParams.ACCEL_MIN, CarControllerParams.ACCEL_MAX))
      # a release that has not moved the car keeps the ramp alive past the plan, whose creep
      # value is not always enough to break away; the latched-hold freeze below still applies
      if self.release_ramp is None or not CS.out.standstill:
        self.breakaway_frames = 0
      else:
        self.breakaway_frames += 1
      breakaway = CS.out.standstill and self.breakaway_frames <= BREAKAWAY_FRAMES
      # bounded both absolutely (stock's worst breakaway) and relative to the plan, so a small
      # plan behind a close lead gets a firm nudge, not a full-authority launch
      ramp_ceiling = max(accel, min(CarControllerParams.ACCEL_BREAKAWAY_MAX,
                                    accel + CarControllerParams.ACCEL_BREAKAWAY_OVERSHOOT))
      if self.release_ramp is not None and (self.release_ramp < accel or breakaway):
        # the release owns the command until its ramp catches the plan. A latched release does
        # not climb until the body lets go: stock pins raw -1 until GEAR.BRAKE_HOLD drops.
        accel = self.release_ramp
        if not (sm.latched_release and CS.brake_hold):
          # a plan that shrinks mid-climb lowers the ceiling; walk down at the winddown limit
          self.release_ramp = max(min(self.release_ramp + CarControllerParams.ACCEL_RELEASE_RAMP * DT_CTRL, ramp_ceiling),
                                  self.release_ramp + CarControllerParams.ACCEL_WINDDOWN_LIMIT)
      else:
        self.release_ramp = None
        # accel_last is tracked through overrides too, so taking control back ramps in
        accel = rate_limit(accel, self.accel_last, CarControllerParams.ACCEL_WINDDOWN_LIMIT,
                           CarControllerParams.ACCEL_WINDUP_LIMIT)
      if sm.car_has_hold:
        # the body ECU is holding the brakes itself, so stop asking for them like stock does
        accel = CarControllerParams.ACCEL_HOLD_LATCHED
      elif sm.holding:
        # the hold command is the plan's own while it brakes, and freezes the moment the plan
        # turns positive: stock never lets ACCEL_CMD climb while STOPPING is asserted
        accel = min(accel, 0.) if CC.actuators.accel <= 0. else min(self.accel_last, 0.)
      if sm.resume_unlatching:
        # stock's latched pulse shape, raw -1 to +0.25 m/s2; the floor keeps a re-hold that
        # lands mid-pulse from braking under the pulse frames
        accel = min(max(accel, CarControllerParams.ACCEL_HOLD_LATCHED),
                    CarControllerParams.ACCEL_RESUME_PULSE_MAX)
    self.accel_last = accel

    if radar_master and self.frame % CarControllerParams.RADAR_STEP == 0:
      for bus in LONG_BUSES:
        can_sends.extend(mazdacan.create_radar_frames(bus, self.radar_counter, self.lead_adv.lead))
      self.radar_counter += 1

    if radar_master and self.frame % CarControllerParams.LONG_STEP == 0:
      acc_available = CS.out.cruiseState.available
      # mirror the driver's distance setting on the dash; stock shows gap 2 by default
      gap = (int(CC.hudControl.leadDistanceBars) or 2) if (long_engaged or acc_available) else 0
      acc_active_2 = sm.acc_active_2 if long_engaged else False
      for bus in LONG_BUSES:
        can_sends.append(mazdacan.create_acc_command(self.packer, bus, self.long_counter, accel,
                                                     long_active=long_engaged, acc_available=acc_available,
                                                     brake_pressed=CS.out.brakePressed,
                                                     stopping=sm.stop_bits, resume_unlatching=sm.resume_unlatching))
        can_sends.append(mazdacan.create_crz_ctrl(self.packer, bus, long_engaged, acc_available, gap,
                                                  self.lead_adv.has_lead, self.lead_adv.ctrl_phase,
                                                  acc_active_2))
      self.long_counter += 1

    return can_sends
