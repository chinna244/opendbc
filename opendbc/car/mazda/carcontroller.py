from collections import deque

import numpy as np

from opendbc.can import CANPacker
from opendbc.car import Bus, DT_CTRL, rate_limit, structs
from opendbc.car.lateral import apply_driver_steer_torque_limits
from opendbc.car.interfaces import CarControllerBase
from opendbc.car.mazda import mazdacan
from opendbc.car.mazda.longitudinal import BREAKAWAY_FRAMES, RESUME_UNLATCH_LATCHED_FRAMES, AdvertisedLead, StandstillHold
from opendbc.car.mazda.radar_session import RadarSessionManager, RadarSessionState
from opendbc.car.mazda.values import CarControllerParams, Buttons, MazdaFlags
from opendbc.sunnypilot.car.mazda.values import MazdaFlagsSP

from opendbc.sunnypilot.car.mazda.icbm import BUTTONS, IntelligentCruiseButtonManagementInterface
from opendbc.sunnypilot.car.stock_ecu import StockEcuState

VisualAlert = structs.CarControl.HUDControl.VisualAlert
LongCtrlState = structs.CarControl.Actuators.LongControlState
ButtonType = structs.CarState.ButtonEvent.Type

# Send synthetic radar frames to both consumers; panda does not forward locally generated frames.
LONG_BUSES = (0, 2)
TJA_MRCC_MAX_TX_FRAMES = 3
TJA_MRCC_RAW_OFF_CONFIRM_FRAMES = 5
# The body ECU drops discrete presses faster than one per 200 ms (measured, docs/zoompilot/icbm.md);
# the arm clears in PEDALS ~80 ms after a press plus the 50 ms confirm.
TJA_MRCC_TX_PERIOD = 0.2  # s between undo presses
# The press-induced arm appears ~80 ms after the press edge (docs/zoompilot/mazda-lateral.md).
TJA_MRCC_ARM_WAIT_FRAMES = int(1.0 / DT_CTRL)
# The white wheel waits this long on a fully-off, quiet cruise before it displays.
MADS_WHITE_HUD_OFF_CONFIRM_FRAMES = int(0.5 / DT_CTRL)
# The alert frame's cadence.
HUD_ALERT_FRAMES = 50
CANCEL_SETTLE_FRAMES = int(CarControllerParams.CANCEL_SETTLE_T / DT_CTRL)
# The dash hands-on-wheel frame waits this long after lateral or cruise comes on: the driver is
# on the button, and the EPS-standby banner (1.8 s) can outlive the block that raised it.
DASH_STEER_WARNING_QUIET_FRAMES = int(2.0 / DT_CTRL)
# The driver asking for alpha long: until RES is pressed with the main switch on, before the car's
# own cruise has run since main came on, the stock radar, and its AEB, stays in its default
# session. SET+/SET- start the car's cruise; once it has run, RES is its resume (it remembers a speed).
ALPHA_LONG_REQUEST_BUTTONS = (ButtonType.resumeCruise,)
# The press may arm stock MRCC ~80 ms later: let that engagement show before the takeover gate opens.
CRUISE_REQUEST_SETTLE_FRAMES = int(1.0 / DT_CTRL)
# The main switch turned off after a takeover hands the radar, and its AEB, back. The takeover
# holds at least this long first, so a quick off/on never alternates radar sessions.
RADAR_HANDBACK_HOLD_FRAMES = int(5.0 / DT_CTRL)
# Back from a hand-back the radar withholds SET for ~12 s (route 0000012b, twice 11.9 s). Bounded,
# since the same bit also drops for other reasons (a cancel, the brake).
RADAR_WARMUP_MAX_FRAMES = int(20.0 / DT_CTRL)
# Back from a hand-back, the radar's cruise needs ~12 s (11.9 s twice, route 0000012b). Whether its
# AEB is back sooner is unknown, and no other module reports it (routes 0000012b/c: no bit outside
# the radar's own frames follows its mode), so AEB counts as off for that long. Off when it may be
# on is the safe error; the reverse is not.
RADAR_RESTART_FRAMES = int(12.0 / DT_CTRL)
# A takeover starts only at a real stop: one shorter than this (creeping traffic) would be cut short
# by the next roll, turning AEB off and back on for nothing. Standing still since start counts.
TAKEOVER_STOP_FRAMES = int(2.0 / DT_CTRL)


def stock_ecu_driver_view(status: StockEcuState, session: RadarSessionState, *, stock_cruise_engaged: bool,
                          main_on: bool, radar_warming: bool, stock_radar_live: bool) -> StockEcuState:
  """What the driver is told about the radar, from the session manager's own status. First match wins:

    radar the car's, its own cruise engaged, and no
      hand-back standing (RESTORED is the lifecycle's
      answer; main is off for the driver's)          -> STOCK_CRUISE_ON  chimes on its edges, SET silent
    manager would say park-to-take-over, and
      main on, radar still refusing SET (warm-up)    -> RESTORING        "Wait ~12 s", refuse sound
      stock radar heard                              -> NOT_NEEDED       SET silent: it works on the car
    anything else                                    -> the manager's status unchanged
  """
  if session == RadarSessionState.STOCK and stock_cruise_engaged and status != StockEcuState.RESTORED:
    return StockEcuState.STOCK_CRUISE_ON
  if status == StockEcuState.PARK_TO_TAKE_OVER:
    if main_on and radar_warming:
      return StockEcuState.RESTORING
    if stock_radar_live:
      return StockEcuState.NOT_NEEDED
  return status


class DashSteerWarning:
  """When the steerRequired alert also reaches the car's own dash.

  The device alert is the primary channel and is never touched here. The dash frame (the
  hands-on-wheel code in CAM_LANEINFO, see mazdacan.create_alert_command) mirrors the alert
  only while openpilot is steering, which keeps it off the dash at a standstill and through an
  EPS block; never inside the quiet window after lateral or cruise comes on; and, once the
  driver cancels or touches the brake while the alert is up, not again until the alert clears.
  The cluster chimes on the frame, so each rule is one chime less at a moment the driver is
  already acting.
  """

  def __init__(self):
    self.lat_active_prev = False
    self.enabled_prev = False
    self.brake_prev = False
    self.quiet_until = 0
    self.held = False

  def update(self, frame: int, steer_required: bool, lat_active: bool, enabled: bool, brake_pressed: bool) -> bool:
    if (lat_active and not self.lat_active_prev) or (enabled and not self.enabled_prev):
      self.quiet_until = frame + DASH_STEER_WARNING_QUIET_FRAMES
    if steer_required and ((self.enabled_prev and not enabled) or brake_pressed != self.brake_prev):
      self.held = True
    if not steer_required:
      self.held = False
    self.lat_active_prev, self.enabled_prev, self.brake_prev = lat_active, enabled, brake_pressed
    return steer_required and lat_active and not self.held and frame >= self.quiet_until


class CarController(CarControllerBase, IntelligentCruiseButtonManagementInterface):
  def __init__(self, dbc_names, CP, CP_SP):
    CarControllerBase.__init__(self, dbc_names, CP, CP_SP)
    IntelligentCruiseButtonManagementInterface.__init__(self, CP, CP_SP)
    if not CP.flags & MazdaFlags.GEN1:
      # mazdacan message builders require GEN1 frame layouts.
      raise NotImplementedError(f"unsupported platform: {CP.carFingerprint}")
    self.params = CarControllerParams(CP)
    # values.py selects the measured EPS envelope from the hardware mask; the speed-dependent
    # scale and the non-delivery latch belong to the steer-to-zero firmware alone.
    self.eps_2022 = bool(CP.flags & MazdaFlags.EPS_HW)
    self.steer_to_zero = bool(CP.flags & MazdaFlags.STEER_TO_ZERO_EPS)
    self.g46l = bool(CP.flags & MazdaFlags.G46L_RADAR)
    self.apply_torque_last = 0
    self.driver_torque_samples: deque[float] = deque(maxlen=self.params.STEER_DRIVER_SAMPLES if self.eps_2022 else 1)
    self.packer = CANPacker(dbc_names[Bus.pt])
    self.cancel_counter = 0
    self.stop_and_go = StandstillHold()
    self.lead_adv = AdvertisedLead()
    self.long_counter = 0
    self.radar_counter = 0
    self.radar_session = RadarSessionManager(moving_takeover=bool(CP.flags & MazdaFlags.MOVING_TAKEOVER))
    self.cruise_requested = False  # latched for the drive on the first SET/RES press
    # nothing asked for yet: no "remain parked" banner before the first controls frame either
    self.radar_session.status = StockEcuState.PARK_TO_TAKE_OVER
    self.cruise_off_handback = False  # the driver's hand-back: main off after a takeover
    self.radar_silenced_frames = 0
    self.radar_session_prev = RadarSessionState.STOCK
    self.handback_by_driver = False  # the hand-back in flight is the driver's main-off one
    self.main_on = True  # until the first frame says otherwise
    self.main_seen_on = False  # main has been on this drive: only then can a request exist to withdraw
    self.car_cruise_used = False  # the car's own cruise has run since main came on: RES is its resume
    self.radar_off_prev = False  # the radar left its default session last frame (programming sent or held)
    self.radar_warmup_frames = 0  # the stock radar, back from a hand-back, does not allow SET yet
    self.stock_radar_live = True  # until the first frame says otherwise
    self.stock_cruise_engaged = False
    self.radar_restart_frames = 0  # back from a hand-back, its AEB not yet counted as on
    self.cruise_request_frames = 0
    self.standstill_frames = TAKEOVER_STOP_FRAMES  # standing still since start is a real stop
    self.accel_last = 0.
    self.release_ramp = None
    self.breakaway_frames = 0
    self.tja_button_prev = False
    self.dash_steer_warning = DashSteerWarning()
    self.dash_warning_on_bus = False
    self.mrcc_armed_prev: bool | None = None
    self.mrcc_undo_pending = False
    self.mrcc_undo_saw_armed = False
    self.mrcc_undo_frames = 0
    self.mrcc_raw_off_frames = 0
    self.mrcc_arm_wait_frames = 0
    # The white wheel rides the alert frame; on_bus tracks that the white bit is on the wire.
    self.mads_white_hud_off_frames = 0
    self.mads_white_hud_on_bus = False

  def update(self, CC, CC_SP, CS, now_nanos):
    can_sends = []

    apply_torque = 0

    self.driver_torque_samples.append(CS.out.steeringTorque)
    if CS.lkas_rejected:
      # The panda reports every 0x243 it refused back on the can stream (src 192). A rejection
      # zeroes its rate-limit reference, so a controller that keeps ramping is refused on every
      # later frame and the EPS loses its stream: LKAS_FAULT about 0.6 s in, the camera fault
      # 5.3 s after that, neither clearing before the next ignition cycle. Only a command
      # within one step of zero is accepted next, so the ramp restarts there. A nonzero stream
      # refused because the panda's lateral is not armed becomes a one-step sawtooth instead of
      # a blind ramp to the rail; the camera's own 0x243 is forwarded meanwhile. See
      # docs/zoompilot/mazda-lateral.md, "LKAS_FAULT".
      self.apply_torque_last = 0

    if CC.latActive:
      # calculate steer and also set limits due to driver torque
      new_torque = int(round(CC.actuators.torque * self.params.STEER_MAX))

      # Clamp to applied EPS authority so controlsd can detect saturation. Keep this separate
      # from STEER_MAX because the torque parameters are expressed on STEER_MAX.
      if self.eps_2022:
        eps_ceiling = round(float(np.interp(CS.out.vEgoRaw, self.params.EPS_CEILING_LOOKUP[0],
                                            self.params.EPS_CEILING_LOOKUP[1])))
        new_torque = int(np.clip(new_torque, -eps_ceiling, eps_ceiling))

      # Use the worst sample plus margin to stay inside panda's fresher driver-torque envelope.
      margin = self.params.STEER_DRIVER_MARGIN if self.eps_2022 else 0
      if new_torque >= 0:
        driver_torque = min(self.driver_torque_samples) - margin
      else:
        driver_torque = max(self.driver_torque_samples) + margin

      apply_torque = apply_driver_steer_torque_limits(new_torque, self.apply_torque_last,
                                                      driver_torque, self.params)

    # Stop requesting torque while carstate says the EPS will not take it: after the
    # non-delivery latch, or through its first engagement of the cycle. Recovery ramps from zero.
    if self.steer_to_zero and (CS.steer_undelivered or CS.steer_first_engage_hold):
      apply_torque = 0

    # Do not cancel a stock MRCC engagement while the stock radar still owns the bus.
    stock_mrcc_owns_cruise = self.CP.openpilotLongitudinalControl and not CS.radar_was_silenced
    if CC.cruiseControl.cancel and not stock_mrcc_owns_cruise:
      # A CANCEL landing with cruise already off is the stock main-off. The car answers its own
      # cancels (brake, the wheel button) 60 to 90 ms after openpilot disengages on them, and the
      # request falls with cruise; upstream waited 70 ms for the brake only. Wait it out for all.
      self.cancel_counter = self.cancel_counter + 1
      if self.frame % 10 == 0 and self.cancel_counter >= CANCEL_SETTLE_FRAMES:
        # Cancel Stock ACC if it's enabled while OP is disengaged
        # Send at a rate of 10hz until we sync with stock ACC state
        can_sends.append(mazdacan.create_button_cmd(self.packer, self.CP, CS.crz_btns_counter, Buttons.CANCEL))
    else:
      self.cancel_counter = 0
      if self.resume_requested(CC) and self.frame % 5 == 0:
        can_sends.append(mazdacan.create_button_cmd(self.packer, self.CP, CS.crz_btns_counter, Buttons.RESUME))

    if self.CP_SP.flags & MazdaFlagsSP.TJA_BUTTON:
      can_sends.extend(self.update_mrcc_cleanup(CC, CC_SP, CS))

    self.apply_torque_last = apply_torque

    if self.CP.openpilotLongitudinalControl:
      can_sends.extend(self.update_longitudinal(CC, CC_SP, CS))

    can_sends.extend(self.update_hud(CC, CC_SP, CS))

    # send steering command
    can_sends.append(mazdacan.create_steering_control(self.packer, self.CP,
                                                      self.frame, apply_torque, CS.cam_lkas))

    # Suppress ICBM while cancel/resume is active or the MRCC cleanup owns CRZ_BTNS:
    # the wheel's press pattern owns the counter stream until release.
    icbm_suppress = (CC.cruiseControl.cancel or CC.cruiseControl.resume or CS.cancel_button == 1 or
                     self.mrcc_undo_pending or CS.tja_button == 1)
    if not icbm_suppress:
      can_sends.extend(IntelligentCruiseButtonManagementInterface.update(self, CC_SP, CS, self.packer, self.frame, self.last_button_frame))

    new_actuators = CC.actuators.as_builder()
    new_actuators.torque = apply_torque / self.params.STEER_MAX
    new_actuators.torqueOutputCan = apply_torque
    # Report the command sent on the wire after clipping, holds, slew, and overrides.
    new_actuators.accel = self.accel_last

    self.frame += 1
    return new_actuators, can_sends

  @property
  def stock_ecu_state(self):
    """The stock ECU transition contract (opendbc/sunnypilot/car/stock_ecu.py): card reads this
    one name and nothing brand-specific. NOT_NEEDED under stock longitudinal, as on any brand
    that silences nothing."""
    if not self.CP.openpilotLongitudinalControl:
      return StockEcuState.NOT_NEEDED
    return stock_ecu_driver_view(self.radar_session.status, self.radar_session.state,
                                 stock_cruise_engaged=self.stock_cruise_engaged, main_on=self.main_on,
                                 radar_warming=self.radar_warmup_frames > 0, stock_radar_live=self.stock_radar_live)

  @property
  def alpha_long_requested(self) -> bool:
    """The driver asked for alpha long (RES), main is on, and the radar is not held yet: card
    publishes it for the "alpha long at the next stop" line. It stays up through a takeover that
    a roll cuts short, so one request is announced once."""
    return self.CP.openpilotLongitudinalControl and self.cruise_requested and self.main_on and \
           self.radar_session.state != RadarSessionState.SILENCED and not self.radar_session.silencing_failed

  @property
  def stock_ecu_off(self) -> bool:
    """The radar's own functions, AEB with them, off or not yet counted as back: programming
    requested, silenced, mid hand-back, or within RADAR_RESTART_FRAMES of one. card publishes it
    for the UI's AEB-off marker and the "AEB on" line."""
    if not self.CP.openpilotLongitudinalControl:
      return False
    return self.radar_off_prev or self.radar_restart_frames > 0

  def _track_radar_session(self, session_state: RadarSessionState, CS, CC_SP) -> None:
    """Per-frame bookkeeping after the session manager ran: what carstate is told, and the timers
    that follow a hand-back (SET warm-up, AEB restart) and a takeover (the hand-back hold)."""
    entered_handback = session_state == RadarSessionState.HANDBACK and self.radar_session_prev != RadarSessionState.HANDBACK
    restored = self.radar_session_prev == RadarSessionState.HANDBACK and session_state == RadarSessionState.STOCK
    rs = self.radar_session
    radar_off = session_state in (RadarSessionState.SILENCED, RadarSessionState.HANDBACK) or \
                (session_state == RadarSessionState.SILENCING and rs.programming_sent)
    if entered_handback:
      self.handback_by_driver = self.cruise_off_handback and not CC_SP.stockEcuHandBack
    # A driver's hand-back the radar never answered is no cruise fault: main is off, engagement
    # stays blocked without the owned radar, and the fault would take MADS lateral with it.
    CS.radar_restore_failed = self.radar_session.handback_failed and not self.handback_by_driver
    # A completed hand-back is still in force only for the lifecycle's standing request; the
    # driver's leaves a stock radar, and the main-off HUD, as on a car never taken over.
    CS.radar_handback_active = session_state == RadarSessionState.HANDBACK or \
                               (self.radar_session.handback_completed and CC_SP.stockEcuHandBack)
    if restored:
      CS.radar_was_silenced = False  # stock traffic restored: the radar is the car's again
      self.radar_warmup_frames = RADAR_WARMUP_MAX_FRAMES
    else:
      if CS.stock_set_allowed or CC_SP.stockEcuHandBack or session_state != RadarSessionState.STOCK:
        self.radar_warmup_frames = 0
      else:
        self.radar_warmup_frames = max(self.radar_warmup_frames - 1, 0)
    # Any return to the default session restarts AEB: the driver's or lifecycle's hand-back, or the
    # radar's own recovery, even one re-taken on the same frame. Counted only in STOCK and held
    # otherwise, so a re-take not yet programmed leaves it as it was: no "AEB on" while restarting.
    if self.radar_off_prev and not radar_off:
      self.radar_restart_frames = RADAR_RESTART_FRAMES
    elif session_state == RadarSessionState.STOCK:
      self.radar_restart_frames = max(self.radar_restart_frames - 1, 0)
    self.radar_off_prev = radar_off
    self.radar_session_prev = session_state
    self.radar_silenced_frames = self.radar_silenced_frames + 1 if session_state == RadarSessionState.SILENCED else 0

  def update_mrcc_cleanup(self, CC, CC_SP, CS):
    # Undo the MRCC arm a physical TJA press causes: the wheel's press reaches every bus-0
    # ECU directly, so the button the driver declared as the lateral switch also arms cruise.
    # Answer with the driver's own MRCC master press until raw PEDALS confirms the arm is gone.
    can_sends = []

    raw_armed = bool(CS.mrcc_armed_raw)
    # PEDALS drops both cruise bits through a brake transition; the filtered state bridges
    # those samples, and only sustained raw-off is authoritative.
    self.mrcc_raw_off_frames = 0 if raw_armed else self.mrcc_raw_off_frames + 1
    raw_off_confirmed = self.mrcc_raw_off_frames >= TJA_MRCC_RAW_OFF_CONFIRM_FRAMES
    # PEDALS holds both cruise bits low under braking: the clock restarts there and runs
    # again once the pedal is off.
    self.mrcc_arm_wait_frames = 0 if CS.out.brakePressed else self.mrcc_arm_wait_frames + 1
    if self.CP.openpilotLongitudinalControl:
      filtered_armed = CS.cruise_available
    else:
      filtered_armed = CS.out.cruiseState.available
    mrcc_armed = raw_armed or (filtered_armed and not raw_off_confirmed)

    if CS.tja_button and not self.tja_button_prev:
      # A press before the previous press's arm reconciled sees that arm as its own
      # artifact, not the driver's baseline.
      if self.mrcc_undo_pending:
        self.mrcc_undo_frames = 0
      else:
        # PEDALS can already show the press-induced arm in the same cycle as the edge; the
        # previous stable sample is the state that existed before the press.
        armed_before_press = self.mrcc_armed_prev if self.mrcc_armed_prev is not None else mrcc_armed
        self.mrcc_undo_pending = not armed_before_press
      self.mrcc_undo_saw_armed = False
      self.mrcc_arm_wait_frames = 0
    self.tja_button_prev = bool(CS.tja_button)
    self.mrcc_armed_prev = mrcc_armed

    # The arm is fully reconciled; the budget returns for the next press.
    if not mrcc_armed and not CS.cruise_enabled and not CS.out.cruiseState.enabled:
      self.mrcc_undo_frames = 0

    if not self.mrcc_undo_pending:
      return can_sends

    self.mrcc_undo_saw_armed |= raw_armed

    # The driver's own cruise presses own CRZ_BTNS from this cycle on, and restoring
    # stock ECU ownership changes who owns the arm.
    driver_activity = (CS.cancel_button or CS.resume_button or CS.accel_button or CS.decel_button or
                       CS.mrcc_button or CS.distance_button)
    if driver_activity or CS.radar_handback_active or CC_SP.stockEcuHandBack:
      self.mrcc_undo_pending = False
      return can_sends

    # The arm never appeared: a takeover already disarmed it, or this car does not arm
    # through the press. Past the bounded brake-free wait, stand down.
    if not self.mrcc_undo_saw_armed:
      if self.mrcc_arm_wait_frames >= TJA_MRCC_ARM_WAIT_FRAMES:
        self.mrcc_undo_pending = False
      return can_sends

    if raw_off_confirmed:
      self.mrcc_undo_pending = False
      return can_sends

    # openpilot's own cancel or resume interleaves button frames; wait it out rather
    # than race the counter stream.
    if CC.cruiseControl.cancel or CC.cruiseControl.resume:
      return can_sends

    # Never inside the driver's held press.
    if CS.tja_button:
      return can_sends

    # One press per body-paced slot, within the episode budget; never while PEDALS reads disarmed,
    # where the master press would arm instead of disarm. last_button_frame also paces ICBM.
    if raw_armed and self.mrcc_undo_frames < TJA_MRCC_MAX_TX_FRAMES and \
       (self.frame - self.last_button_frame) * DT_CTRL > TJA_MRCC_TX_PERIOD:
      can_sends.append(mazdacan.create_mrcc_off_cmd(self.packer, CS.crz_btns_counter))
      self.last_button_frame = self.frame
      self.mrcc_undo_frames += 1
      if self.mrcc_undo_frames >= TJA_MRCC_MAX_TX_FRAMES:
        self.mrcc_undo_pending = False

    return can_sends

  def update_hud(self, CC, CC_SP, CS):
    """The alert frame at 2 Hz, mirroring the camera's own HUD state.

    The hands-on-wheel warning is the steerRequired alert above the LKAS speed floor, as
    DashSteerWarning lets it through.

    On a TJA-declared car with MADS active and cruise fully off, the white-wheel TJA=2
    state is XORed into the camera's current payload, but only onto an exact allowlisted
    idle base: TJA=2 is not display-only, the body reads the same frame. Every cruise
    interaction fails closed, and a white wheel HUD state that became unsafe is withdrawn
    immediately, outside the cadence.
    """
    if self.CP.openpilotLongitudinalControl:
      filtered_available, filtered_enabled = CS.cruise_available, CS.cruise_enabled
    else:
      filtered_available, filtered_enabled = CS.out.cruiseState.available, CS.out.cruiseState.enabled

    session_ambiguous = CS.radar_handback_active or CC_SP.stockEcuHandBack
    mrcc_off = (not session_ambiguous and not CS.mrcc_armed_raw and
                not filtered_available and not filtered_enabled)

    # Every button a TJA wheel carries: TJA, MRCC, SET+/-, RES, DISTANCE, plus the
    # synthesized ICBM set presses and openpilot's own cancel/resume.
    button_activity = (CS.tja_button or CS.mrcc_button or CS.cancel_button or CS.resume_button or
                       CS.accel_button or CS.decel_button or CS.distance_button or
                       CC_SP.intelligentCruiseButtonManagement.sendButton in BUTTONS or
                       CC.cruiseControl.cancel or CC.cruiseControl.resume)

    visual_alert = CC.hudControl.visualAlert
    steer_required = visual_alert == VisualAlert.steerRequired
    # TODO: find a way to silence audible warnings so we can add more hud alerts
    steer_required = steer_required and CS.lkas_allowed_speed
    steer_required = self.dash_steer_warning.update(self.frame, steer_required, CC.latActive, CC.enabled,
                                                    CS.out.brakePressed)

    fsc_raw = CS.cam_laneinfo_raw
    hud_base = mazdacan.white_hud_allowlist_base(fsc_raw)
    white_allowed = (
      bool(self.CP_SP.flags & MazdaFlagsSP.TJA_BUTTON) and
      CC_SP.mads.active and
      CS.cam_laneinfo_live and
      hud_base is not None and
      visual_alert == VisualAlert.none and
      not button_activity and
      mrcc_off
    )
    if white_allowed:
      self.mads_white_hud_off_frames = min(self.mads_white_hud_off_frames + 1, MADS_WHITE_HUD_OFF_CONFIRM_FRAMES)
    else:
      self.mads_white_hud_off_frames = 0
    white = white_allowed and self.mads_white_hud_off_frames >= MADS_WHITE_HUD_OFF_CONFIRM_FRAMES

    # Preserve the normal 2 Hz cadence; the exception is the immediate withdraw of anything
    # the last frame asserted.
    withdraw_now = (self.mads_white_hud_on_bus and not white) or (self.dash_warning_on_bus and not steer_required)
    if self.frame % HUD_ALERT_FRAMES == 0 or withdraw_now:
      alert = mazdacan.create_alert_command(self.packer, CS.cam_laneinfo, visual_alert == VisualAlert.ldw, steer_required)
      payload = hud_base if white else alert[1]
      alert = (alert[0], mazdacan.apply_mads_white_hud(fsc_raw, payload, white), alert[2])
      self.mads_white_hud_on_bus = mazdacan.is_mads_white_hud(alert[1])
      self.dash_warning_on_bus = steer_required
      return [alert]
    return []

  def resume_requested(self, CC) -> bool:
    """The resume button belongs to the stock-longitudinal path alone. Under openpilot longitudinal
    the hold is released in-protocol (stop bits drop, RESUME_UNLATCHING pulses, the command ramps),
    which is what stock MRCC does, and ICBM owns CRZ_BTNS. Toyota, Honda and Hyundai gate their
    resume button the same way.
    """
    return not self.CP.openpilotLongitudinalControl and CC.cruiseControl.resume

  def update_longitudinal(self, CC, CC_SP, CS):
    can_sends = []

    # Start takeover only after the FSC boot check and any stock engagement have ended.
    stock_radar_alive = CS.stock_radar_alive
    stock_engaged = stock_radar_alive and CS.cruise_enabled
    bus_healthy = CS.radar_bus_healthy
    # Main on withdraws the hand-back; the next takeover waits for a new SET/RES press and a stop.
    self.main_on = CS.cruise_available
    self.stock_radar_live = stock_radar_alive
    self.stock_cruise_engaged = stock_engaged
    # The car's own cruise running is the driver choosing it: a pending request goes, and until
    # main goes off RES belongs to it (the car remembers its speed), not to alpha long.
    self.car_cruise_used = CS.cruise_available and (self.car_cruise_used or stock_engaged)
    if (self.car_cruise_used or stock_engaged) and self.radar_session.state != RadarSessionState.SILENCED:
      self.cruise_requested = False  # also mid-takeover: the car's cruise at a stop aborts it
      self.cruise_request_frames = 0
    if CS.cruise_available:
      self.cruise_off_handback = False
      self.main_seen_on = True
    elif self.main_seen_on and self.radar_session.state != RadarSessionState.SILENCED:
      # main off with the radar not yet held: the request goes with it. A takeover in progress is
      # abandoned (the manager's own abort, a hand-back if programming was sent) and none follows
      # at the next stop. A held radar is the 5 s hold's, below.
      self.cruise_requested = False
      self.cruise_request_frames = 0
    elif self.radar_silenced_frames > RADAR_HANDBACK_HOLD_FRAMES and not self.cruise_off_handback:
      self.cruise_off_handback = True
      self.cruise_requested = False
      self.cruise_request_frames = 0
    # A press while the radar is being handed back or restarting is the driver reaching for stock
    # cruise (route 0000012c, three times), not a request for the radar again.
    radar_returning = self.radar_session.state == RadarSessionState.HANDBACK or \
                      (self.radar_warmup_frames > 0 and not CS.stock_set_allowed)
    self.cruise_requested |= CS.cruise_available and not radar_returning and not self.car_cruise_used and \
                             not self.radar_session.silencing_failed and \
                             any(be.pressed and be.type in ALPHA_LONG_REQUEST_BUTTONS for be in CS.out.buttonEvents)
    self.cruise_request_frames = self.cruise_request_frames + 1 if self.cruise_requested else 0
    gate_passed = CS.fsc_settled and self.cruise_request_frames > CRUISE_REQUEST_SETTLE_FRAMES
    self.standstill_frames = self.standstill_frames + 1 if CS.out.standstill else 0
    # carstate's guard from the previous frame is the ownership the engagement path sees
    session_state = self.radar_session.update(gate_passed, stock_radar_alive, CC_SP.stockEcuHandBack or self.cruise_off_handback,
                                              standstill=self.standstill_frames >= TAKEOVER_STOP_FRAMES,
                                              session_refused=CS.radar_session_refused,
                                              stock_radar_gone=CS.stock_radar_gone,
                                              bus_healthy=bus_healthy and CS.out.canValid,
                                              session_response=CS.radar_session_response, frame=self.frame,
                                              stock_engaged=stock_engaged, owned=CS.radar_owned,
                                              cruise_requested=self.cruise_requested)
    if self.radar_session.silencing_failed:
      # the radar refused or never answered: no takeover is retried this drive, so nothing pends
      self.cruise_requested = False
      self.cruise_request_frames = 0
    # Continue synthetic radar frames through hand-back to avoid a camera-visible gap.
    radar_master = self.radar_session.replacement_active
    CS.radar_control_active = radar_master and session_state == RadarSessionState.SILENCED
    self._track_radar_session(session_state, CS, CC_SP)

    if self.radar_session.diagnostic_message is not None:
      can_sends.append(self.radar_session.diagnostic_message)

    stopping = CC.actuators.longControlState == LongCtrlState.stopping
    # Engaged bits follow CC.enabled. Gas is an override, not a disengagement.
    control_ready = CS.radar_control_active and bus_healthy and CS.out.canValid
    long_engaged = CC.enabled and control_ready
    long_active = CC.longActive and control_ready
    sm = self.stop_and_go
    sm.update(long_engaged, stopping, CS.out.standstill, CC.actuators.accel, CS.body_hold,
              gas_pressed=CS.out.gasPressed)
    # Lead advertisement represents perception and is independent of engagement.
    self.lead_adv.update(CC.hudControl.leadVisible, CC_SP.leadOne.dRel,
                         CC_SP.leadOne.vRel, sm.holding)

    if sm.just_released:
      # Never-latched stops relax in one frame; latched holds ramp from the relaxed command.
      self.release_ramp = CarControllerParams.ACCEL_HOLD_LATCHED if sm.latched_release else \
                          CarControllerParams.ACCEL_RELEASE_BAND
    elif sm.holding or not long_active:
      # Re-holds and driver overrides terminate the release ramp.
      self.release_ramp = None

    accel = 0.
    if long_active:
      accel = float(np.clip(CC.actuators.accel, CarControllerParams.ACCEL_MIN, CarControllerParams.ACCEL_MAX))
      # Continue a bounded release ramp while stopped because the plan may not break static hold.
      if self.release_ramp is None or not CS.out.standstill:
        self.breakaway_frames = 0
      else:
        self.breakaway_frames += 1
      breakaway = CS.out.standstill and self.breakaway_frames <= BREAKAWAY_FRAMES
      # Bound breakaway by stock authority and by the plan-relative margin.
      ramp_ceiling = max(accel, min(CarControllerParams.ACCEL_BREAKAWAY_MAX,
                                    accel + CarControllerParams.ACCEL_BREAKAWAY_OVERSHOOT))
      if self.release_ramp is not None and (self.release_ramp < accel or breakaway):
        # The release ramp owns the command until it reaches the plan. Body-latched holds remain
        # at the relaxed command until the body lets go.
        accel = self.release_ramp
        if not (sm.latched_release and CS.body_hold):
          # Stock's ramp through the unlatch pulse window and once rolling; quicker while the car
          # still stands after it, so the pull-away does not wait on the ramp.
          ramp_rate = CarControllerParams.ACCEL_RELEASE_RAMP
          if CS.out.standstill and self.breakaway_frames > RESUME_UNLATCH_LATCHED_FRAMES and not sm.resume_unlatching:
            ramp_rate = CarControllerParams.ACCEL_RELEASE_RAMP_STOPPED
          # Follow a falling plan ceiling at the winddown limit.
          self.release_ramp = max(min(self.release_ramp + ramp_rate * DT_CTRL, ramp_ceiling),
                                  self.release_ramp + CarControllerParams.ACCEL_WINDDOWN_LIMIT)
      else:
        self.release_ramp = None
        # Track overrides in accel_last so control resumes through the slew limiter.
        accel = rate_limit(accel, self.accel_last, CarControllerParams.ACCEL_WINDDOWN_LIMIT,
                           CarControllerParams.ACCEL_WINDUP_LIMIT)
        if accel > 0.:
          # Shape positive commands to stock MRCC's ceiling and build rate at this speed.
          v_ego = CS.out.vEgoRaw
          ceiling = float(np.interp(v_ego, CarControllerParams.ACCEL_CEILING_BP, CarControllerParams.ACCEL_CEILING_V))
          build = float(np.interp(v_ego, CarControllerParams.ACCEL_BUILD_BP, CarControllerParams.ACCEL_BUILD_V)) * DT_CTRL
          accel = min(accel, ceiling, max(self.accel_last, 0.) + build)
        if self.accel_last > 0. and CC.actuators.accel >= 0.:
          # Lift the throttle at stock's rate; a brake request bypasses this above.
          accel = max(accel, self.accel_last + CarControllerParams.ACCEL_LIFT_LIMIT * DT_CTRL)
      if sm.car_has_hold:
        # Stop requesting brake hold after the body ECU takes ownership.
        accel = CarControllerParams.ACCEL_HOLD_LATCHED
      elif sm.holding:
        # Freeze the braking command while STOPPING is asserted.
        accel = min(accel, 0.) if CC.actuators.accel <= 0. else min(self.accel_last, 0.)
      if sm.resume_unlatching:
        # Bound the latched release pulse to stock's command range.
        accel = min(max(accel, CarControllerParams.ACCEL_HOLD_LATCHED),
                    CarControllerParams.ACCEL_RESUME_PULSE_MAX)
    self.accel_last = accel

    if radar_master and self.frame % CarControllerParams.RADAR_STEP == 0:
      for bus in LONG_BUSES:
        can_sends.extend(mazdacan.create_radar_frames(bus, self.radar_counter, self.lead_adv.lead, g46l=self.g46l))
      self.radar_counter += 1

    if radar_master and self.frame % CarControllerParams.LONG_STEP == 0:
      # Preserve the driver's main-switch state through restoration without advertising
      # openpilot engagement. CarState's public availability is already revoked.
      acc_available = CS.cruise_available if session_state == RadarSessionState.HANDBACK and bus_healthy else \
                      CS.out.cruiseState.available and control_ready
      # Mirror the driver's distance setting on the dash; stock shows 2 bars by default.
      # leadDistanceBars counts up, 1 closest to 3 farthest, and DISTANCE_SETTING counts down
      # from 4 bars, so the cluster lights 5 - DISTANCE_SETTING bars (stock: the wheel's
      # DISTANCE_LESS raises the raw, 576 segments).
      bars = int(CC.hudControl.leadDistanceBars) or 2
      gap = (5 - bars) if (long_engaged or acc_available) else 0
      acc_active_2 = sm.acc_active_2 if long_engaged else False
      for bus in LONG_BUSES:
        can_sends.append(mazdacan.create_acc_command(self.packer, bus, self.long_counter, accel,
                                                     long_active=long_engaged, acc_available=acc_available,
                                                     brake_pressed=CS.out.brakePressed,
                                                     stopping=sm.stop_bits, resume_unlatching=sm.resume_unlatching))
        can_sends.append(mazdacan.create_crz_ctrl(self.packer, bus, long_engaged, acc_available, gap,
                                                  self.lead_adv.has_lead, self.lead_adv.ctrl_phase,
                                                  acc_active_2, hbc_request=CS.hbc_request,
                                                  lkas_white=bus == 0 and self.mads_white_hud_on_bus))
      self.long_counter += 1

    return can_sends
