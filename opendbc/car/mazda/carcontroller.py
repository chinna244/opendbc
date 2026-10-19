from collections import deque

import numpy as np

from opendbc.can import CANPacker
from opendbc.car import Bus, DT_CTRL, structs
from opendbc.car.lateral import apply_driver_steer_torque_limits
from opendbc.car.interfaces import CarControllerBase
from opendbc.car.mazda import mazdacan
from opendbc.car.mazda.longitudinal import AccelShaper, LeadIcon, StandstillHold
from opendbc.car.mazda.radar_session import RadarSessionManager, RadarSessionState
from opendbc.car.mazda.values import CarControllerParams, Buttons, MazdaFlags

from opendbc.sunnypilot.car.mazda.icbm import IntelligentCruiseButtonManagementInterface
from opendbc.sunnypilot.car.mazda.mads import MadsCarController
from opendbc.sunnypilot.car.stock_ecu import StockEcuState

VisualAlert = structs.CarControl.HUDControl.VisualAlert
LongCtrlState = structs.CarControl.Actuators.LongControlState
ButtonType = structs.CarState.ButtonEvent.Type

# Send synthetic radar frames to both consumers; panda does not forward locally generated frames.
LONG_BUSES = (0, 2)
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


class CarController(CarControllerBase, IntelligentCruiseButtonManagementInterface, MadsCarController):
  def __init__(self, dbc_names, CP, CP_SP):
    CarControllerBase.__init__(self, dbc_names, CP, CP_SP)
    IntelligentCruiseButtonManagementInterface.__init__(self, CP, CP_SP)
    MadsCarController.__init__(self)
    self.params = CarControllerParams(CP)
    # The non-delivery latch belongs to the steer-to-zero firmware alone.
    self.steer_to_zero = bool(CP.flags & MazdaFlags.STEER_TO_ZERO_EPS)
    self.g46l = bool(CP.flags & MazdaFlags.G46L_RADAR)
    self.apply_torque_last = 0
    self.driver_torque_samples: deque[float] = deque(maxlen=self.params.STEER_DRIVER_SAMPLES)
    self.packer = CANPacker(dbc_names[Bus.pt])
    self.cancel_counter = 0
    self.stop_and_go = StandstillHold()
    self.long_counter = 0
    self.radar_counter = 0
    self.radar_session = RadarSessionManager()
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
    self.accel_shaper = AccelShaper()
    self.accel_last = 0.
    self.lead_icon = LeadIcon()
    self.dash_steer_warning = DashSteerWarning()
    self.dash_warning_on_bus = False

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
      eps_ceiling = round(float(np.interp(CS.out.vEgoRaw, self.params.EPS_CEILING_LOOKUP[0],
                                          self.params.EPS_CEILING_LOOKUP[1])))
      new_torque = int(np.clip(new_torque, -eps_ceiling, eps_ceiling))

      # Use the worst sample plus margin to stay inside panda's fresher driver-torque envelope.
      if new_torque >= 0:
        driver_torque = min(self.driver_torque_samples) - self.params.STEER_DRIVER_MARGIN
      else:
        driver_torque = max(self.driver_torque_samples) + self.params.STEER_DRIVER_MARGIN

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

    can_sends.extend(self.update_mrcc_cleanup(CC, CC_SP, CS, self.radar_session.handback_active))

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
    rs.handback_by_driver = self.handback_by_driver
    # A completed hand-back is still in force only for the lifecycle's standing request; the
    # driver's leaves a stock radar, and the main-off HUD, as on a car never taken over.
    rs.handback_active = session_state == RadarSessionState.HANDBACK or (rs.handback_completed and CC_SP.stockEcuHandBack)
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

  def update_hud(self, CC, CC_SP, CS):
    """The alert frame at 2 Hz, mirroring the camera's own HUD state.

    The hands-on-wheel warning is the steerRequired alert above the LKAS speed floor, as
    DashSteerWarning lets it through. The MADS white wheel rides the same frame.
    """
    visual_alert = CC.hudControl.visualAlert
    steer_required = visual_alert == VisualAlert.steerRequired
    # TODO: find a way to silence audible warnings so we can add more hud alerts
    steer_required = steer_required and CS.lkas_allowed_speed
    steer_required = self.dash_steer_warning.update(self.frame, steer_required, CC.latActive, CC.enabled,
                                                    CS.out.brakePressed)
    white = self.update_white_hud(CC, CC_SP, CS, self.radar_session.handback_active)

    # Preserve the normal 2 Hz cadence; the exception is the immediate withdraw of anything
    # the last frame asserted.
    withdraw_now = (self.mads_white_hud_on_bus and not white) or (self.dash_warning_on_bus and not steer_required)
    if self.frame % HUD_ALERT_FRAMES == 0 or withdraw_now:
      alert = mazdacan.create_alert_command(self.packer, CS.cam_laneinfo, visual_alert == VisualAlert.ldw, steer_required)
      self.dash_warning_on_bus = steer_required
      return [self.white_hud_frame(CS, alert, white)]
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
                                              frame=self.frame,
                                              stock_engaged=stock_engaged, owned=CS.radar_owned,
                                              cruise_requested=self.cruise_requested)
    if self.radar_session.silencing_failed:
      # the radar refused or never answered: no takeover is retried this drive, so nothing pends
      self.cruise_requested = False
      self.cruise_request_frames = 0
    # Continue synthetic radar frames through hand-back to avoid a camera-visible gap.
    radar_master = self.radar_session.replacement_active
    self._track_radar_session(session_state, CS, CC_SP)

    if self.radar_session.diagnostic_message is not None:
      can_sends.append(self.radar_session.diagnostic_message)

    stopping = CC.actuators.longControlState == LongCtrlState.stopping
    # Engaged bits follow CC.enabled. Gas is an override, not a disengagement.
    control_ready = self.radar_session.control_active and bus_healthy and CS.out.canValid
    long_engaged = CC.enabled and control_ready
    long_active = CC.longActive and control_ready
    sm = self.stop_and_go
    sm.update(long_engaged, stopping, CS.out.standstill, CC.actuators.accel, CS.body_hold,
              gas_pressed=CS.out.gasPressed)

    accel = self.accel_shaper.update(long_active, CC.actuators.accel, self.accel_last, sm, CS.out.standstill,
                                     CS.body_hold, CS.out.vEgoRaw, CS.gear_box)
    self.accel_last = accel
    lead_shown = self.lead_icon.update(CC.hudControl.leadVisible)

    if radar_master and self.frame % CarControllerParams.RADAR_STEP == 0:
      for bus in LONG_BUSES:
        can_sends.extend(mazdacan.create_radar_frames(bus, self.radar_counter, g46l=self.g46l))
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
      # Show the vision lead on the dash with stock's distance bucket: 3 behind a stopped lead,
      # 2 while following. The tracks stay empty.
      lead_distance = (3 if CS.out.standstill else 2) if lead_shown else 0
      for bus in LONG_BUSES:
        can_sends.append(mazdacan.create_acc_command(self.packer, bus, self.long_counter, accel,
                                                     long_active=long_engaged, acc_available=acc_available,
                                                     brake_pressed=CS.out.brakePressed,
                                                     stopping=sm.stop_bits, resume_unlatching=sm.resume_unlatching))
        # The cruise frame is ours under alpha long, so it also carries the white wheel while openpilot steers
        # (MADS) with cruise engaged, when the camera frame's white state stays off (it fails closed on any cruise).
        # The camera frame itself is untouched, and the car's own cruise never sees this frame.
        lkas_white = bus == 0 and (self.mads_white_hud_on_bus or (long_engaged and CC_SP.mads.active))
        can_sends.append(mazdacan.create_crz_ctrl(self.packer, bus, long_engaged, acc_available, gap, acc_active_2,
                                                  lead_distance=lead_distance, hbc_request=CS.hbc_request,
                                                  lkas_white=lkas_white))
      self.long_counter += 1

    return can_sends
