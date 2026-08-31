import numpy as np

from opendbc.can import CANPacker
from opendbc.car import Bus, DT_CTRL, make_tester_present_msg, rate_limit, structs, uds
from opendbc.car.lateral import apply_driver_steer_torque_limits
from opendbc.car.interfaces import CarControllerBase
from opendbc.car.mazda import mazdacan
from opendbc.car.mazda.longitudinal import (BREAKAWAY_FRAMES, RADAR_ADDR, AdvertisedLead, RadarSessionManager,
                                            RadarSessionState, StandstillHold, create_radar_session_msg)
from opendbc.car.mazda.values import CarControllerParams, Buttons, MazdaFlags, has_tja_mads

from opendbc.sunnypilot.car.mazda.icbm import IntelligentCruiseButtonManagementInterface
from opendbc.sunnypilot.car.mazda.values import MazdaFlagsSP

VisualAlert = structs.CarControl.HUDControl.VisualAlert
LongCtrlState = structs.CarControl.Actuators.LongControlState
SendButtonState = structs.IntelligentCruiseButtonManagement.SendButtonState
ICBM_SET_BUTTONS = (
  SendButtonState.increase,
  SendButtonState.decrease,
  SendButtonState.increaseHold,
  SendButtonState.decreaseHold,
)

# Synthetic radar frames go to the car and to the camera; the panda only forwards
# received frames between those buses, not our own transmissions.
LONG_BUSES = (0, 2)
# PEDALS can briefly report both ACC bits low during a brake transition. Require
# raw-off to persist before it overrides the intentionally brake-held public cruise
# state. Route 56's real TJA cleanup stayed raw-off for seconds, so this remains well
# inside the interval before another deliberate button press.
TJA_MRCC_RAW_OFF_CONFIRM_FRAMES = 5
# Keep BIT1=0 newer than the wheel's ~10 Hz idle CRZ_BTNS. Isolated taps on
# consecutive OEM counters are overwritten (route 00000030 t+38.6). 300 ms is
# long enough to outlast a few idle frames and short enough not to own the bus.
TJA_MRCC_HOLD_TIMEOUT_FRAMES = int(0.3 / DT_CTRL)
# Route 33: physical MRCC holds and successful synth cleanups advance to a fresh
# CTR about every 50–60 ms of active BIT1=0 TX (not every 10 ms controller tick).
# Successful synth episodes needed only 1–2 unique packed CTRs with lead ≤ 1 over
# the concurrent OEM wheel — never a third unique CTR.
TJA_MRCC_CTR_STEP_FRAMES = int(0.06 / DT_CTRL)
TJA_MRCC_MAX_UNIQUE_CTRS = 2
# WHITE uses CAM_LANEINFO.TJA=2, which the Mazda body/MRCC also consumes as
# functional TJA state. Only expose it after cruise has been completely off and
# quiet for 0.5 s; any interaction withdraws it immediately.
MADS_WHITE_HUD_OFF_CONFIRM_FRAMES = int(0.5 / DT_CTRL)


class CarController(CarControllerBase, IntelligentCruiseButtonManagementInterface):
  def __init__(self, dbc_names, CP, CP_SP):
    CarControllerBase.__init__(self, dbc_names, CP, CP_SP)
    IntelligentCruiseButtonManagementInterface.__init__(self, CP, CP_SP)
    if not CP.flags & MazdaFlags.GEN1:
      # every message builder in mazdacan assumes the GEN1 frame layouts; a new platform
      # needs its own before it can be admitted
      raise NotImplementedError(f"unsupported platform: {CP.carFingerprint}")
    self.params = CarControllerParams(CP)
    self.apply_torque_last = 0
    self.steer_undelivered_frames = 0
    self.steer_undelivered = False
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
    self.tja_button_prev = False
    self.tja_mrcc_unarm_pending = False
    self.tja_mrcc_saw_armed = False
    self.tja_mrcc_hold_frames = 0
    self.tja_mrcc_tx_frames = 0
    # Seed for create_button_cmd (packs CTR = seed+1). Re-TX the same seed at 100 Hz;
    # advance seed only every TJA_MRCC_CTR_STEP_FRAMES of active MRCC-OFF TX, and at
    # most TJA_MRCC_MAX_UNIQUE_CTRS unique packed CTRs per episode (route 33).
    self.tja_mrcc_cmd_counter: int | None = None
    self.tja_mrcc_unique_ctrs = 0
    self.tja_mrcc_ctr_tx_frames = 0
    self.tja_mrcc_armed_prev: bool | None = None
    self.tja_mrcc_raw_off_frames = 0
    self.mads_white_hud_off_frames = 0
    self.mads_white_hud_on_bus = False
    self.mads_white_hud_norm_base: bytes | None = None

  def update(self, CC, CC_SP, CS, now_nanos):
    can_sends = []
    tja_mrcc_cleanup_tx = False

    apply_torque = 0

    # Speed-dependent STEER_MAX (CX-5 2022: 1200 below 32 mph, 800 above). This is the scale
    # from the controller's normalized output to CAN counts, so it stays put -- see values.py.
    if hasattr(self.params, 'STEER_MAX_LOOKUP'):
      steer_max = round(float(np.interp(CS.out.vEgoRaw, self.params.STEER_MAX_LOOKUP[0],
                                         self.params.STEER_MAX_LOOKUP[1])))
    else:
      steer_max = self.params.STEER_MAX

    # Stale or faulted FSC CAM_LKAS must drop commanded torque on the wire.
    fsc_ok = bool(getattr(CS, "cam_lkas_live", False)) and mazdacan.fsc_cam_lkas_allows_steer(CS.cam_lkas)
    if CC.latActive and fsc_ok:
      # calculate steer and also set limits due to driver torque
      new_torque = int(round(CC.actuators.torque * steer_max))

      # Clamp to what the EPS will actually apply at this speed. Counts above the ceiling are
      # not delivered (0 of 7.5M frames above 32.5 mph ever exceeded 620), so this costs no
      # torque at the wheel; what it buys is honesty. new_actuators.torque below reports the
      # clamped value, so controlsd's steer_limited_by_safety fires while the EPS is railed and
      # the lateral controller freezes its integrator instead of winding up against a limit it
      # cannot see. Deliberately separate from steer_max: scaling that down would shrink every
      # sub-saturation command and invalidate the speed-dependent latAccelFactor seeds.
      # Applied before apply_driver_steer_torque_limits, whose driver-torque term only ever
      # narrows the window further (max_steer_allowed = min(steer_max, driver_max_torque)).
      if hasattr(self.params, 'EPS_CEILING_LOOKUP'):
        eps_ceiling = round(float(np.interp(CS.out.vEgoRaw, self.params.EPS_CEILING_LOOKUP[0],
                                            self.params.EPS_CEILING_LOOKUP[1])))
        new_torque = int(np.clip(new_torque, -eps_ceiling, eps_ceiling))

      apply_torque = apply_driver_steer_torque_limits(new_torque, self.apply_torque_last,
                                                      CS.out.steeringTorque, self.params, steer_max)

    # Non-delivery latch (values.py STEER_UNDELIVERED_*): once we have watched the EPS apply
    # exactly nothing to a real request, stop sending it. The camera latches ERR_BIT_1 on a
    # budget of requests that go nowhere, and the rate limiter's 12/frame walk to saturation
    # spends that budget fast while the wheel does not move.
    #
    # The latch releases on LKAS_BLOCK rather than on delivery returning, because once we are
    # sending zero there is no request left to observe delivery of -- keying the exit off our
    # own output would ramp back into the block and limit-cycle. The block bit is the EPS's own
    # statement that it is accepting LKAS again, and the entry condition already established
    # that this particular block is total.
    if hasattr(self.params, 'STEER_UNDELIVERED_FRAMES'):
      if not CS.lkas_blocked:
        self.steer_undelivered_frames = 0
        self.steer_undelivered = False
      elif not self.steer_undelivered:
        # steeringPressed is excluded because a driver holding the wheel is a legitimate reason
        # for the EPS to withhold torque, and apply_driver_steer_torque_limits is already
        # unwinding the command in that case
        if (CC.latActive and not CS.out.steeringPressed and CS.lkas_effective == 0 and
            abs(apply_torque) > self.params.STEER_UNDELIVERED_MIN):
          self.steer_undelivered_frames += 1
          self.steer_undelivered = self.steer_undelivered_frames >= self.params.STEER_UNDELIVERED_FRAMES
        else:
          self.steer_undelivered_frames = 0

      if self.steer_undelivered:
        # apply_torque_last follows below, so delivery coming back ramps from zero at
        # STEER_DELTA_UP (~0.8 s to full) instead of stepping into a request the EPS has
        # not seen. That is the cost of the latch, and it is paid in the speed range where
        # the EPS was applying nothing anyway.
        apply_torque = 0

    # Under op-long, controlsd raises cancel whenever cruiseState.enabled has no matching
    # CC.enabled (pcmCruise). While the stock radar still owns the bus -- the pre-teardown
    # settle window and the silencing-failed stay-stock fallback -- that engagement is the
    # driver's own stock MRCC (openpilot cannot engage there: availability is held low), and
    # the 10 Hz CANCEL would turn its main off within ~100 ms. Leave it alone; the teardown
    # gate already waits out a stock engagement. Once the radar has been silenced a stock
    # engagement is impossible and cancel keeps handling state desync. (The deeper home is
    # carstate not reporting a stock engagement as cruiseState.enabled under op-long at all;
    # that needs an audit of every enabled consumer first, so the send is filtered here.)
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
        can_sends.append(mazdacan.create_button_cmd(self.packer, self.CP, CS.crz_btns_counter, Buttons.CANCEL, CS))
    else:
      self.brake_counter = 0
      if self.resume_requested(CC) and self.frame % 5 == 0:
        can_sends.append(mazdacan.create_button_cmd(self.packer, self.CP, CS.crz_btns_counter, Buttons.RESUME, CS))

    # On the CX-5 2022, the physical TJA button also arms Mazda MRCC on bus 0.
    # Panda can strip the camera-forwarded copy, but it cannot hide a frame from ECUs
    # already sharing bus 0. If MRCC was off before TJA, undo only that side effect
    # after TJA release with a 100 Hz MRCC-off hold. Route 33: re-TX the current
    # packed CTR every 10 ms for bus ownership, and advance to a fresh CTR only about
    # every 60 ms of active TX (max 2 unique CTRs), matching physical holds. Preserve
    # MRCC that was already armed before TJA. A TJA-caused arm during
    # selfdriveInitializing is latched in CarState and inherited on the first apply
    # (route 00000031 t+8.4).
    if has_tja_mads(self.CP):
      tja_button = bool(getattr(CS, "tja_button", 0))
      filtered_mrcc_armed = bool(CS.cruise_available) if hasattr(CS, "cruise_available") else \
        bool(getattr(getattr(CS.out, "cruiseState", None), "available", False))
      raw_mrcc_armed = bool(getattr(CS, "mrcc_armed_raw", filtered_mrcc_armed))
      self.tja_mrcc_raw_off_frames = 0 if raw_mrcc_armed else self.tja_mrcc_raw_off_frames + 1
      raw_off_confirmed = self.tja_mrcc_raw_off_frames >= TJA_MRCC_RAW_OFF_CONFIRM_FRAMES
      # The filtered state protects against momentary brake-only dropouts. Once raw-off
      # is sustained, it is authoritative for this cleanup even if cruise_available is
      # deliberately cached until brake release.
      mrcc_armed = raw_mrcc_armed or (filtered_mrcc_armed and not raw_off_confirmed)
      tja_pressed = tja_button and not self.tja_button_prev
      driver_cruise_cmd = (
        getattr(CS, "cancel_button", 0) == 1 or getattr(CS, "resume_button", 0) == 1 or
        bool(getattr(CS, "accel_button", 0)) or bool(getattr(CS, "decel_button", 0)) or
        getattr(CS, "mrcc_button", 0) == 1 or
        getattr(CS, "distance_button_active", 0) == 1 or getattr(CS, "distance_button", 0) == 1
      )

      def end_cleanup():
        self.tja_mrcc_unarm_pending = False
        self.tja_mrcc_saw_armed = False
        self.tja_mrcc_hold_frames = 0
        self.tja_mrcc_tx_frames = 0
        self.tja_mrcc_cmd_counter = None
        self.tja_mrcc_unique_ctrs = 0
        self.tja_mrcc_ctr_tx_frames = 0
        # Drop any CarState observation of this same TJA side-effect so a post-init
        # latch cannot restart a second ~300 ms episode after timeout (route 33).
        if hasattr(CS, "tja_mrcc_side_effect_pending"):
          CS.tja_mrcc_side_effect_pending = False

      def begin_cleanup(*, saw_armed: bool):
        self.tja_mrcc_unarm_pending = True
        self.tja_mrcc_saw_armed = saw_armed
        self.tja_mrcc_hold_frames = 0
        self.tja_mrcc_tx_frames = 0
        self.tja_mrcc_cmd_counter = None
        self.tja_mrcc_unique_ctrs = 0
        self.tja_mrcc_ctr_tx_frames = 0

      # First apply after init: consume a CarState-latched TJA-caused arm.
      if (not self.tja_mrcc_unarm_pending and
          getattr(CS, "tja_mrcc_side_effect_pending", False)):
        begin_cleanup(saw_armed=raw_mrcc_armed)
        CS.tja_mrcc_side_effect_pending = False

      if tja_pressed and not self.tja_mrcc_unarm_pending:
        # PEDALS may already show the TJA-induced arm in the same update as the button
        # edge. The previous stable sample is the state that existed before the press.
        mrcc_armed_before_press = self.tja_mrcc_armed_prev if self.tja_mrcc_armed_prev is not None else mrcc_armed
        if not mrcc_armed_before_press:
          begin_cleanup(saw_armed=False)
          # Same TJA edge also arms the CarState latch once PEDALS catch up. Clear it
          # now so timeout cannot re-inherit this episode.
          if hasattr(CS, "tja_mrcc_side_effect_pending"):
            CS.tja_mrcc_side_effect_pending = False

      if self.tja_mrcc_unarm_pending:
        self.tja_mrcc_saw_armed |= raw_mrcc_armed
        # While we own cleanup, keep eating a late CarState latch for this episode.
        if hasattr(CS, "tja_mrcc_side_effect_pending"):
          CS.tja_mrcc_side_effect_pending = False
        if driver_cruise_cmd or CC.cruiseControl.cancel or CC.cruiseControl.resume:
          end_cleanup()
        elif self.tja_mrcc_saw_armed and raw_off_confirmed:
          end_cleanup()
        elif tja_button:
          # Pause TX while TJA is held. Keep the leftover arm and the episode timer:
          # a later TJA must not restart the 300 ms budget or treat this as pre-armed.
          pass
        else:
          # Count every post-release cycle, including raw-OFF dropouts, so OFF/ON
          # flickers cannot stretch cleanup past the original 300 ms episode.
          self.tja_mrcc_hold_frames += 1
          if self.tja_mrcc_hold_frames > TJA_MRCC_HOLD_TIMEOUT_FRAMES:
            end_cleanup()
          elif raw_mrcc_armed:
            if self.tja_mrcc_cmd_counter is None:
              # First packed CTR = OEM+1 via create_button_cmd, like route 33 successes.
              self.tja_mrcc_cmd_counter = int(CS.crz_btns_counter)
              self.tja_mrcc_unique_ctrs = 1
              self.tja_mrcc_ctr_tx_frames = 0
            elif (self.tja_mrcc_ctr_tx_frames >= TJA_MRCC_CTR_STEP_FRAMES and
                  self.tja_mrcc_unique_ctrs < TJA_MRCC_MAX_UNIQUE_CTRS):
              # ~60 ms of active TX at this packed CTR → one fresh consecutive CTR.
              self.tja_mrcc_cmd_counter = (self.tja_mrcc_cmd_counter + 1) % 16
              self.tja_mrcc_unique_ctrs += 1
              self.tja_mrcc_ctr_tx_frames = 0
            can_sends.append(mazdacan.create_button_cmd(self.packer, self.CP,
                                                        self.tja_mrcc_cmd_counter,
                                                        Buttons.MRCC_OFF, CS))
            self.tja_mrcc_ctr_tx_frames += 1
            tja_mrcc_cleanup_tx = True
            self.tja_mrcc_tx_frames += 1
          # raw OFF: stop TX immediately; keep ownership / CTR sequence until confirm
          # or a re-arm inside the timeout resumes the hold.

      self.tja_button_prev = tja_button
      self.tja_mrcc_armed_prev = mrcc_armed

    self.apply_torque_last = apply_torque

    if self.CP.openpilotLongitudinalControl:
      can_sends.extend(self.update_longitudinal(CC, CC_SP, CS))

    # CAM_LANEINFO.TJA=2 draws the WHITE wheel, but it is not display-only: the
    # Mazda body/MRCC consumes it too. Fail closed around every cruise/TJA
    # interaction. Only expose WHITE after MRCC has been completely off and quiet
    # for 0.5 s; ARMED and ACTIVE are hard denies.
    cruise_state = getattr(CS.out, "cruiseState", None)
    if self.CP.openpilotLongitudinalControl:
      filtered_mrcc_available = bool(getattr(CS, "cruise_available", False))
      filtered_mrcc_enabled = bool(getattr(CS, "cruise_enabled", False))
    else:
      filtered_mrcc_available = (
        cruise_state is not None and bool(getattr(cruise_state, "available", False))
      )
      filtered_mrcc_enabled = (
        cruise_state is not None and bool(getattr(cruise_state, "enabled", False))
      )

    mrcc_off = (
      not bool(getattr(CS, "mrcc_armed_raw", True)) and
      not bool(getattr(CS, "cruise_available", True)) and
      not bool(getattr(CS, "cruise_enabled", False)) and
      cruise_state is not None and
      not filtered_mrcc_available and
      not filtered_mrcc_enabled
    )

    icbm = getattr(CC_SP, "intelligentCruiseButtonManagement", None)
    icbm_set_activity = (
      icbm is not None and icbm.sendButton in ICBM_SET_BUTTONS
    )
    hud_button_activity = (
      bool(getattr(CS, "tja_button", 0)) or
      bool(getattr(CS, "mrcc_button", 0)) or
      bool(getattr(CS, "main_button", 0)) or
      bool(getattr(CS, "mode_x", 0)) or
      bool(getattr(CS, "mode_y", 0)) or
      bool(getattr(CS, "cancel_button", 0)) or
      bool(getattr(CS, "resume_button", 0)) or
      bool(getattr(CS, "accel_button", 0)) or
      bool(getattr(CS, "decel_button", 0)) or
      bool(getattr(CS, "distance_button", 0)) or
      icbm_set_activity or
      CC.cruiseControl.cancel or CC.cruiseControl.resume
    )
    ldw = CC.hudControl.visualAlert == VisualAlert.ldw
    steer_required = CC.hudControl.visualAlert == VisualAlert.steerRequired
    # TODO: find a way to silence audible warnings so we can add more hud alerts
    steer_required = steer_required and CS.lkas_allowed_speed
    alert = mazdacan.create_alert_command(self.packer, getattr(CS, "cam_laneinfo", {}) or {}, ldw, steer_required)
    packed_laneinfo = alert[1]
    fsc_raw = getattr(CS, "cam_laneinfo_raw", None)
    hud_base = mazdacan.white_hud_allowlist_base(fsc_raw)
    normalized_base = hud_base is not None

    white_hud_trusted = (
      has_tja_mads(self.CP) and
      bool(getattr(getattr(CC_SP, "mads", None), "active", False)) and
      getattr(CS, "cam_laneinfo_live", False) and
      normalized_base and
      CC.hudControl.visualAlert == VisualAlert.none and
      not self.tja_mrcc_unarm_pending and
      not tja_mrcc_cleanup_tx and
      not hud_button_activity
    )

    white_hud_off_base_allowed = (
      bool(self.CP_SP.flags & MazdaFlagsSP.EXPERIMENTAL_MADS_WHITE_HUD) and
      white_hud_trusted and
      mrcc_off
    )
    if white_hud_off_base_allowed:
      if self.mads_white_hud_norm_base is not None and self.mads_white_hud_norm_base != hud_base:
        self.mads_white_hud_off_frames = 0
      self.mads_white_hud_norm_base = hud_base
      self.mads_white_hud_off_frames = min(
        self.mads_white_hud_off_frames + 1,
        MADS_WHITE_HUD_OFF_CONFIRM_FRAMES,
      )
    else:
      self.mads_white_hud_off_frames = 0
      self.mads_white_hud_norm_base = None

    white_hud = (
      white_hud_off_base_allowed and
      self.mads_white_hud_off_frames >= MADS_WHITE_HUD_OFF_CONFIRM_FRAMES
    )
    withdraw_white_now = self.mads_white_hud_on_bus and not white_hud

    # Preserve the normal 2 Hz cadence. Exception: immediate OEM withdraw when WHITE
    # becomes unsafe (button / cleanup / ARMED / warning / stale / unknown payload).
    if self.frame % 50 == 0 or withdraw_white_now:
      payload = hud_base if white_hud and hud_base is not None else packed_laneinfo
      alert = (alert[0], mazdacan.apply_mads_white_hud(fsc_raw, payload, white_hud), alert[2])
      can_sends.append(alert)
      self.mads_white_hud_on_bus = mazdacan.is_mads_white_hud(alert[1])

    # send steering command
    can_sends.append(mazdacan.create_steering_control(self.packer, self.CP,
                                                      self.frame, apply_torque, CS.cam_lkas))

    # Intelligent Cruise Button Management
    # Suppress ICBM CRZ_BTNS spam while cancel/resume are in flight or while the driver is
    # holding the wheel cancel button. Without this guard ICBM's interleaved cancel=0 frames
    # race the driver's cancel=1 frames on the bus and the body ECU drops the cancel intent.
    # TJA_MADS only: also suppress while physical TJA is held so ICBM cannot fabricate a
    # TJA 1→0→1 edge on OP CRZ_BTNS. Non-TJA Mazdas must not let the TJA bit affect ICBM.
    icbm_suppress = (
      CC.cruiseControl.cancel or CC.cruiseControl.resume or CS.cancel_button == 1 or
      (has_tja_mads(self.CP) and
       (getattr(CS, "tja_button", 0) == 1 or self.tja_mrcc_unarm_pending or tja_mrcc_cleanup_tx))
    )
    if not icbm_suppress:
      can_sends.extend(IntelligentCruiseButtonManagementInterface.update(self, CC_SP, CS, self.packer, self.frame, self.last_button_frame))

    new_actuators = CC.actuators.as_builder()
    new_actuators.torque = apply_torque / steer_max
    new_actuators.torqueOutputCan = apply_torque
    # report what actually went on the wire, not the plan: the clip, the standstill hold values,
    # the slew limit, and the zero we send through a gas override all live in accel_last
    new_actuators.accel = self.accel_last

    self.frame += 1
    return new_actuators, can_sends

  def resume_requested(self, CC) -> bool:
    """The resume button is the stock ACC's only lever on a standstill hold, so it belongs to the
    stock-longitudinal path alone.

    Under openpilot longitudinal we are the ACC, and the hold is released in-protocol: CRZ_INFO's
    stop bits drop, RESUME_UNLATCHING pulses and the command ramps positive off the plan. That is
    what the car's own MRCC does -- across 23 stock body-latched-hold releases with cruise
    engaged, 0 put a RES press on the bus and all 23 pulsed RESUME_UNLATCHING
    (tools/mazda_long/scan_stock_release.py). Toyota, Honda and Hyundai all gate their resume
    button off openpilotLongitudinalControl the same way and release through their own ACC frame.

    Pressing it here would also put a second writer on CRZ_BTNS at the release: ICBM owns that
    address, and both of its interlocks (icbm_suppress above and the controller's own readiness
    gate) key off CC.cruiseControl.resume, which carstate makes False under openpilot
    longitudinal by construction.
    """
    return not self.CP.openpilotLongitudinalControl and CC.cruiseControl.resume

  def update_longitudinal(self, CC, CC_SP, CS):
    can_sends = []

    # Radar session sequencing (the why lives on RadarSessionManager): hold off the takeover
    # until the FSC's cold-boot radar-presence check has cleared, and never yank the radar
    # out from under an active stock MRCC engagement (driver SET before the gate passed on a
    # warm boot) -- wait for the driver to disengage first.
    stock_radar_alive = CS.stock_radar_alive
    setup_ok = CS.fsc_settled and not (stock_radar_alive and CS.out.cruiseState.enabled)
    session_state = self.radar_session.update(setup_ok, stock_radar_alive, CC_SP.stockEcuHandBack,
                                              standstill=CS.out.standstill,
                                              session_refused=CS.radar_session_refused)
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
    # The engaged bits follow CC.enabled the way Honda drives ACC_CONTROL's CONTROL_ON: a gas
    # press is an override, not a disengagement, so enabled holds while controlsd drops
    # longActive and the command goes to zero. Clearing the bits mid-decel takes the PCM out
    # of ACC mode as the driver adds throttle, so a light pedal input lands as a lurch and a
    # rev flare; stock MRCC holds them through 9 of 11 decel overrides (analyze_gas_override.py,
    # 576 stock segments). (MADS lateral-only sits outside CC.enabled, so this stays False
    # with cruise off.)
    long_engaged = CC.enabled
    sm = self.stop_and_go
    sm.update(long_engaged, stopping, CS.out.standstill, CC.actuators.accel, CS.brake_hold,
              gas_pressed=CS.out.gasPressed)
    # runs engaged or not: the advertisement is perception (see AdvertisedLead)
    self.lead_adv.update(CC.hudControl.leadVisible, CC_SP.leadOne.dRel,
                         CC_SP.leadOne.vRel, sm.holding)

    if sm.just_released:
      # the release command follows stock's shape, not a slew off the hold value: a
      # never-latched stop relax-jumps into the release band in one frame, a latched hold
      # ramps off the relaxed -0.001 (values.py census). Slewing up from -1.024 instead
      # keeps hold-grade braking on the wire underneath the release pulse, a tuple stock
      # never emits.
      self.release_ramp = CarControllerParams.ACCEL_HOLD_LATCHED if sm.latched_release else \
                          CarControllerParams.ACCEL_RELEASE_BAND
    elif sm.holding or not CC.longActive:
      # a re-hold or a driver override takes the command back; the ramp is release-only
      self.release_ramp = None

    accel = 0.
    if CC.longActive:
      accel = float(np.clip(CC.actuators.accel, CarControllerParams.ACCEL_MIN, CarControllerParams.ACCEL_MAX))
      # A release that has not actually moved the car keeps the ramp alive past the plan: the
      # plan's creep value is not always enough to break away (see ACCEL_BREAKAWAY_MAX). This
      # only extends how long the ramp owns the command -- the climb itself still obeys the
      # latched-hold freeze below, so a body-latched release is never leaned on any harder.
      if self.release_ramp is None or not CS.out.standstill:
        self.breakaway_frames = 0
      else:
        self.breakaway_frames += 1
      breakaway = CS.out.standstill and self.breakaway_frames <= BREAKAWAY_FRAMES
      ramp_ceiling = max(accel, CarControllerParams.ACCEL_BREAKAWAY_MAX)
      if self.release_ramp is not None and (self.release_ramp < accel or breakaway):
        # the release owns the command until its ramp catches the plan: stock climbs
        # ~+1.25 m/s3 straight through the blip or pulse and on into the drive-off.
        # A latched release does not start climbing until the body lets go: stock pins
        # the command at -1 raw until GEAR.BRAKE_HOLD drops in every latched release of
        # the corpus, and there is nothing to gain by pushing against brakes the body
        # still owns.
        accel = self.release_ramp
        if not (sm.latched_release and CS.brake_hold):
          self.release_ramp = min(self.release_ramp + CarControllerParams.ACCEL_RELEASE_RAMP * DT_CTRL,
                                  ramp_ceiling)
      else:
        self.release_ramp = None
        # Slew limit the plan-following command. accel_last is tracked through overrides too,
        # so taking control back when the driver lifts off ramps in instead of stepping.
        accel = rate_limit(accel, self.accel_last, CarControllerParams.ACCEL_WINDDOWN_LIMIT,
                           CarControllerParams.ACCEL_WINDUP_LIMIT)
      if sm.car_has_hold:
        # the body ECU is holding the brakes itself, so stop asking for them like stock does
        accel = CarControllerParams.ACCEL_HOLD_LATCHED
      elif sm.holding:
        # while the plan is braking the hold command is the plan's own, but the moment it
        # turns positive (release debounce) the hold freezes where it is:
        # stock never lets ACCEL_CMD climb while STOPPING is asserted, and pre-ramping
        # toward the plan puts the release's zero-cross inside the unlatch pulse, which
        # stock never does either.
        accel = min(accel, 0.) if CC.actuators.accel <= 0. else min(self.accel_last, 0.)
      if sm.resume_unlatching:
        # only a latched release ever arms the pulse, so this is stock's latched shape:
        # -1 raw to +0.25 m/s2. The ceiling is an invariant the ramp already keeps. The
        # floor does real work on a re-hold that lands while the pulse is still playing:
        # the pulse runs out (stock never restarts one), and this keeps the re-hold's
        # braking off the pulse frames, which is the shape stock's own releases have.
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
