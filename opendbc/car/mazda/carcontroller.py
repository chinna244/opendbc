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
from opendbc.sunnypilot.car.mazda.values import MazdaFlagsSP

from opendbc.sunnypilot.car.mazda.icbm import IntelligentCruiseButtonManagementInterface

VisualAlert = structs.CarControl.HUDControl.VisualAlert
LongCtrlState = structs.CarControl.Actuators.LongControlState
SendButtonState = structs.IntelligentCruiseButtonManagement.SendButtonState
ICBM_SET_BUTTONS = (
  SendButtonState.increase,
  SendButtonState.decrease,
  SendButtonState.increaseHold,
  SendButtonState.decreaseHold,
)

# Send synthetic radar frames to both consumers; panda does not forward locally generated frames.
LONG_BUSES = (0, 2)
TJA_MRCC_RELEASE_WAIT_FRAMES = 25
TJA_MRCC_FIRST_TX_DELAY_NANOS = 50_000_000
TJA_MRCC_MAX_TX_FRAMES = 3
# PEDALS can briefly report both ACC bits low during a brake transition. Require
# raw-off to persist before it overrides the intentionally brake-held public cruise
# state. Route 56's real TJA cleanup stayed raw-off for seconds, so this remains well
# inside the interval before another deliberate button press.
TJA_MRCC_RAW_OFF_CONFIRM_FRAMES = 5
MADS_WHITE_HUD_OFF_CONFIRM_FRAMES = int(0.5 / DT_CTRL)


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
    self.tja_mrcc_release_counter: int | None = None
    self.tja_mrcc_release_wait_frames = 0
    self.tja_mrcc_first_tx_not_before_nanos: int | None = None
    self.tja_mrcc_wait_for_fresh_counter_after_op = False
    self.tja_mrcc_press_frames = 0
    self.tja_mrcc_tx_frames = 0
    self.tja_mrcc_armed_prev: bool | None = None
    self.tja_mrcc_raw_off_frames = 0
    self.mads_white_hud_off_frames = 0
    self.mads_white_hud_on_bus = False
    # The camera's own TJA/CTS is pressed off on its bus while openpilot steers, per arming
    # episode: the camera re-arms on the driver's own TJA press (that press is also the MADS
    # switch on declared cars) and drops its arm by itself at times.
    self.tja_press_count = 0
    self.tja_press_frame: int | None = None
    self.tja_episode_alerted = False

  def update(self, CC, CC_SP, CS, now_nanos):
    can_sends = []
    tja_mrcc_cleanup_tx = False

    apply_torque = 0

    # The measured EPS uses a speed-dependent STEER_MAX.
    if self.eps_2022:
      steer_max = round(float(np.interp(CS.out.vEgoRaw, self.params.STEER_MAX_LOOKUP[0],
                                         self.params.STEER_MAX_LOOKUP[1])))
    else:
      steer_max = self.params.STEER_MAX

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
      new_torque = int(round(CC.actuators.torque * steer_max))

      # Clamp to applied EPS authority so controlsd can detect saturation. Keep this separate
      # from steer_max because torque parameter scaling depends on steer_max.
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
                                                      driver_torque, self.params, steer_max)

    # Stop requesting torque while carstate says the EPS will not take it: after the
    # non-delivery latch, or through its first engagement of the cycle. Recovery ramps from zero.
    if self.steer_to_zero and (CS.steer_undelivered or CS.steer_first_engage_hold):
      apply_torque = 0

    # Do not cancel a stock MRCC engagement while the stock radar still owns the bus.
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

    # On the CX-5 2022, the physical TJA button also arms Mazda MRCC on bus 0.
    # Panda can strip the camera-forwarded copy, but it cannot hide a frame from ECUs
    # already sharing bus 0. If MRCC was off before TJA, undo only that side effect
    # after TJA release with a physical-style MRCC-off hold. Route 61 showed three
    # isolated later retries can all be ignored; a physical press stays asserted on
    # consecutive counters. Never send more than three frames total for one ownership
    # episode. A later physical TJA can interrupt the hold; any replacement hold uses
    # only the remaining global budget. Stop immediately on raw-off and preserve MRCC
    # that was already armed before TJA.
    if self.CP_SP.flags & MazdaFlagsSP.TJA_BUTTON:
      tja_button = bool(getattr(CS, "tja_button", 0))
      crz_btns_counter = int(CS.crz_btns_counter)
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
      tja_released = not tja_button and self.tja_button_prev

      if tja_pressed:
        if not self.tja_mrcc_unarm_pending:
          # PEDALS may already show the TJA-induced arm in the same update as the button
          # edge. The previous stable sample is the state that existed before the press.
          mrcc_armed_before_press = self.tja_mrcc_armed_prev if self.tja_mrcc_armed_prev is not None else mrcc_armed
          if not mrcc_armed_before_press:
            # Acquire a new ownership episode. The cumulative three-frame budget resets
            # only here, never for a later TJA while leftover MRCC remains armed.
            self.tja_mrcc_unarm_pending = True
            self.tja_mrcc_saw_armed = False
            self.tja_mrcc_tx_frames = 0
            self.tja_mrcc_press_frames = 0
        elif self.tja_mrcc_press_frames > 0:
          # Route 5d: the second TJA ends this uninterrupted hold, but not ownership
          # of the TJA-caused arm. Any replacement hold uses only the global budget
          # remaining after the already-transmitted frames.
          self.tja_mrcc_press_frames = 0
        # Keep ownership and wait for the newest release. No TX while TJA is held.
        self.tja_mrcc_release_counter = None
        self.tja_mrcc_release_wait_frames = 0
        self.tja_mrcc_first_tx_not_before_nanos = None
        self.tja_mrcc_wait_for_fresh_counter_after_op = False

      if self.tja_mrcc_unarm_pending:
        self.tja_mrcc_saw_armed |= raw_mrcc_armed
        if (CS.cancel_button == 1 or getattr(CS, "resume_button", 0) == 1 or
            CS.accel_button or CS.decel_button or getattr(CS, "mrcc_button", 0) == 1):
          # Driver cruise-button activity owns CRZ_BTNS regardless of whether TJA
          # is held or a cleanup counter has been anchored.
          self.tja_mrcc_unarm_pending = False
          self.tja_mrcc_press_frames = 0
        elif (CC.cruiseControl.cancel or CC.cruiseControl.resume) and self.tja_mrcc_tx_frames > 0:
          # A synthetic hold has already spent budget. Do not allow cancel/resume
          # during TJA hold to resume later as a replacement press.
          self.tja_mrcc_unarm_pending = False
          self.tja_mrcc_press_frames = 0
        elif self.tja_mrcc_saw_armed and raw_off_confirmed:
          self.tja_mrcc_unarm_pending = False
          self.tja_mrcc_press_frames = 0
        elif tja_released:
          if self.tja_mrcc_tx_frames > 0 and not raw_mrcc_armed:
            # Delayed acknowledgement of an interrupted press can arrive while TJA is
            # held. Observe it before arming a replacement press.
            self.tja_mrcc_unarm_pending = False
            self.tja_mrcc_press_frames = 0
          else:
            # Experimentally delay only the first actual MRCC_OFF frame. Replacement
            # holds after transmission retain the existing consecutive-counter behavior.
            self.tja_mrcc_release_counter = crz_btns_counter
            self.tja_mrcc_release_wait_frames = 0
            self.tja_mrcc_first_tx_not_before_nanos = (
              now_nanos + TJA_MRCC_FIRST_TX_DELAY_NANOS if self.tja_mrcc_tx_frames == 0 else None
            )
            self.tja_mrcc_wait_for_fresh_counter_after_op = False
        elif self.tja_mrcc_release_counter is not None:
          if not raw_mrcc_armed:
            # Raw-off is sufficient to stop an in-flight transaction. Waiting for the
            # filtered state here could send another toggle after a manual/accepted off.
            self.tja_mrcc_unarm_pending = False
            self.tja_mrcc_press_frames = 0
          elif CC.cruiseControl.cancel or CC.cruiseControl.resume:
            # No budget has been spent. Wait for a new OEM counter after this
            # command clears rather than sending from an old retained sample.
            self.tja_mrcc_release_counter = crz_btns_counter
            self.tja_mrcc_release_wait_frames = 0
            self.tja_mrcc_wait_for_fresh_counter_after_op = True
          elif self.tja_mrcc_tx_frames >= TJA_MRCC_MAX_TX_FRAMES:
            self.tja_mrcc_unarm_pending = False
            self.tja_mrcc_press_frames = 0
          else:
            self.tja_mrcc_release_wait_frames += 1
            counter_delta = (crz_btns_counter - self.tja_mrcc_release_counter) % 16
            first_tx_waiting = (
              self.tja_mrcc_tx_frames == 0 and
              self.tja_mrcc_first_tx_not_before_nanos is not None
            )
            first_tx_due = (
              first_tx_waiting and
              now_nanos >= self.tja_mrcc_first_tx_not_before_nanos and
              not self.tja_mrcc_wait_for_fresh_counter_after_op
            )
            if first_tx_waiting:
              if self.tja_mrcc_wait_for_fresh_counter_after_op:
                if counter_delta == 1:
                  self.tja_mrcc_wait_for_fresh_counter_after_op = False
                elif counter_delta > 1:
                  self.tja_mrcc_release_counter = crz_btns_counter
              if not self.tja_mrcc_wait_for_fresh_counter_after_op:
                # Keep the release anchor synchronized with the latest OEM counter.
                # At the deadline, create_mrcc_off_cmd packs latest_counter + 1.
                self.tja_mrcc_release_counter = crz_btns_counter
                self.tja_mrcc_release_wait_frames = 0
                first_tx_due = now_nanos >= self.tja_mrcc_first_tx_not_before_nanos

            # Experimental change: the delayed first frame is deadline-gated rather
            # than counter-delta-gated. Follow-ups retain the delta == 1 requirement.
            if first_tx_due or (not first_tx_waiting and counter_delta == 1):
              if raw_mrcc_armed:
                can_sends.append(mazdacan.create_mrcc_off_cmd(self.packer, crz_btns_counter))
                tja_mrcc_cleanup_tx = True
                self.tja_mrcc_tx_frames += 1
                self.tja_mrcc_press_frames += 1
                self.tja_mrcc_release_counter = crz_btns_counter
                self.tja_mrcc_release_wait_frames = 0
                self.tja_mrcc_first_tx_not_before_nanos = None
                self.tja_mrcc_wait_for_fresh_counter_after_op = False
                if self.tja_mrcc_tx_frames >= TJA_MRCC_MAX_TX_FRAMES:
                  self.tja_mrcc_unarm_pending = False
                  self.tja_mrcc_press_frames = 0
              else:
                self.tja_mrcc_unarm_pending = False
                self.tja_mrcc_press_frames = 0
            elif first_tx_waiting:
              if self.tja_mrcc_release_wait_frames > TJA_MRCC_RELEASE_WAIT_FRAMES:
                self.tja_mrcc_unarm_pending = False
                self.tja_mrcc_press_frames = 0
            elif counter_delta > 1:
              if self.tja_mrcc_press_frames > 0:
                # The physical-style hold has been broken.
                self.tja_mrcc_unarm_pending = False
                self.tja_mrcc_press_frames = 0
              elif self.tja_mrcc_release_wait_frames > TJA_MRCC_RELEASE_WAIT_FRAMES:
                # Repeated pre-start jumps must not suppress ICBM forever.
                self.tja_mrcc_unarm_pending = False
                self.tja_mrcc_press_frames = 0
              else:
                # Press has not started. A skipped OEM counter is not a broken hold;
                # wait for the next consecutive counter from here.
                self.tja_mrcc_release_counter = crz_btns_counter
            elif self.tja_mrcc_release_wait_frames > TJA_MRCC_RELEASE_WAIT_FRAMES:
              # A dead/stale CRZ_BTNS stream must not suppress ICBM indefinitely.
              self.tja_mrcc_unarm_pending = False
              self.tja_mrcc_press_frames = 0

      if not self.tja_mrcc_unarm_pending:
        self.tja_mrcc_first_tx_not_before_nanos = None
        self.tja_mrcc_wait_for_fresh_counter_after_op = False

      self.tja_button_prev = tja_button
      self.tja_mrcc_armed_prev = mrcc_armed

    self.apply_torque_last = apply_torque

    if self.CP.openpilotLongitudinalControl:
      can_sends.extend(self.update_longitudinal(CC, CC_SP, CS))

    can_sends.extend(self.update_camera_tja(CC, CS))

    # CAM_LANEINFO.TJA=2 draws the WHITE wheel, but it is not display-only: the
    # Mazda body/MRCC consumes it too. Fail closed around cruise/TJA interaction.
    # Expose WHITE after MRCC has been OFF or ARMED (not ACTIVE) and quiet for
    # 0.5 s. ACTIVE remains a hard deny. stock_tja (camera RX) stays separate.
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
    mrcc_active = (
      bool(getattr(CS, "cruise_enabled", False)) or
      filtered_mrcc_enabled
    )
    # ARMED for this experiment: available/raw-armed without ACTIVE. Accidental
    # TJA-induced arms are excluded while the button is held (hud_button_activity)
    # and by the 0.5 s confirmation once the state is quiet.
    mrcc_armed = (
      not mrcc_active and
      cruise_state is not None and
      (
        bool(getattr(CS, "mrcc_armed_raw", False)) or
        bool(getattr(CS, "cruise_available", False)) or
        filtered_mrcc_available
      )
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

    tja_button_mazda = bool(self.CP_SP.flags & MazdaFlagsSP.TJA_BUTTON)
    white_hud_trusted = (
      tja_button_mazda and
      bool(getattr(getattr(CC_SP, "mads", None), "active", False)) and
      getattr(CS, "cam_laneinfo_live", False) and
      normalized_base and
      CC.hudControl.visualAlert == VisualAlert.none and
      not hud_button_activity
    )

    white_hud_base_allowed = (
      tja_button_mazda and
      white_hud_trusted and
      (mrcc_off or mrcc_armed)
    )
    if white_hud_base_allowed:
      self.mads_white_hud_off_frames = min(
        self.mads_white_hud_off_frames + 1,
        MADS_WHITE_HUD_OFF_CONFIRM_FRAMES,
      )
    else:
      self.mads_white_hud_off_frames = 0

    white_hud = (
      white_hud_base_allowed and
      self.mads_white_hud_off_frames >= MADS_WHITE_HUD_OFF_CONFIRM_FRAMES
    )
    withdraw_white_now = self.mads_white_hud_on_bus and not white_hud

    # Preserve the normal 2 Hz cadence. Exception: immediate OEM withdraw when WHITE
    # becomes unsafe (button / ACTIVE / warning / stale / unknown payload).
    if self.frame % 50 == 0 or withdraw_white_now:
      payload = hud_base if white_hud and hud_base is not None else packed_laneinfo
      alert = (alert[0], mazdacan.apply_mads_white_hud(fsc_raw, payload, white_hud), alert[2])
      can_sends.append(alert)
      self.mads_white_hud_on_bus = mazdacan.is_mads_white_hud(alert[1])

    # send steering command
    can_sends.append(mazdacan.create_steering_control(self.packer, self.CP,
                                                      self.frame, apply_torque, CS.cam_lkas))

    # Suppress ICBM while cancel/resume or cleanup owns CRZ_BTNS.
    icbm_suppress = (
      CC.cruiseControl.cancel or CC.cruiseControl.resume or CS.cancel_button == 1 or
      (tja_button_mazda and
       (getattr(CS, "tja_button", 0) == 1 or self.tja_mrcc_unarm_pending or tja_mrcc_cleanup_tx))
    )
    if not icbm_suppress:
      can_sends.extend(IntelligentCruiseButtonManagementInterface.update(self, CC_SP, CS, self.packer, self.frame, self.last_button_frame))

    new_actuators = CC.actuators.as_builder()
    new_actuators.torque = apply_torque / steer_max
    new_actuators.torqueOutputCan = apply_torque
    # Report the command sent on the wire after clipping, holds, slew, and overrides.
    new_actuators.accel = self.accel_last

    self.frame += 1
    return new_actuators, can_sends

  def update_camera_tja(self, CC, CS):
    """Press the camera's own TJA/CTS off, on its bus, whenever openpilot steers with it armed.

    The two lane-centering systems must never run at once: the panda drops the camera's 0x243
    while openpilot controls, but the camera keeps its state and takes the wheel the moment
    lateral drops (route 00000018--5655da2c1c seg 15). One CRZ_BTNS with the TJA bit over the
    wheel's idle pattern, counter plus one, on bus 2; the forwarded real stream supplies the
    release and the camera acts on the press edge (tja_cts_route_29). At least one 0x440
    period between presses, three per arming episode; the episode resets when the camera
    reads 0, so a driver re-arming it under us is handled again. Not gated on the button
    declaration: any Mazda steering with the camera armed gets the same press.
    """
    can_sends = []
    if CS.stock_tja == 0:
      self.tja_press_count = 0
      self.tja_press_frame = None
      self.tja_episode_alerted = False
    elif CC.latActive:
      interval = int(CarControllerParams.TJA_PRESS_INTERVAL_T / DT_CTRL)
      due = self.tja_press_frame is None or self.frame - self.tja_press_frame >= interval
      if due and self.tja_press_count < CarControllerParams.TJA_PRESS_MAX:
        can_sends.append(mazdacan.create_button_cmd(self.packer, self.CP, CS.crz_btns_counter, Buttons.TJA, bus=2))
        self.tja_press_count += 1
        self.tja_press_frame = self.frame
      elif due and not self.tja_episode_alerted:
        # The camera did not clear: keep steering (its command is blocked) and tell the driver
        # once. carstate turns this into the one-shot stockLkas pulse.
        CS.stock_cts_stuck = True
        self.tja_episode_alerted = True
    return can_sends

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
    setup_ok = CS.fsc_settled and not (stock_radar_alive and CS.out.cruiseState.enabled)
    session_state = self.radar_session.update(setup_ok, stock_radar_alive, CC_SP.stockEcuHandBack,
                                              standstill=CS.out.standstill,
                                              session_refused=CS.radar_session_refused,
                                              stock_radar_gone=CS.stock_radar_gone)
    # Continue synthetic radar frames through hand-back to avoid a camera-visible gap.
    radar_master = session_state in (RadarSessionState.SILENCED, RadarSessionState.HANDBACK)

    if self.frame % CarControllerParams.RADAR_UDS_STEP == 0:
      if session_state == RadarSessionState.SILENCING:
        can_sends.append(create_radar_session_msg(uds.SESSION_TYPE.PROGRAMMING))
      elif session_state == RadarSessionState.HANDBACK:
        can_sends.append(create_radar_session_msg(uds.SESSION_TYPE.DEFAULT))
      elif session_state == RadarSessionState.SILENCED:
        # Tester-present frames keep the radar silent in its diagnostic session.
        can_sends.append(make_tester_present_msg(RADAR_ADDR, 0, suppress_response=True))

    stopping = CC.actuators.longControlState == LongCtrlState.stopping
    # Engaged bits follow CC.enabled. Gas is an override, not a disengagement.
    long_engaged = CC.enabled
    sm = self.stop_and_go
    sm.update(long_engaged, stopping, CS.out.standstill, CC.actuators.accel, CS.brake_hold,
              gas_pressed=CS.out.gasPressed)
    # Lead advertisement represents perception and is independent of engagement.
    self.lead_adv.update(CC.hudControl.leadVisible, CC_SP.leadOne.dRel,
                         CC_SP.leadOne.vRel, sm.holding)

    if sm.just_released:
      # Never-latched stops relax in one frame; latched holds ramp from the relaxed command.
      self.release_ramp = CarControllerParams.ACCEL_HOLD_LATCHED if sm.latched_release else \
                          CarControllerParams.ACCEL_RELEASE_BAND
    elif sm.holding or not CC.longActive:
      # Re-holds and driver overrides terminate the release ramp.
      self.release_ramp = None

    accel = 0.
    if CC.longActive:
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
        # at the relaxed command until GEAR.BRAKE_HOLD clears.
        accel = self.release_ramp
        if not (sm.latched_release and CS.brake_hold):
          # Follow a falling plan ceiling at the winddown limit.
          self.release_ramp = max(min(self.release_ramp + CarControllerParams.ACCEL_RELEASE_RAMP * DT_CTRL, ramp_ceiling),
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
      acc_available = CS.out.cruiseState.available
      # Mirror the driver's distance setting; stock defaults to gap 2.
      gap = (int(CC.hudControl.leadDistanceBars) or 2) if (long_engaged or acc_available) else 0
      acc_active_2 = sm.acc_active_2 if long_engaged else False
      for bus in LONG_BUSES:
        can_sends.append(mazdacan.create_acc_command(self.packer, bus, self.long_counter, accel,
                                                     long_active=long_engaged, acc_available=acc_available,
                                                     brake_pressed=CS.out.brakePressed,
                                                     stopping=sm.stop_bits, resume_unlatching=sm.resume_unlatching))
        crz_ctrl = mazdacan.create_crz_ctrl(self.packer, bus, long_engaged, acc_available, gap,
                                            self.lead_adv.has_lead, self.lead_adv.ctrl_phase,
                                            acc_active_2)
        if (bus == 0 and bool(getattr(getattr(CC_SP, "mads", None), "active", False)) and
            not CS.mrcc_armed_raw and not CS.cruise_available and not CS.cruise_enabled):
          addr, dat, crz_bus = crz_ctrl
          dat = bytearray(dat)
          dat[4] |= 0x20
          crz_ctrl = (addr, bytes(dat), crz_bus)
        can_sends.append(crz_ctrl)
      self.long_counter += 1

    return can_sends
