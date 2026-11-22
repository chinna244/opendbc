from opendbc.car.mazda import context
"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from enum import StrEnum

from opendbc.can.parser import CANParser
from opendbc.car import Bus, DT_CTRL, create_button_events, structs
from opendbc.car.mazda.icbm import BUTTONS
from opendbc.car.mazda.values_ext import MazdaFlagsSP
class MadsCarStateBase:
  def __init__(self, CP, CP_SP):
    self.CP, self.CP_SP = CP, CP_SP
    self.lkas_button = self.prev_lkas_button = 0


VisualAlert = structs.CarControl.HUDControl.VisualAlert
ButtonType = structs.CarState.ButtonEvent.Type

# The ON shortcut becomes a toggle event only after disabled standby has settled.
# The same window rejects feedback that the control loop has stopped refreshing.
MADS_ON_STANDBY_FRAMES = round(0.5 / DT_CTRL)

TJA_MRCC_MAX_TX_FRAMES = 3
TJA_MRCC_RAW_OFF_CONFIRM_FRAMES = 5
# The body ECU drops discrete presses faster than one per 200 ms (measured, docs/zoompilot/icbm.md);
# the arm clears in PEDALS ~80 ms after a press plus the 50 ms confirm.
TJA_MRCC_TX_PERIOD = 0.2  # s between undo presses
# The press-induced arm appears ~80 ms after the press edge (docs/zoompilot/mazda-lateral.md).
TJA_MRCC_ARM_WAIT_FRAMES = int(1.0 / DT_CTRL)
# The white wheel waits this long on a fully-off, quiet cruise before it displays.
MADS_WHITE_HUD_OFF_CONFIRM_FRAMES = int(0.5 / DT_CTRL)

# The camera's own TJA (CAM_LANEINFO.TJA, native) must read nonzero this long, fresh, while
# openpilot steers before it is pressed off: a passed press shows TJA=2 for under 0.5 s on its own.
CAMERA_TJA_CONFIRM_FRAMES = int(1.0 / DT_CTRL)
# Press-off pacing and budget per episode.
CAMERA_TJA_PRESS_FRAMES = int(1.0 / DT_CTRL)
CAMERA_TJA_PRESS_MAX = 3
# The panda's shared budget for every camera-only press (mazda.h MAZDA_CAM_PRESS_*): >= 0.9 s apart,
# 4 per burst, the burst over after 5 s without one. Mirrored with margin across back-to-back
# episodes, so no press is sent that the panda would refuse.
CAMERA_PRESS_MIN_FRAMES = int(1.0 / DT_CTRL)
CAMERA_PRESS_BURST_MAX = 4
CAMERA_PRESS_QUIET_FRAMES = int(5.1 / DT_CTRL)


# The MADS white wheel: the dash draws it when the camera's own HUD frame carries TJA=2,
# and the body reads the same frame. The white bit is only ever XORed into an exact
# camera payload audited to be an idle frame, never a frame we composed. Every observed
# FSC idle family on TJA-declared cars is enumerated: the OFF family with its
# counter-nibble twins, the LINE_VISIBLE families and their high-beam variants, and the
# partial-lane LANE_LINES=3/4 encodings. Exact bases only; do not widen to a field-based
# rule until more captures are audited. No base may carry ERR_BIT, NO_ERR_BIT (byte 1 0x40,
# unsettled camera in carstate's takeover gate), LDW or a hands warning.
MADS_HUD_SAFE_BASE_PAYLOADS = frozenset(bytes.fromhex(h) for h in (
  "4201000000001040", "4201000000001060", "4221000000004040", "4221000000001040",
  "4221000000001060", "4201000000004040", "0221000000000040", "4201000000000040",
  "4221000000000040", "0221000000001040", "4102000000001040",
  "4122000000001040", "4102000000004040", "4122000000004040",
  "4221000000004060", "4122000000000040", "4103000000001040", "4104000000001040",
  "4123000000000040", "4124000000000040", "4123000000001040", "4124000000001040",
  "4123000000004040", "4124000000004040", "4102000000001060", "4102000000004060",
  "4122000000001060", "4122000000004060", "0122000000000040", "0122000000004040",
  "4202000000001040", "4102000000000040",
  # one-line LANE_LINES=3/4 with the high-beam bit (night): routes 00000072/75, steering above 40 mph
  "4103000000004040", "4104000000004040",
))
# OFF to WHITE is TJA 0 to 2 only: one bit, byte 4 0x20, XORed in, never a frame swap.
MADS_HUD_WHITE_TJA_XOR = bytes.fromhex("0000000020000000")
# 64-bit big-endian keep-mask over the bits an idle camera may still move between
# samples: TJA (byte 4, 0x70), TJA_TRANSITION (byte 3, 0x0C), and the unnamed byte-3
# transition bits 0x03. Do not clear byte-4 0x80 or unrelated byte-0 family bits.
# Every allowlisted base carries zero in the masked bits, so no two bases share a key.
CAM_LANEINFO_TJA_NORMALIZE_MASK = 0xFFFFFFF08FFFFFFF
_MADS_HUD_SAFE_BASE_BY_INT = {
  int.from_bytes(b, "big") & CAM_LANEINFO_TJA_NORMALIZE_MASK: b
  for b in MADS_HUD_SAFE_BASE_PAYLOADS
}


def white_hud_allowlist_base(fsc_raw: bytes | None) -> bytes | None:
  """The allowlisted idle base for the camera's current frame, TJA/transition bits ignored."""
  if fsc_raw is None or len(fsc_raw) != 8:
    return None
  return _MADS_HUD_SAFE_BASE_BY_INT.get(
    int.from_bytes(fsc_raw, "big") & CAM_LANEINFO_TJA_NORMALIZE_MASK
  )


def apply_mads_white_hud(fsc_raw: bytes | None, packed_dat: bytes, enabled: bool) -> bytes:
  """Set TJA=2 on the camera's own allowlisted idle frame, and on nothing else.

  packed_dat must be exactly the base the camera's current frame normalizes to: any other
  payload, or an unknown camera frame, passes through untouched.
  """
  if not enabled or len(packed_dat) != 8:
    return packed_dat
  if packed_dat != white_hud_allowlist_base(fsc_raw):
    return packed_dat
  return bytes(a ^ b for a, b in zip(packed_dat, MADS_HUD_WHITE_TJA_XOR, strict=True))


def with_camera_lane_keep(dat: bytes, fsc_raw: bytes | None) -> bytes:
  """The camera's lane-keep state (byte 3 low nibble: TJA_TRANSITION 0x0C, unnamed 0x03) on our frame.

  Routes 0000006e/6f: camera armed, our byte 3 zeroed, no dash lines; with the C4 unplugged the
  dash gets the camera's frame and draws them. A stale or missing camera frame sends 0.
  EXPERIMENT, not a confirmed cause: Sept drives also sent byte 3 zeroed, unexplained.
  """
  if fsc_raw is None or len(fsc_raw) != 8 or len(dat) != 8:
    return dat
  return dat[:3] + bytes([(dat[3] & 0xF0) | (fsc_raw[3] & 0x0F)]) + dat[4:]


def create_mrcc_off_cmd(packer, counter):
  # The wheel's MRCC master press, active-low: every button bit 0 with its inversion 1,
  # the master signature in BIT1/BIT1_INV plus BIT2/BIT3, counter plus one. Only the
  # TJA-press cleanup sends it, and the panda pins the exact bytes in mazda_mrcc_off_msg_valid.
  values = {
    "CAN_OFF_INV": 1, "SET_P_INV": 1, "RES_INV": 1, "SET_M_INV": 1,
    "DISTANCE_LESS_INV": 1, "DISTANCE_MORE_INV": 1, "MODE_X_INV": 1, "MODE_Y_INV": 1,
    "BIT1_INV": 1, "BIT2": 1, "BIT3": 1,
    "CTR": (counter + 1) % 16,
  }
  return packer.make_can_msg("CRZ_BTNS", 0, values)


def create_camera_tja_press(packer, counter):
  # The wheel's TJA press over its idle pattern, on the camera bus only (route 00000097:
  # 0009ffcc00000000). The panda pins the exact bytes in mazda_cam_tja_press_msg_valid.
  values = {
    "TJA_BUTTON": 1,
    "CAN_OFF_INV": 1, "SET_P_INV": 1, "RES_INV": 1, "SET_M_INV": 1,
    "DISTANCE_LESS_INV": 1, "DISTANCE_MORE_INV": 1, "MODE_X_INV": 1, "MODE_Y_INV": 1,
    "BIT1": 1, "BIT2": 1, "BIT3": 1,
    "CTR": (counter + 1) % 16,
  }
  return packer.make_can_msg("CRZ_BTNS", 2, values)


class MadsCarController:
  """The TJA button as the MADS switch, on the CarController (its packer, frame and
  last_button_frame): the press-induced MRCC arm undone, and the white wheel on the camera's
  own HUD frame."""

  def __init__(self):
    self.tja_button_prev = False
    self.mrcc_armed_prev: bool | None = None
    self.mrcc_undo_pending = False
    self.mrcc_undo_saw_armed = False
    self.mrcc_undo_frames = 0
    self.mrcc_raw_off_frames = 0
    self.mrcc_arm_wait_frames = 0
    # The white wheel rides the alert frame; on_bus tracks that the white bit is on the wire.
    self.mads_white_hud_off_frames = 0
    self.mads_white_hud_on_bus = False
    # The camera's own TJA pressed off while openpilot steers (one episode per nonzero stretch).
    self.camera_tja_frames = 0
    self.camera_tja_presses = 0
    self.camera_tja_press_frame: int | None = None
    # Every camera-only press: the panda's burst as this side counts it.
    self.camera_press_burst = 0
    self.camera_press_frame: int | None = None

  def camera_press_ready(self) -> bool:
    """Whether the panda's shared camera-press budget has room for one now."""
    if self.camera_press_frame is None:
      return True
    since = self.frame - self.camera_press_frame
    if since >= CAMERA_PRESS_QUIET_FRAMES:
      self.camera_press_burst = 0
    return since >= CAMERA_PRESS_MIN_FRAMES and self.camera_press_burst < CAMERA_PRESS_BURST_MAX

  def camera_press(self, CS):
    self.camera_press_burst += 1
    self.camera_press_frame = self.frame
    return create_camera_tja_press(self.packer, CS.crz_btns_counter)

  def update_camera_tja(self, CC, CS):
    """Press the camera's own TJA off, on the camera bus only, while openpilot steers.

    Every TJA press reaches the camera again (cruise on or off) so its lane-keep state arms the
    dash lane lines; with MRCC main on a press can also latch the camera's own TJA. Leave native
    TJA=2 (standby) alone to preserve the camera's HUD state. Active camera TJA must not run
    alongside openpilot's steering (route 0000013b: FSC malfunction). Confirm a fresh native state
    other than 0 or 2 for CAMERA_TJA_CONFIRM_FRAMES, then send up to CAMERA_TJA_PRESS_MAX presses
    CAMERA_TJA_PRESS_FRAMES apart, each only when the panda's shared burst has room
    (camera_press_ready). Native TJA=0 or 2 ends the episode. A persistent active state after the
    budget raises camera_tja_stuck. Stale camera data or openpilot not steering sends nothing.
    """
    can_sends = []
    if not self.CP_SP.flags & MazdaFlagsSP.TJA_BUTTON:
      return can_sends

    native_tja = int(CS.cam_laneinfo["TJA"]) if CS.cam_laneinfo_live else None
    if native_tja in (0, 2):
      self.camera_tja_frames = 0
      self.camera_tja_presses = 0
      self.camera_tja_press_frame = None
      CS.camera_tja_stuck = False
      return can_sends

    if native_tja is None or not CC.latActive:
      # Unknown, or the camera steering alone (allowed): keep the episode's budget, restart the
      # confirmation, and let the camera be.
      self.camera_tja_frames = 0
      if not CC.latActive:
        CS.camera_tja_stuck = False
      return can_sends

    self.camera_tja_frames = min(self.camera_tja_frames + 1, CAMERA_TJA_CONFIRM_FRAMES)
    # Never inside the driver's own held press.
    if self.camera_tja_frames < CAMERA_TJA_CONFIRM_FRAMES or CS.tja_button:
      return can_sends

    due = self.camera_tja_press_frame is None or self.frame - self.camera_tja_press_frame >= CAMERA_TJA_PRESS_FRAMES
    if due and self.camera_tja_presses < CAMERA_TJA_PRESS_MAX:
      # An attempt counts only when it is sent; a full panda burst means wait, not a spent attempt.
      if self.camera_press_ready():
        can_sends.append(self.camera_press(CS))
        self.camera_tja_presses += 1
        self.camera_tja_press_frame = self.frame
    elif due:
      CS.camera_tja_stuck = True
    return can_sends

  def update_mrcc_cleanup(self, CC, CC_SP, CS, handback_active: bool):
    # Undo the MRCC arm a physical TJA press causes: the wheel's press reaches every bus-0
    # ECU directly, so the button the driver declared as the lateral switch also arms cruise.
    # Answer with the driver's own MRCC master press until raw PEDALS confirms the arm is gone.
    can_sends = []
    if not self.CP_SP.flags & MazdaFlagsSP.TJA_BUTTON:
      return can_sends

    raw_armed = bool(CS.mrcc_armed_raw)
    # PEDALS drops both cruise bits through a brake transition; the filtered state bridges
    # those samples, and only sustained raw-off is authoritative.
    self.mrcc_raw_off_frames = 0 if raw_armed else self.mrcc_raw_off_frames + 1
    raw_off_confirmed = self.mrcc_raw_off_frames >= TJA_MRCC_RAW_OFF_CONFIRM_FRAMES
    # PEDALS holds both cruise bits low under braking: the clock restarts there and runs
    # again once the pedal is off.
    self.mrcc_arm_wait_frames = 0 if CS.out.brakePressed else self.mrcc_arm_wait_frames + 1
    mrcc_armed = raw_armed or (CS.cruise_available and not raw_off_confirmed)

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
    if driver_activity or handback_active or CC_SP.stockEcuHandBack:
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
      can_sends.append(create_mrcc_off_cmd(self.packer, CS.crz_btns_counter))
      self.last_button_frame = self.frame
      self.mrcc_undo_frames += 1
      if self.mrcc_undo_frames >= TJA_MRCC_MAX_TX_FRAMES:
        self.mrcc_undo_pending = False

    return can_sends

  def update_white_hud(self, CC, CC_SP, CS, handback_active: bool) -> bool:
    """Whether the alert frame carries the white wheel.

    On a TJA-declared car with MADS active and cruise fully off, or with openpilot steering and
    cruising under alpha long that owns cruise (radar held silent, or the Smart CAN Filter), the white-wheel TJA=2 state
    is XORed into the camera's current payload, but only onto an exact allowlisted idle base:
    TJA=2 is not display-only, the body reads the same frame. Every cruise interaction fails
    closed, and a white wheel HUD state that became unsafe is withdrawn immediately, outside
    the cadence.
    """
    session_ambiguous = handback_active or CC_SP.stockEcuHandBack
    mrcc_off = (not session_ambiguous and not CS.mrcc_armed_raw and
                not CS.cruise_available and not CS.cruise_enabled)
    # Alpha long engaged with openpilot owning cruise: openpilot steering and cruising carries the
    # white state too, and with it the dash's lane lines (routes 00000097 to 0000013e drew them only
    # from TJA=2 frames). Owned = the radar held silent (UDS), or the Smart CAN Filter's confirmed
    # mode: that radar stays online and reads this frame with its main on in standby, its cruise
    # frames kept off the car by the board (the user's 2026-11-15 choice).
    owned = self.can_filter.control_active if self.can_filter is not None else self.radar_session.control_active
    cruising = (not session_ambiguous and CC.enabled and CC.latActive and owned and
                CS.radar_bus_healthy and CS.out.canValid)

    # Every button a TJA wheel carries: TJA, MRCC, SET+/-, RES, DISTANCE, plus the
    # synthesized ICBM set presses and openpilot's own cancel/resume.
    button_activity = (CS.tja_button or CS.mrcc_button or CS.cancel_button or CS.resume_button or
                       CS.accel_button or CS.decel_button or CS.distance_button or
                       CC_SP.intelligentCruiseButtonManagement.sendButton in BUTTONS or
                       CC.cruiseControl.cancel or CC.cruiseControl.resume)

    white_allowed = (
      bool(self.CP_SP.flags & MazdaFlagsSP.TJA_BUTTON) and
      CS.cam_laneinfo_live and
      white_hud_allowlist_base(CS.cam_laneinfo_raw) is not None and
      CC.hudControl.visualAlert == VisualAlert.none and
      not button_activity and
      ((CC_SP.mads.active and mrcc_off) or cruising)
    )
    if white_allowed:
      self.mads_white_hud_off_frames = min(self.mads_white_hud_off_frames + 1, MADS_WHITE_HUD_OFF_CONFIRM_FRAMES)
    else:
      self.mads_white_hud_off_frames = 0
    return white_allowed and self.mads_white_hud_off_frames >= MADS_WHITE_HUD_OFF_CONFIRM_FRAMES

  def white_hud_frame(self, CS, alert, white: bool):
    """The alert frame as sent: the camera's own allowlisted idle base with TJA=2 when white."""
    fsc_raw = CS.cam_laneinfo_raw
    payload = white_hud_allowlist_base(fsc_raw) if white else alert[1]
    self.mads_white_hud_on_bus = white
    dat = apply_mads_white_hud(fsc_raw, payload, white)
    return alert[0], with_camera_lane_keep(dat, fsc_raw if CS.cam_laneinfo_live else None), alert[2]


class MadsCarState(MadsCarStateBase):
  def __init__(self, CP: structs.CarParams, CP_SP: context.CarParamsSP):
    super().__init__(CP, CP_SP)
    # Unfiltered PEDALS cruise state; the filtered public state bridges brake dropouts.
    self.mrcc_armed_raw = False
    self.tja_button = 0
    self.prev_tja_button = 0
    # Active-low wheel MRCC master (CRZ_BTNS.BIT1), read from the bus-0 parser: a 0 is a press.
    self.mrcc_button = 0
    self.prev_mrcc_button = 0
    self.crz_btns_seen = False
    self.mads_available = False
    self.mads_enabled = False
    self.mads_feedback_age = MADS_ON_STANDBY_FRAMES
    self.mads_standby_frames = 0
    self.main_on_button = False
    self.main_on_lkas_button = False
    self.prev_main_on_lkas_button = False

  def update_mads_feedback(self, CC: structs.CarControl, CC_SP: context.CarControlSP) -> None:
    self.mads_available = CC_SP.mads.available
    self.mads_enabled = CC_SP.mads.enabled or CC_SP.mads.active or CC.latActive
    self.mads_feedback_age = 0

  def main_on_lkas_events(self, ret: structs.CarState, btns) -> list[structs.CarState.ButtonEvent]:
    self.mads_feedback_age = min(self.mads_feedback_age + 1, MADS_ON_STANDBY_FRAMES)
    on_pressed = btns["MODE_Y"] == 1
    brake_restricted = True  # StarPilot owns brake policy; never rearm from a held ON while braking.
    standby = (not self.CP_SP.flags & MazdaFlagsSP.TJA_BUTTON and self.mads_available and not self.mads_enabled and
               self.mads_feedback_age < MADS_ON_STANDBY_FRAMES and ret.cruiseState.available and self.out.cruiseState.available and
               not self.cruise_enabled and not self.out.cruiseState.enabled and
               not (self.cancel_button or self.resume_button or self.accel_button or self.decel_button) and
               not (brake_restricted and (ret.brakePressed or ret.regenBraking)))
    self.mads_standby_frames = min(self.mads_standby_frames + 1, MADS_ON_STANDBY_FRAMES) if standby else 0

    self.prev_main_on_lkas_button = self.main_on_lkas_button
    if not on_pressed:
      self.main_on_lkas_button = False
    elif not self.main_on_button and self.mads_standby_frames >= MADS_ON_STANDBY_FRAMES:
      # The shared MADS handler toggles on a pressed LKAS event. Emit once, only
      # from settled disabled standby, then wait for feedback before another request.
      self.main_on_lkas_button = True
      self.mads_standby_frames = 0
    self.main_on_button = on_pressed
    return create_button_events(self.main_on_lkas_button, self.prev_main_on_lkas_button, {True: ButtonType.lkas})

  def update_mads(self, ret: structs.CarState, can_parsers: dict[StrEnum, CANParser]) -> None:
    cp = can_parsers[Bus.pt]

    # Both longitudinal modes: the TJA-press cleanup and the white-wheel HUD gate read it.
    self.mrcc_armed_raw = cp.vl["PEDALS"]["ACC_OFF"] == 1 or cp.vl["PEDALS"]["ACC_ACTIVE"] == 1

    self.prev_mrcc_button = self.mrcc_button
    self.prev_tja_button = self.tja_button
    btns = cp.vl["CRZ_BTNS"]
    # BIT1 is active-low: a 0 on the bus-0 parser is the wheel's MRCC master press. Gated on
    # the declaration (an undeclared wheel's idle level is unknown) and held unpressed until
    # the wheel's first frame: parser zeros before it would decode as a phantom press.
    if self.CP_SP.flags & MazdaFlagsSP.TJA_BUTTON:
      self.crz_btns_seen = self.crz_btns_seen or len(cp.vl_all["CRZ_BTNS"]["BIT1"]) > 0
      self.mrcc_button = int(btns["BIT1"] == 0) if self.crz_btns_seen else 0
    else:
      self.mrcc_button = 0
    # Only a car declared to have the physical TJA button reports it as the MADS switch.
    self.tja_button = int(btns["TJA_BUTTON"] == 1) if self.CP_SP.flags & MazdaFlagsSP.TJA_BUTTON else 0
