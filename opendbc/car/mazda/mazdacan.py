from opendbc.car.can_definitions import CanData
from opendbc.car.mazda.values import Buttons, CarControllerParams, MazdaFlags

# Captured empty radar tracks required by the body ECU for stop-and-go. Only the counter
# nibble changes.
# The real radar sends byte 2 0xe0 in every log, cruising or not (Sept-Nov captures); 0xc0 was never stock.
RADAR_STATIC_MSG = (0x499, bytes.fromhex("0008e00000000000"))
RADAR_TRACK_MSGS = {
  0x361: bytes.fromhex("fff7fefe1fc00080"),
  0x362: bytes.fromhex("fff7fefe1fc78c80"),
  0x363: bytes.fromhex("fff7fefe1fc00000"),
  0x364: bytes.fromhex("fff7fefe1fc00000"),
  0x365: bytes.fromhex("fff7fe7ffbff3fc0"),
  0x366: bytes.fromhex("fff7fe7ffbff3fc0"),
}

# The G46L radar (2016.5 bodies) sends only this static frame and no track messages at
# all; fully static — no counter, no checksum.
G46L_RADAR_STATIC_MSG = (0x499, bytes.fromhex("0098400000000000"))


def crz_info_checksum(dat: bytes) -> int:
  # Invert the sum of the first seven bytes, excluding STOPPING and RESUME_UNLATCHING.
  return (0xFF - ((sum(dat[:7]) - (dat[5] & 0x04) - (dat[6] & 0x40)) & 0xFF)) & 0xFF


def create_acc_command(packer, bus, counter, accel, *, long_active, acc_available,
                       brake_pressed=False, stopping=False, resume_unlatching=False):
  # CRZ_INFO replaces the disabled radar's acceleration command and armed-idle state.
  values = {
    "ERROR_STATUS": 1,
    "STATIC_1": 0x7ff,
    "CTR": counter % 16,
    "ACCEL_CMD": accel if long_active else 8190 * CarControllerParams.ACCEL_CMD_LSB - 4.,  # stock non-controlling sentinel, raw 8190
    "NEW_SIGNAL_7": int(long_active or acc_available),
  }
  if long_active:
    values.update({
      "ACC_ACTIVE": 1,
      "ACC_SET_ALLOWED": 1,
      "STOPPING": int(stopping),
      "STOPPING_2": int(stopping),
      "RESUME_UNLATCHING": int(resume_unlatching),
    })
  elif acc_available:
    values["ACC_SET_ALLOWED"] = int(not brake_pressed)

  dat = packer.make_can_msg("CRZ_INFO", bus, values)[1]
  values["CHKSUM"] = crz_info_checksum(dat)
  return packer.make_can_msg("CRZ_INFO", bus, values)


def create_crz_ctrl(packer, bus, long_active, acc_available, gap_setting, acc_active_2, *, lead_distance=0,
                    hbc_request=False, hands_off=False, lkas_white=False, lane_info=None):
  # CRZ_CTRL replaces radar cruise state and mirrors the driver's gap selection. The cluster draws
  # the lead from RADAR_HAS_LEAD; RADAR_LEAD_RELATIVE_DISTANCE is stock's 1 to 5 closeness bucket.
  # NEW_SIGNAL_3 (bit 13) relays the camera's high-beam request (0x440 BIT2) as the stock radar
  # does; the body raises the lamps and the cluster's green HBC light on it.
  # hands_off: the stock radar's CRZ_CTRL while the OEM TJA camera asks for hands on the wheel
  # (route 0000003c, dashcam, seg 6 ~380 s): byte 6 bits 0, 1 and 3 set (HANDS_OFF_STEERING and two
  # unnamed bits), HANDS_ON_STEER_WARN 0. The 1/9 pair (our old 0x440 level 7 mirrored) drew the
  # "Please hold steering wheel" text instead of the wheel icon.
  # MSG_1/MSG_1_INV and MSG_1_INV_COPY/MSG_1_COPY/NEW_SIGNAL_6 are the stock radar's copy of 0x440
  # LINE_VISIBLE/LINE_NOT_VISIBLE and the 3-bit LANE_LINES (99.9% of 488k pre-takeover frames). The
  # dash draws the lane lines from them; the fixed "not visible, LANE_LINES 1" kept them off.
  # lane_info is (LINE_VISIBLE, LINE_NOT_VISIBLE, LANE_LINES); None keeps that fixed state.
  visible, not_visible, lane_lines = lane_info if lane_info is not None else (0, 1, 1)
  values = {
    "MSG_1": int(visible),
    "MSG_1_INV": int(not_visible),
    "MSG_1_INV_COPY": int(lane_lines) & 1,
    "MSG_1_COPY": (int(lane_lines) >> 1) & 1,
    "NEW_SIGNAL_6": (int(lane_lines) >> 2) & 1,
    "NEW_SIGNAL_8": 1,
    "CRZ_ACTIVE": int(long_active),
    "CRZ_AVAILABLE": int(long_active or acc_available),
    "DISTANCE_SETTING": gap_setting,
    "RADAR_HAS_LEAD": int(lead_distance > 0),
    "RADAR_LEAD_RELATIVE_DISTANCE": lead_distance,
    "ACC_ACTIVE_2": int(acc_active_2),
    "NEW_SIGNAL_3": int(hbc_request),
    "HANDS_OFF_STEERING": int(hands_off),
  }
  addr, dat, bus = packer.make_can_msg("CRZ_CTRL", bus, values)
  if lkas_white:
    dat = dat[:4] + bytes([dat[4] | 0x20]) + dat[5:]
  if hands_off:
    dat = dat[:6] + bytes([dat[6] | 0x0a]) + dat[7:]
  return addr, dat, bus


def create_radar_frames(bus, counter, g46l=False):
  if g46l:
    return [CanData(G46L_RADAR_STATIC_MSG[0], G46L_RADAR_STATIC_MSG[1], bus)]
  frames = [CanData(RADAR_STATIC_MSG[0], RADAR_STATIC_MSG[1], bus)]
  for addr, dat in RADAR_TRACK_MSGS.items():
    frames.append(CanData(addr, dat[:7] + bytes([(dat[7] & 0xf0) | (counter % 16)]), bus))
  return frames


def create_steering_control(packer, CP, frame, apply_torque, lkas):

  tmp = apply_torque + 2048

  lo = tmp & 0xFF
  hi = tmp >> 8

  # copy values from camera
  b1 = int(lkas["BIT_1"])
  # Never the camera's error bits: alone they are sun glare, and the EPS stops assisting on them
  # (docs/zoompilot/mazda-lateral.md, "LKAS_FAULT and the camera's ERR_BIT_1").
  er1 = 0
  lnv = 0
  ldw = 0
  er2 = 0

  # Some older models do have these, newer models don't.
  # Either way, they all work just fine if set to zero.
  steering_angle = 0
  b2 = 0

  tmp = steering_angle + 2048
  ahi = tmp >> 10
  amd = (tmp & 0x3FF) >> 2
  amd = (amd >> 4) | ((amd & 0xF) << 4)
  alo = (tmp & 0x3) << 2

  ctr = frame % 16
  # bytes:     [    1  ] [ 2 ] [             3               ]  [           4         ]
  csum = 249 - ctr - hi - lo - (lnv << 3) - er1 - (ldw << 7) - (er2 << 4) - (b1 << 5)

  # bytes      [ 5 ] [ 6 ] [    7   ]
  csum = csum - ahi - amd - alo - b2

  if ahi == 1:
    csum = csum + 15

  if csum < 0:
    if csum < -256:
      csum = csum + 512
    else:
      csum = csum + 256

  csum = csum % 256

  values = {}
  if CP.flags & MazdaFlags.GEN1:
    values = {
      "LKAS_REQUEST": apply_torque,
      "CTR": ctr,
      "ERR_BIT_1": er1,
      "LINE_NOT_VISIBLE": lnv,
      "LDW": ldw,
      "BIT_1": b1,
      "ERR_BIT_2": er2,
      "STEERING_ANGLE": steering_angle,
      "ANGLE_ENABLED": b2,
      "CHKSUM": csum
    }

  return packer.make_can_msg("CAM_LKAS", 0, values)


def create_alert_command(packer, cam_msg: dict, ldw: bool, steer_required: bool):
  # Preserve camera LKAS state. Keep TJA modes clear because its state machine does not own
  # the injected steering command.
  values = {s: cam_msg[s] for s in [
    "LINE_VISIBLE",
    "LINE_NOT_VISIBLE",
    "LANE_LINES",
    "BIT1",
    "BIT2",
    "BIT3",
    "NO_ERR_BIT",
    "ERR_BIT",
    "S1",
    "S1_HBEAM",
  ]}
  values.update({
    # The OEM TJA camera's own hands-on-wheel request (route 0000003c, dashcam, seg 6 ~380 s):
    # HANDS_WARN_3_BITS 5 plus byte 6 bit 0 (no DBC signal), both HANDS_ON_STEER_WARN bits clear.
    # Level 7 with HANDS_ON_STEER_WARN_2 (route 00000267) draws the "Please hold steering wheel"
    # text instead of the wheel icon. The camera's own frames with HANDS_ON_STEER_WARN are
    # lane-departure warnings (LDW_WARN_LL / _RL set alongside), not a hands-off warning.
    "HANDS_WARN_3_BITS": 0b101 if steer_required else 0,
    "HANDS_ON_STEER_WARN": 0,
    "HANDS_ON_STEER_WARN_2": 0,

    # TODO: right lane works, left doesn't
    # TODO: need to do something about L/R
    "LDW_WARN_LL": 0,
    "LDW_WARN_RL": 0,
  })
  addr, dat, bus = packer.make_can_msg("CAM_LANEINFO", 0, values)
  if steer_required:
    dat = dat[:6] + bytes([dat[6] | 0x01]) + dat[7:]
  return addr, dat, bus


def create_button_cmd(packer, CP, counter, button):

  can = int(button == Buttons.CANCEL)
  res = int(button == Buttons.RESUME)
  inc = int(button == Buttons.SET_PLUS)
  dec = int(button == Buttons.SET_MINUS)

  if CP.flags & MazdaFlags.GEN1:
    values = {
      "CAN_OFF": can,
      "CAN_OFF_INV": (can + 1) % 2,

      "SET_P": inc,
      "SET_P_INV": (inc + 1) % 2,

      "RES": res,
      "RES_INV": (res + 1) % 2,

      "SET_M": dec,
      "SET_M_INV": (dec + 1) % 2,

      "DISTANCE_LESS": 0,
      "DISTANCE_LESS_INV": 1,

      "DISTANCE_MORE": 0,
      "DISTANCE_MORE_INV": 1,

      "MODE_X": 0,
      "MODE_X_INV": 1,

      "MODE_Y": 0,
      "MODE_Y_INV": 1,

      "BIT1": 1,
      "BIT2": 1,
      "BIT3": 1,
      "CTR": (counter + 1) % 16,
    }

    return packer.make_can_msg("CRZ_BTNS", 0, values)
