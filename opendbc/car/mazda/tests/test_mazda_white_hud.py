from types import SimpleNamespace

import pytest

from opendbc.car import Bus, DT_CTRL, gen_empty_fingerprint, structs
from opendbc.car.can_definitions import CanData
from opendbc.can import CANParser
from opendbc.car.mazda import mazdacan
from opendbc.car.mazda.carcontroller import MADS_WHITE_HUD_OFF_CONFIRM_FRAMES, CarController
from opendbc.car.mazda.carstate import CAM_LANEINFO_STALE_FRAMES
from opendbc.car.mazda.interface import CarInterface, latch_cam_laneinfo_raw
from opendbc.car.mazda.values import CAR
from opendbc.sunnypilot.car.mazda.values import MazdaFlagsSP

OFF = mazdacan.MADS_HUD_OFF
WHITE = mazdacan.MADS_HUD_WHITE
LANE_VISIBLE_4361 = bytes.fromhex("4361000000000040")
LANE_VISIBLE_4102 = bytes.fromhex("4102000000001040")
LANE_VISIBLE_4361_WHITE = bytes.fromhex("4361000020000040")
LANE_VISIBLE_4102_WHITE = bytes.fromhex("4102000020001040")
COUNTER_1060 = bytes.fromhex("4201000000001060")
COUNTER_1060_WHITE = bytes.fromhex("4201000020001060")
LANE_AHB_4122 = bytes.fromhex("4122000000001040")
LANE_AHB_4122_WHITE = bytes.fromhex("4122000020001040")
COUNTER_4361_0060 = bytes.fromhex("4361000000000060")
COUNTER_4361_0060_WHITE = bytes.fromhex("4361000020000060")
BIT2_4221_1040 = bytes.fromhex("4221000000001040")
BIT2_4221_1040_WHITE = bytes.fromhex("4221000020001040")
BIT2_4221_1060 = bytes.fromhex("4221000000001060")
BIT2_4221_1060_WHITE = bytes.fromhex("4221000020001060")
NEARBY_4221_10E0 = bytes.fromhex("42210000000010E0")  # extra unnamed byte-7 bit; not audited
UNKNOWN = bytes.fromhex("4201000000011040")  # ERR_BIT=1; normalized packed not allowlisted
TRANS_4102 = bytes.fromhex("4102000400001040")  # TJA_TRANSITION=1 (DBC); normalizes to 410200…1040
WHITE_TJA_XOR = mazdacan.MADS_HUD_WHITE_TJA_XOR
SAFE_BASES = sorted(mazdacan.MADS_HUD_SAFE_BASE_PAYLOADS)
_LANEINFO_SIGS = [
  "LINE_VISIBLE", "LINE_NOT_VISIBLE", "LANE_LINES", "BIT1", "BIT2", "BIT3",
  "NO_ERR_BIT", "ERR_BIT", "TJA", "TJA_TRANSITION", "S1", "S1_HBEAM",
]


def _cam_laneinfo_from_raw(raw: bytes) -> dict:
  cp = CANParser("mazda_2017", [("CAM_LANEINFO", 0)], 2)
  cp.update([(0, [CanData(0x440, raw, 2)])])
  return {s: int(cp.vl["CAM_LANEINFO"][s]) for s in _LANEINFO_SIGS}


@pytest.mark.parametrize("base", SAFE_BASES)
def test_allowlist_payload_only_flips_white_tja_bits(base):
  out = mazdacan.apply_mads_white_hud(base, base, True)
  assert bytes(a ^ b for a, b in zip(base, out, strict=True)) == WHITE_TJA_XOR
  assert mazdacan.is_mads_white_hud(out)


def test_allowlist_stays_fourteen_stable_bases():
  assert len(mazdacan.MADS_HUD_SAFE_BASE_PAYLOADS) == 14
  assert bytes.fromhex("4202000000001040") not in mazdacan.MADS_HUD_SAFE_BASE_PAYLOADS
  assert bytes.fromhex("4102000400001040") not in mazdacan.MADS_HUD_SAFE_BASE_PAYLOADS
  assert NEARBY_4221_10E0 not in mazdacan.MADS_HUD_SAFE_BASE_PAYLOADS


@pytest.mark.parametrize(("fsc_raw", "packed_dat", "enabled", "expected"), [
  (OFF, OFF, True, WHITE),
  (OFF, OFF, False, OFF),
  (UNKNOWN, UNKNOWN, True, UNKNOWN),
  (None, OFF, True, OFF),
  # FSC/packer disagreement on relay fields: never paint
  (bytes.fromhex("4221000000001040"), OFF, True, OFF),
  # already-WHITE / nonzero TJA form is not an allowlisted base
  (WHITE, WHITE, True, WHITE),
  # TJA_TRANSITION-only raw difference: paint on normalized packed base
  (TRANS_4102, LANE_VISIBLE_4102, True, LANE_VISIBLE_4102_WHITE),
])
def test_white_hud_payload_gate(fsc_raw, packed_dat, enabled, expected):
  assert mazdacan.apply_mads_white_hud(fsc_raw, packed_dat, enabled) == expected


def test_unknown_payload_passthrough_unchanged():
  assert mazdacan.apply_mads_white_hud(UNKNOWN, UNKNOWN, True) == UNKNOWN
  assert not mazdacan.is_mads_white_hud(UNKNOWN)
  assert not mazdacan.is_white_hud_normalized_base(UNKNOWN, UNKNOWN)


def test_transition_raw_normalizes_to_allowlisted_packed_base():
  assert mazdacan.is_white_hud_normalized_base(TRANS_4102, LANE_VISIBLE_4102)
  out = mazdacan.apply_mads_white_hud(TRANS_4102, LANE_VISIBLE_4102, True)
  assert out == LANE_VISIBLE_4102_WHITE
  assert bytes(a ^ b for a, b in zip(LANE_VISIBLE_4102, out, strict=True)) == WHITE_TJA_XOR


def test_normalize_mask_matches_mazda_2017_dbc():
  from opendbc.can.dbc import DBC
  be_bits = [j + i * 8 for i in range(64) for j in range(7, -1, -1)]
  sigs = DBC("mazda_2017").name_to_msg["CAM_LANEINFO"].sigs
  field_bits = 0
  for name in ("TJA", "TJA_TRANSITION"):
    sig = sigs[name]
    idx = be_bits.index(sig.start_bit)
    for bit in be_bits[idx:idx + sig.size]:
      byte, b = divmod(bit, 8)
      field_bits |= 1 << (8 * (7 - byte) + b)
  keep = ((1 << 64) - 1) ^ field_bits
  assert keep == mazdacan.CAM_LANEINFO_TJA_NORMALIZE_MASK
  assert (keep >> (8 * (7 - mazdacan._CAM_LANEINFO_TJA_BYTE))) & 0xFF == 0xFF ^ mazdacan._CAM_LANEINFO_TJA_BITS
  assert (keep >> (8 * (7 - mazdacan._CAM_LANEINFO_TRANS_BYTE))) & 0xFF == 0xFF ^ mazdacan._CAM_LANEINFO_TRANS_BITS


@pytest.mark.parametrize("base", SAFE_BASES)
@pytest.mark.parametrize("trans", (1, 2, 3))
def test_packer_tja_transition_normalizes_to_same_base(base, trans):
  from opendbc.can.packer import CANPacker
  packer = CANPacker("mazda_2017")
  fields = _cam_laneinfo_from_raw(base)
  packed = packer.make_can_msg("CAM_LANEINFO", 0, {**fields, "TJA": 0, "TJA_TRANSITION": 0})[1]
  raw = packer.make_can_msg("CAM_LANEINFO", 0, {**fields, "TJA": 0, "TJA_TRANSITION": trans})[1]
  assert raw != packed
  assert packed in mazdacan.MADS_HUD_SAFE_BASE_PAYLOADS
  assert mazdacan.cam_laneinfo_matches_normalized(raw, packed)
  assert mazdacan.is_white_hud_normalized_base(raw, packed)


def test_non_tja_bit_flip_fail_closes():
  for i in range(64):
    raw = bytearray(OFF)
    raw[i // 8] ^= 1 << (7 - (i % 8))
    flipped = bytes(raw)
    tja_or_trans = (
      (i // 8 == mazdacan._CAM_LANEINFO_TJA_BYTE and (1 << (7 - (i % 8))) & mazdacan._CAM_LANEINFO_TJA_BITS) or
      (i // 8 == mazdacan._CAM_LANEINFO_TRANS_BYTE and (1 << (7 - (i % 8))) & mazdacan._CAM_LANEINFO_TRANS_BITS)
    )
    if tja_or_trans:
      assert mazdacan.cam_laneinfo_matches_normalized(flipped, OFF)
    else:
      assert not mazdacan.cam_laneinfo_matches_normalized(flipped, OFF)
      # Leave the 14-base allowlist unless this flip lands on another audited payload.
      if flipped not in mazdacan.MADS_HUD_SAFE_BASE_PAYLOADS:
        assert not mazdacan.is_white_hud_normalized_base(flipped, OFF)


def test_white_hud_helpers_do_not_construct_canparser(monkeypatch):
  import inspect
  import opendbc.can.parser as parser_mod

  src = (
    inspect.getsource(mazdacan.cam_laneinfo_matches_normalized) +
    inspect.getsource(mazdacan.white_hud_allowlist_base) +
    inspect.getsource(mazdacan.is_white_hud_normalized_base) +
    inspect.getsource(mazdacan.apply_mads_white_hud)
  )
  assert "CANParser" not in src
  assert not hasattr(mazdacan, "_decode_cam_laneinfo")

  def _boom(*_a, **_k):
    raise AssertionError("CANParser constructed on WHITE HUD path")
  monkeypatch.setattr(parser_mod, "CANParser", _boom)
  assert mazdacan.is_white_hud_normalized_base(TRANS_4102, LANE_VISIBLE_4102)
  assert mazdacan.white_hud_allowlist_base(TRANS_4102) == LANE_VISIBLE_4102
  assert mazdacan.apply_mads_white_hud(OFF, OFF, True) == WHITE


def test_normalized_match_rss_does_not_grow_with_gc_disabled():
  import gc
  def rss_mb():
    with open("/proc/self/status") as f:
      for line in f:
        if line.startswith("VmRSS:"):
          return int(line.split()[1]) / 1024.0
    return 0.0

  gc.collect()
  gc.disable()
  r0 = rss_mb()
  for _ in range(3_000_000):
    mazdacan.is_white_hud_normalized_base(TRANS_4102, LANE_VISIBLE_4102)
  r1 = rss_mb()
  gc.enable()
  gc.collect()
  assert r1 - r0 < 2.0, f"RSS grew {r1 - r0:.1f} MB over 3e6 calls"


def test_nonzero_tja_or_transition_frames_are_not_allowlisted():
  # Existing TJA=2 WHITE form and a crafted TJA_TRANSITION!=0 frame must never paint.
  tja_trans = bytes.fromhex("4201000000001048")  # OFF with TJA_TRANSITION bit flipped (not audited)
  assert WHITE not in mazdacan.MADS_HUD_SAFE_BASE_PAYLOADS
  assert tja_trans not in mazdacan.MADS_HUD_SAFE_BASE_PAYLOADS
  assert mazdacan.apply_mads_white_hud(WHITE, WHITE, True) == WHITE
  assert mazdacan.apply_mads_white_hud(tja_trans, tja_trans, True) == tja_trans


def test_unpacked_lane_lines_variant_420200_remains_blocked():
  rare = bytes.fromhex("4202000000001040")
  assert rare not in mazdacan.MADS_HUD_SAFE_BASE_PAYLOADS
  assert not mazdacan.is_white_hud_normalized_base(rare, rare)
  assert mazdacan.apply_mads_white_hud(rare, rare, True) == rare


def test_4221_1060_is_trusted_white_base_preserving_byte7():
  assert BIT2_4221_1060 in mazdacan.MADS_HUD_SAFE_BASE_PAYLOADS
  assert mazdacan.white_hud_allowlist_base(BIT2_4221_1060) == BIT2_4221_1060
  assert mazdacan.is_white_hud_normalized_base(BIT2_4221_1060, BIT2_4221_1060)
  out = mazdacan.apply_mads_white_hud(BIT2_4221_1060, BIT2_4221_1060, True)
  assert out == BIT2_4221_1060_WHITE
  assert bytes(a ^ b for a, b in zip(BIT2_4221_1060, out, strict=True)) == WHITE_TJA_XOR
  assert out[7] == 0x60
  assert mazdacan.is_mads_white_hud(out)


def test_4221_1040_white_behavior_is_unchanged():
  assert BIT2_4221_1040 in mazdacan.MADS_HUD_SAFE_BASE_PAYLOADS
  out = mazdacan.apply_mads_white_hud(BIT2_4221_1040, BIT2_4221_1040, True)
  assert out == BIT2_4221_1040_WHITE
  assert bytes(a ^ b for a, b in zip(BIT2_4221_1040, out, strict=True)) == WHITE_TJA_XOR
  assert out[7] == 0x40
  assert mazdacan.is_mads_white_hud(out)


def test_nearby_4221_10e0_remains_fail_closed():
  assert NEARBY_4221_10E0 not in mazdacan.MADS_HUD_SAFE_BASE_PAYLOADS
  assert mazdacan.white_hud_allowlist_base(NEARBY_4221_10E0) is None
  assert not mazdacan.is_white_hud_normalized_base(NEARBY_4221_10E0, NEARBY_4221_10E0)
  assert mazdacan.apply_mads_white_hud(NEARBY_4221_10E0, NEARBY_4221_10E0, True) == NEARBY_4221_10E0
  assert not mazdacan.is_mads_white_hud(NEARBY_4221_10E0)


def test_raw_latch_accepts_only_camera_bus_eight_byte_frames():
  packets = [(0, [
    (0x440, OFF, 0),
    (0x440, OFF[:7], 2),
    (0x440, WHITE, 2),
  ])]
  assert latch_cam_laneinfo_raw(packets, None) == (WHITE, True)
  assert latch_cam_laneinfo_raw([(0, [(0x440, UNKNOWN, 0)])], WHITE) == (WHITE, False)


def test_raw_latch_accepts_single_batch_tuple_like_test_models():
  # test_models calls CI.update((t, frames)), not [(t, frames)]
  assert latch_cam_laneinfo_raw((0, [(0x440, WHITE, 2)]), None) == (WHITE, True)


def test_raw_liveness_expires_and_recovers_on_the_receiving_frame():
  fingerprint = gen_empty_fingerprint()
  CP = CarInterface.get_params(CAR.MAZDA_CX5_2022, fingerprint, [], alpha_long=False, is_release=False, docs=False)
  CP_SP = CarInterface.get_params_sp(CP, CAR.MAZDA_CX5_2022, fingerprint, [], False, False, False)
  CI = CarInterface(CP, CP_SP)

  assert not CI.CS.cam_laneinfo_live
  CI.update([(0, [(0x440, OFF, 2)])])
  assert CI.CS.cam_laneinfo_live
  # single-batch tuple shape used by opendbc car/tests/test_models.py
  CI.update((round(DT_CTRL * 1e9), [(0x440, OFF, 2)]))
  assert CI.CS.cam_laneinfo_live
  for frame in range(CAM_LANEINFO_STALE_FRAMES + 1):
    CI.update([(round((frame + 2) * DT_CTRL * 1e9), [])])
  assert not CI.CS.cam_laneinfo_live
  CI.update([(round((CAM_LANEINFO_STALE_FRAMES + 3) * DT_CTRL * 1e9), [(0x440, OFF, 2)])])
  assert CI.CS.cam_laneinfo_live


def test_lane_visible_4361_only_tja_xor_changes():
  out = mazdacan.apply_mads_white_hud(LANE_VISIBLE_4361, LANE_VISIBLE_4361, True)
  assert out == LANE_VISIBLE_4361_WHITE
  assert bytes(a ^ b for a, b in zip(LANE_VISIBLE_4361, out, strict=True)) == WHITE_TJA_XOR


def test_lane_visible_4102_only_tja_xor_changes():
  out = mazdacan.apply_mads_white_hud(LANE_VISIBLE_4102, LANE_VISIBLE_4102, True)
  assert out == LANE_VISIBLE_4102_WHITE
  assert bytes(a ^ b for a, b in zip(LANE_VISIBLE_4102, out, strict=True)) == WHITE_TJA_XOR


def test_counter_1060_only_tja_xor_changes():
  out = mazdacan.apply_mads_white_hud(COUNTER_1060, COUNTER_1060, True)
  assert out == COUNTER_1060_WHITE
  assert bytes(a ^ b for a, b in zip(COUNTER_1060, out, strict=True)) == WHITE_TJA_XOR
  assert COUNTER_1060_WHITE[7] == 0x60


def test_counter_1060_preserves_byte7_nibble_vs_1040_white():
  base_white = mazdacan.apply_mads_white_hud(OFF, OFF, True)
  counter_white = mazdacan.apply_mads_white_hud(COUNTER_1060, COUNTER_1060, True)
  assert base_white[7] == 0x40
  assert counter_white[7] == 0x60
  assert bytes(a ^ b for a, b in zip(OFF, base_white, strict=True)) == WHITE_TJA_XOR
  assert bytes(a ^ b for a, b in zip(COUNTER_1060, counter_white, strict=True)) == WHITE_TJA_XOR


def test_lane_ahb_4122_only_tja_xor_changes():
  out = mazdacan.apply_mads_white_hud(LANE_AHB_4122, LANE_AHB_4122, True)
  assert out == LANE_AHB_4122_WHITE
  assert bytes(a ^ b for a, b in zip(LANE_AHB_4122, out, strict=True)) == WHITE_TJA_XOR
  assert out[1] == 0x22


def test_counter_4361_0060_only_tja_xor_changes():
  out = mazdacan.apply_mads_white_hud(COUNTER_4361_0060, COUNTER_4361_0060, True)
  assert out == COUNTER_4361_0060_WHITE
  assert bytes(a ^ b for a, b in zip(COUNTER_4361_0060, out, strict=True)) == WHITE_TJA_XOR
  assert COUNTER_4361_0060_WHITE[7] == 0x60


class TestWhiteHudController:
  @staticmethod
  def _controller(candidate=CAR.MAZDA_CX5_2022, off_flag=True):
    fingerprint = {0: {}, 1: {}, 2: {}}
    CP = CarInterface.get_params(candidate, fingerprint, [], alpha_long=False, is_release=False, docs=False)
    CP_SP = CarInterface.get_params_sp(CP, candidate, fingerprint, [], False, False, False)
    if off_flag:
      CP_SP.flags |= MazdaFlagsSP.EXPERIMENTAL_MADS_WHITE_HUD.value
    return CarController({Bus.pt: "mazda_2017"}, CP, CP_SP)

  @staticmethod
  def _controls(active=True, visual_alert=structs.CarControl.HUDControl.VisualAlert.none,
                cancel=False, resume=False):
    CC = structs.CarControl()
    CC.hudControl.visualAlert = visual_alert
    CC.cruiseControl.cancel = cancel
    CC.cruiseControl.resume = resume
    CC = CC.as_reader()
    CC_SP = structs.CarControlSP()
    CC_SP.mads.active = active
    return CC, CC_SP

  @staticmethod
  def _carstate(raw=OFF, live=True, raw_armed=False, filtered_available=False,
                filtered_enabled=False, **overrides):
    cs = SimpleNamespace(
      out=SimpleNamespace(vEgoRaw=12.0, steeringTorque=0, brakePressed=False,
                          cruiseState=SimpleNamespace(available=filtered_available, enabled=filtered_enabled)),
      cruise_available=filtered_available,
      cruise_enabled=filtered_enabled,
      mrcc_armed_raw=raw_armed,
      steer_undelivered=False,
      radar_was_silenced=True,
      cam_lkas_live=True,
      cam_lkas={"ERR_BIT_1": 0, "ERR_BIT_2": 0, "LINE_NOT_VISIBLE": 0, "BIT_1": 1},
      cam_laneinfo=_cam_laneinfo_from_raw(raw),
      cam_laneinfo_raw=raw,
      cam_laneinfo_live=live,
      crz_btns_counter=0,
      cancel_button=0,
      resume_button=0,
      tja_button=0,
      main_button=0,
      mode_x=0,
      mode_y=0,
      distance_button=0,
      accel_button=0,
      decel_button=0,
      mrcc_button=0,
      lkas_allowed_speed=True,
      lkas_blocked=False,
      lkas_effective=0,
    )
    for name, value in overrides.items():
      setattr(cs, name, value)
    return cs

  @staticmethod
  def _hud(sends):
    return next(dat for addr, dat, bus in sends if addr == 0x440 and bus == 0)

  @staticmethod
  def _packed(raw: bytes) -> bytes:
    from opendbc.can.packer import CANPacker
    return mazdacan.create_alert_command(CANPacker("mazda_2017"), _cam_laneinfo_from_raw(raw), False, False)[1]

  def _prime_white(self, controller, CC, CC_SP):
    sends = []
    for frame in range(MADS_WHITE_HUD_OFF_CONFIRM_FRAMES + 1):
      _, sends = controller.update(CC, CC_SP, self._carstate(), round(frame * DT_CTRL * 1e9))
    assert self._hud(sends) == WHITE

  def test_active_fresh_exact_off_becomes_white_only_after_stable_mrcc_off(self):
    CC, CC_SP = self._controls(active=True)
    controller = self._controller()
    _, sends = controller.update(CC, CC_SP, self._carstate(), 0)
    assert self._hud(sends) == OFF
    for frame in range(1, MADS_WHITE_HUD_OFF_CONFIRM_FRAMES + 1):
      _, sends = controller.update(CC, CC_SP, self._carstate(), round(frame * DT_CTRL * 1e9))
    assert self._hud(sends) == WHITE

  @pytest.mark.parametrize("base", SAFE_BASES)
  def test_each_allowlist_base_becomes_tja_only_white_after_stable_off(self, base):
    CC, CC_SP = self._controls(active=True)
    controller = self._controller()
    packed = self._packed(base)
    for frame in range(MADS_WHITE_HUD_OFF_CONFIRM_FRAMES + 1):
      _, sends = controller.update(CC, CC_SP, self._carstate(raw=base), round(frame * DT_CTRL * 1e9))
    out = self._hud(sends)
    assert packed in mazdacan.MADS_HUD_SAFE_BASE_PAYLOADS
    assert bytes(a ^ b for a, b in zip(base, out, strict=True)) == WHITE_TJA_XOR
    assert mazdacan.is_mads_white_hud(out)

  @pytest.mark.parametrize("base", [LANE_VISIBLE_4361, LANE_VISIBLE_4102, COUNTER_1060, LANE_AHB_4122, COUNTER_4361_0060, BIT2_4221_1040, BIT2_4221_1060])
  def test_lane_visible_bases_become_tja_only_white_after_stable_off(self, base):
    CC, CC_SP = self._controls(active=True)
    controller = self._controller()
    packed = self._packed(base)
    for frame in range(MADS_WHITE_HUD_OFF_CONFIRM_FRAMES + 1):
      _, sends = controller.update(CC, CC_SP, self._carstate(raw=base), round(frame * DT_CTRL * 1e9))
    out = self._hud(sends)
    assert packed in mazdacan.MADS_HUD_SAFE_BASE_PAYLOADS
    assert bytes(a ^ b for a, b in zip(base, out, strict=True)) == WHITE_TJA_XOR
    assert mazdacan.is_mads_white_hud(out)

  def test_armed_never_emits_white(self):
    CC, CC_SP = self._controls(active=True)
    controller = self._controller()
    for frame in range(MADS_WHITE_HUD_OFF_CONFIRM_FRAMES + 10):
      _, sends = controller.update(
        CC, CC_SP,
        self._carstate(filtered_available=True, filtered_enabled=False, raw_armed=True),
        round(frame * DT_CTRL * 1e9),
      )
      if any(addr == 0x440 for addr, _dat, _bus in sends):
        assert not mazdacan.is_mads_white_hud(self._hud(sends))

  def test_active_never_emits_white(self):
    CC, CC_SP = self._controls(active=True)
    controller = self._controller()
    for frame in range(MADS_WHITE_HUD_OFF_CONFIRM_FRAMES + 10):
      _, sends = controller.update(
        CC, CC_SP,
        self._carstate(filtered_available=True, filtered_enabled=True, raw_armed=True),
        round(frame * DT_CTRL * 1e9),
      )
      if any(addr == 0x440 for addr, _dat, _bus in sends):
        assert not mazdacan.is_mads_white_hud(self._hud(sends))

  @pytest.mark.parametrize(("off_flag", "active", "raw", "live"), [
    (False, True, OFF, True),
    (True, False, OFF, True),
    (True, True, UNKNOWN, True),
    (True, True, OFF, False),
  ])
  def test_all_disabled_or_untrusted_cases_keep_current_off(self, off_flag, active, raw, live):
    CC, CC_SP = self._controls(active=active)
    _, sends = self._controller(off_flag=off_flag).update(CC, CC_SP, self._carstate(raw=raw, live=live), 0)
    hud = self._hud(sends)
    assert not mazdacan.is_mads_white_hud(hud)
    # Unknown FSC: passthrough of packed laneinfo, never forced WHITE.
    if raw == UNKNOWN:
      assert hud != WHITE
    else:
      assert hud == OFF

  def test_existing_steer_required_warning_is_unchanged(self):
    CC, CC_SP = self._controls(active=True, visual_alert=structs.CarControl.HUDControl.VisualAlert.steerRequired)
    _, sends = self._controller().update(CC, CC_SP, self._carstate(), 0)
    hud = self._hud(sends)
    assert hud == bytes.fromhex("4201000000001e49")
    assert hud != WHITE

  def test_flag_cannot_enable_icon_on_non_tja_mazda(self):
    CC, CC_SP = self._controls(active=True)
    _, sends = self._controller(CAR.MAZDA_CX9_2021).update(CC, CC_SP, self._carstate(), 0)
    assert self._hud(sends) == OFF

  def test_hud_cadence_remains_two_hz(self):
    controller = self._controller()
    CC, CC_SP = self._controls(active=True)
    hud_frames = []
    for frame in range(101):
      _, sends = controller.update(CC, CC_SP, self._carstate(), round(frame * DT_CTRL * 1e9))
      if any(addr == 0x440 for addr, _dat, _bus in sends):
        hud_frames.append(frame)
    assert hud_frames == [0, 50, 100]

  @pytest.mark.parametrize("state", [
    {"mrcc_button": 1},
    {"tja_button": 1},
    {"cancel_button": 1},
    {"resume_button": 1},
    {"accel_button": 1},
    {"decel_button": 1},
    {"distance_button": 1},
    {"raw_armed": True},
    {"filtered_available": True},
    {"filtered_enabled": True},
    {"live": False},
    {"raw": UNKNOWN},
  ])
  def test_interaction_or_untrusted_state_withdraws_white_immediately(self, state):
    controller = self._controller()
    CC, CC_SP = self._controls(active=True)
    self._prime_white(controller, CC, CC_SP)

    _, sends = controller.update(CC, CC_SP, self._carstate(**state),
                                 round((MADS_WHITE_HUD_OFF_CONFIRM_FRAMES + 1) * DT_CTRL * 1e9))
    assert not mazdacan.is_mads_white_hud(self._hud(sends))
    assert controller.frame == MADS_WHITE_HUD_OFF_CONFIRM_FRAMES + 2

  def test_cleanup_pending_withdraws_white_immediately(self):
    controller = self._controller()
    CC, CC_SP = self._controls(active=True)
    self._prime_white(controller, CC, CC_SP)
    controller.tja_mrcc_unarm_pending = True

    _, sends = controller.update(CC, CC_SP, self._carstate(),
                                 round((MADS_WHITE_HUD_OFF_CONFIRM_FRAMES + 1) * DT_CTRL * 1e9))
    assert self._hud(sends) == OFF

  def test_mads_pause_withdraws_white_immediately(self):
    controller = self._controller()
    CC, CC_SP = self._controls(active=True)
    self._prime_white(controller, CC, CC_SP)
    _, paused = self._controls(active=False)

    _, sends = controller.update(CC, paused, self._carstate(),
                                 round((MADS_WHITE_HUD_OFF_CONFIRM_FRAMES + 1) * DT_CTRL * 1e9))
    assert self._hud(sends) == OFF

  @pytest.mark.parametrize(("cancel", "resume"), [(True, False), (False, True)])
  def test_synthetic_cruise_activity_withdraws_white_immediately(self, cancel, resume):
    controller = self._controller()
    CC, CC_SP = self._controls(active=True)
    self._prime_white(controller, CC, CC_SP)
    active_CC, _ = self._controls(active=True, cancel=cancel, resume=resume)

    _, sends = controller.update(active_CC, CC_SP, self._carstate(),
                                 round((MADS_WHITE_HUD_OFF_CONFIRM_FRAMES + 1) * DT_CTRL * 1e9))
    assert self._hud(sends) == OFF

  def test_hud_warning_withdraws_white_immediately(self):
    controller = self._controller()
    CC, CC_SP = self._controls(active=True)
    self._prime_white(controller, CC, CC_SP)
    warning_CC, _ = self._controls(
      active=True,
      visual_alert=structs.CarControl.HUDControl.VisualAlert.steerRequired,
    )

    _, sends = controller.update(warning_CC, CC_SP, self._carstate(),
                                 round((MADS_WHITE_HUD_OFF_CONFIRM_FRAMES + 1) * DT_CTRL * 1e9))
    assert self._hud(sends) == bytes.fromhex("4201000000001e49")

  def test_short_mrcc_tap_stays_oem_until_requalified(self):
    controller = self._controller()
    CC, CC_SP = self._controls(active=True)
    self._prime_white(controller, CC, CC_SP)
    frame = MADS_WHITE_HUD_OFF_CONFIRM_FRAMES + 1

    _, sends = controller.update(CC, CC_SP, self._carstate(mrcc_button=1), round(frame * DT_CTRL * 1e9))
    assert self._hud(sends) == OFF
    frame += 1
    _, sends = controller.update(CC, CC_SP, self._carstate(), round(frame * DT_CTRL * 1e9))
    assert not any(addr == 0x440 for addr, _dat, _bus in sends)

    while controller.frame <= 100:
      frame = controller.frame
      _, sends = controller.update(CC, CC_SP, self._carstate(), round(frame * DT_CTRL * 1e9))
    assert self._hud(sends) == OFF

    while controller.frame <= 150:
      frame = controller.frame
      _, sends = controller.update(CC, CC_SP, self._carstate(), round(frame * DT_CTRL * 1e9))
    assert self._hud(sends) == WHITE

  def test_fast_double_mrcc_tap_restarts_off_confirmation(self):
    controller = self._controller()
    CC, CC_SP = self._controls(active=True)
    self._prime_white(controller, CC, CC_SP)

    for pressed in (1, 0, 1, 0):
      frame = controller.frame
      _, sends = controller.update(CC, CC_SP, self._carstate(mrcc_button=pressed), round(frame * DT_CTRL * 1e9))
      if pressed == 1 and any(addr == 0x440 for addr, _dat, _bus in sends):
        assert self._hud(sends) == OFF

    assert controller.mads_white_hud_off_frames == 1
    assert not controller.mads_white_hud_on_bus

  def test_transition_raw_paints_white_on_normalized_packed_base(self):
    CC, CC_SP = self._controls(active=True)
    controller = self._controller()
    for frame in range(MADS_WHITE_HUD_OFF_CONFIRM_FRAMES + 1):
      _, sends = controller.update(
        CC, CC_SP, self._carstate(raw=TRANS_4102), round(frame * DT_CTRL * 1e9),
      )
    assert self._hud(sends) == LANE_VISIBLE_4102_WHITE

  def test_transition_flicker_does_not_reset_off_confirmation(self):
    CC, CC_SP = self._controls(active=True)
    controller = self._controller()
    for frame in range(MADS_WHITE_HUD_OFF_CONFIRM_FRAMES + 1):
      controller.update(CC, CC_SP, self._carstate(raw=LANE_VISIBLE_4102), round(frame * DT_CTRL * 1e9))
    assert controller.mads_white_hud_off_frames == MADS_WHITE_HUD_OFF_CONFIRM_FRAMES

    # Alternate TRANS raw with the same normalized packed base; timer must not reset.
    for i in range(20):
      raw = TRANS_4102 if i % 2 else LANE_VISIBLE_4102
      controller.update(CC, CC_SP, self._carstate(raw=raw), round((MADS_WHITE_HUD_OFF_CONFIRM_FRAMES + 2 + i) * DT_CTRL * 1e9))
    assert controller.mads_white_hud_off_frames == MADS_WHITE_HUD_OFF_CONFIRM_FRAMES
