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
UNKNOWN = bytes.fromhex("4201000a00001040")
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


@pytest.mark.parametrize(("fsc_dat", "current_dat", "enabled", "expected"), [
  (OFF, OFF, True, WHITE),
  (OFF, OFF, False, OFF),
  (UNKNOWN, OFF, True, OFF),
  (OFF, UNKNOWN, True, UNKNOWN),
  (None, OFF, True, OFF),
  # FSC/packer disagreement: never paint
  (OFF, bytes.fromhex("4221000000001040"), True, bytes.fromhex("4221000000001040")),
  # already-WHITE / nonzero TJA form is not an allowlisted base
  (WHITE, WHITE, True, WHITE),
])
def test_white_hud_payload_gate(fsc_dat, current_dat, enabled, expected):
  assert mazdacan.apply_mads_white_hud(fsc_dat, current_dat, enabled) == expected


def test_unknown_payload_passthrough_unchanged():
  assert mazdacan.apply_mads_white_hud(UNKNOWN, UNKNOWN, True) == UNKNOWN
  assert not mazdacan.is_mads_white_hud(UNKNOWN)


def test_nonzero_tja_or_transition_frames_are_not_allowlisted():
  # Existing TJA=2 WHITE form and a crafted TJA_TRANSITION!=0 frame must never paint.
  tja_trans = bytes.fromhex("4201000000001048")  # OFF with TJA_TRANSITION bit flipped (not audited)
  assert WHITE not in mazdacan.MADS_HUD_SAFE_BASE_PAYLOADS
  assert tja_trans not in mazdacan.MADS_HUD_SAFE_BASE_PAYLOADS
  assert mazdacan.apply_mads_white_hud(WHITE, WHITE, True) == WHITE
  assert mazdacan.apply_mads_white_hud(tja_trans, tja_trans, True) == tja_trans


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
      cam_lkas_live=True,
      cam_lkas={"ERR_BIT_1": 0, "ERR_BIT_2": 0, "LINE_NOT_VISIBLE": 0, "BIT_1": 1},
      cam_laneinfo=_cam_laneinfo_from_raw(raw),
      cam_laneinfo_raw=raw,
      cam_laneinfo_live=live,
      crz_btns_counter=0,
      cancel_button=0,
      resume_button=0,
      tja_button=0,
      accel_button=0,
      decel_button=0,
      mrcc_button=0,
      lkas_allowed_speed=True,
    )
    for name, value in overrides.items():
      setattr(cs, name, value)
    return cs

  @staticmethod
  def _hud(sends):
    return next(dat for addr, dat, bus in sends if addr == 0x440 and bus == 0)

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
    for frame in range(MADS_WHITE_HUD_OFF_CONFIRM_FRAMES + 1):
      _, sends = controller.update(CC, CC_SP, self._carstate(raw=base), round(frame * DT_CTRL * 1e9))
    out = self._hud(sends)
    assert bytes(a ^ b for a, b in zip(base, out, strict=True)) == WHITE_TJA_XOR
    assert mazdacan.is_mads_white_hud(out)

  @pytest.mark.parametrize("base", [LANE_VISIBLE_4361, LANE_VISIBLE_4102])
  def test_lane_visible_bases_become_tja_only_white_after_stable_off(self, base):
    CC, CC_SP = self._controls(active=True)
    controller = self._controller()
    for frame in range(MADS_WHITE_HUD_OFF_CONFIRM_FRAMES + 1):
      _, sends = controller.update(CC, CC_SP, self._carstate(raw=base), round(frame * DT_CTRL * 1e9))
    out = self._hud(sends)
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
