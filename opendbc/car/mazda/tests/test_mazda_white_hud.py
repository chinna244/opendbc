from types import SimpleNamespace

import pytest

from opendbc.car import Bus, DT_CTRL, gen_empty_fingerprint, structs
from opendbc.car.mazda import mazdacan
from opendbc.car.mazda.carcontroller import MADS_WHITE_HUD_OFF_CONFIRM_FRAMES, CarController
from opendbc.car.mazda.carstate import CAM_LANEINFO_STALE_FRAMES
from opendbc.car.mazda.interface import CarInterface, latch_cam_laneinfo_raw
from opendbc.car.mazda.values import CAR
from opendbc.sunnypilot.car.mazda.values import MazdaFlagsSP

OFF = mazdacan.MADS_HUD_OFF
WHITE = mazdacan.MADS_HUD_WHITE
UNKNOWN = bytes.fromhex("4201000a00001040")


@pytest.mark.parametrize(("fsc_dat", "current_dat", "enabled", "expected"), [
  (OFF, OFF, True, WHITE),
  (OFF, OFF, False, OFF),
  (UNKNOWN, OFF, True, OFF),
  (OFF, UNKNOWN, True, UNKNOWN),
  (None, OFF, True, OFF),
])
def test_exact_payload_gate(fsc_dat, current_dat, enabled, expected):
  assert mazdacan.apply_mads_white_hud(fsc_dat, current_dat, enabled) == expected


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


class TestWhiteHudController:
  @staticmethod
  def _controller(candidate=CAR.MAZDA_CX5_2022, off_flag=True, active_probe_flag=False):
    fingerprint = {0: {}, 1: {}, 2: {}}
    CP = CarInterface.get_params(candidate, fingerprint, [], alpha_long=False, is_release=False, docs=False)
    CP_SP = CarInterface.get_params_sp(CP, candidate, fingerprint, [], False, False, False)
    if off_flag:
      CP_SP.flags |= MazdaFlagsSP.EXPERIMENTAL_MADS_WHITE_HUD.value
    if active_probe_flag:
      CP_SP.flags |= MazdaFlagsSP.EXPERIMENTAL_MADS_WHITE_HUD_ACTIVE_PROBE.value
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
  def _carstate(raw=OFF, live=True, raw_armed=False, raw_active=False, filtered_available=False,
                filtered_enabled=False, **overrides):
    cs = SimpleNamespace(
      out=SimpleNamespace(vEgoRaw=12.0, steeringTorque=0, brakePressed=False,
                          cruiseState=SimpleNamespace(available=filtered_available, enabled=filtered_enabled)),
      cruise_available=filtered_available,
      cruise_enabled=filtered_enabled,
      mrcc_armed_raw=raw_armed or raw_active,
      mrcc_active_raw=raw_active,
      cam_lkas_live=True,
      cam_lkas={"ERR_BIT_1": 0, "ERR_BIT_2": 0, "LINE_NOT_VISIBLE": 0, "BIT_1": 1},
      cam_laneinfo={"LINE_VISIBLE": 0, "LINE_NOT_VISIBLE": 1, "LANE_LINES": 1,
                    "BIT1": 1, "BIT2": 0, "BIT3": 1, "NO_ERR_BIT": 0, "ERR_BIT": 0, "TJA": 0, "TJA_TRANSITION": 0, "S1": 1, "S1_HBEAM": 0},
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

  @pytest.mark.parametrize(("off_flag", "active", "raw", "live"), [
    (False, True, OFF, True),
    (True, False, OFF, True),
    (True, True, UNKNOWN, True),
    (True, True, OFF, False),
  ])
  def test_all_disabled_or_untrusted_cases_keep_current_off(self, off_flag, active, raw, live):
    CC, CC_SP = self._controls(active=active)
    _, sends = self._controller(off_flag=off_flag).update(CC, CC_SP, self._carstate(raw=raw, live=live), 0)
    assert self._hud(sends) == OFF

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
    assert self._hud(sends) != WHITE
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


class TestWhiteHudActiveProbe:
  @staticmethod
  def _active_carstate(**overrides):
    state = dict(
      raw_active=True,
      raw_armed=True,
      filtered_available=True,
      filtered_enabled=True,
    )
    state.update(overrides)
    return TestWhiteHudController._carstate(**state)

  def test_active_probe_sends_one_white_after_stable_active(self):
    controller = TestWhiteHudController._controller(off_flag=False, active_probe_flag=True)
    CC, CC_SP = TestWhiteHudController._controls(active=True)
    white_frames = []
    for frame in range(MADS_WHITE_HUD_OFF_CONFIRM_FRAMES + 5):
      _, sends = controller.update(CC, CC_SP, self._active_carstate(), round(frame * DT_CTRL * 1e9))
      if any(addr == 0x440 for addr, _dat, _bus in sends):
        hud = TestWhiteHudController._hud(sends)
        if hud == WHITE:
          white_frames.append(frame)

    assert len(white_frames) == 1
    assert white_frames[0] == MADS_WHITE_HUD_OFF_CONFIRM_FRAMES - 1
    assert controller.mads_white_hud_active_probe_sent

    for frame in range(MADS_WHITE_HUD_OFF_CONFIRM_FRAMES + 5, 200):
      _, sends = controller.update(CC, CC_SP, self._active_carstate(), round(frame * DT_CTRL * 1e9))
      if any(addr == 0x440 for addr, _dat, _bus in sends):
        assert TestWhiteHudController._hud(sends) != WHITE

  def test_active_probe_does_not_run_without_flag(self):
    controller = TestWhiteHudController._controller(off_flag=False, active_probe_flag=False)
    CC, CC_SP = TestWhiteHudController._controls(active=True)
    for frame in range(MADS_WHITE_HUD_OFF_CONFIRM_FRAMES + 5):
      _, sends = controller.update(CC, CC_SP, self._active_carstate(), round(frame * DT_CTRL * 1e9))
      if any(addr == 0x440 for addr, _dat, _bus in sends):
        assert TestWhiteHudController._hud(sends) != WHITE

  @pytest.mark.parametrize("state", [
    {"raw_active": False, "raw_armed": True, "filtered_available": True, "filtered_enabled": False},
    {"raw_active": True, "filtered_enabled": False, "filtered_available": True},
    {"raw_active": False, "raw_armed": False, "filtered_enabled": True, "filtered_available": True},
    {"cancel_button": 1},
    {"tja_button": 1},
  ])
  def test_armed_or_transition_never_emits_active_probe_white(self, state):
    controller = TestWhiteHudController._controller(off_flag=False, active_probe_flag=True)
    CC, CC_SP = TestWhiteHudController._controls(active=True)
    for frame in range(MADS_WHITE_HUD_OFF_CONFIRM_FRAMES + 10):
      _, sends = controller.update(CC, CC_SP, self._active_carstate(**state), round(frame * DT_CTRL * 1e9))
      if any(addr == 0x440 for addr, _dat, _bus in sends):
        assert TestWhiteHudController._hud(sends) != WHITE

  def test_icbm_set_activity_blocks_active_probe(self):
    controller = TestWhiteHudController._controller(off_flag=False, active_probe_flag=True)
    CC, CC_SP = TestWhiteHudController._controls(active=True)
    CC_SP.intelligentCruiseButtonManagement.sendButton = (
      structs.IntelligentCruiseButtonManagement.SendButtonState.increase
    )
    for frame in range(MADS_WHITE_HUD_OFF_CONFIRM_FRAMES + 10):
      _, sends = controller.update(CC, CC_SP, self._active_carstate(), round(frame * DT_CTRL * 1e9))
      if any(addr == 0x440 for addr, _dat, _bus in sends):
        assert TestWhiteHudController._hud(sends) != WHITE

  def test_leaving_active_resets_probe_for_next_episode(self):
    controller = TestWhiteHudController._controller(off_flag=False, active_probe_flag=True)
    CC, CC_SP = TestWhiteHudController._controls(active=True)
    for frame in range(MADS_WHITE_HUD_OFF_CONFIRM_FRAMES + 1):
      controller.update(CC, CC_SP, self._active_carstate(), round(frame * DT_CTRL * 1e9))
    assert controller.mads_white_hud_active_probe_sent

    for frame in range(MADS_WHITE_HUD_OFF_CONFIRM_FRAMES + 2, MADS_WHITE_HUD_OFF_CONFIRM_FRAMES + 20):
      controller.update(
        CC, CC_SP,
        TestWhiteHudController._carstate(filtered_available=True, filtered_enabled=False, raw_armed=True),
        round(frame * DT_CTRL * 1e9),
      )
    assert not controller.mads_white_hud_active_probe_sent

    white_count = 0
    start = MADS_WHITE_HUD_OFF_CONFIRM_FRAMES + 20
    for frame in range(start, start + MADS_WHITE_HUD_OFF_CONFIRM_FRAMES + 5):
      _, sends = controller.update(CC, CC_SP, self._active_carstate(), round(frame * DT_CTRL * 1e9))
      if any(addr == 0x440 for addr, _dat, _bus in sends) and TestWhiteHudController._hud(sends) == WHITE:
        white_count += 1
    assert white_count == 1
