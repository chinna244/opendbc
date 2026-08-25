from types import SimpleNamespace

import pytest

from opendbc.car import Bus, DT_CTRL, gen_empty_fingerprint, structs
from opendbc.car.mazda import mazdacan
from opendbc.car.mazda.carcontroller import CarController
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


def test_raw_liveness_expires_and_recovers_on_the_receiving_frame():
  fingerprint = gen_empty_fingerprint()
  CP = CarInterface.get_params(CAR.MAZDA_CX5_2022, fingerprint, [], alpha_long=False, is_release=False, docs=False)
  CP_SP = CarInterface.get_params_sp(CP, CAR.MAZDA_CX5_2022, fingerprint, [], False, False, False)
  CI = CarInterface(CP, CP_SP)

  assert not CI.CS.cam_laneinfo_live
  CI.update([(0, [(0x440, OFF, 2)])])
  assert CI.CS.cam_laneinfo_live
  for frame in range(CAM_LANEINFO_STALE_FRAMES + 1):
    CI.update([(round((frame + 1) * DT_CTRL * 1e9), [])])
  assert not CI.CS.cam_laneinfo_live
  CI.update([(round((CAM_LANEINFO_STALE_FRAMES + 2) * DT_CTRL * 1e9), [(0x440, OFF, 2)])])
  assert CI.CS.cam_laneinfo_live


class TestWhiteHudController:
  @staticmethod
  def _controller(candidate=CAR.MAZDA_CX5_2022, flag=True):
    fingerprint = {0: {}, 1: {}, 2: {}}
    CP = CarInterface.get_params(candidate, fingerprint, [], alpha_long=False, is_release=False, docs=False)
    CP_SP = CarInterface.get_params_sp(CP, candidate, fingerprint, [], False, False, False)
    if flag:
      CP_SP.flags |= MazdaFlagsSP.EXPERIMENTAL_MADS_WHITE_HUD.value
    return CarController({Bus.pt: "mazda_2017"}, CP, CP_SP)

  @staticmethod
  def _controls(active=True, visual_alert=structs.CarControl.HUDControl.VisualAlert.none):
    CC = structs.CarControl()
    CC.hudControl.visualAlert = visual_alert
    CC = CC.as_reader()
    CC_SP = structs.CarControlSP()
    CC_SP.mads.active = active
    return CC, CC_SP

  @staticmethod
  def _carstate(raw=OFF, live=True):
    return SimpleNamespace(
      out=SimpleNamespace(vEgoRaw=12.0, steeringTorque=0, brakePressed=False,
                          cruiseState=SimpleNamespace(available=False)),
      cruise_available=False,
      mrcc_armed_raw=False,
      cam_lkas_live=True,
      cam_lkas={"ERR_BIT_1": 0, "ERR_BIT_2": 0, "LINE_NOT_VISIBLE": 0, "BIT_1": 1},
      cam_laneinfo={"LINE_VISIBLE": 0, "LINE_NOT_VISIBLE": 1, "LANE_LINES": 1,
                    "BIT1": 1, "BIT2": 0, "BIT3": 1, "NO_ERR_BIT": 0, "S1": 1, "S1_HBEAM": 0},
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

  @staticmethod
  def _hud(sends):
    return next(dat for addr, dat, bus in sends if addr == 0x440 and bus == 0)

  def test_active_fresh_exact_off_becomes_white(self):
    CC, CC_SP = self._controls(active=True)
    _, sends = self._controller().update(CC, CC_SP, self._carstate(), 0)
    assert self._hud(sends) == WHITE

  @pytest.mark.parametrize(("flag", "active", "raw", "live"), [
    (False, True, OFF, True),
    (True, False, OFF, True),
    (True, True, UNKNOWN, True),
    (True, True, OFF, False),
  ])
  def test_all_disabled_or_untrusted_cases_keep_current_off(self, flag, active, raw, live):
    CC, CC_SP = self._controls(active=active)
    _, sends = self._controller(flag=flag).update(CC, CC_SP, self._carstate(raw=raw, live=live), 0)
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
