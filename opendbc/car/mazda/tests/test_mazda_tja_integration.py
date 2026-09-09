"""Integration regressions: upstream camera TJA press + WHITE HUD + MRCC cleanup stay separate."""
from opendbc.car import Bus, DT_CTRL, structs
from opendbc.car.mazda import mazdacan
from opendbc.car.mazda.carcontroller import CarController
from opendbc.car.mazda.interface import CarInterface
from opendbc.car.mazda.tests.conftest import CRZ_BTNS, car_controller, frames, mazda_car_state, step
from opendbc.car.mazda.tests.test_mazda_mrcc_cleanup import TestTjaMrccSideEffect
from opendbc.car.mazda.tests.test_mazda_tja_press import INTERVAL, presses
from opendbc.car.mazda.tests.test_mazda_white_hud import TestWhiteHudController
from opendbc.car.mazda.values import CAR
from opendbc.sunnypilot.car.mazda.values import MazdaFlagsSP

OFF = bytes.fromhex("4201000000001040")


def _tja_cc():
  cc = car_controller()
  cc.CP_SP.flags |= MazdaFlagsSP.TJA_BUTTON
  return cc


class TestCameraTjaDoesNotTriggerMrccCleanup:
  """B: synthetic camera-bus TJA TX must not own MRCC cleanup."""

  def test_synthetic_camera_tja_only_no_mrcc_off(self):
    cc = _tja_cc()
    cs = mazda_car_state(cc.CP, cc.CP_SP)
    for _ in range(3 * INTERVAL):
      _, sends = step(cc, cs, lat_active=True, stock_tja=2, cruise_available=False, mrcc_armed_raw=False)
      assert presses(sends) in (0, 1)
      assert not frames(sends, CRZ_BTNS, bus=0), "camera TJA must not produce bus-0 CRZ_BTNS"
      assert not cc.tja_mrcc_unarm_pending
      assert cc.tja_mrcc_tx_frames == 0


class TestPhysicalTjaWithCameraPressAndMrccCleanup:
  """A: physical TJA can run both cleanups without same-bus collision."""

  def test_physical_tja_camera_and_mrcc_use_separate_buses(self):
    side = TestTjaMrccSideEffect()
    cc = side._cc()
    CC, CC_SP = side._controls()
    CC = CC.as_builder()
    CC.latActive = True
    CC = CC.as_reader()

    side._step(cc, CC, CC_SP, tja=0, armed=False, raw_armed=False, stock_tja=0)
    sends = side._step(cc, CC, CC_SP, tja=1, armed=False, raw_armed=False, stock_tja=2)
    assert len(frames(sends, CRZ_BTNS, bus=2)) == 1
    assert not frames(sends, CRZ_BTNS, bus=0)

    sends = side._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True, stock_tja=2)
    assert not frames(sends, CRZ_BTNS, bus=0)

    assert not side._button_payloads(side._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, stock_tja=2))
    payloads = side._mrcc_off_payloads(
      side._step_to_first_tx_deadline(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, stock_tja=2))
    assert len(payloads) == 1
    assert payloads[0].hex() in {
      "0081fed000000000",
      "0081fed400000000",
      "0081fed800000000",
    }

    for _ in range(5):
      sends = side._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, stock_tja=2)
      assert len(frames(sends, CRZ_BTNS, bus=0)) <= 1


class TestWhiteHudDoesNotArmStockTja:
  """C: WHITE HUD TX must not make stock_tja look armed or spur camera retries."""

  def test_white_hud_tx_leaves_stock_tja_from_incoming_camera(self):
    CP = CarInterface.get_params(CAR.MAZDA_CX5_2022, {0: {}, 1: {}, 2: {}}, [], False, False, False)
    CP_SP = CarInterface.get_params_sp(CP, CAR.MAZDA_CX5_2022, {0: {}, 1: {}, 2: {}}, [], False, False, False)
    CP_SP.flags |= MazdaFlagsSP.TJA_BUTTON
    cc = CarController({Bus.pt: "mazda_2017"}, CP, CP_SP)
    cc.mads_white_hud_off_frames = int(0.5 / DT_CTRL)

    CS = TestWhiteHudController._carstate(raw=OFF, live=True, raw_armed=False,
                                      filtered_available=False, filtered_enabled=False, stock_tja=0)
    CC, CC_SP = TestWhiteHudController._controls(active=True)
    _, sends = cc.update(CC, CC_SP, CS, 0)

    assert CS.stock_tja == 0
    assert presses(sends) == 0
    hud = [dat for addr, dat, bus in sends if addr == 0x440 and bus == 0]
    assert hud
    assert mazdacan.is_mads_white_hud(hud[0])


class TestFirstEngageHoldWithCustomFeatures:
  """D: first-engagement hold still zeroes torque with HUD/MRCC present."""

  def test_first_engage_hold_zeroes_torque(self):
    cc = _tja_cc()
    cs = mazda_car_state(cc.CP, cc.CP_SP)
    actuators, sends = step(
      cc, cs,
      lat_active=True,
      torque=-1.0,
      v_ego=0.3,
      steer_first_engage_hold=True,
      stock_tja=0,
      cruise_available=False,
      mrcc_armed_raw=False,
    )
    assert actuators.torqueOutputCan == 0
    assert not frames(sends, CRZ_BTNS, bus=0)
    assert presses(sends) == 0
