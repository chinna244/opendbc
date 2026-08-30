import pytest

from opendbc.car import structs
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.mazda.interface import CarInterface
from opendbc.car.mazda.values import CAR, LKAS_LIMITS, STEER_TO_ZERO_EPS_FW

Ecu = structs.CarParams.Ecu

# The steer-to-zero EPS a swap donates, and a stock pre-2022 CX-5 EPS for contrast
SWAPPED_EPS_FW = sorted(STEER_TO_ZERO_EPS_FW)[0]
STOCK_CX5_EPS_FW = b'K319-3210X-A-00' + b'\x00' * 9


def _eps_fw(version: bytes) -> list[structs.CarParams.CarFw]:
  fw = structs.CarParams.CarFw()
  fw.ecu = Ecu.eps
  fw.address = 0x730
  fw.subAddress = 0
  fw.fwVersion = version
  return [fw]


def _params(candidate, car_fw=None, alpha_long=False):
  return CarInterface.get_params(candidate, {0: {}, 1: {}, 2: {}}, car_fw or [],
                                 alpha_long, is_release=False, docs=False)


class TestMazdaEpsSwap:
  """A 2022+ CX-5 EPS swapped into an older Mazda brings the EPS-derived behaviour with it.

  Pre-2022 Mazdas are dashcam only because their EPS locks steering out after ~5 s hands-off
  and below 45 kph. That lockout lives in the EPS, so the swap lifts it. Everything keyed on
  the radar, camera or vehicle dynamics must stay keyed on the model.
  """

  def test_stock_older_mazda_is_dashcam_only(self):
    CP = _params(CAR.MAZDA_CX5, _eps_fw(STOCK_CX5_EPS_FW))
    assert CP.dashcamOnly
    assert CP.minSteerSpeed == pytest.approx(LKAS_LIMITS.DISABLE_SPEED * CV.KPH_TO_MS)
    assert CP.steerActuatorDelay == pytest.approx(0.1)

  def test_swapped_eps_lifts_dashcam_and_the_speed_floor(self):
    CP = _params(CAR.MAZDA_CX5, _eps_fw(SWAPPED_EPS_FW))
    assert not CP.dashcamOnly
    assert CP.minSteerSpeed == 0
    assert CP.steerActuatorDelay == pytest.approx(0.14)

  def test_swapped_eps_does_not_unlock_longitudinal(self):
    # the radar and camera are not part of an EPS swap, and this car keeps its own pre-2022 pair
    CP = _params(CAR.MAZDA_CX5, _eps_fw(SWAPPED_EPS_FW), alpha_long=True)
    assert not CP.alphaLongitudinalAvailable
    assert not CP.openpilotLongitudinalControl

  def test_swapped_eps_keeps_the_real_vehicle_specs(self):
    # the whole point of fixing detection is that the user no longer forces MAZDA_CX5_2022 and
    # inherits its mass, steer ratio and tire stiffness
    swapped = _params(CAR.MAZDA_CX5, _eps_fw(SWAPPED_EPS_FW))
    cx5_2022 = _params(CAR.MAZDA_CX5_2022)
    assert swapped.mass != cx5_2022.mass
    assert swapped.steerRatio != cx5_2022.steerRatio
    assert swapped.tireStiffnessFactor != cx5_2022.tireStiffnessFactor

  def test_supported_platforms_are_unchanged(self):
    cx5_2022 = _params(CAR.MAZDA_CX5_2022)
    assert not cx5_2022.dashcamOnly
    assert cx5_2022.minSteerSpeed == 0
    assert cx5_2022.steerActuatorDelay == pytest.approx(0.14)
    assert cx5_2022.alphaLongitudinalAvailable

    # the CX-9 2021 is supported without the CX-5 EPS, so it keeps the 45 kph floor
    cx9_2021 = _params(CAR.MAZDA_CX9_2021)
    assert not cx9_2021.dashcamOnly
    assert cx9_2021.minSteerSpeed == pytest.approx(LKAS_LIMITS.DISABLE_SPEED * CV.KPH_TO_MS)
    assert cx9_2021.steerActuatorDelay == pytest.approx(0.1)

  def test_docs_are_generated_without_firmware(self):
    # car_fw is empty when building CARS.md, so the docs must keep advertising dashcam mode
    for candidate in (CAR.MAZDA_CX5, CAR.MAZDA_CX9, CAR.MAZDA_3, CAR.MAZDA_6):
      CP = CarInterface.get_params(candidate, {0: {}, 1: {}, 2: {}}, [], False,
                                   is_release=False, docs=True)
      assert CP.dashcamOnly, candidate
