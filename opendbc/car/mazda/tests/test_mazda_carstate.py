import pytest

from opendbc.car import gen_empty_fingerprint
from opendbc.car.mazda.interface import CarInterface
from opendbc.car.mazda.values import CAR


@pytest.mark.parametrize("alpha_long", [False, True])
def test_carstate_runs_with_real_parsers(alpha_long):
  # vl_all, unlike vl, has no lazy message registration: every message read through it
  # must be listed in get_can_parsers. The op-long FSC settle gate crashed card on its
  # first update when CAM_LANEINFO was missing from the cam parser (KeyError, 2026-07-29).
  fingerprint = gen_empty_fingerprint()
  CP = CarInterface.get_params(CAR.MAZDA_CX5_2022, fingerprint, [], alpha_long=alpha_long, is_release=False, docs=False)
  CP_SP = CarInterface.get_params_sp(CP, CAR.MAZDA_CX5_2022, fingerprint, [], alpha_long=alpha_long, is_release_sp=False, docs=False)
  assert CP.openpilotLongitudinalControl == alpha_long

  CI = CarInterface(CP, CP_SP)
  for _ in range(10):
    CI.update([])
