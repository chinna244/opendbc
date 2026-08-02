import pytest

from opendbc.car import DT_CTRL, gen_empty_fingerprint
from opendbc.car.mazda.interface import CarInterface
from opendbc.car.mazda.values import CAR, CarControllerParams

CAM_LANEINFO = 0x440

# Real CAM_LANEINFO prefixes, captured on two CX-5 2022s running the same FSC firmware
# (GSH7-67XK2-U). Only byte 1 differs: bit 5 is BIT2, bit 6 is NO_ERR_BIT.
BOOTING = bytes([0x42, 0b01000001, 0, 0, 0, 0, 0, 0])       # NO_ERR_BIT set: still booting
SETTLED = bytes([0x42, 0b00000001, 0, 0, 0, 0, 0, 0])       # markers clear: settled
BIT2_LATCHED = bytes([0x41, 0b00100001, 0, 0, 0, 0, 0, 0])  # BIT2 stuck high for a whole cycle
FAULTED = bytes([0x42, 0b00000001, 0, 0, 0, 0x01, 0, 0])    # ERR_BIT (bit 40) set


def _interface(alpha_long=True):
  fingerprint = gen_empty_fingerprint()
  CP = CarInterface.get_params(CAR.MAZDA_CX5_2022, fingerprint, [], alpha_long=alpha_long,
                               is_release=False, docs=False)
  CP_SP = CarInterface.get_params_sp(CP, CAR.MAZDA_CX5_2022, fingerprint, [],
                                     alpha_long=alpha_long, is_release_sp=False, docs=False)
  return CarInterface(CP, CP_SP)


def _feed(CI, payload, seconds):
  frames = int(seconds / DT_CTRL)
  for i in range(frames):
    CI.update([(int(i * DT_CTRL * 1e9), [(CAM_LANEINFO, payload, 2)])])
  return CI.CS.fsc_settled


@pytest.mark.parametrize("alpha_long", [False, True])
def test_carstate_runs_with_real_parsers(alpha_long):
  # vl_all, unlike vl, has no lazy message registration: every message read through it
  # must be listed in get_can_parsers. The op-long FSC settle gate crashed card on its
  # first update when CAM_LANEINFO was missing from the cam parser (KeyError, 2026-07-29).
  CI = _interface(alpha_long)
  assert CI.CP.openpilotLongitudinalControl == alpha_long
  for _ in range(10):
    CI.update([])


class TestFscSettleGate:
  """The gate that defers the radar teardown past the FSC's cold-boot radar-presence check.

  It must hold while the camera is booting or faulted, and must not be vetoed indefinitely
  by a bit that carries no boot information.
  """

  def test_never_settles_while_boot_marker_is_set(self):
    settle = CarControllerParams.FSC_SETTLE_T
    assert not _feed(_interface(), BOOTING, settle * 2)

  def test_never_settles_while_err_bit_is_set(self):
    # a latched i-ACTIVSENSE fault shows the boot markers clear, so ERR_BIT must veto on its own
    settle = CarControllerParams.FSC_SETTLE_T
    assert not _feed(_interface(), FAULTED, settle * 2)

  def test_settles_once_the_boot_marker_clears(self):
    CI = _interface()
    assert not _feed(CI, BOOTING, 3.0)
    assert not _feed(CI, SETTLED, CarControllerParams.FSC_SETTLE_T - 1.0)
    assert _feed(CI, SETTLED, 1.5)

  def test_a_latched_bit2_does_not_block_the_teardown_forever(self):
    # One CX-5 2022 cold-booted with BIT2 high and NO_ERR_BIT clear for an entire ignition
    # cycle (36.5 s, route 7c735af5fce56485|00000011). BIT2 was in the gate, so the radar was
    # never silenced and the two-master guard held accFaulted for the whole drive.
    assert _feed(_interface(), BIT2_LATCHED, CarControllerParams.FSC_SETTLE_T * 1.5)

  def test_gate_starts_closed_before_any_camera_frame(self):
    # the parser reads all-zero before the first frame, which would otherwise look settled
    CI = _interface()
    for i in range(int(CarControllerParams.FSC_SETTLE_T * 2 / DT_CTRL)):
      CI.update([(int(i * DT_CTRL * 1e9), [])])
    assert not CI.CS.fsc_settled
