#!/usr/bin/env python3
"""Tests for the Mazda CX-5 2022+ EPS steering parameters (gated on the EPS, not the model)
and the longitudinal message builders and stop-and-go state machine."""

import numpy as np
import pytest

from opendbc.can import CANPacker
from opendbc.car.mazda import mazdacan
from opendbc.car.mazda.carcontroller import (HOLD_CTRL_LATCH_FRAMES, HOLD_LATCH_FRAMES, HOLD_PASSIVE_FRAMES,
                                             RESUME_REACTIVATE_FRAMES, RESUME_RELEASE_FRAMES, RESUME_UNLATCH_FRAMES,
                                             StopAndGoStateMachine, StopGoState)
from opendbc.car.mazda.values import CAR, CarControllerParams


class TestCarControllerParams:

  @pytest.fixture
  def cx5_2022_params(self):
    class FakeCP:
      carFingerprint = CAR.MAZDA_CX5_2022
      minSteerSpeed = 0.0   # steer_to_zero -> CX-5 2022+ EPS present
    return CarControllerParams(FakeCP())

  @pytest.fixture
  def eps_swap_params(self):
    # A CX-5 2022+ EPS swapped into (or shared by) another Mazda: different model, same EPS.
    class FakeCP:
      carFingerprint = CAR.MAZDA_CX9_2021
      minSteerSpeed = 0.0
    return CarControllerParams(FakeCP())

  @pytest.fixture
  def pre_2022_params(self):
    class FakeCP:
      carFingerprint = CAR.MAZDA_CX5
      minSteerSpeed = 12.5   # no CX-5 EPS -> low-speed lockout, minSteerSpeed > 0
    return CarControllerParams(FakeCP())

  def test_cx5_2022_has_lookup(self, cx5_2022_params):
    assert hasattr(cx5_2022_params, 'STEER_MAX_LOOKUP')
    assert cx5_2022_params.STEER_MAX == 1200

  def test_cx5_2022_low_speed(self, cx5_2022_params):
    p = cx5_2022_params
    for v in [0.0, 5.0, 10.0, 14.2]:
      sm = round(float(np.interp(v, p.STEER_MAX_LOOKUP[0], p.STEER_MAX_LOOKUP[1])))
      assert sm == 1200

  def test_cx5_2022_high_speed(self, cx5_2022_params):
    p = cx5_2022_params
    for v in [14.5, 20.0, 30.0]:
      sm = round(float(np.interp(v, p.STEER_MAX_LOOKUP[0], p.STEER_MAX_LOOKUP[1])))
      assert sm == 800

  def test_cx5_2022_rate_limits(self, cx5_2022_params):
    assert cx5_2022_params.STEER_DELTA_UP == 12
    assert cx5_2022_params.STEER_DELTA_DOWN == 25

  def test_cx5_eps_driver_multiplier(self, cx5_2022_params):
    # 15 is the CX-5-EPS tune (upstream stock is 1)
    assert cx5_2022_params.STEER_DRIVER_MULTIPLIER == 15

  def test_eps_swap_gets_cx5_tune(self, eps_swap_params):
    # EPS present (minSteerSpeed == 0) on a non-CX-5 model still gets the higher-authority tune
    assert eps_swap_params.STEER_MAX == 1200
    assert eps_swap_params.STEER_DRIVER_MULTIPLIER == 15
    assert hasattr(eps_swap_params, 'STEER_MAX_LOOKUP')

  def test_no_eps_no_lookup(self, pre_2022_params):
    assert not hasattr(pre_2022_params, 'STEER_MAX_LOOKUP')
    assert pre_2022_params.STEER_MAX == 800
    assert pre_2022_params.STEER_DRIVER_MULTIPLIER == 1


def crz_info_reference_checksum(dat):
  # independent reimplementation of the CRZ_INFO checksum, validated against 1.94M stock
  # frames including all 10,350 stop-bit frames
  return (0xFF - ((sum(dat[:7]) - (dat[5] & 0x04)) & 0xFF)) & 0xFF


def decode_accel_cmd_raw(dat):
  return (((dat[2] & 0x3) << 11) | (dat[3] << 3) | (dat[4] >> 5)) - 4096


class TestMazdaLongitudinalMessages:
  """The synthetic CRZ_INFO/CRZ_CTRL/radar frames must reproduce stock captures byte for
  byte; the hex values below come from real radar traffic."""

  @pytest.fixture
  def packer(self):
    return CANPacker("mazda_2017")

  def test_crz_info_standby_matches_stock(self, packer):
    for counter in range(16):
      checksum = (0x5d - counter) & 0xff
      expected = f"01ffe3ffc000{counter:02x}{checksum:02x}"
      dat = mazdacan.create_acc_command(packer, 0, counter, 0.0, False, False, False, False)[1]
      assert dat.hex() == expected

  def test_crz_info_available_matches_stock(self, packer):
    for counter in range(16):
      checksum = (0x99 - counter) & 0xff
      expected = f"01ffe2000480{counter:02x}{checksum:02x}"
      dat = mazdacan.create_acc_command(packer, 0, counter, 0.0, False, True, False, False)[1]
      assert dat.hex() == expected

  @pytest.mark.parametrize(("accel", "stopping", "unlatching", "counter", "expected"), [
    (0.0, False, False, 0, "01ffe20006800097"),     # engaged, zero command
    (2.0, False, False, 3, "01ffe2fa0680039a"),     # ISO max accel, raw 2000
    (-3.5, False, False, 7, "01ffe04a868007c8"),    # ISO max brake, raw -3500
    (-1.024, True, False, 5, "01ffe18006841503"),   # standstill hold, raw -1024 + stop bits
    (-0.001, False, False, 9, "01ffe1ffe68009b0"),  # latched hold, raw -1
    (0.0, False, True, 11, "01ffe20006804b4c"),     # resume unlatch pulse
  ])
  def test_crz_info_engaged_golden_bytes(self, packer, accel, stopping, unlatching, counter, expected):
    dat = mazdacan.create_acc_command(packer, 0, counter, accel, True, False, stopping, unlatching)[1]
    assert dat.hex() == expected

  def test_crz_info_accel_encoding_and_checksum(self, packer):
    # the packed command must round-trip at the 0.001 factor and carry a valid masked-bit
    # checksum over the whole command window, stop bits set or not
    for raw in range(-3500, 2001, 137):
      for stopping in (False, True):
        dat = mazdacan.create_acc_command(packer, 0, raw % 16, raw / 1000.0, True, False, stopping, False)[1]
        assert decode_accel_cmd_raw(dat) == raw
        assert dat[7] == crz_info_reference_checksum(dat)
        assert bool(dat[5] & 0x04) == stopping
        assert bool(dat[6] & 0x10) == stopping

  @pytest.mark.parametrize(("long_active", "acc_available", "gap", "has_lead", "phase", "acc_active_2", "expected"), [
    (False, False, 0, False, 0, False, "0201010000000000"),  # standby
    (False, True, 2, False, 0, False, "02010b0000000000"),   # MRCC armed, SET allowed
    (True, True, 2, True, 1, True, "0a018b2000001000"),      # engaged, cruise
    (True, True, 2, True, 2, True, "0a018b4000001000"),      # engaged, following
    (True, True, 2, True, 3, True, "0a018b6000001000"),      # stop-and-go hold / resume
    (True, True, 2, True, 4, True, "0a018b8000001000"),      # hold latched
    (True, True, 2, True, 4, False, "0a018b8000000000"),     # passive hold, ACC_ACTIVE_2 drops
    (True, True, 1, True, 2, True, "0a01874000001000"),      # driver gap 1 mirrored to the dash
  ])
  def test_crz_ctrl_golden_bytes(self, packer, long_active, acc_available, gap, has_lead, phase, acc_active_2, expected):
    dat = mazdacan.create_crz_ctrl(packer, 0, long_active, acc_available, gap, has_lead, phase, acc_active_2)[1]
    assert dat.hex() == expected

  def test_radar_frames_match_stock(self):
    expected = [
      (0x499, "0008c00000000000"),
      (0x361, "fff7fefe1fc00080"),
      (0x362, "fff7fefe1fc78c80"),
      (0x363, "fff7fefe1fc00000"),
      (0x364, "fff7fefe1fc00000"),
      (0x365, "fff7fe7ffbff3fc0"),
      (0x366, "fff7fe7ffbff3fc0"),
    ]
    frames = mazdacan.create_radar_frames(0, 0, synthetic_lead=False)
    assert [(f.address, f.dat.hex()) for f in frames] == expected

  def test_radar_frames_counter_and_synthetic_lead(self):
    frames = mazdacan.create_radar_frames(2, 15, synthetic_lead=True)
    assert all(f.src == 2 for f in frames)
    # counter stamps the low nibble of the last byte on every track
    assert [f.dat[7] & 0x0f for f in frames[1:]] == [15] * 6
    tracks = {f.address: f.dat.hex() for f in frames}
    assert tracks[0x364] == "0a4000001dc0000f"


class TestStopAndGoStateMachine:

  @pytest.fixture
  def sm(self):
    return StopAndGoStateMachine()

  @staticmethod
  def run(sm, frames, **kwargs):
    defaults = dict(long_active=True, stopping=False, standstill=False,
                    resume_pressed=False, virtual_resume=False, gas_override=False)
    defaults.update(kwargs)
    for _ in range(frames):
      state = sm.update(**defaults)
    return state

  def test_full_stop_cycle_virtual_resume(self, sm):
    assert self.run(sm, 1) == StopGoState.CRUISING
    assert self.run(sm, 1, stopping=True) == StopGoState.STOPPING
    assert self.run(sm, 1, stopping=True, standstill=True) == StopGoState.HOLD
    assert sm.stop_bits

    # a virtual resume cannot release the strong hold phase
    assert self.run(sm, HOLD_LATCH_FRAMES - 2, stopping=True, standstill=True, virtual_resume=True) == StopGoState.HOLD

    assert self.run(sm, 2, stopping=True, standstill=True) == StopGoState.HOLD_LATCHED
    assert not sm.stop_bits
    assert self.run(sm, HOLD_PASSIVE_FRAMES, stopping=True, standstill=True) == StopGoState.HOLD_PASSIVE
    assert not sm.acc_active_2

    # resume out of the passive hold: latched-profile blip, then the unlatch pulse
    assert self.run(sm, 1, stopping=True, standstill=True, virtual_resume=True) == StopGoState.RESUMING
    assert sm.ctrl_phase(lead_visible=True) == 4
    assert not sm.resume_unlatching
    self.run(sm, RESUME_REACTIVATE_FRAMES, stopping=True, standstill=True, virtual_resume=True)
    assert sm.ctrl_phase(lead_visible=True) == 3
    assert sm.resume_unlatching
    self.run(sm, RESUME_UNLATCH_FRAMES, stopping=True, standstill=True, virtual_resume=True)
    assert not sm.resume_unlatching

    # car creeps off the hold, request clears, release window runs out
    assert self.run(sm, RESUME_RELEASE_FRAMES, stopping=False, standstill=False) == StopGoState.CRUISING

  def test_ctrl_latch_phase_progression(self, sm):
    self.run(sm, 1, stopping=True)
    self.run(sm, 1, stopping=True, standstill=True)
    assert sm.ctrl_phase(lead_visible=True) == 3
    self.run(sm, HOLD_CTRL_LATCH_FRAMES, stopping=True, standstill=True)
    assert sm.state == StopGoState.HOLD  # CRZ_CTRL latches before the hold command relaxes
    assert sm.ctrl_phase(lead_visible=True) == 4

  def test_physical_res_waits_for_ctrl_latch(self, sm):
    self.run(sm, 1, stopping=True)
    self.run(sm, 1, stopping=True, standstill=True)
    # earlier than any stock-observed release: RES is ignored
    assert self.run(sm, 10, stopping=True, standstill=True, resume_pressed=True) == StopGoState.HOLD
    self.run(sm, HOLD_CTRL_LATCH_FRAMES, stopping=True, standstill=True)
    assert self.run(sm, 1, stopping=True, standstill=True, resume_pressed=True) == StopGoState.RESUMING
    assert sm.ctrl_phase(lead_visible=False) == 3  # no passive-hold blip needed

  def test_gas_releases_hold_immediately(self, sm):
    self.run(sm, 1, stopping=True)
    self.run(sm, 1, stopping=True, standstill=True)
    assert self.run(sm, 1, stopping=True, standstill=True, gas_override=True) == StopGoState.RESUMING

  def test_rehold_when_car_does_not_move(self, sm):
    self.run(sm, 1, stopping=True)
    self.run(sm, HOLD_LATCH_FRAMES + 2, stopping=True, standstill=True)
    self.run(sm, 1, stopping=True, standstill=True, virtual_resume=True)
    # request disappears, car never moved: fall back into a fresh hold
    assert self.run(sm, RESUME_RELEASE_FRAMES, stopping=True, standstill=True) == StopGoState.HOLD
    assert sm.hold_frames == 0
    assert sm.stop_bits

  def test_long_disengage_resets(self, sm):
    self.run(sm, 1, stopping=True)
    self.run(sm, HOLD_LATCH_FRAMES + 2, stopping=True, standstill=True)
    assert self.run(sm, 1, long_active=False) == StopGoState.CRUISING
    assert sm.hold_frames == 0

  def test_stop_abort_returns_to_cruising(self, sm):
    self.run(sm, 1, stopping=True)
    # lead speeds up again before the car reaches standstill
    assert self.run(sm, 1, stopping=False) == StopGoState.CRUISING
