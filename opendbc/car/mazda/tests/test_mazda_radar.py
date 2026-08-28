"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Radar track parsing: empty-slot detection requires ALL THREE sentinel fields.
Each sentinel is also a reachable real value on its own grid (relv -1.0 m/s is
an ordinary closing speed), so any-single-match dropping deletes a live track
and resets radard's Kalman state whenever a lead crosses that value. Measured
over 29k track frames (route 0b): sentinels occur either all-three (empty slot)
or not at all, except real tracks at exactly -1.0 m/s.
"""
import math

import pytest

from opendbc.can import CANPacker
from opendbc.car import gen_empty_fingerprint
from opendbc.car.mazda.interface import CarInterface
from opendbc.car.mazda.radar_interface import (RadarInterface, RADAR_TRACK_ADDRS, RADAR_USABLE_ADDRS,
                                               SENTINEL_DIST, SENTINEL_ANG, SENTINEL_RELV)
from opendbc.car.mazda.values import CAR

PACKER = CANPacker("mazda_2017")


def _radar_interface():
  # stock-long: under alpha long the stock radar is torn down and radarUnavailable is True
  fingerprint = gen_empty_fingerprint()
  CP = CarInterface.get_params(CAR.MAZDA_CX5_2022, fingerprint, [], alpha_long=False,
                               is_release=False, docs=False)
  CP_SP = CarInterface.get_params_sp(CP, CAR.MAZDA_CX5_2022, fingerprint, [],
                                     alpha_long=False, is_release_sp=False, docs=False)
  return RadarInterface(CP, CP_SP)


def track_msg(addr, dist=SENTINEL_DIST, ang=SENTINEL_ANG, relv=SENTINEL_RELV):
  """Pack one track frame by signal name (physical values), defaulting to an empty slot."""
  return PACKER.make_can_msg(f"RADAR_TRACK_{addr:x}", 0, {"DIST_OBJ": dist, "ANG_OBJ": ang, "RELV_OBJ": relv})


def burst(ri, t_ns, overrides=None):
  """Feed one full 0x361-0x366 burst (empty slots unless overridden) and return the RadarData."""
  overrides = overrides or {}
  msgs = [overrides.get(addr, track_msg(addr)) for addr in RADAR_TRACK_ADDRS]
  rr = ri.update([(t_ns, msgs)])
  assert rr is not None, "a full burst ending in the trigger msg must produce a RadarData"
  return rr


class TestRadarSentinels:
  def test_empty_slots_produce_no_points(self):
    ri = _radar_interface()
    rr = burst(ri, 0)
    assert len(rr.points) == 0

  def test_real_track_parses(self):
    ri = _radar_interface()
    rr = burst(ri, 0, {0x361: track_msg(0x361, dist=40.0, ang=2.0, relv=-5.0)})
    assert len(rr.points) == 1
    pt = rr.points[0]
    assert pt.dRel == pytest.approx(math.cos(math.radians(2.0)) * 40.0, rel=1e-6)
    assert pt.vRel == -5.0

  def test_track_at_exactly_minus_one_mps_is_kept(self):
    """-1.0 m/s with a real distance is a live lead, not an empty slot. Dropping it
    would delete and re-create the track each time the closing speed crosses -1.0,
    resetting radard's Kalman state mid-follow."""
    ri = _radar_interface()
    rr = burst(ri, 0, {0x361: track_msg(0x361, dist=40.0, ang=0.0, relv=-1.0625)})
    track_id = rr.points[0].trackId

    # lead decelerates through exactly -1.0 m/s: the track must survive with its identity
    rr = burst(ri, int(0.1e9), {0x361: track_msg(0x361, dist=39.875, ang=0.0, relv=SENTINEL_RELV)})
    assert len(rr.points) == 1
    assert rr.points[0].vRel == SENTINEL_RELV
    assert rr.points[0].trackId == track_id

    rr = burst(ri, int(0.2e9), {0x361: track_msg(0x361, dist=39.75, ang=0.0, relv=-0.9375)})
    assert rr.points[0].trackId == track_id

  def test_track_at_max_range_is_kept(self):
    ri = _radar_interface()
    rr = burst(ri, 0, {0x362: track_msg(0x362, dist=SENTINEL_DIST, ang=0.0, relv=-10.0)})
    assert len(rr.points) == 1
    assert rr.points[0].dRel == SENTINEL_DIST

  def test_track_at_sentinel_angle_is_kept(self):
    ri = _radar_interface()
    rr = burst(ri, 0, {0x363: track_msg(0x363, dist=40.0, ang=SENTINEL_ANG, relv=-5.0)})
    assert len(rr.points) == 1
    assert rr.points[0].yRel == pytest.approx(-math.sin(math.radians(SENTINEL_ANG)) * 40.0, rel=1e-6)

  def test_all_sentinel_slot_deletes_a_prior_track(self):
    ri = _radar_interface()
    burst(ri, 0, {0x361: track_msg(0x361, dist=40.0, ang=0.0, relv=-5.0)})
    rr = burst(ri, int(0.1e9))  # slot empties: all three sentinels
    assert len(rr.points) == 0

  def test_undecoded_relv_addrs_never_produce_points(self):
    ri = _radar_interface()
    overrides = {addr: track_msg(addr, dist=40.0, ang=0.0, relv=-5.0)
                 for addr in RADAR_TRACK_ADDRS if addr not in RADAR_USABLE_ADDRS}
    rr = burst(ri, 0, overrides)
    assert len(rr.points) == 0
