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

from opendbc.car import DT_CTRL
from opendbc.car.mazda.radar_interface import (RadarInterface, RADAR_LOST_FRAMES, RADAR_LOST_STEP, RADAR_TRACK_ADDRS,
                                               RADAR_USABLE_ADDRS, SENTINEL_DIST, SENTINEL_ANG, SENTINEL_RELV)
from opendbc.car.mazda.tests.conftest import car_params, car_params_sp, packer
from opendbc.car.mazda.values import CAR
from opendbc.sunnypilot.car.interfaces import _initialize_mazda

PACKER = packer()


@pytest.fixture
def ri() -> RadarInterface:
  # stock-long: under alpha long the stock radar is torn down and radarUnavailable is True
  CP = car_params(CAR.MAZDA_CX5_2022, alpha_long=False)
  return RadarInterface(CP, car_params_sp(CP, alpha_long=False))


def track_msg(addr, dist=SENTINEL_DIST, ang=SENTINEL_ANG, relv=SENTINEL_RELV):
  """Pack one track frame by signal name (physical values), defaulting to an empty slot."""
  return PACKER.make_can_msg(addr, 0, {"DIST_OBJ": dist, "ANG_OBJ": ang, "RELV_OBJ": relv})


def burst(ri, t_ns, overrides=None):
  """Feed one full 0x361-0x366 burst (empty slots unless overridden) and return the RadarData."""
  overrides = overrides or {}
  msgs = [overrides.get(addr, track_msg(addr)) for addr in RADAR_TRACK_ADDRS]
  rr = ri.update([(t_ns, msgs)])
  assert rr is not None, "a full burst ending in the trigger msg must produce a RadarData"
  return rr


def test_empty_slots_produce_no_points(ri):
  assert len(burst(ri, 0).points) == 0


def test_real_track_parses(ri):
  rr = burst(ri, 0, {0x361: track_msg(0x361, dist=40.0, ang=2.0, relv=-5.0)})
  assert len(rr.points) == 1
  pt = rr.points[0]
  assert pt.dRel == pytest.approx(math.cos(math.radians(2.0)) * 40.0, rel=1e-6)
  assert pt.vRel == -5.0


def test_track_at_exactly_minus_one_mps_is_kept(ri):
  """-1.0 m/s with a real distance is a live lead, not an empty slot. Dropping it
  would delete and re-create the track each time the closing speed crosses -1.0,
  resetting radard's Kalman state mid-follow."""
  rr = burst(ri, 0, {0x361: track_msg(0x361, dist=40.0, ang=0.0, relv=-1.0625)})
  track_id = rr.points[0].trackId

  # lead decelerates through exactly -1.0 m/s: the track must survive with its identity
  rr = burst(ri, int(0.1e9), {0x361: track_msg(0x361, dist=39.875, ang=0.0, relv=SENTINEL_RELV)})
  assert len(rr.points) == 1
  assert rr.points[0].vRel == SENTINEL_RELV
  assert rr.points[0].trackId == track_id

  rr = burst(ri, int(0.2e9), {0x361: track_msg(0x361, dist=39.75, ang=0.0, relv=-0.9375)})
  assert rr.points[0].trackId == track_id


def test_track_at_max_range_is_kept(ri):
  rr = burst(ri, 0, {0x362: track_msg(0x362, dist=SENTINEL_DIST, ang=0.0, relv=-10.0)})
  assert len(rr.points) == 1
  assert rr.points[0].dRel == SENTINEL_DIST


def test_track_at_sentinel_angle_is_kept(ri):
  rr = burst(ri, 0, {0x363: track_msg(0x363, dist=40.0, ang=SENTINEL_ANG, relv=-5.0)})
  assert len(rr.points) == 1
  assert rr.points[0].yRel == pytest.approx(-math.sin(math.radians(SENTINEL_ANG)) * 40.0, rel=1e-6)


def test_all_sentinel_slot_deletes_a_prior_track(ri):
  burst(ri, 0, {0x361: track_msg(0x361, dist=40.0, ang=0.0, relv=-5.0)})
  rr = burst(ri, int(0.1e9))  # slot empties: all three sentinels
  assert len(rr.points) == 0


def test_undecoded_relv_addrs_never_produce_points(ri):
  overrides = {addr: track_msg(addr, dist=40.0, ang=0.0, relv=-5.0)
               for addr in RADAR_TRACK_ADDRS if addr not in RADAR_USABLE_ADDRS}
  assert len(burst(ri, 0, overrides).points) == 0


# Behind the Smart CAN Filter a lost radar falls back to the camera: empty, temporarily-unavailable
# RadarData at the radar's own rate keeps radard's radarTracks alive, never a CAN error.
@pytest.fixture
def filter_ri() -> RadarInterface:
  CP = car_params(CAR.MAZDA_CX5_2022, alpha_long=True)
  CP_SP = car_params_sp(CP, alpha_long=True)
  _initialize_mazda(CP, CP_SP, {"MazdaCanFilter": "1"})
  assert not CP.radarUnavailable
  return RadarInterface(CP, CP_SP)


def silence(ri, t_ns, frames):
  """card's 100 Hz calls with no radar frames: (frame index, RadarData) for each one that publishes."""
  out = []
  for i in range(frames):
    rr = ri.update([(t_ns + int((i + 1) * DT_CTRL * 1e9), [])])
    if rr is not None:
      out.append((i + 1, rr))
  return out


LEAD = {0x361: track_msg(0x361, dist=40.0, ang=0.0, relv=-5.0)}


def test_filter_healthy_bursts_are_unchanged(filter_ri):
  rr = burst(filter_ri, 0, LEAD)
  assert len(rr.points) == 1
  assert not any(rr.errors.to_dict().values())


def test_filter_silence_falls_back_to_the_camera(filter_ri):
  burst(filter_ri, 0, LEAD)
  published = silence(filter_ri, 0, 100)
  # nothing inside a normal gap, then 10 Hz from 0.25 s: well inside radard's 0.5 s alive window
  assert [i for i, _ in published] == list(range(RADAR_LOST_FRAMES, 101, RADAR_LOST_STEP))
  for _, rr in published:
    assert len(rr.points) == 0
    assert rr.errors.radarUnavailableTemporary
    assert not rr.errors.canError


def test_filter_radar_return_restores_tracks(filter_ri):
  burst(filter_ri, 0, LEAD)
  silence(filter_ri, 0, 100)
  rr = burst(filter_ri, int(1.1e9), LEAD)
  assert len(rr.points) == 1
  assert not any(rr.errors.to_dict().values())
  # the silence counter restarts: one normal gap publishes nothing
  assert silence(filter_ri, int(1.1e9), RADAR_LOST_FRAMES - 1) == []


def test_filter_invalid_parser_is_not_a_can_error(filter_ri):
  burst(filter_ri, 0, LEAD)
  rr = None
  # bursts missing 0x361 long enough for the parser to time it out
  for k in range(1, 40):
    msgs = [track_msg(addr) for addr in RADAR_TRACK_ADDRS if addr != 0x361]
    rr = filter_ri.update([(int(k * 0.1e9), msgs)])
  assert rr is not None and not filter_ri.rcp.can_valid
  assert rr.errors.radarUnavailableTemporary
  assert not rr.errors.canError
  assert len(rr.points) == 0


def test_without_the_filter_silence_publishes_nothing(ri):
  burst(ri, 0, LEAD)
  assert silence(ri, 0, 100) == []


def partial_burst(ri, t_ns, omit, overrides=None):
  """A burst ending in the trigger msg that leaves out the `omit` addresses."""
  overrides = overrides or {}
  msgs = [overrides.get(addr, track_msg(addr)) for addr in RADAR_TRACK_ADDRS if addr not in omit]
  rr = ri.update([(t_ns, msgs)])
  assert rr is not None
  return rr


def test_filter_recovery_never_republishes_a_cached_lead(filter_ri):
  """Codex repro: a 10 m lead, the camera-only fallback, then a recovery burst without 0x361. The
  parser still holds the old 0x361 values; republishing them handed radard a phantom lead it took
  at low speed with no camera confidence."""
  near = {0x361: track_msg(0x361, dist=10.0, ang=0.0, relv=0.0)}
  rr = burst(filter_ri, 0, near)
  assert len(rr.points) == 1
  assert silence(filter_ri, 0, RADAR_LOST_FRAMES)[-1][1].errors.radarUnavailableTemporary

  rr = partial_burst(filter_ri, int(0.35e9), omit={0x361})
  assert len(rr.points) == 0, "the stale 10 m lead came back from the parser's cache"
  rr = partial_burst(filter_ri, int(0.45e9), omit={0x361})
  assert len(rr.points) == 0

  # a fresh lead returns as soon as its own frame arrives, as a new track
  rr = burst(filter_ri, int(0.55e9), {0x361: track_msg(0x361, dist=25.0, ang=0.0, relv=-2.0)})
  assert len(rr.points) == 1
  assert rr.points[0].dRel == pytest.approx(25.0)
  assert not any(rr.errors.to_dict().values())


def test_filter_a_frame_missing_mid_stream_drops_its_track(filter_ri):
  burst(filter_ri, 0, LEAD)
  rr = partial_burst(filter_ri, int(0.1e9), omit={0x361})
  assert len(rr.points) == 0


def test_filter_partial_burst_before_a_dropout_is_discarded(filter_ri):
  """Frames of a burst that never completed are not carried into the burst after the dropout."""
  burst(filter_ri, 0, {})
  filter_ri.update([(int(0.05e9), [track_msg(0x361, dist=10.0, ang=0.0, relv=0.0)])])  # no trigger
  silence(filter_ri, int(0.05e9), RADAR_LOST_FRAMES)
  rr = partial_burst(filter_ri, int(0.4e9), omit={0x361})
  assert len(rr.points) == 0


def test_without_the_filter_a_missing_frame_keeps_its_cached_track(ri):
  """Filter off is unchanged: the parser's last 0x361 values still publish."""
  burst(ri, 0, LEAD)
  rr = partial_burst(ri, int(0.1e9), omit={0x361})
  assert len(rr.points) == 1
