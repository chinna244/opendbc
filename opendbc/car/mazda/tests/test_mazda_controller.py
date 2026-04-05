#!/usr/bin/env python3
"""Tests for Mazda CX-5 2022 speed-dependent STEER_MAX."""

import numpy as np
import pytest

from opendbc.car.mazda.values import CAR, CarControllerParams


class TestCarControllerParams:
  """Test CarControllerParams for CX-5 2022 vs pre-2022."""

  @pytest.fixture
  def cx5_2022_params(self):
    class FakeCP:
      carFingerprint = CAR.MAZDA_CX5_2022
    return CarControllerParams(FakeCP())

  @pytest.fixture
  def pre_2022_params(self):
    class FakeCP:
      carFingerprint = CAR.MAZDA_CX5
    return CarControllerParams(FakeCP())

  # ── CX-5 2022: cliff STEER_MAX at 32 mph ──

  def test_cx5_2022_has_steer_max_lookup(self, cx5_2022_params):
    assert hasattr(cx5_2022_params, 'STEER_MAX_LOOKUP')

  def test_cx5_2022_low_speed_steer_max(self, cx5_2022_params):
    """Below 32 mph (14.2 m/s), full authority at 1200."""
    p = cx5_2022_params
    for v_ego in [0.0, 5.0, 10.0, 14.2]:
      steer_max = round(float(np.interp(v_ego, p.STEER_MAX_LOOKUP[0], p.STEER_MAX_LOOKUP[1])))
      assert steer_max == 1200, f"steer_max should be 1200 at {v_ego} m/s, got {steer_max}"

  def test_cx5_2022_high_speed_steer_max(self, cx5_2022_params):
    """Above 33 mph (14.5 m/s), reduced to 620 matching EPS ceiling."""
    p = cx5_2022_params
    for v_ego in [14.5, 20.0, 30.0, 40.0]:
      steer_max = round(float(np.interp(v_ego, p.STEER_MAX_LOOKUP[0], p.STEER_MAX_LOOKUP[1])))
      assert steer_max == 620, f"steer_max should be 620 at {v_ego} m/s, got {steer_max}"

  def test_cx5_2022_cliff_transition(self, cx5_2022_params):
    """Short ramp from 1200 to 620 between 14.2 and 14.5 m/s prevents step discontinuity."""
    p = cx5_2022_params
    mid = 14.35
    steer_max = float(np.interp(mid, p.STEER_MAX_LOOKUP[0], p.STEER_MAX_LOOKUP[1]))
    assert 620 < steer_max < 1200, f"Should interpolate at {mid} m/s, got {steer_max}"

  def test_cx5_2022_constant_rate_limits(self, cx5_2022_params):
    """Rate limits are constant — EPS hardware rate is 12/frame at all speeds."""
    p = cx5_2022_params
    assert p.STEER_DELTA_UP == 12
    assert p.STEER_DELTA_DOWN == 12
    assert not hasattr(p, 'STEER_DELTA_UP_LOOKUP')
    assert not hasattr(p, 'STEER_DELTA_DOWN_LOOKUP')

  def test_cx5_2022_within_panda_safety(self, cx5_2022_params):
    """All lookup values stay within panda safety limits."""
    p = cx5_2022_params
    for v_ego in np.linspace(0, 40, 100):
      steer_max = float(np.interp(v_ego, p.STEER_MAX_LOOKUP[0], p.STEER_MAX_LOOKUP[1]))
      assert steer_max <= 1200, f"steer_max exceeds panda safety at {v_ego:.1f} m/s"
      assert p.STEER_DELTA_UP <= 12, "delta_up exceeds panda safety"
      assert p.STEER_DELTA_DOWN <= 25, "delta_down exceeds panda safety"

  # ── Pre-2022: no lookups, unchanged behavior ──

  def test_pre_2022_no_lookups(self, pre_2022_params):
    assert not hasattr(pre_2022_params, 'STEER_MAX_LOOKUP')

  def test_pre_2022_static_values(self, pre_2022_params):
    assert pre_2022_params.STEER_MAX == 800
    assert pre_2022_params.STEER_DELTA_UP == 10
    assert pre_2022_params.STEER_DELTA_DOWN == 25
