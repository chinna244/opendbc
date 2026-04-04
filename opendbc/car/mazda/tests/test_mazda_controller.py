#!/usr/bin/env python3
"""Tests for Mazda CX-5 2022 speed-dependent STEER_MAX and rate limits."""

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

  # ── CX-5 2022: speed-dependent lookups exist ──

  def test_cx5_2022_has_lookups(self, cx5_2022_params):
    assert hasattr(cx5_2022_params, 'STEER_MAX_LOOKUP')
    assert hasattr(cx5_2022_params, 'STEER_DELTA_UP_LOOKUP')
    assert hasattr(cx5_2022_params, 'STEER_DELTA_DOWN_LOOKUP')

  def test_cx5_2022_low_speed_values(self, cx5_2022_params):
    """Below 48 kph (13.3 m/s), full authority."""
    p = cx5_2022_params
    for v_ego in [0.0, 5.0, 10.0, 13.3]:
      steer_max = round(float(np.interp(v_ego, p.STEER_MAX_LOOKUP[0], p.STEER_MAX_LOOKUP[1])))
      delta_up = round(float(np.interp(v_ego, p.STEER_DELTA_UP_LOOKUP[0], p.STEER_DELTA_UP_LOOKUP[1])))
      delta_down = round(float(np.interp(v_ego, p.STEER_DELTA_DOWN_LOOKUP[0], p.STEER_DELTA_DOWN_LOOKUP[1])))
      assert steer_max == 1500, f"steer_max should be 1500 at {v_ego} m/s, got {steer_max}"
      assert delta_up == 15, f"delta_up should be 15 at {v_ego} m/s, got {delta_up}"
      assert delta_down == 25, f"delta_down should be 25 at {v_ego} m/s, got {delta_down}"

  def test_cx5_2022_high_speed_values(self, cx5_2022_params):
    """Above 52 kph (14.4 m/s), reduced authority matching EPS limit."""
    p = cx5_2022_params
    for v_ego in [14.4, 20.0, 30.0, 40.0]:
      steer_max = round(float(np.interp(v_ego, p.STEER_MAX_LOOKUP[0], p.STEER_MAX_LOOKUP[1])))
      delta_up = round(float(np.interp(v_ego, p.STEER_DELTA_UP_LOOKUP[0], p.STEER_DELTA_UP_LOOKUP[1])))
      delta_down = round(float(np.interp(v_ego, p.STEER_DELTA_DOWN_LOOKUP[0], p.STEER_DELTA_DOWN_LOOKUP[1])))
      assert steer_max == 750, f"steer_max should be 750 at {v_ego} m/s, got {steer_max}"
      assert delta_up == 9, f"delta_up should be 9 at {v_ego} m/s, got {delta_up}"
      assert delta_down == 15, f"delta_down should be 15 at {v_ego} m/s, got {delta_down}"

  def test_cx5_2022_transition_is_smooth(self, cx5_2022_params):
    """Values interpolate smoothly between 13.3 and 14.4 m/s."""
    p = cx5_2022_params
    v_mid = 13.85  # midpoint
    steer_max = float(np.interp(v_mid, p.STEER_MAX_LOOKUP[0], p.STEER_MAX_LOOKUP[1]))
    assert 750 < steer_max < 1500, f"steer_max should interpolate at {v_mid} m/s, got {steer_max}"
    delta_up = float(np.interp(v_mid, p.STEER_DELTA_UP_LOOKUP[0], p.STEER_DELTA_UP_LOOKUP[1]))
    assert 9 < delta_up < 15, f"delta_up should interpolate at {v_mid} m/s, got {delta_up}"

  def test_cx5_2022_rate_scales_with_steer_max(self, cx5_2022_params):
    """Rate limits maintain roughly consistent %/frame across speeds."""
    p = cx5_2022_params
    # Low speed: 15/1500 = 1.0%
    low_pct = 15 / 1500
    # High speed: 9/750 = 1.2%
    high_pct = 9 / 750
    # Should be within 0.5% of each other
    assert abs(high_pct - low_pct) < 0.005, f"Rate % should be similar: low={low_pct:.3f}, high={high_pct:.3f}"

  def test_cx5_2022_within_panda_safety(self, cx5_2022_params):
    """All lookup values stay within panda safety limits (1500 max, 15/25 rates)."""
    p = cx5_2022_params
    for v_ego in np.linspace(0, 40, 100):
      steer_max = float(np.interp(v_ego, p.STEER_MAX_LOOKUP[0], p.STEER_MAX_LOOKUP[1]))
      delta_up = float(np.interp(v_ego, p.STEER_DELTA_UP_LOOKUP[0], p.STEER_DELTA_UP_LOOKUP[1]))
      delta_down = float(np.interp(v_ego, p.STEER_DELTA_DOWN_LOOKUP[0], p.STEER_DELTA_DOWN_LOOKUP[1]))
      assert steer_max <= 1500, f"steer_max exceeds panda safety at {v_ego:.1f} m/s"
      assert delta_up <= 15, f"delta_up exceeds panda safety at {v_ego:.1f} m/s"
      assert delta_down <= 25, f"delta_down exceeds panda safety at {v_ego:.1f} m/s"

  # ── Pre-2022: no lookups, unchanged behavior ──

  def test_pre_2022_no_lookups(self, pre_2022_params):
    assert not hasattr(pre_2022_params, 'STEER_MAX_LOOKUP')
    assert not hasattr(pre_2022_params, 'STEER_DELTA_UP_LOOKUP')
    assert not hasattr(pre_2022_params, 'STEER_DELTA_DOWN_LOOKUP')

  def test_pre_2022_static_values(self, pre_2022_params):
    assert pre_2022_params.STEER_MAX == 800
    assert pre_2022_params.STEER_DELTA_UP == 10
    assert pre_2022_params.STEER_DELTA_DOWN == 25
