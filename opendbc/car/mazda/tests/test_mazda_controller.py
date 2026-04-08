#!/usr/bin/env python3
"""Tests for Mazda CX-5 2022 steering parameters."""

import pytest

from opendbc.car.mazda.values import CAR, CarControllerParams


class TestCarControllerParams:

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

  def test_cx5_2022_flat_steer_max(self, cx5_2022_params):
    assert cx5_2022_params.STEER_MAX == 1200
    assert not hasattr(cx5_2022_params, 'STEER_MAX_LOOKUP')

  def test_cx5_2022_rate_limits(self, cx5_2022_params):
    assert cx5_2022_params.STEER_DELTA_UP == 12
    assert cx5_2022_params.STEER_DELTA_DOWN == 25

  def test_cx5_2022_driver_multiplier_stock(self, cx5_2022_params):
    """Should use class default (15), not overridden."""
    assert cx5_2022_params.STEER_DRIVER_MULTIPLIER == 15

  def test_pre_2022_values(self, pre_2022_params):
    assert pre_2022_params.STEER_MAX == 800
    assert pre_2022_params.STEER_DELTA_UP == 10
    assert pre_2022_params.STEER_DELTA_DOWN == 25
    assert not hasattr(pre_2022_params, 'STEER_MAX_LOOKUP')
