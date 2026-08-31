"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import unittest

from opendbc.car.structs import CarParams
from opendbc.sunnypilot.car.interfaces import get_speed_dep_config_for_car, get_steer_max_schedule, get_steer_rail_schedule


def _cx5_cp():
  cp = CarParams()
  cp.carFingerprint = 'MAZDA_CX5_2022'
  cp.brand = 'mazda'
  cp.minSteerSpeed = 0.0
  return cp


class TestSteerMaxSchedule(unittest.TestCase):
  def test_mazda_steer_to_zero_returns_lookup(self):
    schedule = get_steer_max_schedule(_cx5_cp())
    self.assertEqual(schedule, ([0.0, 14.2, 14.5], [1200.0, 1200.0, 800.0]))

  def test_flat_steer_max_brand_returns_none(self):
    cp = CarParams()
    cp.carFingerprint = 'TOYOTA_RAV4_TSS2'
    cp.brand = 'toyota'
    self.assertIsNone(get_steer_max_schedule(cp))

  def test_unknown_brand_returns_none(self):
    cp = CarParams()
    cp.brand = 'notabrand'
    self.assertIsNone(get_steer_max_schedule(cp))

  def test_schedule_attached_to_active_entry(self):
    cfg = get_speed_dep_config_for_car(_cx5_cp())
    self.assertEqual(cfg['steer_max_schedule'], ([0.0, 14.2, 14.5], [1200.0, 1200.0, 800.0]))
    # the schedule's step must sit inside a bin span, or per-count interp cannot place it
    self.assertLess(cfg['speed_bp'][2], 14.2)
    self.assertLess(14.5, cfg['speed_bp'][3])

  def test_inactive_entry_stays_empty(self):
    cp = CarParams()
    cp.carFingerprint = 'MAZDA_CX9_2021'
    cp.brand = 'mazda'
    cp.minSteerSpeed = 20.0  # stock EPS, requires_steer_to_zero suppresses the entry
    self.assertEqual(get_speed_dep_config_for_car(cp), {})

  def test_config_copy_not_cached_dict(self):
    a = get_speed_dep_config_for_car(_cx5_cp())
    a['steer_max_schedule'] = 'mutated'
    self.assertNotEqual(get_speed_dep_config_for_car(_cx5_cp())['steer_max_schedule'], 'mutated')


class TestSteerRailSchedule(unittest.TestCase):
  def test_mazda_steer_to_zero_rail(self):
    bp, rail = get_steer_rail_schedule(_cx5_cp())
    self.assertTrue(all(0.0 < r <= 1.0 for r in rail))
    # the rail bottoms out just below the cliff (648/1200) and recovers above it (620/800)
    self.assertEqual(rail[bp.index(14.2)], min(rail))
    self.assertAlmostEqual(rail[bp.index(14.2)], 0.54, delta=0.01)
    self.assertAlmostEqual(rail[bp.index(14.5)], 620.0 / 800.0, delta=1e-6)

  def test_no_ceiling_returns_none(self):
    cp = CarParams()
    cp.brand = 'mazda'
    cp.minSteerSpeed = 20.0  # stock EPS params have no ceiling lookup
    self.assertIsNone(get_steer_rail_schedule(cp))
    cp2 = CarParams()
    cp2.brand = 'toyota'
    self.assertIsNone(get_steer_rail_schedule(cp2))
