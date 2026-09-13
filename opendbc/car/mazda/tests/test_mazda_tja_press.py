"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The camera press leaves TJA=0 and the benign TJA=2 FSC/HUD state alone. For any other nonzero
state it presses the camera's button off on its own bus only while openpilot lateral is active.
One frame per press, at least one 0x440 period between presses, three per active episode, then
one stockLkas pulse. TJA=0 or TJA=2 resets the episode; a temporary lateral pause does not.
"""
from opendbc.car import DT_CTRL
from opendbc.car.mazda.tests.conftest import CRZ_BTNS, car_controller, frames, mazda_car_state, step
from opendbc.car.mazda.values import CarControllerParams

INTERVAL = int(CarControllerParams.TJA_PRESS_INTERVAL_T / DT_CTRL)
PRESS = bytes.fromhex("0009ff")


def rig(alpha_long=False):
  cc = car_controller(alpha_long=alpha_long)
  cs = mazda_car_state(cc.CP, cc.CP_SP)
  return cc, cs


def presses(sends):
  # the one frame openpilot may put on the camera-side CRZ_BTNS
  out = frames(sends, CRZ_BTNS, bus=2)
  for dat in out:
    assert dat[:3] == PRESS and dat[4:] == bytes(4) and dat[3] & 0xc3 == 0xc0
  assert not frames(sends, CRZ_BTNS, bus=0), "the TJA button must never go to the car"
  return len(out)


class TestCameraPress:

  def test_one_press_on_the_first_steering_frame_with_the_camera_active(self):
    cc, cs = rig()
    assert presses(step(cc, cs, lat_active=True, stock_tja=4, crz_btns_counter=7)[1]) == 1

  def test_counter_is_the_wheels_plus_one(self):
    cc, cs = rig()
    _, sends = step(cc, cs, lat_active=True, stock_tja=4, crz_btns_counter=7)
    assert frames(sends, CRZ_BTNS, bus=2)[0][3] == 0xc0 | (8 << 2)

  def test_no_press_with_the_camera_off(self):
    cc, cs = rig()
    cc.tja_press_count = 2
    cc.tja_press_frame = 10
    cc.tja_episode_alerted = True
    for lat_active in (True, False):
      assert presses(step(cc, cs, lat_active=lat_active, stock_tja=0)[1]) == 0
      assert (cc.tja_press_count, cc.tja_press_frame, cc.tja_episode_alerted) == (0, None, False)

  def test_benign_tja_two_never_presses_warns_and_resets(self):
    cc, cs = rig()
    for lat_active in (True, False):
      cc.tja_press_count = 2
      cc.tja_press_frame = 10
      cc.tja_episode_alerted = True
      _, sends = step(cc, cs, lat_active=lat_active, stock_tja=2)
      assert presses(sends) == 0
      assert not cs.stock_cts_stuck
      assert (cc.tja_press_count, cc.tja_press_frame, cc.tja_episode_alerted) == (0, None, False)

  def test_active_camera_with_lateral_off_does_nothing_without_reset(self):
    cc, cs = rig()
    cc.tja_press_count = 2
    cc.tja_press_frame = 10
    cc.tja_episode_alerted = True
    for _ in range(3 * INTERVAL):
      assert presses(step(cc, cs, lat_active=False, stock_tja=4)[1]) == 0
      assert not cs.stock_cts_stuck
      assert (cc.tja_press_count, cc.tja_press_frame, cc.tja_episode_alerted) == (2, 10, True)

  def test_cadence_cap_and_the_one_shot_warning(self):
    cc, cs = rig()
    n = 0
    for i in range(5 * INTERVAL):
      _, sends = step(cc, cs, lat_active=True, stock_tja=4)
      n += presses(sends)
      assert n == min(i // INTERVAL + 1, CarControllerParams.TJA_PRESS_MAX), i
      # the warning fires once, one interval after the last press, and openpilot keeps steering
      expect_stuck = i == CarControllerParams.TJA_PRESS_MAX * INTERVAL
      assert cs.stock_cts_stuck == expect_stuck, i
      cs.stock_cts_stuck = False  # carstate consumes it
    assert n == CarControllerParams.TJA_PRESS_MAX

  def test_the_episode_resets_through_benign_tja_two(self):
    cc, cs = rig()
    for _ in range(4 * INTERVAL):
      step(cc, cs, lat_active=True, stock_tja=4)
    cs.stock_cts_stuck = False
    assert presses(step(cc, cs, lat_active=True, stock_tja=2)[1]) == 0
    assert (cc.tja_press_count, cc.tja_press_frame, cc.tja_episode_alerted) == (0, None, False)
    assert presses(step(cc, cs, lat_active=True, stock_tja=4)[1]) == 1
    assert not cs.stock_cts_stuck

  def test_a_pause_in_steering_does_not_reset_the_count(self):
    cc, cs = rig()
    for _ in range(INTERVAL + 1):
      step(cc, cs, lat_active=True, stock_tja=4)
    assert cc.tja_press_count == 2
    press_frame = cc.tja_press_frame
    for _ in range(INTERVAL):
      assert presses(step(cc, cs, lat_active=False, stock_tja=4)[1]) == 0
    assert (cc.tja_press_count, cc.tja_press_frame) == (2, press_frame)
    assert presses(step(cc, cs, lat_active=True, stock_tja=4)[1]) == 1
    assert cc.tja_press_count == CarControllerParams.TJA_PRESS_MAX
    for _ in range(2 * INTERVAL):
      assert presses(step(cc, cs, lat_active=True, stock_tja=4)[1]) == 0

  def test_same_press_under_openpilot_longitudinal(self):
    cc, cs = rig(alpha_long=True)
    _, sends = step(cc, cs, lat_active=True, stock_tja=4, radar_was_silenced=True)
    assert presses(sends) == 1
