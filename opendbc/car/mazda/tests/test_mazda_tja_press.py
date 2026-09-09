"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The camera press: while openpilot steers with the camera's own TJA/CTS armed (0x440 TJA
nonzero), the controller presses the camera's button off on its own bus so the two
lane-centering systems never run at once. One frame per press, at least one 0x440 period
between presses, three per arming episode, then one stockLkas pulse; the episode resets when
the camera reads 0. Not gated on the TJA button declaration.
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

  def test_one_press_on_the_first_steering_frame_with_the_camera_armed(self):
    cc, cs = rig()
    assert presses(step(cc, cs, lat_active=True, stock_tja=2, crz_btns_counter=7)[1]) == 1

  def test_counter_is_the_wheels_plus_one(self):
    cc, cs = rig()
    _, sends = step(cc, cs, lat_active=True, stock_tja=2, crz_btns_counter=7)
    assert frames(sends, CRZ_BTNS, bus=2)[0][3] == 0xc0 | (8 << 2)

  def test_no_press_while_not_steering_or_with_the_camera_off(self):
    cc, cs = rig()
    for _ in range(3 * INTERVAL):
      assert presses(step(cc, cs, lat_active=False, stock_tja=4)[1]) == 0
    for _ in range(3 * INTERVAL):
      assert presses(step(cc, cs, lat_active=True, stock_tja=0)[1]) == 0

  def test_cadence_cap_and_the_one_shot_warning(self):
    cc, cs = rig()
    n = 0
    for i in range(5 * INTERVAL):
      _, sends = step(cc, cs, lat_active=True, stock_tja=2)
      n += presses(sends)
      assert n == min(i // INTERVAL + 1, CarControllerParams.TJA_PRESS_MAX), i
      # the warning fires once, one interval after the last press, and openpilot keeps steering
      expect_stuck = i == CarControllerParams.TJA_PRESS_MAX * INTERVAL
      assert cs.stock_cts_stuck == expect_stuck, i
      cs.stock_cts_stuck = False  # carstate consumes it
    assert n == CarControllerParams.TJA_PRESS_MAX

  def test_the_episode_resets_when_the_camera_reads_off(self):
    cc, cs = rig()
    for _ in range(4 * INTERVAL):
      step(cc, cs, lat_active=True, stock_tja=2)
    cs.stock_cts_stuck = False
    step(cc, cs, lat_active=True, stock_tja=0)
    # the driver arms it again under us: a fresh episode, pressed at once
    assert presses(step(cc, cs, lat_active=True, stock_tja=2)[1]) == 1
    assert not cs.stock_cts_stuck

  def test_a_pause_in_steering_does_not_reset_the_count(self):
    # only the camera reading 0 ends an episode; dropping lateral for a moment does not
    cc, cs = rig()
    for _ in range(4 * INTERVAL):
      step(cc, cs, lat_active=True, stock_tja=2)
    cs.stock_cts_stuck = False
    for _ in range(INTERVAL):
      step(cc, cs, lat_active=False, stock_tja=2)
    for _ in range(2 * INTERVAL):
      assert presses(step(cc, cs, lat_active=True, stock_tja=2)[1]) == 0
      assert not cs.stock_cts_stuck

  def test_same_press_under_openpilot_longitudinal(self):
    cc, cs = rig(alpha_long=True)
    _, sends = step(cc, cs, lat_active=True, stock_tja=3, radar_was_silenced=True)
    assert presses(sends) == 1
