"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

ICBM button send pacing: conservative 0.2s cadence normally, tightened to 0.1s during a
sustained same-direction ramp so deep smart-cruise dips recover in half the time. Any
direction change or pause in sending resets to the conservative cadence.
"""
import unittest
from types import SimpleNamespace

from opendbc.can import CANPacker
from opendbc.car import structs
from opendbc.car.mazda.values import MazdaFlags
from opendbc.sunnypilot.car.mazda.icbm import IntelligentCruiseButtonManagementInterface, RAMP_MIN_SENDS

SendButtonState = structs.IntelligentCruiseButtonManagement.SendButtonState


def make_carparams():
  cp = structs.CarParams()
  cp.carFingerprint = "MAZDA_CX5_2022"
  cp.flags = MazdaFlags.GEN1.value
  return cp


def make_carcontrolsp(send_button):
  ccsp = structs.CarControlSP()
  ccsp.intelligentCruiseButtonManagement.sendButton = send_button
  return ccsp


class TestIcbmPacing(unittest.TestCase):
  def setUp(self):
    self.icbm = IntelligentCruiseButtonManagementInterface(make_carparams(), structs.CarParamsSP())
    self.packer = CANPacker("mazda_2017")
    self.CS = SimpleNamespace(crz_btns_counter=0, accel_button=0, decel_button=0)
    self.frame = 0
    self.last_button_frame = 0

  def run_frames(self, buttons_by_frame):
    """buttons_by_frame: iterable of SendButtonState, one per 100Hz frame. Frame numbering
    continues across calls. Returns the frame numbers on which a button frame was sent."""
    send_frames = []
    for btn in buttons_by_frame:
      sends = self.icbm.update(make_carcontrolsp(btn), self.CS, self.packer, self.frame, self.last_button_frame)
      self.last_button_frame = self.icbm.last_button_frame
      if sends:
        send_frames.append(self.frame)
      self.frame += 1
    return send_frames

  def assert_starts_at_normal_pace(self, sends, why):
    assert len(sends) >= RAMP_MIN_SENDS, sends
    first_gaps = [b - a for a, b in zip(sends, sends[1:], strict=False)][:RAMP_MIN_SENDS - 1]
    assert all(g >= 20 for g in first_gaps), f"{why} must reset to 0.2s pace: {first_gaps}"

  def test_ramp_tightens_pace(self):
    sends = self.run_frames([SendButtonState.increase] * 300)

    gaps = [b - a for a, b in zip(sends, sends[1:], strict=False)]
    early, late = gaps[:RAMP_MIN_SENDS - 1], gaps[RAMP_MIN_SENDS:]
    assert all(g >= 20 for g in early), f"early sends must pace at 0.2s: {gaps}"
    assert all(g <= 12 for g in late), f"sustained ramp must pace at ~0.1s: {gaps}"
    # twice the throughput of the fixed 0.2s pace over the sustained portion
    assert len(sends) > 300 / 21, sends

  def test_direction_change_resets_pace(self):
    self.run_frames([SendButtonState.increase] * 200)

    sends = self.run_frames([SendButtonState.decrease] * 100)
    self.assert_starts_at_normal_pace(sends, "direction change")

  def test_pause_resets_pace(self):
    self.run_frames([SendButtonState.increase] * 200)

    frames = [SendButtonState.none] * 60 + [SendButtonState.increase] * 100
    sends = self.run_frames(frames)
    self.assert_starts_at_normal_pace(sends, "pause")

  def test_idle_sends_nothing(self):
    sends = self.run_frames([SendButtonState.none] * 100)
    assert sends == []

  def test_driver_press_suppresses_sends(self):
    self.CS.accel_button = 1
    sends = self.run_frames([SendButtonState.increase] * 100)
    assert sends == []

    self.CS.accel_button = 0
    sends = self.run_frames([SendButtonState.increase] * 100)
    self.assert_starts_at_normal_pace(sends, "driver-press suppression")


if __name__ == "__main__":
  unittest.main()
