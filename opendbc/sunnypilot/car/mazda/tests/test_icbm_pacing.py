"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

ICBM button emission: discrete taps at the ECU's measured 5 Hz registration floor (a
tighter cadence makes the body ECU drop presses), sustained holds at the CRZ_BTNS native
50 Hz rate, and same-frame suppression while the driver presses a physical button.
"""
import unittest
from types import SimpleNamespace

from opendbc.can import CANPacker
from opendbc.car import structs
from opendbc.car.mazda.values import MazdaFlags
from opendbc.sunnypilot.car.mazda.icbm import IntelligentCruiseButtonManagementInterface

SendButtonState = structs.IntelligentCruiseButtonManagement.SendButtonState


def make_carparams():
  cp = structs.CarParams()
  cp.carFingerprint = "MAZDA_CX5_2022"
  cp.brand = "mazda"
  cp.flags = MazdaFlags.GEN1.value
  return cp


def make_carcontrolsp(send_button):
  ccsp = structs.CarControlSP()
  ccsp.intelligentCruiseButtonManagement.sendButton = send_button
  return ccsp


class TestIcbmEmission(unittest.TestCase):
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

  def gaps(self, sends):
    return [b - a for a, b in zip(sends, sends[1:], strict=False)]

  def test_taps_hold_steady_pace(self):
    """No ramp: the cadence stays at the 5 Hz registration floor no matter how long the
    send runs (the ECU drops presses at tighter spacing; measured, F4)."""
    sends = self.run_frames([SendButtonState.increase] * 600)

    assert all(g >= 20 for g in self.gaps(sends)), f"taps must never exceed 5 Hz: {self.gaps(sends)}"
    assert len(sends) >= 600 / 28, f"cadence unexpectedly slow: {len(sends)} sends"

  def test_direction_change_keeps_pace(self):
    self.run_frames([SendButtonState.increase] * 200)

    sends = self.run_frames([SendButtonState.decrease] * 100)
    assert all(g >= 20 for g in self.gaps(sends))

  def test_hold_emits_at_native_rate(self):
    """A hold must out-shout the wheel's genuine 50 Hz frames: one forged frame per native
    message slot for as long as the servo asserts it."""
    sends = self.run_frames([SendButtonState.increaseHold] * 100)

    assert len(sends) >= 30, f"hold must emit at ~50 Hz, got {len(sends)} in 1 s"
    assert all(g <= 3 for g in self.gaps(sends)), f"hold gaps too wide: {self.gaps(sends)}"

  def test_hold_release_stops_emission(self):
    self.run_frames([SendButtonState.increaseHold] * 60)
    sends = self.run_frames([SendButtonState.none] * 60)
    assert sends == []

  def test_hold_then_taps_repace(self):
    """After a hold, remainder taps go back to the 5 Hz floor."""
    self.run_frames([SendButtonState.increaseHold] * 60)
    sends = self.run_frames([SendButtonState.increase] * 100)
    assert all(g >= 20 for g in self.gaps(sends))

  def test_idle_sends_nothing(self):
    sends = self.run_frames([SendButtonState.none] * 100)
    assert sends == []

  def test_driver_press_suppresses_sends(self):
    self.CS.accel_button = 1
    sends = self.run_frames([SendButtonState.increase] * 100)
    assert sends == []

    self.CS.accel_button = 0
    sends = self.run_frames([SendButtonState.increase] * 100)
    assert len(sends) >= 3
    assert all(g >= 20 for g in self.gaps(sends))

  def test_driver_press_suppresses_hold(self):
    self.CS.decel_button = 1
    sends = self.run_frames([SendButtonState.decreaseHold] * 100)
    assert sends == []


if __name__ == "__main__":
  unittest.main()
