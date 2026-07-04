"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from opendbc.car import structs, DT_CTRL
from opendbc.car.can_definitions import CanData
from opendbc.car.mazda import mazdacan
from opendbc.car.mazda.values import Buttons
from opendbc.sunnypilot.car.intelligent_cruise_button_management_interface_base import IntelligentCruiseButtonManagementInterfaceBase

ButtonType = structs.CarState.ButtonEvent.Type
SendButtonState = structs.IntelligentCruiseButtonManagement.SendButtonState

BUTTONS = {
  SendButtonState.increase: Buttons.SET_PLUS,
  SendButtonState.decrease: Buttons.SET_MINUS,
}

# Send pacing. One press moves the dash 1 mph, so a fixed 0.2s pace caps tracking at ~5 mph/s
# and restoring the set speed after a deep smart-cruise slowdown takes several seconds. The
# dash confirms a press in ~50 ms (p90 ~80 ms, measured on CX-5 2022), so once a ramp is
# clearly sustained in one direction we tighten the pace. A direction change or a pause in
# sending resets to the conservative cadence.
PACE_NORMAL = 0.2  # s between button frames
PACE_RAMP = 0.1  # s between button frames during a sustained same-direction ramp
RAMP_MIN_SENDS = 3  # consecutive same-direction sends before tightening the pace
RAMP_RESET_TIME = 0.5  # s without a send resets the ramp


class IntelligentCruiseButtonManagementInterface(IntelligentCruiseButtonManagementInterfaceBase):
  def __init__(self, CP, CP_SP):
    super().__init__(CP, CP_SP)
    self.ramp_send_button = SendButtonState.none
    self.ramp_sends = 0

  def update(self, CC_SP, CS, packer, frame, last_button_frame) -> list[CanData]:
    can_sends = []
    self.CC_SP = CC_SP
    self.ICBM = CC_SP.intelligentCruiseButtonManagement
    self.frame = frame
    self.last_button_frame = last_button_frame

    # Same-frame suppression while the driver holds SET+/SET-: the selfdrived readiness gate
    # also pauses ICBM on driver presses, but only after a few frames of messaging latency,
    # during which a forged frame (with the driver's button bit at 0) could interleave with
    # the wheel's own frames and make the body ECU drop or miscount the press.
    if CS.accel_button or CS.decel_button:
      return can_sends

    if self.ICBM.sendButton != SendButtonState.none:
      send_button = BUTTONS[self.ICBM.sendButton]

      # The time-gap reset also covers gaps where update() wasn't called or was suppressed
      # (cancel/resume in flight, driver presses); the else branch covers brief pauses.
      since_last_send = (self.frame - self.last_button_frame) * DT_CTRL
      if self.ICBM.sendButton != self.ramp_send_button or since_last_send > RAMP_RESET_TIME:
        self.ramp_send_button = self.ICBM.sendButton
        self.ramp_sends = 0

      pace = PACE_RAMP if self.ramp_sends >= RAMP_MIN_SENDS else PACE_NORMAL
      if since_last_send > pace:
        self.button_frame += 1
        button_counter_offset = [1, 1, 0, None][self.button_frame % 4]
        if button_counter_offset is not None:
          can_sends.append(mazdacan.create_button_cmd(packer, self.CP, CS.crz_btns_counter + button_counter_offset, send_button))
          self.ramp_sends += 1
          self.last_button_frame = self.frame
    else:
      self.ramp_send_button = SendButtonState.none
      self.ramp_sends = 0

    return can_sends
