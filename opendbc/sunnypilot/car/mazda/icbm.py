"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from opendbc.car import structs, DT_CTRL
from opendbc.car.can_definitions import CanData
from opendbc.car.mazda import mazdacan
from opendbc.car.mazda.values import Buttons
from opendbc.sunnypilot.car.icbm_actuation_profile import get_actuation_profile
from opendbc.sunnypilot.car.intelligent_cruise_button_management_interface_base import IntelligentCruiseButtonManagementInterfaceBase

ButtonType = structs.CarState.ButtonEvent.Type
SendButtonState = structs.IntelligentCruiseButtonManagement.SendButtonState

BUTTONS = {
  SendButtonState.increase: Buttons.SET_PLUS,
  SendButtonState.decrease: Buttons.SET_MINUS,
  SendButtonState.increaseHold: Buttons.SET_PLUS,
  SendButtonState.decreaseHold: Buttons.SET_MINUS,
}
HOLD_BUTTONS = (SendButtonState.increaseHold, SendButtonState.decreaseHold)

# The body ECU registers at most ~1 discrete press per 200 ms; a tighter cadence makes it
# drop presses (measured ~0.93 mph/press at 5 Hz vs ~0.47 at 9 Hz, so faster sending gives
# a slower dash). Hold frames go out at the CRZ_BTNS native rate instead: the wheel's
# genuine frames (button bit 0) interleave with ours either way, so a synthesized hold has
# to out-shout them at message rate for the ECU to integrate it as a hold. Whether it does
# is decided on-car; the servo watches the dash and falls back to discrete taps if the
# long-press step never lands.
HOLD_PERIOD = 0.02  # s between hold frames (CRZ_BTNS native 50 Hz)


class IntelligentCruiseButtonManagementInterface(IntelligentCruiseButtonManagementInterfaceBase):
  def __init__(self, CP, CP_SP):
    super().__init__(CP, CP_SP)
    self.tap_period = 1. / get_actuation_profile(CP.brand).tap_rate_hz

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
      since_last_send = (self.frame - self.last_button_frame) * DT_CTRL

      if self.ICBM.sendButton in HOLD_BUTTONS:
        # Sustained hold: one frame per native message slot. The genuine counter advances
        # between consecutive sends at this cadence, so a fixed +1 offset stays unique.
        if since_last_send > HOLD_PERIOD:
          can_sends.append(mazdacan.create_button_cmd(packer, self.CP, CS.crz_btns_counter + 1, send_button, CS))
          self.last_button_frame = self.frame
      else:
        # Discrete tap, paced to the ECU's registration floor
        if since_last_send > self.tap_period:
          self.button_frame += 1
          button_counter_offset = [1, 1, 0, None][self.button_frame % 4]
          if button_counter_offset is not None:
            can_sends.append(mazdacan.create_button_cmd(packer, self.CP, CS.crz_btns_counter + button_counter_offset, send_button, CS))
            self.last_button_frame = self.frame

    return can_sends
