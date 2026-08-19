"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from enum import StrEnum

from opendbc.car import Bus, structs
from opendbc.can.parser import CANParser
from opendbc.car.common.conversions import Conversions as CV


class CarStateExt:
  def __init__(self, CP, CP_SP):
    self.CP = CP
    self.CP_SP = CP_SP

  def update(self, ret: structs.CarState, ret_sp: structs.CarStateSP, can_parsers: dict[StrEnum, CANParser]) -> None:
    cp_cam = can_parsers[Bus.cam]

    # CAM_TRAFFIC_SIGNS comes from the front camera. SPEED_SIGN_ON encodes both display
    # state and unit: 1 = limit displayed in mph (US-market FSC), 2 = limit displayed in
    # km/h (metric-market FSC), 0 = none. The unit comes from the frame, not the car's
    # display setting. Limit displayed = camera-detected or car's internal map fallback
    # (SPEED_SIGN_CAM distinguishes but both are trustworthy).
    # Plausibility: 90 covers the highest US posting (85 mph); SPEED_SIGN is 7 bits so km/h
    # tops out at the field width, with all-ones (127) excluded as an invalid sentinel.
    sign = cp_cam.vl["CAM_TRAFFIC_SIGNS"]
    speed_sign = sign["SPEED_SIGN"]
    if sign["SPEED_SIGN_ON"] == 1 and 0 < speed_sign <= 90:
      ret_sp.speedLimit = float(speed_sign) * CV.MPH_TO_MS
    elif sign["SPEED_SIGN_ON"] == 2 and 0 < speed_sign < 127:
      ret_sp.speedLimit = float(speed_sign) * CV.KPH_TO_MS
    else:
      ret_sp.speedLimit = 0.0
