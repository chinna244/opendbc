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

    # CAM_TRAFFIC_SIGNS comes from the front camera. SPEED_SIGN is in MPH per DBC.
    # SPEED_SIGN_ON=1 means the cluster is displaying a limit (camera-detected or car's
    # internal map fallback — SPEED_SIGN_CAM distinguishes but both are trustworthy).
    sign = cp_cam.vl["CAM_TRAFFIC_SIGNS"]
    speed_sign = sign["SPEED_SIGN"]
    if sign["SPEED_SIGN_ON"] == 1 and 0 < speed_sign <= 120:
      ret_sp.speedLimit = float(speed_sign) * CV.MPH_TO_MS
    else:
      ret_sp.speedLimit = 0.0
