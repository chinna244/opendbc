"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from enum import IntFlag


class MazdaFlagsSP(IntFlag):
  # Default-on trial: show Mazda's captured white steering icon while MADS is active and MRCC is off.
  EXPERIMENTAL_MADS_WHITE_HUD = 1
  # Default-off road probe: emit one WHITE frame after MRCC has been stably ACTIVE.
  EXPERIMENTAL_MADS_WHITE_HUD_ACTIVE_PROBE = 2
