"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from enum import IntFlag


class MazdaSafetyFlagsSP:
  DEFAULT = 0
  # The physical TJA button is the MADS lateral switch; MRCC no longer drives the main edge.
  TJA_BUTTON = 1
  # The Smart CAN Filter: FILTER_CTRL (0x2A0) may go out, the UDS takeover and synthetic radar frames may not.
  CAN_FILTER = 2


class MazdaFlagsSP(IntFlag):
  # Fitted to some trims only and not predicted by the fingerprint, so the driver declares it.
  TJA_BUTTON = 1
  # The cluster shows the held cruise speed over-read, displayed = held / 0.98 + 1 km/h
  # (ADR speedometer calibration, indicated never below true). Measured on a NZ CX-9 2021 (four
  # samples, zero residual, zoompilot/opendbc#7); NA and non-Oceania export clusters show the
  # CAN value exactly. Keyed on the Oceania WMI, so cruiseState.speedCluster is the dash number.
  OCEANIA_CLUSTER = 2
  # A Smart CAN Filter sits between the radar and the car, declared by the driver: under alpha long it keeps
  # the radar online in place of the UDS takeover (opendbc/car/mazda/can_filter.py).
  CAN_FILTER = 4
