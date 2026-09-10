"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The stock ECU transition contract.

A brand that silences a stock ECU for openpilot longitudinal (Mazda's radar) reports where
that ownership stands through one status, so card and the UI never read brand state. The
controller carries it as `stock_ecu_status`, updated in place every control frame; a controller
without the attribute silences nothing.

The state is the driver's view: what, if anything, they have to do. Ownership details (which
prerequisite, which UDS reply) go to carlog. READY is the vehicle's own guard on the silence
(the two-master guard, which completes after the panda's), the point from which normal
engagement can follow; a session acknowledgement is never readiness.
"""
from dataclasses import dataclass
from enum import StrEnum


class StockEcuState(StrEnum):
  NOT_NEEDED = "notNeeded"           # nothing to take over on this platform or in this mode
  STARTING = "starting"              # prerequisites, request or silence guard still pending: wait
  PARK_TO_TAKE_OVER = "parkToTakeOver"  # this session takes over at the next stop
  STOCK_CRUISE_ON = "stockCruiseOn"  # the driver's own stock engagement holds the takeover
  READY = "ready"                    # owned and guarded; engage normally
  RESTORING = "restoring"            # the ordered hand-back, requested or done: stock cruise until the stop
  FAILED = "failed"                  # a bounded attempt ended without the ECU answering


@dataclass
class StockEcuStatus:
  state: StockEcuState = StockEcuState.NOT_NEEDED
  # the ordered hand-back's answer, for the lifecycle that asked for it
  handback_completed: bool = False
  handback_failed: bool = False
