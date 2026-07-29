from enum import StrEnum

from opendbc.car import uds
from opendbc.car.can_definitions import CanData

RADAR_ADDR = 0x764
RADAR_BUS = 0


def create_radar_session_msg(session_type: int) -> CanData:
  """UDS DIAGNOSTIC_SESSION_CONTROL, fire-and-forget single frame.

  The radar does not support COMMUNICATION_CONTROL (0x28 replies NRC 0x11), so
  disable_ecu() cannot be used. A programming session stops all of its periodic frames
  (CRZ_INFO, CRZ_CTRL, 0x499, tracks 0x361-0x366) while CRZ_EVENTS and PEDALS, owned by
  other ECUs, keep transmitting. The radar stays silent as long as tester present keeps
  arriving; it falls back to the default session on its ~5 s S3 timeout otherwise.
  WARNING: the programming session DISABLES AEB while in effect!"""
  return CanData(RADAR_ADDR, bytes([0x02, uds.SERVICE_TYPE.DIAGNOSTIC_SESSION_CONTROL, session_type, 0x00, 0x00, 0x00, 0x00, 0x00]), RADAR_BUS)


class RadarSessionState(StrEnum):
  STOCK = "stock"          # radar broadcasting; nothing transmitted
  SILENCING = "silencing"  # requesting the programming session
  SILENCED = "silenced"    # radar quiet; tester present + synthetic frames
  HANDBACK = "handback"    # requesting the default session; synthetic frames continue


class RadarSessionManager:
  """Sequences the radar in and out of its UDS programming session.

  Setup is deferred until the FSC camera finishes its cold-boot radar-presence
  check: silencing the radar within ~2 s of the FSC's boot-settle broadcast latches
  an i-ACTIVSENSE fault that only a ~15 min power-down clears, while waiting ~8 s
  is proven clean (docs/mazda-alpha-long-setup-teardown.md). The check verdict is
  invisible until first motion, so the gate is carstate's settle-signal timer, not
  any fault bit. Teardown must complete while the processes are still running:
  pandad blocks TX within ~100 ms of an onroad cycle starting, so the hand-back
  runs from the control loop and the restart is requested only once the stock
  radar is heard again (back to STOCK, nothing transmitted).
  """

  def __init__(self):
    self.state = RadarSessionState.STOCK

  def update(self, gate_passed: bool, stock_radar_alive: bool, handback: bool) -> RadarSessionState:
    if handback:
      if self.state == RadarSessionState.SILENCING:
        # nothing was torn down yet; just stop touching the bus
        self.state = RadarSessionState.STOCK
      elif self.state == RadarSessionState.SILENCED:
        self.state = RadarSessionState.HANDBACK
      elif self.state == RadarSessionState.HANDBACK and stock_radar_alive:
        self.state = RadarSessionState.STOCK
    else:
      if self.state == RadarSessionState.HANDBACK:
        # hand-back withdrawn (toggle flipped back before the restart): the radar is
        # stock again, so re-run the normal takeover
        self.state = RadarSessionState.STOCK
      if self.state == RadarSessionState.STOCK and gate_passed:
        self.state = RadarSessionState.SILENCING if stock_radar_alive else RadarSessionState.SILENCED
      elif self.state == RadarSessionState.SILENCING and not stock_radar_alive:
        self.state = RadarSessionState.SILENCED
      elif self.state == RadarSessionState.SILENCED and stock_radar_alive:
        # the radar S3-recovered (e.g. a dropped tester present); re-silence it
        self.state = RadarSessionState.SILENCING

    return self.state
