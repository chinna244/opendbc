from enum import StrEnum

from opendbc.car import DT_CTRL, uds
from opendbc.car.can_definitions import CanData
from opendbc.car.mazda.values import CarControllerParams

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


HOLD_CTRL_LATCH_FRAMES = int(CarControllerParams.HOLD_CTRL_LATCH_T / DT_CTRL)
HOLD_LATCH_FRAMES = int(CarControllerParams.HOLD_LATCH_T / DT_CTRL)
HOLD_PASSIVE_FRAMES = int(CarControllerParams.HOLD_PASSIVE_T / DT_CTRL)
RESUME_RELEASE_FRAMES = int(CarControllerParams.RESUME_RELEASE_T / DT_CTRL)
RESUME_REACTIVATE_FRAMES = int(CarControllerParams.RESUME_REACTIVATE_T / DT_CTRL)
RESUME_UNLATCH_FRAMES = int(CarControllerParams.RESUME_UNLATCH_T / DT_CTRL)


class StopGoState(StrEnum):
  CRUISING = "cruising"
  STOPPING = "stopping"
  HOLD = "hold"
  HOLD_LATCHED = "hold latched"
  HOLD_PASSIVE = "hold passive"
  RESUMING = "resuming"


class StopAndGoStateMachine:
  """Replays stock MRCC's stop-and-go sequence: ramp to a stop, hold at -1.024 m/s2,
  relax to the latched hold, drop to the passive hold on long stops, and release
  through a short brake-release window on resume."""

  def __init__(self):
    self._reset()

  def _reset(self):
    self.state = StopGoState.CRUISING
    self.hold_frames = 0
    self.release_frames = 0
    self.reactivate_frames = 0
    self.unlatch_frames = 0

  def update(self, long_active: bool, stopping: bool, standstill: bool,
             resume_pressed: bool = False, virtual_resume: bool = False,
             gas_override: bool = False) -> StopGoState:
    if not long_active:
      self._reset()
      return self.state

    if self.state == StopGoState.CRUISING:
      if stopping:
        self.state = StopGoState.STOPPING

    elif self.state == StopGoState.STOPPING:
      if standstill:
        self.state = StopGoState.HOLD
        self.hold_frames = 0
      elif not stopping:
        self.state = StopGoState.CRUISING

    elif self.state in (StopGoState.HOLD, StopGoState.HOLD_LATCHED, StopGoState.HOLD_PASSIVE):
      self.hold_frames += 1
      # Stock's earliest observed hold release comes after the 2 s CRZ_CTRL latch, so a
      # physical RES waits for it. A virtual resume additionally waits for the relaxed
      # hold command so a transient plan flicker cannot release the hold early.
      ctrl_latched = self.hold_frames >= HOLD_CTRL_LATCH_FRAMES
      resume = (resume_pressed and ctrl_latched) or (virtual_resume and self.state != StopGoState.HOLD)
      if gas_override or resume:
        # stock briefly re-raises the latched profile when resuming out of a passive hold
        self.reactivate_frames = RESUME_REACTIVATE_FRAMES if self.state == StopGoState.HOLD_PASSIVE else 0
        self.release_frames = RESUME_RELEASE_FRAMES
        self.unlatch_frames = RESUME_UNLATCH_FRAMES
        self.state = StopGoState.RESUMING
      elif self.hold_frames >= HOLD_PASSIVE_FRAMES:
        self.state = StopGoState.HOLD_PASSIVE
      elif self.hold_frames >= HOLD_LATCH_FRAMES:
        self.state = StopGoState.HOLD_LATCHED

    elif self.state == StopGoState.RESUMING:
      if self.reactivate_frames > 0:
        self.reactivate_frames -= 1
      elif self.unlatch_frames > 0:
        self.unlatch_frames -= 1
      if resume_pressed or virtual_resume or gas_override:
        # keep the brake released while the resume request holds
        self.release_frames = RESUME_RELEASE_FRAMES
      else:
        self.release_frames -= 1
      if self.release_frames <= 0:
        # re-hold if the car never moved, otherwise hand control back to the plan
        self.state = StopGoState.HOLD if standstill else StopGoState.CRUISING
        self.hold_frames = 0

    return self.state

  @property
  def stop_bits(self) -> bool:
    # CRZ_INFO stop flags are held through the approach and the strong hold, and clear
    # once the hold command relaxes
    return self.state in (StopGoState.STOPPING, StopGoState.HOLD)

  @property
  def resume_unlatching(self) -> bool:
    return self.state == StopGoState.RESUMING and self.reactivate_frames == 0 and self.unlatch_frames > 0

  @property
  def acc_active_2(self) -> bool:
    # stock drops ACC_ACTIVE_2 together with the command relax at the hold latch
    return self.state not in (StopGoState.HOLD_LATCHED, StopGoState.HOLD_PASSIVE)

  def radar_has_lead(self, lead_visible: bool) -> bool:
    return lead_visible or self.state in (StopGoState.STOPPING, StopGoState.HOLD,
                                          StopGoState.HOLD_LATCHED, StopGoState.HOLD_PASSIVE)

  def ctrl_phase(self, lead_visible: bool) -> int:
    # RADAR_LEAD_RELATIVE_DISTANCE: 1 cruise, 2 follow, 3 stop/hold, 4 hold-far. Stock holds a
    # constant stop phase through the whole hold and drops to follow on resume; without a real
    # lead distance we advertise the near stop phase (3) throughout the hold.
    if self.state == StopGoState.CRUISING:
      return 2 if lead_visible else 1
    return 3
