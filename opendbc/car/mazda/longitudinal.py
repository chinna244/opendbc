from enum import StrEnum

from opendbc.car import DT_CTRL, uds
from opendbc.car.can_definitions import CanData
from opendbc.car.mazda import mazdacan
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


RESUME_UNLATCH_FRAMES = int(CarControllerParams.RESUME_UNLATCH_T / DT_CTRL)
LEAD_DEBOUNCE_FRAMES = int(CarControllerParams.LEAD_DEBOUNCE_T / DT_CTRL)


class StandstillHold:
  """Holds the car stopped until the plan asks to move, the way Toyota and Honda do it.

  Both upstream ports drive the standstill request straight off the plan and off car feedback,
  with no timers in the path: Toyota clears its request on `actuators.accel > 0` and re-asserts
  it whenever the plan is not asking to move, and Honda asserts STANDSTILL for exactly as long
  as long control is in its stopping state. Neither ever substitutes a canned command for the
  plan's own -- LongControl already parks at CP.stopAccel while stopping, which for this car is
  the stock hold value.

  The relax off that hold is the one thing the car, not the plan, decides: stock lets go the
  instant the body ECU latches its own brake hold (GEAR.BRAKE_HOLD), which can take anywhere
  from nothing to several seconds. If the latch never comes we simply keep braking.

  Nothing here latches: `holding` is recomputed every frame, so a plan that changes its mind
  gets the hold straight back.
  """

  def __init__(self):
    self._reset()

  def _reset(self):
    self.holding = False
    self.car_has_hold = False
    self.unlatch_frames = 0

  def update(self, long_active: bool, stopping: bool, standstill: bool,
             plan_accel: float, brake_hold: bool) -> None:
    if not long_active:
      self._reset()
      return

    was_holding = self.holding
    # the plan asking for acceleration is the only thing that releases the hold
    if plan_accel > 0.:
      self.holding = False
    elif stopping or standstill:
      self.holding = True

    if self.unlatch_frames > 0:
      self.unlatch_frames -= 1
    if was_holding and not self.holding and standstill:
      self.unlatch_frames = RESUME_UNLATCH_FRAMES

    # the body only owns the brakes while we are still asking it to hold
    self.car_has_hold = self.holding and standstill and brake_hold

  @property
  def stop_bits(self) -> bool:
    # CRZ_INFO stop flags are held through the approach and the hold, and clear when the car
    # takes over and the command relaxes
    return self.holding and not self.car_has_hold

  @property
  def resume_unlatching(self) -> bool:
    return self.unlatch_frames > 0

  @property
  def acc_active_2(self) -> bool:
    # stock drops ACC_ACTIVE_2 together with the command relax
    return not self.car_has_hold


class AdvertisedLead:
  """The lead we tell the camera about: CRZ_CTRL's two lead fields and the 0x364 track slot.

  All three describe one fact, and stock pairs them absolutely -- RADAR_HAS_LEAD=1 never came
  with all six slots empty, and has_lead=0 always came with phase=0 -- so they are read off one
  piece of state here rather than computed separately and kept in step by hand.

  Two things make that state more than a copy of leadVisible. A marginal vision lead flickers
  faster than any real radar ever would (route 6bb2dc61c4 t+400: 6 toggles in 1.4 s on a 120 m
  lead), so visibility is adopted only once it has held steady, the way Hyundai debounces its
  lead bit. And leadOne drops to zero the instant vision loses the lead, well before that
  debounce expires; advertising a fabricated stand-in over the gap put a stationary object
  10.25 m dead ahead on the bus at 22 m/s, so the last real measurement is coasted across it
  instead, the way a radar coasts a track.
  """

  def __init__(self):
    self._reset()

  def _reset(self):
    self.visible = False
    self.flip_frames = 0
    self.holding = False
    self.lead = None
    self._measured = None

  def update(self, long_engaged: bool, lead_visible: bool, d_rel: float, v_rel: float,
             holding: bool) -> None:
    if not long_engaged:
      self._reset()
      return

    if lead_visible != self.visible:
      self.flip_frames += 1
      if self.flip_frames >= LEAD_DEBOUNCE_FRAMES:
        self.visible = lead_visible
        self.flip_frames = 0
    else:
      self.flip_frames = 0

    if 0. < d_rel <= mazdacan.DIST_OBJ_MAX:
      self._measured = (d_rel, v_rel)
    self.lead = self._measured if self.visible else None
    self.holding = holding

  @property
  def has_lead(self) -> bool:
    return self.lead is not None

  @property
  def ctrl_phase(self) -> int:
    # RADAR_LEAD_RELATIVE_DISTANCE: 0 nothing to describe, 1 cruise, 2 follow, 3 stop/hold.
    # We shipped has_lead=0 with phase=1 for 22-84% of every engaged drive before this was a
    # read off the advertised lead instead of a second, separate computation.
    if not self.has_lead:
      return 0
    return 3 if self.holding else 2
