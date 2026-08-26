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


RADAR_SESSION_LIMIT_FRAMES = int(CarControllerParams.RADAR_SESSION_LIMIT_T / DT_CTRL)


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

  The session request is fire-and-forget, so a refusal (NRC) is indistinguishable
  from a lost frame; both silencing and the hand-back are therefore bounded the
  way disable_ecu bounds its retries. A silencing episode that never quiets the
  radar gives up for the drive and stock keeps the bus; a hand-back the radar
  never answers stops waiting so the process restart can proceed.
  """

  def __init__(self):
    self.state = RadarSessionState.STOCK
    self.state_frames = 0
    self.silencing_failed = False

  def update(self, gate_passed: bool, stock_radar_alive: bool, handback: bool,
             standstill: bool, session_refused: bool = False) -> RadarSessionState:
    prev_state = self.state
    if handback:
      if self.state == RadarSessionState.SILENCING:
        # nothing was torn down yet; just stop touching the bus
        self.state = RadarSessionState.STOCK
      elif self.state == RadarSessionState.SILENCED:
        self.state = RadarSessionState.HANDBACK
      elif self.state == RadarSessionState.HANDBACK and \
           (stock_radar_alive or self.state_frames >= RADAR_SESSION_LIMIT_FRAMES):
        # heard again, or never coming back: either way stop waiting so the restart proceeds
        self.state = RadarSessionState.STOCK
    else:
      if self.state == RadarSessionState.HANDBACK:
        # hand-back withdrawn (toggle flipped back before the restart): the radar is
        # stock again, so re-run the normal takeover
        self.state = RadarSessionState.STOCK
      if self.state == RadarSessionState.STOCK and gate_passed:
        # actively silencing disables AEB, so like every disable_ecu caller it only starts
        # pre-motion; adopting an already-quiet radar disables nothing and proceeds anywhere
        if not stock_radar_alive:
          self.state = RadarSessionState.SILENCED
        elif standstill and not self.silencing_failed:
          self.state = RadarSessionState.SILENCING
      elif self.state == RadarSessionState.SILENCING:
        if not stock_radar_alive:
          self.state = RadarSessionState.SILENCED
        elif session_refused or self.state_frames >= RADAR_SESSION_LIMIT_FRAMES:
          # a negative UDS response is a definitive refusal, the silence budget covers a
          # radar that answers nothing at all; either way give up for the drive and stock
          # keeps the bus (disable_ecu reads the same response to declare its outcome)
          self.state = RadarSessionState.STOCK
          self.silencing_failed = True
      elif self.state == RadarSessionState.SILENCED and stock_radar_alive:
        # the radar S3-recovered (e.g. a dropped tester present); re-silence it without
        # waiting for a stop: two masters on the bus is the greater hazard, this radar has
        # already accepted sessions this drive, and a refusal still exits through the
        # bounded SILENCING episode above
        self.state = RadarSessionState.SILENCING

    self.state_frames = 0 if self.state != prev_state else self.state_frames + 1
    return self.state


RESUME_UNLATCH_FRAMES = int(CarControllerParams.RESUME_UNLATCH_T / DT_CTRL)
LEAD_DEBOUNCE_FRAMES = int(CarControllerParams.LEAD_DEBOUNCE_T / DT_CTRL)

# Trial (2026-08-26): the escort is off to attribute route 000000fe's fault. That release had
# two deviations from stock, not one: no advertised object AND a 0.20 s unlatch pulse below
# stock's observed 0.22-0.38 s floor; the pulse now sits mid-distribution. A clean no-lead
# resume here means the camera never needed the object and the escort can be deleted; a fault
# proves the object requirement single-variable and the escort comes back on.
ESCORT_ENABLED = False

# The escort ghost's kinematics; single consumer, so they live with it
ESCORT_LEAD_IN_FRAMES = int(0.3 / DT_CTRL)  # the ghost pulls away this long before the release fires
ESCORT_RELV_STEP = 1.25 * DT_CTRL           # m/s per frame, opening-rate ramp
ESCORT_RELV_MAX = 2.5                       # m/s, opening-rate cap
ESCORT_DROP_DIST = 18.0                     # m, escort ends here once the car is moving
ESCORT_FAR_DROP_DIST = 36.0                 # m, and here even stopped: an aborted resume's ghost drives away


class ResumeEscort:
  """A departing ghost lead that escorts a no-lead standstill hold release.

  The camera accepts a hold with nothing advertised -- route 000000fe held 6 s at
  has_lead=0/phase=0 with the body latched and HOLD on the dash -- but latched an SCBS fault
  90 ms into the release. Every other observable in that release matched the 23 stock latched
  releases exactly (stop bits, command relax, ACC_ACTIVE_2, pulse width, no RES press); the
  one deviation was resuming with no object, a state stock cannot produce because its ACC
  never stops without a lead. Stock's releases also show the cause before the effect: the lead
  is already pulling away when RESUME_UNLATCHING fires (relv +0.31..+4.75 at the pulse).

  So the fabrication is confined to the one moment the camera demands it. The ghost appears at
  the track template's captured range when the plan first asks to move, recedes through a short
  lead-in before the release is allowed to fire, keeps opening through the drive-off, and drops
  once the car is rolling -- the regime where the camera provably tolerated 20 one-frame track
  discontinuities at up to 22 m/s. A real measurement always takes the slot over.
  """

  def __init__(self):
    self.reset()

  def reset(self):
    self.lead: tuple[float, float] | None = None
    self.lead_in = 0

  def update(self, release_wanted: bool, standstill: bool,
             real_lead: tuple[float, float] | None) -> None:
    if real_lead is not None:
      self.reset()
      return

    if self.lead is None:
      if release_wanted and standstill:
        self.lead = (mazdacan.LEAD_TRACK_DIST, 0.)
        self.lead_in = ESCORT_LEAD_IN_FRAMES
      return

    if self.lead_in > 0:
      self.lead_in -= 1
    d, v = self.lead
    v = min(v + ESCORT_RELV_STEP, ESCORT_RELV_MAX)
    d += v * DT_CTRL
    drop_dist = ESCORT_FAR_DROP_DIST if standstill else ESCORT_DROP_DIST
    if d >= drop_dist:
      self.reset()
    else:
      self.lead = (d, v)

  @property
  def deferring(self) -> bool:
    # the release waits until the ghost has visibly pulled away
    return self.lead_in > 0


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
  gets the hold straight back. A release with no real lead to advertise is the one the plan
  does not fully own: it waits out the escort's lead-in so the camera sees a lead pull away
  before RESUME_UNLATCHING fires, the order every stock release shows it.
  """

  def __init__(self):
    self.escort = ResumeEscort()
    self._reset()

  def _reset(self):
    self.holding = False
    self.car_has_hold = False
    self.unlatch_frames = 0
    self.escort.reset()

  def update(self, long_engaged: bool, stopping: bool, standstill: bool,
             plan_accel: float, brake_hold: bool, gas_pressed: bool,
             real_lead: tuple[float, float] | None = None) -> None:
    if not long_engaged:
      self._reset()
      return

    was_holding = self.holding
    self.escort.update(ESCORT_ENABLED and self.holding and plan_accel > 0., standstill, real_lead)
    # the plan asking for acceleration releases the hold, and so does the driver's pedal:
    # Toyota's PCM lets the pedal outrank its standstill request the same way. Holding the
    # stop bits against the throttle until the car physically moved put an out-of-protocol
    # release on the bus, stop bits dropping at speed with no unlatch pulse (route 0000004d
    # t+210.9). Stock keeps STOPPING strictly to the final creep: 2,078 rolling STOPPING
    # frames in the corpus, all below 0.55 m/s.
    release = gas_pressed or (plan_accel > 0. and not self.escort.deferring)
    self.holding = not release and (stopping or standstill)

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

  The state is perception, not control: a stock radar reports its objects ignition to ignition,
  and stock shows RADAR_HAS_LEAD=1 with cruise disengaged in 19.5% of all frames. Tying the
  advertisement to engagement instead made a real car 4.5 m ahead vanish from the bus in one
  frame when the driver braked out of a creep (route 0000004d t+212); the camera, still watching
  the car close in its own vision, ran its SCBS display six seconds -- a pattern absent from
  50 h of stock driving. So this updates every control frame, engaged or not, for as long as we
  stand in for the radar.

  Two things make the state more than a copy of leadVisible. A marginal vision lead flickers
  faster than any real radar ever would (route 6bb2dc61c4 t+400: 6 toggles in 1.4 s on a 120 m
  lead), so visibility is adopted only once it has held steady, the way Hyundai debounces its
  lead bit. And leadOne drops to zero the instant vision loses the lead, well before that
  debounce expires; advertising a fabricated stand-in over the gap put a stationary object
  10.25 m dead ahead on the bus at 22 m/s, so the last real measurement is coasted across it
  instead, the way a radar coasts a track.
  """

  def __init__(self):
    self.visible = False
    self.flip_frames = 0
    self.holding = False
    self.lead = None
    self.real_lead = None
    self._measured = None

  def update(self, lead_visible: bool, d_rel: float, v_rel: float,
             holding: bool, escort: tuple[float, float] | None = None) -> None:
    if lead_visible != self.visible:
      self.flip_frames += 1
      if self.flip_frames >= LEAD_DEBOUNCE_FRAMES:
        self.visible = lead_visible
        self.flip_frames = 0
    else:
      self.flip_frames = 0

    if 0. < d_rel <= mazdacan.DIST_OBJ_MAX:
      self._measured = (d_rel, v_rel)
    elif not self.visible:
      # a real radar drops a coasted track; expiring here bounds the coast to the debounce
      # window and keeps a minutes-stale measurement from resurfacing on reacquisition
      self._measured = None
    elif self._measured is not None:
      # propagate through the gap rather than repeating one frozen frame; a stock radar
      # re-measures every track every 100 ms, and a frozen range with the car moving is the
      # camera's proven SCBS trigger (create_lead_track's docstring)
      d, v = self._measured
      self._measured = (d + v * DT_CTRL, v)
    # the escort stands in only when there is nothing real to advertise, so has_lead, the
    # phase and the track still come off this one decision
    self.real_lead = self._measured if self.visible else None
    self.lead = self.real_lead or escort
    self.holding = holding

  @property
  def has_lead(self) -> bool:
    return self.lead is not None

  @property
  def ctrl_phase(self) -> int:
    # RADAR_LEAD_RELATIVE_DISTANCE is stock's 1-5 closeness bucket for the lead, broadcast
    # engaged or not (hl=0 <=> phase=0 holds cruise-off too). We emit 2 following and 3 near
    # a hold: both in-distribution (3 is stock's dominant standstill value), and no fault has
    # ever keyed on the bucket value, only on the triple disagreeing.
    if not self.has_lead:
      return 0
    return 3 if self.holding else 2
