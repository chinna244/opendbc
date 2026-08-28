from enum import StrEnum

from opendbc.car import DT_CTRL, uds
from opendbc.car.carlog import carlog
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

  A negative session response fails the silencing episode immediately, and the
  silence budget bounds a radar that answers nothing at all, the way disable_ecu
  bounds its retries; either way the episode gives up for the drive and stock
  keeps the bus. A hand-back the radar never answers stops waiting so the
  process restart can proceed.
  """

  def __init__(self):
    self.state = RadarSessionState.STOCK
    self.state_frames = 0
    self.silencing_failed = False
    self.handback_completed = False

  def update(self, gate_passed: bool, stock_radar_alive: bool, handback: bool,
             standstill: bool, session_refused: bool) -> RadarSessionState:
    prev_state = self.state
    if handback:
      if self.state == RadarSessionState.SILENCING:
        # nothing was torn down yet; just stop touching the bus
        self.state = RadarSessionState.STOCK
      elif self.state == RadarSessionState.SILENCED:
        self.state = RadarSessionState.HANDBACK
      elif self.state == RadarSessionState.HANDBACK and \
           (stock_radar_alive or self.state_frames >= RADAR_SESSION_LIMIT_FRAMES):
        # heard again, or never coming back: either way stop waiting so the restart proceeds.
        # This hand-back ran to completion, so the radar stays stock for the rest of the
        # process. The producer's contract is to hold the assert until the process exits; this
        # latch is the backstop for a producer that does not (a dropped assert would otherwise
        # read as a withdrawal and, parked with the gate still passed, re-silence the radar
        # right before shutdown -- the unattended S3 recovery the hand-back exists to prevent)
        self.state = RadarSessionState.STOCK
        self.handback_completed = True
    else:
      if self.state == RadarSessionState.HANDBACK:
        # hand-back withdrawn (toggle flipped back before the restart): the radar is
        # stock again, so re-run the normal takeover
        self.state = RadarSessionState.STOCK
      if self.state == RadarSessionState.STOCK and gate_passed and not self.handback_completed:
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
          carlog.error(f"radar silencing failed ({'refused' if session_refused else 'no response'}); staying stock")
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


RESUME_UNLATCH_LATCHED_FRAMES = int(CarControllerParams.RESUME_UNLATCH_LATCHED_T / DT_CTRL)
RESUME_PULSE_DEFER_FRAMES = int(CarControllerParams.RESUME_PULSE_DEFER_T / DT_CTRL)
LEAD_DEBOUNCE_FRAMES = int(CarControllerParams.LEAD_DEBOUNCE_T / DT_CTRL)
RELEASE_DEBOUNCE_FRAMES = int(CarControllerParams.RELEASE_DEBOUNCE_T / DT_CTRL)


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
    self.release_frames = 0
    self.latched_release = False
    self.just_released = False
    self.pulse_deferred_frames = 0

  def update(self, long_engaged: bool, stopping: bool, standstill: bool,
             plan_accel: float, brake_hold: bool, gas_pressed: bool) -> None:
    self.just_released = False
    if not long_engaged:
      self._reset()
      return

    was_holding = self.holding
    # the plan's request to move is debounced so a one-frame blip (a lead that inches forward
    # and stops) cannot fire a phantom release pulse at a standstill; the driver's pedal is
    # not debounced, it outranks the hold immediately
    self.release_frames = self.release_frames + 1 if plan_accel > 0. else 0
    plan_wants_go = self.release_frames >= RELEASE_DEBOUNCE_FRAMES
    # the plan asking for acceleration releases the hold, and so does the driver's pedal:
    # Toyota's PCM lets the pedal outrank its standstill request the same way. Holding the
    # stop bits against the throttle until the car physically moved put an out-of-protocol
    # release on the bus, stop bits dropping at speed with no unlatch pulse (route 0000004d
    # t+210.9). Stock keeps STOPPING strictly to the final creep: 2,078 rolling STOPPING
    # frames in the corpus, all below 0.55 m/s.
    release = gas_pressed or plan_wants_go
    self.holding = not release and (stopping or standstill)

    if self.unlatch_frames > 0:
      self.unlatch_frames -= 1
    # one pulse per release, exactly as stock: never restarted while one is still playing.
    # The pulse is the ACC's resume protocol, not the driver's: stock's captured gas-ended
    # hold drops the stop bits with no pulse at all (the pedal is the resume authority), and
    # pulsing there with the override's zeroed command latched an SCBS fault (route 00000103
    # t+163.8; if that fault ever recurs on a gas release, the command side is the next
    # candidate -- stock's steps positive off the pedal where ours holds the override zero)
    if was_holding and not self.holding and standstill and not gas_pressed and self.unlatch_frames == 0:
      # car_has_hold still carries last frame's value here: whether the body owned the brakes
      # going into this release tells us whether there is anything to unlatch at all. A
      # never-latched release has nothing latched, so it emits nothing; a latched release
      # arms the deferred pulse below and lets the relaxing command try first.
      self.latched_release = self.car_has_hold
      self.pulse_deferred_frames = RESUME_PULSE_DEFER_FRAMES if self.latched_release else 0
      self.just_released = True
    elif self.holding:
      # a re-hold cancels a pending pulse: the body is being asked to hold again, so there is
      # nothing to unlatch, and firing the bit over a hold-grade command is the exact tuple
      # the camera latches on (route 00000053)
      self.pulse_deferred_frames = 0

    if self.pulse_deferred_frames > 0 and not self.just_released:
      if not brake_hold:
        # the body let go off the relaxing command alone -- the common case, and the whole
        # point of deferring: no pulse reaches the wire and the camera has nothing to fault
        self.pulse_deferred_frames = 0
      else:
        self.pulse_deferred_frames -= 1
        if self.pulse_deferred_frames == 0:
          # the body is still holding the brakes after the grace period, so fall back to
          # stock's unlatch pulse: a car that will not move is worse than the SCBS latch,
          # and the logs then say plainly that the pulse is load-bearing after all
          self.unlatch_frames = RESUME_UNLATCH_LATCHED_FRAMES

    # the body only owns the brakes while we are still asking it to hold
    self.car_has_hold = self.holding and standstill and brake_hold

  @property
  def stop_bits(self) -> bool:
    # CRZ_INFO stop flags are held through the approach and the hold, and clear when the car
    # takes over and the command relaxes. A re-hold while a release pulse is still playing
    # waits the pulse out: stock never puts STOPPING and RESUME_UNLATCHING on the wire
    # together (its stop bits are already dropped when the pulse fires, every release)
    return self.holding and not self.car_has_hold and self.unlatch_frames == 0

  @property
  def resume_unlatching(self) -> bool:
    # only ever set by the deferred fallback above, and only for a latched release
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

  def update(self, lead_visible: bool, d_rel: float, v_rel: float, holding: bool) -> None:
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
    self.real_lead = self._measured if self.visible else None
    self.lead = self.real_lead
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
