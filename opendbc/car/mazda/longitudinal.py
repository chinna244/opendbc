"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
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

  The radar does not support COMMUNICATION_CONTROL (0x28 replies NRC 0x11), so disable_ecu()
  cannot be used. A programming session stops all of its periodic frames; it stays silent as
  long as tester present keeps arriving and falls back to the default session on its ~5 s S3
  timeout otherwise. The programming session disables AEB while in effect."""
  return CanData(RADAR_ADDR, bytes([0x02, uds.SERVICE_TYPE.DIAGNOSTIC_SESSION_CONTROL, session_type, 0x00, 0x00, 0x00, 0x00, 0x00]), RADAR_BUS)


class RadarSessionState(StrEnum):
  STOCK = "stock"          # radar broadcasting; nothing transmitted
  SILENCING = "silencing"  # requesting the programming session
  SILENCED = "silenced"    # radar quiet; tester present + synthetic frames
  HANDBACK = "handback"    # requesting the default session; synthetic frames continue


RADAR_SESSION_LIMIT_FRAMES = int(CarControllerParams.RADAR_SESSION_LIMIT_T / DT_CTRL)


class RadarSessionManager:
  """Sequences the radar in and out of its UDS programming session.

  Setup waits for the FSC camera's cold-boot radar-presence check (silencing too early latches
  an i-ACTIVSENSE fault); the verdict is invisible until first motion, so the gate is carstate's
  settle timer. The hand-back runs from the control loop because pandad blocks TX within ~100 ms
  of an onroad cycle starting, and the restart is requested once the stock radar is heard again.
  A refused or unanswered session request gives up for the drive and stock keeps the bus.
  See docs/zoompilot/mazda-longitudinal.md, "Radar takeover".
  """

  def __init__(self):
    self.state = RadarSessionState.STOCK
    self.state_frames = 0
    self.silencing_failed = False
    self.handback_completed = False

  def update(self, gate_passed: bool, stock_radar_alive: bool, handback: bool,
             standstill: bool, session_refused: bool, stock_radar_gone: bool) -> RadarSessionState:
    prev_state = self.state
    if handback:
      if self.state == RadarSessionState.SILENCING:
        # nothing was torn down yet; just stop touching the bus
        self.state = RadarSessionState.STOCK
      elif self.state == RadarSessionState.SILENCED:
        self.state = RadarSessionState.HANDBACK
      elif self.state == RadarSessionState.HANDBACK and \
           (stock_radar_alive or self.state_frames >= RADAR_SESSION_LIMIT_FRAMES):
        # heard again, or never coming back: stop waiting so the restart proceeds. The radar
        # stays stock for the rest of the process, in case the assert drops before exit and
        # would otherwise read as a withdrawal and re-silence it right before shutdown.
        self.state = RadarSessionState.STOCK
        self.handback_completed = True
    else:
      if self.state == RadarSessionState.HANDBACK:
        # hand-back withdrawn (toggle flipped back before the restart): the radar is
        # stock again, so re-run the normal takeover
        self.state = RadarSessionState.STOCK
      if self.state == RadarSessionState.STOCK and gate_passed and not self.handback_completed:
        # silencing disables AEB, so like every disable_ecu caller it only starts pre-motion;
        # adopting an already-quiet radar disables nothing and proceeds anywhere. "Quiet" is
        # the guard-long silence, not the alive window: a stock radar drops a few frames in a
        # row now and then, and adopting on one of those made us a second master.
        if stock_radar_gone:
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
        # The radar is broadcasting again under our synthetic frames (S3 recovery, or never
        # really silenced). Our frames stop either way; the session request is gated like the
        # first teardown since it disables AEB. Moving, stock keeps the bus (carstate raises
        # accFaulted) until the next stop.
        self.state = RadarSessionState.SILENCING if (standstill and not self.silencing_failed) else RadarSessionState.STOCK

    self.state_frames = 0 if self.state != prev_state else self.state_frames + 1
    return self.state


RESUME_UNLATCH_LATCHED_FRAMES = int(CarControllerParams.RESUME_UNLATCH_LATCHED_T / DT_CTRL)
RESUME_REPULSE_FRAMES = int(CarControllerParams.RESUME_REPULSE_T / DT_CTRL)
LEAD_DEBOUNCE_FRAMES = int(CarControllerParams.LEAD_DEBOUNCE_T / DT_CTRL)
RELEASE_DEBOUNCE_FRAMES = int(CarControllerParams.RELEASE_DEBOUNCE_T / DT_CTRL)
BREAKAWAY_FRAMES = int(CarControllerParams.ACCEL_BREAKAWAY_T / DT_CTRL)


class StandstillHold:
  """Holds the car stopped until the plan asks to move, the way Toyota and Honda do it: the
  standstill request comes straight off the plan and car feedback, with no timers in the path,
  and the hold command is the plan's own (LongControl parks at CP.stopAccel, the stock hold
  value). The relax off that hold is the car's decision: stock lets go the instant the body ECU
  latches its own brake hold (GEAR.BRAKE_HOLD). `holding` is recomputed every frame.
  See docs/zoompilot/mazda-longitudinal.md, "Stop-and-go".
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
    self.latched_frames = 0  # frames since a latched release with the body still holding
    self.repulsed = False

  def update(self, long_engaged: bool, stopping: bool, standstill: bool,
             plan_accel: float, brake_hold: bool, gas_pressed: bool) -> None:
    self.just_released = False
    if not long_engaged:
      self._reset()
      return

    was_holding = self.holding
    # the plan's request to move is debounced so a one-frame blip cannot fire a phantom
    # release pulse at a standstill; the driver's pedal outranks the hold immediately
    self.release_frames = self.release_frames + 1 if plan_accel > 0. else 0
    plan_wants_go = self.release_frames >= RELEASE_DEBOUNCE_FRAMES
    # the plan asking for acceleration releases the hold, and so does the driver's pedal, as
    # Toyota's PCM does. Holding the stop bits against the throttle until the car moved put
    # an out-of-protocol release on the bus; stock keeps STOPPING strictly to the final creep.
    release = gas_pressed or plan_wants_go
    self.holding = not release and (stopping or standstill)

    if self.unlatch_frames > 0:
      self.unlatch_frames -= 1
    # one pulse per release, as stock, never restarted while one is still playing. A gas-ended
    # hold gets no pulse: stock's captured gas-ended hold drops the stop bits with none.
    # See docs/zoompilot/mazda-longitudinal.md, "Gas-pedal release".
    if was_holding and not self.holding and standstill and not gas_pressed and self.unlatch_frames == 0:
      # car_has_hold still carries last frame's value: whether the body owned the brakes going
      # into this release is whether there is anything to unlatch. A latched release pulses
      # immediately; the body answers nothing else.
      self.latched_release = self.car_has_hold
      if self.latched_release:
        self.unlatch_frames = RESUME_UNLATCH_LATCHED_FRAMES
      self.just_released = True
      self.latched_frames = 0
      self.repulsed = False

    # A latched release the body did not answer (GEAR.BRAKE_HOLD still set, release standing):
    # the carcontroller keeps the command pinned at the relaxed hold while the body holds, so
    # without this the car would sit under a positive plan until the driver's pedal. One
    # retry, the same tuple as the first pulse, then give up.
    if self.latched_release and not self.holding and standstill and brake_hold and not gas_pressed:
      self.latched_frames += 1
      if self.latched_frames >= RESUME_REPULSE_FRAMES and not self.repulsed and self.unlatch_frames == 0:
        self.unlatch_frames = RESUME_UNLATCH_LATCHED_FRAMES
        self.repulsed = True
    else:
      self.latched_frames = 0

    # the body only owns the brakes while we are still asking it to hold
    self.car_has_hold = self.holding and standstill and brake_hold

  @property
  def stop_bits(self) -> bool:
    # CRZ_INFO stop flags are held through the approach and the hold, and clear when the body
    # takes over. A re-hold while a pulse is still playing waits it out: stock never puts
    # STOPPING and RESUME_UNLATCHING on the wire together.
    return self.holding and not self.car_has_hold and self.unlatch_frames == 0

  @property
  def resume_unlatching(self) -> bool:
    # only ever armed at a latched release
    return self.unlatch_frames > 0

  @property
  def acc_active_2(self) -> bool:
    # stock drops ACC_ACTIVE_2 together with the command relax
    return not self.car_has_hold


class AdvertisedLead:
  """The lead we tell the camera about: CRZ_CTRL's two lead fields and the 0x364 track slot.

  Stock pairs all three absolutely, so they are read off one piece of state. The state is
  perception, not control: a stock radar reports its objects engaged or not, and tying the
  advertisement to engagement made a real lead vanish from the bus mid-creep and the camera
  run its collision display. Visibility is debounced (a marginal vision lead flickers faster
  than any radar track) and the last real measurement is coasted across a vision gap, the way
  a radar coasts a track. See docs/zoompilot/mazda-longitudinal.md, "Advertised lead".
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
      # expiring here bounds the coast to the debounce window and keeps a stale measurement
      # from resurfacing on reacquisition
      self._measured = None
    elif self._measured is not None:
      # propagate through the gap rather than repeating one frozen frame (create_lead_track)
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
    # RADAR_LEAD_RELATIVE_DISTANCE is stock's 1-5 closeness bucket: 2 following, 3 near a
    # hold (stock's dominant standstill value); faults key on the triple disagreeing, not on
    # the bucket value
    if not self.has_lead:
      return 0
    return 3 if self.holding else 2
