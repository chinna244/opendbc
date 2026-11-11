"""The Smart CAN Filter: an inline board between the forward radar and the car that keeps the radar
online under alpha long, in place of the UDS takeover. Protocol from the bench-tested
smart-can-filter firmware (README.md, DESIGN.md):

  0x2A0 FILTER_CTRL, car side: ce fa ad de 2e <ctrl> 00 00. ctrl bit 0 MODE holds op-long mode
    (5 s timeout) and idles the radar's CRZ_BTNS so it can never arm; bit 1 FILTER drops the radar's
    CRZ_INFO/CRZ_CTRL toward the car (250 ms timeout, refreshed by our own 0x21b/0x21c). Mode entry
    is refused while the radar's own cruise is engaged, and MODE=0 restores stock at once. Every
    FILTER_CTRL is acked with an immediate status frame.
  0x2A8 status, 5 Hz (10 Hz for 5 s after boot): byte 7 bit 5 FILTERING, bit 6 MODE, bit 7
    RADAR_ENGAGED. A board that resets boots in passthrough, forwarding the radar's frames at once.

MODE and FILTER go out together. Ownership needs an ack showing both and then the radar's 50 Hz
CRZ_INFO/CRZ_CTRL gone from the car side, so the PCM sees a gap of a few frames rather than two
masters. openpilot then stays the cruise master for the whole mode, standby frames included, as under
the UDS takeover. Ownership ends on the first sign the board stopped filtering: a status without
MODE or FILTERING (every one in every CAN read), the radar's cruise frames heard on the car side
again, or no status at all for STATUS_FRESH_FRAMES; each latches on the CAN read that shows it. The board's bootloader,
softloader, reset and feature commands are never sent.
"""
from opendbc.car import DT_CTRL
from opendbc.car.can_definitions import CanData
from opendbc.car.carlog import carlog
from opendbc.sunnypilot.car.stock_ecu import StockEcuState

FILTER_CTRL_ADDR = 0x2A0
FILTER_STATE_ADDR = 0x2A8
FILTER_BUS = 0
# The radar's cruise frames, which reach the car side only while the board forwards them.
FACTORY_CRUISE_ADDRS = (0x21B, 0x21C)
FILTER_CTRL_PREFIX = b"\xce\xfa\xad\xde\x2e"
CTRL_MODE = 1
CTRL_FILTER = 2
STATE_FILTERING = 1 << 5
STATE_MODE = 1 << 6
STATE_CONFIRMED = STATE_MODE | STATE_FILTERING
# A receive-batch event that is not a status byte: one of the radar's cruise frames on the car side.
FACTORY_CRUISE = -1
# Keepalive (and mode-off) at 2 Hz against the board's 5 s mode timeout.
FILTER_CTRL_STEP = int(0.5 / DT_CTRL)
# The beacon runs at 5 Hz: a status older than 2.5 periods is a board gone quiet.
STATUS_FRESH_FRAMES = int(0.5 / DT_CTRL)
# Ownership waits this long past the ack with no radar cruise frame heard, two of their 20 ms periods.
CONFIRM_SILENCE_FRAMES = int(0.04 / DT_CTRL)
# The board's own mode timeout: a mode-off unanswered this long is a failed hand-back.
RELEASE_LIMIT_FRAMES = int(5.0 / DT_CTRL)
# A request unconfirmed this long is given up for the drive, as the UDS takeover's attempt budget: MODE idles
# the radar's buttons, so holding it unowned would leave the driver no cruise at all.
CONFIRM_LIMIT_FRAMES = int(10.0 / DT_CTRL)


def create_filter_ctrl(ctrl: int) -> CanData:
  return CanData(FILTER_CTRL_ADDR, FILTER_CTRL_PREFIX + bytes([ctrl, 0, 0]), FILTER_BUS)


class CanFilterSession:
  """Requests the board's mode while the car's own cruise is off, and owns cruise (control_active)
  only once the board confirmed it and the radar's frames are gone. Losing ownership latches `failed`
  for the drive: openpilot's cruise frames stop, a MODE=0 restores the stock radar at once, and the UDS
  takeover stays off. The lifecycle's hand-back takes the same release path; the latest status heard
  after the MODE=0 showing neither bit answers it (stock_ecu_state RESTORED).

  CarInterface.update hands every CAN read and the status age after it to receive(), so a loss (the
  status timeout included) latches on that read even when no control frame runs in between; the controller runs update(), and carstate reads
  control_active through the same object."""

  def __init__(self):
    self.request_frames = -1  # control frames since the request (or the mode-off) went out, -1 before it
    self.acked_frames = -1    # control frames since the board's ack with the radar's frames gone, -1 without one
    self.control_active = False
    self.failed = False       # ownership lost: no cruise output and no new request this drive
    self.handback = False     # the lifecycle's hand-back request stands
    self.release_frames = 0
    self.released = False     # the latest status after the mode-off showed the board back in passthrough
    self.mode_sent = False    # MODE went out this drive: the board may be in its mode, acked or not
    self.last_state: int | None = None

  @property
  def releasing(self) -> bool:
    return self.failed or self.handback

  @property
  def lost(self) -> bool:
    """Ownership lost or never confirmed: the drive is without the board, unlike a hand-back."""
    return self.failed

  @property
  def stock_ecu_state(self) -> StockEcuState:
    if not self.mode_sent:
      return StockEcuState.NOT_NEEDED
    if self.releasing:
      if self.released:
        return StockEcuState.RESTORED
      if self.release_frames < RELEASE_LIMIT_FRAMES:
        return StockEcuState.RESTORING
      # no status all drive: no board answered, and its mode timeout has run out past the last MODE anyway
      return StockEcuState.FAILED if self.last_state is not None else StockEcuState.NOT_NEEDED
    # the board may be rewriting the radar's buttons: ours, or not yet
    return StockEcuState.READY if self.control_active else StockEcuState.STARTING

  def _release(self) -> None:
    self.control_active = False
    self.acked_frames = -1
    self.request_frames = -1  # the mode-off goes out on the next control frame
    self.release_frames = 0
    self.released = False

  def _lose(self, reason: str, state: int | None) -> None:
    carlog.error({"event": "mazdaCanFilterDropout", "reason": reason, "state": state})
    self.failed = True
    self._release()

  def receive(self, events: list[int], state_age: int) -> None:
    """One CAN read's filter events in order, each a 0x2A8 status byte or FACTORY_CRUISE, and the status
    age in reads after it."""
    for event in events:
      status = event != FACTORY_CRUISE
      if status:
        self.last_state = event
      if self.control_active:
        if not status:
          self._lose("radar cruise frames returned", self.last_state)
        elif (event & STATE_CONFIRMED) != STATE_CONFIRMED:
          self._lose("status lost mode or filtering", event)
      elif self.releasing:
        # only a status heard after the mode-off went out answers it, and the latest one counts
        if status and self.request_frames >= 0:
          self.released = not event & STATE_CONFIRMED
      elif self.request_frames >= 0:
        # only a status heard after the request went out confirms it
        if status and (event & STATE_CONFIRMED) == STATE_CONFIRMED:
          self.acked_frames = max(self.acked_frames, 0)
        else:
          self.acked_frames = -1
    if self.control_active and state_age >= STATUS_FRESH_FRAMES:
      self._lose("status stale", self.last_state)

  def update(self, state_age: int, cruise_engaged: bool, handback: bool = False) -> list[CanData]:
    if handback and not self.handback:
      carlog.info({"event": "mazdaCanFilterHandback", "controlActive": self.control_active})
      self.handback = True
      if not self.failed:
        self._release()
    elif not handback and self.handback:
      # a withdrawn hand-back is a fresh start, unless the drive already lost the board
      self.handback = False
      if not self.failed:
        self.released = False
        self.request_frames = -1

    if self.releasing:
      ctrl = 0
      self.release_frames += 1
    elif self.control_active:
      ctrl = CTRL_MODE | CTRL_FILTER
    elif cruise_engaged:
      # the car's own cruise: never request the mode over it (the board refuses it too)
      self.request_frames = -1
      self.acked_frames = -1
      return []
    elif self.request_frames >= CONFIRM_LIMIT_FRAMES:
      self._lose("ownership not confirmed", self.last_state)
      ctrl = 0
    else:
      if self.acked_frames >= 0:
        self.acked_frames += 1
        if self.acked_frames >= CONFIRM_SILENCE_FRAMES and state_age < STATUS_FRESH_FRAMES:
          self.control_active = True
          carlog.info({"event": "mazdaCanFilterConfirmed", "requestFrames": self.request_frames})
      ctrl = CTRL_MODE | CTRL_FILTER

    self.request_frames += 1
    if self.request_frames % FILTER_CTRL_STEP != 0:
      return []
    self.mode_sent |= bool(ctrl & CTRL_MODE)
    return [create_filter_ctrl(ctrl)]
