"""
The Smart CAN Filter under alpha long: the opt-in, the handoff to openpilot's cruise frames once
the board confirms its mode and the radar's frames are gone, the car's own cruise already on, a board
dropout (status, batch, reset, link), the lifecycle's hand-back, and the UDS path left as it was with
the option off.
"""
import pytest

from opendbc.car import Bus, DT_CTRL
from opendbc.car.mazda import mazdacan
from opendbc.car.mazda.can_filter import CONFIRM_SILENCE_FRAMES, CTRL_FILTER, CTRL_MODE, FACTORY_CRUISE, FILTER_CTRL_ADDR, \
  FILTER_CTRL_STEP, FILTER_STATE_ADDR, RELEASE_LIMIT_FRAMES, STATE_FILTERING, STATE_MODE, STATUS_FRESH_FRAMES, create_filter_ctrl
from opendbc.car.mazda.carcontroller import CarController
from opendbc.car.mazda.interface import CarInterface
from opendbc.car.mazda.tests.conftest import CRZ_CTRL, CRZ_INFO, DBC_NAME, RADAR_STATIC, RADAR_UDS, SESSION_PROG_DAT, \
  car_params, car_params_sp, frames, mazda_car_state, packer, step_long
from opendbc.car.mazda.tests.test_mazda_carstate import feed
from opendbc.car.mazda.values import CAR, MazdaFlags
from opendbc.sunnypilot.car.interfaces import _initialize_mazda
from opendbc.sunnypilot.car.mazda.values import MazdaFlagsSP, MazdaSafetyFlagsSP
from opendbc.sunnypilot.car.stock_ecu import StockEcuState

CONFIRMED = STATE_MODE | STATE_FILTERING
REQUEST = bytes([0xce, 0xfa, 0xad, 0xde, 0x2e, CTRL_MODE | CTRL_FILTER, 0, 0])
MODE_OFF = bytes([0xce, 0xfa, 0xad, 0xde, 0x2e, 0, 0, 0])
LEGACY_RADAR_ADDRS = (RADAR_UDS, RADAR_STATIC, *range(0x361, 0x367))


def filter_params(enabled=True, alpha_long=True, candidate=CAR.MAZDA_CX5_2022):
  CP = car_params(candidate, alpha_long=alpha_long)
  CP_SP = car_params_sp(CP, candidate, alpha_long=alpha_long)
  _initialize_mazda(CP, CP_SP, {"MazdaCanFilter": "1" if enabled else "0"})
  return CP, CP_SP


def filter_controller():
  CP, CP_SP = filter_params()
  return CarController({Bus.pt: DBC_NAME}, CP, CP_SP)


def run(cc, cs, n=1, state=None, age=0, events=None, **kwargs):
  """n controller frames with the board's status byte (None: never heard) at the given age. The batch
  carries that status when age is 0, unless `events` gives the batch explicitly."""
  sends = []
  for _ in range(n):
    cs.can_filter_state, cs.can_filter_age = state, (age if state is not None else STATUS_FRESH_FRAMES)
    cc.can_filter.receive(events if events is not None else ([state] if state is not None and age == 0 else []), cs.can_filter_age)
    sends += step_long(cc, cs, **kwargs)
  return sends


def cruise_frames(sends):
  return [s for s in sends if s[0] in (CRZ_INFO, CRZ_CTRL)]


def crz_active(dat) -> bool:
  return bool(dat[0] & 0x08)  # CRZ_CTRL.CRZ_ACTIVE


def confirm(cc, cs):
  """Request on one frame, the board's ack on the next, then the radar's frames stay gone."""
  run(cc, cs)
  run(cc, cs, state=CONFIRMED)
  for age in range(1, CONFIRM_SILENCE_FRAMES):
    assert not cc.can_filter.control_active
    run(cc, cs, state=CONFIRMED, age=age)
  assert cc.can_filter.control_active


@pytest.fixture
def fcc():
  return filter_controller()


@pytest.fixture
def fcs(fcc):
  return mazda_car_state(fcc.CP, fcc.CP_SP)


class TestOptIn:
  def test_cx5_2022_alpha_long_opts_in(self):
    CP, CP_SP = filter_params()
    assert CP_SP.flags & MazdaFlagsSP.CAN_FILTER
    assert CP_SP.safetyParam & MazdaSafetyFlagsSP.CAN_FILTER
    assert not CP.radarUnavailable

  def test_g46l_radar_stays_vision_only(self):
    CP = car_params(CAR.MAZDA_CX5_2022, alpha_long=True)
    CP.flags |= MazdaFlags.G46L_RADAR.value
    CP_SP = car_params_sp(CP, CAR.MAZDA_CX5_2022, alpha_long=True)
    _initialize_mazda(CP, CP_SP, {"MazdaCanFilter": "1"})
    assert CP_SP.flags & MazdaFlagsSP.CAN_FILTER
    assert CP.radarUnavailable

  @pytest.mark.parametrize("enabled, alpha_long, candidate", [
    (False, True, CAR.MAZDA_CX5_2022),
    (True, False, CAR.MAZDA_CX5_2022),
    (True, True, CAR.MAZDA_CX9_2021),
  ])
  def test_off_everywhere_else(self, enabled, alpha_long, candidate):
    CP, CP_SP = filter_params(enabled, alpha_long, candidate)
    assert not CP_SP.flags & MazdaFlagsSP.CAN_FILTER
    assert not CP_SP.safetyParam & MazdaSafetyFlagsSP.CAN_FILTER
    if CP.openpilotLongitudinalControl:
      assert CP.radarUnavailable  # alpha long without the filter tears the radar down
    assert CarController({Bus.pt: DBC_NAME}, CP, CP_SP).can_filter is None


class TestOptionOff:
  def test_uds_takeover_unchanged_and_no_filter_traffic(self, cc, cs):
    assert cc.can_filter is None
    sends = []
    for _ in range(3 * FILTER_CTRL_STEP):
      sends += step_long(cc, cs, stock_radar_alive=True, standstill=True, long_active=False)
    assert SESSION_PROG_DAT in frames(sends, RADAR_UDS)
    assert FILTER_CTRL_ADDR not in {a for a, _, _ in sends}


class TestHandoff:
  def test_request_goes_out_at_once_and_no_cruise_frames_before_the_ack(self, fcc, fcs):
    sends = run(fcc, fcs, n=FILTER_CTRL_STEP + 1)
    assert frames(sends, FILTER_CTRL_ADDR) == [REQUEST, REQUEST]
    assert cruise_frames(sends) == []
    assert not any(s[0] in LEGACY_RADAR_ADDRS for s in sends)

  @pytest.mark.parametrize("state", [STATE_MODE, STATE_FILTERING, 0])
  def test_partial_status_does_not_confirm(self, fcc, fcs, state):
    run(fcc, fcs)
    assert cruise_frames(run(fcc, fcs, n=10, state=state)) == []
    assert not fcc.can_filter.control_active

  def test_a_status_heard_before_the_request_does_not_confirm(self, fcc, fcs):
    # the request frame's status may predate the request
    run(fcc, fcs, state=CONFIRMED)
    assert cruise_frames(run(fcc, fcs, n=CONFIRM_SILENCE_FRAMES + 5, state=CONFIRMED, age=1)) == []
    assert not fcc.can_filter.control_active

  def test_ownership_waits_for_the_radars_frames_to_stay_gone(self, fcc, fcs):
    run(fcc, fcs)
    run(fcc, fcs, state=CONFIRMED)
    # a radar frame forwarded after the ack: the board is not filtering yet, nothing goes out
    sends = run(fcc, fcs, n=CONFIRM_SILENCE_FRAMES + 5, state=CONFIRMED, age=1, events=[FACTORY_CRUISE])
    assert cruise_frames(sends) == [] and not fcc.can_filter.control_active
    assert not fcc.can_filter.failed
    # the next ack with the radar quiet confirms
    run(fcc, fcs, state=CONFIRMED)
    sends = run(fcc, fcs, n=CONFIRM_SILENCE_FRAMES - 1, state=CONFIRMED, age=1)
    assert fcc.can_filter.control_active and cruise_frames(sends)

  def test_a_stale_status_does_not_confirm(self, fcc, fcs):
    run(fcc, fcs)
    assert cruise_frames(run(fcc, fcs, n=10, state=CONFIRMED, age=1)) == []

  def test_cruise_frames_follow_the_ack_on_both_buses(self, fcc, fcs):
    confirm(fcc, fcs)
    sends = run(fcc, fcs, n=4, state=CONFIRMED, age=1)
    for bus in (0, 2):
      assert len(frames(sends, CRZ_INFO, bus)) == 2 and len(frames(sends, CRZ_CTRL, bus)) == 2
    # engaged once ours: the ACC bits follow CC.enabled
    assert crz_active(frames(sends, CRZ_CTRL)[-1])

  def test_keepalive_holds_the_request_and_legacy_traffic_never_goes_out(self, fcc, fcs):
    confirm(fcc, fcs)
    sends = run(fcc, fcs, n=4 * FILTER_CTRL_STEP, state=CONFIRMED, age=1)
    assert frames(sends, FILTER_CTRL_ADDR) == [REQUEST] * 4
    assert not any(s[0] in LEGACY_RADAR_ADDRS for s in sends)

  def test_the_driver_view(self, fcc, fcs):
    # nothing sent: nothing to hand back
    run(fcc, fcs, n=5, cruise_engaged=True)
    assert fcc.stock_ecu_state == StockEcuState.NOT_NEEDED
    # MODE sent: the board may be in its mode before any ack, rewriting the buttons, not ours yet
    run(fcc, fcs)
    assert fcc.stock_ecu_state == StockEcuState.STARTING
    run(fcc, fcs, n=5, state=STATE_FILTERING)
    assert fcc.stock_ecu_state == StockEcuState.STARTING
    run(fcc, fcs, state=CONFIRMED)
    run(fcc, fcs, n=CONFIRM_SILENCE_FRAMES, state=CONFIRMED, age=1)
    assert fcc.stock_ecu_state == StockEcuState.READY
    # the radar stays online throughout: AEB never counts as off
    assert not fcc.stock_ecu_off
    assert not fcc.alpha_long_requested


class TestFactoryCruiseActive:
  def test_no_request_while_the_cars_cruise_is_on(self, fcc, fcs):
    sends = run(fcc, fcs, n=3 * FILTER_CTRL_STEP, state=CONFIRMED, cruise_engaged=True)
    assert frames(sends, FILTER_CTRL_ADDR) == []
    assert cruise_frames(sends) == []
    assert not fcc.can_filter.control_active

  def test_request_starts_when_it_goes_off(self, fcc, fcs):
    run(fcc, fcs, n=20, cruise_engaged=True)
    assert frames(run(fcc, fcs), FILTER_CTRL_ADDR) == [REQUEST]

  def test_an_unconfirmed_request_is_withdrawn(self, fcc, fcs):
    run(fcc, fcs, n=5)
    sends = run(fcc, fcs, n=2 * FILTER_CTRL_STEP, state=CONFIRMED, cruise_engaged=True)
    assert frames(sends, FILTER_CTRL_ADDR) == []
    assert not fcc.can_filter.control_active


class TestDropout:
  @pytest.mark.parametrize("state, age", [
    (CONFIRMED, STATUS_FRESH_FRAMES),  # the board went quiet
    (STATE_FILTERING, 0),              # mode lost
    (STATE_MODE, 0),                   # filtering lost
  ])
  def test_cruise_frames_stop_for_the_drive(self, fcc, fcs, state, age):
    confirm(fcc, fcs)
    sends = run(fcc, fcs, state=state, age=age)
    assert cruise_frames(sends) == []
    assert frames(sends, FILTER_CTRL_ADDR) == [MODE_OFF]
    assert fcc.can_filter.failed and not fcc.can_filter.control_active
    # a board that comes back confirming changes nothing for this drive: stock radar, mode off,
    # and no UDS takeover in its place
    sends = run(fcc, fcs, n=4 * FILTER_CTRL_STEP, state=CONFIRMED)
    assert cruise_frames(sends) == []
    assert frames(sends, FILTER_CTRL_ADDR) == [MODE_OFF] * 4
    assert not any(s[0] in LEGACY_RADAR_ADDRS for s in sends)
    assert not fcc.can_filter.control_active

  @pytest.mark.parametrize("batch", [
    [0, CONFIRMED],                # off, then on again later in the same receive batch
    [STATE_MODE, CONFIRMED],
    [CONFIRMED, STATE_FILTERING, CONFIRMED],
  ])
  def test_any_status_in_the_batch_without_both_bits_latches(self, fcc, fcs, batch):
    confirm(fcc, fcs)
    sends = run(fcc, fcs, state=CONFIRMED, events=batch)
    assert cruise_frames(sends) == [] and frames(sends, FILTER_CTRL_ADDR) == [MODE_OFF]
    assert fcc.can_filter.failed

  def test_radar_frames_returning_under_a_fresh_status_latch(self, fcc, fcs):
    # the cached status still says filtering; the radar's own CRZ frames on the car side say otherwise
    confirm(fcc, fcs)
    sends = run(fcc, fcs, state=CONFIRMED, age=1, events=[FACTORY_CRUISE])
    assert cruise_frames(sends) == [] and frames(sends, FILTER_CTRL_ADDR) == [MODE_OFF]
    assert fcc.can_filter.failed

  @pytest.mark.parametrize("quiet_frames", [2, 5, 15])
  def test_board_reset_between_statuses(self, fcc, fcs, quiet_frames):
    # A reset boots in passthrough: the radar's 50 Hz frames reach the car side before the 10 Hz boot
    # beacon. Output stops on the frame the first of them is heard, whatever the cached status says.
    confirm(fcc, fcs)
    sends = []
    for age in range(1, quiet_frames + 1):
      sends += run(fcc, fcs, state=CONFIRMED, age=age)  # rebooting: nothing on the car side from it
    assert fcc.can_filter.control_active and cruise_frames(sends)
    sends = run(fcc, fcs, state=CONFIRMED, age=quiet_frames + 1, events=[FACTORY_CRUISE, FACTORY_CRUISE])
    assert cruise_frames(sends) == [] and fcc.can_filter.failed

  def test_interrupted_link_is_caught_only_by_the_stale_status(self, fcc, fcs):
    # Nothing at all from the board's side, radar frames included: the host cannot see the board
    # release, so output holds until the status is STATUS_FRESH_FRAMES old (bench check).
    confirm(fcc, fcs)
    held = 0
    for age in range(1, STATUS_FRESH_FRAMES + 1):
      if cruise_frames(run(fcc, fcs, state=CONFIRMED, age=age)):
        held = age
    assert fcc.can_filter.failed and held == STATUS_FRESH_FRAMES - 2  # last frame on the 50 Hz cadence

  def test_status_ageing_inside_the_window_holds(self, fcc, fcs):
    confirm(fcc, fcs)
    sends = run(fcc, fcs, state=CONFIRMED, age=STATUS_FRESH_FRAMES - 1)
    assert fcc.can_filter.control_active and frames(sends, FILTER_CTRL_ADDR) == []


class TestCarState:
  """Through the real CarInterface and parsers: the status latch, ownership, and a dropout."""

  @staticmethod
  def interface():
    CP, CP_SP = filter_params()
    CI = CarInterface(CP, CP_SP)
    assert CI.CS.can_filter is CI.CC.can_filter
    return CI

  @staticmethod
  def step(CI, i, state=None, acc_active=False, radar=False):
    pk = packer()
    msgs = [pk.make_can_msg("PEDALS", 0, {"ACC_OFF": 0 if acc_active else 1, "ACC_ACTIVE": 1 if acc_active else 0}),
            pk.make_can_msg("ENGINE_DATA", 0, {"SPEED": 0})]
    if state is not None:
      msgs.append((FILTER_STATE_ADDR, bytes([0, 0, 0, 0, 0, 0, 0, state]), 0))
    if radar:
      msgs.append(mazdacan.create_acc_command(pk, 0, i, 0., long_active=False, acc_available=True))
    ret, _ = feed(CI, i, *msgs)
    return ret

  def test_status_latch_and_age(self):
    CI = self.interface()
    assert CI.CS.can_filter_state is None
    self.step(CI, 0, state=CONFIRMED)
    assert (CI.CS.can_filter_state, CI.CS.can_filter_age) == (CONFIRMED, 0)
    self.step(CI, 1)
    assert (CI.CS.can_filter_state, CI.CS.can_filter_age) == (CONFIRMED, 1)
    # the board's frame on the camera bus is not the board
    feed(CI, 2, (FILTER_STATE_ADDR, bytes(8), 2))
    assert (CI.CS.can_filter_state, CI.CS.can_filter_age) == (CONFIRMED, 2)

  def test_engagement_only_while_the_board_confirms(self):
    CI = self.interface()
    # radar online, no mode: the car's cruise is not openpilot's
    for i in range(int(2. / DT_CTRL)):
      ret = self.step(CI, i, radar=True, acc_active=i > 100)
    assert not ret.cruiseState.enabled and not ret.accFaulted
    # mode confirmed: an engagement after idle is adopted at once, no UDS silence guard
    CI.CC.can_filter.control_active = True
    self.step(CI, 300, state=CONFIRMED)
    ret = self.step(CI, 301, state=CONFIRMED, acc_active=True)
    assert ret.cruiseState.enabled and not ret.accFaulted
    # dropout: cruise drops, the radar returns, and no fault takes MADS lateral with it
    CI.CC.can_filter.control_active, CI.CC.can_filter.failed = False, True
    for i in range(302, 352):
      ret = self.step(CI, i, acc_active=True, radar=True)
      assert not ret.cruiseState.enabled and not ret.accFaulted
    assert ret.cruiseState.available


class TestHandback:
  """The lifecycle's hand-back before onroad ends: cruise output stops, MODE=0 goes out, and only a
  status heard after it with neither bit answers RESTORED."""

  def test_successful_handback(self, fcc, fcs):
    confirm(fcc, fcs)
    # a release status heard before the mode-off went out answers nothing
    sends = run(fcc, fcs, state=CONFIRMED, handback=True, long_active=False)
    assert cruise_frames(sends) == [] and frames(sends, FILTER_CTRL_ADDR) == [MODE_OFF]
    assert not fcc.can_filter.control_active and fcc.stock_ecu_state == StockEcuState.RESTORING
    assert not fcc.can_filter.failed
    sends = run(fcc, fcs, state=0, handback=True, long_active=False)
    assert fcc.stock_ecu_state == StockEcuState.RESTORED and cruise_frames(sends) == []
    # held stock for as long as the request stands
    sends = run(fcc, fcs, n=2 * FILTER_CTRL_STEP, state=0, handback=True, long_active=False)
    assert cruise_frames(sends) == [] and frames(sends, FILTER_CTRL_ADDR) == [MODE_OFF] * 2
    assert not any(s[0] in LEGACY_RADAR_ADDRS for s in sends)

  @pytest.mark.parametrize("state", [CONFIRMED, None])
  def test_board_that_never_acknowledges(self, fcc, fcs, state):
    confirm(fcc, fcs)
    sends = run(fcc, fcs, n=RELEASE_LIMIT_FRAMES - 1, state=state, handback=True, long_active=False)
    assert fcc.stock_ecu_state == StockEcuState.RESTORING
    sends += run(fcc, fcs, state=state, handback=True, long_active=False)
    assert fcc.stock_ecu_state == StockEcuState.FAILED
    assert cruise_frames(sends) == []
    assert set(frames(sends, FILTER_CTRL_ADDR)) == {MODE_OFF}
    assert not any(s[0] in LEGACY_RADAR_ADDRS for s in sends)
    # a late release still completes it
    run(fcc, fcs, state=0, handback=True, long_active=False)
    assert fcc.stock_ecu_state == StockEcuState.RESTORED

  def test_nothing_sent_has_nothing_to_hand_back(self, fcc, fcs):
    run(fcc, fcs, n=10, cruise_engaged=True)
    assert fcc.stock_ecu_state == StockEcuState.NOT_NEEDED

  def test_shutdown_before_the_first_ack(self, fcc, fcs):
    # the request went out, the board's ack has not arrived: it may already be in its mode
    sends = run(fcc, fcs)
    assert frames(sends, FILTER_CTRL_ADDR) == [REQUEST]
    assert fcc.stock_ecu_state == StockEcuState.STARTING
    sends = run(fcc, fcs, handback=True, long_active=False)
    assert frames(sends, FILTER_CTRL_ADDR) == [MODE_OFF] and cruise_frames(sends) == []
    assert fcc.stock_ecu_state == StockEcuState.RESTORING
    # the ack to the request arrives late: it answers nothing
    run(fcc, fcs, state=CONFIRMED, handback=True, long_active=False)
    assert fcc.stock_ecu_state == StockEcuState.RESTORING
    run(fcc, fcs, state=0, handback=True, long_active=False)
    assert fcc.stock_ecu_state == StockEcuState.RESTORED

  @pytest.mark.parametrize("on", [CONFIRMED, STATE_MODE, STATE_FILTERING])
  def test_off_then_on_in_one_release_batch_keeps_it_pending(self, fcc, fcs, on):
    confirm(fcc, fcs)
    run(fcc, fcs, state=CONFIRMED, age=1, handback=True, long_active=False)
    run(fcc, fcs, state=on, events=[0, on], handback=True, long_active=False)
    assert fcc.stock_ecu_state == StockEcuState.RESTORING
    # still the existing timeout without a clear status
    run(fcc, fcs, n=RELEASE_LIMIT_FRAMES, state=on, age=1, handback=True, long_active=False)
    assert fcc.stock_ecu_state == StockEcuState.FAILED
    run(fcc, fcs, state=0, handback=True, long_active=False)
    assert fcc.stock_ecu_state == StockEcuState.RESTORED

  def test_withdrawn_handback_requests_again(self, fcc, fcs):
    confirm(fcc, fcs)
    run(fcc, fcs, state=CONFIRMED, age=1, handback=True, long_active=False)
    run(fcc, fcs, state=0, handback=True, long_active=False)
    assert fcc.stock_ecu_state == StockEcuState.RESTORED and not fcc.can_filter.failed
    sends = run(fcc, fcs, state=0, age=1)
    assert frames(sends, FILTER_CTRL_ADDR) == [REQUEST] and not fcc.can_filter.failed
    confirm_again = run(fcc, fcs, state=CONFIRMED) + run(fcc, fcs, n=CONFIRM_SILENCE_FRAMES, state=CONFIRMED, age=1)
    assert fcc.can_filter.control_active and cruise_frames(confirm_again)

  def test_handback_after_a_dropout(self, fcc, fcs):
    confirm(fcc, fcs)
    run(fcc, fcs, state=STATE_MODE)
    run(fcc, fcs, state=0)
    assert fcc.stock_ecu_state == StockEcuState.RESTORED
    run(fcc, fcs, state=0, age=1, handback=True, long_active=False)
    assert fcc.stock_ecu_state == StockEcuState.RESTORED
    # withdrawn, the drive stays without the board
    sends = run(fcc, fcs, n=FILTER_CTRL_STEP, state=CONFIRMED)
    assert cruise_frames(sends) == [] and frames(sends, FILTER_CTRL_ADDR) == [MODE_OFF]


def owned_interface():
  CI = TestCarState.interface()
  session = CI.CC.can_filter
  session.request_frames, session.mode_sent, session.control_active = 0, True, True
  CI.CS.can_filter_state, CI.CS.can_filter_age = CONFIRMED, 0  # owned on a status just heard
  return CI, session


def status_frame(state):
  return FILTER_STATE_ADDR, bytes([0, 0, 0, 0, 0, 0, 0, state]), 0


def test_interface_keeps_every_status_in_a_batch():
  """Off then on inside one receive batch, through the real CarInterface: the off is not lost."""
  CI, session = owned_interface()
  feed(CI, 0, status_frame(0), status_frame(CONFIRMED))
  assert (CI.CS.can_filter_state, CI.CS.can_filter_age) == (CONFIRMED, 0)
  assert session.failed and not session.control_active
  assert session.update(CI.CS.can_filter_age, False) == [create_filter_ctrl(0)]


@pytest.mark.parametrize("loss", [status_frame(0), status_frame(STATE_MODE), (0x21b, bytes(8), 0), (0x21c, bytes(8), 0)])
def test_loss_latches_across_reads_without_a_control_frame(loss):
  """A loss in one CAN read, a confirming status in the next, and no CI.apply between them (card skips it
  while carControl is not alive): the next control frame still stops output and sends MODE off."""
  CI, session = owned_interface()
  feed(CI, 0, loss)
  feed(CI, 1, status_frame(CONFIRMED))
  feed(CI, 2, status_frame(CONFIRMED))
  assert session.failed and not session.control_active and not CI.CS.radar_owned
  assert session.update(CI.CS.can_filter_age, False) == [create_filter_ctrl(0)]
  # the drive-long lockout: a confirming board changes nothing
  for i in range(3, 3 + 2 * FILTER_CTRL_STEP):
    feed(CI, i, status_frame(CONFIRMED))
    assert set(session.update(CI.CS.can_filter_age, False)) <= {create_filter_ctrl(0)}
  assert session.failed and not session.control_active


def test_factory_frames_on_the_camera_bus_are_not_a_loss():
  CI, session = owned_interface()
  feed(CI, 0, (0x21b, bytes(8), 2), status_frame(CONFIRMED))
  assert session.control_active and not session.failed


def test_status_timeout_latches_on_receive_without_a_control_frame():
  """Reads with no status cross the 500 ms timeout while no CI.apply runs, a confirming status follows on
  the next read, and only then a control frame: the dropout holds, output stops and MODE off goes out."""
  CI, session = owned_interface()
  for i in range(STATUS_FRESH_FRAMES - 1):
    feed(CI, i)
  assert session.control_active
  feed(CI, STATUS_FRESH_FRAMES)
  assert session.failed and not session.control_active
  feed(CI, STATUS_FRESH_FRAMES + 1, status_frame(CONFIRMED))
  assert session.failed and not session.control_active
  sends = step_long(CI.CC, CI.CS)
  assert cruise_frames(sends) == [] and frames(sends, FILTER_CTRL_ADDR) == [MODE_OFF]
  # the drive-long lockout
  for i in range(2 * FILTER_CTRL_STEP):
    feed(CI, STATUS_FRESH_FRAMES + 2 + i, status_frame(CONFIRMED))
    sends = step_long(CI.CC, CI.CS)
    assert cruise_frames(sends) == [] and set(frames(sends, FILTER_CTRL_ADDR)) <= {MODE_OFF}
  assert session.failed
