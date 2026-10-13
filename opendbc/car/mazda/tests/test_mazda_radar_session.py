"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The radar UDS session: RadarSessionManager's bounds on its own, then what goes on the bus in
each state driven through the real CarController.update_longitudinal.
"""
import pytest

from opendbc.car import DT_CTRL
from opendbc.car import structs
from opendbc.car.mazda.carcontroller import CRUISE_REQUEST_SETTLE_FRAMES, RADAR_HANDBACK_HOLD_FRAMES, RADAR_RESTART_FRAMES, TAKEOVER_STOP_FRAMES, \
                                           RADAR_WARMUP_MAX_FRAMES, stock_ecu_driver_view
from opendbc.car.mazda.radar_session import RADAR_SESSION_LIMIT_FRAMES, RADAR_RESTORE_FRAMES, RADAR_UDS_STEP, RadarSessionManager, \
  RadarSessionState
from opendbc.car.mazda.tests.conftest import (CRZ_CTRL, CRZ_INFO, RADAR_STATIC, RADAR_UDS, SESSION_DFLT_DAT, SESSION_PROG_DAT,
                                              TESTER_PRESENT_DAT, LongCtrlState, car_controller, car_interface, frames, step_long)
from opendbc.car.mazda.values import CarControllerParams
from opendbc.sunnypilot.car.stock_ecu import StockEcuState


class RadarSessionManager(RadarSessionManager):
  """Direct-driven manager: counts the controller frame the production caller passes in."""

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.n = -1

  def update(self, *args, **kwargs):
    self.n += 1
    return super().update(*args, frame=self.n, **kwargs)


class TestRadarSessionBounds:
  """Diagnostic attempts are bounded; completion requires observed stock recovery."""

  def test_silencing_gives_up_bounded(self):
    m = RadarSessionManager()
    for _ in range(RADAR_SESSION_LIMIT_FRAMES + 2):
      state = m.update(True, True, False, standstill=True, session_refused=False, stock_radar_gone=False)
    assert state == RadarSessionState.HANDBACK and m.silencing_failed
    for _ in range(RADAR_UDS_STEP + RADAR_RESTORE_FRAMES):
      m.update(True, True, False, standstill=True, session_refused=False, stock_radar_gone=False)
    # and stays given up for the drive: stock keeps the bus
    for _ in range(10):
      assert m.update(True, True, False, standstill=True, session_refused=False, stock_radar_gone=False) == RadarSessionState.STOCK

  def test_negative_response_gives_up_immediately(self):
    # route 000000fe t+15.0 shows the radar answers a session request within 10 ms, so a
    # negative response is definitive: no reason to burn the silence budget
    m = RadarSessionManager()
    m.update(True, True, False, standstill=True, session_refused=False, stock_radar_gone=False)
    assert m.state == RadarSessionState.SILENCING
    assert m.update(True, True, False, standstill=True, session_refused=True, stock_radar_gone=False) == RadarSessionState.HANDBACK
    assert m.silencing_failed

  def test_handback_stops_waiting_for_a_dead_radar(self):
    m = RadarSessionManager()
    m.update(True, False, False, standstill=True, session_refused=False, stock_radar_gone=True)
    assert m.state == RadarSessionState.SILENCED
    for _ in range(RADAR_SESSION_LIMIT_FRAMES + 2):
      state = m.update(True, False, True, standstill=True, session_refused=False, stock_radar_gone=True)
    assert state == RadarSessionState.HANDBACK
    assert m.handback_failed and not m.handback_completed
    assert m.diagnostic_message is None

  def test_ordered_handback_stays_stock_while_the_request_stands(self):
    # the lifecycle holds its assert for the whole stop: no re-silencing right before shutdown
    m = RadarSessionManager()
    m.update(True, False, False, standstill=True, session_refused=False, stock_radar_gone=True)
    assert m.state == RadarSessionState.SILENCED
    m.update(True, False, True, standstill=True, session_refused=False, stock_radar_gone=True)
    assert m.state == RadarSessionState.HANDBACK
    for _ in range(RADAR_UDS_STEP + RADAR_RESTORE_FRAMES):
      m.update(True, True, True, standstill=True, session_refused=False, stock_radar_gone=False)
    assert m.state == RadarSessionState.STOCK and m.handback_completed
    for alive in (True, False):
      for _ in range(5):
        assert m.update(True, alive, True, standstill=True, session_refused=False, stock_radar_gone=not alive) == RadarSessionState.STOCK
    assert m.status == StockEcuState.RESTORED

  def test_withdrawn_request_after_the_restore_is_a_fresh_start(self):
    # forced offroad canceled once the radar was handed back: the next takeover is a first one
    m = RadarSessionManager()
    m.update(True, False, False, standstill=True, session_refused=False, stock_radar_gone=True)
    m.update(True, False, True, standstill=True, session_refused=False, stock_radar_gone=True)
    for _ in range(RADAR_UDS_STEP + RADAR_RESTORE_FRAMES):
      m.update(True, True, True, standstill=True, session_refused=False, stock_radar_gone=False)
    assert m.handback_completed
    assert m.update(True, True, False, standstill=True, session_refused=False, stock_radar_gone=False) == RadarSessionState.SILENCING
    assert not m.handback_completed

  def test_withdrawn_handback_finishes_restoration(self):
    # A reversal must finish restoration; card cycles using the latest toggle value.
    m = RadarSessionManager()
    m.update(True, False, False, standstill=True, session_refused=False, stock_radar_gone=True)
    m.update(True, False, True, standstill=True, session_refused=False, stock_radar_gone=True)
    assert m.state == RadarSessionState.HANDBACK
    state = m.update(True, False, False, standstill=True, session_refused=False, stock_radar_gone=True)
    assert state == RadarSessionState.HANDBACK and not m.handback_completed

  def test_silencing_waits_for_standstill_but_adoption_does_not(self):
    # actively silencing disables AEB, so it only starts pre-motion like disable_ecu;
    # adopting an already-quiet radar disables nothing and proceeds anywhere
    m = RadarSessionManager()
    for _ in range(10):
      assert m.update(True, True, False, standstill=False, session_refused=False, stock_radar_gone=False) == RadarSessionState.STOCK
    assert m.update(True, True, False, standstill=True, session_refused=False, stock_radar_gone=False) == RadarSessionState.SILENCING
    m2 = RadarSessionManager()
    assert m2.update(True, False, False, standstill=False, session_refused=False, stock_radar_gone=True) == RadarSessionState.SILENCED

  def test_a_dropped_stock_frame_is_not_adopted(self):
    # 7.25M stock CRZ_INFO frames: 9 inter-arrival gaps past the 50 ms alive window, 2 past
    # 100 ms (max 105.7 ms, one mid-drive at speed). Silence inside the guard window is a
    # dropped frame, not a torn-down radar: nothing goes on the bus, and no session request
    # either while the car is moving
    m = RadarSessionManager()
    for _ in range(int(CarControllerParams.STOCK_RADAR_GUARD_T / DT_CTRL)):
      assert m.update(True, False, False, standstill=False, session_refused=False, stock_radar_gone=False) == RadarSessionState.STOCK
    # the guard-long silence is the adoption
    assert m.update(True, False, False, standstill=False, session_refused=False, stock_radar_gone=True) == RadarSessionState.SILENCED

  def test_a_returned_radar_is_resilenced_under_the_teardown_gate(self):
    # a radar heard again under our synthetic frames: our frames stop at once (two masters),
    # and the session request follows the same pre-motion gate as the first teardown. Moving,
    # stock keeps the bus until the next stop
    m = RadarSessionManager()
    m.update(True, False, False, standstill=False, session_refused=False, stock_radar_gone=True)
    assert m.state == RadarSessionState.SILENCED
    assert m.update(True, True, False, standstill=False, session_refused=False, stock_radar_gone=False) == RadarSessionState.STOCK
    for _ in range(300):
      assert m.update(True, True, False, standstill=False, session_refused=False, stock_radar_gone=False) == RadarSessionState.STOCK
    assert m.update(True, True, False, standstill=True, session_refused=False, stock_radar_gone=False) == RadarSessionState.SILENCING
    while not m.programming_sent:
      m.update(True, True, False, standstill=True, session_refused=False, stock_radar_gone=False)
    # the handover back is on the alive window, not the guard: the radar stops within a frame
    # of accepting the session, and every frame in between is a radar gap for the camera
    assert m.update(True, False, False, standstill=True, session_refused=False, stock_radar_gone=False) == RadarSessionState.SILENCED
    # stopped when it comes back: re-requested right away, as before
    m2 = RadarSessionManager()
    m2.update(True, False, False, standstill=True, session_refused=False, stock_radar_gone=True)
    assert m2.update(True, True, False, standstill=True, session_refused=False, stock_radar_gone=False) == RadarSessionState.SILENCING


def boot_step(cc, cs, stock_radar_alive, fsc_settled, handback=False, cruise_engaged=False, standstill=True,
              stock_radar_gone=None, buttons=(), available=False):
  # standstill=True models the parked boot; actively silencing a live radar is gated on it
  return step_long(cc, cs, long_active=False, accel=0., long_state=LongCtrlState.off, lead_visible=False, available=available,
                   stock_radar_alive=stock_radar_alive, stock_radar_gone=stock_radar_gone, fsc_settled=fsc_settled,
                   handback=handback, cruise_engaged=cruise_engaged, standstill=standstill, buttons=buttons)


def uds(sends):
  return frames(sends, RADAR_UDS)


def synthetic(sends):
  return [a for a, _, _ in sends if a in (CRZ_INFO, CRZ_CTRL, RADAR_STATIC)]


class TestRadarSessionSequencing:
  """Boot teardown deferral and the ordered hand-back: what goes on the bus in each
  radar session state, driven through the real CarController.update_longitudinal."""

  @pytest.mark.parametrize("phase", range(RADAR_UDS_STEP))
  def test_cancel_inflight_teardown_restores_default_before_finishing(self, cc, cs, phase):
    assert SESSION_PROG_DAT in uds(boot_step(cc, cs, stock_radar_alive=True, fsc_settled=True))
    for _ in range(phase):
      boot_step(cc, cs, stock_radar_alive=True, fsc_settled=True)
    # The last radar frame remains fresh when the toggle flips. No synthetic traffic
    # may overlap it, but the outstanding programming request still needs undoing.
    saw_default = False
    for _ in range(RADAR_UDS_STEP + 2):
      sends = boot_step(cc, cs, stock_radar_alive=True, fsc_settled=True, handback=True)
      saw_default |= SESSION_DFLT_DAT in uds(sends)
      assert synthetic(sends) == []
      assert SESSION_PROG_DAT not in uds(sends)
    assert saw_default
    assert cc.radar_session.handback_completed
    # the request withdrawn after the restore (toggle flipped back): a fresh parked takeover
    resumed = False
    for _ in range(RADAR_UDS_STEP + 1):
      resumed |= SESSION_PROG_DAT in uds(boot_step(cc, cs, stock_radar_alive=True, fsc_settled=True))
    assert resumed

  def test_cancel_inflight_teardown_covers_late_silence(self, cc, cs):
    boot_step(cc, cs, stock_radar_alive=True, fsc_settled=True)
    boot_step(cc, cs, stock_radar_alive=True, fsc_settled=True, handback=True)
    saw_default = False
    saw_synthetic = False
    for _ in range(100):
      sends = boot_step(cc, cs, stock_radar_alive=False, fsc_settled=True, handback=True)
      saw_default |= SESSION_DFLT_DAT in uds(sends)
      saw_synthetic |= bool(synthetic(sends))
      assert TESTER_PRESENT_DAT not in uds(sends)
      assert SESSION_PROG_DAT not in uds(sends)
    assert saw_default and saw_synthetic
    assert cc.radar_session.state == RadarSessionState.HANDBACK
    assert boot_step(cc, cs, stock_radar_alive=True, fsc_settled=True, handback=True) == []

  def test_returned_radar_requires_setup_gate_again(self, cc, cs):
    for settled, engaged in ((False, False), (True, True)):
      boot_step(cc, cs, stock_radar_alive=False, fsc_settled=True)
      for _ in range(100):
        assert boot_step(cc, cs, stock_radar_alive=True, fsc_settled=settled, cruise_engaged=engaged) == []

  def test_handback_before_teardown_stops_everything(self, cc, cs):
    # toggle-off while still waiting on the gate: no session ever entered, so no
    # hand-back traffic either
    boot_step(cc, cs, stock_radar_alive=True, fsc_settled=False)
    for _ in range(120):
      assert boot_step(cc, cs, stock_radar_alive=True, fsc_settled=False, handback=True) == []

  def test_stock_cruise_running_drops_the_teardown(self, cc, cs):
    # driver engaged stock MRCC with a request standing (warm boot): no teardown under it, and
    # none after it ends: the car's cruise running is the driver choosing it
    for _ in range(120):
      assert boot_step(cc, cs, stock_radar_alive=True, fsc_settled=True, cruise_engaged=True) == []
    assert not cc.cruise_requested
    cc.frame = 0
    for _ in range(300):
      assert SESSION_PROG_DAT not in uds(boot_step(cc, cs, stock_radar_alive=True, fsc_settled=True, cruise_engaged=False))

  def test_completed_handback_stays_stock_while_asserted(self, cc, cs):
    # the hand-back server holds CC_SP.stockEcuHandBack for the whole stop; nothing goes on the
    # bus once stock traffic is back
    boot_step(cc, cs, stock_radar_alive=False, fsc_settled=True)
    boot_step(cc, cs, stock_radar_alive=False, fsc_settled=True, handback=True)
    for _ in range(RADAR_UDS_STEP + RADAR_RESTORE_FRAMES):
      boot_step(cc, cs, stock_radar_alive=True, fsc_settled=True, handback=True)
    assert cc.radar_session.handback_completed
    for _ in range(200):
      assert boot_step(cc, cs, stock_radar_alive=True, fsc_settled=True, handback=True) == []

  def test_s3_recovery_resilences(self, cc, cs):
    # radar reappears mid-drive (dropped tester present, S3 timeout): re-request the session
    boot_step(cc, cs, stock_radar_alive=False, fsc_settled=True)
    cc.frame = RADAR_UDS_STEP  # align to a session-request frame
    sends = boot_step(cc, cs, stock_radar_alive=True, fsc_settled=True)
    assert SESSION_PROG_DAT in uds(sends)
    # and settles back to silenced once quiet again
    sends = boot_step(cc, cs, stock_radar_alive=False, fsc_settled=True)
    assert SESSION_PROG_DAT not in uds(sends)

  def test_a_returned_radar_at_speed_sends_no_session_request(self, cc, cs):
    # radar heard again while rolling: synthetic frames and tester present stop, and the
    # programming-session request waits for a stop like the first teardown did
    boot_step(cc, cs, stock_radar_alive=False, fsc_settled=True, standstill=False)
    for _ in range(300):
      assert boot_step(cc, cs, stock_radar_alive=True, stock_radar_gone=False, fsc_settled=True, standstill=False) == []
    for _ in range(TAKEOVER_STOP_FRAMES - 1):  # a real stop first
      assert boot_step(cc, cs, stock_radar_alive=True, stock_radar_gone=False, fsc_settled=True, standstill=True) == []
    cc.frame = RADAR_UDS_STEP
    sends = boot_step(cc, cs, stock_radar_alive=True, stock_radar_gone=False, fsc_settled=True, standstill=True)
    assert uds(sends) == [SESSION_PROG_DAT]

  def test_a_stock_frame_gap_at_speed_puts_nothing_on_the_bus(self, cc, cs):
    # gate passed, car rolling, the stock radar drops a few frames (alive window expired,
    # guard window not): no synthetic frames, no tester present, no session request
    for _ in range(300):
      assert boot_step(cc, cs, stock_radar_alive=False, stock_radar_gone=False, fsc_settled=True, standstill=False) == []
    # the guard-long silence is the adoption
    cc.frame = 0
    sends = boot_step(cc, cs, stock_radar_alive=False, stock_radar_gone=True, fsc_settled=True, standstill=False)
    assert len(synthetic(sends)) > 0


def _stock(m, standstill, alive=True, refused=False, gate=True, handback=False, **kw):
  return m.update(gate, alive, handback, standstill=standstill, session_refused=refused, stock_radar_gone=not alive, **kw)


class TestParkedTakeover:
  """A fresh session started with the car rolling (forced offroad exit, process restart) waits
  for the stop, and the status says so."""

  def test_a_rolling_session_waits_for_the_stop(self):
    m = RadarSessionManager()
    for _ in range(300):
      assert _stock(m, standstill=False) == RadarSessionState.STOCK
    assert m.status == StockEcuState.PARK_TO_TAKE_OVER
    assert _stock(m, standstill=True) == RadarSessionState.SILENCING

  def test_parked_refusal_is_definitive(self):
    m = RadarSessionManager()
    _stock(m, standstill=True)
    _stock(m, standstill=True, refused=True)
    assert m.silencing_failed
    for _ in range(RADAR_UDS_STEP + RADAR_RESTORE_FRAMES + 50):
      _stock(m, standstill=True)
    for standstill in (True, False):
      assert _stock(m, standstill=standstill) == RadarSessionState.STOCK
    assert m.status == StockEcuState.FAILED

  def test_pulling_away_undoes_the_queued_request(self):
    m = RadarSessionManager()
    _stock(m, standstill=True)
    while not m.programming_sent:
      _stock(m, standstill=True)
    assert _stock(m, standstill=False) == RadarSessionState.HANDBACK

  def test_undone_request_leaves_the_next_attempt_open(self):
    # undoing our own request (motion on a parked-only radar) is not an ordered hand-back
    m = RadarSessionManager()
    _stock(m, standstill=True)
    while not m.programming_sent:
      _stock(m, standstill=True)
    _stock(m, standstill=False)
    for _ in range(RADAR_UDS_STEP + RADAR_RESTORE_FRAMES):
      _stock(m, standstill=False)
    assert m.state == RadarSessionState.STOCK and not m.handback_completed
    assert _stock(m, standstill=True) == RadarSessionState.SILENCING

  def test_returned_radar_is_only_resilenced_parked(self):
    # a radar heard again under our frames came back through its own S3 timeout
    m = RadarSessionManager()
    _stock(m, standstill=False, alive=False)
    assert m.state == RadarSessionState.SILENCED
    assert _stock(m, standstill=False) == RadarSessionState.STOCK
    for _ in range(200):
      assert _stock(m, standstill=False) == RadarSessionState.STOCK
    assert m.status == StockEcuState.PARK_TO_TAKE_OVER
    assert _stock(m, standstill=True) == RadarSessionState.SILENCING


class TestStockEcuStatus:
  """The transition contract: the driver's view of ownership, in place on the manager."""

  def test_starting_covers_every_prerequisite(self):
    m = RadarSessionManager()
    _stock(m, standstill=True, gate=False)
    assert m.status == StockEcuState.STARTING
    _stock(m, standstill=True, bus_healthy=False)
    assert m.status == StockEcuState.STARTING
    _stock(m, standstill=True, stock_engaged=True)
    assert m.status == StockEcuState.STOCK_CRUISE_ON
    _stock(m, standstill=True)
    assert m.state == RadarSessionState.SILENCING and m.status == StockEcuState.STARTING

  def test_ready_restoring_restored(self):
    m = RadarSessionManager()
    _stock(m, standstill=True)
    while not m.programming_sent:
      _stock(m, standstill=True)
    _stock(m, standstill=True, alive=False, owned=True)
    assert m.status == StockEcuState.READY
    _stock(m, standstill=True, alive=False, handback=True)
    assert m.status == StockEcuState.RESTORING
    for _ in range(RADAR_UDS_STEP + RADAR_RESTORE_FRAMES):
      _stock(m, standstill=True, handback=True)
    assert (m.status, m.handback_completed) == (StockEcuState.RESTORED, True)

  def test_restore_timeout_is_a_failure_until_late_recovery(self):
    m = RadarSessionManager()
    _stock(m, standstill=True, alive=False)
    for _ in range(RADAR_SESSION_LIMIT_FRAMES + 2):
      _stock(m, standstill=True, alive=False, handback=True)
    assert (m.status, m.handback_failed) == (StockEcuState.FAILED, True)
    for _ in range(RADAR_RESTORE_FRAMES):
      _stock(m, standstill=True, handback=True)
    assert (m.status, m.handback_failed, m.handback_completed) == (StockEcuState.RESTORED, False, True)


class TestControllerStatus:
  def test_alpha_long_controller_publishes_the_contract(self, cc, cs):
    boot_step(cc, cs, stock_radar_alive=True, fsc_settled=False)
    assert cc.stock_ecu_state == StockEcuState.STARTING
    boot_step(cc, cs, stock_radar_alive=True, fsc_settled=True)
    assert cc.radar_session.state == RadarSessionState.SILENCING
    car_cruise = car_controller(alpha_long=True)
    boot_step(car_cruise, cs, stock_radar_alive=True, fsc_settled=True, cruise_engaged=True)
    assert car_cruise.stock_ecu_state == StockEcuState.STOCK_CRUISE_ON

  def test_stock_long_controller_needs_nothing(self, stock_cc):
    assert stock_cc.stock_ecu_state == StockEcuState.NOT_NEEDED

  def test_carstate_reads_the_controllers_session(self):
    # one session, run by the controller: carstate never keeps a copy of its own
    CI = car_interface()
    assert CI.CS.radar_session is CI.CC.radar_session

  def test_ready_follows_carstate_guard(self, cc, cs):
    boot_step(cc, cs, stock_radar_alive=False, fsc_settled=True)
    assert cc.radar_session.state == RadarSessionState.SILENCED and cc.stock_ecu_state == StockEcuState.STARTING
    cs.radar_owned = True
    boot_step(cc, cs, stock_radar_alive=False, fsc_settled=True)
    assert cc.stock_ecu_state == StockEcuState.READY


ButtonType = structs.CarState.ButtonEvent.Type


class TestCruiseRequestTakeover:
  """The stock radar, and its AEB, stays in its default session until the driver presses SET/RES
  in the drive; the takeover is at the stop after that press, once the press has settled."""

  @pytest.fixture
  def cc(self):
    return car_controller(alpha_long=True, cruise_requested=False)

  def test_no_banner_before_the_first_frame(self, cc):
    # card publishes the state from ignition, seconds before the first controls frame
    assert cc.radar_session.status == StockEcuState.PARK_TO_TAKE_OVER and cc.stock_ecu_state == StockEcuState.NOT_NEEDED

  def test_no_request_keeps_the_radar_stock(self, cc, cs):
    # a whole parked drive with the camera settled and the main button pressed: nothing on the bus,
    # and no "remain parked" banner state
    for i in range(2000):
      buttons = (ButtonType.mainCruise,) if i == 500 else ()
      assert boot_step(cc, cs, stock_radar_alive=True, fsc_settled=True, buttons=buttons, available=True) == []
      assert cc.radar_session.status == StockEcuState.PARK_TO_TAKE_OVER and cc.stock_ecu_state == StockEcuState.NOT_NEEDED
    assert cc.radar_session.state == RadarSessionState.STOCK

  def test_press_at_a_stop_takes_over_after_the_settle(self, cc, cs):
    boot_step(cc, cs, stock_radar_alive=True, fsc_settled=True, buttons=(ButtonType.resumeCruise,), available=True)
    assert cc.stock_ecu_state == StockEcuState.STARTING
    for _ in range(CRUISE_REQUEST_SETTLE_FRAMES - 1):
      assert boot_step(cc, cs, stock_radar_alive=True, fsc_settled=True, available=True) == []
    requested = False
    for _ in range(RADAR_UDS_STEP + 2):
      requested |= SESSION_PROG_DAT in uds(boot_step(cc, cs, stock_radar_alive=True, fsc_settled=True, available=True))
    assert requested

  def test_press_while_moving_waits_for_the_stop(self, cc, cs):
    boot_step(cc, cs, stock_radar_alive=True, fsc_settled=True, standstill=False, buttons=(ButtonType.resumeCruise,), available=True)
    assert cc.radar_session.status == StockEcuState.PARK_TO_TAKE_OVER and cc.stock_ecu_state == StockEcuState.NOT_NEEDED
    for _ in range(500):
      assert boot_step(cc, cs, stock_radar_alive=True, fsc_settled=True, standstill=False, available=True) == []
    assert cc.radar_session.status == StockEcuState.PARK_TO_TAKE_OVER and cc.stock_ecu_state == StockEcuState.NOT_NEEDED
    requested = False
    for _ in range(TAKEOVER_STOP_FRAMES + RADAR_UDS_STEP + 2):
      requested |= SESSION_PROG_DAT in uds(boot_step(cc, cs, stock_radar_alive=True, fsc_settled=True, available=True))
    assert requested

  def test_press_with_main_off_is_not_a_request(self, cc, cs):
    # SET with the main switch off arms nothing on the car: the radar stays stock
    boot_step(cc, cs, stock_radar_alive=True, fsc_settled=True, buttons=(ButtonType.resumeCruise,))
    for _ in range(500):
      assert boot_step(cc, cs, stock_radar_alive=True, fsc_settled=True, available=True) == []
    assert cc.radar_session.status == StockEcuState.PARK_TO_TAKE_OVER and cc.stock_ecu_state == StockEcuState.NOT_NEEDED

  def test_press_that_engages_stock_cruise_drops_the_request(self, cc, cs):
    # the car's own cruise starting inside the settle is the driver choosing it: no takeover, then
    # or after it ends
    boot_step(cc, cs, stock_radar_alive=True, fsc_settled=True, buttons=(ButtonType.resumeCruise,), available=True)
    for _ in range(10):
      assert boot_step(cc, cs, stock_radar_alive=True, fsc_settled=True, available=True) == []
    for _ in range(500):
      assert boot_step(cc, cs, stock_radar_alive=True, fsc_settled=True, cruise_engaged=True, available=True) == []
    assert cc.stock_ecu_state == StockEcuState.STOCK_CRUISE_ON and not cc.cruise_requested
    for _ in range(500):
      assert boot_step(cc, cs, stock_radar_alive=True, fsc_settled=True, available=True) == []


def radar_step(cc, cs, alive, main=True, standstill=True, buttons=(), cruise_engaged=False, set_allowed=True):
  return step_long(cc, cs, long_active=False, accel=0., long_state=LongCtrlState.off, lead_visible=False,
                   available=main, stock_radar_alive=alive, fsc_settled=True, standstill=standstill,
                   cruise_engaged=cruise_engaged, buttons=buttons,
                   radar_was_silenced=cs.radar_was_silenced,  # carstate's latch carries over, as on the car
                   stock_set_allowed=set_allowed)


class TestCruiseOffHandback:
  """The main switch turned off after a takeover hands the radar back to its default session
  (AEB with it); a new SET/RES press re-takes it at the next stop, as the first takeover."""

  def take_over(self, cc, cs):
    radar_step(cc, cs, alive=True, buttons=(ButtonType.resumeCruise,))
    for _ in range(CRUISE_REQUEST_SETTLE_FRAMES + RADAR_UDS_STEP + 2):
      radar_step(cc, cs, alive=True)
    assert cc.radar_session.programming_sent
    for _ in range(RADAR_HANDBACK_HOLD_FRAMES + 1):
      radar_step(cc, cs, alive=False)
    assert cc.radar_session.state == RadarSessionState.SILENCED
    cs.radar_was_silenced = True  # carstate's guard, as on the car

  @pytest.fixture
  def cc(self):
    return car_controller(alpha_long=True, cruise_requested=False)

  @pytest.mark.parametrize("standstill", [True, False])
  def test_main_off_hands_back(self, cc, cs, standstill):
    self.take_over(cc, cs)
    sends = []
    for _ in range(RADAR_UDS_STEP + 1):
      sends += radar_step(cc, cs, alive=False, main=False, standstill=standstill)
    assert SESSION_DFLT_DAT in uds(sends) and TESTER_PRESENT_DAT not in uds(sends)
    assert synthetic(sends)  # no camera-visible gap while the radar is still quiet
    assert cc.stock_ecu_state == StockEcuState.RESTORING
    for _ in range(RADAR_RESTORE_FRAMES + 1):
      sends = radar_step(cc, cs, alive=True, main=False, standstill=standstill)
    assert cc.radar_session.state == RadarSessionState.STOCK
    assert not cs.radar_was_silenced and not cc.radar_session.handback_active and not cc.radar_session.handback_failed
    for _ in range(200):
      assert radar_step(cc, cs, alive=True, main=False, standstill=standstill) == []

  def test_takeover_holds_before_a_hand_back(self, cc, cs):
    self.take_over(cc, cs)
    cc.radar_silenced_frames = 0  # main off right after the takeover
    for _ in range(RADAR_HANDBACK_HOLD_FRAMES):
      assert SESSION_DFLT_DAT not in uds(radar_step(cc, cs, alive=False, main=False))
    sends = []
    for _ in range(RADAR_UDS_STEP + 2):
      sends += radar_step(cc, cs, alive=False, main=False)
    assert SESSION_DFLT_DAT in uds(sends)

  def test_cancel_alone_keeps_the_radar(self, cc, cs):
    self.take_over(cc, cs)
    for i in range(1000):
      sends = radar_step(cc, cs, alive=False, standstill=False, buttons=(ButtonType.cancel,) if i == 0 else ())
      assert SESSION_DFLT_DAT not in uds(sends)
    assert cc.radar_session.state == RadarSessionState.SILENCED

  def test_retake_after_hand_back(self, cc, cs):
    self.take_over(cc, cs)
    for _ in range(RADAR_UDS_STEP + 1):
      radar_step(cc, cs, alive=False, main=False)
    for _ in range(RADAR_RESTORE_FRAMES + 1):
      radar_step(cc, cs, alive=True, main=False)
    # main back on: nothing until a SET/RES press, then the takeover at the stop after the settle
    for _ in range(500):
      assert radar_step(cc, cs, alive=True) == []
    assert cc.radar_session.status == StockEcuState.PARK_TO_TAKE_OVER and cc.stock_ecu_state == StockEcuState.NOT_NEEDED
    radar_step(cc, cs, alive=True, standstill=False, buttons=(ButtonType.resumeCruise,))
    for _ in range(500):
      assert radar_step(cc, cs, alive=True, standstill=False) == []
    requested = False
    for _ in range(TAKEOVER_STOP_FRAMES + RADAR_UDS_STEP + 2):
      requested |= SESSION_PROG_DAT in uds(radar_step(cc, cs, alive=True))
    assert requested

  def test_unanswered_hand_back_fails_then_recovers_late(self, cc, cs):
    self.take_over(cc, cs)
    for _ in range(RADAR_SESSION_LIMIT_FRAMES + 1):
      radar_step(cc, cs, alive=False, main=False)
    # the driver's view: SET says it failed; no cruise fault, so MADS lateral is untouched
    assert cc.stock_ecu_state == StockEcuState.FAILED and cc.radar_session.handback_by_driver
    for _ in range(200):
      sends = radar_step(cc, cs, alive=False, main=False)
      assert uds(sends) == []  # bounded: no more diagnostics, neutral frames only
    # the radar's own session timeout brings it back: the hand-back completes after all
    for _ in range(RADAR_RESTORE_FRAMES + 1):
      radar_step(cc, cs, alive=True, main=False)
    assert cc.radar_session.state == RadarSessionState.STOCK and not cc.radar_session.handback_failed

  def test_main_off_press_shows_nothing(self, cc, cs):
    # SET with main off does nothing on the car: the state engages normally, so no pop-up
    radar_step(cc, cs, alive=True, main=False, buttons=(ButtonType.resumeCruise,))
    assert cc.stock_ecu_state == StockEcuState.NOT_NEEDED
    radar_step(cc, cs, alive=True)
    assert cc.radar_session.status == StockEcuState.PARK_TO_TAKE_OVER and cc.stock_ecu_state == StockEcuState.NOT_NEEDED

  def test_stock_cruise_warmup_after_hand_back(self, cc, cs):
    # back in its default session the radar refuses SET ~12 s: SET says stock cruise is returning
    self.take_over(cc, cs)
    for _ in range(RADAR_UDS_STEP + 1):
      radar_step(cc, cs, alive=False, main=False)
    for _ in range(RADAR_RESTORE_FRAMES + 1):
      radar_step(cc, cs, alive=True, main=False, set_allowed=False)
    for _ in range(1000):
      radar_step(cc, cs, alive=True, set_allowed=False)
      assert cc.stock_ecu_state == StockEcuState.RESTORING
    radar_step(cc, cs, alive=True, set_allowed=True)
    assert cc.radar_session.status == StockEcuState.PARK_TO_TAKE_OVER and cc.stock_ecu_state == StockEcuState.NOT_NEEDED

  def test_stock_cruise_warmup_is_bounded(self, cc, cs):
    self.take_over(cc, cs)
    for _ in range(RADAR_UDS_STEP + 1):
      radar_step(cc, cs, alive=False, main=False)
    for _ in range(RADAR_RESTORE_FRAMES + 1):
      radar_step(cc, cs, alive=True, main=False, set_allowed=False)
    for _ in range(RADAR_WARMUP_MAX_FRAMES + 1):
      radar_step(cc, cs, alive=True, set_allowed=False)
    assert cc.radar_session.status == StockEcuState.PARK_TO_TAKE_OVER and cc.stock_ecu_state == StockEcuState.NOT_NEEDED

  def test_set_during_the_warmup_is_not_a_request(self, cc, cs):
    self.take_over(cc, cs)
    for _ in range(RADAR_UDS_STEP + 1):
      radar_step(cc, cs, alive=False, main=False)
    # main on and SET straight away, stopped: the radar is still restarting, so no re-take
    radar_step(cc, cs, alive=True, set_allowed=False, buttons=(ButtonType.resumeCruise,))
    for _ in range(RADAR_RESTORE_FRAMES + 1):
      radar_step(cc, cs, alive=True, set_allowed=False)
    radar_step(cc, cs, alive=True, set_allowed=False, buttons=(ButtonType.resumeCruise,))
    assert cc.stock_ecu_state == StockEcuState.RESTORING
    for _ in range(500):
      assert SESSION_PROG_DAT not in uds(radar_step(cc, cs, alive=True, set_allowed=False))
    # the radar allows SET again: a press counts, the re-take follows at this stop
    radar_step(cc, cs, alive=True, buttons=(ButtonType.resumeCruise,))
    requested = False
    for _ in range(CRUISE_REQUEST_SETTLE_FRAMES + RADAR_UDS_STEP + 2):
      requested |= SESSION_PROG_DAT in uds(radar_step(cc, cs, alive=True))
    assert requested

  def test_stock_cruise_on_holds_through_a_stopped_press(self, cc, cs):
    # the car's own cruise holding at a stop, RES pressed: one engagement, so one state throughout
    radar_step(cc, cs, alive=True, cruise_engaged=True)
    for i in range(CRUISE_REQUEST_SETTLE_FRAMES * 2):
      radar_step(cc, cs, alive=True, cruise_engaged=True, buttons=(ButtonType.resumeCruise,) if i == 0 else ())
      assert cc.stock_ecu_state == StockEcuState.STOCK_CRUISE_ON
    radar_step(cc, cs, alive=True)
    assert cc.stock_ecu_state != StockEcuState.STOCK_CRUISE_ON

  def test_silenced_follows_the_radar_session(self, cc, cs, stock_cc):
    # AEB off exactly while the radar is out of its default session
    assert not cc.stock_ecu_off and not stock_cc.stock_ecu_off
    radar_step(cc, cs, alive=True, buttons=(ButtonType.resumeCruise,))
    for _ in range(CRUISE_REQUEST_SETTLE_FRAMES - 1):
      radar_step(cc, cs, alive=True)
      assert not cc.stock_ecu_off  # settle: nothing requested yet
    for _ in range(RADAR_UDS_STEP + 2):
      radar_step(cc, cs, alive=True)
    assert cc.radar_session.programming_sent and cc.stock_ecu_off  # requested away
    for _ in range(RADAR_HANDBACK_HOLD_FRAMES + 1):
      radar_step(cc, cs, alive=False)
    assert cc.stock_ecu_off
    cs.radar_was_silenced = True
    for _ in range(RADAR_UDS_STEP + 1):
      radar_step(cc, cs, alive=False, main=False)
    assert cc.radar_session.state == RadarSessionState.HANDBACK and cc.stock_ecu_off  # not back yet
    for _ in range(RADAR_RESTORE_FRAMES + 1):
      radar_step(cc, cs, alive=True, main=False)
    assert cc.radar_session.state == RadarSessionState.STOCK
    for _ in range(RADAR_RESTART_FRAMES - 1):
      assert cc.stock_ecu_off  # back, but AEB not counted as on before the radar's restart time
      radar_step(cc, cs, alive=True, main=False)
    radar_step(cc, cs, alive=True, main=False)
    assert not cc.stock_ecu_off


  def test_the_radar_returning_at_a_stop_keeps_aeb_off_through_the_retake(self, cc, cs):
    # back by itself at a stop with the request standing: re-taken on the same frame, and AEB
    # counts as off throughout, never a moment's "AEB on" before programming is sent again
    self.take_over(cc, cs)
    for _ in range(RADAR_SESSION_LIMIT_FRAMES):
      radar_step(cc, cs, alive=True)
      assert cc.stock_ecu_off

  @pytest.mark.parametrize("roll", [False, True])
  def test_main_off_mid_takeover_withdraws_the_request(self, cc, cs, roll):
    # SET, a real stop, programming requested; then main off (and, if roll, driving away):
    # the takeover is abandoned and handed back, and no later stop takes the radar with main off
    radar_step(cc, cs, alive=True, buttons=(ButtonType.resumeCruise,))
    for _ in range(CRUISE_REQUEST_SETTLE_FRAMES + RADAR_UDS_STEP + 2):
      radar_step(cc, cs, alive=True)
    assert cc.radar_session.state == RadarSessionState.SILENCING and cc.radar_session.programming_sent
    radar_step(cc, cs, alive=True, main=False, standstill=not roll)
    assert not cc.cruise_requested and cc.radar_session.state == RadarSessionState.HANDBACK
    for _ in range(RADAR_RESTORE_FRAMES + RADAR_UDS_STEP + 2):
      radar_step(cc, cs, alive=True, main=False, standstill=not roll)
    assert cc.radar_session.state == RadarSessionState.STOCK
    for _ in range(TAKEOVER_STOP_FRAMES * 3):  # a long stop, main still off
      assert SESSION_PROG_DAT not in uds(radar_step(cc, cs, alive=True, main=False))
    assert cc.radar_session.state == RadarSessionState.STOCK

  def test_main_off_before_the_takeover_withdraws_the_request(self, cc, cs):
    # SET while moving, main off, then a stop: cruise cannot engage, so the radar stays the car's
    radar_step(cc, cs, alive=True, standstill=False, buttons=(ButtonType.resumeCruise,))
    assert cc.cruise_requested
    radar_step(cc, cs, alive=True, main=False, standstill=False)
    for _ in range(CRUISE_REQUEST_SETTLE_FRAMES + RADAR_UDS_STEP + 2):
      assert SESSION_PROG_DAT not in uds(radar_step(cc, cs, alive=True, main=False))
    assert not cc.cruise_requested and not cc.stock_ecu_off
    assert cc.stock_ecu_state == StockEcuState.NOT_NEEDED

  def test_a_retake_inside_the_restart_window_keeps_aeb_off(self, cc, cs):
    # back from the driver's hand-back, SET allowed again within the 12 s: the re-take starts
    # while AEB is still restarting, so AEB never counts as on in between (no "AEB on")
    self.take_over(cc, cs)
    for _ in range(RADAR_UDS_STEP + 1):
      radar_step(cc, cs, alive=False, main=False)
    for _ in range(RADAR_RESTORE_FRAMES + 1):
      radar_step(cc, cs, alive=True, main=False)
    assert cc.radar_session.state == RadarSessionState.STOCK and cc.stock_ecu_off
    radar_step(cc, cs, alive=True, buttons=(ButtonType.resumeCruise,))
    assert cc.cruise_requested
    for _ in range(CRUISE_REQUEST_SETTLE_FRAMES + RADAR_UDS_STEP + 2):
      radar_step(cc, cs, alive=True)
      assert cc.stock_ecu_off
    assert cc.radar_session.programming_sent

  def test_the_radar_returning_by_itself_restarts_aeb(self, cc, cs):
    # the radar's own recovery from programming: its AEB restarts as after any hand-back
    self.take_over(cc, cs)
    for _ in range(RADAR_SESSION_LIMIT_FRAMES):
      radar_step(cc, cs, alive=True, standstill=False)
      if cc.radar_session.state == RadarSessionState.STOCK:
        break
    assert cc.radar_session.state == RadarSessionState.STOCK
    for _ in range(RADAR_RESTART_FRAMES - 1):  # driving on: no re-take while moving
      assert cc.stock_ecu_off
      radar_step(cc, cs, alive=True, standstill=False)
    radar_step(cc, cs, alive=True, standstill=False)
    assert not cc.stock_ecu_off


class TestRealStop:
  """In creeping traffic a stop shorter than TAKEOVER_STOP_FRAMES starts no takeover: AEB stays on."""

  @pytest.fixture
  def cc(self):
    return car_controller(alpha_long=True, cruise_requested=False)

  def test_short_stops_start_nothing(self, cc, cs):
    radar_step(cc, cs, alive=True, standstill=False, buttons=(ButtonType.resumeCruise,))
    for _ in range(CRUISE_REQUEST_SETTLE_FRAMES + 1):
      radar_step(cc, cs, alive=True, standstill=False)
    for _ in range(5):  # stop just under the limit, then creep on
      for _ in range(TAKEOVER_STOP_FRAMES - 1):
        assert SESSION_PROG_DAT not in uds(radar_step(cc, cs, alive=True))
        assert not cc.stock_ecu_off and cc.radar_session.state == RadarSessionState.STOCK
      for _ in range(50):
        radar_step(cc, cs, alive=True, standstill=False)
    assert cc.stock_ecu_state == StockEcuState.NOT_NEEDED  # quiet throughout

  def test_a_real_stop_takes_over(self, cc, cs):
    radar_step(cc, cs, alive=True, standstill=False, buttons=(ButtonType.resumeCruise,))
    for _ in range(CRUISE_REQUEST_SETTLE_FRAMES + 1):
      radar_step(cc, cs, alive=True, standstill=False)
    requested = False
    for _ in range(TAKEOVER_STOP_FRAMES + RADAR_UDS_STEP + 2):
      requested |= SESSION_PROG_DAT in uds(radar_step(cc, cs, alive=True))
    assert requested


class TestAlphaLongRequest:
  """RES asks for alpha long, only before the car's own cruise has run since main came on; SET+
  and SET- are the car's cruise; main off clears that memory."""

  @pytest.fixture
  def cc(self):
    return car_controller(alpha_long=True, cruise_requested=False)

  def long_stop_takes_over(self, cc, cs):
    sent = False
    for _ in range(TAKEOVER_STOP_FRAMES + CRUISE_REQUEST_SETTLE_FRAMES + RADAR_UDS_STEP + 2):
      sent |= SESSION_PROG_DAT in uds(radar_step(cc, cs, alive=True))
    return sent

  @pytest.mark.parametrize("button", [ButtonType.accelCruise, ButtonType.decelCruise, ButtonType.setCruise])
  def test_set_is_the_car_cruise(self, cc, cs, button):
    radar_step(cc, cs, alive=True, standstill=False, buttons=(button,))
    assert not cc.cruise_requested and not cc.alpha_long_requested
    assert not self.long_stop_takes_over(cc, cs)

  def test_res_asks_and_is_flagged_until_taken(self, cc, cs):
    radar_step(cc, cs, alive=True, standstill=False, buttons=(ButtonType.resumeCruise,))
    for _ in range(100):
      radar_step(cc, cs, alive=True, standstill=False)
      assert cc.alpha_long_requested
    assert self.long_stop_takes_over(cc, cs)
    for _ in range(RADAR_HANDBACK_HOLD_FRAMES):  # the radar goes quiet: held
      radar_step(cc, cs, alive=False)
    assert cc.radar_session.state == RadarSessionState.SILENCED
    assert not cc.alpha_long_requested  # taken: the line is no longer due

  def test_res_after_the_car_cruise_ran_is_its_resume(self, cc, cs):
    for _ in range(50):
      radar_step(cc, cs, alive=True, standstill=False, cruise_engaged=True)
    for _ in range(50):  # braked out of it: the car remembers its speed
      radar_step(cc, cs, alive=True, standstill=False)
    radar_step(cc, cs, alive=True, standstill=False, buttons=(ButtonType.resumeCruise,))
    assert not cc.cruise_requested and not cc.alpha_long_requested
    assert not self.long_stop_takes_over(cc, cs)

  def test_main_off_clears_the_memory(self, cc, cs):
    for _ in range(50):
      radar_step(cc, cs, alive=True, standstill=False, cruise_engaged=True)
    for _ in range(10):
      radar_step(cc, cs, alive=True, standstill=False, main=False)
    radar_step(cc, cs, alive=True, standstill=False, buttons=(ButtonType.resumeCruise,))
    assert cc.alpha_long_requested
    assert self.long_stop_takes_over(cc, cs)

  def test_the_car_cruise_starting_drops_a_pending_request(self, cc, cs):
    radar_step(cc, cs, alive=True, standstill=False, buttons=(ButtonType.resumeCruise,))
    for _ in range(CRUISE_REQUEST_SETTLE_FRAMES * 2):
      radar_step(cc, cs, alive=True, standstill=False)
    for _ in range(50):  # changed mind: SET+ for the car's cruise
      radar_step(cc, cs, alive=True, standstill=False, cruise_engaged=True)
    assert not cc.cruise_requested and not cc.alpha_long_requested
    assert not self.long_stop_takes_over(cc, cs)


class TestAlphaLongRequestEnds:
  """A request that can no longer be served is not left pending, nor announced (third review)."""

  @pytest.fixture
  def cc(self):
    return car_controller(alpha_long=True, cruise_requested=False)

  def to_programming(self, cc, cs):
    radar_step(cc, cs, alive=True, buttons=(ButtonType.resumeCruise,))
    for _ in range(CRUISE_REQUEST_SETTLE_FRAMES + RADAR_UDS_STEP + 2):
      radar_step(cc, cs, alive=True)
    assert cc.radar_session.state == RadarSessionState.SILENCING and cc.radar_session.programming_sent

  def test_a_refused_takeover_ends_the_request(self, cc, cs):
    self.to_programming(cc, cs)
    step_long(cc, cs, long_active=False, accel=0., long_state=LongCtrlState.off, lead_visible=False, available=True,
              stock_radar_alive=True, fsc_settled=True, standstill=True, radar_session_refused=True,
              radar_was_silenced=cs.radar_was_silenced, stock_set_allowed=True)
    assert cc.radar_session.silencing_failed
    for _ in range(RADAR_RESTORE_FRAMES + RADAR_UDS_STEP + 2):
      radar_step(cc, cs, alive=True, standstill=False)
      assert not cc.cruise_requested and not cc.alpha_long_requested
    radar_step(cc, cs, alive=True, standstill=False, buttons=(ButtonType.resumeCruise,))
    assert not cc.cruise_requested and not cc.alpha_long_requested  # nothing to promise

  def test_the_car_cruise_mid_takeover_ends_the_request(self, cc, cs):
    self.to_programming(cc, cs)
    radar_step(cc, cs, alive=True, cruise_engaged=True)
    assert not cc.cruise_requested
    for _ in range(RADAR_RESTORE_FRAMES + RADAR_UDS_STEP + 2):
      radar_step(cc, cs, alive=True, standstill=False)
    assert cc.radar_session.state == RadarSessionState.STOCK
    for _ in range(TAKEOVER_STOP_FRAMES * 3):  # a long stop afterwards: the car's cruise was chosen
      assert SESSION_PROG_DAT not in uds(radar_step(cc, cs, alive=True))

  def test_the_flag_needs_main(self, cc, cs):
    radar_step(cc, cs, alive=True, standstill=False, buttons=(ButtonType.resumeCruise,))
    assert cc.alpha_long_requested
    radar_step(cc, cs, alive=True, standstill=False, main=False)
    assert not cc.alpha_long_requested

  def test_the_flag_holds_through_a_takeover_cut_short(self, cc, cs):
    # one request, one announcement: a roll that aborts the takeover does not re-raise the flag
    radar_step(cc, cs, alive=True, standstill=False, buttons=(ButtonType.resumeCruise,))
    for _ in range(CRUISE_REQUEST_SETTLE_FRAMES + TAKEOVER_STOP_FRAMES + 5):
      radar_step(cc, cs, alive=True)
      assert cc.alpha_long_requested
    assert cc.radar_session.state == RadarSessionState.SILENCING
    for _ in range(RADAR_RESTORE_FRAMES + RADAR_UDS_STEP + 2):
      radar_step(cc, cs, alive=True, standstill=False)
      assert cc.alpha_long_requested
    assert cc.radar_session.state == RadarSessionState.STOCK


class TestDriverView:
  """stock_ecu_driver_view, one case per rule, written out rather than re-derived."""
  S, P = RadarSessionState, StockEcuState

  @pytest.mark.parametrize("status, session, engaged, main_on, warming, live, expected", [
    # the car's own cruise engaged on its own radar
    (P.PARK_TO_TAKE_OVER, S.STOCK, True, True, False, True, P.STOCK_CRUISE_ON),
    (P.STARTING, S.STOCK, True, True, False, True, P.STOCK_CRUISE_ON),
    # the lifecycle's completed hand-back is answered as such, cruise or not
    (P.RESTORED, S.STOCK, True, True, False, True, P.RESTORED),
    # engaged reads only while the radar is the car's
    (P.READY, S.SILENCED, True, True, False, False, P.READY),
    # back from a hand-back, SET still refused
    (P.PARK_TO_TAKE_OVER, S.STOCK, False, True, True, True, P.RESTORING),
    (P.PARK_TO_TAKE_OVER, S.STOCK, False, False, True, True, P.NOT_NEEDED),
    # nothing asked for, stock radar heard: SET works on the car
    (P.PARK_TO_TAKE_OVER, S.STOCK, False, True, False, True, P.NOT_NEEDED),
    (P.PARK_TO_TAKE_OVER, S.STOCK, False, True, False, False, P.PARK_TO_TAKE_OVER),
    # everything else: the manager's status
    (P.STARTING, S.SILENCING, False, True, True, True, P.STARTING),
    (P.FAILED, S.HANDBACK, False, False, False, False, P.FAILED),
  ])
  def test_rules(self, status, session, engaged, main_on, warming, live, expected):
    assert stock_ecu_driver_view(status, session, stock_cruise_engaged=engaged, main_on=main_on,
                                 radar_warming=warming, stock_radar_live=live) == expected
