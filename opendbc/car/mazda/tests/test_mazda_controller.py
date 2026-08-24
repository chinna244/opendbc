#!/usr/bin/env python3
"""Tests for the Mazda CX-5 2022+ EPS steering parameters (gated on the EPS, not the model)
and the longitudinal message builders and standstill hold."""

from types import SimpleNamespace

import numpy as np
import pytest

from opendbc.can import CANPacker, CANParser
from opendbc.car import Bus, DT_CTRL, structs
from opendbc.car.mazda import mazdacan
from opendbc.car.mazda.carcontroller import (TJA_MRCC_FIRST_TX_DELAY_NANOS, TJA_MRCC_MAX_TX_FRAMES,
                                             CarController)
from opendbc.car.mazda.longitudinal import LEAD_DEBOUNCE_FRAMES, RESUME_UNLATCH_FRAMES, StandstillHold
from opendbc.car.mazda.interface import CarInterface
from opendbc.car.mazda.values import CAR, CarControllerParams


class TestCarControllerParams:

  @pytest.fixture
  def cx5_2022_params(self):
    class FakeCP:
      carFingerprint = CAR.MAZDA_CX5_2022
      minSteerSpeed = 0.0   # steer_to_zero -> CX-5 2022+ EPS present
    return CarControllerParams(FakeCP())

  @pytest.fixture
  def eps_swap_params(self):
    # A CX-5 2022+ EPS swapped into (or shared by) another Mazda: different model, same EPS.
    class FakeCP:
      carFingerprint = CAR.MAZDA_CX9_2021
      minSteerSpeed = 0.0
    return CarControllerParams(FakeCP())

  @pytest.fixture
  def pre_2022_params(self):
    class FakeCP:
      carFingerprint = CAR.MAZDA_CX5
      minSteerSpeed = 12.5   # no CX-5 EPS -> low-speed lockout, minSteerSpeed > 0
    return CarControllerParams(FakeCP())

  def test_cx5_2022_has_lookup(self, cx5_2022_params):
    assert hasattr(cx5_2022_params, 'STEER_MAX_LOOKUP')
    assert cx5_2022_params.STEER_MAX == 1200

  def test_cx5_2022_low_speed(self, cx5_2022_params):
    p = cx5_2022_params
    for v in [0.0, 5.0, 10.0, 14.2]:
      sm = round(float(np.interp(v, p.STEER_MAX_LOOKUP[0], p.STEER_MAX_LOOKUP[1])))
      assert sm == 1200

  def test_cx5_2022_high_speed(self, cx5_2022_params):
    p = cx5_2022_params
    for v in [14.5, 20.0, 30.0]:
      sm = round(float(np.interp(v, p.STEER_MAX_LOOKUP[0], p.STEER_MAX_LOOKUP[1])))
      assert sm == 800

  def test_cx5_2022_rate_limits(self, cx5_2022_params):
    assert cx5_2022_params.STEER_DELTA_UP == 12
    assert cx5_2022_params.STEER_DELTA_DOWN == 25

  def test_cx5_eps_driver_multiplier(self, cx5_2022_params):
    # 15 is the CX-5-EPS tune (upstream stock is 1)
    assert cx5_2022_params.STEER_DRIVER_MULTIPLIER == 15

  def test_eps_swap_gets_cx5_tune(self, eps_swap_params):
    # EPS present (minSteerSpeed == 0) on a non-CX-5 model still gets the higher-authority tune
    assert eps_swap_params.STEER_MAX == 1200
    assert eps_swap_params.STEER_DRIVER_MULTIPLIER == 15
    assert hasattr(eps_swap_params, 'STEER_MAX_LOOKUP')

  def test_no_eps_no_lookup(self, pre_2022_params):
    assert not hasattr(pre_2022_params, 'STEER_MAX_LOOKUP')
    assert pre_2022_params.STEER_MAX == 800
    assert pre_2022_params.STEER_DRIVER_MULTIPLIER == 1


def crz_info_reference_checksum(dat):
  # independent reimplementation of the CRZ_INFO checksum, validated against 1.94M stock
  # frames including all 10,350 stop-bit frames
  return (0xFF - ((sum(dat[:7]) - (dat[5] & 0x04)) & 0xFF)) & 0xFF


def decode_accel_cmd_raw(dat):
  return (((dat[2] & 0x3) << 11) | (dat[3] << 3) | (dat[4] >> 5)) - 4096


class TestMazdaLongitudinalMessages:
  """The synthetic CRZ_INFO/CRZ_CTRL/radar frames must reproduce stock captures byte for
  byte; the hex values below come from real radar traffic."""

  @pytest.fixture
  def packer(self):
    return CANPacker("mazda_2017")

  def test_crz_info_standby_matches_stock(self, packer):
    for counter in range(16):
      checksum = (0x5d - counter) & 0xff
      expected = f"01ffe3ffc000{counter:02x}{checksum:02x}"
      dat = mazdacan.create_acc_command(packer, 0, counter, 0.0, False, False, False, False)[1]
      assert dat.hex() == expected

  def test_crz_info_available_matches_stock(self, packer):
    for counter in range(16):
      checksum = (0x99 - counter) & 0xff
      expected = f"01ffe2000480{counter:02x}{checksum:02x}"
      dat = mazdacan.create_acc_command(packer, 0, counter, 0.0, False, True, False, False)[1]
      assert dat.hex() == expected

  @pytest.mark.parametrize(("accel", "stopping", "unlatching", "counter", "expected"), [
    (0.0, False, False, 0, "01ffe20006800097"),     # engaged, zero command
    (2.0, False, False, 3, "01ffe2fa0680039a"),     # ISO max accel, raw 2000
    (-3.5, False, False, 7, "01ffe04a868007c8"),    # ISO max brake, raw -3500
    (-1.024, True, False, 5, "01ffe18006841503"),   # standstill hold, raw -1024 + stop bits
    (-0.001, False, False, 9, "01ffe1ffe68009b0"),  # latched hold, raw -1
    (0.0, False, True, 11, "01ffe20006804b4c"),     # resume unlatch pulse
  ])
  def test_crz_info_engaged_golden_bytes(self, packer, accel, stopping, unlatching, counter, expected):
    dat = mazdacan.create_acc_command(packer, 0, counter, accel, True, False, stopping, unlatching)[1]
    assert dat.hex() == expected

  def test_crz_info_accel_encoding_and_checksum(self, packer):
    # the packed command must round-trip at the 0.001 factor and carry a valid masked-bit
    # checksum over the whole command window, stop bits set or not
    for raw in range(-3500, 2001, 137):
      for stopping in (False, True):
        dat = mazdacan.create_acc_command(packer, 0, raw % 16, raw / 1000.0, True, False, stopping, False)[1]
        assert decode_accel_cmd_raw(dat) == raw
        assert dat[7] == crz_info_reference_checksum(dat)
        assert bool(dat[5] & 0x04) == stopping
        assert bool(dat[6] & 0x10) == stopping

  @pytest.mark.parametrize(("long_active", "acc_available", "gap", "has_lead", "phase", "acc_active_2", "expected"), [
    (False, False, 0, False, 0, False, "0201010000000000"),  # standby
    (False, True, 2, False, 0, False, "02010b0000000000"),   # MRCC armed, SET allowed
    (True, True, 2, True, 1, True, "0a018b2000001000"),      # engaged, cruise, no lead
    (True, True, 2, True, 2, True, "0a018b4000001000"),      # engaged, following a lead
    (True, True, 2, True, 3, True, "0a018b6000001000"),      # stop-and-go hold (near phase)
    (True, True, 2, True, 4, True, "0a018b8000001000"),      # stop-and-go hold (far phase)
    (True, True, 2, True, 3, False, "0a018b6000000000"),     # relaxed hold, ACC_ACTIVE_2 drops
    (True, True, 1, True, 2, True, "0a01874000001000"),      # driver gap 1 mirrored to the dash
  ])
  def test_crz_ctrl_golden_bytes(self, packer, long_active, acc_available, gap, has_lead, phase, acc_active_2, expected):
    dat = mazdacan.create_crz_ctrl(packer, 0, long_active, acc_available, gap, has_lead, phase, acc_active_2)[1]
    assert dat.hex() == expected

  def test_radar_frames_match_stock(self):
    expected = [
      (0x499, "0008c00000000000"),
      (0x361, "fff7fefe1fc00080"),
      (0x362, "fff7fefe1fc78c80"),
      (0x363, "fff7fefe1fc00000"),
      (0x364, "fff7fefe1fc00000"),
      (0x365, "fff7fe7ffbff3fc0"),
      (0x366, "fff7fe7ffbff3fc0"),
    ]
    frames = mazdacan.create_radar_frames(0, 0, None)
    assert [(f.address, f.dat.hex()) for f in frames] == expected

  def test_radar_frames_counter_and_lead_track(self):
    frames = mazdacan.create_radar_frames(2, 15, (mazdacan.LEAD_TRACK_DIST, 0.))
    assert all(f.src == 2 for f in frames)
    # counter stamps the low nibble of the last byte on every track
    assert [f.dat[7] & 0x0f for f in frames[1:]] == [15] * 6
    tracks = {f.address: f.dat.hex() for f in frames}
    assert tracks[0x364] == "0a4000001dc0000f"

  def test_lead_track_at_template_range_is_the_capture(self):
    assert mazdacan.create_lead_track(mazdacan.LEAD_TRACK_DIST, 0.) == mazdacan.LEAD_TRACK_TEMPLATE

  @pytest.mark.parametrize("d_rel,v_rel", [
    (0., 0.), (6.5, 1.5), (10.25, -2.0), (29.4, 2.9375), (255.875, 63.9375), (400., 100.), (5., -80.),
  ])
  def test_lead_track_round_trips_through_the_dbc(self, d_rel, v_rel):
    dat = mazdacan.create_lead_track(d_rel, v_rel)
    cp = CANParser("mazda_2017", [("RADAR_TRACK_364", float("nan"))], 0)
    cp.update([(0, [(0x364, dat, 0)])])
    vl = cp.vl["RADAR_TRACK_364"]
    assert vl["DIST_OBJ"] == pytest.approx(min(max(d_rel, 0.), 255.875), abs=0.0625)
    assert vl["RELV_OBJ"] == pytest.approx(min(max(v_rel, -64.), 63.9375), abs=0.0625)
    # the bits outside the two fields we drive stay exactly as captured
    assert dat[1] & 0x0f == mazdacan.LEAD_TRACK_TEMPLATE[1] & 0x0f
    assert dat[2] == mazdacan.LEAD_TRACK_TEMPLATE[2]
    assert dat[4] & 0x1f == mazdacan.LEAD_TRACK_TEMPLATE[4] & 0x1f
    assert dat[5:] == mazdacan.LEAD_TRACK_TEMPLATE[5:]


class TestStandstillHold:

  @pytest.fixture
  def sm(self):
    return StandstillHold()

  @staticmethod
  def run(sm, frames, **kwargs):
    defaults = dict(long_active=True, stopping=False, standstill=False, plan_accel=-1.024,
                    brake_hold=False, lead_visible=True)
    defaults.update(kwargs)
    for _ in range(frames):
      sm.update(**defaults)
    return sm

  def test_holds_while_the_plan_is_stopping(self, sm):
    self.run(sm, 1)
    assert not sm.holding
    self.run(sm, 1, stopping=True)
    assert sm.holding and sm.stop_bits and sm.acc_active_2
    assert sm.ctrl_phase() == 3
    # arriving at a standstill changes nothing: the plan is still asking for the brakes
    self.run(sm, 500, stopping=True, standstill=True)
    assert sm.holding and sm.stop_bits

  def test_hold_never_relaxes_on_its_own(self, sm):
    # the creep-into-the-lead regression: without the car taking the hold over, the command
    # must stay on the plan's brake no matter how long the stop lasts
    self.run(sm, 1, stopping=True)
    self.run(sm, int(30.0 / DT_CTRL), stopping=True, standstill=True)
    assert sm.holding and sm.stop_bits and sm.acc_active_2
    assert not sm.car_has_hold

  def test_relax_follows_the_car_taking_the_hold(self, sm):
    self.run(sm, 1, stopping=True)
    self.run(sm, 10, stopping=True, standstill=True)
    assert not sm.car_has_hold
    self.run(sm, 1, stopping=True, standstill=True, brake_hold=True)
    # stop bits and ACC_ACTIVE_2 drop with the command, together, exactly as stock does
    assert sm.car_has_hold and not sm.stop_bits and not sm.acc_active_2
    # and it is not a latch: if the car lets go, we brake again
    self.run(sm, 1, stopping=True, standstill=True, brake_hold=False)
    assert not sm.car_has_hold and sm.stop_bits and sm.acc_active_2

  def test_released_when_the_plan_asks_to_move(self, sm):
    self.run(sm, 1, stopping=True)
    self.run(sm, 500, stopping=True, standstill=True, brake_hold=True)
    assert sm.holding
    self.run(sm, 1, standstill=True, plan_accel=0.1)
    assert not sm.holding and not sm.car_has_hold
    assert sm.resume_unlatching
    assert sm.ctrl_phase() == 2

  def test_release_holds_for_as_long_as_the_plan_wants_to_move(self, sm):
    # the failed-resume regression: no release window to run out from under the plan
    self.run(sm, 1, stopping=True)
    self.run(sm, 100, stopping=True, standstill=True)
    self.run(sm, int(5.0 / DT_CTRL), standstill=True, plan_accel=0.4)
    assert not sm.holding and not sm.stop_bits

  def test_hold_comes_back_if_the_plan_changes_its_mind(self, sm):
    self.run(sm, 1, stopping=True)
    self.run(sm, 100, stopping=True, standstill=True)
    self.run(sm, 5, standstill=True, plan_accel=0.2)
    assert not sm.holding
    self.run(sm, 1, stopping=True, standstill=True, plan_accel=-1.0)
    assert sm.holding and sm.stop_bits

  def test_unlatch_pulses_once_at_the_release(self, sm):
    self.run(sm, 1, stopping=True)
    self.run(sm, 100, stopping=True, standstill=True)
    assert not sm.resume_unlatching
    self.run(sm, 1, standstill=True, plan_accel=0.1)
    assert sm.resume_unlatching
    self.run(sm, RESUME_UNLATCH_FRAMES, standstill=True, plan_accel=0.1)
    assert not sm.resume_unlatching

  def test_long_disengage_resets(self, sm):
    self.run(sm, 1, stopping=True)
    self.run(sm, 100, stopping=True, standstill=True, brake_hold=True)
    self.run(sm, 1, long_active=False)
    assert not sm.holding and not sm.car_has_hold and not sm.stop_bits

  def test_stop_abort_releases(self, sm):
    self.run(sm, 1, stopping=True)
    assert sm.holding
    # lead speeds up again before the car reaches standstill
    self.run(sm, 1, stopping=False, plan_accel=0.3)
    assert not sm.holding

  def test_lead_follows_only_a_steady_state(self, sm):
    # a lead is adopted once leadVisible has held for the debounce window, not before
    self.run(sm, LEAD_DEBOUNCE_FRAMES - 1, lead_visible=True)
    assert not sm.radar_has_lead() and sm.ctrl_phase() == 1
    self.run(sm, 1, lead_visible=True)
    assert sm.radar_has_lead() and sm.ctrl_phase() == 2
    # and dropped the same way
    self.run(sm, LEAD_DEBOUNCE_FRAMES - 1, lead_visible=False)
    assert sm.radar_has_lead()
    self.run(sm, 1, lead_visible=False)
    assert not sm.radar_has_lead()

  def test_lead_flicker_never_reaches_the_bus(self, sm):
    # the measured failure: a marginal 120 m vision lead toggled leadVisible 6 times in 1.4 s
    # (route 6bb2dc61c4 t+400); none of it may reach RADAR_HAS_LEAD or the track slot
    for frames, visible in ((15, True), (5, False), (7, True), (13, False), (10, True)):
      self.run(sm, frames, lead_visible=visible)
      assert not sm.radar_has_lead(), "a flickering lead leaked through the debounce"

  def test_disengage_resets_the_lead(self, sm):
    self.run(sm, 2 * LEAD_DEBOUNCE_FRAMES, lead_visible=True)
    assert sm.radar_has_lead()
    self.run(sm, 1, long_active=False)
    assert not sm.radar_has_lead()


def _mock_cc(long_active=True, accel=0.5, long_state=None, standstill=False, gas=False, override=False,
             resume=False, lead_visible=True, gap=2, available=True,
             stock_radar_alive=False, fsc_settled=True, handback=False, cruise_engaged=False,
             enabled=None, lead_d_rel=12.0, lead_v_rel=0.0, brake_hold=False):
  # openpilot is enabled whenever it is longitudinally active; a gas override is the case
  # where it stays enabled with longActive low
  enabled = long_active if enabled is None else enabled
  out = SimpleNamespace(standstill=standstill, gasPressed=gas,
                        cruiseState=SimpleNamespace(available=available, enabled=cruise_engaged))
  actuators = SimpleNamespace(accel=accel, longControlState=long_state)
  cruise = SimpleNamespace(resume=resume, override=override, cancel=False)
  hud = SimpleNamespace(leadVisible=lead_visible, leadDistanceBars=gap)
  cc = SimpleNamespace(enabled=enabled, longActive=long_active, actuators=actuators,
                       cruiseControl=cruise, hudControl=hud)
  cc_sp = SimpleNamespace(stockEcuHandBack=handback,
                          leadOne=SimpleNamespace(dRel=lead_d_rel, vRel=lead_v_rel))
  cs = SimpleNamespace(out=out, resume_button=0, brake_hold=brake_hold,
                       stock_radar_alive=stock_radar_alive, fsc_settled=fsc_settled)
  return cc, cc_sp, cs


@pytest.fixture
def cc():
  CP = CarInterface.get_params(CAR.MAZDA_CX5_2022, {0: {}, 1: {}, 2: {}}, [], alpha_long=True,
                               is_release=False, docs=False)
  CP_SP = CarInterface.get_params_sp(CP, CAR.MAZDA_CX5_2022, {0: {}, 1: {}, 2: {}}, [], True, False, False)
  assert CP.openpilotLongitudinalControl
  return CarController({Bus.pt: "mazda_2017"}, CP, CP_SP)


def _long_frames(sends):
  """(ACCEL_CMD raw, CRZ_INFO.ACC_ACTIVE, CRZ_CTRL.CRZ_ACTIVE) from a bus 0 emission, or None."""
  info = next((d for a, d, b in sends if a == 0x21b and b == 0), None)
  ctrl = next((d for a, d, b in sends if a == 0x21c and b == 0), None)
  if info is None:
    return None
  cp = CANParser("mazda_2017", [("CRZ_INFO", float("nan")), ("CRZ_CTRL", float("nan"))], 0)
  cp.update([(0, [(0x21b, info, 0), (0x21c, ctrl, 0)])])
  return decode_accel_cmd_raw(info), cp.vl["CRZ_INFO"]["ACC_ACTIVE"], cp.vl["CRZ_CTRL"]["CRZ_ACTIVE"]


def _lead_track(dat):
  """(DIST_OBJ, RELV_OBJ) decoded from a 0x364 track frame."""
  cp = CANParser("mazda_2017", [("RADAR_TRACK_364", float("nan"))], 0)
  cp.update([(0, [(0x364, dat, 0)])])
  return cp.vl["RADAR_TRACK_364"]["DIST_OBJ"], cp.vl["RADAR_TRACK_364"]["RELV_OBJ"]


def _step(cc, **kw):
  kw.setdefault("long_state", structs.CarControl.Actuators.LongControlState.pid)
  control, control_sp, carstate = _mock_cc(**kw)
  sends = cc.update_longitudinal(control, control_sp, carstate)
  cc.frame += 1
  return sends


class TestLongitudinalIntegration:
  """Drives the real CarController.update_longitudinal through an engage -> cruise -> stop ->
  hold -> resume timeline and checks the emitted CAN, not just the state machine in isolation."""

  def test_engaged_frame_rates_and_counters(self, cc):
    long = structs.CarControl.Actuators.LongControlState
    crz_info = crz_ctrl = radar_static = tester = 0
    for _ in range(100):  # 1 s at 100 Hz
      sends = _step(cc, long_state=long.pid, accel=1.0, gap=2)
      addrs = [a for a, _, _ in sends]
      buses = {a: [] for a, _, _ in sends}
      for a, _, b in sends:
        buses[a].append(b)
      crz_info += addrs.count(0x21b)
      crz_ctrl += addrs.count(0x21c)
      radar_static += addrs.count(0x499)
      tester += sum(1 for a, _, _ in sends if a == 0x764)
      # CRZ_INFO/CRZ_CTRL, when emitted, always go to both bus 0 and bus 2
      if 0x21b in buses:
        assert sorted(buses[0x21b]) == [0, 2]
        assert sorted(buses[0x21c]) == [0, 2]

    # 100 Hz loop: long msgs at 50 Hz (x2 buses), radar at 10 Hz (x2), tester at 2 Hz
    assert crz_info == crz_ctrl == 100    # 50 frames x 2 buses
    assert radar_static == 20             # 10 frames x 2 buses
    assert tester == 2                    # 2 Hz, single bus
    assert cc.long_counter == 50 and cc.radar_counter == 10

  def test_gap_setting_mirrors_driver(self, cc):
    for gap in (1, 2, 3):
      cc.frame = 0  # force emission on the first step
      sends = _step(cc, gap=gap, long_state=structs.CarControl.Actuators.LongControlState.pid)
      ctrl = next(dat for a, dat, b in sends if a == 0x21c and b == 0)
      cp = CANParser("mazda_2017", [("CRZ_CTRL", float("nan"))], 0)
      cp.update([(0, [(0x21c, ctrl, 0)])])
      assert cp.vl["CRZ_CTRL"]["DISTANCE_SETTING"] == gap

  def test_stop_emits_hold_then_relaxes(self, cc):
    long = structs.CarControl.Actuators.LongControlState

    def accel_cmd(sends):
      dat = next((d for a, d, b in sends if a == 0x21b and b == 0), None)
      return None if dat is None else decode_accel_cmd_raw(dat)

    # approach the stop
    for _ in range(int(0.5 / 0.01)):
      _step(cc, long_state=long.stopping, accel=-1.5, standstill=False)
    # hold at a standstill: the command is the plan's own and must not relax on its own, no
    # matter how long the stop lasts (the creep-into-the-lead regression)
    cmds = []
    for _ in range(int(30.0 / 0.01)):
      cmd = accel_cmd(_step(cc, long_state=long.stopping, accel=-1.024, standstill=True))
      if cmd is not None:
        cmds.append(cmd)
    settled = cmds[len(cmds) // 2:]
    assert settled and set(settled) == {-1024}, f"hold command drifted off the plan: {sorted(set(settled))}"

    # once the body ECU takes the hold over, stock stops asking for the brakes and so do we
    relaxed = []
    for _ in range(int(1.0 / 0.01)):
      cmd = accel_cmd(_step(cc, long_state=long.stopping, accel=-1.024, standstill=True,
                            brake_hold=True))
      if cmd is not None:
        relaxed.append(cmd)
    assert relaxed and set(relaxed) == {round(CarControllerParams.ACCEL_HOLD_LATCHED * 1000)}

  def test_gas_override_stays_engaged(self, cc):
    """A gas press is an override, not a disengagement. The command goes to zero as on every
    other port, but the engaged bits stay set the way Honda drives CONTROL_ON off CC.enabled.
    Clearing them mid-decel takes the PCM out of ACC mode (docs/mazda-gas-override.md)."""
    long = structs.CarControl.Actuators.LongControlState

    # braking hard, then the driver taps the gas
    for _ in range(200):
      _step(cc, long_state=long.pid, accel=-2.0, cruise_engaged=True)
    assert cc.accel_last == pytest.approx(-2.0)

    cmds = []
    for _ in range(100):  # 1 s of override
      sends = _step(cc, long_active=False, enabled=True, long_state=long.off, accel=0.,
                    gas=True, override=True, cruise_engaged=True)
      frame = _long_frames(sends)
      if frame is not None:
        cmds.append(frame)

    raw, acc_active, crz_active = zip(*cmds, strict=True)
    assert all(acc_active), "ACC_ACTIVE dropped during a gas override"
    assert all(crz_active), "CRZ_ACTIVE dropped during a gas override"
    assert set(raw) == {0}, f"command should be zero through the override, got {sorted(set(raw))}"

  def test_command_slew_is_rate_limited(self, cc):
    """The plan can step; the wire should not. Windup is limited tightly because dumping the
    brake in one frame is what the driver feels, winddown loosely so braking is never delayed."""
    long = structs.CarControl.Actuators.LongControlState
    for _ in range(200):
      _step(cc, long_state=long.pid, accel=-2.0, cruise_engaged=True)
    assert cc.accel_last == pytest.approx(-2.0)

    # plan jumps straight to +1.0: the command must ramp, not step
    prev = cc.accel_last
    for _ in range(5):
      _step(cc, long_state=long.pid, accel=1.0, cruise_engaged=True)
      assert cc.accel_last - prev == pytest.approx(CarControllerParams.ACCEL_WINDUP_LIMIT, abs=1e-6)
      prev = cc.accel_last

    # and the other way, at the looser winddown limit
    for _ in range(200):
      _step(cc, long_state=long.pid, accel=1.0, cruise_engaged=True)
    prev = cc.accel_last
    for _ in range(5):
      _step(cc, long_state=long.pid, accel=-3.0, cruise_engaged=True)
      assert cc.accel_last - prev == pytest.approx(CarControllerParams.ACCEL_WINDDOWN_LIMIT, abs=1e-6)
      prev = cc.accel_last

  def test_accel_last_tracks_the_wire_not_the_plan(self, cc):
    # update() reports accel_last as actuatorsOutput.accel, the way Toyota, Ford and Honda
    # report the value they sent. It must be the wire value, clip and hold included.
    long = structs.CarControl.Actuators.LongControlState

    # a plan beyond the envelope is reported clipped, not as asked
    for _ in range(400):
      sends = _step(cc, long_state=long.pid, accel=-9.0, cruise_engaged=True)
    assert cc.accel_last == pytest.approx(CarControllerParams.ACCEL_MIN)
    frame = _long_frames(sends)
    if frame is not None:
      assert frame[0] == round(cc.accel_last * 1000)

    # the standstill hold is the plan's own command, and that is what gets reported
    for _ in range(int(0.5 / 0.01)):
      _step(cc, long_state=long.stopping, accel=-1.5, standstill=True, cruise_engaged=True)
    assert cc.accel_last == pytest.approx(-1.5)

    # through a gas override we report the zero we actually send
    for _ in range(10):
      _step(cc, long_active=False, enabled=True, long_state=long.off, accel=0., gas=True,
            override=True, cruise_engaged=True)
    assert cc.accel_last == 0.

  def test_gas_from_standstill_hold_releases_the_brake(self, cc):
    # gas out of a hold is a resume, not a slow release: the hold command must go straight to
    # zero rather than ramping off at the cruising override rate
    long = structs.CarControl.Actuators.LongControlState
    for _ in range(int(3.0 / 0.01)):
      _step(cc, long_state=long.stopping, accel=-1.5, standstill=True, cruise_engaged=True)
    assert cc.accel_last < -0.5, "never reached the standstill hold"

    for _ in range(20):
      _step(cc, long_active=False, enabled=True, long_state=long.off, accel=0., gas=True,
            override=True, standstill=True, cruise_engaged=True)
    assert cc.accel_last == 0., f"hold not released for the driver's gas: {cc.accel_last}"

  def test_lead_track_follows_the_measured_lead(self, cc):
    # a frozen track is what latches the camera's SCBS fault, so the range we advertise has to
    # move with the lead we are actually following
    long = structs.CarControl.Actuators.LongControlState
    # let the lead debounce adopt the visible lead before sampling the track
    for _ in range(LEAD_DEBOUNCE_FRAMES):
      _step(cc, long_state=long.pid, accel=0.5, lead_visible=True, lead_d_rel=20.0, lead_v_rel=-1.5)
    seen = []
    for i in range(60):
      sends = _step(cc, long_state=long.pid, accel=0.5, lead_visible=True,
                    lead_d_rel=20.0 - 0.1 * i, lead_v_rel=-1.5)
      track = next((d for a, d, b in sends if a == 0x364 and b == 0), None)
      if track is not None:
        seen.append(_lead_track(track))
    assert len(seen) > 1
    dists = [d for d, _ in seen]
    assert all(a > b for a, b in zip(dists, dists[1:], strict=False)), f"range did not close with the lead: {dists}"
    assert all(v == pytest.approx(-1.5, abs=0.0625) for _, v in seen)

  def test_hold_fabricates_a_lead_but_drops_it_on_release(self, cc):
    # with no lead in view the hold still needs something to hold against, but carrying that
    # fabricated object through the release is what the camera latches on
    long = structs.CarControl.Actuators.LongControlState

    def tracks(sends):
      return [d for a, d, b in sends if a == 0x364 and b == 0]

    held = []
    for _ in range(int(3.0 / 0.01)):
      held += tracks(_step(cc, long_state=long.stopping, accel=-1.5, standstill=True,
                           lead_visible=False, cruise_engaged=True))
    assert held
    assert all(_lead_track(d)[0] == pytest.approx(mazdacan.LEAD_TRACK_DIST) for d in held)

    released = []
    for _ in range(50):
      released += tracks(_step(cc, long_state=long.pid, accel=0.3, standstill=True,
                               lead_visible=False, cruise_engaged=True))
    assert released
    empty = mazdacan.RADAR_TRACK_MSGS[0x364]
    assert all(d[:7] == empty[:7] for d in released), \
      f"fabricated lead survived the release: {released[0].hex()}"

  def test_resume_asks_while_the_plan_wants_to_move_and_the_car_has_not(self, cc):
    # the RES press has to outlast cruiseState.standstill, which drops for ~3 s after a press,
    # so it is keyed on the car actually still being stopped
    control, _, carstate = _mock_cc(standstill=True, accel=0.3)
    assert cc.resume_requested(control, carstate)

    # plan still braking: no press, even though the car is sitting in a hold
    control, _, carstate = _mock_cc(standstill=True, accel=-1.024)
    assert not cc.resume_requested(control, carstate)

    # car is rolling: the hold is gone, stop asking
    control, _, carstate = _mock_cc(standstill=False, accel=0.3)
    assert not cc.resume_requested(control, carstate)

    # not longitudinally active: never our press to send
    control, _, carstate = _mock_cc(long_active=False, enabled=True, standstill=True, accel=0.3)
    assert not cc.resume_requested(control, carstate)

  def test_resume_matches_the_hold_release(self, cc):
    # the press and the release run off the same condition, so the body is never asked to let
    # go while we are still commanding the brake
    long = structs.CarControl.Actuators.LongControlState
    for _ in range(200):
      _step(cc, long_state=long.stopping, accel=-1.024, standstill=True, cruise_engaged=True)
    control, _, carstate = _mock_cc(standstill=True, accel=-1.024)
    assert cc.stop_and_go.holding and not cc.resume_requested(control, carstate)

    _step(cc, long_state=long.pid, accel=0.3, standstill=True, cruise_engaged=True)
    control, _, carstate = _mock_cc(standstill=True, accel=0.3)
    assert not cc.stop_and_go.holding and cc.resume_requested(control, carstate)

  def test_gas_pedal_without_cruise_stays_disengaged(self, cc):
    # gas pressed while openpilot is not enabled must not advertise an engaged ACC
    off = structs.CarControl.Actuators.LongControlState.off
    cc.frame = 0
    sends = _step(cc, long_active=False, enabled=False, long_state=off, gas=True, available=True)
    info = next(dat for a, dat, b in sends if a == 0x21b and b == 0)
    assert info.hex().startswith("01ffe2000480")  # armed-but-idle pattern, zero command

  def test_disengaged_emits_stock_patterns(self, cc):
    off = structs.CarControl.Actuators.LongControlState.off
    # main off, not available: the exact standby pattern the panda allowlists byte-for-byte
    cc.frame = 0
    sends = _step(cc, long_active=False, long_state=off, available=False)
    info = next(dat for a, dat, b in sends if a == 0x21b and b == 0)
    assert info.hex().startswith("01ffe3ffc000")
    # MRCC armed but not engaged: stock advertises ACC_SET_ALLOWED with a zero command
    cc.frame = 0
    sends = _step(cc, long_active=False, long_state=off, available=True)
    info = next(dat for a, dat, b in sends if a == 0x21b and b == 0)
    assert info.hex().startswith("01ffe2000480")


SESSION_PROG_DAT = bytes([0x02, 0x10, 0x02, 0, 0, 0, 0, 0])
SESSION_DFLT_DAT = bytes([0x02, 0x10, 0x01, 0, 0, 0, 0, 0])
TESTER_PRESENT_DAT = bytes([0x02, 0x3e, 0x80, 0, 0, 0, 0, 0])


class TestRadarSessionSequencing:
  """Boot teardown deferral and the ordered hand-back: what goes on the bus in each
  radar session state, driven through the real CarController.update_longitudinal."""

  def _step(self, cc, stock_radar_alive, fsc_settled, handback=False, cruise_engaged=False):
    off = structs.CarControl.Actuators.LongControlState.off
    return _step(cc, long_active=False, accel=0., long_state=off, lead_visible=False, available=False,
                 stock_radar_alive=stock_radar_alive, fsc_settled=fsc_settled,
                 handback=handback, cruise_engaged=cruise_engaged)

  @staticmethod
  def _uds(sends):
    return [dat for a, dat, b in sends if a == 0x764]

  @staticmethod
  def _synthetic(sends):
    return [a for a, _, _ in sends if a in (0x21b, 0x21c, 0x499)]

  def test_stock_state_is_silent(self, cc):
    # radar alive, gate not yet passed: nothing at all goes on the bus
    for _ in range(200):
      sends = self._step(cc, stock_radar_alive=True, fsc_settled=False)
      assert sends == []

  def test_boot_teardown_sequence(self, cc):
    # gate passes with the stock radar alive: programming-session requests at 2 Hz,
    # still no synthetic frames and no tester present
    for i in range(100):
      sends = self._step(cc, stock_radar_alive=True, fsc_settled=True)
      if i % CarControllerParams.RADAR_UDS_STEP == 0:
        assert self._uds(sends) == [SESSION_PROG_DAT]
      else:
        assert self._uds(sends) == []
      assert self._synthetic(sends) == []
    # radar goes quiet: synthetic frames + tester present take over, session requests stop
    saw_tester = False
    for _ in range(100):
      frame = cc.frame
      sends = self._step(cc, stock_radar_alive=False, fsc_settled=True)
      assert SESSION_PROG_DAT not in self._uds(sends)
      if frame % CarControllerParams.LONG_STEP == 0:
        assert len(self._synthetic(sends)) > 0
      saw_tester |= TESTER_PRESENT_DAT in self._uds(sends)
    assert saw_tester

  def test_handback_sequence(self, cc):
    # reach SILENCED
    self._step(cc, stock_radar_alive=False, fsc_settled=True)
    # hand-back requested: default-session requests at 2 Hz, tester present stops,
    # synthetic frames continue while the radar is still quiet
    saw_default = False
    for _ in range(100):
      frame = cc.frame
      sends = self._step(cc, stock_radar_alive=False, fsc_settled=True, handback=True)
      assert TESTER_PRESENT_DAT not in self._uds(sends)
      saw_default |= SESSION_DFLT_DAT in self._uds(sends)
      if frame % CarControllerParams.LONG_STEP == 0:
        assert len(self._synthetic(sends)) > 0
    assert saw_default
    # stock radar returns: everything stops
    for _ in range(200):
      sends = self._step(cc, stock_radar_alive=True, fsc_settled=True, handback=True)
      assert sends == []

  def test_handback_before_teardown_stops_everything(self, cc):
    # toggle-off while still waiting on the gate: no session ever entered, so no
    # hand-back traffic either
    self._step(cc, stock_radar_alive=True, fsc_settled=False)
    for _ in range(120):
      sends = self._step(cc, stock_radar_alive=True, fsc_settled=False, handback=True)
      assert sends == []

  def test_teardown_waits_for_stock_cruise_disengage(self, cc):
    # driver engaged stock MRCC before the gate passed (warm boot): hold the teardown
    for _ in range(120):
      sends = self._step(cc, stock_radar_alive=True, fsc_settled=True, cruise_engaged=True)
      assert sends == []
    # driver disengages: teardown proceeds
    cc.frame = 0
    sends = self._step(cc, stock_radar_alive=True, fsc_settled=True, cruise_engaged=False)
    assert SESSION_PROG_DAT in self._uds(sends)

  def test_s3_recovery_resilences(self, cc):
    # radar reappears mid-drive (dropped tester present, S3 timeout): re-request the session
    self._step(cc, stock_radar_alive=False, fsc_settled=True)
    cc.frame = CarControllerParams.RADAR_UDS_STEP  # align to a session-request frame
    sends = self._step(cc, stock_radar_alive=True, fsc_settled=True)
    assert SESSION_PROG_DAT in self._uds(sends)
    # and settles back to silenced once quiet again
    sends = self._step(cc, stock_radar_alive=False, fsc_settled=True)
    assert SESSION_PROG_DAT not in self._uds(sends)


class TestTjaMrccCleanup:
  @staticmethod
  def _controller(candidate=CAR.MAZDA_CX5_2022):
    fingerprint = {0: {}, 1: {}, 2: {}}
    CP = CarInterface.get_params(candidate, fingerprint, [], alpha_long=False, is_release=False, docs=False)
    CP_SP = CarInterface.get_params_sp(CP, candidate, fingerprint, [], False, False, False)
    return CarController({Bus.pt: "mazda_2017"}, CP, CP_SP)

  @staticmethod
  def _controls(*, cancel=False, resume=False):
    control = structs.CarControl()
    control.cruiseControl.cancel = cancel
    control.cruiseControl.resume = resume
    return control.as_reader(), structs.CarControlSP()

  @staticmethod
  def _state(*, tja=False, armed=False, active=False, raw_armed=None, counter=0, **buttons):
    if raw_armed is None:
      raw_armed = armed
    return SimpleNamespace(
      out=SimpleNamespace(
        vEgoRaw=12.0,
        steeringTorque=0,
        brakePressed=False,
        cruiseState=SimpleNamespace(available=armed, enabled=active),
      ),
      cruise_available=armed,
      mrcc_armed_raw=raw_armed,
      cam_lkas={"ERR_BIT_1": 0, "ERR_BIT_2": 0, "LINE_NOT_VISIBLE": 0, "BIT_1": 1},
      cam_laneinfo={
        "LINE_VISIBLE": 0, "LINE_NOT_VISIBLE": 1, "LANE_LINES": 1,
        "BIT1": 0, "BIT2": 0, "BIT3": 0, "NO_ERR_BIT": 0, "S1": 0, "S1_HBEAM": 0,
      },
      crz_btns_counter=counter,
      cancel_button=buttons.get("cancel_button", 0),
      resume_button=buttons.get("resume_button", 0),
      tja_button=int(tja),
      accel_button=buttons.get("accel_button", 0),
      decel_button=buttons.get("decel_button", 0),
      mrcc_button=buttons.get("mrcc_button", 0),
      lkas_allowed_speed=True,
    )

  def _step(self, controller, *, tja=False, armed=False, active=False, raw_armed=None, counter=None,
            advance_nanos=None, control=None, **buttons):
    if counter is None:
      counter = (getattr(controller, "_test_counter", -1) + 1) % 16
    if advance_nanos is None:
      advance_nanos = round(DT_CTRL * 1e9)
    controller._test_counter = counter
    controller._test_now = getattr(controller, "_test_now", 0) + advance_nanos
    CC, CC_SP = control or self._controls()
    sends = controller.update(
      CC, CC_SP,
      self._state(tja=tja, armed=armed, active=active, raw_armed=raw_armed, counter=counter, **buttons),
      controller._test_now,
    )[1]
    assert controller.tja_mrcc_tx_frames <= TJA_MRCC_MAX_TX_FRAMES
    return sends

  @staticmethod
  def _mrcc_off_payloads(sends):
    parser = CANParser("mazda_2017", [("CRZ_BTNS", 10)], 0)
    payloads = []
    for address, data, bus in sends:
      if address != 0x09d or bus != 0:
        continue
      parser.update([(0, [(address, data, bus)])])
      if parser.vl["CRZ_BTNS"]["BIT1"] == 0:
        payloads.append(data)
    return payloads

  @staticmethod
  def _button_payloads(sends):
    return [data for address, data, bus in sends if address == 0x09d and bus == 0]

  @staticmethod
  def _payload_counters(payloads):
    parser = CANParser("mazda_2017", [("CRZ_BTNS", 10)], 0)
    counters = []
    for data in payloads:
      parser.update([(0, [(0x09d, data, 0)])])
      counters.append(int(parser.vl["CRZ_BTNS"]["CTR"]))
    return counters

  def _start_owned_cleanup(self, controller, *, start_counter=0):
    self._step(controller, tja=False, armed=False, raw_armed=False, counter=start_counter)
    self._step(controller, tja=True, armed=True, raw_armed=True, counter=(start_counter + 1) % 16)
    sends = self._step(controller, tja=False, armed=True, raw_armed=True, counter=(start_counter + 2) % 16)
    assert not self._mrcc_off_payloads(sends)

  def _reach_first_deadline(self, controller, *, counter):
    step_nanos = round(DT_CTRL * 1e9)
    assert TJA_MRCC_FIRST_TX_DELAY_NANOS == 5 * step_nanos
    for _ in range(4):
      assert not self._mrcc_off_payloads(
        self._step(controller, armed=True, raw_armed=True, counter=counter, advance_nanos=step_nanos)
      )
    return self._step(controller, armed=True, raw_armed=True, counter=counter, advance_nanos=step_nanos)

  @pytest.mark.parametrize(("state", "prearmed", "active"), [
    ("off", False, False),
    ("armed", True, False),
    ("active", True, True),
  ])
  def test_cleanup_ownership_depends_on_pre_tja_state(self, state, prearmed, active):
    controller = self._controller()
    self._step(controller, armed=prearmed, active=active, raw_armed=prearmed, counter=0)
    self._step(controller, tja=True, armed=True, raw_armed=True, counter=1)
    self._step(controller, armed=True, raw_armed=True, counter=2)
    sends = self._reach_first_deadline(controller, counter=3)
    assert bool(self._mrcc_off_payloads(sends)) == (state == "off")

  def test_first_attempt_is_delayed_and_hold_is_capped_at_three_frames(self):
    controller = self._controller()
    self._start_owned_cleanup(controller)
    payloads = self._mrcc_off_payloads(self._reach_first_deadline(controller, counter=3))
    for counter in range(4, 12):
      payloads.extend(self._mrcc_off_payloads(
        self._step(controller, armed=True, raw_armed=True, counter=counter % 16)
      ))
    assert len(payloads) == 3
    assert controller.tja_mrcc_tx_frames == 3

  @pytest.mark.parametrize("button", [
    {"cancel_button": 1},
    {"resume_button": 1},
    {"accel_button": 1},
    {"decel_button": 1},
    {"mrcc_button": 1},
  ])
  def test_physical_driver_button_aborts_cleanup(self, button):
    controller = self._controller()
    self._start_owned_cleanup(controller)
    self._step(controller, armed=True, raw_armed=True, counter=3, **button)
    assert not controller.tja_mrcc_unarm_pending
    for counter in range(4, 12):
      assert not self._mrcc_off_payloads(
        self._step(controller, armed=True, raw_armed=True, counter=counter, **button)
      )

  def test_raw_off_stops_in_flight_cleanup_despite_filtered_brake_hold(self):
    controller = self._controller()
    self._start_owned_cleanup(controller)
    assert len(self._mrcc_off_payloads(self._reach_first_deadline(controller, counter=3))) == 1
    assert not self._mrcc_off_payloads(
      self._step(controller, armed=True, raw_armed=False, counter=4)
    )
    assert not controller.tja_mrcc_unarm_pending

  @pytest.mark.parametrize("spent_before_repress", [1, 2])
  def test_double_tja_preserves_ownership_and_cumulative_budget(self, spent_before_repress):
    controller = self._controller()
    self._start_owned_cleanup(controller)
    payloads = self._mrcc_off_payloads(self._reach_first_deadline(controller, counter=3))
    counter = 4
    if spent_before_repress == 2:
      payloads.extend(self._mrcc_off_payloads(
        self._step(controller, armed=True, raw_armed=True, counter=counter)
      ))
      counter += 1
    self._step(controller, tja=True, armed=True, raw_armed=True, counter=counter)
    counter += 1
    self._step(controller, tja=False, armed=True, raw_armed=True, counter=counter)
    counter += 1
    for follow_counter in range(counter, counter + 8):
      payloads.extend(self._mrcc_off_payloads(
        self._step(controller, armed=True, raw_armed=True, counter=follow_counter % 16)
      ))
    assert len(payloads) == 3
    assert controller.tja_mrcc_tx_frames == 3

  @pytest.mark.parametrize("action", ["cancel", "resume"])
  def test_op_command_before_first_tx_defers_until_fresh_counter(self, action):
    controller = self._controller()
    self._start_owned_cleanup(controller)
    op_command = self._controls(**{action: True})
    for _ in range(6):
      assert not self._mrcc_off_payloads(
        self._step(controller, armed=True, raw_armed=True, counter=2, control=op_command)
      )
    assert not self._mrcc_off_payloads(
      self._step(controller, armed=True, raw_armed=True, counter=2)
    )
    sends = self._step(controller, armed=True, raw_armed=True, counter=3)
    assert len(self._mrcc_off_payloads(sends)) == 1

  @pytest.mark.parametrize("action", ["cancel", "resume"])
  def test_op_command_after_first_tx_aborts_replacement_hold(self, action):
    controller = self._controller()
    self._start_owned_cleanup(controller)
    assert len(self._mrcc_off_payloads(self._reach_first_deadline(controller, counter=3))) == 1
    sends = self._step(
      controller, armed=True, raw_armed=True, counter=4,
      control=self._controls(**{action: True}),
    )
    assert not self._mrcc_off_payloads(sends)
    assert not controller.tja_mrcc_unarm_pending

  def test_counter_jump_after_hold_starts_aborts_without_budget_reset(self):
    controller = self._controller()
    self._start_owned_cleanup(controller)
    assert len(self._mrcc_off_payloads(self._reach_first_deadline(controller, counter=3))) == 1
    assert not self._mrcc_off_payloads(
      self._step(controller, armed=True, raw_armed=True, counter=6)
    )
    assert not controller.tja_mrcc_unarm_pending
    assert controller.tja_mrcc_tx_frames == 1

  def test_cleanup_counter_wrap_is_consecutive(self):
    controller = self._controller()
    self._start_owned_cleanup(controller, start_counter=12)
    payloads = self._mrcc_off_payloads(self._reach_first_deadline(controller, counter=14))
    payloads.extend(self._mrcc_off_payloads(
      self._step(controller, armed=True, raw_armed=True, counter=15)
    ))
    payloads.extend(self._mrcc_off_payloads(
      self._step(controller, armed=True, raw_armed=True, counter=0)
    ))
    assert self._payload_counters(payloads) == [15, 0, 1]

  @pytest.mark.parametrize("button", [
    {"mrcc_button": 1},
    {"cancel_button": 1},
    {"resume_button": 1},
    {"accel_button": 1},
    {"decel_button": 1},
  ])
  def test_driver_input_wins_at_exact_first_tx_deadline(self, button):
    controller = self._controller()
    self._start_owned_cleanup(controller)
    step_nanos = round(DT_CTRL * 1e9)
    for _ in range(4):
      self._step(controller, armed=True, raw_armed=True, counter=3, advance_nanos=step_nanos)
    sends = self._step(
      controller, armed=True, raw_armed=True, counter=3,
      advance_nanos=step_nanos, **button,
    )
    assert not self._mrcc_off_payloads(sends)
    assert not controller.tja_mrcc_unarm_pending

  def test_tja_repress_wins_at_exact_first_tx_deadline(self):
    controller = self._controller()
    self._start_owned_cleanup(controller)
    step_nanos = round(DT_CTRL * 1e9)
    for _ in range(4):
      self._step(controller, armed=True, raw_armed=True, counter=3, advance_nanos=step_nanos)
    sends = self._step(
      controller, tja=True, armed=True, raw_armed=True, counter=3,
      advance_nanos=step_nanos,
    )
    assert not self._mrcc_off_payloads(sends)
    assert controller.tja_mrcc_unarm_pending
    assert controller.tja_mrcc_release_counter is None

  def test_brief_raw_dropout_does_not_claim_prearmed_mrcc(self):
    controller = self._controller()
    self._step(controller, armed=True, raw_armed=True, counter=0)
    for counter in range(1, 4):
      self._step(controller, armed=True, raw_armed=False, counter=counter)
    self._step(controller, tja=True, armed=True, raw_armed=True, counter=4)
    self._step(controller, armed=True, raw_armed=True, counter=5)
    sends = self._reach_first_deadline(controller, counter=6)
    assert not self._mrcc_off_payloads(sends)
    assert not controller.tja_mrcc_unarm_pending

  def test_final_cleanup_frame_suppresses_icbm(self):
    controller = self._controller()
    self._start_owned_cleanup(controller)
    self._reach_first_deadline(controller, counter=3)
    self._step(controller, armed=True, raw_armed=True, counter=4)

    CC, CC_SP = self._controls()
    CC_SP.intelligentCruiseButtonManagement.sendButton = (
      structs.IntelligentCruiseButtonManagement.SendButtonState.increase
    )
    controller.last_button_frame = -10_000
    sends = self._step(
      controller, armed=True, raw_armed=True, counter=5, control=(CC, CC_SP),
    )
    assert len(self._mrcc_off_payloads(sends)) == 1
    assert len(self._button_payloads(sends)) == 1

  def test_exhausted_budget_cannot_leak_into_later_armed_tja(self):
    controller = self._controller()
    self._start_owned_cleanup(controller)
    self._reach_first_deadline(controller, counter=3)
    self._step(controller, armed=True, raw_armed=True, counter=4)
    self._step(controller, armed=True, raw_armed=True, counter=5)
    assert controller.tja_mrcc_tx_frames == 3

    self._step(controller, tja=True, armed=True, raw_armed=True, counter=6)
    self._step(controller, armed=True, raw_armed=True, counter=7)
    for counter in range(8, 16):
      assert not self._mrcc_off_payloads(
        self._step(controller, armed=True, raw_armed=True, counter=counter)
      )
    assert controller.tja_mrcc_tx_frames == 3

  @pytest.mark.parametrize("candidate,should_suppress", [
    (CAR.MAZDA_CX5_2022, True),
    (CAR.MAZDA_CX5, False),
  ])
  def test_icbm_tja_hold_suppression_is_platform_scoped(self, candidate, should_suppress):
    controller = self._controller(candidate)
    CC, CC_SP = self._controls()
    CC_SP.intelligentCruiseButtonManagement.sendButton = (
      structs.IntelligentCruiseButtonManagement.SendButtonState.increase
    )
    controller.last_button_frame = -10_000
    sends = self._step(
      controller, tja=True, armed=False, raw_armed=False, counter=0,
      control=(CC, CC_SP),
    )
    assert bool(self._button_payloads(sends)) is not should_suppress

  def test_non_target_and_long_repetition_never_leave_stale_ownership(self):
    non_target = self._controller(CAR.MAZDA_CX5)
    for counter in range(40):
      sends = self._step(
        non_target,
        tja=counter % 4 == 1,
        armed=counter % 4 != 0,
        raw_armed=counter % 4 != 0,
        counter=counter % 16,
      )
      assert not self._mrcc_off_payloads(sends)
    assert not non_target.tja_mrcc_unarm_pending

    target = self._controller()
    self._start_owned_cleanup(target)
    for counter in range(3, 40):
      self._step(target, armed=False, raw_armed=False, counter=counter % 16)
    assert not target.tja_mrcc_unarm_pending
