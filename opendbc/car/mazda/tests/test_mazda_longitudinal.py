"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The accel command on the wire, driven through the real CarController.update_longitudinal:
frame rates, the slew limits, the standstill hold and its relax, the release grammar and the
RESUME_UNLATCHING pulse, the breakaway ramp, the gas override and the disengaged patterns.
"""
import pytest

from opendbc.car import DT_CTRL
from opendbc.car.mazda.longitudinal import (BREAKAWAY_FRAMES, RELEASE_DEBOUNCE_FRAMES, RESUME_REPULSE_FRAMES,
                                            RESUME_UNLATCH_LATCHED_FRAMES)
from opendbc.car.mazda.tests.conftest import (CRZ_CTRL, CRZ_INFO, RADAR_STATIC, RADAR_UDS, LongCtrlState, accel_cmd_raw,
                                              crz_info, frame, parse_frame, step_long)
from opendbc.car.mazda.values import CarControllerParams

STOPPING = LongCtrlState.stopping
OFF = LongCtrlState.off
LEAD_4M = dict(lead_visible=True, lead_d_rel=4.0, lead_v_rel=0.0)


def seconds(t):
  return int(t / DT_CTRL)


def crz_info_rows(cc, cs, n, **kwargs):
  """Step n frames and collect (ACCEL_CMD raw, STOPPING, RESUME_UNLATCHING) from every bus 0 CRZ_INFO."""
  rows = []
  for _ in range(n):
    dat = frame(step_long(cc, cs, **kwargs), CRZ_INFO)
    if dat is not None:
      rows.append(crz_info(dat))
  return rows


def long_frames(sends):
  """(ACCEL_CMD raw, CRZ_INFO.ACC_ACTIVE, CRZ_CTRL.CRZ_ACTIVE) from a bus 0 emission, or None."""
  info, ctrl = frame(sends, CRZ_INFO), frame(sends, CRZ_CTRL)
  if info is None:
    return None
  return crz_info(info)[0], parse_frame(CRZ_INFO, info)["ACC_ACTIVE"], parse_frame(CRZ_CTRL, ctrl)["CRZ_ACTIVE"]


def test_engaged_frame_rates_and_counters(cc, cs):
  crz_info_n = crz_ctrl_n = radar_static = tester = 0
  for _ in range(100):  # 1 s at 100 Hz
    sends = step_long(cc, cs, accel=1.0, gap=2)
    addrs = [a for a, _, _ in sends]
    buses = {a: [] for a, _, _ in sends}
    for a, _, b in sends:
      buses[a].append(b)
    crz_info_n += addrs.count(CRZ_INFO)
    crz_ctrl_n += addrs.count(CRZ_CTRL)
    radar_static += addrs.count(RADAR_STATIC)
    tester += addrs.count(RADAR_UDS)
    # CRZ_INFO/CRZ_CTRL, when emitted, always go to both bus 0 and bus 2
    if CRZ_INFO in buses:
      assert sorted(buses[CRZ_INFO]) == [0, 2]
      assert sorted(buses[CRZ_CTRL]) == [0, 2]

  # 100 Hz loop: long msgs at 50 Hz (x2 buses), radar at 10 Hz (x2), tester at 2 Hz
  assert crz_info_n == crz_ctrl_n == 100  # 50 frames x 2 buses
  assert radar_static == 20  # 10 frames x 2 buses
  assert tester == 2  # 2 Hz, single bus
  assert cc.long_counter == 50 and cc.radar_counter == 10


@pytest.mark.parametrize("gap", [1, 2, 3])
def test_gap_setting_mirrors_driver(cc, cs, gap):
  cc.frame = 0  # force emission on the first step
  sends = step_long(cc, cs, gap=gap)
  assert parse_frame(CRZ_CTRL, frame(sends, CRZ_CTRL))["DISTANCE_SETTING"] == gap


def test_stop_emits_hold_then_relaxes(cc, cs):
  # approach the stop
  for _ in range(seconds(0.5)):
    step_long(cc, cs, long_state=STOPPING, accel=-1.5, standstill=False)
  # hold at a standstill: the command is the plan's own and must not relax on its own, no
  # matter how long the stop lasts (the creep-into-the-lead regression)
  cmds = []
  for _ in range(seconds(30.0)):
    cmd = accel_cmd_raw(step_long(cc, cs, long_state=STOPPING, accel=-1.024, standstill=True))
    if cmd is not None:
      cmds.append(cmd)
  settled = cmds[len(cmds) // 2:]
  assert settled and set(settled) == {-1024}, f"hold command drifted off the plan: {sorted(set(settled))}"

  # once the body ECU takes the hold over, stock stops asking for the brakes and so do we
  relaxed = []
  for _ in range(seconds(1.0)):
    cmd = accel_cmd_raw(step_long(cc, cs, long_state=STOPPING, accel=-1.024, standstill=True, brake_hold=True))
    if cmd is not None:
      relaxed.append(cmd)
  assert relaxed and set(relaxed) == {round(CarControllerParams.ACCEL_HOLD_LATCHED * 1000)}


def test_gas_override_stays_engaged(cc, cs):
  """A gas press is an override, not a disengagement. The command goes to zero as on every
  other port, but the engaged bits stay set the way Honda drives CONTROL_ON off CC.enabled.
  Clearing them mid-decel takes the PCM out of ACC mode (docs/mazda-gas-override.md)."""
  # braking hard, then the driver taps the gas
  for _ in range(200):
    step_long(cc, cs, accel=-2.0, cruise_engaged=True)
  assert cc.accel_last == pytest.approx(-2.0, rel=1e-6, abs=1e-12)

  cmds = []
  for _ in range(100):  # 1 s of override
    sends = step_long(cc, cs, long_active=False, enabled=True, long_state=OFF, accel=0., gas=True, cruise_engaged=True)
    row = long_frames(sends)
    if row is not None:
      cmds.append(row)

  raw, acc_active, crz_active = zip(*cmds, strict=True)
  assert all(acc_active), "ACC_ACTIVE dropped during a gas override"
  assert all(crz_active), "CRZ_ACTIVE dropped during a gas override"
  assert set(raw) == {0}, f"command should be zero through the override, got {sorted(set(raw))}"


def test_command_slew_is_rate_limited(cc, cs):
  """The plan can step; the wire should not. Windup is limited tightly because dumping the
  brake in one frame is what the driver feels, winddown loosely so braking is never delayed."""
  for _ in range(200):
    step_long(cc, cs, accel=-2.0, cruise_engaged=True)
  assert cc.accel_last == pytest.approx(-2.0, rel=1e-6, abs=1e-12)

  # plan jumps straight to +1.0: the command must ramp, not step
  prev = cc.accel_last
  for _ in range(5):
    step_long(cc, cs, accel=1.0, cruise_engaged=True)
    assert cc.accel_last - prev == pytest.approx(CarControllerParams.ACCEL_WINDUP_LIMIT, abs=1e-6)
    prev = cc.accel_last

  # and the other way, at the looser winddown limit
  for _ in range(200):
    step_long(cc, cs, accel=1.0, cruise_engaged=True)
  prev = cc.accel_last
  for _ in range(5):
    step_long(cc, cs, accel=-3.0, cruise_engaged=True)
    assert cc.accel_last - prev == pytest.approx(CarControllerParams.ACCEL_WINDDOWN_LIMIT, abs=1e-6)
    prev = cc.accel_last


def test_accel_last_tracks_the_wire_not_the_plan(cc, cs):
  # update() reports accel_last as actuatorsOutput.accel, the way Toyota, Ford and Honda
  # report the value they sent. It must be the wire value, clip and hold included.

  # a plan beyond the envelope is reported clipped, not as asked
  for _ in range(400):
    sends = step_long(cc, cs, accel=-9.0, cruise_engaged=True)
  assert cc.accel_last == pytest.approx(CarControllerParams.ACCEL_MIN, rel=1e-6, abs=1e-12)
  row = long_frames(sends)
  if row is not None:
    assert row[0] == round(cc.accel_last * 1000)

  # the standstill hold is the plan's own command, and that is what gets reported
  for _ in range(seconds(0.5)):
    step_long(cc, cs, long_state=STOPPING, accel=-1.5, standstill=True, cruise_engaged=True)
  assert cc.accel_last == pytest.approx(-1.5, rel=1e-6, abs=1e-12)

  # through a gas override we report the zero we actually send
  for _ in range(10):
    step_long(cc, cs, long_active=False, enabled=True, long_state=OFF, accel=0., gas=True, cruise_engaged=True)
  assert cc.accel_last == 0.


def test_gas_from_standstill_hold_releases_the_brake(cc, cs):
  # gas out of a hold is a resume, not a slow release: the hold command must go straight to
  # zero rather than ramping off at the cruising override rate
  for _ in range(seconds(3.0)):
    step_long(cc, cs, long_state=STOPPING, accel=-1.5, standstill=True, cruise_engaged=True)
  assert cc.accel_last < -0.5, "never reached the standstill hold"

  for _ in range(20):
    step_long(cc, cs, long_active=False, enabled=True, long_state=OFF, accel=0., gas=True, standstill=True, cruise_engaged=True)
  assert cc.accel_last == 0., f"hold not released for the driver's gas: {cc.accel_last}"


def test_release_command_holds_through_the_debounce_then_jumps(cc, cs):
  """Stock never lets ACCEL_CMD climb while STOPPING is asserted: through the release
  debounce the command stays at the hold value. Once the stop bits drop it relax-jumps
  into stock's release band and ramps. Pre-ramping toward the plan during the debounce puts
  the zero-cross inside the pulse; slewing up off the hold value puts hold-grade braking
  under it. Stock emits neither tuple."""
  for _ in range(seconds(0.5)):
    step_long(cc, cs, long_state=STOPPING, accel=-1.5, standstill=False, **LEAD_4M)
  for _ in range(seconds(3.0)):
    step_long(cc, cs, long_state=STOPPING, accel=-1.3, standstill=True, **LEAD_4M)
  assert cc.accel_last == pytest.approx(-1.3, rel=1e-6, abs=1e-12)

  rows = crz_info_rows(cc, cs, seconds(1.5), accel=1.0, standstill=True, **LEAD_4M)

  debounce = [r for r in rows if r[1]]
  assert debounce, "no stop-bit frames through the release debounce"
  assert all(cmd == -1300 for cmd, _, _ in debounce), \
    f"command moved off the hold while STOPPING was asserted: {sorted({c for c, _, _ in debounce})}"

  # nothing was latched, so no unlatch bit goes out at all
  assert not any(unl for _, _, unl in rows), "a never-latched release pulsed"
  assert max(cmd for cmd, _, _ in rows) > 500, "command never ramped up after the release"


def test_near_zero_hold_release_emits_no_pulse(cc, cs):
  # a no-lead hold relaxes the plan to ~0, so the release ramp would cross zero in the first
  # pulse frame -- the shape behind the routes 000000fe t+44.54 / 00000100 t+353.18 latches.
  # Nothing is latched here, so the release now carries no unlatch bit for it to land in.
  for _ in range(seconds(0.5)):
    step_long(cc, cs, long_state=STOPPING, accel=-0.5, standstill=False)
  for _ in range(seconds(2.0)):
    step_long(cc, cs, long_state=STOPPING, accel=-0.02, standstill=True)
  assert cc.accel_last == pytest.approx(-0.02, rel=1e-6, abs=1e-12)

  rows = crz_info_rows(cc, cs, seconds(1.5), accel=1.0, standstill=True)
  assert not any(unl for _, _, unl in rows), "a never-latched release pulsed"
  assert max(cmd for cmd, _, _ in rows) > 500, "command never ramped up after the release"


def test_release_keeps_climbing_until_the_car_actually_moves(cc, cs):
  """Route 00000009--ad9e22f986 t+452.9 (EPS-swapped CX-9): the release ran correctly, the
  ramp caught the plan at +0.47 with a vision lead 2.5 m ahead, handed the command back --
  and the car sat dead still for 1.5 s until the driver used the pedal. Mazda runs ki=0, so
  LongControl's pid state emits a_target verbatim and nothing ever escalates. The ramp must
  keep climbing past the plan while the car has not moved."""
  lead = dict(lead_visible=True, lead_d_rel=2.5, lead_v_rel=0.0)
  for _ in range(seconds(0.5)):
    step_long(cc, cs, long_state=STOPPING, accel=-1.024, standstill=False, **lead)
  for _ in range(seconds(2.0)):
    step_long(cc, cs, long_state=STOPPING, accel=-1.024, standstill=True, **lead)

  # the plan asks for exactly the creep the CX-9 could not move on, and the car stays stopped
  peak = -10.
  for _ in range(seconds(2.0)):
    step_long(cc, cs, accel=0.47, standstill=True, **lead)
    peak = max(peak, cc.accel_last)
  assert peak > 0.47 + 0.2, f"command plateaued at the plan and never asked harder: {peak:.2f}"
  assert peak <= CarControllerParams.ACCEL_BREAKAWAY_MAX + 1e-6, f"climbed past the cap: {peak:.2f}"
  assert peak <= 0.47 + CarControllerParams.ACCEL_BREAKAWAY_OVERSHOOT + 1e-6, f"climbed past the plan-relative cap: {peak:.2f}"
  # the override sits on top of the plan, so it is bounded by what stock itself commands
  # pulling away from a stop: over all 31 stock stop->go episodes the breakaway command
  # spans +0.405..+1.425 (latched median +0.958), so this must not exceed stock's own worst
  assert CarControllerParams.ACCEL_BREAKAWAY_MAX <= 1.45, "breakaway ceiling past stock's own max"

  # once it moves, the plan owns the command again
  for _ in range(seconds(0.5)):
    step_long(cc, cs, accel=0.47, standstill=False, **lead)
  assert cc.accel_last == pytest.approx(0.47, abs=0.01)


@pytest.mark.parametrize("plan, cap", [
  # a small plan behind a barely-rolling lead is a nudge above the plan, not a launch
  (0.11, 0.11 + CarControllerParams.ACCEL_BREAKAWAY_OVERSHOOT),
  # a big plan is bounded by stock's worst breakaway alone
  (1.1, CarControllerParams.ACCEL_BREAKAWAY_MAX),
], ids=["small_plan", "big_plan"])
def test_breakaway_is_bounded_relative_to_the_plan(cc, cs, plan, cap):
  """Route 00000132 t+207: a lead 3.9 m ahead barely rolling, plan +0.11, and the still-
  stopped ramp climbed to +1.16 before the car moved. The override is a nudge above the
  plan, not a launch to the absolute ceiling (values.py ACCEL_BREAKAWAY_OVERSHOOT)."""
  lead = dict(lead_visible=True, lead_d_rel=3.9, lead_v_rel=0.2)
  for _ in range(seconds(2.0)):
    step_long(cc, cs, long_state=STOPPING, accel=-1.024, standstill=True, **lead)
  peak = -10.
  for _ in range(seconds(2.5)):
    step_long(cc, cs, accel=plan, standstill=True, **lead)
    peak = max(peak, cc.accel_last)
  assert peak == pytest.approx(cap, abs=0.02), f"peak {peak:.2f} vs cap {cap:.2f}"
  # the plan-relative cap still clears stock's own first-quartile latched breakaway (+0.744, 34 episodes)
  assert 0.11 + CarControllerParams.ACCEL_BREAKAWAY_OVERSHOOT > 0.744


def test_breakaway_ceiling_lowered_by_the_plan_is_walked_down_not_stepped(cc, cs):
  # the ramp is at +1.2 on a plan of +0.6 when the lead stops again and the plan drops to
  # +0.1: the ceiling falls under the ramp, and the command follows at the winddown limit
  for _ in range(seconds(2.0)):
    step_long(cc, cs, long_state=STOPPING, accel=-1.024, standstill=True)
  for _ in range(seconds(1.5)):
    step_long(cc, cs, accel=0.6, standstill=True)
  assert cc.accel_last > 1.0
  prev = cc.accel_last
  for _ in range(seconds(0.5)):
    step_long(cc, cs, accel=0.1, standstill=True)
    assert cc.accel_last - prev >= CarControllerParams.ACCEL_WINDDOWN_LIMIT - 1e-9, f"stepped down {prev:.3f} -> {cc.accel_last:.3f}"
    prev = cc.accel_last
  assert cc.accel_last == pytest.approx(0.1 + CarControllerParams.ACCEL_BREAKAWAY_OVERSHOOT, abs=0.02)


def test_missed_pulse_retry_keeps_the_stock_latched_tuple(cc, cs):
  """If the body never answers the pulse the command sits at the relaxed hold under a positive
  plan; the one retry is byte-for-byte the first pulse's shape: stop bits down, command at
  -1 raw, RESUME_UNLATCHING set, and nothing climbs while GEAR.BRAKE_HOLD stays up."""
  for _ in range(seconds(2.0)):
    step_long(cc, cs, long_state=STOPPING, accel=-1.3, standstill=True, brake_hold=True, **LEAD_4M)
  assert cc.stop_and_go.car_has_hold
  rows = crz_info_rows(cc, cs, RELEASE_DEBOUNCE_FRAMES + 2 * RESUME_REPULSE_FRAMES + seconds(1.0),
                       accel=1.0, standstill=True, brake_hold=True, **LEAD_4M)
  pulses = sum(1 for a, b in zip(rows, rows[1:], strict=False) if b[2] and not a[2])
  assert pulses == 2, f"expected the pulse and exactly one retry, got {pulses}"
  assert all(cmd == -1 and not stop for cmd, stop, unl in rows if unl), "a pulse frame left the latched tuple"
  assert all(cmd == -1 for cmd, _, _ in rows), "command climbed against a body that never let go"
  # the second pulse starts the retry window after the release, not sooner
  starts = [i for i, (a, b) in enumerate(zip(rows, rows[1:], strict=False)) if b[2] and not a[2]]
  assert (starts[1] - starts[0]) * 2 >= RESUME_REPULSE_FRAMES - 2


def test_breakaway_gives_up_so_a_stuck_car_is_not_leaned_on(cc, cs):
  # something we cannot see is holding the car (kerb, grade). Asking forever is worse than
  # settling back onto the plan, which the driver can then override with the pedal.
  for _ in range(seconds(2.0)):
    step_long(cc, cs, long_state=STOPPING, accel=-1.024, standstill=True)
  for _ in range(BREAKAWAY_FRAMES + seconds(1.0)):
    step_long(cc, cs, accel=0.3, standstill=True)
  assert cc.accel_last == pytest.approx(0.3, abs=0.01), f"still leaning on a car that never moved: {cc.accel_last:.2f}"


def test_breakaway_never_climbs_against_a_latched_body(cc, cs):
  """The freeze that keeps the command pinned while GEAR.BRAKE_HOLD is still set (route
  00000115 t+381.3, camera faulted 90 ms into the pulse) outranks the breakaway climb: a
  body-latched hold is pulsed, never leaned on."""
  for _ in range(seconds(2.0)):
    step_long(cc, cs, long_state=STOPPING, accel=-1.3, standstill=True, brake_hold=True, **LEAD_4M)
  assert cc.stop_and_go.car_has_hold

  for _ in range(RESUME_UNLATCH_LATCHED_FRAMES + seconds(1.0)):
    step_long(cc, cs, accel=0.5, standstill=True, brake_hold=True, **LEAD_4M)
    assert cc.accel_last <= CarControllerParams.ACCEL_RESUME_PULSE_MAX + 1e-6, \
      f"breakaway climbed against a still-latched body: {cc.accel_last:.2f}"


def test_never_latched_release_speaks_the_stock_wire_grammar(cc, cs):
  """Slewing off the hold value under a long pulse put hold-grade braking beneath
  RESUME_UNLATCHING, a (stop, unlatch, cmd) tuple stock never emits (route 00000053 t+714.8).
  Stock's never-latched grammar (33-pulse census): the command relax-jumps into the
  -0.27..-0.11 band in one frame and the ramp climbs ~+25 raw per wire frame. Stock also blips
  RESUME_UNLATCHING here; we do not, since nothing is latched, so the blip assertions are
  replaced by requiring no unlatch bit at all."""
  for _ in range(seconds(0.5)):
    step_long(cc, cs, long_state=STOPPING, accel=-1.0, standstill=False, **LEAD_4M)
  for _ in range(seconds(2.0)):
    step_long(cc, cs, long_state=STOPPING, accel=-1.024, standstill=True, **LEAD_4M)
  assert cc.accel_last == pytest.approx(-1.024, rel=1e-6, abs=1e-12)

  rows = crz_info_rows(cc, cs, seconds(1.5), accel=0.45, standstill=True, **LEAD_4M)

  assert not any(stop and unl for _, stop, unl in rows), "stop bits and pulse on one frame"
  drop = next(i for i, (_, stop, _) in enumerate(rows) if not stop)
  post = rows[drop:]
  # the relax jump: no post-drop frame ever carries hold-grade braking again
  assert all(cmd >= -280 for cmd, _, _ in post), f"command stayed at hold depth after the drop: {min(c for c, _, _ in post)}"
  assert post[0][0] <= -180, f"release did not start inside the stock band: {post[0][0]}"
  assert not any(unl for _, _, unl in rows), "a never-latched release pulsed"
  # the ramp: stock's +25 raw per wire frame, straight through the drive-off
  ramping = [c for c, _, _ in post][:20]
  assert all(20 <= b - a <= 30 for a, b in zip(ramping, ramping[1:], strict=False)), f"off the stock ramp: {ramping}"


@pytest.mark.parametrize("drop_wire_frames", [1, 2, 3])
def test_latched_release_speaks_the_stock_pulse_shape(cc, cs, drop_wire_frames):
  # a body-latched hold releases with a 9-wire-frame pulse: the command sits pinned at the
  # relaxed -1 raw for as long as the body still reports GEAR.BRAKE_HOLD (as in every latched
  # release of the census -- climbing before the drop faulted the camera 90 ms in, route
  # 00000115 t+381.3), then climbs stock's ~+25 raw per frame ramp, peaking inside stock's
  # family and never past the +0.25 ceiling (census: 6-11 frames, hold drop 1-3 frames in,
  # cmd -1 climbing to +24..+342)
  for _ in range(seconds(0.5)):
    step_long(cc, cs, long_state=STOPPING, accel=-1.5, standstill=False, **LEAD_4M)
  for _ in range(seconds(2.0)):
    step_long(cc, cs, long_state=STOPPING, accel=-1.3, standstill=True, brake_hold=True, **LEAD_4M)
  assert cc.accel_last == pytest.approx(CarControllerParams.ACCEL_HOLD_LATCHED, rel=1e-6, abs=1e-12)

  # the body reacts to the pulse: BRAKE_HOLD drops 1-3 wire frames after it starts (census)
  rows = []
  pulse_started = None
  window = RELEASE_DEBOUNCE_FRAMES + RESUME_UNLATCH_LATCHED_FRAMES + seconds(1.0)
  for i in range(window):
    body_holds = pulse_started is None or i < pulse_started + 2 * drop_wire_frames
    dat = frame(step_long(cc, cs, accel=1.0, standstill=True, brake_hold=body_holds, **LEAD_4M), CRZ_INFO)
    if dat is not None:
      cmd, _, unl = crz_info(dat)
      rows.append((cmd, unl, body_holds))
      if pulse_started is None and unl:
        pulse_started = i

  pulse = [(cmd, held) for cmd, unl, held in rows if unl]
  cap = round(CarControllerParams.ACCEL_RESUME_PULSE_MAX * 1000)
  assert len(pulse) == RESUME_UNLATCH_LATCHED_FRAMES // 2, f"pulse ran {len(pulse)} wire frames"
  # the contract the route 115 fault turned on: no pulse frame moves off the relaxed hold
  # while the body still reports its latch
  pinned = [cmd for cmd, held in pulse if held]
  assert len(pinned) == drop_wire_frames and all(cmd == -1 for cmd in pinned), f"command moved under the latched hold: {pinned}"
  # then the ramp: stock's +25 raw per wire frame from the relaxed hold
  ramp = [cmd for cmd, held in pulse if not held]
  assert -1 <= ramp[0] <= 15, f"ramp must start off the relaxed hold: {ramp[0]}"
  assert all(20 <= b - a <= 30 for a, b in zip(ramp, ramp[1:], strict=False)), f"off the stock ramp: {ramp}"
  assert -1 + (len(ramp) - 1) * 20 <= max(ramp) <= cap, f"in-pulse peak outside the ramp's own family: {max(ramp)}"
  assert max(cmd for cmd, _, _ in rows) > cap, "command never ramped past the cap after the pulse"


def test_latched_release_pulse_starts_at_the_release(cc, cs):
  """Routes 0000011d and 0000012c: the body ignores silence and a positive nudge alike,
  and answers the pulse within 2-3 wire frames. The pulse fires with the release, so the
  resume happens when the plan commands it."""
  for _ in range(seconds(0.5)):
    step_long(cc, cs, long_state=STOPPING, accel=-1.5, standstill=False, **LEAD_4M)
  for _ in range(seconds(2.0)):
    step_long(cc, cs, long_state=STOPPING, accel=-1.3, standstill=True, brake_hold=True, **LEAD_4M)

  rows = crz_info_rows(cc, cs, RELEASE_DEBOUNCE_FRAMES + seconds(0.5), accel=1.0, standstill=True, brake_hold=True, **LEAD_4M)

  # the stop bits are already down during a body-latched hold, so the pulse's deadline is
  # the debounce itself: it must start on the first wire frame after the plan's request lands
  assert not any(stop for _, stop, _ in rows), "stop bits reappeared during a latched hold"
  first_unl = next(i for i, (_, _, unl) in enumerate(rows) if unl)
  assert first_unl <= RELEASE_DEBOUNCE_FRAMES // 2 + 1, \
    f"pulse lagged the release: {first_unl - RELEASE_DEBOUNCE_FRAMES // 2} wire frames late"
  assert rows[first_unl][0] == -1, f"pulse did not start at the relaxed hold: {rows[first_unl][0]}"


def test_gas_pedal_without_cruise_stays_disengaged(cc, cs):
  # gas pressed while openpilot is not enabled must not advertise an engaged ACC
  cc.frame = 0
  sends = step_long(cc, cs, long_active=False, enabled=False, long_state=OFF, gas=True, available=True)
  assert frame(sends, CRZ_INFO).hex().startswith("01ffe3ffc480")  # armed-but-idle pattern, command pegged


@pytest.mark.parametrize("available, brake_pressed, prefix", [
  # main off, not available: the exact standby pattern the panda allowlists byte-for-byte
  (False, False, "01ffe3ffc000"),
  # MRCC armed but not engaged: the command stays pegged and ACC_SET_ALLOWED follows the
  # brake, exactly the two patterns stock alternates between at an armed idle
  (True, False, "01ffe3ffc480"),
  (True, True, "01ffe3ffc080"),
], ids=["standby", "armed_idle", "armed_idle_brake"])
def test_disengaged_emits_stock_patterns(cc, cs, available, brake_pressed, prefix):
  cc.frame = 0
  sends = step_long(cc, cs, long_active=False, long_state=OFF, available=available, brake_pressed=brake_pressed)
  assert frame(sends, CRZ_INFO).hex().startswith(prefix)
