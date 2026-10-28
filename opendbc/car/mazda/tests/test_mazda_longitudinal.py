"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The accel command on the wire, driven through the real CarController.update_longitudinal:
frame rates, the slew limits, the standstill hold and its relax, the release grammar and the
RESUME_UNLATCHING pulse, the breakaway ramp, the gas override and the disengaged patterns.
"""
import numpy as np
import pytest

from opendbc.car import DT_CTRL
from opendbc.car.mazda import mazdacan
from opendbc.car.mazda.longitudinal import (BREAKAWAY_FRAMES, LEAD_HIDE_FRAMES, LEAD_SHOW_FRAMES, RELEASE_DEBOUNCE_FRAMES,
                                            RESUME_REPULSE_FRAMES, RESUME_UNLATCH_LATCHED_FRAMES, AccelShaper, LeadIcon,
                                            StandstillHold)
from opendbc.car.mazda.tests.conftest import (CRZ_CTRL, CRZ_INFO, RADAR_STATIC, RADAR_UDS, LongCtrlState, accel_cmd_raw,
                                              car_controller, crz_ctrl_lead, crz_info, frame, mazda_car_state, parse_frame,
                                              step_long)
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


@pytest.mark.parametrize("engaged", [False, True])
def test_crz_ctrl_relays_hbc_arming_on_both_buses(cc, cs, engaged):
  # Stock radar relays camera's HBC arming into CRZ_CTRL; synthetic radar must also carry the state.
  kwargs = dict(accel=0.5) if engaged else dict(enabled=False, long_active=False, accel=0., long_state=OFF)
  for armed in (False, True, True, False):
    cc.frame = 0  # force emission
    sends = step_long(cc, cs, hbc_request=armed, available=True, **kwargs)
    for bus in (0, 2):
      assert parse_frame(CRZ_CTRL, frame(sends, CRZ_CTRL, bus), bus)["NEW_SIGNAL_3"] == armed


# leadDistanceBars counts up (1 closest, 3 farthest); DISTANCE_SETTING counts down (DBC:
# 1 is 4 bars, 4 is 1 bar), so the wire carries 5 - bars. Measured on 576 stock segments:
# DISTANCE_LESS (closer) raises the raw, DISTANCE_MORE lowers it; 0 only while unavailable.
@pytest.mark.parametrize("bars, wire", [(1, 4), (2, 3), (3, 2)])
def test_gap_setting_mirrors_driver(cc, cs, bars, wire):
  cc.frame = 0  # force emission on the first step
  sends = step_long(cc, cs, gap=bars)
  assert parse_frame(CRZ_CTRL, frame(sends, CRZ_CTRL))["DISTANCE_SETTING"] == wire


@pytest.mark.parametrize("lead_visible", [True, False])
@pytest.mark.parametrize("kwargs, bucket", [
  (dict(accel=0.5), 2),
  (dict(long_state=STOPPING, accel=-1.0, standstill=True), 3),
  (dict(enabled=False, long_active=False, accel=0., long_state=OFF), 2),
], ids=["following", "stopped", "disengaged"])
def test_lead_shows_on_the_dash_but_not_in_the_tracks(cc, cs, kwargs, bucket, lead_visible):
  # CRZ_CTRL carries the vision lead for the cluster; the tracks stay the stock empty capture.
  cc.lead_icon.shown = lead_visible  # past the debounce, covered on its own below
  cc.frame = 0  # force emission
  sends = step_long(cc, cs, lead_visible=lead_visible, **kwargs)
  for bus in (0, 2):
    assert crz_ctrl_lead(frame(sends, CRZ_CTRL, bus)) == ((1, bucket) if lead_visible else (0, 0))
    for addr, capture in mazdacan.RADAR_TRACK_MSGS.items():
      assert frame(sends, addr, bus)[:7] == capture[:7], f"{addr:#x} left the empty capture"


def test_lead_icon_rides_out_flicker():
  icon = LeadIcon()

  def run(visible, n):
    return [icon.update(visible) for _ in range(n)]

  # a marginal lead flickering just inside the show debounce never reaches the dash
  for _ in range(5):
    assert not any(run(True, LEAD_SHOW_FRAMES - 1) + run(False, 1))
  assert run(True, LEAD_SHOW_FRAMES) == [False] * (LEAD_SHOW_FRAMES - 1) + [True]
  # dropouts just inside the hide debounce keep it
  for _ in range(5):
    assert all(run(False, LEAD_HIDE_FRAMES - 1) + run(True, 1))
  assert run(False, LEAD_HIDE_FRAMES) == [True] * (LEAD_HIDE_FRAMES - 1) + [False]


def test_stop_emits_hold_then_relaxes(cc, cs):
  # approach the stop
  for _ in range(seconds(0.5)):
    step_long(cc, cs, long_state=STOPPING, accel=-1.5, standstill=False)
  # hold at a standstill: the command is the plan's own and must not relax on its own, no
  # matter how long the stop lasts (the creep-into-the-lead regression)
  cmds = []
  for _ in range(seconds(30.0)):
    cmd = accel_cmd_raw(step_long(cc, cs, long_state=STOPPING, accel=-1.0, standstill=True))
    if cmd is not None:
      cmds.append(cmd)
  settled = cmds[len(cmds) // 2:]
  assert settled and set(settled) == {-1024}, f"hold command drifted off the plan: {sorted(set(settled))}"

  # once the body ECU takes the hold over, stock stops asking for the brakes and so do we
  relaxed = []
  for _ in range(seconds(1.0)):
    cmd = accel_cmd_raw(step_long(cc, cs, long_state=STOPPING, accel=-1.0, standstill=True, body_hold=True))
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


def gas_return(cc, cs, v_ego=8., a_ego=0.7, **kwargs):
  """Half a second of rolling gas override at a_ego, then the first frame back under control."""
  for _ in range(seconds(0.5)):
    step_long(cc, cs, long_active=False, enabled=True, long_state=OFF, accel=0., gas=True, cruise_engaged=True,
              v_ego=v_ego, a_ego=a_ego, **kwargs)
  assert cc.accel_last == 0.


def test_rolling_gas_return_climbs_to_the_pull_the_car_had(cc, cs):
  """Back from a rolling gas override with no lead, the command climbs at HANDOVER_BUILD up to the car's own
  pull (at most HANDOVER_MAX), then builds at the stock rate, instead of rebuilding from zero at the stock rate."""
  P = CarControllerParams
  gas_return(cc, cs, lead_visible=False)
  cmds = []
  for _ in range(seconds(1.0)):
    step_long(cc, cs, accel=0.85, cruise_engaged=True, v_ego=8., a_ego=0.7, lead_visible=False)
    cmds.append(cc.accel_last)
  assert cmds[0] == pytest.approx(P.HANDOVER_BUILD * DT_CTRL, abs=1e-6)
  steps = np.diff([0.] + cmds)
  assert max(steps) <= P.HANDOVER_BUILD * DT_CTRL + 1e-6, "never steps faster than the handover climb"
  at_pull = next(i for i, c in enumerate(cmds) if c >= P.HANDOVER_MAX - 1e-6)
  assert at_pull <= seconds(P.HANDOVER_MAX / P.HANDOVER_BUILD) + 1
  assert all(s == pytest.approx(0.8 * DT_CTRL, abs=1e-6) for s in steps[at_pull + 1:] if cmds[-1] < 0.85), \
    "past the car's pull it builds at the stock rate"


def test_gas_return_behind_a_lead_keeps_the_stock_build(cc, cs):
  """With a lead shown the MPC owns the gap, so the return is unchanged."""
  gas_return(cc, cs, lead_visible=True)
  step_long(cc, cs, accel=0.85, cruise_engaged=True, v_ego=8., a_ego=0.7, lead_visible=True)
  assert cc.accel_last == pytest.approx(0.8 * DT_CTRL, abs=1e-6)


@pytest.mark.parametrize("v_ego,a_ego", [(2.0, 0.7), (8., 0.), (8., -0.3)])
def test_gas_return_slow_or_not_pulling_keeps_the_stock_build(cc, cs, v_ego, a_ego):
  gas_return(cc, cs, v_ego=v_ego, a_ego=a_ego, lead_visible=False)
  step_long(cc, cs, accel=0.85, cruise_engaged=True, v_ego=v_ego, a_ego=a_ego, lead_visible=False)
  build = float(np.interp(v_ego, CarControllerParams.ACCEL_BUILD_BP, CarControllerParams.ACCEL_BUILD_V))
  assert cc.accel_last == pytest.approx(build * DT_CTRL, abs=1e-6)


def test_rolling_reengage_without_the_gas_keeps_the_stock_build(cc, cs):
  """Only a return from a gas override climbs; re-engaging while rolling (RES) builds as before."""
  for _ in range(seconds(0.5)):
    step_long(cc, cs, long_active=False, enabled=False, long_state=OFF, accel=0., cruise_engaged=True, v_ego=8., a_ego=0.7,
              lead_visible=False)
  step_long(cc, cs, accel=0.85, cruise_engaged=True, v_ego=8., a_ego=0.7, lead_visible=False)
  assert cc.accel_last == pytest.approx(0.8 * DT_CTRL, abs=1e-6)


def handover_vs_reference(frames, **kwargs):
  """Commands after a rolling gas return for frames of (accel, lead_visible, a_ego): with the handover, and from a
  reference controller where it never arms (a lead shown throughout, which changes nothing else in the command)."""
  out = []
  for reference in (False, True):
    c = car_controller(alpha_long=True)
    s = mazda_car_state(c.CP, c.CP_SP)
    gas_return(c, s, lead_visible=reference, **kwargs)
    cmds = []
    for accel, lead, a_ego in frames:
      step_long(c, s, accel=accel, cruise_engaged=True, v_ego=8., a_ego=a_ego, lead_visible=lead or reference, **kwargs)
      cmds.append(c.accel_last)
    out.append(np.array(cmds))
  return out


def assert_braking_never_later(frames, new, ref):
  brake = np.array([a <= CarControllerParams.HANDOVER_BRAKE for a, _, _ in frames])
  assert brake.any()
  assert np.all(new[brake] <= ref[brake] + 1e-9), "a brake request answered with more than the reference"
  k0 = int(np.argmax(brake))
  assert np.argmax(new[k0:] <= 0.) <= np.argmax(ref[k0:] <= 0.), "non-positive later than the reference"


@pytest.mark.parametrize("after", [seconds(0.05), seconds(0.1), seconds(0.2), seconds(0.4), seconds(0.9)])
def test_brake_during_a_gas_return_is_never_later(after):
  """A brake request mid-climb is answered with no more than the shaper without the climb would send."""
  frames = [(0.85, False, 0.7)] * after + [(-1.0, False, 0.7)] * seconds(0.3)
  new, ref = handover_vs_reference(frames)
  assert new[after - 1] > ref[after - 1], "the climb was running"
  assert_braking_never_later(frames, new, ref)


@pytest.mark.parametrize("brake_at", [seconds(0.95), seconds(0.99), seconds(1.0), seconds(1.01), seconds(1.05), seconds(1.5)])
def test_brake_at_or_after_the_window_end_is_never_later(brake_at):
  """The climb lasts HANDOVER_FRAMES; braking just before, at and after its end is no later than the reference."""
  frames = [(0.85, False, 0.7)] * brake_at + [(-1.0, False, 0.7)] * seconds(0.3)
  assert_braking_never_later(frames, *handover_vs_reference(frames))


@pytest.mark.parametrize("brake_at", [seconds(1.0), seconds(1.05), seconds(1.1), seconds(1.3)])
def test_late_positive_request_then_braking_is_never_later(brake_at):
  """A low plan through most of the window, a late big ask at 0.95 s, then braking around and after the window's end."""
  frames = [(0.1, False, 0.7)] * seconds(0.95) + [(0.85, False, 0.7)] * (brake_at - seconds(0.95)) + \
           [(-1.0, False, 0.7)] * seconds(0.3)
  assert_braking_never_later(frames, *handover_vs_reference(frames))


def test_after_the_return_the_command_rejoins_the_reference():
  frames = [(0.85, False, 0.7)] * seconds(2.0)
  new, ref = handover_vs_reference(frames)
  assert new[-1] == pytest.approx(ref[-1], abs=1e-9)


@pytest.mark.parametrize("plan", [0., -0.1, -0.29])
def test_easing_off_during_a_gas_return_lifts_as_usual(cc, cs, plan):
  """Only a real brake request skips the unwind; a plan easing toward zero lifts at the usual rates, never stepping."""
  gas_return(cc, cs, lead_visible=False)
  for _ in range(seconds(0.3)):
    step_long(cc, cs, accel=0.85, cruise_engaged=True, v_ego=8., a_ego=0.7, lead_visible=False)
  prev = cc.accel_last
  step_long(cc, cs, accel=plan, cruise_engaged=True, v_ego=8., a_ego=0.7, lead_visible=False)
  limit = CarControllerParams.ACCEL_LIFT_LIMIT * DT_CTRL if plan >= 0. else CarControllerParams.ACCEL_WINDDOWN_LIMIT
  assert cc.accel_last - prev == pytest.approx(limit, abs=1e-6)


def test_a_lead_appearing_mid_return_ends_the_climb(cc, cs):
  gas_return(cc, cs, lead_visible=False)
  for _ in range(seconds(0.05)):
    step_long(cc, cs, accel=0.85, cruise_engaged=True, v_ego=8., a_ego=0.7, lead_visible=False)
  prev = cc.accel_last
  step_long(cc, cs, accel=0.85, cruise_engaged=True, v_ego=8., a_ego=0.7, lead_visible=True)
  assert cc.accel_last - prev == pytest.approx(0.8 * DT_CTRL, abs=1e-6)


def test_a_second_gas_press_rearms_the_return(cc, cs):
  gas_return(cc, cs, lead_visible=False)
  for _ in range(seconds(0.2)):
    step_long(cc, cs, accel=0.85, cruise_engaged=True, v_ego=8., a_ego=0.7, lead_visible=False)
  gas_return(cc, cs, lead_visible=False)
  step_long(cc, cs, accel=0.85, cruise_engaged=True, v_ego=8., a_ego=0.7, lead_visible=False)
  assert cc.accel_last == pytest.approx(CarControllerParams.HANDOVER_BUILD * DT_CTRL, abs=1e-6)


@pytest.mark.parametrize("lead_after", [0, seconds(0.1)])
def test_a_lead_appearing_with_braking_keeps_the_brake_protection(lead_after):
  """Codex review P1: rolling at 8 m/s in 3rd, 0.5 s of gas, 0.2 s of +0.85, then a lead appears; with the brake
  request on the same frame or 0.1 s later, braking is no later than without the climb."""
  frames = [(0.85, False, 0.7)] * seconds(0.2) + [(0.85, True, 0.7)] * lead_after + [(-1.0, True, 0.7)] * seconds(0.3)
  new, ref = handover_vs_reference(frames, gear_box=3)
  assert new[seconds(0.2) - 1] > ref[seconds(0.2) - 1], "the climb was running"
  assert_braking_never_later(frames, new, ref)


@pytest.mark.parametrize("plan", [-0.05, -0.2, -0.29])
def test_a_lead_appearing_with_mild_slowing_is_never_later(plan):
  """Codex re-review: 0.2 s into a return a lead appears asking only -0.2; behind a lead any slowing gets the
  reference's command, so it turns non-positive no later than without the climb (was 60 ms vs 10 ms)."""
  frames = [(0.85, False, 0.7)] * seconds(0.2) + [(plan, True, 0.6)] * seconds(0.3)
  new, ref = handover_vs_reference(frames, gear_box=3)
  k0 = seconds(0.2)
  assert np.all(new[k0:] <= ref[k0:] + 1e-9)
  assert np.argmax(new[k0:] <= 0.) <= np.argmax(ref[k0:] <= 0.)


@pytest.mark.parametrize("disabled_s", [0.1, 10.])
def test_gas_while_disengaged_does_not_arm_a_later_engagement(cc, cs, disabled_s):
  """Codex review P2: gas pressed while openpilot is disabled, then an ordinary rolling engagement: stock build."""
  for _ in range(seconds(0.5)):
    step_long(cc, cs, long_active=False, enabled=False, long_state=OFF, accel=0., gas=True, cruise_engaged=True, v_ego=8.,
              a_ego=0.7, lead_visible=False)
  for _ in range(seconds(disabled_s)):
    step_long(cc, cs, long_active=False, enabled=False, long_state=OFF, accel=0., cruise_engaged=True, v_ego=8., a_ego=0.7,
              lead_visible=False)
  step_long(cc, cs, accel=0.85, cruise_engaged=True, v_ego=8., a_ego=0.7, lead_visible=False)
  assert cc.accel_last == pytest.approx(0.8 * DT_CTRL, abs=1e-6)


def test_gas_override_then_cancel_then_reengage_keeps_the_stock_build(cc, cs):
  gas_return_kw = dict(lead_visible=False)
  for _ in range(seconds(0.5)):
    step_long(cc, cs, long_active=False, enabled=True, long_state=OFF, accel=0., gas=True, cruise_engaged=True, v_ego=8.,
              a_ego=0.7, **gas_return_kw)
  for _ in range(seconds(0.5)):  # cancelled
    step_long(cc, cs, long_active=False, enabled=False, long_state=OFF, accel=0., cruise_engaged=True, v_ego=8., a_ego=0.7,
              **gas_return_kw)
  step_long(cc, cs, accel=0.85, cruise_engaged=True, v_ego=8., a_ego=0.7, **gas_return_kw)
  assert cc.accel_last == pytest.approx(0.8 * DT_CTRL, abs=1e-6)


def test_braking_during_a_return_ends_the_climb(cc, cs):
  """Codex review lifecycle case: 0.2 s of climb, 0.3 s at -0.5, then positive again builds at the stock rate, not
  toward the pull the car had before it braked."""
  kw = dict(cruise_engaged=True, v_ego=8., lead_visible=False)
  gas_return(cc, cs, lead_visible=False)
  for _ in range(seconds(0.2)):
    step_long(cc, cs, accel=0.85, a_ego=0.7, **kw)
  for _ in range(seconds(0.3)):
    step_long(cc, cs, accel=-0.5, a_ego=-0.5, **kw)
  prev = cc.accel_last
  rises = []
  for _ in range(seconds(0.4)):
    step_long(cc, cs, accel=0.85, a_ego=-0.5, **kw)
    if prev >= 0.:
      rises.append(cc.accel_last - prev)
    prev = cc.accel_last
  assert rises and all(r == pytest.approx(0.8 * DT_CTRL, abs=1e-6) for r in rises)


def test_hold_ownership_clearing_without_a_handover_matches_the_original():
  """Codex re-review P1: braking at standstill, the body takes the hold for one frame, then lets go with a -0.3
  request. The reference command went through the hold rules too, so nothing steps: -0.001 then -0.101, as before."""
  sh, sm, last = AccelShaper(), StandstillHold(), 0.

  def step(plan, body):
    nonlocal last
    sm.update(True, True, True, plan, body, False)
    last = sh.update(True, plan, last, sm, True, body, 0., 3, a_ego=0., lead_visible=True, gas_pressed=False, long_engaged=True)
    return last
  for _ in range(200):
    step(-1.5, False)
  assert step(-1.5, True) == pytest.approx(CarControllerParams.ACCEL_HOLD_LATCHED, abs=1e-9)
  assert step(-0.3, False) == pytest.approx(CarControllerParams.ACCEL_HOLD_LATCHED + CarControllerParams.ACCEL_WINDDOWN_LIMIT,
                                            abs=1e-9)


@pytest.mark.parametrize("seed", range(6))
def test_without_a_handover_the_reference_is_the_command(seed):
  """Holds, releases, re-holds, interrupted resume pulses, disengagements and gas with a lead shown: with no handover
  armed, the reference the brake clamp uses is exactly the command sent, every frame."""
  rng = np.random.default_rng(seed)
  sh, sm, last, plan = AccelShaper(), StandstillHold(), 0., 0.
  standstill = body = stopping = False
  active = engaged = True
  for k in range(4000):
    plan = float(np.clip(plan + rng.normal(0., 0.15), -3., 1.5))
    standstill ^= bool(rng.random() < 0.02)
    body ^= bool(rng.random() < 0.03)
    stopping ^= bool(rng.random() < 0.02)
    engaged ^= bool(rng.random() < 0.005)
    gas = bool(rng.random() < 0.02)
    active = engaged and not gas
    v = 0. if standstill else float(rng.uniform(0.5, 15.))
    sm.update(engaged, stopping, standstill, plan, body, gas)
    last = sh.update(active, plan, last, sm, standstill, body, v, int(rng.integers(1, 7)), a_ego=float(rng.uniform(-1, 1.5)),
                     lead_visible=True, gas_pressed=gas, long_engaged=engaged)
    assert sh.handover_frames == 0
    assert sh.baseline == last, f"frame {k}"


@pytest.mark.parametrize("plan,gear,v_ego,cap", [(0.3, 0, 8., 0.3), (0.4, 6, 20., CarControllerParams.SIXTH_GEAR_ACCEL),
                                                 (0.85, 4, 10., CarControllerParams.FOURTH_GEAR_ACCEL)])
def test_gas_return_never_climbs_past_the_plan_or_the_gear_cap(cc, cs, plan, gear, v_ego, cap):
  gas_return(cc, cs, v_ego=v_ego, a_ego=1.2, lead_visible=False, gear_box=gear)
  for _ in range(seconds(1.0)):
    step_long(cc, cs, accel=plan, cruise_engaged=True, v_ego=v_ego, a_ego=1.2, lead_visible=False, gear_box=gear)
    assert cc.accel_last <= cap + 1e-6


def test_command_slew_is_rate_limited(cc, cs):
  """The plan can step; the wire should not. Windup is limited tightly because dumping the
  brake in one frame is what the driver feels, winddown loosely so braking is never delayed."""
  for _ in range(200):
    step_long(cc, cs, accel=-2.0, cruise_engaged=True, v_ego=10.)
  assert cc.accel_last == pytest.approx(-2.0, rel=1e-6, abs=1e-12)

  # plan jumps straight to +1.0: the command must ramp, not step; brake release at the windup
  # limit, then the positive part at stock's rolling build rate
  prev = cc.accel_last
  for _ in range(5):
    step_long(cc, cs, accel=1.0, cruise_engaged=True, v_ego=10.)
    assert cc.accel_last - prev == pytest.approx(CarControllerParams.ACCEL_WINDUP_LIMIT, abs=1e-6)
    prev = cc.accel_last
  while cc.accel_last <= 0.:
    step_long(cc, cs, accel=1.0, cruise_engaged=True, v_ego=10.)
  prev = cc.accel_last
  for _ in range(5):
    step_long(cc, cs, accel=1.0, cruise_engaged=True, v_ego=10.)
    assert cc.accel_last - prev == pytest.approx(0.8 * DT_CTRL, abs=1e-6)
    prev = cc.accel_last

  # and the other way, at the looser winddown limit
  for _ in range(200):
    step_long(cc, cs, accel=1.0, cruise_engaged=True)
  prev = cc.accel_last
  for _ in range(5):
    step_long(cc, cs, accel=-3.0, cruise_engaged=True)
    assert cc.accel_last - prev == pytest.approx(CarControllerParams.ACCEL_WINDDOWN_LIMIT, abs=1e-6)
    prev = cc.accel_last


def test_build_rate_follows_stock_by_speed(cc, cs):
  """Stock pulls away from a stop at its 1.22 m/s3 release ramp (+25 raw per 50 Hz frame) and builds
  accel at 0.59 m/s3 once rolling (+12 raw per frame, 99.3% of rising stock frames across 158 routes); rolling is
  deliberately a third quicker here. The build rate governs the positive region only; brake
  release below zero keeps the windup limit."""
  ramp = CarControllerParams.ACCEL_RELEASE_RAMP
  for v_ego, jerk in ((0., ramp), (3., ramp), (4.5, (ramp + 0.8) / 2), (6., 0.8), (20., 0.8)):
    cc.accel_last = 0.
    step_long(cc, cs, accel=2.0, cruise_engaged=True, v_ego=v_ego)
    assert cc.accel_last == pytest.approx(jerk * DT_CTRL, abs=1e-6), f"v_ego={v_ego}"
  cc.accel_last = -1.0
  step_long(cc, cs, accel=0., cruise_engaged=True, v_ego=20.)
  assert cc.accel_last == pytest.approx(-1.0 + CarControllerParams.ACCEL_WINDUP_LIMIT, abs=1e-6)


def test_throttle_lift_follows_stock_but_braking_does_not_wait(cc, cs):
  """Stock lifts the throttle at no more than 2.0 m/s3 (99.98% of falling positive frames). A plan
  easing off toward zero follows that; a plan asking for brake drops at the winddown limit."""
  for _ in range(seconds(3.0)):
    step_long(cc, cs, accel=0.7, cruise_engaged=True, v_ego=10.)
  assert cc.accel_last == pytest.approx(0.7, abs=1e-6)
  prev = cc.accel_last
  for _ in range(5):
    step_long(cc, cs, accel=0., cruise_engaged=True, v_ego=10.)
    assert cc.accel_last - prev == pytest.approx(CarControllerParams.ACCEL_LIFT_LIMIT * DT_CTRL, abs=1e-6)
    prev = cc.accel_last
  step_long(cc, cs, accel=-1.0, cruise_engaged=True, v_ego=10.)
  assert cc.accel_last - prev == pytest.approx(CarControllerParams.ACCEL_WINDDOWN_LIMIT, abs=1e-6)


def test_ceiling_follows_stock_by_speed(cc, cs):
  """The wire never asks for more than stock's own accelerating command at that speed."""
  assert max(CarControllerParams.ACCEL_CEILING_V) <= CarControllerParams.ACCEL_MAX
  bp, v = CarControllerParams.ACCEL_CEILING_BP, CarControllerParams.ACCEL_CEILING_V
  stock_bp, stock_p99 = (0., 4., 9., 14., 18., 25.), (1.5, 1.75, 1.45, 1.05, 0.85, 0.65)
  assert all(np.interp(x, bp, v) <= np.interp(x, stock_bp, stock_p99) + 1e-9 for x in sorted({*bp, *stock_bp}))
  for v_ego, ceiling in ((4., 1.2), (9., 0.9), (13.5, 0.75), (18., 0.55), (35., 0.55)):
    cc.accel_last = 2.0
    for _ in range(seconds(1.0)):  # settles through the lift limit
      step_long(cc, cs, accel=2.0, cruise_engaged=True, v_ego=v_ego)
    assert cc.accel_last == pytest.approx(ceiling, abs=1e-6), f"v_ego={v_ego}"
  # the ceiling shapes the plan; a plan under it passes through
  cc.accel_last = 0.5
  step_long(cc, cs, accel=0.5, cruise_engaged=True, v_ego=14.)
  assert cc.accel_last == pytest.approx(0.5, abs=1e-6)


def test_low_speed_upshift_eases_the_push_so_the_new_gear_holds(cc, cs):
  """After 3rd to 4th under 25 mph the push eases to UPSHIFT_HOLD_ACCEL for UPSHIFT_HOLD_FRAMES, then rebuilds to
  the settled-4th ceiling (nothing to rebuild while the two are equal)."""
  P = CarControllerParams

  def run(n, gear, v_ego=9.0, accel=0.8):
    for _ in range(n):
      step_long(cc, cs, accel=accel, cruise_engaged=True, v_ego=v_ego, gear_box=gear)
  run(seconds(2.0), 3)
  assert cc.accel_last == pytest.approx(0.8, abs=1e-6)
  run(1, 15)  # mid-shift
  run(seconds(0.5), 4)
  assert cc.accel_last == pytest.approx(P.UPSHIFT_HOLD_ACCEL, abs=1e-6)
  # eased at the lift rate, not stepped
  run(seconds(2.0), 4)
  assert cc.accel_last == pytest.approx(P.UPSHIFT_HOLD_ACCEL, abs=1e-6)
  run(seconds(1.0), 4)  # hold over: rebuilds at the build rate (nothing to rebuild while the two are equal)
  assert P.UPSHIFT_HOLD_ACCEL <= cc.accel_last <= P.FOURTH_GEAR_ACCEL + 1e-6
  run(seconds(1.0), 4)
  assert cc.accel_last == pytest.approx(P.FOURTH_GEAR_ACCEL, abs=1e-6)


@pytest.mark.parametrize("v_ego", [9.0, 12.0])
def test_fourth_gear_passes_a_065_push(cc, cs, v_ego):
  """Gear study (routes 00000147-00000187): in 4th under 30 mph, right after a low-speed upshift and settled, a 0.65
  push goes through; route 00000184's 4-3-4 hunts (0.74-0.76 sent) and a 0.9 plan are held to 0.65."""
  def run(n, gear, accel):
    for _ in range(n):
      step_long(cc, cs, accel=accel, cruise_engaged=True, v_ego=v_ego, gear_box=gear)
  run(seconds(2.0), 3, 0.65)
  run(seconds(0.5), 4, 0.65)
  assert cc.accel_last == pytest.approx(0.65, abs=1e-6)
  run(seconds(4.0), 4, 0.74)
  assert cc.accel_last == pytest.approx(0.65, abs=1e-6)
  run(seconds(4.0), 4, 0.9)
  assert cc.accel_last == pytest.approx(0.65, abs=1e-6)


def test_upshift_hold_eases_at_the_lift_rate(cc, cs):
  P = CarControllerParams
  for _ in range(seconds(2.0)):
    step_long(cc, cs, accel=0.8, cruise_engaged=True, v_ego=9.0, gear_box=3)
  prev = cc.accel_last
  step_long(cc, cs, accel=0.8, cruise_engaged=True, v_ego=9.0, gear_box=4)
  assert cc.accel_last - prev == pytest.approx(P.ACCEL_LIFT_LIMIT * DT_CTRL, abs=1e-6)


@pytest.mark.parametrize("gears, v_ego", [
  ((2, 3), 6.0),     # into 3rd: never kicked back in the logs
  ((3, 4), 12.0),    # above 25 mph
  ((4, 4), 9.0),     # no shift
  ((0, 4), 9.0),     # first gear seen after P
  ((4, 3), 9.0),     # downshift
])
def test_upshift_hold_only_for_low_speed_upshifts_into_4th_and_above(cc, cs, gears, v_ego):
  P = CarControllerParams
  ceiling = float(np.interp(v_ego, P.ACCEL_CEILING_BP, P.ACCEL_CEILING_V))
  if gears[-1] == 4 and v_ego < P.FOURTH_GEAR_SPEED:
    ceiling = min(ceiling, P.FOURTH_GEAR_ACCEL)  # settled in 4th, not the upshift hold
  for g in gears:
    for _ in range(seconds(1.5)):
      step_long(cc, cs, accel=2.0, cruise_engaged=True, v_ego=v_ego, gear_box=g)
  assert cc.accel_last == pytest.approx(ceiling, abs=1e-6)


def test_upshift_hold_ends_on_a_downshift_and_never_holds_braking(cc, cs):
  P = CarControllerParams
  for g in (3, 4):
    for _ in range(seconds(1.0)):
      step_long(cc, cs, accel=0.8, cruise_engaged=True, v_ego=9.0, gear_box=g)
  assert cc.accel_last == pytest.approx(P.UPSHIFT_HOLD_ACCEL, abs=1e-6)
  for _ in range(seconds(1.0)):
    step_long(cc, cs, accel=0.8, cruise_engaged=True, v_ego=9.0, gear_box=3)
  assert cc.accel_last == pytest.approx(0.8, abs=1e-6)
  for g in (3, 4):
    for _ in range(seconds(0.5)):
      step_long(cc, cs, accel=-1.5, cruise_engaged=True, v_ego=9.0, gear_box=g)
  assert cc.accel_last == pytest.approx(-1.5, abs=1e-6)


@pytest.mark.parametrize("gear, v_ego, capped", [
  (4, 11.5, True),    # route 00000167: settled in 4th at 26 mph, 0.79 kicked it down to 3rd
  (4, 9.0, True),
  (4, 13.5, True),    # route 0000018f: 0.69 at 30.2 mph kicked it down to 3rd and back
  (4, 15.5, True),
  (4, 15.7, False),   # over 35 mph: stock's own ceiling
  (3, 11.5, False),
  (5, 11.5, False),
  (0, 11.5, False),   # gear not seen
])
def test_settled_4th_stays_under_the_kickdown_push(cc, cs, gear, v_ego, capped):
  P = CarControllerParams
  ceiling = float(np.interp(v_ego, P.ACCEL_CEILING_BP, P.ACCEL_CEILING_V))
  for _ in range(seconds(2.0)):
    step_long(cc, cs, accel=0.8, cruise_engaged=True, v_ego=v_ego, gear_box=gear)
  expected = min(0.8, ceiling, P.FOURTH_GEAR_ACCEL) if capped else min(0.8, ceiling)
  if gear == 5:
    expected = min(expected, P.FIFTH_GEAR_LOW_ACCEL)  # 26 mph: 5th's low-rpm cap
  assert cc.accel_last == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("gear, v_ego, capped", [
  (6, 32.9, True),    # route 00000163 t=300: 0.43 at 74 mph dropped it to 5th and back in 1 s
  (6, 25.0, True),
  (5, 25.0, False),   # 5th: stock's own ceiling
  (0, 32.9, False),   # gear not seen
])
def test_6th_stays_under_the_downshift_push(cc, cs, gear, v_ego, capped):
  P = CarControllerParams
  ceiling = float(np.interp(v_ego, P.ACCEL_CEILING_BP, P.ACCEL_CEILING_V))
  for _ in range(seconds(2.0)):
    step_long(cc, cs, accel=0.4, cruise_engaged=True, v_ego=v_ego, gear_box=gear)  # under a kick's ask
  expected = min(0.4, ceiling, P.SIXTH_GEAR_ACCEL) if capped else min(0.4, ceiling)
  assert cc.accel_last == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("v_ego, low", [(12., True), (17., True), (18., False), (27., False)])  # 27, 38, 40 and 60 mph
def test_5th_stays_under_the_5_4_hunt_push(cc, cs, v_ego, low):
  # route 00000163 t=732: 0.75 at 38 mph dropped 5th to 4th and back in 6 s
  P = CarControllerParams
  for _ in range(seconds(2.0)):
    step_long(cc, cs, accel=0.9, cruise_engaged=True, v_ego=v_ego, gear_box=5)
  ceiling = float(np.interp(v_ego, P.ACCEL_CEILING_BP, P.ACCEL_CEILING_V))
  cap = P.FIFTH_GEAR_LOW_ACCEL if low else P.FIFTH_GEAR_ACCEL
  assert cc.accel_last == pytest.approx(min(ceiling, cap), abs=1e-6)


@pytest.mark.parametrize("v_ego", [12., 15.])  # 27 and 34 mph
def test_5th_at_low_rpm_stays_under_the_kickdown_push(cc, cs, v_ego):
  # route 0000017f t=366: the set-speed floor's 0.80 took 5th to 4th and 3rd at 29 mph (0.55 sent in 5th);
  # route 00000180 t=231: the gap boost's 0.45 at 38 mph dropped 5th to 4th and back in 2 s
  P = CarControllerParams
  for _ in range(seconds(2.0)):
    step_long(cc, cs, accel=0.45, cruise_engaged=True, v_ego=v_ego, gear_box=5)
  assert cc.accel_last == pytest.approx(P.FIFTH_GEAR_LOW_ACCEL, abs=1e-6)
  step_long(cc, cs, accel=-1.5, cruise_engaged=True, v_ego=v_ego, gear_box=5)
  # braking is never capped or delayed: the first brake frame moves at the full wind-down rate
  assert cc.accel_last == pytest.approx(P.FIFTH_GEAR_LOW_ACCEL + P.ACCEL_WINDDOWN_LIMIT, abs=1e-6)


@pytest.mark.parametrize("v_ego, cap", [(17.89, "low"), (17.91, "high")])  # car state speed is float32
def test_5th_low_rpm_cap_ends_at_its_speed(cc, cs, v_ego, cap):
  P = CarControllerParams
  for _ in range(seconds(2.0)):
    step_long(cc, cs, accel=0.9, cruise_engaged=True, v_ego=v_ego, gear_box=5)
  ceiling = float(np.interp(v_ego, P.ACCEL_CEILING_BP, P.ACCEL_CEILING_V))
  expected = P.FIFTH_GEAR_LOW_ACCEL if cap == "low" else min(ceiling, P.FIFTH_GEAR_ACCEL)
  assert cc.accel_last == pytest.approx(expected, abs=1e-6)


def test_a_gas_return_in_5th_at_low_rpm_climbs_only_to_its_cap(cc, cs):
  # 1 of 220 engaged rolling gas releases (routes 00000160-00000180) was in 5th under 40 mph pulling over 0.35
  P = CarControllerParams
  gas_return(cc, cs, v_ego=15., a_ego=0.5, lead_visible=False, gear_box=5)
  for _ in range(seconds(1.0)):
    step_long(cc, cs, accel=0.85, cruise_engaged=True, v_ego=15., a_ego=0.5, lead_visible=False, gear_box=5)
  assert cc.accel_last == pytest.approx(P.FIFTH_GEAR_LOW_ACCEL, abs=1e-6)


def test_an_upshift_hold_into_5th_still_keeps_5ths_cap(cc, cs):
  # review: the hold's 0.6 overrode 5th's 0.55 for 3 s after a low-speed upshift into 5th
  P = CarControllerParams
  for g in (4, 5):
    for _ in range(seconds(1.0)):
      step_long(cc, cs, accel=0.9, cruise_engaged=True, v_ego=10.5, gear_box=g)
  assert cc.accel_shaper.upshift_hold_frames > 0
  assert cc.accel_last == pytest.approx(min(P.UPSHIFT_HOLD_ACCEL, P.FIFTH_GEAR_LOW_ACCEL), abs=1e-6)


def test_a_sustained_big_ask_in_6th_gets_one_kick(cc, cs):
  P = CarControllerParams
  ceiling = float(np.interp(30., P.ACCEL_CEILING_BP, P.ACCEL_CEILING_V))

  def run(n, accel, gear=6):
    for _ in range(n):
      step_long(cc, cs, accel=accel, cruise_engaged=True, v_ego=30., gear_box=gear)
  run(seconds(2.0), 0.3)
  assert cc.accel_last == pytest.approx(P.SIXTH_GEAR_ACCEL, abs=1e-6)  # a modest ask stays capped
  run(P.SIXTH_KICK_FRAMES - 1, 1.3)
  assert cc.accel_last == pytest.approx(P.SIXTH_GEAR_ACCEL, abs=1e-6)  # not before a full second
  run(seconds(1.0), 1.3)
  assert cc.accel_last == pytest.approx(min(1.3, ceiling), abs=1e-6)  # the kick: the cap is lifted
  run(seconds(1.0), 0.3)
  assert cc.accel_last == pytest.approx(min(0.3, ceiling), abs=1e-6)  # lasts while the plan still asks
  run(seconds(1.0), 0.2)
  assert cc.accel_last == pytest.approx(P.SIXTH_GEAR_ACCEL, abs=1e-6)  # released under SIXTH_KICK_RELEASE


def test_two_short_big_asks_in_6th_do_not_add_up_to_a_kick(cc, cs):
  # review: 0.5 s at 0.6, 30 s at 0.3, 0.5 s at 0.6 kicked on the second half-second (asks now over the 1.2 threshold)
  P = CarControllerParams

  def run(n, accel):
    for _ in range(n):
      step_long(cc, cs, accel=accel, cruise_engaged=True, v_ego=30., gear_box=6)
  run(seconds(1.0), 0.3)
  run(seconds(0.5), 1.3)
  run(seconds(30.0), 0.3)
  run(seconds(0.5), 1.3)
  assert not cc.accel_shaper.kick
  assert cc.accel_last == pytest.approx(P.SIXTH_GEAR_ACCEL, abs=1e-6)


def test_back_in_6th_the_cap_holds_through_the_cooldown(cc, cs):
  P = CarControllerParams

  def run(n, accel, gear):
    for _ in range(n):
      step_long(cc, cs, accel=accel, cruise_engaged=True, v_ego=30., gear_box=gear)
  run(seconds(1.0), 1.3, 5)
  run(P.SIXTH_KICK_COOLDOWN_FRAMES - P.SIXTH_KICK_FRAMES, 1.3, 6)  # still asking, but just upshifted
  assert cc.accel_last == pytest.approx(P.SIXTH_GEAR_ACCEL, abs=1e-6)
  run(P.SIXTH_KICK_FRAMES + seconds(2.0), 1.3, 6)  # the cooldown ends, then a full second of asking
  assert cc.accel_last > P.SIXTH_GEAR_ACCEL + 0.1

def test_a_far_lead_catch_up_in_6th_does_not_kick(cc, cs):
  # routes 00000184 and 00000186: the gap boost and floor asked 0.55-0.65 for over a second (peaks to 0.78) with the
  # model at 0.12-0.42, and the kick took 6th to 5th and 4th; a catch-up ask just under the threshold stays capped
  P = CarControllerParams

  def run(n, accel):
    for _ in range(n):
      step_long(cc, cs, accel=accel, cruise_engaged=True, v_ego=27., gear_box=6)
  run(seconds(1.0), 0.3)
  run(seconds(5.0), 0.69)
  assert not cc.accel_shaper.kick
  # routes 0000018b and 0000018f: the boosts peaked at 0.96 and held 0.74-0.81 for over a second, and kicked at 0.7
  run(seconds(5.0), 1.19)
  assert not cc.accel_shaper.kick
  assert cc.accel_last == pytest.approx(P.SIXTH_GEAR_ACCEL, abs=1e-6)


def test_6th_cap_eases_at_the_lift_rate_and_never_holds_braking(cc, cs):
  P = CarControllerParams
  cc.accel_last = 0.5
  step_long(cc, cs, accel=0.5, cruise_engaged=True, v_ego=30., gear_box=6)
  assert cc.accel_last == pytest.approx(0.5 + P.ACCEL_LIFT_LIMIT * DT_CTRL, abs=1e-6)
  for _ in range(seconds(0.5)):
    step_long(cc, cs, accel=-1.0, cruise_engaged=True, v_ego=30., gear_box=6)
  assert cc.accel_last == pytest.approx(-1.0, abs=1e-6)


def test_settled_4th_cap_eases_at_the_lift_rate_and_never_holds_braking(cc, cs):
  P = CarControllerParams
  cc.accel_last = 0.8
  step_long(cc, cs, accel=0.8, cruise_engaged=True, v_ego=11.5, gear_box=4)
  assert cc.accel_last == pytest.approx(0.8 + P.ACCEL_LIFT_LIMIT * DT_CTRL, abs=1e-6)
  for _ in range(seconds(1.0)):
    step_long(cc, cs, accel=-1.5, cruise_engaged=True, v_ego=11.5, gear_box=4)
  assert cc.accel_last == pytest.approx(-1.5, abs=1e-6)


def test_accel_last_tracks_the_wire_not_the_plan(cc, cs):
  # update() reports accel_last as actuatorsOutput.accel, the way Toyota, Ford and Honda
  # report the value they sent. It must be the wire value, clip and hold included.

  # a plan beyond the envelope is reported clipped, not as asked
  for _ in range(400):
    sends = step_long(cc, cs, accel=-9.0, cruise_engaged=True)
  assert cc.accel_last == pytest.approx(CarControllerParams.ACCEL_MIN, rel=1e-6, abs=1e-12)
  row = long_frames(sends)
  if row is not None:
    assert row[0] == round(cc.accel_last * 1024)

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
  assert all(cmd == round(-1.3 * 1024) for cmd, _, _ in debounce), \
    f"command moved off the hold while STOPPING was asserted: {sorted({c for c, _, _ in debounce})}"

  # Never-latched releases do not send an unlatch pulse.
  assert not any(unl for _, _, unl in rows), "a never-latched release pulsed"
  assert max(cmd for cmd, _, _ in rows) > 500, "command never ramped up after the release"


def test_near_zero_hold_release_emits_no_pulse(cc, cs):
  # A near-zero hold remains a never-latched release and must not send an unlatch pulse.
  for _ in range(seconds(0.5)):
    step_long(cc, cs, long_state=STOPPING, accel=-0.5, standstill=False)
  for _ in range(seconds(2.0)):
    step_long(cc, cs, long_state=STOPPING, accel=-0.02, standstill=True)
  assert cc.accel_last == pytest.approx(-0.02, rel=1e-6, abs=1e-12)

  rows = crz_info_rows(cc, cs, seconds(1.5), accel=1.0, standstill=True)
  assert not any(unl for _, _, unl in rows), "a never-latched release pulsed"
  assert max(cmd for cmd, _, _ in rows) > 500, "command never ramped up after the release"


def test_release_keeps_climbing_until_the_car_actually_moves(cc, cs):
  """Continue the bounded release ramp while stopped because Mazda longitudinal has no integrator."""
  lead = dict(lead_visible=True, lead_d_rel=2.5, lead_v_rel=0.0)
  for _ in range(seconds(0.5)):
    step_long(cc, cs, long_state=STOPPING, accel=-1.0, standstill=False, **lead)
  for _ in range(seconds(2.0)):
    step_long(cc, cs, long_state=STOPPING, accel=-1.0, standstill=True, **lead)

  # Keep the simulated car stopped under a plan command below breakaway authority.
  peak = -10.
  for _ in range(seconds(2.0)):
    step_long(cc, cs, accel=0.47, standstill=True, **lead)
    peak = max(peak, cc.accel_last)
  assert peak > 0.47 + 0.2, f"command plateaued at the plan and never asked harder: {peak:.2f}"
  assert peak <= CarControllerParams.ACCEL_BREAKAWAY_MAX + 1e-6, f"climbed past the cap: {peak:.2f}"
  assert peak <= 0.47 + CarControllerParams.ACCEL_BREAKAWAY_OVERSHOOT + 1e-6, f"climbed past the plan-relative cap: {peak:.2f}"
  # Bound the override by the measured stock breakaway range.
  assert CarControllerParams.ACCEL_BREAKAWAY_MAX <= 1.45, "breakaway ceiling past stock's own max"

  # Return command ownership to the plan after movement begins.
  for _ in range(seconds(0.5)):
    step_long(cc, cs, accel=0.47, standstill=False, **lead)
  assert cc.accel_last == pytest.approx(0.47, abs=0.01)


@pytest.mark.parametrize("plan, cap", [
  # A small plan permits only the configured relative margin. Smaller still never opens the hold.
  (0.3, 0.3 + CarControllerParams.ACCEL_BREAKAWAY_OVERSHOOT),
  # A large plan remains bounded by stock breakaway authority.
  (1.1, CarControllerParams.ACCEL_BREAKAWAY_MAX),
], ids=["small_plan", "big_plan"])
def test_breakaway_is_bounded_relative_to_the_plan(cc, cs, plan, cap):
  """Bound breakaway by both stock authority and the plan-relative margin."""
  lead = dict(lead_visible=True, lead_d_rel=3.9, lead_v_rel=0.2)
  for _ in range(seconds(2.0)):
    step_long(cc, cs, long_state=STOPPING, accel=-1.0, standstill=True, **lead)
  peak = -10.
  for _ in range(seconds(2.5)):
    step_long(cc, cs, accel=plan, standstill=True, **lead)
    peak = max(peak, cc.accel_last)
  assert peak == pytest.approx(cap, abs=0.02), f"peak {peak:.2f} vs cap {cap:.2f}"
  # The relative cap remains above the measured first-quartile stock breakaway command.
  assert 0.11 + CarControllerParams.ACCEL_BREAKAWAY_OVERSHOOT > 0.744


def test_breakaway_ceiling_lowered_by_the_plan_is_walked_down_not_stepped(cc, cs):
  # the ramp is at +1.2 on a plan of +0.6 when the lead stops again and the plan drops to
  # +0.1: the ceiling falls under the ramp, and the command follows at the winddown limit
  for _ in range(seconds(2.0)):
    step_long(cc, cs, long_state=STOPPING, accel=-1.0, standstill=True)
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
  -1 raw, RESUME_UNLATCHING set, and nothing climbs while the body still holds."""
  for _ in range(seconds(2.0)):
    step_long(cc, cs, long_state=STOPPING, accel=-1.3, standstill=True, body_hold=True, **LEAD_4M)
  assert cc.stop_and_go.car_has_hold
  rows = crz_info_rows(cc, cs, RELEASE_DEBOUNCE_FRAMES + 2 * RESUME_REPULSE_FRAMES + seconds(1.0),
                       accel=1.0, standstill=True, body_hold=True, **LEAD_4M)
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
    step_long(cc, cs, long_state=STOPPING, accel=-1.0, standstill=True)
  for _ in range(BREAKAWAY_FRAMES + seconds(1.0)):
    step_long(cc, cs, accel=0.3, standstill=True)
  assert cc.accel_last == pytest.approx(0.3, abs=0.01), f"still leaning on a car that never moved: {cc.accel_last:.2f}"


def test_breakaway_never_climbs_against_a_latched_body(cc, cs):
  """Body-latched holds remain pinned until the body acknowledges the release pulse."""
  for _ in range(seconds(2.0)):
    step_long(cc, cs, long_state=STOPPING, accel=-1.3, standstill=True, body_hold=True, **LEAD_4M)
  assert cc.stop_and_go.car_has_hold

  for _ in range(RESUME_UNLATCH_LATCHED_FRAMES + seconds(1.0)):
    step_long(cc, cs, accel=0.5, standstill=True, body_hold=True, **LEAD_4M)
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
    step_long(cc, cs, long_state=STOPPING, accel=-1.0, standstill=True, **LEAD_4M)
  assert cc.accel_last == pytest.approx(-1.0, rel=1e-6, abs=1e-12)

  rows = crz_info_rows(cc, cs, seconds(1.5), accel=0.45, standstill=True, **LEAD_4M)

  assert not any(stop and unl for _, stop, unl in rows), "stop bits and pulse on one frame"
  drop = next(i for i, (_, stop, _) in enumerate(rows) if not stop)
  post = rows[drop:]
  # No post-release frame may return to hold-grade braking.
  assert all(cmd >= -280 for cmd, _, _ in post), f"command stayed at hold depth after the drop: {min(c for c, _, _ in post)}"
  assert post[0][0] <= -180, f"release did not start inside the stock band: {post[0][0]}"
  assert not any(unl for _, _, unl in rows), "a never-latched release pulsed"
  # Match stock's 25-count-per-wire-frame release ramp through the nine-frame pulse window, then
  # the quicker still-stopped build (50 counts).
  ramping = [c for c, _, _ in post][:20]
  steps = [b - a for a, b in zip(ramping, ramping[1:], strict=False)]
  assert all(20 <= d <= 30 for d in steps[:8]), f"off the stock ramp: {ramping}"
  assert all(20 <= d <= 55 for d in steps[8:10]) and all(45 <= d <= 55 for d in steps[10:]), f"off the still-stopped ramp: {ramping}"


def test_still_stopped_release_builds_faster_only_until_the_car_rolls(cc, cs):
  """Past the pulse window the release builds at ACCEL_RELEASE_RAMP_STOPPED while the car still
  stands, and returns to stock's ramp once it rolls (the wire lags the ramp by one frame)."""
  for _ in range(seconds(2.0)):
    step_long(cc, cs, long_state=STOPPING, accel=-1.024, standstill=True, **LEAD_4M)
  for _ in range(RELEASE_DEBOUNCE_FRAMES + RESUME_UNLATCH_LATCHED_FRAMES + 2):
    step_long(cc, cs, accel=1.0, standstill=True, **LEAD_4M)
  prev = cc.accel_last
  step_long(cc, cs, accel=1.0, standstill=True, **LEAD_4M)
  assert cc.accel_last - prev == pytest.approx(CarControllerParams.ACCEL_RELEASE_RAMP_STOPPED * DT_CTRL, abs=1e-6)
  step_long(cc, cs, accel=1.0, standstill=False, **LEAD_4M)
  prev = cc.accel_last
  step_long(cc, cs, accel=1.0, standstill=False, **LEAD_4M)
  assert cc.accel_last - prev == pytest.approx(CarControllerParams.ACCEL_RELEASE_RAMP * DT_CTRL, abs=1e-6)


@pytest.mark.parametrize("drop_wire_frames", [1, 2, 3])
def test_latched_release_speaks_the_stock_pulse_shape(cc, cs, drop_wire_frames):
  # Hold the relaxed command until body ownership clears, then follow stock's release ramp
  # within its measured pulse ceiling.
  for _ in range(seconds(0.5)):
    step_long(cc, cs, long_state=STOPPING, accel=-1.5, standstill=False, **LEAD_4M)
  for _ in range(seconds(2.0)):
    step_long(cc, cs, long_state=STOPPING, accel=-1.3, standstill=True, body_hold=True, **LEAD_4M)
  assert cc.accel_last == pytest.approx(CarControllerParams.ACCEL_HOLD_LATCHED, rel=1e-6, abs=1e-12)

  # Model the measured one-to-three-frame body response.
  rows = []
  pulse_started = None
  window = RELEASE_DEBOUNCE_FRAMES + RESUME_UNLATCH_LATCHED_FRAMES + seconds(1.0)
  for i in range(window):
    body_holds = pulse_started is None or i < pulse_started + 2 * drop_wire_frames
    dat = frame(step_long(cc, cs, accel=1.0, standstill=True, body_hold=body_holds, **LEAD_4M), CRZ_INFO)
    if dat is not None:
      cmd, _, unl = crz_info(dat)
      rows.append((cmd, unl, body_holds))
      if pulse_started is None and unl:
        pulse_started = i

  pulse = [(cmd, held) for cmd, unl, held in rows if unl]
  cap = round(CarControllerParams.ACCEL_RESUME_PULSE_MAX * 1000)
  assert len(pulse) == RESUME_UNLATCH_LATCHED_FRAMES // 2, f"pulse ran {len(pulse)} wire frames"
  # Keep every pulse frame pinned while the body still reports its latch.
  pinned = [cmd for cmd, held in pulse if held]
  assert len(pinned) == drop_wire_frames and all(cmd == -1 for cmd in pinned), f"command moved under the latched hold: {pinned}"
  # Ramp from the relaxed hold at stock's wire rate.
  ramp = [cmd for cmd, held in pulse if not held]
  assert -1 <= ramp[0] <= 15, f"ramp must start off the relaxed hold: {ramp[0]}"
  assert all(20 <= b - a <= 30 for a, b in zip(ramp, ramp[1:], strict=False)), f"off the stock ramp: {ramp}"
  assert -1 + (len(ramp) - 1) * 20 <= max(ramp) <= cap, f"in-pulse peak outside the ramp's own family: {max(ramp)}"
  assert max(cmd for cmd, _, _ in rows) > cap, "command never ramped past the cap after the pulse"


def test_latched_release_pulse_starts_at_the_release(cc, cs):
  """Start the unlatch pulse with the plan-driven release."""
  for _ in range(seconds(0.5)):
    step_long(cc, cs, long_state=STOPPING, accel=-1.5, standstill=False, **LEAD_4M)
  for _ in range(seconds(2.0)):
    step_long(cc, cs, long_state=STOPPING, accel=-1.3, standstill=True, body_hold=True, **LEAD_4M)

  rows = crz_info_rows(cc, cs, RELEASE_DEBOUNCE_FRAMES + seconds(0.5), accel=1.0, standstill=True, body_hold=True, **LEAD_4M)

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


@pytest.mark.parametrize("mads,engaged", [(True, True), (False, True), (True, False), (False, False)])
def test_cruise_frame_wheel_follows_mads_with_cruise_on_or_off(cc, cs, mads, engaged):
  """Under alpha long the dash's white wheel rides CRZ_CTRL byte 4 bit 0x20 whenever openpilot steers (MADS):
  with cruise off through the camera frame's white state, with cruise engaged on its own. Never on bus 2."""
  sends = []
  for _ in range(seconds(1.0)):
    sends += step_long(cc, cs, accel=0.3, cruise_engaged=engaged, mads_active=mads, v_ego=20.)
  on_car = [d for a, d, b in sends if a == CRZ_CTRL and b == 0][-1:]
  on_radar = [d for a, d, b in sends if a == CRZ_CTRL and b == 2][-1:]
  on_car, on_radar = (on_car or [None])[0], (on_radar or [None])[0]
  assert on_car is not None and on_radar is not None
  assert bool(on_car[4] & 0x20) == mads
  assert not on_radar[4] & 0x20


@pytest.mark.parametrize("mads,available,white", [(True, True, True), (False, True, False), (True, False, False)])
def test_cruise_frame_wheel_with_cruise_on_not_engaged(cc, cs, mads, available, white):
  """Cruise main on but not set (before SET or after a cancel), radar still ours: the wheel rides CRZ_CTRL
  while MADS steers. Main off leaves it to the camera frame's white state. Never on bus 2."""
  sends = []
  for _ in range(seconds(1.0)):
    sends += step_long(cc, cs, long_active=False, enabled=False, available=available, mads_active=mads, v_ego=20.)
  on_car = [d for a, d, b in sends if a == CRZ_CTRL and b == 0][-1:]
  on_radar = [d for a, d, b in sends if a == CRZ_CTRL and b == 2][-1:]
  assert on_car and on_radar
  assert bool(on_car[0][4] & 0x20) == white
  assert not on_radar[0][4] & 0x20
