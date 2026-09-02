"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The lead we advertise to the camera: AdvertisedLead on its own, then CRZ_CTRL's lead fields
and the 0x364 track slot as the controller actually emits them.
"""
import pytest

from opendbc.car import DT_CTRL
from opendbc.car.mazda import mazdacan
from opendbc.car.mazda.longitudinal import LEAD_DEBOUNCE_FRAMES, AdvertisedLead
from opendbc.car.mazda.tests.conftest import (CRZ_CTRL, LEAD_TRACK, LongCtrlState, crz_ctrl_lead, frame, frames, lead_track,
                                              step_long)

# create_radar_frames stamps the counter into the last byte, so an empty slot is the first seven
EMPTY_TRACK = mazdacan.RADAR_TRACK_MSGS[0x364][:7]


def track_occupied(dat):
  return dat[:7] != EMPTY_TRACK


def drive(al, n, **kwargs):
  defaults = dict(lead_visible=True, d_rel=40.0, v_rel=0.0, holding=False)
  defaults.update(kwargs)
  for _ in range(n):
    al.update(**defaults)
  return al


class TestAdvertisedLead:
  """has_lead, the phase and the track slot are one decision, so they are asserted together."""

  def test_lead_follows_only_a_steady_state(self):
    al = AdvertisedLead()
    # a lead is adopted once leadVisible has held for the debounce window, not before
    drive(al, LEAD_DEBOUNCE_FRAMES - 1)
    assert not al.has_lead and al.ctrl_phase == 0
    drive(al, 1)
    assert al.has_lead and al.lead == (40.0, 0.0) and al.ctrl_phase == 2
    # and dropped the same way
    drive(al, LEAD_DEBOUNCE_FRAMES - 1, lead_visible=False, d_rel=0.)
    assert al.has_lead
    drive(al, 1, lead_visible=False, d_rel=0.)
    assert not al.has_lead and al.ctrl_phase == 0

  def test_lead_flicker_never_reaches_the_bus(self):
    al = AdvertisedLead()
    # the measured failure: a marginal 120 m vision lead toggled leadVisible 6 times in 1.4 s
    # (route 6bb2dc61c4 t+400); none of it may reach RADAR_HAS_LEAD or the track slot
    for n, visible in ((15, True), (5, False), (7, True), (13, False), (10, True)):
      drive(al, n, lead_visible=visible)
      assert not al.has_lead, "a flickering lead leaked through the debounce"

  def test_measurement_is_coasted_across_a_dropout(self):
    al = AdvertisedLead()
    # leadOne goes to zero the instant vision drops the lead, well before the debounce expires.
    # Advertising a fabricated stand-in there put a stationary object 10.25 m dead ahead on the
    # bus at 22 m/s; the last real measurement carries the gap instead -- propagated by its own
    # range rate, never repeated frozen (a frozen range is content no radar ever emits)
    drive(al, 2 * LEAD_DEBOUNCE_FRAMES, d_rel=120.0, v_rel=0.5)
    assert al.lead == (120.0, 0.5)
    coast_frames = LEAD_DEBOUNCE_FRAMES - 1
    drive(al, coast_frames, lead_visible=False, d_rel=0., v_rel=0.)
    assert al.lead is not None, "dropped the measurement inside the debounce window"
    d, v = al.lead
    assert v == 0.5
    assert d == pytest.approx(120.0 + 0.5 * coast_frames * DT_CTRL, abs=1e-6), "the coast must propagate the range, not freeze it"

  def test_holding_reports_the_stop_phase_only_with_a_lead(self):
    al = AdvertisedLead()
    drive(al, 2 * LEAD_DEBOUNCE_FRAMES, holding=True)
    assert al.ctrl_phase == 3
    drive(al, 2 * LEAD_DEBOUNCE_FRAMES, lead_visible=False, d_rel=0., holding=True)
    assert not al.has_lead and al.ctrl_phase == 0


class TestLeadOnTheBus:
  """The advertisement through the real update_longitudinal: the track slot and CRZ_CTRL."""

  def test_lead_track_follows_the_measured_lead(self, cc, cs):
    # a real radar re-measures every track every 100 ms, so the range we advertise has to
    # move with the lead we are actually following
    # let the lead debounce adopt the visible lead before sampling the track
    for _ in range(LEAD_DEBOUNCE_FRAMES):
      step_long(cc, cs, accel=0.5, lead_visible=True, lead_d_rel=20.0, lead_v_rel=-1.5)
    seen = []
    for i in range(60):
      sends = step_long(cc, cs, accel=0.5, lead_visible=True, lead_d_rel=20.0 - 0.1 * i, lead_v_rel=-1.5)
      track = frame(sends, LEAD_TRACK)
      if track is not None:
        seen.append(lead_track(track))
    assert len(seen) > 1
    dists = [d for d, _ in seen]
    assert all(a > b for a, b in zip(dists, dists[1:], strict=False)), f"range did not close with the lead: {dists}"
    assert all(abs(v - -1.5) <= 0.0625 for _, v in seen)

  def test_hold_with_nothing_ahead_advertises_nothing(self, cc, cs):
    # No fabricated object. The body does not decide the latch on the advertisement: across 32
    # stock engaged standstills the radar said has_lead=1 in every one, yet 23 latched
    # GEAR.BRAKE_HOLD and 9 did not (one held 104 s), and 89 of 115 stock latches happened at
    # has_lead=0 / phase=0. What is left to avoid is a phantom the camera can refute.
    held, ctrls = [], []
    for _ in range(400):
      sends = step_long(cc, cs, long_state=LongCtrlState.stopping, accel=-1.024, standstill=True,
                        lead_visible=False, lead_d_rel=0.0, cruise_engaged=True)
      held += frames(sends, LEAD_TRACK)
      ctrls += frames(sends, CRZ_CTRL)
    assert held and ctrls
    assert not any(map(track_occupied, held)), "fabricated a lead for a hold with nothing ahead"
    assert all(crz_ctrl_lead(d) == (0, 0) for d in ctrls), "advertised a lead with nothing in view"
    # and the hold itself is untouched: the plan's brake and the stop bits still go out
    assert cc.stop_and_go.holding and cc.stop_and_go.stop_bits

  def test_vision_lead_dropout_does_not_fabricate_a_lead_at_speed(self, cc, cs):
    # leadOne goes to zero the instant the vision lead drops while sm.lead_visible is still
    # latched. Falling through to the hold fallback there put a stationary object 10.25 m dead
    # ahead on the bus at 22 m/s, 20 times across the two 2026-08-25 drives.
    for _ in range(200):  # settle a real lead at 120 m while cruising
      step_long(cc, cs, accel=0.5, lead_visible=True, lead_d_rel=120.0, lead_v_rel=0.5, cruise_engaged=True)

    dropped = []
    for _ in range(int(LEAD_DEBOUNCE_FRAMES * 0.8)):  # inside the debounce window
      dropped += frames(step_long(cc, cs, accel=0.5, lead_visible=False, lead_d_rel=0.0, lead_v_rel=0.0,
                                  cruise_engaged=True), LEAD_TRACK)
    assert dropped
    for d in dropped:
      dist = lead_track(d)[0]
      assert dist == pytest.approx(120.0, abs=1.0), f"track teleported to {dist} m"

  @pytest.mark.parametrize("kw", [
    dict(accel=0.5, lead_visible=True, lead_d_rel=40.0),
    dict(accel=0.5, lead_visible=False, lead_d_rel=0.0),
    dict(long_state=LongCtrlState.stopping, accel=-1.024, standstill=True, lead_visible=False, lead_d_rel=0.0),
    dict(accel=0.3, standstill=True, lead_visible=False, lead_d_rel=0.0),
    dict(accel=0.5, lead_visible=True, lead_d_rel=0.0),
  ], ids=["following", "no_lead", "hold_no_lead", "release_no_lead", "visible_but_unmeasured"])
  def test_has_lead_phase_and_track_never_disagree(self, cc, cs, kw):
    # stock pairs all three absolutely: has_lead=0 <=> phase=0, and RADAR_HAS_LEAD=1 with all six
    # slots empty appears 8 times in 1,095,826 stock samples. We shipped has_lead=0 with phase=1
    # for 22-84% of every engaged drive before this was derived from one decision.
    for _ in range(120):
      sends = step_long(cc, cs, cruise_engaged=True, **kw)
      trk, ctl = frame(sends, LEAD_TRACK), frame(sends, CRZ_CTRL)
      if trk is None or ctl is None:
        continue
      has_lead, phase = crz_ctrl_lead(ctl)
      assert bool(has_lead) == track_occupied(trk), f"has_lead/track disagree for {kw}"
      assert (phase == 0) == (has_lead == 0), f"has_lead/phase disagree for {kw}"

  def test_lead_survives_disengagement(self, cc, cs):
    # perception is engagement-independent: stock advertises RADAR_HAS_LEAD=1 with cruise off in
    # 19.5% of all frames. Dropping the advertisement at disengage made a real car 4.5 m ahead
    # vanish from the bus in one frame while the driver braked toward it, and the camera ran its
    # SCBS display six seconds (route 0000004d t+212)
    for _ in range(120):
      step_long(cc, cs, cruise_engaged=True, lead_d_rel=4.8, accel=-0.5)
    for _ in range(60):
      sends = step_long(cc, cs, long_active=False, enabled=False, long_state=LongCtrlState.off, accel=0., lead_d_rel=4.8)
      trk, ctl = frame(sends, LEAD_TRACK), frame(sends, CRZ_CTRL)
      if ctl is None:
        continue
      has_lead, phase = crz_ctrl_lead(ctl)
      assert has_lead == 1 and phase != 0, "disengaging dropped a real lead off the bus"
      if trk is not None:
        assert lead_track(trk)[0] == pytest.approx(4.8, abs=0.1)
