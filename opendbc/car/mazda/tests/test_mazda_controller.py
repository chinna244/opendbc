#!/usr/bin/env python3
"""Tests for the Mazda CX-5 2022+ EPS steering parameters (gated on the EPS, not the model)
and the longitudinal message builders and standstill hold."""

from types import SimpleNamespace

import numpy as np
import pytest

from opendbc.can import CANPacker, CANParser
from opendbc.car import Bus, DT_CTRL, structs
from opendbc.car.mazda import mazdacan
from opendbc.car.mazda.carcontroller import (CarController, TJA_MRCC_FIRST_TX_DELAY_NANOS,
                                             TJA_MRCC_RAW_OFF_CONFIRM_FRAMES, TJA_MRCC_MAX_TX_FRAMES,
                                             TJA_MRCC_RELEASE_WAIT_FRAMES)
from opendbc.car.mazda.longitudinal import (LEAD_DEBOUNCE_FRAMES, RADAR_SESSION_LIMIT_FRAMES, RELEASE_DEBOUNCE_FRAMES,
                                            RESUME_PULSE_DEFER_FRAMES, RESUME_UNLATCH_LATCHED_FRAMES,
                                            AdvertisedLead, RadarSessionManager, RadarSessionState, StandstillHold)
from opendbc.car.mazda.interface import CarInterface
from opendbc.car.mazda.values import CAR, CarControllerParams, MazdaSafetyFlags


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

  def test_eps_ceiling_never_exceeds_steer_max_scale(self, cx5_2022_params):
    # The ceiling is a clamp on delivered-torque counts; the scale is STEER_MAX. The clamp is
    # only meaningful if it sits at or below the scale at every speed.
    bp, vals = cx5_2022_params.EPS_CEILING_LOOKUP
    for v in np.arange(0.0, 40.0, 0.25):
      ceiling = np.interp(v, bp, vals)
      steer_max = np.interp(v, cx5_2022_params.STEER_MAX_LOOKUP[0],
                            cx5_2022_params.STEER_MAX_LOOKUP[1])
      assert 0 < ceiling <= steer_max, f"ceiling {ceiling} vs steer_max {steer_max} at {v} m/s"

  def test_eps_ceiling_is_monotone_and_matches_the_measured_rails(self, cx5_2022_params):
    # Measured over 11.4M clean frames: 1148 below 18 mph, a monotone rolloff, hard 620 from
    # 32.5 mph up (docs/mazda-lkas-camera-tx-census.md). Nothing above 620 was ever delivered
    # above 32.5 mph in 7.5M frames, so the high-speed leg must not drift back up.
    bp, vals = cx5_2022_params.EPS_CEILING_LOOKUP
    assert list(vals) == sorted(vals, reverse=True), "ceiling must fall monotonically with speed"
    assert np.interp(5.0, bp, vals) == 1148
    assert np.interp(14.5, bp, vals) == 620
    assert np.interp(35.0, bp, vals) == 620

  def test_steer_delta_up_matches_the_eps_rate_limit_at_this_steer_step(self, cx5_2022_params):
    # The EPS rate limit is per unit TIME (~1200 units/s), while STEER_DELTA_UP is per frame,
    # so the two are only matched at STEER_STEP = 1. Changing one without the other silently
    # rescales the commanded slew rate.
    rate_hz = 1.0 / DT_CTRL / CarControllerParams.STEER_STEP
    assert cx5_2022_params.STEER_DELTA_UP * rate_hz == pytest.approx(1200, rel=0.01)

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

  def test_alert_command_relays_state_but_not_the_tja_churn(self, packer):
    # camera error and line state pass through to the dash; the camera's own TJA/CTS state
    # machine churns against steering it did not command (442 TJA_TRANSITION toggles in 22
    # min, route 0000010b) and relaying it flapped the dash, so those two fields stay zeroed
    cam_msg = {"LINE_VISIBLE": 1, "LINE_NOT_VISIBLE": 0, "LANE_LINES": 2, "BIT1": 1,
               "BIT2": 0, "BIT3": 0, "NO_ERR_BIT": 0, "ERR_BIT": 1,
               "TJA": 4, "TJA_TRANSITION": 3, "S1": 1, "S1_HBEAM": 0}
    dat = mazdacan.create_alert_command(packer, cam_msg, ldw=False, steer_required=False)[1]
    cp = CANParser("mazda_2017", [("CAM_LANEINFO", float("nan"))], 0)
    cp.update([(0, [(0x440, dat, 0)])])
    out = cp.vl["CAM_LANEINFO"]
    assert out["ERR_BIT"] == 1 and out["LINE_VISIBLE"] == 1 and out["LANE_LINES"] == 2 and out["S1"] == 1
    assert out["TJA"] == 0 and out["TJA_TRANSITION"] == 0

  def test_crz_info_standby_matches_stock(self, packer):
    for counter in range(16):
      checksum = (0x5d - counter) & 0xff
      expected = f"01ffe3ffc000{counter:02x}{checksum:02x}"
      dat = mazdacan.create_acc_command(packer, 0, counter, 0.0, long_active=False, acc_available=False)[1]
      assert dat.hex() == expected

  def test_crz_info_armed_idle_matches_stock(self, packer):
    # armed-idle pegs the command like standby (47,752/47,752 stock armed-idle frames carry
    # raw 8190) and follows the brake on ACC_SET_ALLOWED; the zero-command armed-idle this
    # used to emit exists nowhere in the stock corpus
    for brake_pressed, byte4, base in ((False, 0xc4, 0xd9), (True, 0xc0, 0xdd)):
      for counter in range(16):
        checksum = (base - counter) & 0xff
        expected = f"01ffe3ff{byte4:02x}80{counter:02x}{checksum:02x}"
        dat = mazdacan.create_acc_command(packer, 0, counter, 0.0, long_active=False, acc_available=True,
                                          brake_pressed=brake_pressed)[1]
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
    dat = mazdacan.create_acc_command(packer, 0, counter, accel, long_active=True, acc_available=False,
                                      stopping=stopping, resume_unlatching=unlatching)[1]
    assert dat.hex() == expected

  def test_crz_info_accel_encoding_and_checksum(self, packer):
    # the packed command must round-trip at the 0.001 factor and carry a valid masked-bit
    # checksum over the whole command window, stop bits set or not
    for raw in range(-3500, 2001, 137):
      for stopping in (False, True):
        dat = mazdacan.create_acc_command(packer, 0, raw % 16, raw / 1000.0, long_active=True, acc_available=False,
                                          stopping=stopping)[1]
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
    defaults = dict(long_engaged=True, stopping=False, standstill=False, plan_accel=-1.024,
                    brake_hold=False, gas_pressed=False)
    defaults.update(kwargs)
    for _ in range(frames):
      sm.update(**defaults)
    return sm

  def test_holds_while_the_plan_is_stopping(self, sm):
    self.run(sm, 1)
    assert not sm.holding
    self.run(sm, 1, stopping=True)
    assert sm.holding and sm.stop_bits and sm.acc_active_2
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
    # the release is debounced: a plan asking to move for less than the window changes nothing
    # (the body keeps its own latch until the pulse plays, so brake_hold stays up here)
    self.run(sm, RELEASE_DEBOUNCE_FRAMES - 1, standstill=True, brake_hold=True, plan_accel=0.1)
    assert sm.holding and not sm.resume_unlatching
    self.run(sm, 1, standstill=True, brake_hold=True, plan_accel=0.1)
    assert not sm.holding and not sm.car_has_hold
    # the body owned the brakes, so this is the latched family -- but the pulse is deferred:
    # the command relaxes first and the body gets RESUME_PULSE_DEFER_T to let go by itself
    assert sm.latched_release and not sm.resume_unlatching
    assert sm.pulse_deferred_frames > 0

  def test_release_holds_for_as_long_as_the_plan_wants_to_move(self, sm):
    # the failed-resume regression: no release window to run out from under the plan
    self.run(sm, 1, stopping=True)
    self.run(sm, 100, stopping=True, standstill=True)
    self.run(sm, int(5.0 / DT_CTRL), standstill=True, plan_accel=0.4)
    assert not sm.holding and not sm.stop_bits

  def test_hold_comes_back_if_the_plan_changes_its_mind(self, sm):
    self.run(sm, 1, stopping=True)
    self.run(sm, 100, stopping=True, standstill=True)
    self.run(sm, RELEASE_DEBOUNCE_FRAMES, standstill=True, plan_accel=0.2)
    assert not sm.holding
    # nothing was latched, so this release emits no unlatch bit at all, deferred or otherwise
    assert sm.unlatch_frames == 0 and not sm.resume_unlatching
    self.run(sm, 1, stopping=True, standstill=True, plan_accel=-1.0)
    assert sm.holding
    assert not sm.resume_unlatching and sm.unlatch_frames == 0
    assert sm.stop_bits

  def test_never_latched_release_emits_no_pulse(self, sm):
    # a never-latched release has nothing latched to unlatch, so it puts no RESUME_UNLATCHING
    # on the wire at all. Stock blips here, but every pulse this port has emitted latched the
    # camera's SCBS fault (4/4), and mimicking a blip that unlatches nothing is not worth one
    self.run(sm, 1, stopping=True)
    self.run(sm, 100, stopping=True, standstill=True)
    assert not sm.resume_unlatching
    self.run(sm, RELEASE_DEBOUNCE_FRAMES, standstill=True, plan_accel=0.1)
    assert not sm.holding and not sm.latched_release
    self.run(sm, int(1.0 / DT_CTRL), standstill=True, plan_accel=0.1)
    assert not sm.resume_unlatching and sm.unlatch_frames == 0

  def test_latched_release_skips_the_pulse_when_the_body_lets_go(self, sm):
    # the common case the deferral exists for: the body drops GEAR.BRAKE_HOLD off the
    # relaxing command, so no unlatch bit ever reaches the camera
    self.run(sm, 1, stopping=True)
    self.run(sm, 100, stopping=True, standstill=True, brake_hold=True)
    self.run(sm, RELEASE_DEBOUNCE_FRAMES, standstill=True, brake_hold=True, plan_accel=0.1)
    assert sm.latched_release and not sm.resume_unlatching
    # body lets go a few frames in, well inside the grace period
    self.run(sm, 5, standstill=True, brake_hold=False, plan_accel=0.1)
    assert sm.pulse_deferred_frames == 0
    self.run(sm, int(1.0 / DT_CTRL), standstill=True, brake_hold=False, plan_accel=0.1)
    assert not sm.resume_unlatching, "pulse fired even though the body had already released"

  def test_latched_release_falls_back_to_the_pulse_if_the_body_holds_on(self, sm):
    # the safety net: a body that will not let go still gets stock's pulse, because a car
    # that will not move is worse than the SCBS latch
    self.run(sm, 1, stopping=True)
    self.run(sm, 100, stopping=True, standstill=True, brake_hold=True)
    self.run(sm, RELEASE_DEBOUNCE_FRAMES, standstill=True, brake_hold=True, plan_accel=0.1)
    assert not sm.resume_unlatching
    self.run(sm, RESUME_PULSE_DEFER_FRAMES - 1, standstill=True, brake_hold=True, plan_accel=0.1)
    assert not sm.resume_unlatching, "pulse fired before the grace period expired"
    self.run(sm, 1, standstill=True, brake_hold=True, plan_accel=0.1)
    assert sm.resume_unlatching
    self.run(sm, RESUME_UNLATCH_LATCHED_FRAMES, standstill=True, brake_hold=True, plan_accel=0.1)
    assert not sm.resume_unlatching, "fallback pulse outran its length"

  def test_long_disengage_resets(self, sm):
    self.run(sm, 1, stopping=True)
    self.run(sm, 100, stopping=True, standstill=True, brake_hold=True)
    self.run(sm, 1, long_engaged=False)
    assert not sm.holding and not sm.car_has_hold and not sm.stop_bits

  def test_gas_override_drive_off_releases_the_hold(self, sm):
    # a driver-gas drive-off under an override zeroes the plan's command, so the plan never
    # asks to move but the car does; the stop bits must not follow it up to speed. Stock keeps
    # STOPPING strictly to the final creep, below 0.55 m/s across all rolling frames.
    self.run(sm, 1, stopping=True)
    self.run(sm, 100, stopping=True, standstill=True)
    assert sm.holding
    self.run(sm, 1, plan_accel=0.0)
    assert not sm.holding and not sm.stop_bits and not sm.resume_unlatching

  def test_stop_abort_releases(self, sm):
    self.run(sm, 1, stopping=True)
    assert sm.holding
    # lead speeds up again before the car reaches standstill
    self.run(sm, 1, stopping=False, plan_accel=0.3)
    assert not sm.holding

  def test_driver_gas_releases_the_hold_without_a_pulse(self, sm):
    # the driver's pedal outranks the hold, the way Toyota's PCM lets the pedal outrank its
    # standstill request -- but the pulse is the ACC's resume protocol, not the driver's:
    # stock's captured gas-ended hold drops the stop bits with no RESUME_UNLATCHING at all,
    # and pulsing there latched an SCBS fault (route 00000103 t+163.8)
    self.run(sm, 1, stopping=True)
    self.run(sm, 100, stopping=True, standstill=True)
    assert sm.holding
    self.run(sm, 1, stopping=True, standstill=True, gas_pressed=True)
    assert not sm.holding and not sm.resume_unlatching, "gas release must not fire the ACC resume pulse"
    # no re-hold while the pedal is down, and a fresh hold once it lifts with the car stopped
    self.run(sm, RESUME_UNLATCH_LATCHED_FRAMES + 5, stopping=True, standstill=True, gas_pressed=True)
    assert not sm.holding and not sm.resume_unlatching
    self.run(sm, 1, stopping=True, standstill=True)
    assert sm.holding

  def test_plan_flap_below_the_debounce_never_releases(self, sm):
    # the SCBS-axis contamination shape: at a held standstill the lead inches forward and
    # stops, the plan flapping across zero. Sub-debounce flaps must not release at all, and
    # no frame may ever carry the stop bits and the release pulse together
    self.run(sm, 1, stopping=True)
    self.run(sm, 100, stopping=True, standstill=True)
    for i in range(600):
      accel = 0.3 if (i // 10) % 2 == 0 else -1.0  # 0.1 s swings, below the 0.2 s debounce
      sm.update(long_engaged=True, stopping=accel < 0, standstill=True, plan_accel=accel,
                brake_hold=False, gas_pressed=False)
      assert not (sm.stop_bits and sm.resume_unlatching), "stop bits and pulse on one frame"
      assert not sm.resume_unlatching, "a sub-debounce flap fired a release pulse"
    assert sm.holding

  @pytest.mark.parametrize("brake_hold", [False, True])
  def test_slow_flap_never_mixes_stop_bits_with_the_pulse(self, sm, brake_hold):
    # swings long enough to release each time. Nothing latched (brake_hold False) must never
    # put an unlatch bit on the wire; a body that holds on through every swing falls back to
    # at most one pulse per release, and a re-hold mid-pulse waits it out before re-asserting
    # the stop bits
    self.run(sm, 1, stopping=True)
    self.run(sm, 100, stopping=True, standstill=True, brake_hold=brake_hold)
    pulses = 0
    prev_unlatch = False
    swing = RELEASE_DEBOUNCE_FRAMES + RESUME_PULSE_DEFER_FRAMES + 30  # long enough to reach the fallback
    for i in range(1200):
      accel = 0.3 if (i // swing) % 2 == 0 else -1.0
      sm.update(long_engaged=True, stopping=accel < 0, standstill=True, plan_accel=accel,
                brake_hold=brake_hold, gas_pressed=False)
      assert not (sm.stop_bits and sm.resume_unlatching), "stop bits and pulse on one frame"
      pulses += int(sm.resume_unlatching and not prev_unlatch)
      prev_unlatch = sm.resume_unlatching
    if brake_hold:
      assert pulses > 0
      assert pulses <= 1 + 1200 // (2 * swing), "more pulses than releases"
    else:
      assert pulses == 0, "a never-latched release put an unlatch bit on the wire"

  def test_latched_pulse_runs_to_completion_through_a_re_hold(self, sm):
    # a latched pulse spans the body's actual unlatch, so a re-hold mid-pulse waits it out
    # (stop bits blocked, stock never emits STOPPING with RESUME_UNLATCHING) instead of
    # cancelling it; a second release cannot fire a fresh pulse before the first ends because
    # the release debounce is at least as long as any pulse window
    assert RELEASE_DEBOUNCE_FRAMES >= RESUME_UNLATCH_LATCHED_FRAMES
    self.run(sm, 1, stopping=True)
    self.run(sm, 100, stopping=True, standstill=True, brake_hold=True)
    self.run(sm, RELEASE_DEBOUNCE_FRAMES, standstill=True, brake_hold=True, plan_accel=0.3)
    assert sm.latched_release and not sm.resume_unlatching  # deferred while the body holds on
    self.run(sm, RESUME_PULSE_DEFER_FRAMES, standstill=True, brake_hold=True, plan_accel=0.3)
    assert sm.resume_unlatching, "the body never let go, so the fallback pulse must fire"
    self.run(sm, 3, standstill=True, plan_accel=0.3)
    self.run(sm, 1, stopping=True, standstill=True)  # re-hold mid-pulse, body already let go
    assert sm.holding and not sm.stop_bits
    assert sm.resume_unlatching, "a latched pulse mid-release must run to completion"
    self.run(sm, RESUME_UNLATCH_LATCHED_FRAMES, stopping=True, standstill=True, plan_accel=-1.0)
    assert sm.holding and sm.stop_bits and not sm.resume_unlatching


class TestAdvertisedLead:
  """has_lead, the phase and the track slot are one decision, so they are asserted together."""

  @pytest.fixture
  def al(self):
    return AdvertisedLead()

  @staticmethod
  def run(al, frames, **kwargs):
    defaults = dict(lead_visible=True, d_rel=40.0, v_rel=0.0, holding=False)
    defaults.update(kwargs)
    for _ in range(frames):
      al.update(**defaults)
    return al

  def test_lead_follows_only_a_steady_state(self, al):
    # a lead is adopted once leadVisible has held for the debounce window, not before
    self.run(al, LEAD_DEBOUNCE_FRAMES - 1)
    assert not al.has_lead and al.ctrl_phase == 0
    self.run(al, 1)
    assert al.has_lead and al.lead == (40.0, 0.0) and al.ctrl_phase == 2
    # and dropped the same way
    self.run(al, LEAD_DEBOUNCE_FRAMES - 1, lead_visible=False, d_rel=0.)
    assert al.has_lead
    self.run(al, 1, lead_visible=False, d_rel=0.)
    assert not al.has_lead and al.ctrl_phase == 0

  def test_lead_flicker_never_reaches_the_bus(self, al):
    # the measured failure: a marginal 120 m vision lead toggled leadVisible 6 times in 1.4 s
    # (route 6bb2dc61c4 t+400); none of it may reach RADAR_HAS_LEAD or the track slot
    for frames, visible in ((15, True), (5, False), (7, True), (13, False), (10, True)):
      self.run(al, frames, lead_visible=visible)
      assert not al.has_lead, "a flickering lead leaked through the debounce"

  def test_measurement_is_coasted_across_a_dropout(self, al):
    # leadOne goes to zero the instant vision drops the lead, well before the debounce expires.
    # Advertising a fabricated stand-in there put a stationary object 10.25 m dead ahead on the
    # bus at 22 m/s; the last real measurement carries the gap instead -- propagated by its own
    # range rate, never repeated frozen (a frozen range is the camera's proven SCBS trigger)
    self.run(al, 2 * LEAD_DEBOUNCE_FRAMES, d_rel=120.0, v_rel=0.5)
    assert al.lead == (120.0, 0.5)
    coast_frames = LEAD_DEBOUNCE_FRAMES - 1
    self.run(al, coast_frames, lead_visible=False, d_rel=0., v_rel=0.)
    assert al.lead is not None, "dropped the measurement inside the debounce window"
    d, v = al.lead
    assert v == 0.5
    assert d == pytest.approx(120.0 + 0.5 * coast_frames * DT_CTRL, abs=1e-6), \
      "the coast must propagate the range, not freeze it"

  def test_holding_reports_the_stop_phase_only_with_a_lead(self, al):
    self.run(al, 2 * LEAD_DEBOUNCE_FRAMES, holding=True)
    assert al.ctrl_phase == 3
    self.run(al, 2 * LEAD_DEBOUNCE_FRAMES, lead_visible=False, d_rel=0., holding=True)
    assert not al.has_lead and al.ctrl_phase == 0

def _mock_cc(long_active=True, accel=0.5, long_state=None, standstill=False, gas=False,
             resume=False, cancel=False, lead_visible=True, gap=2, available=True,
             stock_radar_alive=False, fsc_settled=True, handback=False, cruise_engaged=False,
             enabled=None, lead_d_rel=12.0, lead_v_rel=0.0, brake_hold=False, brake_pressed=False,
             radar_was_silenced=False):
  # openpilot is enabled whenever it is longitudinally active; a gas override is the case
  # where it stays enabled with longActive low. The mock carries everything the full
  # CarController.update() path reads, so tests can drive update() as well as
  # update_longitudinal() from the one builder.
  enabled = long_active if enabled is None else enabled
  out = SimpleNamespace(standstill=standstill, gasPressed=gas, brakePressed=brake_pressed,
                        vEgoRaw=0., steeringTorque=0.,
                        cruiseState=SimpleNamespace(available=available, enabled=cruise_engaged))
  actuators = SimpleNamespace(accel=accel, longControlState=long_state, torque=0.,
                              as_builder=lambda: SimpleNamespace(torque=0., torqueOutputCan=0, accel=0.))
  cruise = SimpleNamespace(resume=resume, cancel=cancel)
  hud = SimpleNamespace(leadVisible=lead_visible, leadDistanceBars=gap, visualAlert=None)
  cc = SimpleNamespace(enabled=enabled, longActive=long_active, latActive=False,
                       actuators=actuators, cruiseControl=cruise, hudControl=hud)
  cc_sp = SimpleNamespace(stockEcuHandBack=handback,
                          leadOne=SimpleNamespace(dRel=lead_d_rel, vRel=lead_v_rel))
  cs = SimpleNamespace(out=out, resume_button=0, brake_hold=brake_hold,
                       stock_radar_alive=stock_radar_alive, fsc_settled=fsc_settled,
                       radar_session_refused=False, radar_was_silenced=radar_was_silenced,
                       crz_btns_counter=0, cancel_button=0, lkas_allowed_speed=True,
                       cam_lkas={"BIT_1": 0, "ERR_BIT_1": 0, "ERR_BIT_2": 0})
  return cc, cc_sp, cs


@pytest.fixture
def cc():
  CP = CarInterface.get_params(CAR.MAZDA_CX5_2022, {0: {}, 1: {}, 2: {}}, [], alpha_long=True,
                               is_release=False, docs=False)
  CP_SP = CarInterface.get_params_sp(CP, CAR.MAZDA_CX5_2022, {0: {}, 1: {}, 2: {}}, [], True, False, False)
  assert CP.openpilotLongitudinalControl
  return CarController({Bus.pt: "mazda_2017"}, CP, CP_SP)


@pytest.fixture
def stock_cc():
  CP = CarInterface.get_params(CAR.MAZDA_CX5_2022, {0: {}, 1: {}, 2: {}}, [], alpha_long=False,
                               is_release=False, docs=False)
  CP_SP = CarInterface.get_params_sp(CP, CAR.MAZDA_CX5_2022, {0: {}, 1: {}, 2: {}}, [], False, False, False)
  assert not CP.openpilotLongitudinalControl
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


CRZ_BTNS = 0x9d

# create_radar_frames stamps the counter into the last byte, so an empty slot is the first seven
_EMPTY_TRACK = mazdacan.RADAR_TRACK_MSGS[0x364][:7]


def _decode(msg, addr, dat):
  """CANParser view of a single frame."""
  cp = CANParser("mazda_2017", [(msg, float("nan"))], 0)
  cp.update([(0, [(addr, dat, 0)])])
  return cp.vl[msg]


def _frames(sends, addr, bus=0):
  return [d for a, d, b in sends if a == addr and b == bus]


def _frame(sends, addr, bus=0):
  return next(iter(_frames(sends, addr, bus)), None)


def _track_occupied(dat):
  return dat[:7] != _EMPTY_TRACK


def _crz_ctrl(dat):
  """(RADAR_HAS_LEAD, RADAR_LEAD_RELATIVE_DISTANCE) from a CRZ_CTRL frame."""
  v = _decode("CRZ_CTRL", 0x21c, dat)
  return int(v["RADAR_HAS_LEAD"]), int(v["RADAR_LEAD_RELATIVE_DISTANCE"])


def _lead_track(dat):
  """(DIST_OBJ, RELV_OBJ) decoded from a 0x364 track frame."""
  v = _decode("RADAR_TRACK_364", 0x364, dat)
  return v["DIST_OBJ"], v["RELV_OBJ"]


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
                    gas=True, cruise_engaged=True)
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
            cruise_engaged=True)
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
            standstill=True, cruise_engaged=True)
    assert cc.accel_last == 0., f"hold not released for the driver's gas: {cc.accel_last}"

  def test_release_command_holds_through_the_debounce_then_jumps(self, cc):
    """Stock never lets ACCEL_CMD climb while STOPPING is asserted: through the release
    debounce the command stays at the hold value. Once the stop bits drop it relax-jumps
    into stock's release band and ramps. Pre-ramping toward the plan during the debounce put
    the zero-cross inside the pulse (route 00000100 t+353); slewing up off the hold value put
    hold-grade braking under it (route 00000053 t+714.8). Both latched SCBS."""
    long = structs.CarControl.Actuators.LongControlState
    lead = dict(lead_visible=True, lead_d_rel=4.0, lead_v_rel=0.0)
    for _ in range(int(0.5 / 0.01)):
      _step(cc, long_state=long.stopping, accel=-1.5, standstill=False, **lead)
    for _ in range(int(3.0 / 0.01)):
      _step(cc, long_state=long.stopping, accel=-1.3, standstill=True, **lead)
    assert cc.accel_last == pytest.approx(-1.3)

    rows = []
    for _ in range(int(1.5 / 0.01)):
      sends = _step(cc, long_state=long.pid, accel=1.0, standstill=True, **lead)
      dat = next((d for a, d, b in sends if a == 0x21b and b == 0), None)
      if dat is not None:
        rows.append((decode_accel_cmd_raw(dat), (dat[5] >> 2) & 1, (dat[6] >> 6) & 1))

    debounce = [r for r in rows if r[1]]
    assert debounce, "no stop-bit frames through the release debounce"
    assert all(cmd == -1300 for cmd, _, _ in debounce), \
      f"command moved off the hold while STOPPING was asserted: {sorted({c for c, _, _ in debounce})}"

    # nothing was latched, so no unlatch bit goes out at all
    assert not any(unl for _, _, unl in rows), "a never-latched release pulsed"
    assert max(cmd for cmd, _, _ in rows) > 500, "command never ramped up after the release"

  def test_near_zero_hold_release_emits_no_pulse(self, cc):
    # a no-lead hold relaxes the plan to ~0, so the release ramp would cross zero in the first
    # pulse frame -- the shape behind the routes 000000fe t+44.54 / 00000100 t+353.18 latches.
    # Nothing is latched here, so the release now carries no unlatch bit for it to land in.
    long = structs.CarControl.Actuators.LongControlState
    for _ in range(int(0.5 / 0.01)):
      _step(cc, long_state=long.stopping, accel=-0.5, standstill=False)
    for _ in range(int(2.0 / 0.01)):
      _step(cc, long_state=long.stopping, accel=-0.02, standstill=True)
    assert cc.accel_last == pytest.approx(-0.02)

    rows = []
    for _ in range(int(1.5 / 0.01)):
      sends = _step(cc, long_state=long.pid, accel=1.0, standstill=True)
      dat = next((d for a, d, b in sends if a == 0x21b and b == 0), None)
      if dat is not None:
        rows.append((decode_accel_cmd_raw(dat), (dat[6] >> 6) & 1))

    assert not any(unl for _, unl in rows), "a never-latched release pulsed"
    assert max(cmd for cmd, _ in rows) > 500, "command never ramped up after the release"

  def test_never_latched_release_speaks_the_stock_wire_grammar(self, cc):
    """Route 00000053 t+714.8 (second CX-5): slewing off the hold value under a 13-frame pulse
    put hold-grade braking beneath RESUME_UNLATCHING, a (stop, unlatch, cmd) tuple stock never
    emits, and the camera latched SCBS 90 ms in with a real departing lead advertised. Stock's
    never-latched grammar (33-pulse census): the command relax-jumps into the -0.27..-0.11 band
    in one frame and the ramp climbs ~+25 raw per wire frame. Stock also blips RESUME_UNLATCHING
    here; we do not -- nothing is latched, and 4 of 4 pulses this port ever emitted latched the
    camera -- so the blip assertions are replaced by requiring no unlatch bit at all."""
    long = structs.CarControl.Actuators.LongControlState
    lead = dict(lead_visible=True, lead_d_rel=4.0, lead_v_rel=0.0)
    for _ in range(int(0.5 / 0.01)):
      _step(cc, long_state=long.stopping, accel=-1.0, standstill=False, **lead)
    for _ in range(int(2.0 / 0.01)):
      _step(cc, long_state=long.stopping, accel=-1.024, standstill=True, **lead)
    assert cc.accel_last == pytest.approx(-1.024)

    rows = []
    for _ in range(int(1.5 / 0.01)):
      sends = _step(cc, long_state=long.pid, accel=0.45, standstill=True, **lead)
      dat = next((d for a, d, b in sends if a == 0x21b and b == 0), None)
      if dat is not None:
        rows.append((decode_accel_cmd_raw(dat), (dat[5] >> 2) & 1, (dat[6] >> 6) & 1))

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
  def test_latched_release_speaks_the_stock_pulse_shape(self, cc, drop_wire_frames):
    # a body-latched hold releases with a 9-wire-frame pulse: the command sits pinned at the
    # relaxed -1 raw for as long as the body still reports GEAR.BRAKE_HOLD (as in every latched
    # release of the census -- climbing before the drop faulted the camera 90 ms in, route
    # 00000115 t+381.3), then climbs stock's ~+25 raw per frame ramp, peaking inside stock's
    # family and never past the +0.25 ceiling (census: 6-11 frames, hold drop 1-3 frames in,
    # cmd -1 climbing to +24..+342)
    long = structs.CarControl.Actuators.LongControlState
    lead = dict(lead_visible=True, lead_d_rel=4.0, lead_v_rel=0.0)
    for _ in range(int(0.5 / 0.01)):
      _step(cc, long_state=long.stopping, accel=-1.5, standstill=False, **lead)
    for _ in range(int(2.0 / 0.01)):
      _step(cc, long_state=long.stopping, accel=-1.3, standstill=True, brake_hold=True, **lead)
    assert cc.accel_last == pytest.approx(CarControllerParams.ACCEL_HOLD_LATCHED)

    # the body reacts to the pulse: BRAKE_HOLD drops 1-3 wire frames after it starts (census)
    rows = []
    pulse_started = None
    for i in range(int(1.5 / 0.01)):
      body_holds = pulse_started is None or i < pulse_started + 2 * drop_wire_frames
      sends = _step(cc, long_state=long.pid, accel=1.0, standstill=True, brake_hold=body_holds, **lead)
      dat = next((d for a, d, b in sends if a == 0x21b and b == 0), None)
      if dat is not None:
        unl = (dat[6] >> 6) & 1
        rows.append((decode_accel_cmd_raw(dat), unl, body_holds))
        if pulse_started is None and unl:
          pulse_started = i

    pulse = [(cmd, held) for cmd, unl, held in rows if unl]
    cap = round(CarControllerParams.ACCEL_RESUME_PULSE_MAX * 1000)
    assert len(pulse) == RESUME_UNLATCH_LATCHED_FRAMES // 2, f"pulse ran {len(pulse)} wire frames"
    # the contract the route 115 fault turned on: no pulse frame moves off the relaxed hold
    # while the body still reports its latch
    pinned = [cmd for cmd, held in pulse if held]
    assert len(pinned) == drop_wire_frames and all(cmd == -1 for cmd in pinned), \
      f"command moved under the latched hold: {pinned}"
    # then the ramp: stock's +25 raw per wire frame from the relaxed hold
    ramp = [cmd for cmd, held in pulse if not held]
    assert -1 <= ramp[0] <= 15, f"ramp must start off the relaxed hold: {ramp[0]}"
    assert all(20 <= b - a <= 30 for a, b in zip(ramp, ramp[1:], strict=False)), f"off the stock ramp: {ramp}"
    assert -1 + (len(ramp) - 1) * 20 <= max(ramp) <= cap, f"in-pulse peak outside the ramp's own family: {max(ramp)}"
    assert max(cmd for cmd, _, _ in rows) > cap, "command never ramped past the cap after the pulse"

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

  def test_hold_with_nothing_ahead_advertises_nothing(self, cc):
    # No fabricated object. The body does not decide the latch on the advertisement: across 32
    # stock engaged standstills the radar said has_lead=1 in every one, yet 23 latched
    # GEAR.BRAKE_HOLD and 9 did not (one held 104 s), and 89 of 115 stock latches happened at
    # has_lead=0 / phase=0. A phantom the camera can refute is the SCBS trigger.
    long = structs.CarControl.Actuators.LongControlState
    held, ctrls = [], []
    for _ in range(400):
      sends = _step(cc, long_state=long.stopping, accel=-1.024, standstill=True,
                    lead_visible=False, lead_d_rel=0.0, cruise_engaged=True)
      held += _frames(sends, 0x364)
      ctrls += _frames(sends, 0x21c)
    assert held and ctrls
    assert not any(map(_track_occupied, held)), "fabricated a lead for a hold with nothing ahead"
    assert all(_crz_ctrl(d) == (0, 0) for d in ctrls), "advertised a lead with nothing in view"
    # and the hold itself is untouched: the plan's brake and the stop bits still go out
    assert cc.stop_and_go.holding and cc.stop_and_go.stop_bits

  def test_vision_lead_dropout_does_not_fabricate_a_lead_at_speed(self, cc):
    # leadOne goes to zero the instant the vision lead drops while sm.lead_visible is still
    # latched. Falling through to the hold fallback there put a stationary object 10.25 m dead
    # ahead on the bus at 22 m/s, 20 times across the two 2026-08-25 drives.
    long = structs.CarControl.Actuators.LongControlState
    for _ in range(200):  # settle a real lead at 120 m while cruising
      _step(cc, long_state=long.pid, accel=0.5, lead_visible=True, lead_d_rel=120.0,
            lead_v_rel=0.5, cruise_engaged=True)

    dropped = []
    for _ in range(int(LEAD_DEBOUNCE_FRAMES * 0.8)):  # inside the debounce window
      dropped += _frames(_step(cc, long_state=long.pid, accel=0.5, lead_visible=False,
                               lead_d_rel=0.0, lead_v_rel=0.0, cruise_engaged=True), 0x364)
    assert dropped
    for d in dropped:
      dist = _lead_track(d)[0]
      assert dist == pytest.approx(120.0, abs=1.0), f"track teleported to {dist} m"

  def test_has_lead_phase_and_track_never_disagree(self, cc):
    # stock pairs all three absolutely: has_lead=0 <=> phase=0, and RADAR_HAS_LEAD=1 with all six
    # slots empty appears 8 times in 1,095,826 stock samples. We shipped has_lead=0 with phase=1
    # for 22-84% of every engaged drive before this was derived from one decision.
    long = structs.CarControl.Actuators.LongControlState
    cases = [
      dict(long_state=long.pid, accel=0.5, lead_visible=True, lead_d_rel=40.0),
      dict(long_state=long.pid, accel=0.5, lead_visible=False, lead_d_rel=0.0),
      dict(long_state=long.stopping, accel=-1.024, standstill=True, lead_visible=False, lead_d_rel=0.0),
      dict(long_state=long.pid, accel=0.3, standstill=True, lead_visible=False, lead_d_rel=0.0),
      dict(long_state=long.pid, accel=0.5, lead_visible=True, lead_d_rel=0.0),
    ]
    for kw in cases:
      for _ in range(120):
        sends = _step(cc, cruise_engaged=True, **kw)
        trk, ctl = _frame(sends, 0x364), _frame(sends, 0x21c)
        if trk is None or ctl is None:
          continue
        has_lead, phase = _crz_ctrl(ctl)
        assert bool(has_lead) == _track_occupied(trk), f"has_lead/track disagree for {kw}"
        assert (phase == 0) == (has_lead == 0), f"has_lead/phase disagree for {kw}"

  def test_lead_survives_disengagement(self, cc):
    # perception is engagement-independent: stock advertises RADAR_HAS_LEAD=1 with cruise off in
    # 19.5% of all frames. Dropping the advertisement at disengage made a real car 4.5 m ahead
    # vanish from the bus in one frame while the driver braked toward it, and the camera ran its
    # SCBS display six seconds (route 0000004d t+212)
    long = structs.CarControl.Actuators.LongControlState
    for _ in range(120):
      _step(cc, cruise_engaged=True, lead_d_rel=4.8, accel=-0.5)
    for _ in range(60):
      sends = _step(cc, long_active=False, enabled=False, long_state=long.off, accel=0.,
                    lead_d_rel=4.8)
      trk, ctl = _frame(sends, 0x364), _frame(sends, 0x21c)
      if ctl is None:
        continue
      has_lead, phase = _crz_ctrl(ctl)
      assert has_lead == 1 and phase != 0, "disengaging dropped a real lead off the bus"
      if trk is not None:
        assert _lead_track(trk)[0] == pytest.approx(4.8, abs=0.1)

  def test_no_resume_button_while_openpilot_owns_longitudinal(self, cc):
    # We are the ACC here, so the hold is released in-protocol. The car's own MRCC never presses
    # RES either: 0 of 23 stock body-latched-hold releases put one on the bus. A press would also
    # put a second writer on CRZ_BTNS, which ICBM owns.
    for accel in (0.3, -1.024):
      for standstill in (True, False):
        control, _, _ = _mock_cc(standstill=standstill, accel=accel, resume=True)
        assert not cc.resume_requested(control)

  def test_resume_button_still_sent_with_stock_longitudinal(self, stock_cc):
    # stock ACC owns the hold there, and the button is the only lever openpilot has on it
    control, _, _ = _mock_cc(standstill=True, accel=0.3, resume=True)
    assert stock_cc.resume_requested(control)

    control, _, _ = _mock_cc(standstill=True, accel=0.3, resume=False)
    assert not stock_cc.resume_requested(control)

  def test_body_latched_hold_releases_in_protocol(self, cc):
    # the release the button used to stand in for: stop bits already relaxed to the body, then
    # the plan asks to move. The unlatch pulse is deferred -- the relaxed command is given
    # RESUME_PULSE_DEFER_T to get the body to let go on its own before we resort to it.
    long = structs.CarControl.Actuators.LongControlState
    for _ in range(200):
      _step(cc, long_state=long.stopping, accel=-1.024, standstill=True,
            cruise_engaged=True, brake_hold=True)
    assert cc.stop_and_go.holding and cc.stop_and_go.car_has_hold
    assert not cc.stop_and_go.stop_bits  # body owns the brakes, stock relaxes here

    for _ in range(RELEASE_DEBOUNCE_FRAMES):
      sends = _step(cc, long_state=long.pid, accel=0.3, standstill=True,
                    cruise_engaged=True, brake_hold=True)
      assert not any(a == CRZ_BTNS for a, _, _ in sends), "CRZ_BTNS written at the release"
    assert not cc.stop_and_go.holding
    assert not cc.stop_and_go.resume_unlatching, "pulse fired before the body was given a chance"
    for _ in range(RESUME_PULSE_DEFER_FRAMES):
      _step(cc, long_state=long.pid, accel=0.3, standstill=True,
            cruise_engaged=True, brake_hold=True)
    assert cc.stop_and_go.resume_unlatching, "body never let go, so the fallback must pulse"

  def test_gas_pedal_without_cruise_stays_disengaged(self, cc):
    # gas pressed while openpilot is not enabled must not advertise an engaged ACC
    off = structs.CarControl.Actuators.LongControlState.off
    cc.frame = 0
    sends = _step(cc, long_active=False, enabled=False, long_state=off, gas=True, available=True)
    info = next(dat for a, dat, b in sends if a == 0x21b and b == 0)
    assert info.hex().startswith("01ffe3ffc480")  # armed-but-idle pattern, command pegged

  def test_disengaged_emits_stock_patterns(self, cc):
    off = structs.CarControl.Actuators.LongControlState.off
    # main off, not available: the exact standby pattern the panda allowlists byte-for-byte
    cc.frame = 0
    sends = _step(cc, long_active=False, long_state=off, available=False)
    info = next(dat for a, dat, b in sends if a == 0x21b and b == 0)
    assert info.hex().startswith("01ffe3ffc000")
    # MRCC armed but not engaged: the command stays pegged and ACC_SET_ALLOWED follows the
    # brake, exactly the two patterns stock alternates between at an armed idle
    cc.frame = 0
    sends = _step(cc, long_active=False, long_state=off, available=True)
    info = next(dat for a, dat, b in sends if a == 0x21b and b == 0)
    assert info.hex().startswith("01ffe3ffc480")
    cc.frame = 0
    sends = _step(cc, long_active=False, long_state=off, available=True, brake_pressed=True)
    info = next(dat for a, dat, b in sends if a == 0x21b and b == 0)
    assert info.hex().startswith("01ffe3ffc080")


class TestCancelCarveOut:
  """controlsd raises cruiseControl.cancel whenever cruiseState.enabled has no matching
  CC.enabled (mazda reports pcmCruise). While the stock radar still owns the bus that
  engagement is the driver's own stock MRCC and a CANCEL turns its main off within ~100 ms,
  so the documented stay-stock fallback used to leave the driver with no cruise at all. Once
  the radar has been silenced a stock engagement is impossible and cancel handles desync."""

  def _full_update(self, cc, cancel, radar_was_silenced, stock_radar_alive):
    control, control_sp, carstate = _mock_cc(long_active=False, enabled=False, accel=0.,
                                             long_state=structs.CarControl.Actuators.LongControlState.off,
                                             available=False, cruise_engaged=True, cancel=cancel,
                                             stock_radar_alive=stock_radar_alive, fsc_settled=False,
                                             radar_was_silenced=radar_was_silenced)
    cc.frame = 10  # off the 50-frame alert cadence, on the 10-frame cancel cadence
    _, sends = cc.update(control, control_sp, carstate, 0)
    return [a for a, _, _ in sends]

  def test_no_cancel_while_the_radar_is_stock(self, cc):
    # pre-teardown settle window, and equally the silencing-failed drive: a driver SET is
    # their own stock MRCC and must be left alone
    addrs = self._full_update(cc, cancel=True, radar_was_silenced=False, stock_radar_alive=True)
    assert CRZ_BTNS not in addrs, "CANCELed the driver's own stock MRCC"

  def test_cancel_still_sent_after_the_teardown(self, cc):
    # post-teardown a stock engagement is impossible: cancel keeps handling state desync
    addrs = self._full_update(cc, cancel=True, radar_was_silenced=True, stock_radar_alive=False)
    assert CRZ_BTNS in addrs

  def test_stock_longitudinal_cancel_unaffected(self, stock_cc):
    addrs = self._full_update(stock_cc, cancel=True, radar_was_silenced=False, stock_radar_alive=True)
    assert CRZ_BTNS in addrs


SESSION_PROG_DAT = bytes([0x02, 0x10, 0x02, 0, 0, 0, 0, 0])
SESSION_DFLT_DAT = bytes([0x02, 0x10, 0x01, 0, 0, 0, 0, 0])
TESTER_PRESENT_DAT = bytes([0x02, 0x3e, 0x80, 0, 0, 0, 0, 0])


class TestRadarSessionBounds:
  """The fire-and-forget UDS session has no readable NRC, so every episode is bounded the
  way disable_ecu bounds its retries."""

  def test_silencing_gives_up_bounded(self):
    m = RadarSessionManager()
    for _ in range(RADAR_SESSION_LIMIT_FRAMES + 2):
      state = m.update(True, True, False, standstill=True, session_refused=False)
    assert state == RadarSessionState.STOCK and m.silencing_failed
    # and stays given up for the drive: stock keeps the bus
    for _ in range(10):
      assert m.update(True, True, False, standstill=True, session_refused=False) == RadarSessionState.STOCK

  def test_negative_response_gives_up_immediately(self):
    # route 000000fe t+15.0 shows the radar answers a session request within 10 ms, so a
    # negative response is definitive: no reason to burn the silence budget
    m = RadarSessionManager()
    m.update(True, True, False, standstill=True, session_refused=False)
    assert m.state == RadarSessionState.SILENCING
    assert m.update(True, True, False, standstill=True, session_refused=True) == RadarSessionState.STOCK
    assert m.silencing_failed

  def test_handback_stops_waiting_for_a_dead_radar(self):
    m = RadarSessionManager()
    m.update(True, False, False, standstill=True, session_refused=False)
    assert m.state == RadarSessionState.SILENCED
    for _ in range(RADAR_SESSION_LIMIT_FRAMES + 2):
      state = m.update(True, False, True, standstill=True, session_refused=False)
    assert state == RadarSessionState.STOCK

  def test_completed_handback_never_resilences(self):
    # the parked toggle-off regression: the monitor's CC_SP assert used to drop after its done
    # latch, the manager read that as a withdrawal, fell to STOCK, and re-entered SILENCING on
    # the same call (parked, gate still passed) -- re-silencing the radar it had just handed
    # back, right before shutdown, leaving it to a degraded unattended S3 recovery
    m = RadarSessionManager()
    m.update(True, False, False, standstill=True, session_refused=False)
    assert m.state == RadarSessionState.SILENCED
    m.update(True, False, True, standstill=True, session_refused=False)
    assert m.state == RadarSessionState.HANDBACK
    assert m.update(True, True, True, standstill=True, session_refused=False) == RadarSessionState.STOCK
    for handback in (True, False):
      for alive in (True, False):
        for _ in range(5):
          assert m.update(True, alive, handback, standstill=True, session_refused=False) == RadarSessionState.STOCK

  def test_withdrawn_handback_allows_retakeover(self):
    # only a hand-back that ran to completion latches: a genuine toggle-flip-back
    # mid-hand-back gets the normal takeover again
    m = RadarSessionManager()
    m.update(True, False, False, standstill=True, session_refused=False)
    m.update(True, False, True, standstill=True, session_refused=False)
    assert m.state == RadarSessionState.HANDBACK
    state = m.update(True, False, False, standstill=True, session_refused=False)
    assert state == RadarSessionState.SILENCED and not m.handback_completed

  def test_silencing_waits_for_standstill_but_adoption_does_not(self):
    # actively silencing disables AEB, so it only starts pre-motion like disable_ecu;
    # adopting an already-quiet radar disables nothing and proceeds anywhere
    m = RadarSessionManager()
    for _ in range(10):
      assert m.update(True, True, False, standstill=False, session_refused=False) == RadarSessionState.STOCK
    assert m.update(True, True, False, standstill=True, session_refused=False) == RadarSessionState.SILENCING
    m2 = RadarSessionManager()
    assert m2.update(True, False, False, standstill=False, session_refused=False) == RadarSessionState.SILENCED


def test_non_gen1_platform_refused_at_admission():
  # one init-time check instead of per-frame guards in the message builders, which every
  # frame layout in mazdacan assumes; the fall-throughs used to emit an all-zero CAM_LKAS
  # and return None from the button builder, straight into can_sends
  CP = CarInterface.get_params(CAR.MAZDA_CX5_2022, {0: {}, 1: {}, 2: {}}, [], alpha_long=False,
                               is_release=False, docs=False)
  CP_SP = CarInterface.get_params_sp(CP, CAR.MAZDA_CX5_2022, {0: {}, 1: {}, 2: {}}, [], False, False, False)
  CP.flags = 0
  with pytest.raises(NotImplementedError):
    CarController({Bus.pt: "mazda_2017"}, CP, CP_SP)


class TestRadarSessionSequencing:
  """Boot teardown deferral and the ordered hand-back: what goes on the bus in each
  radar session state, driven through the real CarController.update_longitudinal."""

  def _step(self, cc, stock_radar_alive, fsc_settled, handback=False, cruise_engaged=False, standstill=True):
    # standstill=True models the parked boot; actively silencing a live radar is gated on it
    off = structs.CarControl.Actuators.LongControlState.off
    return _step(cc, long_active=False, accel=0., long_state=off, lead_visible=False, available=False,
                 stock_radar_alive=stock_radar_alive, fsc_settled=fsc_settled,
                 handback=handback, cruise_engaged=cruise_engaged, standstill=standstill)

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

  def test_completed_handback_stays_stock_after_the_assert_drops(self, cc):
    # CC_SP is rebuilt every frame, so once the toggle monitor's done latch stops asserting
    # the hand-back the manager sees handback=False; a completed hand-back must not turn
    # into a fresh takeover on the very next frame (parked => standstill, gate still passed)
    self._step(cc, stock_radar_alive=False, fsc_settled=True)
    self._step(cc, stock_radar_alive=False, fsc_settled=True, handback=True)
    self._step(cc, stock_radar_alive=True, fsc_settled=True, handback=True)
    for _ in range(200):
      sends = self._step(cc, stock_radar_alive=True, fsc_settled=True, handback=False)
      assert sends == []

  def test_s3_recovery_resilences(self, cc):
    # radar reappears mid-drive (dropped tester present, S3 timeout): re-request the session
    self._step(cc, stock_radar_alive=False, fsc_settled=True)
    cc.frame = CarControllerParams.RADAR_UDS_STEP  # align to a session-request frame
    sends = self._step(cc, stock_radar_alive=True, fsc_settled=True)
    assert SESSION_PROG_DAT in self._uds(sends)
    # and settles back to silenced once quiet again
    sends = self._step(cc, stock_radar_alive=False, fsc_settled=True)
    assert SESSION_PROG_DAT not in self._uds(sends)


def _lkas_request(dat):
  cp = CANParser("mazda_2017", [("CAM_LKAS", float("nan"))], 0)
  cp.update([(0, [(0x243, dat, 0)])])
  return int(cp.vl["CAM_LKAS"]["LKAS_REQUEST"])


class TestCamLkasTorqueGate:
  """Stale CAM_LKAS must drop commanded torque on the wire, not only the liveness flag."""

  @pytest.fixture
  def cc_stock_long(self):
    CP = CarInterface.get_params(CAR.MAZDA_CX5_2022, {0: {}, 1: {}, 2: {}}, [], alpha_long=False,
                                 is_release=False, docs=False)
    CP_SP = CarInterface.get_params_sp(CP, CAR.MAZDA_CX5_2022, {0: {}, 1: {}, 2: {}}, [], False, False, False)
    return CarController({Bus.pt: "mazda_2017"}, CP, CP_SP)

  def test_stale_cam_lkas_sends_zero_torque(self, cc_stock_long):
    CC = structs.CarControl()
    CC.latActive = True
    CC.actuators.torque = 0.4
    CC = CC.as_reader()
    CC_SP = structs.CarControlSP()
    CS = SimpleNamespace(
      out=SimpleNamespace(vEgoRaw=12.0, steeringTorque=0, brakePressed=False),
      cam_lkas_live=True,
      cam_lkas={"ERR_BIT_1": 0, "ERR_BIT_2": 0, "LINE_NOT_VISIBLE": 0, "BIT_1": 1},
      cam_laneinfo={"LINE_VISIBLE": 0, "LINE_NOT_VISIBLE": 1, "LANE_LINES": 1,
                    "BIT1": 0, "BIT2": 0, "BIT3": 0, "NO_ERR_BIT": 0, "ERR_BIT": 0, "TJA": 0, "TJA_TRANSITION": 0, "S1": 0, "S1_HBEAM": 0},
      crz_btns_counter=0,
      cancel_button=0,
      tja_button=0,
      accel_button=0,
      decel_button=0,
      lkas_allowed_speed=True,
    )

    now_ns = 0
    saw_torque = False
    for _ in range(20):
      _, sends = cc_stock_long.update(CC, CC_SP, CS, now_ns)
      now_ns += int(DT_CTRL * 1e9)
      dat = next(d for a, d, _b in sends if a == 0x243)
      if _lkas_request(dat) != 0:
        saw_torque = True
        break
    assert saw_torque

    CS.cam_lkas_live = False
    _, sends = cc_stock_long.update(CC, CC_SP, CS, now_ns)
    dat = next(d for a, d, _b in sends if a == 0x243)
    assert _lkas_request(dat) == 0


class TestTjaIcbmSuppressScoping:
  """ICBM must ignore the physical TJA bit unless TJA_MADS is set."""

  @staticmethod
  def _cc(candidate, *, car_fw=None):
    CP = CarInterface.get_params(candidate, {0: {}, 1: {}, 2: {}}, car_fw or [], alpha_long=False,
                                 is_release=False, docs=False)
    CP_SP = CarInterface.get_params_sp(CP, candidate, {0: {}, 1: {}, 2: {}}, car_fw or [], False, False, False)
    return CarController({Bus.pt: "mazda_2017"}, CP, CP_SP), CP

  @staticmethod
  def _cs(*, tja_button=0, cruise_available=False, mrcc_armed_raw=None, crz_btns_counter=0,
          cancel_button=0, resume_button=0, accel_button=0, decel_button=0, mrcc_button=0):
    if mrcc_armed_raw is None:
      mrcc_armed_raw = cruise_available
    return SimpleNamespace(
      out=SimpleNamespace(vEgoRaw=12.0, steeringTorque=0, brakePressed=False,
                          cruiseState=SimpleNamespace(available=cruise_available)),
      cruise_available=cruise_available,
      mrcc_armed_raw=mrcc_armed_raw,
      cam_lkas_live=True,
      cam_lkas={"ERR_BIT_1": 0, "ERR_BIT_2": 0, "LINE_NOT_VISIBLE": 0, "BIT_1": 1},
      cam_laneinfo={"LINE_VISIBLE": 0, "LINE_NOT_VISIBLE": 1, "LANE_LINES": 1,
                    "BIT1": 0, "BIT2": 0, "BIT3": 0, "NO_ERR_BIT": 0, "ERR_BIT": 0, "TJA": 0, "TJA_TRANSITION": 0, "S1": 0, "S1_HBEAM": 0},
      crz_btns_counter=crz_btns_counter,
      cancel_button=cancel_button,
      resume_button=resume_button,
      tja_button=tja_button,
      accel_button=accel_button,
      decel_button=decel_button,
      mrcc_button=mrcc_button,
      lkas_allowed_speed=True,
    )

  @staticmethod
  def _crz_btns_present(sends):
    return any(addr == 0x09d for addr, _dat, _bus in sends)

  def test_tja_mads_suppresses_icbm_while_tja_held(self):
    cc, CP = self._cc(CAR.MAZDA_CX5_2022)
    assert CP.safetyConfigs[0].safetyParam & MazdaSafetyFlags.TJA_MADS
    CC = structs.CarControl()
    CC.latActive = False
    CC = CC.as_reader()
    CC_SP = structs.CarControlSP()
    CC_SP.intelligentCruiseButtonManagement.sendButton = (
      structs.IntelligentCruiseButtonManagement.SendButtonState.increase
    )
    cc.last_button_frame = -10_000
    _, sends = cc.update(CC, CC_SP, self._cs(tja_button=1), 0)
    assert not self._crz_btns_present(sends)

  def test_non_tja_does_not_suppress_icbm_for_tja_bit(self):
    from opendbc.car.mazda.values import STEER_TO_ZERO_EPS_FW

    swapped = sorted(STEER_TO_ZERO_EPS_FW)[0]
    fw = structs.CarParams.CarFw()
    fw.ecu = structs.CarParams.Ecu.eps
    fw.address = 0x730
    fw.subAddress = 0
    fw.fwVersion = swapped

    for candidate, car_fw in (
      (CAR.MAZDA_CX9_2021, []),
      (CAR.MAZDA_CX5, [fw]),
    ):
      cc, CP = self._cc(candidate, car_fw=car_fw)
      assert not (CP.safetyConfigs[0].safetyParam & MazdaSafetyFlags.TJA_MADS)
      CC = structs.CarControl()
      CC.latActive = False
      CC = CC.as_reader()
      CC_SP = structs.CarControlSP()
      CC_SP.intelligentCruiseButtonManagement.sendButton = (
        structs.IntelligentCruiseButtonManagement.SendButtonState.increase
      )
      cc.last_button_frame = -10_000
      _, sends = cc.update(CC, CC_SP, self._cs(tja_button=1), 0)
      assert self._crz_btns_present(sends), candidate


class TestTjaMrccSideEffect:
  @staticmethod
  def _cc(candidate=CAR.MAZDA_CX5_2022):
    CP = CarInterface.get_params(candidate, {0: {}, 1: {}, 2: {}}, [], alpha_long=False,
                                 is_release=False, docs=False)
    CP_SP = CarInterface.get_params_sp(CP, candidate, {0: {}, 1: {}, 2: {}}, [], False, False, False)
    return CarController({Bus.pt: "mazda_2017"}, CP, CP_SP)

  @staticmethod
  def _controls():
    CC = structs.CarControl()
    return CC.as_reader(), structs.CarControlSP()

  @staticmethod
  def _button_payloads(sends):
    return [dat for addr, dat, bus in sends if addr == 0x09d and bus == 0]

  def _step(self, cc, CC, CC_SP, *, tja, armed, raw_armed=None, counter=None,
            advance_nanos=None, **cs_kw):
    assert TJA_MRCC_MAX_TX_FRAMES == 3
    if advance_nanos is None:
      advance_nanos = round(DT_CTRL * 1e9)
    if counter is None:
      counter = (getattr(cc, "_test_crz_btns_counter", -1) + 1) % 16
    cc._test_crz_btns_counter = counter
    cc._test_now_nanos = getattr(cc, "_test_now_nanos", 0) + advance_nanos
    CS = TestTjaIcbmSuppressScoping._cs(tja_button=tja, cruise_available=armed,
                                        mrcc_armed_raw=raw_armed, crz_btns_counter=counter,
                                        **cs_kw)
    sends = cc.update(CC, CC_SP, CS, cc._test_now_nanos)[1]
    assert 0 <= cc.tja_mrcc_tx_frames <= TJA_MRCC_MAX_TX_FRAMES
    return sends

  @staticmethod
  def _payload_ctrs(payloads):
    cp = CANParser("mazda_2017", [("CRZ_BTNS", 0)], 0)
    counters = []
    for dat in payloads:
      cp.update([(0, [(0x09d, dat, 0)])])
      counters.append(int(cp.vl["CRZ_BTNS"]["CTR"]))
    return counters

  def _mrcc_off_payloads(self, sends):
    payloads = []
    cp = CANParser("mazda_2017", [("CRZ_BTNS", 0)], 0)
    for dat in self._button_payloads(sends):
      cp.update([(0, [(0x09d, dat, 0)])])
      if int(cp.vl["CRZ_BTNS"]["BIT1"]) == 0:
        payloads.append(dat)
    return payloads

  def _step_to_first_tx_deadline(self, cc, CC, CC_SP, *, tja=0, armed=True,
                                 raw_armed=True, counter=None, **cs_kw):
    """Advance five real-cadence controller updates from the latest TJA release."""
    step_nanos = round(DT_CTRL * 1e9)
    assert TJA_MRCC_FIRST_TX_DELAY_NANOS == 5 * step_nanos
    if counter is None:
      counter = (getattr(cc, "_test_crz_btns_counter", -1) + 1) % 16

    for _ in range(4):
      assert not self._mrcc_off_payloads(
        self._step(cc, CC, CC_SP, tja=tja, armed=armed, raw_armed=raw_armed,
                   counter=counter, advance_nanos=step_nanos, **cs_kw))
    return self._step(cc, CC, CC_SP, tja=tja, armed=armed, raw_armed=raw_armed,
                      counter=counter, advance_nanos=step_nanos, **cs_kw)

  @staticmethod
  def _assert_episode_budget(payloads):
    assert TJA_MRCC_MAX_TX_FRAMES == 3
    assert len(payloads) <= TJA_MRCC_MAX_TX_FRAMES

  def test_tja_caused_mrcc_arm_is_undone_after_release(self):
    cc = self._cc()
    CC, CC_SP = self._controls()

    self._step(cc, CC, CC_SP, tja=0, armed=False)
    self._step(cc, CC, CC_SP, tja=1, armed=False)
    self._step(cc, CC, CC_SP, tja=1, armed=True)

    # Release starts the delayed first-TX deadline.
    assert not self._button_payloads(self._step(cc, CC, CC_SP, tja=0, armed=True))
    payloads = self._button_payloads(
      self._step_to_first_tx_deadline(cc, CC, CC_SP, tja=0, armed=True))
    assert len(payloads) == 1

    cp = CANParser("mazda_2017", [("CRZ_BTNS", 0)], 0)
    cp.update([(0, [(0x09d, payloads[0], 0)])])
    assert cp.vl["CRZ_BTNS"]["BIT1"] == 0
    assert cp.vl["CRZ_BTNS"]["BIT1_INV"] == 1
    assert cp.vl["CRZ_BTNS"]["TJA_BUTTON"] == 0

    for _ in range(TJA_MRCC_RAW_OFF_CONFIRM_FRAMES):
      self._step(cc, CC, CC_SP, tja=0, armed=False)
    assert not cc.tja_mrcc_unarm_pending

    # Once raw feedback confirms off, the bounded hold must not continue.
    for _ in range(50):
      assert not self._button_payloads(
        self._step(cc, CC, CC_SP, tja=0, armed=False, raw_armed=False))

  def test_first_tx_real_controller_cadence(self):
    cc = self._cc()
    CC, CC_SP = self._controls()
    step_nanos = round(DT_CTRL * 1e9)

    self._step(cc, CC, CC_SP, tja=0, armed=False, raw_armed=False, counter=13)
    self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True, counter=14)
    self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=14)
    release_nanos = cc._test_now_nanos

    for elapsed, counter in zip((10, 20, 30, 40), (14, 15, 15, 15), strict=True):
      assert not self._mrcc_off_payloads(
        self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True,
                   counter=counter, advance_nanos=step_nanos))
      assert cc._test_now_nanos - release_nanos == elapsed * 1_000_000

    payloads = self._mrcc_off_payloads(
      self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True,
                 counter=15, advance_nanos=step_nanos))
    assert cc._test_now_nanos - release_nanos == 50_000_000
    assert self._payload_ctrs(payloads) == [0]

  def test_release_between_controller_cycles_dispatches_within_window(self):
    cc = self._cc()
    CC, CC_SP = self._controls()
    step_nanos = round(DT_CTRL * 1e9)

    self._step(cc, CC, CC_SP, tja=0, armed=False, raw_armed=False,
               counter=0, advance_nanos=0)
    self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True,
               counter=1, advance_nanos=0)

    physical_release_nanos = cc._test_now_nanos + 5_000_000
    self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True,
               counter=2, advance_nanos=step_nanos)

    for _ in range(4):
      assert not self._mrcc_off_payloads(
        self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True,
                   counter=2, advance_nanos=step_nanos))

    payloads = self._mrcc_off_payloads(
      self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True,
                 counter=2, advance_nanos=step_nanos))
    physical_latency = cc._test_now_nanos - physical_release_nanos
    assert self._payload_ctrs(payloads) == [3]
    assert 50_000_000 <= physical_latency <= 60_000_000

  def test_raw_off_wins_at_exact_first_tx_deadline(self):
    cc = self._cc()
    CC, CC_SP = self._controls()

    self._step(cc, CC, CC_SP, tja=0, armed=False, raw_armed=False, counter=0)
    self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True, counter=1)
    self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=2)
    for _ in range(4):
      assert not self._mrcc_off_payloads(
        self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=3))

    assert not self._mrcc_off_payloads(
      self._step(cc, CC, CC_SP, tja=0, armed=False, raw_armed=False, counter=3))
    assert not cc.tja_mrcc_unarm_pending
    assert cc.tja_mrcc_first_tx_not_before_nanos is None

  def test_physical_mrcc_wins_at_exact_first_tx_deadline(self):
    cc = self._cc()
    CC, CC_SP = self._controls()

    self._step(cc, CC, CC_SP, tja=0, armed=False, raw_armed=False, counter=0)
    self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True, counter=1)
    self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=2)
    for _ in range(4):
      assert not self._mrcc_off_payloads(
        self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=3))

    assert not self._mrcc_off_payloads(
      self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True,
                 counter=3, mrcc_button=1))
    assert not cc.tja_mrcc_unarm_pending
    assert cc.tja_mrcc_first_tx_not_before_nanos is None

  @pytest.mark.parametrize("button", (
    {"accel_button": 1},
    {"decel_button": 1},
    {"resume_button": 1},
    {"cancel_button": 1},
  ))
  def test_set_res_cancel_wins_at_exact_first_tx_deadline(self, button):
    cc = self._cc()
    CC, CC_SP = self._controls()

    self._step(cc, CC, CC_SP, tja=0, armed=False, raw_armed=False, counter=0)
    self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True, counter=1)
    self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=2)
    for _ in range(4):
      assert not self._mrcc_off_payloads(
        self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=3))

    assert not self._mrcc_off_payloads(
      self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True,
                 counter=3, **button))
    assert not cc.tja_mrcc_unarm_pending
    assert cc.tja_mrcc_first_tx_not_before_nanos is None

  def test_tja_repress_wins_at_exact_first_tx_deadline(self):
    cc = self._cc()
    CC, CC_SP = self._controls()

    self._step(cc, CC, CC_SP, tja=0, armed=False, raw_armed=False, counter=0)
    self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True, counter=1)
    self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=2)
    for _ in range(4):
      assert not self._mrcc_off_payloads(
        self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=3))

    assert not self._mrcc_off_payloads(
      self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True, counter=3))
    assert cc.tja_mrcc_unarm_pending
    assert cc.tja_mrcc_first_tx_not_before_nanos is None

    for _ in range(10):
      assert not self._mrcc_off_payloads(
        self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True, counter=4))

  def test_tja_preserves_mrcc_that_was_already_armed(self):
    cc = self._cc()
    CC, CC_SP = self._controls()
    self._step(cc, CC, CC_SP, tja=0, armed=True)
    self._step(cc, CC, CC_SP, tja=1, armed=True)

    payloads = []
    for _ in range(3):
      payloads.extend(self._button_payloads(self._step(cc, CC, CC_SP, tja=0, armed=True)))
    assert not payloads
    assert not cc.tja_mrcc_unarm_pending

  def test_repeated_tja_under_brake_uses_confirmed_raw_state(self):
    """Route 56: cruise_available stays True through a held brake after the first
    automatic tap, but PEDALS.ACC_OFF is already zero. A later TJA press must clean up
    its new MRCC arm instead of treating that cached True as pre-existing MRCC."""
    cc = self._cc()
    CC, CC_SP = self._controls()

    # First TJA press starts with MRCC genuinely off, then TJA arms it.
    self._step(cc, CC, CC_SP, tja=0, armed=False, raw_armed=False)
    self._step(cc, CC, CC_SP, tja=1, armed=False, raw_armed=False)
    self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True)
    self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True)

    # The automatic tap lands in raw PEDALS, while filtered availability remains
    # intentionally held True for the whole brake press.
    for _ in range(TJA_MRCC_RAW_OFF_CONFIRM_FRAMES):
      self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=False)
    assert not cc.tja_mrcc_unarm_pending

    # A second TJA press during the same brake hold must be recognized as starting
    # from MRCC-off and receive another automatic off tap after release.
    self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=False)
    self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True)
    self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True)
    payloads = self._button_payloads(
      self._step_to_first_tx_deadline(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True))
    assert len(payloads) == 1

  def test_brief_raw_dropout_does_not_unarm_prearmed_mrcc(self):
    cc = self._cc()
    CC, CC_SP = self._controls()

    self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True)
    # Include the TJA edge itself in the below-threshold dropout length.
    for _ in range(TJA_MRCC_RAW_OFF_CONFIRM_FRAMES - 2):
      self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=False)
    self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=False)

    payloads = []
    for _ in range(3):
      payloads.extend(self._button_payloads(
        self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True)))
    assert not payloads
    assert not cc.tja_mrcc_unarm_pending

  def test_non_tja_platform_never_sends_mrcc_off_tap(self):
    cc = self._cc(CAR.MAZDA_CX9_2021)
    CC, CC_SP = self._controls()
    self._step(cc, CC, CC_SP, tja=1, armed=False)
    payloads = []
    for _ in range(3):
      payloads.extend(self._button_payloads(self._step(cc, CC, CC_SP, tja=0, armed=True)))
    assert not payloads

  def test_same_frame_tja_arm_uses_pre_press_mrcc_state(self):
    """The TJA frame and PEDALS arm can reach CarState together. The previous
    stable sample, not the already-armed edge sample, owns the cleanup decision."""
    cc = self._cc()
    CC, CC_SP = self._controls()

    self._step(cc, CC, CC_SP, tja=0, armed=False, raw_armed=False)
    self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True)
    self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True)
    assert len(self._button_payloads(
      self._step_to_first_tx_deadline(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True))) == 1

  def test_long_hold_does_not_consume_cleanup(self):
    cc = self._cc()
    CC, CC_SP = self._controls()

    self._step(cc, CC, CC_SP, tja=0, armed=False, raw_armed=False)
    self._step(cc, CC, CC_SP, tja=1, armed=False, raw_armed=False)
    for _ in range(TJA_MRCC_RELEASE_WAIT_FRAMES + 100):
      assert not self._button_payloads(self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True))

    assert not self._button_payloads(self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True))
    assert len(self._button_payloads(
      self._step_to_first_tx_deadline(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True))) == 1

  def test_route_5d_double_tap_before_first_cleanup_preserves_transaction(self):
    """Route 5d at 123.101/123.263: the second press landed before the first
    cleanup frame. Wait for the final release, then start one continuous press."""
    cc = self._cc()
    CC, CC_SP = self._controls()

    self._step(cc, CC, CC_SP, tja=0, armed=False, raw_armed=False, counter=0)
    self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True, counter=1)
    self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=2)
    self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True, counter=2)
    for _ in range(5):
      assert not self._mrcc_off_payloads(
        self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True, counter=3))
    assert cc.tja_mrcc_unarm_pending
    assert cc.tja_mrcc_tx_frames == 0

    assert not self._mrcc_off_payloads(
      self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=4))
    payloads = self._mrcc_off_payloads(
      self._step_to_first_tx_deadline(
        cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=5))
    for counter in (6, 7, 8):
      payloads.extend(self._mrcc_off_payloads(
        self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=counter)))
    assert len(payloads) == TJA_MRCC_MAX_TX_FRAMES
    self._assert_episode_budget(payloads)
    assert self._payload_ctrs(payloads) == [6, 7, 8]
    assert cc.tja_mrcc_tx_frames == TJA_MRCC_MAX_TX_FRAMES
    assert not cc.tja_mrcc_unarm_pending

  def test_route_5d_interrupted_after_one_frame_delayed_raw_off_sends_no_replacement(self):
    """Route 5d: delayed acknowledgement during TJA2 ends ownership before release."""
    cc = self._cc()
    CC, CC_SP = self._controls()

    self._step(cc, CC, CC_SP, tja=0, armed=False, raw_armed=False, counter=0)
    self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True, counter=1)
    self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=2)
    payloads = self._mrcc_off_payloads(
      self._step_to_first_tx_deadline(
        cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=3))
    assert self._payload_ctrs(payloads) == [4]
    assert cc.tja_mrcc_tx_frames == 1
    assert cc.tja_mrcc_press_frames == 1

    self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True, counter=3)
    assert cc.tja_mrcc_unarm_pending
    assert cc.tja_mrcc_tx_frames == 1
    assert cc.tja_mrcc_press_frames == 0
    # Keep this below the existing five-frame confirmed-off threshold so the
    # replacement-release path itself must observe immediate raw-off.
    for _ in range(2):
      assert not self._mrcc_off_payloads(
        self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=False, counter=4))
    assert cc.tja_mrcc_unarm_pending
    assert cc.tja_mrcc_tx_frames == 1

    payloads.extend(self._mrcc_off_payloads(
      self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=False, counter=5)))
    assert self._payload_ctrs(payloads) == [4]
    self._assert_episode_budget(payloads)
    assert not cc.tja_mrcc_unarm_pending
    assert cc.tja_mrcc_tx_frames == 1

  def test_route_5d_interrupted_after_one_frame_uses_only_remaining_budget(self):
    cc = self._cc()
    CC, CC_SP = self._controls()

    self._step(cc, CC, CC_SP, tja=0, armed=False, raw_armed=False, counter=0)
    self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True, counter=1)
    self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=2)
    payloads = self._mrcc_off_payloads(
      self._step_to_first_tx_deadline(
        cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=3))
    assert self._payload_ctrs(payloads) == [4]

    self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True, counter=3)
    assert cc.tja_mrcc_unarm_pending
    assert cc.tja_mrcc_tx_frames == 1
    assert cc.tja_mrcc_press_frames == 0
    for _ in range(3):
      assert not self._mrcc_off_payloads(
        self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True, counter=4))

    assert not self._mrcc_off_payloads(
      self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=5))
    for counter in (6, 7, 8, 9):
      payloads.extend(self._mrcc_off_payloads(
        self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=counter)))
    assert self._payload_ctrs(payloads) == [4, 7, 8]
    self._assert_episode_budget(payloads)
    assert cc.tja_mrcc_tx_frames == TJA_MRCC_MAX_TX_FRAMES
    assert not cc.tja_mrcc_unarm_pending

  def test_route_5d_interrupted_after_two_frames_has_one_remaining(self):
    cc = self._cc()
    CC, CC_SP = self._controls()

    self._step(cc, CC, CC_SP, tja=0, armed=False, raw_armed=False, counter=0)
    self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True, counter=1)
    self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=2)
    payloads = self._mrcc_off_payloads(
      self._step_to_first_tx_deadline(
        cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=3))
    payloads.extend(self._mrcc_off_payloads(
      self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=4)))
    assert self._payload_ctrs(payloads) == [4, 5]
    assert cc.tja_mrcc_tx_frames == 2

    self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True, counter=4)
    assert cc.tja_mrcc_unarm_pending
    assert cc.tja_mrcc_press_frames == 0
    self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=6)
    for counter in (7, 8, 9):
      payloads.extend(self._mrcc_off_payloads(
        self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=counter)))
    assert self._payload_ctrs(payloads) == [4, 5, 8]
    self._assert_episode_budget(payloads)
    assert cc.tja_mrcc_tx_frames == TJA_MRCC_MAX_TX_FRAMES
    assert not cc.tja_mrcc_unarm_pending

  def test_manual_mrcc_off_before_tja_release_is_never_toggled_on(self):
    cc = self._cc()
    CC, CC_SP = self._controls()

    self._step(cc, CC, CC_SP, tja=0, armed=False, raw_armed=False)
    self._step(cc, CC, CC_SP, tja=1, armed=False, raw_armed=False)
    self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True)
    # The driver turns MRCC off while TJA remains held. Filtered availability can
    # still be cached true under braking, so the raw bit must suppress the toggle.
    self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=False)
    self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=False)
    for _ in range(20):
      assert not self._button_payloads(self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=False))

  def test_delayed_first_tx_uses_latest_stable_counter_then_times_out(self):
    cc = self._cc()
    CC, CC_SP = self._controls()

    self._step(cc, CC, CC_SP, tja=0, armed=False, raw_armed=False, counter=3)
    self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True, counter=4)
    self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=5)
    payloads = self._mrcc_off_payloads(
      self._step_to_first_tx_deadline(
        cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=5))
    assert self._payload_ctrs(payloads) == [6]

    for _ in range(TJA_MRCC_RELEASE_WAIT_FRAMES + 1):
      assert not self._button_payloads(
        self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=5))
    assert not cc.tja_mrcc_unarm_pending

  @pytest.mark.parametrize("which", ("cancel", "resume"))
  def test_op_pre_start_fresh_counter_before_deadline_still_waits(self, which):
    cc = self._cc()
    CC, CC_SP = self._controls()

    self._step(cc, CC, CC_SP, tja=0, armed=False, raw_armed=False, counter=3)
    self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True, counter=4)
    self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=5)
    assert cc.tja_mrcc_tx_frames == 0

    CC_btn = structs.CarControl()
    setattr(CC_btn.cruiseControl, which, True)
    assert not self._mrcc_off_payloads(
      self._step(cc, CC_btn.as_reader(), CC_SP, tja=0, armed=True, raw_armed=True, counter=6))

    # Clearing the OP command on its retained counter is not fresh enough.
    assert not self._mrcc_off_payloads(
      self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=6))
    # A fresh counter before 50 ms satisfies only the post-OP freshness requirement.
    assert not self._mrcc_off_payloads(
      self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=7))
    assert not self._mrcc_off_payloads(
      self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=7))
    payloads = self._mrcc_off_payloads(
      self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=7))
    assert self._payload_ctrs(payloads) == [8]

    for counter in (8, 9, 10):
      payloads.extend(self._mrcc_off_payloads(
        self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=counter)))
    assert self._payload_ctrs(payloads) == [8, 9, 10]
    self._assert_episode_budget(payloads)
    assert cc.tja_mrcc_tx_frames == TJA_MRCC_MAX_TX_FRAMES

  @pytest.mark.parametrize("which", ("cancel", "resume"))
  def test_op_pre_start_deadline_expires_before_command_clears(self, which):
    cc = self._cc()
    CC, CC_SP = self._controls()

    self._step(cc, CC, CC_SP, tja=0, armed=False, raw_armed=False, counter=3)
    self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True, counter=4)
    self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=5)

    CC_btn = structs.CarControl()
    setattr(CC_btn.cruiseControl, which, True)
    for _ in range(5):
      assert not self._mrcc_off_payloads(
        self._step(cc, CC_btn.as_reader(), CC_SP, tja=0, armed=True, raw_armed=True, counter=6))

    # Deadline has passed, but clearing on the retained counter still cannot transmit.
    assert not self._mrcc_off_payloads(
      self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=6))
    payloads = self._mrcc_off_payloads(
      self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=7))
    assert self._payload_ctrs(payloads) == [8]

  @pytest.mark.parametrize("which", ("cancel", "resume"))
  def test_op_pre_start_fresh_counter_wait_times_out_and_unsuppresses_icbm(self, which):
    cc = self._cc()
    CC, CC_SP = self._controls()

    self._step(cc, CC, CC_SP, tja=0, armed=False, raw_armed=False, counter=3)
    self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True, counter=4)
    self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=5)

    CC_btn = structs.CarControl()
    setattr(CC_btn.cruiseControl, which, True)
    assert not self._mrcc_off_payloads(
      self._step(cc, CC_btn.as_reader(), CC_SP, tja=0, armed=True, raw_armed=True, counter=6))
    assert not self._mrcc_off_payloads(
      self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=6))

    for _ in range(TJA_MRCC_RELEASE_WAIT_FRAMES + 1):
      assert not self._mrcc_off_payloads(
        self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=6))
    assert not cc.tja_mrcc_unarm_pending

    CC_SP.intelligentCruiseButtonManagement.sendButton = (
      structs.IntelligentCruiseButtonManagement.SendButtonState.increase
    )
    cc.last_button_frame = -10_000
    payloads = self._button_payloads(
      self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=6))
    assert len(payloads) == 1
    cp = CANParser("mazda_2017", [("CRZ_BTNS", 0)], 0)
    cp.update([(0, [(0x09d, payloads[0], 0)])])
    assert int(cp.vl["CRZ_BTNS"]["BIT1"]) == 1
    assert int(cp.vl["CRZ_BTNS"]["SET_P"]) == 1

  def test_op_cancel_and_resume_abort_after_press_started(self):
    for which in ("cancel", "resume"):
      cc = self._cc()
      CC, CC_SP = self._controls()
      self._step(cc, CC, CC_SP, tja=0, armed=False, raw_armed=False, counter=0)
      self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True, counter=1)
      self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=2)
      assert len(self._mrcc_off_payloads(
        self._step_to_first_tx_deadline(
          cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=3))) == 1
      assert cc.tja_mrcc_tx_frames == 1

      CC_btn = structs.CarControl()
      setattr(CC_btn.cruiseControl, which, True)
      assert not self._mrcc_off_payloads(
        self._step(cc, CC_btn.as_reader(), CC_SP, tja=0, armed=True, raw_armed=True, counter=4))
      assert not cc.tja_mrcc_unarm_pending

      leftover = []
      for counter in range(5, 16):
        leftover.extend(self._mrcc_off_payloads(
          self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=counter)))
      assert leftover == [], which

  @pytest.mark.parametrize("which", ("cancel", "resume"))
  def test_op_cancel_and_resume_abort_during_tja_hold_after_press_started(self, which):
    cc = self._cc()
    CC, CC_SP = self._controls()
    self._step(cc, CC, CC_SP, tja=0, armed=False, raw_armed=False, counter=0)
    self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True, counter=1)
    self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=2)
    assert len(self._mrcc_off_payloads(
      self._step_to_first_tx_deadline(
        cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=3))) == 1

    # TJA2 interrupts the press and clears the release anchor.
    self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True, counter=3)
    assert cc.tja_mrcc_release_counter is None
    assert cc.tja_mrcc_unarm_pending

    CC_btn = structs.CarControl()
    setattr(CC_btn.cruiseControl, which, True)
    assert not self._mrcc_off_payloads(
      self._step(cc, CC_btn.as_reader(), CC_SP, tja=1, armed=True, raw_armed=True, counter=4))
    assert not cc.tja_mrcc_unarm_pending
    assert cc.tja_mrcc_press_frames == 0
    assert cc.tja_mrcc_tx_frames == 1

    leftover = []
    for counter in range(5, 13):
      leftover.extend(self._mrcc_off_payloads(
        self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=counter)))
    assert leftover == [], which

  def test_ignored_first_frame_keeps_press_on_next_consecutive_counter(self):
    """If the first asserted frame is ignored, hold the same press on the next
    consecutive counter, then stop immediately when raw PEDALS confirms off."""
    cc = self._cc()
    CC, CC_SP = self._controls()

    self._step(cc, CC, CC_SP, tja=0, armed=False, raw_armed=False, counter=7)
    self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True, counter=8)
    self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True, counter=9)
    self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=10)

    payloads = self._button_payloads(
      self._step_to_first_tx_deadline(
        cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=11))
    payloads.extend(self._button_payloads(
      self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=12)))
    payloads.extend(self._button_payloads(
      self._step(cc, CC, CC_SP, tja=0, armed=False, raw_armed=False, counter=13)))

    assert len(payloads) == 2
    assert self._payload_ctrs(payloads) == [12, 13]
    assert not cc.tja_mrcc_unarm_pending

  def test_cleanup_is_hard_capped_at_physical_button_length(self):
    cc = self._cc()
    CC, CC_SP = self._controls()

    self._step(cc, CC, CC_SP, tja=0, armed=False, raw_armed=False, counter=0)
    self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True, counter=1)
    self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=2)

    payloads = self._button_payloads(
      self._step_to_first_tx_deadline(
        cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=3))
    for counter in range(4, 16):
      payloads.extend(self._button_payloads(
        self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=counter)))

    assert len(payloads) == TJA_MRCC_MAX_TX_FRAMES
    assert self._payload_ctrs(payloads) == [4, 5, 6]
    assert not cc.tja_mrcc_unarm_pending

  def test_hard_cap_sends_nothing_after_three_frames_despite_delayed_feedback(self):
    """Route 61: after the three-frame press is spent, leftover armed PEDALS
    must not start another synthetic press."""
    cc = self._cc()
    CC, CC_SP = self._controls()

    self._step(cc, CC, CC_SP, tja=0, armed=False, raw_armed=False, counter=0)
    self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True, counter=1)
    self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=2)
    assert len(self._button_payloads(
      self._step_to_first_tx_deadline(
        cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=3))) == 1
    for counter in range(4, 6):
      assert len(self._button_payloads(
        self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=counter))) == 1
    assert cc.tja_mrcc_tx_frames == TJA_MRCC_MAX_TX_FRAMES
    assert not cc.tja_mrcc_unarm_pending

    leftover = []
    for counter in range(6, 16):
      leftover.extend(self._button_payloads(
        self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=counter)))
    assert leftover == []

  def test_driver_set_aborts_in_flight_cleanup(self):
    cc = self._cc()
    CC, CC_SP = self._controls()

    self._step(cc, CC, CC_SP, tja=0, armed=False, raw_armed=False, counter=0)
    self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True, counter=1)
    self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=2)
    assert len(self._button_payloads(
      self._step_to_first_tx_deadline(
        cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=3))) == 1
    assert not self._button_payloads(
      self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=4, accel_button=1))
    assert not cc.tja_mrcc_unarm_pending
    leftover = []
    for counter in range(5, 16):
      leftover.extend(self._button_payloads(
        self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=counter)))
    assert leftover == []

  def test_counter_jump_after_first_frame_aborts_without_continuation(self):
    cc = self._cc()
    CC, CC_SP = self._controls()

    self._step(cc, CC, CC_SP, tja=0, armed=False, raw_armed=False, counter=0)
    self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True, counter=1)
    self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=2)
    assert len(self._button_payloads(
      self._step_to_first_tx_deadline(
        cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=3))) == 1
    assert cc.tja_mrcc_tx_frames == 1
    assert not self._button_payloads(
      self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=5))
    assert not cc.tja_mrcc_unarm_pending
    leftover = []
    for counter in (6, 7, 8):
      leftover.extend(self._button_payloads(
        self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=counter)))
    assert leftover == []

    # The partial spent budget is not reset or resurrected by a later TJA while
    # the unresolved MRCC state remains armed.
    self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True, counter=9)
    for counter in (10, 11, 12):
      leftover.extend(self._mrcc_off_payloads(
        self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=counter)))
    assert leftover == []
    assert cc.tja_mrcc_tx_frames == 1

  def test_counter_advancement_during_delay_uses_latest_counter(self):
    cc = self._cc()
    CC, CC_SP = self._controls()

    self._step(cc, CC, CC_SP, tja=0, armed=False, raw_armed=False, counter=0)
    self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True, counter=1)
    self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=2)
    for counter in (4, 5, 6, 7):
      assert not self._mrcc_off_payloads(
        self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=counter))

    payloads = self._mrcc_off_payloads(
      self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=8))
    assert self._payload_ctrs(payloads) == [9]
    for counter in (9, 10):
      payloads.extend(self._mrcc_off_payloads(
        self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=counter)))
    assert self._payload_ctrs(payloads) == [9, 10, 11]
    assert cc.tja_mrcc_tx_frames == TJA_MRCC_MAX_TX_FRAMES
    assert not cc.tja_mrcc_unarm_pending

  def test_repeated_counter_jumps_cannot_suppress_icbm_indefinitely(self):
    cc = self._cc()
    CC, CC_SP = self._controls()

    self._step(cc, CC, CC_SP, tja=0, armed=False, raw_armed=False, counter=0)
    self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True, counter=1)
    self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=2)
    assert cc.tja_mrcc_unarm_pending
    assert cc.tja_mrcc_tx_frames == 0

    counter = 4
    for _ in range(4):
      assert not self._mrcc_off_payloads(
        self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=counter))
      counter = (counter + 2) % 16

    payloads = self._mrcc_off_payloads(
      self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=counter))
    assert len(payloads) == 1

    counter = (counter + 2) % 16
    assert not self._mrcc_off_payloads(
      self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=counter))
    assert not cc.tja_mrcc_unarm_pending

    CC_SP.intelligentCruiseButtonManagement.sendButton = (
      structs.IntelligentCruiseButtonManagement.SendButtonState.increase
    )
    cc.last_button_frame = -10_000
    payloads = self._button_payloads(
      self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=counter))
    assert len(payloads) == 1
    cp = CANParser("mazda_2017", [("CRZ_BTNS", 0)], 0)
    cp.update([(0, [(0x09d, payloads[0], 0)])])
    assert int(cp.vl["CRZ_BTNS"]["BIT1"]) == 1
    assert int(cp.vl["CRZ_BTNS"]["SET_P"]) == 1

  def test_route_61_tja_from_off_uses_consecutive_not_spaced_counters(self):
    """Route 61 at 98.760: TJA from MRCC-off, Mazda stayed armed. Old spaced
    retries used delta 2. The held press must occupy consecutive CS counters."""
    cc = self._cc()
    CC, CC_SP = self._controls()

    self._step(cc, CC, CC_SP, tja=0, armed=False, raw_armed=False, counter=8)
    self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True, counter=9)
    self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=10)
    payloads = self._button_payloads(
      self._step_to_first_tx_deadline(
        cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=11))
    for counter in (12, 13, 14, 15, 0, 1):
      payloads.extend(self._button_payloads(
        self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=counter)))
    assert len(payloads) == TJA_MRCC_MAX_TX_FRAMES
    assert self._payload_ctrs(payloads) == [12, 13, 14]
    assert not cc.tja_mrcc_unarm_pending

  def test_physical_mrcc_during_hold_aborts_before_pedals_off(self):
    cc = self._cc()
    CC, CC_SP = self._controls()

    self._step(cc, CC, CC_SP, tja=0, armed=False, raw_armed=False, counter=0)
    self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True, counter=1)
    self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=2)
    assert len(self._button_payloads(
      self._step_to_first_tx_deadline(
        cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=3))) == 1
    assert not self._button_payloads(
      self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=4, mrcc_button=1))
    assert not cc.tja_mrcc_unarm_pending
    leftover = []
    for counter in range(5, 12):
      leftover.extend(self._button_payloads(
        self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=counter)))
    assert leftover == []

  @pytest.mark.parametrize("button", (
    {"accel_button": 1},
    {"decel_button": 1},
    {"resume_button": 1},
    {"cancel_button": 1},
    {"mrcc_button": 1},
  ))
  @pytest.mark.parametrize("phase, spent", (("initial", 0), ("after_one", 1), ("after_two", 2)))
  def test_physical_button_aborts_ownership_while_tja_held(self, button, phase, spent):
    cc = self._cc()
    CC, CC_SP = self._controls()
    self._step(cc, CC, CC_SP, tja=0, armed=False, raw_armed=False, counter=0)
    self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True, counter=1)

    if phase == "initial":
      physical_counter = 2
    else:
      self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=2)
      assert len(self._mrcc_off_payloads(
        self._step_to_first_tx_deadline(
          cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=3))) == 1
      for counter in range(4, 3 + spent):
        assert len(self._mrcc_off_payloads(
          self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=counter))) == 1
      # TJA2 interrupts the current press and clears the release anchor.
      interrupted_counter = 2 + spent
      self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True, counter=interrupted_counter)
      assert cc.tja_mrcc_release_counter is None
      physical_counter = interrupted_counter + 1

    assert cc.tja_mrcc_unarm_pending
    assert not self._mrcc_off_payloads(
      self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True,
                 counter=physical_counter, **button))
    assert not cc.tja_mrcc_unarm_pending
    assert cc.tja_mrcc_press_frames == 0
    assert cc.tja_mrcc_tx_frames == spent

    leftover = []
    counter = physical_counter + 1
    for _ in range(8):
      leftover.extend(self._mrcc_off_payloads(
        self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=counter % 16)))
      counter += 1
    assert leftover == [], (button, phase)

  def test_raw_off_after_first_frame_stops_before_second(self):
    cc = self._cc()
    CC, CC_SP = self._controls()

    self._step(cc, CC, CC_SP, tja=0, armed=False, raw_armed=False, counter=0)
    self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True, counter=1)
    self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=2)
    assert len(self._button_payloads(
      self._step_to_first_tx_deadline(
        cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=3))) == 1
    assert not self._button_payloads(
      self._step(cc, CC, CC_SP, tja=0, armed=False, raw_armed=False, counter=4))
    assert not cc.tja_mrcc_unarm_pending

  def test_manual_mrcc_off_during_press_never_sends_followup(self):
    cc = self._cc()
    CC, CC_SP = self._controls()

    self._step(cc, CC, CC_SP, tja=0, armed=False, raw_armed=False, counter=0)
    self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True, counter=1)
    self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=2)
    assert len(self._button_payloads(
      self._step_to_first_tx_deadline(
        cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=3))) == 1
    leftover = []
    for counter in range(4, 10):
      leftover.extend(self._button_payloads(
        self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=False, counter=counter)))
    assert leftover == []
    assert not cc.tja_mrcc_unarm_pending

  def test_counter_wrap_sends_consecutive_15_then_0(self):
    cc = self._cc()
    CC, CC_SP = self._controls()

    self._step(cc, CC, CC_SP, tja=0, armed=False, raw_armed=False, counter=13)
    self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True, counter=14)
    self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=15)
    payloads = self._button_payloads(
      self._step_to_first_tx_deadline(
        cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=0))
    for counter in (1, 2):
      payloads.extend(self._button_payloads(
        self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=counter)))
    assert len(payloads) == 3
    assert self._payload_ctrs(payloads) == [1, 2, 3]
    leftover = []
    for counter in range(3, 8):
      leftover.extend(self._button_payloads(
        self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=counter)))
    assert leftover == []

  def test_cleanup_output_counters_wrap_15_0_1(self):
    cc = self._cc()
    CC, CC_SP = self._controls()

    self._step(cc, CC, CC_SP, tja=0, armed=False, raw_armed=False, counter=12)
    self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True, counter=13)
    self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=13)
    payloads = self._mrcc_off_payloads(
      self._step_to_first_tx_deadline(
        cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=14))
    for counter in (15, 0, 1, 2):
      payloads.extend(self._mrcc_off_payloads(
        self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=counter)))
    assert self._payload_ctrs(payloads) == [15, 0, 1]
    assert cc.tja_mrcc_tx_frames == TJA_MRCC_MAX_TX_FRAMES
    assert not cc.tja_mrcc_unarm_pending

  def test_physical_res_and_cancel_abort_in_flight_cleanup(self):
    for kw in ({"resume_button": 1}, {"cancel_button": 1}, {"decel_button": 1}):
      cc = self._cc()
      CC, CC_SP = self._controls()
      self._step(cc, CC, CC_SP, tja=0, armed=False, raw_armed=False, counter=0)
      self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True, counter=1)
      self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=2)
      assert len(self._button_payloads(
        self._step_to_first_tx_deadline(
          cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=3))) == 1
      assert not self._button_payloads(
        self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=4, **kw))
      assert not cc.tja_mrcc_unarm_pending
      leftover = []
      for counter in range(5, 16):
        leftover.extend(self._button_payloads(
          self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=counter)))
      assert leftover == [], kw

  def test_failed_three_frame_press_does_not_leak_into_later_tja(self):
    """Budget exhaustion survives repeated TJA presses until confirmed raw-off."""
    cc = self._cc()
    CC, CC_SP = self._controls()

    self._step(cc, CC, CC_SP, tja=0, armed=False, raw_armed=False, counter=0)
    self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True, counter=1)
    self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=2)
    self._step_to_first_tx_deadline(
      cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=3)
    for counter in range(4, 6):
      self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=counter)
    assert cc.tja_mrcc_tx_frames == TJA_MRCC_MAX_TX_FRAMES
    assert not cc.tja_mrcc_unarm_pending

    # TJA2 after all three frames, then TJA3/TJA4: no replacement, no reset.
    leftover = []
    counter = 6
    for _ in range(3):
      self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True, counter=counter)
      counter += 1
      for _ in range(2):
        leftover.extend(self._mrcc_off_payloads(
          self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=counter)))
        counter += 1
    assert leftover == []
    assert not cc.tja_mrcc_unarm_pending
    assert cc.tja_mrcc_tx_frames == TJA_MRCC_MAX_TX_FRAMES

    # Confirmed raw-off permits a genuinely new ownership episode and only that
    # physical TJA rising edge resets the cumulative budget.
    for _ in range(TJA_MRCC_RAW_OFF_CONFIRM_FRAMES):
      self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=False, counter=counter)
    assert cc.tja_mrcc_tx_frames == TJA_MRCC_MAX_TX_FRAMES
    self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True, counter=counter)
    assert cc.tja_mrcc_unarm_pending
    assert cc.tja_mrcc_tx_frames == 0
    counter = (counter + 1) % 16
    self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=counter)
    counter = (counter + 1) % 16
    assert len(self._button_payloads(
      self._step_to_first_tx_deadline(
        cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=counter))) == 1
    assert cc.tja_mrcc_tx_frames == 1

  def test_brief_raw_dropout_does_not_reset_spent_budget(self):
    cc = self._cc()
    CC, CC_SP = self._controls()

    self._step(cc, CC, CC_SP, tja=0, armed=False, raw_armed=False, counter=0)
    self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True, counter=1)
    self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=2)
    payloads = self._mrcc_off_payloads(
      self._step_to_first_tx_deadline(
        cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=3))
    for counter in (4, 5):
      payloads.extend(self._mrcc_off_payloads(
        self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=counter)))
    self._assert_episode_budget(payloads)
    assert cc.tja_mrcc_tx_frames == TJA_MRCC_MAX_TX_FRAMES

    # Four raw-off samples including the TJA edge are below the five-frame
    # ownership-classification threshold.
    for counter in (6, 7, 8):
      self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=False, counter=counter)
    self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=False, counter=9)
    assert not cc.tja_mrcc_unarm_pending
    assert cc.tja_mrcc_tx_frames == TJA_MRCC_MAX_TX_FRAMES

    leftover = []
    for counter in (10, 11, 12, 13):
      leftover.extend(self._mrcc_off_payloads(
        self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=counter)))
    assert leftover == []
    assert cc.tja_mrcc_tx_frames == TJA_MRCC_MAX_TX_FRAMES

  def test_final_cleanup_frame_still_suppresses_icbm(self):
    cc = self._cc()
    CC, CC_SP = self._controls()

    self._step(cc, CC, CC_SP, tja=0, armed=False, raw_armed=False, counter=0)
    self._step(cc, CC, CC_SP, tja=1, armed=True, raw_armed=True, counter=1)
    self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=2)
    self._step_to_first_tx_deadline(
      cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=3)
    self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=4)

    CC_SP.intelligentCruiseButtonManagement.sendButton = (
      structs.IntelligentCruiseButtonManagement.SendButtonState.increase
    )
    cc.last_button_frame = -10_000
    payloads = self._button_payloads(
      self._step(cc, CC, CC_SP, tja=0, armed=True, raw_armed=True, counter=5))
    assert len(payloads) == 1

    cp = CANParser("mazda_2017", [("CRZ_BTNS", 0)], 0)
    cp.update([(0, [(0x09d, payloads[0], 0)])])
    assert cp.vl["CRZ_BTNS"]["BIT1"] == 0
    assert cp.vl["CRZ_BTNS"]["SET_P"] == 0
