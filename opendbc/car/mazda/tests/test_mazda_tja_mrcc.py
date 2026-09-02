#!/usr/bin/env python3
"""TJA/MADS independence: ICBM suppress scoping and TJA-caused MRCC cleanup."""
from types import SimpleNamespace

import pytest

from opendbc.can import CANParser
from opendbc.car import Bus, DT_CTRL, structs
from opendbc.car.mazda import mazdacan
from opendbc.car.mazda.carcontroller import (CarController, TJA_MRCC_FIRST_TX_DELAY_NANOS,
                                             TJA_MRCC_RAW_OFF_CONFIRM_FRAMES, TJA_MRCC_MAX_TX_FRAMES,
                                             TJA_MRCC_RELEASE_WAIT_FRAMES)
from opendbc.car.mazda.interface import CarInterface
from opendbc.car.mazda.values import CAR, Buttons, MazdaSafetyFlags

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
          cancel_button=0, resume_button=0, accel_button=0, decel_button=0, mrcc_button=0,
          distance_button=0, distance_button_active=0):
    if mrcc_armed_raw is None:
      mrcc_armed_raw = cruise_available
    return SimpleNamespace(
      out=SimpleNamespace(vEgoRaw=12.0, steeringTorque=0, brakePressed=False,
                          cruiseState=SimpleNamespace(available=cruise_available)),
      cruise_available=cruise_available,
      distance_button=distance_button,
      distance_button_active=distance_button_active,
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
      lkas_blocked=False,
      lkas_effective=0,
      steer_undelivered=False,
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
