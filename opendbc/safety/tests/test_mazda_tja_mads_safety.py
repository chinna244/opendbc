#!/usr/bin/env python3
"""Mazda TJA_MADS / MRCC independence safety tests on current upstream Mazda safety."""
import functools
import random
import unittest

from opendbc.car.mazda.values import MazdaSafetyFlags
from opendbc.car.structs import CarParams
from opendbc.safety.tests.libsafety import libsafety_py
import opendbc.safety.tests.common as common
from opendbc.safety.tests.test_mazda import TestMazdaSafety, TestMazdaSteerToZeroEpsSafety


def require_tja_mads(func):
  @functools.wraps(func)
  def wrapped(self, *args, **kwargs):
    if not (int(self.SAFETY_PARAM) & MazdaSafetyFlags.TJA_MADS):
      self.skipTest("requires TJA_MADS")
    return func(self, *args, **kwargs)
  return wrapped


class _MazdaTjaMadsTestHelpers:
  """Shared CRZ_BTNS / fwd_modify helpers for TJA-MADS and stock-steering safety tests."""

  # FSC-only TJA isolation: Intel bit 11 (byte 1 bit 3) on the bus0->bus2 copy.
  _TJA_BYTE = 1
  _TJA_MASK = 0x08

  @staticmethod
  def _pkt_bytes(msg):
    return bytes(msg[0].data[0:8])

  def _mrcc_armed_msg(self, armed):
    values = {"CRZ_AVAILABLE": int(armed)}
    return self.packer.make_can_msg_safety("CRZ_CTRL", 0, values)

  def _mrcc_off_button_msg(self):
    values = {
      "CAN_OFF": 0, "CAN_OFF_INV": 1,
      "SET_P": 0, "SET_P_INV": 1,
      "RES": 0, "RES_INV": 1,
      "SET_M": 0, "SET_M_INV": 1,
      "DISTANCE_LESS": 0, "DISTANCE_LESS_INV": 1,
      "DISTANCE_MORE": 0, "DISTANCE_MORE_INV": 1,
      "TJA_BUTTON": 0,
      "MODE_X": 0, "MODE_X_INV": 1,
      "MODE_Y": 0, "MODE_Y_INV": 1,
      "BIT1": 0, "BIT1_INV": 1, "BIT2": 1, "BIT3": 1,
      "CTR": 5,
    }
    return self.packer.make_can_msg_safety("CRZ_BTNS", 0, values)

  def _fwd_copy(self, src_bus, msg):
    orig = self._pkt_bytes(msg)
    clone = libsafety_py.make_CANPacket(int(msg[0].addr), int(msg[0].bus), orig)
    self.safety.safety_fwd_modify(src_bus, clone)
    return orig, self._pkt_bytes(clone)

  def _assert_only_tja_cleared(self, orig, fwd):
    self.assertEqual(len(orig), 8)
    self.assertEqual(len(fwd), 8)
    expected = bytearray(orig)
    expected[self._TJA_BYTE] &= ~self._TJA_MASK
    self.assertEqual(bytes(expected), fwd)
    for bit in range(64):
      orig_bit = (orig[bit // 8] >> (bit % 8)) & 1
      fwd_bit = (fwd[bit // 8] >> (bit % 8)) & 1
      if bit == 11:
        self.assertEqual(0, fwd_bit)
      else:
        self.assertEqual(orig_bit, fwd_bit, f"bit {bit} changed")


class TestMazdaTjaMadsSafety(_MazdaTjaMadsTestHelpers, TestMazdaSteerToZeroEpsSafety):
  """CX-5 2022: steer-to-zero EPS plus physical TJA as the MADS lateral master."""

  SAFETY_PARAM = MazdaSafetyFlags.STEER_TO_ZERO_EPS | MazdaSafetyFlags.TJA_MADS

  def _lkas_button_msg(self, enabled):
    values = {"TJA_BUTTON": int(enabled), "BIT1": 1, "BIT2": 1, "BIT3": 1}
    return self.packer.make_can_msg_safety("CRZ_BTNS", 0, values)

  @require_tja_mads
  def test_mrcc_off_tap_allowed_only_while_mrcc_armed(self):
    self.safety.set_controls_allowed(False)
    msg = self._mrcc_off_button_msg()

    self._rx(self._mrcc_armed_msg(False))
    self.assertFalse(self._tx(msg))

    self._rx(self._mrcc_armed_msg(True))
    self.assertTrue(self._tx(msg))

    # Only the exact captured active-low master pair is accepted.
    dat = bytearray(msg.data)
    dat[1] &= 0x7f
    self.assertFalse(self._tx(common.make_msg(0, 0x09d, 8, dat)))

  @require_tja_mads
  def test_mrcc_engage_does_not_grant_or_revoke_mads_lateral(self):
    """MRCC/pcm cruise must not authorize or revoke MADS lateral under TJA_MADS.

    Stock-long uses CRZ_CTRL.CRZ_ACTIVE; alpha-long uses PEDALS.ACC_ACTIVE via
    the longitudinal subclass _pcm_status_msg override.
    """
    self.safety.set_mads_params(True, False, False)
    self.assertFalse(self.safety.get_op_controls_allowed_requests_lateral())
    self.assertFalse(self.safety.get_controls_allowed_lateral())

    # Engage actual cruise without a TJA press.
    if hasattr(self, "_press_set"):
      self._press_set()
    self._rx(self._pcm_status_msg(False))
    self._rx(self._pcm_status_msg(True))
    self.assertTrue(self.safety.get_controls_allowed())
    self.assertFalse(self.safety.get_controls_allowed_lateral())

    # TJA rising edge is the lateral authorization source.
    self._rx(self._lkas_button_msg(True))
    self._rx(self._lkas_button_msg(False))
    self.assertTrue(self.safety.get_controls_allowed_lateral())

    # Cycling MRCC must not toggle lateral authorization.
    self._rx(self._pcm_status_msg(False))
    self.assertFalse(self.safety.get_controls_allowed())
    self.assertTrue(self.safety.get_controls_allowed_lateral())
    self._rx(self._pcm_status_msg(True))
    self.assertTrue(self.safety.get_controls_allowed())
    self.assertTrue(self.safety.get_controls_allowed_lateral())
    self._rx(self._pcm_status_msg(False))
    self.assertFalse(self.safety.get_controls_allowed())
    self.assertTrue(self.safety.get_controls_allowed_lateral())

  @require_tja_mads
  def test_tja_button_sets_mads_press_state(self):
    self.safety.set_mads_params(True, False, False)

    self._rx(self._lkas_button_msg(False))
    self.assertEqual(0, self.safety.get_mads_button_press())
    self.assertFalse(self.safety.get_controls_allowed_lateral())

    self._rx(self._lkas_button_msg(True))
    self.assertEqual(1, self.safety.get_mads_button_press())
    self.assertTrue(self.safety.get_controls_allowed_lateral())

    self._rx(self._lkas_button_msg(True))
    self._rx(self._lkas_button_msg(True))
    self.assertEqual(1, self.safety.get_mads_button_press())
    self.assertTrue(self.safety.get_controls_allowed_lateral())

    self._rx(self._lkas_button_msg(False))
    self.assertEqual(0, self.safety.get_mads_button_press())
    self.assertTrue(self.safety.get_controls_allowed_lateral())

  @require_tja_mads
  def test_tja_second_rising_edge_stays_authorized(self):
    self.safety.set_mads_params(True, False, False)
    self._rx(self._lkas_button_msg(True))
    self._rx(self._lkas_button_msg(False))
    self.assertTrue(self.safety.get_controls_allowed_lateral())
    self._rx(self._lkas_button_msg(True))
    self.assertEqual(1, self.safety.get_mads_button_press())
    self.assertTrue(self.safety.get_controls_allowed_lateral())

  @require_tja_mads
  def test_tja_rising_edge_resets_stale_heartbeat_mismatches(self):
    """A quick MADS off->on press must get a fresh heartbeat grace window."""
    self.safety.set_mads_params(True, False, False)
    self._rx(self._lkas_button_msg(True))
    self._rx(self._lkas_button_msg(False))
    self.assertTrue(self.safety.get_controls_allowed_lateral())

    # Model two old disengaged-heartbeat samples accumulated after userspace
    # disabled MADS, while panda still temporarily retains lateral authorization.
    self.safety.set_heartbeat_engaged_mads(False)
    for _ in range(2):
      self.safety.mads_heartbeat_engaged_check()
    self.assertTrue(self.safety.get_controls_allowed_lateral())

    # A new physical TJA request arrives before the heartbeat catches up. The
    # next check must be sample one of a fresh window, not the old third sample.
    self._rx(self._lkas_button_msg(True))
    self.safety.mads_heartbeat_engaged_check()
    self.assertTrue(self.safety.get_controls_allowed_lateral())

    # The new engaged heartbeat catches up and clears the fresh sample.
    self.safety.set_heartbeat_engaged_mads(True)
    self.safety.mads_heartbeat_engaged_check()
    self.assertTrue(self.safety.get_controls_allowed_lateral())

  @require_tja_mads
  def test_tja_rising_edge_does_not_hide_persistent_heartbeat_failure(self):
    """The reset grants grace, not indefinite authorization without a heartbeat."""
    self.safety.set_mads_params(True, False, False)
    self._rx(self._lkas_button_msg(True))
    self._rx(self._lkas_button_msg(False))
    self.safety.set_heartbeat_engaged_mads(False)

    for _ in range(2):
      self.safety.mads_heartbeat_engaged_check()
    self._rx(self._lkas_button_msg(True))

    for _ in range(2):
      self.safety.mads_heartbeat_engaged_check()
      self.assertTrue(self.safety.get_controls_allowed_lateral())

    self.safety.mads_heartbeat_engaged_check()
    self.assertFalse(self.safety.get_controls_allowed_lateral())

  @require_tja_mads
  def test_tja_grants_lateral_while_mrcc_already_armed(self):
    self.safety.set_mads_params(True, False, False)

    self._rx(self._mrcc_armed_msg(True))
    self.assertFalse(self.safety.get_acc_main_on())
    self.assertFalse(self.safety.get_controls_allowed_lateral())

    self._rx(self._lkas_button_msg(True))
    self.assertTrue(self.safety.get_controls_allowed_lateral())

  @require_tja_mads
  def test_tja_grants_lateral_with_acc_main_already_high(self):
    # MRCC/acc_main already high, MADS lateral off: TJA must still authorize
    # without a new acc_main rising edge.
    self.safety.set_mads_params(True, False, False)
    self.safety.set_acc_main_on(True)
    self._rx(self._speed_msg(0))
    self.safety.set_controls_allowed_lateral(False)
    self.safety.set_controls_requested_lateral(False)
    self._rx(self._speed_msg(0))
    self.assertTrue(self.safety.get_acc_main_on())
    self.assertFalse(self.safety.get_controls_allowed_lateral())

    self._rx(self._lkas_button_msg(True))
    self.assertTrue(self.safety.get_controls_allowed_lateral())

  @require_tja_mads
  def test_mrcc_falling_does_not_exit_mads_lateral(self):
    self.safety.set_mads_params(True, False, False)
    self._rx(self._lkas_button_msg(True))
    self._rx(self._lkas_button_msg(False))
    self.assertTrue(self.safety.get_controls_allowed_lateral())

    self._rx(self._mrcc_armed_msg(True))
    self._rx(self._mrcc_armed_msg(False))
    self.assertFalse(self.safety.get_acc_main_on())
    self.assertTrue(self.safety.get_controls_allowed_lateral())

  @require_tja_mads
  def test_set_res_cancel_do_not_grant_mads_lateral(self):
    self.safety.set_mads_params(True, False, False)
    self._rx(self._button_msg(resume=True))
    self.assertEqual(0, self.safety.get_mads_button_press())
    self.assertFalse(self.safety.get_controls_allowed_lateral())

    self._rx(self._button_msg(cancel=True))
    self.assertEqual(0, self.safety.get_mads_button_press())
    self.assertFalse(self.safety.get_controls_allowed_lateral())

  @require_tja_mads
  def test_mode_x_y_do_not_grant_mads_lateral(self):
    self.safety.set_mads_params(True, False, False)
    for values in (
      {"MODE_X": 1, "MODE_Y": 0},
      {"MODE_X": 0, "MODE_Y": 1},
      {"MODE_X": 1, "MODE_Y": 1},
    ):
      msg = self.packer.make_can_msg_safety("CRZ_BTNS", 0, {**values, "BIT1": 1, "BIT2": 1, "BIT3": 1})
      self._rx(msg)
      self.assertEqual(0, self.safety.get_mads_button_press())
      self.assertFalse(self.safety.get_controls_allowed_lateral())

  @require_tja_mads
  def test_fsc_tja_isolation_passthrough_when_mads_feature_disabled(self):
    self.safety.set_mads_params(False, False, False)
    self.safety.set_heartbeat_engaged_mads(True)
    msg = self._lkas_button_msg(True)
    orig, fwd = self._fwd_copy(0, msg)
    self.assertEqual(self._TJA_MASK, orig[self._TJA_BYTE] & self._TJA_MASK)
    self.assertEqual(orig, fwd)

  @require_tja_mads
  def test_fsc_tja_isolation_strips_before_heartbeat_transition(self):
    # MADS feature on, but openpilot heartbeat still reports not engaged.
    self.safety.set_mads_params(True, False, False)
    self.safety.set_heartbeat_engaged_mads(False)
    self.assertTrue(self.safety.get_enable_mads())

    pressed = self._lkas_button_msg(True)
    orig, fwd = self._fwd_copy(0, pressed)
    self._assert_only_tja_cleared(orig, fwd)

    # Heartbeat catches up later; strip policy must not depend on it.
    self.safety.set_heartbeat_engaged_mads(True)
    orig, fwd = self._fwd_copy(0, pressed)
    self._assert_only_tja_cleared(orig, fwd)

  @require_tja_mads
  def test_fsc_tja_isolation_strips_with_stale_heartbeat_on_disable_edge(self):
    # MADS feature on; runtime disengaged but heartbeat still reports engaged.
    self.safety.set_mads_params(True, False, False)
    self.safety.set_heartbeat_engaged_mads(True)

    pressed = self._lkas_button_msg(True)
    orig, fwd = self._fwd_copy(0, pressed)
    self._assert_only_tja_cleared(orig, fwd)

  @require_tja_mads
  def test_fsc_tja_isolation_panda_rx_sees_original_and_fwd_clears_tja(self):
    self.safety.set_mads_params(True, False, False)
    msg = self._lkas_button_msg(True)
    orig = self._pkt_bytes(msg)
    self.assertEqual(self._TJA_MASK, orig[self._TJA_BYTE] & self._TJA_MASK)

    self.assertEqual(2, self.safety.safety_fwd_hook(0, 0x09d))
    orig_fwd, fwd = self._fwd_copy(0, msg)
    self.assertEqual(orig, orig_fwd)
    self.assertEqual(0, fwd[self._TJA_BYTE] & self._TJA_MASK)
    self._assert_only_tja_cleared(orig, fwd)

    # Original bus0 frame is what mazda_rx_hook sees (fdcan RX uses to_push).
    self._rx(msg)
    self.assertEqual(1, self.safety.get_mads_button_press())
    self.assertTrue(self.safety.get_controls_allowed_lateral())
    self.assertEqual(orig, self._pkt_bytes(msg))

  @require_tja_mads
  def test_fsc_tja_isolation_tja_zero_frame_unchanged(self):
    msg = self._lkas_button_msg(False)
    orig, fwd = self._fwd_copy(0, msg)
    self.assertEqual(0, orig[self._TJA_BYTE] & self._TJA_MASK)
    self.assertEqual(orig, fwd)

  @require_tja_mads
  def test_fsc_tja_isolation_preserves_set_res_cancel_mode_bits(self):
    self.safety.set_mads_params(True, False, False)
    combos = (
      {"SET_P": 1, "SET_P_INV": 0},
      {"SET_M": 1, "SET_M_INV": 0},
      {"RES": 1, "RES_INV": 0},
      {"CAN_OFF": 1, "CAN_OFF_INV": 0},
      {"MODE_X": 1, "MODE_X_INV": 0},
      {"MODE_Y": 1, "MODE_Y_INV": 0},
      {"SET_P": 1, "SET_P_INV": 0, "RES": 1, "RES_INV": 0, "CAN_OFF": 1, "CAN_OFF_INV": 0,
       "MODE_X": 1, "MODE_X_INV": 0, "MODE_Y": 1, "MODE_Y_INV": 0, "TJA_BUTTON": 1},
    )
    for values in combos:
      msg = self.packer.make_can_msg_safety("CRZ_BTNS", 0, {**values, "BIT1": 1, "BIT2": 1, "BIT3": 1})
      orig, fwd = self._fwd_copy(0, msg)
      self._assert_only_tja_cleared(orig, fwd)

  @require_tja_mads
  def test_fsc_tja_isolation_reserved_bit_corpus(self):
    self.safety.set_mads_params(True, False, False)
    rng = random.Random(47)
    for _ in range(256):
      dat = bytes(rng.getrandbits(8) for _ in range(8))
      msg = libsafety_py.make_CANPacket(0x09d, 0, dat)
      orig, fwd = self._fwd_copy(0, msg)
      self._assert_only_tja_cleared(orig, fwd)

  @require_tja_mads
  def test_fsc_tja_isolation_does_not_touch_other_addrs_or_bus2(self):
    dat = bytes(range(8))
    for addr in (0x21c, 0x21b, 0x440, 0x243, 0x165, 0x202):
      msg = libsafety_py.make_CANPacket(addr, 0, dat)
      orig, fwd = self._fwd_copy(0, msg)
      self.assertEqual(orig, fwd)

    msg = libsafety_py.make_CANPacket(0x09d, 2, dat)
    orig, fwd = self._fwd_copy(2, msg)
    self.assertEqual(orig, fwd)

  @require_tja_mads
  def test_fsc_tja_isolation_hold_release_does_not_fabricate_mads_edges(self):
    self.safety.set_mads_params(True, False, False)
    pressed = self._lkas_button_msg(True)
    released = self._lkas_button_msg(False)

    self._rx(released)
    self.assertEqual(0, self.safety.get_mads_button_press())
    self.assertFalse(self.safety.get_controls_allowed_lateral())

    for _ in range(4):
      orig, fwd = self._fwd_copy(0, pressed)
      self._assert_only_tja_cleared(orig, fwd)
      self._rx(pressed)
      self.assertEqual(1, self.safety.get_mads_button_press())
    self.assertTrue(self.safety.get_controls_allowed_lateral())

    orig, fwd = self._fwd_copy(0, released)
    self.assertEqual(orig, fwd)
    self._rx(released)
    self.assertEqual(0, self.safety.get_mads_button_press())
    self.assertTrue(self.safety.get_controls_allowed_lateral())

  def _passthrough_probe_msg(self, addr):
    return self._torque_cmd_msg(0) if addr == 0x243 else self._laneinfo_msg()

  def _set_engagement(self, controls_allowed, controls_allowed_lateral):
    self.safety.set_controls_allowed(controls_allowed)
    self.safety.set_controls_allowed_lateral(controls_allowed_lateral)

  def _stock_passthrough_states(self):
    # the camera owns 0x243/0x440 only while openpilot controls neither axis;
    # engaging either axis hands the addresses to openpilot
    return [
      (True, lambda: self._set_engagement(False, False)),
      (False, lambda: self._set_engagement(True, False)),
      (False, lambda: self._set_engagement(False, True)),
    ]




class TestMazdaStockSteeringSafety(_MazdaTjaMadsTestHelpers, TestMazdaSafety):
  """Pre-2022 / stock EPS envelope: 800 Nm, 10/25 rate, driver multiplier 1. No TJA_MADS."""

  MAX_RATE_UP = 10
  MAX_RATE_DOWN = 25
  MAX_TORQUE_LOOKUP = [0], [800]
  DRIVER_TORQUE_FACTOR = 1
  DRIVER_TORQUE_ALLOWANCE = 15
  SAFETY_PARAM = 0

  def _lkas_button_msg(self, enabled):
    raise NotImplementedError

  def test_high_torque_rejected_without_steer_to_zero(self):
    self.safety.set_controls_allowed(True)
    self.assertFalse(self._tx(self._torque_cmd_msg(900)))

  def test_stock_rate_up_rejected(self):
    self.safety.set_controls_allowed(True)
    self.safety.set_desired_torque_last(0)
    self.assertTrue(self._tx(self._torque_cmd_msg(self.MAX_RATE_UP)))
    self.safety.set_desired_torque_last(0)
    self.assertFalse(self._tx(self._torque_cmd_msg(self.MAX_RATE_UP + 1)))

  def test_tja_does_not_grant_mads_without_tja_mads(self):
    self.safety.set_mads_params(True, False, False)
    self.assertTrue(self.safety.get_op_controls_allowed_requests_lateral())
    msg = self.packer.make_can_msg_safety("CRZ_BTNS", 0, {"TJA_BUTTON": 1, "BIT1": 1, "BIT2": 1, "BIT3": 1})
    self._rx(msg)
    self.assertEqual(-1, self.safety.get_mads_button_press())
    self.assertFalse(self.safety.get_controls_allowed_lateral())

  def test_mrcc_off_tap_not_allowed_without_tja_mads(self):
    self.safety.set_controls_allowed(False)
    self._rx(self._mrcc_armed_msg(True))
    self.assertFalse(self._tx(self._mrcc_off_button_msg()))

  def test_mrcc_engage_still_grants_lateral_without_tja_mads(self):
    """Upstream lateral auth via op_controls_allowed rising must remain for non-TJA Mazdas."""
    self.safety.set_mads_params(True, False, False)
    self.assertTrue(self.safety.get_op_controls_allowed_requests_lateral())
    self._rx(self._pcm_status_msg(False))
    self._rx(self._pcm_status_msg(True))
    self.assertTrue(self.safety.get_controls_allowed())
    self.assertTrue(self.safety.get_controls_allowed_lateral())

  def test_op_controls_allowed_requests_lateral_survives_mads_params_and_mode_switch(self):
    """Car-mode config for op_controls_allowed lateral requests must outlive set_mads_params."""
    # TJA_MADS -> disabled as a lateral source; set_mads_params must not re-arm it.
    self.safety.set_safety_hooks(CarParams.SafetyModel.mazda,
                                 int(MazdaSafetyFlags.STEER_TO_ZERO_EPS | MazdaSafetyFlags.TJA_MADS))
    self.safety.init_tests()
    self.assertFalse(self.safety.get_op_controls_allowed_requests_lateral())
    self.safety.set_mads_params(True, False, False)
    self.assertFalse(self.safety.get_op_controls_allowed_requests_lateral())
    self.safety.set_mads_params(False, False, False)
    self.assertFalse(self.safety.get_op_controls_allowed_requests_lateral())

    # Non-TJA Mazda restores the upstream default; set_mads_params must keep it.
    self.safety.set_safety_hooks(CarParams.SafetyModel.mazda, 0)
    self.safety.init_tests()
    self.assertTrue(self.safety.get_op_controls_allowed_requests_lateral())
    self.safety.set_mads_params(True, False, False)
    self.assertTrue(self.safety.get_op_controls_allowed_requests_lateral())
    self.safety.set_mads_params(False, True, False)
    self.assertTrue(self.safety.get_op_controls_allowed_requests_lateral())

    # Switching back to TJA_MADS disables it again.
    self.safety.set_safety_hooks(CarParams.SafetyModel.mazda, int(MazdaSafetyFlags.TJA_MADS))
    self.safety.init_tests()
    self.assertFalse(self.safety.get_op_controls_allowed_requests_lateral())
    self.safety.set_mads_params(True, False, False)
    self.assertFalse(self.safety.get_op_controls_allowed_requests_lateral())

  def test_fsc_tja_isolation_inactive_without_tja_mads(self):
    self.safety.set_mads_params(True, False, False)
    msg = self.packer.make_can_msg_safety("CRZ_BTNS", 0, {"TJA_BUTTON": 1, "BIT1": 1, "BIT2": 1, "BIT3": 1})
    orig, fwd = self._fwd_copy(0, msg)
    self.assertEqual(self._TJA_MASK, orig[self._TJA_BYTE] & self._TJA_MASK)
    self.assertEqual(orig, fwd)


class TestMazdaTjaMadsWithoutSteerToZero(TestMazdaSafety):
  """TJA_MADS must not change the stock steering envelope."""

  MAX_RATE_UP = 10
  MAX_RATE_DOWN = 25
  MAX_TORQUE_LOOKUP = [0], [800]
  DRIVER_TORQUE_FACTOR = 1
  DRIVER_TORQUE_ALLOWANCE = 15
  SAFETY_PARAM = MazdaSafetyFlags.TJA_MADS

  def test_high_torque_rejected_without_steer_to_zero(self):
    self.safety.set_controls_allowed(True)
    self.assertFalse(self._tx(self._torque_cmd_msg(900)))

