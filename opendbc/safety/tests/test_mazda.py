#!/usr/bin/env python3
import functools
import random
import unittest

from opendbc.car.mazda.values import MazdaSafetyFlags
from opendbc.car.structs import CarParams
from opendbc.safety.tests.libsafety import libsafety_py
import opendbc.safety.tests.common as common
from opendbc.safety.tests.common import CANPackerSafety, make_msg


def require_tja_mads(func):
  @functools.wraps(func)
  def wrapped(self, *args, **kwargs):
    if not (int(self.SAFETY_PARAM) & MazdaSafetyFlags.TJA_MADS):
      self.skipTest("requires TJA_MADS")
    return func(self, *args, **kwargs)
  return wrapped


class TestMazdaSafety(common.CarSafetyTest, common.DriverTorqueSteeringSafetyTest):

  TX_MSGS = [[0x243, 0], [0x09d, 0], [0x440, 0]]
  STANDSTILL_THRESHOLD = .1
  RELAY_MALFUNCTION_ADDRS = {0: (0x243, 0x440)}
  FWD_BLACKLISTED_ADDRS = {2: [0x243, 0x440]}

  MAX_RATE_UP = 12
  MAX_RATE_DOWN = 25
  MAX_TORQUE_LOOKUP = [0], [1200]

  MAX_RT_DELTA = 384

  DRIVER_TORQUE_ALLOWANCE = 15
  DRIVER_TORQUE_FACTOR = 15

  # Mazda actually does not set any bit when requesting torque
  NO_STEER_REQ_BIT = True

  SAFETY_PARAM = MazdaSafetyFlags.STEER_TO_ZERO | MazdaSafetyFlags.TJA_MADS

  def setUp(self):
    self.packer = CANPackerSafety("mazda_2017")
    self.safety = libsafety_py.libsafety
    self.safety.set_safety_hooks(CarParams.SafetyModel.mazda, int(self.SAFETY_PARAM))
    self.safety.init_tests()

  def _torque_meas_msg(self, torque):
    values = {"STEER_TORQUE_MOTOR": torque}
    return self.packer.make_can_msg_safety("STEER_TORQUE", 0, values)

  def _torque_driver_msg(self, torque):
    values = {"STEER_TORQUE_SENSOR": torque}
    return self.packer.make_can_msg_safety("STEER_TORQUE", 0, values)

  def _torque_cmd_msg(self, torque, steer_req=1):
    values = {"LKAS_REQUEST": torque}
    return self.packer.make_can_msg_safety("CAM_LKAS", 0, values)

  def _speed_msg(self, speed):
    values = {"SPEED": speed}
    return self.packer.make_can_msg_safety("ENGINE_DATA", 0, values)

  def _user_brake_msg(self, brake):
    values = {"BRAKE_ON": brake}
    return self.packer.make_can_msg_safety("PEDALS", 0, values)

  def _user_gas_msg(self, gas):
    values = {"PEDAL_GAS": gas}
    return self.packer.make_can_msg_safety("ENGINE_DATA", 0, values)

  def _pcm_status_msg(self, enable):
    values = {"CRZ_ACTIVE": enable}
    return self.packer.make_can_msg_safety("CRZ_CTRL", 0, values)

  def _button_msg(self, resume=False, cancel=False, set_m=False, set_p=False):
    values = {
      "CAN_OFF": cancel,
      "CAN_OFF_INV": (cancel + 1) % 2,
      "RES": resume,
      "RES_INV": (resume + 1) % 2,
      "SET_M": set_m,
      "SET_M_INV": (set_m + 1) % 2,
      "SET_P": set_p,
      "SET_P_INV": (set_p + 1) % 2,
    }
    return self.packer.make_can_msg_safety("CRZ_BTNS", 0, values)

  def _lkas_button_msg(self, enabled):
    values = {"TJA_BUTTON": int(enabled), "BIT1": 1, "BIT2": 1, "BIT3": 1}
    return self.packer.make_can_msg_safety("CRZ_BTNS", 0, values)

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

  def test_buttons(self):
    # only cancel allows while controls not allowed
    self.safety.set_controls_allowed(0)
    self.assertTrue(self._tx(self._button_msg(cancel=True)))
    self.assertFalse(self._tx(self._button_msg(resume=True)))

    # do not block resume if we are engaged already
    self.safety.set_controls_allowed(1)
    self.assertTrue(self._tx(self._button_msg(cancel=True)))
    self.assertTrue(self._tx(self._button_msg(resume=True)))

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

  # FSC-only TJA isolation: Intel bit 11 (byte 1 bit 3) on the bus0->bus2 copy.
  _TJA_BYTE = 1
  _TJA_MASK = 0x08

  @staticmethod
  def _pkt_bytes(msg):
    return bytes(msg[0].data[0:8])

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


class TestMazdaLongitudinalSafety(TestMazdaSafety, common.LongitudinalAccelSafetyTest):

  TX_MSGS = [[0x243, 0], [0x09d, 0], [0x440, 0], [0x21b, 0], [0x21c, 0], [0x499, 0],
             [0x361, 0], [0x362, 0], [0x363, 0], [0x364, 0], [0x365, 0], [0x366, 0], [0x764, 0],
             [0x21b, 2], [0x21c, 2], [0x499, 2], [0x361, 2], [0x362, 2], [0x363, 2], [0x364, 2], [0x365, 2], [0x366, 2]]

  SAFETY_PARAM = MazdaSafetyFlags.LONG | MazdaSafetyFlags.STEER_TO_ZERO | MazdaSafetyFlags.TJA_MADS

  def setUp(self):
    self.packer = CANPackerSafety("mazda_2017")
    self.safety = libsafety_py.libsafety
    self.safety.set_safety_hooks(CarParams.SafetyModel.mazda, int(self.SAFETY_PARAM))
    self.safety.init_tests()

  def _pcm_status_msg(self, enable):
    values = {"ACC_ACTIVE": enable, "BRAKE_ON": 0}
    return self.packer.make_can_msg_safety("PEDALS", 0, values)

  def _mrcc_armed_msg(self, armed):
    values = {"ACC_OFF": int(armed), "ACC_ACTIVE": 0, "BRAKE_ON": 0}
    return self.packer.make_can_msg_safety("PEDALS", 0, values)

  def _accel_msg(self, accel: float, bus: int = 0, active: bool = False):
    values = {"ACCEL_CMD": accel, "ACC_ACTIVE": active}
    return self.packer.make_can_msg_safety("CRZ_INFO", bus, values)

  def _crz_ctrl_cmd_msg(self, active: bool, bus: int = 0):
    values = {"CRZ_ACTIVE": active}
    return self.packer.make_can_msg_safety("CRZ_CTRL", bus, values)

  def test_brake_only_sample_does_not_block_pending_mrcc_off_tap(self):
    self.safety.set_controls_allowed(False)
    self._rx(self._mrcc_armed_msg(True))
    self._rx(self._user_brake_msg(True))
    self.assertTrue(self._tx(self._mrcc_off_button_msg()))

  def _press_set(self):
    # arm the driver-intent qualifier the way every logged engagement does: a wheel press
    # lands 30-70 ms before PEDALS.ACC_ACTIVE rises
    self._rx(self._button_msg(set_m=True))

  def test_enable_control_allowed_from_cruise(self):
    # the common test plus the driver-intent qualifier this mode requires
    self._press_set()
    super().test_enable_control_allowed_from_cruise()

  def test_cruise_without_button_never_arms(self):
    # PEDALS.ACC_ACTIVE alone is the body answering our own fabricated frames; without a
    # SET/RES press heard from the wheel it must not arm controls
    self._rx(self._pcm_status_msg(False))
    for _ in range(12):
      self._rx(self._pcm_status_msg(True))
      self.assertFalse(self.safety.get_controls_allowed())

  def test_button_window_expires(self):
    self._press_set()
    # 10 Hz CRZ_BTNS: run the countdown past the 1 s window with idle button frames
    for _ in range(12):
      self._rx(self._button_msg())
    self._rx(self._pcm_status_msg(True))
    self.assertFalse(self.safety.get_controls_allowed())

  def test_armed_controls_latch_past_the_window(self):
    self._press_set()
    self._rx(self._pcm_status_msg(True))
    self.assertTrue(self.safety.get_controls_allowed())
    # the window expiring must not drop an active engagement
    for _ in range(12):
      self._rx(self._button_msg())
      self._rx(self._pcm_status_msg(True))
      self.assertTrue(self.safety.get_controls_allowed())
    self._rx(self._pcm_status_msg(False))
    self.assertFalse(self.safety.get_controls_allowed())

  def test_each_engage_button_arms(self):
    for btn in ("set_m", "set_p", "resume"):
      self._rx(self._button_msg(**{btn: True}))
      self._rx(self._pcm_status_msg(True))
      self.assertTrue(self.safety.get_controls_allowed(), btn)
      self._rx(self._pcm_status_msg(False))

  def test_crz_info_active_gated_on_controls(self):
    # ACC_ACTIVE mirrors CRZ_CTRL's gate: an engaged-claiming accel frame must not flow while
    # controls are not allowed. The body raises PEDALS.ACC_ACTIVE off the SET press before
    # our first engaged frame in every logged engagement, so there is no deadlock.
    for bus in (0, 2):
      for active in (False, True):
        msg = self._accel_msg(self.INACTIVE_ACCEL, bus=bus, active=active)
        self.safety.set_controls_allowed(False)
        self.assertEqual(not active, self._tx(msg))
        self.safety.set_controls_allowed(True)
        self.assertTrue(self._tx(msg))

  def test_synthetic_lead_radar_track_allowed_disengaged(self):
    # DIST_OBJ and RELV_OBJ are free fields; the template bytes must match. The non-template
    # frames are real on-road emissions (route 6bb2dc61c4), which a byte-exact check silently
    # dropped -- 982 asked, 0 transmitted -- starving the camera of the track. The slot is
    # perception, not actuation, so it flows with controls_allowed low the way a stock radar
    # reports objects with cruise off.
    lead_frames = [
      "0a4000001dc00000",  # the fabricated stopped lead at 10.25 m
      "229000007dc0000e",  # lead at 34.56 m, closing slowly
      "22d000ff7dc00004",  # lead at 34.81 m, opening slowly
      "000000001dc00000",  # zero range, zero relv corner
      "fff000fffdc0000f",  # max range, max relv corner
    ]
    for bus in (0, 2):
      for hexdat in lead_frames:
        dat = bytes.fromhex(hexdat)
        for controls_allowed in (False, True):
          self.safety.set_controls_allowed(controls_allowed)
          self.assertTrue(self._tx(common.make_msg(bus, 0x364, 8, dat)))

  def test_camera_bus_accel_actuation_limits(self):
    # the synthetic radar frames are duplicated onto the camera bus; same limits apply there
    for accel in (self.MIN_ACCEL - 1, self.MIN_ACCEL, self.INACTIVE_ACCEL, self.MAX_ACCEL, self.MAX_ACCEL + 1):
      for controls_allowed in (True, False):
        self.safety.set_controls_allowed(controls_allowed)
        should_tx = controls_allowed and self.MIN_ACCEL <= accel <= self.MAX_ACCEL
        should_tx = should_tx or accel == self.INACTIVE_ACCEL
        self.assertEqual(should_tx, self._tx(self._accel_msg(accel, bus=2)))

  def test_stock_crz_info_standby_allowed(self):
    # stock standby pegs the command field high; it must pass byte-exactly, checksum included,
    # instead of being decoded as a huge accel command
    for controls_allowed in (False, True):
      self.safety.set_controls_allowed(controls_allowed)
      for bus in (0, 2):
        for counter in range(16):
          checksum = (0x5d - counter) & 0xff
          dat = bytes.fromhex(f"01ffe3ffc000{counter:02x}{checksum:02x}")
          self.assertTrue(self._tx(common.make_msg(bus, 0x21b, 8, dat)))

        bad_checksum = bytes.fromhex("01ffe3ffc0000000")
        self.assertFalse(self._tx(common.make_msg(bus, 0x21b, 8, bad_checksum)))

  def test_empty_radar_tracks_allowed(self):
    radar_messages = {
      0x499: bytes.fromhex("0008c00000000000"),
      0x361: bytes.fromhex("fff7fefe1fc00080"),
      0x362: bytes.fromhex("fff7fefe1fc78c80"),
      0x363: bytes.fromhex("fff7fefe1fc00000"),
      0x364: bytes.fromhex("fff7fefe1fc00000"),
      0x365: bytes.fromhex("fff7fe7ffbff3fc0"),
      0x366: bytes.fromhex("fff7fe7ffbff3fc0"),
    }

    for controls_allowed in (False, True):
      self.safety.set_controls_allowed(controls_allowed)
      for bus in (0, 2):
        for addr, dat in radar_messages.items():
          self.assertTrue(self._tx(common.make_msg(bus, addr, 8, dat)))


  def test_malformed_lead_radar_track_blocked(self):
    # each corrupts one template-owned field of a valid lead frame
    bad_frames = [
      "229100007dc0000e",  # data[1] low nibble not zero
      "229001007dc0000e",  # data[2] not zero
      "229000007cc0000e",  # data[4] template bits wrong
      "229000007dc1000e",  # data[5] wrong
      "229000007dc0010e",  # data[6] not zero
      "229000007dc0100e",  # data[7] high nibble not zero
    ]
    self.safety.set_controls_allowed(True)
    for bus in (0, 2):
      for hexdat in bad_frames:
        self.assertFalse(self._tx(common.make_msg(bus, 0x364, 8, bytes.fromhex(hexdat))))

  def test_unexpected_radar_tracks_blocked(self):
    bad_messages = {
      0x499: bytes.fromhex("0008c00100000000"),
      0x361: bytes.fromhex("fff7fefe1fc00180"),
      0x362: bytes.fromhex("fff7fefe1fc00080"),
      0x363: bytes.fromhex("fff7fefe1fc00080"),
      0x364: bytes.fromhex("fff7fefe1fc00080"),
      0x365: bytes.fromhex("fff7fe7ffbff3f80"),
      0x366: bytes.fromhex("fff7fe7ffbff3f80"),
    }

    self.safety.set_controls_allowed(True)
    for bus in (0, 2):
      for addr, dat in bad_messages.items():
        self.assertFalse(self._tx(common.make_msg(bus, addr, 8, dat)))

  def test_radar_uds_allowlist(self):
    # tester present and session control only, main bus only
    self.assertTrue(self._tx(common.make_msg(0, 0x764, 8, bytes.fromhex("023e800000000000"))))
    self.assertTrue(self._tx(common.make_msg(0, 0x764, 8, bytes.fromhex("0210020000000000"))))
    self.assertFalse(self._tx(common.make_msg(0, 0x764, 8, bytes.fromhex("0210030000000000"))))
    self.assertFalse(self._tx(common.make_msg(0, 0x764, 8, bytes.fromhex("0227010000000000"))))
    self.assertFalse(self._tx(common.make_msg(2, 0x764, 8, bytes.fromhex("023e800000000000"))))

  def test_crz_ctrl_active_gated_on_controls(self):
    for bus in (0, 2):
      self.safety.set_controls_allowed(False)
      self.assertFalse(self._tx(self._crz_ctrl_cmd_msg(True, bus)))
      self.assertTrue(self._tx(self._crz_ctrl_cmd_msg(False, bus)))

      self.safety.set_controls_allowed(True)
      self.assertTrue(self._tx(self._crz_ctrl_cmd_msg(True, bus)))

  # a stock armed-idle CRZ_INFO standby frame, checksum-correct: what the controller emits
  # from the moment the radar teardown lands
  SYNTHETIC_CRZ_INFO_STANDBY = bytes.fromhex("01ffe3ffc000005d")

  def _acc_armed_msg(self, armed):
    # PEDALS with MRCC armed-but-idle (ACC_OFF), the state that persists across ignition
    values = {"ACC_OFF": armed, "BRAKE_ON": 0}
    return self.packer.make_can_msg_safety("PEDALS", 0, values)

  def test_acc_main_waits_for_the_radar_mastery_latch(self):
    # Routes 116/117 (2026-08-27): MADS keys lateral off acc_main_on's rising edge, and the
    # software gates its availability on 1 s of stock-radar silence. The panda cannot rx the
    # stock CRZ_INFO (deliberately not an rx check: it goes stale at the teardown), so it
    # mirrors the latch off the observable stand-in: our own first synthetic CRZ_INFO tx
    # (= the teardown landing) plus 1 s of the 50 Hz PEDALS clock. Both machines then arm on
    # the same frame; before that, MRCC-armed PEDALS must not raise acc_main_on, or the edge
    # is consumed at boot and the software's later MADS window transmits into rejections
    # that starve the EPS of 0x243.
    self.safety.set_mads_params(True, False, False)
    # boot: teardown not landed yet, MRCC main armed from the first frame
    for _ in range(120):
      self._rx(self._acc_armed_msg(True))
      self.assertFalse(self.safety.get_acc_main_on())
      self.assertFalse(self.safety.get_controls_allowed_lateral())
    self.assertFalse(self._tx(self._torque_cmd_msg(5)))
    # the teardown lands: the controller starts replaying the radar
    self.assertTrue(self._tx(common.make_msg(0, 0x21b, 8, self.SYNTHETIC_CRZ_INFO_STANDBY)))
    # the latch completes after 1 s of the 50 Hz PEDALS clock
    for _ in range(50):
      self.assertFalse(self.safety.get_acc_main_on())
      self._rx(self._acc_armed_msg(True))
    self.assertTrue(self.safety.get_acc_main_on())
    self.assertTrue(self.safety.get_controls_allowed_lateral())
    self.assertTrue(self._tx(self._torque_cmd_msg(5)))

  def test_camera_bus_radar_tx_does_not_master(self):
    # only the main-bus replay marks mastery; the camera-bus copy is a duplicate
    self.safety.set_mads_params(True, False, False)
    self.assertTrue(self._tx(common.make_msg(2, 0x21b, 8, self.SYNTHETIC_CRZ_INFO_STANDBY)))
    for _ in range(60):
      self._rx(self._acc_armed_msg(True))
    self.assertFalse(self.safety.get_acc_main_on())

  def test_acc_main_follows_armed_state_after_the_latch(self):
    # after the latch, acc_main_on tracks PEDALS arming both ways (main off must still exit)
    self.safety.set_mads_params(True, False, False)
    self.assertTrue(self._tx(common.make_msg(0, 0x21b, 8, self.SYNTHETIC_CRZ_INFO_STANDBY)))
    for _ in range(60):
      self._rx(self._acc_armed_msg(True))
    self.assertTrue(self.safety.get_acc_main_on())
    self._rx(self._acc_armed_msg(False))
    self.assertFalse(self.safety.get_acc_main_on())
    self._rx(self._acc_armed_msg(True))
    self.assertTrue(self.safety.get_acc_main_on())

  def test_crz_info_active_gated_on_controls(self):
    # ACC_ACTIVE mirrors CRZ_CTRL's gate: an engaged-claiming accel frame must not flow while
    # controls are not allowed. The body raises PEDALS.ACC_ACTIVE off the SET press before
    # our first engaged frame in every logged engagement, so there is no deadlock.
    for bus in (0, 2):
      for active in (False, True):
        msg = self._accel_msg(self.INACTIVE_ACCEL, bus=bus, active=active)
        self.safety.set_controls_allowed(False)
        self.assertEqual(not active, self._tx(msg))
        self.safety.set_controls_allowed(True)
        self.assertTrue(self._tx(msg))

class TestMazdaIgnition(unittest.TestCase):
  TX_MSGS: list = []

  def setUp(self):
    self.safety = libsafety_py.libsafety
    self.safety.init_tests()

  def _msg(self, byte0):
    return make_msg(0, 0x9E, dat=bytes([byte0]) + b"\x00" * 7)

  # 0x9E byte 0 high 3 bits == 6 (0xC0)
  def test_ignition_on(self):
    self.safety.ignition_can_hook(self._msg(0xC0))
    self.assertTrue(self.safety.get_ignition_can())

  def test_ignition_off(self):
    self.safety.ignition_can_hook(self._msg(0xC0))
    self.assertTrue(self.safety.get_ignition_can())
    self.safety.ignition_can_hook(self._msg(0x20))
    self.assertFalse(self.safety.get_ignition_can())


class TestMazdaStockSteeringSafety(TestMazdaSafety):
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
                                 int(MazdaSafetyFlags.STEER_TO_ZERO | MazdaSafetyFlags.TJA_MADS))
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
