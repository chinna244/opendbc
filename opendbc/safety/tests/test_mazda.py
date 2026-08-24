#!/usr/bin/env python3
import unittest

from opendbc.car.mazda.values import MazdaSafetyFlags
from opendbc.car.structs import CarParams
from opendbc.safety.tests.libsafety import libsafety_py
import opendbc.safety.tests.common as common
from opendbc.safety.tests.common import CANPackerSafety, make_msg


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

  def setUp(self):
    self.packer = CANPackerSafety("mazda_2017")
    self.safety = libsafety_py.libsafety
    self.safety.set_safety_hooks(CarParams.SafetyModel.mazda, 0)
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

  def _button_msg(self, resume=False, cancel=False):
    values = {
      "CAN_OFF": cancel,
      "CAN_OFF_INV": (cancel + 1) % 2,
      "RES": resume,
      "RES_INV": (resume + 1) % 2,
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


class TestMazdaLongitudinalSafety(TestMazdaSafety, common.LongitudinalAccelSafetyTest):

  TX_MSGS = [[0x243, 0], [0x09d, 0], [0x440, 0], [0x21b, 0], [0x21c, 0], [0x499, 0],
             [0x361, 0], [0x362, 0], [0x363, 0], [0x364, 0], [0x365, 0], [0x366, 0], [0x764, 0],
             [0x21b, 2], [0x21c, 2], [0x499, 2], [0x361, 2], [0x362, 2], [0x363, 2], [0x364, 2], [0x365, 2], [0x366, 2]]

  def setUp(self):
    self.packer = CANPackerSafety("mazda_2017")
    self.safety = libsafety_py.libsafety
    self.safety.set_safety_hooks(CarParams.SafetyModel.mazda, MazdaSafetyFlags.LONG)
    self.safety.init_tests()

  def _pcm_status_msg(self, enable):
    values = {"ACC_ACTIVE": enable, "BRAKE_ON": 0}
    return self.packer.make_can_msg_safety("PEDALS", 0, values)

  def _accel_msg(self, accel: float, bus: int = 0):
    values = {"ACCEL_CMD": accel}
    return self.packer.make_can_msg_safety("CRZ_INFO", bus, values)

  def _crz_ctrl_cmd_msg(self, active: bool, bus: int = 0):
    values = {"CRZ_ACTIVE": active}
    return self.packer.make_can_msg_safety("CRZ_CTRL", bus, values)

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

  def test_synthetic_lead_radar_track_gated_on_controls(self):
    # DIST_OBJ and RELV_OBJ are free fields; the template bytes must match. The non-template
    # frames are real on-road emissions (route 6bb2dc61c4), which a byte-exact check silently
    # dropped -- 982 asked, 0 transmitted -- starving the camera of the track.
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
        self.safety.set_controls_allowed(False)
        self.assertFalse(self._tx(common.make_msg(bus, 0x364, 8, dat)))
        self.safety.set_controls_allowed(True)
        self.assertTrue(self._tx(common.make_msg(bus, 0x364, 8, dat)))

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


class TestMazdaTjaMadsSafety(TestMazdaSafety):
  SAFETY_PARAM = MazdaSafetyFlags.TJA_MADS

  def setUp(self):
    self.packer = CANPackerSafety("mazda_2017")
    self.safety = libsafety_py.libsafety
    self.safety.set_safety_hooks(CarParams.SafetyModel.mazda, int(self.SAFETY_PARAM))
    self.safety.init_tests()

  def _tja_msg(self, pressed):
    return self.packer.make_can_msg_safety("CRZ_BTNS", 0, {
      "TJA_BUTTON": pressed, "BIT1": 1, "BIT2": 1, "BIT3": 1,
    })

  @staticmethod
  def _packet_bytes(msg):
    return bytes(msg[0].data[0:8])

  def test_tja_is_the_only_lateral_request_source(self):
    self.safety.set_mads_params(True, False, False)
    self._rx(self._pcm_status_msg(True))
    self.assertTrue(self.safety.get_controls_allowed())
    self.assertFalse(self.safety.get_controls_allowed_lateral())

    self._rx(self._tja_msg(True))
    self.assertTrue(self.safety.get_controls_allowed_lateral())
    self._rx(self._tja_msg(False))
    self._rx(self._pcm_status_msg(False))
    self.assertFalse(self.safety.get_controls_allowed())
    self.assertTrue(self.safety.get_controls_allowed_lateral())

  def test_forward_copy_strips_only_tja_and_preserves_original(self):
    self.safety.set_mads_params(True, False, False)
    msg = self.packer.make_can_msg_safety("CRZ_BTNS", 0, {
      "TJA_BUTTON": 1, "SET_P": 1, "RES": 1, "MODE_X": 1,
      "BIT1": 1, "BIT2": 1, "BIT3": 1,
    })
    original = self._packet_bytes(msg)
    forwarded = libsafety_py.make_CANPacket(0x09d, 0, original)
    self.safety.safety_fwd_modify(0, forwarded)
    forwarded_bytes = self._packet_bytes(forwarded)

    expected = bytearray(original)
    expected[1] &= ~0x08
    self.assertEqual(bytes(expected), forwarded_bytes)
    self.assertEqual(original, self._packet_bytes(msg))

    self._rx(msg)
    self.assertEqual(1, self.safety.get_mads_button_press())
    self.assertTrue(self.safety.get_controls_allowed_lateral())

  def test_forward_mutation_is_scoped_by_bus_address_and_feature(self):
    original = self._packet_bytes(self._tja_msg(True))
    for enabled, bus, addr, should_strip in (
      (False, 0, 0x09d, False),
      (True, 2, 0x09d, False),
      (True, 0, 0x21c, False),
      (True, 0, 0x09d, True),
    ):
      self.safety.set_mads_params(enabled, False, False)
      forwarded = libsafety_py.make_CANPacket(addr, bus, original)
      self.safety.safety_fwd_modify(bus, forwarded)
      self.assertEqual(should_strip, self._packet_bytes(forwarded) != original)

  def test_forward_mutation_preserves_every_non_tja_bit(self):
    self.safety.set_mads_params(True, False, False)
    for seed in range(32):
      original = bytes((seed * 17 + index * 29) & 0xff for index in range(8))
      forwarded = libsafety_py.make_CANPacket(0x09d, 0, original)
      self.safety.safety_fwd_modify(0, forwarded)
      expected = bytearray(original)
      expected[1] &= ~0x08
      self.assertEqual(bytes(expected), self._packet_bytes(forwarded))

  def test_non_tja_mode_keeps_existing_lateral_source(self):
    self.safety.set_safety_hooks(CarParams.SafetyModel.mazda, 0)
    self.safety.init_tests()
    self.safety.set_mads_params(True, False, False)
    self.assertTrue(self.safety.get_op_controls_allowed_requests_lateral())
    self._rx(self._pcm_status_msg(True))
    self.assertTrue(self.safety.get_controls_allowed_lateral())

  def test_non_tja_wheel_buttons_do_not_request_lateral(self):
    self.safety.set_mads_params(True, False, False)
    for values in (
      {"SET_P": 1, "SET_P_INV": 0},
      {"SET_M": 1, "SET_M_INV": 0},
      {"RES": 1, "RES_INV": 0},
      {"CAN_OFF": 1, "CAN_OFF_INV": 0},
      {"MODE_X": 1, "MODE_X_INV": 0, "MODE_Y": 1, "MODE_Y_INV": 0},
    ):
      message = self.packer.make_can_msg_safety("CRZ_BTNS", 0, {
        **values, "TJA_BUTTON": 0, "BIT1": 1, "BIT2": 1, "BIT3": 1,
      })
      self._rx(message)
      self.assertFalse(self.safety.get_controls_allowed_lateral())

  def test_mrcc_falling_does_not_revoke_tja_lateral(self):
    self.safety.set_mads_params(True, False, False)
    self._rx(self._tja_msg(True))
    self._rx(self._tja_msg(False))
    self.assertTrue(self.safety.get_controls_allowed_lateral())

    self._rx(self.packer.make_can_msg_safety("CRZ_CTRL", 0, {"CRZ_AVAILABLE": 1}))
    self._rx(self.packer.make_can_msg_safety("CRZ_CTRL", 0, {"CRZ_AVAILABLE": 0}))
    self.assertTrue(self.safety.get_controls_allowed_lateral())


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


if __name__ == "__main__":
  unittest.main()
