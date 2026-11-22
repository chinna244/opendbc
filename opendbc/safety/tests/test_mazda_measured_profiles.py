#!/usr/bin/env python3
"""The shipped Mazda measured-profile words (2, 3, 10, 11) through StarPilot's standard safety suite.

Word 0 stays StarPilot's base path (modes/mazda.h + mazda_stock_aol.h) and is proven by test_mazda.py.
These classes port the pinned source suite (sunnypilot opendbc 7b96732d, opendbc/safety/tests/test_mazda.py)
onto the port's C (modes/mazda_port.h), without dropping its assertions.

MADS -> StarPilot AOL mapping used throughout (deliberate; every row is an API, not a behavior, change):
  source                                      port
  set_mads_params(True, ..) (MADS enabled)    alternative_experience 32: mazda_aol_enabled (mazda_port.h:386)
  set_mads_params(False, ..)                  alternative_experience 0: lateral follows controls_allowed (safety.h:107-114)
  controls_allowed_lateral                    lateral_controls_allowed() (mazda_port.h:229-230): AOL permission bit 0
  set_controls_allowed_lateral(True)          a real physical grant (main edge, ON/SET/RES/CANCEL, TJA, accepted cruise
                                              edge) plus a current host request, aol_set_host_request(1)
  set_controls_allowed_lateral(False)         host request 0 (or no grant)
  get_mads_button_press()                     no counterpart; asserted through its effect, the lateral permission
  set_heartbeat_engaged_mads(False) + check   set_aol_test_heartbeat(False)
  set_current_safety_param_sp(TJA_BUTTON)     safety param bit 8, words 10/11 (mazda_port.h:469)
  safety_fwd_modify                           safety_fwd_modify_hook
  word 0 + sp TJA (stock long)                word 10;  word 1 (LONG alone) -> word 3 (only 2/3/10/11 are admitted,
                                              mazda_port.h:381-383)

Where StarPilot AOL intentionally differs from MADS, the test asserts StarPilot's behavior and names the code.
"""
import unittest

from opendbc.car.mazda.values import CarControllerParams, MazdaSafetyFlags
from opendbc.car.structs import CarParams
from opendbc.safety.tests.libsafety import libsafety_py
import opendbc.safety.tests.common as common
from opendbc.safety.tests.common import CANPackerSafety, make_msg

# both-low PEDALS samples before main falls, as carstate counts them
DEBOUNCE = CarControllerParams.MAIN_OFF_DEBOUNCE_SAMPLES
AOL = 32
EPS = int(MazdaSafetyFlags.EPS_HW)
LONG = int(MazdaSafetyFlags.LONG | MazdaSafetyFlags.EPS_HW)
TJA = int(MazdaSafetyFlags.TJA_BUTTON | MazdaSafetyFlags.EPS_HW)
TJA_LONG = int(MazdaSafetyFlags.TJA_BUTTON | MazdaSafetyFlags.LONG | MazdaSafetyFlags.EPS_HW)
WORDS = (EPS, LONG, TJA, TJA_LONG)

LAT_TX = [[0x243, 0], [0x09d, 0], [0x440, 0]]
LONG_TX = LAT_TX + [[0x21b, 0], [0x21c, 0], [0x499, 0], [0x361, 0], [0x362, 0], [0x363, 0], [0x364, 0], [0x365, 0],
                    [0x366, 0], [0x764, 0], [0x21b, 2], [0x21c, 2], [0x499, 2], [0x361, 2], [0x362, 2], [0x363, 2],
                    [0x364, 2], [0x365, 2], [0x366, 2]]
RADAR_CAPTURES = {
  # 2022 family static frame as the real radar sends it (develop 34ca73f8, for the dash's lane lines)
  0x499: bytes.fromhex("0008e00000000000"),
  0x361: bytes.fromhex("fff7fefe1fc00080"),
  0x362: bytes.fromhex("fff7fefe1fc78c80"),
  0x363: bytes.fromhex("fff7fefe1fc00000"),
  0x364: bytes.fromhex("fff7fefe1fc00000"),
  0x365: bytes.fromhex("fff7fe7ffbff3fc0"),
  0x366: bytes.fromhex("fff7fe7ffbff3fc0"),
}


def _is_long(word):
  return bool(word & MazdaSafetyFlags.LONG)


def _is_tja(word):
  return bool(word & MazdaSafetyFlags.TJA_BUTTON)


class MazdaMeasuredBase(common.CarSafetyTest, common.DriverTorqueSteeringSafetyTest):
  """Source TestMazdaEpsSafety on the port: the measured envelope, alternative_experience 0 (MADS off)."""

  DBC = "mazda_2017"
  SAFETY_MODEL = CarParams.SafetyModel.mazda
  TX_MSGS = LAT_TX
  STANDSTILL_THRESHOLD = .1
  RELAY_MALFUNCTION_ADDRS = {0: (0x243, 0x440)}
  # camera 0x243/0x440 frames forward while openpilot is not controlling (lateral is off at setUp)
  FWD_BLACKLISTED_ADDRS = {2: []}

  MAX_RATE_UP = CarControllerParams.STEER_DELTA_UP
  MAX_RATE_DOWN = CarControllerParams.STEER_DELTA_DOWN
  MAX_TORQUE_LOOKUP = [0], [CarControllerParams.EPS_STEER_MAX]
  MAX_RT_DELTA = 320
  DRIVER_TORQUE_ALLOWANCE = CarControllerParams.STEER_DRIVER_ALLOWANCE
  DRIVER_TORQUE_FACTOR = CarControllerParams.STEER_DRIVER_MULTIPLIER
  NO_STEER_REQ_BIT = True

  def setUp(self):
    libsafety_py.libsafety.set_alternative_experience(0)
    super().setUp()

  def tearDown(self):
    self.safety.set_alternative_experience(0)
    self.safety.set_aol_test_heartbeat(False)

  def _init(self, word=None, experience=0):
    self.safety.set_alternative_experience(experience)
    self.safety.set_safety_hooks(self.SAFETY_MODEL, self.SAFETY_PARAM if word is None else word)
    self.safety.init_tests()

  def _torque_meas_msg(self, torque):
    return self.packer.make_can_msg_safety("STEER_TORQUE", 0, {"STEER_TORQUE_MOTOR": torque})

  def _torque_driver_msg(self, torque):
    return self.packer.make_can_msg_safety("STEER_TORQUE", 0, {"STEER_TORQUE_SENSOR": torque})

  def _torque_cmd_msg(self, torque, steer_req=1, bus=0):
    return self.packer.make_can_msg_safety("CAM_LKAS", bus, {"LKAS_REQUEST": torque})

  def _laneinfo_msg(self):
    return self.packer.make_can_msg_safety("CAM_LANEINFO", 0, {"LINE_VISIBLE": 0})

  def _speed_msg(self, speed):
    return self.packer.make_can_msg_safety("ENGINE_DATA", 0, {"SPEED": speed})

  def _user_brake_msg(self, brake):
    return self.packer.make_can_msg_safety("PEDALS", 0, {"BRAKE_ON": brake})

  def _user_gas_msg(self, gas):
    return self.packer.make_can_msg_safety("ENGINE_DATA", 0, {"PEDAL_GAS": gas})

  def _pcm_status_msg(self, enable):
    return self.packer.make_can_msg_safety("CRZ_CTRL", 0, {"CRZ_ACTIVE": enable})

  def _button_msg(self, resume=False, cancel=False, set_m=False, set_p=False, tja=False, bus=0):
    values = {
      "TJA_BUTTON": tja,
      "CAN_OFF": cancel, "CAN_OFF_INV": (cancel + 1) % 2,
      "RES": resume, "RES_INV": (resume + 1) % 2,
      "SET_M": set_m, "SET_M_INV": (set_m + 1) % 2,
      "SET_P": set_p, "SET_P_INV": (set_p + 1) % 2,
    }
    return self.packer.make_can_msg_safety("CRZ_BTNS", bus, values)

  def _cam_tja_press(self, ctr=5):
    # the wheel's idle pattern with the TJA bit: 00 09 ff Cx 00 00 00 00
    values = {"TJA_BUTTON": 1, "DISTANCE_LESS_INV": 1, "BIT1": 1, "BIT2": 1, "BIT3": 1, "CAN_OFF_INV": 1, "RES_INV": 1,
              "SET_P_INV": 1, "SET_M_INV": 1, "DISTANCE_MORE_INV": 1, "MODE_X_INV": 1, "MODE_Y_INV": 1, "CTR": ctr}
    return self.packer.make_can_msg_safety("CRZ_BTNS", 2, values)

  @staticmethod
  def _truncate(msg, length=7):
    return libsafety_py.make_CANPacket(int(msg[0].addr), int(msg[0].bus), bytes(msg[0].data)[:length])

  # ***** source TestMazdaSafety / TestMazdaEpsSafety *****

  def test_buttons(self):
    self.safety.set_controls_allowed(0)
    self.assertTrue(self._tx(self._button_msg(cancel=True)))
    self.assertFalse(self._tx(self._button_msg(resume=True)))
    self.safety.set_controls_allowed(1)
    self.assertTrue(self._tx(self._button_msg(cancel=True)))
    self.assertTrue(self._tx(self._button_msg(resume=True)))

  def test_steer_safety_check(self):
    # the common test, except that disengaged the camera owns 0x243 (test_stock_passthrough), so the
    # zero-torque frame upstream's rule lets through is vetoed too
    for speed in self._torque_speed_range:
      self._reset_speed_measurement(speed)
      max_torque = self._get_max_torque(speed)
      for enabled in [0, 1]:
        for t in range(int(-max_torque * 1.5), int(max_torque * 1.5)):
          self.safety.set_controls_allowed(enabled)
          self._set_prev_torque(t)
          self.assertEqual(bool(enabled) and abs(t) <= max_torque, self._tx(self._torque_cmd_msg(t)))

  def test_measured_envelope_boundaries(self):
    # absolute 1200, rate up 10, driver winddown 10, rt 320, driver allowance 15 x multiplier 15, at each boundary
    lim = CarControllerParams
    self.safety.set_controls_allowed(True)
    for sign in (1, -1):
      self._reset_torque_driver_measurement(0)
      self._set_prev_torque(sign * lim.EPS_STEER_MAX)
      self.assertTrue(self._tx(self._torque_cmd_msg(sign * lim.EPS_STEER_MAX)))
      self.assertFalse(self._tx(self._torque_cmd_msg(sign * (lim.EPS_STEER_MAX + 1))))
      self._set_prev_torque(0)
      self.assertTrue(self._tx(self._torque_cmd_msg(sign * lim.STEER_DELTA_UP)))
      self._set_prev_torque(0)
      self.assertFalse(self._tx(self._torque_cmd_msg(sign * (lim.STEER_DELTA_UP + 1))))
      # max_rate_down is the slowest winddown a heavy opposing driver may force: at least 10 per frame
      self._reset_torque_driver_measurement(-sign * 100)
      self._set_prev_torque(sign * 500)
      self.assertTrue(self._tx(self._torque_cmd_msg(sign * (500 - lim.STEER_DELTA_DOWN))))
      self._set_prev_torque(sign * 500)
      self.assertFalse(self._tx(self._torque_cmd_msg(sign * (500 - lim.STEER_DELTA_DOWN + 1))))
    # rt: 320 counts from the rt reference inside the 250 ms window, 321 is refused
    self._reset_torque_driver_measurement(0)
    self._set_prev_torque(0)
    for t in range(lim.STEER_DELTA_UP, self.MAX_RT_DELTA + 1, lim.STEER_DELTA_UP):
      self.assertTrue(self._tx(self._torque_cmd_msg(t)))
    self.safety.set_desired_torque_last(self.MAX_RT_DELTA)
    self.assertFalse(self._tx(self._torque_cmd_msg(self.MAX_RT_DELTA + 1)))
    # driver limit: opposing driver torque past the allowance pulls the ceiling in by 15 per count
    driver = lim.STEER_DRIVER_ALLOWANCE + 10
    self._reset_torque_driver_measurement(-driver)
    ceiling = lim.EPS_STEER_MAX - (driver - lim.STEER_DRIVER_ALLOWANCE) * lim.STEER_DRIVER_MULTIPLIER
    self._set_prev_torque(ceiling)
    self.assertTrue(self._tx(self._torque_cmd_msg(ceiling)))
    self._set_prev_torque(ceiling + 1)
    self.assertFalse(self._tx(self._torque_cmd_msg(ceiling + 1)))

  def test_lkas_wrong_bus_or_length_blocked(self):
    self.safety.set_controls_allowed(True)
    self._set_prev_torque(0)
    self.assertTrue(self._tx(self._torque_cmd_msg(5)))
    for bus in (1, 2, 3):
      self.assertFalse(self._tx(self._torque_cmd_msg(5, bus=bus)), bus)
    self.assertFalse(self._tx(self._truncate(self._torque_cmd_msg(5))))
    self.assertFalse(self._tx(self._truncate(self._button_msg(cancel=True))))
    self.assertFalse(self._tx(make_msg(0, 0x440, 7)))

  def test_stock_passthrough(self):
    # MADS off (alternative_experience 0): lateral follows cruise, so the camera owns 0x243/0x440 exactly
    # while controls are not allowed. The MADS-on half of the source test is TestMazdaMeasuredAol.
    for controls_allowed in (False, True):
      self.safety.set_controls_allowed(controls_allowed)
      for addr, msg in ((0x243, self._torque_cmd_msg(0)), (0x440, self._laneinfo_msg())):
        self.assertEqual(-1 if controls_allowed else 0, self.safety.safety_fwd_hook(2, addr))
        self.assertEqual(controls_allowed, self._tx(msg), f"{controls_allowed=} {addr=:#x}")

  def test_no_button_reaches_the_camera(self):
    # Without the declared TJA button openpilot never writes CRZ_BTNS on the camera bus. With it, only the exact
    # camera-only TJA press-off while openpilot steers (develop 0f6ce31f; pacing in test_mazda_camera_tja.py).
    tja = _is_tja(self.SAFETY_PARAM)
    for experience in (0, AOL):
      for controls_allowed in (False, True):
        for ctr in range(16):
          self._init(experience=experience)
          self.safety.set_controls_allowed(controls_allowed)
          steering = bool(self.safety.get_controls_allowed()) and experience == 0
          self.assertEqual(tja and steering, self._tx(self._cam_tja_press(ctr=ctr)), f"{experience=} {controls_allowed=} {ctr=}")
        self.assertFalse(self._tx(make_msg(2, 0x09d, 8)))

  def test_tja_button_never_pressed_on_the_car_side(self):
    for controls_allowed in (False, True):
      self.safety.set_controls_allowed(controls_allowed)
      self.assertFalse(self._tx(self._button_msg(tja=True)))
      self.assertFalse(self._tx(self._button_msg(tja=True, resume=True)))
      self.assertFalse(self._tx(self._button_msg(tja=True, cancel=True)))
      self.assertEqual(controls_allowed, self._tx(self._button_msg(resume=True)))

  def test_upstream_envelope_without_the_bit(self):
    # Source: 0 and 4 give upstream's stricter 800 envelope. Port: 0 is StarPilot's base envelope (800);
    # 4 is not an admitted word and gets the zeroed config (mazda_port.h:381-383), stricter still.
    for param in (0, 4):
      with self.subTest(param=param):
        self._init(word=param)
        self.safety.set_controls_allowed(True)
        self._reset_torque_driver_measurement(0)
        self._set_prev_torque(0)
        self.assertFalse(self._tx(self._torque_cmd_msg(11)))
        self._set_prev_torque(800)
        self.assertFalse(self._tx(self._torque_cmd_msg(801)))

  # ***** RX: wrong bus / length are ignored, stale required inputs drop controls *****

  def test_rx_wrong_bus_or_length_ignored(self):
    # a brake press on the wrong bus or with the wrong length never reaches the rx hook (safety.h:267-274)
    for bad in (self.packer.make_can_msg_safety("PEDALS", 1, {"BRAKE_ON": 1}),
                self.packer.make_can_msg_safety("PEDALS", 2, {"BRAKE_ON": 1}),
                self._truncate(self._user_brake_msg(1))):
      self._rx(self._user_brake_msg(0))
      self.safety.set_controls_allowed(True)
      self._rx(bad)
      self.assertTrue(self.safety.get_controls_allowed())
      self.assertFalse(self.safety.get_brake_pressed_prev())
    # nor can a speed frame mark the car moving
    for bad in (self.packer.make_can_msg_safety("ENGINE_DATA", 2, {"SPEED": 50}), self._truncate(self._speed_msg(50))):
      self._rx(bad)
      self.assertFalse(self.safety.get_vehicle_moving())

  def _required(self):
    msgs = {
      "CRZ_BTNS": self._button_msg(),
      "STEER_TORQUE": self._torque_driver_msg(0),
      "ENGINE_DATA": self._speed_msg(0),
    }
    if _is_long(self.SAFETY_PARAM):
      msgs["PEDALS"] = self.packer.make_can_msg_safety("PEDALS", 0, {"ACC_ACTIVE": 1})
      msgs["CRZ_CTRL"] = self.packer.make_can_msg_safety("CRZ_CTRL", 0, {"CRZ_ACTIVE": 0})
    else:
      msgs["PEDALS"] = self._user_brake_msg(0)
      msgs["CRZ_CTRL"] = self.packer.make_can_msg_safety("CRZ_CTRL", 0, {"CRZ_ACTIVE": 1})
    return msgs

  def test_stale_required_input_drops_controls(self):
    # Each rx-checked input lagging past max(10 frames, 1 s) disables controls (safety.h:394-414). Under
    # LONG the radar's CRZ_CTRL is intentionally not required (mazda_port.h:474): it disappears at teardown.
    for missing in ("CRZ_CTRL", "CRZ_BTNS", "STEER_TORQUE", "ENGINE_DATA", "PEDALS"):
      with self.subTest(missing=missing):
        self._init()
        t = 1_000_000
        self.safety.set_timer(t)
        for msg in self._required().values():
          self._rx(msg)
        self.safety.set_controls_allowed(True)
        self.safety.safety_tick()
        self.assertTrue(self.safety.safety_config_valid())
        self.assertTrue(self.safety.get_controls_allowed())
        while t < 2_200_000:
          t += 10_000
          self.safety.set_timer(t)
          for name, msg in self._required().items():
            if name != missing:
              self._rx(msg)
        self.safety.safety_tick()
        required = not (missing == "CRZ_CTRL" and _is_long(self.SAFETY_PARAM))
        self.assertEqual(not required, self.safety.get_controls_allowed())
        self.assertEqual(not required, self.safety.safety_config_valid())


class MazdaMeasuredLateralBase(MazdaMeasuredBase):
  def test_diagnostics_need_long(self):
    # 0x764 is not in the lateral-only tables, so no diagnostic of any shape leaves on any bus
    for controls_allowed in (False, True):
      self.safety.set_controls_allowed(controls_allowed)
      for bus in range(4):
        for dat in ("023e800000000000", "0210010000000000", "0210020000000000"):
          self.assertFalse(self._tx(make_msg(bus, 0x764, 8, bytes.fromhex(dat))))


class MazdaMeasuredLongBase(MazdaMeasuredBase, common.LongitudinalAccelSafetyTest):
  """Source TestMazdaLongitudinalSafety on the port (LONG always travels with EPS_HW)."""

  TX_MSGS = LONG_TX

  def _pcm_status_msg(self, enable):
    return self.packer.make_can_msg_safety("PEDALS", 0, {"ACC_ACTIVE": enable, "BRAKE_ON": 0})

  def _accel_msg(self, accel: float, bus: int = 0, active: bool = False):
    return self.packer.make_can_msg_safety("CRZ_INFO", bus, {"ACCEL_CMD": accel, "ACC_ACTIVE": active})

  def _crz_ctrl_cmd_msg(self, active: bool, bus: int = 0):
    return self.packer.make_can_msg_safety("CRZ_CTRL", bus, {"CRZ_ACTIVE": active})

  def _pedals_msg(self, armed, brake=0):
    return self.packer.make_can_msg_safety("PEDALS", 0, {"ACC_OFF": armed, "BRAKE_ON": brake})

  def _press_set(self):
    # the wheel press lands 30-70 ms before PEDALS.ACC_ACTIVE rises in every logged engagement
    self._rx(self._button_msg(set_m=True))

  def _main_off(self, samples, brake):
    for _ in range(samples):
      self._rx(self._pedals_msg(armed=False, brake=brake))

  def test_enable_control_allowed_from_cruise(self):
    self._press_set()
    super().test_enable_control_allowed_from_cruise()

  def test_cruise_without_button_never_arms(self):
    self._rx(self._pcm_status_msg(False))
    for _ in range(12):
      self._rx(self._pcm_status_msg(True))
      self.assertFalse(self.safety.get_controls_allowed())

  def test_button_window_expires(self):
    self._press_set()
    for _ in range(12):
      self._rx(self._button_msg())
    self._rx(self._pcm_status_msg(True))
    self.assertFalse(self.safety.get_controls_allowed())

  def test_button_window_boundary(self):
    # MAZDA_ENGAGE_BTN_WINDOW = 10: the press frame plus 9 idle frames still qualify, the 10th idle does not
    for idle, qualifies in ((9, True), (10, False)):
      self._init()
      self._rx(self._pcm_status_msg(False))
      self._press_set()
      for _ in range(idle):
        self._rx(self._button_msg())
      self._rx(self._pcm_status_msg(True))
      self.assertEqual(qualifies, self.safety.get_controls_allowed(), idle)

  def test_engage_press_on_wrong_bus_or_length_never_qualifies(self):
    for press in (self._button_msg(set_m=True, bus=2), self._button_msg(set_m=True, bus=1),
                  self._truncate(self._button_msg(set_m=True))):
      self._init()
      self._rx(self._pcm_status_msg(False))
      self._rx(press)
      self._rx(self._pcm_status_msg(True))
      self.assertFalse(self.safety.get_controls_allowed())
    # and ACC_ACTIVE on the camera bus is not the body's answer
    self._init()
    self._press_set()
    self._rx(self.packer.make_can_msg_safety("PEDALS", 2, {"ACC_ACTIVE": 1}))
    self.assertFalse(self.safety.get_controls_allowed())

  def test_armed_controls_latch_past_the_window(self):
    self._press_set()
    self._rx(self._pcm_status_msg(True))
    self.assertTrue(self.safety.get_controls_allowed())
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

  def test_cancel_button_exits_controls(self):
    self._press_set()
    self._rx(self._pcm_status_msg(True))
    self.assertTrue(self.safety.get_controls_allowed())
    self._rx(self._button_msg(cancel=True))
    self.assertFalse(self.safety.get_controls_allowed())
    self._rx(self._pcm_status_msg(True))
    self.assertFalse(self.safety.get_controls_allowed())

  def test_cancel_exits_controls_at_once(self):
    self._rx(self._pedals_msg(True))
    self.safety.set_controls_allowed(True)
    self._rx(self._button_msg(cancel=True))
    self.assertFalse(self.safety.get_controls_allowed())
    # undeclared: PEDALS armed is main; declared TJA: MRCC never drives main (mazda_port.h:156-157)
    self.assertEqual(not _is_tja(self.SAFETY_PARAM), self.safety.get_acc_main_on())

  def test_brake_exits_controls_while_moving(self):
    self._rx(self._speed_msg(20))
    self._press_set()
    self._rx(self._pcm_status_msg(True))
    self.assertTrue(self.safety.get_controls_allowed())
    self._rx(self.packer.make_can_msg_safety("PEDALS", 0, {"ACC_ACTIVE": 1, "BRAKE_ON": 1}))
    self.assertFalse(self.safety.get_controls_allowed())
    self.assertFalse(self._tx(self._accel_msg(-1.0, active=True)))
    self.assertFalse(self._tx(self._crz_ctrl_cmd_msg(True)))

  def test_camera_bus_accel_actuation_limits(self):
    for accel in (self.MIN_ACCEL - 1, self.MIN_ACCEL, self.INACTIVE_ACCEL, self.MAX_ACCEL, self.MAX_ACCEL + 1):
      for controls_allowed in (True, False):
        self.safety.set_controls_allowed(controls_allowed)
        should_tx = (controls_allowed and self.MIN_ACCEL <= accel <= self.MAX_ACCEL) or accel == self.INACTIVE_ACCEL
        self.assertEqual(should_tx, self._tx(self._accel_msg(accel, bus=2)))

  def test_accel_raw_boundaries(self):
    # MAZDA_LONG_LIMITS in raw 1/1024 m/s2 units: 2048 and -3584 pass, one count beyond fails
    lsb = CarControllerParams.ACCEL_CMD_LSB
    self.safety.set_controls_allowed(True)
    for bus in (0, 2):
      for raw, ok in ((2048, True), (2049, False), (-3584, True), (-3585, False)):
        self.assertEqual(ok, self._tx(self._accel_msg(raw * lsb, bus=bus, active=True)), (bus, raw))

  def test_long_wrong_bus_or_length_blocked(self):
    self.safety.set_controls_allowed(True)
    for msg in (self._accel_msg(0.5, bus=1), self._crz_ctrl_cmd_msg(True, bus=1), make_msg(1, 0x499, 8, RADAR_CAPTURES[0x499]),
                self._truncate(self._accel_msg(0.5)), self._truncate(self._crz_ctrl_cmd_msg(False)),
                make_msg(0, 0x361, 7, RADAR_CAPTURES[0x361][:7])):
      self.assertFalse(self._tx(msg))

  def test_stock_crz_info_standby_allowed(self):
    def pegged_frame(d4, d5, counter):
      dat = bytes([0x01, 0xff, 0xe3, 0xff, d4, d5, counter])
      return dat + bytes([(0xff - sum(dat)) & 0xff])

    for controls_allowed in (False, True):
      self.safety.set_controls_allowed(controls_allowed)
      for bus in (0, 2):
        for d4, d5 in ((0xc0, 0x00), (0xc0, 0x80), (0xc4, 0x80)):
          for counter in range(16):
            self.assertTrue(self._tx(make_msg(bus, 0x21b, 8, pegged_frame(d4, d5, counter))))
        self.assertFalse(self._tx(make_msg(bus, 0x21b, 8, bytes.fromhex("01ffe3ffc0000000"))))
        self.assertFalse(self._tx(make_msg(bus, 0x21b, 8, pegged_frame(0xc6, 0x80, 0x00))))
        self.assertFalse(self._tx(make_msg(bus, 0x21b, 8, pegged_frame(0xc0, 0x84, 0x00))))

  def test_empty_radar_tracks_allowed(self):
    for controls_allowed in (False, True):
      self.safety.set_controls_allowed(controls_allowed)
      for bus in (0, 2):
        for addr, dat in RADAR_CAPTURES.items():
          self.assertTrue(self._tx(make_msg(bus, addr, 8, dat)))
    self.safety.set_controls_allowed(True)
    for addr, dat in RADAR_CAPTURES.items():
      for counter in range(1, 16):
        counted = dat[:7] + bytes([dat[7] | counter])
        self.assertEqual(addr != 0x499, self._tx(make_msg(0, addr, 8, counted)), f"{addr=:#x} {counter=}")

  def test_g46l_radar_static_allowed(self):
    for controls_allowed in (False, True):
      self.safety.set_controls_allowed(controls_allowed)
      for bus in (0, 2):
        self.assertTrue(self._tx(make_msg(bus, 0x499, 8, bytes.fromhex("0098400000000000"))))
    self.safety.set_controls_allowed(True)
    for bus in (0, 2):
      self.assertFalse(self._tx(make_msg(bus, 0x499, 8, bytes.fromhex("0098400100000000"))))

  def test_unexpected_radar_tracks_blocked(self):
    bad_messages = {
      0x499: bytes.fromhex("0008e00100000000"),
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
        self.assertFalse(self._tx(make_msg(bus, addr, 8, dat)))
    for controls_allowed in (False, True):
      self.safety.set_controls_allowed(controls_allowed)
      for bus in (0, 2):
        for addr in range(0x361, 0x367):
          for hexdat in ("0a4000001c000000", "229e00007c00000e"):
            self.assertFalse(self._tx(make_msg(bus, addr, 8, bytes.fromhex(hexdat))))

  def test_radar_uds_allowlist(self):
    self.assertTrue(self._tx(make_msg(0, 0x764, 8, bytes.fromhex("023e800000000000"))))
    self.assertTrue(self._tx(make_msg(0, 0x764, 8, bytes.fromhex("0210020000000000"))))
    self.assertFalse(self._tx(make_msg(0, 0x764, 8, bytes.fromhex("0210030000000000"))))
    self.assertFalse(self._tx(make_msg(0, 0x764, 8, bytes.fromhex("0227010000000000"))))
    self.assertFalse(self._tx(make_msg(2, 0x764, 8, bytes.fromhex("023e800000000000"))))

  def test_crz_ctrl_active_gated_on_controls(self):
    for bus in (0, 2):
      self.safety.set_controls_allowed(False)
      self.assertFalse(self._tx(self._crz_ctrl_cmd_msg(True, bus)))
      self.assertTrue(self._tx(self._crz_ctrl_cmd_msg(False, bus)))
      self.safety.set_controls_allowed(True)
      self.assertTrue(self._tx(self._crz_ctrl_cmd_msg(True, bus)))

  def test_crz_info_active_gated_on_controls(self):
    for bus in (0, 2):
      for active in (False, True):
        msg = self._accel_msg(self.INACTIVE_ACCEL, bus=bus, active=active)
        self.safety.set_controls_allowed(False)
        self.assertEqual(not active, self._tx(msg))
        self.safety.set_controls_allowed(True)
        self.assertTrue(self._tx(msg))

  def test_gas_blocks_accel_not_lateral(self):
    self.safety.set_controls_allowed(True)
    self._rx(self._user_gas_msg(10))
    self._rx(self._user_gas_msg(10))
    self.assertFalse(self._tx(self._accel_msg(1.0, active=True)))
    self.assertTrue(self._tx(self._accel_msg(0.0, active=True)))
    self._set_prev_torque(0)
    self.assertTrue(self._tx(self._torque_cmd_msg(5)))


class TestMazdaMeasuredEps(MazdaMeasuredLateralBase):
  SAFETY_PARAM = EPS

  def test_main_follows_mrcc(self):
    for main in (True, False, True):
      self._rx(self.packer.make_can_msg_safety("CRZ_CTRL", 0, {"CRZ_AVAILABLE": main}))
      self.assertEqual(main, self.safety.get_acc_main_on())


class TestMazdaMeasuredTja(MazdaMeasuredLateralBase):
  SAFETY_PARAM = TJA

  def test_declared_button_mrcc_does_not_drive_the_main_edge(self):
    # source TestMazdaTjaMads: declared, MRCC never drives main (mazda_port.h:120)
    self._rx(self.packer.make_can_msg_safety("CRZ_CTRL", 0, {"CRZ_AVAILABLE": 1}))
    self.assertFalse(self.safety.get_acc_main_on())

  def test_declaration_is_read_at_init(self):
    for word, main in ((TJA, False), (EPS, True)):
      self._init(word=word)
      self._rx(self.packer.make_can_msg_safety("CRZ_CTRL", 0, {"CRZ_AVAILABLE": 1}))
      self.assertEqual(main, self.safety.get_acc_main_on())


class TestMazdaMeasuredLong(MazdaMeasuredLongBase):
  SAFETY_PARAM = LONG

  def _armed(self):
    self._rx(self._pedals_msg(True))
    self.assertTrue(self.safety.get_acc_main_on())

  def test_acc_main_follows_armed_state_from_boot(self):
    # Source: MADS lateral needs only main, so torque 5 flows with controls_allowed false. With MADS off
    # (alternative_experience 0) StarPilot lateral follows cruise, so the same frame is refused; the MADS-on
    # (AOL) equivalent is TestMazdaMeasuredAol.test_long_main_from_boot_grants_lateral_only.
    self._armed()
    self.assertFalse(self.safety.get_controls_allowed())
    self.assertFalse(self._tx(self._torque_cmd_msg(5)))

  def test_acc_main_follows_armed_state_both_ways(self):
    self._armed()
    self._main_off(DEBOUNCE, brake=False)
    self.assertFalse(self.safety.get_acc_main_on())
    self._rx(self._pedals_msg(True))
    self.assertTrue(self.safety.get_acc_main_on())

  def test_main_off_lands_after_the_debounce(self):
    for brake in (False, True):
      self._armed()
      self._main_off(DEBOUNCE - 1, brake)
      self.assertTrue(self.safety.get_acc_main_on())
      self._main_off(1, brake)
      self.assertFalse(self.safety.get_acc_main_on())

  def test_short_dropout_is_held(self):
    self._armed()
    for _ in range(3):
      self._main_off(DEBOUNCE - 1, brake=True)
      self.assertTrue(self.safety.get_acc_main_on())
      self._rx(self._pedals_msg(armed=True, brake=True))

  def test_main_toggled_at_a_stop_re_arms(self):
    self._armed()
    self._main_off(DEBOUNCE, brake=True)
    self.assertFalse(self.safety.get_acc_main_on())
    self._rx(self._pedals_msg(armed=True, brake=True))
    self.assertTrue(self.safety.get_acc_main_on())


class TestMazdaMeasuredTjaLong(MazdaMeasuredLongBase):
  SAFETY_PARAM = TJA_LONG

  def test_declared_button_under_openpilot_longitudinal(self):
    # source: the PEDALS-derived main edge is guarded the same way (mazda_port.h:156-157)
    self._rx(self._pedals_msg(True))
    self.assertFalse(self.safety.get_acc_main_on())
    self._main_off(DEBOUNCE, brake=False)
    self.assertFalse(self.safety.get_acc_main_on())


class MazdaAolHarness:
  """A healthy StarPilot AOL session for one word: every required and optional Mazda AOL source current,
  moving at 0.8 kph, gear D, heartbeat engaged."""

  def __init__(self, tc, word, experience=AOL):
    self.tc, self.word = tc, word
    self.safety = libsafety_py.libsafety
    self.packer = CANPackerSafety("mazda_2017")
    self.safety.init_tests()
    self.safety.set_alternative_experience(experience)
    self.safety.set_safety_hooks(CarParams.SafetyModel.mazda, word)
    self.time = 1_000_000
    self.safety.set_timer(self.time)
    self.safety.set_aol_test_heartbeat(True)
    self.request = 1
    self.cruise = False

  def msg(self, name, fields, bus=0):
    return self.packer.make_can_msg_safety(name, bus, fields)

  def tick(self, us=10_000):
    self.time += us
    self.safety.set_timer(self.time)

  def feed(self, main=True, speed=80, brake=False, gear=4, lkas_block=0, cam_err=0, btns=None, omit=(), request=None, refresh=True):
    long = _is_long(self.word)
    frames = (
      ("CRZ_CTRL", 0, {"CRZ_AVAILABLE": int(main), "CRZ_ACTIVE": int(self.cruise and not long)}),
      ("CRZ_BTNS", 0, dict(btns or {})),
      ("STEER_TORQUE", 0, {"STEER_TORQUE_SENSOR": 0}),
      ("ENGINE_DATA", 0, {"SPEED": speed, "PEDAL_GAS": 0}),
      ("PEDALS", 0, {"BRAKE_ON": int(brake), "ACC_OFF": int(main and long), "ACC_ACTIVE": int(self.cruise and long)}),
      ("GEAR", 0, {"GEAR": gear}),
      ("STEER_RATE", 0, {"LKAS_BLOCK": lkas_block}),
      ("CAM_LKAS", 2, {"ERR_BIT_1": cam_err, "LKAS_REQUEST": 0}),
    )
    for name, bus, fields in frames:
      if name not in omit:
        self.safety.safety_rx_hook(self.msg(name, fields, bus))
    self.safety.safety_tick()
    if refresh:
      self.safety.aol_set_host_request(self.request if request is None else request)

  def prime(self, **kwargs):
    # two rounds: a frame heard before the optional sources are current can neither grant nor go neutral
    self.feed(**kwargs)
    self.feed(**kwargs)

  def main_off(self):
    for _ in range(DEBOUNCE if _is_long(self.word) else 1):
      self.feed(main=False)

  def arm(self):
    """A real physical lateral grant: the main edge (MRCC/PEDALS) or, declared, the TJA press."""
    self.prime()
    if _is_tja(self.word):
      self.feed(btns={"TJA_BUTTON": 1})
    else:
      self.main_off()
      self.feed()
    self.tc.assertEqual(1, self.lat())

  def engage_cruise(self):
    if _is_long(self.word):
      self.safety.safety_rx_hook(self.msg("PEDALS", {"ACC_OFF": 1, "ACC_ACTIVE": 0}))
      self.safety.safety_rx_hook(self.msg("CRZ_BTNS", {"SET_M": 1}))
      self.safety.safety_rx_hook(self.msg("PEDALS", {"ACC_OFF": 1, "ACC_ACTIVE": 1}))
    else:
      self.safety.safety_rx_hook(self.msg("CRZ_CTRL", {"CRZ_AVAILABLE": 1, "CRZ_ACTIVE": 0}))
      self.safety.safety_rx_hook(self.msg("CRZ_CTRL", {"CRZ_AVAILABLE": 1, "CRZ_ACTIVE": 1}))
    self.tc.assertTrue(self.safety.get_controls_allowed())
    self.cruise = True

  def lat(self):
    return self.safety.aol_get_permission_mask() & 1

  def torque(self, t):
    return self.msg("CAM_LKAS", {"LKAS_REQUEST": t})

  def mode(self, on=False, off=False, bus=0):
    return self.msg("CRZ_BTNS", {"MODE_Y": on, "MODE_X": off}, bus)

  def standby(self):
    """Source _standby: main on and the boot-time grant expired with the heartbeat (lateral off)."""
    self.arm()
    self.safety.set_aol_test_heartbeat(False)
    self.feed()
    self.tc.assertEqual(0, self.lat())
    self.safety.set_aol_test_heartbeat(True)
    self.feed()
    self.tc.assertEqual(0, self.lat())


class TestMazdaMeasuredAol(unittest.TestCase):
  """The source's MADS-on scenarios, re-expressed for StarPilot AOL, over the measured words."""

  def tearDown(self):
    lib = libsafety_py.libsafety
    lib.init_tests()
    lib.set_aol_test_heartbeat(False)
    lib.set_alternative_experience(0)
    lib.set_safety_hooks(CarParams.SafetyModel.noOutput, 0)

  def test_stock_passthrough_under_aol(self):
    # source (mads=True): the camera owns 0x243/0x440 unless lateral is allowed; cruise alone does not claim
    # them. AOL: lateral is permission bit 0 = (token or cruise) and the host's lateral request
    # (mazda_stock_aol.h:95-106), so cruise with the host's lateral request off leaves the camera in charge.
    for word in WORDS:
      for controls_allowed, lateral in ((False, False), (True, False), (False, True), (True, True)):
        with self.subTest(word=word, controls_allowed=controls_allowed, lateral=lateral):
          h = MazdaAolHarness(self, word)
          if lateral and not controls_allowed:
            h.arm()
          else:
            h.request = 0
            h.prime()
          if controls_allowed:
            h.engage_cruise()
          h.safety.aol_set_host_request(1 if lateral else 0)
          self.assertEqual(int(lateral), h.lat())
          self.assertEqual(controls_allowed, h.safety.get_controls_allowed())
          for addr, msg in ((0x243, h.torque(0)), (0x440, h.msg("CAM_LANEINFO", {"LINE_VISIBLE": 0}))):
            self.assertEqual(-1 if lateral else 0, h.safety.safety_fwd_hook(2, addr), hex(addr))
            self.assertEqual(lateral, h.safety.safety_tx_hook(msg), hex(addr))

  def test_measured_envelope_under_aol_permission(self):
    for word in WORDS:
      with self.subTest(word=word):
        h = MazdaAolHarness(self, word)
        h.arm()
        self.assertFalse(h.safety.get_controls_allowed())
        h.safety.set_torque_driver(0, 0)
        h.safety.set_desired_torque_last(1200)
        h.safety.set_rt_torque_last(1200)
        self.assertTrue(h.safety.safety_tx_hook(h.torque(1200)))
        self.assertFalse(h.safety.safety_tx_hook(h.torque(1201)))
        h.safety.set_desired_torque_last(0)
        h.safety.set_rt_torque_last(0)
        self.assertTrue(h.safety.safety_tx_hook(h.torque(10)))
        self.assertFalse(h.safety.safety_tx_hook(h.torque(21)))
        h.safety.aol_set_host_request(0)
        self.assertFalse(h.safety.safety_tx_hook(h.torque(10)))

  def test_long_main_from_boot_grants_lateral_only(self):
    # source test_acc_main_follows_armed_state_from_boot: one armed PEDALS sample is main, lateral is allowed,
    # cruise is not. AOL: the boot main grant needs motion, healthy sources and a strictly later host ack
    # (mazda_stock_aol.h:147-154 and pending_ack in mazda_aol_request_mask).
    h = MazdaAolHarness(self, LONG)
    h.safety.set_aol_test_heartbeat(False)
    h.prime()
    self.assertTrue(h.safety.get_acc_main_on())
    h.safety.set_aol_test_heartbeat(True)
    self.assertEqual(0, h.lat())
    h.tick(1)
    h.safety.aol_set_host_request(1)
    self.assertEqual(1, h.lat())
    self.assertFalse(h.safety.get_controls_allowed())
    self.assertTrue(h.safety.safety_tx_hook(h.torque(5)))

  def test_main_edges_drive_lateral(self):
    # source test_acc_main_follows_armed_state_both_ways / test_main_off_lands_after_the_debounce /
    # TestMazdaTjaMads.test_undeclared_keeps_mrcc_path: undeclared words, main off exits lateral after the PEDALS
    # debounce (stock words: on the CRZ_CTRL main-off) and the next main rising edge re-grants.
    for word in (EPS, LONG):
      for brake in (False, True):
        with self.subTest(word=word, brake=brake):
          h = MazdaAolHarness(self, word)
          h.arm()
          if _is_long(word):
            for _ in range(DEBOUNCE - 1):
              h.safety.safety_rx_hook(h.msg("PEDALS", {"ACC_OFF": 0, "BRAKE_ON": int(brake)}))
              self.assertTrue(h.safety.get_acc_main_on())
              self.assertEqual(1, h.lat())
            h.safety.safety_rx_hook(h.msg("PEDALS", {"ACC_OFF": 0, "BRAKE_ON": int(brake)}))
          else:
            h.safety.safety_rx_hook(h.msg("CRZ_CTRL", {"CRZ_AVAILABLE": 0}))
          self.assertFalse(h.safety.get_acc_main_on())
          self.assertEqual(0, h.lat())
          h.feed()
          self.assertTrue(h.safety.get_acc_main_on())
          self.assertEqual(1, h.lat())

  def test_main_toggled_at_a_stop_grants_but_waits_for_motion(self):
    # source: main toggled at a stop with the brake held re-arms lateral there. AOL: the main edge re-grants
    # at the stop, but permission is withheld at standstill (mazda_stock_aol.h:98, vehicle_moving) and returns
    # on motion without another press. StarPilot policy, shared with word 0.
    for word in (EPS, LONG):
      with self.subTest(word=word):
        h = MazdaAolHarness(self, word)
        h.arm()
        for _ in range(DEBOUNCE):
          h.feed(main=False, speed=0, brake=True)
        self.assertEqual(0, h.lat())
        h.feed(main=True, speed=0, brake=True)
        self.assertTrue(h.safety.get_acc_main_on())
        self.assertEqual(0, h.lat())
        h.feed(main=True, speed=80, brake=False)
        self.assertEqual(1, h.lat())

  def test_on_rearms_standby_without_engaging_cruise(self):
    # source TestMazdaMainOnMads: after the boot grant expires (heartbeat exit), MODE_Y ON re-grants lateral only
    for word in (EPS, LONG):
      with self.subTest(word=word):
        h = MazdaAolHarness(self, word)
        h.standby()
        h.safety.safety_rx_hook(h.mode(on=True))
        self.assertEqual(1, h.lat())
        self.assertFalse(h.safety.get_controls_allowed())
        self.assertTrue(h.safety.get_acc_main_on())
        h.safety.safety_rx_hook(h.mode())
        h.main_off()
        self.assertEqual(0, h.lat())

  def test_held_on_does_not_rearm_after_heartbeat_exit(self):
    for word in (EPS, LONG):
      with self.subTest(word=word):
        h = MazdaAolHarness(self, word)
        h.standby()
        h.safety.safety_rx_hook(h.mode(on=True))
        self.assertEqual(1, h.lat())
        h.safety.set_aol_test_heartbeat(False)
        self.assertEqual(0, h.lat())
        h.feed(btns={"MODE_Y": 1})
        h.safety.set_aol_test_heartbeat(True)
        h.feed(btns={"MODE_Y": 1})
        self.assertEqual(0, h.lat())
        h.feed()
        h.feed(btns={"MODE_Y": 1})
        self.assertEqual(1, h.lat())

  def test_off_and_camera_bus_do_not_rearm(self):
    for word in (EPS, LONG):
      with self.subTest(word=word):
        h = MazdaAolHarness(self, word)
        h.standby()
        for msg in (h.mode(off=True), h.mode(), h.mode(on=True, bus=2), h.mode(on=True, bus=1),
                    libsafety_py.make_CANPacket(0x9d, 0, bytes(h.mode(on=True)[0].data)[:7])):
          h.safety.safety_rx_hook(msg)
          self.assertEqual(0, h.lat())

  def test_on_requires_main_on_and_aol(self):
    for word in (EPS, LONG):
      with self.subTest(word=word):
        h = MazdaAolHarness(self, word)
        h.standby()
        h.main_off()
        h.safety.safety_rx_hook(h.mode())
        h.safety.safety_rx_hook(h.mode(on=True))
        self.assertEqual(0, h.lat())
        # MADS disabled maps to alternative_experience 0: ON is not a lateral grant, lateral follows cruise
        h = MazdaAolHarness(self, word, experience=0)
        h.prime()
        h.safety.safety_rx_hook(h.mode(on=True))
        self.assertFalse(h.safety.lateral_controls_allowed())
        self.assertFalse(h.safety.safety_tx_hook(h.torque(5)))

  def test_declared_tja_ignores_on(self):
    for word in (TJA, TJA_LONG):
      with self.subTest(word=word):
        h = MazdaAolHarness(self, word)
        h.prime()
        h.safety.safety_rx_hook(h.mode(on=True))
        self.assertEqual(0, h.lat())

  def test_undeclared_ignores_tja_bit(self):
    # source test_undeclared_keeps_mrcc_path_and_ignores_the_bit: bit 11 is no lateral switch undeclared
    for word in (EPS, LONG):
      with self.subTest(word=word):
        h = MazdaAolHarness(self, word)
        h.standby()
        h.feed(btns={"TJA_BUTTON": 1})
        h.feed()
        self.assertEqual(0, h.lat())

  def test_declared_button_allows_lateral_without_mrcc(self):
    # source: the TJA press is the lateral switch and release keeps it; MRCC never drives the edge. AOL also needs
    # healthy sources and a current host request; acc_main_on is held true for AOL (mazda_port.h:184-186, 209).
    for word in (TJA, TJA_LONG):
      with self.subTest(word=word):
        h = MazdaAolHarness(self, word)
        h.prime(main=False)
        self.assertEqual(0, h.lat())
        self.assertTrue(h.safety.get_acc_main_on())  # architecture: held for AOL, never an MRCC edge
        h.feed(main=False, btns={"TJA_BUTTON": 1})
        self.assertEqual(1, h.lat())
        h.feed(main=False)
        self.assertEqual(1, h.lat())
        # the MRCC master off/on never exits a declared lateral
        for main in (True, False, True):
          for _ in range(DEBOUNCE):
            h.feed(main=main)
          self.assertEqual(1, h.lat())
          self.assertTrue(h.safety.get_acc_main_on())

  def test_tja_press_on_camera_bus_or_wrong_length_is_ignored(self):
    for word in (TJA, TJA_LONG):
      with self.subTest(word=word):
        h = MazdaAolHarness(self, word)
        h.prime()
        press = h.msg("CRZ_BTNS", {"TJA_BUTTON": 1})
        for bad in (h.msg("CRZ_BTNS", {"TJA_BUTTON": 1}, 2), h.msg("CRZ_BTNS", {"TJA_BUTTON": 1}, 1),
                    libsafety_py.make_CANPacket(0x9d, 0, bytes(press[0].data)[:7])):
          h.safety.safety_rx_hook(bad)
          self.assertEqual(0, h.lat())
        h.safety.safety_rx_hook(press)
        self.assertEqual(1, h.lat())

  def test_brake_does_not_pause_native_lateral(self):
    # Source test_pause_on_brake_keeps_request_until_release: MADS pauses lateral in the panda while braking.
    # StarPilot AOL has no panda brake pause: the permission has no brake term (mazda_stock_aol.h:95-106); the
    # pause is a host decision (carstate publishes the raw pause under AOL). The brake still exits cruise.
    for word in WORDS:
      with self.subTest(word=word):
        h = MazdaAolHarness(self, word)
        h.arm()
        h.engage_cruise()
        h.feed(brake=True)
        self.assertFalse(h.safety.get_controls_allowed())
        self.assertEqual(1, h.lat())
        self.assertTrue(h.safety.safety_tx_hook(h.torque(5)))
        h.feed(brake=False)
        self.assertEqual(1, h.lat())

  def test_cancel_exits_cruise_keeps_lateral(self):
    for word in WORDS:
      with self.subTest(word=word):
        h = MazdaAolHarness(self, word)
        h.arm()
        h.engage_cruise()
        h.safety.aol_set_host_request(3)
        self.assertEqual(3, h.safety.aol_get_permission_mask())
        h.safety.safety_rx_hook(h.msg("CRZ_BTNS", {"CAN_OFF": 1}))
        if not _is_long(word):
          # stock long: the cancel press reaches cruise through the radar's CRZ_CTRL
          h.safety.safety_rx_hook(h.msg("CRZ_CTRL", {"CRZ_AVAILABLE": 1, "CRZ_ACTIVE": 0}))
        self.assertFalse(h.safety.get_controls_allowed())
        self.assertEqual(1, h.safety.aol_get_permission_mask())

  def test_faults_withhold_lateral(self):
    # gear-not-D is temporary. LKAS_BLOCK (the EPS's re-arm/standby state, delivering partly above ~4 m/s) never
    # withholds lateral: gating on it kept AOL from ever starting. Nor does the camera's ERR_BIT_1 alone (sun glare):
    # the EPS keeps assisting through it, as on develop.
    for word in WORDS:
      with self.subTest(word=word):
        h = MazdaAolHarness(self, word)
        h.arm()
        h.feed(gear=2)
        self.assertEqual(0, h.lat())
        self.assertFalse(h.safety.safety_tx_hook(h.torque(5)))
        h.feed()
        self.assertEqual(1, h.lat())
        h.feed(lkas_block=1)
        self.assertEqual(1, h.lat())
        self.assertTrue(h.safety.safety_tx_hook(h.torque(5)))
        h.feed()
        h.feed(cam_err=1)
        self.assertEqual(1, h.lat())
        self.assertTrue(h.safety.safety_tx_hook(h.torque(5)))
        h.feed()
        self.assertEqual(1, h.lat())

  def test_stale_sources_withhold_lateral(self):
    required = ["CRZ_BTNS", "STEER_TORQUE", "ENGINE_DATA", "PEDALS", "GEAR", "STEER_RATE", "CAM_LKAS"]
    for word in WORDS:
      for name in required + ([] if _is_long(word) else ["CRZ_CTRL"]):
        with self.subTest(word=word, missing=name):
          h = MazdaAolHarness(self, word)
          h.arm()
          for _ in range(120):
            h.tick()
            h.feed(omit=(name,))
          self.assertEqual(0, h.lat())
          self.assertFalse(h.safety.safety_tx_hook(h.torque(5)))
          h.feed()
          self.assertEqual(0, h.lat())
    # under LONG the radar's CRZ_CTRL is not a source: its absence keeps lateral (mazda_port.h:197-198, 474)
    for word in (LONG, TJA_LONG):
      with self.subTest(word=word, missing="CRZ_CTRL"):
        h = MazdaAolHarness(self, word)
        h.arm()
        for _ in range(120):
          h.tick()
          h.feed(omit=("CRZ_CTRL",))
        self.assertEqual(1, h.lat())

  def test_relay_malfunction_under_aol(self):
    # AOL selects the native tables with check_relay false (mazda_port.h:408-412, 441-463), as StarPilot's base
    # word 0 does (mazda.h:119-120): a car-side 0x243 does not latch; a latched malfunction blocks everything.
    for word in WORDS:
      with self.subTest(word=word):
        h = MazdaAolHarness(self, word)
        h.arm()
        h.safety.safety_rx_hook(make_msg(0, 0x243, 8))
        self.assertFalse(h.safety.get_relay_malfunction())
        h.safety.set_relay_malfunction(True)
        self.assertEqual(0, h.lat())
        self.assertFalse(h.safety.safety_tx_hook(h.torque(5)))
        for addr in (0x243, 0x440, 0x9d, 0x21b):
          self.assertEqual(-1, h.safety.safety_fwd_hook(2, addr))
          self.assertEqual(-1, h.safety.safety_fwd_hook(0, addr))

  def test_aol_alpha_long_accel_holds_through_lateral_restrictions(self):
    # Under AOL the port's longitudinal_accel_checks read longitudinal_controls_allowed() (safety.h), the AOL
    # permission bit 1. With alpha long that bit follows the cruise engagement alone, as alternative_experience 0
    # does (mazda_aol_alpha_long_allowed): the -1.0 standstill hold, an EPS LKAS_BLOCK and out of D keep the accel
    # while the lateral bit is withheld there.
    for word in (LONG, TJA_LONG):
      for experience in (0, AOL):
        with self.subTest(word=word, experience=experience):
          h = MazdaAolHarness(self, word, experience=experience)
          h.prime()
          h.engage_cruise()
          h.tick(1)
          h.request = 3
          h.safety.aol_set_host_request(3)
          accel = h.msg("CRZ_INFO", {"ACCEL_CMD": -1.0, "ACC_ACTIVE": 1})
          zero = h.msg("CRZ_INFO", {"ACCEL_CMD": 0.0, "ACC_ACTIVE": 1})
          self.assertTrue(h.safety.safety_tx_hook(accel))
          for fault in ({"speed": 0}, {"lkas_block": 1}, {"gear": 2}):
            h.feed(**fault)
            self.assertTrue(h.safety.get_controls_allowed(), fault)
            self.assertTrue(h.safety.safety_tx_hook(accel), fault)
            if experience == AOL and "lkas_block" not in fault:
              self.assertEqual(0, h.lat(), fault)
              self.assertEqual(2, h.safety.aol_get_permission_mask() & 2, fault)
            self.assertTrue(h.safety.safety_tx_hook(zero))
            self.assertTrue(h.safety.safety_tx_hook(h.msg("CRZ_CTRL", {"CRZ_ACTIVE": 1})))
            h.feed()
            self.assertTrue(h.safety.safety_tx_hook(accel), fault)

  def test_aol_alpha_long_bit_needs_cruise_request_and_heartbeat(self):
    # Only the lateral restrictions are lifted: no cruise engagement, no host long request, a stale request,
    # a lost heartbeat or a relay malfunction each refuse the nonzero accel, at a stop and moving.
    for word in (LONG, TJA_LONG):
      for speed in (0, 80):
        for fault in ("cruise", "no_request", "stale", "heartbeat", "relay"):
          with self.subTest(word=word, speed=speed, fault=fault):
            h = MazdaAolHarness(self, word)
            h.prime()
            h.engage_cruise()
            h.request = 3
            h.feed(speed=speed)
            accel = h.msg("CRZ_INFO", {"ACCEL_CMD": -1.0, "ACC_ACTIVE": 1})
            self.assertTrue(h.safety.safety_tx_hook(accel))
            if fault == "cruise":
              h.cruise = False
              h.safety.safety_rx_hook(h.msg("PEDALS", {"ACC_OFF": 1, "ACC_ACTIVE": 0}))
              self.assertFalse(h.safety.get_controls_allowed())
            elif fault == "no_request":
              h.safety.aol_set_host_request(1)
            elif fault == "stale":
              # CAN stays current (RX healthy), only the host request ages past AOL_HOST_REQUEST_TIMEOUT_US
              for _ in range(31):
                h.tick()
                h.feed(speed=speed, refresh=False)
            elif fault == "heartbeat":
              h.safety.set_aol_test_heartbeat(False)
            else:
              h.safety.set_relay_malfunction(True)
            self.assertEqual(0, h.safety.aol_get_permission_mask() & 2)
            self.assertFalse(h.safety.safety_tx_hook(accel))
            if fault == "stale":
              h.feed(speed=speed)  # a fresh request alone restores it: the request age was the only gate
              self.assertEqual(2, h.safety.aol_get_permission_mask() & 2)

  def test_stock_long_aol_keeps_the_long_bit_on_the_lateral_gate(self):
    # Stock-long words are unchanged: their long bit still needs motion, gear D and no LKAS_BLOCK.
    for word in (EPS, TJA):
      with self.subTest(word=word):
        h = MazdaAolHarness(self, word)
        h.prime()
        h.engage_cruise()
        h.request = 3
        h.feed()
        self.assertEqual(3, h.safety.aol_get_permission_mask())
        h.feed(speed=0)
        self.assertEqual(0, h.safety.aol_get_permission_mask())


class TestMazdaMeasuredTjaForwardCopy(unittest.TestCase):
  """Source TestMazdaTjaHiddenFromCamera: declared and under AOL (MADS system_enabled), the bus 0 -> bus 2 copy
  of CRZ_BTNS loses bit 11 and nothing else; the car's side and the panda's own RX keep the press."""

  def setUp(self):
    self.safety = libsafety_py.libsafety
    self.packer = CANPackerSafety("mazda_2017")

  def tearDown(self):
    self.safety.init_tests()
    self.safety.set_aol_test_heartbeat(False)
    self.safety.set_alternative_experience(0)
    self.safety.set_safety_hooks(CarParams.SafetyModel.noOutput, 0)

  def _init(self, word, experience=AOL):
    self.safety.init_tests()
    self.safety.set_alternative_experience(experience)
    self.safety.set_safety_hooks(CarParams.SafetyModel.mazda, word)

  def _fwd_copy(self, src_bus, msg):
    orig = bytes(msg[0].data[0:8])
    clone = libsafety_py.make_CANPacket(int(msg[0].addr), int(msg[0].bus), orig)
    self.safety.safety_fwd_modify_hook(src_bus, clone)
    return orig, bytes(clone[0].data[0:8])

  def _btns(self, **values):
    return self.packer.make_can_msg_safety("CRZ_BTNS", 0, values)

  def test_declared_button_is_cleared_and_nothing_else(self):
    # develop 0f6ce31f: the driver's TJA press reaches the camera again, forwarded unmodified
    for word in (TJA, TJA_LONG):
      self._init(word)
      for over in ({}, {"SET_P": 1}, {"RES": 1}, {"CAN_OFF": 1}, {"CTR": 9}):
        orig, fwd = self._fwd_copy(0, self._btns(TJA_BUTTON=1, **over))
        self.assertTrue(orig[1] & 0x08)
        self.assertEqual(orig, fwd, f"{word=} {over=}")

  def test_a_frame_without_the_press_is_untouched(self):
    for word in (TJA, TJA_LONG):
      self._init(word)
      orig, fwd = self._fwd_copy(0, self._btns(TJA_BUTTON=0, SET_P=1))
      self.assertEqual(orig, fwd)

  def test_the_panda_still_sees_the_press(self):
    for word in (TJA, TJA_LONG):
      with self.subTest(word=word):
        h = MazdaAolHarness(self, word)
        h.prime()
        press = h.msg("CRZ_BTNS", {"TJA_BUTTON": 1})
        self._fwd_copy(0, press)
        h.safety.safety_rx_hook(press)
        self.assertEqual(1, h.lat())

  def test_undeclared_or_aol_off_passes_the_press_through(self):
    for word, experience in ((EPS, AOL), (LONG, AOL), (TJA, 0), (TJA_LONG, 0), (0, AOL), (0, 0)):
      self._init(word, experience)
      orig, fwd = self._fwd_copy(0, self._btns(TJA_BUTTON=1))
      self.assertEqual(orig, fwd, (word, experience))

  def test_other_buses_and_addresses_are_untouched(self):
    for word in (TJA, TJA_LONG):
      self._init(word)
      orig, fwd = self._fwd_copy(2, self.packer.make_can_msg_safety("CRZ_BTNS", 2, {"TJA_BUTTON": 1}))
      self.assertEqual(orig, fwd)
      dat = bytes([0xff] * 8)
      for addr in (0x9c, 0x9e, 0x165, 0x21c, 0x440):
        clone = libsafety_py.make_CANPacket(addr, 0, dat)
        self.safety.safety_fwd_modify_hook(0, clone)
        self.assertEqual(dat, bytes(clone[0].data[0:8]), f"{addr=:#x}")


class TestMazdaMeasuredMrccOffCleanup(unittest.TestCase):
  """Source TestMazdaMrccOffCleanup (declared TJA, stock long = word 10; alpha long = word 11), with and
  without AOL: the exception does not depend on the lateral axis (mazda_port.h:338)."""

  MRCC_OFF_VALUES = {
    "CAN_OFF_INV": 1, "SET_P_INV": 1, "RES_INV": 1, "SET_M_INV": 1,
    "DISTANCE_LESS_INV": 1, "DISTANCE_MORE_INV": 1, "MODE_X_INV": 1, "MODE_Y_INV": 1,
    "BIT1_INV": 1, "BIT2": 1, "BIT3": 1, "CTR": 4,
  }

  def setUp(self):
    self.safety = libsafety_py.libsafety
    self.packer = CANPackerSafety("mazda_2017")

  def tearDown(self):
    self.safety.init_tests()
    self.safety.set_alternative_experience(0)
    self.safety.set_safety_hooks(CarParams.SafetyModel.noOutput, 0)

  def _cases(self, words=(TJA,)):
    return [(word, experience) for word in words for experience in (0, AOL)]

  def _setup(self, word, experience):
    self.safety.set_alternative_experience(experience)
    self.safety.set_safety_hooks(CarParams.SafetyModel.mazda, word)
    self.safety.init_tests()

  def _mrcc_off(self, **over):
    return self.packer.make_can_msg_safety("CRZ_BTNS", 0, {**self.MRCC_OFF_VALUES, **over})

  def _pedals(self, acc_off):
    return self.packer.make_can_msg_safety("PEDALS", 0, {"ACC_OFF": acc_off})

  def _crz_ctrl(self, main_on):
    return self.packer.make_can_msg_safety("CRZ_CTRL", 0, {"CRZ_AVAILABLE": main_on})

  def test_exact_tap_allowed_declared_and_armed(self):
    for word, experience in self._cases((TJA, TJA_LONG)):
      with self.subTest(word=word, experience=experience):
        self._setup(word, experience)
        self.safety.safety_rx_hook(self._pedals(True))
        self.assertFalse(self.safety.get_controls_allowed())
        self.assertTrue(self.safety.safety_tx_hook(self._mrcc_off()))

  def test_exact_tap_blocked_undeclared(self):
    for word, experience in self._cases((EPS, LONG)):
      with self.subTest(word=word, experience=experience):
        self._setup(word, experience)
        self.safety.safety_rx_hook(self._pedals(True))
        self.assertFalse(self.safety.safety_tx_hook(self._mrcc_off()))

  def test_exact_tap_blocked_when_cruise_not_armed(self):
    for word, experience in self._cases((TJA, TJA_LONG)):
      with self.subTest(word=word, experience=experience):
        self._setup(word, experience)
        self.safety.safety_rx_hook(self._pedals(False))
        self.assertFalse(self.safety.safety_tx_hook(self._mrcc_off()))

  def test_arm_reads_pedals_not_the_radar_under_stock_long(self):
    for word, experience in self._cases():
      with self.subTest(word=word, experience=experience):
        self._setup(word, experience)
        self.safety.safety_rx_hook(self._crz_ctrl(True))
        self.assertFalse(self.safety.safety_tx_hook(self._mrcc_off()))
        self.safety.safety_rx_hook(self._pedals(True))
        self.assertTrue(self.safety.safety_tx_hook(self._mrcc_off()))
        self.safety.safety_rx_hook(self._crz_ctrl(False))
        self.assertTrue(self.safety.safety_tx_hook(self._mrcc_off()))
        self.safety.safety_rx_hook(self._pedals(False))
        self.assertFalse(self.safety.safety_tx_hook(self._mrcc_off()))

  def test_armed_tracks_pedals_after_teardown(self):
    for word, experience in self._cases((TJA_LONG,)):
      with self.subTest(word=word, experience=experience):
        self._setup(word, experience)
        self.safety.safety_rx_hook(self._pedals(True))
        self.assertTrue(self.safety.safety_tx_hook(self._mrcc_off()))
        self.safety.safety_rx_hook(self._pedals(False))
        self.assertFalse(self.safety.safety_tx_hook(self._mrcc_off()))

  def test_lookalike_tap_is_blocked_everywhere(self):
    for word, experience in self._cases((TJA, TJA_LONG)):
      with self.subTest(word=word, experience=experience):
        self._setup(word, experience)
        self.safety.safety_rx_hook(self._pedals(True))
        for over in ({"SET_P": 1}, {"RES": 1}, {"TJA_BUTTON": 1}, {"BIT2": 0}, {"CTR": 12}):
          self.assertEqual(over == {"CTR": 12}, self.safety.safety_tx_hook(self._mrcc_off(**over)), over)
        self.assertFalse(self.safety.safety_tx_hook(make_msg(0, 0x9d, 7, bytes(self._mrcc_off()[0].data)[:7])))
        self.assertFalse(self.safety.safety_tx_hook(self.packer.make_can_msg_safety("CRZ_BTNS", 2, self.MRCC_OFF_VALUES)))

  def test_cancel_still_allowed_while_not_controlling(self):
    for word, experience in self._cases((TJA, TJA_LONG)):
      with self.subTest(word=word, experience=experience):
        self._setup(word, experience)
        self.safety.safety_rx_hook(self._crz_ctrl(False))
        self.assertTrue(self.safety.safety_tx_hook(self.packer.make_can_msg_safety("CRZ_BTNS", 0, {"CAN_OFF": 1})))

  def test_tja_bit_still_blocked_on_the_main_bus(self):
    for word, experience in self._cases((TJA, TJA_LONG)):
      with self.subTest(word=word, experience=experience):
        self._setup(word, experience)
        self.safety.safety_rx_hook(self._crz_ctrl(True))
        self.assertFalse(self.safety.safety_tx_hook(self.packer.make_can_msg_safety("CRZ_BTNS", 0, {"TJA_BUTTON": 1})))


class TestMazdaMeasuredDiagnostics(unittest.TestCase):
  """0x764 radar UDS admission (mazda_port.h:324-332): single-frame tester present (02 3E 80) and session
  control 01/02 only, LONG words only, bus 0 only, exactly 8 bytes."""

  ALLOWED = ("023e800000000000", "0210010000000000", "0210020000000000")
  REFUSED = (
    "023e000000000000",  # tester present without suppress-positive-response
    "0210030000000000",  # extended session
    "0210820000000000",  # programming session, suppress bit
    "0227010000000000",  # security access
    "0211010000000000",  # ECU reset
    "0328010000000000",  # communication control
    "032e010000000000",  # write data by identifier
    "033e800000000000",  # a 3-byte single frame around a tester-present body
    "1008100200000000",  # ISO-TP first frame carrying a session-control body
    "100a2e0102030405",  # ISO-TP first frame of a write
    "2110020000000000",  # consecutive frame
    "3000000000000000",  # flow control
    "0000000000000000",
  )

  def setUp(self):
    self.safety = libsafety_py.libsafety
    self.packer = CANPackerSafety("mazda_2017")

  def tearDown(self):
    self.safety.init_tests()
    self.safety.set_alternative_experience(0)
    self.safety.set_safety_hooks(CarParams.SafetyModel.noOutput, 0)

  @staticmethod
  def _cases(words):
    return [(word, experience) for word in words for experience in (0, AOL)]

  def _setup(self, word, experience):
    self.safety.set_alternative_experience(experience)
    self.safety.set_safety_hooks(CarParams.SafetyModel.mazda, word)
    self.safety.init_tests()

  def _tx(self, dat, bus=0):
    return self.safety.safety_tx_hook(libsafety_py.make_CANPacket(0x764, bus, bytes.fromhex(dat)))

  def test_allowlist_on_long_words(self):
    for word, experience in self._cases((LONG, TJA_LONG)):
      with self.subTest(word=word, experience=experience):
        self._setup(word, experience)
        for controls_allowed in (False, True):
          self.safety.set_controls_allowed(controls_allowed)
          for dat in self.ALLOWED:
            self.assertTrue(self._tx(dat), dat)
          for dat in self.REFUSED:
            self.assertFalse(self._tx(dat), dat)

  def test_padding_after_the_service_bytes_is_not_inspected(self):
    # characterization: only bytes 0-2 are pinned, the padding is free (source identical)
    for word, experience in self._cases((LONG, TJA_LONG)):
      with self.subTest(word=word, experience=experience):
        self._setup(word, experience)
        self.assertTrue(self._tx("0210020000ffffff"))
        self.assertTrue(self._tx("023e80aaaaaaaaaa"))

  def test_wrong_bus_or_length_refused(self):
    for word, experience in self._cases((LONG, TJA_LONG)):
      with self.subTest(word=word, experience=experience):
        self._setup(word, experience)
        for dat in self.ALLOWED:
          for bus in (1, 2, 3):
            self.assertFalse(self._tx(dat, bus), (dat, bus))
          for length in (3, 4, 7):
            packet = libsafety_py.make_CANPacket(0x764, 0, bytes.fromhex(dat)[:length])
            self.assertFalse(self.safety.safety_tx_hook(packet), (dat, length))

  def test_lateral_only_words_send_no_diagnostics(self):
    for word, experience in self._cases((EPS, TJA)):
      with self.subTest(word=word, experience=experience):
        self._setup(word, experience)
        self.safety.set_controls_allowed(True)
        for dat in self.ALLOWED:
          for bus in range(4):
            self.assertFalse(self._tx(dat, bus))

  def test_programming_session_admitted_while_moving(self):
    # CHARACTERIZATION (inherited from the source, mazda.h at 7b96732d: same 0x764 block): 10 02 silences the
    # radar (and its AEB) and is admitted at any speed, engaged or not, AOL or not. Nothing in the panda ties
    # it to standstill; the controller's own teardown sequencing is the only gate.
    for word, experience in self._cases((LONG, TJA_LONG)):
      with self.subTest(word=word, experience=experience):
        self._setup(word, experience)
        for speed in (0, 30, 130):
          self.safety.safety_rx_hook(self.packer.make_can_msg_safety("ENGINE_DATA", 0, {"SPEED": speed}))
          self.assertEqual(speed > 0.1, self.safety.get_vehicle_moving())
          for controls_allowed in (False, True):
            self.safety.set_controls_allowed(controls_allowed)
            self.assertTrue(self._tx("0210020000000000"), (speed, controls_allowed))
            self.assertTrue(self._tx("0210010000000000"), (speed, controls_allowed))


LONG_FILTER = LONG | int(MazdaSafetyFlags.CAN_FILTER)
FILTER_MAGIC = bytes([0xce, 0xfa, 0xad, 0xde])


class TestMazdaMeasuredLongFilter(TestMazdaMeasuredLong):
  """develop d61310c0, Alpha long through the Smart CAN Filter: every longitudinal check above still holds, the
  UDS takeover and the synthetic radar frames are gone, and FILTER_CTRL is the one extra message. The port
  carries the filter in its one safety word (MazdaSafetyFlags.CAN_FILTER); develop in its second word."""

  SAFETY_PARAM = LONG_FILTER
  TX_MSGS = LAT_TX + [[0x21b, 0], [0x21c, 0], [0x2a0, 0], [0x21b, 2], [0x21c, 2]]

  def test_filter_ctrl_mode_and_filter_bits_only(self):
    for ctrl in range(4):
      self.assertTrue(self._tx(make_msg(0, 0x2a0, 8, FILTER_MAGIC + bytes([0x2e, ctrl, 0, 0]))))
    for ctrl in (0x04, 0x80, 0xff):
      self.assertFalse(self._tx(make_msg(0, 0x2a0, 8, FILTER_MAGIC + bytes([0x2e, ctrl, 0, 0]))))
    # reserved bytes must be zero, the magic exact, and only the main bus
    self.assertFalse(self._tx(make_msg(0, 0x2a0, 8, FILTER_MAGIC + bytes([0x2e, 1, 1, 0]))))
    self.assertFalse(self._tx(make_msg(0, 0x2a0, 8, FILTER_MAGIC + bytes([0x2e, 1, 0, 1]))))
    self.assertFalse(self._tx(make_msg(0, 0x2a0, 8, bytes([0xce, 0xfa, 0xad, 0xdf, 0x2e, 1, 0, 0]))))
    self.assertFalse(self._tx(make_msg(2, 0x2a0, 8, FILTER_MAGIC + bytes([0x2e, 1, 0, 0]))))

  def test_board_reset_flash_and_feature_commands_blocked(self):
    for cmd in ("1e0bb002", "1e0bb00a", "1e0bb001", "1f010001", "1f000000"):
      self.assertFalse(self._tx(make_msg(0, 0x2a0, 8, FILTER_MAGIC + bytes.fromhex(cmd))), cmd)
    for addr in (0x2a1, 0x2a8, 0x2a9):
      self.assertFalse(self._tx(make_msg(0, addr, 8, FILTER_MAGIC + bytes([0x2e, 1, 0, 0]))), hex(addr))

  def test_filter_ctrl_independent_of_controls(self):
    msg = FILTER_MAGIC + bytes([0x2e, 3, 0, 0])
    for allowed in (False, True):
      self.safety.set_controls_allowed(allowed)
      self.assertTrue(self._tx(make_msg(0, 0x2a0, 8, msg)))

  def test_aol_uses_the_filter_table_too(self):
    self._init(experience=AOL)
    self.assertTrue(self._tx(make_msg(0, 0x2a0, 8, FILTER_MAGIC + bytes([0x2e, 3, 0, 0]))))
    self.assertFalse(self._tx(make_msg(0, 0x764, 8, bytes.fromhex("0210020000000000"))))
    self.assertFalse(self._tx(make_msg(0, 0x499, 8, RADAR_CAPTURES[0x499])))

  # the UDS takeover and its synthetic radar frames are blocked outright
  def test_empty_radar_tracks_allowed(self):
    for allowed in (False, True):
      self.safety.set_controls_allowed(allowed)
      for bus in (0, 2):
        for addr, dat in RADAR_CAPTURES.items():
          self.assertFalse(self._tx(make_msg(bus, addr, 8, dat)), f"{addr=:#x} {bus=}")

  def test_g46l_radar_static_allowed(self):
    for bus in (0, 2):
      self.assertFalse(self._tx(make_msg(bus, 0x499, 8, bytes.fromhex("0098400000000000"))))

  def test_radar_uds_allowlist(self):
    self.safety.set_controls_allowed(True)
    for dat in ("023e800000000000", "0210020000000000", "0210010000000000"):
      self.assertFalse(self._tx(make_msg(0, 0x764, 8, bytes.fromhex(dat))))


class TestMazdaMeasuredFilterNeedsLongitudinal(TestMazdaMeasuredEps):
  """The filter bit alone, without alpha long, changes nothing: FILTER_CTRL stays blocked."""

  SAFETY_PARAM = EPS | int(MazdaSafetyFlags.CAN_FILTER)

  def test_filter_ctrl_blocked(self):
    for allowed in (False, True):
      self.safety.set_controls_allowed(allowed)
      self.assertFalse(self._tx(make_msg(0, 0x2a0, 8, FILTER_MAGIC + bytes([0x2e, 3, 0, 0]))))


class TestMazdaMeasuredLongBlocksFilterCtrl(TestMazdaMeasuredLong):
  def test_filter_ctrl_blocked_without_the_filter(self):
    self.assertFalse(self._tx(make_msg(0, 0x2a0, 8, FILTER_MAGIC + bytes([0x2e, 3, 0, 0]))))


if __name__ == "__main__":
  unittest.main()
