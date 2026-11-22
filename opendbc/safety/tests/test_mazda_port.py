import pytest
from opendbc.car import structs
from opendbc.safety.tests.common import CANPackerSafety
from opendbc.safety.tests.libsafety import libsafety_py


@pytest.fixture
def safety():
  lib = libsafety_py.libsafety
  lib.init_tests()
  lib.set_alternative_experience(0)
  lib.set_safety_hooks(structs.CarParams.SafetyModel.mazda, 3)
  yield lib
  lib.set_alternative_experience(0)
  lib.set_safety_hooks(structs.CarParams.SafetyModel.noOutput, 0)


def test_measured_profile_accepts_its_limit_and_blocks_excess(safety):
  packer = CANPackerSafety('mazda_2017')
  safety.set_controls_allowed(True)
  safety.set_torque_driver(0, 0)
  safety.set_desired_torque_last(1200)
  safety.set_rt_torque_last(1200)
  assert safety.safety_tx_hook(packer.make_can_msg_safety('CAM_LKAS', 0, {'LKAS_REQUEST': 1200}))
  assert not safety.safety_tx_hook(packer.make_can_msg_safety('CAM_LKAS', 0, {'LKAS_REQUEST': 1201}))


@pytest.mark.parametrize('payload,allowed', [(b'\x02\x10\x01'+bytes(5), True), (b'\x02\x10\x02'+bytes(5), True),
                                           (b'\x02\x3e\x80'+bytes(5), True), (b'\x02\x11\x01'+bytes(5), False),
                                           (b'\x02\x34\x00'+bytes(5), False)])
def test_only_radar_session_diagnostics_are_allowed(safety, payload, allowed):
  packet = libsafety_py.make_CANPacket(0x764, 0, payload)
  assert bool(safety.safety_tx_hook(packet)) == allowed


@pytest.mark.parametrize('param', [0, 2, 3, 10, 11])
def test_forward_copy_only_strips_declared_tja(safety, param):
  safety.set_alternative_experience(32)
  safety.set_safety_hooks(structs.CarParams.SafetyModel.mazda, param)
  original = b'\x00\x08\x00\x00\x00\x00\x00\x00'
  packet = libsafety_py.make_CANPacket(0x9d, 0, original)
  safety.safety_fwd_modify_hook(0, packet)
  # develop 0f6ce31f: the driver's TJA press reaches the camera again, forwarded unmodified
  assert bytes(packet.data)[:8] == original
  camera = libsafety_py.make_CANPacket(0x9d, 2, original)
  safety.safety_fwd_modify_hook(2, camera)
  assert bytes(camera.data)[:8] == original


@pytest.mark.parametrize('param', [1, 4, 6, 8, 16, 0xffff])
def test_unsupported_profile_cannot_send_diagnostics_or_torque(safety, param):
  safety.set_safety_hooks(structs.CarParams.SafetyModel.mazda, param)
  safety.set_controls_allowed(True)
  assert not safety.safety_tx_hook(libsafety_py.make_CANPacket(0x764, 0, b'\x02\x10\x02'+bytes(5)))
  packet = CANPackerSafety('mazda_2017').make_can_msg_safety('CAM_LKAS', 0, {'LKAS_REQUEST': 10})
  assert not safety.safety_tx_hook(packet)


def test_physical_on_rearms_only_lateral_after_native_session_loss(safety):
  from opendbc.safety.tests.test_mazda_stock_aol import TestMazdaStockAol
  fixture = TestMazdaStockAol()
  fixture.configure(word=2)
  fixture.arm()
  fixture.safety.set_aol_test_heartbeat(False)
  fixture.feed()
  assert fixture.safety.aol_get_permission_mask() == 0
  fixture.safety.set_aol_test_heartbeat(True)
  fixture.feed()
  fixture.safety.safety_rx_hook(fixture.packet('CRZ_BTNS', {'MODE_Y': 1}))
  assert fixture.safety.aol_get_permission_mask() == 1
  assert not fixture.safety.get_controls_allowed()


def test_alpha_native_main_uses_the_same_pedals_debounce_as_the_controller(safety):
  from opendbc.safety.tests.test_mazda_stock_aol import TestMazdaStockAol
  fixture = TestMazdaStockAol()
  fixture.configure(word=3)
  for _ in range(5):
    fixture.time += 10_000
    fixture.safety.set_timer(fixture.time)
    fixture.feed()
    fixture.safety.safety_rx_hook(fixture.packet('PEDALS', {'ACC_OFF': 1, 'BRAKE_ON': 0}))
  assert fixture.safety.aol_get_permission_mask() == 1
  for _ in range(9):
    fixture.feed()
    assert fixture.safety.get_acc_main_on()
    assert fixture.safety.aol_get_permission_mask() == 1
  fixture.feed()
  assert not fixture.safety.get_acc_main_on()
  assert fixture.safety.aol_get_permission_mask() == 0


@pytest.mark.parametrize('word', [2, 3, 10, 11])
def test_physical_cruise_grant_retains_only_lateral_after_brake(safety, word):
  from opendbc.safety.tests.test_mazda_stock_aol import TestMazdaStockAol
  fixture = TestMazdaStockAol()
  fixture.configure(word=word)
  fixture.safety.set_aol_test_heartbeat(False)
  for _ in range(20):
    fixture.time += 10_000
    fixture.safety.set_timer(fixture.time)
    fixture.feed()
    fixture.safety.aol_set_host_request(0)
  if word & 1:
    fixture.safety.safety_rx_hook(fixture.packet('CRZ_BTNS', {'RES': 1}))
  name, fields = ('CRZ_CTRL', {'CRZ_AVAILABLE': 1, 'CRZ_ACTIVE': 1}) if not word & 1 else ('PEDALS', {'ACC_ACTIVE': 1})
  fixture.safety.safety_rx_hook(fixture.packet(name, fields))
  assert fixture.safety.get_controls_allowed()
  fixture.time += 10_000
  fixture.safety.set_timer(fixture.time)
  fixture.safety.set_aol_test_heartbeat(True)
  fixture.safety.aol_set_host_request(3)
  assert fixture.safety.aol_get_permission_mask() == 3
  fixture.safety.safety_rx_hook(fixture.packet('PEDALS', {'BRAKE_ON': 1, 'ACC_ACTIVE': 1}))
  assert not fixture.safety.get_controls_allowed()
  assert fixture.safety.aol_get_permission_mask() == 1


@pytest.mark.parametrize('word', [3, 11, 19, 27])
def test_can_filter_word_keeps_aol_lateral(safety, word):
  """develop keeps MADS under the Smart CAN Filter: alpha long words with the filter bit (16) arm AOL lateral and
  steer as the same words without it do."""
  import unittest
  from opendbc.safety.tests.test_mazda_measured_profiles import MazdaAolHarness
  h = MazdaAolHarness(unittest.TestCase(), word)
  h.arm()  # asserts the lateral grant
  assert h.lat() == 1


@pytest.mark.parametrize('word', [3, 11, 19, 27])
def test_aol_cruise_active_bits_need_native_long_permission(safety, word):
  """Under AOL, physical cruise permission alone cannot keep cruise active or release a standstill hold: ACC_ACTIVE,
  the standstill bits and CRZ_ACTIVE need the host's current native long permission, like a nonzero accel does."""
  from opendbc.safety.tests.test_mazda_stock_aol import TestMazdaStockAol
  fixture = TestMazdaStockAol()
  fixture.configure(word=word)
  fixture.safety.set_aol_test_heartbeat(False)
  for _ in range(20):
    fixture.time += 10_000
    fixture.safety.set_timer(fixture.time)
    fixture.feed()
    fixture.safety.aol_set_host_request(0)
  fixture.safety.safety_rx_hook(fixture.packet('CRZ_BTNS', {'RES': 1}))
  fixture.safety.safety_rx_hook(fixture.packet('PEDALS', {'ACC_ACTIVE': 1}))
  assert fixture.safety.get_controls_allowed()
  fixture.time += 10_000
  fixture.safety.set_timer(fixture.time)
  fixture.safety.set_aol_test_heartbeat(True)

  frames = [('CRZ_INFO', {'ACC_ACTIVE': 1, 'ACCEL_CMD': 0.0}), ('CRZ_INFO', {'ACC_ACTIVE': 1, 'STOPPING': 1, 'ACCEL_CMD': 0.0}),
            ('CRZ_INFO', {'ACC_ACTIVE': 1, 'RESUME_UNLATCHING': 1, 'ACCEL_CMD': 0.0}), ('CRZ_CTRL', {'CRZ_ACTIVE': 1})]
  fixture.safety.aol_set_host_request(3)
  assert fixture.safety.aol_get_permission_mask() == 3
  for name, fields in frames:
    assert fixture.safety.safety_tx_hook(fixture.packet(name, fields)), (name, fields)
  fixture.safety.aol_set_host_request(1)  # the host withdraws long, lateral stays
  assert fixture.safety.get_controls_allowed()
  assert fixture.safety.aol_get_permission_mask() == 1
  for name, fields in frames:
    assert not fixture.safety.safety_tx_hook(fixture.packet(name, fields)), (name, fields)


@pytest.mark.parametrize('word', [11, 27])
def test_aol_camera_press_off_needs_the_lateral_grant(safety, word):
  """The car's config (word 27 under AOL): the camera-only TJA press-off (mads update_camera_tja) is refused until
  the AOL lateral grant and passes with it. Active camera TJA alongside openpilot steering faults the FSC (0000013b)."""
  import unittest
  from opendbc.safety.tests.test_mazda_measured_profiles import MazdaAolHarness
  h = MazdaAolHarness(unittest.TestCase(), word)
  idle = {'TJA_BUTTON': 1, 'DISTANCE_LESS_INV': 1, 'BIT1': 1, 'BIT2': 1, 'BIT3': 1, 'CAN_OFF_INV': 1, 'RES_INV': 1,
          'SET_P_INV': 1, 'SET_M_INV': 1, 'DISTANCE_MORE_INV': 1, 'MODE_X_INV': 1, 'MODE_Y_INV': 1}
  h.prime()
  assert not h.safety.safety_tx_hook(h.msg('CRZ_BTNS', dict(idle, CTR=5), 2))
  h.arm()
  for _ in range(30):
    h.tick()
    h.feed()
  assert h.lat() == 1
  assert h.safety.safety_tx_hook(h.msg('CRZ_BTNS', dict(idle, CTR=6), 2))
