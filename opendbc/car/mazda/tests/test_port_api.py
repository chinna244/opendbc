from opendbc.can import CANPacker
from opendbc.car import gen_empty_fingerprint, structs
from opendbc.car.mazda.interface import CarInterface
from opendbc.car.mazda.values import CAR, CarControllerParams


def test_standard_interface_offers_matched_mazda_longitudinal():
  cp = CarInterface.get_params(CAR.MAZDA_CX5_2022, gen_empty_fingerprint(), [], True, False, False)
  assert cp.alphaLongitudinalAvailable and cp.openpilotLongitudinalControl
  ci = CarInterface(cp)
  assert ci.CS.radar_session is ci.CC.radar_session
  assert CarControllerParams(cp).STEER_MAX == 1200
  assert cp.safetyConfigs[0].safetyParam & 3 == 3
  state = ci.update([])
  assert state.schema == structs.CarState.schema
  actuators, messages = ci.apply(structs.CarControl().as_reader(), 1_000_000_000)
  assert actuators.torque == 0
  assert not any(addr == 0x764 for addr, _, _ in messages)


def test_standard_parser_preserves_blind_spot_signals():
  cp = CarInterface.get_params(CAR.MAZDA_CX5_2022, gen_empty_fingerprint(), [], False, False, False)
  ci = CarInterface(cp)
  ci.update([])
  packet = CANPacker('mazda_2017').make_can_msg('BSM', 0, {'LEFT_BS_STATUS': 1, 'RIGHT_BS_STATUS': 0})
  state = ci.update([(100_000_000, [packet])])
  assert state.leftBlindspot and not state.rightBlindspot


def test_standard_parser_keeps_a_raw_eps_block_out_of_the_steering_fault():
  # With or without AOL a single raw block is
  # not a steerFaultTemporary (test_mazda_production_eps covers delivery, re-arm and the delayed fault).
  cp = CarInterface.get_params(CAR.MAZDA_CX5_2022, gen_empty_fingerprint(), [], False, False, False)
  ci = CarInterface(cp)
  packer = CANPacker('mazda_2017')
  ci.update([])
  blocked = packer.make_can_msg('STEER_RATE', 0, {'LKAS_BLOCK': 1})
  state = ci.update([(100_000_000, [blocked])])
  assert ci.CS.lkas_blocked and not ci.CS.steer_undelivered_alert
  assert not state.steerFaultTemporary
