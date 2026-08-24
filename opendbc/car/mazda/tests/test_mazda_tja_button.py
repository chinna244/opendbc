#!/usr/bin/env python3
import pytest

from opendbc.can import CANPacker, CANParser
from opendbc.car import gen_empty_fingerprint, structs
from opendbc.car.mazda import mazdacan
from opendbc.car.mazda.interface import CarInterface
from opendbc.car.mazda.values import CAR, Buttons

ButtonType = structs.CarState.ButtonEvent.Type


def _interface(candidate=CAR.MAZDA_CX5_2022):
  fingerprint = gen_empty_fingerprint()
  CP = CarInterface.get_params(candidate, fingerprint, [], alpha_long=False, is_release=False, docs=False)
  CP_SP = CarInterface.get_params_sp(CP, candidate, fingerprint, [], alpha_long=False,
                                     is_release_sp=False, docs=False)
  return CarInterface(CP, CP_SP)


class ButtonHarness:
  def __init__(self, candidate=CAR.MAZDA_CX5_2022):
    self.ci = _interface(candidate)
    self.packer = CANPacker("mazda_2017")
    self.time = 0

  def step(self, *, tja=0, mode_x=0, mode_y=0, available=0, active=0):
    self.time += 10_000_000
    buttons = self.packer.make_can_msg("CRZ_BTNS", 0, {
      "TJA_BUTTON": tja,
      "MODE_X": mode_x,
      "MODE_Y": mode_y,
      "BIT1": 1,
      "BIT2": 1,
      "BIT3": 1,
    })
    cruise = self.packer.make_can_msg("CRZ_CTRL", 0, {
      "CRZ_AVAILABLE": available,
      "CRZ_ACTIVE": active,
    })
    return self.ci.update([(self.time, [buttons, cruise])])[0]


def _events(state, event_type):
  return [event for event in state.buttonEvents if event.type == event_type]


@pytest.mark.parametrize("available,active", [(0, 0), (1, 0), (1, 1)])
def test_physical_tja_emits_lkas_without_changing_cruise(available, active):
  harness = ButtonHarness()
  harness.step()
  baseline = harness.step(available=available, active=active)
  pressed = harness.step(tja=1, available=available, active=active)
  held = harness.step(tja=1, available=available, active=active)
  released = harness.step(available=available, active=active)
  pressed_again = harness.step(tja=1, available=available, active=active)

  assert [(event.pressed, event.type) for event in _events(pressed, ButtonType.lkas)] == [(True, ButtonType.lkas)]
  assert not _events(held, ButtonType.lkas)
  assert [(event.pressed, event.type) for event in _events(released, ButtonType.lkas)] == [(False, ButtonType.lkas)]
  assert [(event.pressed, event.type) for event in _events(pressed_again, ButtonType.lkas)] == [(True, ButtonType.lkas)]
  assert not any(_events(state, ButtonType.mainCruise) for state in (pressed, held, released, pressed_again))
  for state in (pressed, held, released, pressed_again):
    assert state.cruiseState.available == baseline.cruiseState.available
    assert state.cruiseState.enabled == baseline.cruiseState.enabled


def test_mrcc_mode_bits_do_not_emit_tja_or_change_cruise_on_target():
  harness = ButtonHarness()
  harness.step()
  state = harness.step(mode_x=1, mode_y=1)
  assert not _events(state, ButtonType.lkas)
  assert not _events(state, ButtonType.mainCruise)
  assert not state.cruiseState.available
  assert not state.cruiseState.enabled


def test_non_target_mazda_retains_main_cruise_semantics():
  harness = ButtonHarness(CAR.MAZDA_CX5)
  harness.step()
  mode = harness.step(mode_x=1, mode_y=1)
  tja = harness.step(tja=1)
  assert len(_events(mode, ButtonType.mainCruise)) == 1
  assert not _events(mode, ButtonType.lkas)
  assert not _events(tja, ButtonType.lkas)


def _decode_buttons(data):
  parser = CANParser("mazda_2017", [("CRZ_BTNS", 10)], 0)
  parser.update([(0, [(0x09d, data, 0)])])
  return parser.vl["CRZ_BTNS"]


@pytest.mark.parametrize("button", [Buttons.CANCEL, Buttons.RESUME, Buttons.SET_PLUS, Buttons.SET_MINUS])
def test_synthetic_buttons_preserve_live_tja_and_mode(button):
  ci = _interface()
  packer = CANPacker("mazda_2017")
  state = type("State", (), {"tja_button": 1, "mode_x": 1, "mode_y": 0})()
  _, data, _ = mazdacan.create_button_cmd(packer, ci.CP, 3, button, state)
  decoded = _decode_buttons(data)
  assert decoded["TJA_BUTTON"] == 1
  assert decoded["MODE_X"] == 1
  assert decoded["MODE_Y"] == 0


def test_non_target_synthetic_buttons_keep_upstream_zero_bits():
  ci = _interface(CAR.MAZDA_CX5)
  packer = CANPacker("mazda_2017")
  state = type("State", (), {"tja_button": 1, "mode_x": 1, "mode_y": 1})()
  _, data, _ = mazdacan.create_button_cmd(packer, ci.CP, 3, Buttons.CANCEL, state)
  decoded = _decode_buttons(data)
  assert decoded["TJA_BUTTON"] == 0
  assert decoded["MODE_X"] == 0
  assert decoded["MODE_Y"] == 0
