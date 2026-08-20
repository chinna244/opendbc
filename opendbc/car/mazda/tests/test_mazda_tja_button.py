#!/usr/bin/env python3
"""TJA-only MADS button contract for Mazda CX-5 2022+.

Physical TJA rising edge emits ButtonType.lkas once on stock long and openpilot
long. Held TJA does not repeat. Release emits unpressed only. MRCC is OEM-only
and must not emit lkas or mainCruise. cruiseState follows OEM PEDALS (op long)
or CRZ_CTRL (stock long), never a TJA override.
"""

import random

import pytest

from opendbc.can import CANPacker, CANParser
from opendbc.car import DT_CTRL, gen_empty_fingerprint, structs
from opendbc.car.mazda import mazdacan
from opendbc.car.mazda.interface import CarInterface
from opendbc.car.mazda.values import CAR, Buttons, CarControllerParams, MazdaSafetyFlags, STEER_TO_ZERO_EPS_FW

ButtonType = structs.CarState.ButtonEvent.Type

# ff7df7d6f9c3403b|00000033--7b0201ce40 CRZ_BTNS TJA_BUTTON run-length.
# 27 complete 0→1 pulses (each high 2-4 frames / 100-300ms). Minimum spacing
# between rises is 470ms, so this is not intra-press bounce.
ROUTE_TJA_RLE = (
  (0, 217), (1, 3), (0, 115), (1, 3), (0, 4), (1, 3), (0, 50), (1, 2), (0, 4), (1, 3),
  (0, 18), (1, 2), (0, 30), (1, 3), (0, 102), (1, 3), (0, 8), (1, 3), (0, 12), (1, 3),
  (0, 19), (1, 3), (0, 5), (1, 3), (0, 10), (1, 3), (0, 286), (1, 3), (0, 5), (1, 2),
  (0, 18), (1, 3), (0, 12), (1, 3), (0, 33), (1, 3), (0, 6), (1, 3), (0, 6), (1, 3),
  (0, 6), (1, 3), (0, 10), (1, 4), (0, 13), (1, 3), (0, 16), (1, 3), (0, 23), (1, 3),
  (0, 14), (1, 3), (0, 8), (1, 4), (0, 22),
)
ROUTE_TJA_RISING_EDGES = 27


def _ci(*, alpha_long=True):
  fingerprint = gen_empty_fingerprint()
  CP = CarInterface.get_params(CAR.MAZDA_CX5_2022, fingerprint, [], alpha_long=alpha_long,
                               is_release=False, docs=False)
  CP_SP = CarInterface.get_params_sp(CP, CAR.MAZDA_CX5_2022, fingerprint, [],
                                     alpha_long=alpha_long, is_release_sp=False, docs=False)
  assert CP.openpilotLongitudinalControl == alpha_long
  return CarInterface(CP, CP_SP)


def _crz_btns(packer, *, tja=0, set_p=0, set_m=0, res=0, can_off=0, mode_x=0, mode_y=0):
  return packer.make_can_msg("CRZ_BTNS", 0, {
    "TJA_BUTTON": tja,
    "SET_P": set_p,
    "SET_M": set_m,
    "RES": res,
    "CAN_OFF": can_off,
    "MODE_X": mode_x,
    "MODE_Y": mode_y,
    "BIT1": 1,
    "BIT2": 1,
    "BIT3": 1,
  })


class ButtonHarness:
  def __init__(self, *, alpha_long=True):
    self.ci = _ci(alpha_long=alpha_long)
    self.packer = CANPacker("mazda_2017")
    self.t = 0
    if alpha_long:
      # Op-long cruise available is gated on the two-master radar-silence latch.
      for _ in range(int(CarControllerParams.STOCK_RADAR_GUARD_T / DT_CTRL) + 1):
        self.step()

  def step(self, *, tja=0, acc_off=0, acc_active=0, set_p=0, set_m=0,
           res=0, can_off=0, mode_x=0, mode_y=0, crz_available=0, crz_active=0):
    self.t += 10_000_000
    crz = _crz_btns(self.packer, tja=tja, set_p=set_p, set_m=set_m,
                    res=res, can_off=can_off, mode_x=mode_x, mode_y=mode_y)
    pedals = self.packer.make_can_msg("PEDALS", 0, {
      "ACC_OFF": acc_off,
      "ACC_ACTIVE": acc_active,
      "BRAKE_ON": 0,
    })
    msgs = [crz, pedals]
    if not self.ci.CP.openpilotLongitudinalControl:
      msgs.append(self.packer.make_can_msg("CRZ_CTRL", 0, {
        "CRZ_AVAILABLE": crz_available,
        "CRZ_ACTIVE": crz_active,
      }))
    cs, _ = self.ci.update([(self.t, msgs)])
    return cs


def _events(cs, button_type):
  return [be for be in cs.buttonEvents if be.type == button_type]


def _assert_one_lkas_press(cs):
  ev = _events(cs, ButtonType.lkas)
  assert len(ev) == 1
  assert ev[0].pressed
  assert not _events(cs, ButtonType.mainCruise)


@pytest.mark.parametrize("alpha_long", [False, True])
class TestMazdaTjaButton:
  def test_tja_rising_edge_emits_one_lkas(self, alpha_long):
    h = ButtonHarness(alpha_long=alpha_long)
    h.step()
    cs = h.step(tja=1)
    _assert_one_lkas_press(cs)

  def test_tja_held_emits_no_additional_lkas(self, alpha_long):
    h = ButtonHarness(alpha_long=alpha_long)
    h.step()
    h.step(tja=1)
    for _ in range(5):
      held = h.step(tja=1)
      assert not _events(held, ButtonType.lkas)
      assert not _events(held, ButtonType.mainCruise)

  def test_tja_release_does_not_press_lkas(self, alpha_long):
    h = ButtonHarness(alpha_long=alpha_long)
    h.step()
    h.step(tja=1)
    release = h.step(tja=0)
    ev = _events(release, ButtonType.lkas)
    assert len(ev) == 1
    assert not ev[0].pressed
    assert not _events(release, ButtonType.mainCruise)

  def test_second_tja_rising_edge_emits_one_lkas(self, alpha_long):
    h = ButtonHarness(alpha_long=alpha_long)
    h.step()
    h.step(tja=1)
    h.step(tja=0)
    cs = h.step(tja=1)
    _assert_one_lkas_press(cs)

  def test_one_lkas_press_per_rising_edge(self, alpha_long):
    h = ButtonHarness(alpha_long=alpha_long)
    h.step()
    presses = 0
    for tja in (1, 0, 1, 0, 1, 0, 1, 0):
      cs = h.step(tja=tja)
      presses += sum(1 for be in _events(cs, ButtonType.lkas) if be.pressed)
    assert presses == 4

  def test_mode_x_y_do_not_emit_lkas_or_maincruise(self, alpha_long):
    h = ButtonHarness(alpha_long=alpha_long)
    h.step()
    for kwargs in ({"mode_x": 1}, {"mode_y": 1}, {"mode_x": 1, "mode_y": 1}):
      h.step()
      cs = h.step(**kwargs)
      assert not _events(cs, ButtonType.lkas)
      assert not _events(cs, ButtonType.mainCruise)
      held = h.step(**kwargs)
      assert not _events(held, ButtonType.lkas)
      assert not _events(held, ButtonType.mainCruise)
      release = h.step()
      assert not _events(release, ButtonType.lkas)
      assert not _events(release, ButtonType.mainCruise)

  def test_mrcc_does_not_emit_lkas_or_maincruise(self, alpha_long):
    h = ButtonHarness(alpha_long=alpha_long)
    h.step()
    cs = h.step(mode_x=1, mode_y=1)
    assert not _events(cs, ButtonType.lkas)
    assert not _events(cs, ButtonType.mainCruise)
    held = h.step(mode_x=1, mode_y=1)
    assert not _events(held, ButtonType.lkas)
    assert not _events(held, ButtonType.mainCruise)
    release = h.step()
    assert not _events(release, ButtonType.lkas)
    assert not _events(release, ButtonType.mainCruise)

  def test_set_res_cancel_do_not_emit_lkas(self, alpha_long):
    h = ButtonHarness(alpha_long=alpha_long)
    h.step()
    for kwargs, expected in (
      ({"set_p": 1}, ButtonType.accelCruise),
      ({"set_m": 1}, ButtonType.decelCruise),
      ({"res": 1}, ButtonType.resumeCruise),
      ({"can_off": 1}, ButtonType.cancel),
    ):
      h.step()
      cs = h.step(**kwargs)
      assert _events(cs, expected)
      assert not _events(cs, ButtonType.lkas)
      assert not _events(cs, ButtonType.mainCruise)

  def test_route_tja_pulses_emit_one_lkas_each(self, alpha_long):
    h = ButtonHarness(alpha_long=alpha_long)
    presses = 0
    for tja, n in ROUTE_TJA_RLE:
      for _ in range(n):
        cs = h.step(tja=tja)
        presses += sum(1 for be in _events(cs, ButtonType.lkas) if be.pressed)
        if tja == 1:
          assert not _events(cs, ButtonType.mainCruise)
    assert presses == ROUTE_TJA_RISING_EDGES

  def test_randomized_button_fuzz(self, alpha_long):
    rng = random.Random(0)
    h = ButtonHarness(alpha_long=alpha_long)
    h.step()
    prev_tja = 0
    expected = 0
    actual = 0
    for _ in range(400):
      tja = rng.choice((0, 0, 0, 1))
      kwargs = {
        "tja": tja,
        "mode_x": rng.choice((0, 0, 1)),
        "mode_y": rng.choice((0, 0, 1)),
        "set_p": rng.choice((0, 0, 1)),
        "set_m": rng.choice((0, 0, 1)),
        "res": rng.choice((0, 0, 1)),
        "can_off": rng.choice((0, 0, 1)),
      }
      cs = h.step(**kwargs)
      if tja == 1 and prev_tja == 0:
        expected += 1
      actual += sum(1 for be in _events(cs, ButtonType.lkas) if be.pressed)
      assert not _events(cs, ButtonType.mainCruise)
      if tja == prev_tja:
        assert not any(be.pressed for be in _events(cs, ButtonType.lkas))
      if tja == 0 and prev_tja == 1:
        ev = _events(cs, ButtonType.lkas)
        assert len(ev) == 1
        assert not ev[0].pressed
      prev_tja = tja
    assert actual == expected
    assert expected > 0


class TestMazdaTjaCruiseState:
  def test_tja_does_not_fabricate_cruise_available_or_enabled(self):
    h = ButtonHarness()
    off = h.step()
    assert not off.cruiseState.available
    assert not off.cruiseState.enabled
    tja_off = h.step(tja=1)
    assert not tja_off.cruiseState.available
    assert not tja_off.cruiseState.enabled
    h.step(tja=0)

    armed = h.step(acc_off=1)
    assert armed.cruiseState.available
    assert not armed.cruiseState.enabled
    tja_armed = h.step(tja=1, acc_off=1)
    assert tja_armed.cruiseState.available
    assert not tja_armed.cruiseState.enabled
    h.step(acc_off=1)

    active = h.step(acc_off=1, acc_active=1)
    assert active.cruiseState.available
    assert active.cruiseState.enabled
    tja_active = h.step(tja=1, acc_off=1, acc_active=1)
    assert tja_active.cruiseState.available
    assert tja_active.cruiseState.enabled

  def test_stock_long_tja_does_not_override_crz_ctrl(self):
    h = ButtonHarness(alpha_long=False)
    off = h.step()
    assert not off.cruiseState.available
    assert not off.cruiseState.enabled
    tja_off = h.step(tja=1)
    assert not tja_off.cruiseState.available
    assert not tja_off.cruiseState.enabled
    _assert_one_lkas_press(tja_off)
    h.step(tja=0)

    armed = h.step(crz_available=1)
    assert armed.cruiseState.available
    assert not armed.cruiseState.enabled
    tja_armed = h.step(tja=1, crz_available=1)
    assert tja_armed.cruiseState.available
    assert not tja_armed.cruiseState.enabled
    h.step(tja=0, crz_available=1)

    active = h.step(crz_available=1, crz_active=1)
    assert active.cruiseState.available
    assert active.cruiseState.enabled
    tja_active = h.step(tja=1, crz_available=1, crz_active=1)
    assert tja_active.cruiseState.available
    assert tja_active.cruiseState.enabled

  def test_mrcc_oem_cruise_parsing_follows_pedals(self):
    h = ButtonHarness()
    h.step()
    # MRCC press with OEM still off: cruise stays off.
    mrcc_off = h.step(mode_x=1, mode_y=1)
    assert not mrcc_off.cruiseState.available
    assert not mrcc_off.cruiseState.enabled
    h.step()
    # OEM arms via PEDALS, with or without the MRCC bit.
    armed = h.step(mode_x=1, mode_y=1, acc_off=1)
    assert armed.cruiseState.available
    assert not armed.cruiseState.enabled
    assert not _events(armed, ButtonType.lkas)
    h.step(acc_off=1)
    active = h.step(mode_x=1, mode_y=1, acc_off=1, acc_active=1)
    assert active.cruiseState.available
    assert active.cruiseState.enabled
    h.step(acc_off=1, acc_active=1)
    off = h.step(mode_x=1, mode_y=1, acc_off=0, acc_active=0)
    assert not off.cruiseState.available
    assert not off.cruiseState.enabled
    assert not _events(off, ButtonType.lkas)

  def test_stock_long_mrcc_does_not_emit_lkas(self):
    h = ButtonHarness(alpha_long=False)
    h.step()
    armed = h.step(mode_x=1, mode_y=1, crz_available=1)
    assert armed.cruiseState.available
    assert not armed.cruiseState.enabled
    assert not _events(armed, ButtonType.lkas)
    active = h.step(mode_x=1, mode_y=1, crz_available=1, crz_active=1)
    assert active.cruiseState.available
    assert active.cruiseState.enabled
    assert not _events(active, ButtonType.lkas)


def _decode_crz_btns(dat):
  cp = CANParser("mazda_2017", [("CRZ_BTNS", float("nan"))], 0)
  cp.update([(0, [(0x09d, dat, 0)])])
  return cp.vl["CRZ_BTNS"]


class TestCreateButtonCmdPreservesWheelBits:
  def test_zeroing_tja_would_fabricate_an_edge(self):
    # Failure path: OP cancel/resume historically packed TJA_BUTTON=0. If that TX
    # is visible on bus 0 (OEM always; CarState if looped), a held TJA becomes 1→0→1.
    h = ButtonHarness()
    h.step()
    h.step(tja=1)
    fake_zero = _crz_btns(h.packer, tja=0, can_off=1)
    cs = h.ci.update([(h.t + 10_000_000, [fake_zero])])[0]
    falling = _events(cs, ButtonType.lkas)
    assert len(falling) == 1 and not falling[0].pressed
    restored = h.step(tja=1)
    rising = _events(restored, ButtonType.lkas)
    assert len(rising) == 1 and rising[0].pressed

  def test_op_cancel_preserves_held_tja_and_mode(self):
    ci = _ci()
    packer = CANPacker("mazda_2017")
    cs_src = type("CS", (), {"tja_button": 1, "mode_x": 1, "mode_y": 0, "crz_btns_counter": 3})()
    addr, dat, bus = mazdacan.create_button_cmd(packer, ci.CP, 3, Buttons.CANCEL, cs_src)
    assert addr == 0x09d and bus == 0
    decoded = _decode_crz_btns(dat)
    assert decoded["CAN_OFF"] == 1
    assert decoded["TJA_BUTTON"] == 1
    assert decoded["MODE_X"] == 1
    assert decoded["MODE_Y"] == 0
    assert decoded["RES"] == 0

    # Same TX fed to CarState while TJA is already held: no extra lkas press.
    h = ButtonHarness()
    h.step()
    h.step(tja=1)
    cs = h.ci.update([(h.t + 10_000_000, [(addr, dat, bus)])])[0]
    assert not any(be.pressed for be in _events(cs, ButtonType.lkas))
    assert h.ci.CS.tja_button == 1

  def test_op_resume_idle_wheel_stays_tja_zero(self):
    ci = _ci()
    packer = CANPacker("mazda_2017")
    cs_src = type("CS", (), {"tja_button": 0, "mode_x": 0, "mode_y": 0, "crz_btns_counter": 1})()
    _, dat, _ = mazdacan.create_button_cmd(packer, ci.CP, 1, Buttons.RESUME, cs_src)
    decoded = _decode_crz_btns(dat)
    assert decoded["RES"] == 1
    assert decoded["TJA_BUTTON"] == 0
    assert decoded["CAN_OFF"] == 0


class TestTjaMadsPlatformScoping:
  """Every TJA/MADS-independence path must stay inactive without MazdaSafetyFlags.TJA_MADS.

  Negative platforms:
  - CX-9 2021: supported, alpha-long capable, no TJA_MADS
  - older CX-5 + steer-to-zero EPS swap: STEER_TO_ZERO without TJA_MADS
  """

  SWAPPED_EPS_FW = sorted(STEER_TO_ZERO_EPS_FW)[0]

  @staticmethod
  def _eps_fw(version: bytes):
    fw = structs.CarParams.CarFw()
    fw.ecu = structs.CarParams.Ecu.eps
    fw.address = 0x730
    fw.subAddress = 0
    fw.fwVersion = version
    return [fw]

  @staticmethod
  def _ci(candidate, *, car_fw=None, alpha_long=False):
    fingerprint = gen_empty_fingerprint()
    CP = CarInterface.get_params(candidate, fingerprint, car_fw or [], alpha_long=alpha_long,
                                 is_release=False, docs=False)
    CP_SP = CarInterface.get_params_sp(CP, candidate, fingerprint, car_fw or [],
                                       alpha_long=alpha_long, is_release_sp=False, docs=False)
    return CarInterface(CP, CP_SP)

  @classmethod
  def _non_tja_platforms(cls):
    yield "cx9_2021", cls._ci(CAR.MAZDA_CX9_2021, alpha_long=False)
    yield "cx5_eps_swap", cls._ci(CAR.MAZDA_CX5, car_fw=cls._eps_fw(cls.SWAPPED_EPS_FW), alpha_long=False)

  def test_flags_are_platform_scoped(self):
    cx5_2022 = self._ci(CAR.MAZDA_CX5_2022).CP
    assert cx5_2022.safetyConfigs[0].safetyParam & MazdaSafetyFlags.TJA_MADS
    assert cx5_2022.safetyConfigs[0].safetyParam & MazdaSafetyFlags.STEER_TO_ZERO

    for name, ci in self._non_tja_platforms():
      param = ci.CP.safetyConfigs[0].safetyParam
      assert not (param & MazdaSafetyFlags.TJA_MADS), name
      if name == "cx5_eps_swap":
        assert param & MazdaSafetyFlags.STEER_TO_ZERO, name
      else:
        assert not (param & MazdaSafetyFlags.STEER_TO_ZERO), name

  def test_non_tja_emits_maincruise_not_lkas(self):
    for name, ci in self._non_tja_platforms():
      packer = CANPacker("mazda_2017")
      t = [0]

      def step(ci=ci, packer=packer, t=t, *, mode_x=0, mode_y=0, tja=0):
        t[0] += 10_000_000
        crz = packer.make_can_msg("CRZ_BTNS", 0, {
          "TJA_BUTTON": tja, "MODE_X": mode_x, "MODE_Y": mode_y, "BIT1": 1, "BIT2": 1, "BIT3": 1,
        })
        return ci.update([(t[0], [crz])])[0]

      step()
      cs = step(mode_x=1, mode_y=1)
      assert _events(cs, ButtonType.mainCruise), name
      assert not _events(cs, ButtonType.lkas), name

      # Held / released / second TJA must never become ButtonType.lkas.
      for tja in (1, 1, 0, 1):
        cs = step(tja=tja)
        assert not _events(cs, ButtonType.lkas), name

  def test_non_tja_op_crz_btns_keeps_upstream_tja_mode_zeros(self):
    for name, ci in self._non_tja_platforms():
      packer = CANPacker("mazda_2017")
      # Even if CarState would report live TJA/MODE high, OP TX must stay upstream zeros.
      cs_src = type("CS", (), {"tja_button": 1, "mode_x": 1, "mode_y": 1, "crz_btns_counter": 3})()
      _, dat, _ = mazdacan.create_button_cmd(packer, ci.CP, 3, Buttons.CANCEL, cs_src)
      decoded = _decode_crz_btns(dat)
      assert decoded["CAN_OFF"] == 1, name
      assert decoded["TJA_BUTTON"] == 0, name
      assert decoded["MODE_X"] == 0, name
      assert decoded["MODE_Y"] == 0, name
      assert decoded["MODE_X_INV"] == 1, name
      assert decoded["MODE_Y_INV"] == 1, name
