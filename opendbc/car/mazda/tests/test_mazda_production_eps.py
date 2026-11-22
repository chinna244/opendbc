"""Production-path EPS delivery: the unmodified CarInterface.update/apply on the CAN a steer-to-zero
CX-5 sends, with the EPS closing the loop on the torque it was sent. steerFaultTemporary must keep the
source's semantics (docs/zoompilot/mazda-lateral.md, "LKAS_BLOCK is not all-or-nothing"): partial
delivery keeps steering, only sustained road-speed zero delivery is a fault, and the lane-keep re-arm
block is never one. The same holds under StarPilot's native lateral owner (AOL).
"""
from opendbc.can import CANPacker
from opendbc.car import DT_CTRL
from opendbc.car.mazda.carstate import LKAS_REARM_FAULT_FRAMES
from opendbc.car.mazda.interface import CarInterface
from opendbc.car.mazda.tests.conftest import DBC_NAME, car_control, car_params
from opendbc.car.mazda.tests.test_mazda_golden_tx import D, SimCar
from opendbc.car.mazda.values import CAR, CarControllerParams, MazdaFlags

STEER_RATE = 0x241


class DeliveringSimCar(SimCar):
  """SimCar whose EPS delivers a fraction of the request through LKAS_BLOCK, as real >4 m/s blocks do."""

  def __init__(self, delivered=0.):
    super().__init__()
    self.delivered = delivered
    self.steer_packer = CANPacker(DBC_NAME)

  def can(self):
    msgs = super().can()
    s = self.state
    effective = round(self.eps_request * self.delivered / 4) * 4 if s["lkas_block"] else self.eps_request
    return [self.steer_packer.make_can_msg("STEER_RATE", 0, {
              "LKAS_REQUEST": self.eps_request, "LKAS_EFFECTIVE": effective,
              "LKAS_BLOCK": s["lkas_block"], "LKAS_TRACK_STATE": s["lkas_track"]})
            if addr == STEER_RATE else (addr, dat, bus) for addr, dat, bus in msgs]


def production_interface(aol=False) -> CarInterface:
  cp = car_params(CAR.MAZDA_CX5_2022)
  assert cp.flags & MazdaFlags.STEER_TO_ZERO_EPS
  if aol:
    cp.alternativeExperience = 32  # card marks the CP it shares with the interface
  return CarInterface(cp)


def drive(ci, sim, frames, *, torque=0.3, **state):
  """Engaged at 15 m/s: carstate parses, the controller answers, the EPS sees what was sent."""
  sim.state.update(dict(v=15., gear=D, brake=False, armed=True, active=True), **state)
  out = []
  for _ in range(frames):
    t = int(sim.frame * DT_CTRL * 1e9)
    cs = ci.update([(t, sim.can())])
    actuators, sends = ci.apply(car_control(enabled=True, long_active=False, lat_active=True, torque=torque), t)
    out.append((cs.steerFaultTemporary, int(actuators.torqueOutputCan), ci.CS.lkas_blocked))
    sim.react(sends, [True] * len(sends))
  return out


def settled(ci, sim):
  drive(ci, sim, 200, lkas_block=False)


def test_lane_keep_reenable_at_15_mps_is_not_a_fault():
  # the review's scenario: lane keep off, then back on while engaged at 15 m/s; the EPS holds its
  # re-arm block for ~3 s whatever it is sent
  ci, sim = production_interface(), DeliveringSimCar()
  settled(ci, sim)
  drive(ci, sim, 100, lane_keep=False)
  back = drive(ci, sim, 305, lane_keep=True, lkas_block=True)
  assert all(blocked for _, _, blocked in back)
  assert not any(fault for fault, _, _ in back[:LKAS_REARM_FAULT_FRAMES])
  assert ci.CS.lkas_arming or not ci.CS.lkas_blocked
  recovered = drive(ci, sim, 100, lane_keep=True, lkas_block=False)
  assert not any(fault for fault, _, _ in recovered)
  assert all(torque != 0 for _, torque, _ in recovered[10:])


def test_partial_delivery_keeps_steering_without_a_fault():
  ci, sim = production_interface(), DeliveringSimCar(delivered=0.4)
  settled(ci, sim)
  blocked = drive(ci, sim, 300, lkas_block=True)
  assert not any(fault for fault, _, _ in blocked)
  assert not ci.CS.steer_undelivered
  # the controller keeps commanding through the partial block
  assert all(abs(torque) > CarControllerParams(ci.CP).STEER_UNDELIVERED_MIN for _, torque, _ in blocked[50:])


def test_sustained_zero_delivery_latches_then_faults_then_recovers():
  ci, sim = production_interface(), DeliveringSimCar(delivered=0.)
  params = CarControllerParams(ci.CP)
  settled(ci, sim)
  blocked = drive(ci, sim, 200, lkas_block=True)
  latch = params.STEER_UNDELIVERED_FRAMES
  alert = latch + params.STEER_UNDELIVERED_ALERT_FRAMES
  assert not any(fault for fault, _, _ in blocked[:alert - 5])
  assert all(fault for fault, _, _ in blocked[alert + 5:])
  # the latch stops commanding the undelivered torque before the camera faults
  assert all(torque == 0 for _, torque, _ in blocked[latch + 5:])
  recovered = drive(ci, sim, 100, lkas_block=False)
  assert not any(fault for fault, _, _ in recovered[2:])
  assert all(torque != 0 for _, torque, _ in recovered[10:])


def test_aol_keeps_steering_through_a_partial_block():
  # AOL takes the source's semantics: a raw block with partial delivery is no fault (route 00000002--29b68afa8a)
  ci, sim = production_interface(aol=True), DeliveringSimCar(delivered=0.4)
  settled(ci, sim)
  assert not any(fault for fault, _, _ in drive(ci, sim, 300, lkas_block=True))
  assert not ci.CS.steer_undelivered


def test_legacy_firmware_under_aol_keeps_the_speed_gated_rule():
  # older EPS firmware raises LKAS_BLOCK below its 45 kph floor by design; AOL's restriction covers it
  cp = car_params(CAR.MAZDA_CX9_2021)
  assert cp.flags & MazdaFlags.LEGACY_FW_EPS
  cp.alternativeExperience = 32
  ci, sim = CarInterface(cp), DeliveringSimCar()
  drive(ci, sim, 200, v=8., lkas_block=False)
  assert not any(fault for fault, _, _ in drive(ci, sim, 100, v=8., lkas_block=True))
