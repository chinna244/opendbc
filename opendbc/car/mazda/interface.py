#!/usr/bin/env python3
from opendbc.car import Bus, get_safety_config, structs
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.interfaces import CarInterfaceBase
from opendbc.car.mazda.carcontroller import CarController
from opendbc.car.mazda.carstate import CarState
from opendbc.car.mazda.radar_interface import RadarInterface
from opendbc.car.mazda.values import CAR, DBC, LKAS_LIMITS, STEER_TO_ZERO_EPS_FW, MazdaSafetyFlags

CAM_LANEINFO_ADDR = 0x440
CAM_LANEINFO_RX_BUS = 2


def normalize_can_packets(can_packets):
  """card passes a list of (t, frames) batches; test_models passes one batch tuple."""
  if isinstance(can_packets, tuple) and len(can_packets) == 2 and isinstance(can_packets[0], int):
    return [can_packets]
  return list(can_packets)


def latch_cam_laneinfo_raw(can_packets, prev: bytes | None) -> tuple[bytes | None, bool]:
  """Return the last valid FSC CAM_LANEINFO payload and whether one arrived this cycle."""
  raw = prev
  received = False
  for _t, frames in normalize_can_packets(can_packets):
    for msg in frames:
      addr, dat, src = (msg.address, msg.dat, msg.src) if hasattr(msg, "address") else msg
      if addr == CAM_LANEINFO_ADDR and src == CAM_LANEINFO_RX_BUS and len(dat) == 8:
        raw = bytes(dat)
        received = True
  return raw, received


class CarInterface(CarInterfaceBase):
  CarState = CarState
  CarController = CarController
  RadarInterface = RadarInterface

  def update(self, can_packets):
    can_packets = normalize_can_packets(can_packets)
    raw, received = latch_cam_laneinfo_raw(can_packets, self.CS.cam_laneinfo_raw)
    self.CS.cam_laneinfo_raw = raw
    self.CS.cam_laneinfo_stale_frames = 0 if received else self.CS.cam_laneinfo_stale_frames + 1
    return super().update(can_packets)

  @staticmethod
  def _get_params(ret: structs.CarParams, candidate, fingerprint, car_fw, alpha_long, is_release, docs) -> structs.CarParams:
    ret.brand = "mazda"
    ret.safetyConfigs = [get_safety_config(structs.CarParams.SafetyModel.mazda)]

    ret.radarUnavailable = Bus.radar not in DBC[candidate]

    # 2022+ CX-5 EPS can steer to zero and has no hands-off lockout. Detected by EPS firmware
    # rather than by model, so an EPS swapped into an older Mazda is recognized as what it is.
    steer_to_zero = candidate == CAR.MAZDA_CX5_2022 or \
      any(fw.ecu == 'eps' and fw.fwVersion in STEER_TO_ZERO_EPS_FW for fw in car_fw)
    if not steer_to_zero:
      ret.minSteerSpeed = LKAS_LIMITS.DISABLE_SPEED * CV.KPH_TO_MS

    if steer_to_zero:
      ret.safetyConfigs[0].safetyParam |= MazdaSafetyFlags.STEER_TO_ZERO.value
    # Physical TJA as MADS is verified on CX-5 2022. An EPS swap does not prove the
    # same TJA button or CRZ_BTNS layout, so TJA_MADS stays platform-scoped.
    if candidate == CAR.MAZDA_CX5_2022:
      ret.safetyConfigs[0].safetyParam |= MazdaSafetyFlags.TJA_MADS.value

    # CX-9 2021 verified against route 00000004--97e4328f4f: same message set at the same
    # rates, CRZ_INFO checksum holds on all 54k stock frames, radar UDS at 0x764, and the
    # same FSC camera firmware (GSH7-67XK2-U) as the CX-5 2022 this was developed on.
    ret.alphaLongitudinalAvailable = candidate in (CAR.MAZDA_CX5_2022, CAR.MAZDA_CX9_2021)
    ret.openpilotLongitudinalControl = alpha_long and ret.alphaLongitudinalAvailable
    if ret.openpilotLongitudinalControl:
      ret.safetyConfigs[0].safetyParam |= MazdaSafetyFlags.LONG.value
      # engagement stays with the car: the driver SETs on the wheel, the body ECU raises
      # PEDALS.ACC_ACTIVE, and the dash-owned CRZ_EVENTS setpoint survives the radar teardown
      ret.pcmCruise = True
      ret.radarUnavailable = True
      ret.stopAccel = -1.024  # stock MRCC holds raw -1024 at a stop; the plan parks here and we send it as-is
      ret.longitudinalActuatorDelay = 0.36  # measured ~0.3 s dead time + ~0.3 s first-order lag

    # Older Mazdas are dashcam only for one reason: their EPS locks steering out after ~5 s of
    # hands-off and below 45 kph. That is a property of the EPS, not of the car, so a car with
    # the 2022+ EPS swapped in is controllable and lifts with it. Longitudinal stays keyed on the
    # model above: the radar and camera are not part of an EPS swap.
    ret.dashcamOnly = candidate not in (CAR.MAZDA_CX5_2022, CAR.MAZDA_CX9_2021) and not steer_to_zero

    ret.enableBsm = 0x477 in fingerprint[0]

    # command-to-torque lag is EPS firmware, so it follows the EPS. lagd learns the rest
    # (0.338 total on a CX-5 2022; initial = this + 0.2)
    ret.steerActuatorDelay = 0.14 if steer_to_zero else 0.1
    ret.steerLimitTimer = 0.8

    CarInterfaceBase.configure_torque_tune(candidate, ret.lateralTuning)

    ret.centerToFront = ret.wheelbase * 0.41

    return ret

  @staticmethod
  def _get_params_sp(stock_cp: structs.CarParams, ret: structs.CarParamsSP, candidate, fingerprint: dict[int, dict[int, int]],
                     car_fw: list[structs.CarParams.CarFw], alpha_long: bool, is_release_sp: bool, docs: bool) -> structs.CarParamsSP:
    ret.intelligentCruiseButtonManagementAvailable = True

    return ret
