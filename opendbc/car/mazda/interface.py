#!/usr/bin/env python3
from opendbc.car import Bus, get_safety_config, structs
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.interfaces import CarInterfaceBase
from opendbc.car.mazda.carcontroller import CarController
from opendbc.car.mazda.carstate import CarState
from opendbc.car.mazda.longitudinal import enter_radar_programming_session, request_radar_default_session
from opendbc.car.mazda.radar_interface import RadarInterface
from opendbc.car.mazda.values import CAR, DBC, LKAS_LIMITS, STEER_TO_ZERO_EPS_FW, MazdaSafetyFlags


class CarInterface(CarInterfaceBase):
  CarState = CarState
  CarController = CarController
  RadarInterface = RadarInterface

  @staticmethod
  def _get_params(ret: structs.CarParams, candidate, fingerprint, car_fw, alpha_long, is_release, docs) -> structs.CarParams:
    ret.brand = "mazda"
    ret.safetyConfigs = [get_safety_config(structs.CarParams.SafetyModel.mazda)]

    ret.alphaLongitudinalAvailable = candidate == CAR.MAZDA_CX5_2022
    ret.openpilotLongitudinalControl = alpha_long and ret.alphaLongitudinalAvailable
    if ret.openpilotLongitudinalControl:
      ret.safetyConfigs[0].safetyParam |= MazdaSafetyFlags.LONG.value
      # engagement stays with the car: the driver SETs on the wheel, the body ECU raises
      # PEDALS.ACC_ACTIVE, and the dash-owned CRZ_EVENTS setpoint survives the radar teardown
      ret.pcmCruise = True
      ret.radarUnavailable = True
      ret.stopAccel = -1.0  # stock MRCC holds raw -1024 (-1.024 m/s2) at a stop
      ret.longitudinalActuatorDelay = 0.36  # measured ~0.3 s dead time + ~0.3 s first-order lag
    else:
      ret.radarUnavailable = Bus.radar not in DBC[candidate]

    ret.dashcamOnly = candidate not in (CAR.MAZDA_CX5_2022, CAR.MAZDA_CX9_2021)

    ret.enableBsm = 0x477 in fingerprint[0]

    ret.steerActuatorDelay = 0.1
    if candidate in (CAR.MAZDA_CX5_2022,):
      ret.steerActuatorDelay = 0.14  # lagd learns 0.338 total (initial = this + 0.2)
    ret.steerLimitTimer = 0.8

    CarInterfaceBase.configure_torque_tune(candidate, ret.lateralTuning)

    # 2022+ CX-5 EPS can steer to zero; detect by EPS firmware so an EPS
    # swapped into another Mazda keeps full-speed steering.
    steer_to_zero = candidate == CAR.MAZDA_CX5_2022 or \
      any(fw.ecu == 'eps' and fw.fwVersion in STEER_TO_ZERO_EPS_FW for fw in car_fw)
    if not steer_to_zero:
      ret.minSteerSpeed = LKAS_LIMITS.DISABLE_SPEED * CV.KPH_TO_MS

    ret.centerToFront = ret.wheelbase * 0.41

    return ret

  @staticmethod
  def _get_params_sp(stock_cp: structs.CarParams, ret: structs.CarParamsSP, candidate, fingerprint: dict[int, dict[int, int]],
                     car_fw: list[structs.CarParams.CarFw], alpha_long: bool, is_release_sp: bool, docs: bool) -> structs.CarParamsSP:
    ret.intelligentCruiseButtonManagementAvailable = True

    return ret

  @staticmethod
  def init(CP, CP_SP, can_recv, can_send):
    if CP.openpilotLongitudinalControl:
      # Failure leaves the stock radar broadcasting; carstate keeps accFaulted set while
      # stock CRZ_INFO is heard, so longitudinal cannot engage against a live radar.
      enter_radar_programming_session(can_recv, can_send)

  @staticmethod
  def deinit(CP, can_recv, can_send):
    # nothing calls this today; the radar also recovers on its own via the UDS S3
    # timeout a few seconds after tester present stops
    if CP.openpilotLongitudinalControl:
      request_radar_default_session(can_recv, can_send)
