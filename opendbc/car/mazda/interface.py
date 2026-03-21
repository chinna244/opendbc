#!/usr/bin/env python3
from opendbc.car import get_safety_config, structs
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.interfaces import CarInterfaceBase
from opendbc.car.mazda.carcontroller import CarController
from opendbc.car.mazda.carstate import CarState
from opendbc.car.mazda.values import CAR, LKAS_LIMITS
from opendbc.sunnypilot.car.mazda.interface_ext import CarInterfaceExt


class CarInterface(CarInterfaceBase):
  CarState = CarState
  CarController = CarController

  def __init__(self, CP, CP_SP):
    super().__init__(CP, CP_SP)
    self._ext = CarInterfaceExt(CP, self)

  @property
  def v_ego(self):
    return self._ext.v_ego

  @v_ego.setter
  def v_ego(self, value):
    self._ext.v_ego = value

  def torque_from_lateral_accel(self):
    if self._ext.speed_dep:
      return self._ext.torque_from_lateral_accel_speed_dep_closure
    return self.torque_from_lateral_accel_linear

  def lateral_accel_from_torque(self):
    if self._ext.speed_dep:
      return self._ext.lateral_accel_from_torque_speed_dep_closure
    return self.lateral_accel_from_torque_linear

  def torque_from_lateral_accel_in_torque_space(self):
    return self._ext.torque_from_lateral_accel_in_torque_space()

  def update_speed_dep_laf(self, speed_bp, laf_bp, friction_bp, valid_bp):
    self._ext.update_speed_dep_laf(speed_bp, laf_bp, friction_bp, valid_bp)

  @staticmethod
  def _get_params(ret: structs.CarParams, candidate, fingerprint, car_fw, alpha_long, is_release, docs) -> structs.CarParams:
    ret.brand = "mazda"
    ret.safetyConfigs = [get_safety_config(structs.CarParams.SafetyModel.mazda)]
    ret.radarUnavailable = True

    ret.dashcamOnly = candidate not in (CAR.MAZDA_CX5_2022, CAR.MAZDA_CX9_2021)

    ret.steerActuatorDelay = 0.1
    ret.steerLimitTimer = 0.8

    CarInterfaceBase.configure_torque_tune(candidate, ret.lateralTuning)

    if candidate not in (CAR.MAZDA_CX5_2022,):
      ret.minSteerSpeed = LKAS_LIMITS.DISABLE_SPEED * CV.KPH_TO_MS

    ret.centerToFront = ret.wheelbase * 0.41

    return ret

  @staticmethod
  def _get_params_sp(stock_cp: structs.CarParams, ret: structs.CarParamsSP, candidate, fingerprint: dict[int, dict[int, int]],
                     car_fw: list[structs.CarParams.CarFw], alpha_long: bool, is_release_sp: bool, docs: bool) -> structs.CarParamsSP:
    ret.intelligentCruiseButtonManagementAvailable = True

    return ret
