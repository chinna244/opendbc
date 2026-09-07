#!/usr/bin/env python3
from opendbc.car import Bus, get_safety_config, structs
from opendbc.car.carlog import carlog
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.interfaces import CarInterfaceBase
from opendbc.car.mazda.carcontroller import CarController
from opendbc.car.mazda.carstate import CarState
from opendbc.car.mazda.fingerprints import FW_VERSIONS
from opendbc.car.mazda.radar_interface import RadarInterface
from opendbc.car.mazda.values import CAR, DBC, LKAS_LIMITS, REPLAY_RADAR_DIALECTS, STEER_TO_ZERO_EPS_FW, MazdaFlags, MazdaSafetyFlags

# Radar firmware whose bus publishes the 0x361-0x366 track dialect: every radar the
# database lists except the registered replay dialects, which it lists for fingerprinting
# even though those radars never send tracks on bus 0. Stored null-stripped so UDS
# response padding of any length compares equal.
TRACK_RADAR_FW = {fw.rstrip(b'\x00') for fw in set().union(
  *(fw.get((structs.CarParams.Ecu.fwdRadar, 0x764, None), []) for fw in FW_VERSIONS.values())
)} - frozenset().union(*(d.fw for d in REPLAY_RADAR_DIALECTS))


class CarInterface(CarInterfaceBase):
  CarState = CarState
  CarController = CarController
  RadarInterface = RadarInterface

  @staticmethod
  def _get_params(ret: structs.CarParams, candidate, fingerprint, car_fw, alpha_long, is_release, docs) -> structs.CarParams:
    ret.brand = "mazda"
    ret.safetyConfigs = [get_safety_config(structs.CarParams.SafetyModel.mazda)]

    # A talking radar outside the track dialects sends no tracks we can parse: run
    # vision-only instead of starving radarTracks behind a parser that never goes valid.
    foreign_radar = any(fw.ecu == 'fwdRadar' and fw.fwVersion.rstrip(b'\x00') not in TRACK_RADAR_FW for fw in car_fw)
    ret.radarUnavailable = Bus.radar not in DBC[candidate] or foreign_radar

    # Every gen1 Mazda EPS is the same hardware; only the firmware differs. Steer-to-zero follows
    # the EPS firmware, so a donor-EPS swap carries it and older firmware in a 2022 body loses it.
    # Only an unread EPS (docs, a failed query) falls back to the platform: a forced CX-5 2022
    # fingerprint on an unlisted older EPS then gets the floor and its banner, not a silent latch.
    eps_fw = {fw.fwVersion for fw in car_fw if fw.ecu == 'eps'}
    steer_to_zero = bool(eps_fw & STEER_TO_ZERO_EPS_FW) or (not eps_fw and candidate == CAR.MAZDA_CX5_2022)
    if steer_to_zero:
      # Select panda's matching torque envelope from the detected EPS.
      ret.flags |= MazdaFlags.STEER_TO_ZERO_EPS.value
      ret.safetyConfigs[0].safetyParam |= MazdaSafetyFlags.STEER_TO_ZERO_EPS.value
    else:
      # Same envelope and tune; only the firmware's floor, latch semantics and alpha long differ.
      ret.minSteerSpeed = LKAS_LIMITS.DISABLE_SPEED * CV.KPH_TO_MS
      ret.flags |= MazdaFlags.LEGACY_FW_EPS.value
      ret.safetyConfigs[0].safetyParam |= MazdaSafetyFlags.LEGACY_FW_EPS.value

    # Resolve the detected radar to a registered replay dialect (mazdacan.py replays
    # its own wire behavior, not the 2022 captures).
    radar_fw = {fw.fwVersion.rstrip(b'\x00') for fw in car_fw if fw.ecu == 'fwdRadar'}
    dialect = next((d for d in REPLAY_RADAR_DIALECTS if radar_fw & d.fw), None)
    if dialect is not None:
      ret.flags |= int(dialect.flag)

    # Alpha-long silences the radar and stands in for it, so it needs the radar's dialect,
    # not its tracks: offer it wherever the platform's radar speaks the 2022 family dialect
    # (its DBC claims a radar bus) or the detected radar has a registered replay dialect.
    # The EPS gate stays: a stock older EPS cuts lateral below 45 kph, so stop-and-go would
    # run unsteered.
    ret.alphaLongitudinalAvailable = steer_to_zero and (Bus.radar in DBC[candidate] or dialect is not None)
    ret.openpilotLongitudinalControl = alpha_long and ret.alphaLongitudinalAvailable
    if ret.openpilotLongitudinalControl:
      ret.safetyConfigs[0].safetyParam |= MazdaSafetyFlags.LONG.value
      # The car owns engagement and preserves its setpoint through radar teardown.
      ret.pcmCruise = True
      ret.radarUnavailable = True
      ret.stopAccel = -1.024  # stock MRCC standstill command
      ret.longitudinalActuatorDelay = 0.36  # measured ~0.3 s dead time + ~0.3 s first-order lag

    # Older EPS firmware enforces hands-off and low-speed steering lockouts.
    # Docs mode carries no real EPS firmware, so leave dashcamOnly at the default.
    if not docs:
      ret.dashcamOnly = candidate not in (CAR.MAZDA_CX5_2022, CAR.MAZDA_CX9_2021) and not steer_to_zero

    carlog.info({"event": "mazdaRadarVerdict", "radarUnavailable": ret.radarUnavailable,
                 "platformClaim": Bus.radar in DBC[candidate], "foreignRadarFw": foreign_radar,
                 "replayDialect": dialect.name if dialect is not None else None, "steerToZeroEps": steer_to_zero})

    ret.enableBsm = 0x477 in fingerprint[0]

    # Command-to-torque lag measured on the EPS hardware; lagd learns the remaining delay.
    ret.steerActuatorDelay = 0.14
    ret.steerLimitTimer = 0.8

    CarInterfaceBase.configure_torque_tune(candidate, ret.lateralTuning)

    ret.centerToFront = ret.wheelbase * 0.41

    return ret

  @staticmethod
  def _get_params_sp(stock_cp: structs.CarParams, ret: structs.CarParamsSP, candidate, fingerprint: dict[int, dict[int, int]],
                     car_fw: list[structs.CarParams.CarFw], alpha_long: bool, is_release_sp: bool, docs: bool) -> structs.CarParamsSP:
    ret.intelligentCruiseButtonManagementAvailable = True

    return ret
