from opendbc.car import structs
from opendbc.car.mazda.values import CAR, MazdaFlags, MazdaSafetyFlags

CARS = frozenset((CAR.MAZDA_CX5_2022, CAR.MAZDA_CX9_2021))


def qualified(cp, *, marked_only=False):
  if not (cp.brand == "mazda" and cp.carFingerprint in CARS and not cp.passive and not cp.notCar and not cp.dashcamOnly and
          cp.transmissionType != structs.CarParams.TransmissionType.manual and
          cp.fingerprintSource != structs.CarParams.FingerprintSource.fixed and cp.pcmCruise and len(cp.safetyConfigs) == 1 and
          cp.safetyConfigs[0].safetyModel == structs.CarParams.SafetyModel.mazda and
          cp.alternativeExperience in ((32,) if marked_only else (0, 32))):
    return False
  flags, word = int(cp.flags), int(cp.safetyConfigs[0].safetyParam)
  # the Smart CAN Filter bit (can_filter.prepare): alpha long on the CX-5 2022 only, which `measured` already requires;
  # MADS lateral is the same with it as without (develop keeps MADS under the filter)
  filtered = bool(word & int(MazdaSafetyFlags.CAN_FILTER))
  word &= ~int(MazdaSafetyFlags.CAN_FILTER)
  if filtered and not cp.openpilotLongitudinalControl:
    return False
  legacy = flags == int(MazdaFlags.GEN1) and word == 0 and not cp.openpilotLongitudinalControl
  known = int(MazdaFlags.GEN1 | MazdaFlags.EPS_HW | MazdaFlags.G46L_RADAR | MazdaFlags.HAS_BSM)
  measured = (flags & ~known == 0 and flags & int(MazdaFlags.GEN1) and
              flags & int(MazdaFlags.EPS_HW) in (int(MazdaFlags.STEER_TO_ZERO_EPS), int(MazdaFlags.LEGACY_FW_EPS)) and
              word in (2, 3, 10, 11) and bool(word & 1) == bool(cp.openpilotLongitudinalControl) and
              (not cp.openpilotLongitudinalControl or
               cp.carFingerprint == CAR.MAZDA_CX5_2022 and bool(flags & int(MazdaFlags.STEER_TO_ZERO_EPS)) and cp.alphaLongitudinalAvailable))
  return bool(legacy or measured)


# The panda grants Mazda lateral only while moving (vehicle_moving, > 0.1 kph). The host withdraws its request a
# little above that (starpilot/car/mazda/aol.py stopped_lateral_inhibit), and the whole band is a restriction, so a
# stop or pull-away with cruise engaged pauses lateral instead of reading as a native denial.
STOPPED_LATERAL_SPEED = 0.3  # m/s


def temporary_restriction(cp, state):
  return bool(state is not None and qualified(cp, marked_only=True) and state.canValid and not state.canTimeout and
              not state.steerFaultPermanent and (state.gearShifter != structs.CarState.GearShifter.drive or
                                                 state.steerFaultTemporary or state.standstill or state.vEgo < STOPPED_LATERAL_SPEED or
                                                 (cp.minSteerSpeed > 0 and state.vEgo < cp.minSteerSpeed)))


def native_observation(native, *, latched, panda_ready, restricted, now_ns, pending_since_ns):
  lost = not panda_ready or native is None
  if not latched or lost or restricted:
    return 0, lost, False
  if native.requestedLateral and native.lateralAllowed:
    return 0, False, False
  pending_since_ns = pending_since_ns or now_ns
  return pending_since_ns, False, now_ns - pending_since_ns >= 200_000_000
