from dataclasses import dataclass, field
from enum import IntFlag, StrEnum

from opendbc.car import Bus, CarSpecs, DbcDict, DT_CTRL, PlatformConfig, Platforms
from opendbc.car.carlog import carlog
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.structs import CarParams
from opendbc.car.docs_definitions import CarHarness, CarDocs, CarParts
from opendbc.car.fw_query_definitions import FwQueryConfig, Request, StdQueries
from opendbc.car.vin import Vin, is_valid_vin

Ecu = CarParams.Ecu


# Steer torque limits

class CarControllerParams:
  # The measured EPS envelope, mazda.h's MAZDA_EPS_HW_STEERING_LIMITS: the panda's safety tests take
  # their numbers from here. Evidence for every constant below: the Constants tables in
  # docs/zoompilot/mazda-lateral.md and mazda-longitudinal.md.
  EPS_STEER_MAX = 1200            # theoretical max_steer 2047
  STEER_DELTA_UP = 12             # torque increase per refresh, the EPS's own slew
  STEER_DELTA_DOWN = 12           # torque decrease per refresh
  STEER_DRIVER_ALLOWANCE = 15     # allowed driver torque before start limiting
  STEER_DRIVER_MULTIPLIER = 15    # weight driver torque, tuned for the 2022 EPS
  STEER_DRIVER_FACTOR = 1         # from dbc
  STEER_STEP = 1  # 100 Hz
  # Upstream's STEER_MAX, the scale params.toml's tunes, NNLC models and the manual torque override
  # use: latAccelFactor x TUNE_SCALE, friction / TUNE_SCALE.
  TUNE_STEER_MAX = 800
  TUNE_SCALE = EPS_STEER_MAX / TUNE_STEER_MAX

  ACCEL_MAX = 2.0   # m/s2
  ACCEL_MIN = -3.5  # m/s2

  # Longitudinal message periods in 100 Hz control frames.
  LONG_STEP = 2        # CRZ_INFO/CRZ_CTRL at 50 Hz, matching stock
  RADAR_STEP = 10      # radar static + track frames at 10 Hz

  FSC_SETTLE_T = 7.0           # the camera's cold-boot radar check settled before the teardown may start
  STOCK_RADAR_ALIVE_T = 0.05   # a normal CRZ_INFO gap; it does not establish ownership
  STOCK_RADAR_GUARD_T = 1.27   # silence before ownership is trusted, ~12x the longest stock gap
  CAM_LANEINFO_FRESH_T = 1.5   # longer than one ~2 Hz CAM_LANEINFO period

  # The car's lane keep back on, the EPS holds LKAS_BLOCK for ~3 s whatever it is sent.
  LKAS_REARM_T = 3.0           # no lift of the block counts before this
  LKAS_REARM_FAULT_T = 4.0     # the block it raises is not a fault for this long

  # Both PEDALS cruise bits low this many samples (100 Hz) is a main-off; mazda.h's MAZDA_MAIN_OFF_DEBOUNCE.
  MAIN_OFF_DEBOUNCE_SAMPLES = 10
  CANCEL_SETTLE_T = 0.2       # s a cancel request must hold before the first press; the car answers its own inside it

  # Relax the command after the body ECU takes ownership of the brake hold.
  ACCEL_HOLD_LATCHED = -0.001  # m/s2

  # Match stock's ACCEL_CMD ceiling during a latched release pulse.
  ACCEL_RESUME_PULSE_MAX = 0.25  # m/s2, latched releases only

  # Match stock's one-frame relaxation and subsequent release ramp.
  ACCEL_RELEASE_BAND = -0.26  # m/s2, the one-frame relax target at a never-latched release
  ACCEL_RELEASE_RAMP = 1.25   # m/s3, stock's release ramp (+25 raw per 50 Hz frame)
  # Past the unlatch pulse window with the car still standing, build faster than stock: stock's
  # ramp alone took 0.8 s to break away after the hold opened (routes 0000013a, 0000013b).
  ACCEL_RELEASE_RAMP_STOPPED = 2.5  # m/s3

  # Permit a bounded breakaway ramp because Mazda longitudinal control has no integrator.
  ACCEL_BREAKAWAY_MAX = 1.45  # m/s2, ceiling for the still-stopped release ramp
  ACCEL_BREAKAWAY_OVERSHOOT = 0.75  # m/s2 above the plan the still-stopped ramp may climb

  # Shape positive commands like stock MRCC: its accelerating command's ceiling by speed, and its
  # build rate (taken a third quicker once rolling: the plan sees a lead pull away before the radar
  # would). Positive commands only, so braking is never held longer than the plan asks.
  # The ceiling is held under stock's p99 ([1.5, 1.75, 1.45, 1.05, 0.85, 0.65] at [0, 4, 9, 14, 18, 25]
  # m/s). Up to 1.2 m/s2 from a stop, easing from there: the engaged pull-aways of routes 00000160 and
  # 00000163 took 6 to 7.5 s from 6 to 25 mph at 1.1 to 1.4 m/s2 mean, this driver's own took 8 to 23 s
  # (median 11); this table takes about 9 to 10 s.
  # From 40 mph up it is held at 0.55: in routes 00000160 and 00000163 every engaged downshift of two gears
  # (5th to 4th, 4th to 3rd) came at a command of 0.61 or more and every one-gear downshift at 0.54 or less,
  # and holding 0.55 gives up about 6 mph over 19 engaged minutes.
  ACCEL_CEILING_BP = [0., 4., 5.5, 7., 9., 11., 13.5, 18., 25.]  # m/s
  ACCEL_CEILING_V = [1.2, 1.2, 1.1, 1.0, 0.9, 0.8, 0.75, 0.55, 0.55]  # m/s2
  # Under the plan the transmission takes 4th at 19-20 mph and about 1600 rpm (this driver's own 3rd to
  # 4th: 23-26 mph, about 2000 rpm). In routes 00000164-00000166 both 4th-to-3rd kickdowns within 3 s of
  # such an upshift came at a command of 0.80, and every upshift held at 0.53-0.61 stayed in 4th. For
  # UPSHIFT_HOLD_FRAMES after an upshift into UPSHIFT_HOLD_MIN_GEAR or above under UPSHIFT_HOLD_SPEED the
  # ceiling is UPSHIFT_HOLD_ACCEL, reached at the lift rate. Raised from 0.6 with FOURTH_GEAR_ACCEL: together with the
  # planner floor they held town pickups to 0.6 (20-28 mph 6.2 s against this driver's own 5.3 s), and a useful
  # kickdown under a steady push is allowed. At the 4th-gear value the hold adds nothing in 4th or 5th.
  UPSHIFT_HOLD_ACCEL = 0.75  # m/s2
  UPSHIFT_HOLD_FRAMES = int(3.0 / DT_CTRL)
  UPSHIFT_HOLD_SPEED = 11.2  # m/s, 25 mph
  UPSHIFT_HOLD_MIN_GEAR = 4
  # Settled in 4th the transmission still kicks down to 3rd under a sustained push: in routes 00000160-00000167,
  # engaged in 4th under 30 mph, 3 of 9 pushes peaking at 0.75 or more kicked down (route 00000167: 0.79 at 27
  # mph, 1485 to 2283 rpm) and none of 7 peaking at 0.70 or less did. Under FOURTH_GEAR_SPEED in 4th the
  # ceiling is FOURTH_GEAR_ACCEL. At 0.75 (routes 00000147-00000177): 23 of 25 pushes at 0.75 or less stayed in
  # 4th; of 5 at 0.75 or more, 4 shifted, each followed by a steady pull and none shifting down twice in one pull.
  FOURTH_GEAR_ACCEL = 0.75  # m/s2
  FOURTH_GEAR_SPEED = 13.4  # m/s, 30 mph
  # In 6th the transmission drops to 5th under a small push and comes back once it eases: in routes 00000160 and
  # 00000163 every engaged 6->5 came from a push of 0.23 or more (60-100% of pushes from 0.20 up) and none of 81
  # under 0.20 did; 20 of the 24 engaged highway hunts were the set-speed or gap boost. This driver's own push in
  # 6th is 0.17 at p90. In 6th the ceiling is SIXTH_GEAR_ACCEL.
  SIXTH_GEAR_ACCEL = 0.2  # m/s2
  # A plan that asks SIXTH_KICK_ACCEL or more for SIXTH_KICK_FRAMES gets one deliberate downshift, as a driver
  # flooring it to pass would; it lasts until the plan asks less than SIXTH_KICK_RELEASE or the gear changes.
  # Back in 6th the cap holds for SIXTH_KICK_COOLDOWN_FRAMES, so a kick can never turn into a hunt. Routes
  # 00000160 and 00000163: none in 16 min engaged in 6th. It sits above the set-speed floor's highway ceiling (0.5
  # from 56 mph; the floor alone can reach it below about 52 mph): this driver catches up in 6th under 0.17 at p90 and
  # almost never kicks down. With the floor at 0.8, a replay of the same 16 min still had no kick.
  SIXTH_KICK_ACCEL = 0.55  # m/s2
  SIXTH_KICK_RELEASE = 0.25  # m/s2
  SIXTH_KICK_FRAMES = int(1.0 / DT_CTRL)
  SIXTH_KICK_COOLDOWN_FRAMES = int(15.0 / DT_CTRL)
  # In 5th the transmission drops to 4th and back under a firm push: routes 00000160 and 00000163, all four
  # engaged 5-4-5 hunts (38-60 mph) came from pushes of 0.59-0.75, and pushes up to 0.58 held. In 5th the
  # ceiling is FIFTH_GEAR_ACCEL, the highway ceiling at any speed.
  FIFTH_GEAR_ACCEL = 0.55  # m/s2
  ACCEL_BUILD_BP = [3., 6.]   # m/s
  ACCEL_BUILD_V = [1.25, 0.8]  # m/s3
  # A rolling gas override zeroes the command, and the plan comes back at the car's own acceleration: the build rate
  # alone took 0.6-1.0 s to catch up, the car sagging from +0.63 to +0.11 m/s2 (median of 29 rolling returns, routes
  # 0000016d-00000175). With no lead shown and the car still pulling, the command climbs at HANDOVER_BUILD up to that
  # pull, at most HANDOVER_MAX, under the same ceilings. Only an override while engaged counts. A lead, a brake request
  # or the end of HANDOVER_FRAMES ends the climb. At any time, a brake request of HANDOVER_BRAKE or more, or any slowing
  # with a lead shown, is answered with no more than the shaper without the climb would send, so braking is never later
  # than before, at the window's end or after it; with no lead, a plan only easing off lifts at the usual rates.
  HANDOVER_BUILD = 4.0  # m/s3
  HANDOVER_BRAKE = -0.3  # m/s2
  HANDOVER_MAX = 0.6  # m/s2
  HANDOVER_MIN_SPEED = 3.0  # m/s
  HANDOVER_FRAMES = int(1.0 / DT_CTRL)
  # Stock's throttle lift rate, while the plan is still >= 0; a brake request skips it.
  ACCEL_LIFT_LIMIT = -2.0  # m/s3
  # Limit upward plan-command slew in the brake region without delaying braking response.
  ACCEL_WINDUP_LIMIT = 4.0 * DT_CTRL     # m/s2 per frame
  ACCEL_WINDDOWN_LIMIT = -10.0 * DT_CTRL  # m/s2 per frame, clips only the p99.9+ steps

  def __init__(self, CP):
    # Use a sample window and margin to stay inside panda's fresher driver-torque envelope.
    self.STEER_DRIVER_SAMPLES = 10
    self.STEER_DRIVER_MARGIN = 2

    # One scale at every speed keeps the learned torque parameters in one unit (docs/zoompilot/lateral-tune.md).
    self.STEER_MAX = self.EPS_STEER_MAX
    # Clamp to the measured applied-torque ceiling so controlsd can detect saturation.
    self.EPS_CEILING_LOOKUP = ([8.0, 8.5, 9.4, 10.3, 11.2, 12.1, 13.0, 13.9, 14.5],
                               [1148, 1132, 1092, 1048, 1012,  920,  808,  676,  620])

    if CP.flags & MazdaFlags.STEER_TO_ZERO_EPS:
      # Stop commanding after sustained zero delivery to avoid a camera steering fault. Use
      # LKAS_EFFECTIVE because LKAS_BLOCK may still permit partial delivery.
      self.STEER_UNDELIVERED_MIN = 200      # counts; below this the EPS rounds to zero anyway
      self.STEER_UNDELIVERED_FRAMES = 20    # 200 ms at 100 Hz

      # Alert only after sustained non-delivery above maneuvering speed. Suppress normal
      # low-speed standby blocks identified by LKAS_TRACK_STATE.
      self.STEER_UNDELIVERED_ALERT_FRAMES = 80    # 0.8 s at 100 Hz, on top of the latch's 0.2
      self.STEER_UNDELIVERED_ALERT_MIN_SPEED = 12. * CV.MPH_TO_MS
      # A block that began below this speed is the EPS's standby from a stop; only one that began
      # rolling can be a dropout. carstate's first-engagement hold uses the same boundary.
      self.STEER_UNDELIVERED_ALERT_ORIGIN_SPEED = 1.0  # m/s


@dataclass
class MazdaCarDocs(CarDocs):
  package: str = "All"
  car_parts: CarParts = field(default_factory=CarParts.common([CarHarness.mazda]))


@dataclass(frozen=True, kw_only=True)
class MazdaCarSpecs(CarSpecs):
  tireStiffnessFactor: float = 0.7  # not optimized yet


@dataclass(frozen=True, kw_only=True)
class MazdaCX5_2022CarSpecs(CarSpecs):
  tireStiffnessFactor: float = 1.0


class MazdaFlags(IntFlag):
  # GEN1 platforms share CAN messages and camera hardware.
  GEN1 = 1

  # EPS firmware that steers to zero: no speed floor, the 1200-count low-speed scale, and
  # LKAS_TRACK_STATE standby semantics.
  STEER_TO_ZERO_EPS = 2
  # Every other gen1 Mazda EPS: the same hardware on firmware that keeps the 45 kph floor. Same
  # envelope and tune; the floor, the non-delivery latch and alpha long stay with steer-to-zero.
  LEGACY_FW_EPS = 4
  # Everything keyed on the measured hardware rather than on what the firmware permits.
  EPS_HW = STEER_TO_ZERO_EPS | LEGACY_FW_EPS

  # The G46L radar's dialect bit; see G46L_RADAR_FW below.
  G46L_RADAR = 8


class MazdaSafetyFlags(IntFlag):
  LONG = 1
  # Selects the measured EPS envelope in panda safety; the interface sets it on every gen1 EPS.
  # 4 was the legacy-firmware copy of this bit until 2026-10: do not reuse it, an older car side
  # may still send it.
  EPS_HW = 2


class WMI(StrEnum):
  JAPAN_PASSENGER = "JM1"   # Japan-built passenger cars
  JAPAN_CROSSOVER = "JM3"   # Japan-built crossovers, North America
  EXPORT_CROSSOVER = "JM7"  # Japan-built crossovers, export markets; same chassis and year fields
  MEXICO_PASSENGER = "3MZ"  # Mazda de Mexico (Mazda 3)
  # Export VINs without a model-year field use the EPS-swap fallback.
  OCEANIA_EXPORT = "JM0"


@dataclass
class MazdaPlatformConfig(PlatformConfig):
  dbc_dict: DbcDict = field(default_factory=lambda: {Bus.pt: 'mazda_2017', Bus.radar: 'mazda_2017'})
  flags: int = MazdaFlags.GEN1
  wmis: set[WMI] = field(default_factory=set)
  chassis_codes: set[str] = field(default_factory=set)
  years: set[str] = field(default_factory=set)


class CAR(Platforms):
  MAZDA_CX5_KE = MazdaPlatformConfig(
    [MazdaCarDocs("Mazda CX-5 2012-16")],
    MazdaCarSpecs(mass=3433 * CV.LB_TO_KG, wheelbase=2.7, steerRatio=18.1),  # steer ratio from the 2022 CX-5: same rack hardware
    # This radar does not publish 0x361-0x366 tracks on bus 0.
    dbc_dict={Bus.pt: 'mazda_2017'},
    wmis={WMI.JAPAN_CROSSOVER, WMI.EXPORT_CROSSOVER}, chassis_codes={'KE'}, years={'C', 'D', 'E', 'F', 'G'},  # 2012-16
  )
  MAZDA_CX5 = MazdaPlatformConfig(
    [MazdaCarDocs("Mazda CX-5 2017-21")],
    MazdaCarSpecs(mass=3655 * CV.LB_TO_KG, wheelbase=2.7, steerRatio=18.1),  # steer ratio from the 2022 CX-5: same rack hardware
    wmis={WMI.JAPAN_CROSSOVER, WMI.EXPORT_CROSSOVER}, chassis_codes={'KF'}, years={'H', 'J', 'K', 'L', 'M'},  # 2017-21
  )
  MAZDA_CX9 = MazdaPlatformConfig(
    [MazdaCarDocs("Mazda CX-9 2016-20")],
    MazdaCarSpecs(mass=4217 * CV.LB_TO_KG, wheelbase=2.93, steerRatio=17.6),
    # This radar does not publish 0x361-0x366 tracks on bus 0.
    dbc_dict={Bus.pt: 'mazda_2017'},
    wmis={WMI.JAPAN_CROSSOVER, WMI.EXPORT_CROSSOVER}, chassis_codes={'TC'}, years={'G', 'H', 'J', 'K', 'L'},  # 2016-20
  )
  MAZDA_3 = MazdaPlatformConfig(
    [MazdaCarDocs("Mazda 3 2017-18")],
    MazdaCarSpecs(mass=2875 * CV.LB_TO_KG, wheelbase=2.7, steerRatio=14.0),
    wmis={WMI.JAPAN_PASSENGER, WMI.MEXICO_PASSENGER}, chassis_codes={'BN'}, years={'H', 'J'},  # 2017-18
  )
  MAZDA_6 = MazdaPlatformConfig(
    [MazdaCarDocs("Mazda 6 2017-20")],
    MazdaCarSpecs(mass=3443 * CV.LB_TO_KG, wheelbase=2.83, steerRatio=15.5),
    wmis={WMI.JAPAN_PASSENGER}, chassis_codes={'GL'}, years={'H', 'J', 'K', 'L', 'M'},  # 2017-21
  )
  MAZDA_CX9_2021 = MazdaPlatformConfig(
    [MazdaCarDocs("Mazda CX-9 2021-23", video="https://youtu.be/dA3duO4a0O4")],
    MazdaCarSpecs(mass=4409 * CV.LB_TO_KG, wheelbase=2.93, steerRatio=17.6),
    # 2021-23 in North America; export markets kept the TC through 2025 (a JM7 TC S VIN attested)
    wmis={WMI.JAPAN_CROSSOVER, WMI.EXPORT_CROSSOVER}, chassis_codes={'TC'}, years={'M', 'N', 'P', 'S'},
  )
  MAZDA_CX5_2022 = MazdaPlatformConfig(
    [MazdaCarDocs("Mazda CX-5 2022-25")],
    MazdaCX5_2022CarSpecs(mass=3728 * CV.LB_TO_KG, wheelbase=2.698, steerRatio=18.1),  # learned; factory spec 15.5
    wmis={WMI.JAPAN_CROSSOVER, WMI.EXPORT_CROSSOVER}, chassis_codes={'KF'}, years={'N', 'P', 'R', 'S'},  # 2022-25
  )
  MAZDA_CX8_2023 = MazdaPlatformConfig(
    [MazdaCarDocs("Mazda CX-8 2023")],
    # Three-row CX-5 derivative on the CX-9 wheelbase (chassis KG): the CX-9 specs stand in. It has no
    # VIN year field to decode, so it fingerprints by firmware alone (docs/zoompilot/mazda-fingerprinting.md).
    MAZDA_CX9_2021.specs,
  )


class LKAS_LIMITS:
  STEER_THRESHOLD = 15
  DISABLE_SPEED = 45    # kph
  ENABLE_SPEED = 52     # kph


# (latAccelFactor, friction) on params.toml's scale for a platform whose params.toml entry is
# borrowed: the CX-5 2022's own learned values, not the CX-9 2021's (docs/zoompilot/lateral-tune.md).
TORQUE_TUNES = {
  CAR.MAZDA_CX5_2022: (1.222, 0.154),
}


# Keep steer-to-zero firmware synchronized with the STEER_TO_ZERO_PLATFORMS EPS entries in fingerprints.py.
STEER_TO_ZERO_EPS_FW = {
  b'K0A1-3210X-A-00\x00\x00\x00\x00\x00\x00\x00\x00\x00',  # CX-8 2023 (Japan)
  b'KBST-3210X-A-00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
  b'KSD5-3210X-C-00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
}

# Platforms that ship the steer-to-zero EPS from the factory: what an unread EPS falls back to.
STEER_TO_ZERO_PLATFORMS = frozenset({CAR.MAZDA_CX5_2022, CAR.MAZDA_CX8_2023})

# The 2016.5-era radar an EPS-swapped older body can keep: no tracks on bus 0, one static frame
# (docs/zoompilot/mazda-longitudinal.md). Stored unpadded and matched with nulls stripped.
G46L_RADAR_FW = {
  b'G46L-67XA1-C',
}


class Buttons:
  NONE = 0
  SET_PLUS = 1
  SET_MINUS = 2
  RESUME = 3
  CANCEL = 4


def platform_from_vin(vin: str) -> str | None:
  """The one platform the VIN's fields identify, or None when the VIN is unknown to
  every platform or ambiguous.

  Shared by the fuzzy firmware fallback and the selected-car/VIN mismatch warning.
  """
  if not is_valid_vin(vin):
    return None

  vin_obj = Vin(vin)
  chassis_code = vin_obj.vds[0:2]
  year = vin_obj.vis[0]

  candidates = {platform for platform in CAR
                if vin_obj.wmi in platform.config.wmis and chassis_code in platform.config.chassis_codes
                and year in platform.config.years}
  return str(next(iter(candidates))) if len(candidates) == 1 else None


def match_fw_to_car_fuzzy(live_fw_versions, vin, offline_fw_versions) -> set[str]:
  # After firmware matching fails, require VIN fields to identify one chassis platform.
  platform = platform_from_vin(vin)
  if platform is not None:
    carlog.error(f"Fingerprinted {platform} by VIN")
    return {platform}

  # Only export VINs without model-year data continue to the EPS-swap fallback.
  if not is_valid_vin(vin) or Vin(vin).wmi != WMI.OCEANIA_EXPORT:
    return set()

  # Export-car swaps require a recognized EPS and an engine that identifies one platform.
  eps_fw = live_fw_versions.get((0x730, None), set())
  if not eps_fw & STEER_TO_ZERO_EPS_FW:
    return set()

  engine_fw = live_fw_versions.get((0x7e0, None), set())
  candidates = {platform for platform, ecus in offline_fw_versions.items()
                if engine_fw & set(ecus.get((Ecu.engine, 0x7e0, None), []))}
  if len(candidates) != 1:
    return set()

  carlog.error(f"Fingerprinted {next(iter(candidates))} by engine firmware behind a steer-to-zero EPS swap")
  return {str(c) for c in candidates}

FW_QUERY_CONFIG = FwQueryConfig(
  fw_version_regex=br"[A-Z0-9-]{11,16}\x00{8,13}",
  requests=[
    # TODO: check data to ensure ABS does not skip ISO-TP frames on bus 0
    Request(
      [StdQueries.MANUFACTURER_SOFTWARE_VERSION_REQUEST],
      [StdQueries.MANUFACTURER_SOFTWARE_VERSION_RESPONSE],
      bus=0,
    ),
  ],
  match_fw_to_car_fuzzy=match_fw_to_car_fuzzy,
)

DBC = CAR.create_dbc_map()
