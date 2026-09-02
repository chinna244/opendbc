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
  STEER_DRIVER_ALLOWANCE = 15     # allowed driver torque before start limiting
  STEER_DRIVER_FACTOR = 1         # from dbc
  # 100 Hz. The EPS rate limit is per unit time (~1200 counts/s), so STEER_DELTA_UP/DOWN scale
  # with this: 12 counts * 100 Hz. Change one only with the other.
  STEER_STEP = 1

  ACCEL_MAX = 2.0   # m/s2
  ACCEL_MIN = -3.5  # m/s2

  # Longitudinal message rates, 100 Hz frames
  LONG_STEP = 2        # CRZ_INFO/CRZ_CTRL at 50 Hz, matching stock
  RADAR_STEP = 10      # radar static + track frames at 10 Hz
  RADAR_UDS_STEP = 50  # radar UDS traffic at 2 Hz: session control or tester present

  # Radar session timing, seconds. Silencing the radar before the camera's cold-boot
  # radar-presence check has passed latches an i-ACTIVSENSE fault.
  # See docs/zoompilot/mazda-longitudinal.md, "FSC settle gate".
  FSC_SETTLE_T = 10.0          # observed-settled time before the teardown may start
  # Stock CRZ_INFO runs at 50 Hz; this is the SILENCING -> SILENCED handover and the accFaulted
  # watch, not long enough to adopt a radar we never silenced (stock drops a few frames in a
  # row now and then). See docs/zoompilot/mazda-longitudinal.md, "Stock radar gap census".
  STOCK_RADAR_ALIVE_T = 0.05
  # Two-master guard: engagement (and MADS lateral) stays blocked until the stock radar has been
  # silent this long. The panda runs the same guard (mazda.h MAZDA_RADAR_SILENT_FRAMES) clocked
  # from our first synthetic CRZ_INFO, and ours must complete strictly after it or the panda
  # rejects the torque ramp and starves the EPS. Arming late is harmless.
  # See docs/zoompilot/mazda-longitudinal.md, "Guard ordering".
  PANDA_RADAR_SILENT_T = 1.0            # mazda.h MAZDA_RADAR_SILENT_FRAMES / 50 Hz PEDALS
  STOCK_RADAR_GUARD_MARGIN_T = 0.2
  STOCK_RADAR_GUARD_T = STOCK_RADAR_ALIVE_T + LONG_STEP * DT_CTRL + PANDA_RADAR_SILENT_T + STOCK_RADAR_GUARD_MARGIN_T  # 1.27 s
  RADAR_SESSION_LIMIT_T = 10.0  # per-episode UDS budget: a silent radar gives up here
  # CAM_LANEINFO is a ~2 Hz message; a freshness window shorter than one period reads every
  # gap as a dropout and the teardown gate never opens
  CAM_LANEINFO_PERIOD_T = 0.563
  CAM_LANEINFO_FRESH_T = 1.5

  # RESUME_UNLATCHING pulse length for a body-latched hold release: stock pulses 6-11 wire
  # frames, mode 9. A never-latched stop gets no pulse; there is nothing to unlatch.
  # See docs/zoompilot/mazda-longitudinal.md, "Release grammar".
  RESUME_UNLATCH_LATCHED_T = 0.18  # s, 9 wire frames, the latched-family mode
  # A latched release the body has not answered gets exactly one more pulse after this long,
  # then gives up; the body has answered every pulse so far within 51 ms.
  RESUME_REPULSE_T = 1.0  # s after a latched release, GEAR.BRAKE_HOLD still set

  CANCEL_CONTEXT_T = 0.5      # a wheel CANCEL keeps availability drops landing this long after release

  # The plan must ask to move this long before the hold releases, so a one-frame flap at a
  # standstill cannot fire a phantom pulse; stock's releases lag the lead by at least this
  RELEASE_DEBOUNCE_T = 0.2

  # A marginal vision lead flickers leadVisible faster than any radar track would, so the
  # advertised lead follows a state that has held steady (Hyundai debounces its lead bit too)
  LEAD_DEBOUNCE_T = 0.5

  # Stock relaxes its standstill command the instant the body ECU latches its own brake hold;
  # this is the relaxed value, the hold itself is the plan's own (CP.stopAccel)
  ACCEL_HOLD_LATCHED = -0.001  # m/s2

  # ACCEL_CMD ceiling while a latched release's unlatch pulse plays (stock peaks +0.25)
  ACCEL_RESUME_PULSE_MAX = 0.25  # m/s2, latched releases only

  # The release command follows stock's shape: a never-latched release relax-jumps into this
  # band in one frame, then ramps at this rate through the drive-off; a latched release ramps
  # at the same rate off the relaxed hold. See docs/zoompilot/mazda-longitudinal.md,
  # "Release command shape".
  ACCEL_RELEASE_BAND = -0.26  # m/s2, the one-frame relax target at a never-latched release
  ACCEL_RELEASE_RAMP = 1.25   # m/s3, stock's release ramp (+25 raw per 50 Hz frame)

  # The release ramp keeps climbing past the plan while the car is still stopped, because the
  # plan's creep value is not always enough to break away and Mazda's long control has no
  # integrator (ki = 0). Bounded by stock's own worst breakaway, by time, and relative to the
  # plan. See docs/zoompilot/mazda-longitudinal.md, "Breakaway".
  ACCEL_BREAKAWAY_MAX = 1.45  # m/s2, ceiling for the still-stopped release ramp
  ACCEL_BREAKAWAY_T = 3.0  # s
  ACCEL_BREAKAWAY_OVERSHOOT = 0.75  # m/s2 above the plan the still-stopped ramp may climb

  # Command slew limits, m/s3, on the plan-following command only. Asymmetric: the windup limit
  # keeps the command from dumping the brake in one frame, a tight winddown would only delay
  # real braking. 4.0 clears the p99 of the plan's own up-slew; Toyota uses 4.0 both ways.
  ACCEL_WINDUP_LIMIT = 4.0 * DT_CTRL     # m/s2 per frame
  ACCEL_WINDDOWN_LIMIT = -10.0 * DT_CTRL  # m/s2 per frame, clips only the p99.9+ steps

  def __init__(self, CP):
    # The 2022 CX-5 EPS block, keyed on the EPS (interface.py sets the flag from the EPS
    # firmware) so an EPS swapped into another Mazda gets the same tune
    if CP.flags & MazdaFlags.STEER_TO_ZERO_EPS:
      # STEER_MAX is the scale from normalized torque to counts, not just a ceiling, and the
      # latAccelFactor seeds depend on it; the EPS's real ceiling is EPS_CEILING_LOOKUP below
      self.STEER_MAX = 1200        # theoretical max_steer 2047
      # 1200 below 32 mph for full low-speed authority and feedforward overshoot.
      # 800 above for smoother highway steering.
      self.STEER_MAX_LOOKUP = ([0., 14.2, 14.5], [1200, 1200, 800])
      # EPS hardware slew: 12 counts per 100 Hz frame in both directions, whoever commands it.
      # A faster winddown only lets the command run ahead of the wheel. Must equal the panda's
      # max_rate_up/down for this EPS (mazda.h MAZDA_STEER_TO_ZERO_EPS_STEERING_LIMITS): a
      # looser panda rate-down rejects every frame of a driver-override winddown.
      # See docs/zoompilot/mazda-lateral.md, "The torque envelope".
      self.STEER_DELTA_UP = 12
      self.STEER_DELTA_DOWN = 12
      self.STEER_DRIVER_MULTIPLIER = 15   # weight driver torque (tuned for the CX-5 EPS; upstream stock is 1)
      # Torque the EPS will actually apply, by speed, from its own LKAS_EFFECTIVE report over
      # 11.4M frames. Commanding above it delivers nothing extra and hides the saturation from
      # controlsd's steer_limited_by_safety, so the integrator winds up.
      # See docs/zoompilot/mazda-lateral.md, "EPS ceiling clamp".
      self.EPS_CEILING_LOOKUP = ([8.0, 8.5, 9.4, 10.3, 11.2, 12.1, 13.0, 13.9, 14.5],
                                 [1148, 1132, 1092, 1048, 1012,  920,  808,  676,  620])

      # Non-delivery latch: once the EPS has applied exactly nothing to a real request for this
      # long, stop commanding, so the camera does not latch ERR_BIT_1 (steerFaultPermanent).
      # Read from LKAS_EFFECTIVE, not LKAS_BLOCK: a block above ~4 m/s still delivers a third to
      # a half of the request. Defense in depth; the faults captured so far were 0x243
      # starvation. See docs/zoompilot/mazda-lateral.md, "LKAS_BLOCK and the non-delivery latch".
      self.STEER_UNDELIVERED_MIN = 200      # counts; below this the EPS rounds to zero anyway
      self.STEER_UNDELIVERED_FRAMES = 20    # 200 ms at 100 Hz

      # Telling the driver is the slower half: hold the latch a further 0.8 s and only report
      # above manoeuvring speed, where an EPS applying nothing is abnormal (like Honda's
      # LOW_SPEED_LOCKOUT suppression), and never for a block the EPS flags as its own
      # low-speed standby (LKAS_TRACK_STATE): that one releases on time since 3 m/s, not on
      # speed, so a brisk launch carries it past any gate.
      self.STEER_UNDELIVERED_ALERT_FRAMES = 80    # 0.8 s at 100 Hz, on top of the latch's 0.2
      self.STEER_UNDELIVERED_ALERT_MIN_SPEED = 12. * CV.MPH_TO_MS

      # Driver-torque headroom. The panda enforces the same envelope from the min/max of its
      # own last 6 samples, while the controller sees one sample a control cycle old; at a
      # multiplier of 15 a few counts of staleness put every frame over the panda's line. So
      # bound the ceiling with the most adverse sample in a window spanning the panda's, plus a
      # margin. See docs/zoompilot/mazda-lateral.md, "Driver-torque headroom".
      self.STEER_DRIVER_SAMPLES = 10
      self.STEER_DRIVER_MARGIN = 2
    else:
      # upstream stock, equal to the panda's MAZDA_STEERING_LIMITS (no safety param bit)
      self.STEER_MAX = 800         # theoretical max_steer 2047
      self.STEER_DELTA_UP = 10
      self.STEER_DELTA_DOWN = 25
      self.STEER_DRIVER_MULTIPLIER = 1    # upstream stock


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
  # Static flags
  # Gen 1 hardware: same CAN messages and same camera
  GEN1 = 1

  # Dynamic flags
  # 2022 CX-5 EPS present, by EPS firmware (interface.py). Keys the whole 2022 EPS block:
  # CarControllerParams, carstate's fault handling, alpha-long availability, and the panda's
  # MazdaSafetyFlags.STEER_TO_ZERO_EPS
  STEER_TO_ZERO_EPS = 2


class MazdaSafetyFlags(IntFlag):
  LONG = 1
  # selects MAZDA_STEER_TO_ZERO_EPS_STEERING_LIMITS in safety/modes/mazda.h
  STEER_TO_ZERO_EPS = 2


class WMI(StrEnum):
  JAPAN_PASSENGER = "JM1"   # Japan-built passenger cars
  JAPAN_CROSSOVER = "JM3"   # Japan-built crossovers
  MEXICO_PASSENGER = "3MZ"  # Mazda de Mexico (Mazda 3)
  # Export VINs (Australia, New Zealand) carry no model year field and never decode
  # through the platform table; only the EPS-swap fallback below accepts them
  OCEANIA_EXPORT = "JM0"


@dataclass
class MazdaPlatformConfig(PlatformConfig):
  dbc_dict: DbcDict = field(default_factory=lambda: {Bus.pt: 'mazda_2017', Bus.radar: 'mazda_2017'})
  flags: int = MazdaFlags.GEN1
  wmis: set[WMI] = field(default_factory=set)
  chassis_codes: set[str] = field(default_factory=set)
  years: set[str] = field(default_factory=set)


class CAR(Platforms):
  MAZDA_CX5 = MazdaPlatformConfig(
    [MazdaCarDocs("Mazda CX-5 2017-21")],
    MazdaCarSpecs(mass=3655 * CV.LB_TO_KG, wheelbase=2.7, steerRatio=15.5),
    wmis={WMI.JAPAN_CROSSOVER}, chassis_codes={'KF'}, years={'H', 'J', 'K', 'L', 'M'},  # 2017-21
  )
  MAZDA_CX9 = MazdaPlatformConfig(
    [MazdaCarDocs("Mazda CX-9 2016-20")],
    MazdaCarSpecs(mass=4217 * CV.LB_TO_KG, wheelbase=2.93, steerRatio=17.6),
    # no radar bus: this radar does not put the 0x361-0x366 tracks on bus 0
    dbc_dict={Bus.pt: 'mazda_2017'},
    wmis={WMI.JAPAN_CROSSOVER}, chassis_codes={'TC'}, years={'G', 'H', 'J', 'K', 'L'},  # 2016-20
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
    wmis={WMI.JAPAN_CROSSOVER}, chassis_codes={'TC'}, years={'M', 'N', 'P'},  # 2021-23
  )
  MAZDA_CX5_2022 = MazdaPlatformConfig(
    [MazdaCarDocs("Mazda CX-5 2022-25")],
    MazdaCX5_2022CarSpecs(mass=3728 * CV.LB_TO_KG, wheelbase=2.698, steerRatio=18.1),  # 15.5 is factory spec; 18.1 from paramsd learner (2.9M samples)
    wmis={WMI.JAPAN_CROSSOVER}, chassis_codes={'KF'}, years={'N', 'P', 'R', 'S'},  # 2022-25
  )


class LKAS_LIMITS:
  STEER_THRESHOLD = 15
  DISABLE_SPEED = 45    # kph
  ENABLE_SPEED = 52     # kph


# EPS firmware versions with steer-to-zero capability (2022+ CX-5 EPS). Matched against
# car_fw rather than the fingerprinted platform so the same EPS swapped into another Mazda
# keeps full-speed steering. Keep in sync with the CAR.MAZDA_CX5_2022 (Ecu.eps, 0x730) block
# in fingerprints.py.
STEER_TO_ZERO_EPS_FW = {
  b'KBST-3210X-A-00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
  b'KSD5-3210X-C-00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
}


class Buttons:
  NONE = 0
  SET_PLUS = 1
  SET_MINUS = 2
  RESUME = 3
  CANCEL = 4


def match_fw_to_car_fuzzy(live_fw_versions, vin, offline_fw_versions) -> set[str]:
  # Runs after exact and generic fuzzy FW matching fail, which a donor EPS (steer-to-zero swap)
  # guarantees. The VIN names the chassis through any ECU swap: WMI, model line (positions 4-5)
  # and model year (position 10) must all name the same single platform.
  # See docs/zoompilot/mazda-fingerprinting.md.
  if not is_valid_vin(vin):
    return set()

  vin_obj = Vin(vin)
  chassis_code = vin_obj.vds[0:2]
  year = vin_obj.vis[0]

  candidates = set()
  for platform in CAR:
    platform_config = platform.config
    if vin_obj.wmi in platform_config.wmis and chassis_code in platform_config.chassis_codes and year in platform_config.years:
      candidates.add(platform)

  if len(candidates) == 1:
    carlog.error(f"Fingerprinted {next(iter(candidates))} by VIN")
    return {str(c) for c in candidates}

  # A decodable WMI that names no platform is an unsupported model; only export VINs, which
  # carry no model year, go on to the swap fallback
  if vin_obj.wmi != WMI.OCEANIA_EXPORT:
    return set()

  # EPS-swap fallback for export cars. Like upstream's generic fuzzy match it needs two
  # recognised ECUs: an EPS in STEER_TO_ZERO_EPS_FW (the only EPS this port grants lateral
  # through) and an engine whose firmware names exactly one platform.
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
