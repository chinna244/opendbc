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
  # 100 Hz. The stock camera commands CAM_LKAS at 16.6 Hz (60 ms); we send 6x that. The EPS
  # rate limit is per unit TIME (~1200 units/s), not per received frame -- measured on a stock
  # drive where the camera commands at 16.6 Hz and the EPS still steps 12 units per 10 ms
  # (docs/mazda-lkas-camera-tx-census.md). So the cadence buys no extra authority, but
  # STEER_DELTA_UP/DOWN are per-frame, which makes this constant load-bearing:
  # STEER_DELTA_UP * (1/DT_CTRL/STEER_STEP) = 12 * 100 = 1200 units/s matches the EPS exactly.
  # Changing STEER_STEP without rescaling STEER_DELTA_UP by the same factor silently cuts the
  # commanded slew rate (STEER_STEP=6 would give 200 units/s, 6x slower than the hardware).
  STEER_STEP = 1

  ACCEL_MAX = 2.0   # m/s2
  ACCEL_MIN = -3.5  # m/s2

  # Longitudinal message rates, 100 Hz frames
  LONG_STEP = 2        # CRZ_INFO/CRZ_CTRL at 50 Hz, matching stock
  RADAR_STEP = 10      # radar static + track frames at 10 Hz
  RADAR_UDS_STEP = 50  # radar UDS traffic at 2 Hz: session control or tester present

  # Radar session timing, seconds. The FSC's radar-presence check faulted when the radar
  # went quiet 1.9 s after the camera's boot settle and passed from 5.8 s
  # (docs/mazda-alpha-long-setup-teardown.md), hence the 10 s settle requirement.
  FSC_SETTLE_T = 10.0          # observed-settled time before the teardown may start
  STOCK_RADAR_ALIVE_T = 0.05   # stock CRZ_INFO runs at 50 Hz; silent this long = torn down
  STOCK_RADAR_GUARD_T = 1.0    # two-master guard: block engagement until silent this long
  RADAR_SESSION_LIMIT_T = 10.0  # per-episode UDS budget: a silent radar gives up here
  # CAM_LANEINFO is a ~2 Hz message (longest period measured 0.563 s across 26+ segments on
  # two cars), so freshness has to be judged against that cadence: a window shorter than one
  # period reads every inter-frame gap as a dropout, zeroes the settle timer each time, and
  # the teardown gate never opens. The window keeps 2.7x margin over the longest observed
  # period and still catches a genuine camera dropout.
  CAM_LANEINFO_PERIOD_T = 0.563
  CAM_LANEINFO_FRESH_T = 1.5

  # RESUME_UNLATCHING at the release comes in two families (33-pulse census over the stock
  # corpus, tools/mazda_long release grammar scan, 2026-08-27):
  #   - a body-latched hold pulses 6-11 wire frames (0.12-0.20 s, mode 9) while the command
  #     ramps off the relaxed -0.001; nothing stock ever pulsed longer than 0.20 s
  #   - a never-latched stop only blips 1-6 wire frames (mostly 2-3), starting ~3 wire frames
  #     AFTER the stop bits drop, once the command has relax-jumped into its release band
  # Treating every release as a long latched-style pulse held the unlatch bit over hold-grade
  # braking, a tuple stock never emits, and the camera latched SCBS 90 ms in
  # (route 00000053 t+714.8, second CX-5, with a real departing lead advertised)
  RESUME_UNLATCH_LATCHED_T = 0.18  # s, 9 wire frames, the latched-family mode
  # The unlatch pulse is a last resort, not the release protocol. Of the 11 RESUME_UNLATCHING
  # pulses this port has ever put on the wire, 8 went out to a healthy camera and all 8 latched
  # the SCBS fault 80-90 ms in (routes 0000007e, 0000007f, 000000fe, 00000100, 00000103,
  # 00000115, 00000118, 00000053); the other 3 hit a camera already faulted. There is no clean
  # pulse anywhere in our history, across builds whose shape converged to a byte-level twin of
  # a stock latched release.
  #
  # Across those 8 every candidate signal takes both values and still latches -- phase 0/2/3,
  # RADAR_HAS_LEAD 0/1, ACC_ACTIVE_2 0/1, GEAR.BRAKE_HOLD 0/1 (4 latched-family, 4 not),
  # longActive 0/1, lead 0.0-7.5 m. Seven rounds of matching stock's pulse grammar each found a
  # deviation to fix and each was falsified by the next drive; the only invariant is the pulse.
  # So the command relaxes first and the body is given this long to drop GEAR.BRAKE_HOLD on
  # its own; the pulse only fires if it does not. Stock's never-latched blip is dropped
  # outright: nothing is latched there, so it unlatches nothing.
  # Route 0000011d proved the body will not let go on its own: the command is already at
  # ACCEL_HOLD_LATCHED and the stop bits are already dropped during a body-latched hold, so a
  # deferral that only withholds the pulse puts nothing new on the wire at all. The body sat
  # through 300 ms of that silence and released 40 ms after the pulse finally fired. So the
  # deferral now makes an actual request -- a small positive command, the one release signal
  # we have never tried -- and only falls back to the pulse if the body ignores it too.
  # The nudge gets a long window on purpose. Every second spent here is a second the camera is
  # not being shown the one signal that has latched it 8 times out of 8, and the cost of
  # waiting is only a slower drive-off -- the driver's pedal outranks the hold throughout. The
  # 0.5 s this started at was sized against the *silent* deferral it replaced (route 0000011d,
  # where nothing new went on the wire at all), not against a deferral that actually asks.
  RESUME_PULSE_DEFER_T = 2.0       # s the nudge gets before we resort to the pulse
  ACCEL_DEFER_NUDGE = 0.15         # m/s2, the release request made while the body still holds

  CANCEL_CONTEXT_T = 0.5       # a wheel CANCEL keeps availability drops landing this long after release

  # The plan flapping across zero at a held standstill (a lead inches forward and stops) used
  # to fire a fresh RESUME_UNLATCHING pulse per flap and re-assert the stop bits mid-pulse, a
  # combination stock never emits (stock pulses exactly once per release, stop bits already
  # dropped). The plan must ask to move this long before the hold releases; stock's releases
  # lag the lead's departure by at least this much (all 23 latched releases show the lead
  # already opening at >= +0.31 m/s at the pulse, ~0.2 s into a typical drive-off).
  RELEASE_DEBOUNCE_T = 0.2

  # A marginal vision lead flickers leadVisible faster than the camera can be shown a track
  # appearing and vanishing (route 6bb2dc61c4 t+400: 6 toggles in 1.4 s on a 120 m lead), so the
  # advertised lead only follows a state that has held steady, the way Hyundai debounces its
  # lead bit for 50 frames
  LEAD_DEBOUNCE_T = 0.5

  # Stock relaxes its standstill command the instant the body ECU takes the hold over, not on any
  # schedule: across 13 stock holds >= 4.5 s the relax and GEAR.BRAKE_HOLD agreed to within
  # +-0.02 s in all 9 where both were visible, and the latch itself landed anywhere from 0.01 s
  # to 7.6 s after standstill. The command through the hold is the plan's own, which parks at
  # CP.stopAccel; this is only the relaxed value we send once the car has the brakes.
  ACCEL_HOLD_LATCHED = -0.001  # m/s2

  # ACCEL_CMD ceiling while a body-latched release's RESUME_UNLATCHING pulse plays: stock's
  # latched releases peak at +0.24-0.25 m/s2 (raw +182/+195) in the pulse tail, +0.34 worst
  # case. Never-latched blips are capped at zero -- stock's command is still negative in every
  # never-latched pulse frame of the corpus.
  ACCEL_RESUME_PULSE_MAX = 0.25  # m/s2, latched releases only

  # The release command itself follows stock's shape, not a slew off the hold value. At a
  # never-latched release stock relax-jumps the command in ONE frame from the hold value into
  # a -0.27..-0.18 start (pulse-frame commands span -0.269..-0.111 across the whole corpus)
  # and then ramps ~+25 raw per 50 Hz frame straight through the drive-off; a latched release
  # ramps at the same rate off the relaxed -0.001. Slewing up from -1.024 instead kept
  # hold-grade braking on the wire beneath the release pulse (route 00000053 t+714.8), and
  # pre-ramping toward the plan crossed zero inside it (route 00000100 t+353) -- the band
  # between those two edges is what the camera accepts.
  ACCEL_RELEASE_BAND = -0.26  # m/s2, the one-frame relax target at a never-latched release
  ACCEL_RELEASE_RAMP = 1.25   # m/s3, stock's release ramp (+25 raw per 50 Hz frame)

  # The ramp above hands the command back the moment it catches the plan, which assumes the
  # plan's own value is enough to get the car rolling. It is not always. On the EPS-swapped
  # CX-9 the plan parked at +0.42..+0.47 behind a lead 2.5 m ahead and the car sat dead still
  # for the whole 1.5 s the command was held there, until the driver used the pedal (route
  # 00000009--ad9e22f986 t+452.9); the CX-5 releases in the corpus only ever broke away once
  # the command had climbed past +1.0 (route_118 t+581.6: the only clean corpus breakaway,
  # last still frame at +1.17, first rolling frame at +1.07). It was not the grade: the
  # accelerometer puts that stop at +0.41 deg nose-up, 0.07 m/s2 of gravity, effectively flat
  # (the three stops the driver gassed out of were on up to +2.4 deg). So the ramp keeps
  # climbing past the plan for as long as the car is still stopped. LongControl cannot do this
  # itself: Mazda runs the default ki of 0, so its pid state emits the plan's a_target verbatim
  # with no integrator to wind up against a car that is not moving.
  #
  # The ceiling is deliberately modest. This overrides the plan, so every 0.1 above what the
  # car needed is comfort the model did not ask for, spent creeping toward a lead that can be
  # 2.5 m away -- and the ~0.3 s actuator dead time already carries the command ~0.38 past the
  # value that actually broke the car free before standstill clears. +0.8 is a gentle pull-away
  # that still clears the +0.47 the CX-9 sat through. It is NOT yet known whether that car
  # needs more: the qlog cannot show GEAR.BRAKE_HOLD or the stop bits, so a body brake latch
  # invisible to us is still on the table and would not be cured by asking harder anyway.
  # Raise this only against an rlog that rules that out.
  ACCEL_BREAKAWAY_MAX = 0.8  # m/s2, ceiling for the still-stopped release ramp
  # ...and it gives up after this long, so a car held by something we cannot see -- a kerb, a
  # steep grade, a foot on the brake -- settles back onto the plan instead of being leaned on
  # indefinitely.
  ACCEL_BREAKAWAY_T = 3.0  # s

  # Command slew limits, m/s3, on the plan-following command only. Asymmetric on purpose: the
  # windup limit is what keeps the command from dumping the brake in one frame (the driver-felt
  # problem), while a tight winddown limit would delay real braking for no measured benefit.
  # 4.0 sits above the p99 of the plan's own up-slew on the reporter's route (p99 +3.2, p99.9
  # +6.3, max +34 m/s3), so it only clips state-transition steps. Toyota uses 4.0 both ways.
  ACCEL_WINDUP_LIMIT = 4.0 * DT_CTRL     # m/s2 per frame
  ACCEL_WINDDOWN_LIMIT = -10.0 * DT_CTRL  # m/s2 per frame, clips only the p99.9+ steps

  def __init__(self, CP):
    # Gate the higher-authority steering tune on the CX-5 2022+ EPS, not the car model, so the
    # CX-9 that shares this EPS and CX-5-EPS swaps keep it. steer_to_zero sets minSteerSpeed == 0
    # (interface.py / STEER_TO_ZERO_EPS_FW) — the same EPS-present proxy carstate.py gates on.
    if CP.minSteerSpeed == 0:
      # STEER_MAX is the SCALE from the controller's normalized output to CAN counts
      # (carcontroller: new_torque = actuators.torque * steer_max), not just a ceiling, and
      # latAccelFactor is proportional to it. Changing it rescales every sub-saturation
      # command and invalidates every speed_dependent.toml LAF seed at once, so it is left
      # alone; the EPS's real ceiling is enforced separately by EPS_CEILING_LOOKUP below.
      self.STEER_MAX = 1200        # theoretical max_steer 2047
      # 1200 below 32 mph for full low-speed authority and feedforward overshoot.
      # 800 above for smoother highway steering.
      self.STEER_MAX_LOOKUP = ([0., 14.2, 14.5], [1200, 1200, 800])
      # EPS hardware rate limit: 12 units/frame at 100 Hz (4-unit quantization, max 3 steps).
      # Per unit time, not per frame -- see the STEER_STEP note above before changing either.
      self.STEER_DELTA_UP = 12
      self.STEER_DELTA_DOWN = 25
      self.STEER_DRIVER_MULTIPLIER = 15   # weight driver torque (tuned for the CX-5 EPS; upstream stock is 1)
      # Torque the EPS will actually apply, by speed. Measured over 11,408,748 clean frames
      # (4798 segments, not LKAS_BLOCK / not steeringPressed / vEgo > 2) from 0x241
      # STEER_RATE, which the EPS itself transmits: LKAS_EFFECTIVE is what it applied.
      # Above 32.5 mph ZERO of 7,490,617 frames exceeded 620; below 18 mph none exceeded
      # 1148. The rail is a function of instantaneous speed with no memory -- decel, steady
      # and accel rails are identical (spread 0) from 32-60 mph -- and is left/right
      # symmetric. Derivation: tools/mazda_long/eps_ceiling_curve.py, and
      # docs/mazda-lkas-camera-tx-census.md.
      #
      # Commanding above this delivers no extra torque at the wheel, it only hides actuator
      # saturation from the controller: controlsd derives steer_limited_by_safety from
      # actuators.torque vs actuatorsOutput.torque, so without the clamp the request and the
      # report agree at 1.0 while the EPS sits railed, the integrator never freezes, and it
      # winds up to be paid back as overshoot on release.
      self.EPS_CEILING_LOOKUP = ([8.0, 8.5, 9.4, 10.3, 11.2, 12.1, 13.0, 13.9, 14.5],
                                 [1148, 1132, 1092, 1048, 1012,  920,  808,  676,  620])
    else:
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


class MazdaSafetyFlags(IntFlag):
  LONG = 1


class WMI(StrEnum):
  JAPAN_PASSENGER = "JM1"   # Japan-built passenger cars
  JAPAN_CROSSOVER = "JM3"   # Japan-built crossovers
  MEXICO_PASSENGER = "3MZ"  # Mazda de Mexico (Mazda 3)


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
  # A donor EPS (steer-to-zero swaps) breaks every exact FW match; the VIN names
  # the chassis through any ECU swap. Runs only after exact and fuzzy FW fail.
  # Model line is VIN positions 4-5, model year code is position 10.
  if is_valid_vin(vin):
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

    # a known Mazda WMI that names no platform identified an unsupported model
    # (BP, DM, KE, out-of-range years): never second-guess it with the engine.
    # WMIs outside the table (e.g. 7MM, CX-50) keep the fallback and its
    # collision risk; pinned by test.
    if vin_obj.wmi in {wmi for platform in CAR for wmi in platform.config.wmis}:
      return set()

  # Oceania VINs encode no model year and never decode; engine firmware is
  # unique per platform (asserted by test), so it names the chassis instead.
  # A lone responding address is not a car to name.
  if len(live_fw_versions) < 2:
    return set()

  engine_fw = live_fw_versions.get((0x7e0, None), set())
  candidates = set()
  for platform, ecus in offline_fw_versions.items():
    if engine_fw & set(ecus.get((Ecu.engine, 0x7e0, None), [])):
      candidates.add(platform)

  if len(candidates) == 1:
    carlog.error(f"Fingerprinted {next(iter(candidates))} by engine firmware")
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
