from opendbc.can import CANDefine, CANParser
from opendbc.car import Bus, DT_CTRL, create_button_events, structs, uds
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.interfaces import CarStateBase
from opendbc.car.mazda.values import DBC, LKAS_LIMITS, CarControllerParams, MazdaFlags
from opendbc.sunnypilot.car.mazda.carstate_ext import CarStateExt

ButtonType = structs.CarState.ButtonEvent.Type

FSC_SETTLE_FRAMES = int(CarControllerParams.FSC_SETTLE_T / DT_CTRL)
STOCK_RADAR_ALIVE_FRAMES = int(CarControllerParams.STOCK_RADAR_ALIVE_T / DT_CTRL)
STOCK_RADAR_GUARD_FRAMES = round(CarControllerParams.STOCK_RADAR_GUARD_T / DT_CTRL)
CANCEL_CONTEXT_FRAMES = int(CarControllerParams.CANCEL_CONTEXT_T / DT_CTRL)
CAM_LANEINFO_FRESH_FRAMES = int(CarControllerParams.CAM_LANEINFO_FRESH_T / DT_CTRL)


class CarState(CarStateBase, CarStateExt):
  def __init__(self, CP, CP_SP):
    CarStateBase.__init__(self, CP, CP_SP)
    CarStateExt.__init__(self, CP, CP_SP)

    can_define = CANDefine(DBC[CP.carFingerprint][Bus.pt])
    self.shifter_values = can_define.dv["GEAR"]["GEAR"]

    self.crz_btns_counter = 0
    self.acc_active_last = False
    self.lkas_allowed_speed = False
    self.lkas_blocked = False
    self.lkas_effective = 0
    # LKAS non-delivery latch (update_steer_undelivered), 2022+ EPS only
    self.params = CarControllerParams(CP)
    self.steer_undelivered_frames = 0
    self.steer_undelivered = False
    self.steer_undelivered_alert = False

    self.distance_button = 0
    self.accel_button = 0
    self.decel_button = 0
    self.cancel_button = 0
    self.resume_button = 0
    self.main_button = 0

    self.cruise_available = False
    self.cruise_enabled = False
    self.cruise_enabled_blocked = True
    self.brake_pressed_prev = False
    self.stock_radar_silent_frames = 0
    self.radar_was_silenced = False
    self.cancel_context_frames = 0
    self.cam_laneinfo_seen = False
    self.cam_laneinfo_silent_frames = 0
    self.cam_empty_seen = False
    self.radar_session_refused = False
    self.fsc_settled_frames = 0
    # the body ECU has taken the standstill hold over and is holding the brakes itself
    self.brake_hold = False

  @property
  def fsc_settled(self) -> bool:
    return self.fsc_settled_frames >= FSC_SETTLE_FRAMES

  @property
  def stock_radar_alive(self) -> bool:
    return self.stock_radar_silent_frames < STOCK_RADAR_ALIVE_FRAMES

  @property
  def stock_radar_gone(self) -> bool:
    # the two-master guard's window: silent this long is a torn-down radar, not a dropped
    # frame, so the session manager may adopt a radar it did not silence itself
    return self.stock_radar_silent_frames >= STOCK_RADAR_GUARD_FRAMES

  def update_steer_undelivered(self, v_ego_raw: float, lkas_request: float, lkas_blocked: bool, lkas_track_state: bool) -> None:
    # Non-delivery latch, from the EPS's own STEER_RATE report (LKAS_REQUEST received,
    # LKAS_EFFECTIVE applied): once it has applied nothing to a real request for long enough
    # the carcontroller stops sending one, before the camera latches ERR_BIT_1. Releases on
    # LKAS_BLOCK clearing, since a zeroed command has no delivery to observe. steeringPressed
    # does not gate entry: a driver steering with the request never unwinds it.
    # See docs/zoompilot/mazda-lateral.md, "LKAS_BLOCK and the non-delivery latch".
    if not lkas_blocked:
      self.steer_undelivered_frames = 0
      self.steer_undelivered = False
      self.steer_undelivered_alert = False
    elif not self.steer_undelivered:
      if self.lkas_effective == 0 and abs(lkas_request) > self.params.STEER_UNDELIVERED_MIN:
        self.steer_undelivered_frames += 1
        self.steer_undelivered = self.steer_undelivered_frames >= self.params.STEER_UNDELIVERED_FRAMES
      else:
        self.steer_undelivered_frames = 0

    if self.steer_undelivered:
      # alerting is the slower half: hold the latch a full second and only speak above
      # manoeuvring speed, where an EPS applying nothing is abnormal. The speed is read once,
      # when the alert arms, so a car hovering at the threshold does not flicker.
      # LKAS_TRACK_STATE is the EPS's own low-speed standby: set for the whole of a block that
      # began at standstill, which releases about a second after 3 m/s and so outruns any speed
      # gate on a brisk launch; clear for a block that began at speed (an EPS that dropped
      # LKAS mid-delivery, the shape of every real fault). Only the latter is worth a word.
      self.steer_undelivered_frames += 1
      if (not self.steer_undelivered_alert and not lkas_track_state and
          self.steer_undelivered_frames >= self.params.STEER_UNDELIVERED_FRAMES + self.params.STEER_UNDELIVERED_ALERT_FRAMES and
          v_ego_raw >= self.params.STEER_UNDELIVERED_ALERT_MIN_SPEED):
        self.steer_undelivered_alert = True

  def update(self, can_parsers) -> tuple[structs.CarState, structs.CarStateSP]:
    cp = can_parsers[Bus.pt]
    cp_cam = can_parsers[Bus.cam]

    ret = structs.CarState()
    ret_sp = structs.CarStateSP()

    self.parse_wheel_speeds(ret,
      cp.vl["WHEEL_SPEEDS"]["FL"],
      cp.vl["WHEEL_SPEEDS"]["FR"],
      cp.vl["WHEEL_SPEEDS"]["RL"],
      cp.vl["WHEEL_SPEEDS"]["RR"],
    )

    # Match panda speed reading: standstill comes off ENGINE_DATA, the field the panda's
    # vehicle_moving reads, so the two agree on when the car counts as stopped
    speed_kph = cp.vl["ENGINE_DATA"]["SPEED"]
    ret.standstill = speed_kph <= .1

    can_gear = int(cp.vl["GEAR"]["GEAR"])
    ret.gearShifter = self.parse_gear_shifter(self.shifter_values.get(can_gear, None))
    self.brake_hold = cp.vl["GEAR"]["BRAKE_HOLD"] == 1

    ret.genericToggle = bool(cp.vl["BLINK_INFO"]["HIGH_BEAMS"])
    ret.leftBlindspot = cp.vl["BSM"]["LEFT_BS_STATUS"] != 0
    ret.rightBlindspot = cp.vl["BSM"]["RIGHT_BS_STATUS"] != 0
    ret.leftBlinker, ret.rightBlinker = self.update_blinker_from_lamp(40, cp.vl["BLINK_INFO"]["LEFT_BLINK"] == 1,
                                                                      cp.vl["BLINK_INFO"]["RIGHT_BLINK"] == 1)

    ret.steeringAngleDeg = cp.vl["STEER"]["STEER_ANGLE"]
    ret.steeringTorque = cp.vl["STEER_TORQUE"]["STEER_TORQUE_SENSOR"]
    ret.steeringPressed = self.update_steering_pressed(abs(ret.steeringTorque) > LKAS_LIMITS.STEER_THRESHOLD, 5)

    ret.steeringTorqueEps = cp.vl["STEER_TORQUE"]["STEER_TORQUE_MOTOR"]
    ret.steeringRateDeg = cp.vl["STEER_RATE"]["STEER_ANGLE_RATE"]

    ret.brakePressed = cp.vl["PEDALS"]["BRAKE_ON"] == 1

    ret.seatbeltUnlatched = cp.vl["SEATBELT"]["DRIVER_SEATBELT"] == 0
    ret.doorOpen = any([cp.vl["DOORS"]["FL"], cp.vl["DOORS"]["FR"],
                        cp.vl["DOORS"]["BL"], cp.vl["DOORS"]["BR"]])

    # TODO: this should be from 0 - 1.
    ret.gasPressed = cp.vl["ENGINE_DATA"]["PEDAL_GAS"] > 0

    # Either due to low speed or hands off
    lkas_blocked = cp.vl["STEER_RATE"]["LKAS_BLOCK"] == 1

    # what the EPS actually applied: a block above ~4 m/s usually still delivers part of the
    # request, and LKAS_BLOCK alone cannot tell that from a total block
    self.lkas_blocked = lkas_blocked
    self.lkas_effective = cp.vl["STEER_RATE"]["LKAS_EFFECTIVE"]
    if self.CP.flags & MazdaFlags.STEER_TO_ZERO_EPS:
      self.update_steer_undelivered(ret.vEgoRaw, cp.vl["STEER_RATE"]["LKAS_REQUEST"], lkas_blocked,
                                    cp.vl["STEER_RATE"]["LKAS_TRACK_STATE"] == 1)

    if not self.CP.flags & MazdaFlags.STEER_TO_ZERO_EPS:
      # LKAS is enabled at 52kph going up and disabled at 45kph going down
      # wait for LKAS_BLOCK signal to clear when going up since it lags behind the speed sometimes
      if speed_kph > LKAS_LIMITS.ENABLE_SPEED and not lkas_blocked:
        self.lkas_allowed_speed = True
      elif speed_kph < LKAS_LIMITS.DISABLE_SPEED:
        self.lkas_allowed_speed = False
    else:
      self.lkas_allowed_speed = True

    # CAM_LANEINFO freshness: before the first frame the parser reads all-zero, and through a
    # camera dropout it repeats stale values; neither may drive the FSC settle gate
    if len(cp_cam.vl_all["CAM_LANEINFO"]["LANE_LINES"]) > 0:
      self.cam_laneinfo_seen = True
      self.cam_laneinfo_silent_frames = 0
    else:
      self.cam_laneinfo_silent_frames += 1
    cam_laneinfo_fresh = self.cam_laneinfo_seen and self.cam_laneinfo_silent_frames < CAM_LANEINFO_FRESH_FRAMES

    # camera collision display -> stockFcw: 0x21d leaves its idle 0x7f status only while
    # showing it. A seen latch is guard enough, a stale value only strands a log flag.
    if not self.cam_empty_seen:
      self.cam_empty_seen = len(cp_cam.vl_all["CAM_EMPTY"]["STATUS"]) > 0
    cam_empty = cp_cam.vl["CAM_EMPTY"]
    ped = cp_cam.vl["CAM_PEDESTRIAN"]
    ret.stockFcw = (self.cam_empty_seen and cam_empty["STATUS"] != 0x7F) or \
                   ped["PED_WARNING"] == 1 or ped["BRAKE_WARNING"] == 1

    if self.CP.openpilotLongitudinalControl:
      # The radar teardown silences the radar-owned CRZ_CTRL frame, so cruise state comes
      # from PEDALS: ACC_OFF means MRCC is armed but idle, ACC_ACTIVE means it is engaged.
      # Brake-only samples can arrive with both bits low mid-press; mirror the panda rx
      # guard and hold the previous state through them, else MADS sees a false
      # availability drop and force-disengages lateral.
      acc_armed = cp.vl["PEDALS"]["ACC_OFF"] == 1
      acc_active = cp.vl["PEDALS"]["ACC_ACTIVE"] == 1
      brake_free = not ret.brakePressed and not self.brake_pressed_prev
      # a wheel CANCEL turns the main state off for real and has to land even with the brake
      # down; PEDALS reacts a few frames behind the button, so the context outlives the press
      if cp.vl["CRZ_BTNS"]["CAN_OFF"] == 1:
        self.cancel_context_frames = CANCEL_CONTEXT_FRAMES
      elif self.cancel_context_frames > 0:
        self.cancel_context_frames -= 1
      if acc_armed or acc_active:
        self.cruise_available = True
      elif brake_free or self.cancel_context_frames > 0:
        self.cruise_available = False
      if acc_armed or acc_active or self.cruise_enabled or brake_free:
        self.cruise_enabled = acc_active

      # Two-master guard: engagement stays blocked until the stock radar has been silent for
      # STOCK_RADAR_GUARD_T. Before the first teardown that is the expected boot phase, not a
      # fault; once the radar has been silenced, hearing it again is a real accFaulted (and the
      # alpha-long toggle monitor's "stock radar heard" edge).
      # See docs/zoompilot/mazda-longitudinal.md, "The block wears two hats".
      if len(cp.vl_all["CRZ_INFO"]["CTR"]) > 0:
        self.stock_radar_silent_frames = 0
      else:
        self.stock_radar_silent_frames += 1

      # the radar answers a session request within ~10 ms and the session manager consumes
      # this on the same frame, so no freshness window. NRC 0x78 (response pending) is the
      # one negative response UDS clients wait through, not fail on.
      resp = cp.vl_all["RADAR_UDS_RESPONSE"]
      self.radar_session_refused = any(
        sid == 0x7F and sub == uds.SERVICE_TYPE.DIAGNOSTIC_SESSION_CONTROL and nrc != 0x78
        for sid, sub, nrc in zip(resp["SID"], resp["SUB"], resp["NRC"], strict=True))
      silenced = self.stock_radar_gone
      ret.accFaulted = self.radar_was_silenced and not silenced
      self.radar_was_silenced |= silenced

      # Gate enabled together with available: MADS engages off the enabled edge but only
      # releases off an availability falling edge, so a stock engage inside the guard window
      # would latch lateral on with no way out. A live engagement is not adopted the instant
      # the guard lifts either; the stock state has to pass through idle once.
      if not self.radar_was_silenced:
        self.cruise_enabled_blocked = True
      elif not self.cruise_enabled:
        self.cruise_enabled_blocked = False

      ret.cruiseState.available = self.cruise_available and self.radar_was_silenced
      ret.cruiseState.enabled = self.cruise_enabled and not self.cruise_enabled_blocked

      # FSC settle timer (the radar teardown gate): CAM_LANEINFO.NO_ERR_BIT is the camera's
      # boot-in-progress marker and its radar-presence check follows it; a latched ERR_BIT
      # also shows the marker clear, so it holds the timer at zero too. Not BIT2: it can stay
      # latched high for a whole ignition. See docs/zoompilot/mazda-longitudinal.md, "FSC
      # settle gate".
      laneinfo = cp_cam.vl["CAM_LANEINFO"]
      settled = cam_laneinfo_fresh and not (laneinfo["NO_ERR_BIT"] or laneinfo["ERR_BIT"])
      self.fsc_settled_frames = self.fsc_settled_frames + 1 if settled else 0
    else:
      # TODO: the signal used for available seems to be the adaptive cruise signal, instead of the main on
      #       it should be used for carState.cruiseState.nonAdaptive instead
      ret.cruiseState.available = cp.vl["CRZ_CTRL"]["CRZ_AVAILABLE"] == 1
      ret.cruiseState.enabled = cp.vl["CRZ_CTRL"]["CRZ_ACTIVE"] == 1
    self.brake_pressed_prev = ret.brakePressed
    # PEDALS.STANDSTILL is the PCM's "wheels are stopped" bit, not a stock-ACC hold state.
    # LongControl gates its starting condition on it, so reporting it under openpilot
    # longitudinal deadlocks every stop; Hyundai and Tesla report False for the same reason.
    ret.cruiseState.standstill = cp.vl["PEDALS"]["STANDSTILL"] == 1 and not self.CP.openpilotLongitudinalControl
    ret.cruiseState.speed = cp.vl["CRZ_EVENTS"]["CRZ_SPEED"] * CV.KPH_TO_MS

    # stock lkas should be on
    # TODO: is this needed?
    ret.invalidLkasSetting = cam_laneinfo_fresh and cp_cam.vl["CAM_LANEINFO"]["LANE_LINES"] == 0

    if ret.cruiseState.enabled:
      if not self.lkas_allowed_speed and self.acc_active_last:
        self.low_speed_alert = True
      else:
        self.low_speed_alert = False
    ret.lowSpeedAlert = self.low_speed_alert

    # Check if LKAS is disabled due to lack of driver torque when all other states indicate
    # it should be enabled (steer lockout). Don't warn until we actually get lkas active
    # and lose it again, i.e, after initial lkas activation
    if not self.CP.flags & MazdaFlags.STEER_TO_ZERO_EPS:
      ret.steerFaultTemporary = self.lkas_allowed_speed and lkas_blocked
    else:
      # 2022 EPS: LKAS_BLOCK alone is not a fault, a blocked EPS usually still delivers part
      # of the request. Only a total block held long enough at road speed is surfaced, once
      # the non-delivery latch has zeroed the command and nobody is steering.
      ret.steerFaultTemporary = self.steer_undelivered_alert

    self.acc_active_last = ret.cruiseState.enabled

    self.crz_btns_counter = cp.vl["CRZ_BTNS"]["CTR"]

    # camera signals
    self.cam_lkas = cp_cam.vl["CAM_LKAS"]
    self.cam_laneinfo = cp_cam.vl["CAM_LANEINFO"]
    ret.steerFaultPermanent = cp_cam.vl["CAM_LKAS"]["ERR_BIT_1"] == 1

    # cruise control button events: distance, inc, dec, resume, cancel, and main
    prev_distance_button = self.distance_button
    prev_accel_button = self.accel_button
    prev_decel_button = self.decel_button
    prev_cancel_button = self.cancel_button
    prev_resume_button = self.resume_button
    prev_main_button = self.main_button
    self.distance_button = cp.vl["CRZ_BTNS"]["DISTANCE_LESS"]
    # the wheel "+" button is SET_P, not RES; RES is the resume button
    self.accel_button = cp.vl["CRZ_BTNS"]["SET_P"]
    self.decel_button = cp.vl["CRZ_BTNS"]["SET_M"]
    # CAN_OFF carries the cancel intent; without the event ICBM's readiness gate keeps
    # sending cancel=0 frames over the driver's press
    self.cancel_button = cp.vl["CRZ_BTNS"]["CAN_OFF"]
    self.resume_button = cp.vl["CRZ_BTNS"]["RES"]
    self.main_button = int(cp.vl["CRZ_BTNS"]["MODE_X"] == 1 and cp.vl["CRZ_BTNS"]["MODE_Y"] == 1)

    ret.buttonEvents = [
      *create_button_events(self.distance_button, prev_distance_button, {1: ButtonType.gapAdjustCruise}),
      *create_button_events(self.accel_button, prev_accel_button, {1: ButtonType.accelCruise}),
      *create_button_events(self.decel_button, prev_decel_button, {1: ButtonType.decelCruise}),
      *create_button_events(self.cancel_button, prev_cancel_button, {1: ButtonType.cancel}),
      *create_button_events(self.resume_button, prev_resume_button, {1: ButtonType.resumeCruise}),
      *create_button_events(self.main_button, prev_main_button, {1: ButtonType.mainCruise}),
    ]

    CarStateExt.update(self, ret, ret_sp, can_parsers)

    return ret, ret_sp

  @staticmethod
  def get_can_parsers(CP, CP_SP):
    pt_messages = []
    if CP.openpilotLongitudinalControl:
      # no liveness checks: the stock frame disappears after the radar teardown (the
      # two-master guard watches for it), and the UDS response only arrives when answered
      pt_messages.append(("CRZ_INFO", float("nan")))
      pt_messages.append(("RADAR_UDS_RESPONSE", float("nan")))
    cam_messages = [
      # no liveness checks: read opportunistically, and not every Mazda camera sends them
      # (the 2016-20 CX-9 sends none), so a missing frame must not fail canValid
      ("CAM_LANEINFO", float("nan")),
      ("CAM_TRAFFIC_SIGNS", float("nan")),
      ("CAM_EMPTY", float("nan")),
      ("CAM_PEDESTRIAN", float("nan")),
    ]
    return {
      Bus.pt: CANParser(DBC[CP.carFingerprint][Bus.pt], pt_messages, 0),
      Bus.cam: CANParser(DBC[CP.carFingerprint][Bus.pt], cam_messages, 2),
    }
