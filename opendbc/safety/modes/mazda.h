#pragma once

#include "opendbc/safety/declarations.h"

// CAN msgs we care about
#define MAZDA_LKAS          0x243U
#define MAZDA_LKAS_HUD      0x440U
#define MAZDA_CRZ_INFO      0x21bU
#define MAZDA_CRZ_CTRL      0x21cU
#define MAZDA_CRZ_BTNS      0x09dU
// TJA_BUTTON: DBC start bit 11, 1-bit Motorola == Intel bit 11 (byte 1 bit 3).
#define MAZDA_TJA_BUTTON_BIT 11U
#define MAZDA_RADAR_STATIC  0x499U
#define MAZDA_RADAR_TRACK_1 0x361U
#define MAZDA_RADAR_TRACK_2 0x362U
#define MAZDA_RADAR_TRACK_3 0x363U
#define MAZDA_RADAR_TRACK_4 0x364U
#define MAZDA_RADAR_TRACK_5 0x365U
#define MAZDA_RADAR_TRACK_6 0x366U
#define MAZDA_RADAR_UDS     0x764U
#define MAZDA_STEER_TORQUE  0x240U
#define MAZDA_ENGINE_DATA   0x202U
#define MAZDA_PEDALS        0x165U

// CAN bus numbers
#define MAZDA_MAIN 0
#define MAZDA_CAM  2

#define MAZDA_PARAM_LONGITUDINAL 1U
#define MAZDA_PARAM_STEER_TO_ZERO 2U
#define MAZDA_PARAM_TJA_MADS 4U

// CRZ_BTNS frames (10 Hz) an engage press stays fresh. Every logged engagement shows the
// press 30-70 ms before PEDALS.ACC_ACTIVE rises (104-engagement census, zero genuine
// button-less engagements), so 1 s is generous.
#define MAZDA_ENGAGE_BTN_WINDOW 10U

static bool mazda_longitudinal = false;
static bool mazda_steer_to_zero = false;
static bool mazda_tja_mads = false;
static bool mazda_acc_armed = false;
static uint32_t mazda_engage_btn_frames = 0U;

static bool mazda_mrcc_off_msg_valid(const CANPacket_t *msg) {
  // Exact active-low MRCC master tap captured on CX-5 2022. CTR occupies the
  // variable bits in byte 3; all other button and payload bits stay pinned.
  return (GET_LEN(msg) == 8U) && (msg->data[0] == 0x00U) &&
         (msg->data[1] == 0x81U) && (msg->data[2] == 0xfeU) &&
         ((msg->data[3] & 0xc3U) == 0xc0U) && (msg->data[4] == 0x00U) &&
         (msg->data[5] == 0x00U) && (msg->data[6] == 0x00U) && (msg->data[7] == 0x00U);
}

// Radar-mastery latch mirrored from carstate: the software gates cruise availability on the
// stock radar having been silent for 1 s (STOCK_RADAR_GUARD_T), and MADS keys lateral off
// acc_main_on's rising edge. Without the same latch here the panda's edge fires at boot
// (MRCC main persists over ignition), is consumed and exited long before the software
// engages, and the software's whole MADS window then transmits into rejections -- starving
// the EPS of 0x243 while the camera's own copy is relay-blocked, which latches the dash
// LKAS error (routes 00000116/00000117, 2026-08-27). The rx hook never sees the stock
// CRZ_INFO (it is deliberately not an rx check: it goes stale at the teardown), so the
// observable stand-in is our own first synthetic CRZ_INFO tx -- the controller starts
// emitting it the moment the UDS teardown lands, the same moment the stock radar goes
// quiet, and the software's silence guard runs 1 s from there. PEDALS is the 50 Hz clock.
#define MAZDA_RADAR_SILENT_FRAMES 50U
static bool mazda_radar_mastered = false;
static uint32_t mazda_mastered_pedals_frames = 0U;
static bool mazda_radar_was_silenced = false;

// With longitudinal control the stock radar is silenced and openpilot replays its frames,
// so allowed tx patterns are pinned to byte-exact stock captures wherever possible.

static bool mazda_radar_static_msg_valid(const CANPacket_t *msg) {
  return (msg->data[0] == 0x00U) && (msg->data[1] == 0x08U) &&
         (msg->data[2] == 0xc0U) && (msg->data[3] == 0x00U) &&
         (msg->data[4] == 0x00U) && (msg->data[5] == 0x00U) &&
         (msg->data[6] == 0x00U) && (msg->data[7] == 0x00U);
}

static bool mazda_empty_radar_track_msg_valid(const CANPacket_t *msg) {
  bool valid = false;

  if ((msg->addr == MAZDA_RADAR_TRACK_1) || (msg->addr == MAZDA_RADAR_TRACK_2) ||
      (msg->addr == MAZDA_RADAR_TRACK_3) || (msg->addr == MAZDA_RADAR_TRACK_4)) {
    valid = (msg->data[0] == 0xffU) && (msg->data[1] == 0xf7U) &&
            (msg->data[2] == 0xfeU) && (msg->data[3] == 0xfeU) &&
            (msg->data[4] == 0x1fU);

    if (msg->addr == MAZDA_RADAR_TRACK_2) {
      valid = valid && (msg->data[5] == 0xc7U) && (msg->data[6] == 0x8cU) &&
              ((msg->data[7] & 0xf0U) == 0x80U);
    } else if ((msg->addr == MAZDA_RADAR_TRACK_3) || (msg->addr == MAZDA_RADAR_TRACK_4)) {
      valid = valid && (msg->data[5] == 0xc0U) && (msg->data[6] == 0x00U) &&
              ((msg->data[7] & 0xf0U) == 0x00U);
    } else {
      valid = valid && (msg->data[5] == 0xc0U) && (msg->data[6] == 0x00U) &&
              ((msg->data[7] & 0xf0U) == 0x80U);
    }
  } else if ((msg->addr == MAZDA_RADAR_TRACK_5) || (msg->addr == MAZDA_RADAR_TRACK_6)) {
    valid = (msg->data[0] == 0xffU) && (msg->data[1] == 0xf7U) &&
            (msg->data[2] == 0xfeU) && (msg->data[3] == 0x7fU) &&
            (msg->data[4] == 0xfbU) && (msg->data[5] == 0xffU) &&
            (msg->data[6] == 0x3fU) && ((msg->data[7] & 0xf0U) == 0xc0U);
  } else {
    valid = false;
  }

  return valid;
}

static bool mazda_synthetic_lead_radar_track_msg_valid(const CANPacket_t *msg) {
  // The controller writes the lead it is following into the occupied-slot capture:
  // DIST_OBJ fills data[0] and the high nibble of data[1], RELV_OBJ fills data[3] and the
  // high 3 bits of data[4]. Those fields are free; every bit the template owns must still
  // match it exactly. A byte-exact check here silently dropped every real-lead frame and
  // starved the camera of the track (route 6bb2dc61c4: 982 asked, 0 transmitted).
  return (msg->addr == MAZDA_RADAR_TRACK_4) &&
         ((msg->data[1] & 0x0fU) == 0x00U) && (msg->data[2] == 0x00U) &&
         ((msg->data[4] & 0x1fU) == 0x1dU) && (msg->data[5] == 0xc0U) &&
         (msg->data[6] == 0x00U) && ((msg->data[7] & 0xf0U) == 0x00U);
}

static bool mazda_radar_track_msg_valid(const CANPacket_t *msg) {
  // ignition to ignition, engaged or not, and the controller mirrors that. Gating it on
  // controls_allowed silently killed 0x364 at every disengagement while CRZ_CTRL still said
  // has_lead=1, the exact track/ctrl disagreement the camera faults on.
  return mazda_empty_radar_track_msg_valid(msg) ||
         mazda_synthetic_lead_radar_track_msg_valid(msg);
}

// track msgs coming from OP so that we know what CAM msgs to drop and what to forward
static void mazda_rx_hook(const CANPacket_t *msg) {
  if ((int)msg->bus == MAZDA_MAIN) {
    if (msg->addr == MAZDA_ENGINE_DATA) {
      // sample speed: scale by 0.01 to get kph
      int speed = (msg->data[2] << 8) | msg->data[3];
      vehicle_moving = speed > 10; // moving when speed > 0.1 kph
    }

    if (msg->addr == MAZDA_STEER_TORQUE) {
      int torque_driver_new = msg->data[0] - 127U;
      // update array of samples
      update_sample(&torque_driver, torque_driver_new);
    }

    // Longitudinal PCM: enter/exit controls_allowed on OEM MRCC engaged (CRZ_ACTIVE).
    // When TJA_MADS is set, do not map CRZ_AVAILABLE onto acc_main_on. Mazda MADS
    // lateral authorization is TJA-only; feeding MRCC into acc_main would grant
    // lateral without TJA and revoke it when OEM cruise drops.
    if ((msg->addr == MAZDA_CRZ_CTRL) && !mazda_longitudinal) {
      bool cruise_engaged = msg->data[0] & 0x8U;
      pcm_cruise_check(cruise_engaged);
      mazda_acc_armed = GET_BIT(msg, 17U);
      if (!mazda_tja_mads) {
        acc_main_on = GET_BIT(msg, 17U);
      }
    }

    if (msg->addr == MAZDA_CRZ_BTNS) {
      if (mazda_tja_mads) {
        mads_button_press = GET_BIT(msg, MAZDA_TJA_BUTTON_BIT) ? MADS_BUTTON_PRESSED : MADS_BUTTON_NOT_PRESSED;
      }

      if (mazda_longitudinal) {
        // ensure the driver's cancel press always exits longitudinal controls
        bool cancel = GET_BIT(msg, 0U);
        if (cancel) {
          controls_allowed = false;
        }
        // RES, SET_P or SET_M: the driver-intent half of the engagement qualifier below
        if (GET_BIT(msg, 2U) || GET_BIT(msg, 4U) || GET_BIT(msg, 5U)) {
          mazda_engage_btn_frames = MAZDA_ENGAGE_BTN_WINDOW;
        } else if (mazda_engage_btn_frames > 0U) {
          mazda_engage_btn_frames -= 1U;
        }
      }
    }

    if (msg->addr == MAZDA_ENGINE_DATA) {
      gas_pressed = (msg->data[4] || (msg->data[5] & 0xF0U));
    }

    if (msg->addr == MAZDA_PEDALS) {
      bool brake = (msg->data[0] & 0x10U);
      if (mazda_longitudinal) {
        // PEDALS clocks the silence guard from the mastery point: sticky once set, like
        // carstate's radar_was_silenced (a returning radar is carstate's accFaulted, not a
        // MADS exit)
        if (mazda_radar_mastered && (mazda_mastered_pedals_frames < MAZDA_RADAR_SILENT_FRAMES)) {
          mazda_mastered_pedals_frames += 1U;
        }
        mazda_radar_was_silenced = mazda_radar_was_silenced ||
                                   (mazda_mastered_pedals_frames >= MAZDA_RADAR_SILENT_FRAMES);

        // The radar teardown removes the stock CRZ_CTRL frame, so derive cruise state from
        // PEDALS: ACC_OFF (bit 2) means MRCC is armed but idle, ACC_ACTIVE (bit 3) means
        // engaged. Brake-only samples can arrive with both bits low mid-press; skip those
        // so they are not mistaken for an ACC-off edge.
        bool cruise_engaged = GET_BIT(msg, 3U);
        bool acc_armed = GET_BIT(msg, 2U) || cruise_engaged;

        if (acc_armed || cruise_engaged_prev || (!brake && !brake_pressed_prev)) {
          mazda_acc_armed = acc_armed;
          if (!mazda_tja_mads) {
            // gated on the latch so the MADS arming edge lands on the same frame as the
            // software's cruiseState.available, and both machines arm together
            acc_main_on = acc_armed && mazda_radar_was_silenced;
          }
          // Arm only on an engaged rising edge backed by a recent SET/RES press, the
          // hyundai_common form: ACC_ACTIVE alone is the body answering frames we fabricate.
          if (cruise_engaged && !cruise_engaged_prev && (mazda_engage_btn_frames > 0U)) {
            controls_allowed = true;
          }
          if (!cruise_engaged) {
            controls_allowed = false;
          }
          cruise_engaged_prev = cruise_engaged;
        }
      }
      brake_pressed = brake;
    }
  }
}

static bool mazda_tx_hook(const CANPacket_t *msg) {
  // Lateral envelope follows STEER_TO_ZERO in safetyParam, independently of TJA_MADS.
  const TorqueSteeringLimits MAZDA_STEERING_LIMITS_HIGH = {
    .max_torque = 1200,
    .max_rate_up = 12,
    .max_rate_down = 25,
    .max_rt_delta = 384,
    .driver_torque_multiplier = 15,
    .driver_torque_allowance = 15,
    .type = TorqueDriverLimited,
  };
  const TorqueSteeringLimits MAZDA_STEERING_LIMITS_STOCK = {
    .max_torque = 800,
    .max_rate_up = 10,
    .max_rate_down = 25,
    .max_rt_delta = 384,
    .driver_torque_multiplier = 1,
    .driver_torque_allowance = 15,
    .type = TorqueDriverLimited,
  };
  const TorqueSteeringLimits limits = mazda_steer_to_zero ? MAZDA_STEERING_LIMITS_HIGH :
                                                        MAZDA_STEERING_LIMITS_STOCK;

  // CRZ_INFO.ACCEL_CMD is raw units of 0.001 m/s2 (offset removed below), so this is the
  // ISO window: 2.0 / -3.5 m/s2. Stock MRCC itself commands down to raw -3891 in lead stops.
  const LongitudinalLimits MAZDA_LONG_LIMITS = {
    .max_accel = 2000,
    .min_accel = -3500,
    .inactive_accel = 0,
  };

  bool tx = true;
  bool main_bus = msg->bus == (unsigned char)MAZDA_MAIN;
  bool long_replacement_bus = main_bus || (msg->bus == (unsigned char)MAZDA_CAM);

  // steer cmd checks
  if (main_bus && (msg->addr == MAZDA_LKAS)) {
    int desired_torque = (((msg->data[0] & 0x0FU) << 8) | msg->data[1]) - 2048U;

    if (steer_torque_cmd_checks(desired_torque, -1, limits)) {
      tx = false;
    }
  }

  if (mazda_longitudinal && long_replacement_bus && (msg->addr == MAZDA_CRZ_INFO)) {
    // the stock standby pattern pegs the command field high; allow it byte-exactly
    // (checksum included) instead of decoding it as a huge accel command
    bool stock_standby = (msg->data[0] == 0x01U) && (msg->data[1] == 0xffU) &&
                         (msg->data[2] == 0xe3U) && (msg->data[3] == 0xffU) &&
                         (msg->data[4] == 0xc0U) && (msg->data[5] == 0x00U) &&
                         ((msg->data[6] & 0xf0U) == 0x00U) &&
                         (msg->data[7] == ((0x5dU - msg->data[6]) & 0xffU));

    // 13-bit ACCEL_CMD: data[2] low bits, data[3], data[4] high bits, offset 4096.
    // Decode unsigned first (MISRA), then subtract the DBC offset.
    const uint32_t raw_accel = (((uint32_t)msg->data[2] & 0x3U) << 11) |
                               ((uint32_t)msg->data[3] << 3) |
                               ((uint32_t)msg->data[4] >> 5);
    int desired_accel = (int)raw_accel - 4096;
    if (!stock_standby && longitudinal_accel_checks(desired_accel, MAZDA_LONG_LIMITS)) {
      tx = false;
    }

    // ACC_ACTIVE (bit 33) mirrors CRZ_CTRL's CRZ_ACTIVE gate: an engaged-claiming accel
    // frame must not flow while controls are not allowed.
    bool acc_active = GET_BIT(msg, 33U);
    if (!controls_allowed && acc_active) {
      tx = false;
    }
  }

  if (mazda_longitudinal && long_replacement_bus && (msg->addr == MAZDA_CRZ_CTRL)) {
    bool cruise_active = GET_BIT(msg, 3U);
    if (!controls_allowed && cruise_active) {
      tx = false;
    }
  }

  if (mazda_longitudinal && long_replacement_bus && (msg->addr == MAZDA_RADAR_STATIC)) {
    if (!mazda_radar_static_msg_valid(msg)) {
      tx = false;
    }
  }

  if (mazda_longitudinal && long_replacement_bus && (msg->addr >= MAZDA_RADAR_TRACK_1) && (msg->addr <= MAZDA_RADAR_TRACK_6)) {
    if (!mazda_radar_track_msg_valid(msg)) {
      tx = false;
    }
  }

  if (mazda_longitudinal && main_bus && (msg->addr == MAZDA_RADAR_UDS)) {
    // only tester present and default/programming session control; flashing services stay blocked
    bool tester_present = (msg->data[0] == 0x02U) && (msg->data[1] == 0x3eU) && (msg->data[2] == 0x80U);
    bool session_control = (msg->data[0] == 0x02U) && (msg->data[1] == 0x10U) &&
                           ((msg->data[2] == 0x01U) || (msg->data[2] == 0x02U));
    if (!tester_present && !session_control) {
      tx = false;
    }
  }

  // cruise buttons check
  if (main_bus && (msg->addr == MAZDA_CRZ_BTNS)) {
    // allow resume spamming while controls allowed, but
    // only allow cancel while controls not allowed
    bool cancel_cmd = (msg->data[0] == 0x1U);
    // TJA also arms MRCC on the shared main bus. Permit only the byte-exact
    // active-low MRCC-off tap, and only while Mazda reports MRCC already armed.
    const bool mrcc_off_cmd = mazda_tja_mads && mazda_acc_armed && mazda_mrcc_off_msg_valid(msg);
    if (!controls_allowed && !cancel_cmd && !mrcc_off_cmd) {
      tx = false;
    }
  }

  // radar mastery: our first synthetic CRZ_INFO on the main bus marks the teardown landing,
  // the same moment the stock radar goes quiet
  if (tx && main_bus && (msg->addr == MAZDA_CRZ_INFO) && mazda_longitudinal) {
    mazda_radar_mastered = true;
  }

  return tx;
}

// Mutate only the bus0->bus2 forward copy of CRZ_BTNS. Panda RX and the OEM body/MRCC
// keep the original bus0 frame, including physical TJA. Requires TJA_MADS and the MADS
// feature (system_enabled), not heartbeat_engaged_mads, so the first button edge cannot
// leak to the FSC during a USB-heartbeat transition.
static void mazda_fwd_modify(int bus_num, CANPacket_t *msg) {
  if (mazda_tja_mads && m_mads_state.system_enabled && (bus_num == MAZDA_MAIN) &&
      (msg->addr == MAZDA_CRZ_BTNS) && (GET_LEN(msg) >= 2U)) {
    msg->data[MAZDA_TJA_BUTTON_BIT / 8U] &= (uint8_t)~(1U << (MAZDA_TJA_BUTTON_BIT % 8U));
  }
}

static safety_config mazda_init(uint16_t param) {
  mazda_engage_btn_frames = 0U;
  mazda_radar_mastered = false;
  mazda_mastered_pedals_frames = 0U;
  mazda_radar_was_silenced = false;
  static const CanMsg MAZDA_TX_MSGS[] = {
    {MAZDA_LKAS, 0, 8, .check_relay = true},
    {MAZDA_CRZ_BTNS, 0, 8, .check_relay = false},
    {MAZDA_LKAS_HUD, 0, 8, .check_relay = true},
  };

  static const CanMsg MAZDA_LONG_TX_MSGS[] = {
    {MAZDA_LKAS, 0, 8, .check_relay = true},
    {MAZDA_CRZ_BTNS, 0, 8, .check_relay = false},
    {MAZDA_LKAS_HUD, 0, 8, .check_relay = true},
    {MAZDA_CRZ_INFO, 0, 8, .check_relay = false},
    {MAZDA_CRZ_CTRL, 0, 8, .check_relay = false},
    {MAZDA_RADAR_STATIC, 0, 8, .check_relay = false},
    {MAZDA_RADAR_TRACK_1, 0, 8, .check_relay = false},
    {MAZDA_RADAR_TRACK_2, 0, 8, .check_relay = false},
    {MAZDA_RADAR_TRACK_3, 0, 8, .check_relay = false},
    {MAZDA_RADAR_TRACK_4, 0, 8, .check_relay = false},
    {MAZDA_RADAR_TRACK_5, 0, 8, .check_relay = false},
    {MAZDA_RADAR_TRACK_6, 0, 8, .check_relay = false},
    {MAZDA_RADAR_UDS, 0, 8, .check_relay = false},
    {MAZDA_CRZ_INFO, MAZDA_CAM, 8, .check_relay = false},
    {MAZDA_CRZ_CTRL, MAZDA_CAM, 8, .check_relay = false},
    {MAZDA_RADAR_STATIC, MAZDA_CAM, 8, .check_relay = false},
    {MAZDA_RADAR_TRACK_1, MAZDA_CAM, 8, .check_relay = false},
    {MAZDA_RADAR_TRACK_2, MAZDA_CAM, 8, .check_relay = false},
    {MAZDA_RADAR_TRACK_3, MAZDA_CAM, 8, .check_relay = false},
    {MAZDA_RADAR_TRACK_4, MAZDA_CAM, 8, .check_relay = false},
    {MAZDA_RADAR_TRACK_5, MAZDA_CAM, 8, .check_relay = false},
    {MAZDA_RADAR_TRACK_6, MAZDA_CAM, 8, .check_relay = false},
  };

  static RxCheck mazda_rx_checks[] = {
    {.msg = {{MAZDA_CRZ_CTRL,     0, 8, 50U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},
    {.msg = {{MAZDA_CRZ_BTNS,     0, 8, 10U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},
    {.msg = {{MAZDA_STEER_TORQUE, 0, 8, 83U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},
    {.msg = {{MAZDA_ENGINE_DATA,  0, 8, 100U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},
    {.msg = {{MAZDA_PEDALS,       0, 8, 50U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},
  };

  // no CRZ_CTRL check: the stock radar frame disappears after the teardown
  static RxCheck mazda_long_rx_checks[] = {
    {.msg = {{MAZDA_CRZ_BTNS,     0, 8, 10U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},
    {.msg = {{MAZDA_STEER_TORQUE, 0, 8, 83U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},
    {.msg = {{MAZDA_ENGINE_DATA,  0, 8, 100U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},
    {.msg = {{MAZDA_PEDALS,       0, 8, 50U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},
  };

  mazda_longitudinal = GET_FLAG(param, MAZDA_PARAM_LONGITUDINAL);
  mazda_steer_to_zero = GET_FLAG(param, MAZDA_PARAM_STEER_TO_ZERO);
  mazda_tja_mads = GET_FLAG(param, MAZDA_PARAM_TJA_MADS);
  mazda_acc_armed = false;
  mazda_engage_btn_frames = 0U;
  acc_main_on = false;
  // TJA is the only lateral authorization source; MRCC/pcm cruise must not grant it.
  mads_set_op_controls_allowed_requests_lateral(!mazda_tja_mads);

  return mazda_longitudinal ? BUILD_SAFETY_CFG(mazda_long_rx_checks, MAZDA_LONG_TX_MSGS) :
                              BUILD_SAFETY_CFG(mazda_rx_checks, MAZDA_TX_MSGS);
}

const safety_hooks mazda_hooks = {
  .init = mazda_init,
  .rx = mazda_rx_hook,
  .tx = mazda_tx_hook,
  .fwd_modify = mazda_fwd_modify,
};
