#pragma once

#include "opendbc/safety/declarations.h"

// CAN msgs we care about
#define MAZDA_LKAS          0x243U
#define MAZDA_LKAS_HUD      0x440U
#define MAZDA_CRZ_INFO      0x21bU
#define MAZDA_CRZ_CTRL      0x21cU
#define MAZDA_CRZ_BTNS      0x09dU
// Physical TJA button, DBC start bit 11 (byte 1, bit 3); trims without the button hold it low.
#define MAZDA_TJA_BUTTON_BIT 11U
// MRCC ON intent (MODE_Y), DBC start bit 13; the KE's OFF button sets only MODE_X.
#define MAZDA_MODE_Y_BIT 13U
// sunnypilot safety param: the TJA button is the MADS lateral switch
#define MAZDA_PARAM_TJA_BUTTON 8U
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
// The measured gen1 EPS envelope; the interface sets it on every Mazda it admits.
#define MAZDA_PARAM_EPS_HW 2U
// A Smart CAN Filter keeps the radar online under alpha long (develop's MAZDA_PARAM_SP_CAN_FILTER, in the
// second safety word there): FILTER_CTRL may go out, the UDS takeover and synthetic radar frames may not.
#define MAZDA_PARAM_CAN_FILTER 16U
// Smart CAN Filter control: magic, subcommand 0x2e (FILTER_CTRL)
#define MAZDA_CAN_FILTER_CTRL 0x2A0U
// The camera-only TJA press-off (develop 0f6ce31f): at least this far apart, at most MAZDA_CAM_PRESS_MAX in a
// burst, and the burst ends after MAZDA_CAM_PRESS_QUIET_US without one.
#define MAZDA_CAM_PRESS_MIN_US 900000U
#define MAZDA_CAM_PRESS_MAX 4U
#define MAZDA_CAM_PRESS_QUIET_US 5000000U

// Keep SET/RES intent fresh until PEDALS reports engagement.
#define MAZDA_ENGAGE_BTN_WINDOW 10U
// Both PEDALS cruise bits low for this many samples is a main-off: values'
// MAIN_OFF_DEBOUNCE_SAMPLES, which carstate counts the same way.
#define MAZDA_MAIN_OFF_DEBOUNCE 10U

static bool mazda_port_longitudinal = false;
// Declared by the driver: the TJA button owns lateral and MRCC no longer drives the main edge.
static bool mazda_port_eps_hw = false;
// Live cruise arming from PEDALS, both longitudinal modes (carstate mrcc_armed_raw).
static bool mazda_port_acc_armed = false;
static uint32_t mazda_port_engage_btn_frames = 0U;
static uint32_t mazda_port_main_off_samples = 0U;
static uint32_t mazda_port_cam_press_ts = 0U;     // camera-only TJA press-off: last press sent
static uint32_t mazda_port_cam_press_count = 0U;  // presses in the current burst

static bool mazda_port_cam_tja_press_msg_valid(const CANPacket_t *msg) {
  // Exact wheel TJA press over the idle pattern (route 00000097: 0009ffcc00000000): every other
  // button bit 0, its inversions 1, CTR the only free bits in byte 3.
  return (GET_LEN(msg) == 8U) && (msg->data[0] == 0x00U) &&
         (msg->data[1] == 0x09U) && (msg->data[2] == 0xffU) &&
         ((msg->data[3] & 0xc3U) == 0xc0U) && (msg->data[4] == 0x00U) &&
         (msg->data[5] == 0x00U) && (msg->data[6] == 0x00U) && (msg->data[7] == 0x00U);
}

static bool mazda_port_mrcc_off_msg_valid(const CANPacket_t *msg) {
  // Exact active-low MRCC master tap. CTR occupies the variable bits in byte 3;
  // all other buttons and payload bits remain pinned.
  return (GET_LEN(msg) == 8U) && (msg->data[0] == 0x00U) &&
         (msg->data[1] == 0x81U) && (msg->data[2] == 0xfeU) &&
         ((msg->data[3] & 0xc3U) == 0xc0U) && (msg->data[4] == 0x00U) &&
         (msg->data[5] == 0x00U) && (msg->data[6] == 0x00U) && (msg->data[7] == 0x00U);
}

// Replaced-radar frames must be one of the stock captures (mazdacan.py): every byte pinned, and on
// the tracks only the counter nibble in byte 7 free. The 0x499 static frame has one capture per
// radar generation; the controller picks the dialect.
typedef struct {
  unsigned int addr;
  uint8_t data[8];
  uint8_t last_byte_mask;
} MazdaPortRadarCapture;

static bool mazda_port_radar_capture_msg_valid(const CANPacket_t *msg) {
  static const MazdaPortRadarCapture MAZDA_RADAR_CAPTURES[] = {
    {MAZDA_RADAR_STATIC,  {0x00U, 0x08U, 0xe0U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U}, 0xffU},  // 2022 family
    {MAZDA_RADAR_STATIC,  {0x00U, 0x98U, 0x40U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U}, 0xffU},  // G46L
    {MAZDA_RADAR_TRACK_1, {0xffU, 0xf7U, 0xfeU, 0xfeU, 0x1fU, 0xc0U, 0x00U, 0x80U}, 0xf0U},
    {MAZDA_RADAR_TRACK_2, {0xffU, 0xf7U, 0xfeU, 0xfeU, 0x1fU, 0xc7U, 0x8cU, 0x80U}, 0xf0U},
    {MAZDA_RADAR_TRACK_3, {0xffU, 0xf7U, 0xfeU, 0xfeU, 0x1fU, 0xc0U, 0x00U, 0x00U}, 0xf0U},
    {MAZDA_RADAR_TRACK_4, {0xffU, 0xf7U, 0xfeU, 0xfeU, 0x1fU, 0xc0U, 0x00U, 0x00U}, 0xf0U},
    {MAZDA_RADAR_TRACK_5, {0xffU, 0xf7U, 0xfeU, 0x7fU, 0xfbU, 0xffU, 0x3fU, 0xc0U}, 0xf0U},
    {MAZDA_RADAR_TRACK_6, {0xffU, 0xf7U, 0xfeU, 0x7fU, 0xfbU, 0xffU, 0x3fU, 0xc0U}, 0xf0U},
  };

  bool valid = false;

  for (unsigned int i = 0U; i < (sizeof(MAZDA_RADAR_CAPTURES) / sizeof(MazdaPortRadarCapture)); i++) {
    const MazdaPortRadarCapture *capture = &MAZDA_RADAR_CAPTURES[i];
    if (capture->addr == msg->addr) {
      bool match = (msg->data[7] & capture->last_byte_mask) == capture->data[7];
      for (int j = 0; j < 7; j++) {
        match = match && (msg->data[j] == capture->data[j]);
      }
      valid = valid || match;
    }
  }

  return valid;
}

// track msgs coming from OP so that we know what CAM msgs to drop and what to forward
static void mazda_port_rx_hook(const CANPacket_t *msg) {
  const bool was_allowed = controls_allowed;
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

    // enter controls on rising edge of ACC, exit controls on ACC off
    if ((msg->addr == MAZDA_CRZ_CTRL) && !mazda_port_longitudinal) {
      bool cruise_engaged = msg->data[0] & 0x8U;
      pcm_cruise_check(cruise_engaged);
      // With the TJA button owning lateral, MRCC no longer drives the MADS main edge: its
      // falling edge would exit the panda's lateral while the software's MADS stays on.
      if (!mazda_port_tja_button) {
        acc_main_on = GET_BIT(msg, 17U);
      }
    }


    if ((msg->addr == MAZDA_CRZ_BTNS) && mazda_port_longitudinal) {
      // A physical cancel press always exits controls.
      if (GET_BIT(msg, 0U)) {
        controls_allowed = false;
      }
      // Record SET/RES intent for the engagement qualifier below.
      if (GET_BIT(msg, 2U) || GET_BIT(msg, 4U) || GET_BIT(msg, 5U)) {
        mazda_port_engage_btn_frames = MAZDA_ENGAGE_BTN_WINDOW;
      } else if (mazda_port_engage_btn_frames > 0U) {
        mazda_port_engage_btn_frames -= 1U;
      } else {
      }
    }

    if (msg->addr == MAZDA_ENGINE_DATA) {
      gas_pressed = (msg->data[4] || (msg->data[5] & 0xF0U));
    }

    if (msg->addr == MAZDA_PEDALS) {
      bool brake = (msg->data[0] & 0x10U);
      // The live MRCC arm from the body's own PEDALS bits, the sample mrcc_armed_raw reads, so the
      // MRCC-off exception opens on the frame the controller first sends (CRZ_CTRL lags it).
      mazda_port_acc_armed = GET_BIT(msg, 2U) || GET_BIT(msg, 3U);
      if (mazda_port_longitudinal) {
        // Derive cruise state from PEDALS after radar teardown.
        bool cruise_engaged = GET_BIT(msg, 3U);
        bool brake_free = !brake && !brake_pressed_prev;

        // Main mirrors carstate's cruise_available sample for sample: a main that falls on one side
        // only steers MADS into rejected frames (docs/zoompilot/mazda-longitudinal.md).
        if (mazda_port_tja_button) {
          // the button is the lateral switch; MRCC is cruise only
        } else if (mazda_port_acc_armed) {
          // The radar takeover gates cruise (controls_allowed below), never main.
          acc_main_on = true;
          mazda_port_main_off_samples = 0U;
        } else {
          if (mazda_port_main_off_samples < MAZDA_MAIN_OFF_DEBOUNCE) {
            mazda_port_main_off_samples += 1U;
          }
          if (mazda_port_main_off_samples >= MAZDA_MAIN_OFF_DEBOUNCE) {
            acc_main_on = false;
          }
        }

        if (mazda_port_acc_armed || cruise_engaged_prev || brake_free) {
          // Require recent SET/RES intent on the engaged edge; ACC_ACTIVE alone may acknowledge
          // synthetic traffic rather than a driver request.
          if (cruise_engaged && !cruise_engaged_prev && (mazda_port_engage_btn_frames > 0U)) {
            controls_allowed = true;
          }
          if (!cruise_engaged) {
            controls_allowed = false;
          }
          cruise_engaged_prev = cruise_engaged;
        }
      }
      brake_pressed = brake;
      if (mazda_port_tja_button && mazda_aol_enabled) {
        acc_main_on = true;
      }
    }
  }
  if (mazda_aol_enabled) {
    CANPacket_t observed = *msg;
    if (mazda_port_longitudinal && msg_matches(msg, MAZDA_PEDALS, 0U, 8U)) {
      observed.addr = MAZDA_CRZ_CTRL;
      observed.data[0] = GET_BIT(msg, 3U) ? 8U : 0U;
      observed.data[2] = (mazda_port_tja_button || acc_main_on) ? 2U : 0U;
      mazda_aol_rx(&observed);
      mazda_aol_engagement(&observed, GET_BIT(msg, 3U), !was_allowed && controls_allowed);
    } else if (mazda_port_longitudinal && msg_matches(msg, MAZDA_CRZ_CTRL, 0U, 8U)) {
      // PEDALS remains the main-switch witness while radar frames are replaced.
    } else if (mazda_port_tja_button && msg_matches(msg, MAZDA_CRZ_BTNS, 0U, 8U)) {
      observed.data[0] = GET_BIT(msg, MAZDA_TJA_BUTTON_BIT) ? 0x10U : 0U;
      mazda_aol_rx(&observed);
    } else if (!mazda_port_tja_button && msg_matches(msg, MAZDA_CRZ_BTNS, 0U, 8U)) {
      // MODE_Y is physical ON intent for lateral only. The raw SET/RES qualifier below is untouched.
      if (acc_main_on && GET_BIT(msg, MAZDA_MODE_Y_BIT)) {
        observed.data[0] |= 0x10U;
      }
      mazda_aol_rx(&observed);
    } else if (mazda_port_tja_button && msg_matches(msg, MAZDA_CRZ_CTRL, 0U, 8U)) {
      observed.data[2] |= 2U;
      mazda_aol_rx(&observed);
      mazda_aol_engagement(&observed, GET_BIT(msg, 3U), !was_allowed && controls_allowed);
    } else {
      mazda_aol_rx(msg);
      if (!mazda_port_longitudinal && msg_matches(msg, MAZDA_CRZ_CTRL, 0U, 8U)) {
        mazda_aol_engagement(msg, GET_BIT(msg, 3U), !was_allowed && controls_allowed);
      }
    }
  }

}

static bool mazda_port_is_lka_addr(int addr) {
  return (((unsigned int)addr == MAZDA_LKAS) || ((unsigned int)addr == MAZDA_LKAS_HUD));
}

// The camera owns the LKAS addresses whenever openpilot is not steering. Lateral is its own
// axis under MADS (controls_allowed_lateral); with MADS off it follows cruise. Cruise alone must
// not claim them, or it silences the camera's own TJA/CTS while the dash shows nothing.
static bool mazda_port_openpilot_controlling(void) {
  return lateral_controls_allowed();
}

static bool mazda_port_tx_hook(const CANPacket_t *msg) {
  // Upstream's envelope, the no-param default.
  const TorqueSteeringLimits MAZDA_STEERING_LIMITS = {
    .max_torque = 800,
    .max_rate_up = 10,
    .max_rate_down = 25,
    .max_rt_delta = 300,
    .driver_torque_multiplier = 1,
    .driver_torque_allowance = 15,
    .type = TorqueDriverLimited,
  };

  // The measured EPS envelope: the EPS's ~1000 counts/s hardware slew (12 counts per 12 ms 0x241
  // frame) at 100 Hz, with max_rate_down equal to the controller retreat rate so driver-limit
  // winddown frames remain valid.
  const TorqueSteeringLimits MAZDA_EPS_HW_STEERING_LIMITS = {
    .max_torque = 1200,
    .max_rate_up = 10,
    .max_rate_down = 10,
    .max_rt_delta = 320,
    .driver_torque_multiplier = 15,
    .driver_torque_allowance = 15,
    .type = TorqueDriverLimited,
  };

  // CRZ_INFO.ACCEL_CMD uses 1/1024 m/s2 raw units after removing the offset: 2.0 and -3.5 m/s2.
  const LongitudinalLimits MAZDA_LONG_LIMITS = {
    .max_accel = 2048,
    .min_accel = -3584,
    .inactive_accel = 0,
  };

  bool tx = true;
  bool main_bus = msg->bus == (unsigned char)MAZDA_MAIN;
  bool long_replacement_bus = main_bus || (msg->bus == (unsigned char)MAZDA_CAM);

  if (main_bus && (msg->addr == MAZDA_LKAS)) {
    int desired_torque = (((msg->data[0] & 0x0FU) << 8) | msg->data[1]) - 2048U;

    const TorqueSteeringLimits *limits = &MAZDA_STEERING_LIMITS;
    if (mazda_port_eps_hw) {
      limits = &MAZDA_EPS_HW_STEERING_LIMITS;
    }
    if (steer_torque_cmd_checks(desired_torque, -1, *limits)) {
      tx = false;
    }
  }

  // Run after steering checks, which reset rate-limit state while disengaged.
  if (main_bus && mazda_port_is_lka_addr(msg->addr) && !mazda_port_openpilot_controlling()) {
    tx = false;
  }

  if (mazda_port_longitudinal && long_replacement_bus && (msg->addr == MAZDA_CRZ_INFO)) {
    // Allow byte-exact stock standby patterns with the raw 8190 command sentinel. Byte 5 bit 0 is the
    // radar's echo of the dash frame's white TJA=2, outside the factory checksum (Sept-Oct captures).
    bool stock_standby = (msg->data[0] == 0x01U) && (msg->data[1] == 0xffU) &&
                         (msg->data[2] == 0xe3U) && (msg->data[3] == 0xffU) &&
                         ((msg->data[4] & 0xfbU) == 0xc0U) &&
                         ((msg->data[5] & 0x7eU) == 0x00U) &&
                         ((msg->data[6] & 0xf0U) == 0x00U) &&
                         (msg->data[7] == ((0xffU - ((msg->data[0] + msg->data[1] + msg->data[2] + msg->data[3] +
                                                     msg->data[4] + (msg->data[5] & 0xfeU) + msg->data[6]) & 0xffU)) & 0xffU));

    // Assemble the 13-bit ACCEL_CMD unsigned for MISRA 10.1, then remove its 4096 offset.
    uint32_t accel_raw = (((uint32_t)msg->data[2] & 0x3U) << 11) | ((uint32_t)msg->data[3] << 3) | ((uint32_t)msg->data[4] >> 5);
    int desired_accel = (int)accel_raw - 4096;
    if (!stock_standby && longitudinal_accel_checks(desired_accel, MAZDA_LONG_LIMITS)) {
      tx = false;
    }

    // ACC_ACTIVE requires prior controls permission established from the physical SET edge.
    // So do the standstill bits sent only with it: STOPPING, STOPPING_2 and RESUME_UNLATCHING (a hold release).
    // longitudinal_controls_allowed() is controls_allowed without AOL; under AOL it is the host's current
    // native long permission, so a withdrawn or expired long request cannot keep cruise active or release a hold.
    bool acc_active = GET_BIT(msg, 33U);
    bool standstill_bits = GET_BIT(msg, 42U) || GET_BIT(msg, 52U) || GET_BIT(msg, 54U);
    if (!longitudinal_controls_allowed() && (acc_active || standstill_bits)) {
      tx = false;
    }
  }

  if (mazda_port_longitudinal && long_replacement_bus && (msg->addr == MAZDA_CRZ_CTRL)) {
    // CRZ_ACTIVE and ACC_ACTIVE_2, which is sent only with it
    bool cruise_active = GET_BIT(msg, 3U) || GET_BIT(msg, 52U);
    if (!longitudinal_controls_allowed() && cruise_active) {
      tx = false;
    }
  }

  bool radar_frame = (msg->addr == MAZDA_RADAR_STATIC) || ((msg->addr >= MAZDA_RADAR_TRACK_1) && (msg->addr <= MAZDA_RADAR_TRACK_6));
  if (mazda_port_longitudinal && long_replacement_bus && radar_frame) {
    if (!mazda_port_radar_capture_msg_valid(msg)) {
      tx = false;
    }
  }

  if (mazda_port_longitudinal && main_bus && (msg->addr == MAZDA_RADAR_UDS)) {
    // Allow tester-present and default/programming session control only.
    bool tester_present = (msg->data[0] == 0x02U) && (msg->data[1] == 0x3eU) && (msg->data[2] == 0x80U);
    bool session_control = (msg->data[0] == 0x02U) && (msg->data[1] == 0x10U) &&
                           ((msg->data[2] == 0x01U) || (msg->data[2] == 0x02U));
    if (!tester_present && !session_control) {
      tx = false;
    }
  }

  if (main_bus && (msg->addr == MAZDA_CRZ_BTNS)) {
    // Permit resume only while controlling and cancel only while not controlling.
    bool cancel_cmd = (msg->data[0] == 0x1U);
    const bool mrcc_off_candidate = !GET_BIT(msg, 16U) && GET_BIT(msg, 15U);
    const bool mrcc_off_cmd = mazda_port_tja_button && mazda_port_acc_armed && mazda_port_mrcc_off_msg_valid(msg);
    if (!controls_allowed && !cancel_cmd && !mrcc_off_cmd) {
      tx = false;
    }
    // An MRCC-off-shaped frame is either the exact narrow exception or invalid; it cannot
    // borrow the normal cancel authorization as a composite command.
    if (mrcc_off_candidate && !mrcc_off_cmd) {
      tx = false;
    }
    // The TJA button is never pressed on the car's side: it would toggle MADS through the
    // rx hook and arm MRCC in the body.
    if (GET_BIT(msg, MAZDA_TJA_BUTTON_BIT)) {
      tx = false;
    }
  }

  if (main_bus && (msg->addr == MAZDA_CAN_FILTER_CTRL)) {
    // FILTER_CTRL only: MODE/FILTER bits, reserved bytes zero. The board's bootloader, softloader,
    // reset (0x1e) and feature-save (0x1f) subcommands never pass. Listed only in the filter TX tables.
    bool filter_ctrl = (msg->data[0] == 0xceU) && (msg->data[1] == 0xfaU) && (msg->data[2] == 0xadU) &&
                       (msg->data[3] == 0xdeU) && (msg->data[4] == 0x2eU) && ((msg->data[5] & 0xfcU) == 0U) &&
                       (msg->data[6] == 0U) && (msg->data[7] == 0U);
    if (!filter_ctrl) {
      tx = false;
    }
  }

  if ((msg->bus == (unsigned char)MAZDA_CAM) && (msg->addr == MAZDA_CRZ_BTNS)) {
    // Camera-only TJA press (mads update_camera_tja: press the camera's own TJA off): the exact
    // press, only while openpilot steers, paced and capped per burst. It reaches the camera alone,
    // never the body or the panda's own TJA latch (rx reads bus 0).
    uint32_t ts = microsecond_timer_get();
    uint32_t since = safety_get_ts_elapsed(ts, mazda_port_cam_press_ts);
    if ((mazda_port_cam_press_count > 0U) && (since >= MAZDA_CAM_PRESS_QUIET_US)) {
      mazda_port_cam_press_count = 0U;
    }
    bool paced = (mazda_port_cam_press_count == 0U) || (since >= MAZDA_CAM_PRESS_MIN_US);
    if (mazda_port_tja_button && mazda_port_cam_tja_press_msg_valid(msg) && mazda_port_openpilot_controlling() &&
        paced && (mazda_port_cam_press_count < MAZDA_CAM_PRESS_MAX)) {
      mazda_port_cam_press_count += 1U;
      mazda_port_cam_press_ts = ts;
    } else {
      tx = false;
    }
  }

  mazda_aol_tx_commit(msg, tx && !relay_malfunction);
  return tx;
}

static bool mazda_port_fwd_hook(int bus_num, int addr) {
  bool block_msg = false;

  if (bus_num == MAZDA_CAM) {
    if (mazda_port_is_lka_addr(addr)) {
      block_msg = mazda_port_openpilot_controlling();
    }
  }

  return block_msg;
}

static safety_config mazda_port_init(uint16_t param) {
  mazda_aol_reset();
  mazda_aol_alpha_long = false;
  if (((param & ~27U) != 0U) || ((param & MAZDA_PARAM_EPS_HW) == 0U)) {
    return (safety_config){0};
  }
  mazda_aol_enabled = (unsigned int)alternative_experience == 32U;
  mazda_aol_alpha_long = mazda_aol_enabled && GET_FLAG(param, MAZDA_PARAM_LONGITUDINAL);
  if (mazda_aol_enabled) {
    static const AolSafetyPolicy policy = {
      .reset = mazda_aol_reset,
      .host_request = mazda_aol_request,
      .request_mask = mazda_aol_request_mask,
      .permission_mask = mazda_aol_permission,
      .rx_invalid = mazda_aol_clear,
    };
    aol_policy = &policy;
    mazda_aol_startup_pending = (param & MAZDA_PARAM_TJA_BUTTON) == 0U;
  }
  mazda_port_engage_btn_frames = 0U;
  mazda_port_main_off_samples = 0U;
  mazda_port_acc_armed = false;
  mazda_port_cam_press_ts = 0U;
  mazda_port_cam_press_count = 0U;

  // CRZ_BTNS on the camera bus is the camera-only TJA press; no relay check, so the real wheel
  // stream, the driver's TJA press included, keeps forwarding to the camera.
  static const CanMsg MAZDA_TX_MSGS[] = {
    {MAZDA_LKAS, 0, 8, .check_relay = true, .disable_static_blocking = true},
    {MAZDA_CRZ_BTNS, 0, 8, .check_relay = false},
    {MAZDA_LKAS_HUD, 0, 8, .check_relay = true, .disable_static_blocking = true},
    {MAZDA_CRZ_BTNS, MAZDA_CAM, 8, .check_relay = false},
  };

  static const CanMsg MAZDA_NATIVE_TX_MSGS[] = {
    {MAZDA_LKAS, 0, 8, .check_relay = false, .disable_static_blocking = true},
    {MAZDA_CRZ_BTNS, 0, 8, .check_relay = false},
    {MAZDA_LKAS_HUD, 0, 8, .check_relay = false, .disable_static_blocking = true},
    {MAZDA_CRZ_BTNS, MAZDA_CAM, 8, .check_relay = false},
  };

// Replaced-radar addresses omit relay checks because the radar remains live during boot and
// hand-back. carstate enforces single ownership instead.
  static const CanMsg MAZDA_LONG_TX_MSGS[] = {
    {MAZDA_LKAS, 0, 8, .check_relay = true, .disable_static_blocking = true},
    {MAZDA_CRZ_BTNS, 0, 8, .check_relay = false},
    {MAZDA_LKAS_HUD, 0, 8, .check_relay = true, .disable_static_blocking = true},
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
    {MAZDA_CRZ_BTNS, MAZDA_CAM, 8, .check_relay = false},
  };

  static const CanMsg MAZDA_NATIVE_LONG_TX_MSGS[] = {
    {MAZDA_LKAS, 0, 8, .check_relay = false, .disable_static_blocking = true},
    {MAZDA_CRZ_BTNS, 0, 8, .check_relay = false},
    {MAZDA_LKAS_HUD, 0, 8, .check_relay = false, .disable_static_blocking = true},
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
    {MAZDA_CRZ_BTNS, MAZDA_CAM, 8, .check_relay = false},
  };


  // The Smart CAN Filter keeps the radar online: no UDS takeover and no synthetic radar frames, and
  // its FILTER_CTRL is the one extra message.
  static const CanMsg MAZDA_LONG_FILTER_TX_MSGS[] = {
    {MAZDA_LKAS, 0, 8, .check_relay = true, .disable_static_blocking = true},
    {MAZDA_CRZ_BTNS, 0, 8, .check_relay = false},
    {MAZDA_LKAS_HUD, 0, 8, .check_relay = true, .disable_static_blocking = true},
    {MAZDA_CRZ_INFO, 0, 8, .check_relay = false},
    {MAZDA_CRZ_CTRL, 0, 8, .check_relay = false},
    {MAZDA_CAN_FILTER_CTRL, 0, 8, .check_relay = false},
    {MAZDA_CRZ_BTNS, MAZDA_CAM, 8, .check_relay = false},
    {MAZDA_CRZ_INFO, MAZDA_CAM, 8, .check_relay = false},
    {MAZDA_CRZ_CTRL, MAZDA_CAM, 8, .check_relay = false},
  };

  static const CanMsg MAZDA_NATIVE_LONG_FILTER_TX_MSGS[] = {
    {MAZDA_LKAS, 0, 8, .check_relay = false, .disable_static_blocking = true},
    {MAZDA_CRZ_BTNS, 0, 8, .check_relay = false},
    {MAZDA_LKAS_HUD, 0, 8, .check_relay = false, .disable_static_blocking = true},
    {MAZDA_CRZ_INFO, 0, 8, .check_relay = false},
    {MAZDA_CRZ_CTRL, 0, 8, .check_relay = false},
    {MAZDA_CAN_FILTER_CTRL, 0, 8, .check_relay = false},
    {MAZDA_CRZ_BTNS, MAZDA_CAM, 8, .check_relay = false},
    {MAZDA_CRZ_INFO, MAZDA_CAM, 8, .check_relay = false},
    {MAZDA_CRZ_CTRL, MAZDA_CAM, 8, .check_relay = false},
  };

  mazda_port_longitudinal = GET_FLAG(param, MAZDA_PARAM_LONGITUDINAL);
  mazda_port_eps_hw = GET_FLAG(param, MAZDA_PARAM_EPS_HW);
  mazda_port_tja_button = GET_FLAG(param, MAZDA_PARAM_TJA_BUTTON);
  acc_main_on = false;

  safety_config ret = BUILD_SAFETY_CFG(mazda_rx_checks, MAZDA_TX_MSGS);
  if (mazda_port_longitudinal) {
    ret.rx_checks = &mazda_rx_checks[1];
    ret.rx_checks_len -= 1;
    if (GET_FLAG(param, MAZDA_PARAM_CAN_FILTER)) {
      // Alpha long through the Smart CAN Filter instead of the radar's UDS takeover.
      ret.tx_msgs = mazda_aol_enabled ? MAZDA_NATIVE_LONG_FILTER_TX_MSGS : MAZDA_LONG_FILTER_TX_MSGS;
      ret.tx_msgs_len = sizeof(MAZDA_LONG_FILTER_TX_MSGS) / sizeof(CanMsg);
    } else {
      ret.tx_msgs = mazda_aol_enabled ? MAZDA_NATIVE_LONG_TX_MSGS : MAZDA_LONG_TX_MSGS;
      ret.tx_msgs_len = sizeof(MAZDA_LONG_TX_MSGS) / sizeof(CanMsg);
    }
  } else if (mazda_aol_enabled) {
    ret.tx_msgs = MAZDA_NATIVE_TX_MSGS;
  }
  return ret;
}
