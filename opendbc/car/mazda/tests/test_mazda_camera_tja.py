"""The camera's own TJA pressed off on the camera bus while openpilot steers (mads.update_camera_tja).

Route 00000097: one camera-only press took the camera TJA 2 -> 0 in 0.28 s and its lane-keep state
(CAM_LANEINFO byte 3) stayed armed. Replay cannot show the camera's answer; these pin the controller.
"""
import pytest

from opendbc.car.mazda.tests.conftest import (CAM_LANEINFO, car_controller, frames, mazda_car_state, step)
from opendbc.car.structs import CarParams
from opendbc.safety.tests.libsafety import libsafety_py
from opendbc.car.mazda.mads import CAMERA_TJA_CONFIRM_FRAMES, CAMERA_TJA_PRESS_FRAMES, CAMERA_TJA_PRESS_MAX
from opendbc.car.mazda.values_ext import MazdaFlagsSP

from opendbc.car.mazda.values import MazdaSafetyFlags as _PortFlags
# the port panda's param word: the measured EPS envelope (required) and the declared TJA button
PORT_TJA = _PortFlags.EPS_HW | _PortFlags.TJA_BUTTON

CRZ_BTNS = 0x9d
IDLE_LANEINFO = bytes.fromhex("4201000000001040")
ARMED_LANEINFO = bytes.fromhex("4201000a00001040")  # byte 3 = 0x0a: lane-keep on
HWY = 25.  # m/s


def controller(tja_button=True):
  cc = car_controller()
  if tja_button:
    cc.CP_SP.flags |= MazdaFlagsSP.TJA_BUTTON
  cs = mazda_car_state(cc.CP, cc.CP_SP)
  return cc, cs


def run(cc, cs, n, *, tja, lat_active=True, live=True, tja_button=0, counter=5, v_ego=0., raw=IDLE_LANEINFO, lane_lines=1):
  """n frames with the camera's native TJA at `tja`; returns the camera-bus CRZ_BTNS sent per frame."""
  cs.cam_laneinfo["TJA"] = tja
  cs.cam_laneinfo["LANE_LINES"] = lane_lines
  out = []
  for _ in range(n):
    _, sends = step(cc, cs, lat_active=lat_active, enabled=lat_active, long_active=False, crz_btns_counter=counter,
                    cam_laneinfo_live=live, cam_laneinfo_raw=raw, tja_button=tja_button, v_ego=v_ego)
    out.append(frames(sends, CRZ_BTNS, bus=2))
  return out


def press_frames(out):
  return [i for i, f in enumerate(out) if f]


def test_undeclared_never_presses():
  cc, cs = controller(tja_button=False)
  assert not press_frames(run(cc, cs, 500, tja=3))


@pytest.mark.parametrize("tja", [0, 2])
def test_native_off_or_standby_never_presses(tja):
  cc, cs = controller()
  assert not press_frames(run(cc, cs, 500, tja=tja))
  assert not cs.camera_tja_stuck


@pytest.mark.parametrize("tja", [3, 4, 5])
def test_confirmed_active_tja_presses_paced_capped_then_alerts(tja):
  cc, cs = controller()
  out = run(cc, cs, CAMERA_TJA_CONFIRM_FRAMES + CAMERA_TJA_PRESS_FRAMES * (CAMERA_TJA_PRESS_MAX + 2), tja=tja)
  presses = press_frames(out)
  first = CAMERA_TJA_CONFIRM_FRAMES - 1
  assert presses == [first + i * CAMERA_TJA_PRESS_FRAMES for i in range(CAMERA_TJA_PRESS_MAX)]
  assert all(len(out[i]) == 1 for i in presses)
  assert cs.camera_tja_stuck


def test_alert_waits_a_full_period_after_the_last_press():
  cc, cs = controller()
  last = CAMERA_TJA_CONFIRM_FRAMES - 1 + (CAMERA_TJA_PRESS_MAX - 1) * CAMERA_TJA_PRESS_FRAMES
  run(cc, cs, last + CAMERA_TJA_PRESS_FRAMES, tja=3)
  assert not cs.camera_tja_stuck
  run(cc, cs, 1, tja=3)
  assert cs.camera_tja_stuck


def test_press_bytes_on_the_camera_bus():
  cc, cs = controller()
  out = run(cc, cs, CAMERA_TJA_CONFIRM_FRAMES, tja=3, counter=2)
  assert out[-1] == [bytes.fromhex("0009ffcc00000000")]


def test_transient_after_a_passed_press_is_left_alone():
  # A passed press shows TJA=2 for under 0.5 s and falls back to 0 on its own (routes 00000097..13a).
  cc, cs = controller()
  out = run(cc, cs, 45, tja=2) + run(cc, cs, 300, tja=0)
  assert not press_frames(out)


def test_stale_camera_sends_nothing_and_restarts_confirmation():
  cc, cs = controller()
  out = run(cc, cs, CAMERA_TJA_CONFIRM_FRAMES - 1, tja=3)
  out += run(cc, cs, 300, tja=3, live=False)
  assert not press_frames(out)
  # fresh again: a full confirmation before the first press
  out = run(cc, cs, CAMERA_TJA_CONFIRM_FRAMES, tja=3)
  assert press_frames(out) == [CAMERA_TJA_CONFIRM_FRAMES - 1]


def test_not_steering_leaves_the_camera_alone_and_clears_the_alert():
  cc, cs = controller()
  assert not press_frames(run(cc, cs, 500, tja=3, lat_active=False))
  run(cc, cs, CAMERA_TJA_CONFIRM_FRAMES + CAMERA_TJA_PRESS_FRAMES * CAMERA_TJA_PRESS_MAX, tja=3)
  assert cs.camera_tja_stuck
  assert not press_frames(run(cc, cs, 300, tja=3, lat_active=False))
  assert not cs.camera_tja_stuck


@pytest.mark.parametrize("tja", [0, 2])
def test_off_or_standby_ends_the_episode(tja):
  cc, cs = controller()
  run(cc, cs, CAMERA_TJA_CONFIRM_FRAMES + CAMERA_TJA_PRESS_FRAMES * CAMERA_TJA_PRESS_MAX, tja=3)
  assert cs.camera_tja_stuck
  run(cc, cs, 1, tja=tja)
  assert not cs.camera_tja_stuck
  # a fresh budget of three, paced by the panda's shared burst (test_back_to_back_episodes_*)
  out = run(cc, cs, 1500, tja=3)
  assert len(press_frames(out)) == CAMERA_TJA_PRESS_MAX


def test_one_press_ends_it():
  cc, cs = controller()
  out = run(cc, cs, CAMERA_TJA_CONFIRM_FRAMES, tja=3) + run(cc, cs, 500, tja=0)
  assert len(press_frames(out)) == 1
  assert not cs.camera_tja_stuck


def test_never_inside_the_drivers_press():
  cc, cs = controller()
  out = run(cc, cs, CAMERA_TJA_CONFIRM_FRAMES + 50, tja=3, tja_button=1)
  assert not press_frames(out)
  out = run(cc, cs, 1, tja=3)
  assert press_frames(out) == [0]


def test_cruise_engagement_after_a_permitted_press():
  """MADS steering, the camera's lane-keep armed by an earlier press (native TJA back to 0), then
  cruise engages: nothing is pressed while TJA stays 0; if the camera's TJA rises under cruise
  (the 0000013e path) it is pressed off after the confirmation."""
  cc, cs = controller()
  out = run(cc, cs, 30, tja=2) + run(cc, cs, 500, tja=0)
  cs.cam_laneinfo["TJA"] = 0
  for _ in range(500):
    _, sends = step(cc, cs, lat_active=True, enabled=True, long_active=True, cruise_engaged=True,
                    cam_laneinfo_live=True, cam_laneinfo_raw=IDLE_LANEINFO)
    out.append(frames(sends, CRZ_BTNS, bus=2))
  assert not press_frames(out)
  later = run(cc, cs, CAMERA_TJA_CONFIRM_FRAMES, tja=3)
  assert press_frames(later) == [CAMERA_TJA_CONFIRM_FRAMES - 1]


def test_every_controller_press_passes_the_panda():
  """Together: the controller's cadence against the panda's camera-bus rule, openpilot steering."""
  safety = libsafety_py.libsafety
  # the port's panda: one safety word, TJA_BUTTON in it (no sunnypilot second word or MADS)
  # the libsafety instance is shared: no AOL (alternative experience) left by an earlier test
  safety.set_alternative_experience(0)
  safety.set_safety_hooks(CarParams.SafetyModel.mazda, PORT_TJA)
  safety.init_tests()
  try:
    cc, cs = controller()
    for episode in range(3):
      out = run(cc, cs, CAMERA_TJA_CONFIRM_FRAMES + CAMERA_TJA_PRESS_FRAMES * CAMERA_TJA_PRESS_MAX, tja=3)
      base = episode * 10_000_000
      for i, sent in enumerate(out):
        for dat in sent:
          safety.set_timer(base + i * 10_000)
          safety.set_controls_allowed(True)
          assert safety.safety_tx_hook(libsafety_py.make_CANPacket(CRZ_BTNS, 2, dat)), f"{episode=} frame {i}"
      run(cc, cs, 1, tja=0)
  finally:
    safety.set_controls_allowed(False)


def test_laneinfo_frame_untouched_by_the_press_off():
  # the HUD frame (byte 3 included) is not part of this change: same frames with or without a press-off
  def run(native_tja):
    cc, cs = controller()
    sends = []
    cs.cam_laneinfo["TJA"] = native_tja
    for _ in range(CAMERA_TJA_CONFIRM_FRAMES + 60):
      _, s = step(cc, cs, lat_active=True, enabled=True, long_active=False, cam_laneinfo_live=True, cam_laneinfo_raw=IDLE_LANEINFO)
      sends += s
    return sends
  pressed, quiet = run(3), run(0)
  assert frames(pressed, CRZ_BTNS, bus=2) and not frames(quiet, CRZ_BTNS, bus=2)
  assert frames(pressed, CAM_LANEINFO)
  assert frames(pressed, CAM_LANEINFO) == frames(quiet, CAM_LANEINFO)


# Shared panda burst budget across episodes and both paths, on continuous frame timing

class PandaFeed:
  """Every camera-bus press the controller sends, offered to the panda at its real frame time."""

  def __init__(self, param=0):
    self.safety = libsafety_py.libsafety
    self.safety.set_alternative_experience(0)
    self.safety.set_safety_hooks(CarParams.SafetyModel.mazda, PORT_TJA | param)
    self.safety.init_tests()
    self.frame = 0
    self.sent, self.refused = [], []

  def feed(self, out):
    for f in out:
      for dat in f:
        self.safety.set_timer(1_000_000 + self.frame * 10_000)
        self.safety.set_controls_allowed(True)
        ok = self.safety.safety_tx_hook(libsafety_py.make_CANPacket(CRZ_BTNS, 2, dat))
        (self.sent if ok else self.refused).append(self.frame)
      self.frame += 1

  def close(self):
    self.safety.set_controls_allowed(False)


def test_back_to_back_episodes_never_exceed_the_panda_burst():
  # one frame of TJA=0 between two episodes: the second's presses wait for the burst to clear
  cc, cs = controller()
  panda = PandaFeed()
  try:
    panda.feed(run(cc, cs, 400, tja=3))
    panda.feed(run(cc, cs, 1, tja=0))
    out = run(cc, cs, 1400, tja=3)
    panda.feed(out)
    assert panda.refused == []
    assert len(panda.sent) == 2 * CAMERA_TJA_PRESS_MAX
    # the second episode spent its whole budget on presses that got through, then alerted
    assert cc.camera_tja_presses == CAMERA_TJA_PRESS_MAX
    assert cs.camera_tja_stuck
  finally:
    panda.close()


def test_second_episode_waits_instead_of_spending_attempts():
  cc, cs = controller()
  panda = PandaFeed()
  try:
    panda.feed(run(cc, cs, 400, tja=3) + run(cc, cs, 1, tja=0))
    panda.feed(run(cc, cs, 500, tja=3))
    # burst full after the second episode's first press (frame 500): no attempt is spent, no alert yet
    assert panda.refused == []
    assert cc.camera_tja_presses == 1
    assert not cs.camera_tja_stuck
  finally:
    panda.close()


@pytest.mark.parametrize("seed", range(20))
def test_random_sequences_never_refused(seed):
  import random
  rng = random.Random(seed)
  cc, cs = controller()
  panda = PandaFeed()
  try:
    for _ in range(60):
      panda.feed(run(cc, cs, rng.choice([1, 5, 50, 120, 300, 900, 2100]), tja=rng.choice([0, 0, 2, 3, 4]),
                     lat_active=rng.random() > 0.1, v_ego=rng.choice([0., 10., HWY]), live=rng.random() > 0.05,
                     raw=rng.choice([IDLE_LANEINFO, ARMED_LANEINFO])))
    assert panda.refused == [], f"{seed=}"
    assert panda.sent
  finally:
    panda.close()


def test_smart_can_filter_config_takes_every_press():
  # the car's config: alpha long through the Smart CAN Filter, with its own TX table
  from opendbc.car.mazda.values import MazdaSafetyFlags
  cc, cs = controller()
  panda = PandaFeed(MazdaSafetyFlags.LONG | MazdaSafetyFlags.CAN_FILTER)
  try:
    panda.feed(run(cc, cs, 400, tja=3, v_ego=HWY) + run(cc, cs, 1, tja=0, v_ego=HWY) + run(cc, cs, 1400, tja=3, v_ego=HWY))
    assert panda.refused == []
    assert len(panda.sent) == 2 * CAMERA_TJA_PRESS_MAX
  finally:
    panda.close()


def test_no_automatic_rearm():
  # First-drive candidate: the camera's lane-keep follows the driver's own TJA presses only. Nothing
  # is pressed while its TJA reads 0, however long byte 3 stays zero at speed with openpilot steering.
  cc, cs = controller()
  assert not press_frames(run(cc, cs, 9000, tja=0, v_ego=HWY))
  assert not press_frames(run(cc, cs, 3000, tja=0, v_ego=HWY, raw=ARMED_LANEINFO))


@pytest.mark.parametrize("burst,ago,allowed", [(4, 150, False),   # burst full
                                               (1, 95, False),    # inside the shared spacing
                                               (4, 510, True),    # burst over after the quiet time
                                               (1, 100, True)])
def test_press_off_respects_the_shared_budget(burst, ago, allowed):
  # The episode's own pacing never meets these today; set the shared budget directly so a later change
  # to the episode timing cannot outrun the panda.
  cc, cs = controller()
  run(cc, cs, CAMERA_TJA_CONFIRM_FRAMES - 1, tja=3)
  cc.camera_press_burst = burst
  cc.camera_press_frame = cc.frame - ago
  assert bool(press_frames(run(cc, cs, 1, tja=3))) == allowed
