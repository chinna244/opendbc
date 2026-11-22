"""
The camera's lane-keep state (CAM_LANEINFO byte 3, low nibble: TJA_TRANSITION 0x0C and the
unnamed 0x03) reaches the dash on openpilot's own 0x440 frame.

Routes 0000006e/0000006f (2026-11-17): the TJA press reached the camera and its byte 3 armed
(00 <-> 0a/0c), yet the dash drew no lane lines. With MADS the panda blocks the camera's
0x440 whenever openpilot steers, and our frame was the camera's with byte 3 zeroed, the one
content difference left (TJA, hands-on and LDW are 0 in both). With the C4 unplugged the dash
gets the camera's frame and draws the lines. EXPERIMENT: Sept drives also sent byte 3 zeroed
(unexplained), so byte 3 is not a confirmed cause.
"""

from opendbc.car import Bus
from opendbc.car.mazda.carstate import CarState
from opendbc.car.mazda.tests.conftest import CAM_LANEINFO, car_controller, mazda_car_state, step
from opendbc.car.mazda.values_ext import MazdaFlagsSP

BASE = bytes.fromhex("4201000000001040")


def hud(cc, cs, cycles=120, **kwargs):
  # the parsed camera dict and the raw latch come from the same camera frame on the car
  raw = kwargs.get("cam_laneinfo_raw")
  if raw is not None:
    parser = CarState.get_can_parsers(cc.CP, cc.CP_SP)[Bus.cam]
    parser.update([(0, [(CAM_LANEINFO, raw, 2)])])
    cs.cam_laneinfo = parser.vl["CAM_LANEINFO"]
  out = []
  for _ in range(cycles):
    _, sends = step(cc, cs, **kwargs)
    out.extend(d for a, d, b in sends if a == CAM_LANEINFO and b == 0)
  return out


def setup(tja=False):
  cc = car_controller(alpha_long=True)
  if tja:
    cc.CP_SP.flags |= MazdaFlagsSP.TJA_BUTTON
  return cc, mazda_car_state(cc.CP, cc.CP_SP)


def with_b3(base: bytes, b3: int) -> bytes:
  return base[:3] + bytes([b3]) + base[4:]


def test_armed_camera_byte3_reaches_the_dash_while_openpilot_steers():
  for b3 in (0x0A, 0x0C, 0x04):
    cc, cs = setup()
    frames = hud(cc, cs, lat_active=True, enabled=True, cam_laneinfo_raw=with_b3(BASE, b3), cam_laneinfo_live=True)
    assert frames and all(f[3] == b3 for f in frames), (hex(b3), [f.hex() for f in frames])


def test_only_byte3_changes():
  cc, cs = setup()
  ref = hud(cc, cs, lat_active=True, enabled=True, cam_laneinfo_raw=BASE, cam_laneinfo_live=True)
  cc, cs = setup()
  armed = hud(cc, cs, lat_active=True, enabled=True, cam_laneinfo_raw=with_b3(BASE, 0x0C), cam_laneinfo_live=True)
  assert ref and len(ref) == len(armed)
  for a, b in zip(ref, armed, strict=True):
    assert a[:3] == b[:3] and a[4:] == b[4:] and a[3] == 0


def test_high_nibble_never_copied():
  cc, cs = setup()
  frames = hud(cc, cs, lat_active=True, enabled=True, cam_laneinfo_raw=with_b3(BASE, 0xFA), cam_laneinfo_live=True)
  assert frames and all(f[3] == 0x0A for f in frames)


def test_stale_or_missing_camera_frame_sends_byte3_zero():
  cc, cs = setup()
  frames = hud(cc, cs, lat_active=True, enabled=True, cam_laneinfo_raw=with_b3(BASE, 0x0C), cam_laneinfo_live=False)
  assert frames and all(f == BASE for f in frames)
  cc, cs = setup()
  frames = hud(cc, cs, lat_active=True, enabled=True, cam_laneinfo_raw=None, cam_laneinfo_live=True)
  assert frames and all(f[3] == 0 for f in frames)


def test_white_wheel_keeps_its_tja2_and_carries_byte3():
  cc, cs = setup(tja=True)
  frames = hud(cc, cs, mads_active=True, cam_laneinfo_raw=with_b3(BASE, 0x0A), cam_laneinfo_live=True,
               available=False, mrcc_armed_raw=False)
  assert bytes.fromhex("4201000a20001040") in frames, [f.hex() for f in frames]
