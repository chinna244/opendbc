#!/usr/bin/env python3
"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import math

from opendbc.can import CANParser
from opendbc.car import Bus, DT_CTRL, structs
from opendbc.car.interfaces import RadarInterfaceBase
from opendbc.car.mazda.values import DBC
from opendbc.sunnypilot.car.mazda.values import MazdaFlagsSP

# Six tracks arrive on 0x361-0x366 at 10 Hz. Exclude 0x365 and 0x366 until their relative
# velocity encoding is known; radard requires a valid vRel.
RADAR_TRACK_ADDRS = list(range(0x361, 0x367))
RADAR_USABLE_ADDRS = set(range(0x361, 0x365))  # only tracks with reliable RELV
RADAR_TRIGGER_MSG = RADAR_TRACK_ADDRS[-1]  # 0x366, last in the burst
SENTINEL_DIST = 4095 * 0.0625   # 255.9375 m, raw 4095
SENTINEL_ANG = 2046 * 0.015625  # 31.96875 deg, raw 2046
SENTINEL_RELV = -16 * 0.0625    # -1.0 m/s, raw -16

# Behind the Smart CAN Filter a lost radar falls back to the camera instead of a CAN error: past
# this silence (bursts come every 0.1 s, radard's radarTracks alive window is 0.5 s) card keeps
# publishing empty, temporarily-unavailable RadarData at the radar's own 10 Hz.
RADAR_LOST_FRAMES = int(0.25 / DT_CTRL)
RADAR_LOST_STEP = int(0.1 / DT_CTRL)


# Track IDs preserve the existing DBC message names for 0x361-0x366.
def _create_radar_can_parser(car_fingerprint):
  messages = [(addr, 10) for addr in RADAR_TRACK_ADDRS]
  return CANParser(DBC[car_fingerprint][Bus.radar], messages, 0)


class RadarInterface(RadarInterfaceBase):
  def __init__(self, CP, CP_SP):
    super().__init__(CP, CP_SP)
    self.track_id = 0
    self.updated_messages = set()
    self.trigger_msg = RADAR_TRIGGER_MSG
    self.rcp = None if CP.radarUnavailable else _create_radar_can_parser(CP.carFingerprint)
    self.camera_fallback = bool(CP_SP.flags & MazdaFlagsSP.CAN_FILTER)
    self.silent_frames = 0

  def update(self, can_strings):
    if self.rcp is None:
      return super().update(None)

    vls = self.rcp.update(can_strings)
    self.updated_messages.update(vls)

    if self.trigger_msg not in self.updated_messages:
      if not self.camera_fallback:
        return None
      self.silent_frames += 1
      if self.silent_frames < RADAR_LOST_FRAMES or (self.silent_frames - RADAR_LOST_FRAMES) % RADAR_LOST_STEP:
        return None
      return self._lost()

    self.silent_frames = 0
    rr = self._update(self.updated_messages)
    self.updated_messages.clear()
    return rr

  def _lost(self):
    """No tracks this burst: radard falls back to the camera's leads and selfdrived warns. A burst
    left incomplete by the dropout is discarded, so none of its frames reach the next one."""
    self.pts.clear()
    self.updated_messages.clear()
    ret = structs.RadarData()
    ret.errors.radarUnavailableTemporary = True
    return ret

  def _update(self, updated_messages):
    if not self.rcp.can_valid and self.camera_fallback:
      return self._lost()

    ret = structs.RadarData()
    if not self.rcp.can_valid:
      ret.errors.canError = True

    for addr in RADAR_TRACK_ADDRS:
      # Behind the filter a slot counts only from a frame of this burst: the parser keeps every
      # address's last values, so a frame missing from a burst would republish a stale track.
      if self.camera_fallback and addr not in updated_messages:
        self.pts.pop(addr, None)
        continue

      msg = self.rcp.vl[addr]

      dist = msg['DIST_OBJ']
      ang = msg['ANG_OBJ']
      relv = msg['RELV_OBJ']

      # Each sentinel is individually valid, so require the complete empty-slot signature.
      slot_empty = dist == SENTINEL_DIST and ang == SENTINEL_ANG and relv == SENTINEL_RELV
      if slot_empty or addr not in RADAR_USABLE_ADDRS:
        if addr in self.pts:
          del self.pts[addr]
        continue

      if addr not in self.pts:
        self.pts[addr] = structs.RadarData.RadarPoint()
        self.pts[addr].trackId = self.track_id
        self.track_id += 1

      azimuth = math.radians(ang)
      self.pts[addr].dRel = math.cos(azimuth) * dist
      self.pts[addr].yRel = -math.sin(azimuth) * dist
      self.pts[addr].vRel = relv

    ret.points = list(self.pts.values())
    return ret
