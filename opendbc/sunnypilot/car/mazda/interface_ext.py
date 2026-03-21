"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import numpy as np

from opendbc.car.interfaces import get_speed_dependent_torque_params


class CarInterfaceExt:
  """Speed-dependent torque callbacks for cars configured in speed_dependent.toml.

  For configured cars, overrides the linear torque model with a piecewise
  LAF lookup based on current speed (v_ego). For unconfigured cars,
  delegates to the base CI's linear callbacks.
  """

  def __init__(self, CP, CI_Base):
    self.CP = CP
    self.CI_Base = CI_Base
    self.v_ego = 0.0

    speed_dep_params = get_speed_dependent_torque_params()
    cfg = speed_dep_params.get(CP.carFingerprint)

    if cfg is not None:
      self.speed_dep = True
      self.speed_dep_speed_bp = list(cfg['speed_bp'])
      self.speed_dep_laf_v = list(cfg['laf_bp'])
      self.speed_dep_friction_v = list(cfg.get('friction_bp', [0.1] * len(cfg['speed_bp'])))
      # Store originals for +/-30% sanity bounds on live-learned updates
      self._original_laf_v = list(cfg['laf_bp'])
    else:
      self.speed_dep = False

  def torque_from_lateral_accel_speed_dep_closure(self, lateral_acceleration, torque_params):
    """Upstream closure: torque = lat_accel / LAF(v_ego)."""
    laf = float(np.interp(self.v_ego, self.speed_dep_speed_bp, self.speed_dep_laf_v))
    return lateral_acceleration / laf

  def lateral_accel_from_torque_speed_dep_closure(self, torque, torque_params):
    """Upstream closure (inverse): lat_accel = torque * LAF(v_ego)."""
    laf = float(np.interp(self.v_ego, self.speed_dep_speed_bp, self.speed_dep_laf_v))
    return torque * laf

  def _torque_from_lateral_accel_speed_dep_torque_space(self, latcontrol_inputs, torque_params, gravity_adjusted):
    """SP torque-space callback: uses latcontrol_inputs.vego for speed."""
    laf = float(np.interp(latcontrol_inputs.vego, self.speed_dep_speed_bp, self.speed_dep_laf_v))
    return latcontrol_inputs.lateral_acceleration / laf

  def torque_from_lateral_accel_in_torque_space(self):
    """Return the appropriate torque-space callback based on config."""
    if self.speed_dep:
      return self._torque_from_lateral_accel_speed_dep_torque_space
    return self.CI_Base.torque_from_lateral_accel_linear_in_torque_space

  def update_speed_dep_laf(self, speed_bp, laf_bp, friction_bp, valid_bp):
    """Apply live-learned LAF values with +/-30% sanity bounds per bin.
    Only bins marked valid (cal% >= threshold in torqued) are updated."""
    for i in range(len(self.speed_dep_laf_v)):
      if i < len(valid_bp) and valid_bp[i]:
        lo = self._original_laf_v[i] * 0.7
        hi = self._original_laf_v[i] * 1.3
        if lo <= laf_bp[i] <= hi:
          self.speed_dep_laf_v[i] = laf_bp[i]
