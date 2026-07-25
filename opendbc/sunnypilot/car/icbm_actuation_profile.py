"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Per-car actuation profile for Intelligent Cruise Button Management (ICBM).

On button-actuated (non-pcmCruiseSpeed) cars the stock ECU integrates the cruise buttons,
and every ECU does it differently: how fast discrete presses register, whether a held
button snaps the set speed along a coarse grid, and whether the ACC delays deceleration
while the set speed is still changing. The servo that plans button moves reads these
characteristics from here.

Cars without a measured profile get DEFAULT_PROFILE: discrete taps only, no grid, no hold;
the long-standing ICBM behavior. A brand only changes behavior by adding a measured entry.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ICBMActuationProfile:
  # Discrete taps. tap_rate_hz is the fastest cadence the body ECU reliably registers;
  # pushing beyond it makes the ECU drop presses (measured on Mazda: ~9 Hz sends register
  # at ~0.47 steps/press vs ~0.93 at 5 Hz; faster is slower). The per-tap increment is
  # not modeled: the servo is closed-loop on the dash, so tap size is implicit.
  tap_rate_hz: float = 5.

  # Held-button behavior of the stock ECU. 0 disables hold planning entirely (taps only).
  # A hold first snaps the set speed to the next multiple of longpress_step, then steps by
  # longpress_step per period. Step k lands at ~(first_step_s + (k-1) * step_period_s).
  longpress_step: int = 0  # display units per hold step; also the alignment grid
  longpress_first_step_s: float = 0.
  longpress_step_period_s: float = 0.
  # Grid/steps measured in imperial display units only so far; until a brand confirms the
  # metric grid, metric users plan with taps only.
  longpress_metric_confirmed: bool = False

  # The stock ACC does not commit to decelerating while the set speed is still moving; the
  # servo makes one decisive move and then goes silent instead of tracking continuously.
  decel_needs_stable_setpoint: bool = False

  @property
  def has_longpress(self) -> bool:
    return self.longpress_step > 0

  def supports_longpress(self, is_metric: bool) -> bool:
    return self.has_longpress and (self.longpress_metric_confirmed or not is_metric)


# Mazda CX-5 2022, measured from a 52-episode driver long-press corpus and injected-press
# efficiency analysis (674 rlog segments): taps register reliably at 5 Hz and move 1 mph;
# holds snap to multiples of 5 mph, first step ~0.6 s into the hold, ~0.55 s per further
# step; MRCC will not start decelerating until the set speed stops changing.
ICBM_ACTUATION_PROFILES: dict[str, ICBMActuationProfile] = {
  'mazda': ICBMActuationProfile(
    tap_rate_hz=5.,
    longpress_step=5,
    longpress_first_step_s=0.6,
    longpress_step_period_s=0.55,
    longpress_metric_confirmed=False,
    decel_needs_stable_setpoint=True,
  ),
}

DEFAULT_PROFILE = ICBMActuationProfile()


def get_actuation_profile(brand: str) -> ICBMActuationProfile:
  return ICBM_ACTUATION_PROFILES.get(brand, DEFAULT_PROFILE)
