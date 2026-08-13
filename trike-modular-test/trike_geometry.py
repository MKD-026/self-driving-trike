"""Measured geometry for THIS trike. Single source of truth.

Every number here was measured, not assumed - see HANDOFF.md section 3 in
the trike/ project. Do not "tidy" them toward round values.

WHY THIS FILE EXISTS
  bev.py's PurePursuitConfig defaults to wheelbase_m=1.20 and
  max_wheel_angle_deg=34.0. Neither matches this vehicle. Those two
  numbers appear in bev's normalisation:

      raw = -atan(wheelbase_m * kappa) / radians(max_wheel_angle_deg)

  and again in any code that inverts it to recover curvature. If the two
  sites disagree, the error is applied twice and the trike steers a
  different amount than anything intended. Importing both from here makes
  that impossible.

  With the TRUE values the round trip is an exact identity:
      kappa -> raw -> kappa
  and bev's own +/-1 clamp lands precisely on the mechanical limit,
  because tan(30.5 deg) / 0.965 = 0.610 = KAPPA_MECH. With the 1.20/34.0
  defaults it clamped at 0.562, quietly giving up 8% of the available
  steering while also being wrong.
"""

import math

# --- vehicle ------------------------------------------------------
WHEELBASE_M = 0.965            # measured
MAX_WHEEL_ANGLE_DEG = 30.5     # measured full lock
KAPPA_MECH = 0.610             # 1/m at full lock; R_min 1.64 m

# --- actuator / firmware limits (trike_controller.ino) ------------
KDOT_MAX = 0.12                # 1/m/s. Below the SLOWER direction's
                               # small-signal speed: 73,down = 398 ct/s
                               # = 0.135 1/m/s. Raising it does not make
                               # the trike faster, it makes the actuator
                               # rate-saturate and voids the
                               # equivalent-delay stability argument.
LOOKAHEAD_M = 6.1              # from Ld >= 3.54*v*tau_eq, tau_eq 0.43 s
                               # at v_max 4 m/s. Provisional until the
                               # HIL sweep.
AY_LIMIT = 1.47                # m/s^2, 0.15 g rollover clamp numerator

# --- serial -------------------------------------------------------
ESP_BAUD = 921600              # trike_controller.ino Serial.begin()
VN_BAUD = 115200               # VN-200 register 05


def kappa_to_normalized(kappa: float) -> float:
    """Curvature -> bev-style normalised steer. + = RIGHT, matching bev."""
    a = math.atan(WHEELBASE_M * float(kappa))
    return max(-1.0, min(1.0, -a / math.radians(MAX_WHEEL_ANGLE_DEG)))


def normalized_to_kappa(steering: float) -> float:
    """bev-style normalised steer -> curvature. + = LEFT, matching firmware.

    Exact inverse of kappa_to_normalized. No extra sign flip: the minus
    sign already lives inside tan(-raw*A), so negating again double-inverts
    and steers into whatever was being avoided.
    """
    s = max(-1.0, min(1.0, float(steering)))
    return math.tan(-s * math.radians(MAX_WHEEL_ANGLE_DEG)) / WHEELBASE_M


def clamp_kappa(kappa: float) -> float:
    return max(-KAPPA_MECH, min(KAPPA_MECH, float(kappa)))


def rollover_kappa_limit(v_mps: float) -> float:
    """kappa_max = 1.47/v^2, the firmware's 0.15 g clamp. Mirrored here so
    the host can see when it is about to be clamped rather than being
    surprised by flags & 8."""
    return AY_LIMIT / max(1.0, float(v_mps) ** 2)
