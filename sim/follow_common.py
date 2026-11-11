"""follow_common.py -- S7 follow-a-target PURE core (docs/S7_FOLLOW_DESIGN.md).

PURE module: numpy + stdlib only. Must run under bare /usr/bin/python3 on sable
(no gymnasium, no projectairsim, no SB3) so the offline checkers
(check_follow_reward.py, test_ibvs_follow.py) work without the sim rig. The
sim-bound follow_env.py imports the same primitives -- one source of truth for
projection / target kinematics / detector noise / depth grid / reward.

Spec sections implemented here:
  D2  scripted virtual targets (person / bicycle / car), ground level,
      horizontal range U[6, 12] m inside the camera FOV at episode start.
  D4  pinhole camera rigid to the body: 720x480, fx=fy=640, cx0=360, cy0=240,
      optical axis pitched 20 deg BELOW horizon (Mini gimbal-neutral); the box
      is the projection of the vertical segment base -> base+[0,0,-H] (NED),
      h_px = |v_base - v_top| (clip), cy = midpixel of the two projections,
      in_fov = both points in front of the camera plane and cx/cy inside the
      frame. All params env-overridable (FPV_CAM_*), explicit `params` dict
      always wins.
  D5  depth grid 3 rows x 5 cols, row-major: per pixel cell the MIN distance
      (1..20 m) to any scene-object AABB whose projected cone covers the cell,
      normalized (d-1)/19; free cone = 1.0.
  D6  detector noise model: hit ~ Bernoulli(1-p_miss) (p_miss=0.15), jitter
      cx,cy ~ N(0, 0.5% of frame dim), h *= (1+N(0,0.05)),
      conf = clip(N(0.85,0.08), 0, 1); on miss the box is frozen and age++;
      latency: the box shown at step t is the GT box of step t-L, L ~ U{0,1}
      or set from FPV_LAT_MS hi/lo by the caller.
  D7  reward v5 -> v6 (survival-shaped, NO positive terminal, terminals
      bad-only). v6 (D11): the range term is GOAL-CONDITIONED on a
      per-episode desired range and a close-guard subtracts CLOSE_PENALTY
      while the true range is inside CLOSE_GUARD:
      r_step = 0.5*in_view + 0.4*center + 0.2*max(0,1-|log(dist/desired)|)
               - 0.05*||a|| - 0.03*||a - a_prev|| - 0.02*||v||
               - CLOSE_PENALTY if dist < CLOSE_GUARD (per-step, NOT terminal)
      terminals: landed -10, flew-out -10, collision -20 (flew-out radius
      scales with desired per D11 in the env).
      Dominance is PROVEN offline by check_follow_reward.py (card hard rule).
  D11 goal-conditioned desired range U[FPV_RANGE_MIN, FPV_RANGE_MAX]
      (default 1.5..15 m), sampled per episode by the env from its rng;
      FPV_DIST_STAR set => FIXED desired for every episode (back-compat eval).
  D12 apparent-size range estimate identical to the phone app's `Range`
      object (VISION_ONBOARD_RESEARCH.md section 7): range_estimate() pure
      formula + RangeEstimator (EMA + +-noise + frame-edge-dropout aware).

Coordinates everywhere are AirSim NED (x north, y east, z DOWN).
"""
import math
import os
from collections import deque

import numpy as np

# ---------------------------------------------------------------------------
# camera model (D4)
# ---------------------------------------------------------------------------

def _env_f(name, default):
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def default_cam():
    """Camera params (D4), env-overridable; read at CALL time so FPV_CAM_*
    set by the trainer/launcher is honored in spawned sub-env processes."""
    fx = _env_f("FPV_CAM_FX", 640.0)
    return {
        "W": _env_f("FPV_CAM_W", 720.0),
        "H": _env_f("FPV_CAM_H", 480.0),
        "fx": fx,
        "fy": _env_f("FPV_CAM_FY", fx),
        "cx0": _env_f("FPV_CAM_CX0", 360.0),
        "cy0": _env_f("FPV_CAM_CY0", 240.0),
        # pitch of the optical axis vs horizon; NEGATIVE = below horizon (D4)
        "pitch_deg": _env_f("FPV_CAM_PITCH_DEG", -20.0),
    }


def cam_axes(drone_yaw, p):
    """Camera frame axes in NED world coords: (forward, right, down).

    NED yaw psi rotates forward from north toward east. Right-handedness:
    x_cam(right) x y_cam(down) = z_cam(forward). With pitch=e (deg, negative
    = below horizon) the forward axis dips: f_z = -sin(e) > 0 when e < 0.
    """
    e = math.radians(p["pitch_deg"])
    f = np.array([math.cos(drone_yaw) * math.cos(e),
                  math.sin(drone_yaw) * math.cos(e),
                  -math.sin(e)])
    r = np.array([-math.sin(drone_yaw), math.cos(drone_yaw), 0.0])
    d = np.cross(f, r)
    return f, r, d


def project_target(drone_pos, drone_yaw, target_pos, target_h, params=None):
    """GT detector box per D4: project the vertical segment
    base=target_pos -> top=target_pos+[0,0,-target_h] (NED up is -z) through
    the pinhole. Returns dict(cx, cy, h_px, dist, in_fov, edge_cut).

    cx/cy are the segment-midpoint pixel of the two projected endpoints,
    clipped to the frame (D4 "cy ... clip to frame"); h_px is the projected
    segment length clipped to [0, H]; in_fov requires BOTH endpoints in front
    of the camera plane and the (unclipped) center inside the frame. dist is
    the 3D distance drone -> target base (D7 range uses this). When the target
    is not in front of the camera plane: in_fov=False, h_px=0, cx/cy at the
    principal point (harmless -- the detector treats in_fov=False as a miss).
    edge_cut (D12): True when either RAW (unclipped) endpoint pixel lies
    outside [0,W)x[0,H) -- the visible segment is cut by the frame edge, so
    h_px under-reports and the phone app would drop the box from the range
    estimator. in_fov logic is UNCHANGED by this flag.
    """
    p = params or default_cam()
    f, r, d = cam_axes(drone_yaw, p)
    drone = np.asarray(drone_pos, dtype=np.float64)
    base = np.asarray(target_pos, dtype=np.float64)
    top = base + np.array([0.0, 0.0, -float(target_h)])
    dist = float(np.linalg.norm(base - drone))
    rb, rt = base - drone, top - drone
    zb, zt = float(rb @ f), float(rt @ f)
    if zb <= 1e-6 or zt <= 1e-6:            # endpoint behind camera plane
        return {"cx": p["cx0"], "cy": p["cy0"], "h_px": 0.0,
                "dist": dist, "in_fov": False, "edge_cut": True}
    ub = p["cx0"] + p["fx"] * float(rb @ r) / zb
    vb = p["cy0"] + p["fy"] * float(rb @ d) / zb
    ut = p["cx0"] + p["fx"] * float(rt @ r) / zt
    vt = p["cy0"] + p["fy"] * float(rt @ d) / zt
    cx_raw, cy_raw = 0.5 * (ub + ut), 0.5 * (vb + vt)
    h_px = float(np.clip(abs(vb - vt), 0.0, p["H"]))
    in_fov = bool(0.0 <= cx_raw < p["W"] and 0.0 <= cy_raw < p["H"])
    edge_cut = bool(not (0.0 <= ub < p["W"] and 0.0 <= vb < p["H"]
                         and 0.0 <= ut < p["W"] and 0.0 <= vt < p["H"]))
    return {"cx": float(np.clip(cx_raw, 0.0, p["W"] - 1.0)),
            "cy": float(np.clip(cy_raw, 0.0, p["H"] - 1.0)),
            "h_px": h_px, "dist": dist, "in_fov": in_fov,
            "edge_cut": edge_cut}


def h_star_for(target_h, dist_star=6.0, params=None):
    """h_star (D5): projected box height at the desired range dist_star."""
    p = params or default_cam()
    return p["fx"] * float(target_h) / float(dist_star)


# ---------------------------------------------------------------------------
# scripted virtual targets (D2)
# ---------------------------------------------------------------------------

KINDS = {
    "person":  {"speed": 0.8, "height": 1.7, "swing_deg": 0.0,  "period": None},
    "bicycle": {"speed": 2.5, "height": 1.1, "swing_deg": 45.0, "period": 8.0},
    "car":     {"speed": 6.0, "height": 1.5, "swing_deg": 0.0,  "period": None},
}
KIND_ALIAS = {"person": "person", "bicycle": "bicycle", "bike": "bicycle",
              "car": "car", "rand": "rand"}


def target_speed_scale():
    """Global target-speed multiplier: FPV_TARGET_SPEED_SCALE env var (float,
    default 1.0; invalid/missing falls back to 1.0). Read at CALL time so a
    trainer/launcher env override is honored in spawned sub-env processes.
    KINDS speeds (person 0.8 / bicycle 2.5 / car 6.0 m/s) exceed the drone's
    +-1 m/s action box; scaling down widens the trainable chase windows."""
    return _env_f("FPV_TARGET_SPEED_SCALE", 1.0)


def sample_target(rng, kind="rand", drone_start=(0.0, 0.0, 0.0),
                  drone_yaw=0.0, ground_z=-2.7, r_min=6.0, r_max=12.0,
                  duration=240.0, dt=0.05, params=None, speed_scale=None,
                  desired=None):
    """Sample one per-episode target script (D2). rng must expose
    random.Random-style random()/uniform()/randint()/. choice is done here.
    The trajectory is pre-integrated at dt so target_at(script, t) is a pure
    function of elapsed sim time (the bicycle +-45 deg / period-8 s swing has
    no closed form, so it is integrated once here).

    Spawn: ground level, horizontal range U[r_min, r_max] = U[6,12] m, bearing
    within +-25 deg of the drone's start yaw so the target starts inside the
    camera FOV (horizontal half-FOV = atan(cx0/fx) ~= 29.4 deg).

    desired (D11): when an episode desired range is given, the spawn band
    scales with it: U[max(2.0, 0.6*desired), max(3.0, 1.4*desired)] so a 15 m
    follow target does not start at a silly 6 m and a 1.5 m target does not
    start at 12 m. desired=None keeps the legacy U[r_min, r_max] band so
    existing checker code paths are untouched.

    Target speed = KINDS[kind]["speed"] * scale, where scale is the explicit
    speed_scale argument when not None, else FPV_TARGET_SPEED_SCALE (default
    1.0). The scaled speed is used BOTH for the trajectory integration AND as
    script["speed"], so target_at()'s straight-line extrapolation past
    t_end stays consistent.
    Returns dict(kind, speed, speed_scale, height, ground_z, dt, xy (N,2),
    end_heading, t_end).
    """
    p = params or default_cam()
    scale = target_speed_scale() if speed_scale is None else float(speed_scale)
    kind = KIND_ALIAS.get(str(kind).strip().lower(), "rand")
    if kind == "rand":
        kind = list(KINDS)[rng.randint(0, 2)]
    spec = KINDS[kind]
    fov_half = math.degrees(math.atan(p["cx0"] / p["fx"]))
    margin = math.radians(max(0.0, fov_half - 4.0))        # inside FOV (D2)
    if desired is not None:                   # D11 spawn band scales with goal
        r_min = max(2.0, 0.6 * float(desired))
        r_max = max(3.0, 1.4 * float(desired))
    rng0 = rng.uniform(r_min, r_max)
    bearing = drone_yaw + rng.uniform(-margin, margin)
    ds = np.asarray(drone_start, dtype=np.float64)
    x0 = ds[0] + rng0 * math.cos(bearing)
    y0 = ds[1] + rng0 * math.sin(bearing)
    heading0 = rng.uniform(0.0, 2.0 * math.pi)             # person heading /
    #                                        bicycle base heading / car road
    n = int(duration / dt) + 1
    t = np.arange(n) * dt
    sp = spec["speed"] * scale
    if spec["swing_deg"] > 0.0:
        psi = heading0 + math.radians(spec["swing_deg"]) * np.sin(
            2.0 * math.pi * t / spec["period"])
        x = x0 + np.cumsum(np.cos(psi)) * sp * dt
        y = y0 + np.cumsum(np.sin(psi)) * sp * dt
        end_h = float(psi[-1])
    else:
        x = x0 + sp * t * math.cos(heading0)
        y = y0 + sp * t * math.sin(heading0)
        end_h = heading0
    return {"kind": kind, "speed": sp, "speed_scale": scale,
            "height": spec["height"],
            "ground_z": float(ground_z), "dt": dt,
            "xy": np.stack([x, y], axis=1), "end_heading": end_h,
            "t_end": t[-1]}


def target_at(script, t):
    """Target base position (NED) at elapsed sim time t seconds (pure)."""
    dt, xy = script["dt"], script["xy"]
    if t >= script["t_end"]:                    # extrapolate straight on
        e = script["t_end"]
        dx = math.cos(script["end_heading"]) * script["speed"] * (t - e)
        dy = math.sin(script["end_heading"]) * script["speed"] * (t - e)
        return np.array([xy[-1, 0] + dx, xy[-1, 1] + dy, script["ground_z"]])
    i = min(int(t / dt), len(xy) - 2)
    a = (t - i * dt) / dt
    x = xy[i, 0] + a * (xy[i + 1, 0] - xy[i, 0])
    y = xy[i, 1] + a * (xy[i + 1, 1] - xy[i, 1])
    return np.array([x, y, script["ground_z"]])


# ---------------------------------------------------------------------------
# detector noise + latency (D6)
# ---------------------------------------------------------------------------

class DetectorNoise:
    """Per-step noisy box over the GT stream (D6). rng: random.Random-style.

    latency: update() pushes each GT box; the box the agent sees is the GT
    box L steps back (L = lat_steps, deque window). in_fov=False counts as a
    forced miss (physically invisible -> nothing to detect). On miss the box
    is FROZEN at the last hit and age++ ; on hit age resets to 0.
    .age counts steps since last hit; consumers use in_view = age <= K (D7
    uses K=20). .hits counts Bernoulli hits (for offline stat tests).
    """

    def __init__(self, rng, p_miss=0.15, jitter=0.005, lat_steps=0,
                 h_sigma=0.05, conf_mu=0.85, conf_sig=0.08, params=None):
        self.rng = rng
        self.p_miss = p_miss
        self.jitter = jitter
        self.h_sigma = h_sigma
        self.conf_mu = conf_mu
        self.conf_sig = conf_sig
        self.params = params or default_cam()
        self.buf = deque(maxlen=max(0, int(lat_steps)) + 1)
        self.age = 0
        self.last = None
        self.hits = 0

    def update(self, gt_box):
        """Push this step's GT box; return the (noisy, delayed) box the agent
        sees at this step, or None if no hit has ever occurred."""
        self.buf.append(gt_box)
        src = self.buf[0]
        if (not src.get("in_fov", False)) or self.rng.random() < self.p_miss:
            self.age += 1
            if self.last is None:
                return None
            return dict(self.last, age=self.age)          # frozen box (D6)
        p = self.params
        box = {
            "cx": src["cx"] + self.rng.gauss(0.0, self.jitter * p["W"]),
            "cy": src["cy"] + self.rng.gauss(0.0, self.jitter * p["H"]),
            "h_px": max(0.0, src["h_px"] * (1.0 + self.rng.gauss(0.0,
                                                                 self.h_sigma))),
            "conf": min(1.0, max(0.0, self.rng.gauss(self.conf_mu,
                                                     self.conf_sig))),
        }
        self.last = box
        self.age = 0
        self.hits += 1
        return dict(box, age=0)


def sample_latency_steps(rng, step_dt=0.5, lat_ms_lo=None, lat_ms_hi=None):
    """Per-episode latency L in steps (D6). With FPV_LAT_MS hi/lo given:
    L uniform over [round(lo/dt), round(hi/dt)]; else L ~ U{0,1}."""
    if lat_ms_lo is None or lat_ms_hi is None:
        return rng.randint(0, 1)
    lo = int(round(float(lat_ms_lo) / 1000.0 / step_dt))
    hi = max(lo, int(round(float(lat_ms_hi) / 1000.0 / step_dt)))
    return rng.randint(lo, hi)


def dr_detector_params(rng):
    """FPV_DR=1 domain randomization ranges (D6)."""
    return {"p_miss": rng.uniform(0.05, 0.25),
            "jitter": rng.uniform(0.0025, 0.01)}


# ---------------------------------------------------------------------------
# apparent-size range estimate (D12) -- the SAME measurement the phone app
# computes (VISION_ONBOARD_RESEARCH.md section 7, app 0.3.0 `Range` object)
# ---------------------------------------------------------------------------

def range_estimate(h_px, real_height, params=None):
    """Phone-identical apparent-size range (D12).

    C = H/fy equals 2*tan(vertical FOV / 2) for this pinhole model
    (vFOV = 2*atan(H/(2*fy))). frac = h_px/H is the box-height fraction of
    frame height, so range = real_height / (frac * C) inverts the projection
    h_px = fy*real_height/z exactly when z is the depth along the optical
    axis (that slant-range semantics is also what the phone produces --
    distance along the camera axis, not ground range). frac <= 0 -> inf
    (nothing measurable). Exactly the app formula; no sim-only extras.
    """
    p = params or default_cam()
    C = p["H"] / p["fy"]
    frac = float(h_px) / p["H"]
    if frac <= 0.0:
        return float("inf")
    return float(real_height) / (frac * C)


class RangeEstimator:
    """Smoothed, noisy apparent-size range estimate (D12) mirroring the phone
    app: EMA per track, +-noise (multiplicative), frame-edge-cut boxes and
    missing boxes are IGNORED (no update -- stale estimate persists but is
    flagged invalid), first valid hit seeds the EMA. rng: random.Random-style,
    shared with the episode rng so same-seed streams stay identical.

    update(box, edge_cut) -> (est, valid): box None or edge_cut True =>
    (self.est, False) WITHOUT updating. Freshness (box age) is the CALLER's
    business (D13): pass None for stale/frozen boxes if you want the app's
    "usable hit this step" semantics. est starts +inf (no measurement yet);
    a non-finite raw estimate is reported invalid and never poisons the EMA.
    Knobs FPV_RANGE_NOISE (0.2) / FPV_RANGE_EMA (0.3) are read at CALL time
    via _env_f when the explicit args are None (launcher overrides honored in
    spawned sub-env processes).
    """

    def __init__(self, rng, real_height, params=None, noise=None,
                 ema_alpha=None):
        self.rng = rng
        self.h = float(real_height)
        self.params = params or default_cam()
        self.noise = _env_f("FPV_RANGE_NOISE", 0.2) if noise is None \
            else float(noise)
        self.alpha = _env_f("FPV_RANGE_EMA", 0.3) if ema_alpha is None \
            else float(ema_alpha)
        self.est = float("inf")

    def update(self, box, edge_cut=False):
        """One step. Returns (est, valid) -- see class docstring."""
        if box is None or edge_cut:
            return self.est, False
        raw = range_estimate(box["h_px"], self.h, self.params) \
            * self.rng.uniform(1.0 - self.noise, 1.0 + self.noise)
        if not math.isfinite(raw):
            return self.est, False
        self.est = raw if not math.isfinite(self.est) \
            else self.alpha * raw + (1.0 - self.alpha) * self.est
        return self.est, True


# ---------------------------------------------------------------------------
# depth grid (D5)
# ---------------------------------------------------------------------------

def depth_grid(drone_pos, drone_yaw, objects, params=None,
               rows=3, cols=5, d_min=1.0, d_max=20.0):
    """3x5 depth grid (row-major, length rows*cols) per D5.

    objects: iterable of (name, pos(3,), half(3,)) scene AABBs (half-extents,
    metres). Each AABB's 8 corners are projected; the 2D bbox of the in-front
    corners approximates the pixel cone; every cell whose rectangle intersects
    that bbox gets min(current, (d-1)/(d_max-d_min)) where d is the point-to-
    AABB distance from the drone. Free cells stay 1.0 (D5: no-obstacle = 1.0).
    All values clipped to [0, 1]. Honest approximation of the pixel cone --
    documented in spec open-risk 4 (may be near-useless in the sparse Blocks
    scene; no faking).
    """
    p = params or default_cam()
    grid = np.ones(rows * cols, dtype=np.float32)
    f, r, d = cam_axes(drone_yaw, p)
    drone = np.asarray(drone_pos, dtype=np.float64)
    for _name, pos, half in objects:
        pos = np.asarray(pos, dtype=np.float64)
        half = np.broadcast_to(np.asarray(half, dtype=np.float64), (3,))
        dd = float(np.linalg.norm(np.maximum(np.abs(drone - pos) - half, 0.0)))
        val = (min(max(dd, 0.0), d_max) - d_min) / (d_max - d_min)
        val = min(1.0, max(0.0, val))
        us, vs = [], []
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    rel = pos + np.array([sx * half[0], sy * half[1],
                                          sz * half[2]]) - drone
                    z = float(rel @ f)
                    if z <= 0.1:
                        continue
                    us.append(min(max(p["cx0"] + p["fx"] * float(rel @ r) / z,
                                      0.0), p["W"] - 1.0))
                    vs.append(min(max(p["cy0"] + p["fy"] * float(rel @ d) / z,
                                      0.0), p["H"] - 1.0))
        if len(us) < 1:
            continue                         # entirely behind the camera
        u0, u1 = min(us), max(us)
        v0, v1 = min(vs), max(vs)
        c0 = min(max(int(u0 / p["W"] * cols), 0), cols - 1)
        c1 = min(max(int(u1 / p["W"] * cols), 0), cols - 1)
        r0 = min(max(int(v0 / p["H"] * rows), 0), rows - 1)
        r1 = min(max(int(v1 / p["H"] * rows), 0), rows - 1)
        for ri in range(r0, r1 + 1):
            for ci in range(c0, c1 + 1):
                k = ri * cols + ci
                if val < grid[k]:
                    grid[k] = val
    return grid


# ---------------------------------------------------------------------------
# detour-required eval layout (D15) -- PURE, used by follow_env FPV_DETOUR
# ---------------------------------------------------------------------------

def detour_blocker(boxes, start, yaw, lo=3.0, hi=10.0, cone_deg=60.0):
    """Pick the 'detour-required' blocker among the enumerated scene AABBs
    (D15). boxes: iterable of (name, center(3,), half) -- the env's
    _grid_objs, NO world mutation (pure selection over geometry). Returns
    (name, center) of the FIRST box whose center is
      * lo..hi metres (3-D) from the drone's start position, AND
      * within +-cone_deg degrees of the drone's start yaw,
    else None. Deterministic: first match in enumeration order -- NO rng
    draw here; the caller controls rng consumption via detour_target().
    """
    s = np.asarray(start, dtype=np.float64)
    for name, center, _half in boxes:
        c = np.asarray(center, dtype=np.float64)
        d = float(np.linalg.norm(c - s))
        if not (lo <= d <= hi):
            continue
        bearing = math.atan2(float(c[1] - s[1]), float(c[0] - s[0]))
        diff = (bearing - float(yaw) + math.pi) % (2.0 * math.pi) - math.pi
        if abs(math.degrees(diff)) <= cone_deg:
            return str(name), c
    return None


def detour_target(script, center, drone_start, rng, extra_lo=3.0,
                  extra_hi=6.0):
    """Override the scripted target's START so it lies BEHIND the blocker
    (D15): on the ray drone_start -> center, extended past the centre to
    cube_dist + U[extra_lo, extra_hi] metres; ground z stays the script's.
    The whole pre-integrated trajectory is shifted RIGIDLY (script['xy'] is
    replaced by xy + delta) so the script's motion is otherwise UNCHANGED.
    rng draws EXACTLY ONE uniform (the extension) and the caller runs this
    AFTER the existing reset() draws (desired -> detector knobs -> script),
    so with FPV_DETOUR off (never called) same-seed streams are untouched.
    Returns the start->target-start ray distance (m) for logging.
    """
    s = np.asarray(drone_start, dtype=np.float64)
    c = np.asarray(center, dtype=np.float64)
    d = max(float(np.linalg.norm(c - s)), 1e-6)
    ray = d + float(rng.uniform(extra_lo, extra_hi))
    p = s + (c - s) / d * ray                 # ground z comes from the script
    xy = script["xy"]
    script["xy"] = xy + (p[:2] - xy[0])
    return float(ray)


# ---------------------------------------------------------------------------
# reward v6 (D7 + D11) -- PURE, dominance-proven by check_follow_reward.py
# ---------------------------------------------------------------------------

R_STEP_CEILING = 1.1          # 0.5 + 0.4 + 0.2, penalties only subtract
TERM_LANDED = -10.0           # drone landed mid-episode (D7)
TERM_FLEW_OUT = -10.0         # > leave-radius from start (D7, radius per D11)
TERM_COLLISION = -20.0        # collision with anything (D7)
DIST_STAR = 6.0               # fallback desired range only (D11: goal per-ep)
CLOSE_GUARD = 1.5             # metres (v6): never-this-close "danger zone"
CLOSE_PENALTY = 2.0           # per-step subtraction inside CLOSE_GUARD (v6)
K_VIEW = 20                   # steps until a lost box "ages out" (D7)


def follow_reward(cx_err, cy_err, dist, in_view, a_norm, da_norm, v_norm,
                  landed=False, flew_out=False, collided=False,
                  desired=DIST_STAR):
    """S7 follow reward v6 (D7 + D11) -- PURE function, no sim imports.

    Survival-shaped, NO positive terminal anywhere; terminals are bad-only
    and their magnitudes exceed anything that early termination can harvest
    (proved by check_follow_reward.py -- the failure mode that killed all the
    Q3 hover stages). check order: collided -> landed/flew_out -> per-step.

        r = 0.5*in_view + 0.4*max(0,1-|cx_err,cy_err|)
            + 0.2*max(0,1-|log(dist/desired)|)
            - 0.05*||a|| - 0.03*||a-a_prev|| - 0.02*||v||
            - CLOSE_PENALTY if dist < CLOSE_GUARD   (per-step, NOT terminal)

    GOAL-CONDITIONED (D11): `desired` is the per-episode follow range the
    policy is conditioned on (obs block carries it, D13). the owner's literal ask
    is -|true_range - desired|; we keep the LOG form |log(dist/desired)|
    because it is scale-free -- an absolute-metre term would make a 1.5 m
    follow episode ~10x harder to earn range reward than a 15 m one, so the
    gradient would skew toward the far end of the desired sweep. The log form
    gives 1.5 m and 15 m episodes EQUAL weighting (same relative error =
    same term), which is the whole point of goal-conditioning.

    Close-guard (v6): inside CLOSE_GUARD metres the step loses CLOSE_PENALTY
    every step -- NOT a terminal (terminals stay bad-only landed/flew-out -10,
    collision -20). The checker proves 2.0 > the max 0.2 range-term gain, so
    parking on top of the target can never pay even with range_term = 0.2.
    """
    if collided:
        return TERM_COLLISION
    if landed or flew_out:
        return TERM_LANDED
    center = max(0.0, 1.0 - math.hypot(cx_err, cy_err))
    range_t = max(0.0, 1.0 - abs(math.log(max(dist, 0.05) / desired)))
    r = (0.5 * (1.0 if in_view else 0.0) + 0.4 * center + 0.2 * range_t
         - 0.05 * a_norm - 0.03 * da_norm - 0.02 * v_norm)
    if dist < CLOSE_GUARD:
        r -= CLOSE_PENALTY
    return r
