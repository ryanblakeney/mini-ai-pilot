#!/usr/bin/env python3
"""nav1_simlink.py - NAV1-R SimLink bridge for the Unreal Blocks sim (Yoga).

Locked design: NAV1R_RIG_SPEC.md (kanban card t_70db6699).

TCP server on 127.0.0.1:9930, SimLink framing both directions, per SimLink.kt:
    [type:1 byte ASCII 'T'|'F'|'V'|'C'][length:4 bytes BIG-endian][payload]
  bridge->phone  T  telemetry JSON, sampled at 5 Hz but SENT to the phone at
                    1 per TELEM_SEND_INTERVAL_S (0.70 s); wire log keeps 5 Hz
                    (F/JPEG frames NOT sent here).
  phone->bridge  V  {"fwd","right","down","yaw"} body-frame velocity (20 Hz).
  phone->bridge  C  {"cmd":"enable|disable|takeoff|land|home|cancel_home"}.

Honesty notes (S6-report style, see spec section "Honesty requirements"):
- lat/lon are a FICTIONAL flat-earth spot around (30.395, -85.670); mapping
  replicates Geo.kt exactly: n=(lat-lat0)*111320, e=(lon-lon0)*111320*cos(lat0).
  sim NED: n=x, e=y, alt=-z. No OSM/GPS truth.
- battery=100 and sats=12 are CONSTANTS (sim has neither).
- The installed projectairsim has NO get_ground_truth_velocity (checked on the
  Yoga venv 2026-11-11), so vn/ve/vd are FINITE DIFFERENCES of
  get_ground_truth_kinematics() position at the 5 Hz telemetry rate.
- Attitude comes from pose.orientation quaternion via the gym_env.py euler
  convention: yaw = degrees clockwise from north (0 = north, CW+, mod 360).
  pitch/roll are display-only; NED raw pitch is +nose-DOWN, we FLIP it here so
  the app shows + = nose-up. Roll is raw NED (+ = right wing down).
- 'V'.yaw is a yaw RATE (deg/s). The spec bodyToNe formula needs a heading, so
  the conversion uses the drone's CURRENT heading (from kinematics), not the
  rate; the commanded yaw rate is logged but NOT actuated (Blocks would cancel
  the velocity task if we interleaved rotate commands; route legs are
  translation-only in this rehearsal). Deviation noted in the report.

--no-sim dry mode: no projectairsim import at all (import is lazy); a synthetic
drone orbits the origin at 5 m AGL and integrates accepted 'V' commands at
10 Hz. Wire log + status sentinels behave identically. Used for wiring
validation from sable.
"""
import argparse
import asyncio
import json
import math
import os
import socket
import struct
import sys
import threading
import time

LAT0 = 30.395
LON0 = -85.670
M_PER_DEG = 111320.0
COS0 = math.cos(math.radians(LAT0))
SCENE = "scene_basic_drone.jsonc"

# Rehearsal automation: the Android app rewrites its HUD on every T frame, so a
# 5 Hz app-facing send rate never lets uiautomator reach UI idle ("could not
# get idle state"). Throttle only the socket write (sampling + wire log stay at
# 5 Hz). The app's SafetyConfig.telemTimeoutMs is 1000 ms, so the effective
# interval must stay well under ~0.85 s — 0.70 s it is.
TELEM_SEND_INTERVAL_S = 0.70


def now_ms():
    return int(time.time() * 1000)


def ned_to_latlon(n, e):
    """Inverse of the app's Geo.kt flat-earth mapping (see docstring)."""
    return LAT0 + n / M_PER_DEG, LON0 + e / (M_PER_DEG * COS0)


def quat_to_rpy(w, x, y, z):
    """Same convention as sim/gym_env.py quat_to_rpy (validated against obs).
    yaw is NED heading: 0 = north, CW positive."""
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)
    s = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(s)
    sinp = 2.0 * (w * z + x * y)
    cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(sinp, cosp)
    return roll, pitch, yaw


def body_to_ne(fwd, right, heading_deg):
    """Spec bodyToNe, heading = CURRENT drone heading in deg (0=N, CW+)."""
    r = math.radians(heading_deg)
    return fwd * math.cos(r) - right * math.sin(r), fwd * math.sin(r) + right * math.cos(r)


def frame(typ, payload):
    return typ.encode("ascii") + struct.pack(">I", len(payload)) + payload


class _LoopThread(threading.Thread):
    """ONE persistent asyncio loop thread (pynng sockets are loop-affine —
    proven pattern from D:\\fpv\\rl\\gym_env.py)."""

    def __init__(self):
        super().__init__(daemon=True)
        self.loop = asyncio.new_event_loop()

    def run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def sync(self, coro, timeout=90):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def stop(self):
        self.loop.call_soon_threadsafe(self.loop.stop)


async def _chain(coro):
    """Await move_*_async's inner RPC, then the returned command task."""
    return await (await coro)


class StallWatchdog:
    """Pure decision logic for the NAV1-STALL watchdog (card t_429279c8).

    Rehearsal run 2 defect: the sim silently stopped applying MoveByVelocity
    after the drone stopped at the final waypoint (sim-side autopilot
    takeover: uncommanded yaw drift, commands 'accepted' but no motion).
    Detection: control_on + command fresh + commanded horizontal norm
    >= CMD_MIN while observed horizontal norm <= OBS_STALL continuously for
    STALL_WINDOW s. Recovery ladder: hover, then enable_api_control+arm;
    each step is followed by VERIFY_S s of normal command re-issue before
    testing whether motion returned (obs > OBS_RECOVER while cmd >= CMD_MIN).
    Ladder steps run ONCE per detection; after unrecovered the flag is sticky
    (LIVE line carries stall=1) and re-arms after REARM_S.

    All timing comes from the caller's `now` (monotonic seconds), so
    test_simlink_watchdog.py can drive it with a scripted clock. update()
    returns a list of event strings: 'detected', 'step_hover', 'step_enable',
    'recovered:NAME', 'unrecovered', 'rearmed'.
    """
    CMD_MIN = 0.35       # m/s horizontal command norm to count as "commanded to move"
    OBS_STALL = 0.12     # m/s observed at/below this counts as stalled
    OBS_RECOVER = 0.30   # m/s observed above this counts as motion returned
    STALL_WINDOW = 2.5   # s of sustained stall before detection
    VERIFY_S = 2.0       # s of command re-issue after each ladder step
    REARM_S = 30.0       # s before a sticky unrecovered flag re-arms
    LADDER = ("hover", "enable_api")
    STEP_EVENT = {"hover": "step_hover", "enable_api": "step_enable"}

    def __init__(self):
        self.phase = "idle"          # idle | verify | failed
        self.stall_since = None
        self.step = 0                # number of ladder steps already issued
        self.verify_since = 0.0
        self.failed_at = 0.0
        self.stall_sticky = False    # True while stuck -> LIVE line adds stall=1
        self.last_detect = None      # {"cmd","obs"} norms at detection (logging)

    def update(self, now, commanding, cmd_norm, obs_norm):
        ev = []
        stall = bool(commanding and cmd_norm >= self.CMD_MIN
                     and obs_norm <= self.OBS_STALL)
        if stall:
            if self.stall_since is None:
                self.stall_since = now  # set once at stall onset, not per tick
        else:
            self.stall_since = None

        if obs_norm > self.OBS_RECOVER and self.phase != "idle":
            # motion observed — reset the watchdog no matter the phase
            was, last_step = self.phase, self.step
            self.phase, self.stall_sticky, self.step = "idle", False, 0
            if was == "failed":
                ev.append("recovered:auto")
            elif was == "verify":
                ev.append("recovered:" + self.LADDER[last_step - 1])
        if self.phase == "failed":
            if stall and now - self.failed_at >= self.REARM_S:
                self.phase, self.stall_sticky, self.step = "idle", False, 0
                self.stall_since = now
                ev.append("rearmed")
            else:
                return ev
        if not stall:
            return ev

        if self.phase == "idle":
            if now - self.stall_since >= self.STALL_WINDOW:
                self.phase = "verify"
                self.step = 1
                self.verify_since = now
                self.last_detect = {"cmd": cmd_norm, "obs": obs_norm}
                ev.append("detected")
                ev.append(self.STEP_EVENT[self.LADDER[0]])
        elif self.phase == "verify":
            if now - self.verify_since >= self.VERIFY_S:
                if self.step < len(self.LADDER):
                    ev.append(self.STEP_EVENT[self.LADDER[self.step]])
                    self.step += 1
                    self.verify_since = now
                else:
                    self.phase = "failed"
                    self.stall_sticky = True
                    self.failed_at = now
                    ev.append("unrecovered")
        return ev


class Status:
    """Append-only status file; last line is the current sentinel."""

    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)

    def write(self, line):
        with self.lock:
            try:
                with open(self.path, "a") as f:
                    f.write("%s %s\n" % (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), line))
            except OSError:
                pass


class SimDrone:
    """Real Blocks-backed drone (proven client pattern from gym_env.py)."""

    def __init__(self):
        self.lp = _LoopThread()
        self.lp.start()
        self.client = self.world = self.drone = None
        self.spawn = None  # {"x","y","z"} at connect
        self.kin_fails = 0
        self.note = lambda line: None  # bridge sets this to its own note()
        self.last_vresp = None   # (monotonic, repr(result-or-exception))
        self.vresp_fails = 0
        self._vresp_seen_first = False
        self._vresp_err_last = 0.0

    def connect(self):
        from projectairsim import ProjectAirSimClient  # lazy
        last = None
        for _ in range(30):
            try:
                self.client = ProjectAirSimClient(port_topics=8989, port_services=8990)
                self.client.connect()
                break
            except Exception as ex:  # sim may still be booting
                last = ex
                self.client = None
                time.sleep(2)
        if self.client is None:
            raise RuntimeError("sim connect failed after retries: %s" % last)
        self.lp.sync(self._init_world(), timeout=120)

    async def _init_world(self):
        from projectairsim import Drone, World  # lazy
        self.world = World(self.client, SCENE, delay_after_load_sec=3)
        self.drone = Drone(self.client, self.world, "Drone1")
        self.drone.enable_api_control()

    def _kin(self):
        try:
            k = self.drone.get_ground_truth_kinematics()
        except AttributeError:
            k = None
        if not k:
            k = self.drone.get_kinematics()
        return k

    def kin(self):
        async def go():
            return self._kin()
        return self.lp.sync(go(), timeout=8)

    def state(self, prev):
        k = self.kin()
        self.kin_fails = 0
        p = k["pose"]["position"]
        o = k["pose"]["orientation"]
        n, e, alt = float(p["x"]), float(p["y"]), -float(p["z"])
        roll, pitch, yaw = quat_to_rpy(o["w"], o["x"], o["y"], o["z"])
        t = time.monotonic()
        if prev is None or not (0.0 < t - prev["t"] <= 1.0):
            vn = ve = vd = 0.0
        else:  # finite difference at telemetry rate (no GT velocity API)
            dt = t - prev["t"]
            # vd follows the app 'V.down' convention: POSITIVE = descending.
            vn, ve, vd = (n - prev["n"]) / dt, (e - prev["e"]) / dt, (prev["alt"] - alt) / dt
        return {"n": n, "e": e, "alt": alt, "vn": vn, "ve": ve, "vd": vd,
                "yaw_deg": math.degrees(yaw) % 360.0,
                "pitch_deg": -math.degrees(pitch),
                "roll_deg": math.degrees(roll),
                "z": float(p["z"])}

    def set_velocity(self, vn, ve, vd, duration=0.5):
        async def go():
            inner = await asyncio.wait_for(
                self.drone.move_by_velocity_async(vn, ve, vd, duration=duration),
                timeout=1.5)
            # NAV1-STALL: the outer await only SENT the RPC; `inner` is the
            # task that awaits the sim's actual answer. Record what it says
            # instead of discarding it (rehearsal run 2: sim silently stopped
            # applying these commands — response was never inspected).
            inner.add_done_callback(self._on_vresp)
        self.lp.sync(go(), timeout=2.5)

    def _on_vresp(self, task):
        try:
            if task.cancelled():
                res, failed = "cancelled", True
            else:
                exc = task.exception()
                if exc is not None:
                    res, failed = repr(exc), True
                else:
                    res, failed = repr(task.result()), False
        except Exception as ex:  # callback must never raise
            res, failed = "callback_err:%r" % ex, True
        self.last_vresp = (time.monotonic(), res)
        if not self._vresp_seen_first:
            self._vresp_seen_first = True
            self.note("VRESP first=" + res)
        if failed:
            self.vresp_fails += 1
            if time.monotonic() - self._vresp_err_last >= 2.0:  # <= 1 note / 2 s
                self._vresp_err_last = time.monotonic()
                self.note("VRESP_ERR %s" % res)

    def hover(self):
        """Recovery step 1: Hover RPC (double-await: outer sends, inner answers)."""
        async def go():
            await asyncio.wait_for(_chain(self.drone.hover_async()), timeout=10)
        self.lp.sync(go(), timeout=15)

    def reenable(self):
        """Recovery step 2: re-assert API control + arm (sync requests)."""
        async def go():
            self.drone.enable_api_control()
            self.drone.arm()
        self.lp.sync(go(), timeout=10)

    def takeoff(self):
        async def go():
            self.drone.arm()
            await asyncio.wait_for(self.drone.takeoff_async(timeout_sec=60), timeout=90)
            k = self._kin()
            p = k["pose"]["position"]
            z = self.spawn["z"] - 5.0  # 5 m AGL above spawn pad
            await asyncio.wait_for(_chain(
                self.drone.move_to_position_async(p["x"], p["y"], z, velocity=2.0)),
                timeout=120)
        self.lp.sync(go(), timeout=240)

    def land(self):
        async def go():
            try:
                await asyncio.wait_for(self.drone.land_async(timeout_sec=60), timeout=90)
            except Exception:
                pass
        self.lp.sync(go(), timeout=150)

    def go_home(self, cur_z):
        async def go():
            await asyncio.wait_for(_chain(
                self.drone.move_to_position_async(self.spawn["x"], self.spawn["y"],
                                                 cur_z, velocity=2.0)),
                timeout=180)
        self.lp.sync(go(), timeout=240)

    def cancel(self):
        async def go():
            try:
                self.drone.cancel_last_task()
            except Exception:
                pass
        self.lp.sync(go(), timeout=5)

    def shutdown(self):
        try:
            self.lp.sync(self._shutdown_async(), timeout=30)
        except Exception:
            pass
        self.lp.stop()

    async def _shutdown_async(self):
        try:
            self.drone.cancel_last_task()
        except Exception:
            pass
        try:
            self.client.disconnect()  # mandatory or python hangs (gym_env lesson)
        except Exception:
            pass


class SyntheticDrone:
    """--no-sim mode: orbit origin r=5 m, alt 5 m AGL; integrate V at 10 Hz."""

    def __init__(self, bridge):
        self.bridge = bridge
        self.kin_fails = 0
        self.n, self.e, self.alt = 0.0, 5.0, 5.0
        self.yaw_deg = 90.0
        self._prev = None

    def run(self, stop):
        while not stop.is_set():
            t0 = time.monotonic()
            cmd = self.bridge.active_cmd()
            if cmd is not None:
                self.n += cmd["vn"] * 0.1
                self.e += cmd["ve"] * 0.1
                self.alt -= cmd["vd"] * 0.1  # vd is +down
                self.yaw_deg = (self.yaw_deg + cmd.get("yaw", 0.0) * 0.1) % 360.0
            else:
                self._th = getattr(self, "_th", math.atan2(self.e, self.n))
                self._th += 0.01  # ~0.1 rad/s -> 0.5 m/s tangential at r=5 m
                self.n, self.e = 5.0 * math.cos(self._th), 5.0 * math.sin(self._th)
                self.alt = 5.0
                self.yaw_deg = math.degrees(self._th + math.pi / 2) % 360.0
            stop.wait(max(0.0, 0.1 - (time.monotonic() - t0)))

    def state(self, prev):
        t = time.monotonic()
        if prev is None or not (0.0 < t - prev["t"] <= 1.0):
            vn = ve = vd = 0.0
        else:
            dt = t - prev["t"]
            vn, ve = (self.n - prev["n"]) / dt, (self.e - prev["e"]) / dt
            vd = -(self.alt - prev["alt"]) / dt
        return {"n": self.n, "e": self.e, "alt": self.alt,
                "vn": vn, "ve": ve, "vd": vd,
                "yaw_deg": self.yaw_deg, "pitch_deg": 0.0, "roll_deg": 0.0, "z": -self.alt}

    def set_velocity(self, *a, **k):
        pass

    def hover(self):
        pass

    def reenable(self):
        pass

    def takeoff(self):
        pass

    def land(self):
        pass

    def go_home(self, cur_z):
        pass

    def cancel(self):
        pass

    def shutdown(self):
        pass


class Bridge:
    def __init__(self, args):
        self.args = args
        self.status = Status(args.status_file)
        mode = "no-sim" if args.no_sim else "sim"
        self.mode = mode
        os.makedirs(args.wirelog_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        self.wlog_path = os.path.join(args.wirelog_dir, "nav1_wire_%s.jsonl" % stamp)
        self.wf = open(self.wlog_path, "a")
        self.wlock = threading.Lock()
        self.seq = 0
        self.send_lock = threading.Lock()
        self.client = None
        self.control_on = False
        self.flying = False
        self.busy = False          # takeoff/land/home in flight
        self.cmd_ne = None         # {"vn","ve","vd","yaw"} last V converted
        self.cmd_ts = 0.0
        self.deadman_sent = True
        self.stop = threading.Event()
        self.prev_state = None
        self.last_state = None
        self.last_live = 0.0
        self.last_telem_send = 0.0  # monotonic; app-facing T-frame throttle
        self.watchdog = StallWatchdog()
        if args.no_sim:
            self.drone = SyntheticDrone(self)
            self.vel_src = "no-sim-integrated-velocity-at-10Hz"
        else:
            self.drone = SimDrone()
            self.drone.note = self.note  # VRESP/VRESP_ERR notes
            self.vel_src = ("finite-difference of get_ground_truth_kinematics at 5 Hz "
                            "(no get_ground_truth_velocity in installed projectairsim)")

    @property
    def last_vresp(self):
        """Latest MoveByVelocity RPC answer: (monotonic, repr) or None."""
        return getattr(self.drone, "last_vresp", None)

    # ---- logging ----
    def wirelog(self, dirn, k, payload=None):
        with self.wlock:
            self.seq += 1
            rec = {"seq": self.seq, "t": now_ms(), "dir": dirn, "k": k}
            if isinstance(payload, dict):
                rec.update(payload)
            self.wf.write(json.dumps(rec, separators=(",", ":")) + "\n")
            self.wf.flush()

    def note(self, line):
        self.status.write("NOTE " + line)

    # ---- networking ----
    def bind(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            srv.bind((self.args.host, self.args.port))
        except OSError as ex:
            self.status.write("NAV1R_ABORT bind failed: %s" % ex)
            raise
        srv.listen(1)
        self.status.write("BRIDGE_LISTENING mode=%s port=%d wirelog=%s"
                          % (self.mode, self.args.port, self.wlog_path))
        self.note("battery=100 sats=12 are constants; velocity source: %s; "
                  "no camera (F) frames sent; fictional lat/lon spot" % self.vel_src)
        return srv

    @staticmethod
    def _recv_exact(sock, n):
        buf = b""
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("client closed")
            buf += chunk
        return buf

    def serve_client(self, conn, addr):
        with self.send_lock:
            self.client = conn
        self.status.write("APP_CONNECTED %s:%d" % addr)
        self.note("battery=100 and sats=12 constants (sim has neither)")
        # Auto-takeoff shortly after connect so the drone is settled hovering at
        # 5 m before the operator engages (C enable -> app HOLD zero-velocity at
        # 20 Hz would otherwise race the takeoff task). The enable trigger in
        # handle_cmd stays as a safety net (guarded by flying/busy flags).
        def _auto_takeoff_on_connect():
            if self.mode != "no-sim" and self.stop.wait(3.0):
                return
            if not self.stop.is_set() and not self.flying and not self.busy:
                self.busy = True
                threading.Thread(target=self._do_takeoff, daemon=True).start()
        threading.Thread(target=_auto_takeoff_on_connect, daemon=True).start()
        try:
            while not self.stop.is_set():
                hdr = self._recv_exact(conn, 5)
                typ = hdr[0:1].decode("ascii", "replace")
                ln = struct.unpack(">I", hdr[1:5])[0]
                if ln > 65536:
                    raise ValueError("frame too long: %d" % ln)
                payload = self._recv_exact(conn, ln) if ln else b""
                try:
                    data = json.loads(payload.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as ex:
                    self.wirelog("rx", typ if typ in "TFVC" else "?")
                    self.note("BAD_PAYLOAD type=%s len=%d: %s" % (typ, ln, ex))
                    continue
                if typ == "V" and isinstance(data, dict):
                    fwd = float(data.get("fwd", 0.0))
                    right = float(data.get("right", 0.0))
                    down = float(data.get("down", 0.0))
                    yaw_rate = float(data.get("yaw", 0.0))
                    st = self.last_state or {"yaw_deg": 0.0}
                    vn, ve = body_to_ne(fwd, right, st.get("yaw_deg", 0.0))
                    self.cmd_ne = {"vn": vn, "ve": ve, "vd": down, "yaw": yaw_rate}
                    self.cmd_ts = time.monotonic()
                    self.deadman_sent = False
                    self.wirelog("rx", "V", {**data, "vn": round(vn, 4), "ve": round(ve, 4)})
                elif typ == "C" and isinstance(data, dict):
                    self.wirelog("rx", "C", data)
                    self.handle_cmd(str(data.get("cmd", "")))
                else:
                    self.note("UNKNOWN_FRAME type=%s payload=%s" % (typ, payload[:80]))
        except (ConnectionError, ConnectionResetError, OSError) as ex:
            self.status.write("APP_DISCONNECTED %s" % type(ex).__name__)
        except ValueError as ex:
            self.status.write("APP_PROTOCOL_ERROR %s" % ex)
            try:
                conn.close()
            except OSError:
                pass
        with self.send_lock:
            if self.client is conn:
                self.client = None
        try:
            conn.close()
        except OSError:
            pass

    def accept_loop(self, srv):
        while not self.stop.is_set():
            try:
                conn, addr = srv.accept()
            except OSError:
                break
            with self.send_lock:
                if self.client is not None:
                    self.status.write("REFUSED_EXTRA_CONN %s:%d" % addr)
                    try:
                        conn.close()
                    except OSError:
                        pass
                    continue
            threading.Thread(target=self.serve_client, args=(conn, addr), daemon=True).start()

    def send_telemetry(self, telem):
        cli = self.client
        if cli is None:
            return
        data = json.dumps(telem).encode("utf-8")
        try:
            # Throttle the SOCKET write only (uiautomator idle — see
            # TELEM_SEND_INTERVAL_S); the wire-log row below keeps the 5 Hz
            # sampling cadence unchanged.
            now = time.monotonic()
            if now - self.last_telem_send >= TELEM_SEND_INTERVAL_S:
                self.last_telem_send = now
                with self.send_lock:
                    cli.sendall(frame("T", data))
            self.wirelog("tx", "T", telem)
        except OSError:
            self.status.write("APP_SEND_FAILED")
            with self.send_lock:
                if self.client is cli:
                    self.client = None
            try:
                cli.close()
            except OSError:
                pass

    # ---- command semantics (spec section "Command apply semantics") ----
    def active_cmd(self):
        if (self.control_on and self.cmd_ne is not None
                and (time.monotonic() - self.cmd_ts) < 1.5):
            return self.cmd_ne
        return None

    def handle_cmd(self, cmd):
        if cmd == "enable":
            self.control_on = True
            self.note("CMD_ENABLE control_on=1")
            if not self.flying and not self.busy:
                self.busy = True
                threading.Thread(target=self._do_takeoff, daemon=True).start()
        elif cmd == "disable":
            self.control_on = False
            try:
                self.drone.set_velocity(0.0, 0.0, 0.0, duration=2.0)
            except Exception:
                pass
            self.note("CMD_DISABLE control_on=0 hover(0,0,0)")
        elif cmd == "takeoff":
            if not self.flying and not self.busy:
                self.busy = True
                threading.Thread(target=self._do_takeoff, daemon=True).start()
            self.note("CMD_TAKEOFF")
        elif cmd == "land":
            if self.flying and not self.busy:
                self.busy = True
                threading.Thread(target=self._do_land, daemon=True).start()
            self.note("CMD_LAND")
        elif cmd == "home":
            if not self.busy:
                self.busy = True
                threading.Thread(target=self._do_home, daemon=True).start()
            self.note("CMD_HOME rtl-to-spawn")
        elif cmd == "cancel_home":
            self.drone.cancel()
            self.note("CMD_CANCEL_HOME")
        else:
            self.note("CMD_UNKNOWN %s" % cmd)

    def _do_takeoff(self):
        try:
            self.drone.takeoff()
            self.flying = True
            self.status.write("AUTO_TOOK_OFF")
        except Exception as ex:
            self.note("TAKEOFF_FAILED %s" % ex)
        finally:
            self.busy = False

    def _do_land(self):
        try:
            self.drone.land()
            self.flying = False
            self.note("LANDED")
        except Exception as ex:
            self.note("LAND_FAILED %s" % ex)
        finally:
            self.busy = False

    def _do_home(self):
        try:
            z = self.last_state.get("z", -5.0) if self.last_state else -5.0
            self.drone.go_home(z)
            self.note("HOME_DONE")
        except Exception as ex:
            self.note("HOME_FAILED %s" % ex)
        finally:
            self.busy = False

    # ---- loops ----
    def telemetry_loop(self):
        self.last_state = None
        while not self.stop.is_set():
            t0 = time.monotonic()
            try:
                st = self.drone.state(self.prev_state)
                self.prev_state = {"t": t0, "n": st["n"], "e": st["e"], "alt": st["alt"]}
                self.last_state = st
            except Exception:
                self.drone.kin_fails += 1
                if self.drone.kin_fails >= 10:
                    self.status.write("NAV1R_SIM_LOST kinematics failed 10x")
                    self.stop.set()
                    break
                st = None
            if st is not None:
                lat, lon = ned_to_latlon(st["n"], st["e"])
                telem = {"t": now_ms(),
                         "pitch": round(st["pitch_deg"], 2), "roll": round(st["roll_deg"], 2),
                         "yaw": round(st["yaw_deg"], 2),
                         "lat": round(lat, 7), "lon": round(lon, 7),
                         "alt": round(st["alt"], 3),
                         "vn": round(st["vn"], 3), "ve": round(st["ve"], 3),
                         "vd": round(st["vd"], 3),
                         "batt": 100, "gimbal": 0.0, "flying": self.flying, "sats": 12,
                         "homeSet": True, "homeLat": LAT0, "homeLon": LON0}
                self.send_telemetry(telem)
                if time.monotonic() - self.last_live >= 2.0:
                    self.last_live = time.monotonic()
                    age = time.monotonic() - self.cmd_ts if self.cmd_ts else -1.0
                    stall_flag = " stall=1" if getattr(self.watchdog, "stall_sticky",
                                                        False) else ""
                    self.status.write("LIVE client=%d control_on=%d flying=%d cmd_age_s=%.1f "
                                      "n=%.2f e=%.2f alt=%.2f yaw=%.0f%s"
                                      % (1 if self.client else 0, int(self.control_on),
                                         int(self.flying), age, st["n"], st["e"], st["alt"],
                                         st["yaw_deg"], stall_flag))
            self.stop.wait(max(0.0, 0.2 - (time.monotonic() - t0)))

    def velocity_loop(self):
        while not self.stop.wait(0.2):
            if not self.control_on or self.cmd_ne is None:
                continue
            fresh = (time.monotonic() - self.cmd_ts) < 1.5
            try:
                if fresh:
                    self.drone.set_velocity(self.cmd_ne["vn"], self.cmd_ne["ve"],
                                            self.cmd_ne["vd"], duration=0.5)
                    self.deadman_sent = False
                elif not self.deadman_sent:
                    self.drone.set_velocity(0.0, 0.0, 0.0, duration=2.0)  # deadman hover
                    self.status.write("DEADMAN_HOVER cmd_age_s>1.5")
                    self.deadman_sent = True
            except Exception as ex:
                self.note("VELOCITY_FAIL %s (continuing)" % ex)
            self._watchdog_tick(fresh)

    def _watchdog_tick(self, fresh):
        """NAV1-STALL watchdog: sim mode only (this loop only runs in sim mode).
        Commanded to move >= 0.35 m/s but observed <= 0.12 m/s for >= 2.5 s ->
        STALL_DETECTED, then recovery ladder hover -> enable_api+arm."""
        if self.mode != "sim":
            return
        try:
            cmd_norm = 0.0
            commanding = bool(self.control_on and fresh and self.cmd_ne is not None)
            if commanding:
                cmd_norm = math.hypot(self.cmd_ne["vn"], self.cmd_ne["ve"])
            obs_norm = 0.0
            st = self.last_state
            if st is not None:
                obs_norm = math.hypot(st["vn"], st["ve"])
            for ev in self.watchdog.update(time.monotonic(), commanding,
                                           cmd_norm, obs_norm):
                self._watchdog_action(ev, cmd_norm, obs_norm)
        except Exception as ex:
            self.note("WATCHDOG_FAIL %s (continuing)" % ex)

    def _watchdog_action(self, ev, cmd_norm, obs_norm):
        if ev == "detected":
            lv = self.last_vresp
            self.note("STALL_DETECTED cmd=%.2f obs=%.2f last_vresp=%s vresp_fails=%d"
                      % (cmd_norm, obs_norm,
                         (lv[1] if lv else "none"),
                         getattr(self.drone, "vresp_fails", 0)))
        elif ev == "step_hover":
            try:
                self.drone.hover()
                self.note("STALL_STEP hover issued")
            except Exception as ex:
                self.note("STALL_STEP hover FAILED %s" % ex)
        elif ev == "step_enable":
            try:
                self.drone.reenable()
                self.note("STALL_STEP enable_api+arm issued")
            except Exception as ex:
                self.note("STALL_STEP enable FAILED %s" % ex)
        elif ev.startswith("recovered:"):
            self.note("STALL_RECOVERED via=%s" % ev.split(":", 1)[1])
        elif ev == "unrecovered":
            self.note("STALL_UNRECOVERED ladder exhausted; sticky until motion "
                      "returns or 30 s re-arm")
        elif ev == "rearmed":
            self.note("STALL_REARMED re-detecting")

    def run(self):
        if self.mode == "sim":
            try:
                self.drone.connect()
                k = self.drone.kin()
                p = k["pose"]["position"]
                self.drone.spawn = {"x": float(p["x"]), "y": float(p["y"]), "z": float(p["z"])}
                self.note("SIM_CONNECTED kin=%s" % json.dumps(p))
            except Exception as ex:
                self.status.write("NAV1R_SIM_LOST connect: %s" % ex)
                self.stop.set()
                return 3
        srv = self.bind()
        threading.Thread(target=self.accept_loop, args=(srv,), daemon=True).start()
        threading.Thread(target=self.telemetry_loop, daemon=True).start()
        if self.mode == "sim":
            threading.Thread(target=self.velocity_loop, daemon=True).start()
        else:
            threading.Thread(target=self.drone.run, args=(self.stop,), daemon=True).start()
        deadline = time.monotonic() + self.args.duration if self.args.duration else None
        try:
            while not self.stop.is_set():
                if deadline and time.monotonic() > deadline:
                    self.note("DURATION_ELAPSED")
                    break
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        self.status.write("BRIDGE_EXIT seq=%d wirelog=%s" % (self.seq, self.wlog_path))
        self.stop.set()
        try:
            srv.close()
        except OSError:
            pass
        self.drone.shutdown()
        try:
            self.wf.flush()
            self.wf.close()
        except OSError:
            pass
        return 0


def main():
    ap = argparse.ArgumentParser(description="NAV1-R SimLink bridge (spec NAV1R_RIG_SPEC.md)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9930)
    ap.add_argument("--no-sim", action="store_true", help="synthetic drone, no projectairsim")
    ap.add_argument("--status-file", default=r"D:\fpv\ops\nav1_r_status.txt")
    ap.add_argument("--wirelog-dir", default=r"D:\fpv\ops")
    ap.add_argument("--duration", type=float, default=0.0, help="seconds, 0 = forever")
    args = ap.parse_args()
    # projectairsim import is LAZY (inside SimDrone.connect) so --no-sim works
    # without touching Blocks or the venv's sim stack.
    bridge = Bridge(args)
    rc = bridge.run()
    sys.stdout.flush()
    sys.stderr.flush()
    if args.no_sim:
        sys.exit(rc)
    # sim mode: hard exit — pending sim RPC tasks hang python teardown
    # (gym_env lesson: segfault with pending land task).
    os._exit(rc)


if __name__ == "__main__":
    main()
