#!/usr/bin/env python3
"""nav1_stall_probe.py - NAV1-STALL standalone reproduction probe (card t_429279c8).

Standalone: same single-loop-thread + lazy projectairsim import pattern as
nav1_simlink.py (pynng sockets are loop-affine). Run ONLY via the FPVNAV1Probe
interactive scheduled task (DXGI wall) with cwd
D:\\fpv\\ProjectAirSim\\client\\python\\example_user_scripts.

Rehearsal run 2 (2026-11-15): the drone stopped at the final waypoint and then
became uncommandable - MoveByVelocity every 0.2 s for minutes, sim accepted
everything, ZERO horizontal motion, uncommanded yaw drift. Plain mid-route
zero-hover dwells did NOT trigger it. This probe reproduces the minimal
sequence without the phone app:

  phaseA leg (0.5,0,0) 12 s  -> should move
  phaseB zero-hover    30 s
  phaseC leg (0.5,0,0) 12 s  -> if disp < 1 m the stall is REPRODUCED
  recovery ladder (only if C stalled): hover -> re-test 6 s;
    enable_api_control+arm -> re-test 6 s; log RECOVERY=hover|enable_api|none
  phaseD zero-hover 20 s + leg 12 s -> check it can happen twice

Deliberately awaits a few MoveByVelocity inner response tasks (3 s cap) so we
see what the sim actually answers (the bridge discards these today).

Outputs: D:\\fpv\\ops\\nav1_probe_status.txt (one line per event) and
D:\\fpv\\ops\\nav1_probe_02s.jsonl (tick rows t,phase,n,e,alt,yaw,cmd).
Self-timeout 900 s; always ends os._exit(0) (pending RPC tasks hang teardown).
"""
import asyncio
import json
import math
import os
import sys
import threading
import time

STATUS = r"D:\fpv\ops\nav1_probe_status.txt"
TICKS = r"D:\fpv\ops\nav1_probe_02s.jsonl"
SCENE = "scene_basic_drone.jsonc"
DEADLINE_S = 900.0

T0 = time.monotonic()


def log(line):
    try:
        with open(STATUS, "a") as f:
            f.write("%s %s\n" % (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), line))
    except OSError:
        pass


class _LoopThread(threading.Thread):
    """ONE persistent asyncio loop thread (pynng loop-affinity, bridge pattern)."""

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


class Probe:
    def __init__(self):
        self.lp = _LoopThread()
        self.lp.start()
        self.client = self.world = self.drone = None
        self._resp_n = 0
        self._resp_tasks = set()
        self.tf = open(TICKS, "a")

    # ---- client plumbing (bridge-identical) ----
    def connect(self):
        from projectairsim import ProjectAirSimClient  # lazy
        last = None
        for _ in range(30):
            try:
                self.client = ProjectAirSimClient(port_topics=8989, port_services=8990)
                self.client.connect()
                break
            except Exception as ex:
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

    async def _kin_raw(self):
        try:
            k = self.drone.get_ground_truth_kinematics()
        except AttributeError:
            k = None
        if not k:
            k = self.drone.get_kinematics()
        return k

    def kin(self):
        """returns (n, e, alt_up, yaw_deg_cw_from_north)"""
        async def go():
            k = await self._kin_raw()
            p = k["pose"]["position"]
            o = k["pose"]["orientation"]
            w, x, y, z = o["w"], o["x"], o["y"], o["z"]
            yaw = math.degrees(math.atan2(2.0 * (w * z + x * y),
                                          1.0 - 2.0 * (y * y + z * z))) % 360.0
            return float(p["x"]), float(p["y"]), -float(p["z"]), yaw
        return self.lp.sync(go(), timeout=8)

    def cmd(self, vn, ve, vd, dur=0.5, sample=False):
        async def go():
            inner = await asyncio.wait_for(
                self.drone.move_by_velocity_async(vn, ve, vd, duration=dur),
                timeout=1.5)
            if sample:
                tk = asyncio.ensure_future(self._sample(inner))
                self._resp_tasks.add(tk)
                tk.add_done_callback(self._resp_tasks.discard)
        self.lp.sync(go(), timeout=2.5)

    async def _sample(self, inner):
        """Deliberately await the MoveByVelocity response with a 3 s cap."""
        self._resp_n += 1
        n = self._resp_n
        try:
            res = await asyncio.wait_for(asyncio.shield(inner), timeout=3.0)
            log("VRESP #%d ok=%r" % (n, res))
        except asyncio.TimeoutError:
            log("VRESP #%d NO_RESPONSE_AFTER_3S" % n)
        except asyncio.CancelledError:
            log("VRESP #%d cancelled" % n)
        except Exception as ex:
            log("VRESP #%d ERR %r" % (n, ex))

    async def _chain(self, coro):
        return await (await coro)

    def hover(self):
        async def go():
            await asyncio.wait_for(self._chain(self.drone.hover_async()), timeout=10)
        self.lp.sync(go(), timeout=15)

    def reenable(self):
        async def go():
            self.drone.enable_api_control()
            self.drone.arm()
        self.lp.sync(go(), timeout=10)

    def takeoff_5m(self):
        async def go():
            self.drone.arm()
            await asyncio.wait_for(self.drone.takeoff_async(timeout_sec=60), timeout=90)
            k = await self._kin_raw()
            p = k["pose"]["position"]
            z = float(p["z"]) - 5.0  # 5 m AGL above current (spawn pad)
            await asyncio.wait_for(self._chain(
                self.drone.move_to_position_async(float(p["x"]), float(p["y"]), z,
                                                  velocity=2.0)),
                timeout=120)
        self.lp.sync(go(), timeout=240)

    def _disconnect(self):
        async def go():
            try:
                self.client.disconnect()
            except Exception:
                pass
        return go()

    # ---- phase runner ----
    def run_phase(self, label, dur, vn, ve, vd, sample_first=0):
        t0 = time.monotonic()
        i = 0
        n0, e0, alt0, yaw0 = self.kin()
        while (time.monotonic() - t0 < dur) and (time.monotonic() - T0 < DEADLINE_S):
            tc0 = time.monotonic()
            try:
                self.cmd(vn, ve, vd, dur=0.5, sample=(i < sample_first))
            except Exception as ex:
                log("%s CMD_FAIL %r" % (label, ex))
            try:
                n, e, alt, yaw = self.kin()
                self.tf.write(json.dumps({"t": round(time.monotonic() - T0, 2),
                                          "phase": label, "n": round(n, 3),
                                          "e": round(e, 3), "alt": round(alt, 3),
                                          "yaw": round(yaw, 1),
                                          "cmd": [vn, ve]}) + "\n")
            except Exception as ex:
                log("%s KIN_FAIL %r" % (label, ex))
            i += 1
            time.sleep(max(0.0, 0.2 - (time.monotonic() - tc0)))
        try:
            self.tf.flush()
        except OSError:
            pass
        try:
            n1, e1, alt1, yaw1 = self.kin()
        except Exception:
            n1, e1, alt1, yaw1 = n0, e0, alt0, yaw0
        disp = math.hypot(n1 - n0, e1 - e0)
        dyaw = (yaw1 - yaw0 + 180.0) % 360.0 - 180.0
        log("PHASE %s dur=%.1fs disp=%.2fm yaw %.0f->%.0f (d=%+.0f) alt %.2f->%.2f ticks=%d"
            % (label, time.monotonic() - t0, disp, yaw0, yaw1, dyaw, alt0, alt1, i))
        return disp

    def recovery_ladder(self, tag):
        """Run hover then enable_api_control+arm, each followed by a 6 s leg
        re-test. Returns True if motion was restored."""
        try:
            self.hover()
        except Exception as ex:
            log("%s RECOVERY_HOVER_FAIL %r" % (tag, ex))
        d = self.run_phase("%s1_leg_after_hover" % tag, 6, 0.5, 0, 0)
        if d >= 1.0:
            log("RECOVERY_%s=hover disp=%.2f" % (tag, d))
            return True
        try:
            self.reenable()
        except Exception as ex:
            log("%s RECOVERY_ENABLE_FAIL %r" % (tag, ex))
        d2 = self.run_phase("%s2_leg_after_enable" % tag, 6, 0.5, 0, 0)
        if d2 >= 1.0:
            log("RECOVERY_%s=enable_api disp=%.2f" % (tag, d2))
            return True
        log("RECOVERY_%s=none" % tag)
        return False


def main():
    p = Probe()
    rc = 0
    try:
        log("PROBE_START")
        p.connect()
        n, e, alt, yaw = p.kin()
        log("CONNECTED n=%.2f e=%.2f alt=%.2f yaw=%.0f" % (n, e, alt, yaw))
        p.takeoff_5m()
        n, e, alt, yaw = p.kin()
        log("TOOKOFF_5M alt=%.2f" % alt)

        p.run_phase("A_leg", 12, 0.5, 0, 0, sample_first=2)
        p.run_phase("B_zero", 30, 0.0, 0.0, 0.0, sample_first=2)
        dispC = p.run_phase("C_leg", 12, 0.5, 0, 0, sample_first=2)
        if dispC < 1.0:
            log("C_STALLED disp=%.2f — running recovery ladder" % dispC)
            p.recovery_ladder("C")
        else:
            log("C_FLEW disp=%.2f — no stall after zero-hover dwell" % dispC)

        p.run_phase("D_zero", 20, 0.0, 0.0, 0.0)
        dispD = p.run_phase("D_leg", 12, 0.5, 0, 0)
        if dispD < 1.0:
            log("PHASE_D disp=%.2f STALL_AGAIN — running recovery ladder" % dispD)
            p.recovery_ladder("D")
        else:
            log("PHASE_D disp=%.2f FLEW" % dispD)

        if time.monotonic() - T0 >= DEADLINE_S:
            log("SELF_TIMEOUT_HIT")
        log("FINAL_HOVER")
        try:
            p.hover()
        except Exception as ex:
            log("HOVER_FAIL %r" % ex)
        log("PROBE_DONE")
    except Exception as ex:
        import traceback
        log("PROBE_FAILED %r" % ex)
        try:
            log("TRACEBACK " + traceback.format_exc().replace("\n", " | ")[-800:])
        except Exception:
            pass
        rc = 1
    finally:
        try:
            p.tf.flush()
            p.tf.close()
        except OSError:
            pass
        try:
            p.lp.sync(p._disconnect(), timeout=10)
        except Exception:
            pass
        log("WRAPPER python exiting rc=%d" % rc)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
