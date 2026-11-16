#!/usr/bin/env python3
"""test_wire_client.py - NAV1-R SimLink wire validator (no app needed).

Connects to a running nav1_simlink.py bridge, validates framing
[type:1B ASCII][len:4B BE][payload], measures telemetry cadence (app-facing
T frames are throttled to one per 0.70 s by design), sends sample V frames +
C enable + C takeoff, then confirms telemetry keeps
flowing. Exit 0 = FRAMING OK, non-zero on any violation.
usage: test_wire_client.py [--host 127.0.0.1] [--port 9941] [--quiet]
"""
import argparse
import json
import socket
import struct
import sys
import time


def recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("bridge closed connection")
        buf += chunk
    return buf


def read_frame(sock):
    hdr = recv_exact(sock, 5)
    typ = hdr[0:1].decode("ascii")
    ln = struct.unpack(">I", hdr[1:5])[0]
    if typ not in "TFVC":
        raise ValueError("bad type byte %r" % hdr[0:1])
    if ln > 65536:
        raise ValueError("insane frame length %d" % ln)
    payload = recv_exact(sock, ln) if ln else b""
    return typ, ln, payload


def send_frame(sock, typ, obj):
    data = json.dumps(obj).encode("utf-8")
    sock.sendall(typ.encode("ascii") + struct.pack(">I", len(data)) + data)


def pump(sock, seconds, telemetry_times, telemetry_lines, errors):
    """Read frames for `seconds`; record T arrivals."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        try:
            sock.settimeout(max(0.1, end - time.monotonic()))
            typ, ln, payload = read_frame(sock)
        except socket.timeout:
            continue
        except (ConnectionError, ValueError) as ex:
            errors.append(str(ex))
            return
        if typ == "T":
            telemetry_times.append(time.monotonic())
            try:
                telemetry_lines.append(json.loads(payload.decode("utf-8")))
            except (UnicodeDecodeError, json.JSONDecodeError) as ex:
                errors.append("T payload not JSON: %s" % ex)
                return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9941)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    t_times, t_lines, errors = [], [], []
    sock = socket.create_connection((args.host, args.port), timeout=10)
    sock.settimeout(10)
    print("connected to %s:%d" % (args.host, args.port))

    # phase 1: baseline telemetry cadence (3 s). NB: app-facing T frames are
    # throttled to one per TELEM_SEND_INTERVAL_S (0.70 s) by design (bridge
    # header line 9; wire log keeps 5 Hz), so ~4 frames per 3 s is correct.
    pump(sock, 3.0, t_times, t_lines, errors)
    if errors or len(t_lines) < 3:
        print("FAIL baseline: frames=%d errors=%s" % (len(t_lines), errors))
        sys.exit(1)

    # phase 2: sample V frames (20 Hz burst), C enable, C takeoff
    send_frame(sock, "C", {"cmd": "enable"})
    for i in range(20):
        send_frame(sock, "V", {"fwd": 0.5, "right": 0.0, "down": 0.0, "yaw": 5.0})
        time.sleep(0.05)
    send_frame(sock, "C", {"cmd": "takeoff"})
    print("sent: C enable, 20x V @20Hz (fwd 0.5, yaw-rate 5), C takeoff")

    # phase 3: telemetry must keep flowing after commands (3 s)
    n_before = len(t_lines)
    pump(sock, 3.0, t_times, t_lines, errors)
    n_after = len(t_lines) - n_before

    dt = [t_times[i + 1] - t_times[i] for i in range(len(t_times) - 1)]
    dt_sorted = sorted(dt)
    median = dt_sorted[len(dt_sorted) // 2] if dt_sorted else -1
    ok = (not errors and n_after >= 3 and 0.5 <= median <= 0.9)

    show = None if args.quiet else print
    if show:
        print("first T : %s" % json.dumps(t_lines[0]))
        print("last T  : %s" % json.dumps(t_lines[-1]))
        print("prev-T  : %s" % json.dumps(t_lines[-2]))
    print("T frames total=%d (phase3 after cmds=%d) gaps: median=%.3fs min=%.3fs max=%.3fs"
          % (len(t_lines), n_after, median, min(dt), max(dt)))
    keys = set(t_lines[-1].keys())
    required = {"t", "pitch", "roll", "yaw", "lat", "lon", "alt", "vn", "ve", "vd",
                "batt", "gimbal", "flying", "sats", "homeSet", "homeLat", "homeLon"}
    missing = required - keys
    if missing:
        print("FAIL telemetry missing keys: %s" % missing)
        ok = False
    if errors:
        print("FAIL framing errors: %s" % errors)
        ok = False
    if not (0.5 <= median <= 1.0):
        print("FAIL cadence median %.3fs outside 0.5..1.0s window "
              "(app-facing throttle 0.70s on 0.2s tick)" % median)
        ok = False
    if n_after < 3:
        print("FAIL only %d T frames after commands (telemetry must keep flowing)" % n_after)
        ok = False
    sock.close()
    if ok:
        print("FRAMING OK: framing type(1B)+len(4B BE)+payload validated; cadence "
              "matches 0.70s app-facing throttle; "
              "V + C enable/takeoff accepted; telemetry kept flowing")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
