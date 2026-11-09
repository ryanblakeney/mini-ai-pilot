#!/usr/bin/env python3
"""Mavic Mini link helper (verified 2026-11-08). Talks DUML through the Pi bridge tunnel.
Tunnel first:  ssh -f -N -L 9910:127.0.0.1:9910 pi@<pi-address>
CLI: mavic_link.py read | gimbal <delta_deg>   (gimbal is RELATIVE, one command, clipped to -90..+20 using the pushed angle)
Flight commands are NOT implemented on purpose (need the pilot's per-stage OK, props off)."""
import socket, struct, sys, time
def crc8(b):
    c = 0x77
    for x in b:
        c ^= x
        for _ in range(8): c = (c >> 1) ^ 0x8C if c & 1 else c >> 1
    return c
def crc16(b):
    c = 0x3692
    for x in b:
        c ^= x
        for _ in range(8): c = (c >> 1) ^ 0x8408 if c & 1 else c >> 1
    return c
def duml(snd, rcv, seq, ctype, cset, cid, pl=b''):
    n = 13 + len(pl)
    h = bytes([0x55, n & 255, 0x04 | (n >> 8)]); h += bytes([crc8(h), snd, rcv]) + struct.pack('<H', seq) + bytes([ctype, cset, cid]) + pl
    return h + struct.pack('<H', crc16(h))
def wrap(d): return b'\x55\xcc\x49\x57' + struct.pack('<I', len(d)) + d
class Link:
    def __init__(self, port=9910):
        self.s = socket.create_connection(('127.0.0.1', port), timeout=0.5); self.buf = b''; self.seq = 0x700
    def frames(self, sec):
        end = time.time() + sec
        while time.time() < end:
            try: self.buf += self.s.recv(65536)
            except socket.timeout: pass
        out = []
        while True:
            j = self.buf.find(b'\x55\xcc')
            if j < 0 or len(self.buf) < j + 8: break
            l = struct.unpack('<I', self.buf[j + 4:j + 8])[0]
            if len(self.buf) < j + 8 + l: break
            body = self.buf[j + 8:j + 8 + l]; self.buf = self.buf[j + 8 + l:]; k = 0
            while k + 13 <= len(body) and body[k] == 0x55:
                L = body[k + 1] | ((body[k + 2] & 3) << 8); out.append(body[k:k + L]); k += max(L, 1)
        return out
    def gimbal_pitch(self, sec=1.2):
        v = [struct.unpack('<h', f[11:13])[0] for f in self.frames(sec) if len(f) > 14 and f[4] == 4 and f[9] == 4 and f[10] == 5]
        return v[-1] / 10 if v else None
    def gimbal_move(self, delta_deg):
        """ONE relative pitch move (sender 0x02 = APP; the PC sender 0x0a is ignored). Never send twice."""
        cur = self.gimbal_pitch()
        if cur is None: raise RuntimeError('no gimbal angle: aircraft not linked')
        d = max(-90 - cur, min(20 - cur, delta_deg))   # clip to the Mini's range
        self.seq += 1
        self.s.sendall(wrap(duml(2, 4, self.seq, 0x40, 4, 0x14, struct.pack('<hhh', 0, 0, int(round(d * 10))) + bytes([0, 1]))))
        time.sleep(3); return self.gimbal_pitch()
if __name__ == '__main__':
    L = Link()
    if sys.argv[1] == 'read': print('gimbal pitch deg:', L.gimbal_pitch(2))
    elif sys.argv[1] == 'gimbal': print('gimbal pitch deg now:', L.gimbal_move(float(sys.argv[2])))
