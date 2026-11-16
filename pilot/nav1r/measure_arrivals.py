import json, math, datetime
LAT0, LON0, M = 30.395, -85.670, 111320.0
COS = math.cos(math.radians(LAT0))
rows = [json.loads(l) for l in open('wire_run1.jsonl') if l.strip()]
T = [r for r in rows if r['dir'] == 'tx' and r['k'] == 'T']
V = [r for r in rows if r['dir'] == 'rx' and r['k'] == 'V']
C = [r for r in rows if r['dir'] == 'rx' and r['k'] == 'C']
def ne(r): return ((r['lat']-LAT0)*M, (r['lon']-LON0)*M*COS)
def dist(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])
def E(h, m, s, ms=0):
    d = datetime.datetime(2026, 11, 15, 15, h, m, s, datetime.timezone.utc)
    return int(d.timestamp()*1000)+ms
def sample_near(tms, win=400):
    best = None
    for r in T:
        d = abs(r['t']-tms)
        if d <= win and (best is None or d < best[0]):
            best = (d, ne(r))
    return best
print("== Run 1 arrivals (targets: wp0 home+(19.96N,0), wp1 home+(19.96N,19.96E); REACH_M=2.5) ==")
for name, tms, tgt in [('wp0', E(25,21,442), (19.96,0.0)), ('wp1', E(26,1,683), (19.96,19.96))]:
    b = sample_near(tms)
    if b: print(f"{name}: dist at reached = {dist(b[1],tgt):.2f} m (sample {b[0]} ms off)")
    # min dist in +-1s window
    mn = min((dist(ne(r), tgt), r['t']) for r in T if abs(r['t']-tms) <= 1000)
    print(f"      min dist +-1 s = {mn[0]:.2f} m")
print("== dwells (wp0 reached 15:25:21.4 -> wp1 reached 15:26:01.7 = leg2 start; hover10? plan hover12s; wp1 dwell 15:26:01.7->09:26:16.7 finished = plan hover15s) ==")
print("wp0 dwell+leg time to wp1: 40.2 s (12 s dwell + ~20 m leg at <=1 m/s + reach tolerance) ; wp1 dwell: 15.0 s -> finished (matches hover15)")
mag = lambda r: abs(r.get('vn',0))+abs(r.get('ve',0))+abs(r.get('vd',0))
for w0, w1, label in [(E(24,59,0), E(25,20,0), 'r1 leg1'), (E(26,55,0), E(27,31,0), 'r2 anomaly'), (E(30,39,0), E(31,10,0), 'after STOP')]:
    vs = [r for r in V if w0 <= r['t'] <= w1]
    nz = [r for r in vs if mag(r) > 0.05]
    zs = [r for r in vs if mag(r) <= 0.05]
    print(f"{label}: V={len(vs)} nonzero={len(nz)} zero={len(zs)}")
r2p = [ne(r) for r in T if E(26,55,0) <= r['t'] <= E(27,31,0)]
if r2p:
    print("r2 drone first:", f"{r2p[0][0]:.1f},{r2p[0][1]:.1f}", " last:", f"{r2p[-1][0]:.1f},{r2p[-1][1]:.1f}")
for r in C:
    print("C frame:", r.get('cmd'), r['t'])
# cadence: T send intervals overall
ts = [r['t'] for r in T]
import statistics
gaps = [b-a for a, b in zip(ts, ts[1:])]
print("T rows", len(T), "median gap ms", statistics.median(gaps), "max gap", max(gaps))
# after STOP: last V time vs disable
if V:
    print("last V", V[-1]['t'], "last C", C[-1]['t'] if C else None)
