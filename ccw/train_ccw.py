"""CCW-Net training for the drone hover policy: Algorithm 1 of the paper
(warmup -> mediator intermission -> alignment), plus the no-alignment baseline
(same arch, same warmup, same total epochs, alignment losses identically zero).

Centerpiece paper: Rountree, Verma, So, Fan, Shah. "Causal Concept Explanations
for Deep Neural Models." NeurIPS 2026. arXiv 2611.14407.

Continuous adaptation vs the StS harness: Yt is the expert SAC mean action
A (N,3) float32 (NO one-hot); L_task stays F.mse_loss(yh, yb); the concept
supervision is presence BCE on slot lengths; test metric tMSE = mean((yh-yb)^2)
(per-element mean) and act_agree_pct = ||yh-yb||_inf < 0.1 vs the expert.
L_suf/L_ind/L_pres/L_nec math is IDENTICAL to the harness (they are output-dim
agnostic and work for R=3).

Losses (Eq 2/3): L_task MSE + L_cls BCE + 0.1*(L_nec + L_suf + L_ind + L_pres).
Prop.1 unit check runs before training (mandatory).
"""
import sys, time, json, argparse
sys.path.insert(0, r'D:\fpv\ccw')
import numpy as np
import torch
import torch.nn.functional as F
import ccw_net as cn
from ccw_net import Encoder, Head, MediatorFlow, make_cond, singleton_deltas, flow_fit, prop1_checks

torch.set_num_threads(6)


def get_batches(n, bs, gen):
    perm = torch.randperm(n, generator=gen)
    for st in range(0, n, bs):
        yield perm[st:st + bs]


def compute_test(enc, head, Z, C, Y, test_idx, bs=4096):
    """tMSE (per-element mean squared error) + act_agree_pct on the test split."""
    enc.eval(); head.eval()
    sq = 0.0
    agree = 0
    ntot = 0
    with torch.no_grad():
        for st in range(0, len(test_idx), bs):
            idx = test_idx[st:st + bs]
            M = enc(Z[idx])
            yh = head(M)
            yb = Y[idx]
            sq += float(((yh - yb) ** 2).sum())
            agree += int(((yh - yb).abs().max(dim=1).values < 0.1).sum())
            ntot += len(idx)
    enc.train(); head.train()
    return sq / ntot / cn.R, 100.0 * agree / ntot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=r'D:\fpv\ccw\ccw_data.npz')
    ap.add_argument('--align', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--flow-epochs', type=int, default=10)
    ap.add_argument('--refit-epochs', type=int, default=1)
    ap.add_argument('--refit-every', type=int, default=10)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--flow-batch', type=int, default=2048)
    ap.add_argument('--S', type=int, default=2)
    ap.add_argument('--out', default=None)
    ap.add_argument('--log', default=None)
    a = ap.parse_args()
    tag = 'aligned' if a.align else 'noalign'
    if a.out is None:
        a.out = rf'D:\fpv\ccw\ccw_{tag}_s{a.seed}.pt'
    shared_log = a.log is not None
    if a.log is None:
        a.log = rf'D:\fpv\ccw\train_{tag}_seed{a.seed}.log'
    # shared per-run log (passed by the task wrapper): append so all four runs
    # land in one file and the LAST line is always a TRAIN_DONE line.
    lf = open(a.log, 'a' if shared_log else 'w', buffering=1)

    def P(*s):
        msg = " ".join(str(x) for x in s)
        print(msg, flush=True)
        lf.write(msg + "\n")

    P("=== CCW-Net train (drone hover, continuous R=%d): align=%d seed=%d E=%d Ew=%d batch=%d S=%d ===" %
      (cn.R, a.align, a.seed, a.epochs, a.warmup, a.batch, a.S))
    const = prop1_checks()
    P(f"Prop.1 checks PASSED (h(0) constant = tanh(b), mean={const:.4f})")

    d = np.load(a.data, allow_pickle=False)
    keep = d['keep_mask'].astype(bool)
    keep_idx = np.nonzero(keep)[0]
    J = len(keep_idx)
    Zt = torch.tensor(d['Z'])
    Ct = torch.tensor(d['labels'][:, keep_idx].astype(np.float32))
    Yt = torch.tensor(d['A'].astype(np.float32))   # (N,3) expert mean actions
    N = Zt.shape[0]
    rng = np.random.RandomState(0)
    perm = rng.permutation(N)
    n_test = int(0.15 * N)
    test_idx = torch.tensor(perm[:n_test])
    train_idx = torch.tensor(perm[n_test:])
    P(f"N={N} J={J} (concepts kept idx {keep_idx.tolist()}) "
      f"train={len(train_idx)} test={n_test}")

    torch.manual_seed(a.seed)
    enc = Encoder(J)
    head = Head(J)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=5e-4)
    flow = MediatorFlow(J) if a.align else None
    Ztr = Zt[train_idx]
    Ctr = Ct[train_idx]
    Ytr = Yt[train_idx]
    ntr = len(train_idx)
    gen = torch.Generator()
    gen.manual_seed(a.seed * 1000 + 1)

    t0 = time.time()
    last_log = t0
    hist = []

    def fit_flow(which='initial'):
        t = time.time()
        with torch.no_grad():
            Mparts = []
            for st in range(0, ntr, 4096):
                Mparts.append(enc(Ztr[st:st + 4096]))
            M_all = torch.cat(Mparts)
        eps = a.flow_epochs if which == 'initial' else a.refit_epochs
        h = flow_fit(flow, M_all, Ztr, Ctr, J, eps, a.flow_batch, 'cpu',
                     log_fn=None)
        P(f"[flow {which}] epochs={eps} nll {h[0]:.2f}->{h[-1]:.2f} wall={time.time() - t:.0f}s")

    def epoch(epno, align_on):
        nonlocal last_log
        ep_t0 = time.time()
        acc = dict(task=0.0, cls=0.0, align=0.0, n=0)
        for idx in get_batches(ntr, a.batch, gen):
            zb, cb, yb = Ztr[idx], Ctr[idx], Ytr[idx]
            opt.zero_grad(set_to_none=True)
            M = enc(zb)
            yh = head(M)
            L_task = F.mse_loss(yh, yb)
            length = M.norm(dim=-1).clamp(1e-6, 1 - 1e-6)
            L_cls = F.binary_cross_entropy(length, cb)
            total = L_task + L_cls
            L_al = None
            if align_on:
                dj, h0 = singleton_deltas(head, M)
                ysum = torch.einsum('brj,bj->br', dj, cb)
                Delta0 = yh - h0
                L_suf = ((h0 + ysum - yb) ** 2).sum(1).mean()
                L_ind = ((Delta0 - ysum) ** 2).sum(1).mean()
                beta = 1.0 / cb.sum(1).clamp_min(1.0)
                n0 = Delta0.norm(dim=1, keepdim=True)
                ndj = dj.norm(dim=1)
                L_pres = ((F.relu(beta[:, None] * n0 - ndj) * cb).sum(1)).mean()
                cnt = (1 - cb).sum().clamp_min(1.0)
                L_nec = torch.zeros(())
                B = zb.shape[0]
                for j in range(J):
                    if float((1 - cb[:, j]).sum()) <= 0:
                        continue
                    with torch.no_grad():   # no grad through the mediator flow
                        cf = flow.sample(a.S, make_cond(zb, cb, j).detach()).view(-1, cn.D)
                    Mrep = M.repeat_interleave(a.S, dim=0)   # grad flows via non-cf slots
                    Mrep[:, j, :] = cf
                    ycf = head(Mrep)
                    dd = ((ycf - yh.repeat_interleave(a.S, dim=0)) ** 2).sum(1).view(B, a.S).mean(1)
                    L_nec = L_nec + (dd * (1 - cb[:, j])).sum()
                L_nec = L_nec / cnt
                L_al = 0.1 * (L_nec + L_suf + L_ind + L_pres)
                total = total + L_al
            if not torch.isfinite(total):
                continue  # numerical safety net (cf draws were sanitized)
            total.backward()
            torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(head.parameters()), 10.0)
            opt.step()
            acc['task'] += float(L_task) * len(idx)
            acc['cls'] += float(L_cls) * len(idx)
            acc['align'] += float(L_al) if L_al is not None else 0.0
            acc['n'] += len(idx)
        tm, ag = compute_test(enc, head, Zt, Ct, Yt, test_idx)
        row = dict(ep=epno, align=int(align_on), task=acc['task'] / acc['n'],
                   cls=acc['cls'] / acc['n'], alignL=acc['align'] / acc['n'],
                   tMSE=tm, agree=ag, wall_s=round(time.time() - t0, 1))
        hist.append(row)
        now = time.time()
        if now - last_log > 30 or epno < 3 or not align_on:
            last_log = now
            P(f"[{epno:4d}] al={int(align_on)} task={row['task']:.4f} cls={row['cls']:.4f} "
              f"alignL={row['alignL']:.4f} tMSE={tm:.4f} agree={ag:.1f}% wall={row['wall_s']}s")
        return row

    # ---- warmup (both models identical, L_task + L_cls only)
    for e in range(a.warmup):
        epoch(e, False)
    # ---- intermission: fit mediator on frozen encoder concepts
    if a.align:
        fit_flow('initial')
    # ---- alignment phase (baseline: same epochs, alignment weight 0 forever)
    align_start = a.warmup
    for e in range(align_start, a.epochs):
        if a.align and (e - align_start) > 0 and (e - align_start) % a.refit_every == 0:
            fit_flow(f'refit@{e}')
        epoch(e, bool(a.align))

    tm, ag = compute_test(enc, head, Zt, Ct, Yt, test_idx)
    P(f"FINAL {tag} seed={a.seed}: tMSE={tm:.5f} agree={ag:.2f}% wall={time.time() - t0:.0f}s")
    torch.save(dict(enc=enc.state_dict(), head=head.state_dict(),
                    flow=flow.state_dict() if flow else None,
                    J=J, keep_idx=keep_idx, test_idx=test_idx.numpy(),
                    train_idx=train_idx.numpy(), args=vars(a),
                    history=hist, wall_s=time.time() - t0), a.out)
    P(f"SAVED {a.out}")
    # watchdog line: LAST line of the shared log must contain TRAIN_DONE
    P(f"TRAIN_DONE {tag} seed={a.seed} tMSE={tm:.5f} agree={ag:.2f}")
    lf.close()


if __name__ == '__main__':
    main()
