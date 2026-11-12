"""CCW-Net evaluation on the drone hover policy: tMSE, act_agree_pct vs the
expert mean action, concept accuracy, zero-ablation causal alignment
(rho_active / rho_inactive = paper's All Active / All Inactive in the CUB-style
formulation), the "correlated-not-causal" exhibit and the money table.

Centerpiece paper: Rountree, Verma, So, Fan, Shah. "Causal Concept Explanations
for Deep Neural Models." NeurIPS 2026. arXiv 2611.14407.

Estimator deviation (pre-declared): the A/B match estimator needs mutually
exclusive concept groups which our ontology lacks, so we use the paper's own
zero-ablation attribution (Sec 5.4, App D.3): delta_j = ||h(M) - h(M_j := 0)||,
rho_j = delta_j / sum_k delta_k.

Continuous adaptation: Yt = expert action A (N,3); "match" is replaced by
act_agree = ||yh - yb||_inf < 0.1; the money table is diverse-bucketed over the
aligned seed-0 test set (30 rows).
"""
import sys, json, argparse, time, csv
sys.path.insert(0, r'D:\fpv\ccw')
import numpy as np
import torch
import torch.nn.functional as F
import ccw_net as cn
from ccw_net import Encoder, Head, zero_ablation_deltas

torch.set_num_threads(6)

CONCEPTS = ["above_target", "below_target", "horizontal_error_large",
            "moving_away_from_target", "speed_high", "disturbance_active",
            "disturbance_pushes_north", "disturbance_pushes_east",
            "disturbance_pushes_down", "tilted"]


def load_model(ckpt_path):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    enc = Encoder(ck['J'])
    head = Head(ck['J'])
    enc.load_state_dict(ck['enc'])
    head.load_state_dict(ck['head'])
    enc.eval(); head.eval()
    return ck, enc, head


def fmt_vec(v):
    return "(%.2f,%.2f,%.2f)" % (float(v[0]), float(v[1]), float(v[2]))


def state_text(d, ti):
    return "ep%d st%d err=%s vel=%s rpy=%s f=%s act=%s" % (
        int(d['ep'][ti]), int(d['step_idx'][ti]), fmt_vec(d['err'][ti]),
        fmt_vec(d['vel'][ti]), fmt_vec(d['rpy'][ti]), fmt_vec(d['force'][ti]),
        fmt_vec(d['A'][ti]))


def bucket_of(lab10):
    """Diversity bucket for money-table sampling (first match wins)."""
    if lab10[5] > 0.5:
        return 'disturbance'
    if lab10[9] > 0.5:
        return 'tilted'
    if lab10[2] > 0.5:
        return 'horiz_err_large'
    if lab10[3] > 0.5:
        return 'moving_away'
    return 'plain_near_hover'


def eval_pair(tag, pt_a, pt_b, d, Zt, Yt, hp=None):
    t0 = time.time()
    out = {'tag': tag, 'wall_s': None}
    results = {}
    details = {}
    for name, pt in (('aligned', pt_a), ('noalign', pt_b)):
        ck, enc, head = load_model(pt)
        J = ck['J']
        keep_idx = ck['keep_idx']
        test_idx = torch.tensor(ck['test_idx'])
        C = torch.tensor(d['labels'][:, keep_idx].astype(np.float32))
        with torch.no_grad():
            M = torch.cat([enc(Zt[test_idx[st:st + 4096]]) for st in range(0, len(test_idx), 4096)])
            yh = head(M)
        yb = Yt[test_idx]
        tmse = float(((yh - yb) ** 2).mean())
        agree = float(((yh - yb).abs().max(dim=1).values < 0.1).float().mean() * 100)
        present = M.norm(dim=-1) > 0.5
        per_acc = ((present == (C[test_idx] > 0.5)).float().mean(0)).numpy()
        rho = zero_ablation_deltas(head, M)          # (Ntest, J) deltas
        s = rho.sum(-1, keepdim=True)
        rho = torch.where(s > 0, rho / s.clamp_min(1e-12),
                          torch.full_like(rho, 1.0 / J))
        act_mask = C[test_idx] > 0.5
        rho_active = (rho * act_mask).sum(-1)
        out_ = dict(tMSE=round(tmse, 5), act_agree_pct=round(agree, 2),
                    concept_acc_mean=round(float(per_acc.mean()), 4),
                    concept_acc_per={str(CONCEPTS[keep_idx[j]]): round(float(per_acc[j]), 4)
                                     for j in range(J)},
                    rho_active=round(float(rho_active.mean()), 4),
                    rho_inactive=round(float((1 - rho_active).mean()), 4))
        results[name] = out_
        details[name] = dict(M=M, yh=yh, rho=rho, rho_active=rho_active, agree=agree,
                             per_acc=per_acc, test_idx=test_idx.numpy(), head=head, ck=ck,
                             keep_idx=keep_idx)
        print(f"[eval {tag} {name}] tMSE={out_['tMSE']} agree={out_['act_agree_pct']}% "
              f"cacc={out_['concept_acc_mean']} rho_active={out_['rho_active']} "
              f"rho_inactive={out_['rho_inactive']} wall={time.time() - t0:.0f}s", flush=True)

    # per-concept rho table for model A (mean rho_j across test)
    A = details['aligned']
    ki = A['keep_idx']
    per_concept_rho = {str(CONCEPTS[ki[j]]): round(float(A['rho'][:, j].mean()), 4)
                       for j in range(len(ki))}
    # exhibit: model B (no-alignment) states where active concepts carry <50%
    # of the zero-ablation effect, yet the wrapper still agrees with the expert
    B = details['noalign']
    kb = B['keep_idx']
    Btest = B['test_idx']
    agree_mask = (B['yh'] - Yt[torch.tensor(Btest)]).abs().max(dim=1).values < 0.1
    cand = [int(i) for i in np.flatnonzero(agree_mask.numpy())
            if float(B['rho_active'][int(i)]) < 0.5]
    exhibit = []
    for i in cand[:5]:
        ti = int(Btest[i])
        exhibit.append(dict(model='noalign', state=state_text(d, ti),
                            expert_action=fmt_vec(d['A'][ti]),
                            rho_active=round(float(B['rho_active'][i]), 3),
                            rho_by_concept={str(CONCEPTS[kb[j]]):
                                            round(float(B['rho'][i, j]), 3)
                                            for j in range(len(kb))}))
    # money table: 30 DIVERSE rows from model A (aligned seed0) test set,
    # bucketed by which key concepts are active, up to ~6 per bucket.
    At = A['test_idx']
    buckets = {}
    money = []
    per_bucket = {}
    for i in range(len(At)):
        ti = int(At[i])
        b = bucket_of(d['labels'][ti])
        per_bucket[b] = per_bucket.get(b, 0) + 1
        if per_bucket[b] >= 6 or len(money) >= 30:
            continue
        order = np.argsort(-A['rho'][i].numpy())
        top3 = []
        for j in order[:3]:
            top3.append(dict(concept=str(CONCEPTS[ki[j]]),
                             delta=round(float(A['rho'][i, j]), 3),
                             concept_acc=round(float(A['per_acc'][j]), 4)))
        money.append(dict(model='aligned', row=f"{b}#{ti}", bucket=b,
                          state=state_text(d, ti),
                          action=fmt_vec(d['A'][ti]), top3=top3,
                          act_agree=bool(bool(((A['yh'][i] - Yt[torch.tensor([ti])]).abs().max() < 0.1)))))
        if len(money) >= 30:
            break
    out['metrics'] = results
    out['per_concept_rho_aligned'] = per_concept_rho
    out['exhibit_rows'] = exhibit
    out['money_table'] = money
    out['money_bucket_counts'] = per_bucket
    out['epochs_used'] = dict(aligned=len(details['aligned']['ck']['history']),
                              noalign=len(details['noalign']['ck']['history']))
    out['wall_min'] = dict(aligned=round(details['aligned']['ck']['wall_s'] / 60, 2),
                           noalign=round(details['noalign']['ck']['wall_s'] / 60, 2))
    out['wall_s'] = round(time.time() - t0, 1)
    del details
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--aligned', default=r'D:\fpv\ccw\ccw_aligned_s0.pt')
    ap.add_argument('--noalign', default=r'D:\fpv\ccw\ccw_noalign_s0.pt')
    ap.add_argument('--aligned2', default='')
    ap.add_argument('--noalign2', default='')
    ap.add_argument('--data', default=r'D:\fpv\ccw\ccw_data.npz')
    ap.add_argument('--out', default=r'D:\fpv\ccw\results_drone.json')
    ap.add_argument('--csv', default=r'D:\fpv\ccw\money_table_drone.csv')
    a = ap.parse_args()
    d = np.load(a.data, allow_pickle=False)
    Zt = torch.tensor(d['Z'])
    Yt = torch.tensor(d['A'].astype(np.float32))   # (N,3) expert mean actions
    res1 = eval_pair('seed0', a.aligned, a.noalign, d, Zt, Yt)
    allres = [res1]
    if a.aligned2:
        allres.append(eval_pair('seed1', a.aligned2, a.noalign2, d, Zt, Yt))
    # aggregate
    def avg(d1, d2, key):
        return round((d1[key] + d2[key]) / 2, 5)
    final = dict(data=dict(n_decisions=int(len(Zt)), n_episodes=int(d['ep'].max() + 1),
                           active_rates={n: round(float(d['labels'][:, i].mean()), 4)
                                         for i, n in enumerate(CONCEPTS)}),
                 runs=[r['metrics'] | dict(epochs_used=r['epochs_used'], wall_min=r['wall_min'],
                                           tag=r['tag']) for r in allres])
    if len(allres) == 2:
        final['metrics'] = {m: {k: avg(allres[0]['metrics'][m], allres[1]['metrics'][m], k)
                                for k in ('tMSE', 'act_agree_pct', 'concept_acc_mean',
                                          'rho_active', 'rho_inactive')}
                            for m in ('aligned', 'noalign')}
    else:
        final['metrics'] = allres[0]['metrics']
    final['per_concept_rho'] = allres[0]['per_concept_rho_aligned']
    final['exhibit_rows'] = allres[0]['exhibit_rows']
    final['money_table'] = allres[0]['money_table']
    final['n_seeds'] = len(allres)
    json.dump(final, open(a.out, 'w'), indent=1)
    with open(a.csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'row', 'state', 'action', 'concept1', 'delta1', 'acc1',
                    'concept2', 'delta2', 'acc2', 'concept3', 'delta3', 'acc3'])
        for row in final['money_table']:
            t = row['top3'] + [dict(concept='', delta='', concept_acc='')] * (3 - len(row['top3']))
            w.writerow([row['model'], row['row'], row['state'], row['action']] +
                       sum([[c['concept'], c['delta'], f"{float(c['concept_acc']):.4f}" if c['concept'] else '']
                            for c in t], []))
    print(json.dumps(final['metrics'], indent=1), flush=True)
    print(f"WROTE {a.out} and {a.csv}", flush=True)
    print("EVAL_DONE", flush=True)


if __name__ == '__main__':
    main()
