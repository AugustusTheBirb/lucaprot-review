#!/usr/bin/env python
"""The third independent reason LucaProt's test set is uninformative.

03 shows the test set is separable by homology. 06 shows it is separable by
length. This shows it is separable by dipeptide statistics: 400 hand-made
features counting adjacent residue pairs, fed to an off-the-shelf gradient
booster.

    probe_test AUC   0.9966      near-perfect on the benchmark the paper validates against
    orphan recall    3.76%       finds essentially none of the actual novel RdRPs

That gap is the whole point. A model can look excellent on LucaProt's test set
while being useless at the task the test set is cited as evidence for. Three
unrelated shortcuts - homology, length, dipeptide composition - all clear it,
so clearing it is not evidence of anything in particular.

The features are raw counts, not frequencies, which means they encode length
implicitly (they sum to it). That is deliberate: the comparison against the
frequency version isolates how much of the AUC is length, and the full sweep in
../composition_baseline/fit_composition.py runs both plus three other feature
sets. Only the row the post quotes is recomputed here.

Needs numpy, scikit-learn and xgboost, plus the manifest and input FASTAs.
No embeddings and no GPU - this is the cheap experiment.

    python reproduce/08_composition_baseline.py
"""
import csv, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ROOT, emit, need, section

LP = os.path.join(ROOT, 'linear_probe')
AA = 'ACDEFGHIKLMNPQRSTVWY'
AIDX = {a: i for i, a in enumerate(AA)}
TARGET_FPR = 0.0016
ROUNDS = 400
SEED = 20260810


def read_fasta(path):
    out, h, buf = {}, None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if h:
                    out[h] = ''.join(buf)
                h, buf = line[1:], []
            elif line:
                buf.append(line)
    if h:
        out[h] = ''.join(buf)
    return out


def main():
    try:
        import numpy as np
        import xgboost as xgb
        from sklearn.metrics import roc_auc_score
    except ImportError as e:
        print('SKIP\tneeds numpy + scikit-learn + xgboost (%s)' % e)
        sys.exit(3)

    need(os.path.join(LP, 'inputs', 'manifest.csv'),
         'run linear_probe/prepare_inputs.py first')

    man, seqs = {}, {}
    with open(os.path.join(LP, 'inputs', 'manifest.csv')) as f:
        for r in csv.DictReader(f):
            man[r['uid']] = (r['set'], int(r['label']))
    for name in ['probe_train', 'probe_test', 'orphans', 'control_neg',
                 'control_pos']:
        p = os.path.join(LP, 'inputs', name + '.faa')
        if os.path.exists(p):
            seqs.update(read_fasta(p))

    uids = [u for u in man if u in seqs]
    uids.sort()          # xgboost subsamples rows, so row order changes the fit
    sets = np.array([man[u][0] for u in uids])
    y = np.array([man[u][1] for u in uids])
    tr = sets == 'probe_train'
    tm = sets == 'probe_test'
    neg = tm & (y == 0)
    orph = (sets == 'orphans') & (y == 1)
    print('%d sequences: %d train, %d test, %d orphans'
          % (len(uids), tr.sum(), tm.sum(), orph.sum()), flush=True)

    # ---- 400 dipeptide counts --------------------------------------------
    print('counting dipeptides ...', flush=True)
    X = np.zeros((len(uids), 400), dtype=np.float32)
    t0 = time.time()
    for i, u in enumerate(uids):
        s = seqs[u]
        prev = AIDX.get(s[0]) if s else None
        for ch in s[1:]:
            cur = AIDX.get(ch)
            if prev is not None and cur is not None:
                X[i, prev * 20 + cur] += 1
            prev = cur
        if (i + 1) % 5000 == 0:
            el = time.time() - t0
            sys.stdout.write('\r  %d/%d  %.0fs elapsed, ETA %.0fs   '
                             % (i + 1, len(uids), el,
                                el / (i + 1) * (len(uids) - i - 1)))
            sys.stdout.flush()
    print('\r  %d/%d done in %.0fs                  ' % (len(uids), len(uids),
                                                         time.time() - t0))

    # ---- gradient booster ------------------------------------------------
    section('DIPEPTIDE COUNTS + XGBOOST')
    pw = float((y[tr] == 0).sum()) / max(1, int((y[tr] == 1).sum()))
    params = {'objective': 'binary:logistic', 'max_depth': 6, 'eta': 0.1,
              'subsample': 0.8, 'colsample_bytree': 0.8, 'scale_pos_weight': pw,
              'nthread': os.cpu_count() or 4, 'eval_metric': 'auc',
              'seed': SEED}
    print('  training %d rounds ...' % ROUNDS, end='', flush=True)
    t0 = time.time()
    bst = xgb.train(params, xgb.DMatrix(X[tr], label=y[tr]),
                    num_boost_round=ROUNDS, verbose_eval=False)
    print(' %.0fs' % (time.time() - t0), flush=True)

    prob = bst.predict(xgb.DMatrix(X))
    auc = roc_auc_score(y[tm], prob[tm])
    t = float(np.quantile(prob[neg], 1.0 - TARGET_FPR))
    recall = 100.0 * float((prob[orph] >= t).mean())

    print('\n  AUC on LucaProt\'s test set   %.4f' % auc)
    print('  orphan recall at 0.16%% FPR   %.2f%%' % recall)
    print('  ESM2 probe, same protocol    97.74%%')
    emit('comp.dipep_auc', '%.4f' % auc)
    emit('comp.dipep_orphan_recall', '%.2f' % recall)

    print("""
  400 hand-made features and a gradient booster look near-perfect on the
  benchmark the paper validates against, and find almost nothing real. Taken
  with 03 (homology) and 06 (length), that is three independent ways to score
  well on LucaProt's test set without doing the task. Any one would undercut
  Figure S2; together they mean the test number carries almost no information
  about novel-virus discovery.""")


if __name__ == '__main__':
    main()
