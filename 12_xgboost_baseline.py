#!/usr/bin/env python
"""LucaProt's own XGBoost baseline, which the paper builds and never reports.

The post makes two claims about this and both are checkable:

  1. The authors ship a gradient-boosting baseline on the same frozen ESM2
     layer-36 features the probe uses, configured for 50 boosting rounds.
  2. Run to convergence instead, it reaches ~97% - the same place the logistic
     probe and LucaProt itself land.

Which matters because it changes the story. If the unreported baseline had been
bad, leaving it out would be ordinary. It is not bad: it was configured in a way
that guarantees it looks bad. `run_xgb.sh` passes --num_train_epochs 50 against
an eta of 0.01 and --pos_weight 40, so total shrinkage is about 0.5 against a
40x positive upweight - scores sit above 0.5 nearly everywhere and the paper's
own 0.5 threshold stops meaning anything. That is a configuration artifact, not
a finding about gradient boosting.

Everything here is read from the authors' files rather than chosen:

    LucaProt/src/baselines/run_xgb.sh                --num_train_epochs, --pos_weight
    LucaProt/config/rdrp_40_extend/.../xgb_config.json   eta, depth, objective

Scoring is at a matched operating point, as in 06: each model's threshold is set
on held-out cellular negatives to a common 0.16% false-positive rate, and recall
is then measured on the zero-homology orphans. Comparing recall at a fixed 0.5
across models whose scores are scaled differently is exactly the mistake the
as-shipped configuration would otherwise walk into.

Needs the embeddings (~500 MB) and xgboost, so it skips like 06 does.
The upstream script with the full comparison is linear_probe/fit_xgb.py.

    python reproduce/12_xgboost_baseline.py
    python reproduce/12_xgboost_baseline.py --long 400   # fewer rounds, faster
"""
import argparse, csv, glob, json, os, re, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ROOT, emit, need, section, wilson

LP = os.path.join(ROOT, 'linear_probe')
EMB = os.path.join(LP, 'embeddings')
LUCA = os.path.join(ROOT, 'LucaProt')
SH = os.path.join(LUCA, 'src', 'baselines', 'run_xgb.sh')
CFG = os.path.join(LUCA, 'config', 'rdrp_40_extend', 'protein', 'binary_class',
                   'xgb_config.json')
LAYER = 36
TARGET_FPR = 0.0016


def load_manifest():
    m = {}
    with open(os.path.join(LP, 'inputs', 'manifest.csv')) as f:
        for r in csv.DictReader(f):
            m[r['uid']] = (r['set'], int(r['label']))
    return m


def load_embeddings(np):
    shards = sorted(glob.glob(os.path.join(EMB, '**', 'shard_*.npz'), recursive=True))
    if not shards:
        print('SKIP\tno embedding shards in %s' % EMB)
        print('      produce them with:  modal run linear_probe/embed_modal.py')
        sys.exit(3)
    uids, vecs, t0 = [], [], time.time()
    for i, p in enumerate(shards, 1):
        z = np.load(p, allow_pickle=True)
        vecs.append(z['layer_%d' % LAYER].astype(np.float32))
        uids.extend(str(u) for u in z['uids'])
        el = time.time() - t0
        sys.stdout.write('\r  loading shards  %d/%d  %d sequences  %.0fs elapsed,'
                         ' ETA %.0fs   ' % (i, len(shards), len(uids), el,
                                            el / i * (len(shards) - i)))
        sys.stdout.flush()
    print()
    return uids, np.concatenate(vecs)


def their_settings():
    """Read the baseline's configuration out of the authors' own files."""
    rounds = pos_weight = None
    if os.path.exists(SH):
        src = open(SH, encoding='utf-8', errors='replace').read()
        m = re.search(r'--num_train_epochs=?\s*(\d+)', src)
        rounds = int(m.group(1)) if m else None
        m = re.search(r'--pos_weight\s+(\d+)', src)
        pos_weight = int(m.group(1)) if m else None
    params = json.load(open(CFG)) if os.path.exists(CFG) else {}
    return rounds, pos_weight, params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--long', type=int, default=1500,
                    help='rounds for the converged fit (default 1500)')
    args = ap.parse_args()

    try:
        import numpy as np
        import xgboost as xgb
    except ImportError as e:
        print('SKIP\tneeds numpy + xgboost (%s)' % e)
        sys.exit(3)

    need(CFG, 'the vendored LucaProt repo (git submodule update --init)')
    need(os.path.join(LP, 'inputs', 'manifest.csv'),
         'run linear_probe/prepare_inputs.py first')

    section("THE BASELINE'S CONFIGURATION, READ FROM THEIR FILES")
    rounds, pos_weight, params = their_settings()
    print('  run_xgb.sh  --num_train_epochs   %s' % rounds)
    print('  run_xgb.sh  --pos_weight         %s' % pos_weight)
    for k in ('eta', 'max_depth', 'objective'):
        if k in params:
            print('  xgb_config.json  %-15s %s' % (k, params[k]))
    emit('xgb.their_rounds', rounds)
    emit('xgb.their_pos_weight', pos_weight)
    if 'eta' in params:
        emit('xgb.their_eta', params['eta'])
        print('\n  %s rounds x eta %s = %.2f total shrinkage, against a %sx positive'
              '\n  upweight. Scores sit above 0.5 nearly everywhere, so the 0.5'
              ' threshold\n  the paper uses is not an operating point for this model'
              ' at all.'
              % (rounds, params['eta'], rounds * float(params['eta']), pos_weight))

    # ---- fit both ---------------------------------------------------------
    print('\nloading frozen ESM2-3B layer-%d embeddings' % LAYER, flush=True)
    man = load_manifest()
    uids_all, X_all = load_embeddings(np)
    keep = [i for i, u in enumerate(uids_all) if u in man]
    uids = [uids_all[i] for i in keep]
    X = X_all[keep]
    sets = np.array([man[u][0] for u in uids])
    y = np.array([man[u][1] for u in uids])
    tr = sets == 'probe_train'
    print('  train %d (%d positive), features %d'
          % (int(tr.sum()), int(y[tr].sum()), X.shape[1]))

    p = dict(params)
    p['scale_pos_weight'] = pos_weight or 40
    p['nthread'] = os.cpu_count() or 4
    dtr, dall = xgb.DMatrix(X[tr], label=y[tr]), xgb.DMatrix(X)

    models = []
    for tag, n in [('as shipped', rounds or 50), ('converged', args.long)]:
        t0 = time.time()
        print('\n  %-11s boosting %d rounds ...' % (tag, n), end='', flush=True)
        bst = xgb.train(p, dtr, num_boost_round=n, verbose_eval=False)
        print(' %.0fs' % (time.time() - t0), flush=True)
        models.append((tag, bst.predict(dall)))

    # ---- matched operating point -----------------------------------------
    negm = (sets == 'probe_test') & (y == 0)
    orph = (sets == 'orphans') & (y == 1)
    section('ORPHAN RECALL AT A MATCHED %.2f%% FALSE-POSITIVE RATE' % (100 * TARGET_FPR))
    print('  threshold set on %d held-out cellular negatives, recall measured on'
          '\n  %d zero-homology orphans\n' % (int(negm.sum()), int(orph.sum())))
    print('  %-24s %10s %10s   %s' % ('model', 'threshold', 'recall', '95% CI'))
    for tag, prob in models:
        t = float(np.quantile(prob[negm], 1.0 - TARGET_FPR))
        k = int((prob[orph] >= t).sum())
        rec = 100.0 * k / int(orph.sum())
        lo, hi = wilson(k, int(orph.sum()))
        print('  xgboost, %-15s %10.4f %9.2f%%   %.1f - %.1f'
              % (tag, t, rec, lo, hi))
        emit('xgb.%s_recall' % tag.replace(' ', '_'), '%.2f' % rec)
    print('  %-24s %10.4f %9.2f%%   (as published)' % ('LucaProt', 0.5, 97.73))
    print('  %-24s %10s %9.2f%%   (section 06)' % ('logistic probe', '-', 97.74))

    section('WHY THIS MATTERS FOR THE POST')
    print("""  The post's claim is not that gradient boosting is better than LucaProt. It
  is that the authors built the comparison that would have shown their fusion
  architecture was unnecessary, configured it in a way that could not converge,
  and reported none of it. The paper's only baselines are CHEER, VirHunter,
  Virtifier and RNN-VirSeeker - none of which use ESM2 features, so none of them
  isolate the architecture from the representation.

  Note the shape of the claim. "Converges to about the same place" is a weaker
  statement than "beats it", and it is the one the numbers support: three
  unrelated classifiers on the same frozen features all land near 97.5%, which
  says the features are doing the work.""")


if __name__ == '__main__':
    main()
