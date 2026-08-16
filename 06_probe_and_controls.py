#!/usr/bin/env python
"""Does LucaProt's architecture beat a logistic regression on the same features?

This is the second half of the post. 05 shows LucaProt's recall holds on
sequences it has no homology to. This asks whether the 23.7M-parameter fusion
network is why.

The comparison is deliberately unfair to the probe. It gets frozen ESM2-3B
layer-36 embeddings, mean-pooled over residues - which throws away sequence
order entirely - and one linear layer on top. 2,561 parameters against 23.7M,
and no access to the from-scratch transformer channel at all.

Everything is scored at a MATCHED OPERATING POINT, which matters more than it
sounds. Recall at a fixed 0.5 threshold is meaningless across models whose
scores are scaled differently: an undertrained model whose scores sit above 0.5
everywhere scores 100% recall while firing on everything. So the threshold for
every model below is set to whatever gives a 0.16% false-positive rate on the
3,160 held-out negatives in probe_test - a set no model here trained on, and
disjoint from the orphans whose recall is being measured.

Four controls, because "a linear model matches the 23.7M network" is the kind of
claim that is usually a bug:

    composition       21 hand-made features, no ESM2   -> is the benchmark trivial?
    shuffled labels   real features, labels permuted   -> is the pipeline leaking?
    log-length        one feature                      -> is it a length shortcut?
    embedding L2 norm one feature                      -> is it a norm shortcut?

If any control scores well, the headline result means nothing.

Requires numpy + scikit-learn, the embeddings (linear_probe/embeddings/,
~500 MB, produced by linear_probe/embed_modal.py on a GPU) and the manifest.
This is the only script here that is not stdlib-only and not instant; budget a
few minutes. The upstream scripts it condenses are linear_probe/fit_probe.py,
sanity_controls.py, length_check.py and weight_analysis.py, which report more
than the post uses.

    python reproduce/06_probe_and_controls.py
    python reproduce/06_probe_and_controls.py --quick   # skip the L1 sweep
"""
import argparse, csv, glob, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ROOT, emit, need, section

LP = os.path.join(ROOT, 'linear_probe')
EMB = os.path.join(LP, 'embeddings')
LAYER = 36
TARGET_FPR = 0.0016      # 0.16%, the operating point used throughout the post
AA = 'ACDEFGHIKLMNPQRSTVWY'


def load_manifest():
    m = {}
    with open(os.path.join(LP, 'inputs', 'manifest.csv')) as f:
        for r in csv.DictReader(f):
            m[r['uid']] = (r['set'], int(r['label']), r['source'], int(r['seq_len']))
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
        key = 'layer_%d' % LAYER
        if key not in z:
            sys.exit('shard %s has no %s' % (os.path.basename(p), key))
        vecs.append(z[key].astype(np.float32))
        uids.extend(str(u) for u in z['uids'])
        el = time.time() - t0
        sys.stdout.write('\r  loading shards  %d/%d  %d sequences  %.0fs elapsed,'
                         ' ETA %.0fs   ' % (i, len(shards), len(uids), el,
                                            el / i * (len(shards) - i)))
        sys.stdout.flush()
    print()
    return uids, np.concatenate(vecs)


def read_fasta_seqs(path):
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
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true',
                    help='skip the L1 dimension sweep (the slow part)')
    args = ap.parse_args()

    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import roc_auc_score
    except ImportError as e:
        print('SKIP\tneeds numpy + scikit-learn (%s)' % e)
        sys.exit(3)

    need(os.path.join(LP, 'inputs', 'manifest.csv'),
         'run linear_probe/prepare_inputs.py first')

    print('loading frozen ESM2-3B layer-%d embeddings' % LAYER, flush=True)
    man = load_manifest()
    uids, X = load_embeddings(np)

    keep = [i for i, u in enumerate(uids) if u in man]
    X = X[keep]
    uids = [uids[i] for i in keep]
    sets = np.array([man[u][0] for u in uids])
    y = np.array([man[u][1] for u in uids])
    ln = np.array([man[u][3] for u in uids], dtype=np.float64)

    trm = sets == 'probe_train'
    tem = sets == 'probe_test'
    orm = (sets == 'orphans') & (y == 1)
    print('  %d sequences: %d train, %d test, %d orphans'
          % (len(uids), trm.sum(), tem.sum(), orm.sum()), flush=True)
    emit('probe.dims', X.shape[1])
    emit('probe.params', X.shape[1] + 1)

    def evaluate(tag, prob, cid=None):
        """Threshold on held-out probe_test negatives, then score orphans."""
        negm = tem & (y == 0)
        t = float(np.quantile(prob[negm], 1.0 - TARGET_FPR))
        k, n = int((prob[orm] >= t).sum()), int(orm.sum())
        r = 100.0 * k / n
        print('  %-34s %6.2f%%   [threshold %.4f]' % (tag, r, t))
        if cid:
            emit(cid, '%.2f' % r)
        return r

    # ---- the probe -------------------------------------------------------
    section('THE PROBE, AND THE CONTROLS, ALL AT 0.16%% FPR')
    print('  %-34s %7s' % ('what the classifier is given', 'recall'))

    # class_weight='balanced' throughout: probe_train is 4,900 positives against
    # 25,274 negatives, and every upstream script in linear_probe/ uses it. The
    # numbers below do not reproduce without it.
    def logit(Xf, yf, C=1.0, penalty='l2'):
        return LogisticRegression(max_iter=5000, C=C, penalty=penalty,
                                  class_weight='balanced',
                                  solver='liblinear' if penalty == 'l1'
                                  else 'lbfgs').fit(Xf, yf)

    sc = StandardScaler().fit(X[trm])
    Xs = sc.transform(X)
    print('  fitting logistic regression on %d x %d ...' % (trm.sum(), X.shape[1]),
          end='', flush=True)
    t0 = time.time()
    clf = logit(Xs[trm], y[trm])
    print(' %.0fs' % (time.time() - t0), flush=True)
    p_real = clf.predict_proba(Xs)[:, 1]

    # ---- control 1: composition -----------------------------------------
    seqs = {}
    for f in ['probe_train', 'probe_test', 'orphans']:
        p = os.path.join(LP, 'inputs', f + '.faa')
        if os.path.exists(p):
            seqs.update(read_fasta_seqs(p))
    C = np.zeros((len(uids), len(AA) + 1), dtype=np.float32)
    for i, u in enumerate(uids):
        s = seqs.get(u, '')
        n = max(1, len(s))
        for j, a in enumerate(AA):
            C[i, j] = s.count(a) / n
        C[i, -1] = np.log10(max(1, len(s)))
    cs = StandardScaler().fit(C[trm])
    p_comp = logit(cs.transform(C[trm]), y[trm]).predict_proba(
        cs.transform(C))[:, 1]

    # ---- control 2: shuffled labels -------------------------------------
    # Ten permutations, and the WORST case (highest recall) is reported. Taking
    # the best of ten is the conservative direction for a control: it makes the
    # control look as strong as possible, so a low number means more.
    negm_ = tem & (y == 0)
    best_shuf, p_shuf = -1.0, None
    for seed in range(10):
        rng = np.random.default_rng(seed)
        ysh = y.copy()
        idx = np.where(trm)[0]
        ysh[idx] = rng.permutation(y[idx])
        ps = logit(Xs[trm], ysh[trm]).predict_proba(Xs)[:, 1]
        t = float(np.quantile(ps[negm_], 1.0 - TARGET_FPR))
        r = 100.0 * float((ps[orm] >= t).mean())
        if r > best_shuf:
            best_shuf, p_shuf = r, ps
        sys.stdout.write('\r  shuffled-label control  seed %d/10  worst so far'
                         ' %.2f%%   ' % (seed + 1, best_shuf))
        sys.stdout.flush()
    print()

    # ---- controls 3 and 4: single features ------------------------------
    def one_feature(v):
        f = v.reshape(-1, 1)
        s = StandardScaler().fit(f[trm])
        return logit(s.transform(f[trm]), y[trm]).predict_proba(
            s.transform(f))[:, 1]

    p_len = one_feature(np.log10(ln))
    p_nrm = one_feature(np.linalg.norm(X, axis=1))

    evaluate('amino-acid composition (21 feat)', p_comp, 'ctrl.composition')
    evaluate('ESM2 features, labels shuffled', p_shuf, 'ctrl.shuffled')
    evaluate('log-length alone', p_len, 'ctrl.length')
    evaluate('embedding L2 norm alone', p_nrm, 'ctrl.l2norm')
    evaluate('ESM2 layer 36, real labels', p_real, 'probe.orphan_recall')
    print('  %-34s %6.2f%%   [reported in the paper / measured in 05]'
          % ('LucaProt, 23.7M parameters', 97.73))

    # ---- the separate length finding ------------------------------------
    section('IS LUCAPROT\'S TEST SET SEPARABLE ON LENGTH ALONE?')
    tp, tn = tem & (y == 1), tem & (y == 0)
    auc = roc_auc_score(y[tem], np.log10(ln[tem]))
    print('  median length   positives %5.0f aa   negatives %5.0f aa'
          % (np.median(ln[tp]), np.median(ln[tn])))
    print('  AUC of log-length alone on probe_test   %.4f' % auc)
    print('  median orphan length                    %5.0f aa' % np.median(ln[orm]))
    emit('len.median_pos', '%.0f' % np.median(ln[tp]))
    emit('len.median_neg', '%.0f' % np.median(ln[tn]))
    emit('len.auc_probe_test', '%.4f' % auc)
    print("""
  Length alone ranks the test set at %.4f without looking at the sequence. Note
  the contrast with the control above: the same feature gets ~0%% on the orphans.
  Both are true and they say different things - the test set is separable on a
  feature carrying no biological claim, while the orphan benchmark is not.""" % auc)

    # ---- how few dimensions are needed -----------------------------------
    if args.quick:
        print('\n(skipping the L1 sweep; drop --quick to run it)')
        return

    section('HOW MANY OF ESM2\'S 2,560 DIMENSIONS ARE ACTUALLY NEEDED')
    order = np.argsort(-np.abs(clf.coef_.ravel()))
    for k in [5, 10]:
        cols = order[:k]
        pk = logit(Xs[trm][:, cols], y[trm]).predict_proba(Xs[:, cols])[:, 1]
        evaluate('top %d dims by |weight|' % k, pk, 'dims.top%d' % k)

    print('  fitting L1 (C=0.0003) ...', end='', flush=True)
    t0 = time.time()
    l1 = logit(Xs[trm], y[trm], C=0.0003, penalty='l1')
    nz = int((l1.coef_.ravel() != 0).sum())
    print(' %.0fs -> %d nonzero dimensions' % (time.time() - t0, nz), flush=True)
    evaluate('%d dims selected by L1' % nz, l1.predict_proba(Xs)[:, 1],
             'dims.l1_recall')
    emit('dims.l1_count', nz)
    print("""
  A sharp phase transition: essentially nothing at 5 dimensions, most of the
  performance by ~14. RdRP identity is not in any single direction, but it lives
  in a very low-dimensional subspace of ESM2's representation.

  (L1 selection is mildly sensitive to array memory layout and sklearn version -
  it lands on 13 or 14 dimensions depending on both. The post says 14. If this
  prints 13, that is the known wobble, not a discrepancy.)""")


if __name__ == '__main__':
    main()
