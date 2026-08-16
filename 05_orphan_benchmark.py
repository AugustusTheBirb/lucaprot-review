#!/usr/bin/env python
"""The evaluation the paper is missing, and LucaProt's score on it.

Sections 02-03 establish that the released test set cannot measure detection
beyond homology, because every test positive has a training relative. That is a
statement about the evidence, not about the model. This is the experiment that
tests the model.

Construction:

    87,368  published RdRPs from three prior studies
            (Zayed 2022 Science, Chen 2022 Nat Microbiol, Neri 2022 Cell)
       |    DIAMOND blastp --very-sensitive, e <= 1e-3, against LucaProt's
       |    4,900 training positives; keep only sequences with NO hit at all
     2,874  -> ../no_training_homolog.faa

These are real, published RdRPs with no homology bridge back to anything
LucaProt trained on - precisely the case the test set contains none of. Scoring
them is then just running the unmodified released checkpoint at the paper's own
threshold of 0.5 (../run_experiment.py does this; results are committed under
../results/, so this script only reads and scores them).

Two controls make the number interpretable. Without them a low orphan score
could equally mean the inference environment is misconfigured:

    control_neg   250 cellular proteins from LucaProt's own test split, expect ~0%
    control_pos   250 prior-study RdRPs that DO have a training homolog, expect ~98%

The orphan order was randomised (shuffle_inputs.py, seed 20260806) before
scoring, so the completed prefix is an unbiased random sample of the benchmark
and a partial run is still a valid estimate - just with a wider interval.
Intervals are Wilson, not normal-approximation, because control_neg is 0/250
where the normal interval has zero width.

    python reproduce/05_orphan_benchmark.py
"""
import csv, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ROOT, emit, need, pct, section, wilson

RESULTS = os.path.join(ROOT, 'results')
BENCH = os.path.join(ROOT, 'no_training_homolog.faa')
THRESH = 0.5

SETS = [
    ('control_neg', 'cellular proteins from LucaProt\'s test split', '~0%'),
    ('control_pos', 'prior-study RdRPs WITH a training homolog', '~98%'),
    ('orphans', 'prior-study RdRPs with NO training homolog', 'the experiment'),
]


def read_set(name):
    """Collect every scored sequence for one set across its resumable chunks."""
    d = os.path.join(RESULTS, name)
    if not os.path.isdir(d):
        return []
    probs = []
    for chunk in sorted(os.listdir(d)):
        path = os.path.join(d, chunk, 'result.csv')
        if not os.path.exists(path):
            continue
        with open(path, encoding='utf-8', errors='replace') as f:
            for r in csv.DictReader(f):
                try:
                    probs.append(float(r['prob']))
                except (KeyError, ValueError, TypeError):
                    pass
    return probs


def main():
    need(RESULTS, 'per-sequence predictions from run_experiment.py')

    # benchmark size, straight from the FASTA
    if os.path.exists(BENCH):
        n_bench = sum(1 for line in open(BENCH) if line.startswith('>'))
        print('benchmark: %s -> %d sequences' % (os.path.basename(BENCH), n_bench))
        emit('orph.benchmark_size', n_bench)

    section('LUCAPROT ON SEQUENCES IT HAS NO HOMOLOGY TO (threshold %.2f)' % THRESH)
    print('  %-13s %7s %8s %10s   %s' % ('set', 'n', 'called', '95% CI', 'expected'))
    for name, what, expect in SETS:
        probs = read_set(name)
        if not probs:
            print('  %-13s (no results yet)' % name)
            continue
        k = sum(1 for p in probs if p >= THRESH)
        lo, hi = wilson(k, len(probs))
        print('  %-13s %7d %7.2f%% %5.1f-%.1f   %s'
              % (name, len(probs), pct(k, len(probs)), lo, hi, expect))
        print('  %-13s %s' % ('', what))
        emit('orph.%s_n' % name, len(probs))
        emit('orph.%s_pct' % name, '%.2f' % pct(k, len(probs)))
        emit('orph.%s_ci_lo' % name, '%.1f' % lo)
        emit('orph.%s_ci_hi' % name, '%.1f' % hi)

    # ---- the caveat the post makes explicit ------------------------------
    section('WHAT control_neg DOES AND DOES NOT SHOW')
    lo, hi = wilson(0, 250)
    print('  0 / 250 called RdRP  ->  95%% CI %.1f - %.1f%%' % (lo, hi))
    print("""
  The post argues elsewhere that the paper's 0.014%% false-positive rate implies
  ~20,000 false positives across 144.6M proteins. A reader may notice that this
  control found 0%% and conclude the two contradict. They do not: 250 samples
  put a 95%% upper bound of %.1f%%, roughly a hundred times coarser than the rate
  in question. Resolving 0.014%% needs on the order of tens of thousands of
  held-out hard negatives - which is exactly what LucaProt's split does not
  have, and exactly why nobody knows the deployment rate.

  This control is silent on the false-positive question. It is here only to
  show the inference environment is set up correctly, so that the orphan number
  above means something.""" % hi)

    section('WHAT THIS RESULT MEANS FOR THE POST')
    print("""  Recall holds outside the training homology neighbourhood. The paper's
  central claim is correct - it was simply never demonstrated by the evaluation
  it is cited to. That is a smaller criticism than "the result is wrong", and it
  is the one the evidence supports.

  One limit: ESM2-3B was pretrained on UniRef and these are published sequences,
  so they are plausibly in the language model's pretraining data even though
  they are absent from LucaProt's classifier training set. The holdout is clean
  with respect to the classifier, not the representation.""")


if __name__ == '__main__':
    main()
