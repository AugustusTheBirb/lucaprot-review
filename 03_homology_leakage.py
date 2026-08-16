#!/usr/bin/env python
"""How close every test positive is to its nearest training positive.

This is the core measurement behind the post's central claim about the
evaluation. The argument runs:

  - the split was made randomly, without regard to sequence similarity
  - therefore test positives may have near-relatives in training
  - if *every* test positive does, the test set cannot distinguish "detects
    RdRPs" from "recognises things similar to what it was trained on"
  - and in fact every one does

The alignment itself is DIAMOND blastp, run once and committed as
../analysis/test_vs_train.tsv, because DIAMOND is a compiled binary that not
every reader will have. The command that produced it:

    diamond makedb --in analysis/train_pos.faa -d train_pos
    diamond blastp -q test_pos.faa -d train_pos --ultra-sensitive \\
        --outfmt 6 qseqid sseqid pident length qlen slen bitscore evalue \\
        -o analysis/test_vs_train.tsv

`--ultra-sensitive` rather than defaults. Tuning a baseline is legitimate, but
the post says so explicitly, and so does this docstring. Whether 100% survives
at default sensitivity is untested.

Identity is reported over the *full query length* - pident * length / qlen,
capped at 100 - not over the aligned region. A 40-residue alignment at 95%
identity is not a 95%-identical protein, and using the raw pident column would
inflate every number in the table below.

    python reproduce/03_homology_leakage.py
"""
import os, statistics, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ROOT, emit, need, pct, section

TSV = os.path.join(ROOT, 'analysis', 'test_vs_train.tsv')
N_TEST_POS = 540          # from 02; hard-coded here so 03 runs standalone


def main():
    need(TSV, 'DIAMOND output of the 540 test positives vs the 4,900 training '
              'positives; committed, so this runs without DIAMOND installed')

    best = {}
    with open(TSV) as f:
        for line in f:
            q, s, pid, alen, qlen, slen, bits, ev = line.rstrip('\n').split('\t')
            pid, alen, qlen = float(pid), int(alen), int(qlen)
            bits, ev = float(bits), float(ev)
            gid = min(100.0, pid * alen / qlen)     # identity over the full query
            if q not in best or gid > best[q][0]:
                best[q] = (gid, pid, bits, ev)

    section('DOES ANY TEST POSITIVE LACK A TRAINING HOMOLOG?')
    print('  test positives with a detectable training homolog : %d / %d'
          % (len(best), N_TEST_POS))
    print('  test positives with none                          : %d / %d'
          % (N_TEST_POS - len(best), N_TEST_POS))
    emit('leak.with_homolog', len(best))
    emit('leak.without_homolog', N_TEST_POS - len(best))

    section('E-VALUE OF THE BEST HIT')
    for cut, lab in [(1e-3, '1e-3'), (1e-10, '1e-10'), (1e-30, '1e-30'),
                     (1e-50, '1e-50'), (1e-100, '1e-100')]:
        n = sum(1 for v in best.values() if v[3] <= cut)
        print('  e <= %-8s %3d / %d  (%5.1f%%)' % (lab, n, N_TEST_POS,
                                                   pct(n, N_TEST_POS)))
        emit('leak.e%s' % lab.replace('1e-', ''), n)
    n0 = sum(1 for v in best.values() if v[3] == 0.0)
    print('  e == 0.0     %3d / %d  (%5.1f%%)' % (n0, N_TEST_POS,
                                                  pct(n0, N_TEST_POS)))
    emit('leak.e_exactly_zero', n0)
    emit('leak.e_exactly_zero_pct', '%.1f' % pct(n0, N_TEST_POS))

    ev = sorted(v[3] for v in best.values())
    bits = sorted(v[2] for v in best.values())
    print('\n  worst single test positive : e = %.1e' % ev[-1])
    print('  median e-value             : %.3g' % ev[len(ev) // 2])
    print('  min / median bitscore      : %.0f / %.0f' % (bits[0],
                                                          bits[len(bits) // 2]))
    emit('leak.worst_evalue', '%.1e' % ev[-1])

    section('IDENTITY TO THE NEAREST TRAINING NEIGHBOUR')
    gids = sorted(v[0] for v in best.values())
    print('  median %.1f%%   mean %.1f%%   min %.1f%%'
          % (statistics.median(gids), statistics.mean(gids), gids[0]))
    emit('leak.median_identity', '%.1f' % statistics.median(gids))

    print('\n  %-12s %6s %8s' % ('threshold', 'count', 'share'))
    for t in [100, 99, 95, 90, 80, 70, 50, 30]:
        n = sum(1 for g in gids if g >= t)
        print('  >= %3d%%      %6d %7.1f%%' % (t, n, pct(n, N_TEST_POS)))
        if t in (90, 30):
            emit('leak.ge%d_pct' % t, '%.1f' % pct(n, N_TEST_POS))

    # ---- the baseline ----------------------------------------------------
    section('THE HOMOLOGY BASELINE')
    recall = pct(sum(1 for v in best.values() if v[3] <= 1e-3), N_TEST_POS)
    print('  DIAMOND alone, e <= 1e-3, no machine learning : %.1f%% recall'
          % recall)
    print('  LucaProt, same test set                       : %.1f%% (539/540)'
          % pct(539, 540))
    emit('baseline.diamond_recall_pct', '%.1f' % recall)
    print("""
  The post claims recall only, not precision. Counting this baseline's false
  positives needs DIAMOND run over the 21,744 test negatives as well, and that
  output (analysis/testall_vs_trainpos.tsv) is not committed - so
  analysis/baseline.py and analysis/sweep.py cannot run here. Recall needs only
  the positives, which the committed TSV covers in full, so the 100% figure is
  solid and the precision figure is simply absent.""")


if __name__ == '__main__':
    main()
