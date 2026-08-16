#!/usr/bin/env python
"""Where the training data actually went, and how much of the test set is a copy.

Two claims in the post come from here.

1. "All 10,000 sequences labelled `rt` and all 2,000 labelled `dna` are in
   training, and none are in test." Every released row carries a `source` field,
   so this is a group-by, not an inference.

2. "30 of 540 test positives are byte-identical to a sequence in the training
   set." Pure string comparison after applying LucaProt's own clean_seq, so
   there is no aligner involved and no similarity threshold to argue about.
   This is the claim that needs no interpretation at all: those sequences were
   memorised, not generalised to.

Note the wording the post uses. Every row *labelled* rt is in training - which
is exactly true - but the test split is not literally free of reverse
transcriptases, because ~50 were already present under the generic `non_virus`
label before the hard negatives were appended (see 04). This script counts both
so the distinction stays visible.

Input: linear_probe/data/{train,test}.csv from the figshare release.

    python reproduce/02_split_composition.py
"""
import collections, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import emit, load_split, pct, section

# Annotation text for RT-like proteins. Matched against prot_id, which in this
# release carries the original description line. Used only for the incidental-RT
# count; the authors' own build log gives the Pfam-accession count (see 04).
RT_PAT = re.compile(r'reverse transcriptase|RNA-directed DNA polymerase|'
                    r'retrovirus-related pol|retrotransposon|transposon', re.I)


def main():
    print('reading LucaProt released splits (~30 s, train.csv is 125 MB)',
          flush=True)
    print('  train.csv ...', end='', flush=True)
    tr = load_split('train')
    print(' %d rows' % len(tr), flush=True)
    print('  test.csv  ...', end='', flush=True)
    te = load_split('test')
    print(' %d rows' % len(te), flush=True)

    emit('split.train_rows', len(tr))
    emit('split.test_rows', len(te))

    # ---- 1. composition by source label ----------------------------------
    section('SPLIT COMPOSITION BY SOURCE LABEL')
    ctr = collections.Counter(s for _, _, _, s in tr)
    cte = collections.Counter(s for _, _, _, s in te)
    print('  %-22s %9s %9s' % ('source', 'train', 'test'))
    for k in sorted(set(ctr) | set(cte)):
        print('  %-22s %9d %9d' % (k, ctr.get(k, 0), cte.get(k, 0)))
    print('  %-22s %9d %9d' % ('TOTAL', len(tr), len(te)))

    trpos_n = sum(1 for _, _, l, _ in tr if l == 1)
    tepos_n = sum(1 for _, _, l, _ in te if l == 1)
    teneg_n = len(te) - tepos_n
    emit('split.train_pos', trpos_n)
    emit('split.test_pos', tepos_n)
    emit('split.test_neg', teneg_n)
    emit('split.rt_train', ctr.get('rt', 0))
    emit('split.rt_test', cte.get('rt', 0))
    emit('split.dna_train', ctr.get('dna', 0))
    emit('split.dna_test', cte.get('dna', 0))

    hard_te = cte.get('other_virus', 0) + cte.get('other_virus_domain', 0)
    emit('split.test_hard_neg', hard_te)
    emit('split.test_cellular', cte.get('non_virus', 0))
    # the post's "170k random cellular proteins for good measure"
    emit('split.train_cellular', ctr.get('non_virus', 0))

    print('\n  positives: %d train, %d test' % (trpos_n, tepos_n))
    print('  the test set\'s declared hard negatives are %d non-RdRP viral'
          ' proteins' % hard_te)
    print('  against %d cellular ones.' % cte.get('non_virus', 0))

    # ---- 2. the RT that are there anyway ---------------------------------
    section('REVERSE TRANSCRIPTASES NOT LABELLED "rt"')
    for nm, rows, cid in [('train', tr, 'rt_incidental.train'),
                          ('test', te, 'rt_incidental.test')]:
        c = collections.Counter(s for pid, _, _, s in rows
                                if s != 'rt' and RT_PAT.search(pid))
        n = sum(c.values())
        print('  %-6s %4d  %s' % (nm, n, dict(c)))
        emit(cid, n)
    print('\n  So "no reverse transcriptases in test" would be false. The correct'
          '\n  claim is that every row *labelled* rt is in training, leaving ~50'
          '\n  incidental ones in test under the non_virus label - 0.2%% of test'
          '\n  negatives against 5.2%% of the training set.')

    # ---- 3. byte-identical overlap ---------------------------------------
    section('BYTE-IDENTICAL OVERLAP (no aligner, no threshold)')
    trpos = set(s for _, s, l, _ in tr if l == 1)
    trall = set(s for _, s, _, _ in tr)
    tepos = [s for _, s, l, _ in te if l == 1]
    teall = [s for _, s, _, _ in te]

    d1 = sum(1 for s in tepos if s in trpos)
    d2 = sum(1 for s in teall if s in trall)
    print('  test positives identical to a training positive : %3d / %d  (%.1f%%)'
          % (d1, len(tepos), pct(d1, len(tepos))))
    print('  all test rows identical to any training row     : %3d / %d  (%.2f%%)'
          % (d2, len(teall), pct(d2, len(teall))))
    emit('dup.test_pos_identical', d1)
    emit('dup.test_pos_identical_pct', '%.1f' % pct(d1, len(tepos)))

    c = collections.Counter(s for _, s, l, _ in tr if l == 1)
    print('\n  incidentally, the training positives are not distinct either:')
    print('    %d rows -> %d distinct sequences (%d appear more than once)'
          % (trpos_n, len(trpos), sum(1 for v in c.values() if v > 1)))
    emit('dup.train_pos_distinct', len(trpos))


if __name__ == '__main__':
    main()
