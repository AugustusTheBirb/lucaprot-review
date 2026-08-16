#!/usr/bin/env python
"""Where the hard negatives went, read off the authors' own preprocessing script.

02 shows the *outcome* - all 10,000 reverse transcriptases in training, none in
test. This script shows the *mechanism*, which is more damning and also more
mundane than the outcome alone suggests.

The file is vendored in this repo at

    LucaProt/src/data_preprocess/data_preprocess_for_rdrp_v3_extend.py

and lines 182-208 are a comment block containing the authors' captured stdout
from the run that produced the released dataset. Rather than paraphrase it, this
script locates that block and the three code facts that interpret it, so the
evidence is quoted from the source at runtime and cannot drift.

The three findings:

  1. Two-stage build. A base split of 178,262 / 22,284 / 22,284 - an exact
     8:1:1 - was extended by appending positives and hard negatives to the
     TRAINING list only; dev and test were copied through untouched. The
     "8.5:1:1" the paper reports is a by-product of one-sided augmentation, not
     a split design.

  2. Deliberate subsampling. 81,867 reverse transcriptases were retrieved,
     31,843 survived a length filter, and the list was shuffled and truncated to
     exactly 10,000. Two thirds of the available hard negatives were discarded,
     and every one kept was placed where it could not be tested against.

  3. `rt exists: [439, 64, 53]` - reverse transcriptases already present in the
     base splits, skipped by the augmentation because their ids were taken, and
     left carrying the generic non_virus label. This is the ~50 in test that 02
     finds independently by annotation text.

This script parses; it does not execute the preprocessing pipeline. Doing that
would need source FASTAs (data/rdrp/non_virus_sequence.fasta, non-virus.info.txt)
that are not in the released repo - which is itself why the paper's stated
n=48,490 reverse transcriptases cannot be reconciled with the released 10,000.

    python reproduce/04_how_the_split_was_built.py
"""
import os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ROOT, emit, need, section

SRC = os.path.join(ROOT, 'LucaProt', 'src', 'data_preprocess',
                   'data_preprocess_for_rdrp_v3_extend.py')

# claim id -> regex over the authors' captured stdout
LOG_FIELDS = [
    ('build.base_train',      r'train\s+train\s+(\d+)'),
    ('build.base_dev',        r'dev:\s*(\d+)'),
    ('build.base_test',       r'test:\s*(\d+)'),
    ('build.rt_retrieved',    r'rt sequence number:\s*(\d+)'),
    ('build.rt_filtered',     r'total rt size:\s*(\d+)'),
    ('build.rt_kept',         r'append_negative_rt len:\s*(\d+)'),
    ('build.dna_candidates',  r'total dna size:\s*(\d+)'),
    ('build.dna_kept',        r'append_negative_dna len:\s*(\d+)'),
    ('build.pos_appended',    r'append_positive_rdrp len:\s*(\d+)'),
]


def main():
    need(SRC, 'the vendored LucaProt source; clone it to LucaProt/ if absent')
    src = open(SRC, encoding='utf-8', errors='replace').read()

    section("THE AUTHORS' OWN BUILD LOG, QUOTED FROM THEIR SOURCE")
    for cid, pat in LOG_FIELDS:
        m = re.search(pat, src)
        val = m.group(1) if m else 'NOT-FOUND'
        print('  %-24s %s' % (cid.split('.')[1], val))
        emit(cid, val)

    m = re.search(r'rt exists:\s*\[(\d+),\s*(\d+),\s*(\d+)\]', src)
    if m:
        print('  %-24s train %s / dev %s / test %s'
              % ('rt already in splits', m.group(1), m.group(2), m.group(3)))
        emit('build.rt_preexisting_train', m.group(1))
        emit('build.rt_preexisting_test', m.group(3))

    # The log prints a running total after each append, so the training set can
    # be watched growing. First value is the base split, last is what shipped.
    growth = re.findall(r'cur train_dataset size:\s*(\d+)', src)
    if growth:
        print('\n  training set size after each append step:')
        print('    %s' % '  ->  '.join(growth))
        emit('build.base_train_running', growth[0])
        emit('build.final_train', growth[-1])
        print('    (the first two steps fold dev and test back in; the'
              ' +584 / +10,000 / +2,000\n     steps are the positives, the reverse'
              ' transcriptases and the DNA-virus\n     proteins. The released'
              ' train/dev/test split is cut from the result.)')

        # ---- what the cross-validation was run on ------------------------
        # The post says the cross-validation set is "short about 12k - almost
        # exactly how many were added after the fact". The two numbers that
        # bracket that are both in the log above.
        base = int(growth[2])            # dev and test folded in, nothing appended
        final = int(growth[-1])
        released = 190846 + 22284 + 22284
        print('\n  the base dataset before any augmentation   %s' % format(base, ','))
        print('  the same dataset after augmentation        %s' % format(final, ','))
        print('  appended                                   %s'
              % format(final - base, ','))
        print('  released train + dev + test                %s   (%d more than the'
              ' log)' % (format(released, ','), released - final))
        emit('build.base_dataset', base)
        emit('build.appended_total', final - base)
        print("""
  So ~12.6k sequences exist in the released data that are absent from the
  pre-augmentation dataset, and they are exactly the hard negatives and extra
  positives. Whether the cross-validation in Figure S2 was run before or after
  that step decides whether it saw them at all.

  The count on the Figure S2 panel itself is tier E - it lives in a raster
  image, like the accuracy and AUC values, and is not extractable here. This
  script establishes the size of the gap; reading which side of it the
  cross-validation sits on is an eyeball check on page 23.""")

    # ---- the code facts that interpret the log ---------------------------
    section('THE THREE CODE FACTS')

    trunc = re.search(r'if len\(append_negative_rt_fasta_info\) >= (\d+):', src)
    print('  1. RT truncation threshold in code           : %s'
          % (trunc.group(1) if trunc else 'NOT-FOUND'))
    shuf = re.search(r'for _ in range\((\d+)\):\s*\n\s*random\.shuffle', src)
    print('     shuffles before truncating                : %s'
          % (shuf.group(1) if shuf else 'NOT-FOUND'))

    # the augmentation appends to train_dataset and nothing else
    appends = re.findall(r'(\w+)_dataset\.extend\(\[\[row\[0\], row\[1\]\]', src)
    print('  2. datasets the hard negatives are appended to: %s'
          % (', '.join(sorted(set(appends))) or 'NOT-FOUND'))
    emit('build.augmented_splits', ','.join(sorted(set(appends))))
    print('     -> dev and test are absent from that list, i.e. copied unchanged')

    print('  3. the ~50 RT left in test under the non_virus label are counted'
          '\n     independently by annotation text in 02_split_composition.py')

    # ---- what cannot be reconciled ---------------------------------------
    section('WHAT THIS DOES NOT EXPLAIN')
    print("""  The paper states n=48,490 reverse transcriptases among the negatives.
  The RT counts that exist anywhere in this pipeline are:

      81,867  retrieved by Pfam accession
      31,843  after the length filter
      10,000  sampled into the released training set
        ~556  already present in the base splits

  48,490 is none of them, and the source files that would settle it are not in
  the release. Both the paper's stated composition and the released splits sum
  to 229,434 negatives, with incompatible category breakdowns in between.

  The post says exactly this and asserts nothing about which is correct.""")


if __name__ == '__main__':
    main()
