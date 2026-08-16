#!/usr/bin/env python
"""Numbers the post quotes from the paper, plus the arithmetic done on them.

Nothing here is a re-analysis. The point is narrower and still worth automating:
every figure the post attributes to Hou et al. is checked to actually appear in
the paper, so a misremembered or drifted number fails loudly instead of sitting
in the post looking authoritative.

The text is extracted from ../paper.pdf with pdftotext and cached in .cache/.
Two Figure S2 values (accuracy 0.999955, AUC 0.99999991) live inside a raster
panel and cannot be extracted from text at all; they are read off the figure by
eye and are flagged `tier=E` (eyeballed) in claims.tsv rather than checked here.

The second half computes the false-positive argument in the post: the paper's
own reported rate applied to the paper's own screening scale.

    python reproduce/01_paper_claims.py
"""
import os, re, subprocess, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ROOT, CACHE, emit, section

PDF = os.path.join(ROOT, 'paper.pdf')
TXT = os.path.join(CACHE, 'paper.txt')

# Each entry: claim id, the literal string that must appear in the paper text,
# and what it is. The string is matched verbatim, including the comma grouping
# the typesetter used, which is deliberate - a loose regex would defeat the
# purpose of the check.
QUOTED = [
    ('paper.fpr_pct',            '0.014% false positives',   'reported false-positive rate on the test set'),
    ('paper.fnr_pct',            '1.72% false negatives',    'reported false-negative rate on the test set'),
    ('paper.n_proteins_screened','144.6 million proteins',   'proteins screened across all metatranscriptomes'),
    ('paper.n_calls',            '792,436',                  'sequences LucaProt called RdRP'),
    ('paper.n_shared_contigs',   '512,690',                  'contigs found by both LucaProt and ClstrSearch'),
    ('paper.n_unique_contigs',   '444',                      'contigs found ONLY by LucaProt'),
    ('paper.n_unique_supergroups','23 supergroups',          'supergroups found only by LucaProt'),
    ('paper.n_metatranscriptomes','10,487',                  'metatranscriptomes analysed'),
    ('paper.n_species',          '161,979',                  'putative virus species claimed'),
    ('paper.n_supergroups',      '180 RNA viral super-',     'virus supergroups claimed'),
    ('paper.n_negatives',        '229,434',                  'negative training samples'),
    ('paper.rt_stated',          'n=48,490',                 'reverse transcriptases the paper says are in the negatives'),
    ('paper.ablation_no_esm2',   '44.5%',                    'recall with the ESM2 channel removed'),
    ('paper.fold_expansion',     '8.6-fold expansion',       'claimed expansion at the supergroup level'),
    ('paper.ictv_supergroups',   '21 contained those from viral phyla/classes',
                                                             'supergroups containing anything ICTV classifies'),
    ('paper.rtpcr_supergroups',  '17 of the 115 RNA viral supergroups',
                                                             'supergroups put through RT-PCR validation'),
    ('paper.rtpcr_new',          'Supergroup102, Supergroup167, Supergroup175',
                                                             'the five NEW supergroups among them'),
]


def paper_text():
    if os.path.exists(TXT):
        return open(TXT, encoding='utf-8', errors='replace').read()
    if not os.path.exists(PDF):
        print('SKIP\tmissing %s - the paper PDF is not redistributed here' % PDF)
        sys.exit(3)
    os.makedirs(CACHE, exist_ok=True)
    print('  extracting text from paper.pdf ...', end='', flush=True)
    try:
        subprocess.run(['pdftotext', '-layout', PDF, TXT], check=True,
                       capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        print('\nSKIP\tpdftotext not available; install poppler-utils')
        sys.exit(3)
    print(' ok', flush=True)
    return open(TXT, encoding='utf-8', errors='replace').read()


def main():
    txt = paper_text()
    # collapse hyphenation and whitespace introduced by the two-column layout
    flat = re.sub(r'\s+', ' ', txt)

    section('NUMBERS QUOTED FROM THE PAPER (verified present in the text)')
    missing = 0
    for cid, literal, what in QUOTED:
        ok = literal in flat
        missing += not ok
        print('  %-8s %-26s %s' % ('FOUND' if ok else 'ABSENT', literal, what))
        emit(cid, literal if ok else 'NOT-FOUND-IN-PAPER')

    if missing:
        print('\n  %d quoted value(s) not found. Either the post has drifted from the'
              '\n  paper or the PDF differs from the one this was written against.'
              % missing)

    # ---- the two shares the post quotes ----------------------------------
    section('THE SHARE OF CONTIGS BOTH PIPELINES FOUND')
    shared, unique = 512690, 444
    total = shared + unique
    print('  found by both LucaProt and ClstrSearch   %s' % format(shared, ','))
    print('  found by LucaProt only                   %d' % unique)
    print('  share found by both                      %.1f%%'
          % (100.0 * shared / total))
    emit('paper.shared_contig_pct', '%.1f' % (100.0 * shared / total))
    print('\n  The post says "99.9%% of the viral contigs were found by both".'
          '\n  That is this ratio, not a figure the paper states in those terms.')

    # ---- the false-positive argument -------------------------------------
    section('THE FALSE-POSITIVE ARITHMETIC')
    n_screened = 144.6e6

    # The paper offers two false-positive rates and the post uses both.
    # 0.014% appears in the text and is checked above. 0.023% does not appear
    # in any extractable text - it lives inside the Figure 3A panel and is
    # read off it by eye, so it is tier E in claims.tsv exactly like the two
    # Figure S2 values. The arithmetic on it is checked here; the number it
    # starts from is not, and cannot be.
    for label, fpr, cid in [('text, "evaluated on the test dataset"', 0.00014, 'fp'),
                            ('Figure 3A, 50 metatranscriptomes', 0.00023, 'fp_fig3a')]:
        expected = n_screened * fpr
        print('\n  %s' % label)
        print('    reported FPR                 %.3f%%' % (100 * fpr))
        print('    x 144.6M proteins            %s expected false positives'
              % format(int(round(expected)), ','))
        print('    contigs unique to LucaProt   %d' % 444)
        print('    ratio                        %.0fx' % (expected / 444))
        emit('%s.expected' % cid, '%.0f' % expected)
        emit('%s.ratio_to_unique' % cid, '%.0f' % (expected / 444))

    expected = n_screened * 0.00014

    print("""
  Two caveats the post states, and neither is resolvable from published data:

  (a) The %0.f is expected false positives among the 792,436 raw calls, while
      444 is a count after merging against the ClstrSearch homology pipeline.
      Different stages. Whether that merge preferentially removes false
      positives is unknown; it plausibly removes some.
  (b) 0.014%% is measured on the test set that 02/03 below show to be leaky, so
      it is an optimistic estimate of the deployment rate, not a conservative
      one.

  So this is not a demonstration that the 444 are spurious. It is a
  demonstration that the paper's own numbers do not rule it out. That holds
  under either rate, which is why the post can quote both without the argument
  depending on which one is right.""" % expected)


if __name__ == '__main__':
    main()
