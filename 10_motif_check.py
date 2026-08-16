#!/usr/bin/env python
"""Do the paper's 513,134 claimed RdRPs carry an RdRP catalytic core?

This is the check that could have been the strongest result in the post, and it
came back clean. It is here because a post arguing "verify the evaluation" has
to report the verification that went the other way.

The paper states its own criterion (STAR Methods, "Virus verification"):

    motif A [DxxxxD], motif B [(S/T)Gxxx(T/G)xxxN], motif C [(S/G/N)DD]
    "Sequences that failed to cover these motifs were removed."

So by their own account every sequence in the released ours_checked_rdrp_final.csv
should carry all three. Testing that means implementing their regexes, requiring
A -> B -> C in order with palm-like spacing (without the spacing, three short
degenerate motifs anywhere in a 4,000 aa protein co-occur by chance almost
always, and the test measures nothing).

The whole experiment is the two controls, not the headline percentage:

  POSITIVE CONTROL   LucaProt's own 4,900 training positives - curated,
                     reference-genome RdRPs. This is the detector's ceiling.
  POSITIVE CONTROL   the 2,874 published orphan RdRPs from section 05.
  NULL               39,098 cellular / phage / RT proteins from the section-09
                     panel, none of which is an RdRP. This is the chance rate.

Without both, "X% carry motifs" is uninterpretable in either direction. With
them the reading is unambiguous, and it is not the one that would help the post:
the claimed set scores well above chance and above published orphan RdRPs, while
the detector misses half of the authors' own reference RdRPs - so it is far too
insensitive to support any negative claim.

    python reproduce/10_motif_check.py

Needs ours_checked_rdrp_final.csv (408 MB) from the figshare release,
doi 10.6084/m9.figshare.26298802.v14, in fp_panel/motif/. Everything except the
last row runs without it.
"""
import csv, glob, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ROOT, emit, need, pct, section

csv.field_size_limit(10 ** 9)

PANEL = os.path.join(ROOT, 'fp_panel')
CLAIMED = os.path.join(PANEL, 'motif', 'ours_checked_rdrp_final.csv')
TRAIN_POS = os.path.join(ROOT, 'analysis', 'train_pos.faa')
ORPHANS = os.path.join(ROOT, 'no_training_homolog.faa')

# The paper writes motif C two ways: [(S/G/N/A)D(D/N)] in the results and
# [(S/G/N)DD] in the Methods. The stricter Methods form is used, since that is
# the one stated as the removal criterion; the looser form is reported beside it
# to show the choice does not carry the conclusion.
A = re.compile(r'D.{4}D')
B = re.compile(r'[ST]G.{3}[TG].{3}N')
C = re.compile(r'[SGN]DD')
C_LOOSE = re.compile(r'[SGNA]D[DN]')

AB_MIN, AB_MAX = 20, 150
BC_MIN, BC_MAX = 10, 100

# the null is every panel set that is definitely not viral RdRP material
NULL_PREFIXES = ('long_bact', 'short_bact', 'short_phage', 'short_prok',
                 'short_archaea')


def has_core(seq, c_re=C):
    """True if A, B and C occur in order with palm-like spacing."""
    for ma in A.finditer(seq):
        aend = ma.end()
        for mb in B.finditer(seq, aend + AB_MIN, min(len(seq), aend + AB_MAX + 20)):
            if not (AB_MIN <= mb.start() - aend <= AB_MAX):
                continue
            bend = mb.end()
            for mc in c_re.finditer(seq, bend + BC_MIN,
                                    min(len(seq), bend + BC_MAX + 10)):
                if BC_MIN <= mc.start() - bend <= BC_MAX:
                    return True
    return False


def read_fasta(path):
    seqs, buf, seen = [], [], False
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if seen:
                    seqs.append(''.join(buf))
                seen, buf = True, []
            elif line:
                buf.append(line.upper())
    if seen:
        seqs.append(''.join(buf))
    return seqs


def score(label, seqs, key, note='', dp=1):
    """dp is the precision the post quotes for this row, so that run_all.py can
    compare exactly rather than against a rounding of our own choosing."""
    n = len(seqs)
    if not n:
        return
    core = sum(1 for s in seqs if has_core(s))
    loose = sum(1 for s in seqs if has_core(s, C_LOOSE))
    a = sum(1 for s in seqs if A.search(s))
    b = sum(1 for s in seqs if B.search(s))
    c = sum(1 for s in seqs if C.search(s))
    print('  %-40s %9s %8.2f%% %9.2f%%   %5.1f %5.1f %5.1f  %s'
          % (label, format(n, ','), pct(core, n), pct(loose, n),
             pct(a, n), pct(b, n), pct(c, n), note))
    emit('motif.%s_n' % key, n)
    emit('motif.%s_pct' % key, '%.*f' % (dp, pct(core, n)))


def main():
    print('Motif definitions, verbatim from the paper\'s STAR Methods:')
    print('  A  D.{4}D      B  [ST]G.{3}[TG].{3}N      C  [SGN]DD')
    print('  required in order, spacing A->B %d-%d aa, B->C %d-%d aa'
          % (AB_MIN, AB_MAX, BC_MIN, BC_MAX))

    section('CATALYTIC CORE BY SET')
    print('  %-40s %9s %8s %10s   %-17s'
          % ('set', 'n', 'core', 'loose C', 'A / B / C alone'))

    if os.path.exists(TRAIN_POS):
        score("POS CTRL - LucaProt train positives",
              read_fasta(TRAIN_POS), 'train_pos', '<- the ceiling')
    if os.path.exists(ORPHANS):
        score('POS CTRL - published orphan RdRPs',
              read_fasta(ORPHANS), 'orphan')

    null = []
    for p in sorted(glob.glob(os.path.join(PANEL, 'sets', '*.faa'))):
        if os.path.basename(p).startswith(NULL_PREFIXES):
            null.extend(read_fasta(p))
    if null:
        # two decimals: the null is the one row where the post quotes 0.13%,
        # and one decimal would round it to 0.1% and hide the specificity
        score('NULL - cellular / phage / RT, not RdRP', null, 'null',
              '<- chance rate', dp=2)

    if not os.path.exists(CLAIMED):
        print('\n  (the claimed set is not present locally)')
        need(CLAIMED, 'ours_checked_rdrp_final.csv, 408 MB, from the figshare '
                      'release doi 10.6084/m9.figshare.26298802.v14')

    seqs = []
    with open(CLAIMED, encoding='utf-8', errors='replace') as f:
        for r in csv.DictReader(f):
            s = (r.get('seq') or '').strip().upper()
            if s:
                seqs.append(s)
    print()
    score("THE PAPER'S OWN claimed RdRPs", seqs, 'claimed',
          '<- should be ~100% by their Methods')

    n_null = len(null) or 1
    null_rate = sum(1 for s in null if has_core(s)) / n_null
    claimed_rate = sum(1 for s in seqs if has_core(s)) / max(1, len(seqs))
    if null_rate:
        enrich = claimed_rate / null_rate
        emit('motif.enrichment_over_null', '%d' % round(enrich))
        print('\n  enrichment of the claimed set over the null: %.0fx' % enrich)

    section('WHAT THIS RESULT MEANS FOR THE POST')
    print("""  It does not support the criticism, and the controls are why.

  The detector finds only half of LucaProt's own curated, reference-genome
  RdRPs, so its false-negative rate is enormous and no strong negative claim can
  rest on it. More decisively, the claimed set scores *higher* than published
  orphan RdRPs from prior literature, which are unambiguously real.

  What the null establishes is that the detector is specific - a fraction of a
  percent on 39,098 non-RdRPs - so the claimed set is enriched far above chance.
  By this test the paper's claimed RdRPs look real, and the post should say so
  in those words.

  Sensitivity is limited by motif B, present in only ~59% of training positives,
  and by the spacing constraints, which are certainly wrong for some families.
  Loosening motif C to the paper's other stated form changes nothing material -
  the 'loose C' column above moves the claimed set by about a third of a point.

  This does not speak to whether any individual supergroup is real. Section 11
  is where that question goes.""")


if __name__ == '__main__':
    main()
