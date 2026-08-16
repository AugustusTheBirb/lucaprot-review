#!/usr/bin/env python
"""What LucaProt calls an RdRP when it shouldn't - measured on held-out negatives.

Sections 02-05 are about recall: the test set can't measure detection beyond
homology (02, 03), and when you build an evaluation that can, LucaProt passes it
(05). None of that touches the false-positive side, and the false-positive side
is what the paper's discovery claim rests on - 444 contigs found by LucaProt and
nothing else.

The paper offers two false-positive rates, 0.014% and 0.023%, and neither is
measured on anything held out: the first is not reconstructable from its own
confusion matrix, and the second comes from a benchmark whose positives are
defined by the tools being benchmarked. So this measures one directly.

Construction, per class (fp_panel/fetch_uniprot.py then fp_panel/screen_sets.py):

    UniProt, by taxonomy and length band
       |   1. accession absent from train.csv and test.csv
       |   2. cleaned sequence not byte-identical to any split row
       |   3. no DIAMOND blastp hit (--very-sensitive, e <= 1e-3) to a reference
       |      of 8,024 RdRPs = 4,900 training positives + 250 control positives
       |      + 2,874 published orphans. Broader than the paper's own
       |      5,979-sequence curated RdRP database, so exclusion is conservative.
       v
    53,709 sequences that are not RdRPs and that LucaProt did not train on

Scoring is the released, unmodified predict_many_samples.py at the paper's own
threshold of 0.5, checkpoint sefn/20230201140320/checkpoint-100000, moved onto
A100s by fp_panel/lucaprot_modal.py. Nothing in the model path is reimplemented,
and the port is checked against the laptop CPU path below.

Two things this script deliberately does not do, both because they were wrong
when tried: pool the classes into one rate, and multiply any rate by 144.6M. See
the guardrail section at the bottom.

    python reproduce/09_false_positive_panel.py
"""
import collections, csv, glob, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ROOT, emit, need, pct, section, wilson

PANEL = os.path.join(ROOT, 'fp_panel')
PREDS = os.path.join(PANEL, 'preds')
SETS = os.path.join(PANEL, 'sets')
RESULTS = os.path.join(PANEL, 'results')
THRESH = 0.5

# class -> claim key, the UniProt sets that make it up, one-line description.
# The grouping is taxonomic because the taxonomy is what the result turned out
# to be about; an earlier length-based grouping is the retracted conclusion
# recorded at the bottom of this file.
CLASSES = [
    ('cellular, short (<2,500 aa)', 'cell_short', [
        'long_bact_100_299', 'long_bact_300_499', 'long_bact_500_999',
        'long_bact_1000_1499', 'long_bact_1500_1999', 'long_bact_2000_2499',
        'short_bact_bulk'],
     'bacterial proteins, six length bands plus bulk'),
    ('bacteriophage', 'phage', ['short_phage_caudo'],
     'Caudoviricetes - DNA viruses of bacteria'),
    ('prokaryotic mobile-element RT', 'rt', ['short_prok_rt'],
     'group II intron maturases, retrons, DGRs'),
    ('cellular, long (>=2,500 aa)', 'cell_long', [
        'long_bact_2500_2999', 'long_bact_3000_3499', 'long_bact_3500_3999',
        'long_bact_4000_4499', 'long_bact_4500_4999', 'long_bact_5000_100000',
        'long_sp_cellular'],
     'bacterial >=2,500 aa plus SwissProt long cellular'),
    ('eukaryotic DNA virus, 300-999 aa', 'dnavirus_short',
     ['short_dnavirus_300_999'], 'parvo-, herpes-, poxviruses'),
    ('eukaryotic RNA virus non-RdRP, short', 'rnavirus', ['short_rnavirus'],
     'RNA virus proteins that are not RdRPs'),
    ('eukaryotic DNA virus, 1,000-2,499 aa', 'dnavirus_mid',
     ['short_dnavirus_1000_2499'], 'baculo-, herpes-, giant viruses'),
]

# reported separately: not UniProt draws, so not part of the panel proper
EXTRAS = [
    ('viral non-RdRP, long (their own negatives)', 'other_virus',
     ['other_virus_277'], "LucaProt's own labelled non-RdRP viral negatives"),
    ('long viral, RdRP-homology-screened', 'long_viral', ['long_viral'],
     'n=15 after screening; here for completeness, not a rate'),
]

# the pooled row the post quotes, and the only pooling that is defensible:
# these three classes share a taxonomic domain, so combining them is a
# statement about prokaryotes rather than about an arbitrary mixture
PROK = ['cell_short', 'phage', 'rt']


def load_preds(name):
    p = os.path.join(PREDS, name + '.csv')
    if not os.path.exists(p):
        return {}
    with open(p, newline='') as f:
        return {r['uid']: float(r['prob']) for r in csv.DictReader(f)}


def load_meta(name):
    p = os.path.join(SETS, name + '.meta.tsv')
    if not os.path.exists(p):
        return {}
    out = {}
    with open(p, newline='', encoding='utf-8', errors='replace') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            ident = r.get('ident_to_nearest_split') or ''
            out[r['uid']] = (int(r['length']),
                             float(ident) if ident else None,
                             r.get('header', ''))
    return out


def load_cpu(name):
    """The laptop CPU run, where one exists - used only to check the GPU port."""
    out = {}
    for p in sorted(glob.glob(os.path.join(RESULTS, name, 'chunk_*', 'result.csv'))):
        with open(p, newline='', encoding='utf-8', errors='replace') as f:
            for r in csv.DictReader(f):
                if r.get('protein_id'):
                    out[r['protein_id'].lstrip('>')] = float(r['prob'])
    return out


def collect(names):
    """-> [(prob, length, ident_to_nearest_split, header)] over a class."""
    rows = []
    for name in names:
        probs, meta = load_preds(name), load_meta(name)
        for uid, p in probs.items():
            ln, ident, hdr = meta.get(uid, (None, None, uid))
            rows.append((p, ln, ident, hdr))
    return rows


def desc(header):
    """UniProt header -> just the protein description."""
    h = re.sub(r'^(sp|tr)\|[A-Z0-9]+\|[A-Z0-9_]+ ', '', header)
    return re.sub(r' (OS=|OX=).*$', '', h).strip()


def organism(header):
    m = re.search(r'OS=(.+?) (OX=|GN=|PE=|SV=)', header)
    return m.group(1) if m else '?'


def accession(header):
    m = re.match(r'(?:sp|tr)\|([A-Z0-9]+)\|', header)
    return m.group(1) if m else header


def row(label, k, n, note=''):
    lo, hi = wilson(k, n)
    print('  %-38s %7s %6d %8.3f%%  %6.3f - %-6.3f %s'
          % (label, format(n, ','), k, pct(k, n), lo, hi, note))
    return lo, hi


def main():
    need(PREDS, 'GPU predictions from fp_panel/lucaprot_modal.py')
    if not glob.glob(os.path.join(PREDS, '*.csv')):
        print('SKIP\tno predictions in %s' % PREDS)
        sys.exit(3)

    # ---- provenance: is this really held out? ----------------------------
    section('IS THE PANEL ACTUALLY HELD OUT?')
    idents = [ident
              for _, _, names, _ in CLASSES + EXTRAS
              for _, _, ident, _ in collect(names)
              if ident is not None]
    if idents:
        idents.sort()
        print('  %s sequences carry a DIAMOND identity to their nearest neighbour'
              % format(len(idents), ','))
        print('  in LucaProt\'s train+test splits:  median %.1f%%   max %.1f%%'
              % (idents[len(idents) // 2], idents[-1]))
        print('  %d of them (%.2f%%) sit at >= 90%% identity.'
              % (sum(1 for i in idents if i >= 90),
                 pct(sum(1 for i in idents if i >= 90), len(idents))))
        emit('panel.max_ident_to_splits', '%.1f' % idents[-1])
        print("""
  Identity to a split sequence is not disqualifying on its own, and the max of
  100% is not a leak: resembling one of the 170k cellular negatives is expected
  and harmless. Criterion 2 removes byte-identical duplicates, and criterion 3
  is what makes the rate a false-positive rate - none of these has any homology
  to an RdRP, so none of them can be a true positive that was scored correctly.
  dev.csv was not available, so the
  exclusion covers 213,130 of 235,414 split rows; the validation split drove
  checkpoint selection rather than gradient updates, so its leakage risk is
  lower, not zero.""")

    # ---- the panel -------------------------------------------------------
    section('FALSE-POSITIVE RATE BY CLASS (threshold %.2f)' % THRESH)
    print('  %-38s %7s %6s %9s  %-15s' % ('class', 'n', 'FP', 'FPR', '95% CI'))

    got, total_n, total_fp = {}, 0, 0
    for label, key, names, what in CLASSES:
        rows = collect(names)
        if not rows:
            continue
        n = len(rows)
        k = sum(1 for p, _, _, _ in rows if p >= THRESH)
        got[key] = rows
        row(label, k, n)
        emit('panel.%s_n' % key, n)
        emit('panel.%s_fp' % key, k)
        emit('panel.%s_pct' % key, '%.3f' % pct(k, n))
        total_n += n
        total_fp += k

    prok_rows = [r for k in PROK for r in got.get(k, [])]
    if prok_rows:
        n = len(prok_rows)
        k = sum(1 for p, _, _, _ in prok_rows if p >= THRESH)
        print('  ' + '-' * 78)
        row('ALL PROKARYOTIC + PHAGE', k, n, '<- the headline')
        emit('panel.prok_all_n', n)
        emit('panel.prok_all_fp', k)
        emit('panel.prok_all_pct', '%.3f' % pct(k, n))

    print('\n  reported separately (not UniProt draws):')
    for label, key, names, what in EXTRAS:
        rows = collect(names)
        if not rows:
            continue
        n = len(rows)
        k = sum(1 for p, _, _, _ in rows if p >= THRESH)
        got[key] = rows
        row(label, k, n)
        emit('panel.%s_n' % key, n)
        emit('panel.%s_fp' % key, k)
        emit('panel.%s_pct' % key, '%.3f' % pct(k, n))
        total_n += n
        total_fp += k

    emit('panel.total_scored', total_n)
    emit('panel.total_fp', total_fp)
    print('\n  %s sequences scored, %d called RdRP.'
          % (format(total_n, ','), total_fp))

    print("""
  Zero false positives in 28,350 prokaryotic and phage proteins; 1.4-5.1% on
  eukaryotic viruses. Phage at exactly zero while other DNA viruses run 1.4-5.1%
  is the sharpest contrast in the panel - both are DNA viruses, and the
  difference is host domain.""")

    # ---- what fires ------------------------------------------------------
    section('WHAT IT FIRES ON')
    print('  Not what anyone would predict, which is the substance of the result.\n')

    for label, key in [('eukaryotic DNA virus, 1,000-2,499 aa', 'dnavirus_mid'),
                       ('eukaryotic DNA virus, 300-999 aa', 'dnavirus_short')]:
        fired = [r for r in got.get(key, []) if r[0] >= THRESH]
        if not fired:
            continue
        words = collections.Counter()
        for _, _, _, hdr in fired:
            for w in re.findall(r'[a-z]{5,}', desc(hdr).lower()):
                words[w] += 1
        print('  %s - %d false positives' % (label, len(fired)))
        print('    top description words: %s'
              % ', '.join('%s (%d)' % (w, c) for w, c in words.most_common(5)))
        for p, ln, _, hdr in sorted(fired, reverse=True)[:3]:
            print('    %.4f  %5s aa  %-46s [%s]'
                  % (p, ln, desc(hdr)[:46], organism(hdr)[:28]))
        print()

    fired = [r for r in got.get('cell_long', []) if r[0] >= THRESH]
    if fired:
        def count(pat):
            return sum(1 for _, _, _, h in fired
                       if re.search(pat, desc(h), re.I))
        ubi, seca, poly = (count(r'ubiquitinyl hydrolase'), count(r'SecA'),
                           count(r'polymerase'))
        print('  long cellular proteins - %d false positives' % len(fired))
        print('    %d ubiquitinyl hydrolase 1, %d SecA-family, %d polymerases'
              % (ubi, seca, poly))
        emit('panel.cell_long_ubiquitinyl', ubi)
        emit('panel.cell_long_seca', seca)
        emit('panel.cell_long_polymerase', poly)
        intra = sum(1 for _, _, _, h in fired
                    if re.search(r'Legionella|Berkiella|Criblamydia|Estrella',
                                 organism(h)))
        print('    %d of them are Legionella / Berkiella / Criblamydia / Estrella'
              % intra)
        print('    - intracellular bacteria with long, eukaryote-like proteomes')
        longest = sorted((r for r in fired if r[1]), key=lambda r: -r[1])[:2]
        if longest:
            print('    longest two: %s aa, both TrEMBL gene-model artifacts'
                  % ' and '.join(format(r[1], ',') for r in longest))
        print("""
    Only %d of %d are polymerases. If the model had learned the catalytic core
    it would be confused by other polymerases, which share it; instead it is
    confused by long proteins with no relationship to nucleotide synthesis. That
    is the argument for a learned proxy rather than a learned mechanism.""" % (poly, len(fired)))

        dup = collections.Counter(accession(h) for _, _, _, h in fired)
        if any(c > 1 for c in dup.values()):
            print('\n    (%d of the %d is the same UniProt accession drawn into'
                  ' two overlapping sets - bacterial >=5,000 aa and SwissProt'
                  '\n    long cellular - so %d distinct proteins. The count of 24'
                  ' is over calls, not\n    over proteins; both sets were scored'
                  ' independently.)'
                  % (sum(c - 1 for c in dup.values() if c > 1), len(fired),
                     len(dup)))

    # ---- RT, since the post makes a claim about it -----------------------
    rt_rows = got.get('rt', [])
    if rt_rows:
        section('THE CLASS THE PAPER CALLS ITS HARD NEGATIVE')
        mx = max(p for p, _, _, _ in rt_rows)
        print('  %s prokaryotic mobile-element reverse transcriptases,'
              ' maximum score %.4f' % (format(len(rt_rows), ','), mx))
        emit('panel.rt_max_prob', '%.4f' % mx)
        print("""  Complete separation, not near-misses. RT is the *easiest* class in the
  panel, not the hardest. The methodological criticism survives untouched - all
  10,000 RTs went into training, so the authors never measured this - but the
  empirical claim that RT is what confuses RdRP detection is false, and the post
  should not make it.""")

    # ---- the port ---------------------------------------------------------
    section('IS THE GPU PORT THE SAME MODEL?')
    checked = 0
    for _, key, names, _ in CLASSES + EXTRAS:
        for name in names:
            gpu, cpu = load_preds(name), load_cpu(name)
            both = set(gpu) & set(cpu)
            if not both:
                continue
            dis = sum(1 for u in both if (gpu[u] >= THRESH) != (cpu[u] >= THRESH))
            mx = max(abs(gpu[u] - cpu[u]) for u in both)
            print('  %-24s %4d shared   %d call-disagreements   max |dprob| %.1e'
                  % (name, len(both), dis, mx))
            checked += len(both)
            emit('panel.cpu_gpu_shared', len(both))
            emit('panel.cpu_gpu_disagree', dis)
    if not checked:
        print('  no set has both a CPU and a GPU run committed')

    # ---- guardrails -------------------------------------------------------
    section('WHAT THESE NUMBERS DO NOT SUPPORT')
    print("""  Two conclusions were reached during this work and then retracted. Both were
  artifacts of which classes had been measured at the time, and both are easy to
  fall into again from the table above.

  1. "There is no length effect." From the >=2,500 aa bands alone the rate looks
     flat - 0.375, 0.875, 0.501, 0.000, 0.125, 0.377. It is flat because those
     bands are all on the plateau.

  2. "False positives switch on at 2,500 aa." Adding six short bands gave
     0/4,762 below 2,500 aa against 18/4,790 above, which looks like a step
     function. It was an artifact of testing only *cellular* proteins at short
     lengths. Short viral proteins fire at 1.4-5.1% between 300 and 2,499 aa,
     inside the region that had been declared clean.

  The split is taxonomic, not dimensional. fp_panel/length_vs_fpr.png is correct
  as far as it goes but is a cellular-only curve and must be labelled as one.

  3. Do not multiply any rate here by 144.6 million. An earlier draft gave ~825
     expected false positives by weighting the >=2,500 aa rate by the 0.152% of
     the corpus that is that long. That is retracted: it assumed the driver was
     length. A corpus-wide estimate needs the eukaryotic-virus protein fraction
     of the 144.6M, which nothing here measures. The honest statement is a
     conditional rate per class, which is what the table gives.""")


if __name__ == '__main__':
    main()
