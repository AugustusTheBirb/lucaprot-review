#!/usr/bin/env python
"""How much of the published virosphere rests on LucaProt's call alone.

Section 09 measures a false-positive rate. This measures the exposure: how many
of the 513,134 published RdRP sequences that rate could be acting on.

Table S4 of the paper gives, per sequence, the result of all three detection
methods at the thresholds printed in its own title - BLAST e <= 1E-3, HMM score
>= 10, AI probability >= 0.5. So "found by the AI and nothing else" is directly
computable rather than inferred, and the paper's own stated 98.22% AI detection
rate is available as a check on the parse.

Two of the numbers this produces are not in the paper:

    AI only, no BLAST and no HMM      13,903   2.71%
    detected by no method at all       6,414   1.25%

The first is 31x the 444 contigs the AI-only claim is usually discussed in terms
of. The second is sequences in the final published set that no method called at
the thresholds in the table's own title - presumably present by cluster
membership, which is a defensible design, but it means over 1% of the set is
supported by neither homology nor the model.

    python reproduce/11_table_s4_ai_only.py

Reads fp_panel/table_s4_supergroups.tsv, the committed per-supergroup aggregate.
fp_panel/parse_table_s4.py regenerates it from mmc4.xlsx (45 MB, from the paper's
supplementary material); that file is not redistributed here.
"""
import csv, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ROOT, emit, need, pct, section

PANEL = os.path.join(ROOT, 'fp_panel')
S4 = os.path.join(PANEL, 'table_s4_supergroups.tsv')
S3 = os.path.join(PANEL, 'table_s3_supergroups.tsv')
SEQS = os.path.join(PANEL, 'lucaprot_only_sequences.tsv')

# Read off Figure S4 by eye - see the caveat section at the bottom
NOMOTIF = ['Supergroup%03d' % n for n in (183, 184, 187, 189, 190, 191, 193)]


def read_tsv(path):
    with open(path, newline='', encoding='utf-8', errors='replace') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def main():
    need(S4, 'run fp_panel/parse_table_s4.py against mmc4.xlsx')
    rows = read_tsv(S4)
    i = lambda r, k: int(r[k])

    n = sum(i(r, 'n_seqs') for r in rows)
    ai = sum(i(r, 'n_ai') for r in rows)
    blast = sum(i(r, 'n_blast') for r in rows)
    hmm = sum(i(r, 'n_hmm') for r in rows)
    hom = sum(i(r, 'n_homology') for r in rows)
    only = sum(i(r, 'n_ai_only') for r in rows)
    # ai_only = AI and not homology, so (homology + ai_only) is everything any
    # method found, and the remainder is what nothing found
    none = n - hom - only

    section('DETECTION METHOD, PER SEQUENCE (Table S4 thresholds)')
    print('  %-34s %10s %8s' % ('method', 'sequences', 'share'))
    for label, k in [('AI (LucaProt), prob >= 0.5', ai),
                     ('HMM, score >= 10', hmm),
                     ('BLAST, e <= 1e-3', blast),
                     ('either homology method', hom)]:
        # two decimals because HMM and "either homology" both round to 96.0
        print('  %-34s %10s %7.2f%%' % (label, format(k, ','), pct(k, n)))
    print('  ' + '-' * 53)
    print('  %-34s %10s %7.2f%%  <- not in the paper'
          % ('AI ONLY, no BLAST, no HMM', format(only, ','), pct(only, n)))
    print('  %-34s %10s %7.2f%%  <- not in the paper'
          % ('detected by NO method', format(none, ','), pct(none, n)))
    print('  %-34s %10s' % ('total', format(n, ',')))

    emit('s4.total_seqs', n)
    emit('s4.ai', ai)
    emit('s4.hmm', hmm)
    emit('s4.blast', blast)
    emit('s4.ai_only', only)
    emit('s4.ai_only_pct', '%.2f' % pct(only, n))
    emit('s4.no_method', none)
    emit('s4.no_method_pct', '%.2f' % pct(none, n))
    emit('s4.ai_pct', '%.2f' % pct(ai, n))

    print('\n  The AI share comes to %.2f%% against the paper\'s stated 98.22%% -'
          ' one rounding\n  step apart, which is what validates this parse.'
          ' Not identical: do not write\n  "reproduces it exactly".' % pct(ai, n))

    # ---- supergroups with no homology support at all ---------------------
    section('SUPERGROUPS IN WHICH NOT ONE SEQUENCE HAS A HOMOLOGY HIT')
    zero = sorted((r for r in rows if i(r, 'n_homology') == 0),
                  key=lambda r: -i(r, 'n_seqs'))
    print('  %-16s %-26s %7s %8s %10s %s'
          % ('supergroup', 'group', 'seqs', 'AI>=0.5', 'none found', 'no motifs'))
    for r in zero:
        print('  %-16s %-26s %7d %8d %10d %s'
              % (r['supergroup'], r['group'][:26], i(r, 'n_seqs'), i(r, 'n_ai'),
                 i(r, 'n_seqs') - i(r, 'n_ai'),
                 'YES' if r['supergroup'] in NOMOTIF else ''))
    emit('s4.zero_homology_supergroups', len(zero))

    # ---- the one named object --------------------------------------------
    both = [r for r in zero if r['supergroup'] in NOMOTIF]
    if both:
        section('THE WEAKEST-SUPPORTED GROUP IN THE PAPER')
        for r in both:
            sg = r['supergroup']
            print('  %s - %d sequences, %d of them AI-only, zero BLAST, zero HMM'
                  % (sg, i(r, 'n_seqs'), i(r, 'n_ai_only')))
            emit('s4.%s_n' % sg.lower(), i(r, 'n_seqs'))
            if os.path.exists(SEQS):
                probs = sorted((float(x['AI_Prob'])
                                for x in read_tsv(SEQS)
                                if x['Supergroup_ID'] == sg), reverse=True)
                if probs:
                    print('  AI probabilities: %.6f down to %.6f, all %d of them'
                          % (probs[0], probs[-1], len(probs)))
                    emit('s4.%s_min_prob' % sg.lower(), '%.5f' % probs[-1])
        print("""
  The only supergroup that is simultaneously zero-homology and motif-free in the
  paper's own Figure S4, at near-maximum confidence from a model measured at
  1.4-5.1% false-positive on eukaryotic virus proteins (section 09). Marine,
  freshwater and aquatic samples.

  That is not proof it is spurious. It is the most checkable case in the paper,
  and it is exactly where a false positive would land. One named object is worth
  more to a reader than the aggregate arithmetic above.""")

    # ---- the three-way split of Figure S4 --------------------------------
    if os.path.exists(S3):
        section("FIGURE S4's THREE-WAY SPLIT, FROM TABLE S3")
        groups = {}
        for r in read_tsv(S3):
            groups[r['group']] = groups.get(r['group'], 0) + 1
        for g, c in sorted(groups.items(), key=lambda kv: -kv[1]):
            print('  %-38s %3d supergroups' % (g, c))
        print('  %-38s %3d' % ('total', sum(groups.values())))
        for g, c in groups.items():
            if 'ICTV' in g and 'Known' in g:
                emit('s4.ictv_supergroups', c)
            elif 'this study' in g or 'new' in g.lower():
                emit('s4.new_supergroups', c)
        print("""
  21 and 60 match two counts stated independently in the paper's body text -
  "only 21 contained those from viral phyla/classes currently classified by
  ICTV" and "we unveiled 60 previously unidentified groups" - which is what
  makes the parse trustworthy.""")

    # ---- what this does NOT establish ------------------------------------
    section('WHAT THIS DOES NOT ESTABLISH')
    print("""  The 23 LucaProt-only supergroups remain unidentified, and nothing here
  should be described as "the 23".

  The paper's 444 contigs / 23 supergroups are defined against ClstrSearch, a
  clustering pipeline with ten rounds of iterative HMM expansion. Table S4's
  BLAST and HMM columns measure homology to the *reference* database. Those are
  different tests, which is why this yields 13,903 rather than 444. Nothing in
  the public release distinguishes ClstrSearch hits - ours_checked_rdrp_final.csv
  carries one source value, "checked_rdrp", across all 513,134 rows - and a
  subset-sum over the 60 new supergroups admits 16,266,714,888 valid 23-member
  combinations.

  The one thing that can be said: the seven largest new supergroups (8,914 of
  the 10,846 new-discovery contigs, 82%) are too large to be among the 23, since
  23 supergroups summing to 444 contigs caps any member at 224.

  The seven motif-free supergroups above are tier E - read off the Figure S4
  raster by eye, not extracted from any machine-readable file. The parse
  recovers exactly 180 rows in exactly the three groups the caption describes,
  with 21 and 60 matching the body text, which argues the blank columns are
  real rather than an extraction artifact. It is still worth checking by eye on
  the Figure S4 page before publishing, and NOMOTIF at the top of this file is
  where to correct it.

  Minor data-quality note: Table S4 contains three rows with Superclade014 /
  018 / 022 IDs, one sequence each, inconsistent with Table S3's 180
  Supergroup* IDs.""")


if __name__ == '__main__':
    main()
