#!/usr/bin/env python
"""Run every reproduction script and diff the results against claims.tsv.

claims.tsv is the registry of every number that appears in the blog post. Each
script emits CLAIM lines for what it computes. This driver runs them all, joins
the two, and prints one table:

    OK        computed value matches the post
    MISMATCH  the post and the code disagree - one of them is wrong
    MISSING   a claim in the post that nothing computed (input unavailable)
    EXTRA     a computed value with no corresponding claim in the post

MISMATCH is the row that matters. The scripts do not read claims.tsv and
claims.tsv does not know how anything is computed, so a match is a real check
rather than a tautology.

    python reproduce/run_all.py              # everything, ~8 min
    python reproduce/run_all.py --fast       # only the stdlib scripts, ~1 min
    python reproduce/run_all.py --only 03    # one script, full output
"""
import argparse, csv, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
CLAIMS = os.path.join(HERE, 'claims.tsv')

# script, one-line description, roughly how long, whether --fast skips it
SCRIPTS = [
    ('01_paper_claims.py', 'numbers quoted from the paper + FP arithmetic', 'instant', False),
    ('02_split_composition.py', 'where the hard negatives went; exact duplicates', '~40 s', False),
    ('03_homology_leakage.py', 'test-vs-train homology, the core measurement', 'instant', False),
    ('04_how_the_split_was_built.py', 'the preprocessing script that made the split', 'instant', False),
    ('05_orphan_benchmark.py', 'LucaProt on 2,874 zero-homology RdRPs', 'instant', False),
    ('06_probe_and_controls.py', 'logistic probe vs LucaProt, plus four controls', '~4 min', True),
    ('07_model_parameters.py', 'what the 23.7M trainable parameters are spent on', '~30 s', True),
    ('08_composition_baseline.py', 'dipeptide counts: high test AUC, no real detection', '~1 min', True),
    ('09_false_positive_panel.py', 'measured FPR on 53,709 held-out non-RdRPs', '~5 s', False),
    ('10_motif_check.py', 'catalytic motifs in their 513,134 claimed RdRPs', '~30 s', True),
    ('11_table_s4_ai_only.py', 'how much of the virosphere rests on the AI alone', 'instant', False),
    ('12_xgboost_baseline.py', "their own unreported baseline, run to convergence", '~4 min', True),
]


def load_claims():
    """claim id -> (value, tolerance, tier, where it appears in the post)."""
    out = {}
    with open(CLAIMS, encoding='utf-8') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            cid = (r.get('id') or '').strip()
            if not cid or cid.startswith('#'):
                continue
            out[cid] = ((r.get('value') or '').strip(),
                        (r.get('tol') or '').strip(),
                        (r.get('tier') or '').strip(),
                        (r.get('where') or '').strip())
    return out


def same(a, b, tol=''):
    """Compare as numbers when both parse, else as strings. Exact by default: a
    claim written as 97.73 must be computed as 97.73, not 97.7. A tolerance
    applies only where claims.tsv sets one and the README says why."""
    a, b = a.strip(), b.strip()
    if a == b:
        return True
    try:
        return abs(float(a) - float(b)) <= (float(tol) if tol else 1e-9)
    except ValueError:
        return False


def run(script, verbose):
    t0 = time.time()
    p = subprocess.run([sys.executable, os.path.join(HERE, script)],
                       capture_output=True, text=True, encoding='utf-8',
                       errors='replace')
    out = (p.stdout or '') + (p.stderr or '')
    computed = {}
    for line in out.splitlines():
        if line.startswith('CLAIM\t'):
            parts = line.split('\t')
            if len(parts) >= 3:
                computed[parts[1]] = parts[2]
    if verbose:
        print('\n'.join(l for l in out.splitlines()
                        if not l.startswith('CLAIM\t')))
    return computed, p.returncode == 3, out, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fast', action='store_true',
                    help='skip the scripts needing embeddings / torch / xgboost')
    ap.add_argument('--only', help='run one script by number, e.g. 03')
    args = ap.parse_args()

    claims = load_claims()
    todo = SCRIPTS
    if args.only:
        todo = [s for s in SCRIPTS if s[0].startswith(args.only)]
        if not todo:
            sys.exit('no script matching %r' % args.only)
    elif args.fast:
        todo = [s for s in SCRIPTS if not s[3]]

    print('reproducing %d of %d scripts against %d claims\n'
          % (len(todo), len(SCRIPTS), len(claims)))

    computed, skipped = {}, []
    for i, (script, what, cost, _) in enumerate(todo, 1):
        print('[%d/%d] %-32s %-9s %s' % (i, len(todo), script, cost, what),
              flush=True)
        got, was_skipped, out, el = run(script, verbose=bool(args.only))
        if was_skipped:
            reason = next((l for l in out.splitlines() if l.startswith('SKIP')),
                          'SKIP')
            print('        %s' % reason.replace('SKIP\t', 'skipped: '))
            skipped.append(script)
        else:
            print('        %d values in %.0fs' % (len(got), el))
        computed.update(got)

    # ---- the comparison --------------------------------------------------
    print('\n' + '=' * 78)
    print('%-32s %13s %13s  %s' % ('claim', 'in post', 'computed', 'status'))
    print('=' * 78)

    ok = bad = missing = drift = 0
    for cid, (want, tol, tier, where) in sorted(claims.items()):
        if cid not in computed:
            if tier in ('E', 'M'):    # eyeballed / manual: never computed here
                continue
            missing += 1
            print('%-32s %13s %13s  MISSING' % (cid, want, '-'))
            continue
        got = computed[cid]
        if want == got.strip():
            ok += 1
        elif same(want, got, tol):
            ok += 1
            drift += 1
            print('%-32s %13s %13s  ok, within tol %s' % (cid, want, got, tol))
        else:
            bad += 1
            print('%-32s %13s %13s  MISMATCH  %s' % (cid, want, got, where))

    extra = sorted(c for c in computed if c not in claims)
    for c in extra:
        print('%-32s %13s %13s  EXTRA' % (c, '-', computed[c]))

    print('=' * 78)
    print('%d OK (%d within tolerance)   %d MISMATCH   %d MISSING   %d EXTRA'
          % (ok, drift, bad, missing, len(extra)))
    if skipped:
        print('skipped: %s' % ', '.join(skipped))
    manual = sum(1 for v in claims.values() if v[2] in ('E', 'M'))
    if manual:
        print('%d claim(s) are tier E/M - read off a raster figure or run by '
              'hand; see README' % manual)

    if bad:
        print('\nMISMATCH means the post and the code disagree about a number.'
              '\nOne of them is wrong. Do not publish until this is empty.')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
