"""Shared helpers for the reproduce/ scripts.

Every script here prints ordinary human-readable output *and* one machine-readable
line per claim:

    CLAIM<TAB>claim.id<TAB>value

`run_all.py` collects those lines and diffs them against `claims.tsv`, which is
the registry of every number that appears in the blog post. That is the whole
mechanism: the scripts do not know what the post says, and the registry does not
know how the numbers are computed, so agreement between them is meaningful.
"""
import csv, math, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, '.cache')

csv.field_size_limit(10 ** 9)


def emit(claim_id, value):
    """Record a computed value under a claim id."""
    print('CLAIM\t%s\t%s' % (claim_id, value))


def section(title):
    print('\n' + title)
    print('-' * len(title))


def need(path, why):
    """Exit with a useful message rather than a traceback when an input is absent."""
    if not os.path.exists(path):
        print('SKIP\tmissing input: %s' % path)
        print('      %s' % why)
        sys.exit(3)
    return path


def clean_seq(seq):
    """LucaProt's own src/utils.py:clean_seq - keep A-Z except J.

    Applied everywhere a sequence is compared, so that duplicate counts are over
    exactly the strings the model was fed rather than over the raw CSV field.
    """
    return ''.join(c for c in seq.upper() if 'A' <= c <= 'Z' and c != 'J')


def load_split(name):
    """Read one of LucaProt's released splits -> [(prot_id, seq, label, source)]."""
    path = os.path.join(ROOT, 'linear_probe', 'data', name + '.csv')
    need(path, 'copy the figshare splits into linear_probe/data/ '
               '(doi 10.6084/m9.figshare.26298802.v14)')
    rows = []
    with open(path, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append((r['prot_id'], clean_seq(r['seq'].strip()),
                         int(r['label']), r['source']))
    return rows


def wilson(k, n, z=1.96):
    """Wilson score interval, as a percentage. Used instead of the normal
    approximation because several counts here are 0 or n, where the normal
    interval has zero width and is simply wrong."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100.0 * max(0.0, c - h), 100.0 * min(1.0, c + h)


def pct(k, n):
    return 100.0 * k / n if n else 0.0
