#!/usr/bin/env python
"""What the 23.7M trainable parameters are actually spent on.

The post says 84.4% of LucaProt's trainable parameters are the head sitting on
top of ESM2's output, and the bespoke from-scratch transformer is ~1.1M. That
matters to the argument: if most of the trainable capacity is a pooling head
over a frozen language model, then "a logistic regression on the same
embeddings matches it" stops being surprising and starts being the expected
result.

This reads the released checkpoint's state dict directly with torch.load and
buckets tensors by name prefix. It does not instantiate the model class, so it
needs torch but nothing from LucaProt's source tree.

Checkpoint: LucaProt/models/rdrp_40_extend/protein/binary_class/sefn/
            20230201140320/checkpoint-100000/pytorch_model.bin

Note the 3B ESM2 backbone is not in this file and not counted here - it is
frozen and loaded separately from torch hub. 23.7M is the trainable part.

    python reproduce/07_model_parameters.py
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ROOT, emit, need, section

CKPT = os.path.join(ROOT, 'LucaProt', 'models', 'rdrp_40_extend', 'protein',
                    'binary_class', 'sefn', '20230201140320',
                    'checkpoint-100000', 'pytorch_model.bin')

# Tensor-name prefix -> which channel it belongs to. The names are unambiguous
# in this checkpoint: `embedding_*` is the value-attention pooler and projection
# over ESM2's frozen output, `seq_encoder.*` is the from-scratch BPE transformer.
# Anything unmatched lands in 'everything else', which is printed, so a silent
# misattribution cannot hide.
BUCKETS = [
    ('embedding_pooler', 'ESM2 projection head'),
    ('embedding_linear', 'ESM2 projection head'),
    ('seq_encoder',      'from-scratch BPE transformer'),
]


def bucket_of(name):
    for pfx, label in BUCKETS:
        if name.startswith(pfx):
            return label
    return 'everything else'


def main():
    need(CKPT, 'the released checkpoint; see LucaProt/README.md for the download')
    try:
        import torch
    except ImportError:
        print('SKIP\tneeds torch')
        sys.exit(3)

    print('loading checkpoint (~100 MB) ...', end='', flush=True)
    sd = torch.load(CKPT, map_location='cpu', weights_only=True)
    if hasattr(sd, 'state_dict'):
        sd = sd.state_dict()
    print(' %d tensors' % len(sd), flush=True)

    totals, total = {}, 0
    for name, t in sd.items():
        try:
            n = int(t.numel())
        except AttributeError:
            continue
        totals[bucket_of(name)] = totals.get(bucket_of(name), 0) + n
        total += n

    section('TRAINABLE PARAMETERS BY CHANNEL')
    print('  %-32s %12s %8s' % ('component', 'params', 'share'))
    for label, n in sorted(totals.items(), key=lambda kv: -kv[1]):
        print('  %-32s %12d %7.1f%%' % (label, n, 100.0 * n / total))
    print('  %-32s %12d %7.1f%%' % ('TOTAL', total, 100.0))

    emit('model.params_total', total)
    for label, n in totals.items():
        if 'ESM2' in label:
            emit('model.esm2_head_params', n)
            emit('model.esm2_head_pct', '%.1f' % (100.0 * n / total))
        elif 'from-scratch' in label:
            emit('model.scratch_params', n)
            emit('model.scratch_pct', '%.1f' % (100.0 * n / total))

    # How much of the "from-scratch transformer" is actually a transformer: the
    # BPE word-embedding matrix is a lookup table, not computation.
    lut = int(sd['seq_encoder.embeddings.word_embeddings.weight'].numel())
    scratch = totals.get('from-scratch BPE transformer', 0)
    print('\n  of the from-scratch channel, the BPE lookup table is %d params,'
          '\n  leaving %d of actual transformer (4 layers, 128-dim, 4 heads).'
          % (lut, scratch - lut))
    emit('model.bpe_lookup_params', lut)
    emit('model.scratch_transformer_params', scratch - lut)

    print("""
  So the bespoke "sequence channel" the paper's architecture section is named
  for is ~1.1M parameters of real transformer, against ~20M of pooling head over
  a frozen language model. The 3B ESM2 backbone is not counted here - it is not
  in this file and it is not trained.

  This is the context for the probe result in 06: when 84% of the trainable
  capacity is a head over ESM2 embeddings, a linear model on those same
  embeddings matching the whole thing is close to the expected outcome.""")


if __name__ == '__main__':
    main()
