# Reproducing every number in the post

Note on AI use: all of the code for this project was written with the help of claude code with Opus 5. Design, ideas and mistakes are my own. 

The post argues that LucaProt's evaluation doesn't measure what it's cited for.
A post making that argument shouldn't ask you to take its own numbers on trust.

So: `claims.tsv` lists every figure that appears in the post. Twelve scripts
recompute them from primary sources. `run_all.py` runs the scripts and diffs the
results against the registry. The scripts never read `claims.tsv` and the
registry knows nothing about how anything is computed, so agreement between them
is a real check rather than a tautology.

```bash
python reproduce/run_all.py            # everything, ~15 min
python reproduce/run_all.py --fast     # stdlib-only scripts, ~1 min
python reproduce/run_all.py --only 03  # one script, with its full explanation
```

Current state, on the machine this was written on:

```
126 OK (3 within tolerance)   0 MISMATCH   0 MISSING   46 EXTRA
```

`MISMATCH` is the row that matters — it means the post and the code disagree
about a number, and one of them is wrong. It should always be zero.
`EXTRA` is a value a script computes that the post doesn't quote; those are
supporting figures, and they're informational, not failures.

## What each script does

Run them individually — each one prints its reasoning, not just its numbers.
The docstring at the top of each file is the long-form explanation.

| script | what it establishes | needs |
|---|---|---|
| `01_paper_claims.py` | Every figure the post attributes to Hou et al. actually appears in the paper. Then the false-positive arithmetic: the paper's own 0.014% against its own 144.6M proteins. | `paper.pdf`, `pdftotext` |
| `02_split_composition.py` | All 10,000 `rt` and 2,000 `dna` rows are in training and none in test. 30 of 540 test positives are byte-identical to a training sequence. | released splits |
| `03_homology_leakage.py` | Every one of the 540 test positives has a detectable training homolog. Identity distribution, and the DIAMOND-only baseline that beats LucaProt on its own test set. | committed DIAMOND output |
| `04_how_the_split_was_built.py` | The mechanism behind 02, quoted from the authors' own preprocessing script and its captured stdout. | vendored LucaProt source |
| `05_orphan_benchmark.py` | LucaProt scores 97.73% on 2,874 published RdRPs with zero homology to its training set. Recall holds; the claim is right. | committed predictions |
| `06_probe_and_controls.py` | A 2,561-parameter logistic regression on the same frozen embeddings matches it, and four controls rule out the obvious bugs. | embeddings (~500 MB), sklearn |
| `07_model_parameters.py` | 84.4% of the trainable parameters are a head over ESM2; the bespoke transformer is ~1.1M. | checkpoint, torch |
| `08_composition_baseline.py` | 400 dipeptide counts reach AUC 0.9966 on the test set while finding 3.76% of real novel RdRPs. | manifest + FASTAs, xgboost |
| `09_false_positive_panel.py` | The measured false-positive rate: 0 in 28,350 prokaryotic and phage proteins, 1.4–5.1% on eukaryotic viruses. What it fires on, and the three extrapolations it does not support. | committed predictions + set metadata |
| `10_motif_check.py` | The paper's own motif criterion applied to all 513,134 of its claimed RdRPs, with a positive control and a null. Comes back clean. | 408 MB figshare CSV |
| `11_table_s4_ai_only.py` | 2.71% of the published set is AI-only and 1.25% was called by no method; seven supergroups have no homology support, one of which is also motif-free. | committed per-supergroup aggregate |
| `12_xgboost_baseline.py` | The authors' own unreported gradient-boosting baseline, configured from their shell script and config file: 50 rounds as shipped, 97.49% run to convergence. | embeddings (~500 MB), xgboost |

The dependency order is 01–05 and 09, 11 (cheap, mostly stdlib) then 06–08, 10, 12
(need a scientific stack or large local artifacts). `--fast` runs the first
group.

## Why the argument needs all twelve

The sections build on each other and are individually insufficient:

- **02 and 03** show the test set can't measure detection beyond homology. That
  is a claim about the *evidence*, not about the model.
- **04** shows this wasn't a subtle statistical accident — the hard negatives
  were appended to training and the test file was copied through untouched.
- **05** is the constructive half, and it cuts against the criticism: build the
  missing evaluation and LucaProt passes it. The paper's central claim is
  correct; it just was never demonstrated by the figure it's cited to.
- **06, 07, 08** are a separate argument: the capability is real, but it belongs
  to ESM2 rather than to the architecture, and the test set can be cleared by
  three unrelated shortcuts.
- **09** is the other half of 05. Recall survives the missing evaluation;
  precision is where it doesn't, and the failures are taxonomic rather than the
  polymerase confusions anyone would have predicted.
- **10** is 05's counterpart for the published output: the check that could have
  been the strongest result in the post, run against controls, reported as the
  clean result it turned out to be.
- **11** sizes what 09's rate could be acting on, and names the one supergroup
  where every weakness in the paper coincides.
- **12** closes the architecture argument. 06 shows a linear probe matches
  LucaProt; 12 shows the authors built that comparison themselves, in a
  configuration that could not converge, and published none of it.

Sections 05 and 10 are the two to read first if you only read two. They're the
parts that could have gone the other way, and one of them did.

## Inputs and where they come from

| path | what | in git |
|---|---|---|
| `linear_probe/data/{train,test}.csv` | LucaProt's released splits, 190,846 + 22,284 rows | no, 133 MB |
| `analysis/test_vs_train.tsv` | DIAMOND blastp output, 540 test positives vs 4,900 training positives | yes |
| `results/{orphans,control_pos,control_neg}/` | per-sequence LucaProt predictions | yes |
| `no_training_homolog.faa` | the 2,874-sequence zero-homology benchmark | yes |
| `linear_probe/embeddings/` | frozen ESM2-3B layer-36 vectors, 37,248 sequences | no, 505 MB |
| `linear_probe/inputs/` | FASTAs + manifest, from `prepare_inputs.py` | yes |
| `fp_panel/preds/*.csv` | per-sequence LucaProt scores for the 53,709-sequence panel | yes |
| `fp_panel/sets/*.{faa,meta.tsv}` | the screened panel itself, with length and identity-to-splits per sequence | yes |
| `fp_panel/table_s{3,4}_supergroups.tsv` | per-supergroup aggregates of Tables S3 and S4 | yes |
| `fp_panel/lucaprot_only_sequences.tsv` | the 13,903 AI-only rows of Table S4 | yes |
| `fp_panel/motif/ours_checked_rdrp_final.csv` | the 513,134 claimed RdRPs | no, 408 MB |
| `LucaProt/` | the authors' released repo, vendored | submodule |
| `paper.pdf` | Hou et al. 2024 | no, not ours to redistribute |

The two split CSVs and `ours_checked_rdrp_final.csv` all come from the paper's
figshare release,
[doi 10.6084/m9.figshare.26298802.v14](https://doi.org/10.6084/m9.figshare.26298802.v14).
Drop the splits in `linear_probe/data/` and 02 runs; drop the claimed-RdRP CSV in
`fp_panel/motif/` and 10 runs. `mmc3.xlsx` and `mmc4.xlsx` are the paper's
supplementary Tables S3 and S4; `fp_panel/parse_table_s{3,4}.py` reduce them to
the committed TSVs that 11 reads, so 11 runs without them.

Embeddings are regenerated with `modal run linear_probe/embed_modal.py` — a GPU
job costing roughly $4–6. **Smoke-test it first**: `--sets control_neg` is 250
sequences and a few cents, and exercises image build, GPU, batching, volume
write and download. Skipping that step cost about $4 in failed runs here.

The false-positive panel is regenerated the same way and costs a great deal more
— about €30 for 53,709 sequences on 8×A100-40GB, which is why the predictions are
committed rather than recomputed. `fp_panel/RESULTS.md` documents the full
pipeline, the four Modal build failures that preceded the first working run, and
why residues rather than sequence count track the bill.

Anything absent makes its script exit 3 with a message naming the missing file.
`run_all.py` reports that as `skipped` and carries on, so a partial checkout
still verifies everything it can.

## Tiers, and the three numbers that aren't checked here

`claims.tsv` tags every claim with how it's obtained:

- **P** quoted from the paper, verified present in its extracted text
- **A** arithmetic on paper-reported values
- **D** computed from the released data
- **S** read out of the authors' released source code
- **R** computed from our own model runs
- **E** eyeballed off a raster figure panel
- **M** run by hand

Four claims are **tier E** and cannot be automated: accuracy 0.999955, AUC
0.99999991, six-of-ten cross-validation folds at exactly 1.0, and the **0.023%**
false-positive rate. The first three live inside the image of Figure S2 and the
fourth inside Figure 3A, not in any extractable text, so all four were read off
the panels by eye. `run_all.py` skips them rather than pretending to check them,
but it does check the arithmetic done on them — `fp_fig3a.expected` is 0.023% ×
144.6M = 33,258. If you're verifying the post, those four are the ones to check
yourself, on pages 23 and 8.

There is also one claim nothing here can check at all: **"all of the inference I
needed for this project ran me 38 euros on Modal."** That is a billing figure,
not a computed one. `fp_panel/RESULTS.md` records ~€30 for the false-positive
panel and the embedding job adds $4–6, so it is consistent, but the repo cannot
verify it and does not try.

## What `[cut]` means in `claims.tsv`

The post was rewritten shorter late on, and about a third of the registry is
now marked `[cut]` in its `where` column: values that were quoted in an earlier
draft, still compute correctly, and are no longer in the published text. The
DIAMOND e-value breakdown, the byte-identical duplicate counts, the control
tables, the dipeptide baseline and the parameter counts are all in this
category.

They are kept rather than deleted, because they're the evidence behind
sentences the post still makes qualitatively, and because a number that leaves
the post can come back. But the distinction matters when something fails: a
`MISMATCH` on a live row means the post is wrong, while a `MISMATCH` on a
`[cut]` row means the underlying data moved and the post never mentioned it.

A fourth thing is eyeballed but isn't in `claims.tsv` because the post doesn't
quote it yet: which supergroups have blank motif columns in Figure S4. Script 11
hard-codes the seven at the top of the file and says so. The parse recovering
exactly 180 rows in exactly the three groups the caption describes, with 21 and
60 matching the body text, is what argues the blanks are real — but it is a
raster figure, and if you're going to publish the Supergroup193 argument, look
at the page.

## The three tolerances

Everything is compared exactly, except three rows that carry a `tol` in
`claims.tsv`. Each is a genuinely unstable quantity, and the instability is the
reason rather than an excuse:

| claim | post | typical | why it moves |
|---|---|---|---|
| `ctrl.shuffled` | 0.03% | 0.00% | best-of-ten label permutations; different seeds land on 0 or 1 sequences out of 2,874 |
| `dims.l1_count` | 14 | 13–14 | one coefficient sits exactly on the L1 boundary; the count flips with array memory layout and sklearn version |
| `dims.l1_recall` | 95.34% | 95.27–95.37% | follows from the above |

No other claim is permitted to drift. If a fourth starts needing a tolerance,
that is a finding about the result, not a reason to widen the check.

## Known gaps

**The homology baseline's precision is not reproduced.** 03 shows DIAMOND alone
gets 540/540 recall, which needs only the positives and is solid. Counting its
false positives needs DIAMOND run over the 21,744 test negatives, and that
output (`analysis/testall_vs_trainpos.tsv`) is missing — so `analysis/baseline.py`
and `analysis/sweep.py` don't run. The post claims recall only, deliberately.

**DIAMOND isn't installed on this machine**, so `analysis/test_vs_train.tsv` is
committed rather than regenerated. It was produced with `--ultra-sensitive`, not
defaults; the post says so, and so does the docstring in 03. Whether the 100%
survives at default sensitivity is untested.

**The orphan benchmark is 2,380 of 2,874 scored.** The laptop run was stopped
early. Order was randomised beforehand (`shuffle_inputs.py`, seed 20260806), so
the completed prefix is an unbiased sample and the Wilson interval already
accounts for the smaller n.

**48,490 is not diagnosable.** The paper states that many reverse transcriptases
among the negatives; the release contains 10,000. 04 shows every RT count that
exists anywhere in the pipeline and none of them is 48,490. The source FASTAs
that would settle it aren't in the released repo. The post documents this and
asserts nothing about which is correct.

**The panel's RdRP exclusion is homology-based, not domain-aware.** A panel
sequence is "not an RdRP" because DIAMOND finds no hit in an 8,024-sequence RdRP
reference. That is broader than the paper's own 5,979-sequence curated database,
so exclusion is conservative — but it is not the same as "contains no RdRP
domain", and a sufficiently divergent RdRP would slip through. The top firing
hits are helicases and parvovirus NS1 initiators, which are definitively not
RdRPs, so the headline survives; exact per-class rates want a palmprint or Pfam
check they haven't had.

**`dev.csv` was unavailable**, so the panel's split-exclusion covers train +
test only, 213,130 of 235,414 rows. The validation split drove checkpoint
selection rather than gradient updates, so its leakage risk is lower — not zero.
This affects 09 and, through the null set, 10.

**The 23 LucaProt-only supergroups cannot be identified from public data.** 11
computes 13,903 AI-only *sequences*, which is a different quantity: the paper's
444/23 are defined against ClstrSearch, and nothing in the release distinguishes
ClstrSearch hits. A subset-sum over the 60 new supergroups admits 16,266,714,888
valid combinations. Never write "7 of the 23" — the seven motif-free supergroups
are 7 of the 60 newly discovered ones.

**Four panel classes were specified and never run**: `nrps_bact` and `pks_bact`
(soil-abundant megasynthases, ~3,000 sequences, all ≥2,500 aa), `short_archaea`
and `short_euk`. The screened FASTAs are committed, so they cost GPU time and
nothing else. Archaea appears in 10's null set but has no LucaProt scores.

**No deployment-wide false-positive rate is offered, deliberately.** It would
need the eukaryotic-virus protein fraction of the 144.6M-protein corpus.
`fp_panel/corpus_length_hist.py` measures the length distribution from the
authors' own release but not the taxonomic one, and the corpus prefix it reads
is a single ecosystem. 09 prints the retracted estimate that assumed otherwise.

## Relationship to the rest of the repo

`analysis/`, `linear_probe/`, `composition_baseline/`, `occlusion/` and
`fp_panel/` are the original exploratory work, and they compute more than the
post uses — layer sweeps, weight histograms, bootstrap selection stability,
per-source false positive rates, an occlusion scan over catalytic motifs, a
length-vs-FPR curve, the deployment corpus histogram, the contradictory-label
finding in LucaProt's own negatives.

`reproduce/` is the audited subset: only what the post claims, condensed into
stdlib-where-possible scripts that explain themselves. Where the two overlap
they agree, and getting them to agree exposed one thing worth recording — every
upstream fit uses `class_weight='balanced'`, and dropping it moves orphan recall
from 97.74% to 97.77% and the top-10-dimension figure from 29.44% to 18.89%.
Small print in an upstream script, several points of difference downstream.
Reimplementing from the description rather than the code would have quietly
produced different numbers.
