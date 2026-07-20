# AlphaFold viral vs cellular accuracy study

Measures AlphaFold2 prediction accuracy on viral versus human cellular proteins
by scoring AlphaFold models against experimental PDB structures (TM-score, RMSD,
pLDDT, predicted disorder, and sequence coverage).

## Layout

    src/                   pipeline and tooling (all run from the repo root)
      build_registry.py    select a balanced protein set from RCSB -> proteins.csv
      af_study.py          download, metrics, per-residue, stats, figures, regression
      make_batches.py      split proteins.csv into HPC job batches -> jobs/
      merge_results.py     combine results/batch_*/ outputs into single tables
      extra_stats.py       extra tests appended to results/stats.txt
      make_captions.py     figure captions from existing results
    jobs/                  generated SLURM batch files (batch_NN.csv + batch_NN.sh)
    proteins.csv           the committed protein registry the jobs are built from

Not tracked (regenerated locally): data/, results/, logs/

## Run (from the repo root)

    python src/build_registry.py     # rebuild proteins.csv from RCSB
    python src/af_study.py           # full study on proteins.csv -> results/

Post-hoc analysis on existing results:

    python src/extra_stats.py
    python src/make_captions.py

## HPC batches

    python src/make_batches.py       # proteins.csv -> jobs/batch_NN.csv + .sh

Each job processes one batch into its own results/batch_NN/ folder. After the
jobs finish, recombine them:

    python src/merge_results.py      # -> results/combined_metrics.csv, ...

Before submitting on a cluster:

  1. Fill in account, partition, and the module/venv lines in each
     jobs/batch_NN.sh (they ship as safe placeholders).
  2. Compute nodes are usually offline. Pre-download the structures on a login
     node first (run af_study.py once so ./data is populated on shared storage),
     since the jobs reuse ./data and do not re-download.

All commands are run from the repo root.
