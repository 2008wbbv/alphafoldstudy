# AlphaFold viral vs cellular accuracy study

Measures AlphaFold2 prediction accuracy on viral versus human cellular proteins
by scoring AlphaFold models against experimental PDB structures (TM-score, RMSD,
pLDDT, predicted disorder, and sequence coverage).

## Quickstart

    git clone https://github.com/2008wbbv/alphafoldstudy.git
    cd alphafoldstudy
    bash setup_env.sh

`setup_env.sh` installs Miniconda if the machine does not already have conda,
builds the environment from `environment.yml`, and verifies every import. It is
safe to re-run.

Then, in any new shell:

    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate alphafold-study
    python src/af_study.py

Full instructions, including cluster setup and troubleshooting, are in
[INSTALL.txt](INSTALL.txt).

## Layout

    src/                   pipeline and tooling (all run from the repo root)
      build_registry.py    select a balanced protein set from RCSB -> proteins.csv
      af_study.py          download, metrics, per-residue, stats, figures, regression
      compare_metrics.py   box chart of TM-score, pLDDT and disorder + Mann-Whitney
      make_batches.py      split proteins.csv into HPC job batches -> jobs/
      merge_results.py     combine results/batch_*/ outputs into single tables
      extra_stats.py       extra robustness tests appended to results/stats.txt
      make_captions.py     figure captions from existing results
    jobs/                  generated SLURM batch files (batch_NN.csv + batch_NN.sh)
    environment.yml        conda environment specification
    setup_env.sh           one-command installer
    INSTALL.txt            plain text setup and troubleshooting guide
    proteins.csv           the protein registry the jobs are built from

Not tracked (regenerated locally): `data/`, `results/`, `logs/`

## Run (from the repo root)

    python src/build_registry.py     # rebuild proteins.csv from RCSB
    python src/af_study.py           # full study on proteins.csv -> results/

Post-hoc analysis on existing results:

    python src/compare_metrics.py
    python src/extra_stats.py
    python src/make_captions.py

## HPC batches

    python src/make_batches.py       # proteins.csv -> jobs/batch_NN.csv + .sh

Each job processes one batch into its own `results/batch_NN/` folder. After the
jobs finish, recombine them:

    python src/merge_results.py      # -> results/combined_metrics.csv, ...

Before submitting on a cluster:

  1. Pre-download on a login node, since compute nodes are usually offline:

         bash setup_env.sh --download

     The jobs reuse `./data` and never re-download.

  2. In each `jobs/batch_NN.sh`, fill in the account and partition, and
     uncomment the two conda activation lines.

All commands are run from the repo root.

## Method in one paragraph

Proteins are selected programmatically from RCSB rather than by hand, filtered
by resolution and length, deduplicated by UniProt accession, and balanced across
four predicted-disorder bins. Entries whose parent UniProt sequence exceeds the
AlphaFold single-model limit are excluded, because those are served as fragments
that may not contain the crystallised region at all; a sequence coverage check
during analysis catches any that slip through. Each experimental chain is
superposed onto its AlphaFold model with TM-align, and the resulting TM-score,
RMSD, mean pLDDT, disorder fraction and coverage are compared between the two
groups, both per protein and per residue.
