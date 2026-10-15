# Delivery validation

This document records software checks, not Galaxy10 scientific results.

- Environment: Linux, Python 3.12, torch 2.14.1+cpu, torchvision 0.29.1+cpu,
  numpy 2.3.5, h5py 3.16.0, matplotlib 3.10.8.
- Downloaded official ResNet18 ImageNet weights and executed frozen forward
  inference with the actual full-frame 224 x 224 preprocessing; output was
  finite with shape (2, 512).
- Passed 8 offline algorithm checks: entropy order; farthest-first coverage;
  no repeated picks with identical features; hybrid shortlist constraint;
  seeded random selection; oracle access restrictions; ten-class metrics;
  budget endpoints and normalized learning-curve area.
- Ran the complete pipeline on a 120-image synthetic HDF5 fixture using
  two seeds, all four strategies, three annotation budgets, and a full-label
  reference for each seed. Generated 52 split-level metric records, per-class
  metrics, figures, reports, query histories, and checkpoints.
- Verified the initial labeled images AND initial validation metrics were
  identical across all four strategies within each seed.
- Verified exact annotation budgets and no repeated queries.
- Restarted the completed pipeline: cached features and 24 round checkpoints
  were reused, with unchanged checkpoint bytes and modification times.
- Simulated an interrupted feature extraction and uncommitted final training
  round in the synthetic fixture. Recovery reconstructed the exact original
  feature bytes, selected source indices, validation metrics and checkpoint.
- Ran the actual --smoke command-line branch successfully.
- Exercised diversity and hybrid selection at the real pool dimensions
  (12,330 x 512, 1,800 labeled centers, 200 queries), using synthetic feature
  vectors to check feasibility rather than scientific performance.
- Compiled all Python modules successfully.

The user's complete 2.7 GB Galaxy10 HDF5 remains on their computer. No claim is
made that the formal Galaxy10 experiment has already run, that hybrid will
outperform the baselines, or that the Windows launcher was executed on this
Linux host. The launcher's smoke stage checks the user's local installation
before beginning formal acquisition trajectories.
