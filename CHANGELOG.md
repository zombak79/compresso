# Changelog

All notable changes to Compresso are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Releases marked *not published* exist as versions in the repository but were
never uploaded to PyPI, so `pip install compresso-pytorch` never resolved to
them. See [Release history notes](#release-history-notes) at the end.

## [0.1.7] — 2026-08-27

### Added

- An injectable logger for `TopKSAETrainer`: `TopKSAETrainer(config, logger=...)`
  accepts anything with an `info(str)` method, so a containerised run can report
  itself as structured log lines instead of a tqdm bar that needs a tty.
  `fit()` emits one line describing the run, one per epoch, and one for the
  outcome. Two new `TopKSAEConfig` fields, `log_prefix` and
  `log_every_n_steps`, tag the lines and optionally add per-batch progress
  inside a long epoch.
- Epoch lines carry every key of the epoch's `history` record rather than a
  curated subset, so `dead_features` and `active_count` are always in the
  stream and a metric added to `history` later needs no change here.
- `fit()`, `fit_transform()`, `encode()`, `reconstruct()`, and `transform()`
  each accept `logger` and `show_progress`, overriding the constructor and
  `config.show_progress` for one call. Omitting either inherits the trainer's
  value; passing `logger=None` silences that call even on a trainer that has a
  logger, including the tqdm bar it would otherwise inherit, unless the same
  call passes `show_progress` too. `fit_transform()` forwards both to its fit and its transform.
- Reporting is resolved per call rather than held on the trainer, so a sink that
  fails stops logging only for the call that hit the failure instead of
  silencing the trainer for good, and two threads sharing a trainer cannot
  disturb each other's reporting. A logger is never written to `state_dict()`,
  which keeps a checkpoint picklable when the sink holds a socket or a session.
- `encode()`, `reconstruct()`, and `transform()` log the start and end of their
  pass, and honour `log_every_n_steps`. These three share the progress helper
  with `fit()`, so suppressing tqdm for a logger would otherwise have left them
  reporting nothing at all — worst in `fit_transform()`, which would log its fit
  as finished and then pack a large catalog in silence.

### Changed

- Passing a `logger` suppresses the tqdm bar, since the two would report the
  same numbers. Callers who pass no logger are unaffected: `show_progress` and
  the bar behave exactly as before.
- Combining `standard_scaler_scale` with a normalizing `post_sparsify`
  (`L1Normalize` or `L2Normalize`) now raises a `RuntimeWarning`. Unit-norm
  codes carry no magnitude, so the rescale has to be undone by the decoder
  alone and converges several times worse, which more epochs do not recover.
  It warns rather than raising, since it is a bad trade rather than a
  contradiction, and it still trains. `standard_scaler_mean` is unaffected,
  because centering barely moves the magnitude.
- `log_prefix` and `log_every_n_steps` are declared after `srp_score_mode`
  rather than beside `show_progress`, so every field published in 0.1.6 keeps
  its position. Inserting them earlier had silently redirected the arguments of
  a positional `TopKSAEConfig(...)` caller — a wrong `srp_score_mode` with no
  error, not a `TypeError`.

### Fixed

- The warning raised when a logger fails is itself suppressed if a strict
  warning filter (`-W error`, `warnings.simplefilter("error")`) would turn it
  into an exception. Otherwise the notice that logging cannot end a fit was
  exactly what ended it.

## [0.1.6] — 2026-08-17

### Added

- Optional standard scaling for `TopKSAETrainer`, applied to inputs before the
  SAE and undone on its reconstruction. Three new `TopKSAEConfig` fields:
  `standard_scaler_mean`, `standard_scaler_scale`, and
  `standard_scaler_loss_space`.
- `standard_scaler_scale` chooses how magnitudes are handled. `"feature_std"`
  divides each feature by its own standard deviation, matching
  `sklearn.preprocessing.StandardScaler`. `"global_rms"` divides everything by
  one scalar, the root mean per-feature variance: coordinates land at unit
  scale for the encoder while every angle and every distance ratio is preserved
  exactly, which suits L2-normalized embeddings whose geometry is the signal.
- `standard_scaler_loss_space` selects whether the reconstruction loss is
  measured in original or standardized space. `"original"`, the default, keeps
  the objective and the reported metrics identical to an unscaled run.
- `input_scaler_mean` and `input_scaler_scale` attributes carrying the fitted
  statistics. `state_dict()` moved to `format_version` 4 and includes them;
  versions 1 through 3 still load.
- Statistics are fitted on the training rows only, with `correction=0`, and a
  constant feature keeps a scale of `1` rather than dividing by zero.

### Changed

- `standard_scaler_mean` now accepts every `noise_scale`. Centering leaves
  per-feature variance untouched, so adaptive noise statistics stay exactly as
  meaningful as they were on raw inputs. Only `"feature_std"` flattens
  variances to `1`, which is what collapses both adaptive modes into absolute,
  and it alone is still rejected alongside them.
- Adaptive noise scales under `"global_rms"` are derived analytically as
  `var_raw / scale ** 2` instead of being refitted. Translation leaves variance
  alone and scaling divides it by a known constant, so this is exact and costs
  no extra pass.

### Fixed

- Fitting no longer materializes the input. `EmbeddingsDataset` keeps the
  source exactly as handed over and converts per batch, validation splits are
  returned as row indices rather than two copied halves, and statistics stream
  in one chunked pass. A memory-mapped export is a usable source at last: on a
  439 MiB float16 memmap with standard scaling and a validation split, peak RSS
  drops from 3138 MiB to 843 MiB.
- Streaming statistics merge chunks with Chan's parallel formula instead of
  `E[x**2] - E[x]**2`. In float32 the latter returns exactly zero for a feature
  whose mean dwarfs its spread, and the zero-variance guard then read that as a
  constant feature and left it unscaled — so the features most in need of
  standardizing passed through untouched.
- `transform()` packs each batch as it is produced instead of densifying the
  whole `(n, hidden_dim)` code matrix first, where `torch.cat` held it twice at
  the peak. On 60k rows with `hidden_dim=4096` and `k=128`, peak memory drops
  from 2413 MiB to 185 MiB. Results are unchanged, since top-k runs per row.

## [0.1.5] — 2026-08-06

### Added

- Early stopping for `TopKSAETrainer`, driven by a held-out validation loss.
  Four new `TopKSAEConfig` fields: `validation_frac`, `patience`, `min_delta`,
  and `restore_best_weights`.
- `fit()` and `fit_transform()` accept a `validation_embeddings` keyword for
  callers who hold out their own rows. Mutually exclusive with
  `validation_frac`; `patience` without either raises rather than being
  silently ignored.
- `TopKSAETrainer.eval_step()`, which scores a batch without corruption. Used
  for the validation pass so the monitored loss carries no per-epoch noise
  draw.
- `best_epoch`, `best_val_loss`, and `stopped_epoch` attributes on the trainer,
  plus `val_`-prefixed metrics in `history`. All three round-trip through
  `state_dict()`, now `format_version` 3. Versions 1 and 2 still load.
- With `validation_frac`, rows are permuted using `seed` before the split, and
  the split happens before adaptive noise statistics are fitted, so validation
  rows never influence training.

### Fixed

- Device-to-host copies no longer pass `non_blocking=True`. The flag does not
  synchronize, so these copies could return host memory before the transfer
  landed: corrupted training batches, corrupted sparse index tensors, and
  training that was nondeterministic despite a fixed seed. Five sites, in
  `EmbeddingsDataset.__getitem__`, `GatedMaskedParam.spawn_compacted_gate`,
  `CooSparseParam.build_coo`, and its two packed-selection helpers. Reachable
  whenever a tensor sat on an accelerator and the copy targeted the CPU.
- The configuration reference table in `docs/source/advanced-usage.rst` was a
  malformed reStructuredText table and failed to render at all, so the whole
  table was missing from the built documentation.
- The same table described `min_delta` as a "validation gain", which reads as
  the loss increasing. It is a required decrease.

## [0.1.4] — 2026-08-05 *(not published)*

### Added

- Denoising training for `TopKSAETrainer`. Each training batch can be corrupted
  with Gaussian noise while the clean embedding stays the reconstruction
  target. Three new `TopKSAEConfig` fields: `noise_type`, `noise_scale`, and
  `noise_level`.
- `noise_scale` selects how the noise is scaled: `"absolute"` uses embedding
  coordinate units, `"global_rms"` derives one scale from the training set, and
  `"feature_std"` scales each input feature separately. Adaptive statistics are
  fitted once per `fit()` and exposed as `input_feature_mean` and
  `input_feature_variance`.
- `TopKSAETrainer.load_state_dict()` and `TopKSAETrainer.from_state_dict()`.
  `state_dict()` moved to `format_version` 2, carrying the fitted noise
  statistics and the noise generator state; version 1 states still load and
  default to no corruption.
- Corruption uses a generator seeded from `seed`, independent of the global
  Torch RNG, so unrelated random operations do not shift the noise sequence.
  `encode`, `reconstruct`, and `transform` never corrupt their inputs.

## [0.1.3] — 2026-07-29

### Added

- Row selection on sparse parameters: `SRPTensor.select_rows()`,
  `MaskedParam.select_rows()`, and `MaskedParam.to_srp_param()` for converting
  a completed or frozen mask into a packed sparse representation.
- Shared row-index normalization behind the new selection paths.
- Documentation pages for sparse-tensor I/O and citation guidance.
- Test coverage for sparse parameter indexing and conversion.

### Changed

- `SRPParam.select_rows()` now returns an `SRPTensor` rather than an
  `SRPParam`, and accepts the wider row-index type shared by the new selection
  paths instead of only a `torch.Tensor`.

## [0.1.2] — 2026-07-13

### Added

- PyPI installation instructions, README badges, and project logo.
- Docstrings for `TopKSparsify.forward` and the other `forward` methods.
- Extra classifier topics in package metadata.

### Changed

- Simplified the documented examples and improved the documentation build
  workflow.

## [0.1.1] — 2026-07-13 *(not published)*

Tagged, but `pyproject.toml` was left at `0.1.0`, so the publish workflow had
nothing new to upload. Its contents shipped in 0.1.2.

## [0.1.0] — 2026-07-07

Initial release.

### Added

- `TopKSAE`, a top-k sparse autoencoder returning a
  `(reconstruction, codes, stats)` triple, with optional tied decoder weights
  and custom encoder/decoder modules.
- `TopKSAETrainer` and `TopKSAEConfig`, a scikit-learn-style wrapper with
  `fit`, `transform`, `fit_transform`, `encode`, and `reconstruct`.
- `TopKSparsify` and the functional `topk_ste`, a hard top-k with a
  straight-through estimator and configurable leakage for non-selected entries.
- Sparse representation types `SRPTensor` and `SRPParam`, with `save_srp_tensor`
  and `load_srp_tensor` for on-disk round-trips.
- `MaskedParam` for sparse parameters, with `SparsityController` and
  `exponential_decay` for pruning schedules.
- `L1Normalize` and `L2Normalize` post-sparsification hooks.
- A `compresso.clustering` subpackage: `ClusteringPipeline` plus clustering,
  merging, tagging, and serialization components for analyzing sparse codes.

## Release history notes

- **0.1.1 and 0.1.4 were never published to PyPI.** 0.1.1 was tagged without
  bumping `pyproject.toml`. 0.1.4 was bumped and merged to `main` but never
  tagged, and both the publish and documentation workflows trigger only on
  `v*` tags.
- **Dates** for published releases are their PyPI upload dates. 0.1.4 is dated
  by when it landed on `main`; its version bump commit predates that by a week.
  0.1.1 is dated by its tag.
- **History before 0.1.0** was squashed into a single commit. The original
  development commits are preserved on the `main-before-squash` branch.

[0.1.7]: https://github.com/zombak79/compresso/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/zombak79/compresso/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/zombak79/compresso/compare/v0.1.3...v0.1.5
[0.1.4]: https://github.com/zombak79/compresso/compare/v0.1.3...main
[0.1.3]: https://github.com/zombak79/compresso/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/zombak79/compresso/compare/v0.1.0...v0.1.2
[0.1.1]: https://github.com/zombak79/compresso/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/zombak79/compresso/releases/tag/v0.1.0
