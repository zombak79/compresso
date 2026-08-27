Advanced Usage
==============

:class:`~compresso.TopKSAETrainer` is the easy path, but it is a thin wrapper.
When you need a custom training loop, a different loss, a non-linear encoder, or
direct control over the sparsification, drop down to the building blocks. This
page covers the lower-level objects exported at the top level of ``compresso``.

The raw model: ``TopKSAE``
--------------------------

:class:`~compresso.TopKSAE` is a plain ``nn.Module``. Its ``forward`` returns a
``(reconstruction, codes, stats)`` triple, where ``codes`` already has exactly
``k`` non-zeros per row and ``stats`` is a dict of monitoring metrics:

.. code-block:: python

   import torch
   from compresso import TopKSAE

   model = TopKSAE(input_dim=128, hidden_dim=512, k=32, tied=False)
   x = torch.randn(256, 128)

   reconstruction, codes, stats = model(x)
   # stats keys: reconstruction_mse, cosine_similarity,
   #             active_count, activation_freq, dead_features

Writing your own training loop is then completely standard PyTorch:

.. code-block:: python

   opt = torch.optim.Adam(model.parameters(), lr=1e-3)

   for epoch in range(50):
       perm = torch.randperm(x.size(0))
       for i in range(0, x.size(0), 128):
           batch = x[perm[i : i + 128]]
           _recon, _codes, stats = model(batch)
           loss = stats["reconstruction_mse"]
           opt.zero_grad()
           loss.backward()
           opt.step()

The ``stats`` dictionary
------------------------

The metrics returned each forward pass are useful both as losses and as health
checks:

``reconstruction_mse``
    Mean squared error between input and reconstruction.
``cosine_similarity``
    Mean per-row cosine similarity; the trainer optimizes a blend of
    ``(1 - cosine_similarity)`` and MSE (see ``alpha_loss``).
``active_count``
    Mean number of active features per row (equals ``k`` for a standard
    top-k SAE).
``activation_freq``
    Per-feature firing rate over the batch, shape ``(hidden_dim,)``.
``dead_features``
    Count of features that never fired in the batch. A large value means part of
    your dictionary is wasted — lower ``k``, lower ``hidden_dim``, or train
    longer.

Encoder, decoder, and tied weights
----------------------------------

By default the encoder and decoder are single ``nn.Linear`` layers, but you can
supply any modules — for example a deeper, non-linear encoder — as long as the
shapes line up:

.. code-block:: python

   import torch.nn as nn
   from compresso import TopKSAE

   encoder = nn.Sequential(
       nn.Linear(784, 256), nn.GELU(), nn.Linear(256, 512),
   )
   model = TopKSAE(input_dim=784, hidden_dim=512, k=16, encoder=encoder)

Set ``tied=True`` to make the decoder reuse the encoder weight (transposed),
which halves the parameter count and is common for SAEs. Use
``model.get_decoder_weight()`` to fetch the effective decoder matrix in either
case (this is what the :doc:`basic-example` plots as dictionary atoms).

Controlling sparsification
--------------------------

The bottleneck is a reusable layer, :class:`~compresso.TopKSparsify`, backed by
the functional :func:`~compresso.topk_ste`. You can drop either into any model:

.. code-block:: python

   from compresso import TopKSparsify, topk_ste

   sparsify = TopKSparsify(k=8, score_mode="abs", ste_alpha=0.01)
   z = sparsify(torch.randn(4, 64))     # exactly 8 non-zeros per row

   z2 = topk_ste(torch.randn(4, 64), k=8, score_mode="abs", ste_alpha=0.01)

Two knobs matter:

* ``score_mode`` selects *which* entries survive the top-k:

  * ``"abs"`` keeps the largest-magnitude values (signed features; the default).
  * ``"raw"`` keeps the largest signed values.
  * ``"relu"`` keeps the largest positive values and discards negatives.

* ``ste_alpha`` is the **straight-through estimator** leak. The forward pass is a
  hard top-k (non-differentiable), so the backward pass routes a fraction
  ``ste_alpha`` of the gradient to the *non-selected* entries and full gradient
  to the selected ones. ``ste_alpha=0`` is a pure hard mask; a small value such
  as ``0.01`` keeps unused features learning and reduces dead features.

These are surfaced on the config as ``sparsify_score_mode`` /
``sparsify_ste_alpha`` and ``srp_score_mode`` (the latter is used by
``transform`` when packing into an :class:`~compresso.SRPTensor`).

Denoising training
------------------

The trainer can optionally corrupt each training batch with Gaussian noise
while keeping the original embedding as the reconstruction target:

.. code-block:: python

   from compresso import TopKSAEConfig, TopKSAETrainer

   cfg = TopKSAEConfig(
       hidden_dim=4096,
       k=128,
       noise_type="gaussian",
       noise_scale="feature_std",
       noise_level=0.05,
   )
   trainer = TopKSAETrainer(cfg).fit(embeddings)

``noise_scale="absolute"`` interprets ``noise_level`` directly in embedding
coordinate units. ``"global_rms"`` derives one scale from the RMS feature
standard deviation of the training embeddings. ``"feature_std"`` uses each
feature's own standard deviation, so differently scaled dimensions receive
proportional noise. Adaptive statistics are computed once at the start of each
``fit``.

Noise is applied only by the training loop. ``encode``, ``reconstruct``, and
``transform`` always use the embeddings exactly as provided. The trainer uses
its own generator seeded by ``seed``, so unrelated Torch random operations do
not change the corruption sequence.

The fitted noise statistics and generator state are included in
``trainer.state_dict()``. Restore the complete trainer with:

.. code-block:: python

   restored = TopKSAETrainer.from_state_dict(trainer.state_dict())

This restores the model, optimizer, history, adaptive noise scale, and, when
restored on the same device type, future noise sequence. States created before
denoising support remain loadable and default to no corruption.

Standard scaling
----------------

Inputs can be standardized before the SAE sees them and un-standardized on the
reconstruction, so the scaling stays an internal detail rather than something
callers apply themselves:

.. code-block:: python

   from compresso import TopKSAEConfig, TopKSAETrainer

   cfg = TopKSAEConfig(
       hidden_dim=4096,
       k=128,
       standard_scaler_mean=True,
       standard_scaler_scale="global_rms",
   )
   trainer = TopKSAETrainer(cfg).fit(embeddings)

   trainer.input_scaler_mean    # fitted per-feature mean, or None
   trainer.input_scaler_scale   # fitted divisor, or None

Both parts are off by default and independent. ``standard_scaler_mean``
subtracts the per-feature training mean. ``standard_scaler_scale`` divides, and
its two modes are not variations of one setting:

``"feature_std"``
   Divides each feature by its own standard deviation, matching
   ``sklearn.preprocessing.StandardScaler``. Every feature ends at unit
   variance, which flattens the relative importance of coordinates and bends
   the space the embeddings live in — a poor fit when that geometry *is* the
   signal.

``"global_rms"``
   Divides everything by one scalar, the root mean per-feature variance. Every
   angle stays identical and every distance ratio constant, because a uniform
   scale is not a distortion; only magnitude changes.

``"global_rms"`` is the one to reach for with L2-normalized embeddings, and the
reason is the learning curve rather than the optimum. An L2-normalized vector
spreads its norm across every dimension, so at 1152 dimensions coordinates sit
near ``0.023``: a freshly initialized ``nn.Linear`` starts with pre-activations
around ``0.02`` and spends early epochs merely growing weights. Scaling lifts
those to roughly ``0.58``. A uniform input scale can be absorbed into the
weights, so the optimum itself does not move — but ``l1_penalty`` and
``weight_decay`` are relative to the data scale, so they may want retuning.

Statistics are fitted on the training rows only, with ``correction=0``, in one
streaming pass that never materializes the source. A constant feature keeps a
scale of ``1`` rather than dividing by zero. Both are carried in
``trainer.state_dict()``.

``standard_scaler_loss_space`` decides where the reconstruction loss is
measured. The default ``"original"`` un-scales the reconstruction and compares
it against the raw input, so the objective and every reported metric stay
identical to an unscaled run. ``"scaled"`` compares in standardized space
instead, weighting every feature equally rather than by its variance. It is
rejected unless some scaling is active.

Two combinations get in each other's way:

* ``standard_scaler_scale="feature_std"`` is **rejected** alongside an adaptive
  ``noise_scale``. Unit variance everywhere makes ``"global_rms"`` and
  ``"feature_std"`` noise indistinguishable from ``"absolute"``. Mean-only
  scaling accepts every noise mode, since centering leaves variance alone.
* Either scaling mode with a normalizing ``post_sparsify`` **warns**. Unit-norm
  codes carry no magnitude, so the rescale has to be undone by the decoder
  alone, and it converges several times worse. That is a bad trade rather than
  a contradiction, so it trains anyway.

Early stopping
--------------

Training can stop as soon as a held-out validation loss stops improving. Supply
the validation rows either as a fraction of the input, or as a separate matrix
when you already hold out your own:

.. code-block:: python

   from compresso import TopKSAEConfig, TopKSAETrainer

   cfg = TopKSAEConfig(hidden_dim=4096, k=128, epochs=200, validation_frac=0.1, patience=10)
   trainer = TopKSAETrainer(cfg).fit(embeddings)

   cfg = TopKSAEConfig(hidden_dim=4096, k=128, epochs=200, patience=10)
   trainer = TopKSAETrainer(cfg).fit(train_embeddings, validation_embeddings=val_embeddings)

``validation_frac`` and ``validation_embeddings`` are mutually exclusive. When
neither is given no validation pass runs, and ``patience`` is then rejected
rather than silently ignored.

With ``validation_frac``, rows are permuted using ``seed`` before the split, so
an ordered input does not put a biased slice in the validation part. The split
happens before any training statistics are fitted, including adaptive noise
scales, so validation rows never influence training.

Validation batches are never corrupted, even under ``noise_type="gaussian"``.
The monitored loss therefore carries no per-epoch noise draw, which would
otherwise make patience counting erratic.

``epochs`` becomes an upper bound. Every epoch appends ``val_``-prefixed
metrics to ``trainer.history`` next to the training metrics:

.. code-block:: python

   trainer.history[-1]["val_loss"]
   trainer.best_epoch      # 1-based epoch with the lowest validation loss
   trainer.best_val_loss
   trainer.stopped_epoch   # None when training ran all of ``epochs``

An epoch counts as an improvement only when the validation loss falls by more
than ``min_delta``. Training stops after ``patience`` consecutive non-improving
epochs. With ``restore_best_weights=True``, the default, the best epoch's
weights are reloaded once training ends, so the model you get back is never the
worse final epoch. ``best_epoch``, ``best_val_loss``, and ``stopped_epoch`` are
carried in ``trainer.state_dict()``.

Logging a long run
------------------

``fit`` reports itself two ways. By default it draws a tqdm progress bar, which
needs a tty; inside a container every refresh becomes its own log line instead.
Pass a ``logger`` to get structured lines and no bar:

.. code-block:: python

   import logging

   from compresso import TopKSAEConfig, TopKSAETrainer

   cfg = TopKSAEConfig(hidden_dim=4096, k=128, epochs=30, validation_frac=0.1, log_prefix="SAE")
   trainer = TopKSAETrainer(cfg, logger=logging.getLogger(__name__)).fit(embeddings)

The logger is duck-typed: anything with an ``info(str)`` method works, so a
``logging.Logger``, a service's own logger, or a shim around ``print`` all fit
and compresso needs no logging dependency of its own. Passing one suppresses
tqdm, since a bar and a log stream would carry the same numbers. That rule is
absolute: a logger always wins, so asking for a bar in the same breath does not
get you both.

Reporting is resolved per call. ``fit``, ``fit_transform``, ``encode``,
``reconstruct``, and ``transform`` each accept ``logger`` and ``show_progress``,
which override the constructor and ``config.show_progress`` for that call only:

.. code-block:: python

   trainer = TopKSAETrainer(cfg, logger=job_logger)

   trainer.fit(embeddings)                        # reports to job_logger
   trainer.transform(embeddings, logger=None)     # this one call stays quiet
   trainer.transform(embeddings, logger=other)    # reports somewhere else

   quiet = TopKSAETrainer(cfg)                    # no default sink
   quiet.fit(embeddings, logger=job_logger)       # ...supplied per call

Omitting either argument inherits the trainer's own value, which is why
``logger=None`` has to mean something distinct: it silences that one call even
on a trainer that has a logger. Silence means silence — an explicit
``logger=None`` drops the bar as well, rather than inheriting the one the
logger had been suppressing, since a bar is not what a caller asking for quiet
wants and a container has no tty to draw it on anyway. Pass
``show_progress=True`` in the same call if a bar is what you meant:

.. code-block:: python

   trainer.transform(x, logger=None)                      # nothing at all
   trainer.transform(x, logger=None, show_progress=True)  # bar, no log lines

A trainer with no logger is unaffected: it keeps drawing whatever
``config.show_progress`` asks for. Because the resolution is per call, a sink
that fails does not poison the trainer — logging stops for the call that hit
the failure, and the next call starts fresh.

A sink is deliberately not part of the model. It describes the job that is
running, so it is never written to ``state_dict()`` and never restored by
``from_state_dict()``; a trainer loaded from a checkpoint starts with no logger
until one is given. That is also what keeps checkpoints picklable when the sink
holds a socket or an HTTP session.

One line opens the run with its shape, one closes it with the outcome, and one
lands per epoch carrying *every* key of that epoch's ``history`` record::

   [SAE] fit started: input_dim 32 | hidden_dim 64 | k 8 | 320 train rows / 80 validation rows | ...
   [SAE] epoch 1/3: 11ms/epoch | 11ms elapsed | 22ms remaining | loss: 1.0292 | ... | dead_features: 0.2000 | val_loss: 0.9780 | ...
   [SAE] fit finished: 14ms total | epochs_run 3 | best_epoch 3 | best_val_loss 0.9236 | early stopping did not fire

Dumping the whole record rather than a chosen few means ``dead_features`` is
always in the stream — the number that says whether ``hidden_dim`` is too wide
for the catalog — and a metric added to ``history`` later shows up on its own.

Each inference pass opens and closes with a line of its own. That matters most
for ``fit_transform``, which passes its ``logger`` to both phases, because
packing a large catalog can take longer than the fit that preceded it::

   [SAE] transform started: 60000 rows | 469 batches of 128 | device cpu
   [SAE] transform finished: 41s total | 60000 rows

When a single epoch or pass runs for minutes, ``log_every_n_steps=N`` adds a
line every ``N``-th batch with time per step and time remaining. It stays off at
the default ``0``.

A logger that raises never ends a fit. The failure is reported once as a
``RuntimeWarning``, logging switches itself off, and training continues — and
because a strict warning filter would otherwise re-raise that notice as an
error, the notice is suppressed too rather than the fit being lost.

Post-sparsification hooks
-------------------------

A ``post_sparsify`` module runs on the codes *after* the top-k. The built-in
:class:`~compresso.L1Normalize` and :class:`~compresso.L2Normalize` rescale each
code to unit L1/L2 norm, which is handy when codes feed a downstream similarity
or retrieval step:

.. code-block:: python

   from compresso import TopKSAEConfig, TopKSAETrainer, L1Normalize

   cfg = TopKSAEConfig(hidden_dim=4096, k=128, post_sparsify=L1Normalize())
   trainer = TopKSAETrainer(cfg)

Full config reference
---------------------

Every trainer hyperparameter lives on :class:`~compresso.TopKSAEConfig`:

==============================  ================  ========================================================================================
Field                           Default           Meaning
==============================  ================  ========================================================================================
``hidden_dim``                  ``4096``          Number of dictionary features ``H``.
``k``                           ``128``           Active features kept per row.
``decoder_bias``                ``False``         Add a bias to the default decoder.
``pre_act``                     ``None``          Module applied before sparsification.
``post_sparsify``               ``None``          Module applied to codes after top-k.
``encoder`` / ``decoder``       ``None``          Custom modules (else linear layers).
``sparsify_score_mode``         ``"abs"``         Top-k scoring: ``abs`` / ``raw`` / ``relu``.
``sparsify_ste_alpha``          ``0.01``          Straight-through leak for non-selected entries.
``noise_type``                  ``"none"``        Training corruption: ``none`` / ``gaussian``.
``noise_scale``                 ``"global_rms"``  Gaussian scaling: ``absolute`` / ``global_rms`` / ``feature_std``.
``noise_level``                 ``0.1``           Gaussian scale or adaptive scale multiplier.
``standard_scaler_mean``        ``False``         Subtract the per-feature training mean before the SAE.
``standard_scaler_scale``       ``"none"``        Divisor after centering: ``none`` / ``feature_std`` / ``global_rms``.
``standard_scaler_loss_space``  ``"original"``    Space the reconstruction loss is measured in: ``original`` / ``scaled``.
``alpha_loss``                  ``0.01``          Cosine/MSE mixture weight in the training loss.
``l1_penalty``                  ``0.0``           Extra L1 penalty on code activations.
``batch_size``                  ``128``           Rows per batch.
``shuffle``                     ``True``          Shuffle rows between epochs.
``seed``                        ``42``            Seed for shuffling, init, and training noise.
``epochs``                      ``10``            Maximum training epochs.
``validation_frac``             ``None``          Fraction of rows held out for validation.
``patience``                    ``None``          Non-improving epochs tolerated before stopping.
``min_delta``                   ``0.0``           Smallest decrease in validation loss counted as improvement.
``restore_best_weights``        ``True``          Reload the best epoch's weights when training ends.
``lr`` / ``weight_decay``       ``1e-3`` / 0      AdamW parameters.
``decay``                       ``False``         Cosine LR decay to zero over training.
``compile``                     ``False``         ``torch.compile`` the model when available.
``device``                      ``"cpu"``         Training/transform device.
``show_progress``               ``True``          tqdm progress bar when tqdm is installed. Ignored when a ``logger`` is passed.
``log_prefix``                  ``"TopKSAE"``     Bracketed tag on every logged line.
``log_every_n_steps``           ``0``             With a ``logger``, also log every ``N``-th batch. ``0`` logs epoch/pass boundaries only.
``srp_score_mode``              ``"abs"``         Score mode for ``SRPTensor.from_dense`` in transform.
==============================  ================  ========================================================================================

Sparse parameters and pruning
-----------------------------

Beyond representation learning, Compresso ships sparse *parameter* types for
compressing model weights:

* :class:`~compresso.MaskedParam` — a weight with a learned/scheduled binary
  mask for magnitude pruning.
* :class:`~compresso.SRPParam` — a parameter backed by the same fixed-k sparse
  layout as :class:`~compresso.SRPTensor`.
* :class:`~compresso.SparsityController` — a global dispatcher that advances and
  rewinds :class:`~compresso.MaskedParam` masks during training, and
  :func:`~compresso.exponential_decay`, a helper for sparsity schedules.

Both sparse parameter types support localized row access:

.. code-block:: python

   dense_rows = masked_param[row_indices]
   sparse_rows = srp_param[row_indices]  # returns SRPTensor

For a row-wise ``MaskedParam``, selection performs the current top-k projection
only for the requested rows. It is therefore equivalent to
``masked_param()[row_indices]`` without materializing the complete masked
parameter. ``SRPParam`` selection uses gradient-preserving ``index_select``;
backward passes through ``sparse_rows.vals`` update the original
``srp_param.values``, including accumulation for duplicate requested rows.

After a ``MaskedParam`` schedule is complete, or after its mask is frozen,
convert the exact stored boolean mask to a trainable fixed structure:

.. code-block:: python

   srp_param = masked_param.to_srp_param()

Conversion preserves selected zero-valued and tied entries without recomputing
top-k. It creates a new optimizer-owned parameter, so the optimizer should be
restarted at this lifecycle boundary.

.. note::

   The pruning stack (and the broader ``compresso.layers`` package of sparse
   ``Linear``/``Embedding``/attention layers) is **experimental** and not part
   of the stable first-release surface. The representation-learning API on this
   page and in :doc:`io` is the supported path; expect the parameter/pruning
   APIs to change.
