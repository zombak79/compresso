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

==========================  ============  ====================================================
Field                       Default       Meaning
==========================  ============  ====================================================
``hidden_dim``              ``4096``      Number of dictionary features ``H``.
``k``                       ``128``       Active features kept per row.
``decoder_bias``            ``False``     Add a bias to the default decoder.
``pre_act``                 ``None``      Module applied before sparsification.
``post_sparsify``           ``None``      Module applied to codes after top-k.
``encoder`` / ``decoder``   ``None``      Custom modules (else linear layers).
``sparsify_score_mode``     ``"abs"``     Top-k scoring: ``abs`` / ``raw`` / ``relu``.
``sparsify_ste_alpha``      ``0.01``      Straight-through leak for non-selected entries.
``noise_type``              ``"none"``    Training corruption: ``none`` / ``gaussian``.
``noise_scale``             "global_rms"    Gaussian scaling: absolute / global RMS / feature std.
``noise_level``             ``0.1``       Gaussian scale or adaptive scale multiplier.
``alpha_loss``              ``0.01``      Cosine/MSE mixture weight in the training loss.
``l1_penalty``              ``0.0``       Extra L1 penalty on code activations.
``batch_size``              ``128``       Rows per batch.
``shuffle``                 ``True``      Shuffle rows between epochs.
``seed``                    ``42``        Seed for shuffling, init, and training noise.
``epochs``                  ``10``        Training epochs.
``lr`` / ``weight_decay``   ``1e-3`` / 0  AdamW parameters.
``decay``                   ``False``     Cosine LR decay to zero over training.
``compile``                 ``False``     ``torch.compile`` the model when available.
``device``                  ``"cpu"``     Training/transform device.
``show_progress``           ``True``      tqdm progress bar when tqdm is installed.
``srp_score_mode``          ``"abs"``     Score mode for ``SRPTensor.from_dense`` in transform.
==========================  ============  ====================================================

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

.. note::

   The pruning stack (and the broader ``compresso.layers`` package of sparse
   ``Linear``/``Embedding``/attention layers) is **experimental** and not part
   of the stable first-release surface. The representation-learning API on this
   page and in :doc:`io` is the supported path; expect the parameter/pruning
   APIs to change.
