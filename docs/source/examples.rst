Examples
========

Sparse Autoencoder
------------------

This compact example uses a smaller model so it can run quickly while exploring
the API:

.. code-block:: python

   import numpy as np
   from compresso import L1Normalize, TopKSAEConfig, TopKSAETrainer

   embeddings = np.random.default_rng(0).normal(size=(256, 32)).astype("float32")

   trainer = TopKSAETrainer(
       TopKSAEConfig(
           hidden_dim=128,
           k=8,
           batch_size=64,
           epochs=3,
           post_sparsify=L1Normalize(),
           show_progress=False,
       )
   )

   srp = trainer.fit_transform(embeddings)
   print(srp.shape)

Saving Sparse Representations
-----------------------------

Use :func:`compresso.save_srp_tensor` and :func:`compresso.load_srp_tensor` to
round-trip sparse representation tensors:

.. code-block:: python

   from compresso import load_srp_tensor, save_srp_tensor

   save_srp_tensor(srp, "codes.srpz")
   restored = load_srp_tensor("codes.srpz")

   assert restored.shape == srp.shape

Clustering
----------

This notebook walks through an end-to-end clustering workflow using Compresso
and Compresso Recsys: building a checkpoint, embedding item metadata, training
a sparse autoencoder, discovering clusters, labeling them with an LLM, and
visualizing example items.

.. toctree::
   :maxdepth: 1

   clustering
