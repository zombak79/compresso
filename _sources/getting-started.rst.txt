Getting Started
===============

Compresso focuses on dense-to-sparse representation workflows in PyTorch. The
first-release public API is centered on four areas:

* fixed-k sparse representation tensors with :class:`compresso.SRPTensor`
* sparse autoencoder training with :class:`compresso.TopKSAETrainer`
* sparse parameter utilities such as :class:`compresso.MaskedParam` and
  :class:`compresso.SRPParam`
* sparse-representation clustering through :mod:`compresso.clustering`

Basic Workflow
--------------

A typical workflow starts with dense embeddings, trains a top-k sparse
autoencoder, and returns sparse codes as an :class:`compresso.SRPTensor`.

.. code-block:: python

   import numpy as np
   from compresso import TopKSAEConfig, TopKSAETrainer

   embeddings = np.random.randn(1_000, 128).astype("float32")

   trainer = TopKSAETrainer(
       TopKSAEConfig(
           hidden_dim=512,
           k=16,
           batch_size=128,
           epochs=5,
           show_progress=False,
       )
   )

   srp = trainer.fit_transform(embeddings)

The resulting ``srp`` object stores exactly ``k`` sparse values per row and can
be passed to clustering utilities or saved for later use.

Clustering Workflow
-------------------

Clustering is available as a subpackage:

.. code-block:: python

   import compresso.clustering as cc

   graph = cc.ClusteringPipeline(
       [
           cc.TopMSignedClustering(top_m=4, min_cluster_size=5),
           cc.EntityContainmentLink(threshold=1.0),
           cc.MaterializeLinkMerges(parent_scope="active"),
           cc.PruneRedundantRoots(),
           cc.SizeFilter(min_cluster_size=20),
       ]
   ).fit(srp)

The clustering API is under active development, but the documented surface is
the intended public entry point.
