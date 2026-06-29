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

Clustering Sparse Codes
-----------------------

The clustering API works with sparse representations produced by Compresso:

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

   print(len(graph.clusters))
