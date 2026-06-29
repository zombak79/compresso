Quickstart
==========

Train a top-k sparse autoencoder on dense embeddings:

.. code-block:: python

   import numpy as np
   from compresso import TopKSAEConfig, TopKSAETrainer

   embeddings = np.random.randn(10_000, 512).astype("float32")

   trainer = TopKSAETrainer(
       TopKSAEConfig(
           hidden_dim=4096,
           k=32,
           batch_size=1024,
           epochs=50,
           sparsify_score_mode="abs",
           sparsify_ste_alpha=0.01,
       )
   )

   srp = trainer.fit_transform(embeddings)
   print(srp)

``fit_transform`` trains the model and returns an :class:`compresso.SRPTensor`
with shape ``(10_000, 4096)`` and ``32`` stored values per row.

For default models, non-``float32`` embedding inputs are converted to the
model's floating dtype before training and encoding.
