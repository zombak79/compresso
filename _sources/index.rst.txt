Welcome to Compresso's documentation!
=====================================

Compresso is a small PyTorch library for fixed-k sparse representations,
sparse autoencoders, sparse parameters, and sparse-representation clustering.

The project distribution is named ``compresso-pytorch`` and the Python package
is imported as ``compresso``.

.. code-block:: python

   import numpy as np
   from compresso import TopKSAEConfig, TopKSAETrainer

   embeddings = np.random.randn(10_000, 512).astype("float32")
   srp = TopKSAETrainer(TopKSAEConfig(hidden_dim=4096, k=32, epochs=50)).fit_transform(embeddings)
   print(srp)   # SRPTensor(shape=(10000, 4096), k=32, ...)

New here? Read :doc:`getting-started` for the concepts, then :doc:`basic-example`
to *see* what a sparse autoencoder learns.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   getting-started

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   basic-example
   advanced-usage
   io
   clustering-visualization
   clustering

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
