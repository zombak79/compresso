Citing Compresso
================

If you use Compresso in academic work, please consider citing the framework
and the papers corresponding to the methods used in your research.

Compresso
---------

For the Compresso sparse-representation framework, cite the project:

.. code-block:: bibtex

  @inproceedings{10.1145/3773078.3841254,
      author = {Van{\v c}ura, Vojt{\v e}ch and Medda, Giacomo and Spi{\v s}{\'a}k, Martin and Pe{\v s}ka, Ladislav},
      title = {COMPRESSO: Espresso-Style Sparse Representation Learning for Interpretable Recommender Systems},
      year = {2026},
      isbn = {9798400722844},
      publisher = {Association for Computing Machinery},
      address = {New York, NY, USA},
      url = {https://doi.org/10.1145/3773078.3841254},
      doi = {10.1145/3773078.3841254},
      abstract = {Sparse representations can make recommender-system embeddings more compact and inspectable, but developing sparse-learning workflows typically requires substantial engineering around sparsification, training, pruning, storage, and analysis. We present Compresso, an open-source PyTorch framework that exposes this functionality through a simple and modular interface. Inspired by Italian espresso culture, where one orders a caff{\`e} while the barista handles the machinery, Compresso lets researchers focus on sparse models rather than infrastructure. The framework provides high-level sparse autoencoder training, reusable sparse tensor representations, differentiable top-k operators, sparse and masked neural parameters, pruning schedules, and composable clustering tools.},
      booktitle = {Proceedings of the 20th ACM Conference on Recommender Systems},
      pages = {1793–1795},
      numpages = {3},
      keywords = {Sparse Representations, Embedding Compression, Interpretability},
      location = {
      },
      series = {RecSys '26}
   }

.. _sae-trainer-citation:

Sparse embedding compression and ``TopKSAETrainer``
---------------------------------------------------

:class:`compresso.TopKSAETrainer` builds on the sparse embedding compression
method described in:

.. code-block:: bibtex

   @inproceedings{kasalicky2025future,
     title={The Future is Sparse: Embedding Compression for Scalable Retrieval in Recommender Systems},
     author={Kasalick{\`y}, Petr and Spi{\v{s}}{\'a}k, Martin and Van{\v{c}}ura, Vojt{\v{e}}ch and Bohun{\v{e}}k, Daniel and Alves, Rodrigo and Kord{\'\i}k, Pavel},
     booktitle={Proceedings of the Nineteenth ACM Conference on Recommender Systems},
     pages={1099--1103},
     year={2025}
   }
