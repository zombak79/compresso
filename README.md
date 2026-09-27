# Compresso: A PyTorch Framework for Sparse Representation Learning

[![PyPI](https://img.shields.io/pypi/v/compresso-pytorch.svg)](https://pypi.org/project/compresso-pytorch/)
[![Python](https://img.shields.io/pypi/pyversions/compresso-pytorch.svg)](https://pypi.org/project/compresso-pytorch/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://zombak79.github.io/compresso/)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://compreapp-demo.streamlit.app)
[![Documentation build](https://github.com/zombak79/compresso/actions/workflows/docs.yml/badge.svg)](https://github.com/zombak79/compresso/actions/workflows/docs.yml)

<img
  src="docs/source/_static/compresso.jpg"
  alt="Compresso logo"
  align="left"
  width="97"
  hspace="16"
/>

Compresso is an open-source PyTorch framework for sparse representation learning. It provides reusable building blocks for learning sparse neural representations, dynamic sparsification, sparse inference, and semantic analysis, enabling researchers to rapidly prototype sparse neural architectures while focusing on models rather than infrastructure.
<br clear="left">
## Why Compresso?

Sparse representations are becoming increasingly important across machine learning due to their efficiency, interpretability, and ability to capture semantically meaningful concepts. Yet building sparse models often requires implementing pruning schedules, sparse kernels, training loops, device management, and visualization from scratch.

Compresso hides this complexity behind a simple, modular API.

The name is inspired by Italian espresso culture: when you order a coffee in Italy, you simply ask for a caffè. The barista handles the beans, pressure, and brewing; you just enjoy the result. Compresso follows the same philosophy: researchers should be able to train and analyze sparse representations without worrying about the underlying engineering.

## Install

Using pip:

```bash
pip install compresso-pytorch
```

For local development:

```bash
git clone https://github.com/zombak79/compresso.git
cd compresso
pip install -e ".[test]"
```

## Documentation

Documentation is available at https://zombak79.github.io/compresso/.

## Minimal Example

You can train a sparse autoencoder through one high-level class `TopKSAETrainer` with a scikit-learn-style wrapper: `fit` trains on a dense matrix, `transform` returns sparse codes, and `fit_transform` does both. All hyperparameters live in the `TopKSAEConfig` dataclass.

```python
import numpy as np
from compresso import TopKSAEConfig, TopKSAETrainer

embeddings = np.random.randn(10_000, 512).astype("float32")

trainer = TopKSAETrainer(
    TopKSAEConfig(
        hidden_dim=4096,
        k=32,
    )
)

srp = trainer.fit_transform(embeddings)
print(srp)
```

To train a denoising SAE, enable Gaussian corruption. The trainer adds noise
only to training inputs and still reconstructs the original clean embeddings:

```python
trainer = TopKSAETrainer(
    TopKSAEConfig(
        hidden_dim=4096,
        k=32,
        noise_type="gaussian",
    )
)
srp = trainer.fit_transform(embeddings)
```

Clustering and cluster labeling can be run through the clustering pipeline:

```python
from compresso import clustering as cc

cluster_graph = cc.ClusteringPipeline(
    [
        cc.DominantSignedClustering(min_cluster_size=20),
        cc.LabelClusters(...),
    ]
)(srp)
```

See full example at https://zombak79.github.io/compresso/clustering.html.

## Recommender Systems Add-on

For recommender-system experiments, see [compresso-recsys](https://github.com/zombak79/compresso-recsys), the companion package built on top of Compresso.

It provides recommender-specific dataset loaders, checkpoint management, and retrieval metrics such as Recall and nDCG. It can be installed with:

```bash
pip install compresso-recsys
```

Compresso contains the general sparse representation learning components, while `compresso-recsys` provides the infrastructure needed to apply and evaluate them in recommender-system experiments.


## Citation

If you find this project helpful or use it in your academic work, please consider citing it. This helps us continue to maintain and develop this project. You can find the citation format below.

For method-specific references, including the sparse embedding compression
work behind `TopKSAETrainer`, see the
[citation guide](https://zombak79.github.io/compresso/citing.html).

```bibtex
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
```
