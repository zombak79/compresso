# Compresso: A PyTorch Framework for Sparse Representation Learning

Compresso is an open-source PyTorch framework for sparse representation learning. It provides reusable building blocks for learning sparse neural representations, dynamic sparsification, sparse inference, and semantic analysis, enabling researchers to rapidly prototype sparse neural architectures while focusing on models rather than infrastructure.

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
        batch_size=1024,
        epochs=50,
        sparsify_score_mode="abs",
        sparsify_ste_alpha=0.01,
    )
)

srp = trainer.fit_transform(embeddings)
print(srp)
```

Clustering and cluster labeling can be run through the clustering pipeline:

```python
from compresso import clustering as cc

cluster_graph = cc.ClusteringPipeline(
    [
        cc.SRPSimilarityClustering(
            threshold=0.5,            # minimum similarity between items inside a cluster
            top_k=None,               # None = all pairs above threshold
            min_cluster_size=20,      # smaller clusters are discarded
            normalize_rows=True,      # centroids of the cluster will be normalized
            min_local_density=None,   # optional cleanup
            centroid_top_k=8,         # how many top features are included in centroid definition
            batch_size=32,
            show_progress=True,
        ),
        cc.LabelClusters(
            entity_metadata=meta,       # metadata for srp rows
            text_extractor=texts_fn,    # function that converts entity metadata into a single string (entity description)
            label_fn=label_cluster,     # function that converts a list of (cluster members) entity descriptions to one label
            cluster_scope="all",        # scope of nodes in clustering graph
            show_progress=True,
        ),
    ]
)(srp)
```

See full example at https://zombak79.github.io/compresso/clustering.html.

## Citation

If you find this project helpful or use it in your academic work, please consider citing it. This helps us continue to maintain and develop this project. You can find the citation format below.

```bibtex
@misc{compresso,
  title  = {Compresso: A PyTorch Framework for Sparse Representation Learning},
  author = {Van{\v{c}}ura, Vojt{\v{e}}ch and Giacomo Medda and Spi{\v{s}}{\'a}k, Martin and Ladislav Pe{\v{s}}ka},
  year   = {2026},
  url    = {https://github.com/zombak79/compresso}
}
```

