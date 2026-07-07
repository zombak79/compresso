# Compresso: A PyTorch Framework for Sparse Representation Learning

Compresso is a small PyTorch library for fixed-k sparse representations,
sparse autoencoders, sparse parameters, and sparse-representation clustering.

The PyPI/project distribution name is `compresso-pytorch`, but the Python
package is still imported as `compresso`:

```python
import compresso
from compresso import SRPTensor, TopKSAE, TopKSAETrainer
```

## Install

Until the first PyPI release is published, install directly from GitHub:

```bash
pip install "compresso-pytorch@git+https://github.com/zombak79/compresso.git"
```

For local development:

```bash
git clone https://github.com/zombak79/compresso.git
cd compresso
pip install -e ".[test]"
```

## Documentation

The documentation can be built locally with Sphinx:

```bash
pip install -r docs/requirements.txt
pip install -e .
sphinx-build -b html docs/source docs/build/html
```

After GitHub Pages is enabled for the `gh-pages` branch, release documentation
will be available at:

```text
https://zombak79.github.io/compresso/
```

## First-Release Scope

The top-level `compresso` API intentionally focuses on the stable sparse
learning pieces:

```python
from compresso import (
    MaskedParam,
    SRPParam,
    SRPTensor,
    TopKSAE,
    TopKSAEConfig,
    TopKSAETrainer,
    TopKSparsify,
    topk_ste,
    SparsityController,
    save_srp_tensor,
    load_srp_tensor,
)
```

Clustering is available as a subpackage:

```python
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
```

## Minimal SAE Example

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

## Project Split

Compresso is the core sparse-learning library. Recommender-system experiments,
datasets, ELSA/CompressedELSA pipelines, and checkpoint tooling live in a
separate companion package:

```text
compresso-recsys
```

That package will depend on Compresso and expose recommender-specific utilities
as:

```python
import compresso_recsys as cr
```

This keeps `compresso-pytorch` lightweight and focused.
