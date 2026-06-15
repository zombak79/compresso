# Compresso Demo

Standalone Streamlit demo workspace for exploring sparse representations and discovered clusters.

Current data bundle:

```text
data/goodbooks/goodbooks_demo.zip
```

The bundle is generated from the larger experiment checkpoint `data/exp_descriptions.zip` and keeps only metadata, split information, sparse item representations, and cluster graphs needed by the demo.

Validate from inside this directory:

```bash
python scripts/validate_bundle.py
```

Initial graph stages:

- `semantic_clustering_merged` - default explorer graph
- `topm_clustering_merged` - optional comparison graph
