from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.bundle import DEFAULT_GRAPH_STAGE, extracted_bundle, list_graph_stages, load_graph, load_item_srp, load_manifest, load_metadata, load_split
from lib.cluster_view import child_cluster_table, cluster_features, cluster_items, recommend_clusters_for_user, root_cluster_table
from lib.paths import DEFAULT_BUNDLE


def main() -> None:
    print(f"bundle={DEFAULT_BUNDLE}")
    print(f"bundle_size_mb={DEFAULT_BUNDLE.stat().st_size / 1024 / 1024:.2f}")
    with extracted_bundle(DEFAULT_BUNDLE) as root:
        manifest = load_manifest(root)
        metadata = load_metadata(root)
        split = load_split(root)
        srp = load_item_srp(root)
        stages = list_graph_stages(root)
        print(f"dataset={manifest.get('dataset')}")
        print(f"graph_stages={stages}")
        print(f"metadata_shape={metadata.shape}")
        print(f"srp_shape={srp.shape} k={srp.k} cols_dtype={srp.cols.dtype} vals_dtype={srp.vals.dtype}")
        assert metadata.shape[0] == srp.rows, "metadata rows must align with SRP rows"
        assert len(split["item_ids"]) == srp.rows, "item_ids must align with SRP rows"
        assert split["test_source_indices"], "test users are required"
        for stage in stages:
            graph = load_graph(root, stage)
            print(f"graph[{stage}] nodes={len(graph.clusters)} active={len(graph.active_clusters)} features={graph.n_features}")
            assert graph.n_entities == metadata.shape[0], f"{stage}: graph entities must align with metadata"
            assert graph.n_features == srp.cols_total, f"{stage}: graph feature dim must align with SRP"
            roots = root_cluster_table(graph, min_items=1)
            assert not roots.empty, f"{stage}: roots should not be empty"
            first_cluster_id = roots.iloc[0]["cluster_id"]
            cluster = graph.cluster_by_id[first_cluster_id]
            print(f"  first_root={first_cluster_id!r} label={cluster.label!r} items={cluster.entity_count}")
            print(f"  children={len(child_cluster_table(graph, first_cluster_id))} features={len(cluster_features(cluster))} item_rows={len(cluster_items(metadata, cluster, limit=5))}")
        graph = load_graph(root, DEFAULT_GRAPH_STAGE)
        recs = recommend_clusters_for_user(split["test_source_indices"][0], srp, graph, top_k=5)
        print("recommended_clusters")
        print(recs.to_string(index=False))
        assert not recs.empty, "expected non-empty cluster recommendations for first test user"
    print("OK")


if __name__ == "__main__":
    main()
