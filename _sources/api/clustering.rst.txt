Clustering API
==============

The clustering API is under active development. The objects documented here are
the intended public entry points exported by :mod:`compresso.clustering`.

Types
-----

.. autoclass:: compresso.clustering.SparseVector
   :members:

.. autoclass:: compresso.clustering.ScoredTag
   :members:

.. autoclass:: compresso.clustering.SparseCluster
   :members:

.. autoclass:: compresso.clustering.SparseClusterGraph
   :members:

.. autoclass:: compresso.clustering.SparseClusterSet
   :members:

Pipeline
--------

.. autoclass:: compresso.clustering.ClusteringPipeline
   :members:

.. autoclass:: compresso.clustering.AbstractClusterTransform
   :members:

.. autoclass:: compresso.clustering.AbstractClustering
   :members:

.. autoclass:: compresso.clustering.AbstractMerging
   :members:

Clustering Steps
----------------

.. autoclass:: compresso.clustering.DominantSignedClustering
   :members:

.. autoclass:: compresso.clustering.TopMSignedClustering
   :members:

.. autoclass:: compresso.clustering.ComboSignedClustering
   :members:

.. autoclass:: compresso.clustering.FeaturePathClustering
   :members:

.. autoclass:: compresso.clustering.SRPSimilarityClustering
   :members:

Linking and Merging
-------------------

.. autoclass:: compresso.clustering.EntityContainmentLink
   :members:

.. autoclass:: compresso.clustering.FeatureContainmentLink
   :members:

.. autoclass:: compresso.clustering.MaterializeLinkMerges
   :members:

.. autoclass:: compresso.clustering.EntityIoUMerge
   :members:

.. autoclass:: compresso.clustering.EntityContainmentMerge
   :members:

.. autoclass:: compresso.clustering.FeatureContainmentMerge
   :members:

.. autoclass:: compresso.clustering.CentroidSimilarityMerge
   :members:

.. autoclass:: compresso.clustering.LabelDuplicateMerge
   :members:

.. autoclass:: compresso.clustering.SemanticSimilarityMerge
   :members:

.. autoclass:: compresso.clustering.TagSimilarityMerge
   :members:

Post-processing
---------------

.. autoclass:: compresso.clustering.CompactHiddenClusters
   :members:

.. autoclass:: compresso.clustering.PruneRedundantRoots
   :members:

.. autoclass:: compresso.clustering.AssignTags
   :members:

.. autoclass:: compresso.clustering.LabelClusters
   :members:

.. autoclass:: compresso.clustering.AssignUnclusteredToNearestCluster
   :members:

.. autoclass:: compresso.clustering.SizeFilter
   :members:

Persistence and Helpers
-----------------------

.. autofunction:: compresso.clustering.save_cluster_graph

.. autofunction:: compresso.clustering.load_cluster_graph

.. autofunction:: compresso.clustering.graph_to_dict

.. autofunction:: compresso.clustering.graph_from_dict

.. autofunction:: compresso.clustering.cluster_srp

.. autofunction:: compresso.clustering.run_clustering_pipeline
