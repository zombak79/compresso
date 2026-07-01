Input and Output
================

This page is the reference for the two ends of a Compresso workflow: the dense
**input** the models expect, and the :class:`~compresso.SRPTensor` **output**
format — how it is laid out, how to convert it, and how to save it so another
run or project can reload it.

Input data
----------

Every model and the trainer expect a single 2D dense matrix of shape
``(n_samples, dim)``:

* **NumPy array or Torch tensor** are both accepted; arrays are wrapped with
  ``torch.as_tensor``.
* **Floating dtype.** Integer inputs are promoted to float automatically. By
  default the models run in ``float32``; non-``float32`` floating inputs are
  cast to the model dtype before training and encoding.
* **2D only.** A clear error is raised for other ranks — flatten higher-rank
  data yourself (as the :doc:`basic-example` flattens 28×28 images to length
  784).

A common pattern is to *fit* on a training subset and *transform* everything,
including rows the model never saw during training:

.. code-block:: python

   trainer.fit(embeddings[train_idx])
   codes = trainer.transform(embeddings)     # SRPTensor for all rows

The SRPTensor format
--------------------

:class:`~compresso.SRPTensor` ("Sparse Representation") stores a row-packed
fixed-``k`` matrix. Instead of a dense ``(rows, cols_total)`` array, it keeps two
compact ``(rows, k)`` tensors:

* ``cols`` — ``int64`` column indices of the active entries in each row.
* ``vals`` — the signed values at those indices.

plus the logical dense ``shape`` and an optional ``prefix_shape`` (so a batched
tensor ``(*prefix, cols_total)`` can be restored). For a matrix with ``H``
columns and ``k`` active entries, storage is ``rows × k × 2`` instead of
``rows × H`` — a large saving when ``H`` is in the thousands.

.. code-block:: python

   srp = trainer.transform(embeddings)

   srp.shape         # (rows, cols_total) logical dense shape
   srp.k             # active entries per row
   srp.nnz           # total stored values = rows * k
   srp.rows          # number of rows
   srp.cols_total    # logical number of columns
   srp.device, srp.dtype
   srp.cols          # (rows, k) int64 indices
   srp.vals          # (rows, k) values

Like a tensor, it supports ``.to(...)``, ``.cpu()``, ``.cuda()``, ``.detach()``,
``.clone()``, and ``.contiguous()``, each returning a new ``SRPTensor``.

Building one directly
---------------------

You do not need the trainer to make an ``SRPTensor`` — any dense tensor can be
projected to its top-``k`` entries with :meth:`~compresso.SRPTensor.from_dense`:

.. code-block:: python

   import torch
   from compresso import SRPTensor

   dense = torch.randn(1000, 4096)
   srp = SRPTensor.from_dense(dense, k=32, score_mode="abs")
   # score_mode: "abs" (largest magnitude), "raw" (largest signed), "relu"

Converting to other formats
---------------------------

When a downstream tool needs a standard layout, convert on demand:

.. code-block:: python

   srp.to_dense()        # torch.Tensor (rows, cols_total)
   srp.to_coo()          # torch.sparse_coo_tensor
   srp.to_csr()          # torch.sparse_csr_tensor
   srp.to_csc()          # torch.sparse_csc_tensor
   srp.to_bsr((br, bc))  # block-sparse rows
   srp.to_scipy_coo()    # scipy.sparse.coo_matrix
   srp.to_scipy_csr()    # scipy.sparse.csr_matrix
   srp.to_numpy_dict()   # {"cols", "vals", "shape", "prefix_shape"} as NumPy

The SciPy conversions are convenient for handing codes to scikit-learn or other
sparse-matrix tooling.

Saving and loading
------------------

The custom on-disk format round-trips an ``SRPTensor`` losslessly. The
convention is the ``.srp.pt`` extension:

.. code-block:: python

   from compresso import save_srp_tensor, load_srp_tensor

   save_srp_tensor("codes.srp.pt", srp)        # note: path first, then tensor
   restored = load_srp_tensor("codes.srp.pt")

   assert restored.shape == srp.shape
   assert restored.k == srp.k

``load_srp_tensor`` accepts ``map_location`` (forwarded to ``torch.load``) so you
can move tensors between devices on load, and ``validate=False`` to skip bounds
checks for trusted files.

Using codes in another run or project
-------------------------------------

Because the format is self-contained, a typical division of labour is to encode
once and analyze later — possibly in a different codebase that only needs the
sparse codes, not the model:

.. code-block:: python

   # --- producer: train once, persist the codes ---
   srp = TopKSAETrainer(cfg).fit_transform(embeddings)
   save_srp_tensor("artifacts/items.srp.pt", srp)

   # --- consumer: a separate script/project ---
   import compresso.clustering as cc
   from compresso import load_srp_tensor

   srp = load_srp_tensor("artifacts/items.srp.pt")
   clusters = cc.ClusteringPipeline([...]).fit(srp)

For storage you control yourself (a database blob, an archive, a custom
serializer), :meth:`~compresso.SRPTensor.to_dict` returns a flat, versioned
payload — ``shape``/``prefix_shape`` plus the ``cols``/``vals`` tensors — and
:meth:`~compresso.SRPTensor.from_dict` rebuilds the tensor from it. For a pure
NumPy view (e.g. to feed a non-Torch store), use
:meth:`~compresso.SRPTensor.to_numpy_dict`:

.. code-block:: python

   payload = srp.to_dict()                 # {"version", "layout", "shape", "cols", "vals", ...}
   srp2 = SRPTensor.from_dict(payload)

The :doc:`clustering-visualization` page picks up exactly here: it loads sparse codes and turns
them into interpretable clusters.
