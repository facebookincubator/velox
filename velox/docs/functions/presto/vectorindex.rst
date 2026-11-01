======================
Vector Index Functions
======================

A ``VECTOR_INDEX`` is a local vector index: a FAISS index over embeddings, with
each embedding's id, serialized as ``faiss::write_index`` writes it and stored
as ``VARBINARY``. :func:`vector_index_build` and :func:`vector_index_pack`
return it, and :func:`vector_count` counts the embeddings of many.

This function is available when Velox is built with FAISS
(``VELOX_ENABLE_FAISS``).

.. function:: cardinality(index) -> bigint
   :noindex:

    Returns the number of embeddings in the vector index ``index``.
