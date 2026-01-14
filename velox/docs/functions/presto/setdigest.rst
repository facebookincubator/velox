======================
SetDigest Functions
======================

SetDigest is a data sketch for estimating set cardinality and performing set
operations like intersection cardinality and Jaccard index. It combines HyperLogLog
for cardinality estimation with MinHash for exact counting and intersection operations.

SetDigests may be merged, and for storage and retrieval they may be cast to/from ``VARBINARY``.

Data Structures
---------------

A SetDigest is a data sketch which stores approximate set membership and cardinality
information. The Velox type for this data structure is called ``SetDigest``.
SetDigests support two element types internally:

* ``bigint`` - for integer values (all numeric types are converted to bigint)
* ``varchar`` - for string values

When a SetDigest is exact (cardinality is less than the maximum hash limit),
operations like intersection cardinality return exact results. When the digest
becomes approximate (high cardinality), it uses HyperLogLog and MinHash estimation.

Serialization format is compatible with Presto's.

Aggregate Functions
-------------------

.. function:: make_set_digest(x) -> SetDigest

    Returns the ``SetDigest`` sketch which summarizes the input data set of ``x``.
    Supported input types include: ``boolean``, ``tinyint``, ``smallint``, ``integer``,
    ``bigint``, ``real``, ``double``, ``date``, ``varchar``, and ``varbinary``.

.. function:: merge_set_digest(SetDigest) -> SetDigest

    Returns the ``SetDigest`` of the aggregate union of the individual ``SetDigest``
    structures.

Scalar Functions
----------------

.. function:: cardinality(setdigest) -> bigint

    Returns the estimated cardinality of the set represented by the ``SetDigest`` sketch.
    If the digest is exact (low cardinality), returns the exact count.
    Otherwise, returns an approximation using HyperLogLog.

.. function:: intersection_cardinality(setdigest1, setdigest2) -> bigint

    Returns the estimated intersection cardinality between two ``SetDigest`` sketches.

    * If both digests are exact: returns the exact intersection count
    * If either digest is approximate: returns an estimation using the Jaccard index

    The result is capped at the minimum cardinality of the two input digests to
    ensure logical consistency.

.. function:: jaccard_index(setdigest1, setdigest2) -> double

    Returns the Jaccard index (similarity coefficient) between two ``SetDigest`` sketches.
    The Jaccard index is a value in [0, 1] where:

    * 1.0 means the sets are identical
    * 0.0 means the sets are disjoint (no overlap)

    Uses MinHash estimation for efficient computation.
