=========================
KHyperLogLog Functions
=========================

KHyperLogLog is a data sketch for estimating reidentifiability and joinability within a dataset.
Based on the `KHyperLogLog paper <https://research.google/pubs/khyperloglog-estimating-reidentifiability-and-joinability-of-large-data-at-scale/>`_,
it maintains a map of K number of HyperLogLog structures, where each entry corresponds to a unique key from one column,
and the HLL estimates the cardinality of the associated unique identifiers from another column.

Data Structures
---------------

A KHyperLogLog is a data sketch which stores approximate cardinality information for key-value
associations. The Velox type for this data structure is called ``KHyperLogLog``.
For storage and retrieval, KHyperLogLog values may be cast to/from ``VARBINARY``.

Serialization format is compatible with Presto's.

Aggregate Functions
-------------------

.. function:: khyperloglog_agg(x, uii) -> KHyperLogLog

    Returns the ``KHyperLogLog`` sketch which summarizes the association between
    the key column ``x`` and the unique identifier column ``uii``.
    The ``x`` parameter represents the key values and ``uii`` represents
    the unique identifiers associated with each key.

.. function:: merge(KHyperLogLog) -> KHyperLogLog

    Returns the ``KHyperLogLog`` of the aggregate union of the individual ``KHyperLogLog``
    structures.

Scalar Functions
----------------

.. function:: cardinality(khll) -> bigint

    Returns the estimated total cardinality (number of unique keys) from the
    ``KHyperLogLog`` sketch ``khll``.

.. function:: intersection_cardinality(khll1, khll2) -> bigint

    Returns the estimated intersection cardinality between two ``KHyperLogLog`` sketches.
    If both sketches are exact (small cardinality), returns the exact intersection count.
    Otherwise, returns an approximation using the Jaccard index.

.. function:: jaccard_index(khll1, khll2) -> double

    Returns the Jaccard index (similarity coefficient) between two ``KHyperLogLog`` sketches.
    The Jaccard index is a value in [0, 1] where:

    * 1.0 means the sets are identical
    * 0.0 means the sets are disjoint (no overlap)

.. function:: merge_khll(array(KHyperLogLog)) -> KHyperLogLog

    Returns the ``KHyperLogLog`` of the union of an array of ``KHyperLogLog`` structures.

    * Returns ``NULL`` if the input array is ``NULL``, empty, or contains only ``NULL`` elements
    * Ignores ``NULL`` elements and merges only valid ``KHyperLogLog`` structures when the array contains a mix of ``NULL`` and non-null elements

.. function:: reidentification_potential(khll, threshold) -> double

    Returns the reidentification potential of the ``KHyperLogLog`` sketch ``khll``
    at the given ``threshold``. This measures the fraction of keys that have
    cardinality at or below the threshold, which indicates how easily those
    keys could be reidentified.

.. function:: uniqueness_distribution(khll) -> map(bigint, double)

    Returns a histogram map representing the distribution of uniqueness values
    in the ``KHyperLogLog`` sketch ``khll``. Each key in the map represents a
    cardinality bucket, and the value represents the fraction of keys falling
    into that bucket. The histogram size defaults to the minhash size of the
    KHyperLogLog instance.

.. function:: uniqueness_distribution(khll, histogramSize) -> map(bigint, double)
   :noindex:

    Returns a histogram map representing the distribution of uniqueness values
    in the ``KHyperLogLog`` sketch ``khll`` with the specified ``histogramSize``.
    Each key in the map represents a cardinality bucket, and the value represents
    the fraction of keys falling into that bucket.
