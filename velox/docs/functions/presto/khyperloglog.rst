======================
KHyperLogLog Functions
======================

Like Presto, Velox implements the `KHyperLogLog <https://research.google/pubs/pub47664/>`_
algorithm and data structure. ``KHyperLogLog`` data structure can be created
through :func:`!khyperloglog_agg`.


Data Structures
---------------

KHyperLogLog is a data sketch that compactly represents the association of two
columns. It is implemented in Presto as a two-level data structure composed of
a MinHash structure whose entries map to ``HyperLogLog``.

Serialization
-------------

KHyperLogLog sketches can be cast to and from ``varbinary``. This allows them to
be stored for later use.

Functions
---------

.. function:: uniqueness_distribution(khll) ->  map<bigint,double>

    For a certain value ``x'``, uniqueness is understood as how many ``y'`` values are
    associated with it in the source dataset. This is obtained with the cardinality
    of the HyperLogLog that is mapped from the MinHash bucket that corresponds to
    ``x'``. This function returns a histogram that represents the uniqueness
    distribution, the X-axis being the ``uniqueness`` and the Y-axis being the relative
    frequency of ``x`` values.

.. function:: uniqueness_distribution(khll, histogramSize) ->  map<bigint,double>

    Returns the uniqueness histogram with the given amount of buckets. If omitted,
    the value defaults to 256. All ``uniqueness`` values greater than ``histogramSize`` are
    accumulated in the last bucket.
