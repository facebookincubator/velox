====================================
Miscellaneous Functions
====================================

.. spark:function:: monotonically_increasing_id() -> bigint

    Returns monotonically increasing 64-bit integers. The generated ID is
    guaranteed to be monotonically increasing and unique, but not consecutive.
    The current implementation puts the partition ID in the upper 31 bits, and
    the lower 33 bits represent the record number within each partition.
    The assumption is that the data frame has less than 1 billion partitions,
    and each partition has less than 8 billion records.
    The function relies on partition IDs, which are provided by the framework
    via the configuration 'spark.partition_id'.

.. spark:function:: spark_partition_id() -> integer

    Returns the current partition id.
    The framework provides partition id through the configuration
    'spark.partition_id'.
    It ensures deterministic data partitioning and assigns a unique partition
    id to each task in a deterministic way. Consequently, this function is
    marked as deterministic, enabling Velox to perform constant folding on it.

.. spark:function:: uuid(seed) -> string

    Returns an universally unique identifier (UUID) string. The value is
    returned as a canonical UUID 36-character string. The UUID is generated
    from Pseudo-Random Numbers with the seed by combining user-specified
    ``seed`` and the configuration `spark.partition_id`.
    ``seed`` must be constant. ::

        SELECT uuid(0);    -- "8c7f0aac-97c4-4a2f-b716-a675d821ccc0"
