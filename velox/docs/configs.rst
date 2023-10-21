========================
Configuration properties
========================

Generic Configuration
---------------------
.. list-table::
   :widths: 20 10 10 70
   :header-rows: 1

   * - Property Name
     - Type
     - Default Value
     - Description
   * - preferred_output_batch_bytes
     - integer
     - 10MB
     - Preferred size of batches in bytes to be returned by operators from Operator::getOutput. It is used when an
       estimate of average row size is known. Otherwise preferred_output_batch_rows is used.
   * - preferred_output_batch_rows
     - integer
     - 1024
     - Preferred number of rows to be returned by operators from Operator::getOutput. It is used when an estimate of
       average row size is not known. When the estimate of average row size is known, preferred_output_batch_bytes is used.
   * - max_output_batch_rows
     - integer
     - 10000
     - Max number of rows that could be return by operators from Operator::getOutput. It is used when an estimate of
       average row size is known and preferred_output_batch_bytes is used to compute the number of output rows.
   * - table_scan_getoutput_time_limit_ms
     - integer
     - 5000
     - TableScan operator will exit getOutput() method after this many milliseconds even if it has no data to return yet. Zero means 'no time limit'.
   * - abandon_partial_aggregation_min_rows
     - integer
     - 100,000
     - Min number of rows when we check if a partial aggregation is not reducing the cardinality well and might be
       a subject to being abandoned.
   * - abandon_partial_aggregation_min_pct
     - integer
     - 80
     - If a partial aggregation's number of output rows constitues this or highler percentage of the number of input rows,
       then this partial aggregation will be a subject to being abandoned.
   * - session_timezone
     - string
     -
     - User provided session timezone. Stores a string with the actual timezone name, e.g: "America/Los_Angeles".
   * - adjust_timestamp_to_session_timezone
     - bool
     - false
     - If true, timezone-less timestamp conversions (e.g. string to timestamp, when the string does not specify a timezone)
       will be adjusted to the user provided `session_timezone` (if any). For instance: if this option is true and user
       supplied "America/Los_Angeles", then "1970-01-01" will be converted to -28800 instead of 0. Similarly, timestamp
       to date conversions will adhere to user 'session_timezone', e.g: Timestamp(0) to Date will be -1 (number of days
       since epoch) for "America/Los_Angeles".
   * - track_operator_cpu_usage
     - bool
     - true
     - Whether to track CPU usage for stages of individual operators. Can be expensive when processing small batches,
       e.g. < 10K rows.
   * - hash_adaptivity_enabled
     - bool
     - true
     - If false, the 'group by' code is forced to use generic hash mode hashtable.
   * - adaptive_filter_reordering_enabled
     - bool
     - true
     - If true, the conjunction expression can reorder inputs based on the time taken to calculate them.
   * - max_local_exchange_buffer_size
     - integer
     - 32MB
     - Used for backpressure to block local exchange producers when the local exchange buffer reaches or exceeds this size.
   * - exchange.max_buffer_size
     - integer
     - 32MB
     - Size of buffer in the exchange client that holds data fetched from other nodes before it is processed.
       A larger buffer can increase network throughput for larger clusters and thus decrease query processing time
       at the expense of reducing the amount of memory available for other usage.
   * - max_page_partitioning_buffer_size
     - integer
     - 32MB
     - The target size for a Task's buffered output. The producer Drivers are blocked when the buffered size exceeds this.
       The Drivers are resumed when the buffered size goes below PartitionedOutputBufferManager::kContinuePct (90)% of this.
   * - min_table_rows_for_parallel_join_build
     - integer
     - 1000
     - The minimum number of table rows that can trigger the parallel hash join table build.
   * - debug.validate_output_from_operators
     - bool
     - false
     - If set to true, then during execution of tasks, the output vectors of every operator are validated for consistency.
       This is an expensive check so should only be used for debugging. It can help debug issues where malformed vector
       cause failures or crashes by helping identify which operator is generating them.
   * - enable_expression_evaluation_cache
     - bool
     - true
     - Whether to enable caches in expression evaluation. If set to true, optimizations including vector pools and
       evalWithMemo are enabled.

.. _expression-evaluation-conf:

Expression Evaluation Configuration
-----------------------------------
.. list-table::
   :widths: 20 10 10 70
   :header-rows: 1

   * - Property Name
     - Type
     - Default Value
     - Description
   * - expression.eval_simplified
     - boolean
     - false
     - Whether to use the simplified expression evaluation path.
   * - expression.track_cpu_usage
     - boolean
     - false
     - Whether to track CPU usage for individual expressions (supported by call and cast expressions). Can be expensive
       when processing small batches, e.g. < 10K rows.
   * - cast_match_struct_by_name
     - bool
     - false
     - This flag makes the Row conversion to by applied in a way that the casting row field are matched by name instead of position.
   * - cast_to_int_by_truncate
     - bool
     - false
     - This flags forces the cast from float/double/decimal/string to integer to be performed by truncating the decimal part instead of rounding.
   * - cast_string_to_date_is_iso_8601
     - bool
     - true
     - If set, cast from string to date allows only ISO 8601 formatted strings: ``[+-](YYYY-MM-DD)``.
       Otherwise, allows all patterns supported by Spark:
         * ``[+-]yyyy*``
         * ``[+-]yyyy*-[m]m``
         * ``[+-]yyyy*-[m]m-[d]d``
         * ``[+-]yyyy*-[m]m-[d]d *``
         * ``[+-]yyyy*-[m]m-[d]dT*``
       The asterisk ``*`` in ``yyyy*`` stands for any numbers.
       For the last two patterns, the trailing ``*`` can represent none or any sequence of characters, e.g:
         * "1970-01-01 123"
         * "1970-01-01 (BC)"
       Regardless of this setting's value, leading spaces will be trimmed.

Memory Management
-----------------
.. list-table::
   :widths: 20 10 10 70
   :header-rows: 1

   * - Property Name
     - Type
     - Default Value
     - Description
   * - max_partial_aggregation_memory
     - integer
     - 16MB
     - Maximum amount of memory in bytes for partial aggregation results. Increasing this value can result in less
       network transfer and lower CPU utilization by allowing more groups to be kept locally before being flushed,
       at the cost of additional memory usage.
   * - max_extended_partial_aggregation_memory
     - integer
     - 16MB
     - Maximum amount of memory in bytes for partial aggregation results if cardinality reduction is below
       `partial_aggregation_reduction_ratio_threshold`. Every time partial aggregate results size reaches
       `max_partial_aggregation_memory` bytes, the results are flushed. If cardinality reduction is below
       `partial_aggregation_reduction_ratio_threshold`,
       i.e. `number of result rows / number of input rows > partial_aggregation_reduction_ratio_threshold`,
       memory limit for partial aggregation is automatically doubled up to `max_extended_partial_aggregation_memory`.
       This adaptation is disabled by default, since the value of `max_extended_partial_aggregation_memory` equals the
       value of `max_partial_aggregation_memory`. Specify higher value for `max_extended_partial_aggregation_memory` to enable.

Spilling
--------
.. list-table::
   :widths: 20 10 10 70
   :header-rows: 1

   * - Property Name
     - Type
     - Default Value
     - Description
   * - spill_enabled
     - boolean
     - false
     - Spill memory to disk to avoid exceeding memory limits for the query.
   * - aggregation_spill_enabled
     - boolean
     - true
     - When `spill_enabled` is true, determines whether HashAggregation operator can spill to disk under memory pressure.
       memory limits for the query.
   * - join_spill_enabled
     - boolean
     - true
     - When `spill_enabled` is true, determines whether HashBuild and HashProbe operators can spill to disk under memory pressure.
       limits for the query.
   * - order_by_spill_enabled
     - boolean
     - true
     - When `spill_enabled` is true, determines whether OrderBy operator can spill to disk under memory pressure.
       limits for the query.
   * - row_number_spill_enabled
     - boolean
     - true
     - When `spill_enabled` is true, determines whether RowNumber operator can spill to disk under memory pressure.
   * - topn_row_number_spill_enabled
     - boolean
     - true
     - When `spill_enabled` is true, determines whether TopNRowNumber operator can spill to disk under memory pressure.
   * - writer_spill_enabled
     - boolean
     - true
     - When `writer_spill_enabled` is true, determines whether TableWriter operator can spill to disk under memory pressure.
   * - aggregation_spill_memory_threshold
     - integer
     - 0
     - Maximum amount of memory in bytes that a final aggregation can use before spilling. 0 means unlimited.
   * - aggregation_spill_all
     - boolean
     - false
     - If true and spilling has been triggered during the input processing, the spiller will spill all the remaining in-memory state to disk before output processing. This is to simplify the aggregation query OOM prevention in output processing stage.
   * - join_spill_memory_threshold
     - integer
     - 0
     - Maximum amount of memory in bytes that a hash join build side can use before spilling. 0 means unlimited.
   * - order_by_spill_memory_threshold
     - integer
     - 0
     - Maximum amount of memory in bytes that an order by can use before spilling. 0 means unlimited.
   * - min_spillable_reservation_pct
     - integer
     - 5
     - The minimal available spillable memory reservation in percentage of the current memory usage. Suppose the current
       memory usage size of M, available memory reservation size of N and min reservation percentage of P,
       if M * P / 100 > N, then spiller operator needs to grow the memory reservation with percentage of
       'spillable_reservation_growth_pct' (see below). This ensures we have sufficient amount of memory reservation to
       process the large input outlier.
   * - spillable_reservation_growth_pct
     - integer
     - 10
     - The spillable memory reservation growth percentage of the current memory usage. Suppose a growth percentage of N
       and the current memory usage size of M, the next memory reservation size will be M * (1 + N / 100). After growing
       the memory reservation K times, the memory reservation size will be M * (1 + N / 100) ^ K. Hence the memory
       reservation grows along a series of powers of (1 + N / 100). If the memory reservation fails, it starts spilling.
   * - max_spill_level
     - integer
     - 4
     - The maximum allowed spilling level with zero being the initial spilling level. Applies to hash join build
       spilling which might use recursive spilling when the build table is very large. -1 means unlimited.
       In this case an extremely large query might run out of spilling partition bits. The max spill level
       can be used to prevent a query from using too much io and cpu resources.
   * - max_spill_file_size
     - integer
     - 0
     - The maximum allowed spill file size. Zero means unlimited.
   * - spill_write_buffer_size
     - integer
     - 4MB
     - The maximum size in bytes to buffer the serialized spill data before write to disk for IO efficiency.
       If set to zero, buffering is disabled.
   * - min_spill_run_size
     - integer
     - 256MB
     - The minimum spill run size (bytes) limit used to select partitions for spilling. The spiller tries to spill a
       previously spilled partitions if its data size exceeds this limit, otherwise it spills the partition with most data.
       If the limit is zero, then the spiller always spills a previously spilled partition if it has any data. This is
       to avoid spill from a partition with a small amount of data which might result in generating too many small
       spilled files.
   * - spill_compression_codec
     - string
     - none
     - Specifies the compression algorithm type to compress the spilled data before write to disk to trade CPU for IO
       efficiency. The supported compression codecs are: ZLIB, SNAPPY, LZO, ZSTD, LZ4 and GZIP.
       NONE means no compression.
   * - spiller_start_partition_bit
     - integer
     - 29
     - The start partition bit which is used with `spiller_partition_bits` together to calculate the spilling partition number.
   * - join_spiller_partition_bits
     - integer
     - 2
     - The number of bits (N) used to calculate the spilling partition number for hash join and RowNumber: 2 ^ N. At the moment the maximum
       value is 3, meaning we only support up to 8-way spill partitioning.
   * - aggregation_spiller_partition_bits
     - integer
     - 0
     - The number of bits (N) used to calculate the spilling partition number for hash aggregation: 2 ^ N. At the moment the
       maximum value is 3, meaning we only support up to 8-way spill partitioning.
   * - testing.spill_pct
     - integer
     - 0
     - Percentage of aggregation or join input batches that will be forced to spill for testing. 0 means no extra spilling.

Table Writer
------------
.. list-table::
   :widths: 20 10 10 70
   :header-rows: 1

   * - Property Name
     - Type
     - Default Value
     - Description
   * - task_writer_count
     - integer
     - 1
     - The number of parallel table writer threads per task.
   * - task_partitioned_writer_count
     - integer
     - task_writer_count
     - The number of parallel table writer threads per task for bucketed table writes. If not set, use 'task_writer_count' as default.

Codegen Configuration
---------------------
.. list-table::
   :widths: 20 10 10 70
   :header-rows: 1

   * - Property Name
     - Type
     - Default Value
     - Description
   * - codegen.enabled
     - boolean
     - false
     - Along with `codegen.configuration_file_path` enables codegen in task execution path.
   * - codegen.configuration_file_path
     - string
     -
     - A path to the file contaning codegen options.
   * - codegen.lazy_loading
     - boolean
     - true
     - Triggers codegen initialization tests upon loading if false. Otherwise skips them.

Hive Connector
--------------
.. list-table::
   :widths: 20 10 10 70
   :header-rows: 1

   * - Property Name
     - Type
     - Default Value
     - Description
   * - max_partitions_per_writers
     - integer
     - 100
     - Maximum number of (bucketed) partitions per a single table writer instance.
   * - insert_existing_partitions_behavior
     - string
     - ERROR
     - **Allowed values:** ``OVERWRITE``, ``ERROR``. The behavior on insert existing partitions. This property only derives
       the update mode field of the table writer operator output. ``OVERWRITE``
       sets the update mode to indicate overwriting a partition if exists. ``ERROR`` sets the update mode to indicate
       error throwing if writing to an existing partition.
   * - hive.immutable-partitions
     - bool
     - false
     - True if appending data to an existing unpartitioned table is allowed. Currently this configuration does not
       support appending to existing partitions.
   * - file_column_names_read_as_lower_case
     - bool
     - false
     - True if reading the source file column names as lower case, and planner should guarantee
       the input column name and filter is also lower case to achive case-insensitive read.
   * - max-coalesced-bytes
     - integer
     - 512KB
     - Maximum size in bytes to coalesce requests to be fetched in a single request.
   * - max-coalesced-distance-bytes
     - integer
     - 128MB
     - Maximum distance in bytes between chunks to be fetched that may be coalesced into a single request.
   * - file_writer_flush_threshold_bytes
     - integer
     - 96MB
     - Minimum memory footprint size required to reclaim memory from a file writer by flushing its buffered data to disk.

``Amazon S3 Configuration``
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. list-table::
   :widths: 30 10 10 70
   :header-rows: 1

   * - Property Name
     - Type
     - Default Value
     - Description
   * - hive.s3.use-instance-credentials
     - bool
     - true
     - Use the EC2 metadata service to retrieve API credentials. This works with IAM roles in EC2.
   * - hive.s3.aws-access-key
     - string
     -
     - Default AWS access key to use.
   * - hive.s3.aws-secret-key
     - string
     -
     - Default AWS secret key to use.
   * - hive.s3.endpoint
     - string
     -
     - The S3 storage endpoint server. This can be used to connect to an S3-compatible storage system instead of AWS.
   * - hive.s3.path-style-access
     - bool
     - false
     - Use path-style access for all requests to the S3-compatible storage. This is for S3-compatible storage that
       doesn't support virtual-hosted-style access.
   * - hive.s3.ssl.enabled
     - bool
     - true
     - Use HTTPS to communicate with the S3 API.
   * - hive.s3.log-level
     - string
     - FATAL
     - **Allowed values:** "OFF", "FATAL", "ERROR", "WARN", "INFO", "DEBUG", "TRACE"
       Granularity of logging generated by the AWS C++ SDK library.
   * - hive.s3.iam-role
     - string
     -
     - IAM role to assume.
   * - hive.s3.iam-role-session-name
     - string
     - velox-session
     - Session name associated with the IAM role.

``Google Cloud Storage Configuration``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. list-table::
   :widths: 30 10 10 60
   :header-rows: 1

   * - Property Name
     - Type
     - Default Value
     - Description
   * - hive.gcs.endpoint
     - string
     -
     - The GCS storage endpoint server.
   * - hive.gcs.scheme
     - string
     -
     - The GCS storage scheme, https for default credentials.
   * - hive.gcs.credentials
     - string
     -
     - The GCS service account configuration as json string.

Presto-specific Configuration
-----------------------------
.. list-table::
   :widths: 20 10 10 70
   :header-rows: 1

   * - Property Name
     - Type
     - Default Value
     - Description
   * - presto.array_agg.ignore_nulls
     - bool
     - false
     - If true, ``array_agg`` function ignores null inputs.

Spark-specific Configuration
----------------------------
.. list-table::
   :widths: 20 10 10 70
   :header-rows: 1

   * - Property Name
     - Type
     - Default Value
     - Description
   * - spark.legacy_size_of_null
     - bool
     - true
     - If false, ``size`` function returns null for null input.
   * - spark.bloom_filter.expected_num_items
     - integer
     - 1000000
     - The default number of expected items for the bloom filter in :spark:func:`bloom_filter_agg` function.
   * - spark.bloom_filter.num_bits
     - integer
     - 8388608
     - The default number of bits to use for the bloom filter in :spark:func:`bloom_filter_agg` function.
   * - spark.bloom_filter.max_num_bits
     - integer
     - 4194304
     - The maximum number of bits to use for the bloom filter in :spark:func:`bloom_filter_agg` function,
       the value of this config can not exceed the default value.
