=============
Runtime Stats
=============

Runtime stats are used to collect the per-query velox runtime events for
offline query analysis purpose. The collected stats can provide insights into
the operator level query execution internals, such as how much time a query
operator spent in disk spilling. The collected stats are organized in a
free-form key-value for easy extension. The key is the event name and the
value is defined as RuntimeCounter which is used to store and aggregate a
particular event occurrences during the operator execution. RuntimeCounter has
three types: kNone used to record event count, kNanos used to record event time
in nanoseconds and kBytes used to record memory or storage size in bytes. It
records the count of events, and the min/max/sum of the event values. The stats
are stored in OperatorStats structure. The query system can aggregate the
operator level stats collected from each driver by pipeline and task for
analysis.

Memory Arbitration
------------------
These stats are reported by all operators.

.. list-table::
   :widths: 50 25 50
   :header-rows: 1

   * - Stats
     - Unit
     - Description
   * - memoryReclaimCount
     -
     - The number of times that the memory arbitration to reclaim memory from
       an spillable operator.
       This stats only applies for spillable operators.
   * - memoryReclaimWallNanos
     - nanos
     - The memory reclaim execution time of an operator during the memory
       arbitration. It collects time spent on disk spilling or file write.
       This stats only applies for spillable operators.
   * - reclaimedMemoryBytes
     - bytes
     - The reclaimed memory bytes of an operator during the memory arbitration.
       This stats only applies for spillable operators.

HashBuild, HashAggregation
--------------------------
These stats are reported only by HashBuild and HashAggregation operators.

.. list-table::
   :widths: 50 25 50
   :header-rows: 1

   * - Stats
     - Unit
     - Description
   * - hashtable.capacity
     -
     - Number of slots across all buckets in the hash table.
   * - hashtable.numRehashes
     -
     - Number of rehash() calls.
   * - hashtable.numDistinct
     -
     - Number of distinct keys in the hash table.
   * - hashtable.numTombstones
     -
     - Number of tombstone slots in the hash table.
