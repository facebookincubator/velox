==================
printPlanWithStats
==================

Velox collects a number of valuable statistics during query execution.  These
counters are exposed via Task::taskStats() API for programmatic access and can
be printed in a human-friendly format for manual inspection. We use these to
reason about the query execution dynamic and troubleshoot performance issues.

If you are familiar with Presto, the tools described below would look very
similar to the PrestoQueryLookup tool available via bunnylol presto
<query-id>.

PlanNode::toString()
--------------------

PlanNode::toString() method prints a query plan as a tree of plan nodes. This
API can be used before or after executing a query.

PlanNode::toString() method takes two optional flags: detailed and recursive.
When detailed is true, the output includes extra details about each plan node.
When recursive is true, the output includes the whole plan tree, otherwise only
a single plan node is shown.

In “detailed” mode, Project node shows projection expressions, Filter node shows
filter expression, Join node shows join type and join keys, Aggregation node
shows grouping keys and aggregate functions, OrderBy node shows sorting keys
and orders, etc.

Let's use a simple join query as an example:

`plan->toString(false /*detailed*/, true /*recursive*/)` prints a plan tree using plan node names:

.. code-block::

    -> Project
      -> HashJoin
        -> TableScan
        -> Project
          -> Values

`plan->toString(true /*detailed*/, true /*recursive*/)` adds plan node details to each plan node.

.. code-block::

    -> Project[expressions: (c0:INTEGER, ROW["c0"]), (p1:BIGINT, plus(ROW["c1"],1)), (p2:BIGINT, plus(ROW["c1"],ROW["u_c1"])), ]
      -> HashJoin[INNER c0=u_c0]
        -> TableScan[]
        -> Project[expressions: (u_c0:INTEGER, ROW["c0"]), (u_c1:BIGINT, ROW["c1"]), ]
          -> Values[100 rows in 1 vectors]

Let’s also look at an aggregation query:

`plan->toString(false /*detailed*/, true /*recursive*/)`

.. code-block::

    -> Aggregation
      -> TableScan

`plan->toString(true /*detailed*/, true /*recursive*/)`

.. code-block::

    -> Aggregation[PARTIAL [c5] a0 := max(ROW["c0"]), a1 := sum(ROW["c1"]), a2 := sum(ROW["c2"]), a3 := sum(ROW["c3"]), a4 := sum(ROW["c4"])]
      -> TableScan[]

printPlanWithStats()
--------------------

printPlanWithStats() function prints a query plan annotated with runtime
statistics. This function can be used after the query finishes processing. It
takes a root plan node and a TaskStats struct.

By default, printPlanWithStats shows a number of output rows, CPU time, number of threads used, and peak
memory usage for each plan node.

`printPlanWithStats(*plan, task->taskStats())`

.. code-block::

   -- Project[4][expressions: (c0:INTEGER, ROW["c0"]), (p1:BIGINT, plus(ROW["c1"],1)), (p2:BIGINT, plus(ROW["c1"],ROW["u_c1"]))] -> c0:INTEGER, p1:BIGINT, p2:BIGINT
      Output: 2000 rows (154.34KB, 20 batches), Cpu time: 907.80us, Blocked wall time: 0ns, Peak memory: 2.00KB, Memory allocations: 40, Threads: 1, CPU breakdown: I/O/F (27.24us/872.82us/7.74us)
   -- HashJoin[3][INNER c0=u_c0] -> c0:INTEGER, c1:BIGINT, u_c1:BIGINT
      Output: 2000 rows (136.23KB, 20 batches), Cpu time: 508.74us, Blocked wall time: 0ns, Peak memory: 88.50KB, Memory allocations: 7, CPU breakdown: I/O/F (177.87us/329.20us/1.66us)
      HashBuild: Input: 100 rows (1.31KB, 1 batches), Output: 0 rows (0B, 0 batches), Cpu time: 41.77us, Blocked wall time: 0ns, Peak memory: 68.00KB, Memory allocations: 2, Threads: 1, CPU breakdown: I/O/F (40.18us/1.59us/0ns)
      HashProbe: Input: 2000 rows (118.12KB, 20 batches), Output: 2000 rows (136.23KB, 20 batches), Cpu time: 466.97us, Blocked wall time: 0ns, Peak memory: 20.50KB, Memory allocations: 5, Threads: 1, CPU breakdown: I/O/F (137.69us/327.61us/1.66us)
      -- TableScan[2][table: hive_table] -> c0:INTEGER, c1:BIGINT
         Input: 2000 rows (118.12KB, 20 batches), Raw Input: 20480 rows (72.79KB), Output: 2000 rows (118.12KB, 20 batches), Cpu time: 8.89ms, Blocked wall time: 10.00us, Peak memory: 80.38KB, Memory allocations: 262, Threads: 1, Splits: 20, DynamicFilter producer plan nodes: 3, CPU breakdown: I/O/F (0ns/8.88ms/4.93us)
      -- Project[1][expressions: (u_c0:INTEGER, ROW["c0"]), (u_c1:BIGINT, ROW["c1"])] -> u_c0:INTEGER, u_c1:BIGINT
         Output: 100 rows (1.31KB, 1 batches), Cpu time: 43.22us, Blocked wall time: 0ns, Peak memory: 0B, Memory allocations: 0, Threads: 1, CPU breakdown: I/O/F (691ns/5.54us/36.98us)
         -- Values[0][100 rows in 1 vectors] -> c0:INTEGER, c1:BIGINT
            Input: 0 rows (0B, 0 batches), Output: 100 rows (1.31KB, 1 batches), Cpu time: 3.05us, Blocked wall time: 0ns, Peak memory: 0B, Memory allocations: 0, Threads: 1, CPU breakdown: I/O/F (0ns/2.48us/568ns)

With includeCustomStats flag enabled, printPlanWithStats adds operator-specific
statistics for each plan node, e.g. number of distinct values for the join key,
number of row groups skipped in table scan, amount of data read from cache and
storage in table scan, number of rows processed via aggregation pushdown into
scan, etc.

Here is the output for the join query from above.

`printPlanWithStats(*plan, task->taskStats(), true)` shows custom operator statistics.

.. code-block::

   -- Project[4][expressions: (c0:INTEGER, ROW["c0"]), (p1:BIGINT, plus(ROW["c1"],1)), (p2:BIGINT, plus(ROW["c1"],ROW["u_c1"]))] -> c0:INTEGER, p1:BIGINT, p2:BIGINT
      Output: 2000 rows (154.34KB, 20 batches), Cpu time: 907.80us, Blocked wall time: 0ns, Peak memory: 2.00KB, Memory allocations: 40, Threads: 1, CPU breakdown: I/O/F (27.24us/872.82us/7.74us)
         dataSourceLazyCpuNanos       sum: 745.18us, count: 20, min: 28.42us, max: 57.84us
         dataSourceLazyWallNanos      sum: 765.26us, count: 1, min: 765.26us, max: 765.26us
         runningAddInputWallNanos     sum: 41.22us, count: 1, min: 41.22us, max: 41.22us
         runningFinishWallNanos       sum: 8.42us, count: 1, min: 8.42us, max: 8.42us
         runningGetOutputWallNanos    sum: 888.74us, count: 1, min: 888.74us, max: 888.74us
   -- HashJoin[3][INNER c0=u_c0] -> c0:INTEGER, c1:BIGINT, u_c1:BIGINT
      Output: 2000 rows (136.23KB, 20 batches), Cpu time: 508.74us, Blocked wall time: 0ns, Peak memory: 88.50KB, Memory allocations: 7, CPU breakdown: I/O/F (177.87us/329.20us/1.66us)
      HashBuild: Input: 100 rows (1.31KB, 1 batches), Output: 0 rows (0B, 0 batches), Cpu time: 41.77us, Blocked wall time: 0ns, Peak memory: 68.00KB, Memory allocations: 2, Threads: 1, CPU breakdown: I/O/F (40.18us/1.59us/0ns)
         distinctKey0                 sum: 101, count: 1, min: 101, max: 101
         hashtable.buildWallNanos     sum: 28.77us, count: 1, min: 28.77us, max: 28.77us
         hashtable.capacity           sum: 200, count: 1, min: 200, max: 200
         hashtable.numDistinct        sum: 100, count: 1, min: 100, max: 100
         hashtable.numRehashes        sum: 1, count: 1, min: 1, max: 1
         queuedWallNanos              sum: 131.00us, count: 1, min: 131.00us, max: 131.00us
         rangeKey0                    sum: 200, count: 1, min: 200, max: 200
         runningAddInputWallNanos     sum: 40.41us, count: 1, min: 40.41us, max: 40.41us
         runningFinishWallNanos       sum: 0ns, count: 1, min: 0ns, max: 0ns
         runningGetOutputWallNanos    sum: 2.80us, count: 1, min: 2.80us, max: 2.80us
      HashProbe: Input: 2000 rows (118.12KB, 20 batches), Output: 2000 rows (136.23KB, 20 batches), Cpu time: 466.97us, Blocked wall time: 0ns, Peak memory: 20.50KB, Memory allocations: 5, Threads: 1, CPU breakdown: I/O/F (137.69us/327.61us/1.66us)
         dynamicFiltersProduced       sum: 1, count: 1, min: 1, max: 1
         runningAddInputWallNanos     sum: 154.36us, count: 1, min: 154.36us, max: 154.36us
         runningFinishWallNanos       sum: 2.35us, count: 1, min: 2.35us, max: 2.35us
         runningGetOutputWallNanos    sum: 361.26us, count: 1, min: 361.26us, max: 361.26us
      -- TableScan[2][table: hive_table] -> c0:INTEGER, c1:BIGINT
         Input: 2000 rows (118.12KB, 20 batches), Raw Input: 20480 rows (72.79KB), Output: 2000 rows (118.12KB, 20 batches), Cpu time: 8.89ms, Blocked wall time: 0ns, Peak memory: 80.38KB, Memory allocations: 262, Threads: 1, Splits: 20, DynamicFilter producer plan nodes: 3, CPU breakdown: I/O/F (0ns/8.88ms/4.93us)
            dataSourceAddSplitWallNanos      sum: 464.00us, count: 1, min: 464.00us, max: 464.00us
            dataSourceReadWallNanos          sum: 1.88ms, count: 1, min: 1.88ms, max: 1.88ms
            dynamicFiltersAccepted           sum: 1, count: 1, min: 1, max: 1
            flattenStringDictionaryValues    sum: 0, count: 1, min: 0, max: 0
            ioWaitWallNanos                  sum: 337.00us, count: 1, min: 337.00us, max: 337.00us
            localReadBytes                   sum: 0B, count: 1, min: 0B, max: 0B
            maxSingleIoWaitWallNanos         sum: 32.00us, count: 1, min: 32.00us, max: 32.00us
            numLocalRead                     sum: 0, count: 1, min: 0, max: 0
            numPrefetch                      sum: 18, count: 1, min: 18, max: 18
            numRamRead                       sum: 60, count: 1, min: 60, max: 60
            numStorageRead                   sum: 100, count: 1, min: 100, max: 100
            overreadBytes                    sum: 0B, count: 1, min: 0B, max: 0B
            prefetchBytes                    sum: 58.34KB, count: 1, min: 58.34KB, max: 58.34KB
            preloadedSplits                  sum: 19, count: 19, min: 1, max: 1
            ramReadBytes                     sum: 1.48KB, count: 1, min: 1.48KB, max: 1.48KB
            readyPreloadedSplits             sum: 17, count: 17, min: 1, max: 1
            runningAddInputWallNanos         sum: 0ns, count: 1, min: 0ns, max: 0ns
            runningFinishWallNanos           sum: 5.76us, count: 1, min: 5.76us, max: 5.76us
            runningGetOutputWallNanos        sum: 9.83ms, count: 1, min: 9.83ms, max: 9.83ms
            skippedSplitBytes                sum: 0B, count: 1, min: 0B, max: 0B
            skippedSplits                    sum: 0, count: 1, min: 0, max: 0
            skippedStrides                   sum: 0, count: 1, min: 0, max: 0
            storageReadBytes                 sum: 72.79KB, count: 1, min: 72.79KB, max: 72.79KB
            totalRemainingFilterTime         sum: 0ns, count: 1, min: 0ns, max: 0ns
            totalScanTime                    sum: 309.00us, count: 1, min: 309.00us, max: 309.00us
      -- Project[1][expressions: (u_c0:INTEGER, ROW["c0"]), (u_c1:BIGINT, ROW["c1"])] -> u_c0:INTEGER, u_c1:BIGINT
         Output: 100 rows (1.31KB, 1 batches), Cpu time: 43.22us, Blocked wall time: 0ns, Peak memory: 0B, Memory allocations: 0, Threads: 1, CPU breakdown: I/O/F (691ns/5.54us/36.98us)
            runningAddInputWallNanos     sum: 982ns, count: 1, min: 982ns, max: 982ns
            runningFinishWallNanos       sum: 37.20us, count: 1, min: 37.20us, max: 37.20us
            runningGetOutputWallNanos    sum: 6.34us, count: 1, min: 6.34us, max: 6.34us
         -- Values[0][100 rows in 1 vectors] -> c0:INTEGER, c1:BIGINT
            Input: 0 rows (0B, 0 batches), Output: 100 rows (1.31KB, 1 batches), Cpu time: 3.05us, Blocked wall time: 0ns, Peak memory: 0B, Memory allocations: 0, Threads: 1, CPU breakdown: I/O/F (0ns/2.48us/568ns)
               runningAddInputWallNanos     sum: 0ns, count: 1, min: 0ns, max: 0ns
               runningFinishWallNanos       sum: 782ns, count: 1, min: 782ns, max: 782ns
               runningGetOutputWallNanos    sum: 2.87us, count: 1, min: 2.87us, max: 2.87us

And this is the output for the aggregation query from above.

`printPlanWithStats(*plan, task->taskStats())` shows basic statistics:

.. code-block::

   -- Aggregation[1][PARTIAL [c5] a0 := max(ROW["c0"]), a1 := sum(ROW["c1"]), a2 := sum(ROW["c2"]), a3 := sum(ROW["c3"]), a4 := sum(ROW["c4"])] -> c5:VARCHAR, a0:BIGINT, a1:BIGINT, a2:BIGINT, a3:DOUBLE, a4:DOUBLE
      Output: 835 rows (672.38KB, 1 batches), Cpu time: 1.96ms, Blocked wall time: 0ns, Peak memory: 757.25KB, Memory allocations: 20, Threads: 1, CPU breakdown: I/O/F (1.38ms/579.12us/6.82us)
   -- TableScan[0][table: hive_table] -> c0:BIGINT, c1:INTEGER, c2:SMALLINT, c3:REAL, c4:DOUBLE, c5:VARCHAR
      Input: 10000 rows (0B, 1 batches), Output: 10000 rows (0B, 1 batches), Cpu time: 2.89ms, Blocked wall time: 0ns, Peak memory: 539.88KB, Memory allocations: 69, Threads: 1, Splits: 1, CPU breakdown: I/O/F (0ns/2.89ms/3.35us)

`printPlanWithStats(*plan, task->taskStats(), true)` includes custom statistics:

.. code-block::

   -- Aggregation[1][PARTIAL [c5] a0 := max(ROW["c0"]), a1 := sum(ROW["c1"]), a2 := sum(ROW["c2"]), a3 := sum(ROW["c3"]), a4 := sum(ROW["c4"])] -> c5:VARCHAR, a0:BIGINT, a1:BIGINT, a2:BIGINT, a3:DOUBLE, a4:DOUBLE
      Output: 835 rows (672.38KB, 1 batches), Cpu time: 1.96ms, Blocked wall time: 0ns, Peak memory: 757.25KB, Memory allocations: 20, Threads: 1, CPU breakdown: I/O/F (1.38ms/579.12us/6.82us)
         dataSourceLazyCpuNanos       sum: 2.10ms, count: 6, min: 245.71us, max: 554.09us
         dataSourceLazyWallNanos      sum: 2.10ms, count: 1, min: 2.10ms, max: 2.10ms
         distinctKey0                 sum: 835, count: 1, min: 835, max: 835
         hashtable.capacity           sum: 1252, count: 1, min: 1252, max: 1252
         hashtable.numDistinct        sum: 835, count: 1, min: 835, max: 835
         hashtable.numRehashes        sum: 1, count: 1, min: 1, max: 1
         hashtable.numTombstones      sum: 0, count: 1, min: 0, max: 0
         loadedToValueHook            sum: 50000, count: 5, min: 10000, max: 10000
         runningAddInputWallNanos     sum: 1.37ms, count: 1, min: 1.37ms, max: 1.37ms
         runningFinishWallNanos       sum: 7.57us, count: 1, min: 7.57us, max: 7.57us
         runningGetOutputWallNanos    sum: 582.59us, count: 1, min: 582.59us, max: 582.59us
   -- TableScan[0][table: hive_table] -> c0:BIGINT, c1:INTEGER, c2:SMALLINT, c3:REAL, c4:DOUBLE, c5:VARCHAR
      Input: 10000 rows (0B, 1 batches), Output: 10000 rows (0B, 1 batches), Cpu time: 2.89ms, Blocked wall time: 0ns, Peak memory: 539.88KB, Memory allocations: 69, Threads: 1, Splits: 1, CPU breakdown: I/O/F (0ns/2.89ms/3.35us)
         dataSourceAddSplitWallNanos      sum: 513.00us, count: 1, min: 513.00us, max: 513.00us
         dataSourceReadWallNanos          sum: 180.00us, count: 1, min: 180.00us, max: 180.00us
         flattenStringDictionaryValues    sum: 0, count: 1, min: 0, max: 0
         ioWaitWallNanos                  sum: 142.00us, count: 1, min: 142.00us, max: 142.00us
         localReadBytes                   sum: 0B, count: 1, min: 0B, max: 0B
         maxSingleIoWaitWallNanos         sum: 105.00us, count: 1, min: 105.00us, max: 105.00us
         numLocalRead                     sum: 0, count: 1, min: 0, max: 0
         numPrefetch                      sum: 0, count: 1, min: 0, max: 0
         numRamRead                       sum: 7, count: 1, min: 7, max: 7
         numStorageRead                   sum: 5, count: 1, min: 5, max: 5
         overreadBytes                    sum: 0B, count: 1, min: 0B, max: 0B
         prefetchBytes                    sum: 0B, count: 1, min: 0B, max: 0B
         ramReadBytes                     sum: 295B, count: 1, min: 295B, max: 295B
         runningAddInputWallNanos         sum: 0ns, count: 1, min: 0ns, max: 0ns
         runningFinishWallNanos           sum: 4.09us, count: 1, min: 4.09us, max: 4.09us
         runningGetOutputWallNanos        sum: 2.90ms, count: 1, min: 2.90ms, max: 2.90ms
         skippedSplitBytes                sum: 0B, count: 1, min: 0B, max: 0B
         skippedSplits                    sum: 0, count: 1, min: 0, max: 0
         skippedStrides                   sum: 0, count: 1, min: 0, max: 0
         storageReadBytes                 sum: 29.16KB, count: 1, min: 29.16KB, max: 29.16KB
         totalRemainingFilterTime         sum: 0ns, count: 1, min: 0ns, max: 0ns
         totalScanTime                    sum: 37.00us, count: 1, min: 37.00us, max: 37.00us

Common operator statistics
--------------------------

Let's take a closer look at statistics that are collected for all operators.

For each operator, Velox tracks the total number of input rows, output rows,
their estimated sizes, cpu time, blocked wall time, and the number of threads used to run the operator.

.. code-block::

   -- TableScan[2][table: hive_table] -> c0:INTEGER, c1:BIGINT
         Input: 2000 rows (118.12KB, 20 batches), Raw Input: 20480 rows (72.79KB), Output: 2000 rows (118.12KB, 20 batches), Cpu time: 8.89ms, Blocked wall time: 10.00us, Peak memory: 80.38KB, Memory allocations: 262, Threads: 1, Splits: 20, DynamicFilter producer plan nodes: 3, CPU breakdown: I/O/F (0ns/8.88ms/4.93us)

printPlanWithStats shows output rows and
sizes for each plan node and shows input rows and sizes for leaf nodes and nodes
that expand to multiple operators. Showing input rows for other nodes is redundant
since the number of input rows equals the number of output rows of the immediate child plan node.

.. code-block::

	Input: 2000 rows (118.12KB), Output: 2000 rows (118.12KB, 20 batches)

When rows are pruned for a TableScan with filters, Velox reports the number
of raw input rows and their total size. These are the rows processed before
applying the pushed down filters.
TableScan also reports the number of splits assigned.

.. code-block::

	Raw Input: 20480 rows (72.79KB), Splits: 20

Velox also measures CPU time and the breakdown of CPU time which including addInput, getOutput and finish time,
, peak memory usage and the mumber of memory allocations for each operator. This information is shown for all plan nodes.

.. code-block::

	Cpu time: 8.89ms, Peak memory: 80.38KB, Memory allocations: 262, CPU breakdown: I/O/F (0ns/8.88ms/4.93us)

Some operators like TableScan and HashProbe may be blocked waiting for splits or
hash tables. Velox records the total wall time an operator was blocked and
printPlanWithStats shows this information as “Blocked wall time”.

.. code-block::

	Blocked wall time: 10.00us

Some operators like TableScan may produce the dynamic filter, reports the plan node ids.
There may be several plan node ids showed with separater ``,`` while the example plan only contains 1 plan node id.

.. code-block::

	DynamicFilter producer plan nodes: 3

Custom operator statistics
--------------------------

Operators also collect and report operator-specific statistics.

TableScan operator reports statistics that show how much data has been read from
cache vs. durable storage, how much data was prefetched, how many files and row
groups were skipped via stats-based pruning.

.. code-block::

  -- TableScan[0][table: hive_table] -> c0:BIGINT, c1:INTEGER, c2:SMALLINT, c3:REAL, c4:DOUBLE, c5:VARCHAR
        dataSourceAddSplitWallNanos      sum: 513.00us, count: 1, min: 513.00us, max: 513.00us
        dataSourceReadWallNanos          sum: 180.00us, count: 1, min: 180.00us, max: 180.00us
        flattenStringDictionaryValues    sum: 0, count: 1, min: 0, max: 0
        ioWaitWallNanos                  sum: 142.00us, count: 1, min: 142.00us, max: 142.00us
        localReadBytes                   sum: 0B, count: 1, min: 0B, max: 0B
        maxSingleIoWaitWallNanos         sum: 105.00us, count: 1, min: 105.00us, max: 105.00us
        numLocalRead                     sum: 0, count: 1, min: 0, max: 0
        numPrefetch                      sum: 0, count: 1, min: 0, max: 0
        numRamRead                       sum: 7, count: 1, min: 7, max: 7
        numStorageRead                   sum: 5, count: 1, min: 5, max: 5
        overreadBytes                    sum: 0B, count: 1, min: 0B, max: 0B
        prefetchBytes                    sum: 0B, count: 1, min: 0B, max: 0B
        ramReadBytes                     sum: 295B, count: 1, min: 295B, max: 295B
        runningAddInputWallNanos         sum: 0ns, count: 1, min: 0ns, max: 0ns
        runningFinishWallNanos           sum: 4.09us, count: 1, min: 4.09us, max: 4.09us
        runningGetOutputWallNanos        sum: 2.90ms, count: 1, min: 2.90ms, max: 2.90ms
        skippedSplitBytes                sum: 0B, count: 1, min: 0B, max: 0B
        skippedSplits                    sum: 0, count: 1, min: 0, max: 0
        skippedStrides                   sum: 0, count: 1, min: 0, max: 0
        storageReadBytes                 sum: 29.16KB, count: 1, min: 29.16KB, max: 29.16KB
        totalRemainingFilterTime         sum: 0ns, count: 1, min: 0ns, max: 0ns
        totalScanTime                    sum: 37.00us, count: 1, min: 37.00us, max: 37.00us

HashBuild operator reports range, number of distinct values for the join keys, and the running time wall nanos.

.. code-block::

   -- HashJoin[3][INNER c0=u_c0] -> c0:INTEGER, c1:BIGINT, u_c1:BIGINT
      HashBuild:
         distinctKey0                 sum: 101, count: 1, min: 101, max: 101
         hashtable.buildWallNanos     sum: 28.77us, count: 1, min: 28.77us, max: 28.77us
         hashtable.capacity           sum: 200, count: 1, min: 200, max: 200
         hashtable.numDistinct        sum: 100, count: 1, min: 100, max: 100
         hashtable.numRehashes        sum: 1, count: 1, min: 1, max: 1
         queuedWallNanos              sum: 131.00us, count: 1, min: 131.00us, max: 131.00us
         rangeKey0                    sum: 200, count: 1, min: 200, max: 200
         runningAddInputWallNanos     sum: 40.41us, count: 1, min: 40.41us, max: 40.41us
         runningFinishWallNanos       sum: 0ns, count: 1, min: 0ns, max: 0ns
         runningGetOutputWallNanos    sum: 2.80us, count: 1, min: 2.80us, max: 2.80us

HashProbe operator reports whether it generated dynamic filter and the running time wall nanos. And TableScan
operator reports whether it received dynamic filter pushed down from the join.

.. code-block::

   -- HashJoin[3][INNER c0=u_c0] -> c0:INTEGER, c1:BIGINT, u_c1:BIGINT
      HashProbe:
         dynamicFiltersProduced       sum: 1, count: 1, min: 1, max: 1
         runningAddInputWallNanos     sum: 154.36us, count: 1, min: 154.36us, max: 154.36us
         runningFinishWallNanos       sum: 2.35us, count: 1, min: 2.35us, max: 2.35us
         runningGetOutputWallNanos    sum: 361.26us, count: 1, min: 361.26us, max: 361.26us

Aggregation operator shows datasource time, number of distinct values for the group keys, hash table information
and how many rows were processed by pushing down aggregation into TableScan.

.. code-block::

   -- Aggregation[1][PARTIAL [c5] a0 := max(ROW["c0"]), a1 := sum(ROW["c1"]), a2 := sum(ROW["c2"]), a3 := sum(ROW["c3"]), a4 := sum(ROW["c4"])] -> c5:VARCHAR, a0:BIGINT, a1:BIGINT, a2:BIGINT, a3:DOUBLE, a4:DOUBLE
         dataSourceLazyCpuNanos       sum: 2.10ms, count: 6, min: 245.71us, max: 554.09us
         dataSourceLazyWallNanos      sum: 2.10ms, count: 1, min: 2.10ms, max: 2.10ms
         distinctKey0                 sum: 835, count: 1, min: 835, max: 835
         hashtable.capacity           sum: 1252, count: 1, min: 1252, max: 1252
         hashtable.numDistinct        sum: 835, count: 1, min: 835, max: 835
         hashtable.numRehashes        sum: 1, count: 1, min: 1, max: 1
         hashtable.numTombstones      sum: 0, count: 1, min: 0, max: 0
         loadedToValueHook            sum: 50000, count: 5, min: 10000, max: 10000
         runningAddInputWallNanos     sum: 1.37ms, count: 1, min: 1.37ms, max: 1.37ms
         runningFinishWallNanos       sum: 7.57us, count: 1, min: 7.57us, max: 7.57us
         runningGetOutputWallNanos    sum: 582.59us, count: 1, min: 582.59us, max: 582.59us
