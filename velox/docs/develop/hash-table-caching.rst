===================
Hash Table Caching
===================

Background
----------

In materialized execution engines like Spark and Presto on Spark, for broadcast joins,
the build side splits are replicated to all join tasks due to upfront split planning.
This kind of upfront split planning allows these engines to provide task level fault tolerance
as the input splits of the tasks are tracked and output data can be discarded,
thus enabling task level retries.

But due to this, each task independently builds an identical hash table from the
same data. For large build sides this is wasteful: every task spends CPU and memory
constructing the same hash table that another task in the same query has already built.

The Build IO Tax
^^^^^^^^^^^^^^^^

Also, the broadcast data follows a write-once-read-many I/O pattern. Each task re-reads
the build side data independently. When the number of tasks is large ---
O(100k) tasks across 10k+ workers --- these concurrent reads overwhelm the I/O
service layer, leading to throttling.

Throttling causes tasks to stall for seconds to minutes waiting for I/O. When
queries are charged for reserved workers, these stalls mean reserved resources
sit idle, increasing query cost. Beyond I/O fetch delays, when the hash table is
large (in the gigabyte range), the CPU cost of rebuilding it per task is also
significant and wasteful.

Hash table caching eliminates this redundant work by allowing the first task to
build the hash table and making it available to all subsequent tasks in the same
Velox instance. This is a build-once, reuse-many paradigm. In Sapphire-Velox,
this implements a once-per-worker model that yields more than an order of
magnitude savings, since the number of tasks far exceeds the number of workers.

Enabling Hash Table Caching
----------------------------

Hash table caching is enabled by setting the ``useHashTableCache`` flag to
``true`` on the ``HashJoinNode``:

.. code-block:: c++

    auto joinNode =
        core::HashJoinNode::Builder()
            .id(planNodeIdGenerator->next())
            .joinType(core::JoinType::kInner)
            .nullAware(false)
            .leftKeys({leftKeyField})
            .rightKeys({rightKeyField})
            .left(probeNode)
            .right(buildNode)
            .outputType(outputType)
            .useHashTableCache(true)
            .build();

When ``useHashTableCache`` is false (the default), the hash join behaves
exactly as before. The flag is only intended for broadcast joins and is
currently used by Presto-on-Spark.

Overall Design
--------------

Hash table caching introduces a global singleton ``HashTableCache`` that stores
built hash tables keyed by ``queryId:planNodeId``. The cache coordinates
between tasks so that exactly one task builds the hash table while other tasks
wait and then reuse the result.

The ``HashTableCache`` is a process-wide singleton in the Velox instance,
alongside the ``AsyncDataCache`` and ``MemoryManager``. The cache and its
methods provide building blocks for drivers within a task and tasks within a
worker to coordinate hash table construction and reuse.

The design has three main components:

1. **HashTableCache** - A process-wide singleton that stores and manages cached
   hash table entries.
2. **HashTableCacheEntry** - A cache entry that holds the hash table, build
   coordination state, and a dedicated memory pool.
3. **HashBuild operator integration** - Logic in the HashBuild operator to
   check the cache, build or wait, and store the result.

Cache Structure
---------------

``HashTableCache`` is a thread-safe singleton that maps cache keys to cache
entries:

.. code-block:: c++

    class HashTableCache {
        std::mutex lock_;
        std::unordered_map<std::string,
                           std::shared_ptr<HashTableCacheEntry>> tables_;
    };

Each ``HashTableCacheEntry`` contains:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Field
     - Description
   * - ``cacheKey``
     - The key used to look up this entry (``queryId:planNodeId``).
   * - ``builderTaskId``
     - The task ID of the task that is responsible for building the table.
   * - ``tablePool``
     - A leaf memory pool under the query pool used for table allocations.
   * - ``table``
     - The built ``BaseHashTable``, set once build is complete.
   * - ``hasNullKeys``
     - Whether the build side contained null join keys.
   * - ``buildComplete``
     - Atomic flag indicating whether the table has been fully built.
   * - ``buildPromises``
     - Promises used to notify waiting tasks when build completes.

Cache API
---------

The ``HashTableCache`` exposes three methods. All decisions are made under a
single ``std::mutex``. The lock is held only for map lookups, inserts, and
promise creation --- never during table building or memory allocation.

get()
^^^^^

The ``get()`` method is the central coordination point. It is called by every
``HashBuild`` operator during ``initialize()`` and determines the caller's
role under the mutex:

.. code-block:: text

    get(key, taskId, queryCtx, *future):
        lock(lock_)
        Case 1 – No entry:    create entry, set builderTaskId → return entry (Builder)
        Case 2 – Same task:   return entry (Builder, coordinate via JoinBridge)
        Case 3 – Diff task, not complete: push promise → return entry + future (Waiter)
        Case 4 – Complete:    return entry (Late Arrival)

When creating a new entry, ``get()`` allocates a ``tablePool`` as a leaf child
of the query pool and registers a ``QueryCtx`` release callback that calls
``drop()`` on query destruction.

Key design decisions in ``get()``:

- **Memory pool ownership**: The ``tablePool`` is a leaf child of the first
  caller's ``QueryCtx`` root pool. All drivers in the builder task share this
  pool for partial table allocations (via ``HashBuild::tableMemoryPool()``),
  tying the cached table's memory accounting to the originating query.
- **Cleanup callback**: ``QueryCtx::addReleaseCallback`` ensures the cache entry
  is dropped when the query finishes. ``drop()`` resets the
  ``shared_ptr<BaseHashTable>`` outside the lock to free memory before the entry
  is destroyed.
- **Lock scope**: All decisions are made under a single ``std::mutex``. The lock
  is held only for map lookups/inserts and promise creation --- never during
  table building or memory allocation.

put()
^^^^^

Called by the last driver of the builder task after merging all partial tables.
Publishes the table and wakes all waiters:

.. code-block:: text

    put(key, table, hasNullKeys):
        lock(lock_)
        entry.table = table
        entry.buildComplete = true
        promises = move(entry.buildPromises)
        unlock(lock_)
        for each promise: promise.setValue()   // wake waiters outside lock

drop()
^^^^^^

Removes a cache entry and frees the table memory. Called by the ``QueryCtx``
cleanup callback when the query is destroyed:

.. code-block:: text

    drop(key):
        lock(lock_)
        entry = move(tables_[key])
        tables_.erase(key)
        unlock(lock_)
        entry.table.reset()   // free memory outside lock

Build Coordination
------------------

When hash table caching is enabled, the HashBuild operator calls
``HashTableCache::get()`` during initialization. The cache uses the first
caller's task as the builder and makes subsequent callers wait.

Builder Task
^^^^^^^^^^^^

The first task to call ``get()`` for a given key creates the cache entry and
becomes the builder. This task proceeds through the normal HashBuild flow:
all its drivers build partial hash tables, the last driver merges them, and
the merged table is stored in the cache via ``HashTableCache::put()``.

Drivers within the builder task coordinate with each other through the
existing ``HashJoinBridge`` mechanism. The cache does not interfere with
intra-task driver synchronization.

Waiter Tasks
^^^^^^^^^^^^

When a task calls ``get()`` and finds that another task is already building the
table (``builderTaskId`` differs from its own task ID and ``buildComplete`` is
false), it receives a ``ContinueFuture`` and transitions to the
``kWaitForBuild`` state. The task is suspended until the builder task calls
``put()``, which fulfills all waiting promises.

Once notified, the waiter task calls ``noMoreInput()`` which finds the table
in the cache and passes it directly to the ``HashJoinBridge`` without building
anything. The probe side then runs normally against the cached table.

.. code-block:: text

    Task 1 (Builder)              Task 2 (Waiter)              Task 3 (Waiter)
    ────────────────              ───────────────              ───────────────
    get() → creates entry         get() → sees builder         get() → sees builder
    builds hash table             receives future              receives future
    put() → sets table            (suspended)                  (suspended)
    notifies waiters ──────────→  wakes up                     wakes up
                                  uses cached table            uses cached table

Cache Hit
^^^^^^^^^

If a task calls ``get()`` and finds ``buildComplete`` is already true, the
cached table is returned immediately. The HashBuild operator skips all build
logic and passes the table to the ``HashJoinBridge``.

The HashBuild operator reports cache hits and misses via runtime statistics:

- ``hashtable.cacheHit`` - Table was found in the cache and reused.
- ``hashtable.cacheMiss`` - Table was not in the cache; this task built it.

Usage by HashBuild
------------------

The HashBuild operator uses the cache in a three-phase protocol: build, synchronize,
and probe.

Step 1: Build Phase (Producer)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When a ``HashBuild`` operator is initialized, it checks the cache via
``setupCachedHashTable()``.

- **Cache Miss (Builder)**: The first task to find a miss becomes the Builder.
  It creates a cache entry, pulls data from storage, builds the
  ``BaseHashTable``, and calls ``put()`` to publish it. Within the Builder task,
  subsequent drivers also call ``get()`` and receive the same entry (since
  ``builderTaskId == taskId``). Each driver calls ``setupTable()`` to allocate
  its own partial ``BaseHashTable`` using ``cacheEntry->tablePool``, receives its
  subset of input via ``addInput()``, and builds a partial table. Intra-task
  coordination between these drivers uses the standard ``allPeersFinished()`` /
  ``JoinBridge`` mechanism, not the cache.

- **The Wait (Waiters)**: If other tasks arrive while the Builder is building,
  they encounter the pending state and transition to ``kWaitForBuild``, waiting
  on a ``ContinueFuture`` provided by the cache.

- **Short-circuiting Upstream**: Once the Builder publishes the table, waiters
  are unblocked. Upon receiving the cached table, waiter tasks set their
  no-more-input flags. This short-circuits their source operators (e.g.,
  ``TableScan``), immediately stopping further data retrieval.

Step 2: Synchronization (JoinBridge)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``HashJoinBridge`` acts as the hand-off point between the build and probe
sides. Even if the table was retrieved from the cache rather than built locally,
the bridge ensures the probe side is notified that the data is ready for
processing. Both builder and waiter tasks call
``joinBridge.setHashTable()`` to publish the table (or cached table) to the
probe operators.

Step 3: Probe Phase (Consumer)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``HashProbe`` operator takes the cached table from the bridge and executes
as usual. Because the table is held as a ``shared_ptr``, the probe operator's
reference prevents the cache from freeing the table while a join is actively
scanning it. Once the probe finishes, the reference count is decremented. The
table is ultimately freed when the ``QueryCtx`` release callback calls
``drop()``.

HashBuild Lifecycle
-------------------

The following pseudocode shows the complete lifecycle of a ``HashBuild``
operator when hash table caching is enabled. Only the key function calls are
shown.

Initialization
^^^^^^^^^^^^^^

.. code-block:: text

    initialize():
        cacheKey = "queryId:planNodeId"
        cacheEntry = HashTableCache::instance()->get(cacheKey, taskId, queryCtx, &future_)

        if cacheEntry.buildComplete:        // Late Arrival
            noMoreInput()                   // → finishHashBuild() → getHashTableFromCache()
            return

        if future_.valid():                 // Waiter
            state = kWaitForBuild
            return

        // Builder: proceed with normal table setup
        setupTable()                        // allocate BaseHashTable using cacheEntry.tablePool
        setupSpiller()                      // no-op: canSpill() returns false with cache

Build and Publish
^^^^^^^^^^^^^^^^^

.. code-block:: text

    noMoreInput() → finishHashBuild():
        if not allPeersFinished():          // wait for peer drivers in same task
            state = kWaitForBuild
            return

        if getHashTableFromCache():         // Waiter or Late Arrival: cache has table
            joinBridge.setHashTable(cacheEntry.table, hasNullKeys)
            return

        // Builder (last driver): merge and publish
        table_.prepareJoinTable(otherTables)
        HashTableCache::instance()->put(cacheKey, table_, hasNullKeys)
        joinBridge.setHashTable(table_, hasNullKeys)

Waiter Wake-up
^^^^^^^^^^^^^^

.. code-block:: text

    isBlocked():
        case kWaitForBuild:
            if receivedCachedHashTable():   // future_ fulfilled, buildComplete == true
                setRunning()
                noMoreInput()               // → finishHashBuild() → getHashTableFromCache()

Skipping Source Reads
---------------------

Waiter tasks never read any data from storage. No splits are fetched, no
exchanges are initiated.

In Velox, a build-side pipeline is a chain of operators ending with
``HashBuild`` as the sink:

.. code-block:: text

    [TableScan / Exchange] → ... → [HashBuild]
         operators_[0]               operators_[last]

The ``Driver::runInternal()`` loop iterates through operator pairs ``(op,
nextOp)`` and, for each pair, follows this sequence:

1. Check ``op->isBlocked()`` --- if blocked, suspend the Driver.
2. Check ``nextOp->isBlocked()`` --- if blocked, suspend the Driver.
3. Check ``nextOp->needsInput()`` --- if false, skip pulling from ``op``.
4. Call ``op->getOutput()`` and feed the result to ``nextOp->addInput()``.

The critical point is that ``nextOp->isBlocked()`` is checked **before**
``op->getOutput()`` is ever called. When the ``HashBuild`` operator is in the
``kWaitForBuild`` state, it returns a blocked status, which prevents the
driver from pulling data from any upstream operator (e.g., ``TableScan`` or
``Exchange``). Once the cached table arrives and the waiter calls
``noMoreInput()``, source operators are short-circuited immediately ---
they never execute at all.

This is a key benefit of the caching design: waiter tasks incur zero I/O cost.

Memory Management
-----------------

Cached hash tables must outlive the task that built them because waiter tasks
from the same query need to access the table after the builder task has
finished. To support this, cached hash tables use a dedicated leaf memory pool
created under the **query** memory pool rather than the operator's task-level
pool.

Pool Hierarchy
^^^^^^^^^^^^^^

.. code-block:: text

    Query Pool
    ├── Task 1 Pool (builder - may finish first)
    │   └── Operator Pool
    └── cached_table_<key> Pool  ← hash table lives here
        (created by HashTableCache)

The ``tablePool`` is created by the first call to ``get()`` as a leaf child of
the caller's ``QueryCtx`` root pool. All drivers in the builder task share this
pool for their partial table allocations via ``HashBuild::tableMemoryPool()``.
This ties the cached table's memory accounting to the originating query rather
than to any individual task, allowing the table to survive task completion.

Cleanup Callback
^^^^^^^^^^^^^^^^

When a cache entry is created, ``HashTableCache::get()`` registers a release
callback on the ``QueryCtx``. When the query context is destroyed, this
callback calls ``HashTableCache::drop()`` to remove the entry and free the
table's memory before the query pool is torn down. ``drop()`` resets the
``shared_ptr<BaseHashTable>`` outside the lock to free memory before the entry
itself is destroyed. This ensures there are no dangling references to
destroyed memory pools.

``HashBuild::tableMemoryPool()`` returns the cache entry's ``tablePool`` when
caching is enabled, or the operator's own ``pool()`` for regular joins.

Ownership and Shared Pointers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Without caching, the hash table is transferred to the ``HashJoinBridge`` as a
``unique_ptr``. With caching enabled, the table is stored as a ``shared_ptr``
in the cache entry and a copy of the ``shared_ptr`` is passed to the bridge.
This allows the cache to retain ownership while the bridge and probe operator
also hold references. The ``HashJoinBridge::setHashTable()`` signature was
changed to accept ``shared_ptr`` to support this.

Reference counting ensures that the table is not freed while any probe operator
is actively scanning it. Once the probe finishes, the reference count is
decremented. The table is ultimately freed when the ``QueryCtx`` release
callback calls ``drop()``.

Spilling
--------

Spilling is not supported when hash table caching is enabled. Both
``HashBuild::canSpill()`` and ``HashBuild::canReclaim()`` return false when
``useHashTableCache`` is true:

.. code-block:: c++

    bool HashBuild::canSpill() const {
        // ...
        if (useHashTableCache()) {
            return false;
        }
        // ...
    }

This is because spilling clears the hash table from memory and rebuilds it
later, which would corrupt the cached table that other tasks may be using.
Specifically:

- **Builder task**: Cannot spill because the table is shared via the cache.
  Spilling would invalidate the ``shared_ptr`` held by waiter tasks.
- **Waiter tasks**: Cannot spill because they use the cached table directly
  and never build their own.
- **Coordination complexity**: Rebuild-after-spill would require re-coordinating
  across all tasks sharing the cached table.

Broadcast joins (the primary use case for this cache) are generally expected to
fit in memory. If a build-side relation is large enough to require spilling, it
should bypass the cache and use a standard partitioned hash join with spilling
enabled.

Eviction
--------

Cache eviction is not currently supported. Entries remain in the cache until
the query context is destroyed, at which point the release callback removes
them.

Future memory pressure-based eviction would need to address:

1. **Tracking total memory**: Summing the memory held by all cached tables.
2. **Eviction policy**: Deciding which entries to evict (e.g., LRU, by size).
3. **Reference invalidation**: Safely handling eviction while probe operators
   hold references via ``shared_ptr``.
4. **Rebuild fallback**: Allowing tasks to re-build the table if it was evicted.

The ``drop()`` method already provides the mechanism for removing individual
entries and could be extended to support eviction driven by the memory manager
or arbitration framework.

Limitations and Future Work
---------------------------

- **No spilling**: Cached tables must reside entirely in memory. See
  :ref:`Spilling <spilling>` above.
- **No eviction**: Cached entries live for the full query lifetime. Memory
  pressure-based eviction is planned.
- **Single-query scope**: The cache key includes the ``queryId``, so tables are
  not shared across different queries even if the build side data is identical.
  Cross-query sharing is a potential future optimization.
- **No sanity checks on table sharing during probe**: For right joins, we rely on
  the planner to not do a broadcast join and skip using cached tables. But velox
  as a library does not do checks during probe that it is in fact running a join
  that does not mutate the hash table. Mutating the cached hash table can cause
  incorrect execution results. We should add this check
