=====================================
AsyncDataCache (In Memory Data Cache)
=====================================

Background
----------
Velox provides a transparent data cache (AsyncDataCache) to accelerate table scans operators through hot data reuse and prefetch algorithms. 
This data cache is integrated with the memory system to achieve dynamic memory sharing between the data cache and query memory. 
When a query fails to allocate memory, Velox retries the allocation by shrinking the data cache. 
Therefore, the data cache size is automatically adjusted in response to the query memory usage change. 
See `Memory Management - Velox Documentation <https://facebookincubator.github.io/velox/develop/memory.html>`_  
for more information about Velox's data cache.

Configuration Properties
------------------------
See `Configuration Properties 
<https://github.com/prestodb/presto/blob/master/presto-docs/src/main/sphinx/presto_cpp/properties.rst#cache-properties>`_ 
for AsyncDataCache related configuration properties.

=========
SSD Cache
=========

Background
----------
The in-memory data cache (AsyncDataCache) is configured to use SSD when provided.
The SSD serves as an extension for the AsyncDataCache.
This helps mitigate the number of reads from slower storage.

Configuration Properties
------------------------
See `Configuration Properties 
<https://github.com/prestodb/presto/blob/master/presto-docs/src/main/sphinx/presto_cpp/properties.rst#cache-properties>`_ 
for SSD Cache related configuration properties.

Metrics
-------
There are SSD cache relevant metrics that Velox emits during query execution and runtime. 
See `Debugging Metrics <./debugging/metrics.rst>`_ and `Monitoring Metrics <../monitoring/metrics.rst>`_ for more details.
