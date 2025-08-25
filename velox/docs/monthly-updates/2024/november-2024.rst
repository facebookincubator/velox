********************
November 2024 Update
********************

This update was generated with the assistance of AI. While we strive for accuracy, please note
that AI–generated content may not always be error-free. We encourage you to verify any information
that is important to you.

Core Library
============

* Add query trace TaskRunner for comprehensive query analysis. :pr:`11927`
* Add support for API for setting spill directory callback in task. :pr:`11572`
* Add support for specifying target tracing driver IDs. :pr:`11560`
* Add skewed partition balancer for improved data distribution. :pr:`11635`
* Add auto table scan scaling based on memory usage. :pr:`11879`
* Add table evolution fuzzer for testing schema changes. :pr:`11872`
* Add support for fine-grained per-split cache control. :pr:`11978`, :pr:`11984`
* Add priority-based memory reclaim framework. :pr:`11598`
* Add support for auto scale writer functionality. :pr:`11702`
* Add arbitration lock timeout to shared arbitrator. :pr:`11376`
* Fix driver block hanging issue in serialized execution mode. :pr:`11647`, :pr:`11681`
* Fix memory reclaim bytes accounting in hash join operations. :pr:`11624`, :pr:`11642`
* Fix flaky HashJoinTest.reclaimDuringOutputProcessing. :pr:`11993`
* Fix integer overflow while skipping on a stream. :pr:`11477`
* Fix task hanging under serialized execution mode. :pr:`11747`

Presto Functions
================

* Add :func:`ip_prefix` function for IP address prefix operations. :pr:`11514`
* Add :func:`get_json_object` Spark function. :pr:`11691`
* Add :func:`locate` Spark function. :pr:`8863`
* Add :func:`concat_ws` Spark function. :pr:`8854`
* Add :func:`map_key_exists` function for map operations. :pr:`11735`
* Add classification functions for machine learning support. :pr:`11792`, :pr:`11864`
* Add support for canonicalization of JSON data. :pr:`11284`
* Add support for parsing illegal unicode in json_parse function. :pr:`11744`
* Fix :func:`date_add` throws or produces wrong results for nonexistent time in timezone. :pr:`11845`
* Fix :func:`from_unixtime` with TimestampWithTimeZone precision issues. :pr:`11426`
* Fix Presto URL functions to match Java behavior more closely. :pr:`11488`, :pr:`11535`, :pr:`11540`, :pr:`11604`
* Fix :func:`array_distinct` under-allocation with overlapping input arrays. :pr:`11817`
* Fix :func:`array_intersect` with single argument null handling in dictionary encoded arrays. :pr:`11807`

Spark Functions
===============

* Add support for :spark:func:`cast` with decimal type for unary minus function. :pr:`11454`
* Add support for shuffle compression. :pr:`11914`
* Fix :spark:func:`get_json_object` JSON tape failure. :pr:`11831`

Connectors
==========

* Add support for per-split fine-grained cache control in Hive connector. :pr:`11978`
* Add support for viewfs file system in velox. :pr:`11811`
* Add ABFS dependency for Auth support. :pr:`11633`
* Add support for ABFS with SAS and OAuth configuration. :pr:`11623`
* Add support for timestamp type partition filter. :pr:`11754`
* Add support for S3 bucket configuration. :pr:`11321`
* Add support for fallocate for file size extension when supported. :pr:`11403`, :pr:`11541`
* Fix Parquet schema evolution issues. :pr:`11595`
* Fix DWRF footer IO read count and size optimization in Hive connector. :pr:`11798`
* Fix importing long decimal vector from Arrow. :pr:`11404`

Performance and Correctness
===========================

* Add T-Digest data structure for statistical computations. :pr:`11665`
* Add utilities for combining dictionary wrappers. :pr:`11944`
* Add support for prefix sort with string type key. :pr:`11527`
* Add fast row size estimation for hash probe operations. :pr:`11558`
* Add indexed priority queue for auto writer thread scaling. :pr:`11584`
* Add support for constrained input generators in fuzzers. :pr:`11368`
* Add support for testing peeling in expression fuzzer. :pr:`11379`
* Add fault injection in cache fuzzer for robustness testing. :pr:`11969`
* Optimize DWRF footer IO read count and size in Hive connector. :pr:`11798`
* Optimize IndexedPriorityQueue::addOrUpdate by 20x performance improvement. :pr:`11955`
* Enable RowContainer column stats by default for performance. :pr:`11731`

Credits
=======

Abdullah Ozturk, Amit Dutta, Bikramjeet Vig, Bryan Cutler, Chengcheng Jin, Christian Zentgraf, David Reveman, Deepak Majeti, Eric Liu, Guilherme Kunigami, Heidi Han, Huameng (Michael) Jiang, Jacob Wujciak-Jens, Jaime Pan, Jia Ke, Jialiang Tan, Jiaqi Zhang, Jimmy Lu, Joe Abraham, Joe Giardino, Ke, Kevin Wilfong, Krishna Pai, Marcus D. Hanwell, Max Ma, Mike Lui, Minhan Cao, NEUpanning, Orri Erling, Pedro Eugenio Rocha Pedreira, Pedro Pedreira, Pramod, Richard Barnes, Rong Ma, Satadru Pan, Sergey Pershin, Wei He, Wenbin Lin, Xiaoxuan Meng, Xuedong Luan, Yang Zhang, Yenda Li, Zac Wen, Zhaokuo, Zuyu ZHANG, aditi-pandit, dependabot[bot], duanmeng, hengjiang.ly, mohsaka, rui-mo, yingsu00, zhli1142015, zuyu
