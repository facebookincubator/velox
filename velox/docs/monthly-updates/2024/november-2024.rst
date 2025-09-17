********************
November 2024 Update
********************

This update was generated with the assistance of AI. While we strive for accuracy, please note
that AI–generated content may not always be error-free. We encourage you to verify any information
that is important to you.

Core Library
============

* Add query trace TaskRunner. :pr:`11927`
* Add spill directory callback API in task. :pr:`11572`
* Add target tracing driver IDs. :pr:`11560`
* Add skewed partition balancer. :pr:`11635`
* Add auto table scan scaling. :pr:`11879`
* Add fine-grained per-split cache control. :pr:`11978`, :pr:`11984`
* Add priority-based memory reclaim framework. :pr:`11598`
* Add auto scale writer. :pr:`11702`
* Add arbitration lock timeout to shared arbitrator. :pr:`11376`
* Fix driver block hanging in serialized execution mode. :pr:`11647`, :pr:`11681`
* Fix memory reclaim bytes accounting in hash join. :pr:`11624`, :pr:`11642`
* Fix integer overflow while skipping on stream. :pr:`11477`
* Fix task hanging in serialized execution mode. :pr:`11747`

Presto Functions
================

* Add :func:`ip_prefix` function. :pr:`11514`
* Add :func:`map_key_exists` function. :pr:`11735`
* Add :func:`normal_cdf`, :func:`inverse_normal_cdf`, :func:`beta_cdf`, :func:`inverse_beta_cdf` functions. :pr:`11792`, :pr:`11864`
* Add canonicalization of JSON data. :pr:`11284`
* Add illegal unicode parsing in :func:`json_parse`. :pr:`11744`
* Fix :func:`date_add` nonexistent time in time zones. :pr:`11845`
* Fix :func:`from_unixtime` TimestampWithTimeZone precision. :pr:`11426`
* Fix Presto URL functions to match Java behavior. :pr:`11488`, :pr:`11535`, :pr:`11540`, :pr:`11604`
* Fix :func:`array_distinct` under-allocation with overlapping arrays. :pr:`11817`
* Fix :func:`array_intersect` null handling in dictionary encoded arrays. :pr:`11807`

Spark Functions
===============

* Add :spark:func:`get_json_object` function. :pr:`11691`
* Add :spark:func:`locate` function. :pr:`8863`
* Add :spark:func:`concat_ws` function. :pr:`8854`
* Add :spark:func:`cast` with decimal type for unary minus. :pr:`11454`
* Add shuffle compression. :pr:`11914`
* Fix :spark:func:`get_json_object` JSON tape failure. :pr:`11831`

Connectors
==========

* Add per-split fine-grained cache control in Hive connector. :pr:`11978`
* Add viewfs file system. :pr:`11811`
* Add ABFS with SAS and OAuth authentication. :pr:`11623`
* Add timestamp type partition filtering. :pr:`11754`
* Add S3 bucket configuration. :pr:`11321`
* Add fallocate for file size extension. :pr:`11403`, :pr:`11541`
* Fix Parquet schema evolution issues. :pr:`11595`
* Fix DWRF footer IO read optimization in Hive connector. :pr:`11798`
* Fix importing long decimal vector from Arrow. :pr:`11404`

Performance and Correctness
===========================

* Add T-Digest data structure. :pr:`11665`
* Add table schema evolution fuzzer. :pr:`11872`
* Add utilities for combining dictionary wrappers. :pr:`11944`
* Add prefix sort with string type key. :pr:`11527`
* Add fast row size estimation for hash probe. :pr:`11558`
* Add indexed priority queue for auto writer thread scaling. :pr:`11584`
* Add constrained input generators in fuzzers. :pr:`11368`
* Add testing peeling in expression fuzzer. :pr:`11379`
* Add fault injection in cache fuzzer. :pr:`11969`
* Optimize DWRF footer IO read in Hive connector. :pr:`11798`
* Optimize IndexedPriorityQueue::addOrUpdate for 20x performance improvement. :pr:`11955`
* Enable RowContainer column stats by default. :pr:`11731`

Credits
=======

Abdullah Ozturk, Amit Dutta, Bikramjeet Vig, Bryan Cutler, Chengcheng Jin, Christian Zentgraf, David Reveman, Deepak Majeti, Eric Liu, Guilherme Kunigami, Heidi Han, Huameng (Michael) Jiang, Jacob Wujciak-Jens, Jaime Pan, Jia Ke, Jialiang Tan, Jiaqi Zhang, Jimmy Lu, Joe Abraham, Joe Giardino, Ke, Kevin Wilfong, Krishna Pai, Marcus D. Hanwell, Max Ma, Mike Lui, Minhan Cao, NEUpanning, Orri Erling, Pedro Eugenio Rocha Pedreira, Pedro Pedreira, Pramod, Richard Barnes, Rong Ma, Satadru Pan, Sergey Pershin, Wei He, Wenbin Lin, Xiaoxuan Meng, Xuedong Luan, Yang Zhang, Yenda Li, Zac Wen, Zhaokuo, Zuyu ZHANG, aditi-pandit, dependabot[bot], duanmeng, hengjiang.ly, mohsaka, rui-mo, yingsu00, zhli1142015, zuyu
