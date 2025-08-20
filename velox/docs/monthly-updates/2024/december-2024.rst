********************
December 2024 Update
********************

This update was generated with the assistance of AI. While we strive for accuracy, please note
that AI–generated content may not always be error-free. We encourage you to verify any information
that is important to you.

Core Library
============

* Add support for spiller abstraction with better memory management. :pr:`11656`
* Add priority-based memory reclaim framework for improved resource management. :pr:`11598`
* Add support for auto scale writer with load balancing. :pr:`11702`
* Add support for key/value operations in radix sort for breeze. :pr:`11733`
* Add support for string type key in prefix sort operations. :pr:`11527`
* Add T-Digest data structure for advanced statistical computations. :pr:`11665`
* Add support for UnsafeRow and CompactRow compression. :pr:`11497`
* Add utilities for combining dictionary wrappers. :pr:`11944`
* Add support for viewfs file system in velox. :pr:`11811`
* Fix flaky HashJoinTest.reclaimDuringOutputProcessing test. :pr:`11993`
* Fix writer reclaim failure by clearing dict chained buffers during arbitration. :pr:`11988`
* Fix BitSet size to be larger than int32 for large datasets. :pr:`11986`
* Fix simd build for different platform compatibility. :pr:`11971`
* Fix memory manager test instance initialization. :pr:`11926`
* Fix task hanging under serialized execution mode. :pr:`11747`
* Fix zero reclaimable bytes handling in priority reclaiming. :pr:`11925`

Presto Functions
================

* Add :func:`ip_prefix` function with cast operators for IP address operations. :pr:`11514`, :pr:`11481`, :pr:`11546`
* Add :func:`map_key_exists` function for map key validation. :pr:`11735`
* Add classification functions for machine learning workflows. :pr:`11792`, :pr:`11864`
* Add support for negative array indices in JSON path operations. :pr:`11451`
* Add support for canonicalization of JSON data processing. :pr:`11284`
* Fix :func:`date_add` behavior for nonexistent time in time zones. :pr:`11845`
* Fix :func:`array_distinct` under-allocation with overlapping input arrays. :pr:`11817`
* Fix :func:`array_intersect` single argument null handling in dictionary encoded arrays. :pr:`11807`
* Fix :func:`IN` expression handling of user errors during evaluation. :pr:`11823`
* Fix :func:`width_bucket` indices in elements null checks. :pr:`11810`
* Fix parsing of fractions of a second in :func:`parse_datetime`. :pr:`11723`
* Fix casting Varchar to Timestamp with unrecognized time zone offsets. :pr:`11849`

Spark Functions
===============

* Add :spark:func:`get_json_object` function for JSON data extraction. :pr:`11691`
* Add :spark:func:`concat_ws` function for string concatenation with separator. :pr:`8854`
* Add support for shuffle compression in Spark operations. :pr:`11914`
* Fix :spark:func:`get_json_object` JSON tape failure handling. :pr:`11831`

Connectors
==========

* Add support for per-split fine-grained cache control. :pr:`11978`, :pr:`11984`
* Add support for ABFS with SAS and OAuth authentication. :pr:`11623`
* Add support for timestamp type partition filtering. :pr:`11754`
* Add support for SST new file format implementation. :pr:`11847`
* Add support for Int64 Timestamp in Parquet reader. :pr:`11530`
* Add support for converted type in Parquet timestamp reader. :pr:`11964`
* Fix Parquet schema evolution compatibility issues. :pr:`11595`
* Fix table writer to allow structs to be written as flat maps. :pr:`11909`
* Fix copying vector preserving encodings for LazyVector. :pr:`11855`
* Fix TPCH benchmark reader crash for improved stability. :pr:`11833`

Performance and Correctness
===========================

* Add fault injection in cache fuzzer for robustness testing. :pr:`11969`
* Add support for constrained input generators for comprehensive fuzzing. :pr:`11368`
* Add ability to run multiple batches in expression fuzzer. :pr:`11903`
* Add support for join filters in join fuzzer testing. :pr:`11473`
* Add table evolution fuzzer for schema change validation. :pr:`11872`
* Add auto table scan scaling based on memory usage patterns. :pr:`11879`
* Optimize IndexedPriorityQueue::addOrUpdate for 20x performance improvement. :pr:`11955`
* Optimize prefix sort to exclude null byte if column has no nulls. :pr:`11583`
* Enable RowContainer column stats by default for better performance monitoring. :pr:`11731`
* Support separate null count and minmax from column stats. :pr:`11860`

Credits
=======

Amit Dutta, Andrii Rosa, Bikramjeet Vig, Chengcheng Jin, Christian Zentgraf, Daniel Bauer, Daniel Hunte, Dark Knight, David Reveman, Deepak Majeti, Emily (Xuetong) Sun, Ge Gao, Guilherme Kunigami, Harsha Rastogi, Hongze Zhang, Huameng (Michael) Jiang, Jacob Wujciak-Jens, Jia Ke, Jialiang Tan, Jiaqi Zhang, Jimmy Lu, Joe Giardino, Karthikeyan, Kevin Wilfong, Kk Pulla, Kostas Xirogiannopoulos, Krishna Pai, Masha Basmanova, Mingyu Zhang, Minhan Cao, Orri Erling, PHILO-HE, Pavel Solodovnikov, Pedro Eugenio Rocha Pedreira, Prasoon Telang, Raymond Wu, Richard Barnes, Sergey Pershin, Wei He, Wenbin Lin, Xiaoxuan Meng, Yang Zhang, Yenda Li, Yizhuo Liang, Zac Wen, Zuyu ZHANG, aditi-pandit, duanmeng, lingbin, mohsaka, rui-mo, wypb, xiaodou, yingsu00, zhli1142015, zuyu
