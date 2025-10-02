********************
September 2024 Update
********************

This update was generated with the assistance of AI. While we strive for accuracy, please note
that AI–generated content may not always be error-free. We encourage you to verify any information
that is important to you.

Core Library
============

* Add query replayer. :pr:`10897`
* Add trace support for TableWriter operator. :pr:`10910`
* Add arbitration participant and operation objects for global memory arbitration optimization. :pr:`11074`
* Add custom comparison functions in ContainerRowSerde, RowContainer, and VectorHasher. :pr:`11023`, :pr:`11024`, :pr:`11051`
* Add full outer join in SMJ (Sort Merge Join). :pr:`10247`
* Add batch version of RowContainer::store API. :pr:`10812`
* Add spill metric spillExtractVectorTimeNanos. :pr:`11049`
* Fix right join result mismatch. :pr:`11027`
* Fix memory deadlock when failed to create root memory pool. :pr:`11042`
* Fix hash build not able to reclaim in finishHashBuild. :pr:`11016`
* Fix recursive memory arbitration caused by parallel spill. :pr:`10975`
* Fix casting varchar to integral types by trimming unicode and control characters. :pr:`10957`
* Fix TRY_CAST(VARCHAR as TIMESTAMP) incorrectly suppress errors outside supported ranges. :pr:`10928`
* Fix timestamp and interval arithmetic bug. :pr:`11017`

Presto Functions
================

* Add :func:`substr`, :func:`trim`, :func:`ltrim`, :func:`rtrim`, :func:`reverse` functions.
* Add custom comparison in Presto's :func:`IN` function. :pr:`11032`
* Fix :func:`date_diff` bug across time zone boundaries. :pr:`11053`
* Fix :func:`round` function to handle large numbers. :pr:`10922`, :pr:`10937`
* Fix :func:`concat` function to throw when there are less than 2 arguments. :pr:`11076`
* Fix :func:`array_join` for Date type. :pr:`11003`
* Enable unicode escaping in JSON processing. :pr:`10887`

Spark Functions
===============

* Add :spark:func:`at_least_n_non_nulls` function. :pr:`10508`
* Add :spark:func:`array_insert` function. :pr:`9851`
* Fix invalid UTF-8 character in Spark :spark:func:`translate` function. :pr:`10891`
* Handle multi-char delimiters in :spark:func:`split_to_map` function. :pr:`10861`

Connectors
==========

* Fix ParquetReader initialize schema failed for ARRAY/MAP column. :pr:`10681`
* Fix inaccurate statistical data for parquet-251. :pr:`10823`
* Enable reading explicit row number column from Prism connector. :pr:`11072`
* Add S3FileSystem CRC32 checksum on AWS S3. :pr:`10918`
* Add config to use selective Nimble reader. :pr:`10990`
* Extend filesystem APIs. :pr:`10504`
* Enable Parquet LazyVector support. :pr:`11010`

Performance and Correctness
===========================

* Add query configs to turn off expression evaluation optimizations. :pr:`10902`
* Add smallint to PrefixSort. :pr:`10946`
* Allow pushdown of dynamic filters through HashAggregation. :pr:`10988`
* Optimize duplicate row memory allocations. :pr:`10865`
* Enable probe spill with dynamic filter replaced. :pr:`10849`
* Ignore time zones not recognizable by OS when building time zone database. :pr:`10654`
* Extend expression fuzzer test to support decimal. :pr:`9149`

Credits
=======

Bikramjeet Vig, Chengcheng Jin, Christian Zentgraf, Daniel Hunte, David Tolnay, Deepak Majeti, Eric Liu, Ge Gao, Heidi Han, Hongze Zhang, Jia Ke, Jialiang Tan, Jimmy Lu, Karteek, Ke, Kevin Wilfong, Krishna Pai, Mahadevuni Naveen Kumar, Marcus D. Hanwell, NEUpanning, Orri Erling, PHILO-HE, Pedro Eugenio Rocha Pedreira, Pramod, Satadru Pan, Sergey Pershin, Tengfei Huang, Wei He, Xiaoxuan Meng, Zac Wen, duanmeng, hengjiang.ly, joey.ljy, kevincmchen, liangyongyuan, lingbin, mohsaka, rui-mo, wypb, xiaodou, xiaoxmeng, yingsu00, zhli1142015
