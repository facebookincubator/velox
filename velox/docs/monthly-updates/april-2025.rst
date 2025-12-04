*****************
April 2025 Update
*****************

This update was generated with the assistance of AI. While we strive for accuracy, please note
that AI–generated content may not always be error-free. We encourage you to verify any information
that is important to you.

Core Library
============

* Add merge source start processing mechanism for PySpark-Velox integration. :pr:`13139`
* Add insertable iteratable spill partition set. :pr:`13130`
* Add SpillPartitionFunction and SpillPartitionIdLookup. :pr:`13079`, :pr:`13077`
* Add producing filters as expressions. :pr:`13178`
* Add Builder classes for PlanNodes. :pr:`13093`
* Fix child vector resize length in row vector operations. :pr:`13195`
* Fix memory allocation in freeNonContiguous operations. :pr:`13136`
* Fix overflow prevention in NegatedBigintValuesUsingBitmask. :pr:`13179`
* Fix initialization of MemoryManager with proper options. :pr:`13182`
* Fix constant filter results handling in processFilterResults. :pr:`13155`
* Fix right join result mismatch. :pr:`13065`
* Fix merge join issues and source test value naming. :pr:`13104`
* Fix raw_vector allocation bytes overflow. :pr:`13110`
* Enable hashing of opaque variants. :pr:`13069`

Presto Functions
================

* Add :func:`inverse_poisson_cdf`, :func:`inverse_binomial_cdf` functions. :pr:`12984`, :pr:`12983`
* Add Merge() TDigest Type implementation. :pr:`13101`
* Add scale support for TDigest operations. :pr:`12725`
* Add QDigest data type. :pr:`13030`
* Add field_names_in_json_cast_enabled when casting rows to JSON. :pr:`13108`
* Add unknown values in maps and arrays when casting to JSON. :pr:`13154`
* Fix float precision in rescaleFloatingPoint operations. :pr:`12852`
* Add maximum elements configuration for sequence and repeat functions. :pr:`13160`, :pr:`13126`
* Add evaluating array/map match lambda functions over batches of elements. :pr:`13100`

Spark Functions
===============

* Add :spark:func:`array_prepend`. :pr:`12730`
* Add :spark:func:`CAST` for timestamp to integral types. :pr:`11468`
* Fix timestamp precision in :spark:func:`in` function. :pr:`11812`
* Add date type in :spark:func:`from_json` function. :pr:`12848`
* Fix :spark:func:`unix_timestamp` function for date type handling. :pr:`12881`
* Fix Spark :func:`get_json_object` to parse incomplete JSON. :pr:`12417`
* Fix Spark central moments functions to handle legacy behavior. :pr:`12566`

Breeze
======

* Add CUDA 12.8 atomics when available. :pr:`12932`
* Add cuDF based OrderBy operator for GPU processing. :pr:`12735`
* Add Arm Neoverse V2 CPU architecture. :pr:`13006`

Connectors
==========

* Add dictionary and dictionary page size configuration in Parquet. :pr:`12766`
* Fix INT64 timestamp precision conversion in Parquet reader. :pr:`12953`
* Fix dictionary vector handling when loading lazy delta for FULL_REWRITE. :pr:`13094`

Performance and Correctness
===========================

* Add TimestampWithTimeZone in Aggregation, Window, and Join Fuzzers. :pr:`11897`
* Add unified compression API with multiple codec support. :pr:`7589`
* Add optimization for Hash Table atomic bools performance. :pr:`13054`
* Add array_agg and arbitrary benchmarks with Streaming Aggregation. :pr:`12935`
* Add lazily-allocated CPUThreadPoolExecutor. :pr:`13070`
* Optimize UnsafeRow to RowVector conversion. :pr:`12841`
* Fix Velox buffer copy and serialization interfaces to use i64 instead of i32. :pr:`13083`
* Fix semi merge join with duplicate match vectors. :pr:`13096`

Credits
=======

Artem Selishchev, Bikramjeet Vig, Carl Shapiro, Chengcheng Jin, Christian Zentgraf, Daniel Munoz, David Reveman, Deepak Majeti, Devavret Makkar, Emily (Xuetong) Sun, Eric Jia, Facebook Community Bot, Gary Helmling, Haiping Xue, Hung-Ching Lee, Ian Petersen, Jacob Wujciak-Jens, James Gill, Jialiang Tan, Jimmy Lu, Joe Abraham, Joe Giardino, Junjie Wang, Ke Jia, Ke Wang, Kevin Wilfong, Kk Pulla, Krishna Pai, Masha Basmanova, Mingyu Zhang, Minhan Cao, Natasha Sehgal, Nathan Phan, Nikhil Tarte, Orri Erling, Patrick Sullivan, Pedro Eugenio Rocha Pedreira, Peter Enescu, Pradeep Vaka, Pramod Satya, Qian Sun, Richard Barnes, Rong Ma, Rui Mo, Sergey Pershin, Shakyan Kushwaha, Sutou Kouhei, Wei He, Xiao Du, Xiaoxuan Meng, Xin Zhang, Xuedong Luan, Yenda Li, Yuan Zhou, Yun Wu, Zac Wen, aditi-pandit, boyao.zby, dependabot[bot], generatedunixname647790274085263, generatedunixname89002005287564, joey.ljy, kavinli, lingbin, rexan, rui-mo, xwei19, yumwang@ebay.com, zhli1142015, zml1206
