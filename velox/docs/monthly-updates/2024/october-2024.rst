*******************
October 2024 Update
*******************

This update was generated with the assistance of AI. While we strive for accuracy, please note
that AI–generated content may not always be error-free. We encourage you to verify any information
that is important to you.

Core Library
============

* Add global memory arbitration optimization framework. :pr:`11262`
* Add shutdown support for memory arbitrator. :pr:`11325`
* Add support for expression transformers in expression fuzzer. :pr:`11264`
* Add support for IPPREFIX type with comparison and cast operators. :pr:`11122`
* Add support for stripe with row count greater than int32. :pr:`11314`
* Add support for runtime compilation features. :pr:`11225`
* Add max_local_exchange_partition_count config property. :pr:`11292`
* Add support for opaque type in vector fuzzer. :pr:`11189`
* Fix global arbitration check failure when disabled. :pr:`11364`
* Fix memory reclaim and arbitration issues in HashJoinBridge. :pr:`11324`
* Fix security vulnerabilities in codebase. :pr:`11260`
* Fix flaky arbitration tests and improve stability. :pr:`11316`, :pr:`11346`
* Fix sort and spill memory tracking causing OOM. :pr:`11129`
* Fix spill related issues in TopNRowNumber operator. :pr:`11310`

Presto Functions
================

* Add :func:`fb_regex_match_any` function for pattern matching. :pr:`11273`
* Add :func:`replace_first` function for string replacement. :pr:`11288`
* Add :func:`trail` function. :pr:`11265`, :pr:`11327`
* Add support for single parameter array_intersect function. :pr:`11305`
* Add comparison functions for complex types. :pr:`11241`
* Fix :func:`to_iso8601` to use Z for UTC timezone. :pr:`11279`
* Fix :func:`format_datetime` and :func:`parse_datetime` with time zone support. :pr:`11283`, :pr:`11312`, :pr:`11323`, :pr:`11330`, :pr:`11331`, :pr:`11337`
* Fix :func:`date_add` and :func:`date_diff` UDFs with TimestampAndTimeZone and DST. :pr:`11353`, :pr:`11380`
* Fix :func:`power` function Inf and NaN handling. :pr:`11295`, :pr:`11210`

Spark Functions
===============

* Add :spark:func:`cast` support for integral types to timestamp. :pr:`11089`
* Fix overflow behavior for Spark decimal sum aggregate function. :pr:`11127`

Connectors
==========

* Add DELTA_BYTE_ARRAY encoding support in native Parquet reader. :pr:`10589`
* Add Row ID column reading support. :pr:`11363`
* Add support for jvm version libhdfs. :pr:`9835`
* Fix nullable bug of Arrow MapVector in Bridge.cpp. :pr:`11214`
* Fix local file sink to use velox fs and fail on existing file. :pr:`11322`
* Fix memory arbitration in writer operations. :pr:`11319`

Performance and Correctness
===========================

* Add operator tracing and make it work E2E. :pr:`11360`
* Add monthly update template for consistent documentation. :pr:`11195`
* Reserve memory for SortWindowBuild sort process. :pr:`11370`
* Support unary and binary arithmetic operators in expression fuzzer against Presto. :pr:`11313`
* Support PrefixSort in Window operator for improved performance. :pr:`10417`
* Support yield in bucket sort table write to prevent stuck driver detection. :pr:`11229`
* Avoid local shuffle and global shuffle using same hash value to prevent data skew. :pr:`11338`
* Use nested namespace definition for code organization. :pr:`11147`

Credits
=======

Amit Dutta, Andrii Rosa, Bikramjeet Vig, Brady Kilpatrick, Chengcheng Jin, Christian Zentgraf, Daniel Hunte, Deepak Majeti, Duc Nguyen, Emmanuel Ferdman, Eric Liu, Guilherme Kunigami, Heidi Han, JackyWoo, Jacob Wujciak-Jens, Jenson, Jia Ke, Jialiang Tan, Jimmy Lu, Joe Giardino, Ke, Ke Wang, Kevin Wilfong, MacVincent Agha-Oko, Mingyu Zhang, NEUpanning, Orri Erling, PHILO-HE, Pedro Eugenio Rocha Pedreira, Pramod, Reetika Agrawal, Richard Barnes, Satadru Pan, Sergey Pershin, Tao Yang, Wei He, Xiaoxuan Meng, Yenda Li, Zuyu ZHANG, aditi-pandit, duanmeng, joey.ljy, lingbin, mohsaka, mwish, rexan, wypb, yingsu00, zhli1142015, zml1206
