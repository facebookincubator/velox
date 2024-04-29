********************
February 2024 Update
********************

Core Library
============

* Add support for aggregations over distinct inputs to StreamingAggregation.
* Add stats for null skew in join operator.
* Add background profiler that starts Linux perf on the Velox process.
* Fix ``out of range in dynamic array`` error in Task::toJson.
* Delete unused ``max_arbitrary_buffer_size`` config.

Presto Functions
================

* Add :func:`typeof`, :func:`from_iso8601_date` scalar functions.
* Add support for deserializing a single column in Presto page format.
* Add support for deserializing an all-null column serialized as UNKNOWN type in Presto page format.
* Add support for DECIMAL input type to :func:`set_agg` and :func:`set_union`.
* Add support for UNKNOWN input type to :func:`checksum`.
* Add support for :func:`date` +/- INTERVAL YEAR MONTH.
* Improve :func:`parse_datetime` to allow ``UCT|UCT|GMT|GMT0`` as ``Z``
* Convert TIMESTAMP_WITH_TIME_ZONE type to a primitive type.

Spark Functions
===============

* Add :spark:func:`array_repeat`, :spark:func:`date_from_unix_date`, :spark:func:`weekday`, :spark:func:`minute`, :spark:func:`second` scalar functions.
* Add :spark:func:`ntile` window function.

Hive Connector
==============

* Add ``ignore_missing_files`` config.
* Add write support in ABFSFileSystem.
* Add support for proxy in S3FileSystem.

Arrow
=====

* Add support to export UNKNOWN type to Arrow array.
* Add support to convert Arrow REE arrays to Velox Vectors.

Performance and Correctness
===========================

* Add FieldReference benchmark.
* Add fuzzer test for Window functions.
* Fix ``Too many open files`` error in JoinFuzzer.

Build System
============

* Add ``VELOX_BUILD_MINIMAL_WITH_DWIO`` CMake option.
* Move documentation, header and format check to Github Action.

Credits
=======

Aaron Feldman, Ankita Victor, Bikramjeet Vig, Christian Zentgraf, Daniel Munoz,
David McKnight, Deepak Majeti, Ge Gao, Hongze Zhang, Jacob Wujciak-Jens, Jia Ke,
Jialiang Tan, Jimmy Lu, Kevin Wilfong, Krishna Pai, Lu Niu, Masha Basmanova,
Nick Terrell, Orri Erling, PHILO-HE, Pedro Pedreira, Pramod, Pranjal Shankhdhar,
Richard Barnes, Schierbeck, Cody, Sergey Pershin, Wei He, Yedidya Feldblum,
Zac Wen, Zhenyuan Zhao, aditi-pandit, duanmeng, gayangya, hengjiang.ly, hitarth,
lingbin, mwish, rrando901, rui-mo, xiaodou, xiaoxmeng, xumingming, yingsu00,
zhli1142015, 高阳阳
