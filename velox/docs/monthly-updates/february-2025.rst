*******************
February 2025 Update
*******************

This update was generated with the assistance of AI. While we strive for accuracy, please note
that AI–generated content may not always be error-free. We encourage you to verify any information
that is important to you.

Core Library
============

* Add index join conditions. :pr:`12461`
* Add JSON SerDe code for LocationHandle::targetFileName_ and HiveTableHandle::tableParameters_. :pr:`12431`, :pr:`12177`
* Add HashStringAllocator::InputStream. :pr:`12364`
* Add Global Config in place of gflags for SSD configuration. :pr:`12181`
* Add custom type with parameter in signature parser. :pr:`12445`
* Add mathematical operators for IntervalYearMonth type. :pr:`11612`
* Add registering components dynamically. :pr:`11439`
* Fix RowVector::copyRanges not preserving values in gap between ranges. :pr:`12472`
* Fix ambiguous sources serde logic for TableWriteNode and TableWriteMergeNode. :pr:`12429`
* Fix DwrfStreamIdentifier compilation on operator== method. :pr:`12356`
* Fix dual linking of gflags when CMake variable BUILD_SHARED_LIBS is cached. :pr:`12359`
* Fix memory arbitration issues. :pr:`12388`

Presto Functions
================

* Add :func:`map_keys_by_top_n_values` function. :pr:`12209`
* Add :func:`ip_subnet_range`, :func:`is_subnet_of` functions. :pr:`11777`
* Add :func:`array_top_n` function. :pr:`12105`
* Add UNKNOWN type in :func:`set_union` function. :pr:`12559`
* Add custom comparison in :func:`map_union_sum` function. :pr:`12273`
* Fix :func:`date_diff` and :func:`from_unixtime` precision and timezone handling. :pr:`12426`, :pr:`12411`, :pr:`12409`
* Fix JSON functions to add backslash to unescaped character list. :pr:`12442`
* Add TDigest data structure with scale, merge, and value_at_quantile functions. :pr:`12725`, :pr:`12326`, :pr:`12529`
* Add QDigest data type. :pr:`13030`
* Add mathematical CDF functions: :func:`inverse_poisson_cdf`, :func:`inverse_binomial_cdf`, :func:`inverse_gamma_cdf`. :pr:`12984`, :pr:`12983`, :pr:`12867`

Spark Functions
===============

* Add :spark:func:`unix_timestamp` for timestamp and date types. :pr:`11128`
* Add :spark:func:`CAST` for double/float to timestamp conversion. :pr:`12041`
* Add :spark:func:`from_json` Spark function. :pr:`11709`
* Add :spark:func:`json_array_length` Spark function. :pr:`11946`
* Add :spark:func:`array_append` Spark function. :pr:`12043`
* Fix NullHandlingMode for Spark min/max aggregate functions. :pr:`12384`
* Fix Spark :spark:func:`regex_extract` on mismatched group behavior. :pr:`12162`
* Fix Spark :spark:func:`date_trunc` function initialization. :pr:`12922`

Connectors
==========

* Add FilesystemStatistics for runtime counters. :pr:`12424`
* Add storageParameters inclusion in Hive split configuration. :pr:`12443`
* Add parquet writer page size and batch size configuration. :pr:`12755`, :pr:`12766`
* Add timestamp type partition filtering. :pr:`12754`
* Add decimal column writer for ORC file format. :pr:`11431`
* Fix Parquet precision conversion for INT64 timestamps. :pr:`12953`
* Fix initialization of memory manager in remote function operations. :pr:`12095`
* Fix TPCH benchmark reader crash. :pr:`12833`
* Add unified compression API with lz4_frame/lz4_raw/lz4_hadoop codec support. :pr:`7589`

Performance and Correctness
===========================

* Add type deduplication during serialization. :pr:`12361`
* Add batched deserialization in RowSerializer. :pr:`13032`
* Optimize UnsafeRow to RowVector conversion performance. :pr:`12841`
* Add TopNRowNumberFuzzer. :pr:`12103`
* Optimize ContainerRowSerde deserialization for string, array, and map types. :pr:`12362`
* Enable aggregation fuzzer test with FB-only functions. :pr:`12288`

Credits
=======

Bikramjeet Vig, Christian Zentgraf, CodemodService Bot, Dark Knight, David Reveman, Deepak Majeti, Emily (Xuetong) Sun, Gaurav Mogre, Heidi Han, Hongze Zhang, James Gill, Jialiang Tan, Jiaqi Zhang, Jim Meyering, Jimmy Lu, Joe Abraham, Ke, Ke Jia, Ke Wang, Kevin Wilfong, Krishna Pai, Mahadevuni Naveen Kumar, Masha Basmanova, Natasha Sehgal, Nicholas Ormrod, Orri Erling, PHILO-HE, Pedro Eugenio Rocha Pedreira, Peter Enescu, Pradeep Vaka, Pramod Satya, Richard Barnes, Rui Mo, Serge Druzkin, Sergey Pershin, Shakyan Kushwaha, Soumya Duriseti, Wei He, Xiao Du, Xiaochong Wei, Xiaoxuan Meng, Xin Zhang, Xuedong Luan, Yedidya Feldblum, Yenda Li, Yiyang Chen, Zac Wen, aditi-pandit, duanmeng, joey.ljy, lifulong, lingbin, mwish, rexan, rui-mo, svm1, wangguangxin.cn, wecharyu, zhli1142015
