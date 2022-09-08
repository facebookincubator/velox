******************
August 2022 Update
******************

Documentation
=============

* Add documentation for :func:`combinations` Presto function.
* Add documentation for :doc:`/functions/window`.
* Add documentation for :func:`array_sort`.
* Add `website`_ for Velox.

.. _website: https://velox-lib.io

Core Library
============

* Add support for Right Semi Join.
* Add support for a simple in-memory Window operator.
* Add support for `month` string in Joda library.
* Add support for DECIMAL arithmetic addition, subtraction functions.
* Add support for spilling `order by` to disk.
* Add support for zero-copy vector view.
* Add support for casting DECIMAL type to DECIMAL type.
* Add support for DECIMAL type sum aggregation.
* Improve spilling to disk by avoiding redundant computations, better test coverage.
* Resolve to vector functions over simple functions when their signatures match.
* Pre-load lazy vectors that are referenced by multiple sub-expressions.

Core Extensions
===============
* Add support for ROW type, ARRAY type, and MAP type in Substrait to Velox conversion.
* Improve Arrow to Velox conversion for ARRAY and MAP types, and add support for dictionary.

Presto Functions
================

* Add :func:`transform_keys` and :func:`transform_values` functions.
* Add :func:`array_sum` function.
* Add :func:`max_data_size_for_stats` aggregate function that is used for analyzing statistics.
* Add :func:`is_json_scalar`, :func:`json_array_length`, :func:`json_array_contains` functions.
* Add support for TIMESTAMPWITHTIMEZONE in :func:`date_trunc` function.
* Update :func:`min`, :func:`max` aggregate functions to use the same type for input, intermediate, and final results.
* Update :func:`sum` aggregate function to check for integer overflow.
* Add simd support for :func:`eq`, :func:`neq`, :func:`lt`, :func:`gt`, :func:`lte`, :func:`gte` functions.

Hive Connector
==============

* Add INTEGER dictionary, FLOAT type, DOUBLE type, STRING type support to native Parquet reader.
* Add GZIP, snappy compression support to native Parquet reader.
* Add support for DATE type in ORC reader.

Performance and Correctness
===========================

* Add q9, q15, q16 to TPC-H benchmark.
* Optimize memory allocation by specializing vector readers for constant and flat primitives based on the arguments.
* Add benchmark for vector slice.
* Publish microbenchmark results to `conbench`_.

.. _conbench: https://velox-conbench.voltrondata.run/

Debugging Experience
====================

* Add `BaseVector::toString(bool)` API to print all layers of encodings.

Credits
=======

Aditi Pandit, Barson, Behnam Robatmili, Bikramjeet Vig, Chad Austin, Connor Devlin,
Daniel Munoz, Deepak Majeti, Ge Gao, Huameng Jiang, James Wyles, Jialiang Tan,
Jimmy Lu, Jonathan Keane, Karteek Murthy Samba Murthy, Katie Mancini, Kimberly Yang,
Kk Pulla, Krishna Pai, Laith Sakka, Masha Basmanova, Michael Shang, Orri Erling,
Orvid King, Parvez Shaikh, Paul Saab, Pedro Eugenio Rocha Pedreira, Pramod,
Pyre Bot Jr, Raúl Cumplido, Serge Druzkin, Sergey Pershin, Shiyu Gan,
Shrikrishna (Shri) Khare, Taras Boiko, Victor Zverovich, Wei He, Wei Zheng,
Xiaoxuan Meng, Yuan Chao Chou, Zhenyuan Zhao, erdembilegt.j, jiyu.cy, leoluan2009,
muniao, tanjialiang, usurai, yingsu00, 学东栾.