******************
August 2022 Update
******************

Documentation
=============

* Add `website`_ for Velox.
* Add documentation for :doc:`/functions/window`.
* Add documentation for :func:`combinations` and :func:`array_sort` Presto functions.

.. _website: https://velox-lib.io

Core Library
============

* Add support for Right Semi Join.
* Add basic :ref:`Window operator<window-node>`.
* Add support for parsing `month` string (prefix "Jan" or the full name "January") in Joda library.
* Add support for spilling `order by`.
* Add support for zero-copy `vector view`_.
* Add support for DECIMAL addition and subtraction functions.
* Add support for casting one DECIMAL to another DECIMAL with different precision and scale.
* Add support for DECIMAL sum aggregation.
* Improve spilling by avoiding redundant computations, better test coverage.
* Resolve to vector functions over simple functions when their signatures match.

.. _vector view: https://github.com/facebookincubator/velox/discussions/2212

Substrait Extensions
====================

* Add support for ROW, ARRAY, and MAP types in Substrait to Velox conversion.


Arrow Extensions
================

* Improve Arrow to Velox conversion for ARRAY and MAP types, and add support for dictionary.

Presto Functions
================

* Add :func:`transform_keys` and :func:`transform_values` functions.
* Add :func:`array_sum` function.
* Add :func:`max_data_size_for_stats` aggregate function that is used for computing statistics.
* Add :func:`is_json_scalar`, :func:`json_array_length`, :func:`json_array_contains` functions.
* Add support for TIMESTAMP WITH TIME ZONE in :func:`date_trunc` function.
* Update :func:`min`, :func:`max` aggregate functions to use the same type for input, intermediate, and final results.
* Update :func:`sum` aggregate function to check for integer overflow.
* Add simd support for :func:`eq`, :func:`neq`, :func:`lt`, :func:`gt`, :func:`lte`, :func:`gte` functions.

Hive Connector
==============

* Add support for INTEGER dictionary, FLOAT, DOUBLE, STRING types to native Parquet reader.
* Add GZIP, Snappy compression support to native Parquet reader.
* Add support for DATE type in ORC reader.

Performance and Correctness
===========================

* Add q9, q15, q16 to TPC-H benchmark.
* Optimize memory allocation by `specializing vector readers`_ for constant and flat primitives based on the arguments.
* Add benchmark for vector view.
* Publish microbenchmark results to `conbench`_.

.. _conbench: https://velox-conbench.voltrondata.run/
.. _specializing vector readers: https://github.com/facebookincubator/velox/pull/1956

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