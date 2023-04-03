********************
March 2023 Update
********************

Documentation
=============

 * Add :doc:`Chapter 1  of the Programming Guide <../programming-guuide>`.
 * Add blog post on `continuous integration and packaging.<https://velox-lib.io/blog/velox-build-experience>`_


Core Library
============

 * Remove usage of CppToType.
 * Add `printVector` helper function.


Presto Functions
================

 * Add :func:`from_utf8`, :func:`hmac_md5` functions.
 * Add :func:`any_match`, :func:`all_match` and :func:`none_match` functions.
 * Add support for DECIMAL types in :func:`min` and :func:`max` aggregate functions.


Spark Functions
===============

 * Add :spark:func:`sha1` function.


Hive Connector
==============

 * Add support for single-level subfield pruning. :pr:`3949`
 * Add support for BOOLEAN and DATE type in native Parquet reader.
 * Add options to open and prepare file splits asynchronously.
 * Fix reading of VARBINARY columns in Parquet reader.


Substrait
=========

 * Update Substrait to 0.23.0.
 * Add support for `emit <https://substrait.io/tutorial/sql_to_substrait/#column-selection-and-emit>`_
 * Add support for DATE type.


Arrow
=====

 * Add :ref:`ArrowStream operator`.
 * Add support for DATE type.


Performance and Correctness
===========================

 * Add support for custom types in VectorSaver.
 *


Build System
============

 * Move re2, gtest, gmock, xsimd to use the new dependency resolution system.
 * Refactored dependency resolution system into separate modules.
 * Add support for Mac M1 machines during continuous integration.


Credits
=======

Aditi Pandit, Barys Skarabahaty, Benjamin Kietzman, Chandrashekhar Kumar Singh, Chen Zhang, Christian Zentgraf, Daniel Munoz, David Tolnay, David Vu, Deepak Majeti, Denis Yaroshevskiy, Ge Gao, Huameng Jiang, Ivan Sadikov, Jacob Wujciak-Jens, Jake Jung, Jeff Palm, Jialiang Tan, Jialing Zhou, Jimmy Lu, Jonathan Kron, Karteek Murthy Samba Murthy, Krishna Pai, Laith Sakka, Masha Basmanova, Matthew William Edwards, Naveen Kumar Mahadevuni, Oguz Ulgen, Open Source Bot, Orri Erling, Patrick Sullivan, Pedro Eugenio Rocha Pedreira, Pramod, Sergey Pershin, Shengxuan Liu, Siva Muthusamy, Srikrishna Gopu, Wei He, Xiaoxuan Meng, Zac, Zhaolong Zhu, cambyzju, dependabot[bot], lingbin, macduan, wuxiaolong26, xiaoxmeng, yangchuan, yingsu00, zhejiangxiaomai, 张政豪
