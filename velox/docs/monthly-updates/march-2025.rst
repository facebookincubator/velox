****************
March 2025 Update
****************

This update was generated with the assistance of AI. While we strive for accuracy, please note
that AI–generated content may not always be error-free. We encourage you to verify any information
that is important to you.

Core Library
============

* Add index join lookup constant filter condition support for SS connector. :pr:`12839`
* Add BingTile geometric functions including construction, property, and conversion operations. :pr:`12821`, :pr:`12708`, :pr:`12580`, :pr:`12505`, :pr:`12419`
* Add Geometry Presto type support for spatial operations. :pr:`12274`
* Add support for INTERVAL YEAR/MONTH and DAYS parsing. :pr:`12828`, :pr:`12826`
* Add decimal support in histogram aggregate function. :pr:`12811`
* Add memory order support to breeze atomics for improved concurrency. :pr:`12793`
* Add :func:`is_private_ip` function to check if IP address is private. :pr:`12807`
* Add :func:`ip_prefix_subnets` function for IP prefix operations. :pr:`12801`
* Add support for custom AWSCredentialsProvider registration in S3. :pr:`12774`
* Add processedStrides and processedSplits runtime statistics. :pr:`12647`
* Fix incorrect result when casting double to decimal. :pr:`12600`
* Fix JSON casting of NaN values and unicode character handling. :pr:`12825`, :pr:`12806`
* Fix boolean sorting bug in :func:`array_sort_desc` function. :pr:`12770`
* Fix null-deref in complex vector during memory operations. :pr:`12767`
* Fix concurrency issues in TPC-H dbgen. :pr:`12819`

Presto Functions
================

* Add :func:`array_max_by`, :func:`array_min_by` functions for array processing. :pr:`12652`
* Add :func:`map_top_n_values` function for map operations. :pr:`12822`
* Add :func:`split_to_multimap` function for string processing. :pr:`12727`, :pr:`12779`
* Add :func:`combine_hash` internal operator. :pr:`12244`
* Add :func:`murmur3_x64_128` hash function implementation. :pr:`12889`
* Add support for UUID to Varbinary casting operations. :pr:`12544`
* Add support for unknown type in maps and arrays when casting to JSON. :pr:`13154`
* Fix JSON array_get function to always canonicalize output. :pr:`12814`
* Fix JSON extract performance improvements for large JSON documents. :pr:`12796`
* Fix wildcard support in JSON extract using simdjson. :pr:`12281`
* Fix Spark legacy behavior for central moments functions. :pr:`12566`

Spark Functions
===============

* Add :spark:func:`get_json_object` function for JSON data extraction. :pr:`12691`
* Add :spark:func:`locate` function for string operations. :pr:`8863`
* Add :spark:func:`concat_ws` function for string concatenation with separator. :pr:`8854`
* Fix Spark JSON object_keys function to return NULL for invalid JSON. :pr:`12679`
* Fix Spark collect_set to handle NaN values correctly. :pr:`12335`

Connectors
==========

* Add GEOS library as optional dependency for spatial operations. :pr:`12243`
* Add S3 filesystem metrics collection and reporting. :pr:`12213`
* Add support for S3 log location configuration. :pr:`12534`
* Add Parquet reserved keywords handling. :pr:`12625`
* Add boolean RLE decoder for Parquet format. :pr:`11282`
* Fix Iceberg positional delete upper bound check bug. :pr:`12453`
* Fix Arrow bundled dependency build issues on macOS. :pr:`12658`
* Fix handling of escaped separators in URL functions. :pr:`11540`

Performance and Correctness
===========================

* Add custom fuzzer input generator for phone number and canonical inputs. :pr:`12724`, :pr:`12769`
* Add support for constrained input generators in VectorFuzzer. :pr:`11466`
* Add expression evaluation logging in fuzzer for debugging. :pr:`12706`
* Add custom input generator for JSON path operations. :pr:`12312`
* Add TopNRowNumberFuzzer to GitHub workflow runs. :pr:`12662`
* Enable CCache for Manylinux builds to improve build performance. :pr:`12710`
* Optimize deserialize UnsafeRows to RowVector conversion. :pr:`11936`
* Optimize JSON extract implementation to conform to jayway standards. :pr:`12483`
* Support recursive JSON path operator for advanced JSON querying. :pr:`12568`

Credits
=======

Amit Dutta, Bikramjeet Vig, Bradley Dice, Chandrashekhar Kumar Singh, Chengcheng Jin, Christian Zentgraf, David Reveman, Deepak Majeti, Heidi Han, Hongze Zhang, Jacob Khaliqi, Jacob Wujciak-Jens, Jaime Pan, James Gill, Jialiang Tan, Jiaqi Zhang, Jim Meyering, Jimmy Lu, Joe Giardino, Karteek Murthy, Ke, Kevin Stichter, Kevin Wilfong, Kk Pulla, Krishna Pai, Mahadevuni Naveen Kumar, Masha Basmanova, Muhammad Haseeb, NEUpanning, Natasha Sehgal, Orri Erling, PHILO-HE, Patrick Sullivan, Pedro Eugenio Rocha Pedreira, Peter Enescu, Prashant Golash, Qian Sun, Richard Barnes, Rong Ma, Rui Mo, Sebastiano Peluso, Serge Druzkin, Sergey Pershin, Shakyan Kushwaha, Sutou Kouhei, Wei He, Wenqi Wu, Xiao Du, Xiaoxuan Meng, Xin Zhang, Yedidya Feldblum, Yenda Li, Yuan, Yun Wu, Yuxuan Chen, Zac Wen, aditi-pandit, alileclerc, duanmeng, iamorchid, mwish, peterenescu, rwang22, wangguangxin.cn, yingsu00, zhaokuo03, zhli1142015
