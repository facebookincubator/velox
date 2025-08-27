******************
January 2025 Update
******************

This update was generated with the assistance of AI. While we strive for accuracy, please note
that AI–generated content may not always be error-free. We encourage you to verify any information
that is important to you.

Core Library
============

* Add initial index lookup join operator implementation. :pr:`12218`
* Add index lookup join plan node and connector interface support. :pr:`12163`, :pr:`12187`
* Add test index storage connector for index lookup join unit tests. :pr:`12215`
* Add Global Config in place of gflags for expressions and core components. :pr:`12127`, :pr:`11745`
* Add IoStatistics in ReadFile to collect storage statistics. :pr:`12160`
* Add BufferedOutputStream for improved I/O operations. :pr:`12052`
* Add support for UUID comparison functions. :pr:`10791`
* Add support for dynamic filters for Right Join operations. :pr:`12057`
* Add support for UNKNOWN type in Spark collect_set aggregate function. :pr:`12013`
* Fix null bit set in left index lookup join. :pr:`12226`
* Fix build failure and change lookup result iterator return type. :pr:`12224`
* Fix memory arbitration fuzzer failure and improve stability. :pr:`12005`
* Fix operator addInput stats collection for better monitoring. :pr:`11960`
* Fix json_parse to clear state after encountering errors. :pr:`12150`
* Fix unsafe deserialization in RemoteFunctionServiceMain initialization. :pr:`12095`

Presto Functions
================

* Add :func:`parse_duration` function for time duration parsing. :pr:`12500`
* Add :func:`MAP` signature for MapFunction with improved functionality. :pr:`12115`
* Add support for $internal$_json_string_to_array/map/cast functions. :pr:`12159`
* Add support for UNKNOWN value in :func:`map_entries` function. :pr:`12622`
* Add support for boolean type in :func:`approx_distinct` function. :pr:`12518`
* Fix :func:`regexp_extract_all` to not return match in mismatched group. :pr:`12143`
* Fix :func:`regexp_extract` to return match in mismatched group behavior. :pr:`12109`
* Fix :func:`split_part` to match Presto behavior for empty string delimiter. :pr:`12583`
* Fix :func:`set_agg` to not throw on nested nulls. :pr:`12093`
* Fix :func:`approx_set` to use murmur3 hash function to match Presto. :pr:`12374`

Spark Functions
===============

* Add :spark:func:`date_format` function registration. :pr:`11953`
* Add :spark:func:`array_join` function registration. :pr:`11948`
* Add :spark:func:`sign` function registration. :pr:`12464`
* Add :spark:func:`array_union` function registration. :pr:`12449`
* Add :spark:func:`concat` array function. :pr:`12454`
* Add :spark:func:`get_struct_field` function. :pr:`12166`
* Add support for decimal type in :spark:func:`in`, :spark:func:`floor`, :spark:func:`ceil` functions. :pr:`11947`, :pr:`11951`, :pr:`12056`
* Fix :spark:func:`make_date` to return null for invalid inputs. :pr:`11950`
* Fix :spark:func:`json_object_keys` to return NULL for invalid JSON. :pr:`12679`

Connectors
==========

* Add file format setting for LocalRunnerTestBase. :pr:`12783`
* Add write filesink registration for ABFS connector. :pr:`11973`
* Add support for S3 region configuration. :pr:`12063`
* Add support for reading local files asynchronously. :pr:`11869`
* Add support for o_direct flag in read file operations. :pr:`12138`
* Add support for timestamp and date types in Spark unix_timestamp function. :pr:`11128`
* Fix Parquet SkippedStrides runtime stats reporting. :pr:`12777`
* Fix partition filters with timestamp value handling. :pr:`12368`
* Fix text writer abort test stability. :pr:`12004`

Performance and Correctness
===========================

* Add trace file operation tool for debugging and analysis. :pr:`12021`
* Add build metrics collection and reporting. :pr:`12142`
* Add OrderBy benchmark for performance testing. :pr:`10041`
* Add custom input generator for VectorFuzzer. :pr:`11466`
* Add support for multiple joins in join node toSql methods for reference query runners. :pr:`11801`
* Add support for custom result verifier with compare() API in window fuzzer. :pr:`12148`
* Avoid small batches in Exchange for better throughput. :pr:`12010`
* Enable trace tool test and improve debugging capabilities. :pr:`12139`
* Optimize json_parse performance for better JSON processing. :pr:`11924`
* Upgrade Presto version for fuzzer comparisons to 0.290. :pr:`12096`

Credits
=======

Andrii Rosa, Ankita Victor, Bryan Cutler, Chengcheng Jin, Christian Zentgraf, Daniel Hunte, Darren Fu, Deepak Majeti, Emily (Xuetong) Sun, Guilherme Kunigami, Harsha Rastogi, HolyLow, Hongze Zhang, Jacob Khaliqi, Jacob Wujciak-Jens, Jenson, Jia Ke, Jialiang Tan, Jiaqi Zhang, Jimmy Lu, Ke, Ke Wang, Kevin Wilfong, Kk Pulla, Krishna Pai, Leonid Chistov, Minhan Cao, Natasha Sehgal, Orri Erling, PHILO-HE, Pedro Eugenio Rocha Pedreira, Peter Enescu, Pramod Satya, Rong Ma, Ryan Johnson, Sergey Pershin, Wei He, Xiao Du, Xiaoxuan Meng, Yedidya Feldblum, Yenda Li, Yuan Zhou, Zac Wen, aditi-pandit, dependabot[bot], duanmeng, generatedunixname89002005232357, generatedunixname89002005307016, rui-mo, wangguangxin.cn, xiaodou, zhli1142015
