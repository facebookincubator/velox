********************
December 2021 Update
********************

Documentation
-------------

* Add :doc:`Aggregation-related optimizations <../develop/aggregations>` guide.
* Add :doc:`json_extract_scalar <../functions/json>` function.
* Update :doc:`Mathematical Functions <../functions/math>`.
* Update :doc:`Comparison Functions <../functions/comparison>`.
* Update :doc:`How to add an aggregate function? <../develop/aggregate-functions>` guide with new registry.
* Update :doc:`How to add a scalar function? <../develop/scalar-functions>` guide with complex types.


Core Library
------------

* Fix a bug in HashJoin filter to enable TPC-H query 19.
* Add support for partition key aliases to Hive Connector.
* Add support for conversions between Velox string and Arrow string.
* Add support for 'timestamp with timezone'.
* Add support for masks to StreamingAggregation operator.
* Add support for SSD-file data cache.
* Other bug-fixes and code improvements.

Presto Functions
----------------

* Add :func:`corr`, :func:`covar_samp`, :func:`covar_pop`, :func:`every` aggregate functions.
* Add :func:`sort_array`, :func:`array_sort`, :func:`array_position` array functions.

Credits
-------

Aditi Pandit, Alex Hornby, Amit Dutta, Andres Suarez, Andrew Gallagher,
Chao Chen, Cheng Su, Deepak Majeti, Huameng Jiang, Jack Qiao, Kevin Wilfong,
Krishna Pai, Laith Sakka, Marc Fisher, Masha Basmanova, Michael Shang,
Naresh Kumar, Orri Erling, Pedro Eugenio Rocha Pedreira, Sergey Pershin,
Wei He, Wei Zheng, Xavier Deguillard, Yating Zhou, Yuan Chao Chou, Zhenyuan Zhao 
