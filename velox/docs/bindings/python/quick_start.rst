===========================================================
PyVelox Quick Start - The Power of Velox at your Fingertips
===========================================================

PyVelox is a thin Python layer on top of the Velox C++ library. It provides Python language bindings that allow Velox query plans to be easily constructed and executed, without requiring users to build, install, and learn other idiosyncrasies of the C++ library. 

PyVelox is meant to be used by engineers when developing and debugging Velox and other parts of the data stack; it’s not meant to be used by end data users. 

Installation
------------

.. code-block:: python

      // The PyPI page for PyVelox is available at: https://pypi.org/project/PyVelox/

      $ pip install pyvelox


Managing Files
--------------

On a high-level, PyVelox allows users to define and execute local query plans. Query plans often start by reading data from a set of files. Let’s look at a simple example first. 

Velox supports file formats like Dwrf, Nimble, Parquet, ORC, Json, CSV, and others. You can define files to be read and written using pyvelox.file:

.. code-block:: python

      // Register input files to PyVelox

      from velox.py.file import NIMBLE, PARQUET

      parquet_file = PARQUET("/tmp/my_file.parquet")
      nimble_file = NIMBLE("/tmp/my_efficient_file.nimble")

By default, these files are read from the local filesystem. Files can also be read from remote storage systems like S3, ABFS, GCS, and others by registering the filesystem and using the proper path.


Scanning a File
---------------

To read (scan) data in a file or a set of files, you can define a query plan containing a table scan node:

.. code-block:: python

      // Read input file using PyVelox

      from pyvelox.file import DWRF
      from pyvelox.plan_builder import PlanBuilder
      from pyvelox.type import BIGINT, ROW, VARCHAR

      plan_builder = PlanBuilder()
      plan_builder.table_scan(
          output=ROW(
             ["col1", "col2"],
             [VARCHAR(), BIGINT()],
          )
          input_files=[DWRF(file_path)],
      )
      print(plan_builder.get_plan_node())

Velox (and hence PyVelox) are designed to be used by large-scale compute engines, such that table schema is often stored in a separate metadata store. Therefore, when creating the query plan, the user needs to explicitly specify the columns and types to be read from the file.

Alternatively, a file schema can be inspected by using the following API:

.. code-block:: python

      // Read input file schema

      from pyvelox.file import NIMBLE, PARQUET

      dwrf_file = DWRF("ws://ws.dw.ftw0dw0/namespace/...")
      row_type = dwrf_file.get_schema()
      print(row_type)


