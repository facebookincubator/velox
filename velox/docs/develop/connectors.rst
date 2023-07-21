==========
Connectors
==========

Connectors allow to read and write data to and from external sources.
This concept is similar to `Presto Connectors <https://prestodb.io/docs/current/develop/connectors.html>`_.
The :ref:`TableScanNode<TableScanNode>` operator reads external data via a connector.
The :ref:`TableWriteNode<TableWriteNode>` operator writes data externally via a connector.
The various connector interfaces in Velox are described below.

Connector Interface
-------------------

.. list-table::
   :widths: 10 40
   :header-rows: 1

   * - Interface Name
     - Description
   * - ConnectorSplit
     - Basic unit that describes a chunk of data for processing.
   * - DataSource
     - Provides methods to consume and process a split. A DataSource can optionally consume a
       dynamic filter during execution to prune some rows from the output vector.
   * - DataSink
     - Provides methods to write a Velox vector externally.
   * - Connector
     - Allows creating instances of a DataSource or a DataSink.
   * - Connector Factory
     - Enables creating instances of a particular connector.

Velox currently has in-built support for the Hive Connector and the TPC-H Connector.
Let's see how the above connector interfaces are implemented in the Hive Connector in detail below.

Hive Connector
--------------
The Hive Connector can be used to read and write data files (Parquet, DWRF) residing on
an external storage (S3, HDFS, GCS, Linux FS).

HiveConnectorSplit
~~~~~~~~~~~~~~~~~~
The HiveConnectorSplit describes a data chunk using parameters including `file-path`,
`file-format`, `start position`, `length`, `storage format`, etc..
Given a set of Parquet files, users or applications are responsible for defining the splits.

HiveDataSource
~~~~~~~~~~~~~~
The HiveDataSource implements the `addSplit` API that consumes a HiveConnectorSplit.
It creates a file reader based on the file format, offset, and length. The supported file formats
are DWRF and Parquet.
The `next` API processes the split and returns a batch of rows. Users can continue to call
`next` until all the rows in the split are fully read.
HiveDataSource allows adding a dynamic filter using the `addDynamicFilter` API. This allows
supporting :ref:`Dynamic Filter Pushdown<DynamicFilterPushdown>`.

HiveDataSink
~~~~~~~~~~~~
The HiveDataSink writes vectors to files on disk. The supported file formats are DWRF and Parquet.
The parameters to HiveDataSink also include column names, sorting, partitioning, and bucketing information.
The `appendData` API instantiates a file writer based on the above parameters and writes a vector to disk.

HiveConnector
~~~~~~~~~~~~~
The HiveConnector extends the `createDataSource` connector API to create instances of HiveDataSource.
It also extends the `createDataSink` connector API to create instances of HiveDataSink.
One of the parameters to these APIs is `ConnectorQueryCtx`, which provides means to specify a
memory pool and connector configuration.

HiveConnectorFactory
~~~~~~~~~~~~~~~~~~~~
The HiveConnectorFactory enabled creating instances of the HiveConnector. A `connector name` say "hive"
is required to register the HiveConnectorFactory. Multiple instances of the HiveConnector can then be
created by using the `newConnector` API by specifying a `connectorId` and connector config listed
:doc:`here</configs>`.

Storage Adapters
~~~~~~~~~~~~~~~~
The Hive Connector file reading and writing is supporting on a variety of distributed storage APIs.
The supported storage API are S3, HDFS, GCS, Linux FS.

S3 is supported using the `AWS SDK for C++ <https://github.com/aws/aws-sdk-cpp>`_ library.
S3 supported schemes are `s3://` (Amazon S3, Minio), `s3a://` (Hadoop 3.x), `s3n://` (Deprecated in Hadoop 3.x),
`oss://` (Alibaba cloud storage), and `cos://`, `cosn://` (Tencent cloud storage).

HDFS is supported using the
`Apache Hawk libhdfs3 <https://github.com/apache/hawq/tree/master/depends/libhdfs3>`_ library. HDFS supported schemes
are `hdfs://`.

GCS is supported using the
`Google Cloud Platform C++ Client Libraries <https://github.com/googleapis/google-cloud-cpp>`_. GCS supported schemes
are `gs://`.