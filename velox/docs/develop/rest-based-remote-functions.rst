============================================
Using REST-based Remote Functions in Velox
============================================

Overview
--------
This document provides an overview of how to register and use REST-based remote functions
in Velox. REST-based remote functions are external functions that Velox can call via an HTTP
endpoint, where function execution is offloaded to a remote service. 

Registration
------------
Before you can call the remote function in a query, you need to register it with Velox.
Here is an example of how to create the necessary metadata, build the function signature,
and register the remote function, below is the example of registering an absolute function :

.. code-block:: c++

    auto absSignature = {exec::FunctionSignatureBuilder()
                             .returnType("integer")
                             .argumentType("integer")
                             .build()};

    RemoteVectorFunctionMetadata metadata;
    metadata.serdeFormat = remote::PageFormat::PRESTO_PAGE;
    metadata.location = restServerUrl + '/' + "remote_abs";

    registerRemoteFunction("remote_abs", signatures, metadata);

.. note::

   - ``metadata.serdeFormat`` must be set to the PRESTO_PAGE format i.e, ``PageFormat::PRESTO_PAGE``.
   - ``metadata.location`` is the REST endpoint to which Velox will send the function invocation
     requests.

Query Execution
---------------
Once the remote function is registered, it can be used in velox expressions.
During query execution:

1. Velox packs the function input arguments into a request payload using the
   PrestoVectorSerde format.
2. This request is sent to the REST endpoint specified in ``metadata.location``.
3. The remote service executes the function and returns a response payload, also serialized
   in the PrestoVectorSerde format.
4. Velox then deserializes the result payload and proceeds with further query processing.

Serialization Details
---------------------
The request and response payloads are transferred using a ``folly::IOBuf`` under the hood.
Because the format is ``PageFormat::PRESTO_PAGE``, the serialization and deserialization
are done by the PrestoVectorSerde implementation. This means that the remote function server
must be able to understand the Presto page format and return the results in the same format.

Summary
-------
Using REST-based remote functions in Velox involves two major steps:

1. **Registration**: Provide the remote server's endpoint and other metadata via
   ``registerRemoteFunction()`` so that Velox knows how to connect and what format to use.
2. **Execution**: During query execution, Velox serializes function inputs, sends them
   to the remote server, and deserializes the results.
