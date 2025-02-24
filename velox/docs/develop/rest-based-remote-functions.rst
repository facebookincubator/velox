============================================
Using REST-based Remote Functions in Velox
============================================

Overview
--------
This document provides an overview of how to register and use REST-based remote functions
in Velox. REST-based remote functions are external functions that Velox can call via an HTTP
endpoint, where function execution is offloaded to a remote service.

Using REST-based remote functions in Velox involves two major steps:

1. **Registration**: Provide the remote server's endpoint and other metadata via
   ``registerRemoteFunction()`` so that Velox knows how to connect and what format to use.
2. **Execution**: During query execution, Velox serializes function inputs, sends them
   to the remote server, and deserializes the results in the ``serdeFormat`` provided at the time
   of function registration.

Registration
------------
Before you can call the remote function in a query, you need to register it with Velox.
Here is an example of how to create the necessary metadata, build the function signature,
and register the remote function. Below is an example of registering an absolute function:

.. code-block:: c++

    auto absSignature = exec::FunctionSignatureBuilder()
                             .returnType("integer")
                             .argumentType("integer")
                             .build();

    RemoteVectorFunctionMetadata metadata;
    metadata.serdeFormat = remote::PageFormat::PRESTO_PAGE;
    metadata.location = restServerUrl + '/' + "remote_abs";

    registerRemoteFunction("remote_abs", {absSignature}, metadata);

.. note::

   - ``metadata.location`` is the REST endpoint to which Velox will send the function invocation
     requests (POST requests).

Query Execution
---------------
Once the remote function is registered, it can be used in Velox expressions.
During query execution:

1. Velox packs the function input arguments into a request payload using
   ``metadata.serdeFormat``.
2. This request is sent to the REST endpoint specified in ``metadata.location``.
3. The remote service executes the function and returns a response payload,
   also serialized in ``metadata.serdeFormat``.
4. Velox then deserializes the result payload and proceeds with further
   query processing.

Serialization Details
---------------------
The request and response payloads are transferred using a ``folly::IOBuf`` under the hood.
Serialization and deserialization are purely based on the SerDe (serializer-deserializer)
provided during the function registration process. The server assumes that any data it processes
has been serialized and deserialized according to the SerDe logic, without needing additional
hints or metadata.

.. note::
   - Serialization - Deserialization information is passed as headers in the HTTP request.
