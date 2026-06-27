================
Remote Functions
================

Overview
--------

Remote functions allow Velox to execute scalar functions in a separate process,
enabling language interoperability, process isolation, and resource separation.
A remote function is a :doc:`VectorFunction </develop/scalar-functions>` that
serializes its input batch, sends it to a remote server via Thrift RPC, and
deserializes the result.

This is useful when:

- UDFs are written in a different language (e.g., Java SparkSQL functions
  called from a C++ Velox client).
- Functions need process isolation for security or resource management.
- Functions depend on libraries that cannot be linked into the Velox process.

Architecture
------------

The remote function framework consists of a client-side proxy (registered as a
regular Velox vector function) and a server-side handler that evaluates the
actual function.

.. code-block:: text

    Client                              Server
    ------                              ------
    registerRemoteFunction()            Functions registered locally
           |                                  |
    RemoteVectorFunction.apply()        RemoteFunctionServiceHandler
           |                                  |
    RowVector -> serialize -> IOBuf     IOBuf -> deserialize -> RowVector
           |                                  |
    Thrift RPC  ---------------------->  ExprSet.eval()
           |                                  |
    IOBuf -> deserialize  <-----------  RowVector -> serialize -> IOBuf

**Serialization formats**: Two formats are supported via the ``PageFormat`` enum:

- ``PRESTO_PAGE`` — Uses ``PrestoVectorSerde``. Supports encoding
  preservation. Recommended for Presto-based systems.
- ``SPARK_UNSAFE_ROW`` — Uses ``UnsafeRowVectorSerde``. Compatible with
  Spark's internal row format.

Only active (selected) rows are serialized, reducing network and server
overhead when the function is called within conditional expressions.

Registering Remote Functions
----------------------------

Use ``registerRemoteFunction()`` to register a remote function as a standard
Velox vector function:

.. code-block:: c++

    #include "velox/functions/remote/client/Remote.h"

    RemoteThriftVectorFunctionMetadata metadata;
    metadata.location = folly::SocketAddress::makeFromPath("/tmp/remote.socket");
    metadata.serdeFormat = remote::PageFormat::PRESTO_PAGE;

    auto signatures = {exec::FunctionSignatureBuilder()
                           .returnType("bigint")
                           .argumentType("bigint")
                           .argumentType("bigint")
                           .build()};

    registerRemoteFunction("my_remote_add", signatures, metadata);

After registration, ``my_remote_add`` can be used in expressions just like any
other Velox function.

Configuration
~~~~~~~~~~~~~

``RemoteThriftVectorFunctionMetadata`` extends ``RemoteVectorFunctionMetadata``
(which extends ``VectorFunctionMetadata``) and provides the following options:

.. list-table::
   :header-rows: 1

   * - Field
     - Default
     - Description
   * - ``location``
     - (none)
     - ``folly::SocketAddress`` of the remote server. Supports IP:port or Unix
       domain sockets (via ``SocketAddress::makeFromPath()``).
   * - ``serdeFormat``
     - ``PRESTO_PAGE``
     - Serialization format for input/output data.
   * - ``preserveEncoding``
     - ``false``
     - Whether to preserve vector encodings (e.g., dictionary) during
       serialization. Only effective with ``PRESTO_PAGE``.
   * - ``deterministic``
     - ``true``
     - Whether the function is deterministic. Set to ``false`` for functions
       like ``random()`` or ``uuid()``.
   * - ``defaultNullBehavior``
     - ``true``
     - When ``true``, null inputs automatically produce null output without
       calling the remote function. Set to ``false`` if the remote function
       needs to handle nulls explicitly.
   * - ``clientFactory``
     - (none)
     - Optional factory for custom ``IRemoteFunctionClient`` implementations.
       Useful for dependency injection in tests.

Server Implementation
---------------------

The server-side handler ``RemoteFunctionServiceHandler`` receives Thrift
requests, deserializes input, evaluates the function using Velox's expression
engine, and returns serialized results.

.. code-block:: c++

    #include "velox/functions/remote/server/RemoteFunctionService.h"

    // Create handler with an optional function name prefix.
    auto handler = std::make_shared<RemoteFunctionServiceHandler>(
        "my_prefix");

    // Set up and start the Thrift server.
    auto server = std::make_shared<ThriftServer>();
    server->setInterface(handler);
    server->setAddress(folly::SocketAddress::makeFromPath("/tmp/remote.socket"));
    server->serve();

**Function prefix**: The server prepends an optional prefix to the function
name before looking it up in the local registry. This allows the same process
to host both the remote proxy and the actual implementation (useful for
testing). For example, with prefix ``"my_prefix"``, a request for function
``"add"`` resolves to ``"my_prefix.add"`` on the server.

Error Handling
--------------

The ``throwOnError`` flag in the request controls error behavior:

- **When true** (default): Exceptions from the remote function propagate
  directly to the caller.
- **When false** (e.g., inside a ``TRY()`` expression): The server captures
  errors per row and serializes them as VARCHAR strings in an error payload.
  The client deserializes these errors and reports them per-row via
  ``EvalCtx::setError()``, allowing ``TRY()`` to convert them to nulls.

Testing with Mock Clients
-------------------------

The ``RemoteFunctionClientFactory`` typedef enables dependency injection for
testing without a real Thrift server:

.. code-block:: c++

    #include "velox/functions/remote/client/ThriftClient.h"

    // Create a mock client.
    class MockClient : public IRemoteFunctionClient {
     public:
      void invokeFunction(
          remote::RemoteFunctionResponse& response,
          const remote::RemoteFunctionRequest& request) override {
        // Build mock response...
      }
    };

    // Register with mock client factory.
    RemoteThriftVectorFunctionMetadata metadata;
    metadata.location = folly::SocketAddress("127.0.0.1", 12345);
    metadata.clientFactory =
        [](const folly::SocketAddress&, folly::EventBase*) {
          return std::make_unique<MockClient>();
        };

    registerRemoteFunction("mock_fn", signatures, metadata);

Build Configuration
-------------------

Remote functions are gated by the CMake option ``VELOX_ENABLE_REMOTE_FUNCTIONS``
(default: ``OFF``). Enable it to build the remote function client, server, and
tests:

.. code-block:: bash

    cmake -DVELOX_ENABLE_REMOTE_FUNCTIONS=ON ...

Custom Transport
----------------

To use a transport other than Thrift, extend the ``RemoteVectorFunction``
abstract class and implement:

- ``invokeRemoteFunction()`` — Sends the serialized request and returns the
  response.
- ``remoteLocationToString()`` — Returns a human-readable description of the
  remote endpoint for error messages.
