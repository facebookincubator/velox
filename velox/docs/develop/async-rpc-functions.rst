=================================
How to add an async RPC function?
=================================

This guide describes the async RPC extension point in Velox: the mechanism used
to call an external service once per input row (or once per batch of rows) and
turn the responses back into a Velox vector. It is used for remote inference
workloads such as LLM completion and text embeddings.

For a runnable example, see **velox/exec/rpc/tests/DemoRPCFunction.h** and
**velox/exec/rpc/tests/DemoRPCFunctionRegistration.cpp**.

Overview
--------

Async RPC execution is made of three layers, each in its own directory:

* **Transport** (``velox/common/rpc/``) — the low-level client that sends a
  request and returns a future (``IRPCClient``, ``RPCRequest``, ``RPCResponse``,
  ``RPCStreamingMode``). Transport-specific implementations (e.g. a particular
  inference backend) live here.
* **Function** (``velox/expression/rpc/``) — the business-logic extension point,
  :ref:`AsyncRPCFunction <AsyncRPCFunction>`. It defines *what* an RPC function
  is: how to build requests from input rows and how to interpret responses into
  a result column. This is the interface you implement to add a new RPC
  function, analogous to ``VectorFunction`` for scalar expressions.
* **Execution** (``velox/exec/rpc/``) — the :ref:`RPCOperator <RPCNode>` that
  drives async dispatch: rate limiting, timeouts, row-id assignment, congestion
  control, and passing through the non-RPC columns. You do not implement this;
  it is shared by all RPC functions.

At plan time an :ref:`RPCNode <RPCNode>` is created for the call; at execution
time ``RPCPlanNodeTranslator`` turns it into an ``RPCOperator``, which looks up
your registered ``AsyncRPCFunction`` by name and drives it.

The table below summarizes where each concern lives. Note in particular that
**per-row vs. batch dispatch is a transport-layer concern** (it reflects what the
backend client supports), that **flow control and timeouts live in the execution
layer**, and that **retries live in the transport** — these are deliberately kept
separate.

.. list-table:: Where each concern lives
   :widths: 22 22 56
   :align: left
   :header-rows: 1

   * - Layer
     - Directory
     - Owns
   * - Transport
     - ``velox/common/rpc/``
     - Wire request/response; retries and backoff; PER_ROW vs. BATCH capability
       (native batch API vs. client-side per-row fan-out); backend batch-size /
       token limits.
   * - Function
     - ``velox/expression/rpc/``
     - Building requests from input rows; interpreting responses into the result
       vector; result type; tier key; classifying a response as overloaded
       (the congestion policy).
   * - Execution (operator)
     - ``velox/exec/rpc/``
     - Adaptive concurrency / flow control (AIMD — additive-increase /
       multiplicative-decrease); per-unit timeout; row-id assignment;
       passthrough of the non-RPC columns.

.. _AsyncRPCFunction:

The AsyncRPCFunction interface
------------------------------

An async RPC function subclasses
``facebook::velox::exec::rpc::AsyncRPCFunction``
(``velox/expression/rpc/AsyncRPCFunction.h``) and implements:

.. list-table::
   :widths: 20 40
   :align: left
   :header-rows: 1

   * - Method
     - Description
   * - ``name()``
     - The registered function name.
   * - ``resultType()``
     - The Velox type of the result column produced by the call.
   * - ``initialize(queryConfig, inputTypes, constantInputs)``
     - Called once during operator init, before any dispatch. Create/cache the
       transport client, read session properties from ``queryConfig``, and
       inspect constant arguments. ``constantInputs`` is aligned with
       ``inputTypes``: non-constant arguments are ``nullptr``; constant
       arguments (e.g. model name, options JSON) are single-element
       ``ConstantVector`` objects.
   * - ``tierKey()``
     - Service tier key used for rate limiting. Empty string means "use the
       global default limit".
   * - ``dispatchPerRow(rows, args)``
     - PER_ROW mode. Dispatch one RPC per active row and return one future per
       row, keyed by the original row index. Null-input rows should return an
       immediate ``RPCResponse`` with ``error="null_input"``.
   * - ``accumulateBatch(rows, args)`` / ``flushBatch()``
     - BATCH mode. ``accumulateBatch`` unpacks and stores rows across
       ``addInput()`` calls (returning the original row indices for all rows,
       null and non-null); ``flushBatch`` dispatches the accumulated rows as one
       request. Only required if the function supports BATCH.
   * - ``buildOutput(...)``
     - Convert the collected ``RPCResponse`` objects into the result column vector
       (of type ``resultType()``). See the base class header and the demo for
       the exact signature.

.. note::
   ``buildOutput`` must produce a vector of exactly ``resultType()``. It does
   **not** coerce to any type the caller may have declared elsewhere; if the
   caller needs a different type (e.g. ``ARRAY(DOUBLE)`` from an
   ``ARRAY(REAL)`` result), that conversion belongs in a projection above the
   RPC node, not in the function.

Registration
------------

Register the function with the ``VELOX_REGISTER_RPC_FUNCTION`` macro
(``velox/expression/rpc/AsyncRPCFunctionRegistry.h``):

.. code-block:: c++

  #include "velox/exec/rpc/tests/DemoRPCFunction.h"
  #include "velox/expression/rpc/AsyncRPCFunctionRegistry.h"

  using namespace facebook::velox::exec::rpc;

  VELOX_REGISTER_RPC_FUNCTION(demo_rpc, DemoAsyncRPCFunction);

The first argument is the SQL-visible function name; the second is the
``AsyncRPCFunction`` subclass. The macro registers a factory that the
``RPCOperator`` uses to instantiate the function.

You must also register the plan-node translator once at startup:

.. code-block:: c++

  facebook::velox::exec::rpc::registerRPCPlanNodeTranslator();

Streaming modes and dispatch batch size
---------------------------------------

An :ref:`RPCNode <RPCNode>` carries two knobs that control how input rows are
turned into requests:

* ``streamingMode`` — ``kPerRow`` (one RPC per row, dispatched concurrently) or
  ``kBatch`` (rows grouped into a single request).
* ``dispatchBatchSize`` — in ``kBatch`` mode, the number of rows per request
  (``0`` means "send the whole input as one request"). It is not read in
  ``kPerRow`` mode.

**Per-row vs. batch dispatch is fundamentally a transport-layer concept.** The
``streamingMode`` enum on the plan node only selects which transport path to take;
whether ``kBatch`` maps to a *single native request* or to a client-side fan-out
of per-row calls is a **transport capability**, not a plan-level choice.
``IRPCClient::callBatch`` (``velox/common/rpc/IRPCClient.h``) defaults to fanning
out to individual ``call()`` invocations, and a function only supports BATCH if it
overrides ``accumulateBatch``/``flushBatch``. So the same ``kBatch`` plan runs as
a native batch on backends that have a batch API and degrades to a per-row
fan-out on backends that do not. Which mode is appropriate therefore depends on
the backend's batch API, its maximum batch size / token budget, and its
per-request overhead.

Note that ``streamingMode`` / ``dispatchBatchSize`` do **not** control
concurrency — how many requests are outstanding is governed adaptively by the
flow control described below, independently of the mode. ``dispatchBatchSize``
only sets the flush granularity within ``kBatch``.

.. note::
   These two knobs are being revisited. Because concurrency is already adaptive
   (see below), the open question is expressing the caller's *latency budget*
   rather than a fixed batch size. Treat ``streamingMode`` / ``dispatchBatchSize``
   as current behavior, not a long-term contract.

Adaptive flow control
---------------------

The ``RPCOperator`` regulates how much work is outstanding to the backend with
two AIMD-style (additive-increase / multiplicative-decrease) adaptive
controllers. Both operate on *in-flight units* — a unit
is one row in ``kPerRow`` mode and one batch in ``kBatch`` mode — so the same
control applies **regardless of streaming mode or** ``dispatchBatchSize``:

* **Per-driver window** — ``CongestionController`` (held by ``RPCState``): a
  latency-gradient window over units in flight for a single driver. It shrinks
  multiplicatively on an overload verdict (``effective / 2``) and grows
  additively on healthy latency
  (``effective * gradient + stepCoef * sqrt(effective)``, where ``gradient`` is
  derived from the ratio of baseline to observed RTT). Tunables:
  ``rpc.congestion.*`` (see :doc:`/configs`).
* **Process-global, per-tier limiter** — ``RPCRateLimiter`` (keyed by
  ``tierKey()``): a classic AIMD limit shared across all drivers hitting the same
  backend tier — multiplicative decrease on rate-limiting, additive increase on
  success. Tunables: ``rpc_rate_limiter_*``.

Both controllers back off on the same signal: the function's congestion policy
classifies each response batch and reports overload on rate-limiting (HTTP 429),
timeout, or a majority of errors (ignoring null-input rows).

Putting it together, for each unit the operator:

#. **Admits** it only when *both* the per-driver window and the per-tier limiter
   have headroom; otherwise it waits.
#. **Dispatches** it under a per-unit timeout.
#. **On completion**, feeds the observed round-trip time to the per-driver window
   and the outcome (healthy vs. overloaded) to both controllers: an overload
   verdict shrinks them multiplicatively, sustained healthy latency grows them.

Because a unit is a row in ``kPerRow`` and a batch in ``kBatch``, the adaptation
math is identical in both modes — flow control is independent of the transport's
per-row/batch choice. The streaming mode only seeds the per-driver window's
initial and maximum size.

Retries and error handling
--------------------------

Retries are a **transport** concern, not the operator's. The transport client
(behind ``IRPCClient``) applies its own bounded, backoff retry to each request;
the ``RPCOperator`` only wraps a per-unit timeout and runs the flow control
above. A timeout or error that reaches the operator is therefore one that already
survived the transport's retries, which is why it is treated as an overload
signal. This keeps request-level reliability (retries, in the transport) separate
from cluster-level flow control (concurrency, in the operator).

See also
--------

* :ref:`RPCNode <RPCNode>` in :doc:`/develop/operators` — the plan node and its
  properties.
* ``velox/exec/rpc/RPCOperator.h`` — operator lifecycle and threading model.
* ``velox/exec/rpc/CongestionController.h`` and
  ``velox/exec/rpc/RPCRateLimiter.h`` — the adaptive flow-control controllers.
* ``velox/common/rpc/IRPCClient.h`` — the transport interface (batch capability
  and retries).
* ``velox/exec/rpc/tests/DemoRPCFunction.h`` — a minimal worked example.
