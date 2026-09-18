=================================
How to add an async RPC function?
=================================

This guide describes the async RPC extension point in Velox: the mechanism used
to call an external service once per input row (or once per batch of rows) and
turn the responses back into a Velox vector. It is used for remote inference
workloads such as LLM completion and text embeddings.

For a runnable example, see ``velox/exec/rpc/tests/DemoRPCFunction.h`` and
``velox/exec/rpc/tests/DemoRPCFunctionRegistration.cpp``.

Overview
--------

Async RPC execution is made of three layers, each in its own directory:

* **Wire types** (``velox/common/rpc/``) — the small vocabulary the framework and
  the function share about a call's outcome: ``RPCResponse``,
  ``RpcPayload``, ``RPCErrorKind``, and ``RPCStreamingMode``. Velox does
  not define a client or a request type; both belong to the function.
* **Function** (``velox/expression/rpc/``) — the business-logic extension point,
  :ref:`AsyncRPCFunction <AsyncRPCFunction>`. It defines *what* an RPC function
  is: how to reach a backend, how to turn input rows into requests, and how to
  interpret responses into a result column. This is the interface you implement
  to add a new RPC function, analogous to ``VectorFunction`` for scalar
  expressions.
* **Execution** (``velox/exec/rpc/``) — the :ref:`RPCOperator <RPCNode>` that
  drives async dispatch: admission control, timeouts, row-id assignment,
  congestion control, and passing through the non-RPC columns. You do not
  implement this; it is shared by all RPC functions.

Directory placement and C++ namespaces are independent: the function-layer
interfaces under ``velox/expression/rpc/`` are declared in the
``facebook::velox::exec::rpc`` namespace.

At plan time an :ref:`RPCNode <RPCNode>` is created for the call; at execution
time ``RPCPlanNodeTranslator`` turns it into an ``RPCOperator``, which looks up
your registered ``AsyncRPCFunction`` by name and drives it.

The dividing line is deliberate: **a decision belongs to the layer that holds the
knowledge it needs.** The operator knows how many rows are in flight and how the
backend is responding, so it owns concurrency and timeouts. Only the function
knows which backend it resolved and what that backend's API can do, so it owns
the request format, the dispatch path, and any request-size limits. The operator
never asks which backend is behind a function.

.. list-table:: Where each concern lives
   :widths: 22 22 56
   :align: left
   :header-rows: 1

   * - Layer
     - Directory
     - Owns
   * - Wire types
     - ``velox/common/rpc/``
     - The response envelope: a row id and exactly one function-owned payload
       or ``RpcError`` outcome; the execution mode the coordinator resolved.
   * - Function
     - ``velox/expression/rpc/``
     - The client and the request format; which dispatch path serves the
       requested mode on the backend it resolved; per-request size bounds;
       retries and backoff; interpreting responses into the result vector;
       result type; admission key; classifying a response as overloaded.
   * - Execution (operator)
     - ``velox/exec/rpc/``
     - Adaptive concurrency / flow control (AIMD — additive-increase /
       multiplicative-decrease); per-backend admission; per-unit timeout; row-id
       assignment; passthrough of the non-RPC columns.

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
   * - ``initialize(queryConfig, inputTypes, constantInputs, instruction)``
     - Called once during operator init, before any dispatch. Create/cache the
       client, read session properties from ``queryConfig``, and inspect
       constant arguments. ``constantInputs`` is aligned with ``inputTypes``:
       non-constant arguments are ``nullptr``; constant arguments (e.g. model
       name, options JSON) are single-element ``ConstantVector`` objects.
       ``instruction`` is what the query asked for — see
       :ref:`Execution mode and dispatch path <RPCDispatchPath>`.
   * - ``admissionKey()``
     - Key used to select the shared admission bucket. Empty string means "use
       the global default limit". Override this method when requests share a
       backend-specific capacity pool.
   * - ``dispatchPerRow(rows, args)``
     - Per-row dispatch. Send one RPC per active row and return one future per
       row, keyed by the original row index. Null-input rows should return an
       immediate ``RPCResponse`` carrying ``RPCErrorKind::kNullInput``.
   * - ``accumulateBatch(rows, args)`` / ``flushBatch(maxRows)`` /
       ``pendingBatchSize()``
     - Batch dispatch. ``accumulateBatch`` unpacks and stores rows across
       ``addInput()`` calls (returning the original row indices for all rows,
       null and non-null); ``flushBatch`` dispatches at most ``maxRows`` of the
       accumulated rows, oldest first, and stamps each response's ``rowId``
       with its index within the flushed batch; ``pendingBatchSize`` reports how
       many rows are waiting. Required only if the function batches.
   * - ``maxRowsPerFlush(desired)``
     - Optional. Return fewer than ``desired`` when the backend caps a single
       request — by serialized bytes, tokens, or a protocol maximum. Which of
       those it is stays inside the function; the framework only ever counts
       rows. Must return at least 1 so the drain always progresses; a row that
       alone exceeds the cap is failed in ``flushBatch()``. The default accepts
       whatever is asked for.
   * - ``buildOutput(responses, pool)``
     - Convert the collected ``RPCResponse`` objects into the result column
       vector (of type ``resultType()``).
   * - ``evaluateCongestion(responses)``
     - Classify a completed unit — see
       :ref:`Adaptive flow control <RPCFlowControl>`.

.. note::
   ``buildOutput`` must produce a vector of exactly ``resultType()``. It does
   **not** coerce to any type the caller may have declared elsewhere; if the
   caller needs a different type (e.g. ``ARRAY(DOUBLE)`` from an
   ``ARRAY(REAL)`` result), that conversion belongs in a projection above the
   RPC node, not in the function.

Responses and payloads
----------------------

``RPCResponse`` (``velox/common/rpc/RPCTypes.h``) carries the row id and exactly
one outcome: a function-owned ``RpcPayload`` on success, or an ``RpcError``
containing an ``RPCErrorKind`` and diagnostic message on failure. A
default-constructed response contains the defensive ``kUnset`` error, so
"neither payload nor error" is not representable.

``RpcPayload`` is a move-only, type-erased value stored inline in the response.
The framework moves it from the function's dispatch to the function's
``buildOutput`` and never inspects it. Payload types must fit the 32-byte inline
storage and be nothrow move-constructible; larger values should be represented
by a handle.

Each function defines its own payload type and casts back on the way out:

.. code-block:: c++

  struct TextPayload {
    std::string text;
  };

  // In buildOutput():
  result->set(i, StringView(responseAs<TextPayload>(responses[i]).text));

``responseAs<T>()`` always checks that a payload is present and has exactly the
type requested.

Keeping the payload out of the framework's vocabulary is what lets a function
hand back the representation it already has: an embedding function carries its
vector of floats directly rather than rendering it to text for a field nothing
in the framework reads. Payloads are plain C++ objects rather than Velox vectors
because they are built on transport threads, while Velox allocation must happen
on the driver thread — ``buildOutput`` is where the crossing happens.

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

.. _RPCDispatchPath:

Execution mode and dispatch path
--------------------------------

Three things are easy to confuse, so they have separate names:

* **Objective** — what the query wants, stated by the caller: lowest latency,
  highest throughput, lowest cost, or "decide for me". Objectives are resolved
  during planning and never reach a worker. The policy may inspect table
  statistics; the OSS default resolves ``AUTOMATIC`` to ``PER_ROW``.
* **Execution mode** (``RPCStreamingMode``, on the :ref:`RPCNode <RPCNode>` as
  ``streamingMode``) — the mechanism the coordinator's policy resolved the
  objective to: ``kPerRow`` or ``kBatch``. This is what crosses to the worker,
  and it is what ``initialize()`` receives as ``instruction``.
* **Dispatch path** (``RpcDispatchPath``) — the function-internal description
  of what it will actually do about that instruction on the backend it resolved:

  * ``kPerRow`` — one call per row.
  * ``kNativeBatch`` — one call carrying many rows, answered on the same call.
  * ``kAsyncJob`` — a distinct submit / poll / fetch protocol. Not simply a
    faster batch: its round trip measures queue and execution time rather than
    backend load, which is why it is named separately.

The function makes the last step because it is the only party holding both
facts — the mode the query asked for and what its backend can do. A backend with
no multi-row API serves ``kBatch`` per-row; a backend whose multi-row API is an
offline job serves it as ``kAsyncJob``. The operator never learns which, and
does not branch on the path; a function expresses the consequences of its
choice through ``maxRowsPerFlush()`` and ``evaluateCongestion()`` instead.

``dispatchBatchSize`` on the node sets the flush granularity in ``kBatch`` mode.
``0`` means wait until input closes, then drain everything pending;
``maxRowsPerFlush()`` may split that backlog into smaller requests. It is not
read in ``kPerRow`` mode. If the function also reports a byte budget, each flush
is capped by whichever of the two is smaller, so one request never exceeds the
backend's size limit.

Note that neither ``streamingMode`` nor ``dispatchBatchSize`` controls
concurrency — how many requests are outstanding is governed adaptively by the
flow control described below, independently of the mode.

.. note::
   The user-facing spellings are deliberately unchanged: the SQL option is still
   ``streaming_mode`` and the session property is still ``rpc_streaming_mode``.
   Renaming them would break existing queries.

.. _RPCFlowControl:

Adaptive flow control
---------------------

The ``RPCOperator`` regulates how much work is outstanding to the backend with
two AIMD-style (additive-increase / multiplicative-decrease) adaptive
controllers. Both operate on *in-flight units* — a unit is one row on a per-row
path and one batch on a batch path — so the same control applies regardless of
dispatch path or ``dispatchBatchSize``:

* **Per-driver window** — ``CongestionController`` (held by ``RPCState``): a
  latency-gradient window over units in flight for a single driver. It shrinks
  multiplicatively on an overload verdict (``effective / 2``) and grows
  additively on healthy latency
  (``effective * gradient + stepCoef * sqrt(effective)``, where ``gradient`` is
  derived from the ratio of baseline to observed RTT). Tunables:
  ``rpc.congestion.*`` (see :doc:`/configs`).
* **Per-backend admission** — ``RPCRateLimiter``
  (``velox/exec/rpc/RPCRateLimiter.h``), one instance per admission key,
  obtained from the process-scoped ``RPCRateLimiterRegistry`` and keyed by
  ``admissionKey()``:
  an AIMD limit shared across every driver hitting that backend —
  multiplicative decrease on overload, additive increase on success. Two
  backends in one worker adapt on their own schedules. Tunables:
  ``rpc_rate_limiter_*``.

  Despite the name it bounds *concurrency*, not a rate: capacity is a semaphore
  over in-flight units. The name matches the session properties and runtime
  stats, which cannot be renamed without breaking queries and dashboards.

Both controllers back off on the same signal. ``evaluateCongestion()`` returns
one of five verdicts, and only one of them backs off:

.. list-table::
   :widths: 22 58
   :align: left
   :header-rows: 1

   * - Signal
     - Meaning
   * - ``kSuccess``
     - The unit completed cleanly; feed its latency to the gradient window and
       let the limiter recover.
   * - ``kSuccessNoLatency``
     - The unit completed cleanly; let the limiter recover without feeding its
       latency to the gradient window.
   * - ``kOverloaded``
     - The backend shed load — rate limited, or timed out under pressure.
       **The only signal that shrinks either controller.**
   * - ``kError``
     - The unit failed for something the backend is not responsible for, such as
       a malformed request or a bad key. Retrying more slowly does not make an
       invalid request valid, so neither controller reacts.
   * - ``kNone``
     - Nothing to evaluate.

The distinction matters: a majority of rows failing on bad input is not a reason
to throttle against a healthy backend. For the same reason, a function on a
``kAsyncJob`` path reports ``kSuccessNoLatency`` for a successful batch, but
preserves typed ``kOverloaded`` and ``kError`` signals. Its successful round
trip includes queue and execution time, so feeding that latency to a latency
window would read a slow queue as an overloaded backend. The success still
recovers shared admission capacity.

Putting it together, for each unit the operator:

#. **Admits** it only when *both* the per-driver window and the per-backend
   limiter have headroom; otherwise it waits.
#. **Dispatches** it under a per-unit timeout.
#. **On completion**, feeds the observed round-trip time to the per-driver window
   and the verdict to both controllers.

Retries and error handling
--------------------------

Retries belong to the **function's client**, not the operator. The client applies
its own bounded, backoff retry to each request; the ``RPCOperator`` only wraps a
per-unit timeout and runs the flow control above. A timeout or error that reaches
the operator is therefore one that already survived those retries, which is why
it is treated as a congestion signal rather than a transient blip. This keeps
request-level reliability separate from cluster-level flow control.

Backend and transport failures become per-row errored responses rather than
failing the query, so the configured per-row error policy applies downstream.

A framework failure is different: a wrong response count, a duplicate or
out-of-range batch position, or a response carrying ``kUnset`` or
``kInternalError`` violates the framework contract and fails the query. The
operator cannot safely recover a row-level result from these failures, so the
configured error policy does not apply.

See also
--------

* ``velox/docs/designs/async-rpc.md`` — the design: layering, the threading and
  allocation rule, the flush-boundary row-id protocol, and what the congestion
  signals mean.
* :ref:`RPCNode <RPCNode>` in :doc:`/develop/operators` — the plan node and its
  properties.
* ``velox/exec/rpc/RPCOperator.h`` — operator lifecycle and threading model.
* ``velox/exec/rpc/CongestionController.h`` and
  ``velox/exec/rpc/RPCRateLimiter.h`` — the adaptive flow-control controllers.
* ``velox/common/rpc/RPCTypes.h`` — the response envelope and inline,
  type-erased payload container.
* ``velox/expression/rpc/AsyncRPCFunction.h`` — the extension point itself.
* ``velox/exec/rpc/tests/DemoRPCFunction.h`` — a minimal worked example.
