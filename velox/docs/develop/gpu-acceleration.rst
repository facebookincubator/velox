================
GPU Acceleration
================

Velox includes several **experimental** components for executing query plans on
GPUs. They live under ``velox/experimental/`` and fall into three groups:

* **Execution backends** that run Velox operators on the GPU — :ref:`Wave
  <gpu-wave>` and :ref:`cuDF <gpu-cudf>`.
* A **portable primitives library** — :ref:`Breeze <gpu-breeze>` — that provides
  the data-parallel building blocks (reduce, scan, sort, ...) used to build GPU
  kernels.
* A **GPU-to-GPU data exchange** — the :ref:`UCX exchange <gpu-ucx>` — that
  shuffles data between workers without staging through host memory.

.. note::

   Everything described here is experimental. Operator, type, and function
   coverage is still evolving, the components require a CUDA toolchain (and, to
   run, a GPU), and APIs may change. Operators that a backend does not support
   fall back to CPU execution.

Both execution backends plug into Velox through the same extension point — the
``DriverAdapter`` interface (see :doc:`task`) — which rewrites a query plan to
replace CPU operators with GPU operators. Because the rewrite happens at the
operator boundary, the algorithm and plan above it are unchanged; only the
backend that executes the operators differs. The two backends take opposite
approaches: Wave fuses a run of operators into generated kernels, while cuDF
swaps operators one-to-one for library calls.

Overview
--------

.. list-table::
   :header-rows: 1
   :widths: 16 30 30 24

   * - Component
     - Role
     - How it integrates
     - Notes
   * - Wave
     - Whole-pipeline execution backend
     - JIT-compiles a contiguous run of operators into CUDA kernels
     - Highest fusion potential; early-stage coverage
   * - cuDF
     - Operator-level execution backend
     - One-to-one operator replacement backed by NVIDIA libcudf
     - Broad relational coverage; single-node, single-GPU
   * - Breeze
     - Portable primitives library
     - Building blocks (reduce/scan/sort) used by GPU kernels
     - Not an operator backend; multi-platform
   * - UCX exchange
     - GPU-to-GPU shuffle transport
     - Replaces the inter-worker exchange; selected per plan node
     - Engine-agnostic; complements either backend

.. _gpu-wave:

Wave
----

`Wave <https://github.com/facebookincubator/velox/tree/main/velox/experimental/wave>`_
(``velox/experimental/wave``) is a whole-pipeline GPU backend. A ``DriverAdapter``
inspects a Driver's operators, lowers a contiguous run of supported operators
into an intermediate representation, generates CUDA C++ from it, compiles that at
query time with NVRTC, and replaces the original operators with a single source
operator that launches the generated kernels and streams results back to the
host. Generated modules are cached and compiled in the background; operators
that are not (yet) supported terminate the offloaded run and execute on CPU.

Key characteristics:

* **Kernel fusion.** A run such as scan → filter → project → aggregate is fused
  into a small number of generated kernels, minimizing materialization between
  operators.
* **On-GPU decode.** Wave includes a GPU columnar-decode path, so a table scan
  can decode encoded columns (including dictionary encoding) on the device
  rather than decoding on the CPU and copying decoded values across PCIe.
* **Built on Breeze.** Wave's kernels build on the platform/atomics/primitive
  layer provided by :ref:`Breeze <gpu-breeze>`.

Wave is built when ``VELOX_ENABLE_WAVE`` is set. Several targets link the CUDA
driver; they can be built without a GPU using the CUDA "stub" packages, but the
resulting binaries only run on a machine with a real CUDA driver.

Wave is at an early stage: the set of supported operators, scalar functions,
aggregates, and column types is still narrow, and distributed exchange is not
yet offloaded to the GPU. It is the direction for full-pipeline GPU offload, not
a drop-in replacement for the CPU engine today.

.. _gpu-cudf:

cuDF
----

The `cuDF backend
<https://github.com/facebookincubator/velox/tree/main/velox/experimental/cudf>`_
(``velox/experimental/cudf``) executes Velox plans using NVIDIA's
`RAPIDS libcudf <https://github.com/rapidsai/cudf>`_, the CUDA C++ core of cuDF.
libcudf uses Arrow-compatible data layouts and provides single-node, single-GPU
algorithms for data processing.

``CudfDriverAdapter`` rewrites a plan to run on the GPU, **generally replacing
operators one-to-one** with libcudf-backed equivalents and inserting conversions
at GPU/CPU boundaries (with CPU fallback where an operator is unsupported). For
end-to-end GPU execution, cuDF relies on Velox's pipeline-based execution model
(see :doc:`task`) to separate stages, partition work across drivers, and schedule
concurrent work on the GPU.

Compared with Wave, cuDF offers broader relational coverage today (filter and
project, hash and streaming aggregation, hash and nested-loop joins, order by,
top-n, limit, and common aggregates/expressions) because it reuses a mature,
externally maintained library. The trade-off is that the one-to-one model does
less cross-operator fusion than Wave's generated kernels.

Building and configuration:

* The backend is included when ``VELOX_ENABLE_CUDF`` is set. cuDF supports Linux
  and WSL2 (not Windows or macOS) and has minimum CUDA, driver, and GPU
  architecture requirements (see the `RAPIDS Installation Guide
  <https://docs.rapids.ai/install/>`_). The ``adapters-cuda`` Docker image in the
  Velox repository is a convenient starting point.
* cuDF-specific runtime properties (GPU execution behavior, memory management,
  debugging) are documented in the cuDF-specific configuration section of the
  :doc:`configuration guide </configs>`.

For a deeper walk-through, see the blog post `Extending Velox — GPU Acceleration
with cuDF <https://velox-lib.io/blog/extending-velox-with-cudf>`_ and the module
``README``.

.. _gpu-breeze:

Breeze
------

`Breeze <https://github.com/facebookincubator/velox/tree/main/velox/experimental/breeze>`_
(``velox/experimental/breeze``) is a standalone, portable library of
data-parallel primitives — block- and device-level ``load``/``store``,
``reduce``, ``scan`` (decoupled look-back), and radix ``sort``. A single source
implementation maps onto multiple backends — CUDA, HIP, OpenCL, SYCL, Metal, and
OpenMP — through a thin platform-abstraction layer, so the same primitive can run
across heterogeneous hardware.

Breeze is **not** an operator backend and is not selected directly by a query
plan. It has no dependency on Velox (and does not wrap CUB, Thrust, or libcudf),
so it can be built and tested on its own. Within Velox it provides the low-level
building blocks that GPU kernels are written against; Wave builds on it. Breeze
is compiled together with the GPU backends (``VELOX_ENABLE_WAVE`` or
``VELOX_ENABLE_CUDF``).

.. _gpu-ucx:

UCX Exchange
------------

The `UCX exchange
<https://github.com/facebookincubator/velox/tree/main/velox/experimental/ucx-exchange>`_
(``velox/experimental/ucx-exchange``) is a GPU-aware replacement for Velox's
inter-worker exchange — the data movement between a task that ends in a
``PartitionedOutput`` operator and a task whose source is an ``Exchange``
operator. Using `UCX <https://openucx.org/>`_ (via UCXX), it transfers device
buffers **directly GPU-to-GPU**, avoiding the round trip through host memory that
a host-staged shuffle would incur.

The transport is engine-agnostic: it moves GPU column buffers regardless of which
backend produced them, so the same exchange serves both Wave and cuDF. Transport
is chosen per plan node — ``TransportKind::kUcx`` versus ``kHttp`` — and nodes
default to the standard HTTP path (the regular serializer over host memory)
unless explicitly opted into UCX, so it can be adopted incrementally.

The UCX exchange is experimental and is built separately from the core library
(it requires a system UCX installation and benefits from GPUDirect/RDMA-capable
hardware).

Choosing an approach
--------------------

* **cuDF** is the most complete option today for general relational SQL on a
  single GPU, reusing a mature external library.
* **Wave** targets the highest performance ceiling through whole-pipeline kernel
  fusion and on-GPU decode, but its coverage is still early.
* **Breeze** is a foundation, not an alternative — choose it when implementing
  new portable GPU primitives rather than as a way to run a plan.
* **The UCX exchange** is orthogonal to the compute backend: it addresses data
  movement between workers and can be combined with either Wave or cuDF when
  shuffle cost dominates.

Because all of these are experimental, validate coverage and correctness for
your specific plans (CPU fallback makes partial support transparent but can mask
where execution actually runs), and expect APIs and capabilities to evolve.
