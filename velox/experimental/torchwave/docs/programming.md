# TorchWave Programming Guide

## Introduction

This guide is for two audiences:

1. **Operator developers** who want to add support for new PyTorch operations (aten ops or custom ops) to TorchWave.
2. **TorchWave engine developers** who work on the compiler, executor, and runtime.

It covers the executor API, the operator registration system, kernel calling conventions, and the three execution modes (single-block, multi-kernel, and cooperative grid). The running example is `aten.masked_select`, which has implementations in all three modes.

## Executor API

### WaveGraphExecutor

`WaveGraphExecutor` is the main entry point for running an exported PyTorch model through TorchWave. It takes a `ModelContext` (which owns the nativert `Graph`, `Weights`, and config), compiles it into a `WaveGraph` of fused kernel operations, and executes it on the GPU.

```cpp
#include "velox/experimental/torchwave/Executor.h"

// 1. Load the model
auto fixture = ModelFixture::load("model.pt2");

// 2. Create the executor (compiles the WaveGraph)
WaveGraphExecutor executor(fixture->makeModelContext());

// 3. Get a pooled device frame (contains weights on GPU)
auto frame = executor.getFrame();

// 4. Fill user inputs
fillWaveFrame(executor.graph(), *frame, deviceInputs);

// 5. Execute
auto outputs = executor.executeWithPrefilledFrame(*frame);

// 6. Return the frame to the pool for reuse
executor.returnFrame(std::move(frame));
```

Key points:

- The `WaveGraphExecutor` constructor takes ownership of the graph and mutates it during compilation. The graph must not be used externally after construction.
- `getFrame()` / `returnFrame()` pool execution frames so that device-side weight buffers survive across calls. Always return frames after use.
- `executeWithPrefilledFrame` runs the compiled WaveGraph: it fills the parameter buffer, launches CUDA kernels, runs standalone ops, and returns the output IValues.
- `errorString()` returns a human-readable description of any device-side errors from the most recent execution.
- `makePerfReport()` produces a timing breakdown when `kTiming` tracing is on.

### Thread Safety and WaveThreadInfo

A single `WaveGraphExecutor` can be called from multiple threads concurrently — each thread obtains its own pooled frame via `getFrame()` and executes independently. However, the `executeWithPrefilledFrame` API is inherited from `nativert::GraphExecutorBase`, which returns only a `vector<IValue>`. There is no way to pass back execution metadata (errors, debug info, timing) through the return value.

TorchWave solves this with a thread-local: `WaveThreadInfo`. After each execution, the executor writes per-execution results into the calling thread's `WaveThreadInfo`. The caller retrieves it via `waveThreadInfo()`:

```cpp
auto outputs = executor.executeWithPrefilledFrame(*frame);

const auto& info = waveThreadInfo();
if (!info.errors.empty()) {
  LOG(ERROR) << "Kernel errors: " << info.errors;
}
```

`WaveThreadInfo` contains:

| Field | Type | Description |
|-------|------|-------------|
| `debugInfo` | `vector<vector<DebugInfo>>` | Per-launch, per-block debug info (clock cycles, op index, error line, error message). Outer vector is one entry per kernel launch; inner vector is one `DebugInfo` per thread block. |
| `launchMeta` | `vector<LaunchMeta>` | Per-launch metadata: sequence number, step index, block count, and timing breakdown (gather, grid, alloc, fill, kernel, standalone microseconds), plus input/output byte counts. |
| `errors` | `string` | Human-readable error string from `errorString()`. Maps device-side errors (non-zero `DebugInfo::line`) back to the originating kernel op and formats them with the node expression, block index, and error message. |
| `standaloneTimes` | `vector<int64_t>` | Execution times for standalone ops, sorted descending. |
| `standaloneLabels` | `vector<string>` | Node descriptions for standalone ops, parallel to `standaloneTimes`. |
| `perfReport` | `string` | Structured performance report (E2E throughput, per-node timing, thread block balance, top consumers). Populated when `kTiming` trace flag is on. |

The thread-local is populated by `executeWave` when `WaveConfig::keepStatsOnThread` is true (the default). Debug info is transferred from device to host via `collectDebugInfo`, which queues D2H copies and synchronizes before the execution state is returned to the pool.

## Processing Stages

This section describes the compilation and execution pipeline, following the data from an FX graph through code generation and into GPU execution.

### WaveGraph.cpp — Entry Point

`WaveGraph` is the top-level container that drives compilation. Its constructor (`WaveGraph::WaveGraph(ModelContext*)`) performs these stages in order:

1. **Normalize** (`normalizeAndAnnotateGraph`) — Fills in missing attribute defaults from `c10::FunctionSchema` (e.g. a `clamp` node missing its `max` argument gets `None` filled in). Substitutes operations that have multi-part implementations: if a node's `Metadata` has `makeMultiKernelVariant`, the variant subgraph is recorded in `multiKernelVariants_` for later use.

2. **Optimize** (`Optimizer`) — Propagates `ValueConstraint` (rank, dtype) backward from outputs through producers. This lets TorchWave know that a `view` or `reshape` is a no-op when the input is already 1D, and enables eliding redundant view chains.

3. **Partition** (`ParallelNodes::makeParallelNodes`) — Splits the graph into layers (see below).

4. **Compile** (`CompileCtx::compileNode` per layer) — Generates CUDA code for each layer (see below).

5. **Build execution plan** — Produces `CompiledNode` objects that the `WaveGraphExecutor` iterates at runtime.

The result is a vector of `CompiledNode`, each owning a `CompositeInvocation` that knows how to launch the compiled kernel and run standalone ops.

### ParallelExpr.cpp — Layering

`ParallelNodes::makeParallelNodes` divides the FX graph into consecutive layers, each represented by a `ProjectNode`. Within a layer, the top-level expressions are **data-independent** — they share no intermediate values and can execute in any order or in parallel.

The algorithm works by reference counting: it walks the graph's nodes and identifies which ones have no unsatisfied data dependencies (their inputs are all produced by earlier layers or are graph inputs). These become the current layer's expressions. Once placed, their consumers' reference counts drop, revealing the next layer.

Each `ProjectNode` stores:
- `nodes_` — the top-level expression roots for this layer.
- `inputs_` — the boundary nodes from earlier layers that this layer reads.
- `input_` — pointer to the preceding `ProjectNode`.

The layering determines execution order: layers execute sequentially, but within a layer everything is free to run in parallel.

### Compile.cpp — Grid Generation and Code Generation

`CompileCtx::compileNode` takes a single `ProjectNode` (layer) and produces a `CompiledNode`. For each top-level expression in the layer:

1. **Extract subgraph** (`extractSubgraph`) — Walks the expression tree from root to its leaf inputs, collecting all nodes and their input values.

2. **Deduplicate** (`makeProjectionOperation` with `SubgraphMap`) — Expressions that have the same tree structure, same operation types, same ranks and dtypes, but differ only in their leaf input values or scalar constants, are deduplicated. The first occurrence creates a `ProjectOperation` with a **formal** subgraph. Subsequent matches reuse the same `ProjectOperation` and bind their **actual** values through an `OpInvocation`. The formal subgraph defines the code; each invocation maps formal value IDs to actual value IDs via `FormalToActual` and formal nodes to actual nodes via `NodeMap`. Constants from the actual subgraph are stored in the `OpInvocation` and patched into the parameter buffer at launch time. This deduplication dramatically reduces generated code size — a model with 100 identical `a + b * c` expressions generates code once and binds it 100 times.

3. **Place kernels** (`placeKernels`) — Walks each subgraph and decides which nodes run as fused GPU kernel code and which run as standalone OpKernel invocations. Elementwise chains are fused together. Nodes marked `isStandalone` or with unsupported rank are pushed to standalone execution.

4. **Generate variant grids** — For each `ProjectOperation`, up to three `LaunchGrid` variants are generated:
   - **Default multi-block grid** (`grid_`): Each fused subgraph becomes a `KernelOperation` that can span multiple thread blocks. Multi-kernel ops (like `masked_select`) are expanded via `makeMultiKernelVariant` into multi-step launch sequences.
   - **Single-block grid** (`singleBlockGrid_`): A variant where ops use their single-block implementations. Used when the input is small enough that one block can handle it.
   - **CG grid** (`cgGrid_`): A cooperative-grid variant where multi-stage ops use `opBarrier` instead of kernel breaks. Created via `cgVariant`.

5. **Generate CUDA code** — For each `KernelOperation`, `CompileCtx` emits CUDA code:
   - Elementwise operations go through `generateElementwise` which produces grid-strided loops calling per-element device functions.
   - Non-elementwise ops go through `fusedCode` → `functionLoop` → `makeCall` which emit a device function call with the correct parameter bindings.
   - The generated code for all `KernelOperations` in a layer is combined into a single `CompositeKernel` — one CUDA translation unit compiled via NVRTC.

### Elementwise.cpp — Elementwise Code Generation

The elementwise code generator (`generateElementwise`) produces a grid-strided loop that processes one element per thread across all fused elementwise operations. Several mechanisms handle the complexity of real-world tensor shapes:

**Complex indices and broadcast.** When all input tensors are contiguous and have the same shape, the loop index directly addresses elements — this is the "fast path". When tensors are non-contiguous (strided) or have different shapes that broadcast against each other, the "slow path" uses `complexIdx`:

- Each non-contiguous tensor has an `IntDivider`-based index calculator (`initIndexCalculator`) that decomposes a linear index into per-dimension coordinates and applies strides via `indexToOffset`.
- For broadcast, a lower-rank tensor's index calculator is initialized with the *output* tensor's dimensions (via `dimOffset = output->rank - rank`), so that its coordinates are derived from the output's linear index rather than its own.
- The fast path is guarded by a runtime bitmask: each tensor input has a bit that is set to 1 if the tensor is contiguous and matches the output shape. If all bits are set, the fast path runs; otherwise the slow path runs with `complexIdx` calls. The `--no_elementwise_fast_path` flag forces the slow path for debugging.

**Output shapes for non-materialized results.** Elementwise expressions may produce intermediate values that are never written to memory — they stay in registers and flow directly into the next fused operation. Even so, an output shape is computed and an `ElementExpr` is created for each such value. This is because the output shape determines the loop bounds: the element count of the elementwise section is the maximum element count across all inputs and outputs. Without tracking the shape of non-materialized intermediates, the loop could iterate over the wrong number of elements when broadcast is involved.

**ElementExpr.** Each `ElementExpr` represents one output of the elementwise section, recording:
- `output` — the output value.
- `inputs` — the input values that need index computation for this particular output.
- `altParamOffset` — maps values to alternative parameter offsets for broadcast adjustments. When a lower-rank tensor broadcasts against a higher-rank output, its `Tensor` descriptor may need a separate copy with the output's dims for index calculation.
- `shapeFromThisOp` — true if the output shape is determined by this operation (as opposed to being inherited from an input).

The code generator produces one loop per elementwise section. Each iteration computes all fused elementwise expressions for one element index. The generated code reads inputs, evaluates the expression tree bottom-up, and writes outputs.

### KernelOperation.cpp — Managing a Fused Kernel

A `KernelOperation` represents a single operation within a `CompositeKernel` — for example, one elementwise fusion group or one `masked_select` call. It manages:

**Parameter layout.** Each input and output value is assigned an offset in the kernel's parameter buffer (`BlockInfo::params`). Tensor arguments occupy `sizeof(Tensor)` bytes, scalars occupy 8 bytes. The `paramOffsets_` map records these. At runtime, the host writes `Tensor` descriptors (storage pointer, dims, strides) and scalar values at these offsets in a pinned buffer, which is then copied to device memory before the kernel launches.

**OutputDesc — what to allocate.** `setOutputs` walks the subgraph and produces an `OutputDesc` for each output value. The `OutputDesc` tells the runtime:
- `reserveShape`: a function that computes the output tensor dimensions from the inputs in the execution frame.
- `shapeSetOnDevice`: the kernel writes the actual shape (e.g. stream compaction result size). The host allocates worst-case and reads back the real size after execution.
- `neededOnHost`: the output must be D2H copied after the kernel finishes.
- `shapeOnly`: a fake tensor is needed to carry shape metadata but no backing memory is allocated.
- `sizeExpr`: a `SizeExpr` tree that computes the element count from input values using `kMax` (elementwise) or `kSum` (concat) shortcuts, without calling `reserveShape`.

**SizeExpr — fast element count.** The `SizeExpr` tree lets the runtime compute the number of thread blocks without calling `reserveShape` for each output. For elementwise ops, it is `kMax` over the leaf input sizes. For concat, it is `kSum`. This avoids redundant shape computation and enables grid caching — if all input sizes fall within cached bounds, the previous grid is reused.

### CompiledOp.cpp — Execution at Runtime

A `CompositeInvocation` executes one compiled layer at runtime. The `execute` method processes a sequence of **steps** (corresponding to stages of multi-kernel or CG operations):

**Step execution.** For each step index (0, 1, 2, ...):

1. **Gather launches** (`gatherLaunches`) — Collects `LaunchData` for all kernel ops and standalone ops at this step. On the first execution, this builds the `LaunchData` from `OpInvocation` bindings, resolving formal value IDs to actual frame slots. On subsequent executions with unchanged input sizes, the cached `LaunchData` is reused.

2. **Grid selection** (`selectGrid` + `makeGrid`) — For each kernel launch, decides between single-block and multi-block based on element count. Computes the number of blocks per launch, balancing work across the available SMs. The grid is cached: if input sizes haven't changed, the previous block assignment is reused.

3. **Allocate outputs** — For each kernel launch, allocates output tensors in the execution frame. Uses `OutputDesc::reserveShape` for the first execution, then reuses allocations if shapes haven't changed. Shape-only outputs get placeholder tensors with no backing memory.

4. **Fill parameter buffer** — Copies `Tensor` descriptors (storage pointer, rank, dims, strides, element size, element type) and scalar values from the execution frame into a contiguous pinned buffer at the offsets defined by `KernelOperation::paramOffsets_`. Constant values from the `OpInvocation` are also written. `BlockInfo` headers (op code, block index, block count, params pointer) are written for each block.

5. **H2D transfer** — The pinned buffer is copied to device memory asynchronously on the CUDA stream.

6. **Launch kernel** — The `CompositeKernel` is launched with the device buffer as the parameter argument. In CG mode, `launchCooperative` is used.

7. **Run standalones** — While the kernel is executing on the GPU, standalone ops for this step are executed on the host via their `OpKernel::compute` calls. This overlaps host-side standalone computation with GPU kernel execution. Standalone ops that produce values consumed by later steps write directly into the execution frame.

8. **Process returns** — After the kernel completes, any `neededOnHost` or `shapeSetOnDevice` outputs are read from the pinned buffer (which was filled by a queued D2H copy) back into the execution frame. This updates `at::Tensor` sizes for compacted outputs and copies scalar results to the host.

**Data transfer overlap.** The execution pipeline overlaps GPU and CPU work. After the kernel launch is enqueued, standalone ops for the *same step* run immediately on the CPU while the GPU processes the kernel. If there are no standalones, the host simply moves to the next step. The stream is only synchronized when needed: at the end of all steps, when `shapeSetOnDevice` outputs must be read, or when reference frame verification requires host-side comparison.

**Caching.** Multiple levels of caching avoid redundant work across repeated executions:
- **Grid cache**: If all input element counts fall within the bounds from the previous run, the block assignment is reused.
- **Launch cache**: If grid choices haven't changed, the `LaunchData` (value IDs, parameter offsets) is reused.
- **Frame generation**: A generation counter tracks frame reuse. When the frame is returned to the pool and re-obtained, the counter increments to invalidate caches.

## Declaring Operators

Every operation that TorchWave can fuse or execute must be registered in the `Registry` via a `Metadata` entry. Registration happens in `Builtins.cpp` at startup.

### The Metadata Structure

`Metadata` describes everything TorchWave needs to know about an operation:

- **`functionSchema`**: The `c10::FunctionSchema` that defines the operation's argument names, types, and defaults. For aten ops this is looked up from the PyTorch dispatcher. For TorchWave intrinsics (like `tw.masked_select_head`), a custom schema is constructed.

- **`argumentMeta`**: A vector of `ArgumentMeta`, one per schema argument. Controls how each argument is passed to device code. Must match `functionSchema->arguments().size()`.

- **`returnMeta`**: A vector of `ArgumentMeta`, one per schema return. Controls output allocation, size computation, and device-to-host transfer. Must match `functionSchema->returns().size()`.

- **`deviceFunc`**: Name of the `__device__` function to call (e.g. `"masked_select"`).

- **`headerFile`**: Path to the `.cuh` file containing the device function.

### ArgumentMeta

`ArgumentMeta` controls how a single argument or return value is handled:

| Field | Default | Meaning |
|-------|---------|---------|
| `isRegister` | false | If true, the value is passed as a scalar register (per-element for elementwise ops). If false, passed as a `Tensor*` pointer. |
| `reserveShape` | nullptr | Function that determines the output tensor shape given the inputs. Called at runtime before kernel launch. |
| `shapeSetOnDevice` | false | If true, the actual output size is written by the kernel (e.g. stream compaction). The host allocates a worst-case buffer and reads the final size after execution. |
| `neededOnHost` | false | If true, the output must be transferred back to the host after the kernel completes. Triggers a D2H copy and host-side sync. |
| `cpuOnly` | false | If true, this argument stays on CPU even when the graph targets GPU. |
| `wholeTensor` | false | For elementwise ops: pass the entire `Tensor*` instead of just the element at the current index. Used for ops like `index_elt` that need random access into a tensor. |
| `randomAccess` | false | If true and the value is produced in the same kernel, a kernel-wide barrier is required between producer and consumer. |
| `sizeShortcut` | kNone | How to compute the output element count: `kMax` = largest input (elementwise default), `kSum` = sum of inputs (concatenation). `kNone` = use `reserveShape`. |
| `sizeArgs` | {} | Which argument ordinals determine the output size. For example, `{.ordinal = {0}}` means "use argument 0's element count". |
| `hasPresentTemplateParam` | false | Emits a `bool` template parameter indicating whether this argument has a non-None value. |

### OutputReserveFunc

The `reserveShape` function determines the output shape at runtime:

```cpp
using OutputReserveFunc = std::function<std::vector<std::vector<Dim>>(
    NodeCP node,
    nativert::ExecutionFrame& frame,
    const FormalToActual& map,
    NodeCP originalFormalNode,
    const NodeMap& nodeMap)>;
```

- **`node`**: The formal node from the kernel's subgraph.
- **`frame`**: The execution frame containing input tensors.
- **`map`**: Maps formal value IDs to actual value IDs in the frame.
- **`originalFormalNode`**: The original formal node (before variant expansion). Used by `paramIntByName` and `paramIntListByName` to look up attributes on the actual node via `nodeMap`.
- **`nodeMap`**: Maps formal nodes to actual nodes. Needed when multiple invocations of the same kernel subgraph have different attribute values.

Returns a vector of shapes (one per output element for tuple returns). Each shape is a `vector<Dim>` giving the dimensions.

### Size Shortcuts

For common cases you don't need a custom `reserveShape`:

- **`SizeShortcut::kMax`** with `sizeArgs = {{0}}`: Output has the same number of elements as argument 0. This is the default for elementwise ops.
- **`SizeShortcut::kSum`** with `sizeArgs = {{0, 1}}`: Output element count is the sum of arguments 0 and 1. Used for concatenation.
- **`SizeShortcut::kNone`**: Use the `reserveShape` function.

### MetadataBuilder

The fluent `MetadataBuilder` API is the preferred way to register ops:

```cpp
// Simple elementwise op — add, mul, sub, etc.
Registry::registerElementwise("torch.ops.aten.add.Tensor");

// Op with custom metadata
MetadataBuilder("torch.ops.aten.masked_select.default")
    .sizeOrdinal({0})
    .hasBarrier()
    .singleBlockIfFused()
    .argumentMeta({{.isRegister = true}, {.isRegister = true}})
    .returnMeta(
        {{.isRegister = false,
          .reserveShape = myReserveFunc,
          .shapeSetOnDevice = true}})
    .makeMultiKernelVariant(makeMaskedSelectVariant)
    .cgVariant(makeMaskedSelectCgVariant)
    .headerFile("Scan.cuh")
    .deviceFunc("masked_select")
    .sharedDecls({{"Int32X32", "warpSums"}, {"uint32_t", "counter"}})
    .typeTemplateParams({0})
    .hasBlockSizeTemplateParam()
    .alwaysSingleBlock()
    .registerOp();
```

Key builder methods:

| Method | Purpose |
|--------|---------|
| `elementwise()` | Mark as elementwise; sets `isRegister` on all args, `sizeArgs={0}`, `inPlaceIfLastUse`. |
| `deviceFunc(name)` | Name of the `__device__` function. |
| `headerFile(path)` | `.cuh` file to include. |
| `typeTemplateParams({0})` | Argument 0's dtype becomes a template parameter. |
| `hasBlockSizeTemplateParam()` | Block size becomes the first template parameter. |
| `sharedDecls({...})` | Declare `__shared__` variables passed as trailing args. |
| `makeMultiKernelVariant(fn)` | Function to create the multi-kernel variant graph. |
| `cgVariant(fn)` | Function to create the cooperative grid variant graph. |
| `hasBarrier()` | The op needs all block values ready before executing. |
| `singleBlockIfFused()` | When fused, force single-block grid. |
| `alwaysSingleBlock()` | Always use single-block grid regardless of context. |
| `multiBlockReturnBarrier()` | Needs a barrier after output before consumers can read. |
| `inputFromPreviousKernel(n)` | Argument `n` is always an output of the previous kernel stage. |

## Standalone Kernels

When TorchWave encounters an operation it cannot fuse — either because it has no `Metadata` entry, or because the metadata marks it as standalone — it falls back to executing the operation through the nativert `OpKernel` mechanism (PyTorch's standard per-op dispatch).

Even for standalone ops, TorchWave needs to know the **rank and type** of each output so it can:

1. Allocate output tensors in the correct shape before launching subsequent fused kernels that consume them.
2. Propagate type and rank constraints through the graph during optimization.

If an op lacks metadata entirely, TorchWave infers what it can from the graph's `TensorMeta` annotations (which come from `torch.export`'s shape inference). But when metadata exists, the `outputConstraints` function provides more precise information:

```cpp
MetadataBuilder("torch.ops.aten.some_standalone_op.default")
    .isStandalone()
    .outputConstraints([](NodeCP node, const ValueTypes& types)
        -> std::vector<ValueConstraint> {
      // Output has same rank and dtype as input 0
      return {types.constraint(node->inputs()[0].value)};
    })
    .registerOp();
```

The `only1d` flag provides a middle ground: the op is fusable for 1D inputs but falls back to standalone when any input has rank > 1.

## Writing Kernels

### Device-Side Data Types

TorchWave kernels operate on these device-side types defined in `KernelParams.h`:

**`Tensor`** — A device-side tensor descriptor. Not a PyTorch `at::Tensor`; it is a lightweight struct that lives in the kernel parameter buffer:

```cpp
struct Tensor {
  void* storage;        // Pointer to the raw data on device
  int8_t rank;          // Number of dimensions (max 3)
  uint8_t elementSize;  // Bytes per element
  uint8_t elementType;  // c10::ScalarType enum value
  int32_t dims[3];      // Dimension sizes
  int32_t strides[3];   // Strides in elements
  uint32_t numEl;       // Total number of elements
  bool contiguous;      // True if row-major contiguous
  IntDivider sizes[3];  // Fast dividers for index decomposition
};
```

Access the data pointer with `storage<T>(tensor)`. Get the element count with `numEl(tensor)`.

**`TensorList`** — A list of tensor pointers:

```cpp
struct TensorList {
  int64_t size;       // Number of tensors
  Tensor** tensors;   // Array of Tensor pointers
};
```

**`ScalarList`** — A list of int64 scalars:

```cpp
struct ScalarList {
  int64_t size;
  int64_t* data;
};
```

**`BlockInfo`** — Per-block execution context, loaded from shared memory at kernel entry:

```cpp
struct BlockInfo {
  void* params;           // Base pointer to the parameter buffer
  int32_t op;             // Operation index within the kernel
  int32_t blockInOp;      // This block's index within the operation
  int32_t numBlocksInOp;  // Total blocks assigned to this operation
  DebugInfo* debugInfo;   // Per-block debug/error output
  // ... timing fields
};
```

### Calling Conventions

A TorchWave device function receives arguments in this order:

1. **Template parameters** (compile-time):
   - Block size (`kBlockSize`) if `hasBlockSizeTemplateParam` is set.
   - Element types from `typeTemplateParams` (e.g. `typename T` resolved from input 0's dtype).
   - Dtype template if `hasDtypeTemplateParam` is set.
   - Attribute templates from `templateAttrs`.
   - Presence booleans from `hasPresentTemplateParam`.

2. **Input/output arguments** (one per schema argument and return):
   - `Tensor*` for tensor arguments (when `isRegister` is false).
   - Scalar types (`int32_t*`, `float`, etc.) for scalar arguments.
   - For elementwise with `isRegister = true`, the element value (e.g. `T input`) rather than a pointer.

3. **Shared memory variables** (from `sharedDecls` and `dynamicSharedDecls`):
   - Declared as `__shared__` at the top of the containing kernel.
   - Passed by reference as trailing arguments.

4. **BlockInfo&** — always the last argument.

### Example: Single-Block masked_select

```cpp
template <int32_t kBlockSize, typename T>
__device__ void masked_select(
    Tensor* input,       // arg 0: input tensor
    Tensor* mask,        // arg 1: mask tensor
    Tensor* output,      // return 0: output tensor
    void* temp,          // shared: Int32X32 warpSums (cast to void*)
    uint32_t& counter,   // shared: uint32_t counter
    uint32_t idx,        // loop index (from grid-strided loop)
    uint32_t size,       // total element count
    BlockInfo& block) {  // always last
  // ...
}
```

The generated kernel wraps this in a grid-strided loop:

```cuda
__shared__ Int32X32 warpSums;
__shared__ uint32_t counter;
// ...
for (uint32_t idx = threadIdx.x; idx < size; idx += blockDim.x) {
  masked_select<256, float>(input, mask, output, warpSums, counter,
                            idx, size, blockInfo);
}
```

### Elementwise Kernel Functions

Elementwise functions take scalar values (not tensor pointers) and return a scalar:

```cpp
// In Elementwise.cuh
template <typename T>
__device__ inline T __add(T a, T b) {
  return a + b;
}
```

These are called per-element inside a grid-strided loop. The framework handles loading from and storing to tensors, including non-contiguous access via `indexToOffset`.

When an elementwise argument has `wholeTensor = true`, that argument is passed as `Tensor*` instead of as a scalar element. This is used for operations like `index_elt` that need to index into the tensor at arbitrary offsets.

### Shared Memory

Shared memory variables are declared via `sharedDecls` in the metadata:

```cpp
.sharedDecls({{"Int32X32", "warpSums"}, {"uint32_t", "counter"}})
```

This declares `__shared__ Int32X32 warpSums;` and `__shared__ uint32_t counter;` at the top of the generated kernel function, and passes them as trailing arguments (before `BlockInfo`) to the device function.

`dynamicSharedDecls` works similarly but the type depends on the runtime dtype of an input argument. The variable name is suffixed with the type name to avoid collisions when multiple types appear in one kernel.

### The `Int32X32` Type

`Int32X32` is a 256-byte shared memory scratch buffer used as the `void* temp` argument for scan and reduce operations:

```cpp
struct Int32X32 {
  uint64_t data[32];
  operator void*() { return data; }
};
```

It is large enough to hold one warp-reduce result per warp (up to 32 warps for a 1024-thread block).

## Execution Modes

TorchWave supports three execution modes. The runtime picks the best mode based on input size and available metadata. Each mode handles inter-block coordination differently.

### Single-Block Mode

A single thread block processes the entire input. The block iterates over all elements in a grid-strided loop. Synchronization within the block uses `__syncthreads()`.

This is the simplest mode. It works well for small inputs where launch overhead dominates. Many operations only have a single-block implementation (e.g. `aten.cumsum` with `singleBlockIfFused`).

The single-block `masked_select` does an inclusive prefix sum in one pass:

```cpp
template <int32_t kBlockSize, typename T>
__device__ void masked_select(
    T input, bool flag, Tensor* output,
    void* temp, uint32_t& counter,
    uint32_t idx, uint32_t size, BlockInfo& block) {
  T* out = storage<T>(output);
  if (idx == 0) counter = 0;
  if (idx >= size) flag = false;
  int32_t sum = inclusiveSum<uint32_t, kBlockSize>(
      threadIdx.x == 0 ? uint32_t(flag) + counter : uint32_t(flag),
      &counter, (uint32_t*)temp);
  if (flag) out[sum - 1] = input;
  if (idx == size - 1) output->dims[0] = sum;
  __syncthreads();
}
```

The `counter` shared variable carries the running sum across iterations of the grid-strided loop. `__syncthreads()` at the end ensures the counter is visible to all threads before the next iteration.

### Multi-Kernel Mode

For large inputs, a single block cannot saturate the GPU. Multi-kernel mode splits the work into multiple kernel launches separated by implicit host-side barriers (the kernel launch boundary itself).

A multi-kernel variant is created by the `makeMultiKernelVariant` function, which takes the original single node and creates a sequence of nodes connected by intermediate tensors. Each node in the sequence becomes a separate kernel launch.

**`masked_select` multi-kernel variant** has three stages:

1. **`tw.masked_select_head`** — Each of N blocks counts the number of selected elements in its chunk. Writes one `int32_t` per block to a `counts` tensor. Marked with `multiBlockReturnBarrier` to force a kernel break.

2. **`tw.add_sizes`** — A single block computes the inclusive prefix sum of the `counts` tensor, converting per-block counts into per-block output offsets. Also marked with `multiBlockReturnBarrier`. Writes the total count to a scalar output.

3. **`tw.masked_select_final`** — Each block scatters its selected elements to the correct positions using the prefix-summed offsets. Takes `input`, `mask`, `counts`, and `total` as inputs.

The variant graph is constructed in `makeMaskedSelectVariant`:

```cpp
nativert::Node* makeMaskedSelectVariant(NodeCP single, WaveGraph* waveGraph) {
  auto* graph = variantNodeGraph(waveGraph);

  // Stage 1: per-block counts
  auto* headNode = graph->createNode("tw.masked_select_head",
      {{"input", single->inputs()[0].value},
       {"mask", single->inputs()[1].value}});
  auto* headOutput = newVariantTensorValue(headNode, waveGraph,
      "masked_select_counts", c10::ScalarType::Int);

  // Stage 2: prefix sum of counts
  auto* addSizesNode = graph->createNode("tw.add_sizes",
      {{"input", headOutput}});
  auto* addSizesOutput = newVariantScalarValue(addSizesNode, waveGraph,
      "masked_select_total", c10::ScalarType::Int);

  // Stage 3: scatter
  auto* finalNode = graph->createNode("tw.masked_select_final",
      {{"input", single->inputs()[0].value},
       {"mask", single->inputs()[1].value},
       {"counts", headOutput},
       {"total", addSizesOutput}});
  copyOriginalOutputs(finalNode, single, waveGraph);
  return finalNode;
}
```

Each stage must be registered as its own intrinsic with a custom `c10::FunctionSchema`:

```cpp
MetadataBuilder(std::make_unique<c10::FunctionSchema>(
    "tw.masked_select_head", "",
    {c10::Argument("input", c10::TensorType::get()),
     c10::Argument("mask", c10::TensorType::get())},
    {c10::Argument("output", c10::TensorType::get())}))
    .sizeOrdinal({0})
    .hasBarrier()
    .returnMeta({{.isRegister = false, .reserveShape = numBlocksShape}})
    .headerFile("Scan.cuh")
    .deviceFunc("masked_select_head")
    .sharedDecls({{"Int32X32", "warpSums"},
                  {"uint32_t", "size"},
                  {"uint32_t", "rounded"}})
    .typeTemplateParams({0})
    .hasBlockSizeTemplateParam()
    .multiBlockReturnBarrier()
    .registerOp();
```

The `inputFromPreviousKernel` field on later stages tells the compiler which input must be produced by the preceding stage, ensuring correct ordering.

### Cooperative Grid (CG) Mode

Cooperative grid mode uses `cudaLaunchCooperativeKernel` so that all thread blocks in a kernel launch can synchronize with each other via device-side barriers — without returning to the host between stages.

A CG variant is created by the `cgVariant` function. Unlike multi-kernel variants, a CG variant is a single node that performs all stages internally, using `opBarrier(block, barN)` calls between stages.

**`masked_select_cg`** combines all three stages into one function:

```cpp
template <int32_t kBlockSize, typename T>
__device__ void masked_select_cg(
    Tensor* input, Tensor* mask,
    Tensor* output, Tensor* counts,
    void* temp,
    uint32_t& size, uint32_t& rounded, uint32_t& counter,
    int32_t bar0, int32_t bar1, int32_t bar2,
    BlockInfo& block) {
  // Stage 1: per-block counts
  masked_select_head<kBlockSize, T>(
      input, mask, counts, temp, size, rounded, block);
  opBarrier(block, bar0);

  // Stage 2: prefix sum (single block only)
  if (block.blockInOp == 0) {
    add_sizes<kBlockSize>(counts, (int32_t*)nullptr,
        temp, size, rounded, counter, block);
  }
  opBarrier(block, bar1);

  // Stage 3: scatter
  masked_select_final<kBlockSize, T>(
      input, mask, counts, (int32_t*)nullptr,
      output, temp, size, rounded, block);
  opBarrier(block, bar2);
}
```

Key CG concepts:

- **`opBarrier(block, barN)`** is a device-side inter-block barrier. All blocks assigned to this operation must reach the barrier before any can proceed. The `barN` argument is an integer barrier index, managed by the runtime.

- **`numBarriers(N)`** in the metadata tells the compiler how many `opBarrier` calls the function makes. The runtime allocates barrier slots accordingly.

- **Single-block stages**: When only one block should execute a stage (like `add_sizes`), guard with `if (block.blockInOp == 0)`. All blocks still must call the surrounding `opBarrier` calls.

- **Extra outputs**: CG variants often need extra tensor outputs for intermediate storage (e.g. `counts`). These are added via `newVariantTensorValue` in the `cgVariant` function and appear as additional return values in the schema.

### Choosing Between Modes

The runtime chooses the execution mode based on:

1. **Input size**: Small inputs use single-block mode. The threshold is roughly `blockSize` elements.
2. **`alwaysSingleBlock`**: Forces single-block mode regardless of input size.
3. **`singleBlockIfFused`**: Uses single-block when the op is fused with others in the same kernel.
4. **CG availability**: CG mode is used when `--cg 1` is set or auto-detected as beneficial, and a `cgVariant` exists.
5. **Fallback**: If no multi-kernel variant exists and the input is too large for single-block, the op may need to run as a standalone.

### Adding New Temporary State

To introduce new shared or scratch storage for a kernel:

1. **Shared memory**: Add entries to `sharedDecls`:
   ```cpp
   .sharedDecls({{"uint32_t", "counter"}, {"Int32X32", "warpSums"}})
   ```
   These become `__shared__` variables and trailing arguments.

2. **Dynamic shared memory** (type depends on input dtype): Use `dynamicSharedDecls`:
   ```cpp
   .dynamicSharedDecls({{0, "accumulator"}})
   ```
   The type is resolved from input 0's dtype. The name is suffixed (e.g. `accumulatorFloat`).

3. **Intermediate tensors** (for multi-kernel and CG variants): Create them in the variant function using `newVariantTensorValue` or `newVariantScalarValue`. They are allocated by the runtime between stages.

4. **Barrier indices** (CG only): Barrier slots are passed as `int32_t bar0, bar1, ...` arguments. Declare the count with `.numBarriers(N)`.

### Summary: masked_select in Three Modes

| Mode | Kernel launches | Synchronization | Registration |
|------|----------------|-----------------|--------------|
| Single-block | 1 | `__syncthreads()` in loop, `counter` shared var | Base `masked_select` with `alwaysSingleBlock` |
| Multi-kernel | 3 (`head`, `add_sizes`, `final`) | Implicit (kernel launch boundary) | `makeMultiKernelVariant` + 3 intrinsic registrations |
| Cooperative grid | 1 | `opBarrier(block, barN)` between stages | `cgVariant` + 1 intrinsic registration with `numBarriers(3)` |

All three modes produce the same output. The runtime picks the best mode based on input size and configuration.
