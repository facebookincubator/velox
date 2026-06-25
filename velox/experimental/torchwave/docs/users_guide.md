# TorchWave User's Guide

## Introduction

### What is TorchWave?

TorchWave is a fused GPU execution engine for PyTorch FX graphs. It takes an exported PyTorch model (a `.pt2` file produced by `torch.export`) and executes it on the GPU using fused CUDA kernels. Instead of launching one CUDA kernel per operation — as PyTorch's eager mode or the nativert serial executor does — TorchWave fuses multiple operations into a single kernel launch, reducing launch overhead, improving data locality, and increasing GPU utilization.

TorchWave is designed for feature preprocessing workloads in recommendation models. These workloads typically consist of many small elementwise operations, scatter/gather indexing, masked selects, and concatenations — exactly the kind of operations that benefit most from kernel fusion.

Key capabilities:

- **Elementwise fusion**: Chains of pointwise operations (add, mul, clamp, relu, etc.) are compiled into a single grid-strided CUDA kernel.
- **Multi-block cooperative kernels**: Operations like `cat`, `index_put`, and `clone` that write to shared output buffers use cooperative grid launches with inter-block barriers.
- **Automatic grid sizing**: The runtime selects between single-block and multi-block grid variants based on input size and balances thread blocks across fused operations.
- **JIT CUDA compilation**: Elementwise kernels are compiled at runtime via NVRTC, with optional on-disk caching.
- **Reference frame verification**: Every intermediate value can be compared against a CPU reference to pinpoint the first divergent operation.

## Getting Started

### Test Model Workflow

The typical workflow has two steps:

1. **Export the model** from Python to a `.pt2` archive (and a `_results.pt` reference file).
2. **Run the model** through `executor_test` to compare serial CPU, serial GPU, and Wave execution.

#### Step 1: Write a Test Model

Create an `nn.Module` that exercises the operations you want to test. Place it in `torchwave/tests/`. For example, `element_test.py`:

```python
import torch
from torch import nn, Tensor

class ElementTest(nn.Module):
    def forward(self, a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
        s = a + b
        d = a - b
        return s * d, torch.clamp(s, 0, 100)
```

Then write a companion `_run_pt2.py` script that creates sample inputs, runs the module eagerly to produce reference outputs, and exports the model:

```python
import torch
from velox.experimental.torchwave.tests.element_test import ElementTest

module = ElementTest()
a = torch.arange(10000, dtype=torch.float)
b = torch.arange(10000, dtype=torch.float) * 0.5
results = module(a, b)
torch.save(list(results), "data/element_test_results.pt")

with torch.no_grad():
    exported = torch.export.export(module, (a, b), strict=False)
torch.export.save(exported, "data/element_test.pt2")
```

Add corresponding `python_library` and `python_binary` targets to the `BUCK` file (see existing entries for the pattern).

Generate the test data:

```bash
buck run velox/experimental/torchwave/tests:element_test_run_pt2
```

Copy the generated `.pt2` and `_results.pt` files from the buck-out link tree into `torchwave/tests/data/`.

#### Step 2: Run with executor_test

```bash
buck run @//mode/opt velox/experimental/torchwave/tests:executor_test -- \
  --gtest_filter="*.custom" --custom data/element_test
```

The `--custom` flag takes a base name (without the `.pt2` / `_results.pt` suffix). The test:

1. Loads the `.pt2` model and `_results.pt` reference outputs.
2. Runs the model through the **nativert serial CPU executor** and verifies outputs match the reference.
3. Runs through the **nativert serial GPU executor** (one PyTorch kernel launch per op) and verifies.
4. Compiles the model into a **WaveGraph** and runs through the **Wave executor**, then verifies.
5. Prints timing for all three paths.

Add `--wave_only` to skip the serial CPU and serial GPU runs (useful when iterating on Wave-specific issues):

```bash
buck run @//mode/opt velox/experimental/torchwave/tests:executor_test -- \
  --gtest_filter="*.custom" --custom data/element_test --wave_only
```

### Using graph_tool

`graph_tool` is a standalone binary for inspecting `.pt2` files without running them:

```bash
buck run @//mode/opt velox/experimental/torchwave/tests:graph_tool -- \
  --pt2 path/to/model.pt2 --print_graph
```

Useful flags:

| Flag | Description |
|------|-------------|
| `--pt2 PATH` | Path to the `.pt2` file (required). |
| `--list_models` | List model names in the archive and exit. |
| `--print_graph` | Print the graph signature and output node as an expression tree. |
| `--print_params` | Print model parameters and sample input metadata. |
| `--compile` | Compile the graph into a WaveGraph and print the result. |
| `--optimize` | Apply graph optimization passes before printing. |
| `--value_meta` | Show value type and rank annotations. |
| `--describe_pt PATH` | Describe the contents of a `.pt` file (tensor shapes, dtypes). |
| `--print_options OPTS` | Set NodePrinter formatting (see [Print Options](#print-options)). |

Example — inspect a model's compiled WaveGraph structure:

```bash
buck run @//mode/opt velox/experimental/torchwave/tests:graph_tool -- \
  --pt2 data/element_test.pt2 --compile
```

## Common Flags

### WaveConfig Flags

These flags map directly to fields on `WaveConfig` and control how the Wave executor compiles and runs the graph. They are set via gflags on `executor_test`.

| Flag | Default | WaveConfig Field | Description |
|------|---------|-----------------|-------------|
| `--block_dim N` | 256 | `blockSize` | CUDA thread block size (threads per block). |
| `--standalone_kernels` | false | `allStandalone` | Treat all operations as standalone kernel launches (no fusion). Useful for isolating bugs to individual ops. |
| `--single_block N` | -1 | `useSingleBlock` | Force grid choice: -1 = auto, 0 = always multi-block, 1 = always single-block. |
| `--cg N` | -1 | `isCg` | Force cooperative grid: -1 = auto, 0 = off, 1 = on. |
| `--trace N` | 0 | `trace` | Trace bit mask (see [Using Tracing](#using-tracing)). |
| `--trace_values IDS` | "" | `traceValues` | Comma-separated value ids to trace (see [Using Tracing](#using-tracing)). |
| `--tensor_print_limit N` | 100 | `tensorPrintElementLimit` | Max elements printed per tensor when tracing. 0 = no limit. |
| `--kernel_cache_dir DIR` | "" | `kernelCacheDir` | Directory for caching compiled CUDA kernels (cubin files). Avoids recompilation across runs. |
| `--max_elementwise_vars N` | 7 | `maxElementwiseVars` | Max pointer variables in elementwise codegen. Beyond this threshold, storage expressions are inlined instead of using named variables. |
| `--out_of_line_expr_size N` | 10000 | `outOfLineExprSize` | Character threshold for extracting elementwise subtrees into `__noinline__` device helper functions. |
| `--no_elementwise_fast_path` | false | `noElementwiseFastPath` | Skip the elementwise fast path; always generate the slow path with `complexIdx`. Useful for debugging fast-path-specific bugs. |
| `--throw_on_error` | true | `throwOnError` | Throw after execution if a device-side error is detected. Set to false to print the error and continue. |
| `--continue_after_mismatch` | false | `continueAfterMismatch` | Log reference frame mismatches but continue instead of throwing. Useful for seeing all mismatches at once. |
| `--kernel_debug_output` | false | `kernelDebugOutput` | Enable device-side debug `printf`. Emergency use only — floods output. |
| `--debug_single_ops` | false | `debugSingleOps` | Launch each kernel op as a standalone invocation, waiting between launches. Makes device-side errors attributable to a single op. |
| `--reverify` | false | `reverify` | Re-verify all previously-passed reference values on every step to detect value corruption. |
| `--auto_adjust_cost` | false | `autoAdjustCost` | Adjust per-op cost multipliers after each execution based on actual thread block clock distribution. Improves block allocation balance across repeated runs. |

### Execution Flags

These flags control `executor_test` behavior but do not map to `WaveConfig`:

| Flag | Default | Description |
|------|---------|-------------|
| `--custom NAME` | "" | Base name of the test model (e.g. `data/element_test`). The test appends `.pt2` and `_results.pt`. |
| `--wave_only` | false | Skip serial CPU and serial GPU runs. |
| `--num_repeats N` | 1 | Number of timed repetitions. Timing stats (min/avg/p90/max) are printed when N > 1. |
| `--single_ops N` | -1 | -1 = run both normal and debug-single-ops modes, 0 = normal only, 1 = single-ops only. |
| `--standalone_timing` | false | Print per-op standalone execution times after the Wave run. |
| `--print_timing` | false | Print per-step timing breakdown. |
| `--print_full_mismatch` | false | Print full tensor contents for mismatched outputs. |
| `--list N` | 0 | List the WaveGraph before execution: 0 = off, 1 = expressions (kExprs), 2 = grid info (kGrids). |
| `--save_reference_frame PATH` | "" | Save a reference frame after the serial CPU run (see [Using Reference Frames](#using-reference-frames)). |
| `--reference_frame PATH` | "" | Load a reference frame and verify Wave intermediates against it. |

### Print Options

The `--print_options` flag (available on both `graph_tool` and `executor_test`) controls how node expressions are formatted. It takes a comma-separated list of tokens:

| Token | Effect |
|-------|--------|
| `D<n>` | Set max recursion depth. `D3` shows at most 3 levels of nested expressions. |
| `L<n>` | Set max argument list length. `L4` truncates argument lists beyond 4 entries. |
| `S` | Short names — strip `torch.ops.aten.` prefix and `.default` / `.Tensor` suffixes. |
| `V` | Per-line values — each node is printed on its own line with output ids, rather than inlined. |
| `NA` | No attributes — suppress node attributes in the output. |
| `VN` | Value names — print values by their graph name instead of `%id`. |
| `II` | Inline intermediates (default). |
| `OI` | Show output ids before expressions. |
| `CR` | Compress consecutive output id ranges (e.g. `[%5 - %8]`). |
| `T` | Show type annotations (e.g. `(2Df)` for 2D float). |
| `OF` | Show output descriptor flags. |

Example:

```bash
buck run @//mode/opt velox/experimental/torchwave/tests:graph_tool -- \
  --pt2 data/element_test.pt2 --print_graph --print_options "D3,S,VN"
```

## Using Reference Frames

Reference frames let you compare every intermediate value produced by the Wave executor against the values produced by the nativert serial CPU executor. This is the primary tool for debugging correctness issues.

### Step 1: Generate a Reference Frame

Run the model through `executor_test` with `--save_reference_frame` to capture all intermediate values from the serial CPU execution:

```bash
buck run @//mode/opt velox/experimental/torchwave/tests:executor_test -- \
  --gtest_filter="*.custom" --custom data/element_test \
  --save_reference_frame /tmp/ref_frame.pt
```

This saves a `.pt` file mapping each `ValueId` to its tensor or scalar value as computed by the nativert serial executor on CPU.

### Step 2: Run Wave with the Reference Frame

Pass the saved frame back with `--reference_frame`:

```bash
buck run @//mode/opt velox/experimental/torchwave/tests:executor_test -- \
  --gtest_filter="*.custom" --custom data/element_test --wave_only \
  --reference_frame /tmp/ref_frame.pt
```

After each execution step (each kernel launch or standalone op), the Wave executor compares every output value against the reference. The first mismatch throws with a message identifying the value id, the producing node, and the magnitude of the difference.

### Related Flags

- **`--continue_after_mismatch`**: Instead of throwing on the first mismatch, log all mismatches and continue. This is useful for seeing the full scope of a problem — sometimes a single root cause (e.g. a bad constant cast) manifests as mismatches on many downstream nodes.

- **`--reverify`**: On every step, re-check all previously-verified values. This detects corruption — a value that was correct after step N but gets overwritten before step N+2 (e.g. because two ops share a buffer or a barrier is missing). Without `--reverify`, each value is checked only once, immediately after it is produced.

- **`--throw_on_error`**: Controls whether device-side errors (out-of-range indices, etc.) throw or are printed. When debugging with reference frames, you usually want this set to `true` (the default) so execution stops at the first error. Set to `false` when you want to see all errors and mismatches in a single run.

### Workflow Tips

A typical debugging session:

```bash
# 1. Generate the reference frame (runs serial CPU, saves intermediates)
buck run @//mode/opt velox/experimental/torchwave/tests:executor_test -- \
  --gtest_filter="*.custom" --custom data/my_model \
  --save_reference_frame /tmp/ref.pt

# 2. Run Wave with the reference, see all mismatches
buck run @//mode/opt velox/experimental/torchwave/tests:executor_test -- \
  --gtest_filter="*.custom" --custom data/my_model --wave_only \
  --reference_frame /tmp/ref.pt --continue_after_mismatch

# 3. Narrow down: add --reverify to detect corruption
buck run @//mode/opt velox/experimental/torchwave/tests:executor_test -- \
  --gtest_filter="*.custom" --custom data/my_model --wave_only \
  --reference_frame /tmp/ref.pt --reverify

# 4. Debug a specific value: trace it
buck run @//mode/opt velox/experimental/torchwave/tests:executor_test -- \
  --gtest_filter="*.custom" --custom data/my_model --wave_only \
  --reference_frame /tmp/ref.pt --trace_values "42,43,44"
```

## Using the Select Benchmark

The select benchmark measures TorchWave's performance on `masked_select` workloads with varying numbers of fused operations and input sizes.

### Generating Benchmark Data

The benchmark script generates `.pt2` and `_results.pt` files for a matrix of configurations:

- **N** (number of `masked_select` operations): 1, 2, 4, 8, 16, 32, 64, 128
- **Size** (input tensor length): 1K to 1M elements

```bash
buck run velox/experimental/torchwave/tests:select_benchmark
```

This writes files named `select_<N>_<size>.pt2` and `select_<N>_<size>_results.pt` into `torchwave/tests/data/`.

### Running the Benchmark

Run individual configurations through `executor_test`:

```bash
# 32 masked_selects on 64K-element inputs
buck run @//mode/opt velox/experimental/torchwave/tests:executor_test -- \
  --gtest_filter="*.custom" --custom data/select_32_64000 \
  --wave_only --num_repeats 10
```

The output shows timing for the Wave executor with min/avg/p90/max over the repeated runs. Compare against the serial GPU timing (drop `--wave_only`) to see the speedup from fusion.

### Running the Full Sweep

The `select_report.sh` script sweeps all N x size combinations and prints a table with p90 timings for serial GPU, wave (multi-block), wave (CG), and wave (single-block) modes. Each configuration runs 40 repeats.

```bash
bash velox/experimental/torchwave/tests/select_report.sh path/to/data/directory
```

The script takes the directory containing the generated `select_*.pt2` files as its argument. Output is a formatted table:

```
n     size     serial_gpu_p90      wave_p90      wave_cg_p90  wave_1blk_p90
----  --------  ---------------  ---------------  ---------------  ---------------
1      1000              12              8             10              7
1      2000              14              9             11              8
...
```

## Using Tracing

### The `--trace` Flag

The `--trace` flag takes a bit mask that controls what execution information is printed:

| Bit | Value | Name | Output |
|-----|-------|------|--------|
| 0 | 1 | `kNodes` | Node headers — prints the node target and its expression tree for each operation as it executes. |
| 1 | 2 | `kLaunches` | Launch details — prints per-launch info including thread block count, element count, and the operations in each launch. Also prints standalone node invocations. |
| 2 | 4 | `kTensors` | Tensor outputs — after each standalone node, prints the shape and first elements of each output tensor. |
| 3 | 8 | `kFrame` | Frame values — prints value ids and tensor summaries for output slots after standalone execution. |
| 4 | 16 | `kTiming` | Performance report — at the end of execution, prints a structured report with E2E throughput, per-node timing, per-step breakdown (gather/grid/alloc/fill/kernel), thread block balance, and top time consumers. |

Combine bits by adding them. For example, `--trace 3` enables both node headers and launch details. `--trace 16` enables only the timing report.

```bash
# Node headers + launch details
buck run @//mode/opt velox/experimental/torchwave/tests:executor_test -- \
  --gtest_filter="*.custom" --custom data/element_test --wave_only --trace 3

# Timing report only
buck run @//mode/opt velox/experimental/torchwave/tests:executor_test -- \
  --gtest_filter="*.custom" --custom data/element_test --wave_only --trace 16
```

### The `--trace_values` Flag

`--trace_values` takes a comma-separated list of value ids (the `%N` numbers from the graph) and prints the full tensor contents for those values as they flow through the serial CPU and Wave executors. Value tracing does not apply to the serial GPU executor.

```bash
buck run @//mode/opt velox/experimental/torchwave/tests:executor_test -- \
  --gtest_filter="*.custom" --custom data/element_test \
  --trace_values "5,10,15"
```

Each traced value is printed once — the first time it appears in the frame after being computed. The output includes a label indicating which executor and phase produced it: `serial input`, `serial in`, and `serial out` for the CPU serial path; `input` and `output` for the Wave path.

Use `--tensor_print_limit N` to control how many elements are printed per tensor. The default is 100. Set to 0 for no limit (prints the entire tensor), but be cautious with large tensors.

```bash
# Print up to 500 elements per traced tensor
buck run @//mode/opt velox/experimental/torchwave/tests:executor_test -- \
  --gtest_filter="*.custom" --custom data/element_test \
  --trace_values "5,10" --tensor_print_limit 500
```

### Finding Value IDs

To figure out which value ids to trace, use `graph_tool` to print the graph with value ids visible:

```bash
buck run @//mode/opt velox/experimental/torchwave/tests:graph_tool -- \
  --pt2 data/element_test.pt2 --print_graph --print_options "V,S"
```

The `V` option prints each node on its own line with output ids (e.g. `(%5) = add(%1, %2)`). The `S` option shortens operation names. You can then pick the ids of interest for `--trace_values`.

### The `--list` Flag

Use `--list` to print the compiled WaveGraph structure before execution:

```bash
# Print fused expression trees (how ops are grouped)
buck run @//mode/opt velox/experimental/torchwave/tests:executor_test -- \
  --gtest_filter="*.custom" --custom data/element_test --wave_only --list 1

# Print grid info (block counts, element counts per launch)
buck run @//mode/opt velox/experimental/torchwave/tests:executor_test -- \
  --gtest_filter="*.custom" --custom data/element_test --wave_only --list 2
```

## Understanding the Performance Report

The performance report is printed when `--trace 16` (`kTiming`) is set. It provides a structured breakdown of where time is spent during Wave execution. With `--num_repeats N`, the report is printed for every iteration so you can observe how the cost adjustment converges.

### Top-Level Throughput

```
=== Performance Report ===
E2E wall time: 113318 us (0.113 s)
Input throughput: 18.79 GB/s (2129.2 MB input)
Internal throughput: 187.03 GB/s (21193.8 MB total data)
```

- **E2E wall time**: Wall clock time of the `executeWave` call, from the first kernel launch through the final stream synchronization.
- **Input throughput**: Total user input tensor bytes divided by wall time. This is the rate at which input data is consumed. When this exceeds the host-to-device transfer bandwidth, the preproc is faster than the data can arrive.
- **Internal throughput**: Sum of all kernel input and output bytes (from the parameter buffer `Tensor` descriptors) divided by wall time. This measures the total memory bandwidth utilization of all fused kernels.

### Nodes

Nodes correspond to `CompiledNode` objects -- the layers of independent computation in the WaveGraph. They are listed in order of **decreasing wall time** (most expensive first).

```
Node 10: 25948 us
  step 0: 13207 us  blocks=422  in=1748572.6KB out=2215952.5KB  312.1 GB/s
    [gather=152 grid=0 alloc=0 fill=48 kernel=13007] standalone=182
```

Each node's wall time includes all its steps (kernel launches, standalone ops, and data transfers).

### Steps

Within a node, steps execute sequentially. Multi-kernel operations (like multi-block reductions) produce multiple steps separated by kernel boundaries. Steps are listed in **execution order**.

Each kernel step shows:

- **blocks**: Total thread blocks launched across all ops in this step.
- **in/out**: Input and output data volume from the kernel parameter tensors.
- **GB/s**: Effective kernel bandwidth (input+output bytes / kernel time).
- **[gather=... grid=... alloc=... fill=... kernel=...]**: Time breakdown in microseconds:
  - **gather**: Collecting `LaunchData` from `OpInvocation` bindings. Resolves formal value IDs to actual frame slots.
  - **grid**: Computing block assignments via `makeGrid`. Determines how many blocks each op gets, proportional to its cost. Zero on cache hits.
  - **alloc**: Allocating output tensors in the execution frame.
  - **fill**: Copying `Tensor` descriptors and scalar values into the pinned parameter buffer.
  - **kernel**: The kernel launch plus H2D transfer, GPU execution, and D2H transfer (if any). When `kTiming` is set, includes a stream sync for accurate measurement.
- **standalone=N**: Microseconds spent on standalone ops that overlap with this kernel step. A trailing `*` means standalone time exceeded kernel time (the step was standalone-bound, not GPU-bound).
- **noDtoH**: No device-to-host transfer was needed for this step (all outputs stay on device).

Standalone-only steps (no kernel launch) show a simpler format:

```
  step 2: 1 standalones  35444 us
```

### Thread Block Balance

For kernel steps, the balance report shows how evenly work is distributed across thread blocks:

```
    balance: util=71.5% sync=4.8% maxClk=18212443 blocks=422
```

- **util**: Utilization -- `totalClocks / (maxClocks * numBlocks) * 100`. 100% means all blocks finished at the same time. Lower values mean some blocks finished early and waited.
- **sync**: Percentage of total clock cycles spent in barrier synchronization (CG `opBarrier` calls).
- **maxClk**: The slowest thread block's clock cycle count. This determines the step's GPU time.
- **blocks**: Total thread blocks in this step.

### Per-Op Breakdown

Within each step, ops are listed in order of **decreasing max thread block clock** (the op with the highest max time first):

```
      op 33 (1 blocks, 42.0K): clk max/avg/min=18212443/18212443/18212443 barrier=0
      op 50 (379 blocks, 4.6M): clk max/avg/min=13108060/13088815/13062912 barrier=266094907
```

- **op N**: The op code (index into the kernel's operation table).
- **blocks**: Number of thread blocks assigned to this op.
- **42.0K / 4.6M**: Number of elements this op processes (K = thousands, M = millions).
- **clk max/avg/min**: Thread block clock cycles. max is the slowest block, min is the fastest. Large max/min spread indicates uneven data distribution or memory access patterns.
- **barrier**: Total barrier clock cycles across all blocks (from CG `opBarrier` calls).

When an op's max clock is much higher than other ops in the same step, it's the bottleneck. The `--auto_adjust_cost` flag dynamically increases the cost multiplier for such ops so they get more blocks on subsequent iterations.

### Top Consumers

```
=== Top Consumers ===
  Node 10: 25948 us (22.8%)
  Node 8: 4561 us (4.0%)
```

Lists the top 10 nodes by wall time, with percentage of total node time. Helps identify which compilation layers dominate execution.

### Top Standalones

```
Top standalones (% wall time):
  35443 us (31.3%): (%9907, ...) = torch.ops.fb.fused_datafm_merge_and_dedup_by_reference.default(...)
```

Lists the top 10 standalone ops by execution time, with percentage of E2E wall time. These are ops that run through PyTorch's standard dispatch rather than fused TorchWave kernels. They represent optimization opportunities -- either by fusing them into kernels or by optimizing the standalone implementation.

### Op Legend

```
=== Op Legend ===
  Op 0 cost=98.0 (%351) = to.dtype(%350, ...) ...
  Op 50 cost=2172.0 (%5538) = to.dtype(...)  ...
```

Maps each op code referenced in the per-op breakdown to its expression tree and cost. Sorted by op code. The cost is the per-element weight used by `makeGrid` to distribute thread blocks proportionally. Higher cost means more blocks per element. The expression tree shows what the op computes, with inner nodes abbreviated when the tree is deep.
