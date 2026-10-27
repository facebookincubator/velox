# Explicit cuDF physical plans

The physical-plan path selects GPU implementations before Velox builds drivers.
It is opt-in while engines migrate from the general cuDF driver adapter.
All code in this layer lives under `experimental/cudf`; no Velox core hook is
required.

## Entry points

Register cuDF with `CudfConfig::enableDriverAdapter = false` before constructing
tasks. For configuration-file users, set `cudf.driver_adapter_enabled=false`.
The general adapter and physical planning are mutually exclusive.

Engines with ordinary Velox plans can use the compatibility rewriter:

```cpp
CudfPlanRewriter::Config config;
config.queryCtx = queryCtx; // The actual context used by the task.
auto physicalPlan = CudfPlanRewriter::rewrite(cpuPlan, config);
```

Engines that construct complete physical plans directly do not need the
rewriter. They can call `validateCudfPlan(plan, *queryCtx)` before creating the
task. This checks representation boundaries, source arity, node IDs, conversion
types, and terminal output. It is not an exhaustive expression or aggregation
state-layout validator.

The rewriter preserves supported GPU execution for aggregation, filter/project,
hash and nested-loop joins, sorting, limits, top-N, top-N row number, window,
grouping sets, distinct markers, unique IDs, single-row enforcement, and local
boundaries. GPU scans are recognized through the connector; exchanges are
recognized through their explicit transport. Unsupported ordinary operators
remain CPU operators when fallback is allowed, and the rewriter logs the reason
for each one.

The rewriter and the general driver adapter decide placement with the same
rules, in `GpuPlacement.h`. A Filter followed by a Project is decided as one
unit, because LocalPlanner fuses the two into one FilterProject operator. Core
plan nodes that stay on the CPU are rebuilt with their rewritten sources.

## Driver counts

`cudf.max_drivers` (`CudfConfig::maxDrivers`) limits the drivers of every
pipeline that contains a cuDF plan node. Pipelines without one keep the task's
driver count. When it is unset, both use the task's count. Nodes that must run
single-threaded keep doing so: `CudfLocalMergeNode` and `CudfLocalPartitionNode`
inherit `requiresSingleThread()` from their core base classes, and the other
cuDF plan nodes override it the same way as their core counterparts. The limit
applies only to physical plans, because the general driver adapter runs after
LocalPlanner has fixed the driver counts.

## Application-owned nodes

The engine selects the implementation of its custom node. This library does
not register automatic GPU replacement rules for application nodes.

Implement `CudfPlanNode`, or register a named `CudfPlanNodeDescriptor` for a
class owned by another library. Its contract declares:

- One input requirement per source: Velox, cuDF, or either representation.
- A fixed output representation, an `outputSource` whose representation is
  preserved, or neither for a terminal sink.
- A rebuilding method that installs new sources while preserving the operation,
  node ID, dynamic type, and output schema.

For example, a two-input implementation taking a device input and a host input,
and producing device vectors, declares:

```cpp
return CudfPlanNodeTraits{
    {InputRepresentation::kCudf, InputRepresentation::kVelox},
    VectorRepresentation::kCudf,
    {}};
```

A representation-preserving unary implementation declares:

```cpp
return CudfPlanNodeTraits{{InputRepresentation::kEither}, {}, 0};
```

The compatibility rewriter uses these contracts to insert `CudfFromVeloxNode`
and `CudfToVeloxNode` on individual edges. An unknown CPU node whose sources
change needs a rebuilding descriptor; the rewriter fails instead of retaining
stale sources. Operator construction stays in the application's ordinary
`Operator::PlanNodeTranslator`.

Register descriptors before planning. Duplicate registration names fail;
intrinsic node contracts take precedence, followed by registered descriptors
in registration order. Unregister by name for targeted test cleanup.

## Aggregation types

`CudfAggregationNode` is constructed from its source, output schema, aggregation
step, grouping keys, aggregate descriptions, intermediate types, and execution
settings. It neither accepts nor retains a CPU aggregation node.

Only the compatibility rewriter resolves CPU aggregate intermediate types.
cuDF aggregation execution reads the types declared by the physical node.
An aggregation rewrite can therefore construct new physical nodes and update
the surrounding schemas without first constructing modified CPU nodes.

This iteration preserves existing state encodings. In particular, declaring a
different type does not itself implement a new decimal SUM/AVG state encoding.
A compact-state rewrite must supply matching producer and consumer execution,
preserve SQL semantics, and either keep that state inside its supported region
or explicitly convert it at CPU boundaries. Vector representation and aggregate
state format are separate contracts.

## Temporary local boundaries and transports

Core LocalPlanner still owns local-exchange queues and pipeline splitting.
`CudfLocalPartitionNode` and `CudfLocalMergeNode` preserve that infrastructure.
A mandatory narrow adapter constructs their GPU operators after driver creation.
It performs no capability checks or CPU/GPU selection. The general
operator-adapter registry is not used by this path.

HTTP exchange consumes and produces host vectors. UCX exchange consumes and
produces cuDF vectors; the rewriter inserts the appropriate conversions.
Coordinator-facing Presto output remains HTTP. A UCX merge input is represented
as an ordinary UCX exchange followed by an explicit GPU sort, with the original
exchange ID retained for split delivery.

Filter/project fusion is performed by constructing one `CudfFilterProjectNode`
during rewriting. This does not introduce a general multi-node LocalPlanner
fusion hook.

## Current scope

This is an in-process physical-plan API and migration path. Coordinator wire
formats, complete physical-node serialization/trace replay, new compact
aggregation encodings, and removal of the remaining local-boundary adapter are
separate follow-ups. Existing legacy execution stays the default until an
engine explicitly opts in.

Tests should pass the intended plan explicitly, for example
`AssertQueryBuilder(rewriteToCudfPlan(plan))` using the test helper, or
`AssertQueryBuilder(physicalPlan)`. AssertQueryBuilder itself does not translate
plans.
