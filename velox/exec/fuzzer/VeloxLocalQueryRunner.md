# VeloxLocalQueryRunner

`VeloxLocalQueryRunner` is a reference query runner that serializes a Velox plan, sends it to a background LocalRunnerService, and gets the results back. It's similar to `PrestoQueryRunner` but uses Velox's local execution engine instead of Presto.

## Overview

The `VeloxLocalQueryRunner` works as follows:

1. It takes a `PlanNodePtr` and serializes it to JSON
2. It sends the serialized plan to a background LocalRunnerService via HTTP
3. It receives the results from LocalRunnerService and converts them back to Velox vectors

## Usage

### Starting the LocalRunnerService

Before using `VeloxLocalQueryRunner`, you need to start the LocalRunnerService:

```bash
cd fbsource/fbcode
buck run velox/runner/tests:local_runner_service
```

By default, the service listens on port 9090. You can change this using the `--port` flag.

### Creating a VeloxLocalQueryRunner

```cpp
#include "velox/exec/fuzzer/VeloxLocalQueryRunner.h"

// Create a memory pool
auto pool = memory::getDefaultScopedMemoryPool();

// Create a VeloxLocalQueryRunner
auto queryRunner = std::make_unique<VeloxLocalQueryRunner>(
    pool.get(), "http://127.0.0.1:9090", std::chrono::milliseconds(5000));
```

### Executing a Plan

```cpp
// Create a plan using PlanBuilder
auto plan = PlanBuilder()
    .values(someValues)
    .project({"c0 + 10", "c1"})
    .planNode();

// Execute the plan
auto result = queryRunner->executeAndReturnVector(plan);

// Check the result
if (result.first.has_value() && result.second == ReferenceQueryErrorCode::kSuccess) {
    // Process the results
    auto vectors = result.first.value();
    // ...
}
```

## Comparison with PrestoQueryRunner

While `PrestoQueryRunner` converts a plan to SQL and sends it to a Presto server, `VeloxLocalQueryRunner` serializes the plan directly and sends it to a LocalRunnerService. This has several advantages:

1. No need to convert the plan to SQL, which can be complex and error-prone
2. Support for all Velox features, not just those supported by Presto
3. Faster execution since there's no SQL parsing overhead
4. No need for a full Presto server, just a lightweight LocalRunnerService

## Implementation Details

The `VeloxLocalQueryRunner` uses the following components:

- `core::PlanNode::serialize()` to serialize the plan to JSON
- `cpr` library for HTTP communication with the LocalRunnerService
- `folly::parseJson` to parse the JSON response
- Custom logic to convert the JSON results back to Velox vectors

## Example

See `velox/exec/fuzzer/tests/VeloxLocalQueryRunnerTest.cpp` for a complete example of how to use `VeloxLocalQueryRunner`.
