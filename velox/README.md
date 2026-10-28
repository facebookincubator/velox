# Velox

Velox is a composable execution engine distributed as an open source C++
library. It provides reusable, extensible, and high-performance data
processing components — types, columnar vectors, vectorized expression
evaluation, operators, connectors, and file readers/writers — used to build
data management systems for batch, interactive, and ML workloads. This folder
holds the Velox engine source, its connectors, functions, benchmarks,
examples, Python bindings, and docs.

This folder is a collaboration between several teams, including `velox`,
`spark_gluten`, `dwios`, `velox_wave`, and `presto_cpp` (see the `oncall()`
in each sub-package's `BUCK` file for the owner of that part).
