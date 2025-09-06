# Velox-cuDF

Velox-cuDF is a Velox extension module that uses the cuDF library to implement a GPU-accelerated backend for executing Velox plans. [cuDF](https://github.com/rapidsai/cudf) is an open source library for GPU data processing, and Velox-cuDF integrates with "[libcudf](https://github.com/rapidsai/cudf/tree/branch-25.10/cpp)", the CUDA C++ core of cuDF. libcudf uses [Arrow](https://arrow.apache.org)-compatible data layouts and includes single-node, single-GPU algorithms for data processing.

## How Velox and cuDF work together

Velox-cuDF uses the [DriverAdapter](https://github.com/facebookincubator/velox/blob/2a9c9043264a60c9a1b01324f8371c64bd095af9/velox/experimental/cudf/exec/ToCudf.cpp#L293) interface to rewrite query plans for GPU execution. Generally the cuDF DriverAdapter replaces operators one-to-one. For end-to-end GPU execution where cuDF replaces all of the Velox CPU operators, cuDF relies on Velox's [pipeline-based execution model](https://facebookincubator.github.io/velox/develop/task.html) to separate stages of execution, partition the work across drivers, and schedule concurrent work on the GPU.

For more information please refer to our blog: "[Extending Velox - GPU Acceleration with cuDF](https://velox-lib.io/blog/extending-velox-with-cudf)."

## Getting started with Velox-cuDF

cuDF supports Linux but not Windows or MacOS, and requires CUDA 12.0+ with a compatible NVIDIA driver. cuDF runs on NVIDIA GPUs with Volta architecture or better (Compute Capability >=7.0). Please refer to cuDF's [readme](https://github.com/rapidsai/cudf) and [developer guide](https://github.com/rapidsai/cudf/blob/branch-25.10/cpp/doxygen/developer_guide/DEVELOPER_GUIDE.md) for more information.

### Building Velox with cuDF

The cuDF backend is included in Velox builds when the [VELOX_ENABLE_CUDF](https://github.com/facebookincubator/velox/blob/43df50c4f24bcbfa96f5739c072ab0894d41cf4c/CMakeLists.txt#L455)  flag is set. The `adapters-cuda` service in Velox's [docker-compose.yml](https://github.com/facebookincubator/velox/blob/43df50c4f24bcbfa96f5739c072ab0894d41cf4c/docker-compose.yml#L69) is an excellent starting point for Velox builds with cuDF. 

1. Use `docker compose` to run an `adapters-cuda` image.
```
$ docker compose -f docker-compose.yml run -e NUM_THREADS=64 --rm -v "$(pwd):/velox" adapters-cuda /bin/bash
```
2. Once inside the image, build cuDF with the following flags:
```
$ CUDA_ARCHITECTURES="native" EXTRA_CMAKE_FLAGS="-DVELOX_ENABLE_ARROW=ON -DVELOX_ENABLE_PARQUET=ON -DVELOX_ENABLE_BENCHMARKS=ON -DVELOX_ENABLE_BENCHMARKS_BASIC=ON" make cudf
```
3. After cuDF is built, verify the build by running the unit tests.
```
$ cd _build/release
$ ctest -R cudf -V
```

### Testing Velox with cuDF

The Velox-cuDF tests in [experimental/cudf/tests](https://github.com/facebookincubator/velox/blob/main/velox/experimental/cudf/tests) include several types of tests:
* operator tests
* function tests
* fuzz tests (not yet implemented)

#### Operator tests

Many of tests for cuDF are "operator tests" which confirm correct execution of simple query plans. cuDF's operator tests use the cuDF `DriverAdapter` to modify the test plan with GPU operators before executing it. The operator tests for cuDF include both tests that assert successful GPU operator replacement, and tests that pass with CPU fallback. 

#### Function tests

Velox-cuDF also includes "function tests" which cover the behavior of shared functions that could be called in multiple operators. Velox-cuDF function tests assess the correctness of functions using one or more cuDF API calls to provide the output. [SubfieldFilterAstTest](https://github.com/facebookincubator/velox/blob/99a04b94eed42d1c35ae99101da3bf77b31652e8/velox/experimental/cudf/tests/SubfieldFilterAstTest.cpp#L158) includes several examples of function tests. Please note that unit tests for cuDF APIs are included in [cudf/cpp/tests](https://github.com/rapidsai/cudf/tree/branch-25.10/cpp/tests) rather than Velox.

#### Fuzz tests

Velox includes components for "fuzz testing" to ensure robustness of Velox operators. For instance, the [Join Fuzzer](https://github.com/facebookincubator/velox/blob/99a04b94eed42d1c35ae99101da3bf77b31652e8/velox/docs/develop/testing/join-fuzzer.rst) executes a random join type with random inputs and compares the Velox results with a reference query engine. Fuzz testing tools have been used for cuDF operator development, but fuzz testing for cuDF is yet integrated into Velox mainline.

### Benchmarking Velox with cuDF

Benchmarking Velox-cuDF will run as part of nightly automation workflows (outside of CI). Velox's cuDF backend can execute the hand-built query plans located at [TpchQueryBuilder](https://github.com/facebookincubator/velox/blob/43df50c4f24bcbfa96f5739c072ab0894d41cf4c/velox/exec/tests/utils/TpchQueryBuilder.cpp). Velox [PR 13695](https://github.com/facebookincubator/velox/pull/13695) includes changes to extend Velox benchmarks to the cuDF backend. Please note that the hand-built query plans require the data set to have floating-point types in place of the fixed-point types defined in the standard. Further development of Velox's TpchBenchmark could allow correct behavior with both fixed-point and floating-point types.

## Contributing

Velox-cuDF's development priorities are documented as Velox issues using the "[cuDF]" prefix. Please check out the [open issues](https://github.com/facebookincubator/velox/issues?q=is%3Aissue%20state%3Aopen%20%5BcuDF%5D) to learn more.

We would love to hear from you in Velox's Slack workspace, please see Velox discussion [11348](https://github.com/facebookincubator/velox/discussions/11348) for information on joining.
