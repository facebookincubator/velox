# CuDF Exchange

The CuDF Exchange is a replacement for the inter-worker velox exchange that happens between tasks ending in a partitioned output operator and tasks having an exchange operator as source node. In Velox, this exchange is partially implemented; both the exchange server and the actual exchange source object implementations are outside the scope of Velox.

The velox cudf exchange implements all the necessary components to efficiently transfer cudf-vectors between tasks. At the core is a UCXX based transfer that directly copies the raw vector data from GPU memory to GPU memory.

## CMake Configuration

When building for the first time, this error occurs:

```
CMake Error at _build/release/_deps/geos-src/CMakeLists.txt:396 (add_custom_target):
  add_custom_target cannot create target "uninstall" because another target
  with the same name already exists.  The existing target is a custom target
  created in source directory
  "/gpfs/zc2/u/dnb/rapidsai-velox/_build/release/_deps/zstd-src/build/cmake/lib".
  See documentation for policy CMP0002 for more details.
```

Edit the file _build/release/_deps/geos-src/CMakeLists.txt, line 396 and change the target name from `uninstall` to `uninstall_geos`.

```
CUDA_ARCHITECTURES="80" EXTRA_CMAKE_FLAGS="-DVELOX_ENABLE_ARROW=ON -DVELOX_ENABLE_PARQUET=ON -DVELOX_ENABLE_BENCHMARKS=OFF -DVELOX_ENABLE_BENCHMARKS_BASIC=OFF" make cudf
```

In order to build the Cudf exchange tests, add `-DVELOX_BUILD_TESTING=ON` to EXTRA_CMAKE_FLAGS above. To build only the exchange and exchange tests, run:
`cmake --build _build/release -j --target=velox_cudf_queue_mgr_test`.

Then for example torun tests:
`_build/release/velox/experimental/cudf-exchange/tests/velox_cudf_queue_mgr_test`


To build the UCXX tests:

```
cmake --build _build/release -j --target=exchange_client_tst
cmake --build _build/release -j --target=exchange_srv_tst

_build/release/velox/experimental/cudf-exchange/tests/exchange_srv_tst -logtostdout -v=3 -port <PORT>
_build/release/velox/experimental/cudf-exchange/tests/exchange_client_tst -logtostdout -v=3 -port <PORT>
```


```
To build the velox test:

cmake --build _build/release -j --target=1brc_server
cmake --build _build/release -j --target=1brc_client



CUDA_VISIBLE_DEVICES=4  UCX_TLS=^ib _build/release/velox/experimental/cudf-exchange/tests/1brc_server -inputfile /gpfs/zc2/zoltan/1brc/measurements.parquet  -logtostdout -v=3 -velox_cudf_enabled=true -velox_cudf_table_scan=true -velox_cudf_debug=true  -velox_cudf_memory_resource=pool -port <PORT> -cuda_device=0 cudfChunkSizeGB=1

CUDA_VISIBLE_DEVICES=5 UCX_LOG_LEVEL=error UCX_TCP_KEEPINTVL=1ms UCX_KEEPALIVE_INTERVAL=1ms  UCX_TLS=^ib _build/release/velox/experimental/cudf-exchange/tests/1brc_client -logtostdout -v=3 -velox_cudf_enabled=true -velox_cudf_table_scan=true -velox_cudf_debug=true  -velox_cudf_memory_resource=pool -port <PORT> -cuda_device=0

```

There's a common target called `exchange_tests` that builds all of the above:
```
cmake --build _build/release -j --target=exchange_tests
```

## Code Formatting

We use the same code formating as velox.
The pre-commit hook needs to be activate once. Then formatting happens on every commit. These are the steps:

 -   Install the Python pre-commit package locally: pip install pre-commit
 -   Change to the velox directory where the velox repository is checked out, e.g. cd ~/velox
 -   Install the pre-commit hook in that directory: ~/.local/bin/pre-commit install
    Optionally, run the pre-commit hook "offline" to format some files, e.g.: ~/.local/bin/pre-commit run --files ./velox/experimental/cudf-exchange/*

When a file is ready to commit, the pre-commit hook runs and stops the commit whenever it has re-formatted the file, then the file can be re-viewed or directly added again and committed.
