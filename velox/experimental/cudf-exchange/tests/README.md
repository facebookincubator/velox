# How to run the tests:

Run all tests:
CUDA_VISIBLE_DEVICES=7 UCX_TCP_CM_REUSEADDR=y ./_build/release/velox/experimental/cudf-exchange/tests/cudf_exchange_test -v=3 -logtostdout  

Run only a selected test:
CUDA_VISIBLE_DEVICES=7 UCX_TCP_CM_REUSEADDR=y ./_build/release/velox/experimental/cudf-exchange/tests/cudf_exchange_test -v=3 -logtostdout  --gtest_filter=CudfExchangeTest.basicTest


# Running the 1brc test

The 1brc server can serve multiple files, each of the files will be added as a split to the parquet table reader.
The 1brc client can connect to multiple upstream 1brc servers, each will provide data for computing the aggregation.

## Starting the 1brc test

Here's an example of how to start the server with 2 files (on Sally, for example):

```
UCX_RNDV_PIPELINE_ERROR_HANDLING=y CUDA_VISIBLE_DEVICES=6 \
  ./_build/release/velox/experimental/cudf-exchange/tests/1brc_server \
  -inputfiles=/gpfs/zc2/data/tpch/tpch-sf1-parquet/one_brc_parquet/measurements.parquet,/gpfs/zc2/data/tpch/tpch-sf1-parquet/one_brc_parquet/measurements.parquet \
  -v=3 --logtostdout -velox_cudf_memory_resource=async
```

And the corresponding command to start the client (on Monty, for example):

```
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/gpfs/zc2/u/sro/lib

UCX_RNDV_PIPELINE_ERROR_HANDLING=y CUDA_VISIBLE_DEVICES=7 \
./_build/release/velox/experimental/cudf-exchange/tests/1brc_client -v=3 --logtostdout -velox_cudf_memory_resource=async -nodes="http://sally:24356"
```

## Starting multiple servers

To start multiple servers, each server is given one file and the servers need to use different ports:

```
UCX_RNDV_PIPELINE_ERROR_HANDLING=y CUDA_VISIBLE_DEVICES=6 \
./_build/release/velox/experimental/cudf-exchange/tests/1brc_server \
-inputfiles=/gpfs/zc2/data/tpch/tpch-sf1-parquet/one_brc_parquet/measurements.parquet \
-v=3 --logtostdout -velox_cudf_memory_resource=async

UCX_RNDV_PIPELINE_ERROR_HANDLING=y CUDA_VISIBLE_DEVICES=7 \
./_build/release/velox/experimental/cudf-exchange/tests/1brc_server \
-inputfiles=/gpfs/zc2/data/tpch/tpch-sf1-parquet/one_brc_parquet/measurements.parquet \
-v=3 --logtostdout -velox_cudf_memory_resource=async -port 24360
```

To start the client, it needs to connect to both servers. Note that the client must correct the port number by -3:
```
UCX_RNDV_PIPELINE_ERROR_HANDLING=y CUDA_VISIBLE_DEVICES=7 \
./_build/release/velox/experimental/cudf-exchange/tests/1brc_client \
-v=3 --logtostdout -velox_cudf_memory_resource=async -nodes="http://sally:24356,http://sally:24357"
```
