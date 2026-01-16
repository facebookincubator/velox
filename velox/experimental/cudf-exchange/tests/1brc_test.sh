#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
SERVER_LISTEN_PORT=20000
CLIENT_LISTEN_PORT=40000
BUILD_DIR=/gpfs/zc2/u/sro/git/zrl_velox/velox/_build/release/velox

COUNTER=1

while [ $COUNTER -le 50 ]; do

  echo "########## Running test with $COUNTER"

  CUDA_VISIBLE_DEVICES=4 UCX_TLS=^ib $BUILD_DIR/experimental/cudf-exchange/tests/1brc_server -inputfile /gpfs/zc2/zoltan/1brc/measurements.parquet -logtostdout -v=3 -velox_cudf_enabled=true -velox_cudf_table_scan=true -velox_cudf_debug=true -velox_cudf_memory_resource=pool -port $SERVER_LISTEN_PORT -cuda_device=0 cudfChunkSizeGB=1 &>/dev/null &

  sleep_time=$((RANDOM % 5))
  echo "Sleeping $sleep_time"
  sleep $sleep_time

  CUDA_VISIBLE_DEVICES=5 UCX_LOG_LEVEL=error UCX_TCP_KEEPINTVL=1ms UCX_KEEPALIVE_INTERVAL=1ms UCX_TLS=^ib $BUILD_DIR/experimental/cudf-exchange/tests/1brc_client -logtostdout -v=3 -velox_cudf_enabled=true -velox_cudf_table_scan=true -velox_cudf_debug=true -velox_cudf_memory_resource=pool -port $SERVER_LISTEN_PORT -cuda_device=0 -srv_port $CLIENT_LISTEN_PORT

  RETURN_CODE=$?

  echo "RETURN_CODE is $RETURN_CODE"

  if [ $RETURN_CODE -ne 0 ]; then
    echo "FAILED !"
    exit 1
  else
    COUNTER=$((COUNTER + 1))
    sleep 5
  fi

done

echo "SUCCEEDED"
