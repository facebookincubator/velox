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

IMG=velox-dev-ucx-test-1.19.0.img:latest
NAME=velox-dev-ucx-test

docker run -d --rm -it --gpus all --network=host --device /dev/infiniband/rdma_cm \
  --device=/dev/infiniband/uverbs0 --device=/dev/infiniband/uverbs1 \
  --device=/dev/infiniband/uverbs2 --device=/dev/infiniband/uverbs3 \
  --device=/dev/infiniband/uverbs4 --device=/dev/infiniband/uverbs5 \
  --device=/dev/infiniband/uverbs6 --device=/dev/infiniband/uverbs7 \
  --device=/dev/infiniband/uverbs8 --device=/dev/infiniband/uverbs9 \
  --cap-add=IPC_LOCK \
  --shm-size=1g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --pid host \
  --name ${NAME} \
  --entrypoint='' \
  -v /gpfs/zc2/data/tpch/tpch-sf1-parquet/one_brc_parquet:/data \
  ${IMG} \
  tail -f /dev/null
