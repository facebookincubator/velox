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
UCX_TLS=tcp,cuda_copy,cuda_ipc UCX_MAX_RNDV_RAILS=1 UCX_LOG_LEVEL=info UCX_PROTO_INFO=n UCX_RNDV_PIPELINE_ERROR_HANDLING=y CUDA_VISIBLE_DEVICES=7 /workspace/velox/_build/release/velox/experimental/cudf-exchange/tests/1brc_client -v=3 --logtostdout -velox_cudf_memory_resource=async -nodes="http://127.0.0.1:50000"
