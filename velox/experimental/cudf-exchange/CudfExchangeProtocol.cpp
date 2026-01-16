/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cuda.h"

#include "velox/experimental/cudf-exchange/CudfExchangeProtocol.h"

namespace facebook::velox::cudf_exchange {

uint32_t fnv1a_32(const std::string& s) {
  uint32_t hash = 0x811C9DC5u; // FNV offset basis
  for (unsigned char c : s) {
    hash ^= c;
    hash *= 0x01000193u; // FNV prime
  }
  return hash;
}

void cudaCheck(CUresult result) {
  if (result != CUDA_SUCCESS) {
    const char* err_msg;
    cuGetErrorName(result, &err_msg);
    std::cout << "Cuda error: " << err_msg << std::endl;
    exit(-1);
  }
}

} // namespace facebook::velox::cudf_exchange
