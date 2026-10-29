# Copyright (c) Meta Platforms, Inc. and its affiliates.
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

# The vector search join plan node, which builds without FAISS, holds and
# validates these settings.
velox_add_library(
  velox_vector_index_settings
  LocalVectorIndexOptions.cpp
  VectorIndexMetric.cpp
  VectorIndexType.cpp
  HEADERS
  LocalVectorIndexOptions.h
  VectorIndexMetric.h
  VectorIndexType.h
)

velox_link_libraries(
  velox_vector_index_settings
  velox_enum_declare
  velox_enum_define
  velox_exception
  Folly::folly
)

if(VELOX_ENABLE_FAISS)
  find_package(OpenMP REQUIRED)

  velox_add_library(
    velox_vector_index
    Embeddings.cpp
    LocalVectorIndex.cpp
    HEADERS
    CallbackWriter.h
    Embeddings.h
    LocalVectorIndex.h
    SingleThreadedFaiss.h
  )

  velox_link_libraries(
    velox_vector_index
    velox_vector_index_settings
    velox_enum_declare
    velox_enum_define
    velox_exception
    Folly::folly
    FAISS::faiss
    OpenMP::OpenMP_CXX
  )

  if(${VELOX_BUILD_TESTING})
    add_subdirectory(tests)
  endif()
endif()
