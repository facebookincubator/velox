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

add_subdirectory(util)

if(${VELOX_BUILD_TESTING} AND NOT WIN32)
  # Parquet writer tests require arrow_testing library which is not
  # available in our vcpkg static build on Windows.
  add_subdirectory(tests)
endif()

velox_add_library(
  velox_dwio_arrow_parquet_writer_lib
  ArrowSchema.cpp
  ArrowSchemaInternal.cpp
  ColumnWriter.cpp
  Encoding.cpp
  Encryption.cpp
  EncryptionInternal.cpp
  Exception.cpp
  FileEncryptorInternal.cpp
  FileDecryptorInternal.cpp
  FileWriter.cpp
  Metadata.cpp
  PageIndex.cpp
  PathInternal.cpp
  Platform.cpp
  Properties.cpp
  Schema.cpp
  Statistics.cpp
  StringTruncation.cpp
  Types.cpp
  Writer.cpp
  HEADERS
  ArrowSchema.h
  ArrowSchemaInternal.h
  ColumnPage.h
  ColumnWriter.h
  Encoding.h
  Encryption.h
  EncryptionInternal.h
  Exception.h
  FileDecryptorInternal.h
  FileEncryptorInternal.h
  FileWriter.h
  Metadata.h
  PageIndex.h
  PathInternal.h
  Platform.h
  Properties.h
  Schema.h
  SchemaInternal.h
  Statistics.h
  StringTruncation.h
  ThriftInternal.h
  Types.h
  Writer.h
)

velox_link_libraries(
  velox_dwio_arrow_parquet_writer_lib
  velox_dwio_arrow_parquet_writer_util_lib
  velox_dwio_parquet_common
  velox_dwio_common
  velox_arrow_bridge
  arrow
  fmt::fmt
)

if(MSVC)
  # Parquet writer is built as a static library on Windows
  velox_compile_definitions(velox_dwio_arrow_parquet_writer_lib PRIVATE PARQUET_STATIC)
  # Prevent glog ERROR macro conflict
  velox_compile_definitions(velox_dwio_arrow_parquet_writer_lib PRIVATE GLOG_NO_ABBREVIATED_SEVERITIES)
endif()
