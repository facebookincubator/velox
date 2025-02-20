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

velox_add_library(
  velox_dwio_parquet_crypto
  CryptoFactory.cpp
  CryptoFactory.h
  Exception.h
  KeyRetriever.h
  InMemoryKMSClient.cpp
  InMemoryKMSClient.h
  Utils.h
  EncryptionKey.h
  KeyMetadataAssembler.cpp
  KeyMetadataAssembler.h
  Cache.cpp
  Cache.h
  AesEncryption.cpp
  AesEncryption.h
  FileDecryptionProperties.cpp
  FileDecryptionProperties.h
  ColumnPath.cpp
  ColumnPath.h
  FileDecryptor.cpp
  FileDecryptor.h
  KMSClient.cpp
  KMSClient.h
)

velox_link_libraries(
  velox_dwio_parquet_crypto
  CURL::libcurl
  velox_dwio_parquet_thrift
)
