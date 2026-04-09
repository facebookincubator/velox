/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#pragma once

/*
 * Some conda / minimal CUDA stacks ship a <cufile.h> without the cuFile 1.7+
 * batch + stream + async declarations. KvikIO 26.06 still uses those symbols at
 * compile time (decltype in cufile.hpp). Layout and prototypes are aligned with
 * NVIDIA libcufile linux-sbsa 1.13.1 public headers.
 *
 * Include this only after <cufile.h> when a CMake probe shows the batch API is
 * missing. Requires CUfileHandle_t, CUfileError_t, and CUstream from cuda.h /
 * the partial cufile.h.
 */

#include <cuda.h>
#include <stddef.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum CUfileOpcode {
  CUFILE_READ = 0,
  CUFILE_WRITE
} CUfileOpcode_t;

typedef enum CUFILEStatus_enum {
  CUFILE_WAITING   = 0x000001,
  CUFILE_PENDING   = 0x000002,
  CUFILE_INVALID   = 0x000004,
  CUFILE_CANCELED  = 0x000008,
  CUFILE_COMPLETE  = 0x0000010,
  CUFILE_TIMEOUT   = 0x0000020,
  CUFILE_FAILED    = 0x0000040
} CUfileStatus_t;

typedef enum cufileBatchMode {
  CUFILE_BATCH = 1,
} CUfileBatchMode_t;

typedef struct CUfileIOParams {
  CUfileBatchMode_t mode;
  union {
    struct {
      void* devPtr_base;
      off_t file_offset;
      off_t devPtr_offset;
      size_t size;
    } batch;
  } u;
  CUfileHandle_t fh;
  CUfileOpcode_t opcode;
  void* cookie;
} CUfileIOParams_t;

typedef struct CUfileIOEvents {
  void* cookie;
  CUfileStatus_t status;
  size_t ret;
} CUfileIOEvents_t;

typedef void* CUfileBatchHandle_t;

CUfileError_t cuFileBatchIOSetUp(CUfileBatchHandle_t* batch_idp, unsigned nr);
CUfileError_t cuFileBatchIOSubmit(CUfileBatchHandle_t batch_idp,
                                  unsigned nr,
                                  CUfileIOParams_t* iocbp,
                                  unsigned int flags);
CUfileError_t cuFileBatchIOGetStatus(CUfileBatchHandle_t batch_idp,
                                     unsigned min_nr,
                                     unsigned* nr,
                                     CUfileIOEvents_t* iocbp,
                                     struct timespec* timeout);
CUfileError_t cuFileBatchIOCancel(CUfileBatchHandle_t batch_idp);
void cuFileBatchIODestroy(CUfileBatchHandle_t batch_idp);

CUfileError_t cuFileReadAsync(CUfileHandle_t fh,
                              void* bufPtr_base,
                              size_t* size_p,
                              off_t* file_offset_p,
                              off_t* bufPtr_offset_p,
                              ssize_t* bytes_read_p,
                              CUstream stream);

CUfileError_t cuFileWriteAsync(CUfileHandle_t fh,
                               void* bufPtr_base,
                               size_t* size_p,
                               off_t* file_offset_p,
                               off_t* bufPtr_offset_p,
                               ssize_t* bytes_written_p,
                               CUstream stream);

CUfileError_t cuFileStreamRegister(CUstream stream, unsigned flags);
CUfileError_t cuFileStreamDeregister(CUstream stream);

#ifdef __cplusplus
}
#endif
