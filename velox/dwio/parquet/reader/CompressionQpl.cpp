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
#include <sys/time.h>

#include <glog/logging.h>
#include "CompressionQpl.h"
#include "velox/dwio/common/BufferUtil.h"

#define MAX_COMPRESS_CHUNK_SIZE 262144
#define MAX_DECOMPRESS_CHUNK_SIZE 262144
#define QPL_MAX_TRANS_SIZE 2097152

/*
* QPL compression/decompression need init a job buffer,
* the job buffer will be initialized only once.
* We set a lock to initialize job buffer, the default QPL execute path will be
hardware.
* If hardware path init failed, call back to software path init.
* If software path init failed, return error and exit.

* QPL_STS_INTL_HARDWARE_TIMEOUT means init hardware timeout.
* QPL_STS_INIT_HW_NOT_SUPPORTED means no IAA device.
* QPL_STS_INIT_WORK_QUEUES_NOT_AVAILABLE means Supported and enabled work queues
* are not found (May be due to lack of privileges e.g. lack of sudo on linux).
*/
void Qplcodec::Initjob() {
  if (jobInitialize_) {
    return;
  }

  qpl_status status;
  uint32_t size;

  status = qpl_get_job_size(executePath_, &size);
  if (status != QPL_STS_OK) {
    VELOX_FAIL("QPL::An error acquired during job size getting.");
  }

  jobBuffer_.resize(size);
  job_ = reinterpret_cast<qpl_job*>(jobBuffer_.data());

  status = qpl_init_job(executePath_, job_);

  if (status == QPL_STS_INTL_HARDWARE_TIMEOUT ||
      status == QPL_STS_INIT_HW_NOT_SUPPORTED ||
      status == QPL_STS_INIT_WORK_QUEUES_NOT_AVAILABLE) {
    executePath_ = qpl_path_software;
    status = qpl_get_job_size(executePath_, &size);
    if (status != QPL_STS_OK) {
      VELOX_FAIL("QPL::An error acquired during job size getting.");
    }

    jobBuffer_.resize(size);
    job_ = reinterpret_cast<qpl_job*>(jobBuffer_.data());
    status = qpl_init_job(executePath_, job_);
    if (status != QPL_STS_OK) {
      VELOX_FAIL("QPL::An error acquired during compression job initializing.");
    }
  }
  jobInitialize_ = true;
  return;
}

Qplcodec::~Qplcodec() {
  qpl_status status = qpl_fini_job(job_);
  if (status != QPL_STS_OK) {
    LOG(ERROR)
        << "QPL::An error acquired during compression job initializing finalization.";
  }
}
/*
 * The QPL can't compress/decompress the block over QPL_MAX_TRANS_SIZE,
 * We partition the input to several 256KB blocks.
 * So that QPL can compress/decompress with a high throughput.
 */
void Qplcodec::Compress(
    int64_t input_length,
    const uint8_t* input,
    int64_t output_buffer_length,
    uint8_t* output) {
  Initjob();

  int64_t out_size = output_buffer_length;
  job_->total_out = 0;
  job_->total_in = 0;

  job_->op = qpl_op_compress;
  job_->level = (qpl_compression_levels)compression_level_;
  job_->next_out_ptr = output;
  job_->next_in_ptr = const_cast<uint8_t*>(input);
  job_->flags =
      QPL_FLAG_FIRST | QPL_FLAG_DYNAMIC_HUFFMAN | QPL_FLAG_OMIT_VERIFY;

  int64_t current_chunk_size = MAX_COMPRESS_CHUNK_SIZE;
  uint32_t iteration_count = 0;
  int64_t source_bytes_left = input_length;

  while (source_bytes_left > 0) {
    if (current_chunk_size >= source_bytes_left) {
      job_->flags |= QPL_FLAG_LAST;
      current_chunk_size = source_bytes_left;
    }

    job_->available_in = current_chunk_size;
    job_->available_out = QPL_MAX_TRANS_SIZE < out_size - job_->total_out
        ? QPL_MAX_TRANS_SIZE
        : out_size - job_->total_out;

    qpl_status status = qpl_execute_job(job_);

    if (status != QPL_STS_OK) {
      VELOX_FAIL("QPL::Error while QPL compression occurred.");
    }

    source_bytes_left = input_length - job_->total_in;
    job_->flags &= ~QPL_FLAG_FIRST;
    iteration_count++;
  }
}

void Qplcodec::Decompress(
    int64_t input_length,
    const uint8_t* input,
    int64_t output_buffer_length,
    uint8_t* output) {
  if (output_buffer_length == 0) {
    VELOX_FAIL("QPL::Error, the Decompression size is 0.");
  }

  Initjob();

  int64_t out_size = output_buffer_length;
  job_->total_out = 0;
  job_->total_in = 0;
  job_->op = qpl_op_decompress;
  job_->next_out_ptr = output;
  job_->next_in_ptr = const_cast<uint8_t*>(input);
  job_->flags = QPL_FLAG_FIRST;

  int64_t current_chunk_size = MAX_DECOMPRESS_CHUNK_SIZE;
  uint32_t iteration_count = 0;
  int64_t source_bytes_left = input_length;

  while (source_bytes_left > 0) {
    if (current_chunk_size >= source_bytes_left) {
      job_->flags |= QPL_FLAG_LAST;
      current_chunk_size = source_bytes_left;
    }
    source_bytes_left -= current_chunk_size;
    job_->available_in = current_chunk_size;
    job_->available_out = QPL_MAX_TRANS_SIZE < out_size - job_->total_out
        ? QPL_MAX_TRANS_SIZE
        : out_size - job_->total_out;

    qpl_status status = qpl_execute_job(job_);
    if (status != QPL_STS_OK) {
      VELOX_FAIL("QPL::Error while decompression occurred.");
    }
    source_bytes_left = input_length - job_->total_in;
    job_->flags &= ~QPL_FLAG_FIRST;
    iteration_count++;
  }
}
