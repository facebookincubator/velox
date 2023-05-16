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

#pragma once

#include <iostream>
#include <vector>

#include "velox/dwio/common/BufferUtil.h"
#include "velox/dwio/common/QplJobPool.h"
#include "velox/dwio/common/TypeUtil.h"
#include "velox/dwio/parquet/thrift/ParquetThriftTypes.h"
#include "velox/dwio/parquet/reader/ParquetTypeWithId.h"
#include "velox/vector/BaseVector.h"

#include <unistd.h>       // for syscall()
#include <sys/syscall.h> 

#ifdef VELOX_ENABLE_QPL
namespace facebook::velox::parquet {

// This class will be used for dictionary Ids or other data that is
// RLE/BP encoded and inflate compressed data

class DeflateRleBpDecoder {
 public:
    DeflateRleBpDecoder(
        const char* FOLLY_NONNULL pageData,
        thrift::PageHeader pageHeader,
        ParquetTypeWithIdPtr type,
        memory::MemoryPool& pool)
        : pageData_(pageData),
          pageHeader_(pageHeader),
          type_(type),
          pool_(pool),
          decoded_values_(NULL),
          cur_visitor_pos(0) {}


      template <bool hasNulls, typename Visitor>
  uint32_t decodeWithVisitor(const uint64_t* FOLLY_NULLABLE nulls, Visitor visitor, uint32_t visitBase) {
    if (useQplPath<hasNulls>(visitor)) {
        constexpr bool hasFilter =
            !std::is_same_v<typename Visitor::FilterType, common::AlwaysTrue>;
        constexpr bool hasHook =
            !std::is_same_v<typename Visitor::HookType, dwio::common::NoHook>;
        auto rows = visitor.rows();
        auto numRows = visitor.numRows();
        auto rowsAsRange = folly::Range<const int32_t*>(rows, numRows);

        return rleDecode<hasFilter, hasHook, false>(rowsAsRange, nullptr, visitor, visitBase);
    } else {
        std::cout << "not support yet!" << std::endl;
    }
    return 0;
  }

    template <bool hasNulls, typename Visitor>
  void filterWithVisitor(const uint64_t* FOLLY_NULLABLE nulls, Visitor visitor) {
    if (useQplPath<hasNulls>(visitor)) {
        constexpr bool hasFilter =
            !std::is_same_v<typename Visitor::FilterType, common::AlwaysTrue>;
        constexpr bool hasHook =
            !std::is_same_v<typename Visitor::HookType, dwio::common::NoHook>;
        auto rows = visitor.rows();
        auto numRows = visitor.numRows();
        auto rowsAsRange = folly::Range<const int32_t*>(rows, numRows);

        filterScan<hasFilter, hasHook, false>(rowsAsRange, nullptr, visitor);
    } else {
        std::cout << "not support yet!" << std::endl;
    }
    return;
  }
  
  void skip(uint64_t numValues) {
    return;
  }

 protected:
    const char* FOLLY_NULLABLE pageData_{nullptr};    // uncompressed page data
    thrift::PageHeader pageHeader_;

    ParquetTypeWithIdPtr type_;

    BufferPtr decoded_values_;

    memory::MemoryPool& pool_;

    uint32_t cur_visitor_pos;

 private:
  template <bool hasNulls, typename ColumnVisitor>
  bool useQplPath(const ColumnVisitor& visitor) {
    if (hasNulls) {
        return false;
    }

    // TODO: check data is uint16 uint32

    // auto numRows = visitor.numRows();
    // auto values = visitor.rawValues(numRows);
    // using TValues = typename std::remove_reference<decltype(values[0])>::type;
    // using TIndex = typename std::make_signed_t<typename dwio::common::make_index<TValues>::type>;

    // if (sizeof(TIndex) <= sizeof(uint32_t)) {
    //   return true;
    // }

    // TODO: check filter type
    return true;
  }


  template <bool hasFilter, bool hasHook, bool scatter,  typename ColumnVisitor>
  uint32_t rleDecode(folly::Range<const int32_t*> nonNullRows,
      const int32_t* FOLLY_NULLABLE scatterRows,
      ColumnVisitor& visitor,
      uint32_t visitBase) {

    // Step1. uncompress + scan with mask data pate to get output vector
    auto numRows = visitor.numRows();
    auto numAllRows = visitor.numRows();
    auto values = visitor.rawValues(numRows);
    using TValues = typename std::remove_reference<decltype(values[0])>::type;
    using TIndex = typename std::make_signed_t<typename dwio::common::make_index<TValues>::type>;
    uint8_t* out_ptr;

    if (decoded_values_ == NULL) {
      if (numRows < pageHeader_.data_page_header.num_values) {
        facebook::velox::dwio::common::ensureCapacity<TValues>(decoded_values_, pageHeader_.data_page_header.num_values, &pool_);
        decoded_values_->setSize(pageHeader_.data_page_header.num_values * sizeof(TValues));
        out_ptr = decoded_values_->asMutable<uint8_t>();
        // std::cout << "numRows: " << numRows << ", pageHeader_.data_page_header.num_values" << pageHeader_.data_page_header.num_values << std::endl;
      } else {
        out_ptr = reinterpret_cast<uint8_t*>(values);
      }
      memset(out_ptr, 0, pageHeader_.data_page_header.num_values * sizeof(TValues));
    } else {
      return 0;
    }

    dwio::common::QplJobHWPool& qpl_job_pool = dwio::common::QplJobHWPool::GetInstance();
    uint32_t job_id = 0;

    qpl_job* job = qpl_job_pool.AcquireJob(job_id);

    job->op = qpl_op_extract;
    job->next_in_ptr = reinterpret_cast<uint8_t*>(const_cast<char*>(pageData_));
    job->available_in = pageHeader_.compressed_page_size;
    job->parser = qpl_p_parquet_rle;
    job->param_low = 0;
    job->param_high = pageHeader_.data_page_header.num_values;
    job->out_bit_width = qpl_ow_32;
    job->next_out_ptr = out_ptr;
    job->available_out = static_cast<uint32_t>(pageHeader_.data_page_header.num_values * sizeof(TIndex));
    job->num_input_elements = pageHeader_.data_page_header.num_values;
    job->flags   = QPL_FLAG_DECOMPRESS_ENABLE | QPL_FLAG_FIRST | QPL_FLAG_LAST;
    job->numa_id = 1;
    // std::cout << "pageHeader_.compressed_page_size: " << pageHeader_.compressed_page_size << std::endl;

    auto status = qpl_submit_job(job);
    uint32_t check_time = 0;
    while (status == QPL_STS_QUEUES_ARE_BUSY_ERR && check_time < UINT32_MAX - 1) {
      _umwait(1, __rdtsc() + 7000);
      check_time++;
      status = qpl_submit_job(job);
      // std::cout << "submit deflate+prle decode job error : check_time " << check_time << ", status: " << (int)status << std::endl;
    }
    VELOX_DCHECK(status == QPL_STS_OK, "Execturion of QPL Job failed, status {}, job_id {}", status, (int)job_id);
    if (status != QPL_STS_OK) {
    throw std::runtime_error(
        "submition of QPL Job failed, status:" + std::to_string(status) + " job_id: " + std::to_string(job_id));
    }

    // int id = syscall(SYS_gettid);
    // std::cout << "submit decode job: " << (int)job_id << ", sys id: " << id << std::endl;
    return job_id;
  }

  template <bool hasFilter, bool hasHook, bool scatter,  typename ColumnVisitor>
  void filterScan(folly::Range<const int32_t*> nonNullRows,
      const int32_t* FOLLY_NULLABLE scatterRows,
      ColumnVisitor& visitor) {
      auto numRows = visitor.numRows();
      auto numAllRows = visitor.numRows();
      auto values = visitor.rawValues(numRows);
      using TValues = typename std::remove_reference<decltype(values[0])>::type;
      using TIndex = typename std::make_signed_t<typename dwio::common::make_index<TValues>::type>;
      //  filter dictionary to get output
      int32_t numValues = 0;
      auto filterHits = hasFilter ? visitor.outputRows(numRows) : nullptr;
      TValues* input;
      if (decoded_values_ != NULL) {
        // std::cout << "decode is no null" << std::endl;
        input = decoded_values_->asMutable<TValues>() + cur_visitor_pos;
        cur_visitor_pos += numAllRows;
        memcpy((void*)values, input, numAllRows * sizeof(TValues));
      }

  uint32_t check_time = 0;
  while (check_time < 100) {
    try {
        visitor.template processRun<hasFilter, hasHook, scatter>(
            values,
            numRows,
            nullptr,
            filterHits,
            values,
            numValues);
        if (visitor.atEnd()) {
          visitor.setNumValues(hasFilter ? numValues : numAllRows);
          return;
        }  
        return;     
    } catch (const std::runtime_error& e) {
      check_time++;
      // We cannot throw an exception from the destructor. Warn instead.
        if (decoded_values_ != NULL) {
          // std::cout << "decode is no null" << std::endl;
          input = decoded_values_->asMutable<TValues>() + cur_visitor_pos - numAllRows;
          cur_visitor_pos += numAllRows;
          memcpy((void*)values, input, numAllRows * sizeof(TValues));
        }
        std::cout << "check time: " << check_time << folly::to<std::string>(e.what()) << std::endl;
    }
  }      
  }  

};

}  // namespace facebook::velox::parquet::reader
#endif
