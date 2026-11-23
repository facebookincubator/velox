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

#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>
#include <cuda/stream_ref>
#include <cuda_runtime_api.h>

#include <gmock/gmock.h>

#include <array>
#include <memory>
#include <vector>

using namespace facebook::velox;
using namespace facebook::velox::cudf_velox;
using namespace facebook::velox::test;

namespace {

class TestCudaStream {
 public:
  TestCudaStream() {
    auto status = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
    VELOX_CHECK(
        status == cudaSuccess,
        "cudaStreamCreateWithFlags failed: {} ({})",
        cudaGetErrorString(status),
        static_cast<int>(status));
  }

  ~TestCudaStream() {
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
    }
  }

  cuda::stream_ref view() const {
    return cuda::stream_ref{stream_};
  }

  cudaStream_t value() const {
    return stream_;
  }

 private:
  cudaStream_t stream_{nullptr};
};

struct RecordingAsyncResourceState {
  std::vector<cudaStream_t> deallocationStreams;
};

class RecordingAsyncDeviceResource {
 public:
  void* allocate(
      cuda::stream_ref stream,
      std::size_t bytes,
      std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) {
    return asyncUpstream_.allocate(stream, bytes, alignment);
  }

  void deallocate(
      cuda::stream_ref stream,
      void* ptr,
      std::size_t bytes,
      std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept {
    state_->deallocationStreams.push_back(stream.get());
    asyncUpstream_.deallocate(stream, ptr, bytes, alignment);
  }

  void* allocate_sync(
      std::size_t bytes,
      std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) {
    return asyncUpstream_.allocate_sync(bytes, alignment);
  }

  void deallocate_sync(
      void* ptr,
      std::size_t bytes,
      std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept {
    asyncUpstream_.deallocate_sync(ptr, bytes, alignment);
  }

  void reset() {
    state_->deallocationStreams.clear();
  }

  // Streams of every deallocation since the last reset(), in order.
  const std::vector<cudaStream_t>& deallocationStreams() const {
    return state_->deallocationStreams;
  }

  bool operator==(const RecordingAsyncDeviceResource& other) const noexcept {
    return state_ == other.state_;
  }

 private:
  std::shared_ptr<RecordingAsyncResourceState> state_{
      std::make_shared<RecordingAsyncResourceState>()};
  rmm::mr::cuda_async_memory_resource asyncUpstream_;
};

void get_property(
    const RecordingAsyncDeviceResource&,
    cuda::mr::device_accessible) noexcept {}

std::unique_ptr<cudf::table> makeTable(
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  std::array<int32_t, 4> values{{1, 2, 3, 4}};
  rmm::device_buffer data(values.size() * sizeof(int32_t), stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      data.data(),
      values.data(),
      values.size() * sizeof(int32_t),
      cudaMemcpyHostToDevice,
      stream.get()));

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(
      std::make_unique<cudf::column>(
          cudf::data_type{cudf::type_id::INT32},
          static_cast<cudf::size_type>(values.size()),
          std::move(data),
          cudf::create_null_mask(
              0,
              cudf::mask_state::UNALLOCATED,
              stream,
              cudf::memory_resources{mr, mr}),
          0));
  return std::make_unique<cudf::table>(std::move(columns));
}

// Number of device buffers in a table from makeNestedTable(): data and null
// mask for 'c0' and for the struct's child, plus the struct's null mask. A
// struct column has no data buffer.
constexpr std::size_t kNumNestedTableBuffers{5};

// Makes a table of ROW(c0 INTEGER, c1 ROW(c2 INTEGER)) whose columns all own an
// allocated null mask.
std::unique_ptr<cudf::table> makeNestedTable(
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  constexpr cudf::size_type kNumRows{4};
  auto makeIntegers = [&]() {
    return cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32},
        kNumRows,
        cudf::mask_state::ALL_VALID,
        stream,
        mr);
  };

  std::vector<std::unique_ptr<cudf::column>> fields;
  fields.push_back(makeIntegers());
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(makeIntegers());
  columns.push_back(
      cudf::make_structs_column(
          kNumRows,
          std::move(fields),
          0,
          cudf::create_null_mask(
              kNumRows,
              cudf::mask_state::ALL_VALID,
              stream,
              cudf::memory_resources{mr, mr}),
          stream,
          mr));
  return std::make_unique<cudf::table>(std::move(columns));
}

RowTypePtr nestedTableType() {
  return ROW({{"c0", INTEGER()}, {"c1", ROW("c2", INTEGER())}});
}

std::unique_ptr<cudf::packed_table> makePackedTable(
    cuda::stream_ref stream,
    RecordingAsyncDeviceResource& resource) {
  auto table = makeTable(stream, cudf::get_current_device_resource_ref());
  auto packedColumns = cudf::pack(
      table->view(),
      stream,
      rmm::to_device_async_resource_ref_checked(&resource));
  // CudfVector does not join producer streams. Synchronize the packing stream
  // before handing the packed table to CudfVector.
  stream.sync();
  auto tableView = cudf::unpack(packedColumns);
  return std::make_unique<cudf::packed_table>(
      cudf::packed_table{tableView, std::move(packedColumns)});
}

class CudfVectorTest : public ::testing::Test, public VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }
};

TEST_F(CudfVectorTest, retainedSizeReportsDeviceStorage) {
  TestCudaStream stream;
  auto table =
      makeTable(stream.view(), cudf::get_current_device_resource_ref());
  stream.view().sync();

  CudfVector vector(
      pool_.get(),
      ROW({"c0"}, {INTEGER()}),
      table->num_rows(),
      std::move(table),
      stream.view());

  EXPECT_EQ(vector.estimateFlatSize(), 4 * sizeof(int32_t));
  EXPECT_EQ(vector.retainedSize(), vector.estimateFlatSize());
}

TEST_F(CudfVectorTest, rebindOwnedTableDeallocationStream) {
  TestCudaStream allocationStream;
  TestCudaStream targetStream;
  RecordingAsyncDeviceResource resource;

  auto table = makeNestedTable(
      allocationStream.view(),
      rmm::to_device_async_resource_ref_checked(&resource));
  allocationStream.view().sync();

  auto vector = std::make_shared<CudfVector>(
      pool_.get(),
      nestedTableType(),
      table->num_rows(),
      std::move(table),
      allocationStream.view());
  resource.reset();

  ASSERT_TRUE(vector->rebindStream(targetStream.view()));
  vector.reset();

  EXPECT_THAT(
      resource.deallocationStreams(),
      testing::AllOf(
          testing::SizeIs(kNumNestedTableBuffers),
          testing::Each(targetStream.value())));
}

TEST_F(CudfVectorTest, ownedTableDestructionUsesConsumerStream) {
  TestCudaStream allocationStream;
  TestCudaStream consumerStream;
  RecordingAsyncDeviceResource resource;

  auto table = makeNestedTable(
      allocationStream.view(),
      rmm::to_device_async_resource_ref_checked(&resource));
  allocationStream.view().sync();

  // The table's buffers were allocated on allocationStream, but downstream
  // operators read the vector on consumerStream. Dropping the vector must
  // order every free, including null masks and children, after that work.
  auto vector = std::make_shared<CudfVector>(
      pool_.get(),
      nestedTableType(),
      table->num_rows(),
      std::move(table),
      consumerStream.view());
  resource.reset();

  vector.reset();

  EXPECT_THAT(
      resource.deallocationStreams(),
      testing::AllOf(
          testing::SizeIs(kNumNestedTableBuffers),
          testing::Each(consumerStream.value())));
}

TEST_F(CudfVectorTest, packedTableDestructionUsesConsumerStream) {
  TestCudaStream allocationStream;
  TestCudaStream consumerStream;
  RecordingAsyncDeviceResource resource;
  auto packedTable = makePackedTable(allocationStream.view(), resource);

  // Model the intra-node UCX path: the producer allocated the packed buffer on
  // its own stream, and the consumer wraps it with a different stream on which
  // downstream operators read it. Dropping the vector must order the free after
  // that work.
  auto vector = std::make_shared<CudfVector>(
      pool_.get(),
      ROW({"c0"}, {INTEGER()}),
      packedTable->table.num_rows(),
      std::move(packedTable),
      consumerStream.view());
  resource.reset();

  vector.reset();

  EXPECT_THAT(
      resource.deallocationStreams(),
      testing::ElementsAre(consumerStream.value()));
}

TEST_F(CudfVectorTest, rebindPackedTableDeallocationStream) {
  TestCudaStream allocationStream;
  TestCudaStream targetStream;
  RecordingAsyncDeviceResource resource;
  auto packedTable = makePackedTable(allocationStream.view(), resource);

  // Rebinding after construction must move the packed buffer's deallocation
  // to the new consumer stream.
  auto vector = std::make_shared<CudfVector>(
      pool_.get(),
      ROW({"c0"}, {INTEGER()}),
      packedTable->table.num_rows(),
      std::move(packedTable),
      allocationStream.view());
  resource.reset();

  ASSERT_TRUE(vector->rebindStream(targetStream.view()));
  EXPECT_EQ(vector->stream().get(), targetStream.value());
  vector.reset();

  EXPECT_THAT(
      resource.deallocationStreams(),
      testing::ElementsAre(targetStream.value()));
}

TEST_F(CudfVectorTest, orderDeallocationsRejectsReleasedInput) {
  TestCudaStream allocationStream;
  TestCudaStream targetStream;
  RecordingAsyncDeviceResource resource;
  auto mr = rmm::to_device_async_resource_ref_checked(&resource);

  auto makeVector = [&]() {
    auto table = makeNestedTable(allocationStream.view(), mr);
    return std::make_shared<CudfVector>(
        pool_.get(),
        nestedTableType(),
        table->num_rows(),
        std::move(table),
        allocationStream.view());
  };
  auto released = makeVector();
  auto retained = makeVector();
  allocationStream.view().sync();
  auto releasedTable = released->release();
  resource.reset();

  // The released input comes first, so the check must not skip rebinding the
  // inputs after it.
  std::vector<CudfVectorPtr> inputs{released, retained};
  VELOX_ASSERT_THROW(
      orderCudfVectorDeallocationsAfterStream(inputs, targetStream.view()),
      "Cannot rebind a released CudfVector");

  inputs.clear();
  retained.reset();

  EXPECT_THAT(
      resource.deallocationStreams(),
      testing::AllOf(
          testing::SizeIs(kNumNestedTableBuffers),
          testing::Each(targetStream.value())));
}

TEST_F(CudfVectorTest, packedTableReleaseUsesMaterializationStream) {
  TestCudaStream allocationStream;
  TestCudaStream targetStream;
  RecordingAsyncDeviceResource resource;
  auto packedTable = makePackedTable(allocationStream.view(), resource);

  // CudfVector::release materializes from a table_view that points into the
  // packed buffer. The packed buffer must be freed on the materialization
  // stream, not the original packing stream.
  CudfVector vector(
      pool_.get(),
      ROW({"c0"}, {INTEGER()}),
      packedTable->table.num_rows(),
      std::move(packedTable),
      targetStream.view());
  resource.reset();

  auto materialized = vector.release();

  EXPECT_THAT(
      resource.deallocationStreams(),
      testing::ElementsAre(targetStream.value()));
  targetStream.view().sync();
  EXPECT_EQ(materialized->num_columns(), 1);
  EXPECT_EQ(materialized->num_rows(), 4);
}

TEST_F(CudfVectorTest, ownedTableRejectsUseAfterRelease) {
  TestCudaStream stream;
  auto table =
      makeTable(stream.view(), cudf::get_current_device_resource_ref());
  stream.view().sync();
  CudfVector vector(
      pool_.get(),
      ROW({"c0"}, {INTEGER()}),
      table->num_rows(),
      std::move(table),
      stream.view());

  auto released = vector.release();

  VELOX_ASSERT_THROW(vector.release(), "CudfVector was already released");
  VELOX_ASSERT_THROW(vector.getTableView(), "CudfVector was already released");
  // The vector's own stream would take the equal-stream early return.
  EXPECT_FALSE(vector.rebindStream(stream.view()));
}

TEST_F(CudfVectorTest, packedTableRejectsUseAfterRelease) {
  TestCudaStream stream;
  RecordingAsyncDeviceResource resource;
  auto packedTable = makePackedTable(stream.view(), resource);
  CudfVector vector(
      pool_.get(),
      ROW({"c0"}, {INTEGER()}),
      packedTable->table.num_rows(),
      std::move(packedTable),
      stream.view());

  auto materialized = vector.release();

  VELOX_ASSERT_THROW(vector.release(), "CudfVector was already released");
  VELOX_ASSERT_THROW(vector.getTableView(), "CudfVector was already released");
  EXPECT_FALSE(vector.rebindStream(stream.view()));
}

} // namespace
