#include "velox4j/iterator/BlockingQueue.h"
#include <gtest/gtest.h>
#include <velox/vector/tests/utils/VectorTestBase.h>
#include "velox4j/test/Init.h"

namespace velox4j {
using namespace facebook::velox;
using namespace facebook::velox::exec::test;

class BlockingQueueTest : public testing::Test, public test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    testingEnsureInitializedForSpark();
  }

  BlockingQueueTest() {
    data_ = {
        makeRowVector({
            makeFlatVector<int64_t>({1, 2, 3}),
            makeFlatVector<int32_t>({10, 20, 30}),
            makeConstant(true, 3),
        }),
        makeRowVector({
            makeFlatVector<int64_t>({2, 3, 4, 5}),
            makeFlatVector<int32_t>({20, 30, 40, 50}),
            makeConstant(false, 4),
        })};
  }

  std::vector<RowVectorPtr> data_;
};

TEST_F(BlockingQueueTest, sanity) {
  BlockingQueue queue;

  // Put data into the queue.
  for (auto& row : data_) {
    queue.put(row);
  }
  queue.noMoreInput();

  // Read the data back.
  for (auto& expectedRow : data_) {
    ContinueFuture future;
    auto result = queue.read(future);
    ASSERT_FALSE(!future.valid());
    ASSERT_EQ(result.value()->size(), expectedRow->size());
  }

  ContinueFuture future;
  auto result = queue.read(future);
  ASSERT_FALSE(!future.valid());
  ASSERT_EQ(result, nullptr);

  ASSERT_TRUE(queue.empty());
}

TEST_F(BlockingQueueTest, concurrentPutAndRead) {
  BlockingQueue queue;
  const int numIterations = 10;

  // Consumer thread.
  std::thread consumer([&]() {
    for (int i = 0; i < numIterations; ++i) {
      for (auto& expectedRow : data_) {
        while (true) {
          ContinueFuture future = ContinueFuture::makeEmpty();
          auto result = queue.read(future);
          if (future.valid()) {
            future.wait();
            continue;
          }
          ASSERT_EQ(result.value()->size(), expectedRow->size());
          break;
        }
      }
    }
    while (true) {
      ContinueFuture future = ContinueFuture::makeEmpty();
      auto result = queue.read(future);
      if (future.valid()) {
        future.wait();
        continue;
      }
      ASSERT_EQ(result, nullptr);
      break;
    }
  });

  // Producer thread.
  std::thread producer([&]() {
    for (int i = 0; i < numIterations; ++i) {
      // Put data into the queue.
      for (auto& row : data_) {
        // Insert some delay to block the consumer thread.
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        queue.put(row);
      }
    }
    queue.noMoreInput();
  });

  producer.join();
  consumer.join();
  ASSERT_TRUE(queue.empty());
}
} // namespace velox4j
