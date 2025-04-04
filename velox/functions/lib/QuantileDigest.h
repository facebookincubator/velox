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

#include <folly/Bits.h>

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/Portability.h"
#include "velox/common/memory/HashStringAllocator.h"

namespace facebook::velox::functions {

namespace qdigest {

constexpr double kZeroWeightThreshold = 1.0E-5;

} // namespace qdigest

template <typename T>
class QuantileDigest {
  static_assert(
      std::is_same_v<T, int64_t> || std::is_same_v<T, double> ||
      std::is_same_v<T, float>);

 public:
  using U = std::conditional_t<
      std::is_same_v<T, int64_t> || std::is_same_v<T, double>,
      int64_t,
      int32_t>;

  explicit QuantileDigest(
      HashStringAllocator* allocator,
      double maxError = 0.01);

  QuantileDigest(HashStringAllocator* allocator, const char* serialized);

  void add(T value, double weight);

  void merge(const QuantileDigest& other);

  double getCount() const;

  void scale(double scaleFactor);

  void compress();

  std::vector<T> estimateQuantiles(const std::vector<double>& quantiles);

  T estimateQuantile(double quantile);

  int64_t estimatedInMemorySizeInBytes() const;

  int64_t serialize(char* out);

  T getMin();

  T getMax();

 private:
  void insert(U value, double count);

  void setChild(int32_t parent, U branch, int32_t child);

  int32_t makeSiblings(int32_t first, int32_t second);

  int32_t createLeaf(U value, double count);

  int32_t createNode(U value, int8_t level, double count);

  bool inSameSubtree(U bitsA, U bitsB, int32_t level);

  U preprocessByType(T value) const;

  T postprocessByType(U bits) const;

  U longToBits(U value) const;

  U bitsToLong(U bits) const;

  U getBranchMask(int8_t level);

  int32_t calculateCompressionFactor() const;

  int32_t tryRemove(int32_t node);

  void remove(int32_t node);

  void pushFree(int32_t node);

  int32_t popFree();

  template <typename Func>
  bool postOrderTraverse(
      int32_t node,
      Func callback,
      const std::vector<int32_t, StlAllocator<int32_t>>& firstChildren,
      const std::vector<int32_t, StlAllocator<int32_t>>& secondChildren) {
    if (node == -1) {
      return false;
    } else {
      auto first = firstChildren[node];
      auto second = secondChildren[node];
      if (first != -1 &&
          !postOrderTraverse(first, callback, firstChildren, secondChildren)) {
        return false;
      } else {
        return second != -1 &&
                !postOrderTraverse(
                    second, callback, firstChildren, secondChildren)
            ? false
            : callback(node);
      }
    }
  }

  int32_t mergeRecursive(int32_t node, QuantileDigest other, int32_t otherNode);

  int32_t copyRecursive(QuantileDigest other, int32_t otherNode);

  U lowerBound(int32_t node) const;

  U upperBound(int32_t node) const;

  double maxError_{0.01};
  double weightedCount_;
  U min_;
  U max_;
  int32_t root_;
  int32_t nextNode_;
  int32_t firstFree_;
  int32_t freeCount_;
  std::vector<double, StlAllocator<double>> counts_;
  std::vector<int8_t, StlAllocator<int8_t>> levels_;
  std::vector<U, StlAllocator<U>> values_;
  std::vector<int32_t, StlAllocator<int32_t>> lefts_;
  std::vector<int32_t, StlAllocator<int32_t>> rights_;
};

template <typename T>
QuantileDigest<T>::QuantileDigest(
    HashStringAllocator* allocator,
    double maxError)
    : maxError_{maxError},
      weightedCount_{0},
      min_{std::numeric_limits<U>::max()},
      max_{std::numeric_limits<U>::min()},
      root_{-1},
      nextNode_{0},
      firstFree_{-1},
      freeCount_{0},
      counts_{1, 0, StlAllocator<double>(allocator)},
      levels_{1, 0, StlAllocator<int8_t>(allocator)},
      values_{1, 0, StlAllocator<U>(allocator)},
      lefts_{1, -1, StlAllocator<int32_t>(allocator)},
      rights_{1, -1, StlAllocator<int32_t>(allocator)} {}

template <typename T>
void read(const char*& input, T& value) {
  value = folly::loadUnaligned<T>(input);
  input += sizeof(T);
}

template <typename T>
T read(const char*& input) {
  T value = folly::loadUnaligned<T>(input);
  input += sizeof(T);
  return value;
}

template <typename T>
QuantileDigest<T>::QuantileDigest(
    HashStringAllocator* allocator,
    const char* serialized)
    : weightedCount_{0},
      root_{-1},
      firstFree_{-1},
      freeCount_{0},
      counts_{StlAllocator<double>(allocator)},
      levels_{StlAllocator<int8_t>(allocator)},
      values_{StlAllocator<U>(allocator)},
      lefts_{StlAllocator<int32_t>(allocator)},
      rights_{StlAllocator<int32_t>(allocator)} {
  const char* in = serialized;
  int8_t version;
  read<int8_t>(in, version);
  VELOX_CHECK_EQ(version, 0);

  read<double>(in, maxError_);

  auto alpha = read<double>(in);
  VELOX_CHECK_EQ(alpha, 0.0);

  auto landmark = read<int64_t>(in);
  VELOX_CHECK_EQ(landmark, 0);

  int64_t data;
  read<int64_t>(in, data);
  min_ = static_cast<U>(data);
  read<int64_t>(in, data);
  max_ = static_cast<U>(data);

  auto nodeCount = read<int32_t>(in);
  int32_t height;
  if constexpr (std::is_same_v<U, int64_t>) {
    height = 64 - count_leading_zeros(min_ ^ max_) + 1;
    VELOX_CHECK(
        height >= 64 || static_cast<int64_t>(nodeCount) <= (1L << height) - 1L,
        "Too many nodes in deserialized tree. Possible corruption");
  } else {
    height = 32 - count_leading_zeros_32bits(min_ ^ max_) + 1;
    VELOX_CHECK(
        height >= 32 || static_cast<int64_t>(nodeCount) <= (1L << height) - 1L,
        "Too many nodes in deserialized tree. Possible corruption");
  }

  counts_.resize(nodeCount, 0);
  levels_.resize(nodeCount, 0);
  values_.resize(nodeCount, 0);
  lefts_.resize(nodeCount, -1);
  rights_.resize(nodeCount, -1);

  std::vector<int32_t, StlAllocator<int32_t>> stack(
      height, StlAllocator<int32_t>(allocator));
  int32_t top = -1;
  for (auto i = 0; i < nodeCount; ++i) {
    auto nodeStructure = read<int8_t>(in);
    bool hasRight = (nodeStructure & 2) != 0;
    bool hasLeft = (nodeStructure & 1) != 0;
    levels_[i] =
        static_cast<int8_t>(static_cast<uint8_t>(nodeStructure) >> 2 & 63);
    if constexpr (std::is_same_v<U, int32_t>) {
      levels_[i] = (levels_[i] == 64) ? 32 : levels_[i];
    }
    if (hasLeft || hasRight) {
      levels_[i]++;
    }

    if (hasRight) {
      rights_[i] = stack[top--];
    } else {
      rights_[i] = -1;
    }
    if (hasLeft) {
      lefts_[i] = stack[top--];
    } else {
      lefts_[i] = -1;
    }

    ++top;
    stack[top] = i;
    read<double>(in, counts_[i]);
    weightedCount_ += counts_[i];
    if constexpr (std::is_same_v<U, int64_t>) {
      read<int64_t>(in, values_[i]);
    } else {
      int64_t bits;
      read<int64_t>(in, bits);
      values_[i] = (bits ^ std::numeric_limits<int64_t>::min()) ^
          std::numeric_limits<int32_t>::min();
    }
  }
  VELOX_CHECK(
      nodeCount == 0 || top == 0,
      "Tree is corrupted. Expected a single root node");
  root_ = nodeCount - 1;
  nextNode_ = nodeCount;
}

template <typename T>
double QuantileDigest<T>::getCount() const {
  return weightedCount_;
}

template <typename T>
void QuantileDigest<T>::scale(double scaleFactor) {
  VELOX_USER_CHECK(scaleFactor > 0.0, "scale factor must be > 0");
  for (auto i = 0; i < counts_.size(); ++i) {
    counts_[i] *= scaleFactor;
  }

  weightedCount_ *= scaleFactor;
  compress();
}

template <typename T>
QuantileDigest<T>::U QuantileDigest<T>::preprocessByType(T value) const {
  if constexpr (std::is_same_v<T, int64_t>) {
    return value;
  }
  if constexpr (std::is_same_v<T, double>) {
    auto bits = *reinterpret_cast<int64_t*>(&value);
    return bits ^ ((bits >> 63) & std::numeric_limits<int64_t>::max());
  } else {
    auto bits = *reinterpret_cast<int32_t*>(&value);
    return bits ^ ((bits >> 31) & std::numeric_limits<int32_t>::max());
  }
}

template <typename T>
T QuantileDigest<T>::postprocessByType(U bits) const {
  if constexpr (std::is_same_v<T, int64_t>) {
    return bits;
  } else if constexpr (std::is_same_v<T, double>) {
    bits = bits ^ ((bits >> 63) & std::numeric_limits<int64_t>::max());
    return *reinterpret_cast<double*>(&bits);
  } else {
    bits = bits ^ ((bits >> 31) & std::numeric_limits<int32_t>::max());
    return *reinterpret_cast<float*>(&bits);
  }
}

template <typename T>
void QuantileDigest<T>::add(T value, double weight) {
  VELOX_USER_CHECK(weight > 0.0, "weight must be > 0");
  bool needsCompression{false};
  auto processedValue = preprocessByType(value);
  max_ = std::max(max_, processedValue);
  min_ = std::min(min_, processedValue);
  auto previousCount = weightedCount_;
  insert(longToBits(processedValue), weight);
  auto compressionFactor = calculateCompressionFactor();
  if (needsCompression ||
      static_cast<int64_t>(previousCount) /
              static_cast<int64_t>(compressionFactor) !=
          static_cast<int64_t>(weightedCount_) /
              static_cast<int64_t>(compressionFactor)) {
    compress();
  }
}

template <typename T>
QuantileDigest<T>::U QuantileDigest<T>::longToBits(U value) const {
  return value ^ std::numeric_limits<U>::min();
}

template <typename T>
QuantileDigest<T>::U QuantileDigest<T>::bitsToLong(U bits) const {
  return bits ^ std::numeric_limits<U>::min();
}

template <typename T>
int32_t QuantileDigest<T>::calculateCompressionFactor() const {
  if constexpr (std::is_same_v<U, int64_t>) {
    return root_ == -1
        ? 1
        : std::max(
              static_cast<int>(
                  static_cast<double>(levels_[root_] + 1) / maxError_),
              1);
  } else {
    return root_ == -1
        ? 1
        : std::max(
              static_cast<int>(
                  static_cast<double>(
                      (levels_[root_] == 32 ? 64 : levels_[root_]) + 1) /
                  maxError_),
              1);
  }
}

template <typename T>
void QuantileDigest<T>::insert(U value, double count) {
  if (count < qdigest::kZeroWeightThreshold) {
    return;
  }
  U lastBranch = 0;
  int32_t parent = -1;
  int32_t current = root_;
  while (current != -1) {
    auto currentValue = values_[current];
    auto currentLevel = levels_[current];
    if (!inSameSubtree(value, currentValue, currentLevel)) {
      setChild(
          parent, lastBranch, makeSiblings(current, createLeaf(value, count)));
      return;
    }

    if (currentLevel == 0 && currentValue == value) {
      counts_[current] += count;
      weightedCount_ += count;
      return;
    }

    U branch = value & getBranchMask(currentLevel);
    parent = current;
    lastBranch = branch;
    if (branch == 0) {
      current = lefts_[current];
    } else {
      current = rights_[current];
    }
  }
  setChild(parent, lastBranch, createLeaf(value, count));
}

template <typename T>
int32_t QuantileDigest<T>::createLeaf(U value, double count) {
  return createNode(value, 0, count);
}

template <typename T>
int32_t QuantileDigest<T>::createNode(U value, int8_t level, double count) {
  auto node = popFree();
  if (node == -1) {
    if (nextNode_ == counts_.size()) {
      int32_t newSize = counts_.size() +
          std::min(counts_.size(),
                   static_cast<uint64_t>(calculateCompressionFactor() / 5 + 1));
      counts_.resize(newSize);
      levels_.resize(newSize);
      values_.resize(newSize);
      lefts_.resize(newSize);
      rights_.resize(newSize);
    }

    node = nextNode_;
    nextNode_++;
  }
  weightedCount_ += count;
  values_[node] = value;
  levels_[node] = level;
  counts_[node] = count;
  lefts_[node] = -1;
  rights_[node] = -1;
  return node;
}

template <typename T>
bool QuantileDigest<T>::inSameSubtree(U bitsA, U bitsB, int32_t level) {
  if constexpr (std::is_same_v<U, int64_t>) {
    return (level == 64) ||
        ((static_cast<uint64_t>(bitsA) >> level) ==
         (static_cast<uint64_t>(bitsB) >> level));
  } else {
    return (level == 32) ||
        ((static_cast<uint32_t>(bitsA) >> level) ==
         (static_cast<uint32_t>(bitsB) >> level));
  }
}

template <typename T>
QuantileDigest<T>::U QuantileDigest<T>::getBranchMask(int8_t level) {
  return static_cast<U>(1) << (level - 1);
}

template <typename T>
int32_t QuantileDigest<T>::makeSiblings(int32_t first, int32_t second) {
  auto firstValue = values_[first];
  auto secondValue = values_[second];
  int32_t parentLevel;
  if constexpr (std::is_same_v<U, int64_t>) {
    parentLevel = 64 - count_leading_zeros(firstValue ^ secondValue);
  } else {
    parentLevel = 32 - count_leading_zeros_32bits(firstValue ^ secondValue);
  }
  auto parent = createNode(firstValue, parentLevel, 0.0);
  auto branch = firstValue & getBranchMask(levels_[parent]);
  if (branch == 0) {
    lefts_[parent] = first;
    rights_[parent] = second;
  } else {
    lefts_[parent] = second;
    rights_[parent] = first;
  }
  return parent;
}

template <typename T>
void QuantileDigest<T>::setChild(int32_t parent, U branch, int32_t child) {
  if (parent == -1) {
    root_ = child;
  } else if (branch == 0) {
    lefts_[parent] = child;
  } else {
    rights_[parent] = child;
  }
}

template <typename T>
void QuantileDigest<T>::compress() {
  double bound = std::floor(
      weightedCount_ / static_cast<double>(calculateCompressionFactor()));
  postOrderTraverse(
      root_,
      [this, bound](int32_t node) mutable {
        auto left = lefts_[node];
        auto right = rights_[node];
        if (left == -1 && right == -1) {
          return true;
        } else {
          double leftCount = (left == -1) ? 0.0 : counts_[left];
          double rightCount = (right == -1) ? 0.0 : counts_[right];
          bool shouldCompress =
              (counts_[node] + leftCount + rightCount) < bound;
          if (left != -1 &&
              (shouldCompress || leftCount < qdigest::kZeroWeightThreshold)) {
            lefts_[node] = tryRemove(left);
            counts_[node] += leftCount;
          }

          if (right != -1 &&
              (shouldCompress || rightCount < qdigest::kZeroWeightThreshold)) {
            rights_[node] = tryRemove(right);
            counts_[node] += rightCount;
          }

          return true;
        }
      },
      lefts_,
      rights_);
  if (root_ != -1 && counts_[root_] < qdigest::kZeroWeightThreshold) {
    root_ = tryRemove(root_);
  }
}

template <typename T>
int32_t QuantileDigest<T>::tryRemove(int32_t node) {
  VELOX_USER_CHECK_NE(node, -1, "node is -1");
  auto left = lefts_[node];
  auto right = rights_[node];
  if (left == -1 && right == -1) {
    remove(node);
    return -1;
  } else if (left != -1 && right != -1) {
    counts_[node] = 0.0;
    return node;
  } else {
    remove(node);
    return left != -1 ? left : right;
  }
}

template <typename T>
void QuantileDigest<T>::remove(int32_t node) {
  if (node == nextNode_ - 1) {
    --nextNode_;
  } else {
    pushFree(node);
  }
  if (node == root_) {
    root_ = -1;
  }
}

template <typename T>
void QuantileDigest<T>::pushFree(int32_t node) {
  lefts_[node] = firstFree_;
  firstFree_ = node;
  ++freeCount_;
}

template <typename T>
int32_t QuantileDigest<T>::popFree() {
  auto node = firstFree_;
  if (node == -1) {
    return node;
  } else {
    firstFree_ = lefts_[firstFree_];
    --freeCount_;
    return node;
  }
}

template <typename T>
void QuantileDigest<T>::merge(const QuantileDigest<T>& other) {
  // this.rescaleToCommonLandmark(this, other);
  root_ = mergeRecursive(root_, other, other.root_);
  max_ = std::max(max_, other.max_);
  min_ = std::min(min_, other.min_);
  compress();
}

template <typename T>
int32_t QuantileDigest<T>::mergeRecursive(
    int32_t node,
    QuantileDigest<T> other,
    int32_t otherNode) {
  if (otherNode == -1) {
    return node;
  } else if (node == -1) {
    return copyRecursive(other, otherNode);
  } else if (!inSameSubtree(
                 values_[node],
                 other.values_[otherNode],
                 std::max(levels_[node], other.levels_[otherNode]))) {
    return makeSiblings(node, copyRecursive(other, otherNode));
  } else {
    if (levels_[node] > other.levels_[otherNode]) {
      int32_t left;
      auto branch = other.values_[otherNode] & getBranchMask(levels_[node]);
      if (branch == 0) {
        left = mergeRecursive(lefts_[node], other, otherNode);
        lefts_[node] = left;
      } else {
        left = mergeRecursive(rights_[node], other, otherNode);
        rights_[node] = left;
      }
      return node;
    } else if (levels_[node] < other.levels_[otherNode]) {
      auto branch = values_[node] & getBranchMask(other.levels_[otherNode]);
      int32_t left, right;
      if (branch == 0) {
        left = mergeRecursive(node, other, other.lefts_[otherNode]);
        right = copyRecursive(other, other.rights_[otherNode]);
      } else {
        left = copyRecursive(other, other.lefts_[otherNode]);
        right = mergeRecursive(node, other, other.rights_[otherNode]);
      }

      auto result = createNode(
          other.values_[otherNode],
          other.levels_[otherNode],
          other.counts_[otherNode]);
      lefts_[result] = left;
      rights_[result] = right;
      return result;
    } else {
      weightedCount_ += other.counts_[otherNode];
      counts_[node] += other.counts_[otherNode];
      auto left = mergeRecursive(lefts_[node], other, other.lefts_[otherNode]);
      auto right =
          mergeRecursive(rights_[node], other, other.rights_[otherNode]);
      lefts_[node] = left;
      rights_[node] = right;
      return node;
    }
  }
}

template <typename T>
int32_t QuantileDigest<T>::copyRecursive(
    QuantileDigest other,
    int32_t otherNode) {
  if (otherNode == -1) {
    return otherNode;
  } else {
    auto node = createNode(
        other.values_[otherNode],
        other.levels_[otherNode],
        other.counts_[otherNode]);
    if (other.lefts_[otherNode] != -1) {
      lefts_[node] = copyRecursive(other, other.lefts_[otherNode]);
    }
    if (other.rights_[otherNode] != -1) {
      rights_[node] = copyRecursive(other, other.rights_[otherNode]);
    }
    return node;
  }
}

inline bool validateQuantiles(const std::vector<double>& quantiles) {
  VELOX_CHECK(!quantiles.empty());
  VELOX_USER_CHECK_GE(quantiles[0], 0.0);
  VELOX_USER_CHECK_LE(quantiles[0], 1.0);
  for (auto i = 1; i < quantiles.size(); ++i) {
    VELOX_USER_CHECK_GE(quantiles[i], quantiles[i - 1]);
    VELOX_USER_CHECK_GE(quantiles[i], 0.0);
    VELOX_USER_CHECK_LE(quantiles[i], 1.0);
  }
  return true;
}

template <typename T>
std::vector<T> QuantileDigest<T>::estimateQuantiles(
    const std::vector<double>& quantiles) {
  VELOX_DCHECK(validateQuantiles(quantiles));
  std::vector<T> result;
  int i = -1;
  double sum = 0.0;
  postOrderTraverse(
      root_,
      [this, &result, &quantiles, &i, &sum](int32_t node) {
        sum += counts_[node];
        while (i + 1 < quantiles.size() &&
               sum > quantiles[i + 1] * weightedCount_) {
          result.push_back(postprocessByType(std::min(upperBound(node), max_)));
          i++;
        }
        return i < static_cast<int64_t>(quantiles.size());
      },
      lefts_,
      rights_);
  for (; i + 1 < quantiles.size(); ++i) {
    result.push_back(postprocessByType(max_));
  }
  return result;
}

template <typename T>
T QuantileDigest<T>::estimateQuantile(double quantile) {
  return estimateQuantiles({quantile})[0];
}

template <typename T>
T QuantileDigest<T>::getMin() {
  T result = std::numeric_limits<T>::min();
  postOrderTraverse(
      root_,
      [this, &result](int32_t node) {
        if (counts_[node] >= qdigest::kZeroWeightThreshold) {
          result = postprocessByType(lowerBound(node));
          return false;
        } else {
          return true;
        }
      },
      lefts_,
      rights_);
  return std::max(postprocessByType(min_), result);
}

template <typename T>
T QuantileDigest<T>::getMax() {
  T result = std::numeric_limits<T>::max();
  postOrderTraverse(
      root_,
      [this, &result](int32_t node) {
        if (counts_[node] >= qdigest::kZeroWeightThreshold) {
          result = postprocessByType(upperBound(node));
          return false;
        } else {
          return true;
        }
      },
      rights_,
      lefts_);
  return std::min(postprocessByType(max_), result);
}

template <typename T>
QuantileDigest<T>::U QuantileDigest<T>::lowerBound(int32_t node) const {
  if constexpr (std::is_same_v<U, int64_t>) {
    uint64_t mask = 0L;
    if (levels_[node] > 0) {
      mask = static_cast<uint64_t>(-1L) >> (64 - levels_[node]);
    }
    return bitsToLong(values_[node] & ~mask);
  } else {
    uint32_t mask = 0;
    if (levels_[node] > 0) {
      mask = static_cast<uint32_t>(-1L) >> (32 - levels_[node]);
    }
    return bitsToLong(values_[node] & ~mask);
  }
}

template <typename T>
QuantileDigest<T>::U QuantileDigest<T>::upperBound(int32_t node) const {
  if constexpr (std::is_same_v<U, int64_t>) {
    uint64_t mask = 0L;
    if (levels_[node] > 0) {
      mask = static_cast<uint64_t>(-1L) >> (64 - levels_[node]);
    }
    return bitsToLong(values_[node] | mask);
  } else {
    uint32_t mask = 0L;
    if (levels_[node] > 0) {
      mask = static_cast<uint32_t>(-1) >> (32 - levels_[node]);
    }
    return bitsToLong(values_[node] | mask);
  }
}

template <typename T>
int64_t QuantileDigest<T>::estimatedInMemorySizeInBytes() const {
  auto nodeCount = nextNode_ - freeCount_;
  return /*version*/ sizeof(char) + sizeof(maxError_) +
      /*alpha*/ sizeof(double) + /*landmarkInSeconds*/ sizeof(int64_t) +
      /*min*/ sizeof(int64_t) + /*max*/ sizeof(int64_t) +
      /*nodeCount*/ sizeof(int32_t) +
      (sizeof(typename decltype(counts_)::value_type) * nodeCount) +
      (sizeof(typename decltype(levels_)::value_type) * nodeCount) +
      /*values*/ (sizeof(int64_t) * nodeCount);
}

template <typename T>
void write(T value, char*& out) {
  folly::storeUnaligned(out, value);
  out += sizeof(T);
}

template <typename T>
int64_t QuantileDigest<T>::serialize(char* out) {
  compress();
  const char* outStart = out;
  write<char>(0, out); // version
  write<double>(maxError_, out);
  write<double>(0.0, out); // alpha
  write<int64_t>(0, out); // landmarkInSeconds
  write<int64_t>(static_cast<int64_t>(min_), out);
  write<int64_t>(static_cast<int64_t>(max_), out);
  write<int32_t>(nextNode_ - freeCount_, out);
  postOrderTraverse(
      root_,
      [&](int32_t node) {
        if constexpr (std::is_same_v<U, int64_t>) {
          auto nodeStructure =
              static_cast<int8_t>(std::max(levels_[node] - 1, 0) << 2);
          if (lefts_[node] != -1) {
            nodeStructure = static_cast<int8_t>(nodeStructure | 1);
          }
          if (rights_[node] != -1) {
            nodeStructure = static_cast<int8_t>(nodeStructure | 2);
          }
          write<int8_t>(nodeStructure, out);
          write<double>(counts_[node], out);
          write<int64_t>(values_[node], out);

        } else {
          auto nodeStructure = static_cast<int8_t>(
              std::max((levels_[node] == 32 ? 64 : levels_[node]) - 1, 0) << 2);
          if (lefts_[node] != -1) {
            nodeStructure = static_cast<int8_t>(nodeStructure | 1);
          }
          if (rights_[node] != -1) {
            nodeStructure = static_cast<int8_t>(nodeStructure | 2);
          }
          write<int8_t>(nodeStructure, out);
          write<double>(counts_[node], out);
          write<int64_t>(
              (values_[node] ^ std::numeric_limits<int32_t>::min()) ^
                  std::numeric_limits<int64_t>::min(),
              out);
        }

        return true;
      },
      lefts_,
      rights_);
  VELOX_CHECK_LE(out - outStart, estimatedInMemorySizeInBytes());
  return out - outStart;
}

} // namespace facebook::velox::functions
