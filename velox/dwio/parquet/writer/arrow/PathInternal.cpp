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

// Adapted from Apache Arrow.

// Overview.
//
// The strategy used for this code for repetition/definition.
// Is to dissect the top level array into a list of paths.
// From the top level array to the final primitive (possibly.
// Dictionary encoded array). It then evaluates each one of.
// Those paths to produce results for the callback iteratively.
//
// This approach was taken to reduce the aggregate memory required if we were.
// To build all def/rep levels in parallel as apart of a tree traversal.  It.
// Also allows for straightforward parallelization at the path level if that is.
// Desired in the future.
//
// The main downside to this approach is it duplicates effort for nodes.
// That share common ancestors. This can be mitigated to some degree.
// By adding in optimizations that detect leaf arrays that share.
// The same common list ancestor and reuse the repetition levels.
// From the first leaf encountered (only definition levels greater.
// The list ancestor need to be re-evaluated. This is left for future.
// Work.
//
// Algorithm.
//
// As mentioned above this code dissects arrays into constituent parts:
// Nullability data, and list offset data. It tries to optimize for.
// Some special cases, where it is known ahead of time that a step.
// Can be skipped (e.g. a nullable array happens to have all of its.
// Values) or batch filled (a nullable array has all null values).
// One further optimization that is not implemented but could be done.
// In the future is special handling for nested list arrays that.
// Have some intermediate data which indicates the final array contains only.
// Nulls.
//
// In general, the algorithm attempts to batch work at each node as much.
// As possible.  For nullability nodes this means finding runs of null.
// Values and batch filling those interspersed with finding runs of non-null.
// Values to process in batch at the next column.
//
// Similarly, list runs of empty lists are all processed in one batch.
// Followed by either:
//    - A single list entry for non-terminal lists (i.e. the upper part of a.
//    nested list)
//    - Runs of non-empty lists for the terminal list (i.e. the lowest part of
//    a. Nested list).
//
// This makes use of the following observations.
// 1.  Null values at any node on the path are terminal (repetition and.
// Definition.
//     Level can be set directly when a Null value is encountered).
// 2.  Empty lists share this eager termination property with Null values.
// 3.  In order to keep repetition/definition level populated the algorithm is.
// Lazy.
//     In assigning repetition levels. The algorithm tracks whether it is.
//     Currently in the middle of a list by comparing the lengths of.
//     Repetition/definition levels. If it is currently in the middle of a list.
//     The the number of repetition levels populated will be greater than.
//     Definition levels (the start of a List requires adding the first.
//     Element). If there are equal numbers of definition and repetition levels.
//     Populated this indicates a list is waiting to be started and the next.
//     List encountered will have its repetition level signify the beginning of.
//     The list.
//
//     Other implementation notes.
//
//     This code hasn't been benchmarked (or assembly analyzed) but did the.
//     Following as optimizations (yes premature optimization is the root of
//     all. Evil).
//     - This code does not use recursion, instead it constructs its own stack.
//     And manages.
//       Updating elements accordingly.
//     - It tries to avoid using Status for common return states.
//     - Avoids virtual dispatch in favor of if/else statements on a set of
//     well. Known classes.

#include "velox/dwio/parquet/writer/arrow/PathInternal.h"

#include <atomic>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "arrow/array.h"
#include "arrow/buffer.h"
#include "arrow/buffer_builder.h"
#include "arrow/extension_type.h"
#include "arrow/memory_pool.h"
#include "arrow/type.h"
#include "arrow/type_traits.h"
#include "arrow/util/bit_run_reader.h"
#include "arrow/util/bit_util.h"
#include "arrow/util/bitmap_visit.h"
#include "arrow/util/macros.h"
#include "arrow/visit_array_inline.h"

#include "velox/common/base/Exceptions.h"
#include "velox/dwio/parquet/writer/arrow/Properties.h"

namespace facebook::velox::parquet::arrow::arrow {

namespace {

using ::arrow::Array;
using ::arrow::Status;
using ::arrow::TypedBufferBuilder;

constexpr static int16_t kLevelNotSet = -1;

/// \brief Simple result of a iterating over a column to determine values.
enum IterationResult { /// Processing is done at this node. Move back up the
                       /// path. To continue processing.
  kDone = -1,
  /// Move down towards the leaf for processing.
  kNext = 1,
  /// An error occurred while processing.
  kError = 2
};

#define RETURN_IF_ERROR(iterationResult)                  \
  do {                                                    \
    if (ARROW_PREDICT_FALSE(iterationResult == kError)) { \
      return iterationResult;                             \
    }                                                     \
  } while (false)

int64_t lazyNullCount(const Array& array) {
  return array.data()->null_count.load();
}

bool lazyNoNulls(const Array& array) {
  int64_t nullCount = lazyNullCount(array);
  return nullCount == 0 ||
      // KUnkownNullCount comparison is needed to account.
      // For null arrays.
      (nullCount == ::arrow::kUnknownNullCount &&
       array.null_bitmap_data() == nullptr);
}

struct PathWriteContext {
  PathWriteContext(
      ::arrow::MemoryPool* pool,
      std::shared_ptr<::arrow::ResizableBuffer> defLevelsBuffer)
      : repLevels(pool), defLevels(std::move(defLevelsBuffer), pool) {}
  IterationResult reserveDefLevels(int64_t elements) {
    lastStatus = defLevels.Reserve(elements);
    if (ARROW_PREDICT_TRUE(lastStatus.ok())) {
      return kDone;
    }
    return kError;
  }

  IterationResult appendDefLevel(int16_t defLevel) {
    lastStatus = defLevels.Append(defLevel);
    if (ARROW_PREDICT_TRUE(lastStatus.ok())) {
      return kDone;
    }
    return kError;
  }

  IterationResult appendDefLevels(int64_t count, int16_t defLevel) {
    lastStatus = defLevels.Append(count, defLevel);
    if (ARROW_PREDICT_TRUE(lastStatus.ok())) {
      return kDone;
    }
    return kError;
  }

  void unsafeAppendDefLevel(int16_t defLevel) {
    defLevels.UnsafeAppend(defLevel);
  }

  IterationResult appendRepLevel(int16_t repLevel) {
    lastStatus = repLevels.Append(repLevel);

    if (ARROW_PREDICT_TRUE(lastStatus.ok())) {
      return kDone;
    }
    return kError;
  }

  IterationResult appendRepLevels(int64_t count, int16_t repLevel) {
    lastStatus = repLevels.Append(count, repLevel);
    if (ARROW_PREDICT_TRUE(lastStatus.ok())) {
      return kDone;
    }
    return kError;
  }

  bool equalRepDefLevelsLengths() const {
    return repLevels.length() == defLevels.length();
  }

  // Incorporates |range| into visited elements. If the |range| is contiguous.
  // With the last range, extend the last range, otherwise add |range|.
  // Separately to the list.
  void recordPostListVisit(const ElementRange& range) {
    if (!visitedElements.empty() && range.start == visitedElements.back().end) {
      visitedElements.back().end = range.end;
      return;
    }
    visitedElements.push_back(range);
  }

  Status lastStatus;
  TypedBufferBuilder<int16_t> repLevels;
  TypedBufferBuilder<int16_t> defLevels;
  std::vector<ElementRange> visitedElements;
};

IterationResult
fillRepLevels(int64_t count, int16_t repLevel, PathWriteContext* context) {
  if (repLevel == kLevelNotSet) {
    return kDone;
  }
  int64_t fillCount = count;
  // This condition occurs (rep and dep levels equals), in one of.
  // In a few cases:
  // 1.  Before any list is encountered.
  // 2.  After rep-level has been filled in due to null/empty.
  //     Values above it.
  // 3.  After finishing a list.
  if (!context->equalRepDefLevelsLengths()) {
    fillCount--;
  }
  return context->appendRepLevels(fillCount, repLevel);
}

// A node for handling an array that is discovered to have all.
// Null elements. It is referred to as a TerminalNode because.
// Traversal of nodes will not continue it when generating.
// Rep/def levels. However, there could be many nested children.
// Elements beyond it in the Array that is being processed.
class AllNullsTerminalNode {
 public:
  explicit AllNullsTerminalNode(
      int16_t defLevel,
      int16_t repLevel = kLevelNotSet)
      : defLevel_(defLevel), repLevel_(repLevel) {}
  void setRepLevelIfNull(int16_t repLevel) {
    repLevel_ = repLevel;
  }
  IterationResult run(const ElementRange& range, PathWriteContext* context) {
    int64_t size = range.size();
    RETURN_IF_ERROR(fillRepLevels(size, repLevel_, context));
    return context->appendDefLevels(size, defLevel_);
  }

 private:
  int16_t defLevel_;
  int16_t repLevel_;
};

// Handles the case where all remaining arrays until the leaf have no nulls.
// (And are not interrupted by lists). Unlike AllNullsTerminalNode this is.
// Always the last node in a path. We don't need an analogue to the.
// AllNullsTerminalNode because if all values are present at an intermediate.
// Array no node is added for it (the def-level for the next nullable node is.
// Incremented).
struct AllPresentTerminalNode {
  IterationResult run(const ElementRange& range, PathWriteContext* context) {
    return context->appendDefLevels(range.end - range.start, defLevel);
    // No need to worry about rep levels, because this state should.
    // Only be applicable for after all list/repeated values.
    // Have been evaluated in the path.
  }
  int16_t defLevel;
};

/// Node for handling the case when the leaf-array is nullable.
/// And contains null elements.
struct NullableTerminalNode {
  NullableTerminalNode() = default;

  NullableTerminalNode(
      const uint8_t* bitmap,
      int64_t elementOffset,
      int16_t defLevelIfPresent)
      : bitmap_(bitmap),
        elementOffset_(elementOffset),
        defLevelIfPresent_(defLevelIfPresent),
        defLevelIfNull_(defLevelIfPresent - 1) {}

  IterationResult run(const ElementRange& range, PathWriteContext* context) {
    int64_t elements = range.size();
    RETURN_IF_ERROR(context->reserveDefLevels(elements));

    VELOX_DCHECK_GT(elements, 0);

    auto bitVisitor = [&](bool isSet) {
      context->unsafeAppendDefLevel(
          isSet ? defLevelIfPresent_ : defLevelIfNull_);
    };

    if (elements > 16) { // 16 guarantees at least one unrolled loop.
      ::arrow::internal::VisitBitsUnrolled(
          bitmap_, range.start + elementOffset_, elements, bitVisitor);
    } else {
      ::arrow::internal::VisitBits(
          bitmap_, range.start + elementOffset_, elements, bitVisitor);
    }
    return kDone;
  }
  const uint8_t* bitmap_;
  int64_t elementOffset_;
  int16_t defLevelIfPresent_;
  int16_t defLevelIfNull_;
};

// List nodes handle populating rep_level for Arrow Lists and def-level for.
// Empty lists. Nullability (both list and children) is handled by other Nodes.
// By construction all list nodes will be intermediate nodes (they will always.
// Be followed by at least one other node).
//
// Type parameters:
//    |RangeSelector| - A strategy for determine the the range of the child
//    node. To process.
//       This varies depending on the type of list (int32_t* offsets, int64_t*.
//       Offsets of fixed.
template <typename RangeSelector>
class ListPathNode {
 public:
  ListPathNode(RangeSelector selector, int16_t repLev, int16_t defLevelIfEmpty)
      : selector_(std::move(selector)),
        prevRepLevel_(repLev - 1),
        repLevel_(repLev),
        defLevelIfEmpty_(defLevelIfEmpty) {}

  int16_t repLevel() const {
    return repLevel_;
  }

  IterationResult run(
      ElementRange* range,
      ElementRange* childRange,
      PathWriteContext* context) {
    if (range->empty()) {
      return kDone;
    }
    // Find the first non-empty list (skipping a run of empties).
    int64_t emptyElements = 0;
    do {
      // Retrieve the range of elements that this list contains.
      *childRange = selector_.getRange(range->start);
      if (!childRange->empty()) {
        break;
      }
      ++emptyElements;
      ++range->start;
    } while (!range->empty());

    // Post condition:
    //   * range is either empty (we are done processing at this node)
    //     Or start corresponds a non-empty list.
    //   * If range is non-empty child_range contains.
    //     The bounds of non-empty list.

    // Handle any skipped over empty lists.
    if (emptyElements > 0) {
      RETURN_IF_ERROR(fillRepLevels(emptyElements, prevRepLevel_, context));
      RETURN_IF_ERROR(
          context->appendDefLevels(emptyElements, defLevelIfEmpty_));
    }
    // Start of a new list. Note that for nested lists adding the element.
    // Here effectively suppresses this code until we either encounter null.
    // Elements or empty lists between here and the innermost list (since.
    // We make the rep levels repetition and definition levels unequal).
    // Similarly when we are backtracking up the stack the repetition and.
    // Definition levels are again equal so if we encounter an intermediate
    // list. With more elements this will detect it as a new list.
    if (context->equalRepDefLevelsLengths() && !range->empty()) {
      RETURN_IF_ERROR(context->appendRepLevel(prevRepLevel_));
    }

    if (range->empty()) {
      return kDone;
    }

    ++range->start;
    if (isLast_) {
      // If this is the last repeated node, we can extend try.
      // To extend the child range as wide as possible before.
      // Continuing to the next node.
      return fillForLast(range, childRange, context);
    }
    return kNext;
  }

  void setLast() {
    isLast_ = true;
  }

 private:
  IterationResult fillForLast(
      ElementRange* range,
      ElementRange* childRange,
      PathWriteContext* context) {
    // First fill int the remainder of the list.
    RETURN_IF_ERROR(fillRepLevels(childRange->size(), repLevel_, context));
    // Once we've reached this point the following preconditions should hold:
    // 1.  There are no more repeated path nodes to deal with.
    // 2.  All elements in |range| represent contiguous elements in the.
    //     Child array (Null values would have shortened the range to ensure.
    //     All remaining list elements are present (though they may be empty.
    //     Lists)).
    // 3.  No element of range spans a parent list (intermediate.
    //     List nodes only handle one list entry at a time).
    //
    // Given these preconditions it should be safe to fill runs on non-empty.
    // Lists here and expand the range in the child node accordingly.

    while (!range->empty()) {
      ElementRange sizeCheck = selector_.getRange(range->start);
      if (sizeCheck.empty()) {
        // The empty range will need to be handled after we pass down the.
        // Accumulated range because it affects def_level placement and we need.
        // To get the children def_levels entered first.
        break;
      }
      // This is the start of a new list. We can be sure it only applies.
      // To the previous list (and doesn't jump to the start of any list.
      // Further up in nesting due to the constraints mentioned at the start.
      // Of the function).
      RETURN_IF_ERROR(context->appendRepLevel(prevRepLevel_));
      RETURN_IF_ERROR(
          context->appendRepLevels(sizeCheck.size() - 1, repLevel_));
      VELOX_DCHECK_EQ(sizeCheck.start, childRange->end);
      childRange->end = sizeCheck.end;
      ++range->start;
    }

    // Do book-keeping to track the elements of the arrays that are actually.
    // Visited beyond this point.  This is necessary to identify "gaps" in.
    // Values that should not be processed (written out to parquet).
    context->recordPostListVisit(*childRange);
    return kNext;
  }

  RangeSelector selector_;
  int16_t prevRepLevel_;
  int16_t repLevel_;
  int16_t defLevelIfEmpty_;
  bool isLast_ = false;
};

template <typename OffsetType>
struct VarRangeSelector {
  ElementRange getRange(int64_t index) const {
    return ElementRange{offsets[index], offsets[index + 1]};
  }

  // Either int32_t* or int64_t*.
  const OffsetType* offsets;
};

struct FixedSizedRangeSelector {
  ElementRange getRange(int64_t index) const {
    int64_t start = index * listSize;
    return ElementRange{start, start + listSize};
  }
  int listSize;
};

// An intermediate node that handles null values.
class NullableNode {
 public:
  NullableNode(
      const uint8_t* nullBitmap,
      int64_t entryOffset,
      int16_t defLevelIfNull,
      int16_t repLevelIfNull = kLevelNotSet)
      : nullBitmap_(nullBitmap),
        entryOffset_(entryOffset),
        validBitsReader_(makeReader(ElementRange{0, 0})),
        defLevelIfNull_(defLevelIfNull),
        repLevelIfNull_(repLevelIfNull),
        newRange_(true) {}

  void setRepLevelIfNull(int16_t repLevel) {
    repLevelIfNull_ = repLevel;
  }

  ::arrow::internal::BitRunReader makeReader(const ElementRange& range) {
    return ::arrow::internal::BitRunReader(
        nullBitmap_, entryOffset_ + range.start, range.size());
  }

  IterationResult run(
      ElementRange* range,
      ElementRange* childRange,
      PathWriteContext* context) {
    if (newRange_) {
      // Reset the reader each time we are starting fresh on a range.
      // We can't rely on continuity because nulls above can.
      // Cause discontinuities.
      validBitsReader_ = makeReader(*range);
    }
    childRange->start = range->start;
    ::arrow::internal::BitRun run = validBitsReader_.NextRun();
    if (!run.set) {
      range->start += run.length;
      RETURN_IF_ERROR(fillRepLevels(run.length, repLevelIfNull_, context));
      RETURN_IF_ERROR(context->appendDefLevels(run.length, defLevelIfNull_));
      run = validBitsReader_.NextRun();
    }
    if (range->empty()) {
      newRange_ = true;
      return kDone;
    }
    childRange->end = childRange->start = range->start;
    childRange->end += run.length;

    VELOX_DCHECK(!childRange->empty());
    range->start += childRange->size();
    newRange_ = false;
    return kNext;
  }

  const uint8_t* nullBitmap_;
  int64_t entryOffset_;
  ::arrow::internal::BitRunReader validBitsReader_;
  int16_t defLevelIfNull_;
  int16_t repLevelIfNull_;

  // Whether the next invocation will be a new range.
  bool newRange_ = true;
};

using ListNode = ListPathNode<VarRangeSelector<int32_t>>;
using LargeListNode = ListPathNode<VarRangeSelector<int64_t>>;
using FixedSizeListNode = ListPathNode<FixedSizedRangeSelector>;

// Contains static information derived from traversing the schema.
struct PathInfo {
  // The vectors are expected to the same length info.

  // Note index order matters here.
  using Node = std::variant<
      NullableTerminalNode,
      ListNode,
      LargeListNode,
      FixedSizeListNode,
      NullableNode,
      AllPresentTerminalNode,
      AllNullsTerminalNode>;

  std::vector<Node> path;
  std::shared_ptr<Array> primitiveArray;
  int16_t maxDefLevel = 0;
  int16_t maxRepLevel = 0;
  bool hasDictionary = false;
  bool leafIsNullable = false;
};

/// Contains logic for writing a single leaf node to parquet.
/// This tracks the path from root to leaf.
///
/// |Writer| will be called after all of the definition/repetition.
/// Values have been calculated for root_range with the calculated.
/// Values. It is intended to abstract the complexity of writing.
/// The levels and values to parquet.
Status writePath(
    ElementRange rootRange,
    PathInfo* pathInfo,
    ArrowWriteContext* arrowContext,
    MultipathLevelBuilder::CallbackFunction writer) {
  std::vector<ElementRange> stack(pathInfo->path.size());
  MultipathLevelBuilderResult builderResult;
  builderResult.leafArray = pathInfo->primitiveArray;
  builderResult.leafIsNullable = pathInfo->leafIsNullable;

  if (pathInfo->maxDefLevel == 0) {
    // This case only occurs when there are no nullable or repeated.
    // Columns in the path from the root to leaf.
    int64_t leafLength = builderResult.leafArray->length();
    builderResult.defRepLevelCount = leafLength;
    builderResult.postListVisitedElements.push_back({0, leafLength});
    return writer(builderResult);
  }
  stack[0] = rootRange;
  RETURN_NOT_OK(arrowContext->defLevelsBuffer->Resize(
      /*new_size=*/0, /*shrink_to_fit*/ false));
  PathWriteContext context(
      arrowContext->memoryPool, arrowContext->defLevelsBuffer);
  // We should need at least this many entries so reserve the space ahead of.
  // Time.
  RETURN_NOT_OK(context.defLevels.Reserve(rootRange.size()));
  if (pathInfo->maxRepLevel > 0) {
    RETURN_NOT_OK(context.repLevels.Reserve(rootRange.size()));
  }

  auto stackBase = &stack[0];
  auto stackPosition = stackBase;
  // This is the main loop for calculated rep/def levels. The nodes.
  // In the path implement a chain-of-responsibility like pattern.
  // Where each node can add some number of repetition/definition.
  // Levels to PathWriteContext and also delegate to the next node.
  // in the path to add values. The values are added through each Run(...)
  // Call and the choice to delegate to the next node (or return to the.
  // Previous node) is communicated by the return value of Run(...).
  // The loop terminates after the first node indicates all values in.
  // |Root_range| are processed.
  while (stackPosition >= stackBase) {
    PathInfo::Node& Node = pathInfo->path[stackPosition - stackBase];
    struct {
      IterationResult operator()(NullableNode& Node) {
        return Node.run(stackPosition, stackPosition + 1, context);
      }
      IterationResult operator()(ListNode& Node) {
        return Node.run(stackPosition, stackPosition + 1, context);
      }
      IterationResult operator()(NullableTerminalNode& Node) {
        return Node.run(*stackPosition, context);
      }
      IterationResult operator()(FixedSizeListNode& Node) {
        return Node.run(stackPosition, stackPosition + 1, context);
      }
      IterationResult operator()(AllPresentTerminalNode& Node) {
        return Node.run(*stackPosition, context);
      }
      IterationResult operator()(AllNullsTerminalNode& Node) {
        return Node.run(*stackPosition, context);
      }
      IterationResult operator()(LargeListNode& Node) {
        return Node.run(stackPosition, stackPosition + 1, context);
      }
      ElementRange* stackPosition;
      PathWriteContext* context;
    } visitor = {stackPosition, &context};

    IterationResult result = std::visit(visitor, Node);

    if (ARROW_PREDICT_FALSE(result == kError)) {
      VELOX_DCHECK(!context.lastStatus.ok());
      return context.lastStatus;
    }
    stackPosition += static_cast<int>(result);
  }
  RETURN_NOT_OK(context.lastStatus);
  builderResult.defRepLevelCount = context.defLevels.length();

  if (context.repLevels.length() > 0) {
    // This case only occurs when there was a repeated element that needs to be.
    // Processed.
    builderResult.repLevels = context.repLevels.data();
    std::swap(builderResult.postListVisitedElements, context.visitedElements);
    // If it is possible when processing lists that all lists where empty. In.
    // This case no elements would have been added to.
    // Post_list_visited_elements. By added an empty element we avoid special.
    // Casing in downstream consumers.
    if (builderResult.postListVisitedElements.empty()) {
      builderResult.postListVisitedElements.push_back({0, 0});
    }
  } else {
    builderResult.postListVisitedElements.push_back(
        {0, builderResult.leafArray->length()});
    builderResult.repLevels = nullptr;
  }

  builderResult.defLevels = context.defLevels.data();
  return writer(builderResult);
}

struct FixupVisitor {
  int maxRepLevel = -1;
  int16_t repLevelIfNull = kLevelNotSet;

  template <typename T>
  void handleListNode(T& arg) {
    if (arg.repLevel() == maxRepLevel) {
      arg.setLast();
      // After the last list node we don't need to fill.
      // Rep levels on null.
      repLevelIfNull = kLevelNotSet;
    } else {
      repLevelIfNull = arg.repLevel();
    }
  }
  void operator()(ListNode& Node) {
    handleListNode(Node);
  }
  void operator()(LargeListNode& Node) {
    handleListNode(Node);
  }
  void operator()(FixedSizeListNode& Node) {
    handleListNode(Node);
  }

  // For non-list intermediate nodes.
  template <typename T>
  void handleIntermediateNode(T& arg) {
    if (repLevelIfNull != kLevelNotSet) {
      arg.setRepLevelIfNull(repLevelIfNull);
    }
  }

  void operator()(NullableNode& arg) {
    handleIntermediateNode(arg);
  }

  void operator()(AllNullsTerminalNode& arg) {
    // Even though no processing happens past this point we.
    // Still need to adjust it if a list occurred after an.
    // All null array.
    handleIntermediateNode(arg);
  }

  void operator()(NullableTerminalNode&) {}
  void operator()(AllPresentTerminalNode&) {}
};

PathInfo fixup(PathInfo info) {
  // We only need to fixup the path if there were repeated.
  // Elements on it.
  if (info.maxRepLevel == 0) {
    return info;
  }
  FixupVisitor visitor;
  visitor.maxRepLevel = info.maxRepLevel;
  if (visitor.maxRepLevel > 0) {
    visitor.repLevelIfNull = 0;
  }
  for (size_t x = 0; x < info.path.size(); x++) {
    std::visit(visitor, info.path[x]);
  }
  return info;
}

class PathBuilder {
 public:
  explicit PathBuilder(bool startNullable) : nullableInParent_(startNullable) {}
  template <typename T>
  void addTerminalInfo(const T& array) {
    info_.leafIsNullable = nullableInParent_;
    if (nullableInParent_) {
      info_.maxDefLevel++;
    }
    // We don't use null_count() because if the null_count isn't known.
    // And the array does in fact contain nulls, we will end up.
    // Traversing the null bitmap twice (once here and once when calculating.
    // Rep/def levels).
    if (lazyNoNulls(array)) {
      info_.path.emplace_back(AllPresentTerminalNode{info_.maxDefLevel});
    } else if (lazyNullCount(array) == array.length()) {
      info_.path.emplace_back(AllNullsTerminalNode(info_.maxDefLevel - 1));
    } else {
      info_.path.emplace_back(NullableTerminalNode(
          array.null_bitmap_data(), array.offset(), info_.maxDefLevel));
    }
    info_.primitiveArray = std::make_shared<T>(array.data());
    paths_.push_back(fixup(info_));
  }

  template <typename T>
  ::arrow::enable_if_t<std::is_base_of<::arrow::FlatArray, T>::value, Status>
  Visit(const T& array) {
    addTerminalInfo(array);
    return Status::OK();
  }

  template <typename T>
  ::arrow::enable_if_t<
      std::is_same<::arrow::ListArray, T>::value ||
          std::is_same<::arrow::LargeListArray, T>::value,
      Status>
  Visit(const T& array) {
    maybeAddNullable(array);
    // Increment necessary due to empty lists.
    info_.maxDefLevel++;
    info_.maxRepLevel++;
    // Raw_value_offsets() accounts for any slice offset.
    ListPathNode<VarRangeSelector<typename T::offset_type>> Node(
        VarRangeSelector<typename T::offset_type>{array.raw_value_offsets()},
        info_.maxRepLevel,
        info_.maxDefLevel - 1);
    info_.path.emplace_back(std::move(Node));
    nullableInParent_ = array.list_type()->value_field()->nullable();
    return VisitInline(*array.values());
  }

  Status Visit(const ::arrow::DictionaryArray& array) {
    // Only currently handle DictionaryArray where the dictionary is a.
    // Primitive type.
    if (array.dict_type()->value_type()->num_fields() > 0) {
      return Status::NotImplemented(
          "Writing DictionaryArray with nested dictionary "
          "type not yet supported");
    }
    if (array.dictionary()->null_count() > 0) {
      return Status::NotImplemented(
          "Writing DictionaryArray with null encoded in dictionary "
          "type not yet supported");
    }
    addTerminalInfo(array);
    return Status::OK();
  }

  void maybeAddNullable(const Array& array) {
    if (!nullableInParent_) {
      return;
    }
    info_.maxDefLevel++;
    // We don't use null_count() because if the null_count isn't known.
    // And the array does in fact contain nulls, we will end up.
    // Traversing the null bitmap twice (once here and once when calculating.
    // Rep/def levels). Because this isn't terminal this might not be.
    // The right decision for structs that share the same nullable.
    // Parents.
    if (lazyNoNulls(array)) {
      // Don't add anything because there won't be any point checking.
      // Null values for the array.  There will always be at least.
      // One more array to handle nullability.
      return;
    }
    if (lazyNullCount(array) == array.length()) {
      info_.path.emplace_back(AllNullsTerminalNode(info_.maxDefLevel - 1));
      return;
    }
    info_.path.emplace_back(NullableNode(
        array.null_bitmap_data(),
        array.offset(),
        /* def_level_if_null = */ info_.maxDefLevel - 1));
  }

  Status VisitInline(const Array& array);

  Status Visit(const ::arrow::MapArray& array) {
    return Visit(static_cast<const ::arrow::ListArray&>(array));
  }

  Status Visit(const ::arrow::StructArray& array) {
    maybeAddNullable(array);
    PathInfo infoBackup = info_;
    for (int x = 0; x < array.num_fields(); x++) {
      nullableInParent_ = array.type()->field(x)->nullable();
      RETURN_NOT_OK(VisitInline(*array.field(x)));
      info_ = infoBackup;
    }
    return Status::OK();
  }

  Status Visit(const ::arrow::FixedSizeListArray& array) {
    maybeAddNullable(array);
    int32_t listSize = array.list_type()->list_size();
    // Technically we could encode fixed size lists with two level encodings.
    // But since we always use 3 level encoding we increment def levels as.
    // Well.
    info_.maxDefLevel++;
    info_.maxRepLevel++;
    info_.path.emplace_back(FixedSizeListNode(
        FixedSizedRangeSelector{listSize},
        info_.maxRepLevel,
        info_.maxDefLevel));
    nullableInParent_ = array.list_type()->value_field()->nullable();
    if (array.offset() > 0) {
      return VisitInline(*array.values()->Slice(array.value_offset(0)));
    }
    return VisitInline(*array.values());
  }

  Status Visit(const ::arrow::ExtensionArray& array) {
    return VisitInline(*array.storage());
  }

#define NOT_IMPLEMENTED_VISIT(ArrowTypePrefix)                          \
  Status Visit(const ::arrow::ArrowTypePrefix##Array& array) {          \
    return Status::NotImplemented(                                      \
        "Level generation for " #ArrowTypePrefix " not supported yet"); \
  }

  // Types not yet supported in Parquet.
  NOT_IMPLEMENTED_VISIT(Union)
  NOT_IMPLEMENTED_VISIT(RunEndEncoded);
  NOT_IMPLEMENTED_VISIT(ListView);
  NOT_IMPLEMENTED_VISIT(LargeListView);

#undef NOT_IMPLEMENTED_VISIT
  std::vector<PathInfo>& paths() {
    return paths_;
  }

 private:
  PathInfo info_;
  std::vector<PathInfo> paths_;
  bool nullableInParent_;
};

Status PathBuilder::VisitInline(const Array& array) {
  return ::arrow::VisitArrayInline(array, this);
}

#undef RETURN_IF_ERROR
} // namespace

class MultipathLevelBuilderImpl : public MultipathLevelBuilder {
 public:
  MultipathLevelBuilderImpl(
      std::shared_ptr<::arrow::ArrayData> data,
      std::unique_ptr<PathBuilder> pathBuilder)
      : rootRange_{0, data->length},
        data_(std::move(data)),
        pathBuilder_(std::move(pathBuilder)) {}

  int getLeafCount() const override {
    return static_cast<int>(pathBuilder_->paths().size());
  }

  ::arrow::Status write(
      int leafIndex,
      ArrowWriteContext* context,
      CallbackFunction writeLeafCallback) override {
    if (ARROW_PREDICT_FALSE(leafIndex < 0 || leafIndex >= getLeafCount())) {
      return Status::Invalid(
          "Column index out of bounds (got ",
          leafIndex,
          ", should be "
          "between 0 and ",
          getLeafCount(),
          ")");
    }

    return writePath(
        rootRange_,
        &pathBuilder_->paths()[leafIndex],
        context,
        std::move(writeLeafCallback));
  }

 private:
  ElementRange rootRange_;
  // Reference holder to ensure the data stays valid.
  std::shared_ptr<::arrow::ArrayData> data_;
  std::unique_ptr<PathBuilder> pathBuilder_;
};

// Static.
::arrow::Result<std::unique_ptr<MultipathLevelBuilder>>
MultipathLevelBuilder::make(
    const ::arrow::Array& array,
    bool arrayFieldNullable) {
  auto constructor = std::make_unique<PathBuilder>(arrayFieldNullable);
  RETURN_NOT_OK(::arrow::VisitArrayInline(array, constructor.get()));
  return std::make_unique<MultipathLevelBuilderImpl>(
      array.data(), std::move(constructor));
}

// Static.
Status MultipathLevelBuilder::write(
    const Array& array,
    bool arrayFieldNullable,
    ArrowWriteContext* context,
    MultipathLevelBuilder::CallbackFunction callback) {
  ARROW_ASSIGN_OR_RAISE(
      std::unique_ptr<MultipathLevelBuilder> Builder,
      MultipathLevelBuilder::make(array, arrayFieldNullable));
  for (int leafIdx = 0; leafIdx < Builder->getLeafCount(); leafIdx++) {
    RETURN_NOT_OK(Builder->write(leafIdx, context, callback));
  }
  return Status::OK();
}

} // namespace facebook::velox::parquet::arrow::arrow
