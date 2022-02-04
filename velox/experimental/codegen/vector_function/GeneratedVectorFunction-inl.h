/*
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
#include <thread>
#include <type_traits>
#include <utility>
#include "f4d/experimental/codegen/vector_function/ConcatExpression-inl.h"
#include "f4d/experimental/codegen/vector_function/Perf.h"
#include "f4d/experimental/codegen/vector_function/VectorReader-inl.h"
#include "f4d/expression/VectorFunction.h"
#include "f4d/vector/SelectivityVector.h"

namespace facebook {
namespace f4d {
namespace codegen {

template <bool isDefaultNull>
struct InputReaderConfig {
  static constexpr bool isWriter_ = false;
  // when false, the reader will never read a nullvalue
  static constexpr bool mayReadNull_ = !isDefaultNull;

  // irrelevent for reader
  static constexpr bool mayWriteNull_ = false;
  static constexpr bool intializedWithNullSet_ = false;

  constexpr static bool inputStringBuffersShared = false;
  constexpr static bool constantStringBuffersShared = false;
};

template <bool isDefaultNull, bool isDefaultNullStrict>
struct OutputReaderConfig {
  static constexpr bool isWriter_ = true;
  // true means set to null, false means not null
  static constexpr bool intializedWithNullSet_ =
      isDefaultNull || isDefaultNullStrict;
  // when true, the reader will never reveive a null value to write
  static constexpr bool mayWriteNull_ = !isDefaultNullStrict;
  static constexpr bool mayReadNull_ = mayWriteNull_;

  constexpr static bool inputStringBuffersShared = true;
  constexpr static bool constantStringBuffersShared = false;
};

// #define DEBUG_CODEGEN
// Debugging/printing functions
namespace {
template <typename ReferenceType>
std::ostream& printReference(
    std::ostream& out,
    const ReferenceType& reference) {
  auto value = reference.toOptionalValueType();

  std::stringstream referenceStream;
  if (!value.has_value()) {
    referenceStream << "Null";
  } else {
    referenceStream << value.value();
  }
  const std::string format =
      "ReferenceType [ rowIndex {}, vector address {}, value {} ]";
  out << fmt::format(
      format,
      reference.rowIndex_,
      static_cast<void*>(reference.reader_.vector_.get()),
      referenceStream.str());
  return out;
}

template <typename ValType>
std::ostream& printValue(std::ostream& out, const ValType& value) {
  std::stringstream referenceStream;
  if (!value.has_value()) {
    referenceStream << "Null";
  } else {
    referenceStream << value.value();
  }
  const std::string format = "[ value {} ]";
  out << fmt::format(format, referenceStream.str());
  return out;
}

template <typename... Types>
void printTuple(const std::tuple<Types...>& tuple) {
  auto printElement = [](/*auto && first,*/ auto&&... elements) {
    (printValue(std::cerr, elements), ...);
  };
  std::apply(printElement, tuple);
};

// HelperFunction
template <typename... Types>
std::ostream& operator<<(std::ostream& out, const std::tuple<Types...>& tuple) {
  auto printElement = [&out](auto&& first, auto&&... elements) {
    out << std::forward<decltype(first)>(first);
    ((out << ", " << std::forward<decltype(elements)>(elements)), ...);
  };
  std::apply(printElement, tuple);
  return out;
};
} // namespace

/// Base class for all generated function
/// This add a new apply method to support multiple output;
class GeneratedVectorFunctionBase : public facebook::f4d::exec::VectorFunction {
 public:
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Woverloaded-virtual"
  // Multiple output apply method.
  virtual void apply(
      const facebook::f4d::SelectivityVector& rows,
      std::vector<facebook::f4d::VectorPtr>& args,
      [[maybe_unused]] facebook::f4d::exec::Expr* caller,
      facebook::f4d::exec::EvalCtx* context,
      std::vector<facebook::f4d::VectorPtr>& results) const = 0;
#pragma clang diagnostic pop

  void setRowType(const std::shared_ptr<const RowType>& rowType) {
    rowType_ = rowType;
  }

 protected:
  std::shared_ptr<const RowType> rowType_;
};

// helper templates/functions
namespace {
template <
    typename Func,
    typename T,
    std::size_t... Indices,
    typename... ExtraArgs>
void applyLambdaToVector(
    Func func,
    std::vector<T>& args,
    [[maybe_unused]] const std::index_sequence<Indices...>& unused,
    ExtraArgs&... extraArgs) {
  (func(args[Indices], extraArgs...), ...);
}

template <size_t... Indices>
constexpr bool indexIn(const size_t i, const std::index_sequence<Indices...>&) {
  return ((i == Indices) || ...);
}

template <size_t... Indices, typename OtherIndices>
constexpr bool isSubset(
    const std::index_sequence<Indices...>&,
    const OtherIndices& other) {
  return (indexIn(Indices, other) && ...);
}

} // namespace

///
/// \tparam GeneratedCode
template <typename GeneratedCodeConfig>
class GeneratedVectorFunction : public GeneratedVectorFunctionBase {
 public:
  using GeneratedCode = typename GeneratedCodeConfig::GeneratedCodeClass;
  using OutputTypes = typename GeneratedCode::VeloxOutputType;
  using InputTypes = typename GeneratedCode::VeloxInputType;

  // only useful when there's filter
  using FilterInputIndices =
      typename GeneratedCode::template InputMapAtIndex<0>;

  // projection input is at index 0 when no filter and 1 when has filter
  using ProjectionInputIndices =
      typename GeneratedCode::template InputMapAtIndex<
          GeneratedCode::hasFilter>;

  constexpr static bool isFilterDefaultNull =
      GeneratedCode::hasFilter && GeneratedCodeConfig::isFilterDefaultNull;

  /// when it's filter default null and projection attributes is
  /// a subset of filter attributes, in this case even if it's projection
  /// default null we can ignore it (and optimize some computations later),
  /// because the null bits of the projection attributes is going to be filtered
  /// out at filter stage already
  constexpr static bool isProjectionDefaultNull =
      (GeneratedCodeConfig::isProjectionDefaultNull ||
       GeneratedCodeConfig::isProjectionDefaultNullStrict) &&
      (!isFilterDefaultNull ||
       !isSubset(ProjectionInputIndices{}, FilterInputIndices{}));

  GeneratedVectorFunction() : generatedExpression_() {}

 public:
  virtual void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      exec::Expr* caller,
      exec::EvalCtx* context,
      VectorPtr* result) const override {
    VELOX_CHECK(result != nullptr);
    VELOX_CHECK(rowType_ != nullptr);
    VELOX_CHECK(
        (*result == nullptr) or (result->get()->as<RowVector>() != nullptr));

    // all-constant expressions shouldn't be compiled
    VELOX_CHECK(args.size() > 0);

    BaseVector::ensureWritable(rows, rowType_, context->pool(), result);

    // TODO: We should probably move this loop inside ensureWritable
    for (size_t columnIndex = 0; columnIndex < rowType_->size();
         ++columnIndex) {
      BaseVector::ensureWritable(
          rows,
          rowType_->childAt(columnIndex),
          context->pool(),
          &(result->get()->as<RowVector>()->childAt(columnIndex)));
    }

    // Constuct nulls
    for (auto& child : result->get()->as<RowVector>()->children()) {
      child->setCodegenOutput();
      if constexpr (isProjectionDefaultNull) {
        // preset all with nulls
        child->addNulls(nullptr, rows);
      } else {
        // preset all not nulls
        child->mutableRawNulls();
      }
    }

    (*result)->setCodegenOutput();

    VELOX_CHECK(result->get()->as<RowVector>() != nullptr);

    // Shared string input buffer
    // TODO: write now we are sharing everything, this not ideal. We should do
    // a static analysis reachability pass and acquire shared buffers
    // accordingly to reduce memory lifetime.
    for (auto& arg : args) {
      if (arg->type()->kind() == TypeKind::VARCHAR) {
        for (size_t columnIndex = 0; columnIndex < rowType_->size();
             ++columnIndex) {
          // Ensures that the results vectors are nullables.
          if (result->get()
                  ->as<RowVector>()
                  ->childAt(columnIndex)
                  ->type()
                  ->kind() == TypeKind::VARCHAR) {
            result->get()
                ->as<RowVector>()
                ->childAt(columnIndex)
                ->template asFlatVector<StringView>()
                ->acquireSharedStringBuffers(arg.get());
          }
        }
      }
    }

    apply(
        rows,
        args,
        caller,
        context,
        result->get()->as<RowVector>()->children());
  }

  void apply(
      const facebook::f4d::SelectivityVector& rows,
      std::vector<facebook::f4d::VectorPtr>& args,
      [[maybe_unused]] facebook::f4d::exec::Expr* caller,
      facebook::f4d::exec::EvalCtx* context,
      std::vector<facebook::f4d::VectorPtr>& results) const override {
    VELOX_CHECK(rowType_ != nullptr);
    VELOX_CHECK(context != nullptr);
    VELOX_CHECK(results.size() == rowType_->size());

    /// in some (constexpr) cases copies aren't needed, can we assume compilers
    /// can pick up those cases and avoid instantiating here?
    auto filterSel = rows, projectionSel = rows;

    auto deselectNull =
        [&rows](const VectorPtr& arg, SelectivityVector& rowsNotNull) {
          if (arg->mayHaveNulls() && arg->getNullCount() != 0) {
            rowsNotNull.deselectNulls(
                arg->flatRawNulls(rows), rows.begin(), rows.end());
          }
        };

    // build filter selectivity vector if it's filter default null
    if constexpr (isFilterDefaultNull) {
      applyLambdaToVector(deselectNull, args, FilterInputIndices{}, filterSel);
    }

    // build projection selectivity vector if it's projection default null
    if constexpr (isProjectionDefaultNull) {
      applyLambdaToVector(
          deselectNull, args, ProjectionInputIndices{}, projectionSel);
    }

    auto inReaders = createReaders(
        args,
        std::make_integer_sequence<
            std::size_t,
            std::tuple_size_v<InputTypes>>{});

    auto outReaders = createWriters(
        results,
        std::make_integer_sequence<
            std::size_t,
            std::tuple_size_v<OutputTypes>>{});

#ifdef DEBUG_CODEGEN
#define DEBUG_PRINT()       \
  std::cerr << "Input : ";  \
  printTuple(inputs);       \
  std::cerr << std::endl;   \
  std::cerr << "Output : "; \
  printTuple(outputs);      \
  std::cerr << std::endl
#else
#define DEBUG_PRINT()
#endif
    // some duplication for better readability here
    if constexpr (!GeneratedCode::hasFilter) {
      // projection only
      auto computeRow = [&inReaders, &outReaders, this](size_t rowIndex) {
        auto inputs = makeInputTuple(rowIndex, inReaders);
        auto outputs = makeOutputTuple(rowIndex, outReaders);
        // Apply the generated function.
        generatedExpression_(inputs, outputs);
        DEBUG_PRINT();
        return true;
      };
      {
        Perf perf;
        projectionSel.applyToSelected(computeRow);
      }
    } else {
      // merged filter + projection
      size_t outIndex = 0;
      auto computeRow =
          [&inReaders, &outReaders, this, &outIndex, &projectionSel](
              size_t rowIndex) {
            auto inputs = makeInputTuple(rowIndex, inReaders);
            auto outputs = makeOutputTuple(outIndex, outReaders);
            bool projectionSelected = 1;
            if constexpr (isProjectionDefaultNull) {
              projectionSelected = projectionSel.isValid(rowIndex);
            }
            // Apply the generated function.
            if (generatedExpression_(inputs, outputs, projectionSelected)) {
              outIndex++;
            }
            DEBUG_PRINT();
            return true;
          };
      {
        Perf perf;
        filterSel.applyToSelected(computeRow);
      }
      if (outIndex != args[0]->size()) {
        for (auto& child : results) {
          child->resize(outIndex);
        }
      }
    }
#undef DEBUG_PRINT
  }

  // TODO: Missing implementation;
  virtual bool isDeterministic() const override {
    return true;
  }

  virtual bool isDefaultNullBehavior() const override {
    return false;
  }

  // Builds a new Instance
  static std::shared_ptr<GeneratedVectorFunction<GeneratedCode>> newInstance(
      const std::vector<TypePtr>& inputTypes,
      const std::vector<TypePtr>& outputTypes) {
    return std::make_shared<GeneratedVectorFunction<GeneratedCode>>(
        inputTypes, outputTypes);
  }

 private:
  template <typename... SQLImplType>
  auto toVectorType(const std::tuple<SQLImplType...>&) const {
    return std::vector<TypePtr>{std::make_shared<SQLImplType>()...};
  }

  template <size_t... Indices>
  auto createReaders(
      std::vector<facebook::f4d::VectorPtr>& args,
      const std::index_sequence<Indices...>&) const {
    /// situations where we know that we won't read null value:
    /// 1. if an attribute is in filter, and filter condition is default null
    /// 2. if an attribute is ONLY in projection, and projection is default null
    return std::make_tuple(
        VectorReader<
            std::tuple_element_t<Indices, InputTypes>,
            InputReaderConfig<
                (isFilterDefaultNull &&
                 indexIn(Indices, FilterInputIndices{})) ||
                (isProjectionDefaultNull &&
                 indexIn(Indices, ProjectionInputIndices{}) &&
                 !indexIn(Indices, FilterInputIndices{}))>>(args[Indices])...);
  }

  template <size_t... Indices>
  auto createWriters(
      std::vector<facebook::f4d::VectorPtr>& results,
      const std::index_sequence<Indices...>&) const {
    return std::make_tuple(
        VectorReader<
            std::tuple_element_t<Indices, OutputTypes>,
            OutputReaderConfig<
                GeneratedCodeConfig::isProjectionDefaultNull,
                GeneratedCodeConfig::isProjectionDefaultNullStrict>>(
            results[Indices])...);
  }

  template <typename Func, typename Tuple, size_t... Index>
  inline auto applyToTuple(
      Func&& fun,
      Tuple&& args,
      [[maybe_unused]] const std::index_sequence<Index...>& indices =
          std::index_sequence<Index...>{}) const {
    /// TODO:  This might need to be improved in case func return ref types.
    return std::make_tuple(fun(std::get<Index>(std::forward<Tuple>(args)))...);
  }

  template <typename Tuple>
  inline auto makeInputTuple(size_t& rowIndex, Tuple&& inReaders) const {
    return applyToTuple(
        [rowIndex](auto&& reader) -> auto {
          return (const typename std::remove_reference_t<
                  decltype(reader)>::InputType)reader[rowIndex];
        },
        inReaders,
        std::make_integer_sequence<
            std::size_t,
            std::tuple_size_v<InputTypes>>{});
  }

  template <typename Tuple>
  inline auto makeOutputTuple(size_t& rowIndex, Tuple&& outReaders) const {
    return applyToTuple(
        [rowIndex](auto&& reader) ->
        typename std::remove_reference_t<decltype(reader)>::PointerType {
          return reader[rowIndex];
        },
        outReaders,
        std::make_integer_sequence<
            std::size_t,
            std::tuple_size_v<OutputTypes>>{});
  }

  mutable GeneratedCode generatedExpression_;
};

} // namespace codegen
} // namespace f4d
} // namespace facebook
