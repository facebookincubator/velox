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

#include <concepts>

#include "velox/exec/Aggregate.h"
#include "velox/expression/VectorReaders.h"
#include "velox/expression/VectorWriters.h"
#include "velox/type/SimpleFunctionApi.h"

namespace facebook::velox::exec {

// AggregateInputType is similar to Row but allows Variadic as the last element.
// Unlike Row, this type is not meant to be serialized or used as a data type
// in Velox vectors - it's only used for type deduction in
// SimpleAggregateAdapter.
template <typename... T>
struct AggregateInputType {
  template <size_t idx>
  using type_at = typename std::tuple_element<idx, std::tuple<T...>>::type;

  static const size_t size_ = sizeof...(T);

  // Verify that Variadic, if present, is only at the last position.
  static_assert(
      []() constexpr {
        if constexpr (sizeof...(T) <= 1) {
          return true;
        } else {
          // Check that no element except possibly the last is Variadic
          constexpr bool checks[] = {!isVariadicType<T>::value...};
          for (size_t i = 0; i < sizeof...(T) - 1; ++i) {
            if (!checks[i]) {
              return false;
            }
          }
          return true;
        }
      }(),
      "Variadic can only appear at the end of aggregation input type");

 private:
  AggregateInputType() {}
};

// The writer type of T used in simple UDAF interface. An instance of
// out_type allows writing one row into the output vector.
template <typename T>
using out_type = typename VectorExec::template resolver<T>::out_type;

// The reader type of T used in simple UDAF interface. An instance of
// arg_type allows reading one row from the input vector. This is used for UDAFs
// that have the default null behavior.
template <typename T>
using arg_type = typename VectorExec::template resolver<T>::in_type;

// The reader type of T used in simple UDAF interface. An instance of
// arg_type allows reading one row from the input vector. This is used for UDAFs
// that have non-default null behavior.
template <typename T>
using optional_arg_type = OptionalAccessor<T>;

namespace detail {

// Whether 'F' has default null behavior: true unless 'F' sets
// default_null_behavior_ to false.
template <typename F>
constexpr bool hasDefaultNullBehavior() {
  if constexpr (requires { F::default_null_behavior_; }) {
    return F::default_null_behavior_;
  } else {
    return true;
  }
}

template <typename Accumulator, typename... T>
concept AcceptsInput =
    requires(Accumulator& accumulator, HashStringAllocator* allocator) {
      accumulator.addInput(allocator, std::declval<arg_type<T>>()...);
    };

template <typename Accumulator, typename... T>
concept AcceptsOptionalInput =
    requires(Accumulator& accumulator, HashStringAllocator* allocator) {
      {
        accumulator.addInput(allocator, std::declval<optional_arg_type<T>>()...)
      } -> std::convertible_to<bool>;
    };

// Accept a pointer to Row<T...> or AggregateInputType<T...> when
// 'Accumulator' accepts the arguments 'T...'. Unpacking the arguments here
// lets a failed check name the addInput call that does not compile.
template <
    typename Accumulator,
    template <typename...> class TInput,
    typename... T>
  requires AcceptsInput<Accumulator, T...>
void checkInput(TInput<T...>*);

template <
    typename Accumulator,
    template <typename...> class TInput,
    typename... T>
  requires AcceptsOptionalInput<Accumulator, T...>
void checkOptionalInput(TInput<T...>*);

} // namespace detail

/// What a write method returns: bool, true when it wrote a non-null value and
/// false when the result is null; or void, when the result is never null.
template <typename T>
concept AggregateWriteResult = std::same_as<T, bool> || std::same_as<T, void>;

/// The accumulator of a simple aggregation function 'F' with default null
/// behavior. T1, ... are the arguments of F::InputType. Groups start null.
///
/// - addInput(allocator, arg_type<T1>, ...) adds one row of raw input. Rows
///   with a null argument never reach it; for a Variadic argument, a null in
///   any of its elements skips the row too.
/// - combine(allocator, arg_type<IntermediateType>) adds one intermediate
///   result. Null intermediate results never reach it.
/// - writeIntermediateResult(out) and writeFinalResult(out) write the
///   accumulator out. They run only for non-null groups; null groups produce
///   null without a call. See SimpleAggregateFunction for what they must
///   preserve.
///
/// After addInput() or combine(), the group is non-null.
template <typename F, typename Accumulator = typename F::AccumulatorType>
concept DefaultNullAccumulator =
    requires(typename F::InputType* input) {
      detail::checkInput<Accumulator>(input);
    } &&
    requires(
        Accumulator& accumulator,
        HashStringAllocator* allocator,
        out_type<typename F::IntermediateType>& intermediateOut,
        out_type<typename F::OutputType>& finalOut) {
      accumulator.combine(
          allocator, std::declval<arg_type<typename F::IntermediateType>>());
      {
        accumulator.writeIntermediateResult(intermediateOut)
      } -> AggregateWriteResult;
      { accumulator.writeFinalResult(finalOut) } -> AggregateWriteResult;
    };

/// The accumulator of a simple aggregation function 'F' that sets
/// default_null_behavior_ to false. T1, ... are the arguments of F::InputType.
/// Groups start null.
///
/// - addInput(allocator, optional_arg_type<T1>, ...) adds one row of raw
///   input, nulls included.
/// - combine(allocator, optional_arg_type<IntermediateType>) adds one
///   intermediate result, nulls included.
///
/// Both return whether the group is non-null after the call; returning false
/// leaves a group that is already non-null as it is.
///
/// - writeIntermediateResult(nonNullGroup, out) and writeFinalResult(
///   nonNullGroup, out) write the accumulator out, for every group, with
///   'nonNullGroup' saying whether the group is non-null. See
///   SimpleAggregateFunction for what they must preserve.
template <typename F, typename Accumulator = typename F::AccumulatorType>
concept NonDefaultNullAccumulator =
    requires(typename F::InputType* input) {
      detail::checkOptionalInput<Accumulator>(input);
    } &&
    requires(
        Accumulator& accumulator,
        HashStringAllocator* allocator,
        bool nonNullGroup,
        out_type<typename F::IntermediateType>& intermediateOut,
        out_type<typename F::OutputType>& finalOut) {
      {
        accumulator.combine(
            allocator,
            std::declval<optional_arg_type<typename F::IntermediateType>>())
      } -> std::convertible_to<bool>;
      {
        accumulator.writeIntermediateResult(nonNullGroup, intermediateOut)
      } -> AggregateWriteResult;
      {
        accumulator.writeFinalResult(nonNullGroup, finalOut)
      } -> AggregateWriteResult;
    };

/// A simple aggregation function: a class that SimpleAggregateAdapter turns
/// into an Aggregate. For example, see GeometricMeanAggregate in
/// velox/functions/prestosql/aggregates.
///
/// Required members:
/// - InputType: the raw input arguments, Row<T1, ...>; AggregateInputType<T1,
///   ...> allows a Variadic last argument.
/// - IntermediateType and OutputType: the types of the intermediate and final
///   results.
/// - AccumulatorType: one group's state, constructed for every group, null or
///   not, with (HashStringAllocator*, F*). F* is the function object shared by
///   all groups; it carries function-level state such as the result type.
///   The methods are DefaultNullAccumulator, or NonDefaultNullAccumulator when
///   default_null_behavior_ is false.
///
/// The write methods must leave the accumulator representing the same input.
/// Window functions over a frame that grows row by row, such as ROWS BETWEEN
/// UNBOUNDED PRECEDING AND CURRENT ROW, add each new row to one accumulator
/// and write the result after every row. Streaming systems that embed Velox
/// aggregates do the same over unbounded input. A write may rearrange the
/// accumulator's contents, as long as later calls behave as if no write had
/// happened.
///
/// Optional members of F:
/// - default_null_behavior_ (default true) and is_reducing_ (default true:
///   the state is smaller than its input; set to false when the state grows
///   with the input, as when collecting values into an array).
/// - initialize(step, argTypes, resultType[, config]): called once, at
///   construction.
/// - setConstantInputs(constantInputs): called at most once, before any
///   input, with one entry per raw argument, null for non-constant ones.
///   Called only in steps that read raw input, partial and single
///   aggregation; steps that merge intermediate results have no raw
///   arguments.
/// - static bool toIntermediate(out, args...): converts one raw input row to
///   an intermediate result without an accumulator, for plans that abandon
///   partial aggregation. Returns false for a null result. 'args' are
///   arg_type<T1>, ... with default null behavior, where rows with a null
///   argument produce null without a call; otherwise optional_arg_type<T1>,
///   ....
///
/// Optional members of AccumulatorType:
/// - is_fixed_size_ (default true): false when the accumulator allocates from
///   the HashStringAllocator; the adapter then tracks each group's memory.
/// - use_external_memory_ (default false): true when the accumulator uses
///   memory that Velox does not track.
/// - is_aligned_ (default false): true when the accumulator requires
///   alignof(AccumulatorType) alignment.
/// - destroy(allocator): releases the memory of a non-null group. When
///   destroy() is defined, the destructor never runs, so memory the
///   constructor allocates leaks; without it, the destructor runs for every
///   group.
template <typename F>
concept SimpleAggregateFunction = std::default_initializable<F> &&
    requires {
      typename F::InputType;
      typename F::IntermediateType;
      typename F::OutputType;
      typename F::AccumulatorType;
    } &&
    std::constructible_from<typename F::AccumulatorType,
                            HashStringAllocator*,
                            F*> &&
    ((detail::hasDefaultNullBehavior<F>() && DefaultNullAccumulator<F>) ||
     (!detail::hasDefaultNullBehavior<F>() && NonDefaultNullAccumulator<F>)) &&
    (!requires { &F::setConstantInputs; } ||
     requires(F& function, const std::vector<VectorPtr>& constantInputs) {
       function.setConstantInputs(constantInputs);
     });

template <SimpleAggregateFunction FUNC>
class SimpleAggregateAdapter : public Aggregate {
 public:
  explicit SimpleAggregateAdapter(
      core::AggregationNode::Step step,
      const std::vector<TypePtr>& argTypes,
      TypePtr resultType,
      const core::QueryConfig* config = nullptr)
      : Aggregate(std::move(resultType)), fn_{std::make_unique<FUNC>()} {
    if constexpr (support_initialize_with_config_) {
      VELOX_CHECK_NOT_NULL(
          config,
          "QueryConfig is required to initialize this aggregate function.");
      fn_->initialize(step, argTypes, resultType_, *config);
    } else if constexpr (support_initialize_) {
      fn_->initialize(step, argTypes, resultType_);
    }
  }

  // Detect the optional members described in SimpleAggregateFunction.
  template <typename T, typename = void>
  struct accumulator_is_fixed_size : std::true_type {};

  template <typename T>
  struct accumulator_is_fixed_size<T, std::void_t<decltype(T::is_fixed_size_)>>
      : std::integral_constant<bool, T::is_fixed_size_> {};

  template <typename T, typename = void>
  struct aggregate_is_reducing : std::true_type {};

  template <typename T>
  struct aggregate_is_reducing<T, std::void_t<decltype(T::is_reducing_)>>
      : std::integral_constant<bool, T::is_reducing_> {};

  template <typename T, typename = void>
  struct accumulator_use_external_memory : std::false_type {};

  template <typename T>
  struct accumulator_use_external_memory<
      T,
      std::void_t<decltype(T::use_external_memory_)>>
      : std::integral_constant<bool, T::use_external_memory_> {};

  // Whether the accumulator type defines its destroy() method or not. If it is
  // defined, we call the accumulator's destroy() in
  // SimpleAggregateAdapter::destroy().
  template <typename T, typename = void>
  struct accumulator_custom_destroy : std::false_type {};

  template <typename T>
  struct accumulator_custom_destroy<T, std::void_t<decltype(&T::destroy)>>
      : std::true_type {};

  // Whether the function defines its toIntermediate() method or not. If it is
  // defined, SimpleAggregateAdapter::supportToIntermediate() returns true.
  // Otherwise, SimpleAggregateAdapter::supportToIntermediate() returns false
  // and SimpleAggregateAdapter::toIntermediate() is empty.
  template <typename T, typename = void>
  struct support_to_intermediate : std::false_type {};

  template <typename T>
  struct support_to_intermediate<T, std::void_t<decltype(&T::toIntermediate)>>
      : std::true_type {};

  template <typename T, typename = void>
  struct support_fill_empty_intermediate : std::false_type {};

  template <typename T>
  struct support_fill_empty_intermediate<
      T,
      std::void_t<decltype(&T::fillEmptyIntermediate)>> : std::true_type {};

  template <typename T, typename = void>
  struct support_initialize : std::false_type {};

  template <typename T>
  struct support_initialize<T, std::void_t<decltype(&T::initialize)>>
      : std::true_type {};

  // Whether the function defines an initialize() method that accepts a
  // QueryConfig parameter. If so, the config is forwarded during construction.
  template <typename T, typename = void>
  struct support_initialize_with_config : std::false_type {};

  template <typename T>
  struct support_initialize_with_config<
      T,
      std::void_t<decltype(std::declval<T&>().initialize(
          std::declval<core::AggregationNode::Step>(),
          std::declval<const std::vector<TypePtr>&>(),
          std::declval<const TypePtr&>(),
          std::declval<const core::QueryConfig&>()))>> : std::true_type {};

  // Whether the function defines setConstantInputs(). If it does, the adapter
  // forwards Aggregate::setConstantInputs() to it.
  template <typename T, typename = void>
  struct support_set_constant_inputs : std::false_type {};

  template <typename T>
  struct support_set_constant_inputs<
      T,
      std::void_t<decltype(std::declval<T&>().setConstantInputs(
          std::declval<const std::vector<VectorPtr>&>()))>> : std::true_type {};

  // Whether the accumulator requires aligned access. If it is defined,
  // SimpleAggregateAdapter::accumulatorAlignmentSize() returns
  // alignof(typename FUNC::AccumulatorType).
  // Otherwise, SimpleAggregateAdapter::accumulatorAlignmentSize() returns
  // Aggregate::accumulatorAlignmentSize(), with a default value of 1.
  template <typename T, typename = void>
  struct accumulator_is_aligned : std::false_type {};

  template <typename T>
  struct accumulator_is_aligned<T, std::void_t<decltype(T::is_aligned_)>>
      : std::integral_constant<bool, T::is_aligned_> {};

  static constexpr bool aggregate_default_null_behavior_ =
      detail::hasDefaultNullBehavior<FUNC>();

  static constexpr bool accumulator_is_fixed_size_ =
      accumulator_is_fixed_size<typename FUNC::AccumulatorType>::value;

  static constexpr bool aggregate_is_reducing_ =
      aggregate_is_reducing<FUNC>::value;

  static constexpr bool accumulator_use_external_memory_ =
      accumulator_use_external_memory<typename FUNC::AccumulatorType>::value;

  static constexpr bool accumulator_custom_destroy_ =
      accumulator_custom_destroy<typename FUNC::AccumulatorType>::value;

  static constexpr bool support_to_intermediate_ =
      support_to_intermediate<FUNC>::value;

  static constexpr bool support_fill_empty_intermediate_ =
      support_fill_empty_intermediate<FUNC>::value;

  static constexpr bool support_initialize_ = support_initialize<FUNC>::value;

  static constexpr bool support_initialize_with_config_ =
      support_initialize_with_config<FUNC>::value;

  static constexpr bool accumulator_is_aligned_ =
      accumulator_is_aligned<typename FUNC::AccumulatorType>::value;

  static constexpr bool support_set_constant_inputs_ =
      support_set_constant_inputs<FUNC>::value;

  bool isFixedSize() const override {
    return accumulator_is_fixed_size_;
  }

  bool isReducing() const override {
    return aggregate_is_reducing_;
  }

  bool accumulatorUsesExternalMemory() const override {
    return accumulator_use_external_memory_;
  }

  int32_t accumulatorFixedWidthSize() const override {
    return sizeof(typename FUNC::AccumulatorType);
  }

  int32_t accumulatorAlignmentSize() const override {
    if constexpr (accumulator_is_aligned_) {
      return alignof(typename FUNC::AccumulatorType);
    }
    return Aggregate::accumulatorAlignmentSize();
  }

  void setConstantInputs(
      const std::vector<VectorPtr>& constantInputs) override {
    if constexpr (support_set_constant_inputs_) {
      fn_->setConstantInputs(constantInputs);
    } else {
      Aggregate::setConstantInputs(constantInputs);
    }
  }

  // Add raw input to accumulators. If the simple aggregation function has
  // default null behavior, input rows that has nulls are skipped. Otherwise,
  // the accumulator type's addInput() method handles null inputs.
  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    if (inputDecoded_.size() < args.size()) {
      inputDecoded_.resize(args.size());
    }

    for (column_index_t i = 0; i < args.size(); ++i) {
      inputDecoded_[i].decode(*args[i], rows);
    }

    addRawInputImpl(groups, rows);
  }

  // Similar to addRawInput, but add inputs to one single accumulator.
  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    if (inputDecoded_.size() < args.size()) {
      inputDecoded_.resize(args.size());
    }

    for (column_index_t i = 0; i < args.size(); ++i) {
      inputDecoded_[i].decode(*args[i], rows);
    }

    addSingleGroupRawInputImpl(group, rows);
  }

  bool supportsToIntermediate() const override {
    return support_to_intermediate_;
  }

  bool supportsFillEmptyIntermediate() const override {
    return support_fill_empty_intermediate_;
  }

  void fillEmptyIntermediate(const SelectivityVector& rows, VectorPtr& result)
      const override {
    if constexpr (support_fill_empty_intermediate_) {
      using IntermediateType = typename FUNC::IntermediateType;
      auto* vector =
          result->as<typename TypeToFlatVector<IntermediateType>::type>();
      VectorWriter<IntermediateType> writer;
      writer.init(*vector);
      rows.applyToSelected([&](vector_size_t row) {
        writer.setOffset(row);
        FUNC::fillEmptyIntermediate(writer.current());
        writer.commit();
      });
      writer.finish();
    } else {
      VELOX_UNREACHABLE("fillEmptyIntermediate is not supported");
    }
  }

  void toIntermediate(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      VectorPtr& result) const override {
    if constexpr (support_to_intermediate_) {
      std::vector<DecodedVector> inputDecoded{args.size()};
      for (column_index_t i = 0; i < args.size(); ++i) {
        inputDecoded[i].decode(*args[i], rows);
      }

      toIntermediateImpl(inputDecoded, rows, result);
    } else {
      VELOX_UNREACHABLE(
          "toIntermediate should only be called when support_to_intermediate_ is true.");
    }
  }

  // Add intermediate results to accumulators. If the simple aggregation
  // function has default null behavior, intermediate result rows that has nulls
  // are skipped. Otherwise, the accumulator type's combine() method handles
  // nulls.
  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    intermediateDecoded_.decode(*args[0], rows);

    addIntermediateResultsImpl(groups, rows);
  }

  // Similar to addIntermediateResults, but add intermediate results to one
  // single accumulator.
  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /* mayPushdown */) override {
    intermediateDecoded_.decode(*args[0], rows);

    addSingleGroupIntermediateResultsImpl(group, rows);
  }

  // Extract intermediate results to a vector.
  void extractAccumulators(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    VectorWriter<typename FUNC::IntermediateType> writer;
    auto vector = (*result)
                      ->as<typename TypeToFlatVector<
                          typename FUNC::IntermediateType>::type>();
    vector->resize(numGroups);
    writer.init(*vector);

    for (auto i = 0; i < numGroups; ++i) {
      writer.setOffset(i);
      auto group = value<typename FUNC::AccumulatorType>(groups[i]);

      if constexpr (aggregate_default_null_behavior_) {
        if (isNull(groups[i])) {
          writer.commitNull();
        } else {
          writer.commit(callWrite([&] {
            return group->writeIntermediateResult(writer.current());
          }));
        }
      } else {
        writer.commit(callWrite([&] {
          return group->writeIntermediateResult(
              !isNull(groups[i]), writer.current());
        }));
      }
    }
    writer.finish();
  }

  // Extract final results to a vector.
  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    auto flatResult =
        (*result)
            ->as<typename TypeToFlatVector<typename FUNC::OutputType>::type>();
    flatResult->resize(numGroups);

    VectorWriter<typename FUNC::OutputType> writer;
    writer.init(*flatResult);

    for (auto i = 0; i < numGroups; ++i) {
      writer.setOffset(i);
      auto group = value<typename FUNC::AccumulatorType>(groups[i]);

      if constexpr (aggregate_default_null_behavior_) {
        if (isNull(groups[i])) {
          writer.commitNull();
        } else {
          writer.commit(callWrite(
              [&] { return group->writeFinalResult(writer.current()); }));
        }
      } else {
        writer.commit(callWrite([&] {
          return group->writeFinalResult(!isNull(groups[i]), writer.current());
        }));
      }
    }
    writer.finish();
  }

 protected:
  // Calls 'write', a writeIntermediateResult() or writeFinalResult() call, and
  // returns whether the result is non-NULL: what 'write' returns, or true when
  // it returns void.
  template <typename TWrite>
  static bool callWrite(TWrite&& write) {
    if constexpr (std::is_void_v<std::invoke_result_t<TWrite>>) {
      write();
      return true;
    } else {
      return write();
    }
  }

  void initializeNewGroupsInternal(
      char** groups,
      folly::Range<const vector_size_t*> indices) override {
    setAllNulls(groups, indices);
    for (auto i : indices) {
      new (groups[i] + offset_)
          typename FUNC::AccumulatorType(allocator_, fn_.get());
    }
  }

  void destroyInternal(folly::Range<char**> groups) override {
    if constexpr (accumulator_custom_destroy_) {
      for (auto group : groups) {
        auto accumulator = value<typename FUNC::AccumulatorType>(group);
        if (isInitialized(group) && !isNull(group)) {
          accumulator->destroy(allocator_);
        }
      }
    } else {
      destroyAccumulators<typename FUNC::AccumulatorType>(groups);
    }
  }

 private:
  // Check if InputType has a variadic argument (must be at the last position).
  static constexpr bool hasVariadicInput() {
    if constexpr (FUNC::InputType::size_ == 0) {
      return false;
    } else {
      return isVariadicType<typename FUNC::InputType::template type_at<
          FUNC::InputType::size_ - 1>>::value;
    }
  }

  static constexpr bool has_variadic_input_ = hasVariadicInput();

  // The position in InputType where the variadic argument starts.
  static constexpr int32_t variadicStartPosition_ =
      has_variadic_input_ ? FUNC::InputType::size_ - 1 : FUNC::InputType::size_;

  // Check if an input at POSITION is non-null for the given row. For
  // non-variadic inputs this checks isSet(). For variadic inputs this checks
  // hasTopLevelNull(), which inspects each variadic element's top-level
  // nullity. This matches the expression layer's per-input null deselection
  // for simple scalar functions.
  template <size_t POSITION, typename TReader>
  static bool isInputNonNull(const TReader& reader, vector_size_t row) {
    if constexpr (POSITION >= variadicStartPosition_) {
      return !reader.hasTopLevelNull(row);
    } else {
      return reader.isSet(row);
    }
  }

  // Generic recursive unpacking function that creates VectorReaders for each
  // input type and invokes the callback with all accumulated readers.
  template <std::size_t POSITION, typename Callback, typename... TReader>
  void unpackInputs(
      const std::vector<DecodedVector>& inputDecoded,
      const Callback& callback,
      TReader&... readers) const {
    if constexpr (POSITION == FUNC::InputType::size_) {
      callback(readers...);
    } else if constexpr (isVariadicType<
                             typename FUNC::InputType::template type_at<
                                 POSITION>>::value) {
      static_assert(
          POSITION == FUNC::InputType::size_ - 1,
          "Variadic must be the last argument");
      using VariadicT = typename FUNC::InputType::template type_at<POSITION>;
      VectorReader<VariadicT> variadicReader(
          inputDecoded, variadicStartPosition_);
      callback(readers..., variadicReader);
    } else {
      VectorReader<typename FUNC::InputType::template type_at<POSITION>> reader(
          &inputDecoded[POSITION]);
      unpackInputs<POSITION + 1>(inputDecoded, callback, readers..., reader);
    }
  }

  void addRawInputImpl(char** groups, const SelectivityVector& rows) {
    unpackInputs<0>(inputDecoded_, [&](auto&... readers) {
      addRawInputWithReaders(groups, rows, readers...);
    });
  }

  template <typename... TReader>
  void addRawInputWithReaders(
      char** groups,
      const SelectivityVector& rows,
      TReader&... readers) {
    addRawInputWithReadersImpl(
        groups,
        rows,
        std::make_index_sequence<sizeof...(TReader)>{},
        readers...);
  }

  // Process rows with the accumulated readers. Use Is to access the original
  // input types to create OptionalAccessor.
  template <std::size_t... Is, typename... TReader>
  void addRawInputWithReadersImpl(
      char** groups,
      const SelectivityVector& rows,
      std::index_sequence<Is...>,
      TReader&... readers) {
    static_assert(
        sizeof...(Is) == FUNC::InputType::size_,
        "Reader count must match InputType size");

    if constexpr (aggregate_default_null_behavior_) {
      rows.applyToSelected([&](vector_size_t row) {
        // If any input is null, we ignore the whole row. For variadic
        // arguments, this includes nulls within the variadic elements.
        if (!(isInputNonNull<Is>(readers, row) && ...)) {
          return;
        }
        std::optional<RowSizeTracker<char, uint32_t>> tracker;
        if constexpr (!accumulator_is_fixed_size_) {
          tracker.emplace(groups[row][rowSizeOffset_], *allocator_);
        }
        auto group = value<typename FUNC::AccumulatorType>(groups[row]);
        group->addInput(allocator_, readers[row]...);
        clearNull(groups[row]);
      });
    } else {
      rows.applyToSelected([&](vector_size_t row) {
        std::optional<RowSizeTracker<char, uint32_t>> tracker;
        if constexpr (!accumulator_is_fixed_size_) {
          tracker.emplace(groups[row][rowSizeOffset_], *allocator_);
        }
        auto group = value<typename FUNC::AccumulatorType>(groups[row]);
        bool nonNull = group->addInput(
            allocator_,
            OptionalAccessor<typename FUNC::InputType::template type_at<Is>>{
                &readers, (int64_t)row}...);
        if (nonNull) {
          clearNull(groups[row]);
        }
      });
    }
  }

  void addSingleGroupRawInputImpl(char* group, const SelectivityVector& rows) {
    unpackInputs<0>(inputDecoded_, [&](auto&... readers) {
      addSingleGroupRawInputWithReaders(group, rows, readers...);
    });
  }

  template <typename... TReader>
  void addSingleGroupRawInputWithReaders(
      char* group,
      const SelectivityVector& rows,
      TReader&... readers) {
    addSingleGroupRawInputWithReadersImpl(
        group,
        rows,
        std::make_index_sequence<sizeof...(TReader)>{},
        readers...);
  }

  // Process rows for a single group with the accumulated readers. Use Is to
  // access the original input types to create OptionalAccessor.
  template <std::size_t... Is, typename... TReader>
  void addSingleGroupRawInputWithReadersImpl(
      char* group,
      const SelectivityVector& rows,
      std::index_sequence<Is...>,
      TReader&... readers) {
    static_assert(
        sizeof...(Is) == FUNC::InputType::size_,
        "Reader count must match InputType size");

    auto accumulator = value<typename FUNC::AccumulatorType>(group);

    if constexpr (aggregate_default_null_behavior_) {
      rows.applyToSelected([&](vector_size_t row) {
        // If any input is null, we ignore the whole row. For variadic
        // arguments, this includes nulls within the variadic elements.
        if (!(isInputNonNull<Is>(readers, row) && ...)) {
          return;
        }
        std::optional<RowSizeTracker<char, uint32_t>> tracker;
        if constexpr (!accumulator_is_fixed_size_) {
          tracker.emplace(group[rowSizeOffset_], *allocator_);
        }
        accumulator->addInput(allocator_, readers[row]...);
        clearNull(group);
      });
    } else {
      rows.applyToSelected([&](vector_size_t row) {
        std::optional<RowSizeTracker<char, uint32_t>> tracker;
        if constexpr (!accumulator_is_fixed_size_) {
          tracker.emplace(group[rowSizeOffset_], *allocator_);
        }
        bool nonNull = accumulator->addInput(
            allocator_,
            OptionalAccessor<typename FUNC::InputType::template type_at<Is>>{
                &readers, (int64_t)row}...);
        if (nonNull) {
          clearNull(group);
        }
      });
    }
  }

  void toIntermediateImpl(
      const std::vector<DecodedVector>& inputDecoded,
      const SelectivityVector& rows,
      VectorPtr& result) const {
    VELOX_CHECK(result);
    result->ensureWritable(rows);
    auto* rawNulls = result->mutableRawNulls();
    bits::fillBits(rawNulls, 0, result->size(), bits::kNull);

    constexpr auto intermediateKind =
        SimpleTypeTrait<typename FUNC::IntermediateType>::typeKind;
    auto* flatResult =
        result->as<typename KindToFlatVector<intermediateKind>::type>();
    exec::VectorWriter<typename FUNC::IntermediateType> writer;
    writer.init(*flatResult);

    unpackInputs<0>(inputDecoded, [&](auto&... readers) {
      toIntermediateWithReaders(rows, writer, readers...);
    });
    writer.finish();
  }

  template <typename... TReader>
  void toIntermediateWithReaders(
      const SelectivityVector& rows,
      exec::VectorWriter<typename FUNC::IntermediateType>& writer,
      TReader&... readers) const {
    toIntermediateWithReadersImpl(
        rows,
        writer,
        std::make_index_sequence<sizeof...(TReader)>{},
        readers...);
  }

  // Process rows for toIntermediate with the accumulated readers.
  template <std::size_t... Is, typename... TReader>
  void toIntermediateWithReadersImpl(
      const SelectivityVector& rows,
      exec::VectorWriter<typename FUNC::IntermediateType>& writer,
      std::index_sequence<Is...>,
      TReader&... readers) const {
    static_assert(
        sizeof...(Is) == FUNC::InputType::size_,
        "Reader count must match InputType size");

    if constexpr (aggregate_default_null_behavior_) {
      rows.applyToSelected([&](vector_size_t row) {
        writer.setOffset(row);
        // If any input is null, we ignore the whole row. For variadic
        // arguments, this includes nulls within the variadic elements.
        if (!(isInputNonNull<Is>(readers, row) && ...)) {
          writer.commitNull();
          return;
        }
        bool nonNull = FUNC::toIntermediate(writer.current(), readers[row]...);
        writer.commit(nonNull);
      });
    } else {
      rows.applyToSelected([&](vector_size_t row) {
        writer.setOffset(row);
        bool nonNull = FUNC::toIntermediate(
            writer.current(),
            OptionalAccessor<typename FUNC::InputType::template type_at<Is>>{
                &readers, (int64_t)row}...);
        writer.commit(nonNull);
      });
    }
  }

  // Implementation of addIntermediateResults when the intermediate type is not
  // a Row type.
  void addIntermediateResultsImpl(
      char** groups,
      const SelectivityVector& rows) {
    VectorReader<typename FUNC::IntermediateType> reader(&intermediateDecoded_);

    if constexpr (aggregate_default_null_behavior_) {
      rows.applyToSelected([&](vector_size_t row) {
        if (!reader.isSet(row)) {
          return;
        }
        std::optional<RowSizeTracker<char, uint32_t>> tracker;
        if constexpr (!accumulator_is_fixed_size_) {
          tracker.emplace(groups[row][rowSizeOffset_], *allocator_);
        }
        auto group = value<typename FUNC::AccumulatorType>(groups[row]);
        group->combine(allocator_, reader[row]);
        clearNull(groups[row]);
      });
    } else {
      rows.applyToSelected([&](vector_size_t row) {
        std::optional<RowSizeTracker<char, uint32_t>> tracker;
        if constexpr (!accumulator_is_fixed_size_) {
          tracker.emplace(groups[row][rowSizeOffset_], *allocator_);
        }
        auto group = value<typename FUNC::AccumulatorType>(groups[row]);
        bool nonNull = group->combine(
            allocator_,
            OptionalAccessor<typename FUNC::IntermediateType>{
                &reader, (int64_t)row});
        if (nonNull) {
          clearNull(groups[row]);
        }
      });
    }
  }

  // Implementation of addSingleGroupIntermediateResults when the intermediate
  // type is not a Row type.
  void addSingleGroupIntermediateResultsImpl(
      char* group,
      const SelectivityVector& rows) {
    VectorReader<typename FUNC::IntermediateType> reader(&intermediateDecoded_);
    auto accumulator = value<typename FUNC::AccumulatorType>(group);

    if constexpr (aggregate_default_null_behavior_) {
      rows.applyToSelected([&](vector_size_t row) {
        if (!reader.isSet(row)) {
          return;
        }
        std::optional<RowSizeTracker<char, uint32_t>> tracker;
        if constexpr (!accumulator_is_fixed_size_) {
          tracker.emplace(group[rowSizeOffset_], *allocator_);
        }
        accumulator->combine(allocator_, reader[row]);
        clearNull(group);
      });
    } else {
      rows.applyToSelected([&](vector_size_t row) {
        std::optional<RowSizeTracker<char, uint32_t>> tracker;
        if constexpr (!accumulator_is_fixed_size_) {
          tracker.emplace(group[rowSizeOffset_], *allocator_);
        }
        bool nonNull = accumulator->combine(
            allocator_,
            OptionalAccessor<typename FUNC::IntermediateType>{
                &reader, (int64_t)row});
        if (nonNull) {
          clearNull(group);
        }
      });
    }
  }

  std::vector<DecodedVector> inputDecoded_;
  DecodedVector intermediateDecoded_;

  std::unique_ptr<FUNC> fn_;
};

} // namespace facebook::velox::exec
