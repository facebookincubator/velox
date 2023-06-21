// See https://raw.githubusercontent.com/duckdb/duckdb/master/LICENSE for licensing information

#include "duckdb.hpp"
#include "duckdb-internal.hpp"
#ifndef DUCKDB_AMALGAMATION
#error header mismatch
#endif







namespace duckdb {

struct ArgMinMaxStateBase {
	ArgMinMaxStateBase() : is_initialized(false) {
	}

	template <class T>
	static inline void CreateValue(T &value) {
	}

	template <class T>
	static inline void DestroyValue(T &value) {
	}

	template <class T>
	static inline void AssignValue(T &target, T new_value, bool is_initialized) {
		target = new_value;
	}

	template <typename T>
	static inline void ReadValue(Vector &result, T &arg, T *target, idx_t idx) {
		target[idx] = arg;
	}

	bool is_initialized;
};

// Out-of-line specialisations
template <>
void ArgMinMaxStateBase::CreateValue(Vector *&value) {
	value = nullptr;
}

template <>
void ArgMinMaxStateBase::DestroyValue(string_t &value) {
	if (!value.IsInlined()) {
		delete[] value.GetData();
	}
}

template <>
void ArgMinMaxStateBase::DestroyValue(Vector *&value) {
	delete value;
	value = nullptr;
}

template <>
void ArgMinMaxStateBase::AssignValue(string_t &target, string_t new_value, bool is_initialized) {
	if (is_initialized) {
		DestroyValue(target);
	}
	if (new_value.IsInlined()) {
		target = new_value;
	} else {
		// non-inlined string, need to allocate space for it
		auto len = new_value.GetSize();
		auto ptr = new char[len];
		memcpy(ptr, new_value.GetData(), len);

		target = string_t(ptr, len);
	}
}

template <>
void ArgMinMaxStateBase::ReadValue(Vector &result, string_t &arg, string_t *target, idx_t idx) {
	target[idx] = StringVector::AddStringOrBlob(result, arg);
}

template <class A, class B>
struct ArgMinMaxState : public ArgMinMaxStateBase {
	using ARG_TYPE = A;
	using BY_TYPE = B;

	ARG_TYPE arg;
	BY_TYPE value;

	ArgMinMaxState() {
		CreateValue(arg);
		CreateValue(value);
	}

	~ArgMinMaxState() {
		if (is_initialized) {
			DestroyValue(arg);
			DestroyValue(value);
			is_initialized = false;
		}
	}
};

template <class COMPARATOR>
struct ArgMinMaxBase {
	template <class STATE>
	static void Destroy(AggregateInputData &aggr_input_data, STATE *state) {
		state->~STATE();
	}

	template <class STATE>
	static void Initialize(STATE *state) {
		new (state) STATE;
	}

	template <class A_TYPE, class B_TYPE, class STATE, class OP>
	static void Operation(STATE *state, AggregateInputData &, A_TYPE *x_data, B_TYPE *y_data, ValidityMask &amask,
	                      ValidityMask &bmask, idx_t xidx, idx_t yidx) {
		if (!state->is_initialized) {
			STATE::template AssignValue<A_TYPE>(state->arg, x_data[xidx], false);
			STATE::template AssignValue<B_TYPE>(state->value, y_data[yidx], false);
			state->is_initialized = true;
		} else {
			OP::template Execute<A_TYPE, B_TYPE, STATE>(state, x_data[xidx], y_data[yidx]);
		}
	}

	template <class A_TYPE, class B_TYPE, class STATE>
	static void Execute(STATE *state, A_TYPE x_data, B_TYPE y_data) {
		if (COMPARATOR::Operation(y_data, state->value)) {
			STATE::template AssignValue<A_TYPE>(state->arg, x_data, true);
			STATE::template AssignValue<B_TYPE>(state->value, y_data, true);
		}
	}

	template <class STATE, class OP>
	static void Combine(const STATE &source, STATE *target, AggregateInputData &) {
		if (!source.is_initialized) {
			return;
		}
		if (!target->is_initialized || COMPARATOR::Operation(source.value, target->value)) {
			STATE::template AssignValue(target->arg, source.arg, target->is_initialized);
			STATE::template AssignValue(target->value, source.value, target->is_initialized);
			target->is_initialized = true;
		}
	}

	template <class T, class STATE>
	static void Finalize(Vector &result, AggregateInputData &, STATE *state, T *target, ValidityMask &mask, idx_t idx) {
		if (!state->is_initialized) {
			mask.SetInvalid(idx);
		} else {
			STATE::template ReadValue(result, state->arg, target, idx);
		}
	}

	static bool IgnoreNull() {
		return true;
	}
};

template <typename COMPARATOR>
struct VectorArgMinMaxBase : ArgMinMaxBase<COMPARATOR> {
	template <class STATE>
	static void AssignVector(STATE *state, Vector &arg, const idx_t idx) {
		if (!state->is_initialized) {
			state->arg = new Vector(arg.GetType());
			state->arg->SetVectorType(VectorType::CONSTANT_VECTOR);
		}
		sel_t selv = idx;
		SelectionVector sel(&selv);
		VectorOperations::Copy(arg, *state->arg, sel, 1, 0, 0);
	}

	template <class STATE>
	static void Update(Vector inputs[], AggregateInputData &, idx_t input_count, Vector &state_vector, idx_t count) {
		auto &arg = inputs[0];
		UnifiedVectorFormat adata;
		arg.ToUnifiedFormat(count, adata);

		using BY_TYPE = typename STATE::BY_TYPE;
		auto &by = inputs[1];
		UnifiedVectorFormat bdata;
		by.ToUnifiedFormat(count, bdata);
		const auto bys = (BY_TYPE *)bdata.data;

		UnifiedVectorFormat sdata;
		state_vector.ToUnifiedFormat(count, sdata);

		auto states = (STATE **)sdata.data;
		for (idx_t i = 0; i < count; i++) {
			const auto bidx = bdata.sel->get_index(i);
			if (!bdata.validity.RowIsValid(bidx)) {
				continue;
			}
			const auto bval = bys[bidx];

			const auto sidx = sdata.sel->get_index(i);
			auto state = states[sidx];
			if (!state->is_initialized) {
				STATE::template AssignValue<BY_TYPE>(state->value, bval, false);
				AssignVector(state, arg, i);
				state->is_initialized = true;

			} else if (COMPARATOR::template Operation<BY_TYPE>(bval, state->value)) {
				STATE::template AssignValue<BY_TYPE>(state->value, bval, true);
				AssignVector(state, arg, i);
			}
		}
	}

	template <class STATE, class OP>
	static void Combine(const STATE &source, STATE *target, AggregateInputData &) {
		if (!source.is_initialized) {
			return;
		}
		if (!target->is_initialized || COMPARATOR::Operation(source.value, target->value)) {
			STATE::template AssignValue(target->value, source.value, target->is_initialized);
			AssignVector(target, *source.arg, 0);
			target->is_initialized = true;
		}
	}

	template <class T, class STATE>
	static void Finalize(Vector &result, AggregateInputData &, STATE *state, T *target, ValidityMask &mask, idx_t idx) {
		if (!state->is_initialized) {
			// we need to use SetNull here
			// since for STRUCT columns only setting the validity mask of the struct is incorrect
			// as for a struct column, we need to also set ALL child columns to NULL
			switch (result.GetVectorType()) {
			case VectorType::FLAT_VECTOR:
				FlatVector::SetNull(result, idx, true);
				break;
			case VectorType::CONSTANT_VECTOR:
				ConstantVector::SetNull(result, true);
				break;
			default:
				throw InternalException("Invalid result vector type for nested arg_min/max");
			}
		} else {
			VectorOperations::Copy(*state->arg, result, 1, 0, idx);
		}
	}

	static unique_ptr<FunctionData> Bind(ClientContext &context, AggregateFunction &function,
	                                     vector<unique_ptr<Expression>> &arguments) {
		function.arguments[0] = arguments[0]->return_type;
		function.return_type = arguments[0]->return_type;
		return nullptr;
	}
};

template <class OP, class ARG_TYPE, class BY_TYPE>
AggregateFunction GetVectorArgMinMaxFunctionInternal(const LogicalType &by_type, const LogicalType &type) {
	using STATE = ArgMinMaxState<ARG_TYPE, BY_TYPE>;
	return AggregateFunction({type, by_type}, type, AggregateFunction::StateSize<STATE>,
	                         AggregateFunction::StateInitialize<STATE, OP>, OP::template Update<STATE>,
	                         AggregateFunction::StateCombine<STATE, OP>,
	                         AggregateFunction::StateFinalize<STATE, void, OP>, nullptr, OP::Bind,
	                         AggregateFunction::StateDestroy<STATE, OP>);
}

template <class OP, class ARG_TYPE>
AggregateFunction GetVectorArgMinMaxFunctionBy(const LogicalType &by_type, const LogicalType &type) {
	switch (by_type.InternalType()) {
	case PhysicalType::INT32:
		return GetVectorArgMinMaxFunctionInternal<OP, ARG_TYPE, int32_t>(by_type, type);
	case PhysicalType::INT64:
		return GetVectorArgMinMaxFunctionInternal<OP, ARG_TYPE, int64_t>(by_type, type);
	case PhysicalType::DOUBLE:
		return GetVectorArgMinMaxFunctionInternal<OP, ARG_TYPE, double>(by_type, type);
	case PhysicalType::VARCHAR:
		return GetVectorArgMinMaxFunctionInternal<OP, ARG_TYPE, string_t>(by_type, type);
	default:
		throw InternalException("Unimplemented arg_min/arg_max aggregate");
	}
}

template <class OP, class ARG_TYPE>
void AddVectorArgMinMaxFunctionBy(AggregateFunctionSet &fun, const LogicalType &type) {
	fun.AddFunction(GetVectorArgMinMaxFunctionBy<OP, ARG_TYPE>(LogicalType::INTEGER, type));
	fun.AddFunction(GetVectorArgMinMaxFunctionBy<OP, ARG_TYPE>(LogicalType::BIGINT, type));
	fun.AddFunction(GetVectorArgMinMaxFunctionBy<OP, ARG_TYPE>(LogicalType::DOUBLE, type));
	fun.AddFunction(GetVectorArgMinMaxFunctionBy<OP, ARG_TYPE>(LogicalType::VARCHAR, type));
	fun.AddFunction(GetVectorArgMinMaxFunctionBy<OP, ARG_TYPE>(LogicalType::DATE, type));
	fun.AddFunction(GetVectorArgMinMaxFunctionBy<OP, ARG_TYPE>(LogicalType::TIMESTAMP, type));
	fun.AddFunction(GetVectorArgMinMaxFunctionBy<OP, ARG_TYPE>(LogicalType::TIMESTAMP_TZ, type));
	fun.AddFunction(GetVectorArgMinMaxFunctionBy<OP, ARG_TYPE>(LogicalType::BLOB, type));
}

template <class OP, class ARG_TYPE, class BY_TYPE>
AggregateFunction GetArgMinMaxFunctionInternal(const LogicalType &by_type, const LogicalType &type) {
	using STATE = ArgMinMaxState<ARG_TYPE, BY_TYPE>;
	auto function = AggregateFunction::BinaryAggregate<STATE, ARG_TYPE, BY_TYPE, ARG_TYPE, OP>(type, by_type, type);
	if (type.InternalType() == PhysicalType::VARCHAR || by_type.InternalType() == PhysicalType::VARCHAR) {
		function.destructor = AggregateFunction::StateDestroy<STATE, OP>;
	}
	return function;
}

template <class OP, class ARG_TYPE>
AggregateFunction GetArgMinMaxFunctionBy(const LogicalType &by_type, const LogicalType &type) {
	switch (by_type.InternalType()) {
	case PhysicalType::INT32:
		return GetArgMinMaxFunctionInternal<OP, ARG_TYPE, int32_t>(by_type, type);
	case PhysicalType::INT64:
		return GetArgMinMaxFunctionInternal<OP, ARG_TYPE, int64_t>(by_type, type);
	case PhysicalType::DOUBLE:
		return GetArgMinMaxFunctionInternal<OP, ARG_TYPE, double>(by_type, type);
	case PhysicalType::VARCHAR:
		return GetArgMinMaxFunctionInternal<OP, ARG_TYPE, string_t>(by_type, type);
	default:
		throw InternalException("Unimplemented arg_min/arg_max aggregate");
	}
}

template <class OP, class ARG_TYPE>
void AddArgMinMaxFunctionBy(AggregateFunctionSet &fun, const LogicalType &type) {
	fun.AddFunction(GetArgMinMaxFunctionBy<OP, ARG_TYPE>(LogicalType::INTEGER, type));
	fun.AddFunction(GetArgMinMaxFunctionBy<OP, ARG_TYPE>(LogicalType::BIGINT, type));
	fun.AddFunction(GetArgMinMaxFunctionBy<OP, ARG_TYPE>(LogicalType::DOUBLE, type));
	fun.AddFunction(GetArgMinMaxFunctionBy<OP, ARG_TYPE>(LogicalType::VARCHAR, type));
	fun.AddFunction(GetArgMinMaxFunctionBy<OP, ARG_TYPE>(LogicalType::DATE, type));
	fun.AddFunction(GetArgMinMaxFunctionBy<OP, ARG_TYPE>(LogicalType::TIMESTAMP, type));
	fun.AddFunction(GetArgMinMaxFunctionBy<OP, ARG_TYPE>(LogicalType::TIMESTAMP_TZ, type));
	fun.AddFunction(GetArgMinMaxFunctionBy<OP, ARG_TYPE>(LogicalType::BLOB, type));
}

template <class COMPARATOR>
static void AddArgMinMaxFunctions(AggregateFunctionSet &fun) {
	using OP = ArgMinMaxBase<COMPARATOR>;
	AddArgMinMaxFunctionBy<OP, int32_t>(fun, LogicalType::INTEGER);
	AddArgMinMaxFunctionBy<OP, int64_t>(fun, LogicalType::BIGINT);
	AddArgMinMaxFunctionBy<OP, double>(fun, LogicalType::DOUBLE);
	AddArgMinMaxFunctionBy<OP, string_t>(fun, LogicalType::VARCHAR);
	AddArgMinMaxFunctionBy<OP, date_t>(fun, LogicalType::DATE);
	AddArgMinMaxFunctionBy<OP, timestamp_t>(fun, LogicalType::TIMESTAMP);
	AddArgMinMaxFunctionBy<OP, timestamp_t>(fun, LogicalType::TIMESTAMP_TZ);
	AddArgMinMaxFunctionBy<OP, string_t>(fun, LogicalType::BLOB);

	using VECTOR_OP = VectorArgMinMaxBase<COMPARATOR>;
	AddVectorArgMinMaxFunctionBy<VECTOR_OP, Vector *>(fun, LogicalType::ANY);
}

AggregateFunctionSet ArgMinFun::GetFunctions() {
	AggregateFunctionSet fun;
	AddArgMinMaxFunctions<LessThan>(fun);
	return fun;
}

AggregateFunctionSet ArgMaxFun::GetFunctions() {
	AggregateFunctionSet fun;
	AddArgMinMaxFunctions<GreaterThan>(fun);
	return fun;
}

} // namespace duckdb








namespace duckdb {

template <class T>
struct BitState {
	bool is_set;
	T value;
};

template <class OP>
static AggregateFunction GetBitfieldUnaryAggregate(LogicalType type) {
	switch (type.id()) {
	case LogicalTypeId::TINYINT:
		return AggregateFunction::UnaryAggregate<BitState<uint8_t>, int8_t, int8_t, OP>(type, type);
	case LogicalTypeId::SMALLINT:
		return AggregateFunction::UnaryAggregate<BitState<uint16_t>, int16_t, int16_t, OP>(type, type);
	case LogicalTypeId::INTEGER:
		return AggregateFunction::UnaryAggregate<BitState<uint32_t>, int32_t, int32_t, OP>(type, type);
	case LogicalTypeId::BIGINT:
		return AggregateFunction::UnaryAggregate<BitState<uint64_t>, int64_t, int64_t, OP>(type, type);
	case LogicalTypeId::HUGEINT:
		return AggregateFunction::UnaryAggregate<BitState<hugeint_t>, hugeint_t, hugeint_t, OP>(type, type);
	case LogicalTypeId::UTINYINT:
		return AggregateFunction::UnaryAggregate<BitState<uint8_t>, uint8_t, uint8_t, OP>(type, type);
	case LogicalTypeId::USMALLINT:
		return AggregateFunction::UnaryAggregate<BitState<uint16_t>, uint16_t, uint16_t, OP>(type, type);
	case LogicalTypeId::UINTEGER:
		return AggregateFunction::UnaryAggregate<BitState<uint32_t>, uint32_t, uint32_t, OP>(type, type);
	case LogicalTypeId::UBIGINT:
		return AggregateFunction::UnaryAggregate<BitState<uint64_t>, uint64_t, uint64_t, OP>(type, type);
	default:
		throw InternalException("Unimplemented bitfield type for unary aggregate");
	}
}

struct BitwiseOperation {
	template <class STATE>
	static void Initialize(STATE *state) {
		//  If there are no matching rows, returns a null value.
		state->is_set = false;
	}

	template <class INPUT_TYPE, class STATE, class OP>
	static void Operation(STATE *state, AggregateInputData &, INPUT_TYPE *input, ValidityMask &mask, idx_t idx) {
		if (!state->is_set) {
			OP::template Assign(state, input[idx]);
			state->is_set = true;
		} else {
			OP::template Execute(state, input[idx]);
		}
	}

	template <class INPUT_TYPE, class STATE, class OP>
	static void ConstantOperation(STATE *state, AggregateInputData &aggr_input_data, INPUT_TYPE *input,
	                              ValidityMask &mask, idx_t count) {
		OP::template Operation<INPUT_TYPE, STATE, OP>(state, aggr_input_data, input, mask, 0);
	}

	template <class INPUT_TYPE, class STATE>
	static void Assign(STATE *state, INPUT_TYPE input) {
		state->value = input;
	}

	template <class STATE, class OP>
	static void Combine(const STATE &source, STATE *target, AggregateInputData &) {
		if (!source.is_set) {
			// source is NULL, nothing to do.
			return;
		}
		if (!target->is_set) {
			// target is NULL, use source value directly.
			OP::template Assign(target, source.value);
			target->is_set = true;
		} else {
			OP::template Execute(target, source.value);
		}
	}

	template <class T, class STATE>
	static void Finalize(Vector &result, AggregateInputData &, STATE *state, T *target, ValidityMask &mask, idx_t idx) {
		if (!state->is_set) {
			mask.SetInvalid(idx);
		} else {
			target[idx] = state->value;
		}
	}

	static bool IgnoreNull() {
		return true;
	}
};

struct BitAndOperation : public BitwiseOperation {
	template <class INPUT_TYPE, class STATE>
	static void Execute(STATE *state, INPUT_TYPE input) {
		state->value &= input;
	}
};

struct BitOrOperation : public BitwiseOperation {
	template <class INPUT_TYPE, class STATE>
	static void Execute(STATE *state, INPUT_TYPE input) {
		state->value |= input;
	}
};

struct BitXorOperation : public BitwiseOperation {
	template <class INPUT_TYPE, class STATE>
	static void Execute(STATE *state, INPUT_TYPE input) {
		state->value ^= input;
	}

	template <class INPUT_TYPE, class STATE, class OP>
	static void ConstantOperation(STATE *state, AggregateInputData &aggr_input_data, INPUT_TYPE *input,
	                              ValidityMask &mask, idx_t count) {
		for (idx_t i = 0; i < count; i++) {
			Operation<INPUT_TYPE, STATE, OP>(state, aggr_input_data, input, mask, 0);
		}
	}
};

struct BitStringBitwiseOperation : public BitwiseOperation {
	template <class STATE>
	static void Destroy(AggregateInputData &aggr_input_data, STATE *state) {
		if (state->is_set && !state->value.IsInlined()) {
			delete[] state->value.GetData();
		}
	}

	template <class INPUT_TYPE, class STATE>
	static void Assign(STATE *state, INPUT_TYPE input) {
		D_ASSERT(state->is_set == false);
		if (input.IsInlined()) {
			state->value = input;
		} else { // non-inlined string, need to allocate space for it
			auto len = input.GetSize();
			auto ptr = new char[len];
			memcpy(ptr, input.GetData(), len);

			state->value = string_t(ptr, len);
		}
	}

	template <class T, class STATE>
	static void Finalize(Vector &result, AggregateInputData &, STATE *state, T *target, ValidityMask &mask, idx_t idx) {
		if (!state->is_set) {
			mask.SetInvalid(idx);
		} else {
			target[idx] = StringVector::AddStringOrBlob(result, state->value);
		}
	}
};

struct BitStringAndOperation : public BitStringBitwiseOperation {

	template <class INPUT_TYPE, class STATE>
	static void Execute(STATE *state, INPUT_TYPE input) {
		Bit::BitwiseAnd(input, state->value, state->value);
	}
};

struct BitStringOrOperation : public BitStringBitwiseOperation {

	template <class INPUT_TYPE, class STATE>
	static void Execute(STATE *state, INPUT_TYPE input) {
		Bit::BitwiseOr(input, state->value, state->value);
	}
};

struct BitStringXorOperation : public BitStringBitwiseOperation {
	template <class INPUT_TYPE, class STATE>
	static void Execute(STATE *state, INPUT_TYPE input) {
		Bit::BitwiseXor(input, state->value, state->value);
	}

	template <class INPUT_TYPE, class STATE, class OP>
	static void ConstantOperation(STATE *state, AggregateInputData &aggr_input_data, INPUT_TYPE *input,
	                              ValidityMask &mask, idx_t count) {
		for (idx_t i = 0; i < count; i++) {
			Operation<INPUT_TYPE, STATE, OP>(state, aggr_input_data, input, mask, 0);
		}
	}
};

AggregateFunctionSet BitAndFun::GetFunctions() {
	AggregateFunctionSet bit_and;
	for (auto &type : LogicalType::Integral()) {
		bit_and.AddFunction(GetBitfieldUnaryAggregate<BitAndOperation>(type));
	}

	bit_and.AddFunction(
	    AggregateFunction::UnaryAggregateDestructor<BitState<string_t>, string_t, string_t, BitStringAndOperation>(
	        LogicalType::BIT, LogicalType::BIT));
	return bit_and;
}

AggregateFunctionSet BitOrFun::GetFunctions() {
	AggregateFunctionSet bit_or;
	for (auto &type : LogicalType::Integral()) {
		bit_or.AddFunction(GetBitfieldUnaryAggregate<BitOrOperation>(type));
	}
	bit_or.AddFunction(
	    AggregateFunction::UnaryAggregateDestructor<BitState<string_t>, string_t, string_t, BitStringOrOperation>(
	        LogicalType::BIT, LogicalType::BIT));
	return bit_or;
}

AggregateFunctionSet BitXorFun::GetFunctions() {
	AggregateFunctionSet bit_xor;
	for (auto &type : LogicalType::Integral()) {
		bit_xor.AddFunction(GetBitfieldUnaryAggregate<BitXorOperation>(type));
	}
	bit_xor.AddFunction(
	    AggregateFunction::UnaryAggregateDestructor<BitState<string_t>, string_t, string_t, BitStringXorOperation>(
	        LogicalType::BIT, LogicalType::BIT));
	return bit_xor;
}

} // namespace duckdb










namespace duckdb {

template <class INPUT_TYPE>
struct BitAggState {
	bool is_set;
	string_t value;
	INPUT_TYPE min;
	INPUT_TYPE max;
};

struct BitstringAggBindData : public FunctionData {
	Value min;
	Value max;

	BitstringAggBindData() {
	}

	BitstringAggBindData(Value min, Value max) : min(std::move(min)), max(std::move(max)) {
	}

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<BitstringAggBindData>(*this);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<BitstringAggBindData>();
		if (min.IsNull() && other.min.IsNull() && max.IsNull() && other.max.IsNull()) {
			return true;
		}
		if (Value::NotDistinctFrom(min, other.min) && Value::NotDistinctFrom(max, other.max)) {
			return true;
		}
		return false;
	}
};

struct BitStringAggOperation {
	static constexpr const idx_t MAX_BIT_RANGE = 1000000000; // for now capped at 1 billion bits

	template <class STATE>
	static void Initialize(STATE *state) {
		state->is_set = false;
	}

	template <class INPUT_TYPE, class STATE, class OP>
	static void Operation(STATE *state, AggregateInputData &data, INPUT_TYPE *input, ValidityMask &mask, idx_t idx) {
		auto bind_agg_data = (BitstringAggBindData *)data.bind_data;
		if (!state->is_set) {
			if (bind_agg_data->min.IsNull() || bind_agg_data->max.IsNull()) {
				throw BinderException(
				    "Could not retrieve required statistics. Alternatively, try by providing the statistics "
				    "explicitly: BITSTRING_AGG(col, min, max) ");
			}
			state->min = bind_agg_data->min.GetValue<INPUT_TYPE>();
			state->max = bind_agg_data->max.GetValue<INPUT_TYPE>();
			idx_t bit_range =
			    GetRange(bind_agg_data->min.GetValue<INPUT_TYPE>(), bind_agg_data->max.GetValue<INPUT_TYPE>());
			if (bit_range > MAX_BIT_RANGE) {
				throw OutOfRangeException(
				    "The range between min and max value (%s <-> %s) is too large for bitstring aggregation",
				    NumericHelper::ToString(state->min), NumericHelper::ToString(state->max));
			}
			idx_t len = Bit::ComputeBitstringLen(bit_range);
			auto target = len > string_t::INLINE_LENGTH ? string_t(new char[len], len) : string_t(len);
			Bit::SetEmptyBitString(target, bit_range);

			state->value = target;
			state->is_set = true;
		}
		if (input[idx] >= state->min && input[idx] <= state->max) {
			Execute(state, input[idx], bind_agg_data->min.GetValue<INPUT_TYPE>());
		} else {
			throw OutOfRangeException("Value %s is outside of provided min and max range (%s <-> %s)",
			                          NumericHelper::ToString(input[idx]), NumericHelper::ToString(state->min),
			                          NumericHelper::ToString(state->max));
		}
	}

	template <class INPUT_TYPE, class STATE, class OP>
	static void ConstantOperation(STATE *state, AggregateInputData &aggr_input_data, INPUT_TYPE *input,
	                              ValidityMask &mask, idx_t count) {
		OP::template Operation<INPUT_TYPE, STATE, OP>(state, aggr_input_data, input, mask, 0);
	}

	template <class INPUT_TYPE>
	static idx_t GetRange(INPUT_TYPE min, INPUT_TYPE max) {
		D_ASSERT(max >= min);
		INPUT_TYPE result;
		if (!TrySubtractOperator::Operation(max, min, result)) {
			return NumericLimits<idx_t>::Maximum();
		}
		idx_t val(result);
		if (val == NumericLimits<idx_t>::Maximum()) {
			return val;
		}
		return val + 1;
	}

	template <class INPUT_TYPE, class STATE>
	static void Execute(STATE *state, INPUT_TYPE input, INPUT_TYPE min) {
		Bit::SetBit(state->value, input - min, 1);
	}

	template <class STATE, class OP>
	static void Combine(const STATE &source, STATE *target, AggregateInputData &) {
		if (!source.is_set) {
			return;
		}
		if (!target->is_set) {
			Assign(target, source.value);
			target->is_set = true;
			target->min = source.min;
			target->max = source.max;
		} else {
			Bit::BitwiseOr(source.value, target->value, target->value);
		}
	}

	template <class INPUT_TYPE, class STATE>
	static void Assign(STATE *state, INPUT_TYPE input) {
		D_ASSERT(state->is_set == false);
		if (input.IsInlined()) {
			state->value = input;
		} else { // non-inlined string, need to allocate space for it
			auto len = input.GetSize();
			auto ptr = new char[len];
			memcpy(ptr, input.GetData(), len);
			state->value = string_t(ptr, len);
		}
	}

	template <class T, class STATE>
	static void Finalize(Vector &result, AggregateInputData &, STATE *state, T *target, ValidityMask &mask, idx_t idx) {
		if (!state->is_set) {
			mask.SetInvalid(idx);
		} else {
			target[idx] = StringVector::AddStringOrBlob(result, state->value);
		}
	}

	template <class STATE>
	static void Destroy(AggregateInputData &aggr_input_data, STATE *state) {
		if (state->is_set && !state->value.IsInlined()) {
			delete[] state->value.GetData();
		}
	}

	static bool IgnoreNull() {
		return true;
	}
};

template <>
void BitStringAggOperation::Execute(BitAggState<hugeint_t> *state, hugeint_t input, hugeint_t min) {
	idx_t val;
	if (Hugeint::TryCast(input - min, val)) {
		Bit::SetBit(state->value, val, 1);
	} else {
		throw OutOfRangeException("Range too large for bitstring aggregation");
	}
}

template <>
idx_t BitStringAggOperation::GetRange(hugeint_t min, hugeint_t max) {
	hugeint_t result;
	if (!TrySubtractOperator::Operation(max, min, result)) {
		return NumericLimits<idx_t>::Maximum();
	}
	idx_t range;
	if (!Hugeint::TryCast(result + 1, range)) {
		return NumericLimits<idx_t>::Maximum();
	}
	return range;
}

unique_ptr<BaseStatistics> BitstringPropagateStats(ClientContext &context, BoundAggregateExpression &expr,
                                                   AggregateStatisticsInput &input) {

	if (!NumericStats::HasMinMax(input.child_stats[0])) {
		throw BinderException("Could not retrieve required statistics. Alternatively, try by providing the statistics "
		                      "explicitly: BITSTRING_AGG(col, min, max) ");
	}
	auto bind_agg_data = (BitstringAggBindData *)input.bind_data;
	bind_agg_data->min = NumericStats::Min(input.child_stats[0]);
	bind_agg_data->max = NumericStats::Max(input.child_stats[0]);
	return nullptr;
}

unique_ptr<FunctionData> BindBitstringAgg(ClientContext &context, AggregateFunction &function,
                                          vector<unique_ptr<Expression>> &arguments) {
	if (arguments.size() == 3) {
		if (!arguments[1]->IsFoldable() && !arguments[2]->IsFoldable()) {
			throw BinderException("bitstring_agg requires a constant min and max argument");
		}
		auto min = ExpressionExecutor::EvaluateScalar(context, *arguments[1]);
		auto max = ExpressionExecutor::EvaluateScalar(context, *arguments[2]);
		Function::EraseArgument(function, arguments, 2);
		Function::EraseArgument(function, arguments, 1);
		return make_uniq<BitstringAggBindData>(min, max);
	}
	return make_uniq<BitstringAggBindData>();
}

template <class TYPE>
static void BindBitString(AggregateFunctionSet &bitstring_agg, const LogicalTypeId &type) {
	auto function =
	    AggregateFunction::UnaryAggregateDestructor<BitAggState<TYPE>, TYPE, string_t, BitStringAggOperation>(
	        type, LogicalType::BIT);
	function.bind = BindBitstringAgg;              // create new a 'BitstringAggBindData'
	function.statistics = BitstringPropagateStats; // stores min and max from column stats in BitstringAggBindData
	bitstring_agg.AddFunction(function); // uses the BitstringAggBindData to access statistics for creating bitstring
	function.arguments = {type, type, type};
	function.statistics = nullptr; // min and max are provided as arguments
	bitstring_agg.AddFunction(function);
}

void GetBitStringAggregate(const LogicalType &type, AggregateFunctionSet &bitstring_agg) {
	switch (type.id()) {
	case LogicalType::TINYINT: {
		return BindBitString<int8_t>(bitstring_agg, type.id());
	}
	case LogicalType::SMALLINT: {
		return BindBitString<int16_t>(bitstring_agg, type.id());
	}
	case LogicalType::INTEGER: {
		return BindBitString<int32_t>(bitstring_agg, type.id());
	}
	case LogicalType::BIGINT: {
		return BindBitString<int64_t>(bitstring_agg, type.id());
	}
	case LogicalType::HUGEINT: {
		return BindBitString<hugeint_t>(bitstring_agg, type.id());
	}
	case LogicalType::UTINYINT: {
		return BindBitString<uint8_t>(bitstring_agg, type.id());
	}
	case LogicalType::USMALLINT: {
		return BindBitString<uint16_t>(bitstring_agg, type.id());
	}
	case LogicalType::UINTEGER: {
		return BindBitString<uint32_t>(bitstring_agg, type.id());
	}
	case LogicalType::UBIGINT: {
		return BindBitString<uint64_t>(bitstring_agg, type.id());
	}
	default:
		throw InternalException("Unimplemented bitstring aggregate");
	}
}

AggregateFunctionSet BitstringAggFun::GetFunctions() {
	AggregateFunctionSet bitstring_agg("bitstring_agg");
	for (auto &type : LogicalType::Integral()) {
		GetBitStringAggregate(type, bitstring_agg);
	}
	return bitstring_agg;
}

} // namespace duckdb






namespace duckdb {

struct BoolState {
	bool empty;
	bool val;
};

struct BoolAndFunFunction {
	template <class STATE>
	static void Initialize(STATE *state) {
		state->val = true;
		state->empty = true;
	}

	template <class STATE, class OP>
	static void Combine(const STATE &source, STATE *target, AggregateInputData &) {
		target->val = target->val && source.val;
		target->empty = target->empty && source.empty;
	}

	template <class T, class STATE>
	static void Finalize(Vector &result, AggregateInputData &, STATE *state, T *target, ValidityMask &mask, idx_t idx) {
		if (state->empty) {
			mask.SetInvalid(idx);
			return;
		}
		target[idx] = state->val;
	}

	template <class INPUT_TYPE, class STATE, class OP>
	static void Operation(STATE *state, AggregateInputData &, INPUT_TYPE *input, ValidityMask &mask, idx_t idx) {
		state->empty = false;
		state->val = input[idx] && state->val;
	}

	template <class INPUT_TYPE, class STATE, class OP>
	static void ConstantOperation(STATE *state, AggregateInputData &aggr_input_data, INPUT_TYPE *input,
	                              ValidityMask &mask, idx_t count) {
		for (idx_t i = 0; i < count; i++) {
			Operation<INPUT_TYPE, STATE, OP>(state, aggr_input_data, input, mask, 0);
		}
	}
	static bool IgnoreNull() {
		return true;
	}
};

struct BoolOrFunFunction {
	template <class STATE>
	static void Initialize(STATE *state) {
		state->val = false;
		state->empty = true;
	}

	template <class STATE, class OP>
	static void Combine(const STATE &source, STATE *target, AggregateInputData &) {
		target->val = target->val || source.val;
		target->empty = target->empty && source.empty;
	}

	template <class T, class STATE>
	static void Finalize(Vector &result, AggregateInputData &, STATE *state, T *target, ValidityMask &mask, idx_t idx) {
		if (state->empty) {
			mask.SetInvalid(idx);
			return;
		}
		target[idx] = state->val;
	}
	template <class INPUT_TYPE, class STATE, class OP>
	static void Operation(STATE *state, AggregateInputData &, INPUT_TYPE *input, ValidityMask &mask, idx_t idx) {
		state->empty = false;
		state->val = input[idx] || state->val;
	}

	template <class INPUT_TYPE, class STATE, class OP>
	static void ConstantOperation(STATE *state, AggregateInputData &aggr_input_data, INPUT_TYPE *input,
	                              ValidityMask &mask, idx_t count) {
		for (idx_t i = 0; i < count; i++) {
			Operation<INPUT_TYPE, STATE, OP>(state, aggr_input_data, input, mask, 0);
		}
	}

	static bool IgnoreNull() {
		return true;
	}
};

AggregateFunction BoolOrFun::GetFunction() {
	auto fun = AggregateFunction::UnaryAggregate<BoolState, bool, bool, BoolOrFunFunction>(
	    LogicalType(LogicalTypeId::BOOLEAN), LogicalType::BOOLEAN);
	fun.order_dependent = AggregateOrderDependent::NOT_ORDER_DEPENDENT;
	return fun;
}

AggregateFunction BoolAndFun::GetFunction() {
	auto fun = AggregateFunction::UnaryAggregate<BoolState, bool, bool, BoolAndFunFunction>(
	    LogicalType(LogicalTypeId::BOOLEAN), LogicalType::BOOLEAN);
	fun.order_dependent = AggregateOrderDependent::NOT_ORDER_DEPENDENT;
	return fun;
}

} // namespace duckdb








namespace duckdb {

template <class T>
struct EntropyState {
	using DistinctMap = unordered_map<T, idx_t>;

	idx_t count;
	DistinctMap *distinct;

	EntropyState &operator=(const EntropyState &other) = delete;

	EntropyState &Assign(const EntropyState &other) {
		D_ASSERT(!distinct);
		distinct = new DistinctMap(*other.distinct);
		count = other.count;
		return *this;
	}
};

struct EntropyFunctionBase {
	template <class STATE>
	static void Initialize(STATE *state) {
		state->distinct = nullptr;
		state->count = 0;
	}

	template <class STATE, class OP>
	static void Combine(const STATE &source, STATE *target, AggregateInputData &) {
		if (!source.distinct) {
			return;
		}
		if (!target->distinct) {
			target->Assign(source);
			return;
		}
		for (auto &val : *source.distinct) {
			auto value = val.first;
			(*target->distinct)[value] += val.second;
		}
		target->count += source.count;
	}

	template <class T, class STATE>
	static void Finalize(Vector &result, AggregateInputData &, STATE *state, T *target, ValidityMask &mask, idx_t idx) {
		double count = state->count;
		if (state->distinct) {
			double entropy = 0;
			for (auto &val : *state->distinct) {
				entropy += (val.second / count) * log2(count / val.second);
			}
			target[idx] = entropy;
		} else {
			target[idx] = 0;
		}
	}

	static bool IgnoreNull() {
		return true;
	}
	template <class STATE>
	static void Destroy(AggregateInputData &aggr_input_data, STATE *state) {
		if (state->distinct) {
			delete state->distinct;
		}
	}
};

struct EntropyFunction : EntropyFunctionBase {
	template <class INPUT_TYPE, class STATE, class OP>
	static void Operation(STATE *state, AggregateInputData &, INPUT_TYPE *input, ValidityMask &mask, idx_t idx) {
		if (!state->distinct) {
			state->distinct = new unordered_map<INPUT_TYPE, idx_t>();
		}
		(*state->distinct)[input[idx]]++;
		state->count++;
	}
	template <class INPUT_TYPE, class STATE, class OP>
	static void ConstantOperation(STATE *state, AggregateInputData &aggr_input_data, INPUT_TYPE *input,
	                              ValidityMask &mask, idx_t count) {
		for (idx_t i = 0; i < count; i++) {
			Operation<INPUT_TYPE, STATE, OP>(state, aggr_input_data, input, mask, 0);
		}
	}
};

struct EntropyFunctionString : EntropyFunctionBase {
	template <class INPUT_TYPE, class STATE, class OP>
	static void Operation(STATE *state, AggregateInputData &, INPUT_TYPE *input, ValidityMask &mask, idx_t idx) {
		if (!state->distinct) {
			state->distinct = new unordered_map<string, idx_t>();
		}
		auto value = input[idx].GetString();
		(*state->distinct)[value]++;
		state->count++;
	}

	template <class INPUT_TYPE, class STATE, class OP>
	static void ConstantOperation(STATE *state, AggregateInputData &aggr_input_data, INPUT_TYPE *input,
	                              ValidityMask &mask, idx_t count) {
		for (idx_t i = 0; i < count; i++) {
			Operation<INPUT_TYPE, STATE, OP>(state, aggr_input_data, input, mask, 0);
		}
	}
};

template <typename INPUT_TYPE, typename RESULT_TYPE>
AggregateFunction GetEntropyFunction(const LogicalType &input_type, const LogicalType &result_type) {
	auto fun =
	    AggregateFunction::UnaryAggregateDestructor<EntropyState<INPUT_TYPE>, INPUT_TYPE, RESULT_TYPE, EntropyFunction>(
	        input_type, result_type);
	fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	return fun;
}

AggregateFunction GetEntropyFunctionInternal(PhysicalType type) {
	switch (type) {
	case PhysicalType::UINT16:
		return AggregateFunction::UnaryAggregateDestructor<EntropyState<uint16_t>, uint16_t, double, EntropyFunction>(
		    LogicalType::USMALLINT, LogicalType::DOUBLE);
	case PhysicalType::UINT32:
		return AggregateFunction::UnaryAggregateDestructor<EntropyState<uint32_t>, uint32_t, double, EntropyFunction>(
		    LogicalType::UINTEGER, LogicalType::DOUBLE);
	case PhysicalType::UINT64:
		return AggregateFunction::UnaryAggregateDestructor<EntropyState<uint64_t>, uint64_t, double, EntropyFunction>(
		    LogicalType::UBIGINT, LogicalType::DOUBLE);
	case PhysicalType::INT16:
		return AggregateFunction::UnaryAggregateDestructor<EntropyState<int16_t>, int16_t, double, EntropyFunction>(
		    LogicalType::SMALLINT, LogicalType::DOUBLE);
	case PhysicalType::INT32:
		return AggregateFunction::UnaryAggregateDestructor<EntropyState<int32_t>, int32_t, double, EntropyFunction>(
		    LogicalType::INTEGER, LogicalType::DOUBLE);
	case PhysicalType::INT64:
		return AggregateFunction::UnaryAggregateDestructor<EntropyState<int64_t>, int64_t, double, EntropyFunction>(
		    LogicalType::BIGINT, LogicalType::DOUBLE);
	case PhysicalType::FLOAT:
		return AggregateFunction::UnaryAggregateDestructor<EntropyState<float>, float, double, EntropyFunction>(
		    LogicalType::FLOAT, LogicalType::DOUBLE);
	case PhysicalType::DOUBLE:
		return AggregateFunction::UnaryAggregateDestructor<EntropyState<double>, double, double, EntropyFunction>(
		    LogicalType::DOUBLE, LogicalType::DOUBLE);
	case PhysicalType::VARCHAR:
		return AggregateFunction::UnaryAggregateDestructor<EntropyState<string>, string_t, double,
		                                                   EntropyFunctionString>(LogicalType::VARCHAR,
		                                                                          LogicalType::DOUBLE);

	default:
		throw InternalException("Unimplemented approximate_count aggregate");
	}
}

AggregateFunction GetEntropyFunction(PhysicalType type) {
	auto fun = GetEntropyFunctionInternal(type);
	fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	return fun;
}

AggregateFunctionSet EntropyFun::GetFunctions() {
	AggregateFunctionSet entropy("entropy");
	entropy.AddFunction(GetEntropyFunction(PhysicalType::UINT16));
	entropy.AddFunction(GetEntropyFunction(PhysicalType::UINT32));
	entropy.AddFunction(GetEntropyFunction(PhysicalType::UINT64));
	entropy.AddFunction(GetEntropyFunction(PhysicalType::FLOAT));
	entropy.AddFunction(GetEntropyFunction(PhysicalType::INT16));
	entropy.AddFunction(GetEntropyFunction(PhysicalType::INT32));
	entropy.AddFunction(GetEntropyFunction(PhysicalType::INT64));
	entropy.AddFunction(GetEntropyFunction(PhysicalType::DOUBLE));
	entropy.AddFunction(GetEntropyFunction(PhysicalType::VARCHAR));
	entropy.AddFunction(GetEntropyFunction<int64_t, double>(LogicalType::TIMESTAMP, LogicalType::DOUBLE));
	entropy.AddFunction(GetEntropyFunction<int64_t, double>(LogicalType::TIMESTAMP_TZ, LogicalType::DOUBLE));
	return entropy;
}

} // namespace duckdb






namespace duckdb {

struct KurtosisState {
	idx_t n;
	double sum;
	double sum_sqr;
	double sum_cub;
	double sum_four;
};

struct KurtosisOperation {
	template <class STATE>
	static void Initialize(STATE *state) {
		state->n = 0;
		state->sum = state->sum_sqr = state->sum_cub = state->sum_four = 0.0;
	}

	template <class INPUT_TYPE, class STATE, class OP>
	static void ConstantOperation(STATE *state, AggregateInputData &aggr_input_data, INPUT_TYPE *input,
	                              ValidityMask &mask, idx_t count) {
		for (idx_t i = 0; i < count; i++) {
			Operation<INPUT_TYPE, STATE, OP>(state, aggr_input_data, input, mask, 0);
		}
	}

	template <class INPUT_TYPE, class STATE, class OP>
	static void Operation(STATE *state, AggregateInputData &, INPUT_TYPE *data, ValidityMask &mask, idx_t idx) {
		state->n++;
		state->sum += data[idx];
		state->sum_sqr += pow(data[idx], 2);
		state->sum_cub += pow(data[idx], 3);
		state->sum_four += pow(data[idx], 4);
	}

	template <class STATE, class OP>
	static void Combine(const STATE &source, STATE *target, AggregateInputData &) {
		if (source.n == 0) {
			return;
		}
		target->n += source.n;
		target->sum += source.sum;
		target->sum_sqr += source.sum_sqr;
		target->sum_cub += source.sum_cub;
		target->sum_four += source.sum_four;
	}

	template <class TARGET_TYPE, class STATE>
	static void Finalize(Vector &result, AggregateInputData &, STATE *state, TARGET_TYPE *target, ValidityMask &mask,
	                     idx_t idx) {
		auto n = (double)state->n;
		if (n <= 3) {
			mask.SetInvalid(idx);
			return;
		}
		double temp = 1 / n;
		//! This is necessary due to linux 32 bits
		long double temp_aux = 1 / n;
		if (state->sum_sqr - state->sum * state->sum * temp == 0 ||
		    state->sum_sqr - state->sum * state->sum * temp_aux == 0) {
			mask.SetInvalid(idx);
			return;
		}
		double m4 =
		    temp * (state->sum_four - 4 * state->sum_cub * state->sum * temp +
		            6 * state->sum_sqr * state->sum * state->sum * temp * temp - 3 * pow(state->sum, 4) * pow(temp, 3));

		double m2 = temp * (state->sum_sqr - state->sum * state->sum * temp);
		if (m2 <= 0 || ((n - 2) * (n - 3)) == 0) { // m2 shouldn't be below 0 but floating points are weird
			mask.SetInvalid(idx);
			return;
		}
		target[idx] = (n - 1) * ((n + 1) * m4 / (m2 * m2) - 3 * (n - 1)) / ((n - 2) * (n - 3));
		if (!Value::DoubleIsFinite(target[idx])) {
			throw OutOfRangeException("Kurtosis is out of range!");
		}
	}

	static bool IgnoreNull() {
		return true;
	}
};

AggregateFunction KurtosisFun::GetFunction() {
	return AggregateFunction::UnaryAggregate<KurtosisState, double, double, KurtosisOperation>(LogicalType::DOUBLE,
	                                                                                           LogicalType::DOUBLE);
}

} // namespace duckdb







namespace duckdb {

template <class T>
struct MinMaxState {
	T value;
	bool isset;
};

template <class OP>
static AggregateFunction GetUnaryAggregate(LogicalType type) {
	switch (type.InternalType()) {
	case PhysicalType::BOOL:
		return AggregateFunction::UnaryAggregate<MinMaxState<int8_t>, int8_t, int8_t, OP>(type, type);
	case PhysicalType::INT8:
		return AggregateFunction::UnaryAggregate<MinMaxState<int8_t>, int8_t, int8_t, OP>(type, type);
	case PhysicalType::INT16:
		return AggregateFunction::UnaryAggregate<MinMaxState<int16_t>, int16_t, int16_t, OP>(type, type);
	case PhysicalType::INT32:
		return AggregateFunction::UnaryAggregate<MinMaxState<int32_t>, int32_t, int32_t, OP>(type, type);
	case PhysicalType::INT64:
		return AggregateFunction::UnaryAggregate<MinMaxState<int64_t>, int64_t, int64_t, OP>(type, type);
	case PhysicalType::UINT8:
		return AggregateFunction::UnaryAggregate<MinMaxState<uint8_t>, uint8_t, uint8_t, OP>(type, type);
	case PhysicalType::UINT16:
		return AggregateFunction::UnaryAggregate<MinMaxState<uint16_t>, uint16_t, uint16_t, OP>(type, type);
	case PhysicalType::UINT32:
		return AggregateFunction::UnaryAggregate<MinMaxState<uint32_t>, uint32_t, uint32_t, OP>(type, type);
	case PhysicalType::UINT64:
		return AggregateFunction::UnaryAggregate<MinMaxState<uint64_t>, uint64_t, uint64_t, OP>(type, type);
	case PhysicalType::INT128:
		return AggregateFunction::UnaryAggregate<MinMaxState<hugeint_t>, hugeint_t, hugeint_t, OP>(type, type);
	case PhysicalType::FLOAT:
		return AggregateFunction::UnaryAggregate<MinMaxState<float>, float, float, OP>(type, type);
	case PhysicalType::DOUBLE:
		return AggregateFunction::UnaryAggregate<MinMaxState<double>, double, double, OP>(type, type);
	case PhysicalType::INTERVAL:
		return AggregateFunction::UnaryAggregate<MinMaxState<interval_t>, interval_t, interval_t, OP>(type, type);
	default:
		throw InternalException("Unimplemented type for min/max aggregate");
	}
}

struct MinMaxBase {
	template <class STATE>
	static void Initialize(STATE *state) {
		state->isset = false;
	}

	template <class INPUT_TYPE, class STATE, class OP>
	static void ConstantOperation(STATE *state, AggregateInputData &input_data, INPUT_TYPE *input, ValidityMask &mask,
	                              idx_t count) {
		D_ASSERT(mask.RowIsValid(0));
		if (!state->isset) {
			OP::template Assign<INPUT_TYPE, STATE>(state, input_data, input[0]);
			state->isset = true;
		} else {
			OP::template Execute<INPUT_TYPE, STATE>(state, input_data, input[0]);
		}
	}

	template <class INPUT_TYPE, class STATE, class OP>
	static void Operation(STATE *state, AggregateInputData &input_data, INPUT_TYPE *input, ValidityMask &mask,
	                      idx_t idx) {
		if (!state->isset) {
			OP::template Assign<INPUT_TYPE, STATE>(state, input_data, input[idx]);
			state->isset = true;
		} else {
			OP::template Execute<INPUT_TYPE, STATE>(state, input_data, input[idx]);
		}
	}

	static bool IgnoreNull() {
		return true;
	}
};

struct NumericMinMaxBase : public MinMaxBase {
	template <class INPUT_TYPE, class STATE>
	static void Assign(STATE *state, AggregateInputData &, INPUT_TYPE input) {
		state->value = input;
	}

	template <class T, class STATE>
	static void Finalize(Vector &result, AggregateInputData &, STATE *state, T *target, ValidityMask &mask, idx_t idx) {
		mask.Set(idx, state->isset);
		target[idx] = state->value;
	}
};

struct MinOperation : public NumericMinMaxBase {
	template <class INPUT_TYPE, class STATE>
	static void Execute(STATE *state, AggregateInputData &, INPUT_TYPE input) {
		if (LessThan::Operation<INPUT_TYPE>(input, state->value)) {
			state->value = input;
		}
	}

	template <class STATE, class OP>
	static void Combine(const STATE &source, STATE *target, AggregateInputData &) {
		if (!source.isset) {
			// source is NULL, nothing to do
			return;
		}
		if (!target->isset) {
			// target is NULL, use source value directly
			*target = source;
		} else if (GreaterThan::Operation(target->value, source.value)) {
			target->value = source.value;
		}
	}
};

struct MaxOperation : public NumericMinMaxBase {
	template <class INPUT_TYPE, class STATE>
	static void Execute(STATE *state, AggregateInputData &, INPUT_TYPE input) {
		if (GreaterThan::Operation<INPUT_TYPE>(input, state->value)) {
			state->value = input;
		}
	}

	template <class STATE, class OP>
	static void Combine(const STATE &source, STATE *target, AggregateInputData &) {
		if (!source.isset) {
			// source is NULL, nothing to do
			return;
		}
		if (!target->isset) {
			// target is NULL, use source value directly
			*target = source;
		} else if (LessThan::Operation(target->value, source.value)) {
			target->value = source.value;
		}
	}
};

struct StringMinMaxBase : public MinMaxBase {
	template <class STATE>
	static void Destroy(AggregateInputData &aggr_input_data, STATE *state) {
		if (state->isset && !state->value.IsInlined()) {
			delete[] state->value.GetData();
		}
	}

	template <class INPUT_TYPE, class STATE>
	static void Assign(STATE *state, AggregateInputData &input_data, INPUT_TYPE input) {
		Destroy(input_data, state);
		if (input.IsInlined()) {
			state->value = input;
		} else {
			// non-inlined string, need to allocate space for it
			auto len = input.GetSize();
			auto ptr = new char[len];
			memcpy(ptr, input.GetData(), len);

			state->value = string_t(ptr, len);
		}
	}

	template <class T, class STATE>
	static void Finalize(Vector &result, AggregateInputData &, STATE *state, T *target, ValidityMask &mask, idx_t idx) {
		if (!state->isset) {
			mask.SetInvalid(idx);
		} else {
			target[idx] = StringVector::AddStringOrBlob(result, state->value);
		}
	}

	template <class STATE, class OP>
	static void Combine(const STATE &source, STATE *target, AggregateInputData &input_data) {
		if (!source.isset) {
			// source is NULL, nothing to do
			return;
		}
		if (!target->isset) {
			// target is NULL, use source value directly
			Assign(target, input_data, source.value);
			target->isset = true;
		} else {
			OP::template Execute<string_t, STATE>(target, input_data, source.value);
		}
	}
};

struct MinOperationString : public StringMinMaxBase {
	template <class INPUT_TYPE, class STATE>
	static void Execute(STATE *state, AggregateInputData &input_data, INPUT_TYPE input) {
		if (LessThan::Operation<INPUT_TYPE>(input, state->value)) {
			Assign(state, input_data, input);
		}
	}
};

struct MaxOperationString : public StringMinMaxBase {
	template <class INPUT_TYPE, class STATE>
	static void Execute(STATE *state, AggregateInputData &input_data, INPUT_TYPE input) {
		if (GreaterThan::Operation<INPUT_TYPE>(input, state->value)) {
			Assign(state, input_data, input);
		}
	}
};

template <typename T, class OP>
static bool TemplatedOptimumType(Vector &left, idx_t lidx, idx_t lcount, Vector &right, idx_t ridx, idx_t rcount) {
	UnifiedVectorFormat lvdata, rvdata;
	left.ToUnifiedFormat(lcount, lvdata);
	right.ToUnifiedFormat(rcount, rvdata);

	lidx = lvdata.sel->get_index(lidx);
	ridx = rvdata.sel->get_index(ridx);

	auto ldata = (const T *)lvdata.data;
	auto rdata = (const T *)rvdata.data;

	auto &lval = ldata[lidx];
	auto &rval = rdata[ridx];

	auto lnull = !lvdata.validity.RowIsValid(lidx);
	auto rnull = !rvdata.validity.RowIsValid(ridx);

	return OP::Operation(lval, rval, lnull, rnull);
}

template <class OP>
static bool TemplatedOptimumList(Vector &left, idx_t lidx, idx_t lcount, Vector &right, idx_t ridx, idx_t rcount);

template <class OP>
static bool TemplatedOptimumStruct(Vector &left, idx_t lidx, idx_t lcount, Vector &right, idx_t ridx, idx_t rcount);

template <class OP>
static bool TemplatedOptimumValue(Vector &left, idx_t lidx, idx_t lcount, Vector &right, idx_t ridx, idx_t rcount) {
	D_ASSERT(left.GetType() == right.GetType());
	switch (left.GetType().InternalType()) {
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
		return TemplatedOptimumType<int8_t, OP>(left, lidx, lcount, right, ridx, rcount);
	case PhysicalType::INT16:
		return TemplatedOptimumType<int16_t, OP>(left, lidx, lcount, right, ridx, rcount);
	case PhysicalType::INT32:
		return TemplatedOptimumType<int32_t, OP>(left, lidx, lcount, right, ridx, rcount);
	case PhysicalType::INT64:
		return TemplatedOptimumType<int64_t, OP>(left, lidx, lcount, right, ridx, rcount);
	case PhysicalType::UINT8:
		return TemplatedOptimumType<uint8_t, OP>(left, lidx, lcount, right, ridx, rcount);
	case PhysicalType::UINT16:
		return TemplatedOptimumType<uint16_t, OP>(left, lidx, lcount, right, ridx, rcount);
	case PhysicalType::UINT32:
		return TemplatedOptimumType<uint32_t, OP>(left, lidx, lcount, right, ridx, rcount);
	case PhysicalType::UINT64:
		return TemplatedOptimumType<uint64_t, OP>(left, lidx, lcount, right, ridx, rcount);
	case PhysicalType::INT128:
		return TemplatedOptimumType<hugeint_t, OP>(left, lidx, lcount, right, ridx, rcount);
	case PhysicalType::FLOAT:
		return TemplatedOptimumType<float, OP>(left, lidx, lcount, right, ridx, rcount);
	case PhysicalType::DOUBLE:
		return TemplatedOptimumType<double, OP>(left, lidx, lcount, right, ridx, rcount);
	case PhysicalType::INTERVAL:
		return TemplatedOptimumType<interval_t, OP>(left, lidx, lcount, right, ridx, rcount);
	case PhysicalType::VARCHAR:
		return TemplatedOptimumType<string_t, OP>(left, lidx, lcount, right, ridx, rcount);
	case PhysicalType::LIST:
		return TemplatedOptimumList<OP>(left, lidx, lcount, right, ridx, rcount);
	case PhysicalType::STRUCT:
		return TemplatedOptimumStruct<OP>(left, lidx, lcount, right, ridx, rcount);
	default:
		throw InternalException("Invalid type for distinct comparison");
	}
}

template <class OP>
static bool TemplatedOptimumStruct(Vector &left, idx_t lidx_p, idx_t lcount, Vector &right, idx_t ridx_p,
                                   idx_t rcount) {
	// STRUCT dictionaries apply to all the children
	// so map the indexes first
	UnifiedVectorFormat lvdata, rvdata;
	left.ToUnifiedFormat(lcount, lvdata);
	right.ToUnifiedFormat(rcount, rvdata);

	idx_t lidx = lvdata.sel->get_index(lidx_p);
	idx_t ridx = rvdata.sel->get_index(ridx_p);

	// DISTINCT semantics are in effect for nested types
	auto lnull = !lvdata.validity.RowIsValid(lidx);
	auto rnull = !rvdata.validity.RowIsValid(ridx);
	if (lnull || rnull) {
		return OP::Operation(0, 0, lnull, rnull);
	}

	auto &lchildren = StructVector::GetEntries(left);
	auto &rchildren = StructVector::GetEntries(right);

	D_ASSERT(lchildren.size() == rchildren.size());
	for (idx_t col_no = 0; col_no < lchildren.size(); ++col_no) {
		auto &lchild = *lchildren[col_no];
		auto &rchild = *rchildren[col_no];

		// Strict comparisons use the OP for definite
		if (TemplatedOptimumValue<OP>(lchild, lidx_p, lcount, rchild, ridx_p, rcount)) {
			return true;
		}

		if (col_no == lchildren.size() - 1) {
			break;
		}

		// Strict comparisons use IS NOT DISTINCT for possible
		if (!TemplatedOptimumValue<NotDistinctFrom>(lchild, lidx_p, lcount, rchild, ridx_p, rcount)) {
			return false;
		}
	}

	return false;
}

template <class OP>
static bool TemplatedOptimumList(Vector &left, idx_t lidx, idx_t lcount, Vector &right, idx_t ridx, idx_t rcount) {
	UnifiedVectorFormat lvdata, rvdata;
	left.ToUnifiedFormat(lcount, lvdata);
	right.ToUnifiedFormat(rcount, rvdata);

	// Update the indexes and vector sizes for recursion.
	lidx = lvdata.sel->get_index(lidx);
	ridx = rvdata.sel->get_index(ridx);

	lcount = ListVector::GetListSize(left);
	rcount = ListVector::GetListSize(right);

	// DISTINCT semantics are in effect for nested types
	auto lnull = !lvdata.validity.RowIsValid(lidx);
	auto rnull = !rvdata.validity.RowIsValid(ridx);
	if (lnull || rnull) {
		return OP::Operation(0, 0, lnull, rnull);
	}

	auto &lchild = ListVector::GetEntry(left);
	auto &rchild = ListVector::GetEntry(right);

	auto ldata = (const list_entry_t *)lvdata.data;
	auto rdata = (const list_entry_t *)rvdata.data;

	auto &lval = ldata[lidx];
	auto &rval = rdata[ridx];

	for (idx_t pos = 0;; ++pos) {
		// Tie-breaking uses the OP
		if (pos == lval.length || pos == rval.length) {
			return OP::Operation(lval.length, rval.length, false, false);
		}

		// Strict comparisons use the OP for definite
		lidx = lval.offset + pos;
		ridx = rval.offset + pos;
		if (TemplatedOptimumValue<OP>(lchild, lidx, lcount, rchild, ridx, rcount)) {
			return true;
		}

		// Strict comparisons use IS NOT DISTINCT for possible
		if (!TemplatedOptimumValue<NotDistinctFrom>(lchild, lidx, lcount, rchild, ridx, rcount)) {
			return false;
		}
	}

	return false;
}

struct VectorMinMaxState {
	Vector *value;
};

struct VectorMinMaxBase {
	static bool IgnoreNull() {
		return true;
	}

	template <class STATE>
	static void Initialize(STATE *state) {
		state->value = nullptr;
	}

	template <class STATE>
	static void Destroy(AggregateInputData &aggr_input_data, STATE *state) {
		if (state->value) {
			delete state->value;
		}
		state->value = nullptr;
	}

	template <class STATE>
	static void Assign(STATE *state, Vector &input, const idx_t idx) {
		if (!state->value) {
			state->value = new Vector(input.GetType());
			state->value->SetVectorType(VectorType::CONSTANT_VECTOR);
		}
		sel_t selv = idx;
		SelectionVector sel(&selv);
		VectorOperations::Copy(input, *state->value, sel, 1, 0, 0);
	}

	template <class STATE>
	static void Execute(STATE *state, Vector &input, const idx_t idx, const idx_t count) {
		Assign(state, input, idx);
	}

	template <class STATE, class OP>
	static void Update(Vector inputs[], AggregateInputData &, idx_t input_count, Vector &state_vector, idx_t count) {
		auto &input = inputs[0];
		UnifiedVectorFormat idata;
		input.ToUnifiedFormat(count, idata);

		UnifiedVectorFormat sdata;
		state_vector.ToUnifiedFormat(count, sdata);

		auto states = (STATE **)sdata.data;
		for (idx_t i = 0; i < count; i++) {
			const auto idx = idata.sel->get_index(i);
			if (!idata.validity.RowIsValid(idx)) {
				continue;
			}
			const auto sidx = sdata.sel->get_index(i);
			auto state = states[sidx];
			if (!state->value) {
				Assign(state, input, i);
			} else {
				OP::template Execute(state, input, i, count);
			}
		}
	}

	template <class STATE, class OP>
	static void Combine(const STATE &source, STATE *target, AggregateInputData &) {
		if (!source.value) {
			return;
		} else if (!target->value) {
			Assign(target, *source.value, 0);
		} else {
			OP::template Execute(target, *source.value, 0, 1);
		}
	}

	template <class T, class STATE>
	static void Finalize(Vector &result, AggregateInputData &, STATE *state, T *target, ValidityMask &mask, idx_t idx) {
		if (!state->value) {
			// we need to use SetNull here
			// since for STRUCT columns only setting the validity mask of the struct is incorrect
			// as for a struct column, we need to also set ALL child columns to NULL
			switch (result.GetVectorType()) {
			case VectorType::FLAT_VECTOR:
				FlatVector::SetNull(result, idx, true);
				break;
			case VectorType::CONSTANT_VECTOR:
				ConstantVector::SetNull(result, true);
				break;
			default:
				throw InternalException("Invalid result vector type for nested min/max");
			}
		} else {
			VectorOperations::Copy(*state->value, result, 1, 0, idx);
		}
	}

	static unique_ptr<FunctionData> Bind(ClientContext &context, AggregateFunction &function,
	                                     vector<unique_ptr<Expression>> &arguments) {
		function.arguments[0] = arguments[0]->return_type;
		function.return_type = arguments[0]->return_type;
		return nullptr;
	}
};

struct MinOperationVector : public VectorMinMaxBase {
	template <class STATE>
	static void Execute(STATE *state, Vector &input, const idx_t idx, const idx_t count) {
		if (TemplatedOptimumValue<DistinctLessThan>(input, idx, count, *state->value, 0, 1)) {
			Assign(state, input, idx);
		}
	}
};

struct MaxOperationVector : public VectorMinMaxBase {
	template <class STATE>
	static void Execute(STATE *state, Vector &input, const idx_t idx, const idx_t count) {
		if (TemplatedOptimumValue<DistinctGreaterThan>(input, idx, count, *state->value, 0, 1)) {
			Assign(state, input, idx);
		}
	}
};

template <class OP>
unique_ptr<FunctionData> BindDecimalMinMax(ClientContext &context, AggregateFunction &function,
                                           vector<unique_ptr<Expression>> &arguments) {
	auto decimal_type = arguments[0]->return_type;
	auto name = function.name;
	switch (decimal_type.InternalType()) {
	case PhysicalType::INT16:
		function = GetUnaryAggregate<OP>(LogicalType::SMALLINT);
		break;
	case PhysicalType::INT32:
		function = GetUnaryAggregate<OP>(LogicalType::INTEGER);
		break;
	case PhysicalType::INT64:
		function = GetUnaryAggregate<OP>(LogicalType::BIGINT);
		break;
	default:
		function = GetUnaryAggregate<OP>(LogicalType::HUGEINT);
		break;
	}
	function.name = std::move(name);
	function.arguments[0] = decimal_type;
	function.return_type = decimal_type;
	function.order_dependent = AggregateOrderDependent::NOT_ORDER_DEPENDENT;
	return nullptr;
}

template <typename OP, typename STATE>
static AggregateFunction GetMinMaxFunction(const LogicalType &type) {
	return AggregateFunction({type}, type, AggregateFunction::StateSize<STATE>,
	                         AggregateFunction::StateInitialize<STATE, OP>, OP::template Update<STATE, OP>,
	                         AggregateFunction::StateCombine<STATE, OP>,
	                         AggregateFunction::StateFinalize<STATE, void, OP>, nullptr, OP::Bind,
	                         AggregateFunction::StateDestroy<STATE, OP>);
}

template <class OP, class OP_STRING, class OP_VECTOR>
static AggregateFunction GetMinMaxOperator(const LogicalType &type) {
	if (type.InternalType() == PhysicalType::VARCHAR) {
		return AggregateFunction::UnaryAggregateDestructor<MinMaxState<string_t>, string_t, string_t, OP_STRING>(
		    type.id(), type.id());
	} else if (type.InternalType() == PhysicalType::LIST || type.InternalType() == PhysicalType::STRUCT) {
		return GetMinMaxFunction<OP_VECTOR, VectorMinMaxState>(type);
	} else {
		return GetUnaryAggregate<OP>(type);
	}
}

template <class OP, class OP_STRING, class OP_VECTOR>
unique_ptr<FunctionData> BindMinMax(ClientContext &context, AggregateFunction &function,
                                    vector<unique_ptr<Expression>> &arguments) {
	auto input_type = arguments[0]->return_type;
	auto name = std::move(function.name);
	function = GetMinMaxOperator<OP, OP_STRING, OP_VECTOR>(input_type);
	function.name = std::move(name);
	function.order_dependent = AggregateOrderDependent::NOT_ORDER_DEPENDENT;
	if (function.bind) {
		return function.bind(context, function, arguments);
	} else {
		return nullptr;
	}
}

template <class OP, class OP_STRING, class OP_VECTOR>
static void AddMinMaxOperator(AggregateFunctionSet &set) {
	set.AddFunction(AggregateFunction({LogicalTypeId::DECIMAL}, LogicalTypeId::DECIMAL, nullptr, nullptr, nullptr,
	                                  nullptr, nullptr, nullptr, BindDecimalMinMax<OP>));
	set.AddFunction(AggregateFunction({LogicalType::ANY}, LogicalType::ANY, nullptr, nullptr, nullptr, nullptr, nullptr,
	                                  nullptr, BindMinMax<OP, OP_STRING, OP_VECTOR>));
}

AggregateFunctionSet MinFun::GetFunctions() {
	AggregateFunctionSet min("min");
	AddMinMaxOperator<MinOperation, MinOperationString, MinOperationVector>(min);
	return min;
}

AggregateFunctionSet MaxFun::GetFunctions() {
	AggregateFunctionSet max("max");
	AddMinMaxOperator<MaxOperation, MaxOperationString, MaxOperationVector>(max);
	return max;
}

} // namespace duckdb






namespace duckdb {

struct ProductState {
	bool empty;
	double val;
};

struct ProductFunction {
	template <class STATE>
	static void Initialize(STATE *state) {
		state->val = 1;
		state->empty = true;
	}

	template <class STATE, class OP>
	static void Combine(const STATE &source, STATE *target, AggregateInputData &) {
		target->val *= source.val;
		target->empty = target->empty && source.empty;
	}

	template <class T, class STATE>
	static void Finalize(Vector &result, AggregateInputData &, STATE *state, T *target, ValidityMask &mask, idx_t idx) {
		if (state->empty) {
			mask.SetInvalid(idx);
			return;
		}
		target[idx] = state->val;
	}
	template <class INPUT_TYPE, class STATE, class OP>
	static void Operation(STATE *state, AggregateInputData &, INPUT_TYPE *input, ValidityMask &mask, idx_t idx) {
		if (state->empty) {
			state->empty = false;
		}
		state->val *= input[idx];
	}

	template <class INPUT_TYPE, class STATE, class OP>
	static void ConstantOperation(STATE *state, AggregateInputData &aggr_input_data, INPUT_TYPE *input,
	                              ValidityMask &mask, idx_t count) {
		for (idx_t i = 0; i < count; i++) {
			Operation<INPUT_TYPE, STATE, OP>(state, aggr_input_data, input, mask, 0);
		}
	}

	static bool IgnoreNull() {
		return true;
	}
};

AggregateFunction ProductFun::GetFunction() {
	return AggregateFunction::UnaryAggregate<ProductState, double, double, ProductFunction>(
	    LogicalType(LogicalTypeId::DOUBLE), LogicalType::DOUBLE);
}

} // namespace duckdb






namespace duckdb {

struct SkewState {
	size_t n;
	double sum;
	double sum_sqr;
	double sum_cub;
};

struct SkewnessOperation {
	template <class STATE>
	static void Initialize(STATE *state) {
		state->n = 0;
		state->sum = state->sum_sqr = state->sum_cub = 0;
	}

	template <class INPUT_TYPE, class STATE, class OP>
	static void ConstantOperation(STATE *state, AggregateInputData &aggr_input_data, INPUT_TYPE *input,
	                              ValidityMask &mask, idx_t count) {
		for (idx_t i = 0; i < count; i++) {
			Operation<INPUT_TYPE, STATE, OP>(state, aggr_input_data, input, mask, 0);
		}
	}

	template <class INPUT_TYPE, class STATE, class OP>
	static void Operation(STATE *state, AggregateInputData &, INPUT_TYPE *data, ValidityMask &mask, idx_t idx) {
		state->n++;
		state->sum += data[idx];
		state->sum_sqr += pow(data[idx], 2);
		state->sum_cub += pow(data[idx], 3);
	}

	template <class STATE, class OP>
	static void Combine(const STATE &source, STATE *target, AggregateInputData &) {
		if (source.n == 0) {
			return;
		}

		target->n += source.n;
		target->sum += source.sum;
		target->sum_sqr += source.sum_sqr;
		target->sum_cub += source.sum_cub;
	}

	template <class TARGET_TYPE, class STATE>
	static void Finalize(Vector &result, AggregateInputData &, STATE *state, TARGET_TYPE *target, ValidityMask &mask,
	                     idx_t idx) {
		if (state->n <= 2) {
			mask.SetInvalid(idx);
			return;
		}
		double n = state->n;
		double temp = 1 / n;
		auto p = std::pow(temp * (state->sum_sqr - state->sum * state->sum * temp), 3);
		if (p < 0) {
			p = 0; // Shouldn't be below 0 but floating points are weird
		}
		double div = std::sqrt(p);
		if (div == 0) {
			mask.SetInvalid(idx);
			return;
		}
		double temp1 = std::sqrt(n * (n - 1)) / (n - 2);
		target[idx] = temp1 * temp *
		              (state->sum_cub - 3 * state->sum_sqr * state->sum * temp + 2 * pow(state->sum, 3) * temp * temp) /
		              div;
		if (!Value::DoubleIsFinite(target[idx])) {
			throw OutOfRangeException("SKEW is out of range!");
		}
	}

	static bool IgnoreNull() {
		return true;
	}
};

AggregateFunction SkewnessFun::GetFunction() {
	return AggregateFunction::UnaryAggregate<SkewState, double, double, SkewnessOperation>(LogicalType::DOUBLE,
	                                                                                       LogicalType::DOUBLE);
}

} // namespace duckdb









namespace duckdb {

struct StringAggState {
	idx_t size;
	idx_t alloc_size;
	char *dataptr;
};

struct StringAggBindData : public FunctionData {
	explicit StringAggBindData(string sep_p) : sep(std::move(sep_p)) {
	}

	string sep;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<StringAggBindData>(sep);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<StringAggBindData>();
		return sep == other.sep;
	}
};

struct StringAggFunction {
	template <class STATE>
	static void Initialize(STATE *state) {
		state->dataptr = nullptr;
		state->alloc_size = 0;
		state->size = 0;
	}

	template <class T, class STATE>
	static void Finalize(Vector &result, AggregateInputData &, STATE *state, T *target, ValidityMask &mask, idx_t idx) {
		if (!state->dataptr) {
			mask.SetInvalid(idx);
		} else {
			target[idx] = StringVector::AddString(result, state->dataptr, state->size);
		}
	}

	template <class STATE>
	static void Destroy(AggregateInputData &aggr_input_data, STATE *state) {
		if (state->dataptr) {
			delete[] state->dataptr;
		}
	}

	static bool IgnoreNull() {
		return true;
	}

	static inline void PerformOperation(StringAggState *state, const char *str, const char *sep, idx_t str_size,
	                                    idx_t sep_size) {
		if (!state->dataptr) {
			// first iteration: allocate space for the string and copy it into the state
			state->alloc_size = MaxValue<idx_t>(8, NextPowerOfTwo(str_size));
			state->dataptr = new char[state->alloc_size];
			state->size = str_size;
			memcpy(state->dataptr, str, str_size);
		} else {
			// subsequent iteration: first check if we have space to place the string and separator
			idx_t required_size = state->size + str_size + sep_size;
			if (required_size > state->alloc_size) {
				// no space! allocate extra space
				while (state->alloc_size < required_size) {
					state->alloc_size *= 2;
				}
				auto new_data = new char[state->alloc_size];
				memcpy(new_data, state->dataptr, state->size);
				delete[] state->dataptr;
				state->dataptr = new_data;
			}
			// copy the separator
			memcpy(state->dataptr + state->size, sep, sep_size);
			state->size += sep_size;
			// copy the string
			memcpy(state->dataptr + state->size, str, str_size);
			state->size += str_size;
		}
	}

	static inline void PerformOperation(StringAggState *state, string_t str, FunctionData *data_p) {
		auto &data = data_p->Cast<StringAggBindData>();
		PerformOperation(state, str.GetData(), data.sep.c_str(), str.GetSize(), data.sep.size());
	}

	template <class INPUT_TYPE, class STATE, class OP>
	static void Operation(STATE *state, AggregateInputData &aggr_input_data, INPUT_TYPE *str_data,
	                      ValidityMask &str_mask, idx_t str_idx) {
		PerformOperation(state, str_data[str_idx], aggr_input_data.bind_data);
	}

	template <class INPUT_TYPE, class STATE, class OP>
	static void ConstantOperation(STATE *state, AggregateInputData &aggr_input_data, INPUT_TYPE *input,
	                              ValidityMask &mask, idx_t count) {
		for (idx_t i = 0; i < count; i++) {
			Operation<INPUT_TYPE, STATE, OP>(state, aggr_input_data, input, mask, 0);
		}
	}

	template <class STATE, class OP>
	static void Combine(const STATE &source, STATE *target, AggregateInputData &aggr_input_data) {
		if (!source.dataptr) {
			// source is not set: skip combining
			return;
		}
		PerformOperation(target, string_t(source.dataptr, source.size), aggr_input_data.bind_data);
	}
};

unique_ptr<FunctionData> StringAggBind(ClientContext &context, AggregateFunction &function,
                                       vector<unique_ptr<Expression>> &arguments) {
	if (arguments.size() == 1) {
		// single argument: default to comma
		return make_uniq<StringAggBindData>(",");
	}
	D_ASSERT(arguments.size() == 2);
	if (arguments[1]->HasParameter()) {
		throw ParameterNotResolvedException();
	}
	if (!arguments[1]->IsFoldable()) {
		throw BinderException("Separator argument to StringAgg must be a constant");
	}
	auto separator_val = ExpressionExecutor::EvaluateScalar(context, *arguments[1]);
	string separator_string = ",";
	if (separator_val.IsNull()) {
		arguments[0] = make_uniq<BoundConstantExpression>(Value(LogicalType::VARCHAR));
	} else {
		separator_string = separator_val.ToString();
	}
	Function::EraseArgument(function, arguments, arguments.size() - 1);
	return make_uniq<StringAggBindData>(std::move(separator_string));
}

static void StringAggSerialize(FieldWriter &writer, const FunctionData *bind_data_p,
                               const AggregateFunction &function) {
	D_ASSERT(bind_data_p);
	auto bind_data = bind_data_p->Cast<StringAggBindData>();
	writer.WriteString(bind_data.sep);
}

unique_ptr<FunctionData> StringAggDeserialize(ClientContext &context, FieldReader &reader,
                                              AggregateFunction &bound_function) {
	auto sep = reader.ReadRequired<string>();
	return make_uniq<StringAggBindData>(std::move(sep));
}

AggregateFunctionSet StringAggFun::GetFunctions() {
	AggregateFunctionSet string_agg;
	AggregateFunction string_agg_param(
	    {LogicalType::VARCHAR}, LogicalType::VARCHAR, AggregateFunction::StateSize<StringAggState>,
	    AggregateFunction::StateInitialize<StringAggState, StringAggFunction>,
	    AggregateFunction::UnaryScatterUpdate<StringAggState, string_t, StringAggFunction>,
	    AggregateFunction::StateCombine<StringAggState, StringAggFunction>,
	    AggregateFunction::StateFinalize<StringAggState, string_t, StringAggFunction>,
	    AggregateFunction::UnaryUpdate<StringAggState, string_t, StringAggFunction>, StringAggBind,
	    AggregateFunction::StateDestroy<StringAggState, StringAggFunction>);
	string_agg_param.serialize = StringAggSerialize;
	string_agg_param.deserialize = StringAggDeserialize;
	string_agg.AddFunction(string_agg_param);
	string_agg_param.arguments.emplace_back(LogicalType::VARCHAR);
	string_agg.AddFunction(string_agg_param);
	return string_agg;
}

} // namespace duckdb






namespace duckdb {

struct SumSetOperation {
	template <class STATE>
	static void Initialize(STATE *state) {
		state->Initialize();
	}
	template <class STATE>
	static void Combine(const STATE &source, STATE *target, AggregateInputData &) {
		target->Combine(source);
	}
	template <class STATE>
	static void AddValues(STATE *state, idx_t count) {
		state->isset = true;
	}
};

struct IntegerSumOperation : public BaseSumOperation<SumSetOperation, RegularAdd> {
	template <class T, class STATE>
	static void Finalize(Vector &result, AggregateInputData &, STATE *state, T *target, ValidityMask &mask, idx_t idx) {
		if (!state->isset) {
			mask.SetInvalid(idx);
		} else {
			target[idx] = Hugeint::Convert(state->value);
		}
	}
};

struct SumToHugeintOperation : public BaseSumOperation<SumSetOperation, HugeintAdd> {
	template <class T, class STATE>
	static void Finalize(Vector &result, AggregateInputData &, STATE *state, T *target, ValidityMask &mask, idx_t idx) {
		if (!state->isset) {
			mask.SetInvalid(idx);
		} else {
			target[idx] = state->value;
		}
	}
};

template <class ADD_OPERATOR>
struct DoubleSumOperation : public BaseSumOperation<SumSetOperation, ADD_OPERATOR> {
	template <class T, class STATE>
	static void Finalize(Vector &result, AggregateInputData &, STATE *state, T *target, ValidityMask &mask, idx_t idx) {
		if (!state->isset) {
			mask.SetInvalid(idx);
		} else {
			target[idx] = state->value;
		}
	}
};

using NumericSumOperation = DoubleSumOperation<RegularAdd>;
using KahanSumOperation = DoubleSumOperation<KahanAdd>;

struct HugeintSumOperation : public BaseSumOperation<SumSetOperation, RegularAdd> {
	template <class T, class STATE>
	static void Finalize(Vector &result, AggregateInputData &, STATE *state, T *target, ValidityMask &mask, idx_t idx) {
		if (!state->isset) {
			mask.SetInvalid(idx);
		} else {
			target[idx] = state->value;
		}
	}
};

AggregateFunction GetSumAggregateNoOverflow(PhysicalType type) {
	switch (type) {
	case PhysicalType::INT32: {
		auto function = AggregateFunction::UnaryAggregate<SumState<int64_t>, int32_t, hugeint_t, IntegerSumOperation>(
		    LogicalType::INTEGER, LogicalType::HUGEINT);
		function.name = "sum_no_overflow";
		function.order_dependent = AggregateOrderDependent::NOT_ORDER_DEPENDENT;
		return function;
	}
	case PhysicalType::INT64: {
		auto function = AggregateFunction::UnaryAggregate<SumState<int64_t>, int64_t, hugeint_t, IntegerSumOperation>(
		    LogicalType::BIGINT, LogicalType::HUGEINT);
		function.name = "sum_no_overflow";
		function.order_dependent = AggregateOrderDependent::NOT_ORDER_DEPENDENT;
		return function;
	}
	default:
		throw BinderException("Unsupported internal type for sum_no_overflow");
	}
}

unique_ptr<BaseStatistics> SumPropagateStats(ClientContext &context, BoundAggregateExpression &expr,
                                             AggregateStatisticsInput &input) {
	if (input.node_stats && input.node_stats->has_max_cardinality) {
		auto &numeric_stats = input.child_stats[0];
		if (!NumericStats::HasMinMax(numeric_stats)) {
			return nullptr;
		}
		auto internal_type = numeric_stats.GetType().InternalType();
		hugeint_t max_negative;
		hugeint_t max_positive;
		switch (internal_type) {
		case PhysicalType::INT32:
			max_negative = NumericStats::Min(numeric_stats).GetValueUnsafe<int32_t>();
			max_positive = NumericStats::Max(numeric_stats).GetValueUnsafe<int32_t>();
			break;
		case PhysicalType::INT64:
			max_negative = NumericStats::Min(numeric_stats).GetValueUnsafe<int64_t>();
			max_positive = NumericStats::Max(numeric_stats).GetValueUnsafe<int64_t>();
			break;
		default:
			throw InternalException("Unsupported type for propagate sum stats");
		}
		auto max_sum_negative = max_negative * hugeint_t(input.node_stats->max_cardinality);
		auto max_sum_positive = max_positive * hugeint_t(input.node_stats->max_cardinality);
		if (max_sum_positive >= NumericLimits<int64_t>::Maximum() ||
		    max_sum_negative <= NumericLimits<int64_t>::Minimum()) {
			// sum can potentially exceed int64_t bounds: use hugeint sum
			return nullptr;
		}
		// total sum is guaranteed to fit in a single int64: use int64 sum instead of hugeint sum
		expr.function = GetSumAggregateNoOverflow(internal_type);
	}
	return nullptr;
}

AggregateFunction GetSumAggregate(PhysicalType type) {
	switch (type) {
	case PhysicalType::INT16: {
		auto function = AggregateFunction::UnaryAggregate<SumState<int64_t>, int16_t, hugeint_t, IntegerSumOperation>(
		    LogicalType::SMALLINT, LogicalType::HUGEINT);
		function.order_dependent = AggregateOrderDependent::NOT_ORDER_DEPENDENT;
		return function;
	}

	case PhysicalType::INT32: {
		auto function =
		    AggregateFunction::UnaryAggregate<SumState<hugeint_t>, int32_t, hugeint_t, SumToHugeintOperation>(
		        LogicalType::INTEGER, LogicalType::HUGEINT);
		function.statistics = SumPropagateStats;
		function.order_dependent = AggregateOrderDependent::NOT_ORDER_DEPENDENT;
		return function;
	}
	case PhysicalType::INT64: {
		auto function =
		    AggregateFunction::UnaryAggregate<SumState<hugeint_t>, int64_t, hugeint_t, SumToHugeintOperation>(
		        LogicalType::BIGINT, LogicalType::HUGEINT);
		function.statistics = SumPropagateStats;
		function.order_dependent = AggregateOrderDependent::NOT_ORDER_DEPENDENT;
		return function;
	}
	case PhysicalType::INT128: {
		auto function =
		    AggregateFunction::UnaryAggregate<SumState<hugeint_t>, hugeint_t, hugeint_t, HugeintSumOperation>(
		        LogicalType::HUGEINT, LogicalType::HUGEINT);
		function.order_dependent = AggregateOrderDependent::NOT_ORDER_DEPENDENT;
		return function;
	}
	default:
		throw InternalException("Unimplemented sum aggregate");
	}
}

unique_ptr<FunctionData> BindDecimalSum(ClientContext &context, AggregateFunction &function,
                                        vector<unique_ptr<Expression>> &arguments) {
	auto decimal_type = arguments[0]->return_type;
	function = GetSumAggregate(decimal_type.InternalType());
	function.name = "sum";
	function.arguments[0] = decimal_type;
	function.return_type = LogicalType::DECIMAL(Decimal::MAX_WIDTH_DECIMAL, DecimalType::GetScale(decimal_type));
	function.order_dependent = AggregateOrderDependent::NOT_ORDER_DEPENDENT;
	return nullptr;
}

unique_ptr<FunctionData> BindDecimalSumNoOverflow(ClientContext &context, AggregateFunction &function,
                                                  vector<unique_ptr<Expression>> &arguments) {
	auto decimal_type = arguments[0]->return_type;
	function = GetSumAggregateNoOverflow(decimal_type.InternalType());
	function.name = "sum_no_overflow";
	function.arguments[0] = decimal_type;
	function.return_type = LogicalType::DECIMAL(Decimal::MAX_WIDTH_DECIMAL, DecimalType::GetScale(decimal_type));
	function.order_dependent = AggregateOrderDependent::NOT_ORDER_DEPENDENT;
	return nullptr;
}

AggregateFunctionSet SumFun::GetFunctions() {
	AggregateFunctionSet sum;
	// decimal
	sum.AddFunction(AggregateFunction({LogicalTypeId::DECIMAL}, LogicalTypeId::DECIMAL, nullptr, nullptr, nullptr,
	                                  nullptr, nullptr, FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr,
	                                  BindDecimalSum));
	sum.AddFunction(GetSumAggregate(PhysicalType::INT16));
	sum.AddFunction(GetSumAggregate(PhysicalType::INT32));
	sum.AddFunction(GetSumAggregate(PhysicalType::INT64));
	sum.AddFunction(GetSumAggregate(PhysicalType::INT128));
	sum.AddFunction(AggregateFunction::UnaryAggregate<SumState<double>, double, double, NumericSumOperation>(
	    LogicalType::DOUBLE, LogicalType::DOUBLE));
	return sum;
}

AggregateFunctionSet SumNoOverflowFun::GetFunctions() {
	AggregateFunctionSet sum_no_overflow;
	sum_no_overflow.AddFunction(GetSumAggregateNoOverflow(PhysicalType::INT32));
	sum_no_overflow.AddFunction(GetSumAggregateNoOverflow(PhysicalType::INT64));
	sum_no_overflow.AddFunction(
	    AggregateFunction({LogicalTypeId::DECIMAL}, LogicalTypeId::DECIMAL, nullptr, nullptr, nullptr, nullptr, nullptr,
	                      FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr, BindDecimalSumNoOverflow));
	return sum_no_overflow;
}

AggregateFunction KahanSumFun::GetFunction() {
	return AggregateFunction::UnaryAggregate<KahanSumState, double, double, KahanSumOperation>(LogicalType::DOUBLE,
	                                                                                           LogicalType::DOUBLE);
}

} // namespace duckdb







#include <algorithm>
#include <cmath>
#include <stdlib.h>

namespace duckdb {

struct ApproxQuantileState {
	duckdb_tdigest::TDigest *h;
	idx_t pos;
};

struct ApproximateQuantileBindData : public FunctionData {
	explicit ApproximateQuantileBindData(float quantile_p) : quantiles(1, quantile_p) {
	}

	explicit ApproximateQuantileBindData(vector<float> quantiles_p) : quantiles(std::move(quantiles_p)) {
	}

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<ApproximateQuantileBindData>(quantiles);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<ApproximateQuantileBindData>();
		//		return quantiles == other.quantiles;
		if (quantiles != other.quantiles) {
			return false;
		}
		return true;
	}

	static void Serialize(FieldWriter &writer, const FunctionData *bind_data_p, const AggregateFunction &function) {
		D_ASSERT(bind_data_p);
		auto bind_data = (ApproximateQuantileBindData *)bind_data_p;
		writer.WriteList<float>(bind_data->quantiles);
	}

	static unique_ptr<FunctionData> Deserialize(ClientContext &context, FieldReader &reader,
	                                            AggregateFunction &bound_function) {
		auto quantiles = reader.ReadRequiredList<float>();
		return make_uniq<ApproximateQuantileBindData>(std::move(quantiles));
	}

	vector<float> quantiles;
};

struct ApproxQuantileOperation {
	using SAVE_TYPE = duckdb_tdigest::Value;

	template <class STATE>
	static void Initialize(STATE *state) {
		state->pos = 0;
		state->h = nullptr;
	}

	template <class INPUT_TYPE, class STATE, class OP>
	static void ConstantOperation(STATE *state, AggregateInputData &aggr_input_data, INPUT_TYPE *input,
	                              ValidityMask &mask, idx_t count) {
		for (idx_t i = 0; i < count; i++) {
			Operation<INPUT_TYPE, STATE, OP>(state, aggr_input_data, input, mask, 0);
		}
	}

	template <class INPUT_TYPE, class STATE, class OP>
	static void Operation(STATE *state, AggregateInputData &, INPUT_TYPE *data, ValidityMask &mask, idx_t idx) {
		auto val = Cast::template Operation<INPUT_TYPE, SAVE_TYPE>(data[idx]);
		if (!Value::DoubleIsFinite(val)) {
			return;
		}
		if (!state->h) {
			state->h = new duckdb_tdigest::TDigest(100);
		}
		state->h->add(val);
		state->pos++;
	}

	template <class STATE, class OP>
	static void Combine(const STATE &source, STATE *target, AggregateInputData &) {
		if (source.pos == 0) {
			return;
		}
		D_ASSERT(source.h);
		if (!target->h) {
			target->h = new duckdb_tdigest::TDigest(100);
		}
		target->h->merge(source.h);
		target->pos += source.pos;
	}

	template <class STATE>
	static void Destroy(AggregateInputData &aggr_input_data, STATE *state) {
		if (state->h) {
			delete state->h;
		}
	}

	static bool IgnoreNull() {
		return true;
	}
};

struct ApproxQuantileScalarOperation : public ApproxQuantileOperation {

	template <class TARGET_TYPE, class STATE>
	static void Finalize(Vector &result, AggregateInputData &aggr_input_data, STATE *state, TARGET_TYPE *target,
	                     ValidityMask &mask, idx_t idx) {

		if (state->pos == 0) {
			mask.SetInvalid(idx);
			return;
		}
		D_ASSERT(state->h);
		D_ASSERT(aggr_input_data.bind_data);
		state->h->compress();
		auto bind_data = (ApproximateQuantileBindData *)aggr_input_data.bind_data;
		D_ASSERT(bind_data->quantiles.size() == 1);
		target[idx] = Cast::template Operation<SAVE_TYPE, TARGET_TYPE>(state->h->quantile(bind_data->quantiles[0]));
	}
};

AggregateFunction GetApproximateQuantileAggregateFunction(PhysicalType type) {
	switch (type) {
	case PhysicalType::INT16:
		return AggregateFunction::UnaryAggregateDestructor<ApproxQuantileState, int16_t, int16_t,
		                                                   ApproxQuantileScalarOperation>(LogicalType::SMALLINT,
		                                                                                  LogicalType::SMALLINT);
	case PhysicalType::INT32:
		return AggregateFunction::UnaryAggregateDestructor<ApproxQuantileState, int32_t, int32_t,
		                                                   ApproxQuantileScalarOperation>(LogicalType::INTEGER,
		                                                                                  LogicalType::INTEGER);
	case PhysicalType::INT64:
		return AggregateFunction::UnaryAggregateDestructor<ApproxQuantileState, int64_t, int64_t,
		                                                   ApproxQuantileScalarOperation>(LogicalType::BIGINT,
		                                                                                  LogicalType::BIGINT);
	case PhysicalType::INT128:
		return AggregateFunction::UnaryAggregateDestructor<ApproxQuantileState, hugeint_t, hugeint_t,
		                                                   ApproxQuantileScalarOperation>(LogicalType::HUGEINT,
		                                                                                  LogicalType::HUGEINT);
	case PhysicalType::DOUBLE:
		return AggregateFunction::UnaryAggregateDestructor<ApproxQuantileState, double, double,
		                                                   ApproxQuantileScalarOperation>(LogicalType::DOUBLE,
		                                                                                  LogicalType::DOUBLE);
	default:
		throw InternalException("Unimplemented quantile aggregate");
	}
}

static float CheckApproxQuantile(const Value &quantile_val) {
	if (quantile_val.IsNull()) {
		throw BinderException("APPROXIMATE QUANTILE parameter cannot be NULL");
	}
	auto quantile = quantile_val.GetValue<float>();
	if (quantile < 0 || quantile > 1) {
		throw BinderException("APPROXIMATE QUANTILE can only take parameters in range [0, 1]");
	}

	return quantile;
}

unique_ptr<FunctionData> BindApproxQuantile(ClientContext &context, AggregateFunction &function,
                                            vector<unique_ptr<Expression>> &arguments) {
	if (arguments[1]->HasParameter()) {
		throw ParameterNotResolvedException();
	}
	if (!arguments[1]->IsFoldable()) {
		throw BinderException("APPROXIMATE QUANTILE can only take constant quantile parameters");
	}
	Value quantile_val = ExpressionExecutor::EvaluateScalar(context, *arguments[1]);

	vector<float> quantiles;
	if (quantile_val.type().id() != LogicalTypeId::LIST) {
		quantiles.push_back(CheckApproxQuantile(quantile_val));
	} else {
		for (const auto &element_val : ListValue::GetChildren(quantile_val)) {
			quantiles.push_back(CheckApproxQuantile(element_val));
		}
	}

	// remove the quantile argument so we can use the unary aggregate
	Function::EraseArgument(function, arguments, arguments.size() - 1);
	return make_uniq<ApproximateQuantileBindData>(quantiles);
}

unique_ptr<FunctionData> BindApproxQuantileDecimal(ClientContext &context, AggregateFunction &function,
                                                   vector<unique_ptr<Expression>> &arguments) {
	auto bind_data = BindApproxQuantile(context, function, arguments);
	function = GetApproximateQuantileAggregateFunction(arguments[0]->return_type.InternalType());
	function.name = "approx_quantile";
	function.serialize = ApproximateQuantileBindData::Serialize;
	function.deserialize = ApproximateQuantileBindData::Deserialize;
	return bind_data;
}

AggregateFunction GetApproximateQuantileAggregate(PhysicalType type) {
	auto fun = GetApproximateQuantileAggregateFunction(type);
	fun.bind = BindApproxQuantile;
	fun.serialize = ApproximateQuantileBindData::Serialize;
	fun.deserialize = ApproximateQuantileBindData::Deserialize;
	// temporarily push an argument so we can bind the actual quantile
	fun.arguments.emplace_back(LogicalType::FLOAT);
	return fun;
}

template <class CHILD_TYPE>
struct ApproxQuantileListOperation : public ApproxQuantileOperation {

	template <class RESULT_TYPE, class STATE>
	static void Finalize(Vector &result_list, AggregateInputData &aggr_input_data, STATE *state, RESULT_TYPE *target,
	                     ValidityMask &mask, idx_t idx) {
		if (state->pos == 0) {
			mask.SetInvalid(idx);
			return;
		}

		D_ASSERT(aggr_input_data.bind_data);
		auto bind_data = (ApproximateQuantileBindData *)aggr_input_data.bind_data;

		auto &result = ListVector::GetEntry(result_list);
		auto ridx = ListVector::GetListSize(result_list);
		ListVector::Reserve(result_list, ridx + bind_data->quantiles.size());
		auto rdata = FlatVector::GetData<CHILD_TYPE>(result);

		D_ASSERT(state->h);
		state->h->compress();

		auto &entry = target[idx];
		entry.offset = ridx;
		entry.length = bind_data->quantiles.size();
		for (size_t q = 0; q < entry.length; ++q) {
			const auto &quantile = bind_data->quantiles[q];
			rdata[ridx + q] = Cast::template Operation<SAVE_TYPE, CHILD_TYPE>(state->h->quantile(quantile));
		}

		ListVector::SetListSize(result_list, entry.offset + entry.length);
	}

	template <class STATE_TYPE, class RESULT_TYPE>
	static void FinalizeList(Vector &states, AggregateInputData &aggr_input_data, Vector &result, idx_t count, // NOLINT
	                         idx_t offset) {
		D_ASSERT(result.GetType().id() == LogicalTypeId::LIST);

		D_ASSERT(aggr_input_data.bind_data);
		auto bind_data = (ApproximateQuantileBindData *)aggr_input_data.bind_data;

		if (states.GetVectorType() == VectorType::CONSTANT_VECTOR) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
			ListVector::Reserve(result, bind_data->quantiles.size());

			auto sdata = ConstantVector::GetData<STATE_TYPE *>(states);
			auto rdata = ConstantVector::GetData<RESULT_TYPE>(result);
			auto &mask = ConstantVector::Validity(result);
			Finalize<RESULT_TYPE, STATE_TYPE>(result, aggr_input_data, sdata[0], rdata, mask, 0);
		} else {
			D_ASSERT(states.GetVectorType() == VectorType::FLAT_VECTOR);
			result.SetVectorType(VectorType::FLAT_VECTOR);
			ListVector::Reserve(result, (offset + count) * bind_data->quantiles.size());

			auto sdata = FlatVector::GetData<STATE_TYPE *>(states);
			auto rdata = FlatVector::GetData<RESULT_TYPE>(result);
			auto &mask = FlatVector::Validity(result);
			for (idx_t i = 0; i < count; i++) {
				Finalize<RESULT_TYPE, STATE_TYPE>(result, aggr_input_data, sdata[i], rdata, mask, i + offset);
			}
		}

		result.Verify(count);
	}
};

template <class STATE, class INPUT_TYPE, class RESULT_TYPE, class OP>
static AggregateFunction ApproxQuantileListAggregate(const LogicalType &input_type, const LogicalType &child_type) {
	LogicalType result_type = LogicalType::LIST(child_type);
	return AggregateFunction(
	    {input_type}, result_type, AggregateFunction::StateSize<STATE>, AggregateFunction::StateInitialize<STATE, OP>,
	    AggregateFunction::UnaryScatterUpdate<STATE, INPUT_TYPE, OP>, AggregateFunction::StateCombine<STATE, OP>,
	    OP::template FinalizeList<STATE, RESULT_TYPE>, AggregateFunction::UnaryUpdate<STATE, INPUT_TYPE, OP>, nullptr,
	    AggregateFunction::StateDestroy<STATE, OP>);
}

template <typename INPUT_TYPE, typename SAVE_TYPE>
AggregateFunction GetTypedApproxQuantileListAggregateFunction(const LogicalType &type) {
	using STATE = ApproxQuantileState;
	using OP = ApproxQuantileListOperation<INPUT_TYPE>;
	auto fun = ApproxQuantileListAggregate<STATE, INPUT_TYPE, list_entry_t, OP>(type, type);
	fun.serialize = ApproximateQuantileBindData::Serialize;
	fun.deserialize = ApproximateQuantileBindData::Deserialize;
	return fun;
}

AggregateFunction GetApproxQuantileListAggregateFunction(const LogicalType &type) {
	switch (type.id()) {
	case LogicalTypeId::TINYINT:
		return GetTypedApproxQuantileListAggregateFunction<int8_t, int8_t>(type);
	case LogicalTypeId::SMALLINT:
		return GetTypedApproxQuantileListAggregateFunction<int16_t, int16_t>(type);
	case LogicalTypeId::INTEGER:
		return GetTypedApproxQuantileListAggregateFunction<int32_t, int32_t>(type);
	case LogicalTypeId::BIGINT:
		return GetTypedApproxQuantileListAggregateFunction<int64_t, int64_t>(type);
	case LogicalTypeId::HUGEINT:
		return GetTypedApproxQuantileListAggregateFunction<hugeint_t, hugeint_t>(type);
	case LogicalTypeId::FLOAT:
		return GetTypedApproxQuantileListAggregateFunction<float, float>(type);
	case LogicalTypeId::DOUBLE:
		return GetTypedApproxQuantileListAggregateFunction<double, double>(type);
	case LogicalTypeId::DECIMAL:
		switch (type.InternalType()) {
		case PhysicalType::INT16:
			return GetTypedApproxQuantileListAggregateFunction<int16_t, int16_t>(type);
		case PhysicalType::INT32:
			return GetTypedApproxQuantileListAggregateFunction<int32_t, int32_t>(type);
		case PhysicalType::INT64:
			return GetTypedApproxQuantileListAggregateFunction<int64_t, int64_t>(type);
		case PhysicalType::INT128:
			return GetTypedApproxQuantileListAggregateFunction<hugeint_t, hugeint_t>(type);
		default:
			throw NotImplementedException("Unimplemented approximate quantile list aggregate");
		}
	default:
		// TODO: Add quantitative temporal types
		throw NotImplementedException("Unimplemented approximate quantile list aggregate");
	}
}

unique_ptr<FunctionData> BindApproxQuantileDecimalList(ClientContext &context, AggregateFunction &function,
                                                       vector<unique_ptr<Expression>> &arguments) {
	auto bind_data = BindApproxQuantile(context, function, arguments);
	function = GetApproxQuantileListAggregateFunction(arguments[0]->return_type);
	function.name = "approx_quantile";
	function.serialize = ApproximateQuantileBindData::Serialize;
	function.deserialize = ApproximateQuantileBindData::Deserialize;
	return bind_data;
}

AggregateFunction GetApproxQuantileListAggregate(const LogicalType &type) {
	auto fun = GetApproxQuantileListAggregateFunction(type);
	fun.bind = BindApproxQuantile;
	fun.serialize = ApproximateQuantileBindData::Serialize;
	fun.deserialize = ApproximateQuantileBindData::Deserialize;
	// temporarily push an argument so we can bind the actual quantile
	auto list_of_float = LogicalType::LIST(LogicalType::FLOAT);
	fun.arguments.push_back(list_of_float);
	return fun;
}

AggregateFunctionSet ApproxQuantileFun::GetFunctions() {
	AggregateFunctionSet approx_quantile;
	approx_quantile.AddFunction(AggregateFunction({LogicalTypeId::DECIMAL, LogicalType::FLOAT}, LogicalTypeId::DECIMAL,
	                                              nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
	                                              BindApproxQuantileDecimal));

	approx_quantile.AddFunction(GetApproximateQuantileAggregate(PhysicalType::INT16));
	approx_quantile.AddFunction(GetApproximateQuantileAggregate(PhysicalType::INT32));
	approx_quantile.AddFunction(GetApproximateQuantileAggregate(PhysicalType::INT64));
	approx_quantile.AddFunction(GetApproximateQuantileAggregate(PhysicalType::INT128));
	approx_quantile.AddFunction(GetApproximateQuantileAggregate(PhysicalType::DOUBLE));

	// List variants
	approx_quantile.AddFunction(AggregateFunction({LogicalTypeId::DECIMAL, LogicalType::LIST(LogicalType::FLOAT)},
	                                              LogicalType::LIST(LogicalTypeId::DECIMAL), nullptr, nullptr, nullptr,
	                                              nullptr, nullptr, nullptr, BindApproxQuantileDecimalList));

	approx_quantile.AddFunction(GetApproxQuantileListAggregate(LogicalTypeId::TINYINT));
	approx_quantile.AddFunction(GetApproxQuantileListAggregate(LogicalTypeId::SMALLINT));
	approx_quantile.AddFunction(GetApproxQuantileListAggregate(LogicalTypeId::INTEGER));
	approx_quantile.AddFunction(GetApproxQuantileListAggregate(LogicalTypeId::BIGINT));
	approx_quantile.AddFunction(GetApproxQuantileListAggregate(LogicalTypeId::HUGEINT));
	approx_quantile.AddFunction(GetApproxQuantileListAggregate(LogicalTypeId::FLOAT));
	approx_quantile.AddFunction(GetApproxQuantileListAggregate(LogicalTypeId::DOUBLE));
	return approx_quantile;
}

} // namespace duckdb
// MODE( <expr1> )
// Returns the most frequent value for the values within expr1.
// NULL values are ignored. If all the values are NULL, or there are 0 rows, then the function returns NULL.








#include <functional>

namespace std {

template <>
struct hash<duckdb::interval_t> {
	inline size_t operator()(const duckdb::interval_t &val) const {
		return hash<int32_t> {}(val.days) ^ hash<int32_t> {}(val.months) ^ hash<int64_t> {}(val.micros);
	}
};

template <>
struct hash<duckdb::hugeint_t> {
	inline size_t operator()(const duckdb::hugeint_t &val) const {
		return hash<int64_t> {}(val.upper) ^ hash<int64_t> {}(val.lower);
	}
};

} // namespace std

namespace duckdb {

using FrameBounds = std::pair<idx_t, idx_t>;

template <class KEY_TYPE>
struct ModeState {
	struct ModeAttr {
		ModeAttr() : count(0), first_row(std::numeric_limits<idx_t>::max()) {
		}
		size_t count;
		idx_t first_row;
	};
	using Counts = unordered_map<KEY_TYPE, ModeAttr>;

	Counts *frequency_map;
	KEY_TYPE *mode;
	size_t nonzero;
	bool valid;
	size_t count;

	void Initialize() {
		frequency_map = nullptr;
		mode = nullptr;
		nonzero = 0;
		valid = false;
		count = 0;
	}

	void Destroy() {
		if (frequency_map) {
			delete frequency_map;
		}
		if (mode) {
			delete mode;
		}
	}

	void Reset() {
		Counts empty;
		frequency_map->swap(empty);
		nonzero = 0;
		count = 0;
		valid = false;
	}

	void ModeAdd(const KEY_TYPE &key, idx_t row) {
		auto &attr = (*frequency_map)[key];
		auto new_count = (attr.count += 1);
		if (new_count == 1) {
			++nonzero;
			attr.first_row = row;
		} else {
			attr.first_row = MinValue(row, attr.first_row);
		}
		if (new_count > count) {
			valid = true;
			count = new_count;
			if (mode) {
				*mode = key;
			} else {
				mode = new KEY_TYPE(key);
			}
		}
	}

	void ModeRm(const KEY_TYPE &key, idx_t frame) {
		auto &attr = (*frequency_map)[key];
		auto old_count = attr.count;
		nonzero -= int(old_count == 1);

		attr.count -= 1;
		if (count == old_count && key == *mode) {
			valid = false;
		}
	}

	typename Counts::const_iterator Scan() const {
		//! Initialize control variables to first variable of the frequency map
		auto highest_frequency = frequency_map->begin();
		for (auto i = highest_frequency; i != frequency_map->end(); ++i) {
			// Tie break with the lowest insert position
			if (i->second.count > highest_frequency->second.count ||
			    (i->second.count == highest_frequency->second.count &&
			     i->second.first_row < highest_frequency->second.first_row)) {
				highest_frequency = i;
			}
		}
		return highest_frequency;
	}
};

struct ModeIncluded {
	inline explicit ModeIncluded(const ValidityMask &fmask_p, const ValidityMask &dmask_p, idx_t bias_p)
	    : fmask(fmask_p), dmask(dmask_p), bias(bias_p) {
	}

	inline bool operator()(const idx_t &idx) const {
		return fmask.RowIsValid(idx) && dmask.RowIsValid(idx - bias);
	}
	const ValidityMask &fmask;
	const ValidityMask &dmask;
	const idx_t bias;
};

struct ModeAssignmentStandard {
	template <class INPUT_TYPE, class RESULT_TYPE>
	static RESULT_TYPE Assign(Vector &result, INPUT_TYPE input) {
		return RESULT_TYPE(input);
	}
};

struct ModeAssignmentString {
	template <class INPUT_TYPE, class RESULT_TYPE>
	static RESULT_TYPE Assign(Vector &result, INPUT_TYPE input) {
		return StringVector::AddString(result, input);
	}
};

template <typename KEY_TYPE, typename ASSIGN_OP>
struct ModeFunction {
	template <class STATE>
	static void Initialize(STATE *state) {
		state->Initialize();
	}

	template <class INPUT_TYPE, class STATE, class OP>
	static void Operation(STATE *state, AggregateInputData &, INPUT_TYPE *input, ValidityMask &mask, idx_t idx) {
		if (!state->frequency_map) {
			state->frequency_map = new typename STATE::Counts();
		}
		auto key = KEY_TYPE(input[idx]);
		auto &i = (*state->frequency_map)[key];
		i.count++;
		i.first_row = MinValue<idx_t>(i.first_row, state->count);
		state->count++;
	}

	template <class STATE, class OP>
	static void Combine(const STATE &source, STATE *target, AggregateInputData &) {
		if (!source.frequency_map) {
			return;
		}
		if (!target->frequency_map) {
			// Copy - don't destroy! Otherwise windowing will break.
			target->frequency_map = new typename STATE::Counts(*source.frequency_map);
			return;
		}
		for (auto &val : *source.frequency_map) {
			auto &i = (*target->frequency_map)[val.first];
			i.count += val.second.count;
			i.first_row = MinValue(i.first_row, val.second.first_row);
		}
		target->count += source.count;
	}

	template <class INPUT_TYPE, class STATE>
	static void Finalize(Vector &result, AggregateInputData &, STATE *state, INPUT_TYPE *target, ValidityMask &mask,
	                     idx_t idx) {
		if (!state->frequency_map) {
			mask.SetInvalid(idx);
			return;
		}
		auto highest_frequency = state->Scan();
		if (highest_frequency != state->frequency_map->end()) {
			target[idx] = ASSIGN_OP::template Assign<INPUT_TYPE, INPUT_TYPE>(result, highest_frequency->first);
		} else {
			mask.SetInvalid(idx);
		}
	}
	template <class INPUT_TYPE, class STATE, class OP>
	static void ConstantOperation(STATE *state, AggregateInputData &, INPUT_TYPE *input, ValidityMask &mask,
	                              idx_t count) {
		if (!state->frequency_map) {
			state->frequency_map = new typename STATE::Counts();
		}
		auto key = KEY_TYPE(input[0]);
		auto &i = (*state->frequency_map)[key];
		i.count += count;
		i.first_row = MinValue<idx_t>(i.first_row, state->count);
		state->count += count;
	}

	template <class STATE, class INPUT_TYPE, class RESULT_TYPE>
	static void Window(const INPUT_TYPE *data, const ValidityMask &fmask, const ValidityMask &dmask,
	                   AggregateInputData &, STATE *state, const FrameBounds &frame, const FrameBounds &prev,
	                   Vector &result, idx_t rid, idx_t bias) {
		auto rdata = FlatVector::GetData<RESULT_TYPE>(result);
		auto &rmask = FlatVector::Validity(result);

		ModeIncluded included(fmask, dmask, bias);

		if (!state->frequency_map) {
			state->frequency_map = new typename STATE::Counts;
		}
		const double tau = .25;
		if (state->nonzero <= tau * state->frequency_map->size()) {
			state->Reset();
			// for f ∈ F do
			for (auto f = frame.first; f < frame.second; ++f) {
				if (included(f)) {
					state->ModeAdd(KEY_TYPE(data[f]), f);
				}
			}
		} else {
			// for f ∈ P \ F do
			for (auto p = prev.first; p < frame.first; ++p) {
				if (included(p)) {
					state->ModeRm(KEY_TYPE(data[p]), p);
				}
			}
			for (auto p = frame.second; p < prev.second; ++p) {
				if (included(p)) {
					state->ModeRm(KEY_TYPE(data[p]), p);
				}
			}

			// for f ∈ F \ P do
			for (auto f = frame.first; f < prev.first; ++f) {
				if (included(f)) {
					state->ModeAdd(KEY_TYPE(data[f]), f);
				}
			}
			for (auto f = prev.second; f < frame.second; ++f) {
				if (included(f)) {
					state->ModeAdd(KEY_TYPE(data[f]), f);
				}
			}
		}

		if (!state->valid) {
			// Rescan
			auto highest_frequency = state->Scan();
			if (highest_frequency != state->frequency_map->end()) {
				*(state->mode) = highest_frequency->first;
				state->count = highest_frequency->second.count;
				state->valid = (state->count > 0);
			}
		}

		if (state->valid) {
			rdata[rid] = ASSIGN_OP::template Assign<INPUT_TYPE, RESULT_TYPE>(result, *state->mode);
		} else {
			rmask.Set(rid, false);
		}
	}

	static bool IgnoreNull() {
		return true;
	}

	template <class STATE>
	static void Destroy(AggregateInputData &aggr_input_data, STATE *state) {
		state->Destroy();
	}
};

template <typename INPUT_TYPE, typename KEY_TYPE, typename ASSIGN_OP = ModeAssignmentStandard>
AggregateFunction GetTypedModeFunction(const LogicalType &type) {
	using STATE = ModeState<KEY_TYPE>;
	using OP = ModeFunction<KEY_TYPE, ASSIGN_OP>;
	auto func = AggregateFunction::UnaryAggregateDestructor<STATE, INPUT_TYPE, INPUT_TYPE, OP>(type, type);
	func.window = AggregateFunction::UnaryWindow<STATE, INPUT_TYPE, INPUT_TYPE, OP>;
	return func;
}

AggregateFunction GetModeAggregate(const LogicalType &type) {
	switch (type.InternalType()) {
	case PhysicalType::INT8:
		return GetTypedModeFunction<int8_t, int8_t>(type);
	case PhysicalType::UINT8:
		return GetTypedModeFunction<uint8_t, uint8_t>(type);
	case PhysicalType::INT16:
		return GetTypedModeFunction<int16_t, int16_t>(type);
	case PhysicalType::UINT16:
		return GetTypedModeFunction<uint16_t, uint16_t>(type);
	case PhysicalType::INT32:
		return GetTypedModeFunction<int32_t, int32_t>(type);
	case PhysicalType::UINT32:
		return GetTypedModeFunction<uint32_t, uint32_t>(type);
	case PhysicalType::INT64:
		return GetTypedModeFunction<int64_t, int64_t>(type);
	case PhysicalType::UINT64:
		return GetTypedModeFunction<uint64_t, uint64_t>(type);
	case PhysicalType::INT128:
		return GetTypedModeFunction<hugeint_t, hugeint_t>(type);

	case PhysicalType::FLOAT:
		return GetTypedModeFunction<float, float>(type);
	case PhysicalType::DOUBLE:
		return GetTypedModeFunction<double, double>(type);

	case PhysicalType::INTERVAL:
		return GetTypedModeFunction<interval_t, interval_t>(type);

	case PhysicalType::VARCHAR:
		return GetTypedModeFunction<string_t, string, ModeAssignmentString>(type);

	default:
		throw NotImplementedException("Unimplemented mode aggregate");
	}
}

unique_ptr<FunctionData> BindModeDecimal(ClientContext &context, AggregateFunction &function,
                                         vector<unique_ptr<Expression>> &arguments) {
	function = GetModeAggregate(arguments[0]->return_type);
	function.name = "mode";
	return nullptr;
}

AggregateFunctionSet ModeFun::GetFunctions() {
	const vector<LogicalType> TEMPORAL = {LogicalType::DATE,         LogicalType::TIMESTAMP, LogicalType::TIME,
	                                      LogicalType::TIMESTAMP_TZ, LogicalType::TIME_TZ,   LogicalType::INTERVAL};

	AggregateFunctionSet mode;
	mode.AddFunction(AggregateFunction({LogicalTypeId::DECIMAL}, LogicalTypeId::DECIMAL, nullptr, nullptr, nullptr,
	                                   nullptr, nullptr, nullptr, BindModeDecimal));

	for (const auto &type : LogicalType::Numeric()) {
		if (type.id() != LogicalTypeId::DECIMAL) {
			mode.AddFunction(GetModeAggregate(type));
		}
	}

	for (const auto &type : TEMPORAL) {
		mode.AddFunction(GetModeAggregate(type));
	}

	mode.AddFunction(GetModeAggregate(LogicalType::VARCHAR));
	return mode;
}
} // namespace duckdb











#include <algorithm>
#include <stdlib.h>
#include <utility>

namespace duckdb {

// Hugeint arithmetic
static hugeint_t operator*(const hugeint_t &h, const double &d) {
	D_ASSERT(d >= 0 && d <= 1);
	return Hugeint::Convert(Hugeint::Cast<double>(h) * d);
}

// Interval arithmetic
static interval_t operator*(const interval_t &i, const double &d) { // NOLINT
	D_ASSERT(d >= 0 && d <= 1);
	return Interval::FromMicro(std::llround(Interval::GetMicro(i) * d));
}

inline interval_t operator+(const interval_t &lhs, const interval_t &rhs) {
	return Interval::FromMicro(Interval::GetMicro(lhs) + Interval::GetMicro(rhs));
}

inline interval_t operator-(const interval_t &lhs, const interval_t &rhs) {
	return Interval::FromMicro(Interval::GetMicro(lhs) - Interval::GetMicro(rhs));
}

using FrameBounds = std::pair<idx_t, idx_t>;

template <typename SAVE_TYPE>
struct QuantileState {
	using SaveType = SAVE_TYPE;

	// Regular aggregation
	vector<SaveType> v;

	// Windowed Quantile indirection
	vector<idx_t> w;
	idx_t pos;

	// Windowed MAD indirection
	vector<idx_t> m;

	QuantileState() : pos(0) {
	}

	~QuantileState() {
	}

	inline void SetPos(size_t pos_p) {
		pos = pos_p;
		if (pos >= w.size()) {
			w.resize(pos);
		}
	}
};

struct QuantileIncluded {
	inline explicit QuantileIncluded(const ValidityMask &fmask_p, const ValidityMask &dmask_p, idx_t bias_p)
	    : fmask(fmask_p), dmask(dmask_p), bias(bias_p) {
	}

	inline bool operator()(const idx_t &idx) const {
		return fmask.RowIsValid(idx) && dmask.RowIsValid(idx - bias);
	}

	inline bool AllValid() const {
		return fmask.AllValid() && dmask.AllValid();
	}

	const ValidityMask &fmask;
	const ValidityMask &dmask;
	const idx_t bias;
};

void ReuseIndexes(idx_t *index, const FrameBounds &frame, const FrameBounds &prev) {
	idx_t j = 0;

	//  Copy overlapping indices
	for (idx_t p = 0; p < (prev.second - prev.first); ++p) {
		auto idx = index[p];

		//  Shift down into any hole
		if (j != p) {
			index[j] = idx;
		}

		//  Skip overlapping values
		if (frame.first <= idx && idx < frame.second) {
			++j;
		}
	}

	//  Insert new indices
	if (j > 0) {
		// Overlap: append the new ends
		for (auto f = frame.first; f < prev.first; ++f, ++j) {
			index[j] = f;
		}
		for (auto f = prev.second; f < frame.second; ++f, ++j) {
			index[j] = f;
		}
	} else {
		//  No overlap: overwrite with new values
		for (auto f = frame.first; f < frame.second; ++f, ++j) {
			index[j] = f;
		}
	}
}

static idx_t ReplaceIndex(idx_t *index, const FrameBounds &frame, const FrameBounds &prev) { // NOLINT
	D_ASSERT(index);

	idx_t j = 0;
	for (idx_t p = 0; p < (prev.second - prev.first); ++p) {
		auto idx = index[p];
		if (j != p) {
			break;
		}

		if (frame.first <= idx && idx < frame.second) {
			++j;
		}
	}
	index[j] = frame.second - 1;

	return j;
}

template <class INPUT_TYPE>
static inline int CanReplace(const idx_t *index, const INPUT_TYPE *fdata, const idx_t j, const idx_t k0, const idx_t k1,
                             const QuantileIncluded &validity) {
	D_ASSERT(index);

	// NULLs sort to the end, so if we have inserted a NULL,
	// it must be past the end of the quantile to be replaceable.
	// Note that the quantile values are never NULL.
	const auto ij = index[j];
	if (!validity(ij)) {
		return k1 < j ? 1 : 0;
	}

	auto curr = fdata[ij];
	if (k1 < j) {
		auto hi = fdata[index[k0]];
		return hi < curr ? 1 : 0;
	} else if (j < k0) {
		auto lo = fdata[index[k1]];
		return curr < lo ? -1 : 0;
	}

	return 0;
}

template <class INPUT_TYPE>
struct IndirectLess {
	inline explicit IndirectLess(const INPUT_TYPE *inputs_p) : inputs(inputs_p) {
	}

	inline bool operator()(const idx_t &lhi, const idx_t &rhi) const {
		return inputs[lhi] < inputs[rhi];
	}

	const INPUT_TYPE *inputs;
};

struct CastInterpolation {

	template <class INPUT_TYPE, class TARGET_TYPE>
	static inline TARGET_TYPE Cast(const INPUT_TYPE &src, Vector &result) {
		return Cast::Operation<INPUT_TYPE, TARGET_TYPE>(src);
	}
	template <typename TARGET_TYPE>
	static inline TARGET_TYPE Interpolate(const TARGET_TYPE &lo, const double d, const TARGET_TYPE &hi) {
		const auto delta = hi - lo;
		return lo + delta * d;
	}
};

template <>
interval_t CastInterpolation::Cast(const dtime_t &src, Vector &result) {
	return {0, 0, src.micros};
}

template <>
double CastInterpolation::Interpolate(const double &lo, const double d, const double &hi) {
	return lo * (1.0 - d) + hi * d;
}

template <>
dtime_t CastInterpolation::Interpolate(const dtime_t &lo, const double d, const dtime_t &hi) {
	return dtime_t(std::llround(lo.micros * (1.0 - d) + hi.micros * d));
}

template <>
timestamp_t CastInterpolation::Interpolate(const timestamp_t &lo, const double d, const timestamp_t &hi) {
	return timestamp_t(std::llround(lo.value * (1.0 - d) + hi.value * d));
}

template <>
string_t CastInterpolation::Cast(const std::string &src, Vector &result) {
	return StringVector::AddString(result, src);
}

template <>
string_t CastInterpolation::Cast(const string_t &src, Vector &result) {
	return StringVector::AddString(result, src);
}

// Direct access
template <typename T>
struct QuantileDirect {
	using INPUT_TYPE = T;
	using RESULT_TYPE = T;

	inline const INPUT_TYPE &operator()(const INPUT_TYPE &x) const {
		return x;
	}
};

// Indirect access
template <typename T>
struct QuantileIndirect {
	using INPUT_TYPE = idx_t;
	using RESULT_TYPE = T;
	const RESULT_TYPE *data;

	explicit QuantileIndirect(const RESULT_TYPE *data_p) : data(data_p) {
	}

	inline RESULT_TYPE operator()(const idx_t &input) const {
		return data[input];
	}
};

// Composed access
template <typename OUTER, typename INNER>
struct QuantileComposed {
	using INPUT_TYPE = typename INNER::INPUT_TYPE;
	using RESULT_TYPE = typename OUTER::RESULT_TYPE;

	const OUTER &outer;
	const INNER &inner;

	explicit QuantileComposed(const OUTER &outer_p, const INNER &inner_p) : outer(outer_p), inner(inner_p) {
	}

	inline RESULT_TYPE operator()(const idx_t &input) const {
		return outer(inner(input));
	}
};

// Accessed comparison
template <typename ACCESSOR>
struct QuantileCompare {
	using INPUT_TYPE = typename ACCESSOR::INPUT_TYPE;
	const ACCESSOR &accessor;
	const bool desc;
	explicit QuantileCompare(const ACCESSOR &accessor_p, bool desc_p) : accessor(accessor_p), desc(desc_p) {
	}

	inline bool operator()(const INPUT_TYPE &lhs, const INPUT_TYPE &rhs) const {
		const auto lval = accessor(lhs);
		const auto rval = accessor(rhs);

		return desc ? (rval < lval) : (lval < rval);
	}
};

// Continuous interpolation
template <bool DISCRETE>
struct Interpolator {
	Interpolator(const Value &q, const idx_t n_p, const bool desc_p)
	    : desc(desc_p), RN((double)(n_p - 1) * q.GetValue<double>()), FRN(floor(RN)), CRN(ceil(RN)), begin(0),
	      end(n_p) {
	}

	template <class INPUT_TYPE, class TARGET_TYPE, typename ACCESSOR = QuantileDirect<INPUT_TYPE>>
	TARGET_TYPE Operation(INPUT_TYPE *v_t, Vector &result, const ACCESSOR &accessor = ACCESSOR()) const {
		using ACCESS_TYPE = typename ACCESSOR::RESULT_TYPE;
		QuantileCompare<ACCESSOR> comp(accessor, desc);
		if (CRN == FRN) {
			std::nth_element(v_t + begin, v_t + FRN, v_t + end, comp);
			return CastInterpolation::Cast<ACCESS_TYPE, TARGET_TYPE>(accessor(v_t[FRN]), result);
		} else {
			std::nth_element(v_t + begin, v_t + FRN, v_t + end, comp);
			std::nth_element(v_t + FRN, v_t + CRN, v_t + end, comp);
			auto lo = CastInterpolation::Cast<ACCESS_TYPE, TARGET_TYPE>(accessor(v_t[FRN]), result);
			auto hi = CastInterpolation::Cast<ACCESS_TYPE, TARGET_TYPE>(accessor(v_t[CRN]), result);
			return CastInterpolation::Interpolate<TARGET_TYPE>(lo, RN - FRN, hi);
		}
	}

	template <class INPUT_TYPE, class TARGET_TYPE, typename ACCESSOR = QuantileDirect<INPUT_TYPE>>
	TARGET_TYPE Replace(const INPUT_TYPE *v_t, Vector &result, const ACCESSOR &accessor = ACCESSOR()) const {
		using ACCESS_TYPE = typename ACCESSOR::RESULT_TYPE;
		if (CRN == FRN) {
			return CastInterpolation::Cast<ACCESS_TYPE, TARGET_TYPE>(accessor(v_t[FRN]), result);
		} else {
			auto lo = CastInterpolation::Cast<ACCESS_TYPE, TARGET_TYPE>(accessor(v_t[FRN]), result);
			auto hi = CastInterpolation::Cast<ACCESS_TYPE, TARGET_TYPE>(accessor(v_t[CRN]), result);
			return CastInterpolation::Interpolate<TARGET_TYPE>(lo, RN - FRN, hi);
		}
	}

	const bool desc;
	const double RN;
	const idx_t FRN;
	const idx_t CRN;

	idx_t begin;
	idx_t end;
};

// Discrete "interpolation"
template <>
struct Interpolator<true> {
	static inline idx_t Index(const Value &q, const idx_t n) {
		idx_t floored;
		const auto &type = q.type();
		switch (type.id()) {
		case LogicalTypeId::DECIMAL: {
			//	Integer arithmetic for accuracy
			const auto integral = IntegralValue::Get(q);
			const auto scaling = Hugeint::POWERS_OF_TEN[DecimalType::GetScale(type)];
			const auto scaled_q = DecimalMultiplyOverflowCheck::Operation<hugeint_t, hugeint_t, hugeint_t>(n, integral);
			const auto scaled_n = DecimalMultiplyOverflowCheck::Operation<hugeint_t, hugeint_t, hugeint_t>(n, scaling);
			floored = Cast::Operation<hugeint_t, idx_t>((scaled_n - scaled_q) / scaling);
			break;
		}
		default:
			const auto scaled_q = (double)(n * q.GetValue<double>());
			floored = floor(n - scaled_q);
			break;
		}

		return MaxValue<idx_t>(1, n - floored) - 1;
	}

	Interpolator(const Value &q, const idx_t n_p, bool desc_p)
	    : desc(desc_p), FRN(Index(q, n_p)), CRN(FRN), begin(0), end(n_p) {
	}

	template <class INPUT_TYPE, class TARGET_TYPE, typename ACCESSOR = QuantileDirect<INPUT_TYPE>>
	TARGET_TYPE Operation(INPUT_TYPE *v_t, Vector &result, const ACCESSOR &accessor = ACCESSOR()) const {
		using ACCESS_TYPE = typename ACCESSOR::RESULT_TYPE;
		QuantileCompare<ACCESSOR> comp(accessor, desc);
		std::nth_element(v_t + begin, v_t + FRN, v_t + end, comp);
		return CastInterpolation::Cast<ACCESS_TYPE, TARGET_TYPE>(accessor(v_t[FRN]), result);
	}

	template <class INPUT_TYPE, class TARGET_TYPE, typename ACCESSOR = QuantileDirect<INPUT_TYPE>>
	TARGET_TYPE Replace(const INPUT_TYPE *v_t, Vector &result, const ACCESSOR &accessor = ACCESSOR()) const {
		using ACCESS_TYPE = typename ACCESSOR::RESULT_TYPE;
		return CastInterpolation::Cast<ACCESS_TYPE, TARGET_TYPE>(accessor(v_t[FRN]), result);
	}

	const bool desc;
	const idx_t FRN;
	const idx_t CRN;

	idx_t begin;
	idx_t end;
};

template <typename T>
static inline T QuantileAbs(const T &t) {
	return AbsOperator::Operation<T, T>(t);
}

template <>
inline Value QuantileAbs(const Value &v) {
	const auto &type = v.type();
	switch (type.id()) {
	case LogicalTypeId::DECIMAL: {
		const auto integral = IntegralValue::Get(v);
		const auto width = DecimalType::GetWidth(type);
		const auto scale = DecimalType::GetScale(type);
		switch (type.InternalType()) {
		case PhysicalType::INT16:
			return Value::DECIMAL(QuantileAbs<int16_t>(Cast::Operation<hugeint_t, int16_t>(integral)), width, scale);
		case PhysicalType::INT32:
			return Value::DECIMAL(QuantileAbs<int32_t>(Cast::Operation<hugeint_t, int32_t>(integral)), width, scale);
		case PhysicalType::INT64:
			return Value::DECIMAL(QuantileAbs<int64_t>(Cast::Operation<hugeint_t, int64_t>(integral)), width, scale);
		case PhysicalType::INT128:
			return Value::DECIMAL(QuantileAbs<hugeint_t>(integral), width, scale);
		default:
			throw InternalException("Unknown DECIMAL type");
		}
	}
	default:
		return Value::DOUBLE(QuantileAbs<double>(v.GetValue<double>()));
	}
}

struct QuantileBindData : public FunctionData {

	explicit QuantileBindData(const Value &quantile_p)
	    : quantiles(1, QuantileAbs(quantile_p)), order(1, 0), desc(quantile_p < 0) {
	}

	explicit QuantileBindData(const vector<Value> &quantiles_p) {
		size_t pos = 0;
		size_t neg = 0;
		for (idx_t i = 0; i < quantiles_p.size(); ++i) {
			const auto &q = quantiles_p[i];
			pos += (q > 0);
			neg += (q < 0);
			quantiles.emplace_back(QuantileAbs(q));
			order.push_back(i);
		}
		if (pos && neg) {
			throw BinderException("QUANTILE parameters must have consistent signs");
		}
		desc = (neg > 0);

		IndirectLess<Value> lt(quantiles.data());
		std::sort(order.begin(), order.end(), lt);
	}

	QuantileBindData(const QuantileBindData &other) : order(other.order), desc(other.desc) {
		for (const auto &q : other.quantiles) {
			quantiles.emplace_back(q);
		}
	}

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<QuantileBindData>(*this);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<QuantileBindData>();
		return desc == other.desc && quantiles == other.quantiles && order == other.order;
	}

	vector<Value> quantiles;
	vector<idx_t> order;
	bool desc;
};

struct QuantileOperation {
	template <class STATE>
	static void Initialize(STATE *state) {
		new (state) STATE;
	}

	template <class INPUT_TYPE, class STATE, class OP>
	static void ConstantOperation(STATE *state, AggregateInputData &aggr_input_data, INPUT_TYPE *input,
	                              ValidityMask &mask, idx_t count) {
		for (idx_t i = 0; i < count; i++) {
			Operation<INPUT_TYPE, STATE, OP>(state, aggr_input_data, input, mask, 0);
		}
	}

	template <class INPUT_TYPE, class STATE, class OP>
	static void Operation(STATE *state, AggregateInputData &, INPUT_TYPE *data, ValidityMask &mask, idx_t idx) {
		state->v.emplace_back(data[idx]);
	}

	template <class STATE, class OP>
	static void Combine(const STATE &source, STATE *target, AggregateInputData &) {
		if (source.v.empty()) {
			return;
		}
		target->v.insert(target->v.end(), source.v.begin(), source.v.end());
	}

	template <class STATE>
	static void Destroy(AggregateInputData &aggr_input_data, STATE *state) {
		state->~STATE();
	}

	static bool IgnoreNull() {
		return true;
	}
};

template <class STATE_TYPE, class RESULT_TYPE, class OP>
static void ExecuteListFinalize(Vector &states, AggregateInputData &aggr_input_data, Vector &result,
                                idx_t count, // NOLINT
                                idx_t offset) {
	D_ASSERT(result.GetType().id() == LogicalTypeId::LIST);

	D_ASSERT(aggr_input_data.bind_data);
	auto bind_data = (QuantileBindData *)aggr_input_data.bind_data;

	if (states.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
		ListVector::Reserve(result, bind_data->quantiles.size());

		auto sdata = ConstantVector::GetData<STATE_TYPE *>(states);
		auto rdata = ConstantVector::GetData<RESULT_TYPE>(result);
		auto &mask = ConstantVector::Validity(result);
		OP::template Finalize<RESULT_TYPE, STATE_TYPE>(result, aggr_input_data, sdata[0], rdata, mask, 0);
	} else {
		D_ASSERT(states.GetVectorType() == VectorType::FLAT_VECTOR);
		result.SetVectorType(VectorType::FLAT_VECTOR);
		ListVector::Reserve(result, (offset + count) * bind_data->quantiles.size());

		auto sdata = FlatVector::GetData<STATE_TYPE *>(states);
		auto rdata = FlatVector::GetData<RESULT_TYPE>(result);
		auto &mask = FlatVector::Validity(result);
		for (idx_t i = 0; i < count; i++) {
			OP::template Finalize<RESULT_TYPE, STATE_TYPE>(result, aggr_input_data, sdata[i], rdata, mask, i + offset);
		}
	}

	result.Verify(count);
}

template <class STATE, class INPUT_TYPE, class RESULT_TYPE, class OP>
static AggregateFunction QuantileListAggregate(const LogicalType &input_type, const LogicalType &child_type) { // NOLINT
	LogicalType result_type = LogicalType::LIST(child_type);
	return AggregateFunction(
	    {input_type}, result_type, AggregateFunction::StateSize<STATE>, AggregateFunction::StateInitialize<STATE, OP>,
	    AggregateFunction::UnaryScatterUpdate<STATE, INPUT_TYPE, OP>, AggregateFunction::StateCombine<STATE, OP>,
	    ExecuteListFinalize<STATE, RESULT_TYPE, OP>, AggregateFunction::UnaryUpdate<STATE, INPUT_TYPE, OP>, nullptr,
	    AggregateFunction::StateDestroy<STATE, OP>);
}

template <bool DISCRETE>
struct QuantileScalarOperation : public QuantileOperation {

	template <class RESULT_TYPE, class STATE>
	static void Finalize(Vector &result, AggregateInputData &aggr_input_data, STATE *state, RESULT_TYPE *target,
	                     ValidityMask &mask, idx_t idx) {
		if (state->v.empty()) {
			mask.SetInvalid(idx);
			return;
		}
		D_ASSERT(aggr_input_data.bind_data);
		auto bind_data = (QuantileBindData *)aggr_input_data.bind_data;
		D_ASSERT(bind_data->quantiles.size() == 1);
		Interpolator<DISCRETE> interp(bind_data->quantiles[0], state->v.size(), bind_data->desc);
		target[idx] = interp.template Operation<typename STATE::SaveType, RESULT_TYPE>(state->v.data(), result);
	}

	template <class STATE, class INPUT_TYPE, class RESULT_TYPE>
	static void Window(const INPUT_TYPE *data, const ValidityMask &fmask, const ValidityMask &dmask,
	                   AggregateInputData &aggr_input_data, STATE *state, const FrameBounds &frame,
	                   const FrameBounds &prev, Vector &result, idx_t ridx, idx_t bias) {
		auto rdata = FlatVector::GetData<RESULT_TYPE>(result);
		auto &rmask = FlatVector::Validity(result);

		QuantileIncluded included(fmask, dmask, bias);

		//  Lazily initialise frame state
		auto prev_pos = state->pos;
		state->SetPos(frame.second - frame.first);

		auto index = state->w.data();
		D_ASSERT(index);

		D_ASSERT(aggr_input_data.bind_data);
		auto bind_data = (QuantileBindData *)aggr_input_data.bind_data;

		// Find the two positions needed
		const auto q = bind_data->quantiles[0];

		bool replace = false;
		if (frame.first == prev.first + 1 && frame.second == prev.second + 1) {
			//  Fixed frame size
			const auto j = ReplaceIndex(index, frame, prev);
			//	We can only replace if the number of NULLs has not changed
			if (included.AllValid() || included(prev.first) == included(prev.second)) {
				Interpolator<DISCRETE> interp(q, prev_pos, false);
				replace = CanReplace(index, data, j, interp.FRN, interp.CRN, included);
				if (replace) {
					state->pos = prev_pos;
				}
			}
		} else {
			ReuseIndexes(index, frame, prev);
		}

		if (!replace && !included.AllValid()) {
			// Remove the NULLs
			state->pos = std::partition(index, index + state->pos, included) - index;
		}
		if (state->pos) {
			Interpolator<DISCRETE> interp(q, state->pos, false);

			using ID = QuantileIndirect<INPUT_TYPE>;
			ID indirect(data);
			rdata[ridx] = replace ? interp.template Replace<idx_t, RESULT_TYPE, ID>(index, result, indirect)
			                      : interp.template Operation<idx_t, RESULT_TYPE, ID>(index, result, indirect);
		} else {
			rmask.Set(ridx, false);
		}
	}
};

template <typename INPUT_TYPE, typename SAVED_TYPE>
AggregateFunction GetTypedDiscreteQuantileAggregateFunction(const LogicalType &type) {
	using STATE = QuantileState<SAVED_TYPE>;
	using OP = QuantileScalarOperation<true>;
	auto fun = AggregateFunction::UnaryAggregateDestructor<STATE, INPUT_TYPE, INPUT_TYPE, OP>(type, type);
	fun.window = AggregateFunction::UnaryWindow<STATE, INPUT_TYPE, INPUT_TYPE, OP>;
	return fun;
}

AggregateFunction GetDiscreteQuantileAggregateFunction(const LogicalType &type) {
	switch (type.id()) {
	case LogicalTypeId::TINYINT:
		return GetTypedDiscreteQuantileAggregateFunction<int8_t, int8_t>(type);
	case LogicalTypeId::SMALLINT:
		return GetTypedDiscreteQuantileAggregateFunction<int16_t, int16_t>(type);
	case LogicalTypeId::INTEGER:
		return GetTypedDiscreteQuantileAggregateFunction<int32_t, int32_t>(type);
	case LogicalTypeId::BIGINT:
		return GetTypedDiscreteQuantileAggregateFunction<int64_t, int64_t>(type);
	case LogicalTypeId::HUGEINT:
		return GetTypedDiscreteQuantileAggregateFunction<hugeint_t, hugeint_t>(type);
	case LogicalTypeId::FLOAT:
		return GetTypedDiscreteQuantileAggregateFunction<float, float>(type);
	case LogicalTypeId::DOUBLE:
		return GetTypedDiscreteQuantileAggregateFunction<double, double>(type);
	case LogicalTypeId::DECIMAL:
		switch (type.InternalType()) {
		case PhysicalType::INT16:
			return GetTypedDiscreteQuantileAggregateFunction<int16_t, int16_t>(type);
		case PhysicalType::INT32:
			return GetTypedDiscreteQuantileAggregateFunction<int32_t, int32_t>(type);
		case PhysicalType::INT64:
			return GetTypedDiscreteQuantileAggregateFunction<int64_t, int64_t>(type);
		case PhysicalType::INT128:
			return GetTypedDiscreteQuantileAggregateFunction<hugeint_t, hugeint_t>(type);
		default:
			throw NotImplementedException("Unimplemented discrete quantile aggregate");
		}
	case LogicalTypeId::DATE:
		return GetTypedDiscreteQuantileAggregateFunction<int32_t, int32_t>(type);
	case LogicalTypeId::TIMESTAMP:
	case LogicalTypeId::TIMESTAMP_TZ:
		return GetTypedDiscreteQuantileAggregateFunction<int64_t, int64_t>(type);
	case LogicalTypeId::TIME:
	case LogicalTypeId::TIME_TZ:
		return GetTypedDiscreteQuantileAggregateFunction<int64_t, int64_t>(type);
	case LogicalTypeId::INTERVAL:
		return GetTypedDiscreteQuantileAggregateFunction<interval_t, interval_t>(type);

	case LogicalTypeId::VARCHAR:
		return GetTypedDiscreteQuantileAggregateFunction<string_t, std::string>(type);

	default:
		throw NotImplementedException("Unimplemented discrete quantile aggregate");
	}
}

template <class CHILD_TYPE, bool DISCRETE>
struct QuantileListOperation : public QuantileOperation {

	template <class RESULT_TYPE, class STATE>
	static void Finalize(Vector &result_list, AggregateInputData &aggr_input_data, STATE *state, RESULT_TYPE *target,
	                     ValidityMask &mask, idx_t idx) {
		if (state->v.empty()) {
			mask.SetInvalid(idx);
			return;
		}

		D_ASSERT(aggr_input_data.bind_data);
		auto bind_data = (QuantileBindData *)aggr_input_data.bind_data;

		auto &result = ListVector::GetEntry(result_list);
		auto ridx = ListVector::GetListSize(result_list);
		ListVector::Reserve(result_list, ridx + bind_data->quantiles.size());
		auto rdata = FlatVector::GetData<CHILD_TYPE>(result);

		auto v_t = state->v.data();
		D_ASSERT(v_t);

		auto &entry = target[idx];
		entry.offset = ridx;
		idx_t lower = 0;
		for (const auto &q : bind_data->order) {
			const auto &quantile = bind_data->quantiles[q];
			Interpolator<DISCRETE> interp(quantile, state->v.size(), bind_data->desc);
			interp.begin = lower;
			rdata[ridx + q] = interp.template Operation<typename STATE::SaveType, CHILD_TYPE>(v_t, result);
			lower = interp.FRN;
		}
		entry.length = bind_data->quantiles.size();

		ListVector::SetListSize(result_list, entry.offset + entry.length);
	}

	template <class STATE, class INPUT_TYPE, class RESULT_TYPE>
	static void Window(const INPUT_TYPE *data, const ValidityMask &fmask, const ValidityMask &dmask,
	                   AggregateInputData &aggr_input_data, STATE *state, const FrameBounds &frame,
	                   const FrameBounds &prev, Vector &list, idx_t lidx, idx_t bias) {
		D_ASSERT(aggr_input_data.bind_data);
		auto bind_data = (QuantileBindData *)aggr_input_data.bind_data;

		QuantileIncluded included(fmask, dmask, bias);

		// Result is a constant LIST<RESULT_TYPE> with a fixed length
		auto ldata = FlatVector::GetData<RESULT_TYPE>(list);
		auto &lmask = FlatVector::Validity(list);
		auto &lentry = ldata[lidx];
		lentry.offset = ListVector::GetListSize(list);
		lentry.length = bind_data->quantiles.size();

		ListVector::Reserve(list, lentry.offset + lentry.length);
		ListVector::SetListSize(list, lentry.offset + lentry.length);
		auto &result = ListVector::GetEntry(list);
		auto rdata = FlatVector::GetData<CHILD_TYPE>(result);

		//  Lazily initialise frame state
		auto prev_pos = state->pos;
		state->SetPos(frame.second - frame.first);

		auto index = state->w.data();

		// We can generalise replacement for quantile lists by observing that when a replacement is
		// valid for a single quantile, it is valid for all quantiles greater/less than that quantile
		// based on whether the insertion is below/above the quantile location.
		// So if a replaced index in an IQR is located between Q25 and Q50, but has a value below Q25,
		// then Q25 must be recomputed, but Q50 and Q75 are unaffected.
		// For a single element list, this reduces to the scalar case.
		std::pair<idx_t, idx_t> replaceable {state->pos, 0};
		if (frame.first == prev.first + 1 && frame.second == prev.second + 1) {
			//  Fixed frame size
			const auto j = ReplaceIndex(index, frame, prev);
			//	We can only replace if the number of NULLs has not changed
			if (included.AllValid() || included(prev.first) == included(prev.second)) {
				for (const auto &q : bind_data->order) {
					const auto &quantile = bind_data->quantiles[q];
					Interpolator<DISCRETE> interp(quantile, prev_pos, false);
					const auto replace = CanReplace(index, data, j, interp.FRN, interp.CRN, included);
					if (replace < 0) {
						//	Replacement is before this quantile, so the rest will be replaceable too.
						replaceable.first = MinValue(replaceable.first, interp.FRN);
						replaceable.second = prev_pos;
						break;
					} else if (replace > 0) {
						//	Replacement is after this quantile, so everything before it is replaceable too.
						replaceable.first = 0;
						replaceable.second = MaxValue(replaceable.second, interp.CRN);
					}
				}
				if (replaceable.first < replaceable.second) {
					state->pos = prev_pos;
				}
			}
		} else {
			ReuseIndexes(index, frame, prev);
		}

		if (replaceable.first >= replaceable.second && !included.AllValid()) {
			// Remove the NULLs
			state->pos = std::partition(index, index + state->pos, included) - index;
		}

		if (state->pos) {
			using ID = QuantileIndirect<INPUT_TYPE>;
			ID indirect(data);
			for (const auto &q : bind_data->order) {
				const auto &quantile = bind_data->quantiles[q];
				Interpolator<DISCRETE> interp(quantile, state->pos, false);
				if (replaceable.first <= interp.FRN && interp.CRN <= replaceable.second) {
					rdata[lentry.offset + q] = interp.template Replace<idx_t, CHILD_TYPE, ID>(index, result, indirect);
				} else {
					// Make sure we don't disturb any replacements
					if (replaceable.first < replaceable.second) {
						if (interp.FRN < replaceable.first) {
							interp.end = replaceable.first;
						}
						if (replaceable.second < interp.CRN) {
							interp.begin = replaceable.second;
						}
					}
					rdata[lentry.offset + q] =
					    interp.template Operation<idx_t, CHILD_TYPE, ID>(index, result, indirect);
				}
			}
		} else {
			lmask.Set(lidx, false);
		}
	}
};

template <typename INPUT_TYPE, typename SAVE_TYPE>
AggregateFunction GetTypedDiscreteQuantileListAggregateFunction(const LogicalType &type) {
	using STATE = QuantileState<SAVE_TYPE>;
	using OP = QuantileListOperation<INPUT_TYPE, true>;
	auto fun = QuantileListAggregate<STATE, INPUT_TYPE, list_entry_t, OP>(type, type);
	fun.order_dependent = AggregateOrderDependent::NOT_ORDER_DEPENDENT;
	fun.window = AggregateFunction::UnaryWindow<STATE, INPUT_TYPE, list_entry_t, OP>;
	return fun;
}

AggregateFunction GetDiscreteQuantileListAggregateFunction(const LogicalType &type) {
	switch (type.id()) {
	case LogicalTypeId::TINYINT:
		return GetTypedDiscreteQuantileListAggregateFunction<int8_t, int8_t>(type);
	case LogicalTypeId::SMALLINT:
		return GetTypedDiscreteQuantileListAggregateFunction<int16_t, int16_t>(type);
	case LogicalTypeId::INTEGER:
		return GetTypedDiscreteQuantileListAggregateFunction<int32_t, int32_t>(type);
	case LogicalTypeId::BIGINT:
		return GetTypedDiscreteQuantileListAggregateFunction<int64_t, int64_t>(type);
	case LogicalTypeId::HUGEINT:
		return GetTypedDiscreteQuantileListAggregateFunction<hugeint_t, hugeint_t>(type);
	case LogicalTypeId::FLOAT:
		return GetTypedDiscreteQuantileListAggregateFunction<float, float>(type);
	case LogicalTypeId::DOUBLE:
		return GetTypedDiscreteQuantileListAggregateFunction<double, double>(type);
	case LogicalTypeId::DECIMAL:
		switch (type.InternalType()) {
		case PhysicalType::INT16:
			return GetTypedDiscreteQuantileListAggregateFunction<int16_t, int16_t>(type);
		case PhysicalType::INT32:
			return GetTypedDiscreteQuantileListAggregateFunction<int32_t, int32_t>(type);
		case PhysicalType::INT64:
			return GetTypedDiscreteQuantileListAggregateFunction<int64_t, int64_t>(type);
		case PhysicalType::INT128:
			return GetTypedDiscreteQuantileListAggregateFunction<hugeint_t, hugeint_t>(type);
		default:
			throw NotImplementedException("Unimplemented discrete quantile list aggregate");
		}
	case LogicalTypeId::DATE:
		return GetTypedDiscreteQuantileListAggregateFunction<date_t, date_t>(type);
	case LogicalTypeId::TIMESTAMP:
	case LogicalTypeId::TIMESTAMP_TZ:
		return GetTypedDiscreteQuantileListAggregateFunction<timestamp_t, timestamp_t>(type);
	case LogicalTypeId::TIME:
	case LogicalTypeId::TIME_TZ:
		return GetTypedDiscreteQuantileListAggregateFunction<dtime_t, dtime_t>(type);
	case LogicalTypeId::INTERVAL:
		return GetTypedDiscreteQuantileListAggregateFunction<interval_t, interval_t>(type);
	case LogicalTypeId::VARCHAR:
		return GetTypedDiscreteQuantileListAggregateFunction<string_t, std::string>(type);
	default:
		throw NotImplementedException("Unimplemented discrete quantile list aggregate");
	}
}

template <typename INPUT_TYPE, typename TARGET_TYPE>
AggregateFunction GetTypedContinuousQuantileAggregateFunction(const LogicalType &input_type,
                                                              const LogicalType &target_type) {
	using STATE = QuantileState<INPUT_TYPE>;
	using OP = QuantileScalarOperation<false>;
	auto fun = AggregateFunction::UnaryAggregateDestructor<STATE, INPUT_TYPE, TARGET_TYPE, OP>(input_type, target_type);
	fun.order_dependent = AggregateOrderDependent::NOT_ORDER_DEPENDENT;
	fun.window = AggregateFunction::UnaryWindow<STATE, INPUT_TYPE, TARGET_TYPE, OP>;
	return fun;
}

AggregateFunction GetContinuousQuantileAggregateFunction(const LogicalType &type) {
	switch (type.id()) {
	case LogicalTypeId::TINYINT:
		return GetTypedContinuousQuantileAggregateFunction<int8_t, double>(type, LogicalType::DOUBLE);
	case LogicalTypeId::SMALLINT:
		return GetTypedContinuousQuantileAggregateFunction<int16_t, double>(type, LogicalType::DOUBLE);
	case LogicalTypeId::INTEGER:
		return GetTypedContinuousQuantileAggregateFunction<int32_t, double>(type, LogicalType::DOUBLE);
	case LogicalTypeId::BIGINT:
		return GetTypedContinuousQuantileAggregateFunction<int64_t, double>(type, LogicalType::DOUBLE);
	case LogicalTypeId::HUGEINT:
		return GetTypedContinuousQuantileAggregateFunction<hugeint_t, double>(type, LogicalType::DOUBLE);
	case LogicalTypeId::FLOAT:
		return GetTypedContinuousQuantileAggregateFunction<float, float>(type, type);
	case LogicalTypeId::DOUBLE:
		return GetTypedContinuousQuantileAggregateFunction<double, double>(type, type);
	case LogicalTypeId::DECIMAL:
		switch (type.InternalType()) {
		case PhysicalType::INT16:
			return GetTypedContinuousQuantileAggregateFunction<int16_t, int16_t>(type, type);
		case PhysicalType::INT32:
			return GetTypedContinuousQuantileAggregateFunction<int32_t, int32_t>(type, type);
		case PhysicalType::INT64:
			return GetTypedContinuousQuantileAggregateFunction<int64_t, int64_t>(type, type);
		case PhysicalType::INT128:
			return GetTypedContinuousQuantileAggregateFunction<hugeint_t, hugeint_t>(type, type);
		default:
			throw NotImplementedException("Unimplemented continuous quantile DECIMAL aggregate");
		}
	case LogicalTypeId::DATE:
		return GetTypedContinuousQuantileAggregateFunction<date_t, timestamp_t>(type, LogicalType::TIMESTAMP);
	case LogicalTypeId::TIMESTAMP:
	case LogicalTypeId::TIMESTAMP_TZ:
		return GetTypedContinuousQuantileAggregateFunction<timestamp_t, timestamp_t>(type, type);
	case LogicalTypeId::TIME:
	case LogicalTypeId::TIME_TZ:
		return GetTypedContinuousQuantileAggregateFunction<dtime_t, dtime_t>(type, type);

	default:
		throw NotImplementedException("Unimplemented continuous quantile aggregate");
	}
}

template <typename INPUT_TYPE, typename CHILD_TYPE>
AggregateFunction GetTypedContinuousQuantileListAggregateFunction(const LogicalType &input_type,
                                                                  const LogicalType &result_type) {
	using STATE = QuantileState<INPUT_TYPE>;
	using OP = QuantileListOperation<CHILD_TYPE, false>;
	auto fun = QuantileListAggregate<STATE, INPUT_TYPE, list_entry_t, OP>(input_type, result_type);
	fun.order_dependent = AggregateOrderDependent::NOT_ORDER_DEPENDENT;
	fun.window = AggregateFunction::UnaryWindow<STATE, INPUT_TYPE, list_entry_t, OP>;
	return fun;
}

AggregateFunction GetContinuousQuantileListAggregateFunction(const LogicalType &type) {
	switch (type.id()) {
	case LogicalTypeId::TINYINT:
		return GetTypedContinuousQuantileListAggregateFunction<int8_t, double>(type, LogicalType::DOUBLE);
	case LogicalTypeId::SMALLINT:
		return GetTypedContinuousQuantileListAggregateFunction<int16_t, double>(type, LogicalType::DOUBLE);
	case LogicalTypeId::INTEGER:
		return GetTypedContinuousQuantileListAggregateFunction<int32_t, double>(type, LogicalType::DOUBLE);
	case LogicalTypeId::BIGINT:
		return GetTypedContinuousQuantileListAggregateFunction<int64_t, double>(type, LogicalType::DOUBLE);
	case LogicalTypeId::HUGEINT:
		return GetTypedContinuousQuantileListAggregateFunction<hugeint_t, double>(type, LogicalType::DOUBLE);

	case LogicalTypeId::FLOAT:
		return GetTypedContinuousQuantileListAggregateFunction<float, float>(type, type);
	case LogicalTypeId::DOUBLE:
		return GetTypedContinuousQuantileListAggregateFunction<double, double>(type, type);
	case LogicalTypeId::DECIMAL:
		switch (type.InternalType()) {
		case PhysicalType::INT16:
			return GetTypedContinuousQuantileListAggregateFunction<int16_t, int16_t>(type, type);
		case PhysicalType::INT32:
			return GetTypedContinuousQuantileListAggregateFunction<int32_t, int32_t>(type, type);
		case PhysicalType::INT64:
			return GetTypedContinuousQuantileListAggregateFunction<int64_t, int64_t>(type, type);
		case PhysicalType::INT128:
			return GetTypedContinuousQuantileListAggregateFunction<hugeint_t, hugeint_t>(type, type);
		default:
			throw NotImplementedException("Unimplemented discrete quantile DECIMAL list aggregate");
		}
		break;

	case LogicalTypeId::DATE:
		return GetTypedContinuousQuantileListAggregateFunction<date_t, timestamp_t>(type, LogicalType::TIMESTAMP);
	case LogicalTypeId::TIMESTAMP:
	case LogicalTypeId::TIMESTAMP_TZ:
		return GetTypedContinuousQuantileListAggregateFunction<timestamp_t, timestamp_t>(type, type);
	case LogicalTypeId::TIME:
	case LogicalTypeId::TIME_TZ:
		return GetTypedContinuousQuantileListAggregateFunction<dtime_t, dtime_t>(type, type);

	default:
		throw NotImplementedException("Unimplemented discrete quantile list aggregate");
	}
}

template <typename T, typename R, typename MEDIAN_TYPE>
struct MadAccessor {
	using INPUT_TYPE = T;
	using RESULT_TYPE = R;
	const MEDIAN_TYPE &median;
	explicit MadAccessor(const MEDIAN_TYPE &median_p) : median(median_p) {
	}

	inline RESULT_TYPE operator()(const INPUT_TYPE &input) const {
		const auto delta = input - median;
		return TryAbsOperator::Operation<RESULT_TYPE, RESULT_TYPE>(delta);
	}
};

// hugeint_t - double => undefined
template <>
struct MadAccessor<hugeint_t, double, double> {
	using INPUT_TYPE = hugeint_t;
	using RESULT_TYPE = double;
	using MEDIAN_TYPE = double;
	const MEDIAN_TYPE &median;
	explicit MadAccessor(const MEDIAN_TYPE &median_p) : median(median_p) {
	}
	inline RESULT_TYPE operator()(const INPUT_TYPE &input) const {
		const auto delta = Hugeint::Cast<double>(input) - median;
		return TryAbsOperator::Operation<double, double>(delta);
	}
};

// date_t - timestamp_t => interval_t
template <>
struct MadAccessor<date_t, interval_t, timestamp_t> {
	using INPUT_TYPE = date_t;
	using RESULT_TYPE = interval_t;
	using MEDIAN_TYPE = timestamp_t;
	const MEDIAN_TYPE &median;
	explicit MadAccessor(const MEDIAN_TYPE &median_p) : median(median_p) {
	}
	inline RESULT_TYPE operator()(const INPUT_TYPE &input) const {
		const auto dt = Cast::Operation<date_t, timestamp_t>(input);
		const auto delta = dt - median;
		return Interval::FromMicro(TryAbsOperator::Operation<int64_t, int64_t>(delta));
	}
};

// timestamp_t - timestamp_t => int64_t
template <>
struct MadAccessor<timestamp_t, interval_t, timestamp_t> {
	using INPUT_TYPE = timestamp_t;
	using RESULT_TYPE = interval_t;
	using MEDIAN_TYPE = timestamp_t;
	const MEDIAN_TYPE &median;
	explicit MadAccessor(const MEDIAN_TYPE &median_p) : median(median_p) {
	}
	inline RESULT_TYPE operator()(const INPUT_TYPE &input) const {
		const auto delta = input - median;
		return Interval::FromMicro(TryAbsOperator::Operation<int64_t, int64_t>(delta));
	}
};

// dtime_t - dtime_t => int64_t
template <>
struct MadAccessor<dtime_t, interval_t, dtime_t> {
	using INPUT_TYPE = dtime_t;
	using RESULT_TYPE = interval_t;
	using MEDIAN_TYPE = dtime_t;
	const MEDIAN_TYPE &median;
	explicit MadAccessor(const MEDIAN_TYPE &median_p) : median(median_p) {
	}
	inline RESULT_TYPE operator()(const INPUT_TYPE &input) const {
		const auto delta = input - median;
		return Interval::FromMicro(TryAbsOperator::Operation<int64_t, int64_t>(delta));
	}
};

template <typename MEDIAN_TYPE>
struct MedianAbsoluteDeviationOperation : public QuantileOperation {

	template <class RESULT_TYPE, class STATE>
	static void Finalize(Vector &result, AggregateInputData &, STATE *state, RESULT_TYPE *target, ValidityMask &mask,
	                     idx_t idx) {
		if (state->v.empty()) {
			mask.SetInvalid(idx);
			return;
		}
		using SAVE_TYPE = typename STATE::SaveType;
		Interpolator<false> interp(0.5, state->v.size(), false);
		const auto med = interp.template Operation<SAVE_TYPE, MEDIAN_TYPE>(state->v.data(), result);

		MadAccessor<SAVE_TYPE, RESULT_TYPE, MEDIAN_TYPE> accessor(med);
		target[idx] = interp.template Operation<SAVE_TYPE, RESULT_TYPE>(state->v.data(), result, accessor);
	}

	template <class STATE, class INPUT_TYPE, class RESULT_TYPE>
	static void Window(const INPUT_TYPE *data, const ValidityMask &fmask, const ValidityMask &dmask,
	                   AggregateInputData &, STATE *state, const FrameBounds &frame, const FrameBounds &prev,
	                   Vector &result, idx_t ridx, idx_t bias) {
		auto rdata = FlatVector::GetData<RESULT_TYPE>(result);
		auto &rmask = FlatVector::Validity(result);

		QuantileIncluded included(fmask, dmask, bias);

		//  Lazily initialise frame state
		auto prev_pos = state->pos;
		state->SetPos(frame.second - frame.first);

		auto index = state->w.data();
		D_ASSERT(index);

		// We need a second index for the second pass.
		if (state->pos > state->m.size()) {
			state->m.resize(state->pos);
		}

		auto index2 = state->m.data();
		D_ASSERT(index2);

		// The replacement trick does not work on the second index because if
		// the median has changed, the previous order is not correct.
		// It is probably close, however, and so reuse is helpful.
		ReuseIndexes(index2, frame, prev);
		std::partition(index2, index2 + state->pos, included);

		// Find the two positions needed for the median
		const float q = 0.5;

		bool replace = false;
		if (frame.first == prev.first + 1 && frame.second == prev.second + 1) {
			//  Fixed frame size
			const auto j = ReplaceIndex(index, frame, prev);
			//	We can only replace if the number of NULLs has not changed
			if (included.AllValid() || included(prev.first) == included(prev.second)) {
				Interpolator<false> interp(q, prev_pos, false);
				replace = CanReplace(index, data, j, interp.FRN, interp.CRN, included);
				if (replace) {
					state->pos = prev_pos;
				}
			}
		} else {
			ReuseIndexes(index, frame, prev);
		}

		if (!replace && !included.AllValid()) {
			// Remove the NULLs
			state->pos = std::partition(index, index + state->pos, included) - index;
		}

		if (state->pos) {
			Interpolator<false> interp(q, state->pos, false);

			// Compute or replace median from the first index
			using ID = QuantileIndirect<INPUT_TYPE>;
			ID indirect(data);
			const auto med = replace ? interp.template Replace<idx_t, MEDIAN_TYPE, ID>(index, result, indirect)
			                         : interp.template Operation<idx_t, MEDIAN_TYPE, ID>(index, result, indirect);

			// Compute mad from the second index
			using MAD = MadAccessor<INPUT_TYPE, RESULT_TYPE, MEDIAN_TYPE>;
			MAD mad(med);

			using MadIndirect = QuantileComposed<MAD, ID>;
			MadIndirect mad_indirect(mad, indirect);
			rdata[ridx] = interp.template Operation<idx_t, RESULT_TYPE, MadIndirect>(index2, result, mad_indirect);
		} else {
			rmask.Set(ridx, false);
		}
	}
};

template <typename INPUT_TYPE, typename MEDIAN_TYPE, typename TARGET_TYPE>
AggregateFunction GetTypedMedianAbsoluteDeviationAggregateFunction(const LogicalType &input_type,
                                                                   const LogicalType &target_type) {
	using STATE = QuantileState<INPUT_TYPE>;
	using OP = MedianAbsoluteDeviationOperation<MEDIAN_TYPE>;
	auto fun = AggregateFunction::UnaryAggregateDestructor<STATE, INPUT_TYPE, TARGET_TYPE, OP>(input_type, target_type);
	fun.order_dependent = AggregateOrderDependent::NOT_ORDER_DEPENDENT;
	fun.window = AggregateFunction::UnaryWindow<STATE, INPUT_TYPE, TARGET_TYPE, OP>;
	return fun;
}

AggregateFunction GetMedianAbsoluteDeviationAggregateFunction(const LogicalType &type) {
	switch (type.id()) {
	case LogicalTypeId::FLOAT:
		return GetTypedMedianAbsoluteDeviationAggregateFunction<float, float, float>(type, type);
	case LogicalTypeId::DOUBLE:
		return GetTypedMedianAbsoluteDeviationAggregateFunction<double, double, double>(type, type);
	case LogicalTypeId::DECIMAL:
		switch (type.InternalType()) {
		case PhysicalType::INT16:
			return GetTypedMedianAbsoluteDeviationAggregateFunction<int16_t, int16_t, int16_t>(type, type);
		case PhysicalType::INT32:
			return GetTypedMedianAbsoluteDeviationAggregateFunction<int32_t, int32_t, int32_t>(type, type);
		case PhysicalType::INT64:
			return GetTypedMedianAbsoluteDeviationAggregateFunction<int64_t, int64_t, int64_t>(type, type);
		case PhysicalType::INT128:
			return GetTypedMedianAbsoluteDeviationAggregateFunction<hugeint_t, hugeint_t, hugeint_t>(type, type);
		default:
			throw NotImplementedException("Unimplemented Median Absolute Deviation DECIMAL aggregate");
		}
		break;

	case LogicalTypeId::DATE:
		return GetTypedMedianAbsoluteDeviationAggregateFunction<date_t, timestamp_t, interval_t>(type,
		                                                                                         LogicalType::INTERVAL);
	case LogicalTypeId::TIMESTAMP:
	case LogicalTypeId::TIMESTAMP_TZ:
		return GetTypedMedianAbsoluteDeviationAggregateFunction<timestamp_t, timestamp_t, interval_t>(
		    type, LogicalType::INTERVAL);
	case LogicalTypeId::TIME:
	case LogicalTypeId::TIME_TZ:
		return GetTypedMedianAbsoluteDeviationAggregateFunction<dtime_t, dtime_t, interval_t>(type,
		                                                                                      LogicalType::INTERVAL);

	default:
		throw NotImplementedException("Unimplemented Median Absolute Deviation aggregate");
	}
}

static void QuantileSerialize(FieldWriter &writer, const FunctionData *bind_data_p, const AggregateFunction &function) {
	D_ASSERT(bind_data_p);
	throw NotImplementedException("FIXME: serializing quantiles is not supported right now");
	//
	//	auto bind_data = (QuantileBindData *)bind_data_p;
	//	writer.WriteList<Value>(bind_data->quantiles);
}

unique_ptr<FunctionData> QuantileDeserialize(ClientContext &context, FieldReader &reader,
                                             AggregateFunction &bound_function) {
	auto quantiles = reader.ReadRequiredList<Value>();
	return make_uniq<QuantileBindData>(std::move(quantiles));
}

unique_ptr<FunctionData> BindMedian(ClientContext &context, AggregateFunction &function,
                                    vector<unique_ptr<Expression>> &arguments) {
	return make_uniq<QuantileBindData>(Value::DECIMAL(int16_t(5), 2, 1));
}

unique_ptr<FunctionData> BindMedianDecimal(ClientContext &context, AggregateFunction &function,
                                           vector<unique_ptr<Expression>> &arguments) {
	auto bind_data = BindMedian(context, function, arguments);

	function = GetDiscreteQuantileAggregateFunction(arguments[0]->return_type);
	function.name = "median";
	function.serialize = QuantileSerialize;
	function.deserialize = QuantileDeserialize;
	function.order_dependent = AggregateOrderDependent::NOT_ORDER_DEPENDENT;
	return bind_data;
}

unique_ptr<FunctionData> BindMedianAbsoluteDeviationDecimal(ClientContext &context, AggregateFunction &function,
                                                            vector<unique_ptr<Expression>> &arguments) {
	function = GetMedianAbsoluteDeviationAggregateFunction(arguments[0]->return_type);
	function.name = "mad";
	function.order_dependent = AggregateOrderDependent::NOT_ORDER_DEPENDENT;
	return nullptr;
}

static const Value &CheckQuantile(const Value &quantile_val) {
	if (quantile_val.IsNull()) {
		throw BinderException("QUANTILE parameter cannot be NULL");
	}
	auto quantile = quantile_val.GetValue<double>();
	if (quantile < -1 || quantile > 1) {
		throw BinderException("QUANTILE can only take parameters in the range [-1, 1]");
	}
	if (Value::IsNan(quantile)) {
		throw BinderException("QUANTILE parameter cannot be NaN");
	}

	return quantile_val;
}

unique_ptr<FunctionData> BindQuantile(ClientContext &context, AggregateFunction &function,
                                      vector<unique_ptr<Expression>> &arguments) {
	if (arguments[1]->HasParameter()) {
		throw ParameterNotResolvedException();
	}
	if (!arguments[1]->IsFoldable()) {
		throw BinderException("QUANTILE can only take constant parameters");
	}
	Value quantile_val = ExpressionExecutor::EvaluateScalar(context, *arguments[1]);
	vector<Value> quantiles;
	if (quantile_val.type().id() != LogicalTypeId::LIST) {
		quantiles.push_back(CheckQuantile(quantile_val));
	} else {
		for (const auto &element_val : ListValue::GetChildren(quantile_val)) {
			quantiles.push_back(CheckQuantile(element_val));
		}
	}

	Function::EraseArgument(function, arguments, arguments.size() - 1);
	return make_uniq<QuantileBindData>(quantiles);
}

static void QuantileDecimalSerialize(FieldWriter &writer, const FunctionData *bind_data_p,
                                     const AggregateFunction &function) {
	throw NotImplementedException("FIXME: serializing quantiles with decimals is not supported right now");
}

unique_ptr<FunctionData> BindDiscreteQuantileDecimal(ClientContext &context, AggregateFunction &function,
                                                     vector<unique_ptr<Expression>> &arguments) {
	auto bind_data = BindQuantile(context, function, arguments);
	function = GetDiscreteQuantileAggregateFunction(arguments[0]->return_type);
	function.name = "quantile_disc";
	function.serialize = QuantileDecimalSerialize;
	function.deserialize = QuantileDeserialize;
	function.order_dependent = AggregateOrderDependent::NOT_ORDER_DEPENDENT;
	return bind_data;
}

unique_ptr<FunctionData> BindDiscreteQuantileDecimalList(ClientContext &context, AggregateFunction &function,
                                                         vector<unique_ptr<Expression>> &arguments) {
	auto bind_data = BindQuantile(context, function, arguments);
	function = GetDiscreteQuantileListAggregateFunction(arguments[0]->return_type);
	function.name = "quantile_disc";
	function.serialize = QuantileDecimalSerialize;
	function.deserialize = QuantileDeserialize;
	function.order_dependent = AggregateOrderDependent::NOT_ORDER_DEPENDENT;
	return bind_data;
}

unique_ptr<FunctionData> BindContinuousQuantileDecimal(ClientContext &context, AggregateFunction &function,
                                                       vector<unique_ptr<Expression>> &arguments) {
	auto bind_data = BindQuantile(context, function, arguments);
	function = GetContinuousQuantileAggregateFunction(arguments[0]->return_type);
	function.name = "quantile_cont";
	function.serialize = QuantileDecimalSerialize;
	function.deserialize = QuantileDeserialize;
	function.order_dependent = AggregateOrderDependent::NOT_ORDER_DEPENDENT;
	return bind_data;
}

unique_ptr<FunctionData> BindContinuousQuantileDecimalList(ClientContext &context, AggregateFunction &function,
                                                           vector<unique_ptr<Expression>> &arguments) {
	auto bind_data = BindQuantile(context, function, arguments);
	function = GetContinuousQuantileListAggregateFunction(arguments[0]->return_type);
	function.name = "quantile_cont";
	function.serialize = QuantileDecimalSerialize;
	function.deserialize = QuantileDeserialize;
	function.order_dependent = AggregateOrderDependent::NOT_ORDER_DEPENDENT;
	return bind_data;
}

static bool CanInterpolate(const LogicalType &type) {
	switch (type.id()) {
	case LogicalTypeId::INTERVAL:
	case LogicalTypeId::VARCHAR:
		return false;
	default:
		return true;
	}
}

AggregateFunction GetMedianAggregate(const LogicalType &type) {
	auto fun = CanInterpolate(type) ? GetContinuousQuantileAggregateFunction(type)
	                                : GetDiscreteQuantileAggregateFunction(type);
	fun.bind = BindMedian;
	fun.serialize = QuantileSerialize;
	fun.deserialize = QuantileDeserialize;
	return fun;
}

AggregateFunction GetDiscreteQuantileAggregate(const LogicalType &type) {
	auto fun = GetDiscreteQuantileAggregateFunction(type);
	fun.bind = BindQuantile;
	fun.serialize = QuantileSerialize;
	fun.deserialize = QuantileDeserialize;
	// temporarily push an argument so we can bind the actual quantile
	fun.arguments.emplace_back(LogicalType::DOUBLE);
	fun.order_dependent = AggregateOrderDependent::NOT_ORDER_DEPENDENT;
	return fun;
}

AggregateFunction GetDiscreteQuantileListAggregate(const LogicalType &type) {
	auto fun = GetDiscreteQuantileListAggregateFunction(type);
	fun.bind = BindQuantile;
	fun.serialize = QuantileSerialize;
	fun.deserialize = QuantileDeserialize;
	// temporarily push an argument so we can bind the actual quantile
	auto list_of_double = LogicalType::LIST(LogicalType::DOUBLE);
	fun.arguments.push_back(list_of_double);
	fun.order_dependent = AggregateOrderDependent::NOT_ORDER_DEPENDENT;
	return fun;
}

AggregateFunction GetContinuousQuantileAggregate(const LogicalType &type) {
	auto fun = GetContinuousQuantileAggregateFunction(type);
	fun.bind = BindQuantile;
	fun.serialize = QuantileSerialize;
	fun.deserialize = QuantileDeserialize;
	// temporarily push an argument so we can bind the actual quantile
	fun.arguments.emplace_back(LogicalType::DOUBLE);
	fun.order_dependent = AggregateOrderDependent::NOT_ORDER_DEPENDENT;
	return fun;
}

AggregateFunction GetContinuousQuantileListAggregate(const LogicalType &type) {
	auto fun = GetContinuousQuantileListAggregateFunction(type);
	fun.bind = BindQuantile;
	fun.serialize = QuantileSerialize;
	fun.deserialize = QuantileDeserialize;
	// temporarily push an argument so we can bind the actual quantile
	auto list_of_double = LogicalType::LIST(LogicalType::DOUBLE);
	fun.arguments.push_back(list_of_double);
	fun.order_dependent = AggregateOrderDependent::NOT_ORDER_DEPENDENT;
	return fun;
}

AggregateFunction GetQuantileDecimalAggregate(const vector<LogicalType> &arguments, const LogicalType &return_type,
                                              bind_aggregate_function_t bind) {
	AggregateFunction fun(arguments, return_type, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, bind);
	fun.bind = bind;
	fun.serialize = QuantileSerialize;
	fun.deserialize = QuantileDeserialize;
	fun.order_dependent = AggregateOrderDependent::NOT_ORDER_DEPENDENT;
	return fun;
}

vector<LogicalType> GetQuantileTypes() {
	return {LogicalType::TINYINT,   LogicalType::SMALLINT, LogicalType::INTEGER,      LogicalType::BIGINT,
	        LogicalType::HUGEINT,   LogicalType::FLOAT,    LogicalType::DOUBLE,       LogicalType::DATE,
	        LogicalType::TIMESTAMP, LogicalType::TIME,     LogicalType::TIMESTAMP_TZ, LogicalType::TIME_TZ,
	        LogicalType::INTERVAL,  LogicalType::VARCHAR};
}

AggregateFunctionSet MedianFun::GetFunctions() {
	AggregateFunctionSet median("median");
	median.AddFunction(
	    GetQuantileDecimalAggregate({LogicalTypeId::DECIMAL}, LogicalTypeId::DECIMAL, BindMedianDecimal));
	for (const auto &type : GetQuantileTypes()) {
		median.AddFunction(GetMedianAggregate(type));
	}
	return median;
}

AggregateFunctionSet QuantileDiscFun::GetFunctions() {
	AggregateFunctionSet quantile_disc("quantile_disc");
	quantile_disc.AddFunction(GetQuantileDecimalAggregate({LogicalTypeId::DECIMAL, LogicalType::DOUBLE},
	                                                      LogicalTypeId::DECIMAL, BindDiscreteQuantileDecimal));
	quantile_disc.AddFunction(
	    GetQuantileDecimalAggregate({LogicalTypeId::DECIMAL, LogicalType::LIST(LogicalType::DOUBLE)},
	                                LogicalType::LIST(LogicalTypeId::DECIMAL), BindDiscreteQuantileDecimalList));
	for (const auto &type : GetQuantileTypes()) {
		quantile_disc.AddFunction(GetDiscreteQuantileAggregate(type));
		quantile_disc.AddFunction(GetDiscreteQuantileListAggregate(type));
	}
	return quantile_disc;
	// quantile
}

AggregateFunctionSet QuantileContFun::GetFunctions() {
	AggregateFunctionSet quantile_cont("quantile_cont");
	quantile_cont.AddFunction(GetQuantileDecimalAggregate({LogicalTypeId::DECIMAL, LogicalType::DOUBLE},
	                                                      LogicalTypeId::DECIMAL, BindContinuousQuantileDecimal));
	quantile_cont.AddFunction(
	    GetQuantileDecimalAggregate({LogicalTypeId::DECIMAL, LogicalType::LIST(LogicalType::DOUBLE)},
	                                LogicalType::LIST(LogicalTypeId::DECIMAL), BindContinuousQuantileDecimalList));

	for (const auto &type : GetQuantileTypes()) {
		if (CanInterpolate(type)) {
			quantile_cont.AddFunction(GetContinuousQuantileAggregate(type));
			quantile_cont.AddFunction(GetContinuousQuantileListAggregate(type));
		}
	}
	return quantile_cont;
}

AggregateFunctionSet MadFun::GetFunctions() {
	AggregateFunctionSet mad("mad");
	mad.AddFunction(AggregateFunction({LogicalTypeId::DECIMAL}, LogicalTypeId::DECIMAL, nullptr, nullptr, nullptr,
	                                  nullptr, nullptr, nullptr, BindMedianAbsoluteDeviationDecimal));

	const vector<LogicalType> MAD_TYPES = {LogicalType::FLOAT,     LogicalType::DOUBLE, LogicalType::DATE,
	                                       LogicalType::TIMESTAMP, LogicalType::TIME,   LogicalType::TIMESTAMP_TZ,
	                                       LogicalType::TIME_TZ};
	for (const auto &type : MAD_TYPES) {
		mad.AddFunction(GetMedianAbsoluteDeviationAggregateFunction(type));
	}
	return mad;
}

} // namespace duckdb







#include <algorithm>
#include <stdlib.h>

namespace duckdb {

template <typename T>
struct ReservoirQuantileState {
	T *v;
	idx_t len;
	idx_t pos;
	BaseReservoirSampling *r_samp;

	void Resize(idx_t new_len) {
		if (new_len <= len) {
			return;
		}
		v = (T *)realloc(v, new_len * sizeof(T));
		if (!v) {
			throw InternalException("Memory allocation failure");
		}
		len = new_len;
	}

	void ReplaceElement(T &input) {
		v[r_samp->min_entry] = input;
		r_samp->ReplaceElement();
	}

	void FillReservoir(idx_t sample_size, T element) {
		if (pos < sample_size) {
			v[pos++] = element;
			r_samp->InitializeReservoir(pos, len);
		} else {
			D_ASSERT(r_samp->next_index >= r_samp->current_count);
			if (r_samp->next_index == r_samp->current_count) {
				ReplaceElement(element);
			}
		}
	}
};

struct ReservoirQuantileBindData : public FunctionData {
	ReservoirQuantileBindData(double quantile_p, int32_t sample_size_p)
	    : quantiles(1, quantile_p), sample_size(sample_size_p) {
	}

	ReservoirQuantileBindData(vector<double> quantiles_p, int32_t sample_size_p)
	    : quantiles(std::move(quantiles_p)), sample_size(sample_size_p) {
	}

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<ReservoirQuantileBindData>(quantiles, sample_size);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<ReservoirQuantileBindData>();
		return quantiles == other.quantiles && sample_size == other.sample_size;
	}

	static void Serialize(FieldWriter &writer, const FunctionData *bind_data_p, const AggregateFunction &function) {
		D_ASSERT(bind_data_p);
		auto bind_data = (ReservoirQuantileBindData *)bind_data_p;
		writer.WriteList<double>(bind_data->quantiles);
		writer.WriteField<int32_t>(bind_data->sample_size);
	}

	static unique_ptr<FunctionData> Deserialize(ClientContext &context, FieldReader &reader,
	                                            AggregateFunction &bound_function) {
		auto quantiles = reader.ReadRequiredList<double>();
		auto sample_size = reader.ReadRequired<int32_t>();
		return make_uniq<ReservoirQuantileBindData>(std::move(quantiles), sample_size);
	}

	vector<double> quantiles;
	int32_t sample_size;
};

struct ReservoirQuantileOperation {
	template <class STATE>
	static void Initialize(STATE *state) {
		state->v = nullptr;
		state->len = 0;
		state->pos = 0;
		state->r_samp = nullptr;
	}

	template <class INPUT_TYPE, class STATE, class OP>
	static void ConstantOperation(STATE *state, AggregateInputData &aggr_input_data, INPUT_TYPE *input,
	                              ValidityMask &mask, idx_t count) {
		for (idx_t i = 0; i < count; i++) {
			Operation<INPUT_TYPE, STATE, OP>(state, aggr_input_data, input, mask, 0);
		}
	}

	template <class INPUT_TYPE, class STATE, class OP>
	static void Operation(STATE *state, AggregateInputData &aggr_input_data, INPUT_TYPE *data, ValidityMask &mask,
	                      idx_t idx) {
		auto bind_data = (ReservoirQuantileBindData *)aggr_input_data.bind_data;
		D_ASSERT(bind_data);
		if (state->pos == 0) {
			state->Resize(bind_data->sample_size);
		}
		if (!state->r_samp) {
			state->r_samp = new BaseReservoirSampling();
		}
		D_ASSERT(state->v);
		state->FillReservoir(bind_data->sample_size, data[idx]);
	}

	template <class STATE, class OP>
	static void Combine(const STATE &source, STATE *target, AggregateInputData &) {
		if (source.pos == 0) {
			return;
		}
		if (target->pos == 0) {
			target->Resize(source.len);
		}
		if (!target->r_samp) {
			target->r_samp = new BaseReservoirSampling();
		}
		for (idx_t src_idx = 0; src_idx < source.pos; src_idx++) {
			target->FillReservoir(target->len, source.v[src_idx]);
		}
	}

	template <class STATE>
	static void Destroy(AggregateInputData &aggr_input_data, STATE *state) {
		if (state->v) {
			free(state->v);
			state->v = nullptr;
		}
		if (state->r_samp) {
			delete state->r_samp;
			state->r_samp = nullptr;
		}
	}

	static bool IgnoreNull() {
		return true;
	}
};

struct ReservoirQuantileScalarOperation : public ReservoirQuantileOperation {
	template <class TARGET_TYPE, class STATE>
	static void Finalize(Vector &result, AggregateInputData &aggr_input_data, STATE *state, TARGET_TYPE *target,
	                     ValidityMask &mask, idx_t idx) {
		if (state->pos == 0) {
			mask.SetInvalid(idx);
			return;
		}
		D_ASSERT(state->v);
		D_ASSERT(aggr_input_data.bind_data);
		auto bind_data = (ReservoirQuantileBindData *)aggr_input_data.bind_data;
		auto v_t = state->v;
		D_ASSERT(bind_data->quantiles.size() == 1);
		auto offset = (idx_t)((double)(state->pos - 1) * bind_data->quantiles[0]);
		std::nth_element(v_t, v_t + offset, v_t + state->pos);
		target[idx] = v_t[offset];
	}
};

AggregateFunction GetReservoirQuantileAggregateFunction(PhysicalType type) {
	switch (type) {
	case PhysicalType::INT8:
		return AggregateFunction::UnaryAggregateDestructor<ReservoirQuantileState<int8_t>, int8_t, int8_t,
		                                                   ReservoirQuantileScalarOperation>(LogicalType::TINYINT,
		                                                                                     LogicalType::TINYINT);

	case PhysicalType::INT16:
		return AggregateFunction::UnaryAggregateDestructor<ReservoirQuantileState<int16_t>, int16_t, int16_t,
		                                                   ReservoirQuantileScalarOperation>(LogicalType::SMALLINT,
		                                                                                     LogicalType::SMALLINT);

	case PhysicalType::INT32:
		return AggregateFunction::UnaryAggregateDestructor<ReservoirQuantileState<int32_t>, int32_t, int32_t,
		                                                   ReservoirQuantileScalarOperation>(LogicalType::INTEGER,
		                                                                                     LogicalType::INTEGER);

	case PhysicalType::INT64:
		return AggregateFunction::UnaryAggregateDestructor<ReservoirQuantileState<int64_t>, int64_t, int64_t,
		                                                   ReservoirQuantileScalarOperation>(LogicalType::BIGINT,
		                                                                                     LogicalType::BIGINT);

	case PhysicalType::INT128:
		return AggregateFunction::UnaryAggregateDestructor<ReservoirQuantileState<hugeint_t>, hugeint_t, hugeint_t,
		                                                   ReservoirQuantileScalarOperation>(LogicalType::HUGEINT,
		                                                                                     LogicalType::HUGEINT);
	case PhysicalType::FLOAT:
		return AggregateFunction::UnaryAggregateDestructor<ReservoirQuantileState<float>, float, float,
		                                                   ReservoirQuantileScalarOperation>(LogicalType::FLOAT,
		                                                                                     LogicalType::FLOAT);
	case PhysicalType::DOUBLE:
		return AggregateFunction::UnaryAggregateDestructor<ReservoirQuantileState<double>, double, double,
		                                                   ReservoirQuantileScalarOperation>(LogicalType::DOUBLE,
		                                                                                     LogicalType::DOUBLE);
	default:
		throw InternalException("Unimplemented reservoir quantile aggregate");
	}
}

template <class CHILD_TYPE>
struct ReservoirQuantileListOperation : public ReservoirQuantileOperation {

	template <class RESULT_TYPE, class STATE>
	static void Finalize(Vector &result_list, AggregateInputData &aggr_input_data, STATE *state, RESULT_TYPE *target,
	                     ValidityMask &mask, idx_t idx) {
		if (state->pos == 0) {
			mask.SetInvalid(idx);
			return;
		}

		D_ASSERT(aggr_input_data.bind_data);
		auto bind_data = (ReservoirQuantileBindData *)aggr_input_data.bind_data;

		auto &result = ListVector::GetEntry(result_list);
		auto ridx = ListVector::GetListSize(result_list);
		ListVector::Reserve(result_list, ridx + bind_data->quantiles.size());
		auto rdata = FlatVector::GetData<CHILD_TYPE>(result);

		auto v_t = state->v;
		D_ASSERT(v_t);

		auto &entry = target[idx];
		entry.offset = ridx;
		entry.length = bind_data->quantiles.size();
		for (size_t q = 0; q < entry.length; ++q) {
			const auto &quantile = bind_data->quantiles[q];
			auto offset = (idx_t)((double)(state->pos - 1) * quantile);
			std::nth_element(v_t, v_t + offset, v_t + state->pos);
			rdata[ridx + q] = v_t[offset];
		}

		ListVector::SetListSize(result_list, entry.offset + entry.length);
	}

	template <class STATE_TYPE, class RESULT_TYPE>
	static void FinalizeList(Vector &states, AggregateInputData &aggr_input_data, Vector &result, idx_t count, // NOLINT
	                         idx_t offset) {
		D_ASSERT(result.GetType().id() == LogicalTypeId::LIST);

		D_ASSERT(aggr_input_data.bind_data);
		auto bind_data = (ReservoirQuantileBindData *)aggr_input_data.bind_data;

		if (states.GetVectorType() == VectorType::CONSTANT_VECTOR) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
			ListVector::Reserve(result, bind_data->quantiles.size());

			auto sdata = ConstantVector::GetData<STATE_TYPE *>(states);
			auto rdata = ConstantVector::GetData<RESULT_TYPE>(result);
			auto &mask = ConstantVector::Validity(result);
			Finalize<RESULT_TYPE, STATE_TYPE>(result, aggr_input_data, sdata[0], rdata, mask, 0);
		} else {
			D_ASSERT(states.GetVectorType() == VectorType::FLAT_VECTOR);
			result.SetVectorType(VectorType::FLAT_VECTOR);
			ListVector::Reserve(result, (offset + count) * bind_data->quantiles.size());

			auto sdata = FlatVector::GetData<STATE_TYPE *>(states);
			auto rdata = FlatVector::GetData<RESULT_TYPE>(result);
			auto &mask = FlatVector::Validity(result);
			for (idx_t i = 0; i < count; i++) {
				Finalize<RESULT_TYPE, STATE_TYPE>(result, aggr_input_data, sdata[i], rdata, mask, i + offset);
			}
		}

		result.Verify(count);
	}
};

template <class STATE, class INPUT_TYPE, class RESULT_TYPE, class OP>
static AggregateFunction ReservoirQuantileListAggregate(const LogicalType &input_type, const LogicalType &child_type) {
	LogicalType result_type = LogicalType::LIST(child_type);
	return AggregateFunction(
	    {input_type}, result_type, AggregateFunction::StateSize<STATE>, AggregateFunction::StateInitialize<STATE, OP>,
	    AggregateFunction::UnaryScatterUpdate<STATE, INPUT_TYPE, OP>, AggregateFunction::StateCombine<STATE, OP>,
	    OP::template FinalizeList<STATE, RESULT_TYPE>, AggregateFunction::UnaryUpdate<STATE, INPUT_TYPE, OP>, nullptr,
	    AggregateFunction::StateDestroy<STATE, OP>);
}

template <typename INPUT_TYPE, typename SAVE_TYPE>
AggregateFunction GetTypedReservoirQuantileListAggregateFunction(const LogicalType &type) {
	using STATE = ReservoirQuantileState<SAVE_TYPE>;
	using OP = ReservoirQuantileListOperation<INPUT_TYPE>;
	auto fun = ReservoirQuantileListAggregate<STATE, INPUT_TYPE, list_entry_t, OP>(type, type);
	return fun;
}

AggregateFunction GetReservoirQuantileListAggregateFunction(const LogicalType &type) {
	switch (type.id()) {
	case LogicalTypeId::TINYINT:
		return GetTypedReservoirQuantileListAggregateFunction<int8_t, int8_t>(type);
	case LogicalTypeId::SMALLINT:
		return GetTypedReservoirQuantileListAggregateFunction<int16_t, int16_t>(type);
	case LogicalTypeId::INTEGER:
		return GetTypedReservoirQuantileListAggregateFunction<int32_t, int32_t>(type);
	case LogicalTypeId::BIGINT:
		return GetTypedReservoirQuantileListAggregateFunction<int64_t, int64_t>(type);
	case LogicalTypeId::HUGEINT:
		return GetTypedReservoirQuantileListAggregateFunction<hugeint_t, hugeint_t>(type);
	case LogicalTypeId::FLOAT:
		return GetTypedReservoirQuantileListAggregateFunction<float, float>(type);
	case LogicalTypeId::DOUBLE:
		return GetTypedReservoirQuantileListAggregateFunction<double, double>(type);
	case LogicalTypeId::DECIMAL:
		switch (type.InternalType()) {
		case PhysicalType::INT16:
			return GetTypedReservoirQuantileListAggregateFunction<int16_t, int16_t>(type);
		case PhysicalType::INT32:
			return GetTypedReservoirQuantileListAggregateFunction<int32_t, int32_t>(type);
		case PhysicalType::INT64:
			return GetTypedReservoirQuantileListAggregateFunction<int64_t, int64_t>(type);
		case PhysicalType::INT128:
			return GetTypedReservoirQuantileListAggregateFunction<hugeint_t, hugeint_t>(type);
		default:
			throw NotImplementedException("Unimplemented reservoir quantile list aggregate");
		}
	default:
		// TODO: Add quantitative temporal types
		throw NotImplementedException("Unimplemented reservoir quantile list aggregate");
	}
}

static double CheckReservoirQuantile(const Value &quantile_val) {
	if (quantile_val.IsNull()) {
		throw BinderException("RESERVOIR_QUANTILE QUANTILE parameter cannot be NULL");
	}
	auto quantile = quantile_val.GetValue<double>();
	if (quantile < 0 || quantile > 1) {
		throw BinderException("RESERVOIR_QUANTILE can only take parameters in the range [0, 1]");
	}
	return quantile;
}

unique_ptr<FunctionData> BindReservoirQuantile(ClientContext &context, AggregateFunction &function,
                                               vector<unique_ptr<Expression>> &arguments) {
	D_ASSERT(arguments.size() >= 2);
	if (arguments[1]->HasParameter()) {
		throw ParameterNotResolvedException();
	}
	if (!arguments[1]->IsFoldable()) {
		throw BinderException("RESERVOIR_QUANTILE can only take constant quantile parameters");
	}
	Value quantile_val = ExpressionExecutor::EvaluateScalar(context, *arguments[1]);
	vector<double> quantiles;
	if (quantile_val.type().id() != LogicalTypeId::LIST) {
		quantiles.push_back(CheckReservoirQuantile(quantile_val));
	} else {
		for (const auto &element_val : ListValue::GetChildren(quantile_val)) {
			quantiles.push_back(CheckReservoirQuantile(element_val));
		}
	}

	if (arguments.size() == 2) {
		if (function.arguments.size() == 2) {
			Function::EraseArgument(function, arguments, arguments.size() - 1);
		} else {
			arguments.pop_back();
		}
		return make_uniq<ReservoirQuantileBindData>(quantiles, 8192);
	}
	if (!arguments[2]->IsFoldable()) {
		throw BinderException("RESERVOIR_QUANTILE can only take constant sample size parameters");
	}
	Value sample_size_val = ExpressionExecutor::EvaluateScalar(context, *arguments[2]);
	if (sample_size_val.IsNull()) {
		throw BinderException("Size of the RESERVOIR_QUANTILE sample cannot be NULL");
	}
	auto sample_size = sample_size_val.GetValue<int32_t>();

	if (sample_size_val.IsNull() || sample_size <= 0) {
		throw BinderException("Size of the RESERVOIR_QUANTILE sample must be bigger than 0");
	}

	// remove the quantile argument so we can use the unary aggregate
	Function::EraseArgument(function, arguments, arguments.size() - 1);
	Function::EraseArgument(function, arguments, arguments.size() - 1);
	return make_uniq<ReservoirQuantileBindData>(quantiles, sample_size);
}

unique_ptr<FunctionData> BindReservoirQuantileDecimal(ClientContext &context, AggregateFunction &function,
                                                      vector<unique_ptr<Expression>> &arguments) {
	function = GetReservoirQuantileAggregateFunction(arguments[0]->return_type.InternalType());
	auto bind_data = BindReservoirQuantile(context, function, arguments);
	function.name = "reservoir_quantile";
	function.serialize = ReservoirQuantileBindData::Serialize;
	function.deserialize = ReservoirQuantileBindData::Deserialize;
	return bind_data;
}

AggregateFunction GetReservoirQuantileAggregate(PhysicalType type) {
	auto fun = GetReservoirQuantileAggregateFunction(type);
	fun.bind = BindReservoirQuantile;
	fun.serialize = ReservoirQuantileBindData::Serialize;
	fun.deserialize = ReservoirQuantileBindData::Deserialize;
	// temporarily push an argument so we can bind the actual quantile
	fun.arguments.emplace_back(LogicalType::DOUBLE);
	return fun;
}

unique_ptr<FunctionData> BindReservoirQuantileDecimalList(ClientContext &context, AggregateFunction &function,
                                                          vector<unique_ptr<Expression>> &arguments) {
	function = GetReservoirQuantileListAggregateFunction(arguments[0]->return_type);
	auto bind_data = BindReservoirQuantile(context, function, arguments);
	function.serialize = ReservoirQuantileBindData::Serialize;
	function.deserialize = ReservoirQuantileBindData::Deserialize;
	function.name = "reservoir_quantile";
	return bind_data;
}

AggregateFunction GetReservoirQuantileListAggregate(const LogicalType &type) {
	auto fun = GetReservoirQuantileListAggregateFunction(type);
	fun.bind = BindReservoirQuantile;
	fun.serialize = ReservoirQuantileBindData::Serialize;
	fun.deserialize = ReservoirQuantileBindData::Deserialize;
	// temporarily push an argument so we can bind the actual quantile
	auto list_of_double = LogicalType::LIST(LogicalType::DOUBLE);
	fun.arguments.push_back(list_of_double);
	return fun;
}

static void DefineReservoirQuantile(AggregateFunctionSet &set, const LogicalType &type) {
	//	Four versions: type, scalar/list[, count]
	auto fun = GetReservoirQuantileAggregate(type.InternalType());
	set.AddFunction(fun);

	fun.arguments.emplace_back(LogicalType::INTEGER);
	set.AddFunction(fun);

	// List variants
	fun = GetReservoirQuantileListAggregate(type);
	set.AddFunction(fun);

	fun.arguments.emplace_back(LogicalType::INTEGER);
	set.AddFunction(fun);
}

static void GetReservoirQuantileDecimalFunction(AggregateFunctionSet &set, const vector<LogicalType> &arguments,
                                                const LogicalType &return_value) {
	AggregateFunction fun(arguments, return_value, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
	                      BindReservoirQuantileDecimal);
	fun.serialize = ReservoirQuantileBindData::Serialize;
	fun.deserialize = ReservoirQuantileBindData::Deserialize;
	set.AddFunction(fun);

	fun.arguments.emplace_back(LogicalType::INTEGER);
	set.AddFunction(fun);
}

AggregateFunctionSet ReservoirQuantileFun::GetFunctions() {
	AggregateFunctionSet reservoir_quantile;

	// DECIMAL
	GetReservoirQuantileDecimalFunction(reservoir_quantile, {LogicalTypeId::DECIMAL, LogicalType::DOUBLE},
	                                    LogicalTypeId::DECIMAL);
	GetReservoirQuantileDecimalFunction(reservoir_quantile,
	                                    {LogicalTypeId::DECIMAL, LogicalType::LIST(LogicalType::DOUBLE)},
	                                    LogicalType::LIST(LogicalTypeId::DECIMAL));

	DefineReservoirQuantile(reservoir_quantile, LogicalTypeId::TINYINT);
	DefineReservoirQuantile(reservoir_quantile, LogicalTypeId::SMALLINT);
	DefineReservoirQuantile(reservoir_quantile, LogicalTypeId::INTEGER);
	DefineReservoirQuantile(reservoir_quantile, LogicalTypeId::BIGINT);
	DefineReservoirQuantile(reservoir_quantile, LogicalTypeId::HUGEINT);
	DefineReservoirQuantile(reservoir_quantile, LogicalTypeId::FLOAT);
	DefineReservoirQuantile(reservoir_quantile, LogicalTypeId::DOUBLE);
	return reservoir_quantile;
}

} // namespace duckdb







namespace duckdb {

struct HistogramFunctor {
	template <class T, class MAP_TYPE = map<T, idx_t>>
	static void HistogramUpdate(UnifiedVectorFormat &sdata, UnifiedVectorFormat &input_data, idx_t count) {

		auto states = (HistogramAggState<T, MAP_TYPE> **)sdata.data;
		for (idx_t i = 0; i < count; i++) {
			if (input_data.validity.RowIsValid(input_data.sel->get_index(i))) {
				auto state = states[sdata.sel->get_index(i)];
				if (!state->hist) {
					state->hist = new MAP_TYPE();
				}
				auto value = (T *)input_data.data;
				(*state->hist)[value[input_data.sel->get_index(i)]]++;
			}
		}
	}

	template <class T>
	static Value HistogramFinalize(T first) {
		return Value::CreateValue(first);
	}
};

struct HistogramStringFunctor {
	template <class T, class MAP_TYPE = map<T, idx_t>>
	static void HistogramUpdate(UnifiedVectorFormat &sdata, UnifiedVectorFormat &input_data, idx_t count) {

		auto states = (HistogramAggState<T, MAP_TYPE> **)sdata.data;
		for (idx_t i = 0; i < count; i++) {
			if (input_data.validity.RowIsValid(input_data.sel->get_index(i))) {
				auto state = states[sdata.sel->get_index(i)];
				if (!state->hist) {
					state->hist = new MAP_TYPE();
				}
				auto value = (string_t *)input_data.data;
				(*state->hist)[value[input_data.sel->get_index(i)].GetString()]++;
			}
		}
	}

	template <class T>
	static Value HistogramFinalize(T first) {
		string_t value = first;
		return Value::CreateValue(value);
	}
};

struct HistogramFunction {
	template <class STATE>
	static void Initialize(STATE *state) {
		state->hist = nullptr;
	}

	template <class STATE>
	static void Destroy(AggregateInputData &aggr_input_data, STATE *state) {
		if (state->hist) {
			delete state->hist;
		}
	}

	static bool IgnoreNull() {
		return true;
	}
};

template <class OP, class T, class MAP_TYPE>
static void HistogramUpdateFunction(Vector inputs[], AggregateInputData &, idx_t input_count, Vector &state_vector,
                                    idx_t count) {

	D_ASSERT(input_count == 1);

	auto &input = inputs[0];
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	UnifiedVectorFormat input_data;
	input.ToUnifiedFormat(count, input_data);

	OP::template HistogramUpdate<T, MAP_TYPE>(sdata, input_data, count);
}

template <class T, class MAP_TYPE>
static void HistogramCombineFunction(Vector &state, Vector &combined, AggregateInputData &, idx_t count) {

	UnifiedVectorFormat sdata;
	state.ToUnifiedFormat(count, sdata);
	auto states_ptr = (HistogramAggState<T, MAP_TYPE> **)sdata.data;

	auto combined_ptr = FlatVector::GetData<HistogramAggState<T, MAP_TYPE> *>(combined);

	for (idx_t i = 0; i < count; i++) {
		auto state = states_ptr[sdata.sel->get_index(i)];
		if (!state->hist) {
			continue;
		}
		if (!combined_ptr[i]->hist) {
			combined_ptr[i]->hist = new MAP_TYPE();
		}
		D_ASSERT(combined_ptr[i]->hist);
		D_ASSERT(state->hist);
		for (auto &entry : *state->hist) {
			(*combined_ptr[i]->hist)[entry.first] += entry.second;
		}
	}
}

template <class OP, class T, class MAP_TYPE>
static void HistogramFinalizeFunction(Vector &state_vector, AggregateInputData &, Vector &result, idx_t count,
                                      idx_t offset) {

	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (HistogramAggState<T, MAP_TYPE> **)sdata.data;

	auto &mask = FlatVector::Validity(result);
	auto old_len = ListVector::GetListSize(result);

	for (idx_t i = 0; i < count; i++) {
		const auto rid = i + offset;
		auto state = states[sdata.sel->get_index(i)];
		if (!state->hist) {
			mask.SetInvalid(rid);
			continue;
		}

		for (auto &entry : *state->hist) {
			Value bucket_value = OP::template HistogramFinalize<T>(entry.first);
			auto count_value = Value::CreateValue(entry.second);
			auto struct_value =
			    Value::STRUCT({std::make_pair("key", bucket_value), std::make_pair("value", count_value)});
			ListVector::PushBack(result, struct_value);
		}

		auto list_struct_data = ListVector::GetData(result);
		list_struct_data[rid].length = ListVector::GetListSize(result) - old_len;
		list_struct_data[rid].offset = old_len;
		old_len += list_struct_data[rid].length;
	}
}

unique_ptr<FunctionData> HistogramBindFunction(ClientContext &context, AggregateFunction &function,
                                               vector<unique_ptr<Expression>> &arguments) {

	D_ASSERT(arguments.size() == 1);

	if (arguments[0]->return_type.id() == LogicalTypeId::LIST ||
	    arguments[0]->return_type.id() == LogicalTypeId::STRUCT ||
	    arguments[0]->return_type.id() == LogicalTypeId::MAP) {
		throw NotImplementedException("Unimplemented type for histogram %s", arguments[0]->return_type.ToString());
	}

	auto struct_type = LogicalType::MAP(arguments[0]->return_type, LogicalType::UBIGINT);

	function.return_type = struct_type;
	return make_uniq<VariableReturnBindData>(function.return_type);
}

template <class OP, class T, class MAP_TYPE = map<T, idx_t>>
static AggregateFunction GetHistogramFunction(const LogicalType &type) {

	using STATE_TYPE = HistogramAggState<T, MAP_TYPE>;

	return AggregateFunction("histogram", {type}, LogicalTypeId::MAP, AggregateFunction::StateSize<STATE_TYPE>,
	                         AggregateFunction::StateInitialize<STATE_TYPE, HistogramFunction>,
	                         HistogramUpdateFunction<OP, T, MAP_TYPE>, HistogramCombineFunction<T, MAP_TYPE>,
	                         HistogramFinalizeFunction<OP, T, MAP_TYPE>, nullptr, HistogramBindFunction,
	                         AggregateFunction::StateDestroy<STATE_TYPE, HistogramFunction>);
}

template <class OP, class T, bool IS_ORDERED>
AggregateFunction GetMapType(const LogicalType &type) {

	if (IS_ORDERED) {
		return GetHistogramFunction<OP, T>(type);
	}
	return GetHistogramFunction<OP, T, unordered_map<T, idx_t>>(type);
}

template <bool IS_ORDERED = true>
AggregateFunction GetHistogramFunction(const LogicalType &type) {

	switch (type.id()) {
	case LogicalType::BOOLEAN:
		return GetMapType<HistogramFunctor, bool, IS_ORDERED>(type);
	case LogicalType::UTINYINT:
		return GetMapType<HistogramFunctor, uint8_t, IS_ORDERED>(type);
	case LogicalType::USMALLINT:
		return GetMapType<HistogramFunctor, uint16_t, IS_ORDERED>(type);
	case LogicalType::UINTEGER:
		return GetMapType<HistogramFunctor, uint32_t, IS_ORDERED>(type);
	case LogicalType::UBIGINT:
		return GetMapType<HistogramFunctor, uint64_t, IS_ORDERED>(type);
	case LogicalType::TINYINT:
		return GetMapType<HistogramFunctor, int8_t, IS_ORDERED>(type);
	case LogicalType::SMALLINT:
		return GetMapType<HistogramFunctor, int16_t, IS_ORDERED>(type);
	case LogicalType::INTEGER:
		return GetMapType<HistogramFunctor, int32_t, IS_ORDERED>(type);
	case LogicalType::BIGINT:
		return GetMapType<HistogramFunctor, int64_t, IS_ORDERED>(type);
	case LogicalType::FLOAT:
		return GetMapType<HistogramFunctor, float, IS_ORDERED>(type);
	case LogicalType::DOUBLE:
		return GetMapType<HistogramFunctor, double, IS_ORDERED>(type);
	case LogicalType::VARCHAR:
		return GetMapType<HistogramStringFunctor, string, IS_ORDERED>(type);
	case LogicalType::TIMESTAMP:
		return GetMapType<HistogramFunctor, timestamp_t, IS_ORDERED>(type);
	case LogicalType::TIMESTAMP_TZ:
		return GetMapType<HistogramFunctor, timestamp_tz_t, IS_ORDERED>(type);
	case LogicalType::TIMESTAMP_S:
		return GetMapType<HistogramFunctor, timestamp_sec_t, IS_ORDERED>(type);
	case LogicalType::TIMESTAMP_MS:
		return GetMapType<HistogramFunctor, timestamp_ms_t, IS_ORDERED>(type);
	case LogicalType::TIMESTAMP_NS:
		return GetMapType<HistogramFunctor, timestamp_ns_t, IS_ORDERED>(type);
	case LogicalType::TIME:
		return GetMapType<HistogramFunctor, dtime_t, IS_ORDERED>(type);
	case LogicalType::TIME_TZ:
		return GetMapType<HistogramFunctor, dtime_tz_t, IS_ORDERED>(type);
	case LogicalType::DATE:
		return GetMapType<HistogramFunctor, date_t, IS_ORDERED>(type);
	default:
		throw InternalException("Unimplemented histogram aggregate");
	}
}

AggregateFunctionSet HistogramFun::GetFunctions() {
	AggregateFunctionSet fun;
	fun.AddFunction(GetHistogramFunction<>(LogicalType::BOOLEAN));
	fun.AddFunction(GetHistogramFunction<>(LogicalType::UTINYINT));
	fun.AddFunction(GetHistogramFunction<>(LogicalType::USMALLINT));
	fun.AddFunction(GetHistogramFunction<>(LogicalType::UINTEGER));
	fun.AddFunction(GetHistogramFunction<>(LogicalType::UBIGINT));
	fun.AddFunction(GetHistogramFunction<>(LogicalType::TINYINT));
	fun.AddFunction(GetHistogramFunction<>(LogicalType::SMALLINT));
	fun.AddFunction(GetHistogramFunction<>(LogicalType::INTEGER));
	fun.AddFunction(GetHistogramFunction<>(LogicalType::BIGINT));
	fun.AddFunction(GetHistogramFunction<>(LogicalType::FLOAT));
	fun.AddFunction(GetHistogramFunction<>(LogicalType::DOUBLE));
	fun.AddFunction(GetHistogramFunction<>(LogicalType::VARCHAR));
	fun.AddFunction(GetHistogramFunction<>(LogicalType::TIMESTAMP));
	fun.AddFunction(GetHistogramFunction<>(LogicalType::TIMESTAMP_TZ));
	fun.AddFunction(GetHistogramFunction<>(LogicalType::TIMESTAMP_S));
	fun.AddFunction(GetHistogramFunction<>(LogicalType::TIMESTAMP_MS));
	fun.AddFunction(GetHistogramFunction<>(LogicalType::TIMESTAMP_NS));
	fun.AddFunction(GetHistogramFunction<>(LogicalType::TIME));
	fun.AddFunction(GetHistogramFunction<>(LogicalType::TIME_TZ));
	fun.AddFunction(GetHistogramFunction<>(LogicalType::DATE));
	return fun;
}

AggregateFunction HistogramFun::GetHistogramUnorderedMap(LogicalType &type) {
	const auto &const_type = type;
	return GetHistogramFunction<false>(const_type);
}

} // namespace duckdb





namespace duckdb {

static void RecursiveFlatten(Vector &vector, idx_t &count) {
	if (vector.GetVectorType() != VectorType::FLAT_VECTOR) {
		vector.Flatten(count);
	}

	auto internal_type = vector.GetType().InternalType();
	if (internal_type == PhysicalType::LIST) {
		auto &child_vector = ListVector::GetEntry(vector);
		auto child_vector_count = ListVector::GetListSize(vector);
		RecursiveFlatten(child_vector, child_vector_count);
	} else if (internal_type == PhysicalType::STRUCT) {
		auto &children = StructVector::GetEntries(vector);
		for (auto &child : children) {
			RecursiveFlatten(*child, count);
		}
	}
}

struct ListBindData : public FunctionData {
	explicit ListBindData(const LogicalType &stype_p);
	~ListBindData() override;

	LogicalType stype;
	ListSegmentFunctions functions;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<ListBindData>(stype);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = (const ListBindData &)other_p;
		return stype == other.stype;
	}
};

ListBindData::ListBindData(const LogicalType &stype_p) : stype(stype_p) {
	// always unnest once because the result vector is of type LIST
	auto type = ListType::GetChildType(stype_p);
	GetSegmentDataFunctions(functions, type);
}

ListBindData::~ListBindData() {
}

struct ListAggState {
	LinkedList linked_list;
};

struct ListFunction {
	template <class STATE>
	static void Initialize(STATE *state) {
		state->linked_list.total_capacity = 0;
		state->linked_list.first_segment = nullptr;
		state->linked_list.last_segment = nullptr;
	}

	template <class STATE>
	static void Destroy(AggregateInputData &aggr_input_data, STATE *state) {
		D_ASSERT(state);
		auto &list_bind_data = aggr_input_data.bind_data->Cast<ListBindData>();
		list_bind_data.functions.Destroy(aggr_input_data.allocator, state->linked_list);
	}
	static bool IgnoreNull() {
		return false;
	}
};

static void ListUpdateFunction(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                               Vector &state_vector, idx_t count) {
	D_ASSERT(input_count == 1);

	auto &input = inputs[0];
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);

	auto states = (ListAggState **)sdata.data;
	RecursiveFlatten(input, count);

	auto &list_bind_data = aggr_input_data.bind_data->Cast<ListBindData>();

	for (idx_t i = 0; i < count; i++) {
		auto state = states[sdata.sel->get_index(i)];
		list_bind_data.functions.AppendRow(aggr_input_data.allocator, state->linked_list, input, i, count);
	}
}

static void ListCombineFunction(Vector &state, Vector &combined, AggregateInputData &aggr_input_data, idx_t count) {
	UnifiedVectorFormat sdata;
	state.ToUnifiedFormat(count, sdata);
	auto states_ptr = (ListAggState **)sdata.data;

	auto &list_bind_data = aggr_input_data.bind_data->Cast<ListBindData>();

	auto combined_ptr = FlatVector::GetData<ListAggState *>(combined);
	for (idx_t i = 0; i < count; i++) {
		auto state = states_ptr[sdata.sel->get_index(i)];
		if (state->linked_list.total_capacity == 0) {
			// NULL, no need to append.
			continue;
		}

		// copy the linked list of the state
		auto copied_linked_list = LinkedList(state->linked_list.total_capacity, nullptr, nullptr);
		list_bind_data.functions.CopyLinkedList(state->linked_list, copied_linked_list, aggr_input_data.allocator);

		// append the copied linked list to the combined state
		if (combined_ptr[i]->linked_list.last_segment) {
			combined_ptr[i]->linked_list.last_segment->next = copied_linked_list.first_segment;
		} else {
			combined_ptr[i]->linked_list.first_segment = copied_linked_list.first_segment;
		}
		combined_ptr[i]->linked_list.last_segment = copied_linked_list.last_segment;
		combined_ptr[i]->linked_list.total_capacity += copied_linked_list.total_capacity;
	}
}

static void ListFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                         idx_t offset) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (ListAggState **)sdata.data;

	D_ASSERT(result.GetType().id() == LogicalTypeId::LIST);

	auto &mask = FlatVector::Validity(result);
	auto result_data = FlatVector::GetData<list_entry_t>(result);
	size_t total_len = ListVector::GetListSize(result);

	auto &list_bind_data = aggr_input_data.bind_data->Cast<ListBindData>();
	// first iterate over all of the entries and set up the list entries, plus get the newly required total length
	for (idx_t i = 0; i < count; i++) {
		auto state = states[sdata.sel->get_index(i)];
		const auto rid = i + offset;
		result_data[rid].offset = total_len;
		if (state->linked_list.total_capacity == 0) {
			mask.SetInvalid(rid);
			result_data[rid].length = 0;
			continue;
		}
		// set the length and offset of this list in the result vector
		auto total_capacity = state->linked_list.total_capacity;
		result_data[rid].length = total_capacity;
		total_len += total_capacity;
	}
	// reserve capacity, then iterate over all of the entries again and copy over the data tot he child vector
	ListVector::Reserve(result, total_len);
	auto &result_child = ListVector::GetEntry(result);
	for (idx_t i = 0; i < count; i++) {
		auto state = states[sdata.sel->get_index(i)];
		const auto rid = i + offset;
		if (state->linked_list.total_capacity == 0) {
			continue;
		}

		idx_t current_offset = result_data[rid].offset;
		list_bind_data.functions.BuildListVector(state->linked_list, result_child, current_offset);
	}
	ListVector::SetListSize(result, total_len);
}

unique_ptr<FunctionData> ListBindFunction(ClientContext &context, AggregateFunction &function,
                                          vector<unique_ptr<Expression>> &arguments) {
	D_ASSERT(arguments.size() == 1);
	D_ASSERT(function.arguments.size() == 1);

	if (arguments[0]->return_type.id() == LogicalTypeId::UNKNOWN) {
		function.arguments[0] = LogicalTypeId::UNKNOWN;
		function.return_type = LogicalType::SQLNULL;
		return nullptr;
	}

	function.return_type = LogicalType::LIST(arguments[0]->return_type);
	return make_uniq<ListBindData>(function.return_type);
}

AggregateFunction ListFun::GetFunction() {
	return AggregateFunction({LogicalType::ANY}, LogicalTypeId::LIST, AggregateFunction::StateSize<ListAggState>,
	                         AggregateFunction::StateInitialize<ListAggState, ListFunction>, ListUpdateFunction,
	                         ListCombineFunction, ListFinalize, nullptr, ListBindFunction,
	                         AggregateFunction::StateDestroy<ListAggState, ListFunction>, nullptr, nullptr);
}

} // namespace duckdb






namespace duckdb {
struct RegrState {
	double sum;
	size_t count;
};

struct RegrAvgFunction {
	template <class STATE>
	static void Initialize(STATE *state) {
		state->sum = 0;
		state->count = 0;
	}

	template <class STATE, class OP>
	static void Combine(const STATE &source, STATE *target, AggregateInputData &) {
		target->sum += source.sum;
		target->count += source.count;
	}

	template <class T, class STATE>
	static void Finalize(Vector &result, AggregateInputData &, STATE *state, T *target, ValidityMask &mask, idx_t idx) {
		if (state->count == 0) {
			mask.SetInvalid(idx);
		} else {
			target[idx] = state->sum / (double)state->count;
		}
	}
	static bool IgnoreNull() {
		return true;
	}
};
struct RegrAvgXFunction : RegrAvgFunction {
	template <class A_TYPE, class B_TYPE, class STATE, class OP>
	static void Operation(STATE *state, AggregateInputData &, A_TYPE *x_data, B_TYPE *y_data, ValidityMask &amask,
	                      ValidityMask &bmask, idx_t xidx, idx_t yidx) {
		state->sum += y_data[yidx];
		state->count++;
	}
};

struct RegrAvgYFunction : RegrAvgFunction {
	template <class A_TYPE, class B_TYPE, class STATE, class OP>
	static void Operation(STATE *state, AggregateInputData &, A_TYPE *x_data, B_TYPE *y_data, ValidityMask &amask,
	                      ValidityMask &bmask, idx_t xidx, idx_t yidx) {
		state->sum += x_data[xidx];
		state->count++;
	}
};

AggregateFunction RegrAvgxFun::GetFunction() {
	return AggregateFunction::BinaryAggregate<RegrState, double, double, double, RegrAvgXFunction>(
	    LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE);
}

AggregateFunction RegrAvgyFun::GetFunction() {
	return AggregateFunction::BinaryAggregate<RegrState, double, double, double, RegrAvgYFunction>(
	    LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE);
}

} // namespace duckdb







namespace duckdb {

AggregateFunction RegrCountFun::GetFunction() {
	auto regr_count = AggregateFunction::BinaryAggregate<size_t, double, double, uint32_t, RegrCountFunction>(
	    LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::UINTEGER);
	regr_count.name = "regr_count";
	regr_count.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	return regr_count;
}

} // namespace duckdb
//! AVG(y)-REGR_SLOPE(y,x)*AVG(x)





namespace duckdb {

struct RegrInterceptState {
	size_t count;
	double sum_x;
	double sum_y;
	RegrSlopeState slope;
};

struct RegrInterceptOperation {
	template <class STATE>
	static void Initialize(STATE *state) {
		state->count = 0;
		state->sum_x = 0;
		state->sum_y = 0;
		RegrSlopeOperation::Initialize<RegrSlopeState>(&state->slope);
	}

	template <class A_TYPE, class B_TYPE, class STATE, class OP>
	static void Operation(STATE *state, AggregateInputData &aggr_input_data, A_TYPE *x_data, B_TYPE *y_data,
	                      ValidityMask &amask, ValidityMask &bmask, idx_t xidx, idx_t yidx) {
		state->count++;
		state->sum_x += y_data[yidx];
		state->sum_y += x_data[xidx];
		RegrSlopeOperation::Operation<A_TYPE, B_TYPE, RegrSlopeState, OP>(&state->slope, aggr_input_data, x_data,
		                                                                  y_data, amask, bmask, xidx, yidx);
	}

	template <class STATE, class OP>
	static void Combine(const STATE &source, STATE *target, AggregateInputData &aggr_input_data) {
		target->count += source.count;
		target->sum_x += source.sum_x;
		target->sum_y += source.sum_y;
		RegrSlopeOperation::Combine<RegrSlopeState, OP>(source.slope, &target->slope, aggr_input_data);
	}

	template <class T, class STATE>
	static void Finalize(Vector &result, AggregateInputData &aggr_input_data, STATE *state, T *target,
	                     ValidityMask &mask, idx_t idx) {
		if (state->count == 0) {
			mask.SetInvalid(idx);
			return;
		}
		RegrSlopeOperation::Finalize<T, RegrSlopeState>(result, aggr_input_data, &state->slope, target, mask, idx);
		auto x_avg = state->sum_x / state->count;
		auto y_avg = state->sum_y / state->count;
		target[idx] = y_avg - target[idx] * x_avg;
	}

	static bool IgnoreNull() {
		return true;
	}
};

AggregateFunction RegrInterceptFun::GetFunction() {
	return AggregateFunction::BinaryAggregate<RegrInterceptState, double, double, double, RegrInterceptOperation>(
	    LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE);
}

} // namespace duckdb
// Returns the coefficient of determination for non-null pairs in a group.
// It is computed for non-null pairs using the following formula:
// null                 if var_pop(x) = 0, else
// 1                    if var_pop(y) = 0 and var_pop(x) <> 0, else
// power(corr(y,x), 2)





namespace duckdb {
struct RegrR2State {
	CorrState corr;
	StddevState var_pop_x;
	StddevState var_pop_y;
};

struct RegrR2Operation {
	template <class STATE>
	static void Initialize(STATE *state) {
		CorrOperation::Initialize<CorrState>(&state->corr);
		STDDevBaseOperation::Initialize<StddevState>(&state->var_pop_x);
		STDDevBaseOperation::Initialize<StddevState>(&state->var_pop_y);
	}

	template <class A_TYPE, class B_TYPE, class STATE, class OP>
	static void Operation(STATE *state, AggregateInputData &aggr_input_data, A_TYPE *x_data, B_TYPE *y_data,
	                      ValidityMask &amask, ValidityMask &bmask, idx_t xidx, idx_t yidx) {
		CorrOperation::Operation<A_TYPE, B_TYPE, CorrState, OP>(&state->corr, aggr_input_data, y_data, x_data, bmask,
		                                                        amask, yidx, xidx);
		STDDevBaseOperation::Operation<A_TYPE, StddevState, OP>(&state->var_pop_x, aggr_input_data, y_data, bmask,
		                                                        yidx);
		STDDevBaseOperation::Operation<A_TYPE, StddevState, OP>(&state->var_pop_y, aggr_input_data, x_data, amask,
		                                                        xidx);
	}

	template <class STATE, class OP>
	static void Combine(const STATE &source, STATE *target, AggregateInputData &aggr_input_data) {
		CorrOperation::Combine<CorrState, OP>(source.corr, &target->corr, aggr_input_data);
		STDDevBaseOperation::Combine<StddevState, OP>(source.var_pop_x, &target->var_pop_x, aggr_input_data);
		STDDevBaseOperation::Combine<StddevState, OP>(source.var_pop_y, &target->var_pop_y, aggr_input_data);
	}

	template <class T, class STATE>
	static void Finalize(Vector &result, AggregateInputData &aggr_input_data, STATE *state, T *target,
	                     ValidityMask &mask, idx_t idx) {
		auto var_pop_x = state->var_pop_x.count > 1 ? (state->var_pop_x.dsquared / state->var_pop_x.count) : 0;
		if (!Value::DoubleIsFinite(var_pop_x)) {
			throw OutOfRangeException("VARPOP(X) is out of range!");
		}
		if (var_pop_x == 0) {
			mask.SetInvalid(idx);
			return;
		}
		auto var_pop_y = state->var_pop_y.count > 1 ? (state->var_pop_y.dsquared / state->var_pop_y.count) : 0;
		if (!Value::DoubleIsFinite(var_pop_y)) {
			throw OutOfRangeException("VARPOP(Y) is out of range!");
		}
		if (var_pop_y == 0) {
			target[idx] = 1;
			return;
		}
		CorrOperation::Finalize<T, CorrState>(result, aggr_input_data, &state->corr, target, mask, idx);
		target[idx] = pow(target[idx], 2);
	}

	static bool IgnoreNull() {
		return true;
	}
};

AggregateFunction RegrR2Fun::GetFunction() {
	return AggregateFunction::BinaryAggregate<RegrR2State, double, double, double, RegrR2Operation>(
	    LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE);
}
} // namespace duckdb
// REGR_SLOPE(y, x)
// Returns the slope of the linear regression line for non-null pairs in a group.
// It is computed for non-null pairs using the following formula:
// COVAR_POP(x,y) / VAR_POP(x)

//! Input : Any numeric type
//! Output : Double





namespace duckdb {

AggregateFunction RegrSlopeFun::GetFunction() {
	return AggregateFunction::BinaryAggregate<RegrSlopeState, double, double, double, RegrSlopeOperation>(
	    LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE);
}

} // namespace duckdb
// regr_sxx
// Returns REGR_COUNT(y, x) * VAR_POP(x) for non-null pairs.
// regrsyy
// Returns REGR_COUNT(y, x) * VAR_POP(y) for non-null pairs.





namespace duckdb {

struct RegrSState {
	size_t count;
	StddevState var_pop;
};

struct RegrBaseOperation {
	template <class STATE>
	static void Initialize(STATE *state) {
		RegrCountFunction::Initialize<size_t>(&state->count);
		STDDevBaseOperation::Initialize<StddevState>(&state->var_pop);
	}

	template <class STATE, class OP>
	static void Combine(const STATE &source, STATE *target, AggregateInputData &aggr_input_data) {
		RegrCountFunction::Combine<size_t, OP>(source.count, &target->count, aggr_input_data);
		STDDevBaseOperation::Combine<StddevState, OP>(source.var_pop, &target->var_pop, aggr_input_data);
	}

	template <class T, class STATE>
	static void Finalize(Vector &result, AggregateInputData &aggr_input_data, STATE *state, T *target,
	                     ValidityMask &mask, idx_t idx) {
		if (state->var_pop.count == 0) {
			mask.SetInvalid(idx);
			return;
		}
		auto var_pop = state->var_pop.count > 1 ? (state->var_pop.dsquared / state->var_pop.count) : 0;
		if (!Value::DoubleIsFinite(var_pop)) {
			throw OutOfRangeException("VARPOP is out of range!");
		}
		RegrCountFunction::Finalize<T, size_t>(result, aggr_input_data, &state->count, target, mask, idx);
		target[idx] *= var_pop;
	}

	static bool IgnoreNull() {
		return true;
	}
};

struct RegrSXXOperation : RegrBaseOperation {
	template <class A_TYPE, class B_TYPE, class STATE, class OP>
	static void Operation(STATE *state, AggregateInputData &aggr_input_data, A_TYPE *x_data, B_TYPE *y_data,
	                      ValidityMask &amask, ValidityMask &bmask, idx_t xidx, idx_t yidx) {
		RegrCountFunction::Operation<A_TYPE, B_TYPE, size_t, OP>(&state->count, aggr_input_data, y_data, x_data, bmask,
		                                                         amask, yidx, xidx);
		STDDevBaseOperation::Operation<A_TYPE, StddevState, OP>(&state->var_pop, aggr_input_data, y_data, bmask, yidx);
	}
};

struct RegrSYYOperation : RegrBaseOperation {
	template <class A_TYPE, class B_TYPE, class STATE, class OP>
	static void Operation(STATE *state, AggregateInputData &aggr_input_data, A_TYPE *x_data, B_TYPE *y_data,
	                      ValidityMask &amask, ValidityMask &bmask, idx_t xidx, idx_t yidx) {
		RegrCountFunction::Operation<A_TYPE, B_TYPE, size_t, OP>(&state->count, aggr_input_data, y_data, x_data, bmask,
		                                                         amask, yidx, xidx);
		STDDevBaseOperation::Operation<A_TYPE, StddevState, OP>(&state->var_pop, aggr_input_data, x_data, bmask, xidx);
	}
};

AggregateFunction RegrSXXFun::GetFunction() {
	return AggregateFunction::BinaryAggregate<RegrSState, double, double, double, RegrSXXOperation>(
	    LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE);
}

AggregateFunction RegrSYYFun::GetFunction() {
	return AggregateFunction::BinaryAggregate<RegrSState, double, double, double, RegrSYYOperation>(
	    LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE);
}

} // namespace duckdb
// Returns REGR_COUNT(expr1, expr2) * COVAR_POP(expr1, expr2) for non-null pairs.






namespace duckdb {

struct RegrSXyState {
	size_t count;
	CovarState cov_pop;
};

struct RegrSXYOperation {
	template <class STATE>
	static void Initialize(STATE *state) {
		RegrCountFunction::Initialize<size_t>(&state->count);
		CovarOperation::Initialize<CovarState>(&state->cov_pop);
	}

	template <class A_TYPE, class B_TYPE, class STATE, class OP>
	static void Operation(STATE *state, AggregateInputData &aggr_input_data, A_TYPE *x_data, B_TYPE *y_data,
	                      ValidityMask &amask, ValidityMask &bmask, idx_t xidx, idx_t yidx) {
		RegrCountFunction::Operation<A_TYPE, B_TYPE, size_t, OP>(&state->count, aggr_input_data, y_data, x_data, bmask,
		                                                         amask, yidx, xidx);
		CovarOperation::Operation<A_TYPE, B_TYPE, CovarState, OP>(&state->cov_pop, aggr_input_data, x_data, y_data,
		                                                          amask, bmask, xidx, yidx);
	}

	template <class STATE, class OP>
	static void Combine(const STATE &source, STATE *target, AggregateInputData &aggr_input_data) {
		CovarOperation::Combine<CovarState, OP>(source.cov_pop, &target->cov_pop, aggr_input_data);
		RegrCountFunction::Combine<size_t, OP>(source.count, &target->count, aggr_input_data);
	}

	template <class T, class STATE>
	static void Finalize(Vector &result, AggregateInputData &aggr_input_data, STATE *state, T *target,
	                     ValidityMask &mask, idx_t idx) {
		CovarPopOperation::Finalize<T, CovarState>(result, aggr_input_data, &state->cov_pop, target, mask, idx);
		auto cov_pop = target[idx];
		RegrCountFunction::Finalize<T, size_t>(result, aggr_input_data, &state->count, target, mask, idx);
		target[idx] *= cov_pop;
	}

	static bool IgnoreNull() {
		return true;
	}
};

AggregateFunction RegrSXYFun::GetFunction() {
	return AggregateFunction::BinaryAggregate<RegrSXyState, double, double, double, RegrSXYOperation>(
	    LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE);
}

} // namespace duckdb





namespace duckdb {

template <class T>
void FillExtraInfo(StaticFunctionDefinition &function, T &info) {
	info.internal = true;
	info.description = function.description;
	info.parameter_names = StringUtil::Split(function.parameters, ",");
	info.example = function.example;
}

void CoreFunctions::RegisterFunctions(Catalog &catalog, CatalogTransaction transaction) {
	auto functions = StaticFunctionDefinition::GetFunctionList();
	for (idx_t i = 0; functions[i].name; i++) {
		auto &function = functions[i];
		if (function.get_function || function.get_function_set) {
			// scalar function
			ScalarFunctionSet result;
			if (function.get_function) {
				result.AddFunction(function.get_function());
			} else {
				result = function.get_function_set();
			}
			result.name = function.name;
			CreateScalarFunctionInfo info(result);
			FillExtraInfo(function, info);
			catalog.CreateFunction(transaction, info);
		} else if (function.get_aggregate_function || function.get_aggregate_function_set) {
			// aggregate function
			AggregateFunctionSet result;
			if (function.get_aggregate_function) {
				result.AddFunction(function.get_aggregate_function());
			} else {
				result = function.get_aggregate_function_set();
			}
			result.name = function.name;
			CreateAggregateFunctionInfo info(result);
			FillExtraInfo(function, info);
			catalog.CreateFunction(transaction, info);
		} else {
			throw InternalException("Do not know how to register function of this type");
		}
	}
}

} // namespace duckdb




















namespace duckdb {

// Scalar Function
#define DUCKDB_SCALAR_FUNCTION_BASE(_PARAM, _NAME)                                                                     \
	{ _NAME, _PARAM::Parameters, _PARAM::Description, _PARAM::Example, _PARAM::GetFunction, nullptr, nullptr, nullptr }
#define DUCKDB_SCALAR_FUNCTION(_PARAM)       DUCKDB_SCALAR_FUNCTION_BASE(_PARAM, _PARAM::Name)
#define DUCKDB_SCALAR_FUNCTION_ALIAS(_PARAM) DUCKDB_SCALAR_FUNCTION_BASE(_PARAM::ALIAS, _PARAM::Name)
// Scalar Function Set
#define DUCKDB_SCALAR_FUNCTION_SET_BASE(_PARAM, _NAME)                                                                 \
	{ _NAME, _PARAM::Parameters, _PARAM::Description, _PARAM::Example, nullptr, _PARAM::GetFunctions, nullptr, nullptr }
#define DUCKDB_SCALAR_FUNCTION_SET(_PARAM)       DUCKDB_SCALAR_FUNCTION_SET_BASE(_PARAM, _PARAM::Name)
#define DUCKDB_SCALAR_FUNCTION_SET_ALIAS(_PARAM) DUCKDB_SCALAR_FUNCTION_SET_BASE(_PARAM::ALIAS, _PARAM::Name)
// Aggregate Function
#define DUCKDB_AGGREGATE_FUNCTION_BASE(_PARAM, _NAME)                                                                  \
	{ _NAME, _PARAM::Parameters, _PARAM::Description, _PARAM::Example, nullptr, nullptr, _PARAM::GetFunction, nullptr }
#define DUCKDB_AGGREGATE_FUNCTION(_PARAM)       DUCKDB_AGGREGATE_FUNCTION_BASE(_PARAM, _PARAM::Name)
#define DUCKDB_AGGREGATE_FUNCTION_ALIAS(_PARAM) DUCKDB_AGGREGATE_FUNCTION_BASE(_PARAM::ALIAS, _PARAM::Name)
// Aggregate Function Set
#define DUCKDB_AGGREGATE_FUNCTION_SET_BASE(_PARAM, _NAME)                                                              \
	{ _NAME, _PARAM::Parameters, _PARAM::Description, _PARAM::Example, nullptr, nullptr, nullptr, _PARAM::GetFunctions }
#define DUCKDB_AGGREGATE_FUNCTION_SET(_PARAM)       DUCKDB_AGGREGATE_FUNCTION_SET_BASE(_PARAM, _PARAM::Name)
#define DUCKDB_AGGREGATE_FUNCTION_SET_ALIAS(_PARAM) DUCKDB_AGGREGATE_FUNCTION_SET_BASE(_PARAM::ALIAS, _PARAM::Name)
#define FINAL_FUNCTION                                                                                                 \
	{ nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr }

// this list is generated by scripts/generate_functions.py
static StaticFunctionDefinition internal_functions[] = {
	DUCKDB_SCALAR_FUNCTION(FactorialOperatorFun),
	DUCKDB_SCALAR_FUNCTION_SET(BitwiseAndFun),
	DUCKDB_SCALAR_FUNCTION(PowOperatorFun),
	DUCKDB_SCALAR_FUNCTION_SET(LeftShiftFun),
	DUCKDB_SCALAR_FUNCTION_SET(RightShiftFun),
	DUCKDB_SCALAR_FUNCTION_SET(AbsOperatorFun),
	DUCKDB_SCALAR_FUNCTION_ALIAS(PowOperatorFunAlias),
	DUCKDB_SCALAR_FUNCTION(StartsWithOperatorFun),
	DUCKDB_SCALAR_FUNCTION_SET_ALIAS(AbsFun),
	DUCKDB_SCALAR_FUNCTION(AcosFun),
	DUCKDB_SCALAR_FUNCTION_SET(AgeFun),
	DUCKDB_SCALAR_FUNCTION_ALIAS(AggregateFun),
	DUCKDB_SCALAR_FUNCTION(AliasFun),
	DUCKDB_SCALAR_FUNCTION_ALIAS(ApplyFun),
	DUCKDB_AGGREGATE_FUNCTION_SET(ApproxCountDistinctFun),
	DUCKDB_AGGREGATE_FUNCTION_SET(ApproxQuantileFun),
	DUCKDB_AGGREGATE_FUNCTION_SET(ArgMaxFun),
	DUCKDB_AGGREGATE_FUNCTION_SET(ArgMinFun),
	DUCKDB_AGGREGATE_FUNCTION_SET_ALIAS(ArgmaxFun),
	DUCKDB_AGGREGATE_FUNCTION_SET_ALIAS(ArgminFun),
	DUCKDB_AGGREGATE_FUNCTION_ALIAS(ArrayAggFun),
	DUCKDB_SCALAR_FUNCTION_ALIAS(ArrayAggrFun),
	DUCKDB_SCALAR_FUNCTION_ALIAS(ArrayAggregateFun),
	DUCKDB_SCALAR_FUNCTION_ALIAS(ArrayApplyFun),
	DUCKDB_SCALAR_FUNCTION_ALIAS(ArrayDistinctFun),
	DUCKDB_SCALAR_FUNCTION_ALIAS(ArrayFilterFun),
	DUCKDB_SCALAR_FUNCTION_SET_ALIAS(ArrayReverseSortFun),
	DUCKDB_SCALAR_FUNCTION_ALIAS(ArraySliceFun),
	DUCKDB_SCALAR_FUNCTION_SET_ALIAS(ArraySortFun),
	DUCKDB_SCALAR_FUNCTION_ALIAS(ArrayTransformFun),
	DUCKDB_SCALAR_FUNCTION_ALIAS(ArrayUniqueFun),
	DUCKDB_SCALAR_FUNCTION(ASCIIFun),
	DUCKDB_SCALAR_FUNCTION(AsinFun),
	DUCKDB_SCALAR_FUNCTION(AtanFun),
	DUCKDB_SCALAR_FUNCTION(Atan2Fun),
	DUCKDB_AGGREGATE_FUNCTION_SET(AvgFun),
	DUCKDB_SCALAR_FUNCTION_SET(BarFun),
	DUCKDB_SCALAR_FUNCTION_ALIAS(Base64Fun),
	DUCKDB_SCALAR_FUNCTION_SET(BinFun),
	DUCKDB_AGGREGATE_FUNCTION_SET(BitAndFun),
	DUCKDB_SCALAR_FUNCTION_SET(BitCountFun),
	DUCKDB_AGGREGATE_FUNCTION_SET(BitOrFun),
	DUCKDB_SCALAR_FUNCTION(BitPositionFun),
	DUCKDB_AGGREGATE_FUNCTION_SET(BitXorFun),
	DUCKDB_SCALAR_FUNCTION(BitStringFun),
	DUCKDB_AGGREGATE_FUNCTION_SET(BitstringAggFun),
	DUCKDB_AGGREGATE_FUNCTION(BoolAndFun),
	DUCKDB_AGGREGATE_FUNCTION(BoolOrFun),
	DUCKDB_SCALAR_FUNCTION(CardinalityFun),
	DUCKDB_SCALAR_FUNCTION(CbrtFun),
	DUCKDB_SCALAR_FUNCTION_SET(CeilFun),
	DUCKDB_SCALAR_FUNCTION_SET_ALIAS(CeilingFun),
	DUCKDB_SCALAR_FUNCTION_SET(CenturyFun),
	DUCKDB_SCALAR_FUNCTION(ChrFun),
	DUCKDB_AGGREGATE_FUNCTION(CorrFun),
	DUCKDB_SCALAR_FUNCTION(CosFun),
	DUCKDB_SCALAR_FUNCTION(CotFun),
	DUCKDB_AGGREGATE_FUNCTION(CovarPopFun),
	DUCKDB_AGGREGATE_FUNCTION(CovarSampFun),
	DUCKDB_SCALAR_FUNCTION(CurrentDatabaseFun),
	DUCKDB_SCALAR_FUNCTION(CurrentDateFun),
	DUCKDB_SCALAR_FUNCTION(CurrentQueryFun),
	DUCKDB_SCALAR_FUNCTION(CurrentSchemaFun),
	DUCKDB_SCALAR_FUNCTION(CurrentSchemasFun),
	DUCKDB_SCALAR_FUNCTION(CurrentSettingFun),
	DUCKDB_SCALAR_FUNCTION(DamerauLevenshteinFun),
	DUCKDB_SCALAR_FUNCTION_SET(DateDiffFun),
	DUCKDB_SCALAR_FUNCTION_SET(DatePartFun),
	DUCKDB_SCALAR_FUNCTION_SET(DateSubFun),
	DUCKDB_SCALAR_FUNCTION_SET(DateTruncFun),
	DUCKDB_SCALAR_FUNCTION_SET_ALIAS(DatediffFun),
	DUCKDB_SCALAR_FUNCTION_SET_ALIAS(DatepartFun),
	DUCKDB_SCALAR_FUNCTION_SET_ALIAS(DatesubFun),
	DUCKDB_SCALAR_FUNCTION_SET_ALIAS(DatetruncFun),
	DUCKDB_SCALAR_FUNCTION_SET(DayFun),
	DUCKDB_SCALAR_FUNCTION_SET(DayNameFun),
	DUCKDB_SCALAR_FUNCTION_SET(DayOfMonthFun),
	DUCKDB_SCALAR_FUNCTION_SET(DayOfWeekFun),
	DUCKDB_SCALAR_FUNCTION_SET(DayOfYearFun),
	DUCKDB_SCALAR_FUNCTION_SET(DecadeFun),
	DUCKDB_SCALAR_FUNCTION(DecodeFun),
	DUCKDB_SCALAR_FUNCTION(DegreesFun),
	DUCKDB_SCALAR_FUNCTION_ALIAS(Editdist3Fun),
	DUCKDB_SCALAR_FUNCTION_ALIAS(ElementAtFun),
	DUCKDB_SCALAR_FUNCTION(EncodeFun),
	DUCKDB_AGGREGATE_FUNCTION_SET(EntropyFun),
	DUCKDB_SCALAR_FUNCTION(EnumCodeFun),
	DUCKDB_SCALAR_FUNCTION(EnumFirstFun),
	DUCKDB_SCALAR_FUNCTION(EnumLastFun),
	DUCKDB_SCALAR_FUNCTION(EnumRangeFun),
	DUCKDB_SCALAR_FUNCTION(EnumRangeBoundaryFun),
	DUCKDB_SCALAR_FUNCTION_SET(EpochFun),
	DUCKDB_SCALAR_FUNCTION(EpochMsFun),
	DUCKDB_SCALAR_FUNCTION_SET(EraFun),
	DUCKDB_SCALAR_FUNCTION(ErrorFun),
	DUCKDB_SCALAR_FUNCTION(EvenFun),
	DUCKDB_SCALAR_FUNCTION(ExpFun),
	DUCKDB_SCALAR_FUNCTION_ALIAS(FactorialFun),
	DUCKDB_AGGREGATE_FUNCTION(FAvgFun),
	DUCKDB_SCALAR_FUNCTION_ALIAS(FilterFun),
	DUCKDB_SCALAR_FUNCTION(ListFlattenFun),
	DUCKDB_SCALAR_FUNCTION_SET(FloorFun),
	DUCKDB_SCALAR_FUNCTION(FormatFun),
	DUCKDB_SCALAR_FUNCTION_ALIAS(FormatreadabledecimalsizeFun),
	DUCKDB_SCALAR_FUNCTION(FormatBytesFun),
	DUCKDB_SCALAR_FUNCTION(FromBase64Fun),
	DUCKDB_SCALAR_FUNCTION_ALIAS(FromBinaryFun),
	DUCKDB_SCALAR_FUNCTION_ALIAS(FromHexFun),
	DUCKDB_AGGREGATE_FUNCTION_ALIAS(FsumFun),
	DUCKDB_SCALAR_FUNCTION(GammaFun),
	DUCKDB_SCALAR_FUNCTION_SET_ALIAS(GcdFun),
	DUCKDB_SCALAR_FUNCTION_ALIAS(GenRandomUuidFun),
	DUCKDB_SCALAR_FUNCTION_SET(GenerateSeriesFun),
	DUCKDB_SCALAR_FUNCTION(GetBitFun),
	DUCKDB_SCALAR_FUNCTION(CurrentTimeFun),
	DUCKDB_SCALAR_FUNCTION(GetCurrentTimestampFun),
	DUCKDB_SCALAR_FUNCTION_SET(GreatestFun),
	DUCKDB_SCALAR_FUNCTION_SET(GreatestCommonDivisorFun),
	DUCKDB_AGGREGATE_FUNCTION_SET_ALIAS(GroupConcatFun),
	DUCKDB_SCALAR_FUNCTION(HammingFun),
	DUCKDB_SCALAR_FUNCTION(HashFun),
	DUCKDB_SCALAR_FUNCTION_SET(HexFun),
	DUCKDB_AGGREGATE_FUNCTION_SET(HistogramFun),
	DUCKDB_SCALAR_FUNCTION_SET(HoursFun),
	DUCKDB_SCALAR_FUNCTION(InstrFun),
	DUCKDB_SCALAR_FUNCTION_SET(IsFiniteFun),
	DUCKDB_SCALAR_FUNCTION_SET(IsInfiniteFun),
	DUCKDB_SCALAR_FUNCTION_SET(IsNanFun),
	DUCKDB_SCALAR_FUNCTION_SET(ISODayOfWeekFun),
	DUCKDB_SCALAR_FUNCTION_SET(ISOYearFun),
	DUCKDB_SCALAR_FUNCTION(JaccardFun),
	DUCKDB_SCALAR_FUNCTION(JaroSimilarityFun),
	DUCKDB_SCALAR_FUNCTION(JaroWinklerSimilarityFun),
	DUCKDB_AGGREGATE_FUNCTION(KahanSumFun),
	DUCKDB_AGGREGATE_FUNCTION(KurtosisFun),
	DUCKDB_SCALAR_FUNCTION_SET(LastDayFun),
	DUCKDB_SCALAR_FUNCTION_SET_ALIAS(LcmFun),
	DUCKDB_SCALAR_FUNCTION_SET(LeastFun),
	DUCKDB_SCALAR_FUNCTION_SET(LeastCommonMultipleFun),
	DUCKDB_SCALAR_FUNCTION(LeftFun),
	DUCKDB_SCALAR_FUNCTION(LeftGraphemeFun),
	DUCKDB_SCALAR_FUNCTION(LevenshteinFun),
	DUCKDB_SCALAR_FUNCTION(LogGammaFun),
	DUCKDB_AGGREGATE_FUNCTION(ListFun),
	DUCKDB_SCALAR_FUNCTION_ALIAS(ListAggrFun),
	DUCKDB_SCALAR_FUNCTION(ListAggregateFun),
	DUCKDB_SCALAR_FUNCTION_ALIAS(ListApplyFun),
	DUCKDB_SCALAR_FUNCTION(ListDistinctFun),
	DUCKDB_SCALAR_FUNCTION(ListFilterFun),
	DUCKDB_SCALAR_FUNCTION_ALIAS(ListPackFun),
	DUCKDB_SCALAR_FUNCTION_SET(ListReverseSortFun),
	DUCKDB_SCALAR_FUNCTION(ListSliceFun),
	DUCKDB_SCALAR_FUNCTION_SET(ListSortFun),
	DUCKDB_SCALAR_FUNCTION(ListTransformFun),
	DUCKDB_SCALAR_FUNCTION(ListUniqueFun),
	DUCKDB_SCALAR_FUNCTION(ListValueFun),
	DUCKDB_SCALAR_FUNCTION(LnFun),
	DUCKDB_SCALAR_FUNCTION_ALIAS(LogFun),
	DUCKDB_SCALAR_FUNCTION(Log10Fun),
	DUCKDB_SCALAR_FUNCTION(Log2Fun),
	DUCKDB_SCALAR_FUNCTION(LpadFun),
	DUCKDB_SCALAR_FUNCTION_SET(LtrimFun),
	DUCKDB_AGGREGATE_FUNCTION_SET(MadFun),
	DUCKDB_SCALAR_FUNCTION_SET(MakeDateFun),
	DUCKDB_SCALAR_FUNCTION(MakeTimeFun),
	DUCKDB_SCALAR_FUNCTION(MakeTimestampFun),
	DUCKDB_SCALAR_FUNCTION(MapFun),
	DUCKDB_SCALAR_FUNCTION(MapConcatFun),
	DUCKDB_SCALAR_FUNCTION(MapEntriesFun),
	DUCKDB_SCALAR_FUNCTION(MapExtractFun),
	DUCKDB_SCALAR_FUNCTION(MapFromEntriesFun),
	DUCKDB_SCALAR_FUNCTION(MapKeysFun),
	DUCKDB_SCALAR_FUNCTION(MapValuesFun),
	DUCKDB_AGGREGATE_FUNCTION_SET(MaxFun),
	DUCKDB_AGGREGATE_FUNCTION_SET_ALIAS(MaxByFun),
	DUCKDB_SCALAR_FUNCTION(MD5Fun),
	DUCKDB_SCALAR_FUNCTION(MD5NumberFun),
	DUCKDB_SCALAR_FUNCTION(MD5NumberLowerFun),
	DUCKDB_SCALAR_FUNCTION(MD5NumberUpperFun),
	DUCKDB_AGGREGATE_FUNCTION_SET_ALIAS(MeanFun),
	DUCKDB_AGGREGATE_FUNCTION_SET(MedianFun),
	DUCKDB_SCALAR_FUNCTION_SET(MicrosecondsFun),
	DUCKDB_SCALAR_FUNCTION_SET(MillenniumFun),
	DUCKDB_SCALAR_FUNCTION_SET(MillisecondsFun),
	DUCKDB_AGGREGATE_FUNCTION_SET(MinFun),
	DUCKDB_AGGREGATE_FUNCTION_SET_ALIAS(MinByFun),
	DUCKDB_SCALAR_FUNCTION_SET(MinutesFun),
	DUCKDB_SCALAR_FUNCTION_ALIAS(MismatchesFun),
	DUCKDB_AGGREGATE_FUNCTION_SET(ModeFun),
	DUCKDB_SCALAR_FUNCTION_SET(MonthFun),
	DUCKDB_SCALAR_FUNCTION_SET(MonthNameFun),
	DUCKDB_SCALAR_FUNCTION_SET(NextAfterFun),
	DUCKDB_SCALAR_FUNCTION_ALIAS(NowFun),
	DUCKDB_SCALAR_FUNCTION_ALIAS(OrdFun),
	DUCKDB_SCALAR_FUNCTION(PiFun),
	DUCKDB_SCALAR_FUNCTION_ALIAS(PositionFun),
	DUCKDB_SCALAR_FUNCTION_ALIAS(PowFun),
	DUCKDB_SCALAR_FUNCTION_ALIAS(PowerFun),
	DUCKDB_SCALAR_FUNCTION(PrintfFun),
	DUCKDB_AGGREGATE_FUNCTION(ProductFun),
	DUCKDB_AGGREGATE_FUNCTION_SET_ALIAS(QuantileFun),
	DUCKDB_AGGREGATE_FUNCTION_SET(QuantileContFun),
	DUCKDB_AGGREGATE_FUNCTION_SET(QuantileDiscFun),
	DUCKDB_SCALAR_FUNCTION_SET(QuarterFun),
	DUCKDB_SCALAR_FUNCTION(RadiansFun),
	DUCKDB_SCALAR_FUNCTION(RandomFun),
	DUCKDB_SCALAR_FUNCTION_SET(ListRangeFun),
	DUCKDB_SCALAR_FUNCTION_SET_ALIAS(RegexpSplitToArrayFun),
	DUCKDB_AGGREGATE_FUNCTION(RegrAvgxFun),
	DUCKDB_AGGREGATE_FUNCTION(RegrAvgyFun),
	DUCKDB_AGGREGATE_FUNCTION(RegrCountFun),
	DUCKDB_AGGREGATE_FUNCTION(RegrInterceptFun),
	DUCKDB_AGGREGATE_FUNCTION(RegrR2Fun),
	DUCKDB_AGGREGATE_FUNCTION(RegrSlopeFun),
	DUCKDB_AGGREGATE_FUNCTION(RegrSXXFun),
	DUCKDB_AGGREGATE_FUNCTION(RegrSXYFun),
	DUCKDB_AGGREGATE_FUNCTION(RegrSYYFun),
	DUCKDB_SCALAR_FUNCTION(RepeatFun),
	DUCKDB_SCALAR_FUNCTION(ReplaceFun),
	DUCKDB_AGGREGATE_FUNCTION_SET(ReservoirQuantileFun),
	DUCKDB_SCALAR_FUNCTION(ReverseFun),
	DUCKDB_SCALAR_FUNCTION(RightFun),
	DUCKDB_SCALAR_FUNCTION(RightGraphemeFun),
	DUCKDB_SCALAR_FUNCTION_SET(RoundFun),
	DUCKDB_SCALAR_FUNCTION_ALIAS(RowFun),
	DUCKDB_SCALAR_FUNCTION(RpadFun),
	DUCKDB_SCALAR_FUNCTION_SET(RtrimFun),
	DUCKDB_SCALAR_FUNCTION_SET(SecondsFun),
	DUCKDB_AGGREGATE_FUNCTION(StandardErrorOfTheMeanFun),
	DUCKDB_SCALAR_FUNCTION(SetBitFun),
	DUCKDB_SCALAR_FUNCTION(SetseedFun),
	DUCKDB_SCALAR_FUNCTION_SET(SignFun),
	DUCKDB_SCALAR_FUNCTION_SET(SignBitFun),
	DUCKDB_SCALAR_FUNCTION(SinFun),
	DUCKDB_AGGREGATE_FUNCTION(SkewnessFun),
	DUCKDB_SCALAR_FUNCTION_ALIAS(SplitFun),
	DUCKDB_SCALAR_FUNCTION(SqrtFun),
	DUCKDB_SCALAR_FUNCTION_ALIAS(StartsWithFun),
	DUCKDB_SCALAR_FUNCTION(StatsFun),
	DUCKDB_AGGREGATE_FUNCTION_ALIAS(StddevFun),
	DUCKDB_AGGREGATE_FUNCTION(StdDevPopFun),
	DUCKDB_AGGREGATE_FUNCTION(StdDevSampFun),
	DUCKDB_SCALAR_FUNCTION_ALIAS(StrSplitFun),
	DUCKDB_SCALAR_FUNCTION_SET_ALIAS(StrSplitRegexFun),
	DUCKDB_SCALAR_FUNCTION_SET(StrfTimeFun),
	DUCKDB_AGGREGATE_FUNCTION_SET(StringAggFun),
	DUCKDB_SCALAR_FUNCTION(StringSplitFun),
	DUCKDB_SCALAR_FUNCTION_SET(StringSplitRegexFun),
	DUCKDB_SCALAR_FUNCTION_ALIAS(StringToArrayFun),
	DUCKDB_SCALAR_FUNCTION_ALIAS(StrposFun),
	DUCKDB_SCALAR_FUNCTION_SET(StrpTimeFun),
	DUCKDB_SCALAR_FUNCTION(StructInsertFun),
	DUCKDB_SCALAR_FUNCTION(StructPackFun),
	DUCKDB_AGGREGATE_FUNCTION_SET(SumFun),
	DUCKDB_AGGREGATE_FUNCTION_SET(SumNoOverflowFun),
	DUCKDB_AGGREGATE_FUNCTION_ALIAS(SumkahanFun),
	DUCKDB_SCALAR_FUNCTION(TanFun),
	DUCKDB_SCALAR_FUNCTION_SET(TimeBucketFun),
	DUCKDB_SCALAR_FUNCTION_SET(TimezoneFun),
	DUCKDB_SCALAR_FUNCTION_SET(TimezoneHourFun),
	DUCKDB_SCALAR_FUNCTION_SET(TimezoneMinuteFun),
	DUCKDB_SCALAR_FUNCTION(ToBase64Fun),
	DUCKDB_SCALAR_FUNCTION_SET_ALIAS(ToBinaryFun),
	DUCKDB_SCALAR_FUNCTION(ToDaysFun),
	DUCKDB_SCALAR_FUNCTION_SET_ALIAS(ToHexFun),
	DUCKDB_SCALAR_FUNCTION(ToHoursFun),
	DUCKDB_SCALAR_FUNCTION(ToMicrosecondsFun),
	DUCKDB_SCALAR_FUNCTION(ToMillisecondsFun),
	DUCKDB_SCALAR_FUNCTION(ToMinutesFun),
	DUCKDB_SCALAR_FUNCTION(ToMonthsFun),
	DUCKDB_SCALAR_FUNCTION(ToSecondsFun),
	DUCKDB_SCALAR_FUNCTION(ToTimestampFun),
	DUCKDB_SCALAR_FUNCTION(ToYearsFun),
	DUCKDB_SCALAR_FUNCTION_ALIAS(TodayFun),
	DUCKDB_SCALAR_FUNCTION_ALIAS(TransactionTimestampFun),
	DUCKDB_SCALAR_FUNCTION(TranslateFun),
	DUCKDB_SCALAR_FUNCTION_SET(TrimFun),
	DUCKDB_SCALAR_FUNCTION_SET(TruncFun),
	DUCKDB_SCALAR_FUNCTION_SET(TryStrpTimeFun),
	DUCKDB_SCALAR_FUNCTION(CurrentTransactionIdFun),
	DUCKDB_SCALAR_FUNCTION(TypeOfFun),
	DUCKDB_SCALAR_FUNCTION(UnbinFun),
	DUCKDB_SCALAR_FUNCTION(UnhexFun),
	DUCKDB_SCALAR_FUNCTION(UnicodeFun),
	DUCKDB_SCALAR_FUNCTION(UnionExtractFun),
	DUCKDB_SCALAR_FUNCTION(UnionTagFun),
	DUCKDB_SCALAR_FUNCTION(UnionValueFun),
	DUCKDB_SCALAR_FUNCTION(UUIDFun),
	DUCKDB_AGGREGATE_FUNCTION(VarPopFun),
	DUCKDB_AGGREGATE_FUNCTION(VarSampFun),
	DUCKDB_AGGREGATE_FUNCTION_ALIAS(VarianceFun),
	DUCKDB_SCALAR_FUNCTION(VersionFun),
	DUCKDB_SCALAR_FUNCTION_SET(WeekFun),
	DUCKDB_SCALAR_FUNCTION_SET(WeekDayFun),
	DUCKDB_SCALAR_FUNCTION_SET(WeekOfYearFun),
	DUCKDB_SCALAR_FUNCTION_SET(BitwiseXorFun),
	DUCKDB_SCALAR_FUNCTION_SET(YearFun),
	DUCKDB_SCALAR_FUNCTION_SET(YearWeekFun),
	DUCKDB_SCALAR_FUNCTION_SET(BitwiseOrFun),
	DUCKDB_SCALAR_FUNCTION_SET(BitwiseNotFun),
	FINAL_FUNCTION
};

StaticFunctionDefinition *StaticFunctionDefinition::GetFunctionList() {
	return internal_functions;
}

} // namespace duckdb




namespace duckdb {

//===--------------------------------------------------------------------===//
// BitStringFunction
//===--------------------------------------------------------------------===//
static void BitStringFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	BinaryExecutor::Execute<string_t, int32_t, string_t>(
	    args.data[0], args.data[1], result, args.size(), [&](string_t input, int32_t n) {
		    if (n < 0) {
			    throw InvalidInputException("The bitstring length cannot be negative");
		    }
		    if (idx_t(n) < input.GetSize()) {
			    throw InvalidInputException("Length must be equal or larger than input string");
		    }
		    idx_t len;
		    Bit::TryGetBitStringSize(input, len, nullptr); // string verification

		    len = Bit::ComputeBitstringLen(n);
		    string_t target = StringVector::EmptyString(result, len);
		    Bit::BitString(input, n, target);
		    target.Finalize();
		    return target;
	    });
}

ScalarFunction BitStringFun::GetFunction() {
	return ScalarFunction({LogicalType::VARCHAR, LogicalType::INTEGER}, LogicalType::BIT, BitStringFunction);
}

//===--------------------------------------------------------------------===//
// get_bit
//===--------------------------------------------------------------------===//
struct GetBitOperator {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA input, TB n) {
		if (n < 0 || (idx_t)n > Bit::BitLength(input) - 1) {
			throw OutOfRangeException("bit index %s out of valid range (0..%s)", NumericHelper::ToString(n),
			                          NumericHelper::ToString(Bit::BitLength(input) - 1));
		}
		return Bit::GetBit(input, n);
	}
};

ScalarFunction GetBitFun::GetFunction() {
	return ScalarFunction({LogicalType::BIT, LogicalType::INTEGER}, LogicalType::INTEGER,
	                      ScalarFunction::BinaryFunction<string_t, int32_t, int32_t, GetBitOperator>);
}

//===--------------------------------------------------------------------===//
// set_bit
//===--------------------------------------------------------------------===//
static void SetBitOperation(DataChunk &args, ExpressionState &state, Vector &result) {
	TernaryExecutor::Execute<string_t, int32_t, int32_t, string_t>(
	    args.data[0], args.data[1], args.data[2], result, args.size(),
	    [&](string_t input, int32_t n, int32_t new_value) {
		    if (new_value != 0 && new_value != 1) {
			    throw InvalidInputException("The new bit must be 1 or 0");
		    }
		    if (n < 0 || (idx_t)n > Bit::BitLength(input) - 1) {
			    throw OutOfRangeException("bit index %s out of valid range (0..%s)", NumericHelper::ToString(n),
			                              NumericHelper::ToString(Bit::BitLength(input) - 1));
		    }
		    string_t target = StringVector::EmptyString(result, input.GetSize());
		    memcpy(target.GetDataWriteable(), input.GetData(), input.GetSize());
		    Bit::SetBit(target, n, new_value);
		    return target;
	    });
}

ScalarFunction SetBitFun::GetFunction() {
	return ScalarFunction({LogicalType::BIT, LogicalType::INTEGER, LogicalType::INTEGER}, LogicalType::BIT,
	                      SetBitOperation);
}

//===--------------------------------------------------------------------===//
// bit_position
//===--------------------------------------------------------------------===//
struct BitPositionOperator {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA substring, TB input) {
		if (substring.GetSize() > input.GetSize()) {
			return 0;
		}
		return Bit::BitPosition(substring, input);
	}
};

ScalarFunction BitPositionFun::GetFunction() {
	return ScalarFunction({LogicalType::BIT, LogicalType::BIT}, LogicalType::INTEGER,
	                      ScalarFunction::BinaryFunction<string_t, string_t, int32_t, BitPositionOperator>);
}

} // namespace duckdb



namespace duckdb {

struct Base64EncodeOperator {
	template <class INPUT_TYPE, class RESULT_TYPE>
	static RESULT_TYPE Operation(INPUT_TYPE input, Vector &result) {
		auto result_str = StringVector::EmptyString(result, Blob::ToBase64Size(input));
		Blob::ToBase64(input, result_str.GetDataWriteable());
		result_str.Finalize();
		return result_str;
	}
};

struct Base64DecodeOperator {
	template <class INPUT_TYPE, class RESULT_TYPE>
	static RESULT_TYPE Operation(INPUT_TYPE input, Vector &result) {
		auto result_size = Blob::FromBase64Size(input);
		auto result_blob = StringVector::EmptyString(result, result_size);
		Blob::FromBase64(input, (data_ptr_t)result_blob.GetDataWriteable(), result_size);
		result_blob.Finalize();
		return result_blob;
	}
};

static void Base64EncodeFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	// decode is also a nop cast, but requires verification if the provided string is actually
	UnaryExecutor::ExecuteString<string_t, string_t, Base64EncodeOperator>(args.data[0], result, args.size());
}

static void Base64DecodeFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	// decode is also a nop cast, but requires verification if the provided string is actually
	UnaryExecutor::ExecuteString<string_t, string_t, Base64DecodeOperator>(args.data[0], result, args.size());
}

ScalarFunction ToBase64Fun::GetFunction() {
	return ScalarFunction({LogicalType::BLOB}, LogicalType::VARCHAR, Base64EncodeFunction);
}

ScalarFunction FromBase64Fun::GetFunction() {
	return ScalarFunction({LogicalType::VARCHAR}, LogicalType::BLOB, Base64DecodeFunction);
}

} // namespace duckdb



namespace duckdb {

static void EncodeFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	// encode is essentially a nop cast from varchar to blob
	// we only need to reinterpret the data using the blob type
	result.Reinterpret(args.data[0]);
}

struct BlobDecodeOperator {
	template <class INPUT_TYPE, class RESULT_TYPE>
	static RESULT_TYPE Operation(INPUT_TYPE input) {
		auto input_data = input.GetData();
		auto input_length = input.GetSize();
		if (Utf8Proc::Analyze(input_data, input_length) == UnicodeType::INVALID) {
			throw ConversionException(
			    "Failure in decode: could not convert blob to UTF8 string, the blob contained invalid UTF8 characters");
		}
		return input;
	}
};

static void DecodeFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	// decode is also a nop cast, but requires verification if the provided string is actually
	UnaryExecutor::Execute<string_t, string_t, BlobDecodeOperator>(args.data[0], result, args.size());
	StringVector::AddHeapReference(result, args.data[0]);
}

ScalarFunction EncodeFun::GetFunction() {
	return ScalarFunction({LogicalType::VARCHAR}, LogicalType::BLOB, EncodeFunction);
}

ScalarFunction DecodeFun::GetFunction() {
	return ScalarFunction({LogicalType::BLOB}, LogicalType::VARCHAR, DecodeFunction);
}

} // namespace duckdb








namespace duckdb {

static void AgeFunctionStandard(DataChunk &input, ExpressionState &state, Vector &result) {
	D_ASSERT(input.ColumnCount() == 1);
	auto current_timestamp = Timestamp::GetCurrentTimestamp();

	UnaryExecutor::ExecuteWithNulls<timestamp_t, interval_t>(input.data[0], result, input.size(),
	                                                         [&](timestamp_t input, ValidityMask &mask, idx_t idx) {
		                                                         if (Timestamp::IsFinite(input)) {
			                                                         return Interval::GetAge(current_timestamp, input);
		                                                         } else {
			                                                         mask.SetInvalid(idx);
			                                                         return interval_t();
		                                                         }
	                                                         });
}

static void AgeFunction(DataChunk &input, ExpressionState &state, Vector &result) {
	D_ASSERT(input.ColumnCount() == 2);

	BinaryExecutor::ExecuteWithNulls<timestamp_t, timestamp_t, interval_t>(
	    input.data[0], input.data[1], result, input.size(),
	    [&](timestamp_t input1, timestamp_t input2, ValidityMask &mask, idx_t idx) {
		    if (Timestamp::IsFinite(input1) && Timestamp::IsFinite(input2)) {
			    return Interval::GetAge(input1, input2);
		    } else {
			    mask.SetInvalid(idx);
			    return interval_t();
		    }
	    });
}

ScalarFunctionSet AgeFun::GetFunctions() {
	ScalarFunctionSet age("age");
	age.AddFunction(ScalarFunction({LogicalType::TIMESTAMP}, LogicalType::INTERVAL, AgeFunctionStandard));
	age.AddFunction(
	    ScalarFunction({LogicalType::TIMESTAMP, LogicalType::TIMESTAMP}, LogicalType::INTERVAL, AgeFunction));
	return age;
}

} // namespace duckdb









namespace duckdb {

static timestamp_t GetTransactionTimestamp(ExpressionState &state) {
	return MetaTransaction::Get(state.GetContext()).start_timestamp;
}

static void CurrentTimeFunction(DataChunk &input, ExpressionState &state, Vector &result) {
	D_ASSERT(input.ColumnCount() == 0);
	auto val = Value::TIME(Timestamp::GetTime(GetTransactionTimestamp(state)));
	result.Reference(val);
}

static void CurrentDateFunction(DataChunk &input, ExpressionState &state, Vector &result) {
	D_ASSERT(input.ColumnCount() == 0);

	auto val = Value::DATE(Timestamp::GetDate(GetTransactionTimestamp(state)));
	result.Reference(val);
}

static void CurrentTimestampFunction(DataChunk &input, ExpressionState &state, Vector &result) {
	D_ASSERT(input.ColumnCount() == 0);

	auto val = Value::TIMESTAMPTZ(GetTransactionTimestamp(state));
	result.Reference(val);
}

ScalarFunction CurrentTimeFun::GetFunction() {
	ScalarFunction current_time({}, LogicalType::TIME, CurrentTimeFunction);
	current_time.side_effects = FunctionSideEffects::HAS_SIDE_EFFECTS;
	return current_time;
}

ScalarFunction CurrentDateFun::GetFunction() {
	ScalarFunction current_date({}, LogicalType::DATE, CurrentDateFunction);
	current_date.side_effects = FunctionSideEffects::HAS_SIDE_EFFECTS;
	return current_date;
}

ScalarFunction GetCurrentTimestampFun::GetFunction() {
	ScalarFunction current_timestamp({}, LogicalType::TIMESTAMP_TZ, CurrentTimestampFunction);
	current_timestamp.side_effects = FunctionSideEffects::HAS_SIDE_EFFECTS;
	return current_timestamp;
}

} // namespace duckdb












namespace duckdb {

// This function is an implementation of the "period-crossing" date difference function from T-SQL
// https://docs.microsoft.com/en-us/sql/t-sql/functions/datediff-transact-sql?view=sql-server-ver15
struct DateDiff {
	template <class TA, class TB, class TR, class OP>
	static inline void BinaryExecute(Vector &left, Vector &right, Vector &result, idx_t count) {
		BinaryExecutor::ExecuteWithNulls<TA, TB, TR>(
		    left, right, result, count, [&](TA startdate, TB enddate, ValidityMask &mask, idx_t idx) {
			    if (Value::IsFinite(startdate) && Value::IsFinite(enddate)) {
				    return OP::template Operation<TA, TB, TR>(startdate, enddate);
			    } else {
				    mask.SetInvalid(idx);
				    return TR();
			    }
		    });
	}

	struct YearOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA startdate, TB enddate) {
			return Date::ExtractYear(enddate) - Date::ExtractYear(startdate);
		}
	};

	struct MonthOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA startdate, TB enddate) {
			int32_t start_year, start_month, start_day;
			Date::Convert(startdate, start_year, start_month, start_day);
			int32_t end_year, end_month, end_day;
			Date::Convert(enddate, end_year, end_month, end_day);

			return (end_year * 12 + end_month - 1) - (start_year * 12 + start_month - 1);
		}
	};

	struct DayOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA startdate, TB enddate) {
			return TR(Date::EpochDays(enddate)) - TR(Date::EpochDays(startdate));
		}
	};

	struct DecadeOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA startdate, TB enddate) {
			return Date::ExtractYear(enddate) / 10 - Date::ExtractYear(startdate) / 10;
		}
	};

	struct CenturyOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA startdate, TB enddate) {
			return Date::ExtractYear(enddate) / 100 - Date::ExtractYear(startdate) / 100;
		}
	};

	struct MilleniumOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA startdate, TB enddate) {
			return Date::ExtractYear(enddate) / 1000 - Date::ExtractYear(startdate) / 1000;
		}
	};

	struct QuarterOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA startdate, TB enddate) {
			int32_t start_year, start_month, start_day;
			Date::Convert(startdate, start_year, start_month, start_day);
			int32_t end_year, end_month, end_day;
			Date::Convert(enddate, end_year, end_month, end_day);

			return (end_year * 12 + end_month - 1) / Interval::MONTHS_PER_QUARTER -
			       (start_year * 12 + start_month - 1) / Interval::MONTHS_PER_QUARTER;
		}
	};

	struct WeekOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA startdate, TB enddate) {
			return Date::Epoch(Date::GetMondayOfCurrentWeek(enddate)) / Interval::SECS_PER_WEEK -
			       Date::Epoch(Date::GetMondayOfCurrentWeek(startdate)) / Interval::SECS_PER_WEEK;
		}
	};

	struct ISOYearOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA startdate, TB enddate) {
			return Date::ExtractISOYearNumber(enddate) - Date::ExtractISOYearNumber(startdate);
		}
	};

	struct MicrosecondsOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA startdate, TB enddate) {
			return Date::EpochMicroseconds(enddate) - Date::EpochMicroseconds(startdate);
		}
	};

	struct MillisecondsOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA startdate, TB enddate) {
			return Date::EpochMicroseconds(enddate) / Interval::MICROS_PER_MSEC -
			       Date::EpochMicroseconds(startdate) / Interval::MICROS_PER_MSEC;
		}
	};

	struct SecondsOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA startdate, TB enddate) {
			return Date::Epoch(enddate) - Date::Epoch(startdate);
		}
	};

	struct MinutesOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA startdate, TB enddate) {
			return Date::Epoch(enddate) / Interval::SECS_PER_MINUTE -
			       Date::Epoch(startdate) / Interval::SECS_PER_MINUTE;
		}
	};

	struct HoursOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA startdate, TB enddate) {
			return Date::Epoch(enddate) / Interval::SECS_PER_HOUR - Date::Epoch(startdate) / Interval::SECS_PER_HOUR;
		}
	};
};

// TIMESTAMP specialisations
template <>
int64_t DateDiff::YearOperator::Operation(timestamp_t startdate, timestamp_t enddate) {
	return YearOperator::Operation<date_t, date_t, int64_t>(Timestamp::GetDate(startdate), Timestamp::GetDate(enddate));
}

template <>
int64_t DateDiff::MonthOperator::Operation(timestamp_t startdate, timestamp_t enddate) {
	return MonthOperator::Operation<date_t, date_t, int64_t>(Timestamp::GetDate(startdate),
	                                                         Timestamp::GetDate(enddate));
}

template <>
int64_t DateDiff::DayOperator::Operation(timestamp_t startdate, timestamp_t enddate) {
	return DayOperator::Operation<date_t, date_t, int64_t>(Timestamp::GetDate(startdate), Timestamp::GetDate(enddate));
}

template <>
int64_t DateDiff::DecadeOperator::Operation(timestamp_t startdate, timestamp_t enddate) {
	return DecadeOperator::Operation<date_t, date_t, int64_t>(Timestamp::GetDate(startdate),
	                                                          Timestamp::GetDate(enddate));
}

template <>
int64_t DateDiff::CenturyOperator::Operation(timestamp_t startdate, timestamp_t enddate) {
	return CenturyOperator::Operation<date_t, date_t, int64_t>(Timestamp::GetDate(startdate),
	                                                           Timestamp::GetDate(enddate));
}

template <>
int64_t DateDiff::MilleniumOperator::Operation(timestamp_t startdate, timestamp_t enddate) {
	return MilleniumOperator::Operation<date_t, date_t, int64_t>(Timestamp::GetDate(startdate),
	                                                             Timestamp::GetDate(enddate));
}

template <>
int64_t DateDiff::QuarterOperator::Operation(timestamp_t startdate, timestamp_t enddate) {
	return QuarterOperator::Operation<date_t, date_t, int64_t>(Timestamp::GetDate(startdate),
	                                                           Timestamp::GetDate(enddate));
}

template <>
int64_t DateDiff::WeekOperator::Operation(timestamp_t startdate, timestamp_t enddate) {
	return WeekOperator::Operation<date_t, date_t, int64_t>(Timestamp::GetDate(startdate), Timestamp::GetDate(enddate));
}

template <>
int64_t DateDiff::ISOYearOperator::Operation(timestamp_t startdate, timestamp_t enddate) {
	return ISOYearOperator::Operation<date_t, date_t, int64_t>(Timestamp::GetDate(startdate),
	                                                           Timestamp::GetDate(enddate));
}

template <>
int64_t DateDiff::MicrosecondsOperator::Operation(timestamp_t startdate, timestamp_t enddate) {
	const auto start = Timestamp::GetEpochMicroSeconds(startdate);
	const auto end = Timestamp::GetEpochMicroSeconds(enddate);
	return SubtractOperatorOverflowCheck::Operation<int64_t, int64_t, int64_t>(end, start);
}

template <>
int64_t DateDiff::MillisecondsOperator::Operation(timestamp_t startdate, timestamp_t enddate) {
	return Timestamp::GetEpochMs(enddate) - Timestamp::GetEpochMs(startdate);
}

template <>
int64_t DateDiff::SecondsOperator::Operation(timestamp_t startdate, timestamp_t enddate) {
	return Timestamp::GetEpochSeconds(enddate) - Timestamp::GetEpochSeconds(startdate);
}

template <>
int64_t DateDiff::MinutesOperator::Operation(timestamp_t startdate, timestamp_t enddate) {
	return Timestamp::GetEpochSeconds(enddate) / Interval::SECS_PER_MINUTE -
	       Timestamp::GetEpochSeconds(startdate) / Interval::SECS_PER_MINUTE;
}

template <>
int64_t DateDiff::HoursOperator::Operation(timestamp_t startdate, timestamp_t enddate) {
	return Timestamp::GetEpochSeconds(enddate) / Interval::SECS_PER_HOUR -
	       Timestamp::GetEpochSeconds(startdate) / Interval::SECS_PER_HOUR;
}

// TIME specialisations
template <>
int64_t DateDiff::YearOperator::Operation(dtime_t startdate, dtime_t enddate) {
	throw NotImplementedException("\"time\" units \"year\" not recognized");
}

template <>
int64_t DateDiff::MonthOperator::Operation(dtime_t startdate, dtime_t enddate) {
	throw NotImplementedException("\"time\" units \"month\" not recognized");
}

template <>
int64_t DateDiff::DayOperator::Operation(dtime_t startdate, dtime_t enddate) {
	throw NotImplementedException("\"time\" units \"day\" not recognized");
}

template <>
int64_t DateDiff::DecadeOperator::Operation(dtime_t startdate, dtime_t enddate) {
	throw NotImplementedException("\"time\" units \"decade\" not recognized");
}

template <>
int64_t DateDiff::CenturyOperator::Operation(dtime_t startdate, dtime_t enddate) {
	throw NotImplementedException("\"time\" units \"century\" not recognized");
}

template <>
int64_t DateDiff::MilleniumOperator::Operation(dtime_t startdate, dtime_t enddate) {
	throw NotImplementedException("\"time\" units \"millennium\" not recognized");
}

template <>
int64_t DateDiff::QuarterOperator::Operation(dtime_t startdate, dtime_t enddate) {
	throw NotImplementedException("\"time\" units \"quarter\" not recognized");
}

template <>
int64_t DateDiff::WeekOperator::Operation(dtime_t startdate, dtime_t enddate) {
	throw NotImplementedException("\"time\" units \"week\" not recognized");
}

template <>
int64_t DateDiff::ISOYearOperator::Operation(dtime_t startdate, dtime_t enddate) {
	throw NotImplementedException("\"time\" units \"isoyear\" not recognized");
}

template <>
int64_t DateDiff::MicrosecondsOperator::Operation(dtime_t startdate, dtime_t enddate) {
	return enddate.micros - startdate.micros;
}

template <>
int64_t DateDiff::MillisecondsOperator::Operation(dtime_t startdate, dtime_t enddate) {
	return enddate.micros / Interval::MICROS_PER_MSEC - startdate.micros / Interval::MICROS_PER_MSEC;
}

template <>
int64_t DateDiff::SecondsOperator::Operation(dtime_t startdate, dtime_t enddate) {
	return enddate.micros / Interval::MICROS_PER_SEC - startdate.micros / Interval::MICROS_PER_SEC;
}

template <>
int64_t DateDiff::MinutesOperator::Operation(dtime_t startdate, dtime_t enddate) {
	return enddate.micros / Interval::MICROS_PER_MINUTE - startdate.micros / Interval::MICROS_PER_MINUTE;
}

template <>
int64_t DateDiff::HoursOperator::Operation(dtime_t startdate, dtime_t enddate) {
	return enddate.micros / Interval::MICROS_PER_HOUR - startdate.micros / Interval::MICROS_PER_HOUR;
}

template <typename TA, typename TB, typename TR>
static int64_t DifferenceDates(DatePartSpecifier type, TA startdate, TB enddate) {
	switch (type) {
	case DatePartSpecifier::YEAR:
		return DateDiff::YearOperator::template Operation<TA, TB, TR>(startdate, enddate);
	case DatePartSpecifier::MONTH:
		return DateDiff::MonthOperator::template Operation<TA, TB, TR>(startdate, enddate);
	case DatePartSpecifier::DAY:
	case DatePartSpecifier::DOW:
	case DatePartSpecifier::ISODOW:
	case DatePartSpecifier::DOY:
		return DateDiff::DayOperator::template Operation<TA, TB, TR>(startdate, enddate);
	case DatePartSpecifier::DECADE:
		return DateDiff::DecadeOperator::template Operation<TA, TB, TR>(startdate, enddate);
	case DatePartSpecifier::CENTURY:
		return DateDiff::CenturyOperator::template Operation<TA, TB, TR>(startdate, enddate);
	case DatePartSpecifier::MILLENNIUM:
		return DateDiff::MilleniumOperator::template Operation<TA, TB, TR>(startdate, enddate);
	case DatePartSpecifier::QUARTER:
		return DateDiff::QuarterOperator::template Operation<TA, TB, TR>(startdate, enddate);
	case DatePartSpecifier::WEEK:
	case DatePartSpecifier::YEARWEEK:
		return DateDiff::WeekOperator::template Operation<TA, TB, TR>(startdate, enddate);
	case DatePartSpecifier::ISOYEAR:
		return DateDiff::ISOYearOperator::template Operation<TA, TB, TR>(startdate, enddate);
	case DatePartSpecifier::MICROSECONDS:
		return DateDiff::MicrosecondsOperator::template Operation<TA, TB, TR>(startdate, enddate);
	case DatePartSpecifier::MILLISECONDS:
		return DateDiff::MillisecondsOperator::template Operation<TA, TB, TR>(startdate, enddate);
	case DatePartSpecifier::SECOND:
	case DatePartSpecifier::EPOCH:
		return DateDiff::SecondsOperator::template Operation<TA, TB, TR>(startdate, enddate);
	case DatePartSpecifier::MINUTE:
		return DateDiff::MinutesOperator::template Operation<TA, TB, TR>(startdate, enddate);
	case DatePartSpecifier::HOUR:
		return DateDiff::HoursOperator::template Operation<TA, TB, TR>(startdate, enddate);
	default:
		throw NotImplementedException("Specifier type not implemented for DATEDIFF");
	}
}

struct DateDiffTernaryOperator {
	template <typename TS, typename TA, typename TB, typename TR>
	static inline TR Operation(TS part, TA startdate, TB enddate, ValidityMask &mask, idx_t idx) {
		if (Value::IsFinite(startdate) && Value::IsFinite(enddate)) {
			return DifferenceDates<TA, TB, TR>(GetDatePartSpecifier(part.GetString()), startdate, enddate);
		} else {
			mask.SetInvalid(idx);
			return TR();
		}
	}
};

template <typename TA, typename TB, typename TR>
static void DateDiffBinaryExecutor(DatePartSpecifier type, Vector &left, Vector &right, Vector &result, idx_t count) {
	switch (type) {
	case DatePartSpecifier::YEAR:
		DateDiff::BinaryExecute<TA, TB, TR, DateDiff::YearOperator>(left, right, result, count);
		break;
	case DatePartSpecifier::MONTH:
		DateDiff::BinaryExecute<TA, TB, TR, DateDiff::MonthOperator>(left, right, result, count);
		break;
	case DatePartSpecifier::DAY:
	case DatePartSpecifier::DOW:
	case DatePartSpecifier::ISODOW:
	case DatePartSpecifier::DOY:
		DateDiff::BinaryExecute<TA, TB, TR, DateDiff::DayOperator>(left, right, result, count);
		break;
	case DatePartSpecifier::DECADE:
		DateDiff::BinaryExecute<TA, TB, TR, DateDiff::DecadeOperator>(left, right, result, count);
		break;
	case DatePartSpecifier::CENTURY:
		DateDiff::BinaryExecute<TA, TB, TR, DateDiff::CenturyOperator>(left, right, result, count);
		break;
	case DatePartSpecifier::MILLENNIUM:
		DateDiff::BinaryExecute<TA, TB, TR, DateDiff::MilleniumOperator>(left, right, result, count);
		break;
	case DatePartSpecifier::QUARTER:
		DateDiff::BinaryExecute<TA, TB, TR, DateDiff::QuarterOperator>(left, right, result, count);
		break;
	case DatePartSpecifier::WEEK:
	case DatePartSpecifier::YEARWEEK:
		DateDiff::BinaryExecute<TA, TB, TR, DateDiff::WeekOperator>(left, right, result, count);
		break;
	case DatePartSpecifier::ISOYEAR:
		DateDiff::BinaryExecute<TA, TB, TR, DateDiff::ISOYearOperator>(left, right, result, count);
		break;
	case DatePartSpecifier::MICROSECONDS:
		DateDiff::BinaryExecute<TA, TB, TR, DateDiff::MicrosecondsOperator>(left, right, result, count);
		break;
	case DatePartSpecifier::MILLISECONDS:
		DateDiff::BinaryExecute<TA, TB, TR, DateDiff::MillisecondsOperator>(left, right, result, count);
		break;
	case DatePartSpecifier::SECOND:
	case DatePartSpecifier::EPOCH:
		DateDiff::BinaryExecute<TA, TB, TR, DateDiff::SecondsOperator>(left, right, result, count);
		break;
	case DatePartSpecifier::MINUTE:
		DateDiff::BinaryExecute<TA, TB, TR, DateDiff::MinutesOperator>(left, right, result, count);
		break;
	case DatePartSpecifier::HOUR:
		DateDiff::BinaryExecute<TA, TB, TR, DateDiff::HoursOperator>(left, right, result, count);
		break;
	default:
		throw NotImplementedException("Specifier type not implemented for DATEDIFF");
	}
}

template <typename T>
static void DateDiffFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(args.ColumnCount() == 3);
	auto &part_arg = args.data[0];
	auto &start_arg = args.data[1];
	auto &end_arg = args.data[2];

	if (part_arg.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		// Common case of constant part.
		if (ConstantVector::IsNull(part_arg)) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
			ConstantVector::SetNull(result, true);
		} else {
			const auto type = GetDatePartSpecifier(ConstantVector::GetData<string_t>(part_arg)->GetString());
			DateDiffBinaryExecutor<T, T, int64_t>(type, start_arg, end_arg, result, args.size());
		}
	} else {
		TernaryExecutor::ExecuteWithNulls<string_t, T, T, int64_t>(
		    part_arg, start_arg, end_arg, result, args.size(),
		    DateDiffTernaryOperator::Operation<string_t, T, T, int64_t>);
	}
}

ScalarFunctionSet DateDiffFun::GetFunctions() {
	ScalarFunctionSet date_diff("date_diff");
	date_diff.AddFunction(ScalarFunction({LogicalType::VARCHAR, LogicalType::DATE, LogicalType::DATE},
	                                     LogicalType::BIGINT, DateDiffFunction<date_t>));
	date_diff.AddFunction(ScalarFunction({LogicalType::VARCHAR, LogicalType::TIMESTAMP, LogicalType::TIMESTAMP},
	                                     LogicalType::BIGINT, DateDiffFunction<timestamp_t>));
	date_diff.AddFunction(ScalarFunction({LogicalType::VARCHAR, LogicalType::TIME, LogicalType::TIME},
	                                     LogicalType::BIGINT, DateDiffFunction<dtime_t>));
	return date_diff;
}

} // namespace duckdb















namespace duckdb {

DatePartSpecifier GetDateTypePartSpecifier(const string &specifier, LogicalType &type) {
	const auto part = GetDatePartSpecifier(specifier);
	switch (type.id()) {
	case LogicalType::TIMESTAMP:
	case LogicalType::TIMESTAMP_TZ:
		return part;
	case LogicalType::DATE:
		switch (part) {
		case DatePartSpecifier::YEAR:
		case DatePartSpecifier::MONTH:
		case DatePartSpecifier::DAY:
		case DatePartSpecifier::DECADE:
		case DatePartSpecifier::CENTURY:
		case DatePartSpecifier::MILLENNIUM:
		case DatePartSpecifier::DOW:
		case DatePartSpecifier::ISODOW:
		case DatePartSpecifier::ISOYEAR:
		case DatePartSpecifier::WEEK:
		case DatePartSpecifier::QUARTER:
		case DatePartSpecifier::DOY:
		case DatePartSpecifier::YEARWEEK:
		case DatePartSpecifier::ERA:
			return part;
		default:
			break;
		}
		break;
	case LogicalType::TIME:
		switch (part) {
		case DatePartSpecifier::MICROSECONDS:
		case DatePartSpecifier::MILLISECONDS:
		case DatePartSpecifier::SECOND:
		case DatePartSpecifier::MINUTE:
		case DatePartSpecifier::HOUR:
		case DatePartSpecifier::EPOCH:
		case DatePartSpecifier::TIMEZONE:
		case DatePartSpecifier::TIMEZONE_HOUR:
		case DatePartSpecifier::TIMEZONE_MINUTE:
			return part;
		default:
			break;
		}
		break;
	case LogicalType::INTERVAL:
		switch (part) {
		case DatePartSpecifier::YEAR:
		case DatePartSpecifier::MONTH:
		case DatePartSpecifier::DAY:
		case DatePartSpecifier::DECADE:
		case DatePartSpecifier::CENTURY:
		case DatePartSpecifier::QUARTER:
		case DatePartSpecifier::MILLENNIUM:
		case DatePartSpecifier::MICROSECONDS:
		case DatePartSpecifier::MILLISECONDS:
		case DatePartSpecifier::SECOND:
		case DatePartSpecifier::MINUTE:
		case DatePartSpecifier::HOUR:
		case DatePartSpecifier::EPOCH:
			return part;
		default:
			break;
		}
		break;
	default:
		break;
	}

	throw NotImplementedException("\"%s\" units \"%s\" not recognized", EnumUtil::ToString(type.id()), specifier);
}

template <int64_t MIN, int64_t MAX>
static unique_ptr<BaseStatistics> PropagateSimpleDatePartStatistics(vector<BaseStatistics> &child_stats) {
	// we can always propagate simple date part statistics
	// since the min and max can never exceed these bounds
	auto result = NumericStats::CreateEmpty(LogicalType::BIGINT);
	result.CopyValidity(child_stats[0]);
	NumericStats::SetMin(result, Value::BIGINT(MIN));
	NumericStats::SetMax(result, Value::BIGINT(MAX));
	return result.ToUnique();
}

struct DatePart {
	template <class T, class OP>
	static unique_ptr<BaseStatistics> PropagateDatePartStatistics(vector<BaseStatistics> &child_stats) {
		// we can only propagate complex date part stats if the child has stats
		auto &nstats = child_stats[0];
		if (!NumericStats::HasMinMax(nstats)) {
			return nullptr;
		}
		// run the operator on both the min and the max, this gives us the [min, max] bound
		auto min = NumericStats::GetMin<T>(nstats);
		auto max = NumericStats::GetMax<T>(nstats);
		if (min > max) {
			return nullptr;
		}
		// Infinities prevent us from computing generic ranges
		if (!Value::IsFinite(min) || !Value::IsFinite(max)) {
			return nullptr;
		}
		auto min_part = OP::template Operation<T, int64_t>(min);
		auto max_part = OP::template Operation<T, int64_t>(max);
		auto result = NumericStats::CreateEmpty(LogicalType::BIGINT);
		NumericStats::SetMin(result, Value::BIGINT(min_part));
		NumericStats::SetMax(result, Value::BIGINT(max_part));
		result.CopyValidity(child_stats[0]);
		return result.ToUnique();
	}

	template <typename OP>
	struct PartOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input, ValidityMask &mask, idx_t idx, void *dataptr) {
			if (Value::IsFinite(input)) {
				return OP::template Operation<TA, TR>(input);
			} else {
				mask.SetInvalid(idx);
				return TR();
			}
		}
	};

	template <class TA, class TR, class OP>
	static void UnaryFunction(DataChunk &input, ExpressionState &state, Vector &result) {
		D_ASSERT(input.ColumnCount() >= 1);
		using IOP = PartOperator<OP>;
		UnaryExecutor::GenericExecute<TA, TR, IOP>(input.data[0], result, input.size(), nullptr, true);
	}

	struct YearOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			return Date::ExtractYear(input);
		}

		template <class T>
		static unique_ptr<BaseStatistics> PropagateStatistics(ClientContext &context, FunctionStatisticsInput &input) {
			return PropagateDatePartStatistics<T, YearOperator>(input.child_stats);
		}
	};

	struct MonthOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			return Date::ExtractMonth(input);
		}

		template <class T>
		static unique_ptr<BaseStatistics> PropagateStatistics(ClientContext &context, FunctionStatisticsInput &input) {
			// min/max of month operator is [1, 12]
			return PropagateSimpleDatePartStatistics<1, 12>(input.child_stats);
		}
	};

	struct DayOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			return Date::ExtractDay(input);
		}

		template <class T>
		static unique_ptr<BaseStatistics> PropagateStatistics(ClientContext &context, FunctionStatisticsInput &input) {
			// min/max of day operator is [1, 31]
			return PropagateSimpleDatePartStatistics<1, 31>(input.child_stats);
		}
	};

	struct DecadeOperator {
		// From the PG docs: "The year field divided by 10"
		template <typename TR>
		static inline TR DecadeFromYear(TR yyyy) {
			return yyyy / 10;
		}

		template <class TA, class TR>
		static inline TR Operation(TA input) {
			return DecadeFromYear(YearOperator::Operation<TA, TR>(input));
		}

		template <class T>
		static unique_ptr<BaseStatistics> PropagateStatistics(ClientContext &context, FunctionStatisticsInput &input) {
			return PropagateDatePartStatistics<T, DecadeOperator>(input.child_stats);
		}
	};

	struct CenturyOperator {
		// From the PG docs:
		// "The first century starts at 0001-01-01 00:00:00 AD, although they did not know it at the time.
		// This definition applies to all Gregorian calendar countries.
		// There is no century number 0, you go from -1 century to 1 century.
		// If you disagree with this, please write your complaint to: Pope, Cathedral Saint-Peter of Roma, Vatican."
		// (To be fair, His Holiness had nothing to do with this -
		// it was the lack of zero in the counting systems of the time...)
		template <typename TR>
		static inline TR CenturyFromYear(TR yyyy) {
			if (yyyy > 0) {
				return ((yyyy - 1) / 100) + 1;
			} else {
				return (yyyy / 100) - 1;
			}
		}

		template <class TA, class TR>
		static inline TR Operation(TA input) {
			return CenturyFromYear(YearOperator::Operation<TA, TR>(input));
		}

		template <class T>
		static unique_ptr<BaseStatistics> PropagateStatistics(ClientContext &context, FunctionStatisticsInput &input) {
			return PropagateDatePartStatistics<T, CenturyOperator>(input.child_stats);
		}
	};

	struct MillenniumOperator {
		// See the century comment
		template <typename TR>
		static inline TR MillenniumFromYear(TR yyyy) {
			if (yyyy > 0) {
				return ((yyyy - 1) / 1000) + 1;
			} else {
				return (yyyy / 1000) - 1;
			}
		}

		template <class TA, class TR>
		static inline TR Operation(TA input) {
			return MillenniumFromYear<TR>(YearOperator::Operation<TA, TR>(input));
		}

		template <class T>
		static unique_ptr<BaseStatistics> PropagateStatistics(ClientContext &context, FunctionStatisticsInput &input) {
			return PropagateDatePartStatistics<T, MillenniumOperator>(input.child_stats);
		}
	};

	struct QuarterOperator {
		template <class TR>
		static inline TR QuarterFromMonth(TR mm) {
			return (mm - 1) / Interval::MONTHS_PER_QUARTER + 1;
		}

		template <class TA, class TR>
		static inline TR Operation(TA input) {
			return QuarterFromMonth(Date::ExtractMonth(input));
		}

		template <class T>
		static unique_ptr<BaseStatistics> PropagateStatistics(ClientContext &context, FunctionStatisticsInput &input) {
			// min/max of quarter operator is [1, 4]
			return PropagateSimpleDatePartStatistics<1, 4>(input.child_stats);
		}
	};

	struct DayOfWeekOperator {
		template <class TR>
		static inline TR DayOfWeekFromISO(TR isodow) {
			// day of the week (Sunday = 0, Saturday = 6)
			// turn sunday into 0 by doing mod 7
			return isodow % 7;
		}

		template <class TA, class TR>
		static inline TR Operation(TA input) {
			return DayOfWeekFromISO(Date::ExtractISODayOfTheWeek(input));
		}

		template <class T>
		static unique_ptr<BaseStatistics> PropagateStatistics(ClientContext &context, FunctionStatisticsInput &input) {
			return PropagateSimpleDatePartStatistics<0, 6>(input.child_stats);
		}
	};

	struct ISODayOfWeekOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			// isodow (Monday = 1, Sunday = 7)
			return Date::ExtractISODayOfTheWeek(input);
		}

		template <class T>
		static unique_ptr<BaseStatistics> PropagateStatistics(ClientContext &context, FunctionStatisticsInput &input) {
			return PropagateSimpleDatePartStatistics<1, 7>(input.child_stats);
		}
	};

	struct DayOfYearOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			return Date::ExtractDayOfTheYear(input);
		}

		template <class T>
		static unique_ptr<BaseStatistics> PropagateStatistics(ClientContext &context, FunctionStatisticsInput &input) {
			return PropagateSimpleDatePartStatistics<1, 366>(input.child_stats);
		}
	};

	struct WeekOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			return Date::ExtractISOWeekNumber(input);
		}

		template <class T>
		static unique_ptr<BaseStatistics> PropagateStatistics(ClientContext &context, FunctionStatisticsInput &input) {
			return PropagateSimpleDatePartStatistics<1, 54>(input.child_stats);
		}
	};

	struct ISOYearOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			return Date::ExtractISOYearNumber(input);
		}

		template <class T>
		static unique_ptr<BaseStatistics> PropagateStatistics(ClientContext &context, FunctionStatisticsInput &input) {
			return PropagateDatePartStatistics<T, ISOYearOperator>(input.child_stats);
		}
	};

	struct YearWeekOperator {
		template <class TR>
		static inline TR YearWeekFromParts(TR yyyy, TR ww) {
			return yyyy * 100 + ((yyyy > 0) ? ww : -ww);
		}

		template <class TA, class TR>
		static inline TR Operation(TA input) {
			int32_t yyyy, ww;
			Date::ExtractISOYearWeek(input, yyyy, ww);
			return YearWeekFromParts(yyyy, ww);
		}

		template <class T>
		static unique_ptr<BaseStatistics> PropagateStatistics(ClientContext &context, FunctionStatisticsInput &input) {
			return PropagateDatePartStatistics<T, YearWeekOperator>(input.child_stats);
		}
	};

	struct MicrosecondsOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			return 0;
		}

		template <class T>
		static unique_ptr<BaseStatistics> PropagateStatistics(ClientContext &context, FunctionStatisticsInput &input) {
			return PropagateSimpleDatePartStatistics<0, 60000000>(input.child_stats);
		}
	};

	struct MillisecondsOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			return 0;
		}

		template <class T>
		static unique_ptr<BaseStatistics> PropagateStatistics(ClientContext &context, FunctionStatisticsInput &input) {
			return PropagateSimpleDatePartStatistics<0, 60000>(input.child_stats);
		}
	};

	struct SecondsOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			return 0;
		}

		template <class T>
		static unique_ptr<BaseStatistics> PropagateStatistics(ClientContext &context, FunctionStatisticsInput &input) {
			return PropagateSimpleDatePartStatistics<0, 60>(input.child_stats);
		}
	};

	struct MinutesOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			return 0;
		}

		template <class T>
		static unique_ptr<BaseStatistics> PropagateStatistics(ClientContext &context, FunctionStatisticsInput &input) {
			return PropagateSimpleDatePartStatistics<0, 60>(input.child_stats);
		}
	};

	struct HoursOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			return 0;
		}

		template <class T>
		static unique_ptr<BaseStatistics> PropagateStatistics(ClientContext &context, FunctionStatisticsInput &input) {
			return PropagateSimpleDatePartStatistics<0, 24>(input.child_stats);
		}
	};

	struct EpochOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			return Date::Epoch(input);
		}

		template <class T>
		static unique_ptr<BaseStatistics> PropagateStatistics(ClientContext &context, FunctionStatisticsInput &input) {
			return PropagateDatePartStatistics<T, EpochOperator>(input.child_stats);
		}
	};

	struct EraOperator {
		template <class TR>
		static inline TR EraFromYear(TR yyyy) {
			return yyyy > 0 ? 1 : 0;
		}

		template <class TA, class TR>
		static inline TR Operation(TA input) {
			return EraFromYear(Date::ExtractYear(input));
		}

		template <class T>
		static unique_ptr<BaseStatistics> PropagateStatistics(ClientContext &context, FunctionStatisticsInput &input) {
			return PropagateSimpleDatePartStatistics<0, 1>(input.child_stats);
		}
	};

	struct TimezoneOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			// Regular timestamps are UTC.
			return 0;
		}

		template <class T>
		static unique_ptr<BaseStatistics> PropagateStatistics(ClientContext &context, FunctionStatisticsInput &input) {
			return PropagateSimpleDatePartStatistics<0, 0>(input.child_stats);
		}
	};

	// These are all zero and have the same restrictions
	using TimezoneHourOperator = TimezoneOperator;
	using TimezoneMinuteOperator = TimezoneOperator;

	struct StructOperator {
		using part_codes_t = vector<DatePartSpecifier>;
		using part_mask_t = uint64_t;

		enum MaskBits : uint8_t {
			YMD = 1 << 0,
			DOW = 1 << 1,
			DOY = 1 << 2,
			EPOCH = 1 << 3,
			TIME = 1 << 4,
			ZONE = 1 << 5,
			ISO = 1 << 6
		};

		static part_mask_t GetMask(const part_codes_t &part_codes) {
			part_mask_t mask = 0;
			for (const auto &part_code : part_codes) {
				switch (part_code) {
				case DatePartSpecifier::YEAR:
				case DatePartSpecifier::MONTH:
				case DatePartSpecifier::DAY:
				case DatePartSpecifier::DECADE:
				case DatePartSpecifier::CENTURY:
				case DatePartSpecifier::MILLENNIUM:
				case DatePartSpecifier::QUARTER:
				case DatePartSpecifier::ERA:
					mask |= YMD;
					break;
				case DatePartSpecifier::YEARWEEK:
				case DatePartSpecifier::WEEK:
				case DatePartSpecifier::ISOYEAR:
					mask |= ISO;
					break;
				case DatePartSpecifier::DOW:
				case DatePartSpecifier::ISODOW:
					mask |= DOW;
					break;
				case DatePartSpecifier::DOY:
					mask |= DOY;
					break;
				case DatePartSpecifier::EPOCH:
					mask |= EPOCH;
					break;
				case DatePartSpecifier::MICROSECONDS:
				case DatePartSpecifier::MILLISECONDS:
				case DatePartSpecifier::SECOND:
				case DatePartSpecifier::MINUTE:
				case DatePartSpecifier::HOUR:
					mask |= TIME;
					break;
				case DatePartSpecifier::TIMEZONE:
				case DatePartSpecifier::TIMEZONE_HOUR:
				case DatePartSpecifier::TIMEZONE_MINUTE:
					mask |= ZONE;
					break;
				}
			}
			return mask;
		}

		template <typename P>
		static inline P HasPartValue(P *part_values, DatePartSpecifier part) {
			return part_values[int(part)];
		}

		template <class TA, class TR>
		static inline void Operation(TR **part_values, const TA &input, const idx_t idx, const part_mask_t mask) {
			TR *part_data;
			// YMD calculations
			int32_t yyyy = 1970;
			int32_t mm = 0;
			int32_t dd = 1;
			if (mask & YMD) {
				Date::Convert(input, yyyy, mm, dd);
				if ((part_data = HasPartValue(part_values, DatePartSpecifier::YEAR))) {
					part_data[idx] = yyyy;
				}
				if ((part_data = HasPartValue(part_values, DatePartSpecifier::MONTH))) {
					part_data[idx] = mm;
				}
				if ((part_data = HasPartValue(part_values, DatePartSpecifier::DAY))) {
					part_data[idx] = dd;
				}
				if ((part_data = HasPartValue(part_values, DatePartSpecifier::DECADE))) {
					part_data[idx] = DecadeOperator::DecadeFromYear(yyyy);
				}
				if ((part_data = HasPartValue(part_values, DatePartSpecifier::CENTURY))) {
					part_data[idx] = CenturyOperator::CenturyFromYear(yyyy);
				}
				if ((part_data = HasPartValue(part_values, DatePartSpecifier::MILLENNIUM))) {
					part_data[idx] = MillenniumOperator::MillenniumFromYear(yyyy);
				}
				if ((part_data = HasPartValue(part_values, DatePartSpecifier::QUARTER))) {
					part_data[idx] = QuarterOperator::QuarterFromMonth(mm);
				}
				if ((part_data = HasPartValue(part_values, DatePartSpecifier::ERA))) {
					part_data[idx] = EraOperator::EraFromYear(yyyy);
				}
			}

			// Week calculations
			if (mask & DOW) {
				auto isodow = Date::ExtractISODayOfTheWeek(input);
				if ((part_data = HasPartValue(part_values, DatePartSpecifier::DOW))) {
					part_data[idx] = DayOfWeekOperator::DayOfWeekFromISO(isodow);
				}
				if ((part_data = HasPartValue(part_values, DatePartSpecifier::ISODOW))) {
					part_data[idx] = isodow;
				}
			}

			// ISO calculations
			if (mask & ISO) {
				int32_t ww = 0;
				int32_t iyyy = 0;
				Date::ExtractISOYearWeek(input, iyyy, ww);
				if ((part_data = HasPartValue(part_values, DatePartSpecifier::WEEK))) {
					part_data[idx] = ww;
				}
				if ((part_data = HasPartValue(part_values, DatePartSpecifier::ISOYEAR))) {
					part_data[idx] = iyyy;
				}
				if ((part_data = HasPartValue(part_values, DatePartSpecifier::YEARWEEK))) {
					part_data[idx] = YearWeekOperator::YearWeekFromParts(iyyy, ww);
				}
			}

			if (mask & EPOCH) {
				if ((part_data = HasPartValue(part_values, DatePartSpecifier::EPOCH))) {
					part_data[idx] = Date::Epoch(input);
				}
			}
			if (mask & DOY) {
				if ((part_data = HasPartValue(part_values, DatePartSpecifier::DOY))) {
					part_data[idx] = Date::ExtractDayOfTheYear(input);
				}
			}
		}
	};
};

template <class T>
static void LastYearFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	int32_t last_year = 0;
	UnaryExecutor::ExecuteWithNulls<T, int64_t>(args.data[0], result, args.size(),
	                                            [&](T input, ValidityMask &mask, idx_t idx) {
		                                            if (Value::IsFinite(input)) {
			                                            return Date::ExtractYear(input, &last_year);
		                                            } else {
			                                            mask.SetInvalid(idx);
			                                            return 0;
		                                            }
	                                            });
}

template <>
int64_t DatePart::YearOperator::Operation(timestamp_t input) {
	return YearOperator::Operation<date_t, int64_t>(Timestamp::GetDate(input));
}

template <>
int64_t DatePart::YearOperator::Operation(interval_t input) {
	return input.months / Interval::MONTHS_PER_YEAR;
}

template <>
int64_t DatePart::YearOperator::Operation(dtime_t input) {
	throw NotImplementedException("\"time\" units \"year\" not recognized");
}

template <>
int64_t DatePart::MonthOperator::Operation(timestamp_t input) {
	return MonthOperator::Operation<date_t, int64_t>(Timestamp::GetDate(input));
}

template <>
int64_t DatePart::MonthOperator::Operation(interval_t input) {
	return input.months % Interval::MONTHS_PER_YEAR;
}

template <>
int64_t DatePart::MonthOperator::Operation(dtime_t input) {
	throw NotImplementedException("\"time\" units \"month\" not recognized");
}

template <>
int64_t DatePart::DayOperator::Operation(timestamp_t input) {
	return DayOperator::Operation<date_t, int64_t>(Timestamp::GetDate(input));
}

template <>
int64_t DatePart::DayOperator::Operation(interval_t input) {
	return input.days;
}

template <>
int64_t DatePart::DayOperator::Operation(dtime_t input) {
	throw NotImplementedException("\"time\" units \"day\" not recognized");
}

template <>
int64_t DatePart::DecadeOperator::Operation(interval_t input) {
	return input.months / Interval::MONTHS_PER_DECADE;
}

template <>
int64_t DatePart::DecadeOperator::Operation(dtime_t input) {
	throw NotImplementedException("\"time\" units \"decade\" not recognized");
}

template <>
int64_t DatePart::CenturyOperator::Operation(interval_t input) {
	return input.months / Interval::MONTHS_PER_CENTURY;
}

template <>
int64_t DatePart::CenturyOperator::Operation(dtime_t input) {
	throw NotImplementedException("\"time\" units \"century\" not recognized");
}

template <>
int64_t DatePart::MillenniumOperator::Operation(interval_t input) {
	return input.months / Interval::MONTHS_PER_MILLENIUM;
}

template <>
int64_t DatePart::MillenniumOperator::Operation(dtime_t input) {
	throw NotImplementedException("\"time\" units \"millennium\" not recognized");
}

template <>
int64_t DatePart::QuarterOperator::Operation(timestamp_t input) {
	return QuarterOperator::Operation<date_t, int64_t>(Timestamp::GetDate(input));
}

template <>
int64_t DatePart::QuarterOperator::Operation(interval_t input) {
	return MonthOperator::Operation<interval_t, int64_t>(input) / Interval::MONTHS_PER_QUARTER + 1;
}

template <>
int64_t DatePart::QuarterOperator::Operation(dtime_t input) {
	throw NotImplementedException("\"time\" units \"quarter\" not recognized");
}

template <>
int64_t DatePart::DayOfWeekOperator::Operation(timestamp_t input) {
	return DayOfWeekOperator::Operation<date_t, int64_t>(Timestamp::GetDate(input));
}

template <>
int64_t DatePart::DayOfWeekOperator::Operation(interval_t input) {
	throw NotImplementedException("interval units \"dow\" not recognized");
}

template <>
int64_t DatePart::DayOfWeekOperator::Operation(dtime_t input) {
	throw NotImplementedException("\"time\" units \"dow\" not recognized");
}

template <>
int64_t DatePart::ISODayOfWeekOperator::Operation(timestamp_t input) {
	return ISODayOfWeekOperator::Operation<date_t, int64_t>(Timestamp::GetDate(input));
}

template <>
int64_t DatePart::ISODayOfWeekOperator::Operation(interval_t input) {
	throw NotImplementedException("interval units \"isodow\" not recognized");
}

template <>
int64_t DatePart::ISODayOfWeekOperator::Operation(dtime_t input) {
	throw NotImplementedException("\"time\" units \"isodow\" not recognized");
}

template <>
int64_t DatePart::DayOfYearOperator::Operation(timestamp_t input) {
	return DayOfYearOperator::Operation<date_t, int64_t>(Timestamp::GetDate(input));
}

template <>
int64_t DatePart::DayOfYearOperator::Operation(interval_t input) {
	throw NotImplementedException("interval units \"doy\" not recognized");
}

template <>
int64_t DatePart::DayOfYearOperator::Operation(dtime_t input) {
	throw NotImplementedException("\"time\" units \"doy\" not recognized");
}

template <>
int64_t DatePart::WeekOperator::Operation(timestamp_t input) {
	return WeekOperator::Operation<date_t, int64_t>(Timestamp::GetDate(input));
}

template <>
int64_t DatePart::WeekOperator::Operation(interval_t input) {
	throw NotImplementedException("interval units \"week\" not recognized");
}

template <>
int64_t DatePart::WeekOperator::Operation(dtime_t input) {
	throw NotImplementedException("\"time\" units \"week\" not recognized");
}

template <>
int64_t DatePart::ISOYearOperator::Operation(timestamp_t input) {
	return ISOYearOperator::Operation<date_t, int64_t>(Timestamp::GetDate(input));
}

template <>
int64_t DatePart::ISOYearOperator::Operation(interval_t input) {
	throw NotImplementedException("interval units \"isoyear\" not recognized");
}

template <>
int64_t DatePart::ISOYearOperator::Operation(dtime_t input) {
	throw NotImplementedException("\"time\" units \"isoyear\" not recognized");
}

template <>
int64_t DatePart::YearWeekOperator::Operation(timestamp_t input) {
	return YearWeekOperator::Operation<date_t, int64_t>(Timestamp::GetDate(input));
}

template <>
int64_t DatePart::YearWeekOperator::Operation(interval_t input) {
	const auto yyyy = YearOperator::Operation<interval_t, int64_t>(input);
	const auto ww = WeekOperator::Operation<interval_t, int64_t>(input);
	return YearWeekOperator::YearWeekFromParts<int64_t>(yyyy, ww);
}

template <>
int64_t DatePart::YearWeekOperator::Operation(dtime_t input) {
	throw NotImplementedException("\"time\" units \"yearweek\" not recognized");
}

template <>
int64_t DatePart::MicrosecondsOperator::Operation(timestamp_t input) {
	auto time = Timestamp::GetTime(input);
	// remove everything but the second & microsecond part
	return time.micros % Interval::MICROS_PER_MINUTE;
}

template <>
int64_t DatePart::MicrosecondsOperator::Operation(interval_t input) {
	// remove everything but the second & microsecond part
	return input.micros % Interval::MICROS_PER_MINUTE;
}

template <>
int64_t DatePart::MicrosecondsOperator::Operation(dtime_t input) {
	// remove everything but the second & microsecond part
	return input.micros % Interval::MICROS_PER_MINUTE;
}

template <>
int64_t DatePart::MillisecondsOperator::Operation(timestamp_t input) {
	return MicrosecondsOperator::Operation<timestamp_t, int64_t>(input) / Interval::MICROS_PER_MSEC;
}

template <>
int64_t DatePart::MillisecondsOperator::Operation(interval_t input) {
	return MicrosecondsOperator::Operation<interval_t, int64_t>(input) / Interval::MICROS_PER_MSEC;
}

template <>
int64_t DatePart::MillisecondsOperator::Operation(dtime_t input) {
	return MicrosecondsOperator::Operation<dtime_t, int64_t>(input) / Interval::MICROS_PER_MSEC;
}

template <>
int64_t DatePart::SecondsOperator::Operation(timestamp_t input) {
	return MicrosecondsOperator::Operation<timestamp_t, int64_t>(input) / Interval::MICROS_PER_SEC;
}

template <>
int64_t DatePart::SecondsOperator::Operation(interval_t input) {
	return MicrosecondsOperator::Operation<interval_t, int64_t>(input) / Interval::MICROS_PER_SEC;
}

template <>
int64_t DatePart::SecondsOperator::Operation(dtime_t input) {
	return MicrosecondsOperator::Operation<dtime_t, int64_t>(input) / Interval::MICROS_PER_SEC;
}

template <>
int64_t DatePart::MinutesOperator::Operation(timestamp_t input) {
	auto time = Timestamp::GetTime(input);
	// remove the hour part, and truncate to minutes
	return (time.micros % Interval::MICROS_PER_HOUR) / Interval::MICROS_PER_MINUTE;
}

template <>
int64_t DatePart::MinutesOperator::Operation(interval_t input) {
	// remove the hour part, and truncate to minutes
	return (input.micros % Interval::MICROS_PER_HOUR) / Interval::MICROS_PER_MINUTE;
}

template <>
int64_t DatePart::MinutesOperator::Operation(dtime_t input) {
	// remove the hour part, and truncate to minutes
	return (input.micros % Interval::MICROS_PER_HOUR) / Interval::MICROS_PER_MINUTE;
}

template <>
int64_t DatePart::HoursOperator::Operation(timestamp_t input) {
	return Timestamp::GetTime(input).micros / Interval::MICROS_PER_HOUR;
}

template <>
int64_t DatePart::HoursOperator::Operation(interval_t input) {
	return input.micros / Interval::MICROS_PER_HOUR;
}

template <>
int64_t DatePart::HoursOperator::Operation(dtime_t input) {
	return input.micros / Interval::MICROS_PER_HOUR;
}

template <>
int64_t DatePart::EpochOperator::Operation(timestamp_t input) {
	return Timestamp::GetEpochSeconds(input);
}

template <>
int64_t DatePart::EpochOperator::Operation(interval_t input) {
	int64_t interval_years = input.months / Interval::MONTHS_PER_YEAR;
	int64_t interval_days;
	interval_days = Interval::DAYS_PER_YEAR * interval_years;
	interval_days += Interval::DAYS_PER_MONTH * (input.months % Interval::MONTHS_PER_YEAR);
	interval_days += input.days;
	int64_t interval_epoch;
	interval_epoch = interval_days * Interval::SECS_PER_DAY;
	// we add 0.25 days per year to sort of account for leap days
	interval_epoch += interval_years * (Interval::SECS_PER_DAY / 4);
	interval_epoch += input.micros / Interval::MICROS_PER_SEC;
	return interval_epoch;
}

template <>
int64_t DatePart::EpochOperator::Operation(dtime_t input) {
	return input.micros / Interval::MICROS_PER_SEC;
}

template <>
unique_ptr<BaseStatistics> DatePart::EpochOperator::PropagateStatistics<dtime_t>(ClientContext &context,
                                                                                 FunctionStatisticsInput &input) {
	// time seconds range over a single day
	return PropagateSimpleDatePartStatistics<0, 86400>(input.child_stats);
}

template <>
int64_t DatePart::EraOperator::Operation(timestamp_t input) {
	return EraOperator::Operation<date_t, int64_t>(Timestamp::GetDate(input));
}

template <>
int64_t DatePart::EraOperator::Operation(interval_t input) {
	throw NotImplementedException("interval units \"era\" not recognized");
}

template <>
int64_t DatePart::EraOperator::Operation(dtime_t input) {
	throw NotImplementedException("\"time\" units \"era\" not recognized");
}

template <>
int64_t DatePart::TimezoneOperator::Operation(date_t input) {
	throw NotImplementedException("\"date\" units \"timezone\" not recognized");
}

template <>
int64_t DatePart::TimezoneOperator::Operation(interval_t input) {
	throw NotImplementedException("\"interval\" units \"timezone\" not recognized");
}

template <>
int64_t DatePart::TimezoneOperator::Operation(dtime_t input) {
	return 0;
}

template <>
void DatePart::StructOperator::Operation(int64_t **part_values, const dtime_t &input, const idx_t idx,
                                         const part_mask_t mask) {
	int64_t *part_data;
	if (mask & TIME) {
		const auto micros = MicrosecondsOperator::Operation<dtime_t, int64_t>(input);
		if ((part_data = HasPartValue(part_values, DatePartSpecifier::MICROSECONDS))) {
			part_data[idx] = micros;
		}
		if ((part_data = HasPartValue(part_values, DatePartSpecifier::MILLISECONDS))) {
			part_data[idx] = micros / Interval::MICROS_PER_MSEC;
		}
		if ((part_data = HasPartValue(part_values, DatePartSpecifier::SECOND))) {
			part_data[idx] = micros / Interval::MICROS_PER_SEC;
		}
		if ((part_data = HasPartValue(part_values, DatePartSpecifier::MINUTE))) {
			part_data[idx] = MinutesOperator::Operation<dtime_t, int64_t>(input);
		}
		if ((part_data = HasPartValue(part_values, DatePartSpecifier::HOUR))) {
			part_data[idx] = HoursOperator::Operation<dtime_t, int64_t>(input);
		}
	}

	if (mask & EPOCH) {
		if ((part_data = HasPartValue(part_values, DatePartSpecifier::EPOCH))) {
			part_data[idx] = EpochOperator::Operation<dtime_t, int64_t>(input);
			;
		}
	}

	if (mask & ZONE) {
		if ((part_data = HasPartValue(part_values, DatePartSpecifier::TIMEZONE))) {
			part_data[idx] = 0;
		}
		if ((part_data = HasPartValue(part_values, DatePartSpecifier::TIMEZONE_HOUR))) {
			part_data[idx] = 0;
		}
		if ((part_data = HasPartValue(part_values, DatePartSpecifier::TIMEZONE_MINUTE))) {
			part_data[idx] = 0;
		}
	}
}

template <>
void DatePart::StructOperator::Operation(int64_t **part_values, const timestamp_t &input, const idx_t idx,
                                         const part_mask_t mask) {
	date_t d;
	dtime_t t;
	Timestamp::Convert(input, d, t);

	// Both define epoch, and the correct value is the sum.
	// So mask it out and compute it separately.
	Operation(part_values, d, idx, mask & ~EPOCH);
	Operation(part_values, t, idx, mask & ~EPOCH);

	if (mask & EPOCH) {
		auto part_data = HasPartValue(part_values, DatePartSpecifier::EPOCH);
		if (part_data) {
			part_data[idx] = EpochOperator::Operation<timestamp_t, int64_t>(input);
		}
	}
}

template <>
void DatePart::StructOperator::Operation(int64_t **part_values, const interval_t &input, const idx_t idx,
                                         const part_mask_t mask) {
	int64_t *part_data;
	if (mask & YMD) {
		const auto mm = input.months % Interval::MONTHS_PER_YEAR;
		if ((part_data = HasPartValue(part_values, DatePartSpecifier::YEAR))) {
			part_data[idx] = input.months / Interval::MONTHS_PER_YEAR;
		}
		if ((part_data = HasPartValue(part_values, DatePartSpecifier::MONTH))) {
			part_data[idx] = mm;
		}
		if ((part_data = HasPartValue(part_values, DatePartSpecifier::DAY))) {
			part_data[idx] = input.days;
		}
		if ((part_data = HasPartValue(part_values, DatePartSpecifier::DECADE))) {
			part_data[idx] = input.months / Interval::MONTHS_PER_DECADE;
		}
		if ((part_data = HasPartValue(part_values, DatePartSpecifier::CENTURY))) {
			part_data[idx] = input.months / Interval::MONTHS_PER_CENTURY;
		}
		if ((part_data = HasPartValue(part_values, DatePartSpecifier::MILLENNIUM))) {
			part_data[idx] = input.months / Interval::MONTHS_PER_MILLENIUM;
		}
		if ((part_data = HasPartValue(part_values, DatePartSpecifier::QUARTER))) {
			part_data[idx] = mm / Interval::MONTHS_PER_QUARTER + 1;
		}
	}

	if (mask & TIME) {
		const auto micros = MicrosecondsOperator::Operation<interval_t, int64_t>(input);
		if ((part_data = HasPartValue(part_values, DatePartSpecifier::MICROSECONDS))) {
			part_data[idx] = micros;
		}
		if ((part_data = HasPartValue(part_values, DatePartSpecifier::MILLISECONDS))) {
			part_data[idx] = micros / Interval::MICROS_PER_MSEC;
		}
		if ((part_data = HasPartValue(part_values, DatePartSpecifier::SECOND))) {
			part_data[idx] = micros / Interval::MICROS_PER_SEC;
		}
		if ((part_data = HasPartValue(part_values, DatePartSpecifier::MINUTE))) {
			part_data[idx] = MinutesOperator::Operation<interval_t, int64_t>(input);
		}
		if ((part_data = HasPartValue(part_values, DatePartSpecifier::HOUR))) {
			part_data[idx] = HoursOperator::Operation<interval_t, int64_t>(input);
		}
	}

	if (mask & EPOCH) {
		if ((part_data = HasPartValue(part_values, DatePartSpecifier::EPOCH))) {
			part_data[idx] = EpochOperator::Operation<interval_t, int64_t>(input);
		}
	}
}

template <typename T>
static int64_t ExtractElement(DatePartSpecifier type, T element) {
	switch (type) {
	case DatePartSpecifier::YEAR:
		return DatePart::YearOperator::template Operation<T, int64_t>(element);
	case DatePartSpecifier::MONTH:
		return DatePart::MonthOperator::template Operation<T, int64_t>(element);
	case DatePartSpecifier::DAY:
		return DatePart::DayOperator::template Operation<T, int64_t>(element);
	case DatePartSpecifier::DECADE:
		return DatePart::DecadeOperator::template Operation<T, int64_t>(element);
	case DatePartSpecifier::CENTURY:
		return DatePart::CenturyOperator::template Operation<T, int64_t>(element);
	case DatePartSpecifier::MILLENNIUM:
		return DatePart::MillenniumOperator::template Operation<T, int64_t>(element);
	case DatePartSpecifier::QUARTER:
		return DatePart::QuarterOperator::template Operation<T, int64_t>(element);
	case DatePartSpecifier::DOW:
		return DatePart::DayOfWeekOperator::template Operation<T, int64_t>(element);
	case DatePartSpecifier::ISODOW:
		return DatePart::ISODayOfWeekOperator::template Operation<T, int64_t>(element);
	case DatePartSpecifier::DOY:
		return DatePart::DayOfYearOperator::template Operation<T, int64_t>(element);
	case DatePartSpecifier::WEEK:
		return DatePart::WeekOperator::template Operation<T, int64_t>(element);
	case DatePartSpecifier::ISOYEAR:
		return DatePart::ISOYearOperator::template Operation<T, int64_t>(element);
	case DatePartSpecifier::YEARWEEK:
		return DatePart::YearWeekOperator::template Operation<T, int64_t>(element);
	case DatePartSpecifier::EPOCH:
		return DatePart::EpochOperator::template Operation<T, int64_t>(element);
	case DatePartSpecifier::MICROSECONDS:
		return DatePart::MicrosecondsOperator::template Operation<T, int64_t>(element);
	case DatePartSpecifier::MILLISECONDS:
		return DatePart::MillisecondsOperator::template Operation<T, int64_t>(element);
	case DatePartSpecifier::SECOND:
		return DatePart::SecondsOperator::template Operation<T, int64_t>(element);
	case DatePartSpecifier::MINUTE:
		return DatePart::MinutesOperator::template Operation<T, int64_t>(element);
	case DatePartSpecifier::HOUR:
		return DatePart::HoursOperator::template Operation<T, int64_t>(element);
	case DatePartSpecifier::ERA:
		return DatePart::EraOperator::template Operation<T, int64_t>(element);
	case DatePartSpecifier::TIMEZONE:
	case DatePartSpecifier::TIMEZONE_HOUR:
	case DatePartSpecifier::TIMEZONE_MINUTE:
		return DatePart::TimezoneOperator::template Operation<T, int64_t>(element);
	default:
		throw NotImplementedException("Specifier type not implemented for DATEPART");
	}
}

template <typename T>
static void DatePartFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(args.ColumnCount() == 2);
	auto &spec_arg = args.data[0];
	auto &date_arg = args.data[1];

	BinaryExecutor::ExecuteWithNulls<string_t, T, int64_t>(
	    spec_arg, date_arg, result, args.size(), [&](string_t specifier, T date, ValidityMask &mask, idx_t idx) {
		    if (Value::IsFinite(date)) {
			    return ExtractElement<T>(GetDatePartSpecifier(specifier.GetString()), date);
		    } else {
			    mask.SetInvalid(idx);
			    return int64_t(0);
		    }
	    });
}

ScalarFunctionSet GetGenericDatePartFunction(scalar_function_t date_func, scalar_function_t ts_func,
                                             scalar_function_t interval_func, function_statistics_t date_stats,
                                             function_statistics_t ts_stats) {
	ScalarFunctionSet operator_set;
	operator_set.AddFunction(
	    ScalarFunction({LogicalType::DATE}, LogicalType::BIGINT, std::move(date_func), nullptr, nullptr, date_stats));
	operator_set.AddFunction(
	    ScalarFunction({LogicalType::TIMESTAMP}, LogicalType::BIGINT, std::move(ts_func), nullptr, nullptr, ts_stats));
	operator_set.AddFunction(ScalarFunction({LogicalType::INTERVAL}, LogicalType::BIGINT, std::move(interval_func)));
	return operator_set;
}

template <class OP>
static ScalarFunctionSet GetDatePartFunction() {
	return GetGenericDatePartFunction(
	    DatePart::UnaryFunction<date_t, int64_t, OP>, DatePart::UnaryFunction<timestamp_t, int64_t, OP>,
	    ScalarFunction::UnaryFunction<interval_t, int64_t, OP>, OP::template PropagateStatistics<date_t>,
	    OP::template PropagateStatistics<timestamp_t>);
}

ScalarFunctionSet GetGenericTimePartFunction(scalar_function_t date_func, scalar_function_t ts_func,
                                             scalar_function_t interval_func, scalar_function_t time_func,
                                             function_statistics_t date_stats, function_statistics_t ts_stats,
                                             function_statistics_t time_stats) {
	ScalarFunctionSet operator_set;
	operator_set.AddFunction(
	    ScalarFunction({LogicalType::DATE}, LogicalType::BIGINT, std::move(date_func), nullptr, nullptr, date_stats));
	operator_set.AddFunction(
	    ScalarFunction({LogicalType::TIMESTAMP}, LogicalType::BIGINT, std::move(ts_func), nullptr, nullptr, ts_stats));
	operator_set.AddFunction(ScalarFunction({LogicalType::INTERVAL}, LogicalType::BIGINT, std::move(interval_func)));
	operator_set.AddFunction(
	    ScalarFunction({LogicalType::TIME}, LogicalType::BIGINT, std::move(time_func), nullptr, nullptr, time_stats));
	return operator_set;
}

template <class OP>
static ScalarFunctionSet GetTimePartFunction() {
	return GetGenericTimePartFunction(
	    DatePart::UnaryFunction<date_t, int64_t, OP>, DatePart::UnaryFunction<timestamp_t, int64_t, OP>,
	    ScalarFunction::UnaryFunction<interval_t, int64_t, OP>, ScalarFunction::UnaryFunction<dtime_t, int64_t, OP>,
	    OP::template PropagateStatistics<date_t>, OP::template PropagateStatistics<timestamp_t>,
	    OP::template PropagateStatistics<dtime_t>);
}

struct LastDayOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		int32_t yyyy, mm, dd;
		Date::Convert(input, yyyy, mm, dd);
		yyyy += (mm / 12);
		mm %= 12;
		++mm;
		return Date::FromDate(yyyy, mm, 1) - 1;
	}
};

template <>
date_t LastDayOperator::Operation(timestamp_t input) {
	return LastDayOperator::Operation<date_t, date_t>(Timestamp::GetDate(input));
}

struct MonthNameOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		return Date::MONTH_NAMES[DatePart::MonthOperator::Operation<TA, int64_t>(input) - 1];
	}
};

struct DayNameOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		return Date::DAY_NAMES[DatePart::DayOfWeekOperator::Operation<TA, int64_t>(input)];
	}
};

struct StructDatePart {
	using part_codes_t = vector<DatePartSpecifier>;

	struct BindData : public VariableReturnBindData {
		part_codes_t part_codes;

		explicit BindData(const LogicalType &stype, const part_codes_t &part_codes_p)
		    : VariableReturnBindData(stype), part_codes(part_codes_p) {
		}

		unique_ptr<FunctionData> Copy() const override {
			return make_uniq<BindData>(stype, part_codes);
		}
	};

	static unique_ptr<FunctionData> Bind(ClientContext &context, ScalarFunction &bound_function,
	                                     vector<unique_ptr<Expression>> &arguments) {
		// collect names and deconflict, construct return type
		if (arguments[0]->HasParameter()) {
			throw ParameterNotResolvedException();
		}
		if (!arguments[0]->IsFoldable()) {
			throw BinderException("%s can only take constant lists of part names", bound_function.name);
		}

		case_insensitive_set_t name_collision_set;
		child_list_t<LogicalType> struct_children;
		part_codes_t part_codes;

		Value parts_list = ExpressionExecutor::EvaluateScalar(context, *arguments[0]);
		if (parts_list.type().id() == LogicalTypeId::LIST) {
			auto &list_children = ListValue::GetChildren(parts_list);
			if (list_children.empty()) {
				throw BinderException("%s requires non-empty lists of part names", bound_function.name);
			}
			for (const auto &part_value : list_children) {
				if (part_value.IsNull()) {
					throw BinderException("NULL struct entry name in %s", bound_function.name);
				}
				const auto part_name = part_value.ToString();
				const auto part_code = GetDateTypePartSpecifier(part_name, arguments[1]->return_type);
				if (name_collision_set.find(part_name) != name_collision_set.end()) {
					throw BinderException("Duplicate struct entry name \"%s\" in %s", part_name, bound_function.name);
				}
				name_collision_set.insert(part_name);
				part_codes.emplace_back(part_code);
				struct_children.emplace_back(make_pair(part_name, LogicalType::BIGINT));
			}
		} else {
			throw BinderException("%s can only take constant lists of part names", bound_function.name);
		}

		Function::EraseArgument(bound_function, arguments, 0);
		bound_function.return_type = LogicalType::STRUCT(struct_children);
		return make_uniq<BindData>(bound_function.return_type, part_codes);
	}

	template <typename INPUT_TYPE>
	static void Function(DataChunk &args, ExpressionState &state, Vector &result) {
		auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
		auto &info = (BindData &)*func_expr.bind_info;
		D_ASSERT(args.ColumnCount() == 1);

		const auto count = args.size();
		Vector &input = args.data[0];
		vector<int64_t *> part_values(int(DatePartSpecifier::TIMEZONE_MINUTE) + 1, nullptr);
		const auto part_mask = DatePart::StructOperator::GetMask(info.part_codes);

		auto &child_entries = StructVector::GetEntries(result);

		// The first computer of a part "owns" it
		// and other requestors just reference the owner
		vector<size_t> owners(int(DatePartSpecifier::TIMEZONE_MINUTE) + 1, child_entries.size());
		for (size_t col = 0; col < child_entries.size(); ++col) {
			const auto part_index = size_t(info.part_codes[col]);
			if (owners[part_index] == child_entries.size()) {
				owners[part_index] = col;
			}
		}

		if (input.GetVectorType() == VectorType::CONSTANT_VECTOR) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);

			if (ConstantVector::IsNull(input)) {
				ConstantVector::SetNull(result, true);
			} else {
				ConstantVector::SetNull(result, false);
				for (size_t col = 0; col < child_entries.size(); ++col) {
					auto &child_entry = child_entries[col];
					ConstantVector::SetNull(*child_entry, false);
					const auto part_index = size_t(info.part_codes[col]);
					if (owners[part_index] == col) {
						part_values[part_index] = ConstantVector::GetData<int64_t>(*child_entry);
					}
				}
				auto tdata = ConstantVector::GetData<INPUT_TYPE>(input);
				if (Value::IsFinite(tdata[0])) {
					DatePart::StructOperator::Operation(part_values.data(), tdata[0], 0, part_mask);
				} else {
					for (auto &child_entry : child_entries) {
						ConstantVector::SetNull(*child_entry, true);
					}
				}
			}
		} else {
			UnifiedVectorFormat rdata;
			input.ToUnifiedFormat(count, rdata);

			const auto &arg_valid = rdata.validity;
			auto tdata = (const INPUT_TYPE *)rdata.data;

			// Start with a valid flat vector
			result.SetVectorType(VectorType::FLAT_VECTOR);
			auto &res_valid = FlatVector::Validity(result);
			if (res_valid.GetData()) {
				res_valid.SetAllValid(count);
			}

			// Start with valid children
			for (size_t col = 0; col < child_entries.size(); ++col) {
				auto &child_entry = child_entries[col];
				child_entry->SetVectorType(VectorType::FLAT_VECTOR);
				auto &child_validity = FlatVector::Validity(*child_entry);
				if (child_validity.GetData()) {
					child_validity.SetAllValid(count);
				}

				// Pre-multiplex
				const auto part_index = size_t(info.part_codes[col]);
				if (owners[part_index] == col) {
					part_values[part_index] = FlatVector::GetData<int64_t>(*child_entry);
				}
			}

			for (idx_t i = 0; i < count; ++i) {
				const auto idx = rdata.sel->get_index(i);
				if (arg_valid.RowIsValid(idx)) {
					if (Value::IsFinite(tdata[idx])) {
						DatePart::StructOperator::Operation(part_values.data(), tdata[idx], i, part_mask);
					} else {
						for (auto &child_entry : child_entries) {
							FlatVector::Validity(*child_entry).SetInvalid(i);
						}
					}
				} else {
					res_valid.SetInvalid(i);
					for (auto &child_entry : child_entries) {
						FlatVector::Validity(*child_entry).SetInvalid(i);
					}
				}
			}
		}

		// Reference any duplicate parts
		for (size_t col = 0; col < child_entries.size(); ++col) {
			const auto part_index = size_t(info.part_codes[col]);
			const auto owner = owners[part_index];
			if (owner != col) {
				child_entries[col]->Reference(*child_entries[owner]);
			}
		}

		result.Verify(count);
	}

	static void SerializeFunction(FieldWriter &writer, const FunctionData *bind_data_p,
	                              const ScalarFunction &function) {
		D_ASSERT(bind_data_p);
		auto &info = (BindData &)*bind_data_p;
		writer.WriteSerializable(info.stype);
		writer.WriteList<DatePartSpecifier>(info.part_codes);
	}

	static unique_ptr<FunctionData> DeserializeFunction(ClientContext &context, FieldReader &reader,
	                                                    ScalarFunction &bound_function) {
		auto stype = reader.ReadRequiredSerializable<LogicalType, LogicalType>();
		auto part_codes = reader.ReadRequiredList<DatePartSpecifier>();
		return make_uniq<BindData>(std::move(stype), std::move(part_codes));
	}

	template <typename INPUT_TYPE>
	static ScalarFunction GetFunction(const LogicalType &temporal_type) {
		auto part_type = LogicalType::LIST(LogicalType::VARCHAR);
		auto result_type = LogicalType::STRUCT({});
		ScalarFunction result({part_type, temporal_type}, result_type, Function<INPUT_TYPE>, Bind);
		result.serialize = SerializeFunction;
		result.deserialize = DeserializeFunction;
		return result;
	}
};

ScalarFunctionSet YearFun::GetFunctions() {
	return GetGenericDatePartFunction(LastYearFunction<date_t>, LastYearFunction<timestamp_t>,
	                                  ScalarFunction::UnaryFunction<interval_t, int64_t, DatePart::YearOperator>,
	                                  DatePart::YearOperator::PropagateStatistics<date_t>,
	                                  DatePart::YearOperator::PropagateStatistics<timestamp_t>);
}

ScalarFunctionSet MonthFun::GetFunctions() {
	return GetDatePartFunction<DatePart::MonthOperator>();
}

ScalarFunctionSet DayFun::GetFunctions() {
	return GetDatePartFunction<DatePart::DayOperator>();
}

ScalarFunctionSet DecadeFun::GetFunctions() {
	return GetDatePartFunction<DatePart::DecadeOperator>();
}

ScalarFunctionSet CenturyFun::GetFunctions() {
	return GetDatePartFunction<DatePart::CenturyOperator>();
}

ScalarFunctionSet MillenniumFun::GetFunctions() {
	return GetDatePartFunction<DatePart::MillenniumOperator>();
}

ScalarFunctionSet QuarterFun::GetFunctions() {
	return GetDatePartFunction<DatePart::QuarterOperator>();
}

ScalarFunctionSet DayOfWeekFun::GetFunctions() {
	return GetDatePartFunction<DatePart::DayOfWeekOperator>();
}

ScalarFunctionSet ISODayOfWeekFun::GetFunctions() {
	return GetDatePartFunction<DatePart::ISODayOfWeekOperator>();
}

ScalarFunctionSet DayOfYearFun::GetFunctions() {
	return GetDatePartFunction<DatePart::DayOfYearOperator>();
}

ScalarFunctionSet WeekFun::GetFunctions() {
	return GetDatePartFunction<DatePart::WeekOperator>();
}

ScalarFunctionSet ISOYearFun::GetFunctions() {
	return GetDatePartFunction<DatePart::ISOYearOperator>();
}

ScalarFunctionSet EraFun::GetFunctions() {
	return GetDatePartFunction<DatePart::EraOperator>();
}

ScalarFunctionSet TimezoneFun::GetFunctions() {
	return GetDatePartFunction<DatePart::TimezoneOperator>();
}

ScalarFunctionSet TimezoneHourFun::GetFunctions() {
	return GetDatePartFunction<DatePart::TimezoneHourOperator>();
}

ScalarFunctionSet TimezoneMinuteFun::GetFunctions() {
	return GetDatePartFunction<DatePart::TimezoneMinuteOperator>();
}

ScalarFunctionSet EpochFun::GetFunctions() {
	return GetTimePartFunction<DatePart::EpochOperator>();
}

ScalarFunctionSet MicrosecondsFun::GetFunctions() {
	return GetTimePartFunction<DatePart::MicrosecondsOperator>();
}

ScalarFunctionSet MillisecondsFun::GetFunctions() {
	return GetTimePartFunction<DatePart::MillisecondsOperator>();
}

ScalarFunctionSet SecondsFun::GetFunctions() {
	return GetTimePartFunction<DatePart::SecondsOperator>();
}

ScalarFunctionSet MinutesFun::GetFunctions() {
	return GetTimePartFunction<DatePart::MinutesOperator>();
}

ScalarFunctionSet HoursFun::GetFunctions() {
	return GetTimePartFunction<DatePart::HoursOperator>();
}

ScalarFunctionSet YearWeekFun::GetFunctions() {
	return GetDatePartFunction<DatePart::YearWeekOperator>();
}

ScalarFunctionSet DayOfMonthFun::GetFunctions() {
	return GetDatePartFunction<DatePart::DayOperator>();
}

ScalarFunctionSet WeekDayFun::GetFunctions() {
	return GetDatePartFunction<DatePart::DayOfWeekOperator>();
}

ScalarFunctionSet WeekOfYearFun::GetFunctions() {
	return GetDatePartFunction<DatePart::WeekOperator>();
}

ScalarFunctionSet LastDayFun::GetFunctions() {
	ScalarFunctionSet last_day;
	last_day.AddFunction(ScalarFunction({LogicalType::DATE}, LogicalType::DATE,
	                                    DatePart::UnaryFunction<date_t, date_t, LastDayOperator>));
	last_day.AddFunction(ScalarFunction({LogicalType::TIMESTAMP}, LogicalType::DATE,
	                                    DatePart::UnaryFunction<timestamp_t, date_t, LastDayOperator>));
	return last_day;
}

ScalarFunctionSet MonthNameFun::GetFunctions() {
	ScalarFunctionSet monthname;
	monthname.AddFunction(ScalarFunction({LogicalType::DATE}, LogicalType::VARCHAR,
	                                     DatePart::UnaryFunction<date_t, string_t, MonthNameOperator>));
	monthname.AddFunction(ScalarFunction({LogicalType::TIMESTAMP}, LogicalType::VARCHAR,
	                                     DatePart::UnaryFunction<timestamp_t, string_t, MonthNameOperator>));
	return monthname;
}

ScalarFunctionSet DayNameFun::GetFunctions() {
	ScalarFunctionSet dayname;
	dayname.AddFunction(ScalarFunction({LogicalType::DATE}, LogicalType::VARCHAR,
	                                   DatePart::UnaryFunction<date_t, string_t, DayNameOperator>));
	dayname.AddFunction(ScalarFunction({LogicalType::TIMESTAMP}, LogicalType::VARCHAR,
	                                   DatePart::UnaryFunction<timestamp_t, string_t, DayNameOperator>));
	return dayname;
}

ScalarFunctionSet DatePartFun::GetFunctions() {
	ScalarFunctionSet date_part;
	date_part.AddFunction(
	    ScalarFunction({LogicalType::VARCHAR, LogicalType::DATE}, LogicalType::BIGINT, DatePartFunction<date_t>));
	date_part.AddFunction(ScalarFunction({LogicalType::VARCHAR, LogicalType::TIMESTAMP}, LogicalType::BIGINT,
	                                     DatePartFunction<timestamp_t>));
	date_part.AddFunction(
	    ScalarFunction({LogicalType::VARCHAR, LogicalType::TIME}, LogicalType::BIGINT, DatePartFunction<dtime_t>));
	date_part.AddFunction(ScalarFunction({LogicalType::VARCHAR, LogicalType::INTERVAL}, LogicalType::BIGINT,
	                                     DatePartFunction<interval_t>));

	// struct variants
	date_part.AddFunction(StructDatePart::GetFunction<date_t>(LogicalType::DATE));
	date_part.AddFunction(StructDatePart::GetFunction<timestamp_t>(LogicalType::TIMESTAMP));
	date_part.AddFunction(StructDatePart::GetFunction<dtime_t>(LogicalType::TIME));
	date_part.AddFunction(StructDatePart::GetFunction<interval_t>(LogicalType::INTERVAL));

	return date_part;
}

} // namespace duckdb












namespace duckdb {

struct DateSub {
	static int64_t SubtractMicros(timestamp_t startdate, timestamp_t enddate) {
		const auto start = Timestamp::GetEpochMicroSeconds(startdate);
		const auto end = Timestamp::GetEpochMicroSeconds(enddate);
		return SubtractOperatorOverflowCheck::Operation<int64_t, int64_t, int64_t>(end, start);
	}

	template <class TA, class TB, class TR, class OP>
	static inline void BinaryExecute(Vector &left, Vector &right, Vector &result, idx_t count) {
		BinaryExecutor::ExecuteWithNulls<TA, TB, TR>(
		    left, right, result, count, [&](TA startdate, TB enddate, ValidityMask &mask, idx_t idx) {
			    if (Value::IsFinite(startdate) && Value::IsFinite(enddate)) {
				    return OP::template Operation<TA, TB, TR>(startdate, enddate);
			    } else {
				    mask.SetInvalid(idx);
				    return TR();
			    }
		    });
	}

	struct MonthOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA start_ts, TB end_ts) {

			if (start_ts > end_ts) {
				return -MonthOperator::Operation<TA, TB, TR>(end_ts, start_ts);
			}
			// The number of complete months depends on whether end_ts is on the last day of the month.
			date_t end_date;
			dtime_t end_time;
			Timestamp::Convert(end_ts, end_date, end_time);

			int32_t yyyy, mm, dd;
			Date::Convert(end_date, yyyy, mm, dd);
			const auto end_days = Date::MonthDays(yyyy, mm);
			if (end_days == dd) {
				// Now check whether the start day is after the end day
				date_t start_date;
				dtime_t start_time;
				Timestamp::Convert(start_ts, start_date, start_time);
				Date::Convert(start_date, yyyy, mm, dd);
				if (dd > end_days || (dd == end_days && start_time < end_time)) {
					// Move back to the same time on the last day of the (shorter) end month
					start_date = Date::FromDate(yyyy, mm, end_days);
					start_ts = Timestamp::FromDatetime(start_date, start_time);
				}
			}

			// Our interval difference will now give the correct result.
			// Note that PG gives different interval subtraction results,
			// so if we change this we will have to reimplement.
			return Interval::GetAge(end_ts, start_ts).months;
		}
	};

	struct QuarterOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA start_ts, TB end_ts) {
			return MonthOperator::Operation<TA, TB, TR>(start_ts, end_ts) / Interval::MONTHS_PER_QUARTER;
		}
	};

	struct YearOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA start_ts, TB end_ts) {
			return MonthOperator::Operation<TA, TB, TR>(start_ts, end_ts) / Interval::MONTHS_PER_YEAR;
		}
	};

	struct DecadeOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA start_ts, TB end_ts) {
			return MonthOperator::Operation<TA, TB, TR>(start_ts, end_ts) / Interval::MONTHS_PER_DECADE;
		}
	};

	struct CenturyOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA start_ts, TB end_ts) {
			return MonthOperator::Operation<TA, TB, TR>(start_ts, end_ts) / Interval::MONTHS_PER_CENTURY;
		}
	};

	struct MilleniumOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA start_ts, TB end_ts) {
			return MonthOperator::Operation<TA, TB, TR>(start_ts, end_ts) / Interval::MONTHS_PER_MILLENIUM;
		}
	};

	struct DayOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA startdate, TB enddate) {
			return SubtractMicros(startdate, enddate) / Interval::MICROS_PER_DAY;
		}
	};

	struct WeekOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA startdate, TB enddate) {
			return SubtractMicros(startdate, enddate) / Interval::MICROS_PER_WEEK;
		}
	};

	struct MicrosecondsOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA startdate, TB enddate) {
			return SubtractMicros(startdate, enddate);
		}
	};

	struct MillisecondsOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA startdate, TB enddate) {
			return SubtractMicros(startdate, enddate) / Interval::MICROS_PER_MSEC;
		}
	};

	struct SecondsOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA startdate, TB enddate) {
			return SubtractMicros(startdate, enddate) / Interval::MICROS_PER_SEC;
		}
	};

	struct MinutesOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA startdate, TB enddate) {
			return SubtractMicros(startdate, enddate) / Interval::MICROS_PER_MINUTE;
		}
	};

	struct HoursOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA startdate, TB enddate) {
			return SubtractMicros(startdate, enddate) / Interval::MICROS_PER_HOUR;
		}
	};
};

// DATE specialisations
template <>
int64_t DateSub::YearOperator::Operation(date_t startdate, date_t enddate) {
	dtime_t t0(0);
	return YearOperator::Operation<timestamp_t, timestamp_t, int64_t>(Timestamp::FromDatetime(startdate, t0),
	                                                                  Timestamp::FromDatetime(enddate, t0));
}

template <>
int64_t DateSub::MonthOperator::Operation(date_t startdate, date_t enddate) {
	dtime_t t0(0);
	return MonthOperator::Operation<timestamp_t, timestamp_t, int64_t>(Timestamp::FromDatetime(startdate, t0),
	                                                                   Timestamp::FromDatetime(enddate, t0));
}

template <>
int64_t DateSub::DayOperator::Operation(date_t startdate, date_t enddate) {
	dtime_t t0(0);
	return DayOperator::Operation<timestamp_t, timestamp_t, int64_t>(Timestamp::FromDatetime(startdate, t0),
	                                                                 Timestamp::FromDatetime(enddate, t0));
}

template <>
int64_t DateSub::DecadeOperator::Operation(date_t startdate, date_t enddate) {
	dtime_t t0(0);
	return DecadeOperator::Operation<timestamp_t, timestamp_t, int64_t>(Timestamp::FromDatetime(startdate, t0),
	                                                                    Timestamp::FromDatetime(enddate, t0));
}

template <>
int64_t DateSub::CenturyOperator::Operation(date_t startdate, date_t enddate) {
	dtime_t t0(0);
	return CenturyOperator::Operation<timestamp_t, timestamp_t, int64_t>(Timestamp::FromDatetime(startdate, t0),
	                                                                     Timestamp::FromDatetime(enddate, t0));
}

template <>
int64_t DateSub::MilleniumOperator::Operation(date_t startdate, date_t enddate) {
	dtime_t t0(0);
	return MilleniumOperator::Operation<timestamp_t, timestamp_t, int64_t>(Timestamp::FromDatetime(startdate, t0),
	                                                                       Timestamp::FromDatetime(enddate, t0));
}

template <>
int64_t DateSub::QuarterOperator::Operation(date_t startdate, date_t enddate) {
	dtime_t t0(0);
	return QuarterOperator::Operation<timestamp_t, timestamp_t, int64_t>(Timestamp::FromDatetime(startdate, t0),
	                                                                     Timestamp::FromDatetime(enddate, t0));
}

template <>
int64_t DateSub::WeekOperator::Operation(date_t startdate, date_t enddate) {
	dtime_t t0(0);
	return WeekOperator::Operation<timestamp_t, timestamp_t, int64_t>(Timestamp::FromDatetime(startdate, t0),
	                                                                  Timestamp::FromDatetime(enddate, t0));
}

template <>
int64_t DateSub::MicrosecondsOperator::Operation(date_t startdate, date_t enddate) {
	dtime_t t0(0);
	return MicrosecondsOperator::Operation<timestamp_t, timestamp_t, int64_t>(Timestamp::FromDatetime(startdate, t0),
	                                                                          Timestamp::FromDatetime(enddate, t0));
}

template <>
int64_t DateSub::MillisecondsOperator::Operation(date_t startdate, date_t enddate) {
	dtime_t t0(0);
	return MillisecondsOperator::Operation<timestamp_t, timestamp_t, int64_t>(Timestamp::FromDatetime(startdate, t0),
	                                                                          Timestamp::FromDatetime(enddate, t0));
}

template <>
int64_t DateSub::SecondsOperator::Operation(date_t startdate, date_t enddate) {
	dtime_t t0(0);
	return SecondsOperator::Operation<timestamp_t, timestamp_t, int64_t>(Timestamp::FromDatetime(startdate, t0),
	                                                                     Timestamp::FromDatetime(enddate, t0));
}

template <>
int64_t DateSub::MinutesOperator::Operation(date_t startdate, date_t enddate) {
	dtime_t t0(0);
	return MinutesOperator::Operation<timestamp_t, timestamp_t, int64_t>(Timestamp::FromDatetime(startdate, t0),
	                                                                     Timestamp::FromDatetime(enddate, t0));
}

template <>
int64_t DateSub::HoursOperator::Operation(date_t startdate, date_t enddate) {
	dtime_t t0(0);
	return HoursOperator::Operation<timestamp_t, timestamp_t, int64_t>(Timestamp::FromDatetime(startdate, t0),
	                                                                   Timestamp::FromDatetime(enddate, t0));
}

// TIME specialisations
template <>
int64_t DateSub::YearOperator::Operation(dtime_t startdate, dtime_t enddate) {
	throw NotImplementedException("\"time\" units \"year\" not recognized");
}

template <>
int64_t DateSub::MonthOperator::Operation(dtime_t startdate, dtime_t enddate) {
	throw NotImplementedException("\"time\" units \"month\" not recognized");
}

template <>
int64_t DateSub::DayOperator::Operation(dtime_t startdate, dtime_t enddate) {
	throw NotImplementedException("\"time\" units \"day\" not recognized");
}

template <>
int64_t DateSub::DecadeOperator::Operation(dtime_t startdate, dtime_t enddate) {
	throw NotImplementedException("\"time\" units \"decade\" not recognized");
}

template <>
int64_t DateSub::CenturyOperator::Operation(dtime_t startdate, dtime_t enddate) {
	throw NotImplementedException("\"time\" units \"century\" not recognized");
}

template <>
int64_t DateSub::MilleniumOperator::Operation(dtime_t startdate, dtime_t enddate) {
	throw NotImplementedException("\"time\" units \"millennium\" not recognized");
}

template <>
int64_t DateSub::QuarterOperator::Operation(dtime_t startdate, dtime_t enddate) {
	throw NotImplementedException("\"time\" units \"quarter\" not recognized");
}

template <>
int64_t DateSub::WeekOperator::Operation(dtime_t startdate, dtime_t enddate) {
	throw NotImplementedException("\"time\" units \"week\" not recognized");
}

template <>
int64_t DateSub::MicrosecondsOperator::Operation(dtime_t startdate, dtime_t enddate) {
	return enddate.micros - startdate.micros;
}

template <>
int64_t DateSub::MillisecondsOperator::Operation(dtime_t startdate, dtime_t enddate) {
	return (enddate.micros - startdate.micros) / Interval::MICROS_PER_MSEC;
}

template <>
int64_t DateSub::SecondsOperator::Operation(dtime_t startdate, dtime_t enddate) {
	return (enddate.micros - startdate.micros) / Interval::MICROS_PER_SEC;
}

template <>
int64_t DateSub::MinutesOperator::Operation(dtime_t startdate, dtime_t enddate) {
	return (enddate.micros - startdate.micros) / Interval::MICROS_PER_MINUTE;
}

template <>
int64_t DateSub::HoursOperator::Operation(dtime_t startdate, dtime_t enddate) {
	return (enddate.micros - startdate.micros) / Interval::MICROS_PER_HOUR;
}

template <typename TA, typename TB, typename TR>
static int64_t SubtractDateParts(DatePartSpecifier type, TA startdate, TB enddate) {
	switch (type) {
	case DatePartSpecifier::YEAR:
	case DatePartSpecifier::ISOYEAR:
		return DateSub::YearOperator::template Operation<TA, TB, TR>(startdate, enddate);
	case DatePartSpecifier::MONTH:
		return DateSub::MonthOperator::template Operation<TA, TB, TR>(startdate, enddate);
	case DatePartSpecifier::DAY:
	case DatePartSpecifier::DOW:
	case DatePartSpecifier::ISODOW:
	case DatePartSpecifier::DOY:
		return DateSub::DayOperator::template Operation<TA, TB, TR>(startdate, enddate);
	case DatePartSpecifier::DECADE:
		return DateSub::DecadeOperator::template Operation<TA, TB, TR>(startdate, enddate);
	case DatePartSpecifier::CENTURY:
		return DateSub::CenturyOperator::template Operation<TA, TB, TR>(startdate, enddate);
	case DatePartSpecifier::MILLENNIUM:
		return DateSub::MilleniumOperator::template Operation<TA, TB, TR>(startdate, enddate);
	case DatePartSpecifier::QUARTER:
		return DateSub::QuarterOperator::template Operation<TA, TB, TR>(startdate, enddate);
	case DatePartSpecifier::WEEK:
	case DatePartSpecifier::YEARWEEK:
		return DateSub::WeekOperator::template Operation<TA, TB, TR>(startdate, enddate);
	case DatePartSpecifier::MICROSECONDS:
		return DateSub::MicrosecondsOperator::template Operation<TA, TB, TR>(startdate, enddate);
	case DatePartSpecifier::MILLISECONDS:
		return DateSub::MillisecondsOperator::template Operation<TA, TB, TR>(startdate, enddate);
	case DatePartSpecifier::SECOND:
	case DatePartSpecifier::EPOCH:
		return DateSub::SecondsOperator::template Operation<TA, TB, TR>(startdate, enddate);
	case DatePartSpecifier::MINUTE:
		return DateSub::MinutesOperator::template Operation<TA, TB, TR>(startdate, enddate);
	case DatePartSpecifier::HOUR:
		return DateSub::HoursOperator::template Operation<TA, TB, TR>(startdate, enddate);
	default:
		throw NotImplementedException("Specifier type not implemented for DATESUB");
	}
}

struct DateSubTernaryOperator {
	template <typename TS, typename TA, typename TB, typename TR>
	static inline TR Operation(TS part, TA startdate, TB enddate, ValidityMask &mask, idx_t idx) {
		if (Value::IsFinite(startdate) && Value::IsFinite(enddate)) {
			return SubtractDateParts<TA, TB, TR>(GetDatePartSpecifier(part.GetString()), startdate, enddate);
		} else {
			mask.SetInvalid(idx);
			return TR();
		}
	}
};

template <typename TA, typename TB, typename TR>
static void DateSubBinaryExecutor(DatePartSpecifier type, Vector &left, Vector &right, Vector &result, idx_t count) {
	switch (type) {
	case DatePartSpecifier::YEAR:
	case DatePartSpecifier::ISOYEAR:
		DateSub::BinaryExecute<TA, TB, TR, DateSub::YearOperator>(left, right, result, count);
		break;
	case DatePartSpecifier::MONTH:
		DateSub::BinaryExecute<TA, TB, TR, DateSub::MonthOperator>(left, right, result, count);
		break;
	case DatePartSpecifier::DAY:
	case DatePartSpecifier::DOW:
	case DatePartSpecifier::ISODOW:
	case DatePartSpecifier::DOY:
		DateSub::BinaryExecute<TA, TB, TR, DateSub::DayOperator>(left, right, result, count);
		break;
	case DatePartSpecifier::DECADE:
		DateSub::BinaryExecute<TA, TB, TR, DateSub::DecadeOperator>(left, right, result, count);
		break;
	case DatePartSpecifier::CENTURY:
		DateSub::BinaryExecute<TA, TB, TR, DateSub::CenturyOperator>(left, right, result, count);
		break;
	case DatePartSpecifier::MILLENNIUM:
		DateSub::BinaryExecute<TA, TB, TR, DateSub::MilleniumOperator>(left, right, result, count);
		break;
	case DatePartSpecifier::QUARTER:
		DateSub::BinaryExecute<TA, TB, TR, DateSub::QuarterOperator>(left, right, result, count);
		break;
	case DatePartSpecifier::WEEK:
	case DatePartSpecifier::YEARWEEK:
		DateSub::BinaryExecute<TA, TB, TR, DateSub::WeekOperator>(left, right, result, count);
		break;
	case DatePartSpecifier::MICROSECONDS:
		DateSub::BinaryExecute<TA, TB, TR, DateSub::MicrosecondsOperator>(left, right, result, count);
		break;
	case DatePartSpecifier::MILLISECONDS:
		DateSub::BinaryExecute<TA, TB, TR, DateSub::MillisecondsOperator>(left, right, result, count);
		break;
	case DatePartSpecifier::SECOND:
	case DatePartSpecifier::EPOCH:
		DateSub::BinaryExecute<TA, TB, TR, DateSub::SecondsOperator>(left, right, result, count);
		break;
	case DatePartSpecifier::MINUTE:
		DateSub::BinaryExecute<TA, TB, TR, DateSub::MinutesOperator>(left, right, result, count);
		break;
	case DatePartSpecifier::HOUR:
		DateSub::BinaryExecute<TA, TB, TR, DateSub::HoursOperator>(left, right, result, count);
		break;
	default:
		throw NotImplementedException("Specifier type not implemented for DATESUB");
	}
}

template <typename T>
static void DateSubFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(args.ColumnCount() == 3);
	auto &part_arg = args.data[0];
	auto &start_arg = args.data[1];
	auto &end_arg = args.data[2];

	if (part_arg.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		// Common case of constant part.
		if (ConstantVector::IsNull(part_arg)) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
			ConstantVector::SetNull(result, true);
		} else {
			const auto type = GetDatePartSpecifier(ConstantVector::GetData<string_t>(part_arg)->GetString());
			DateSubBinaryExecutor<T, T, int64_t>(type, start_arg, end_arg, result, args.size());
		}
	} else {
		TernaryExecutor::ExecuteWithNulls<string_t, T, T, int64_t>(
		    part_arg, start_arg, end_arg, result, args.size(),
		    DateSubTernaryOperator::Operation<string_t, T, T, int64_t>);
	}
}

ScalarFunctionSet DateSubFun::GetFunctions() {
	ScalarFunctionSet date_sub("date_sub");
	date_sub.AddFunction(ScalarFunction({LogicalType::VARCHAR, LogicalType::DATE, LogicalType::DATE},
	                                    LogicalType::BIGINT, DateSubFunction<date_t>));
	date_sub.AddFunction(ScalarFunction({LogicalType::VARCHAR, LogicalType::TIMESTAMP, LogicalType::TIMESTAMP},
	                                    LogicalType::BIGINT, DateSubFunction<timestamp_t>));
	date_sub.AddFunction(ScalarFunction({LogicalType::VARCHAR, LogicalType::TIME, LogicalType::TIME},
	                                    LogicalType::BIGINT, DateSubFunction<dtime_t>));
	return date_sub;
}

} // namespace duckdb










namespace duckdb {

struct DateTrunc {
	template <class TA, class TR, class OP>
	static inline TR UnaryFunction(TA input) {
		if (Value::IsFinite(input)) {
			return OP::template Operation<TA, TR>(input);
		} else {
			return Cast::template Operation<TA, TR>(input);
		}
	}

	template <class TA, class TR, class OP>
	static inline void UnaryExecute(Vector &left, Vector &result, idx_t count) {
		UnaryExecutor::Execute<TA, TR>(left, result, count, UnaryFunction<TA, TR, OP>);
	}

	struct MillenniumOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			return Date::FromDate((Date::ExtractYear(input) / 1000) * 1000, 1, 1);
		}
	};

	struct CenturyOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			return Date::FromDate((Date::ExtractYear(input) / 100) * 100, 1, 1);
		}
	};

	struct DecadeOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			return Date::FromDate((Date::ExtractYear(input) / 10) * 10, 1, 1);
		}
	};

	struct YearOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			return Date::FromDate(Date::ExtractYear(input), 1, 1);
		}
	};

	struct QuarterOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			int32_t yyyy, mm, dd;
			Date::Convert(input, yyyy, mm, dd);
			mm = 1 + (((mm - 1) / 3) * 3);
			return Date::FromDate(yyyy, mm, 1);
		}
	};

	struct MonthOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			return Date::FromDate(Date::ExtractYear(input), Date::ExtractMonth(input), 1);
		}
	};

	struct WeekOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			return Date::GetMondayOfCurrentWeek(input);
		}
	};

	struct ISOYearOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			date_t date = Date::GetMondayOfCurrentWeek(input);
			date.days -= (Date::ExtractISOWeekNumber(date) - 1) * Interval::DAYS_PER_WEEK;

			return date;
		}
	};

	struct DayOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			return input;
		}
	};

	struct HourOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			int32_t hour, min, sec, micros;
			date_t date;
			dtime_t time;
			Timestamp::Convert(input, date, time);
			Time::Convert(time, hour, min, sec, micros);
			return Timestamp::FromDatetime(date, Time::FromTime(hour, 0, 0, 0));
		}
	};

	struct MinuteOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			int32_t hour, min, sec, micros;
			date_t date;
			dtime_t time;
			Timestamp::Convert(input, date, time);
			Time::Convert(time, hour, min, sec, micros);
			return Timestamp::FromDatetime(date, Time::FromTime(hour, min, 0, 0));
		}
	};

	struct SecondOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			int32_t hour, min, sec, micros;
			date_t date;
			dtime_t time;
			Timestamp::Convert(input, date, time);
			Time::Convert(time, hour, min, sec, micros);
			return Timestamp::FromDatetime(date, Time::FromTime(hour, min, sec, 0));
		}
	};

	struct MillisecondOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			int32_t hour, min, sec, micros;
			date_t date;
			dtime_t time;
			Timestamp::Convert(input, date, time);
			Time::Convert(time, hour, min, sec, micros);
			micros -= micros % Interval::MICROS_PER_MSEC;
			return Timestamp::FromDatetime(date, Time::FromTime(hour, min, sec, micros));
		}
	};

	struct MicrosecondOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			return input;
		}
	};
};

// DATE specialisations
template <>
date_t DateTrunc::MillenniumOperator::Operation(timestamp_t input) {
	return MillenniumOperator::Operation<date_t, date_t>(Timestamp::GetDate(input));
}

template <>
timestamp_t DateTrunc::MillenniumOperator::Operation(date_t input) {
	return Timestamp::FromDatetime(MillenniumOperator::Operation<date_t, date_t>(input), dtime_t(0));
}

template <>
timestamp_t DateTrunc::MillenniumOperator::Operation(timestamp_t input) {
	return MillenniumOperator::Operation<date_t, timestamp_t>(Timestamp::GetDate(input));
}

template <>
date_t DateTrunc::CenturyOperator::Operation(timestamp_t input) {
	return CenturyOperator::Operation<date_t, date_t>(Timestamp::GetDate(input));
}

template <>
timestamp_t DateTrunc::CenturyOperator::Operation(date_t input) {
	return Timestamp::FromDatetime(CenturyOperator::Operation<date_t, date_t>(input), dtime_t(0));
}

template <>
timestamp_t DateTrunc::CenturyOperator::Operation(timestamp_t input) {
	return CenturyOperator::Operation<date_t, timestamp_t>(Timestamp::GetDate(input));
}

template <>
date_t DateTrunc::DecadeOperator::Operation(timestamp_t input) {
	return DecadeOperator::Operation<date_t, date_t>(Timestamp::GetDate(input));
}

template <>
timestamp_t DateTrunc::DecadeOperator::Operation(date_t input) {
	return Timestamp::FromDatetime(DecadeOperator::Operation<date_t, date_t>(input), dtime_t(0));
}

template <>
timestamp_t DateTrunc::DecadeOperator::Operation(timestamp_t input) {
	return DecadeOperator::Operation<date_t, timestamp_t>(Timestamp::GetDate(input));
}

template <>
date_t DateTrunc::YearOperator::Operation(timestamp_t input) {
	return YearOperator::Operation<date_t, date_t>(Timestamp::GetDate(input));
}

template <>
timestamp_t DateTrunc::YearOperator::Operation(date_t input) {
	return Timestamp::FromDatetime(YearOperator::Operation<date_t, date_t>(input), dtime_t(0));
}

template <>
timestamp_t DateTrunc::YearOperator::Operation(timestamp_t input) {
	return YearOperator::Operation<date_t, timestamp_t>(Timestamp::GetDate(input));
}

template <>
date_t DateTrunc::QuarterOperator::Operation(timestamp_t input) {
	return QuarterOperator::Operation<date_t, date_t>(Timestamp::GetDate(input));
}

template <>
timestamp_t DateTrunc::QuarterOperator::Operation(date_t input) {
	return Timestamp::FromDatetime(QuarterOperator::Operation<date_t, date_t>(input), dtime_t(0));
}

template <>
timestamp_t DateTrunc::QuarterOperator::Operation(timestamp_t input) {
	return QuarterOperator::Operation<date_t, timestamp_t>(Timestamp::GetDate(input));
}

template <>
date_t DateTrunc::MonthOperator::Operation(timestamp_t input) {
	return MonthOperator::Operation<date_t, date_t>(Timestamp::GetDate(input));
}

template <>
timestamp_t DateTrunc::MonthOperator::Operation(date_t input) {
	return Timestamp::FromDatetime(MonthOperator::Operation<date_t, date_t>(input), dtime_t(0));
}

template <>
timestamp_t DateTrunc::MonthOperator::Operation(timestamp_t input) {
	return MonthOperator::Operation<date_t, timestamp_t>(Timestamp::GetDate(input));
}

template <>
date_t DateTrunc::WeekOperator::Operation(timestamp_t input) {
	return WeekOperator::Operation<date_t, date_t>(Timestamp::GetDate(input));
}

template <>
timestamp_t DateTrunc::WeekOperator::Operation(date_t input) {
	return Timestamp::FromDatetime(WeekOperator::Operation<date_t, date_t>(input), dtime_t(0));
}

template <>
timestamp_t DateTrunc::WeekOperator::Operation(timestamp_t input) {
	return WeekOperator::Operation<date_t, timestamp_t>(Timestamp::GetDate(input));
}

template <>
date_t DateTrunc::ISOYearOperator::Operation(timestamp_t input) {
	return ISOYearOperator::Operation<date_t, date_t>(Timestamp::GetDate(input));
}

template <>
timestamp_t DateTrunc::ISOYearOperator::Operation(date_t input) {
	return Timestamp::FromDatetime(ISOYearOperator::Operation<date_t, date_t>(input), dtime_t(0));
}

template <>
timestamp_t DateTrunc::ISOYearOperator::Operation(timestamp_t input) {
	return ISOYearOperator::Operation<date_t, timestamp_t>(Timestamp::GetDate(input));
}

template <>
date_t DateTrunc::DayOperator::Operation(timestamp_t input) {
	return DayOperator::Operation<date_t, date_t>(Timestamp::GetDate(input));
}

template <>
timestamp_t DateTrunc::DayOperator::Operation(date_t input) {
	return Timestamp::FromDatetime(DayOperator::Operation<date_t, date_t>(input), dtime_t(0));
}

template <>
timestamp_t DateTrunc::DayOperator::Operation(timestamp_t input) {
	return DayOperator::Operation<date_t, timestamp_t>(Timestamp::GetDate(input));
}

template <>
date_t DateTrunc::HourOperator::Operation(date_t input) {
	return DayOperator::Operation<date_t, date_t>(input);
}

template <>
timestamp_t DateTrunc::HourOperator::Operation(date_t input) {
	return DayOperator::Operation<date_t, timestamp_t>(input);
}

template <>
date_t DateTrunc::HourOperator::Operation(timestamp_t input) {
	return Timestamp::GetDate(HourOperator::Operation<timestamp_t, timestamp_t>(input));
}

template <>
date_t DateTrunc::MinuteOperator::Operation(date_t input) {
	return DayOperator::Operation<date_t, date_t>(input);
}

template <>
timestamp_t DateTrunc::MinuteOperator::Operation(date_t input) {
	return DayOperator::Operation<date_t, timestamp_t>(input);
}

template <>
date_t DateTrunc::MinuteOperator::Operation(timestamp_t input) {
	return Timestamp::GetDate(HourOperator::Operation<timestamp_t, timestamp_t>(input));
}

template <>
date_t DateTrunc::SecondOperator::Operation(date_t input) {
	return DayOperator::Operation<date_t, date_t>(input);
}

template <>
timestamp_t DateTrunc::SecondOperator::Operation(date_t input) {
	return DayOperator::Operation<date_t, timestamp_t>(input);
}

template <>
date_t DateTrunc::SecondOperator::Operation(timestamp_t input) {
	return Timestamp::GetDate(DayOperator::Operation<timestamp_t, timestamp_t>(input));
}

template <>
date_t DateTrunc::MillisecondOperator::Operation(date_t input) {
	return DayOperator::Operation<date_t, date_t>(input);
}

template <>
timestamp_t DateTrunc::MillisecondOperator::Operation(date_t input) {
	return DayOperator::Operation<date_t, timestamp_t>(input);
}

template <>
date_t DateTrunc::MillisecondOperator::Operation(timestamp_t input) {
	return Timestamp::GetDate(MillisecondOperator::Operation<timestamp_t, timestamp_t>(input));
}

template <>
date_t DateTrunc::MicrosecondOperator::Operation(date_t input) {
	return DayOperator::Operation<date_t, date_t>(input);
}

template <>
timestamp_t DateTrunc::MicrosecondOperator::Operation(date_t input) {
	return DayOperator::Operation<date_t, timestamp_t>(input);
}

template <>
date_t DateTrunc::MicrosecondOperator::Operation(timestamp_t input) {
	return Timestamp::GetDate(MicrosecondOperator::Operation<timestamp_t, timestamp_t>(input));
}

// INTERVAL specialisations
template <>
interval_t DateTrunc::MillenniumOperator::Operation(interval_t input) {
	input.days = 0;
	input.micros = 0;
	input.months = (input.months / Interval::MONTHS_PER_MILLENIUM) * Interval::MONTHS_PER_MILLENIUM;
	return input;
}

template <>
interval_t DateTrunc::CenturyOperator::Operation(interval_t input) {
	input.days = 0;
	input.micros = 0;
	input.months = (input.months / Interval::MONTHS_PER_CENTURY) * Interval::MONTHS_PER_CENTURY;
	return input;
}

template <>
interval_t DateTrunc::DecadeOperator::Operation(interval_t input) {
	input.days = 0;
	input.micros = 0;
	input.months = (input.months / Interval::MONTHS_PER_DECADE) * Interval::MONTHS_PER_DECADE;
	return input;
}

template <>
interval_t DateTrunc::YearOperator::Operation(interval_t input) {
	input.days = 0;
	input.micros = 0;
	input.months = (input.months / Interval::MONTHS_PER_YEAR) * Interval::MONTHS_PER_YEAR;
	return input;
}

template <>
interval_t DateTrunc::QuarterOperator::Operation(interval_t input) {
	input.days = 0;
	input.micros = 0;
	input.months = (input.months / Interval::MONTHS_PER_QUARTER) * Interval::MONTHS_PER_QUARTER;
	return input;
}

template <>
interval_t DateTrunc::MonthOperator::Operation(interval_t input) {
	input.days = 0;
	input.micros = 0;
	return input;
}

template <>
interval_t DateTrunc::WeekOperator::Operation(interval_t input) {
	input.micros = 0;
	input.days = (input.days / Interval::DAYS_PER_WEEK) * Interval::DAYS_PER_WEEK;
	return input;
}

template <>
interval_t DateTrunc::ISOYearOperator::Operation(interval_t input) {
	return YearOperator::Operation<interval_t, interval_t>(input);
}

template <>
interval_t DateTrunc::DayOperator::Operation(interval_t input) {
	input.micros = 0;
	return input;
}

template <>
interval_t DateTrunc::HourOperator::Operation(interval_t input) {
	input.micros = (input.micros / Interval::MICROS_PER_HOUR) * Interval::MICROS_PER_HOUR;
	return input;
}

template <>
interval_t DateTrunc::MinuteOperator::Operation(interval_t input) {
	input.micros = (input.micros / Interval::MICROS_PER_MINUTE) * Interval::MICROS_PER_MINUTE;
	return input;
}

template <>
interval_t DateTrunc::SecondOperator::Operation(interval_t input) {
	input.micros = (input.micros / Interval::MICROS_PER_SEC) * Interval::MICROS_PER_SEC;
	return input;
}

template <>
interval_t DateTrunc::MillisecondOperator::Operation(interval_t input) {
	input.micros = (input.micros / Interval::MICROS_PER_MSEC) * Interval::MICROS_PER_MSEC;
	return input;
}

template <>
interval_t DateTrunc::MicrosecondOperator::Operation(interval_t input) {
	return input;
}

template <class TA, class TR>
static TR TruncateElement(DatePartSpecifier type, TA element) {
	if (!Value::IsFinite(element)) {
		return Cast::template Operation<TA, TR>(element);
	}

	switch (type) {
	case DatePartSpecifier::MILLENNIUM:
		return DateTrunc::MillenniumOperator::Operation<TA, TR>(element);
	case DatePartSpecifier::CENTURY:
		return DateTrunc::CenturyOperator::Operation<TA, TR>(element);
	case DatePartSpecifier::DECADE:
		return DateTrunc::DecadeOperator::Operation<TA, TR>(element);
	case DatePartSpecifier::YEAR:
		return DateTrunc::YearOperator::Operation<TA, TR>(element);
	case DatePartSpecifier::QUARTER:
		return DateTrunc::QuarterOperator::Operation<TA, TR>(element);
	case DatePartSpecifier::MONTH:
		return DateTrunc::MonthOperator::Operation<TA, TR>(element);
	case DatePartSpecifier::WEEK:
	case DatePartSpecifier::YEARWEEK:
		return DateTrunc::WeekOperator::Operation<TA, TR>(element);
	case DatePartSpecifier::ISOYEAR:
		return DateTrunc::ISOYearOperator::Operation<TA, TR>(element);
	case DatePartSpecifier::DAY:
	case DatePartSpecifier::DOW:
	case DatePartSpecifier::ISODOW:
	case DatePartSpecifier::DOY:
		return DateTrunc::DayOperator::Operation<TA, TR>(element);
	case DatePartSpecifier::HOUR:
		return DateTrunc::HourOperator::Operation<TA, TR>(element);
	case DatePartSpecifier::MINUTE:
		return DateTrunc::MinuteOperator::Operation<TA, TR>(element);
	case DatePartSpecifier::SECOND:
	case DatePartSpecifier::EPOCH:
		return DateTrunc::SecondOperator::Operation<TA, TR>(element);
	case DatePartSpecifier::MILLISECONDS:
		return DateTrunc::MillisecondOperator::Operation<TA, TR>(element);
	case DatePartSpecifier::MICROSECONDS:
		return DateTrunc::MicrosecondOperator::Operation<TA, TR>(element);
	default:
		throw NotImplementedException("Specifier type not implemented for DATETRUNC");
	}
}

struct DateTruncBinaryOperator {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA specifier, TB date) {
		return TruncateElement<TB, TR>(GetDatePartSpecifier(specifier.GetString()), date);
	}
};

template <typename TA, typename TR>
static void DateTruncUnaryExecutor(DatePartSpecifier type, Vector &left, Vector &result, idx_t count) {
	switch (type) {
	case DatePartSpecifier::MILLENNIUM:
		DateTrunc::UnaryExecute<TA, TR, DateTrunc::MillenniumOperator>(left, result, count);
		break;
	case DatePartSpecifier::CENTURY:
		DateTrunc::UnaryExecute<TA, TR, DateTrunc::CenturyOperator>(left, result, count);
		break;
	case DatePartSpecifier::DECADE:
		DateTrunc::UnaryExecute<TA, TR, DateTrunc::DecadeOperator>(left, result, count);
		break;
	case DatePartSpecifier::YEAR:
		DateTrunc::UnaryExecute<TA, TR, DateTrunc::YearOperator>(left, result, count);
		break;
	case DatePartSpecifier::QUARTER:
		DateTrunc::UnaryExecute<TA, TR, DateTrunc::QuarterOperator>(left, result, count);
		break;
	case DatePartSpecifier::MONTH:
		DateTrunc::UnaryExecute<TA, TR, DateTrunc::MonthOperator>(left, result, count);
		break;
	case DatePartSpecifier::WEEK:
	case DatePartSpecifier::YEARWEEK:
		DateTrunc::UnaryExecute<TA, TR, DateTrunc::WeekOperator>(left, result, count);
		break;
	case DatePartSpecifier::ISOYEAR:
		DateTrunc::UnaryExecute<TA, TR, DateTrunc::ISOYearOperator>(left, result, count);
		break;
	case DatePartSpecifier::DAY:
	case DatePartSpecifier::DOW:
	case DatePartSpecifier::ISODOW:
	case DatePartSpecifier::DOY:
		DateTrunc::UnaryExecute<TA, TR, DateTrunc::DayOperator>(left, result, count);
		break;
	case DatePartSpecifier::HOUR:
		DateTrunc::UnaryExecute<TA, TR, DateTrunc::HourOperator>(left, result, count);
		break;
	case DatePartSpecifier::MINUTE:
		DateTrunc::UnaryExecute<TA, TR, DateTrunc::MinuteOperator>(left, result, count);
		break;
	case DatePartSpecifier::SECOND:
	case DatePartSpecifier::EPOCH:
		DateTrunc::UnaryExecute<TA, TR, DateTrunc::SecondOperator>(left, result, count);
		break;
	case DatePartSpecifier::MILLISECONDS:
		DateTrunc::UnaryExecute<TA, TR, DateTrunc::MillisecondOperator>(left, result, count);
		break;
	case DatePartSpecifier::MICROSECONDS:
		DateTrunc::UnaryExecute<TA, TR, DateTrunc::MicrosecondOperator>(left, result, count);
		break;
	default:
		throw NotImplementedException("Specifier type not implemented for DATETRUNC");
	}
}

template <typename TA, typename TR>
static void DateTruncFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(args.ColumnCount() == 2);
	auto &part_arg = args.data[0];
	auto &date_arg = args.data[1];

	if (part_arg.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		// Common case of constant part.
		if (ConstantVector::IsNull(part_arg)) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
			ConstantVector::SetNull(result, true);
		} else {
			const auto type = GetDatePartSpecifier(ConstantVector::GetData<string_t>(part_arg)->GetString());
			DateTruncUnaryExecutor<TA, TR>(type, date_arg, result, args.size());
		}
	} else {
		BinaryExecutor::ExecuteStandard<string_t, TA, TR, DateTruncBinaryOperator>(part_arg, date_arg, result,
		                                                                           args.size());
	}
}

template <class TA, class TR, class OP>
static unique_ptr<BaseStatistics> DateTruncStatistics(vector<BaseStatistics> &child_stats) {
	// we can only propagate date stats if the child has stats
	auto &nstats = child_stats[1];
	if (!NumericStats::HasMinMax(nstats)) {
		return nullptr;
	}
	// run the operator on both the min and the max, this gives us the [min, max] bound
	auto min = NumericStats::GetMin<TA>(nstats);
	auto max = NumericStats::GetMax<TA>(nstats);
	if (min > max) {
		return nullptr;
	}

	// Infinite values are unmodified
	auto min_part = DateTrunc::UnaryFunction<TA, TR, OP>(min);
	auto max_part = DateTrunc::UnaryFunction<TA, TR, OP>(max);

	auto min_value = Value::CreateValue(min_part);
	auto max_value = Value::CreateValue(max_part);
	auto result = NumericStats::CreateEmpty(min_value.type());
	NumericStats::SetMin(result, min_value);
	NumericStats::SetMax(result, max_value);
	result.CopyValidity(child_stats[0]);
	return result.ToUnique();
}

template <class TA, class TR, class OP>
static unique_ptr<BaseStatistics> PropagateDateTruncStatistics(ClientContext &context, FunctionStatisticsInput &input) {
	return DateTruncStatistics<TA, TR, OP>(input.child_stats);
}

template <typename TA, typename TR>
static function_statistics_t DateTruncStats(DatePartSpecifier type) {
	switch (type) {
	case DatePartSpecifier::MILLENNIUM:
		return PropagateDateTruncStatistics<TA, TR, DateTrunc::MillenniumOperator>;
	case DatePartSpecifier::CENTURY:
		return PropagateDateTruncStatistics<TA, TR, DateTrunc::CenturyOperator>;
	case DatePartSpecifier::DECADE:
		return PropagateDateTruncStatistics<TA, TR, DateTrunc::DecadeOperator>;
	case DatePartSpecifier::YEAR:
		return PropagateDateTruncStatistics<TA, TR, DateTrunc::YearOperator>;
	case DatePartSpecifier::QUARTER:
		return PropagateDateTruncStatistics<TA, TR, DateTrunc::QuarterOperator>;
	case DatePartSpecifier::MONTH:
		return PropagateDateTruncStatistics<TA, TR, DateTrunc::MonthOperator>;
	case DatePartSpecifier::WEEK:
	case DatePartSpecifier::YEARWEEK:
		return PropagateDateTruncStatistics<TA, TR, DateTrunc::WeekOperator>;
	case DatePartSpecifier::ISOYEAR:
		return PropagateDateTruncStatistics<TA, TR, DateTrunc::ISOYearOperator>;
	case DatePartSpecifier::DAY:
	case DatePartSpecifier::DOW:
	case DatePartSpecifier::ISODOW:
	case DatePartSpecifier::DOY:
		return PropagateDateTruncStatistics<TA, TR, DateTrunc::DayOperator>;
	case DatePartSpecifier::HOUR:
		return PropagateDateTruncStatistics<TA, TR, DateTrunc::HourOperator>;
	case DatePartSpecifier::MINUTE:
		return PropagateDateTruncStatistics<TA, TR, DateTrunc::MinuteOperator>;
	case DatePartSpecifier::SECOND:
	case DatePartSpecifier::EPOCH:
		return PropagateDateTruncStatistics<TA, TR, DateTrunc::SecondOperator>;
	case DatePartSpecifier::MILLISECONDS:
		return PropagateDateTruncStatistics<TA, TR, DateTrunc::MillisecondOperator>;
	case DatePartSpecifier::MICROSECONDS:
		return PropagateDateTruncStatistics<TA, TR, DateTrunc::MicrosecondOperator>;
	default:
		throw NotImplementedException("Specifier type not implemented for DATETRUNC statistics");
	}
}

static unique_ptr<FunctionData> DateTruncBind(ClientContext &context, ScalarFunction &bound_function,
                                              vector<unique_ptr<Expression>> &arguments) {
	if (!arguments[0]->IsFoldable()) {
		return nullptr;
	}

	// Rebind to return a date if we are truncating that far
	Value part_value = ExpressionExecutor::EvaluateScalar(context, *arguments[0]);
	if (part_value.IsNull()) {
		return nullptr;
	}
	const auto part_name = part_value.ToString();
	const auto part_code = GetDatePartSpecifier(part_name);
	switch (part_code) {
	case DatePartSpecifier::MILLENNIUM:
	case DatePartSpecifier::CENTURY:
	case DatePartSpecifier::DECADE:
	case DatePartSpecifier::YEAR:
	case DatePartSpecifier::QUARTER:
	case DatePartSpecifier::MONTH:
	case DatePartSpecifier::WEEK:
	case DatePartSpecifier::YEARWEEK:
	case DatePartSpecifier::ISOYEAR:
	case DatePartSpecifier::DAY:
	case DatePartSpecifier::DOW:
	case DatePartSpecifier::ISODOW:
	case DatePartSpecifier::DOY:
		switch (bound_function.arguments[1].id()) {
		case LogicalType::TIMESTAMP:
			bound_function.function = DateTruncFunction<timestamp_t, date_t>;
			bound_function.statistics = DateTruncStats<timestamp_t, date_t>(part_code);
			break;
		case LogicalType::DATE:
			bound_function.function = DateTruncFunction<date_t, date_t>;
			bound_function.statistics = DateTruncStats<date_t, date_t>(part_code);
			break;
		default:
			throw NotImplementedException("Temporal argument type for DATETRUNC");
		}
		bound_function.return_type = LogicalType::DATE;
		break;
	default:
		switch (bound_function.arguments[1].id()) {
		case LogicalType::TIMESTAMP:
			bound_function.statistics = DateTruncStats<timestamp_t, timestamp_t>(part_code);
			break;
		case LogicalType::DATE:
			bound_function.statistics = DateTruncStats<date_t, timestamp_t>(part_code);
			break;
		default:
			throw NotImplementedException("Temporal argument type for DATETRUNC");
		}
		break;
	}

	return nullptr;
}

ScalarFunctionSet DateTruncFun::GetFunctions() {
	ScalarFunctionSet date_trunc("date_trunc");
	date_trunc.AddFunction(ScalarFunction({LogicalType::VARCHAR, LogicalType::TIMESTAMP}, LogicalType::TIMESTAMP,
	                                      DateTruncFunction<timestamp_t, timestamp_t>, DateTruncBind));
	date_trunc.AddFunction(ScalarFunction({LogicalType::VARCHAR, LogicalType::DATE}, LogicalType::TIMESTAMP,
	                                      DateTruncFunction<date_t, timestamp_t>, DateTruncBind));
	date_trunc.AddFunction(ScalarFunction({LogicalType::VARCHAR, LogicalType::INTERVAL}, LogicalType::INTERVAL,
	                                      DateTruncFunction<interval_t, interval_t>));
	return date_trunc;
}

} // namespace duckdb







namespace duckdb {

struct EpochSecOperator {
	template <class INPUT_TYPE, class RESULT_TYPE>
	static RESULT_TYPE Operation(INPUT_TYPE input) {
		return Timestamp::FromEpochSeconds(input);
	}
};

static void EpochSecFunction(DataChunk &input, ExpressionState &state, Vector &result) {
	D_ASSERT(input.ColumnCount() == 1);

	UnaryExecutor::Execute<int64_t, timestamp_t, EpochSecOperator>(input.data[0], result, input.size());
}

struct EpochMillisOperator {
	template <class INPUT_TYPE, class RESULT_TYPE>
	static RESULT_TYPE Operation(INPUT_TYPE input) {
		return Timestamp::FromEpochMs(input);
	}
};

static void EpochMillisFunction(DataChunk &input, ExpressionState &state, Vector &result) {
	D_ASSERT(input.ColumnCount() == 1);

	UnaryExecutor::Execute<int64_t, timestamp_t, EpochMillisOperator>(input.data[0], result, input.size());
}

ScalarFunction EpochMsFun::GetFunction() {
	return ScalarFunction({LogicalType::BIGINT}, LogicalType::TIMESTAMP, EpochMillisFunction);
}

ScalarFunction ToTimestampFun::GetFunction() {
	// to_timestamp is an alias from Postgres that converts the time in seconds to a timestamp
	return ScalarFunction({LogicalType::BIGINT}, LogicalType::TIMESTAMP, EpochSecFunction);
}

} // namespace duckdb







#include <cmath>

namespace duckdb {

struct MakeDateOperator {
	template <typename YYYY, typename MM, typename DD, typename RESULT_TYPE>
	static RESULT_TYPE Operation(YYYY yyyy, MM mm, DD dd) {
		return Date::FromDate(yyyy, mm, dd);
	}
};

template <typename T>
static void ExecuteMakeDate(DataChunk &input, ExpressionState &state, Vector &result) {
	D_ASSERT(input.ColumnCount() == 3);
	auto &yyyy = input.data[0];
	auto &mm = input.data[1];
	auto &dd = input.data[2];

	TernaryExecutor::Execute<T, T, T, date_t>(yyyy, mm, dd, result, input.size(),
	                                          MakeDateOperator::Operation<T, T, T, date_t>);
}

template <typename T>
static void ExecuteStructMakeDate(DataChunk &input, ExpressionState &state, Vector &result) {
	// this should be guaranteed by the binder
	D_ASSERT(input.ColumnCount() == 1);
	auto &vec = input.data[0];

	auto &children = StructVector::GetEntries(vec);
	D_ASSERT(children.size() == 3);
	auto &yyyy = *children[0];
	auto &mm = *children[1];
	auto &dd = *children[2];

	TernaryExecutor::Execute<T, T, T, date_t>(yyyy, mm, dd, result, input.size(), Date::FromDate);
}

struct MakeTimeOperator {
	template <typename HH, typename MM, typename SS, typename RESULT_TYPE>
	static RESULT_TYPE Operation(HH hh, MM mm, SS ss) {
		int64_t secs = ss;
		int64_t micros = std::round((ss - secs) * Interval::MICROS_PER_SEC);
		return Time::FromTime(hh, mm, secs, micros);
	}
};

template <typename T>
static void ExecuteMakeTime(DataChunk &input, ExpressionState &state, Vector &result) {
	D_ASSERT(input.ColumnCount() == 3);
	auto &yyyy = input.data[0];
	auto &mm = input.data[1];
	auto &dd = input.data[2];

	TernaryExecutor::Execute<T, T, double, dtime_t>(yyyy, mm, dd, result, input.size(),
	                                                MakeTimeOperator::Operation<T, T, double, dtime_t>);
}

struct MakeTimestampOperator {
	template <typename YYYY, typename MM, typename DD, typename HR, typename MN, typename SS, typename RESULT_TYPE>
	static RESULT_TYPE Operation(YYYY yyyy, MM mm, DD dd, HR hr, MN mn, SS ss) {
		const auto d = MakeDateOperator::Operation<YYYY, MM, DD, date_t>(yyyy, mm, dd);
		const auto t = MakeTimeOperator::Operation<HR, MN, SS, dtime_t>(hr, mn, ss);
		return Timestamp::FromDatetime(d, t);
	}
};

template <typename T>
static void ExecuteMakeTimestamp(DataChunk &input, ExpressionState &state, Vector &result) {
	D_ASSERT(input.ColumnCount() == 6);

	auto func = MakeTimestampOperator::Operation<T, T, T, T, T, double, timestamp_t>;
	SenaryExecutor::Execute<T, T, T, T, T, double, timestamp_t>(input, result, func);
}

ScalarFunctionSet MakeDateFun::GetFunctions() {
	ScalarFunctionSet make_date("make_date");
	make_date.AddFunction(ScalarFunction({LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT},
	                                     LogicalType::DATE, ExecuteMakeDate<int64_t>));

	child_list_t<LogicalType> make_date_children {
	    {"year", LogicalType::BIGINT}, {"month", LogicalType::BIGINT}, {"day", LogicalType::BIGINT}};
	make_date.AddFunction(
	    ScalarFunction({LogicalType::STRUCT(make_date_children)}, LogicalType::DATE, ExecuteStructMakeDate<int64_t>));
	return make_date;
}

ScalarFunction MakeTimeFun::GetFunction() {
	return ScalarFunction({LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::DOUBLE}, LogicalType::TIME,
	                      ExecuteMakeTime<int64_t>);
}

ScalarFunction MakeTimestampFun::GetFunction() {
	return ScalarFunction({LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT,
	                       LogicalType::BIGINT, LogicalType::DOUBLE},
	                      LogicalType::TIMESTAMP, ExecuteMakeTimestamp<int64_t>);
}

} // namespace duckdb








#include <cctype>
#include <utility>

namespace duckdb {

struct StrfTimeBindData : public FunctionData {
	explicit StrfTimeBindData(StrfTimeFormat format_p, string format_string_p, bool is_null)
	    : format(std::move(format_p)), format_string(std::move(format_string_p)), is_null(is_null) {
	}

	StrfTimeFormat format;
	string format_string;
	bool is_null;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<StrfTimeBindData>(format, format_string, is_null);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = (const StrfTimeBindData &)other_p;
		return format_string == other.format_string;
	}
};

template <bool REVERSED>
static unique_ptr<FunctionData> StrfTimeBindFunction(ClientContext &context, ScalarFunction &bound_function,
                                                     vector<unique_ptr<Expression>> &arguments) {
	auto format_idx = REVERSED ? 0 : 1;
	auto &format_arg = arguments[format_idx];
	if (format_arg->HasParameter()) {
		throw ParameterNotResolvedException();
	}
	if (!format_arg->IsFoldable()) {
		throw InvalidInputException("strftime format must be a constant");
	}
	Value options_str = ExpressionExecutor::EvaluateScalar(context, *format_arg);
	auto format_string = options_str.GetValue<string>();
	StrfTimeFormat format;
	bool is_null = options_str.IsNull();
	if (!is_null) {
		string error = StrTimeFormat::ParseFormatSpecifier(format_string, format);
		if (!error.empty()) {
			throw InvalidInputException("Failed to parse format specifier %s: %s", format_string, error);
		}
	}
	return make_uniq<StrfTimeBindData>(format, format_string, is_null);
}

template <bool REVERSED>
static void StrfTimeFunctionDate(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &info = func_expr.bind_info->Cast<StrfTimeBindData>();

	if (info.is_null) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
		ConstantVector::SetNull(result, true);
		return;
	}
	info.format.ConvertDateVector(args.data[REVERSED ? 1 : 0], result, args.size());
}

template <bool REVERSED>
static void StrfTimeFunctionTimestamp(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &info = func_expr.bind_info->Cast<StrfTimeBindData>();

	if (info.is_null) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
		ConstantVector::SetNull(result, true);
		return;
	}
	info.format.ConvertTimestampVector(args.data[REVERSED ? 1 : 0], result, args.size());
}

ScalarFunctionSet StrfTimeFun::GetFunctions() {
	ScalarFunctionSet strftime;

	strftime.AddFunction(ScalarFunction({LogicalType::DATE, LogicalType::VARCHAR}, LogicalType::VARCHAR,
	                                    StrfTimeFunctionDate<false>, StrfTimeBindFunction<false>));
	strftime.AddFunction(ScalarFunction({LogicalType::TIMESTAMP, LogicalType::VARCHAR}, LogicalType::VARCHAR,
	                                    StrfTimeFunctionTimestamp<false>, StrfTimeBindFunction<false>));
	strftime.AddFunction(ScalarFunction({LogicalType::VARCHAR, LogicalType::DATE}, LogicalType::VARCHAR,
	                                    StrfTimeFunctionDate<true>, StrfTimeBindFunction<true>));
	strftime.AddFunction(ScalarFunction({LogicalType::VARCHAR, LogicalType::TIMESTAMP}, LogicalType::VARCHAR,
	                                    StrfTimeFunctionTimestamp<true>, StrfTimeBindFunction<true>));
	return strftime;
}

struct StrpTimeBindData : public FunctionData {
	StrpTimeBindData(const StrpTimeFormat &format, const string &format_string)
	    : formats(1, format), format_strings(1, format_string) {
	}

	StrpTimeBindData(vector<StrpTimeFormat> formats_p, vector<string> format_strings_p)
	    : formats(std::move(formats_p)), format_strings(std::move(format_strings_p)) {
	}

	vector<StrpTimeFormat> formats;
	vector<string> format_strings;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<StrpTimeBindData>(formats, format_strings);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = (const StrpTimeBindData &)other_p;
		return format_strings == other.format_strings;
	}
};

static unique_ptr<FunctionData> StrpTimeBindFunction(ClientContext &context, ScalarFunction &bound_function,
                                                     vector<unique_ptr<Expression>> &arguments) {
	if (arguments[1]->HasParameter()) {
		throw ParameterNotResolvedException();
	}
	if (!arguments[1]->IsFoldable()) {
		throw InvalidInputException("strptime format must be a constant");
	}
	Value format_value = ExpressionExecutor::EvaluateScalar(context, *arguments[1]);
	string format_string;
	StrpTimeFormat format;
	if (format_value.IsNull()) {
		return make_uniq<StrpTimeBindData>(format, format_string);
	} else if (format_value.type().id() == LogicalTypeId::VARCHAR) {
		format_string = format_value.ToString();
		format.format_specifier = format_string;
		string error = StrTimeFormat::ParseFormatSpecifier(format_string, format);
		if (!error.empty()) {
			throw InvalidInputException("Failed to parse format specifier %s: %s", format_string, error);
		}
		if (format.HasFormatSpecifier(StrTimeSpecifier::UTC_OFFSET)) {
			bound_function.return_type = LogicalType::TIMESTAMP_TZ;
		}
		return make_uniq<StrpTimeBindData>(format, format_string);
	} else if (format_value.type() == LogicalType::LIST(LogicalType::VARCHAR)) {
		const auto &children = ListValue::GetChildren(format_value);
		if (children.empty()) {
			throw InvalidInputException("strptime format list must not be empty");
		}
		vector<string> format_strings;
		vector<StrpTimeFormat> formats;
		for (const auto &child : children) {
			format_string = child.ToString();
			format.format_specifier = format_string;
			string error = StrTimeFormat::ParseFormatSpecifier(format_string, format);
			if (!error.empty()) {
				throw InvalidInputException("Failed to parse format specifier %s: %s", format_string, error);
			}
			// If any format has UTC offsets, then we have to produce TSTZ
			if (format.HasFormatSpecifier(StrTimeSpecifier::UTC_OFFSET)) {
				bound_function.return_type = LogicalType::TIMESTAMP_TZ;
			}
			format_strings.emplace_back(format_string);
			formats.emplace_back(format);
		}
		return make_uniq<StrpTimeBindData>(formats, format_strings);
	} else {
		throw InvalidInputException("strptime format must be a string");
	}
}

struct StrpTimeFunction {

	static void Parse(DataChunk &args, ExpressionState &state, Vector &result) {
		auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
		auto &info = func_expr.bind_info->Cast<StrpTimeBindData>();

		if (ConstantVector::IsNull(args.data[1])) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
			ConstantVector::SetNull(result, true);
			return;
		}
		UnaryExecutor::Execute<string_t, timestamp_t>(args.data[0], result, args.size(), [&](string_t input) {
			StrpTimeFormat::ParseResult result;
			for (auto &format : info.formats) {
				if (format.Parse(input, result)) {
					return result.ToTimestamp();
				}
			}
			throw InvalidInputException(result.FormatError(input, info.formats[0].format_specifier));
		});
	}

	static void TryParse(DataChunk &args, ExpressionState &state, Vector &result) {
		auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
		auto &info = func_expr.bind_info->Cast<StrpTimeBindData>();

		if (ConstantVector::IsNull(args.data[1])) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
			ConstantVector::SetNull(result, true);
			return;
		}

		UnaryExecutor::ExecuteWithNulls<string_t, timestamp_t>(
		    args.data[0], result, args.size(), [&](string_t input, ValidityMask &mask, idx_t idx) {
			    timestamp_t result;
			    string error;
			    for (auto &format : info.formats) {
				    if (format.TryParseTimestamp(input, result, error)) {
					    return result;
				    }
			    }

			    mask.SetInvalid(idx);
			    return timestamp_t();
		    });
	}
};

ScalarFunctionSet StrpTimeFun::GetFunctions() {
	ScalarFunctionSet strptime;

	const auto list_type = LogicalType::LIST(LogicalType::VARCHAR);
	auto fun = ScalarFunction({LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::TIMESTAMP,
	                          StrpTimeFunction::Parse, StrpTimeBindFunction);
	fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	strptime.AddFunction(fun);

	fun = ScalarFunction({LogicalType::VARCHAR, list_type}, LogicalType::TIMESTAMP, StrpTimeFunction::Parse,
	                     StrpTimeBindFunction);
	fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	strptime.AddFunction(fun);
	return strptime;
}

ScalarFunctionSet TryStrpTimeFun::GetFunctions() {
	ScalarFunctionSet try_strptime;

	const auto list_type = LogicalType::LIST(LogicalType::VARCHAR);
	auto fun = ScalarFunction({LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::TIMESTAMP,
	                          StrpTimeFunction::TryParse, StrpTimeBindFunction);
	fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	try_strptime.AddFunction(fun);

	fun = ScalarFunction({LogicalType::VARCHAR, list_type}, LogicalType::TIMESTAMP, StrpTimeFunction::TryParse,
	                     StrpTimeBindFunction);
	fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	try_strptime.AddFunction(fun);

	return try_strptime;
}

} // namespace duckdb













namespace duckdb {

struct TimeBucket {

	// Use 2000-01-03 00:00:00 (Monday) as origin when bucket_width is days, hours, ... for TimescaleDB compatibility
	// There are 10959 days between 1970-01-01 and 2000-01-03
	constexpr static const int64_t DEFAULT_ORIGIN_MICROS = 10959 * Interval::MICROS_PER_DAY;
	// Use 2000-01-01 as origin when bucket_width is months, years, ... for TimescaleDB compatibility
	// There are 360 months between 1970-01-01 and 2000-01-01
	constexpr static const int32_t DEFAULT_ORIGIN_MONTHS = 360;

	enum struct BucketWidthType { CONVERTIBLE_TO_MICROS, CONVERTIBLE_TO_MONTHS, UNCLASSIFIED };

	static inline BucketWidthType ClassifyBucketWidth(const interval_t bucket_width) {
		if (bucket_width.months == 0 && Interval::GetMicro(bucket_width) > 0) {
			return BucketWidthType::CONVERTIBLE_TO_MICROS;
		} else if (bucket_width.months > 0 && bucket_width.days == 0 && bucket_width.micros == 0) {
			return BucketWidthType::CONVERTIBLE_TO_MONTHS;
		} else {
			return BucketWidthType::UNCLASSIFIED;
		}
	}

	static inline BucketWidthType ClassifyBucketWidthErrorThrow(const interval_t bucket_width) {
		if (bucket_width.months == 0) {
			int64_t bucket_width_micros = Interval::GetMicro(bucket_width);
			if (bucket_width_micros <= 0) {
				throw NotImplementedException("Period must be greater than 0");
			}
			return BucketWidthType::CONVERTIBLE_TO_MICROS;
		} else if (bucket_width.months != 0 && bucket_width.days == 0 && bucket_width.micros == 0) {
			if (bucket_width.months < 0) {
				throw NotImplementedException("Period must be greater than 0");
			}
			return BucketWidthType::CONVERTIBLE_TO_MONTHS;
		} else {
			throw NotImplementedException("Month intervals cannot have day or time component");
		}
	}

	template <typename T>
	static inline int32_t EpochMonths(T ts) {
		date_t ts_date = Cast::template Operation<T, date_t>(ts);
		return (Date::ExtractYear(ts_date) - 1970) * 12 + Date::ExtractMonth(ts_date) - 1;
	}

	static inline timestamp_t WidthConvertibleToMicrosCommon(int64_t bucket_width_micros, int64_t ts_micros,
	                                                         int64_t origin_micros) {
		origin_micros %= bucket_width_micros;
		ts_micros = SubtractOperatorOverflowCheck::Operation<int64_t, int64_t, int64_t>(ts_micros, origin_micros);

		int64_t result_micros = (ts_micros / bucket_width_micros) * bucket_width_micros;
		if (ts_micros < 0 && ts_micros % bucket_width_micros != 0) {
			result_micros =
			    SubtractOperatorOverflowCheck::Operation<int64_t, int64_t, int64_t>(result_micros, bucket_width_micros);
		}
		result_micros += origin_micros;

		return Timestamp::FromEpochMicroSeconds(result_micros);
	}

	static inline date_t WidthConvertibleToMonthsCommon(int32_t bucket_width_months, int32_t ts_months,
	                                                    int32_t origin_months) {
		origin_months %= bucket_width_months;
		ts_months = SubtractOperatorOverflowCheck::Operation<int32_t, int32_t, int32_t>(ts_months, origin_months);

		int32_t result_months = (ts_months / bucket_width_months) * bucket_width_months;
		if (ts_months < 0 && ts_months % bucket_width_months != 0) {
			result_months =
			    SubtractOperatorOverflowCheck::Operation<int32_t, int32_t, int32_t>(result_months, bucket_width_months);
		}
		result_months += origin_months;

		int32_t year =
		    (result_months < 0 && result_months % 12 != 0) ? 1970 + result_months / 12 - 1 : 1970 + result_months / 12;
		int32_t month =
		    (result_months < 0 && result_months % 12 != 0) ? result_months % 12 + 13 : result_months % 12 + 1;

		return Date::FromDate(year, month, 1);
	}

	struct WidthConvertibleToMicrosBinaryOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA bucket_width, TB ts) {
			if (!Value::IsFinite(ts)) {
				return Cast::template Operation<TB, TR>(ts);
			}
			int64_t bucket_width_micros = Interval::GetMicro(bucket_width);
			int64_t ts_micros = Timestamp::GetEpochMicroSeconds(Cast::template Operation<TB, timestamp_t>(ts));
			return Cast::template Operation<timestamp_t, TR>(
			    WidthConvertibleToMicrosCommon(bucket_width_micros, ts_micros, DEFAULT_ORIGIN_MICROS));
		}
	};

	struct WidthConvertibleToMonthsBinaryOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA bucket_width, TB ts) {
			if (!Value::IsFinite(ts)) {
				return Cast::template Operation<TB, TR>(ts);
			}
			int32_t ts_months = EpochMonths(ts);
			return Cast::template Operation<date_t, TR>(
			    WidthConvertibleToMonthsCommon(bucket_width.months, ts_months, DEFAULT_ORIGIN_MONTHS));
		}
	};

	struct BinaryOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA bucket_width, TB ts) {
			BucketWidthType bucket_width_type = ClassifyBucketWidthErrorThrow(bucket_width);
			switch (bucket_width_type) {
			case BucketWidthType::CONVERTIBLE_TO_MICROS:
				return WidthConvertibleToMicrosBinaryOperator::Operation<TA, TB, TR>(bucket_width, ts);
			case BucketWidthType::CONVERTIBLE_TO_MONTHS:
				return WidthConvertibleToMonthsBinaryOperator::Operation<TA, TB, TR>(bucket_width, ts);
			default:
				throw NotImplementedException("Bucket type not implemented for TIME_BUCKET");
			}
		}
	};

	struct OffsetWidthConvertibleToMicrosTernaryOperator {
		template <class TA, class TB, class TC, class TR>
		static inline TR Operation(TA bucket_width, TB ts, TC offset) {
			if (!Value::IsFinite(ts)) {
				return Cast::template Operation<TB, TR>(ts);
			}
			int64_t bucket_width_micros = Interval::GetMicro(bucket_width);
			int64_t ts_micros = Timestamp::GetEpochMicroSeconds(
			    Interval::Add(Cast::template Operation<TB, timestamp_t>(ts), Interval::Invert(offset)));
			return Cast::template Operation<timestamp_t, TR>(Interval::Add(
			    WidthConvertibleToMicrosCommon(bucket_width_micros, ts_micros, DEFAULT_ORIGIN_MICROS), offset));
		}
	};

	struct OffsetWidthConvertibleToMonthsTernaryOperator {
		template <class TA, class TB, class TC, class TR>
		static inline TR Operation(TA bucket_width, TB ts, TC offset) {
			if (!Value::IsFinite(ts)) {
				return Cast::template Operation<TB, TR>(ts);
			}
			int32_t ts_months = EpochMonths(Interval::Add(ts, Interval::Invert(offset)));
			return Interval::Add(Cast::template Operation<date_t, TR>(WidthConvertibleToMonthsCommon(
			                         bucket_width.months, ts_months, DEFAULT_ORIGIN_MONTHS)),
			                     offset);
		}
	};

	struct OffsetTernaryOperator {
		template <class TA, class TB, class TC, class TR>
		static inline TR Operation(TA bucket_width, TB ts, TC offset) {
			BucketWidthType bucket_width_type = ClassifyBucketWidthErrorThrow(bucket_width);
			switch (bucket_width_type) {
			case BucketWidthType::CONVERTIBLE_TO_MICROS:
				return OffsetWidthConvertibleToMicrosTernaryOperator::Operation<TA, TB, TC, TR>(bucket_width, ts,
				                                                                                offset);
			case BucketWidthType::CONVERTIBLE_TO_MONTHS:
				return OffsetWidthConvertibleToMonthsTernaryOperator::Operation<TA, TB, TC, TR>(bucket_width, ts,
				                                                                                offset);
			default:
				throw NotImplementedException("Bucket type not implemented for TIME_BUCKET");
			}
		}
	};

	struct OriginWidthConvertibleToMicrosTernaryOperator {
		template <class TA, class TB, class TC, class TR>
		static inline TR Operation(TA bucket_width, TB ts, TC origin) {
			if (!Value::IsFinite(ts)) {
				return Cast::template Operation<TB, TR>(ts);
			}
			int64_t bucket_width_micros = Interval::GetMicro(bucket_width);
			int64_t ts_micros = Timestamp::GetEpochMicroSeconds(Cast::template Operation<TB, timestamp_t>(ts));
			int64_t origin_micros = Timestamp::GetEpochMicroSeconds(Cast::template Operation<TB, timestamp_t>(origin));
			return Cast::template Operation<timestamp_t, TR>(
			    WidthConvertibleToMicrosCommon(bucket_width_micros, ts_micros, origin_micros));
		}
	};

	struct OriginWidthConvertibleToMonthsTernaryOperator {
		template <class TA, class TB, class TC, class TR>
		static inline TR Operation(TA bucket_width, TB ts, TC origin) {
			if (!Value::IsFinite(ts)) {
				return Cast::template Operation<TB, TR>(ts);
			}
			int32_t ts_months = EpochMonths(ts);
			int32_t origin_months = EpochMonths(origin);
			return Cast::template Operation<date_t, TR>(
			    WidthConvertibleToMonthsCommon(bucket_width.months, ts_months, origin_months));
		}
	};

	struct OriginTernaryOperator {
		template <class TA, class TB, class TC, class TR>
		static inline TR Operation(TA bucket_width, TB ts, TC origin, ValidityMask &mask, idx_t idx) {
			if (!Value::IsFinite(origin)) {
				mask.SetInvalid(idx);
				return TR();
			}
			BucketWidthType bucket_width_type = ClassifyBucketWidthErrorThrow(bucket_width);
			switch (bucket_width_type) {
			case BucketWidthType::CONVERTIBLE_TO_MICROS:
				return OriginWidthConvertibleToMicrosTernaryOperator::Operation<TA, TB, TC, TR>(bucket_width, ts,
				                                                                                origin);
			case BucketWidthType::CONVERTIBLE_TO_MONTHS:
				return OriginWidthConvertibleToMonthsTernaryOperator::Operation<TA, TB, TC, TR>(bucket_width, ts,
				                                                                                origin);
			default:
				throw NotImplementedException("Bucket type not implemented for TIME_BUCKET");
			}
		}
	};
};

template <typename T>
static void TimeBucketFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(args.ColumnCount() == 2);

	auto &bucket_width_arg = args.data[0];
	auto &ts_arg = args.data[1];

	if (bucket_width_arg.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		if (ConstantVector::IsNull(bucket_width_arg)) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
			ConstantVector::SetNull(result, true);
		} else {
			interval_t bucket_width = *ConstantVector::GetData<interval_t>(bucket_width_arg);
			TimeBucket::BucketWidthType bucket_width_type = TimeBucket::ClassifyBucketWidth(bucket_width);
			switch (bucket_width_type) {
			case TimeBucket::BucketWidthType::CONVERTIBLE_TO_MICROS:
				BinaryExecutor::Execute<interval_t, T, T>(
				    bucket_width_arg, ts_arg, result, args.size(),
				    TimeBucket::WidthConvertibleToMicrosBinaryOperator::Operation<interval_t, T, T>);
				break;
			case TimeBucket::BucketWidthType::CONVERTIBLE_TO_MONTHS:
				BinaryExecutor::Execute<interval_t, T, T>(
				    bucket_width_arg, ts_arg, result, args.size(),
				    TimeBucket::WidthConvertibleToMonthsBinaryOperator::Operation<interval_t, T, T>);
				break;
			case TimeBucket::BucketWidthType::UNCLASSIFIED:
				BinaryExecutor::Execute<interval_t, T, T>(bucket_width_arg, ts_arg, result, args.size(),
				                                          TimeBucket::BinaryOperator::Operation<interval_t, T, T>);
				break;
			default:
				throw NotImplementedException("Bucket type not implemented for TIME_BUCKET");
			}
		}
	} else {
		BinaryExecutor::Execute<interval_t, T, T>(bucket_width_arg, ts_arg, result, args.size(),
		                                          TimeBucket::BinaryOperator::Operation<interval_t, T, T>);
	}
}

template <typename T>
static void TimeBucketOffsetFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(args.ColumnCount() == 3);

	auto &bucket_width_arg = args.data[0];
	auto &ts_arg = args.data[1];
	auto &offset_arg = args.data[2];

	if (bucket_width_arg.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		if (ConstantVector::IsNull(bucket_width_arg)) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
			ConstantVector::SetNull(result, true);
		} else {
			interval_t bucket_width = *ConstantVector::GetData<interval_t>(bucket_width_arg);
			TimeBucket::BucketWidthType bucket_width_type = TimeBucket::ClassifyBucketWidth(bucket_width);
			switch (bucket_width_type) {
			case TimeBucket::BucketWidthType::CONVERTIBLE_TO_MICROS:
				TernaryExecutor::Execute<interval_t, T, interval_t, T>(
				    bucket_width_arg, ts_arg, offset_arg, result, args.size(),
				    TimeBucket::OffsetWidthConvertibleToMicrosTernaryOperator::Operation<interval_t, T, interval_t, T>);
				break;
			case TimeBucket::BucketWidthType::CONVERTIBLE_TO_MONTHS:
				TernaryExecutor::Execute<interval_t, T, interval_t, T>(
				    bucket_width_arg, ts_arg, offset_arg, result, args.size(),
				    TimeBucket::OffsetWidthConvertibleToMonthsTernaryOperator::Operation<interval_t, T, interval_t, T>);
				break;
			case TimeBucket::BucketWidthType::UNCLASSIFIED:
				TernaryExecutor::Execute<interval_t, T, interval_t, T>(
				    bucket_width_arg, ts_arg, offset_arg, result, args.size(),
				    TimeBucket::OffsetTernaryOperator::Operation<interval_t, T, interval_t, T>);
				break;
			default:
				throw NotImplementedException("Bucket type not implemented for TIME_BUCKET");
			}
		}
	} else {
		TernaryExecutor::Execute<interval_t, T, interval_t, T>(
		    bucket_width_arg, ts_arg, offset_arg, result, args.size(),
		    TimeBucket::OffsetTernaryOperator::Operation<interval_t, T, interval_t, T>);
	}
}

template <typename T>
static void TimeBucketOriginFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(args.ColumnCount() == 3);

	auto &bucket_width_arg = args.data[0];
	auto &ts_arg = args.data[1];
	auto &origin_arg = args.data[2];

	if (bucket_width_arg.GetVectorType() == VectorType::CONSTANT_VECTOR &&
	    origin_arg.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		if (ConstantVector::IsNull(bucket_width_arg) || ConstantVector::IsNull(origin_arg) ||
		    !Value::IsFinite(*ConstantVector::GetData<T>(origin_arg))) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
			ConstantVector::SetNull(result, true);
		} else {
			interval_t bucket_width = *ConstantVector::GetData<interval_t>(bucket_width_arg);
			TimeBucket::BucketWidthType bucket_width_type = TimeBucket::ClassifyBucketWidth(bucket_width);
			switch (bucket_width_type) {
			case TimeBucket::BucketWidthType::CONVERTIBLE_TO_MICROS:
				TernaryExecutor::Execute<interval_t, T, T, T>(
				    bucket_width_arg, ts_arg, origin_arg, result, args.size(),
				    TimeBucket::OriginWidthConvertibleToMicrosTernaryOperator::Operation<interval_t, T, T, T>);
				break;
			case TimeBucket::BucketWidthType::CONVERTIBLE_TO_MONTHS:
				TernaryExecutor::Execute<interval_t, T, T, T>(
				    bucket_width_arg, ts_arg, origin_arg, result, args.size(),
				    TimeBucket::OriginWidthConvertibleToMonthsTernaryOperator::Operation<interval_t, T, T, T>);
				break;
			case TimeBucket::BucketWidthType::UNCLASSIFIED:
				TernaryExecutor::ExecuteWithNulls<interval_t, T, T, T>(
				    bucket_width_arg, ts_arg, origin_arg, result, args.size(),
				    TimeBucket::OriginTernaryOperator::Operation<interval_t, T, T, T>);
				break;
			default:
				throw NotImplementedException("Bucket type not implemented for TIME_BUCKET");
			}
		}
	} else {
		TernaryExecutor::ExecuteWithNulls<interval_t, T, T, T>(
		    bucket_width_arg, ts_arg, origin_arg, result, args.size(),
		    TimeBucket::OriginTernaryOperator::Operation<interval_t, T, T, T>);
	}
}

ScalarFunctionSet TimeBucketFun::GetFunctions() {
	ScalarFunctionSet time_bucket;
	time_bucket.AddFunction(
	    ScalarFunction({LogicalType::INTERVAL, LogicalType::DATE}, LogicalType::DATE, TimeBucketFunction<date_t>));
	time_bucket.AddFunction(ScalarFunction({LogicalType::INTERVAL, LogicalType::TIMESTAMP}, LogicalType::TIMESTAMP,
	                                       TimeBucketFunction<timestamp_t>));
	time_bucket.AddFunction(ScalarFunction({LogicalType::INTERVAL, LogicalType::DATE, LogicalType::INTERVAL},
	                                       LogicalType::DATE, TimeBucketOffsetFunction<date_t>));
	time_bucket.AddFunction(ScalarFunction({LogicalType::INTERVAL, LogicalType::TIMESTAMP, LogicalType::INTERVAL},
	                                       LogicalType::TIMESTAMP, TimeBucketOffsetFunction<timestamp_t>));
	time_bucket.AddFunction(ScalarFunction({LogicalType::INTERVAL, LogicalType::DATE, LogicalType::DATE},
	                                       LogicalType::DATE, TimeBucketOriginFunction<date_t>));
	time_bucket.AddFunction(ScalarFunction({LogicalType::INTERVAL, LogicalType::TIMESTAMP, LogicalType::TIMESTAMP},
	                                       LogicalType::TIMESTAMP, TimeBucketOriginFunction<timestamp_t>));
	return time_bucket;
}

} // namespace duckdb




namespace duckdb {

struct ToYearsOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		interval_t result;
		result.days = 0;
		result.micros = 0;
		if (!TryMultiplyOperator::Operation<int32_t, int32_t, int32_t>(input, Interval::MONTHS_PER_YEAR,
		                                                               result.months)) {
			throw OutOfRangeException("Interval value %d years out of range", input);
		}
		return result;
	}
};

struct ToMonthsOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		interval_t result;
		result.months = input;
		result.days = 0;
		result.micros = 0;
		return result;
	}
};

struct ToDaysOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		interval_t result;
		result.months = 0;
		result.days = input;
		result.micros = 0;
		return result;
	}
};

struct ToHoursOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		interval_t result;
		result.months = 0;
		result.days = 0;
		if (!TryMultiplyOperator::Operation<int64_t, int64_t, int64_t>(input, Interval::MICROS_PER_HOUR,
		                                                               result.micros)) {
			throw OutOfRangeException("Interval value %d hours out of range", input);
		}
		return result;
	}
};

struct ToMinutesOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		interval_t result;
		result.months = 0;
		result.days = 0;
		if (!TryMultiplyOperator::Operation<int64_t, int64_t, int64_t>(input, Interval::MICROS_PER_MINUTE,
		                                                               result.micros)) {
			throw OutOfRangeException("Interval value %d minutes out of range", input);
		}
		return result;
	}
};

struct ToSecondsOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		interval_t result;
		result.months = 0;
		result.days = 0;
		if (!TryMultiplyOperator::Operation<int64_t, int64_t, int64_t>(input, Interval::MICROS_PER_SEC,
		                                                               result.micros)) {
			throw OutOfRangeException("Interval value %d seconds out of range", input);
		}
		return result;
	}
};

struct ToMilliSecondsOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		interval_t result;
		result.months = 0;
		result.days = 0;
		if (!TryMultiplyOperator::Operation<int64_t, int64_t, int64_t>(input, Interval::MICROS_PER_MSEC,
		                                                               result.micros)) {
			throw OutOfRangeException("Interval value %d milliseconds out of range", input);
		}
		return result;
	}
};

struct ToMicroSecondsOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		interval_t result;
		result.months = 0;
		result.days = 0;
		result.micros = input;
		return result;
	}
};

ScalarFunction ToYearsFun::GetFunction() {
	return ScalarFunction({LogicalType::INTEGER}, LogicalType::INTERVAL,
	                      ScalarFunction::UnaryFunction<int32_t, interval_t, ToYearsOperator>);
}

ScalarFunction ToMonthsFun::GetFunction() {
	return ScalarFunction({LogicalType::INTEGER}, LogicalType::INTERVAL,
	                      ScalarFunction::UnaryFunction<int32_t, interval_t, ToMonthsOperator>);
}

ScalarFunction ToDaysFun::GetFunction() {
	return ScalarFunction({LogicalType::INTEGER}, LogicalType::INTERVAL,
	                      ScalarFunction::UnaryFunction<int32_t, interval_t, ToDaysOperator>);
}

ScalarFunction ToHoursFun::GetFunction() {
	return ScalarFunction({LogicalType::BIGINT}, LogicalType::INTERVAL,
	                      ScalarFunction::UnaryFunction<int64_t, interval_t, ToHoursOperator>);
}

ScalarFunction ToMinutesFun::GetFunction() {
	return ScalarFunction({LogicalType::BIGINT}, LogicalType::INTERVAL,
	                      ScalarFunction::UnaryFunction<int64_t, interval_t, ToMinutesOperator>);
}

ScalarFunction ToSecondsFun::GetFunction() {
	return ScalarFunction({LogicalType::BIGINT}, LogicalType::INTERVAL,
	                      ScalarFunction::UnaryFunction<int64_t, interval_t, ToSecondsOperator>);
}

ScalarFunction ToMillisecondsFun::GetFunction() {
	return ScalarFunction({LogicalType::BIGINT}, LogicalType::INTERVAL,
	                      ScalarFunction::UnaryFunction<int64_t, interval_t, ToMilliSecondsOperator>);
}

ScalarFunction ToMicrosecondsFun::GetFunction() {
	return ScalarFunction({LogicalType::BIGINT}, LogicalType::INTERVAL,
	                      ScalarFunction::UnaryFunction<int64_t, interval_t, ToMicroSecondsOperator>);
}

} // namespace duckdb


namespace duckdb {

static void EnumFirstFunction(DataChunk &input, ExpressionState &state, Vector &result) {
	D_ASSERT(input.GetTypes().size() == 1);
	auto &enum_vector = EnumType::GetValuesInsertOrder(input.GetTypes()[0]);
	auto val = Value(enum_vector.GetValue(0));
	result.Reference(val);
}

static void EnumLastFunction(DataChunk &input, ExpressionState &state, Vector &result) {
	D_ASSERT(input.GetTypes().size() == 1);
	auto enum_size = EnumType::GetSize(input.GetTypes()[0]);
	auto &enum_vector = EnumType::GetValuesInsertOrder(input.GetTypes()[0]);
	auto val = Value(enum_vector.GetValue(enum_size - 1));
	result.Reference(val);
}

static void EnumRangeFunction(DataChunk &input, ExpressionState &state, Vector &result) {
	D_ASSERT(input.GetTypes().size() == 1);
	auto enum_size = EnumType::GetSize(input.GetTypes()[0]);
	auto &enum_vector = EnumType::GetValuesInsertOrder(input.GetTypes()[0]);
	vector<Value> enum_values;
	for (idx_t i = 0; i < enum_size; i++) {
		enum_values.emplace_back(enum_vector.GetValue(i));
	}
	auto val = Value::LIST(enum_values);
	result.Reference(val);
}

static void EnumRangeBoundaryFunction(DataChunk &input, ExpressionState &state, Vector &result) {
	D_ASSERT(input.GetTypes().size() == 2);
	idx_t start, end;
	auto first_param = input.GetValue(0, 0);
	auto second_param = input.GetValue(1, 0);

	auto &enum_vector = first_param.IsNull() ? EnumType::GetValuesInsertOrder(input.GetTypes()[1])
	                                         : EnumType::GetValuesInsertOrder(input.GetTypes()[0]);

	if (first_param.IsNull()) {
		start = 0;
	} else {
		start = first_param.GetValue<uint32_t>();
	}
	if (second_param.IsNull()) {
		end = EnumType::GetSize(input.GetTypes()[0]);
	} else {
		end = second_param.GetValue<uint32_t>() + 1;
	}
	vector<Value> enum_values;
	for (idx_t i = start; i < end; i++) {
		enum_values.emplace_back(enum_vector.GetValue(i));
	}
	Value val;
	if (enum_values.empty()) {
		val = Value::EMPTYLIST(LogicalType::VARCHAR);
	} else {
		val = Value::LIST(enum_values);
	}
	result.Reference(val);
}

static void EnumCodeFunction(DataChunk &input, ExpressionState &state, Vector &result) {
	D_ASSERT(input.GetTypes().size() == 1);
	result.Reinterpret(input.data[0]);
}

static void CheckEnumParameter(const Expression &expr) {
	if (expr.HasParameter()) {
		throw ParameterNotResolvedException();
	}
}

unique_ptr<FunctionData> BindEnumFunction(ClientContext &context, ScalarFunction &bound_function,
                                          vector<unique_ptr<Expression>> &arguments) {
	CheckEnumParameter(*arguments[0]);
	if (arguments[0]->return_type.id() != LogicalTypeId::ENUM) {
		throw BinderException("This function needs an ENUM as an argument");
	}
	return nullptr;
}

unique_ptr<FunctionData> BindEnumCodeFunction(ClientContext &context, ScalarFunction &bound_function,
                                              vector<unique_ptr<Expression>> &arguments) {
	CheckEnumParameter(*arguments[0]);
	if (arguments[0]->return_type.id() != LogicalTypeId::ENUM) {
		throw BinderException("This function needs an ENUM as an argument");
	}

	auto phy_type = EnumType::GetPhysicalType(arguments[0]->return_type);
	switch (phy_type) {
	case PhysicalType::UINT8:
		bound_function.return_type = LogicalType(LogicalTypeId::UTINYINT);
		break;
	case PhysicalType::UINT16:
		bound_function.return_type = LogicalType(LogicalTypeId::USMALLINT);
		break;
	case PhysicalType::UINT32:
		bound_function.return_type = LogicalType(LogicalTypeId::UINTEGER);
		break;
	case PhysicalType::UINT64:
		bound_function.return_type = LogicalType(LogicalTypeId::UBIGINT);
		break;
	default:
		throw InternalException("Unsupported Enum Internal Type");
	}

	return nullptr;
}

unique_ptr<FunctionData> BindEnumRangeBoundaryFunction(ClientContext &context, ScalarFunction &bound_function,
                                                       vector<unique_ptr<Expression>> &arguments) {
	CheckEnumParameter(*arguments[0]);
	CheckEnumParameter(*arguments[1]);
	if (arguments[0]->return_type.id() != LogicalTypeId::ENUM && arguments[0]->return_type != LogicalType::SQLNULL) {
		throw BinderException("This function needs an ENUM as an argument");
	}
	if (arguments[1]->return_type.id() != LogicalTypeId::ENUM && arguments[1]->return_type != LogicalType::SQLNULL) {
		throw BinderException("This function needs an ENUM as an argument");
	}
	if (arguments[0]->return_type == LogicalType::SQLNULL && arguments[1]->return_type == LogicalType::SQLNULL) {
		throw BinderException("This function needs an ENUM as an argument");
	}
	if (arguments[0]->return_type.id() == LogicalTypeId::ENUM &&
	    arguments[1]->return_type.id() == LogicalTypeId::ENUM &&
	    arguments[0]->return_type != arguments[1]->return_type) {
		throw BinderException("The parameters need to link to ONLY one enum OR be NULL ");
	}
	return nullptr;
}

ScalarFunction EnumFirstFun::GetFunction() {
	auto fun = ScalarFunction({LogicalType::ANY}, LogicalType::VARCHAR, EnumFirstFunction, BindEnumFunction);
	fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	return fun;
}

ScalarFunction EnumLastFun::GetFunction() {
	auto fun = ScalarFunction({LogicalType::ANY}, LogicalType::VARCHAR, EnumLastFunction, BindEnumFunction);
	fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	return fun;
}

ScalarFunction EnumCodeFun::GetFunction() {
	auto fun = ScalarFunction({LogicalType::ANY}, LogicalType::ANY, EnumCodeFunction, BindEnumCodeFunction);
	fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	return fun;
}

ScalarFunction EnumRangeFun::GetFunction() {
	auto fun = ScalarFunction({LogicalType::ANY}, LogicalType::LIST(LogicalType::VARCHAR), EnumRangeFunction,
	                          BindEnumFunction);
	fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	return fun;
}

ScalarFunction EnumRangeBoundaryFun::GetFunction() {
	auto fun = ScalarFunction({LogicalType::ANY, LogicalType::ANY}, LogicalType::LIST(LogicalType::VARCHAR),
	                          EnumRangeBoundaryFunction, BindEnumRangeBoundaryFunction);
	fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	return fun;
}

} // namespace duckdb



namespace duckdb {

static void AliasFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	Value v(state.expr.alias.empty() ? func_expr.children[0]->GetName() : state.expr.alias);
	result.Reference(v);
}

ScalarFunction AliasFun::GetFunction() {
	auto fun = ScalarFunction({LogicalType::ANY}, LogicalType::VARCHAR, AliasFunction);
	fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	return fun;
}

} // namespace duckdb







namespace duckdb {

struct CurrentSettingBindData : public FunctionData {
	explicit CurrentSettingBindData(Value value_p) : value(std::move(value_p)) {
	}

	Value value;

public:
	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<CurrentSettingBindData>(value);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = (const CurrentSettingBindData &)other_p;
		return Value::NotDistinctFrom(value, other.value);
	}
};

static void CurrentSettingFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &info = func_expr.bind_info->Cast<CurrentSettingBindData>();
	result.Reference(info.value);
}

unique_ptr<FunctionData> CurrentSettingBind(ClientContext &context, ScalarFunction &bound_function,
                                            vector<unique_ptr<Expression>> &arguments) {

	auto &key_child = arguments[0];
	if (key_child->return_type.id() == LogicalTypeId::UNKNOWN) {
		throw ParameterNotResolvedException();
	}
	if (key_child->return_type.id() != LogicalTypeId::VARCHAR ||
	    key_child->return_type.id() != LogicalTypeId::VARCHAR || !key_child->IsFoldable()) {
		throw ParserException("Key name for current_setting needs to be a constant string");
	}
	Value key_val = ExpressionExecutor::EvaluateScalar(context, *key_child);
	D_ASSERT(key_val.type().id() == LogicalTypeId::VARCHAR);
	auto &key_str = StringValue::Get(key_val);
	if (key_val.IsNull() || key_str.empty()) {
		throw ParserException("Key name for current_setting needs to be neither NULL nor empty");
	}

	auto key = StringUtil::Lower(key_str);
	Value val;
	if (!context.TryGetCurrentSetting(key, val)) {
		throw Catalog::UnrecognizedConfigurationError(context, key);
	}

	bound_function.return_type = val.type();
	return make_uniq<CurrentSettingBindData>(val);
}

ScalarFunction CurrentSettingFun::GetFunction() {
	auto fun = ScalarFunction({LogicalType::VARCHAR}, LogicalType::ANY, CurrentSettingFunction, CurrentSettingBind);
	fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	return fun;
}

} // namespace duckdb

#include <iostream>

namespace duckdb {

struct ErrorOperator {
	template <class TA, class TR>
	static inline TR Operation(const TA &input) {
		throw Exception(input.GetString());
	}
};

ScalarFunction ErrorFun::GetFunction() {
	auto fun = ScalarFunction({LogicalType::VARCHAR}, LogicalType::BOOLEAN,
	                          ScalarFunction::UnaryFunction<string_t, bool, ErrorOperator>);
	// Set the function with side effects to avoid the optimization.
	fun.side_effects = FunctionSideEffects::HAS_SIDE_EFFECTS;
	return fun;
}

} // namespace duckdb


namespace duckdb {

static void HashFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	args.Hash(result);
}

ScalarFunction HashFun::GetFunction() {
	auto hash_fun = ScalarFunction({LogicalType::ANY}, LogicalType::HASH, HashFunction);
	hash_fun.varargs = LogicalType::ANY;
	hash_fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	return hash_fun;
}

} // namespace duckdb



namespace duckdb {

template <class OP>
struct LeastOperator {
	template <class T>
	static T Operation(T left, T right) {
		return OP::Operation(left, right) ? left : right;
	}
};

template <class T, class OP, bool IS_STRING = false>
static void LeastGreatestFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	if (args.ColumnCount() == 1) {
		// single input: nop
		result.Reference(args.data[0]);
		return;
	}
	auto result_type = VectorType::CONSTANT_VECTOR;
	for (idx_t col_idx = 0; col_idx < args.ColumnCount(); col_idx++) {
		if (args.data[col_idx].GetVectorType() != VectorType::CONSTANT_VECTOR) {
			// non-constant input: result is not a constant vector
			result_type = VectorType::FLAT_VECTOR;
		}
		if (IS_STRING) {
			// for string vectors we add a reference to the heap of the children
			StringVector::AddHeapReference(result, args.data[col_idx]);
		}
	}

	auto result_data = FlatVector::GetData<T>(result);
	auto &result_mask = FlatVector::Validity(result);
	// copy over the first column
	bool result_has_value[STANDARD_VECTOR_SIZE];
	{
		UnifiedVectorFormat vdata;
		args.data[0].ToUnifiedFormat(args.size(), vdata);
		auto input_data = (T *)vdata.data;
		for (idx_t i = 0; i < args.size(); i++) {
			auto vindex = vdata.sel->get_index(i);
			if (vdata.validity.RowIsValid(vindex)) {
				result_data[i] = input_data[vindex];
				result_has_value[i] = true;
			} else {
				result_has_value[i] = false;
			}
		}
	}
	// now handle the remainder of the columns
	for (idx_t col_idx = 1; col_idx < args.ColumnCount(); col_idx++) {
		if (args.data[col_idx].GetVectorType() == VectorType::CONSTANT_VECTOR &&
		    ConstantVector::IsNull(args.data[col_idx])) {
			// ignore null vector
			continue;
		}

		UnifiedVectorFormat vdata;
		args.data[col_idx].ToUnifiedFormat(args.size(), vdata);

		auto input_data = (T *)vdata.data;
		if (!vdata.validity.AllValid()) {
			// potential new null entries: have to check the null mask
			for (idx_t i = 0; i < args.size(); i++) {
				auto vindex = vdata.sel->get_index(i);
				if (vdata.validity.RowIsValid(vindex)) {
					// not a null entry: perform the operation and add to new set
					auto ivalue = input_data[vindex];
					if (!result_has_value[i] || OP::template Operation<T>(ivalue, result_data[i])) {
						result_has_value[i] = true;
						result_data[i] = ivalue;
					}
				}
			}
		} else {
			// no new null entries: only need to perform the operation
			for (idx_t i = 0; i < args.size(); i++) {
				auto vindex = vdata.sel->get_index(i);

				auto ivalue = input_data[vindex];
				if (!result_has_value[i] || OP::template Operation<T>(ivalue, result_data[i])) {
					result_has_value[i] = true;
					result_data[i] = ivalue;
				}
			}
		}
	}
	for (idx_t i = 0; i < args.size(); i++) {
		if (!result_has_value[i]) {
			result_mask.SetInvalid(i);
		}
	}
	result.SetVectorType(result_type);
}

template <typename T, class OP>
ScalarFunction GetLeastGreatestFunction(const LogicalType &type) {
	return ScalarFunction({type}, type, LeastGreatestFunction<T, OP>, nullptr, nullptr, nullptr, nullptr, type,
	                      FunctionSideEffects::NO_SIDE_EFFECTS, FunctionNullHandling::SPECIAL_HANDLING);
}

template <class OP>
static ScalarFunctionSet GetLeastGreatestFunctions() {
	ScalarFunctionSet fun_set;
	fun_set.AddFunction(ScalarFunction({LogicalType::BIGINT}, LogicalType::BIGINT, LeastGreatestFunction<int64_t, OP>,
	                                   nullptr, nullptr, nullptr, nullptr, LogicalType::BIGINT,
	                                   FunctionSideEffects::NO_SIDE_EFFECTS, FunctionNullHandling::SPECIAL_HANDLING));
	fun_set.AddFunction(ScalarFunction(
	    {LogicalType::HUGEINT}, LogicalType::HUGEINT, LeastGreatestFunction<hugeint_t, OP>, nullptr, nullptr, nullptr,
	    nullptr, LogicalType::HUGEINT, FunctionSideEffects::NO_SIDE_EFFECTS, FunctionNullHandling::SPECIAL_HANDLING));
	fun_set.AddFunction(ScalarFunction({LogicalType::DOUBLE}, LogicalType::DOUBLE, LeastGreatestFunction<double, OP>,
	                                   nullptr, nullptr, nullptr, nullptr, LogicalType::DOUBLE,
	                                   FunctionSideEffects::NO_SIDE_EFFECTS, FunctionNullHandling::SPECIAL_HANDLING));
	fun_set.AddFunction(ScalarFunction({LogicalType::VARCHAR}, LogicalType::VARCHAR,
	                                   LeastGreatestFunction<string_t, OP, true>, nullptr, nullptr, nullptr, nullptr,
	                                   LogicalType::VARCHAR, FunctionSideEffects::NO_SIDE_EFFECTS,
	                                   FunctionNullHandling::SPECIAL_HANDLING));

	fun_set.AddFunction(GetLeastGreatestFunction<timestamp_t, OP>(LogicalType::TIMESTAMP));
	fun_set.AddFunction(GetLeastGreatestFunction<time_t, OP>(LogicalType::TIME));
	fun_set.AddFunction(GetLeastGreatestFunction<date_t, OP>(LogicalType::DATE));

	fun_set.AddFunction(GetLeastGreatestFunction<timestamp_t, OP>(LogicalType::TIMESTAMP_TZ));
	fun_set.AddFunction(GetLeastGreatestFunction<time_t, OP>(LogicalType::TIME_TZ));
	return fun_set;
}

ScalarFunctionSet LeastFun::GetFunctions() {
	return GetLeastGreatestFunctions<duckdb::LessThan>();
}

ScalarFunctionSet GreatestFun::GetFunctions() {
	return GetLeastGreatestFunctions<duckdb::GreaterThan>();
}

} // namespace duckdb



namespace duckdb {

struct StatsBindData : public FunctionData {
	explicit StatsBindData(string stats_p = string()) : stats(std::move(stats_p)) {
	}

	string stats;

public:
	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<StatsBindData>(stats);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = (const StatsBindData &)other_p;
		return stats == other.stats;
	}
};

static void StatsFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &info = func_expr.bind_info->Cast<StatsBindData>();
	if (info.stats.empty()) {
		info.stats = "No statistics";
	}
	Value v(info.stats);
	result.Reference(v);
}

unique_ptr<FunctionData> StatsBind(ClientContext &context, ScalarFunction &bound_function,
                                   vector<unique_ptr<Expression>> &arguments) {
	return make_uniq<StatsBindData>();
}

static unique_ptr<BaseStatistics> StatsPropagateStats(ClientContext &context, FunctionStatisticsInput &input) {
	auto &child_stats = input.child_stats;
	auto &bind_data = input.bind_data;
	auto &info = bind_data->Cast<StatsBindData>();
	info.stats = child_stats[0].ToString();
	return nullptr;
}

ScalarFunction StatsFun::GetFunction() {
	ScalarFunction stats({LogicalType::ANY}, LogicalType::VARCHAR, StatsFunction, StatsBind, nullptr,
	                     StatsPropagateStats);
	stats.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	stats.side_effects = FunctionSideEffects::HAS_SIDE_EFFECTS;
	return stats;
}

} // namespace duckdb










namespace duckdb {

// current_query
static void CurrentQueryFunction(DataChunk &input, ExpressionState &state, Vector &result) {
	Value val(state.GetContext().GetCurrentQuery());
	result.Reference(val);
}

// current_schema
static void CurrentSchemaFunction(DataChunk &input, ExpressionState &state, Vector &result) {
	Value val(ClientData::Get(state.GetContext()).catalog_search_path->GetDefault().schema);
	result.Reference(val);
}

// current_database
static void CurrentDatabaseFunction(DataChunk &input, ExpressionState &state, Vector &result) {
	Value val(DatabaseManager::GetDefaultDatabase(state.GetContext()));
	result.Reference(val);
}

// current_schemas
static void CurrentSchemasFunction(DataChunk &input, ExpressionState &state, Vector &result) {
	if (!input.AllConstant()) {
		throw NotImplementedException("current_schemas requires a constant input");
	}
	if (ConstantVector::IsNull(input.data[0])) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
		ConstantVector::SetNull(result, true);
		return;
	}
	auto implicit_schemas = *ConstantVector::GetData<bool>(input.data[0]);
	vector<Value> schema_list;
	auto &catalog_search_path = ClientData::Get(state.GetContext()).catalog_search_path;
	auto &search_path = implicit_schemas ? catalog_search_path->Get() : catalog_search_path->GetSetPaths();
	std::transform(search_path.begin(), search_path.end(), std::back_inserter(schema_list),
	               [](const CatalogSearchEntry &s) -> Value { return Value(s.schema); });

	auto val = Value::LIST(LogicalType::VARCHAR, schema_list);
	result.Reference(val);
}

// txid_current
static void TransactionIdCurrent(DataChunk &input, ExpressionState &state, Vector &result) {
	auto &context = state.GetContext();
	auto &catalog = Catalog::GetCatalog(context, DatabaseManager::GetDefaultDatabase(context));
	auto &transaction = DuckTransaction::Get(context, catalog);
	auto val = Value::BIGINT(transaction.start_time);
	result.Reference(val);
}

// version
static void VersionFunction(DataChunk &input, ExpressionState &state, Vector &result) {
	auto val = Value(DuckDB::LibraryVersion());
	result.Reference(val);
}

ScalarFunction CurrentQueryFun::GetFunction() {
	ScalarFunction current_query({}, LogicalType::VARCHAR, CurrentQueryFunction);
	current_query.side_effects = FunctionSideEffects::HAS_SIDE_EFFECTS;
	return current_query;
}

ScalarFunction CurrentSchemaFun::GetFunction() {
	return ScalarFunction({}, LogicalType::VARCHAR, CurrentSchemaFunction);
}

ScalarFunction CurrentDatabaseFun::GetFunction() {
	return ScalarFunction({}, LogicalType::VARCHAR, CurrentDatabaseFunction);
}

ScalarFunction CurrentSchemasFun::GetFunction() {
	auto varchar_list_type = LogicalType::LIST(LogicalType::VARCHAR);
	return ScalarFunction({LogicalType::BOOLEAN}, varchar_list_type, CurrentSchemasFunction);
}

ScalarFunction CurrentTransactionIdFun::GetFunction() {
	return ScalarFunction({}, LogicalType::BIGINT, TransactionIdCurrent);
}

ScalarFunction VersionFun::GetFunction() {
	return ScalarFunction({}, LogicalType::VARCHAR, VersionFunction);
}

} // namespace duckdb


namespace duckdb {

static void TypeOfFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	Value v(args.data[0].GetType().ToString());
	result.Reference(v);
}

ScalarFunction TypeOfFun::GetFunction() {
	auto fun = ScalarFunction({LogicalType::ANY}, LogicalType::VARCHAR, TypeOfFunction);
	fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	return fun;
}

} // namespace duckdb









namespace duckdb {

template <typename INPUT_TYPE, typename INDEX_TYPE>
INDEX_TYPE ValueLength(const INPUT_TYPE &value) {
	return 0;
}

template <>
int64_t ValueLength(const list_entry_t &value) {
	return value.length;
}

template <>
int32_t ValueLength(const string_t &value) {
	return LengthFun::Length<string_t, int32_t>(value);
}

template <typename INPUT_TYPE, typename INDEX_TYPE>
bool ClampIndex(INDEX_TYPE &index, const INPUT_TYPE &value) {
	const auto length = ValueLength<INPUT_TYPE, INDEX_TYPE>(value);
	if (index < 0) {
		if (-index > length) {
			return false;
		}
		index = length + index;
	} else if (index > length) {
		index = length;
	}
	return true;
}

template <typename INPUT_TYPE, typename INDEX_TYPE>
static bool ClampSlice(const INPUT_TYPE &value, INDEX_TYPE &begin, INDEX_TYPE &end, bool begin_valid, bool end_valid) {
	// Clamp offsets
	begin = begin_valid ? begin : 0;
	begin = (begin > 0) ? begin - 1 : begin;
	end = end_valid ? end : ValueLength<INPUT_TYPE, INDEX_TYPE>(value);
	if (!ClampIndex(begin, value) || !ClampIndex(end, value)) {
		return false;
	}
	end = MaxValue<INDEX_TYPE>(begin, end);

	return true;
}

template <typename INPUT_TYPE, typename INDEX_TYPE>
INPUT_TYPE SliceValue(Vector &result, INPUT_TYPE input, INDEX_TYPE begin, INDEX_TYPE end) {
	return input;
}

template <>
list_entry_t SliceValue(Vector &result, list_entry_t input, int64_t begin, int64_t end) {
	input.offset += begin;
	input.length = end - begin;
	return input;
}

template <>
string_t SliceValue(Vector &result, string_t input, int32_t begin, int32_t end) {
	// one-based - zero has strange semantics
	return SubstringFun::SubstringUnicode(result, input, begin + 1, end - begin);
}

template <typename INPUT_TYPE, typename INDEX_TYPE>
static void ExecuteSlice(Vector &result, Vector &s, Vector &b, Vector &e, const idx_t count) {
	if (result.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		auto rdata = ConstantVector::GetData<INPUT_TYPE>(result);
		auto sdata = ConstantVector::GetData<INPUT_TYPE>(s);
		auto bdata = ConstantVector::GetData<INDEX_TYPE>(b);
		auto edata = ConstantVector::GetData<INDEX_TYPE>(e);

		auto sliced = sdata[0];
		auto begin = bdata[0];
		auto end = edata[0];

		auto svalid = !ConstantVector::IsNull(s);
		auto bvalid = !ConstantVector::IsNull(b);
		auto evalid = !ConstantVector::IsNull(e);

		// Try to slice
		if (!svalid || !ClampSlice(sliced, begin, end, bvalid, evalid)) {
			ConstantVector::SetNull(result, true);
		} else {
			rdata[0] = SliceValue<INPUT_TYPE, INDEX_TYPE>(result, sliced, begin, end);
		}
	} else {
		UnifiedVectorFormat sdata, bdata, edata;

		s.ToUnifiedFormat(count, sdata);
		b.ToUnifiedFormat(count, bdata);
		e.ToUnifiedFormat(count, edata);

		auto rdata = FlatVector::GetData<INPUT_TYPE>(result);
		auto &rmask = FlatVector::Validity(result);

		for (idx_t i = 0; i < count; ++i) {
			auto sidx = sdata.sel->get_index(i);
			auto bidx = bdata.sel->get_index(i);
			auto eidx = edata.sel->get_index(i);

			auto sliced = ((INPUT_TYPE *)sdata.data)[sidx];
			auto begin = ((INDEX_TYPE *)bdata.data)[bidx];
			auto end = ((INDEX_TYPE *)edata.data)[eidx];

			auto svalid = sdata.validity.RowIsValid(sidx);
			auto bvalid = bdata.validity.RowIsValid(bidx);
			auto evalid = edata.validity.RowIsValid(eidx);

			// Try to slice
			if (!svalid || !ClampSlice(sliced, begin, end, bvalid, evalid)) {
				rmask.SetInvalid(i);
			} else {
				rdata[i] = SliceValue<INPUT_TYPE, INDEX_TYPE>(result, sliced, begin, end);
			}
		}
	}

	result.Verify(count);
}

static void ArraySliceFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(args.ColumnCount() == 3);
	D_ASSERT(args.data.size() == 3);
	auto count = args.size();

	Vector &s = args.data[0];
	Vector &b = args.data[1];
	Vector &e = args.data[2];

	result.SetVectorType(args.AllConstant() ? VectorType::CONSTANT_VECTOR : VectorType::FLAT_VECTOR);
	switch (result.GetType().id()) {
	case LogicalTypeId::LIST:
		// Share the value dictionary as we are just going to slice it
		if (s.GetVectorType() != VectorType::FLAT_VECTOR && s.GetVectorType() != VectorType::CONSTANT_VECTOR) {
			s.Flatten(count);
		}
		ListVector::ReferenceEntry(result, s);
		ExecuteSlice<list_entry_t, int64_t>(result, s, b, e, count);
		break;
	case LogicalTypeId::VARCHAR:
		ExecuteSlice<string_t, int32_t>(result, s, b, e, count);
		break;
	default:
		throw NotImplementedException("Specifier type not implemented");
	}
}

static unique_ptr<FunctionData> ArraySliceBind(ClientContext &context, ScalarFunction &bound_function,
                                               vector<unique_ptr<Expression>> &arguments) {
	D_ASSERT(bound_function.arguments.size() == 3);
	switch (arguments[0]->return_type.id()) {
	case LogicalTypeId::LIST:
		// The result is the same type
		bound_function.return_type = arguments[0]->return_type;
		break;
	case LogicalTypeId::VARCHAR:
		// string slice returns a string, but can only accept 32 bit integers
		bound_function.return_type = arguments[0]->return_type;
		bound_function.arguments[1] = LogicalType::INTEGER;
		bound_function.arguments[2] = LogicalType::INTEGER;
		break;
	case LogicalTypeId::SQLNULL:
	case LogicalTypeId::UNKNOWN:
		bound_function.arguments[0] = LogicalTypeId::UNKNOWN;
		bound_function.return_type = LogicalType::SQLNULL;
		break;
	default:
		throw BinderException("ARRAY_SLICE can only operate on LISTs and VARCHARs");
	}

	return make_uniq<VariableReturnBindData>(bound_function.return_type);
}

ScalarFunction ListSliceFun::GetFunction() {
	// the arguments and return types are actually set in the binder function
	ScalarFunction fun({LogicalType::ANY, LogicalType::BIGINT, LogicalType::BIGINT}, LogicalType::ANY,
	                   ArraySliceFunction, ArraySliceBind);
	fun.varargs = LogicalType::ANY;
	fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	return fun;
}

} // namespace duckdb






namespace duckdb {

void ListFlattenFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(args.ColumnCount() == 1);

	Vector &input = args.data[0];
	if (input.GetType().id() == LogicalTypeId::SQLNULL) {
		result.Reference(input);
		return;
	}

	idx_t count = args.size();

	UnifiedVectorFormat list_data;
	input.ToUnifiedFormat(count, list_data);
	auto list_entries = (list_entry_t *)list_data.data;

	auto &child_vector = ListVector::GetEntry(input);

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_entries = FlatVector::GetData<list_entry_t>(result);
	auto &result_validity = FlatVector::Validity(result);

	if (child_vector.GetType().id() == LogicalTypeId::SQLNULL) {
		auto result_entries = FlatVector::GetData<list_entry_t>(result);
		for (idx_t i = 0; i < count; i++) {
			auto list_index = list_data.sel->get_index(i);
			if (!list_data.validity.RowIsValid(list_index)) {
				result_validity.SetInvalid(i);
				continue;
			}
			result_entries[i].offset = 0;
			result_entries[i].length = 0;
		}
		if (args.AllConstant()) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
		}
		return;
	}

	auto child_size = ListVector::GetListSize(input);
	UnifiedVectorFormat child_data;
	child_vector.ToUnifiedFormat(child_size, child_data);
	auto child_entries = (list_entry_t *)child_data.data;
	auto &data_vector = ListVector::GetEntry(child_vector);

	idx_t offset = 0;
	for (idx_t i = 0; i < count; i++) {
		auto list_index = list_data.sel->get_index(i);
		if (!list_data.validity.RowIsValid(list_index)) {
			result_validity.SetInvalid(i);
			continue;
		}
		auto list_entry = list_entries[list_index];

		idx_t source_offset = 0;
		// Find first valid child list entry to get offset
		for (idx_t j = 0; j < list_entry.length; j++) {
			auto child_list_index = child_data.sel->get_index(list_entry.offset + j);
			if (child_data.validity.RowIsValid(child_list_index)) {
				source_offset = child_entries[child_list_index].offset;
				break;
			}
		}

		idx_t length = 0;
		// Find last valid child list entry to get length
		for (idx_t j = list_entry.length - 1; j != (idx_t)-1; j--) {
			auto child_list_index = child_data.sel->get_index(list_entry.offset + j);
			if (child_data.validity.RowIsValid(child_list_index)) {
				auto child_entry = child_entries[child_list_index];
				length = child_entry.offset + child_entry.length - source_offset;
				break;
			}
		}
		ListVector::Append(result, data_vector, source_offset + length, source_offset);

		result_entries[i].offset = offset;
		result_entries[i].length = length;
		offset += length;
	}

	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

static unique_ptr<FunctionData> ListFlattenBind(ClientContext &context, ScalarFunction &bound_function,
                                                vector<unique_ptr<Expression>> &arguments) {
	D_ASSERT(bound_function.arguments.size() == 1);

	auto &input_type = arguments[0]->return_type;
	bound_function.arguments[0] = input_type;
	if (input_type.id() == LogicalTypeId::UNKNOWN) {
		bound_function.arguments[0] = LogicalType(LogicalTypeId::UNKNOWN);
		bound_function.return_type = LogicalType(LogicalTypeId::SQLNULL);
		return nullptr;
	}
	D_ASSERT(input_type.id() == LogicalTypeId::LIST);

	auto child_type = ListType::GetChildType(input_type);
	if (child_type.id() == LogicalType::SQLNULL) {
		bound_function.return_type = input_type;
		return make_uniq<VariableReturnBindData>(bound_function.return_type);
	}
	if (child_type.id() == LogicalTypeId::UNKNOWN) {
		bound_function.arguments[0] = LogicalType(LogicalTypeId::UNKNOWN);
		bound_function.return_type = LogicalType(LogicalTypeId::SQLNULL);
		return nullptr;
	}
	D_ASSERT(child_type.id() == LogicalTypeId::LIST);

	bound_function.return_type = child_type;
	return make_uniq<VariableReturnBindData>(bound_function.return_type);
}

static unique_ptr<BaseStatistics> ListFlattenStats(ClientContext &context, FunctionStatisticsInput &input) {
	auto &child_stats = input.child_stats;
	auto &list_child_stats = ListStats::GetChildStats(child_stats[0]);
	auto child_copy = list_child_stats.Copy();
	child_copy.Set(StatsInfo::CAN_HAVE_NULL_VALUES);
	return child_copy.ToUnique();
}

ScalarFunction ListFlattenFun::GetFunction() {
	return ScalarFunction({LogicalType::LIST(LogicalType::LIST(LogicalType::ANY))}, LogicalType::LIST(LogicalType::ANY),
	                      ListFlattenFunction, ListFlattenBind, nullptr, ListFlattenStats);
}

} // namespace duckdb












namespace duckdb {

// FIXME: use a local state for each thread to increase performance?
// FIXME: benchmark the use of simple_update against using update (if applicable)

struct ListAggregatesBindData : public FunctionData {
	ListAggregatesBindData(const LogicalType &stype_p, unique_ptr<Expression> aggr_expr_p);
	~ListAggregatesBindData() override;

	LogicalType stype;
	unique_ptr<Expression> aggr_expr;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<ListAggregatesBindData>(stype, aggr_expr->Copy());
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = (const ListAggregatesBindData &)other_p;
		return stype == other.stype && aggr_expr->Equals(other.aggr_expr.get());
	}
	static void Serialize(FieldWriter &writer, const FunctionData *bind_data_p, const ScalarFunction &function) {
		throw NotImplementedException("FIXME: list aggr serialize");
	}
	static unique_ptr<FunctionData> Deserialize(ClientContext &context, FieldReader &reader,
	                                            ScalarFunction &bound_function) {
		throw NotImplementedException("FIXME: list aggr deserialize");
	}
};

ListAggregatesBindData::ListAggregatesBindData(const LogicalType &stype_p, unique_ptr<Expression> aggr_expr_p)
    : stype(stype_p), aggr_expr(std::move(aggr_expr_p)) {
}

ListAggregatesBindData::~ListAggregatesBindData() {
}

struct StateVector {
	StateVector(idx_t count_p, unique_ptr<Expression> aggr_expr_p)
	    : count(count_p), aggr_expr(std::move(aggr_expr_p)), state_vector(Vector(LogicalType::POINTER, count_p)) {
	}

	~StateVector() { // NOLINT
		// destroy objects within the aggregate states
		auto &aggr = aggr_expr->Cast<BoundAggregateExpression>();
		if (aggr.function.destructor) {
			AggregateInputData aggr_input_data(aggr.bind_info.get(), Allocator::DefaultAllocator());
			aggr.function.destructor(state_vector, aggr_input_data, count);
		}
	}

	idx_t count;
	unique_ptr<Expression> aggr_expr;
	Vector state_vector;
};

struct FinalizeValueFunctor {
	template <class T>
	static Value FinalizeValue(T first) {
		return Value::CreateValue(first);
	}
};

struct FinalizeStringValueFunctor {
	template <class T>
	static Value FinalizeValue(T first) {
		string_t value = first;
		return Value::CreateValue(value);
	}
};

struct AggregateFunctor {
	template <class OP, class T, class MAP_TYPE = unordered_map<T, idx_t>>
	static void ListExecuteFunction(Vector &result, Vector &state_vector, idx_t count) {
	}
};

struct DistinctFunctor {
	template <class OP, class T, class MAP_TYPE = unordered_map<T, idx_t>>
	static void ListExecuteFunction(Vector &result, Vector &state_vector, idx_t count) {

		UnifiedVectorFormat sdata;
		state_vector.ToUnifiedFormat(count, sdata);
		auto states = (HistogramAggState<T, MAP_TYPE> **)sdata.data;

		auto result_data = FlatVector::GetData<list_entry_t>(result);

		idx_t offset = 0;
		for (idx_t i = 0; i < count; i++) {

			auto state = states[sdata.sel->get_index(i)];
			result_data[i].offset = offset;

			if (!state->hist) {
				result_data[i].length = 0;
				continue;
			}

			result_data[i].length = state->hist->size();
			offset += state->hist->size();

			for (auto &entry : *state->hist) {
				Value bucket_value = OP::template FinalizeValue<T>(entry.first);
				ListVector::PushBack(result, bucket_value);
			}
		}
		result.Verify(count);
	}
};

struct UniqueFunctor {
	template <class OP, class T, class MAP_TYPE = unordered_map<T, idx_t>>
	static void ListExecuteFunction(Vector &result, Vector &state_vector, idx_t count) {

		UnifiedVectorFormat sdata;
		state_vector.ToUnifiedFormat(count, sdata);
		auto states = (HistogramAggState<T, MAP_TYPE> **)sdata.data;

		auto result_data = FlatVector::GetData<uint64_t>(result);

		for (idx_t i = 0; i < count; i++) {

			auto state = states[sdata.sel->get_index(i)];

			if (!state->hist) {
				result_data[i] = 0;
				continue;
			}

			result_data[i] = state->hist->size();
		}
		result.Verify(count);
	}
};

template <class FUNCTION_FUNCTOR, bool IS_AGGR = false>
static void ListAggregatesFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto count = args.size();
	Vector &lists = args.data[0];

	// set the result vector
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto &result_validity = FlatVector::Validity(result);

	if (lists.GetType().id() == LogicalTypeId::SQLNULL) {
		result_validity.SetInvalid(0);
		return;
	}

	// get the aggregate function
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &info = func_expr.bind_info->Cast<ListAggregatesBindData>();
	auto &aggr = info.aggr_expr->Cast<BoundAggregateExpression>();
	AggregateInputData aggr_input_data(aggr.bind_info.get(), Allocator::DefaultAllocator());

	D_ASSERT(aggr.function.update);

	auto lists_size = ListVector::GetListSize(lists);
	auto &child_vector = ListVector::GetEntry(lists);
	child_vector.Flatten(lists_size);

	UnifiedVectorFormat child_data;
	child_vector.ToUnifiedFormat(lists_size, child_data);

	UnifiedVectorFormat lists_data;
	lists.ToUnifiedFormat(count, lists_data);
	auto list_entries = (list_entry_t *)lists_data.data;

	// state_buffer holds the state for each list of this chunk
	idx_t size = aggr.function.state_size();
	auto state_buffer = make_unsafe_uniq_array<data_t>(size * count);

	// state vector for initialize and finalize
	StateVector state_vector(count, info.aggr_expr->Copy());
	auto states = FlatVector::GetData<data_ptr_t>(state_vector.state_vector);

	// state vector of STANDARD_VECTOR_SIZE holds the pointers to the states
	Vector state_vector_update = Vector(LogicalType::POINTER);
	auto states_update = FlatVector::GetData<data_ptr_t>(state_vector_update);

	// selection vector pointing to the data
	SelectionVector sel_vector(STANDARD_VECTOR_SIZE);
	idx_t states_idx = 0;

	for (idx_t i = 0; i < count; i++) {

		// initialize the state for this list
		auto state_ptr = state_buffer.get() + size * i;
		states[i] = state_ptr;
		aggr.function.initialize(states[i]);

		auto lists_index = lists_data.sel->get_index(i);
		const auto &list_entry = list_entries[lists_index];

		// nothing to do for this list
		if (!lists_data.validity.RowIsValid(lists_index)) {
			result_validity.SetInvalid(i);
			continue;
		}

		// skip empty list
		if (list_entry.length == 0) {
			continue;
		}

		for (idx_t child_idx = 0; child_idx < list_entry.length; child_idx++) {
			// states vector is full, update
			if (states_idx == STANDARD_VECTOR_SIZE) {
				// update the aggregate state(s)
				Vector slice(child_vector, sel_vector, states_idx);
				aggr.function.update(&slice, aggr_input_data, 1, state_vector_update, states_idx);

				// reset values
				states_idx = 0;
			}

			auto source_idx = child_data.sel->get_index(list_entry.offset + child_idx);
			sel_vector.set_index(states_idx, source_idx);
			states_update[states_idx] = state_ptr;
			states_idx++;
		}
	}

	// update the remaining elements of the last list(s)
	if (states_idx != 0) {
		Vector slice(child_vector, sel_vector, states_idx);
		aggr.function.update(&slice, aggr_input_data, 1, state_vector_update, states_idx);
	}

	if (IS_AGGR) {
		// finalize all the aggregate states
		aggr.function.finalize(state_vector.state_vector, aggr_input_data, result, count, 0);

	} else {
		// finalize manually to use the map
		D_ASSERT(aggr.function.arguments.size() == 1);
		auto key_type = aggr.function.arguments[0];

		switch (key_type.InternalType()) {
		case PhysicalType::BOOL:
			FUNCTION_FUNCTOR::template ListExecuteFunction<FinalizeValueFunctor, bool>(
			    result, state_vector.state_vector, count);
			break;
		case PhysicalType::UINT8:
			FUNCTION_FUNCTOR::template ListExecuteFunction<FinalizeValueFunctor, uint8_t>(
			    result, state_vector.state_vector, count);
			break;
		case PhysicalType::UINT16:
			FUNCTION_FUNCTOR::template ListExecuteFunction<FinalizeValueFunctor, uint16_t>(
			    result, state_vector.state_vector, count);
			break;
		case PhysicalType::UINT32:
			FUNCTION_FUNCTOR::template ListExecuteFunction<FinalizeValueFunctor, uint32_t>(
			    result, state_vector.state_vector, count);
			break;
		case PhysicalType::UINT64:
			FUNCTION_FUNCTOR::template ListExecuteFunction<FinalizeValueFunctor, uint64_t>(
			    result, state_vector.state_vector, count);
			break;
		case PhysicalType::INT8:
			FUNCTION_FUNCTOR::template ListExecuteFunction<FinalizeValueFunctor, int8_t>(
			    result, state_vector.state_vector, count);
			break;
		case PhysicalType::INT16:
			FUNCTION_FUNCTOR::template ListExecuteFunction<FinalizeValueFunctor, int16_t>(
			    result, state_vector.state_vector, count);
			break;
		case PhysicalType::INT32:
			if (key_type.id() == LogicalTypeId::DATE) {
				FUNCTION_FUNCTOR::template ListExecuteFunction<FinalizeValueFunctor, date_t>(
				    result, state_vector.state_vector, count);
			} else {
				FUNCTION_FUNCTOR::template ListExecuteFunction<FinalizeValueFunctor, int32_t>(
				    result, state_vector.state_vector, count);
			}
			break;
		case PhysicalType::INT64:
			switch (key_type.id()) {
			case LogicalTypeId::TIME:
				FUNCTION_FUNCTOR::template ListExecuteFunction<FinalizeValueFunctor, dtime_t>(
				    result, state_vector.state_vector, count);
				break;
			case LogicalTypeId::TIME_TZ:
				FUNCTION_FUNCTOR::template ListExecuteFunction<FinalizeValueFunctor, dtime_tz_t>(
				    result, state_vector.state_vector, count);
				break;
			case LogicalTypeId::TIMESTAMP:
				FUNCTION_FUNCTOR::template ListExecuteFunction<FinalizeValueFunctor, timestamp_t>(
				    result, state_vector.state_vector, count);
				break;
			case LogicalTypeId::TIMESTAMP_MS:
				FUNCTION_FUNCTOR::template ListExecuteFunction<FinalizeValueFunctor, timestamp_ms_t>(
				    result, state_vector.state_vector, count);
				break;
			case LogicalTypeId::TIMESTAMP_NS:
				FUNCTION_FUNCTOR::template ListExecuteFunction<FinalizeValueFunctor, timestamp_ns_t>(
				    result, state_vector.state_vector, count);
				break;
			case LogicalTypeId::TIMESTAMP_SEC:
				FUNCTION_FUNCTOR::template ListExecuteFunction<FinalizeValueFunctor, timestamp_sec_t>(
				    result, state_vector.state_vector, count);
				break;
			case LogicalTypeId::TIMESTAMP_TZ:
				FUNCTION_FUNCTOR::template ListExecuteFunction<FinalizeValueFunctor, timestamp_tz_t>(
				    result, state_vector.state_vector, count);
				break;
			default:
				FUNCTION_FUNCTOR::template ListExecuteFunction<FinalizeValueFunctor, int64_t>(
				    result, state_vector.state_vector, count);
				break;
			}
			break;
		case PhysicalType::FLOAT:
			FUNCTION_FUNCTOR::template ListExecuteFunction<FinalizeValueFunctor, float>(
			    result, state_vector.state_vector, count);
			break;
		case PhysicalType::DOUBLE:
			FUNCTION_FUNCTOR::template ListExecuteFunction<FinalizeValueFunctor, double>(
			    result, state_vector.state_vector, count);
			break;
		case PhysicalType::VARCHAR:
			FUNCTION_FUNCTOR::template ListExecuteFunction<FinalizeStringValueFunctor, string>(
			    result, state_vector.state_vector, count);
			break;
		default:
			throw InternalException("Unimplemented histogram aggregate");
		}
	}

	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

static void ListAggregateFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(args.ColumnCount() >= 2);
	ListAggregatesFunction<AggregateFunctor, true>(args, state, result);
}

static void ListDistinctFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(args.ColumnCount() == 1);
	ListAggregatesFunction<DistinctFunctor>(args, state, result);
}

static void ListUniqueFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(args.ColumnCount() == 1);
	ListAggregatesFunction<UniqueFunctor>(args, state, result);
}

template <bool IS_AGGR = false>
static unique_ptr<FunctionData>
ListAggregatesBindFunction(ClientContext &context, ScalarFunction &bound_function, const LogicalType &list_child_type,
                           AggregateFunction &aggr_function, vector<unique_ptr<Expression>> &arguments) {

	// create the child expression and its type
	vector<unique_ptr<Expression>> children;
	auto expr = make_uniq<BoundConstantExpression>(Value(list_child_type));
	children.push_back(std::move(expr));
	// push any extra arguments into the list aggregate bind
	if (arguments.size() > 2) {
		for (idx_t i = 2; i < arguments.size(); i++) {
			children.push_back(std::move(arguments[i]));
		}
		arguments.resize(2);
	}

	FunctionBinder function_binder(context);
	auto bound_aggr_function = function_binder.BindAggregateFunction(aggr_function, std::move(children));
	bound_function.arguments[0] = LogicalType::LIST(bound_aggr_function->function.arguments[0]);

	if (IS_AGGR) {
		bound_function.return_type = bound_aggr_function->function.return_type;
	}
	// check if the aggregate function consumed all the extra input arguments
	if (bound_aggr_function->children.size() > 1) {
		throw InvalidInputException(
		    "Aggregate function %s is not supported for list_aggr: extra arguments were not removed during bind",
		    bound_aggr_function->ToString());
	}

	return make_uniq<ListAggregatesBindData>(bound_function.return_type, std::move(bound_aggr_function));
}

template <bool IS_AGGR = false>
static unique_ptr<FunctionData> ListAggregatesBind(ClientContext &context, ScalarFunction &bound_function,
                                                   vector<unique_ptr<Expression>> &arguments) {
	if (arguments[0]->return_type.id() == LogicalTypeId::SQLNULL) {
		bound_function.arguments[0] = LogicalType::SQLNULL;
		bound_function.return_type = LogicalType::SQLNULL;
		return make_uniq<VariableReturnBindData>(bound_function.return_type);
	}

	bool is_parameter = arguments[0]->return_type.id() == LogicalTypeId::UNKNOWN;
	auto list_child_type = is_parameter ? LogicalTypeId::UNKNOWN : ListType::GetChildType(arguments[0]->return_type);

	string function_name = "histogram";
	if (IS_AGGR) { // get the name of the aggregate function
		if (!arguments[1]->IsFoldable()) {
			throw InvalidInputException("Aggregate function name must be a constant");
		}
		// get the function name
		Value function_value = ExpressionExecutor::EvaluateScalar(context, *arguments[1]);
		function_name = function_value.ToString();
	}

	// look up the aggregate function in the catalog
	QueryErrorContext error_context(nullptr, 0);
	auto &func = Catalog::GetSystemCatalog(context).GetEntry<AggregateFunctionCatalogEntry>(
	    context, DEFAULT_SCHEMA, function_name, error_context);
	D_ASSERT(func.type == CatalogType::AGGREGATE_FUNCTION_ENTRY);

	if (is_parameter) {
		bound_function.arguments[0] = LogicalTypeId::UNKNOWN;
		bound_function.return_type = LogicalType::SQLNULL;
		return nullptr;
	}

	// find a matching aggregate function
	string error;
	vector<LogicalType> types;
	types.push_back(list_child_type);
	// push any extra arguments into the type list
	for (idx_t i = 2; i < arguments.size(); i++) {
		types.push_back(arguments[i]->return_type);
	}

	FunctionBinder function_binder(context);
	auto best_function_idx = function_binder.BindFunction(func.name, func.functions, types, error);
	if (best_function_idx == DConstants::INVALID_INDEX) {
		throw BinderException("No matching aggregate function\n%s", error);
	}

	// found a matching function, bind it as an aggregate
	auto best_function = func.functions.GetFunctionByOffset(best_function_idx);
	if (IS_AGGR) {
		return ListAggregatesBindFunction<IS_AGGR>(context, bound_function, list_child_type, best_function, arguments);
	}

	// create the unordered map histogram function
	D_ASSERT(best_function.arguments.size() == 1);
	auto key_type = best_function.arguments[0];
	auto aggr_function = HistogramFun::GetHistogramUnorderedMap(key_type);
	return ListAggregatesBindFunction<IS_AGGR>(context, bound_function, list_child_type, aggr_function, arguments);
}

static unique_ptr<FunctionData> ListAggregateBind(ClientContext &context, ScalarFunction &bound_function,
                                                  vector<unique_ptr<Expression>> &arguments) {

	// the list column and the name of the aggregate function
	D_ASSERT(bound_function.arguments.size() >= 2);
	D_ASSERT(arguments.size() >= 2);

	return ListAggregatesBind<true>(context, bound_function, arguments);
}

static unique_ptr<FunctionData> ListDistinctBind(ClientContext &context, ScalarFunction &bound_function,
                                                 vector<unique_ptr<Expression>> &arguments) {

	D_ASSERT(bound_function.arguments.size() == 1);
	D_ASSERT(arguments.size() == 1);
	bound_function.return_type = arguments[0]->return_type;

	return ListAggregatesBind<>(context, bound_function, arguments);
}

static unique_ptr<FunctionData> ListUniqueBind(ClientContext &context, ScalarFunction &bound_function,
                                               vector<unique_ptr<Expression>> &arguments) {

	D_ASSERT(bound_function.arguments.size() == 1);
	D_ASSERT(arguments.size() == 1);
	bound_function.return_type = LogicalType::UBIGINT;

	return ListAggregatesBind<>(context, bound_function, arguments);
}

ScalarFunction ListAggregateFun::GetFunction() {
	auto result = ScalarFunction({LogicalType::LIST(LogicalType::ANY), LogicalType::VARCHAR}, LogicalType::ANY,
	                             ListAggregateFunction, ListAggregateBind);
	result.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	result.varargs = LogicalType::ANY;
	result.serialize = ListAggregatesBindData::Serialize;
	result.deserialize = ListAggregatesBindData::Deserialize;
	return result;
}

ScalarFunction ListDistinctFun::GetFunction() {
	return ScalarFunction({LogicalType::LIST(LogicalType::ANY)}, LogicalType::LIST(LogicalType::ANY),
	                      ListDistinctFunction, ListDistinctBind);
}

ScalarFunction ListUniqueFun::GetFunction() {
	return ScalarFunction({LogicalType::LIST(LogicalType::ANY)}, LogicalType::UBIGINT, ListUniqueFunction,
	                      ListUniqueBind);
}

} // namespace duckdb











namespace duckdb {

struct ListLambdaBindData : public FunctionData {
	ListLambdaBindData(const LogicalType &stype_p, unique_ptr<Expression> lambda_expr);
	~ListLambdaBindData() override;

	LogicalType stype;
	unique_ptr<Expression> lambda_expr;

public:
	bool Equals(const FunctionData &other_p) const override;
	unique_ptr<FunctionData> Copy() const override;
	static void Serialize(FieldWriter &writer, const FunctionData *bind_data_p, const ScalarFunction &function) {
		throw NotImplementedException("FIXME: list lambda serialize");
	}
	static unique_ptr<FunctionData> Deserialize(ClientContext &context, FieldReader &reader,
	                                            ScalarFunction &bound_function) {
		throw NotImplementedException("FIXME: list lambda deserialize");
	}
};

ListLambdaBindData::ListLambdaBindData(const LogicalType &stype_p, unique_ptr<Expression> lambda_expr_p)
    : stype(stype_p), lambda_expr(std::move(lambda_expr_p)) {
}

unique_ptr<FunctionData> ListLambdaBindData::Copy() const {
	return make_uniq<ListLambdaBindData>(stype, lambda_expr->Copy());
}

bool ListLambdaBindData::Equals(const FunctionData &other_p) const {
	auto &other = other_p.Cast<ListLambdaBindData>();
	return lambda_expr->Equals(other.lambda_expr.get()) && stype == other.stype;
}

ListLambdaBindData::~ListLambdaBindData() {
}

static void AppendTransformedToResult(Vector &lambda_vector, idx_t &elem_cnt, Vector &result) {

	// append the lambda_vector to the result list
	UnifiedVectorFormat lambda_child_data;
	lambda_vector.ToUnifiedFormat(elem_cnt, lambda_child_data);
	ListVector::Append(result, lambda_vector, *lambda_child_data.sel, elem_cnt, 0);
}

static void AppendFilteredToResult(Vector &lambda_vector, list_entry_t *result_entries, idx_t &elem_cnt, Vector &result,
                                   idx_t &curr_list_len, idx_t &curr_list_offset, idx_t &appended_lists_cnt,
                                   vector<idx_t> &lists_len, idx_t &curr_original_list_len, DataChunk &input_chunk) {

	idx_t true_count = 0;
	SelectionVector true_sel(elem_cnt);
	UnifiedVectorFormat lambda_data;
	lambda_vector.ToUnifiedFormat(elem_cnt, lambda_data);

	auto lambda_values = (bool *)lambda_data.data;
	auto &lambda_validity = lambda_data.validity;

	// compute the new lengths and offsets, and create a selection vector
	for (idx_t i = 0; i < elem_cnt; i++) {
		auto entry = lambda_data.sel->get_index(i);

		while (appended_lists_cnt < lists_len.size() && lists_len[appended_lists_cnt] == 0) {
			result_entries[appended_lists_cnt].offset = curr_list_offset;
			result_entries[appended_lists_cnt].length = 0;
			appended_lists_cnt++;
		}

		// found a true value
		if (lambda_validity.RowIsValid(entry) && lambda_values[entry]) {
			true_sel.set_index(true_count++, i);
			curr_list_len++;
		}

		curr_original_list_len++;

		if (lists_len[appended_lists_cnt] == curr_original_list_len) {
			result_entries[appended_lists_cnt].offset = curr_list_offset;
			result_entries[appended_lists_cnt].length = curr_list_len;
			curr_list_offset += curr_list_len;
			appended_lists_cnt++;
			curr_list_len = 0;
			curr_original_list_len = 0;
		}
	}

	while (appended_lists_cnt < lists_len.size() && lists_len[appended_lists_cnt] == 0) {
		result_entries[appended_lists_cnt].offset = curr_list_offset;
		result_entries[appended_lists_cnt].length = 0;
		appended_lists_cnt++;
	}

	// slice to get the new lists and append them to the result
	Vector new_lists(input_chunk.data[0], true_sel, true_count);
	new_lists.Flatten(true_count);
	UnifiedVectorFormat new_lists_child_data;
	new_lists.ToUnifiedFormat(true_count, new_lists_child_data);
	ListVector::Append(result, new_lists, *new_lists_child_data.sel, true_count, 0);
}

static void ExecuteExpression(vector<LogicalType> &types, vector<LogicalType> &result_types, idx_t &elem_cnt,
                              SelectionVector &sel, vector<SelectionVector> &sel_vectors, DataChunk &input_chunk,
                              DataChunk &lambda_chunk, Vector &child_vector, DataChunk &args,
                              ExpressionExecutor &expr_executor) {

	input_chunk.SetCardinality(elem_cnt);
	lambda_chunk.SetCardinality(elem_cnt);

	// set the list child vector
	Vector slice(child_vector, sel, elem_cnt);
	Vector second_slice(child_vector, sel, elem_cnt);
	slice.Flatten(elem_cnt);
	second_slice.Flatten(elem_cnt);

	input_chunk.data[0].Reference(slice);
	input_chunk.data[1].Reference(second_slice);

	// set the other vectors
	vector<Vector> slices;
	for (idx_t col_idx = 0; col_idx < args.ColumnCount() - 1; col_idx++) {
		slices.emplace_back(args.data[col_idx + 1], sel_vectors[col_idx], elem_cnt);
		slices[col_idx].Flatten(elem_cnt);
		input_chunk.data[col_idx + 2].Reference(slices[col_idx]);
	}

	// execute the lambda expression
	expr_executor.Execute(input_chunk, lambda_chunk);
}

template <bool IS_TRANSFORM = true>
static void ListLambdaFunction(DataChunk &args, ExpressionState &state, Vector &result) {

	// always at least the list argument
	D_ASSERT(args.ColumnCount() >= 1);

	auto count = args.size();
	Vector &lists = args.data[0];

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_entries = FlatVector::GetData<list_entry_t>(result);
	auto &result_validity = FlatVector::Validity(result);

	if (lists.GetType().id() == LogicalTypeId::SQLNULL) {
		result_validity.SetInvalid(0);
		return;
	}

	// e.g. window functions in sub queries return dictionary vectors, which segfault on expression execution
	// if not flattened first
	for (idx_t i = 1; i < args.ColumnCount(); i++) {
		if (args.data[i].GetVectorType() != VectorType::FLAT_VECTOR &&
		    args.data[i].GetVectorType() != VectorType::CONSTANT_VECTOR) {
			args.data[i].Flatten(count);
		}
	}

	// get the lists data
	UnifiedVectorFormat lists_data;
	lists.ToUnifiedFormat(count, lists_data);
	auto list_entries = (list_entry_t *)lists_data.data;

	// get the lambda expression
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &info = func_expr.bind_info->Cast<ListLambdaBindData>();
	auto &lambda_expr = info.lambda_expr;

	// get the child vector and child data
	auto lists_size = ListVector::GetListSize(lists);
	auto &child_vector = ListVector::GetEntry(lists);
	child_vector.Flatten(lists_size);
	UnifiedVectorFormat child_data;
	child_vector.ToUnifiedFormat(lists_size, child_data);

	// to slice the child vector
	SelectionVector sel(STANDARD_VECTOR_SIZE);

	// this vector never contains more than one element
	vector<LogicalType> result_types;
	result_types.push_back(lambda_expr->return_type);

	// non-lambda parameter columns
	vector<UnifiedVectorFormat> columns;
	vector<idx_t> indexes;
	vector<SelectionVector> sel_vectors;

	vector<LogicalType> types;
	types.push_back(child_vector.GetType());
	types.push_back(child_vector.GetType());

	// skip the list column
	for (idx_t i = 1; i < args.ColumnCount(); i++) {
		columns.emplace_back();
		args.data[i].ToUnifiedFormat(count, columns[i - 1]);
		indexes.push_back(0);
		sel_vectors.emplace_back(STANDARD_VECTOR_SIZE);
		types.push_back(args.data[i].GetType());
	}

	// get the expression executor
	ExpressionExecutor expr_executor(state.GetContext(), *lambda_expr);

	// these are only for the list_filter
	vector<idx_t> lists_len;
	idx_t curr_list_len = 0;
	idx_t curr_list_offset = 0;
	idx_t appended_lists_cnt = 0;
	idx_t curr_original_list_len = 0;

	if (!IS_TRANSFORM) {
		lists_len.reserve(count);
	}

	DataChunk input_chunk;
	DataChunk lambda_chunk;
	input_chunk.InitializeEmpty(types);
	lambda_chunk.Initialize(Allocator::DefaultAllocator(), result_types);

	// loop over the child entries and create chunks to be executed by the expression executor
	idx_t elem_cnt = 0;
	idx_t offset = 0;
	for (idx_t row_idx = 0; row_idx < count; row_idx++) {

		auto lists_index = lists_data.sel->get_index(row_idx);
		const auto &list_entry = list_entries[lists_index];

		// set the result to NULL for this row
		if (!lists_data.validity.RowIsValid(lists_index)) {
			result_validity.SetInvalid(row_idx);
			if (!IS_TRANSFORM) {
				lists_len.push_back(0);
			}
			continue;
		}

		// set the length and offset of the resulting lists of list_transform
		if (IS_TRANSFORM) {
			result_entries[row_idx].offset = offset;
			result_entries[row_idx].length = list_entry.length;
			offset += list_entry.length;
		} else {
			lists_len.push_back(list_entry.length);
		}

		// empty list, nothing to execute
		if (list_entry.length == 0) {
			continue;
		}

		// get the data indexes
		for (idx_t col_idx = 0; col_idx < args.ColumnCount() - 1; col_idx++) {
			indexes[col_idx] = columns[col_idx].sel->get_index(row_idx);
		}

		// iterate list elements and create transformed expression columns
		for (idx_t child_idx = 0; child_idx < list_entry.length; child_idx++) {
			// reached STANDARD_VECTOR_SIZE elements
			if (elem_cnt == STANDARD_VECTOR_SIZE) {
				lambda_chunk.Reset();
				ExecuteExpression(types, result_types, elem_cnt, sel, sel_vectors, input_chunk, lambda_chunk,
				                  child_vector, args, expr_executor);

				auto &lambda_vector = lambda_chunk.data[0];

				if (IS_TRANSFORM) {
					AppendTransformedToResult(lambda_vector, elem_cnt, result);
				} else {
					AppendFilteredToResult(lambda_vector, result_entries, elem_cnt, result, curr_list_len,
					                       curr_list_offset, appended_lists_cnt, lists_len, curr_original_list_len,
					                       input_chunk);
				}
				elem_cnt = 0;
			}

			// to slice the child vector
			auto source_idx = child_data.sel->get_index(list_entry.offset + child_idx);
			sel.set_index(elem_cnt, source_idx);

			// for each column, set the index of the selection vector to slice properly
			for (idx_t col_idx = 0; col_idx < args.ColumnCount() - 1; col_idx++) {
				sel_vectors[col_idx].set_index(elem_cnt, indexes[col_idx]);
			}
			elem_cnt++;
		}
	}

	lambda_chunk.Reset();
	ExecuteExpression(types, result_types, elem_cnt, sel, sel_vectors, input_chunk, lambda_chunk, child_vector, args,
	                  expr_executor);
	auto &lambda_vector = lambda_chunk.data[0];

	if (IS_TRANSFORM) {
		AppendTransformedToResult(lambda_vector, elem_cnt, result);
	} else {
		AppendFilteredToResult(lambda_vector, result_entries, elem_cnt, result, curr_list_len, curr_list_offset,
		                       appended_lists_cnt, lists_len, curr_original_list_len, input_chunk);
	}

	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

static void ListTransformFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	ListLambdaFunction<>(args, state, result);
}

static void ListFilterFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	ListLambdaFunction<false>(args, state, result);
}

template <int64_t LAMBDA_PARAM_CNT>
static unique_ptr<FunctionData> ListLambdaBind(ClientContext &context, ScalarFunction &bound_function,
                                               vector<unique_ptr<Expression>> &arguments) {

	auto &bound_lambda_expr = (BoundLambdaExpression &)*arguments[1];
	if (bound_lambda_expr.parameter_count != LAMBDA_PARAM_CNT) {
		throw BinderException("Incorrect number of parameters in lambda function! " + bound_function.name +
		                      " expects " + to_string(LAMBDA_PARAM_CNT) + " parameter(s).");
	}

	if (arguments[0]->return_type.id() == LogicalTypeId::SQLNULL) {
		bound_function.arguments.pop_back();
		bound_function.arguments[0] = LogicalType::SQLNULL;
		bound_function.return_type = LogicalType::SQLNULL;
		return make_uniq<VariableReturnBindData>(bound_function.return_type);
	}

	if (arguments[0]->return_type.id() == LogicalTypeId::UNKNOWN) {
		throw ParameterNotResolvedException();
	}

	D_ASSERT(arguments[0]->return_type.id() == LogicalTypeId::LIST);

	// get the lambda expression and put it in the bind info
	auto lambda_expr = std::move(bound_lambda_expr.lambda_expr);
	return make_uniq<ListLambdaBindData>(bound_function.return_type, std::move(lambda_expr));
}

static unique_ptr<FunctionData> ListTransformBind(ClientContext &context, ScalarFunction &bound_function,
                                                  vector<unique_ptr<Expression>> &arguments) {

	// at least the list column and the lambda function
	D_ASSERT(arguments.size() == 2);
	if (arguments[1]->expression_class != ExpressionClass::BOUND_LAMBDA) {
		throw BinderException("Invalid lambda expression!");
	}

	auto &bound_lambda_expr = (BoundLambdaExpression &)*arguments[1];
	bound_function.return_type = LogicalType::LIST(bound_lambda_expr.lambda_expr->return_type);
	return ListLambdaBind<1>(context, bound_function, arguments);
}

static unique_ptr<FunctionData> ListFilterBind(ClientContext &context, ScalarFunction &bound_function,
                                               vector<unique_ptr<Expression>> &arguments) {

	// at least the list column and the lambda function
	D_ASSERT(arguments.size() == 2);
	if (arguments[1]->expression_class != ExpressionClass::BOUND_LAMBDA) {
		throw BinderException("Invalid lambda expression!");
	}

	// try to cast to boolean, if the return type of the lambda filter expression is not already boolean
	auto &bound_lambda_expr = (BoundLambdaExpression &)*arguments[1];
	if (bound_lambda_expr.lambda_expr->return_type != LogicalType::BOOLEAN) {
		auto cast_lambda_expr =
		    BoundCastExpression::AddCastToType(context, std::move(bound_lambda_expr.lambda_expr), LogicalType::BOOLEAN);
		bound_lambda_expr.lambda_expr = std::move(cast_lambda_expr);
	}

	bound_function.return_type = arguments[0]->return_type;
	return ListLambdaBind<1>(context, bound_function, arguments);
}

ScalarFunction ListTransformFun::GetFunction() {
	ScalarFunction fun({LogicalType::LIST(LogicalType::ANY), LogicalType::LAMBDA}, LogicalType::LIST(LogicalType::ANY),
	                   ListTransformFunction, ListTransformBind, nullptr, nullptr);
	fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	fun.serialize = ListLambdaBindData::Serialize;
	fun.deserialize = ListLambdaBindData::Deserialize;
	return fun;
}

ScalarFunction ListFilterFun::GetFunction() {
	ScalarFunction fun({LogicalType::LIST(LogicalType::ANY), LogicalType::LAMBDA}, LogicalType::LIST(LogicalType::ANY),
	                   ListFilterFunction, ListFilterBind, nullptr, nullptr);
	fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	fun.serialize = ListLambdaBindData::Serialize;
	fun.deserialize = ListLambdaBindData::Deserialize;
	return fun;
}

} // namespace duckdb









namespace duckdb {

struct ListSortBindData : public FunctionData {
	ListSortBindData(OrderType order_type_p, OrderByNullType null_order_p, const LogicalType &return_type_p,
	                 const LogicalType &child_type_p, ClientContext &context_p);
	~ListSortBindData() override;

	OrderType order_type;
	OrderByNullType null_order;
	LogicalType return_type;
	LogicalType child_type;

	vector<LogicalType> types;
	vector<LogicalType> payload_types;

	ClientContext &context;
	RowLayout payload_layout;
	vector<BoundOrderByNode> orders;

public:
	bool Equals(const FunctionData &other_p) const override;
	unique_ptr<FunctionData> Copy() const override;
};

ListSortBindData::ListSortBindData(OrderType order_type_p, OrderByNullType null_order_p,
                                   const LogicalType &return_type_p, const LogicalType &child_type_p,
                                   ClientContext &context_p)
    : order_type(order_type_p), null_order(null_order_p), return_type(return_type_p), child_type(child_type_p),
      context(context_p) {

	// get the vector types
	types.emplace_back(LogicalType::USMALLINT);
	types.emplace_back(child_type);
	D_ASSERT(types.size() == 2);

	// get the payload types
	payload_types.emplace_back(LogicalType::UINTEGER);
	D_ASSERT(payload_types.size() == 1);

	// initialize the payload layout
	payload_layout.Initialize(payload_types);

	// get the BoundOrderByNode
	auto idx_col_expr = make_uniq_base<Expression, BoundReferenceExpression>(LogicalType::USMALLINT, 0);
	auto lists_col_expr = make_uniq_base<Expression, BoundReferenceExpression>(child_type, 1);
	orders.emplace_back(OrderType::ASCENDING, OrderByNullType::ORDER_DEFAULT, std::move(idx_col_expr));
	orders.emplace_back(order_type, null_order, std::move(lists_col_expr));
}

unique_ptr<FunctionData> ListSortBindData::Copy() const {
	return make_uniq<ListSortBindData>(order_type, null_order, return_type, child_type, context);
}

bool ListSortBindData::Equals(const FunctionData &other_p) const {
	auto &other = other_p.Cast<ListSortBindData>();
	return order_type == other.order_type && null_order == other.null_order;
}

ListSortBindData::~ListSortBindData() {
}

// create the key_chunk and the payload_chunk and sink them into the local_sort_state
void SinkDataChunk(Vector *child_vector, SelectionVector &sel, idx_t offset_lists_indices, vector<LogicalType> &types,
                   vector<LogicalType> &payload_types, Vector &payload_vector, LocalSortState &local_sort_state,
                   bool &data_to_sort, Vector &lists_indices) {

	// slice the child vector
	Vector slice(*child_vector, sel, offset_lists_indices);

	// initialize and fill key_chunk
	DataChunk key_chunk;
	key_chunk.InitializeEmpty(types);
	key_chunk.data[0].Reference(lists_indices);
	key_chunk.data[1].Reference(slice);
	key_chunk.SetCardinality(offset_lists_indices);

	// initialize and fill key_chunk and payload_chunk
	DataChunk payload_chunk;
	payload_chunk.InitializeEmpty(payload_types);
	payload_chunk.data[0].Reference(payload_vector);
	payload_chunk.SetCardinality(offset_lists_indices);

	key_chunk.Verify();
	payload_chunk.Verify();

	// sink
	key_chunk.Flatten();
	local_sort_state.SinkChunk(key_chunk, payload_chunk);
	data_to_sort = true;
}

static void ListSortFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(args.ColumnCount() >= 1 && args.ColumnCount() <= 3);
	auto count = args.size();
	Vector &input_lists = args.data[0];

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto &result_validity = FlatVector::Validity(result);

	if (input_lists.GetType().id() == LogicalTypeId::SQLNULL) {
		result_validity.SetInvalid(0);
		return;
	}

	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &info = func_expr.bind_info->Cast<ListSortBindData>();

	// initialize the global and local sorting state
	auto &buffer_manager = BufferManager::GetBufferManager(info.context);
	GlobalSortState global_sort_state(buffer_manager, info.orders, info.payload_layout);
	LocalSortState local_sort_state;
	local_sort_state.Initialize(global_sort_state, buffer_manager);

	// this ensures that we do not change the order of the entries in the input chunk
	VectorOperations::Copy(input_lists, result, count, 0, 0);

	// get the child vector
	auto lists_size = ListVector::GetListSize(result);
	auto &child_vector = ListVector::GetEntry(result);
	UnifiedVectorFormat child_data;
	child_vector.ToUnifiedFormat(lists_size, child_data);

	// get the lists data
	UnifiedVectorFormat lists_data;
	result.ToUnifiedFormat(count, lists_data);
	auto list_entries = (list_entry_t *)lists_data.data;

	// create the lists_indices vector, this contains an element for each list's entry,
	// the element corresponds to the list's index, e.g. for [1, 2, 4], [5, 4]
	// lists_indices contains [0, 0, 0, 1, 1]
	Vector lists_indices(LogicalType::USMALLINT);
	auto lists_indices_data = FlatVector::GetData<uint16_t>(lists_indices);

	// create the payload_vector, this is just a vector containing incrementing integers
	// this will later be used as the 'new' selection vector of the child_vector, after
	// rearranging the payload according to the sorting order
	Vector payload_vector(LogicalType::UINTEGER);
	auto payload_vector_data = FlatVector::GetData<uint32_t>(payload_vector);

	// selection vector pointing to the data of the child vector,
	// used for slicing the child_vector correctly
	SelectionVector sel(STANDARD_VECTOR_SIZE);

	idx_t offset_lists_indices = 0;
	uint32_t incr_payload_count = 0;
	bool data_to_sort = false;

	for (idx_t i = 0; i < count; i++) {
		auto lists_index = lists_data.sel->get_index(i);
		const auto &list_entry = list_entries[lists_index];

		// nothing to do for this list
		if (!lists_data.validity.RowIsValid(lists_index)) {
			result_validity.SetInvalid(i);
			continue;
		}

		// empty list, no sorting required
		if (list_entry.length == 0) {
			continue;
		}

		for (idx_t child_idx = 0; child_idx < list_entry.length; child_idx++) {
			// lists_indices vector is full, sink
			if (offset_lists_indices == STANDARD_VECTOR_SIZE) {
				SinkDataChunk(&child_vector, sel, offset_lists_indices, info.types, info.payload_types, payload_vector,
				              local_sort_state, data_to_sort, lists_indices);
				offset_lists_indices = 0;
			}

			auto source_idx = list_entry.offset + child_idx;
			sel.set_index(offset_lists_indices, source_idx);
			lists_indices_data[offset_lists_indices] = (uint32_t)i;
			payload_vector_data[offset_lists_indices] = source_idx;
			offset_lists_indices++;
			incr_payload_count++;
		}
	}

	if (offset_lists_indices != 0) {
		SinkDataChunk(&child_vector, sel, offset_lists_indices, info.types, info.payload_types, payload_vector,
		              local_sort_state, data_to_sort, lists_indices);
	}

	if (data_to_sort) {
		// add local state to global state, which sorts the data
		global_sort_state.AddLocalState(local_sort_state);
		global_sort_state.PrepareMergePhase();

		// selection vector that is to be filled with the 'sorted' payload
		SelectionVector sel_sorted(incr_payload_count);
		idx_t sel_sorted_idx = 0;

		// scan the sorted row data
		PayloadScanner scanner(*global_sort_state.sorted_blocks[0]->payload_data, global_sort_state);
		for (;;) {
			DataChunk result_chunk;
			result_chunk.Initialize(Allocator::DefaultAllocator(), info.payload_types);
			result_chunk.SetCardinality(0);
			scanner.Scan(result_chunk);
			if (result_chunk.size() == 0) {
				break;
			}

			// construct the selection vector with the new order from the result vectors
			Vector result_vector(result_chunk.data[0]);
			auto result_data = FlatVector::GetData<uint32_t>(result_vector);
			auto row_count = result_chunk.size();

			for (idx_t i = 0; i < row_count; i++) {
				sel_sorted.set_index(sel_sorted_idx, result_data[i]);
				D_ASSERT(result_data[i] < lists_size);
				sel_sorted_idx++;
			}
		}

		D_ASSERT(sel_sorted_idx == incr_payload_count);
		child_vector.Slice(sel_sorted, sel_sorted_idx);
		child_vector.Flatten(sel_sorted_idx);
	}

	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

static unique_ptr<FunctionData> ListSortBind(ClientContext &context, ScalarFunction &bound_function,
                                             vector<unique_ptr<Expression>> &arguments, OrderType &order,
                                             OrderByNullType &null_order) {
	bound_function.arguments[0] = arguments[0]->return_type;
	bound_function.return_type = arguments[0]->return_type;
	auto child_type = ListType::GetChildType(arguments[0]->return_type);

	return make_uniq<ListSortBindData>(order, null_order, bound_function.return_type, child_type, context);
}

template <class T>
static T GetOrder(ClientContext &context, Expression &expr) {
	if (!expr.IsFoldable()) {
		throw InvalidInputException("Sorting order must be a constant");
	}
	Value order_value = ExpressionExecutor::EvaluateScalar(context, expr);
	auto order_name = StringUtil::Upper(order_value.ToString());
	return EnumUtil::FromString<T>(order_name.c_str());
}

static unique_ptr<FunctionData> ListNormalSortBind(ClientContext &context, ScalarFunction &bound_function,
                                                   vector<unique_ptr<Expression>> &arguments) {
	D_ASSERT(!arguments.empty() && arguments.size() <= 3);
	auto order = OrderType::ORDER_DEFAULT;
	auto null_order = OrderByNullType::ORDER_DEFAULT;

	// get the sorting order
	if (arguments.size() >= 2) {
		order = GetOrder<OrderType>(context, *arguments[1]);
	}
	// get the null sorting order
	if (arguments.size() == 3) {
		null_order = GetOrder<OrderByNullType>(context, *arguments[2]);
	}
	auto &config = DBConfig::GetConfig(context);
	order = config.ResolveOrder(order);
	null_order = config.ResolveNullOrder(order, null_order);
	return ListSortBind(context, bound_function, arguments, order, null_order);
}

static unique_ptr<FunctionData> ListReverseSortBind(ClientContext &context, ScalarFunction &bound_function,
                                                    vector<unique_ptr<Expression>> &arguments) {
	auto order = OrderType::ORDER_DEFAULT;
	auto null_order = OrderByNullType::ORDER_DEFAULT;

	if (arguments.size() == 2) {
		null_order = GetOrder<OrderByNullType>(context, *arguments[1]);
	}
	auto &config = DBConfig::GetConfig(context);
	order = config.ResolveOrder(order);
	switch (order) {
	case OrderType::ASCENDING:
		order = OrderType::DESCENDING;
		break;
	case OrderType::DESCENDING:
		order = OrderType::ASCENDING;
		break;
	default:
		throw InternalException("Unexpected order type in list reverse sort");
	}
	null_order = config.ResolveNullOrder(order, null_order);
	return ListSortBind(context, bound_function, arguments, order, null_order);
}

ScalarFunctionSet ListSortFun::GetFunctions() {
	// one parameter: list
	ScalarFunction sort({LogicalType::LIST(LogicalType::ANY)}, LogicalType::LIST(LogicalType::ANY), ListSortFunction,
	                    ListNormalSortBind);

	// two parameters: list, order
	ScalarFunction sort_order({LogicalType::LIST(LogicalType::ANY), LogicalType::VARCHAR},
	                          LogicalType::LIST(LogicalType::ANY), ListSortFunction, ListNormalSortBind);

	// three parameters: list, order, null order
	ScalarFunction sort_orders({LogicalType::LIST(LogicalType::ANY), LogicalType::VARCHAR, LogicalType::VARCHAR},
	                           LogicalType::LIST(LogicalType::ANY), ListSortFunction, ListNormalSortBind);

	ScalarFunctionSet list_sort;
	list_sort.AddFunction(sort);
	list_sort.AddFunction(sort_order);
	list_sort.AddFunction(sort_orders);
	return list_sort;
}

ScalarFunctionSet ListReverseSortFun::GetFunctions() {
	// one parameter: list
	ScalarFunction sort_reverse({LogicalType::LIST(LogicalType::ANY)}, LogicalType::LIST(LogicalType::ANY),
	                            ListSortFunction, ListReverseSortBind);

	// two parameters: list, null order
	ScalarFunction sort_reverse_null_order({LogicalType::LIST(LogicalType::ANY), LogicalType::VARCHAR},
	                                       LogicalType::LIST(LogicalType::ANY), ListSortFunction, ListReverseSortBind);

	ScalarFunctionSet list_reverse_sort;
	list_reverse_sort.AddFunction(sort_reverse);
	list_reverse_sort.AddFunction(sort_reverse_null_order);
	return list_reverse_sort;
}

} // namespace duckdb










namespace duckdb {

static void ListValueFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(result.GetType().id() == LogicalTypeId::LIST);
	auto &child_type = ListType::GetChildType(result.GetType());

	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	for (idx_t i = 0; i < args.ColumnCount(); i++) {
		if (args.data[i].GetVectorType() != VectorType::CONSTANT_VECTOR) {
			result.SetVectorType(VectorType::FLAT_VECTOR);
		}
	}

	auto result_data = FlatVector::GetData<list_entry_t>(result);
	for (idx_t i = 0; i < args.size(); i++) {
		result_data[i].offset = ListVector::GetListSize(result);
		for (idx_t col_idx = 0; col_idx < args.ColumnCount(); col_idx++) {
			auto val = args.GetValue(col_idx, i).DefaultCastAs(child_type);
			ListVector::PushBack(result, val);
		}
		result_data[i].length = args.ColumnCount();
	}
	result.Verify(args.size());
}

static unique_ptr<FunctionData> ListValueBind(ClientContext &context, ScalarFunction &bound_function,
                                              vector<unique_ptr<Expression>> &arguments) {
	// collect names and deconflict, construct return type
	LogicalType child_type = arguments.empty() ? LogicalType::SQLNULL : arguments[0]->return_type;
	for (idx_t i = 1; i < arguments.size(); i++) {
		child_type = LogicalType::MaxLogicalType(child_type, arguments[i]->return_type);
	}

	// this is more for completeness reasons
	bound_function.varargs = child_type;
	bound_function.return_type = LogicalType::LIST(child_type);
	return make_uniq<VariableReturnBindData>(bound_function.return_type);
}

unique_ptr<BaseStatistics> ListValueStats(ClientContext &context, FunctionStatisticsInput &input) {
	auto &child_stats = input.child_stats;
	auto &expr = input.expr;
	auto list_stats = ListStats::CreateEmpty(expr.return_type);
	auto &list_child_stats = ListStats::GetChildStats(list_stats);
	for (idx_t i = 0; i < child_stats.size(); i++) {
		list_child_stats.Merge(child_stats[i]);
	}
	return list_stats.ToUnique();
}

ScalarFunction ListValueFun::GetFunction() {
	// the arguments and return types are actually set in the binder function
	ScalarFunction fun("list_value", {}, LogicalTypeId::LIST, ListValueFunction, ListValueBind, nullptr,
	                   ListValueStats);
	fun.varargs = LogicalType::ANY;
	fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	return fun;
}

} // namespace duckdb






namespace duckdb {

struct NumericRangeInfo {
	using TYPE = int64_t;
	using INCREMENT_TYPE = int64_t;

	static int64_t DefaultStart() {
		return 0;
	}
	static int64_t DefaultIncrement() {
		return 1;
	}

	static uint64_t ListLength(int64_t start_value, int64_t end_value, int64_t increment_value, bool inclusive_bound) {
		if (increment_value == 0) {
			return 0;
		}
		if (start_value > end_value && increment_value > 0) {
			return 0;
		}
		if (start_value < end_value && increment_value < 0) {
			return 0;
		}
		hugeint_t total_diff = AbsValue(hugeint_t(end_value) - hugeint_t(start_value));
		hugeint_t increment = AbsValue(hugeint_t(increment_value));
		hugeint_t total_values = total_diff / increment;
		if (total_diff % increment == 0) {
			if (inclusive_bound) {
				total_values += 1;
			}
		} else {
			total_values += 1;
		}
		if (total_values > NumericLimits<uint32_t>::Maximum()) {
			throw InvalidInputException("Lists larger than 2^32 elements are not supported");
		}
		return Hugeint::Cast<uint64_t>(total_values);
	}

	static void Increment(int64_t &input, int64_t increment) {
		input += increment;
	}
};
struct TimestampRangeInfo {
	using TYPE = timestamp_t;
	using INCREMENT_TYPE = interval_t;

	static timestamp_t DefaultStart() {
		throw InternalException("Default start not implemented for timestamp range");
	}
	static interval_t DefaultIncrement() {
		throw InternalException("Default increment not implemented for timestamp range");
	}
	static uint64_t ListLength(timestamp_t start_value, timestamp_t end_value, interval_t increment_value,
	                           bool inclusive_bound) {
		bool is_positive = increment_value.months > 0 || increment_value.days > 0 || increment_value.micros > 0;
		bool is_negative = increment_value.months < 0 || increment_value.days < 0 || increment_value.micros < 0;
		if (!is_negative && !is_positive) {
			// interval is 0: no result
			return 0;
		}
		// We don't allow infinite bounds because they generate errors or infinite loops
		if (!Timestamp::IsFinite(start_value) || !Timestamp::IsFinite(end_value)) {
			throw InvalidInputException("Interval infinite bounds not supported");
		}

		if (is_negative && is_positive) {
			// we don't allow a mix of
			throw InvalidInputException("Interval with mix of negative/positive entries not supported");
		}
		if (start_value > end_value && is_positive) {
			return 0;
		}
		if (start_value < end_value && is_negative) {
			return 0;
		}
		int64_t total_values = 0;
		if (is_negative) {
			// negative interval, start_value is going down
			while (inclusive_bound ? start_value >= end_value : start_value > end_value) {
				start_value = Interval::Add(start_value, increment_value);
				total_values++;
				if (total_values > NumericLimits<uint32_t>::Maximum()) {
					throw InvalidInputException("Lists larger than 2^32 elements are not supported");
				}
			}
		} else {
			// positive interval, start_value is going up
			while (inclusive_bound ? start_value <= end_value : start_value < end_value) {
				start_value = Interval::Add(start_value, increment_value);
				total_values++;
				if (total_values > NumericLimits<uint32_t>::Maximum()) {
					throw InvalidInputException("Lists larger than 2^32 elements are not supported");
				}
			}
		}
		return total_values;
	}

	static void Increment(timestamp_t &input, interval_t increment) {
		input = Interval::Add(input, increment);
	}
};

template <class OP, bool INCLUSIVE_BOUND>
class RangeInfoStruct {
public:
	explicit RangeInfoStruct(DataChunk &args_p) : args(args_p) {
		switch (args.ColumnCount()) {
		case 1:
			args.data[0].ToUnifiedFormat(args.size(), vdata[0]);
			break;
		case 2:
			args.data[0].ToUnifiedFormat(args.size(), vdata[0]);
			args.data[1].ToUnifiedFormat(args.size(), vdata[1]);
			break;
		case 3:
			args.data[0].ToUnifiedFormat(args.size(), vdata[0]);
			args.data[1].ToUnifiedFormat(args.size(), vdata[1]);
			args.data[2].ToUnifiedFormat(args.size(), vdata[2]);
			break;
		default:
			throw InternalException("Unsupported number of parameters for range");
		}
	}

	bool RowIsValid(idx_t row_idx) {
		for (idx_t i = 0; i < args.ColumnCount(); i++) {
			auto idx = vdata[i].sel->get_index(row_idx);
			if (!vdata[i].validity.RowIsValid(idx)) {
				return false;
			}
		}
		return true;
	}

	typename OP::TYPE StartListValue(idx_t row_idx) {
		if (args.ColumnCount() == 1) {
			return OP::DefaultStart();
		} else {
			auto data = (typename OP::TYPE *)vdata[0].data;
			auto idx = vdata[0].sel->get_index(row_idx);
			return data[idx];
		}
	}

	typename OP::TYPE EndListValue(idx_t row_idx) {
		idx_t vdata_idx = args.ColumnCount() == 1 ? 0 : 1;
		auto data = (typename OP::TYPE *)vdata[vdata_idx].data;
		auto idx = vdata[vdata_idx].sel->get_index(row_idx);
		return data[idx];
	}

	typename OP::INCREMENT_TYPE ListIncrementValue(idx_t row_idx) {
		if (args.ColumnCount() < 3) {
			return OP::DefaultIncrement();
		} else {
			auto data = (typename OP::INCREMENT_TYPE *)vdata[2].data;
			auto idx = vdata[2].sel->get_index(row_idx);
			return data[idx];
		}
	}

	void GetListValues(idx_t row_idx, typename OP::TYPE &start_value, typename OP::TYPE &end_value,
	                   typename OP::INCREMENT_TYPE &increment_value) {
		start_value = StartListValue(row_idx);
		end_value = EndListValue(row_idx);
		increment_value = ListIncrementValue(row_idx);
	}

	uint64_t ListLength(idx_t row_idx) {
		typename OP::TYPE start_value;
		typename OP::TYPE end_value;
		typename OP::INCREMENT_TYPE increment_value;
		GetListValues(row_idx, start_value, end_value, increment_value);
		return OP::ListLength(start_value, end_value, increment_value, INCLUSIVE_BOUND);
	}

private:
	DataChunk &args;
	UnifiedVectorFormat vdata[3];
};

template <class OP, bool INCLUSIVE_BOUND>
static void ListRangeFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(result.GetType().id() == LogicalTypeId::LIST);

	RangeInfoStruct<OP, INCLUSIVE_BOUND> info(args);
	idx_t args_size = 1;
	auto result_type = VectorType::CONSTANT_VECTOR;
	for (idx_t i = 0; i < args.ColumnCount(); i++) {
		if (args.data[i].GetVectorType() != VectorType::CONSTANT_VECTOR) {
			args_size = args.size();
			result_type = VectorType::FLAT_VECTOR;
			break;
		}
	}
	auto list_data = FlatVector::GetData<list_entry_t>(result);
	auto &result_validity = FlatVector::Validity(result);
	int64_t total_size = 0;
	for (idx_t i = 0; i < args_size; i++) {
		if (!info.RowIsValid(i)) {
			result_validity.SetInvalid(i);
			list_data[i].offset = total_size;
			list_data[i].length = 0;
		} else {
			list_data[i].offset = total_size;
			list_data[i].length = info.ListLength(i);
			total_size += list_data[i].length;
		}
	}

	// now construct the child vector of the list
	ListVector::Reserve(result, total_size);
	auto range_data = FlatVector::GetData<typename OP::TYPE>(ListVector::GetEntry(result));
	idx_t total_idx = 0;
	for (idx_t i = 0; i < args_size; i++) {
		typename OP::TYPE start_value = info.StartListValue(i);
		typename OP::INCREMENT_TYPE increment = info.ListIncrementValue(i);

		typename OP::TYPE range_value = start_value;
		for (idx_t range_idx = 0; range_idx < list_data[i].length; range_idx++) {
			if (range_idx > 0) {
				OP::Increment(range_value, increment);
			}
			range_data[total_idx++] = range_value;
		}
	}

	ListVector::SetListSize(result, total_size);
	result.SetVectorType(result_type);

	result.Verify(args.size());
}

ScalarFunctionSet ListRangeFun::GetFunctions() {
	// the arguments and return types are actually set in the binder function
	ScalarFunctionSet range_set;
	range_set.AddFunction(ScalarFunction({LogicalType::BIGINT}, LogicalType::LIST(LogicalType::BIGINT),
	                                     ListRangeFunction<NumericRangeInfo, false>));
	range_set.AddFunction(ScalarFunction({LogicalType::BIGINT, LogicalType::BIGINT},
	                                     LogicalType::LIST(LogicalType::BIGINT),
	                                     ListRangeFunction<NumericRangeInfo, false>));
	range_set.AddFunction(ScalarFunction({LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT},
	                                     LogicalType::LIST(LogicalType::BIGINT),
	                                     ListRangeFunction<NumericRangeInfo, false>));
	range_set.AddFunction(ScalarFunction({LogicalType::TIMESTAMP, LogicalType::TIMESTAMP, LogicalType::INTERVAL},
	                                     LogicalType::LIST(LogicalType::TIMESTAMP),
	                                     ListRangeFunction<TimestampRangeInfo, false>));
	return range_set;
}

ScalarFunctionSet GenerateSeriesFun::GetFunctions() {
	ScalarFunctionSet generate_series;
	generate_series.AddFunction(ScalarFunction({LogicalType::BIGINT}, LogicalType::LIST(LogicalType::BIGINT),
	                                           ListRangeFunction<NumericRangeInfo, true>));
	generate_series.AddFunction(ScalarFunction({LogicalType::BIGINT, LogicalType::BIGINT},
	                                           LogicalType::LIST(LogicalType::BIGINT),
	                                           ListRangeFunction<NumericRangeInfo, true>));
	generate_series.AddFunction(ScalarFunction({LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT},
	                                           LogicalType::LIST(LogicalType::BIGINT),
	                                           ListRangeFunction<NumericRangeInfo, true>));
	generate_series.AddFunction(ScalarFunction({LogicalType::TIMESTAMP, LogicalType::TIMESTAMP, LogicalType::INTERVAL},
	                                           LogicalType::LIST(LogicalType::TIMESTAMP),
	                                           ListRangeFunction<TimestampRangeInfo, true>));
	return generate_series;
}

} // namespace duckdb







namespace duckdb {

static void CardinalityFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &map = args.data[0];
	UnifiedVectorFormat map_data;
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_data = FlatVector::GetData<uint64_t>(result);
	auto &result_validity = FlatVector::Validity(result);

	map.ToUnifiedFormat(args.size(), map_data);
	for (idx_t row = 0; row < args.size(); row++) {
		auto list_entry = ((list_entry_t *)map_data.data)[map_data.sel->get_index(row)];
		result_data[row] = list_entry.length;
		result_validity.Set(row, map_data.validity.RowIsValid(map_data.sel->get_index(row)));
	}

	if (args.size() == 1) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

static unique_ptr<FunctionData> CardinalityBind(ClientContext &context, ScalarFunction &bound_function,
                                                vector<unique_ptr<Expression>> &arguments) {
	if (arguments.size() != 1) {
		throw BinderException("Cardinality must have exactly one arguments");
	}

	if (arguments[0]->return_type.id() != LogicalTypeId::MAP) {
		throw BinderException("Cardinality can only operate on MAPs");
	}

	bound_function.return_type = LogicalType::UBIGINT;
	return make_uniq<VariableReturnBindData>(bound_function.return_type);
}

ScalarFunction CardinalityFun::GetFunction() {
	ScalarFunction fun({LogicalType::ANY}, LogicalType::UBIGINT, CardinalityFunction, CardinalityBind);
	fun.varargs = LogicalType::ANY;
	fun.null_handling = FunctionNullHandling::DEFAULT_NULL_HANDLING;
	return fun;
}

} // namespace duckdb









namespace duckdb {

// Example:
// source: [1,2,3], expansion_factor: 4
// target (result): [1,2,3,1,2,3,1,2,3,1,2,3]
static void CreateExpandedVector(const Vector &source, Vector &target, idx_t expansion_factor) {
	idx_t count = ListVector::GetListSize(source);
	auto &entry = ListVector::GetEntry(source);

	idx_t target_idx = 0;
	for (idx_t copy = 0; copy < expansion_factor; copy++) {
		for (idx_t key_idx = 0; key_idx < count; key_idx++) {
			target.SetValue(target_idx, entry.GetValue(key_idx));
			target_idx++;
		}
	}
	D_ASSERT(target_idx == count * expansion_factor);
}

static void AlignVectorToReference(const Vector &original, const Vector &reference, idx_t tuple_count, Vector &result) {
	auto original_length = ListVector::GetListSize(original);
	auto new_length = ListVector::GetListSize(reference);

	Vector expanded_const(ListType::GetChildType(original.GetType()), new_length);

	auto expansion_factor = new_length / original_length;
	if (expansion_factor != tuple_count) {
		throw InvalidInputException("Error in MAP creation: key list and value list do not align. i.e. different "
		                            "size or incompatible structure");
	}
	CreateExpandedVector(original, expanded_const, expansion_factor);
	result.Reference(expanded_const);
}

static bool ListEntriesEqual(Vector &keys, Vector &values, idx_t count) {
	auto key_count = ListVector::GetListSize(keys);
	auto value_count = ListVector::GetListSize(values);
	bool same_vector_type = keys.GetVectorType() == values.GetVectorType();

	D_ASSERT(keys.GetType().id() == LogicalTypeId::LIST);
	D_ASSERT(values.GetType().id() == LogicalTypeId::LIST);

	UnifiedVectorFormat keys_data;
	UnifiedVectorFormat values_data;

	keys.ToUnifiedFormat(count, keys_data);
	values.ToUnifiedFormat(count, values_data);

	auto keys_entries = (list_entry_t *)keys_data.data;
	auto values_entries = (list_entry_t *)values_data.data;

	if (same_vector_type) {
		const auto key_data = keys_data.data;
		const auto value_data = values_data.data;

		if (keys.GetVectorType() == VectorType::CONSTANT_VECTOR) {
			D_ASSERT(values.GetVectorType() == VectorType::CONSTANT_VECTOR);
			// Only need to compare one entry in this case
			return memcmp(key_data, value_data, sizeof(list_entry_t)) == 0;
		}

		// Fast path if the vector types are equal, can just check if the entries are the same
		if (key_count != value_count) {
			return false;
		}
		return memcmp(key_data, value_data, count * sizeof(list_entry_t)) == 0;
	}

	// Compare the list_entries one by one
	for (idx_t i = 0; i < count; i++) {
		auto keys_idx = keys_data.sel->get_index(i);
		auto values_idx = values_data.sel->get_index(i);

		if (keys_entries[keys_idx] != values_entries[values_idx]) {
			return false;
		}
	}
	return true;
}

static void MapFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(result.GetType().id() == LogicalTypeId::MAP);

	auto &key_vector = MapVector::GetKeys(result);
	auto &value_vector = MapVector::GetValues(result);
	auto result_data = ListVector::GetData(result);

	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	if (args.data.empty()) {
		ListVector::SetListSize(result, 0);
		result_data->offset = 0;
		result_data->length = 0;
		result.Verify(args.size());
		return;
	}

	bool keys_are_const = args.data[0].GetVectorType() == VectorType::CONSTANT_VECTOR;
	bool values_are_const = args.data[1].GetVectorType() == VectorType::CONSTANT_VECTOR;
	if (!keys_are_const || !values_are_const) {
		result.SetVectorType(VectorType::FLAT_VECTOR);
	}

	auto key_count = ListVector::GetListSize(args.data[0]);
	auto value_count = ListVector::GetListSize(args.data[1]);
	auto key_data = ListVector::GetData(args.data[0]);
	auto value_data = ListVector::GetData(args.data[1]);
	auto src_data = key_data;

	if (keys_are_const && !values_are_const) {
		AlignVectorToReference(args.data[0], args.data[1], args.size(), key_vector);
		src_data = value_data;
	} else if (values_are_const && !keys_are_const) {
		AlignVectorToReference(args.data[1], args.data[0], args.size(), value_vector);
	} else {
		if (!ListEntriesEqual(args.data[0], args.data[1], args.size())) {
			throw InvalidInputException("Error in MAP creation: key list and value list do not align. i.e. different "
			                            "size or incompatible structure");
		}
	}

	ListVector::SetListSize(result, MaxValue(key_count, value_count));

	result_data = ListVector::GetData(result);
	for (idx_t i = 0; i < args.size(); i++) {
		result_data[i] = src_data[i];
	}

	// check whether one of the vectors has already been referenced to an expanded vector in the case of const/non-const
	// combination. If not, then referencing is still necessary
	if (!(keys_are_const && !values_are_const)) {
		key_vector.Reference(ListVector::GetEntry(args.data[0]));
	}
	if (!(values_are_const && !keys_are_const)) {
		value_vector.Reference(ListVector::GetEntry(args.data[1]));
	}

	MapVector::MapConversionVerify(result, args.size());
	result.Verify(args.size());
}

static unique_ptr<FunctionData> MapBind(ClientContext &context, ScalarFunction &bound_function,
                                        vector<unique_ptr<Expression>> &arguments) {
	child_list_t<LogicalType> child_types;

	if (arguments.size() != 2 && !arguments.empty()) {
		throw Exception("We need exactly two lists for a map");
	}
	if (arguments.size() == 2) {
		if (arguments[0]->return_type.id() != LogicalTypeId::LIST) {
			throw Exception("First argument is not a list");
		}
		if (arguments[1]->return_type.id() != LogicalTypeId::LIST) {
			throw Exception("Second argument is not a list");
		}
		child_types.push_back(make_pair("key", arguments[0]->return_type));
		child_types.push_back(make_pair("value", arguments[1]->return_type));
	}

	if (arguments.empty()) {
		auto empty = LogicalType::LIST(LogicalTypeId::SQLNULL);
		child_types.push_back(make_pair("key", empty));
		child_types.push_back(make_pair("value", empty));
	}

	bound_function.return_type =
	    LogicalType::MAP(ListType::GetChildType(child_types[0].second), ListType::GetChildType(child_types[1].second));

	return make_uniq<VariableReturnBindData>(bound_function.return_type);
}

ScalarFunction MapFun::GetFunction() {
	//! the arguments and return types are actually set in the binder function
	ScalarFunction fun({}, LogicalTypeId::MAP, MapFunction, MapBind);
	fun.varargs = LogicalType::ANY;
	fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	return fun;
}

} // namespace duckdb










namespace duckdb {

namespace {

struct MapKeyIndexPair {
	MapKeyIndexPair(idx_t map, idx_t key) : map_index(map), key_index(key) {
	}
	// The index of the map that this key comes from
	idx_t map_index;
	// The index within the maps key_list
	idx_t key_index;
};

} // namespace

vector<Value> GetListEntries(vector<Value> keys, vector<Value> values) {
	D_ASSERT(keys.size() == values.size());
	vector<Value> entries;
	for (idx_t i = 0; i < keys.size(); i++) {
		child_list_t<Value> children;
		children.emplace_back(make_pair("key", std::move(keys[i])));
		children.emplace_back(make_pair("value", std::move(values[i])));
		entries.push_back(Value::STRUCT(std::move(children)));
	}
	return entries;
}

static void MapConcatFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	if (result.GetType().id() == LogicalTypeId::SQLNULL) {
		// All inputs are NULL, just return NULL
		auto &validity = FlatVector::Validity(result);
		validity.SetInvalid(0);
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
		return;
	}
	D_ASSERT(result.GetType().id() == LogicalTypeId::MAP);
	auto count = args.size();

	auto map_count = args.ColumnCount();
	vector<UnifiedVectorFormat> map_formats(map_count);
	for (idx_t i = 0; i < map_count; i++) {
		auto &map = args.data[i];
		map.ToUnifiedFormat(count, map_formats[i]);
	}
	auto result_data = FlatVector::GetData<list_entry_t>(result);

	for (idx_t i = 0; i < count; i++) {
		// Loop through all the maps per list
		// we cant do better because all the entries of the child vector have to be contiguous
		// so we cant start the next row before we have finished the one before it
		auto &result_entry = result_data[i];
		vector<MapKeyIndexPair> index_to_map;
		vector<Value> keys_list;
		for (idx_t map_idx = 0; map_idx < map_count; map_idx++) {
			if (args.data[map_idx].GetType().id() == LogicalTypeId::SQLNULL) {
				continue;
			}
			auto &map_format = map_formats[map_idx];
			auto &keys = MapVector::GetKeys(args.data[map_idx]);

			auto index = map_format.sel->get_index(i);
			auto entry = ((list_entry_t *)map_format.data)[index];

			// Update the list for this row
			for (idx_t list_idx = 0; list_idx < entry.length; list_idx++) {
				auto key_index = entry.offset + list_idx;
				auto key = keys.GetValue(key_index);
				auto entry = std::find(keys_list.begin(), keys_list.end(), key);
				if (entry == keys_list.end()) {
					// Result list does not contain this value yet
					keys_list.push_back(key);
					index_to_map.emplace_back(map_idx, key_index);
				} else {
					// Result list already contains this, update where to find the value at
					auto distance = std::distance(keys_list.begin(), entry);
					auto &mapping = *(index_to_map.begin() + distance);
					mapping.key_index = key_index;
					mapping.map_index = map_idx;
				}
			}
		}
		vector<Value> values_list;
		D_ASSERT(keys_list.size() == index_to_map.size());
		// Get the values from the mapping
		for (auto &mapping : index_to_map) {
			auto &map = args.data[mapping.map_index];
			auto &values = MapVector::GetValues(map);
			values_list.push_back(values.GetValue(mapping.key_index));
		}
		idx_t entries_count = keys_list.size();
		D_ASSERT(values_list.size() == keys_list.size());
		result_entry.offset = ListVector::GetListSize(result);
		result_entry.length = values_list.size();
		auto list_entries = GetListEntries(std::move(keys_list), std::move(values_list));
		for (auto &list_entry : list_entries) {
			ListVector::PushBack(result, list_entry);
		}
		ListVector::SetListSize(result, ListVector::GetListSize(result) + entries_count);
	}

	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
	result.Verify(count);
}

static bool IsEmptyMap(const LogicalType &map) {
	D_ASSERT(map.id() == LogicalTypeId::MAP);
	auto &key_type = MapType::KeyType(map);
	auto &value_type = MapType::ValueType(map);
	return key_type.id() == LogicalType::SQLNULL && value_type.id() == LogicalType::SQLNULL;
}

static unique_ptr<FunctionData> MapConcatBind(ClientContext &context, ScalarFunction &bound_function,
                                              vector<unique_ptr<Expression>> &arguments) {

	auto arg_count = arguments.size();
	if (arg_count < 2) {
		throw InvalidInputException("The provided amount of arguments is incorrect, please provide 2 or more maps");
	}

	if (arguments[0]->return_type.id() == LogicalTypeId::UNKNOWN) {
		// Prepared statement
		bound_function.arguments.emplace_back(LogicalTypeId::UNKNOWN);
		bound_function.return_type = LogicalType(LogicalTypeId::SQLNULL);
		return nullptr;
	}

	LogicalType expected = LogicalType::SQLNULL;

	bool is_null = true;
	// Check and verify that all the maps are of the same type
	for (idx_t i = 0; i < arg_count; i++) {
		auto &arg = arguments[i];
		auto &map = arg->return_type;
		if (map.id() == LogicalTypeId::UNKNOWN) {
			// Prepared statement
			bound_function.arguments.emplace_back(LogicalTypeId::UNKNOWN);
			bound_function.return_type = LogicalType(LogicalTypeId::SQLNULL);
			return nullptr;
		}
		if (map.id() == LogicalTypeId::SQLNULL) {
			// The maps are allowed to be NULL
			continue;
		}
		is_null = false;
		if (IsEmptyMap(map)) {
			// Map is allowed to be empty
			continue;
		}

		if (expected.id() == LogicalTypeId::SQLNULL) {
			expected = map;
		} else if (map != expected) {
			throw InvalidInputException(
			    "'value' type of map differs between arguments, expected '%s', found '%s' instead", expected.ToString(),
			    map.ToString());
		}
	}

	if (expected.id() == LogicalTypeId::SQLNULL && is_null == false) {
		expected = LogicalType::MAP(LogicalType::SQLNULL, LogicalType::SQLNULL);
	}
	bound_function.return_type = expected;
	return make_uniq<VariableReturnBindData>(bound_function.return_type);
}

ScalarFunction MapConcatFun::GetFunction() {
	//! the arguments and return types are actually set in the binder function
	ScalarFunction fun("map_concat", {}, LogicalTypeId::LIST, MapConcatFunction, MapConcatBind);
	fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	fun.varargs = LogicalType::ANY;
	return fun;
}

} // namespace duckdb








namespace duckdb {

// Reverse of map_from_entries
static void MapEntriesFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	idx_t count = args.size();

	result.Reinterpret(args.data[0]);

	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
	result.Verify(count);
}

static unique_ptr<FunctionData> MapEntriesBind(ClientContext &context, ScalarFunction &bound_function,
                                               vector<unique_ptr<Expression>> &arguments) {
	child_list_t<LogicalType> child_types;

	if (arguments.size() != 1) {
		throw InvalidInputException("Too many arguments provided, only expecting a single map");
	}
	auto &map = arguments[0]->return_type;

	if (map.id() == LogicalTypeId::UNKNOWN) {
		// Prepared statement
		bound_function.arguments.emplace_back(LogicalTypeId::UNKNOWN);
		bound_function.return_type = LogicalType(LogicalTypeId::SQLNULL);
		return nullptr;
	}

	if (map.id() != LogicalTypeId::MAP) {
		throw InvalidInputException("The provided argument is not a map");
	}
	auto &key_type = MapType::KeyType(map);
	auto &value_type = MapType::ValueType(map);

	child_types.push_back(make_pair("k", key_type));
	child_types.push_back(make_pair("v", value_type));

	auto row_type = LogicalType::STRUCT(child_types);

	bound_function.return_type = LogicalType::LIST(row_type);
	return make_uniq<VariableReturnBindData>(bound_function.return_type);
}

ScalarFunction MapEntriesFun::GetFunction() {
	//! the arguments and return types are actually set in the binder function
	ScalarFunction fun({}, LogicalTypeId::LIST, MapEntriesFunction, MapEntriesBind);
	fun.null_handling = FunctionNullHandling::DEFAULT_NULL_HANDLING;
	fun.varargs = LogicalType::ANY;
	return fun;
}

} // namespace duckdb







namespace duckdb {

struct MapKeyArgFunctor {
	// MAP is a LIST(STRUCT(K,V))
	// meaning the MAP itself is a List, but the child vector that we're interested in (the keys)
	// are a level deeper than the initial child vector

	static Vector &GetList(Vector &map) {
		return map;
	}
	static idx_t GetListSize(Vector &map) {
		return ListVector::GetListSize(map);
	}
	static Vector &GetEntry(Vector &map) {
		return MapVector::GetKeys(map);
	}
};

void FillResult(Vector &map, Vector &offsets, Vector &result, idx_t count) {
	UnifiedVectorFormat map_data;
	map.ToUnifiedFormat(count, map_data);

	UnifiedVectorFormat offset_data;
	offsets.ToUnifiedFormat(count, offset_data);

	UnifiedVectorFormat result_data;
	result.ToUnifiedFormat(count, result_data);

	auto entry_count = ListVector::GetListSize(map);
	auto &values_entries = MapVector::GetValues(map);
	UnifiedVectorFormat values_entry_data;
	// Note: this vector can have a different size than the map
	values_entries.ToUnifiedFormat(entry_count, values_entry_data);

	for (idx_t row = 0; row < count; row++) {
		idx_t offset_idx = offset_data.sel->get_index(row);
		auto offset = ((int32_t *)offset_data.data)[offset_idx];

		// Get the current size of the list, for the offset
		idx_t current_offset = ListVector::GetListSize(result);
		if (!offset_data.validity.RowIsValid(offset_idx) || !offset) {
			// Set the entry data for this result row
			idx_t result_index = result_data.sel->get_index(row);
			auto &entry = ((list_entry_t *)result_data.data)[result_index];
			entry.length = 0;
			entry.offset = current_offset;
			continue;
		}
		// All list indices start at 1, reduce by 1 to get the actual index
		offset--;

		// Get the 'values' list entry corresponding to the offset
		idx_t value_index = map_data.sel->get_index(row);
		auto &value_list_entry = ((list_entry_t *)map_data.data)[value_index];

		// Add the values to the result
		idx_t list_offset = value_list_entry.offset + offset;
		// All keys are unique, only one will ever match
		idx_t length = 1;
		ListVector::Append(result, values_entries, length + list_offset, list_offset);

		// Set the entry data for this result row
		idx_t result_index = result_data.sel->get_index(row);
		auto &entry = ((list_entry_t *)result_data.data)[result_index];
		entry.length = length;
		entry.offset = current_offset;
	}
}

static void MapExtractFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(args.data.size() == 2);
	D_ASSERT(args.data[0].GetType().id() == LogicalTypeId::MAP);
	result.SetVectorType(VectorType::FLAT_VECTOR);

	idx_t tuple_count = args.size();
	// Optimization: because keys are not allowed to be NULL, we can early-out
	if (args.data[1].GetType().id() == LogicalTypeId::SQLNULL) {
		//! We don't need to look through the map if the 'key' to look for is NULL
		ListVector::SetListSize(result, 0);
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
		auto list_data = ConstantVector::GetData<list_entry_t>(result);
		list_data->offset = 0;
		list_data->length = 0;
		result.Verify(tuple_count);
		return;
	}

	auto &map = args.data[0];
	auto &key = args.data[1];

	UnifiedVectorFormat map_data;

	// Create the chunk we'll feed to ListPosition
	DataChunk list_position_chunk;
	vector<LogicalType> chunk_types;
	chunk_types.reserve(2);
	chunk_types.push_back(map.GetType());
	chunk_types.push_back(key.GetType());
	list_position_chunk.InitializeEmpty(chunk_types.begin(), chunk_types.end());

	// Populate it with the map keys list and the key vector
	list_position_chunk.data[0].Reference(map);
	list_position_chunk.data[1].Reference(key);
	list_position_chunk.SetCardinality(tuple_count);

	Vector position_vector(LogicalType::LIST(LogicalType::INTEGER), tuple_count);
	// We can pass around state as it's not used by ListPositionFunction anyways
	ListContainsOrPosition<int32_t, PositionFunctor, MapKeyArgFunctor>(list_position_chunk, position_vector);

	FillResult(map, position_vector, result, tuple_count);

	if (tuple_count == 1) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}

	result.Verify(tuple_count);
}

static unique_ptr<FunctionData> MapExtractBind(ClientContext &context, ScalarFunction &bound_function,
                                               vector<unique_ptr<Expression>> &arguments) {
	if (arguments.size() != 2) {
		throw BinderException("MAP_EXTRACT must have exactly two arguments");
	}
	if (arguments[0]->return_type.id() != LogicalTypeId::MAP) {
		throw BinderException("MAP_EXTRACT can only operate on MAPs");
	}
	auto &value_type = MapType::ValueType(arguments[0]->return_type);

	//! Here we have to construct the List Type that will be returned
	bound_function.return_type = LogicalType::LIST(value_type);
	auto key_type = MapType::KeyType(arguments[0]->return_type);
	if (key_type.id() != LogicalTypeId::SQLNULL && arguments[1]->return_type.id() != LogicalTypeId::SQLNULL) {
		bound_function.arguments[1] = MapType::KeyType(arguments[0]->return_type);
	}
	return make_uniq<VariableReturnBindData>(value_type);
}

ScalarFunction MapExtractFun::GetFunction() {
	ScalarFunction fun({LogicalType::ANY, LogicalType::ANY}, LogicalType::ANY, MapExtractFunction, MapExtractBind);
	fun.varargs = LogicalType::ANY;
	fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	return fun;
}

} // namespace duckdb







namespace duckdb {

static void MapFromEntriesFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto count = args.size();

	result.Reinterpret(args.data[0]);

	MapVector::MapConversionVerify(result, count);
	result.Verify(count);

	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

static unique_ptr<FunctionData> MapFromEntriesBind(ClientContext &context, ScalarFunction &bound_function,
                                                   vector<unique_ptr<Expression>> &arguments) {
	if (arguments.size() != 1) {
		throw InvalidInputException("The input argument must be a list of structs.");
	}
	auto &list = arguments[0]->return_type;

	if (list.id() == LogicalTypeId::UNKNOWN) {
		bound_function.arguments.emplace_back(LogicalTypeId::UNKNOWN);
		bound_function.return_type = LogicalType(LogicalTypeId::SQLNULL);
		return nullptr;
	}

	if (list.id() != LogicalTypeId::LIST) {
		throw InvalidInputException("The provided argument is not a list of structs");
	}
	auto &elem_type = ListType::GetChildType(list);
	if (elem_type.id() != LogicalTypeId::STRUCT) {
		throw InvalidInputException("The elements of the list must be structs");
	}
	auto &children = StructType::GetChildTypes(elem_type);
	if (children.size() != 2) {
		throw InvalidInputException("The provided struct type should only contain 2 fields, a key and a value");
	}

	bound_function.return_type = LogicalType::MAP(elem_type);
	return make_uniq<VariableReturnBindData>(bound_function.return_type);
}

ScalarFunction MapFromEntriesFun::GetFunction() {
	//! the arguments and return types are actually set in the binder function
	ScalarFunction fun({}, LogicalTypeId::MAP, MapFromEntriesFunction, MapFromEntriesBind);
	fun.null_handling = FunctionNullHandling::DEFAULT_NULL_HANDLING;
	fun.varargs = LogicalType::ANY;
	return fun;
}

} // namespace duckdb








namespace duckdb {

static void MapKeyValueFunction(DataChunk &args, ExpressionState &state, Vector &result,
                                Vector &(*get_child_vector)(Vector &)) {
	D_ASSERT(result.GetType().id() == LogicalTypeId::LIST);
	auto count = args.size();

	auto &map = args.data[0];
	D_ASSERT(map.GetType().id() == LogicalTypeId::MAP);
	auto child = get_child_vector(map);

	auto &entries = ListVector::GetEntry(result);
	entries.Reference(child);

	UnifiedVectorFormat map_data;
	map.ToUnifiedFormat(count, map_data);

	D_ASSERT(result.GetVectorType() == VectorType::FLAT_VECTOR);
	FlatVector::SetData(result, map_data.data);
	FlatVector::SetValidity(result, map_data.validity);
	auto list_size = ListVector::GetListSize(map);
	ListVector::SetListSize(result, list_size);
	if (map.GetVectorType() == VectorType::DICTIONARY_VECTOR) {
		result.Slice(*map_data.sel, count);
	}
	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
	result.Verify(count);
}

static void MapKeysFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	MapKeyValueFunction(args, state, result, MapVector::GetKeys);
}

static void MapValuesFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	MapKeyValueFunction(args, state, result, MapVector::GetValues);
}

static unique_ptr<FunctionData> MapKeyValueBind(ClientContext &context, ScalarFunction &bound_function,
                                                vector<unique_ptr<Expression>> &arguments,
                                                const LogicalType &(*type_func)(const LogicalType &)) {
	if (arguments.size() != 1) {
		throw InvalidInputException("Too many arguments provided, only expecting a single map");
	}
	auto &map = arguments[0]->return_type;

	if (map.id() == LogicalTypeId::UNKNOWN) {
		// Prepared statement
		bound_function.arguments.emplace_back(LogicalTypeId::UNKNOWN);
		bound_function.return_type = LogicalType(LogicalTypeId::SQLNULL);
		return nullptr;
	}

	if (map.id() != LogicalTypeId::MAP) {
		throw InvalidInputException("The provided argument is not a map");
	}

	auto &type = type_func(map);

	bound_function.return_type = LogicalType::LIST(type);
	return make_uniq<VariableReturnBindData>(bound_function.return_type);
}

static unique_ptr<FunctionData> MapKeysBind(ClientContext &context, ScalarFunction &bound_function,
                                            vector<unique_ptr<Expression>> &arguments) {
	return MapKeyValueBind(context, bound_function, arguments, MapType::KeyType);
}

static unique_ptr<FunctionData> MapValuesBind(ClientContext &context, ScalarFunction &bound_function,
                                              vector<unique_ptr<Expression>> &arguments) {
	return MapKeyValueBind(context, bound_function, arguments, MapType::ValueType);
}

ScalarFunction MapKeysFun::GetFunction() {
	//! the arguments and return types are actually set in the binder function
	ScalarFunction fun({}, LogicalTypeId::LIST, MapKeysFunction, MapKeysBind);
	fun.null_handling = FunctionNullHandling::DEFAULT_NULL_HANDLING;
	fun.varargs = LogicalType::ANY;
	return fun;
}

ScalarFunction MapValuesFun::GetFunction() {
	ScalarFunction fun({}, LogicalTypeId::LIST, MapValuesFunction, MapValuesBind);
	fun.null_handling = FunctionNullHandling::DEFAULT_NULL_HANDLING;
	fun.varargs = LogicalType::ANY;
	return fun;
}

} // namespace duckdb











#include <cmath>
#include <errno.h>

namespace duckdb {

template <class TR, class OP>
static scalar_function_t GetScalarIntegerUnaryFunctionFixedReturn(const LogicalType &type) {
	scalar_function_t function;
	switch (type.id()) {
	case LogicalTypeId::TINYINT:
		function = &ScalarFunction::UnaryFunction<int8_t, TR, OP>;
		break;
	case LogicalTypeId::SMALLINT:
		function = &ScalarFunction::UnaryFunction<int16_t, TR, OP>;
		break;
	case LogicalTypeId::INTEGER:
		function = &ScalarFunction::UnaryFunction<int32_t, TR, OP>;
		break;
	case LogicalTypeId::BIGINT:
		function = &ScalarFunction::UnaryFunction<int64_t, TR, OP>;
		break;
	case LogicalTypeId::HUGEINT:
		function = &ScalarFunction::UnaryFunction<hugeint_t, TR, OP>;
		break;
	default:
		throw NotImplementedException("Unimplemented type for GetScalarIntegerUnaryFunctionFixedReturn");
	}
	return function;
}

//===--------------------------------------------------------------------===//
// nextafter
//===--------------------------------------------------------------------===//
struct NextAfterOperator {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA base, TB exponent) {
		throw NotImplementedException("Unimplemented type for NextAfter Function");
	}

	template <class TA, class TB, class TR>
	static inline double Operation(double input, double approximate_to) {
		return nextafter(input, approximate_to);
	}
	template <class TA, class TB, class TR>
	static inline float Operation(float input, float approximate_to) {
		return nextafterf(input, approximate_to);
	}
};

ScalarFunctionSet NextAfterFun::GetFunctions() {
	ScalarFunctionSet next_after_fun;
	next_after_fun.AddFunction(
	    ScalarFunction({LogicalType::DOUBLE, LogicalType::DOUBLE}, LogicalType::DOUBLE,
	                   ScalarFunction::BinaryFunction<double, double, double, NextAfterOperator>));
	next_after_fun.AddFunction(ScalarFunction({LogicalType::FLOAT, LogicalType::FLOAT}, LogicalType::FLOAT,
	                                          ScalarFunction::BinaryFunction<float, float, float, NextAfterOperator>));
	return next_after_fun;
}

//===--------------------------------------------------------------------===//
// abs
//===--------------------------------------------------------------------===//
static unique_ptr<BaseStatistics> PropagateAbsStats(ClientContext &context, FunctionStatisticsInput &input) {
	auto &child_stats = input.child_stats;
	auto &expr = input.expr;
	D_ASSERT(child_stats.size() == 1);
	// can only propagate stats if the children have stats
	auto &lstats = child_stats[0];
	Value new_min, new_max;
	bool potential_overflow = true;
	if (NumericStats::HasMinMax(lstats)) {
		switch (expr.return_type.InternalType()) {
		case PhysicalType::INT8:
			potential_overflow = NumericStats::Min(lstats).GetValue<int8_t>() == NumericLimits<int8_t>::Minimum();
			break;
		case PhysicalType::INT16:
			potential_overflow = NumericStats::Min(lstats).GetValue<int16_t>() == NumericLimits<int16_t>::Minimum();
			break;
		case PhysicalType::INT32:
			potential_overflow = NumericStats::Min(lstats).GetValue<int32_t>() == NumericLimits<int32_t>::Minimum();
			break;
		case PhysicalType::INT64:
			potential_overflow = NumericStats::Min(lstats).GetValue<int64_t>() == NumericLimits<int64_t>::Minimum();
			break;
		default:
			return nullptr;
		}
	}
	if (potential_overflow) {
		new_min = Value(expr.return_type);
		new_max = Value(expr.return_type);
	} else {
		// no potential overflow

		// compute stats
		auto current_min = NumericStats::Min(lstats).GetValue<int64_t>();
		auto current_max = NumericStats::Max(lstats).GetValue<int64_t>();

		int64_t min_val, max_val;

		if (current_min < 0 && current_max < 0) {
			// if both min and max are below zero, then min=abs(cur_max) and max=abs(cur_min)
			min_val = AbsValue(current_max);
			max_val = AbsValue(current_min);
		} else if (current_min < 0) {
			D_ASSERT(current_max >= 0);
			// if min is below zero and max is above 0, then min=0 and max=max(cur_max, abs(cur_min))
			min_val = 0;
			max_val = MaxValue(AbsValue(current_min), current_max);
		} else {
			// if both current_min and current_max are > 0, then the abs is a no-op and can be removed entirely
			*input.expr_ptr = std::move(input.expr.children[0]);
			return child_stats[0].ToUnique();
		}
		new_min = Value::Numeric(expr.return_type, min_val);
		new_max = Value::Numeric(expr.return_type, max_val);
		expr.function.function = ScalarFunction::GetScalarUnaryFunction<AbsOperator>(expr.return_type);
	}
	auto stats = NumericStats::CreateEmpty(expr.return_type);
	NumericStats::SetMin(stats, new_min);
	NumericStats::SetMax(stats, new_max);
	stats.CopyValidity(lstats);
	return stats.ToUnique();
}

template <class OP>
unique_ptr<FunctionData> DecimalUnaryOpBind(ClientContext &context, ScalarFunction &bound_function,
                                            vector<unique_ptr<Expression>> &arguments) {
	auto decimal_type = arguments[0]->return_type;
	switch (decimal_type.InternalType()) {
	case PhysicalType::INT16:
		bound_function.function = ScalarFunction::GetScalarUnaryFunction<OP>(LogicalTypeId::SMALLINT);
		break;
	case PhysicalType::INT32:
		bound_function.function = ScalarFunction::GetScalarUnaryFunction<OP>(LogicalTypeId::INTEGER);
		break;
	case PhysicalType::INT64:
		bound_function.function = ScalarFunction::GetScalarUnaryFunction<OP>(LogicalTypeId::BIGINT);
		break;
	default:
		bound_function.function = ScalarFunction::GetScalarUnaryFunction<OP>(LogicalTypeId::HUGEINT);
		break;
	}
	bound_function.arguments[0] = decimal_type;
	bound_function.return_type = decimal_type;
	return nullptr;
}

ScalarFunctionSet AbsOperatorFun::GetFunctions() {
	ScalarFunctionSet abs;
	for (auto &type : LogicalType::Numeric()) {
		switch (type.id()) {
		case LogicalTypeId::DECIMAL:
			abs.AddFunction(ScalarFunction({type}, type, nullptr, DecimalUnaryOpBind<AbsOperator>));
			break;
		case LogicalTypeId::TINYINT:
		case LogicalTypeId::SMALLINT:
		case LogicalTypeId::INTEGER:
		case LogicalTypeId::BIGINT: {
			ScalarFunction func({type}, type, ScalarFunction::GetScalarUnaryFunction<TryAbsOperator>(type));
			func.statistics = PropagateAbsStats;
			abs.AddFunction(func);
			break;
		}
		case LogicalTypeId::UTINYINT:
		case LogicalTypeId::USMALLINT:
		case LogicalTypeId::UINTEGER:
		case LogicalTypeId::UBIGINT:
			abs.AddFunction(ScalarFunction({type}, type, ScalarFunction::NopFunction));
			break;
		default:
			abs.AddFunction(ScalarFunction({type}, type, ScalarFunction::GetScalarUnaryFunction<AbsOperator>(type)));
			break;
		}
	}
	return abs;
}

//===--------------------------------------------------------------------===//
// bit_count
//===--------------------------------------------------------------------===//
struct BitCntOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		using TU = typename std::make_unsigned<TA>::type;
		TR count = 0;
		for (auto value = TU(input); value; ++count) {
			value &= (value - 1);
		}
		return count;
	}
};

struct HugeIntBitCntOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		using TU = typename std::make_unsigned<int64_t>::type;
		TR count = 0;

		for (auto value = TU(input.upper); value; ++count) {
			value &= (value - 1);
		}
		for (auto value = TU(input.lower); value; ++count) {
			value &= (value - 1);
		}
		return count;
	}
};

struct BitStringBitCntOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		TR count = Bit::BitCount(input);
		return count;
	}
};

ScalarFunctionSet BitCountFun::GetFunctions() {
	ScalarFunctionSet functions;
	functions.AddFunction(ScalarFunction({LogicalType::TINYINT}, LogicalType::TINYINT,
	                                     ScalarFunction::UnaryFunction<int8_t, int8_t, BitCntOperator>));
	functions.AddFunction(ScalarFunction({LogicalType::SMALLINT}, LogicalType::TINYINT,
	                                     ScalarFunction::UnaryFunction<int16_t, int8_t, BitCntOperator>));
	functions.AddFunction(ScalarFunction({LogicalType::INTEGER}, LogicalType::TINYINT,
	                                     ScalarFunction::UnaryFunction<int32_t, int8_t, BitCntOperator>));
	functions.AddFunction(ScalarFunction({LogicalType::BIGINT}, LogicalType::TINYINT,
	                                     ScalarFunction::UnaryFunction<int64_t, int8_t, BitCntOperator>));
	functions.AddFunction(ScalarFunction({LogicalType::HUGEINT}, LogicalType::TINYINT,
	                                     ScalarFunction::UnaryFunction<hugeint_t, int8_t, HugeIntBitCntOperator>));
	functions.AddFunction(ScalarFunction({LogicalType::BIT}, LogicalType::BIGINT,
	                                     ScalarFunction::UnaryFunction<string_t, idx_t, BitStringBitCntOperator>));
	return functions;
}

//===--------------------------------------------------------------------===//
// sign
//===--------------------------------------------------------------------===//
struct SignOperator {
	template <class TA, class TR>
	static TR Operation(TA input) {
		if (input == TA(0)) {
			return 0;
		} else if (input > TA(0)) {
			return 1;
		} else {
			return -1;
		}
	}
};

template <>
int8_t SignOperator::Operation(float input) {
	if (input == 0 || Value::IsNan(input)) {
		return 0;
	} else if (input > 0) {
		return 1;
	} else {
		return -1;
	}
}

template <>
int8_t SignOperator::Operation(double input) {
	if (input == 0 || Value::IsNan(input)) {
		return 0;
	} else if (input > 0) {
		return 1;
	} else {
		return -1;
	}
}

ScalarFunctionSet SignFun::GetFunctions() {
	ScalarFunctionSet sign;
	for (auto &type : LogicalType::Numeric()) {
		if (type.id() == LogicalTypeId::DECIMAL) {
			continue;
		} else {
			sign.AddFunction(
			    ScalarFunction({type}, LogicalType::TINYINT,
			                   ScalarFunction::GetScalarUnaryFunctionFixedReturn<int8_t, SignOperator>(type)));
		}
	}
	return sign;
}

//===--------------------------------------------------------------------===//
// ceil
//===--------------------------------------------------------------------===//
struct CeilOperator {
	template <class TA, class TR>
	static inline TR Operation(TA left) {
		return std::ceil(left);
	}
};

template <class T, class POWERS_OF_TEN, class OP>
static void GenericRoundFunctionDecimal(DataChunk &input, ExpressionState &state, Vector &result) {
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	OP::template Operation<T, POWERS_OF_TEN>(input, DecimalType::GetScale(func_expr.children[0]->return_type), result);
}

template <class OP>
unique_ptr<FunctionData> BindGenericRoundFunctionDecimal(ClientContext &context, ScalarFunction &bound_function,
                                                         vector<unique_ptr<Expression>> &arguments) {
	// ceil essentially removes the scale
	auto &decimal_type = arguments[0]->return_type;
	auto scale = DecimalType::GetScale(decimal_type);
	auto width = DecimalType::GetWidth(decimal_type);
	if (scale == 0) {
		bound_function.function = ScalarFunction::NopFunction;
	} else {
		switch (decimal_type.InternalType()) {
		case PhysicalType::INT16:
			bound_function.function = GenericRoundFunctionDecimal<int16_t, NumericHelper, OP>;
			break;
		case PhysicalType::INT32:
			bound_function.function = GenericRoundFunctionDecimal<int32_t, NumericHelper, OP>;
			break;
		case PhysicalType::INT64:
			bound_function.function = GenericRoundFunctionDecimal<int64_t, NumericHelper, OP>;
			break;
		default:
			bound_function.function = GenericRoundFunctionDecimal<hugeint_t, Hugeint, OP>;
			break;
		}
	}
	bound_function.arguments[0] = decimal_type;
	bound_function.return_type = LogicalType::DECIMAL(width, 0);
	return nullptr;
}

struct CeilDecimalOperator {
	template <class T, class POWERS_OF_TEN_CLASS>
	static void Operation(DataChunk &input, uint8_t scale, Vector &result) {
		T power_of_ten = POWERS_OF_TEN_CLASS::POWERS_OF_TEN[scale];
		UnaryExecutor::Execute<T, T>(input.data[0], result, input.size(), [&](T input) {
			if (input < 0) {
				// below 0 we floor the number (e.g. -10.5 -> -10)
				return input / power_of_ten;
			} else {
				// above 0 we ceil the number
				return ((input - 1) / power_of_ten) + 1;
			}
		});
	}
};

ScalarFunctionSet CeilFun::GetFunctions() {
	ScalarFunctionSet ceil;
	for (auto &type : LogicalType::Numeric()) {
		scalar_function_t func = nullptr;
		bind_scalar_function_t bind_func = nullptr;
		if (type.IsIntegral()) {
			// no ceil for integral numbers
			continue;
		}
		switch (type.id()) {
		case LogicalTypeId::FLOAT:
			func = ScalarFunction::UnaryFunction<float, float, CeilOperator>;
			break;
		case LogicalTypeId::DOUBLE:
			func = ScalarFunction::UnaryFunction<double, double, CeilOperator>;
			break;
		case LogicalTypeId::DECIMAL:
			bind_func = BindGenericRoundFunctionDecimal<CeilDecimalOperator>;
			break;
		default:
			throw InternalException("Unimplemented numeric type for function \"ceil\"");
		}
		ceil.AddFunction(ScalarFunction({type}, type, func, bind_func));
	}
	return ceil;
}

//===--------------------------------------------------------------------===//
// floor
//===--------------------------------------------------------------------===//
struct FloorOperator {
	template <class TA, class TR>
	static inline TR Operation(TA left) {
		return std::floor(left);
	}
};

struct FloorDecimalOperator {
	template <class T, class POWERS_OF_TEN_CLASS>
	static void Operation(DataChunk &input, uint8_t scale, Vector &result) {
		T power_of_ten = POWERS_OF_TEN_CLASS::POWERS_OF_TEN[scale];
		UnaryExecutor::Execute<T, T>(input.data[0], result, input.size(), [&](T input) {
			if (input < 0) {
				// below 0 we ceil the number (e.g. -10.5 -> -11)
				return ((input + 1) / power_of_ten) - 1;
			} else {
				// above 0 we floor the number
				return input / power_of_ten;
			}
		});
	}
};

ScalarFunctionSet FloorFun::GetFunctions() {
	ScalarFunctionSet floor;
	for (auto &type : LogicalType::Numeric()) {
		scalar_function_t func = nullptr;
		bind_scalar_function_t bind_func = nullptr;
		if (type.IsIntegral()) {
			// no floor for integral numbers
			continue;
		}
		switch (type.id()) {
		case LogicalTypeId::FLOAT:
			func = ScalarFunction::UnaryFunction<float, float, FloorOperator>;
			break;
		case LogicalTypeId::DOUBLE:
			func = ScalarFunction::UnaryFunction<double, double, FloorOperator>;
			break;
		case LogicalTypeId::DECIMAL:
			bind_func = BindGenericRoundFunctionDecimal<FloorDecimalOperator>;
			break;
		default:
			throw InternalException("Unimplemented numeric type for function \"floor\"");
		}
		floor.AddFunction(ScalarFunction({type}, type, func, bind_func));
	}
	return floor;
}

//===--------------------------------------------------------------------===//
// trunc
//===--------------------------------------------------------------------===//
struct TruncOperator {
	// Integer truncation is a NOP
	template <class TA, class TR>
	static inline TR Operation(TA left) {
		return std::trunc(left);
	}
};

struct TruncDecimalOperator {
	template <class T, class POWERS_OF_TEN_CLASS>
	static void Operation(DataChunk &input, uint8_t scale, Vector &result) {
		T power_of_ten = POWERS_OF_TEN_CLASS::POWERS_OF_TEN[scale];
		UnaryExecutor::Execute<T, T>(input.data[0], result, input.size(), [&](T input) {
			//	Always floor
			return (input / power_of_ten);
		});
	}
};

ScalarFunctionSet TruncFun::GetFunctions() {
	ScalarFunctionSet trunc;
	for (auto &type : LogicalType::Numeric()) {
		scalar_function_t func = nullptr;
		bind_scalar_function_t bind_func = nullptr;
		//	Truncation of integers gets generated by some tools (e.g., Tableau/JDBC:Postgres)
		switch (type.id()) {
		case LogicalTypeId::FLOAT:
			func = ScalarFunction::UnaryFunction<float, float, TruncOperator>;
			break;
		case LogicalTypeId::DOUBLE:
			func = ScalarFunction::UnaryFunction<double, double, TruncOperator>;
			break;
		case LogicalTypeId::DECIMAL:
			bind_func = BindGenericRoundFunctionDecimal<TruncDecimalOperator>;
			break;
		case LogicalTypeId::TINYINT:
		case LogicalTypeId::SMALLINT:
		case LogicalTypeId::INTEGER:
		case LogicalTypeId::BIGINT:
		case LogicalTypeId::HUGEINT:
		case LogicalTypeId::UTINYINT:
		case LogicalTypeId::USMALLINT:
		case LogicalTypeId::UINTEGER:
		case LogicalTypeId::UBIGINT:
			func = ScalarFunction::NopFunction;
			break;
		default:
			throw InternalException("Unimplemented numeric type for function \"trunc\"");
		}
		trunc.AddFunction(ScalarFunction({type}, type, func, bind_func));
	}
	return trunc;
}

//===--------------------------------------------------------------------===//
// round
//===--------------------------------------------------------------------===//
struct RoundOperatorPrecision {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA input, TB precision) {
		double rounded_value;
		if (precision < 0) {
			double modifier = std::pow(10, -precision);
			rounded_value = (std::round(input / modifier)) * modifier;
			if (std::isinf(rounded_value) || std::isnan(rounded_value)) {
				return 0;
			}
		} else {
			double modifier = std::pow(10, precision);
			rounded_value = (std::round(input * modifier)) / modifier;
			if (std::isinf(rounded_value) || std::isnan(rounded_value)) {
				return input;
			}
		}
		return rounded_value;
	}
};

struct RoundOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		double rounded_value = round(input);
		if (std::isinf(rounded_value) || std::isnan(rounded_value)) {
			return input;
		}
		return rounded_value;
	}
};

struct RoundDecimalOperator {
	template <class T, class POWERS_OF_TEN_CLASS>
	static void Operation(DataChunk &input, uint8_t scale, Vector &result) {
		T power_of_ten = POWERS_OF_TEN_CLASS::POWERS_OF_TEN[scale];
		T addition = power_of_ten / 2;
		// regular round rounds towards the nearest number
		// in case of a tie we round away from zero
		// i.e. -10.5 -> -11, 10.5 -> 11
		// we implement this by adding (positive) or subtracting (negative) 0.5
		// and then flooring the number
		// e.g. 10.5 + 0.5 = 11, floor(11) = 11
		//      10.4 + 0.5 = 10.9, floor(10.9) = 10
		UnaryExecutor::Execute<T, T>(input.data[0], result, input.size(), [&](T input) {
			if (input < 0) {
				input -= addition;
			} else {
				input += addition;
			}
			return input / power_of_ten;
		});
	}
};

struct RoundPrecisionFunctionData : public FunctionData {
	explicit RoundPrecisionFunctionData(int32_t target_scale) : target_scale(target_scale) {
	}

	int32_t target_scale;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<RoundPrecisionFunctionData>(target_scale);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = (const RoundPrecisionFunctionData &)other_p;
		return target_scale == other.target_scale;
	}
};

template <class T, class POWERS_OF_TEN_CLASS>
static void DecimalRoundNegativePrecisionFunction(DataChunk &input, ExpressionState &state, Vector &result) {
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &info = (RoundPrecisionFunctionData &)*func_expr.bind_info;
	auto source_scale = DecimalType::GetScale(func_expr.children[0]->return_type);
	auto width = DecimalType::GetWidth(func_expr.children[0]->return_type);
	if (-info.target_scale >= width) {
		// scale too big for width
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
		result.SetValue(0, Value::INTEGER(0));
		return;
	}
	T divide_power_of_ten = POWERS_OF_TEN_CLASS::POWERS_OF_TEN[-info.target_scale + source_scale];
	T multiply_power_of_ten = POWERS_OF_TEN_CLASS::POWERS_OF_TEN[-info.target_scale];
	T addition = divide_power_of_ten / 2;

	UnaryExecutor::Execute<T, T>(input.data[0], result, input.size(), [&](T input) {
		if (input < 0) {
			input -= addition;
		} else {
			input += addition;
		}
		return input / divide_power_of_ten * multiply_power_of_ten;
	});
}

template <class T, class POWERS_OF_TEN_CLASS>
static void DecimalRoundPositivePrecisionFunction(DataChunk &input, ExpressionState &state, Vector &result) {
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &info = (RoundPrecisionFunctionData &)*func_expr.bind_info;
	auto source_scale = DecimalType::GetScale(func_expr.children[0]->return_type);
	T power_of_ten = POWERS_OF_TEN_CLASS::POWERS_OF_TEN[source_scale - info.target_scale];
	T addition = power_of_ten / 2;
	UnaryExecutor::Execute<T, T>(input.data[0], result, input.size(), [&](T input) {
		if (input < 0) {
			input -= addition;
		} else {
			input += addition;
		}
		return input / power_of_ten;
	});
}

unique_ptr<FunctionData> BindDecimalRoundPrecision(ClientContext &context, ScalarFunction &bound_function,
                                                   vector<unique_ptr<Expression>> &arguments) {
	auto &decimal_type = arguments[0]->return_type;
	if (arguments[1]->HasParameter()) {
		throw ParameterNotResolvedException();
	}
	if (!arguments[1]->IsFoldable()) {
		throw NotImplementedException("ROUND(DECIMAL, INTEGER) with non-constant precision is not supported");
	}
	Value val = ExpressionExecutor::EvaluateScalar(context, *arguments[1]).DefaultCastAs(LogicalType::INTEGER);
	if (val.IsNull()) {
		throw NotImplementedException("ROUND(DECIMAL, INTEGER) with non-constant precision is not supported");
	}
	// our new precision becomes the round value
	// e.g. ROUND(DECIMAL(18,3), 1) -> DECIMAL(18,1)
	// but ONLY if the round value is positive
	// if it is negative the scale becomes zero
	// i.e. ROUND(DECIMAL(18,3), -1) -> DECIMAL(18,0)
	int32_t round_value = IntegerValue::Get(val);
	uint8_t target_scale;
	auto width = DecimalType::GetWidth(decimal_type);
	auto scale = DecimalType::GetScale(decimal_type);
	if (round_value < 0) {
		target_scale = 0;
		switch (decimal_type.InternalType()) {
		case PhysicalType::INT16:
			bound_function.function = DecimalRoundNegativePrecisionFunction<int16_t, NumericHelper>;
			break;
		case PhysicalType::INT32:
			bound_function.function = DecimalRoundNegativePrecisionFunction<int32_t, NumericHelper>;
			break;
		case PhysicalType::INT64:
			bound_function.function = DecimalRoundNegativePrecisionFunction<int64_t, NumericHelper>;
			break;
		default:
			bound_function.function = DecimalRoundNegativePrecisionFunction<hugeint_t, Hugeint>;
			break;
		}
	} else {
		if (round_value >= (int32_t)scale) {
			// if round_value is bigger than or equal to scale we do nothing
			bound_function.function = ScalarFunction::NopFunction;
			target_scale = scale;
		} else {
			target_scale = round_value;
			switch (decimal_type.InternalType()) {
			case PhysicalType::INT16:
				bound_function.function = DecimalRoundPositivePrecisionFunction<int16_t, NumericHelper>;
				break;
			case PhysicalType::INT32:
				bound_function.function = DecimalRoundPositivePrecisionFunction<int32_t, NumericHelper>;
				break;
			case PhysicalType::INT64:
				bound_function.function = DecimalRoundPositivePrecisionFunction<int64_t, NumericHelper>;
				break;
			default:
				bound_function.function = DecimalRoundPositivePrecisionFunction<hugeint_t, Hugeint>;
				break;
			}
		}
	}
	bound_function.arguments[0] = decimal_type;
	bound_function.return_type = LogicalType::DECIMAL(width, target_scale);
	return make_uniq<RoundPrecisionFunctionData>(round_value);
}

ScalarFunctionSet RoundFun::GetFunctions() {
	ScalarFunctionSet round;
	for (auto &type : LogicalType::Numeric()) {
		scalar_function_t round_prec_func = nullptr;
		scalar_function_t round_func = nullptr;
		bind_scalar_function_t bind_func = nullptr;
		bind_scalar_function_t bind_prec_func = nullptr;
		if (type.IsIntegral()) {
			// no round for integral numbers
			continue;
		}
		switch (type.id()) {
		case LogicalTypeId::FLOAT:
			round_func = ScalarFunction::UnaryFunction<float, float, RoundOperator>;
			round_prec_func = ScalarFunction::BinaryFunction<float, int32_t, float, RoundOperatorPrecision>;
			break;
		case LogicalTypeId::DOUBLE:
			round_func = ScalarFunction::UnaryFunction<double, double, RoundOperator>;
			round_prec_func = ScalarFunction::BinaryFunction<double, int32_t, double, RoundOperatorPrecision>;
			break;
		case LogicalTypeId::DECIMAL:
			bind_func = BindGenericRoundFunctionDecimal<RoundDecimalOperator>;
			bind_prec_func = BindDecimalRoundPrecision;
			break;
		default:
			throw InternalException("Unimplemented numeric type for function \"floor\"");
		}
		round.AddFunction(ScalarFunction({type}, type, round_func, bind_func));
		round.AddFunction(ScalarFunction({type, LogicalType::INTEGER}, type, round_prec_func, bind_prec_func));
	}
	return round;
}

//===--------------------------------------------------------------------===//
// exp
//===--------------------------------------------------------------------===//
struct ExpOperator {
	template <class TA, class TR>
	static inline TR Operation(TA left) {
		return std::exp(left);
	}
};

ScalarFunction ExpFun::GetFunction() {
	return ScalarFunction({LogicalType::DOUBLE}, LogicalType::DOUBLE,
	                      ScalarFunction::UnaryFunction<double, double, ExpOperator>);
}

//===--------------------------------------------------------------------===//
// pow
//===--------------------------------------------------------------------===//
struct PowOperator {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA base, TB exponent) {
		return std::pow(base, exponent);
	}
};

ScalarFunction PowOperatorFun::GetFunction() {
	return ScalarFunction({LogicalType::DOUBLE, LogicalType::DOUBLE}, LogicalType::DOUBLE,
	                      ScalarFunction::BinaryFunction<double, double, double, PowOperator>);
}

//===--------------------------------------------------------------------===//
// sqrt
//===--------------------------------------------------------------------===//
struct SqrtOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		if (input < 0) {
			throw OutOfRangeException("cannot take square root of a negative number");
		}
		return std::sqrt(input);
	}
};

ScalarFunction SqrtFun::GetFunction() {
	return ScalarFunction({LogicalType::DOUBLE}, LogicalType::DOUBLE,
	                      ScalarFunction::UnaryFunction<double, double, SqrtOperator>);
}

//===--------------------------------------------------------------------===//
// cbrt
//===--------------------------------------------------------------------===//
struct CbRtOperator {
	template <class TA, class TR>
	static inline TR Operation(TA left) {
		return std::cbrt(left);
	}
};

ScalarFunction CbrtFun::GetFunction() {
	return ScalarFunction({LogicalType::DOUBLE}, LogicalType::DOUBLE,
	                      ScalarFunction::UnaryFunction<double, double, CbRtOperator>);
}

//===--------------------------------------------------------------------===//
// ln
//===--------------------------------------------------------------------===//

struct LnOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		if (input < 0) {
			throw OutOfRangeException("cannot take logarithm of a negative number");
		}
		if (input == 0) {
			throw OutOfRangeException("cannot take logarithm of zero");
		}
		return std::log(input);
	}
};

ScalarFunction LnFun::GetFunction() {
	return ScalarFunction({LogicalType::DOUBLE}, LogicalType::DOUBLE,
	                      ScalarFunction::UnaryFunction<double, double, LnOperator>);
}

//===--------------------------------------------------------------------===//
// log
//===--------------------------------------------------------------------===//
struct Log10Operator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		if (input < 0) {
			throw OutOfRangeException("cannot take logarithm of a negative number");
		}
		if (input == 0) {
			throw OutOfRangeException("cannot take logarithm of zero");
		}
		return std::log10(input);
	}
};

ScalarFunction Log10Fun::GetFunction() {
	return ScalarFunction({LogicalType::DOUBLE}, LogicalType::DOUBLE,
	                      ScalarFunction::UnaryFunction<double, double, Log10Operator>);
}

//===--------------------------------------------------------------------===//
// log2
//===--------------------------------------------------------------------===//
struct Log2Operator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		if (input < 0) {
			throw OutOfRangeException("cannot take logarithm of a negative number");
		}
		if (input == 0) {
			throw OutOfRangeException("cannot take logarithm of zero");
		}
		return std::log2(input);
	}
};

ScalarFunction Log2Fun::GetFunction() {
	return ScalarFunction({LogicalType::DOUBLE}, LogicalType::DOUBLE,
	                      ScalarFunction::UnaryFunction<double, double, Log2Operator>);
}

//===--------------------------------------------------------------------===//
// pi
//===--------------------------------------------------------------------===//
static void PiFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(args.ColumnCount() == 0);
	Value pi_value = Value::DOUBLE(PI);
	result.Reference(pi_value);
}

ScalarFunction PiFun::GetFunction() {
	return ScalarFunction({}, LogicalType::DOUBLE, PiFunction);
}

//===--------------------------------------------------------------------===//
// degrees
//===--------------------------------------------------------------------===//
struct DegreesOperator {
	template <class TA, class TR>
	static inline TR Operation(TA left) {
		return left * (180 / PI);
	}
};

ScalarFunction DegreesFun::GetFunction() {
	return ScalarFunction({LogicalType::DOUBLE}, LogicalType::DOUBLE,
	                      ScalarFunction::UnaryFunction<double, double, DegreesOperator>);
}

//===--------------------------------------------------------------------===//
// radians
//===--------------------------------------------------------------------===//
struct RadiansOperator {
	template <class TA, class TR>
	static inline TR Operation(TA left) {
		return left * (PI / 180);
	}
};

ScalarFunction RadiansFun::GetFunction() {
	return ScalarFunction({LogicalType::DOUBLE}, LogicalType::DOUBLE,
	                      ScalarFunction::UnaryFunction<double, double, RadiansOperator>);
}

//===--------------------------------------------------------------------===//
// isnan
//===--------------------------------------------------------------------===//
struct IsNanOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		return Value::IsNan(input);
	}
};

ScalarFunctionSet IsNanFun::GetFunctions() {
	ScalarFunctionSet funcs;
	funcs.AddFunction(ScalarFunction({LogicalType::FLOAT}, LogicalType::BOOLEAN,
	                                 ScalarFunction::UnaryFunction<float, bool, IsNanOperator>));
	funcs.AddFunction(ScalarFunction({LogicalType::DOUBLE}, LogicalType::BOOLEAN,
	                                 ScalarFunction::UnaryFunction<double, bool, IsNanOperator>));
	return funcs;
}

//===--------------------------------------------------------------------===//
// signbit
//===--------------------------------------------------------------------===//
struct SignBitOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		return std::signbit(input);
	}
};

ScalarFunctionSet SignBitFun::GetFunctions() {
	ScalarFunctionSet funcs;
	funcs.AddFunction(ScalarFunction({LogicalType::FLOAT}, LogicalType::BOOLEAN,
	                                 ScalarFunction::UnaryFunction<float, bool, SignBitOperator>));
	funcs.AddFunction(ScalarFunction({LogicalType::DOUBLE}, LogicalType::BOOLEAN,
	                                 ScalarFunction::UnaryFunction<double, bool, SignBitOperator>));
	return funcs;
}

//===--------------------------------------------------------------------===//
// isinf
//===--------------------------------------------------------------------===//
struct IsInfiniteOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		return !Value::IsNan(input) && !Value::IsFinite(input);
	}
};

template <>
bool IsInfiniteOperator::Operation(date_t input) {
	return !Value::IsFinite(input);
}

template <>
bool IsInfiniteOperator::Operation(timestamp_t input) {
	return !Value::IsFinite(input);
}

ScalarFunctionSet IsInfiniteFun::GetFunctions() {
	ScalarFunctionSet funcs("isinf");
	funcs.AddFunction(ScalarFunction({LogicalType::FLOAT}, LogicalType::BOOLEAN,
	                                 ScalarFunction::UnaryFunction<float, bool, IsInfiniteOperator>));
	funcs.AddFunction(ScalarFunction({LogicalType::DOUBLE}, LogicalType::BOOLEAN,
	                                 ScalarFunction::UnaryFunction<double, bool, IsInfiniteOperator>));
	funcs.AddFunction(ScalarFunction({LogicalType::DATE}, LogicalType::BOOLEAN,
	                                 ScalarFunction::UnaryFunction<date_t, bool, IsInfiniteOperator>));
	funcs.AddFunction(ScalarFunction({LogicalType::TIMESTAMP}, LogicalType::BOOLEAN,
	                                 ScalarFunction::UnaryFunction<timestamp_t, bool, IsInfiniteOperator>));
	funcs.AddFunction(ScalarFunction({LogicalType::TIMESTAMP_TZ}, LogicalType::BOOLEAN,
	                                 ScalarFunction::UnaryFunction<timestamp_t, bool, IsInfiniteOperator>));
	return funcs;
}

//===--------------------------------------------------------------------===//
// isfinite
//===--------------------------------------------------------------------===//
struct IsFiniteOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		return Value::IsFinite(input);
	}
};

ScalarFunctionSet IsFiniteFun::GetFunctions() {
	ScalarFunctionSet funcs;
	funcs.AddFunction(ScalarFunction({LogicalType::FLOAT}, LogicalType::BOOLEAN,
	                                 ScalarFunction::UnaryFunction<float, bool, IsFiniteOperator>));
	funcs.AddFunction(ScalarFunction({LogicalType::DOUBLE}, LogicalType::BOOLEAN,
	                                 ScalarFunction::UnaryFunction<double, bool, IsFiniteOperator>));
	funcs.AddFunction(ScalarFunction({LogicalType::DATE}, LogicalType::BOOLEAN,
	                                 ScalarFunction::UnaryFunction<date_t, bool, IsFiniteOperator>));
	funcs.AddFunction(ScalarFunction({LogicalType::TIMESTAMP}, LogicalType::BOOLEAN,
	                                 ScalarFunction::UnaryFunction<timestamp_t, bool, IsFiniteOperator>));
	funcs.AddFunction(ScalarFunction({LogicalType::TIMESTAMP_TZ}, LogicalType::BOOLEAN,
	                                 ScalarFunction::UnaryFunction<timestamp_t, bool, IsFiniteOperator>));
	return funcs;
}

//===--------------------------------------------------------------------===//
// sin
//===--------------------------------------------------------------------===//
template <class OP>
struct NoInfiniteDoubleWrapper {
	template <class INPUT_TYPE, class RESULT_TYPE>
	static RESULT_TYPE Operation(INPUT_TYPE input) {
		if (DUCKDB_UNLIKELY(!Value::IsFinite(input))) {
			if (Value::IsNan(input)) {
				return input;
			}
			throw OutOfRangeException("input value %lf is out of range for numeric function", input);
		}
		return OP::template Operation<INPUT_TYPE, RESULT_TYPE>(input);
	}
};

struct SinOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		return std::sin(input);
	}
};

ScalarFunction SinFun::GetFunction() {
	return ScalarFunction({LogicalType::DOUBLE}, LogicalType::DOUBLE,
	                      ScalarFunction::UnaryFunction<double, double, NoInfiniteDoubleWrapper<SinOperator>>);
}

//===--------------------------------------------------------------------===//
// cos
//===--------------------------------------------------------------------===//
struct CosOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		return (double)std::cos(input);
	}
};

ScalarFunction CosFun::GetFunction() {
	return ScalarFunction({LogicalType::DOUBLE}, LogicalType::DOUBLE,
	                      ScalarFunction::UnaryFunction<double, double, NoInfiniteDoubleWrapper<CosOperator>>);
}

//===--------------------------------------------------------------------===//
// tan
//===--------------------------------------------------------------------===//
struct TanOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		return (double)std::tan(input);
	}
};

ScalarFunction TanFun::GetFunction() {
	return ScalarFunction({LogicalType::DOUBLE}, LogicalType::DOUBLE,
	                      ScalarFunction::UnaryFunction<double, double, NoInfiniteDoubleWrapper<TanOperator>>);
}

//===--------------------------------------------------------------------===//
// asin
//===--------------------------------------------------------------------===//
struct ASinOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		if (input < -1 || input > 1) {
			throw Exception("ASIN is undefined outside [-1,1]");
		}
		return (double)std::asin(input);
	}
};

ScalarFunction AsinFun::GetFunction() {
	return ScalarFunction({LogicalType::DOUBLE}, LogicalType::DOUBLE,
	                      ScalarFunction::UnaryFunction<double, double, NoInfiniteDoubleWrapper<ASinOperator>>);
}

//===--------------------------------------------------------------------===//
// atan
//===--------------------------------------------------------------------===//
struct ATanOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		return (double)std::atan(input);
	}
};

ScalarFunction AtanFun::GetFunction() {
	return ScalarFunction({LogicalType::DOUBLE}, LogicalType::DOUBLE,
	                      ScalarFunction::UnaryFunction<double, double, ATanOperator>);
}

//===--------------------------------------------------------------------===//
// atan2
//===--------------------------------------------------------------------===//
struct ATan2 {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA left, TB right) {
		return (double)std::atan2(left, right);
	}
};

ScalarFunction Atan2Fun::GetFunction() {
	return ScalarFunction({LogicalType::DOUBLE, LogicalType::DOUBLE}, LogicalType::DOUBLE,
	                      ScalarFunction::BinaryFunction<double, double, double, ATan2>);
}

//===--------------------------------------------------------------------===//
// acos
//===--------------------------------------------------------------------===//
struct ACos {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		return (double)std::acos(input);
	}
};

ScalarFunction AcosFun::GetFunction() {
	return ScalarFunction({LogicalType::DOUBLE}, LogicalType::DOUBLE,
	                      ScalarFunction::UnaryFunction<double, double, NoInfiniteDoubleWrapper<ACos>>);
}

//===--------------------------------------------------------------------===//
// cot
//===--------------------------------------------------------------------===//
struct CotOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		return 1.0 / (double)std::tan(input);
	}
};

ScalarFunction CotFun::GetFunction() {
	return ScalarFunction({LogicalType::DOUBLE}, LogicalType::DOUBLE,
	                      ScalarFunction::UnaryFunction<double, double, NoInfiniteDoubleWrapper<CotOperator>>);
}

//===--------------------------------------------------------------------===//
// gamma
//===--------------------------------------------------------------------===//
struct GammaOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		if (input == 0) {
			throw OutOfRangeException("cannot take gamma of zero");
		}
		return std::tgamma(input);
	}
};

ScalarFunction GammaFun::GetFunction() {
	return ScalarFunction({LogicalType::DOUBLE}, LogicalType::DOUBLE,
	                      ScalarFunction::UnaryFunction<double, double, GammaOperator>);
}

//===--------------------------------------------------------------------===//
// gamma
//===--------------------------------------------------------------------===//
struct LogGammaOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		if (input == 0) {
			throw OutOfRangeException("cannot take log gamma of zero");
		}
		return std::lgamma(input);
	}
};

ScalarFunction LogGammaFun::GetFunction() {
	return ScalarFunction({LogicalType::DOUBLE}, LogicalType::DOUBLE,
	                      ScalarFunction::UnaryFunction<double, double, LogGammaOperator>);
}

//===--------------------------------------------------------------------===//
// factorial(), !
//===--------------------------------------------------------------------===//
struct FactorialOperator {
	template <class TA, class TR>
	static inline TR Operation(TA left) {
		TR ret = 1;
		for (TA i = 2; i <= left; i++) {
			ret *= i;
		}
		return ret;
	}
};

ScalarFunction FactorialOperatorFun::GetFunction() {
	return ScalarFunction({LogicalType::INTEGER}, LogicalType::HUGEINT,
	                      ScalarFunction::UnaryFunction<int32_t, hugeint_t, FactorialOperator>);
}

//===--------------------------------------------------------------------===//
// even
//===--------------------------------------------------------------------===//
struct EvenOperator {
	template <class TA, class TR>
	static inline TR Operation(TA left) {
		double value;
		if (left >= 0) {
			value = std::ceil(left);
		} else {
			value = std::ceil(-left);
			value = -value;
		}
		if (std::floor(value / 2) * 2 != value) {
			if (left >= 0) {
				return value += 1;
			}
			return value -= 1;
		}
		return value;
	}
};

ScalarFunction EvenFun::GetFunction() {
	return ScalarFunction({LogicalType::DOUBLE}, LogicalType::DOUBLE,
	                      ScalarFunction::UnaryFunction<double, double, EvenOperator>);
}

//===--------------------------------------------------------------------===//
// gcd
//===--------------------------------------------------------------------===//

// should be replaced with std::gcd in a newer C++ standard
template <class TA>
TA GreatestCommonDivisor(TA left, TA right) {
	TA a = left;
	TA b = right;

	// This protects the following modulo operations from a corner case,
	// where we would get a runtime error due to an integer overflow.
	if ((left == NumericLimits<TA>::Minimum() && right == -1) ||
	    (left == -1 && right == NumericLimits<TA>::Minimum())) {
		return 1;
	}

	while (true) {
		if (a == 0) {
			return TryAbsOperator::Operation<TA, TA>(b);
		}
		b %= a;

		if (b == 0) {
			return TryAbsOperator::Operation<TA, TA>(a);
		}
		a %= b;
	}
}

struct GreatestCommonDivisorOperator {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA left, TB right) {
		return GreatestCommonDivisor(left, right);
	}
};

ScalarFunctionSet GreatestCommonDivisorFun::GetFunctions() {
	ScalarFunctionSet funcs;
	funcs.AddFunction(
	    ScalarFunction({LogicalType::BIGINT, LogicalType::BIGINT}, LogicalType::BIGINT,
	                   ScalarFunction::BinaryFunction<int64_t, int64_t, int64_t, GreatestCommonDivisorOperator>));
	funcs.AddFunction(
	    ScalarFunction({LogicalType::HUGEINT, LogicalType::HUGEINT}, LogicalType::HUGEINT,
	                   ScalarFunction::BinaryFunction<hugeint_t, hugeint_t, hugeint_t, GreatestCommonDivisorOperator>));
	return funcs;
}

//===--------------------------------------------------------------------===//
// lcm
//===--------------------------------------------------------------------===//

// should be replaced with std::lcm in a newer C++ standard
struct LeastCommonMultipleOperator {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA left, TB right) {
		if (left == 0 || right == 0) {
			return 0;
		}
		TR result;
		if (!TryMultiplyOperator::Operation<TA, TB, TR>(left, right / GreatestCommonDivisor(left, right), result)) {
			throw OutOfRangeException("lcm value is out of range");
		}
		return TryAbsOperator::Operation<TR, TR>(result);
	}
};

ScalarFunctionSet LeastCommonMultipleFun::GetFunctions() {
	ScalarFunctionSet funcs;

	funcs.AddFunction(
	    ScalarFunction({LogicalType::BIGINT, LogicalType::BIGINT}, LogicalType::BIGINT,
	                   ScalarFunction::BinaryFunction<int64_t, int64_t, int64_t, LeastCommonMultipleOperator>));
	funcs.AddFunction(
	    ScalarFunction({LogicalType::HUGEINT, LogicalType::HUGEINT}, LogicalType::HUGEINT,
	                   ScalarFunction::BinaryFunction<hugeint_t, hugeint_t, hugeint_t, LeastCommonMultipleOperator>));
	return funcs;
}

} // namespace duckdb





namespace duckdb {

template <class OP>
static scalar_function_t GetScalarIntegerUnaryFunction(const LogicalType &type) {
	scalar_function_t function;
	switch (type.id()) {
	case LogicalTypeId::TINYINT:
		function = &ScalarFunction::UnaryFunction<int8_t, int8_t, OP>;
		break;
	case LogicalTypeId::SMALLINT:
		function = &ScalarFunction::UnaryFunction<int16_t, int16_t, OP>;
		break;
	case LogicalTypeId::INTEGER:
		function = &ScalarFunction::UnaryFunction<int32_t, int32_t, OP>;
		break;
	case LogicalTypeId::BIGINT:
		function = &ScalarFunction::UnaryFunction<int64_t, int64_t, OP>;
		break;
	case LogicalTypeId::UTINYINT:
		function = &ScalarFunction::UnaryFunction<uint8_t, uint8_t, OP>;
		break;
	case LogicalTypeId::USMALLINT:
		function = &ScalarFunction::UnaryFunction<uint16_t, uint16_t, OP>;
		break;
	case LogicalTypeId::UINTEGER:
		function = &ScalarFunction::UnaryFunction<uint32_t, uint32_t, OP>;
		break;
	case LogicalTypeId::UBIGINT:
		function = &ScalarFunction::UnaryFunction<uint64_t, uint64_t, OP>;
		break;
	case LogicalTypeId::HUGEINT:
		function = &ScalarFunction::UnaryFunction<hugeint_t, hugeint_t, OP>;
		break;
	default:
		throw NotImplementedException("Unimplemented type for GetScalarIntegerUnaryFunction");
	}
	return function;
}

template <class OP>
static scalar_function_t GetScalarIntegerBinaryFunction(const LogicalType &type) {
	scalar_function_t function;
	switch (type.id()) {
	case LogicalTypeId::TINYINT:
		function = &ScalarFunction::BinaryFunction<int8_t, int8_t, int8_t, OP>;
		break;
	case LogicalTypeId::SMALLINT:
		function = &ScalarFunction::BinaryFunction<int16_t, int16_t, int16_t, OP>;
		break;
	case LogicalTypeId::INTEGER:
		function = &ScalarFunction::BinaryFunction<int32_t, int32_t, int32_t, OP>;
		break;
	case LogicalTypeId::BIGINT:
		function = &ScalarFunction::BinaryFunction<int64_t, int64_t, int64_t, OP>;
		break;
	case LogicalTypeId::UTINYINT:
		function = &ScalarFunction::BinaryFunction<uint8_t, uint8_t, uint8_t, OP>;
		break;
	case LogicalTypeId::USMALLINT:
		function = &ScalarFunction::BinaryFunction<uint16_t, uint16_t, uint16_t, OP>;
		break;
	case LogicalTypeId::UINTEGER:
		function = &ScalarFunction::BinaryFunction<uint32_t, uint32_t, uint32_t, OP>;
		break;
	case LogicalTypeId::UBIGINT:
		function = &ScalarFunction::BinaryFunction<uint64_t, uint64_t, uint64_t, OP>;
		break;
	case LogicalTypeId::HUGEINT:
		function = &ScalarFunction::BinaryFunction<hugeint_t, hugeint_t, hugeint_t, OP>;
		break;
	default:
		throw NotImplementedException("Unimplemented type for GetScalarIntegerBinaryFunction");
	}
	return function;
}

//===--------------------------------------------------------------------===//
// & [bitwise_and]
//===--------------------------------------------------------------------===//
struct BitwiseANDOperator {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA left, TB right) {
		return left & right;
	}
};

static void BitwiseANDOperation(DataChunk &args, ExpressionState &state, Vector &result) {
	BinaryExecutor::Execute<string_t, string_t, string_t>(
	    args.data[0], args.data[1], result, args.size(), [&](string_t rhs, string_t lhs) {
		    string_t target = StringVector::EmptyString(result, rhs.GetSize());

		    Bit::BitwiseAnd(rhs, lhs, target);
		    return target;
	    });
}

ScalarFunctionSet BitwiseAndFun::GetFunctions() {
	ScalarFunctionSet functions;
	for (auto &type : LogicalType::Integral()) {
		functions.AddFunction(
		    ScalarFunction({type, type}, type, GetScalarIntegerBinaryFunction<BitwiseANDOperator>(type)));
	}
	functions.AddFunction(ScalarFunction({LogicalType::BIT, LogicalType::BIT}, LogicalType::BIT, BitwiseANDOperation));
	return functions;
}

//===--------------------------------------------------------------------===//
// | [bitwise_or]
//===--------------------------------------------------------------------===//
struct BitwiseOROperator {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA left, TB right) {
		return left | right;
	}
};

static void BitwiseOROperation(DataChunk &args, ExpressionState &state, Vector &result) {
	BinaryExecutor::Execute<string_t, string_t, string_t>(
	    args.data[0], args.data[1], result, args.size(), [&](string_t rhs, string_t lhs) {
		    string_t target = StringVector::EmptyString(result, rhs.GetSize());

		    Bit::BitwiseOr(rhs, lhs, target);
		    return target;
	    });
}

ScalarFunctionSet BitwiseOrFun::GetFunctions() {
	ScalarFunctionSet functions;
	for (auto &type : LogicalType::Integral()) {
		functions.AddFunction(
		    ScalarFunction({type, type}, type, GetScalarIntegerBinaryFunction<BitwiseOROperator>(type)));
	}
	functions.AddFunction(ScalarFunction({LogicalType::BIT, LogicalType::BIT}, LogicalType::BIT, BitwiseOROperation));
	return functions;
}

//===--------------------------------------------------------------------===//
// # [bitwise_xor]
//===--------------------------------------------------------------------===//
struct BitwiseXOROperator {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA left, TB right) {
		return left ^ right;
	}
};

static void BitwiseXOROperation(DataChunk &args, ExpressionState &state, Vector &result) {
	BinaryExecutor::Execute<string_t, string_t, string_t>(
	    args.data[0], args.data[1], result, args.size(), [&](string_t rhs, string_t lhs) {
		    string_t target = StringVector::EmptyString(result, rhs.GetSize());

		    Bit::BitwiseXor(rhs, lhs, target);
		    return target;
	    });
}

ScalarFunctionSet BitwiseXorFun::GetFunctions() {
	ScalarFunctionSet functions;
	for (auto &type : LogicalType::Integral()) {
		functions.AddFunction(
		    ScalarFunction({type, type}, type, GetScalarIntegerBinaryFunction<BitwiseXOROperator>(type)));
	}
	functions.AddFunction(ScalarFunction({LogicalType::BIT, LogicalType::BIT}, LogicalType::BIT, BitwiseXOROperation));
	return functions;
}

//===--------------------------------------------------------------------===//
// ~ [bitwise_not]
//===--------------------------------------------------------------------===//
struct BitwiseNotOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		return ~input;
	}
};

static void BitwiseNOTOperation(DataChunk &args, ExpressionState &state, Vector &result) {
	UnaryExecutor::Execute<string_t, string_t>(args.data[0], result, args.size(), [&](string_t input) {
		string_t target = StringVector::EmptyString(result, input.GetSize());

		Bit::BitwiseNot(input, target);
		return target;
	});
}

ScalarFunctionSet BitwiseNotFun::GetFunctions() {
	ScalarFunctionSet functions;
	for (auto &type : LogicalType::Integral()) {
		functions.AddFunction(ScalarFunction({type}, type, GetScalarIntegerUnaryFunction<BitwiseNotOperator>(type)));
	}
	functions.AddFunction(ScalarFunction({LogicalType::BIT}, LogicalType::BIT, BitwiseNOTOperation));
	return functions;
}

//===--------------------------------------------------------------------===//
// << [bitwise_left_shift]
//===--------------------------------------------------------------------===//

struct BitwiseShiftLeftOperator {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA input, TB shift) {
		TA max_shift = TA(sizeof(TA) * 8);
		if (input < 0) {
			throw OutOfRangeException("Cannot left-shift negative number %s", NumericHelper::ToString(input));
		}
		if (shift < 0) {
			throw OutOfRangeException("Cannot left-shift by negative number %s", NumericHelper::ToString(shift));
		}
		if (shift >= max_shift) {
			if (input == 0) {
				return 0;
			}
			throw OutOfRangeException("Left-shift value %s is out of range", NumericHelper::ToString(shift));
		}
		if (shift == 0) {
			return input;
		}
		TA max_value = (TA(1) << (max_shift - shift - 1));
		if (input >= max_value) {
			throw OutOfRangeException("Overflow in left shift (%s << %s)", NumericHelper::ToString(input),
			                          NumericHelper::ToString(shift));
		}
		return input << shift;
	}
};

static void BitwiseShiftLeftOperation(DataChunk &args, ExpressionState &state, Vector &result) {
	BinaryExecutor::Execute<string_t, int32_t, string_t>(
	    args.data[0], args.data[1], result, args.size(), [&](string_t input, int32_t shift) {
		    int32_t max_shift = Bit::BitLength(input);
		    if (shift == 0) {
			    return input;
		    }
		    if (shift < 0) {
			    throw OutOfRangeException("Cannot left-shift by negative number %s", NumericHelper::ToString(shift));
		    }
		    string_t target = StringVector::EmptyString(result, input.GetSize());

		    if (shift >= max_shift) {
			    Bit::SetEmptyBitString(target, input);
			    return target;
		    }
		    Bit::LeftShift(input, shift, target);
		    return target;
	    });
}

ScalarFunctionSet LeftShiftFun::GetFunctions() {
	ScalarFunctionSet functions;
	for (auto &type : LogicalType::Integral()) {
		functions.AddFunction(
		    ScalarFunction({type, type}, type, GetScalarIntegerBinaryFunction<BitwiseShiftLeftOperator>(type)));
	}
	functions.AddFunction(
	    ScalarFunction({LogicalType::BIT, LogicalType::INTEGER}, LogicalType::BIT, BitwiseShiftLeftOperation));
	return functions;
}

//===--------------------------------------------------------------------===//
// >> [bitwise_right_shift]
//===--------------------------------------------------------------------===//
template <class T>
bool RightShiftInRange(T shift) {
	return shift >= 0 && shift < T(sizeof(T) * 8);
}

struct BitwiseShiftRightOperator {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA input, TB shift) {
		return RightShiftInRange(shift) ? input >> shift : 0;
	}
};

static void BitwiseShiftRightOperation(DataChunk &args, ExpressionState &state, Vector &result) {
	BinaryExecutor::Execute<string_t, int32_t, string_t>(
	    args.data[0], args.data[1], result, args.size(), [&](string_t input, int32_t shift) {
		    int32_t max_shift = Bit::BitLength(input);
		    if (shift == 0) {
			    return input;
		    }
		    string_t target = StringVector::EmptyString(result, input.GetSize());
		    if (shift < 0 || shift >= max_shift) {
			    Bit::SetEmptyBitString(target, input);
			    return target;
		    }
		    Bit::RightShift(input, shift, target);
		    return target;
	    });
}

ScalarFunctionSet RightShiftFun::GetFunctions() {
	ScalarFunctionSet functions;
	for (auto &type : LogicalType::Integral()) {
		functions.AddFunction(
		    ScalarFunction({type, type}, type, GetScalarIntegerBinaryFunction<BitwiseShiftRightOperator>(type)));
	}
	functions.AddFunction(
	    ScalarFunction({LogicalType::BIT, LogicalType::INTEGER}, LogicalType::BIT, BitwiseShiftRightOperation));
	return functions;
}

} // namespace duckdb








namespace duckdb {

struct RandomLocalState : public FunctionLocalState {
	explicit RandomLocalState(uint32_t seed) : random_engine(seed) {
	}

	RandomEngine random_engine;
};

static void RandomFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(args.ColumnCount() == 0);
	auto &lstate = ExecuteFunctionState::GetFunctionState(state)->Cast<RandomLocalState>();

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_data = FlatVector::GetData<double>(result);
	for (idx_t i = 0; i < args.size(); i++) {
		result_data[i] = lstate.random_engine.NextRandom();
	}
}

static unique_ptr<FunctionLocalState> RandomInitLocalState(ExpressionState &state, const BoundFunctionExpression &expr,
                                                           FunctionData *bind_data) {
	auto &random_engine = RandomEngine::Get(state.GetContext());
	lock_guard<mutex> guard(random_engine.lock);
	return make_uniq<RandomLocalState>(random_engine.NextRandomInteger());
}

ScalarFunction RandomFun::GetFunction() {
	ScalarFunction random("random", {}, LogicalType::DOUBLE, RandomFunction, nullptr, nullptr, nullptr,
	                      RandomInitLocalState);
	random.side_effects = FunctionSideEffects::HAS_SIDE_EFFECTS;
	return random;
}

static void GenerateUUIDFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(args.ColumnCount() == 0);
	auto &lstate = ExecuteFunctionState::GetFunctionState(state)->Cast<RandomLocalState>();

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_data = FlatVector::GetData<hugeint_t>(result);

	for (idx_t i = 0; i < args.size(); i++) {
		result_data[i] = UUID::GenerateRandomUUID(lstate.random_engine);
	}
}

ScalarFunction UUIDFun::GetFunction() {
	ScalarFunction uuid_function({}, LogicalType::UUID, GenerateUUIDFunction, nullptr, nullptr, nullptr,
	                             RandomInitLocalState);
	// generate a random uuid
	uuid_function.side_effects = FunctionSideEffects::HAS_SIDE_EFFECTS;
	return uuid_function;
}

} // namespace duckdb









namespace duckdb {

struct SetseedBindData : public FunctionData {
	//! The client context for the function call
	ClientContext &context;

	explicit SetseedBindData(ClientContext &context) : context(context) {
	}

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<SetseedBindData>(context);
	}

	bool Equals(const FunctionData &other_p) const override {
		return true;
	}
};

static void SetSeedFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &info = func_expr.bind_info->Cast<SetseedBindData>();
	auto &input = args.data[0];
	input.Flatten(args.size());

	auto input_seeds = FlatVector::GetData<double>(input);
	uint32_t half_max = NumericLimits<uint32_t>::Maximum() / 2;

	auto &random_engine = RandomEngine::Get(info.context);
	for (idx_t i = 0; i < args.size(); i++) {
		if (input_seeds[i] < -1.0 || input_seeds[i] > 1.0 || Value::IsNan(input_seeds[i])) {
			throw Exception("SETSEED accepts seed values between -1.0 and 1.0, inclusive");
		}
		uint32_t norm_seed = (input_seeds[i] + 1.0) * half_max;
		random_engine.SetSeed(norm_seed);
	}

	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	ConstantVector::SetNull(result, true);
}

unique_ptr<FunctionData> SetSeedBind(ClientContext &context, ScalarFunction &bound_function,
                                     vector<unique_ptr<Expression>> &arguments) {
	return make_uniq<SetseedBindData>(context);
}

ScalarFunction SetseedFun::GetFunction() {
	ScalarFunction setseed("setseed", {LogicalType::DOUBLE}, LogicalType::SQLNULL, SetSeedFunction, SetSeedBind);
	setseed.side_effects = FunctionSideEffects::HAS_SIDE_EFFECTS;
	return setseed;
}

} // namespace duckdb




namespace duckdb {

struct AsciiOperator {
	template <class TA, class TR>
	static inline TR Operation(const TA &input) {
		auto str = input.GetData();
		if (Utf8Proc::Analyze(str, input.GetSize()) == UnicodeType::ASCII) {
			return str[0];
		}
		int utf8_bytes = 4;
		return Utf8Proc::UTF8ToCodepoint(str, utf8_bytes);
	}
};

ScalarFunction ASCIIFun::GetFunction() {
	return ScalarFunction({LogicalType::VARCHAR}, LogicalType::INTEGER,
	                      ScalarFunction::UnaryFunction<string_t, int32_t, AsciiOperator>);
}

} // namespace duckdb









namespace duckdb {

static string_t BarScalarFunction(double x, double min, double max, double max_width, string &result) {
	static const char *FULL_BLOCK = UnicodeBar::FullBlock();
	static const char *const *PARTIAL_BLOCKS = UnicodeBar::PartialBlocks();
	static const idx_t PARTIAL_BLOCKS_COUNT = UnicodeBar::PartialBlocksCount();

	if (!Value::IsFinite(max_width)) {
		throw ValueOutOfRangeException("Max bar width must not be NaN or infinity");
	}
	if (max_width < 1) {
		throw ValueOutOfRangeException("Max bar width must be >= 1");
	}
	if (max_width > 1000) {
		throw ValueOutOfRangeException("Max bar width must be <= 1000");
	}

	double width;

	if (Value::IsNan(x) || Value::IsNan(min) || Value::IsNan(max) || x <= min) {
		width = 0;
	} else if (x >= max) {
		width = max_width;
	} else {
		width = max_width * (x - min) / (max - min);
	}

	if (!Value::IsFinite(width)) {
		throw ValueOutOfRangeException("Bar width must not be NaN or infinity");
	}

	result.clear();

	int32_t width_as_int = static_cast<int32_t>(width * PARTIAL_BLOCKS_COUNT);
	idx_t full_blocks_count = (width_as_int / PARTIAL_BLOCKS_COUNT);
	for (idx_t i = 0; i < full_blocks_count; i++) {
		result += FULL_BLOCK;
	}

	idx_t remaining = width_as_int % PARTIAL_BLOCKS_COUNT;

	if (remaining) {
		result += PARTIAL_BLOCKS[remaining];
	}

	return string_t(result);
}

static void BarFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(args.ColumnCount() == 3 || args.ColumnCount() == 4);
	auto &x_arg = args.data[0];
	auto &min_arg = args.data[1];
	auto &max_arg = args.data[2];
	string buffer;

	if (args.ColumnCount() == 3) {
		GenericExecutor::ExecuteTernary<PrimitiveType<double>, PrimitiveType<double>, PrimitiveType<double>,
		                                PrimitiveType<string_t>>(
		    x_arg, min_arg, max_arg, result, args.size(),
		    [&](PrimitiveType<double> x, PrimitiveType<double> min, PrimitiveType<double> max) {
			    return StringVector::AddString(result, BarScalarFunction(x.val, min.val, max.val, 80, buffer));
		    });
	} else {
		auto &width_arg = args.data[3];
		GenericExecutor::ExecuteQuaternary<PrimitiveType<double>, PrimitiveType<double>, PrimitiveType<double>,
		                                   PrimitiveType<double>, PrimitiveType<string_t>>(
		    x_arg, min_arg, max_arg, width_arg, result, args.size(),
		    [&](PrimitiveType<double> x, PrimitiveType<double> min, PrimitiveType<double> max,
		        PrimitiveType<double> width) {
			    return StringVector::AddString(result, BarScalarFunction(x.val, min.val, max.val, width.val, buffer));
		    });
	}
}

ScalarFunctionSet BarFun::GetFunctions() {
	ScalarFunctionSet bar;
	bar.AddFunction(ScalarFunction({LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE},
	                               LogicalType::VARCHAR, BarFunction));
	bar.AddFunction(ScalarFunction({LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE},
	                               LogicalType::VARCHAR, BarFunction));
	return bar;
}

} // namespace duckdb




namespace duckdb {

struct ChrOperator {
	static void GetCodepoint(int32_t input, char c[], int &utf8_bytes) {
		if (input < 0 || !Utf8Proc::CodepointToUtf8(input, utf8_bytes, &c[0])) {
			throw InvalidInputException("Invalid UTF8 Codepoint %d", input);
		}
	}

	template <class TA, class TR>
	static inline TR Operation(const TA &input) {
		char c[5] = {'\0', '\0', '\0', '\0', '\0'};
		int utf8_bytes;
		GetCodepoint(input, c, utf8_bytes);
		return string_t(&c[0], utf8_bytes);
	}
};

#ifdef DUCKDB_DEBUG_NO_INLINE
// the chr function depends on the data always being inlined (which is always possible, since it outputs max 4 bytes)
// to enable chr when string inlining is disabled we create a special function here
static void ChrFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &code_vec = args.data[0];

	char c[5] = {'\0', '\0', '\0', '\0', '\0'};
	int utf8_bytes;
	UnaryExecutor::Execute<int32_t, string_t>(code_vec, result, args.size(), [&](int32_t input) {
		ChrOperator::GetCodepoint(input, c, utf8_bytes);
		return StringVector::AddString(result, &c[0], utf8_bytes);
	});
}
#endif

ScalarFunction ChrFun::GetFunction() {
	return ScalarFunction("chr", {LogicalType::INTEGER}, LogicalType::VARCHAR,
#ifdef DUCKDB_DEBUG_NO_INLINE
	                      ChrFunction
#else
	                      ScalarFunction::UnaryFunction<int32_t, string_t, ChrOperator>
#endif
	);
}

} // namespace duckdb




namespace duckdb {

// Using Lowrance-Wagner (LW) algorithm: https://doi.org/10.1145%2F321879.321880
// Can't calculate as trivial modification to levenshtein algorithm
// as we need to potentially know about earlier in the string
static idx_t DamerauLevenshteinDistance(const string_t &source, const string_t &target) {
	// costs associated with each type of edit, to aid readability
	constexpr uint8_t COST_SUBSTITUTION = 1;
	constexpr uint8_t COST_INSERTION = 1;
	constexpr uint8_t COST_DELETION = 1;
	constexpr uint8_t COST_TRANSPOSITION = 1;
	const auto source_len = source.GetSize();
	const auto target_len = target.GetSize();

	// If one string is empty, the distance equals the length of the other string
	// either through target_len insertions
	// or source_len deletions
	if (source_len == 0) {
		return target_len * COST_INSERTION;
	} else if (target_len == 0) {
		return source_len * COST_DELETION;
	}

	const auto source_str = source.GetData();
	const auto target_str = target.GetData();

	// larger than the largest possible value:
	const auto inf = source_len * COST_DELETION + target_len * COST_INSERTION + 1;
	// minimum edit distance from prefix of source string to prefix of target string
	// same object as H in LW paper (with indices offset by 1)
	vector<vector<idx_t>> distance(source_len + 2, vector<idx_t>(target_len + 2, inf));
	// keeps track of the largest string indices of source string matching each character
	// same as DA in LW paper
	map<char, idx_t> largest_source_chr_matching;

	// initialise row/column corresponding to zero-length strings
	// partial string -> empty requires a deletion for each character
	for (idx_t source_idx = 0; source_idx <= source_len; source_idx++) {
		distance[source_idx + 1][1] = source_idx * COST_DELETION;
	}
	// and empty -> partial string means simply inserting characters
	for (idx_t target_idx = 1; target_idx <= target_len; target_idx++) {
		distance[1][target_idx + 1] = target_idx * COST_INSERTION;
	}
	// loop through string indices - these are offset by 2 from distance indices
	for (idx_t source_idx = 0; source_idx < source_len; source_idx++) {
		// keeps track of the largest string indices of target string matching current source character
		// same as DB in LW paper
		idx_t largest_target_chr_matching;
		largest_target_chr_matching = 0;
		for (idx_t target_idx = 0; target_idx < target_len; target_idx++) {
			// correspond to i1 and j1 in LW paper respectively
			idx_t largest_source_chr_matching_target;
			idx_t largest_target_chr_matching_source;
			// cost associated to diagnanl shift in distance matrix
			// corresponds to d in LW paper
			uint8_t cost_diagonal_shift;
			largest_source_chr_matching_target = largest_source_chr_matching[target_str[target_idx]];
			largest_target_chr_matching_source = largest_target_chr_matching;
			// if characters match, diagonal move costs nothing and we update our largest target index
			// otherwise move is substitution and costs as such
			if (source_str[source_idx] == target_str[target_idx]) {
				cost_diagonal_shift = 0;
				largest_target_chr_matching = target_idx + 1;
			} else {
				cost_diagonal_shift = COST_SUBSTITUTION;
			}
			distance[source_idx + 2][target_idx + 2] = MinValue(
			    distance[source_idx + 1][target_idx + 1] + cost_diagonal_shift,
			    MinValue(distance[source_idx + 2][target_idx + 1] + COST_INSERTION,
			             MinValue(distance[source_idx + 1][target_idx + 2] + COST_DELETION,
			                      distance[largest_source_chr_matching_target][largest_target_chr_matching_source] +
			                          (source_idx - largest_source_chr_matching_target) * COST_DELETION +
			                          COST_TRANSPOSITION +
			                          (target_idx - largest_target_chr_matching_source) * COST_INSERTION)));
		}
		largest_source_chr_matching[source_str[source_idx]] = source_idx + 1;
	}
	return distance[source_len + 1][target_len + 1];
}

static int64_t DamerauLevenshteinScalarFunction(Vector &result, const string_t source, const string_t target) {
	return (int64_t)DamerauLevenshteinDistance(source, target);
}

static void DamerauLevenshteinFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &source_vec = args.data[0];
	auto &target_vec = args.data[1];

	BinaryExecutor::Execute<string_t, string_t, int64_t>(
	    source_vec, target_vec, result, args.size(),
	    [&](string_t source, string_t target) { return DamerauLevenshteinScalarFunction(result, source, target); });
}

ScalarFunction DamerauLevenshteinFun::GetFunction() {
	return ScalarFunction({LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::BIGINT,
	                      DamerauLevenshteinFunction);
}

} // namespace duckdb




namespace duckdb {

static void FormatBytesFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	UnaryExecutor::Execute<int64_t, string_t>(args.data[0], result, args.size(), [&](int64_t bytes) {
		bool is_negative = bytes < 0;
		idx_t unsigned_bytes;
		if (bytes < 0) {
			if (bytes == NumericLimits<int64_t>::Minimum()) {
				unsigned_bytes = idx_t(NumericLimits<int64_t>::Maximum()) + 1;
			} else {
				unsigned_bytes = idx_t(-bytes);
			}
		} else {
			unsigned_bytes = idx_t(bytes);
		}
		return StringVector::AddString(result, (is_negative ? "-" : "") +
		                                           StringUtil::BytesToHumanReadableString(unsigned_bytes));
	});
}

ScalarFunction FormatBytesFun::GetFunction() {
	return ScalarFunction({LogicalType::BIGINT}, LogicalType::VARCHAR, FormatBytesFunction);
}

} // namespace duckdb



#include <ctype.h>
#include <algorithm>

namespace duckdb {

static int64_t MismatchesScalarFunction(Vector &result, const string_t str, string_t tgt) {
	idx_t str_len = str.GetSize();
	idx_t tgt_len = tgt.GetSize();

	if (str_len != tgt_len) {
		throw InvalidInputException("Mismatch Function: Strings must be of equal length!");
	}
	if (str_len < 1) {
		throw InvalidInputException("Mismatch Function: Strings must be of length > 0!");
	}

	idx_t mismatches = 0;
	auto str_str = str.GetData();
	auto tgt_str = tgt.GetData();

	for (idx_t idx = 0; idx < str_len; ++idx) {
		if (str_str[idx] != tgt_str[idx]) {
			mismatches++;
		}
	}
	return (int64_t)mismatches;
}

static void MismatchesFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &str_vec = args.data[0];
	auto &tgt_vec = args.data[1];

	BinaryExecutor::Execute<string_t, string_t, int64_t>(
	    str_vec, tgt_vec, result, args.size(),
	    [&](string_t str, string_t tgt) { return MismatchesScalarFunction(result, str, tgt); });
}

ScalarFunction HammingFun::GetFunction() {
	return ScalarFunction({LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::BIGINT, MismatchesFunction);
}

} // namespace duckdb









namespace duckdb {

static void WriteHexBytes(uint64_t x, char *&output, idx_t buffer_size) {
	idx_t offset = buffer_size * 4;

	for (; offset >= 4; offset -= 4) {
		uint8_t byte = (x >> (offset - 4)) & 0x0F;
		*output = Blob::HEX_TABLE[byte];
		output++;
	}
}

static void WriteHugeIntHexBytes(hugeint_t x, char *&output, idx_t buffer_size) {
	idx_t offset = buffer_size * 4;
	auto upper = x.upper;
	auto lower = x.lower;

	for (; offset >= 68; offset -= 4) {
		uint8_t byte = (upper >> (offset - 68)) & 0x0F;
		*output = Blob::HEX_TABLE[byte];
		output++;
	}

	for (; offset >= 4; offset -= 4) {
		uint8_t byte = (lower >> (offset - 4)) & 0x0F;
		*output = Blob::HEX_TABLE[byte];
		output++;
	}
}

static void WriteBinBytes(uint64_t x, char *&output, idx_t buffer_size) {
	idx_t offset = buffer_size;
	for (; offset >= 1; offset -= 1) {
		*output = ((x >> (offset - 1)) & 0x01) + '0';
		output++;
	}
}

static void WriteHugeIntBinBytes(hugeint_t x, char *&output, idx_t buffer_size) {
	auto upper = x.upper;
	auto lower = x.lower;
	idx_t offset = buffer_size;

	for (; offset >= 65; offset -= 1) {
		*output = ((upper >> (offset - 65)) & 0x01) + '0';
		output++;
	}

	for (; offset >= 1; offset -= 1) {
		*output = ((lower >> (offset - 1)) & 0x01) + '0';
		output++;
	}
}

struct HexStrOperator {
	template <class INPUT_TYPE, class RESULT_TYPE>
	static RESULT_TYPE Operation(INPUT_TYPE input, Vector &result) {
		auto data = input.GetData();
		auto size = input.GetSize();

		// Allocate empty space
		auto target = StringVector::EmptyString(result, size * 2);
		auto output = target.GetDataWriteable();

		for (idx_t i = 0; i < size; ++i) {
			*output = Blob::HEX_TABLE[(data[i] >> 4) & 0x0F];
			output++;
			*output = Blob::HEX_TABLE[data[i] & 0x0F];
			output++;
		}

		target.Finalize();
		return target;
	}
};

struct HexIntegralOperator {
	template <class INPUT_TYPE, class RESULT_TYPE>
	static RESULT_TYPE Operation(INPUT_TYPE input, Vector &result) {

		idx_t num_leading_zero = CountZeros<uint64_t>::Leading(input);
		idx_t num_bits_to_check = 64 - num_leading_zero;
		D_ASSERT(num_bits_to_check <= sizeof(INPUT_TYPE) * 8);

		idx_t buffer_size = (num_bits_to_check + 3) / 4;

		// Special case: All bits are zero
		if (buffer_size == 0) {
			auto target = StringVector::EmptyString(result, 1);
			auto output = target.GetDataWriteable();
			*output = '0';
			target.Finalize();
			return target;
		}

		D_ASSERT(buffer_size > 0);
		auto target = StringVector::EmptyString(result, buffer_size);
		auto output = target.GetDataWriteable();

		WriteHexBytes(input, output, buffer_size);

		target.Finalize();
		return target;
	}
};

struct HexHugeIntOperator {
	template <class INPUT_TYPE, class RESULT_TYPE>
	static RESULT_TYPE Operation(INPUT_TYPE input, Vector &result) {

		idx_t num_leading_zero = CountZeros<hugeint_t>::Leading(input);
		idx_t buffer_size = sizeof(INPUT_TYPE) * 2 - (num_leading_zero / 4);

		// Special case: All bits are zero
		if (buffer_size == 0) {
			auto target = StringVector::EmptyString(result, 1);
			auto output = target.GetDataWriteable();
			*output = '0';
			target.Finalize();
			return target;
		}

		D_ASSERT(buffer_size > 0);
		auto target = StringVector::EmptyString(result, buffer_size);
		auto output = target.GetDataWriteable();

		WriteHugeIntHexBytes(input, output, buffer_size);

		target.Finalize();
		return target;
	}
};

template <class INPUT, class OP>
static void ToHexFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(args.ColumnCount() == 1);
	auto &input = args.data[0];
	idx_t count = args.size();
	UnaryExecutor::ExecuteString<INPUT, string_t, OP>(input, result, count);
}

struct BinaryStrOperator {
	template <class INPUT_TYPE, class RESULT_TYPE>
	static RESULT_TYPE Operation(INPUT_TYPE input, Vector &result) {
		auto data = input.GetData();
		auto size = input.GetSize();

		// Allocate empty space
		auto target = StringVector::EmptyString(result, size * 8);
		auto output = target.GetDataWriteable();

		for (idx_t i = 0; i < size; ++i) {
			uint8_t byte = data[i];
			for (idx_t i = 8; i >= 1; --i) {
				*output = ((byte >> (i - 1)) & 0x01) + '0';
				output++;
			}
		}

		target.Finalize();
		return target;
	}
};

struct BinaryIntegralOperator {
	template <class INPUT_TYPE, class RESULT_TYPE>
	static RESULT_TYPE Operation(INPUT_TYPE input, Vector &result) {

		idx_t num_leading_zero = CountZeros<uint64_t>::Leading(input);
		idx_t num_bits_to_check = 64 - num_leading_zero;
		D_ASSERT(num_bits_to_check <= sizeof(INPUT_TYPE) * 8);

		idx_t buffer_size = num_bits_to_check;

		// Special case: All bits are zero
		if (buffer_size == 0) {
			auto target = StringVector::EmptyString(result, 1);
			auto output = target.GetDataWriteable();
			*output = '0';
			target.Finalize();
			return target;
		}

		D_ASSERT(buffer_size > 0);
		auto target = StringVector::EmptyString(result, buffer_size);
		auto output = target.GetDataWriteable();

		WriteBinBytes(input, output, buffer_size);

		target.Finalize();
		return target;
	}
};

struct BinaryHugeIntOperator {
	template <class INPUT_TYPE, class RESULT_TYPE>
	static RESULT_TYPE Operation(INPUT_TYPE input, Vector &result) {
		idx_t num_leading_zero = CountZeros<hugeint_t>::Leading(input);
		idx_t buffer_size = sizeof(INPUT_TYPE) * 8 - num_leading_zero;

		// Special case: All bits are zero
		if (buffer_size == 0) {
			auto target = StringVector::EmptyString(result, 1);
			auto output = target.GetDataWriteable();
			*output = '0';
			target.Finalize();
			return target;
		}

		auto target = StringVector::EmptyString(result, buffer_size);
		auto output = target.GetDataWriteable();

		WriteHugeIntBinBytes(input, output, buffer_size);

		target.Finalize();
		return target;
	}
};

struct FromHexOperator {
	template <class INPUT_TYPE, class RESULT_TYPE>
	static RESULT_TYPE Operation(INPUT_TYPE input, Vector &result) {
		auto data = input.GetData();
		auto size = input.GetSize();

		if (size > NumericLimits<uint32_t>::Maximum()) {
			throw InvalidInputException("Hexadecimal input length larger than 2^32 are not supported");
		}

		D_ASSERT(size <= NumericLimits<uint32_t>::Maximum());
		auto buffer_size = (size + 1) / 2;

		// Allocate empty space
		auto target = StringVector::EmptyString(result, buffer_size);
		auto output = target.GetDataWriteable();

		// Treated as a single byte
		idx_t i = 0;
		if (size % 2 != 0) {
			*output = StringUtil::GetHexValue(data[i]);
			i++;
			output++;
		}

		for (; i < size; i += 2) {
			uint8_t major = StringUtil::GetHexValue(data[i]);
			uint8_t minor = StringUtil::GetHexValue(data[i + 1]);
			*output = (major << 4) | minor;
			output++;
		}

		target.Finalize();
		return target;
	}
};

struct FromBinaryOperator {
	template <class INPUT_TYPE, class RESULT_TYPE>
	static RESULT_TYPE Operation(INPUT_TYPE input, Vector &result) {
		auto data = input.GetData();
		auto size = input.GetSize();

		if (size > NumericLimits<uint32_t>::Maximum()) {
			throw InvalidInputException("Binary input length larger than 2^32 are not supported");
		}

		D_ASSERT(size <= NumericLimits<uint32_t>::Maximum());
		auto buffer_size = (size + 7) / 8;

		// Allocate empty space
		auto target = StringVector::EmptyString(result, buffer_size);
		auto output = target.GetDataWriteable();

		// Treated as a single byte
		idx_t i = 0;
		if (size % 8 != 0) {
			uint8_t byte = 0;
			for (idx_t j = size % 8; j > 0; --j) {
				byte |= StringUtil::GetBinaryValue(data[i]) << (j - 1);
				i++;
			}
			*output = byte;
			output++;
		}

		while (i < size) {
			uint8_t byte = 0;
			for (idx_t j = 8; j > 0; --j) {
				byte |= StringUtil::GetBinaryValue(data[i]) << (j - 1);
				i++;
			}
			*output = byte;
			output++;
		}

		target.Finalize();
		return target;
	}
};

template <class INPUT, class OP>
static void ToBinaryFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(args.ColumnCount() == 1);
	auto &input = args.data[0];
	idx_t count = args.size();
	UnaryExecutor::ExecuteString<INPUT, string_t, OP>(input, result, count);
}

static void FromBinaryFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(args.ColumnCount() == 1);
	D_ASSERT(args.data[0].GetType().InternalType() == PhysicalType::VARCHAR);
	auto &input = args.data[0];
	idx_t count = args.size();

	UnaryExecutor::ExecuteString<string_t, string_t, FromBinaryOperator>(input, result, count);
}

static void FromHexFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(args.ColumnCount() == 1);
	D_ASSERT(args.data[0].GetType().InternalType() == PhysicalType::VARCHAR);
	auto &input = args.data[0];
	idx_t count = args.size();

	UnaryExecutor::ExecuteString<string_t, string_t, FromHexOperator>(input, result, count);
}

ScalarFunctionSet HexFun::GetFunctions() {
	ScalarFunctionSet to_hex;
	to_hex.AddFunction(
	    ScalarFunction({LogicalType::VARCHAR}, LogicalType::VARCHAR, ToHexFunction<string_t, HexStrOperator>));

	to_hex.AddFunction(
	    ScalarFunction({LogicalType::BIGINT}, LogicalType::VARCHAR, ToHexFunction<int64_t, HexIntegralOperator>));

	to_hex.AddFunction(
	    ScalarFunction({LogicalType::UBIGINT}, LogicalType::VARCHAR, ToHexFunction<uint64_t, HexIntegralOperator>));

	to_hex.AddFunction(
	    ScalarFunction({LogicalType::HUGEINT}, LogicalType::VARCHAR, ToHexFunction<hugeint_t, HexHugeIntOperator>));
	return to_hex;
}

ScalarFunction UnhexFun::GetFunction() {
	return ScalarFunction({LogicalType::VARCHAR}, LogicalType::BLOB, FromHexFunction);
}

ScalarFunctionSet BinFun::GetFunctions() {
	ScalarFunctionSet to_binary;

	to_binary.AddFunction(
	    ScalarFunction({LogicalType::VARCHAR}, LogicalType::VARCHAR, ToBinaryFunction<string_t, BinaryStrOperator>));
	to_binary.AddFunction(ScalarFunction({LogicalType::UBIGINT}, LogicalType::VARCHAR,
	                                     ToBinaryFunction<uint64_t, BinaryIntegralOperator>));
	to_binary.AddFunction(
	    ScalarFunction({LogicalType::BIGINT}, LogicalType::VARCHAR, ToBinaryFunction<int64_t, BinaryIntegralOperator>));
	to_binary.AddFunction(ScalarFunction({LogicalType::HUGEINT}, LogicalType::VARCHAR,
	                                     ToBinaryFunction<hugeint_t, BinaryHugeIntOperator>));
	return to_binary;
}

ScalarFunction UnbinFun::GetFunction() {
	return ScalarFunction({LogicalType::VARCHAR}, LogicalType::BLOB, FromBinaryFunction);
}

} // namespace duckdb








namespace duckdb {

struct InstrOperator {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA haystack, TB needle) {
		int64_t string_position = 0;

		auto location = ContainsFun::Find(haystack, needle);
		if (location != DConstants::INVALID_INDEX) {
			auto len = (utf8proc_ssize_t)location;
			auto str = reinterpret_cast<const utf8proc_uint8_t *>(haystack.GetData());
			D_ASSERT(len <= (utf8proc_ssize_t)haystack.GetSize());
			for (++string_position; len > 0; ++string_position) {
				utf8proc_int32_t codepoint;
				auto bytes = utf8proc_iterate(str, len, &codepoint);
				str += bytes;
				len -= bytes;
			}
		}
		return string_position;
	}
};

struct InstrAsciiOperator {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA haystack, TB needle) {
		auto location = ContainsFun::Find(haystack, needle);
		return location == DConstants::INVALID_INDEX ? 0 : location + 1;
	}
};

static unique_ptr<BaseStatistics> InStrPropagateStats(ClientContext &context, FunctionStatisticsInput &input) {
	auto &child_stats = input.child_stats;
	auto &expr = input.expr;
	D_ASSERT(child_stats.size() == 2);
	// can only propagate stats if the children have stats
	// for strpos, we only care if the FIRST string has unicode or not
	if (!StringStats::CanContainUnicode(child_stats[0])) {
		expr.function.function = ScalarFunction::BinaryFunction<string_t, string_t, int64_t, InstrAsciiOperator>;
	}
	return nullptr;
}

ScalarFunction InstrFun::GetFunction() {
	return ScalarFunction({LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::BIGINT,
	                      ScalarFunction::BinaryFunction<string_t, string_t, int64_t, InstrOperator>, nullptr, nullptr,
	                      InStrPropagateStats);
}

} // namespace duckdb




#include <ctype.h>

namespace duckdb {

static inline map<char, idx_t> GetSet(const string_t &str) {
	auto map_of_chars = map<char, idx_t> {};
	idx_t str_len = str.GetSize();
	auto s = str.GetData();

	for (idx_t pos = 0; pos < str_len; pos++) {
		map_of_chars.insert(std::make_pair(s[pos], 1));
	}
	return map_of_chars;
}

static double JaccardSimilarity(const string_t &str, const string_t &txt) {
	if (str.GetSize() < 1 || txt.GetSize() < 1) {
		throw InvalidInputException("Jaccard Function: An argument too short!");
	}
	map<char, idx_t> m_str, m_txt;

	m_str = GetSet(str);
	m_txt = GetSet(txt);

	if (m_str.size() > m_txt.size()) {
		m_str.swap(m_txt);
	}

	for (auto const &achar : m_str) {
		++m_txt[achar.first];
	}
	// m_txt.size is now size of union.

	idx_t size_intersect = 0;
	for (const auto &apair : m_txt) {
		if (apair.second > 1) {
			size_intersect++;
		}
	}

	return (double)size_intersect / (double)m_txt.size();
}

static double JaccardScalarFunction(Vector &result, const string_t str, string_t tgt) {
	return (double)JaccardSimilarity(str, tgt);
}

static void JaccardFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &str_vec = args.data[0];
	auto &tgt_vec = args.data[1];

	BinaryExecutor::Execute<string_t, string_t, double>(
	    str_vec, tgt_vec, result, args.size(),
	    [&](string_t str, string_t tgt) { return JaccardScalarFunction(result, str, tgt); });
}

ScalarFunction JaccardFun::GetFunction() {
	return ScalarFunction({LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::DOUBLE, JaccardFunction);
}

} // namespace duckdb




namespace duckdb {

static inline double JaroScalarFunction(const string_t &s1, const string_t &s2) {
	auto s1_begin = s1.GetData();
	auto s2_begin = s2.GetData();
	return duckdb_jaro_winkler::jaro_similarity(s1_begin, s1_begin + s1.GetSize(), s2_begin, s2_begin + s2.GetSize());
}

static inline double JaroWinklerScalarFunction(const string_t &s1, const string_t &s2) {
	auto s1_begin = s1.GetData();
	auto s2_begin = s2.GetData();
	return duckdb_jaro_winkler::jaro_winkler_similarity(s1_begin, s1_begin + s1.GetSize(), s2_begin,
	                                                    s2_begin + s2.GetSize());
}

template <class CACHED_SIMILARITY>
static void CachedFunction(Vector &constant, Vector &other, Vector &result, idx_t count) {
	auto val = constant.GetValue(0);
	if (val.IsNull()) {
		auto &result_validity = FlatVector::Validity(result);
		result_validity.SetAllInvalid(count);
		return;
	}

	auto str_val = StringValue::Get(val);
	auto cached = CACHED_SIMILARITY(str_val);
	UnaryExecutor::Execute<string_t, double>(other, result, count, [&](const string_t &other_str) {
		auto other_str_begin = other_str.GetData();
		return cached.similarity(other_str_begin, other_str_begin + other_str.GetSize());
	});
}

template <class CACHED_SIMILARITY, class SIMILARITY_FUNCTION = std::function<double(string_t, string_t)>>
static void TemplatedJaroWinklerFunction(DataChunk &args, Vector &result, SIMILARITY_FUNCTION fun) {
	bool arg0_constant = args.data[0].GetVectorType() == VectorType::CONSTANT_VECTOR;
	bool arg1_constant = args.data[1].GetVectorType() == VectorType::CONSTANT_VECTOR;
	if (!(arg0_constant ^ arg1_constant)) {
		// We can't optimize by caching one of the two strings
		BinaryExecutor::Execute<string_t, string_t, double>(args.data[0], args.data[1], result, args.size(), fun);
		return;
	}

	if (arg0_constant) {
		CachedFunction<CACHED_SIMILARITY>(args.data[0], args.data[1], result, args.size());
	} else {
		CachedFunction<CACHED_SIMILARITY>(args.data[1], args.data[0], result, args.size());
	}
}

static void JaroFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	TemplatedJaroWinklerFunction<duckdb_jaro_winkler::CachedJaroSimilarity<char>>(args, result, JaroScalarFunction);
}

static void JaroWinklerFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	TemplatedJaroWinklerFunction<duckdb_jaro_winkler::CachedJaroWinklerSimilarity<char>>(args, result,
	                                                                                     JaroWinklerScalarFunction);
}

ScalarFunction JaroSimilarityFun::GetFunction() {
	return ScalarFunction({LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::DOUBLE, JaroFunction);
}

ScalarFunction JaroWinklerSimilarityFun::GetFunction() {
	return ScalarFunction({LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::DOUBLE, JaroWinklerFunction);
}

} // namespace duckdb





#include <ctype.h>
#include <algorithm>

namespace duckdb {

struct LeftRightUnicode {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		return LengthFun::Length<TA, TR>(input);
	}

	static string_t Substring(Vector &result, string_t input, int64_t offset, int64_t length) {
		return SubstringFun::SubstringUnicode(result, input, offset, length);
	}
};

struct LeftRightGrapheme {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		return LengthFun::GraphemeCount<TA, TR>(input);
	}

	static string_t Substring(Vector &result, string_t input, int64_t offset, int64_t length) {
		return SubstringFun::SubstringGrapheme(result, input, offset, length);
	}
};

template <class OP>
static string_t LeftScalarFunction(Vector &result, const string_t str, int64_t pos) {
	if (pos >= 0) {
		return OP::Substring(result, str, 1, pos);
	}

	int64_t num_characters = OP::template Operation<string_t, int64_t>(str);
	pos = MaxValue<int64_t>(0, num_characters + pos);
	return OP::Substring(result, str, 1, pos);
}

template <class OP>
static void LeftFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &str_vec = args.data[0];
	auto &pos_vec = args.data[1];

	BinaryExecutor::Execute<string_t, int64_t, string_t>(
	    str_vec, pos_vec, result, args.size(),
	    [&](string_t str, int64_t pos) { return LeftScalarFunction<OP>(result, str, pos); });
}

ScalarFunction LeftFun::GetFunction() {
	return ScalarFunction({LogicalType::VARCHAR, LogicalType::BIGINT}, LogicalType::VARCHAR,
	                      LeftFunction<LeftRightUnicode>);
}

ScalarFunction LeftGraphemeFun::GetFunction() {
	return ScalarFunction({LogicalType::VARCHAR, LogicalType::BIGINT}, LogicalType::VARCHAR,
	                      LeftFunction<LeftRightGrapheme>);
}

template <class OP>
static string_t RightScalarFunction(Vector &result, const string_t str, int64_t pos) {
	int64_t num_characters = OP::template Operation<string_t, int64_t>(str);
	if (pos >= 0) {
		int64_t len = MinValue<int64_t>(num_characters, pos);
		int64_t start = num_characters - len + 1;
		return OP::Substring(result, str, start, len);
	}

	int64_t len = 0;
	if (pos != std::numeric_limits<int64_t>::min()) {
		len = num_characters - MinValue<int64_t>(num_characters, -pos);
	}
	int64_t start = num_characters - len + 1;
	return OP::Substring(result, str, start, len);
}

template <class OP>
static void RightFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &str_vec = args.data[0];
	auto &pos_vec = args.data[1];
	BinaryExecutor::Execute<string_t, int64_t, string_t>(
	    str_vec, pos_vec, result, args.size(),
	    [&](string_t str, int64_t pos) { return RightScalarFunction<OP>(result, str, pos); });
}

ScalarFunction RightFun::GetFunction() {
	return ScalarFunction({LogicalType::VARCHAR, LogicalType::BIGINT}, LogicalType::VARCHAR,
	                      RightFunction<LeftRightUnicode>);
}

ScalarFunction RightGraphemeFun::GetFunction() {
	return ScalarFunction({LogicalType::VARCHAR, LogicalType::BIGINT}, LogicalType::VARCHAR,
	                      RightFunction<LeftRightGrapheme>);
}

} // namespace duckdb




#include <ctype.h>
#include <algorithm>

namespace duckdb {

// See: https://www.kdnuggets.com/2020/10/optimizing-levenshtein-distance-measuring-text-similarity.html
// And: Iterative 2-row algorithm: https://en.wikipedia.org/wiki/Levenshtein_distance
// Note: A first implementation using the array algorithm version resulted in an error raised by duckdb
// (too muach memory usage)

static idx_t LevenshteinDistance(const string_t &txt, const string_t &tgt) {
	auto txt_len = txt.GetSize();
	auto tgt_len = tgt.GetSize();

	// If one string is empty, the distance equals the length of the other string
	if (txt_len == 0) {
		return tgt_len;
	} else if (tgt_len == 0) {
		return txt_len;
	}

	auto txt_str = txt.GetData();
	auto tgt_str = tgt.GetData();

	// Create two working vectors
	vector<idx_t> distances0(tgt_len + 1, 0);
	vector<idx_t> distances1(tgt_len + 1, 0);

	idx_t cost_substitution = 0;
	idx_t cost_insertion = 0;
	idx_t cost_deletion = 0;

	// initialize distances0 vector
	// edit distance for an empty txt string is just the number of characters to delete from tgt
	for (idx_t pos_tgt = 0; pos_tgt <= tgt_len; pos_tgt++) {
		distances0[pos_tgt] = pos_tgt;
	}

	for (idx_t pos_txt = 0; pos_txt < txt_len; pos_txt++) {
		// calculate distances1 (current raw distances) from the previous row

		distances1[0] = pos_txt + 1;

		for (idx_t pos_tgt = 0; pos_tgt < tgt_len; pos_tgt++) {
			cost_deletion = distances0[pos_tgt + 1] + 1;
			cost_insertion = distances1[pos_tgt] + 1;
			cost_substitution = distances0[pos_tgt];

			if (txt_str[pos_txt] != tgt_str[pos_tgt]) {
				cost_substitution += 1;
			}

			distances1[pos_tgt + 1] = MinValue(cost_deletion, MinValue(cost_substitution, cost_insertion));
		}
		// copy distances1 (current row) to distances0 (previous row) for next iteration
		// since data in distances1 is always invalidated, a swap without copy is more efficient
		distances0 = distances1;
	}

	return distances0[tgt_len];
}

static int64_t LevenshteinScalarFunction(Vector &result, const string_t str, string_t tgt) {
	return (int64_t)LevenshteinDistance(str, tgt);
}

static void LevenshteinFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &str_vec = args.data[0];
	auto &tgt_vec = args.data[1];

	BinaryExecutor::Execute<string_t, string_t, int64_t>(
	    str_vec, tgt_vec, result, args.size(),
	    [&](string_t str, string_t tgt) { return LevenshteinScalarFunction(result, str, tgt); });
}

ScalarFunction LevenshteinFun::GetFunction() {
	return ScalarFunction({LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::BIGINT, LevenshteinFunction);
}

} // namespace duckdb






namespace duckdb {

struct MD5Operator {
	template <class INPUT_TYPE, class RESULT_TYPE>
	static RESULT_TYPE Operation(INPUT_TYPE input, Vector &result) {
		auto hash = StringVector::EmptyString(result, MD5Context::MD5_HASH_LENGTH_TEXT);
		MD5Context context;
		context.Add(input);
		context.FinishHex(hash.GetDataWriteable());
		hash.Finalize();
		return hash;
	}
};

struct MD5Number128Operator {
	template <class INPUT_TYPE, class RESULT_TYPE>
	static RESULT_TYPE Operation(INPUT_TYPE input) {
		data_t digest[MD5Context::MD5_HASH_LENGTH_BINARY];

		MD5Context context;
		context.Add(input);
		context.Finish(digest);
		return *reinterpret_cast<hugeint_t *>(digest);
	}
};

template <bool lower>
struct MD5Number64Operator {
	template <class INPUT_TYPE, class RESULT_TYPE>
	static RESULT_TYPE Operation(INPUT_TYPE input) {
		data_t digest[MD5Context::MD5_HASH_LENGTH_BINARY];

		MD5Context context;
		context.Add(input);
		context.Finish(digest);
		return *reinterpret_cast<uint64_t *>(&digest[lower ? 8 : 0]);
	}
};

static void MD5Function(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &input = args.data[0];

	UnaryExecutor::ExecuteString<string_t, string_t, MD5Operator>(input, result, args.size());
}

static void MD5NumberFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &input = args.data[0];

	UnaryExecutor::Execute<string_t, hugeint_t, MD5Number128Operator>(input, result, args.size());
}

static void MD5NumberUpperFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &input = args.data[0];

	UnaryExecutor::Execute<string_t, uint64_t, MD5Number64Operator<false>>(input, result, args.size());
}

static void MD5NumberLowerFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &input = args.data[0];

	UnaryExecutor::Execute<string_t, uint64_t, MD5Number64Operator<true>>(input, result, args.size());
}

ScalarFunction MD5Fun::GetFunction() {
	return ScalarFunction({LogicalType::VARCHAR}, LogicalType::VARCHAR, MD5Function);
}

ScalarFunction MD5NumberFun::GetFunction() {
	return ScalarFunction({LogicalType::VARCHAR}, LogicalType::HUGEINT, MD5NumberFunction);
}

ScalarFunction MD5NumberUpperFun::GetFunction() {
	return ScalarFunction({LogicalType::VARCHAR}, LogicalType::UBIGINT, MD5NumberUpperFunction);
}

ScalarFunction MD5NumberLowerFun::GetFunction() {
	return ScalarFunction({LogicalType::VARCHAR}, LogicalType::UBIGINT, MD5NumberLowerFunction);
}

} // namespace duckdb










namespace duckdb {

static pair<idx_t, idx_t> PadCountChars(const idx_t len, const char *data, const idx_t size) {
	//  Count how much of str will fit in the output
	auto str = reinterpret_cast<const utf8proc_uint8_t *>(data);
	idx_t nbytes = 0;
	idx_t nchars = 0;
	for (; nchars < len && nbytes < size; ++nchars) {
		utf8proc_int32_t codepoint;
		auto bytes = utf8proc_iterate(str + nbytes, size - nbytes, &codepoint);
		D_ASSERT(bytes > 0);
		nbytes += bytes;
	}

	return pair<idx_t, idx_t>(nbytes, nchars);
}

static bool InsertPadding(const idx_t len, const string_t &pad, vector<char> &result) {
	//  Copy the padding until the output is long enough
	auto data = pad.GetData();
	auto size = pad.GetSize();

	//  Check whether we need data that we don't have
	if (len > 0 && size == 0) {
		return false;
	}

	//  Insert characters until we have all we need.
	auto str = reinterpret_cast<const utf8proc_uint8_t *>(data);
	idx_t nbytes = 0;
	for (idx_t nchars = 0; nchars < len; ++nchars) {
		//  If we are at the end of the pad, flush all of it and loop back
		if (nbytes >= size) {
			result.insert(result.end(), data, data + size);
			nbytes = 0;
		}

		//  Write the next character
		utf8proc_int32_t codepoint;
		auto bytes = utf8proc_iterate(str + nbytes, size - nbytes, &codepoint);
		D_ASSERT(bytes > 0);
		nbytes += bytes;
	}

	//  Flush the remaining pad
	result.insert(result.end(), data, data + nbytes);

	return true;
}

static string_t LeftPadFunction(const string_t &str, const int32_t len, const string_t &pad, vector<char> &result) {
	//  Reuse the buffer
	result.clear();

	// Get information about the base string
	auto data_str = str.GetData();
	auto size_str = str.GetSize();

	//  Count how much of str will fit in the output
	auto written = PadCountChars(len, data_str, size_str);

	//  Left pad by the number of characters still needed
	if (!InsertPadding(len - written.second, pad, result)) {
		throw Exception("Insufficient padding in LPAD.");
	}

	//  Append as much of the original string as fits
	result.insert(result.end(), data_str, data_str + written.first);

	return string_t(result.data(), result.size());
}

struct LeftPadOperator {
	static inline string_t Operation(const string_t &str, const int32_t len, const string_t &pad,
	                                 vector<char> &result) {
		return LeftPadFunction(str, len, pad, result);
	}
};

static string_t RightPadFunction(const string_t &str, const int32_t len, const string_t &pad, vector<char> &result) {
	//  Reuse the buffer
	result.clear();

	// Get information about the base string
	auto data_str = str.GetData();
	auto size_str = str.GetSize();

	// Count how much of str will fit in the output
	auto written = PadCountChars(len, data_str, size_str);

	//  Append as much of the original string as fits
	result.insert(result.end(), data_str, data_str + written.first);

	//  Right pad by the number of characters still needed
	if (!InsertPadding(len - written.second, pad, result)) {
		throw Exception("Insufficient padding in RPAD.");
	};

	return string_t(result.data(), result.size());
}

struct RightPadOperator {
	static inline string_t Operation(const string_t &str, const int32_t len, const string_t &pad,
	                                 vector<char> &result) {
		return RightPadFunction(str, len, pad, result);
	}
};

template <class OP>
static void PadFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &str_vector = args.data[0];
	auto &len_vector = args.data[1];
	auto &pad_vector = args.data[2];

	vector<char> buffer;
	TernaryExecutor::Execute<string_t, int32_t, string_t, string_t>(
	    str_vector, len_vector, pad_vector, result, args.size(), [&](string_t str, int32_t len, string_t pad) {
		    len = MaxValue<int32_t>(len, 0);
		    return StringVector::AddString(result, OP::Operation(str, len, pad, buffer));
	    });
}

ScalarFunction LpadFun::GetFunction() {
	return ScalarFunction({LogicalType::VARCHAR, LogicalType::INTEGER, LogicalType::VARCHAR}, LogicalType::VARCHAR,
	                      PadFunction<LeftPadOperator>);
}

ScalarFunction RpadFun::GetFunction() {
	return ScalarFunction({LogicalType::VARCHAR, LogicalType::INTEGER, LogicalType::VARCHAR}, LogicalType::VARCHAR,
	                      PadFunction<RightPadOperator>);
}

} // namespace duckdb






namespace duckdb {

struct FMTPrintf {
	template <class CTX>
	static string OP(const char *format_str, vector<duckdb_fmt::basic_format_arg<CTX>> &format_args) {
		return duckdb_fmt::vsprintf(
		    format_str, duckdb_fmt::basic_format_args<CTX>(format_args.data(), static_cast<int>(format_args.size())));
	}
};

struct FMTFormat {
	template <class CTX>
	static string OP(const char *format_str, vector<duckdb_fmt::basic_format_arg<CTX>> &format_args) {
		return duckdb_fmt::vformat(
		    format_str, duckdb_fmt::basic_format_args<CTX>(format_args.data(), static_cast<int>(format_args.size())));
	}
};

unique_ptr<FunctionData> BindPrintfFunction(ClientContext &context, ScalarFunction &bound_function,
                                            vector<unique_ptr<Expression>> &arguments) {
	for (idx_t i = 1; i < arguments.size(); i++) {
		switch (arguments[i]->return_type.id()) {
		case LogicalTypeId::BOOLEAN:
		case LogicalTypeId::TINYINT:
		case LogicalTypeId::SMALLINT:
		case LogicalTypeId::INTEGER:
		case LogicalTypeId::BIGINT:
		case LogicalTypeId::FLOAT:
		case LogicalTypeId::DOUBLE:
		case LogicalTypeId::VARCHAR:
			// these types are natively supported
			bound_function.arguments.push_back(arguments[i]->return_type);
			break;
		case LogicalTypeId::DECIMAL:
			// decimal type: add cast to double
			bound_function.arguments.emplace_back(LogicalType::DOUBLE);
			break;
		case LogicalTypeId::UNKNOWN:
			// parameter: accept any input and rebind later
			bound_function.arguments.emplace_back(LogicalType::ANY);
			break;
		default:
			// all other types: add cast to string
			bound_function.arguments.emplace_back(LogicalType::VARCHAR);
			break;
		}
	}
	return nullptr;
}

template <class FORMAT_FUN, class CTX>
static void PrintfFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &format_string = args.data[0];
	auto &result_validity = FlatVector::Validity(result);
	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	result_validity.Initialize(args.size());
	for (idx_t i = 0; i < args.ColumnCount(); i++) {
		switch (args.data[i].GetVectorType()) {
		case VectorType::CONSTANT_VECTOR:
			if (ConstantVector::IsNull(args.data[i])) {
				// constant null! result is always NULL regardless of other input
				result.SetVectorType(VectorType::CONSTANT_VECTOR);
				ConstantVector::SetNull(result, true);
				return;
			}
			break;
		default:
			// FLAT VECTOR, we can directly OR the nullmask
			args.data[i].Flatten(args.size());
			result.SetVectorType(VectorType::FLAT_VECTOR);
			result_validity.Combine(FlatVector::Validity(args.data[i]), args.size());
			break;
		}
	}
	idx_t count = result.GetVectorType() == VectorType::CONSTANT_VECTOR ? 1 : args.size();

	auto format_data = FlatVector::GetData<string_t>(format_string);
	auto result_data = FlatVector::GetData<string_t>(result);
	for (idx_t idx = 0; idx < count; idx++) {
		if (result.GetVectorType() == VectorType::FLAT_VECTOR && FlatVector::IsNull(result, idx)) {
			// this entry is NULL: skip it
			continue;
		}

		// first fetch the format string
		auto fmt_idx = format_string.GetVectorType() == VectorType::CONSTANT_VECTOR ? 0 : idx;
		auto format_string = format_data[fmt_idx].GetString();

		// now gather all the format arguments
		vector<duckdb_fmt::basic_format_arg<CTX>> format_args;
		vector<unsafe_unique_array<data_t>> string_args;

		for (idx_t col_idx = 1; col_idx < args.ColumnCount(); col_idx++) {
			auto &col = args.data[col_idx];
			idx_t arg_idx = col.GetVectorType() == VectorType::CONSTANT_VECTOR ? 0 : idx;
			switch (col.GetType().id()) {
			case LogicalTypeId::BOOLEAN: {
				auto arg_data = FlatVector::GetData<bool>(col);
				format_args.emplace_back(duckdb_fmt::internal::make_arg<CTX>(arg_data[arg_idx]));
				break;
			}
			case LogicalTypeId::TINYINT: {
				auto arg_data = FlatVector::GetData<int8_t>(col);
				format_args.emplace_back(duckdb_fmt::internal::make_arg<CTX>(arg_data[arg_idx]));
				break;
			}
			case LogicalTypeId::SMALLINT: {
				auto arg_data = FlatVector::GetData<int16_t>(col);
				format_args.emplace_back(duckdb_fmt::internal::make_arg<CTX>(arg_data[arg_idx]));
				break;
			}
			case LogicalTypeId::INTEGER: {
				auto arg_data = FlatVector::GetData<int32_t>(col);
				format_args.emplace_back(duckdb_fmt::internal::make_arg<CTX>(arg_data[arg_idx]));
				break;
			}
			case LogicalTypeId::BIGINT: {
				auto arg_data = FlatVector::GetData<int64_t>(col);
				format_args.emplace_back(duckdb_fmt::internal::make_arg<CTX>(arg_data[arg_idx]));
				break;
			}
			case LogicalTypeId::FLOAT: {
				auto arg_data = FlatVector::GetData<float>(col);
				format_args.emplace_back(duckdb_fmt::internal::make_arg<CTX>(arg_data[arg_idx]));
				break;
			}
			case LogicalTypeId::DOUBLE: {
				auto arg_data = FlatVector::GetData<double>(col);
				format_args.emplace_back(duckdb_fmt::internal::make_arg<CTX>(arg_data[arg_idx]));
				break;
			}
			case LogicalTypeId::VARCHAR: {
				auto arg_data = FlatVector::GetData<string_t>(col);
				auto string_view =
				    duckdb_fmt::basic_string_view<char>(arg_data[arg_idx].GetData(), arg_data[arg_idx].GetSize());
				format_args.emplace_back(duckdb_fmt::internal::make_arg<CTX>(string_view));
				break;
			}
			default:
				throw InternalException("Unexpected type for printf format");
			}
		}
		// finally actually perform the format
		string dynamic_result = FORMAT_FUN::template OP<CTX>(format_string.c_str(), format_args);
		result_data[idx] = StringVector::AddString(result, dynamic_result);
	}
}

ScalarFunction PrintfFun::GetFunction() {
	// duckdb_fmt::printf_context, duckdb_fmt::vsprintf
	ScalarFunction printf_fun({LogicalType::VARCHAR}, LogicalType::VARCHAR,
	                          PrintfFunction<FMTPrintf, duckdb_fmt::printf_context>, BindPrintfFunction);
	printf_fun.varargs = LogicalType::ANY;
	return printf_fun;
}

ScalarFunction FormatFun::GetFunction() {
	// duckdb_fmt::format_context, duckdb_fmt::vformat
	ScalarFunction format_fun({LogicalType::VARCHAR}, LogicalType::VARCHAR,
	                          PrintfFunction<FMTFormat, duckdb_fmt::format_context>, BindPrintfFunction);
	format_fun.varargs = LogicalType::ANY;
	return format_fun;
}

} // namespace duckdb





#include <string.h>
#include <ctype.h>

namespace duckdb {

static string_t RepeatScalarFunction(const string_t &str, const int64_t cnt, vector<char> &result) {
	// Get information about the repeated string
	auto input_str = str.GetData();
	auto size_str = str.GetSize();

	//  Reuse the buffer
	result.clear();
	for (auto remaining = cnt; remaining-- > 0;) {
		result.insert(result.end(), input_str, input_str + size_str);
	}

	return string_t(result.data(), result.size());
}

static void RepeatFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &str_vector = args.data[0];
	auto &cnt_vector = args.data[1];

	vector<char> buffer;
	BinaryExecutor::Execute<string_t, int64_t, string_t>(
	    str_vector, cnt_vector, result, args.size(), [&](string_t str, int64_t cnt) {
		    return StringVector::AddString(result, RepeatScalarFunction(str, cnt, buffer));
	    });
}

ScalarFunction RepeatFun::GetFunction() {
	return ScalarFunction({LogicalType::VARCHAR, LogicalType::BIGINT}, LogicalType::VARCHAR, RepeatFunction);
}

} // namespace duckdb






#include <string.h>
#include <ctype.h>
#include <unordered_map>

namespace duckdb {

static idx_t NextNeedle(const char *input_haystack, idx_t size_haystack, const char *input_needle,
                        const idx_t size_needle) {
	// Needle needs something to proceed
	if (size_needle > 0) {
		// Haystack should be bigger or equal size to the needle
		for (idx_t string_position = 0; (size_haystack - string_position) >= size_needle; ++string_position) {
			// Compare Needle to the Haystack
			if ((memcmp(input_haystack + string_position, input_needle, size_needle) == 0)) {
				return string_position;
			}
		}
	}
	// Did not find the needle
	return size_haystack;
}

static string_t ReplaceScalarFunction(const string_t &haystack, const string_t &needle, const string_t &thread,
                                      vector<char> &result) {
	// Get information about the needle, the haystack and the "thread"
	auto input_haystack = haystack.GetData();
	auto size_haystack = haystack.GetSize();

	auto input_needle = needle.GetData();
	auto size_needle = needle.GetSize();

	auto input_thread = thread.GetData();
	auto size_thread = thread.GetSize();

	//  Reuse the buffer
	result.clear();

	for (;;) {
		//  Append the non-matching characters
		auto string_position = NextNeedle(input_haystack, size_haystack, input_needle, size_needle);
		result.insert(result.end(), input_haystack, input_haystack + string_position);
		input_haystack += string_position;
		size_haystack -= string_position;

		//  Stop when we have read the entire haystack
		if (size_haystack == 0) {
			break;
		}

		//  Replace the matching characters
		result.insert(result.end(), input_thread, input_thread + size_thread);
		input_haystack += size_needle;
		size_haystack -= size_needle;
	}

	return string_t(result.data(), result.size());
}

static void ReplaceFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &haystack_vector = args.data[0];
	auto &needle_vector = args.data[1];
	auto &thread_vector = args.data[2];

	vector<char> buffer;
	TernaryExecutor::Execute<string_t, string_t, string_t, string_t>(
	    haystack_vector, needle_vector, thread_vector, result, args.size(),
	    [&](string_t input_string, string_t needle_string, string_t thread_string) {
		    return StringVector::AddString(result,
		                                   ReplaceScalarFunction(input_string, needle_string, thread_string, buffer));
	    });
}

ScalarFunction ReplaceFun::GetFunction() {
	return ScalarFunction({LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::VARCHAR,
	                      ReplaceFunction);
}

} // namespace duckdb







#include <string.h>

namespace duckdb {

//! Fast ASCII string reverse, returns false if the input data is not ascii
static bool StrReverseASCII(const char *input, idx_t n, char *output) {
	for (idx_t i = 0; i < n; i++) {
		if (input[i] & 0x80) {
			// non-ascii character
			return false;
		}
		output[n - i - 1] = input[i];
	}
	return true;
}

//! Unicode string reverse using grapheme breakers
static void StrReverseUnicode(const char *input, idx_t n, char *output) {
	utf8proc_grapheme_callback(input, n, [&](size_t start, size_t end) {
		memcpy(output + n - end, input + start, end - start);
		return true;
	});
}

struct ReverseOperator {
	template <class INPUT_TYPE, class RESULT_TYPE>
	static RESULT_TYPE Operation(INPUT_TYPE input, Vector &result) {
		auto input_data = input.GetData();
		auto input_length = input.GetSize();

		auto target = StringVector::EmptyString(result, input_length);
		auto target_data = target.GetDataWriteable();
		if (!StrReverseASCII(input_data, input_length, target_data)) {
			StrReverseUnicode(input_data, input_length, target_data);
		}
		target.Finalize();
		return target;
	}
};

static void ReverseFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	UnaryExecutor::ExecuteString<string_t, string_t, ReverseOperator>(args.data[0], result, args.size());
}

ScalarFunction ReverseFun::GetFunction() {
	return ScalarFunction("reverse", {LogicalType::VARCHAR}, LogicalType::VARCHAR, ReverseFunction);
}

} // namespace duckdb






namespace duckdb {

static bool StartsWith(const unsigned char *haystack, idx_t haystack_size, const unsigned char *needle,
                       idx_t needle_size) {
	D_ASSERT(needle_size > 0);
	if (needle_size > haystack_size) {
		// needle is bigger than haystack: haystack cannot start with needle
		return false;
	}
	return memcmp(haystack, needle, needle_size) == 0;
}

static bool StartsWith(const string_t &haystack_s, const string_t &needle_s) {
	auto haystack = (const unsigned char *)haystack_s.GetData();
	auto haystack_size = haystack_s.GetSize();
	auto needle = (const unsigned char *)needle_s.GetData();
	auto needle_size = needle_s.GetSize();
	if (needle_size == 0) {
		// empty needle: always true
		return true;
	}
	return StartsWith(haystack, haystack_size, needle, needle_size);
}

struct StartsWithOperator {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA left, TB right) {
		return StartsWith(left, right);
	}
};

ScalarFunction StartsWithOperatorFun::GetFunction() {
	return ScalarFunction({LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::BOOLEAN,
	                      ScalarFunction::BinaryFunction<string_t, string_t, bool, StartsWithOperator>);
}

} // namespace duckdb









namespace duckdb {

struct StringSplitInput {
	StringSplitInput(Vector &result_list, Vector &result_child, idx_t offset)
	    : result_list(result_list), result_child(result_child), offset(offset) {
	}

	Vector &result_list;
	Vector &result_child;
	idx_t offset;

	void AddSplit(const char *split_data, idx_t split_size, idx_t list_idx) {
		auto list_entry = offset + list_idx;
		if (list_entry >= ListVector::GetListCapacity(result_list)) {
			ListVector::SetListSize(result_list, offset + list_idx);
			ListVector::Reserve(result_list, ListVector::GetListCapacity(result_list) * 2);
		}
		FlatVector::GetData<string_t>(result_child)[list_entry] =
		    StringVector::AddString(result_child, split_data, split_size);
	}
};

struct RegularStringSplit {
	static idx_t Find(const char *input_data, idx_t input_size, const char *delim_data, idx_t delim_size,
	                  idx_t &match_size, void *data) {
		match_size = delim_size;
		if (delim_size == 0) {
			return 0;
		}
		return ContainsFun::Find((const unsigned char *)input_data, input_size, (const unsigned char *)delim_data,
		                         delim_size);
	}
};

struct ConstantRegexpStringSplit {
	static idx_t Find(const char *input_data, idx_t input_size, const char *delim_data, idx_t delim_size,
	                  idx_t &match_size, void *data) {
		D_ASSERT(data);
		auto regex = (duckdb_re2::RE2 *)data;
		duckdb_re2::StringPiece match;
		if (!regex->Match(duckdb_re2::StringPiece(input_data, input_size), 0, input_size, RE2::UNANCHORED, &match, 1)) {
			return DConstants::INVALID_INDEX;
		}
		match_size = match.size();
		return match.data() - input_data;
	}
};

struct RegexpStringSplit {
	static idx_t Find(const char *input_data, idx_t input_size, const char *delim_data, idx_t delim_size,
	                  idx_t &match_size, void *data) {
		duckdb_re2::RE2 regex(duckdb_re2::StringPiece(delim_data, delim_size));
		if (!regex.ok()) {
			throw InvalidInputException(regex.error());
		}
		return ConstantRegexpStringSplit::Find(input_data, input_size, delim_data, delim_size, match_size, &regex);
	}
};

struct StringSplitter {
	template <class OP>
	static idx_t Split(string_t input, string_t delim, StringSplitInput &state, void *data) {
		auto input_data = input.GetData();
		auto input_size = input.GetSize();
		auto delim_data = delim.GetData();
		auto delim_size = delim.GetSize();
		idx_t list_idx = 0;
		while (input_size > 0) {
			idx_t match_size = 0;
			auto pos = OP::Find(input_data, input_size, delim_data, delim_size, match_size, data);
			if (pos > input_size) {
				break;
			}
			if (match_size == 0 && pos == 0) {
				// special case: 0 length match and pos is 0
				// move to the next character
				for (pos++; pos < input_size; pos++) {
					if (LengthFun::IsCharacter(input_data[pos])) {
						break;
					}
				}
				if (pos == input_size) {
					break;
				}
			}
			D_ASSERT(input_size >= pos + match_size);
			state.AddSplit(input_data, pos, list_idx);

			list_idx++;
			input_data += (pos + match_size);
			input_size -= (pos + match_size);
		}
		state.AddSplit(input_data, input_size, list_idx);
		list_idx++;
		return list_idx;
	}
};

template <class OP>
static void StringSplitExecutor(DataChunk &args, ExpressionState &state, Vector &result, void *data = nullptr) {
	UnifiedVectorFormat input_data;
	args.data[0].ToUnifiedFormat(args.size(), input_data);
	auto inputs = (string_t *)input_data.data;

	UnifiedVectorFormat delim_data;
	args.data[1].ToUnifiedFormat(args.size(), delim_data);
	auto delims = (string_t *)delim_data.data;

	D_ASSERT(result.GetType().id() == LogicalTypeId::LIST);

	result.SetVectorType(VectorType::FLAT_VECTOR);
	ListVector::SetListSize(result, 0);

	auto list_struct_data = FlatVector::GetData<list_entry_t>(result);

	// count all the splits and set up the list entries
	auto &child_entry = ListVector::GetEntry(result);
	auto &result_mask = FlatVector::Validity(result);
	idx_t total_splits = 0;
	for (idx_t i = 0; i < args.size(); i++) {
		auto input_idx = input_data.sel->get_index(i);
		auto delim_idx = delim_data.sel->get_index(i);
		if (!input_data.validity.RowIsValid(input_idx)) {
			result_mask.SetInvalid(i);
			continue;
		}
		StringSplitInput split_input(result, child_entry, total_splits);
		if (!delim_data.validity.RowIsValid(delim_idx)) {
			// delim is NULL: copy the complete entry
			split_input.AddSplit(inputs[input_idx].GetData(), inputs[input_idx].GetSize(), 0);
			list_struct_data[i].length = 1;
			list_struct_data[i].offset = total_splits;
			total_splits++;
			continue;
		}
		auto list_length = StringSplitter::Split<OP>(inputs[input_idx], delims[delim_idx], split_input, data);
		list_struct_data[i].length = list_length;
		list_struct_data[i].offset = total_splits;
		total_splits += list_length;
	}
	ListVector::SetListSize(result, total_splits);
	D_ASSERT(ListVector::GetListSize(result) == total_splits);

	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

static void StringSplitFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	StringSplitExecutor<RegularStringSplit>(args, state, result, nullptr);
}

static void StringSplitRegexFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &info = func_expr.bind_info->Cast<RegexpMatchesBindData>();
	if (info.constant_pattern) {
		// fast path: pre-compiled regex
		auto &lstate = ExecuteFunctionState::GetFunctionState(state)->Cast<RegexLocalState>();
		StringSplitExecutor<ConstantRegexpStringSplit>(args, state, result, &lstate.constant_pattern);
	} else {
		// slow path: have to re-compile regex for every row
		StringSplitExecutor<RegexpStringSplit>(args, state, result);
	}
}

ScalarFunction StringSplitFun::GetFunction() {
	auto varchar_list_type = LogicalType::LIST(LogicalType::VARCHAR);

	ScalarFunction string_split({LogicalType::VARCHAR, LogicalType::VARCHAR}, varchar_list_type, StringSplitFunction);
	string_split.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	return string_split;
}

ScalarFunctionSet StringSplitRegexFun::GetFunctions() {
	auto varchar_list_type = LogicalType::LIST(LogicalType::VARCHAR);
	ScalarFunctionSet regexp_split;
	ScalarFunction regex_fun({LogicalType::VARCHAR, LogicalType::VARCHAR}, varchar_list_type, StringSplitRegexFunction,
	                         RegexpMatchesBind, nullptr, nullptr, RegexInitLocalState, LogicalType::INVALID,
	                         FunctionSideEffects::NO_SIDE_EFFECTS, FunctionNullHandling::SPECIAL_HANDLING);
	regexp_split.AddFunction(regex_fun);
	// regexp options
	regex_fun.arguments.emplace_back(LogicalType::VARCHAR);
	regexp_split.AddFunction(regex_fun);
	return regexp_split;
}

} // namespace duckdb








#include <string.h>
#include <ctype.h>
#include <unordered_map>
#include <unordered_set>

namespace duckdb {

static string_t TranslateScalarFunction(const string_t &haystack, const string_t &needle, const string_t &thread,
                                        vector<char> &result) {
	// Get information about the haystack, the needle and the "thread"
	auto input_haystack = haystack.GetData();
	auto size_haystack = haystack.GetSize();

	auto input_needle = needle.GetData();
	auto size_needle = needle.GetSize();

	auto input_thread = thread.GetData();
	auto size_thread = thread.GetSize();

	// Reuse the buffer
	result.clear();
	result.reserve(size_haystack);

	idx_t i = 0, j = 0;
	int sz = 0, c_sz = 0;

	// Character to be replaced
	unordered_map<int32_t, int32_t> to_replace;
	while (i < size_needle && j < size_thread) {
		auto codepoint_needle = Utf8Proc::UTF8ToCodepoint(input_needle, sz);
		input_needle += sz;
		i += sz;
		auto codepoint_thread = Utf8Proc::UTF8ToCodepoint(input_thread, sz);
		input_thread += sz;
		j += sz;
		// Ignore unicode character that is existed in to_replace
		if (to_replace.count(codepoint_needle) == 0) {
			to_replace[codepoint_needle] = codepoint_thread;
		}
	}

	// Character to be deleted
	unordered_set<int32_t> to_delete;
	while (i < size_needle) {
		auto codepoint_needle = Utf8Proc::UTF8ToCodepoint(input_needle, sz);
		input_needle += sz;
		i += sz;
		// Add unicode character that will be deleted
		if (to_replace.count(codepoint_needle) == 0) {
			to_delete.insert(codepoint_needle);
		}
	}

	char c[5] = {'\0', '\0', '\0', '\0', '\0'};
	for (i = 0; i < size_haystack; i += sz) {
		auto codepoint_haystack = Utf8Proc::UTF8ToCodepoint(input_haystack, sz);
		if (to_replace.count(codepoint_haystack) != 0) {
			Utf8Proc::CodepointToUtf8(to_replace[codepoint_haystack], c_sz, c);
			result.insert(result.end(), c, c + c_sz);
		} else if (to_delete.count(codepoint_haystack) == 0) {
			result.insert(result.end(), input_haystack, input_haystack + sz);
		}
		input_haystack += sz;
	}

	return string_t(result.data(), result.size());
}

static void TranslateFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &haystack_vector = args.data[0];
	auto &needle_vector = args.data[1];
	auto &thread_vector = args.data[2];

	vector<char> buffer;
	TernaryExecutor::Execute<string_t, string_t, string_t, string_t>(
	    haystack_vector, needle_vector, thread_vector, result, args.size(),
	    [&](string_t input_string, string_t needle_string, string_t thread_string) {
		    return StringVector::AddString(result,
		                                   TranslateScalarFunction(input_string, needle_string, thread_string, buffer));
	    });
}

ScalarFunction TranslateFun::GetFunction() {
	return ScalarFunction({LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::VARCHAR,
	                      TranslateFunction);
}

} // namespace duckdb







#include <string.h>

namespace duckdb {

template <bool LTRIM, bool RTRIM>
struct TrimOperator {
	template <class INPUT_TYPE, class RESULT_TYPE>
	static RESULT_TYPE Operation(INPUT_TYPE input, Vector &result) {
		auto data = input.GetData();
		auto size = input.GetSize();

		utf8proc_int32_t codepoint;
		auto str = reinterpret_cast<const utf8proc_uint8_t *>(data);

		// Find the first character that is not left trimmed
		idx_t begin = 0;
		if (LTRIM) {
			while (begin < size) {
				auto bytes = utf8proc_iterate(str + begin, size - begin, &codepoint);
				D_ASSERT(bytes > 0);
				if (utf8proc_category(codepoint) != UTF8PROC_CATEGORY_ZS) {
					break;
				}
				begin += bytes;
			}
		}

		// Find the last character that is not right trimmed
		idx_t end;
		if (RTRIM) {
			end = begin;
			for (auto next = begin; next < size;) {
				auto bytes = utf8proc_iterate(str + next, size - next, &codepoint);
				D_ASSERT(bytes > 0);
				next += bytes;
				if (utf8proc_category(codepoint) != UTF8PROC_CATEGORY_ZS) {
					end = next;
				}
			}
		} else {
			end = size;
		}

		// Copy the trimmed string
		auto target = StringVector::EmptyString(result, end - begin);
		auto output = target.GetDataWriteable();
		memcpy(output, data + begin, end - begin);

		target.Finalize();
		return target;
	}
};

template <bool LTRIM, bool RTRIM>
static void UnaryTrimFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	UnaryExecutor::ExecuteString<string_t, string_t, TrimOperator<LTRIM, RTRIM>>(args.data[0], result, args.size());
}

static void GetIgnoredCodepoints(string_t ignored, unordered_set<utf8proc_int32_t> &ignored_codepoints) {
	auto dataptr = (utf8proc_uint8_t *)ignored.GetData();
	auto size = ignored.GetSize();
	idx_t pos = 0;
	while (pos < size) {
		utf8proc_int32_t codepoint;
		pos += utf8proc_iterate(dataptr + pos, size - pos, &codepoint);
		ignored_codepoints.insert(codepoint);
	}
}

template <bool LTRIM, bool RTRIM>
static void BinaryTrimFunction(DataChunk &input, ExpressionState &state, Vector &result) {
	BinaryExecutor::Execute<string_t, string_t, string_t>(
	    input.data[0], input.data[1], result, input.size(), [&](string_t input, string_t ignored) {
		    auto data = input.GetData();
		    auto size = input.GetSize();

		    unordered_set<utf8proc_int32_t> ignored_codepoints;
		    GetIgnoredCodepoints(ignored, ignored_codepoints);

		    utf8proc_int32_t codepoint;
		    auto str = reinterpret_cast<const utf8proc_uint8_t *>(data);

		    // Find the first character that is not left trimmed
		    idx_t begin = 0;
		    if (LTRIM) {
			    while (begin < size) {
				    auto bytes = utf8proc_iterate(str + begin, size - begin, &codepoint);
				    if (ignored_codepoints.find(codepoint) == ignored_codepoints.end()) {
					    break;
				    }
				    begin += bytes;
			    }
		    }

		    // Find the last character that is not right trimmed
		    idx_t end;
		    if (RTRIM) {
			    end = begin;
			    for (auto next = begin; next < size;) {
				    auto bytes = utf8proc_iterate(str + next, size - next, &codepoint);
				    D_ASSERT(bytes > 0);
				    next += bytes;
				    if (ignored_codepoints.find(codepoint) == ignored_codepoints.end()) {
					    end = next;
				    }
			    }
		    } else {
			    end = size;
		    }

		    // Copy the trimmed string
		    auto target = StringVector::EmptyString(result, end - begin);
		    auto output = target.GetDataWriteable();
		    memcpy(output, data + begin, end - begin);

		    target.Finalize();
		    return target;
	    });
}

ScalarFunctionSet TrimFun::GetFunctions() {
	ScalarFunctionSet trim;
	trim.AddFunction(ScalarFunction({LogicalType::VARCHAR}, LogicalType::VARCHAR, UnaryTrimFunction<true, true>));

	trim.AddFunction(ScalarFunction({LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::VARCHAR,
	                                BinaryTrimFunction<true, true>));
	return trim;
}

ScalarFunctionSet LtrimFun::GetFunctions() {
	ScalarFunctionSet ltrim;
	ltrim.AddFunction(ScalarFunction({LogicalType::VARCHAR}, LogicalType::VARCHAR, UnaryTrimFunction<true, false>));
	ltrim.AddFunction(ScalarFunction({LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::VARCHAR,
	                                 BinaryTrimFunction<true, false>));
	return ltrim;
}

ScalarFunctionSet RtrimFun::GetFunctions() {
	ScalarFunctionSet rtrim;
	rtrim.AddFunction(ScalarFunction({LogicalType::VARCHAR}, LogicalType::VARCHAR, UnaryTrimFunction<false, true>));

	rtrim.AddFunction(ScalarFunction({LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::VARCHAR,
	                                 BinaryTrimFunction<false, true>));
	return rtrim;
}

} // namespace duckdb







#include <string.h>

namespace duckdb {

struct UnicodeOperator {
	template <class TA, class TR>
	static inline TR Operation(const TA &input) {
		auto str = reinterpret_cast<const utf8proc_uint8_t *>(input.GetData());
		auto len = input.GetSize();
		utf8proc_int32_t codepoint;
		(void)utf8proc_iterate(str, len, &codepoint);
		return codepoint;
	}
};

ScalarFunction UnicodeFun::GetFunction() {
	return ScalarFunction({LogicalType::VARCHAR}, LogicalType::INTEGER,
	                      ScalarFunction::UnaryFunction<string_t, int32_t, UnicodeOperator>);
}

} // namespace duckdb









namespace duckdb {

static void StructInsertFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &starting_vec = args.data[0];

	starting_vec.Verify(args.size());

	auto &starting_child_entries = StructVector::GetEntries(starting_vec);
	auto &result_child_entries = StructVector::GetEntries(result);

	// Assign the starting vector entries to the result vector
	for (size_t i = 0; i < starting_child_entries.size(); i++) {
		auto &starting_child = starting_child_entries[i];
		result_child_entries[i]->Reference(*starting_child);
	}

	// Assign the new entries to the result vector
	for (size_t i = 1; i < args.ColumnCount(); i++) {
		result_child_entries[starting_child_entries.size() + i - 1]->Reference(args.data[i]);
	}

	result.Verify(args.size());

	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

static unique_ptr<FunctionData> StructInsertBind(ClientContext &context, ScalarFunction &bound_function,
                                                 vector<unique_ptr<Expression>> &arguments) {
	case_insensitive_set_t name_collision_set;

	if (arguments.empty()) {
		throw Exception("Missing required arguments for struct_insert function.");
	}

	if (LogicalTypeId::STRUCT != arguments[0]->return_type.id()) {
		throw Exception("The first argument to struct_insert must be a STRUCT");
	}

	if (arguments.size() < 2) {
		throw Exception("Can't insert nothing into a struct");
	}

	child_list_t<LogicalType> new_struct_children;

	auto &existing_struct_children = StructType::GetChildTypes(arguments[0]->return_type);

	for (size_t i = 0; i < existing_struct_children.size(); i++) {
		auto &child = existing_struct_children[i];
		name_collision_set.insert(child.first);
		new_struct_children.push_back(make_pair(child.first, child.second));
	}

	// Loop through the additional arguments (name/value pairs)
	for (idx_t i = 1; i < arguments.size(); i++) {
		auto &child = arguments[i];
		if (child->alias.empty() && bound_function.name == "struct_insert") {
			throw BinderException("Need named argument for struct insert, e.g. STRUCT_PACK(a := b)");
		}
		if (name_collision_set.find(child->alias) != name_collision_set.end()) {
			throw BinderException("Duplicate struct entry name \"%s\"", child->alias);
		}
		name_collision_set.insert(child->alias);
		new_struct_children.push_back(make_pair(child->alias, arguments[i]->return_type));
	}

	// this is more for completeness reasons
	bound_function.return_type = LogicalType::STRUCT(new_struct_children);
	return make_uniq<VariableReturnBindData>(bound_function.return_type);
}

unique_ptr<BaseStatistics> StructInsertStats(ClientContext &context, FunctionStatisticsInput &input) {
	auto &child_stats = input.child_stats;
	auto &expr = input.expr;
	auto new_struct_stats = StructStats::CreateUnknown(expr.return_type);

	auto existing_count = StructType::GetChildCount(child_stats[0].GetType());
	auto existing_stats = StructStats::GetChildStats(child_stats[0]);
	for (idx_t i = 0; i < existing_count; i++) {
		StructStats::SetChildStats(new_struct_stats, i, existing_stats[i]);
	}
	auto new_count = StructType::GetChildCount(expr.return_type);
	auto offset = new_count - child_stats.size();
	for (idx_t i = 1; i < child_stats.size(); i++) {
		StructStats::SetChildStats(new_struct_stats, offset + i, child_stats[i]);
	}
	return new_struct_stats.ToUnique();
}

ScalarFunction StructInsertFun::GetFunction() {
	// the arguments and return types are actually set in the binder function
	ScalarFunction fun({}, LogicalTypeId::STRUCT, StructInsertFunction, StructInsertBind, nullptr, StructInsertStats);
	fun.varargs = LogicalType::ANY;
	fun.serialize = VariableReturnBindData::Serialize;
	fun.deserialize = VariableReturnBindData::Deserialize;
	return fun;
}

} // namespace duckdb









namespace duckdb {

static void StructPackFunction(DataChunk &args, ExpressionState &state, Vector &result) {
#ifdef DEBUG
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &info = func_expr.bind_info->Cast<VariableReturnBindData>();
	// this should never happen if the binder below is sane
	D_ASSERT(args.ColumnCount() == StructType::GetChildTypes(info.stype).size());
#endif
	bool all_const = true;
	auto &child_entries = StructVector::GetEntries(result);
	for (size_t i = 0; i < args.ColumnCount(); i++) {
		if (args.data[i].GetVectorType() != VectorType::CONSTANT_VECTOR) {
			all_const = false;
		}
		// same holds for this
		child_entries[i]->Reference(args.data[i]);
	}
	result.SetVectorType(all_const ? VectorType::CONSTANT_VECTOR : VectorType::FLAT_VECTOR);

	result.Verify(args.size());
}

static unique_ptr<FunctionData> StructPackBind(ClientContext &context, ScalarFunction &bound_function,
                                               vector<unique_ptr<Expression>> &arguments) {
	case_insensitive_set_t name_collision_set;

	// collect names and deconflict, construct return type
	if (arguments.empty()) {
		throw Exception("Can't pack nothing into a struct");
	}
	child_list_t<LogicalType> struct_children;
	for (idx_t i = 0; i < arguments.size(); i++) {
		auto &child = arguments[i];
		if (child->alias.empty() && bound_function.name == "struct_pack") {
			throw BinderException("Need named argument for struct pack, e.g. STRUCT_PACK(a := b)");
		}
		if (child->alias.empty() && bound_function.name == "row") {
			child->alias = "v" + std::to_string(i + 1);
		}
		if (name_collision_set.find(child->alias) != name_collision_set.end()) {
			throw BinderException("Duplicate struct entry name \"%s\"", child->alias);
		}
		name_collision_set.insert(child->alias);
		struct_children.push_back(make_pair(child->alias, arguments[i]->return_type));
	}

	// this is more for completeness reasons
	bound_function.return_type = LogicalType::STRUCT(struct_children);
	return make_uniq<VariableReturnBindData>(bound_function.return_type);
}

unique_ptr<BaseStatistics> StructPackStats(ClientContext &context, FunctionStatisticsInput &input) {
	auto &child_stats = input.child_stats;
	auto &expr = input.expr;
	auto struct_stats = StructStats::CreateUnknown(expr.return_type);
	for (idx_t i = 0; i < child_stats.size(); i++) {
		StructStats::SetChildStats(struct_stats, i, child_stats[i]);
	}
	return struct_stats.ToUnique();
}

ScalarFunction StructPackFun::GetFunction() {
	// the arguments and return types are actually set in the binder function
	ScalarFunction fun("struct_pack", {}, LogicalTypeId::STRUCT, StructPackFunction, StructPackBind, nullptr,
	                   StructPackStats);
	fun.varargs = LogicalType::ANY;
	fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	fun.serialize = VariableReturnBindData::Serialize;
	fun.deserialize = VariableReturnBindData::Deserialize;
	return fun;
}

} // namespace duckdb






namespace duckdb {

struct UnionExtractBindData : public FunctionData {
	UnionExtractBindData(string key, idx_t index, LogicalType type)
	    : key(std::move(key)), index(index), type(std::move(type)) {
	}

	string key;
	idx_t index;
	LogicalType type;

public:
	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<UnionExtractBindData>(key, index, type);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &other = (const UnionExtractBindData &)other_p;
		return key == other.key && index == other.index && type == other.type;
	}
};

static void UnionExtractFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &info = func_expr.bind_info->Cast<UnionExtractBindData>();

	// this should be guaranteed by the binder
	auto &vec = args.data[0];
	vec.Verify(args.size());

	D_ASSERT(info.index < UnionType::GetMemberCount(vec.GetType()));
	auto &member = UnionVector::GetMember(vec, info.index);
	result.Reference(member);
	result.Verify(args.size());
}

static unique_ptr<FunctionData> UnionExtractBind(ClientContext &context, ScalarFunction &bound_function,
                                                 vector<unique_ptr<Expression>> &arguments) {
	D_ASSERT(bound_function.arguments.size() == 2);
	if (arguments[0]->return_type.id() == LogicalTypeId::UNKNOWN) {
		throw ParameterNotResolvedException();
	}
	D_ASSERT(LogicalTypeId::UNION == arguments[0]->return_type.id());
	idx_t union_member_count = UnionType::GetMemberCount(arguments[0]->return_type);
	if (union_member_count == 0) {
		throw InternalException("Can't extract something from an empty union");
	}
	bound_function.arguments[0] = arguments[0]->return_type;

	auto &key_child = arguments[1];
	if (key_child->HasParameter()) {
		throw ParameterNotResolvedException();
	}

	if (key_child->return_type.id() != LogicalTypeId::VARCHAR || !key_child->IsFoldable()) {
		throw BinderException("Key name for union_extract needs to be a constant string");
	}
	Value key_val = ExpressionExecutor::EvaluateScalar(context, *key_child);
	D_ASSERT(key_val.type().id() == LogicalTypeId::VARCHAR);
	auto &key_str = StringValue::Get(key_val);
	if (key_val.IsNull() || key_str.empty()) {
		throw BinderException("Key name for union_extract needs to be neither NULL nor empty");
	}
	string key = StringUtil::Lower(key_str);

	LogicalType return_type;
	idx_t key_index = 0;
	bool found_key = false;

	for (size_t i = 0; i < union_member_count; i++) {
		auto &member_name = UnionType::GetMemberName(arguments[0]->return_type, i);
		if (StringUtil::Lower(member_name) == key) {
			found_key = true;
			key_index = i;
			return_type = UnionType::GetMemberType(arguments[0]->return_type, i);
			break;
		}
	}

	if (!found_key) {
		vector<string> candidates;
		candidates.reserve(union_member_count);
		for (idx_t i = 0; i < union_member_count; i++) {
			candidates.push_back(UnionType::GetMemberName(arguments[0]->return_type, i));
		}
		auto closest_settings = StringUtil::TopNLevenshtein(candidates, key);
		auto message = StringUtil::CandidatesMessage(closest_settings, "Candidate Entries");
		throw BinderException("Could not find key \"%s\" in union\n%s", key, message);
	}

	bound_function.return_type = return_type;
	return make_uniq<UnionExtractBindData>(key, key_index, return_type);
}

ScalarFunction UnionExtractFun::GetFunction() {
	// the arguments and return types are actually set in the binder function
	return ScalarFunction({LogicalTypeId::UNION, LogicalType::VARCHAR}, LogicalType::ANY, UnionExtractFunction,
	                      UnionExtractBind, nullptr, nullptr);
}

} // namespace duckdb






namespace duckdb {

static unique_ptr<FunctionData> UnionTagBind(ClientContext &context, ScalarFunction &bound_function,
                                             vector<unique_ptr<Expression>> &arguments) {

	if (arguments.empty()) {
		throw BinderException("Missing required arguments for union_tag function.");
	}

	if (LogicalTypeId::UNKNOWN == arguments[0]->return_type.id()) {
		throw ParameterNotResolvedException();
	}

	if (LogicalTypeId::UNION != arguments[0]->return_type.id()) {
		throw BinderException("First argument to union_tag function must be a union type.");
	}

	if (arguments.size() > 1) {
		throw BinderException("Too many arguments, union_tag takes at most one argument.");
	}

	auto member_count = UnionType::GetMemberCount(arguments[0]->return_type);
	if (member_count == 0) {
		// this should never happen, empty unions are not allowed
		throw InternalException("Can't get tags from an empty union");
	}

	bound_function.arguments[0] = arguments[0]->return_type;

	auto varchar_vector = Vector(LogicalType::VARCHAR, member_count);
	for (idx_t i = 0; i < member_count; i++) {
		auto str = string_t(UnionType::GetMemberName(arguments[0]->return_type, i));
		FlatVector::GetData<string_t>(varchar_vector)[i] =
		    str.IsInlined() ? str : StringVector::AddString(varchar_vector, str);
	}
	auto enum_type = LogicalType::ENUM("", varchar_vector, member_count);
	bound_function.return_type = enum_type;

	return nullptr;
}

static void UnionTagFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(result.GetType().id() == LogicalTypeId::ENUM);
	result.Reinterpret(UnionVector::GetTags(args.data[0]));
}

ScalarFunction UnionTagFun::GetFunction() {
	return ScalarFunction({LogicalTypeId::UNION}, LogicalTypeId::ANY, UnionTagFunction, UnionTagBind, nullptr,
	                      nullptr); // TODO: Statistics?
}

} // namespace duckdb







namespace duckdb {

struct UnionValueBindData : public FunctionData {
	UnionValueBindData() {
	}

public:
	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<UnionValueBindData>();
	}
	bool Equals(const FunctionData &other_p) const override {
		return true;
	}
};

static void UnionValueFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	// Assign the new entries to the result vector
	UnionVector::GetMember(result, 0).Reference(args.data[0]);

	// Set the result tag vector to a constant value
	auto &tag_vector = UnionVector::GetTags(result);
	tag_vector.SetVectorType(VectorType::CONSTANT_VECTOR);
	ConstantVector::GetData<union_tag_t>(tag_vector)[0] = 0;

	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}

	result.Verify(args.size());
}

static unique_ptr<FunctionData> UnionValueBind(ClientContext &context, ScalarFunction &bound_function,
                                               vector<unique_ptr<Expression>> &arguments) {

	if (arguments.size() != 1) {
		throw BinderException("union_value takes exactly one argument");
	}
	auto &child = arguments[0];

	if (child->alias.empty()) {
		throw BinderException("Need named argument for union tag, e.g. UNION_VALUE(a := b)");
	}

	child_list_t<LogicalType> union_members;

	union_members.push_back(make_pair(child->alias, child->return_type));

	bound_function.return_type = LogicalType::UNION(std::move(union_members));
	return make_uniq<VariableReturnBindData>(bound_function.return_type);
}

ScalarFunction UnionValueFun::GetFunction() {
	ScalarFunction fun("union_value", {}, LogicalTypeId::UNION, UnionValueFunction, UnionValueBind, nullptr, nullptr);
	fun.varargs = LogicalType::ANY;
	fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	fun.serialize = VariableReturnBindData::Serialize;
	fun.deserialize = VariableReturnBindData::Deserialize;
	return fun;
}

} // namespace duckdb





namespace duckdb {

AdaptiveFilter::AdaptiveFilter(const Expression &expr)
    : iteration_count(0), observe_interval(10), execute_interval(20), warmup(true) {
	auto &conj_expr = (const BoundConjunctionExpression &)expr;
	D_ASSERT(conj_expr.children.size() > 1);
	for (idx_t idx = 0; idx < conj_expr.children.size(); idx++) {
		permutation.push_back(idx);
		if (idx != conj_expr.children.size() - 1) {
			swap_likeliness.push_back(100);
		}
	}
	right_random_border = 100 * (conj_expr.children.size() - 1);
}

AdaptiveFilter::AdaptiveFilter(TableFilterSet *table_filters)
    : iteration_count(0), observe_interval(10), execute_interval(20), warmup(true) {
	for (auto &table_filter : table_filters->filters) {
		permutation.push_back(table_filter.first);
		swap_likeliness.push_back(100);
	}
	swap_likeliness.pop_back();
	right_random_border = 100 * (table_filters->filters.size() - 1);
}
void AdaptiveFilter::AdaptRuntimeStatistics(double duration) {
	iteration_count++;
	runtime_sum += duration;

	if (!warmup) {
		// the last swap was observed
		if (observe && iteration_count == observe_interval) {
			// keep swap if runtime decreased, else reverse swap
			if (prev_mean - (runtime_sum / iteration_count) <= 0) {
				// reverse swap because runtime didn't decrease
				std::swap(permutation[swap_idx], permutation[swap_idx + 1]);

				// decrease swap likeliness, but make sure there is always a small likeliness left
				if (swap_likeliness[swap_idx] > 1) {
					swap_likeliness[swap_idx] /= 2;
				}
			} else {
				// keep swap because runtime decreased, reset likeliness
				swap_likeliness[swap_idx] = 100;
			}
			observe = false;

			// reset values
			iteration_count = 0;
			runtime_sum = 0.0;
		} else if (!observe && iteration_count == execute_interval) {
			// save old mean to evaluate swap
			prev_mean = runtime_sum / iteration_count;

			// get swap index and swap likeliness
			std::uniform_int_distribution<int> distribution(1, right_random_border); // a <= i <= b
			idx_t random_number = distribution(generator) - 1;

			swap_idx = random_number / 100;                    // index to be swapped
			idx_t likeliness = random_number - 100 * swap_idx; // random number between [0, 100)

			// check if swap is going to happen
			if (swap_likeliness[swap_idx] > likeliness) { // always true for the first swap of an index
				// swap
				std::swap(permutation[swap_idx], permutation[swap_idx + 1]);

				// observe whether swap will be applied
				observe = true;
			}

			// reset values
			iteration_count = 0;
			runtime_sum = 0.0;
		}
	} else {
		if (iteration_count == 5) {
			// initially set all values
			iteration_count = 0;
			runtime_sum = 0.0;
			observe = false;
			warmup = false;
		}
	}
}

} // namespace duckdb















#include <cmath>

namespace duckdb {

using ValidityBytes = TupleDataLayout::ValidityBytes;

GroupedAggregateHashTable::GroupedAggregateHashTable(ClientContext &context, Allocator &allocator,
                                                     vector<LogicalType> group_types, vector<LogicalType> payload_types,
                                                     const vector<BoundAggregateExpression *> &bindings,
                                                     HtEntryType entry_type, idx_t initial_capacity)
    : GroupedAggregateHashTable(context, allocator, std::move(group_types), std::move(payload_types),
                                AggregateObject::CreateAggregateObjects(bindings), entry_type, initial_capacity) {
}

GroupedAggregateHashTable::GroupedAggregateHashTable(ClientContext &context, Allocator &allocator,
                                                     vector<LogicalType> group_types)
    : GroupedAggregateHashTable(context, allocator, std::move(group_types), {}, vector<AggregateObject>()) {
}

AggregateHTAppendState::AggregateHTAppendState()
    : ht_offsets(LogicalTypeId::BIGINT), hash_salts(LogicalTypeId::SMALLINT),
      group_compare_vector(STANDARD_VECTOR_SIZE), no_match_vector(STANDARD_VECTOR_SIZE),
      empty_vector(STANDARD_VECTOR_SIZE), new_groups(STANDARD_VECTOR_SIZE), addresses(LogicalType::POINTER),
      chunk_state_initialized(false) {
}

GroupedAggregateHashTable::GroupedAggregateHashTable(ClientContext &context, Allocator &allocator,
                                                     vector<LogicalType> group_types_p,
                                                     vector<LogicalType> payload_types_p,
                                                     vector<AggregateObject> aggregate_objects_p,
                                                     HtEntryType entry_type, idx_t initial_capacity)
    : BaseAggregateHashTable(context, allocator, aggregate_objects_p, std::move(payload_types_p)),
      entry_type(entry_type), capacity(0), is_finalized(false),
      aggregate_allocator(make_shared<ArenaAllocator>(allocator)) {
	// Append hash column to the end and initialise the row layout
	group_types_p.emplace_back(LogicalType::HASH);
	layout.Initialize(std::move(group_types_p), std::move(aggregate_objects_p));
	tuple_size = layout.GetRowWidth();
	tuples_per_block = Storage::BLOCK_SIZE / tuple_size;

	// HT layout
	hash_offset = layout.GetOffsets()[layout.ColumnCount() - 1];
	data_collection = make_uniq<TupleDataCollection>(buffer_manager, layout);
	data_collection->InitializeAppend(td_pin_state, TupleDataPinProperties::KEEP_EVERYTHING_PINNED);

	hashes_hdl = buffer_manager.Allocate(Storage::BLOCK_SIZE);
	hashes_hdl_ptr = hashes_hdl.Ptr();

	switch (entry_type) {
	case HtEntryType::HT_WIDTH_64: {
		hash_prefix_shift = (HASH_WIDTH - sizeof(aggr_ht_entry_64::salt)) * 8;
		Resize<aggr_ht_entry_64>(initial_capacity);
		break;
	}
	case HtEntryType::HT_WIDTH_32: {
		hash_prefix_shift = (HASH_WIDTH - sizeof(aggr_ht_entry_32::salt)) * 8;
		Resize<aggr_ht_entry_32>(initial_capacity);
		break;
	}
	default:
		throw InternalException("Unknown HT entry width");
	}

	predicates.resize(layout.ColumnCount() - 1, ExpressionType::COMPARE_EQUAL);
}

GroupedAggregateHashTable::~GroupedAggregateHashTable() {
	Destroy();
}

void GroupedAggregateHashTable::Destroy() {
	if (data_collection->Count() == 0) {
		return;
	}

	// Check if there is an aggregate with a destructor
	bool has_destructor = false;
	for (auto &aggr : layout.GetAggregates()) {
		if (aggr.function.destructor) {
			has_destructor = true;
		}
	}
	if (!has_destructor) {
		return;
	}

	// There are aggregates with destructors: Call the destructor for each of the aggregates
	RowOperationsState state(aggregate_allocator->GetAllocator());
	TupleDataChunkIterator iterator(*data_collection, TupleDataPinProperties::DESTROY_AFTER_DONE, false);
	auto &row_locations = iterator.GetChunkState().row_locations;
	do {
		RowOperations::DestroyStates(state, layout, row_locations, iterator.GetCurrentChunkCount());
	} while (iterator.Next());
	data_collection->Reset();
}

template <class ENTRY>
void GroupedAggregateHashTable::VerifyInternal() {
	auto hashes_ptr = (ENTRY *)hashes_hdl_ptr;
	idx_t count = 0;
	for (idx_t i = 0; i < capacity; i++) {
		if (hashes_ptr[i].page_nr > 0) {
			D_ASSERT(hashes_ptr[i].page_offset < tuples_per_block);
			D_ASSERT(hashes_ptr[i].page_nr <= payload_hds_ptrs.size());
			auto ptr = payload_hds_ptrs[hashes_ptr[i].page_nr - 1] + ((hashes_ptr[i].page_offset) * tuple_size);
			auto hash = Load<hash_t>(ptr + hash_offset);
			D_ASSERT((hashes_ptr[i].salt) == (hash >> hash_prefix_shift));

			count++;
		}
	}
	(void)count;
	D_ASSERT(count == Count());
}

idx_t GroupedAggregateHashTable::InitialCapacity() {
	return STANDARD_VECTOR_SIZE * 2ULL;
}

idx_t GroupedAggregateHashTable::GetMaxCapacity(HtEntryType entry_type, idx_t tuple_size) {
	idx_t max_pages;
	idx_t max_tuples;

	switch (entry_type) {
	case HtEntryType::HT_WIDTH_32:
		max_pages = NumericLimits<uint8_t>::Maximum();
		max_tuples = NumericLimits<uint16_t>::Maximum();
		break;
	case HtEntryType::HT_WIDTH_64:
		max_pages = NumericLimits<uint32_t>::Maximum();
		max_tuples = NumericLimits<uint16_t>::Maximum();
		break;
	default:
		throw InternalException("Unsupported hash table width");
	}

	return max_pages * MinValue(max_tuples, (idx_t)Storage::BLOCK_SIZE / tuple_size);
}

idx_t GroupedAggregateHashTable::MaxCapacity() {
	return GetMaxCapacity(entry_type, tuple_size);
}

void GroupedAggregateHashTable::Verify() {
#ifdef DEBUG
	switch (entry_type) {
	case HtEntryType::HT_WIDTH_32:
		VerifyInternal<aggr_ht_entry_32>();
		break;
	case HtEntryType::HT_WIDTH_64:
		VerifyInternal<aggr_ht_entry_64>();
		break;
	}
#endif
}

template <class ENTRY>
void GroupedAggregateHashTable::Resize(idx_t size) {
	D_ASSERT(!is_finalized);
	D_ASSERT(size >= STANDARD_VECTOR_SIZE);
	D_ASSERT(IsPowerOfTwo(size));

	if (size < capacity) {
		throw InternalException("Cannot downsize a hash table!");
	}
	capacity = size;

	bitmask = capacity - 1;
	const auto byte_size = capacity * sizeof(ENTRY);
	if (byte_size > (idx_t)Storage::BLOCK_SIZE) {
		hashes_hdl = buffer_manager.Allocate(byte_size);
		hashes_hdl_ptr = hashes_hdl.Ptr();
	}
	memset(hashes_hdl_ptr, 0, byte_size);

	if (Count() != 0) {
		D_ASSERT(!payload_hds_ptrs.empty());
		auto hashes_arr = (ENTRY *)hashes_hdl_ptr;

		idx_t block_id = 0;
		auto block_pointer = payload_hds_ptrs[block_id];
		auto block_end = block_pointer + tuples_per_block * tuple_size;

		TupleDataChunkIterator iterator(*data_collection, TupleDataPinProperties::ALREADY_PINNED, false);
		const auto row_locations = iterator.GetRowLocations();
		do {
			for (idx_t i = 0; i < iterator.GetCurrentChunkCount(); i++) {
				const auto &row_location = row_locations[i];
				if (row_location > block_end || row_location < block_pointer) {
					block_id++;
					D_ASSERT(block_id < payload_hds_ptrs.size());
					block_pointer = payload_hds_ptrs[block_id];
					block_end = block_pointer + tuples_per_block * tuple_size;
				}
				D_ASSERT(row_location >= block_pointer && row_location < block_end);
				D_ASSERT((row_location - block_pointer) % tuple_size == 0);

				const auto hash = Load<hash_t>(row_location + hash_offset);
				D_ASSERT((hash & bitmask) == (hash % capacity));
				D_ASSERT(hash >> hash_prefix_shift <= NumericLimits<uint16_t>::Maximum());

				auto entry_idx = (idx_t)hash & bitmask;
				while (hashes_arr[entry_idx].page_nr > 0) {
					entry_idx++;
					if (entry_idx >= capacity) {
						entry_idx = 0;
					}
				}

				auto &ht_entry = hashes_arr[entry_idx];
				D_ASSERT(!ht_entry.page_nr);
				ht_entry.salt = hash >> hash_prefix_shift;
				ht_entry.page_nr = block_id + 1;
				ht_entry.page_offset = (row_location - block_pointer) / tuple_size;
			}
		} while (iterator.Next());
	}

	Verify();
}

idx_t GroupedAggregateHashTable::AddChunk(AggregateHTAppendState &state, DataChunk &groups, DataChunk &payload,
                                          AggregateType filter) {
	unsafe_vector<idx_t> aggregate_filter;

	auto &aggregates = layout.GetAggregates();
	for (idx_t i = 0; i < aggregates.size(); i++) {
		auto &aggregate = aggregates[i];
		if (aggregate.aggr_type == filter) {
			aggregate_filter.push_back(i);
		}
	}
	return AddChunk(state, groups, payload, aggregate_filter);
}

idx_t GroupedAggregateHashTable::AddChunk(AggregateHTAppendState &state, DataChunk &groups, DataChunk &payload,
                                          const unsafe_vector<idx_t> &filter) {
	Vector hashes(LogicalType::HASH);
	groups.Hash(hashes);

	return AddChunk(state, groups, hashes, payload, filter);
}

idx_t GroupedAggregateHashTable::AddChunk(AggregateHTAppendState &state, DataChunk &groups, Vector &group_hashes,
                                          DataChunk &payload, const unsafe_vector<idx_t> &filter) {
	D_ASSERT(!is_finalized);
	if (groups.size() == 0) {
		return 0;
	}

#ifdef DEBUG
	D_ASSERT(groups.ColumnCount() + 1 == layout.ColumnCount());
	for (idx_t i = 0; i < groups.ColumnCount(); i++) {
		D_ASSERT(groups.GetTypes()[i] == layout.GetTypes()[i]);
	}
#endif

	auto new_group_count = FindOrCreateGroups(state, groups, group_hashes, state.addresses, state.new_groups);
	VectorOperations::AddInPlace(state.addresses, layout.GetAggrOffset(), payload.size());

	// Now every cell has an entry, update the aggregates
	auto &aggregates = layout.GetAggregates();
	idx_t filter_idx = 0;
	idx_t payload_idx = 0;
	RowOperationsState row_state(aggregate_allocator->GetAllocator());
	for (idx_t i = 0; i < aggregates.size(); i++) {
		auto &aggr = aggregates[i];
		if (filter_idx >= filter.size() || i < filter[filter_idx]) {
			// Skip all the aggregates that are not in the filter
			payload_idx += aggr.child_count;
			VectorOperations::AddInPlace(state.addresses, aggr.payload_size, payload.size());
			continue;
		}
		D_ASSERT(i == filter[filter_idx]);

		if (aggr.aggr_type != AggregateType::DISTINCT && aggr.filter) {
			RowOperations::UpdateFilteredStates(row_state, filter_set.GetFilterData(i), aggr, state.addresses, payload,
			                                    payload_idx);
		} else {
			RowOperations::UpdateStates(row_state, aggr, state.addresses, payload, payload_idx, payload.size());
		}

		// Move to the next aggregate
		payload_idx += aggr.child_count;
		VectorOperations::AddInPlace(state.addresses, aggr.payload_size, payload.size());
		filter_idx++;
	}

	Verify();
	return new_group_count;
}

void GroupedAggregateHashTable::FetchAggregates(DataChunk &groups, DataChunk &result) {
	groups.Verify();
	D_ASSERT(groups.ColumnCount() + 1 == layout.ColumnCount());
	for (idx_t i = 0; i < result.ColumnCount(); i++) {
		D_ASSERT(result.data[i].GetType() == payload_types[i]);
	}
	result.SetCardinality(groups);
	if (groups.size() == 0) {
		return;
	}

	// find the groups associated with the addresses
	// FIXME: this should not use the FindOrCreateGroups, creating them is unnecessary
	AggregateHTAppendState append_state;
	Vector addresses(LogicalType::POINTER);
	FindOrCreateGroups(append_state, groups, addresses);
	// now fetch the aggregates
	RowOperationsState row_state(aggregate_allocator->GetAllocator());
	RowOperations::FinalizeStates(row_state, layout, addresses, result, 0);
}

idx_t GroupedAggregateHashTable::ResizeThreshold() {
	return capacity / LOAD_FACTOR;
}

template <class ENTRY>
idx_t GroupedAggregateHashTable::FindOrCreateGroupsInternal(AggregateHTAppendState &state, DataChunk &groups,
                                                            Vector &group_hashes_v, Vector &addresses_v,
                                                            SelectionVector &new_groups_out) {
	D_ASSERT(!is_finalized);
	D_ASSERT(groups.ColumnCount() + 1 == layout.ColumnCount());
	D_ASSERT(group_hashes_v.GetType() == LogicalType::HASH);
	D_ASSERT(state.ht_offsets.GetVectorType() == VectorType::FLAT_VECTOR);
	D_ASSERT(state.ht_offsets.GetType() == LogicalType::BIGINT);
	D_ASSERT(addresses_v.GetType() == LogicalType::POINTER);
	D_ASSERT(state.hash_salts.GetType() == LogicalType::SMALLINT);

	if (Count() + groups.size() > MaxCapacity()) {
		throw InternalException("Hash table capacity reached");
	}

	// Resize at 50% capacity, also need to fit the entire vector
	if (capacity - Count() <= groups.size() || Count() > ResizeThreshold()) {
		Verify();
		Resize<ENTRY>(capacity * 2);
	}
	D_ASSERT(capacity - Count() >= groups.size()); // we need to be able to fit at least one vector of data

	group_hashes_v.Flatten(groups.size());
	auto group_hashes = FlatVector::GetData<hash_t>(group_hashes_v);

	addresses_v.Flatten(groups.size());
	auto addresses = FlatVector::GetData<data_ptr_t>(addresses_v);

	// Compute the entry in the table based on the hash using a modulo,
	// and precompute the hash salts for faster comparison below
	auto ht_offsets_ptr = FlatVector::GetData<uint64_t>(state.ht_offsets);
	auto hash_salts_ptr = FlatVector::GetData<uint16_t>(state.hash_salts);
	for (idx_t r = 0; r < groups.size(); r++) {
		auto element = group_hashes[r];
		D_ASSERT((element & bitmask) == (element % capacity));
		ht_offsets_ptr[r] = element & bitmask;
		hash_salts_ptr[r] = element >> hash_prefix_shift;
	}
	// we start out with all entries [0, 1, 2, ..., groups.size()]
	const SelectionVector *sel_vector = FlatVector::IncrementalSelectionVector();

	// Make a chunk that references the groups and the hashes and convert to unified format
	if (state.group_chunk.ColumnCount() == 0) {
		state.group_chunk.InitializeEmpty(layout.GetTypes());
	}
	D_ASSERT(state.group_chunk.ColumnCount() == layout.GetTypes().size());
	for (idx_t grp_idx = 0; grp_idx < groups.ColumnCount(); grp_idx++) {
		state.group_chunk.data[grp_idx].Reference(groups.data[grp_idx]);
	}
	state.group_chunk.data[groups.ColumnCount()].Reference(group_hashes_v);
	state.group_chunk.SetCardinality(groups);

	// convert all vectors to unified format
	if (!state.chunk_state_initialized) {
		data_collection->InitializeAppend(state.chunk_state);
		state.chunk_state_initialized = true;
	}
	TupleDataCollection::ToUnifiedFormat(state.chunk_state, state.group_chunk);
	if (!state.group_data) {
		state.group_data = make_unsafe_uniq_array<UnifiedVectorFormat>(state.group_chunk.ColumnCount());
	}
	TupleDataCollection::GetVectorData(state.chunk_state, state.group_data.get());

	idx_t new_group_count = 0;
	idx_t remaining_entries = groups.size();
	while (remaining_entries > 0) {
		idx_t new_entry_count = 0;
		idx_t need_compare_count = 0;
		idx_t no_match_count = 0;

		// For each remaining entry, figure out whether or not it belongs to a full or empty group
		for (idx_t i = 0; i < remaining_entries; i++) {
			const idx_t index = sel_vector->get_index(i);
			auto &ht_entry = *(((ENTRY *)this->hashes_hdl_ptr) + ht_offsets_ptr[index]);
			if (ht_entry.page_nr == 0) { // Cell is unoccupied (we use page number 0 as a "unused marker")
				D_ASSERT(group_hashes[index] >> hash_prefix_shift <= NumericLimits<uint16_t>::Maximum());
				D_ASSERT(payload_hds_ptrs.size() < NumericLimits<uint32_t>::Maximum());

				// Set page nr to 1 for now to mark it as occupied (will be corrected later) and set the salt
				ht_entry.page_nr = 1;
				ht_entry.salt = group_hashes[index] >> hash_prefix_shift;

				// Update selection lists for outer loops
				state.empty_vector.set_index(new_entry_count++, index);
				new_groups_out.set_index(new_group_count++, index);
			} else { // Cell is occupied: Compare salts
				if (ht_entry.salt == hash_salts_ptr[index]) {
					state.group_compare_vector.set_index(need_compare_count++, index);
				} else {
					state.no_match_vector.set_index(no_match_count++, index);
				}
			}
		}

		if (new_entry_count != 0) {
			// Append everything that belongs to an empty group
			data_collection->AppendUnified(td_pin_state, state.chunk_state, state.group_chunk, state.empty_vector,
			                               new_entry_count);
			RowOperations::InitializeStates(layout, state.chunk_state.row_locations,
			                                *FlatVector::IncrementalSelectionVector(), new_entry_count);

			// Get the pointers to the (possibly) newly created blocks of the data collection
			idx_t block_id = payload_hds_ptrs.empty() ? 0 : payload_hds_ptrs.size() - 1;
			UpdateBlockPointers();
			auto block_pointer = payload_hds_ptrs[block_id];
			auto block_end = block_pointer + tuples_per_block * tuple_size;

			// Set the page nrs/offsets in the 1st part of the HT now that the data has been appended
			const auto row_locations = FlatVector::GetData<data_ptr_t>(state.chunk_state.row_locations);
			for (idx_t new_entry_idx = 0; new_entry_idx < new_entry_count; new_entry_idx++) {
				const auto &row_location = row_locations[new_entry_idx];
				if (row_location > block_end || row_location < block_pointer) {
					block_id++;
					D_ASSERT(block_id < payload_hds_ptrs.size());
					block_pointer = payload_hds_ptrs[block_id];
					block_end = block_pointer + tuples_per_block * tuple_size;
				}
				D_ASSERT(row_location >= block_pointer && row_location < block_end);
				D_ASSERT((row_location - block_pointer) % tuple_size == 0);
				const auto index = state.empty_vector.get_index(new_entry_idx);
				auto &ht_entry = *(((ENTRY *)this->hashes_hdl_ptr) + ht_offsets_ptr[index]);
				ht_entry.page_nr = block_id + 1;
				ht_entry.page_offset = (row_location - block_pointer) / tuple_size;
				addresses[index] = row_location;
			}
		}

		if (need_compare_count != 0) {
			// Get the pointers to the rows that need to be compared
			for (idx_t need_compare_idx = 0; need_compare_idx < need_compare_count; need_compare_idx++) {
				const auto index = state.group_compare_vector.get_index(need_compare_idx);
				const auto &ht_entry = *(((ENTRY *)this->hashes_hdl_ptr) + ht_offsets_ptr[index]);
				auto page_ptr = payload_hds_ptrs[ht_entry.page_nr - 1];
				auto page_offset = ht_entry.page_offset * tuple_size;
				addresses[index] = page_ptr + page_offset;
			}

			// Perform group comparisons
			RowOperations::Match(state.group_chunk, state.group_data.get(), layout, addresses_v, predicates,
			                     state.group_compare_vector, need_compare_count, &state.no_match_vector,
			                     no_match_count);
		}

		// Linear probing: each of the entries that do not match move to the next entry in the HT
		for (idx_t i = 0; i < no_match_count; i++) {
			idx_t index = state.no_match_vector.get_index(i);
			ht_offsets_ptr[index]++;
			if (ht_offsets_ptr[index] >= capacity) {
				ht_offsets_ptr[index] = 0;
			}
		}
		sel_vector = &state.no_match_vector;
		remaining_entries = no_match_count;
	}

	return new_group_count;
}

void GroupedAggregateHashTable::UpdateBlockPointers() {
	for (const auto &id_and_handle : td_pin_state.row_handles) {
		const auto &id = id_and_handle.first;
		const auto &handle = id_and_handle.second;
		if (payload_hds_ptrs.empty() || id > payload_hds_ptrs.size() - 1) {
			payload_hds_ptrs.resize(id + 1);
		}
		payload_hds_ptrs[id] = handle.Ptr();
	}
}

// this is to support distinct aggregations where we need to record whether we
// have already seen a value for a group
idx_t GroupedAggregateHashTable::FindOrCreateGroups(AggregateHTAppendState &state, DataChunk &groups,
                                                    Vector &group_hashes, Vector &addresses_out,
                                                    SelectionVector &new_groups_out) {
	switch (entry_type) {
	case HtEntryType::HT_WIDTH_64:
		return FindOrCreateGroupsInternal<aggr_ht_entry_64>(state, groups, group_hashes, addresses_out, new_groups_out);
	case HtEntryType::HT_WIDTH_32:
		return FindOrCreateGroupsInternal<aggr_ht_entry_32>(state, groups, group_hashes, addresses_out, new_groups_out);
	default:
		throw InternalException("Unknown HT entry width");
	}
}

void GroupedAggregateHashTable::FindOrCreateGroups(AggregateHTAppendState &state, DataChunk &groups,
                                                   Vector &addresses) {
	// create a dummy new_groups sel vector
	FindOrCreateGroups(state, groups, addresses, state.new_groups);
}

idx_t GroupedAggregateHashTable::FindOrCreateGroups(AggregateHTAppendState &state, DataChunk &groups,
                                                    Vector &addresses_out, SelectionVector &new_groups_out) {
	Vector hashes(LogicalType::HASH);
	groups.Hash(hashes);
	return FindOrCreateGroups(state, groups, hashes, addresses_out, new_groups_out);
}

struct FlushMoveState {
	explicit FlushMoveState(TupleDataCollection &collection_p)
	    : collection(collection_p), hashes(LogicalType::HASH), group_addresses(LogicalType::POINTER),
	      new_groups_sel(STANDARD_VECTOR_SIZE) {
		const auto &layout = collection.GetLayout();
		vector<column_t> column_ids;
		column_ids.reserve(layout.ColumnCount() - 1);
		for (idx_t col_idx = 0; col_idx < layout.ColumnCount() - 1; col_idx++) {
			column_ids.emplace_back(col_idx);
		}
		// FIXME DESTROY_AFTER_DONE if we make it possible to pass a selection vector to RowOperations::DestroyStates?
		collection.InitializeScan(scan_state, column_ids, TupleDataPinProperties::UNPIN_AFTER_DONE);
		collection.InitializeScanChunk(scan_state, groups);
		hash_col_idx = layout.ColumnCount() - 1;
	}

	bool Scan();

	TupleDataCollection &collection;
	TupleDataScanState scan_state;
	DataChunk groups;

	idx_t hash_col_idx;
	Vector hashes;

	AggregateHTAppendState append_state;
	Vector group_addresses;
	SelectionVector new_groups_sel;
};

bool FlushMoveState::Scan() {
	if (collection.Scan(scan_state, groups)) {
		collection.Gather(scan_state.chunk_state.row_locations, *FlatVector::IncrementalSelectionVector(),
		                  groups.size(), hash_col_idx, hashes, *FlatVector::IncrementalSelectionVector());
		return true;
	}

	collection.FinalizePinState(scan_state.pin_state);
	return false;
}

void GroupedAggregateHashTable::Combine(GroupedAggregateHashTable &other) {
	D_ASSERT(!is_finalized);

	D_ASSERT(other.layout.GetAggrWidth() == layout.GetAggrWidth());
	D_ASSERT(other.layout.GetDataWidth() == layout.GetDataWidth());
	D_ASSERT(other.layout.GetRowWidth() == layout.GetRowWidth());

	if (other.Count() == 0) {
		return;
	}

	FlushMoveState state(*other.data_collection);
	RowOperationsState row_state(aggregate_allocator->GetAllocator());
	while (state.Scan()) {
		FindOrCreateGroups(state.append_state, state.groups, state.hashes, state.group_addresses, state.new_groups_sel);
		RowOperations::CombineStates(row_state, layout, state.scan_state.chunk_state.row_locations,
		                             state.group_addresses, state.groups.size());
	}

	Verify();
}

void GroupedAggregateHashTable::Partition(vector<GroupedAggregateHashTable *> &partition_hts, idx_t radix_bits) {
	const auto num_partitions = RadixPartitioning::NumberOfPartitions(radix_bits);
	D_ASSERT(partition_hts.size() == num_partitions);

	// Partition the data
	auto partitioned_data =
	    make_uniq<RadixPartitionedTupleData>(buffer_manager, layout, radix_bits, layout.ColumnCount() - 1);
	partitioned_data->Partition(*data_collection, TupleDataPinProperties::KEEP_EVERYTHING_PINNED);
	D_ASSERT(partitioned_data->GetPartitions().size() == num_partitions);

	// Move the partitioned data collections to the partitioned hash tables and initialize the 1st part of the HT
	auto &partitions = partitioned_data->GetPartitions();
	for (idx_t partition_idx = 0; partition_idx < num_partitions; partition_idx++) {
		auto &partition_ht = *partition_hts[partition_idx];
		partition_ht.data_collection = std::move(partitions[partition_idx]);
		partition_ht.aggregate_allocator = aggregate_allocator;
		partition_ht.InitializeFirstPart();
		partition_ht.Verify();
	}
}

void GroupedAggregateHashTable::InitializeFirstPart() {
	data_collection->GetBlockPointers(payload_hds_ptrs);
	auto size = MaxValue<idx_t>(NextPowerOfTwo(Count() * 2L), capacity);
	switch (entry_type) {
	case HtEntryType::HT_WIDTH_64:
		Resize<aggr_ht_entry_64>(size);
		break;
	case HtEntryType::HT_WIDTH_32:
		Resize<aggr_ht_entry_32>(size);
		break;
	default:
		throw InternalException("Unknown HT entry width");
	}
}

idx_t GroupedAggregateHashTable::Scan(TupleDataParallelScanState &gstate, TupleDataLocalScanState &lstate,
                                      DataChunk &result) {
	data_collection->Scan(gstate, lstate, result);

	RowOperationsState row_state(aggregate_allocator->GetAllocator());
	const auto group_cols = layout.ColumnCount() - 1;
	RowOperations::FinalizeStates(row_state, layout, lstate.chunk_state.row_locations, result, group_cols);

	return result.size();
}

void GroupedAggregateHashTable::Finalize() {
	if (is_finalized) {
		return;
	}

	// Early release hashes (not needed for partition/scan) and data collection (will be pinned again when scanning)
	hashes_hdl.Destroy();
	data_collection->FinalizePinState(td_pin_state);
	data_collection->Unpin();

	is_finalized = true;
}

} // namespace duckdb




namespace duckdb {

BaseAggregateHashTable::BaseAggregateHashTable(ClientContext &context, Allocator &allocator,
                                               const vector<AggregateObject> &aggregates,
                                               vector<LogicalType> payload_types_p)
    : allocator(allocator), buffer_manager(BufferManager::GetBufferManager(context)),
      payload_types(std::move(payload_types_p)) {
	filter_set.Initialize(context, aggregates, payload_types);
}

} // namespace duckdb














namespace duckdb {

ColumnBindingResolver::ColumnBindingResolver() {
}

void ColumnBindingResolver::VisitOperator(LogicalOperator &op) {
	switch (op.type) {
	case LogicalOperatorType::LOGICAL_ASOF_JOIN:
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
	case LogicalOperatorType::LOGICAL_DELIM_JOIN: {
		// special case: comparison join
		auto &comp_join = op.Cast<LogicalComparisonJoin>();
		// first get the bindings of the LHS and resolve the LHS expressions
		VisitOperator(*comp_join.children[0]);
		for (auto &cond : comp_join.conditions) {
			VisitExpression(&cond.left);
		}
		if (op.type == LogicalOperatorType::LOGICAL_DELIM_JOIN) {
			// visit the duplicate eliminated columns on the LHS, if any
			auto &delim_join = op.Cast<LogicalDelimJoin>();
			for (auto &expr : delim_join.duplicate_eliminated_columns) {
				VisitExpression(&expr);
			}
		}
		// then get the bindings of the RHS and resolve the RHS expressions
		VisitOperator(*comp_join.children[1]);
		for (auto &cond : comp_join.conditions) {
			VisitExpression(&cond.right);
		}
		// finally update the bindings with the result bindings of the join
		bindings = op.GetColumnBindings();
		return;
	}
	case LogicalOperatorType::LOGICAL_ANY_JOIN: {
		// ANY join, this join is different because we evaluate the expression on the bindings of BOTH join sides at
		// once i.e. we set the bindings first to the bindings of the entire join, and then resolve the expressions of
		// this operator
		VisitOperatorChildren(op);
		bindings = op.GetColumnBindings();
		auto &any_join = op.Cast<LogicalAnyJoin>();
		if (any_join.join_type == JoinType::SEMI || any_join.join_type == JoinType::ANTI) {
			auto right_bindings = op.children[1]->GetColumnBindings();
			bindings.insert(bindings.end(), right_bindings.begin(), right_bindings.end());
		}
		VisitOperatorExpressions(op);
		return;
	}
	case LogicalOperatorType::LOGICAL_CREATE_INDEX: {
		// CREATE INDEX statement, add the columns of the table with table index 0 to the binding set
		// afterwards bind the expressions of the CREATE INDEX statement
		auto &create_index = op.Cast<LogicalCreateIndex>();
		bindings = LogicalOperator::GenerateColumnBindings(0, create_index.table.GetColumns().LogicalColumnCount());
		VisitOperatorExpressions(op);
		return;
	}
	case LogicalOperatorType::LOGICAL_GET: {
		//! We first need to update the current set of bindings and then visit operator expressions
		bindings = op.GetColumnBindings();
		VisitOperatorExpressions(op);
		return;
	}
	case LogicalOperatorType::LOGICAL_INSERT: {
		//! We want to execute the normal path, but also add a dummy 'excluded' binding if there is a
		// ON CONFLICT DO UPDATE clause
		auto &insert_op = op.Cast<LogicalInsert>();
		if (insert_op.action_type != OnConflictAction::THROW) {
			// Get the bindings from the children
			VisitOperatorChildren(op);
			auto column_count = insert_op.table.GetColumns().PhysicalColumnCount();
			auto dummy_bindings = LogicalOperator::GenerateColumnBindings(insert_op.excluded_table_index, column_count);
			// Now insert our dummy bindings at the start of the bindings,
			// so the first 'column_count' indices of the chunk are reserved for our 'excluded' columns
			bindings.insert(bindings.begin(), dummy_bindings.begin(), dummy_bindings.end());
			if (insert_op.on_conflict_condition) {
				VisitExpression(&insert_op.on_conflict_condition);
			}
			if (insert_op.do_update_condition) {
				VisitExpression(&insert_op.do_update_condition);
			}
			VisitOperatorExpressions(op);
			bindings = op.GetColumnBindings();
			return;
		}
	}
	default:
		break;
	}
	// general case
	// first visit the children of this operator
	VisitOperatorChildren(op);
	// now visit the expressions of this operator to resolve any bound column references
	VisitOperatorExpressions(op);
	// finally update the current set of bindings to the current set of column bindings
	bindings = op.GetColumnBindings();
}

unique_ptr<Expression> ColumnBindingResolver::VisitReplace(BoundColumnRefExpression &expr,
                                                           unique_ptr<Expression> *expr_ptr) {
	D_ASSERT(expr.depth == 0);
	// check the current set of column bindings to see which index corresponds to the column reference
	for (idx_t i = 0; i < bindings.size(); i++) {
		if (expr.binding == bindings[i]) {
			return make_uniq<BoundReferenceExpression>(expr.alias, expr.return_type, i);
		}
	}
	// LCOV_EXCL_START
	// could not bind the column reference, this should never happen and indicates a bug in the code
	// generate an error message
	string bound_columns = "[";
	for (idx_t i = 0; i < bindings.size(); i++) {
		if (i != 0) {
			bound_columns += " ";
		}
		bound_columns += to_string(bindings[i].table_index) + "." + to_string(bindings[i].column_index);
	}
	bound_columns += "]";

	throw InternalException("Failed to bind column reference \"%s\" [%d.%d] (bindings: %s)", expr.alias,
	                        expr.binding.table_index, expr.binding.column_index, bound_columns);
	// LCOV_EXCL_STOP
}

unordered_set<idx_t> ColumnBindingResolver::VerifyInternal(LogicalOperator &op) {
	unordered_set<idx_t> result;
	for (auto &child : op.children) {
		auto child_indexes = VerifyInternal(*child);
		for (auto index : child_indexes) {
			D_ASSERT(index != DConstants::INVALID_INDEX);
			if (result.find(index) != result.end()) {
				throw InternalException("Duplicate table index \"%lld\" found", index);
			}
			result.insert(index);
		}
	}
	auto indexes = op.GetTableIndex();
	for (auto index : indexes) {
		D_ASSERT(index != DConstants::INVALID_INDEX);
		if (result.find(index) != result.end()) {
			throw InternalException("Duplicate table index \"%lld\" found", index);
		}
		result.insert(index);
	}
	return result;
}

void ColumnBindingResolver::Verify(LogicalOperator &op) {
#ifdef DEBUG
	VerifyInternal(op);
#endif
}

} // namespace duckdb






namespace duckdb {

struct BothInclusiveBetweenOperator {
	template <class T>
	static inline bool Operation(T input, T lower, T upper) {
		return GreaterThanEquals::Operation<T>(input, lower) && LessThanEquals::Operation<T>(input, upper);
	}
};

struct LowerInclusiveBetweenOperator {
	template <class T>
	static inline bool Operation(T input, T lower, T upper) {
		return GreaterThanEquals::Operation<T>(input, lower) && LessThan::Operation<T>(input, upper);
	}
};

struct UpperInclusiveBetweenOperator {
	template <class T>
	static inline bool Operation(T input, T lower, T upper) {
		return GreaterThan::Operation<T>(input, lower) && LessThanEquals::Operation<T>(input, upper);
	}
};

struct ExclusiveBetweenOperator {
	template <class T>
	static inline bool Operation(T input, T lower, T upper) {
		return GreaterThan::Operation<T>(input, lower) && LessThan::Operation<T>(input, upper);
	}
};

template <class OP>
static idx_t BetweenLoopTypeSwitch(Vector &input, Vector &lower, Vector &upper, const SelectionVector *sel, idx_t count,
                                   SelectionVector *true_sel, SelectionVector *false_sel) {
	switch (input.GetType().InternalType()) {
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
		return TernaryExecutor::Select<int8_t, int8_t, int8_t, OP>(input, lower, upper, sel, count, true_sel,
		                                                           false_sel);
	case PhysicalType::INT16:
		return TernaryExecutor::Select<int16_t, int16_t, int16_t, OP>(input, lower, upper, sel, count, true_sel,
		                                                              false_sel);
	case PhysicalType::INT32:
		return TernaryExecutor::Select<int32_t, int32_t, int32_t, OP>(input, lower, upper, sel, count, true_sel,
		                                                              false_sel);
	case PhysicalType::INT64:
		return TernaryExecutor::Select<int64_t, int64_t, int64_t, OP>(input, lower, upper, sel, count, true_sel,
		                                                              false_sel);
	case PhysicalType::INT128:
		return TernaryExecutor::Select<hugeint_t, hugeint_t, hugeint_t, OP>(input, lower, upper, sel, count, true_sel,
		                                                                    false_sel);
	case PhysicalType::UINT8:
		return TernaryExecutor::Select<uint8_t, uint8_t, uint8_t, OP>(input, lower, upper, sel, count, true_sel,
		                                                              false_sel);
	case PhysicalType::UINT16:
		return TernaryExecutor::Select<uint16_t, uint16_t, uint16_t, OP>(input, lower, upper, sel, count, true_sel,
		                                                                 false_sel);
	case PhysicalType::UINT32:
		return TernaryExecutor::Select<uint32_t, uint32_t, uint32_t, OP>(input, lower, upper, sel, count, true_sel,
		                                                                 false_sel);
	case PhysicalType::UINT64:
		return TernaryExecutor::Select<uint64_t, uint64_t, uint64_t, OP>(input, lower, upper, sel, count, true_sel,
		                                                                 false_sel);
	case PhysicalType::FLOAT:
		return TernaryExecutor::Select<float, float, float, OP>(input, lower, upper, sel, count, true_sel, false_sel);
	case PhysicalType::DOUBLE:
		return TernaryExecutor::Select<double, double, double, OP>(input, lower, upper, sel, count, true_sel,
		                                                           false_sel);
	case PhysicalType::VARCHAR:
		return TernaryExecutor::Select<string_t, string_t, string_t, OP>(input, lower, upper, sel, count, true_sel,
		                                                                 false_sel);
	case PhysicalType::INTERVAL:
		return TernaryExecutor::Select<interval_t, interval_t, interval_t, OP>(input, lower, upper, sel, count,
		                                                                       true_sel, false_sel);
	default:
		throw InvalidTypeException(input.GetType(), "Invalid type for BETWEEN");
	}
}

unique_ptr<ExpressionState> ExpressionExecutor::InitializeState(const BoundBetweenExpression &expr,
                                                                ExpressionExecutorState &root) {
	auto result = make_uniq<ExpressionState>(expr, root);
	result->AddChild(expr.input.get());
	result->AddChild(expr.lower.get());
	result->AddChild(expr.upper.get());
	result->Finalize();
	return result;
}

void ExpressionExecutor::Execute(const BoundBetweenExpression &expr, ExpressionState *state, const SelectionVector *sel,
                                 idx_t count, Vector &result) {
	// resolve the children
	state->intermediate_chunk.Reset();

	auto &input = state->intermediate_chunk.data[0];
	auto &lower = state->intermediate_chunk.data[1];
	auto &upper = state->intermediate_chunk.data[2];

	Execute(*expr.input, state->child_states[0].get(), sel, count, input);
	Execute(*expr.lower, state->child_states[1].get(), sel, count, lower);
	Execute(*expr.upper, state->child_states[2].get(), sel, count, upper);

	Vector intermediate1(LogicalType::BOOLEAN);
	Vector intermediate2(LogicalType::BOOLEAN);

	if (expr.upper_inclusive && expr.lower_inclusive) {
		VectorOperations::GreaterThanEquals(input, lower, intermediate1, count);
		VectorOperations::LessThanEquals(input, upper, intermediate2, count);
	} else if (expr.lower_inclusive) {
		VectorOperations::GreaterThanEquals(input, lower, intermediate1, count);
		VectorOperations::LessThan(input, upper, intermediate2, count);
	} else if (expr.upper_inclusive) {
		VectorOperations::GreaterThan(input, lower, intermediate1, count);
		VectorOperations::LessThanEquals(input, upper, intermediate2, count);
	} else {
		VectorOperations::GreaterThan(input, lower, intermediate1, count);
		VectorOperations::LessThan(input, upper, intermediate2, count);
	}
	VectorOperations::And(intermediate1, intermediate2, result, count);
}

idx_t ExpressionExecutor::Select(const BoundBetweenExpression &expr, ExpressionState *state, const SelectionVector *sel,
                                 idx_t count, SelectionVector *true_sel, SelectionVector *false_sel) {
	// resolve the children
	Vector input(state->intermediate_chunk.data[0]);
	Vector lower(state->intermediate_chunk.data[1]);
	Vector upper(state->intermediate_chunk.data[2]);

	Execute(*expr.input, state->child_states[0].get(), sel, count, input);
	Execute(*expr.lower, state->child_states[1].get(), sel, count, lower);
	Execute(*expr.upper, state->child_states[2].get(), sel, count, upper);

	if (expr.upper_inclusive && expr.lower_inclusive) {
		return BetweenLoopTypeSwitch<BothInclusiveBetweenOperator>(input, lower, upper, sel, count, true_sel,
		                                                           false_sel);
	} else if (expr.lower_inclusive) {
		return BetweenLoopTypeSwitch<LowerInclusiveBetweenOperator>(input, lower, upper, sel, count, true_sel,
		                                                            false_sel);
	} else if (expr.upper_inclusive) {
		return BetweenLoopTypeSwitch<UpperInclusiveBetweenOperator>(input, lower, upper, sel, count, true_sel,
		                                                            false_sel);
	} else {
		return BetweenLoopTypeSwitch<ExclusiveBetweenOperator>(input, lower, upper, sel, count, true_sel, false_sel);
	}
}

} // namespace duckdb




namespace duckdb {

struct CaseExpressionState : public ExpressionState {
	CaseExpressionState(const Expression &expr, ExpressionExecutorState &root)
	    : ExpressionState(expr, root), true_sel(STANDARD_VECTOR_SIZE), false_sel(STANDARD_VECTOR_SIZE) {
	}

	SelectionVector true_sel;
	SelectionVector false_sel;
};

unique_ptr<ExpressionState> ExpressionExecutor::InitializeState(const BoundCaseExpression &expr,
                                                                ExpressionExecutorState &root) {
	auto result = make_uniq<CaseExpressionState>(expr, root);
	for (auto &case_check : expr.case_checks) {
		result->AddChild(case_check.when_expr.get());
		result->AddChild(case_check.then_expr.get());
	}
	result->AddChild(expr.else_expr.get());
	result->Finalize();
	return std::move(result);
}

void ExpressionExecutor::Execute(const BoundCaseExpression &expr, ExpressionState *state_p, const SelectionVector *sel,
                                 idx_t count, Vector &result) {
	auto state = (CaseExpressionState *)state_p;

	state->intermediate_chunk.Reset();

	// first execute the check expression
	auto current_true_sel = &state->true_sel;
	auto current_false_sel = &state->false_sel;
	auto current_sel = sel;
	idx_t current_count = count;
	for (idx_t i = 0; i < expr.case_checks.size(); i++) {
		auto &case_check = expr.case_checks[i];
		auto &intermediate_result = state->intermediate_chunk.data[i * 2 + 1];
		auto check_state = state->child_states[i * 2].get();
		auto then_state = state->child_states[i * 2 + 1].get();

		idx_t tcount =
		    Select(*case_check.when_expr, check_state, current_sel, current_count, current_true_sel, current_false_sel);
		if (tcount == 0) {
			// everything is false: do nothing
			continue;
		}
		idx_t fcount = current_count - tcount;
		if (fcount == 0 && current_count == count) {
			// everything is true in the first CHECK statement
			// we can skip the entire case and only execute the TRUE side
			Execute(*case_check.then_expr, then_state, sel, count, result);
			return;
		} else {
			// we need to execute and then fill in the desired tuples in the result
			Execute(*case_check.then_expr, then_state, current_true_sel, tcount, intermediate_result);
			FillSwitch(intermediate_result, result, *current_true_sel, tcount);
		}
		// continue with the false tuples
		current_sel = current_false_sel;
		current_count = fcount;
		if (fcount == 0) {
			// everything is true: we are done
			break;
		}
	}
	if (current_count > 0) {
		auto else_state = state->child_states.back().get();
		if (current_count == count) {
			// everything was false, we can just evaluate the else expression directly
			Execute(*expr.else_expr, else_state, sel, count, result);
			return;
		} else {
			auto &intermediate_result = state->intermediate_chunk.data[expr.case_checks.size() * 2];

			D_ASSERT(current_sel);
			Execute(*expr.else_expr, else_state, current_sel, current_count, intermediate_result);
			FillSwitch(intermediate_result, result, *current_sel, current_count);
		}
	}
	if (sel) {
		result.Slice(*sel, count);
	}
}

template <class T>
void TemplatedFillLoop(Vector &vector, Vector &result, const SelectionVector &sel, sel_t count) {
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto res = FlatVector::GetData<T>(result);
	auto &result_mask = FlatVector::Validity(result);
	if (vector.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		auto data = ConstantVector::GetData<T>(vector);
		if (ConstantVector::IsNull(vector)) {
			for (idx_t i = 0; i < count; i++) {
				result_mask.SetInvalid(sel.get_index(i));
			}
		} else {
			for (idx_t i = 0; i < count; i++) {
				res[sel.get_index(i)] = *data;
			}
		}
	} else {
		UnifiedVectorFormat vdata;
		vector.ToUnifiedFormat(count, vdata);
		auto data = (T *)vdata.data;
		for (idx_t i = 0; i < count; i++) {
			auto source_idx = vdata.sel->get_index(i);
			auto res_idx = sel.get_index(i);

			res[res_idx] = data[source_idx];
			result_mask.Set(res_idx, vdata.validity.RowIsValid(source_idx));
		}
	}
}

void ValidityFillLoop(Vector &vector, Vector &result, const SelectionVector &sel, sel_t count) {
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto &result_mask = FlatVector::Validity(result);
	if (vector.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		if (ConstantVector::IsNull(vector)) {
			for (idx_t i = 0; i < count; i++) {
				result_mask.SetInvalid(sel.get_index(i));
			}
		}
	} else {
		UnifiedVectorFormat vdata;
		vector.ToUnifiedFormat(count, vdata);
		if (vdata.validity.AllValid()) {
			return;
		}
		for (idx_t i = 0; i < count; i++) {
			auto source_idx = vdata.sel->get_index(i);
			if (!vdata.validity.RowIsValid(source_idx)) {
				result_mask.SetInvalid(sel.get_index(i));
			}
		}
	}
}

void ExpressionExecutor::FillSwitch(Vector &vector, Vector &result, const SelectionVector &sel, sel_t count) {
	switch (result.GetType().InternalType()) {
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
		TemplatedFillLoop<int8_t>(vector, result, sel, count);
		break;
	case PhysicalType::INT16:
		TemplatedFillLoop<int16_t>(vector, result, sel, count);
		break;
	case PhysicalType::INT32:
		TemplatedFillLoop<int32_t>(vector, result, sel, count);
		break;
	case PhysicalType::INT64:
		TemplatedFillLoop<int64_t>(vector, result, sel, count);
		break;
	case PhysicalType::UINT8:
		TemplatedFillLoop<uint8_t>(vector, result, sel, count);
		break;
	case PhysicalType::UINT16:
		TemplatedFillLoop<uint16_t>(vector, result, sel, count);
		break;
	case PhysicalType::UINT32:
		TemplatedFillLoop<uint32_t>(vector, result, sel, count);
		break;
	case PhysicalType::UINT64:
		TemplatedFillLoop<uint64_t>(vector, result, sel, count);
		break;
	case PhysicalType::INT128:
		TemplatedFillLoop<hugeint_t>(vector, result, sel, count);
		break;
	case PhysicalType::FLOAT:
		TemplatedFillLoop<float>(vector, result, sel, count);
		break;
	case PhysicalType::DOUBLE:
		TemplatedFillLoop<double>(vector, result, sel, count);
		break;
	case PhysicalType::INTERVAL:
		TemplatedFillLoop<interval_t>(vector, result, sel, count);
		break;
	case PhysicalType::VARCHAR:
		TemplatedFillLoop<string_t>(vector, result, sel, count);
		StringVector::AddHeapReference(result, vector);
		break;
	case PhysicalType::STRUCT: {
		auto &vector_entries = StructVector::GetEntries(vector);
		auto &result_entries = StructVector::GetEntries(result);
		ValidityFillLoop(vector, result, sel, count);
		D_ASSERT(vector_entries.size() == result_entries.size());
		for (idx_t i = 0; i < vector_entries.size(); i++) {
			FillSwitch(*vector_entries[i], *result_entries[i], sel, count);
		}
		break;
	}
	case PhysicalType::LIST: {
		idx_t offset = ListVector::GetListSize(result);
		auto &list_child = ListVector::GetEntry(vector);
		ListVector::Append(result, list_child, ListVector::GetListSize(vector));

		// all the false offsets need to be incremented by true_child.count
		TemplatedFillLoop<list_entry_t>(vector, result, sel, count);
		if (offset == 0) {
			break;
		}

		auto result_data = FlatVector::GetData<list_entry_t>(result);
		for (idx_t i = 0; i < count; i++) {
			auto result_idx = sel.get_index(i);
			result_data[result_idx].offset += offset;
		}

		Vector::Verify(result, sel, count);
		break;
	}
	default:
		throw NotImplementedException("Unimplemented type for case expression: %s", result.GetType().ToString());
	}
}

} // namespace duckdb





namespace duckdb {

unique_ptr<ExpressionState> ExpressionExecutor::InitializeState(const BoundCastExpression &expr,
                                                                ExpressionExecutorState &root) {
	auto result = make_uniq<ExecuteFunctionState>(expr, root);
	result->AddChild(expr.child.get());
	result->Finalize();
	if (expr.bound_cast.init_local_state) {
		CastLocalStateParameters parameters(root.executor->GetContext(), expr.bound_cast.cast_data);
		result->local_state = expr.bound_cast.init_local_state(parameters);
	}
	return std::move(result);
}

void ExpressionExecutor::Execute(const BoundCastExpression &expr, ExpressionState *state, const SelectionVector *sel,
                                 idx_t count, Vector &result) {
	auto lstate = ExecuteFunctionState::GetFunctionState(*state);

	// resolve the child
	state->intermediate_chunk.Reset();

	auto &child = state->intermediate_chunk.data[0];
	auto child_state = state->child_states[0].get();

	Execute(*expr.child, child_state, sel, count, child);
	if (expr.try_cast) {
		string error_message;
		CastParameters parameters(expr.bound_cast.cast_data.get(), false, &error_message, lstate);
		expr.bound_cast.function(child, result, count, parameters);
	} else {
		// cast it to the type specified by the cast expression
		D_ASSERT(result.GetType() == expr.return_type);
		CastParameters parameters(expr.bound_cast.cast_data.get(), false, nullptr, lstate);
		expr.bound_cast.function(child, result, count, parameters);
	}
}

} // namespace duckdb






#include <algorithm>

namespace duckdb {

unique_ptr<ExpressionState> ExpressionExecutor::InitializeState(const BoundComparisonExpression &expr,
                                                                ExpressionExecutorState &root) {
	auto result = make_uniq<ExpressionState>(expr, root);
	result->AddChild(expr.left.get());
	result->AddChild(expr.right.get());
	result->Finalize();
	return result;
}

void ExpressionExecutor::Execute(const BoundComparisonExpression &expr, ExpressionState *state,
                                 const SelectionVector *sel, idx_t count, Vector &result) {
	// resolve the children
	state->intermediate_chunk.Reset();
	auto &left = state->intermediate_chunk.data[0];
	auto &right = state->intermediate_chunk.data[1];

	Execute(*expr.left, state->child_states[0].get(), sel, count, left);
	Execute(*expr.right, state->child_states[1].get(), sel, count, right);

	switch (expr.type) {
	case ExpressionType::COMPARE_EQUAL:
		VectorOperations::Equals(left, right, result, count);
		break;
	case ExpressionType::COMPARE_NOTEQUAL:
		VectorOperations::NotEquals(left, right, result, count);
		break;
	case ExpressionType::COMPARE_LESSTHAN:
		VectorOperations::LessThan(left, right, result, count);
		break;
	case ExpressionType::COMPARE_GREATERTHAN:
		VectorOperations::GreaterThan(left, right, result, count);
		break;
	case ExpressionType::COMPARE_LESSTHANOREQUALTO:
		VectorOperations::LessThanEquals(left, right, result, count);
		break;
	case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
		VectorOperations::GreaterThanEquals(left, right, result, count);
		break;
	case ExpressionType::COMPARE_DISTINCT_FROM:
		VectorOperations::DistinctFrom(left, right, result, count);
		break;
	case ExpressionType::COMPARE_NOT_DISTINCT_FROM:
		VectorOperations::NotDistinctFrom(left, right, result, count);
		break;
	default:
		throw InternalException("Unknown comparison type!");
	}
}

template <typename OP>
static idx_t NestedSelectOperation(Vector &left, Vector &right, const SelectionVector *sel, idx_t count,
                                   SelectionVector *true_sel, SelectionVector *false_sel);

template <class OP>
static idx_t TemplatedSelectOperation(Vector &left, Vector &right, const SelectionVector *sel, idx_t count,
                                      SelectionVector *true_sel, SelectionVector *false_sel) {
	// the inplace loops take the result as the last parameter
	switch (left.GetType().InternalType()) {
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
		return BinaryExecutor::Select<int8_t, int8_t, OP>(left, right, sel, count, true_sel, false_sel);
	case PhysicalType::INT16:
		return BinaryExecutor::Select<int16_t, int16_t, OP>(left, right, sel, count, true_sel, false_sel);
	case PhysicalType::INT32:
		return BinaryExecutor::Select<int32_t, int32_t, OP>(left, right, sel, count, true_sel, false_sel);
	case PhysicalType::INT64:
		return BinaryExecutor::Select<int64_t, int64_t, OP>(left, right, sel, count, true_sel, false_sel);
	case PhysicalType::UINT8:
		return BinaryExecutor::Select<uint8_t, uint8_t, OP>(left, right, sel, count, true_sel, false_sel);
	case PhysicalType::UINT16:
		return BinaryExecutor::Select<uint16_t, uint16_t, OP>(left, right, sel, count, true_sel, false_sel);
	case PhysicalType::UINT32:
		return BinaryExecutor::Select<uint32_t, uint32_t, OP>(left, right, sel, count, true_sel, false_sel);
	case PhysicalType::UINT64:
		return BinaryExecutor::Select<uint64_t, uint64_t, OP>(left, right, sel, count, true_sel, false_sel);
	case PhysicalType::INT128:
		return BinaryExecutor::Select<hugeint_t, hugeint_t, OP>(left, right, sel, count, true_sel, false_sel);
	case PhysicalType::FLOAT:
		return BinaryExecutor::Select<float, float, OP>(left, right, sel, count, true_sel, false_sel);
	case PhysicalType::DOUBLE:
		return BinaryExecutor::Select<double, double, OP>(left, right, sel, count, true_sel, false_sel);
	case PhysicalType::INTERVAL:
		return BinaryExecutor::Select<interval_t, interval_t, OP>(left, right, sel, count, true_sel, false_sel);
	case PhysicalType::VARCHAR:
		return BinaryExecutor::Select<string_t, string_t, OP>(left, right, sel, count, true_sel, false_sel);
	case PhysicalType::LIST:
	case PhysicalType::STRUCT:
		return NestedSelectOperation<OP>(left, right, sel, count, true_sel, false_sel);
	default:
		throw InternalException("Invalid type for comparison");
	}
}

struct NestedSelector {
	// Select the matching rows for the values of a nested type that are not both NULL.
	// Those semantics are the same as the corresponding non-distinct comparator
	template <typename OP>
	static idx_t Select(Vector &left, Vector &right, const SelectionVector &sel, idx_t count, SelectionVector *true_sel,
	                    SelectionVector *false_sel) {
		throw InvalidTypeException(left.GetType(), "Invalid operation for nested SELECT");
	}
};

template <>
idx_t NestedSelector::Select<duckdb::Equals>(Vector &left, Vector &right, const SelectionVector &sel, idx_t count,
                                             SelectionVector *true_sel, SelectionVector *false_sel) {
	return VectorOperations::NestedEquals(left, right, sel, count, true_sel, false_sel);
}

template <>
idx_t NestedSelector::Select<duckdb::NotEquals>(Vector &left, Vector &right, const SelectionVector &sel, idx_t count,
                                                SelectionVector *true_sel, SelectionVector *false_sel) {
	return VectorOperations::NestedNotEquals(left, right, sel, count, true_sel, false_sel);
}

template <>
idx_t NestedSelector::Select<duckdb::LessThan>(Vector &left, Vector &right, const SelectionVector &sel, idx_t count,
                                               SelectionVector *true_sel, SelectionVector *false_sel) {
	return VectorOperations::DistinctLessThan(left, right, &sel, count, true_sel, false_sel);
}

template <>
idx_t NestedSelector::Select<duckdb::LessThanEquals>(Vector &left, Vector &right, const SelectionVector &sel,
                                                     idx_t count, SelectionVector *true_sel,
                                                     SelectionVector *false_sel) {
	return VectorOperations::DistinctLessThanEquals(left, right, &sel, count, true_sel, false_sel);
}

template <>
idx_t NestedSelector::Select<duckdb::GreaterThan>(Vector &left, Vector &right, const SelectionVector &sel, idx_t count,
                                                  SelectionVector *true_sel, SelectionVector *false_sel) {
	return VectorOperations::DistinctGreaterThan(left, right, &sel, count, true_sel, false_sel);
}

template <>
idx_t NestedSelector::Select<duckdb::GreaterThanEquals>(Vector &left, Vector &right, const SelectionVector &sel,
                                                        idx_t count, SelectionVector *true_sel,
                                                        SelectionVector *false_sel) {
	return VectorOperations::DistinctGreaterThanEquals(left, right, &sel, count, true_sel, false_sel);
}

static inline idx_t SelectNotNull(Vector &left, Vector &right, const idx_t count, const SelectionVector &sel,
                                  SelectionVector &maybe_vec, OptionalSelection &false_opt) {

	UnifiedVectorFormat lvdata, rvdata;
	left.ToUnifiedFormat(count, lvdata);
	right.ToUnifiedFormat(count, rvdata);

	auto &lmask = lvdata.validity;
	auto &rmask = rvdata.validity;

	// For top-level comparisons, NULL semantics are in effect,
	// so filter out any NULLs
	idx_t remaining = 0;
	if (lmask.AllValid() && rmask.AllValid()) {
		//	None are NULL, distinguish values.
		for (idx_t i = 0; i < count; ++i) {
			const auto idx = sel.get_index(i);
			maybe_vec.set_index(remaining++, idx);
		}
		return remaining;
	}

	// Slice the Vectors down to the rows that are not determined (i.e., neither is NULL)
	SelectionVector slicer(count);
	idx_t false_count = 0;
	for (idx_t i = 0; i < count; ++i) {
		const auto result_idx = sel.get_index(i);
		const auto lidx = lvdata.sel->get_index(i);
		const auto ridx = rvdata.sel->get_index(i);
		if (!lmask.RowIsValid(lidx) || !rmask.RowIsValid(ridx)) {
			false_opt.Append(false_count, result_idx);
		} else {
			//	Neither is NULL, distinguish values.
			slicer.set_index(remaining, i);
			maybe_vec.set_index(remaining++, result_idx);
		}
	}
	false_opt.Advance(false_count);

	if (remaining && remaining < count) {
		left.Slice(slicer, remaining);
		right.Slice(slicer, remaining);
	}

	return remaining;
}

static void ScatterSelection(SelectionVector *target, const idx_t count, const SelectionVector &dense_vec) {
	if (target) {
		for (idx_t i = 0; i < count; ++i) {
			target->set_index(i, dense_vec.get_index(i));
		}
	}
}

template <typename OP>
static idx_t NestedSelectOperation(Vector &left, Vector &right, const SelectionVector *sel, idx_t count,
                                   SelectionVector *true_sel, SelectionVector *false_sel) {
	// The Select operations all use a dense pair of input vectors to partition
	// a selection vector in a single pass. But to implement progressive comparisons,
	// we have to make multiple passes, so we need to keep track of the original input positions
	// and then scatter the output selections when we are done.
	if (!sel) {
		sel = FlatVector::IncrementalSelectionVector();
	}

	// Make buffered selections for progressive comparisons
	// TODO: Remove unnecessary allocations
	SelectionVector true_vec(count);
	OptionalSelection true_opt(&true_vec);

	SelectionVector false_vec(count);
	OptionalSelection false_opt(&false_vec);

	SelectionVector maybe_vec(count);

	// Handle NULL nested values
	Vector l_not_null(left);
	Vector r_not_null(right);

	auto match_count = SelectNotNull(l_not_null, r_not_null, count, *sel, maybe_vec, false_opt);
	auto no_match_count = count - match_count;
	count = match_count;

	//	Now that we have handled the NULLs, we can use the recursive nested comparator for the rest.
	match_count = NestedSelector::Select<OP>(l_not_null, r_not_null, maybe_vec, count, true_opt, false_opt);
	no_match_count += (count - match_count);

	// Copy the buffered selections to the output selections
	ScatterSelection(true_sel, match_count, true_vec);
	ScatterSelection(false_sel, no_match_count, false_vec);

	return match_count;
}

idx_t VectorOperations::Equals(Vector &left, Vector &right, const SelectionVector *sel, idx_t count,
                               SelectionVector *true_sel, SelectionVector *false_sel) {
	return TemplatedSelectOperation<duckdb::Equals>(left, right, sel, count, true_sel, false_sel);
}

idx_t VectorOperations::NotEquals(Vector &left, Vector &right, const SelectionVector *sel, idx_t count,
                                  SelectionVector *true_sel, SelectionVector *false_sel) {
	return TemplatedSelectOperation<duckdb::NotEquals>(left, right, sel, count, true_sel, false_sel);
}

idx_t VectorOperations::GreaterThan(Vector &left, Vector &right, const SelectionVector *sel, idx_t count,
                                    SelectionVector *true_sel, SelectionVector *false_sel) {
	return TemplatedSelectOperation<duckdb::GreaterThan>(left, right, sel, count, true_sel, false_sel);
}

idx_t VectorOperations::GreaterThanEquals(Vector &left, Vector &right, const SelectionVector *sel, idx_t count,
                                          SelectionVector *true_sel, SelectionVector *false_sel) {
	return TemplatedSelectOperation<duckdb::GreaterThanEquals>(left, right, sel, count, true_sel, false_sel);
}

idx_t VectorOperations::LessThan(Vector &left, Vector &right, const SelectionVector *sel, idx_t count,
                                 SelectionVector *true_sel, SelectionVector *false_sel) {
	return TemplatedSelectOperation<duckdb::GreaterThan>(right, left, sel, count, true_sel, false_sel);
}

idx_t VectorOperations::LessThanEquals(Vector &left, Vector &right, const SelectionVector *sel, idx_t count,
                                       SelectionVector *true_sel, SelectionVector *false_sel) {
	return TemplatedSelectOperation<duckdb::GreaterThanEquals>(right, left, sel, count, true_sel, false_sel);
}

idx_t ExpressionExecutor::Select(const BoundComparisonExpression &expr, ExpressionState *state,
                                 const SelectionVector *sel, idx_t count, SelectionVector *true_sel,
                                 SelectionVector *false_sel) {
	// resolve the children
	state->intermediate_chunk.Reset();
	auto &left = state->intermediate_chunk.data[0];
	auto &right = state->intermediate_chunk.data[1];

	Execute(*expr.left, state->child_states[0].get(), sel, count, left);
	Execute(*expr.right, state->child_states[1].get(), sel, count, right);

	switch (expr.type) {
	case ExpressionType::COMPARE_EQUAL:
		return VectorOperations::Equals(left, right, sel, count, true_sel, false_sel);
	case ExpressionType::COMPARE_NOTEQUAL:
		return VectorOperations::NotEquals(left, right, sel, count, true_sel, false_sel);
	case ExpressionType::COMPARE_LESSTHAN:
		return VectorOperations::LessThan(left, right, sel, count, true_sel, false_sel);
	case ExpressionType::COMPARE_GREATERTHAN:
		return VectorOperations::GreaterThan(left, right, sel, count, true_sel, false_sel);
	case ExpressionType::COMPARE_LESSTHANOREQUALTO:
		return VectorOperations::LessThanEquals(left, right, sel, count, true_sel, false_sel);
	case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
		return VectorOperations::GreaterThanEquals(left, right, sel, count, true_sel, false_sel);
	case ExpressionType::COMPARE_DISTINCT_FROM:
		return VectorOperations::DistinctFrom(left, right, sel, count, true_sel, false_sel);
	case ExpressionType::COMPARE_NOT_DISTINCT_FROM:
		return VectorOperations::NotDistinctFrom(left, right, sel, count, true_sel, false_sel);
	default:
		throw InternalException("Unknown comparison type!");
	}
}

} // namespace duckdb






#include <random>

namespace duckdb {

struct ConjunctionState : public ExpressionState {
	ConjunctionState(const Expression &expr, ExpressionExecutorState &root) : ExpressionState(expr, root) {
		adaptive_filter = make_uniq<AdaptiveFilter>(expr);
	}
	unique_ptr<AdaptiveFilter> adaptive_filter;
};

unique_ptr<ExpressionState> ExpressionExecutor::InitializeState(const BoundConjunctionExpression &expr,
                                                                ExpressionExecutorState &root) {
	auto result = make_uniq<ConjunctionState>(expr, root);
	for (auto &child : expr.children) {
		result->AddChild(child.get());
	}
	result->Finalize();
	return std::move(result);
}

void ExpressionExecutor::Execute(const BoundConjunctionExpression &expr, ExpressionState *state,
                                 const SelectionVector *sel, idx_t count, Vector &result) {
	// execute the children
	state->intermediate_chunk.Reset();
	for (idx_t i = 0; i < expr.children.size(); i++) {
		auto &current_result = state->intermediate_chunk.data[i];
		Execute(*expr.children[i], state->child_states[i].get(), sel, count, current_result);
		if (i == 0) {
			// move the result
			result.Reference(current_result);
		} else {
			Vector intermediate(LogicalType::BOOLEAN);
			// AND/OR together
			switch (expr.type) {
			case ExpressionType::CONJUNCTION_AND:
				VectorOperations::And(current_result, result, intermediate, count);
				break;
			case ExpressionType::CONJUNCTION_OR:
				VectorOperations::Or(current_result, result, intermediate, count);
				break;
			default:
				throw InternalException("Unknown conjunction type!");
			}
			result.Reference(intermediate);
		}
	}
}

idx_t ExpressionExecutor::Select(const BoundConjunctionExpression &expr, ExpressionState *state_p,
                                 const SelectionVector *sel, idx_t count, SelectionVector *true_sel,
                                 SelectionVector *false_sel) {
	auto state = (ConjunctionState *)state_p;

	if (expr.type == ExpressionType::CONJUNCTION_AND) {
		// get runtime statistics
		auto start_time = high_resolution_clock::now();

		const SelectionVector *current_sel = sel;
		idx_t current_count = count;
		idx_t false_count = 0;

		unique_ptr<SelectionVector> temp_true, temp_false;
		if (false_sel) {
			temp_false = make_uniq<SelectionVector>(STANDARD_VECTOR_SIZE);
		}
		if (!true_sel) {
			temp_true = make_uniq<SelectionVector>(STANDARD_VECTOR_SIZE);
			true_sel = temp_true.get();
		}
		for (idx_t i = 0; i < expr.children.size(); i++) {
			idx_t tcount = Select(*expr.children[state->adaptive_filter->permutation[i]],
			                      state->child_states[state->adaptive_filter->permutation[i]].get(), current_sel,
			                      current_count, true_sel, temp_false.get());
			idx_t fcount = current_count - tcount;
			if (fcount > 0 && false_sel) {
				// move failing tuples into the false_sel
				// tuples passed, move them into the actual result vector
				for (idx_t i = 0; i < fcount; i++) {
					false_sel->set_index(false_count++, temp_false->get_index(i));
				}
			}
			current_count = tcount;
			if (current_count == 0) {
				break;
			}
			if (current_count < count) {
				// tuples were filtered out: move on to using the true_sel to only evaluate passing tuples in subsequent
				// iterations
				current_sel = true_sel;
			}
		}

		// adapt runtime statistics
		auto end_time = high_resolution_clock::now();
		state->adaptive_filter->AdaptRuntimeStatistics(duration_cast<duration<double>>(end_time - start_time).count());
		return current_count;
	} else {
		// get runtime statistics
		auto start_time = high_resolution_clock::now();

		const SelectionVector *current_sel = sel;
		idx_t current_count = count;
		idx_t result_count = 0;

		unique_ptr<SelectionVector> temp_true, temp_false;
		if (true_sel) {
			temp_true = make_uniq<SelectionVector>(STANDARD_VECTOR_SIZE);
		}
		if (!false_sel) {
			temp_false = make_uniq<SelectionVector>(STANDARD_VECTOR_SIZE);
			false_sel = temp_false.get();
		}
		for (idx_t i = 0; i < expr.children.size(); i++) {
			idx_t tcount = Select(*expr.children[state->adaptive_filter->permutation[i]],
			                      state->child_states[state->adaptive_filter->permutation[i]].get(), current_sel,
			                      current_count, temp_true.get(), false_sel);
			if (tcount > 0) {
				if (true_sel) {
					// tuples passed, move them into the actual result vector
					for (idx_t i = 0; i < tcount; i++) {
						true_sel->set_index(result_count++, temp_true->get_index(i));
					}
				}
				// now move on to check only the non-passing tuples
				current_count -= tcount;
				current_sel = false_sel;
			}
		}

		// adapt runtime statistics
		auto end_time = high_resolution_clock::now();
		state->adaptive_filter->AdaptRuntimeStatistics(duration_cast<duration<double>>(end_time - start_time).count());
		return result_count;
	}
}

} // namespace duckdb




namespace duckdb {

unique_ptr<ExpressionState> ExpressionExecutor::InitializeState(const BoundConstantExpression &expr,
                                                                ExpressionExecutorState &root) {
	auto result = make_uniq<ExpressionState>(expr, root);
	result->Finalize();
	return result;
}

void ExpressionExecutor::Execute(const BoundConstantExpression &expr, ExpressionState *state,
                                 const SelectionVector *sel, idx_t count, Vector &result) {
	D_ASSERT(expr.value.type() == expr.return_type);
	result.Reference(expr.value);
}

} // namespace duckdb



namespace duckdb {

ExecuteFunctionState::ExecuteFunctionState(const Expression &expr, ExpressionExecutorState &root)
    : ExpressionState(expr, root) {
}

ExecuteFunctionState::~ExecuteFunctionState() {
}

unique_ptr<ExpressionState> ExpressionExecutor::InitializeState(const BoundFunctionExpression &expr,
                                                                ExpressionExecutorState &root) {
	auto result = make_uniq<ExecuteFunctionState>(expr, root);
	for (auto &child : expr.children) {
		result->AddChild(child.get());
	}
	result->Finalize();
	if (expr.function.init_local_state) {
		result->local_state = expr.function.init_local_state(*result, expr, expr.bind_info.get());
	}
	return std::move(result);
}

static void VerifyNullHandling(const BoundFunctionExpression &expr, DataChunk &args, Vector &result) {
#ifdef DEBUG
	if (args.data.empty() || expr.function.null_handling != FunctionNullHandling::DEFAULT_NULL_HANDLING) {
		return;
	}

	// Combine all the argument validity masks into a flat validity mask
	idx_t count = args.size();
	ValidityMask combined_mask(count);
	for (auto &arg : args.data) {
		UnifiedVectorFormat arg_data;
		arg.ToUnifiedFormat(count, arg_data);

		for (idx_t i = 0; i < count; i++) {
			auto idx = arg_data.sel->get_index(i);
			if (!arg_data.validity.RowIsValid(idx)) {
				combined_mask.SetInvalid(i);
			}
		}
	}

	// Default is that if any of the arguments are NULL, the result is also NULL
	UnifiedVectorFormat result_data;
	result.ToUnifiedFormat(count, result_data);
	for (idx_t i = 0; i < count; i++) {
		if (!combined_mask.RowIsValid(i)) {
			auto idx = result_data.sel->get_index(i);
			D_ASSERT(!result_data.validity.RowIsValid(idx));
		}
	}
#endif
}

void ExpressionExecutor::Execute(const BoundFunctionExpression &expr, ExpressionState *state,
                                 const SelectionVector *sel, idx_t count, Vector &result) {
	state->intermediate_chunk.Reset();
	auto &arguments = state->intermediate_chunk;
	if (!state->types.empty()) {
		for (idx_t i = 0; i < expr.children.size(); i++) {
			D_ASSERT(state->types[i] == expr.children[i]->return_type);
			Execute(*expr.children[i], state->child_states[i].get(), sel, count, arguments.data[i]);
#ifdef DEBUG
			if (expr.children[i]->return_type.id() == LogicalTypeId::VARCHAR) {
				arguments.data[i].UTFVerify(count);
			}
#endif
		}
		arguments.Verify();
	}
	arguments.SetCardinality(count);

	state->profiler.BeginSample();
	D_ASSERT(expr.function.function);
	expr.function.function(arguments, *state, result);
	state->profiler.EndSample(count);

	VerifyNullHandling(expr, arguments, result);
	D_ASSERT(result.GetType() == expr.return_type);
}

} // namespace duckdb




namespace duckdb {

unique_ptr<ExpressionState> ExpressionExecutor::InitializeState(const BoundOperatorExpression &expr,
                                                                ExpressionExecutorState &root) {
	auto result = make_uniq<ExpressionState>(expr, root);
	for (auto &child : expr.children) {
		result->AddChild(child.get());
	}
	result->Finalize();
	return result;
}

void ExpressionExecutor::Execute(const BoundOperatorExpression &expr, ExpressionState *state,
                                 const SelectionVector *sel, idx_t count, Vector &result) {
	// special handling for special snowflake 'IN'
	// IN has n children
	if (expr.type == ExpressionType::COMPARE_IN || expr.type == ExpressionType::COMPARE_NOT_IN) {
		if (expr.children.size() < 2) {
			throw InvalidInputException("IN needs at least two children");
		}

		Vector left(expr.children[0]->return_type);
		// eval left side
		Execute(*expr.children[0], state->child_states[0].get(), sel, count, left);

		// init result to false
		Vector intermediate(LogicalType::BOOLEAN);
		Value false_val = Value::BOOLEAN(false);
		intermediate.Reference(false_val);

		// in rhs is a list of constants
		// for every child, OR the result of the comparision with the left
		// to get the overall result.
		for (idx_t child = 1; child < expr.children.size(); child++) {
			Vector vector_to_check(expr.children[child]->return_type);
			Vector comp_res(LogicalType::BOOLEAN);

			Execute(*expr.children[child], state->child_states[child].get(), sel, count, vector_to_check);
			VectorOperations::Equals(left, vector_to_check, comp_res, count);

			if (child == 1) {
				// first child: move to result
				intermediate.Reference(comp_res);
			} else {
				// otherwise OR together
				Vector new_result(LogicalType::BOOLEAN, true, false);
				VectorOperations::Or(intermediate, comp_res, new_result, count);
				intermediate.Reference(new_result);
			}
		}
		if (expr.type == ExpressionType::COMPARE_NOT_IN) {
			// NOT IN: invert result
			VectorOperations::Not(intermediate, result, count);
		} else {
			// directly use the result
			result.Reference(intermediate);
		}
	} else if (expr.type == ExpressionType::OPERATOR_COALESCE) {
		SelectionVector sel_a(count);
		SelectionVector sel_b(count);
		SelectionVector slice_sel(count);
		SelectionVector result_sel(count);
		SelectionVector *next_sel = &sel_a;
		const SelectionVector *current_sel = sel;
		idx_t remaining_count = count;
		idx_t next_count;
		for (idx_t child = 0; child < expr.children.size(); child++) {
			Vector vector_to_check(expr.children[child]->return_type);
			Execute(*expr.children[child], state->child_states[child].get(), current_sel, remaining_count,
			        vector_to_check);

			UnifiedVectorFormat vdata;
			vector_to_check.ToUnifiedFormat(remaining_count, vdata);

			idx_t result_count = 0;
			next_count = 0;
			for (idx_t i = 0; i < remaining_count; i++) {
				auto base_idx = current_sel ? current_sel->get_index(i) : i;
				auto idx = vdata.sel->get_index(i);
				if (vdata.validity.RowIsValid(idx)) {
					slice_sel.set_index(result_count, i);
					result_sel.set_index(result_count++, base_idx);
				} else {
					next_sel->set_index(next_count++, base_idx);
				}
			}
			if (result_count > 0) {
				vector_to_check.Slice(slice_sel, result_count);
				FillSwitch(vector_to_check, result, result_sel, result_count);
			}
			current_sel = next_sel;
			next_sel = next_sel == &sel_a ? &sel_b : &sel_a;
			remaining_count = next_count;
			if (next_count == 0) {
				break;
			}
		}
		if (remaining_count > 0) {
			for (idx_t i = 0; i < remaining_count; i++) {
				FlatVector::SetNull(result, current_sel->get_index(i), true);
			}
		}
		if (sel) {
			result.Slice(*sel, count);
		} else if (count == 1) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
		}
	} else if (expr.children.size() == 1) {
		state->intermediate_chunk.Reset();
		auto &child = state->intermediate_chunk.data[0];

		Execute(*expr.children[0], state->child_states[0].get(), sel, count, child);
		switch (expr.type) {
		case ExpressionType::OPERATOR_NOT: {
			VectorOperations::Not(child, result, count);
			break;
		}
		case ExpressionType::OPERATOR_IS_NULL: {
			VectorOperations::IsNull(child, result, count);
			break;
		}
		case ExpressionType::OPERATOR_IS_NOT_NULL: {
			VectorOperations::IsNotNull(child, result, count);
			break;
		}
		default:
			throw NotImplementedException("Unsupported operator type with 1 child!");
		}
	} else {
		throw NotImplementedException("operator");
	}
}

} // namespace duckdb




namespace duckdb {

unique_ptr<ExpressionState> ExpressionExecutor::InitializeState(const BoundParameterExpression &expr,
                                                                ExpressionExecutorState &root) {
	auto result = make_uniq<ExpressionState>(expr, root);
	result->Finalize();
	return result;
}

void ExpressionExecutor::Execute(const BoundParameterExpression &expr, ExpressionState *state,
                                 const SelectionVector *sel, idx_t count, Vector &result) {
	D_ASSERT(expr.parameter_data);
	D_ASSERT(expr.parameter_data->return_type == expr.return_type);
	D_ASSERT(expr.parameter_data->value.type() == expr.return_type);
	result.Reference(expr.parameter_data->value);
}

} // namespace duckdb



namespace duckdb {

unique_ptr<ExpressionState> ExpressionExecutor::InitializeState(const BoundReferenceExpression &expr,
                                                                ExpressionExecutorState &root) {
	auto result = make_uniq<ExpressionState>(expr, root);
	result->Finalize();
	return result;
}

void ExpressionExecutor::Execute(const BoundReferenceExpression &expr, ExpressionState *state,
                                 const SelectionVector *sel, idx_t count, Vector &result) {
	D_ASSERT(expr.index != DConstants::INVALID_INDEX);
	D_ASSERT(expr.index < chunk->ColumnCount());

	if (sel) {
		result.Slice(chunk->data[expr.index], *sel, count);
	} else {
		result.Reference(chunk->data[expr.index]);
	}
}

} // namespace duckdb







namespace duckdb {

ExpressionExecutor::ExpressionExecutor(ClientContext &context) : context(&context) {
}

ExpressionExecutor::ExpressionExecutor(ClientContext &context, const Expression *expression)
    : ExpressionExecutor(context) {
	D_ASSERT(expression);
	AddExpression(*expression);
}

ExpressionExecutor::ExpressionExecutor(ClientContext &context, const Expression &expression)
    : ExpressionExecutor(context) {
	AddExpression(expression);
}

ExpressionExecutor::ExpressionExecutor(ClientContext &context, const vector<unique_ptr<Expression>> &exprs)
    : ExpressionExecutor(context) {
	D_ASSERT(exprs.size() > 0);
	for (auto &expr : exprs) {
		AddExpression(*expr);
	}
}

ExpressionExecutor::ExpressionExecutor(const vector<unique_ptr<Expression>> &exprs) : context(nullptr) {
	D_ASSERT(exprs.size() > 0);
	for (auto &expr : exprs) {
		AddExpression(*expr);
	}
}

ExpressionExecutor::ExpressionExecutor() : context(nullptr) {
}

bool ExpressionExecutor::HasContext() {
	return context;
}

ClientContext &ExpressionExecutor::GetContext() {
	if (!context) {
		throw InternalException("Calling ExpressionExecutor::GetContext on an expression executor without a context");
	}
	return *context;
}

Allocator &ExpressionExecutor::GetAllocator() {
	return context ? Allocator::Get(*context) : Allocator::DefaultAllocator();
}

void ExpressionExecutor::AddExpression(const Expression &expr) {
	expressions.push_back(&expr);
	auto state = make_uniq<ExpressionExecutorState>();
	Initialize(expr, *state);
	state->Verify();
	states.push_back(std::move(state));
}

void ExpressionExecutor::Initialize(const Expression &expression, ExpressionExecutorState &state) {
	state.executor = this;
	state.root_state = InitializeState(expression, state);
}

void ExpressionExecutor::Execute(DataChunk *input, DataChunk &result) {
	SetChunk(input);
	D_ASSERT(expressions.size() == result.ColumnCount());
	D_ASSERT(!expressions.empty());

	for (idx_t i = 0; i < expressions.size(); i++) {
		ExecuteExpression(i, result.data[i]);
	}
	result.SetCardinality(input ? input->size() : 1);
	result.Verify();
}

void ExpressionExecutor::ExecuteExpression(DataChunk &input, Vector &result) {
	SetChunk(&input);
	ExecuteExpression(result);
}

idx_t ExpressionExecutor::SelectExpression(DataChunk &input, SelectionVector &sel) {
	D_ASSERT(expressions.size() == 1);
	SetChunk(&input);
	states[0]->profiler.BeginSample();
	idx_t selected_tuples = Select(*expressions[0], states[0]->root_state.get(), nullptr, input.size(), &sel, nullptr);
	states[0]->profiler.EndSample(chunk ? chunk->size() : 0);
	return selected_tuples;
}

void ExpressionExecutor::ExecuteExpression(Vector &result) {
	D_ASSERT(expressions.size() == 1);
	ExecuteExpression(0, result);
}

void ExpressionExecutor::ExecuteExpression(idx_t expr_idx, Vector &result) {
	D_ASSERT(expr_idx < expressions.size());
	D_ASSERT(result.GetType().id() == expressions[expr_idx]->return_type.id());
	states[expr_idx]->profiler.BeginSample();
	Execute(*expressions[expr_idx], states[expr_idx]->root_state.get(), nullptr, chunk ? chunk->size() : 1, result);
	states[expr_idx]->profiler.EndSample(chunk ? chunk->size() : 0);
}

Value ExpressionExecutor::EvaluateScalar(ClientContext &context, const Expression &expr, bool allow_unfoldable) {
	D_ASSERT(allow_unfoldable || expr.IsFoldable());
	D_ASSERT(expr.IsScalar());
	// use an ExpressionExecutor to execute the expression
	ExpressionExecutor executor(context, expr);

	Vector result(expr.return_type);
	executor.ExecuteExpression(result);

	D_ASSERT(allow_unfoldable || result.GetVectorType() == VectorType::CONSTANT_VECTOR);
	auto result_value = result.GetValue(0);
	D_ASSERT(result_value.type().InternalType() == expr.return_type.InternalType());
	return result_value;
}

bool ExpressionExecutor::TryEvaluateScalar(ClientContext &context, const Expression &expr, Value &result) {
	try {
		result = EvaluateScalar(context, expr);
		return true;
	} catch (InternalException &ex) {
		throw ex;
	} catch (...) {
		return false;
	}
}

void ExpressionExecutor::Verify(const Expression &expr, Vector &vector, idx_t count) {
	D_ASSERT(expr.return_type.id() == vector.GetType().id());
	vector.Verify(count);
	if (expr.verification_stats) {
		expr.verification_stats->Verify(vector, count);
	}
}

unique_ptr<ExpressionState> ExpressionExecutor::InitializeState(const Expression &expr,
                                                                ExpressionExecutorState &state) {
	switch (expr.expression_class) {
	case ExpressionClass::BOUND_REF:
		return InitializeState((const BoundReferenceExpression &)expr, state);
	case ExpressionClass::BOUND_BETWEEN:
		return InitializeState((const BoundBetweenExpression &)expr, state);
	case ExpressionClass::BOUND_CASE:
		return InitializeState((const BoundCaseExpression &)expr, state);
	case ExpressionClass::BOUND_CAST:
		return InitializeState((const BoundCastExpression &)expr, state);
	case ExpressionClass::BOUND_COMPARISON:
		return InitializeState((const BoundComparisonExpression &)expr, state);
	case ExpressionClass::BOUND_CONJUNCTION:
		return InitializeState((const BoundConjunctionExpression &)expr, state);
	case ExpressionClass::BOUND_CONSTANT:
		return InitializeState((const BoundConstantExpression &)expr, state);
	case ExpressionClass::BOUND_FUNCTION:
		return InitializeState((const BoundFunctionExpression &)expr, state);
	case ExpressionClass::BOUND_OPERATOR:
		return InitializeState((const BoundOperatorExpression &)expr, state);
	case ExpressionClass::BOUND_PARAMETER:
		return InitializeState((const BoundParameterExpression &)expr, state);
	default:
		throw InternalException("Attempting to initialize state of expression of unknown type!");
	}
}

void ExpressionExecutor::Execute(const Expression &expr, ExpressionState *state, const SelectionVector *sel,
                                 idx_t count, Vector &result) {
#ifdef DEBUG
	//! The result Vector must be "clean"
	if (result.GetVectorType() == VectorType::FLAT_VECTOR) {
		D_ASSERT(FlatVector::Validity(result).CheckAllValid(count));
	}
#endif

	if (count == 0) {
		return;
	}
	switch (expr.expression_class) {
	case ExpressionClass::BOUND_BETWEEN:
		Execute((const BoundBetweenExpression &)expr, state, sel, count, result);
		break;
	case ExpressionClass::BOUND_REF:
		Execute((const BoundReferenceExpression &)expr, state, sel, count, result);
		break;
	case ExpressionClass::BOUND_CASE:
		Execute((const BoundCaseExpression &)expr, state, sel, count, result);
		break;
	case ExpressionClass::BOUND_CAST:
		Execute((const BoundCastExpression &)expr, state, sel, count, result);
		break;
	case ExpressionClass::BOUND_COMPARISON:
		Execute((const BoundComparisonExpression &)expr, state, sel, count, result);
		break;
	case ExpressionClass::BOUND_CONJUNCTION:
		Execute((const BoundConjunctionExpression &)expr, state, sel, count, result);
		break;
	case ExpressionClass::BOUND_CONSTANT:
		Execute((const BoundConstantExpression &)expr, state, sel, count, result);
		break;
	case ExpressionClass::BOUND_FUNCTION:
		Execute((const BoundFunctionExpression &)expr, state, sel, count, result);
		break;
	case ExpressionClass::BOUND_OPERATOR:
		Execute((const BoundOperatorExpression &)expr, state, sel, count, result);
		break;
	case ExpressionClass::BOUND_PARAMETER:
		Execute((const BoundParameterExpression &)expr, state, sel, count, result);
		break;
	default:
		throw InternalException("Attempting to execute expression of unknown type!");
	}
	Verify(expr, result, count);
}

idx_t ExpressionExecutor::Select(const Expression &expr, ExpressionState *state, const SelectionVector *sel,
                                 idx_t count, SelectionVector *true_sel, SelectionVector *false_sel) {
	if (count == 0) {
		return 0;
	}
	D_ASSERT(true_sel || false_sel);
	D_ASSERT(expr.return_type.id() == LogicalTypeId::BOOLEAN);
	switch (expr.expression_class) {
	case ExpressionClass::BOUND_BETWEEN:
		return Select(expr.Cast<BoundBetweenExpression>(), state, sel, count, true_sel, false_sel);
	case ExpressionClass::BOUND_COMPARISON:
		return Select(expr.Cast<BoundComparisonExpression>(), state, sel, count, true_sel, false_sel);
	case ExpressionClass::BOUND_CONJUNCTION:
		return Select(expr.Cast<BoundConjunctionExpression>(), state, sel, count, true_sel, false_sel);
	default:
		return DefaultSelect(expr, state, sel, count, true_sel, false_sel);
	}
}

template <bool NO_NULL, bool HAS_TRUE_SEL, bool HAS_FALSE_SEL>
static inline idx_t DefaultSelectLoop(const SelectionVector *bsel, uint8_t *__restrict bdata, ValidityMask &mask,
                                      const SelectionVector *sel, idx_t count, SelectionVector *true_sel,
                                      SelectionVector *false_sel) {
	idx_t true_count = 0, false_count = 0;
	for (idx_t i = 0; i < count; i++) {
		auto bidx = bsel->get_index(i);
		auto result_idx = sel->get_index(i);
		if (bdata[bidx] > 0 && (NO_NULL || mask.RowIsValid(bidx))) {
			if (HAS_TRUE_SEL) {
				true_sel->set_index(true_count++, result_idx);
			}
		} else {
			if (HAS_FALSE_SEL) {
				false_sel->set_index(false_count++, result_idx);
			}
		}
	}
	if (HAS_TRUE_SEL) {
		return true_count;
	} else {
		return count - false_count;
	}
}

template <bool NO_NULL>
static inline idx_t DefaultSelectSwitch(UnifiedVectorFormat &idata, const SelectionVector *sel, idx_t count,
                                        SelectionVector *true_sel, SelectionVector *false_sel) {
	if (true_sel && false_sel) {
		return DefaultSelectLoop<NO_NULL, true, true>(idata.sel, (uint8_t *)idata.data, idata.validity, sel, count,
		                                              true_sel, false_sel);
	} else if (true_sel) {
		return DefaultSelectLoop<NO_NULL, true, false>(idata.sel, (uint8_t *)idata.data, idata.validity, sel, count,
		                                               true_sel, false_sel);
	} else {
		D_ASSERT(false_sel);
		return DefaultSelectLoop<NO_NULL, false, true>(idata.sel, (uint8_t *)idata.data, idata.validity, sel, count,
		                                               true_sel, false_sel);
	}
}

idx_t ExpressionExecutor::DefaultSelect(const Expression &expr, ExpressionState *state, const SelectionVector *sel,
                                        idx_t count, SelectionVector *true_sel, SelectionVector *false_sel) {
	// generic selection of boolean expression:
	// resolve the true/false expression first
	// then use that to generate the selection vector
	bool intermediate_bools[STANDARD_VECTOR_SIZE];
	Vector intermediate(LogicalType::BOOLEAN, (data_ptr_t)intermediate_bools);
	Execute(expr, state, sel, count, intermediate);

	UnifiedVectorFormat idata;
	intermediate.ToUnifiedFormat(count, idata);

	if (!sel) {
		sel = FlatVector::IncrementalSelectionVector();
	}
	if (!idata.validity.AllValid()) {
		return DefaultSelectSwitch<false>(idata, sel, count, true_sel, false_sel);
	} else {
		return DefaultSelectSwitch<true>(idata, sel, count, true_sel, false_sel);
	}
}

vector<unique_ptr<ExpressionExecutorState>> &ExpressionExecutor::GetStates() {
	return states;
}

} // namespace duckdb





namespace duckdb {

void ExpressionState::AddChild(Expression *expr) {
	types.push_back(expr->return_type);
	child_states.push_back(ExpressionExecutor::InitializeState(*expr, root));
}

void ExpressionState::Finalize() {
	if (!types.empty()) {
		intermediate_chunk.Initialize(GetAllocator(), types);
	}
}

Allocator &ExpressionState::GetAllocator() {
	return root.executor->GetAllocator();
}

bool ExpressionState::HasContext() {
	return root.executor->HasContext();
}

ClientContext &ExpressionState::GetContext() {
	if (!HasContext()) {
		throw BinderException("Cannot use %s in this context", (expr.Cast<BoundFunctionExpression>()).function.name);
	}
	return root.executor->GetContext();
}

ExpressionState::ExpressionState(const Expression &expr, ExpressionExecutorState &root) : expr(expr), root(root) {
}

ExpressionExecutorState::ExpressionExecutorState() : profiler() {
}

void ExpressionState::Verify(ExpressionExecutorState &root_executor) {
	D_ASSERT(&root_executor == &root);
	for (auto &entry : child_states) {
		entry->Verify(root_executor);
	}
}

void ExpressionExecutorState::Verify() {
	D_ASSERT(executor);
	root_state->Verify(*this);
}

} // namespace duckdb



















#include <algorithm>
#include <cstring>

namespace duckdb {

struct ARTIndexScanState : public IndexScanState {

	//! Scan predicates (single predicate scan or range scan)
	Value values[2];
	//! Expressions of the scan predicates
	ExpressionType expressions[2];
	bool checked = false;
	//! All scanned row IDs
	vector<row_t> result_ids;
	Iterator iterator;
};

ART::ART(const vector<column_t> &column_ids, TableIOManager &table_io_manager,
         const vector<unique_ptr<Expression>> &unbound_expressions, const IndexConstraintType constraint_type,
         AttachedDatabase &db, const idx_t block_id, const idx_t block_offset)

    : Index(db, IndexType::ART, table_io_manager, column_ids, unbound_expressions, constraint_type) {

	if (!Radix::IsLittleEndian()) {
		throw NotImplementedException("ART indexes are not supported on big endian architectures");
	}

	// initialize all allocators
	allocators.emplace_back(make_uniq<FixedSizeAllocator>(sizeof(PrefixSegment), buffer_manager.GetBufferAllocator()));
	allocators.emplace_back(make_uniq<FixedSizeAllocator>(sizeof(LeafSegment), buffer_manager.GetBufferAllocator()));
	allocators.emplace_back(make_uniq<FixedSizeAllocator>(sizeof(Leaf), buffer_manager.GetBufferAllocator()));
	allocators.emplace_back(make_uniq<FixedSizeAllocator>(sizeof(Node4), buffer_manager.GetBufferAllocator()));
	allocators.emplace_back(make_uniq<FixedSizeAllocator>(sizeof(Node16), buffer_manager.GetBufferAllocator()));
	allocators.emplace_back(make_uniq<FixedSizeAllocator>(sizeof(Node48), buffer_manager.GetBufferAllocator()));
	allocators.emplace_back(make_uniq<FixedSizeAllocator>(sizeof(Node256), buffer_manager.GetBufferAllocator()));

	// set the root node of the tree
	tree = make_uniq<Node>();
	if (block_id != DConstants::INVALID_INDEX) {
		tree->buffer_id = block_id;
		tree->offset = block_offset;
		tree->Deserialize(*this);
	}
	serialized_data_pointer = BlockPointer(block_id, block_offset);

	// validate the types of the key columns
	for (idx_t i = 0; i < types.size(); i++) {
		switch (types[i]) {
		case PhysicalType::BOOL:
		case PhysicalType::INT8:
		case PhysicalType::INT16:
		case PhysicalType::INT32:
		case PhysicalType::INT64:
		case PhysicalType::INT128:
		case PhysicalType::UINT8:
		case PhysicalType::UINT16:
		case PhysicalType::UINT32:
		case PhysicalType::UINT64:
		case PhysicalType::FLOAT:
		case PhysicalType::DOUBLE:
		case PhysicalType::VARCHAR:
			break;
		default:
			throw InvalidTypeException(logical_types[i], "Invalid type for index key.");
		}
	}
}

ART::~ART() {
	tree->Reset();
}

//===--------------------------------------------------------------------===//
// Initialize Predicate Scans
//===--------------------------------------------------------------------===//

unique_ptr<IndexScanState> ART::InitializeScanSinglePredicate(const Transaction &transaction, const Value &value,
                                                              const ExpressionType expression_type) {
	// initialize point lookup
	auto result = make_uniq<ARTIndexScanState>();
	result->values[0] = value;
	result->expressions[0] = expression_type;
	return std::move(result);
}

unique_ptr<IndexScanState> ART::InitializeScanTwoPredicates(const Transaction &transaction, const Value &low_value,
                                                            const ExpressionType low_expression_type,
                                                            const Value &high_value,
                                                            const ExpressionType high_expression_type) {
	// initialize range lookup
	auto result = make_uniq<ARTIndexScanState>();
	result->values[0] = low_value;
	result->expressions[0] = low_expression_type;
	result->values[1] = high_value;
	result->expressions[1] = high_expression_type;
	return std::move(result);
}

//===--------------------------------------------------------------------===//
// Keys
//===--------------------------------------------------------------------===//

template <class T>
static void TemplatedGenerateKeys(ArenaAllocator &allocator, Vector &input, idx_t count, vector<ARTKey> &keys) {
	UnifiedVectorFormat idata;
	input.ToUnifiedFormat(count, idata);

	D_ASSERT(keys.size() >= count);
	auto input_data = (T *)idata.data;
	for (idx_t i = 0; i < count; i++) {
		auto idx = idata.sel->get_index(i);
		if (idata.validity.RowIsValid(idx)) {
			ARTKey::CreateARTKey<T>(allocator, input.GetType(), keys[i], input_data[idx]);
		} else {
			// we need to possibly reset the former key value in the keys vector
			keys[i] = ARTKey();
		}
	}
}

template <class T>
static void ConcatenateKeys(ArenaAllocator &allocator, Vector &input, idx_t count, vector<ARTKey> &keys) {
	UnifiedVectorFormat idata;
	input.ToUnifiedFormat(count, idata);

	auto input_data = (T *)idata.data;
	for (idx_t i = 0; i < count; i++) {
		auto idx = idata.sel->get_index(i);

		// key is not NULL (no previous column entry was NULL)
		if (!keys[i].Empty()) {
			if (!idata.validity.RowIsValid(idx)) {
				// this column entry is NULL, set whole key to NULL
				keys[i] = ARTKey();
			} else {
				auto other_key = ARTKey::CreateARTKey<T>(allocator, input.GetType(), input_data[idx]);
				keys[i].ConcatenateARTKey(allocator, other_key);
			}
		}
	}
}

void ART::GenerateKeys(ArenaAllocator &allocator, DataChunk &input, vector<ARTKey> &keys) {
	// generate keys for the first input column
	switch (input.data[0].GetType().InternalType()) {
	case PhysicalType::BOOL:
		TemplatedGenerateKeys<bool>(allocator, input.data[0], input.size(), keys);
		break;
	case PhysicalType::INT8:
		TemplatedGenerateKeys<int8_t>(allocator, input.data[0], input.size(), keys);
		break;
	case PhysicalType::INT16:
		TemplatedGenerateKeys<int16_t>(allocator, input.data[0], input.size(), keys);
		break;
	case PhysicalType::INT32:
		TemplatedGenerateKeys<int32_t>(allocator, input.data[0], input.size(), keys);
		break;
	case PhysicalType::INT64:
		TemplatedGenerateKeys<int64_t>(allocator, input.data[0], input.size(), keys);
		break;
	case PhysicalType::INT128:
		TemplatedGenerateKeys<hugeint_t>(allocator, input.data[0], input.size(), keys);
		break;
	case PhysicalType::UINT8:
		TemplatedGenerateKeys<uint8_t>(allocator, input.data[0], input.size(), keys);
		break;
	case PhysicalType::UINT16:
		TemplatedGenerateKeys<uint16_t>(allocator, input.data[0], input.size(), keys);
		break;
	case PhysicalType::UINT32:
		TemplatedGenerateKeys<uint32_t>(allocator, input.data[0], input.size(), keys);
		break;
	case PhysicalType::UINT64:
		TemplatedGenerateKeys<uint64_t>(allocator, input.data[0], input.size(), keys);
		break;
	case PhysicalType::FLOAT:
		TemplatedGenerateKeys<float>(allocator, input.data[0], input.size(), keys);
		break;
	case PhysicalType::DOUBLE:
		TemplatedGenerateKeys<double>(allocator, input.data[0], input.size(), keys);
		break;
	case PhysicalType::VARCHAR:
		TemplatedGenerateKeys<string_t>(allocator, input.data[0], input.size(), keys);
		break;
	default:
		throw InternalException("Invalid type for index");
	}

	for (idx_t i = 1; i < input.ColumnCount(); i++) {
		// for each of the remaining columns, concatenate
		switch (input.data[i].GetType().InternalType()) {
		case PhysicalType::BOOL:
			ConcatenateKeys<bool>(allocator, input.data[i], input.size(), keys);
			break;
		case PhysicalType::INT8:
			ConcatenateKeys<int8_t>(allocator, input.data[i], input.size(), keys);
			break;
		case PhysicalType::INT16:
			ConcatenateKeys<int16_t>(allocator, input.data[i], input.size(), keys);
			break;
		case PhysicalType::INT32:
			ConcatenateKeys<int32_t>(allocator, input.data[i], input.size(), keys);
			break;
		case PhysicalType::INT64:
			ConcatenateKeys<int64_t>(allocator, input.data[i], input.size(), keys);
			break;
		case PhysicalType::INT128:
			ConcatenateKeys<hugeint_t>(allocator, input.data[i], input.size(), keys);
			break;
		case PhysicalType::UINT8:
			ConcatenateKeys<uint8_t>(allocator, input.data[i], input.size(), keys);
			break;
		case PhysicalType::UINT16:
			ConcatenateKeys<uint16_t>(allocator, input.data[i], input.size(), keys);
			break;
		case PhysicalType::UINT32:
			ConcatenateKeys<uint32_t>(allocator, input.data[i], input.size(), keys);
			break;
		case PhysicalType::UINT64:
			ConcatenateKeys<uint64_t>(allocator, input.data[i], input.size(), keys);
			break;
		case PhysicalType::FLOAT:
			ConcatenateKeys<float>(allocator, input.data[i], input.size(), keys);
			break;
		case PhysicalType::DOUBLE:
			ConcatenateKeys<double>(allocator, input.data[i], input.size(), keys);
			break;
		case PhysicalType::VARCHAR:
			ConcatenateKeys<string_t>(allocator, input.data[i], input.size(), keys);
			break;
		default:
			throw InternalException("Invalid type for index");
		}
	}
}

//===--------------------------------------------------------------------===//
// Construct from sorted data (only during CREATE (UNIQUE) INDEX statements)
//===--------------------------------------------------------------------===//

struct KeySection {
	KeySection(idx_t start_p, idx_t end_p, idx_t depth_p, data_t key_byte_p)
	    : start(start_p), end(end_p), depth(depth_p), key_byte(key_byte_p) {};
	KeySection(idx_t start_p, idx_t end_p, vector<ARTKey> &keys, KeySection &key_section)
	    : start(start_p), end(end_p), depth(key_section.depth + 1), key_byte(keys[end_p].data[key_section.depth]) {};
	idx_t start;
	idx_t end;
	idx_t depth;
	data_t key_byte;
};

void GetChildSections(vector<KeySection> &child_sections, vector<ARTKey> &keys, KeySection &key_section) {

	idx_t child_start_idx = key_section.start;
	for (idx_t i = key_section.start + 1; i <= key_section.end; i++) {
		if (keys[i - 1].data[key_section.depth] != keys[i].data[key_section.depth]) {
			child_sections.emplace_back(child_start_idx, i - 1, keys, key_section);
			child_start_idx = i;
		}
	}
	child_sections.emplace_back(child_start_idx, key_section.end, keys, key_section);
}

bool Construct(ART &art, vector<ARTKey> &keys, row_t *row_ids, Node &node, KeySection &key_section,
               bool &has_constraint) {

	D_ASSERT(key_section.start < keys.size());
	D_ASSERT(key_section.end < keys.size());
	D_ASSERT(key_section.start <= key_section.end);

	auto &start_key = keys[key_section.start];
	auto &end_key = keys[key_section.end];

	// increment the depth until we reach a leaf or find a mismatching byte
	auto prefix_start = key_section.depth;
	while (start_key.len != key_section.depth && start_key.ByteMatches(end_key, key_section.depth)) {
		key_section.depth++;
	}

	// we reached a leaf, i.e. all the bytes of start_key and end_key match
	if (start_key.len == key_section.depth) {
		// end_idx is inclusive
		auto num_row_ids = key_section.end - key_section.start + 1;

		// check for possible constraint violation
		auto single_row_id = num_row_ids == 1;
		if (has_constraint && !single_row_id) {
			return false;
		}

		if (single_row_id) {
			Leaf::New(art, node, start_key, prefix_start, row_ids[key_section.start]);
		} else {
			Leaf::New(art, node, start_key, prefix_start, row_ids + key_section.start, num_row_ids);
		}
		return true;
	}

	// create a new node and recurse

	// we will find at least two child entries of this node, otherwise we'd have reached a leaf
	vector<KeySection> child_sections;
	GetChildSections(child_sections, keys, key_section);

	auto node_type = Node::GetARTNodeTypeByCount(child_sections.size());
	Node::New(art, node, node_type);

	auto prefix_length = key_section.depth - prefix_start;
	node.GetPrefix(art).Initialize(art, start_key, prefix_start, prefix_length);

	// recurse on each child section
	for (auto &child_section : child_sections) {
		Node new_child;
		auto no_violation = Construct(art, keys, row_ids, new_child, child_section, has_constraint);
		Node::InsertChild(art, node, child_section.key_byte, new_child);
		if (!no_violation) {
			return false;
		}
	}
	return true;
}

bool ART::ConstructFromSorted(idx_t count, vector<ARTKey> &keys, Vector &row_identifiers) {

	// prepare the row_identifiers
	row_identifiers.Flatten(count);
	auto row_ids = FlatVector::GetData<row_t>(row_identifiers);

	auto key_section = KeySection(0, count - 1, 0, 0);
	auto has_constraint = IsUnique();
	return Construct(*this, keys, row_ids, *this->tree, key_section, has_constraint);
}

//===--------------------------------------------------------------------===//
// Insert / Verification / Constraint Checking
//===--------------------------------------------------------------------===//
PreservedError ART::Insert(IndexLock &lock, DataChunk &input, Vector &row_ids) {

	D_ASSERT(row_ids.GetType().InternalType() == ROW_TYPE);
	D_ASSERT(logical_types[0] == input.data[0].GetType());

	// generate the keys for the given input
	ArenaAllocator arena_allocator(BufferAllocator::Get(db));
	vector<ARTKey> keys(input.size());
	GenerateKeys(arena_allocator, input, keys);

	// get the corresponding row IDs
	row_ids.Flatten(input.size());
	auto row_identifiers = FlatVector::GetData<row_t>(row_ids);

	// now insert the elements into the index
	idx_t failed_index = DConstants::INVALID_INDEX;
	for (idx_t i = 0; i < input.size(); i++) {
		if (keys[i].Empty()) {
			continue;
		}

		row_t row_id = row_identifiers[i];
		if (!Insert(*tree, keys[i], 0, row_id)) {
			// failed to insert because of constraint violation
			failed_index = i;
			break;
		}
	}

	// failed to insert because of constraint violation: remove previously inserted entries
	if (failed_index != DConstants::INVALID_INDEX) {
		for (idx_t i = 0; i < failed_index; i++) {
			if (keys[i].Empty()) {
				continue;
			}
			row_t row_id = row_identifiers[i];
			Erase(*tree, keys[i], 0, row_id);
		}
	}

	if (failed_index != DConstants::INVALID_INDEX) {
		return PreservedError(ConstraintException("PRIMARY KEY or UNIQUE constraint violated: duplicate key \"%s\"",
		                                          AppendRowError(input, failed_index)));
	}
	return PreservedError();
}

PreservedError ART::Append(IndexLock &lock, DataChunk &appended_data, Vector &row_identifiers) {
	DataChunk expression_result;
	expression_result.Initialize(Allocator::DefaultAllocator(), logical_types);

	// first resolve the expressions for the index
	ExecuteExpressions(appended_data, expression_result);

	// now insert into the index
	return Insert(lock, expression_result, row_identifiers);
}

void ART::VerifyAppend(DataChunk &chunk) {
	ConflictManager conflict_manager(VerifyExistenceType::APPEND, chunk.size());
	CheckConstraintsForChunk(chunk, conflict_manager);
}

void ART::VerifyAppend(DataChunk &chunk, ConflictManager &conflict_manager) {
	D_ASSERT(conflict_manager.LookupType() == VerifyExistenceType::APPEND);
	CheckConstraintsForChunk(chunk, conflict_manager);
}

bool ART::InsertToLeaf(Node &leaf_node, const row_t &row_id) {

	auto &leaf = Leaf::Get(*this, leaf_node);

#ifdef DEBUG
	for (idx_t k = 0; k < leaf.count; k++) {
		D_ASSERT(leaf.GetRowId(*this, k) != row_id);
	}
#endif
	if (IsUnique() && leaf.count != 0) {
		return false;
	}
	leaf.Insert(*this, row_id);
	return true;
}

bool ART::Insert(Node &node, const ARTKey &key, idx_t depth, const row_t &row_id) {

	if (!node.IsSet()) {
		// node is currently empty, create a leaf here with the key
		Leaf::New(*this, node, key, depth, row_id);
		return true;
	}

	if (node.DecodeARTNodeType() == NType::LEAF) {

		// add a row ID to a leaf, if they have the same key
		auto &leaf = Leaf::Get(*this, node);
		auto mismatch_position = leaf.prefix.KeyMismatchPosition(*this, key, depth);
		if (mismatch_position == leaf.prefix.count && depth + leaf.prefix.count == key.len) {
			return InsertToLeaf(node, row_id);
		}

		// replace leaf with Node4 and store both leaves in it
		auto old_node = node;
		auto &new_n4 = Node4::New(*this, node);
		new_n4.prefix.Initialize(*this, key, depth, mismatch_position);

		auto key_byte = old_node.GetPrefix(*this).Reduce(*this, mismatch_position);
		Node4::InsertChild(*this, node, key_byte, old_node);

		Node leaf_node;
		Leaf::New(*this, leaf_node, key, depth + mismatch_position + 1, row_id);
		Node4::InsertChild(*this, node, key[depth + mismatch_position], leaf_node);

		return true;
	}

	// handle prefix of inner node
	auto &old_node_prefix = node.GetPrefix(*this);
	if (old_node_prefix.count) {

		auto mismatch_position = old_node_prefix.KeyMismatchPosition(*this, key, depth);
		if (mismatch_position != old_node_prefix.count) {

			// prefix differs, create new node
			auto old_node = node;
			auto &new_n4 = Node4::New(*this, node);
			new_n4.prefix.Initialize(*this, key, depth, mismatch_position);

			auto key_byte = old_node_prefix.Reduce(*this, mismatch_position);
			Node4::InsertChild(*this, node, key_byte, old_node);

			Node leaf_node;
			Leaf::New(*this, leaf_node, key, depth + mismatch_position + 1, row_id);
			Node4::InsertChild(*this, node, key[depth + mismatch_position], leaf_node);

			return true;
		}
		depth += node.GetPrefix(*this).count;
	}

	// recurse
	D_ASSERT(depth < key.len);
	auto child = node.GetChild(*this, key[depth]);
	if (child) {
		bool success = Insert(*child, key, depth + 1, row_id);
		node.ReplaceChild(*this, key[depth], *child);
		return success;
	}

	// insert at position
	Node leaf_node;
	Leaf::New(*this, leaf_node, key, depth + 1, row_id);
	Node::InsertChild(*this, node, key[depth], leaf_node);
	return true;
}

//===--------------------------------------------------------------------===//
// Delete
//===--------------------------------------------------------------------===//

void ART::Delete(IndexLock &state, DataChunk &input, Vector &row_ids) {

	DataChunk expression;
	expression.Initialize(Allocator::DefaultAllocator(), logical_types);

	// first resolve the expressions
	ExecuteExpressions(input, expression);

	// then generate the keys for the given input
	ArenaAllocator arena_allocator(BufferAllocator::Get(db));
	vector<ARTKey> keys(expression.size());
	GenerateKeys(arena_allocator, expression, keys);

	// now erase the elements from the database
	row_ids.Flatten(input.size());
	auto row_identifiers = FlatVector::GetData<row_t>(row_ids);

	for (idx_t i = 0; i < input.size(); i++) {
		if (keys[i].Empty()) {
			continue;
		}
		Erase(*tree, keys[i], 0, row_identifiers[i]);
#ifdef DEBUG
		auto node = Lookup(*tree, keys[i], 0);
		if (node.IsSet()) {
			auto &leaf = Leaf::Get(*this, node);
			for (idx_t k = 0; k < leaf.count; k++) {
				D_ASSERT(leaf.GetRowId(*this, k) != row_identifiers[i]);
			}
		}
#endif
	}
}

void ART::Erase(Node &node, const ARTKey &key, idx_t depth, const row_t &row_id) {

	if (!node.IsSet()) {
		return;
	}

	// delete a row ID from a leaf
	if (node.DecodeARTNodeType() == NType::LEAF) {
		auto &leaf = Leaf::Get(*this, node);
		leaf.Remove(*this, row_id);

		if (leaf.count == 0) {
			Node::Free(*this, node);
			node.Reset();
		}
		return;
	}

	// handle prefix
	auto &node_prefix = node.GetPrefix(*this);
	if (node_prefix.count) {
		if (node_prefix.KeyMismatchPosition(*this, key, depth) != node_prefix.count) {
			return;
		}
		depth += node_prefix.count;
	}

	auto child = node.GetChild(*this, key[depth]);
	if (child) {
		D_ASSERT(child->IsSet());

		if (child->DecodeARTNodeType() == NType::LEAF) {
			// leaf found, remove entry
			auto &leaf = Leaf::Get(*this, *child);
			leaf.Remove(*this, row_id);

			if (leaf.count == 0) {
				// leaf is empty, delete leaf, decrement node counter and maybe shrink node
				Node::DeleteChild(*this, node, key[depth]);
			}
			return;
		}

		// recurse
		Erase(*child, key, depth + 1, row_id);
		node.ReplaceChild(*this, key[depth], *child);
	}
}

//===--------------------------------------------------------------------===//
// Point Query (Equal)
//===--------------------------------------------------------------------===//

static ARTKey CreateKey(ArenaAllocator &allocator, PhysicalType type, Value &value) {
	D_ASSERT(type == value.type().InternalType());
	switch (type) {
	case PhysicalType::BOOL:
		return ARTKey::CreateARTKey<bool>(allocator, value.type(), value);
	case PhysicalType::INT8:
		return ARTKey::CreateARTKey<int8_t>(allocator, value.type(), value);
	case PhysicalType::INT16:
		return ARTKey::CreateARTKey<int16_t>(allocator, value.type(), value);
	case PhysicalType::INT32:
		return ARTKey::CreateARTKey<int32_t>(allocator, value.type(), value);
	case PhysicalType::INT64:
		return ARTKey::CreateARTKey<int64_t>(allocator, value.type(), value);
	case PhysicalType::UINT8:
		return ARTKey::CreateARTKey<uint8_t>(allocator, value.type(), value);
	case PhysicalType::UINT16:
		return ARTKey::CreateARTKey<uint16_t>(allocator, value.type(), value);
	case PhysicalType::UINT32:
		return ARTKey::CreateARTKey<uint32_t>(allocator, value.type(), value);
	case PhysicalType::UINT64:
		return ARTKey::CreateARTKey<uint64_t>(allocator, value.type(), value);
	case PhysicalType::INT128:
		return ARTKey::CreateARTKey<hugeint_t>(allocator, value.type(), value);
	case PhysicalType::FLOAT:
		return ARTKey::CreateARTKey<float>(allocator, value.type(), value);
	case PhysicalType::DOUBLE:
		return ARTKey::CreateARTKey<double>(allocator, value.type(), value);
	case PhysicalType::VARCHAR:
		return ARTKey::CreateARTKey<string_t>(allocator, value.type(), value);
	default:
		throw InternalException("Invalid type for the ART key");
	}
}

bool ART::SearchEqual(ARTKey &key, idx_t max_count, vector<row_t> &result_ids) {

	auto leaf_node = Lookup(*tree, key, 0);
	if (!leaf_node.IsSet()) {
		return true;
	}

	auto &leaf = Leaf::Get(*this, leaf_node);
	if (leaf.count > max_count) {
		return false;
	}
	for (idx_t i = 0; i < leaf.count; i++) {
		row_t row_id = leaf.GetRowId(*this, i);
		result_ids.push_back(row_id);
	}
	return true;
}

void ART::SearchEqualJoinNoFetch(ARTKey &key, idx_t &result_size) {

	// we need to look for a leaf
	auto leaf_node = Lookup(*tree, key, 0);
	if (!leaf_node.IsSet()) {
		result_size = 0;
		return;
	}

	auto &leaf = Leaf::Get(*this, leaf_node);
	result_size = leaf.count;
}

//===--------------------------------------------------------------------===//
// Lookup
//===--------------------------------------------------------------------===//

Node ART::Lookup(Node node, const ARTKey &key, idx_t depth) {

	while (node.IsSet()) {
		if (node.DecodeARTNodeType() == NType::LEAF) {
			auto &leaf = Leaf::Get(*this, node);

			// check if leaf contains key
			for (idx_t i = 0; i < leaf.prefix.count; i++) {
				if (leaf.prefix.GetByte(*this, i) != key[i + depth]) {
					return Node();
				}
			}
			return node;
		}
		auto &node_prefix = node.GetPrefix(*this);
		if (node_prefix.count) {
			for (idx_t pos = 0; pos < node_prefix.count; pos++) {
				if (key[depth + pos] != node_prefix.GetByte(*this, pos)) {
					// prefix mismatch, subtree of node does not contain key
					return Node();
				}
			}
			depth += node_prefix.count;
		}

		// prefix matches key, but no child at byte, does not contain key
		auto child = node.GetChild(*this, key[depth]);
		if (!child) {
			return Node();
		}

		// recurse into child
		node = *child;
		D_ASSERT(node.IsSet());
		depth++;
	}

	return Node();
}

//===--------------------------------------------------------------------===//
// Greater Than
// Returns: True (If found leaf >= key)
//          False (Otherwise)
//===--------------------------------------------------------------------===//

bool ART::SearchGreater(ARTIndexScanState *state, ARTKey &key, bool inclusive, idx_t max_count,
                        vector<row_t> &result_ids) {

	Iterator *it = &state->iterator;

	// greater than scan: first set the iterator to the node at which we will start our scan by finding the lowest node
	// that satisfies our requirement
	if (!it->art) {
		it->art = this;
		if (!it->LowerBound(*tree, key, inclusive)) {
			return true;
		}
	}

	// after that we continue the scan; we don't need to check the bounds as any value following this value is
	// automatically bigger and hence satisfies our predicate
	ARTKey empty_key = ARTKey();
	return it->Scan(empty_key, max_count, result_ids, false);
}

//===--------------------------------------------------------------------===//
// Less Than
//===--------------------------------------------------------------------===//

bool ART::SearchLess(ARTIndexScanState *state, ARTKey &upper_bound, bool inclusive, idx_t max_count,
                     vector<row_t> &result_ids) {

	if (!tree->IsSet()) {
		return true;
	}

	Iterator *it = &state->iterator;

	if (!it->art) {
		it->art = this;
		// first find the minimum value in the ART: we start scanning from this value
		it->FindMinimum(*tree);
		// early out min value higher than upper bound query
		if (it->cur_key > upper_bound) {
			return true;
		}
	}

	// now continue the scan until we reach the upper bound
	return it->Scan(upper_bound, max_count, result_ids, inclusive);
}

//===--------------------------------------------------------------------===//
// Closed Range Query
//===--------------------------------------------------------------------===//

bool ART::SearchCloseRange(ARTIndexScanState *state, ARTKey &lower_bound, ARTKey &upper_bound, bool left_inclusive,
                           bool right_inclusive, idx_t max_count, vector<row_t> &result_ids) {

	Iterator *it = &state->iterator;

	// first find the first node that satisfies the left predicate
	if (!it->art) {
		it->art = this;
		if (!it->LowerBound(*tree, lower_bound, left_inclusive)) {
			return true;
		}
	}

	// now continue the scan until we reach the upper bound
	return it->Scan(upper_bound, max_count, result_ids, right_inclusive);
}

bool ART::Scan(const Transaction &transaction, const DataTable &table, IndexScanState &table_state,
               const idx_t max_count, vector<row_t> &result_ids) {

	auto state = (ARTIndexScanState *)&table_state;
	vector<row_t> row_ids;
	bool success;

	// FIXME: the key directly owning the data for a single key might be more efficient
	D_ASSERT(state->values[0].type().InternalType() == types[0]);
	ArenaAllocator arena_allocator(Allocator::Get(db));
	auto key = CreateKey(arena_allocator, types[0], state->values[0]);

	if (state->values[1].IsNull()) {

		// single predicate
		lock_guard<mutex> l(lock);
		switch (state->expressions[0]) {
		case ExpressionType::COMPARE_EQUAL:
			success = SearchEqual(key, max_count, row_ids);
			break;
		case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
			success = SearchGreater(state, key, true, max_count, row_ids);
			break;
		case ExpressionType::COMPARE_GREATERTHAN:
			success = SearchGreater(state, key, false, max_count, row_ids);
			break;
		case ExpressionType::COMPARE_LESSTHANOREQUALTO:
			success = SearchLess(state, key, true, max_count, row_ids);
			break;
		case ExpressionType::COMPARE_LESSTHAN:
			success = SearchLess(state, key, false, max_count, row_ids);
			break;
		default:
			throw InternalException("Operation not implemented");
		}

	} else {

		// two predicates
		lock_guard<mutex> l(lock);

		D_ASSERT(state->values[1].type().InternalType() == types[0]);
		auto upper_bound = CreateKey(arena_allocator, types[0], state->values[1]);

		bool left_inclusive = state->expressions[0] == ExpressionType ::COMPARE_GREATERTHANOREQUALTO;
		bool right_inclusive = state->expressions[1] == ExpressionType ::COMPARE_LESSTHANOREQUALTO;
		success = SearchCloseRange(state, key, upper_bound, left_inclusive, right_inclusive, max_count, row_ids);
	}

	if (!success) {
		return false;
	}
	if (row_ids.empty()) {
		return true;
	}

	// sort the row ids
	sort(row_ids.begin(), row_ids.end());
	// duplicate eliminate the row ids and append them to the row ids of the state
	result_ids.reserve(row_ids.size());

	result_ids.push_back(row_ids[0]);
	for (idx_t i = 1; i < row_ids.size(); i++) {
		if (row_ids[i] != row_ids[i - 1]) {
			result_ids.push_back(row_ids[i]);
		}
	}
	return true;
}

//===--------------------------------------------------------------------===//
// More Verification / Constraint Checking
//===--------------------------------------------------------------------===//

string ART::GenerateErrorKeyName(DataChunk &input, idx_t row) {

	// FIXME: why exactly can we not pass the expression_chunk as an argument to this
	// FIXME: function instead of re-executing?
	// re-executing the expressions is not very fast, but we're going to throw, so we don't care
	DataChunk expression_chunk;
	expression_chunk.Initialize(Allocator::DefaultAllocator(), logical_types);
	ExecuteExpressions(input, expression_chunk);

	string key_name;
	for (idx_t k = 0; k < expression_chunk.ColumnCount(); k++) {
		if (k > 0) {
			key_name += ", ";
		}
		key_name += unbound_expressions[k]->GetName() + ": " + expression_chunk.data[k].GetValue(row).ToString();
	}
	return key_name;
}

string ART::GenerateConstraintErrorMessage(VerifyExistenceType verify_type, const string &key_name) {
	switch (verify_type) {
	case VerifyExistenceType::APPEND: {
		// APPEND to PK/UNIQUE table, but node/key already exists in PK/UNIQUE table
		string type = IsPrimary() ? "primary key" : "unique";
		return StringUtil::Format(
		    "Duplicate key \"%s\" violates %s constraint. "
		    "If this is an unexpected constraint violation please double "
		    "check with the known index limitations section in our documentation (docs - sql - indexes).",
		    key_name, type);
	}
	case VerifyExistenceType::APPEND_FK: {
		// APPEND_FK to FK table, node/key does not exist in PK/UNIQUE table
		return StringUtil::Format(
		    "Violates foreign key constraint because key \"%s\" does not exist in the referenced table", key_name);
	}
	case VerifyExistenceType::DELETE_FK: {
		// DELETE_FK that still exists in a FK table, i.e., not a valid delete
		return StringUtil::Format("Violates foreign key constraint because key \"%s\" is still referenced by a foreign "
		                          "key in a different table",
		                          key_name);
	}
	default:
		throw NotImplementedException("Type not implemented for VerifyExistenceType");
	}
}

void ART::CheckConstraintsForChunk(DataChunk &input, ConflictManager &conflict_manager) {

	// don't alter the index during constraint checking
	lock_guard<mutex> l(lock);

	// first resolve the expressions for the index
	DataChunk expression_chunk;
	expression_chunk.Initialize(Allocator::DefaultAllocator(), logical_types);
	ExecuteExpressions(input, expression_chunk);

	// generate the keys for the given input
	ArenaAllocator arena_allocator(BufferAllocator::Get(db));
	vector<ARTKey> keys(expression_chunk.size());
	GenerateKeys(arena_allocator, expression_chunk, keys);

	idx_t found_conflict = DConstants::INVALID_INDEX;
	for (idx_t i = 0; found_conflict == DConstants::INVALID_INDEX && i < input.size(); i++) {

		if (keys[i].Empty()) {
			if (conflict_manager.AddNull(i)) {
				found_conflict = i;
			}
			continue;
		}

		auto leaf_node = Lookup(*tree, keys[i], 0);
		if (!leaf_node.IsSet()) {
			if (conflict_manager.AddMiss(i)) {
				found_conflict = i;
			}
			continue;
		}

		// When we find a node, we need to update the 'matches' and 'row_ids'
		// NOTE: Leafs can have more than one row_id, but for UNIQUE/PRIMARY KEY they will only have one
		Leaf &leaf = Leaf::Get(*this, leaf_node);
		D_ASSERT(leaf.count == 1);
		auto row_id = leaf.GetRowId(*this, 0);
		if (conflict_manager.AddHit(i, row_id)) {
			found_conflict = i;
		}
	}

	conflict_manager.FinishLookup();

	if (found_conflict == DConstants::INVALID_INDEX) {
		return;
	}

	auto key_name = GenerateErrorKeyName(input, found_conflict);
	auto exception_msg = GenerateConstraintErrorMessage(conflict_manager.LookupType(), key_name);
	throw ConstraintException(exception_msg);
}

//===--------------------------------------------------------------------===//
// Serialization
//===--------------------------------------------------------------------===//

BlockPointer ART::Serialize(MetaBlockWriter &writer) {

	lock_guard<mutex> l(lock);
	if (tree->IsSet()) {
		serialized_data_pointer = tree->Serialize(*this, writer);
	} else {
		serialized_data_pointer = {(block_id_t)DConstants::INVALID_INDEX, (uint32_t)DConstants::INVALID_INDEX};
	}

	return serialized_data_pointer;
}

//===--------------------------------------------------------------------===//
// Vacuum
//===--------------------------------------------------------------------===//

void ART::InitializeVacuum(ARTFlags &flags) {

	flags.vacuum_flags.reserve(allocators.size());
	for (auto &allocator : allocators) {
		flags.vacuum_flags.push_back(allocator->InitializeVacuum());
	}
}

void ART::FinalizeVacuum(const ARTFlags &flags) {

	for (idx_t i = 0; i < allocators.size(); i++) {
		if (flags.vacuum_flags[i]) {
			allocators[i]->FinalizeVacuum();
		}
	}
}

void ART::Vacuum(IndexLock &state) {

	if (!tree->IsSet()) {
		for (auto &allocator : allocators) {
			allocator->Reset();
		}
		return;
	}

	// holds true, if an allocator needs a vacuum, and false otherwise
	ARTFlags flags;
	InitializeVacuum(flags);

	// skip vacuum if no allocators require it
	auto perform_vacuum = false;
	for (const auto &vacuum_flag : flags.vacuum_flags) {
		if (vacuum_flag) {
			perform_vacuum = true;
			break;
		}
	}
	if (!perform_vacuum) {
		return;
	}

	// traverse the allocated memory of the tree to perform a vacuum
	Node::Vacuum(*this, *tree, flags);

	// finalize the vacuum operation
	FinalizeVacuum(flags);
}

//===--------------------------------------------------------------------===//
// Merging
//===--------------------------------------------------------------------===//

void ART::InitializeMerge(ARTFlags &flags) {

	flags.merge_buffer_counts.reserve(allocators.size());
	for (auto &allocator : allocators) {
		flags.merge_buffer_counts.emplace_back(allocator->buffers.size());
	}
}

bool ART::MergeIndexes(IndexLock &state, Index &other_index) {

	auto &other_art = other_index.Cast<ART>();
	if (!other_art.tree->IsSet()) {
		return true;
	}

	if (tree->IsSet()) {
		//  fully deserialize other_index, and traverse it to increment its buffer IDs
		ARTFlags flags;
		InitializeMerge(flags);
		other_art.tree->InitializeMerge(other_art, flags);
	}

	// merge the node storage
	for (idx_t i = 0; i < allocators.size(); i++) {
		allocators[i]->Merge(*other_art.allocators[i]);
	}

	// merge the ARTs
	if (!tree->Merge(*this, *other_art.tree)) {
		return false;
	}
	return true;
}

//===--------------------------------------------------------------------===//
// Utility
//===--------------------------------------------------------------------===//

string ART::ToString() {
	if (tree->IsSet()) {
		return tree->ToString(*this);
	}
	return "[empty]";
}

} // namespace duckdb


namespace duckdb {

ARTKey::ARTKey() : len(0) {
}

ARTKey::ARTKey(const data_ptr_t &data, const uint32_t &len) : len(len), data(data) {
}

ARTKey::ARTKey(ArenaAllocator &allocator, const uint32_t &len) : len(len) {
	data = allocator.Allocate(len);
}

template <>
ARTKey ARTKey::CreateARTKey(ArenaAllocator &allocator, const LogicalType &type, string_t value) {
	uint32_t len = value.GetSize() + 1;
	auto data = allocator.Allocate(len);
	memcpy(data, value.GetData(), len - 1);

	// FIXME: rethink this
	if (type == LogicalType::BLOB || type == LogicalType::VARCHAR) {
		// indexes cannot contain BLOBs (or BLOBs cast to VARCHARs) that contain null-terminated bytes
		for (uint32_t i = 0; i < len - 1; i++) {
			if (data[i] == '\0') {
				throw NotImplementedException("Indexes cannot contain BLOBs that contain null-terminated bytes.");
			}
		}
	}

	data[len - 1] = '\0';
	return ARTKey(data, len);
}

template <>
ARTKey ARTKey::CreateARTKey(ArenaAllocator &allocator, const LogicalType &type, const char *value) {
	return ARTKey::CreateARTKey(allocator, type, string_t(value, strlen(value)));
}

template <>
void ARTKey::CreateARTKey(ArenaAllocator &allocator, const LogicalType &type, ARTKey &key, string_t value) {
	key.len = value.GetSize() + 1;
	key.data = allocator.Allocate(key.len);
	memcpy(key.data, value.GetData(), key.len - 1);

	// FIXME: rethink this
	if (type == LogicalType::BLOB || type == LogicalType::VARCHAR) {
		// indexes cannot contain BLOBs (or BLOBs cast to VARCHARs) that contain null-terminated bytes
		for (uint32_t i = 0; i < key.len - 1; i++) {
			if (key.data[i] == '\0') {
				throw NotImplementedException("Indexes cannot contain BLOBs that contain null-terminated bytes.");
			}
		}
	}

	key.data[key.len - 1] = '\0';
}

template <>
void ARTKey::CreateARTKey(ArenaAllocator &allocator, const LogicalType &type, ARTKey &key, const char *value) {
	ARTKey::CreateARTKey(allocator, type, key, string_t(value, strlen(value)));
}

bool ARTKey::operator>(const ARTKey &k) const {
	for (uint32_t i = 0; i < MinValue<uint32_t>(len, k.len); i++) {
		if (data[i] > k.data[i]) {
			return true;
		} else if (data[i] < k.data[i]) {
			return false;
		}
	}
	return len > k.len;
}

bool ARTKey::operator<(const ARTKey &k) const {
	for (uint32_t i = 0; i < MinValue<uint32_t>(len, k.len); i++) {
		if (data[i] < k.data[i]) {
			return true;
		} else if (data[i] > k.data[i]) {
			return false;
		}
	}
	return len < k.len;
}

bool ARTKey::operator>=(const ARTKey &k) const {
	for (uint32_t i = 0; i < MinValue<uint32_t>(len, k.len); i++) {
		if (data[i] > k.data[i]) {
			return true;
		} else if (data[i] < k.data[i]) {
			return false;
		}
	}
	return len >= k.len;
}

bool ARTKey::operator==(const ARTKey &k) const {
	if (len != k.len) {
		return false;
	}
	for (uint32_t i = 0; i < len; i++) {
		if (data[i] != k.data[i]) {
			return false;
		}
	}
	return true;
}

void ARTKey::ConcatenateARTKey(ArenaAllocator &allocator, ARTKey &other_key) {

	auto compound_data = allocator.Allocate(len + other_key.len);
	memcpy(compound_data, data, len);
	memcpy(compound_data + len, other_key.data, other_key.len);
	len += other_key.len;
	data = compound_data;
}
} // namespace duckdb






namespace duckdb {

constexpr idx_t FixedSizeAllocator::BASE[];
constexpr uint8_t FixedSizeAllocator::SHIFT[];

FixedSizeAllocator::FixedSizeAllocator(const idx_t allocation_size, Allocator &allocator)
    : allocation_size(allocation_size), total_allocations(0), allocator(allocator) {

	// calculate how many allocations fit into one buffer

	idx_t bits_per_value = sizeof(validity_t) * 8;
	idx_t curr_alloc_size = 0;

	bitmask_count = 0;
	allocations_per_buffer = 0;

	while (curr_alloc_size < BUFFER_ALLOC_SIZE) {
		if (!bitmask_count || (bitmask_count * bits_per_value) % allocations_per_buffer == 0) {
			bitmask_count++;
			curr_alloc_size += sizeof(validity_t);
		}

		auto remaining_alloc_size = BUFFER_ALLOC_SIZE - curr_alloc_size;
		auto remaining_allocations = MinValue(remaining_alloc_size / allocation_size, bits_per_value);

		if (remaining_allocations == 0) {
			break;
		}

		allocations_per_buffer += remaining_allocations;
		curr_alloc_size += remaining_allocations * allocation_size;
	}

	allocation_offset = bitmask_count * sizeof(validity_t);
}

FixedSizeAllocator::~FixedSizeAllocator() {
	for (auto &buffer : buffers) {
		allocator.FreeData(buffer.ptr, BUFFER_ALLOC_SIZE);
	}
}

SwizzleablePointer FixedSizeAllocator::New() {

	// no more free pointers
	if (buffers_with_free_space.empty()) {

		// add a new buffer
		idx_t buffer_id = buffers.size();
		D_ASSERT(buffer_id <= (uint32_t)DConstants::INVALID_INDEX);
		auto buffer = allocator.AllocateData(BUFFER_ALLOC_SIZE);
		buffers.emplace_back(buffer, 0);
		buffers_with_free_space.insert(buffer_id);

		// set the bitmask
		ValidityMask mask((validity_t *)buffer);
		mask.SetAllValid(allocations_per_buffer);
	}

	// return a pointer
	D_ASSERT(!buffers_with_free_space.empty());
	auto buffer_id = (uint32_t)*buffers_with_free_space.begin();

	auto bitmask_ptr = (validity_t *)buffers[buffer_id].ptr;
	ValidityMask mask(bitmask_ptr);
	auto offset = GetOffset(mask, buffers[buffer_id].allocation_count);

	buffers[buffer_id].allocation_count++;
	total_allocations++;
	if (buffers[buffer_id].allocation_count == allocations_per_buffer) {
		buffers_with_free_space.erase(buffer_id);
	}

	return SwizzleablePointer(offset, buffer_id);
}

void FixedSizeAllocator::Free(const SwizzleablePointer ptr) {

	auto bitmask_ptr = (validity_t *)buffers[ptr.buffer_id].ptr;
	ValidityMask mask(bitmask_ptr);
	D_ASSERT(!mask.RowIsValid(ptr.offset));
	mask.SetValid(ptr.offset);
	buffers_with_free_space.insert(ptr.buffer_id);

	D_ASSERT(total_allocations > 0);
	D_ASSERT(buffers[ptr.buffer_id].allocation_count > 0);
	buffers[ptr.buffer_id].allocation_count--;
	total_allocations--;
}

void FixedSizeAllocator::Reset() {

	for (auto &buffer : buffers) {
		allocator.FreeData(buffer.ptr, BUFFER_ALLOC_SIZE);
	}
	buffers.clear();
	buffers_with_free_space.clear();
	total_allocations = 0;
}

void FixedSizeAllocator::Merge(FixedSizeAllocator &other) {

	D_ASSERT(allocation_size == other.allocation_size);

	// remember the buffer count and merge the buffers
	idx_t buffer_count = buffers.size();
	for (auto &buffer : other.buffers) {
		buffers.push_back(buffer);
	}
	other.buffers.clear();

	// merge the vectors containing all buffers with free space
	for (auto &buffer_id : other.buffers_with_free_space) {
		buffers_with_free_space.insert(buffer_id + buffer_count);
	}
	other.buffers_with_free_space.clear();

	// add the total allocations
	total_allocations += other.total_allocations;
}

bool FixedSizeAllocator::InitializeVacuum() {

	auto total_available_allocations = allocations_per_buffer * buffers.size();
	auto total_free_positions = total_available_allocations - total_allocations;

	// vacuum_count buffers can be freed
	auto vacuum_count = total_free_positions / allocations_per_buffer / 2;

	// calculate the vacuum threshold adaptively
	idx_t memory_usage = GetMemoryUsage();
	idx_t excess_memory_usage = vacuum_count * BUFFER_ALLOC_SIZE;
	auto excess_percentage = (double)excess_memory_usage / (double)memory_usage;
	auto threshold = (double)VACUUM_THRESHOLD / 100.0;
	if (excess_percentage < threshold) {
		return false;
	}

	min_vacuum_buffer_id = buffers.size() - vacuum_count;

	// remove all invalid buffers from the available buffer list to ensure that we do not reuse them
	auto it = buffers_with_free_space.begin();
	while (it != buffers_with_free_space.end()) {
		if (*it >= min_vacuum_buffer_id) {
			it = buffers_with_free_space.erase(it);
		} else {
			it++;
		}
	}

	return true;
}

void FixedSizeAllocator::FinalizeVacuum() {

	// free all (now unused) buffers
	while (min_vacuum_buffer_id < buffers.size()) {
		allocator.FreeData(buffers.back().ptr, BUFFER_ALLOC_SIZE);
		buffers.pop_back();
	}
}

SwizzleablePointer FixedSizeAllocator::VacuumPointer(const SwizzleablePointer ptr) {

	// we do not need to adjust the bitmask of the old buffer, because we will free the entire
	// buffer after the vacuum operation

	auto new_ptr = New();
	memcpy(Get(new_ptr), Get(ptr), allocation_size);
	return new_ptr;
}

uint32_t FixedSizeAllocator::GetOffset(ValidityMask &mask, const idx_t allocation_count) {

	auto data = mask.GetData();

	// fills up a buffer sequentially before searching for free bits
	if (mask.RowIsValid(allocation_count)) {
		mask.SetInvalid(allocation_count);
		return allocation_count;
	}

	// get an entry with free bits
	for (idx_t entry_idx = 0; entry_idx < bitmask_count; entry_idx++) {
		if (data[entry_idx] != 0) {

			// find the position of the free bit
			auto entry = data[entry_idx];
			idx_t first_valid_bit = 0;

			// this loop finds the position of the rightmost set bit in entry and stores it
			// in first_valid_bit
			for (idx_t i = 0; i < 6; i++) {
				// set the left half of the bits of this level to zero and test if the entry is still not zero
				if (entry & BASE[i]) {
					// first valid bit is in the rightmost s[i] bits
					// permanently set the left half of the bits to zero
					entry &= BASE[i];
				} else {
					// first valid bit is in the leftmost s[i] bits
					// shift by s[i] for the next iteration and add s[i] to the position of the rightmost set bit
					entry >>= SHIFT[i];
					first_valid_bit += SHIFT[i];
				}
			}
			D_ASSERT(entry);

			auto prev_bits = entry_idx * sizeof(validity_t) * 8;
			D_ASSERT(mask.RowIsValid(prev_bits + first_valid_bit));
			mask.SetInvalid(prev_bits + first_valid_bit);
			return (prev_bits + first_valid_bit);
		}
	}

	throw InternalException("Invalid bitmask of FixedSizeAllocator");
}

} // namespace duckdb







namespace duckdb {

void IteratorCurrentKey::Push(const uint8_t byte) {
	if (cur_key_pos == key.size()) {
		key.push_back(byte);
	}
	D_ASSERT(cur_key_pos < key.size());
	key[cur_key_pos++] = byte;
}

void IteratorCurrentKey::Pop(const idx_t n) {
	cur_key_pos -= n;
	D_ASSERT(cur_key_pos <= key.size());
}

uint8_t &IteratorCurrentKey::operator[](idx_t idx) {
	if (idx >= key.size()) {
		key.push_back(0);
	}
	D_ASSERT(idx < key.size());
	return key[idx];
}

bool IteratorCurrentKey::operator>(const ARTKey &k) const {
	for (idx_t i = 0; i < MinValue<idx_t>(cur_key_pos, k.len); i++) {
		if (key[i] > k.data[i]) {
			return true;
		} else if (key[i] < k.data[i]) {
			return false;
		}
	}
	return cur_key_pos > k.len;
}

bool IteratorCurrentKey::operator>=(const ARTKey &k) const {
	for (idx_t i = 0; i < MinValue<idx_t>(cur_key_pos, k.len); i++) {
		if (key[i] > k.data[i]) {
			return true;
		} else if (key[i] < k.data[i]) {
			return false;
		}
	}
	return cur_key_pos >= k.len;
}

bool IteratorCurrentKey::operator==(const ARTKey &k) const {
	if (cur_key_pos != k.len) {
		return false;
	}
	for (idx_t i = 0; i < cur_key_pos; i++) {
		if (key[i] != k.data[i]) {
			return false;
		}
	}
	return true;
}

void Iterator::FindMinimum(Node &node) {

	// reconstruct the prefix
	// FIXME: get all bytes at once to increase performance
	auto &node_prefix = node.GetPrefix(*art);
	for (idx_t i = 0; i < node_prefix.count; i++) {
		cur_key.Push(node_prefix.GetByte(*art, i));
	}

	// found the minimum
	if (node.DecodeARTNodeType() == NType::LEAF) {
		last_leaf = Node::GetAllocator(*art, NType::LEAF).Get<Leaf>(node);
		return;
	}

	// go to the leftmost entry in the current node
	uint8_t byte = 0;
	auto next = node.GetNextChild(*art, byte);
	D_ASSERT(next);
	cur_key.Push(byte);

	// recurse
	nodes.emplace(node, byte);
	FindMinimum(*next);
}

void Iterator::PushKey(const Node &node, const uint8_t byte) {
	if (node.DecodeARTNodeType() != NType::LEAF) {
		cur_key.Push(byte);
	}
}

bool Iterator::Scan(const ARTKey &key, const idx_t &max_count, vector<row_t> &result_ids, const bool &is_inclusive) {

	bool has_next;
	do {
		if (!key.Empty()) {
			// no more row IDs within the key bounds
			if (is_inclusive) {
				if (cur_key > key) {
					return true;
				}
			} else {
				if (cur_key >= key) {
					return true;
				}
			}
		}

		// adding more elements would exceed the max count
		if (result_ids.size() + last_leaf->count > max_count) {
			return false;
		}

		// FIXME: copy all at once to improve performance
		for (idx_t i = 0; i < last_leaf->count; i++) {
			row_t row_id = last_leaf->GetRowId(*art, i);
			result_ids.push_back(row_id);
		}

		// get the next leaf
		has_next = Next();

	} while (has_next);

	return true;
}

void Iterator::PopNode() {
	auto cur_node = nodes.top();
	idx_t elements_to_pop = cur_node.node.GetPrefix(*art).count + (nodes.size() != 1);
	cur_key.Pop(elements_to_pop);
	nodes.pop();
}

bool Iterator::Next() {
	if (!nodes.empty()) {
		auto cur_node = nodes.top().node;
		if (cur_node.DecodeARTNodeType() == NType::LEAF) {
			// pop leaf
			// we must pop the prefix size + the key to the node, unless we are popping the root
			PopNode();
		}
	}

	// look for the next leaf
	while (!nodes.empty()) {

		// cur_node
		auto &top = nodes.top();
		Node node = top.node;

		// found a leaf: move to next node
		if (node.DecodeARTNodeType() == NType::LEAF) {
			last_leaf = Node::GetAllocator(*art, NType::LEAF).Get<Leaf>(node);
			return true;
		}

		// find next node
		if (top.byte == NumericLimits<uint8_t>::Maximum()) {
			// no node found: move up the tree, pop prefix and key of current node
			PopNode();
			continue;
		}

		top.byte == 0 ? top.byte : top.byte++;
		auto next_node = node.GetNextChild(*art, top.byte);

		if (next_node) {
			// add the next node's key byte
			PushKey(node, top.byte);

			// add prefix of new node
			// FIXME: get all bytes at once to increase performance
			auto &next_node_prefix = next_node->GetPrefix(*art);
			for (idx_t i = 0; i < next_node_prefix.count; i++) {
				cur_key.Push(next_node_prefix.GetByte(*art, i));
			}

			// next node found: push it
			nodes.emplace(*next_node, 0);
		} else {

			// no node found: move up the tree, pop prefix and key of current node
			PopNode();
		}
	}
	return false;
}

bool Iterator::LowerBound(Node node, const ARTKey &key, const bool &is_inclusive) {

	if (!node.IsSet()) {
		return false;
	}

	idx_t depth = 0;
	bool equal = true;
	while (true) {

		nodes.emplace(node, 0);
		auto &top = nodes.top();

		// reconstruct the prefix
		// FIXME: get all bytes at once to increase performance
		reference<Prefix> node_prefix(top.node.GetPrefix(*art));
		for (idx_t i = 0; i < node_prefix.get().count; i++) {
			cur_key.Push(node_prefix.get().GetByte(*art, i));
		}

		// greater case: find leftmost leaf node directly
		if (!equal) {
			while (node.DecodeARTNodeType() != NType::LEAF) {

				uint8_t byte = 0;
				auto next_node = *node.GetNextChild(*art, byte);
				D_ASSERT(next_node.IsSet());

				PushKey(node, byte);
				nodes.emplace(node, byte);
				node = next_node;

				// reconstruct the prefix
				node_prefix = node.GetPrefix(*art);
				for (idx_t i = 0; i < node_prefix.get().count; i++) {
					cur_key.Push(node_prefix.get().GetByte(*art, i));
				}

				auto &c_top = nodes.top();
				c_top.node = node;
			}
		}

		if (node.DecodeARTNodeType() == NType::LEAF) {
			// found a leaf node: check if it is bigger or equal than the current key
			last_leaf = Node::GetAllocator(*art, NType::LEAF).Get<Leaf>(node);

			// if the search is not inclusive the leaf node could still be equal to the current value
			// check if leaf is equal to the current key
			if (cur_key == key) {
				// if it's not inclusive check if there is a next leaf
				if (!is_inclusive && !Next()) {
					return false;
				} else {
					return true;
				}
			}

			if (cur_key > key) {
				return true;
			}
			// Case1: When the ART has only one leaf node, the Next() will return false
			// Case2: This means the previous node prefix(if any) + a_key(one element of of key array of previous node)
			// == key[q..=w].
			// But key[w+1..=z] maybe greater than leaf node prefix.
			// One fact is key[w] is alawys equal to a_key and the next element
			// of key array of previous node is always > a_key So we just call Next() once.

			return Next();
		}

		// equal case:
		node_prefix = node.GetPrefix(*art);
		auto mismatch_pos = node_prefix.get().KeyMismatchPosition(*art, key, depth);
		if (mismatch_pos != node_prefix.get().count) {
			if (node_prefix.get().GetByte(*art, mismatch_pos) < key[depth + mismatch_pos]) {
				// less
				PopNode();
				return Next();
			}
			// greater
			top.byte = 0;
			return Next();
		}

		// prefix matches, search inside the child for the key
		depth += node_prefix.get().count;
		top.byte = key[depth];
		auto child = node.GetNextChild(*art, top.byte);
		equal = key[depth] == top.byte;

		// the maximum key byte of the current node is less than the key
		// fall back to the previous node
		if (!child) {
			PopNode();
			return Next();
		}

		PushKey(node, top.byte);
		node = *child;

		// all children of this node qualify as greater or equal
		depth++;
	}
}

} // namespace duckdb









namespace duckdb {

Leaf &Leaf::New(ART &art, Node &node, const ARTKey &key, const uint32_t depth, const row_t row_id) {

	node.SetPtr(Node::GetAllocator(art, NType::LEAF).New());
	node.type = (uint8_t)NType::LEAF;
	auto &leaf = Leaf::Get(art, node);

	// set the fields of the leaf
	leaf.count = 1;
	leaf.row_ids.inlined = row_id;

	// initialize the prefix
	D_ASSERT(key.len >= depth);
	leaf.prefix.Initialize(art, key, depth, key.len - depth);

	return leaf;
}

Leaf &Leaf::New(ART &art, Node &node, const ARTKey &key, const uint32_t depth, const row_t *row_ids,
                const idx_t count) {

	// inlined leaf
	D_ASSERT(count >= 1);
	if (count == 1) {
		return Leaf::New(art, node, key, depth, row_ids[0]);
	}

	node.SetPtr(Node::GetAllocator(art, NType::LEAF).New());
	node.type = (uint8_t)NType::LEAF;
	auto &leaf = Leaf::Get(art, node);

	// set the fields of the leaf
	leaf.count = 0;

	// copy the row IDs
	reference<LeafSegment> segment(LeafSegment::New(art, leaf.row_ids.ptr));
	for (idx_t i = 0; i < count; i++) {
		segment = segment.get().Append(art, leaf.count, row_ids[i]);
	}

	// set the prefix
	D_ASSERT(key.len >= depth);
	leaf.prefix.Initialize(art, key, depth, key.len - depth);

	return leaf;
}

void Leaf::Free(ART &art, Node &node) {

	D_ASSERT(node.IsSet());
	D_ASSERT(!node.IsSwizzled());

	auto &leaf = Leaf::Get(art, node);

	// delete all leaf segments
	if (!leaf.IsInlined()) {
		auto ptr = leaf.row_ids.ptr;
		while (ptr.IsSet()) {
			auto next_ptr = LeafSegment::Get(art, ptr).next;
			Node::Free(art, ptr);
			ptr = next_ptr;
		}
	}
}

void Leaf::InitializeMerge(const ART &art, const idx_t buffer_count) {

	if (IsInlined()) {
		return;
	}

	reference<LeafSegment> segment(LeafSegment::Get(art, row_ids.ptr));
	row_ids.ptr.buffer_id += buffer_count;

	auto ptr = segment.get().next;
	while (ptr.IsSet()) {
		segment.get().next.buffer_id += buffer_count;
		segment = LeafSegment::Get(art, ptr);
		ptr = segment.get().next;
	}
}

void Leaf::Merge(ART &art, Node &other) {

	auto &other_leaf = Leaf::Get(art, other);

	// copy inlined row ID
	if (other_leaf.IsInlined()) {
		Insert(art, other_leaf.row_ids.inlined);
		Node::Free(art, other);
		return;
	}

	// row ID was inlined, move to a new segment
	if (IsInlined()) {
		auto row_id = row_ids.inlined;
		auto &segment = LeafSegment::New(art, row_ids.ptr);
		segment.row_ids[0] = row_id;
	}

	// get the first segment to copy to
	reference<LeafSegment> segment(LeafSegment::Get(art, row_ids.ptr).GetTail(art));

	// initialize loop variables
	auto other_ptr = other_leaf.row_ids.ptr;
	auto remaining = other_leaf.count;

	// copy row IDs
	while (other_ptr.IsSet()) {
		auto &other_segment = LeafSegment::Get(art, other_ptr);
		auto copy_count = MinValue(Node::LEAF_SEGMENT_SIZE, remaining);

		// copy the data
		for (idx_t i = 0; i < copy_count; i++) {
			segment = segment.get().Append(art, count, other_segment.row_ids[i]);
		}

		// adjust the loop variables
		other_ptr = other_segment.next;
		remaining -= copy_count;
	}
	D_ASSERT(remaining == 0);

	Node::Free(art, other);
}

void Leaf::Insert(ART &art, const row_t row_id) {

	if (count == 0) {
		row_ids.inlined = row_id;
		count++;
		return;
	}

	if (count == 1) {
		MoveInlinedToSegment(art);
	}

	// append to the tail
	auto &first_segment = LeafSegment::Get(art, row_ids.ptr);
	auto &tail = first_segment.GetTail(art);
	tail.Append(art, count, row_id);
}

void Leaf::Remove(ART &art, const row_t row_id) {

	if (count == 0) {
		return;
	}

	if (IsInlined()) {
		if (row_ids.inlined == row_id) {
			count--;
		}
		return;
	}

	// possibly inline the row ID
	if (count == 2) {
		auto &segment = LeafSegment::Get(art, row_ids.ptr);
		if (segment.row_ids[0] != row_id && segment.row_ids[1] != row_id) {
			return;
		}

		auto remaining_row_id = segment.row_ids[0] == row_id ? segment.row_ids[1] : segment.row_ids[0];
		Node::Free(art, row_ids.ptr);
		row_ids.inlined = remaining_row_id;
		count--;
		return;
	}

	// find the row ID, and the segment containing that row ID (stored in ptr)
	auto ptr = row_ids.ptr;
	auto copy_idx = FindRowId(art, ptr, row_id);
	if (copy_idx == (uint32_t)DConstants::INVALID_INDEX) {
		return;
	}
	copy_idx++;

	// iterate all remaining segments and move the row IDs one field to the left
	reference<LeafSegment> segment(LeafSegment::Get(art, ptr));
	reference<LeafSegment> prev_segment(LeafSegment::Get(art, ptr));
	while (copy_idx < count) {

		auto copy_start = copy_idx % Node::LEAF_SEGMENT_SIZE;
		D_ASSERT(copy_start != 0);
		auto copy_end = MinValue(copy_start + count - copy_idx, Node::LEAF_SEGMENT_SIZE);

		// copy row IDs
		for (idx_t i = copy_start; i < copy_end; i++) {
			segment.get().row_ids[i - 1] = segment.get().row_ids[i];
			copy_idx++;
		}

		// adjust loop variables
		if (segment.get().next.IsSet()) {
			prev_segment = segment;
			segment = LeafSegment::Get(art, segment.get().next);
			// this segment has at least one element, and we need to copy it into the previous segment
			prev_segment.get().row_ids[Node::LEAF_SEGMENT_SIZE - 1] = segment.get().row_ids[0];
			copy_idx++;
		}
	}

	// this evaluates to true, if we need to delete the last segment
	if (count % Node::LEAF_SEGMENT_SIZE == 1) {
		ptr = row_ids.ptr;
		while (ptr.IsSet()) {

			// get the segment succeeding the current segment
			auto &current_segment = LeafSegment::Get(art, ptr);
			D_ASSERT(current_segment.next.IsSet());
			auto &next_segment = LeafSegment::Get(art, current_segment.next);

			// next_segment is the tail of the segment list
			if (!next_segment.next.IsSet()) {
				Node::Free(art, current_segment.next);
			}

			// adjust loop variables
			ptr = current_segment.next;
		}
	}
	count--;
}

row_t Leaf::GetRowId(const ART &art, const idx_t position) const {

	D_ASSERT(position < count);
	if (IsInlined()) {
		return row_ids.inlined;
	}

	// get the correct segment
	reference<LeafSegment> segment(LeafSegment::Get(art, row_ids.ptr));
	for (idx_t i = 0; i < position / Node::LEAF_SEGMENT_SIZE; i++) {
		D_ASSERT(segment.get().next.IsSet());
		segment = LeafSegment::Get(art, segment.get().next);
	}

	return segment.get().row_ids[position % Node::LEAF_SEGMENT_SIZE];
}

uint32_t Leaf::FindRowId(const ART &art, Node &ptr, const row_t row_id) const {

	D_ASSERT(!IsInlined());

	auto remaining = count;
	while (ptr.IsSet()) {

		auto &segment = LeafSegment::Get(art, ptr);
		auto search_count = MinValue(Node::LEAF_SEGMENT_SIZE, remaining);

		// search in this segment
		for (idx_t i = 0; i < search_count; i++) {
			if (segment.row_ids[i] == row_id) {
				return count - remaining + i;
			}
		}

		// adjust loop variables
		remaining -= search_count;
		ptr = segment.next;
	}
	return (uint32_t)DConstants::INVALID_INDEX;
}

string Leaf::ToString(const ART &art) const {

	if (IsInlined()) {
		return "Leaf (" + to_string(count) + "): [" + to_string(row_ids.inlined) + "]";
	}

	auto ptr = row_ids.ptr;
	auto remaining = count;
	string str = "";
	uint32_t this_count = 0;
	while (ptr.IsSet()) {
		auto &segment = LeafSegment::Get(art, ptr);
		auto to_string_count = Node::LEAF_SEGMENT_SIZE < remaining ? Node::LEAF_SEGMENT_SIZE : remaining;

		for (idx_t i = 0; i < to_string_count; i++) {
			str += ", " + to_string(segment.row_ids[i]);
			this_count++;
		}
		remaining -= to_string_count;
		ptr = segment.next;
	}
	return "Leaf (" + to_string(this_count) + ", " + to_string(count) + "): [" + str + "] \n";
}

BlockPointer Leaf::Serialize(const ART &art, MetaBlockWriter &writer) const {

	// get pointer and write fields
	auto block_pointer = writer.GetBlockPointer();
	writer.Write(NType::LEAF);
	writer.Write<uint32_t>(count);
	prefix.Serialize(art, writer);

	if (IsInlined()) {
		writer.Write(row_ids.inlined);
		return block_pointer;
	}

	D_ASSERT(row_ids.ptr.IsSet());
	auto ptr = row_ids.ptr;
	auto remaining = count;

	// iterate all leaf segments and write their row IDs
	while (ptr.IsSet()) {
		auto &segment = LeafSegment::Get(art, ptr);
		auto write_count = MinValue(Node::LEAF_SEGMENT_SIZE, remaining);

		// write the row IDs
		for (idx_t i = 0; i < write_count; i++) {
			writer.Write(segment.row_ids[i]);
		}

		// adjust loop variables
		remaining -= write_count;
		ptr = segment.next;
	}
	D_ASSERT(remaining == 0);

	return block_pointer;
}

void Leaf::Deserialize(ART &art, MetaBlockReader &reader) {

	auto count_p = reader.Read<uint32_t>();
	prefix.Deserialize(art, reader);

	// inlined
	if (count_p == 1) {
		row_ids.inlined = reader.Read<row_t>();
		count = count_p;
		return;
	}

	// copy into segments
	count = 0;
	reference<LeafSegment> segment(LeafSegment::New(art, row_ids.ptr));
	for (idx_t i = 0; i < count_p; i++) {
		segment = segment.get().Append(art, count, reader.Read<row_t>());
	}
	D_ASSERT(count_p == count);
}

void Leaf::Vacuum(ART &art) {

	if (IsInlined()) {
		return;
	}

	// first pointer has special treatment because we don't obtain it from a leaf segment
	auto &allocator = Node::GetAllocator(art, NType::LEAF_SEGMENT);
	if (allocator.NeedsVacuum(row_ids.ptr)) {
		row_ids.ptr.SetPtr(allocator.VacuumPointer(row_ids.ptr));
	}

	auto ptr = row_ids.ptr;
	while (ptr.IsSet()) {
		auto &segment = LeafSegment::Get(art, ptr);
		ptr = segment.next;
		if (ptr.IsSet() && allocator.NeedsVacuum(ptr)) {
			segment.next.SetPtr(allocator.VacuumPointer(ptr));
			ptr = segment.next;
		}
	}
}

void Leaf::MoveInlinedToSegment(ART &art) {

	D_ASSERT(IsInlined());

	auto row_id = row_ids.inlined;
	auto &segment = LeafSegment::New(art, row_ids.ptr);
	segment.row_ids[0] = row_id;
}

} // namespace duckdb





namespace duckdb {

LeafSegment &LeafSegment::New(ART &art, Node &node) {

	node.SetPtr(Node::GetAllocator(art, NType::LEAF_SEGMENT).New());
	node.type = (uint8_t)NType::LEAF_SEGMENT;

	auto &segment = LeafSegment::Get(art, node);
	segment.next.Reset();
	return segment;
}

LeafSegment &LeafSegment::Append(ART &art, uint32_t &count, const row_t row_id) {

	reference<LeafSegment> segment(*this);
	auto position = count % Node::LEAF_SEGMENT_SIZE;

	// we need a new segment
	if (position == 0 && count != 0) {
		segment = LeafSegment::New(art, next);
	}

	segment.get().row_ids[position] = row_id;
	count++;
	return segment.get();
}

LeafSegment &LeafSegment::GetTail(const ART &art) {

	reference<LeafSegment> segment(*this);
	while (segment.get().next.IsSet()) {
		segment = LeafSegment::Get(art, segment.get().next);
	}
	return segment.get();
}

} // namespace duckdb
















namespace duckdb {

//===--------------------------------------------------------------------===//
// Constructors / Destructors
//===--------------------------------------------------------------------===//

Node::Node() : SwizzleablePointer() {
}

Node::Node(MetaBlockReader &reader) : SwizzleablePointer(reader) {
}

void Node::New(ART &art, Node &node, const NType type) {

	switch (type) {
	case NType::PREFIX_SEGMENT:
		PrefixSegment::New(art, node);
		break;
	case NType::LEAF_SEGMENT:
		LeafSegment::New(art, node);
		break;
	case NType::NODE_4:
		Node4::New(art, node);
		break;
	case NType::NODE_16:
		Node16::New(art, node);
		break;
	case NType::NODE_48:
		Node48::New(art, node);
		break;
	case NType::NODE_256:
		Node256::New(art, node);
		break;
	default:
		throw InternalException("Invalid node type for New.");
	}
}

void Node::Free(ART &art, Node &node) {

	// recursively free all nodes that are in-memory, and skip swizzled and empty nodes

	if (!node.IsSet()) {
		return;
	}

	if (!node.IsSwizzled()) {

		auto type = node.DecodeARTNodeType();
		if (type != NType::PREFIX_SEGMENT && type != NType::LEAF_SEGMENT) {
			node.GetPrefix(art).Free(art);
		}

		// free the prefixes and children of the nodes
		switch (type) {
		case NType::LEAF:
			Leaf::Free(art, node);
			break;
		case NType::NODE_4:
			Node4::Free(art, node);
			break;
		case NType::NODE_16:
			Node16::Free(art, node);
			break;
		case NType::NODE_48:
			Node48::Free(art, node);
			break;
		case NType::NODE_256:
			Node256::Free(art, node);
			break;
		default:
			break;
		}

		Node::GetAllocator(art, type).Free(node);
	}

	// overwrite with an empty ART node
	node.Reset();
}

//===--------------------------------------------------------------------===//
// Inserts
//===--------------------------------------------------------------------===//

void Node::ReplaceChild(const ART &art, const uint8_t byte, const Node child) {

	D_ASSERT(!IsSwizzled());

	switch (DecodeARTNodeType()) {
	case NType::NODE_4:
		return Node4::Get(art, *this).ReplaceChild(byte, child);
	case NType::NODE_16:
		return Node16::Get(art, *this).ReplaceChild(byte, child);
	case NType::NODE_48:
		return Node48::Get(art, *this).ReplaceChild(byte, child);
	case NType::NODE_256:
		return Node256::Get(art, *this).ReplaceChild(byte, child);
	default:
		throw InternalException("Invalid node type for ReplaceChild.");
	}
}

void Node::InsertChild(ART &art, Node &node, const uint8_t byte, const Node child) {

	switch (node.DecodeARTNodeType()) {
	case NType::NODE_4:
		return Node4::InsertChild(art, node, byte, child);
	case NType::NODE_16:
		return Node16::InsertChild(art, node, byte, child);
	case NType::NODE_48:
		return Node48::InsertChild(art, node, byte, child);
	case NType::NODE_256:
		return Node256::InsertChild(art, node, byte, child);
	default:
		throw InternalException("Invalid node type for InsertChild.");
	}
}

//===--------------------------------------------------------------------===//
// Deletes
//===--------------------------------------------------------------------===//

void Node::DeleteChild(ART &art, Node &node, const uint8_t byte) {

	switch (node.DecodeARTNodeType()) {
	case NType::NODE_4:
		return Node4::DeleteChild(art, node, byte);
	case NType::NODE_16:
		return Node16::DeleteChild(art, node, byte);
	case NType::NODE_48:
		return Node48::DeleteChild(art, node, byte);
	case NType::NODE_256:
		return Node256::DeleteChild(art, node, byte);
	default:
		throw InternalException("Invalid node type for DeleteChild.");
	}
}

//===--------------------------------------------------------------------===//
// Get functions
//===--------------------------------------------------------------------===//

optional_ptr<Node> Node::GetChild(ART &art, const uint8_t byte) const {

	D_ASSERT(!IsSwizzled());

	optional_ptr<Node> child;
	switch (DecodeARTNodeType()) {
	case NType::NODE_4: {
		child = Node4::Get(art, *this).GetChild(byte);
		break;
	}
	case NType::NODE_16: {
		child = Node16::Get(art, *this).GetChild(byte);
		break;
	}
	case NType::NODE_48: {
		child = Node48::Get(art, *this).GetChild(byte);
		break;
	}
	case NType::NODE_256: {
		child = Node256::Get(art, *this).GetChild(byte);
		break;
	}
	default:
		throw InternalException("Invalid node type for GetChild.");
	}

	// unswizzle the ART node before returning it
	if (child && child->IsSwizzled()) {
		child->Deserialize(art);
	}
	return child;
}

optional_ptr<Node> Node::GetNextChild(ART &art, uint8_t &byte) const {

	D_ASSERT(!IsSwizzled());

	optional_ptr<Node> child;
	switch (DecodeARTNodeType()) {
	case NType::NODE_4: {
		child = Node4::Get(art, *this).GetNextChild(byte);
		break;
	}
	case NType::NODE_16: {
		child = Node16::Get(art, *this).GetNextChild(byte);
		break;
	}
	case NType::NODE_48: {
		child = Node48::Get(art, *this).GetNextChild(byte);
		break;
	}
	case NType::NODE_256: {
		child = Node256::Get(art, *this).GetNextChild(byte);
		break;
	}
	default:
		throw InternalException("Invalid node type for GetNextChild.");
	}

	// unswizzle the ART node before returning it
	if (child && child->IsSwizzled()) {
		child->Deserialize(art);
	}
	return child;
}

//===--------------------------------------------------------------------===//
// (De)serialization
//===--------------------------------------------------------------------===//

BlockPointer Node::Serialize(ART &art, MetaBlockWriter &writer) {

	if (!IsSet()) {
		return {(block_id_t)DConstants::INVALID_INDEX, 0};
	}

	if (IsSwizzled()) {
		Deserialize(art);
	}

	switch (DecodeARTNodeType()) {
	case NType::LEAF:
		return Leaf::Get(art, *this).Serialize(art, writer);
	case NType::NODE_4:
		return Node4::Get(art, *this).Serialize(art, writer);
	case NType::NODE_16:
		return Node16::Get(art, *this).Serialize(art, writer);
	case NType::NODE_48:
		return Node48::Get(art, *this).Serialize(art, writer);
	case NType::NODE_256:
		return Node256::Get(art, *this).Serialize(art, writer);
	default:
		throw InternalException("Invalid node type for Serialize.");
	}
}

void Node::Deserialize(ART &art) {

	MetaBlockReader reader(art.table_io_manager.GetIndexBlockManager(), buffer_id);
	reader.offset = offset;
	type = reader.Read<uint8_t>();
	swizzle_flag = 0;

	auto type = DecodeARTNodeType();
	SetPtr(Node::GetAllocator(art, type).New());

	switch (type) {
	case NType::LEAF:
		return Leaf::Get(art, *this).Deserialize(art, reader);
	case NType::NODE_4:
		return Node4::Get(art, *this).Deserialize(art, reader);
	case NType::NODE_16:
		return Node16::Get(art, *this).Deserialize(art, reader);
	case NType::NODE_48:
		return Node48::Get(art, *this).Deserialize(art, reader);
	case NType::NODE_256:
		return Node256::Get(art, *this).Deserialize(art, reader);
	default:
		throw InternalException("Invalid node type for Deserialize.");
	}
}

//===--------------------------------------------------------------------===//
// Utility
//===--------------------------------------------------------------------===//

string Node::ToString(ART &art) const {

	D_ASSERT(!IsSwizzled());

	if (DecodeARTNodeType() == NType::LEAF) {
		return Leaf::Get(art, *this).ToString(art);
	}

	string str = "Node" + to_string(GetCapacity()) + ": [";

	uint8_t byte = 0;
	auto child = GetNextChild(art, byte);
	while (child) {
		str += "(" + to_string(byte) + ", " + child->ToString(art) + ")";
		if (byte == NumericLimits<uint8_t>::Maximum()) {
			break;
		}
		byte++;
		child = GetNextChild(art, byte);
	}

	return str + "]";
}

idx_t Node::GetCapacity() const {

	D_ASSERT(!IsSwizzled());

	switch (DecodeARTNodeType()) {
	case NType::NODE_4:
		return Node::NODE_4_CAPACITY;
	case NType::NODE_16:
		return Node::NODE_16_CAPACITY;
	case NType::NODE_48:
		return Node::NODE_48_CAPACITY;
	case NType::NODE_256:
		return Node::NODE_256_CAPACITY;
	default:
		throw InternalException("Invalid node type for GetCapacity.");
	}
}

Prefix &Node::GetPrefix(ART &art) {

	if (IsSwizzled()) {
		Deserialize(art);
	}

	switch (DecodeARTNodeType()) {
	case NType::LEAF:
		return Leaf::Get(art, *this).prefix;
	case NType::NODE_4:
		return Node4::Get(art, *this).prefix;
	case NType::NODE_16:
		return Node16::Get(art, *this).prefix;
	case NType::NODE_48:
		return Node48::Get(art, *this).prefix;
	case NType::NODE_256:
		return Node256::Get(art, *this).prefix;
	default:
		throw InternalException("Invalid node type for GetPrefix.");
	}
}

NType Node::GetARTNodeTypeByCount(const idx_t count) {

	if (count <= NODE_4_CAPACITY) {
		return NType::NODE_4;
	} else if (count <= NODE_16_CAPACITY) {
		return NType::NODE_16;
	} else if (count <= NODE_48_CAPACITY) {
		return NType::NODE_48;
	}
	return NType::NODE_256;
}

FixedSizeAllocator &Node::GetAllocator(const ART &art, NType type) {
	return *art.allocators[(uint8_t)type - 1];
}

//===--------------------------------------------------------------------===//
// Merging
//===--------------------------------------------------------------------===//

void Node::InitializeMerge(ART &art, const ARTFlags &flags) {

	if (!IsSet()) {
		return;
	}

	if (IsSwizzled()) {
		Deserialize(art);
	}

	// if not all prefixes are inlined
	if (flags.merge_buffer_counts[(uint8_t)NType::PREFIX_SEGMENT - 1] != 0) {
		// initialize prefix segments
		GetPrefix(art).InitializeMerge(art, flags.merge_buffer_counts[(uint8_t)NType::PREFIX_SEGMENT - 1]);
	}

	auto type = DecodeARTNodeType();
	switch (type) {
	case NType::LEAF:
		// if not all leaves are inlined
		if (flags.merge_buffer_counts[(uint8_t)NType::LEAF_SEGMENT - 1] != 0) {
			// initialize leaf segments
			Leaf::Get(art, *this).InitializeMerge(art, flags.merge_buffer_counts[(uint8_t)NType::LEAF_SEGMENT - 1]);
		}
		break;
	case NType::NODE_4:
		Node4::Get(art, *this).InitializeMerge(art, flags);
		break;
	case NType::NODE_16:
		Node16::Get(art, *this).InitializeMerge(art, flags);
		break;
	case NType::NODE_48:
		Node48::Get(art, *this).InitializeMerge(art, flags);
		break;
	case NType::NODE_256:
		Node256::Get(art, *this).InitializeMerge(art, flags);
		break;
	default:
		throw InternalException("Invalid node type for InitializeMerge.");
	}

	buffer_id += flags.merge_buffer_counts[(uint8_t)type - 1];
}

bool Node::Merge(ART &art, Node &other) {

	if (!IsSet()) {
		*this = other;
		other = Node();
		return true;
	}

	return ResolvePrefixes(art, other);
}

bool Node::ResolvePrefixes(ART &art, Node &other) {

	// NOTE: we always merge into the left ART

	D_ASSERT(IsSet());
	D_ASSERT(other.IsSet());

	// make sure that r_node has the longer (or equally long) prefix
	if (GetPrefix(art).count > other.GetPrefix(art).count) {
		swap(*this, other);
	}

	auto &l_node = *this;
	auto &r_node = other;
	auto &l_prefix = l_node.GetPrefix(art);
	auto &r_prefix = r_node.GetPrefix(art);

	auto mismatch_position = l_prefix.MismatchPosition(art, r_prefix);

	// both nodes have no prefix or the same prefix
	if (mismatch_position == l_prefix.count && l_prefix.count == r_prefix.count) {
		return MergeInternal(art, r_node);
	}

	if (mismatch_position == l_prefix.count) {
		// r_node's prefix contains l_node's prefix
		// l_node cannot be a leaf, otherwise the key represented by l_node would be a subset of another key
		// which is not possible by our construction
		D_ASSERT(l_node.DecodeARTNodeType() != NType::LEAF);

		// test if the next byte (mismatch_position) in r_node (longer prefix) exists in l_node
		auto mismatch_byte = r_prefix.GetByte(art, mismatch_position);
		auto child_node = l_node.GetChild(art, mismatch_byte);

		// update the prefix of r_node to only consist of the bytes after mismatch_position
		r_prefix.Reduce(art, mismatch_position);

		// insert r_node as a child of l_node at empty position
		if (!child_node) {
			Node::InsertChild(art, l_node, mismatch_byte, r_node);
			r_node.Reset();
			return true;
		}

		// recurse
		return child_node->ResolvePrefixes(art, r_node);
	}

	// prefixes differ, create new node and insert both nodes as children

	// create new node
	auto old_l_node = l_node;
	auto &new_n4 = Node4::New(art, l_node);
	new_n4.prefix.Initialize(art, l_prefix, mismatch_position);

	// insert old l_node, break up prefix of old l_node
	auto key_byte = l_prefix.Reduce(art, mismatch_position);
	Node4::InsertChild(art, l_node, key_byte, old_l_node);

	// insert r_node, break up prefix of r_node
	key_byte = r_prefix.Reduce(art, mismatch_position);
	Node4::InsertChild(art, l_node, key_byte, r_node);

	r_node.Reset();
	return true;
}

bool Node::MergeInternal(ART &art, Node &other) {

	D_ASSERT(IsSet());
	D_ASSERT(other.IsSet());

	// always try to merge the smaller node into the bigger node
	// because maybe there is enough free space in the bigger node to fit the smaller one
	// without too much recursion
	if (this->DecodeARTNodeType() < other.DecodeARTNodeType()) {
		swap(*this, other);
	}

	Node empty_node;
	auto &l_node = *this;
	auto &r_node = other;

	if (r_node.DecodeARTNodeType() == NType::LEAF) {
		D_ASSERT(l_node.DecodeARTNodeType() == NType::LEAF);

		if (art.IsUnique()) {
			return false;
		}

		Leaf::Get(art, *this).Merge(art, r_node);
		return true;
	}

	uint8_t byte = 0;
	auto r_child = r_node.GetNextChild(art, byte);

	// while r_node still has children to merge
	while (r_child) {
		auto l_child = l_node.GetChild(art, byte);
		if (!l_child) {
			// insert child at empty byte
			Node::InsertChild(art, l_node, byte, *r_child);
			r_node.ReplaceChild(art, byte, empty_node);

		} else {
			// recurse
			if (!l_child->ResolvePrefixes(art, *r_child)) {
				return false;
			}
		}

		if (byte == NumericLimits<uint8_t>::Maximum()) {
			break;
		}
		byte++;
		r_child = r_node.GetNextChild(art, byte);
	}

	Node::Free(art, r_node);
	return true;
}

//===--------------------------------------------------------------------===//
// Vacuum
//===--------------------------------------------------------------------===//

void Node::Vacuum(ART &art, Node &node, const ARTFlags &flags) {

	if (node.IsSwizzled()) {
		return;
	}

	// possibly vacuum prefix segments, if not all prefixes are inlined
	bool needs_vacuum = flags.vacuum_flags[(uint8_t)NType::PREFIX_SEGMENT - 1];
	if (needs_vacuum) {
		// vacuum prefix segments
		node.GetPrefix(art).Vacuum(art);
	}

	auto type = node.DecodeARTNodeType();
	auto &allocator = Node::GetAllocator(art, type);
	needs_vacuum = flags.vacuum_flags[node.type - 1] && allocator.NeedsVacuum(node);
	if (needs_vacuum) {
		node.SetPtr(allocator.VacuumPointer(node));
	}

	switch (type) {
	case NType::LEAF: {
		// possibly vacuum leaf segments, if not all leaves are inlined
		if (flags.vacuum_flags[(uint8_t)NType::LEAF_SEGMENT - 1]) {
			Leaf::Get(art, node).Vacuum(art);
		}
		return;
	}
	case NType::NODE_4:
		return Node4::Get(art, node).Vacuum(art, flags);
	case NType::NODE_16:
		return Node16::Get(art, node).Vacuum(art, flags);
	case NType::NODE_48:
		return Node48::Get(art, node).Vacuum(art, flags);
	case NType::NODE_256:
		return Node256::Get(art, node).Vacuum(art, flags);
	default:
		throw InternalException("Invalid node type for Vacuum.");
	}
}

} // namespace duckdb









namespace duckdb {

Node16 &Node16::New(ART &art, Node &node) {

	node.SetPtr(Node::GetAllocator(art, NType::NODE_16).New());
	node.type = (uint8_t)NType::NODE_16;
	auto &n16 = Node16::Get(art, node);

	n16.count = 0;
	n16.prefix.Initialize();
	return n16;
}

void Node16::Free(ART &art, Node &node) {

	D_ASSERT(node.IsSet());
	D_ASSERT(!node.IsSwizzled());

	auto &n16 = Node16::Get(art, node);

	// free all children
	for (idx_t i = 0; i < n16.count; i++) {
		Node::Free(art, n16.children[i]);
	}
}

Node16 &Node16::GrowNode4(ART &art, Node &node16, Node &node4) {

	auto &n4 = Node4::Get(art, node4);
	auto &n16 = Node16::New(art, node16);

	n16.count = n4.count;
	n16.prefix.Move(n4.prefix);

	for (idx_t i = 0; i < n4.count; i++) {
		n16.key[i] = n4.key[i];
		n16.children[i] = n4.children[i];
	}

	n4.count = 0;
	Node::Free(art, node4);
	return n16;
}

Node16 &Node16::ShrinkNode48(ART &art, Node &node16, Node &node48) {

	auto &n16 = Node16::New(art, node16);
	auto &n48 = Node48::Get(art, node48);

	n16.count = 0;
	n16.prefix.Move(n48.prefix);

	for (idx_t i = 0; i < Node::NODE_256_CAPACITY; i++) {
		if (n48.child_index[i] != Node::EMPTY_MARKER) {
			n16.key[n16.count] = i;
			n16.children[n16.count] = n48.children[n48.child_index[i]];
			n16.count++;
		}
	}

	n48.count = 0;
	Node::Free(art, node48);
	return n16;
}

void Node16::InitializeMerge(ART &art, const ARTFlags &flags) {

	for (idx_t i = 0; i < count; i++) {
		children[i].InitializeMerge(art, flags);
	}
}

void Node16::InsertChild(ART &art, Node &node, const uint8_t byte, const Node child) {

	D_ASSERT(node.IsSet());
	D_ASSERT(!node.IsSwizzled());
	auto &n16 = Node16::Get(art, node);

	// ensure that there is no other child at the same byte
	for (idx_t i = 0; i < n16.count; i++) {
		D_ASSERT(n16.key[i] != byte);
	}

	// insert new child node into node
	if (n16.count < Node::NODE_16_CAPACITY) {
		// still space, just insert the child
		idx_t child_pos = 0;
		while (child_pos < n16.count && n16.key[child_pos] < byte) {
			child_pos++;
		}
		// move children backwards to make space
		for (idx_t i = n16.count; i > child_pos; i--) {
			n16.key[i] = n16.key[i - 1];
			n16.children[i] = n16.children[i - 1];
		}

		n16.key[child_pos] = byte;
		n16.children[child_pos] = child;
		n16.count++;

	} else {
		// node is full, grow to Node48
		auto node16 = node;
		Node48::GrowNode16(art, node, node16);
		Node48::InsertChild(art, node, byte, child);
	}
}

void Node16::DeleteChild(ART &art, Node &node, const uint8_t byte) {

	D_ASSERT(node.IsSet());
	D_ASSERT(!node.IsSwizzled());
	auto &n16 = Node16::Get(art, node);

	idx_t child_pos = 0;
	for (; child_pos < n16.count; child_pos++) {
		if (n16.key[child_pos] == byte) {
			break;
		}
	}

	D_ASSERT(child_pos < n16.count);

	// free the child and decrease the count
	Node::Free(art, n16.children[child_pos]);
	n16.count--;

	// potentially move any children backwards
	for (idx_t i = child_pos; i < n16.count; i++) {
		n16.key[i] = n16.key[i + 1];
		n16.children[i] = n16.children[i + 1];
	}

	// shrink node to Node4
	if (n16.count < Node::NODE_4_CAPACITY) {
		auto node16 = node;
		Node4::ShrinkNode16(art, node, node16);
	}
}

void Node16::ReplaceChild(const uint8_t byte, const Node child) {
	for (idx_t i = 0; i < count; i++) {
		if (key[i] == byte) {
			children[i] = child;
			return;
		}
	}
}

optional_ptr<Node> Node16::GetChild(const uint8_t byte) {

	for (idx_t i = 0; i < count; i++) {
		if (key[i] == byte) {
			return &children[i];
		}
	}
	return nullptr;
}

optional_ptr<Node> Node16::GetNextChild(uint8_t &byte) {

	for (idx_t i = 0; i < count; i++) {
		if (key[i] >= byte) {
			byte = key[i];
			return &children[i];
		}
	}
	return nullptr;
}

BlockPointer Node16::Serialize(ART &art, MetaBlockWriter &writer) {

	// recurse into children and retrieve child block pointers
	vector<BlockPointer> child_block_pointers;
	for (idx_t i = 0; i < count; i++) {
		child_block_pointers.push_back(children[i].Serialize(art, writer));
	}
	for (idx_t i = count; i < Node::NODE_16_CAPACITY; i++) {
		child_block_pointers.emplace_back((block_id_t)DConstants::INVALID_INDEX, 0);
	}

	// get pointer and write fields
	auto block_pointer = writer.GetBlockPointer();
	writer.Write(NType::NODE_16);
	writer.Write<uint8_t>(count);
	prefix.Serialize(art, writer);

	// write key values
	for (idx_t i = 0; i < Node::NODE_16_CAPACITY; i++) {
		writer.Write(key[i]);
	}

	// write child block pointers
	for (auto &child_block_pointer : child_block_pointers) {
		writer.Write(child_block_pointer.block_id);
		writer.Write(child_block_pointer.offset);
	}

	return block_pointer;
}

void Node16::Deserialize(ART &art, MetaBlockReader &reader) {

	count = reader.Read<uint8_t>();
	prefix.Deserialize(art, reader);

	// read key values
	for (idx_t i = 0; i < Node::NODE_16_CAPACITY; i++) {
		key[i] = reader.Read<uint8_t>();
	}

	// read child block pointers
	for (idx_t i = 0; i < Node::NODE_16_CAPACITY; i++) {
		children[i] = Node(reader);
	}
}

void Node16::Vacuum(ART &art, const ARTFlags &flags) {

	for (idx_t i = 0; i < count; i++) {
		Node::Vacuum(art, children[i], flags);
	}
}

} // namespace duckdb








namespace duckdb {

Node256 &Node256::New(ART &art, Node &node) {

	node.SetPtr(Node::GetAllocator(art, NType::NODE_256).New());
	node.type = (uint8_t)NType::NODE_256;
	auto &n256 = Node256::Get(art, node);

	n256.count = 0;
	n256.prefix.Initialize();

	for (idx_t i = 0; i < Node::NODE_256_CAPACITY; i++) {
		n256.children[i].Reset();
	}

	return n256;
}

void Node256::Free(ART &art, Node &node) {

	D_ASSERT(node.IsSet());
	D_ASSERT(!node.IsSwizzled());

	auto &n256 = Node256::Get(art, node);

	if (!n256.count) {
		return;
	}

	// free all children
	for (idx_t i = 0; i < Node::NODE_256_CAPACITY; i++) {
		if (n256.children[i].IsSet()) {
			Node::Free(art, n256.children[i]);
		}
	}
}

Node256 &Node256::GrowNode48(ART &art, Node &node256, Node &node48) {

	auto &n48 = Node48::Get(art, node48);
	auto &n256 = Node256::New(art, node256);

	n256.count = n48.count;
	n256.prefix.Move(n48.prefix);

	for (idx_t i = 0; i < Node::NODE_256_CAPACITY; i++) {
		if (n48.child_index[i] != Node::EMPTY_MARKER) {
			n256.children[i] = n48.children[n48.child_index[i]];
		} else {
			n256.children[i].Reset();
		}
	}

	n48.count = 0;
	Node::Free(art, node48);
	return n256;
}

void Node256::InitializeMerge(ART &art, const ARTFlags &flags) {

	for (idx_t i = 0; i < Node::NODE_256_CAPACITY; i++) {
		if (children[i].IsSet()) {
			children[i].InitializeMerge(art, flags);
		}
	}
}

void Node256::InsertChild(ART &art, Node &node, const uint8_t byte, const Node child) {

	D_ASSERT(node.IsSet());
	D_ASSERT(!node.IsSwizzled());
	auto &n256 = Node256::Get(art, node);

	// ensure that there is no other child at the same byte
	D_ASSERT(!n256.children[byte].IsSet());

	n256.count++;
	n256.children[byte] = child;
}

void Node256::DeleteChild(ART &art, Node &node, const uint8_t byte) {

	D_ASSERT(node.IsSet());
	D_ASSERT(!node.IsSwizzled());
	auto &n256 = Node256::Get(art, node);

	// free the child and decrease the count
	Node::Free(art, n256.children[byte]);
	n256.count--;

	// shrink node to Node48
	if (n256.count <= Node::NODE_256_SHRINK_THRESHOLD) {
		auto node256 = node;
		Node48::ShrinkNode256(art, node, node256);
	}
}

optional_ptr<Node> Node256::GetNextChild(uint8_t &byte) {

	for (idx_t i = byte; i < Node::NODE_256_CAPACITY; i++) {
		if (children[i].IsSet()) {
			byte = i;
			return &children[i];
		}
	}
	return nullptr;
}

BlockPointer Node256::Serialize(ART &art, MetaBlockWriter &writer) {

	// recurse into children and retrieve child block pointers
	vector<BlockPointer> child_block_pointers;
	for (idx_t i = 0; i < Node::NODE_256_CAPACITY; i++) {
		child_block_pointers.push_back(children[i].Serialize(art, writer));
	}

	// get pointer and write fields
	auto block_pointer = writer.GetBlockPointer();
	writer.Write(NType::NODE_256);
	writer.Write<uint16_t>(count);
	prefix.Serialize(art, writer);

	// write child block pointers
	for (auto &child_block_pointer : child_block_pointers) {
		writer.Write(child_block_pointer.block_id);
		writer.Write(child_block_pointer.offset);
	}

	return block_pointer;
}

void Node256::Deserialize(ART &art, MetaBlockReader &reader) {

	count = reader.Read<uint16_t>();
	prefix.Deserialize(art, reader);

	// read child block pointers
	for (idx_t i = 0; i < Node::NODE_256_CAPACITY; i++) {
		children[i] = Node(reader);
	}
}

void Node256::Vacuum(ART &art, const ARTFlags &flags) {

	for (idx_t i = 0; i < Node::NODE_256_CAPACITY; i++) {
		if (children[i].IsSet()) {
			Node::Vacuum(art, children[i], flags);
		}
	}
}

} // namespace duckdb








namespace duckdb {

Node4 &Node4::New(ART &art, Node &node) {

	node.SetPtr(Node::GetAllocator(art, NType::NODE_4).New());
	node.type = (uint8_t)NType::NODE_4;
	auto &n4 = Node4::Get(art, node);

	n4.count = 0;
	n4.prefix.Initialize();
	return n4;
}

void Node4::Free(ART &art, Node &node) {

	D_ASSERT(node.IsSet());
	D_ASSERT(!node.IsSwizzled());

	auto &n4 = Node4::Get(art, node);

	// free all children
	for (idx_t i = 0; i < n4.count; i++) {
		Node::Free(art, n4.children[i]);
	}
}

Node4 &Node4::ShrinkNode16(ART &art, Node &node4, Node &node16) {

	auto &n4 = Node4::New(art, node4);
	auto &n16 = Node16::Get(art, node16);

	n4.count = n16.count;
	n4.prefix.Move(n16.prefix);

	for (idx_t i = 0; i < n16.count; i++) {
		n4.key[i] = n16.key[i];
		n4.children[i] = n16.children[i];
	}

	n16.count = 0;
	Node::Free(art, node16);
	return n4;
}

void Node4::InitializeMerge(ART &art, const ARTFlags &flags) {

	for (idx_t i = 0; i < count; i++) {
		children[i].InitializeMerge(art, flags);
	}
}

void Node4::InsertChild(ART &art, Node &node, const uint8_t byte, const Node child) {

	D_ASSERT(node.IsSet());
	D_ASSERT(!node.IsSwizzled());
	auto &n4 = Node4::Get(art, node);

	// ensure that there is no other child at the same byte
	for (idx_t i = 0; i < n4.count; i++) {
		D_ASSERT(n4.key[i] != byte);
	}

	// insert new child node into node
	if (n4.count < Node::NODE_4_CAPACITY) {
		// still space, just insert the child
		idx_t child_pos = 0;
		while (child_pos < n4.count && n4.key[child_pos] < byte) {
			child_pos++;
		}
		// move children backwards to make space
		for (idx_t i = n4.count; i > child_pos; i--) {
			n4.key[i] = n4.key[i - 1];
			n4.children[i] = n4.children[i - 1];
		}

		n4.key[child_pos] = byte;
		n4.children[child_pos] = child;
		n4.count++;

	} else {
		// node is full, grow to Node16
		auto node4 = node;
		Node16::GrowNode4(art, node, node4);
		Node16::InsertChild(art, node, byte, child);
	}
}

void Node4::DeleteChild(ART &art, Node &node, const uint8_t byte) {

	D_ASSERT(node.IsSet());
	D_ASSERT(!node.IsSwizzled());
	auto &n4 = Node4::Get(art, node);

	idx_t child_pos = 0;
	for (; child_pos < n4.count; child_pos++) {
		if (n4.key[child_pos] == byte) {
			break;
		}
	}

	D_ASSERT(child_pos < n4.count);
	D_ASSERT(n4.count > 1);

	// free the child and decrease the count
	Node::Free(art, n4.children[child_pos]);
	n4.count--;

	// potentially move any children backwards
	for (idx_t i = child_pos; i < n4.count; i++) {
		n4.key[i] = n4.key[i + 1];
		n4.children[i] = n4.children[i + 1];
	}

	// this is a one way node, compress
	if (n4.count == 1) {

		// get only child and concatenate prefixes
		auto child = *n4.GetChild(n4.key[0]);
		child.GetPrefix(art).Concatenate(art, n4.key[0], n4.prefix);
		n4.count--;

		Node::Free(art, node);
		node = child;
	}
}

void Node4::ReplaceChild(const uint8_t byte, const Node child) {
	for (idx_t i = 0; i < count; i++) {
		if (key[i] == byte) {
			children[i] = child;
			return;
		}
	}
}

optional_ptr<Node> Node4::GetChild(const uint8_t byte) {

	for (idx_t i = 0; i < count; i++) {
		if (key[i] == byte) {
			return &children[i];
		}
	}
	return nullptr;
}

optional_ptr<Node> Node4::GetNextChild(uint8_t &byte) {

	for (idx_t i = 0; i < count; i++) {
		if (key[i] >= byte) {
			byte = key[i];
			return &children[i];
		}
	}
	return nullptr;
}

BlockPointer Node4::Serialize(ART &art, MetaBlockWriter &writer) {

	// recurse into children and retrieve child block pointers
	vector<BlockPointer> child_block_pointers;
	for (idx_t i = 0; i < count; i++) {
		child_block_pointers.push_back(children[i].Serialize(art, writer));
	}
	for (idx_t i = count; i < Node::NODE_4_CAPACITY; i++) {
		child_block_pointers.emplace_back((block_id_t)DConstants::INVALID_INDEX, 0);
	}

	// get pointer and write fields
	auto block_pointer = writer.GetBlockPointer();
	writer.Write(NType::NODE_4);
	writer.Write<uint8_t>(count);
	prefix.Serialize(art, writer);

	// write key values
	for (idx_t i = 0; i < Node::NODE_4_CAPACITY; i++) {
		writer.Write(key[i]);
	}

	// write child block pointers
	for (auto &child_block_pointer : child_block_pointers) {
		writer.Write(child_block_pointer.block_id);
		writer.Write(child_block_pointer.offset);
	}

	return block_pointer;
}

void Node4::Deserialize(ART &art, MetaBlockReader &reader) {

	count = reader.Read<uint8_t>();
	prefix.Deserialize(art, reader);

	// read key values
	for (idx_t i = 0; i < Node::NODE_4_CAPACITY; i++) {
		key[i] = reader.Read<uint8_t>();
	}

	// read child block pointers
	for (idx_t i = 0; i < Node::NODE_4_CAPACITY; i++) {
		children[i] = Node(reader);
	}
}

void Node4::Vacuum(ART &art, const ARTFlags &flags) {

	for (idx_t i = 0; i < count; i++) {
		Node::Vacuum(art, children[i], flags);
	}
}

} // namespace duckdb









namespace duckdb {

Node48 &Node48::New(ART &art, Node &node) {

	node.SetPtr(Node::GetAllocator(art, NType::NODE_48).New());
	node.type = (uint8_t)NType::NODE_48;
	auto &n48 = Node48::Get(art, node);

	n48.count = 0;
	n48.prefix.Initialize();

	for (idx_t i = 0; i < Node::NODE_256_CAPACITY; i++) {
		n48.child_index[i] = Node::EMPTY_MARKER;
	}

	// necessary for faster child insertion/deletion
	for (idx_t i = 0; i < Node::NODE_48_CAPACITY; i++) {
		n48.children[i].Reset();
	}

	return n48;
}

void Node48::Free(ART &art, Node &node) {

	D_ASSERT(node.IsSet());
	D_ASSERT(!node.IsSwizzled());

	auto &n48 = Node48::Get(art, node);

	if (!n48.count) {
		return;
	}

	// free all children
	for (idx_t i = 0; i < Node::NODE_256_CAPACITY; i++) {
		if (n48.child_index[i] != Node::EMPTY_MARKER) {
			Node::Free(art, n48.children[n48.child_index[i]]);
		}
	}
}

Node48 &Node48::GrowNode16(ART &art, Node &node48, Node &node16) {

	auto &n16 = Node16::Get(art, node16);
	auto &n48 = Node48::New(art, node48);

	n48.count = n16.count;
	n48.prefix.Move(n16.prefix);

	for (idx_t i = 0; i < Node::NODE_256_CAPACITY; i++) {
		n48.child_index[i] = Node::EMPTY_MARKER;
	}

	for (idx_t i = 0; i < n16.count; i++) {
		n48.child_index[n16.key[i]] = i;
		n48.children[i] = n16.children[i];
	}

	// necessary for faster child insertion/deletion
	for (idx_t i = n16.count; i < Node::NODE_48_CAPACITY; i++) {
		n48.children[i].Reset();
	}

	n16.count = 0;
	Node::Free(art, node16);
	return n48;
}

Node48 &Node48::ShrinkNode256(ART &art, Node &node48, Node &node256) {

	auto &n48 = Node48::New(art, node48);
	auto &n256 = Node256::Get(art, node256);

	n48.count = 0;
	n48.prefix.Move(n256.prefix);

	for (idx_t i = 0; i < Node::NODE_256_CAPACITY; i++) {
		if (n256.children[i].IsSet()) {
			n48.child_index[i] = n48.count;
			n48.children[n48.count] = n256.children[i];
			n48.count++;
		} else {
			n48.child_index[i] = Node::EMPTY_MARKER;
		}
	}

	// necessary for faster child insertion/deletion
	for (idx_t i = n48.count; i < Node::NODE_48_CAPACITY; i++) {
		n48.children[i].Reset();
	}

	n256.count = 0;
	Node::Free(art, node256);
	return n48;
}

void Node48::InitializeMerge(ART &art, const ARTFlags &flags) {

	for (idx_t i = 0; i < Node::NODE_256_CAPACITY; i++) {
		if (child_index[i] != Node::EMPTY_MARKER) {
			children[child_index[i]].InitializeMerge(art, flags);
		}
	}
}

void Node48::InsertChild(ART &art, Node &node, const uint8_t byte, const Node child) {

	D_ASSERT(node.IsSet());
	D_ASSERT(!node.IsSwizzled());
	auto &n48 = Node48::Get(art, node);

	// ensure that there is no other child at the same byte
	D_ASSERT(n48.child_index[byte] == Node::EMPTY_MARKER);

	// insert new child node into node
	if (n48.count < Node::NODE_48_CAPACITY) {
		// still space, just insert the child
		idx_t child_pos = n48.count;
		if (n48.children[child_pos].IsSet()) {
			// find an empty position in the node list if the current position is occupied
			child_pos = 0;
			while (n48.children[child_pos].IsSet()) {
				child_pos++;
			}
		}
		n48.children[child_pos] = child;
		n48.child_index[byte] = child_pos;
		n48.count++;

	} else {
		// node is full, grow to Node256
		auto node48 = node;
		Node256::GrowNode48(art, node, node48);
		Node256::InsertChild(art, node, byte, child);
	}
}

void Node48::DeleteChild(ART &art, Node &node, const uint8_t byte) {

	D_ASSERT(node.IsSet());
	D_ASSERT(!node.IsSwizzled());
	auto &n48 = Node48::Get(art, node);

	// free the child and decrease the count
	Node::Free(art, n48.children[n48.child_index[byte]]);
	n48.child_index[byte] = Node::EMPTY_MARKER;
	n48.count--;

	// shrink node to Node16
	if (n48.count < Node::NODE_48_SHRINK_THRESHOLD) {
		auto node48 = node;
		Node16::ShrinkNode48(art, node, node48);
	}
}

optional_ptr<Node> Node48::GetNextChild(uint8_t &byte) {

	for (idx_t i = byte; i < Node::NODE_256_CAPACITY; i++) {
		if (child_index[i] != Node::EMPTY_MARKER) {
			byte = i;
			return &children[child_index[i]];
		}
	}
	return nullptr;
}

BlockPointer Node48::Serialize(ART &art, MetaBlockWriter &writer) {

	// recurse into children and retrieve child block pointers
	vector<BlockPointer> child_block_pointers;
	for (idx_t i = 0; i < Node::NODE_48_CAPACITY; i++) {
		child_block_pointers.push_back(children[i].Serialize(art, writer));
	}

	// get pointer and write fields
	auto block_pointer = writer.GetBlockPointer();
	writer.Write(NType::NODE_48);
	writer.Write<uint8_t>(count);
	prefix.Serialize(art, writer);

	// write key values
	for (idx_t i = 0; i < Node::NODE_256_CAPACITY; i++) {
		writer.Write(child_index[i]);
	}

	// write child block pointers
	for (auto &child_block_pointer : child_block_pointers) {
		writer.Write(child_block_pointer.block_id);
		writer.Write(child_block_pointer.offset);
	}

	return block_pointer;
}

void Node48::Deserialize(ART &art, MetaBlockReader &reader) {

	count = reader.Read<uint8_t>();
	prefix.Deserialize(art, reader);

	// read key values
	for (idx_t i = 0; i < Node::NODE_256_CAPACITY; i++) {
		child_index[i] = reader.Read<uint8_t>();
	}

	// read child block pointers
	for (idx_t i = 0; i < Node::NODE_48_CAPACITY; i++) {
		children[i] = Node(reader);
	}
}

void Node48::Vacuum(ART &art, const ARTFlags &flags) {

	for (idx_t i = 0; i < Node::NODE_256_CAPACITY; i++) {
		if (child_index[i] != Node::EMPTY_MARKER) {
			Node::Vacuum(art, children[child_index[i]], flags);
		}
	}
}

} // namespace duckdb









namespace duckdb {

void Prefix::Free(ART &art) {

	if (IsInlined()) {
		return Initialize();
	}

	// delete all prefix segments
	auto ptr = data.ptr;
	while (ptr.IsSet()) {
		auto next_ptr = PrefixSegment::Get(art, ptr).next;
		Node::Free(art, ptr);
		ptr = next_ptr;
	}

	Initialize();
}

void Prefix::Initialize(ART &art, const ARTKey &key, const uint32_t depth, const uint32_t count_p) {

	// prefix can be inlined
	if (count_p <= Node::PREFIX_INLINE_BYTES) {
		memcpy(data.inlined, key.data + depth, count_p);
		count = count_p;
		return;
	}

	// prefix cannot be inlined, copy to segment(s)
	count = 0;
	reference<PrefixSegment> segment(PrefixSegment::New(art, data.ptr));
	for (idx_t i = 0; i < count_p; i++) {
		segment = segment.get().Append(art, count, key.data[depth + i]);
	}
	D_ASSERT(count == count_p);
}

void Prefix::Initialize(ART &art, const Prefix &other, const uint32_t count_p) {

	D_ASSERT(count_p <= other.count);

	// copy inlined data
	if (other.IsInlined()) {
		memcpy(data.inlined, other.data.inlined, count_p);
		count = count_p;
		return;
	}

	// initialize the count and get the first segment
	count = 0;
	reference<PrefixSegment> segment(PrefixSegment::New(art, data.ptr));

	// iterate the segments of the other prefix and copy their data
	auto other_ptr = other.data.ptr;
	auto remaining = count_p;

	while (remaining != 0) {
		D_ASSERT(other_ptr.IsSet());
		auto &other_segment = PrefixSegment::Get(art, other_ptr);
		auto copy_count = MinValue(Node::PREFIX_SEGMENT_SIZE, remaining);

		// copy the data
		for (idx_t i = 0; i < copy_count; i++) {
			segment = segment.get().Append(art, count, other_segment.bytes[i]);
		}

		// adjust the loop variables
		other_ptr = other_segment.next;
		remaining -= copy_count;
	}
	D_ASSERT(count == count_p);
}

void Prefix::InitializeMerge(ART &art, const idx_t buffer_count) {

	if (IsInlined()) {
		return;
	}

	reference<PrefixSegment> segment(PrefixSegment::Get(art, data.ptr));
	data.ptr.buffer_id += buffer_count;

	auto ptr = segment.get().next;
	while (ptr.IsSet()) {
		segment.get().next.buffer_id += buffer_count;
		segment = PrefixSegment::Get(art, ptr);
		ptr = segment.get().next;
	}
}

void Prefix::Append(ART &art, const Prefix &other) {

	// result fits into inlined data, i.e., both prefixes are also inlined
	if (count + other.count <= Node::PREFIX_INLINE_BYTES) {
		memcpy(data.inlined + count, other.data.inlined, other.count);
		count += other.count;
		return;
	}

	// this prefix is inlined, but will no longer be after appending the other prefix,
	// move the inlined bytes to the first prefix segment
	if (IsInlined()) {
		MoveInlinedToSegment(art);
	}

	// get the tail of the segments of this prefix
	reference<PrefixSegment> segment(PrefixSegment::Get(art, data.ptr).GetTail(art));

	// the other prefix is inlined
	if (other.IsInlined()) {
		for (idx_t i = 0; i < other.count; i++) {
			segment = segment.get().Append(art, count, other.data.inlined[i]);
		}
		return;
	}

	// iterate all segments of the other prefix and copy their data
	auto other_ptr = other.data.ptr;
	auto remaining = other.count;

	while (other_ptr.IsSet()) {
		auto &other_segment = PrefixSegment::Get(art, other_ptr);
		auto copy_count = MinValue(Node::PREFIX_SEGMENT_SIZE, remaining);

		// copy the data
		for (idx_t i = 0; i < copy_count; i++) {
			segment = segment.get().Append(art, count, other_segment.bytes[i]);
		}

		// adjust the loop variables
		other_ptr = other_segment.next;
		remaining -= copy_count;
	}
	D_ASSERT(remaining == 0);
}

void Prefix::Concatenate(ART &art, const uint8_t byte, const Prefix &other) {

	auto new_size = count + 1 + other.count;

	// overwrite into this prefix (both are inlined)
	if (new_size <= Node::PREFIX_INLINE_BYTES) {
		// move this prefix backwards
		memmove(data.inlined + other.count + 1, data.inlined, count);
		// copy byte
		data.inlined[other.count] = byte;
		// copy the other prefix into this prefix
		memcpy(data.inlined, other.data.inlined, other.count);
		count = new_size;
		return;
	}

	auto this_inlined = IsInlined();
	auto this_count = count;
	auto this_data = data;
	Initialize();

	// append the other prefix and possibly move the data to a segment
	Append(art, other);
	if (IsInlined()) {
		MoveInlinedToSegment(art);
	}

	// get the tail
	reference<PrefixSegment> segment(PrefixSegment::Get(art, data.ptr).GetTail(art));
	// append the byte
	segment = segment.get().Append(art, count, byte);

	if (this_inlined) {
		// append this prefix
		for (idx_t i = 0; i < this_count; i++) {
			segment = segment.get().Append(art, count, this_data.inlined[i]);
		}
		return;
	}

	// iterate all segments of this prefix, copy their data, and free them
	auto this_ptr = this_data.ptr;
	auto remaining = this_count;

	while (this_ptr.IsSet()) {
		auto &this_segment = PrefixSegment::Get(art, this_ptr);
		auto copy_count = MinValue(Node::PREFIX_SEGMENT_SIZE, remaining);

		// copy the data
		for (idx_t i = 0; i < copy_count; i++) {
			segment = segment.get().Append(art, count, this_segment.bytes[i]);
		}

		// adjust the loop variables
		Node::Free(art, this_ptr);
		this_ptr = this_segment.next;
		remaining -= copy_count;
	}
	D_ASSERT(remaining == 0);
}

uint8_t Prefix::Reduce(ART &art, const idx_t reduce_count) {

	auto new_count = count - reduce_count - 1;
	auto new_first_byte = GetByte(art, reduce_count);

	// prefix is now empty
	if (new_count == 0) {
		Free(art);
		return new_first_byte;
	}

	// was inlined, just move bytes
	if (IsInlined()) {
		memmove(data.inlined, data.inlined + reduce_count + 1, new_count);
		count = new_count;
		return new_first_byte;
	}

	count = 0;
	auto start = reduce_count + 1;
	auto offset = start % Node::PREFIX_SEGMENT_SIZE;
	auto remaining = new_count;

	// get the source segment, i.e., the segment that contains the byte at start
	reference<PrefixSegment> src_segment(PrefixSegment::Get(art, data.ptr));
	for (idx_t i = 0; i < start / Node::PREFIX_SEGMENT_SIZE; i++) {
		D_ASSERT(src_segment.get().next.IsSet());
		src_segment = PrefixSegment::Get(art, src_segment.get().next);
	}

	// iterate all segments starting at the source segment and shift their data
	reference<PrefixSegment> dst_segment(PrefixSegment::Get(art, data.ptr));
	while (true) {
		auto copy_count = MinValue(Node::PREFIX_SEGMENT_SIZE - offset, remaining);

		// copy the data
		for (idx_t i = offset; i < offset + copy_count; i++) {
			dst_segment = dst_segment.get().Append(art, count, src_segment.get().bytes[i]);
		}

		// adjust the loop variables
		offset = 0;
		remaining -= copy_count;
		if (remaining == 0) {
			break;
		}
		D_ASSERT(src_segment.get().next.IsSet());
		src_segment = PrefixSegment::Get(art, src_segment.get().next);
	}

	// possibly inline the data
	if (IsInlined()) {
		MoveSegmentToInlined(art);
	}

	return new_first_byte;
}

uint8_t Prefix::GetByte(const ART &art, const idx_t position) const {

	D_ASSERT(position < count);
	if (IsInlined()) {
		return data.inlined[position];
	}

	// get the correct segment
	reference<PrefixSegment> segment(PrefixSegment::Get(art, data.ptr));
	for (idx_t i = 0; i < position / Node::PREFIX_SEGMENT_SIZE; i++) {
		D_ASSERT(segment.get().next.IsSet());
		segment = PrefixSegment::Get(art, segment.get().next);
	}

	return segment.get().bytes[position % Node::PREFIX_SEGMENT_SIZE];
}

uint32_t Prefix::KeyMismatchPosition(const ART &art, const ARTKey &key, const uint32_t depth) const {

	if (IsInlined()) {
		for (idx_t mismatch_position = 0; mismatch_position < count; mismatch_position++) {
			D_ASSERT(depth + mismatch_position < key.len);
			if (key[depth + mismatch_position] != data.inlined[mismatch_position]) {
				return mismatch_position;
			}
		}
		return count;
	}

	uint32_t mismatch_position = 0;
	auto ptr = data.ptr;

	while (mismatch_position != count) {
		D_ASSERT(depth + mismatch_position < key.len);
		D_ASSERT(ptr.IsSet());

		auto &segment = PrefixSegment::Get(art, ptr);
		auto compare_count = MinValue(Node::PREFIX_SEGMENT_SIZE, count - mismatch_position);

		// compare bytes
		for (uint32_t i = 0; i < compare_count; i++) {
			if (key[depth + mismatch_position] != segment.bytes[i]) {
				return mismatch_position;
			}
			mismatch_position++;
		}

		// adjust loop variables
		ptr = segment.next;
	}
	return count;
}

uint32_t Prefix::MismatchPosition(const ART &art, const Prefix &other) const {

	D_ASSERT(count <= other.count);

	// case 1: both prefixes are inlined
	if (IsInlined() && other.IsInlined()) {
		for (uint32_t i = 0; i < count; i++) {
			if (data.inlined[i] != other.data.inlined[i]) {
				return i;
			}
		}
		return count;
	}

	// case 2: only this prefix is inlined
	if (IsInlined()) {
		// we only need the first segment of the other prefix
		auto &segment = PrefixSegment::Get(art, other.data.ptr);
		for (uint32_t i = 0; i < count; i++) {
			if (data.inlined[i] != segment.bytes[i]) {
				return i;
			}
		}
		return count;
	}

	// case 3: both prefixes are not inlined
	auto ptr = data.ptr;
	auto other_ptr = other.data.ptr;

	// iterate segments and compare bytes
	uint32_t mismatch_position = 0;
	while (ptr.IsSet()) {
		D_ASSERT(other_ptr.IsSet());
		auto &segment = PrefixSegment::Get(art, ptr);
		auto &other_segment = PrefixSegment::Get(art, other_ptr);

		// compare bytes
		auto compare_count = MinValue(Node::PREFIX_SEGMENT_SIZE, count - mismatch_position);
		for (uint32_t i = 0; i < compare_count; i++) {
			if (segment.bytes[i] != other_segment.bytes[i]) {
				return mismatch_position;
			}
			mismatch_position++;
		}

		// adjust loop variables
		ptr = segment.next;
		other_ptr = other_segment.next;
	}
	return count;
}

void Prefix::Serialize(const ART &art, MetaBlockWriter &writer) const {

	writer.Write(count);

	// write inlined data
	if (IsInlined()) {
		writer.WriteData(data.inlined, count);
		return;
	}

	D_ASSERT(data.ptr.IsSet());
	auto ptr = data.ptr;
	auto remaining = count;

	// iterate all prefix segments and write their bytes
	while (ptr.IsSet()) {
		auto &segment = PrefixSegment::Get(art, ptr);
		auto copy_count = MinValue(Node::PREFIX_SEGMENT_SIZE, remaining);

		// write the bytes
		writer.WriteData(segment.bytes, copy_count);

		// adjust loop variables
		remaining -= copy_count;
		ptr = segment.next;
	}
	D_ASSERT(remaining == 0);
}

void Prefix::Deserialize(ART &art, MetaBlockReader &reader) {

	auto count_p = reader.Read<uint32_t>();

	// copy into inlined data
	if (count_p <= Node::PREFIX_INLINE_BYTES) {
		reader.ReadData(data.inlined, count_p);
		count = count_p;
		return;
	}

	// copy into segments
	count = 0;
	reference<PrefixSegment> segment(PrefixSegment::New(art, data.ptr));
	for (idx_t i = 0; i < count_p; i++) {
		segment = segment.get().Append(art, count, reader.Read<uint8_t>());
	}
	D_ASSERT(count_p == count);
}

void Prefix::Vacuum(ART &art) {

	if (IsInlined()) {
		return;
	}

	// first pointer has special treatment because we don't obtain it from a prefix segment
	auto &allocator = Node::GetAllocator(art, NType::PREFIX_SEGMENT);
	if (allocator.NeedsVacuum(data.ptr)) {
		data.ptr.SetPtr(allocator.VacuumPointer(data.ptr));
	}

	auto ptr = data.ptr;
	while (ptr.IsSet()) {
		auto &segment = PrefixSegment::Get(art, ptr);
		ptr = segment.next;
		if (ptr.IsSet() && allocator.NeedsVacuum(ptr)) {
			segment.next.SetPtr(allocator.VacuumPointer(ptr));
			ptr = segment.next;
		}
	}
}

PrefixSegment &Prefix::MoveInlinedToSegment(ART &art) {

	D_ASSERT(IsInlined());

	Node ptr;
	auto &segment = PrefixSegment::New(art, ptr);

	// move data
	D_ASSERT(Node::PREFIX_SEGMENT_SIZE >= Node::PREFIX_INLINE_BYTES);
	memcpy(segment.bytes, data.inlined, count);
	data.ptr = ptr;
	return segment;
}

void Prefix::MoveSegmentToInlined(ART &art) {

	D_ASSERT(IsInlined());
	D_ASSERT(data.ptr.IsSet());

	auto ptr = data.ptr;
	auto &segment = PrefixSegment::Get(art, data.ptr);

	memcpy(data.inlined, segment.bytes, count);
	Node::Free(art, ptr);
}

} // namespace duckdb





namespace duckdb {

PrefixSegment &PrefixSegment::New(ART &art, Node &node) {

	node.SetPtr(Node::GetAllocator(art, NType::PREFIX_SEGMENT).New());
	node.type = (uint8_t)NType::PREFIX_SEGMENT;

	auto &segment = PrefixSegment::Get(art, node);
	segment.next.Reset();
	return segment;
}

PrefixSegment &PrefixSegment::Append(ART &art, uint32_t &count, const uint8_t byte) {

	reference<PrefixSegment> segment(*this);
	auto position = count % Node::PREFIX_SEGMENT_SIZE;

	// we need a new segment
	if (position == 0 && count != 0) {
		segment = PrefixSegment::New(art, next);
	}

	segment.get().bytes[position] = byte;
	count++;
	return segment.get();
}

PrefixSegment &PrefixSegment::GetTail(const ART &art) {

	reference<PrefixSegment> segment(*this);
	while (segment.get().next.IsSet()) {
		segment = PrefixSegment::Get(art, segment.get().next);
	}
	return segment.get();
}

} // namespace duckdb




namespace duckdb {

SwizzleablePointer::SwizzleablePointer(MetaBlockReader &reader) {

	idx_t block_id = reader.Read<block_id_t>();
	offset = reader.Read<uint32_t>();
	type = 0;

	if (block_id == DConstants::INVALID_INDEX) {
		swizzle_flag = 0;
		return;
	}

	buffer_id = (uint32_t)block_id;
	swizzle_flag = 1;
}

} // namespace duckdb









namespace duckdb {

using ValidityBytes = JoinHashTable::ValidityBytes;
using ScanStructure = JoinHashTable::ScanStructure;
using ProbeSpill = JoinHashTable::ProbeSpill;
using ProbeSpillLocalState = JoinHashTable::ProbeSpillLocalAppendState;

JoinHashTable::JoinHashTable(BufferManager &buffer_manager_p, const vector<JoinCondition> &conditions_p,
                             vector<LogicalType> btypes, JoinType type_p)
    : buffer_manager(buffer_manager_p), conditions(conditions_p), build_types(std::move(btypes)), entry_size(0),
      tuple_size(0), vfound(Value::BOOLEAN(false)), join_type(type_p), finalized(false), has_null(false),
      external(false), radix_bits(4), partition_start(0), partition_end(0) {
	for (auto &condition : conditions) {
		D_ASSERT(condition.left->return_type == condition.right->return_type);
		auto type = condition.left->return_type;
		if (condition.comparison == ExpressionType::COMPARE_EQUAL ||
		    condition.comparison == ExpressionType::COMPARE_NOT_DISTINCT_FROM ||
		    condition.comparison == ExpressionType::COMPARE_DISTINCT_FROM) {
			// all equality conditions should be at the front
			// all other conditions at the back
			// this assert checks that
			D_ASSERT(equality_types.size() == condition_types.size());
			equality_types.push_back(type);
		}

		predicates.push_back(condition.comparison);
		null_values_are_equal.push_back(condition.comparison == ExpressionType::COMPARE_DISTINCT_FROM ||
		                                condition.comparison == ExpressionType::COMPARE_NOT_DISTINCT_FROM);

		condition_types.push_back(type);
	}
	// at least one equality is necessary
	D_ASSERT(!equality_types.empty());

	// Types for the layout
	vector<LogicalType> layout_types(condition_types);
	layout_types.insert(layout_types.end(), build_types.begin(), build_types.end());
	if (IsRightOuterJoin(join_type)) {
		// full/right outer joins need an extra bool to keep track of whether or not a tuple has found a matching entry
		// we place the bool before the NEXT pointer
		layout_types.emplace_back(LogicalType::BOOLEAN);
	}
	layout_types.emplace_back(LogicalType::HASH);
	layout.Initialize(layout_types, false);

	const auto &offsets = layout.GetOffsets();
	tuple_size = offsets[condition_types.size() + build_types.size()];
	pointer_offset = offsets.back();
	entry_size = layout.GetRowWidth();

	data_collection = make_uniq<TupleDataCollection>(buffer_manager, layout);
	sink_collection =
	    make_uniq<RadixPartitionedTupleData>(buffer_manager, layout, radix_bits, layout.ColumnCount() - 1);
}

JoinHashTable::~JoinHashTable() {
}

void JoinHashTable::Merge(JoinHashTable &other) {
	{
		lock_guard<mutex> guard(data_lock);
		data_collection->Combine(*other.data_collection);
	}

	if (join_type == JoinType::MARK) {
		auto &info = correlated_mark_join_info;
		lock_guard<mutex> mj_lock(info.mj_lock);
		has_null = has_null || other.has_null;
		if (!info.correlated_types.empty()) {
			auto &other_info = other.correlated_mark_join_info;
			info.correlated_counts->Combine(*other_info.correlated_counts);
		}
	}

	sink_collection->Combine(*other.sink_collection);
}

void JoinHashTable::ApplyBitmask(Vector &hashes, idx_t count) {
	if (hashes.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		D_ASSERT(!ConstantVector::IsNull(hashes));
		auto indices = ConstantVector::GetData<hash_t>(hashes);
		*indices = *indices & bitmask;
	} else {
		hashes.Flatten(count);
		auto indices = FlatVector::GetData<hash_t>(hashes);
		for (idx_t i = 0; i < count; i++) {
			indices[i] &= bitmask;
		}
	}
}

void JoinHashTable::ApplyBitmask(Vector &hashes, const SelectionVector &sel, idx_t count, Vector &pointers) {
	UnifiedVectorFormat hdata;
	hashes.ToUnifiedFormat(count, hdata);

	auto hash_data = (hash_t *)hdata.data;
	auto result_data = FlatVector::GetData<data_ptr_t *>(pointers);
	auto main_ht = (data_ptr_t *)hash_map.get();
	for (idx_t i = 0; i < count; i++) {
		auto rindex = sel.get_index(i);
		auto hindex = hdata.sel->get_index(rindex);
		auto hash = hash_data[hindex];
		result_data[rindex] = main_ht + (hash & bitmask);
	}
}

void JoinHashTable::Hash(DataChunk &keys, const SelectionVector &sel, idx_t count, Vector &hashes) {
	if (count == keys.size()) {
		// no null values are filtered: use regular hash functions
		VectorOperations::Hash(keys.data[0], hashes, keys.size());
		for (idx_t i = 1; i < equality_types.size(); i++) {
			VectorOperations::CombineHash(hashes, keys.data[i], keys.size());
		}
	} else {
		// null values were filtered: use selection vector
		VectorOperations::Hash(keys.data[0], hashes, sel, count);
		for (idx_t i = 1; i < equality_types.size(); i++) {
			VectorOperations::CombineHash(hashes, keys.data[i], sel, count);
		}
	}
}

static idx_t FilterNullValues(UnifiedVectorFormat &vdata, const SelectionVector &sel, idx_t count,
                              SelectionVector &result) {
	idx_t result_count = 0;
	for (idx_t i = 0; i < count; i++) {
		auto idx = sel.get_index(i);
		auto key_idx = vdata.sel->get_index(idx);
		if (vdata.validity.RowIsValid(key_idx)) {
			result.set_index(result_count++, idx);
		}
	}
	return result_count;
}

idx_t JoinHashTable::PrepareKeys(DataChunk &keys, unsafe_unique_array<UnifiedVectorFormat> &key_data,
                                 const SelectionVector *&current_sel, SelectionVector &sel, bool build_side) {
	key_data = keys.ToUnifiedFormat();

	// figure out which keys are NULL, and create a selection vector out of them
	current_sel = FlatVector::IncrementalSelectionVector();
	idx_t added_count = keys.size();
	if (build_side && IsRightOuterJoin(join_type)) {
		// in case of a right or full outer join, we cannot remove NULL keys from the build side
		return added_count;
	}
	for (idx_t i = 0; i < keys.ColumnCount(); i++) {
		if (!null_values_are_equal[i]) {
			if (key_data[i].validity.AllValid()) {
				continue;
			}
			added_count = FilterNullValues(key_data[i], *current_sel, added_count, sel);
			// null values are NOT equal for this column, filter them out
			current_sel = &sel;
		}
	}
	return added_count;
}

void JoinHashTable::Build(PartitionedTupleDataAppendState &append_state, DataChunk &keys, DataChunk &payload) {
	D_ASSERT(!finalized);
	D_ASSERT(keys.size() == payload.size());
	if (keys.size() == 0) {
		return;
	}
	// special case: correlated mark join
	if (join_type == JoinType::MARK && !correlated_mark_join_info.correlated_types.empty()) {
		auto &info = correlated_mark_join_info;
		lock_guard<mutex> mj_lock(info.mj_lock);
		// Correlated MARK join
		// for the correlated mark join we need to keep track of COUNT(*) and COUNT(COLUMN) for each of the correlated
		// columns push into the aggregate hash table
		D_ASSERT(info.correlated_counts);
		info.group_chunk.SetCardinality(keys);
		for (idx_t i = 0; i < info.correlated_types.size(); i++) {
			info.group_chunk.data[i].Reference(keys.data[i]);
		}
		if (info.correlated_payload.data.empty()) {
			vector<LogicalType> types;
			types.push_back(keys.data[info.correlated_types.size()].GetType());
			info.correlated_payload.InitializeEmpty(types);
		}
		info.correlated_payload.SetCardinality(keys);
		info.correlated_payload.data[0].Reference(keys.data[info.correlated_types.size()]);
		AggregateHTAppendState append_state;
		info.correlated_counts->AddChunk(append_state, info.group_chunk, info.correlated_payload,
		                                 AggregateType::NON_DISTINCT);
	}

	// prepare the keys for processing
	unsafe_unique_array<UnifiedVectorFormat> key_data;
	const SelectionVector *current_sel;
	SelectionVector sel(STANDARD_VECTOR_SIZE);
	idx_t added_count = PrepareKeys(keys, key_data, current_sel, sel, true);
	if (added_count < keys.size()) {
		has_null = true;
	}
	if (added_count == 0) {
		return;
	}

	// hash the keys and obtain an entry in the list
	// note that we only hash the keys used in the equality comparison
	Vector hash_values(LogicalType::HASH);
	Hash(keys, *current_sel, added_count, hash_values);

	// build a chunk to append to the data collection [keys, payload, (optional "found" boolean), hash]
	DataChunk source_chunk;
	source_chunk.InitializeEmpty(layout.GetTypes());
	for (idx_t i = 0; i < keys.ColumnCount(); i++) {
		source_chunk.data[i].Reference(keys.data[i]);
	}
	idx_t col_offset = keys.ColumnCount();
	D_ASSERT(build_types.size() == payload.ColumnCount());
	for (idx_t i = 0; i < payload.ColumnCount(); i++) {
		source_chunk.data[col_offset + i].Reference(payload.data[i]);
	}
	col_offset += payload.ColumnCount();
	if (IsRightOuterJoin(join_type)) {
		// for FULL/RIGHT OUTER joins initialize the "found" boolean to false
		source_chunk.data[col_offset].Reference(vfound);
		col_offset++;
	}
	source_chunk.data[col_offset].Reference(hash_values);
	source_chunk.SetCardinality(keys);

	if (added_count < keys.size()) {
		source_chunk.Slice(*current_sel, added_count);
	}
	sink_collection->Append(append_state, source_chunk);
}

template <bool PARALLEL>
static inline void InsertHashesLoop(atomic<data_ptr_t> pointers[], const hash_t indices[], const idx_t count,
                                    const data_ptr_t key_locations[], const idx_t pointer_offset) {
	for (idx_t i = 0; i < count; i++) {
		const auto index = indices[i];
		if (PARALLEL) {
			data_ptr_t head;
			do {
				head = pointers[index];
				Store<data_ptr_t>(head, key_locations[i] + pointer_offset);
			} while (!std::atomic_compare_exchange_weak(&pointers[index], &head, key_locations[i]));
		} else {
			// set prev in current key to the value (NOTE: this will be nullptr if there is none)
			Store<data_ptr_t>(pointers[index], key_locations[i] + pointer_offset);

			// set pointer to current tuple
			pointers[index] = key_locations[i];
		}
	}
}

void JoinHashTable::InsertHashes(Vector &hashes, idx_t count, data_ptr_t key_locations[], bool parallel) {
	D_ASSERT(hashes.GetType().id() == LogicalType::HASH);

	// use bitmask to get position in array
	ApplyBitmask(hashes, count);

	hashes.Flatten(count);
	D_ASSERT(hashes.GetVectorType() == VectorType::FLAT_VECTOR);

	auto pointers = (atomic<data_ptr_t> *)hash_map.get();
	auto indices = FlatVector::GetData<hash_t>(hashes);

	if (parallel) {
		InsertHashesLoop<true>(pointers, indices, count, key_locations, pointer_offset);
	} else {
		InsertHashesLoop<false>(pointers, indices, count, key_locations, pointer_offset);
	}
}

void JoinHashTable::InitializePointerTable() {
	idx_t capacity = PointerTableCapacity(Count());
	D_ASSERT(IsPowerOfTwo(capacity));

	if (hash_map.get()) {
		// There is already a hash map
		auto current_capacity = hash_map.GetSize() / sizeof(data_ptr_t);
		if (capacity > current_capacity) {
			// Need more space
			hash_map = buffer_manager.GetBufferAllocator().Allocate(capacity * sizeof(data_ptr_t));
		} else {
			// Just use the current hash map
			capacity = current_capacity;
		}
	} else {
		// Allocate a hash map
		hash_map = buffer_manager.GetBufferAllocator().Allocate(capacity * sizeof(data_ptr_t));
	}
	D_ASSERT(hash_map.GetSize() == capacity * sizeof(data_ptr_t));

	// initialize HT with all-zero entries
	std::fill_n((data_ptr_t *)hash_map.get(), capacity, nullptr);

	bitmask = capacity - 1;
}

void JoinHashTable::Finalize(idx_t chunk_idx_from, idx_t chunk_idx_to, bool parallel) {
	// Pointer table should be allocated
	D_ASSERT(hash_map.get());

	Vector hashes(LogicalType::HASH);
	auto hash_data = FlatVector::GetData<hash_t>(hashes);

	TupleDataChunkIterator iterator(*data_collection, TupleDataPinProperties::KEEP_EVERYTHING_PINNED, chunk_idx_from,
	                                chunk_idx_to, false);
	const auto row_locations = iterator.GetRowLocations();
	do {
		const auto count = iterator.GetCurrentChunkCount();
		for (idx_t i = 0; i < count; i++) {
			hash_data[i] = Load<hash_t>(row_locations[i] + pointer_offset);
		}
		InsertHashes(hashes, count, row_locations, parallel);
	} while (iterator.Next());
}

unique_ptr<ScanStructure> JoinHashTable::InitializeScanStructure(DataChunk &keys, const SelectionVector *&current_sel) {
	D_ASSERT(Count() > 0); // should be handled before
	D_ASSERT(finalized);

	// set up the scan structure
	auto ss = make_uniq<ScanStructure>(*this);

	if (join_type != JoinType::INNER) {
		ss->found_match = make_unsafe_uniq_array<bool>(STANDARD_VECTOR_SIZE);
		memset(ss->found_match.get(), 0, sizeof(bool) * STANDARD_VECTOR_SIZE);
	}

	// first prepare the keys for probing
	ss->count = PrepareKeys(keys, ss->key_data, current_sel, ss->sel_vector, false);
	return ss;
}

unique_ptr<ScanStructure> JoinHashTable::Probe(DataChunk &keys, Vector *precomputed_hashes) {
	const SelectionVector *current_sel;
	auto ss = InitializeScanStructure(keys, current_sel);
	if (ss->count == 0) {
		return ss;
	}

	if (precomputed_hashes) {
		ApplyBitmask(*precomputed_hashes, *current_sel, ss->count, ss->pointers);
	} else {
		// hash all the keys
		Vector hashes(LogicalType::HASH);
		Hash(keys, *current_sel, ss->count, hashes);

		// now initialize the pointers of the scan structure based on the hashes
		ApplyBitmask(hashes, *current_sel, ss->count, ss->pointers);
	}

	// create the selection vector linking to only non-empty entries
	ss->InitializeSelectionVector(current_sel);

	return ss;
}

ScanStructure::ScanStructure(JoinHashTable &ht)
    : pointers(LogicalType::POINTER), sel_vector(STANDARD_VECTOR_SIZE), ht(ht), finished(false) {
}

void ScanStructure::Next(DataChunk &keys, DataChunk &left, DataChunk &result) {
	if (finished) {
		return;
	}
	switch (ht.join_type) {
	case JoinType::INNER:
	case JoinType::RIGHT:
		NextInnerJoin(keys, left, result);
		break;
	case JoinType::SEMI:
		NextSemiJoin(keys, left, result);
		break;
	case JoinType::MARK:
		NextMarkJoin(keys, left, result);
		break;
	case JoinType::ANTI:
		NextAntiJoin(keys, left, result);
		break;
	case JoinType::OUTER:
	case JoinType::LEFT:
		NextLeftJoin(keys, left, result);
		break;
	case JoinType::SINGLE:
		NextSingleJoin(keys, left, result);
		break;
	default:
		throw InternalException("Unhandled join type in JoinHashTable");
	}
}

idx_t ScanStructure::ResolvePredicates(DataChunk &keys, SelectionVector &match_sel, SelectionVector *no_match_sel) {
	// Start with the scan selection
	for (idx_t i = 0; i < this->count; ++i) {
		match_sel.set_index(i, this->sel_vector.get_index(i));
	}
	idx_t no_match_count = 0;

	return RowOperations::Match(keys, key_data.get(), ht.layout, pointers, ht.predicates, match_sel, this->count,
	                            no_match_sel, no_match_count);
}

idx_t ScanStructure::ScanInnerJoin(DataChunk &keys, SelectionVector &result_vector) {
	while (true) {
		// resolve the predicates for this set of keys
		idx_t result_count = ResolvePredicates(keys, result_vector, nullptr);

		// after doing all the comparisons set the found_match vector
		if (found_match) {
			for (idx_t i = 0; i < result_count; i++) {
				auto idx = result_vector.get_index(i);
				found_match[idx] = true;
			}
		}
		if (result_count > 0) {
			return result_count;
		}
		// no matches found: check the next set of pointers
		AdvancePointers();
		if (this->count == 0) {
			return 0;
		}
	}
}

void ScanStructure::AdvancePointers(const SelectionVector &sel, idx_t sel_count) {
	// now for all the pointers, we move on to the next set of pointers
	idx_t new_count = 0;
	auto ptrs = FlatVector::GetData<data_ptr_t>(this->pointers);
	for (idx_t i = 0; i < sel_count; i++) {
		auto idx = sel.get_index(i);
		ptrs[idx] = Load<data_ptr_t>(ptrs[idx] + ht.pointer_offset);
		if (ptrs[idx]) {
			this->sel_vector.set_index(new_count++, idx);
		}
	}
	this->count = new_count;
}

void ScanStructure::InitializeSelectionVector(const SelectionVector *&current_sel) {
	idx_t non_empty_count = 0;
	auto ptrs = FlatVector::GetData<data_ptr_t>(pointers);
	auto cnt = count;
	for (idx_t i = 0; i < cnt; i++) {
		const auto idx = current_sel->get_index(i);
		ptrs[idx] = Load<data_ptr_t>(ptrs[idx]);
		if (ptrs[idx]) {
			sel_vector.set_index(non_empty_count++, idx);
		}
	}
	count = non_empty_count;
}

void ScanStructure::AdvancePointers() {
	AdvancePointers(this->sel_vector, this->count);
}

void ScanStructure::GatherResult(Vector &result, const SelectionVector &result_vector,
                                 const SelectionVector &sel_vector, const idx_t count, const idx_t col_no) {
	ht.data_collection->Gather(pointers, sel_vector, count, col_no, result, result_vector);
}

void ScanStructure::GatherResult(Vector &result, const SelectionVector &sel_vector, const idx_t count,
                                 const idx_t col_idx) {
	GatherResult(result, *FlatVector::IncrementalSelectionVector(), sel_vector, count, col_idx);
}

void ScanStructure::NextInnerJoin(DataChunk &keys, DataChunk &left, DataChunk &result) {
	D_ASSERT(result.ColumnCount() == left.ColumnCount() + ht.build_types.size());
	if (this->count == 0) {
		// no pointers left to chase
		return;
	}

	SelectionVector result_vector(STANDARD_VECTOR_SIZE);

	idx_t result_count = ScanInnerJoin(keys, result_vector);
	if (result_count > 0) {
		if (IsRightOuterJoin(ht.join_type)) {
			// full/right outer join: mark join matches as FOUND in the HT
			auto ptrs = FlatVector::GetData<data_ptr_t>(pointers);
			for (idx_t i = 0; i < result_count; i++) {
				auto idx = result_vector.get_index(i);
				// NOTE: threadsan reports this as a data race because this can be set concurrently by separate threads
				// Technically it is, but it does not matter, since the only value that can be written is "true"
				Store<bool>(true, ptrs[idx] + ht.tuple_size);
			}
		}
		// matches were found
		// construct the result
		// on the LHS, we create a slice using the result vector
		result.Slice(left, result_vector, result_count);

		// on the RHS, we need to fetch the data from the hash table
		for (idx_t i = 0; i < ht.build_types.size(); i++) {
			auto &vector = result.data[left.ColumnCount() + i];
			D_ASSERT(vector.GetType() == ht.build_types[i]);
			GatherResult(vector, result_vector, result_count, i + ht.condition_types.size());
		}
		AdvancePointers();
	}
}

void ScanStructure::ScanKeyMatches(DataChunk &keys) {
	// the semi-join, anti-join and mark-join we handle a differently from the inner join
	// since there can be at most STANDARD_VECTOR_SIZE results
	// we handle the entire chunk in one call to Next().
	// for every pointer, we keep chasing pointers and doing comparisons.
	// this results in a boolean array indicating whether or not the tuple has a match
	SelectionVector match_sel(STANDARD_VECTOR_SIZE), no_match_sel(STANDARD_VECTOR_SIZE);
	while (this->count > 0) {
		// resolve the predicates for the current set of pointers
		idx_t match_count = ResolvePredicates(keys, match_sel, &no_match_sel);
		idx_t no_match_count = this->count - match_count;

		// mark each of the matches as found
		for (idx_t i = 0; i < match_count; i++) {
			found_match[match_sel.get_index(i)] = true;
		}
		// continue searching for the ones where we did not find a match yet
		AdvancePointers(no_match_sel, no_match_count);
	}
}

template <bool MATCH>
void ScanStructure::NextSemiOrAntiJoin(DataChunk &keys, DataChunk &left, DataChunk &result) {
	D_ASSERT(left.ColumnCount() == result.ColumnCount());
	D_ASSERT(keys.size() == left.size());
	// create the selection vector from the matches that were found
	SelectionVector sel(STANDARD_VECTOR_SIZE);
	idx_t result_count = 0;
	for (idx_t i = 0; i < keys.size(); i++) {
		if (found_match[i] == MATCH) {
			// part of the result
			sel.set_index(result_count++, i);
		}
	}
	// construct the final result
	if (result_count > 0) {
		// we only return the columns on the left side
		// reference the columns of the left side from the result
		result.Slice(left, sel, result_count);
	} else {
		D_ASSERT(result.size() == 0);
	}
}

void ScanStructure::NextSemiJoin(DataChunk &keys, DataChunk &left, DataChunk &result) {
	// first scan for key matches
	ScanKeyMatches(keys);
	// then construct the result from all tuples with a match
	NextSemiOrAntiJoin<true>(keys, left, result);

	finished = true;
}

void ScanStructure::NextAntiJoin(DataChunk &keys, DataChunk &left, DataChunk &result) {
	// first scan for key matches
	ScanKeyMatches(keys);
	// then construct the result from all tuples that did not find a match
	NextSemiOrAntiJoin<false>(keys, left, result);

	finished = true;
}

void ScanStructure::ConstructMarkJoinResult(DataChunk &join_keys, DataChunk &child, DataChunk &result) {
	// for the initial set of columns we just reference the left side
	result.SetCardinality(child);
	for (idx_t i = 0; i < child.ColumnCount(); i++) {
		result.data[i].Reference(child.data[i]);
	}
	auto &mark_vector = result.data.back();
	mark_vector.SetVectorType(VectorType::FLAT_VECTOR);
	// first we set the NULL values from the join keys
	// if there is any NULL in the keys, the result is NULL
	auto bool_result = FlatVector::GetData<bool>(mark_vector);
	auto &mask = FlatVector::Validity(mark_vector);
	for (idx_t col_idx = 0; col_idx < join_keys.ColumnCount(); col_idx++) {
		if (ht.null_values_are_equal[col_idx]) {
			continue;
		}
		UnifiedVectorFormat jdata;
		join_keys.data[col_idx].ToUnifiedFormat(join_keys.size(), jdata);
		if (!jdata.validity.AllValid()) {
			for (idx_t i = 0; i < join_keys.size(); i++) {
				auto jidx = jdata.sel->get_index(i);
				mask.Set(i, jdata.validity.RowIsValidUnsafe(jidx));
			}
		}
	}
	// now set the remaining entries to either true or false based on whether a match was found
	if (found_match) {
		for (idx_t i = 0; i < child.size(); i++) {
			bool_result[i] = found_match[i];
		}
	} else {
		memset(bool_result, 0, sizeof(bool) * child.size());
	}
	// if the right side contains NULL values, the result of any FALSE becomes NULL
	if (ht.has_null) {
		for (idx_t i = 0; i < child.size(); i++) {
			if (!bool_result[i]) {
				mask.SetInvalid(i);
			}
		}
	}
}

void ScanStructure::NextMarkJoin(DataChunk &keys, DataChunk &input, DataChunk &result) {
	D_ASSERT(result.ColumnCount() == input.ColumnCount() + 1);
	D_ASSERT(result.data.back().GetType() == LogicalType::BOOLEAN);
	// this method should only be called for a non-empty HT
	D_ASSERT(ht.Count() > 0);

	ScanKeyMatches(keys);
	if (ht.correlated_mark_join_info.correlated_types.empty()) {
		ConstructMarkJoinResult(keys, input, result);
	} else {
		auto &info = ht.correlated_mark_join_info;
		// there are correlated columns
		// first we fetch the counts from the aggregate hashtable corresponding to these entries
		D_ASSERT(keys.ColumnCount() == info.group_chunk.ColumnCount() + 1);
		info.group_chunk.SetCardinality(keys);
		for (idx_t i = 0; i < info.group_chunk.ColumnCount(); i++) {
			info.group_chunk.data[i].Reference(keys.data[i]);
		}
		info.correlated_counts->FetchAggregates(info.group_chunk, info.result_chunk);

		// for the initial set of columns we just reference the left side
		result.SetCardinality(input);
		for (idx_t i = 0; i < input.ColumnCount(); i++) {
			result.data[i].Reference(input.data[i]);
		}
		// create the result matching vector
		auto &last_key = keys.data.back();
		auto &result_vector = result.data.back();
		// first set the nullmask based on whether or not there were NULL values in the join key
		result_vector.SetVectorType(VectorType::FLAT_VECTOR);
		auto bool_result = FlatVector::GetData<bool>(result_vector);
		auto &mask = FlatVector::Validity(result_vector);
		switch (last_key.GetVectorType()) {
		case VectorType::CONSTANT_VECTOR:
			if (ConstantVector::IsNull(last_key)) {
				mask.SetAllInvalid(input.size());
			}
			break;
		case VectorType::FLAT_VECTOR:
			mask.Copy(FlatVector::Validity(last_key), input.size());
			break;
		default: {
			UnifiedVectorFormat kdata;
			last_key.ToUnifiedFormat(keys.size(), kdata);
			for (idx_t i = 0; i < input.size(); i++) {
				auto kidx = kdata.sel->get_index(i);
				mask.Set(i, kdata.validity.RowIsValid(kidx));
			}
			break;
		}
		}

		auto count_star = FlatVector::GetData<int64_t>(info.result_chunk.data[0]);
		auto count = FlatVector::GetData<int64_t>(info.result_chunk.data[1]);
		// set the entries to either true or false based on whether a match was found
		for (idx_t i = 0; i < input.size(); i++) {
			D_ASSERT(count_star[i] >= count[i]);
			bool_result[i] = found_match ? found_match[i] : false;
			if (!bool_result[i] && count_star[i] > count[i]) {
				// RHS has NULL value and result is false: set to null
				mask.SetInvalid(i);
			}
			if (count_star[i] == 0) {
				// count == 0, set nullmask to false (we know the result is false now)
				mask.SetValid(i);
			}
		}
	}
	finished = true;
}

void ScanStructure::NextLeftJoin(DataChunk &keys, DataChunk &left, DataChunk &result) {
	// a LEFT OUTER JOIN is identical to an INNER JOIN except all tuples that do
	// not have a match must return at least one tuple (with the right side set
	// to NULL in every column)
	NextInnerJoin(keys, left, result);
	if (result.size() == 0) {
		// no entries left from the normal join
		// fill in the result of the remaining left tuples
		// together with NULL values on the right-hand side
		idx_t remaining_count = 0;
		SelectionVector sel(STANDARD_VECTOR_SIZE);
		for (idx_t i = 0; i < left.size(); i++) {
			if (!found_match[i]) {
				sel.set_index(remaining_count++, i);
			}
		}
		if (remaining_count > 0) {
			// have remaining tuples
			// slice the left side with tuples that did not find a match
			result.Slice(left, sel, remaining_count);

			// now set the right side to NULL
			for (idx_t i = left.ColumnCount(); i < result.ColumnCount(); i++) {
				Vector &vec = result.data[i];
				vec.SetVectorType(VectorType::CONSTANT_VECTOR);
				ConstantVector::SetNull(vec, true);
			}
		}
		finished = true;
	}
}

void ScanStructure::NextSingleJoin(DataChunk &keys, DataChunk &input, DataChunk &result) {
	// single join
	// this join is similar to the semi join except that
	// (1) we actually return data from the RHS and
	// (2) we return NULL for that data if there is no match
	idx_t result_count = 0;
	SelectionVector result_sel(STANDARD_VECTOR_SIZE);
	SelectionVector match_sel(STANDARD_VECTOR_SIZE), no_match_sel(STANDARD_VECTOR_SIZE);
	while (this->count > 0) {
		// resolve the predicates for the current set of pointers
		idx_t match_count = ResolvePredicates(keys, match_sel, &no_match_sel);
		idx_t no_match_count = this->count - match_count;

		// mark each of the matches as found
		for (idx_t i = 0; i < match_count; i++) {
			// found a match for this index
			auto index = match_sel.get_index(i);
			found_match[index] = true;
			result_sel.set_index(result_count++, index);
		}
		// continue searching for the ones where we did not find a match yet
		AdvancePointers(no_match_sel, no_match_count);
	}
	// reference the columns of the left side from the result
	D_ASSERT(input.ColumnCount() > 0);
	for (idx_t i = 0; i < input.ColumnCount(); i++) {
		result.data[i].Reference(input.data[i]);
	}
	// now fetch the data from the RHS
	for (idx_t i = 0; i < ht.build_types.size(); i++) {
		auto &vector = result.data[input.ColumnCount() + i];
		// set NULL entries for every entry that was not found
		for (idx_t j = 0; j < input.size(); j++) {
			if (!found_match[j]) {
				FlatVector::SetNull(vector, j, true);
			}
		}
		// for the remaining values we fetch the values
		GatherResult(vector, result_sel, result_sel, result_count, i + ht.condition_types.size());
	}
	result.SetCardinality(input.size());

	// like the SEMI, ANTI and MARK join types, the SINGLE join only ever does one pass over the HT per input chunk
	finished = true;
}

void JoinHashTable::ScanFullOuter(JoinHTScanState &state, Vector &addresses, DataChunk &result) {
	// scan the HT starting from the current position and check which rows from the build side did not find a match
	auto key_locations = FlatVector::GetData<data_ptr_t>(addresses);
	idx_t found_entries = 0;

	auto &iterator = state.iterator;
	if (iterator.Done()) {
		return;
	}

	const auto row_locations = iterator.GetRowLocations();
	do {
		const auto count = iterator.GetCurrentChunkCount();
		for (idx_t i = state.offset_in_chunk; i < count; i++) {
			auto found_match = Load<bool>(row_locations[i] + tuple_size);
			if (!found_match) {
				key_locations[found_entries++] = row_locations[i];
				if (found_entries == STANDARD_VECTOR_SIZE) {
					state.offset_in_chunk = i + 1;
					break;
				}
			}
		}
		if (found_entries == STANDARD_VECTOR_SIZE) {
			break;
		}
		state.offset_in_chunk = 0;
	} while (iterator.Next());

	// now gather from the found rows
	if (found_entries == 0) {
		return;
	}
	result.SetCardinality(found_entries);
	idx_t left_column_count = result.ColumnCount() - build_types.size();
	const auto &sel_vector = *FlatVector::IncrementalSelectionVector();
	// set the left side as a constant NULL
	for (idx_t i = 0; i < left_column_count; i++) {
		Vector &vec = result.data[i];
		vec.SetVectorType(VectorType::CONSTANT_VECTOR);
		ConstantVector::SetNull(vec, true);
	}

	// gather the values from the RHS
	for (idx_t i = 0; i < build_types.size(); i++) {
		auto &vector = result.data[left_column_count + i];
		D_ASSERT(vector.GetType() == build_types[i]);
		const auto col_no = condition_types.size() + i;
		data_collection->Gather(addresses, sel_vector, found_entries, col_no, vector, sel_vector);
	}
}

idx_t JoinHashTable::FillWithHTOffsets(JoinHTScanState &state, Vector &addresses) {
	// iterate over HT
	auto key_locations = FlatVector::GetData<data_ptr_t>(addresses);
	idx_t key_count = 0;

	auto &iterator = state.iterator;
	const auto row_locations = iterator.GetRowLocations();
	do {
		const auto count = iterator.GetCurrentChunkCount();
		for (idx_t i = 0; i < count; i++) {
			key_locations[key_count + i] = row_locations[i];
		}
		key_count += count;
	} while (iterator.Next());

	return key_count;
}

bool JoinHashTable::RequiresExternalJoin(ClientConfig &config, vector<unique_ptr<JoinHashTable>> &local_hts) {
	total_count = 0;
	idx_t data_size = 0;
	for (auto &ht : local_hts) {
		auto &local_sink_collection = ht->GetSinkCollection();
		total_count += local_sink_collection.Count();
		data_size += local_sink_collection.SizeInBytes();
	}

	if (total_count == 0) {
		return false;
	}

	if (config.force_external) {
		// Do ~3 rounds if forcing external join to test all code paths
		auto data_size_per_round = (data_size + 2) / 3;
		auto count_per_round = (total_count + 2) / 3;
		max_ht_size = data_size_per_round + PointerTableSize(count_per_round);
		external = true;
	} else {
		auto ht_size = data_size + PointerTableSize(total_count);
		external = ht_size > max_ht_size;
	}
	return external;
}

void JoinHashTable::Unpartition() {
	for (auto &partition : sink_collection->GetPartitions()) {
		data_collection->Combine(*partition);
	}
}

bool JoinHashTable::RequiresPartitioning(ClientConfig &config, vector<unique_ptr<JoinHashTable>> &local_hts) {
	D_ASSERT(total_count != 0);
	D_ASSERT(external);

	idx_t num_partitions = RadixPartitioning::NumberOfPartitions(radix_bits);
	vector<idx_t> partition_counts(num_partitions, 0);
	vector<idx_t> partition_sizes(num_partitions, 0);
	for (auto &ht : local_hts) {
		const auto &local_partitions = ht->GetSinkCollection().GetPartitions();
		for (idx_t partition_idx = 0; partition_idx < num_partitions; partition_idx++) {
			auto &local_partition = local_partitions[partition_idx];
			partition_counts[partition_idx] += local_partition->Count();
			partition_sizes[partition_idx] += local_partition->SizeInBytes();
		}
	}

	// Figure out if we can fit all single partitions in memory
	idx_t max_partition_idx = 0;
	idx_t max_partition_size = 0;
	for (idx_t partition_idx = 0; partition_idx < num_partitions; partition_idx++) {
		const auto &partition_count = partition_counts[partition_idx];
		const auto &partition_size = partition_sizes[partition_idx];
		auto partition_ht_size = partition_size + PointerTableSize(partition_count);
		if (partition_ht_size > max_partition_size) {
			max_partition_size = partition_ht_size;
			max_partition_idx = partition_idx;
		}
	}

	if (config.force_external || max_partition_size > max_ht_size) {
		const auto partition_count = partition_counts[max_partition_idx];
		const auto partition_size = partition_sizes[max_partition_idx];

		const auto max_added_bits = 8 - radix_bits;
		idx_t added_bits;
		for (added_bits = 1; added_bits < max_added_bits; added_bits++) {
			double partition_multiplier = RadixPartitioning::NumberOfPartitions(added_bits);

			auto new_estimated_count = double(partition_count) / partition_multiplier;
			auto new_estimated_size = double(partition_size) / partition_multiplier;
			auto new_estimated_ht_size = new_estimated_size + PointerTableSize(new_estimated_count);

			if (new_estimated_ht_size <= double(max_ht_size) / 4) {
				// Aim for an estimated partition size of max_ht_size / 4
				break;
			}
		}
		radix_bits += added_bits;
		sink_collection =
		    make_uniq<RadixPartitionedTupleData>(buffer_manager, layout, radix_bits, layout.ColumnCount() - 1);
		return true;
	} else {
		return false;
	}
}

void JoinHashTable::Partition(JoinHashTable &global_ht) {
	auto new_sink_collection =
	    make_uniq<RadixPartitionedTupleData>(buffer_manager, layout, global_ht.radix_bits, layout.ColumnCount() - 1);
	sink_collection->Repartition(*new_sink_collection);
	sink_collection = std::move(new_sink_collection);
	global_ht.Merge(*this);
}

void JoinHashTable::Reset() {
	data_collection->Reset();
	finalized = false;
}

bool JoinHashTable::PrepareExternalFinalize() {
	if (finalized) {
		Reset();
	}

	const auto num_partitions = RadixPartitioning::NumberOfPartitions(radix_bits);
	if (partition_end == num_partitions) {
		return false;
	}

	// Start where we left off
	auto &partitions = sink_collection->GetPartitions();
	partition_start = partition_end;

	// Determine how many partitions we can do next (at least one)
	idx_t count = 0;
	idx_t data_size = 0;
	idx_t partition_idx;
	for (partition_idx = partition_start; partition_idx < num_partitions; partition_idx++) {
		auto incl_count = count + partitions[partition_idx]->Count();
		auto incl_data_size = data_size + partitions[partition_idx]->SizeInBytes();
		auto incl_ht_size = incl_data_size + PointerTableSize(incl_count);
		if (count > 0 && incl_ht_size > max_ht_size) {
			break;
		}
		count = incl_count;
		data_size = incl_data_size;
	}
	partition_end = partition_idx;

	// Move the partitions to the main data collection
	for (partition_idx = partition_start; partition_idx < partition_end; partition_idx++) {
		data_collection->Combine(*partitions[partition_idx]);
	}
	D_ASSERT(Count() == count);

	return true;
}

static void CreateSpillChunk(DataChunk &spill_chunk, DataChunk &keys, DataChunk &payload, Vector &hashes) {
	spill_chunk.Reset();
	idx_t spill_col_idx = 0;
	for (idx_t col_idx = 0; col_idx < keys.ColumnCount(); col_idx++) {
		spill_chunk.data[col_idx].Reference(keys.data[col_idx]);
	}
	spill_col_idx += keys.ColumnCount();
	for (idx_t col_idx = 0; col_idx < payload.data.size(); col_idx++) {
		spill_chunk.data[spill_col_idx + col_idx].Reference(payload.data[col_idx]);
	}
	spill_col_idx += payload.ColumnCount();
	spill_chunk.data[spill_col_idx].Reference(hashes);
}

unique_ptr<ScanStructure> JoinHashTable::ProbeAndSpill(DataChunk &keys, DataChunk &payload, ProbeSpill &probe_spill,
                                                       ProbeSpillLocalAppendState &spill_state,
                                                       DataChunk &spill_chunk) {
	// hash all the keys
	Vector hashes(LogicalType::HASH);
	Hash(keys, *FlatVector::IncrementalSelectionVector(), keys.size(), hashes);

	// find out which keys we can match with the current pinned partitions
	SelectionVector true_sel;
	SelectionVector false_sel;
	true_sel.Initialize();
	false_sel.Initialize();
	auto true_count = RadixPartitioning::Select(hashes, FlatVector::IncrementalSelectionVector(), keys.size(),
	                                            radix_bits, partition_end, &true_sel, &false_sel);
	auto false_count = keys.size() - true_count;

	CreateSpillChunk(spill_chunk, keys, payload, hashes);

	// can't probe these values right now, append to spill
	spill_chunk.Slice(false_sel, false_count);
	spill_chunk.Verify();
	probe_spill.Append(spill_chunk, spill_state);

	// slice the stuff we CAN probe right now
	hashes.Slice(true_sel, true_count);
	keys.Slice(true_sel, true_count);
	payload.Slice(true_sel, true_count);

	const SelectionVector *current_sel;
	auto ss = InitializeScanStructure(keys, current_sel);
	if (ss->count == 0) {
		return ss;
	}

	// now initialize the pointers of the scan structure based on the hashes
	ApplyBitmask(hashes, *current_sel, ss->count, ss->pointers);

	// create the selection vector linking to only non-empty entries
	ss->InitializeSelectionVector(current_sel);

	return ss;
}

ProbeSpill::ProbeSpill(JoinHashTable &ht, ClientContext &context, const vector<LogicalType> &probe_types)
    : ht(ht), context(context), probe_types(probe_types) {
	auto remaining_count = ht.GetSinkCollection().Count();
	auto remaining_data_size = ht.GetSinkCollection().SizeInBytes();
	auto remaining_ht_size = remaining_data_size + ht.PointerTableSize(remaining_count);
	if (remaining_ht_size <= ht.max_ht_size) {
		// No need to partition as we will only have one more probe round
		partitioned = false;
	} else {
		// More than one probe round to go, so we need to partition
		partitioned = true;
		global_partitions =
		    make_uniq<RadixPartitionedColumnData>(context, probe_types, ht.radix_bits, probe_types.size() - 1);
	}
	column_ids.reserve(probe_types.size());
	for (column_t column_id = 0; column_id < probe_types.size(); column_id++) {
		column_ids.emplace_back(column_id);
	}
}

ProbeSpillLocalState ProbeSpill::RegisterThread() {
	ProbeSpillLocalAppendState result;
	lock_guard<mutex> guard(lock);
	if (partitioned) {
		local_partitions.emplace_back(global_partitions->CreateShared());
		local_partition_append_states.emplace_back(make_uniq<PartitionedColumnDataAppendState>());
		local_partitions.back()->InitializeAppendState(*local_partition_append_states.back());

		result.local_partition = local_partitions.back().get();
		result.local_partition_append_state = local_partition_append_states.back().get();
	} else {
		local_spill_collections.emplace_back(
		    make_uniq<ColumnDataCollection>(BufferManager::GetBufferManager(context), probe_types));
		local_spill_append_states.emplace_back(make_uniq<ColumnDataAppendState>());
		local_spill_collections.back()->InitializeAppend(*local_spill_append_states.back());

		result.local_spill_collection = local_spill_collections.back().get();
		result.local_spill_append_state = local_spill_append_states.back().get();
	}
	return result;
}

void ProbeSpill::Append(DataChunk &chunk, ProbeSpillLocalAppendState &local_state) {
	if (partitioned) {
		local_state.local_partition->Append(*local_state.local_partition_append_state, chunk);
	} else {
		local_state.local_spill_collection->Append(*local_state.local_spill_append_state, chunk);
	}
}

void ProbeSpill::Finalize() {
	if (partitioned) {
		D_ASSERT(local_partitions.size() == local_partition_append_states.size());
		for (idx_t i = 0; i < local_partition_append_states.size(); i++) {
			local_partitions[i]->FlushAppendState(*local_partition_append_states[i]);
		}
		for (auto &local_partition : local_partitions) {
			global_partitions->Combine(*local_partition);
		}
		local_partitions.clear();
		local_partition_append_states.clear();
	} else {
		if (local_spill_collections.empty()) {
			global_spill_collection =
			    make_uniq<ColumnDataCollection>(BufferManager::GetBufferManager(context), probe_types);
		} else {
			global_spill_collection = std::move(local_spill_collections[0]);
			for (idx_t i = 1; i < local_spill_collections.size(); i++) {
				global_spill_collection->Combine(*local_spill_collections[i]);
			}
		}
		local_spill_collections.clear();
		local_spill_append_states.clear();
	}
}

void ProbeSpill::PrepareNextProbe() {
	if (partitioned) {
		auto &partitions = global_partitions->GetPartitions();
		if (partitions.empty() || ht.partition_start == partitions.size()) {
			// Can't probe, just make an empty one
			global_spill_collection =
			    make_uniq<ColumnDataCollection>(BufferManager::GetBufferManager(context), probe_types);
		} else {
			// Move specific partitions to the global spill collection
			global_spill_collection = std::move(partitions[ht.partition_start]);
			for (idx_t i = ht.partition_start + 1; i < ht.partition_end; i++) {
				auto &partition = partitions[i];
				if (global_spill_collection->Count() == 0) {
					global_spill_collection = std::move(partition);
				} else {
					global_spill_collection->Combine(*partition);
				}
			}
		}
	}
	consumer = make_uniq<ColumnDataConsumer>(*global_spill_collection, column_ids);
	consumer->InitializeScan();
}

} // namespace duckdb
