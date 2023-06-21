// See https://raw.githubusercontent.com/duckdb/duckdb/master/LICENSE for licensing information

#include "duckdb.hpp"
#include "duckdb-internal.hpp"
#ifndef DUCKDB_AMALGAMATION
#error header mismatch
#endif





namespace duckdb {

template <class T>
struct FirstState {
	T value;
	bool is_set;
	bool is_null;
};

struct FirstFunctionBase {
	template <class STATE>
	static void Initialize(STATE *state) {
		state->is_set = false;
		state->is_null = false;
	}

	static bool IgnoreNull() {
		return false;
	}
};

template <bool LAST, bool SKIP_NULLS>
struct FirstFunction : public FirstFunctionBase {
	template <class INPUT_TYPE, class STATE, class OP>
	static void Operation(STATE *state, AggregateInputData &, INPUT_TYPE *input, ValidityMask &mask, idx_t idx) {
		if (LAST || !state->is_set) {
			if (!mask.RowIsValid(idx)) {
				if (!SKIP_NULLS) {
					state->is_set = true;
				}
				state->is_null = true;
			} else {
				state->is_set = true;
				state->is_null = false;
				state->value = input[idx];
			}
		}
	}

	template <class INPUT_TYPE, class STATE, class OP>
	static void ConstantOperation(STATE *state, AggregateInputData &aggr_input_data, INPUT_TYPE *input,
	                              ValidityMask &mask, idx_t count) {
		Operation<INPUT_TYPE, STATE, OP>(state, aggr_input_data, input, mask, 0);
	}

	template <class STATE, class OP>
	static void Combine(const STATE &source, STATE *target, AggregateInputData &) {
		if (!target->is_set) {
			*target = source;
		}
	}

	template <class T, class STATE>
	static void Finalize(Vector &result, AggregateInputData &, STATE *state, T *target, ValidityMask &mask, idx_t idx) {
		if (!state->is_set || state->is_null) {
			mask.SetInvalid(idx);
		} else {
			target[idx] = state->value;
		}
	}
};

template <bool LAST, bool SKIP_NULLS>
struct FirstFunctionString : public FirstFunctionBase {
	template <class STATE>
	static void SetValue(STATE *state, AggregateInputData &input_data, string_t value, bool is_null) {
		if (LAST && state->is_set) {
			Destroy(input_data, state);
		}
		if (is_null) {
			if (!SKIP_NULLS) {
				state->is_set = true;
				state->is_null = true;
			}
		} else {
			state->is_set = true;
			state->is_null = false;
			if (value.IsInlined()) {
				state->value = value;
			} else {
				// non-inlined string, need to allocate space for it
				auto len = value.GetSize();
				auto ptr = new char[len];
				memcpy(ptr, value.GetData(), len);

				state->value = string_t(ptr, len);
			}
		}
	}

	template <class INPUT_TYPE, class STATE, class OP>
	static void Operation(STATE *state, AggregateInputData &input_data, INPUT_TYPE *input, ValidityMask &mask,
	                      idx_t idx) {
		if (LAST || !state->is_set) {
			SetValue(state, input_data, input[idx], !mask.RowIsValid(idx));
		}
	}

	template <class INPUT_TYPE, class STATE, class OP>
	static void ConstantOperation(STATE *state, AggregateInputData &aggr_input_data, INPUT_TYPE *input,
	                              ValidityMask &mask, idx_t count) {
		Operation<INPUT_TYPE, STATE, OP>(state, aggr_input_data, input, mask, 0);
	}

	template <class STATE, class OP>
	static void Combine(const STATE &source, STATE *target, AggregateInputData &input_data) {
		if (source.is_set && (LAST || !target->is_set)) {
			SetValue(target, input_data, source.value, source.is_null);
		}
	}

	template <class T, class STATE>
	static void Finalize(Vector &result, AggregateInputData &, STATE *state, T *target, ValidityMask &mask, idx_t idx) {
		if (!state->is_set || state->is_null) {
			mask.SetInvalid(idx);
		} else {
			target[idx] = StringVector::AddStringOrBlob(result, state->value);
		}
	}

	template <class STATE>
	static void Destroy(AggregateInputData &aggr_input_data, STATE *state) {
		if (state->is_set && !state->is_null && !state->value.IsInlined()) {
			delete[] state->value.GetData();
		}
	}
};

struct FirstStateVector {
	Vector *value;
};

template <bool LAST, bool SKIP_NULLS>
struct FirstVectorFunction {
	template <class STATE>
	static void Initialize(STATE *state) {
		state->value = nullptr;
	}

	template <class STATE>
	static void Destroy(AggregateInputData &aggr_input_data, STATE *state) {
		if (state->value) {
			delete state->value;
		}
	}
	static bool IgnoreNull() {
		return SKIP_NULLS;
	}

	template <class STATE>
	static void SetValue(STATE *state, Vector &input, const idx_t idx) {
		if (!state->value) {
			state->value = new Vector(input.GetType());
			state->value->SetVectorType(VectorType::CONSTANT_VECTOR);
		}
		sel_t selv = idx;
		SelectionVector sel(&selv);
		VectorOperations::Copy(input, *state->value, sel, 1, 0, 0);
	}

	static void Update(Vector inputs[], AggregateInputData &, idx_t input_count, Vector &state_vector, idx_t count) {
		auto &input = inputs[0];
		UnifiedVectorFormat idata;
		input.ToUnifiedFormat(count, idata);

		UnifiedVectorFormat sdata;
		state_vector.ToUnifiedFormat(count, sdata);

		auto states = (FirstStateVector **)sdata.data;
		for (idx_t i = 0; i < count; i++) {
			const auto idx = idata.sel->get_index(i);
			if (SKIP_NULLS && !idata.validity.RowIsValid(idx)) {
				continue;
			}
			auto state = states[sdata.sel->get_index(i)];
			if (LAST || !state->value) {
				SetValue(state, input, i);
			}
		}
	}

	template <class STATE, class OP>
	static void Combine(const STATE &source, STATE *target, AggregateInputData &) {
		if (source.value && (LAST || !target->value)) {
			SetValue(target, *source.value, 0);
		}
	}

	template <class T, class STATE>
	static void Finalize(Vector &result, AggregateInputData &, STATE *state, T *target, ValidityMask &mask, idx_t idx) {
		if (!state->value) {
			// we need to use FlatVector::SetNull here
			// since for STRUCT columns only setting the validity mask of the struct is incorrect
			// as for a struct column, we need to also set ALL child columns to NULL
			if (result.GetVectorType() == VectorType::CONSTANT_VECTOR) {
				ConstantVector::SetNull(result, true);
			} else {
				FlatVector::SetNull(result, idx, true);
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

template <class T, bool LAST, bool SKIP_NULLS>
static AggregateFunction GetFirstAggregateTemplated(LogicalType type) {
	return AggregateFunction::UnaryAggregate<FirstState<T>, T, T, FirstFunction<LAST, SKIP_NULLS>>(type, type);
}

template <bool LAST, bool SKIP_NULLS>
static AggregateFunction GetFirstFunction(const LogicalType &type);

template <bool LAST, bool SKIP_NULLS>
AggregateFunction GetDecimalFirstFunction(const LogicalType &type) {
	D_ASSERT(type.id() == LogicalTypeId::DECIMAL);
	switch (type.InternalType()) {
	case PhysicalType::INT16:
		return GetFirstFunction<LAST, SKIP_NULLS>(LogicalType::SMALLINT);
	case PhysicalType::INT32:
		return GetFirstFunction<LAST, SKIP_NULLS>(LogicalType::INTEGER);
	case PhysicalType::INT64:
		return GetFirstFunction<LAST, SKIP_NULLS>(LogicalType::BIGINT);
	default:
		return GetFirstFunction<LAST, SKIP_NULLS>(LogicalType::HUGEINT);
	}
}

template <bool LAST, bool SKIP_NULLS>
static AggregateFunction GetFirstFunction(const LogicalType &type) {
	switch (type.id()) {
	case LogicalTypeId::BOOLEAN:
		return GetFirstAggregateTemplated<int8_t, LAST, SKIP_NULLS>(type);
	case LogicalTypeId::TINYINT:
		return GetFirstAggregateTemplated<int8_t, LAST, SKIP_NULLS>(type);
	case LogicalTypeId::SMALLINT:
		return GetFirstAggregateTemplated<int16_t, LAST, SKIP_NULLS>(type);
	case LogicalTypeId::INTEGER:
	case LogicalTypeId::DATE:
		return GetFirstAggregateTemplated<int32_t, LAST, SKIP_NULLS>(type);
	case LogicalTypeId::BIGINT:
	case LogicalTypeId::TIME:
	case LogicalTypeId::TIMESTAMP:
	case LogicalTypeId::TIME_TZ:
	case LogicalTypeId::TIMESTAMP_TZ:
		return GetFirstAggregateTemplated<int64_t, LAST, SKIP_NULLS>(type);
	case LogicalTypeId::UTINYINT:
		return GetFirstAggregateTemplated<uint8_t, LAST, SKIP_NULLS>(type);
	case LogicalTypeId::USMALLINT:
		return GetFirstAggregateTemplated<uint16_t, LAST, SKIP_NULLS>(type);
	case LogicalTypeId::UINTEGER:
		return GetFirstAggregateTemplated<uint32_t, LAST, SKIP_NULLS>(type);
	case LogicalTypeId::UBIGINT:
		return GetFirstAggregateTemplated<uint64_t, LAST, SKIP_NULLS>(type);
	case LogicalTypeId::HUGEINT:
		return GetFirstAggregateTemplated<hugeint_t, LAST, SKIP_NULLS>(type);
	case LogicalTypeId::FLOAT:
		return GetFirstAggregateTemplated<float, LAST, SKIP_NULLS>(type);
	case LogicalTypeId::DOUBLE:
		return GetFirstAggregateTemplated<double, LAST, SKIP_NULLS>(type);
	case LogicalTypeId::INTERVAL:
		return GetFirstAggregateTemplated<interval_t, LAST, SKIP_NULLS>(type);
	case LogicalTypeId::VARCHAR:
	case LogicalTypeId::BLOB:
		return AggregateFunction::UnaryAggregateDestructor<FirstState<string_t>, string_t, string_t,
		                                                   FirstFunctionString<LAST, SKIP_NULLS>>(type, type);
	case LogicalTypeId::DECIMAL: {
		type.Verify();
		AggregateFunction function = GetDecimalFirstFunction<LAST, SKIP_NULLS>(type);
		function.arguments[0] = type;
		function.return_type = type;
		// TODO set_key here?
		return function;
	}
	default: {
		using OP = FirstVectorFunction<LAST, SKIP_NULLS>;
		return AggregateFunction({type}, type, AggregateFunction::StateSize<FirstStateVector>,
		                         AggregateFunction::StateInitialize<FirstStateVector, OP>, OP::Update,
		                         AggregateFunction::StateCombine<FirstStateVector, OP>,
		                         AggregateFunction::StateFinalize<FirstStateVector, void, OP>, nullptr, OP::Bind,
		                         AggregateFunction::StateDestroy<FirstStateVector, OP>, nullptr, nullptr);
	}
	}
}

AggregateFunction FirstFun::GetFunction(const LogicalType &type) {
	auto fun = GetFirstFunction<false, false>(type);
	fun.name = "first";
	return fun;
}

template <bool LAST, bool SKIP_NULLS>
unique_ptr<FunctionData> BindDecimalFirst(ClientContext &context, AggregateFunction &function,
                                          vector<unique_ptr<Expression>> &arguments) {
	auto decimal_type = arguments[0]->return_type;
	function = GetFirstFunction<LAST, SKIP_NULLS>(decimal_type);
	function.name = "first";
	function.return_type = decimal_type;
	return nullptr;
}

template <bool LAST, bool SKIP_NULLS>
static AggregateFunction GetFirstOperator(const LogicalType &type) {
	if (type.id() == LogicalTypeId::DECIMAL) {
		throw InternalException("FIXME: this shouldn't happen...");
	}
	return GetFirstFunction<LAST, SKIP_NULLS>(type);
}

template <bool LAST, bool SKIP_NULLS>
unique_ptr<FunctionData> BindFirst(ClientContext &context, AggregateFunction &function,
                                   vector<unique_ptr<Expression>> &arguments) {
	auto input_type = arguments[0]->return_type;
	auto name = std::move(function.name);
	function = GetFirstOperator<LAST, SKIP_NULLS>(input_type);
	function.name = std::move(name);
	if (function.bind) {
		return function.bind(context, function, arguments);
	} else {
		return nullptr;
	}
}

template <bool LAST, bool SKIP_NULLS>
static void AddFirstOperator(AggregateFunctionSet &set) {
	set.AddFunction(AggregateFunction({LogicalTypeId::DECIMAL}, LogicalTypeId::DECIMAL, nullptr, nullptr, nullptr,
	                                  nullptr, nullptr, nullptr, BindDecimalFirst<LAST, SKIP_NULLS>));
	set.AddFunction(AggregateFunction({LogicalType::ANY}, LogicalType::ANY, nullptr, nullptr, nullptr, nullptr, nullptr,
	                                  nullptr, BindFirst<LAST, SKIP_NULLS>));
}

void FirstFun::RegisterFunction(BuiltinFunctions &set) {
	AggregateFunctionSet first("first");
	AggregateFunctionSet last("last");
	AggregateFunctionSet any_value("any_value");

	AddFirstOperator<false, false>(first);
	AddFirstOperator<true, false>(last);
	AddFirstOperator<false, true>(any_value);

	set.AddFunction(first);
	first.name = "arbitrary";
	set.AddFunction(first);

	set.AddFunction(last);

	set.AddFunction(any_value);
}

} // namespace duckdb






namespace duckdb {

void BuiltinFunctions::RegisterDistributiveAggregates() {
	Register<CountStarFun>();
	Register<CountFun>();
	Register<FirstFun>();
}

} // namespace duckdb










namespace duckdb {

struct SortedAggregateBindData : public FunctionData {
	SortedAggregateBindData(ClientContext &context, BoundAggregateExpression &expr)
	    : buffer_manager(BufferManager::GetBufferManager(context)), function(expr.function),
	      bind_info(std::move(expr.bind_info)), threshold(ClientConfig::GetConfig(context).ordered_aggregate_threshold),
	      external(ClientConfig::GetConfig(context).force_external) {
		auto &children = expr.children;
		arg_types.reserve(children.size());
		for (const auto &child : children) {
			arg_types.emplace_back(child->return_type);
		}
		auto &order_bys = *expr.order_bys;
		sort_types.reserve(order_bys.orders.size());
		for (auto &order : order_bys.orders) {
			orders.emplace_back(order.Copy());
			sort_types.emplace_back(order.expression->return_type);
		}
		sorted_on_args = (children.size() == order_bys.orders.size());
		for (size_t i = 0; sorted_on_args && i < children.size(); ++i) {
			sorted_on_args = children[i]->Equals(order_bys.orders[i].expression.get());
		}
	}

	SortedAggregateBindData(const SortedAggregateBindData &other)
	    : buffer_manager(other.buffer_manager), function(other.function), arg_types(other.arg_types),
	      sort_types(other.sort_types), sorted_on_args(other.sorted_on_args), threshold(other.threshold),
	      external(other.external) {
		if (other.bind_info) {
			bind_info = other.bind_info->Copy();
		}
		for (auto &order : other.orders) {
			orders.emplace_back(order.Copy());
		}
	}

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<SortedAggregateBindData>(*this);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = (const SortedAggregateBindData &)other_p;
		if (bind_info && other.bind_info) {
			if (!bind_info->Equals(*other.bind_info)) {
				return false;
			}
		} else if (bind_info || other.bind_info) {
			return false;
		}
		if (function != other.function) {
			return false;
		}
		if (orders.size() != other.orders.size()) {
			return false;
		}
		for (size_t i = 0; i < orders.size(); ++i) {
			if (!orders[i].Equals(other.orders[i])) {
				return false;
			}
		}
		return true;
	}

	BufferManager &buffer_manager;
	AggregateFunction function;
	vector<LogicalType> arg_types;
	unique_ptr<FunctionData> bind_info;

	vector<BoundOrderByNode> orders;
	vector<LogicalType> sort_types;
	bool sorted_on_args;

	//! The sort flush threshold
	const idx_t threshold;
	const bool external;
};

struct SortedAggregateState {
	//! Default buffer size, optimised for small group to avoid blowing out memory.
	static const idx_t BUFFER_CAPACITY = 16;

	SortedAggregateState() : count(0), nsel(0), offset(0) {
	}

	static inline void InitializeBuffer(DataChunk &chunk, const vector<LogicalType> &types) {
		if (!chunk.ColumnCount() && !types.empty()) {
			chunk.Initialize(Allocator::DefaultAllocator(), types, BUFFER_CAPACITY);
		}
	}

	//! Make sure the buffer is large enough for slicing
	static inline void ResetBuffer(DataChunk &chunk, const vector<LogicalType> &types) {
		chunk.Reset();
		chunk.Destroy();
		chunk.Initialize(Allocator::DefaultAllocator(), types);
	}

	void Flush(const SortedAggregateBindData &order_bind) {
		if (ordering) {
			return;
		}

		ordering = make_uniq<ColumnDataCollection>(order_bind.buffer_manager, order_bind.sort_types);
		InitializeBuffer(sort_buffer, order_bind.sort_types);
		ordering->Append(sort_buffer);
		ResetBuffer(sort_buffer, order_bind.sort_types);

		if (!order_bind.sorted_on_args) {
			arguments = make_uniq<ColumnDataCollection>(order_bind.buffer_manager, order_bind.arg_types);
			InitializeBuffer(arg_buffer, order_bind.arg_types);
			arguments->Append(arg_buffer);
			ResetBuffer(arg_buffer, order_bind.arg_types);
		}
	}

	void Update(const SortedAggregateBindData &order_bind, DataChunk &sort_chunk, DataChunk &arg_chunk) {
		count += sort_chunk.size();

		// Lazy instantiation of the buffer chunks
		InitializeBuffer(sort_buffer, order_bind.sort_types);
		if (!order_bind.sorted_on_args) {
			InitializeBuffer(arg_buffer, order_bind.arg_types);
		}

		if (sort_chunk.size() + sort_buffer.size() > STANDARD_VECTOR_SIZE) {
			Flush(order_bind);
		}
		if (arguments) {
			ordering->Append(sort_chunk);
			arguments->Append(arg_chunk);
		} else if (ordering) {
			ordering->Append(sort_chunk);
		} else if (order_bind.sorted_on_args) {
			sort_buffer.Append(sort_chunk, true);
		} else {
			sort_buffer.Append(sort_chunk, true);
			arg_buffer.Append(arg_chunk, true);
		}
	}

	void UpdateSlice(const SortedAggregateBindData &order_bind, DataChunk &sort_inputs, DataChunk &arg_inputs) {
		count += nsel;

		// Lazy instantiation of the buffer chunks
		InitializeBuffer(sort_buffer, order_bind.sort_types);
		if (!order_bind.sorted_on_args) {
			InitializeBuffer(arg_buffer, order_bind.arg_types);
		}

		if (nsel + sort_buffer.size() > STANDARD_VECTOR_SIZE) {
			Flush(order_bind);
		}
		if (arguments) {
			sort_buffer.Reset();
			sort_buffer.Slice(sort_inputs, sel, nsel);
			ordering->Append(sort_buffer);

			arg_buffer.Reset();
			arg_buffer.Slice(arg_inputs, sel, nsel);
			arguments->Append(arg_buffer);
		} else if (ordering) {
			sort_buffer.Reset();
			sort_buffer.Slice(sort_inputs, sel, nsel);
			ordering->Append(sort_buffer);
		} else if (order_bind.sorted_on_args) {
			sort_buffer.Append(sort_inputs, true, &sel, nsel);
		} else {
			sort_buffer.Append(sort_inputs, true, &sel, nsel);
			arg_buffer.Append(arg_inputs, true, &sel, nsel);
		}

		nsel = 0;
		offset = 0;
	}

	void Combine(SortedAggregateBindData &order_bind, SortedAggregateState &other) {
		if (other.arguments) {
			// Force CDC if the other has it
			Flush(order_bind);
			ordering->Combine(*other.ordering);
			arguments->Combine(*other.arguments);
			count += other.count;
		} else if (other.ordering) {
			// Force CDC if the other has it
			Flush(order_bind);
			ordering->Combine(*other.ordering);
			count += other.count;
		} else if (other.sort_buffer.size()) {
			Update(order_bind, other.sort_buffer, other.arg_buffer);
		}
	}

	void PrefixSortBuffer(DataChunk &prefixed) {
		for (column_t col_idx = 0; col_idx < sort_buffer.ColumnCount(); ++col_idx) {
			prefixed.data[col_idx + 1].Reference(sort_buffer.data[col_idx]);
		}
		prefixed.SetCardinality(sort_buffer);
	}

	void Finalize(const SortedAggregateBindData &order_bind, DataChunk &prefixed, LocalSortState &local_sort) {
		if (arguments) {
			ColumnDataScanState sort_state;
			ordering->InitializeScan(sort_state);
			ColumnDataScanState arg_state;
			arguments->InitializeScan(arg_state);
			for (sort_buffer.Reset(); ordering->Scan(sort_state, sort_buffer); sort_buffer.Reset()) {
				PrefixSortBuffer(prefixed);
				arg_buffer.Reset();
				arguments->Scan(arg_state, arg_buffer);
				local_sort.SinkChunk(prefixed, arg_buffer);
			}
			ordering->Reset();
			arguments->Reset();
		} else if (ordering) {
			ColumnDataScanState sort_state;
			ordering->InitializeScan(sort_state);
			for (sort_buffer.Reset(); ordering->Scan(sort_state, sort_buffer); sort_buffer.Reset()) {
				PrefixSortBuffer(prefixed);
				local_sort.SinkChunk(prefixed, sort_buffer);
			}
			ordering->Reset();
		} else if (order_bind.sorted_on_args) {
			PrefixSortBuffer(prefixed);
			local_sort.SinkChunk(prefixed, sort_buffer);
		} else {
			PrefixSortBuffer(prefixed);
			local_sort.SinkChunk(prefixed, arg_buffer);
		}
	}

	idx_t count;
	unique_ptr<ColumnDataCollection> arguments;
	unique_ptr<ColumnDataCollection> ordering;

	DataChunk sort_buffer;
	DataChunk arg_buffer;

	// Selection for scattering
	SelectionVector sel;
	idx_t nsel;
	idx_t offset;
};

struct SortedAggregateFunction {
	template <typename STATE>
	static void Initialize(STATE *state) {
		new (state) STATE();
	}

	template <typename STATE>
	static void Destroy(AggregateInputData &aggr_input_data, STATE *state) {
		state->~STATE();
	}

	static void ProjectInputs(Vector inputs[], const SortedAggregateBindData &order_bind, idx_t input_count,
	                          idx_t count, DataChunk &arg_chunk, DataChunk &sort_chunk) {
		idx_t col = 0;

		if (!order_bind.sorted_on_args) {
			arg_chunk.InitializeEmpty(order_bind.arg_types);
			for (auto &dst : arg_chunk.data) {
				dst.Reference(inputs[col++]);
			}
			arg_chunk.SetCardinality(count);
		}

		sort_chunk.InitializeEmpty(order_bind.sort_types);
		for (auto &dst : sort_chunk.data) {
			dst.Reference(inputs[col++]);
		}
		sort_chunk.SetCardinality(count);
	}

	static void SimpleUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count, data_ptr_t state,
	                         idx_t count) {
		const auto order_bind = aggr_input_data.bind_data->Cast<SortedAggregateBindData>();
		DataChunk arg_chunk;
		DataChunk sort_chunk;
		ProjectInputs(inputs, order_bind, input_count, count, arg_chunk, sort_chunk);

		const auto order_state = (SortedAggregateState *)state;
		order_state->Update(order_bind, sort_chunk, arg_chunk);
	}

	static void ScatterUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count, Vector &states,
	                          idx_t count) {
		if (!count) {
			return;
		}

		// Append the arguments to the two sub-collections
		const auto &order_bind = aggr_input_data.bind_data->Cast<SortedAggregateBindData>();
		DataChunk arg_inputs;
		DataChunk sort_inputs;
		ProjectInputs(inputs, order_bind, input_count, count, arg_inputs, sort_inputs);

		// We have to scatter the chunks one at a time
		// so build a selection vector for each one.
		UnifiedVectorFormat svdata;
		states.ToUnifiedFormat(count, svdata);

		// Size the selection vector for each state.
		auto sdata = (SortedAggregateState **)svdata.data;
		for (idx_t i = 0; i < count; ++i) {
			auto sidx = svdata.sel->get_index(i);
			auto order_state = sdata[sidx];
			order_state->nsel++;
		}

		// Build the selection vector for each state.
		vector<sel_t> sel_data(count);
		idx_t start = 0;
		for (idx_t i = 0; i < count; ++i) {
			auto sidx = svdata.sel->get_index(i);
			auto order_state = sdata[sidx];
			if (!order_state->offset) {
				//	First one
				order_state->offset = start;
				order_state->sel.Initialize(sel_data.data() + order_state->offset);
				start += order_state->nsel;
			}
			sel_data[order_state->offset++] = sidx;
		}

		// Append nonempty slices to the arguments
		for (idx_t i = 0; i < count; ++i) {
			auto sidx = svdata.sel->get_index(i);
			auto order_state = sdata[sidx];
			if (!order_state->nsel) {
				continue;
			}

			order_state->UpdateSlice(order_bind, sort_inputs, arg_inputs);
		}
	}

	template <class STATE, class OP>
	static void Combine(const STATE &source, STATE *target, AggregateInputData &aggr_input_data) {
		const auto order_bind = (SortedAggregateBindData *)aggr_input_data.bind_data;
		auto &other = const_cast<STATE &>(source);
		target->Combine(*order_bind, other);
	}

	static void Window(Vector inputs[], const ValidityMask &filter_mask, AggregateInputData &aggr_input_data,
	                   idx_t input_count, data_ptr_t state, const FrameBounds &frame, const FrameBounds &prev,
	                   Vector &result, idx_t rid, idx_t bias) {
		throw InternalException("Sorted aggregates should not be generated for window clauses");
	}

	static void Finalize(Vector &states, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
	                     const idx_t offset) {
		const auto &order_bind = aggr_input_data.bind_data->Cast<SortedAggregateBindData>();
		auto &buffer_manager = order_bind.buffer_manager;
		RowLayout payload_layout;
		payload_layout.Initialize(order_bind.arg_types);
		DataChunk chunk;
		chunk.Initialize(Allocator::DefaultAllocator(), order_bind.arg_types);
		DataChunk sliced;
		sliced.Initialize(Allocator::DefaultAllocator(), order_bind.arg_types);

		//	 Reusable inner state
		vector<data_t> agg_state(order_bind.function.state_size());
		Vector agg_state_vec(Value::POINTER((idx_t)agg_state.data()));

		// State variables
		auto bind_info = order_bind.bind_info.get();
		AggregateInputData aggr_bind_info(bind_info, Allocator::DefaultAllocator());

		// Inner aggregate APIs
		auto initialize = order_bind.function.initialize;
		auto destructor = order_bind.function.destructor;
		auto simple_update = order_bind.function.simple_update;
		auto update = order_bind.function.update;
		auto finalize = order_bind.function.finalize;

		auto sdata = FlatVector::GetData<SortedAggregateState *>(states);

		vector<idx_t> state_unprocessed(count, 0);
		for (idx_t i = 0; i < count; ++i) {
			state_unprocessed[i] = sdata[i]->count;
		}

		// Sort the input payloads on (state_idx ASC, orders)
		vector<BoundOrderByNode> orders;
		orders.emplace_back(BoundOrderByNode(OrderType::ASCENDING, OrderByNullType::NULLS_FIRST,
		                                     make_uniq<BoundConstantExpression>(Value::USMALLINT(0))));
		for (const auto &order : order_bind.orders) {
			orders.emplace_back(order.Copy());
		}

		auto global_sort = make_uniq<GlobalSortState>(buffer_manager, orders, payload_layout);
		global_sort->external = order_bind.external;
		auto local_sort = make_uniq<LocalSortState>();
		local_sort->Initialize(*global_sort, global_sort->buffer_manager);

		DataChunk prefixed;
		prefixed.Initialize(Allocator::DefaultAllocator(), global_sort->sort_layout.logical_types);

		//	Go through the states accumulating values to sort until we hit the sort threshold
		idx_t unsorted_count = 0;
		idx_t sorted = 0;
		for (idx_t finalized = 0; finalized < count;) {
			if (unsorted_count < order_bind.threshold) {
				auto state = sdata[finalized];
				prefixed.Reset();
				prefixed.data[0].Reference(Value::USMALLINT(finalized));
				state->Finalize(order_bind, prefixed, *local_sort);
				unsorted_count += state_unprocessed[finalized];

				// Go to the next aggregate unless this is the last one
				if (++finalized < count) {
					continue;
				}
			}

			//	If they were all empty (filtering) flush them
			//	(This can only happen on the last range)
			if (!unsorted_count) {
				break;
			}

			//	Sort all the data
			global_sort->AddLocalState(*local_sort);
			global_sort->PrepareMergePhase();
			while (global_sort->sorted_blocks.size() > 1) {
				global_sort->InitializeMergeRound();
				MergeSorter merge_sorter(*global_sort, global_sort->buffer_manager);
				merge_sorter.PerformInMergeRound();
				global_sort->CompleteMergeRound(false);
			}

			auto scanner = make_uniq<PayloadScanner>(*global_sort);
			initialize(agg_state.data());
			while (scanner->Remaining()) {
				chunk.Reset();
				scanner->Scan(chunk);
				idx_t consumed = 0;

				// Distribute the scanned chunk to the aggregates
				while (consumed < chunk.size()) {
					//	Find the next aggregate that needs data
					for (; !state_unprocessed[sorted]; ++sorted) {
						// Finalize a single value at the next offset
						agg_state_vec.SetVectorType(states.GetVectorType());
						finalize(agg_state_vec, aggr_bind_info, result, 1, sorted + offset);
						if (destructor) {
							destructor(agg_state_vec, aggr_bind_info, 1);
						}

						initialize(agg_state.data());
					}
					const auto input_count = MinValue(state_unprocessed[sorted], chunk.size() - consumed);
					for (column_t col_idx = 0; col_idx < chunk.ColumnCount(); ++col_idx) {
						sliced.data[col_idx].Slice(chunk.data[col_idx], consumed, consumed + input_count);
					}
					sliced.SetCardinality(input_count);

					// These are all simple updates, so use it if available
					if (simple_update) {
						simple_update(sliced.data.data(), aggr_bind_info, 1, agg_state.data(), sliced.size());
					} else {
						// We are only updating a constant state
						agg_state_vec.SetVectorType(VectorType::CONSTANT_VECTOR);
						update(sliced.data.data(), aggr_bind_info, 1, agg_state_vec, sliced.size());
					}

					consumed += input_count;
					state_unprocessed[sorted] -= input_count;
				}
			}

			//	Finalize the last state for this sort
			agg_state_vec.SetVectorType(states.GetVectorType());
			finalize(agg_state_vec, aggr_bind_info, result, 1, sorted + offset);
			if (destructor) {
				destructor(agg_state_vec, aggr_bind_info, 1);
			}
			++sorted;

			//	Stop if we are done
			if (finalized >= count) {
				break;
			}

			//	Create a new sort
			scanner.reset();
			global_sort = make_uniq<GlobalSortState>(buffer_manager, orders, payload_layout);
			global_sort->external = order_bind.external;
			local_sort = make_uniq<LocalSortState>();
			local_sort->Initialize(*global_sort, global_sort->buffer_manager);
			unsorted_count = 0;
		}

		for (; sorted < count; ++sorted) {
			initialize(agg_state.data());

			// Finalize a single value at the next offset
			agg_state_vec.SetVectorType(states.GetVectorType());
			finalize(agg_state_vec, aggr_bind_info, result, 1, sorted + offset);

			if (destructor) {
				destructor(agg_state_vec, aggr_bind_info, 1);
			}
		}

		result.Verify(count);
	}

	static void Serialize(FieldWriter &writer, const FunctionData *bind_data, const AggregateFunction &function) {
		throw NotImplementedException("FIXME: serialize sorted aggregate not supported");
	}
	static unique_ptr<FunctionData> Deserialize(ClientContext &context, FieldReader &reader,
	                                            AggregateFunction &function) {
		throw NotImplementedException("FIXME: deserialize sorted aggregate not supported");
	}
};

void FunctionBinder::BindSortedAggregate(ClientContext &context, BoundAggregateExpression &expr,
                                         const vector<unique_ptr<Expression>> &groups) {
	if (!expr.order_bys || expr.order_bys->orders.empty() || expr.children.empty()) {
		// not a sorted aggregate: return
		return;
	}
	if (context.config.enable_optimizer) {
		// for each ORDER BY - check if it is actually necessary
		// expressions that are in the groups do not need to be ORDERED BY
		// `ORDER BY` on a group has no effect, because for each aggregate, the group is unique
		// similarly, we only need to ORDER BY each aggregate once
		expression_set_t seen_expressions;
		for (auto &target : groups) {
			seen_expressions.insert(*target);
		}
		vector<BoundOrderByNode> new_order_nodes;
		for (auto &order_node : expr.order_bys->orders) {
			if (seen_expressions.find(*order_node.expression) != seen_expressions.end()) {
				// we do not need to order by this node
				continue;
			}
			seen_expressions.insert(*order_node.expression);
			new_order_nodes.push_back(std::move(order_node));
		}
		if (new_order_nodes.empty()) {
			expr.order_bys.reset();
			return;
		}
		expr.order_bys->orders = std::move(new_order_nodes);
	}
	auto &bound_function = expr.function;
	auto &children = expr.children;
	auto &order_bys = *expr.order_bys;
	auto sorted_bind = make_uniq<SortedAggregateBindData>(context, expr);

	if (!sorted_bind->sorted_on_args) {
		// The arguments are the children plus the sort columns.
		for (auto &order : order_bys.orders) {
			children.emplace_back(std::move(order.expression));
		}
	}

	vector<LogicalType> arguments;
	arguments.reserve(children.size());
	for (const auto &child : children) {
		arguments.emplace_back(child->return_type);
	}

	// Replace the aggregate with the wrapper
	AggregateFunction ordered_aggregate(
	    bound_function.name, arguments, bound_function.return_type, AggregateFunction::StateSize<SortedAggregateState>,
	    AggregateFunction::StateInitialize<SortedAggregateState, SortedAggregateFunction>,
	    SortedAggregateFunction::ScatterUpdate,
	    AggregateFunction::StateCombine<SortedAggregateState, SortedAggregateFunction>,
	    SortedAggregateFunction::Finalize, bound_function.null_handling, SortedAggregateFunction::SimpleUpdate, nullptr,
	    AggregateFunction::StateDestroy<SortedAggregateState, SortedAggregateFunction>, nullptr,
	    SortedAggregateFunction::Window, SortedAggregateFunction::Serialize, SortedAggregateFunction::Deserialize);

	expr.function = std::move(ordered_aggregate);
	expr.bind_info = std::move(sorted_bind);
	expr.order_bys.reset();
}

} // namespace duckdb










namespace duckdb {

BuiltinFunctions::BuiltinFunctions(CatalogTransaction transaction, Catalog &catalog)
    : transaction(transaction), catalog(catalog) {
}

BuiltinFunctions::~BuiltinFunctions() {
}

void BuiltinFunctions::AddCollation(string name, ScalarFunction function, bool combinable,
                                    bool not_required_for_equality) {
	CreateCollationInfo info(std::move(name), std::move(function), combinable, not_required_for_equality);
	info.internal = true;
	catalog.CreateCollation(transaction, info);
}

void BuiltinFunctions::AddFunction(AggregateFunctionSet set) {
	CreateAggregateFunctionInfo info(std::move(set));
	info.internal = true;
	catalog.CreateFunction(transaction, info);
}

void BuiltinFunctions::AddFunction(AggregateFunction function) {
	CreateAggregateFunctionInfo info(std::move(function));
	info.internal = true;
	catalog.CreateFunction(transaction, info);
}

void BuiltinFunctions::AddFunction(PragmaFunction function) {
	CreatePragmaFunctionInfo info(std::move(function));
	info.internal = true;
	catalog.CreatePragmaFunction(transaction, info);
}

void BuiltinFunctions::AddFunction(const string &name, PragmaFunctionSet functions) {
	CreatePragmaFunctionInfo info(name, std::move(functions));
	info.internal = true;
	catalog.CreatePragmaFunction(transaction, info);
}

void BuiltinFunctions::AddFunction(ScalarFunction function) {
	CreateScalarFunctionInfo info(std::move(function));
	info.internal = true;
	catalog.CreateFunction(transaction, info);
}

void BuiltinFunctions::AddFunction(const vector<string> &names, ScalarFunction function) { // NOLINT: false positive
	for (auto &name : names) {
		function.name = name;
		AddFunction(function);
	}
}

void BuiltinFunctions::AddFunction(ScalarFunctionSet set) {
	CreateScalarFunctionInfo info(std::move(set));
	info.internal = true;
	catalog.CreateFunction(transaction, info);
}

void BuiltinFunctions::AddFunction(TableFunction function) {
	CreateTableFunctionInfo info(std::move(function));
	info.internal = true;
	catalog.CreateTableFunction(transaction, info);
}

void BuiltinFunctions::AddFunction(TableFunctionSet set) {
	CreateTableFunctionInfo info(std::move(set));
	info.internal = true;
	catalog.CreateTableFunction(transaction, info);
}

void BuiltinFunctions::AddFunction(CopyFunction function) {
	CreateCopyFunctionInfo info(std::move(function));
	info.internal = true;
	catalog.CreateCopyFunction(transaction, info);
}

} // namespace duckdb



namespace duckdb {

BoundCastInfo DefaultCasts::BitCastSwitch(BindCastInput &input, const LogicalType &source, const LogicalType &target) {
	// now switch on the result type
	switch (target.id()) {
	case LogicalTypeId::VARCHAR:
		// bit to varchar
		return BoundCastInfo(&VectorCastHelpers::StringCast<string_t, duckdb::CastFromBit>);
	default:
		return DefaultCasts::TryVectorNullCast;
	}
}

} // namespace duckdb



namespace duckdb {

BoundCastInfo DefaultCasts::BlobCastSwitch(BindCastInput &input, const LogicalType &source, const LogicalType &target) {
	// now switch on the result type
	switch (target.id()) {
	case LogicalTypeId::VARCHAR:
		// blob to varchar
		return BoundCastInfo(&VectorCastHelpers::StringCast<string_t, duckdb::CastFromBlob>);
	case LogicalTypeId::AGGREGATE_STATE:
		return DefaultCasts::ReinterpretCast;
	default:
		return DefaultCasts::TryVectorNullCast;
	}
}

} // namespace duckdb







namespace duckdb {

BindCastInput::BindCastInput(CastFunctionSet &function_set, optional_ptr<BindCastInfo> info,
                             optional_ptr<ClientContext> context)
    : function_set(function_set), info(info), context(context) {
}

BoundCastInfo BindCastInput::GetCastFunction(const LogicalType &source, const LogicalType &target) {
	GetCastFunctionInput input(context);
	return function_set.GetCastFunction(source, target, input);
}

BindCastFunction::BindCastFunction(bind_cast_function_t function_p, unique_ptr<BindCastInfo> info_p)
    : function(function_p), info(std::move(info_p)) {
}

CastFunctionSet::CastFunctionSet() : map_info(nullptr) {
	bind_functions.emplace_back(DefaultCasts::GetDefaultCastFunction);
}

CastFunctionSet &CastFunctionSet::Get(ClientContext &context) {
	return DBConfig::GetConfig(context).GetCastFunctions();
}

CastFunctionSet &CastFunctionSet::Get(DatabaseInstance &db) {
	return DBConfig::GetConfig(db).GetCastFunctions();
}

BoundCastInfo CastFunctionSet::GetCastFunction(const LogicalType &source, const LogicalType &target,
                                               GetCastFunctionInput &get_input) {
	if (source == target) {
		return DefaultCasts::NopCast;
	}
	// the first function is the default
	// we iterate the set of bind functions backwards
	for (idx_t i = bind_functions.size(); i > 0; i--) {
		auto &bind_function = bind_functions[i - 1];
		BindCastInput input(*this, bind_function.info.get(), get_input.context);
		auto result = bind_function.function(input, source, target);
		if (result.function) {
			// found a cast function! return it
			return result;
		}
	}
	// no cast found: return the default null cast
	return DefaultCasts::TryVectorNullCast;
}

struct MapCastNode {
	MapCastNode(BoundCastInfo info, int64_t implicit_cast_cost)
	    : cast_info(std::move(info)), bind_function(nullptr), implicit_cast_cost(implicit_cast_cost) {
	}
	MapCastNode(bind_cast_function_t func, int64_t implicit_cast_cost)
	    : cast_info(nullptr), bind_function(func), implicit_cast_cost(implicit_cast_cost) {
	}

	BoundCastInfo cast_info;
	bind_cast_function_t bind_function;
	int64_t implicit_cast_cost;
};

template <class MAP_VALUE_TYPE>
static auto RelaxedTypeMatch(type_map_t<MAP_VALUE_TYPE> &map, const LogicalType &type) -> decltype(map.find(type)) {
	D_ASSERT(map.find(type) == map.end()); // we shouldn't be here
	switch (type.id()) {
	case LogicalTypeId::LIST:
		return map.find(LogicalType::LIST(LogicalType::ANY));
	case LogicalTypeId::STRUCT:
		return map.find(LogicalType::STRUCT({{"any", LogicalType::ANY}}));
	case LogicalTypeId::MAP:
		for (auto it = map.begin(); it != map.end(); it++) {
			const auto &entry_type = it->first;
			if (entry_type.id() != LogicalTypeId::MAP) {
				continue;
			}
			auto &entry_key_type = MapType::KeyType(entry_type);
			auto &entry_val_type = MapType::ValueType(entry_type);
			if ((entry_key_type == LogicalType::ANY || entry_key_type == MapType::KeyType(type)) &&
			    (entry_val_type == LogicalType::ANY || entry_val_type == MapType::ValueType(type))) {
				return it;
			}
		}
		return map.end();
	case LogicalTypeId::UNION:
		return map.find(LogicalType::UNION({{"any", LogicalType::ANY}}));
	default:
		return map.find(LogicalType::ANY);
	}
}

struct MapCastInfo : public BindCastInfo {
public:
	const optional_ptr<MapCastNode> GetEntry(const LogicalType &source, const LogicalType &target) {
		auto source_type_id_entry = casts.find(source.id());
		if (source_type_id_entry == casts.end()) {
			source_type_id_entry = casts.find(LogicalTypeId::ANY);
			if (source_type_id_entry == casts.end()) {
				return nullptr;
			}
		}

		auto &source_type_entries = source_type_id_entry->second;
		auto source_type_entry = source_type_entries.find(source);
		if (source_type_entry == source_type_entries.end()) {
			source_type_entry = RelaxedTypeMatch(source_type_entries, source);
			if (source_type_entry == source_type_entries.end()) {
				return nullptr;
			}
		}

		auto &target_type_id_entries = source_type_entry->second;
		auto target_type_id_entry = target_type_id_entries.find(target.id());
		if (target_type_id_entry == target_type_id_entries.end()) {
			target_type_id_entry = target_type_id_entries.find(LogicalTypeId::ANY);
			if (target_type_id_entry == target_type_id_entries.end()) {
				return nullptr;
			}
		}

		auto &target_type_entries = target_type_id_entry->second;
		auto target_type_entry = target_type_entries.find(target);
		if (target_type_entry == target_type_entries.end()) {
			target_type_entry = RelaxedTypeMatch(target_type_entries, target);
			if (target_type_entry == target_type_entries.end()) {
				return nullptr;
			}
		}

		return &target_type_entry->second;
	}

	void AddEntry(const LogicalType &source, const LogicalType &target, MapCastNode node) {
		casts[source.id()][source][target.id()].insert(make_pair(target, std::move(node)));
	}

private:
	type_id_map_t<type_map_t<type_id_map_t<type_map_t<MapCastNode>>>> casts;
};

int64_t CastFunctionSet::ImplicitCastCost(const LogicalType &source, const LogicalType &target) {
	// check if a cast has been registered
	if (map_info) {
		auto entry = map_info->GetEntry(source, target);
		if (entry) {
			return entry->implicit_cast_cost;
		}
	}
	// if not, fallback to the default implicit cast rules
	return CastRules::ImplicitCast(source, target);
}

BoundCastInfo MapCastFunction(BindCastInput &input, const LogicalType &source, const LogicalType &target) {
	D_ASSERT(input.info);
	auto &map_info = (MapCastInfo &)*input.info;
	auto entry = map_info.GetEntry(source, target);
	if (entry) {
		if (entry->bind_function) {
			return entry->bind_function(input, source, target);
		}
		return entry->cast_info.Copy();
	}
	return nullptr;
}

void CastFunctionSet::RegisterCastFunction(const LogicalType &source, const LogicalType &target, BoundCastInfo function,
                                           int64_t implicit_cast_cost) {
	RegisterCastFunction(source, target, MapCastNode(std::move(function), implicit_cast_cost));
}

void CastFunctionSet::RegisterCastFunction(const LogicalType &source, const LogicalType &target,
                                           bind_cast_function_t bind_function, int64_t implicit_cast_cost) {
	RegisterCastFunction(source, target, MapCastNode(bind_function, implicit_cast_cost));
}

void CastFunctionSet::RegisterCastFunction(const LogicalType &source, const LogicalType &target, MapCastNode node) {
	if (!map_info) {
		// create the cast map and the cast map function
		auto info = make_uniq<MapCastInfo>();
		map_info = info.get();
		bind_functions.emplace_back(MapCastFunction, std::move(info));
	}
	map_info->AddEntry(source, target, std::move(node));
}

} // namespace duckdb








namespace duckdb {

template <class T>
static bool FromDecimalCast(Vector &source, Vector &result, idx_t count, CastParameters &parameters) {
	auto &source_type = source.GetType();
	auto width = DecimalType::GetWidth(source_type);
	auto scale = DecimalType::GetScale(source_type);
	switch (source_type.InternalType()) {
	case PhysicalType::INT16:
		return VectorCastHelpers::TemplatedDecimalCast<int16_t, T, TryCastFromDecimal>(
		    source, result, count, parameters.error_message, width, scale);
	case PhysicalType::INT32:
		return VectorCastHelpers::TemplatedDecimalCast<int32_t, T, TryCastFromDecimal>(
		    source, result, count, parameters.error_message, width, scale);
	case PhysicalType::INT64:
		return VectorCastHelpers::TemplatedDecimalCast<int64_t, T, TryCastFromDecimal>(
		    source, result, count, parameters.error_message, width, scale);
	case PhysicalType::INT128:
		return VectorCastHelpers::TemplatedDecimalCast<hugeint_t, T, TryCastFromDecimal>(
		    source, result, count, parameters.error_message, width, scale);
	default:
		throw InternalException("Unimplemented internal type for decimal");
	}
}

template <class LIMIT_TYPE, class FACTOR_TYPE = LIMIT_TYPE>
struct DecimalScaleInput {
	DecimalScaleInput(Vector &result_p, FACTOR_TYPE factor_p) : result(result_p), factor(factor_p) {
	}
	DecimalScaleInput(Vector &result_p, LIMIT_TYPE limit_p, FACTOR_TYPE factor_p, string *error_message_p,
	                  uint8_t source_width_p, uint8_t source_scale_p)
	    : result(result_p), limit(limit_p), factor(factor_p), error_message(error_message_p),
	      source_width(source_width_p), source_scale(source_scale_p) {
	}

	Vector &result;
	LIMIT_TYPE limit;
	FACTOR_TYPE factor;
	bool all_converted = true;
	string *error_message;
	uint8_t source_width;
	uint8_t source_scale;
};

struct DecimalScaleUpOperator {
	template <class INPUT_TYPE, class RESULT_TYPE>
	static RESULT_TYPE Operation(INPUT_TYPE input, ValidityMask &mask, idx_t idx, void *dataptr) {
		auto data = (DecimalScaleInput<INPUT_TYPE, RESULT_TYPE> *)dataptr;
		return Cast::Operation<INPUT_TYPE, RESULT_TYPE>(input) * data->factor;
	}
};

struct DecimalScaleUpCheckOperator {
	template <class INPUT_TYPE, class RESULT_TYPE>
	static RESULT_TYPE Operation(INPUT_TYPE input, ValidityMask &mask, idx_t idx, void *dataptr) {
		auto data = (DecimalScaleInput<INPUT_TYPE, RESULT_TYPE> *)dataptr;
		if (input >= data->limit || input <= -data->limit) {
			auto error = StringUtil::Format("Casting value \"%s\" to type %s failed: value is out of range!",
			                                Decimal::ToString(input, data->source_width, data->source_scale),
			                                data->result.GetType().ToString());
			return HandleVectorCastError::Operation<RESULT_TYPE>(std::move(error), mask, idx, data->error_message,
			                                                     data->all_converted);
		}
		return Cast::Operation<INPUT_TYPE, RESULT_TYPE>(input) * data->factor;
	}
};

template <class SOURCE, class DEST, class POWERS_SOURCE, class POWERS_DEST>
bool TemplatedDecimalScaleUp(Vector &source, Vector &result, idx_t count, string *error_message) {
	auto source_scale = DecimalType::GetScale(source.GetType());
	auto source_width = DecimalType::GetWidth(source.GetType());
	auto result_scale = DecimalType::GetScale(result.GetType());
	auto result_width = DecimalType::GetWidth(result.GetType());
	D_ASSERT(result_scale >= source_scale);
	idx_t scale_difference = result_scale - source_scale;
	DEST multiply_factor = POWERS_DEST::POWERS_OF_TEN[scale_difference];
	idx_t target_width = result_width - scale_difference;
	if (source_width < target_width) {
		DecimalScaleInput<SOURCE, DEST> input(result, multiply_factor);
		// type will always fit: no need to check limit
		UnaryExecutor::GenericExecute<SOURCE, DEST, DecimalScaleUpOperator>(source, result, count, &input);
		return true;
	} else {
		// type might not fit: check limit
		auto limit = POWERS_SOURCE::POWERS_OF_TEN[target_width];
		DecimalScaleInput<SOURCE, DEST> input(result, limit, multiply_factor, error_message, source_width,
		                                      source_scale);
		UnaryExecutor::GenericExecute<SOURCE, DEST, DecimalScaleUpCheckOperator>(source, result, count, &input,
		                                                                         error_message);
		return input.all_converted;
	}
}

struct DecimalScaleDownOperator {
	template <class INPUT_TYPE, class RESULT_TYPE>
	static RESULT_TYPE Operation(INPUT_TYPE input, ValidityMask &mask, idx_t idx, void *dataptr) {
		auto data = (DecimalScaleInput<INPUT_TYPE> *)dataptr;
		return Cast::Operation<INPUT_TYPE, RESULT_TYPE>(input / data->factor);
	}
};

struct DecimalScaleDownCheckOperator {
	template <class INPUT_TYPE, class RESULT_TYPE>
	static RESULT_TYPE Operation(INPUT_TYPE input, ValidityMask &mask, idx_t idx, void *dataptr) {
		auto data = (DecimalScaleInput<INPUT_TYPE> *)dataptr;
		if (input >= data->limit || input <= -data->limit) {
			auto error = StringUtil::Format("Casting value \"%s\" to type %s failed: value is out of range!",
			                                Decimal::ToString(input, data->source_width, data->source_scale),
			                                data->result.GetType().ToString());
			return HandleVectorCastError::Operation<RESULT_TYPE>(std::move(error), mask, idx, data->error_message,
			                                                     data->all_converted);
		}
		return Cast::Operation<INPUT_TYPE, RESULT_TYPE>(input / data->factor);
	}
};

template <class SOURCE, class DEST, class POWERS_SOURCE>
bool TemplatedDecimalScaleDown(Vector &source, Vector &result, idx_t count, string *error_message) {
	auto source_scale = DecimalType::GetScale(source.GetType());
	auto source_width = DecimalType::GetWidth(source.GetType());
	auto result_scale = DecimalType::GetScale(result.GetType());
	auto result_width = DecimalType::GetWidth(result.GetType());
	D_ASSERT(result_scale < source_scale);
	idx_t scale_difference = source_scale - result_scale;
	idx_t target_width = result_width + scale_difference;
	SOURCE divide_factor = POWERS_SOURCE::POWERS_OF_TEN[scale_difference];
	if (source_width < target_width) {
		DecimalScaleInput<SOURCE> input(result, divide_factor);
		// type will always fit: no need to check limit
		UnaryExecutor::GenericExecute<SOURCE, DEST, DecimalScaleDownOperator>(source, result, count, &input);
		return true;
	} else {
		// type might not fit: check limit
		auto limit = POWERS_SOURCE::POWERS_OF_TEN[target_width];
		DecimalScaleInput<SOURCE> input(result, limit, divide_factor, error_message, source_width, source_scale);
		UnaryExecutor::GenericExecute<SOURCE, DEST, DecimalScaleDownCheckOperator>(source, result, count, &input,
		                                                                           error_message);
		return input.all_converted;
	}
}

template <class SOURCE, class POWERS_SOURCE>
static bool DecimalDecimalCastSwitch(Vector &source, Vector &result, idx_t count, CastParameters &parameters) {
	auto source_scale = DecimalType::GetScale(source.GetType());
	auto result_scale = DecimalType::GetScale(result.GetType());
	source.GetType().Verify();
	result.GetType().Verify();

	// we need to either multiply or divide by the difference in scales
	if (result_scale >= source_scale) {
		// multiply
		switch (result.GetType().InternalType()) {
		case PhysicalType::INT16:
			return TemplatedDecimalScaleUp<SOURCE, int16_t, POWERS_SOURCE, NumericHelper>(source, result, count,
			                                                                              parameters.error_message);
		case PhysicalType::INT32:
			return TemplatedDecimalScaleUp<SOURCE, int32_t, POWERS_SOURCE, NumericHelper>(source, result, count,
			                                                                              parameters.error_message);
		case PhysicalType::INT64:
			return TemplatedDecimalScaleUp<SOURCE, int64_t, POWERS_SOURCE, NumericHelper>(source, result, count,
			                                                                              parameters.error_message);
		case PhysicalType::INT128:
			return TemplatedDecimalScaleUp<SOURCE, hugeint_t, POWERS_SOURCE, Hugeint>(source, result, count,
			                                                                          parameters.error_message);
		default:
			throw NotImplementedException("Unimplemented internal type for decimal");
		}
	} else {
		// divide
		switch (result.GetType().InternalType()) {
		case PhysicalType::INT16:
			return TemplatedDecimalScaleDown<SOURCE, int16_t, POWERS_SOURCE>(source, result, count,
			                                                                 parameters.error_message);
		case PhysicalType::INT32:
			return TemplatedDecimalScaleDown<SOURCE, int32_t, POWERS_SOURCE>(source, result, count,
			                                                                 parameters.error_message);
		case PhysicalType::INT64:
			return TemplatedDecimalScaleDown<SOURCE, int64_t, POWERS_SOURCE>(source, result, count,
			                                                                 parameters.error_message);
		case PhysicalType::INT128:
			return TemplatedDecimalScaleDown<SOURCE, hugeint_t, POWERS_SOURCE>(source, result, count,
			                                                                   parameters.error_message);
		default:
			throw NotImplementedException("Unimplemented internal type for decimal");
		}
	}
}

struct DecimalCastInput {
	DecimalCastInput(Vector &result_p, uint8_t width_p, uint8_t scale_p)
	    : result(result_p), width(width_p), scale(scale_p) {
	}

	Vector &result;
	uint8_t width;
	uint8_t scale;
};

struct StringCastFromDecimalOperator {
	template <class INPUT_TYPE, class RESULT_TYPE>
	static RESULT_TYPE Operation(INPUT_TYPE input, ValidityMask &mask, idx_t idx, void *dataptr) {
		auto data = (DecimalCastInput *)dataptr;
		return StringCastFromDecimal::Operation<INPUT_TYPE>(input, data->width, data->scale, data->result);
	}
};

template <class SRC>
static bool DecimalToStringCast(Vector &source, Vector &result, idx_t count, CastParameters &parameters) {
	auto &source_type = source.GetType();
	auto width = DecimalType::GetWidth(source_type);
	auto scale = DecimalType::GetScale(source_type);
	DecimalCastInput input(result, width, scale);

	UnaryExecutor::GenericExecute<SRC, string_t, StringCastFromDecimalOperator>(source, result, count, (void *)&input);
	return true;
}

BoundCastInfo DefaultCasts::DecimalCastSwitch(BindCastInput &input, const LogicalType &source,
                                              const LogicalType &target) {
	// now switch on the result type
	switch (target.id()) {
	case LogicalTypeId::BOOLEAN:
		return FromDecimalCast<bool>;
	case LogicalTypeId::TINYINT:
		return FromDecimalCast<int8_t>;
	case LogicalTypeId::SMALLINT:
		return FromDecimalCast<int16_t>;
	case LogicalTypeId::INTEGER:
		return FromDecimalCast<int32_t>;
	case LogicalTypeId::BIGINT:
		return FromDecimalCast<int64_t>;
	case LogicalTypeId::UTINYINT:
		return FromDecimalCast<uint8_t>;
	case LogicalTypeId::USMALLINT:
		return FromDecimalCast<uint16_t>;
	case LogicalTypeId::UINTEGER:
		return FromDecimalCast<uint32_t>;
	case LogicalTypeId::UBIGINT:
		return FromDecimalCast<uint64_t>;
	case LogicalTypeId::HUGEINT:
		return FromDecimalCast<hugeint_t>;
	case LogicalTypeId::DECIMAL: {
		// decimal to decimal cast
		// first we need to figure out the source and target internal types
		switch (source.InternalType()) {
		case PhysicalType::INT16:
			return DecimalDecimalCastSwitch<int16_t, NumericHelper>;
		case PhysicalType::INT32:
			return DecimalDecimalCastSwitch<int32_t, NumericHelper>;
		case PhysicalType::INT64:
			return DecimalDecimalCastSwitch<int64_t, NumericHelper>;
		case PhysicalType::INT128:
			return DecimalDecimalCastSwitch<hugeint_t, Hugeint>;
		default:
			throw NotImplementedException("Unimplemented internal type for decimal in decimal_decimal cast");
		}
	}
	case LogicalTypeId::FLOAT:
		return FromDecimalCast<float>;
	case LogicalTypeId::DOUBLE:
		return FromDecimalCast<double>;
	case LogicalTypeId::VARCHAR: {
		switch (source.InternalType()) {
		case PhysicalType::INT16:
			return DecimalToStringCast<int16_t>;
		case PhysicalType::INT32:
			return DecimalToStringCast<int32_t>;
		case PhysicalType::INT64:
			return DecimalToStringCast<int64_t>;
		case PhysicalType::INT128:
			return DecimalToStringCast<hugeint_t>;
		default:
			throw InternalException("Unimplemented internal decimal type");
		}
	}
	default:
		return DefaultCasts::TryVectorNullCast;
	}
}

} // namespace duckdb












namespace duckdb {

BindCastInfo::~BindCastInfo() {
}

BoundCastData::~BoundCastData() {
}

BoundCastInfo::BoundCastInfo(cast_function_t function_p, unique_ptr<BoundCastData> cast_data_p,
                             init_cast_local_state_t init_local_state_p)
    : function(function_p), init_local_state(init_local_state_p), cast_data(std::move(cast_data_p)) {
}

BoundCastInfo BoundCastInfo::Copy() const {
	return BoundCastInfo(function, cast_data ? cast_data->Copy() : nullptr, init_local_state);
}

bool DefaultCasts::NopCast(Vector &source, Vector &result, idx_t count, CastParameters &parameters) {
	result.Reference(source);
	return true;
}

static string UnimplementedCastMessage(const LogicalType &source_type, const LogicalType &target_type) {
	return StringUtil::Format("Unimplemented type for cast (%s -> %s)", source_type.ToString(), target_type.ToString());
}

// NULL cast only works if all values in source are NULL, otherwise an unimplemented cast exception is thrown
bool DefaultCasts::TryVectorNullCast(Vector &source, Vector &result, idx_t count, CastParameters &parameters) {
	bool success = true;
	if (VectorOperations::HasNotNull(source, count)) {
		HandleCastError::AssignError(UnimplementedCastMessage(source.GetType(), result.GetType()),
		                             parameters.error_message);
		success = false;
	}
	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	ConstantVector::SetNull(result, true);
	return success;
}

bool DefaultCasts::ReinterpretCast(Vector &source, Vector &result, idx_t count, CastParameters &parameters) {
	result.Reinterpret(source);
	return true;
}

static bool AggregateStateToBlobCast(Vector &source, Vector &result, idx_t count, CastParameters &parameters) {
	if (result.GetType().id() != LogicalTypeId::BLOB) {
		throw TypeMismatchException(source.GetType(), result.GetType(),
		                            "Cannot cast AGGREGATE_STATE to anything but BLOB");
	}
	result.Reinterpret(source);
	return true;
}

static bool NullTypeCast(Vector &source, Vector &result, idx_t count, CastParameters &parameters) {
	// cast a NULL to another type, just copy the properties and change the type
	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	ConstantVector::SetNull(result, true);
	return true;
}

BoundCastInfo DefaultCasts::GetDefaultCastFunction(BindCastInput &input, const LogicalType &source,
                                                   const LogicalType &target) {
	D_ASSERT(source != target);

	// first check if were casting to a union
	if (source.id() != LogicalTypeId::UNION && source.id() != LogicalTypeId::SQLNULL &&
	    target.id() == LogicalTypeId::UNION) {
		return ImplicitToUnionCast(input, source, target);
	}

	// else, switch on source type
	switch (source.id()) {
	case LogicalTypeId::BOOLEAN:
	case LogicalTypeId::TINYINT:
	case LogicalTypeId::SMALLINT:
	case LogicalTypeId::INTEGER:
	case LogicalTypeId::BIGINT:
	case LogicalTypeId::UTINYINT:
	case LogicalTypeId::USMALLINT:
	case LogicalTypeId::UINTEGER:
	case LogicalTypeId::UBIGINT:
	case LogicalTypeId::HUGEINT:
	case LogicalTypeId::FLOAT:
	case LogicalTypeId::DOUBLE:
		return NumericCastSwitch(input, source, target);
	case LogicalTypeId::POINTER:
		return PointerCastSwitch(input, source, target);
	case LogicalTypeId::UUID:
		return UUIDCastSwitch(input, source, target);
	case LogicalTypeId::DECIMAL:
		return DecimalCastSwitch(input, source, target);
	case LogicalTypeId::DATE:
		return DateCastSwitch(input, source, target);
	case LogicalTypeId::TIME:
		return TimeCastSwitch(input, source, target);
	case LogicalTypeId::TIME_TZ:
		return TimeTzCastSwitch(input, source, target);
	case LogicalTypeId::TIMESTAMP:
		return TimestampCastSwitch(input, source, target);
	case LogicalTypeId::TIMESTAMP_TZ:
		return TimestampTzCastSwitch(input, source, target);
	case LogicalTypeId::TIMESTAMP_NS:
		return TimestampNsCastSwitch(input, source, target);
	case LogicalTypeId::TIMESTAMP_MS:
		return TimestampMsCastSwitch(input, source, target);
	case LogicalTypeId::TIMESTAMP_SEC:
		return TimestampSecCastSwitch(input, source, target);
	case LogicalTypeId::INTERVAL:
		return IntervalCastSwitch(input, source, target);
	case LogicalTypeId::VARCHAR:
		return StringCastSwitch(input, source, target);
	case LogicalTypeId::BLOB:
		return BlobCastSwitch(input, source, target);
	case LogicalTypeId::BIT:
		return BitCastSwitch(input, source, target);
	case LogicalTypeId::SQLNULL:
		return NullTypeCast;
	case LogicalTypeId::MAP:
		return MapCastSwitch(input, source, target);
	case LogicalTypeId::STRUCT:
		return StructCastSwitch(input, source, target);
	case LogicalTypeId::LIST:
		return ListCastSwitch(input, source, target);
	case LogicalTypeId::UNION:
		return UnionCastSwitch(input, source, target);
	case LogicalTypeId::ENUM:
		return EnumCastSwitch(input, source, target);
	case LogicalTypeId::AGGREGATE_STATE:
		return AggregateStateToBlobCast;
	default:
		return nullptr;
	}
}

} // namespace duckdb




namespace duckdb {

template <class SRC_TYPE, class RES_TYPE>
bool EnumEnumCast(Vector &source, Vector &result, idx_t count, CastParameters &parameters) {
	bool all_converted = true;
	result.SetVectorType(VectorType::FLAT_VECTOR);

	auto &str_vec = EnumType::GetValuesInsertOrder(source.GetType());
	auto str_vec_ptr = FlatVector::GetData<string_t>(str_vec);

	auto res_enum_type = result.GetType();

	UnifiedVectorFormat vdata;
	source.ToUnifiedFormat(count, vdata);

	auto source_data = (SRC_TYPE *)vdata.data;
	auto source_sel = vdata.sel;
	auto source_mask = vdata.validity;

	auto result_data = FlatVector::GetData<RES_TYPE>(result);
	auto &result_mask = FlatVector::Validity(result);

	for (idx_t i = 0; i < count; i++) {
		auto src_idx = source_sel->get_index(i);
		if (!source_mask.RowIsValid(src_idx)) {
			result_mask.SetInvalid(i);
			continue;
		}
		auto key = EnumType::GetPos(res_enum_type, str_vec_ptr[source_data[src_idx]]);
		if (key == -1) {
			// key doesn't exist on result enum
			if (!parameters.error_message) {
				result_data[i] = HandleVectorCastError::Operation<RES_TYPE>(
				    CastExceptionText<SRC_TYPE, RES_TYPE>(source_data[src_idx]), result_mask, i,
				    parameters.error_message, all_converted);
			} else {
				result_mask.SetInvalid(i);
			}
			continue;
		}
		result_data[i] = key;
	}
	return all_converted;
}

template <class SRC_TYPE>
BoundCastInfo EnumEnumCastSwitch(BindCastInput &input, const LogicalType &source, const LogicalType &target) {
	switch (target.InternalType()) {
	case PhysicalType::UINT8:
		return EnumEnumCast<SRC_TYPE, uint8_t>;
	case PhysicalType::UINT16:
		return EnumEnumCast<SRC_TYPE, uint16_t>;
	case PhysicalType::UINT32:
		return EnumEnumCast<SRC_TYPE, uint32_t>;
	default:
		throw InternalException("ENUM can only have unsigned integers (except UINT64) as physical types");
	}
}

template <class SRC>
static bool EnumToVarcharCast(Vector &source, Vector &result, idx_t count, CastParameters &parameters) {
	auto &enum_dictionary = EnumType::GetValuesInsertOrder(source.GetType());
	auto dictionary_data = FlatVector::GetData<string_t>(enum_dictionary);
	auto result_data = FlatVector::GetData<string_t>(result);
	auto &result_mask = FlatVector::Validity(result);

	UnifiedVectorFormat vdata;
	source.ToUnifiedFormat(count, vdata);

	auto source_data = (SRC *)vdata.data;
	for (idx_t i = 0; i < count; i++) {
		auto source_idx = vdata.sel->get_index(i);
		if (!vdata.validity.RowIsValid(source_idx)) {
			result_mask.SetInvalid(i);
			continue;
		}
		auto enum_idx = source_data[source_idx];
		result_data[i] = dictionary_data[enum_idx];
	}
	if (source.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	} else {
		result.SetVectorType(VectorType::FLAT_VECTOR);
	}
	return true;
}

struct EnumBoundCastData : public BoundCastData {
	EnumBoundCastData(BoundCastInfo to_varchar_cast, BoundCastInfo from_varchar_cast)
	    : to_varchar_cast(std::move(to_varchar_cast)), from_varchar_cast(std::move(from_varchar_cast)) {
	}

	BoundCastInfo to_varchar_cast;
	BoundCastInfo from_varchar_cast;

public:
	unique_ptr<BoundCastData> Copy() const override {
		return make_uniq<EnumBoundCastData>(to_varchar_cast.Copy(), from_varchar_cast.Copy());
	}
};

unique_ptr<BoundCastData> BindEnumCast(BindCastInput &input, const LogicalType &source, const LogicalType &target) {
	auto to_varchar_cast = input.GetCastFunction(source, LogicalType::VARCHAR);
	auto from_varchar_cast = input.GetCastFunction(LogicalType::VARCHAR, target);
	return make_uniq<EnumBoundCastData>(std::move(to_varchar_cast), std::move(from_varchar_cast));
}

struct EnumCastLocalState : public FunctionLocalState {
public:
	unique_ptr<FunctionLocalState> to_varchar_local;
	unique_ptr<FunctionLocalState> from_varchar_local;
};

static unique_ptr<FunctionLocalState> InitEnumCastLocalState(CastLocalStateParameters &parameters) {
	auto &cast_data = parameters.cast_data->Cast<EnumBoundCastData>();
	auto result = make_uniq<EnumCastLocalState>();

	if (cast_data.from_varchar_cast.init_local_state) {
		CastLocalStateParameters from_varchar_params(parameters, cast_data.from_varchar_cast.cast_data);
		result->from_varchar_local = cast_data.from_varchar_cast.init_local_state(from_varchar_params);
	}
	if (cast_data.to_varchar_cast.init_local_state) {
		CastLocalStateParameters from_varchar_params(parameters, cast_data.to_varchar_cast.cast_data);
		result->from_varchar_local = cast_data.to_varchar_cast.init_local_state(from_varchar_params);
	}
	return std::move(result);
}

static bool EnumToAnyCast(Vector &source, Vector &result, idx_t count, CastParameters &parameters) {
	auto &cast_data = parameters.cast_data->Cast<EnumBoundCastData>();
	auto &lstate = parameters.local_state->Cast<EnumCastLocalState>();

	Vector varchar_cast(LogicalType::VARCHAR, count);

	// cast to varchar
	CastParameters to_varchar_params(parameters, cast_data.to_varchar_cast.cast_data, lstate.to_varchar_local);
	cast_data.to_varchar_cast.function(source, varchar_cast, count, to_varchar_params);

	// cast from varchar to the target
	CastParameters from_varchar_params(parameters, cast_data.from_varchar_cast.cast_data, lstate.from_varchar_local);
	cast_data.from_varchar_cast.function(varchar_cast, result, count, from_varchar_params);
	return true;
}

BoundCastInfo DefaultCasts::EnumCastSwitch(BindCastInput &input, const LogicalType &source, const LogicalType &target) {
	auto enum_physical_type = source.InternalType();
	switch (target.id()) {
	case LogicalTypeId::ENUM: {
		// This means they are both ENUMs, but of different types.
		switch (enum_physical_type) {
		case PhysicalType::UINT8:
			return EnumEnumCastSwitch<uint8_t>(input, source, target);
		case PhysicalType::UINT16:
			return EnumEnumCastSwitch<uint16_t>(input, source, target);
		case PhysicalType::UINT32:
			return EnumEnumCastSwitch<uint32_t>(input, source, target);
		default:
			throw InternalException("ENUM can only have unsigned integers (except UINT64) as physical types");
		}
	}
	case LogicalTypeId::VARCHAR:
		switch (enum_physical_type) {
		case PhysicalType::UINT8:
			return EnumToVarcharCast<uint8_t>;
		case PhysicalType::UINT16:
			return EnumToVarcharCast<uint16_t>;
		case PhysicalType::UINT32:
			return EnumToVarcharCast<uint32_t>;
		default:
			throw InternalException("ENUM can only have unsigned integers (except UINT64) as physical types");
		}
	default: {
		return BoundCastInfo(EnumToAnyCast, BindEnumCast(input, source, target), InitEnumCastLocalState);
	}
	}
}

} // namespace duckdb




namespace duckdb {

unique_ptr<BoundCastData> ListBoundCastData::BindListToListCast(BindCastInput &input, const LogicalType &source,
                                                                const LogicalType &target) {
	vector<BoundCastInfo> child_cast_info;
	auto &source_child_type = ListType::GetChildType(source);
	auto &result_child_type = ListType::GetChildType(target);
	auto child_cast = input.GetCastFunction(source_child_type, result_child_type);
	return make_uniq<ListBoundCastData>(std::move(child_cast));
}

unique_ptr<FunctionLocalState> ListBoundCastData::InitListLocalState(CastLocalStateParameters &parameters) {
	auto &cast_data = parameters.cast_data->Cast<ListBoundCastData>();
	if (!cast_data.child_cast_info.init_local_state) {
		return nullptr;
	}
	CastLocalStateParameters child_parameters(parameters, cast_data.child_cast_info.cast_data);
	return cast_data.child_cast_info.init_local_state(child_parameters);
}

bool ListCast::ListToListCast(Vector &source, Vector &result, idx_t count, CastParameters &parameters) {
	auto &cast_data = parameters.cast_data->Cast<ListBoundCastData>();

	// only handle constant and flat vectors here for now
	if (source.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		result.SetVectorType(source.GetVectorType());
		ConstantVector::SetNull(result, ConstantVector::IsNull(source));

		auto ldata = ConstantVector::GetData<list_entry_t>(source);
		auto tdata = ConstantVector::GetData<list_entry_t>(result);
		*tdata = *ldata;
	} else {
		source.Flatten(count);
		result.SetVectorType(VectorType::FLAT_VECTOR);
		FlatVector::SetValidity(result, FlatVector::Validity(source));

		auto ldata = FlatVector::GetData<list_entry_t>(source);
		auto tdata = FlatVector::GetData<list_entry_t>(result);
		for (idx_t i = 0; i < count; i++) {
			tdata[i] = ldata[i];
		}
	}
	auto &source_cc = ListVector::GetEntry(source);
	auto source_size = ListVector::GetListSize(source);

	ListVector::Reserve(result, source_size);
	auto &append_vector = ListVector::GetEntry(result);

	CastParameters child_parameters(parameters, cast_data.child_cast_info.cast_data, parameters.local_state);
	if (!cast_data.child_cast_info.function(source_cc, append_vector, source_size, child_parameters)) {
		return false;
	}
	ListVector::SetListSize(result, source_size);
	D_ASSERT(ListVector::GetListSize(result) == source_size);
	return true;
}

static bool ListToVarcharCast(Vector &source, Vector &result, idx_t count, CastParameters &parameters) {
	auto constant = source.GetVectorType() == VectorType::CONSTANT_VECTOR;
	// first cast the child vector to varchar
	Vector varchar_list(LogicalType::LIST(LogicalType::VARCHAR), count);
	ListCast::ListToListCast(source, varchar_list, count, parameters);

	// now construct the actual varchar vector
	varchar_list.Flatten(count);
	auto &child = ListVector::GetEntry(varchar_list);
	auto list_data = FlatVector::GetData<list_entry_t>(varchar_list);
	auto &validity = FlatVector::Validity(varchar_list);

	child.Flatten(count);
	auto child_data = FlatVector::GetData<string_t>(child);
	auto &child_validity = FlatVector::Validity(child);

	auto result_data = FlatVector::GetData<string_t>(result);
	static constexpr const idx_t SEP_LENGTH = 2;
	static constexpr const idx_t NULL_LENGTH = 4;
	for (idx_t i = 0; i < count; i++) {
		if (!validity.RowIsValid(i)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}
		auto list = list_data[i];
		// figure out how long the result needs to be
		idx_t list_length = 2; // "[" and "]"
		for (idx_t list_idx = 0; list_idx < list.length; list_idx++) {
			auto idx = list.offset + list_idx;
			if (list_idx > 0) {
				list_length += SEP_LENGTH; // ", "
			}
			// string length, or "NULL"
			list_length += child_validity.RowIsValid(idx) ? child_data[idx].GetSize() : NULL_LENGTH;
		}
		result_data[i] = StringVector::EmptyString(result, list_length);
		auto dataptr = result_data[i].GetDataWriteable();
		auto offset = 0;
		dataptr[offset++] = '[';
		for (idx_t list_idx = 0; list_idx < list.length; list_idx++) {
			auto idx = list.offset + list_idx;
			if (list_idx > 0) {
				memcpy(dataptr + offset, ", ", SEP_LENGTH);
				offset += SEP_LENGTH;
			}
			if (child_validity.RowIsValid(idx)) {
				auto len = child_data[idx].GetSize();
				memcpy(dataptr + offset, child_data[idx].GetData(), len);
				offset += len;
			} else {
				memcpy(dataptr + offset, "NULL", NULL_LENGTH);
				offset += NULL_LENGTH;
			}
		}
		dataptr[offset] = ']';
		result_data[i].Finalize();
	}

	if (constant) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
	return true;
}

BoundCastInfo DefaultCasts::ListCastSwitch(BindCastInput &input, const LogicalType &source, const LogicalType &target) {
	switch (target.id()) {
	case LogicalTypeId::LIST:
		return BoundCastInfo(ListCast::ListToListCast, ListBoundCastData::BindListToListCast(input, source, target),
		                     ListBoundCastData::InitListLocalState);
	case LogicalTypeId::VARCHAR:
		return BoundCastInfo(
		    ListToVarcharCast,
		    ListBoundCastData::BindListToListCast(input, source, LogicalType::LIST(LogicalType::VARCHAR)),
		    ListBoundCastData::InitListLocalState);
	default:
		return DefaultCasts::TryVectorNullCast;
	}
}

} // namespace duckdb




namespace duckdb {

unique_ptr<BoundCastData> MapBoundCastData::BindMapToMapCast(BindCastInput &input, const LogicalType &source,
                                                             const LogicalType &target) {
	vector<BoundCastInfo> child_cast_info;
	auto source_key = MapType::KeyType(source);
	auto target_key = MapType::KeyType(target);
	auto source_val = MapType::ValueType(source);
	auto target_val = MapType::ValueType(target);
	auto key_cast = input.GetCastFunction(source_key, target_key);
	auto value_cast = input.GetCastFunction(source_val, target_val);
	return make_uniq<MapBoundCastData>(std::move(key_cast), std::move(value_cast));
}

static bool MapToVarcharCast(Vector &source, Vector &result, idx_t count, CastParameters &parameters) {
	auto constant = source.GetVectorType() == VectorType::CONSTANT_VECTOR;
	auto varchar_type = LogicalType::MAP(LogicalType::VARCHAR, LogicalType::VARCHAR);
	Vector varchar_map(varchar_type, count);

	// since map's physical type is a list, the ListCast can be utilized
	ListCast::ListToListCast(source, varchar_map, count, parameters);

	varchar_map.Flatten(count);
	auto &validity = FlatVector::Validity(varchar_map);
	auto &key_str = MapVector::GetKeys(varchar_map);
	auto &val_str = MapVector::GetValues(varchar_map);

	key_str.Flatten(ListVector::GetListSize(source));
	val_str.Flatten(ListVector::GetListSize(source));

	auto list_data = ListVector::GetData(varchar_map);
	auto key_data = FlatVector::GetData<string_t>(key_str);
	auto val_data = FlatVector::GetData<string_t>(val_str);
	auto &key_validity = FlatVector::Validity(key_str);
	auto &val_validity = FlatVector::Validity(val_str);
	auto &struct_validity = FlatVector::Validity(ListVector::GetEntry(varchar_map));

	auto result_data = FlatVector::GetData<string_t>(result);
	for (idx_t i = 0; i < count; i++) {
		if (!validity.RowIsValid(i)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}
		auto list = list_data[i];
		string ret = "{";
		for (idx_t list_idx = 0; list_idx < list.length; list_idx++) {
			if (list_idx > 0) {
				ret += ", ";
			}
			auto idx = list.offset + list_idx;

			if (!struct_validity.RowIsValid(idx)) {
				ret += "NULL";
				continue;
			}
			if (!key_validity.RowIsValid(idx)) {
				// throw InternalException("Error in map: key validity invalid?!");
				ret += "invalid";
				continue;
			}
			ret += key_data[idx].GetString();
			ret += "=";
			ret += val_validity.RowIsValid(idx) ? val_data[idx].GetString() : "NULL";
		}
		ret += "}";
		result_data[i] = StringVector::AddString(result, ret);
	}

	if (constant) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
	return true;
}

BoundCastInfo DefaultCasts::MapCastSwitch(BindCastInput &input, const LogicalType &source, const LogicalType &target) {
	switch (target.id()) {
	case LogicalTypeId::MAP:
		return BoundCastInfo(ListCast::ListToListCast, ListBoundCastData::BindListToListCast(input, source, target),
		                     ListBoundCastData::InitListLocalState);
	case LogicalTypeId::VARCHAR: {
		auto varchar_type = LogicalType::MAP(LogicalType::VARCHAR, LogicalType::VARCHAR);
		return BoundCastInfo(MapToVarcharCast, ListBoundCastData::BindListToListCast(input, source, varchar_type),
		                     ListBoundCastData::InitListLocalState);
	}
	default:
		return TryVectorNullCast;
	}
}

} // namespace duckdb





namespace duckdb {

template <class SRC>
static BoundCastInfo InternalNumericCastSwitch(const LogicalType &source, const LogicalType &target) {
	// now switch on the result type
	switch (target.id()) {
	case LogicalTypeId::BOOLEAN:
		return BoundCastInfo(&VectorCastHelpers::TryCastLoop<SRC, bool, duckdb::NumericTryCast>);
	case LogicalTypeId::TINYINT:
		return BoundCastInfo(&VectorCastHelpers::TryCastLoop<SRC, int8_t, duckdb::NumericTryCast>);
	case LogicalTypeId::SMALLINT:
		return BoundCastInfo(&VectorCastHelpers::TryCastLoop<SRC, int16_t, duckdb::NumericTryCast>);
	case LogicalTypeId::INTEGER:
		return BoundCastInfo(&VectorCastHelpers::TryCastLoop<SRC, int32_t, duckdb::NumericTryCast>);
	case LogicalTypeId::BIGINT:
		return BoundCastInfo(&VectorCastHelpers::TryCastLoop<SRC, int64_t, duckdb::NumericTryCast>);
	case LogicalTypeId::UTINYINT:
		return BoundCastInfo(&VectorCastHelpers::TryCastLoop<SRC, uint8_t, duckdb::NumericTryCast>);
	case LogicalTypeId::USMALLINT:
		return BoundCastInfo(&VectorCastHelpers::TryCastLoop<SRC, uint16_t, duckdb::NumericTryCast>);
	case LogicalTypeId::UINTEGER:
		return BoundCastInfo(&VectorCastHelpers::TryCastLoop<SRC, uint32_t, duckdb::NumericTryCast>);
	case LogicalTypeId::UBIGINT:
		return BoundCastInfo(&VectorCastHelpers::TryCastLoop<SRC, uint64_t, duckdb::NumericTryCast>);
	case LogicalTypeId::HUGEINT:
		return BoundCastInfo(&VectorCastHelpers::TryCastLoop<SRC, hugeint_t, duckdb::NumericTryCast>);
	case LogicalTypeId::FLOAT:
		return BoundCastInfo(&VectorCastHelpers::TryCastLoop<SRC, float, duckdb::NumericTryCast>);
	case LogicalTypeId::DOUBLE:
		return BoundCastInfo(&VectorCastHelpers::TryCastLoop<SRC, double, duckdb::NumericTryCast>);
	case LogicalTypeId::DECIMAL:
		return BoundCastInfo(&VectorCastHelpers::ToDecimalCast<SRC>);
	case LogicalTypeId::VARCHAR:
		return BoundCastInfo(&VectorCastHelpers::StringCast<SRC, duckdb::StringCast>);
	default:
		return DefaultCasts::TryVectorNullCast;
	}
}

BoundCastInfo DefaultCasts::NumericCastSwitch(BindCastInput &input, const LogicalType &source,
                                              const LogicalType &target) {
	switch (source.id()) {
	case LogicalTypeId::BOOLEAN:
		return InternalNumericCastSwitch<bool>(source, target);
	case LogicalTypeId::TINYINT:
		return InternalNumericCastSwitch<int8_t>(source, target);
	case LogicalTypeId::SMALLINT:
		return InternalNumericCastSwitch<int16_t>(source, target);
	case LogicalTypeId::INTEGER:
		return InternalNumericCastSwitch<int32_t>(source, target);
	case LogicalTypeId::BIGINT:
		return InternalNumericCastSwitch<int64_t>(source, target);
	case LogicalTypeId::UTINYINT:
		return InternalNumericCastSwitch<uint8_t>(source, target);
	case LogicalTypeId::USMALLINT:
		return InternalNumericCastSwitch<uint16_t>(source, target);
	case LogicalTypeId::UINTEGER:
		return InternalNumericCastSwitch<uint32_t>(source, target);
	case LogicalTypeId::UBIGINT:
		return InternalNumericCastSwitch<uint64_t>(source, target);
	case LogicalTypeId::HUGEINT:
		return InternalNumericCastSwitch<hugeint_t>(source, target);
	case LogicalTypeId::FLOAT:
		return InternalNumericCastSwitch<float>(source, target);
	case LogicalTypeId::DOUBLE:
		return InternalNumericCastSwitch<double>(source, target);
	default:
		throw InternalException("NumericCastSwitch called with non-numeric argument");
	}
}

} // namespace duckdb



namespace duckdb {

BoundCastInfo DefaultCasts::PointerCastSwitch(BindCastInput &input, const LogicalType &source,
                                              const LogicalType &target) {
	// now switch on the result type
	switch (target.id()) {
	case LogicalTypeId::VARCHAR:
		// pointer to varchar
		return BoundCastInfo(&VectorCastHelpers::StringCast<uintptr_t, duckdb::CastFromPointer>);
	default:
		return nullptr;
	}
}

} // namespace duckdb







namespace duckdb {

template <class T>
bool StringEnumCastLoop(string_t *source_data, ValidityMask &source_mask, const LogicalType &source_type,
                        T *result_data, ValidityMask &result_mask, const LogicalType &result_type, idx_t count,
                        string *error_message, const SelectionVector *sel) {
	bool all_converted = true;
	for (idx_t i = 0; i < count; i++) {
		idx_t source_idx = i;
		if (sel) {
			source_idx = sel->get_index(i);
		}
		if (source_mask.RowIsValid(source_idx)) {
			auto pos = EnumType::GetPos(result_type, source_data[source_idx]);
			if (pos == -1) {
				result_data[i] =
				    HandleVectorCastError::Operation<T>(CastExceptionText<string_t, T>(source_data[source_idx]),
				                                        result_mask, i, error_message, all_converted);
			} else {
				result_data[i] = pos;
			}
		} else {
			result_mask.SetInvalid(i);
		}
	}
	return all_converted;
}

template <class T>
bool StringEnumCast(Vector &source, Vector &result, idx_t count, CastParameters &parameters) {
	D_ASSERT(source.GetType().id() == LogicalTypeId::VARCHAR);
	auto enum_name = EnumType::GetTypeName(result.GetType());
	switch (source.GetVectorType()) {
	case VectorType::CONSTANT_VECTOR: {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);

		auto source_data = ConstantVector::GetData<string_t>(source);
		auto source_mask = ConstantVector::Validity(source);
		auto result_data = ConstantVector::GetData<T>(result);
		auto &result_mask = ConstantVector::Validity(result);

		return StringEnumCastLoop(source_data, source_mask, source.GetType(), result_data, result_mask,
		                          result.GetType(), 1, parameters.error_message, nullptr);
	}
	default: {
		UnifiedVectorFormat vdata;
		source.ToUnifiedFormat(count, vdata);

		result.SetVectorType(VectorType::FLAT_VECTOR);

		auto source_data = (string_t *)vdata.data;
		auto source_sel = vdata.sel;
		auto source_mask = vdata.validity;
		auto result_data = FlatVector::GetData<T>(result);
		auto &result_mask = FlatVector::Validity(result);

		return StringEnumCastLoop(source_data, source_mask, source.GetType(), result_data, result_mask,
		                          result.GetType(), count, parameters.error_message, source_sel);
	}
	}
}

static BoundCastInfo VectorStringCastNumericSwitch(BindCastInput &input, const LogicalType &source,
                                                   const LogicalType &target) {
	// now switch on the result type
	switch (target.id()) {
	case LogicalTypeId::ENUM: {
		switch (target.InternalType()) {
		case PhysicalType::UINT8:
			return StringEnumCast<uint8_t>;
		case PhysicalType::UINT16:
			return StringEnumCast<uint16_t>;
		case PhysicalType::UINT32:
			return StringEnumCast<uint32_t>;
		default:
			throw InternalException("ENUM can only have unsigned integers (except UINT64) as physical types");
		}
	}
	case LogicalTypeId::BOOLEAN:
		return BoundCastInfo(&VectorCastHelpers::TryCastStrictLoop<string_t, bool, duckdb::TryCast>);
	case LogicalTypeId::TINYINT:
		return BoundCastInfo(&VectorCastHelpers::TryCastStrictLoop<string_t, int8_t, duckdb::TryCast>);
	case LogicalTypeId::SMALLINT:
		return BoundCastInfo(&VectorCastHelpers::TryCastStrictLoop<string_t, int16_t, duckdb::TryCast>);
	case LogicalTypeId::INTEGER:
		return BoundCastInfo(&VectorCastHelpers::TryCastStrictLoop<string_t, int32_t, duckdb::TryCast>);
	case LogicalTypeId::BIGINT:
		return BoundCastInfo(&VectorCastHelpers::TryCastStrictLoop<string_t, int64_t, duckdb::TryCast>);
	case LogicalTypeId::UTINYINT:
		return BoundCastInfo(&VectorCastHelpers::TryCastStrictLoop<string_t, uint8_t, duckdb::TryCast>);
	case LogicalTypeId::USMALLINT:
		return BoundCastInfo(&VectorCastHelpers::TryCastStrictLoop<string_t, uint16_t, duckdb::TryCast>);
	case LogicalTypeId::UINTEGER:
		return BoundCastInfo(&VectorCastHelpers::TryCastStrictLoop<string_t, uint32_t, duckdb::TryCast>);
	case LogicalTypeId::UBIGINT:
		return BoundCastInfo(&VectorCastHelpers::TryCastStrictLoop<string_t, uint64_t, duckdb::TryCast>);
	case LogicalTypeId::HUGEINT:
		return BoundCastInfo(&VectorCastHelpers::TryCastStrictLoop<string_t, hugeint_t, duckdb::TryCast>);
	case LogicalTypeId::FLOAT:
		return BoundCastInfo(&VectorCastHelpers::TryCastStrictLoop<string_t, float, duckdb::TryCast>);
	case LogicalTypeId::DOUBLE:
		return BoundCastInfo(&VectorCastHelpers::TryCastStrictLoop<string_t, double, duckdb::TryCast>);
	case LogicalTypeId::INTERVAL:
		return BoundCastInfo(&VectorCastHelpers::TryCastErrorLoop<string_t, interval_t, duckdb::TryCastErrorMessage>);
	case LogicalTypeId::DECIMAL:
		return BoundCastInfo(&VectorCastHelpers::ToDecimalCast<string_t>);
	default:
		return DefaultCasts::TryVectorNullCast;
	}
}

//===--------------------------------------------------------------------===//
// string -> list casting
//===--------------------------------------------------------------------===//
bool VectorStringToList::StringToNestedTypeCastLoop(string_t *source_data, ValidityMask &source_mask, Vector &result,
                                                    ValidityMask &result_mask, idx_t count, CastParameters &parameters,
                                                    const SelectionVector *sel) {
	idx_t total_list_size = 0;
	for (idx_t i = 0; i < count; i++) {
		idx_t idx = i;
		if (sel) {
			idx = sel->get_index(i);
		}
		if (!source_mask.RowIsValid(idx)) {
			continue;
		}
		total_list_size += VectorStringToList::CountPartsList(source_data[idx]);
	}

	Vector varchar_vector(LogicalType::VARCHAR, total_list_size);

	ListVector::Reserve(result, total_list_size);
	ListVector::SetListSize(result, total_list_size);

	auto list_data = ListVector::GetData(result);
	auto child_data = FlatVector::GetData<string_t>(varchar_vector);

	bool all_converted = true;
	idx_t total = 0;
	for (idx_t i = 0; i < count; i++) {
		idx_t idx = i;
		if (sel) {
			idx = sel->get_index(i);
		}
		if (!source_mask.RowIsValid(idx)) {
			result_mask.SetInvalid(i);
			continue;
		}

		list_data[i].offset = total;
		if (!VectorStringToList::SplitStringList(source_data[idx], child_data, total, varchar_vector)) {
			string text = "Type VARCHAR with value '" + source_data[idx].GetString() +
			              "' can't be cast to the destination type LIST";
			HandleVectorCastError::Operation<string_t>(text, result_mask, idx, parameters.error_message, all_converted);
		}
		list_data[i].length = total - list_data[i].offset; // length is the amount of parts coming from this string
	}
	D_ASSERT(total_list_size == total);

	auto &result_child = ListVector::GetEntry(result);
	auto &cast_data = parameters.cast_data->Cast<ListBoundCastData>();
	CastParameters child_parameters(parameters, cast_data.child_cast_info.cast_data, parameters.local_state);
	return cast_data.child_cast_info.function(varchar_vector, result_child, total_list_size, child_parameters) &&
	       all_converted;
}

static LogicalType InitVarcharStructType(const LogicalType &target) {
	child_list_t<LogicalType> child_types;
	for (auto &child : StructType::GetChildTypes(target)) {
		child_types.push_back(make_pair(child.first, LogicalType::VARCHAR));
	}

	return LogicalType::STRUCT(child_types);
}

//===--------------------------------------------------------------------===//
// string -> struct casting
//===--------------------------------------------------------------------===//
bool VectorStringToStruct::StringToNestedTypeCastLoop(string_t *source_data, ValidityMask &source_mask, Vector &result,
                                                      ValidityMask &result_mask, idx_t count,
                                                      CastParameters &parameters, const SelectionVector *sel) {
	auto varchar_struct_type = InitVarcharStructType(result.GetType());
	Vector varchar_vector(varchar_struct_type, count);
	auto &child_vectors = StructVector::GetEntries(varchar_vector);
	auto &result_children = StructVector::GetEntries(result);

	string_map_t<idx_t> child_names;
	vector<ValidityMask *> child_masks;
	for (idx_t child_idx = 0; child_idx < result_children.size(); child_idx++) {
		child_names.insert({StructType::GetChildName(result.GetType(), child_idx), child_idx});
		child_masks.emplace_back(&FlatVector::Validity(*child_vectors[child_idx]));
		child_masks[child_idx]->SetAllInvalid(count);
	}

	bool all_converted = true;
	for (idx_t i = 0; i < count; i++) {
		idx_t idx = i;
		if (sel) {
			idx = sel->get_index(i);
		}
		if (!source_mask.RowIsValid(idx)) {
			result_mask.SetInvalid(i);
			continue;
		}
		if (!VectorStringToStruct::SplitStruct(source_data[idx], child_vectors, i, child_names, child_masks)) {
			string text = "Type VARCHAR with value '" + source_data[idx].GetString() +
			              "' can't be cast to the destination type STRUCT";
			for (auto &child_mask : child_masks) {
				child_mask->SetInvalid(idx); // some values may have already been found and set valid
			}
			HandleVectorCastError::Operation<string_t>(text, result_mask, idx, parameters.error_message, all_converted);
		}
	}

	auto &cast_data = parameters.cast_data->Cast<StructBoundCastData>();
	auto &lstate = parameters.local_state->Cast<StructCastLocalState>();
	D_ASSERT(cast_data.child_cast_info.size() == result_children.size());

	for (idx_t child_idx = 0; child_idx < result_children.size(); child_idx++) {
		auto &child_varchar_vector = *child_vectors[child_idx];
		auto &result_child_vector = *result_children[child_idx];
		auto &child_cast_info = cast_data.child_cast_info[child_idx];
		CastParameters child_parameters(parameters, child_cast_info.cast_data, lstate.local_states[child_idx]);
		if (!child_cast_info.function(child_varchar_vector, result_child_vector, count, child_parameters)) {
			all_converted = false;
		}
	}
	return all_converted;
}

//===--------------------------------------------------------------------===//
// string -> map casting
//===--------------------------------------------------------------------===//
unique_ptr<FunctionLocalState> InitMapCastLocalState(CastLocalStateParameters &parameters) {
	auto &cast_data = parameters.cast_data->Cast<MapBoundCastData>();
	auto result = make_uniq<MapCastLocalState>();

	if (cast_data.key_cast.init_local_state) {
		CastLocalStateParameters child_params(parameters, cast_data.key_cast.cast_data);
		result->key_state = cast_data.key_cast.init_local_state(child_params);
	}
	if (cast_data.value_cast.init_local_state) {
		CastLocalStateParameters child_params(parameters, cast_data.value_cast.cast_data);
		result->value_state = cast_data.value_cast.init_local_state(child_params);
	}
	return std::move(result);
}

bool VectorStringToMap::StringToNestedTypeCastLoop(string_t *source_data, ValidityMask &source_mask, Vector &result,
                                                   ValidityMask &result_mask, idx_t count, CastParameters &parameters,
                                                   const SelectionVector *sel) {
	idx_t total_elements = 0;
	for (idx_t i = 0; i < count; i++) {
		idx_t idx = i;
		if (sel) {
			idx = sel->get_index(i);
		}
		if (!source_mask.RowIsValid(idx)) {
			continue;
		}
		total_elements += (VectorStringToMap::CountPartsMap(source_data[idx]) + 1) / 2;
	}

	Vector varchar_key_vector(LogicalType::VARCHAR, total_elements);
	Vector varchar_val_vector(LogicalType::VARCHAR, total_elements);
	auto child_key_data = FlatVector::GetData<string_t>(varchar_key_vector);
	auto child_val_data = FlatVector::GetData<string_t>(varchar_val_vector);

	ListVector::Reserve(result, total_elements);
	ListVector::SetListSize(result, total_elements);
	auto list_data = ListVector::GetData(result);

	bool all_converted = true;
	idx_t total = 0;
	for (idx_t i = 0; i < count; i++) {
		idx_t idx = i;
		if (sel) {
			idx = sel->get_index(i);
		}
		if (!source_mask.RowIsValid(idx)) {
			result_mask.SetInvalid(idx);
			continue;
		}

		list_data[i].offset = total;
		if (!VectorStringToMap::SplitStringMap(source_data[idx], child_key_data, child_val_data, total,
		                                       varchar_key_vector, varchar_val_vector)) {
			string text = "Type VARCHAR with value '" + source_data[idx].GetString() +
			              "' can't be cast to the destination type MAP";
			FlatVector::SetNull(result, idx, true);
			HandleVectorCastError::Operation<string_t>(text, result_mask, idx, parameters.error_message, all_converted);
		}
		list_data[i].length = total - list_data[i].offset;
	}
	D_ASSERT(total_elements == total);

	auto &result_key_child = MapVector::GetKeys(result);
	auto &result_val_child = MapVector::GetValues(result);
	auto &cast_data = parameters.cast_data->Cast<MapBoundCastData>();
	auto &lstate = parameters.local_state->Cast<MapCastLocalState>();

	CastParameters key_params(parameters, cast_data.key_cast.cast_data, lstate.key_state);
	if (!cast_data.key_cast.function(varchar_key_vector, result_key_child, total_elements, key_params)) {
		all_converted = false;
	}
	CastParameters val_params(parameters, cast_data.value_cast.cast_data, lstate.value_state);
	if (!cast_data.value_cast.function(varchar_val_vector, result_val_child, total_elements, val_params)) {
		all_converted = false;
	}

	auto &key_validity = FlatVector::Validity(result_key_child);
	if (!all_converted) {
		for (idx_t row_idx = 0; row_idx < count; row_idx++) {
			if (!result_mask.RowIsValid(row_idx)) {
				continue;
			}
			auto list = list_data[row_idx];
			for (idx_t list_idx = 0; list_idx < list.length; list_idx++) {
				auto idx = list.offset + list_idx;
				if (!key_validity.RowIsValid(idx)) {
					result_mask.SetInvalid(row_idx);
				}
			}
		}
	}
	MapVector::MapConversionVerify(result, count);
	return all_converted;
}

template <class T>
bool StringToNestedTypeCast(Vector &source, Vector &result, idx_t count, CastParameters &parameters) {
	D_ASSERT(source.GetType().id() == LogicalTypeId::VARCHAR);

	switch (source.GetVectorType()) {
	case VectorType::CONSTANT_VECTOR: {
		auto source_data = ConstantVector::GetData<string_t>(source);
		auto &source_mask = ConstantVector::Validity(source);
		auto &result_mask = FlatVector::Validity(result);
		auto ret = T::StringToNestedTypeCastLoop(source_data, source_mask, result, result_mask, 1, parameters, nullptr);
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
		return ret;
	}
	default: {
		UnifiedVectorFormat unified_source;

		source.ToUnifiedFormat(count, unified_source);
		auto source_sel = unified_source.sel;
		auto source_data = (string_t *)unified_source.data;
		auto &source_mask = unified_source.validity;
		auto &result_mask = FlatVector::Validity(result);

		return T::StringToNestedTypeCastLoop(source_data, source_mask, result, result_mask, count, parameters,
		                                     source_sel);
	}
	}
}

BoundCastInfo DefaultCasts::StringCastSwitch(BindCastInput &input, const LogicalType &source,
                                             const LogicalType &target) {
	switch (target.id()) {
	case LogicalTypeId::DATE:
		return BoundCastInfo(&VectorCastHelpers::TryCastErrorLoop<string_t, date_t, duckdb::TryCastErrorMessage>);
	case LogicalTypeId::TIME:
	case LogicalTypeId::TIME_TZ:
		return BoundCastInfo(&VectorCastHelpers::TryCastErrorLoop<string_t, dtime_t, duckdb::TryCastErrorMessage>);
	case LogicalTypeId::TIMESTAMP:
	case LogicalTypeId::TIMESTAMP_TZ:
		return BoundCastInfo(&VectorCastHelpers::TryCastErrorLoop<string_t, timestamp_t, duckdb::TryCastErrorMessage>);
	case LogicalTypeId::TIMESTAMP_NS:
		return BoundCastInfo(
		    &VectorCastHelpers::TryCastStrictLoop<string_t, timestamp_t, duckdb::TryCastToTimestampNS>);
	case LogicalTypeId::TIMESTAMP_SEC:
		return BoundCastInfo(
		    &VectorCastHelpers::TryCastStrictLoop<string_t, timestamp_t, duckdb::TryCastToTimestampSec>);
	case LogicalTypeId::TIMESTAMP_MS:
		return BoundCastInfo(
		    &VectorCastHelpers::TryCastStrictLoop<string_t, timestamp_t, duckdb::TryCastToTimestampMS>);
	case LogicalTypeId::BLOB:
		return BoundCastInfo(&VectorCastHelpers::TryCastStringLoop<string_t, string_t, duckdb::TryCastToBlob>);
	case LogicalTypeId::BIT:
		return BoundCastInfo(&VectorCastHelpers::TryCastStringLoop<string_t, string_t, duckdb::TryCastToBit>);
	case LogicalTypeId::UUID:
		return BoundCastInfo(&VectorCastHelpers::TryCastStringLoop<string_t, hugeint_t, duckdb::TryCastToUUID>);
	case LogicalTypeId::SQLNULL:
		return &DefaultCasts::TryVectorNullCast;
	case LogicalTypeId::VARCHAR:
		return &DefaultCasts::ReinterpretCast;
	case LogicalTypeId::LIST:
		// the second argument allows for a secondary casting function to be passed in the CastParameters
		return BoundCastInfo(
		    &StringToNestedTypeCast<VectorStringToList>,
		    ListBoundCastData::BindListToListCast(input, LogicalType::LIST(LogicalType::VARCHAR), target),
		    ListBoundCastData::InitListLocalState);
	case LogicalTypeId::STRUCT:
		return BoundCastInfo(&StringToNestedTypeCast<VectorStringToStruct>,
		                     StructBoundCastData::BindStructToStructCast(input, InitVarcharStructType(target), target),
		                     StructBoundCastData::InitStructCastLocalState);
	case LogicalTypeId::MAP:
		return BoundCastInfo(&StringToNestedTypeCast<VectorStringToMap>,
		                     MapBoundCastData::BindMapToMapCast(
		                         input, LogicalType::MAP(LogicalType::VARCHAR, LogicalType::VARCHAR), target),
		                     InitMapCastLocalState);
	default:
		return VectorStringCastNumericSwitch(input, source, target);
	}
}

} // namespace duckdb




namespace duckdb {

unique_ptr<BoundCastData> StructBoundCastData::BindStructToStructCast(BindCastInput &input, const LogicalType &source,
                                                                      const LogicalType &target) {
	vector<BoundCastInfo> child_cast_info;
	auto &source_child_types = StructType::GetChildTypes(source);
	auto &result_child_types = StructType::GetChildTypes(target);
	if (source_child_types.size() != result_child_types.size()) {
		throw TypeMismatchException(source, target, "Cannot cast STRUCTs of different size");
	}
	for (idx_t i = 0; i < source_child_types.size(); i++) {
		auto child_cast = input.GetCastFunction(source_child_types[i].second, result_child_types[i].second);
		child_cast_info.push_back(std::move(child_cast));
	}
	return make_uniq<StructBoundCastData>(std::move(child_cast_info), target);
}

unique_ptr<FunctionLocalState> StructBoundCastData::InitStructCastLocalState(CastLocalStateParameters &parameters) {
	auto &cast_data = parameters.cast_data->Cast<StructBoundCastData>();
	auto result = make_uniq<StructCastLocalState>();

	for (auto &entry : cast_data.child_cast_info) {
		unique_ptr<FunctionLocalState> child_state;
		if (entry.init_local_state) {
			CastLocalStateParameters child_params(parameters, entry.cast_data);
			child_state = entry.init_local_state(child_params);
		}
		result->local_states.push_back(std::move(child_state));
	}
	return std::move(result);
}

static bool StructToStructCast(Vector &source, Vector &result, idx_t count, CastParameters &parameters) {
	auto &cast_data = parameters.cast_data->Cast<StructBoundCastData>();
	auto &lstate = parameters.local_state->Cast<StructCastLocalState>();
	auto &source_child_types = StructType::GetChildTypes(source.GetType());
	auto &source_children = StructVector::GetEntries(source);
	D_ASSERT(source_children.size() == StructType::GetChildTypes(result.GetType()).size());

	auto &result_children = StructVector::GetEntries(result);
	bool all_converted = true;
	for (idx_t c_idx = 0; c_idx < source_child_types.size(); c_idx++) {
		auto &result_child_vector = *result_children[c_idx];
		auto &source_child_vector = *source_children[c_idx];
		CastParameters child_parameters(parameters, cast_data.child_cast_info[c_idx].cast_data,
		                                lstate.local_states[c_idx]);
		if (!cast_data.child_cast_info[c_idx].function(source_child_vector, result_child_vector, count,
		                                               child_parameters)) {
			all_converted = false;
		}
	}
	if (source.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
		ConstantVector::SetNull(result, ConstantVector::IsNull(source));
	} else {
		source.Flatten(count);
		FlatVector::Validity(result) = FlatVector::Validity(source);
	}
	return all_converted;
}

static bool StructToVarcharCast(Vector &source, Vector &result, idx_t count, CastParameters &parameters) {
	auto constant = source.GetVectorType() == VectorType::CONSTANT_VECTOR;
	// first cast all child elements to varchar
	auto &cast_data = parameters.cast_data->Cast<StructBoundCastData>();
	Vector varchar_struct(cast_data.target, count);
	StructToStructCast(source, varchar_struct, count, parameters);

	// now construct the actual varchar vector
	varchar_struct.Flatten(count);
	auto &child_types = StructType::GetChildTypes(source.GetType());
	auto &children = StructVector::GetEntries(varchar_struct);
	auto &validity = FlatVector::Validity(varchar_struct);
	auto result_data = FlatVector::GetData<string_t>(result);
	static constexpr const idx_t SEP_LENGTH = 2;
	static constexpr const idx_t NAME_SEP_LENGTH = 4;
	static constexpr const idx_t NULL_LENGTH = 4;
	for (idx_t i = 0; i < count; i++) {
		if (!validity.RowIsValid(i)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}
		idx_t string_length = 2; // {}
		for (idx_t c = 0; c < children.size(); c++) {
			if (c > 0) {
				string_length += SEP_LENGTH;
			}
			children[c]->Flatten(count);
			auto &child_validity = FlatVector::Validity(*children[c]);
			auto data = FlatVector::GetData<string_t>(*children[c]);
			auto &name = child_types[c].first;
			string_length += name.size() + NAME_SEP_LENGTH; // "'{name}': "
			string_length += child_validity.RowIsValid(i) ? data[i].GetSize() : NULL_LENGTH;
		}
		result_data[i] = StringVector::EmptyString(result, string_length);
		auto dataptr = result_data[i].GetDataWriteable();
		idx_t offset = 0;
		dataptr[offset++] = '{';
		for (idx_t c = 0; c < children.size(); c++) {
			if (c > 0) {
				memcpy(dataptr + offset, ", ", SEP_LENGTH);
				offset += SEP_LENGTH;
			}
			auto &child_validity = FlatVector::Validity(*children[c]);
			auto data = FlatVector::GetData<string_t>(*children[c]);
			auto &name = child_types[c].first;
			// "'{name}': "
			dataptr[offset++] = '\'';
			memcpy(dataptr + offset, name.c_str(), name.size());
			offset += name.size();
			dataptr[offset++] = '\'';
			dataptr[offset++] = ':';
			dataptr[offset++] = ' ';
			// value
			if (child_validity.RowIsValid(i)) {
				auto len = data[i].GetSize();
				memcpy(dataptr + offset, data[i].GetData(), len);
				offset += len;
			} else {
				memcpy(dataptr + offset, "NULL", NULL_LENGTH);
				offset += NULL_LENGTH;
			}
		}
		dataptr[offset++] = '}';
		result_data[i].Finalize();
	}

	if (constant) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
	return true;
}

BoundCastInfo DefaultCasts::StructCastSwitch(BindCastInput &input, const LogicalType &source,
                                             const LogicalType &target) {
	switch (target.id()) {
	case LogicalTypeId::STRUCT:
		return BoundCastInfo(StructToStructCast, StructBoundCastData::BindStructToStructCast(input, source, target),
		                     StructBoundCastData::InitStructCastLocalState);
	case LogicalTypeId::VARCHAR: {
		// bind a cast in which we convert all child entries to VARCHAR entries
		auto &struct_children = StructType::GetChildTypes(source);
		child_list_t<LogicalType> varchar_children;
		for (auto &child_entry : struct_children) {
			varchar_children.push_back(make_pair(child_entry.first, LogicalType::VARCHAR));
		}
		auto varchar_type = LogicalType::STRUCT(varchar_children);
		return BoundCastInfo(StructToVarcharCast,
		                     StructBoundCastData::BindStructToStructCast(input, source, varchar_type),
		                     StructBoundCastData::InitStructCastLocalState);
	}
	default:
		return TryVectorNullCast;
	}
}

} // namespace duckdb



namespace duckdb {

BoundCastInfo DefaultCasts::DateCastSwitch(BindCastInput &input, const LogicalType &source, const LogicalType &target) {
	// now switch on the result type
	switch (target.id()) {
	case LogicalTypeId::VARCHAR:
		// date to varchar
		return BoundCastInfo(&VectorCastHelpers::StringCast<date_t, duckdb::StringCast>);
	case LogicalTypeId::TIMESTAMP:
	case LogicalTypeId::TIMESTAMP_TZ:
		// date to timestamp
		return BoundCastInfo(&VectorCastHelpers::TryCastLoop<date_t, timestamp_t, duckdb::TryCast>);
	case LogicalTypeId::TIMESTAMP_NS:
		return BoundCastInfo(&VectorCastHelpers::TryCastLoop<date_t, timestamp_t, duckdb::TryCastToTimestampNS>);
	case LogicalTypeId::TIMESTAMP_SEC:
		return BoundCastInfo(&VectorCastHelpers::TryCastLoop<date_t, timestamp_t, duckdb::TryCastToTimestampSec>);
	case LogicalTypeId::TIMESTAMP_MS:
		return BoundCastInfo(&VectorCastHelpers::TryCastLoop<date_t, timestamp_t, duckdb::TryCastToTimestampMS>);
	default:
		return TryVectorNullCast;
	}
}

BoundCastInfo DefaultCasts::TimeCastSwitch(BindCastInput &input, const LogicalType &source, const LogicalType &target) {
	// now switch on the result type
	switch (target.id()) {
	case LogicalTypeId::VARCHAR:
		// time to varchar
		return BoundCastInfo(&VectorCastHelpers::StringCast<dtime_t, duckdb::StringCast>);
	case LogicalTypeId::TIME_TZ:
		// time to time with time zone
		return ReinterpretCast;
	default:
		return TryVectorNullCast;
	}
}

BoundCastInfo DefaultCasts::TimeTzCastSwitch(BindCastInput &input, const LogicalType &source,
                                             const LogicalType &target) {
	// now switch on the result type
	switch (target.id()) {
	case LogicalTypeId::VARCHAR:
		// time with time zone to varchar
		return BoundCastInfo(&VectorCastHelpers::StringCast<dtime_t, duckdb::StringCastTZ>);
	case LogicalTypeId::TIME:
		// time with time zone to time
		return ReinterpretCast;
	default:
		return TryVectorNullCast;
	}
}

BoundCastInfo DefaultCasts::TimestampCastSwitch(BindCastInput &input, const LogicalType &source,
                                                const LogicalType &target) {
	// now switch on the result type
	switch (target.id()) {
	case LogicalTypeId::VARCHAR:
		// timestamp to varchar
		return BoundCastInfo(&VectorCastHelpers::StringCast<timestamp_t, duckdb::StringCast>);
	case LogicalTypeId::DATE:
		// timestamp to date
		return BoundCastInfo(&VectorCastHelpers::TemplatedCastLoop<timestamp_t, date_t, duckdb::Cast>);
	case LogicalTypeId::TIME:
	case LogicalTypeId::TIME_TZ:
		// timestamp to time
		return BoundCastInfo(&VectorCastHelpers::TemplatedCastLoop<timestamp_t, dtime_t, duckdb::Cast>);
	case LogicalTypeId::TIMESTAMP_TZ:
		// timestamp (us) to timestamp with time zone
		return ReinterpretCast;
	case LogicalTypeId::TIMESTAMP_NS:
		// timestamp (us) to timestamp (ns)
		return BoundCastInfo(
		    &VectorCastHelpers::TemplatedCastLoop<timestamp_t, timestamp_t, duckdb::CastTimestampUsToNs>);
	case LogicalTypeId::TIMESTAMP_MS:
		// timestamp (us) to timestamp (ms)
		return BoundCastInfo(
		    &VectorCastHelpers::TemplatedCastLoop<timestamp_t, timestamp_t, duckdb::CastTimestampUsToMs>);
	case LogicalTypeId::TIMESTAMP_SEC:
		// timestamp (us) to timestamp (s)
		return BoundCastInfo(
		    &VectorCastHelpers::TemplatedCastLoop<timestamp_t, timestamp_t, duckdb::CastTimestampUsToSec>);
	default:
		return TryVectorNullCast;
	}
}

BoundCastInfo DefaultCasts::TimestampTzCastSwitch(BindCastInput &input, const LogicalType &source,
                                                  const LogicalType &target) {
	// now switch on the result type
	switch (target.id()) {
	case LogicalTypeId::VARCHAR:
		// timestamp with time zone to varchar
		return BoundCastInfo(&VectorCastHelpers::StringCast<timestamp_t, duckdb::StringCastTZ>);
	case LogicalTypeId::TIME_TZ:
		// timestamp with time zone to time with time zone.
		// TODO: set the offset to +00
		return BoundCastInfo(&VectorCastHelpers::TemplatedCastLoop<timestamp_t, dtime_t, duckdb::Cast>);
	case LogicalTypeId::TIMESTAMP:
		// timestamp with time zone to timestamp (us)
		return ReinterpretCast;
	default:
		return TryVectorNullCast;
	}
}

BoundCastInfo DefaultCasts::TimestampNsCastSwitch(BindCastInput &input, const LogicalType &source,
                                                  const LogicalType &target) {
	// now switch on the result type
	switch (target.id()) {
	case LogicalTypeId::VARCHAR:
		// timestamp (ns) to varchar
		return BoundCastInfo(&VectorCastHelpers::StringCast<timestamp_t, duckdb::CastFromTimestampNS>);
	case LogicalTypeId::TIMESTAMP:
		// timestamp (ns) to timestamp (us)
		return BoundCastInfo(
		    &VectorCastHelpers::TemplatedCastLoop<timestamp_t, timestamp_t, duckdb::CastTimestampNsToUs>);
	default:
		return TryVectorNullCast;
	}
}

BoundCastInfo DefaultCasts::TimestampMsCastSwitch(BindCastInput &input, const LogicalType &source,
                                                  const LogicalType &target) {
	// now switch on the result type
	switch (target.id()) {
	case LogicalTypeId::VARCHAR:
		// timestamp (ms) to varchar
		return BoundCastInfo(&VectorCastHelpers::StringCast<timestamp_t, duckdb::CastFromTimestampMS>);
	case LogicalTypeId::TIMESTAMP:
		// timestamp (ms) to timestamp (us)
		return BoundCastInfo(
		    &VectorCastHelpers::TemplatedCastLoop<timestamp_t, timestamp_t, duckdb::CastTimestampMsToUs>);
	default:
		return TryVectorNullCast;
	}
}

BoundCastInfo DefaultCasts::TimestampSecCastSwitch(BindCastInput &input, const LogicalType &source,
                                                   const LogicalType &target) {
	// now switch on the result type
	switch (target.id()) {
	case LogicalTypeId::VARCHAR:
		// timestamp (sec) to varchar
		return BoundCastInfo(&VectorCastHelpers::StringCast<timestamp_t, duckdb::CastFromTimestampSec>);
	case LogicalTypeId::TIMESTAMP:
		// timestamp (s) to timestamp (us)
		return BoundCastInfo(
		    &VectorCastHelpers::TemplatedCastLoop<timestamp_t, timestamp_t, duckdb::CastTimestampSecToUs>);
	default:
		return TryVectorNullCast;
	}
}
BoundCastInfo DefaultCasts::IntervalCastSwitch(BindCastInput &input, const LogicalType &source,
                                               const LogicalType &target) {
	// now switch on the result type
	switch (target.id()) {
	case LogicalTypeId::VARCHAR:
		// time to varchar
		return BoundCastInfo(&VectorCastHelpers::StringCast<interval_t, duckdb::StringCast>);
	default:
		return TryVectorNullCast;
	}
}

} // namespace duckdb




#include <algorithm> // for std::sort

namespace duckdb {

//--------------------------------------------------------------------------------------------------
// ??? -> UNION
//--------------------------------------------------------------------------------------------------
// if the source can be implicitly cast to a member of the target union, the cast is valid

struct ToUnionBoundCastData : public BoundCastData {
	ToUnionBoundCastData(union_tag_t member_idx, string name, LogicalType type, int64_t cost,
	                     BoundCastInfo member_cast_info)
	    : tag(member_idx), name(std::move(name)), type(std::move(type)), cost(cost),
	      member_cast_info(std::move(member_cast_info)) {
	}

	union_tag_t tag;
	string name;
	LogicalType type;
	int64_t cost;
	BoundCastInfo member_cast_info;

public:
	unique_ptr<BoundCastData> Copy() const override {
		return make_uniq<ToUnionBoundCastData>(tag, name, type, cost, member_cast_info.Copy());
	}

	static bool SortByCostAscending(const ToUnionBoundCastData &left, const ToUnionBoundCastData &right) {
		return left.cost < right.cost;
	}
};

unique_ptr<BoundCastData> BindToUnionCast(BindCastInput &input, const LogicalType &source, const LogicalType &target) {
	D_ASSERT(target.id() == LogicalTypeId::UNION);

	vector<ToUnionBoundCastData> candidates;

	for (idx_t member_idx = 0; member_idx < UnionType::GetMemberCount(target); member_idx++) {
		auto member_type = UnionType::GetMemberType(target, member_idx);
		auto member_name = UnionType::GetMemberName(target, member_idx);
		auto member_cast_cost = input.function_set.ImplicitCastCost(source, member_type);
		if (member_cast_cost != -1) {
			auto member_cast_info = input.GetCastFunction(source, member_type);
			candidates.emplace_back(member_idx, member_name, member_type, member_cast_cost,
			                        std::move(member_cast_info));
		}
	};

	// no possible casts found!
	if (candidates.empty()) {
		auto message = StringUtil::Format(
		    "Type %s can't be cast as %s. %s can't be implicitly cast to any of the union member types: ",
		    source.ToString(), target.ToString(), source.ToString());

		auto member_count = UnionType::GetMemberCount(target);
		for (idx_t member_idx = 0; member_idx < member_count; member_idx++) {
			auto member_type = UnionType::GetMemberType(target, member_idx);
			message += member_type.ToString();
			if (member_idx < member_count - 1) {
				message += ", ";
			}
		}
		throw CastException(message);
	}

	// sort the candidate casts by cost
	std::sort(candidates.begin(), candidates.end(), ToUnionBoundCastData::SortByCostAscending);

	// select the lowest possible cost cast
	auto &selected_cast = candidates[0];
	auto selected_cost = candidates[0].cost;

	// check if the cast is ambiguous (2 or more casts have the same cost)
	if (candidates.size() > 1 && candidates[1].cost == selected_cost) {

		// collect all the ambiguous types
		auto message = StringUtil::Format(
		    "Type %s can't be cast as %s. The cast is ambiguous, multiple possible members in target: ", source,
		    target);
		for (size_t i = 0; i < candidates.size(); i++) {
			if (candidates[i].cost == selected_cost) {
				message += StringUtil::Format("'%s (%s)'", candidates[i].name, candidates[i].type.ToString());
				if (i < candidates.size() - 1) {
					message += ", ";
				}
			}
		}
		message += ". Disambiguate the target type by using the 'union_value(<tag> := <arg>)' function to promote the "
		           "source value to a single member union before casting.";
		throw CastException(message);
	}

	// otherwise, return the selected cast
	return make_uniq<ToUnionBoundCastData>(std::move(selected_cast));
}

unique_ptr<FunctionLocalState> InitToUnionLocalState(CastLocalStateParameters &parameters) {
	auto &cast_data = parameters.cast_data->Cast<ToUnionBoundCastData>();
	if (!cast_data.member_cast_info.init_local_state) {
		return nullptr;
	}
	CastLocalStateParameters child_parameters(parameters, cast_data.member_cast_info.cast_data);
	return cast_data.member_cast_info.init_local_state(child_parameters);
}

static bool ToUnionCast(Vector &source, Vector &result, idx_t count, CastParameters &parameters) {
	D_ASSERT(result.GetType().id() == LogicalTypeId::UNION);
	auto &cast_data = parameters.cast_data->Cast<ToUnionBoundCastData>();
	auto &selected_member_vector = UnionVector::GetMember(result, cast_data.tag);

	CastParameters child_parameters(parameters, cast_data.member_cast_info.cast_data, parameters.local_state);
	if (!cast_data.member_cast_info.function(source, selected_member_vector, count, child_parameters)) {
		return false;
	}

	// cast succeeded, create union vector
	UnionVector::SetToMember(result, cast_data.tag, selected_member_vector, count, true);

	result.Verify(count);

	return true;
}

BoundCastInfo DefaultCasts::ImplicitToUnionCast(BindCastInput &input, const LogicalType &source,
                                                const LogicalType &target) {
	return BoundCastInfo(&ToUnionCast, BindToUnionCast(input, source, target), InitToUnionLocalState);
}

//--------------------------------------------------------------------------------------------------
// UNION -> UNION
//--------------------------------------------------------------------------------------------------
// if the source member tags is a subset of the target member tags, and all the source members can be
// implicitly cast to the corresponding target members, the cast is valid.
//
// VALID: 	UNION(A, B) 	-> 	UNION(A, B, C)
// VALID: 	UNION(A, B) 	-> 	UNION(A, C)		if B can be implicitly cast to C
//
// INVALID: UNION(A, B, C)	->	UNION(A, B)
// INVALID:	UNION(A, B) 	->	UNION(A, C)		if B can't be implicitly cast to C
// INVALID:	UNION(A, B, D) 	->	UNION(A, B, C)

struct UnionToUnionBoundCastData : public BoundCastData {

	// mapping from source member index to target member index
	// these are always the same size as the source member count
	// (since all source members must be present in the target)
	vector<idx_t> tag_map;
	vector<BoundCastInfo> member_casts;

	LogicalType target_type;

	UnionToUnionBoundCastData(vector<idx_t> tag_map, vector<BoundCastInfo> member_casts, LogicalType target_type)
	    : tag_map(std::move(tag_map)), member_casts(std::move(member_casts)), target_type(std::move(target_type)) {
	}

public:
	unique_ptr<BoundCastData> Copy() const override {
		vector<BoundCastInfo> member_casts_copy;
		for (auto &member_cast : member_casts) {
			member_casts_copy.push_back(member_cast.Copy());
		}
		return make_uniq<UnionToUnionBoundCastData>(tag_map, std::move(member_casts_copy), target_type);
	}
};

unique_ptr<BoundCastData> BindUnionToUnionCast(BindCastInput &input, const LogicalType &source,
                                               const LogicalType &target) {
	D_ASSERT(source.id() == LogicalTypeId::UNION);
	D_ASSERT(target.id() == LogicalTypeId::UNION);

	auto source_member_count = UnionType::GetMemberCount(source);

	auto tag_map = vector<idx_t>(source_member_count);
	auto member_casts = vector<BoundCastInfo>();

	for (idx_t source_idx = 0; source_idx < source_member_count; source_idx++) {
		auto &source_member_type = UnionType::GetMemberType(source, source_idx);
		auto &source_member_name = UnionType::GetMemberName(source, source_idx);

		bool found = false;
		for (idx_t target_idx = 0; target_idx < UnionType::GetMemberCount(target); target_idx++) {
			auto &target_member_name = UnionType::GetMemberName(target, target_idx);

			// found a matching member
			if (source_member_name == target_member_name) {
				auto &target_member_type = UnionType::GetMemberType(target, target_idx);
				tag_map[source_idx] = target_idx;
				member_casts.push_back(input.GetCastFunction(source_member_type, target_member_type));
				found = true;
				break;
			}
		}
		if (!found) {
			// no matching member tag found in the target set
			auto message =
			    StringUtil::Format("Type %s can't be cast as %s. The member '%s' is not present in target union",
			                       source.ToString(), target.ToString(), source_member_name);
			throw CastException(message);
		}
	}

	return make_uniq<UnionToUnionBoundCastData>(tag_map, std::move(member_casts), target);
}

unique_ptr<FunctionLocalState> InitUnionToUnionLocalState(CastLocalStateParameters &parameters) {
	auto &cast_data = parameters.cast_data->Cast<UnionToUnionBoundCastData>();
	auto result = make_uniq<StructCastLocalState>();

	for (auto &entry : cast_data.member_casts) {
		unique_ptr<FunctionLocalState> child_state;
		if (entry.init_local_state) {
			CastLocalStateParameters child_params(parameters, entry.cast_data);
			child_state = entry.init_local_state(child_params);
		}
		result->local_states.push_back(std::move(child_state));
	}
	return std::move(result);
}

static bool UnionToUnionCast(Vector &source, Vector &result, idx_t count, CastParameters &parameters) {
	auto &cast_data = parameters.cast_data->Cast<UnionToUnionBoundCastData>();
	auto &lstate = parameters.local_state->Cast<StructCastLocalState>();

	auto source_member_count = UnionType::GetMemberCount(source.GetType());
	auto target_member_count = UnionType::GetMemberCount(result.GetType());

	auto target_member_is_mapped = vector<bool>(target_member_count);

	// Perform the casts from source to target members
	for (idx_t member_idx = 0; member_idx < source_member_count; member_idx++) {
		auto target_member_idx = cast_data.tag_map[member_idx];

		auto &source_member_vector = UnionVector::GetMember(source, member_idx);
		auto &target_member_vector = UnionVector::GetMember(result, target_member_idx);
		auto &member_cast = cast_data.member_casts[member_idx];

		CastParameters child_parameters(parameters, member_cast.cast_data, lstate.local_states[member_idx]);
		if (!member_cast.function(source_member_vector, target_member_vector, count, child_parameters)) {
			return false;
		}

		target_member_is_mapped[target_member_idx] = true;
	}

	// All member casts succeeded!

	// Set the unmapped target members to constant NULL.
	// If we cast UNION(A, B) -> UNION(A, B, C) we need to invalidate C so that
	// the invariants of the result union hold. (only member columns "selected"
	// by the rowwise corresponding tag in the tag vector should be valid)
	for (idx_t target_member_idx = 0; target_member_idx < target_member_count; target_member_idx++) {
		if (!target_member_is_mapped[target_member_idx]) {
			auto &target_member_vector = UnionVector::GetMember(result, target_member_idx);
			target_member_vector.SetVectorType(VectorType::CONSTANT_VECTOR);
			ConstantVector::SetNull(target_member_vector, true);
		}
	}

	// Update the tags in the result vector
	auto &source_tag_vector = UnionVector::GetTags(source);
	auto &result_tag_vector = UnionVector::GetTags(result);

	if (source.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		// Constant vector case optimization
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
		if (ConstantVector::IsNull(source)) {
			ConstantVector::SetNull(result, true);
		} else {
			// map the tag
			auto source_tag = ConstantVector::GetData<union_tag_t>(source_tag_vector)[0];
			auto mapped_tag = cast_data.tag_map[source_tag];
			ConstantVector::GetData<union_tag_t>(result_tag_vector)[0] = mapped_tag;
		}
	} else {
		// Otherwise, use the unified vector format to access the source vector.

		// Ensure that all the result members are flat vectors
		// This is not always the case, e.g. when a member is cast using the default TryNullCast function
		// the resulting member vector will be a constant null vector.
		for (idx_t target_member_idx = 0; target_member_idx < target_member_count; target_member_idx++) {
			UnionVector::GetMember(result, target_member_idx).Flatten(count);
		}

		// We assume that a union tag vector validity matches the union vector validity.
		UnifiedVectorFormat source_tag_format;
		source_tag_vector.ToUnifiedFormat(count, source_tag_format);

		for (idx_t row_idx = 0; row_idx < count; row_idx++) {
			auto source_row_idx = source_tag_format.sel->get_index(row_idx);
			if (source_tag_format.validity.RowIsValid(source_row_idx)) {
				// map the tag
				auto source_tag = ((union_tag_t *)source_tag_format.data)[source_row_idx];
				auto target_tag = cast_data.tag_map[source_tag];
				FlatVector::GetData<union_tag_t>(result_tag_vector)[row_idx] = target_tag;
			} else {

				// Issue: The members of the result is not always flatvectors
				// In the case of TryNullCast, the result member is constant.
				FlatVector::SetNull(result, row_idx, true);
			}
		}
	}

	result.Verify(count);

	return true;
}

static bool UnionToVarcharCast(Vector &source, Vector &result, idx_t count, CastParameters &parameters) {
	auto constant = source.GetVectorType() == VectorType::CONSTANT_VECTOR;
	// first cast all union members to varchar
	auto &cast_data = parameters.cast_data->Cast<UnionToUnionBoundCastData>();
	Vector varchar_union(cast_data.target_type, count);

	UnionToUnionCast(source, varchar_union, count, parameters);

	// now construct the actual varchar vector
	varchar_union.Flatten(count);
	auto &tag_vector = UnionVector::GetTags(source);
	auto tags = FlatVector::GetData<union_tag_t>(tag_vector);

	auto &validity = FlatVector::Validity(varchar_union);
	auto result_data = FlatVector::GetData<string_t>(result);

	for (idx_t i = 0; i < count; i++) {
		if (!validity.RowIsValid(i)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}

		auto &member = UnionVector::GetMember(varchar_union, tags[i]);
		UnifiedVectorFormat member_vdata;
		member.ToUnifiedFormat(count, member_vdata);

		auto mapped_idx = member_vdata.sel->get_index(i);
		auto member_valid = member_vdata.validity.RowIsValid(mapped_idx);
		if (member_valid) {
			auto member_str = ((string_t *)member_vdata.data)[mapped_idx];
			result_data[i] = StringVector::AddString(result, member_str);
		} else {
			result_data[i] = StringVector::AddString(result, "NULL");
		}
	}

	if (constant) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}

	result.Verify(count);
	return true;
}

BoundCastInfo DefaultCasts::UnionCastSwitch(BindCastInput &input, const LogicalType &source,
                                            const LogicalType &target) {
	switch (target.id()) {
	case LogicalTypeId::VARCHAR: {
		// bind a cast in which we convert all members to VARCHAR first
		child_list_t<LogicalType> varchar_members;
		for (idx_t member_idx = 0; member_idx < UnionType::GetMemberCount(source); member_idx++) {
			varchar_members.push_back(make_pair(UnionType::GetMemberName(source, member_idx), LogicalType::VARCHAR));
		}
		auto varchar_type = LogicalType::UNION(std::move(varchar_members));
		return BoundCastInfo(UnionToVarcharCast, BindUnionToUnionCast(input, source, varchar_type),
		                     InitUnionToUnionLocalState);
	}
	case LogicalTypeId::UNION:
		return BoundCastInfo(UnionToUnionCast, BindUnionToUnionCast(input, source, target), InitUnionToUnionLocalState);
	default:
		return TryVectorNullCast;
	}
}

} // namespace duckdb




namespace duckdb {

BoundCastInfo DefaultCasts::UUIDCastSwitch(BindCastInput &input, const LogicalType &source, const LogicalType &target) {
	// now switch on the result type
	switch (target.id()) {
	case LogicalTypeId::VARCHAR:
		// uuid to varchar
		return BoundCastInfo(&VectorCastHelpers::StringCast<hugeint_t, duckdb::CastFromUUID>);
	default:
		return TryVectorNullCast;
	}
}

} // namespace duckdb


namespace duckdb {

// ------- Helper functions for splitting string nested types  -------
static bool IsNull(const char *buf, idx_t start_pos, Vector &child, idx_t row_idx) {
	if (buf[start_pos] == 'N' && buf[start_pos + 1] == 'U' && buf[start_pos + 2] == 'L' && buf[start_pos + 3] == 'L') {
		FlatVector::SetNull(child, row_idx, true);
		return true;
	}
	return false;
}

inline static void SkipWhitespace(const char *buf, idx_t &pos, idx_t len) {
	while (pos < len && StringUtil::CharacterIsSpace(buf[pos])) {
		pos++;
	}
}

static bool SkipToCloseQuotes(idx_t &pos, const char *buf, idx_t &len) {
	char quote = buf[pos];
	pos++;

	while (pos < len) {
		if (buf[pos] == quote) {
			return true;
		}
		pos++;
	}
	return false;
}

static bool SkipToClose(idx_t &idx, const char *buf, idx_t &len, idx_t &lvl, char close_bracket) {
	idx++;

	while (idx < len) {
		if (buf[idx] == '"' || buf[idx] == '\'') {
			if (!SkipToCloseQuotes(idx, buf, len)) {
				return false;
			}
		} else if (buf[idx] == '{') {
			if (!SkipToClose(idx, buf, len, lvl, '}')) {
				return false;
			}
		} else if (buf[idx] == '[') {
			if (!SkipToClose(idx, buf, len, lvl, ']')) {
				return false;
			}
			lvl++;
		} else if (buf[idx] == close_bracket) {
			if (close_bracket == ']') {
				lvl--;
			}
			return true;
		}
		idx++;
	}
	return false;
}

static idx_t StringTrim(const char *buf, idx_t &start_pos, idx_t pos) {
	idx_t trailing_whitespace = 0;
	while (StringUtil::CharacterIsSpace(buf[pos - trailing_whitespace - 1])) {
		trailing_whitespace++;
	}
	if ((buf[start_pos] == '"' && buf[pos - trailing_whitespace - 1] == '"') ||
	    (buf[start_pos] == '\'' && buf[pos - trailing_whitespace - 1] == '\'')) {
		start_pos++;
		trailing_whitespace++;
	}
	return (pos - trailing_whitespace);
}

struct CountPartOperation {
	idx_t count = 0;

	bool HandleKey(const char *buf, idx_t start_pos, idx_t pos) {
		count++;
		return true;
	}
	void HandleValue(const char *buf, idx_t start_pos, idx_t pos) {
		count++;
	}
};

// ------- LIST SPLIT -------
struct SplitStringListOperation {
	SplitStringListOperation(string_t *child_data, idx_t &child_start, Vector &child)
	    : child_data(child_data), child_start(child_start), child(child) {
	}

	string_t *child_data;
	idx_t &child_start;
	Vector &child;

	void HandleValue(const char *buf, idx_t start_pos, idx_t pos) {
		if ((pos - start_pos) == 4 && IsNull(buf, start_pos, child, child_start)) {
			child_start++;
			return;
		}
		child_data[child_start] = StringVector::AddString(child, buf + start_pos, pos - start_pos);
		child_start++;
	}
};

template <class OP>
static bool SplitStringListInternal(const string_t &input, OP &state) {
	const char *buf = input.GetData();
	idx_t len = input.GetSize();
	idx_t lvl = 1;
	idx_t pos = 0;

	SkipWhitespace(buf, pos, len);
	if (pos == len || buf[pos] != '[') {
		return false;
	}

	SkipWhitespace(buf, ++pos, len);
	idx_t start_pos = pos;
	while (pos < len) {
		if (buf[pos] == '[') {
			if (!SkipToClose(pos, buf, len, ++lvl, ']')) {
				return false;
			}
		} else if ((buf[pos] == '"' || buf[pos] == '\'') && pos == start_pos) {
			SkipToCloseQuotes(pos, buf, len);
		} else if (buf[pos] == '{') {
			idx_t struct_lvl = 0;
			SkipToClose(pos, buf, len, struct_lvl, '}');
		} else if (buf[pos] == ',' || buf[pos] == ']') {
			idx_t trailing_whitespace = 0;
			while (StringUtil::CharacterIsSpace(buf[pos - trailing_whitespace - 1])) {
				trailing_whitespace++;
			}
			if (!(buf[pos] == ']' && start_pos == pos)) {
				state.HandleValue(buf, start_pos, pos - trailing_whitespace);
			} // else the list is empty
			if (buf[pos] == ']') {
				lvl--;
				break;
			}
			SkipWhitespace(buf, ++pos, len);
			start_pos = pos;
			continue;
		}
		pos++;
	}
	SkipWhitespace(buf, ++pos, len);
	return (pos == len && lvl == 0);
}

bool VectorStringToList::SplitStringList(const string_t &input, string_t *child_data, idx_t &child_start,
                                         Vector &child) {
	SplitStringListOperation state(child_data, child_start, child);
	return SplitStringListInternal<SplitStringListOperation>(input, state);
}

idx_t VectorStringToList::CountPartsList(const string_t &input) {
	CountPartOperation state;
	SplitStringListInternal<CountPartOperation>(input, state);
	return state.count;
}

// ------- MAP SPLIT -------
struct SplitStringMapOperation {
	SplitStringMapOperation(string_t *child_key_data, string_t *child_val_data, idx_t &child_start, Vector &varchar_key,
	                        Vector &varchar_val)
	    : child_key_data(child_key_data), child_val_data(child_val_data), child_start(child_start),
	      varchar_key(varchar_key), varchar_val(varchar_val) {
	}

	string_t *child_key_data;
	string_t *child_val_data;
	idx_t &child_start;
	Vector &varchar_key;
	Vector &varchar_val;

	bool HandleKey(const char *buf, idx_t start_pos, idx_t pos) {
		if ((pos - start_pos) == 4 && IsNull(buf, start_pos, varchar_key, child_start)) {
			FlatVector::SetNull(varchar_val, child_start, true);
			child_start++;
			return false;
		}
		child_key_data[child_start] = StringVector::AddString(varchar_key, buf + start_pos, pos - start_pos);
		return true;
	}

	void HandleValue(const char *buf, idx_t start_pos, idx_t pos) {
		if ((pos - start_pos) == 4 && IsNull(buf, start_pos, varchar_val, child_start)) {
			child_start++;
			return;
		}
		child_val_data[child_start] = StringVector::AddString(varchar_val, buf + start_pos, pos - start_pos);
		child_start++;
	}
};

template <class OP>
static bool FindKeyOrValueMap(const char *buf, idx_t len, idx_t &pos, OP &state, bool key) {
	auto start_pos = pos;
	idx_t lvl = 0;
	while (pos < len) {
		if (buf[pos] == '"' || buf[pos] == '\'') {
			SkipToCloseQuotes(pos, buf, len);
		} else if (buf[pos] == '{') {
			SkipToClose(pos, buf, len, lvl, '}');
		} else if (buf[pos] == '[') {
			SkipToClose(pos, buf, len, lvl, ']');
		} else if (key && buf[pos] == '=') {
			idx_t end_pos = StringTrim(buf, start_pos, pos);
			return state.HandleKey(buf, start_pos, end_pos); // put string in KEY_child_vector
		} else if (!key && (buf[pos] == ',' || buf[pos] == '}')) {
			idx_t end_pos = StringTrim(buf, start_pos, pos);
			state.HandleValue(buf, start_pos, end_pos); // put string in VALUE_child_vector
			return true;
		}
		pos++;
	}
	return false;
}

template <class OP>
static bool SplitStringMapInternal(const string_t &input, OP &state) {
	const char *buf = input.GetData();
	idx_t len = input.GetSize();
	idx_t pos = 0;

	SkipWhitespace(buf, pos, len);
	if (pos == len || buf[pos] != '{') {
		return false;
	}
	SkipWhitespace(buf, ++pos, len);
	if (pos == len) {
		return false;
	}
	if (buf[pos] == '}') {
		SkipWhitespace(buf, ++pos, len);
		return (pos == len);
	}
	while (pos < len) {
		if (!FindKeyOrValueMap(buf, len, pos, state, true)) {
			return false;
		}
		SkipWhitespace(buf, ++pos, len);
		if (!FindKeyOrValueMap(buf, len, pos, state, false)) {
			return false;
		}
		SkipWhitespace(buf, ++pos, len);
	}
	return true;
}

bool VectorStringToMap::SplitStringMap(const string_t &input, string_t *child_key_data, string_t *child_val_data,
                                       idx_t &child_start, Vector &varchar_key, Vector &varchar_val) {
	SplitStringMapOperation state(child_key_data, child_val_data, child_start, varchar_key, varchar_val);
	return SplitStringMapInternal<SplitStringMapOperation>(input, state);
}

idx_t VectorStringToMap::CountPartsMap(const string_t &input) {
	CountPartOperation state;
	SplitStringMapInternal<CountPartOperation>(input, state);
	return state.count;
}

// ------- STRUCT SPLIT -------
static bool FindKeyStruct(const char *buf, idx_t len, idx_t &pos) {
	while (pos < len) {
		if (buf[pos] == ':') {
			return true;
		}
		pos++;
	}
	return false;
}

static bool FindValueStruct(const char *buf, idx_t len, idx_t &pos, Vector &varchar_child, idx_t &row_idx,
                            ValidityMask *child_mask) {
	auto start_pos = pos;
	idx_t lvl = 0;
	while (pos < len) {
		if (buf[pos] == '"' || buf[pos] == '\'') {
			SkipToCloseQuotes(pos, buf, len);
		} else if (buf[pos] == '{') {
			SkipToClose(pos, buf, len, lvl, '}');
		} else if (buf[pos] == '[') {
			SkipToClose(pos, buf, len, lvl, ']');
		} else if (buf[pos] == ',' || buf[pos] == '}') {
			idx_t end_pos = StringTrim(buf, start_pos, pos);
			if ((end_pos - start_pos) == 4 && IsNull(buf, start_pos, varchar_child, row_idx)) {
				return true;
			}
			FlatVector::GetData<string_t>(varchar_child)[row_idx] =
			    StringVector::AddString(varchar_child, buf + start_pos, end_pos - start_pos);
			child_mask->SetValid(row_idx); // any child not set to valid will remain invalid
			return true;
		}
		pos++;
	}
	return false;
}

bool VectorStringToStruct::SplitStruct(string_t &input, vector<unique_ptr<Vector>> &varchar_vectors, idx_t &row_idx,
                                       string_map_t<idx_t> &child_names, vector<ValidityMask *> &child_masks) {
	const char *buf = input.GetData();
	idx_t len = input.GetSize();
	idx_t pos = 0;
	idx_t child_idx;

	SkipWhitespace(buf, pos, len);
	if (pos == len || buf[pos] != '{') {
		return false;
	}
	SkipWhitespace(buf, ++pos, len);
	if (buf[pos] == '}') {
		pos++;
	} else {
		while (pos < len) {
			auto key_start = pos;
			if (!FindKeyStruct(buf, len, pos)) {
				return false;
			}
			auto key_end = StringTrim(buf, key_start, pos);
			string_t found_key(buf + key_start, key_end - key_start);

			auto it = child_names.find(found_key);
			if (it == child_names.end()) {
				return false; // false key
			}
			child_idx = it->second;
			SkipWhitespace(buf, ++pos, len);
			if (!FindValueStruct(buf, len, pos, *varchar_vectors[child_idx], row_idx, child_masks[child_idx])) {
				return false;
			}
			SkipWhitespace(buf, ++pos, len);
		}
	}
	SkipWhitespace(buf, pos, len);
	return (pos == len);
}

} // namespace duckdb


namespace duckdb {

//! The target type determines the preferred implicit casts
static int64_t TargetTypeCost(const LogicalType &type) {
	switch (type.id()) {
	case LogicalTypeId::INTEGER:
		return 103;
	case LogicalTypeId::BIGINT:
		return 101;
	case LogicalTypeId::DOUBLE:
		return 102;
	case LogicalTypeId::HUGEINT:
		return 120;
	case LogicalTypeId::TIMESTAMP:
		return 120;
	case LogicalTypeId::VARCHAR:
		return 149;
	case LogicalTypeId::DECIMAL:
		return 104;
	case LogicalTypeId::STRUCT:
	case LogicalTypeId::MAP:
	case LogicalTypeId::LIST:
	case LogicalTypeId::UNION:
		return 160;
	default:
		return 110;
	}
}

static int64_t ImplicitCastTinyint(const LogicalType &to) {
	switch (to.id()) {
	case LogicalTypeId::SMALLINT:
	case LogicalTypeId::INTEGER:
	case LogicalTypeId::BIGINT:
	case LogicalTypeId::HUGEINT:
	case LogicalTypeId::FLOAT:
	case LogicalTypeId::DOUBLE:
	case LogicalTypeId::DECIMAL:
		return TargetTypeCost(to);
	default:
		return -1;
	}
}

static int64_t ImplicitCastSmallint(const LogicalType &to) {
	switch (to.id()) {
	case LogicalTypeId::INTEGER:
	case LogicalTypeId::BIGINT:
	case LogicalTypeId::HUGEINT:
	case LogicalTypeId::FLOAT:
	case LogicalTypeId::DOUBLE:
	case LogicalTypeId::DECIMAL:
		return TargetTypeCost(to);
	default:
		return -1;
	}
}

static int64_t ImplicitCastInteger(const LogicalType &to) {
	switch (to.id()) {
	case LogicalTypeId::BIGINT:
	case LogicalTypeId::HUGEINT:
	case LogicalTypeId::FLOAT:
	case LogicalTypeId::DOUBLE:
	case LogicalTypeId::DECIMAL:
		return TargetTypeCost(to);
	default:
		return -1;
	}
}

static int64_t ImplicitCastBigint(const LogicalType &to) {
	switch (to.id()) {
	case LogicalTypeId::FLOAT:
	case LogicalTypeId::DOUBLE:
	case LogicalTypeId::HUGEINT:
	case LogicalTypeId::DECIMAL:
		return TargetTypeCost(to);
	default:
		return -1;
	}
}

static int64_t ImplicitCastUTinyint(const LogicalType &to) {
	switch (to.id()) {
	case LogicalTypeId::USMALLINT:
	case LogicalTypeId::UINTEGER:
	case LogicalTypeId::UBIGINT:
	case LogicalTypeId::SMALLINT:
	case LogicalTypeId::INTEGER:
	case LogicalTypeId::BIGINT:
	case LogicalTypeId::HUGEINT:
	case LogicalTypeId::FLOAT:
	case LogicalTypeId::DOUBLE:
	case LogicalTypeId::DECIMAL:
		return TargetTypeCost(to);
	default:
		return -1;
	}
}

static int64_t ImplicitCastUSmallint(const LogicalType &to) {
	switch (to.id()) {
	case LogicalTypeId::UINTEGER:
	case LogicalTypeId::UBIGINT:
	case LogicalTypeId::INTEGER:
	case LogicalTypeId::BIGINT:
	case LogicalTypeId::HUGEINT:
	case LogicalTypeId::FLOAT:
	case LogicalTypeId::DOUBLE:
	case LogicalTypeId::DECIMAL:
		return TargetTypeCost(to);
	default:
		return -1;
	}
}

static int64_t ImplicitCastUInteger(const LogicalType &to) {
	switch (to.id()) {

	case LogicalTypeId::UBIGINT:
	case LogicalTypeId::BIGINT:
	case LogicalTypeId::HUGEINT:
	case LogicalTypeId::FLOAT:
	case LogicalTypeId::DOUBLE:
	case LogicalTypeId::DECIMAL:
		return TargetTypeCost(to);
	default:
		return -1;
	}
}

static int64_t ImplicitCastUBigint(const LogicalType &to) {
	switch (to.id()) {
	case LogicalTypeId::FLOAT:
	case LogicalTypeId::DOUBLE:
	case LogicalTypeId::HUGEINT:
	case LogicalTypeId::DECIMAL:
		return TargetTypeCost(to);
	default:
		return -1;
	}
}

static int64_t ImplicitCastFloat(const LogicalType &to) {
	switch (to.id()) {
	case LogicalTypeId::DOUBLE:
		return TargetTypeCost(to);
	default:
		return -1;
	}
}

static int64_t ImplicitCastDouble(const LogicalType &to) {
	switch (to.id()) {
	default:
		return -1;
	}
}

static int64_t ImplicitCastDecimal(const LogicalType &to) {
	switch (to.id()) {
	case LogicalTypeId::FLOAT:
	case LogicalTypeId::DOUBLE:
		return TargetTypeCost(to);
	default:
		return -1;
	}
}

static int64_t ImplicitCastHugeint(const LogicalType &to) {
	switch (to.id()) {
	case LogicalTypeId::FLOAT:
	case LogicalTypeId::DOUBLE:
	case LogicalTypeId::DECIMAL:
		return TargetTypeCost(to);
	default:
		return -1;
	}
}

static int64_t ImplicitCastDate(const LogicalType &to) {
	switch (to.id()) {
	case LogicalTypeId::TIMESTAMP:
		return TargetTypeCost(to);
	default:
		return -1;
	}
}

int64_t CastRules::ImplicitCast(const LogicalType &from, const LogicalType &to) {
	if (from.id() == LogicalTypeId::SQLNULL) {
		// NULL expression can be cast to anything
		return TargetTypeCost(to);
	}
	if (from.id() == LogicalTypeId::UNKNOWN) {
		// parameter expression can be cast to anything for no cost
		return 0;
	}
	if (to.id() == LogicalTypeId::ANY) {
		// anything can be cast to ANY type for (almost no) cost
		return 1;
	}
	if (from.GetAlias() != to.GetAlias()) {
		// if aliases are different, an implicit cast is not possible
		return -1;
	}
	if (from.id() == LogicalTypeId::LIST && to.id() == LogicalTypeId::LIST) {
		// Lists can be cast if their child types can be cast
		auto child_cost = ImplicitCast(ListType::GetChildType(from), ListType::GetChildType(to));
		if (child_cost >= 100) {
			// subtract one from the cost because we prefer LIST[X] -> LIST[VARCHAR] over LIST[X] -> VARCHAR
			child_cost--;
		}
		return child_cost;
	}
	if (from.id() == to.id()) {
		// arguments match: do nothing
		return 0;
	}
	if (from.id() == LogicalTypeId::BLOB && to.id() == LogicalTypeId::VARCHAR) {
		// Implicit cast not allowed from BLOB to VARCHAR
		return -1;
	}
	if (to.id() == LogicalTypeId::VARCHAR) {
		// everything can be cast to VARCHAR, but this cast has a high cost
		return TargetTypeCost(to);
	}

	if (from.id() == LogicalTypeId::UNION && to.id() == LogicalTypeId::UNION) {
		// Unions can be cast if the source tags are a subset of the target tags
		// in which case the most expensive cost is used
		int cost = -1;
		for (idx_t from_member_idx = 0; from_member_idx < UnionType::GetMemberCount(from); from_member_idx++) {
			auto &from_member_name = UnionType::GetMemberName(from, from_member_idx);

			bool found = false;
			for (idx_t to_member_idx = 0; to_member_idx < UnionType::GetMemberCount(to); to_member_idx++) {
				auto &to_member_name = UnionType::GetMemberName(to, to_member_idx);

				if (from_member_name == to_member_name) {
					auto &from_member_type = UnionType::GetMemberType(from, from_member_idx);
					auto &to_member_type = UnionType::GetMemberType(to, to_member_idx);

					int child_cost = ImplicitCast(from_member_type, to_member_type);
					if (child_cost > cost) {
						cost = child_cost;
					}
					found = true;
					break;
				}
			}
			if (!found) {
				return -1;
			}
		}
		return cost;
	}

	if (to.id() == LogicalTypeId::UNION) {
		// check that the union type is fully resolved.
		if (to.AuxInfo() == nullptr) {
			return -1;
		}
		// every type can be implicitly be cast to a union if the source type is a member of the union
		for (idx_t i = 0; i < UnionType::GetMemberCount(to); i++) {
			auto member = UnionType::GetMemberType(to, i);
			if (from == member) {
				return 0;
			}
		}
	}

	if ((from.id() == LogicalTypeId::TIMESTAMP_SEC || from.id() == LogicalTypeId::TIMESTAMP_MS ||
	     from.id() == LogicalTypeId::TIMESTAMP_NS) &&
	    to.id() == LogicalTypeId::TIMESTAMP) {
		//! Any timestamp type can be converted to the default (us) type at low cost
		return 101;
	}
	if ((to.id() == LogicalTypeId::TIMESTAMP_SEC || to.id() == LogicalTypeId::TIMESTAMP_MS ||
	     to.id() == LogicalTypeId::TIMESTAMP_NS) &&
	    from.id() == LogicalTypeId::TIMESTAMP) {
		//! Any timestamp type can be converted to the default (us) type at low cost
		return 100;
	}
	switch (from.id()) {
	case LogicalTypeId::TINYINT:
		return ImplicitCastTinyint(to);
	case LogicalTypeId::SMALLINT:
		return ImplicitCastSmallint(to);
	case LogicalTypeId::INTEGER:
		return ImplicitCastInteger(to);
	case LogicalTypeId::BIGINT:
		return ImplicitCastBigint(to);
	case LogicalTypeId::UTINYINT:
		return ImplicitCastUTinyint(to);
	case LogicalTypeId::USMALLINT:
		return ImplicitCastUSmallint(to);
	case LogicalTypeId::UINTEGER:
		return ImplicitCastUInteger(to);
	case LogicalTypeId::UBIGINT:
		return ImplicitCastUBigint(to);
	case LogicalTypeId::HUGEINT:
		return ImplicitCastHugeint(to);
	case LogicalTypeId::FLOAT:
		return ImplicitCastFloat(to);
	case LogicalTypeId::DOUBLE:
		return ImplicitCastDouble(to);
	case LogicalTypeId::DATE:
		return ImplicitCastDate(to);
	case LogicalTypeId::DECIMAL:
		return ImplicitCastDecimal(to);
	default:
		return -1;
	}
}

} // namespace duckdb





namespace duckdb {

typedef CompressionFunction (*get_compression_function_t)(PhysicalType type);
typedef bool (*compression_supports_type_t)(PhysicalType type);

struct DefaultCompressionMethod {
	CompressionType type;
	get_compression_function_t get_function;
	compression_supports_type_t supports_type;
};

static DefaultCompressionMethod internal_compression_methods[] = {
    {CompressionType::COMPRESSION_CONSTANT, ConstantFun::GetFunction, ConstantFun::TypeIsSupported},
    {CompressionType::COMPRESSION_UNCOMPRESSED, UncompressedFun::GetFunction, UncompressedFun::TypeIsSupported},
    {CompressionType::COMPRESSION_RLE, RLEFun::GetFunction, RLEFun::TypeIsSupported},
    {CompressionType::COMPRESSION_BITPACKING, BitpackingFun::GetFunction, BitpackingFun::TypeIsSupported},
    {CompressionType::COMPRESSION_DICTIONARY, DictionaryCompressionFun::GetFunction,
     DictionaryCompressionFun::TypeIsSupported},
    {CompressionType::COMPRESSION_CHIMP, ChimpCompressionFun::GetFunction, ChimpCompressionFun::TypeIsSupported},
    {CompressionType::COMPRESSION_PATAS, PatasCompressionFun::GetFunction, PatasCompressionFun::TypeIsSupported},
    {CompressionType::COMPRESSION_FSST, FSSTFun::GetFunction, FSSTFun::TypeIsSupported},
    {CompressionType::COMPRESSION_AUTO, nullptr, nullptr}};

static optional_ptr<CompressionFunction> FindCompressionFunction(CompressionFunctionSet &set, CompressionType type,
                                                                 PhysicalType data_type) {
	auto &functions = set.functions;
	auto comp_entry = functions.find(type);
	if (comp_entry != functions.end()) {
		auto &type_functions = comp_entry->second;
		auto type_entry = type_functions.find(data_type);
		if (type_entry != type_functions.end()) {
			return &type_entry->second;
		}
	}
	return nullptr;
}

static optional_ptr<CompressionFunction> LoadCompressionFunction(CompressionFunctionSet &set, CompressionType type,
                                                                 PhysicalType data_type) {
	for (idx_t index = 0; internal_compression_methods[index].get_function; index++) {
		const auto &method = internal_compression_methods[index];
		if (method.type == type) {
			// found the correct compression type
			if (!method.supports_type(data_type)) {
				// but it does not support this data type: bail out
				return nullptr;
			}
			// the type is supported: create the function and insert it into the set
			auto function = method.get_function(data_type);
			set.functions[type].insert(make_pair(data_type, function));
			return FindCompressionFunction(set, type, data_type);
		}
	}
	throw InternalException("Unsupported compression function type");
}

static void TryLoadCompression(DBConfig &config, vector<reference<CompressionFunction>> &result, CompressionType type,
                               PhysicalType data_type) {
	auto function = config.GetCompressionFunction(type, data_type);
	if (!function) {
		return;
	}
	result.push_back(*function);
}

vector<reference<CompressionFunction>> DBConfig::GetCompressionFunctions(PhysicalType data_type) {
	vector<reference<CompressionFunction>> result;
	TryLoadCompression(*this, result, CompressionType::COMPRESSION_UNCOMPRESSED, data_type);
	TryLoadCompression(*this, result, CompressionType::COMPRESSION_RLE, data_type);
	TryLoadCompression(*this, result, CompressionType::COMPRESSION_BITPACKING, data_type);
	TryLoadCompression(*this, result, CompressionType::COMPRESSION_DICTIONARY, data_type);
	TryLoadCompression(*this, result, CompressionType::COMPRESSION_CHIMP, data_type);
	TryLoadCompression(*this, result, CompressionType::COMPRESSION_PATAS, data_type);
	TryLoadCompression(*this, result, CompressionType::COMPRESSION_FSST, data_type);
	return result;
}

optional_ptr<CompressionFunction> DBConfig::GetCompressionFunction(CompressionType type, PhysicalType data_type) {
	lock_guard<mutex> l(compression_functions->lock);
	// check if the function is already loaded
	auto function = FindCompressionFunction(*compression_functions, type, data_type);
	if (function) {
		return function;
	}
	// else load the function
	return LoadCompressionFunction(*compression_functions, type, data_type);
}

} // namespace duckdb










namespace duckdb {

FunctionData::~FunctionData() {
}

bool FunctionData::Equals(const FunctionData *left, const FunctionData *right) {
	if (left == right) {
		return true;
	}
	if (!left || !right) {
		return false;
	}
	return left->Equals(*right);
}

TableFunctionData::~TableFunctionData() {
}

unique_ptr<FunctionData> TableFunctionData::Copy() const {
	throw InternalException("Copy not supported for TableFunctionData");
}

bool TableFunctionData::Equals(const FunctionData &other) const {
	return false;
}

Function::Function(string name_p) : name(std::move(name_p)) {
}
Function::~Function() {
}

SimpleFunction::SimpleFunction(string name_p, vector<LogicalType> arguments_p, LogicalType varargs_p)
    : Function(std::move(name_p)), arguments(std::move(arguments_p)), varargs(std::move(varargs_p)) {
}

SimpleFunction::~SimpleFunction() {
}

string SimpleFunction::ToString() const {
	return Function::CallToString(name, arguments);
}

bool SimpleFunction::HasVarArgs() const {
	return varargs.id() != LogicalTypeId::INVALID;
}

SimpleNamedParameterFunction::SimpleNamedParameterFunction(string name_p, vector<LogicalType> arguments_p,
                                                           LogicalType varargs_p)
    : SimpleFunction(std::move(name_p), std::move(arguments_p), std::move(varargs_p)) {
}

SimpleNamedParameterFunction::~SimpleNamedParameterFunction() {
}

string SimpleNamedParameterFunction::ToString() const {
	return Function::CallToString(name, arguments, named_parameters);
}

bool SimpleNamedParameterFunction::HasNamedParameters() const {
	return !named_parameters.empty();
}

BaseScalarFunction::BaseScalarFunction(string name_p, vector<LogicalType> arguments_p, LogicalType return_type_p,
                                       FunctionSideEffects side_effects, LogicalType varargs_p,
                                       FunctionNullHandling null_handling)
    : SimpleFunction(std::move(name_p), std::move(arguments_p), std::move(varargs_p)),
      return_type(std::move(return_type_p)), side_effects(side_effects), null_handling(null_handling) {
}

BaseScalarFunction::~BaseScalarFunction() {
}

string BaseScalarFunction::ToString() const {
	return Function::CallToString(name, arguments, return_type);
}

// add your initializer for new functions here
void BuiltinFunctions::Initialize() {
	RegisterTableScanFunctions();
	RegisterSQLiteFunctions();
	RegisterReadFunctions();
	RegisterTableFunctions();
	RegisterArrowFunctions();

	RegisterDistributiveAggregates();

	RegisterGenericFunctions();
	RegisterOperators();
	RegisterSequenceFunctions();
	RegisterStringFunctions();
	RegisterNestedFunctions();

	RegisterPragmaFunctions();

	// initialize collations
	AddCollation("nocase", LowerFun::GetFunction(), true);
	AddCollation("noaccent", StripAccentsFun::GetFunction());
	AddCollation("nfc", NFCNormalizeFun::GetFunction());
}

hash_t BaseScalarFunction::Hash() const {
	hash_t hash = return_type.Hash();
	for (auto &arg : arguments) {
		duckdb::CombineHash(hash, arg.Hash());
	}
	return hash;
}

string Function::CallToString(const string &name, const vector<LogicalType> &arguments) {
	string result = name + "(";
	result += StringUtil::Join(arguments, arguments.size(), ", ",
	                           [](const LogicalType &argument) { return argument.ToString(); });
	return result + ")";
}

string Function::CallToString(const string &name, const vector<LogicalType> &arguments,
                              const LogicalType &return_type) {
	string result = CallToString(name, arguments);
	result += " -> " + return_type.ToString();
	return result;
}

string Function::CallToString(const string &name, const vector<LogicalType> &arguments,
                              const named_parameter_type_map_t &named_parameters) {
	vector<string> input_arguments;
	input_arguments.reserve(arguments.size() + named_parameters.size());
	for (auto &arg : arguments) {
		input_arguments.push_back(arg.ToString());
	}
	for (auto &kv : named_parameters) {
		input_arguments.push_back(StringUtil::Format("%s : %s", kv.first, kv.second.ToString()));
	}
	return StringUtil::Format("%s(%s)", name, StringUtil::Join(input_arguments, ", "));
}

void Function::EraseArgument(SimpleFunction &bound_function, vector<unique_ptr<Expression>> &arguments,
                             idx_t argument_index) {
	if (bound_function.original_arguments.empty()) {
		bound_function.original_arguments = bound_function.arguments;
	}
	D_ASSERT(arguments.size() == bound_function.arguments.size());
	D_ASSERT(argument_index < arguments.size());
	arguments.erase(arguments.begin() + argument_index);
	bound_function.arguments.erase(bound_function.arguments.begin() + argument_index);
}

} // namespace duckdb














namespace duckdb {

FunctionBinder::FunctionBinder(ClientContext &context) : context(context) {
}

int64_t FunctionBinder::BindVarArgsFunctionCost(const SimpleFunction &func, const vector<LogicalType> &arguments) {
	if (arguments.size() < func.arguments.size()) {
		// not enough arguments to fulfill the non-vararg part of the function
		return -1;
	}
	int64_t cost = 0;
	for (idx_t i = 0; i < arguments.size(); i++) {
		LogicalType arg_type = i < func.arguments.size() ? func.arguments[i] : func.varargs;
		if (arguments[i] == arg_type) {
			// arguments match: do nothing
			continue;
		}
		int64_t cast_cost = CastFunctionSet::Get(context).ImplicitCastCost(arguments[i], arg_type);
		if (cast_cost >= 0) {
			// we can implicitly cast, add the cost to the total cost
			cost += cast_cost;
		} else {
			// we can't implicitly cast: throw an error
			return -1;
		}
	}
	return cost;
}

int64_t FunctionBinder::BindFunctionCost(const SimpleFunction &func, const vector<LogicalType> &arguments) {
	if (func.HasVarArgs()) {
		// special case varargs function
		return BindVarArgsFunctionCost(func, arguments);
	}
	if (func.arguments.size() != arguments.size()) {
		// invalid argument count: check the next function
		return -1;
	}
	int64_t cost = 0;
	for (idx_t i = 0; i < arguments.size(); i++) {
		int64_t cast_cost = CastFunctionSet::Get(context).ImplicitCastCost(arguments[i], func.arguments[i]);
		if (cast_cost >= 0) {
			// we can implicitly cast, add the cost to the total cost
			cost += cast_cost;
		} else {
			// we can't implicitly cast: throw an error
			return -1;
		}
	}
	return cost;
}

template <class T>
vector<idx_t> FunctionBinder::BindFunctionsFromArguments(const string &name, FunctionSet<T> &functions,
                                                         const vector<LogicalType> &arguments, string &error) {
	idx_t best_function = DConstants::INVALID_INDEX;
	int64_t lowest_cost = NumericLimits<int64_t>::Maximum();
	vector<idx_t> candidate_functions;
	for (idx_t f_idx = 0; f_idx < functions.functions.size(); f_idx++) {
		auto &func = functions.functions[f_idx];
		// check the arguments of the function
		int64_t cost = BindFunctionCost(func, arguments);
		if (cost < 0) {
			// auto casting was not possible
			continue;
		}
		if (cost == lowest_cost) {
			candidate_functions.push_back(f_idx);
			continue;
		}
		if (cost > lowest_cost) {
			continue;
		}
		candidate_functions.clear();
		lowest_cost = cost;
		best_function = f_idx;
	}
	if (best_function == DConstants::INVALID_INDEX) {
		// no matching function was found, throw an error
		string call_str = Function::CallToString(name, arguments);
		string candidate_str = "";
		for (auto &f : functions.functions) {
			candidate_str += "\t" + f.ToString() + "\n";
		}
		error = StringUtil::Format("No function matches the given name and argument types '%s'. You might need to add "
		                           "explicit type casts.\n\tCandidate functions:\n%s",
		                           call_str, candidate_str);
		return candidate_functions;
	}
	candidate_functions.push_back(best_function);
	return candidate_functions;
}

template <class T>
idx_t FunctionBinder::MultipleCandidateException(const string &name, FunctionSet<T> &functions,
                                                 vector<idx_t> &candidate_functions,
                                                 const vector<LogicalType> &arguments, string &error) {
	D_ASSERT(functions.functions.size() > 1);
	// there are multiple possible function definitions
	// throw an exception explaining which overloads are there
	string call_str = Function::CallToString(name, arguments);
	string candidate_str = "";
	for (auto &conf : candidate_functions) {
		T f = functions.GetFunctionByOffset(conf);
		candidate_str += "\t" + f.ToString() + "\n";
	}
	error = StringUtil::Format("Could not choose a best candidate function for the function call \"%s\". In order to "
	                           "select one, please add explicit type casts.\n\tCandidate functions:\n%s",
	                           call_str, candidate_str);
	return DConstants::INVALID_INDEX;
}

template <class T>
idx_t FunctionBinder::BindFunctionFromArguments(const string &name, FunctionSet<T> &functions,
                                                const vector<LogicalType> &arguments, string &error) {
	auto candidate_functions = BindFunctionsFromArguments<T>(name, functions, arguments, error);
	if (candidate_functions.empty()) {
		// no candidates
		return DConstants::INVALID_INDEX;
	}
	if (candidate_functions.size() > 1) {
		// multiple candidates, check if there are any unknown arguments
		bool has_parameters = false;
		for (auto &arg_type : arguments) {
			if (arg_type.id() == LogicalTypeId::UNKNOWN) {
				//! there are! we could not resolve parameters in this case
				throw ParameterNotResolvedException();
			}
		}
		if (!has_parameters) {
			return MultipleCandidateException(name, functions, candidate_functions, arguments, error);
		}
	}
	return candidate_functions[0];
}

idx_t FunctionBinder::BindFunction(const string &name, ScalarFunctionSet &functions,
                                   const vector<LogicalType> &arguments, string &error) {
	return BindFunctionFromArguments(name, functions, arguments, error);
}

idx_t FunctionBinder::BindFunction(const string &name, AggregateFunctionSet &functions,
                                   const vector<LogicalType> &arguments, string &error) {
	return BindFunctionFromArguments(name, functions, arguments, error);
}

idx_t FunctionBinder::BindFunction(const string &name, TableFunctionSet &functions,
                                   const vector<LogicalType> &arguments, string &error) {
	return BindFunctionFromArguments(name, functions, arguments, error);
}

idx_t FunctionBinder::BindFunction(const string &name, PragmaFunctionSet &functions, PragmaInfo &info, string &error) {
	vector<LogicalType> types;
	for (auto &value : info.parameters) {
		types.push_back(value.type());
	}
	idx_t entry = BindFunctionFromArguments(name, functions, types, error);
	if (entry == DConstants::INVALID_INDEX) {
		throw BinderException(error);
	}
	auto candidate_function = functions.GetFunctionByOffset(entry);
	// cast the input parameters
	for (idx_t i = 0; i < info.parameters.size(); i++) {
		auto target_type =
		    i < candidate_function.arguments.size() ? candidate_function.arguments[i] : candidate_function.varargs;
		info.parameters[i] = info.parameters[i].CastAs(context, target_type);
	}
	return entry;
}

vector<LogicalType> FunctionBinder::GetLogicalTypesFromExpressions(vector<unique_ptr<Expression>> &arguments) {
	vector<LogicalType> types;
	types.reserve(arguments.size());
	for (auto &argument : arguments) {
		types.push_back(argument->return_type);
	}
	return types;
}

idx_t FunctionBinder::BindFunction(const string &name, ScalarFunctionSet &functions,
                                   vector<unique_ptr<Expression>> &arguments, string &error) {
	auto types = GetLogicalTypesFromExpressions(arguments);
	return BindFunction(name, functions, types, error);
}

idx_t FunctionBinder::BindFunction(const string &name, AggregateFunctionSet &functions,
                                   vector<unique_ptr<Expression>> &arguments, string &error) {
	auto types = GetLogicalTypesFromExpressions(arguments);
	return BindFunction(name, functions, types, error);
}

idx_t FunctionBinder::BindFunction(const string &name, TableFunctionSet &functions,
                                   vector<unique_ptr<Expression>> &arguments, string &error) {
	auto types = GetLogicalTypesFromExpressions(arguments);
	return BindFunction(name, functions, types, error);
}

enum class LogicalTypeComparisonResult { IDENTICAL_TYPE, TARGET_IS_ANY, DIFFERENT_TYPES };

LogicalTypeComparisonResult RequiresCast(const LogicalType &source_type, const LogicalType &target_type) {
	if (target_type.id() == LogicalTypeId::ANY) {
		return LogicalTypeComparisonResult::TARGET_IS_ANY;
	}
	if (source_type == target_type) {
		return LogicalTypeComparisonResult::IDENTICAL_TYPE;
	}
	if (source_type.id() == LogicalTypeId::LIST && target_type.id() == LogicalTypeId::LIST) {
		return RequiresCast(ListType::GetChildType(source_type), ListType::GetChildType(target_type));
	}
	return LogicalTypeComparisonResult::DIFFERENT_TYPES;
}

void FunctionBinder::CastToFunctionArguments(SimpleFunction &function, vector<unique_ptr<Expression>> &children) {
	for (idx_t i = 0; i < children.size(); i++) {
		auto target_type = i < function.arguments.size() ? function.arguments[i] : function.varargs;
		target_type.Verify();
		// don't cast lambda children, they get removed anyways
		if (children[i]->return_type.id() == LogicalTypeId::LAMBDA) {
			continue;
		}
		// check if the type of child matches the type of function argument
		// if not we need to add a cast
		auto cast_result = RequiresCast(children[i]->return_type, target_type);
		// except for one special case: if the function accepts ANY argument
		// in that case we don't add a cast
		if (cast_result == LogicalTypeComparisonResult::DIFFERENT_TYPES) {
			children[i] = BoundCastExpression::AddCastToType(context, std::move(children[i]), target_type);
		}
	}
}

unique_ptr<Expression> FunctionBinder::BindScalarFunction(const string &schema, const string &name,
                                                          vector<unique_ptr<Expression>> children, string &error,
                                                          bool is_operator, Binder *binder) {
	// bind the function
	auto &function =
	    Catalog::GetSystemCatalog(context).GetEntry(context, CatalogType::SCALAR_FUNCTION_ENTRY, schema, name);
	D_ASSERT(function.type == CatalogType::SCALAR_FUNCTION_ENTRY);
	return BindScalarFunction(function.Cast<ScalarFunctionCatalogEntry>(), std::move(children), error, is_operator,
	                          binder);
}

unique_ptr<Expression> FunctionBinder::BindScalarFunction(ScalarFunctionCatalogEntry &func,
                                                          vector<unique_ptr<Expression>> children, string &error,
                                                          bool is_operator, Binder *binder) {
	// bind the function
	idx_t best_function = BindFunction(func.name, func.functions, children, error);
	if (best_function == DConstants::INVALID_INDEX) {
		return nullptr;
	}

	// found a matching function!
	auto bound_function = func.functions.GetFunctionByOffset(best_function);

	if (bound_function.null_handling == FunctionNullHandling::DEFAULT_NULL_HANDLING) {
		for (auto &child : children) {
			if (child->return_type == LogicalTypeId::SQLNULL) {
				return make_uniq<BoundConstantExpression>(Value(LogicalType::SQLNULL));
			}
		}
	}
	return BindScalarFunction(bound_function, std::move(children), is_operator);
}

unique_ptr<BoundFunctionExpression> FunctionBinder::BindScalarFunction(ScalarFunction bound_function,
                                                                       vector<unique_ptr<Expression>> children,
                                                                       bool is_operator) {
	unique_ptr<FunctionData> bind_info;
	if (bound_function.bind) {
		bind_info = bound_function.bind(context, bound_function, children);
	}
	// check if we need to add casts to the children
	CastToFunctionArguments(bound_function, children);

	// now create the function
	auto return_type = bound_function.return_type;
	return make_uniq<BoundFunctionExpression>(std::move(return_type), std::move(bound_function), std::move(children),
	                                          std::move(bind_info), is_operator);
}

unique_ptr<BoundAggregateExpression> FunctionBinder::BindAggregateFunction(AggregateFunction bound_function,
                                                                           vector<unique_ptr<Expression>> children,
                                                                           unique_ptr<Expression> filter,
                                                                           AggregateType aggr_type) {
	unique_ptr<FunctionData> bind_info;
	if (bound_function.bind) {
		bind_info = bound_function.bind(context, bound_function, children);
		// we may have lost some arguments in the bind
		children.resize(MinValue(bound_function.arguments.size(), children.size()));
	}

	// check if we need to add casts to the children
	CastToFunctionArguments(bound_function, children);

	return make_uniq<BoundAggregateExpression>(std::move(bound_function), std::move(children), std::move(filter),
	                                           std::move(bind_info), aggr_type);
}

} // namespace duckdb



namespace duckdb {

ScalarFunctionSet::ScalarFunctionSet() : FunctionSet("") {
}

ScalarFunctionSet::ScalarFunctionSet(string name) : FunctionSet(std::move(name)) {
}

ScalarFunctionSet::ScalarFunctionSet(ScalarFunction fun) : FunctionSet(std::move(fun.name)) {
	functions.push_back(std::move(fun));
}

ScalarFunction ScalarFunctionSet::GetFunctionByArguments(ClientContext &context, const vector<LogicalType> &arguments) {
	string error;
	FunctionBinder binder(context);
	idx_t index = binder.BindFunction(name, *this, arguments, error);
	if (index == DConstants::INVALID_INDEX) {
		throw InternalException("Failed to find function %s(%s)\n%s", name, StringUtil::ToString(arguments, ","),
		                        error);
	}
	return GetFunctionByOffset(index);
}

AggregateFunctionSet::AggregateFunctionSet() : FunctionSet("") {
}

AggregateFunctionSet::AggregateFunctionSet(string name) : FunctionSet(std::move(name)) {
}

AggregateFunctionSet::AggregateFunctionSet(AggregateFunction fun) : FunctionSet(std::move(fun.name)) {
	functions.push_back(std::move(fun));
}

AggregateFunction AggregateFunctionSet::GetFunctionByArguments(ClientContext &context,
                                                               const vector<LogicalType> &arguments) {
	string error;
	FunctionBinder binder(context);
	idx_t index = binder.BindFunction(name, *this, arguments, error);
	if (index == DConstants::INVALID_INDEX) {
		// check if the arguments are a prefix of any of the arguments
		// this is used for functions such as quantile or string_agg that delete part of their arguments during bind
		// FIXME: we should come up with a better solution here
		for (auto &func : functions) {
			if (arguments.size() >= func.arguments.size()) {
				continue;
			}
			bool is_prefix = true;
			for (idx_t k = 0; k < arguments.size(); k++) {
				if (arguments[k] != func.arguments[k]) {
					is_prefix = false;
					break;
				}
			}
			if (is_prefix) {
				return func;
			}
		}
		throw InternalException("Failed to find function %s(%s)\n%s", name, StringUtil::ToString(arguments, ","),
		                        error);
	}
	return GetFunctionByOffset(index);
}

TableFunctionSet::TableFunctionSet(string name) : FunctionSet(std::move(name)) {
}

TableFunctionSet::TableFunctionSet(TableFunction fun) : FunctionSet(std::move(fun.name)) {
	functions.push_back(std::move(fun));
}

TableFunction TableFunctionSet::GetFunctionByArguments(ClientContext &context, const vector<LogicalType> &arguments) {
	string error;
	FunctionBinder binder(context);
	idx_t index = binder.BindFunction(name, *this, arguments, error);
	if (index == DConstants::INVALID_INDEX) {
		throw InternalException("Failed to find function %s(%s)\n%s", name, StringUtil::ToString(arguments, ","),
		                        error);
	}
	return GetFunctionByOffset(index);
}

PragmaFunctionSet::PragmaFunctionSet(string name) : FunctionSet(std::move(name)) {
}

PragmaFunctionSet::PragmaFunctionSet(PragmaFunction fun) : FunctionSet(std::move(fun.name)) {
	functions.push_back(std::move(fun));
}

} // namespace duckdb












namespace duckdb {

// MacroFunction::MacroFunction(unique_ptr<ParsedExpression> expression) : expression(std::move(expression)) {}

MacroFunction::MacroFunction(MacroType type) : type(type) {
}

string MacroFunction::ValidateArguments(MacroFunction &macro_def, const string &name, FunctionExpression &function_expr,
                                        vector<unique_ptr<ParsedExpression>> &positionals,
                                        unordered_map<string, unique_ptr<ParsedExpression>> &defaults) {

	// separate positional and default arguments
	for (auto &arg : function_expr.children) {
		if (!arg->alias.empty()) {
			// default argument
			if (!macro_def.default_parameters.count(arg->alias)) {
				return StringUtil::Format("Macro %s does not have default parameter %s!", name, arg->alias);
			} else if (defaults.count(arg->alias)) {
				return StringUtil::Format("Duplicate default parameters %s!", arg->alias);
			}
			defaults[arg->alias] = std::move(arg);
		} else if (!defaults.empty()) {
			return "Positional parameters cannot come after parameters with a default value!";
		} else {
			// positional argument
			positionals.push_back(std::move(arg));
		}
	}

	// validate if the right number of arguments was supplied
	string error;
	auto &parameters = macro_def.parameters;
	if (parameters.size() != positionals.size()) {
		error = StringUtil::Format(
		    "Macro function '%s(%s)' requires ", name,
		    StringUtil::Join(parameters, parameters.size(), ", ", [](const unique_ptr<ParsedExpression> &p) {
			    return (p->Cast<ColumnRefExpression>()).column_names[0];
		    }));
		error += parameters.size() == 1 ? "a single positional argument"
		                                : StringUtil::Format("%i positional arguments", parameters.size());
		error += ", but ";
		error += positionals.size() == 1 ? "a single positional argument was"
		                                 : StringUtil::Format("%i positional arguments were", positionals.size());
		error += " provided.";
		return error;
	}

	// Add the default values for parameters that have defaults, that were not explicitly assigned to
	for (auto it = macro_def.default_parameters.begin(); it != macro_def.default_parameters.end(); it++) {
		auto &parameter_name = it->first;
		auto &parameter_default = it->second;
		if (!defaults.count(parameter_name)) {
			// This parameter was not set yet, set it with the default value
			defaults[parameter_name] = parameter_default->Copy();
		}
	}

	return error;
}

void MacroFunction::CopyProperties(MacroFunction &other) const {
	other.type = type;
	for (auto &param : parameters) {
		other.parameters.push_back(param->Copy());
	}
	for (auto &kv : default_parameters) {
		other.default_parameters[kv.first] = kv.second->Copy();
	}
}

string MacroFunction::ToSQL(const string &schema, const string &name) const {
	vector<string> param_strings;
	for (auto &param : parameters) {
		param_strings.push_back(param->ToString());
	}
	for (auto &named_param : default_parameters) {
		param_strings.push_back(StringUtil::Format("%s := %s", named_param.first, named_param.second->ToString()));
	}

	return StringUtil::Format("CREATE MACRO %s.%s(%s) AS ", schema, name, StringUtil::Join(param_strings, ", "));
}

void MacroFunction::Serialize(Serializer &main_serializer) const {
	FieldWriter writer(main_serializer);
	writer.WriteField(type);
	writer.WriteSerializableList(parameters);
	writer.WriteField<uint32_t>((uint32_t)default_parameters.size());
	auto &serializer = writer.GetSerializer();
	for (auto &kv : default_parameters) {
		serializer.WriteString(kv.first);
		kv.second->Serialize(serializer);
	}
	SerializeInternal(writer);
	writer.Finalize();
}

unique_ptr<MacroFunction> MacroFunction::Deserialize(Deserializer &main_source) {
	FieldReader reader(main_source);
	auto type = reader.ReadRequired<MacroType>();
	auto parameters = reader.ReadRequiredSerializableList<ParsedExpression>();
	auto default_param_count = reader.ReadRequired<uint32_t>();
	unordered_map<string, unique_ptr<ParsedExpression>> default_parameters;
	auto &source = reader.GetSource();
	for (idx_t i = 0; i < default_param_count; i++) {
		auto name = source.Read<string>();
		default_parameters[name] = ParsedExpression::Deserialize(source);
	}
	unique_ptr<MacroFunction> result;
	switch (type) {
	case MacroType::SCALAR_MACRO:
		result = ScalarMacroFunction::Deserialize(reader);
		break;
	case MacroType::TABLE_MACRO:
		result = TableMacroFunction::Deserialize(reader);
		break;
	default:
		throw InternalException("Cannot deserialize macro type");
	}
	result->parameters = std::move(parameters);
	result->default_parameters = std::move(default_parameters);
	reader.Finalize();
	return result;
}

} // namespace duckdb













#include <cctype>

namespace duckdb {

static void PragmaEnableProfilingStatement(ClientContext &context, const FunctionParameters &parameters) {
	auto &config = ClientConfig::GetConfig(context);
	config.enable_profiler = true;
	config.emit_profiler_output = true;
}

void RegisterEnableProfiling(BuiltinFunctions &set) {
	PragmaFunctionSet functions("");
	functions.AddFunction(PragmaFunction::PragmaStatement(string(), PragmaEnableProfilingStatement));

	set.AddFunction("enable_profile", functions);
	set.AddFunction("enable_profiling", functions);
}

static void PragmaDisableProfiling(ClientContext &context, const FunctionParameters &parameters) {
	auto &config = ClientConfig::GetConfig(context);
	config.enable_profiler = false;
}

static void PragmaEnableProgressBar(ClientContext &context, const FunctionParameters &parameters) {
	ClientConfig::GetConfig(context).enable_progress_bar = true;
}

static void PragmaDisableProgressBar(ClientContext &context, const FunctionParameters &parameters) {
	ClientConfig::GetConfig(context).enable_progress_bar = false;
}

static void PragmaEnablePrintProgressBar(ClientContext &context, const FunctionParameters &parameters) {
	ClientConfig::GetConfig(context).print_progress_bar = true;
}

static void PragmaDisablePrintProgressBar(ClientContext &context, const FunctionParameters &parameters) {
	ClientConfig::GetConfig(context).print_progress_bar = false;
}

static void PragmaEnableVerification(ClientContext &context, const FunctionParameters &parameters) {
	ClientConfig::GetConfig(context).query_verification_enabled = true;
	ClientConfig::GetConfig(context).verify_serializer = true;
}

static void PragmaDisableVerification(ClientContext &context, const FunctionParameters &parameters) {
	ClientConfig::GetConfig(context).query_verification_enabled = false;
	ClientConfig::GetConfig(context).verify_serializer = false;
}

static void PragmaVerifySerializer(ClientContext &context, const FunctionParameters &parameters) {
	ClientConfig::GetConfig(context).verify_serializer = true;
}

static void PragmaDisableVerifySerializer(ClientContext &context, const FunctionParameters &parameters) {
	ClientConfig::GetConfig(context).verify_serializer = false;
}

static void PragmaEnableExternalVerification(ClientContext &context, const FunctionParameters &parameters) {
	ClientConfig::GetConfig(context).verify_external = true;
}

static void PragmaDisableExternalVerification(ClientContext &context, const FunctionParameters &parameters) {
	ClientConfig::GetConfig(context).verify_external = false;
}

static void PragmaEnableForceParallelism(ClientContext &context, const FunctionParameters &parameters) {
	ClientConfig::GetConfig(context).verify_parallelism = true;
}

static void PragmaEnableForceIndexJoin(ClientContext &context, const FunctionParameters &parameters) {
	ClientConfig::GetConfig(context).force_index_join = true;
}

static void PragmaForceCheckpoint(ClientContext &context, const FunctionParameters &parameters) {
	DBConfig::GetConfig(context).options.force_checkpoint = true;
}

static void PragmaDisableForceParallelism(ClientContext &context, const FunctionParameters &parameters) {
	ClientConfig::GetConfig(context).verify_parallelism = false;
}

static void PragmaEnableObjectCache(ClientContext &context, const FunctionParameters &parameters) {
	DBConfig::GetConfig(context).options.object_cache_enable = true;
}

static void PragmaDisableObjectCache(ClientContext &context, const FunctionParameters &parameters) {
	DBConfig::GetConfig(context).options.object_cache_enable = false;
}

static void PragmaEnableCheckpointOnShutdown(ClientContext &context, const FunctionParameters &parameters) {
	DBConfig::GetConfig(context).options.checkpoint_on_shutdown = true;
}

static void PragmaDisableCheckpointOnShutdown(ClientContext &context, const FunctionParameters &parameters) {
	DBConfig::GetConfig(context).options.checkpoint_on_shutdown = false;
}

static void PragmaEnableOptimizer(ClientContext &context, const FunctionParameters &parameters) {
	ClientConfig::GetConfig(context).enable_optimizer = true;
}

static void PragmaDisableOptimizer(ClientContext &context, const FunctionParameters &parameters) {
	ClientConfig::GetConfig(context).enable_optimizer = false;
}

void PragmaFunctions::RegisterFunction(BuiltinFunctions &set) {
	RegisterEnableProfiling(set);

	set.AddFunction(PragmaFunction::PragmaStatement("disable_profile", PragmaDisableProfiling));
	set.AddFunction(PragmaFunction::PragmaStatement("disable_profiling", PragmaDisableProfiling));

	set.AddFunction(PragmaFunction::PragmaStatement("enable_verification", PragmaEnableVerification));
	set.AddFunction(PragmaFunction::PragmaStatement("disable_verification", PragmaDisableVerification));

	set.AddFunction(PragmaFunction::PragmaStatement("verify_external", PragmaEnableExternalVerification));
	set.AddFunction(PragmaFunction::PragmaStatement("disable_verify_external", PragmaDisableExternalVerification));

	set.AddFunction(PragmaFunction::PragmaStatement("verify_serializer", PragmaVerifySerializer));
	set.AddFunction(PragmaFunction::PragmaStatement("disable_verify_serializer", PragmaDisableVerifySerializer));

	set.AddFunction(PragmaFunction::PragmaStatement("verify_parallelism", PragmaEnableForceParallelism));
	set.AddFunction(PragmaFunction::PragmaStatement("disable_verify_parallelism", PragmaDisableForceParallelism));

	set.AddFunction(PragmaFunction::PragmaStatement("enable_object_cache", PragmaEnableObjectCache));
	set.AddFunction(PragmaFunction::PragmaStatement("disable_object_cache", PragmaDisableObjectCache));

	set.AddFunction(PragmaFunction::PragmaStatement("enable_optimizer", PragmaEnableOptimizer));
	set.AddFunction(PragmaFunction::PragmaStatement("disable_optimizer", PragmaDisableOptimizer));

	set.AddFunction(PragmaFunction::PragmaStatement("force_index_join", PragmaEnableForceIndexJoin));
	set.AddFunction(PragmaFunction::PragmaStatement("force_checkpoint", PragmaForceCheckpoint));

	set.AddFunction(PragmaFunction::PragmaStatement("enable_progress_bar", PragmaEnableProgressBar));
	set.AddFunction(PragmaFunction::PragmaStatement("disable_progress_bar", PragmaDisableProgressBar));

	set.AddFunction(PragmaFunction::PragmaStatement("enable_print_progress_bar", PragmaEnablePrintProgressBar));
	set.AddFunction(PragmaFunction::PragmaStatement("disable_print_progress_bar", PragmaDisablePrintProgressBar));

	set.AddFunction(PragmaFunction::PragmaStatement("enable_checkpoint_on_shutdown", PragmaEnableCheckpointOnShutdown));
	set.AddFunction(
	    PragmaFunction::PragmaStatement("disable_checkpoint_on_shutdown", PragmaDisableCheckpointOnShutdown));
}

} // namespace duckdb













namespace duckdb {

string PragmaTableInfo(ClientContext &context, const FunctionParameters &parameters) {
	return StringUtil::Format("SELECT * FROM pragma_table_info('%s');", parameters.values[0].ToString());
}

string PragmaShowTables(ClientContext &context, const FunctionParameters &parameters) {
	auto catalog = DatabaseManager::GetDefaultDatabase(context);
	auto schema = ClientData::Get(context).catalog_search_path->GetDefault().schema;
	schema = (schema == INVALID_SCHEMA) ? DEFAULT_SCHEMA : schema; // NOLINT

	auto where_clause = StringUtil::Format("where ((database_name = '%s') and (schema_name = '%s'))", catalog, schema);
	// clang-format off
	auto pragma_query = StringUtil::Format(R"EOF(
	with "tables" as
	(
		SELECT table_name as "name"
		FROM duckdb_tables %s
	), "views" as
	(
		SELECT view_name as "name"
		FROM duckdb_views %s
	), db_objects as
	(
		SELECT "name" FROM "tables"
		UNION ALL
		SELECT "name" FROM "views"
	)
	SELECT "name"
	FROM db_objects
	ORDER BY "name";)EOF", where_clause, where_clause);
	// clang-format on

	return pragma_query;
}

string PragmaShowTablesExpanded(ClientContext &context, const FunctionParameters &parameters) {
	return R"(
			SELECT
				t.database_name AS database,
				t.schema_name AS schema,
				t.table_name,
				LIST(c.column_name order by c.column_index) AS column_names,
				LIST(c.data_type order by c.column_index) AS column_types,
				FIRST(t.temporary) AS temporary,
			FROM duckdb_tables t
			JOIN duckdb_columns c
			USING (table_oid)
			GROUP BY t.database_name, t.schema_name, t.table_name
			ORDER BY t.database_name, t.schema_name, t.table_name;
	)";
}

string PragmaShowDatabases(ClientContext &context, const FunctionParameters &parameters) {
	return "SELECT database_name FROM duckdb_databases() WHERE NOT internal ORDER BY database_name;";
}

string PragmaAllProfiling(ClientContext &context, const FunctionParameters &parameters) {
	return "SELECT * FROM pragma_last_profiling_output() JOIN pragma_detailed_profiling_output() ON "
	       "(pragma_last_profiling_output.operator_id);";
}

string PragmaDatabaseList(ClientContext &context, const FunctionParameters &parameters) {
	return "SELECT * FROM pragma_database_list;";
}

string PragmaCollations(ClientContext &context, const FunctionParameters &parameters) {
	return "SELECT * FROM pragma_collations() ORDER BY 1;";
}

string PragmaFunctionsQuery(ClientContext &context, const FunctionParameters &parameters) {
	return "SELECT function_name AS name, upper(function_type) AS type, parameter_types AS parameters, varargs, "
	       "return_type, has_side_effects AS side_effects"
	       " FROM duckdb_functions()"
	       " WHERE function_type IN ('scalar', 'aggregate')"
	       " ORDER BY 1;";
}

string PragmaShow(ClientContext &context, const FunctionParameters &parameters) {
	// PRAGMA table_info but with some aliases
	auto table = QualifiedName::Parse(parameters.values[0].ToString());

	// clang-format off
    string sql = R"(
	SELECT
		name AS "column_name",
		type as "column_type",
		CASE WHEN "notnull" THEN 'NO' ELSE 'YES' END AS "null",
		(SELECT 
			MIN(CASE 
				WHEN constraint_type='PRIMARY KEY' THEN 'PRI'
				WHEN constraint_type='UNIQUE' THEN 'UNI' 
				ELSE NULL END) 
		FROM duckdb_constraints() c  
		WHERE c.table_oid=cols.table_oid 
		AND list_contains(constraint_column_names, cols.column_name)) AS "key",
		dflt_value AS "default", 
		NULL AS "extra" 
	FROM pragma_table_info('%func_param_table%') 
	LEFT JOIN duckdb_columns cols 
	ON cols.column_name = pragma_table_info.name 
	AND cols.table_name='%table_name%'
	AND cols.schema_name='%table_schema%';)";
	// clang-format on

	sql = StringUtil::Replace(sql, "%func_param_table%", parameters.values[0].ToString());
	sql = StringUtil::Replace(sql, "%table_name%", table.name);
	sql = StringUtil::Replace(sql, "%table_schema%", table.schema.empty() ? DEFAULT_SCHEMA : table.schema);
	return sql;
}

string PragmaVersion(ClientContext &context, const FunctionParameters &parameters) {
	return "SELECT * FROM pragma_version();";
}

string PragmaImportDatabase(ClientContext &context, const FunctionParameters &parameters) {
	auto &config = DBConfig::GetConfig(context);
	if (!config.options.enable_external_access) {
		throw PermissionException("Import is disabled through configuration");
	}
	auto &fs = FileSystem::GetFileSystem(context);

	string final_query;
	// read the "shema.sql" and "load.sql" files
	vector<string> files = {"schema.sql", "load.sql"};
	for (auto &file : files) {
		auto file_path = fs.JoinPath(parameters.values[0].ToString(), file);
		auto handle = fs.OpenFile(file_path, FileFlags::FILE_FLAGS_READ, FileSystem::DEFAULT_LOCK,
		                          FileSystem::DEFAULT_COMPRESSION);
		auto fsize = fs.GetFileSize(*handle);
		auto buffer = make_unsafe_uniq_array<char>(fsize);
		fs.Read(*handle, buffer.get(), fsize);
		auto query = string(buffer.get(), fsize);
		// Replace the placeholder with the path provided to IMPORT
		if (file == "load.sql") {
			Parser parser;
			parser.ParseQuery(query);
			auto copy_statements = std::move(parser.statements);
			query.clear();
			for (auto &statement_p : copy_statements) {
				D_ASSERT(statement_p->type == StatementType::COPY_STATEMENT);
				auto &statement = statement_p->Cast<CopyStatement>();
				auto &info = *statement.info;
				auto file_name = fs.ExtractName(info.file_path);
				info.file_path = fs.JoinPath(parameters.values[0].ToString(), file_name);
				query += statement.ToString() + ";";
			}
		}
		final_query += query;
	}
	return final_query;
}

string PragmaDatabaseSize(ClientContext &context, const FunctionParameters &parameters) {
	return "SELECT * FROM pragma_database_size();";
}

string PragmaStorageInfo(ClientContext &context, const FunctionParameters &parameters) {
	return StringUtil::Format("SELECT * FROM pragma_storage_info('%s');", parameters.values[0].ToString());
}

void PragmaQueries::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(PragmaFunction::PragmaCall("table_info", PragmaTableInfo, {LogicalType::VARCHAR}));
	set.AddFunction(PragmaFunction::PragmaCall("storage_info", PragmaStorageInfo, {LogicalType::VARCHAR}));
	set.AddFunction(PragmaFunction::PragmaStatement("show_tables", PragmaShowTables));
	set.AddFunction(PragmaFunction::PragmaStatement("show_tables_expanded", PragmaShowTablesExpanded));
	set.AddFunction(PragmaFunction::PragmaStatement("show_databases", PragmaShowDatabases));
	set.AddFunction(PragmaFunction::PragmaStatement("database_list", PragmaDatabaseList));
	set.AddFunction(PragmaFunction::PragmaStatement("collations", PragmaCollations));
	set.AddFunction(PragmaFunction::PragmaCall("show", PragmaShow, {LogicalType::VARCHAR}));
	set.AddFunction(PragmaFunction::PragmaStatement("version", PragmaVersion));
	set.AddFunction(PragmaFunction::PragmaStatement("database_size", PragmaDatabaseSize));
	set.AddFunction(PragmaFunction::PragmaStatement("functions", PragmaFunctionsQuery));
	set.AddFunction(PragmaFunction::PragmaCall("import_database", PragmaImportDatabase, {LogicalType::VARCHAR}));
	set.AddFunction(PragmaFunction::PragmaStatement("all_profiling_output", PragmaAllProfiling));
}

} // namespace duckdb



namespace duckdb {

PragmaFunction::PragmaFunction(string name, PragmaType pragma_type, pragma_query_t query, pragma_function_t function,
                               vector<LogicalType> arguments, LogicalType varargs)
    : SimpleNamedParameterFunction(std::move(name), std::move(arguments), std::move(varargs)), type(pragma_type),
      query(query), function(function) {
}

PragmaFunction PragmaFunction::PragmaCall(const string &name, pragma_query_t query, vector<LogicalType> arguments,
                                          LogicalType varargs) {
	return PragmaFunction(name, PragmaType::PRAGMA_CALL, query, nullptr, std::move(arguments), std::move(varargs));
}

PragmaFunction PragmaFunction::PragmaCall(const string &name, pragma_function_t function, vector<LogicalType> arguments,
                                          LogicalType varargs) {
	return PragmaFunction(name, PragmaType::PRAGMA_CALL, nullptr, function, std::move(arguments), std::move(varargs));
}

PragmaFunction PragmaFunction::PragmaStatement(const string &name, pragma_query_t query) {
	vector<LogicalType> types;
	return PragmaFunction(name, PragmaType::PRAGMA_STATEMENT, query, nullptr, std::move(types), LogicalType::INVALID);
}

PragmaFunction PragmaFunction::PragmaStatement(const string &name, pragma_function_t function) {
	vector<LogicalType> types;
	return PragmaFunction(name, PragmaType::PRAGMA_STATEMENT, nullptr, function, std::move(types),
	                      LogicalType::INVALID);
}

string PragmaFunction::ToString() const {
	switch (type) {
	case PragmaType::PRAGMA_STATEMENT:
		return StringUtil::Format("PRAGMA %s", name);
	case PragmaType::PRAGMA_CALL: {
		return StringUtil::Format("PRAGMA %s", SimpleNamedParameterFunction::ToString());
	}
	default:
		return "UNKNOWN";
	}
}

} // namespace duckdb





namespace duckdb {

struct ConstantOrNullBindData : public FunctionData {
	explicit ConstantOrNullBindData(Value val) : value(std::move(val)) {
	}

	Value value;

public:
	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<ConstantOrNullBindData>(value);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = (const ConstantOrNullBindData &)other_p;
		return value == other.value;
	}
};

static void ConstantOrNullFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &info = func_expr.bind_info->Cast<ConstantOrNullBindData>();
	result.Reference(info.value);
	for (idx_t idx = 1; idx < args.ColumnCount(); idx++) {
		switch (args.data[idx].GetVectorType()) {
		case VectorType::FLAT_VECTOR: {
			auto &input_mask = FlatVector::Validity(args.data[idx]);
			if (!input_mask.AllValid()) {
				// there are null values: need to merge them into the result
				result.Flatten(args.size());
				auto &result_mask = FlatVector::Validity(result);
				result_mask.Combine(input_mask, args.size());
			}
			break;
		}
		case VectorType::CONSTANT_VECTOR: {
			if (ConstantVector::IsNull(args.data[idx])) {
				// input is constant null, return constant null
				result.Reference(info.value);
				ConstantVector::SetNull(result, true);
				return;
			}
			break;
		}
		default: {
			UnifiedVectorFormat vdata;
			args.data[idx].ToUnifiedFormat(args.size(), vdata);
			if (!vdata.validity.AllValid()) {
				result.Flatten(args.size());
				auto &result_mask = FlatVector::Validity(result);
				for (idx_t i = 0; i < args.size(); i++) {
					if (!vdata.validity.RowIsValid(vdata.sel->get_index(i))) {
						result_mask.SetInvalid(i);
					}
				}
			}
			break;
		}
		}
	}
}

ScalarFunction ConstantOrNull::GetFunction(const LogicalType &return_type) {
	return ScalarFunction("constant_or_null", {return_type, LogicalType::ANY}, return_type, ConstantOrNullFunction);
}

unique_ptr<FunctionData> ConstantOrNull::Bind(Value value) {
	return make_uniq<ConstantOrNullBindData>(std::move(value));
}

bool ConstantOrNull::IsConstantOrNull(BoundFunctionExpression &expr, const Value &val) {
	if (expr.function.name != "constant_or_null") {
		return false;
	}
	D_ASSERT(expr.bind_info);
	auto &bind_data = expr.bind_info->Cast<ConstantOrNullBindData>();
	D_ASSERT(bind_data.value.type() == val.type());
	return bind_data.value == val;
}

unique_ptr<FunctionData> ConstantOrNullBind(ClientContext &context, ScalarFunction &bound_function,
                                            vector<unique_ptr<Expression>> &arguments) {
	if (arguments[0]->HasParameter()) {
		throw ParameterNotResolvedException();
	}
	if (!arguments[0]->IsFoldable()) {
		throw BinderException("ConstantOrNull requires a constant input");
	}
	D_ASSERT(arguments.size() >= 2);
	auto value = ExpressionExecutor::EvaluateScalar(context, *arguments[0]);
	bound_function.return_type = arguments[0]->return_type;
	return make_uniq<ConstantOrNullBindData>(std::move(value));
}

void ConstantOrNull::RegisterFunction(BuiltinFunctions &set) {
	auto fun = ConstantOrNull::GetFunction(LogicalType::ANY);
	fun.bind = ConstantOrNullBind;
	fun.varargs = LogicalType::ANY;
	set.AddFunction(fun);
}

} // namespace duckdb


namespace duckdb {

void BuiltinFunctions::RegisterGenericFunctions() {
	Register<ConstantOrNull>();
	Register<ExportAggregateFunction>();
}

} // namespace duckdb





namespace duckdb {

static void ListContainsFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	(void)state;
	return ListContainsOrPosition<bool, ContainsFunctor, ListArgFunctor>(args, result);
}

static void ListPositionFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	(void)state;
	return ListContainsOrPosition<int32_t, PositionFunctor, ListArgFunctor>(args, result);
}

template <LogicalTypeId RETURN_TYPE>
static unique_ptr<FunctionData> ListContainsOrPositionBind(ClientContext &context, ScalarFunction &bound_function,
                                                           vector<unique_ptr<Expression>> &arguments) {
	D_ASSERT(bound_function.arguments.size() == 2);

	const auto &list = arguments[0]->return_type; // change to list
	const auto &value = arguments[1]->return_type;
	if (list.id() == LogicalTypeId::UNKNOWN) {
		bound_function.return_type = RETURN_TYPE;
		if (value.id() != LogicalTypeId::UNKNOWN) {
			// only list is a parameter, cast it to a list of value type
			bound_function.arguments[0] = LogicalType::LIST(value);
			bound_function.arguments[1] = value;
		}
	} else if (value.id() == LogicalTypeId::UNKNOWN) {
		// only value is a parameter: we expect the child type of list
		auto const &child_type = ListType::GetChildType(list);
		bound_function.arguments[0] = list;
		bound_function.arguments[1] = child_type;
		bound_function.return_type = RETURN_TYPE;
	} else {
		auto const &child_type = ListType::GetChildType(list);
		auto max_child_type = LogicalType::MaxLogicalType(child_type, value);
		auto list_type = LogicalType::LIST(max_child_type);

		bound_function.arguments[0] = list_type;
		bound_function.arguments[1] = value == max_child_type ? value : max_child_type;

		// list_contains and list_position only differ in their return type
		bound_function.return_type = RETURN_TYPE;
	}
	return make_uniq<VariableReturnBindData>(bound_function.return_type);
}

static unique_ptr<FunctionData> ListContainsBind(ClientContext &context, ScalarFunction &bound_function,
                                                 vector<unique_ptr<Expression>> &arguments) {
	return ListContainsOrPositionBind<LogicalType::BOOLEAN>(context, bound_function, arguments);
}

static unique_ptr<FunctionData> ListPositionBind(ClientContext &context, ScalarFunction &bound_function,
                                                 vector<unique_ptr<Expression>> &arguments) {
	return ListContainsOrPositionBind<LogicalType::INTEGER>(context, bound_function, arguments);
}

ScalarFunction ListContainsFun::GetFunction() {
	return ScalarFunction({LogicalType::LIST(LogicalType::ANY), LogicalType::ANY}, // argument list
	                      LogicalType::BOOLEAN,                                    // return type
	                      ListContainsFunction, ListContainsBind, nullptr);
}

ScalarFunction ListPositionFun::GetFunction() {
	return ScalarFunction({LogicalType::LIST(LogicalType::ANY), LogicalType::ANY}, // argument list
	                      LogicalType::INTEGER,                                    // return type
	                      ListPositionFunction, ListPositionBind, nullptr);
}

void ListContainsFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction({"list_contains", "array_contains", "list_has", "array_has"}, GetFunction());
}

void ListPositionFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction({"list_position", "list_indexof", "array_position", "array_indexof"}, GetFunction());
}
} // namespace duckdb






namespace duckdb {

static void ListConcatFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(args.ColumnCount() == 2);
	auto count = args.size();

	Vector &lhs = args.data[0];
	Vector &rhs = args.data[1];
	if (lhs.GetType().id() == LogicalTypeId::SQLNULL) {
		result.Reference(rhs);
		return;
	}
	if (rhs.GetType().id() == LogicalTypeId::SQLNULL) {
		result.Reference(lhs);
		return;
	}

	UnifiedVectorFormat lhs_data;
	UnifiedVectorFormat rhs_data;
	lhs.ToUnifiedFormat(count, lhs_data);
	rhs.ToUnifiedFormat(count, rhs_data);
	auto lhs_entries = (list_entry_t *)lhs_data.data;
	auto rhs_entries = (list_entry_t *)rhs_data.data;

	auto lhs_list_size = ListVector::GetListSize(lhs);
	auto rhs_list_size = ListVector::GetListSize(rhs);
	auto &lhs_child = ListVector::GetEntry(lhs);
	auto &rhs_child = ListVector::GetEntry(rhs);
	UnifiedVectorFormat lhs_child_data;
	UnifiedVectorFormat rhs_child_data;
	lhs_child.ToUnifiedFormat(lhs_list_size, lhs_child_data);
	rhs_child.ToUnifiedFormat(rhs_list_size, rhs_child_data);

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_entries = FlatVector::GetData<list_entry_t>(result);
	auto &result_validity = FlatVector::Validity(result);

	idx_t offset = 0;
	for (idx_t i = 0; i < count; i++) {
		auto lhs_list_index = lhs_data.sel->get_index(i);
		auto rhs_list_index = rhs_data.sel->get_index(i);
		if (!lhs_data.validity.RowIsValid(lhs_list_index) && !rhs_data.validity.RowIsValid(rhs_list_index)) {
			result_validity.SetInvalid(i);
			continue;
		}
		result_entries[i].offset = offset;
		result_entries[i].length = 0;
		if (lhs_data.validity.RowIsValid(lhs_list_index)) {
			const auto &lhs_entry = lhs_entries[lhs_list_index];
			result_entries[i].length += lhs_entry.length;
			ListVector::Append(result, lhs_child, *lhs_child_data.sel, lhs_entry.offset + lhs_entry.length,
			                   lhs_entry.offset);
		}
		if (rhs_data.validity.RowIsValid(rhs_list_index)) {
			const auto &rhs_entry = rhs_entries[rhs_list_index];
			result_entries[i].length += rhs_entry.length;
			ListVector::Append(result, rhs_child, *rhs_child_data.sel, rhs_entry.offset + rhs_entry.length,
			                   rhs_entry.offset);
		}
		offset += result_entries[i].length;
	}
	D_ASSERT(ListVector::GetListSize(result) == offset);

	if (lhs.GetVectorType() == VectorType::CONSTANT_VECTOR && rhs.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

static unique_ptr<FunctionData> ListConcatBind(ClientContext &context, ScalarFunction &bound_function,
                                               vector<unique_ptr<Expression>> &arguments) {
	D_ASSERT(bound_function.arguments.size() == 2);

	auto &lhs = arguments[0]->return_type;
	auto &rhs = arguments[1]->return_type;
	if (lhs.id() == LogicalTypeId::UNKNOWN || rhs.id() == LogicalTypeId::UNKNOWN) {
		throw ParameterNotResolvedException();
	} else if (lhs.id() == LogicalTypeId::SQLNULL || rhs.id() == LogicalTypeId::SQLNULL) {
		// we mimic postgres behaviour: list_concat(NULL, my_list) = my_list
		auto return_type = rhs.id() == LogicalTypeId::SQLNULL ? lhs : rhs;
		bound_function.arguments[0] = return_type;
		bound_function.arguments[1] = return_type;
		bound_function.return_type = return_type;
	} else {
		D_ASSERT(lhs.id() == LogicalTypeId::LIST);
		D_ASSERT(rhs.id() == LogicalTypeId::LIST);

		// Resolve list type
		LogicalType child_type = LogicalType::SQLNULL;
		for (const auto &argument : arguments) {
			child_type = LogicalType::MaxLogicalType(child_type, ListType::GetChildType(argument->return_type));
		}
		auto list_type = LogicalType::LIST(child_type);

		bound_function.arguments[0] = list_type;
		bound_function.arguments[1] = list_type;
		bound_function.return_type = list_type;
	}
	return make_uniq<VariableReturnBindData>(bound_function.return_type);
}

static unique_ptr<BaseStatistics> ListConcatStats(ClientContext &context, FunctionStatisticsInput &input) {
	auto &child_stats = input.child_stats;
	D_ASSERT(child_stats.size() == 2);

	auto &left_stats = child_stats[0];
	auto &right_stats = child_stats[1];

	auto stats = left_stats.ToUnique();
	stats->Merge(right_stats);

	return stats;
}

ScalarFunction ListConcatFun::GetFunction() {
	// the arguments and return types are actually set in the binder function
	auto fun = ScalarFunction({LogicalType::LIST(LogicalType::ANY), LogicalType::LIST(LogicalType::ANY)},
	                          LogicalType::LIST(LogicalType::ANY), ListConcatFunction, ListConcatBind, nullptr,
	                          ListConcatStats);
	fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	return fun;
}

void ListConcatFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction({"list_concat", "list_cat", "array_concat", "array_cat"}, GetFunction());
}

} // namespace duckdb











namespace duckdb {

template <class T, bool HEAP_REF = false, bool VALIDITY_ONLY = false>
void ListExtractTemplate(idx_t count, UnifiedVectorFormat &list_data, UnifiedVectorFormat &offsets_data,
                         Vector &child_vector, idx_t list_size, Vector &result) {
	UnifiedVectorFormat child_format;
	child_vector.ToUnifiedFormat(list_size, child_format);

	T *result_data;

	result.SetVectorType(VectorType::FLAT_VECTOR);
	if (!VALIDITY_ONLY) {
		result_data = FlatVector::GetData<T>(result);
	}
	auto &result_mask = FlatVector::Validity(result);

	// heap-ref once
	if (HEAP_REF) {
		StringVector::AddHeapReference(result, child_vector);
	}

	// this is lifted from ExecuteGenericLoop because we can't push the list child data into this otherwise
	// should have gone with GetValue perhaps
	auto child_data = (T *)child_format.data;
	for (idx_t i = 0; i < count; i++) {
		auto list_index = list_data.sel->get_index(i);
		auto offsets_index = offsets_data.sel->get_index(i);
		if (!list_data.validity.RowIsValid(list_index)) {
			result_mask.SetInvalid(i);
			continue;
		}
		if (!offsets_data.validity.RowIsValid(offsets_index)) {
			result_mask.SetInvalid(i);
			continue;
		}
		auto list_entry = ((list_entry_t *)list_data.data)[list_index];
		auto offsets_entry = ((int64_t *)offsets_data.data)[offsets_index];

		// 1-based indexing
		if (offsets_entry == 0) {
			result_mask.SetInvalid(i);
			continue;
		}
		offsets_entry = (offsets_entry > 0) ? offsets_entry - 1 : offsets_entry;

		idx_t child_offset;
		if (offsets_entry < 0) {
			if ((idx_t)-offsets_entry > list_entry.length) {
				result_mask.SetInvalid(i);
				continue;
			}
			child_offset = list_entry.offset + list_entry.length + offsets_entry;
		} else {
			if ((idx_t)offsets_entry >= list_entry.length) {
				result_mask.SetInvalid(i);
				continue;
			}
			child_offset = list_entry.offset + offsets_entry;
		}
		auto child_index = child_format.sel->get_index(child_offset);
		if (child_format.validity.RowIsValid(child_index)) {
			if (!VALIDITY_ONLY) {
				result_data[i] = child_data[child_index];
			}
		} else {
			result_mask.SetInvalid(i);
		}
	}
	if (count == 1) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}
static void ExecuteListExtractInternal(const idx_t count, UnifiedVectorFormat &list, UnifiedVectorFormat &offsets,
                                       Vector &child_vector, idx_t list_size, Vector &result) {
	D_ASSERT(child_vector.GetType() == result.GetType());
	switch (result.GetType().InternalType()) {
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
		ListExtractTemplate<int8_t>(count, list, offsets, child_vector, list_size, result);
		break;
	case PhysicalType::INT16:
		ListExtractTemplate<int16_t>(count, list, offsets, child_vector, list_size, result);
		break;
	case PhysicalType::INT32:
		ListExtractTemplate<int32_t>(count, list, offsets, child_vector, list_size, result);
		break;
	case PhysicalType::INT64:
		ListExtractTemplate<int64_t>(count, list, offsets, child_vector, list_size, result);
		break;
	case PhysicalType::INT128:
		ListExtractTemplate<hugeint_t>(count, list, offsets, child_vector, list_size, result);
		break;
	case PhysicalType::UINT8:
		ListExtractTemplate<uint8_t>(count, list, offsets, child_vector, list_size, result);
		break;
	case PhysicalType::UINT16:
		ListExtractTemplate<uint16_t>(count, list, offsets, child_vector, list_size, result);
		break;
	case PhysicalType::UINT32:
		ListExtractTemplate<uint32_t>(count, list, offsets, child_vector, list_size, result);
		break;
	case PhysicalType::UINT64:
		ListExtractTemplate<uint64_t>(count, list, offsets, child_vector, list_size, result);
		break;
	case PhysicalType::FLOAT:
		ListExtractTemplate<float>(count, list, offsets, child_vector, list_size, result);
		break;
	case PhysicalType::DOUBLE:
		ListExtractTemplate<double>(count, list, offsets, child_vector, list_size, result);
		break;
	case PhysicalType::VARCHAR:
		ListExtractTemplate<string_t, true>(count, list, offsets, child_vector, list_size, result);
		break;
	case PhysicalType::INTERVAL:
		ListExtractTemplate<interval_t>(count, list, offsets, child_vector, list_size, result);
		break;
	case PhysicalType::STRUCT: {
		auto &entries = StructVector::GetEntries(child_vector);
		auto &result_entries = StructVector::GetEntries(result);
		D_ASSERT(entries.size() == result_entries.size());
		// extract the child entries of the struct
		for (idx_t i = 0; i < entries.size(); i++) {
			ExecuteListExtractInternal(count, list, offsets, *entries[i], list_size, *result_entries[i]);
		}
		// extract the validity mask
		ListExtractTemplate<bool, false, true>(count, list, offsets, child_vector, list_size, result);
		break;
	}
	case PhysicalType::LIST: {
		// nested list: we have to reference the child
		auto &child_child_list = ListVector::GetEntry(child_vector);

		ListVector::GetEntry(result).Reference(child_child_list);
		ListVector::SetListSize(result, ListVector::GetListSize(child_vector));
		ListExtractTemplate<list_entry_t>(count, list, offsets, child_vector, list_size, result);
		break;
	}
	default:
		throw NotImplementedException("Unimplemented type for LIST_EXTRACT");
	}
}

static void ExecuteListExtract(Vector &result, Vector &list, Vector &offsets, const idx_t count) {
	D_ASSERT(list.GetType().id() == LogicalTypeId::LIST);
	UnifiedVectorFormat list_data;
	UnifiedVectorFormat offsets_data;

	list.ToUnifiedFormat(count, list_data);
	offsets.ToUnifiedFormat(count, offsets_data);
	ExecuteListExtractInternal(count, list_data, offsets_data, ListVector::GetEntry(list),
	                           ListVector::GetListSize(list), result);
	result.Verify(count);
}

static void ExecuteStringExtract(Vector &result, Vector &input_vector, Vector &subscript_vector, const idx_t count) {
	BinaryExecutor::Execute<string_t, int64_t, string_t>(
	    input_vector, subscript_vector, result, count, [&](string_t input_string, int64_t subscript) {
		    return SubstringFun::SubstringUnicode(result, input_string, subscript, 1);
	    });
}

static void ListExtractFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(args.ColumnCount() == 2);
	auto count = args.size();

	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	for (idx_t i = 0; i < args.ColumnCount(); i++) {
		if (args.data[i].GetVectorType() != VectorType::CONSTANT_VECTOR) {
			result.SetVectorType(VectorType::FLAT_VECTOR);
		}
	}

	Vector &base = args.data[0];
	Vector &subscript = args.data[1];

	switch (base.GetType().id()) {
	case LogicalTypeId::LIST:
		ExecuteListExtract(result, base, subscript, count);
		break;
	case LogicalTypeId::VARCHAR:
		ExecuteStringExtract(result, base, subscript, count);
		break;
	case LogicalTypeId::SQLNULL:
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
		ConstantVector::SetNull(result, true);
		break;
	default:
		throw NotImplementedException("Specifier type not implemented");
	}
}

static unique_ptr<FunctionData> ListExtractBind(ClientContext &context, ScalarFunction &bound_function,
                                                vector<unique_ptr<Expression>> &arguments) {
	D_ASSERT(bound_function.arguments.size() == 2);
	D_ASSERT(LogicalTypeId::LIST == arguments[0]->return_type.id());
	// list extract returns the child type of the list as return type
	bound_function.return_type = ListType::GetChildType(arguments[0]->return_type);
	return make_uniq<VariableReturnBindData>(bound_function.return_type);
}

static unique_ptr<BaseStatistics> ListExtractStats(ClientContext &context, FunctionStatisticsInput &input) {
	auto &child_stats = input.child_stats;
	auto &list_child_stats = ListStats::GetChildStats(child_stats[0]);
	auto child_copy = list_child_stats.Copy();
	// list_extract always pushes a NULL, since if the offset is out of range for a list it inserts a null
	child_copy.Set(StatsInfo::CAN_HAVE_NULL_VALUES);
	return child_copy.ToUnique();
}

void ListExtractFun::RegisterFunction(BuiltinFunctions &set) {
	// the arguments and return types are actually set in the binder function
	ScalarFunction lfun({LogicalType::LIST(LogicalType::ANY), LogicalType::BIGINT}, LogicalType::ANY,
	                    ListExtractFunction, ListExtractBind, nullptr, ListExtractStats);

	ScalarFunction sfun({LogicalType::VARCHAR, LogicalType::BIGINT}, LogicalType::VARCHAR, ListExtractFunction);

	ScalarFunctionSet list_extract("list_extract");
	list_extract.AddFunction(lfun);
	list_extract.AddFunction(sfun);
	set.AddFunction(list_extract);

	ScalarFunctionSet list_element("list_element");
	list_element.AddFunction(lfun);
	list_element.AddFunction(sfun);
	set.AddFunction(list_element);

	ScalarFunctionSet array_extract("array_extract");
	array_extract.AddFunction(lfun);
	array_extract.AddFunction(sfun);
	array_extract.AddFunction(StructExtractFun::GetFunction());
	set.AddFunction(array_extract);
}

} // namespace duckdb


namespace duckdb {

void BuiltinFunctions::RegisterNestedFunctions() {
	Register<StructExtractFun>();
	Register<ListConcatFun>();
	Register<ListContainsFun>();
	Register<ListPositionFun>();
	Register<ListExtractFun>();
}

} // namespace duckdb










namespace duckdb {

//===--------------------------------------------------------------------===//
// + [add]
//===--------------------------------------------------------------------===//
template <>
float AddOperator::Operation(float left, float right) {
	auto result = left + right;
	return result;
}

template <>
double AddOperator::Operation(double left, double right) {
	auto result = left + right;
	return result;
}

template <>
interval_t AddOperator::Operation(interval_t left, interval_t right) {
	left.months = AddOperatorOverflowCheck::Operation<int32_t, int32_t, int32_t>(left.months, right.months);
	left.days = AddOperatorOverflowCheck::Operation<int32_t, int32_t, int32_t>(left.days, right.days);
	left.micros = AddOperatorOverflowCheck::Operation<int64_t, int64_t, int64_t>(left.micros, right.micros);
	return left;
}

template <>
date_t AddOperator::Operation(date_t left, int32_t right) {
	if (!Value::IsFinite(left)) {
		return left;
	}
	int32_t days;
	if (!TryAddOperator::Operation(left.days, right, days)) {
		throw OutOfRangeException("Date out of range");
	}
	date_t result(days);
	if (!Value::IsFinite(result)) {
		throw OutOfRangeException("Date out of range");
	}
	return result;
}

template <>
date_t AddOperator::Operation(int32_t left, date_t right) {
	return AddOperator::Operation<date_t, int32_t, date_t>(right, left);
}

template <>
timestamp_t AddOperator::Operation(date_t left, dtime_t right) {
	if (left == date_t::infinity()) {
		return timestamp_t::infinity();
	} else if (left == date_t::ninfinity()) {
		return timestamp_t::ninfinity();
	}
	timestamp_t result;
	if (!Timestamp::TryFromDatetime(left, right, result)) {
		throw OutOfRangeException("Timestamp out of range");
	}
	return result;
}

template <>
timestamp_t AddOperator::Operation(dtime_t left, date_t right) {
	return AddOperator::Operation<date_t, dtime_t, timestamp_t>(right, left);
}

template <>
date_t AddOperator::Operation(date_t left, interval_t right) {
	return Interval::Add(left, right);
}

template <>
date_t AddOperator::Operation(interval_t left, date_t right) {
	return AddOperator::Operation<date_t, interval_t, date_t>(right, left);
}

template <>
timestamp_t AddOperator::Operation(timestamp_t left, interval_t right) {
	return Interval::Add(left, right);
}

template <>
timestamp_t AddOperator::Operation(interval_t left, timestamp_t right) {
	return AddOperator::Operation<timestamp_t, interval_t, timestamp_t>(right, left);
}

//===--------------------------------------------------------------------===//
// + [add] with overflow check
//===--------------------------------------------------------------------===//
struct OverflowCheckedAddition {
	template <class SRCTYPE, class UTYPE>
	static inline bool Operation(SRCTYPE left, SRCTYPE right, SRCTYPE &result) {
		UTYPE uresult = AddOperator::Operation<UTYPE, UTYPE, UTYPE>(UTYPE(left), UTYPE(right));
		if (uresult < NumericLimits<SRCTYPE>::Minimum() || uresult > NumericLimits<SRCTYPE>::Maximum()) {
			return false;
		}
		result = SRCTYPE(uresult);
		return true;
	}
};

template <>
bool TryAddOperator::Operation(uint8_t left, uint8_t right, uint8_t &result) {
	return OverflowCheckedAddition::Operation<uint8_t, uint16_t>(left, right, result);
}
template <>
bool TryAddOperator::Operation(uint16_t left, uint16_t right, uint16_t &result) {
	return OverflowCheckedAddition::Operation<uint16_t, uint32_t>(left, right, result);
}
template <>
bool TryAddOperator::Operation(uint32_t left, uint32_t right, uint32_t &result) {
	return OverflowCheckedAddition::Operation<uint32_t, uint64_t>(left, right, result);
}

template <>
bool TryAddOperator::Operation(uint64_t left, uint64_t right, uint64_t &result) {
	if (NumericLimits<uint64_t>::Maximum() - left < right) {
		return false;
	}
	return OverflowCheckedAddition::Operation<uint64_t, uint64_t>(left, right, result);
}

template <>
bool TryAddOperator::Operation(int8_t left, int8_t right, int8_t &result) {
	return OverflowCheckedAddition::Operation<int8_t, int16_t>(left, right, result);
}

template <>
bool TryAddOperator::Operation(int16_t left, int16_t right, int16_t &result) {
	return OverflowCheckedAddition::Operation<int16_t, int32_t>(left, right, result);
}

template <>
bool TryAddOperator::Operation(int32_t left, int32_t right, int32_t &result) {
	return OverflowCheckedAddition::Operation<int32_t, int64_t>(left, right, result);
}

template <>
bool TryAddOperator::Operation(int64_t left, int64_t right, int64_t &result) {
#if (__GNUC__ >= 5) || defined(__clang__)
	if (__builtin_add_overflow(left, right, &result)) {
		return false;
	}
#else
	// https://blog.regehr.org/archives/1139
	result = int64_t((uint64_t)left + (uint64_t)right);
	if ((left < 0 && right < 0 && result >= 0) || (left >= 0 && right >= 0 && result < 0)) {
		return false;
	}
#endif
	return true;
}

//===--------------------------------------------------------------------===//
// add decimal with overflow check
//===--------------------------------------------------------------------===//
template <class T, T min, T max>
bool TryDecimalAddTemplated(T left, T right, T &result) {
	if (right < 0) {
		if (min - right > left) {
			return false;
		}
	} else {
		if (max - right < left) {
			return false;
		}
	}
	result = left + right;
	return true;
}

template <>
bool TryDecimalAdd::Operation(int16_t left, int16_t right, int16_t &result) {
	return TryDecimalAddTemplated<int16_t, -9999, 9999>(left, right, result);
}

template <>
bool TryDecimalAdd::Operation(int32_t left, int32_t right, int32_t &result) {
	return TryDecimalAddTemplated<int32_t, -999999999, 999999999>(left, right, result);
}

template <>
bool TryDecimalAdd::Operation(int64_t left, int64_t right, int64_t &result) {
	return TryDecimalAddTemplated<int64_t, -999999999999999999, 999999999999999999>(left, right, result);
}

template <>
bool TryDecimalAdd::Operation(hugeint_t left, hugeint_t right, hugeint_t &result) {
	result = left + right;
	if (result <= -Hugeint::POWERS_OF_TEN[38] || result >= Hugeint::POWERS_OF_TEN[38]) {
		return false;
	}
	return true;
}

template <>
hugeint_t DecimalAddOverflowCheck::Operation(hugeint_t left, hugeint_t right) {
	hugeint_t result;
	if (!TryDecimalAdd::Operation(left, right, result)) {
		throw OutOfRangeException("Overflow in addition of DECIMAL(38) (%s + %s);", left.ToString(), right.ToString());
	}
	return result;
}

//===--------------------------------------------------------------------===//
// add time operator
//===--------------------------------------------------------------------===//
template <>
dtime_t AddTimeOperator::Operation(dtime_t left, interval_t right) {
	date_t date(0);
	return Interval::Add(left, right, date);
}

template <>
dtime_t AddTimeOperator::Operation(interval_t left, dtime_t right) {
	return AddTimeOperator::Operation<dtime_t, interval_t, dtime_t>(right, left);
}

} // namespace duckdb

















#include <limits>

namespace duckdb {

template <class OP>
static scalar_function_t GetScalarIntegerFunction(PhysicalType type) {
	scalar_function_t function;
	switch (type) {
	case PhysicalType::INT8:
		function = &ScalarFunction::BinaryFunction<int8_t, int8_t, int8_t, OP>;
		break;
	case PhysicalType::INT16:
		function = &ScalarFunction::BinaryFunction<int16_t, int16_t, int16_t, OP>;
		break;
	case PhysicalType::INT32:
		function = &ScalarFunction::BinaryFunction<int32_t, int32_t, int32_t, OP>;
		break;
	case PhysicalType::INT64:
		function = &ScalarFunction::BinaryFunction<int64_t, int64_t, int64_t, OP>;
		break;
	case PhysicalType::UINT8:
		function = &ScalarFunction::BinaryFunction<uint8_t, uint8_t, uint8_t, OP>;
		break;
	case PhysicalType::UINT16:
		function = &ScalarFunction::BinaryFunction<uint16_t, uint16_t, uint16_t, OP>;
		break;
	case PhysicalType::UINT32:
		function = &ScalarFunction::BinaryFunction<uint32_t, uint32_t, uint32_t, OP>;
		break;
	case PhysicalType::UINT64:
		function = &ScalarFunction::BinaryFunction<uint64_t, uint64_t, uint64_t, OP>;
		break;
	default:
		throw NotImplementedException("Unimplemented type for GetScalarBinaryFunction");
	}
	return function;
}

template <class OP>
static scalar_function_t GetScalarBinaryFunction(PhysicalType type) {
	scalar_function_t function;
	switch (type) {
	case PhysicalType::INT128:
		function = &ScalarFunction::BinaryFunction<hugeint_t, hugeint_t, hugeint_t, OP>;
		break;
	case PhysicalType::FLOAT:
		function = &ScalarFunction::BinaryFunction<float, float, float, OP>;
		break;
	case PhysicalType::DOUBLE:
		function = &ScalarFunction::BinaryFunction<double, double, double, OP>;
		break;
	default:
		function = GetScalarIntegerFunction<OP>(type);
		break;
	}
	return function;
}

//===--------------------------------------------------------------------===//
// + [add]
//===--------------------------------------------------------------------===//
struct AddPropagateStatistics {
	template <class T, class OP>
	static bool Operation(LogicalType type, BaseStatistics &lstats, BaseStatistics &rstats, Value &new_min,
	                      Value &new_max) {
		T min, max;
		// new min is min+min
		if (!OP::Operation(NumericStats::GetMin<T>(lstats), NumericStats::GetMin<T>(rstats), min)) {
			return true;
		}
		// new max is max+max
		if (!OP::Operation(NumericStats::GetMax<T>(lstats), NumericStats::GetMax<T>(rstats), max)) {
			return true;
		}
		new_min = Value::Numeric(type, min);
		new_max = Value::Numeric(type, max);
		return false;
	}
};

struct SubtractPropagateStatistics {
	template <class T, class OP>
	static bool Operation(LogicalType type, BaseStatistics &lstats, BaseStatistics &rstats, Value &new_min,
	                      Value &new_max) {
		T min, max;
		if (!OP::Operation(NumericStats::GetMin<T>(lstats), NumericStats::GetMax<T>(rstats), min)) {
			return true;
		}
		if (!OP::Operation(NumericStats::GetMax<T>(lstats), NumericStats::GetMin<T>(rstats), max)) {
			return true;
		}
		new_min = Value::Numeric(type, min);
		new_max = Value::Numeric(type, max);
		return false;
	}
};

struct DecimalArithmeticBindData : public FunctionData {
	DecimalArithmeticBindData() : check_overflow(true) {
	}

	unique_ptr<FunctionData> Copy() const override {
		auto res = make_uniq<DecimalArithmeticBindData>();
		res->check_overflow = check_overflow;
		return std::move(res);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto other = other_p.Cast<DecimalArithmeticBindData>();
		return other.check_overflow == check_overflow;
	}

	bool check_overflow;
};

template <class OP, class PROPAGATE, class BASEOP>
static unique_ptr<BaseStatistics> PropagateNumericStats(ClientContext &context, FunctionStatisticsInput &input) {
	auto &child_stats = input.child_stats;
	auto &expr = input.expr;
	D_ASSERT(child_stats.size() == 2);
	// can only propagate stats if the children have stats
	auto &lstats = child_stats[0];
	auto &rstats = child_stats[1];
	Value new_min, new_max;
	bool potential_overflow = true;
	if (NumericStats::HasMinMax(lstats) && NumericStats::HasMinMax(rstats)) {
		switch (expr.return_type.InternalType()) {
		case PhysicalType::INT8:
			potential_overflow =
			    PROPAGATE::template Operation<int8_t, OP>(expr.return_type, lstats, rstats, new_min, new_max);
			break;
		case PhysicalType::INT16:
			potential_overflow =
			    PROPAGATE::template Operation<int16_t, OP>(expr.return_type, lstats, rstats, new_min, new_max);
			break;
		case PhysicalType::INT32:
			potential_overflow =
			    PROPAGATE::template Operation<int32_t, OP>(expr.return_type, lstats, rstats, new_min, new_max);
			break;
		case PhysicalType::INT64:
			potential_overflow =
			    PROPAGATE::template Operation<int64_t, OP>(expr.return_type, lstats, rstats, new_min, new_max);
			break;
		default:
			return nullptr;
		}
	}
	if (potential_overflow) {
		new_min = Value(expr.return_type);
		new_max = Value(expr.return_type);
	} else {
		// no potential overflow: replace with non-overflowing operator
		if (input.bind_data) {
			auto &bind_data = input.bind_data->Cast<DecimalArithmeticBindData>();
			bind_data.check_overflow = false;
		}
		expr.function.function = GetScalarIntegerFunction<BASEOP>(expr.return_type.InternalType());
	}
	auto result = NumericStats::CreateEmpty(expr.return_type);
	NumericStats::SetMin(result, new_min);
	NumericStats::SetMax(result, new_max);
	result.CombineValidity(lstats, rstats);
	return result.ToUnique();
}

template <class OP, class OPOVERFLOWCHECK, bool IS_SUBTRACT = false>
unique_ptr<FunctionData> BindDecimalAddSubtract(ClientContext &context, ScalarFunction &bound_function,
                                                vector<unique_ptr<Expression>> &arguments) {

	auto bind_data = make_uniq<DecimalArithmeticBindData>();

	// get the max width and scale of the input arguments
	uint8_t max_width = 0, max_scale = 0, max_width_over_scale = 0;
	for (idx_t i = 0; i < arguments.size(); i++) {
		if (arguments[i]->return_type.id() == LogicalTypeId::UNKNOWN) {
			continue;
		}
		uint8_t width, scale;
		auto can_convert = arguments[i]->return_type.GetDecimalProperties(width, scale);
		if (!can_convert) {
			throw InternalException("Could not convert type %s to a decimal.", arguments[i]->return_type.ToString());
		}
		max_width = MaxValue<uint8_t>(width, max_width);
		max_scale = MaxValue<uint8_t>(scale, max_scale);
		max_width_over_scale = MaxValue<uint8_t>(width - scale, max_width_over_scale);
	}
	D_ASSERT(max_width > 0);
	// for addition/subtraction, we add 1 to the width to ensure we don't overflow
	auto required_width = MaxValue<uint8_t>(max_scale + max_width_over_scale, max_width) + 1;
	if (required_width > Decimal::MAX_WIDTH_INT64 && max_width <= Decimal::MAX_WIDTH_INT64) {
		// we don't automatically promote past the hugeint boundary to avoid the large hugeint performance penalty
		bind_data->check_overflow = true;
		required_width = Decimal::MAX_WIDTH_INT64;
	}
	if (required_width > Decimal::MAX_WIDTH_DECIMAL) {
		// target width does not fit in decimal at all: truncate the scale and perform overflow detection
		bind_data->check_overflow = true;
		required_width = Decimal::MAX_WIDTH_DECIMAL;
	}
	// arithmetic between two decimal arguments: check the types of the input arguments
	LogicalType result_type = LogicalType::DECIMAL(required_width, max_scale);
	// we cast all input types to the specified type
	for (idx_t i = 0; i < arguments.size(); i++) {
		// first check if the cast is necessary
		// if the argument has a matching scale and internal type as the output type, no casting is necessary
		auto &argument_type = arguments[i]->return_type;
		uint8_t width, scale;
		argument_type.GetDecimalProperties(width, scale);
		if (scale == DecimalType::GetScale(result_type) && argument_type.InternalType() == result_type.InternalType()) {
			bound_function.arguments[i] = argument_type;
		} else {
			bound_function.arguments[i] = result_type;
		}
	}
	bound_function.return_type = result_type;
	// now select the physical function to execute
	if (bind_data->check_overflow) {
		bound_function.function = GetScalarBinaryFunction<OPOVERFLOWCHECK>(result_type.InternalType());
	} else {
		bound_function.function = GetScalarBinaryFunction<OP>(result_type.InternalType());
	}
	if (result_type.InternalType() != PhysicalType::INT128) {
		if (IS_SUBTRACT) {
			bound_function.statistics =
			    PropagateNumericStats<TryDecimalSubtract, SubtractPropagateStatistics, SubtractOperator>;
		} else {
			bound_function.statistics = PropagateNumericStats<TryDecimalAdd, AddPropagateStatistics, AddOperator>;
		}
	}
	return std::move(bind_data);
}

static void SerializeDecimalArithmetic(FieldWriter &writer, const FunctionData *bind_data_p,
                                       const ScalarFunction &function) {
	D_ASSERT(bind_data_p);
	auto bind_data = (DecimalArithmeticBindData *)bind_data_p;
	writer.WriteField(bind_data->check_overflow);
	writer.WriteSerializable(function.return_type);
	writer.WriteRegularSerializableList(function.arguments);
}

// TODO this is partially duplicated from the bind
template <class OP, class OPOVERFLOWCHECK, bool IS_SUBTRACT = false>
unique_ptr<FunctionData> DeserializeDecimalArithmetic(ClientContext &context, FieldReader &reader,
                                                      ScalarFunction &bound_function) {
	// re-change the function pointers
	auto check_overflow = reader.ReadRequired<bool>();
	auto return_type = reader.ReadRequiredSerializable<LogicalType, LogicalType>();
	auto arguments = reader.template ReadRequiredSerializableList<LogicalType, LogicalType>();

	if (check_overflow) {
		bound_function.function = GetScalarBinaryFunction<OPOVERFLOWCHECK>(return_type.InternalType());
	} else {
		bound_function.function = GetScalarBinaryFunction<OP>(return_type.InternalType());
	}
	bound_function.statistics = nullptr; // TODO we likely dont want to do stats prop again
	bound_function.return_type = return_type;
	bound_function.arguments = arguments;

	auto bind_data = make_uniq<DecimalArithmeticBindData>();
	bind_data->check_overflow = check_overflow;
	return std::move(bind_data);
}

unique_ptr<FunctionData> NopDecimalBind(ClientContext &context, ScalarFunction &bound_function,
                                        vector<unique_ptr<Expression>> &arguments) {
	bound_function.return_type = arguments[0]->return_type;
	bound_function.arguments[0] = arguments[0]->return_type;
	return nullptr;
}

ScalarFunction AddFun::GetFunction(const LogicalType &type) {
	D_ASSERT(type.IsNumeric());
	if (type.id() == LogicalTypeId::DECIMAL) {
		return ScalarFunction("+", {type}, type, ScalarFunction::NopFunction, NopDecimalBind);
	} else {
		return ScalarFunction("+", {type}, type, ScalarFunction::NopFunction);
	}
}

ScalarFunction AddFun::GetFunction(const LogicalType &left_type, const LogicalType &right_type) {
	if (left_type.IsNumeric() && left_type.id() == right_type.id()) {
		if (left_type.id() == LogicalTypeId::DECIMAL) {
			auto function = ScalarFunction("+", {left_type, right_type}, left_type, nullptr,
			                               BindDecimalAddSubtract<AddOperator, DecimalAddOverflowCheck>);
			function.serialize = SerializeDecimalArithmetic;
			function.deserialize = DeserializeDecimalArithmetic<AddOperator, DecimalAddOverflowCheck>;
			return function;
		} else if (left_type.IsIntegral() && left_type.id() != LogicalTypeId::HUGEINT) {
			return ScalarFunction("+", {left_type, right_type}, left_type,
			                      GetScalarIntegerFunction<AddOperatorOverflowCheck>(left_type.InternalType()), nullptr,
			                      nullptr, PropagateNumericStats<TryAddOperator, AddPropagateStatistics, AddOperator>);
		} else {
			return ScalarFunction("+", {left_type, right_type}, left_type,
			                      GetScalarBinaryFunction<AddOperator>(left_type.InternalType()));
		}
	}

	switch (left_type.id()) {
	case LogicalTypeId::DATE:
		if (right_type.id() == LogicalTypeId::INTEGER) {
			return ScalarFunction("+", {left_type, right_type}, LogicalType::DATE,
			                      ScalarFunction::BinaryFunction<date_t, int32_t, date_t, AddOperator>);
		} else if (right_type.id() == LogicalTypeId::INTERVAL) {
			return ScalarFunction("+", {left_type, right_type}, LogicalType::DATE,
			                      ScalarFunction::BinaryFunction<date_t, interval_t, date_t, AddOperator>);
		} else if (right_type.id() == LogicalTypeId::TIME) {
			return ScalarFunction("+", {left_type, right_type}, LogicalType::TIMESTAMP,
			                      ScalarFunction::BinaryFunction<date_t, dtime_t, timestamp_t, AddOperator>);
		}
		break;
	case LogicalTypeId::INTEGER:
		if (right_type.id() == LogicalTypeId::DATE) {
			return ScalarFunction("+", {left_type, right_type}, right_type,
			                      ScalarFunction::BinaryFunction<int32_t, date_t, date_t, AddOperator>);
		}
		break;
	case LogicalTypeId::INTERVAL:
		if (right_type.id() == LogicalTypeId::INTERVAL) {
			return ScalarFunction("+", {left_type, right_type}, LogicalType::INTERVAL,
			                      ScalarFunction::BinaryFunction<interval_t, interval_t, interval_t, AddOperator>);
		} else if (right_type.id() == LogicalTypeId::DATE) {
			return ScalarFunction("+", {left_type, right_type}, LogicalType::DATE,
			                      ScalarFunction::BinaryFunction<interval_t, date_t, date_t, AddOperator>);
		} else if (right_type.id() == LogicalTypeId::TIME) {
			return ScalarFunction("+", {left_type, right_type}, LogicalType::TIME,
			                      ScalarFunction::BinaryFunction<interval_t, dtime_t, dtime_t, AddTimeOperator>);
		} else if (right_type.id() == LogicalTypeId::TIMESTAMP) {
			return ScalarFunction("+", {left_type, right_type}, LogicalType::TIMESTAMP,
			                      ScalarFunction::BinaryFunction<interval_t, timestamp_t, timestamp_t, AddOperator>);
		}
		break;
	case LogicalTypeId::TIME:
		if (right_type.id() == LogicalTypeId::INTERVAL) {
			return ScalarFunction("+", {left_type, right_type}, LogicalType::TIME,
			                      ScalarFunction::BinaryFunction<dtime_t, interval_t, dtime_t, AddTimeOperator>);
		} else if (right_type.id() == LogicalTypeId::DATE) {
			return ScalarFunction("+", {left_type, right_type}, LogicalType::TIMESTAMP,
			                      ScalarFunction::BinaryFunction<dtime_t, date_t, timestamp_t, AddOperator>);
		}
		break;
	case LogicalTypeId::TIMESTAMP:
		if (right_type.id() == LogicalTypeId::INTERVAL) {
			return ScalarFunction("+", {left_type, right_type}, LogicalType::TIMESTAMP,
			                      ScalarFunction::BinaryFunction<timestamp_t, interval_t, timestamp_t, AddOperator>);
		}
		break;
	default:
		break;
	}
	// LCOV_EXCL_START
	throw NotImplementedException("AddFun for types %s, %s", EnumUtil::ToString(left_type.id()),
	                              EnumUtil::ToString(right_type.id()));
	// LCOV_EXCL_STOP
}

void AddFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunctionSet functions("+");
	for (auto &type : LogicalType::Numeric()) {
		// unary add function is a nop, but only exists for numeric types
		functions.AddFunction(GetFunction(type));
		// binary add function adds two numbers together
		functions.AddFunction(GetFunction(type, type));
	}
	// we can add integers to dates
	functions.AddFunction(GetFunction(LogicalType::DATE, LogicalType::INTEGER));
	functions.AddFunction(GetFunction(LogicalType::INTEGER, LogicalType::DATE));
	// we can add intervals together
	functions.AddFunction(GetFunction(LogicalType::INTERVAL, LogicalType::INTERVAL));
	// we can add intervals to dates/times/timestamps
	functions.AddFunction(GetFunction(LogicalType::DATE, LogicalType::INTERVAL));
	functions.AddFunction(GetFunction(LogicalType::INTERVAL, LogicalType::DATE));

	functions.AddFunction(GetFunction(LogicalType::TIME, LogicalType::INTERVAL));
	functions.AddFunction(GetFunction(LogicalType::INTERVAL, LogicalType::TIME));

	functions.AddFunction(GetFunction(LogicalType::TIMESTAMP, LogicalType::INTERVAL));
	functions.AddFunction(GetFunction(LogicalType::INTERVAL, LogicalType::TIMESTAMP));

	// we can add times to dates
	functions.AddFunction(GetFunction(LogicalType::TIME, LogicalType::DATE));
	functions.AddFunction(GetFunction(LogicalType::DATE, LogicalType::TIME));

	// we can add lists together
	functions.AddFunction(ListConcatFun::GetFunction());

	set.AddFunction(functions);

	functions.name = "add";
	set.AddFunction(functions);
}

//===--------------------------------------------------------------------===//
// - [subtract]
//===--------------------------------------------------------------------===//
struct NegateOperator {
	template <class T>
	static bool CanNegate(T input) {
		using Limits = std::numeric_limits<T>;
		return !(Limits::is_integer && Limits::is_signed && Limits::lowest() == input);
	}

	template <class TA, class TR>
	static inline TR Operation(TA input) {
		auto cast = (TR)input;
		if (!CanNegate<TR>(cast)) {
			throw OutOfRangeException("Overflow in negation of integer!");
		}
		return -cast;
	}
};

template <>
bool NegateOperator::CanNegate(float input) {
	return true;
}

template <>
bool NegateOperator::CanNegate(double input) {
	return true;
}

template <>
interval_t NegateOperator::Operation(interval_t input) {
	interval_t result;
	result.months = NegateOperator::Operation<int32_t, int32_t>(input.months);
	result.days = NegateOperator::Operation<int32_t, int32_t>(input.days);
	result.micros = NegateOperator::Operation<int64_t, int64_t>(input.micros);
	return result;
}

struct DecimalNegateBindData : public FunctionData {
	DecimalNegateBindData() : bound_type(LogicalTypeId::INVALID) {
	}

	unique_ptr<FunctionData> Copy() const override {
		auto res = make_uniq<DecimalNegateBindData>();
		res->bound_type = bound_type;
		return std::move(res);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto other = other_p.Cast<DecimalNegateBindData>();
		return other.bound_type == bound_type;
	}

	LogicalTypeId bound_type;
};

unique_ptr<FunctionData> DecimalNegateBind(ClientContext &context, ScalarFunction &bound_function,
                                           vector<unique_ptr<Expression>> &arguments) {

	auto bind_data = make_uniq<DecimalNegateBindData>();

	auto &decimal_type = arguments[0]->return_type;
	auto width = DecimalType::GetWidth(decimal_type);
	if (width <= Decimal::MAX_WIDTH_INT16) {
		bound_function.function = ScalarFunction::GetScalarUnaryFunction<NegateOperator>(LogicalTypeId::SMALLINT);
	} else if (width <= Decimal::MAX_WIDTH_INT32) {
		bound_function.function = ScalarFunction::GetScalarUnaryFunction<NegateOperator>(LogicalTypeId::INTEGER);
	} else if (width <= Decimal::MAX_WIDTH_INT64) {
		bound_function.function = ScalarFunction::GetScalarUnaryFunction<NegateOperator>(LogicalTypeId::BIGINT);
	} else {
		D_ASSERT(width <= Decimal::MAX_WIDTH_INT128);
		bound_function.function = ScalarFunction::GetScalarUnaryFunction<NegateOperator>(LogicalTypeId::HUGEINT);
	}
	decimal_type.Verify();
	bound_function.arguments[0] = decimal_type;
	bound_function.return_type = decimal_type;
	return nullptr;
}

struct NegatePropagateStatistics {
	template <class T>
	static bool Operation(LogicalType type, BaseStatistics &istats, Value &new_min, Value &new_max) {
		auto max_value = NumericStats::GetMax<T>(istats);
		auto min_value = NumericStats::GetMin<T>(istats);
		if (!NegateOperator::CanNegate<T>(min_value) || !NegateOperator::CanNegate<T>(max_value)) {
			return true;
		}
		// new min is -max
		new_min = Value::Numeric(type, NegateOperator::Operation<T, T>(max_value));
		// new max is -min
		new_max = Value::Numeric(type, NegateOperator::Operation<T, T>(min_value));
		return false;
	}
};

static unique_ptr<BaseStatistics> NegateBindStatistics(ClientContext &context, FunctionStatisticsInput &input) {
	auto &child_stats = input.child_stats;
	auto &expr = input.expr;
	D_ASSERT(child_stats.size() == 1);
	// can only propagate stats if the children have stats
	auto &istats = child_stats[0];
	Value new_min, new_max;
	bool potential_overflow = true;
	if (NumericStats::HasMinMax(istats)) {
		switch (expr.return_type.InternalType()) {
		case PhysicalType::INT8:
			potential_overflow =
			    NegatePropagateStatistics::Operation<int8_t>(expr.return_type, istats, new_min, new_max);
			break;
		case PhysicalType::INT16:
			potential_overflow =
			    NegatePropagateStatistics::Operation<int16_t>(expr.return_type, istats, new_min, new_max);
			break;
		case PhysicalType::INT32:
			potential_overflow =
			    NegatePropagateStatistics::Operation<int32_t>(expr.return_type, istats, new_min, new_max);
			break;
		case PhysicalType::INT64:
			potential_overflow =
			    NegatePropagateStatistics::Operation<int64_t>(expr.return_type, istats, new_min, new_max);
			break;
		default:
			return nullptr;
		}
	}
	if (potential_overflow) {
		new_min = Value(expr.return_type);
		new_max = Value(expr.return_type);
	}
	auto stats = NumericStats::CreateEmpty(expr.return_type);
	NumericStats::SetMin(stats, new_min);
	NumericStats::SetMax(stats, new_max);
	stats.CopyValidity(istats);
	return stats.ToUnique();
}

ScalarFunction SubtractFun::GetFunction(const LogicalType &type) {
	if (type.id() == LogicalTypeId::INTERVAL) {
		return ScalarFunction("-", {type}, type, ScalarFunction::UnaryFunction<interval_t, interval_t, NegateOperator>);
	} else if (type.id() == LogicalTypeId::DECIMAL) {
		return ScalarFunction("-", {type}, type, nullptr, DecimalNegateBind, nullptr, NegateBindStatistics);
	} else {
		D_ASSERT(type.IsNumeric());
		return ScalarFunction("-", {type}, type, ScalarFunction::GetScalarUnaryFunction<NegateOperator>(type), nullptr,
		                      nullptr, NegateBindStatistics);
	}
}

ScalarFunction SubtractFun::GetFunction(const LogicalType &left_type, const LogicalType &right_type) {
	if (left_type.IsNumeric() && left_type.id() == right_type.id()) {
		if (left_type.id() == LogicalTypeId::DECIMAL) {
			auto function =
			    ScalarFunction("-", {left_type, right_type}, left_type, nullptr,
			                   BindDecimalAddSubtract<SubtractOperator, DecimalSubtractOverflowCheck, true>);
			function.serialize = SerializeDecimalArithmetic;
			function.deserialize = DeserializeDecimalArithmetic<SubtractOperator, DecimalSubtractOverflowCheck>;
			return function;
		} else if (left_type.IsIntegral() && left_type.id() != LogicalTypeId::HUGEINT) {
			return ScalarFunction(
			    "-", {left_type, right_type}, left_type,
			    GetScalarIntegerFunction<SubtractOperatorOverflowCheck>(left_type.InternalType()), nullptr, nullptr,
			    PropagateNumericStats<TrySubtractOperator, SubtractPropagateStatistics, SubtractOperator>);

		} else {
			return ScalarFunction("-", {left_type, right_type}, left_type,
			                      GetScalarBinaryFunction<SubtractOperator>(left_type.InternalType()));
		}
	}

	switch (left_type.id()) {
	case LogicalTypeId::DATE:
		if (right_type.id() == LogicalTypeId::DATE) {
			return ScalarFunction("-", {left_type, right_type}, LogicalType::BIGINT,
			                      ScalarFunction::BinaryFunction<date_t, date_t, int64_t, SubtractOperator>);

		} else if (right_type.id() == LogicalTypeId::INTEGER) {
			return ScalarFunction("-", {left_type, right_type}, LogicalType::DATE,
			                      ScalarFunction::BinaryFunction<date_t, int32_t, date_t, SubtractOperator>);
		} else if (right_type.id() == LogicalTypeId::INTERVAL) {
			return ScalarFunction("-", {left_type, right_type}, LogicalType::DATE,
			                      ScalarFunction::BinaryFunction<date_t, interval_t, date_t, SubtractOperator>);
		}
		break;
	case LogicalTypeId::TIMESTAMP:
		if (right_type.id() == LogicalTypeId::TIMESTAMP) {
			return ScalarFunction(
			    "-", {left_type, right_type}, LogicalType::INTERVAL,
			    ScalarFunction::BinaryFunction<timestamp_t, timestamp_t, interval_t, SubtractOperator>);
		} else if (right_type.id() == LogicalTypeId::INTERVAL) {
			return ScalarFunction(
			    "-", {left_type, right_type}, LogicalType::TIMESTAMP,
			    ScalarFunction::BinaryFunction<timestamp_t, interval_t, timestamp_t, SubtractOperator>);
		}
		break;
	case LogicalTypeId::INTERVAL:
		if (right_type.id() == LogicalTypeId::INTERVAL) {
			return ScalarFunction("-", {left_type, right_type}, LogicalType::INTERVAL,
			                      ScalarFunction::BinaryFunction<interval_t, interval_t, interval_t, SubtractOperator>);
		}
		break;
	case LogicalTypeId::TIME:
		if (right_type.id() == LogicalTypeId::INTERVAL) {
			return ScalarFunction("-", {left_type, right_type}, LogicalType::TIME,
			                      ScalarFunction::BinaryFunction<dtime_t, interval_t, dtime_t, SubtractTimeOperator>);
		}
		break;
	default:
		break;
	}
	// LCOV_EXCL_START
	throw NotImplementedException("SubtractFun for types %s, %s", EnumUtil::ToString(left_type.id()),
	                              EnumUtil::ToString(right_type.id()));
	// LCOV_EXCL_STOP
}

void SubtractFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunctionSet functions("-");
	for (auto &type : LogicalType::Numeric()) {
		// unary subtract function, negates the input (i.e. multiplies by -1)
		functions.AddFunction(GetFunction(type));
		// binary subtract function "a - b", subtracts b from a
		functions.AddFunction(GetFunction(type, type));
	}
	// we can subtract dates from each other
	functions.AddFunction(GetFunction(LogicalType::DATE, LogicalType::DATE));
	// we can subtract integers from dates
	functions.AddFunction(GetFunction(LogicalType::DATE, LogicalType::INTEGER));
	// we can subtract timestamps from each other
	functions.AddFunction(GetFunction(LogicalType::TIMESTAMP, LogicalType::TIMESTAMP));
	// we can subtract intervals from each other
	functions.AddFunction(GetFunction(LogicalType::INTERVAL, LogicalType::INTERVAL));
	// we can subtract intervals from dates/times/timestamps, but not the other way around
	functions.AddFunction(GetFunction(LogicalType::DATE, LogicalType::INTERVAL));
	functions.AddFunction(GetFunction(LogicalType::TIME, LogicalType::INTERVAL));
	functions.AddFunction(GetFunction(LogicalType::TIMESTAMP, LogicalType::INTERVAL));
	// we can negate intervals
	functions.AddFunction(GetFunction(LogicalType::INTERVAL));
	set.AddFunction(functions);

	functions.name = "subtract";
	set.AddFunction(functions);
}

//===--------------------------------------------------------------------===//
// * [multiply]
//===--------------------------------------------------------------------===//
struct MultiplyPropagateStatistics {
	template <class T, class OP>
	static bool Operation(LogicalType type, BaseStatistics &lstats, BaseStatistics &rstats, Value &new_min,
	                      Value &new_max) {
		// statistics propagation on the multiplication is slightly less straightforward because of negative numbers
		// the new min/max depend on the signs of the input types
		// if both are positive the result is [lmin * rmin][lmax * rmax]
		// if lmin/lmax are negative the result is [lmin * rmax][lmax * rmin]
		// etc
		// rather than doing all this switcheroo we just multiply all combinations of lmin/lmax with rmin/rmax
		// and check what the minimum/maximum value is
		T lvals[] {NumericStats::GetMin<T>(lstats), NumericStats::GetMax<T>(lstats)};
		T rvals[] {NumericStats::GetMin<T>(rstats), NumericStats::GetMax<T>(rstats)};
		T min = NumericLimits<T>::Maximum();
		T max = NumericLimits<T>::Minimum();
		// multiplications
		for (idx_t l = 0; l < 2; l++) {
			for (idx_t r = 0; r < 2; r++) {
				T result;
				if (!OP::Operation(lvals[l], rvals[r], result)) {
					// potential overflow
					return true;
				}
				if (result < min) {
					min = result;
				}
				if (result > max) {
					max = result;
				}
			}
		}
		new_min = Value::Numeric(type, min);
		new_max = Value::Numeric(type, max);
		return false;
	}
};

unique_ptr<FunctionData> BindDecimalMultiply(ClientContext &context, ScalarFunction &bound_function,
                                             vector<unique_ptr<Expression>> &arguments) {

	auto bind_data = make_uniq<DecimalArithmeticBindData>();

	uint8_t result_width = 0, result_scale = 0;
	uint8_t max_width = 0;
	for (idx_t i = 0; i < arguments.size(); i++) {
		if (arguments[i]->return_type.id() == LogicalTypeId::UNKNOWN) {
			continue;
		}
		uint8_t width, scale;
		auto can_convert = arguments[i]->return_type.GetDecimalProperties(width, scale);
		if (!can_convert) {
			throw InternalException("Could not convert type %s to a decimal?", arguments[i]->return_type.ToString());
		}
		if (width > max_width) {
			max_width = width;
		}
		result_width += width;
		result_scale += scale;
	}
	D_ASSERT(max_width > 0);
	if (result_scale > Decimal::MAX_WIDTH_DECIMAL) {
		throw OutOfRangeException(
		    "Needed scale %d to accurately represent the multiplication result, but this is out of range of the "
		    "DECIMAL type. Max scale is %d; could not perform an accurate multiplication. Either add a cast to DOUBLE, "
		    "or add an explicit cast to a decimal with a lower scale.",
		    result_scale, Decimal::MAX_WIDTH_DECIMAL);
	}
	if (result_width > Decimal::MAX_WIDTH_INT64 && max_width <= Decimal::MAX_WIDTH_INT64 &&
	    result_scale < Decimal::MAX_WIDTH_INT64) {
		bind_data->check_overflow = true;
		result_width = Decimal::MAX_WIDTH_INT64;
	}
	if (result_width > Decimal::MAX_WIDTH_DECIMAL) {
		bind_data->check_overflow = true;
		result_width = Decimal::MAX_WIDTH_DECIMAL;
	}
	LogicalType result_type = LogicalType::DECIMAL(result_width, result_scale);
	// since our scale is the summation of our input scales, we do not need to cast to the result scale
	// however, we might need to cast to the correct internal type
	for (idx_t i = 0; i < arguments.size(); i++) {
		auto &argument_type = arguments[i]->return_type;
		if (argument_type.InternalType() == result_type.InternalType()) {
			bound_function.arguments[i] = argument_type;
		} else {
			uint8_t width, scale;
			if (!argument_type.GetDecimalProperties(width, scale)) {
				scale = 0;
			}

			bound_function.arguments[i] = LogicalType::DECIMAL(result_width, scale);
		}
	}
	result_type.Verify();
	bound_function.return_type = result_type;
	// now select the physical function to execute
	if (bind_data->check_overflow) {
		bound_function.function = GetScalarBinaryFunction<DecimalMultiplyOverflowCheck>(result_type.InternalType());
	} else {
		bound_function.function = GetScalarBinaryFunction<MultiplyOperator>(result_type.InternalType());
	}
	if (result_type.InternalType() != PhysicalType::INT128) {
		bound_function.statistics =
		    PropagateNumericStats<TryDecimalMultiply, MultiplyPropagateStatistics, MultiplyOperator>;
	}
	return std::move(bind_data);
}

void MultiplyFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunctionSet functions("*");
	for (auto &type : LogicalType::Numeric()) {
		if (type.id() == LogicalTypeId::DECIMAL) {
			ScalarFunction function({type, type}, type, nullptr, BindDecimalMultiply);
			function.serialize = SerializeDecimalArithmetic;
			function.deserialize = DeserializeDecimalArithmetic<MultiplyOperator, DecimalMultiplyOverflowCheck>;
			functions.AddFunction(function);
		} else if (TypeIsIntegral(type.InternalType()) && type.id() != LogicalTypeId::HUGEINT) {
			functions.AddFunction(ScalarFunction(
			    {type, type}, type, GetScalarIntegerFunction<MultiplyOperatorOverflowCheck>(type.InternalType()),
			    nullptr, nullptr,
			    PropagateNumericStats<TryMultiplyOperator, MultiplyPropagateStatistics, MultiplyOperator>));
		} else {
			functions.AddFunction(
			    ScalarFunction({type, type}, type, GetScalarBinaryFunction<MultiplyOperator>(type.InternalType())));
		}
	}
	functions.AddFunction(
	    ScalarFunction({LogicalType::INTERVAL, LogicalType::BIGINT}, LogicalType::INTERVAL,
	                   ScalarFunction::BinaryFunction<interval_t, int64_t, interval_t, MultiplyOperator>));
	functions.AddFunction(
	    ScalarFunction({LogicalType::BIGINT, LogicalType::INTERVAL}, LogicalType::INTERVAL,
	                   ScalarFunction::BinaryFunction<int64_t, interval_t, interval_t, MultiplyOperator>));
	set.AddFunction(functions);

	functions.name = "multiply";
	set.AddFunction(functions);
}

//===--------------------------------------------------------------------===//
// / [divide]
//===--------------------------------------------------------------------===//
template <>
float DivideOperator::Operation(float left, float right) {
	auto result = left / right;
	return result;
}

template <>
double DivideOperator::Operation(double left, double right) {
	auto result = left / right;
	return result;
}

template <>
hugeint_t DivideOperator::Operation(hugeint_t left, hugeint_t right) {
	if (right.lower == 0 && right.upper == 0) {
		throw InternalException("Hugeint division by zero!");
	}
	return left / right;
}

template <>
interval_t DivideOperator::Operation(interval_t left, int64_t right) {
	left.days /= right;
	left.months /= right;
	left.micros /= right;
	return left;
}

struct BinaryNumericDivideWrapper {
	template <class FUNC, class OP, class LEFT_TYPE, class RIGHT_TYPE, class RESULT_TYPE>
	static inline RESULT_TYPE Operation(FUNC fun, LEFT_TYPE left, RIGHT_TYPE right, ValidityMask &mask, idx_t idx) {
		if (left == NumericLimits<LEFT_TYPE>::Minimum() && right == -1) {
			throw OutOfRangeException("Overflow in division of %d / %d", left, right);
		} else if (right == 0) {
			mask.SetInvalid(idx);
			return left;
		} else {
			return OP::template Operation<LEFT_TYPE, RIGHT_TYPE, RESULT_TYPE>(left, right);
		}
	}

	static bool AddsNulls() {
		return true;
	}
};

struct BinaryZeroIsNullWrapper {
	template <class FUNC, class OP, class LEFT_TYPE, class RIGHT_TYPE, class RESULT_TYPE>
	static inline RESULT_TYPE Operation(FUNC fun, LEFT_TYPE left, RIGHT_TYPE right, ValidityMask &mask, idx_t idx) {
		if (right == 0) {
			mask.SetInvalid(idx);
			return left;
		} else {
			return OP::template Operation<LEFT_TYPE, RIGHT_TYPE, RESULT_TYPE>(left, right);
		}
	}

	static bool AddsNulls() {
		return true;
	}
};

struct BinaryZeroIsNullHugeintWrapper {
	template <class FUNC, class OP, class LEFT_TYPE, class RIGHT_TYPE, class RESULT_TYPE>
	static inline RESULT_TYPE Operation(FUNC fun, LEFT_TYPE left, RIGHT_TYPE right, ValidityMask &mask, idx_t idx) {
		if (right.upper == 0 && right.lower == 0) {
			mask.SetInvalid(idx);
			return left;
		} else {
			return OP::template Operation<LEFT_TYPE, RIGHT_TYPE, RESULT_TYPE>(left, right);
		}
	}

	static bool AddsNulls() {
		return true;
	}
};

template <class TA, class TB, class TC, class OP, class ZWRAPPER = BinaryZeroIsNullWrapper>
static void BinaryScalarFunctionIgnoreZero(DataChunk &input, ExpressionState &state, Vector &result) {
	BinaryExecutor::Execute<TA, TB, TC, OP, ZWRAPPER>(input.data[0], input.data[1], result, input.size());
}

template <class OP>
static scalar_function_t GetBinaryFunctionIgnoreZero(const LogicalType &type) {
	switch (type.id()) {
	case LogicalTypeId::TINYINT:
		return BinaryScalarFunctionIgnoreZero<int8_t, int8_t, int8_t, OP, BinaryNumericDivideWrapper>;
	case LogicalTypeId::SMALLINT:
		return BinaryScalarFunctionIgnoreZero<int16_t, int16_t, int16_t, OP, BinaryNumericDivideWrapper>;
	case LogicalTypeId::INTEGER:
		return BinaryScalarFunctionIgnoreZero<int32_t, int32_t, int32_t, OP, BinaryNumericDivideWrapper>;
	case LogicalTypeId::BIGINT:
		return BinaryScalarFunctionIgnoreZero<int64_t, int64_t, int64_t, OP, BinaryNumericDivideWrapper>;
	case LogicalTypeId::UTINYINT:
		return BinaryScalarFunctionIgnoreZero<uint8_t, uint8_t, uint8_t, OP>;
	case LogicalTypeId::USMALLINT:
		return BinaryScalarFunctionIgnoreZero<uint16_t, uint16_t, uint16_t, OP>;
	case LogicalTypeId::UINTEGER:
		return BinaryScalarFunctionIgnoreZero<uint32_t, uint32_t, uint32_t, OP>;
	case LogicalTypeId::UBIGINT:
		return BinaryScalarFunctionIgnoreZero<uint64_t, uint64_t, uint64_t, OP>;
	case LogicalTypeId::HUGEINT:
		return BinaryScalarFunctionIgnoreZero<hugeint_t, hugeint_t, hugeint_t, OP, BinaryZeroIsNullHugeintWrapper>;
	case LogicalTypeId::FLOAT:
		return BinaryScalarFunctionIgnoreZero<float, float, float, OP>;
	case LogicalTypeId::DOUBLE:
		return BinaryScalarFunctionIgnoreZero<double, double, double, OP>;
	default:
		throw NotImplementedException("Unimplemented type for GetScalarUnaryFunction");
	}
}

void DivideFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunctionSet fp_divide("/");
	fp_divide.AddFunction(ScalarFunction({LogicalType::FLOAT, LogicalType::FLOAT}, LogicalType::FLOAT,
	                                     GetBinaryFunctionIgnoreZero<DivideOperator>(LogicalType::FLOAT)));
	fp_divide.AddFunction(ScalarFunction({LogicalType::DOUBLE, LogicalType::DOUBLE}, LogicalType::DOUBLE,
	                                     GetBinaryFunctionIgnoreZero<DivideOperator>(LogicalType::DOUBLE)));
	fp_divide.AddFunction(
	    ScalarFunction({LogicalType::INTERVAL, LogicalType::BIGINT}, LogicalType::INTERVAL,
	                   BinaryScalarFunctionIgnoreZero<interval_t, int64_t, interval_t, DivideOperator>));
	set.AddFunction(fp_divide);

	ScalarFunctionSet full_divide("//");
	for (auto &type : LogicalType::Numeric()) {
		if (type.id() == LogicalTypeId::DECIMAL) {
			continue;
		} else {
			full_divide.AddFunction(
			    ScalarFunction({type, type}, type, GetBinaryFunctionIgnoreZero<DivideOperator>(type)));
		}
	}
	set.AddFunction(full_divide);

	full_divide.name = "divide";
	set.AddFunction(full_divide);
}

//===--------------------------------------------------------------------===//
// % [modulo]
//===--------------------------------------------------------------------===//
template <>
float ModuloOperator::Operation(float left, float right) {
	D_ASSERT(right != 0);
	auto result = std::fmod(left, right);
	return result;
}

template <>
double ModuloOperator::Operation(double left, double right) {
	D_ASSERT(right != 0);
	auto result = std::fmod(left, right);
	return result;
}

template <>
hugeint_t ModuloOperator::Operation(hugeint_t left, hugeint_t right) {
	if (right.lower == 0 && right.upper == 0) {
		throw InternalException("Hugeint division by zero!");
	}
	return left % right;
}

void ModFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunctionSet functions("%");
	for (auto &type : LogicalType::Numeric()) {
		if (type.id() == LogicalTypeId::DECIMAL) {
			continue;
		} else {
			functions.AddFunction(
			    ScalarFunction({type, type}, type, GetBinaryFunctionIgnoreZero<ModuloOperator>(type)));
		}
	}
	set.AddFunction(functions);
	functions.name = "mod";
	set.AddFunction(functions);
}

} // namespace duckdb







#include <limits>
#include <algorithm>

namespace duckdb {

//===--------------------------------------------------------------------===//
// * [multiply]
//===--------------------------------------------------------------------===//
template <>
float MultiplyOperator::Operation(float left, float right) {
	auto result = left * right;
	return result;
}

template <>
double MultiplyOperator::Operation(double left, double right) {
	auto result = left * right;
	return result;
}

template <>
interval_t MultiplyOperator::Operation(interval_t left, int64_t right) {
	left.months = MultiplyOperatorOverflowCheck::Operation<int32_t, int32_t, int32_t>(left.months, right);
	left.days = MultiplyOperatorOverflowCheck::Operation<int32_t, int32_t, int32_t>(left.days, right);
	left.micros = MultiplyOperatorOverflowCheck::Operation<int64_t, int64_t, int64_t>(left.micros, right);
	return left;
}

template <>
interval_t MultiplyOperator::Operation(int64_t left, interval_t right) {
	return MultiplyOperator::Operation<interval_t, int64_t, interval_t>(right, left);
}

//===--------------------------------------------------------------------===//
// * [multiply] with overflow check
//===--------------------------------------------------------------------===//
struct OverflowCheckedMultiply {
	template <class SRCTYPE, class UTYPE>
	static inline bool Operation(SRCTYPE left, SRCTYPE right, SRCTYPE &result) {
		UTYPE uresult = MultiplyOperator::Operation<UTYPE, UTYPE, UTYPE>(UTYPE(left), UTYPE(right));
		if (uresult < NumericLimits<SRCTYPE>::Minimum() || uresult > NumericLimits<SRCTYPE>::Maximum()) {
			return false;
		}
		result = SRCTYPE(uresult);
		return true;
	}
};

template <>
bool TryMultiplyOperator::Operation(uint8_t left, uint8_t right, uint8_t &result) {
	return OverflowCheckedMultiply::Operation<uint8_t, uint16_t>(left, right, result);
}
template <>
bool TryMultiplyOperator::Operation(uint16_t left, uint16_t right, uint16_t &result) {
	return OverflowCheckedMultiply::Operation<uint16_t, uint32_t>(left, right, result);
}
template <>
bool TryMultiplyOperator::Operation(uint32_t left, uint32_t right, uint32_t &result) {
	return OverflowCheckedMultiply::Operation<uint32_t, uint64_t>(left, right, result);
}
template <>
bool TryMultiplyOperator::Operation(uint64_t left, uint64_t right, uint64_t &result) {
	if (left > right) {
		std::swap(left, right);
	}
	if (left > NumericLimits<uint32_t>::Maximum()) {
		return false;
	}
	uint32_t c = right >> 32;
	uint32_t d = NumericLimits<uint32_t>::Maximum() & right;
	uint64_t r = left * c;
	uint64_t s = left * d;
	if (r > NumericLimits<uint32_t>::Maximum()) {
		return false;
	}
	r <<= 32;
	if (NumericLimits<uint64_t>::Maximum() - s < r) {
		return false;
	}
	return OverflowCheckedMultiply::Operation<uint64_t, uint64_t>(left, right, result);
}

template <>
bool TryMultiplyOperator::Operation(int8_t left, int8_t right, int8_t &result) {
	return OverflowCheckedMultiply::Operation<int8_t, int16_t>(left, right, result);
}

template <>
bool TryMultiplyOperator::Operation(int16_t left, int16_t right, int16_t &result) {
	return OverflowCheckedMultiply::Operation<int16_t, int32_t>(left, right, result);
}

template <>
bool TryMultiplyOperator::Operation(int32_t left, int32_t right, int32_t &result) {
	return OverflowCheckedMultiply::Operation<int32_t, int64_t>(left, right, result);
}

template <>
bool TryMultiplyOperator::Operation(int64_t left, int64_t right, int64_t &result) {
#if (__GNUC__ >= 5) || defined(__clang__)
	if (__builtin_mul_overflow(left, right, &result)) {
		return false;
	}
#else
	if (left == std::numeric_limits<int64_t>::min()) {
		if (right == 0) {
			result = 0;
			return true;
		}
		if (right == 1) {
			result = left;
			return true;
		}
		return false;
	}
	if (right == std::numeric_limits<int64_t>::min()) {
		if (left == 0) {
			result = 0;
			return true;
		}
		if (left == 1) {
			result = right;
			return true;
		}
		return false;
	}
	uint64_t left_non_negative = uint64_t(std::abs(left));
	uint64_t right_non_negative = uint64_t(std::abs(right));
	// split values into 2 32-bit parts
	uint64_t left_high_bits = left_non_negative >> 32;
	uint64_t left_low_bits = left_non_negative & 0xffffffff;
	uint64_t right_high_bits = right_non_negative >> 32;
	uint64_t right_low_bits = right_non_negative & 0xffffffff;

	// check the high bits of both
	// the high bits define the overflow
	if (left_high_bits == 0) {
		if (right_high_bits != 0) {
			// only the right has high bits set
			// multiply the high bits of right with the low bits of left
			// multiply the low bits, and carry any overflow to the high bits
			// then check for any overflow
			auto low_low = left_low_bits * right_low_bits;
			auto low_high = left_low_bits * right_high_bits;
			auto high_bits = low_high + (low_low >> 32);
			if (high_bits & 0xffffff80000000) {
				// there is! abort
				return false;
			}
		}
	} else if (right_high_bits == 0) {
		// only the left has high bits set
		// multiply the high bits of left with the low bits of right
		// multiply the low bits, and carry any overflow to the high bits
		// then check for any overflow
		auto low_low = left_low_bits * right_low_bits;
		auto high_low = left_high_bits * right_low_bits;
		auto high_bits = high_low + (low_low >> 32);
		if (high_bits & 0xffffff80000000) {
			// there is! abort
			return false;
		}
	} else {
		// both left and right have high bits set: guaranteed overflow
		// abort!
		return false;
	}
	// now we know that there is no overflow, we can just perform the multiplication
	result = left * right;
#endif
	return true;
}

template <>
bool TryMultiplyOperator::Operation(hugeint_t left, hugeint_t right, hugeint_t &result) {
	return Hugeint::TryMultiply(left, right, result);
}

//===--------------------------------------------------------------------===//
// multiply  decimal with overflow check
//===--------------------------------------------------------------------===//
template <class T, T min, T max>
bool TryDecimalMultiplyTemplated(T left, T right, T &result) {
	if (!TryMultiplyOperator::Operation(left, right, result) || result < min || result > max) {
		return false;
	}
	return true;
}

template <>
bool TryDecimalMultiply::Operation(int16_t left, int16_t right, int16_t &result) {
	return TryDecimalMultiplyTemplated<int16_t, -9999, 9999>(left, right, result);
}

template <>
bool TryDecimalMultiply::Operation(int32_t left, int32_t right, int32_t &result) {
	return TryDecimalMultiplyTemplated<int32_t, -999999999, 999999999>(left, right, result);
}

template <>
bool TryDecimalMultiply::Operation(int64_t left, int64_t right, int64_t &result) {
	return TryDecimalMultiplyTemplated<int64_t, -999999999999999999, 999999999999999999>(left, right, result);
}

template <>
bool TryDecimalMultiply::Operation(hugeint_t left, hugeint_t right, hugeint_t &result) {
	result = left * right;
	if (result <= -Hugeint::POWERS_OF_TEN[38] || result >= Hugeint::POWERS_OF_TEN[38]) {
		return false;
	}
	return true;
}

template <>
hugeint_t DecimalMultiplyOverflowCheck::Operation(hugeint_t left, hugeint_t right) {
	hugeint_t result;
	if (!TryDecimalMultiply::Operation(left, right, result)) {
		throw OutOfRangeException("Overflow in multiplication of DECIMAL(38) (%s * %s). You might want to add an "
		                          "explicit cast to a decimal with a smaller scale.",
		                          left.ToString(), right.ToString());
	}
	return result;
}

} // namespace duckdb









namespace duckdb {

//===--------------------------------------------------------------------===//
// - [subtract]
//===--------------------------------------------------------------------===//
template <>
float SubtractOperator::Operation(float left, float right) {
	auto result = left - right;
	return result;
}

template <>
double SubtractOperator::Operation(double left, double right) {
	auto result = left - right;
	return result;
}

template <>
int64_t SubtractOperator::Operation(date_t left, date_t right) {
	return int64_t(left.days) - int64_t(right.days);
}

template <>
date_t SubtractOperator::Operation(date_t left, int32_t right) {
	if (!Date::IsFinite(left)) {
		return left;
	}
	int32_t days;
	if (!TrySubtractOperator::Operation(left.days, right, days)) {
		throw OutOfRangeException("Date out of range");
	}

	date_t result(days);
	if (!Date::IsFinite(result)) {
		throw OutOfRangeException("Date out of range");
	}
	return result;
}

template <>
interval_t SubtractOperator::Operation(interval_t left, interval_t right) {
	interval_t result;
	result.months = left.months - right.months;
	result.days = left.days - right.days;
	result.micros = left.micros - right.micros;
	return result;
}

template <>
date_t SubtractOperator::Operation(date_t left, interval_t right) {
	return AddOperator::Operation<date_t, interval_t, date_t>(left, Interval::Invert(right));
}

template <>
timestamp_t SubtractOperator::Operation(timestamp_t left, interval_t right) {
	return AddOperator::Operation<timestamp_t, interval_t, timestamp_t>(left, Interval::Invert(right));
}

template <>
interval_t SubtractOperator::Operation(timestamp_t left, timestamp_t right) {
	return Interval::GetDifference(left, right);
}

//===--------------------------------------------------------------------===//
// - [subtract] with overflow check
//===--------------------------------------------------------------------===//
struct OverflowCheckedSubtract {
	template <class SRCTYPE, class UTYPE>
	static inline bool Operation(SRCTYPE left, SRCTYPE right, SRCTYPE &result) {
		UTYPE uresult = SubtractOperator::Operation<UTYPE, UTYPE, UTYPE>(UTYPE(left), UTYPE(right));
		if (uresult < NumericLimits<SRCTYPE>::Minimum() || uresult > NumericLimits<SRCTYPE>::Maximum()) {
			return false;
		}
		result = SRCTYPE(uresult);
		return true;
	}
};

template <>
bool TrySubtractOperator::Operation(uint8_t left, uint8_t right, uint8_t &result) {
	if (right > left) {
		return false;
	}
	return OverflowCheckedSubtract::Operation<uint8_t, uint16_t>(left, right, result);
}

template <>
bool TrySubtractOperator::Operation(uint16_t left, uint16_t right, uint16_t &result) {
	if (right > left) {
		return false;
	}
	return OverflowCheckedSubtract::Operation<uint16_t, uint32_t>(left, right, result);
}

template <>
bool TrySubtractOperator::Operation(uint32_t left, uint32_t right, uint32_t &result) {
	if (right > left) {
		return false;
	}
	return OverflowCheckedSubtract::Operation<uint32_t, uint64_t>(left, right, result);
}

template <>
bool TrySubtractOperator::Operation(uint64_t left, uint64_t right, uint64_t &result) {
	if (right > left) {
		return false;
	}
	return OverflowCheckedSubtract::Operation<uint64_t, uint64_t>(left, right, result);
}

template <>
bool TrySubtractOperator::Operation(int8_t left, int8_t right, int8_t &result) {
	return OverflowCheckedSubtract::Operation<int8_t, int16_t>(left, right, result);
}

template <>
bool TrySubtractOperator::Operation(int16_t left, int16_t right, int16_t &result) {
	return OverflowCheckedSubtract::Operation<int16_t, int32_t>(left, right, result);
}

template <>
bool TrySubtractOperator::Operation(int32_t left, int32_t right, int32_t &result) {
	return OverflowCheckedSubtract::Operation<int32_t, int64_t>(left, right, result);
}

template <>
bool TrySubtractOperator::Operation(int64_t left, int64_t right, int64_t &result) {
#if (__GNUC__ >= 5) || defined(__clang__)
	if (__builtin_sub_overflow(left, right, &result)) {
		return false;
	}
#else
	if (right < 0) {
		if (NumericLimits<int64_t>::Maximum() + right < left) {
			return false;
		}
	} else {
		if (NumericLimits<int64_t>::Minimum() + right > left) {
			return false;
		}
	}
	result = left - right;
#endif
	return true;
}

template <>
bool TrySubtractOperator::Operation(hugeint_t left, hugeint_t right, hugeint_t &result) {
	result = left;
	return Hugeint::SubtractInPlace(result, right);
}

//===--------------------------------------------------------------------===//
// subtract decimal with overflow check
//===--------------------------------------------------------------------===//
template <class T, T min, T max>
bool TryDecimalSubtractTemplated(T left, T right, T &result) {
	if (right < 0) {
		if (max + right < left) {
			return false;
		}
	} else {
		if (min + right > left) {
			return false;
		}
	}
	result = left - right;
	return true;
}

template <>
bool TryDecimalSubtract::Operation(int16_t left, int16_t right, int16_t &result) {
	return TryDecimalSubtractTemplated<int16_t, -9999, 9999>(left, right, result);
}

template <>
bool TryDecimalSubtract::Operation(int32_t left, int32_t right, int32_t &result) {
	return TryDecimalSubtractTemplated<int32_t, -999999999, 999999999>(left, right, result);
}

template <>
bool TryDecimalSubtract::Operation(int64_t left, int64_t right, int64_t &result) {
	return TryDecimalSubtractTemplated<int64_t, -999999999999999999, 999999999999999999>(left, right, result);
}

template <>
bool TryDecimalSubtract::Operation(hugeint_t left, hugeint_t right, hugeint_t &result) {
	result = left - right;
	if (result <= -Hugeint::POWERS_OF_TEN[38] || result >= Hugeint::POWERS_OF_TEN[38]) {
		return false;
	}
	return true;
}

template <>
hugeint_t DecimalSubtractOverflowCheck::Operation(hugeint_t left, hugeint_t right) {
	hugeint_t result;
	if (!TryDecimalSubtract::Operation(left, right, result)) {
		throw OutOfRangeException("Overflow in subtract of DECIMAL(38) (%s - %s);", left.ToString(), right.ToString());
	}
	return result;
}

//===--------------------------------------------------------------------===//
// subtract time operator
//===--------------------------------------------------------------------===//
template <>
dtime_t SubtractTimeOperator::Operation(dtime_t left, interval_t right) {
	right.micros = -right.micros;
	return AddTimeOperator::Operation<dtime_t, interval_t, dtime_t>(left, right);
}

} // namespace duckdb



namespace duckdb {

void BuiltinFunctions::RegisterOperators() {
	Register<AddFun>();
	Register<SubtractFun>();
	Register<MultiplyFun>();
	Register<DivideFun>();
	Register<ModFun>();
}

} // namespace duckdb


namespace duckdb {

void BuiltinFunctions::RegisterPragmaFunctions() {
	Register<PragmaQueries>();
	Register<PragmaFunctions>();
}

} // namespace duckdb














namespace duckdb {

struct NextvalBindData : public FunctionData {
	explicit NextvalBindData(optional_ptr<SequenceCatalogEntry> sequence) : sequence(sequence) {
	}

	//! The sequence to use for the nextval computation; only if the sequence is a constant
	optional_ptr<SequenceCatalogEntry> sequence;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<NextvalBindData>(sequence);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<NextvalBindData>();
		return sequence == other.sequence;
	}
};

struct CurrentSequenceValueOperator {
	static int64_t Operation(DuckTransaction &transaction, SequenceCatalogEntry &seq) {
		lock_guard<mutex> seqlock(seq.lock);
		int64_t result;
		if (seq.usage_count == 0u) {
			throw SequenceException("currval: sequence is not yet defined in this session");
		}
		result = seq.last_value;
		return result;
	}
};

struct NextSequenceValueOperator {
	static int64_t Operation(DuckTransaction &transaction, SequenceCatalogEntry &seq) {
		lock_guard<mutex> seqlock(seq.lock);
		int64_t result;
		result = seq.counter;
		bool overflow = !TryAddOperator::Operation(seq.counter, seq.increment, seq.counter);
		if (seq.cycle) {
			if (overflow) {
				seq.counter = seq.increment < 0 ? seq.max_value : seq.min_value;
			} else if (seq.counter < seq.min_value) {
				seq.counter = seq.max_value;
			} else if (seq.counter > seq.max_value) {
				seq.counter = seq.min_value;
			}
		} else {
			if (result < seq.min_value || (overflow && seq.increment < 0)) {
				throw SequenceException("nextval: reached minimum value of sequence \"%s\" (%lld)", seq.name,
				                        seq.min_value);
			}
			if (result > seq.max_value || overflow) {
				throw SequenceException("nextval: reached maximum value of sequence \"%s\" (%lld)", seq.name,
				                        seq.max_value);
			}
		}
		seq.last_value = result;
		seq.usage_count++;
		if (!seq.temporary) {
			transaction.sequence_usage[&seq] = SequenceValue(seq.usage_count, seq.counter);
		}
		return result;
	}
};

SequenceCatalogEntry &BindSequence(ClientContext &context, const string &name) {
	auto qname = QualifiedName::Parse(name);
	// fetch the sequence from the catalog
	Binder::BindSchemaOrCatalog(context, qname.catalog, qname.schema);
	return Catalog::GetEntry<SequenceCatalogEntry>(context, qname.catalog, qname.schema, qname.name);
}

template <class OP>
static void NextValFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &info = func_expr.bind_info->Cast<NextvalBindData>();
	auto &input = args.data[0];

	auto &context = state.GetContext();
	if (info.sequence) {
		auto &sequence = *info.sequence;
		auto &transaction = DuckTransaction::Get(context, sequence.catalog);
		// sequence to use is hard coded
		// increment the sequence
		result.SetVectorType(VectorType::FLAT_VECTOR);
		auto result_data = FlatVector::GetData<int64_t>(result);
		for (idx_t i = 0; i < args.size(); i++) {
			// get the next value from the sequence
			result_data[i] = OP::Operation(transaction, sequence);
		}
	} else {
		// sequence to use comes from the input
		UnaryExecutor::Execute<string_t, int64_t>(input, result, args.size(), [&](string_t value) {
			// fetch the sequence from the catalog
			auto &sequence = BindSequence(context, value.GetString());
			// finally get the next value from the sequence
			auto &transaction = DuckTransaction::Get(context, sequence.catalog);
			return OP::Operation(transaction, sequence);
		});
	}
}

static unique_ptr<FunctionData> NextValBind(ClientContext &context, ScalarFunction &bound_function,
                                            vector<unique_ptr<Expression>> &arguments) {
	optional_ptr<SequenceCatalogEntry> sequence;
	if (arguments[0]->IsFoldable()) {
		// parameter to nextval function is a foldable constant
		// evaluate the constant and perform the catalog lookup already
		auto seqname = ExpressionExecutor::EvaluateScalar(context, *arguments[0]);
		if (!seqname.IsNull()) {
			sequence = &BindSequence(context, seqname.ToString());
		}
	}
	return make_uniq<NextvalBindData>(sequence);
}

static void NextValDependency(BoundFunctionExpression &expr, DependencyList &dependencies) {
	auto &info = expr.bind_info->Cast<NextvalBindData>();
	if (info.sequence) {
		dependencies.AddDependency(*info.sequence);
	}
}

void NextvalFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunction next_val("nextval", {LogicalType::VARCHAR}, LogicalType::BIGINT,
	                        NextValFunction<NextSequenceValueOperator>, NextValBind, NextValDependency);
	next_val.side_effects = FunctionSideEffects::HAS_SIDE_EFFECTS;
	set.AddFunction(next_val);
}

void CurrvalFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunction curr_val("currval", {LogicalType::VARCHAR}, LogicalType::BIGINT,
	                        NextValFunction<CurrentSequenceValueOperator>, NextValBind, NextValDependency);
	curr_val.side_effects = FunctionSideEffects::HAS_SIDE_EFFECTS;
	set.AddFunction(curr_val);
}

} // namespace duckdb


namespace duckdb {

void BuiltinFunctions::RegisterSequenceFunctions() {
	Register<NextvalFun>();
	Register<CurrvalFun>();
}

} // namespace duckdb








namespace duckdb {

idx_t StrfTimepecifierSize(StrTimeSpecifier specifier) {
	switch (specifier) {
	case StrTimeSpecifier::ABBREVIATED_WEEKDAY_NAME:
	case StrTimeSpecifier::ABBREVIATED_MONTH_NAME:
		return 3;
	case StrTimeSpecifier::WEEKDAY_DECIMAL:
		return 1;
	case StrTimeSpecifier::DAY_OF_MONTH_PADDED:
	case StrTimeSpecifier::MONTH_DECIMAL_PADDED:
	case StrTimeSpecifier::YEAR_WITHOUT_CENTURY_PADDED:
	case StrTimeSpecifier::HOUR_24_PADDED:
	case StrTimeSpecifier::HOUR_12_PADDED:
	case StrTimeSpecifier::MINUTE_PADDED:
	case StrTimeSpecifier::SECOND_PADDED:
	case StrTimeSpecifier::AM_PM:
	case StrTimeSpecifier::WEEK_NUMBER_PADDED_SUN_FIRST:
	case StrTimeSpecifier::WEEK_NUMBER_PADDED_MON_FIRST:
		return 2;
	case StrTimeSpecifier::MICROSECOND_PADDED:
		return 6;
	case StrTimeSpecifier::MILLISECOND_PADDED:
		return 3;
	case StrTimeSpecifier::DAY_OF_YEAR_PADDED:
		return 3;
	default:
		return 0;
	}
}

void StrTimeFormat::AddLiteral(string literal) {
	constant_size += literal.size();
	literals.push_back(std::move(literal));
}

void StrTimeFormat::AddFormatSpecifier(string preceding_literal, StrTimeSpecifier specifier) {
	AddLiteral(std::move(preceding_literal));
	specifiers.push_back(specifier);
}

void StrfTimeFormat::AddFormatSpecifier(string preceding_literal, StrTimeSpecifier specifier) {
	is_date_specifier.push_back(IsDateSpecifier(specifier));
	idx_t specifier_size = StrfTimepecifierSize(specifier);
	if (specifier_size == 0) {
		// variable length specifier
		var_length_specifiers.push_back(specifier);
	} else {
		// constant size specifier
		constant_size += specifier_size;
	}
	StrTimeFormat::AddFormatSpecifier(std::move(preceding_literal), specifier);
}

idx_t StrfTimeFormat::GetSpecifierLength(StrTimeSpecifier specifier, date_t date, dtime_t time, int32_t utc_offset,
                                         const char *tz_name) {
	switch (specifier) {
	case StrTimeSpecifier::FULL_WEEKDAY_NAME:
		return Date::DAY_NAMES[Date::ExtractISODayOfTheWeek(date) % 7].GetSize();
	case StrTimeSpecifier::FULL_MONTH_NAME:
		return Date::MONTH_NAMES[Date::ExtractMonth(date) - 1].GetSize();
	case StrTimeSpecifier::YEAR_DECIMAL: {
		auto year = Date::ExtractYear(date);
		// Be consistent with WriteStandardSpecifier
		if (0 <= year && year <= 9999) {
			return 4;
		} else {
			return NumericHelper::SignedLength<int32_t, uint32_t>(year);
		}
	}
	case StrTimeSpecifier::MONTH_DECIMAL: {
		idx_t len = 1;
		auto month = Date::ExtractMonth(date);
		len += month >= 10;
		return len;
	}
	case StrTimeSpecifier::UTC_OFFSET:
		// ±HH or ±HH:MM
		return (utc_offset % 60) ? 6 : 3;
	case StrTimeSpecifier::TZ_NAME:
		if (tz_name) {
			return strlen(tz_name);
		}
		// empty for now
		return 0;
	case StrTimeSpecifier::HOUR_24_DECIMAL:
	case StrTimeSpecifier::HOUR_12_DECIMAL:
	case StrTimeSpecifier::MINUTE_DECIMAL:
	case StrTimeSpecifier::SECOND_DECIMAL: {
		// time specifiers
		idx_t len = 1;
		int32_t hour, min, sec, msec;
		Time::Convert(time, hour, min, sec, msec);
		switch (specifier) {
		case StrTimeSpecifier::HOUR_24_DECIMAL:
			len += hour >= 10;
			break;
		case StrTimeSpecifier::HOUR_12_DECIMAL:
			hour = hour % 12;
			if (hour == 0) {
				hour = 12;
			}
			len += hour >= 10;
			break;
		case StrTimeSpecifier::MINUTE_DECIMAL:
			len += min >= 10;
			break;
		case StrTimeSpecifier::SECOND_DECIMAL:
			len += sec >= 10;
			break;
		default:
			throw InternalException("Time specifier mismatch");
		}
		return len;
	}
	case StrTimeSpecifier::DAY_OF_MONTH:
		return NumericHelper::UnsignedLength<uint32_t>(Date::ExtractDay(date));
	case StrTimeSpecifier::DAY_OF_YEAR_DECIMAL:
		return NumericHelper::UnsignedLength<uint32_t>(Date::ExtractDayOfTheYear(date));
	case StrTimeSpecifier::YEAR_WITHOUT_CENTURY:
		return NumericHelper::UnsignedLength<uint32_t>(AbsValue(Date::ExtractYear(date)) % 100);
	default:
		throw InternalException("Unimplemented specifier for GetSpecifierLength");
	}
}

//! Returns the total length of the date formatted by this format specifier
idx_t StrfTimeFormat::GetLength(date_t date, dtime_t time, int32_t utc_offset, const char *tz_name) {
	idx_t size = constant_size;
	if (!var_length_specifiers.empty()) {
		for (auto &specifier : var_length_specifiers) {
			size += GetSpecifierLength(specifier, date, time, utc_offset, tz_name);
		}
	}
	return size;
}

char *StrfTimeFormat::WriteString(char *target, const string_t &str) {
	idx_t size = str.GetSize();
	memcpy(target, str.GetData(), size);
	return target + size;
}

// write a value in the range of 0..99 unpadded (e.g. "1", "2", ... "98", "99")
char *StrfTimeFormat::Write2(char *target, uint8_t value) {
	D_ASSERT(value < 100);
	if (value >= 10) {
		return WritePadded2(target, value);
	} else {
		*target = char(uint8_t('0') + value);
		return target + 1;
	}
}

// write a value in the range of 0..99 padded to 2 digits
char *StrfTimeFormat::WritePadded2(char *target, uint32_t value) {
	D_ASSERT(value < 100);
	auto index = static_cast<unsigned>(value * 2);
	*target++ = duckdb_fmt::internal::data::digits[index];
	*target++ = duckdb_fmt::internal::data::digits[index + 1];
	return target;
}

// write a value in the range of 0..999 padded
char *StrfTimeFormat::WritePadded3(char *target, uint32_t value) {
	D_ASSERT(value < 1000);
	if (value >= 100) {
		WritePadded2(target + 1, value % 100);
		*target = char(uint8_t('0') + value / 100);
		return target + 3;
	} else {
		*target = '0';
		target++;
		return WritePadded2(target, value);
	}
}

// write a value in the range of 0..999999 padded to 6 digits
char *StrfTimeFormat::WritePadded(char *target, uint32_t value, size_t padding) {
	D_ASSERT(padding % 2 == 0);
	for (size_t i = 0; i < padding / 2; i++) {
		int decimals = value % 100;
		WritePadded2(target + padding - 2 * (i + 1), decimals);
		value /= 100;
	}
	return target + padding;
}

bool StrfTimeFormat::IsDateSpecifier(StrTimeSpecifier specifier) {
	switch (specifier) {
	case StrTimeSpecifier::ABBREVIATED_WEEKDAY_NAME:
	case StrTimeSpecifier::FULL_WEEKDAY_NAME:
	case StrTimeSpecifier::DAY_OF_YEAR_PADDED:
	case StrTimeSpecifier::DAY_OF_YEAR_DECIMAL:
	case StrTimeSpecifier::WEEK_NUMBER_PADDED_MON_FIRST:
	case StrTimeSpecifier::WEEK_NUMBER_PADDED_SUN_FIRST:
	case StrTimeSpecifier::WEEKDAY_DECIMAL:
		return true;
	default:
		return false;
	}
}

char *StrfTimeFormat::WriteDateSpecifier(StrTimeSpecifier specifier, date_t date, char *target) {
	switch (specifier) {
	case StrTimeSpecifier::ABBREVIATED_WEEKDAY_NAME: {
		auto dow = Date::ExtractISODayOfTheWeek(date);
		target = WriteString(target, Date::DAY_NAMES_ABBREVIATED[dow % 7]);
		break;
	}
	case StrTimeSpecifier::FULL_WEEKDAY_NAME: {
		auto dow = Date::ExtractISODayOfTheWeek(date);
		target = WriteString(target, Date::DAY_NAMES[dow % 7]);
		break;
	}
	case StrTimeSpecifier::WEEKDAY_DECIMAL: {
		auto dow = Date::ExtractISODayOfTheWeek(date);
		*target = char('0' + uint8_t(dow % 7));
		target++;
		break;
	}
	case StrTimeSpecifier::DAY_OF_YEAR_PADDED: {
		int32_t doy = Date::ExtractDayOfTheYear(date);
		target = WritePadded3(target, doy);
		break;
	}
	case StrTimeSpecifier::WEEK_NUMBER_PADDED_MON_FIRST:
		target = WritePadded2(target, Date::ExtractWeekNumberRegular(date, true));
		break;
	case StrTimeSpecifier::WEEK_NUMBER_PADDED_SUN_FIRST:
		target = WritePadded2(target, Date::ExtractWeekNumberRegular(date, false));
		break;
	case StrTimeSpecifier::DAY_OF_YEAR_DECIMAL: {
		uint32_t doy = Date::ExtractDayOfTheYear(date);
		target += NumericHelper::UnsignedLength<uint32_t>(doy);
		NumericHelper::FormatUnsigned(doy, target);
		break;
	}
	default:
		throw InternalException("Unimplemented date specifier for strftime");
	}
	return target;
}

char *StrfTimeFormat::WriteStandardSpecifier(StrTimeSpecifier specifier, int32_t data[], const char *tz_name,
                                             size_t tz_len, char *target) {
	// data contains [0] year, [1] month, [2] day, [3] hour, [4] minute, [5] second, [6] msec, [7] utc
	switch (specifier) {
	case StrTimeSpecifier::DAY_OF_MONTH_PADDED:
		target = WritePadded2(target, data[2]);
		break;
	case StrTimeSpecifier::ABBREVIATED_MONTH_NAME: {
		auto &month_name = Date::MONTH_NAMES_ABBREVIATED[data[1] - 1];
		return WriteString(target, month_name);
	}
	case StrTimeSpecifier::FULL_MONTH_NAME: {
		auto &month_name = Date::MONTH_NAMES[data[1] - 1];
		return WriteString(target, month_name);
	}
	case StrTimeSpecifier::MONTH_DECIMAL_PADDED:
		target = WritePadded2(target, data[1]);
		break;
	case StrTimeSpecifier::YEAR_WITHOUT_CENTURY_PADDED:
		target = WritePadded2(target, AbsValue(data[0]) % 100);
		break;
	case StrTimeSpecifier::YEAR_DECIMAL:
		if (data[0] >= 0 && data[0] <= 9999) {
			target = WritePadded(target, data[0], 4);
		} else {
			int32_t year = data[0];
			if (data[0] < 0) {
				*target = '-';
				year = -year;
				target++;
			}
			auto len = NumericHelper::UnsignedLength<uint32_t>(year);
			NumericHelper::FormatUnsigned(year, target + len);
			target += len;
		}
		break;
	case StrTimeSpecifier::HOUR_24_PADDED: {
		target = WritePadded2(target, data[3]);
		break;
	}
	case StrTimeSpecifier::HOUR_12_PADDED: {
		int hour = data[3] % 12;
		if (hour == 0) {
			hour = 12;
		}
		target = WritePadded2(target, hour);
		break;
	}
	case StrTimeSpecifier::AM_PM:
		*target++ = data[3] >= 12 ? 'P' : 'A';
		*target++ = 'M';
		break;
	case StrTimeSpecifier::MINUTE_PADDED: {
		target = WritePadded2(target, data[4]);
		break;
	}
	case StrTimeSpecifier::SECOND_PADDED:
		target = WritePadded2(target, data[5]);
		break;
	case StrTimeSpecifier::MICROSECOND_PADDED:
		target = WritePadded(target, data[6], 6);
		break;
	case StrTimeSpecifier::MILLISECOND_PADDED:
		target = WritePadded3(target, data[6] / 1000);
		break;
	case StrTimeSpecifier::UTC_OFFSET: {
		*target++ = (data[7] < 0) ? '-' : '+';

		auto offset = abs(data[7]);
		auto offset_hours = offset / Interval::MINS_PER_HOUR;
		auto offset_minutes = offset % Interval::MINS_PER_HOUR;
		target = WritePadded2(target, offset_hours);
		if (offset_minutes) {
			*target++ = ':';
			target = WritePadded2(target, offset_minutes);
		}
		break;
	}
	case StrTimeSpecifier::TZ_NAME:
		if (tz_name) {
			memcpy(target, tz_name, tz_len);
			target += strlen(tz_name);
		}
		break;
	case StrTimeSpecifier::DAY_OF_MONTH: {
		target = Write2(target, data[2] % 100);
		break;
	}
	case StrTimeSpecifier::MONTH_DECIMAL: {
		target = Write2(target, data[1]);
		break;
	}
	case StrTimeSpecifier::YEAR_WITHOUT_CENTURY: {
		target = Write2(target, AbsValue(data[0]) % 100);
		break;
	}
	case StrTimeSpecifier::HOUR_24_DECIMAL: {
		target = Write2(target, data[3]);
		break;
	}
	case StrTimeSpecifier::HOUR_12_DECIMAL: {
		int hour = data[3] % 12;
		if (hour == 0) {
			hour = 12;
		}
		target = Write2(target, hour);
		break;
	}
	case StrTimeSpecifier::MINUTE_DECIMAL: {
		target = Write2(target, data[4]);
		break;
	}
	case StrTimeSpecifier::SECOND_DECIMAL: {
		target = Write2(target, data[5]);
		break;
	}
	default:
		throw InternalException("Unimplemented specifier for WriteStandardSpecifier in strftime");
	}
	return target;
}

void StrfTimeFormat::FormatString(date_t date, int32_t data[8], const char *tz_name, char *target) {
	D_ASSERT(specifiers.size() + 1 == literals.size());
	idx_t i;
	for (i = 0; i < specifiers.size(); i++) {
		// first copy the current literal
		memcpy(target, literals[i].c_str(), literals[i].size());
		target += literals[i].size();
		// now copy the specifier
		if (is_date_specifier[i]) {
			target = WriteDateSpecifier(specifiers[i], date, target);
		} else {
			auto tz_len = tz_name ? strlen(tz_name) : 0;
			target = WriteStandardSpecifier(specifiers[i], data, tz_name, tz_len, target);
		}
	}
	// copy the final literal into the target
	memcpy(target, literals[i].c_str(), literals[i].size());
}

void StrfTimeFormat::FormatString(date_t date, dtime_t time, char *target) {
	int32_t data[8]; // year, month, day, hour, min, sec, µs, offset
	Date::Convert(date, data[0], data[1], data[2]);
	Time::Convert(time, data[3], data[4], data[5], data[6]);
	data[7] = 0;

	FormatString(date, data, nullptr, target);
}

string StrfTimeFormat::Format(timestamp_t timestamp, const string &format_str) {
	StrfTimeFormat format;
	format.ParseFormatSpecifier(format_str, format);

	auto date = Timestamp::GetDate(timestamp);
	auto time = Timestamp::GetTime(timestamp);

	auto len = format.GetLength(date, time, 0, nullptr);
	auto result = make_unsafe_uniq_array<char>(len);
	format.FormatString(date, time, result.get());
	return string(result.get(), len);
}

string StrTimeFormat::ParseFormatSpecifier(const string &format_string, StrTimeFormat &format) {
	if (format_string.empty()) {
		return "Empty format string";
	}
	format.format_specifier = format_string;
	format.specifiers.clear();
	format.literals.clear();
	format.numeric_width.clear();
	format.constant_size = 0;
	idx_t pos = 0;
	string current_literal;
	for (idx_t i = 0; i < format_string.size(); i++) {
		if (format_string[i] == '%') {
			if (i + 1 == format_string.size()) {
				return "Trailing format character %";
			}
			if (i > pos) {
				// push the previous string to the current literal
				current_literal += format_string.substr(pos, i - pos);
			}
			char format_char = format_string[++i];
			if (format_char == '%') {
				// special case: %%
				// set the pos for the next literal and continue
				pos = i;
				continue;
			}
			StrTimeSpecifier specifier;
			if (format_char == '-' && i + 1 < format_string.size()) {
				format_char = format_string[++i];
				switch (format_char) {
				case 'd':
					specifier = StrTimeSpecifier::DAY_OF_MONTH;
					break;
				case 'm':
					specifier = StrTimeSpecifier::MONTH_DECIMAL;
					break;
				case 'y':
					specifier = StrTimeSpecifier::YEAR_WITHOUT_CENTURY;
					break;
				case 'H':
					specifier = StrTimeSpecifier::HOUR_24_DECIMAL;
					break;
				case 'I':
					specifier = StrTimeSpecifier::HOUR_12_DECIMAL;
					break;
				case 'M':
					specifier = StrTimeSpecifier::MINUTE_DECIMAL;
					break;
				case 'S':
					specifier = StrTimeSpecifier::SECOND_DECIMAL;
					break;
				case 'j':
					specifier = StrTimeSpecifier::DAY_OF_YEAR_DECIMAL;
					break;
				default:
					return "Unrecognized format for strftime/strptime: %-" + string(1, format_char);
				}
			} else {
				switch (format_char) {
				case 'a':
					specifier = StrTimeSpecifier::ABBREVIATED_WEEKDAY_NAME;
					break;
				case 'A':
					specifier = StrTimeSpecifier::FULL_WEEKDAY_NAME;
					break;
				case 'w':
					specifier = StrTimeSpecifier::WEEKDAY_DECIMAL;
					break;
				case 'd':
					specifier = StrTimeSpecifier::DAY_OF_MONTH_PADDED;
					break;
				case 'h':
				case 'b':
					specifier = StrTimeSpecifier::ABBREVIATED_MONTH_NAME;
					break;
				case 'B':
					specifier = StrTimeSpecifier::FULL_MONTH_NAME;
					break;
				case 'm':
					specifier = StrTimeSpecifier::MONTH_DECIMAL_PADDED;
					break;
				case 'y':
					specifier = StrTimeSpecifier::YEAR_WITHOUT_CENTURY_PADDED;
					break;
				case 'Y':
					specifier = StrTimeSpecifier::YEAR_DECIMAL;
					break;
				case 'H':
					specifier = StrTimeSpecifier::HOUR_24_PADDED;
					break;
				case 'I':
					specifier = StrTimeSpecifier::HOUR_12_PADDED;
					break;
				case 'p':
					specifier = StrTimeSpecifier::AM_PM;
					break;
				case 'M':
					specifier = StrTimeSpecifier::MINUTE_PADDED;
					break;
				case 'S':
					specifier = StrTimeSpecifier::SECOND_PADDED;
					break;
				case 'f':
					specifier = StrTimeSpecifier::MICROSECOND_PADDED;
					break;
				case 'g':
					specifier = StrTimeSpecifier::MILLISECOND_PADDED;
					break;
				case 'z':
					specifier = StrTimeSpecifier::UTC_OFFSET;
					break;
				case 'Z':
					specifier = StrTimeSpecifier::TZ_NAME;
					break;
				case 'j':
					specifier = StrTimeSpecifier::DAY_OF_YEAR_PADDED;
					break;
				case 'U':
					specifier = StrTimeSpecifier::WEEK_NUMBER_PADDED_SUN_FIRST;
					break;
				case 'W':
					specifier = StrTimeSpecifier::WEEK_NUMBER_PADDED_MON_FIRST;
					break;
				case 'c':
				case 'x':
				case 'X':
				case 'T': {
					string subformat;
					if (format_char == 'c') {
						// %c: Locale’s appropriate date and time representation.
						// we push the ISO timestamp representation here
						subformat = "%Y-%m-%d %H:%M:%S";
					} else if (format_char == 'x') {
						// %x - Locale’s appropriate date representation.
						// we push the ISO date format here
						subformat = "%Y-%m-%d";
					} else if (format_char == 'X' || format_char == 'T') {
						// %X - Locale’s appropriate time representation.
						// we push the ISO time format here
						subformat = "%H:%M:%S";
					}
					// parse the subformat in a separate format specifier
					StrfTimeFormat locale_format;
					string error = StrTimeFormat::ParseFormatSpecifier(subformat, locale_format);
					D_ASSERT(error.empty());
					// add the previous literal to the first literal of the subformat
					locale_format.literals[0] = std::move(current_literal) + locale_format.literals[0];
					current_literal = "";
					// now push the subformat into the current format specifier
					for (idx_t i = 0; i < locale_format.specifiers.size(); i++) {
						format.AddFormatSpecifier(std::move(locale_format.literals[i]), locale_format.specifiers[i]);
					}
					pos = i + 1;
					continue;
				}
				default:
					return "Unrecognized format for strftime/strptime: %" + string(1, format_char);
				}
			}
			format.AddFormatSpecifier(std::move(current_literal), specifier);
			current_literal = "";
			pos = i + 1;
		}
	}
	// add the final literal
	if (pos < format_string.size()) {
		current_literal += format_string.substr(pos, format_string.size() - pos);
	}
	format.AddLiteral(std::move(current_literal));
	return string();
}

void StrfTimeFormat::ConvertDateVector(Vector &input, Vector &result, idx_t count) {
	D_ASSERT(input.GetType().id() == LogicalTypeId::DATE);
	D_ASSERT(result.GetType().id() == LogicalTypeId::VARCHAR);
	UnaryExecutor::ExecuteWithNulls<date_t, string_t>(input, result, count,
	                                                  [&](date_t input, ValidityMask &mask, idx_t idx) {
		                                                  if (Date::IsFinite(input)) {
			                                                  dtime_t time(0);
			                                                  idx_t len = GetLength(input, time, 0, nullptr);
			                                                  string_t target = StringVector::EmptyString(result, len);
			                                                  FormatString(input, time, target.GetDataWriteable());
			                                                  target.Finalize();
			                                                  return target;
		                                                  } else {
			                                                  mask.SetInvalid(idx);
			                                                  return string_t();
		                                                  }
	                                                  });
}

void StrfTimeFormat::ConvertTimestampVector(Vector &input, Vector &result, idx_t count) {
	D_ASSERT(input.GetType().id() == LogicalTypeId::TIMESTAMP || input.GetType().id() == LogicalTypeId::TIMESTAMP_TZ);
	D_ASSERT(result.GetType().id() == LogicalTypeId::VARCHAR);
	UnaryExecutor::ExecuteWithNulls<timestamp_t, string_t>(
	    input, result, count, [&](timestamp_t input, ValidityMask &mask, idx_t idx) {
		    if (Timestamp::IsFinite(input)) {
			    date_t date;
			    dtime_t time;
			    Timestamp::Convert(input, date, time);
			    idx_t len = GetLength(date, time, 0, nullptr);
			    string_t target = StringVector::EmptyString(result, len);
			    FormatString(date, time, target.GetDataWriteable());
			    target.Finalize();
			    return target;
		    } else {
			    mask.SetInvalid(idx);
			    return string_t();
		    }
	    });
}

void StrpTimeFormat::AddFormatSpecifier(string preceding_literal, StrTimeSpecifier specifier) {
	numeric_width.push_back(NumericSpecifierWidth(specifier));
	StrTimeFormat::AddFormatSpecifier(std::move(preceding_literal), specifier);
}

int StrpTimeFormat::NumericSpecifierWidth(StrTimeSpecifier specifier) {
	switch (specifier) {
	case StrTimeSpecifier::WEEKDAY_DECIMAL:
		return 1;
	case StrTimeSpecifier::DAY_OF_MONTH_PADDED:
	case StrTimeSpecifier::DAY_OF_MONTH:
	case StrTimeSpecifier::MONTH_DECIMAL_PADDED:
	case StrTimeSpecifier::MONTH_DECIMAL:
	case StrTimeSpecifier::YEAR_WITHOUT_CENTURY_PADDED:
	case StrTimeSpecifier::YEAR_WITHOUT_CENTURY:
	case StrTimeSpecifier::HOUR_24_PADDED:
	case StrTimeSpecifier::HOUR_24_DECIMAL:
	case StrTimeSpecifier::HOUR_12_PADDED:
	case StrTimeSpecifier::HOUR_12_DECIMAL:
	case StrTimeSpecifier::MINUTE_PADDED:
	case StrTimeSpecifier::MINUTE_DECIMAL:
	case StrTimeSpecifier::SECOND_PADDED:
	case StrTimeSpecifier::SECOND_DECIMAL:
	case StrTimeSpecifier::WEEK_NUMBER_PADDED_SUN_FIRST:
	case StrTimeSpecifier::WEEK_NUMBER_PADDED_MON_FIRST:
		return 2;
	case StrTimeSpecifier::MILLISECOND_PADDED:
	case StrTimeSpecifier::DAY_OF_YEAR_PADDED:
	case StrTimeSpecifier::DAY_OF_YEAR_DECIMAL:
		return 3;
	case StrTimeSpecifier::YEAR_DECIMAL:
		return 4;
	case StrTimeSpecifier::MICROSECOND_PADDED:
		return 6;
	default:
		return -1;
	}
}

enum class TimeSpecifierAMOrPM : uint8_t { TIME_SPECIFIER_NONE = 0, TIME_SPECIFIER_AM = 1, TIME_SPECIFIER_PM = 2 };

int32_t StrpTimeFormat::TryParseCollection(const char *data, idx_t &pos, idx_t size, const string_t collection[],
                                           idx_t collection_count) {
	for (idx_t c = 0; c < collection_count; c++) {
		auto &entry = collection[c];
		auto entry_data = entry.GetData();
		auto entry_size = entry.GetSize();
		// check if this entry matches
		if (pos + entry_size > size) {
			// too big: can't match
			continue;
		}
		// compare the characters
		idx_t i;
		for (i = 0; i < entry_size; i++) {
			if (std::tolower(entry_data[i]) != std::tolower(data[pos + i])) {
				break;
			}
		}
		if (i == entry_size) {
			// full match
			pos += entry_size;
			return c;
		}
	}
	return -1;
}

//! Parses a timestamp using the given specifier
bool StrpTimeFormat::Parse(string_t str, ParseResult &result) {
	auto &result_data = result.data;
	auto &error_message = result.error_message;
	auto &error_position = result.error_position;

	// initialize the result
	result_data[0] = 1900;
	result_data[1] = 1;
	result_data[2] = 1;
	result_data[3] = 0;
	result_data[4] = 0;
	result_data[5] = 0;
	result_data[6] = 0;
	result_data[7] = 0;

	auto data = str.GetData();
	idx_t size = str.GetSize();
	// skip leading spaces
	while (StringUtil::CharacterIsSpace(*data)) {
		data++;
		size--;
	}
	idx_t pos = 0;
	TimeSpecifierAMOrPM ampm = TimeSpecifierAMOrPM::TIME_SPECIFIER_NONE;

	// Year offset state (Year+W/j)
	auto offset_specifier = StrTimeSpecifier::WEEKDAY_DECIMAL;
	uint64_t weekno = 0;
	uint64_t weekday = 0;
	uint64_t yearday = 0;

	for (idx_t i = 0;; i++) {
		D_ASSERT(i < literals.size());
		// first compare the literal
		const auto &literal = literals[i];
		for (size_t l = 0; l < literal.size();) {
			// Match runs of spaces to runs of spaces.
			if (StringUtil::CharacterIsSpace(literal[l])) {
				if (!StringUtil::CharacterIsSpace(data[pos])) {
					error_message = "Space does not match, expected " + literals[i];
					error_position = pos;
					return false;
				}
				for (++pos; pos < size && StringUtil::CharacterIsSpace(data[pos]); ++pos) {
					continue;
				}
				for (++l; l < literal.size() && StringUtil::CharacterIsSpace(literal[l]); ++l) {
					continue;
				}
				continue;
			}
			// literal does not match
			if (data[pos++] != literal[l++]) {
				error_message = "Literal does not match, expected " + literal;
				error_position = pos;
				return false;
			}
		}
		if (i == specifiers.size()) {
			break;
		}
		// now parse the specifier
		if (numeric_width[i] > 0) {
			// numeric specifier: parse a number
			uint64_t number = 0;
			size_t start_pos = pos;
			size_t end_pos = start_pos + numeric_width[i];
			while (pos < size && pos < end_pos && StringUtil::CharacterIsDigit(data[pos])) {
				number = number * 10 + data[pos] - '0';
				pos++;
			}
			if (pos == start_pos) {
				// expected a number here
				error_message = "Expected a number";
				error_position = start_pos;
				return false;
			}
			switch (specifiers[i]) {
			case StrTimeSpecifier::DAY_OF_MONTH_PADDED:
			case StrTimeSpecifier::DAY_OF_MONTH:
				if (number < 1 || number > 31) {
					error_message = "Day out of range, expected a value between 1 and 31";
					error_position = start_pos;
					return false;
				}
				// day of the month
				result_data[2] = number;
				offset_specifier = specifiers[i];
				break;
			case StrTimeSpecifier::MONTH_DECIMAL_PADDED:
			case StrTimeSpecifier::MONTH_DECIMAL:
				if (number < 1 || number > 12) {
					error_message = "Month out of range, expected a value between 1 and 12";
					error_position = start_pos;
					return false;
				}
				// month number
				result_data[1] = number;
				offset_specifier = specifiers[i];
				break;
			case StrTimeSpecifier::YEAR_WITHOUT_CENTURY_PADDED:
			case StrTimeSpecifier::YEAR_WITHOUT_CENTURY:
				// year without century..
				// Python uses 69 as a crossover point (i.e. >= 69 is 19.., < 69 is 20..)
				if (number >= 100) {
					// %y only supports numbers between [0..99]
					error_message = "Year without century out of range, expected a value between 0 and 99";
					error_position = start_pos;
					return false;
				}
				if (number >= 69) {
					result_data[0] = int32_t(1900 + number);
				} else {
					result_data[0] = int32_t(2000 + number);
				}
				break;
			case StrTimeSpecifier::YEAR_DECIMAL:
				// year as full number
				result_data[0] = number;
				break;
			case StrTimeSpecifier::HOUR_24_PADDED:
			case StrTimeSpecifier::HOUR_24_DECIMAL:
				if (number >= 24) {
					error_message = "Hour out of range, expected a value between 0 and 23";
					error_position = start_pos;
					return false;
				}
				// hour as full number
				result_data[3] = number;
				break;
			case StrTimeSpecifier::HOUR_12_PADDED:
			case StrTimeSpecifier::HOUR_12_DECIMAL:
				if (number < 1 || number > 12) {
					error_message = "Hour12 out of range, expected a value between 1 and 12";
					error_position = start_pos;
					return false;
				}
				// 12-hour number: start off by just storing the number
				result_data[3] = number;
				break;
			case StrTimeSpecifier::MINUTE_PADDED:
			case StrTimeSpecifier::MINUTE_DECIMAL:
				if (number >= 60) {
					error_message = "Minutes out of range, expected a value between 0 and 59";
					error_position = start_pos;
					return false;
				}
				// minutes
				result_data[4] = number;
				break;
			case StrTimeSpecifier::SECOND_PADDED:
			case StrTimeSpecifier::SECOND_DECIMAL:
				if (number >= 60) {
					error_message = "Seconds out of range, expected a value between 0 and 59";
					error_position = start_pos;
					return false;
				}
				// seconds
				result_data[5] = number;
				break;
			case StrTimeSpecifier::MICROSECOND_PADDED:
				D_ASSERT(number < 1000000ULL); // enforced by the length of the number
				// milliseconds
				result_data[6] = number;
				break;
			case StrTimeSpecifier::MILLISECOND_PADDED:
				D_ASSERT(number < 1000ULL); // enforced by the length of the number
				// milliseconds
				result_data[6] = number * 1000;
				break;
			case StrTimeSpecifier::WEEK_NUMBER_PADDED_SUN_FIRST:
			case StrTimeSpecifier::WEEK_NUMBER_PADDED_MON_FIRST:
				// m/d overrides WU/w but does not conflict
				switch (offset_specifier) {
				case StrTimeSpecifier::DAY_OF_MONTH_PADDED:
				case StrTimeSpecifier::DAY_OF_MONTH:
				case StrTimeSpecifier::MONTH_DECIMAL_PADDED:
				case StrTimeSpecifier::MONTH_DECIMAL:
					// Just validate, don't use
					break;
				case StrTimeSpecifier::WEEKDAY_DECIMAL:
					// First offset specifier
					offset_specifier = specifiers[i];
					break;
				default:
					error_message = "Multiple year offsets specified";
					error_position = start_pos;
					return false;
				}
				if (number > 53) {
					error_message = "Week out of range, expected a value between 0 and 53";
					error_position = start_pos;
					return false;
				}
				weekno = number;
				break;
			case StrTimeSpecifier::WEEKDAY_DECIMAL:
				if (number > 6) {
					error_message = "Weekday out of range, expected a value between 0 and 6";
					error_position = start_pos;
					return false;
				}
				weekday = number;
				break;
			case StrTimeSpecifier::DAY_OF_YEAR_PADDED:
			case StrTimeSpecifier::DAY_OF_YEAR_DECIMAL:
				// m/d overrides j but does not conflict
				switch (offset_specifier) {
				case StrTimeSpecifier::DAY_OF_MONTH_PADDED:
				case StrTimeSpecifier::DAY_OF_MONTH:
				case StrTimeSpecifier::MONTH_DECIMAL_PADDED:
				case StrTimeSpecifier::MONTH_DECIMAL:
					// Just validate, don't use
					break;
				case StrTimeSpecifier::WEEKDAY_DECIMAL:
					// First offset specifier
					offset_specifier = specifiers[i];
					break;
				default:
					error_message = "Multiple year offsets specified";
					error_position = start_pos;
					return false;
				}
				if (number < 1 || number > 366) {
					error_message = "Year day out of range, expected a value between 1 and 366";
					error_position = start_pos;
					return false;
				}
				yearday = number;
				break;
			default:
				throw NotImplementedException("Unsupported specifier for strptime");
			}
		} else {
			switch (specifiers[i]) {
			case StrTimeSpecifier::AM_PM: {
				// parse the next 2 characters
				if (pos + 2 > size) {
					// no characters left to parse
					error_message = "Expected AM/PM";
					error_position = pos;
					return false;
				}
				char pa_char = char(std::tolower(data[pos]));
				char m_char = char(std::tolower(data[pos + 1]));
				if (m_char != 'm') {
					error_message = "Expected AM/PM";
					error_position = pos;
					return false;
				}
				if (pa_char == 'p') {
					ampm = TimeSpecifierAMOrPM::TIME_SPECIFIER_PM;
				} else if (pa_char == 'a') {
					ampm = TimeSpecifierAMOrPM::TIME_SPECIFIER_AM;
				} else {
					error_message = "Expected AM/PM";
					error_position = pos;
					return false;
				}
				pos += 2;
				break;
			}
			// we parse weekday names, but we don't use them as information
			case StrTimeSpecifier::ABBREVIATED_WEEKDAY_NAME:
				if (TryParseCollection(data, pos, size, Date::DAY_NAMES_ABBREVIATED, 7) < 0) {
					error_message = "Expected an abbreviated day name (Mon, Tue, Wed, Thu, Fri, Sat, Sun)";
					error_position = pos;
					return false;
				}
				break;
			case StrTimeSpecifier::FULL_WEEKDAY_NAME:
				if (TryParseCollection(data, pos, size, Date::DAY_NAMES, 7) < 0) {
					error_message = "Expected a full day name (Monday, Tuesday, etc...)";
					error_position = pos;
					return false;
				}
				break;
			case StrTimeSpecifier::ABBREVIATED_MONTH_NAME: {
				int32_t month = TryParseCollection(data, pos, size, Date::MONTH_NAMES_ABBREVIATED, 12);
				if (month < 0) {
					error_message = "Expected an abbreviated month name (Jan, Feb, Mar, etc..)";
					error_position = pos;
					return false;
				}
				result_data[1] = month + 1;
				break;
			}
			case StrTimeSpecifier::FULL_MONTH_NAME: {
				int32_t month = TryParseCollection(data, pos, size, Date::MONTH_NAMES, 12);
				if (month < 0) {
					error_message = "Expected a full month name (January, February, etc...)";
					error_position = pos;
					return false;
				}
				result_data[1] = month + 1;
				break;
			}
			case StrTimeSpecifier::UTC_OFFSET: {
				int hour_offset, minute_offset;
				if (!Timestamp::TryParseUTCOffset(data, pos, size, hour_offset, minute_offset)) {
					error_message = "Expected +HH[MM] or -HH[MM]";
					error_position = pos;
					return false;
				}
				result_data[7] = hour_offset * Interval::MINS_PER_HOUR + minute_offset;
				break;
			}
			case StrTimeSpecifier::TZ_NAME: {
				// skip leading spaces
				while (pos < size && StringUtil::CharacterIsSpace(data[pos])) {
					pos++;
				}
				const auto tz_begin = data + pos;
				// stop when we encounter a non-tz character
				while (pos < size && Timestamp::CharacterIsTimeZone(data[pos])) {
					pos++;
				}
				const auto tz_end = data + pos;
				// Can't fully validate without a list - caller's responsibility.
				// But tz must not be empty.
				if (tz_end == tz_begin) {
					error_message = "Empty Time Zone name";
					error_position = tz_begin - data;
					return false;
				}
				result.tz.assign(tz_begin, tz_end);
				break;
			}
			default:
				throw NotImplementedException("Unsupported specifier for strptime");
			}
		}
	}
	// skip trailing spaces
	while (pos < size && StringUtil::CharacterIsSpace(data[pos])) {
		pos++;
	}
	if (pos != size) {
		error_message = "Full specifier did not match: trailing characters";
		error_position = pos;
		return false;
	}
	if (ampm != TimeSpecifierAMOrPM::TIME_SPECIFIER_NONE) {
		if (result_data[3] > 12) {
			error_message =
			    "Invalid hour: " + to_string(result_data[3]) + " AM/PM, expected an hour within the range [0..12]";
			return false;
		}
		// adjust the hours based on the AM or PM specifier
		if (ampm == TimeSpecifierAMOrPM::TIME_SPECIFIER_AM) {
			// AM: 12AM=0, 1AM=1, 2AM=2, ..., 11AM=11
			if (result_data[3] == 12) {
				result_data[3] = 0;
			}
		} else {
			// PM: 12PM=12, 1PM=13, 2PM=14, ..., 11PM=23
			if (result_data[3] != 12) {
				result_data[3] += 12;
			}
		}
	}
	switch (offset_specifier) {
	case StrTimeSpecifier::WEEK_NUMBER_PADDED_SUN_FIRST:
	case StrTimeSpecifier::WEEK_NUMBER_PADDED_MON_FIRST: {
		// Adjust weekday to be 0-based for the week type
		weekday = (weekday + 7 - int(offset_specifier == StrTimeSpecifier::WEEK_NUMBER_PADDED_MON_FIRST)) % 7;
		// Get the start of week 1, move back 7 days and then weekno * 7 + weekday gives the date
		const auto jan1 = Date::FromDate(result_data[0], 1, 1);
		auto yeardate = Date::GetMondayOfCurrentWeek(jan1);
		yeardate -= int(offset_specifier == StrTimeSpecifier::WEEK_NUMBER_PADDED_SUN_FIRST);
		// Is there a week 0?
		yeardate -= 7 * int(yeardate >= jan1);
		yeardate += weekno * 7 + weekday;
		Date::Convert(yeardate, result_data[0], result_data[1], result_data[2]);
		break;
	}
	case StrTimeSpecifier::DAY_OF_YEAR_PADDED:
	case StrTimeSpecifier::DAY_OF_YEAR_DECIMAL: {
		auto yeardate = Date::FromDate(result_data[0], 1, 1);
		yeardate += yearday - 1;
		Date::Convert(yeardate, result_data[0], result_data[1], result_data[2]);
		break;
	}
	case StrTimeSpecifier::DAY_OF_MONTH_PADDED:
	case StrTimeSpecifier::DAY_OF_MONTH:
	case StrTimeSpecifier::MONTH_DECIMAL_PADDED:
	case StrTimeSpecifier::MONTH_DECIMAL:
		// m/d overrides UWw/j
		break;
	default:
		D_ASSERT(offset_specifier == StrTimeSpecifier::WEEKDAY_DECIMAL);
		break;
	}

	return true;
}

StrpTimeFormat::ParseResult StrpTimeFormat::Parse(const string &format_string, const string &text) {
	StrpTimeFormat format;
	format.format_specifier = format_string;
	string error = StrTimeFormat::ParseFormatSpecifier(format_string, format);
	if (!error.empty()) {
		throw InvalidInputException("Failed to parse format specifier %s: %s", format_string, error);
	}
	StrpTimeFormat::ParseResult result;
	if (!format.Parse(text, result)) {
		throw InvalidInputException("Failed to parse string \"%s\" with format specifier \"%s\"", text, format_string);
	}
	return result;
}

string StrpTimeFormat::FormatStrpTimeError(const string &input, idx_t position) {
	if (position == DConstants::INVALID_INDEX) {
		return string();
	}
	return input + "\n" + string(position, ' ') + "^";
}

date_t StrpTimeFormat::ParseResult::ToDate() {
	return Date::FromDate(data[0], data[1], data[2]);
}

bool StrpTimeFormat::ParseResult::TryToDate(date_t &result) {
	return Date::TryFromDate(data[0], data[1], data[2], result);
}

timestamp_t StrpTimeFormat::ParseResult::ToTimestamp() {
	date_t date = Date::FromDate(data[0], data[1], data[2]);
	const auto hour_offset = data[7] / Interval::MINS_PER_HOUR;
	const auto mins_offset = data[7] % Interval::MINS_PER_HOUR;
	dtime_t time = Time::FromTime(data[3] - hour_offset, data[4] - mins_offset, data[5], data[6]);
	return Timestamp::FromDatetime(date, time);
}

bool StrpTimeFormat::ParseResult::TryToTimestamp(timestamp_t &result) {
	date_t date;
	if (!TryToDate(date)) {
		return false;
	}
	const auto hour_offset = data[7] / Interval::MINS_PER_HOUR;
	const auto mins_offset = data[7] % Interval::MINS_PER_HOUR;
	dtime_t time = Time::FromTime(data[3] - hour_offset, data[4] - mins_offset, data[5], data[6]);
	return Timestamp::TryFromDatetime(date, time, result);
}

string StrpTimeFormat::ParseResult::FormatError(string_t input, const string &format_specifier) {
	return StringUtil::Format("Could not parse string \"%s\" according to format specifier \"%s\"\n%s\nError: %s",
	                          input.GetString(), format_specifier,
	                          FormatStrpTimeError(input.GetString(), error_position), error_message);
}

bool StrpTimeFormat::TryParseDate(string_t input, date_t &result, string &error_message) {
	ParseResult parse_result;
	if (!Parse(input, parse_result)) {
		error_message = parse_result.FormatError(input, format_specifier);
		return false;
	}
	return parse_result.TryToDate(result);
}

bool StrpTimeFormat::TryParseTimestamp(string_t input, timestamp_t &result, string &error_message) {
	ParseResult parse_result;
	if (!Parse(input, parse_result)) {
		error_message = parse_result.FormatError(input, format_specifier);
		return false;
	}
	return parse_result.TryToTimestamp(result);
}

date_t StrpTimeFormat::ParseDate(string_t input) {
	ParseResult result;
	if (!Parse(input, result)) {
		throw InvalidInputException(result.FormatError(input, format_specifier));
	}
	return result.ToDate();
}

timestamp_t StrpTimeFormat::ParseTimestamp(string_t input) {
	ParseResult result;
	if (!Parse(input, result)) {
		throw InvalidInputException(result.FormatError(input, format_specifier));
	}
	return result.ToTimestamp();
}

} // namespace duckdb









#include <string.h>

namespace duckdb {

uint8_t UpperFun::ascii_to_upper_map[] = {
    0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,
    22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,
    44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,
    66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,
    88,  89,  90,  91,  92,  93,  94,  95,  96,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,
    78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  123, 124, 125, 126, 127, 128, 129, 130, 131,
    132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153,
    154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
    176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197,
    198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219,
    220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241,
    242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254};
uint8_t LowerFun::ascii_to_lower_map[] = {
    0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,
    22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,
    44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  97,
    98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
    120, 121, 122, 91,  92,  93,  94,  95,  96,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
    110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131,
    132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153,
    154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
    176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197,
    198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219,
    220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241,
    242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254};

template <bool IS_UPPER>
static string_t ASCIICaseConvert(Vector &result, const char *input_data, idx_t input_length) {
	idx_t output_length = input_length;
	auto result_str = StringVector::EmptyString(result, output_length);
	auto result_data = result_str.GetDataWriteable();
	for (idx_t i = 0; i < input_length; i++) {
		result_data[i] = IS_UPPER ? UpperFun::ascii_to_upper_map[uint8_t(input_data[i])]
		                          : LowerFun::ascii_to_lower_map[uint8_t(input_data[i])];
	}
	result_str.Finalize();
	return result_str;
}

template <bool IS_UPPER>
static idx_t GetResultLength(const char *input_data, idx_t input_length) {
	idx_t output_length = 0;
	for (idx_t i = 0; i < input_length;) {
		if (input_data[i] & 0x80) {
			// unicode
			int sz = 0;
			int codepoint = utf8proc_codepoint(input_data + i, sz);
			int converted_codepoint = IS_UPPER ? utf8proc_toupper(codepoint) : utf8proc_tolower(codepoint);
			int new_sz = utf8proc_codepoint_length(converted_codepoint);
			D_ASSERT(new_sz >= 0);
			output_length += new_sz;
			i += sz;
		} else {
			// ascii
			output_length++;
			i++;
		}
	}
	return output_length;
}

template <bool IS_UPPER>
static void CaseConvert(const char *input_data, idx_t input_length, char *result_data) {
	for (idx_t i = 0; i < input_length;) {
		if (input_data[i] & 0x80) {
			// non-ascii character
			int sz = 0, new_sz = 0;
			int codepoint = utf8proc_codepoint(input_data + i, sz);
			int converted_codepoint = IS_UPPER ? utf8proc_toupper(codepoint) : utf8proc_tolower(codepoint);
			auto success = utf8proc_codepoint_to_utf8(converted_codepoint, new_sz, result_data);
			D_ASSERT(success);
			(void)success;
			result_data += new_sz;
			i += sz;
		} else {
			// ascii
			*result_data = IS_UPPER ? UpperFun::ascii_to_upper_map[uint8_t(input_data[i])]
			                        : LowerFun::ascii_to_lower_map[uint8_t(input_data[i])];
			result_data++;
			i++;
		}
	}
}

idx_t LowerFun::LowerLength(const char *input_data, idx_t input_length) {
	return GetResultLength<false>(input_data, input_length);
}

void LowerFun::LowerCase(const char *input_data, idx_t input_length, char *result_data) {
	CaseConvert<false>(input_data, input_length, result_data);
}

template <bool IS_UPPER>
static string_t UnicodeCaseConvert(Vector &result, const char *input_data, idx_t input_length) {
	// first figure out the output length
	idx_t output_length = GetResultLength<IS_UPPER>(input_data, input_length);
	auto result_str = StringVector::EmptyString(result, output_length);
	auto result_data = result_str.GetDataWriteable();

	CaseConvert<IS_UPPER>(input_data, input_length, result_data);
	result_str.Finalize();
	return result_str;
}

template <bool IS_UPPER>
struct CaseConvertOperator {
	template <class INPUT_TYPE, class RESULT_TYPE>
	static RESULT_TYPE Operation(INPUT_TYPE input, Vector &result) {
		auto input_data = input.GetData();
		auto input_length = input.GetSize();
		return UnicodeCaseConvert<IS_UPPER>(result, input_data, input_length);
	}
};

template <bool IS_UPPER>
static void CaseConvertFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	UnaryExecutor::ExecuteString<string_t, string_t, CaseConvertOperator<IS_UPPER>>(args.data[0], result, args.size());
}

template <bool IS_UPPER>
struct CaseConvertOperatorASCII {
	template <class INPUT_TYPE, class RESULT_TYPE>
	static RESULT_TYPE Operation(INPUT_TYPE input, Vector &result) {
		auto input_data = input.GetData();
		auto input_length = input.GetSize();
		return ASCIICaseConvert<IS_UPPER>(result, input_data, input_length);
	}
};

template <bool IS_UPPER>
static void CaseConvertFunctionASCII(DataChunk &args, ExpressionState &state, Vector &result) {
	UnaryExecutor::ExecuteString<string_t, string_t, CaseConvertOperatorASCII<IS_UPPER>>(args.data[0], result,
	                                                                                     args.size());
}

template <bool IS_UPPER>
static unique_ptr<BaseStatistics> CaseConvertPropagateStats(ClientContext &context, FunctionStatisticsInput &input) {
	auto &child_stats = input.child_stats;
	auto &expr = input.expr;
	D_ASSERT(child_stats.size() == 1);
	// can only propagate stats if the children have stats
	if (!StringStats::CanContainUnicode(child_stats[0])) {
		expr.function.function = CaseConvertFunctionASCII<IS_UPPER>;
	}
	return nullptr;
}

ScalarFunction LowerFun::GetFunction() {
	return ScalarFunction("lower", {LogicalType::VARCHAR}, LogicalType::VARCHAR, CaseConvertFunction<false>, nullptr,
	                      nullptr, CaseConvertPropagateStats<false>);
}

void LowerFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction({"lower", "lcase"}, LowerFun::GetFunction());
}

void UpperFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction({"upper", "ucase"},
	                ScalarFunction({LogicalType::VARCHAR}, LogicalType::VARCHAR, CaseConvertFunction<true>, nullptr,
	                               nullptr, CaseConvertPropagateStats<true>));
}

} // namespace duckdb







#include <string.h>

namespace duckdb {

static void ConcatFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	// iterate over the vectors to count how large the final string will be
	idx_t constant_lengths = 0;
	vector<idx_t> result_lengths(args.size(), 0);
	for (idx_t col_idx = 0; col_idx < args.ColumnCount(); col_idx++) {
		auto &input = args.data[col_idx];
		D_ASSERT(input.GetType().id() == LogicalTypeId::VARCHAR);
		if (input.GetVectorType() == VectorType::CONSTANT_VECTOR) {
			if (ConstantVector::IsNull(input)) {
				// constant null, skip
				continue;
			}
			auto input_data = ConstantVector::GetData<string_t>(input);
			constant_lengths += input_data->GetSize();
		} else {
			// non-constant vector: set the result type to a flat vector
			result.SetVectorType(VectorType::FLAT_VECTOR);
			// now get the lengths of each of the input elements
			UnifiedVectorFormat vdata;
			input.ToUnifiedFormat(args.size(), vdata);

			auto input_data = (string_t *)vdata.data;
			// now add the length of each vector to the result length
			for (idx_t i = 0; i < args.size(); i++) {
				auto idx = vdata.sel->get_index(i);
				if (!vdata.validity.RowIsValid(idx)) {
					continue;
				}
				result_lengths[i] += input_data[idx].GetSize();
			}
		}
	}

	// first we allocate the empty strings for each of the values
	auto result_data = FlatVector::GetData<string_t>(result);
	for (idx_t i = 0; i < args.size(); i++) {
		// allocate an empty string of the required size
		idx_t str_length = constant_lengths + result_lengths[i];
		result_data[i] = StringVector::EmptyString(result, str_length);
		// we reuse the result_lengths vector to store the currently appended size
		result_lengths[i] = 0;
	}

	// now that the empty space for the strings has been allocated, perform the concatenation
	for (idx_t col_idx = 0; col_idx < args.ColumnCount(); col_idx++) {
		auto &input = args.data[col_idx];

		// loop over the vector and concat to all results
		if (input.GetVectorType() == VectorType::CONSTANT_VECTOR) {
			// constant vector
			if (ConstantVector::IsNull(input)) {
				// constant null, skip
				continue;
			}
			// append the constant vector to each of the strings
			auto input_data = ConstantVector::GetData<string_t>(input);
			auto input_ptr = input_data->GetData();
			auto input_len = input_data->GetSize();
			for (idx_t i = 0; i < args.size(); i++) {
				memcpy(result_data[i].GetDataWriteable() + result_lengths[i], input_ptr, input_len);
				result_lengths[i] += input_len;
			}
		} else {
			// standard vector
			UnifiedVectorFormat idata;
			input.ToUnifiedFormat(args.size(), idata);

			auto input_data = (string_t *)idata.data;
			for (idx_t i = 0; i < args.size(); i++) {
				auto idx = idata.sel->get_index(i);
				if (!idata.validity.RowIsValid(idx)) {
					continue;
				}
				auto input_ptr = input_data[idx].GetData();
				auto input_len = input_data[idx].GetSize();
				memcpy(result_data[i].GetDataWriteable() + result_lengths[i], input_ptr, input_len);
				result_lengths[i] += input_len;
			}
		}
	}
	for (idx_t i = 0; i < args.size(); i++) {
		result_data[i].Finalize();
	}
}

static void ConcatOperator(DataChunk &args, ExpressionState &state, Vector &result) {
	BinaryExecutor::Execute<string_t, string_t, string_t>(
	    args.data[0], args.data[1], result, args.size(), [&](string_t a, string_t b) {
		    auto a_data = a.GetData();
		    auto b_data = b.GetData();
		    auto a_length = a.GetSize();
		    auto b_length = b.GetSize();

		    auto target_length = a_length + b_length;
		    auto target = StringVector::EmptyString(result, target_length);
		    auto target_data = target.GetDataWriteable();

		    memcpy(target_data, a_data, a_length);
		    memcpy(target_data + a_length, b_data, b_length);
		    target.Finalize();
		    return target;
	    });
}

static void TemplatedConcatWS(DataChunk &args, string_t *sep_data, const SelectionVector &sep_sel,
                              const SelectionVector &rsel, idx_t count, Vector &result) {
	vector<idx_t> result_lengths(args.size(), 0);
	vector<bool> has_results(args.size(), false);
	auto orrified_data = make_unsafe_uniq_array<UnifiedVectorFormat>(args.ColumnCount() - 1);
	for (idx_t col_idx = 1; col_idx < args.ColumnCount(); col_idx++) {
		args.data[col_idx].ToUnifiedFormat(args.size(), orrified_data[col_idx - 1]);
	}

	// first figure out the lengths
	for (idx_t col_idx = 1; col_idx < args.ColumnCount(); col_idx++) {
		auto &idata = orrified_data[col_idx - 1];

		auto input_data = (string_t *)idata.data;
		for (idx_t i = 0; i < count; i++) {
			auto ridx = rsel.get_index(i);
			auto sep_idx = sep_sel.get_index(ridx);
			auto idx = idata.sel->get_index(ridx);
			if (!idata.validity.RowIsValid(idx)) {
				continue;
			}
			if (has_results[ridx]) {
				result_lengths[ridx] += sep_data[sep_idx].GetSize();
			}
			result_lengths[ridx] += input_data[idx].GetSize();
			has_results[ridx] = true;
		}
	}

	// first we allocate the empty strings for each of the values
	auto result_data = FlatVector::GetData<string_t>(result);
	for (idx_t i = 0; i < count; i++) {
		auto ridx = rsel.get_index(i);
		// allocate an empty string of the required size
		result_data[ridx] = StringVector::EmptyString(result, result_lengths[ridx]);
		// we reuse the result_lengths vector to store the currently appended size
		result_lengths[ridx] = 0;
		has_results[ridx] = false;
	}

	// now that the empty space for the strings has been allocated, perform the concatenation
	for (idx_t col_idx = 1; col_idx < args.ColumnCount(); col_idx++) {
		auto &idata = orrified_data[col_idx - 1];
		auto input_data = (string_t *)idata.data;
		for (idx_t i = 0; i < count; i++) {
			auto ridx = rsel.get_index(i);
			auto sep_idx = sep_sel.get_index(ridx);
			auto idx = idata.sel->get_index(ridx);
			if (!idata.validity.RowIsValid(idx)) {
				continue;
			}
			if (has_results[ridx]) {
				auto sep_size = sep_data[sep_idx].GetSize();
				auto sep_ptr = sep_data[sep_idx].GetData();
				memcpy(result_data[ridx].GetDataWriteable() + result_lengths[ridx], sep_ptr, sep_size);
				result_lengths[ridx] += sep_size;
			}
			auto input_ptr = input_data[idx].GetData();
			auto input_len = input_data[idx].GetSize();
			memcpy(result_data[ridx].GetDataWriteable() + result_lengths[ridx], input_ptr, input_len);
			result_lengths[ridx] += input_len;
			has_results[ridx] = true;
		}
	}
	for (idx_t i = 0; i < count; i++) {
		auto ridx = rsel.get_index(i);
		result_data[ridx].Finalize();
	}
}

static void ConcatWSFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &separator = args.data[0];
	UnifiedVectorFormat vdata;
	separator.ToUnifiedFormat(args.size(), vdata);

	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	for (idx_t col_idx = 0; col_idx < args.ColumnCount(); col_idx++) {
		if (args.data[col_idx].GetVectorType() != VectorType::CONSTANT_VECTOR) {
			result.SetVectorType(VectorType::FLAT_VECTOR);
			break;
		}
	}
	switch (separator.GetVectorType()) {
	case VectorType::CONSTANT_VECTOR: {
		if (ConstantVector::IsNull(separator)) {
			// constant NULL as separator: return constant NULL vector
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
			ConstantVector::SetNull(result, true);
			return;
		}
		// no null values
		auto sel = FlatVector::IncrementalSelectionVector();
		TemplatedConcatWS(args, (string_t *)vdata.data, *vdata.sel, *sel, args.size(), result);
		return;
	}
	default: {
		// default case: loop over nullmask and create a non-null selection vector
		idx_t not_null_count = 0;
		SelectionVector not_null_vector(STANDARD_VECTOR_SIZE);
		auto &result_mask = FlatVector::Validity(result);
		for (idx_t i = 0; i < args.size(); i++) {
			if (!vdata.validity.RowIsValid(vdata.sel->get_index(i))) {
				result_mask.SetInvalid(i);
			} else {
				not_null_vector.set_index(not_null_count++, i);
			}
		}
		TemplatedConcatWS(args, (string_t *)vdata.data, *vdata.sel, not_null_vector, not_null_count, result);
		return;
	}
	}
}

void ConcatFun::RegisterFunction(BuiltinFunctions &set) {
	// the concat operator and concat function have different behavior regarding NULLs
	// this is strange but seems consistent with postgresql and mysql
	// (sqlite does not support the concat function, only the concat operator)

	// the concat operator behaves as one would expect: any NULL value present results in a NULL
	// i.e. NULL || 'hello' = NULL
	// the concat function, however, treats NULL values as an empty string
	// i.e. concat(NULL, 'hello') = 'hello'
	// concat_ws functions similarly to the concat function, except the result is NULL if the separator is NULL
	// if the separator is not NULL, however, NULL values are counted as empty string
	// there is one separate rule: there are no separators added between NULL values
	// so the NULL value and empty string are different!
	// e.g.:
	// concat_ws(',', NULL, NULL) = ""
	// concat_ws(',', '', '') = ","
	ScalarFunction concat = ScalarFunction("concat", {LogicalType::VARCHAR}, LogicalType::VARCHAR, ConcatFunction);
	concat.varargs = LogicalType::VARCHAR;
	concat.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	set.AddFunction(concat);

	ScalarFunctionSet concat_op("||");
	concat_op.AddFunction(
	    ScalarFunction({LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::VARCHAR, ConcatOperator));
	concat_op.AddFunction(ScalarFunction({LogicalType::BLOB, LogicalType::BLOB}, LogicalType::BLOB, ConcatOperator));
	concat_op.AddFunction(ListConcatFun::GetFunction());
	for (auto &fun : concat_op.functions) {
		fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	}
	set.AddFunction(concat_op);

	ScalarFunction concat_ws = ScalarFunction("concat_ws", {LogicalType::VARCHAR, LogicalType::VARCHAR},
	                                          LogicalType::VARCHAR, ConcatWSFunction);
	concat_ws.varargs = LogicalType::VARCHAR;
	concat_ws.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	set.AddFunction(concat_ws);
}

} // namespace duckdb






namespace duckdb {

template <class UNSIGNED, int NEEDLE_SIZE>
static idx_t ContainsUnaligned(const unsigned char *haystack, idx_t haystack_size, const unsigned char *needle,
                               idx_t base_offset) {
	if (NEEDLE_SIZE > haystack_size) {
		// needle is bigger than haystack: haystack cannot contain needle
		return DConstants::INVALID_INDEX;
	}
	// contains for a small unaligned needle (3/5/6/7 bytes)
	// we perform unsigned integer comparisons to check for equality of the entire needle in a single comparison
	// this implementation is inspired by the memmem implementation of freebsd

	// first we set up the needle and the first NEEDLE_SIZE characters of the haystack as UNSIGNED integers
	UNSIGNED needle_entry = 0;
	UNSIGNED haystack_entry = 0;
	const UNSIGNED start = (sizeof(UNSIGNED) * 8) - 8;
	const UNSIGNED shift = (sizeof(UNSIGNED) - NEEDLE_SIZE) * 8;
	for (int i = 0; i < NEEDLE_SIZE; i++) {
		needle_entry |= UNSIGNED(needle[i]) << UNSIGNED(start - i * 8);
		haystack_entry |= UNSIGNED(haystack[i]) << UNSIGNED(start - i * 8);
	}
	// now we perform the actual search
	for (idx_t offset = NEEDLE_SIZE; offset < haystack_size; offset++) {
		// for this position we first compare the haystack with the needle
		if (haystack_entry == needle_entry) {
			return base_offset + offset - NEEDLE_SIZE;
		}
		// now we adjust the haystack entry by
		// (1) removing the left-most character (shift by 8)
		// (2) adding the next character (bitwise or, with potential shift)
		// this shift is only necessary if the needle size is not aligned with the unsigned integer size
		// (e.g. needle size 3, unsigned integer size 4, we need to shift by 1)
		haystack_entry = (haystack_entry << 8) | ((UNSIGNED(haystack[offset])) << shift);
	}
	if (haystack_entry == needle_entry) {
		return base_offset + haystack_size - NEEDLE_SIZE;
	}
	return DConstants::INVALID_INDEX;
}

template <class UNSIGNED>
static idx_t ContainsAligned(const unsigned char *haystack, idx_t haystack_size, const unsigned char *needle,
                             idx_t base_offset) {
	if (sizeof(UNSIGNED) > haystack_size) {
		// needle is bigger than haystack: haystack cannot contain needle
		return DConstants::INVALID_INDEX;
	}
	// contains for a small needle aligned with unsigned integer (2/4/8)
	// similar to ContainsUnaligned, but simpler because we only need to do a reinterpret cast
	auto needle_entry = Load<UNSIGNED>(needle);
	for (idx_t offset = 0; offset <= haystack_size - sizeof(UNSIGNED); offset++) {
		// for this position we first compare the haystack with the needle
		auto haystack_entry = Load<UNSIGNED>(haystack + offset);
		if (needle_entry == haystack_entry) {
			return base_offset + offset;
		}
	}
	return DConstants::INVALID_INDEX;
}

idx_t ContainsGeneric(const unsigned char *haystack, idx_t haystack_size, const unsigned char *needle,
                      idx_t needle_size, idx_t base_offset) {
	if (needle_size > haystack_size) {
		// needle is bigger than haystack: haystack cannot contain needle
		return DConstants::INVALID_INDEX;
	}
	// this implementation is inspired by Raphael Javaux's faststrstr (https://github.com/RaphaelJ/fast_strstr)
	// generic contains; note that we can't use strstr because we don't have null-terminated strings anymore
	// we keep track of a shifting window sum of all characters with window size equal to needle_size
	// this shifting sum is used to avoid calling into memcmp;
	// we only need to call into memcmp when the window sum is equal to the needle sum
	// when that happens, the characters are potentially the same and we call into memcmp to check if they are
	uint32_t sums_diff = 0;
	for (idx_t i = 0; i < needle_size; i++) {
		sums_diff += haystack[i];
		sums_diff -= needle[i];
	}
	idx_t offset = 0;
	while (true) {
		if (sums_diff == 0 && haystack[offset] == needle[0]) {
			if (memcmp(haystack + offset, needle, needle_size) == 0) {
				return base_offset + offset;
			}
		}
		if (offset >= haystack_size - needle_size) {
			return DConstants::INVALID_INDEX;
		}
		sums_diff -= haystack[offset];
		sums_diff += haystack[offset + needle_size];
		offset++;
	}
}

idx_t ContainsFun::Find(const unsigned char *haystack, idx_t haystack_size, const unsigned char *needle,
                        idx_t needle_size) {
	D_ASSERT(needle_size > 0);
	// start off by performing a memchr to find the first character of the
	auto location = memchr(haystack, needle[0], haystack_size);
	if (location == nullptr) {
		return DConstants::INVALID_INDEX;
	}
	idx_t base_offset = (const unsigned char *)location - haystack;
	haystack_size -= base_offset;
	haystack = (const unsigned char *)location;
	// switch algorithm depending on needle size
	switch (needle_size) {
	case 1:
		return base_offset;
	case 2:
		return ContainsAligned<uint16_t>(haystack, haystack_size, needle, base_offset);
	case 3:
		return ContainsUnaligned<uint32_t, 3>(haystack, haystack_size, needle, base_offset);
	case 4:
		return ContainsAligned<uint32_t>(haystack, haystack_size, needle, base_offset);
	case 5:
		return ContainsUnaligned<uint64_t, 5>(haystack, haystack_size, needle, base_offset);
	case 6:
		return ContainsUnaligned<uint64_t, 6>(haystack, haystack_size, needle, base_offset);
	case 7:
		return ContainsUnaligned<uint64_t, 7>(haystack, haystack_size, needle, base_offset);
	case 8:
		return ContainsAligned<uint64_t>(haystack, haystack_size, needle, base_offset);
	default:
		return ContainsGeneric(haystack, haystack_size, needle, needle_size, base_offset);
	}
}

idx_t ContainsFun::Find(const string_t &haystack_s, const string_t &needle_s) {
	auto haystack = (const unsigned char *)haystack_s.GetData();
	auto haystack_size = haystack_s.GetSize();
	auto needle = (const unsigned char *)needle_s.GetData();
	auto needle_size = needle_s.GetSize();
	if (needle_size == 0) {
		// empty needle: always true
		return 0;
	}
	return ContainsFun::Find(haystack, haystack_size, needle, needle_size);
}

struct ContainsOperator {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA left, TB right) {
		return ContainsFun::Find(left, right) != DConstants::INVALID_INDEX;
	}
};

ScalarFunction ContainsFun::GetFunction() {
	return ScalarFunction("contains",                                   // name of the function
	                      {LogicalType::VARCHAR, LogicalType::VARCHAR}, // argument list
	                      LogicalType::BOOLEAN,                         // return type
	                      ScalarFunction::BinaryFunction<string_t, string_t, bool, ContainsOperator>);
}

void ContainsFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(GetFunction());
}

} // namespace duckdb










namespace duckdb {

// length returns the number of unicode codepoints
struct StringLengthOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		return LengthFun::Length<TA, TR>(input);
	}
};

struct GraphemeCountOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		return LengthFun::GraphemeCount<TA, TR>(input);
	}
};

struct ArrayLengthOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		return input.length;
	}
};

struct ArrayLengthBinaryOperator {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA input, TB dimension) {
		if (dimension != 1) {
			throw NotImplementedException("array_length for dimensions other than 1 not implemented");
		}
		return input.length;
	}
};

// strlen returns the size in bytes
struct StrLenOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		return input.GetSize();
	}
};

struct OctetLenOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		return Bit::OctetLength(input);
	}
};

// bitlen returns the size in bits
struct BitLenOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		return 8 * input.GetSize();
	}
};

// bitstringlen returns the amount of bits in a bitstring
struct BitStringLenOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		return Bit::BitLength(input);
	}
};

static unique_ptr<BaseStatistics> LengthPropagateStats(ClientContext &context, FunctionStatisticsInput &input) {
	auto &child_stats = input.child_stats;
	auto &expr = input.expr;
	D_ASSERT(child_stats.size() == 1);
	// can only propagate stats if the children have stats
	if (!StringStats::CanContainUnicode(child_stats[0])) {
		expr.function.function = ScalarFunction::UnaryFunction<string_t, int64_t, StrLenOperator>;
	}
	return nullptr;
}

static unique_ptr<FunctionData> ListLengthBind(ClientContext &context, ScalarFunction &bound_function,
                                               vector<unique_ptr<Expression>> &arguments) {
	if (arguments[0]->HasParameter()) {
		throw ParameterNotResolvedException();
	}
	bound_function.arguments[0] = arguments[0]->return_type;
	return nullptr;
}

void LengthFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunction array_length_unary =
	    ScalarFunction({LogicalType::LIST(LogicalType::ANY)}, LogicalType::BIGINT,
	                   ScalarFunction::UnaryFunction<list_entry_t, int64_t, ArrayLengthOperator>, ListLengthBind);
	ScalarFunctionSet length("length");
	length.AddFunction(ScalarFunction({LogicalType::VARCHAR}, LogicalType::BIGINT,
	                                  ScalarFunction::UnaryFunction<string_t, int64_t, StringLengthOperator>, nullptr,
	                                  nullptr, LengthPropagateStats));
	length.AddFunction(ScalarFunction({LogicalType::BIT}, LogicalType::BIGINT,
	                                  ScalarFunction::UnaryFunction<string_t, int64_t, BitStringLenOperator>));
	length.AddFunction(array_length_unary);
	set.AddFunction(length);
	length.name = "len";
	set.AddFunction(length);

	ScalarFunctionSet length_grapheme("length_grapheme");
	length_grapheme.AddFunction(ScalarFunction({LogicalType::VARCHAR}, LogicalType::BIGINT,
	                                           ScalarFunction::UnaryFunction<string_t, int64_t, GraphemeCountOperator>,
	                                           nullptr, nullptr, LengthPropagateStats));
	set.AddFunction(length_grapheme);

	ScalarFunctionSet array_length("array_length");
	array_length.AddFunction(array_length_unary);
	array_length.AddFunction(ScalarFunction(
	    {LogicalType::LIST(LogicalType::ANY), LogicalType::BIGINT}, LogicalType::BIGINT,
	    ScalarFunction::BinaryFunction<list_entry_t, int64_t, int64_t, ArrayLengthBinaryOperator>, ListLengthBind));
	set.AddFunction(array_length);

	set.AddFunction(ScalarFunction("strlen", {LogicalType::VARCHAR}, LogicalType::BIGINT,
	                               ScalarFunction::UnaryFunction<string_t, int64_t, StrLenOperator>));
	ScalarFunctionSet bit_length("bit_length");
	bit_length.AddFunction(ScalarFunction({LogicalType::VARCHAR}, LogicalType::BIGINT,
	                                      ScalarFunction::UnaryFunction<string_t, int64_t, BitLenOperator>));
	bit_length.AddFunction(ScalarFunction({LogicalType::BIT}, LogicalType::BIGINT,
	                                      ScalarFunction::UnaryFunction<string_t, int64_t, BitStringLenOperator>));
	set.AddFunction(bit_length);
	// length for BLOB type
	ScalarFunctionSet octet_length("octet_length");
	octet_length.AddFunction(ScalarFunction({LogicalType::BLOB}, LogicalType::BIGINT,
	                                        ScalarFunction::UnaryFunction<string_t, int64_t, StrLenOperator>));
	octet_length.AddFunction(ScalarFunction({LogicalType::BIT}, LogicalType::BIGINT,
	                                        ScalarFunction::UnaryFunction<string_t, int64_t, OctetLenOperator>));
	set.AddFunction(octet_length);
}

} // namespace duckdb







namespace duckdb {

struct StandardCharacterReader {
	static char Operation(const char *data, idx_t pos) {
		return data[pos];
	}
};

struct ASCIILCaseReader {
	static char Operation(const char *data, idx_t pos) {
		return (char)LowerFun::ascii_to_lower_map[(uint8_t)data[pos]];
	}
};

template <char PERCENTAGE, char UNDERSCORE, bool HAS_ESCAPE, class READER = StandardCharacterReader>
bool TemplatedLikeOperator(const char *sdata, idx_t slen, const char *pdata, idx_t plen, char escape) {
	idx_t pidx = 0;
	idx_t sidx = 0;
	for (; pidx < plen && sidx < slen; pidx++) {
		char pchar = READER::Operation(pdata, pidx);
		char schar = READER::Operation(sdata, sidx);
		if (HAS_ESCAPE && pchar == escape) {
			pidx++;
			if (pidx == plen) {
				throw SyntaxException("Like pattern must not end with escape character!");
			}
			if (pdata[pidx] != schar) {
				return false;
			}
			sidx++;
		} else if (pchar == UNDERSCORE) {
			sidx++;
		} else if (pchar == PERCENTAGE) {
			pidx++;
			while (pidx < plen && pdata[pidx] == PERCENTAGE) {
				pidx++;
			}
			if (pidx == plen) {
				return true; /* tail is acceptable */
			}
			for (; sidx < slen; sidx++) {
				if (TemplatedLikeOperator<PERCENTAGE, UNDERSCORE, HAS_ESCAPE, READER>(
				        sdata + sidx, slen - sidx, pdata + pidx, plen - pidx, escape)) {
					return true;
				}
			}
			return false;
		} else if (pchar == schar) {
			sidx++;
		} else {
			return false;
		}
	}
	while (pidx < plen && pdata[pidx] == PERCENTAGE) {
		pidx++;
	}
	return pidx == plen && sidx == slen;
}

struct LikeSegment {
	explicit LikeSegment(string pattern) : pattern(std::move(pattern)) {
	}

	string pattern;
};

struct LikeMatcher : public FunctionData {
	LikeMatcher(string like_pattern_p, vector<LikeSegment> segments, bool has_start_percentage, bool has_end_percentage)
	    : like_pattern(std::move(like_pattern_p)), segments(std::move(segments)),
	      has_start_percentage(has_start_percentage), has_end_percentage(has_end_percentage) {
	}

	bool Match(string_t &str) {
		auto str_data = (const unsigned char *)str.GetData();
		auto str_len = str.GetSize();
		idx_t segment_idx = 0;
		idx_t end_idx = segments.size() - 1;
		if (!has_start_percentage) {
			// no start sample_size: match the first part of the string directly
			auto &segment = segments[0];
			if (str_len < segment.pattern.size()) {
				return false;
			}
			if (memcmp(str_data, segment.pattern.c_str(), segment.pattern.size()) != 0) {
				return false;
			}
			str_data += segment.pattern.size();
			str_len -= segment.pattern.size();
			segment_idx++;
			if (segments.size() == 1) {
				// only one segment, and it matches
				// we have a match if there is an end sample_size, OR if the memcmp was an exact match (remaining str is
				// empty)
				return has_end_percentage || str_len == 0;
			}
		}
		// main match loop: for every segment in the middle, use Contains to find the needle in the haystack
		for (; segment_idx < end_idx; segment_idx++) {
			auto &segment = segments[segment_idx];
			// find the pattern of the current segment
			idx_t next_offset = ContainsFun::Find(str_data, str_len, (const unsigned char *)segment.pattern.c_str(),
			                                      segment.pattern.size());
			if (next_offset == DConstants::INVALID_INDEX) {
				// could not find this pattern in the string: no match
				return false;
			}
			idx_t offset = next_offset + segment.pattern.size();
			str_data += offset;
			str_len -= offset;
		}
		if (!has_end_percentage) {
			end_idx--;
			// no end sample_size: match the final segment now
			auto &segment = segments.back();
			if (str_len < segment.pattern.size()) {
				return false;
			}
			if (memcmp(str_data + str_len - segment.pattern.size(), segment.pattern.c_str(), segment.pattern.size()) !=
			    0) {
				return false;
			}
			return true;
		} else {
			auto &segment = segments.back();
			// find the pattern of the current segment
			idx_t next_offset = ContainsFun::Find(str_data, str_len, (const unsigned char *)segment.pattern.c_str(),
			                                      segment.pattern.size());
			return next_offset != DConstants::INVALID_INDEX;
		}
	}

	static unique_ptr<LikeMatcher> CreateLikeMatcher(string like_pattern, char escape = '\0') {
		vector<LikeSegment> segments;
		idx_t last_non_pattern = 0;
		bool has_start_percentage = false;
		bool has_end_percentage = false;
		for (idx_t i = 0; i < like_pattern.size(); i++) {
			auto ch = like_pattern[i];
			if (ch == escape || ch == '%' || ch == '_') {
				// special character, push a constant pattern
				if (i > last_non_pattern) {
					segments.emplace_back(like_pattern.substr(last_non_pattern, i - last_non_pattern));
				}
				last_non_pattern = i + 1;
				if (ch == escape || ch == '_') {
					// escape or underscore: could not create efficient like matcher
					// FIXME: we could handle escaped percentages here
					return nullptr;
				} else {
					// sample_size
					if (i == 0) {
						has_start_percentage = true;
					}
					if (i + 1 == like_pattern.size()) {
						has_end_percentage = true;
					}
				}
			}
		}
		if (last_non_pattern < like_pattern.size()) {
			segments.emplace_back(like_pattern.substr(last_non_pattern, like_pattern.size() - last_non_pattern));
		}
		if (segments.empty()) {
			return nullptr;
		}
		return make_uniq<LikeMatcher>(std::move(like_pattern), std::move(segments), has_start_percentage,
		                              has_end_percentage);
	}

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<LikeMatcher>(like_pattern, segments, has_start_percentage, has_end_percentage);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = (const LikeMatcher &)other_p;
		return like_pattern == other.like_pattern;
	}

private:
	string like_pattern;
	vector<LikeSegment> segments;
	bool has_start_percentage;
	bool has_end_percentage;
};

static unique_ptr<FunctionData> LikeBindFunction(ClientContext &context, ScalarFunction &bound_function,
                                                 vector<unique_ptr<Expression>> &arguments) {
	// pattern is the second argument. If its constant, we can already prepare the pattern and store it for later.
	D_ASSERT(arguments.size() == 2 || arguments.size() == 3);
	if (arguments[1]->IsFoldable()) {
		Value pattern_str = ExpressionExecutor::EvaluateScalar(context, *arguments[1]);
		if (pattern_str.IsNull()) {
			return nullptr;
		}
		return LikeMatcher::CreateLikeMatcher(pattern_str.ToString());
	}
	return nullptr;
}

bool LikeOperatorFunction(const char *s, idx_t slen, const char *pattern, idx_t plen, char escape) {
	return TemplatedLikeOperator<'%', '_', true>(s, slen, pattern, plen, escape);
}

bool LikeOperatorFunction(const char *s, idx_t slen, const char *pattern, idx_t plen) {
	return TemplatedLikeOperator<'%', '_', false>(s, slen, pattern, plen, '\0');
}

bool LikeOperatorFunction(string_t &s, string_t &pat) {
	return LikeOperatorFunction(s.GetData(), s.GetSize(), pat.GetData(), pat.GetSize());
}

bool LikeOperatorFunction(string_t &s, string_t &pat, char escape) {
	return LikeOperatorFunction(s.GetData(), s.GetSize(), pat.GetData(), pat.GetSize(), escape);
}

bool LikeFun::Glob(const char *string, idx_t slen, const char *pattern, idx_t plen, bool allow_question_mark) {
	idx_t sidx = 0;
	idx_t pidx = 0;
main_loop : {
	// main matching loop
	while (sidx < slen && pidx < plen) {
		char s = string[sidx];
		char p = pattern[pidx];
		switch (p) {
		case '*': {
			// asterisk: match any set of characters
			// skip any subsequent asterisks
			pidx++;
			while (pidx < plen && pattern[pidx] == '*') {
				pidx++;
			}
			// if the asterisk is the last character, the pattern always matches
			if (pidx == plen) {
				return true;
			}
			// recursively match the remainder of the pattern
			for (; sidx < slen; sidx++) {
				if (LikeFun::Glob(string + sidx, slen - sidx, pattern + pidx, plen - pidx)) {
					return true;
				}
			}
			return false;
		}
		case '?':
			// when enabled: matches anything but null
			if (allow_question_mark) {
				break;
			}
			DUCKDB_EXPLICIT_FALLTHROUGH;
		case '[':
			pidx++;
			goto parse_bracket;
		case '\\':
			// escape character, next character needs to match literally
			pidx++;
			// check that we still have a character remaining
			if (pidx == plen) {
				return false;
			}
			p = pattern[pidx];
			if (s != p) {
				return false;
			}
			break;
		default:
			// not a control character: characters need to match literally
			if (s != p) {
				return false;
			}
			break;
		}
		sidx++;
		pidx++;
	}
	while (pidx < plen && pattern[pidx] == '*') {
		pidx++;
	}
	// we are finished only if we have consumed the full pattern
	return pidx == plen && sidx == slen;
}
parse_bracket : {
	// inside a bracket
	if (pidx == plen) {
		return false;
	}
	// check the first character
	// if it is an exclamation mark we need to invert our logic
	char p = pattern[pidx];
	char s = string[sidx];
	bool invert = false;
	if (p == '!') {
		invert = true;
		pidx++;
	}
	bool found_match = invert;
	idx_t start_pos = pidx;
	bool found_closing_bracket = false;
	// now check the remainder of the pattern
	while (pidx < plen) {
		p = pattern[pidx];
		// if the first character is a closing bracket, we match it literally
		// otherwise it indicates an end of bracket
		if (p == ']' && pidx > start_pos) {
			// end of bracket found: we are done
			found_closing_bracket = true;
			pidx++;
			break;
		}
		// we either match a range (a-b) or a single character (a)
		// check if the next character is a dash
		if (pidx + 1 == plen) {
			// no next character!
			break;
		}
		bool matches;
		if (pattern[pidx + 1] == '-') {
			// range! find the next character in the range
			if (pidx + 2 == plen) {
				break;
			}
			char next_char = pattern[pidx + 2];
			// check if the current character is within the range
			matches = s >= p && s <= next_char;
			// shift the pattern forward past the range
			pidx += 3;
		} else {
			// no range! perform a direct match
			matches = p == s;
			// shift the pattern forward past the character
			pidx++;
		}
		if (found_match == invert && matches) {
			// found a match! set the found_matches flag
			// we keep on pattern matching after this until we reach the end bracket
			// however, we don't need to update the found_match flag anymore
			found_match = !invert;
		}
	}
	if (!found_closing_bracket) {
		// no end of bracket: invalid pattern
		return false;
	}
	if (!found_match) {
		// did not match the bracket: return false;
		return false;
	}
	// finished the bracket matching: move forward
	sidx++;
	goto main_loop;
}
}

static char GetEscapeChar(string_t escape) {
	// Only one escape character should be allowed
	if (escape.GetSize() > 1) {
		throw SyntaxException("Invalid escape string. Escape string must be empty or one character.");
	}
	return escape.GetSize() == 0 ? '\0' : *escape.GetData();
}

struct LikeEscapeOperator {
	template <class TA, class TB, class TC>
	static inline bool Operation(TA str, TB pattern, TC escape) {
		char escape_char = GetEscapeChar(escape);
		return LikeOperatorFunction(str.GetData(), str.GetSize(), pattern.GetData(), pattern.GetSize(), escape_char);
	}
};

struct NotLikeEscapeOperator {
	template <class TA, class TB, class TC>
	static inline bool Operation(TA str, TB pattern, TC escape) {
		return !LikeEscapeOperator::Operation(str, pattern, escape);
	}
};

struct LikeOperator {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA str, TB pattern) {
		return LikeOperatorFunction(str, pattern);
	}
};

bool ILikeOperatorFunction(string_t &str, string_t &pattern, char escape = '\0') {
	auto str_data = str.GetData();
	auto str_size = str.GetSize();
	auto pat_data = pattern.GetData();
	auto pat_size = pattern.GetSize();

	// lowercase both the str and the pattern
	idx_t str_llength = LowerFun::LowerLength(str_data, str_size);
	auto str_ldata = make_unsafe_uniq_array<char>(str_llength);
	LowerFun::LowerCase(str_data, str_size, str_ldata.get());

	idx_t pat_llength = LowerFun::LowerLength(pat_data, pat_size);
	auto pat_ldata = make_unsafe_uniq_array<char>(pat_llength);
	LowerFun::LowerCase(pat_data, pat_size, pat_ldata.get());
	string_t str_lcase(str_ldata.get(), str_llength);
	string_t pat_lcase(pat_ldata.get(), pat_llength);
	return LikeOperatorFunction(str_lcase, pat_lcase, escape);
}

struct ILikeEscapeOperator {
	template <class TA, class TB, class TC>
	static inline bool Operation(TA str, TB pattern, TC escape) {
		char escape_char = GetEscapeChar(escape);
		return ILikeOperatorFunction(str, pattern, escape_char);
	}
};

struct NotILikeEscapeOperator {
	template <class TA, class TB, class TC>
	static inline bool Operation(TA str, TB pattern, TC escape) {
		return !ILikeEscapeOperator::Operation(str, pattern, escape);
	}
};

struct ILikeOperator {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA str, TB pattern) {
		return ILikeOperatorFunction(str, pattern);
	}
};

struct NotLikeOperator {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA str, TB pattern) {
		return !LikeOperatorFunction(str, pattern);
	}
};

struct NotILikeOperator {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA str, TB pattern) {
		return !ILikeOperator::Operation<TA, TB, TR>(str, pattern);
	}
};

struct ILikeOperatorASCII {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA str, TB pattern) {
		return TemplatedLikeOperator<'%', '_', false, ASCIILCaseReader>(str.GetData(), str.GetSize(), pattern.GetData(),
		                                                                pattern.GetSize(), '\0');
	}
};

struct NotILikeOperatorASCII {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA str, TB pattern) {
		return !ILikeOperatorASCII::Operation<TA, TB, TR>(str, pattern);
	}
};

struct GlobOperator {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA str, TB pattern) {
		return LikeFun::Glob(str.GetData(), str.GetSize(), pattern.GetData(), pattern.GetSize());
	}
};

// This can be moved to the scalar_function class
template <typename FUNC>
static void LikeEscapeFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &str = args.data[0];
	auto &pattern = args.data[1];
	auto &escape = args.data[2];

	TernaryExecutor::Execute<string_t, string_t, string_t, bool>(
	    str, pattern, escape, result, args.size(), FUNC::template Operation<string_t, string_t, string_t>);
}

template <class ASCII_OP>
static unique_ptr<BaseStatistics> ILikePropagateStats(ClientContext &context, FunctionStatisticsInput &input) {
	auto &child_stats = input.child_stats;
	auto &expr = input.expr;
	D_ASSERT(child_stats.size() >= 1);
	// can only propagate stats if the children have stats
	if (!StringStats::CanContainUnicode(child_stats[0])) {
		expr.function.function = ScalarFunction::BinaryFunction<string_t, string_t, bool, ASCII_OP>;
	}
	return nullptr;
}

template <class OP, bool INVERT>
static void RegularLikeFunction(DataChunk &input, ExpressionState &state, Vector &result) {
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	if (func_expr.bind_info) {
		auto &matcher = (LikeMatcher &)*func_expr.bind_info;
		// use fast like matcher
		UnaryExecutor::Execute<string_t, bool>(input.data[0], result, input.size(), [&](string_t input) {
			return INVERT ? !matcher.Match(input) : matcher.Match(input);
		});
	} else {
		// use generic like matcher
		BinaryExecutor::ExecuteStandard<string_t, string_t, bool, OP>(input.data[0], input.data[1], result,
		                                                              input.size());
	}
}
void LikeFun::RegisterFunction(BuiltinFunctions &set) {
	// like
	set.AddFunction(ScalarFunction("~~", {LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::BOOLEAN,
	                               RegularLikeFunction<LikeOperator, false>, LikeBindFunction));
	// not like
	set.AddFunction(ScalarFunction("!~~", {LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::BOOLEAN,
	                               RegularLikeFunction<NotLikeOperator, true>, LikeBindFunction));
	// glob
	set.AddFunction(ScalarFunction("~~~", {LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::BOOLEAN,
	                               ScalarFunction::BinaryFunction<string_t, string_t, bool, GlobOperator>));
	// ilike
	set.AddFunction(ScalarFunction("~~*", {LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::BOOLEAN,
	                               ScalarFunction::BinaryFunction<string_t, string_t, bool, ILikeOperator>, nullptr,
	                               nullptr, ILikePropagateStats<ILikeOperatorASCII>));
	// not ilike
	set.AddFunction(ScalarFunction("!~~*", {LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::BOOLEAN,
	                               ScalarFunction::BinaryFunction<string_t, string_t, bool, NotILikeOperator>, nullptr,
	                               nullptr, ILikePropagateStats<NotILikeOperatorASCII>));
}

void LikeEscapeFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction({"like_escape"}, ScalarFunction({LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR},
	                                                LogicalType::BOOLEAN, LikeEscapeFunction<LikeEscapeOperator>));
	set.AddFunction({"not_like_escape"},
	                ScalarFunction({LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR},
	                               LogicalType::BOOLEAN, LikeEscapeFunction<NotLikeEscapeOperator>));

	set.AddFunction({"ilike_escape"}, ScalarFunction({LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR},
	                                                 LogicalType::BOOLEAN, LikeEscapeFunction<ILikeEscapeOperator>));
	set.AddFunction({"not_ilike_escape"},
	                ScalarFunction({LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR},
	                               LogicalType::BOOLEAN, LikeEscapeFunction<NotILikeEscapeOperator>));
}
} // namespace duckdb




namespace duckdb {

struct NFCNormalizeOperator {
	template <class INPUT_TYPE, class RESULT_TYPE>
	static RESULT_TYPE Operation(INPUT_TYPE input, Vector &result) {
		auto input_data = input.GetData();
		auto input_length = input.GetSize();
		if (StripAccentsFun::IsAscii(input_data, input_length)) {
			return input;
		}
		auto normalized_str = Utf8Proc::Normalize(input_data, input_length);
		D_ASSERT(normalized_str);
		auto result_str = StringVector::AddString(result, normalized_str);
		free(normalized_str);
		return result_str;
	}
};

static void NFCNormalizeFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(args.ColumnCount() == 1);

	UnaryExecutor::ExecuteString<string_t, string_t, NFCNormalizeOperator>(args.data[0], result, args.size());
	StringVector::AddHeapReference(result, args.data[0]);
}

ScalarFunction NFCNormalizeFun::GetFunction() {
	return ScalarFunction("nfc_normalize", {LogicalType::VARCHAR}, LogicalType::VARCHAR, NFCNormalizeFunction);
}

void NFCNormalizeFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(NFCNormalizeFun::GetFunction());
}

} // namespace duckdb





namespace duckdb {

static bool PrefixFunction(const string_t &str, const string_t &pattern);

struct PrefixOperator {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA left, TB right) {
		return PrefixFunction(left, right);
	}
};
static bool PrefixFunction(const string_t &str, const string_t &pattern) {
	auto str_length = str.GetSize();
	auto patt_length = pattern.GetSize();
	if (patt_length > str_length) {
		return false;
	}
	if (patt_length <= string_t::PREFIX_LENGTH) {
		// short prefix
		if (patt_length == 0) {
			// length = 0, return true
			return true;
		}

		// prefix early out
		const char *str_pref = str.GetPrefix();
		const char *patt_pref = pattern.GetPrefix();
		for (idx_t i = 0; i < patt_length; ++i) {
			if (str_pref[i] != patt_pref[i]) {
				return false;
			}
		}
		return true;
	} else {
		// prefix early out
		const char *str_pref = str.GetPrefix();
		const char *patt_pref = pattern.GetPrefix();
		for (idx_t i = 0; i < string_t::PREFIX_LENGTH; ++i) {
			if (str_pref[i] != patt_pref[i]) {
				// early out
				return false;
			}
		}
		// compare the rest of the prefix
		const char *str_data = str.GetData();
		const char *patt_data = pattern.GetData();
		D_ASSERT(patt_length <= str_length);
		for (idx_t i = string_t::PREFIX_LENGTH; i < patt_length; ++i) {
			if (str_data[i] != patt_data[i]) {
				return false;
			}
		}
		return true;
	}
}

ScalarFunction PrefixFun::GetFunction() {
	return ScalarFunction("prefix",                                     // name of the function
	                      {LogicalType::VARCHAR, LogicalType::VARCHAR}, // argument list
	                      LogicalType::BOOLEAN,                         // return type
	                      ScalarFunction::BinaryFunction<string_t, string_t, bool, PrefixOperator>);
}

void PrefixFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(GetFunction());
}

} // namespace duckdb






namespace duckdb {

using regexp_util::CreateStringPiece;
using regexp_util::Extract;
using regexp_util::ParseRegexOptions;
using regexp_util::TryParseConstantPattern;

unique_ptr<FunctionLocalState>
RegexpExtractAll::InitLocalState(ExpressionState &state, const BoundFunctionExpression &expr, FunctionData *bind_data) {
	auto &info = bind_data->Cast<RegexpBaseBindData>();
	if (info.constant_pattern) {
		return make_uniq<RegexLocalState>(info, true);
	}
	return nullptr;
}

// Forwards startpos automatically
bool ExtractAll(duckdb_re2::StringPiece &input, duckdb_re2::RE2 &pattern, idx_t *startpos,
                duckdb_re2::StringPiece *groups, int ngroups) {

	D_ASSERT(pattern.ok());
	D_ASSERT(pattern.NumberOfCapturingGroups() == ngroups);

	if (!pattern.Match(input, *startpos, input.size(), pattern.Anchored(), groups, ngroups + 1)) {
		return false;
	}
	idx_t consumed = static_cast<size_t>(groups[0].end() - (input.begin() + *startpos));
	if (!consumed) {
		// Empty match found, have to manually forward the input
		// to avoid an infinite loop
		// FIXME: support unicode characters
		consumed++;
		while (*startpos + consumed < input.length() && !LengthFun::IsCharacter(input[*startpos + consumed])) {
			consumed++;
		}
	}
	*startpos += consumed;
	return true;
}

void ExtractSingleTuple(const string_t &string, duckdb_re2::RE2 &pattern, int32_t group, RegexStringPieceArgs &args,
                        Vector &result, idx_t row) {
	auto input = CreateStringPiece(string);

	auto &child_vector = ListVector::GetEntry(result);
	auto list_content = FlatVector::GetData<string_t>(child_vector);
	auto &child_validity = FlatVector::Validity(child_vector);

	auto current_list_size = ListVector::GetListSize(result);
	auto current_list_capacity = ListVector::GetListCapacity(result);

	auto result_data = FlatVector::GetData<list_entry_t>(result);
	auto &list_entry = result_data[row];
	list_entry.offset = current_list_size;

	if (group < 0) {
		list_entry.length = 0;
		return;
	}
	// If the requested group index is out of bounds
	// we want to throw only if there is a match
	bool throw_on_group_found = (idx_t)group > args.size;

	idx_t startpos = 0;
	for (idx_t iteration = 0; ExtractAll(input, pattern, &startpos, args.group_buffer, args.size); iteration++) {
		if (!iteration && throw_on_group_found) {
			throw InvalidInputException("Pattern has %d groups. Cannot access group %d", args.size, group);
		}

		// Make sure we have enough room for the new entries
		if (current_list_size + 1 >= current_list_capacity) {
			ListVector::Reserve(result, current_list_capacity * 2);
			current_list_capacity = ListVector::GetListCapacity(result);
			list_content = FlatVector::GetData<string_t>(child_vector);
		}

		// Write the captured groups into the list-child vector
		auto &match_group = args.group_buffer[group];

		idx_t child_idx = current_list_size;
		if (match_group.empty()) {
			// This group was not matched
			list_content[child_idx] = string_t(string.GetData(), 0);
			if (match_group.begin() == nullptr) {
				// This group is optional
				child_validity.SetInvalid(child_idx);
			}
		} else {
			// Every group is a substring of the original, we can find out the offset using the pointer
			// the 'match_group' address is guaranteed to be bigger than that of the source
			D_ASSERT((const char *)match_group.begin() >= string.GetData());
			idx_t offset = match_group.begin() - string.GetData();
			list_content[child_idx] = string_t(string.GetData() + offset, match_group.size());
		}
		current_list_size++;
		if (startpos > input.size()) {
			// Empty match found at the end of the string
			break;
		}
	}
	list_entry.length = current_list_size - list_entry.offset;
	ListVector::SetListSize(result, current_list_size);
}

int32_t GetGroupIndex(DataChunk &args, idx_t row, int32_t &result) {
	if (args.ColumnCount() < 3) {
		result = 0;
		return true;
	}
	UnifiedVectorFormat format;
	args.data[2].ToUnifiedFormat(args.size(), format);
	idx_t index = format.sel->get_index(row);
	if (!format.validity.RowIsValid(index)) {
		return false;
	}
	result = ((int32_t *)format.data)[index];
	return true;
}

duckdb_re2::RE2 &GetPattern(const RegexpBaseBindData &info, ExpressionState &state,
                            unique_ptr<duckdb_re2::RE2> &pattern_p) {
	if (info.constant_pattern) {
		auto &lstate = ExecuteFunctionState::GetFunctionState(state)->Cast<RegexLocalState>();
		return lstate.constant_pattern;
	}
	D_ASSERT(pattern_p);
	return *pattern_p;
}

RegexStringPieceArgs &GetGroupsBuffer(const RegexpBaseBindData &info, ExpressionState &state,
                                      unique_ptr<RegexStringPieceArgs> &groups_p) {
	if (info.constant_pattern) {
		auto &lstate = ExecuteFunctionState::GetFunctionState(state)->Cast<RegexLocalState>();
		return lstate.group_buffer;
	}
	D_ASSERT(groups_p);
	return *groups_p;
}

void RegexpExtractAll::Execute(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	const auto &info = func_expr.bind_info->Cast<RegexpBaseBindData>();

	auto &strings = args.data[0];
	auto &patterns = args.data[1];
	D_ASSERT(result.GetType().id() == LogicalTypeId::LIST);
	auto &output_child = ListVector::GetEntry(result);

	UnifiedVectorFormat strings_data;
	strings.ToUnifiedFormat(args.size(), strings_data);

	UnifiedVectorFormat pattern_data;
	patterns.ToUnifiedFormat(args.size(), pattern_data);

	ListVector::Reserve(result, STANDARD_VECTOR_SIZE);
	// Reference the 'strings' StringBuffer, because we won't need to allocate new data
	// for the result, all returned strings are substrings of the originals
	output_child.SetAuxiliary(strings.GetAuxiliary());

	// Avoid doing extra work if all the inputs are constant
	idx_t tuple_count = args.AllConstant() ? 1 : args.size();

	unique_ptr<RegexStringPieceArgs> non_const_args;
	unique_ptr<duckdb_re2::RE2> stored_re;
	if (!info.constant_pattern) {
		non_const_args = make_uniq<RegexStringPieceArgs>();
	} else {
		// Verify that the constant pattern is valid
		auto &re = GetPattern(info, state, stored_re);
		auto group_count_p = re.NumberOfCapturingGroups();
		if (group_count_p == -1) {
			throw InvalidInputException("Pattern failed to parse, error: '%s'", re.error());
		}
	}

	for (idx_t row = 0; row < tuple_count; row++) {
		bool pattern_valid = true;
		if (!info.constant_pattern) {
			// Check if the pattern is NULL or not,
			// and compile the pattern if it's not constant
			auto pattern_idx = pattern_data.sel->get_index(row);
			if (!pattern_data.validity.RowIsValid(pattern_idx)) {
				pattern_valid = false;
			} else {
				auto &pattern_p = ((string_t *)pattern_data.data)[pattern_idx];
				auto pattern_strpiece = CreateStringPiece(pattern_p);
				stored_re = make_uniq<duckdb_re2::RE2>(pattern_strpiece, info.options);

				// Increase the size of the args buffer if needed
				auto group_count_p = stored_re->NumberOfCapturingGroups();
				if (group_count_p == -1) {
					throw InvalidInputException("Pattern failed to parse, error: '%s'", stored_re->error());
				}
				non_const_args->SetSize(group_count_p);
			}
		}

		auto string_idx = strings_data.sel->get_index(row);
		int32_t group_index;
		if (!pattern_valid || !strings_data.validity.RowIsValid(string_idx) || !GetGroupIndex(args, row, group_index)) {
			// If something is NULL, the result is NULL
			// FIXME: do we even need 'SPECIAL_HANDLING'?
			auto result_data = FlatVector::GetData<list_entry_t>(result);
			auto &result_validity = FlatVector::Validity(result);
			result_data[row].length = 0;
			result_data[row].offset = ListVector::GetListSize(result);
			result_validity.SetInvalid(row);
			continue;
		}

		auto &re = GetPattern(info, state, stored_re);
		auto &groups = GetGroupsBuffer(info, state, non_const_args);
		auto &string = ((string_t *)strings_data.data)[string_idx];
		ExtractSingleTuple(string, re, group_index, groups, result, row);
	}

	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

unique_ptr<FunctionData> RegexpExtractAll::Bind(ClientContext &context, ScalarFunction &bound_function,
                                                vector<unique_ptr<Expression>> &arguments) {
	D_ASSERT(arguments.size() >= 2);

	duckdb_re2::RE2::Options options;

	string constant_string;
	bool constant_pattern = TryParseConstantPattern(context, *arguments[1], constant_string);

	if (arguments.size() >= 4) {
		ParseRegexOptions(context, *arguments[3], options);
	}
	return make_uniq<RegexpExtractBindData>(options, std::move(constant_string), constant_pattern, "");
}

} // namespace duckdb



namespace duckdb {

namespace regexp_util {

bool TryParseConstantPattern(ClientContext &context, Expression &expr, string &constant_string) {
	if (!expr.IsFoldable()) {
		return false;
	}
	Value pattern_str = ExpressionExecutor::EvaluateScalar(context, expr);
	if (!pattern_str.IsNull() && pattern_str.type().id() == LogicalTypeId::VARCHAR) {
		constant_string = StringValue::Get(pattern_str);
		return true;
	}
	return false;
}

void ParseRegexOptions(const string &options, duckdb_re2::RE2::Options &result, bool *global_replace) {
	for (idx_t i = 0; i < options.size(); i++) {
		switch (options[i]) {
		case 'c':
			// case-sensitive matching
			result.set_case_sensitive(true);
			break;
		case 'i':
			// case-insensitive matching
			result.set_case_sensitive(false);
			break;
		case 'l':
			// literal matching
			result.set_literal(true);
			break;
		case 'm':
		case 'n':
		case 'p':
			// newline-sensitive matching
			result.set_dot_nl(false);
			break;
		case 's':
			// non-newline-sensitive matching
			result.set_dot_nl(true);
			break;
		case 'g':
			// global replace, only available for regexp_replace
			if (global_replace) {
				*global_replace = true;
			} else {
				throw InvalidInputException("Option 'g' (global replace) is only valid for regexp_replace");
			}
			break;
		case ' ':
		case '\t':
		case '\n':
			// ignore whitespace
			break;
		default:
			throw InvalidInputException("Unrecognized Regex option %c", options[i]);
		}
	}
}

void ParseRegexOptions(ClientContext &context, Expression &expr, RE2::Options &target, bool *global_replace) {
	if (expr.HasParameter()) {
		throw ParameterNotResolvedException();
	}
	if (!expr.IsFoldable()) {
		throw InvalidInputException("Regex options field must be a constant");
	}
	Value options_str = ExpressionExecutor::EvaluateScalar(context, expr);
	if (!options_str.IsNull() && options_str.type().id() == LogicalTypeId::VARCHAR) {
		ParseRegexOptions(StringValue::Get(options_str), target, global_replace);
	}
}

} // namespace regexp_util

} // namespace duckdb












namespace duckdb {

using regexp_util::CreateStringPiece;
using regexp_util::Extract;
using regexp_util::ParseRegexOptions;
using regexp_util::TryParseConstantPattern;

static bool RegexOptionsEquals(const duckdb_re2::RE2::Options &opt_a, const duckdb_re2::RE2::Options &opt_b) {
	return opt_a.case_sensitive() == opt_b.case_sensitive();
}

RegexpBaseBindData::RegexpBaseBindData() : constant_pattern(false) {
}
RegexpBaseBindData::RegexpBaseBindData(duckdb_re2::RE2::Options options, string constant_string_p,
                                       bool constant_pattern)
    : options(options), constant_string(std::move(constant_string_p)), constant_pattern(constant_pattern) {
}

RegexpBaseBindData::~RegexpBaseBindData() {
}

bool RegexpBaseBindData::Equals(const FunctionData &other_p) const {
	auto &other = other_p.Cast<RegexpBaseBindData>();
	return constant_pattern == other.constant_pattern && constant_string == other.constant_string &&
	       RegexOptionsEquals(options, other.options);
}

unique_ptr<FunctionLocalState> RegexInitLocalState(ExpressionState &state, const BoundFunctionExpression &expr,
                                                   FunctionData *bind_data) {
	auto &info = bind_data->Cast<RegexpBaseBindData>();
	if (info.constant_pattern) {
		return make_uniq<RegexLocalState>(info);
	}
	return nullptr;
}

//===--------------------------------------------------------------------===//
// Regexp Matches
//===--------------------------------------------------------------------===//
RegexpMatchesBindData::RegexpMatchesBindData(duckdb_re2::RE2::Options options, string constant_string_p,
                                             bool constant_pattern)
    : RegexpBaseBindData(options, std::move(constant_string_p), constant_pattern) {
	if (constant_pattern) {
		auto pattern = make_uniq<RE2>(constant_string, options);
		if (!pattern->ok()) {
			throw Exception(pattern->error());
		}

		range_success = pattern->PossibleMatchRange(&range_min, &range_max, 1000);
	} else {
		range_success = false;
	}
}

RegexpMatchesBindData::RegexpMatchesBindData(duckdb_re2::RE2::Options options, string constant_string_p,
                                             bool constant_pattern, string range_min_p, string range_max_p,
                                             bool range_success)
    : RegexpBaseBindData(options, std::move(constant_string_p), constant_pattern), range_min(std::move(range_min_p)),
      range_max(std::move(range_max_p)), range_success(range_success) {
}

unique_ptr<FunctionData> RegexpMatchesBindData::Copy() const {
	return make_uniq<RegexpMatchesBindData>(options, constant_string, constant_pattern, range_min, range_max,
	                                        range_success);
}

unique_ptr<FunctionData> RegexpMatchesBind(ClientContext &context, ScalarFunction &bound_function,
                                           vector<unique_ptr<Expression>> &arguments) {
	// pattern is the second argument. If its constant, we can already prepare the pattern and store it for later.
	D_ASSERT(arguments.size() == 2 || arguments.size() == 3);
	RE2::Options options;
	options.set_log_errors(false);
	if (arguments.size() == 3) {
		ParseRegexOptions(context, *arguments[2], options);
	}

	string constant_string;
	bool constant_pattern;
	constant_pattern = TryParseConstantPattern(context, *arguments[1], constant_string);
	return make_uniq<RegexpMatchesBindData>(options, std::move(constant_string), constant_pattern);
}

struct RegexPartialMatch {
	static inline bool Operation(const duckdb_re2::StringPiece &input, duckdb_re2::RE2 &re) {
		return duckdb_re2::RE2::PartialMatch(input, re);
	}
};

struct RegexFullMatch {
	static inline bool Operation(const duckdb_re2::StringPiece &input, duckdb_re2::RE2 &re) {
		return duckdb_re2::RE2::FullMatch(input, re);
	}
};

template <class OP>
static void RegexpMatchesFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &strings = args.data[0];
	auto &patterns = args.data[1];

	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &info = func_expr.bind_info->Cast<RegexpMatchesBindData>();

	if (info.constant_pattern) {
		auto &lstate = ExecuteFunctionState::GetFunctionState(state)->Cast<RegexLocalState>();
		UnaryExecutor::Execute<string_t, bool>(strings, result, args.size(), [&](string_t input) {
			return OP::Operation(CreateStringPiece(input), lstate.constant_pattern);
		});
	} else {
		BinaryExecutor::Execute<string_t, string_t, bool>(strings, patterns, result, args.size(),
		                                                  [&](string_t input, string_t pattern) {
			                                                  RE2 re(CreateStringPiece(pattern), info.options);
			                                                  if (!re.ok()) {
				                                                  throw Exception(re.error());
			                                                  }
			                                                  return OP::Operation(CreateStringPiece(input), re);
		                                                  });
	}
}

//===--------------------------------------------------------------------===//
// Regexp Replace
//===--------------------------------------------------------------------===//
RegexpReplaceBindData::RegexpReplaceBindData() : global_replace(false) {
}

RegexpReplaceBindData::RegexpReplaceBindData(duckdb_re2::RE2::Options options, string constant_string_p,
                                             bool constant_pattern, bool global_replace)
    : RegexpBaseBindData(options, std::move(constant_string_p), constant_pattern), global_replace(global_replace) {
}

unique_ptr<FunctionData> RegexpReplaceBindData::Copy() const {
	auto copy = make_uniq<RegexpReplaceBindData>(options, constant_string, constant_pattern, global_replace);
	return std::move(copy);
}

bool RegexpReplaceBindData::Equals(const FunctionData &other_p) const {
	auto &other = (const RegexpReplaceBindData &)other_p;
	return RegexpBaseBindData::Equals(other) && global_replace == other.global_replace;
}

static unique_ptr<FunctionData> RegexReplaceBind(ClientContext &context, ScalarFunction &bound_function,
                                                 vector<unique_ptr<Expression>> &arguments) {
	auto data = make_uniq<RegexpReplaceBindData>();

	data->constant_pattern = TryParseConstantPattern(context, *arguments[1], data->constant_string);
	if (arguments.size() == 4) {
		ParseRegexOptions(context, *arguments[3], data->options, &data->global_replace);
	}
	data->options.set_log_errors(false);
	return std::move(data);
}

static void RegexReplaceFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &info = func_expr.bind_info->Cast<RegexpReplaceBindData>();

	auto &strings = args.data[0];
	auto &patterns = args.data[1];
	auto &replaces = args.data[2];

	if (info.constant_pattern) {
		auto &lstate = ExecuteFunctionState::GetFunctionState(state)->Cast<RegexLocalState>();
		BinaryExecutor::Execute<string_t, string_t, string_t>(
		    strings, replaces, result, args.size(), [&](string_t input, string_t replace) {
			    std::string sstring = input.GetString();
			    if (info.global_replace) {
				    RE2::GlobalReplace(&sstring, lstate.constant_pattern, CreateStringPiece(replace));
			    } else {
				    RE2::Replace(&sstring, lstate.constant_pattern, CreateStringPiece(replace));
			    }
			    return StringVector::AddString(result, sstring);
		    });
	} else {
		TernaryExecutor::Execute<string_t, string_t, string_t, string_t>(
		    strings, patterns, replaces, result, args.size(), [&](string_t input, string_t pattern, string_t replace) {
			    RE2 re(CreateStringPiece(pattern), info.options);
			    std::string sstring = input.GetString();
			    if (info.global_replace) {
				    RE2::GlobalReplace(&sstring, re, CreateStringPiece(replace));
			    } else {
				    RE2::Replace(&sstring, re, CreateStringPiece(replace));
			    }
			    return StringVector::AddString(result, sstring);
		    });
	}
}

//===--------------------------------------------------------------------===//
// Regexp Extract
//===--------------------------------------------------------------------===//
RegexpExtractBindData::RegexpExtractBindData() {
}

RegexpExtractBindData::RegexpExtractBindData(duckdb_re2::RE2::Options options, string constant_string_p,
                                             bool constant_pattern, string group_string_p)
    : RegexpBaseBindData(options, std::move(constant_string_p), constant_pattern),
      group_string(std::move(group_string_p)), rewrite(group_string) {
}

unique_ptr<FunctionData> RegexpExtractBindData::Copy() const {
	return make_uniq<RegexpExtractBindData>(options, constant_string, constant_pattern, group_string);
}

bool RegexpExtractBindData::Equals(const FunctionData &other_p) const {
	auto &other = (const RegexpExtractBindData &)other_p;
	return RegexpBaseBindData::Equals(other) && group_string == other.group_string;
}

static void RegexExtractFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	const auto &info = func_expr.bind_info->Cast<RegexpExtractBindData>();

	auto &strings = args.data[0];
	auto &patterns = args.data[1];
	if (info.constant_pattern) {
		auto &lstate = ExecuteFunctionState::GetFunctionState(state)->Cast<RegexLocalState>();
		UnaryExecutor::Execute<string_t, string_t>(strings, result, args.size(), [&](string_t input) {
			return Extract(input, result, lstate.constant_pattern, info.rewrite);
		});
	} else {
		BinaryExecutor::Execute<string_t, string_t, string_t>(strings, patterns, result, args.size(),
		                                                      [&](string_t input, string_t pattern) {
			                                                      RE2 re(CreateStringPiece(pattern), info.options);
			                                                      return Extract(input, result, re, info.rewrite);
		                                                      });
	}
}

//===--------------------------------------------------------------------===//
// Regexp Extract Struct
//===--------------------------------------------------------------------===//
static void RegexExtractStructFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &lstate = ExecuteFunctionState::GetFunctionState(state)->Cast<RegexLocalState>();

	const auto count = args.size();
	auto &input = args.data[0];

	auto &child_entries = StructVector::GetEntries(result);
	const auto groupSize = child_entries.size();
	// Reference the 'input' StringBuffer, because we won't need to allocate new data
	// for the result, all returned strings are substrings of the originals
	for (auto &child_entry : child_entries) {
		child_entry->SetAuxiliary(input.GetAuxiliary());
	}

	vector<RE2::Arg> argv(groupSize);
	vector<RE2::Arg *> groups(groupSize);
	vector<duckdb_re2::StringPiece> ws(groupSize);
	for (size_t i = 0; i < groupSize; ++i) {
		groups[i] = &argv[i];
		argv[i] = &ws[i];
	}

	if (input.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);

		if (ConstantVector::IsNull(input)) {
			ConstantVector::SetNull(result, true);
		} else {
			ConstantVector::SetNull(result, false);
			auto idata = ConstantVector::GetData<string_t>(input);
			auto str = CreateStringPiece(idata[0]);
			auto match = duckdb_re2::RE2::PartialMatchN(str, lstate.constant_pattern, groups.data(), groups.size());
			for (size_t col = 0; col < child_entries.size(); ++col) {
				auto &child_entry = child_entries[col];
				ConstantVector::SetNull(*child_entry, false);
				auto &extracted = ws[col];
				auto cdata = ConstantVector::GetData<string_t>(*child_entry);
				cdata[0] = string_t(extracted.data(), match ? extracted.size() : 0);
			}
		}
	} else {
		UnifiedVectorFormat iunified;
		input.ToUnifiedFormat(count, iunified);

		const auto &ivalidity = iunified.validity;
		auto idata = (const string_t *)iunified.data;

		// Start with a valid flat vector
		result.SetVectorType(VectorType::FLAT_VECTOR);

		// Start with valid children
		for (size_t col = 0; col < child_entries.size(); ++col) {
			auto &child_entry = child_entries[col];
			child_entry->SetVectorType(VectorType::FLAT_VECTOR);
		}

		for (idx_t i = 0; i < count; ++i) {
			const auto idx = iunified.sel->get_index(i);
			if (ivalidity.RowIsValid(idx)) {
				auto str = CreateStringPiece(idata[idx]);
				auto match = duckdb_re2::RE2::PartialMatchN(str, lstate.constant_pattern, groups.data(), groups.size());
				for (size_t col = 0; col < child_entries.size(); ++col) {
					auto &child_entry = child_entries[col];
					auto cdata = FlatVector::GetData<string_t>(*child_entry);
					auto &extracted = ws[col];
					cdata[i] = string_t(extracted.data(), match ? extracted.size() : 0);
				}
			} else {
				FlatVector::SetNull(result, i, true);
			}
		}
	}
}

static unique_ptr<FunctionData> RegexExtractBind(ClientContext &context, ScalarFunction &bound_function,
                                                 vector<unique_ptr<Expression>> &arguments) {
	D_ASSERT(arguments.size() >= 2);

	duckdb_re2::RE2::Options options;

	string constant_string;
	bool constant_pattern = TryParseConstantPattern(context, *arguments[1], constant_string);

	if (arguments.size() >= 4) {
		ParseRegexOptions(context, *arguments[3], options);
	}

	string group_string = "\\0";
	if (arguments.size() >= 3) {
		if (arguments[2]->HasParameter()) {
			throw ParameterNotResolvedException();
		}
		if (!arguments[2]->IsFoldable()) {
			throw InvalidInputException("Group specification field must be a constant!");
		}
		Value group = ExpressionExecutor::EvaluateScalar(context, *arguments[2]);
		if (group.IsNull()) {
			group_string = "";
		} else if (group.type().id() == LogicalTypeId::LIST) {
			if (!constant_pattern) {
				throw BinderException("%s with LIST requires a constant pattern", bound_function.name);
			}
			auto &list_children = ListValue::GetChildren(group);
			if (list_children.empty()) {
				throw BinderException("%s requires non-empty lists of capture names", bound_function.name);
			}
			case_insensitive_set_t name_collision_set;
			child_list_t<LogicalType> struct_children;
			for (const auto &child : list_children) {
				if (child.IsNull()) {
					throw BinderException("NULL group name in %s", bound_function.name);
				}
				const auto group_name = child.ToString();
				if (name_collision_set.find(group_name) != name_collision_set.end()) {
					throw BinderException("Duplicate group name \"%s\" in %s", group_name, bound_function.name);
				}
				name_collision_set.insert(group_name);
				struct_children.emplace_back(make_pair(group_name, LogicalType::VARCHAR));
			}
			bound_function.return_type = LogicalType::STRUCT(struct_children);

			duckdb_re2::StringPiece constant_piece(constant_string.c_str(), constant_string.size());
			RE2 constant_pattern(constant_piece, options);
			if (size_t(constant_pattern.NumberOfCapturingGroups()) < list_children.size()) {
				throw BinderException("Not enough group names in %s", bound_function.name);
			}
		} else {
			auto group_idx = group.GetValue<int32_t>();
			if (group_idx < 0 || group_idx > 9) {
				throw InvalidInputException("Group index must be between 0 and 9!");
			}
			group_string = "\\" + to_string(group_idx);
		}
	}

	return make_uniq<RegexpExtractBindData>(options, std::move(constant_string), constant_pattern,
	                                        std::move(group_string));
}

void RegexpFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunctionSet regexp_full_match("regexp_full_match");
	regexp_full_match.AddFunction(ScalarFunction(
	    {LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::BOOLEAN, RegexpMatchesFunction<RegexFullMatch>,
	    RegexpMatchesBind, nullptr, nullptr, RegexInitLocalState, LogicalType::INVALID,
	    FunctionSideEffects::NO_SIDE_EFFECTS, FunctionNullHandling::SPECIAL_HANDLING));
	regexp_full_match.AddFunction(ScalarFunction(
	    {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::BOOLEAN,
	    RegexpMatchesFunction<RegexFullMatch>, RegexpMatchesBind, nullptr, nullptr, RegexInitLocalState,
	    LogicalType::INVALID, FunctionSideEffects::NO_SIDE_EFFECTS, FunctionNullHandling::SPECIAL_HANDLING));

	ScalarFunctionSet regexp_partial_match("regexp_matches");
	regexp_partial_match.AddFunction(ScalarFunction(
	    {LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::BOOLEAN, RegexpMatchesFunction<RegexPartialMatch>,
	    RegexpMatchesBind, nullptr, nullptr, RegexInitLocalState, LogicalType::INVALID,
	    FunctionSideEffects::NO_SIDE_EFFECTS, FunctionNullHandling::SPECIAL_HANDLING));
	regexp_partial_match.AddFunction(ScalarFunction(
	    {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::BOOLEAN,
	    RegexpMatchesFunction<RegexPartialMatch>, RegexpMatchesBind, nullptr, nullptr, RegexInitLocalState,
	    LogicalType::INVALID, FunctionSideEffects::NO_SIDE_EFFECTS, FunctionNullHandling::SPECIAL_HANDLING));

	ScalarFunctionSet regexp_replace("regexp_replace");
	regexp_replace.AddFunction(ScalarFunction({LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR},
	                                          LogicalType::VARCHAR, RegexReplaceFunction, RegexReplaceBind, nullptr,
	                                          nullptr, RegexInitLocalState));
	regexp_replace.AddFunction(ScalarFunction(
	    {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::VARCHAR,
	    RegexReplaceFunction, RegexReplaceBind, nullptr, nullptr, RegexInitLocalState));

	ScalarFunctionSet regexp_extract("regexp_extract");
	regexp_extract.AddFunction(
	    ScalarFunction({LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::VARCHAR, RegexExtractFunction,
	                   RegexExtractBind, nullptr, nullptr, RegexInitLocalState, LogicalType::INVALID,
	                   FunctionSideEffects::NO_SIDE_EFFECTS, FunctionNullHandling::SPECIAL_HANDLING));
	regexp_extract.AddFunction(ScalarFunction(
	    {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::INTEGER}, LogicalType::VARCHAR, RegexExtractFunction,
	    RegexExtractBind, nullptr, nullptr, RegexInitLocalState, LogicalType::INVALID,
	    FunctionSideEffects::NO_SIDE_EFFECTS, FunctionNullHandling::SPECIAL_HANDLING));
	regexp_extract.AddFunction(ScalarFunction(
	    {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::INTEGER, LogicalType::VARCHAR}, LogicalType::VARCHAR,
	    RegexExtractFunction, RegexExtractBind, nullptr, nullptr, RegexInitLocalState, LogicalType::INVALID,
	    FunctionSideEffects::NO_SIDE_EFFECTS, FunctionNullHandling::SPECIAL_HANDLING));
	// REGEXP_EXTRACT(<string>, <pattern>, [<group 1 name>[, <group n name>]...])
	regexp_extract.AddFunction(ScalarFunction(
	    {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::LIST(LogicalType::VARCHAR)}, LogicalType::VARCHAR,
	    RegexExtractStructFunction, RegexExtractBind, nullptr, nullptr, RegexInitLocalState, LogicalType::INVALID,
	    FunctionSideEffects::NO_SIDE_EFFECTS, FunctionNullHandling::SPECIAL_HANDLING));
	// REGEXP_EXTRACT(<string>, <pattern>, [<group 1 name>[, <group n name>]...], <options>)
	regexp_extract.AddFunction(ScalarFunction(
	    {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::LIST(LogicalType::VARCHAR), LogicalType::VARCHAR},
	    LogicalType::VARCHAR, RegexExtractStructFunction, RegexExtractBind, nullptr, nullptr, RegexInitLocalState,
	    LogicalType::INVALID, FunctionSideEffects::NO_SIDE_EFFECTS, FunctionNullHandling::SPECIAL_HANDLING));

	ScalarFunctionSet regexp_extract_all("regexp_extract_all");
	regexp_extract_all.AddFunction(ScalarFunction(
	    {LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::LIST(LogicalType::VARCHAR),
	    RegexpExtractAll::Execute, RegexpExtractAll::Bind, nullptr, nullptr, RegexpExtractAll::InitLocalState,
	    LogicalType::INVALID, FunctionSideEffects::NO_SIDE_EFFECTS, FunctionNullHandling::SPECIAL_HANDLING));
	regexp_extract_all.AddFunction(ScalarFunction(
	    {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::INTEGER}, LogicalType::LIST(LogicalType::VARCHAR),
	    RegexpExtractAll::Execute, RegexpExtractAll::Bind, nullptr, nullptr, RegexpExtractAll::InitLocalState,
	    LogicalType::INVALID, FunctionSideEffects::NO_SIDE_EFFECTS, FunctionNullHandling::SPECIAL_HANDLING));
	regexp_extract_all.AddFunction(
	    ScalarFunction({LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::INTEGER, LogicalType::VARCHAR},
	                   LogicalType::LIST(LogicalType::VARCHAR), RegexpExtractAll::Execute, RegexpExtractAll::Bind,
	                   nullptr, nullptr, RegexpExtractAll::InitLocalState, LogicalType::INVALID,
	                   FunctionSideEffects::NO_SIDE_EFFECTS, FunctionNullHandling::SPECIAL_HANDLING));

	set.AddFunction(regexp_full_match);
	set.AddFunction(regexp_partial_match);
	set.AddFunction(regexp_replace);
	set.AddFunction(regexp_extract);
	set.AddFunction(regexp_extract_all);
}

} // namespace duckdb




namespace duckdb {

bool StripAccentsFun::IsAscii(const char *input, idx_t n) {
	for (idx_t i = 0; i < n; i++) {
		if (input[i] & 0x80) {
			// non-ascii character
			return false;
		}
	}
	return true;
}

struct StripAccentsOperator {
	template <class INPUT_TYPE, class RESULT_TYPE>
	static RESULT_TYPE Operation(INPUT_TYPE input, Vector &result) {
		if (StripAccentsFun::IsAscii(input.GetData(), input.GetSize())) {
			return input;
		}

		// non-ascii, perform collation
		auto stripped = utf8proc_remove_accents((const utf8proc_uint8_t *)input.GetData(), input.GetSize());
		auto result_str = StringVector::AddString(result, (const char *)stripped);
		free(stripped);
		return result_str;
	}
};

static void StripAccentsFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(args.ColumnCount() == 1);

	UnaryExecutor::ExecuteString<string_t, string_t, StripAccentsOperator>(args.data[0], result, args.size());
	StringVector::AddHeapReference(result, args.data[0]);
}

ScalarFunction StripAccentsFun::GetFunction() {
	return ScalarFunction("strip_accents", {LogicalType::VARCHAR}, LogicalType::VARCHAR, StripAccentsFunction);
}

void StripAccentsFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(StripAccentsFun::GetFunction());
}

} // namespace duckdb











namespace duckdb {

static const int64_t SUPPORTED_UPPER_BOUND = NumericLimits<uint32_t>::Maximum();
static const int64_t SUPPORTED_LOWER_BOUND = -SUPPORTED_UPPER_BOUND - 1;

static inline void AssertInSupportedRange(idx_t input_size, int64_t offset, int64_t length) {

	if (input_size > (uint64_t)SUPPORTED_UPPER_BOUND) {
		throw OutOfRangeException("Substring input size is too large (> %d)", SUPPORTED_UPPER_BOUND);
	}
	if (offset < SUPPORTED_LOWER_BOUND) {
		throw OutOfRangeException("Substring offset outside of supported range (< %d)", SUPPORTED_LOWER_BOUND);
	}
	if (offset > SUPPORTED_UPPER_BOUND) {
		throw OutOfRangeException("Substring offset outside of supported range (> %d)", SUPPORTED_UPPER_BOUND);
	}
	if (length < SUPPORTED_LOWER_BOUND) {
		throw OutOfRangeException("Substring length outside of supported range (< %d)", SUPPORTED_LOWER_BOUND);
	}
	if (length > SUPPORTED_UPPER_BOUND) {
		throw OutOfRangeException("Substring length outside of supported range (> %d)", SUPPORTED_UPPER_BOUND);
	}
}

string_t SubstringEmptyString(Vector &result) {
	auto result_string = StringVector::EmptyString(result, 0);
	result_string.Finalize();
	return result_string;
}

string_t SubstringSlice(Vector &result, const char *input_data, int64_t offset, int64_t length) {
	auto result_string = StringVector::EmptyString(result, length);
	auto result_data = result_string.GetDataWriteable();
	memcpy(result_data, input_data + offset, length);
	result_string.Finalize();
	return result_string;
}

// compute start and end characters from the given input size and offset/length
bool SubstringStartEnd(int64_t input_size, int64_t offset, int64_t length, int64_t &start, int64_t &end) {
	if (length == 0) {
		return false;
	}
	if (offset > 0) {
		// positive offset: scan from start
		start = MinValue<int64_t>(input_size, offset - 1);
	} else if (offset < 0) {
		// negative offset: scan from end (i.e. start = end + offset)
		start = MaxValue<int64_t>(input_size + offset, 0);
	} else {
		// offset = 0: special case, we start 1 character BEHIND the first character
		start = 0;
		length--;
		if (length <= 0) {
			return false;
		}
	}
	if (length > 0) {
		// positive length: go forward (i.e. end = start + offset)
		end = MinValue<int64_t>(input_size, start + length);
	} else {
		// negative length: go backwards (i.e. end = start, start = start + length)
		end = start;
		start = MaxValue<int64_t>(0, start + length);
	}
	if (start == end) {
		return false;
	}
	D_ASSERT(start < end);
	return true;
}

string_t SubstringASCII(Vector &result, string_t input, int64_t offset, int64_t length) {
	auto input_data = input.GetData();
	auto input_size = input.GetSize();

	AssertInSupportedRange(input_size, offset, length);

	int64_t start, end;
	if (!SubstringStartEnd(input_size, offset, length, start, end)) {
		return SubstringEmptyString(result);
	}
	return SubstringSlice(result, input_data, start, end - start);
}

string_t SubstringFun::SubstringUnicode(Vector &result, string_t input, int64_t offset, int64_t length) {
	auto input_data = input.GetData();
	auto input_size = input.GetSize();

	AssertInSupportedRange(input_size, offset, length);

	if (length == 0) {
		return SubstringEmptyString(result);
	}
	// first figure out which direction we need to scan
	idx_t start_pos;
	idx_t end_pos;
	if (offset < 0) {
		start_pos = 0;
		end_pos = DConstants::INVALID_INDEX;

		// negative offset: scan backwards
		int64_t start, end;

		// we express start and end as unicode codepoints from the back
		offset--;
		if (length < 0) {
			// negative length
			start = -offset - length;
			end = -offset;
		} else {
			// positive length
			start = -offset;
			end = -offset - length;
		}
		if (end <= 0) {
			end_pos = input_size;
		}
		int64_t current_character = 0;
		for (idx_t i = input_size; i > 0; i--) {
			if (LengthFun::IsCharacter(input_data[i - 1])) {
				current_character++;
				if (current_character == start) {
					start_pos = i;
					break;
				} else if (current_character == end) {
					end_pos = i;
				}
			}
		}
		while (!LengthFun::IsCharacter(input_data[start_pos])) {
			start_pos++;
		}
		while (end_pos < input_size && !LengthFun::IsCharacter(input_data[end_pos])) {
			end_pos++;
		}

		if (end_pos == DConstants::INVALID_INDEX) {
			return SubstringEmptyString(result);
		}
	} else {
		start_pos = DConstants::INVALID_INDEX;
		end_pos = input_size;

		// positive offset: scan forwards
		int64_t start, end;

		// we express start and end as unicode codepoints from the front
		offset--;
		if (length < 0) {
			// negative length
			start = MaxValue<int64_t>(0, offset + length);
			end = offset;
		} else {
			// positive length
			start = MaxValue<int64_t>(0, offset);
			end = offset + length;
		}

		int64_t current_character = 0;
		for (idx_t i = 0; i < input_size; i++) {
			if (LengthFun::IsCharacter(input_data[i])) {
				if (current_character == start) {
					start_pos = i;
				} else if (current_character == end) {
					end_pos = i;
					break;
				}
				current_character++;
			}
		}
		if (start_pos == DConstants::INVALID_INDEX || end == 0 || end <= start) {
			return SubstringEmptyString(result);
		}
	}
	D_ASSERT(end_pos >= start_pos);
	// after we have found these, we can slice the substring
	return SubstringSlice(result, input_data, start_pos, end_pos - start_pos);
}

string_t SubstringFun::SubstringGrapheme(Vector &result, string_t input, int64_t offset, int64_t length) {
	auto input_data = input.GetData();
	auto input_size = input.GetSize();

	AssertInSupportedRange(input_size, offset, length);

	// we don't know yet if the substring is ascii, but we assume it is (for now)
	// first get the start and end as if this was an ascii string
	int64_t start, end;
	if (!SubstringStartEnd(input_size, offset, length, start, end)) {
		return SubstringEmptyString(result);
	}

	// now check if all the characters between 0 and end are ascii characters
	// note that we scan one further to check for a potential combining diacritics (e.g. i + diacritic is ï)
	bool is_ascii = true;
	idx_t ascii_end = MinValue<idx_t>(end + 1, input_size);
	for (idx_t i = 0; i < ascii_end; i++) {
		if (input_data[i] & 0x80) {
			// found a non-ascii character: eek
			is_ascii = false;
			break;
		}
	}
	if (is_ascii) {
		// all characters are ascii, we can just slice the substring
		return SubstringSlice(result, input_data, start, end - start);
	}
	// if the characters are not ascii, we need to scan grapheme clusters
	// first figure out which direction we need to scan
	// offset = 0 case is taken care of in SubstringStartEnd
	if (offset < 0) {
		// negative offset, this case is more difficult
		// we first need to count the number of characters in the string
		idx_t num_characters = 0;
		utf8proc_grapheme_callback(input_data, input_size, [&](size_t start, size_t end) {
			num_characters++;
			return true;
		});
		// now call substring start and end again, but with the number of unicode characters this time
		SubstringStartEnd(num_characters, offset, length, start, end);
	}

	// now scan the graphemes of the string to find the positions of the start and end characters
	int64_t current_character = 0;
	idx_t start_pos = DConstants::INVALID_INDEX, end_pos = input_size;
	utf8proc_grapheme_callback(input_data, input_size, [&](size_t gstart, size_t gend) {
		if (current_character == start) {
			start_pos = gstart;
		} else if (current_character == end) {
			end_pos = gstart;
			return false;
		}
		current_character++;
		return true;
	});
	if (start_pos == DConstants::INVALID_INDEX) {
		return SubstringEmptyString(result);
	}
	// after we have found these, we can slice the substring
	return SubstringSlice(result, input_data, start_pos, end_pos - start_pos);
}

struct SubstringUnicodeOp {
	static string_t Substring(Vector &result, string_t input, int64_t offset, int64_t length) {
		return SubstringFun::SubstringUnicode(result, input, offset, length);
	}
};

struct SubstringGraphemeOp {
	static string_t Substring(Vector &result, string_t input, int64_t offset, int64_t length) {
		return SubstringFun::SubstringGrapheme(result, input, offset, length);
	}
};

template <class OP>
static void SubstringFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &input_vector = args.data[0];
	auto &offset_vector = args.data[1];
	if (args.ColumnCount() == 3) {
		auto &length_vector = args.data[2];

		TernaryExecutor::Execute<string_t, int64_t, int64_t, string_t>(
		    input_vector, offset_vector, length_vector, result, args.size(),
		    [&](string_t input_string, int64_t offset, int64_t length) {
			    return OP::Substring(result, input_string, offset, length);
		    });
	} else {
		BinaryExecutor::Execute<string_t, int64_t, string_t>(
		    input_vector, offset_vector, result, args.size(), [&](string_t input_string, int64_t offset) {
			    return OP::Substring(result, input_string, offset, NumericLimits<uint32_t>::Maximum());
		    });
	}
}

static void SubstringFunctionASCII(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &input_vector = args.data[0];
	auto &offset_vector = args.data[1];
	if (args.ColumnCount() == 3) {
		auto &length_vector = args.data[2];

		TernaryExecutor::Execute<string_t, int64_t, int64_t, string_t>(
		    input_vector, offset_vector, length_vector, result, args.size(),
		    [&](string_t input_string, int64_t offset, int64_t length) {
			    return SubstringASCII(result, input_string, offset, length);
		    });
	} else {
		BinaryExecutor::Execute<string_t, int64_t, string_t>(
		    input_vector, offset_vector, result, args.size(), [&](string_t input_string, int64_t offset) {
			    return SubstringASCII(result, input_string, offset, NumericLimits<uint32_t>::Maximum());
		    });
	}
}

static unique_ptr<BaseStatistics> SubstringPropagateStats(ClientContext &context, FunctionStatisticsInput &input) {
	auto &child_stats = input.child_stats;
	auto &expr = input.expr;
	// can only propagate stats if the children have stats
	// we only care about the stats of the first child (i.e. the string)
	if (!StringStats::CanContainUnicode(child_stats[0])) {
		expr.function.function = SubstringFunctionASCII;
	}
	return nullptr;
}

void SubstringFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunctionSet substr("substring");
	substr.AddFunction(ScalarFunction({LogicalType::VARCHAR, LogicalType::BIGINT, LogicalType::BIGINT},
	                                  LogicalType::VARCHAR, SubstringFunction<SubstringUnicodeOp>, nullptr, nullptr,
	                                  SubstringPropagateStats));
	substr.AddFunction(ScalarFunction({LogicalType::VARCHAR, LogicalType::BIGINT}, LogicalType::VARCHAR,
	                                  SubstringFunction<SubstringUnicodeOp>, nullptr, nullptr,
	                                  SubstringPropagateStats));
	set.AddFunction(substr);
	substr.name = "substr";
	set.AddFunction(substr);

	ScalarFunctionSet substr_grapheme("substring_grapheme");
	substr_grapheme.AddFunction(ScalarFunction({LogicalType::VARCHAR, LogicalType::BIGINT, LogicalType::BIGINT},
	                                           LogicalType::VARCHAR, SubstringFunction<SubstringGraphemeOp>, nullptr,
	                                           nullptr, SubstringPropagateStats));
	substr_grapheme.AddFunction(ScalarFunction({LogicalType::VARCHAR, LogicalType::BIGINT}, LogicalType::VARCHAR,
	                                           SubstringFunction<SubstringGraphemeOp>, nullptr, nullptr,
	                                           SubstringPropagateStats));
	set.AddFunction(substr_grapheme);
}

} // namespace duckdb





namespace duckdb {

static bool SuffixFunction(const string_t &str, const string_t &suffix);

struct SuffixOperator {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA left, TB right) {
		return SuffixFunction(left, right);
	}
};

static bool SuffixFunction(const string_t &str, const string_t &suffix) {
	auto suffix_size = suffix.GetSize();
	auto str_size = str.GetSize();
	if (suffix_size > str_size) {
		return false;
	}

	auto suffix_data = suffix.GetData();
	auto str_data = str.GetData();
	int32_t suf_idx = suffix_size - 1;
	idx_t str_idx = str_size - 1;
	for (; suf_idx >= 0; --suf_idx, --str_idx) {
		if (suffix_data[suf_idx] != str_data[str_idx]) {
			return false;
		}
	}
	return true;
}

ScalarFunction SuffixFun::GetFunction() {
	return ScalarFunction("suffix",                                     // name of the function
	                      {LogicalType::VARCHAR, LogicalType::VARCHAR}, // argument list
	                      LogicalType::BOOLEAN,                         // return type
	                      ScalarFunction::BinaryFunction<string_t, string_t, bool, SuffixOperator>);
}

void SuffixFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(GetFunction());
}

} // namespace duckdb


namespace duckdb {

void BuiltinFunctions::RegisterStringFunctions() {
	Register<LowerFun>();
	Register<UpperFun>();
	Register<StripAccentsFun>();
	Register<ConcatFun>();
	Register<ContainsFun>();
	Register<LengthFun>();
	Register<LikeFun>();
	Register<LikeEscapeFun>();
	Register<RegexpFun>();
	Register<SubstringFun>();
	Register<PrefixFun>();
	Register<SuffixFun>();
	Register<NFCNormalizeFun>();
}

} // namespace duckdb







namespace duckdb {

struct StructExtractBindData : public FunctionData {
	StructExtractBindData(string key, idx_t index, LogicalType type)
	    : key(std::move(key)), index(index), type(std::move(type)) {
	}

	string key;
	idx_t index;
	LogicalType type;

public:
	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<StructExtractBindData>(key, index, type);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &other = (const StructExtractBindData &)other_p;
		return key == other.key && index == other.index && type == other.type;
	}
};

static void StructExtractFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &info = func_expr.bind_info->Cast<StructExtractBindData>();

	// this should be guaranteed by the binder
	auto &vec = args.data[0];

	vec.Verify(args.size());
	auto &children = StructVector::GetEntries(vec);
	D_ASSERT(info.index < children.size());
	auto &struct_child = children[info.index];
	result.Reference(*struct_child);
	result.Verify(args.size());
}

static unique_ptr<FunctionData> StructExtractBind(ClientContext &context, ScalarFunction &bound_function,
                                                  vector<unique_ptr<Expression>> &arguments) {
	D_ASSERT(bound_function.arguments.size() == 2);
	if (arguments[0]->return_type.id() == LogicalTypeId::UNKNOWN) {
		throw ParameterNotResolvedException();
	}
	D_ASSERT(LogicalTypeId::STRUCT == arguments[0]->return_type.id());
	auto &struct_children = StructType::GetChildTypes(arguments[0]->return_type);
	if (struct_children.empty()) {
		throw InternalException("Can't extract something from an empty struct");
	}
	bound_function.arguments[0] = arguments[0]->return_type;

	auto &key_child = arguments[1];
	if (key_child->HasParameter()) {
		throw ParameterNotResolvedException();
	}

	if (key_child->return_type.id() != LogicalTypeId::VARCHAR || !key_child->IsFoldable()) {
		throw BinderException("Key name for struct_extract needs to be a constant string");
	}
	Value key_val = ExpressionExecutor::EvaluateScalar(context, *key_child);
	D_ASSERT(key_val.type().id() == LogicalTypeId::VARCHAR);
	auto &key_str = StringValue::Get(key_val);
	if (key_val.IsNull() || key_str.empty()) {
		throw BinderException("Key name for struct_extract needs to be neither NULL nor empty");
	}
	string key = StringUtil::Lower(key_str);

	LogicalType return_type;
	idx_t key_index = 0;
	bool found_key = false;

	for (size_t i = 0; i < struct_children.size(); i++) {
		auto &child = struct_children[i];
		if (StringUtil::Lower(child.first) == key) {
			found_key = true;
			key_index = i;
			return_type = child.second;
			break;
		}
	}

	if (!found_key) {
		vector<string> candidates;
		candidates.reserve(struct_children.size());
		for (auto &struct_child : struct_children) {
			candidates.push_back(struct_child.first);
		}
		auto closest_settings = StringUtil::TopNLevenshtein(candidates, key);
		auto message = StringUtil::CandidatesMessage(closest_settings, "Candidate Entries");
		throw BinderException("Could not find key \"%s\" in struct\n%s", key, message);
	}

	bound_function.return_type = return_type;
	return make_uniq<StructExtractBindData>(std::move(key), key_index, std::move(return_type));
}

static unique_ptr<BaseStatistics> PropagateStructExtractStats(ClientContext &context, FunctionStatisticsInput &input) {
	auto &child_stats = input.child_stats;
	auto &bind_data = input.bind_data;

	auto &info = bind_data->Cast<StructExtractBindData>();
	auto struct_child_stats = StructStats::GetChildStats(child_stats[0]);
	return struct_child_stats[info.index].ToUnique();
}

ScalarFunction StructExtractFun::GetFunction() {
	return ScalarFunction("struct_extract", {LogicalTypeId::STRUCT, LogicalType::VARCHAR}, LogicalType::ANY,
	                      StructExtractFunction, StructExtractBind, nullptr, PropagateStructExtractStats);
}

void StructExtractFun::RegisterFunction(BuiltinFunctions &set) {
	// the arguments and return types are actually set in the binder function
	auto fun = GetFunction();
	set.AddFunction(fun);
}

} // namespace duckdb









namespace duckdb {

// aggregate state export
struct ExportAggregateBindData : public FunctionData {
	AggregateFunction aggr;
	idx_t state_size;

	explicit ExportAggregateBindData(AggregateFunction aggr_p, idx_t state_size_p)
	    : aggr(std::move(aggr_p)), state_size(state_size_p) {
	}

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<ExportAggregateBindData>(aggr, state_size);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = (const ExportAggregateBindData &)other_p;
		return aggr == other.aggr && state_size == other.state_size;
	}

	static ExportAggregateBindData &GetFrom(ExpressionState &state) {
		auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
		return func_expr.bind_info->Cast<ExportAggregateBindData>();
	}
};

struct CombineState : public FunctionLocalState {
	idx_t state_size;

	unsafe_unique_array<data_t> state_buffer0, state_buffer1;
	Vector state_vector0, state_vector1;

	explicit CombineState(idx_t state_size_p)
	    : state_size(state_size_p), state_buffer0(make_unsafe_uniq_array<data_t>(state_size_p)),
	      state_buffer1(make_unsafe_uniq_array<data_t>(state_size_p)),
	      state_vector0(Value::POINTER((uintptr_t)state_buffer0.get())),
	      state_vector1(Value::POINTER((uintptr_t)state_buffer1.get())) {
	}
};

static unique_ptr<FunctionLocalState> InitCombineState(ExpressionState &state, const BoundFunctionExpression &expr,
                                                       FunctionData *bind_data_p) {
	auto &bind_data = *(ExportAggregateBindData *)bind_data_p;
	return make_uniq<CombineState>(bind_data.state_size);
}

struct FinalizeState : public FunctionLocalState {
	idx_t state_size;
	unsafe_unique_array<data_t> state_buffer;
	Vector addresses;

	explicit FinalizeState(idx_t state_size_p)
	    : state_size(state_size_p),
	      state_buffer(make_unsafe_uniq_array<data_t>(STANDARD_VECTOR_SIZE * AlignValue(state_size_p))),
	      addresses(LogicalType::POINTER) {
	}
};

static unique_ptr<FunctionLocalState> InitFinalizeState(ExpressionState &state, const BoundFunctionExpression &expr,
                                                        FunctionData *bind_data_p) {
	auto &bind_data = *(ExportAggregateBindData *)bind_data_p;
	return make_uniq<FinalizeState>(bind_data.state_size);
}

static void AggregateStateFinalize(DataChunk &input, ExpressionState &state_p, Vector &result) {
	auto &bind_data = ExportAggregateBindData::GetFrom(state_p);
	auto &local_state = (FinalizeState &)*((ExecuteFunctionState &)state_p).local_state;

	D_ASSERT(bind_data.state_size == bind_data.aggr.state_size());
	D_ASSERT(input.data.size() == 1);
	D_ASSERT(input.data[0].GetType().id() == LogicalTypeId::AGGREGATE_STATE);
	auto aligned_state_size = AlignValue(bind_data.state_size);

	auto state_vec_ptr = FlatVector::GetData<data_ptr_t>(local_state.addresses);

	UnifiedVectorFormat state_data;
	input.data[0].ToUnifiedFormat(input.size(), state_data);
	for (idx_t i = 0; i < input.size(); i++) {
		auto state_idx = state_data.sel->get_index(i);
		auto state_entry = &((string_t *)state_data.data)[state_idx];
		auto target_ptr = (const char *)local_state.state_buffer.get() + aligned_state_size * i;

		if (state_data.validity.RowIsValid(state_idx)) {
			D_ASSERT(state_entry->GetSize() == bind_data.state_size);
			memcpy((void *)target_ptr, state_entry->GetData(), bind_data.state_size);
		} else {
			// create a dummy state because finalize does not understand NULLs in its input
			// we put the NULL back in explicitly below
			bind_data.aggr.initialize((data_ptr_t)target_ptr);
		}
		state_vec_ptr[i] = (data_ptr_t)target_ptr;
	}

	AggregateInputData aggr_input_data(nullptr, Allocator::DefaultAllocator());
	bind_data.aggr.finalize(local_state.addresses, aggr_input_data, result, input.size(), 0);

	for (idx_t i = 0; i < input.size(); i++) {
		auto state_idx = state_data.sel->get_index(i);
		if (!state_data.validity.RowIsValid(state_idx)) {
			FlatVector::SetNull(result, i, true);
		}
	}
}

static void AggregateStateCombine(DataChunk &input, ExpressionState &state_p, Vector &result) {
	auto &bind_data = ExportAggregateBindData::GetFrom(state_p);
	auto &local_state = (CombineState &)*((ExecuteFunctionState &)state_p).local_state;

	D_ASSERT(bind_data.state_size == bind_data.aggr.state_size());

	D_ASSERT(input.data.size() == 2);
	D_ASSERT(input.data[0].GetType().id() == LogicalTypeId::AGGREGATE_STATE);
	D_ASSERT(input.data[0].GetType() == result.GetType());

	if (input.data[0].GetType().InternalType() != input.data[1].GetType().InternalType()) {
		throw IOException("Aggregate state combine type mismatch, expect %s, got %s",
		                  input.data[0].GetType().ToString(), input.data[1].GetType().ToString());
	}

	UnifiedVectorFormat state0_data, state1_data;
	input.data[0].ToUnifiedFormat(input.size(), state0_data);
	input.data[1].ToUnifiedFormat(input.size(), state1_data);

	auto result_ptr = FlatVector::GetData<string_t>(result);

	for (idx_t i = 0; i < input.size(); i++) {
		auto state0_idx = state0_data.sel->get_index(i);
		auto state1_idx = state1_data.sel->get_index(i);

		auto &state0 = ((string_t *)state0_data.data)[state0_idx];
		auto &state1 = ((string_t *)state1_data.data)[state1_idx];

		// if both are NULL, we return NULL. If either of them is not, the result is that one
		if (!state0_data.validity.RowIsValid(state0_idx) && !state1_data.validity.RowIsValid(state1_idx)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}
		if (state0_data.validity.RowIsValid(state0_idx) && !state1_data.validity.RowIsValid(state1_idx)) {
			result_ptr[i] = StringVector::AddStringOrBlob(result, (const char *)state0.GetData(), bind_data.state_size);
			continue;
		}
		if (!state0_data.validity.RowIsValid(state0_idx) && state1_data.validity.RowIsValid(state1_idx)) {
			result_ptr[i] = StringVector::AddStringOrBlob(result, (const char *)state1.GetData(), bind_data.state_size);
			continue;
		}

		// we actually have to combine
		if (state0.GetSize() != bind_data.state_size || state1.GetSize() != bind_data.state_size) {
			throw IOException("Aggregate state size mismatch, expect %llu, got %llu and %llu", bind_data.state_size,
			                  state0.GetSize(), state1.GetSize());
		}

		memcpy(local_state.state_buffer0.get(), state0.GetData(), bind_data.state_size);
		memcpy(local_state.state_buffer1.get(), state1.GetData(), bind_data.state_size);

		AggregateInputData aggr_input_data(nullptr, Allocator::DefaultAllocator());
		bind_data.aggr.combine(local_state.state_vector0, local_state.state_vector1, aggr_input_data, 1);

		result_ptr[i] =
		    StringVector::AddStringOrBlob(result, (const char *)local_state.state_buffer1.get(), bind_data.state_size);
	}
}

static unique_ptr<FunctionData> BindAggregateState(ClientContext &context, ScalarFunction &bound_function,
                                                   vector<unique_ptr<Expression>> &arguments) {

	// grab the aggregate type and bind the aggregate again

	// the aggregate name and types are in the logical type of the aggregate state, make sure its sane
	auto &arg_return_type = arguments[0]->return_type;
	for (auto &arg_type : bound_function.arguments) {
		arg_type = arg_return_type;
	}

	if (arg_return_type.id() != LogicalTypeId::AGGREGATE_STATE) {
		throw BinderException("Can only FINALIZE aggregate state, not %s", arg_return_type.ToString());
	}
	// combine
	if (arguments.size() == 2 && arguments[0]->return_type != arguments[1]->return_type &&
	    arguments[1]->return_type.id() != LogicalTypeId::BLOB) {
		throw BinderException("Cannot COMBINE aggregate states from different functions, %s <> %s",
		                      arguments[0]->return_type.ToString(), arguments[1]->return_type.ToString());
	}

	// following error states are only reachable when someone messes up creating the state_type
	// which is impossible from SQL

	auto state_type = AggregateStateType::GetStateType(arg_return_type);

	// now we can look up the function in the catalog again and bind it
	auto &func = Catalog::GetSystemCatalog(context).GetEntry(context, CatalogType::SCALAR_FUNCTION_ENTRY,
	                                                         DEFAULT_SCHEMA, state_type.function_name);
	if (func.type != CatalogType::AGGREGATE_FUNCTION_ENTRY) {
		throw InternalException("Could not find aggregate %s", state_type.function_name);
	}
	auto &aggr = func.Cast<AggregateFunctionCatalogEntry>();

	string error;

	FunctionBinder function_binder(context);
	idx_t best_function =
	    function_binder.BindFunction(aggr.name, aggr.functions, state_type.bound_argument_types, error);
	if (best_function == DConstants::INVALID_INDEX) {
		throw InternalException("Could not re-bind exported aggregate %s: %s", state_type.function_name, error);
	}
	auto bound_aggr = aggr.functions.GetFunctionByOffset(best_function);
	if (bound_aggr.bind) {
		// FIXME: this is really hacky
		// but the aggregate state export needs a rework around how it handles more complex aggregates anyway
		vector<unique_ptr<Expression>> args;
		args.reserve(state_type.bound_argument_types.size());
		for (auto &arg_type : state_type.bound_argument_types) {
			args.push_back(make_uniq<BoundConstantExpression>(Value(arg_type)));
		}
		auto bind_info = bound_aggr.bind(context, bound_aggr, args);
		if (bind_info) {
			throw BinderException("Aggregate function with bind info not supported yet in aggregate state export");
		}
	}

	if (bound_aggr.return_type != state_type.return_type || bound_aggr.arguments != state_type.bound_argument_types) {
		throw InternalException("Type mismatch for exported aggregate %s", state_type.function_name);
	}

	if (bound_function.name == "finalize") {
		bound_function.return_type = bound_aggr.return_type;
	} else {
		D_ASSERT(bound_function.name == "combine");
		bound_function.return_type = arg_return_type;
	}

	return make_uniq<ExportAggregateBindData>(bound_aggr, bound_aggr.state_size());
}

static void ExportAggregateFinalize(Vector &state, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                                    idx_t offset) {
	D_ASSERT(offset == 0);
	auto bind_data = (ExportAggregateFunctionBindData *)aggr_input_data.bind_data;
	auto state_size = bind_data->aggregate->function.state_size();
	auto blob_ptr = FlatVector::GetData<string_t>(result);
	auto addresses_ptr = FlatVector::GetData<data_ptr_t>(state);
	for (idx_t row_idx = 0; row_idx < count; row_idx++) {
		auto data_ptr = addresses_ptr[row_idx];
		blob_ptr[row_idx] = StringVector::AddStringOrBlob(result, (const char *)data_ptr, state_size);
	}
}

ExportAggregateFunctionBindData::ExportAggregateFunctionBindData(unique_ptr<Expression> aggregate_p) {
	D_ASSERT(aggregate_p->type == ExpressionType::BOUND_AGGREGATE);
	aggregate = unique_ptr<BoundAggregateExpression>((BoundAggregateExpression *)aggregate_p.release());
}

unique_ptr<FunctionData> ExportAggregateFunctionBindData::Copy() const {
	return make_uniq<ExportAggregateFunctionBindData>(aggregate->Copy());
}

bool ExportAggregateFunctionBindData::Equals(const FunctionData &other_p) const {
	auto &other = (const ExportAggregateFunctionBindData &)other_p;
	return aggregate->Equals(other.aggregate.get());
}

static void ExportStateAggregateSerialize(FieldWriter &writer, const FunctionData *bind_data_p,
                                          const AggregateFunction &function) {
	throw NotImplementedException("FIXME: export state serialize");
}
static unique_ptr<FunctionData> ExportStateAggregateDeserialize(ClientContext &context, FieldReader &reader,
                                                                AggregateFunction &bound_function) {
	throw NotImplementedException("FIXME: export state deserialize");
}

static void ExportStateScalarSerialize(FieldWriter &writer, const FunctionData *bind_data_p,
                                       const ScalarFunction &function) {
	throw NotImplementedException("FIXME: export state serialize");
}
static unique_ptr<FunctionData> ExportStateScalarDeserialize(ClientContext &context, FieldReader &reader,
                                                             ScalarFunction &bound_function) {
	throw NotImplementedException("FIXME: export state deserialize");
}

unique_ptr<BoundAggregateExpression>
ExportAggregateFunction::Bind(unique_ptr<BoundAggregateExpression> child_aggregate) {
	auto &bound_function = child_aggregate->function;
	if (!bound_function.combine) {
		throw BinderException("Cannot use EXPORT_STATE for non-combinable function %s", bound_function.name);
	}
	if (bound_function.bind) {
		throw BinderException("Cannot use EXPORT_STATE on aggregate functions with custom binders");
	}
	if (bound_function.destructor) {
		throw BinderException("Cannot use EXPORT_STATE on aggregate functions with custom destructors");
	}
	// this should be required
	D_ASSERT(bound_function.state_size);
	D_ASSERT(bound_function.finalize);

	D_ASSERT(child_aggregate->function.return_type.id() != LogicalTypeId::INVALID);
#ifdef DEBUG
	for (auto &arg_type : child_aggregate->function.arguments) {
		D_ASSERT(arg_type.id() != LogicalTypeId::INVALID);
	}
#endif
	auto export_bind_data = make_uniq<ExportAggregateFunctionBindData>(child_aggregate->Copy());
	aggregate_state_t state_type(child_aggregate->function.name, child_aggregate->function.return_type,
	                             child_aggregate->function.arguments);
	auto return_type = LogicalType::AGGREGATE_STATE(std::move(state_type));

	auto export_function =
	    AggregateFunction("aggregate_state_export_" + bound_function.name, bound_function.arguments, return_type,
	                      bound_function.state_size, bound_function.initialize, bound_function.update,
	                      bound_function.combine, ExportAggregateFinalize, bound_function.simple_update,
	                      /* can't bind this again */ nullptr, /* no dynamic state yet */ nullptr,
	                      /* can't propagate statistics */ nullptr, nullptr);
	export_function.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	export_function.serialize = ExportStateAggregateSerialize;
	export_function.deserialize = ExportStateAggregateDeserialize;

	return make_uniq<BoundAggregateExpression>(export_function, std::move(child_aggregate->children),
	                                           std::move(child_aggregate->filter), std::move(export_bind_data),
	                                           child_aggregate->aggr_type);
}

ScalarFunction ExportAggregateFunction::GetFinalize() {
	auto result = ScalarFunction("finalize", {LogicalTypeId::AGGREGATE_STATE}, LogicalTypeId::INVALID,
	                             AggregateStateFinalize, BindAggregateState, nullptr, nullptr, InitFinalizeState);
	result.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	result.serialize = ExportStateScalarSerialize;
	result.deserialize = ExportStateScalarDeserialize;
	return result;
}

ScalarFunction ExportAggregateFunction::GetCombine() {
	auto result =
	    ScalarFunction("combine", {LogicalTypeId::AGGREGATE_STATE, LogicalTypeId::ANY}, LogicalTypeId::AGGREGATE_STATE,
	                   AggregateStateCombine, BindAggregateState, nullptr, nullptr, InitCombineState);
	result.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	result.serialize = ExportStateScalarSerialize;
	result.deserialize = ExportStateScalarDeserialize;
	return result;
}

void ExportAggregateFunction::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(ExportAggregateFunction::GetCombine());
	set.AddFunction(ExportAggregateFunction::GetFinalize());
}

} // namespace duckdb


namespace duckdb {

FunctionLocalState::~FunctionLocalState() {
}

ScalarFunction::ScalarFunction(string name, vector<LogicalType> arguments, LogicalType return_type,
                               scalar_function_t function, bind_scalar_function_t bind,
                               dependency_function_t dependency, function_statistics_t statistics,
                               init_local_state_t init_local_state, LogicalType varargs,
                               FunctionSideEffects side_effects, FunctionNullHandling null_handling)
    : BaseScalarFunction(std::move(name), std::move(arguments), std::move(return_type), side_effects,
                         std::move(varargs), null_handling),
      function(std::move(function)), bind(bind), init_local_state(init_local_state), dependency(dependency),
      statistics(statistics), serialize(nullptr), deserialize(nullptr) {
}

ScalarFunction::ScalarFunction(vector<LogicalType> arguments, LogicalType return_type, scalar_function_t function,
                               bind_scalar_function_t bind, dependency_function_t dependency,
                               function_statistics_t statistics, init_local_state_t init_local_state,
                               LogicalType varargs, FunctionSideEffects side_effects,
                               FunctionNullHandling null_handling)
    : ScalarFunction(string(), std::move(arguments), std::move(return_type), std::move(function), bind, dependency,
                     statistics, init_local_state, std::move(varargs), side_effects, null_handling) {
}

bool ScalarFunction::operator==(const ScalarFunction &rhs) const {
	return CompareScalarFunctionT(rhs.function) && bind == rhs.bind && dependency == rhs.dependency &&
	       statistics == rhs.statistics;
}
bool ScalarFunction::operator!=(const ScalarFunction &rhs) const {
	return !(*this == rhs);
}

bool ScalarFunction::Equal(const ScalarFunction &rhs) const {
	// number of types
	if (this->arguments.size() != rhs.arguments.size()) {
		return false;
	}
	// argument types
	for (idx_t i = 0; i < this->arguments.size(); ++i) {
		if (this->arguments[i] != rhs.arguments[i]) {
			return false;
		}
	}
	// return type
	if (this->return_type != rhs.return_type) {
		return false;
	}
	// varargs
	if (this->varargs != rhs.varargs) {
		return false;
	}

	return true; // they are equal
}

bool ScalarFunction::CompareScalarFunctionT(const scalar_function_t &other) const {
	typedef void(scalar_function_ptr_t)(DataChunk &, ExpressionState &, Vector &);

	auto func_ptr = (scalar_function_ptr_t **)function.template target<scalar_function_ptr_t *>();
	auto other_ptr = (scalar_function_ptr_t **)other.template target<scalar_function_ptr_t *>();

	// Case the functions were created from lambdas the target will return a nullptr
	if (!func_ptr && !other_ptr) {
		return true;
	}
	if (func_ptr == nullptr || other_ptr == nullptr) {
		// scalar_function_t (std::functions) from lambdas cannot be compared
		return false;
	}
	return ((size_t)*func_ptr == (size_t)*other_ptr);
}

void ScalarFunction::NopFunction(DataChunk &input, ExpressionState &state, Vector &result) {
	D_ASSERT(input.ColumnCount() >= 1);
	result.Reference(input.data[0]);
}

} // namespace duckdb
//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/function/scalar_macro_function.hpp
//
//
//===----------------------------------------------------------------------===//







namespace duckdb {

ScalarMacroFunction::ScalarMacroFunction(unique_ptr<ParsedExpression> expression)
    : MacroFunction(MacroType::SCALAR_MACRO), expression(std::move(expression)) {
}

ScalarMacroFunction::ScalarMacroFunction(void) : MacroFunction(MacroType::SCALAR_MACRO) {
}

unique_ptr<MacroFunction> ScalarMacroFunction::Copy() const {
	auto result = make_uniq<ScalarMacroFunction>();
	result->expression = expression->Copy();
	CopyProperties(*result);

	return std::move(result);
}

void RemoveQualificationRecursive(unique_ptr<ParsedExpression> &expr) {
	if (expr->GetExpressionType() == ExpressionType::COLUMN_REF) {
		auto &col_ref = expr->Cast<ColumnRefExpression>();
		auto &col_names = col_ref.column_names;
		if (col_names.size() == 2 && col_names[0].find(DummyBinding::DUMMY_NAME) != string::npos) {
			col_names.erase(col_names.begin());
		}
	} else {
		ParsedExpressionIterator::EnumerateChildren(
		    *expr, [](unique_ptr<ParsedExpression> &child) { RemoveQualificationRecursive(child); });
	}
}

string ScalarMacroFunction::ToSQL(const string &schema, const string &name) const {
	// In case of nested macro's we need to fix it a bit
	auto expression_copy = expression->Copy();
	RemoveQualificationRecursive(expression_copy);
	return MacroFunction::ToSQL(schema, name) + StringUtil::Format("(%s);", expression_copy->ToString());
}

void ScalarMacroFunction::SerializeInternal(FieldWriter &writer) const {
	writer.WriteSerializable(*expression);
}

unique_ptr<MacroFunction> ScalarMacroFunction::Deserialize(FieldReader &reader) {
	auto result = make_uniq<ScalarMacroFunction>();
	result->expression = reader.ReadRequiredSerializable<ParsedExpression>();
	return std::move(result);
}

} // namespace duckdb













namespace duckdb {

LogicalType ArrowTableFunction::GetArrowLogicalType(
    ArrowSchema &schema, std::unordered_map<idx_t, unique_ptr<ArrowConvertData>> &arrow_convert_data, idx_t col_idx) {
	auto format = string(schema.format);
	if (arrow_convert_data.find(col_idx) == arrow_convert_data.end()) {
		arrow_convert_data[col_idx] = make_uniq<ArrowConvertData>();
	}
	auto &convert_data = *arrow_convert_data[col_idx];
	if (format == "n") {
		return LogicalType::SQLNULL;
	} else if (format == "b") {
		return LogicalType::BOOLEAN;
	} else if (format == "c") {
		return LogicalType::TINYINT;
	} else if (format == "s") {
		return LogicalType::SMALLINT;
	} else if (format == "i") {
		return LogicalType::INTEGER;
	} else if (format == "l") {
		return LogicalType::BIGINT;
	} else if (format == "C") {
		return LogicalType::UTINYINT;
	} else if (format == "S") {
		return LogicalType::USMALLINT;
	} else if (format == "I") {
		return LogicalType::UINTEGER;
	} else if (format == "L") {
		return LogicalType::UBIGINT;
	} else if (format == "f") {
		return LogicalType::FLOAT;
	} else if (format == "g") {
		return LogicalType::DOUBLE;
	} else if (format[0] == 'd') { //! this can be either decimal128 or decimal 256 (e.g., d:38,0)
		std::string parameters = format.substr(format.find(':'));
		uint8_t width = std::stoi(parameters.substr(1, parameters.find(',')));
		uint8_t scale = std::stoi(parameters.substr(parameters.find(',') + 1));
		if (width > 38) {
			throw NotImplementedException("Unsupported Internal Arrow Type for Decimal %s", format);
		}
		return LogicalType::DECIMAL(width, scale);
	} else if (format == "u") {
		convert_data.variable_sz_type.emplace_back(ArrowVariableSizeType::NORMAL, 0);
		return LogicalType::VARCHAR;
	} else if (format == "U") {
		convert_data.variable_sz_type.emplace_back(ArrowVariableSizeType::SUPER_SIZE, 0);
		return LogicalType::VARCHAR;
	} else if (format == "tsn:") {
		return LogicalTypeId::TIMESTAMP_NS;
	} else if (format == "tsu:") {
		return LogicalTypeId::TIMESTAMP;
	} else if (format == "tsm:") {
		return LogicalTypeId::TIMESTAMP_MS;
	} else if (format == "tss:") {
		return LogicalTypeId::TIMESTAMP_SEC;
	} else if (format == "tdD") {
		convert_data.date_time_precision.emplace_back(ArrowDateTimeType::DAYS);
		return LogicalType::DATE;
	} else if (format == "tdm") {
		convert_data.date_time_precision.emplace_back(ArrowDateTimeType::MILLISECONDS);
		return LogicalType::DATE;
	} else if (format == "tts") {
		convert_data.date_time_precision.emplace_back(ArrowDateTimeType::SECONDS);
		return LogicalType::TIME;
	} else if (format == "ttm") {
		convert_data.date_time_precision.emplace_back(ArrowDateTimeType::MILLISECONDS);
		return LogicalType::TIME;
	} else if (format == "ttu") {
		convert_data.date_time_precision.emplace_back(ArrowDateTimeType::MICROSECONDS);
		return LogicalType::TIME;
	} else if (format == "ttn") {
		convert_data.date_time_precision.emplace_back(ArrowDateTimeType::NANOSECONDS);
		return LogicalType::TIME;
	} else if (format == "tDs") {
		convert_data.date_time_precision.emplace_back(ArrowDateTimeType::SECONDS);
		return LogicalType::INTERVAL;
	} else if (format == "tDm") {
		convert_data.date_time_precision.emplace_back(ArrowDateTimeType::MILLISECONDS);
		return LogicalType::INTERVAL;
	} else if (format == "tDu") {
		convert_data.date_time_precision.emplace_back(ArrowDateTimeType::MICROSECONDS);
		return LogicalType::INTERVAL;
	} else if (format == "tDn") {
		convert_data.date_time_precision.emplace_back(ArrowDateTimeType::NANOSECONDS);
		return LogicalType::INTERVAL;
	} else if (format == "tiD") {
		convert_data.date_time_precision.emplace_back(ArrowDateTimeType::DAYS);
		return LogicalType::INTERVAL;
	} else if (format == "tiM") {
		convert_data.date_time_precision.emplace_back(ArrowDateTimeType::MONTHS);
		return LogicalType::INTERVAL;
	} else if (format == "tin") {
		convert_data.date_time_precision.emplace_back(ArrowDateTimeType::MONTH_DAY_NANO);
		return LogicalType::INTERVAL;
	} else if (format == "+l") {
		convert_data.variable_sz_type.emplace_back(ArrowVariableSizeType::NORMAL, 0);
		auto child_type = GetArrowLogicalType(*schema.children[0], arrow_convert_data, col_idx);
		return LogicalType::LIST(child_type);
	} else if (format == "+L") {
		convert_data.variable_sz_type.emplace_back(ArrowVariableSizeType::SUPER_SIZE, 0);
		auto child_type = GetArrowLogicalType(*schema.children[0], arrow_convert_data, col_idx);
		return LogicalType::LIST(child_type);
	} else if (format[0] == '+' && format[1] == 'w') {
		std::string parameters = format.substr(format.find(':') + 1);
		idx_t fixed_size = std::stoi(parameters);
		convert_data.variable_sz_type.emplace_back(ArrowVariableSizeType::FIXED_SIZE, fixed_size);
		auto child_type = GetArrowLogicalType(*schema.children[0], arrow_convert_data, col_idx);
		return LogicalType::LIST(child_type);
	} else if (format == "+s") {
		child_list_t<LogicalType> child_types;
		for (idx_t type_idx = 0; type_idx < (idx_t)schema.n_children; type_idx++) {
			auto child_type = GetArrowLogicalType(*schema.children[type_idx], arrow_convert_data, col_idx);
			child_types.push_back({schema.children[type_idx]->name, child_type});
		}
		return LogicalType::STRUCT(child_types);

	} else if (format == "+m") {
		convert_data.variable_sz_type.emplace_back(ArrowVariableSizeType::NORMAL, 0);

		auto &arrow_struct_type = *schema.children[0];
		D_ASSERT(arrow_struct_type.n_children == 2);
		auto key_type = GetArrowLogicalType(*arrow_struct_type.children[0], arrow_convert_data, col_idx);
		auto value_type = GetArrowLogicalType(*arrow_struct_type.children[1], arrow_convert_data, col_idx);
		return LogicalType::MAP(key_type, value_type);
	} else if (format == "z") {
		convert_data.variable_sz_type.emplace_back(ArrowVariableSizeType::NORMAL, 0);
		return LogicalType::BLOB;
	} else if (format == "Z") {
		convert_data.variable_sz_type.emplace_back(ArrowVariableSizeType::SUPER_SIZE, 0);
		return LogicalType::BLOB;
	} else if (format[0] == 'w') {
		std::string parameters = format.substr(format.find(':') + 1);
		idx_t fixed_size = std::stoi(parameters);
		convert_data.variable_sz_type.emplace_back(ArrowVariableSizeType::FIXED_SIZE, fixed_size);
		return LogicalType::BLOB;
	} else if (format[0] == 't' && format[1] == 's') {
		// Timestamp with Timezone
		if (format[2] == 'n') {
			convert_data.date_time_precision.emplace_back(ArrowDateTimeType::NANOSECONDS);
		} else if (format[2] == 'u') {
			convert_data.date_time_precision.emplace_back(ArrowDateTimeType::MICROSECONDS);
		} else if (format[2] == 'm') {
			convert_data.date_time_precision.emplace_back(ArrowDateTimeType::MILLISECONDS);
		} else if (format[2] == 's') {
			convert_data.date_time_precision.emplace_back(ArrowDateTimeType::SECONDS);
		} else {
			throw NotImplementedException(" Timestamptz precision of not accepted");
		}
		// TODO right now we just get the UTC value. We probably want to support this properly in the future
		return LogicalType::TIMESTAMP_TZ;
	} else {
		throw NotImplementedException("Unsupported Internal Arrow Type %s", format);
	}
}

void ArrowTableFunction::RenameArrowColumns(vector<string> &names) {
	unordered_map<string, idx_t> name_map;
	for (auto &column_name : names) {
		// put it all lower_case
		auto low_column_name = StringUtil::Lower(column_name);
		if (name_map.find(low_column_name) == name_map.end()) {
			// Name does not exist yet
			name_map[low_column_name]++;
		} else {
			// Name already exists, we add _x where x is the repetition number
			string new_column_name = column_name + "_" + std::to_string(name_map[low_column_name]);
			auto new_column_name_low = StringUtil::Lower(new_column_name);
			while (name_map.find(new_column_name_low) != name_map.end()) {
				// This name is already here due to a previous definition
				name_map[low_column_name]++;
				new_column_name = column_name + "_" + std::to_string(name_map[low_column_name]);
				new_column_name_low = StringUtil::Lower(new_column_name);
			}
			column_name = new_column_name;
			name_map[new_column_name_low]++;
		}
	}
}

unique_ptr<FunctionData> ArrowTableFunction::ArrowScanBind(ClientContext &context, TableFunctionBindInput &input,
                                                           vector<LogicalType> &return_types, vector<string> &names) {
	auto stream_factory_ptr = input.inputs[0].GetPointer();
	auto stream_factory_produce = (stream_factory_produce_t)input.inputs[1].GetPointer();
	auto stream_factory_get_schema = (stream_factory_get_schema_t)input.inputs[2].GetPointer();

	auto res = make_uniq<ArrowScanFunctionData>(stream_factory_produce, stream_factory_ptr);

	auto &data = *res;
	stream_factory_get_schema(stream_factory_ptr, data.schema_root);
	for (idx_t col_idx = 0; col_idx < (idx_t)data.schema_root.arrow_schema.n_children; col_idx++) {
		auto &schema = *data.schema_root.arrow_schema.children[col_idx];
		if (!schema.release) {
			throw InvalidInputException("arrow_scan: released schema passed");
		}
		if (schema.dictionary) {
			res->arrow_convert_data[col_idx] =
			    make_uniq<ArrowConvertData>(GetArrowLogicalType(schema, res->arrow_convert_data, col_idx));
			return_types.emplace_back(GetArrowLogicalType(*schema.dictionary, res->arrow_convert_data, col_idx));
		} else {
			return_types.emplace_back(GetArrowLogicalType(schema, res->arrow_convert_data, col_idx));
		}
		auto format = string(schema.format);
		auto name = string(schema.name);
		if (name.empty()) {
			name = string("v") + to_string(col_idx);
		}
		names.push_back(name);
	}
	RenameArrowColumns(names);
	res->all_types = return_types;
	return std::move(res);
}

unique_ptr<ArrowArrayStreamWrapper> ProduceArrowScan(const ArrowScanFunctionData &function,
                                                     const vector<column_t> &column_ids, TableFilterSet *filters) {
	//! Generate Projection Pushdown Vector
	ArrowStreamParameters parameters;
	D_ASSERT(!column_ids.empty());
	for (idx_t idx = 0; idx < column_ids.size(); idx++) {
		auto col_idx = column_ids[idx];
		if (col_idx != COLUMN_IDENTIFIER_ROW_ID) {
			auto &schema = *function.schema_root.arrow_schema.children[col_idx];
			parameters.projected_columns.projection_map[idx] = schema.name;
			parameters.projected_columns.columns.emplace_back(schema.name);
		}
	}
	parameters.filters = filters;
	return function.scanner_producer(function.stream_factory_ptr, parameters);
}

idx_t ArrowTableFunction::ArrowScanMaxThreads(ClientContext &context, const FunctionData *bind_data_p) {
	return context.db->NumberOfThreads();
}

bool ArrowTableFunction::ArrowScanParallelStateNext(ClientContext &context, const FunctionData *bind_data_p,
                                                    ArrowScanLocalState &state, ArrowScanGlobalState &parallel_state) {
	lock_guard<mutex> parallel_lock(parallel_state.main_mutex);
	if (parallel_state.done) {
		return false;
	}
	state.chunk_offset = 0;
	state.batch_index = ++parallel_state.batch_index;

	auto current_chunk = parallel_state.stream->GetNextChunk();
	while (current_chunk->arrow_array.length == 0 && current_chunk->arrow_array.release) {
		current_chunk = parallel_state.stream->GetNextChunk();
	}
	state.chunk = std::move(current_chunk);
	//! have we run out of chunks? we are done
	if (!state.chunk->arrow_array.release) {
		parallel_state.done = true;
		return false;
	}
	return true;
}

unique_ptr<GlobalTableFunctionState> ArrowTableFunction::ArrowScanInitGlobal(ClientContext &context,
                                                                             TableFunctionInitInput &input) {
	auto &bind_data = (const ArrowScanFunctionData &)*input.bind_data;
	auto result = make_uniq<ArrowScanGlobalState>();
	result->stream = ProduceArrowScan(bind_data, input.column_ids, input.filters.get());
	result->max_threads = ArrowScanMaxThreads(context, input.bind_data.get());
	if (input.CanRemoveFilterColumns()) {
		result->projection_ids = input.projection_ids;
		for (const auto &col_idx : input.column_ids) {
			if (col_idx == COLUMN_IDENTIFIER_ROW_ID) {
				result->scanned_types.emplace_back(LogicalType::ROW_TYPE);
			} else {
				result->scanned_types.push_back(bind_data.all_types[col_idx]);
			}
		}
	}
	return std::move(result);
}

unique_ptr<LocalTableFunctionState>
ArrowTableFunction::ArrowScanInitLocalInternal(ClientContext &context, TableFunctionInitInput &input,
                                               GlobalTableFunctionState *global_state_p) {
	auto &global_state = global_state_p->Cast<ArrowScanGlobalState>();
	auto current_chunk = make_uniq<ArrowArrayWrapper>();
	auto result = make_uniq<ArrowScanLocalState>(std::move(current_chunk));
	result->column_ids = input.column_ids;
	result->filters = input.filters.get();
	if (input.CanRemoveFilterColumns()) {
		auto &asgs = global_state_p->Cast<ArrowScanGlobalState>();
		result->all_columns.Initialize(context, asgs.scanned_types);
	}
	if (!ArrowScanParallelStateNext(context, input.bind_data.get(), *result, global_state)) {
		return nullptr;
	}
	return std::move(result);
}

unique_ptr<LocalTableFunctionState> ArrowTableFunction::ArrowScanInitLocal(ExecutionContext &context,
                                                                           TableFunctionInitInput &input,
                                                                           GlobalTableFunctionState *global_state_p) {
	return ArrowScanInitLocalInternal(context.client, input, global_state_p);
}

void ArrowTableFunction::ArrowScanFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	if (!data_p.local_state) {
		return;
	}
	auto &data = (ArrowScanFunctionData &)*data_p.bind_data;
	auto &state = data_p.local_state->Cast<ArrowScanLocalState>();
	auto &global_state = data_p.global_state->Cast<ArrowScanGlobalState>();

	//! Out of tuples in this chunk
	if (state.chunk_offset >= (idx_t)state.chunk->arrow_array.length) {
		if (!ArrowScanParallelStateNext(context, data_p.bind_data.get(), state, global_state)) {
			return;
		}
	}
	int64_t output_size = MinValue<int64_t>(STANDARD_VECTOR_SIZE, state.chunk->arrow_array.length - state.chunk_offset);
	data.lines_read += output_size;
	if (global_state.CanRemoveFilterColumns()) {
		state.all_columns.Reset();
		state.all_columns.SetCardinality(output_size);
		ArrowToDuckDB(state, data.arrow_convert_data, state.all_columns, data.lines_read - output_size);
		output.ReferenceColumns(state.all_columns, global_state.projection_ids);
	} else {
		output.SetCardinality(output_size);
		ArrowToDuckDB(state, data.arrow_convert_data, output, data.lines_read - output_size);
	}

	output.Verify();
	state.chunk_offset += output.size();
}

unique_ptr<NodeStatistics> ArrowTableFunction::ArrowScanCardinality(ClientContext &context, const FunctionData *data) {
	return make_uniq<NodeStatistics>();
}

idx_t ArrowTableFunction::ArrowGetBatchIndex(ClientContext &context, const FunctionData *bind_data_p,
                                             LocalTableFunctionState *local_state,
                                             GlobalTableFunctionState *global_state) {
	auto &state = local_state->Cast<ArrowScanLocalState>();
	return state.batch_index;
}

void ArrowTableFunction::RegisterFunction(BuiltinFunctions &set) {
	TableFunction arrow("arrow_scan", {LogicalType::POINTER, LogicalType::POINTER, LogicalType::POINTER},
	                    ArrowScanFunction, ArrowScanBind, ArrowScanInitGlobal, ArrowScanInitLocal);
	arrow.cardinality = ArrowScanCardinality;
	arrow.get_batch_index = ArrowGetBatchIndex;
	arrow.projection_pushdown = true;
	arrow.filter_pushdown = true;
	arrow.filter_prune = true;
	set.AddFunction(arrow);

	TableFunction arrow_dumb("arrow_scan_dumb", {LogicalType::POINTER, LogicalType::POINTER, LogicalType::POINTER},
	                         ArrowScanFunction, ArrowScanBind, ArrowScanInitGlobal, ArrowScanInitLocal);
	arrow_dumb.cardinality = ArrowScanCardinality;
	arrow_dumb.get_batch_index = ArrowGetBatchIndex;
	arrow_dumb.projection_pushdown = false;
	arrow_dumb.filter_pushdown = false;
	arrow_dumb.filter_prune = false;
	set.AddFunction(arrow_dumb);
}

void BuiltinFunctions::RegisterArrowFunctions() {
	ArrowTableFunction::RegisterFunction(*this);
}
} // namespace duckdb







namespace {
using duckdb::idx_t;
struct ArrowConvertDataIndices {
	//! The index that refers to 'variable_sz_type' in ArrowConvertData
	idx_t variable_sized_index;
	//! The index that refers to 'date_time_precision' in ArrowConvertData
	idx_t datetime_precision_index;
};
} // namespace

namespace duckdb {

static void ShiftRight(unsigned char *ar, int size, int shift) {
	int carry = 0;
	while (shift--) {
		for (int i = size - 1; i >= 0; --i) {
			int next = (ar[i] & 1) ? 0x80 : 0;
			ar[i] = carry | (ar[i] >> 1);
			carry = next;
		}
	}
}

static void GetValidityMask(ValidityMask &mask, ArrowArray &array, ArrowScanLocalState &scan_state, idx_t size,
                            int64_t nested_offset = -1, bool add_null = false) {
	// In certains we don't need to or cannot copy arrow's validity mask to duckdb.
	//
	// The conditions where we do want to copy arrow's mask to duckdb are:
	// 1. nulls exist
	// 2. n_buffers > 0, meaning the array's arrow type is not `null`
	// 3. the validity buffer (the first buffer) is not a nullptr
	if (array.null_count != 0 && array.n_buffers > 0 && array.buffers[0]) {
		auto bit_offset = scan_state.chunk_offset + array.offset;
		if (nested_offset != -1) {
			bit_offset = nested_offset;
		}
		mask.EnsureWritable();
#if STANDARD_VECTOR_SIZE > 64
		auto n_bitmask_bytes = (size + 8 - 1) / 8;
		if (bit_offset % 8 == 0) {
			//! just memcpy nullmask
			memcpy((void *)mask.GetData(), (uint8_t *)array.buffers[0] + bit_offset / 8, n_bitmask_bytes);
		} else {
			//! need to re-align nullmask
			vector<uint8_t> temp_nullmask(n_bitmask_bytes + 1);
			memcpy(temp_nullmask.data(), (uint8_t *)array.buffers[0] + bit_offset / 8, n_bitmask_bytes + 1);
			ShiftRight(temp_nullmask.data(), n_bitmask_bytes + 1,
			           bit_offset % 8); //! why this has to be a right shift is a mystery to me
			memcpy((void *)mask.GetData(), (data_ptr_t)temp_nullmask.data(), n_bitmask_bytes);
		}
#else
		auto byte_offset = bit_offset / 8;
		auto source_data = (uint8_t *)array.buffers[0];
		bit_offset %= 8;
		for (idx_t i = 0; i < size; i++) {
			mask.Set(i, source_data[byte_offset] & (1 << bit_offset));
			bit_offset++;
			if (bit_offset == 8) {
				bit_offset = 0;
				byte_offset++;
			}
		}
#endif
	}
	if (add_null) {
		//! We are setting a validity mask of the data part of dictionary vector
		//! For some reason, Nulls are allowed to be indexes, hence we need to set the last element here to be null
		//! We might have to resize the mask
		mask.Resize(size, size + 1);
		mask.SetInvalid(size);
	}
}

static void SetValidityMask(Vector &vector, ArrowArray &array, ArrowScanLocalState &scan_state, idx_t size,
                            int64_t nested_offset, bool add_null = false) {
	D_ASSERT(vector.GetVectorType() == VectorType::FLAT_VECTOR);
	auto &mask = FlatVector::Validity(vector);
	GetValidityMask(mask, array, scan_state, size, nested_offset, add_null);
}

static void ColumnArrowToDuckDB(Vector &vector, ArrowArray &array, ArrowScanLocalState &scan_state, idx_t size,
                                std::unordered_map<idx_t, unique_ptr<ArrowConvertData>> &arrow_convert_data,
                                idx_t col_idx, ArrowConvertDataIndices &arrow_convert_idx, int64_t nested_offset = -1,
                                ValidityMask *parent_mask = nullptr);

static void ArrowToDuckDBList(Vector &vector, ArrowArray &array, ArrowScanLocalState &scan_state, idx_t size,
                              std::unordered_map<idx_t, unique_ptr<ArrowConvertData>> &arrow_convert_data,
                              idx_t col_idx, ArrowConvertDataIndices &arrow_convert_idx, int64_t nested_offset,
                              ValidityMask *parent_mask) {
	auto original_type = arrow_convert_data[col_idx]->variable_sz_type[arrow_convert_idx.variable_sized_index++];
	idx_t list_size = 0;
	SetValidityMask(vector, array, scan_state, size, nested_offset);
	idx_t start_offset = 0;
	idx_t cur_offset = 0;
	if (original_type.first == ArrowVariableSizeType::FIXED_SIZE) {
		//! Have to check validity mask before setting this up
		idx_t offset = (scan_state.chunk_offset + array.offset) * original_type.second;
		if (nested_offset != -1) {
			offset = original_type.second * nested_offset;
		}
		start_offset = offset;
		auto list_data = FlatVector::GetData<list_entry_t>(vector);
		for (idx_t i = 0; i < size; i++) {
			auto &le = list_data[i];
			le.offset = cur_offset;
			le.length = original_type.second;
			cur_offset += original_type.second;
		}
		list_size = start_offset + cur_offset;
	} else if (original_type.first == ArrowVariableSizeType::NORMAL) {
		auto offsets = (uint32_t *)array.buffers[1] + array.offset + scan_state.chunk_offset;
		if (nested_offset != -1) {
			offsets = (uint32_t *)array.buffers[1] + nested_offset;
		}
		start_offset = offsets[0];
		auto list_data = FlatVector::GetData<list_entry_t>(vector);
		for (idx_t i = 0; i < size; i++) {
			auto &le = list_data[i];
			le.offset = cur_offset;
			le.length = offsets[i + 1] - offsets[i];
			cur_offset += le.length;
		}
		list_size = offsets[size];
	} else {
		auto offsets = (uint64_t *)array.buffers[1] + array.offset + scan_state.chunk_offset;
		if (nested_offset != -1) {
			offsets = (uint64_t *)array.buffers[1] + nested_offset;
		}
		start_offset = offsets[0];
		auto list_data = FlatVector::GetData<list_entry_t>(vector);
		for (idx_t i = 0; i < size; i++) {
			auto &le = list_data[i];
			le.offset = cur_offset;
			le.length = offsets[i + 1] - offsets[i];
			cur_offset += le.length;
		}
		list_size = offsets[size];
	}
	list_size -= start_offset;
	ListVector::Reserve(vector, list_size);
	ListVector::SetListSize(vector, list_size);
	auto &child_vector = ListVector::GetEntry(vector);
	SetValidityMask(child_vector, *array.children[0], scan_state, list_size, start_offset);
	auto &list_mask = FlatVector::Validity(vector);
	if (parent_mask) {
		//! Since this List is owned by a struct we must guarantee their validity map matches on Null
		if (!parent_mask->AllValid()) {
			for (idx_t i = 0; i < size; i++) {
				if (!parent_mask->RowIsValid(i)) {
					list_mask.SetInvalid(i);
				}
			}
		}
	}
	if (list_size == 0 && start_offset == 0) {
		ColumnArrowToDuckDB(child_vector, *array.children[0], scan_state, list_size, arrow_convert_data, col_idx,
		                    arrow_convert_idx, -1);
	} else {
		ColumnArrowToDuckDB(child_vector, *array.children[0], scan_state, list_size, arrow_convert_data, col_idx,
		                    arrow_convert_idx, start_offset);
	}
}

static void ArrowToDuckDBBlob(Vector &vector, ArrowArray &array, ArrowScanLocalState &scan_state, idx_t size,
                              std::unordered_map<idx_t, unique_ptr<ArrowConvertData>> &arrow_convert_data,
                              idx_t col_idx, ArrowConvertDataIndices &arrow_convert_idx, int64_t nested_offset) {
	auto original_type = arrow_convert_data[col_idx]->variable_sz_type[arrow_convert_idx.variable_sized_index++];
	SetValidityMask(vector, array, scan_state, size, nested_offset);
	if (original_type.first == ArrowVariableSizeType::FIXED_SIZE) {
		//! Have to check validity mask before setting this up
		idx_t offset = (scan_state.chunk_offset + array.offset) * original_type.second;
		if (nested_offset != -1) {
			offset = original_type.second * nested_offset;
		}
		auto cdata = (char *)array.buffers[1];
		for (idx_t row_idx = 0; row_idx < size; row_idx++) {
			if (FlatVector::IsNull(vector, row_idx)) {
				continue;
			}
			auto bptr = cdata + offset;
			auto blob_len = original_type.second;
			FlatVector::GetData<string_t>(vector)[row_idx] = StringVector::AddStringOrBlob(vector, bptr, blob_len);
			offset += blob_len;
		}
	} else if (original_type.first == ArrowVariableSizeType::NORMAL) {
		auto offsets = (uint32_t *)array.buffers[1] + array.offset + scan_state.chunk_offset;
		if (nested_offset != -1) {
			offsets = (uint32_t *)array.buffers[1] + array.offset + nested_offset;
		}
		auto cdata = (char *)array.buffers[2];
		for (idx_t row_idx = 0; row_idx < size; row_idx++) {
			if (FlatVector::IsNull(vector, row_idx)) {
				continue;
			}
			auto bptr = cdata + offsets[row_idx];
			auto blob_len = offsets[row_idx + 1] - offsets[row_idx];
			FlatVector::GetData<string_t>(vector)[row_idx] = StringVector::AddStringOrBlob(vector, bptr, blob_len);
		}
	} else {
		//! Check if last offset is higher than max uint32
		if (((uint64_t *)array.buffers[1])[array.length] > NumericLimits<uint32_t>::Maximum()) { // LCOV_EXCL_START
			throw ConversionException("DuckDB does not support Blobs over 4GB");
		} // LCOV_EXCL_STOP
		auto offsets = (uint64_t *)array.buffers[1] + array.offset + scan_state.chunk_offset;
		if (nested_offset != -1) {
			offsets = (uint64_t *)array.buffers[1] + array.offset + nested_offset;
		}
		auto cdata = (char *)array.buffers[2];
		for (idx_t row_idx = 0; row_idx < size; row_idx++) {
			if (FlatVector::IsNull(vector, row_idx)) {
				continue;
			}
			auto bptr = cdata + offsets[row_idx];
			auto blob_len = offsets[row_idx + 1] - offsets[row_idx];
			FlatVector::GetData<string_t>(vector)[row_idx] = StringVector::AddStringOrBlob(vector, bptr, blob_len);
		}
	}
}

static void ArrowToDuckDBMapVerify(Vector &vector, idx_t count) {
	auto valid_check = MapVector::CheckMapValidity(vector, count);
	switch (valid_check) {
	case MapInvalidReason::VALID:
		break;
	case MapInvalidReason::DUPLICATE_KEY: {
		throw InvalidInputException("Arrow map contains duplicate key, which isn't supported by DuckDB map type");
	}
	case MapInvalidReason::NULL_KEY: {
		throw InvalidInputException("Arrow map contains NULL as map key, which isn't supported by DuckDB map type");
	}
	case MapInvalidReason::NULL_KEY_LIST: {
		throw InvalidInputException("Arrow map contains NULL as key list, which isn't supported by DuckDB map type");
	}
	default: {
		throw InternalException("MapInvalidReason not implemented");
	}
	}
}

template <class T>
static void SetVectorString(Vector &vector, idx_t size, char *cdata, T *offsets) {
	auto strings = FlatVector::GetData<string_t>(vector);
	for (idx_t row_idx = 0; row_idx < size; row_idx++) {
		if (FlatVector::IsNull(vector, row_idx)) {
			continue;
		}
		auto cptr = cdata + offsets[row_idx];
		auto str_len = offsets[row_idx + 1] - offsets[row_idx];
		if (str_len > NumericLimits<uint32_t>::Maximum()) { // LCOV_EXCL_START
			throw ConversionException("DuckDB does not support Strings over 4GB");
		} // LCOV_EXCL_STOP
		strings[row_idx] = string_t(cptr, str_len);
	}
}

static void DirectConversion(Vector &vector, ArrowArray &array, ArrowScanLocalState &scan_state,
                             int64_t nested_offset) {
	auto internal_type = GetTypeIdSize(vector.GetType().InternalType());
	auto data_ptr = (data_ptr_t)array.buffers[1] + internal_type * (scan_state.chunk_offset + array.offset);
	if (nested_offset != -1) {
		data_ptr = (data_ptr_t)array.buffers[1] + internal_type * (array.offset + nested_offset);
	}
	FlatVector::SetData(vector, data_ptr);
}

template <class T>
static void TimeConversion(Vector &vector, ArrowArray &array, ArrowScanLocalState &scan_state, int64_t nested_offset,
                           idx_t size, int64_t conversion) {
	auto tgt_ptr = (dtime_t *)FlatVector::GetData(vector);
	auto &validity_mask = FlatVector::Validity(vector);
	auto src_ptr = (T *)array.buffers[1] + scan_state.chunk_offset + array.offset;
	if (nested_offset != -1) {
		src_ptr = (T *)array.buffers[1] + nested_offset + array.offset;
	}
	for (idx_t row = 0; row < size; row++) {
		if (!validity_mask.RowIsValid(row)) {
			continue;
		}
		if (!TryMultiplyOperator::Operation((int64_t)src_ptr[row], conversion, tgt_ptr[row].micros)) {
			throw ConversionException("Could not convert Time to Microsecond");
		}
	}
}

static void TimestampTZConversion(Vector &vector, ArrowArray &array, ArrowScanLocalState &scan_state,
                                  int64_t nested_offset, idx_t size, int64_t conversion) {
	auto tgt_ptr = (timestamp_t *)FlatVector::GetData(vector);
	auto &validity_mask = FlatVector::Validity(vector);
	auto src_ptr = (int64_t *)array.buffers[1] + scan_state.chunk_offset + array.offset;
	if (nested_offset != -1) {
		src_ptr = (int64_t *)array.buffers[1] + nested_offset + array.offset;
	}
	for (idx_t row = 0; row < size; row++) {
		if (!validity_mask.RowIsValid(row)) {
			continue;
		}
		if (!TryMultiplyOperator::Operation(src_ptr[row], conversion, tgt_ptr[row].value)) {
			throw ConversionException("Could not convert TimestampTZ to Microsecond");
		}
	}
}

static void IntervalConversionUs(Vector &vector, ArrowArray &array, ArrowScanLocalState &scan_state,
                                 int64_t nested_offset, idx_t size, int64_t conversion) {
	auto tgt_ptr = (interval_t *)FlatVector::GetData(vector);
	auto src_ptr = (int64_t *)array.buffers[1] + scan_state.chunk_offset + array.offset;
	if (nested_offset != -1) {
		src_ptr = (int64_t *)array.buffers[1] + nested_offset + array.offset;
	}
	for (idx_t row = 0; row < size; row++) {
		tgt_ptr[row].days = 0;
		tgt_ptr[row].months = 0;
		if (!TryMultiplyOperator::Operation(src_ptr[row], conversion, tgt_ptr[row].micros)) {
			throw ConversionException("Could not convert Interval to Microsecond");
		}
	}
}

static void IntervalConversionMonths(Vector &vector, ArrowArray &array, ArrowScanLocalState &scan_state,
                                     int64_t nested_offset, idx_t size) {
	auto tgt_ptr = (interval_t *)FlatVector::GetData(vector);
	auto src_ptr = (int32_t *)array.buffers[1] + scan_state.chunk_offset + array.offset;
	if (nested_offset != -1) {
		src_ptr = (int32_t *)array.buffers[1] + nested_offset + array.offset;
	}
	for (idx_t row = 0; row < size; row++) {
		tgt_ptr[row].days = 0;
		tgt_ptr[row].micros = 0;
		tgt_ptr[row].months = src_ptr[row];
	}
}

static void IntervalConversionMonthDayNanos(Vector &vector, ArrowArray &array, ArrowScanLocalState &scan_state,
                                            int64_t nested_offset, idx_t size) {
	auto tgt_ptr = (interval_t *)FlatVector::GetData(vector);
	auto src_ptr = (ArrowInterval *)array.buffers[1] + scan_state.chunk_offset + array.offset;
	if (nested_offset != -1) {
		src_ptr = (ArrowInterval *)array.buffers[1] + nested_offset + array.offset;
	}
	for (idx_t row = 0; row < size; row++) {
		tgt_ptr[row].days = src_ptr[row].days;
		tgt_ptr[row].micros = src_ptr[row].nanoseconds / Interval::NANOS_PER_MICRO;
		tgt_ptr[row].months = src_ptr[row].months;
	}
}

static void ColumnArrowToDuckDB(Vector &vector, ArrowArray &array, ArrowScanLocalState &scan_state, idx_t size,
                                std::unordered_map<idx_t, unique_ptr<ArrowConvertData>> &arrow_convert_data,
                                idx_t col_idx, ArrowConvertDataIndices &arrow_convert_idx, int64_t nested_offset,
                                ValidityMask *parent_mask) {
	switch (vector.GetType().id()) {
	case LogicalTypeId::SQLNULL:
		vector.Reference(Value());
		break;
	case LogicalTypeId::BOOLEAN: {
		//! Arrow bit-packs boolean values
		//! Lets first figure out where we are in the source array
		auto src_ptr = (uint8_t *)array.buffers[1] + (scan_state.chunk_offset + array.offset) / 8;

		if (nested_offset != -1) {
			src_ptr = (uint8_t *)array.buffers[1] + (nested_offset + array.offset) / 8;
		}
		auto tgt_ptr = (uint8_t *)FlatVector::GetData(vector);
		int src_pos = 0;
		idx_t cur_bit = scan_state.chunk_offset % 8;
		if (nested_offset != -1) {
			cur_bit = nested_offset % 8;
		}
		for (idx_t row = 0; row < size; row++) {
			if ((src_ptr[src_pos] & (1 << cur_bit)) == 0) {
				tgt_ptr[row] = 0;
			} else {
				tgt_ptr[row] = 1;
			}
			cur_bit++;
			if (cur_bit == 8) {
				src_pos++;
				cur_bit = 0;
			}
		}
		break;
	}
	case LogicalTypeId::TINYINT:
	case LogicalTypeId::SMALLINT:
	case LogicalTypeId::INTEGER:
	case LogicalTypeId::FLOAT:
	case LogicalTypeId::DOUBLE:
	case LogicalTypeId::UTINYINT:
	case LogicalTypeId::USMALLINT:
	case LogicalTypeId::UINTEGER:
	case LogicalTypeId::UBIGINT:
	case LogicalTypeId::BIGINT:
	case LogicalTypeId::HUGEINT:
	case LogicalTypeId::TIMESTAMP:
	case LogicalTypeId::TIMESTAMP_SEC:
	case LogicalTypeId::TIMESTAMP_MS:
	case LogicalTypeId::TIMESTAMP_NS: {
		DirectConversion(vector, array, scan_state, nested_offset);
		break;
	}
	case LogicalTypeId::VARCHAR: {
		auto original_type = arrow_convert_data[col_idx]->variable_sz_type[arrow_convert_idx.variable_sized_index++];
		auto cdata = (char *)array.buffers[2];
		if (original_type.first == ArrowVariableSizeType::SUPER_SIZE) {
			auto offsets = (uint64_t *)array.buffers[1] + array.offset + scan_state.chunk_offset;
			if (nested_offset != -1) {
				offsets = (uint64_t *)array.buffers[1] + array.offset + nested_offset;
			}
			SetVectorString(vector, size, cdata, offsets);
		} else {
			auto offsets = (uint32_t *)array.buffers[1] + array.offset + scan_state.chunk_offset;
			if (nested_offset != -1) {
				offsets = (uint32_t *)array.buffers[1] + array.offset + nested_offset;
			}
			SetVectorString(vector, size, cdata, offsets);
		}
		break;
	}
	case LogicalTypeId::DATE: {
		auto precision = arrow_convert_data[col_idx]->date_time_precision[arrow_convert_idx.datetime_precision_index++];
		switch (precision) {
		case ArrowDateTimeType::DAYS: {
			DirectConversion(vector, array, scan_state, nested_offset);
			break;
		}
		case ArrowDateTimeType::MILLISECONDS: {
			//! convert date from nanoseconds to days
			auto src_ptr = (uint64_t *)array.buffers[1] + scan_state.chunk_offset + array.offset;
			if (nested_offset != -1) {
				src_ptr = (uint64_t *)array.buffers[1] + nested_offset + array.offset;
			}
			auto tgt_ptr = (date_t *)FlatVector::GetData(vector);
			for (idx_t row = 0; row < size; row++) {
				tgt_ptr[row] = date_t(int64_t(src_ptr[row]) / static_cast<int64_t>(1000 * 60 * 60 * 24));
			}
			break;
		}
		default:
			throw NotImplementedException("Unsupported precision for Date Type ");
		}
		break;
	}
	case LogicalTypeId::TIME: {
		auto precision = arrow_convert_data[col_idx]->date_time_precision[arrow_convert_idx.datetime_precision_index++];
		switch (precision) {
		case ArrowDateTimeType::SECONDS: {
			TimeConversion<int32_t>(vector, array, scan_state, nested_offset, size, 1000000);
			break;
		}
		case ArrowDateTimeType::MILLISECONDS: {
			TimeConversion<int32_t>(vector, array, scan_state, nested_offset, size, 1000);
			break;
		}
		case ArrowDateTimeType::MICROSECONDS: {
			TimeConversion<int64_t>(vector, array, scan_state, nested_offset, size, 1);
			break;
		}
		case ArrowDateTimeType::NANOSECONDS: {
			auto tgt_ptr = (dtime_t *)FlatVector::GetData(vector);
			auto src_ptr = (int64_t *)array.buffers[1] + scan_state.chunk_offset + array.offset;
			if (nested_offset != -1) {
				src_ptr = (int64_t *)array.buffers[1] + nested_offset + array.offset;
			}
			for (idx_t row = 0; row < size; row++) {
				tgt_ptr[row].micros = src_ptr[row] / 1000;
			}
			break;
		}
		default:
			throw NotImplementedException("Unsupported precision for Time Type ");
		}
		break;
	}
	case LogicalTypeId::TIMESTAMP_TZ: {
		auto precision = arrow_convert_data[col_idx]->date_time_precision[arrow_convert_idx.datetime_precision_index++];
		switch (precision) {
		case ArrowDateTimeType::SECONDS: {
			TimestampTZConversion(vector, array, scan_state, nested_offset, size, 1000000);
			break;
		}
		case ArrowDateTimeType::MILLISECONDS: {
			TimestampTZConversion(vector, array, scan_state, nested_offset, size, 1000);
			break;
		}
		case ArrowDateTimeType::MICROSECONDS: {
			DirectConversion(vector, array, scan_state, nested_offset);
			break;
		}
		case ArrowDateTimeType::NANOSECONDS: {
			auto tgt_ptr = (timestamp_t *)FlatVector::GetData(vector);
			auto src_ptr = (int64_t *)array.buffers[1] + scan_state.chunk_offset + array.offset;
			if (nested_offset != -1) {
				src_ptr = (int64_t *)array.buffers[1] + nested_offset + array.offset;
			}
			for (idx_t row = 0; row < size; row++) {
				tgt_ptr[row].value = src_ptr[row] / 1000;
			}
			break;
		}
		default:
			throw NotImplementedException("Unsupported precision for TimestampTZ Type ");
		}
		break;
	}
	case LogicalTypeId::INTERVAL: {
		auto precision = arrow_convert_data[col_idx]->date_time_precision[arrow_convert_idx.datetime_precision_index++];
		switch (precision) {
		case ArrowDateTimeType::SECONDS: {
			IntervalConversionUs(vector, array, scan_state, nested_offset, size, 1000000);
			break;
		}
		case ArrowDateTimeType::DAYS:
		case ArrowDateTimeType::MILLISECONDS: {
			IntervalConversionUs(vector, array, scan_state, nested_offset, size, 1000);
			break;
		}
		case ArrowDateTimeType::MICROSECONDS: {
			IntervalConversionUs(vector, array, scan_state, nested_offset, size, 1);
			break;
		}
		case ArrowDateTimeType::NANOSECONDS: {
			auto tgt_ptr = (interval_t *)FlatVector::GetData(vector);
			auto src_ptr = (int64_t *)array.buffers[1] + scan_state.chunk_offset + array.offset;
			if (nested_offset != -1) {
				src_ptr = (int64_t *)array.buffers[1] + nested_offset + array.offset;
			}
			for (idx_t row = 0; row < size; row++) {
				tgt_ptr[row].micros = src_ptr[row] / 1000;
				tgt_ptr[row].days = 0;
				tgt_ptr[row].months = 0;
			}
			break;
		}
		case ArrowDateTimeType::MONTHS: {
			IntervalConversionMonths(vector, array, scan_state, nested_offset, size);
			break;
		}
		case ArrowDateTimeType::MONTH_DAY_NANO: {
			IntervalConversionMonthDayNanos(vector, array, scan_state, nested_offset, size);
			break;
		}
		default:
			throw NotImplementedException("Unsupported precision for Interval/Duration Type ");
		}
		break;
	}
	case LogicalTypeId::DECIMAL: {
		auto val_mask = FlatVector::Validity(vector);
		//! We have to convert from INT128
		auto src_ptr = (hugeint_t *)array.buffers[1] + scan_state.chunk_offset + array.offset;
		if (nested_offset != -1) {
			src_ptr = (hugeint_t *)array.buffers[1] + nested_offset + array.offset;
		}
		switch (vector.GetType().InternalType()) {
		case PhysicalType::INT16: {
			auto tgt_ptr = (int16_t *)FlatVector::GetData(vector);
			for (idx_t row = 0; row < size; row++) {
				if (val_mask.RowIsValid(row)) {
					auto result = Hugeint::TryCast(src_ptr[row], tgt_ptr[row]);
					D_ASSERT(result);
					(void)result;
				}
			}
			break;
		}
		case PhysicalType::INT32: {
			auto tgt_ptr = (int32_t *)FlatVector::GetData(vector);
			for (idx_t row = 0; row < size; row++) {
				if (val_mask.RowIsValid(row)) {
					auto result = Hugeint::TryCast(src_ptr[row], tgt_ptr[row]);
					D_ASSERT(result);
					(void)result;
				}
			}
			break;
		}
		case PhysicalType::INT64: {
			auto tgt_ptr = (int64_t *)FlatVector::GetData(vector);
			for (idx_t row = 0; row < size; row++) {
				if (val_mask.RowIsValid(row)) {
					auto result = Hugeint::TryCast(src_ptr[row], tgt_ptr[row]);
					D_ASSERT(result);
					(void)result;
				}
			}
			break;
		}
		case PhysicalType::INT128: {
			FlatVector::SetData(vector, (data_ptr_t)array.buffers[1] + GetTypeIdSize(vector.GetType().InternalType()) *
			                                                               (scan_state.chunk_offset + array.offset));
			break;
		}
		default:
			throw NotImplementedException("Unsupported physical type for Decimal: %s",
			                              TypeIdToString(vector.GetType().InternalType()));
		}
		break;
	}
	case LogicalTypeId::BLOB: {
		ArrowToDuckDBBlob(vector, array, scan_state, size, arrow_convert_data, col_idx, arrow_convert_idx,
		                  nested_offset);
		break;
	}
	case LogicalTypeId::LIST: {
		ArrowToDuckDBList(vector, array, scan_state, size, arrow_convert_data, col_idx, arrow_convert_idx,
		                  nested_offset, parent_mask);
		break;
	}
	case LogicalTypeId::MAP: {
		ArrowToDuckDBList(vector, array, scan_state, size, arrow_convert_data, col_idx, arrow_convert_idx,
		                  nested_offset, parent_mask);
		ArrowToDuckDBMapVerify(vector, size);
		break;
	}
	case LogicalTypeId::STRUCT: {
		//! Fill the children
		auto &child_entries = StructVector::GetEntries(vector);
		auto &struct_validity_mask = FlatVector::Validity(vector);
		for (idx_t type_idx = 0; type_idx < (idx_t)array.n_children; type_idx++) {
			SetValidityMask(*child_entries[type_idx], *array.children[type_idx], scan_state, size, nested_offset);
			if (!struct_validity_mask.AllValid()) {
				auto &child_validity_mark = FlatVector::Validity(*child_entries[type_idx]);
				for (idx_t i = 0; i < size; i++) {
					if (!struct_validity_mask.RowIsValid(i)) {
						child_validity_mark.SetInvalid(i);
					}
				}
			}
			ColumnArrowToDuckDB(*child_entries[type_idx], *array.children[type_idx], scan_state, size,
			                    arrow_convert_data, col_idx, arrow_convert_idx, nested_offset, &struct_validity_mask);
		}
		break;
	}
	default:
		throw NotImplementedException("Unsupported type %s", vector.GetType().ToString());
	}
}

template <class T>
static void SetSelectionVectorLoop(SelectionVector &sel, data_ptr_t indices_p, idx_t size) {
	auto indices = (T *)indices_p;
	for (idx_t row = 0; row < size; row++) {
		sel.set_index(row, indices[row]);
	}
}

template <class T>
static void SetSelectionVectorLoopWithChecks(SelectionVector &sel, data_ptr_t indices_p, idx_t size) {

	auto indices = (T *)indices_p;
	for (idx_t row = 0; row < size; row++) {
		if (indices[row] > NumericLimits<uint32_t>::Maximum()) {
			throw ConversionException("DuckDB only supports indices that fit on an uint32");
		}
		sel.set_index(row, indices[row]);
	}
}

template <class T>
static void SetMaskedSelectionVectorLoop(SelectionVector &sel, data_ptr_t indices_p, idx_t size, ValidityMask &mask,
                                         idx_t last_element_pos) {
	auto indices = (T *)indices_p;
	for (idx_t row = 0; row < size; row++) {
		if (mask.RowIsValid(row)) {
			sel.set_index(row, indices[row]);
		} else {
			//! Need to point out to last element
			sel.set_index(row, last_element_pos);
		}
	}
}

static void SetSelectionVector(SelectionVector &sel, data_ptr_t indices_p, LogicalType &logical_type, idx_t size,
                               ValidityMask *mask = nullptr, idx_t last_element_pos = 0) {
	sel.Initialize(size);

	if (mask) {
		switch (logical_type.id()) {
		case LogicalTypeId::UTINYINT:
			SetMaskedSelectionVectorLoop<uint8_t>(sel, indices_p, size, *mask, last_element_pos);
			break;
		case LogicalTypeId::TINYINT:
			SetMaskedSelectionVectorLoop<int8_t>(sel, indices_p, size, *mask, last_element_pos);
			break;
		case LogicalTypeId::USMALLINT:
			SetMaskedSelectionVectorLoop<uint16_t>(sel, indices_p, size, *mask, last_element_pos);
			break;
		case LogicalTypeId::SMALLINT:
			SetMaskedSelectionVectorLoop<int16_t>(sel, indices_p, size, *mask, last_element_pos);
			break;
		case LogicalTypeId::UINTEGER:
			if (last_element_pos > NumericLimits<uint32_t>::Maximum()) {
				//! Its guaranteed that our indices will point to the last element, so just throw an error
				throw ConversionException("DuckDB only supports indices that fit on an uint32");
			}
			SetMaskedSelectionVectorLoop<uint32_t>(sel, indices_p, size, *mask, last_element_pos);
			break;
		case LogicalTypeId::INTEGER:
			SetMaskedSelectionVectorLoop<int32_t>(sel, indices_p, size, *mask, last_element_pos);
			break;
		case LogicalTypeId::UBIGINT:
			if (last_element_pos > NumericLimits<uint32_t>::Maximum()) {
				//! Its guaranteed that our indices will point to the last element, so just throw an error
				throw ConversionException("DuckDB only supports indices that fit on an uint32");
			}
			SetMaskedSelectionVectorLoop<uint64_t>(sel, indices_p, size, *mask, last_element_pos);
			break;
		case LogicalTypeId::BIGINT:
			if (last_element_pos > NumericLimits<uint32_t>::Maximum()) {
				//! Its guaranteed that our indices will point to the last element, so just throw an error
				throw ConversionException("DuckDB only supports indices that fit on an uint32");
			}
			SetMaskedSelectionVectorLoop<int64_t>(sel, indices_p, size, *mask, last_element_pos);
			break;

		default:
			throw NotImplementedException("(Arrow) Unsupported type for selection vectors %s", logical_type.ToString());
		}

	} else {
		switch (logical_type.id()) {
		case LogicalTypeId::UTINYINT:
			SetSelectionVectorLoop<uint8_t>(sel, indices_p, size);
			break;
		case LogicalTypeId::TINYINT:
			SetSelectionVectorLoop<int8_t>(sel, indices_p, size);
			break;
		case LogicalTypeId::USMALLINT:
			SetSelectionVectorLoop<uint16_t>(sel, indices_p, size);
			break;
		case LogicalTypeId::SMALLINT:
			SetSelectionVectorLoop<int16_t>(sel, indices_p, size);
			break;
		case LogicalTypeId::UINTEGER:
			SetSelectionVectorLoop<uint32_t>(sel, indices_p, size);
			break;
		case LogicalTypeId::INTEGER:
			SetSelectionVectorLoop<int32_t>(sel, indices_p, size);
			break;
		case LogicalTypeId::UBIGINT:
			if (last_element_pos > NumericLimits<uint32_t>::Maximum()) {
				//! We need to check if our indexes fit in a uint32_t
				SetSelectionVectorLoopWithChecks<uint64_t>(sel, indices_p, size);
			} else {
				SetSelectionVectorLoop<uint64_t>(sel, indices_p, size);
			}
			break;
		case LogicalTypeId::BIGINT:
			if (last_element_pos > NumericLimits<uint32_t>::Maximum()) {
				//! We need to check if our indexes fit in a uint32_t
				SetSelectionVectorLoopWithChecks<int64_t>(sel, indices_p, size);
			} else {
				SetSelectionVectorLoop<int64_t>(sel, indices_p, size);
			}
			break;
		default:
			throw ConversionException("(Arrow) Unsupported type for selection vectors %s", logical_type.ToString());
		}
	}
}

static void ColumnArrowToDuckDBDictionary(Vector &vector, ArrowArray &array, ArrowScanLocalState &scan_state,
                                          idx_t size,
                                          std::unordered_map<idx_t, unique_ptr<ArrowConvertData>> &arrow_convert_data,
                                          idx_t col_idx, ArrowConvertDataIndices &arrow_convert_idx) {
	SelectionVector sel;
	auto &dict_vectors = scan_state.arrow_dictionary_vectors;
	if (!dict_vectors.count(col_idx)) {
		//! We need to set the dictionary data for this column
		auto base_vector = make_uniq<Vector>(vector.GetType(), array.dictionary->length);
		SetValidityMask(*base_vector, *array.dictionary, scan_state, array.dictionary->length, 0, array.null_count > 0);
		ColumnArrowToDuckDB(*base_vector, *array.dictionary, scan_state, array.dictionary->length, arrow_convert_data,
		                    col_idx, arrow_convert_idx);
		dict_vectors[col_idx] = std::move(base_vector);
	}
	auto dictionary_type = arrow_convert_data[col_idx]->dictionary_type;
	//! Get Pointer to Indices of Dictionary
	auto indices = (data_ptr_t)array.buffers[1] +
	               GetTypeIdSize(dictionary_type.InternalType()) * (scan_state.chunk_offset + array.offset);
	if (array.null_count > 0) {
		ValidityMask indices_validity;
		GetValidityMask(indices_validity, array, scan_state, size);
		SetSelectionVector(sel, indices, dictionary_type, size, &indices_validity, array.dictionary->length);
	} else {
		SetSelectionVector(sel, indices, dictionary_type, size);
	}
	vector.Slice(*dict_vectors[col_idx], sel, size);
}

void ArrowTableFunction::ArrowToDuckDB(ArrowScanLocalState &scan_state,
                                       unordered_map<idx_t, unique_ptr<ArrowConvertData>> &arrow_convert_data,
                                       DataChunk &output, idx_t start, bool arrow_scan_is_projected) {
	for (idx_t idx = 0; idx < output.ColumnCount(); idx++) {
		auto col_idx = scan_state.column_ids[idx];

		// If projection was not pushed down into the arrow scanner, but projection pushdown is enabled on the
		// table function, we need to use original column ids here.
		auto arrow_array_idx = arrow_scan_is_projected ? idx : col_idx;

		if (col_idx == COLUMN_IDENTIFIER_ROW_ID) {
			// This column is skipped by the projection pushdown
			continue;
		}

		ArrowConvertDataIndices arrow_convert_idx {0, 0};
		auto &array = *scan_state.chunk->arrow_array.children[arrow_array_idx];
		if (!array.release) {
			throw InvalidInputException("arrow_scan: released array passed");
		}
		if (array.length != scan_state.chunk->arrow_array.length) {
			throw InvalidInputException("arrow_scan: array length mismatch");
		}
		// Make sure this Vector keeps the Arrow chunk alive in case we can zero-copy the data
		output.data[idx].GetBuffer()->SetAuxiliaryData(make_uniq<ArrowAuxiliaryData>(scan_state.chunk));
		if (array.dictionary) {
			ColumnArrowToDuckDBDictionary(output.data[idx], array, scan_state, output.size(), arrow_convert_data,
			                              col_idx, arrow_convert_idx);
		} else {
			SetValidityMask(output.data[idx], array, scan_state, output.size(), -1);
			ColumnArrowToDuckDB(output.data[idx], array, scan_state, output.size(), arrow_convert_data, col_idx,
			                    arrow_convert_idx);
		}
	}
}

} // namespace duckdb







namespace duckdb {

struct CheckpointBindData : public FunctionData {
	explicit CheckpointBindData(optional_ptr<AttachedDatabase> db) : db(db) {
	}

	optional_ptr<AttachedDatabase> db;

public:
	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<CheckpointBindData>(db);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = (const CheckpointBindData &)other_p;
		return db == other.db;
	}
};

static unique_ptr<FunctionData> CheckpointBind(ClientContext &context, TableFunctionBindInput &input,
                                               vector<LogicalType> &return_types, vector<string> &names) {
	return_types.emplace_back(LogicalType::BOOLEAN);
	names.emplace_back("Success");

	optional_ptr<AttachedDatabase> db;
	auto &db_manager = DatabaseManager::Get(context);
	if (!input.inputs.empty()) {
		auto &db_name = StringValue::Get(input.inputs[0]);
		db = db_manager.GetDatabase(context, db_name);
		if (!db) {
			throw BinderException("Database \"%s\" not found", db_name);
		}
	} else {
		db = db_manager.GetDatabase(context, DatabaseManager::GetDefaultDatabase(context));
	}
	return make_uniq<CheckpointBindData>(db);
}

template <bool FORCE>
static void TemplatedCheckpointFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind_data = data_p.bind_data->Cast<CheckpointBindData>();
	auto &transaction_manager = TransactionManager::Get(*bind_data.db.get_mutable());
	transaction_manager.Checkpoint(context, FORCE);
}

void CheckpointFunction::RegisterFunction(BuiltinFunctions &set) {
	TableFunctionSet checkpoint("checkpoint");
	checkpoint.AddFunction(TableFunction({}, TemplatedCheckpointFunction<false>, CheckpointBind));
	checkpoint.AddFunction(TableFunction({LogicalType::VARCHAR}, TemplatedCheckpointFunction<false>, CheckpointBind));
	set.AddFunction(checkpoint);

	TableFunctionSet force_checkpoint("force_checkpoint");
	force_checkpoint.AddFunction(TableFunction({}, TemplatedCheckpointFunction<true>, CheckpointBind));
	force_checkpoint.AddFunction(
	    TableFunction({LogicalType::VARCHAR}, TemplatedCheckpointFunction<true>, CheckpointBind));
	set.AddFunction(force_checkpoint);
}

} // namespace duckdb













#include <limits>

namespace duckdb {

void SubstringDetection(string &str_1, string &str_2, const string &name_str_1, const string &name_str_2) {
	if (str_1.empty() || str_2.empty()) {
		return;
	}
	if ((str_1.find(str_2) != string::npos || str_2.find(str_1) != std::string::npos)) {
		throw BinderException("%s must not appear in the %s specification and vice versa", name_str_1, name_str_2);
	}
}

//===--------------------------------------------------------------------===//
// Bind
//===--------------------------------------------------------------------===//

void BaseCSVData::Finalize() {
	// verify that the options are correct in the final pass
	if (options.escape.empty()) {
		options.escape = options.quote;
	}
	// escape and delimiter must not be substrings of each other
	if (options.has_delimiter && options.has_escape) {
		SubstringDetection(options.delimiter, options.escape, "DELIMITER", "ESCAPE");
	}
	// delimiter and quote must not be substrings of each other
	if (options.has_quote && options.has_delimiter) {
		SubstringDetection(options.quote, options.delimiter, "DELIMITER", "QUOTE");
	}
	// escape and quote must not be substrings of each other (but can be the same)
	if (options.quote != options.escape && options.has_quote && options.has_escape) {
		SubstringDetection(options.quote, options.escape, "QUOTE", "ESCAPE");
	}
	if (!options.null_str.empty()) {
		// null string and delimiter must not be substrings of each other
		if (options.has_delimiter) {
			SubstringDetection(options.delimiter, options.null_str, "DELIMITER", "NULL");
		}
		// quote/escape and nullstr must not be substrings of each other
		if (options.has_quote) {
			SubstringDetection(options.quote, options.null_str, "QUOTE", "NULL");
		}
		if (options.has_escape) {
			SubstringDetection(options.escape, options.null_str, "ESCAPE", "NULL");
		}
	}

	if (!options.prefix.empty() || !options.suffix.empty()) {
		if (options.prefix.empty() || options.suffix.empty()) {
			throw BinderException("COPY ... (FORMAT CSV) must have both PREFIX and SUFFIX, or none at all");
		}
		if (options.header) {
			throw BinderException("COPY ... (FORMAT CSV)'s HEADER cannot be combined with PREFIX/SUFFIX");
		}
	}
}

static unique_ptr<FunctionData> WriteCSVBind(ClientContext &context, CopyInfo &info, vector<string> &names,
                                             vector<LogicalType> &sql_types) {
	auto bind_data = make_uniq<WriteCSVData>(info.file_path, sql_types, names);

	// check all the options in the copy info
	for (auto &option : info.options) {
		auto loption = StringUtil::Lower(option.first);
		auto &set = option.second;
		bind_data->options.SetWriteOption(loption, ConvertVectorToValue(std::move(set)));
	}
	// verify the parsed options
	if (bind_data->options.force_quote.empty()) {
		// no FORCE_QUOTE specified: initialize to false
		bind_data->options.force_quote.resize(names.size(), false);
	}
	bind_data->Finalize();
	bind_data->is_simple = bind_data->options.delimiter.size() == 1 && bind_data->options.escape.size() == 1 &&
	                       bind_data->options.quote.size() == 1;
	if (bind_data->is_simple) {
		bind_data->requires_quotes = make_unsafe_uniq_array<bool>(256);
		memset(bind_data->requires_quotes.get(), 0, sizeof(bool) * 256);
		bind_data->requires_quotes['\n'] = true;
		bind_data->requires_quotes['\r'] = true;
		bind_data->requires_quotes[bind_data->options.delimiter[0]] = true;
		bind_data->requires_quotes[bind_data->options.quote[0]] = true;
	}
	if (!bind_data->options.write_newline.empty()) {
		bind_data->newline = bind_data->options.write_newline;
	}
	return std::move(bind_data);
}

static unique_ptr<FunctionData> ReadCSVBind(ClientContext &context, CopyInfo &info, vector<string> &expected_names,
                                            vector<LogicalType> &expected_types) {
	auto bind_data = make_uniq<ReadCSVData>();
	bind_data->csv_types = expected_types;
	bind_data->csv_names = expected_names;
	bind_data->return_types = expected_types;
	bind_data->return_names = expected_names;
	bind_data->files = MultiFileReader::GetFileList(context, Value(info.file_path), "CSV");

	auto &options = bind_data->options;

	// check all the options in the copy info
	for (auto &option : info.options) {
		auto loption = StringUtil::Lower(option.first);
		auto &set = option.second;
		options.SetReadOption(loption, ConvertVectorToValue(std::move(set)), expected_names);
	}
	// verify the parsed options
	if (options.force_not_null.empty()) {
		// no FORCE_QUOTE specified: initialize to false
		options.force_not_null.resize(expected_types.size(), false);
	}
	bind_data->FinalizeRead(context);
	if (!bind_data->single_threaded && options.auto_detect) {
		options.file_path = bind_data->files[0];
		options.name_list = expected_names;
		auto initial_reader = make_uniq<BufferedCSVReader>(context, options, expected_types);
		options = initial_reader->options;
	}
	return std::move(bind_data);
}

//===--------------------------------------------------------------------===//
// Helper writing functions
//===--------------------------------------------------------------------===//
static string AddEscapes(string &to_be_escaped, const string &escape, const string &val) {
	idx_t i = 0;
	string new_val = "";
	idx_t found = val.find(to_be_escaped);

	while (found != string::npos) {
		while (i < found) {
			new_val += val[i];
			i++;
		}
		new_val += escape;
		found = val.find(to_be_escaped, found + escape.length());
	}
	while (i < val.length()) {
		new_val += val[i];
		i++;
	}
	return new_val;
}

static bool RequiresQuotes(WriteCSVData &csv_data, const char *str, idx_t len) {
	auto &options = csv_data.options;
	// check if the string is equal to the null string
	if (len == options.null_str.size() && memcmp(str, options.null_str.c_str(), len) == 0) {
		return true;
	}
	if (csv_data.is_simple) {
		// simple CSV: check for newlines, quotes and delimiter all at once
		auto str_data = reinterpret_cast<const_data_ptr_t>(str);
		for (idx_t i = 0; i < len; i++) {
			if (csv_data.requires_quotes[str_data[i]]) {
				// this byte requires quotes - write a quoted string
				return true;
			}
		}
		// no newline, quote or delimiter in the string
		// no quoting or escaping necessary
		return false;
	} else {
		// CSV with complex quotes/delimiter (multiple bytes)

		// first check for \n, \r, \n\r in string
		for (idx_t i = 0; i < len; i++) {
			if (str[i] == '\n' || str[i] == '\r') {
				// newline, write a quoted string
				return true;
			}
		}

		// check for delimiter
		if (options.delimiter.length() != 0 &&
		    ContainsFun::Find((const unsigned char *)str, len, (const unsigned char *)options.delimiter.c_str(),
		                      options.delimiter.size()) != DConstants::INVALID_INDEX) {
			return true;
		}
		// check for quote
		if (options.quote.length() != 0 &&
		    ContainsFun::Find((const unsigned char *)str, len, (const unsigned char *)options.quote.c_str(),
		                      options.quote.size()) != DConstants::INVALID_INDEX) {
			return true;
		}
		return false;
	}
}

static void WriteQuotedString(Serializer &serializer, WriteCSVData &csv_data, const char *str, idx_t len,
                              bool force_quote) {
	auto &options = csv_data.options;
	if (!force_quote) {
		// force quote is disabled: check if we need to add quotes anyway
		force_quote = RequiresQuotes(csv_data, str, len);
	}
	if (force_quote) {
		// quoting is enabled: we might need to escape things in the string
		bool requires_escape = false;
		if (csv_data.is_simple) {
			// simple CSV
			// do a single loop to check for a quote or escape value
			for (idx_t i = 0; i < len; i++) {
				if (str[i] == options.quote[0] || str[i] == options.escape[0]) {
					requires_escape = true;
					break;
				}
			}
		} else {
			// complex CSV
			// check for quote or escape separately
			if (options.quote.length() != 0 &&
			    ContainsFun::Find((const unsigned char *)str, len, (const unsigned char *)options.quote.c_str(),
			                      options.quote.size()) != DConstants::INVALID_INDEX) {
				requires_escape = true;
			} else if (options.escape.length() != 0 &&
			           ContainsFun::Find((const unsigned char *)str, len, (const unsigned char *)options.escape.c_str(),
			                             options.escape.size()) != DConstants::INVALID_INDEX) {
				requires_escape = true;
			}
		}
		if (!requires_escape) {
			// fast path: no need to escape anything
			serializer.WriteBufferData(options.quote);
			serializer.WriteData((const_data_ptr_t)str, len);
			serializer.WriteBufferData(options.quote);
			return;
		}

		// slow path: need to add escapes
		string new_val(str, len);
		new_val = AddEscapes(options.escape, options.escape, new_val);
		if (options.escape != options.quote) {
			// need to escape quotes separately
			new_val = AddEscapes(options.quote, options.escape, new_val);
		}
		serializer.WriteBufferData(options.quote);
		serializer.WriteBufferData(new_val);
		serializer.WriteBufferData(options.quote);
	} else {
		serializer.WriteData((const_data_ptr_t)str, len);
	}
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
struct LocalWriteCSVData : public LocalFunctionData {
	//! The thread-local buffer to write data into
	BufferedSerializer serializer;
	//! A chunk with VARCHAR columns to cast intermediates into
	DataChunk cast_chunk;
	//! If we've written any rows yet, allows us to prevent a trailing comma when writing JSON ARRAY
	bool written_anything = false;
};

struct GlobalWriteCSVData : public GlobalFunctionData {
	GlobalWriteCSVData(FileSystem &fs, const string &file_path, FileCompressionType compression)
	    : fs(fs), written_anything(false) {
		handle = fs.OpenFile(file_path, FileFlags::FILE_FLAGS_WRITE | FileFlags::FILE_FLAGS_FILE_CREATE_NEW,
		                     FileLockType::WRITE_LOCK, compression);
	}

	//! Write generic data, e.g., CSV header
	void WriteData(const_data_ptr_t data, idx_t size) {
		lock_guard<mutex> flock(lock);
		handle->Write((void *)data, size);
	}

	//! Write rows
	void WriteRows(const_data_ptr_t data, idx_t size, const string &newline) {
		lock_guard<mutex> flock(lock);
		if (written_anything) {
			handle->Write((void *)newline.c_str(), newline.length());
		} else {
			written_anything = true;
		}
		handle->Write((void *)data, size);
	}

	FileSystem &fs;
	//! The mutex for writing to the physical file
	mutex lock;
	//! The file handle to write to
	unique_ptr<FileHandle> handle;
	//! If we've written any rows yet, allows us to prevent a trailing comma when writing JSON ARRAY
	bool written_anything;
};

static unique_ptr<LocalFunctionData> WriteCSVInitializeLocal(ExecutionContext &context, FunctionData &bind_data) {
	auto &csv_data = bind_data.Cast<WriteCSVData>();
	auto local_data = make_uniq<LocalWriteCSVData>();

	// create the chunk with VARCHAR types
	vector<LogicalType> types;
	types.resize(csv_data.options.name_list.size(), LogicalType::VARCHAR);

	local_data->cast_chunk.Initialize(Allocator::Get(context.client), types);
	return std::move(local_data);
}

static unique_ptr<GlobalFunctionData> WriteCSVInitializeGlobal(ClientContext &context, FunctionData &bind_data,
                                                               const string &file_path) {
	auto &csv_data = bind_data.Cast<WriteCSVData>();
	auto &options = csv_data.options;
	auto global_data =
	    make_uniq<GlobalWriteCSVData>(FileSystem::GetFileSystem(context), file_path, options.compression);

	if (!options.prefix.empty()) {
		global_data->WriteData((const_data_ptr_t)options.prefix.c_str(), options.prefix.size());
	}

	if (options.header) {
		BufferedSerializer serializer;
		// write the header line to the file
		for (idx_t i = 0; i < csv_data.options.name_list.size(); i++) {
			if (i != 0) {
				serializer.WriteBufferData(options.delimiter);
			}
			WriteQuotedString(serializer, csv_data, csv_data.options.name_list[i].c_str(),
			                  csv_data.options.name_list[i].size(), false);
		}
		serializer.WriteBufferData(csv_data.newline);

		global_data->WriteData(serializer.blob.data.get(), serializer.blob.size);
	}

	return std::move(global_data);
}

static void WriteCSVChunkInternal(ClientContext &context, FunctionData &bind_data, DataChunk &cast_chunk,
                                  BufferedSerializer &writer, DataChunk &input, bool &written_anything) {
	auto &csv_data = bind_data.Cast<WriteCSVData>();
	auto &options = csv_data.options;

	// first cast the columns of the chunk to varchar
	cast_chunk.Reset();
	cast_chunk.SetCardinality(input);
	for (idx_t col_idx = 0; col_idx < input.ColumnCount(); col_idx++) {
		if (csv_data.sql_types[col_idx].id() == LogicalTypeId::VARCHAR) {
			// VARCHAR, just reinterpret (cannot reference, because LogicalTypeId::VARCHAR is used by the JSON type too)
			cast_chunk.data[col_idx].Reinterpret(input.data[col_idx]);
		} else if (options.has_format[LogicalTypeId::DATE] && csv_data.sql_types[col_idx].id() == LogicalTypeId::DATE) {
			// use the date format to cast the chunk
			csv_data.options.write_date_format[LogicalTypeId::DATE].ConvertDateVector(
			    input.data[col_idx], cast_chunk.data[col_idx], input.size());
		} else if (options.has_format[LogicalTypeId::TIMESTAMP] &&
		           (csv_data.sql_types[col_idx].id() == LogicalTypeId::TIMESTAMP ||
		            csv_data.sql_types[col_idx].id() == LogicalTypeId::TIMESTAMP_TZ)) {
			// use the timestamp format to cast the chunk
			csv_data.options.write_date_format[LogicalTypeId::TIMESTAMP].ConvertTimestampVector(
			    input.data[col_idx], cast_chunk.data[col_idx], input.size());
		} else {
			// non varchar column, perform the cast
			VectorOperations::Cast(context, input.data[col_idx], cast_chunk.data[col_idx], input.size());
		}
	}

	cast_chunk.Flatten();
	// now loop over the vectors and output the values
	for (idx_t row_idx = 0; row_idx < cast_chunk.size(); row_idx++) {
		if (row_idx == 0 && !written_anything) {
			written_anything = true;
		} else {
			writer.WriteBufferData(csv_data.newline);
		}
		// write values
		for (idx_t col_idx = 0; col_idx < cast_chunk.ColumnCount(); col_idx++) {
			if (col_idx != 0) {
				writer.WriteBufferData(options.delimiter);
			}
			if (FlatVector::IsNull(cast_chunk.data[col_idx], row_idx)) {
				// write null value
				writer.WriteBufferData(options.null_str);
				continue;
			}

			// non-null value, fetch the string value from the cast chunk
			auto str_data = FlatVector::GetData<string_t>(cast_chunk.data[col_idx]);
			// FIXME: we could gain some performance here by checking for certain types if they ever require quotes
			// (e.g. integers only require quotes if the delimiter is a number, decimals only require quotes if the
			// delimiter is a number or "." character)
			WriteQuotedString(writer, csv_data, str_data[row_idx].GetData(), str_data[row_idx].GetSize(),
			                  csv_data.options.force_quote[col_idx]);
		}
	}
}

static void WriteCSVSink(ExecutionContext &context, FunctionData &bind_data, GlobalFunctionData &gstate,
                         LocalFunctionData &lstate, DataChunk &input) {
	auto &csv_data = bind_data.Cast<WriteCSVData>();
	auto &local_data = lstate.Cast<LocalWriteCSVData>();
	auto &global_state = gstate.Cast<GlobalWriteCSVData>();

	// write data into the local buffer
	WriteCSVChunkInternal(context.client, bind_data, local_data.cast_chunk, local_data.serializer, input,
	                      local_data.written_anything);

	// check if we should flush what we have currently written
	auto &writer = local_data.serializer;
	if (writer.blob.size >= csv_data.flush_size) {
		global_state.WriteRows(writer.blob.data.get(), writer.blob.size, csv_data.newline);
		writer.Reset();
		local_data.written_anything = false;
	}
}

//===--------------------------------------------------------------------===//
// Combine
//===--------------------------------------------------------------------===//
static void WriteCSVCombine(ExecutionContext &context, FunctionData &bind_data, GlobalFunctionData &gstate,
                            LocalFunctionData &lstate) {
	auto &local_data = lstate.Cast<LocalWriteCSVData>();
	auto &global_state = gstate.Cast<GlobalWriteCSVData>();
	auto &csv_data = bind_data.Cast<WriteCSVData>();
	auto &writer = local_data.serializer;
	// flush the local writer
	if (local_data.written_anything) {
		global_state.WriteRows(writer.blob.data.get(), writer.blob.size, csv_data.newline);
		writer.Reset();
	}
}

//===--------------------------------------------------------------------===//
// Finalize
//===--------------------------------------------------------------------===//
void WriteCSVFinalize(ClientContext &context, FunctionData &bind_data, GlobalFunctionData &gstate) {
	auto &global_state = gstate.Cast<GlobalWriteCSVData>();
	auto &csv_data = bind_data.Cast<WriteCSVData>();
	auto &options = csv_data.options;

	BufferedSerializer serializer;
	if (!options.suffix.empty()) {
		serializer.WriteBufferData(options.suffix);
	} else if (global_state.written_anything) {
		serializer.WriteBufferData(csv_data.newline);
	}
	global_state.WriteData(serializer.blob.data.get(), serializer.blob.size);

	global_state.handle->Close();
	global_state.handle.reset();
}

//===--------------------------------------------------------------------===//
// Execution Mode
//===--------------------------------------------------------------------===//
CopyFunctionExecutionMode WriteCSVExecutionMode(bool preserve_insertion_order, bool supports_batch_index) {
	if (!preserve_insertion_order) {
		return CopyFunctionExecutionMode::PARALLEL_COPY_TO_FILE;
	}
	if (supports_batch_index) {
		return CopyFunctionExecutionMode::BATCH_COPY_TO_FILE;
	}
	return CopyFunctionExecutionMode::REGULAR_COPY_TO_FILE;
}
//===--------------------------------------------------------------------===//
// Prepare Batch
//===--------------------------------------------------------------------===//
struct WriteCSVBatchData : public PreparedBatchData {
	//! The thread-local buffer to write data into
	BufferedSerializer serializer;
};

unique_ptr<PreparedBatchData> WriteCSVPrepareBatch(ClientContext &context, FunctionData &bind_data,
                                                   GlobalFunctionData &gstate,
                                                   unique_ptr<ColumnDataCollection> collection) {
	auto &csv_data = bind_data.Cast<WriteCSVData>();

	// create the cast chunk with VARCHAR types
	vector<LogicalType> types;
	types.resize(csv_data.options.name_list.size(), LogicalType::VARCHAR);
	DataChunk cast_chunk;
	cast_chunk.Initialize(Allocator::Get(context), types);

	// write CSV chunks to the batch data
	bool written_anything = false;
	auto batch = make_uniq<WriteCSVBatchData>();
	for (auto &chunk : collection->Chunks()) {
		WriteCSVChunkInternal(context, bind_data, cast_chunk, batch->serializer, chunk, written_anything);
	}
	return std::move(batch);
}

//===--------------------------------------------------------------------===//
// Flush Batch
//===--------------------------------------------------------------------===//
void WriteCSVFlushBatch(ClientContext &context, FunctionData &bind_data, GlobalFunctionData &gstate,
                        PreparedBatchData &batch) {
	auto &csv_batch = batch.Cast<WriteCSVBatchData>();
	auto &global_state = gstate.Cast<GlobalWriteCSVData>();
	auto &csv_data = bind_data.Cast<WriteCSVData>();
	auto &writer = csv_batch.serializer;
	global_state.WriteRows(writer.blob.data.get(), writer.blob.size, csv_data.newline);
	writer.Reset();
}

void CSVCopyFunction::RegisterFunction(BuiltinFunctions &set) {
	CopyFunction info("csv");
	info.copy_to_bind = WriteCSVBind;
	info.copy_to_initialize_local = WriteCSVInitializeLocal;
	info.copy_to_initialize_global = WriteCSVInitializeGlobal;
	info.copy_to_sink = WriteCSVSink;
	info.copy_to_combine = WriteCSVCombine;
	info.copy_to_finalize = WriteCSVFinalize;
	info.execution_mode = WriteCSVExecutionMode;
	info.prepare_batch = WriteCSVPrepareBatch;
	info.flush_batch = WriteCSVFlushBatch;

	info.copy_from_bind = ReadCSVBind;
	info.copy_from_function = ReadCSVTableFunction::GetFunction();

	info.extension = "csv";

	set.AddFunction(info);
}

} // namespace duckdb







namespace duckdb {

struct GlobFunctionBindData : public TableFunctionData {
	vector<string> files;
};

static unique_ptr<FunctionData> GlobFunctionBind(ClientContext &context, TableFunctionBindInput &input,
                                                 vector<LogicalType> &return_types, vector<string> &names) {
	auto result = make_uniq<GlobFunctionBindData>();
	result->files = MultiFileReader::GetFileList(context, input.inputs[0], "Globbing", FileGlobOptions::ALLOW_EMPTY);
	return_types.emplace_back(LogicalType::VARCHAR);
	names.emplace_back("file");
	return std::move(result);
}

struct GlobFunctionState : public GlobalTableFunctionState {
	GlobFunctionState() : current_idx(0) {
	}

	idx_t current_idx;
};

static unique_ptr<GlobalTableFunctionState> GlobFunctionInit(ClientContext &context, TableFunctionInitInput &input) {
	return make_uniq<GlobFunctionState>();
}

static void GlobFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind_data = data_p.bind_data->Cast<GlobFunctionBindData>();
	auto &state = (GlobFunctionState &)*data_p.global_state;

	idx_t count = 0;
	idx_t next_idx = MinValue<idx_t>(state.current_idx + STANDARD_VECTOR_SIZE, bind_data.files.size());
	for (; state.current_idx < next_idx; state.current_idx++) {
		output.data[0].SetValue(count, bind_data.files[state.current_idx]);
		count++;
	}
	output.SetCardinality(count);
}

void GlobTableFunction::RegisterFunction(BuiltinFunctions &set) {
	TableFunction glob_function("glob", {LogicalType::VARCHAR}, GlobFunction, GlobFunctionBind, GlobFunctionInit);
	set.AddFunction(MultiFileReader::CreateFunctionSet(glob_function));
}

} // namespace duckdb










namespace duckdb {

struct PragmaDetailedProfilingOutputOperatorData : public GlobalTableFunctionState {
	explicit PragmaDetailedProfilingOutputOperatorData() : initialized(false) {
	}

	ColumnDataScanState scan_state;
	bool initialized;
};

struct PragmaDetailedProfilingOutputData : public TableFunctionData {
	explicit PragmaDetailedProfilingOutputData(vector<LogicalType> &types) : types(types) {
	}
	unique_ptr<ColumnDataCollection> collection;
	vector<LogicalType> types;
};

static unique_ptr<FunctionData> PragmaDetailedProfilingOutputBind(ClientContext &context, TableFunctionBindInput &input,
                                                                  vector<LogicalType> &return_types,
                                                                  vector<string> &names) {
	names.emplace_back("OPERATOR_ID");
	return_types.emplace_back(LogicalType::INTEGER);

	names.emplace_back("ANNOTATION");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("ID");
	return_types.emplace_back(LogicalType::INTEGER);

	names.emplace_back("NAME");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("TIME");
	return_types.emplace_back(LogicalType::DOUBLE);

	names.emplace_back("CYCLES_PER_TUPLE");
	return_types.emplace_back(LogicalType::DOUBLE);

	names.emplace_back("SAMPLE_SIZE");
	return_types.emplace_back(LogicalType::INTEGER);

	names.emplace_back("INPUT_SIZE");
	return_types.emplace_back(LogicalType::INTEGER);

	names.emplace_back("EXTRA_INFO");
	return_types.emplace_back(LogicalType::VARCHAR);

	return make_uniq<PragmaDetailedProfilingOutputData>(return_types);
}

unique_ptr<GlobalTableFunctionState> PragmaDetailedProfilingOutputInit(ClientContext &context,
                                                                       TableFunctionInitInput &input) {
	return make_uniq<PragmaDetailedProfilingOutputOperatorData>();
}

// Insert a row into the given datachunk
static void SetValue(DataChunk &output, int index, int op_id, string annotation, int id, string name, double time,
                     int sample_counter, int tuple_counter, string extra_info) {
	output.SetValue(0, index, op_id);
	output.SetValue(1, index, std::move(annotation));
	output.SetValue(2, index, id);
	output.SetValue(3, index, std::move(name));
#if defined(RDTSC)
	output.SetValue(4, index, Value(nullptr));
	output.SetValue(5, index, time);
#else
	output.SetValue(4, index, time);
	output.SetValue(5, index, Value(nullptr));

#endif
	output.SetValue(6, index, sample_counter);
	output.SetValue(7, index, tuple_counter);
	output.SetValue(8, index, std::move(extra_info));
}

static void ExtractFunctions(ColumnDataCollection &collection, ExpressionInfo &info, DataChunk &chunk, int op_id,
                             int &fun_id) {
	if (info.hasfunction) {
		D_ASSERT(info.sample_tuples_count != 0);
		SetValue(chunk, chunk.size(), op_id, "Function", fun_id++, info.function_name,
		         int(info.function_time) / double(info.sample_tuples_count), info.sample_tuples_count,
		         info.tuples_count, "");

		chunk.SetCardinality(chunk.size() + 1);
		if (chunk.size() == STANDARD_VECTOR_SIZE) {
			collection.Append(chunk);
			chunk.Reset();
		}
	}
	if (info.children.empty()) {
		return;
	}
	// extract the children of this node
	for (auto &child : info.children) {
		ExtractFunctions(collection, *child, chunk, op_id, fun_id);
	}
}

static void PragmaDetailedProfilingOutputFunction(ClientContext &context, TableFunctionInput &data_p,
                                                  DataChunk &output) {
	auto &state = data_p.global_state->Cast<PragmaDetailedProfilingOutputOperatorData>();
	auto &data = data_p.bind_data->CastNoConst<PragmaDetailedProfilingOutputData>();

	if (!state.initialized) {
		// create a ColumnDataCollection
		auto collection = make_uniq<ColumnDataCollection>(context, data.types);

		// create a chunk
		DataChunk chunk;
		chunk.Initialize(context, data.types);

		// Initialize ids
		int operator_counter = 1;
		int function_counter = 1;
		int expression_counter = 1;
		auto &client_data = ClientData::Get(context);
		if (client_data.query_profiler_history->GetPrevProfilers().empty()) {
			return;
		}
		// For each Operator
		auto &tree_map = client_data.query_profiler_history->GetPrevProfilers().back().second->GetTreeMap();
		for (auto op : tree_map) {
			// For each Expression Executor
			for (auto &expr_executor : op.second.get().info.executors_info) {
				// For each Expression tree
				if (!expr_executor) {
					continue;
				}
				for (auto &expr_timer : expr_executor->roots) {
					D_ASSERT(expr_timer->sample_tuples_count != 0);
					SetValue(chunk, chunk.size(), operator_counter, "ExpressionRoot", expression_counter++,
					         // Sometimes, cycle counter is not accurate, too big or too small. return 0 for
					         // those cases
					         expr_timer->name, int(expr_timer->time) / double(expr_timer->sample_tuples_count),
					         expr_timer->sample_tuples_count, expr_timer->tuples_count, expr_timer->extra_info);
					// Increment cardinality
					chunk.SetCardinality(chunk.size() + 1);
					// Check whether data chunk is full or not
					if (chunk.size() == STANDARD_VECTOR_SIZE) {
						collection->Append(chunk);
						chunk.Reset();
					}
					// Extract all functions inside the tree
					ExtractFunctions(*collection, *expr_timer->root, chunk, operator_counter, function_counter);
				}
			}
			operator_counter++;
		}
		collection->Append(chunk);
		data.collection = std::move(collection);
		data.collection->InitializeScan(state.scan_state);
		state.initialized = true;
	}

	data.collection->Scan(state.scan_state, output);
}

void PragmaDetailedProfilingOutput::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(TableFunction("pragma_detailed_profiling_output", {}, PragmaDetailedProfilingOutputFunction,
	                              PragmaDetailedProfilingOutputBind, PragmaDetailedProfilingOutputInit));
}

} // namespace duckdb










namespace duckdb {

struct PragmaLastProfilingOutputOperatorData : public GlobalTableFunctionState {
	PragmaLastProfilingOutputOperatorData() : initialized(false) {
	}

	ColumnDataScanState scan_state;
	bool initialized;
};

struct PragmaLastProfilingOutputData : public TableFunctionData {
	explicit PragmaLastProfilingOutputData(vector<LogicalType> &types) : types(types) {
	}
	unique_ptr<ColumnDataCollection> collection;
	vector<LogicalType> types;
};

static unique_ptr<FunctionData> PragmaLastProfilingOutputBind(ClientContext &context, TableFunctionBindInput &input,
                                                              vector<LogicalType> &return_types,
                                                              vector<string> &names) {
	names.emplace_back("OPERATOR_ID");
	return_types.emplace_back(LogicalType::INTEGER);

	names.emplace_back("NAME");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("TIME");
	return_types.emplace_back(LogicalType::DOUBLE);

	names.emplace_back("CARDINALITY");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("DESCRIPTION");
	return_types.emplace_back(LogicalType::VARCHAR);

	return make_uniq<PragmaLastProfilingOutputData>(return_types);
}

static void SetValue(DataChunk &output, int index, int op_id, string name, double time, int64_t car,
                     string description) {
	output.SetValue(0, index, op_id);
	output.SetValue(1, index, std::move(name));
	output.SetValue(2, index, time);
	output.SetValue(3, index, car);
	output.SetValue(4, index, std::move(description));
}

unique_ptr<GlobalTableFunctionState> PragmaLastProfilingOutputInit(ClientContext &context,
                                                                   TableFunctionInitInput &input) {
	return make_uniq<PragmaLastProfilingOutputOperatorData>();
}

static void PragmaLastProfilingOutputFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &state = data_p.global_state->Cast<PragmaLastProfilingOutputOperatorData>();
	auto &data = (PragmaLastProfilingOutputData &)*data_p.bind_data;
	if (!state.initialized) {
		// create a ColumnDataCollection
		auto collection = make_uniq<ColumnDataCollection>(context, data.types);

		DataChunk chunk;
		chunk.Initialize(context, data.types);
		int operator_counter = 1;
		auto &client_data = ClientData::Get(context);
		if (!client_data.query_profiler_history->GetPrevProfilers().empty()) {
			auto &tree_map = client_data.query_profiler_history->GetPrevProfilers().back().second->GetTreeMap();
			for (auto op : tree_map) {
				auto &tree_info = op.second.get();
				SetValue(chunk, chunk.size(), operator_counter++, tree_info.name, tree_info.info.time,
				         tree_info.info.elements, " ");
				chunk.SetCardinality(chunk.size() + 1);
				if (chunk.size() == STANDARD_VECTOR_SIZE) {
					collection->Append(chunk);
					chunk.Reset();
				}
			}
		}
		collection->Append(chunk);
		data.collection = std::move(collection);
		data.collection->InitializeScan(state.scan_state);
		state.initialized = true;
	}

	data.collection->Scan(state.scan_state, output);
}

void PragmaLastProfilingOutput::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(TableFunction("pragma_last_profiling_output", {}, PragmaLastProfilingOutputFunction,
	                              PragmaLastProfilingOutputBind, PragmaLastProfilingOutputInit));
}

} // namespace duckdb








namespace duckdb {

//===--------------------------------------------------------------------===//
// Range (integers)
//===--------------------------------------------------------------------===//
struct RangeFunctionBindData : public TableFunctionData {
	hugeint_t start;
	hugeint_t end;
	hugeint_t increment;

public:
	bool Equals(const FunctionData &other_p) const override {
		auto &other = (const RangeFunctionBindData &)other_p;
		return other.start == start && other.end == end && other.increment == increment;
	}
};

template <bool GENERATE_SERIES>
static void GenerateRangeParameters(const vector<Value> &inputs, RangeFunctionBindData &result) {
	for (auto &input : inputs) {
		if (input.IsNull()) {
			result.start = GENERATE_SERIES ? 1 : 0;
			result.end = 0;
			result.increment = 1;
			return;
		}
	}
	if (inputs.size() < 2) {
		// single argument: only the end is specified
		result.start = 0;
		result.end = inputs[0].GetValue<int64_t>();
	} else {
		// two arguments: first two arguments are start and end
		result.start = inputs[0].GetValue<int64_t>();
		result.end = inputs[1].GetValue<int64_t>();
	}
	if (inputs.size() < 3) {
		result.increment = 1;
	} else {
		result.increment = inputs[2].GetValue<int64_t>();
	}
	if (result.increment == 0) {
		throw BinderException("interval cannot be 0!");
	}
	if (result.start > result.end && result.increment > 0) {
		throw BinderException("start is bigger than end, but increment is positive: cannot generate infinite series");
	} else if (result.start < result.end && result.increment < 0) {
		throw BinderException("start is smaller than end, but increment is negative: cannot generate infinite series");
	}
}

template <bool GENERATE_SERIES>
static unique_ptr<FunctionData> RangeFunctionBind(ClientContext &context, TableFunctionBindInput &input,
                                                  vector<LogicalType> &return_types, vector<string> &names) {
	auto result = make_uniq<RangeFunctionBindData>();
	auto &inputs = input.inputs;
	GenerateRangeParameters<GENERATE_SERIES>(inputs, *result);

	return_types.emplace_back(LogicalType::BIGINT);
	if (GENERATE_SERIES) {
		// generate_series has inclusive bounds on the RHS
		if (result->increment < 0) {
			result->end = result->end - 1;
		} else {
			result->end = result->end + 1;
		}
		names.emplace_back("generate_series");
	} else {
		names.emplace_back("range");
	}
	return std::move(result);
}

struct RangeFunctionState : public GlobalTableFunctionState {
	RangeFunctionState() : current_idx(0) {
	}

	int64_t current_idx;
};

static unique_ptr<GlobalTableFunctionState> RangeFunctionInit(ClientContext &context, TableFunctionInitInput &input) {
	return make_uniq<RangeFunctionState>();
}

static void RangeFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind_data = data_p.bind_data->Cast<RangeFunctionBindData>();
	auto &state = (RangeFunctionState &)*data_p.global_state;

	auto increment = bind_data.increment;
	auto end = bind_data.end;
	hugeint_t current_value = bind_data.start + increment * state.current_idx;
	int64_t current_value_i64;
	if (!Hugeint::TryCast<int64_t>(current_value, current_value_i64)) {
		return;
	}
	int64_t offset = increment < 0 ? 1 : -1;
	idx_t remaining = MinValue<idx_t>(Hugeint::Cast<idx_t>((end - current_value + (increment + offset)) / increment),
	                                  STANDARD_VECTOR_SIZE);
	// set the result vector as a sequence vector
	output.data[0].Sequence(current_value_i64, Hugeint::Cast<int64_t>(increment), remaining);
	// increment the index pointer by the remaining count
	state.current_idx += remaining;
	output.SetCardinality(remaining);
}

unique_ptr<NodeStatistics> RangeCardinality(ClientContext &context, const FunctionData *bind_data_p) {
	auto &bind_data = bind_data_p->Cast<RangeFunctionBindData>();
	idx_t cardinality = Hugeint::Cast<idx_t>((bind_data.end - bind_data.start) / bind_data.increment);
	return make_uniq<NodeStatistics>(cardinality, cardinality);
}

//===--------------------------------------------------------------------===//
// Range (timestamp)
//===--------------------------------------------------------------------===//
struct RangeDateTimeBindData : public TableFunctionData {
	timestamp_t start;
	timestamp_t end;
	interval_t increment;
	bool inclusive_bound;
	bool greater_than_check;

public:
	bool Equals(const FunctionData &other_p) const override {
		auto &other = (const RangeDateTimeBindData &)other_p;
		return other.start == start && other.end == end && other.increment == increment &&
		       other.inclusive_bound == inclusive_bound && other.greater_than_check == greater_than_check;
	}

	bool Finished(timestamp_t current_value) const {
		if (greater_than_check) {
			if (inclusive_bound) {
				return current_value > end;
			} else {
				return current_value >= end;
			}
		} else {
			if (inclusive_bound) {
				return current_value < end;
			} else {
				return current_value <= end;
			}
		}
	}
};

template <bool GENERATE_SERIES>
static unique_ptr<FunctionData> RangeDateTimeBind(ClientContext &context, TableFunctionBindInput &input,
                                                  vector<LogicalType> &return_types, vector<string> &names) {
	auto result = make_uniq<RangeDateTimeBindData>();
	auto &inputs = input.inputs;
	D_ASSERT(inputs.size() == 3);
	result->start = inputs[0].GetValue<timestamp_t>();
	result->end = inputs[1].GetValue<timestamp_t>();
	result->increment = inputs[2].GetValue<interval_t>();

	// Infinities either cause errors or infinite loops, so just ban them
	if (!Timestamp::IsFinite(result->start) || !Timestamp::IsFinite(result->end)) {
		throw BinderException("RANGE with infinite bounds is not supported");
	}

	if (result->increment.months == 0 && result->increment.days == 0 && result->increment.micros == 0) {
		throw BinderException("interval cannot be 0!");
	}
	// all elements should point in the same direction
	if (result->increment.months > 0 || result->increment.days > 0 || result->increment.micros > 0) {
		if (result->increment.months < 0 || result->increment.days < 0 || result->increment.micros < 0) {
			throw BinderException("RANGE with composite interval that has mixed signs is not supported");
		}
		result->greater_than_check = true;
		if (result->start > result->end) {
			throw BinderException(
			    "start is bigger than end, but increment is positive: cannot generate infinite series");
		}
	} else {
		result->greater_than_check = false;
		if (result->start < result->end) {
			throw BinderException(
			    "start is smaller than end, but increment is negative: cannot generate infinite series");
		}
	}
	return_types.push_back(inputs[0].type());
	if (GENERATE_SERIES) {
		// generate_series has inclusive bounds on the RHS
		result->inclusive_bound = true;
		names.emplace_back("generate_series");
	} else {
		result->inclusive_bound = false;
		names.emplace_back("range");
	}
	return std::move(result);
}

struct RangeDateTimeState : public GlobalTableFunctionState {
	explicit RangeDateTimeState(timestamp_t start_p) : current_state(start_p) {
	}

	timestamp_t current_state;
	bool finished = false;
};

static unique_ptr<GlobalTableFunctionState> RangeDateTimeInit(ClientContext &context, TableFunctionInitInput &input) {
	auto &bind_data = input.bind_data->Cast<RangeDateTimeBindData>();
	return make_uniq<RangeDateTimeState>(bind_data.start);
}

static void RangeDateTimeFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind_data = data_p.bind_data->Cast<RangeDateTimeBindData>();
	auto &state = (RangeDateTimeState &)*data_p.global_state;
	if (state.finished) {
		return;
	}

	idx_t size = 0;
	auto data = FlatVector::GetData<timestamp_t>(output.data[0]);
	while (true) {
		data[size++] = state.current_state;
		state.current_state =
		    AddOperator::Operation<timestamp_t, interval_t, timestamp_t>(state.current_state, bind_data.increment);
		if (bind_data.Finished(state.current_state)) {
			state.finished = true;
			break;
		}
		if (size >= STANDARD_VECTOR_SIZE) {
			break;
		}
	}
	output.SetCardinality(size);
}

void RangeTableFunction::RegisterFunction(BuiltinFunctions &set) {
	TableFunctionSet range("range");

	TableFunction range_function({LogicalType::BIGINT}, RangeFunction, RangeFunctionBind<false>, RangeFunctionInit);
	range_function.cardinality = RangeCardinality;

	// single argument range: (end) - implicit start = 0 and increment = 1
	range.AddFunction(range_function);
	// two arguments range: (start, end) - implicit increment = 1
	range_function.arguments = {LogicalType::BIGINT, LogicalType::BIGINT};
	range.AddFunction(range_function);
	// three arguments range: (start, end, increment)
	range_function.arguments = {LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT};
	range.AddFunction(range_function);
	range.AddFunction(TableFunction({LogicalType::TIMESTAMP, LogicalType::TIMESTAMP, LogicalType::INTERVAL},
	                                RangeDateTimeFunction, RangeDateTimeBind<false>, RangeDateTimeInit));
	set.AddFunction(range);
	// generate_series: similar to range, but inclusive instead of exclusive bounds on the RHS
	TableFunctionSet generate_series("generate_series");
	range_function.bind = RangeFunctionBind<true>;
	range_function.arguments = {LogicalType::BIGINT};
	generate_series.AddFunction(range_function);
	range_function.arguments = {LogicalType::BIGINT, LogicalType::BIGINT};
	generate_series.AddFunction(range_function);
	range_function.arguments = {LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT};
	generate_series.AddFunction(range_function);
	generate_series.AddFunction(TableFunction({LogicalType::TIMESTAMP, LogicalType::TIMESTAMP, LogicalType::INTERVAL},
	                                          RangeDateTimeFunction, RangeDateTimeBind<true>, RangeDateTimeInit));
	set.AddFunction(generate_series);
}

void BuiltinFunctions::RegisterTableFunctions() {
	CheckpointFunction::RegisterFunction(*this);
	GlobTableFunction::RegisterFunction(*this);
	RangeTableFunction::RegisterFunction(*this);
	RepeatTableFunction::RegisterFunction(*this);
	SummaryTableFunction::RegisterFunction(*this);
	UnnestTableFunction::RegisterFunction(*this);
	RepeatRowTableFunction::RegisterFunction(*this);
}

} // namespace duckdb
















#include <limits>

namespace duckdb {

unique_ptr<CSVFileHandle> ReadCSV::OpenCSV(const string &file_path, FileCompressionType compression,
                                           ClientContext &context) {
	auto &fs = FileSystem::GetFileSystem(context);
	auto &allocator = BufferAllocator::Get(context);
	return CSVFileHandle::OpenFile(fs, allocator, file_path, compression, false);
}

void ReadCSVData::FinalizeRead(ClientContext &context) {
	BaseCSVData::Finalize();
	// Here we identify if we can run this CSV file on parallel or not.
	bool null_or_empty = options.delimiter.empty() || options.escape.empty() || options.quote.empty() ||
	                     options.delimiter[0] == '\0' || options.escape[0] == '\0' || options.quote[0] == '\0';
	bool complex_options = options.delimiter.size() > 1 || options.escape.size() > 1 || options.quote.size() > 1;
	bool not_supported_options = options.null_padding;

	auto number_of_threads = TaskScheduler::GetScheduler(context).NumberOfThreads();
	if (options.parallel_mode != ParallelMode::PARALLEL && int64_t(files.size() * 2) >= number_of_threads) {
		single_threaded = true;
	}
	if (options.parallel_mode == ParallelMode::SINGLE_THREADED || null_or_empty || not_supported_options ||
	    complex_options || options.new_line == NewLineIdentifier::MIX) {
		// not supported for parallel CSV reading
		single_threaded = true;
	}
}

uint8_t GetCandidateSpecificity(const LogicalType &candidate_type) {
	//! Const ht with accepted auto_types and their weights in specificity
	const duckdb::unordered_map<uint8_t, uint8_t> auto_type_candidates_specificity {
	    {(uint8_t)LogicalTypeId::VARCHAR, 0},  {(uint8_t)LogicalTypeId::TIMESTAMP, 1},
	    {(uint8_t)LogicalTypeId::DATE, 2},     {(uint8_t)LogicalTypeId::TIME, 3},
	    {(uint8_t)LogicalTypeId::DOUBLE, 4},   {(uint8_t)LogicalTypeId::FLOAT, 5},
	    {(uint8_t)LogicalTypeId::BIGINT, 6},   {(uint8_t)LogicalTypeId::INTEGER, 7},
	    {(uint8_t)LogicalTypeId::SMALLINT, 8}, {(uint8_t)LogicalTypeId::TINYINT, 9},
	    {(uint8_t)LogicalTypeId::BOOLEAN, 10}, {(uint8_t)LogicalTypeId::SQLNULL, 11}};

	auto id = (uint8_t)candidate_type.id();
	auto it = auto_type_candidates_specificity.find(id);
	if (it == auto_type_candidates_specificity.end()) {
		throw BinderException("Auto Type Candidate of type %s is not accepted as a valid input",
		                      EnumUtil::ToString(candidate_type.id()));
	}
	return it->second;
}

static unique_ptr<FunctionData> ReadCSVBind(ClientContext &context, TableFunctionBindInput &input,
                                            vector<LogicalType> &return_types, vector<string> &names) {
	auto result = make_uniq<ReadCSVData>();
	auto &options = result->options;
	result->files = MultiFileReader::GetFileList(context, input.inputs[0], "CSV");

	bool explicitly_set_columns = false;
	for (auto &kv : input.named_parameters) {
		if (MultiFileReader::ParseOption(kv.first, kv.second, options.file_options)) {
			continue;
		}
		auto loption = StringUtil::Lower(kv.first);
		if (loption == "columns") {
			explicitly_set_columns = true;
			auto &child_type = kv.second.type();
			if (child_type.id() != LogicalTypeId::STRUCT) {
				throw BinderException("read_csv columns requires a struct as input");
			}
			auto &struct_children = StructValue::GetChildren(kv.second);
			D_ASSERT(StructType::GetChildCount(child_type) == struct_children.size());
			for (idx_t i = 0; i < struct_children.size(); i++) {
				auto &name = StructType::GetChildName(child_type, i);
				auto &val = struct_children[i];
				names.push_back(name);
				if (val.type().id() != LogicalTypeId::VARCHAR) {
					throw BinderException("read_csv requires a type specification as string");
				}
				return_types.emplace_back(TransformStringToLogicalType(StringValue::Get(val), context));
			}
			if (names.empty()) {
				throw BinderException("read_csv requires at least a single column as input!");
			}
		} else if (loption == "auto_type_candidates") {
			options.auto_type_candidates.clear();
			map<uint8_t, LogicalType> candidate_types;
			// We always have the extremes of Null and Varchar, so we can default to varchar if the
			// sniffer is not able to confidently detect that column type
			candidate_types[GetCandidateSpecificity(LogicalType::VARCHAR)] = LogicalType::VARCHAR;
			candidate_types[GetCandidateSpecificity(LogicalType::SQLNULL)] = LogicalType::SQLNULL;

			auto &child_type = kv.second.type();
			if (child_type.id() != LogicalTypeId::LIST) {
				throw BinderException("read_csv auto_types requires a list as input");
			}
			auto &list_children = ListValue::GetChildren(kv.second);
			if (list_children.empty()) {
				throw BinderException("auto_type_candidates requires at least one type");
			}
			for (auto &child : list_children) {
				if (child.type().id() != LogicalTypeId::VARCHAR) {
					throw BinderException("auto_type_candidates requires a type specification as string");
				}
				auto candidate_type = TransformStringToLogicalType(StringValue::Get(child), context);
				candidate_types[GetCandidateSpecificity(candidate_type)] = candidate_type;
			}
			for (auto &candidate_type : candidate_types) {
				options.auto_type_candidates.emplace_back(candidate_type.second);
			}
		} else if (loption == "column_names" || loption == "names") {
			if (!options.name_list.empty()) {
				throw BinderException("read_csv_auto column_names/names can only be supplied once");
			}
			if (kv.second.IsNull()) {
				throw BinderException("read_csv_auto %s cannot be NULL", kv.first);
			}
			auto &children = ListValue::GetChildren(kv.second);
			for (auto &child : children) {
				options.name_list.push_back(StringValue::Get(child));
			}
		} else if (loption == "column_types" || loption == "types" || loption == "dtypes") {
			auto &child_type = kv.second.type();
			if (child_type.id() != LogicalTypeId::STRUCT && child_type.id() != LogicalTypeId::LIST) {
				throw BinderException("read_csv_auto %s requires a struct or list as input", kv.first);
			}
			if (!options.sql_type_list.empty()) {
				throw BinderException("read_csv_auto column_types/types/dtypes can only be supplied once");
			}
			vector<string> sql_type_names;
			if (child_type.id() == LogicalTypeId::STRUCT) {
				auto &struct_children = StructValue::GetChildren(kv.second);
				D_ASSERT(StructType::GetChildCount(child_type) == struct_children.size());
				for (idx_t i = 0; i < struct_children.size(); i++) {
					auto &name = StructType::GetChildName(child_type, i);
					auto &val = struct_children[i];
					if (val.type().id() != LogicalTypeId::VARCHAR) {
						throw BinderException("read_csv_auto %s requires a type specification as string", kv.first);
					}
					sql_type_names.push_back(StringValue::Get(val));
					options.sql_types_per_column[name] = i;
				}
			} else {
				auto &list_child = ListType::GetChildType(child_type);
				if (list_child.id() != LogicalTypeId::VARCHAR) {
					throw BinderException("read_csv_auto %s requires a list of types (varchar) as input", kv.first);
				}
				auto &children = ListValue::GetChildren(kv.second);
				for (auto &child : children) {
					sql_type_names.push_back(StringValue::Get(child));
				}
			}
			options.sql_type_list.reserve(sql_type_names.size());
			for (auto &sql_type : sql_type_names) {
				auto def_type = TransformStringToLogicalType(sql_type);
				if (def_type.id() == LogicalTypeId::USER) {
					throw BinderException("Unrecognized type \"%s\" for read_csv_auto %s definition", sql_type,
					                      kv.first);
				}
				options.sql_type_list.push_back(std::move(def_type));
			}
		} else if (loption == "all_varchar") {
			options.all_varchar = BooleanValue::Get(kv.second);
		} else if (loption == "normalize_names") {
			options.normalize_names = BooleanValue::Get(kv.second);
		} else if (loption == "parallel") {
			options.parallel_mode =
			    BooleanValue::Get(kv.second) ? ParallelMode::PARALLEL : ParallelMode::SINGLE_THREADED;
		} else {
			options.SetReadOption(loption, kv.second, names);
		}
	}
	if (options.file_options.auto_detect_hive_partitioning) {
		options.file_options.hive_partitioning = MultiFileReaderOptions::AutoDetectHivePartitioning(result->files);
	}

	if (!options.auto_detect && return_types.empty()) {
		throw BinderException("read_csv requires columns to be specified through the 'columns' option. Use "
		                      "read_csv_auto or set read_csv(..., "
		                      "AUTO_DETECT=TRUE) to automatically guess columns.");
	}
	if (options.auto_detect) {
		options.file_path = result->files[0];
		auto initial_reader = make_uniq<BufferedCSVReader>(context, options);
		return_types.assign(initial_reader->return_types.begin(), initial_reader->return_types.end());
		if (names.empty()) {
			names.assign(initial_reader->names.begin(), initial_reader->names.end());
		} else {
			if (explicitly_set_columns) {
				// The user has influenced the names, can't assume they are valid anymore
				if (return_types.size() != names.size()) {
					throw BinderException("The amount of names specified (%d) and the observed amount of types (%d) in "
					                      "the file don't match",
					                      names.size(), return_types.size());
				}
			} else {
				D_ASSERT(return_types.size() == names.size());
			}
			initial_reader->names = names;
		}
		options = initial_reader->options;
		result->initial_reader = std::move(initial_reader);
	} else {
		D_ASSERT(return_types.size() == names.size());
	}
	result->csv_types = return_types;
	result->csv_names = names;

	if (options.file_options.union_by_name) {
		result->reader_bind =
		    MultiFileReader::BindUnionReader<BufferedCSVReader>(context, return_types, names, *result, options);
		if (result->union_readers.size() > 1) {
			result->column_info.emplace_back(result->csv_names, result->csv_types);
			for (idx_t i = 1; i < result->union_readers.size(); i++) {
				result->column_info.emplace_back(result->union_readers[i]->names,
				                                 result->union_readers[i]->return_types);
			}
		}
		if (!options.sql_types_per_column.empty()) {
			auto exception = BufferedCSVReader::ColumnTypesError(options.sql_types_per_column, names);
			if (!exception.empty()) {
				throw BinderException(exception);
			}
		}
	} else {
		result->reader_bind = MultiFileReader::BindOptions(options.file_options, result->files, return_types, names);
	}
	result->return_types = return_types;
	result->return_names = names;
	result->FinalizeRead(context);
	return std::move(result);
}

static unique_ptr<FunctionData> ReadCSVAutoBind(ClientContext &context, TableFunctionBindInput &input,
                                                vector<LogicalType> &return_types, vector<string> &names) {
	input.named_parameters["auto_detect"] = Value::BOOLEAN(true);
	return ReadCSVBind(context, input, return_types, names);
}

//===--------------------------------------------------------------------===//
// Parallel CSV Reader CSV Global State
//===--------------------------------------------------------------------===//

struct ParallelCSVGlobalState : public GlobalTableFunctionState {
public:
	ParallelCSVGlobalState(ClientContext &context, unique_ptr<CSVFileHandle> file_handle_p,
	                       const vector<string> &files_path_p, idx_t system_threads_p, idx_t buffer_size_p,
	                       idx_t rows_to_skip, bool force_parallelism_p, vector<column_t> column_ids_p, bool has_header)
	    : file_handle(std::move(file_handle_p)), system_threads(system_threads_p), buffer_size(buffer_size_p),
	      force_parallelism(force_parallelism_p), column_ids(std::move(column_ids_p)),
	      line_info(main_mutex, batch_to_tuple_end, tuple_start, tuple_end) {
		file_handle->DisableReset();
		current_file_path = files_path_p[0];
		line_info.lines_read[0] = rows_to_skip;
		if (has_header) {
			line_info.lines_read[0]++;
		}
		file_size = file_handle->FileSize();
		first_file_size = file_size;
		on_disk_file = file_handle->OnDiskFile();
		bytes_read = 0;
		if (buffer_size < file_size || file_size == 0) {
			bytes_per_local_state = buffer_size / ParallelCSVGlobalState::MaxThreads();
		} else {
			bytes_per_local_state = file_size / MaxThreads();
		}
		if (bytes_per_local_state == 0) {
			// In practice, I think this won't happen, it only happens because we are mocking up test scenarios
			// this boy needs to be at least one.
			bytes_per_local_state = 1;
		}
		for (idx_t i = 0; i < rows_to_skip; i++) {
			file_handle->ReadLine();
		}
		first_position = current_csv_position;
		current_buffer = make_shared<CSVBuffer>(context, buffer_size, *file_handle, current_csv_position, file_number);
		next_buffer = shared_ptr<CSVBuffer>(
		    current_buffer->Next(*file_handle, buffer_size, current_csv_position, file_number).release());
		running_threads = MaxThreads();

		// Initialize all the book-keeping variables
		auto file_count = files_path_p.size();
		line_info.current_batches.resize(file_count);
		tuple_start.resize(file_count);
		tuple_end.resize(file_count);
		tuple_end_to_batch.resize(file_count);
		batch_to_tuple_end.resize(file_count);
	}
	ParallelCSVGlobalState() : line_info(main_mutex, batch_to_tuple_end, tuple_start, tuple_end) {
		running_threads = MaxThreads();
	}

	~ParallelCSVGlobalState() override {
	}

	//! How many bytes were read up to this point
	atomic<idx_t> bytes_read;
	//! Size of current file
	idx_t file_size;

public:
	idx_t MaxThreads() const override;
	//! Updates the CSV reader with the next buffer to read. Returns false if no more buffers are available.
	bool Next(ClientContext &context, const ReadCSVData &bind_data, unique_ptr<ParallelCSVReader> &reader);
	//! Verify if the CSV File was read correctly
	void Verify();

	void UpdateVerification(VerificationPositions positions, idx_t file_number, idx_t batch_idx);

	void UpdateLinesRead(CSVBufferRead &buffer_read, idx_t file_idx);

	void IncrementThread();

	void DecrementThread();

	bool Finished();

	double GetProgress(const ReadCSVData &bind_data) const {
		idx_t total_files = bind_data.files.size();

		// get the progress WITHIN the current file
		double progress;
		if (file_size == 0) {
			progress = 1.0;
		} else {
			progress = double(bytes_read) / double(file_size);
		}
		// now get the total percentage of files read
		double percentage = double(file_index - 1) / total_files;
		percentage += (double(1) / double(total_files)) * progress;
		return percentage * 100;
	}

private:
	//! File Handle for current file
	unique_ptr<CSVFileHandle> file_handle;
	shared_ptr<CSVBuffer> current_buffer;
	shared_ptr<CSVBuffer> next_buffer;

	//! The index of the next file to read (i.e. current file + 1)
	idx_t file_index = 1;
	string current_file_path;

	//! Mutex to lock when getting next batch of bytes (Parallel Only)
	mutex main_mutex;
	//! Byte set from for last thread
	idx_t next_byte = 0;
	//! How many bytes we should execute per local state
	idx_t bytes_per_local_state;
	//! Size of first file
	idx_t first_file_size;
	//! Whether or not this is an on-disk file
	bool on_disk_file = true;
	//! Basically max number of threads in DuckDB
	idx_t system_threads;
	//! Size of the buffers
	idx_t buffer_size;
	//! Current batch index
	idx_t batch_index = 0;
	idx_t local_batch_index = 0;

	//! Forces parallelism for small CSV Files, should only be used for testing.
	bool force_parallelism = false;
	//! Current (Global) position of CSV
	idx_t current_csv_position = 0;
	//! First Position of First Buffer
	idx_t first_position = 0;
	//! Current File Number
	idx_t file_number = 0;
	idx_t max_tuple_end = 0;
	//! the vector stores positions where threads ended the last line they read in the CSV File, and the set stores
	//! positions where they started reading the first line.
	vector<vector<idx_t>> tuple_end;
	vector<set<idx_t>> tuple_start;
	//! Tuple end to batch
	vector<unordered_map<idx_t, idx_t>> tuple_end_to_batch;
	//! Batch to Tuple End
	vector<unordered_map<idx_t, idx_t>> batch_to_tuple_end;
	idx_t running_threads = 0;
	//! The column ids to read
	vector<column_t> column_ids;
	//! Line Info used in error messages
	LineInfo line_info;
};

idx_t ParallelCSVGlobalState::MaxThreads() const {
	if (force_parallelism || !on_disk_file) {
		return system_threads;
	}
	idx_t one_mb = 1000000; // We initialize max one thread per Mb
	idx_t threads_per_mb = first_file_size / one_mb + 1;
	if (threads_per_mb < system_threads || threads_per_mb == 1) {
		return threads_per_mb;
	}

	return system_threads;
}

void ParallelCSVGlobalState::IncrementThread() {
	lock_guard<mutex> parallel_lock(main_mutex);
	running_threads++;
}

void ParallelCSVGlobalState::DecrementThread() {
	lock_guard<mutex> parallel_lock(main_mutex);
	D_ASSERT(running_threads > 0);
	running_threads--;
}

bool ParallelCSVGlobalState::Finished() {
	lock_guard<mutex> parallel_lock(main_mutex);
	return running_threads == 0;
}

void ParallelCSVGlobalState::Verify() {
	// All threads are done, we run some magic sweet verification code
	if (running_threads == 0) {
		D_ASSERT(tuple_end.size() == tuple_start.size());
		for (idx_t i = 0; i < tuple_start.size(); i++) {
			auto &current_tuple_end = tuple_end[i];
			auto &current_tuple_start = tuple_start[i];
			// figure out max value of last_pos
			if (current_tuple_end.empty()) {
				return;
			}
			auto max_value = *max_element(std::begin(current_tuple_end), std::end(current_tuple_end));
			for (idx_t tpl_idx = 0; tpl_idx < current_tuple_end.size(); tpl_idx++) {
				auto last_pos = current_tuple_end[tpl_idx];
				auto first_pos = current_tuple_start.find(last_pos);
				if (first_pos == current_tuple_start.end()) {
					// this might be necessary due to carriage returns outside buffer scopes.
					first_pos = current_tuple_start.find(last_pos + 1);
				}
				if (first_pos == current_tuple_start.end() && last_pos != max_value) {
					auto batch_idx = tuple_end_to_batch[i][last_pos];
					auto problematic_line = line_info.GetLine(batch_idx);
					throw InvalidInputException(
					    "CSV File not supported for multithreading. This can be a problematic line in your CSV File or "
					    "that this CSV can't be read in Parallel. Please, inspect if the line %llu is correct. If so, "
					    "please run single-threaded CSV Reading by setting parallel=false in the read_csv call.",
					    problematic_line);
				}
			}
		}
	}
}

void LineInfo::Verify(idx_t file_idx, idx_t batch_idx, idx_t cur_first_pos) {
	auto &tuple_start_set = tuple_start[file_idx];
	auto &processed_batches = batch_to_tuple_end[file_idx];
	auto &tuple_end_vec = tuple_end[file_idx];
	bool has_error = false;
	idx_t problematic_line;
	if (batch_idx == 0 || tuple_start_set.empty()) {
		return;
	}
	for (idx_t cur_batch = 0; cur_batch < batch_idx - 1; cur_batch++) {
		auto cur_end = tuple_end_vec[processed_batches[cur_batch]];
		auto first_pos = tuple_start_set.find(cur_end);
		if (first_pos == tuple_start_set.end()) {
			has_error = true;
			problematic_line = GetLine(cur_batch);
			break;
		}
	}
	if (!has_error) {
		auto cur_end = tuple_end_vec[processed_batches[batch_idx - 1]];
		if (cur_end != cur_first_pos) {
			has_error = true;
			problematic_line = GetLine(batch_idx);
		}
	}
	if (has_error) {
		throw InvalidInputException(
		    "CSV File not supported for multithreading. This can be a problematic line in your CSV File or "
		    "that this CSV can't be read in Parallel. Please, inspect if the line %llu is correct. If so, "
		    "please run single-threaded CSV Reading by setting parallel=false in the read_csv call.",
		    problematic_line);
	}
}

bool ParallelCSVGlobalState::Next(ClientContext &context, const ReadCSVData &bind_data,
                                  unique_ptr<ParallelCSVReader> &reader) {
	lock_guard<mutex> parallel_lock(main_mutex);
	if (!current_buffer) {
		// This means we are done with the current file, we need to go to the next one (if exists).
		if (file_index < bind_data.files.size()) {
			current_file_path = bind_data.files[file_index++];
			file_handle = ReadCSV::OpenCSV(current_file_path, bind_data.options.compression, context);
			current_csv_position = 0;
			file_number++;
			local_batch_index = 0;
			current_buffer =
			    make_shared<CSVBuffer>(context, buffer_size, *file_handle, current_csv_position, file_number);
			next_buffer = shared_ptr<CSVBuffer>(
			    current_buffer->Next(*file_handle, buffer_size, current_csv_position, file_number).release());
		} else {
			// We are done scanning.
			reader.reset();
			return false;
		}
	}
	// set up the current buffer
	line_info.current_batches.back().insert(local_batch_index);
	auto result = make_uniq<CSVBufferRead>(current_buffer, next_buffer, next_byte, next_byte + bytes_per_local_state,
	                                       batch_index++, local_batch_index++, &line_info);
	// move the byte index of the CSV reader to the next buffer
	next_byte += bytes_per_local_state;
	if (next_byte >= current_buffer->GetBufferSize()) {
		// We replace the current buffer with the next buffer
		next_byte = 0;
		bytes_read += current_buffer->GetBufferSize();
		current_buffer = next_buffer;
		if (next_buffer) {
			// Next buffer gets the next-next buffer
			next_buffer = shared_ptr<CSVBuffer>(
			    next_buffer->Next(*file_handle, buffer_size, current_csv_position, file_number).release());
		}
	}
	if (!reader || reader->options.file_path != current_file_path) {
		// we either don't have a reader, or the reader was created for a different file
		// we need to create a new reader and instantiate it
		if (file_index > 0 && file_index <= bind_data.union_readers.size() && bind_data.union_readers[file_index - 1]) {
			// we are doing UNION BY NAME - fetch the options from the union reader for this file
			auto &union_reader = *bind_data.union_readers[file_index - 1];
			reader = make_uniq<ParallelCSVReader>(context, union_reader.options, std::move(result), first_position,
			                                      union_reader.GetTypes(), file_index - 1);
			reader->names = union_reader.GetNames();
		} else if (file_index <= bind_data.column_info.size()) {
			// Serialized Union By name
			reader = make_uniq<ParallelCSVReader>(context, bind_data.options, std::move(result), first_position,
			                                      bind_data.column_info[file_index - 1].types, file_index - 1);
			reader->names = bind_data.column_info[file_index - 1].names;
		} else {
			// regular file - use the standard options
			reader = make_uniq<ParallelCSVReader>(context, bind_data.options, std::move(result), first_position,
			                                      bind_data.csv_types, file_index - 1);
			reader->names = bind_data.csv_names;
		}
		reader->options.file_path = current_file_path;
		MultiFileReader::InitializeReader(*reader, bind_data.options.file_options, bind_data.reader_bind,
		                                  bind_data.return_types, bind_data.return_names, column_ids, nullptr,
		                                  bind_data.files.front());
	} else {
		// update the current reader
		reader->SetBufferRead(std::move(result));
	}
	return true;
}
void ParallelCSVGlobalState::UpdateVerification(VerificationPositions positions, idx_t file_number_p, idx_t batch_idx) {
	lock_guard<mutex> parallel_lock(main_mutex);
	if (positions.beginning_of_first_line < positions.end_of_last_line) {
		if (positions.end_of_last_line > max_tuple_end) {
			max_tuple_end = positions.end_of_last_line;
		}
		tuple_end_to_batch[file_number_p][positions.end_of_last_line] = batch_idx;
		batch_to_tuple_end[file_number_p][batch_idx] = tuple_end[file_number_p].size();
		tuple_start[file_number_p].insert(positions.beginning_of_first_line);
		tuple_end[file_number_p].push_back(positions.end_of_last_line);
	}
}

void ParallelCSVGlobalState::UpdateLinesRead(CSVBufferRead &buffer_read, idx_t file_idx) {
	auto batch_idx = buffer_read.local_batch_index;
	auto lines_read = buffer_read.lines_read;
	lock_guard<mutex> parallel_lock(main_mutex);
	line_info.current_batches[file_idx].erase(batch_idx);
	line_info.lines_read[batch_idx] += lines_read;
}

bool LineInfo::CanItGetLine(idx_t file_idx, idx_t batch_idx) {
	lock_guard<mutex> parallel_lock(main_mutex);
	if (current_batches.empty() || done) {
		return true;
	}
	if (file_idx >= current_batches.size() || current_batches[file_idx].empty()) {
		return true;
	}
	auto min_value = *current_batches[file_idx].begin();
	if (min_value >= batch_idx) {
		return true;
	}
	return false;
}

// Returns the 1-indexed line number
idx_t LineInfo::GetLine(idx_t batch_idx, idx_t line_error, idx_t file_idx, idx_t cur_start, bool verify) {
	unique_ptr<lock_guard<mutex>> parallel_lock;
	if (!verify) {
		parallel_lock = duckdb::make_uniq<lock_guard<mutex>>(main_mutex);
	}
	idx_t line_count = 0;
	if (done) {
		// line count is 0-indexed, but we want to return 1-indexed
		return first_line + 1;
	}
	for (idx_t i = 0; i <= batch_idx; i++) {
		if (lines_read.find(i) == lines_read.end() && i != batch_idx) {
			throw InternalException("Missing batch index on Parallel CSV Reader GetLine");
		}
		line_count += lines_read[i];
	}

	// before we are done, if this is not a call in Verify() we must check Verify up to this batch
	if (!verify) {
		Verify(file_idx, batch_idx, cur_start);
	}
	done = true;
	first_line = line_count + line_error;
	// line count is 0-indexed, but we want to return 1-indexed
	return first_line + 1;
}

static unique_ptr<GlobalTableFunctionState> ParallelCSVInitGlobal(ClientContext &context,
                                                                  TableFunctionInitInput &input) {
	auto &bind_data = input.bind_data->CastNoConst<ReadCSVData>();
	if (bind_data.files.empty()) {
		// This can happen when a filename based filter pushdown has eliminated all possible files for this scan.
		return make_uniq<ParallelCSVGlobalState>();
	}
	unique_ptr<CSVFileHandle> file_handle;

	bind_data.options.file_path = bind_data.files[0];

	if (bind_data.initial_reader) {
		file_handle = std::move(bind_data.initial_reader->file_handle);
		file_handle->Reset();
		file_handle->DisableReset();
		bind_data.initial_reader.reset();
	} else {
		file_handle = ReadCSV::OpenCSV(bind_data.options.file_path, bind_data.options.compression, context);
	}
	return make_uniq<ParallelCSVGlobalState>(
	    context, std::move(file_handle), bind_data.files, context.db->NumberOfThreads(), bind_data.options.buffer_size,
	    bind_data.options.skip_rows, ClientConfig::GetConfig(context).verify_parallelism, input.column_ids,
	    bind_data.options.header && bind_data.options.has_header);
}

//===--------------------------------------------------------------------===//
// Read CSV Local State
//===--------------------------------------------------------------------===//
struct ParallelCSVLocalState : public LocalTableFunctionState {
public:
	explicit ParallelCSVLocalState(unique_ptr<ParallelCSVReader> csv_reader_p) : csv_reader(std::move(csv_reader_p)) {
	}

	//! The CSV reader
	unique_ptr<ParallelCSVReader> csv_reader;
	CSVBufferRead previous_buffer;
	bool done = false;
};

unique_ptr<LocalTableFunctionState> ParallelReadCSVInitLocal(ExecutionContext &context, TableFunctionInitInput &input,
                                                             GlobalTableFunctionState *global_state_p) {
	auto &csv_data = input.bind_data->Cast<ReadCSVData>();
	auto &global_state = global_state_p->Cast<ParallelCSVGlobalState>();
	unique_ptr<ParallelCSVReader> csv_reader;
	auto has_next = global_state.Next(context.client, csv_data, csv_reader);
	if (!has_next) {
		global_state.DecrementThread();
		csv_reader.reset();
	}
	return make_uniq<ParallelCSVLocalState>(std::move(csv_reader));
}

static void ParallelReadCSVFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind_data = data_p.bind_data->Cast<ReadCSVData>();
	auto &csv_global_state = data_p.global_state->Cast<ParallelCSVGlobalState>();
	auto &csv_local_state = data_p.local_state->Cast<ParallelCSVLocalState>();

	if (!csv_local_state.csv_reader) {
		// no csv_reader was set, this can happen when a filename-based filter has filtered out all possible files
		return;
	}

	do {
		if (output.size() != 0) {
			MultiFileReader::FinalizeChunk(bind_data.reader_bind, csv_local_state.csv_reader->reader_data, output);
			break;
		}
		if (csv_local_state.csv_reader->finished) {
			auto verification_updates = csv_local_state.csv_reader->GetVerificationPositions();
			if (verification_updates.beginning_of_first_line != verification_updates.end_of_last_line) {
				csv_global_state.UpdateVerification(verification_updates,
				                                    csv_local_state.csv_reader->buffer->buffer->GetFileNumber(),
				                                    csv_local_state.csv_reader->buffer->local_batch_index);
			}
			csv_global_state.UpdateLinesRead(*csv_local_state.csv_reader->buffer, csv_local_state.csv_reader->file_idx);
			auto has_next = csv_global_state.Next(context, bind_data, csv_local_state.csv_reader);
			if (csv_local_state.csv_reader) {
				csv_local_state.csv_reader->linenr = 0;
			}

			if (!has_next) {
				csv_global_state.DecrementThread();
				break;
			}
		}
		csv_local_state.csv_reader->ParseCSV(output);

	} while (true);
	if (csv_global_state.Finished()) {
		csv_global_state.Verify();
	}
}

//===--------------------------------------------------------------------===//
// Single-Threaded CSV Reader
//===--------------------------------------------------------------------===//
struct SingleThreadedCSVState : public GlobalTableFunctionState {
	explicit SingleThreadedCSVState(idx_t total_files) : total_files(total_files), next_file(0), progress_in_files(0) {
	}

	mutex csv_lock;
	unique_ptr<BufferedCSVReader> initial_reader;
	//! The total number of files to read from
	idx_t total_files;
	//! The index of the next file to read (i.e. current file + 1)
	atomic<idx_t> next_file;
	//! How far along we are in reading the current set of open files
	//! This goes from [0...next_file] * 100
	atomic<idx_t> progress_in_files;
	//! The set of SQL types
	vector<LogicalType> csv_types;
	//! The set of SQL names to be read from the file
	vector<string> csv_names;
	//! The column ids to read
	vector<column_t> column_ids;

	idx_t MaxThreads() const override {
		return total_files;
	}

	double GetProgress(const ReadCSVData &bind_data) const {
		D_ASSERT(total_files == bind_data.files.size());
		D_ASSERT(progress_in_files <= total_files * 100);
		return (double(progress_in_files) / double(total_files));
	}

	unique_ptr<BufferedCSVReader> GetCSVReader(ClientContext &context, ReadCSVData &bind_data, idx_t &file_index,
	                                           idx_t &total_size) {
		auto reader = GetCSVReaderInternal(context, bind_data, file_index, total_size);
		if (reader) {
			reader->file_handle->DisableReset();
		}
		return reader;
	}

private:
	unique_ptr<BufferedCSVReader> GetCSVReaderInternal(ClientContext &context, ReadCSVData &bind_data,
	                                                   idx_t &file_index, idx_t &total_size) {
		BufferedCSVReaderOptions options;
		{
			lock_guard<mutex> l(csv_lock);
			if (initial_reader) {
				total_size = initial_reader->file_handle ? initial_reader->file_handle->FileSize() : 0;
				return std::move(initial_reader);
			}
			if (next_file >= total_files) {
				return nullptr;
			}
			options = bind_data.options;
			file_index = next_file;
			next_file++;
		}
		// reuse csv_readers was created during binding
		unique_ptr<BufferedCSVReader> result;
		if (file_index < bind_data.union_readers.size() && bind_data.union_readers[file_index]) {
			result = std::move(bind_data.union_readers[file_index]);
		} else {
			auto union_by_name = options.file_options.union_by_name;
			options.file_path = bind_data.files[file_index];
			result = make_uniq<BufferedCSVReader>(context, std::move(options), csv_types);
			if (!union_by_name) {
				result->names = csv_names;
			}
			MultiFileReader::InitializeReader(*result, bind_data.options.file_options, bind_data.reader_bind,
			                                  bind_data.return_types, bind_data.return_names, column_ids, nullptr,
			                                  bind_data.files.front());
		}
		total_size = result->file_handle->FileSize();
		return result;
	}
};

struct SingleThreadedCSVLocalState : public LocalTableFunctionState {
public:
	explicit SingleThreadedCSVLocalState() : bytes_read(0), total_size(0), current_progress(0), file_index(0) {
	}

	//! The CSV reader
	unique_ptr<BufferedCSVReader> csv_reader;
	//! The current amount of bytes read by this reader
	idx_t bytes_read;
	//! The total amount of bytes in the file
	idx_t total_size;
	//! The current progress from 0..100
	idx_t current_progress;
	//! The file index of this reader
	idx_t file_index;
};

static unique_ptr<GlobalTableFunctionState> SingleThreadedCSVInit(ClientContext &context,
                                                                  TableFunctionInitInput &input) {
	auto &bind_data = input.bind_data->CastNoConst<ReadCSVData>();
	auto result = make_uniq<SingleThreadedCSVState>(bind_data.files.size());
	if (bind_data.files.empty()) {
		// This can happen when a filename based filter pushdown has eliminated all possible files for this scan.
		return std::move(result);
	} else {
		bind_data.options.file_path = bind_data.files[0];
		if (bind_data.initial_reader) {
			// If this is a pipe and an initial reader already exists due to read_csv_auto
			// We must re-use it, since we can't restart the reader due for it being a pipe.
			result->initial_reader = std::move(bind_data.initial_reader);
		} else {
			result->initial_reader = make_uniq<BufferedCSVReader>(context, bind_data.options, bind_data.csv_types);
		}
		if (!bind_data.options.file_options.union_by_name) {
			result->initial_reader->names = bind_data.csv_names;
		}
		if (bind_data.options.auto_detect) {
			bind_data.options = result->initial_reader->options;
		}
	}
	MultiFileReader::InitializeReader(*result->initial_reader, bind_data.options.file_options, bind_data.reader_bind,
	                                  bind_data.return_types, bind_data.return_names, input.column_ids, input.filters,
	                                  bind_data.files.front());
	for (auto &reader : bind_data.union_readers) {
		if (!reader) {
			continue;
		}
		MultiFileReader::InitializeReader(*reader, bind_data.options.file_options, bind_data.reader_bind,
		                                  bind_data.return_types, bind_data.return_names, input.column_ids,
		                                  input.filters, bind_data.files.front());
	}
	result->column_ids = input.column_ids;

	if (!bind_data.options.file_options.union_by_name) {
		// if we are reading multiple files - run auto-detect only on the first file
		// UNLESS union by name is turned on - in that case we assume that different files have different schemas
		// as such, we need to re-run the auto detection on each file
		bind_data.options.auto_detect = false;
	}
	result->csv_types = bind_data.csv_types;
	result->csv_names = bind_data.csv_names;
	result->next_file = 1;
	return std::move(result);
}

unique_ptr<LocalTableFunctionState> SingleThreadedReadCSVInitLocal(ExecutionContext &context,
                                                                   TableFunctionInitInput &input,
                                                                   GlobalTableFunctionState *global_state_p) {
	auto &bind_data = (ReadCSVData &)*input.bind_data;
	auto &data = (SingleThreadedCSVState &)*global_state_p;
	auto result = make_uniq<SingleThreadedCSVLocalState>();
	result->csv_reader = data.GetCSVReader(context.client, bind_data, result->file_index, result->total_size);
	return std::move(result);
}

static void SingleThreadedCSVFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind_data = (ReadCSVData &)*data_p.bind_data;
	auto &data = (SingleThreadedCSVState &)*data_p.global_state;
	auto &lstate = data_p.local_state->Cast<SingleThreadedCSVLocalState>();
	if (!lstate.csv_reader) {
		// no csv_reader was set, this can happen when a filename-based filter has filtered out all possible files
		return;
	}

	do {
		lstate.csv_reader->ParseCSV(output);
		// update the number of bytes read
		D_ASSERT(lstate.bytes_read <= lstate.csv_reader->bytes_in_chunk);
		auto bytes_read = MinValue<idx_t>(lstate.total_size, lstate.csv_reader->bytes_in_chunk);
		auto current_progress = lstate.total_size == 0 ? 100 : 100 * bytes_read / lstate.total_size;
		if (current_progress > lstate.current_progress) {
			if (current_progress > 100) {
				throw InternalException("Progress should never exceed 100");
			}
			data.progress_in_files += current_progress - lstate.current_progress;
			lstate.current_progress = current_progress;
		}
		if (output.size() == 0) {
			// exhausted this file, but we might have more files we can read
			auto csv_reader = data.GetCSVReader(context, bind_data, lstate.file_index, lstate.total_size);
			// add any left-over progress for this file to the progress bar
			if (lstate.current_progress < 100) {
				data.progress_in_files += 100 - lstate.current_progress;
			}
			// reset the current progress
			lstate.current_progress = 0;
			lstate.bytes_read = 0;
			lstate.csv_reader = std::move(csv_reader);
			if (!lstate.csv_reader) {
				// no more files - we are done
				return;
			}
			lstate.bytes_read = 0;
		} else {
			MultiFileReader::FinalizeChunk(bind_data.reader_bind, lstate.csv_reader->reader_data, output);
			break;
		}
	} while (true);
}

//===--------------------------------------------------------------------===//
// Read CSV Functions
//===--------------------------------------------------------------------===//
static unique_ptr<GlobalTableFunctionState> ReadCSVInitGlobal(ClientContext &context, TableFunctionInitInput &input) {
	auto &bind_data = input.bind_data->Cast<ReadCSVData>();
	if (bind_data.single_threaded) {
		return SingleThreadedCSVInit(context, input);
	} else {
		return ParallelCSVInitGlobal(context, input);
	}
}

unique_ptr<LocalTableFunctionState> ReadCSVInitLocal(ExecutionContext &context, TableFunctionInitInput &input,
                                                     GlobalTableFunctionState *global_state_p) {
	auto &csv_data = input.bind_data->Cast<ReadCSVData>();
	if (csv_data.single_threaded) {
		return SingleThreadedReadCSVInitLocal(context, input, global_state_p);
	} else {
		return ParallelReadCSVInitLocal(context, input, global_state_p);
	}
}

static void ReadCSVFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind_data = data_p.bind_data->Cast<ReadCSVData>();
	if (bind_data.single_threaded) {
		SingleThreadedCSVFunction(context, data_p, output);
	} else {
		ParallelReadCSVFunction(context, data_p, output);
	}
}

static idx_t CSVReaderGetBatchIndex(ClientContext &context, const FunctionData *bind_data_p,
                                    LocalTableFunctionState *local_state, GlobalTableFunctionState *global_state) {
	auto &bind_data = bind_data_p->Cast<ReadCSVData>();
	if (bind_data.single_threaded) {
		auto &data = local_state->Cast<SingleThreadedCSVLocalState>();
		return data.file_index;
	}
	auto &data = local_state->Cast<ParallelCSVLocalState>();
	return data.csv_reader->buffer->batch_index;
}

static void ReadCSVAddNamedParameters(TableFunction &table_function) {
	table_function.named_parameters["sep"] = LogicalType::VARCHAR;
	table_function.named_parameters["delim"] = LogicalType::VARCHAR;
	table_function.named_parameters["quote"] = LogicalType::VARCHAR;
	table_function.named_parameters["new_line"] = LogicalType::VARCHAR;
	table_function.named_parameters["escape"] = LogicalType::VARCHAR;
	table_function.named_parameters["nullstr"] = LogicalType::VARCHAR;
	table_function.named_parameters["columns"] = LogicalType::ANY;
	table_function.named_parameters["auto_type_candidates"] = LogicalType::ANY;
	table_function.named_parameters["header"] = LogicalType::BOOLEAN;
	table_function.named_parameters["auto_detect"] = LogicalType::BOOLEAN;
	table_function.named_parameters["sample_size"] = LogicalType::BIGINT;
	table_function.named_parameters["sample_chunk_size"] = LogicalType::BIGINT;
	table_function.named_parameters["sample_chunks"] = LogicalType::BIGINT;
	table_function.named_parameters["all_varchar"] = LogicalType::BOOLEAN;
	table_function.named_parameters["dateformat"] = LogicalType::VARCHAR;
	table_function.named_parameters["timestampformat"] = LogicalType::VARCHAR;
	table_function.named_parameters["normalize_names"] = LogicalType::BOOLEAN;
	table_function.named_parameters["compression"] = LogicalType::VARCHAR;
	table_function.named_parameters["skip"] = LogicalType::BIGINT;
	table_function.named_parameters["max_line_size"] = LogicalType::VARCHAR;
	table_function.named_parameters["maximum_line_size"] = LogicalType::VARCHAR;
	table_function.named_parameters["ignore_errors"] = LogicalType::BOOLEAN;
	table_function.named_parameters["buffer_size"] = LogicalType::UBIGINT;
	table_function.named_parameters["decimal_separator"] = LogicalType::VARCHAR;
	table_function.named_parameters["parallel"] = LogicalType::BOOLEAN;
	table_function.named_parameters["null_padding"] = LogicalType::BOOLEAN;
	table_function.named_parameters["allow_quoted_nulls"] = LogicalType::BOOLEAN;
	table_function.named_parameters["column_types"] = LogicalType::ANY;
	table_function.named_parameters["dtypes"] = LogicalType::ANY;
	table_function.named_parameters["types"] = LogicalType::ANY;
	table_function.named_parameters["names"] = LogicalType::LIST(LogicalType::VARCHAR);
	table_function.named_parameters["column_names"] = LogicalType::LIST(LogicalType::VARCHAR);
	MultiFileReader::AddParameters(table_function);
}

double CSVReaderProgress(ClientContext &context, const FunctionData *bind_data_p,
                         const GlobalTableFunctionState *global_state) {
	auto &bind_data = bind_data_p->Cast<ReadCSVData>();
	if (bind_data.single_threaded) {
		auto &data = global_state->Cast<SingleThreadedCSVState>();
		return data.GetProgress(bind_data);
	} else {
		auto &data = global_state->Cast<ParallelCSVGlobalState>();
		return data.GetProgress(bind_data);
	}
}

void CSVComplexFilterPushdown(ClientContext &context, LogicalGet &get, FunctionData *bind_data_p,
                              vector<unique_ptr<Expression>> &filters) {
	auto &data = bind_data_p->Cast<ReadCSVData>();
	auto reset_reader =
	    MultiFileReader::ComplexFilterPushdown(context, data.files, data.options.file_options, get, filters);
	if (reset_reader) {
		MultiFileReader::PruneReaders(data);
	}
}

unique_ptr<NodeStatistics> CSVReaderCardinality(ClientContext &context, const FunctionData *bind_data_p) {
	auto &bind_data = bind_data_p->Cast<ReadCSVData>();
	idx_t per_file_cardinality = 0;
	if (bind_data.initial_reader && bind_data.initial_reader->file_handle) {
		auto estimated_row_width = (bind_data.csv_types.size() * 5);
		per_file_cardinality = bind_data.initial_reader->file_handle->FileSize() / estimated_row_width;
	} else {
		// determined through the scientific method as the average amount of rows in a CSV file
		per_file_cardinality = 42;
	}
	return make_uniq<NodeStatistics>(bind_data.files.size() * per_file_cardinality);
}

void BufferedCSVReaderOptions::Serialize(FieldWriter &writer) const {
	// common options
	writer.WriteField<bool>(has_delimiter);
	writer.WriteString(delimiter);
	writer.WriteField<bool>(has_quote);
	writer.WriteString(quote);
	writer.WriteField<bool>(has_escape);
	writer.WriteString(escape);
	writer.WriteField<bool>(has_header);
	writer.WriteField<bool>(header);
	writer.WriteField<bool>(ignore_errors);
	writer.WriteField<idx_t>(num_cols);
	writer.WriteField<idx_t>(buffer_sample_size);
	writer.WriteString(null_str);
	writer.WriteField<FileCompressionType>(compression);
	writer.WriteField<NewLineIdentifier>(new_line);
	writer.WriteField<bool>(allow_quoted_nulls);
	// read options
	writer.WriteField<idx_t>(skip_rows);
	writer.WriteField<bool>(skip_rows_set);
	writer.WriteField<idx_t>(maximum_line_size);
	writer.WriteField<bool>(normalize_names);
	writer.WriteListNoReference<bool>(force_not_null);
	writer.WriteField<bool>(all_varchar);
	writer.WriteField<idx_t>(sample_chunk_size);
	writer.WriteField<idx_t>(sample_chunks);
	writer.WriteField<bool>(auto_detect);
	writer.WriteString(file_path);
	writer.WriteString(decimal_separator);
	writer.WriteField<bool>(null_padding);
	writer.WriteField<idx_t>(buffer_size);
	writer.WriteSerializable(file_options);
	// write options
	writer.WriteListNoReference<bool>(force_quote);
	// FIXME: serialize date_format / has_format
	vector<string> csv_formats;
	for (auto &format : date_format) {
		csv_formats.push_back(format.second.format_specifier);
	}
	writer.WriteList<string>(csv_formats);
}

void BufferedCSVReaderOptions::Deserialize(FieldReader &reader) {
	// common options
	has_delimiter = reader.ReadRequired<bool>();
	delimiter = reader.ReadRequired<string>();
	has_quote = reader.ReadRequired<bool>();
	quote = reader.ReadRequired<string>();
	has_escape = reader.ReadRequired<bool>();
	escape = reader.ReadRequired<string>();
	has_header = reader.ReadRequired<bool>();
	header = reader.ReadRequired<bool>();
	ignore_errors = reader.ReadRequired<bool>();
	num_cols = reader.ReadRequired<idx_t>();
	buffer_sample_size = reader.ReadRequired<idx_t>();
	null_str = reader.ReadRequired<string>();
	compression = reader.ReadRequired<FileCompressionType>();
	new_line = reader.ReadRequired<NewLineIdentifier>();
	allow_quoted_nulls = reader.ReadRequired<bool>();
	// read options
	skip_rows = reader.ReadRequired<idx_t>();
	skip_rows_set = reader.ReadRequired<bool>();
	maximum_line_size = reader.ReadRequired<idx_t>();
	normalize_names = reader.ReadRequired<bool>();
	force_not_null = reader.ReadRequiredList<bool>();
	all_varchar = reader.ReadRequired<bool>();
	sample_chunk_size = reader.ReadRequired<idx_t>();
	sample_chunks = reader.ReadRequired<idx_t>();
	auto_detect = reader.ReadRequired<bool>();
	file_path = reader.ReadRequired<string>();
	decimal_separator = reader.ReadRequired<string>();
	null_padding = reader.ReadRequired<bool>();
	buffer_size = reader.ReadRequired<idx_t>();
	file_options = reader.ReadRequiredSerializable<MultiFileReaderOptions, MultiFileReaderOptions>();
	// write options
	force_quote = reader.ReadRequiredList<bool>();
	auto formats = reader.ReadRequiredList<string>();
	vector<LogicalTypeId> format_types {LogicalTypeId::DATE, LogicalTypeId::TIMESTAMP};
	for (idx_t f_idx = 0; f_idx < formats.size(); f_idx++) {
		auto &format = formats[f_idx];
		auto &type = format_types[f_idx];
		if (format.empty()) {
			continue;
		}
		has_format[type] = true;
		StrTimeFormat::ParseFormatSpecifier(format, date_format[type]);
	}
}

static void CSVReaderSerialize(FieldWriter &writer, const FunctionData *bind_data_p, const TableFunction &function) {
	auto &bind_data = bind_data_p->Cast<ReadCSVData>();
	writer.WriteList<string>(bind_data.files);
	writer.WriteRegularSerializableList<LogicalType>(bind_data.csv_types);
	writer.WriteList<string>(bind_data.csv_names);
	writer.WriteRegularSerializableList<LogicalType>(bind_data.return_types);
	writer.WriteList<string>(bind_data.return_names);
	writer.WriteField<idx_t>(bind_data.filename_col_idx);
	writer.WriteField<idx_t>(bind_data.hive_partition_col_idx);
	bind_data.options.Serialize(writer);
	writer.WriteField<bool>(bind_data.single_threaded);
	writer.WriteSerializable(bind_data.reader_bind);
	writer.WriteField<uint32_t>(bind_data.column_info.size());
	for (auto &col : bind_data.column_info) {
		col.Serialize(writer);
	}
}

static unique_ptr<FunctionData> CSVReaderDeserialize(ClientContext &context, FieldReader &reader,
                                                     TableFunction &function) {
	auto result_data = make_uniq<ReadCSVData>();
	result_data->files = reader.ReadRequiredList<string>();
	result_data->csv_types = reader.ReadRequiredSerializableList<LogicalType, LogicalType>();
	result_data->csv_names = reader.ReadRequiredList<string>();
	result_data->return_types = reader.ReadRequiredSerializableList<LogicalType, LogicalType>();
	result_data->return_names = reader.ReadRequiredList<string>();
	result_data->filename_col_idx = reader.ReadRequired<idx_t>();
	result_data->hive_partition_col_idx = reader.ReadRequired<idx_t>();
	result_data->options.Deserialize(reader);
	result_data->single_threaded = reader.ReadField<bool>(true);
	result_data->reader_bind = reader.ReadRequiredSerializable<MultiFileReaderBindData, MultiFileReaderBindData>();
	uint32_t file_number = reader.ReadRequired<uint32_t>();
	for (idx_t i = 0; i < file_number; i++) {
		result_data->column_info.emplace_back(ColumnInfo::Deserialize(reader));
	}
	return std::move(result_data);
}

TableFunction ReadCSVTableFunction::GetFunction() {
	TableFunction read_csv("read_csv", {LogicalType::VARCHAR}, ReadCSVFunction, ReadCSVBind, ReadCSVInitGlobal,
	                       ReadCSVInitLocal);
	read_csv.table_scan_progress = CSVReaderProgress;
	read_csv.pushdown_complex_filter = CSVComplexFilterPushdown;
	read_csv.serialize = CSVReaderSerialize;
	read_csv.deserialize = CSVReaderDeserialize;
	read_csv.get_batch_index = CSVReaderGetBatchIndex;
	read_csv.cardinality = CSVReaderCardinality;
	read_csv.projection_pushdown = true;
	ReadCSVAddNamedParameters(read_csv);
	return read_csv;
}

TableFunction ReadCSVTableFunction::GetAutoFunction() {
	auto read_csv_auto = ReadCSVTableFunction::GetFunction();
	read_csv_auto.name = "read_csv_auto";
	read_csv_auto.bind = ReadCSVAutoBind;
	return read_csv_auto;
}

void ReadCSVTableFunction::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(MultiFileReader::CreateFunctionSet(ReadCSVTableFunction::GetFunction()));
	set.AddFunction(MultiFileReader::CreateFunctionSet(ReadCSVTableFunction::GetAutoFunction()));
}

unique_ptr<TableRef> ReadCSVReplacement(ClientContext &context, const string &table_name, ReplacementScanData *data) {
	auto lower_name = StringUtil::Lower(table_name);
	// remove any compression
	if (StringUtil::EndsWith(lower_name, ".gz")) {
		lower_name = lower_name.substr(0, lower_name.size() - 3);
	} else if (StringUtil::EndsWith(lower_name, ".zst")) {
		lower_name = lower_name.substr(0, lower_name.size() - 4);
	}
	if (!StringUtil::EndsWith(lower_name, ".csv") && !StringUtil::Contains(lower_name, ".csv?") &&
	    !StringUtil::EndsWith(lower_name, ".tsv") && !StringUtil::Contains(lower_name, ".tsv?")) {
		return nullptr;
	}
	auto table_function = make_uniq<TableFunctionRef>();
	vector<unique_ptr<ParsedExpression>> children;
	children.push_back(make_uniq<ConstantExpression>(Value(table_name)));
	table_function->function = make_uniq<FunctionExpression>("read_csv_auto", std::move(children));

	if (!FileSystem::HasGlob(table_name)) {
		table_function->alias = FileSystem::ExtractBaseName(table_name);
	}

	return std::move(table_function);
}

void BuiltinFunctions::RegisterReadFunctions() {
	CSVCopyFunction::RegisterFunction(*this);
	ReadCSVTableFunction::RegisterFunction(*this);
	auto &config = DBConfig::GetConfig(*transaction.db);
	config.replacement_scans.emplace_back(ReadCSVReplacement);
}

} // namespace duckdb



namespace duckdb {

struct RepeatFunctionData : public TableFunctionData {
	RepeatFunctionData(Value value, idx_t target_count) : value(std::move(value)), target_count(target_count) {
	}

	Value value;
	idx_t target_count;
};

struct RepeatOperatorData : public GlobalTableFunctionState {
	RepeatOperatorData() : current_count(0) {
	}
	idx_t current_count;
};

static unique_ptr<FunctionData> RepeatBind(ClientContext &context, TableFunctionBindInput &input,
                                           vector<LogicalType> &return_types, vector<string> &names) {
	// the repeat function returns the type of the first argument
	auto &inputs = input.inputs;
	return_types.push_back(inputs[0].type());
	names.push_back(inputs[0].ToString());
	return make_uniq<RepeatFunctionData>(inputs[0], inputs[1].GetValue<int64_t>());
}

static unique_ptr<GlobalTableFunctionState> RepeatInit(ClientContext &context, TableFunctionInitInput &input) {
	return make_uniq<RepeatOperatorData>();
}

static void RepeatFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind_data = (const RepeatFunctionData &)*data_p.bind_data;
	auto &state = data_p.global_state->Cast<RepeatOperatorData>();

	idx_t remaining = MinValue<idx_t>(bind_data.target_count - state.current_count, STANDARD_VECTOR_SIZE);
	output.data[0].Reference(bind_data.value);
	output.SetCardinality(remaining);
	state.current_count += remaining;
}

static unique_ptr<NodeStatistics> RepeatCardinality(ClientContext &context, const FunctionData *bind_data_p) {
	auto &bind_data = (const RepeatFunctionData &)*bind_data_p;
	return make_uniq<NodeStatistics>(bind_data.target_count, bind_data.target_count);
}

void RepeatTableFunction::RegisterFunction(BuiltinFunctions &set) {
	TableFunction repeat("repeat", {LogicalType::ANY, LogicalType::BIGINT}, RepeatFunction, RepeatBind, RepeatInit);
	repeat.cardinality = RepeatCardinality;
	set.AddFunction(repeat);
}

} // namespace duckdb



namespace duckdb {

struct RepeatRowFunctionData : public TableFunctionData {
	RepeatRowFunctionData(vector<Value> values, idx_t target_count)
	    : values(std::move(values)), target_count(target_count) {
	}

	const vector<Value> values;
	idx_t target_count;
};

struct RepeatRowOperatorData : public GlobalTableFunctionState {
	RepeatRowOperatorData() : current_count(0) {
	}
	idx_t current_count;
};

static unique_ptr<FunctionData> RepeatRowBind(ClientContext &context, TableFunctionBindInput &input,
                                              vector<LogicalType> &return_types, vector<string> &names) {
	auto &inputs = input.inputs;
	for (idx_t input_idx = 0; input_idx < inputs.size(); input_idx++) {
		return_types.push_back(inputs[input_idx].type());
		names.push_back("column" + std::to_string(input_idx));
	}
	return make_uniq<RepeatRowFunctionData>(inputs, input.named_parameters["num_rows"].GetValue<int64_t>());
}

static unique_ptr<GlobalTableFunctionState> RepeatRowInit(ClientContext &context, TableFunctionInitInput &input) {
	return make_uniq<RepeatRowOperatorData>();
}

static void RepeatRowFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind_data = (const RepeatRowFunctionData &)*data_p.bind_data;
	auto &state = data_p.global_state->Cast<RepeatRowOperatorData>();

	idx_t remaining = MinValue<idx_t>(bind_data.target_count - state.current_count, STANDARD_VECTOR_SIZE);
	for (idx_t val_idx = 0; val_idx < bind_data.values.size(); val_idx++) {
		output.data[val_idx].Reference(bind_data.values[val_idx]);
	}
	output.SetCardinality(remaining);
	state.current_count += remaining;
}

static unique_ptr<NodeStatistics> RepeatRowCardinality(ClientContext &context, const FunctionData *bind_data_p) {
	auto &bind_data = (const RepeatRowFunctionData &)*bind_data_p;
	return make_uniq<NodeStatistics>(bind_data.target_count, bind_data.target_count);
}

void RepeatRowTableFunction::RegisterFunction(BuiltinFunctions &set) {
	TableFunction repeat_row("repeat_row", {}, RepeatRowFunction, RepeatRowBind, RepeatRowInit);
	repeat_row.varargs = LogicalType::ANY;
	repeat_row.named_parameters["num_rows"] = LogicalType::BIGINT;
	repeat_row.cardinality = RepeatRowCardinality;
	set.AddFunction(repeat_row);
}

} // namespace duckdb





// this function makes not that much sense on its own but is a demo for table-parameter table-producing functions

namespace duckdb {

static unique_ptr<FunctionData> SummaryFunctionBind(ClientContext &context, TableFunctionBindInput &input,
                                                    vector<LogicalType> &return_types, vector<string> &names) {

	return_types.emplace_back(LogicalType::VARCHAR);
	names.emplace_back("summary");

	for (idx_t i = 0; i < input.input_table_types.size(); i++) {
		return_types.push_back(input.input_table_types[i]);
		names.emplace_back(input.input_table_names[i]);
	}

	return make_uniq<TableFunctionData>();
}

static OperatorResultType SummaryFunction(ExecutionContext &context, TableFunctionInput &data_p, DataChunk &input,
                                          DataChunk &output) {
	output.SetCardinality(input.size());

	for (idx_t row_idx = 0; row_idx < input.size(); row_idx++) {
		string summary_val = "[";

		for (idx_t col_idx = 0; col_idx < input.ColumnCount(); col_idx++) {
			summary_val += input.GetValue(col_idx, row_idx).ToString();
			if (col_idx < input.ColumnCount() - 1) {
				summary_val += ", ";
			}
		}
		summary_val += "]";
		output.SetValue(0, row_idx, Value(summary_val));
	}
	for (idx_t col_idx = 0; col_idx < input.ColumnCount(); col_idx++) {
		output.data[col_idx + 1].Reference(input.data[col_idx]);
	}
	return OperatorResultType::NEED_MORE_INPUT;
}

void SummaryTableFunction::RegisterFunction(BuiltinFunctions &set) {
	TableFunction summary_function("summary", {LogicalType::TABLE}, nullptr, SummaryFunctionBind);
	summary_function.in_out_function = SummaryFunction;
	set.AddFunction(summary_function);
}

} // namespace duckdb










#include <set>

namespace duckdb {

struct DuckDBColumnsData : public GlobalTableFunctionState {
	DuckDBColumnsData() : offset(0), column_offset(0) {
	}

	vector<reference<CatalogEntry>> entries;
	idx_t offset;
	idx_t column_offset;
};

static unique_ptr<FunctionData> DuckDBColumnsBind(ClientContext &context, TableFunctionBindInput &input,
                                                  vector<LogicalType> &return_types, vector<string> &names) {
	names.emplace_back("database_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("database_oid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("schema_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("schema_oid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("table_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("table_oid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("column_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("column_index");
	return_types.emplace_back(LogicalType::INTEGER);

	names.emplace_back("internal");
	return_types.emplace_back(LogicalType::BOOLEAN);

	names.emplace_back("column_default");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("is_nullable");
	return_types.emplace_back(LogicalType::BOOLEAN);

	names.emplace_back("data_type");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("data_type_id");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("character_maximum_length");
	return_types.emplace_back(LogicalType::INTEGER);

	names.emplace_back("numeric_precision");
	return_types.emplace_back(LogicalType::INTEGER);

	names.emplace_back("numeric_precision_radix");
	return_types.emplace_back(LogicalType::INTEGER);

	names.emplace_back("numeric_scale");
	return_types.emplace_back(LogicalType::INTEGER);

	return nullptr;
}

unique_ptr<GlobalTableFunctionState> DuckDBColumnsInit(ClientContext &context, TableFunctionInitInput &input) {
	auto result = make_uniq<DuckDBColumnsData>();

	// scan all the schemas for tables and views and collect them
	auto schemas = Catalog::GetAllSchemas(context);
	for (auto &schema : schemas) {
		schema.get().Scan(context, CatalogType::TABLE_ENTRY,
		                  [&](CatalogEntry &entry) { result->entries.push_back(entry); });
	}
	return std::move(result);
}

class ColumnHelper {
public:
	static unique_ptr<ColumnHelper> Create(CatalogEntry &entry);

	virtual ~ColumnHelper() {
	}

	virtual StandardEntry &Entry() = 0;
	virtual idx_t NumColumns() = 0;
	virtual const string &ColumnName(idx_t col) = 0;
	virtual const LogicalType &ColumnType(idx_t col) = 0;
	virtual const Value ColumnDefault(idx_t col) = 0;
	virtual bool IsNullable(idx_t col) = 0;

	void WriteColumns(idx_t index, idx_t start_col, idx_t end_col, DataChunk &output);
};

class TableColumnHelper : public ColumnHelper {
public:
	explicit TableColumnHelper(TableCatalogEntry &entry) : entry(entry) {
		for (auto &constraint : entry.GetConstraints()) {
			if (constraint->type == ConstraintType::NOT_NULL) {
				auto &not_null = *reinterpret_cast<NotNullConstraint *>(constraint.get());
				not_null_cols.insert(not_null.index.index);
			}
		}
	}

	StandardEntry &Entry() override {
		return entry;
	}
	idx_t NumColumns() override {
		return entry.GetColumns().LogicalColumnCount();
	}
	const string &ColumnName(idx_t col) override {
		return entry.GetColumn(LogicalIndex(col)).Name();
	}
	const LogicalType &ColumnType(idx_t col) override {
		return entry.GetColumn(LogicalIndex(col)).Type();
	}
	const Value ColumnDefault(idx_t col) override {
		auto &column = entry.GetColumn(LogicalIndex(col));
		if (column.DefaultValue()) {
			return Value(column.DefaultValue()->ToString());
		}
		return Value();
	}
	bool IsNullable(idx_t col) override {
		return not_null_cols.find(col) == not_null_cols.end();
	}

private:
	TableCatalogEntry &entry;
	std::set<idx_t> not_null_cols;
};

class ViewColumnHelper : public ColumnHelper {
public:
	explicit ViewColumnHelper(ViewCatalogEntry &entry) : entry(entry) {
	}

	StandardEntry &Entry() override {
		return entry;
	}
	idx_t NumColumns() override {
		return entry.types.size();
	}
	const string &ColumnName(idx_t col) override {
		return entry.aliases[col];
	}
	const LogicalType &ColumnType(idx_t col) override {
		return entry.types[col];
	}
	const Value ColumnDefault(idx_t col) override {
		return Value();
	}
	bool IsNullable(idx_t col) override {
		return true;
	}

private:
	ViewCatalogEntry &entry;
};

unique_ptr<ColumnHelper> ColumnHelper::Create(CatalogEntry &entry) {
	switch (entry.type) {
	case CatalogType::TABLE_ENTRY:
		return make_uniq<TableColumnHelper>(entry.Cast<TableCatalogEntry>());
	case CatalogType::VIEW_ENTRY:
		return make_uniq<ViewColumnHelper>(entry.Cast<ViewCatalogEntry>());
	default:
		throw NotImplementedException("Unsupported catalog type for duckdb_columns");
	}
}

void ColumnHelper::WriteColumns(idx_t start_index, idx_t start_col, idx_t end_col, DataChunk &output) {
	for (idx_t i = start_col; i < end_col; i++) {
		auto index = start_index + (i - start_col);
		auto &entry = Entry();

		idx_t col = 0;
		// database_name, VARCHAR
		output.SetValue(col++, index, entry.catalog.GetName());
		// database_oid, BIGINT
		output.SetValue(col++, index, Value::BIGINT(entry.catalog.GetOid()));
		// schema_name, VARCHAR
		output.SetValue(col++, index, entry.schema.name);
		// schema_oid, BIGINT
		output.SetValue(col++, index, Value::BIGINT(entry.schema.oid));
		// table_name, VARCHAR
		output.SetValue(col++, index, entry.name);
		// table_oid, BIGINT
		output.SetValue(col++, index, Value::BIGINT(entry.oid));
		// column_name, VARCHAR
		output.SetValue(col++, index, Value(ColumnName(i)));
		// column_index, INTEGER
		output.SetValue(col++, index, Value::INTEGER(i + 1));
		// internal, BOOLEAN
		output.SetValue(col++, index, Value::BOOLEAN(entry.internal));
		// column_default, VARCHAR
		output.SetValue(col++, index, Value(ColumnDefault(i)));
		// is_nullable, BOOLEAN
		output.SetValue(col++, index, Value::BOOLEAN(IsNullable(i)));
		// data_type, VARCHAR
		const LogicalType &type = ColumnType(i);
		output.SetValue(col++, index, Value(type.ToString()));
		// data_type_id, BIGINT
		output.SetValue(col++, index, Value::BIGINT(int(type.id())));
		if (type == LogicalType::VARCHAR) {
			// FIXME: need check constraints in place to set this correctly
			// character_maximum_length, INTEGER
			output.SetValue(col++, index, Value());
		} else {
			// "character_maximum_length", PhysicalType::INTEGER
			output.SetValue(col++, index, Value());
		}

		Value numeric_precision, numeric_scale, numeric_precision_radix;
		switch (type.id()) {
		case LogicalTypeId::DECIMAL:
			numeric_precision = Value::INTEGER(DecimalType::GetWidth(type));
			numeric_scale = Value::INTEGER(DecimalType::GetScale(type));
			numeric_precision_radix = Value::INTEGER(10);
			break;
		case LogicalTypeId::HUGEINT:
			numeric_precision = Value::INTEGER(128);
			numeric_scale = Value::INTEGER(0);
			numeric_precision_radix = Value::INTEGER(2);
			break;
		case LogicalTypeId::BIGINT:
			numeric_precision = Value::INTEGER(64);
			numeric_scale = Value::INTEGER(0);
			numeric_precision_radix = Value::INTEGER(2);
			break;
		case LogicalTypeId::INTEGER:
			numeric_precision = Value::INTEGER(32);
			numeric_scale = Value::INTEGER(0);
			numeric_precision_radix = Value::INTEGER(2);
			break;
		case LogicalTypeId::SMALLINT:
			numeric_precision = Value::INTEGER(16);
			numeric_scale = Value::INTEGER(0);
			numeric_precision_radix = Value::INTEGER(2);
			break;
		case LogicalTypeId::TINYINT:
			numeric_precision = Value::INTEGER(8);
			numeric_scale = Value::INTEGER(0);
			numeric_precision_radix = Value::INTEGER(2);
			break;
		case LogicalTypeId::FLOAT:
			numeric_precision = Value::INTEGER(24);
			numeric_scale = Value::INTEGER(0);
			numeric_precision_radix = Value::INTEGER(2);
			break;
		case LogicalTypeId::DOUBLE:
			numeric_precision = Value::INTEGER(53);
			numeric_scale = Value::INTEGER(0);
			numeric_precision_radix = Value::INTEGER(2);
			break;
		default:
			numeric_precision = Value();
			numeric_scale = Value();
			numeric_precision_radix = Value();
			break;
		}

		// numeric_precision, INTEGER
		output.SetValue(col++, index, numeric_precision);
		// numeric_precision_radix, INTEGER
		output.SetValue(col++, index, numeric_precision_radix);
		// numeric_scale, INTEGER
		output.SetValue(col++, index, numeric_scale);
	}
}

void DuckDBColumnsFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = data_p.global_state->Cast<DuckDBColumnsData>();
	if (data.offset >= data.entries.size()) {
		// finished returning values
		return;
	}

	// We need to track the offset of the relation we're writing as well as the last column
	// we wrote from that relation (if any); it's possible that we can fill up the output
	// with a partial list of columns from a relation and will need to pick up processing the
	// next chunk at the same spot.
	idx_t next = data.offset;
	idx_t column_offset = data.column_offset;
	idx_t index = 0;
	while (next < data.entries.size() && index < STANDARD_VECTOR_SIZE) {
		auto column_helper = ColumnHelper::Create(data.entries[next].get());
		idx_t columns = column_helper->NumColumns();

		// Check to see if we are going to exceed the maximum index for a DataChunk
		if (index + (columns - column_offset) > STANDARD_VECTOR_SIZE) {
			idx_t column_limit = column_offset + (STANDARD_VECTOR_SIZE - index);
			output.SetCardinality(STANDARD_VECTOR_SIZE);
			column_helper->WriteColumns(index, column_offset, column_limit, output);

			// Make the current column limit the column offset when we process the next chunk
			column_offset = column_limit;
			break;
		} else {
			// Otherwise, write all of the columns from the current relation and
			// then move on to the next one.
			output.SetCardinality(index + (columns - column_offset));
			column_helper->WriteColumns(index, column_offset, columns, output);
			index += columns - column_offset;
			next++;
			column_offset = 0;
		}
	}
	data.offset = next;
	data.column_offset = column_offset;
}

void DuckDBColumnsFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(TableFunction("duckdb_columns", {}, DuckDBColumnsFunction, DuckDBColumnsBind, DuckDBColumnsInit));
}

} // namespace duckdb


















namespace duckdb {

struct UniqueKeyInfo {
	string schema;
	string table;
	vector<LogicalIndex> columns;

	bool operator==(const UniqueKeyInfo &other) const {
		return (schema == other.schema) && (table == other.table) && (columns == other.columns);
	}
};

} // namespace duckdb

namespace std {

template <>
struct hash<duckdb::UniqueKeyInfo> {
	template <class X>
	static size_t ComputeHash(const X &x) {
		return hash<X>()(x);
	}

	size_t operator()(const duckdb::UniqueKeyInfo &j) const {
		D_ASSERT(j.columns.size() > 0);
		return ComputeHash(j.schema) + ComputeHash(j.table) + ComputeHash(j.columns[0].index);
	}
};

} // namespace std

namespace duckdb {

struct DuckDBConstraintsData : public GlobalTableFunctionState {
	DuckDBConstraintsData() : offset(0), constraint_offset(0), unique_constraint_offset(0) {
	}

	vector<reference<CatalogEntry>> entries;
	idx_t offset;
	idx_t constraint_offset;
	idx_t unique_constraint_offset;
	unordered_map<UniqueKeyInfo, idx_t> known_fk_unique_constraint_offsets;
};

static unique_ptr<FunctionData> DuckDBConstraintsBind(ClientContext &context, TableFunctionBindInput &input,
                                                      vector<LogicalType> &return_types, vector<string> &names) {
	names.emplace_back("database_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("database_oid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("schema_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("schema_oid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("table_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("table_oid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("constraint_index");
	return_types.emplace_back(LogicalType::BIGINT);

	// CHECK, PRIMARY KEY or UNIQUE
	names.emplace_back("constraint_type");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("constraint_text");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("expression");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("constraint_column_indexes");
	return_types.push_back(LogicalType::LIST(LogicalType::BIGINT));

	names.emplace_back("constraint_column_names");
	return_types.push_back(LogicalType::LIST(LogicalType::VARCHAR));

	return nullptr;
}

unique_ptr<GlobalTableFunctionState> DuckDBConstraintsInit(ClientContext &context, TableFunctionInitInput &input) {
	auto result = make_uniq<DuckDBConstraintsData>();

	// scan all the schemas for tables and collect them
	auto schemas = Catalog::GetAllSchemas(context);

	for (auto &schema : schemas) {
		vector<reference<CatalogEntry>> entries;

		schema.get().Scan(context, CatalogType::TABLE_ENTRY, [&](CatalogEntry &entry) {
			if (entry.type == CatalogType::TABLE_ENTRY) {
				entries.push_back(entry);
			}
		});

		sort(entries.begin(), entries.end(), [&](CatalogEntry &x, CatalogEntry &y) { return (x.name < y.name); });

		result->entries.insert(result->entries.end(), entries.begin(), entries.end());
	};

	return std::move(result);
}

void DuckDBConstraintsFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = data_p.global_state->Cast<DuckDBConstraintsData>();
	if (data.offset >= data.entries.size()) {
		// finished returning values
		return;
	}
	// start returning values
	// either fill up the chunk or return all the remaining columns
	idx_t count = 0;
	while (data.offset < data.entries.size() && count < STANDARD_VECTOR_SIZE) {
		auto &entry = data.entries[data.offset].get();
		D_ASSERT(entry.type == CatalogType::TABLE_ENTRY);

		auto &table = entry.Cast<TableCatalogEntry>();
		auto &constraints = table.GetConstraints();
		bool is_duck_table = table.IsDuckTable();
		for (; data.constraint_offset < constraints.size() && count < STANDARD_VECTOR_SIZE; data.constraint_offset++) {
			auto &constraint = constraints[data.constraint_offset];
			// return values:
			// constraint_type, VARCHAR
			// Processing this first due to shortcut (early continue)
			string constraint_type;
			switch (constraint->type) {
			case ConstraintType::CHECK:
				constraint_type = "CHECK";
				break;
			case ConstraintType::UNIQUE: {
				auto &unique = constraint->Cast<UniqueConstraint>();
				constraint_type = unique.is_primary_key ? "PRIMARY KEY" : "UNIQUE";
				break;
			}
			case ConstraintType::NOT_NULL:
				constraint_type = "NOT NULL";
				break;
			case ConstraintType::FOREIGN_KEY: {
				if (!is_duck_table) {
					continue;
				}
				auto &bound_constraints = table.GetBoundConstraints();
				auto &bound_foreign_key = bound_constraints[data.constraint_offset]->Cast<BoundForeignKeyConstraint>();
				if (bound_foreign_key.info.type == ForeignKeyType::FK_TYPE_PRIMARY_KEY_TABLE) {
					// Those are already covered by PRIMARY KEY and UNIQUE entries
					continue;
				}
				constraint_type = "FOREIGN KEY";
				break;
			}
			default:
				throw NotImplementedException("Unimplemented constraint for duckdb_constraints");
			}

			idx_t col = 0;
			// database_name, LogicalType::VARCHAR
			output.SetValue(col++, count, Value(table.schema.catalog.GetName()));
			// database_oid, LogicalType::BIGINT
			output.SetValue(col++, count, Value::BIGINT(table.schema.catalog.GetOid()));
			// schema_name, LogicalType::VARCHAR
			output.SetValue(col++, count, Value(table.schema.name));
			// schema_oid, LogicalType::BIGINT
			output.SetValue(col++, count, Value::BIGINT(table.schema.oid));
			// table_name, LogicalType::VARCHAR
			output.SetValue(col++, count, Value(table.name));
			// table_oid, LogicalType::BIGINT
			output.SetValue(col++, count, Value::BIGINT(table.oid));

			// constraint_index, BIGINT
			UniqueKeyInfo uk_info;

			if (is_duck_table) {
				auto &bound_constraint = *table.GetBoundConstraints()[data.constraint_offset];
				switch (bound_constraint.type) {
				case ConstraintType::UNIQUE: {
					auto &bound_unique = bound_constraint.Cast<BoundUniqueConstraint>();
					uk_info = {table.schema.name, table.name, bound_unique.keys};
					break;
				}
				case ConstraintType::FOREIGN_KEY: {
					const auto &bound_foreign_key = bound_constraint.Cast<BoundForeignKeyConstraint>();
					const auto &info = bound_foreign_key.info;
					// find the other table
					auto table_entry = Catalog::GetEntry<TableCatalogEntry>(
					    context, table.catalog.GetName(), info.schema, info.table, OnEntryNotFound::RETURN_NULL);
					if (!table_entry) {
						throw InternalException("dukdb_constraints: entry %s.%s referenced in foreign key not found",
						                        info.schema, info.table);
					}
					vector<LogicalIndex> index;
					for (auto &key : info.pk_keys) {
						index.push_back(table_entry->GetColumns().PhysicalToLogical(key));
					}
					uk_info = {table_entry->schema.name, table_entry->name, index};
					break;
				}
				default:
					break;
				}
			}

			if (uk_info.columns.empty()) {
				output.SetValue(col++, count, Value::BIGINT(data.unique_constraint_offset++));
			} else {
				auto known_unique_constraint_offset = data.known_fk_unique_constraint_offsets.find(uk_info);
				if (known_unique_constraint_offset == data.known_fk_unique_constraint_offsets.end()) {
					data.known_fk_unique_constraint_offsets.insert(make_pair(uk_info, data.unique_constraint_offset));
					output.SetValue(col++, count, Value::BIGINT(data.unique_constraint_offset));
					data.unique_constraint_offset++;
				} else {
					output.SetValue(col++, count, Value::BIGINT(known_unique_constraint_offset->second));
				}
			}
			output.SetValue(col++, count, Value(constraint_type));

			// constraint_text, VARCHAR
			output.SetValue(col++, count, Value(constraint->ToString()));

			// expression, VARCHAR
			Value expression_text;
			if (constraint->type == ConstraintType::CHECK) {
				auto &check = constraint->Cast<CheckConstraint>();
				expression_text = Value(check.expression->ToString());
			}
			output.SetValue(col++, count, expression_text);

			vector<LogicalIndex> column_index_list;
			if (is_duck_table) {
				auto &bound_constraint = *table.GetBoundConstraints()[data.constraint_offset];
				switch (bound_constraint.type) {
				case ConstraintType::CHECK: {
					auto &bound_check = bound_constraint.Cast<BoundCheckConstraint>();
					for (auto &col_idx : bound_check.bound_columns) {
						column_index_list.push_back(table.GetColumns().PhysicalToLogical(col_idx));
					}
					break;
				}
				case ConstraintType::UNIQUE: {
					auto &bound_unique = bound_constraint.Cast<BoundUniqueConstraint>();
					for (auto &col_idx : bound_unique.keys) {
						column_index_list.push_back(col_idx);
					}
					break;
				}
				case ConstraintType::NOT_NULL: {
					auto &bound_not_null = bound_constraint.Cast<BoundNotNullConstraint>();
					column_index_list.push_back(table.GetColumns().PhysicalToLogical(bound_not_null.index));
					break;
				}
				case ConstraintType::FOREIGN_KEY: {
					auto &bound_foreign_key = bound_constraint.Cast<BoundForeignKeyConstraint>();
					for (auto &col_idx : bound_foreign_key.info.fk_keys) {
						column_index_list.push_back(table.GetColumns().PhysicalToLogical(col_idx));
					}
					break;
				}
				default:
					throw NotImplementedException("Unimplemented constraint for duckdb_constraints");
				}
			}

			vector<Value> index_list;
			vector<Value> column_name_list;
			for (auto column_index : column_index_list) {
				index_list.push_back(Value::BIGINT(column_index.index));
				column_name_list.emplace_back(table.GetColumn(column_index).Name());
			}

			// constraint_column_indexes, LIST
			output.SetValue(col++, count, Value::LIST(LogicalType::BIGINT, std::move(index_list)));

			// constraint_column_names, LIST
			output.SetValue(col++, count, Value::LIST(LogicalType::VARCHAR, std::move(column_name_list)));

			count++;
		}
		if (data.constraint_offset >= constraints.size()) {
			data.constraint_offset = 0;
			data.offset++;
		}
	}
	output.SetCardinality(count);
}

void DuckDBConstraintsFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(TableFunction("duckdb_constraints", {}, DuckDBConstraintsFunction, DuckDBConstraintsBind,
	                              DuckDBConstraintsInit));
}

} // namespace duckdb




namespace duckdb {

struct DuckDBDatabasesData : public GlobalTableFunctionState {
	DuckDBDatabasesData() : offset(0) {
	}

	vector<reference<AttachedDatabase>> entries;
	idx_t offset;
};

static unique_ptr<FunctionData> DuckDBDatabasesBind(ClientContext &context, TableFunctionBindInput &input,
                                                    vector<LogicalType> &return_types, vector<string> &names) {
	names.emplace_back("database_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("database_oid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("path");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("internal");
	return_types.emplace_back(LogicalType::BOOLEAN);

	names.emplace_back("type");
	return_types.emplace_back(LogicalType::VARCHAR);

	return nullptr;
}

unique_ptr<GlobalTableFunctionState> DuckDBDatabasesInit(ClientContext &context, TableFunctionInitInput &input) {
	auto result = make_uniq<DuckDBDatabasesData>();

	// scan all the schemas for tables and collect them and collect them
	auto &db_manager = DatabaseManager::Get(context);
	result->entries = db_manager.GetDatabases(context);
	return std::move(result);
}

void DuckDBDatabasesFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = data_p.global_state->Cast<DuckDBDatabasesData>();
	if (data.offset >= data.entries.size()) {
		// finished returning values
		return;
	}
	// start returning values
	// either fill up the chunk or return all the remaining columns
	idx_t count = 0;
	while (data.offset < data.entries.size() && count < STANDARD_VECTOR_SIZE) {
		auto &entry = data.entries[data.offset++];

		auto &attached = entry.get().Cast<AttachedDatabase>();
		// return values:

		idx_t col = 0;
		// database_name, VARCHAR
		output.SetValue(col++, count, attached.GetName());
		// database_oid, BIGINT
		output.SetValue(col++, count, Value::BIGINT(attached.oid));
		// path, VARCHAR
		bool is_internal = attached.IsSystem() || attached.IsTemporary();
		Value db_path;
		if (!is_internal) {
			bool in_memory = attached.GetCatalog().InMemory();
			if (!in_memory) {
				db_path = Value(attached.GetCatalog().GetDBPath());
			}
		}
		output.SetValue(col++, count, db_path);
		// internal, BOOLEAN
		output.SetValue(col++, count, Value::BOOLEAN(is_internal));
		// type, VARCHAR
		output.SetValue(col++, count, Value(attached.GetCatalog().GetCatalogType()));

		count++;
	}
	output.SetCardinality(count);
}

void DuckDBDatabasesFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(
	    TableFunction("duckdb_databases", {}, DuckDBDatabasesFunction, DuckDBDatabasesBind, DuckDBDatabasesInit));
}

} // namespace duckdb








namespace duckdb {

struct DependencyInformation {
	DependencyInformation(CatalogEntry &object, CatalogEntry &dependent, DependencyType type)
	    : object(object), dependent(dependent), type(type) {
	}

	CatalogEntry &object;
	CatalogEntry &dependent;
	DependencyType type;
};

struct DuckDBDependenciesData : public GlobalTableFunctionState {
	DuckDBDependenciesData() : offset(0) {
	}

	vector<DependencyInformation> entries;
	idx_t offset;
};

static unique_ptr<FunctionData> DuckDBDependenciesBind(ClientContext &context, TableFunctionBindInput &input,
                                                       vector<LogicalType> &return_types, vector<string> &names) {
	names.emplace_back("classid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("objid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("objsubid");
	return_types.emplace_back(LogicalType::INTEGER);

	names.emplace_back("refclassid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("refobjid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("refobjsubid");
	return_types.emplace_back(LogicalType::INTEGER);

	names.emplace_back("deptype");
	return_types.emplace_back(LogicalType::VARCHAR);

	return nullptr;
}

unique_ptr<GlobalTableFunctionState> DuckDBDependenciesInit(ClientContext &context, TableFunctionInitInput &input) {
	auto result = make_uniq<DuckDBDependenciesData>();

	// scan all the schemas and collect them
	auto &catalog = Catalog::GetCatalog(context, INVALID_CATALOG);
	if (catalog.IsDuckCatalog()) {
		auto &duck_catalog = catalog.Cast<DuckCatalog>();
		auto &dependency_manager = duck_catalog.GetDependencyManager();
		dependency_manager.Scan([&](CatalogEntry &obj, CatalogEntry &dependent, DependencyType type) {
			result->entries.emplace_back(obj, dependent, type);
		});
	}

	return std::move(result);
}

void DuckDBDependenciesFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = data_p.global_state->Cast<DuckDBDependenciesData>();
	if (data.offset >= data.entries.size()) {
		// finished returning values
		return;
	}
	// start returning values
	// either fill up the chunk or return all the remaining columns
	idx_t count = 0;
	while (data.offset < data.entries.size() && count < STANDARD_VECTOR_SIZE) {
		auto &entry = data.entries[data.offset];

		// return values:
		// classid, LogicalType::BIGINT
		output.SetValue(0, count, Value::BIGINT(0));
		// objid, LogicalType::BIGINT
		output.SetValue(1, count, Value::BIGINT(entry.object.oid));
		// objsubid, LogicalType::INTEGER
		output.SetValue(2, count, Value::INTEGER(0));
		// refclassid, LogicalType::BIGINT
		output.SetValue(3, count, Value::BIGINT(0));
		// refobjid, LogicalType::BIGINT
		output.SetValue(4, count, Value::BIGINT(entry.dependent.oid));
		// refobjsubid, LogicalType::INTEGER
		output.SetValue(5, count, Value::INTEGER(0));
		// deptype, LogicalType::VARCHAR
		string dependency_type_str;
		switch (entry.type) {
		case DependencyType::DEPENDENCY_REGULAR:
			dependency_type_str = "n";
			break;
		case DependencyType::DEPENDENCY_AUTOMATIC:
			dependency_type_str = "a";
			break;
		default:
			throw NotImplementedException("Unimplemented dependency type");
		}
		output.SetValue(6, count, Value(dependency_type_str));

		data.offset++;
		count++;
	}
	output.SetCardinality(count);
}

void DuckDBDependenciesFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(TableFunction("duckdb_dependencies", {}, DuckDBDependenciesFunction, DuckDBDependenciesBind,
	                              DuckDBDependenciesInit));
}

} // namespace duckdb










namespace duckdb {

struct ExtensionInformation {
	string name;
	bool loaded = false;
	bool installed = false;
	string file_path;
	string description;
	vector<Value> aliases;
};

struct DuckDBExtensionsData : public GlobalTableFunctionState {
	DuckDBExtensionsData() : offset(0) {
	}

	vector<ExtensionInformation> entries;
	idx_t offset;
};

static unique_ptr<FunctionData> DuckDBExtensionsBind(ClientContext &context, TableFunctionBindInput &input,
                                                     vector<LogicalType> &return_types, vector<string> &names) {
	names.emplace_back("extension_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("loaded");
	return_types.emplace_back(LogicalType::BOOLEAN);

	names.emplace_back("installed");
	return_types.emplace_back(LogicalType::BOOLEAN);

	names.emplace_back("install_path");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("description");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("aliases");
	return_types.emplace_back(LogicalType::LIST(LogicalType::VARCHAR));

	return nullptr;
}

unique_ptr<GlobalTableFunctionState> DuckDBExtensionsInit(ClientContext &context, TableFunctionInitInput &input) {
	auto result = make_uniq<DuckDBExtensionsData>();

	auto &fs = FileSystem::GetFileSystem(context);
	auto &db = DatabaseInstance::GetDatabase(context);

	map<string, ExtensionInformation> installed_extensions;
	auto extension_count = ExtensionHelper::DefaultExtensionCount();
	auto alias_count = ExtensionHelper::ExtensionAliasCount();
	for (idx_t i = 0; i < extension_count; i++) {
		auto extension = ExtensionHelper::GetDefaultExtension(i);
		ExtensionInformation info;
		info.name = extension.name;
		info.installed = extension.statically_loaded;
		info.loaded = false;
		info.file_path = extension.statically_loaded ? "(BUILT-IN)" : string();
		info.description = extension.description;
		for (idx_t k = 0; k < alias_count; k++) {
			auto alias = ExtensionHelper::GetExtensionAlias(k);
			if (info.name == alias.extension) {
				info.aliases.emplace_back(alias.alias);
			}
		}
		installed_extensions[info.name] = std::move(info);
	}
#ifndef WASM_LOADABLE_EXTENSIONS
	// scan the install directory for installed extensions
	auto ext_directory = ExtensionHelper::ExtensionDirectory(context);
	fs.ListFiles(ext_directory, [&](const string &path, bool is_directory) {
		if (!StringUtil::EndsWith(path, ".duckdb_extension")) {
			return;
		}
		ExtensionInformation info;
		info.name = fs.ExtractBaseName(path);
		info.loaded = false;
		info.file_path = fs.JoinPath(ext_directory, path);
		auto entry = installed_extensions.find(info.name);
		if (entry == installed_extensions.end()) {
			installed_extensions[info.name] = std::move(info);
		} else {
			if (!entry->second.loaded) {
				entry->second.file_path = info.file_path;
			}
			entry->second.installed = true;
		}
	});
#endif
	// now check the list of currently loaded extensions
	auto &loaded_extensions = db.LoadedExtensions();
	for (auto &ext_name : loaded_extensions) {
		auto entry = installed_extensions.find(ext_name);
		if (entry == installed_extensions.end()) {
			ExtensionInformation info;
			info.name = ext_name;
			info.loaded = true;
			installed_extensions[ext_name] = std::move(info);
		} else {
			entry->second.loaded = true;
		}
	}

	result->entries.reserve(installed_extensions.size());
	for (auto &kv : installed_extensions) {
		result->entries.push_back(std::move(kv.second));
	}
	return std::move(result);
}

void DuckDBExtensionsFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = data_p.global_state->Cast<DuckDBExtensionsData>();
	if (data.offset >= data.entries.size()) {
		// finished returning values
		return;
	}
	// start returning values
	// either fill up the chunk or return all the remaining columns
	idx_t count = 0;
	while (data.offset < data.entries.size() && count < STANDARD_VECTOR_SIZE) {
		auto &entry = data.entries[data.offset];

		// return values:
		// extension_name LogicalType::VARCHAR
		output.SetValue(0, count, Value(entry.name));
		// loaded LogicalType::BOOLEAN
		output.SetValue(1, count, Value::BOOLEAN(entry.loaded));
		// installed LogicalType::BOOLEAN
		output.SetValue(2, count, !entry.installed && entry.loaded ? Value() : Value::BOOLEAN(entry.installed));
		// install_path LogicalType::VARCHAR
		output.SetValue(3, count, Value(entry.file_path));
		// description LogicalType::VARCHAR
		output.SetValue(4, count, Value(entry.description));
		// aliases     LogicalType::LIST(LogicalType::VARCHAR)
		output.SetValue(5, count, Value::LIST(LogicalType::VARCHAR, entry.aliases));

		data.offset++;
		count++;
	}
	output.SetCardinality(count);
}

void DuckDBExtensionsFun::RegisterFunction(BuiltinFunctions &set) {
	TableFunctionSet functions("duckdb_extensions");
	functions.AddFunction(TableFunction({}, DuckDBExtensionsFunction, DuckDBExtensionsBind, DuckDBExtensionsInit));
	set.AddFunction(functions);
}

} // namespace duckdb

















namespace duckdb {

struct DuckDBFunctionsData : public GlobalTableFunctionState {
	DuckDBFunctionsData() : offset(0), offset_in_entry(0) {
	}

	vector<reference<CatalogEntry>> entries;
	idx_t offset;
	idx_t offset_in_entry;
};

static unique_ptr<FunctionData> DuckDBFunctionsBind(ClientContext &context, TableFunctionBindInput &input,
                                                    vector<LogicalType> &return_types, vector<string> &names) {
	names.emplace_back("database_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("schema_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("function_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("function_type");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("description");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("return_type");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("parameters");
	return_types.push_back(LogicalType::LIST(LogicalType::VARCHAR));

	names.emplace_back("parameter_types");
	return_types.push_back(LogicalType::LIST(LogicalType::VARCHAR));

	names.emplace_back("varargs");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("macro_definition");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("has_side_effects");
	return_types.emplace_back(LogicalType::BOOLEAN);

	names.emplace_back("internal");
	return_types.emplace_back(LogicalType::BOOLEAN);

	names.emplace_back("function_oid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("example");
	return_types.emplace_back(LogicalType::VARCHAR);

	return nullptr;
}

static void ExtractFunctionsFromSchema(ClientContext &context, SchemaCatalogEntry &schema,
                                       DuckDBFunctionsData &result) {
	schema.Scan(context, CatalogType::SCALAR_FUNCTION_ENTRY,
	            [&](CatalogEntry &entry) { result.entries.push_back(entry); });
	schema.Scan(context, CatalogType::TABLE_FUNCTION_ENTRY,
	            [&](CatalogEntry &entry) { result.entries.push_back(entry); });
	schema.Scan(context, CatalogType::PRAGMA_FUNCTION_ENTRY,
	            [&](CatalogEntry &entry) { result.entries.push_back(entry); });
}

unique_ptr<GlobalTableFunctionState> DuckDBFunctionsInit(ClientContext &context, TableFunctionInitInput &input) {
	auto result = make_uniq<DuckDBFunctionsData>();

	// scan all the schemas for tables and collect them and collect them
	auto schemas = Catalog::GetAllSchemas(context);
	for (auto &schema : schemas) {
		ExtractFunctionsFromSchema(context, schema.get(), *result);
	};

	std::sort(result->entries.begin(), result->entries.end(),
	          [&](reference<CatalogEntry> a, reference<CatalogEntry> b) {
		          return (int32_t)a.get().type < (int32_t)b.get().type;
	          });
	return std::move(result);
}

struct ScalarFunctionExtractor {
	static idx_t FunctionCount(ScalarFunctionCatalogEntry &entry) {
		return entry.functions.Size();
	}

	static Value GetFunctionType() {
		return Value("scalar");
	}

	static Value GetReturnType(ScalarFunctionCatalogEntry &entry, idx_t offset) {
		return Value(entry.functions.GetFunctionByOffset(offset).return_type.ToString());
	}

	static vector<Value> GetParameters(ScalarFunctionCatalogEntry &entry, idx_t offset) {
		vector<Value> results;
		for (idx_t i = 0; i < entry.functions.GetFunctionByOffset(offset).arguments.size(); i++) {
			results.emplace_back("col" + to_string(i));
		}
		return results;
	}

	static Value GetParameterTypes(ScalarFunctionCatalogEntry &entry, idx_t offset) {
		vector<Value> results;
		auto fun = entry.functions.GetFunctionByOffset(offset);
		for (idx_t i = 0; i < fun.arguments.size(); i++) {
			results.emplace_back(fun.arguments[i].ToString());
		}
		return Value::LIST(LogicalType::VARCHAR, std::move(results));
	}

	static Value GetVarArgs(ScalarFunctionCatalogEntry &entry, idx_t offset) {
		auto fun = entry.functions.GetFunctionByOffset(offset);
		return !fun.HasVarArgs() ? Value() : Value(fun.varargs.ToString());
	}

	static Value GetMacroDefinition(ScalarFunctionCatalogEntry &entry, idx_t offset) {
		return Value();
	}

	static Value HasSideEffects(ScalarFunctionCatalogEntry &entry, idx_t offset) {
		return Value::BOOLEAN(entry.functions.GetFunctionByOffset(offset).side_effects ==
		                      FunctionSideEffects::HAS_SIDE_EFFECTS);
	}
};

struct AggregateFunctionExtractor {
	static idx_t FunctionCount(AggregateFunctionCatalogEntry &entry) {
		return entry.functions.Size();
	}

	static Value GetFunctionType() {
		return Value("aggregate");
	}

	static Value GetReturnType(AggregateFunctionCatalogEntry &entry, idx_t offset) {
		return Value(entry.functions.GetFunctionByOffset(offset).return_type.ToString());
	}

	static vector<Value> GetParameters(AggregateFunctionCatalogEntry &entry, idx_t offset) {
		vector<Value> results;
		for (idx_t i = 0; i < entry.functions.GetFunctionByOffset(offset).arguments.size(); i++) {
			results.emplace_back("col" + to_string(i));
		}
		return results;
	}

	static Value GetParameterTypes(AggregateFunctionCatalogEntry &entry, idx_t offset) {
		vector<Value> results;
		auto fun = entry.functions.GetFunctionByOffset(offset);
		for (idx_t i = 0; i < fun.arguments.size(); i++) {
			results.emplace_back(fun.arguments[i].ToString());
		}
		return Value::LIST(LogicalType::VARCHAR, std::move(results));
	}

	static Value GetVarArgs(AggregateFunctionCatalogEntry &entry, idx_t offset) {
		auto fun = entry.functions.GetFunctionByOffset(offset);
		return !fun.HasVarArgs() ? Value() : Value(fun.varargs.ToString());
	}

	static Value GetMacroDefinition(AggregateFunctionCatalogEntry &entry, idx_t offset) {
		return Value();
	}

	static Value HasSideEffects(AggregateFunctionCatalogEntry &entry, idx_t offset) {
		return Value::BOOLEAN(entry.functions.GetFunctionByOffset(offset).side_effects ==
		                      FunctionSideEffects::HAS_SIDE_EFFECTS);
	}
};

struct MacroExtractor {
	static idx_t FunctionCount(ScalarMacroCatalogEntry &entry) {
		return 1;
	}

	static Value GetFunctionType() {
		return Value("macro");
	}

	static Value GetReturnType(ScalarMacroCatalogEntry &entry, idx_t offset) {
		return Value();
	}

	static vector<Value> GetParameters(ScalarMacroCatalogEntry &entry, idx_t offset) {
		vector<Value> results;
		for (auto &param : entry.function->parameters) {
			D_ASSERT(param->type == ExpressionType::COLUMN_REF);
			auto &colref = param->Cast<ColumnRefExpression>();
			results.emplace_back(colref.GetColumnName());
		}
		for (auto &param_entry : entry.function->default_parameters) {
			results.emplace_back(param_entry.first);
		}
		return results;
	}

	static Value GetParameterTypes(ScalarMacroCatalogEntry &entry, idx_t offset) {
		vector<Value> results;
		for (idx_t i = 0; i < entry.function->parameters.size(); i++) {
			results.emplace_back(LogicalType::VARCHAR);
		}
		for (idx_t i = 0; i < entry.function->default_parameters.size(); i++) {
			results.emplace_back(LogicalType::VARCHAR);
		}
		return Value::LIST(LogicalType::VARCHAR, std::move(results));
	}

	static Value GetVarArgs(ScalarMacroCatalogEntry &entry, idx_t offset) {
		return Value();
	}

	static Value GetMacroDefinition(ScalarMacroCatalogEntry &entry, idx_t offset) {
		D_ASSERT(entry.function->type == MacroType::SCALAR_MACRO);
		auto &func = entry.function->Cast<ScalarMacroFunction>();
		return func.expression->ToString();
	}

	static Value HasSideEffects(ScalarMacroCatalogEntry &entry, idx_t offset) {
		return Value();
	}
};

struct TableMacroExtractor {
	static idx_t FunctionCount(TableMacroCatalogEntry &entry) {
		return 1;
	}

	static Value GetFunctionType() {
		return Value("table_macro");
	}

	static Value GetReturnType(TableMacroCatalogEntry &entry, idx_t offset) {
		return Value();
	}

	static vector<Value> GetParameters(TableMacroCatalogEntry &entry, idx_t offset) {
		vector<Value> results;
		for (auto &param : entry.function->parameters) {
			D_ASSERT(param->type == ExpressionType::COLUMN_REF);
			auto &colref = param->Cast<ColumnRefExpression>();
			results.emplace_back(colref.GetColumnName());
		}
		for (auto &param_entry : entry.function->default_parameters) {
			results.emplace_back(param_entry.first);
		}
		return results;
	}

	static Value GetParameterTypes(TableMacroCatalogEntry &entry, idx_t offset) {
		vector<Value> results;
		for (idx_t i = 0; i < entry.function->parameters.size(); i++) {
			results.emplace_back(LogicalType::VARCHAR);
		}
		for (idx_t i = 0; i < entry.function->default_parameters.size(); i++) {
			results.emplace_back(LogicalType::VARCHAR);
		}
		return Value::LIST(LogicalType::VARCHAR, std::move(results));
	}

	static Value GetVarArgs(TableMacroCatalogEntry &entry, idx_t offset) {
		return Value();
	}

	static Value GetMacroDefinition(TableMacroCatalogEntry &entry, idx_t offset) {
		if (entry.function->type == MacroType::SCALAR_MACRO) {
			auto &func = entry.function->Cast<ScalarMacroFunction>();
			return func.expression->ToString();
		}
		return Value();
	}

	static Value HasSideEffects(TableMacroCatalogEntry &entry, idx_t offset) {
		return Value();
	}
};

struct TableFunctionExtractor {
	static idx_t FunctionCount(TableFunctionCatalogEntry &entry) {
		return entry.functions.Size();
	}

	static Value GetFunctionType() {
		return Value("table");
	}

	static Value GetReturnType(TableFunctionCatalogEntry &entry, idx_t offset) {
		return Value();
	}

	static vector<Value> GetParameters(TableFunctionCatalogEntry &entry, idx_t offset) {
		vector<Value> results;
		auto fun = entry.functions.GetFunctionByOffset(offset);
		for (idx_t i = 0; i < fun.arguments.size(); i++) {
			results.emplace_back("col" + to_string(i));
		}
		for (auto &param : fun.named_parameters) {
			results.emplace_back(param.first);
		}
		return results;
	}

	static Value GetParameterTypes(TableFunctionCatalogEntry &entry, idx_t offset) {
		vector<Value> results;
		auto fun = entry.functions.GetFunctionByOffset(offset);

		for (idx_t i = 0; i < fun.arguments.size(); i++) {
			results.emplace_back(fun.arguments[i].ToString());
		}
		for (auto &param : fun.named_parameters) {
			results.emplace_back(param.second.ToString());
		}
		return Value::LIST(LogicalType::VARCHAR, std::move(results));
	}

	static Value GetVarArgs(TableFunctionCatalogEntry &entry, idx_t offset) {
		auto fun = entry.functions.GetFunctionByOffset(offset);
		return !fun.HasVarArgs() ? Value() : Value(fun.varargs.ToString());
	}

	static Value GetMacroDefinition(TableFunctionCatalogEntry &entry, idx_t offset) {
		return Value();
	}

	static Value HasSideEffects(TableFunctionCatalogEntry &entry, idx_t offset) {
		return Value();
	}
};

struct PragmaFunctionExtractor {
	static idx_t FunctionCount(PragmaFunctionCatalogEntry &entry) {
		return entry.functions.Size();
	}

	static Value GetFunctionType() {
		return Value("pragma");
	}

	static Value GetReturnType(PragmaFunctionCatalogEntry &entry, idx_t offset) {
		return Value();
	}

	static vector<Value> GetParameters(PragmaFunctionCatalogEntry &entry, idx_t offset) {
		vector<Value> results;
		auto fun = entry.functions.GetFunctionByOffset(offset);

		for (idx_t i = 0; i < fun.arguments.size(); i++) {
			results.emplace_back("col" + to_string(i));
		}
		for (auto &param : fun.named_parameters) {
			results.emplace_back(param.first);
		}
		return results;
	}

	static Value GetParameterTypes(PragmaFunctionCatalogEntry &entry, idx_t offset) {
		vector<Value> results;
		auto fun = entry.functions.GetFunctionByOffset(offset);

		for (idx_t i = 0; i < fun.arguments.size(); i++) {
			results.emplace_back(fun.arguments[i].ToString());
		}
		for (auto &param : fun.named_parameters) {
			results.emplace_back(param.second.ToString());
		}
		return Value::LIST(LogicalType::VARCHAR, std::move(results));
	}

	static Value GetVarArgs(PragmaFunctionCatalogEntry &entry, idx_t offset) {
		auto fun = entry.functions.GetFunctionByOffset(offset);
		return !fun.HasVarArgs() ? Value() : Value(fun.varargs.ToString());
	}

	static Value GetMacroDefinition(PragmaFunctionCatalogEntry &entry, idx_t offset) {
		return Value();
	}

	static Value HasSideEffects(PragmaFunctionCatalogEntry &entry, idx_t offset) {
		return Value();
	}
};

template <class T, class OP>
bool ExtractFunctionData(FunctionEntry &entry, idx_t function_idx, DataChunk &output, idx_t output_offset) {
	auto &function = entry.Cast<T>();
	idx_t col = 0;

	// database_name, LogicalType::VARCHAR
	output.SetValue(col++, output_offset, Value(function.schema.catalog.GetName()));

	// schema_name, LogicalType::VARCHAR
	output.SetValue(col++, output_offset, Value(function.schema.name));

	// function_name, LogicalType::VARCHAR
	output.SetValue(col++, output_offset, Value(function.name));

	// function_type, LogicalType::VARCHAR
	output.SetValue(col++, output_offset, Value(OP::GetFunctionType()));

	// function_description, LogicalType::VARCHAR
	output.SetValue(col++, output_offset, entry.description.empty() ? Value() : entry.description);

	// return_type, LogicalType::VARCHAR
	output.SetValue(col++, output_offset, OP::GetReturnType(function, function_idx));

	// parameters, LogicalType::LIST(LogicalType::VARCHAR)
	auto parameters = OP::GetParameters(function, function_idx);
	for (idx_t param_idx = 0; param_idx < function.parameter_names.size() && param_idx < parameters.size();
	     param_idx++) {
		parameters[param_idx] = Value(function.parameter_names[param_idx]);
	}
	output.SetValue(col++, output_offset, Value::LIST(LogicalType::VARCHAR, std::move(parameters)));

	// parameter_types, LogicalType::LIST(LogicalType::VARCHAR)
	output.SetValue(col++, output_offset, OP::GetParameterTypes(function, function_idx));

	// varargs, LogicalType::VARCHAR
	output.SetValue(col++, output_offset, OP::GetVarArgs(function, function_idx));

	// macro_definition, LogicalType::VARCHAR
	output.SetValue(col++, output_offset, OP::GetMacroDefinition(function, function_idx));

	// has_side_effects, LogicalType::BOOLEAN
	output.SetValue(col++, output_offset, OP::HasSideEffects(function, function_idx));

	// internal, LogicalType::BOOLEAN
	output.SetValue(col++, output_offset, Value::BOOLEAN(function.internal));

	// function_oid, LogicalType::BIGINT
	output.SetValue(col++, output_offset, Value::BIGINT(function.oid));

	// example, LogicalType::VARCHAR
	output.SetValue(col++, output_offset, entry.example.empty() ? Value() : entry.example);

	return function_idx + 1 == OP::FunctionCount(function);
}

void DuckDBFunctionsFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = (DuckDBFunctionsData &)*data_p.global_state;
	if (data.offset >= data.entries.size()) {
		// finished returning values
		return;
	}
	// start returning values
	// either fill up the chunk or return all the remaining columns
	idx_t count = 0;
	while (data.offset < data.entries.size() && count < STANDARD_VECTOR_SIZE) {
		auto &entry = data.entries[data.offset].get().Cast<FunctionEntry>();
		bool finished;

		switch (entry.type) {
		case CatalogType::SCALAR_FUNCTION_ENTRY:
			finished = ExtractFunctionData<ScalarFunctionCatalogEntry, ScalarFunctionExtractor>(
			    entry, data.offset_in_entry, output, count);
			break;
		case CatalogType::AGGREGATE_FUNCTION_ENTRY:
			finished = ExtractFunctionData<AggregateFunctionCatalogEntry, AggregateFunctionExtractor>(
			    entry, data.offset_in_entry, output, count);
			break;
		case CatalogType::TABLE_MACRO_ENTRY:
			finished = ExtractFunctionData<TableMacroCatalogEntry, TableMacroExtractor>(entry, data.offset_in_entry,
			                                                                            output, count);
			break;

		case CatalogType::MACRO_ENTRY:
			finished = ExtractFunctionData<ScalarMacroCatalogEntry, MacroExtractor>(entry, data.offset_in_entry, output,
			                                                                        count);
			break;
		case CatalogType::TABLE_FUNCTION_ENTRY:
			finished = ExtractFunctionData<TableFunctionCatalogEntry, TableFunctionExtractor>(
			    entry, data.offset_in_entry, output, count);
			break;
		case CatalogType::PRAGMA_FUNCTION_ENTRY:
			finished = ExtractFunctionData<PragmaFunctionCatalogEntry, PragmaFunctionExtractor>(
			    entry, data.offset_in_entry, output, count);
			break;
		default:
			throw InternalException("FIXME: unrecognized function type in duckdb_functions");
		}
		if (finished) {
			// finished with this function, move to the next function
			data.offset++;
			data.offset_in_entry = 0;
		} else {
			// more functions remain
			data.offset_in_entry++;
		}
		count++;
	}
	output.SetCardinality(count);
}

void DuckDBFunctionsFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(
	    TableFunction("duckdb_functions", {}, DuckDBFunctionsFunction, DuckDBFunctionsBind, DuckDBFunctionsInit));
}

} // namespace duckdb











namespace duckdb {

struct DuckDBIndexesData : public GlobalTableFunctionState {
	DuckDBIndexesData() : offset(0) {
	}

	vector<reference<CatalogEntry>> entries;
	idx_t offset;
};

static unique_ptr<FunctionData> DuckDBIndexesBind(ClientContext &context, TableFunctionBindInput &input,
                                                  vector<LogicalType> &return_types, vector<string> &names) {
	names.emplace_back("database_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("database_oid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("schema_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("schema_oid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("index_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("index_oid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("table_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("table_oid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("is_unique");
	return_types.emplace_back(LogicalType::BOOLEAN);

	names.emplace_back("is_primary");
	return_types.emplace_back(LogicalType::BOOLEAN);

	names.emplace_back("expressions");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("sql");
	return_types.emplace_back(LogicalType::VARCHAR);

	return nullptr;
}

unique_ptr<GlobalTableFunctionState> DuckDBIndexesInit(ClientContext &context, TableFunctionInitInput &input) {
	auto result = make_uniq<DuckDBIndexesData>();

	// scan all the schemas for tables and collect them and collect them
	auto schemas = Catalog::GetAllSchemas(context);
	for (auto &schema : schemas) {
		schema.get().Scan(context, CatalogType::INDEX_ENTRY,
		                  [&](CatalogEntry &entry) { result->entries.push_back(entry); });
	};
	return std::move(result);
}

void DuckDBIndexesFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = data_p.global_state->Cast<DuckDBIndexesData>();
	if (data.offset >= data.entries.size()) {
		// finished returning values
		return;
	}
	// start returning values
	// either fill up the chunk or return all the remaining columns
	idx_t count = 0;
	while (data.offset < data.entries.size() && count < STANDARD_VECTOR_SIZE) {
		auto &entry = data.entries[data.offset++].get();

		auto &index = entry.Cast<IndexCatalogEntry>();
		// return values:

		idx_t col = 0;
		// database_name, VARCHAR
		output.SetValue(col++, count, index.catalog.GetName());
		// database_oid, BIGINT
		output.SetValue(col++, count, Value::BIGINT(index.catalog.GetOid()));
		// schema_name, VARCHAR
		output.SetValue(col++, count, Value(index.schema.name));
		// schema_oid, BIGINT
		output.SetValue(col++, count, Value::BIGINT(index.schema.oid));
		// index_name, VARCHAR
		output.SetValue(col++, count, Value(index.name));
		// index_oid, BIGINT
		output.SetValue(col++, count, Value::BIGINT(index.oid));
		// find the table in the catalog
		auto &table_entry =
		    index.schema.catalog.GetEntry<TableCatalogEntry>(context, index.GetSchemaName(), index.GetTableName());
		// table_name, VARCHAR
		output.SetValue(col++, count, Value(table_entry.name));
		// table_oid, BIGINT
		output.SetValue(col++, count, Value::BIGINT(table_entry.oid));
		if (index.index) {
			// is_unique, BOOLEAN
			output.SetValue(col++, count, Value::BOOLEAN(index.index->IsUnique()));
			// is_primary, BOOLEAN
			output.SetValue(col++, count, Value::BOOLEAN(index.index->IsPrimary()));
		} else {
			output.SetValue(col++, count, Value());
			output.SetValue(col++, count, Value());
		}
		// expressions, VARCHAR
		output.SetValue(col++, count, Value());
		// sql, VARCHAR
		auto sql = index.ToSQL();
		output.SetValue(col++, count, sql.empty() ? Value() : Value(std::move(sql)));

		count++;
	}
	output.SetCardinality(count);
}

void DuckDBIndexesFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(TableFunction("duckdb_indexes", {}, DuckDBIndexesFunction, DuckDBIndexesBind, DuckDBIndexesInit));
}

} // namespace duckdb






namespace duckdb {

struct DuckDBKeywordsData : public GlobalTableFunctionState {
	DuckDBKeywordsData() : offset(0) {
	}

	vector<ParserKeyword> entries;
	idx_t offset;
};

static unique_ptr<FunctionData> DuckDBKeywordsBind(ClientContext &context, TableFunctionBindInput &input,
                                                   vector<LogicalType> &return_types, vector<string> &names) {
	names.emplace_back("keyword_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("keyword_category");
	return_types.emplace_back(LogicalType::VARCHAR);

	return nullptr;
}

unique_ptr<GlobalTableFunctionState> DuckDBKeywordsInit(ClientContext &context, TableFunctionInitInput &input) {
	auto result = make_uniq<DuckDBKeywordsData>();
	result->entries = Parser::KeywordList();
	return std::move(result);
}

void DuckDBKeywordsFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = data_p.global_state->Cast<DuckDBKeywordsData>();
	if (data.offset >= data.entries.size()) {
		// finished returning values
		return;
	}
	// start returning values
	// either fill up the chunk or return all the remaining columns
	idx_t count = 0;
	while (data.offset < data.entries.size() && count < STANDARD_VECTOR_SIZE) {
		auto &entry = data.entries[data.offset++];

		// keyword_name, VARCHAR
		output.SetValue(0, count, Value(entry.name));
		// keyword_category, VARCHAR
		string category_name;
		switch (entry.category) {
		case KeywordCategory::KEYWORD_RESERVED:
			category_name = "reserved";
			break;
		case KeywordCategory::KEYWORD_UNRESERVED:
			category_name = "unreserved";
			break;
		case KeywordCategory::KEYWORD_TYPE_FUNC:
			category_name = "type_function";
			break;
		case KeywordCategory::KEYWORD_COL_NAME:
			category_name = "column_name";
			break;
		default:
			throw InternalException("Unrecognized keyword category");
		}
		output.SetValue(1, count, Value(std::move(category_name)));

		count++;
	}
	output.SetCardinality(count);
}

void DuckDBKeywordsFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(
	    TableFunction("duckdb_keywords", {}, DuckDBKeywordsFunction, DuckDBKeywordsBind, DuckDBKeywordsInit));
}

} // namespace duckdb








namespace duckdb {

struct DuckDBSchemasData : public GlobalTableFunctionState {
	DuckDBSchemasData() : offset(0) {
	}

	vector<reference<SchemaCatalogEntry>> entries;
	idx_t offset;
};

static unique_ptr<FunctionData> DuckDBSchemasBind(ClientContext &context, TableFunctionBindInput &input,
                                                  vector<LogicalType> &return_types, vector<string> &names) {
	names.emplace_back("oid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("database_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("database_oid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("schema_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("internal");
	return_types.emplace_back(LogicalType::BOOLEAN);

	names.emplace_back("sql");
	return_types.emplace_back(LogicalType::VARCHAR);

	return nullptr;
}

unique_ptr<GlobalTableFunctionState> DuckDBSchemasInit(ClientContext &context, TableFunctionInitInput &input) {
	auto result = make_uniq<DuckDBSchemasData>();

	// scan all the schemas and collect them
	result->entries = Catalog::GetAllSchemas(context);

	return std::move(result);
}

void DuckDBSchemasFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = data_p.global_state->Cast<DuckDBSchemasData>();
	if (data.offset >= data.entries.size()) {
		// finished returning values
		return;
	}
	// start returning values
	// either fill up the chunk or return all the remaining columns
	idx_t count = 0;
	while (data.offset < data.entries.size() && count < STANDARD_VECTOR_SIZE) {
		auto &entry = data.entries[data.offset].get();

		// return values:
		idx_t col = 0;
		// "oid", PhysicalType::BIGINT
		output.SetValue(col++, count, Value::BIGINT(entry.oid));
		// database_name, VARCHAR
		output.SetValue(col++, count, entry.catalog.GetName());
		// database_oid, BIGINT
		output.SetValue(col++, count, Value::BIGINT(entry.catalog.GetOid()));
		// "schema_name", PhysicalType::VARCHAR
		output.SetValue(col++, count, Value(entry.name));
		// "internal", PhysicalType::BOOLEAN
		output.SetValue(col++, count, Value::BOOLEAN(entry.internal));
		// "sql", PhysicalType::VARCHAR
		output.SetValue(col++, count, Value());

		data.offset++;
		count++;
	}
	output.SetCardinality(count);
}

void DuckDBSchemasFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(TableFunction("duckdb_schemas", {}, DuckDBSchemasFunction, DuckDBSchemasBind, DuckDBSchemasInit));
}

} // namespace duckdb









namespace duckdb {

struct DuckDBSequencesData : public GlobalTableFunctionState {
	DuckDBSequencesData() : offset(0) {
	}

	vector<reference<SequenceCatalogEntry>> entries;
	idx_t offset;
};

static unique_ptr<FunctionData> DuckDBSequencesBind(ClientContext &context, TableFunctionBindInput &input,
                                                    vector<LogicalType> &return_types, vector<string> &names) {
	names.emplace_back("database_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("database_oid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("schema_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("schema_oid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("sequence_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("sequence_oid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("temporary");
	return_types.emplace_back(LogicalType::BOOLEAN);

	names.emplace_back("start_value");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("min_value");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("max_value");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("increment_by");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("cycle");
	return_types.emplace_back(LogicalType::BOOLEAN);

	names.emplace_back("last_value");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("sql");
	return_types.emplace_back(LogicalType::VARCHAR);

	return nullptr;
}

unique_ptr<GlobalTableFunctionState> DuckDBSequencesInit(ClientContext &context, TableFunctionInitInput &input) {
	auto result = make_uniq<DuckDBSequencesData>();

	// scan all the schemas for tables and collect themand collect them
	auto schemas = Catalog::GetAllSchemas(context);
	for (auto &schema : schemas) {
		schema.get().Scan(context, CatalogType::SEQUENCE_ENTRY,
		                  [&](CatalogEntry &entry) { result->entries.push_back(entry.Cast<SequenceCatalogEntry>()); });
	};
	return std::move(result);
}

void DuckDBSequencesFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = data_p.global_state->Cast<DuckDBSequencesData>();
	if (data.offset >= data.entries.size()) {
		// finished returning values
		return;
	}
	// start returning values
	// either fill up the chunk or return all the remaining columns
	idx_t count = 0;
	while (data.offset < data.entries.size() && count < STANDARD_VECTOR_SIZE) {
		auto &seq = data.entries[data.offset++].get();

		// return values:
		idx_t col = 0;
		// database_name, VARCHAR
		output.SetValue(col++, count, seq.catalog.GetName());
		// database_oid, BIGINT
		output.SetValue(col++, count, Value::BIGINT(seq.catalog.GetOid()));
		// schema_name, VARCHAR
		output.SetValue(col++, count, Value(seq.schema.name));
		// schema_oid, BIGINT
		output.SetValue(col++, count, Value::BIGINT(seq.schema.oid));
		// sequence_name, VARCHAR
		output.SetValue(col++, count, Value(seq.name));
		// sequence_oid, BIGINT
		output.SetValue(col++, count, Value::BIGINT(seq.oid));
		// temporary, BOOLEAN
		output.SetValue(col++, count, Value::BOOLEAN(seq.temporary));
		// start_value, BIGINT
		output.SetValue(col++, count, Value::BIGINT(seq.start_value));
		// min_value, BIGINT
		output.SetValue(col++, count, Value::BIGINT(seq.min_value));
		// max_value, BIGINT
		output.SetValue(col++, count, Value::BIGINT(seq.max_value));
		// increment_by, BIGINT
		output.SetValue(col++, count, Value::BIGINT(seq.increment));
		// cycle, BOOLEAN
		output.SetValue(col++, count, Value::BOOLEAN(seq.cycle));
		// last_value, BIGINT
		output.SetValue(col++, count, seq.usage_count == 0 ? Value() : Value::BOOLEAN(seq.last_value));
		// sql, LogicalType::VARCHAR
		output.SetValue(col++, count, Value(seq.ToSQL()));

		count++;
	}
	output.SetCardinality(count);
}

void DuckDBSequencesFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(
	    TableFunction("duckdb_sequences", {}, DuckDBSequencesFunction, DuckDBSequencesBind, DuckDBSequencesInit));
}

} // namespace duckdb






namespace duckdb {

struct DuckDBSettingValue {
	string name;
	string value;
	string description;
	string input_type;
};

struct DuckDBSettingsData : public GlobalTableFunctionState {
	DuckDBSettingsData() : offset(0) {
	}

	vector<DuckDBSettingValue> settings;
	idx_t offset;
};

static unique_ptr<FunctionData> DuckDBSettingsBind(ClientContext &context, TableFunctionBindInput &input,
                                                   vector<LogicalType> &return_types, vector<string> &names) {
	names.emplace_back("name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("value");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("description");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("input_type");
	return_types.emplace_back(LogicalType::VARCHAR);

	return nullptr;
}

unique_ptr<GlobalTableFunctionState> DuckDBSettingsInit(ClientContext &context, TableFunctionInitInput &input) {
	auto result = make_uniq<DuckDBSettingsData>();

	auto &config = DBConfig::GetConfig(context);
	auto options_count = DBConfig::GetOptionCount();
	for (idx_t i = 0; i < options_count; i++) {
		auto option = DBConfig::GetOptionByIndex(i);
		D_ASSERT(option);
		DuckDBSettingValue value;
		value.name = option->name;
		value.value = option->get_setting(context).ToString();
		value.description = option->description;
		value.input_type = EnumUtil::ToString(option->parameter_type);

		result->settings.push_back(std::move(value));
	}
	for (auto &ext_param : config.extension_parameters) {
		Value setting_val;
		string setting_str_val;
		if (context.TryGetCurrentSetting(ext_param.first, setting_val)) {
			setting_str_val = setting_val.ToString();
		}
		DuckDBSettingValue value;
		value.name = ext_param.first;
		value.value = std::move(setting_str_val);
		value.description = ext_param.second.description;
		value.input_type = ext_param.second.type.ToString();

		result->settings.push_back(std::move(value));
	}
	return std::move(result);
}

void DuckDBSettingsFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = data_p.global_state->Cast<DuckDBSettingsData>();
	if (data.offset >= data.settings.size()) {
		// finished returning values
		return;
	}
	// start returning values
	// either fill up the chunk or return all the remaining columns
	idx_t count = 0;
	while (data.offset < data.settings.size() && count < STANDARD_VECTOR_SIZE) {
		auto &entry = data.settings[data.offset++];

		// return values:
		// name, LogicalType::VARCHAR
		output.SetValue(0, count, Value(entry.name));
		// value, LogicalType::VARCHAR
		output.SetValue(1, count, Value(entry.value));
		// description, LogicalType::VARCHAR
		output.SetValue(2, count, Value(entry.description));
		// input_type, LogicalType::VARCHAR
		output.SetValue(3, count, Value(entry.input_type));
		count++;
	}
	output.SetCardinality(count);
}

void DuckDBSettingsFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(
	    TableFunction("duckdb_settings", {}, DuckDBSettingsFunction, DuckDBSettingsBind, DuckDBSettingsInit));
}

} // namespace duckdb













namespace duckdb {

struct DuckDBTablesData : public GlobalTableFunctionState {
	DuckDBTablesData() : offset(0) {
	}

	vector<reference<CatalogEntry>> entries;
	idx_t offset;
};

static unique_ptr<FunctionData> DuckDBTablesBind(ClientContext &context, TableFunctionBindInput &input,
                                                 vector<LogicalType> &return_types, vector<string> &names) {
	names.emplace_back("database_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("database_oid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("schema_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("schema_oid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("table_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("table_oid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("internal");
	return_types.emplace_back(LogicalType::BOOLEAN);

	names.emplace_back("temporary");
	return_types.emplace_back(LogicalType::BOOLEAN);

	names.emplace_back("has_primary_key");
	return_types.emplace_back(LogicalType::BOOLEAN);

	names.emplace_back("estimated_size");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("column_count");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("index_count");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("check_constraint_count");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("sql");
	return_types.emplace_back(LogicalType::VARCHAR);

	return nullptr;
}

unique_ptr<GlobalTableFunctionState> DuckDBTablesInit(ClientContext &context, TableFunctionInitInput &input) {
	auto result = make_uniq<DuckDBTablesData>();

	// scan all the schemas for tables and collect themand collect them
	auto schemas = Catalog::GetAllSchemas(context);
	for (auto &schema : schemas) {
		schema.get().Scan(context, CatalogType::TABLE_ENTRY,
		                  [&](CatalogEntry &entry) { result->entries.push_back(entry); });
	};
	return std::move(result);
}

static bool TableHasPrimaryKey(TableCatalogEntry &table) {
	for (auto &constraint : table.GetConstraints()) {
		if (constraint->type == ConstraintType::UNIQUE) {
			auto &unique = constraint->Cast<UniqueConstraint>();
			if (unique.is_primary_key) {
				return true;
			}
		}
	}
	return false;
}

static idx_t CheckConstraintCount(TableCatalogEntry &table) {
	idx_t check_count = 0;
	for (auto &constraint : table.GetConstraints()) {
		if (constraint->type == ConstraintType::CHECK) {
			check_count++;
		}
	}
	return check_count;
}

void DuckDBTablesFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = data_p.global_state->Cast<DuckDBTablesData>();
	if (data.offset >= data.entries.size()) {
		// finished returning values
		return;
	}
	// start returning values
	// either fill up the chunk or return all the remaining columns
	idx_t count = 0;
	while (data.offset < data.entries.size() && count < STANDARD_VECTOR_SIZE) {
		auto &entry = data.entries[data.offset++].get();

		if (entry.type != CatalogType::TABLE_ENTRY) {
			continue;
		}
		auto &table = entry.Cast<TableCatalogEntry>();
		auto storage_info = table.GetStorageInfo(context);
		// return values:
		idx_t col = 0;
		// database_name, VARCHAR
		output.SetValue(col++, count, table.catalog.GetName());
		// database_oid, BIGINT
		output.SetValue(col++, count, Value::BIGINT(table.catalog.GetOid()));
		// schema_name, LogicalType::VARCHAR
		output.SetValue(col++, count, Value(table.schema.name));
		// schema_oid, LogicalType::BIGINT
		output.SetValue(col++, count, Value::BIGINT(table.schema.oid));
		// table_name, LogicalType::VARCHAR
		output.SetValue(col++, count, Value(table.name));
		// table_oid, LogicalType::BIGINT
		output.SetValue(col++, count, Value::BIGINT(table.oid));
		// internal, LogicalType::BOOLEAN
		output.SetValue(col++, count, Value::BOOLEAN(table.internal));
		// temporary, LogicalType::BOOLEAN
		output.SetValue(col++, count, Value::BOOLEAN(table.temporary));
		// has_primary_key, LogicalType::BOOLEAN
		output.SetValue(col++, count, Value::BOOLEAN(TableHasPrimaryKey(table)));
		// estimated_size, LogicalType::BIGINT
		Value card_val =
		    storage_info.cardinality == DConstants::INVALID_INDEX ? Value() : Value::BIGINT(storage_info.cardinality);
		output.SetValue(col++, count, card_val);
		// column_count, LogicalType::BIGINT
		output.SetValue(col++, count, Value::BIGINT(table.GetColumns().LogicalColumnCount()));
		// index_count, LogicalType::BIGINT
		output.SetValue(col++, count, Value::BIGINT(storage_info.index_info.size()));
		// check_constraint_count, LogicalType::BIGINT
		output.SetValue(col++, count, Value::BIGINT(CheckConstraintCount(table)));
		// sql, LogicalType::VARCHAR
		output.SetValue(col++, count, Value(table.ToSQL()));

		count++;
	}
	output.SetCardinality(count);
}

void DuckDBTablesFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(TableFunction("duckdb_tables", {}, DuckDBTablesFunction, DuckDBTablesBind, DuckDBTablesInit));
}

} // namespace duckdb



namespace duckdb {

struct DuckDBTemporaryFilesData : public GlobalTableFunctionState {
	DuckDBTemporaryFilesData() : offset(0) {
	}

	vector<TemporaryFileInformation> entries;
	idx_t offset;
};

static unique_ptr<FunctionData> DuckDBTemporaryFilesBind(ClientContext &context, TableFunctionBindInput &input,
                                                         vector<LogicalType> &return_types, vector<string> &names) {
	names.emplace_back("path");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("size");
	return_types.emplace_back(LogicalType::BIGINT);

	return nullptr;
}

unique_ptr<GlobalTableFunctionState> DuckDBTemporaryFilesInit(ClientContext &context, TableFunctionInitInput &input) {
	auto result = make_uniq<DuckDBTemporaryFilesData>();

	result->entries = BufferManager::GetBufferManager(context).GetTemporaryFiles();
	return std::move(result);
}

void DuckDBTemporaryFilesFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = data_p.global_state->Cast<DuckDBTemporaryFilesData>();
	if (data.offset >= data.entries.size()) {
		// finished returning values
		return;
	}
	// start returning values
	// either fill up the chunk or return all the remaining columns
	idx_t count = 0;
	while (data.offset < data.entries.size() && count < STANDARD_VECTOR_SIZE) {
		auto &entry = data.entries[data.offset++];
		// return values:
		idx_t col = 0;
		// database_name, VARCHAR
		output.SetValue(col++, count, entry.path);
		// database_oid, BIGINT
		output.SetValue(col++, count, Value::BIGINT(entry.size));
		count++;
	}
	output.SetCardinality(count);
}

void DuckDBTemporaryFilesFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(TableFunction("duckdb_temporary_files", {}, DuckDBTemporaryFilesFunction, DuckDBTemporaryFilesBind,
	                              DuckDBTemporaryFilesInit));
}

} // namespace duckdb










namespace duckdb {

struct DuckDBTypesData : public GlobalTableFunctionState {
	DuckDBTypesData() : offset(0) {
	}

	vector<reference<TypeCatalogEntry>> entries;
	idx_t offset;
	unordered_set<int64_t> oids;
};

static unique_ptr<FunctionData> DuckDBTypesBind(ClientContext &context, TableFunctionBindInput &input,
                                                vector<LogicalType> &return_types, vector<string> &names) {
	names.emplace_back("database_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("database_oid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("schema_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("schema_oid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("type_oid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("type_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("type_size");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("logical_type");
	return_types.emplace_back(LogicalType::VARCHAR);

	// NUMERIC, STRING, DATETIME, BOOLEAN, COMPOSITE, USER
	names.emplace_back("type_category");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("internal");
	return_types.emplace_back(LogicalType::BOOLEAN);

	names.emplace_back("labels");
	return_types.emplace_back(LogicalType::LIST(LogicalType::VARCHAR));

	return nullptr;
}

unique_ptr<GlobalTableFunctionState> DuckDBTypesInit(ClientContext &context, TableFunctionInitInput &input) {
	auto result = make_uniq<DuckDBTypesData>();
	auto schemas = Catalog::GetAllSchemas(context);
	for (auto &schema : schemas) {
		schema.get().Scan(context, CatalogType::TYPE_ENTRY,
		                  [&](CatalogEntry &entry) { result->entries.push_back(entry.Cast<TypeCatalogEntry>()); });
	};
	return std::move(result);
}

void DuckDBTypesFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = data_p.global_state->Cast<DuckDBTypesData>();
	if (data.offset >= data.entries.size()) {
		// finished returning values
		return;
	}
	// start returning values
	// either fill up the chunk or return all the remaining columns
	idx_t count = 0;
	while (data.offset < data.entries.size() && count < STANDARD_VECTOR_SIZE) {
		auto &type_entry = data.entries[data.offset++].get();
		auto &type = type_entry.user_type;

		// return values:
		idx_t col = 0;
		// database_name, VARCHAR
		output.SetValue(col++, count, type_entry.catalog.GetName());
		// database_oid, BIGINT
		output.SetValue(col++, count, Value::BIGINT(type_entry.catalog.GetOid()));
		// schema_name, LogicalType::VARCHAR
		output.SetValue(col++, count, Value(type_entry.schema.name));
		// schema_oid, LogicalType::BIGINT
		output.SetValue(col++, count, Value::BIGINT(type_entry.schema.oid));
		// type_oid, BIGINT
		int64_t oid;
		if (type_entry.internal) {
			oid = int64_t(type.id());
		} else {
			oid = type_entry.oid;
		}
		Value oid_val;
		if (data.oids.find(oid) == data.oids.end()) {
			data.oids.insert(oid);
			oid_val = Value::BIGINT(oid);
		} else {
			oid_val = Value();
		}
		output.SetValue(col++, count, oid_val);
		// type_name, VARCHAR
		output.SetValue(col++, count, Value(type_entry.name));
		// type_size, BIGINT
		auto internal_type = type.InternalType();
		output.SetValue(col++, count,
		                internal_type == PhysicalType::INVALID ? Value() : Value::BIGINT(GetTypeIdSize(internal_type)));
		// logical_type, VARCHAR
		output.SetValue(col++, count, Value(EnumUtil::ToString(type.id())));
		// type_category, VARCHAR
		string category;
		switch (type.id()) {
		case LogicalTypeId::TINYINT:
		case LogicalTypeId::SMALLINT:
		case LogicalTypeId::INTEGER:
		case LogicalTypeId::BIGINT:
		case LogicalTypeId::DECIMAL:
		case LogicalTypeId::FLOAT:
		case LogicalTypeId::DOUBLE:
		case LogicalTypeId::UTINYINT:
		case LogicalTypeId::USMALLINT:
		case LogicalTypeId::UINTEGER:
		case LogicalTypeId::UBIGINT:
		case LogicalTypeId::HUGEINT:
			category = "NUMERIC";
			break;
		case LogicalTypeId::DATE:
		case LogicalTypeId::TIME:
		case LogicalTypeId::TIMESTAMP_SEC:
		case LogicalTypeId::TIMESTAMP_MS:
		case LogicalTypeId::TIMESTAMP:
		case LogicalTypeId::TIMESTAMP_NS:
		case LogicalTypeId::INTERVAL:
		case LogicalTypeId::TIME_TZ:
		case LogicalTypeId::TIMESTAMP_TZ:
			category = "DATETIME";
			break;
		case LogicalTypeId::CHAR:
		case LogicalTypeId::VARCHAR:
			category = "STRING";
			break;
		case LogicalTypeId::BOOLEAN:
			category = "BOOLEAN";
			break;
		case LogicalTypeId::STRUCT:
		case LogicalTypeId::LIST:
		case LogicalTypeId::MAP:
		case LogicalTypeId::UNION:
			category = "COMPOSITE";
			break;
		default:
			break;
		}
		output.SetValue(col++, count, category.empty() ? Value() : Value(category));
		// internal, BOOLEAN
		output.SetValue(col++, count, Value::BOOLEAN(type_entry.internal));
		// labels, VARCHAR[]
		if (type.id() == LogicalTypeId::ENUM && type.AuxInfo()) {
			auto data = FlatVector::GetData<string_t>(EnumType::GetValuesInsertOrder(type));
			idx_t size = EnumType::GetSize(type);

			vector<Value> labels;
			for (idx_t i = 0; i < size; i++) {
				labels.emplace_back(data[i]);
			}

			output.SetValue(col++, count, Value::LIST(labels));
		} else {
			output.SetValue(col++, count, Value());
		}

		count++;
	}
	output.SetCardinality(count);
}

void DuckDBTypesFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(TableFunction("duckdb_types", {}, DuckDBTypesFunction, DuckDBTypesBind, DuckDBTypesInit));
}

} // namespace duckdb









namespace duckdb {

struct DuckDBViewsData : public GlobalTableFunctionState {
	DuckDBViewsData() : offset(0) {
	}

	vector<reference<CatalogEntry>> entries;
	idx_t offset;
};

static unique_ptr<FunctionData> DuckDBViewsBind(ClientContext &context, TableFunctionBindInput &input,
                                                vector<LogicalType> &return_types, vector<string> &names) {
	names.emplace_back("database_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("database_oid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("schema_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("schema_oid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("view_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("view_oid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("internal");
	return_types.emplace_back(LogicalType::BOOLEAN);

	names.emplace_back("temporary");
	return_types.emplace_back(LogicalType::BOOLEAN);

	names.emplace_back("column_count");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("sql");
	return_types.emplace_back(LogicalType::VARCHAR);

	return nullptr;
}

unique_ptr<GlobalTableFunctionState> DuckDBViewsInit(ClientContext &context, TableFunctionInitInput &input) {
	auto result = make_uniq<DuckDBViewsData>();

	// scan all the schemas for tables and collect them and collect them
	auto schemas = Catalog::GetAllSchemas(context);
	for (auto &schema : schemas) {
		schema.get().Scan(context, CatalogType::VIEW_ENTRY,
		                  [&](CatalogEntry &entry) { result->entries.push_back(entry); });
	};
	return std::move(result);
}

void DuckDBViewsFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = data_p.global_state->Cast<DuckDBViewsData>();
	if (data.offset >= data.entries.size()) {
		// finished returning values
		return;
	}
	// start returning values
	// either fill up the chunk or return all the remaining columns
	idx_t count = 0;
	while (data.offset < data.entries.size() && count < STANDARD_VECTOR_SIZE) {
		auto &entry = data.entries[data.offset++].get();

		if (entry.type != CatalogType::VIEW_ENTRY) {
			continue;
		}
		auto &view = entry.Cast<ViewCatalogEntry>();

		// return values:
		idx_t col = 0;
		// database_name, VARCHAR
		output.SetValue(col++, count, view.catalog.GetName());
		// database_oid, BIGINT
		output.SetValue(col++, count, Value::BIGINT(view.catalog.GetOid()));
		// schema_name, LogicalType::VARCHAR
		output.SetValue(col++, count, Value(view.schema.name));
		// schema_oid, LogicalType::BIGINT
		output.SetValue(col++, count, Value::BIGINT(view.schema.oid));
		// view_name, LogicalType::VARCHAR
		output.SetValue(col++, count, Value(view.name));
		// view_oid, LogicalType::BIGINT
		output.SetValue(col++, count, Value::BIGINT(view.oid));
		// internal, LogicalType::BOOLEAN
		output.SetValue(col++, count, Value::BOOLEAN(view.internal));
		// temporary, LogicalType::BOOLEAN
		output.SetValue(col++, count, Value::BOOLEAN(view.temporary));
		// column_count, LogicalType::BIGINT
		output.SetValue(col++, count, Value::BIGINT(view.types.size()));
		// sql, LogicalType::VARCHAR
		output.SetValue(col++, count, Value(view.ToSQL()));

		count++;
	}
	output.SetCardinality(count);
}

void DuckDBViewsFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(TableFunction("duckdb_views", {}, DuckDBViewsFunction, DuckDBViewsBind, DuckDBViewsInit));
}

} // namespace duckdb







namespace duckdb {

struct PragmaCollateData : public GlobalTableFunctionState {
	PragmaCollateData() : offset(0) {
	}

	vector<string> entries;
	idx_t offset;
};

static unique_ptr<FunctionData> PragmaCollateBind(ClientContext &context, TableFunctionBindInput &input,
                                                  vector<LogicalType> &return_types, vector<string> &names) {
	names.emplace_back("collname");
	return_types.emplace_back(LogicalType::VARCHAR);

	return nullptr;
}

unique_ptr<GlobalTableFunctionState> PragmaCollateInit(ClientContext &context, TableFunctionInitInput &input) {
	auto result = make_uniq<PragmaCollateData>();

	auto schemas = Catalog::GetAllSchemas(context);
	for (auto schema : schemas) {
		schema.get().Scan(context, CatalogType::COLLATION_ENTRY,
		                  [&](CatalogEntry &entry) { result->entries.push_back(entry.name); });
	}
	return std::move(result);
}

static void PragmaCollateFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = data_p.global_state->Cast<PragmaCollateData>();
	if (data.offset >= data.entries.size()) {
		// finished returning values
		return;
	}
	idx_t next = MinValue<idx_t>(data.offset + STANDARD_VECTOR_SIZE, data.entries.size());
	output.SetCardinality(next - data.offset);
	for (idx_t i = data.offset; i < next; i++) {
		auto index = i - data.offset;
		output.SetValue(0, index, Value(data.entries[i]));
	}

	data.offset = next;
}

void PragmaCollations::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(
	    TableFunction("pragma_collations", {}, PragmaCollateFunction, PragmaCollateBind, PragmaCollateInit));
}

} // namespace duckdb










namespace duckdb {

struct PragmaDatabaseSizeData : public GlobalTableFunctionState {
	PragmaDatabaseSizeData() : index(0) {
	}

	idx_t index;
	vector<reference<AttachedDatabase>> databases;
	Value memory_usage;
	Value memory_limit;
};

static unique_ptr<FunctionData> PragmaDatabaseSizeBind(ClientContext &context, TableFunctionBindInput &input,
                                                       vector<LogicalType> &return_types, vector<string> &names) {
	names.emplace_back("database_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("database_size");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("block_size");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("total_blocks");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("used_blocks");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("free_blocks");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("wal_size");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("memory_usage");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("memory_limit");
	return_types.emplace_back(LogicalType::VARCHAR);

	return nullptr;
}

unique_ptr<GlobalTableFunctionState> PragmaDatabaseSizeInit(ClientContext &context, TableFunctionInitInput &input) {
	auto result = make_uniq<PragmaDatabaseSizeData>();
	result->databases = DatabaseManager::Get(context).GetDatabases(context);
	auto &buffer_manager = BufferManager::GetBufferManager(context);
	result->memory_usage = Value(StringUtil::BytesToHumanReadableString(buffer_manager.GetUsedMemory()));
	auto max_memory = buffer_manager.GetMaxMemory();
	result->memory_limit =
	    max_memory == (idx_t)-1 ? Value("Unlimited") : Value(StringUtil::BytesToHumanReadableString(max_memory));

	return std::move(result);
}

void PragmaDatabaseSizeFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = data_p.global_state->Cast<PragmaDatabaseSizeData>();
	idx_t row = 0;
	for (; data.index < data.databases.size() && row < STANDARD_VECTOR_SIZE; data.index++) {
		auto &db = data.databases[data.index].get();
		if (db.IsSystem() || db.IsTemporary()) {
			continue;
		}
		auto ds = db.GetCatalog().GetDatabaseSize(context);
		idx_t col = 0;
		output.data[col++].SetValue(row, Value(db.GetName()));
		output.data[col++].SetValue(row, Value(StringUtil::BytesToHumanReadableString(ds.bytes)));
		output.data[col++].SetValue(row, Value::BIGINT(ds.block_size));
		output.data[col++].SetValue(row, Value::BIGINT(ds.total_blocks));
		output.data[col++].SetValue(row, Value::BIGINT(ds.used_blocks));
		output.data[col++].SetValue(row, Value::BIGINT(ds.free_blocks));
		output.data[col++].SetValue(
		    row, ds.wal_size == idx_t(-1) ? Value() : Value(StringUtil::BytesToHumanReadableString(ds.wal_size)));
		output.data[col++].SetValue(row, data.memory_usage);
		output.data[col++].SetValue(row, data.memory_limit);
		row++;
	}
	output.SetCardinality(row);
}

void PragmaDatabaseSize::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(TableFunction("pragma_database_size", {}, PragmaDatabaseSizeFunction, PragmaDatabaseSizeBind,
	                              PragmaDatabaseSizeInit));
}

} // namespace duckdb















#include <algorithm>

namespace duckdb {

struct PragmaStorageFunctionData : public TableFunctionData {
	explicit PragmaStorageFunctionData(TableCatalogEntry &table_entry) : table_entry(table_entry) {
	}

	TableCatalogEntry &table_entry;
	TableStorageInfo storage_info;
};

struct PragmaStorageOperatorData : public GlobalTableFunctionState {
	PragmaStorageOperatorData() : offset(0) {
	}

	idx_t offset;
};

static unique_ptr<FunctionData> PragmaStorageInfoBind(ClientContext &context, TableFunctionBindInput &input,
                                                      vector<LogicalType> &return_types, vector<string> &names) {
	names.emplace_back("row_group_id");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("column_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("column_id");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("column_path");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("segment_id");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("segment_type");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("start");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("count");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("compression");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("stats");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("has_updates");
	return_types.emplace_back(LogicalType::BOOLEAN);

	names.emplace_back("persistent");
	return_types.emplace_back(LogicalType::BOOLEAN);

	names.emplace_back("block_id");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("block_offset");
	return_types.emplace_back(LogicalType::BIGINT);

	auto qname = QualifiedName::Parse(input.inputs[0].GetValue<string>());

	// look up the table name in the catalog
	Binder::BindSchemaOrCatalog(context, qname.catalog, qname.schema);
	auto &table_entry = Catalog::GetEntry<TableCatalogEntry>(context, qname.catalog, qname.schema, qname.name);
	auto result = make_uniq<PragmaStorageFunctionData>(table_entry);
	result->storage_info = table_entry.GetStorageInfo(context);
	return std::move(result);
}

unique_ptr<GlobalTableFunctionState> PragmaStorageInfoInit(ClientContext &context, TableFunctionInitInput &input) {
	return make_uniq<PragmaStorageOperatorData>();
}

static void PragmaStorageInfoFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind_data = data_p.bind_data->Cast<PragmaStorageFunctionData>();
	auto &data = data_p.global_state->Cast<PragmaStorageOperatorData>();
	idx_t count = 0;
	auto &columns = bind_data.table_entry.GetColumns();
	while (data.offset < bind_data.storage_info.column_segments.size() && count < STANDARD_VECTOR_SIZE) {
		auto &entry = bind_data.storage_info.column_segments[data.offset++];

		idx_t col_idx = 0;
		// row_group_id
		output.SetValue(col_idx++, count, Value::BIGINT(entry.row_group_index));
		// column_name
		auto &col = columns.GetColumn(PhysicalIndex(entry.column_id));
		output.SetValue(col_idx++, count, Value(col.Name()));
		// column_id
		output.SetValue(col_idx++, count, Value::BIGINT(entry.column_id));
		// column_path
		output.SetValue(col_idx++, count, Value(entry.column_path));
		// segment_id
		output.SetValue(col_idx++, count, Value::BIGINT(entry.segment_idx));
		// segment_type
		output.SetValue(col_idx++, count, Value(entry.segment_type));
		// start
		output.SetValue(col_idx++, count, Value::BIGINT(entry.segment_start));
		// count
		output.SetValue(col_idx++, count, Value::BIGINT(entry.segment_count));
		// compression
		output.SetValue(col_idx++, count, Value(entry.compression_type));
		// stats
		output.SetValue(col_idx++, count, Value(entry.segment_stats));
		// has_updates
		output.SetValue(col_idx++, count, Value::BOOLEAN(entry.has_updates));
		// persistent
		output.SetValue(col_idx++, count, Value::BOOLEAN(entry.persistent));
		// block_id
		// block_offset
		if (entry.persistent) {
			output.SetValue(col_idx++, count, Value::BIGINT(entry.block_id));
			output.SetValue(col_idx++, count, Value::BIGINT(entry.block_offset));
		} else {
			output.SetValue(col_idx++, count, Value());
			output.SetValue(col_idx++, count, Value());
		}
		count++;
	}
	output.SetCardinality(count);
}

void PragmaStorageInfo::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(TableFunction("pragma_storage_info", {LogicalType::VARCHAR}, PragmaStorageInfoFunction,
	                              PragmaStorageInfoBind, PragmaStorageInfoInit));
}

} // namespace duckdb














#include <algorithm>

namespace duckdb {

struct PragmaTableFunctionData : public TableFunctionData {
	explicit PragmaTableFunctionData(CatalogEntry &entry_p) : entry(entry_p) {
	}

	CatalogEntry &entry;
};

struct PragmaTableOperatorData : public GlobalTableFunctionState {
	PragmaTableOperatorData() : offset(0) {
	}
	idx_t offset;
};

static unique_ptr<FunctionData> PragmaTableInfoBind(ClientContext &context, TableFunctionBindInput &input,
                                                    vector<LogicalType> &return_types, vector<string> &names) {

	names.emplace_back("cid");
	return_types.emplace_back(LogicalType::INTEGER);

	names.emplace_back("name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("type");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("notnull");
	return_types.emplace_back(LogicalType::BOOLEAN);

	names.emplace_back("dflt_value");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("pk");
	return_types.emplace_back(LogicalType::BOOLEAN);

	auto qname = QualifiedName::Parse(input.inputs[0].GetValue<string>());

	// look up the table name in the catalog
	Binder::BindSchemaOrCatalog(context, qname.catalog, qname.schema);
	auto &entry = Catalog::GetEntry(context, CatalogType::TABLE_ENTRY, qname.catalog, qname.schema, qname.name);
	return make_uniq<PragmaTableFunctionData>(entry);
}

unique_ptr<GlobalTableFunctionState> PragmaTableInfoInit(ClientContext &context, TableFunctionInitInput &input) {
	return make_uniq<PragmaTableOperatorData>();
}

static void CheckConstraints(TableCatalogEntry &table, const ColumnDefinition &column, bool &out_not_null,
                             bool &out_pk) {
	out_not_null = false;
	out_pk = false;
	// check all constraints
	// FIXME: this is pretty inefficient, it probably doesn't matter
	for (auto &constraint : table.GetConstraints()) {
		switch (constraint->type) {
		case ConstraintType::NOT_NULL: {
			auto &not_null = constraint->Cast<NotNullConstraint>();
			if (not_null.index == column.Logical()) {
				out_not_null = true;
			}
			break;
		}
		case ConstraintType::UNIQUE: {
			auto &unique = constraint->Cast<UniqueConstraint>();
			if (unique.is_primary_key) {
				if (unique.index == column.Logical()) {
					out_pk = true;
				}
				if (std::find(unique.columns.begin(), unique.columns.end(), column.GetName()) != unique.columns.end()) {
					out_pk = true;
				}
			}
			break;
		}
		default:
			break;
		}
	}
}

static void PragmaTableInfoTable(PragmaTableOperatorData &data, TableCatalogEntry &table, DataChunk &output) {
	if (data.offset >= table.GetColumns().LogicalColumnCount()) {
		// finished returning values
		return;
	}
	// start returning values
	// either fill up the chunk or return all the remaining columns
	idx_t next = MinValue<idx_t>(data.offset + STANDARD_VECTOR_SIZE, table.GetColumns().LogicalColumnCount());
	output.SetCardinality(next - data.offset);

	for (idx_t i = data.offset; i < next; i++) {
		bool not_null, pk;
		auto index = i - data.offset;
		auto &column = table.GetColumn(LogicalIndex(i));
		D_ASSERT(column.Oid() < (idx_t)NumericLimits<int32_t>::Maximum());
		CheckConstraints(table, column, not_null, pk);

		// return values:
		// "cid", PhysicalType::INT32
		output.SetValue(0, index, Value::INTEGER((int32_t)column.Oid()));
		// "name", PhysicalType::VARCHAR
		output.SetValue(1, index, Value(column.Name()));
		// "type", PhysicalType::VARCHAR
		output.SetValue(2, index, Value(column.Type().ToString()));
		// "notnull", PhysicalType::BOOL
		output.SetValue(3, index, Value::BOOLEAN(not_null));
		// "dflt_value", PhysicalType::VARCHAR
		Value def_value = column.DefaultValue() ? Value(column.DefaultValue()->ToString()) : Value();
		output.SetValue(4, index, def_value);
		// "pk", PhysicalType::BOOL
		output.SetValue(5, index, Value::BOOLEAN(pk));
	}
	data.offset = next;
}

static void PragmaTableInfoView(PragmaTableOperatorData &data, ViewCatalogEntry &view, DataChunk &output) {
	if (data.offset >= view.types.size()) {
		// finished returning values
		return;
	}
	// start returning values
	// either fill up the chunk or return all the remaining columns
	idx_t next = MinValue<idx_t>(data.offset + STANDARD_VECTOR_SIZE, view.types.size());
	output.SetCardinality(next - data.offset);

	for (idx_t i = data.offset; i < next; i++) {
		auto index = i - data.offset;
		auto type = view.types[i];
		auto &name = view.aliases[i];
		// return values:
		// "cid", PhysicalType::INT32

		output.SetValue(0, index, Value::INTEGER((int32_t)i));
		// "name", PhysicalType::VARCHAR
		output.SetValue(1, index, Value(name));
		// "type", PhysicalType::VARCHAR
		output.SetValue(2, index, Value(type.ToString()));
		// "notnull", PhysicalType::BOOL
		output.SetValue(3, index, Value::BOOLEAN(false));
		// "dflt_value", PhysicalType::VARCHAR
		output.SetValue(4, index, Value());
		// "pk", PhysicalType::BOOL
		output.SetValue(5, index, Value::BOOLEAN(false));
	}
	data.offset = next;
}

static void PragmaTableInfoFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind_data = data_p.bind_data->Cast<PragmaTableFunctionData>();
	auto &state = data_p.global_state->Cast<PragmaTableOperatorData>();
	switch (bind_data.entry.type) {
	case CatalogType::TABLE_ENTRY:
		PragmaTableInfoTable(state, bind_data.entry.Cast<TableCatalogEntry>(), output);
		break;
	case CatalogType::VIEW_ENTRY:
		PragmaTableInfoView(state, bind_data.entry.Cast<ViewCatalogEntry>(), output);
		break;
	default:
		throw NotImplementedException("Unimplemented catalog type for pragma_table_info");
	}
}

void PragmaTableInfo::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(TableFunction("pragma_table_info", {LogicalType::VARCHAR}, PragmaTableInfoFunction,
	                              PragmaTableInfoBind, PragmaTableInfoInit));
}

} // namespace duckdb






#include <cmath>
#include <limits>

namespace duckdb {

struct TestAllTypesData : public GlobalTableFunctionState {
	TestAllTypesData() : offset(0) {
	}

	vector<vector<Value>> entries;
	idx_t offset;
};

vector<TestType> TestAllTypesFun::GetTestTypes() {
	vector<TestType> result;
	// scalar types/numerics
	result.emplace_back(LogicalType::BOOLEAN, "bool");
	result.emplace_back(LogicalType::TINYINT, "tinyint");
	result.emplace_back(LogicalType::SMALLINT, "smallint");
	result.emplace_back(LogicalType::INTEGER, "int");
	result.emplace_back(LogicalType::BIGINT, "bigint");
	result.emplace_back(LogicalType::HUGEINT, "hugeint");
	result.emplace_back(LogicalType::UTINYINT, "utinyint");
	result.emplace_back(LogicalType::USMALLINT, "usmallint");
	result.emplace_back(LogicalType::UINTEGER, "uint");
	result.emplace_back(LogicalType::UBIGINT, "ubigint");
	result.emplace_back(LogicalType::DATE, "date");
	result.emplace_back(LogicalType::TIME, "time");
	result.emplace_back(LogicalType::TIMESTAMP, "timestamp");
	result.emplace_back(LogicalType::TIMESTAMP_S, "timestamp_s");
	result.emplace_back(LogicalType::TIMESTAMP_MS, "timestamp_ms");
	result.emplace_back(LogicalType::TIMESTAMP_NS, "timestamp_ns");
	result.emplace_back(LogicalType::TIME_TZ, "time_tz");
	result.emplace_back(LogicalType::TIMESTAMP_TZ, "timestamp_tz");
	result.emplace_back(LogicalType::FLOAT, "float");
	result.emplace_back(LogicalType::DOUBLE, "double");
	result.emplace_back(LogicalType::DECIMAL(4, 1), "dec_4_1");
	result.emplace_back(LogicalType::DECIMAL(9, 4), "dec_9_4");
	result.emplace_back(LogicalType::DECIMAL(18, 6), "dec_18_6");
	result.emplace_back(LogicalType::DECIMAL(38, 10), "dec38_10");
	result.emplace_back(LogicalType::UUID, "uuid");

	// interval
	interval_t min_interval;
	min_interval.months = 0;
	min_interval.days = 0;
	min_interval.micros = 0;

	interval_t max_interval;
	max_interval.months = 999;
	max_interval.days = 999;
	max_interval.micros = 999999999;
	result.emplace_back(LogicalType::INTERVAL, "interval", Value::INTERVAL(min_interval),
	                    Value::INTERVAL(max_interval));
	// strings/blobs/bitstrings
	result.emplace_back(LogicalType::VARCHAR, "varchar", Value("🦆🦆🦆🦆🦆🦆"),
	                    Value(string("goo\x00se", 6)));
	result.emplace_back(LogicalType::BLOB, "blob", Value::BLOB("thisisalongblob\\x00withnullbytes"),
	                    Value::BLOB("\\x00\\x00\\x00a"));
	result.emplace_back(LogicalType::BIT, "bit", Value::BIT("0010001001011100010101011010111"), Value::BIT("10101"));

	// enums
	Vector small_enum(LogicalType::VARCHAR, 2);
	auto small_enum_ptr = FlatVector::GetData<string_t>(small_enum);
	small_enum_ptr[0] = StringVector::AddStringOrBlob(small_enum, "DUCK_DUCK_ENUM");
	small_enum_ptr[1] = StringVector::AddStringOrBlob(small_enum, "GOOSE");
	result.emplace_back(LogicalType::ENUM("small_enum", small_enum, 2), "small_enum");

	Vector medium_enum(LogicalType::VARCHAR, 300);
	auto medium_enum_ptr = FlatVector::GetData<string_t>(medium_enum);
	for (idx_t i = 0; i < 300; i++) {
		medium_enum_ptr[i] = StringVector::AddStringOrBlob(medium_enum, string("enum_") + to_string(i));
	}
	result.emplace_back(LogicalType::ENUM("medium_enum", medium_enum, 300), "medium_enum");

	// this is a big one... not sure if we should push this one here, but it's required for completeness
	Vector large_enum(LogicalType::VARCHAR, 70000);
	auto large_enum_ptr = FlatVector::GetData<string_t>(large_enum);
	for (idx_t i = 0; i < 70000; i++) {
		large_enum_ptr[i] = StringVector::AddStringOrBlob(large_enum, string("enum_") + to_string(i));
	}
	result.emplace_back(LogicalType::ENUM("large_enum", large_enum, 70000), "large_enum");

	// arrays
	auto int_list_type = LogicalType::LIST(LogicalType::INTEGER);
	auto empty_int_list = Value::EMPTYLIST(LogicalType::INTEGER);
	auto int_list = Value::LIST({Value::INTEGER(42), Value::INTEGER(999), Value(LogicalType::INTEGER),
	                             Value(LogicalType::INTEGER), Value::INTEGER(-42)});
	result.emplace_back(int_list_type, "int_array", empty_int_list, int_list);

	auto double_list_type = LogicalType::LIST(LogicalType::DOUBLE);
	auto empty_double_list = Value::EMPTYLIST(LogicalType::DOUBLE);
	auto double_list = Value::LIST(
	    {Value::DOUBLE(42), Value::DOUBLE(NAN), Value::DOUBLE(std::numeric_limits<double>::infinity()),
	     Value::DOUBLE(-std::numeric_limits<double>::infinity()), Value(LogicalType::DOUBLE), Value::DOUBLE(-42)});
	result.emplace_back(double_list_type, "double_array", empty_double_list, double_list);

	auto date_list_type = LogicalType::LIST(LogicalType::DATE);
	auto empty_date_list = Value::EMPTYLIST(LogicalType::DATE);
	auto date_list =
	    Value::LIST({Value::DATE(date_t()), Value::DATE(date_t::infinity()), Value::DATE(date_t::ninfinity()),
	                 Value(LogicalType::DATE), Value::DATE(Date::FromString("2022-05-12"))});
	result.emplace_back(date_list_type, "date_array", empty_date_list, date_list);

	auto timestamp_list_type = LogicalType::LIST(LogicalType::TIMESTAMP);
	auto empty_timestamp_list = Value::EMPTYLIST(LogicalType::TIMESTAMP);
	auto timestamp_list = Value::LIST({Value::TIMESTAMP(timestamp_t()), Value::TIMESTAMP(timestamp_t::infinity()),
	                                   Value::TIMESTAMP(timestamp_t::ninfinity()), Value(LogicalType::TIMESTAMP),
	                                   Value::TIMESTAMP(Timestamp::FromString("2022-05-12 16:23:45"))});
	result.emplace_back(timestamp_list_type, "timestamp_array", empty_timestamp_list, timestamp_list);

	auto timestamptz_list_type = LogicalType::LIST(LogicalType::TIMESTAMP_TZ);
	auto empty_timestamptz_list = Value::EMPTYLIST(LogicalType::TIMESTAMP_TZ);
	auto timestamptz_list = Value::LIST({Value::TIMESTAMPTZ(timestamp_t()), Value::TIMESTAMPTZ(timestamp_t::infinity()),
	                                     Value::TIMESTAMPTZ(timestamp_t::ninfinity()), Value(LogicalType::TIMESTAMP_TZ),
	                                     Value::TIMESTAMPTZ(Timestamp::FromString("2022-05-12 16:23:45-07"))});
	result.emplace_back(timestamptz_list_type, "timestamptz_array", empty_timestamptz_list, timestamptz_list);

	auto varchar_list_type = LogicalType::LIST(LogicalType::VARCHAR);
	auto empty_varchar_list = Value::EMPTYLIST(LogicalType::VARCHAR);
	auto varchar_list =
	    Value::LIST({Value("🦆🦆🦆🦆🦆🦆"), Value("goose"), Value(LogicalType::VARCHAR), Value("")});
	result.emplace_back(varchar_list_type, "varchar_array", empty_varchar_list, varchar_list);

	// nested arrays
	auto nested_list_type = LogicalType::LIST(int_list_type);
	auto empty_nested_list = Value::EMPTYLIST(int_list_type);
	auto nested_int_list = Value::LIST({empty_int_list, int_list, Value(int_list_type), empty_int_list, int_list});
	result.emplace_back(nested_list_type, "nested_int_array", empty_nested_list, nested_int_list);

	// structs
	child_list_t<LogicalType> struct_type_list;
	struct_type_list.push_back(make_pair("a", LogicalType::INTEGER));
	struct_type_list.push_back(make_pair("b", LogicalType::VARCHAR));
	auto struct_type = LogicalType::STRUCT(struct_type_list);

	child_list_t<Value> min_struct_list;
	min_struct_list.push_back(make_pair("a", Value(LogicalType::INTEGER)));
	min_struct_list.push_back(make_pair("b", Value(LogicalType::VARCHAR)));
	auto min_struct_val = Value::STRUCT(std::move(min_struct_list));

	child_list_t<Value> max_struct_list;
	max_struct_list.push_back(make_pair("a", Value::INTEGER(42)));
	max_struct_list.push_back(make_pair("b", Value("🦆🦆🦆🦆🦆🦆")));
	auto max_struct_val = Value::STRUCT(std::move(max_struct_list));

	result.emplace_back(struct_type, "struct", min_struct_val, max_struct_val);

	// structs with lists
	child_list_t<LogicalType> struct_list_type_list;
	struct_list_type_list.push_back(make_pair("a", int_list_type));
	struct_list_type_list.push_back(make_pair("b", varchar_list_type));
	auto struct_list_type = LogicalType::STRUCT(struct_list_type_list);

	child_list_t<Value> min_struct_vl_list;
	min_struct_vl_list.push_back(make_pair("a", Value(int_list_type)));
	min_struct_vl_list.push_back(make_pair("b", Value(varchar_list_type)));
	auto min_struct_val_list = Value::STRUCT(std::move(min_struct_vl_list));

	child_list_t<Value> max_struct_vl_list;
	max_struct_vl_list.push_back(make_pair("a", int_list));
	max_struct_vl_list.push_back(make_pair("b", varchar_list));
	auto max_struct_val_list = Value::STRUCT(std::move(max_struct_vl_list));

	result.emplace_back(struct_list_type, "struct_of_arrays", std::move(min_struct_val_list),
	                    std::move(max_struct_val_list));

	// array of structs
	auto array_of_structs_type = LogicalType::LIST(struct_type);
	auto min_array_of_struct_val = Value::EMPTYLIST(struct_type);
	auto max_array_of_struct_val = Value::LIST({min_struct_val, max_struct_val, Value(struct_type)});
	result.emplace_back(array_of_structs_type, "array_of_structs", std::move(min_array_of_struct_val),
	                    std::move(max_array_of_struct_val));

	// map
	auto map_type = LogicalType::MAP(LogicalType::VARCHAR, LogicalType::VARCHAR);
	auto min_map_value = Value::MAP(ListType::GetChildType(map_type), vector<Value>());

	child_list_t<Value> map_struct1;
	map_struct1.push_back(make_pair("key", Value("key1")));
	map_struct1.push_back(make_pair("value", Value("🦆🦆🦆🦆🦆🦆")));
	child_list_t<Value> map_struct2;
	map_struct2.push_back(make_pair("key", Value("key2")));
	map_struct2.push_back(make_pair("key", Value("goose")));

	vector<Value> map_values;
	map_values.push_back(Value::STRUCT(map_struct1));
	map_values.push_back(Value::STRUCT(map_struct2));

	auto max_map_value = Value::MAP(ListType::GetChildType(map_type), map_values);
	result.emplace_back(map_type, "map", std::move(min_map_value), std::move(max_map_value));

	return result;
}

static unique_ptr<FunctionData> TestAllTypesBind(ClientContext &context, TableFunctionBindInput &input,
                                                 vector<LogicalType> &return_types, vector<string> &names) {
	auto test_types = TestAllTypesFun::GetTestTypes();
	for (auto &test_type : test_types) {
		return_types.push_back(std::move(test_type.type));
		names.push_back(std::move(test_type.name));
	}
	return nullptr;
}

unique_ptr<GlobalTableFunctionState> TestAllTypesInit(ClientContext &context, TableFunctionInitInput &input) {
	auto result = make_uniq<TestAllTypesData>();
	auto test_types = TestAllTypesFun::GetTestTypes();
	// 3 rows: min, max and NULL
	result->entries.resize(3);
	// initialize the values
	for (auto &test_type : test_types) {
		result->entries[0].push_back(std::move(test_type.min_value));
		result->entries[1].push_back(std::move(test_type.max_value));
		result->entries[2].emplace_back(std::move(test_type.type));
	}
	return std::move(result);
}

void TestAllTypesFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = data_p.global_state->Cast<TestAllTypesData>();
	if (data.offset >= data.entries.size()) {
		// finished returning values
		return;
	}
	// start returning values
	// either fill up the chunk or return all the remaining columns
	idx_t count = 0;
	while (data.offset < data.entries.size() && count < STANDARD_VECTOR_SIZE) {
		auto &vals = data.entries[data.offset++];
		for (idx_t col_idx = 0; col_idx < vals.size(); col_idx++) {
			output.SetValue(col_idx, count, vals[col_idx]);
		}
		count++;
	}
	output.SetCardinality(count);
}

void TestAllTypesFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(TableFunction("test_all_types", {}, TestAllTypesFunction, TestAllTypesBind, TestAllTypesInit));
}

} // namespace duckdb




namespace duckdb {

// FLAT, CONSTANT, DICTIONARY, SEQUENCE
struct TestVectorBindData : public TableFunctionData {
	LogicalType type;
	bool all_flat;
};

struct TestVectorTypesData : public GlobalTableFunctionState {
	TestVectorTypesData() : offset(0) {
	}

	vector<unique_ptr<DataChunk>> entries;
	idx_t offset;
};

struct TestVectorInfo {
	TestVectorInfo(const LogicalType &type, const map<LogicalTypeId, TestType> &test_type_map,
	               vector<unique_ptr<DataChunk>> &entries)
	    : type(type), test_type_map(test_type_map), entries(entries) {
	}

	const LogicalType &type;
	const map<LogicalTypeId, TestType> &test_type_map;
	vector<unique_ptr<DataChunk>> &entries;
};

struct TestVectorFlat {
	static constexpr const idx_t TEST_VECTOR_CARDINALITY = 3;

	static vector<Value> GenerateValues(TestVectorInfo &info, const LogicalType &type) {
		vector<Value> result;
		switch (type.InternalType()) {
		case PhysicalType::STRUCT: {
			vector<child_list_t<Value>> struct_children;
			auto &child_types = StructType::GetChildTypes(type);

			struct_children.resize(TEST_VECTOR_CARDINALITY);
			for (auto &child_type : child_types) {
				auto child_values = GenerateValues(info, child_type.second);

				for (idx_t i = 0; i < child_values.size(); i++) {
					struct_children[i].push_back(make_pair(child_type.first, std::move(child_values[i])));
				}
			}
			for (auto &struct_child : struct_children) {
				result.push_back(Value::STRUCT(std::move(struct_child)));
			}
			break;
		}
		case PhysicalType::LIST: {
			auto &child_type = ListType::GetChildType(type);
			auto child_values = GenerateValues(info, child_type);

			result.push_back(Value::LIST(child_type, {child_values[0], child_values[1]}));
			result.push_back(Value::LIST(child_type, {}));
			result.push_back(Value::LIST(child_type, {child_values[2]}));
			break;
		}
		default: {
			auto entry = info.test_type_map.find(type.id());
			if (entry == info.test_type_map.end()) {
				throw NotImplementedException("Unimplemented type for test_vector_types %s", type.ToString());
			}
			result.push_back(entry->second.min_value);
			result.push_back(entry->second.max_value);
			result.emplace_back(type);
			break;
		}
		}
		return result;
	}

	static void Generate(TestVectorInfo &info) {
		vector<Value> result_values = GenerateValues(info, info.type);
		for (idx_t cur_row = 0; cur_row < result_values.size(); cur_row += STANDARD_VECTOR_SIZE) {
			auto result = make_uniq<DataChunk>();
			result->Initialize(Allocator::DefaultAllocator(), {info.type});
			auto cardinality = MinValue<idx_t>(STANDARD_VECTOR_SIZE, result_values.size() - cur_row);
			for (idx_t i = 0; i < cardinality; i++) {
				result->data[0].SetValue(i, result_values[cur_row + i]);
			}
			result->SetCardinality(cardinality);
			info.entries.push_back(std::move(result));
		}
	}
};

struct TestVectorConstant {
	static void Generate(TestVectorInfo &info) {
		auto values = TestVectorFlat::GenerateValues(info, info.type);
		for (idx_t cur_row = 0; cur_row < TestVectorFlat::TEST_VECTOR_CARDINALITY; cur_row += STANDARD_VECTOR_SIZE) {
			auto result = make_uniq<DataChunk>();
			result->Initialize(Allocator::DefaultAllocator(), {info.type});
			auto cardinality = MinValue<idx_t>(STANDARD_VECTOR_SIZE, TestVectorFlat::TEST_VECTOR_CARDINALITY - cur_row);
			result->data[0].SetValue(0, values[0]);
			result->data[0].SetVectorType(VectorType::CONSTANT_VECTOR);
			result->SetCardinality(cardinality);

			info.entries.push_back(std::move(result));
		}
	}
};

struct TestVectorSequence {
	static void GenerateVector(TestVectorInfo &info, const LogicalType &type, Vector &result) {
		D_ASSERT(type == result.GetType());
		switch (type.id()) {
		case LogicalTypeId::TINYINT:
		case LogicalTypeId::SMALLINT:
		case LogicalTypeId::INTEGER:
		case LogicalTypeId::BIGINT:
		case LogicalTypeId::UTINYINT:
		case LogicalTypeId::USMALLINT:
		case LogicalTypeId::UINTEGER:
		case LogicalTypeId::UBIGINT:
			result.Sequence(3, 2, 3);
			return;
		default:
			break;
		}
		switch (type.InternalType()) {
		case PhysicalType::STRUCT: {
			auto &child_entries = StructVector::GetEntries(result);
			for (auto &child_entry : child_entries) {
				GenerateVector(info, child_entry->GetType(), *child_entry);
			}
			break;
		}
		case PhysicalType::LIST: {
			auto data = FlatVector::GetData<list_entry_t>(result);
			data[0].offset = 0;
			data[0].length = 2;
			data[1].offset = 2;
			data[1].length = 0;
			data[2].offset = 2;
			data[2].length = 1;

			GenerateVector(info, ListType::GetChildType(type), ListVector::GetEntry(result));
			ListVector::SetListSize(result, 3);
			break;
		}
		default: {
			auto entry = info.test_type_map.find(type.id());
			if (entry == info.test_type_map.end()) {
				throw NotImplementedException("Unimplemented type for test_vector_types %s", type.ToString());
			}
			result.SetValue(0, entry->second.min_value);
			result.SetValue(1, entry->second.max_value);
			result.SetValue(2, Value(type));
			break;
		}
		}
	}

	static void Generate(TestVectorInfo &info) {
#if STANDARD_VECTOR_SIZE > 2
		auto result = make_uniq<DataChunk>();
		result->Initialize(Allocator::DefaultAllocator(), {info.type});

		GenerateVector(info, info.type, result->data[0]);
		result->SetCardinality(3);
		info.entries.push_back(std::move(result));
#endif
	}
};

struct TestVectorDictionary {
	static void Generate(TestVectorInfo &info) {
		idx_t current_chunk = info.entries.size();

		unordered_set<idx_t> slice_entries {1, 2};

		TestVectorFlat::Generate(info);
		idx_t current_idx = 0;
		for (idx_t i = current_chunk; i < info.entries.size(); i++) {
			auto &chunk = *info.entries[i];
			SelectionVector sel(STANDARD_VECTOR_SIZE);
			idx_t sel_idx = 0;
			for (idx_t k = 0; k < chunk.size(); k++) {
				if (slice_entries.count(current_idx + k) > 0) {
					sel.set_index(sel_idx++, k);
				}
			}
			chunk.Slice(sel, sel_idx);
			current_idx += chunk.size();
		}
	}
};

static unique_ptr<FunctionData> TestVectorTypesBind(ClientContext &context, TableFunctionBindInput &input,
                                                    vector<LogicalType> &return_types, vector<string> &names) {
	auto result = make_uniq<TestVectorBindData>();
	result->type = input.inputs[0].type();
	result->all_flat = BooleanValue::Get(input.inputs[1]);

	return_types.push_back(result->type);
	names.emplace_back("test_vector");
	return std::move(result);
}

unique_ptr<GlobalTableFunctionState> TestVectorTypesInit(ClientContext &context, TableFunctionInitInput &input) {
	auto &bind_data = input.bind_data->Cast<TestVectorBindData>();

	auto result = make_uniq<TestVectorTypesData>();

	auto test_types = TestAllTypesFun::GetTestTypes();

	map<LogicalTypeId, TestType> test_type_map;
	for (auto &test_type : test_types) {
		test_type_map.insert(make_pair(test_type.type.id(), std::move(test_type)));
	}

	TestVectorInfo info(bind_data.type, test_type_map, result->entries);
	TestVectorFlat::Generate(info);
	TestVectorConstant::Generate(info);
	TestVectorDictionary::Generate(info);
	TestVectorSequence::Generate(info);
	for (auto &entry : result->entries) {
		entry->Verify();
	}
	if (bind_data.all_flat) {
		for (auto &entry : result->entries) {
			entry->Flatten();
			entry->Verify();
		}
	}
	return std::move(result);
}

void TestVectorTypesFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = (TestVectorTypesData &)*data_p.global_state;
	if (data.offset >= data.entries.size()) {
		// finished returning values
		return;
	}
	output.Reference(*data.entries[data.offset]);
	data.offset++;
}

void TestVectorTypesFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(TableFunction("test_vector_types", {LogicalType::ANY, LogicalType::BOOLEAN},
	                              TestVectorTypesFunction, TestVectorTypesBind, TestVectorTypesInit));
}

} // namespace duckdb








namespace duckdb {

void BuiltinFunctions::RegisterSQLiteFunctions() {
	PragmaVersion::RegisterFunction(*this);
	PragmaCollations::RegisterFunction(*this);
	PragmaTableInfo::RegisterFunction(*this);
	PragmaStorageInfo::RegisterFunction(*this);
	PragmaDatabaseSize::RegisterFunction(*this);
	PragmaLastProfilingOutput::RegisterFunction(*this);
	PragmaDetailedProfilingOutput::RegisterFunction(*this);

	DuckDBColumnsFun::RegisterFunction(*this);
	DuckDBConstraintsFun::RegisterFunction(*this);
	DuckDBDatabasesFun::RegisterFunction(*this);
	DuckDBFunctionsFun::RegisterFunction(*this);
	DuckDBKeywordsFun::RegisterFunction(*this);
	DuckDBIndexesFun::RegisterFunction(*this);
	DuckDBSchemasFun::RegisterFunction(*this);
	DuckDBDependenciesFun::RegisterFunction(*this);
	DuckDBExtensionsFun::RegisterFunction(*this);
	DuckDBSequencesFun::RegisterFunction(*this);
	DuckDBSettingsFun::RegisterFunction(*this);
	DuckDBTablesFun::RegisterFunction(*this);
	DuckDBTemporaryFilesFun::RegisterFunction(*this);
	DuckDBTypesFun::RegisterFunction(*this);
	DuckDBViewsFun::RegisterFunction(*this);
	TestAllTypesFun::RegisterFunction(*this);
	TestVectorTypesFun::RegisterFunction(*this);
}

} // namespace duckdb


















namespace duckdb {

//===--------------------------------------------------------------------===//
// Table Scan
//===--------------------------------------------------------------------===//
bool TableScanParallelStateNext(ClientContext &context, const FunctionData *bind_data_p,
                                LocalTableFunctionState *local_state, GlobalTableFunctionState *gstate);

struct TableScanLocalState : public LocalTableFunctionState {
	//! The current position in the scan
	TableScanState scan_state;
	//! The DataChunk containing all read columns (even filter columns that are immediately removed)
	DataChunk all_columns;
};

static storage_t GetStorageIndex(TableCatalogEntry &table, column_t column_id) {
	if (column_id == DConstants::INVALID_INDEX) {
		return column_id;
	}
	auto &col = table.GetColumn(LogicalIndex(column_id));
	return col.StorageOid();
}

struct TableScanGlobalState : public GlobalTableFunctionState {
	TableScanGlobalState(ClientContext &context, const FunctionData *bind_data_p) {
		D_ASSERT(bind_data_p);
		auto &bind_data = bind_data_p->Cast<TableScanBindData>();
		max_threads = bind_data.table.GetStorage().MaxThreads(context);
	}

	ParallelTableScanState state;
	idx_t max_threads;

	vector<idx_t> projection_ids;
	vector<LogicalType> scanned_types;

	idx_t MaxThreads() const override {
		return max_threads;
	}

	bool CanRemoveFilterColumns() const {
		return !projection_ids.empty();
	}
};

static unique_ptr<LocalTableFunctionState> TableScanInitLocal(ExecutionContext &context, TableFunctionInitInput &input,
                                                              GlobalTableFunctionState *gstate) {
	auto result = make_uniq<TableScanLocalState>();
	auto &bind_data = input.bind_data->Cast<TableScanBindData>();
	vector<column_t> column_ids = input.column_ids;
	for (auto &col : column_ids) {
		auto storage_idx = GetStorageIndex(bind_data.table, col);
		col = storage_idx;
	}
	result->scan_state.Initialize(std::move(column_ids), input.filters.get());
	TableScanParallelStateNext(context.client, input.bind_data.get(), result.get(), gstate);
	if (input.CanRemoveFilterColumns()) {
		auto &tsgs = gstate->Cast<TableScanGlobalState>();
		result->all_columns.Initialize(context.client, tsgs.scanned_types);
	}
	return std::move(result);
}

unique_ptr<GlobalTableFunctionState> TableScanInitGlobal(ClientContext &context, TableFunctionInitInput &input) {

	D_ASSERT(input.bind_data);
	auto &bind_data = input.bind_data->Cast<TableScanBindData>();
	auto result = make_uniq<TableScanGlobalState>(context, input.bind_data.get());
	bind_data.table.GetStorage().InitializeParallelScan(context, result->state);
	if (input.CanRemoveFilterColumns()) {
		result->projection_ids = input.projection_ids;
		const auto &columns = bind_data.table.GetColumns();
		for (const auto &col_idx : input.column_ids) {
			if (col_idx == COLUMN_IDENTIFIER_ROW_ID) {
				result->scanned_types.emplace_back(LogicalType::ROW_TYPE);
			} else {
				result->scanned_types.push_back(columns.GetColumn(LogicalIndex(col_idx)).Type());
			}
		}
	}
	return std::move(result);
}

static unique_ptr<BaseStatistics> TableScanStatistics(ClientContext &context, const FunctionData *bind_data_p,
                                                      column_t column_id) {
	auto &bind_data = bind_data_p->Cast<TableScanBindData>();
	auto &local_storage = LocalStorage::Get(context, bind_data.table.catalog);
	if (local_storage.Find(bind_data.table.GetStorage())) {
		// we don't emit any statistics for tables that have outstanding transaction-local data
		return nullptr;
	}
	return bind_data.table.GetStatistics(context, column_id);
}

static void TableScanFunc(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind_data = data_p.bind_data->Cast<TableScanBindData>();
	auto &gstate = data_p.global_state->Cast<TableScanGlobalState>();
	auto &state = data_p.local_state->Cast<TableScanLocalState>();
	auto &transaction = DuckTransaction::Get(context, bind_data.table.catalog);
	auto &storage = bind_data.table.GetStorage();
	do {
		if (bind_data.is_create_index) {
			storage.CreateIndexScan(state.scan_state, output,
			                        TableScanType::TABLE_SCAN_COMMITTED_ROWS_OMIT_PERMANENTLY_DELETED);
		} else if (gstate.CanRemoveFilterColumns()) {
			state.all_columns.Reset();
			storage.Scan(transaction, state.all_columns, state.scan_state);
			output.ReferenceColumns(state.all_columns, gstate.projection_ids);
		} else {
			storage.Scan(transaction, output, state.scan_state);
		}
		if (output.size() > 0) {
			return;
		}
		if (!TableScanParallelStateNext(context, data_p.bind_data.get(), data_p.local_state.get(),
		                                data_p.global_state.get())) {
			return;
		}
	} while (true);
}

bool TableScanParallelStateNext(ClientContext &context, const FunctionData *bind_data_p,
                                LocalTableFunctionState *local_state, GlobalTableFunctionState *global_state) {
	auto &bind_data = bind_data_p->Cast<TableScanBindData>();
	auto &parallel_state = global_state->Cast<TableScanGlobalState>();
	auto &state = local_state->Cast<TableScanLocalState>();
	auto &storage = bind_data.table.GetStorage();

	return storage.NextParallelScan(context, parallel_state.state, state.scan_state);
}

double TableScanProgress(ClientContext &context, const FunctionData *bind_data_p,
                         const GlobalTableFunctionState *gstate_p) {
	auto &bind_data = bind_data_p->Cast<TableScanBindData>();
	auto &gstate = gstate_p->Cast<TableScanGlobalState>();
	auto &storage = bind_data.table.GetStorage();
	idx_t total_rows = storage.GetTotalRows();
	if (total_rows == 0) {
		//! Table is either empty or smaller than a vector size, so it is finished
		return 100;
	}
	idx_t scanned_rows = gstate.state.scan_state.processed_rows;
	scanned_rows += gstate.state.local_state.processed_rows;
	auto percentage = 100 * (double(scanned_rows) / total_rows);
	if (percentage > 100) {
		//! In case the last chunk has less elements than STANDARD_VECTOR_SIZE, if our percentage is over 100
		//! It means we finished this table.
		return 100;
	}
	return percentage;
}

idx_t TableScanGetBatchIndex(ClientContext &context, const FunctionData *bind_data_p,
                             LocalTableFunctionState *local_state, GlobalTableFunctionState *gstate_p) {
	auto &state = local_state->Cast<TableScanLocalState>();
	if (state.scan_state.table_state.row_group) {
		return state.scan_state.table_state.batch_index;
	}
	if (state.scan_state.local_state.row_group) {
		return state.scan_state.table_state.batch_index + state.scan_state.local_state.batch_index;
	}
	return 0;
}

BindInfo TableScanGetBindInfo(const FunctionData *bind_data) {
	return BindInfo(ScanType::TABLE);
}

void TableScanDependency(DependencyList &entries, const FunctionData *bind_data_p) {
	auto &bind_data = bind_data_p->Cast<TableScanBindData>();
	entries.AddDependency(bind_data.table);
}

unique_ptr<NodeStatistics> TableScanCardinality(ClientContext &context, const FunctionData *bind_data_p) {
	auto &bind_data = bind_data_p->Cast<TableScanBindData>();
	auto &local_storage = LocalStorage::Get(context, bind_data.table.catalog);
	auto &storage = bind_data.table.GetStorage();
	idx_t estimated_cardinality = storage.info->cardinality + local_storage.AddedRows(bind_data.table.GetStorage());
	return make_uniq<NodeStatistics>(storage.info->cardinality, estimated_cardinality);
}

//===--------------------------------------------------------------------===//
// Index Scan
//===--------------------------------------------------------------------===//
struct IndexScanGlobalState : public GlobalTableFunctionState {
	explicit IndexScanGlobalState(data_ptr_t row_id_data) : row_ids(LogicalType::ROW_TYPE, row_id_data) {
	}

	Vector row_ids;
	ColumnFetchState fetch_state;
	TableScanState local_storage_state;
	vector<storage_t> column_ids;
	bool finished;
};

static unique_ptr<GlobalTableFunctionState> IndexScanInitGlobal(ClientContext &context, TableFunctionInitInput &input) {
	auto &bind_data = input.bind_data->Cast<TableScanBindData>();
	data_ptr_t row_id_data = nullptr;
	if (!bind_data.result_ids.empty()) {
		row_id_data = (data_ptr_t)&bind_data.result_ids[0];
	}
	auto result = make_uniq<IndexScanGlobalState>(row_id_data);
	auto &local_storage = LocalStorage::Get(context, bind_data.table.catalog);

	result->column_ids.reserve(input.column_ids.size());
	for (auto &id : input.column_ids) {
		result->column_ids.push_back(GetStorageIndex(bind_data.table, id));
	}
	result->local_storage_state.Initialize(result->column_ids, input.filters.get());
	local_storage.InitializeScan(bind_data.table.GetStorage(), result->local_storage_state.local_state, input.filters);

	result->finished = false;
	return std::move(result);
}

static void IndexScanFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind_data = data_p.bind_data->Cast<TableScanBindData>();
	auto &state = data_p.global_state->Cast<IndexScanGlobalState>();
	auto &transaction = DuckTransaction::Get(context, bind_data.table.catalog);
	auto &local_storage = LocalStorage::Get(transaction);

	if (!state.finished) {
		bind_data.table.GetStorage().Fetch(transaction, output, state.column_ids, state.row_ids,
		                                   bind_data.result_ids.size(), state.fetch_state);
		state.finished = true;
	}
	if (output.size() == 0) {
		local_storage.Scan(state.local_storage_state.local_state, state.column_ids, output);
	}
}

static void RewriteIndexExpression(Index &index, LogicalGet &get, Expression &expr, bool &rewrite_possible) {
	if (expr.type == ExpressionType::BOUND_COLUMN_REF) {
		auto &bound_colref = expr.Cast<BoundColumnRefExpression>();
		// bound column ref: rewrite to fit in the current set of bound column ids
		bound_colref.binding.table_index = get.table_index;
		column_t referenced_column = index.column_ids[bound_colref.binding.column_index];
		// search for the referenced column in the set of column_ids
		for (idx_t i = 0; i < get.column_ids.size(); i++) {
			if (get.column_ids[i] == referenced_column) {
				bound_colref.binding.column_index = i;
				return;
			}
		}
		// column id not found in bound columns in the LogicalGet: rewrite not possible
		rewrite_possible = false;
	}
	ExpressionIterator::EnumerateChildren(
	    expr, [&](Expression &child) { RewriteIndexExpression(index, get, child, rewrite_possible); });
}

void TableScanPushdownComplexFilter(ClientContext &context, LogicalGet &get, FunctionData *bind_data_p,
                                    vector<unique_ptr<Expression>> &filters) {
	auto &bind_data = bind_data_p->Cast<TableScanBindData>();
	auto &table = bind_data.table;
	auto &storage = table.GetStorage();

	auto &config = ClientConfig::GetConfig(context);
	if (!config.enable_optimizer) {
		// we only push index scans if the optimizer is enabled
		return;
	}
	if (bind_data.is_index_scan) {
		return;
	}
	if (filters.empty()) {
		// no indexes or no filters: skip the pushdown
		return;
	}
	// behold
	storage.info->indexes.Scan([&](Index &index) {
		// first rewrite the index expression so the ColumnBindings align with the column bindings of the current table

		if (index.unbound_expressions.size() > 1) {
			// NOTE: index scans are not (yet) supported for compound index keys
			return false;
		}

		auto index_expression = index.unbound_expressions[0]->Copy();
		bool rewrite_possible = true;
		RewriteIndexExpression(index, get, *index_expression, rewrite_possible);
		if (!rewrite_possible) {
			// could not rewrite!
			return false;
		}

		Value low_value, high_value, equal_value;
		ExpressionType low_comparison_type = ExpressionType::INVALID, high_comparison_type = ExpressionType::INVALID;
		// try to find a matching index for any of the filter expressions
		for (auto &filter : filters) {
			auto &expr = *filter;

			// create a matcher for a comparison with a constant
			ComparisonExpressionMatcher matcher;
			// match on a comparison type
			matcher.expr_type = make_uniq<ComparisonExpressionTypeMatcher>();
			// match on a constant comparison with the indexed expression
			matcher.matchers.push_back(make_uniq<ExpressionEqualityMatcher>(*index_expression));
			matcher.matchers.push_back(make_uniq<ConstantExpressionMatcher>());

			matcher.policy = SetMatcher::Policy::UNORDERED;

			vector<reference<Expression>> bindings;
			if (matcher.Match(expr, bindings)) {
				// range or equality comparison with constant value
				// we can use our index here
				// bindings[0] = the expression
				// bindings[1] = the index expression
				// bindings[2] = the constant
				auto &comparison = bindings[0].get().Cast<BoundComparisonExpression>();
				auto constant_value = bindings[2].get().Cast<BoundConstantExpression>().value;
				auto comparison_type = comparison.type;
				if (comparison.left->type == ExpressionType::VALUE_CONSTANT) {
					// the expression is on the right side, we flip them around
					comparison_type = FlipComparisonExpression(comparison_type);
				}
				if (comparison_type == ExpressionType::COMPARE_EQUAL) {
					// equality value
					// equality overrides any other bounds so we just break here
					equal_value = constant_value;
					break;
				} else if (comparison_type == ExpressionType::COMPARE_GREATERTHANOREQUALTO ||
				           comparison_type == ExpressionType::COMPARE_GREATERTHAN) {
					// greater than means this is a lower bound
					low_value = constant_value;
					low_comparison_type = comparison_type;
				} else {
					// smaller than means this is an upper bound
					high_value = constant_value;
					high_comparison_type = comparison_type;
				}
			} else if (expr.type == ExpressionType::COMPARE_BETWEEN) {
				// BETWEEN expression
				auto &between = expr.Cast<BoundBetweenExpression>();
				if (!between.input->Equals(index_expression.get())) {
					// expression doesn't match the current index expression
					continue;
				}
				if (between.lower->type != ExpressionType::VALUE_CONSTANT ||
				    between.upper->type != ExpressionType::VALUE_CONSTANT) {
					// not a constant comparison
					continue;
				}
				low_value = (between.lower->Cast<BoundConstantExpression>()).value;
				low_comparison_type = between.lower_inclusive ? ExpressionType::COMPARE_GREATERTHANOREQUALTO
				                                              : ExpressionType::COMPARE_GREATERTHAN;
				high_value = (between.upper->Cast<BoundConstantExpression>()).value;
				high_comparison_type = between.upper_inclusive ? ExpressionType::COMPARE_LESSTHANOREQUALTO
				                                               : ExpressionType::COMPARE_LESSTHAN;
				break;
			}
		}
		if (!equal_value.IsNull() || !low_value.IsNull() || !high_value.IsNull()) {
			// we can scan this index using this predicate: try a scan
			auto &transaction = Transaction::Get(context, bind_data.table.catalog);
			unique_ptr<IndexScanState> index_state;
			if (!equal_value.IsNull()) {
				// equality predicate
				index_state =
				    index.InitializeScanSinglePredicate(transaction, equal_value, ExpressionType::COMPARE_EQUAL);
			} else if (!low_value.IsNull() && !high_value.IsNull()) {
				// two-sided predicate
				index_state = index.InitializeScanTwoPredicates(transaction, low_value, low_comparison_type, high_value,
				                                                high_comparison_type);
			} else if (!low_value.IsNull()) {
				// less than predicate
				index_state = index.InitializeScanSinglePredicate(transaction, low_value, low_comparison_type);
			} else {
				D_ASSERT(!high_value.IsNull());
				index_state = index.InitializeScanSinglePredicate(transaction, high_value, high_comparison_type);
			}
			if (index.Scan(transaction, storage, *index_state, STANDARD_VECTOR_SIZE, bind_data.result_ids)) {
				// use an index scan!
				bind_data.is_index_scan = true;
				get.function = TableScanFunction::GetIndexScanFunction();
			} else {
				bind_data.result_ids.clear();
			}
			return true;
		}
		return false;
	});
}

string TableScanToString(const FunctionData *bind_data_p) {
	auto &bind_data = bind_data_p->Cast<TableScanBindData>();
	string result = bind_data.table.name;
	return result;
}

static void TableScanSerialize(FieldWriter &writer, const FunctionData *bind_data_p, const TableFunction &function) {
	auto &bind_data = bind_data_p->Cast<TableScanBindData>();

	writer.WriteString(bind_data.table.schema.name);
	writer.WriteString(bind_data.table.name);
	writer.WriteField<bool>(bind_data.is_index_scan);
	writer.WriteField<bool>(bind_data.is_create_index);
	writer.WriteList<row_t>(bind_data.result_ids);
	writer.WriteString(bind_data.table.schema.catalog.GetName());
}

static unique_ptr<FunctionData> TableScanDeserialize(ClientContext &context, FieldReader &reader,
                                                     TableFunction &function) {
	auto schema_name = reader.ReadRequired<string>();
	auto table_name = reader.ReadRequired<string>();
	auto is_index_scan = reader.ReadRequired<bool>();
	auto is_create_index = reader.ReadRequired<bool>();
	auto result_ids = reader.ReadRequiredList<row_t>();
	auto catalog_name = reader.ReadField<string>(INVALID_CATALOG);

	auto &catalog_entry = Catalog::GetEntry<TableCatalogEntry>(context, catalog_name, schema_name, table_name);
	if (catalog_entry.type != CatalogType::TABLE_ENTRY) {
		throw SerializationException("Cant find table for %s.%s", schema_name, table_name);
	}

	auto result = make_uniq<TableScanBindData>(catalog_entry.Cast<DuckTableEntry>());
	result->is_index_scan = is_index_scan;
	result->is_create_index = is_create_index;
	result->result_ids = std::move(result_ids);
	return std::move(result);
}

TableFunction TableScanFunction::GetIndexScanFunction() {
	TableFunction scan_function("index_scan", {}, IndexScanFunction);
	scan_function.init_local = nullptr;
	scan_function.init_global = IndexScanInitGlobal;
	scan_function.statistics = TableScanStatistics;
	scan_function.dependency = TableScanDependency;
	scan_function.cardinality = TableScanCardinality;
	scan_function.pushdown_complex_filter = nullptr;
	scan_function.to_string = TableScanToString;
	scan_function.table_scan_progress = nullptr;
	scan_function.get_batch_index = nullptr;
	scan_function.projection_pushdown = true;
	scan_function.filter_pushdown = false;
	scan_function.serialize = TableScanSerialize;
	scan_function.deserialize = TableScanDeserialize;
	return scan_function;
}

TableFunction TableScanFunction::GetFunction() {
	TableFunction scan_function("seq_scan", {}, TableScanFunc);
	scan_function.init_local = TableScanInitLocal;
	scan_function.init_global = TableScanInitGlobal;
	scan_function.statistics = TableScanStatistics;
	scan_function.dependency = TableScanDependency;
	scan_function.cardinality = TableScanCardinality;
	scan_function.pushdown_complex_filter = TableScanPushdownComplexFilter;
	scan_function.to_string = TableScanToString;
	scan_function.table_scan_progress = TableScanProgress;
	scan_function.get_batch_index = TableScanGetBatchIndex;
	scan_function.get_batch_info = TableScanGetBindInfo;
	scan_function.projection_pushdown = true;
	scan_function.filter_pushdown = true;
	scan_function.filter_prune = true;
	scan_function.serialize = TableScanSerialize;
	scan_function.deserialize = TableScanDeserialize;
	return scan_function;
}

optional_ptr<TableCatalogEntry> TableScanFunction::GetTableEntry(const TableFunction &function,
                                                                 const optional_ptr<FunctionData> bind_data_p) {
	if (function.function != TableScanFunc || !bind_data_p) {
		return nullptr;
	}
	auto &bind_data = bind_data_p->Cast<TableScanBindData>();
	return &bind_data.table;
}

void TableScanFunction::RegisterFunction(BuiltinFunctions &set) {
	TableFunctionSet table_scan_set("seq_scan");
	table_scan_set.AddFunction(GetFunction());
	set.AddFunction(std::move(table_scan_set));

	set.AddFunction(GetIndexScanFunction());
}

void BuiltinFunctions::RegisterTableScanFunctions() {
	TableScanFunction::RegisterFunction(*this);
}

} // namespace duckdb






namespace duckdb {

struct UnnestBindData : public FunctionData {
	explicit UnnestBindData(LogicalType input_type_p) : input_type(std::move(input_type_p)) {
	}

	LogicalType input_type;

public:
	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<UnnestBindData>(input_type);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = (const UnnestBindData &)other_p;
		return input_type == other.input_type;
	}
};

struct UnnestGlobalState : public GlobalTableFunctionState {
	UnnestGlobalState() {
	}

	vector<unique_ptr<Expression>> select_list;

	idx_t MaxThreads() const override {
		return GlobalTableFunctionState::MAX_THREADS;
	}
};

struct UnnestLocalState : public LocalTableFunctionState {
	UnnestLocalState() {
	}

	unique_ptr<OperatorState> operator_state;
};

static unique_ptr<FunctionData> UnnestBind(ClientContext &context, TableFunctionBindInput &input,
                                           vector<LogicalType> &return_types, vector<string> &names) {
	if (input.input_table_types.size() != 1 || input.input_table_types[0].id() != LogicalTypeId::LIST) {
		throw BinderException("UNNEST requires a single list as input");
	}
	return_types.push_back(ListType::GetChildType(input.input_table_types[0]));
	names.push_back(input.input_table_names[0]);
	return make_uniq<UnnestBindData>(input.input_table_types[0]);
}

static unique_ptr<LocalTableFunctionState> UnnestLocalInit(ExecutionContext &context, TableFunctionInitInput &input,
                                                           GlobalTableFunctionState *global_state) {
	auto &gstate = global_state->Cast<UnnestGlobalState>();

	auto result = make_uniq<UnnestLocalState>();
	result->operator_state = PhysicalUnnest::GetState(context, gstate.select_list);
	return std::move(result);
}

static unique_ptr<GlobalTableFunctionState> UnnestInit(ClientContext &context, TableFunctionInitInput &input) {
	auto &bind_data = input.bind_data->Cast<UnnestBindData>();
	auto result = make_uniq<UnnestGlobalState>();
	auto ref = make_uniq<BoundReferenceExpression>(bind_data.input_type, 0);
	auto bound_unnest = make_uniq<BoundUnnestExpression>(ListType::GetChildType(bind_data.input_type));
	bound_unnest->child = std::move(ref);
	result->select_list.push_back(std::move(bound_unnest));
	return std::move(result);
}

static OperatorResultType UnnestFunction(ExecutionContext &context, TableFunctionInput &data_p, DataChunk &input,
                                         DataChunk &output) {
	auto &state = data_p.global_state->Cast<UnnestGlobalState>();
	auto &lstate = data_p.local_state->Cast<UnnestLocalState>();
	return PhysicalUnnest::ExecuteInternal(context, input, output, *lstate.operator_state, state.select_list, false);
}

void UnnestTableFunction::RegisterFunction(BuiltinFunctions &set) {
	TableFunction unnest_function("unnest", {LogicalTypeId::TABLE}, nullptr, UnnestBind, UnnestInit, UnnestLocalInit);
	unnest_function.in_out_function = UnnestFunction;
	set.AddFunction(unnest_function);
}

} // namespace duckdb



#include <cstdint>

namespace duckdb {

struct PragmaVersionData : public GlobalTableFunctionState {
	PragmaVersionData() : finished(false) {
	}

	bool finished;
};

static unique_ptr<FunctionData> PragmaVersionBind(ClientContext &context, TableFunctionBindInput &input,
                                                  vector<LogicalType> &return_types, vector<string> &names) {
	names.emplace_back("library_version");
	return_types.emplace_back(LogicalType::VARCHAR);
	names.emplace_back("source_id");
	return_types.emplace_back(LogicalType::VARCHAR);
	return nullptr;
}

static unique_ptr<GlobalTableFunctionState> PragmaVersionInit(ClientContext &context, TableFunctionInitInput &input) {
	return make_uniq<PragmaVersionData>();
}

static void PragmaVersionFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = (PragmaVersionData &)*data_p.global_state;
	if (data.finished) {
		// finished returning values
		return;
	}
	output.SetCardinality(1);
	output.SetValue(0, 0, DuckDB::LibraryVersion());
	output.SetValue(1, 0, DuckDB::SourceID());
	data.finished = true;
}

void PragmaVersion::RegisterFunction(BuiltinFunctions &set) {
	TableFunction pragma_version("pragma_version", {}, PragmaVersionFunction);
	pragma_version.bind = PragmaVersionBind;
	pragma_version.init_global = PragmaVersionInit;
	set.AddFunction(pragma_version);
}

idx_t DuckDB::StandardVectorSize() {
	return STANDARD_VECTOR_SIZE;
}

const char *DuckDB::SourceID() {
	return DUCKDB_SOURCE_ID;
}

const char *DuckDB::LibraryVersion() {
	return DUCKDB_VERSION;
}

string DuckDB::Platform() {
	string os = "linux";
#if INTPTR_MAX == INT64_MAX
	string arch = "amd64";
#elif INTPTR_MAX == INT32_MAX
	string arch = "i686";
#else
#error Unknown pointer size or missing size macros!
#endif
	string postfix = "";

#ifdef _WIN32
	os = "windows";
#elif defined(__APPLE__)
	os = "osx";
#endif
#if defined(__aarch64__) || defined(__ARM_ARCH_ISA_A64)
	arch = "arm64";
#endif

#if !defined(_GLIBCXX_USE_CXX11_ABI) || _GLIBCXX_USE_CXX11_ABI == 0
	if (os == "linux") {
		postfix = "_gcc4";
	}
#endif
#ifdef __MINGW32__
	postfix = "_mingw";
#endif
// this is used for the windows R builds which use a separate build environment
#ifdef DUCKDB_PLATFORM_RTOOLS
	postfix = "_rtools";
#endif
	return os + "_" + arch + postfix;
}

} // namespace duckdb


namespace duckdb {

GlobalTableFunctionState::~GlobalTableFunctionState() {
}

LocalTableFunctionState::~LocalTableFunctionState() {
}

TableFunctionInfo::~TableFunctionInfo() {
}

TableFunction::TableFunction(string name, vector<LogicalType> arguments, table_function_t function,
                             table_function_bind_t bind, table_function_init_global_t init_global,
                             table_function_init_local_t init_local)
    : SimpleNamedParameterFunction(std::move(name), std::move(arguments)), bind(bind), bind_replace(nullptr),
      init_global(init_global), init_local(init_local), function(function), in_out_function(nullptr),
      in_out_function_final(nullptr), statistics(nullptr), dependency(nullptr), cardinality(nullptr),
      pushdown_complex_filter(nullptr), to_string(nullptr), table_scan_progress(nullptr), get_batch_index(nullptr),
      get_batch_info(nullptr), serialize(nullptr), deserialize(nullptr), projection_pushdown(false),
      filter_pushdown(false), filter_prune(false) {
}

TableFunction::TableFunction(const vector<LogicalType> &arguments, table_function_t function,
                             table_function_bind_t bind, table_function_init_global_t init_global,
                             table_function_init_local_t init_local)
    : TableFunction(string(), arguments, function, bind, init_global, init_local) {
}
TableFunction::TableFunction()
    : SimpleNamedParameterFunction("", {}), bind(nullptr), bind_replace(nullptr), init_global(nullptr),
      init_local(nullptr), function(nullptr), in_out_function(nullptr), statistics(nullptr), dependency(nullptr),
      cardinality(nullptr), pushdown_complex_filter(nullptr), to_string(nullptr), table_scan_progress(nullptr),
      get_batch_index(nullptr), get_batch_info(nullptr), serialize(nullptr), deserialize(nullptr),
      projection_pushdown(false), filter_pushdown(false), filter_prune(false) {
}

bool TableFunction::Equal(const TableFunction &rhs) const {
	// number of types
	if (this->arguments.size() != rhs.arguments.size()) {
		return false;
	}
	// argument types
	for (idx_t i = 0; i < this->arguments.size(); ++i) {
		if (this->arguments[i] != rhs.arguments[i]) {
			return false;
		}
	}
	// varargs
	if (this->varargs != rhs.varargs) {
		return false;
	}

	return true; // they are equal
}

} // namespace duckdb
//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/function/table_macro_function.hpp
//
//
//===----------------------------------------------------------------------===//
//! The SelectStatement of the view





namespace duckdb {

TableMacroFunction::TableMacroFunction(unique_ptr<QueryNode> query_node)
    : MacroFunction(MacroType::TABLE_MACRO), query_node(std::move(query_node)) {
}

TableMacroFunction::TableMacroFunction(void) : MacroFunction(MacroType::TABLE_MACRO) {
}

unique_ptr<MacroFunction> TableMacroFunction::Copy() const {
	auto result = make_uniq<TableMacroFunction>();
	result->query_node = query_node->Copy();
	this->CopyProperties(*result);
	return std::move(result);
}

string TableMacroFunction::ToSQL(const string &schema, const string &name) const {
	return MacroFunction::ToSQL(schema, name) + StringUtil::Format("TABLE (%s);", query_node->ToString());
}

void TableMacroFunction::SerializeInternal(FieldWriter &writer) const {
	writer.WriteSerializable(*query_node);
}

unique_ptr<MacroFunction> TableMacroFunction::Deserialize(FieldReader &reader) {
	auto result = make_uniq<TableMacroFunction>();
	result->query_node = reader.ReadRequiredSerializable<QueryNode>();
	return std::move(result);
}

} // namespace duckdb







namespace duckdb {

void UDFWrapper::RegisterFunction(string name, vector<LogicalType> args, LogicalType ret_type,
                                  scalar_function_t udf_function, ClientContext &context, LogicalType varargs) {

	ScalarFunction scalar_function(std::move(name), std::move(args), std::move(ret_type), std::move(udf_function));
	scalar_function.varargs = std::move(varargs);
	scalar_function.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	CreateScalarFunctionInfo info(scalar_function);
	info.schema = DEFAULT_SCHEMA;
	context.RegisterFunction(info);
}

void UDFWrapper::RegisterAggrFunction(AggregateFunction aggr_function, ClientContext &context, LogicalType varargs) {
	aggr_function.varargs = std::move(varargs);
	CreateAggregateFunctionInfo info(std::move(aggr_function));
	context.RegisterFunction(info);
}

} // namespace duckdb















namespace duckdb {

BaseAppender::BaseAppender(Allocator &allocator, AppenderType type_p)
    : allocator(allocator), column(0), appender_type(type_p) {
}

BaseAppender::BaseAppender(Allocator &allocator_p, vector<LogicalType> types_p, AppenderType type_p)
    : allocator(allocator_p), types(std::move(types_p)), collection(make_uniq<ColumnDataCollection>(allocator, types)),
      column(0), appender_type(type_p) {
	InitializeChunk();
}

BaseAppender::~BaseAppender() {
}

void BaseAppender::Destructor() {
	if (Exception::UncaughtException()) {
		return;
	}
	// flush any remaining chunks, but only if we are not cleaning up the appender as part of an exception stack unwind
	// wrapped in a try/catch because Close() can throw if the table was dropped in the meantime
	try {
		Close();
	} catch (...) {
	}
}

InternalAppender::InternalAppender(ClientContext &context_p, TableCatalogEntry &table_p)
    : BaseAppender(Allocator::DefaultAllocator(), table_p.GetTypes(), AppenderType::PHYSICAL), context(context_p),
      table(table_p) {
}

InternalAppender::~InternalAppender() {
	Destructor();
}

Appender::Appender(Connection &con, const string &schema_name, const string &table_name)
    : BaseAppender(Allocator::DefaultAllocator(), AppenderType::LOGICAL), context(con.context) {
	description = con.TableInfo(schema_name, table_name);
	if (!description) {
		// table could not be found
		throw CatalogException(StringUtil::Format("Table \"%s.%s\" could not be found", schema_name, table_name));
	}
	for (auto &column : description->columns) {
		types.push_back(column.Type());
	}
	InitializeChunk();
	collection = make_uniq<ColumnDataCollection>(allocator, types);
}

Appender::Appender(Connection &con, const string &table_name) : Appender(con, DEFAULT_SCHEMA, table_name) {
}

Appender::~Appender() {
	Destructor();
}

void BaseAppender::InitializeChunk() {
	chunk.Initialize(allocator, types);
}

void BaseAppender::BeginRow() {
}

void BaseAppender::EndRow() {
	// check that all rows have been appended to
	if (column != chunk.ColumnCount()) {
		throw InvalidInputException("Call to EndRow before all rows have been appended to!");
	}
	column = 0;
	chunk.SetCardinality(chunk.size() + 1);
	if (chunk.size() >= STANDARD_VECTOR_SIZE) {
		FlushChunk();
	}
}

template <class SRC, class DST>
void BaseAppender::AppendValueInternal(Vector &col, SRC input) {
	FlatVector::GetData<DST>(col)[chunk.size()] = Cast::Operation<SRC, DST>(input);
}

template <class SRC, class DST>
void BaseAppender::AppendDecimalValueInternal(Vector &col, SRC input) {
	switch (appender_type) {
	case AppenderType::LOGICAL: {
		auto &type = col.GetType();
		D_ASSERT(type.id() == LogicalTypeId::DECIMAL);
		auto width = DecimalType::GetWidth(type);
		auto scale = DecimalType::GetScale(type);
		TryCastToDecimal::Operation<SRC, DST>(input, FlatVector::GetData<DST>(col)[chunk.size()], nullptr, width,
		                                      scale);
		return;
	}
	case AppenderType::PHYSICAL: {
		AppendValueInternal<SRC, DST>(col, input);
		return;
	}
	default:
		throw InternalException("Type not implemented for AppenderType");
	}
}

template <class T>
void BaseAppender::AppendValueInternal(T input) {
	if (column >= types.size()) {
		throw InvalidInputException("Too many appends for chunk!");
	}
	auto &col = chunk.data[column];
	switch (col.GetType().id()) {
	case LogicalTypeId::BOOLEAN:
		AppendValueInternal<T, bool>(col, input);
		break;
	case LogicalTypeId::UTINYINT:
		AppendValueInternal<T, uint8_t>(col, input);
		break;
	case LogicalTypeId::TINYINT:
		AppendValueInternal<T, int8_t>(col, input);
		break;
	case LogicalTypeId::USMALLINT:
		AppendValueInternal<T, uint16_t>(col, input);
		break;
	case LogicalTypeId::SMALLINT:
		AppendValueInternal<T, int16_t>(col, input);
		break;
	case LogicalTypeId::UINTEGER:
		AppendValueInternal<T, uint32_t>(col, input);
		break;
	case LogicalTypeId::INTEGER:
		AppendValueInternal<T, int32_t>(col, input);
		break;
	case LogicalTypeId::UBIGINT:
		AppendValueInternal<T, uint64_t>(col, input);
		break;
	case LogicalTypeId::BIGINT:
		AppendValueInternal<T, int64_t>(col, input);
		break;
	case LogicalTypeId::HUGEINT:
		AppendValueInternal<T, hugeint_t>(col, input);
		break;
	case LogicalTypeId::FLOAT:
		AppendValueInternal<T, float>(col, input);
		break;
	case LogicalTypeId::DOUBLE:
		AppendValueInternal<T, double>(col, input);
		break;
	case LogicalTypeId::DECIMAL:
		switch (col.GetType().InternalType()) {
		case PhysicalType::INT16:
			AppendDecimalValueInternal<T, int16_t>(col, input);
			break;
		case PhysicalType::INT32:
			AppendDecimalValueInternal<T, int32_t>(col, input);
			break;
		case PhysicalType::INT64:
			AppendDecimalValueInternal<T, int64_t>(col, input);
			break;
		case PhysicalType::INT128:
			AppendDecimalValueInternal<T, hugeint_t>(col, input);
			break;
		default:
			throw InternalException("Internal type not recognized for Decimal");
		}
		break;
	case LogicalTypeId::DATE:
		AppendValueInternal<T, date_t>(col, input);
		break;
	case LogicalTypeId::TIMESTAMP:
	case LogicalTypeId::TIMESTAMP_TZ:
		AppendValueInternal<T, timestamp_t>(col, input);
		break;
	case LogicalTypeId::TIME:
	case LogicalTypeId::TIME_TZ:
		AppendValueInternal<T, dtime_t>(col, input);
		break;
	case LogicalTypeId::INTERVAL:
		AppendValueInternal<T, interval_t>(col, input);
		break;
	case LogicalTypeId::VARCHAR:
		FlatVector::GetData<string_t>(col)[chunk.size()] = StringCast::Operation<T>(input, col);
		break;
	default:
		AppendValue(Value::CreateValue<T>(input));
		return;
	}
	column++;
}

template <>
void BaseAppender::Append(bool value) {
	AppendValueInternal<bool>(value);
}

template <>
void BaseAppender::Append(int8_t value) {
	AppendValueInternal<int8_t>(value);
}

template <>
void BaseAppender::Append(int16_t value) {
	AppendValueInternal<int16_t>(value);
}

template <>
void BaseAppender::Append(int32_t value) {
	AppendValueInternal<int32_t>(value);
}

template <>
void BaseAppender::Append(int64_t value) {
	AppendValueInternal<int64_t>(value);
}

template <>
void BaseAppender::Append(hugeint_t value) {
	AppendValueInternal<hugeint_t>(value);
}

template <>
void BaseAppender::Append(uint8_t value) {
	AppendValueInternal<uint8_t>(value);
}

template <>
void BaseAppender::Append(uint16_t value) {
	AppendValueInternal<uint16_t>(value);
}

template <>
void BaseAppender::Append(uint32_t value) {
	AppendValueInternal<uint32_t>(value);
}

template <>
void BaseAppender::Append(uint64_t value) {
	AppendValueInternal<uint64_t>(value);
}

template <>
void BaseAppender::Append(const char *value) {
	AppendValueInternal<string_t>(string_t(value));
}

void BaseAppender::Append(const char *value, uint32_t length) {
	AppendValueInternal<string_t>(string_t(value, length));
}

template <>
void BaseAppender::Append(string_t value) {
	AppendValueInternal<string_t>(value);
}

template <>
void BaseAppender::Append(float value) {
	AppendValueInternal<float>(value);
}

template <>
void BaseAppender::Append(double value) {
	AppendValueInternal<double>(value);
}

template <>
void BaseAppender::Append(date_t value) {
	AppendValueInternal<date_t>(value);
}

template <>
void BaseAppender::Append(dtime_t value) {
	AppendValueInternal<dtime_t>(value);
}

template <>
void BaseAppender::Append(timestamp_t value) {
	AppendValueInternal<timestamp_t>(value);
}

template <>
void BaseAppender::Append(interval_t value) {
	AppendValueInternal<interval_t>(value);
}

template <>
void BaseAppender::Append(Value value) { // NOLINT: template shtuff
	if (column >= chunk.ColumnCount()) {
		throw InvalidInputException("Too many appends for chunk!");
	}
	AppendValue(value);
}

template <>
void BaseAppender::Append(std::nullptr_t value) {
	if (column >= chunk.ColumnCount()) {
		throw InvalidInputException("Too many appends for chunk!");
	}
	auto &col = chunk.data[column++];
	FlatVector::SetNull(col, chunk.size(), true);
}

void BaseAppender::AppendValue(const Value &value) {
	chunk.SetValue(column, chunk.size(), value);
	column++;
}

void BaseAppender::AppendDataChunk(DataChunk &chunk) {
	if (chunk.GetTypes() != types) {
		throw InvalidInputException("Type mismatch in Append DataChunk and the types required for appender");
	}
	collection->Append(chunk);
	if (collection->Count() >= FLUSH_COUNT) {
		Flush();
	}
}

void BaseAppender::FlushChunk() {
	if (chunk.size() == 0) {
		return;
	}
	collection->Append(chunk);
	chunk.Reset();
	if (collection->Count() >= FLUSH_COUNT) {
		Flush();
	}
}

void BaseAppender::Flush() {
	// check that all vectors have the same length before appending
	if (column != 0) {
		throw InvalidInputException("Failed to Flush appender: incomplete append to row!");
	}

	FlushChunk();
	if (collection->Count() == 0) {
		return;
	}
	FlushInternal(*collection);

	collection->Reset();
	column = 0;
}

void Appender::FlushInternal(ColumnDataCollection &collection) {
	context->Append(*description, collection);
}

void InternalAppender::FlushInternal(ColumnDataCollection &collection) {
	table.GetStorage().LocalAppend(table, context, collection);
}

void BaseAppender::Close() {
	if (column == 0 || column == types.size()) {
		Flush();
	}
}

} // namespace duckdb









namespace duckdb {

AttachedDatabase::AttachedDatabase(DatabaseInstance &db, AttachedDatabaseType type)
    : CatalogEntry(CatalogType::DATABASE_ENTRY,
                   type == AttachedDatabaseType::SYSTEM_DATABASE ? SYSTEM_CATALOG : TEMP_CATALOG, 0),
      db(db), type(type) {
	D_ASSERT(type == AttachedDatabaseType::TEMP_DATABASE || type == AttachedDatabaseType::SYSTEM_DATABASE);
	if (type == AttachedDatabaseType::TEMP_DATABASE) {
		storage = make_uniq<SingleFileStorageManager>(*this, ":memory:", false);
	}
	catalog = make_uniq<DuckCatalog>(*this);
	transaction_manager = make_uniq<DuckTransactionManager>(*this);
	internal = true;
}

AttachedDatabase::AttachedDatabase(DatabaseInstance &db, Catalog &catalog_p, string name_p, string file_path_p,
                                   AccessMode access_mode)
    : CatalogEntry(CatalogType::DATABASE_ENTRY, catalog_p, std::move(name_p)), db(db),
      type(access_mode == AccessMode::READ_ONLY ? AttachedDatabaseType::READ_ONLY_DATABASE
                                                : AttachedDatabaseType::READ_WRITE_DATABASE),
      parent_catalog(&catalog_p) {
	storage = make_uniq<SingleFileStorageManager>(*this, std::move(file_path_p), access_mode == AccessMode::READ_ONLY);
	catalog = make_uniq<DuckCatalog>(*this);
	transaction_manager = make_uniq<DuckTransactionManager>(*this);
	internal = true;
}

AttachedDatabase::AttachedDatabase(DatabaseInstance &db, Catalog &catalog_p, StorageExtension &storage_extension,
                                   string name_p, AttachInfo &info, AccessMode access_mode)
    : CatalogEntry(CatalogType::DATABASE_ENTRY, catalog_p, std::move(name_p)), db(db),
      type(access_mode == AccessMode::READ_ONLY ? AttachedDatabaseType::READ_ONLY_DATABASE
                                                : AttachedDatabaseType::READ_WRITE_DATABASE),
      parent_catalog(&catalog_p) {
	catalog = storage_extension.attach(storage_extension.storage_info.get(), *this, name, info, access_mode);
	if (!catalog) {
		throw InternalException("AttachedDatabase - attach function did not return a catalog");
	}
	transaction_manager =
	    storage_extension.create_transaction_manager(storage_extension.storage_info.get(), *this, *catalog);
	if (!transaction_manager) {
		throw InternalException(
		    "AttachedDatabase - create_transaction_manager function did not return a transaction manager");
	}
	internal = true;
}

AttachedDatabase::~AttachedDatabase() {
	if (Exception::UncaughtException()) {
		return;
	}
	if (!storage) {
		return;
	}

	// shutting down: attempt to checkpoint the database
	// but only if we are not cleaning up as part of an exception unwind
	try {
		if (!storage->InMemory()) {
			auto &config = DBConfig::GetConfig(db);
			if (!config.options.checkpoint_on_shutdown) {
				return;
			}
			storage->CreateCheckpoint(true);
		}
	} catch (...) {
	}
}

bool AttachedDatabase::IsSystem() const {
	D_ASSERT(!storage || type != AttachedDatabaseType::SYSTEM_DATABASE);
	return type == AttachedDatabaseType::SYSTEM_DATABASE;
}

bool AttachedDatabase::IsTemporary() const {
	return type == AttachedDatabaseType::TEMP_DATABASE;
}
bool AttachedDatabase::IsReadOnly() const {
	return type == AttachedDatabaseType::READ_ONLY_DATABASE;
}

string AttachedDatabase::ExtractDatabaseName(const string &dbpath) {
	if (dbpath.empty() || dbpath == ":memory:") {
		return "memory";
	}
	return FileSystem::ExtractBaseName(dbpath);
}

void AttachedDatabase::Initialize() {
	if (IsSystem()) {
		catalog->Initialize(true);
	} else {
		catalog->Initialize(false);
	}
	if (storage) {
		storage->Initialize();
	}
}

StorageManager &AttachedDatabase::GetStorageManager() {
	if (!storage) {
		throw InternalException("Internal system catalog does not have storage");
	}
	return *storage;
}

Catalog &AttachedDatabase::GetCatalog() {
	return *catalog;
}

TransactionManager &AttachedDatabase::GetTransactionManager() {
	return *transaction_manager;
}

Catalog &AttachedDatabase::ParentCatalog() {
	return *parent_catalog;
}

} // namespace duckdb


using duckdb::Appender;
using duckdb::AppenderWrapper;
using duckdb::Connection;
using duckdb::date_t;
using duckdb::dtime_t;
using duckdb::hugeint_t;
using duckdb::interval_t;
using duckdb::string_t;
using duckdb::timestamp_t;

duckdb_state duckdb_appender_create(duckdb_connection connection, const char *schema, const char *table,
                                    duckdb_appender *out_appender) {
	Connection *conn = (Connection *)connection;

	if (!connection || !table || !out_appender) {
		return DuckDBError;
	}
	if (schema == nullptr) {
		schema = DEFAULT_SCHEMA;
	}
	auto wrapper = new AppenderWrapper();
	*out_appender = (duckdb_appender)wrapper;
	try {
		wrapper->appender = duckdb::make_uniq<Appender>(*conn, schema, table);
	} catch (std::exception &ex) {
		wrapper->error = ex.what();
		return DuckDBError;
	} catch (...) { // LCOV_EXCL_START
		wrapper->error = "Unknown create appender error";
		return DuckDBError;
	} // LCOV_EXCL_STOP
	return DuckDBSuccess;
}

duckdb_state duckdb_appender_destroy(duckdb_appender *appender) {
	if (!appender || !*appender) {
		return DuckDBError;
	}
	duckdb_appender_close(*appender);
	auto wrapper = (AppenderWrapper *)*appender;
	if (wrapper) {
		delete wrapper;
	}
	*appender = nullptr;
	return DuckDBSuccess;
}

template <class FUN>
duckdb_state duckdb_appender_run_function(duckdb_appender appender, FUN &&function) {
	if (!appender) {
		return DuckDBError;
	}
	auto wrapper = (AppenderWrapper *)appender;
	if (!wrapper->appender) {
		return DuckDBError;
	}
	try {
		function(*wrapper->appender);
	} catch (std::exception &ex) {
		wrapper->error = ex.what();
		return DuckDBError;
	} catch (...) { // LCOV_EXCL_START
		wrapper->error = "Unknown error";
		return DuckDBError;
	} // LCOV_EXCL_STOP
	return DuckDBSuccess;
}

const char *duckdb_appender_error(duckdb_appender appender) {
	if (!appender) {
		return nullptr;
	}
	auto wrapper = (AppenderWrapper *)appender;
	if (wrapper->error.empty()) {
		return nullptr;
	}
	return wrapper->error.c_str();
}

duckdb_state duckdb_appender_begin_row(duckdb_appender appender) {
	return DuckDBSuccess;
}

duckdb_state duckdb_appender_end_row(duckdb_appender appender) {
	return duckdb_appender_run_function(appender, [&](Appender &appender) { appender.EndRow(); });
}

template <class T>
duckdb_state duckdb_append_internal(duckdb_appender appender, T value) {
	if (!appender) {
		return DuckDBError;
	}
	auto *appender_instance = (AppenderWrapper *)appender;
	try {
		appender_instance->appender->Append<T>(value);
	} catch (std::exception &ex) {
		appender_instance->error = ex.what();
		return DuckDBError;
	} catch (...) {
		return DuckDBError;
	}
	return DuckDBSuccess;
}

duckdb_state duckdb_append_bool(duckdb_appender appender, bool value) {
	return duckdb_append_internal<bool>(appender, value);
}

duckdb_state duckdb_append_int8(duckdb_appender appender, int8_t value) {
	return duckdb_append_internal<int8_t>(appender, value);
}

duckdb_state duckdb_append_int16(duckdb_appender appender, int16_t value) {
	return duckdb_append_internal<int16_t>(appender, value);
}

duckdb_state duckdb_append_int32(duckdb_appender appender, int32_t value) {
	return duckdb_append_internal<int32_t>(appender, value);
}

duckdb_state duckdb_append_int64(duckdb_appender appender, int64_t value) {
	return duckdb_append_internal<int64_t>(appender, value);
}

duckdb_state duckdb_append_hugeint(duckdb_appender appender, duckdb_hugeint value) {
	hugeint_t internal;
	internal.lower = value.lower;
	internal.upper = value.upper;
	return duckdb_append_internal<hugeint_t>(appender, internal);
}

duckdb_state duckdb_append_uint8(duckdb_appender appender, uint8_t value) {
	return duckdb_append_internal<uint8_t>(appender, value);
}

duckdb_state duckdb_append_uint16(duckdb_appender appender, uint16_t value) {
	return duckdb_append_internal<uint16_t>(appender, value);
}

duckdb_state duckdb_append_uint32(duckdb_appender appender, uint32_t value) {
	return duckdb_append_internal<uint32_t>(appender, value);
}

duckdb_state duckdb_append_uint64(duckdb_appender appender, uint64_t value) {
	return duckdb_append_internal<uint64_t>(appender, value);
}

duckdb_state duckdb_append_float(duckdb_appender appender, float value) {
	return duckdb_append_internal<float>(appender, value);
}

duckdb_state duckdb_append_double(duckdb_appender appender, double value) {
	return duckdb_append_internal<double>(appender, value);
}

duckdb_state duckdb_append_date(duckdb_appender appender, duckdb_date value) {
	return duckdb_append_internal<date_t>(appender, date_t(value.days));
}

duckdb_state duckdb_append_time(duckdb_appender appender, duckdb_time value) {
	return duckdb_append_internal<dtime_t>(appender, dtime_t(value.micros));
}

duckdb_state duckdb_append_timestamp(duckdb_appender appender, duckdb_timestamp value) {
	return duckdb_append_internal<timestamp_t>(appender, timestamp_t(value.micros));
}

duckdb_state duckdb_append_interval(duckdb_appender appender, duckdb_interval value) {
	interval_t interval;
	interval.months = value.months;
	interval.days = value.days;
	interval.micros = value.micros;
	return duckdb_append_internal<interval_t>(appender, interval);
}

duckdb_state duckdb_append_null(duckdb_appender appender) {
	return duckdb_append_internal<std::nullptr_t>(appender, nullptr);
}

duckdb_state duckdb_append_varchar(duckdb_appender appender, const char *val) {
	return duckdb_append_internal<const char *>(appender, val);
}

duckdb_state duckdb_append_varchar_length(duckdb_appender appender, const char *val, idx_t length) {
	return duckdb_append_internal<string_t>(appender, string_t(val, length));
}
duckdb_state duckdb_append_blob(duckdb_appender appender, const void *data, idx_t length) {
	auto value = duckdb::Value::BLOB((duckdb::const_data_ptr_t)data, length);
	return duckdb_append_internal<duckdb::Value>(appender, value);
}

duckdb_state duckdb_appender_flush(duckdb_appender appender) {
	return duckdb_appender_run_function(appender, [&](Appender &appender) { appender.Flush(); });
}

duckdb_state duckdb_appender_close(duckdb_appender appender) {
	return duckdb_appender_run_function(appender, [&](Appender &appender) { appender.Close(); });
}

duckdb_state duckdb_append_data_chunk(duckdb_appender appender, duckdb_data_chunk chunk) {
	if (!chunk) {
		return DuckDBError;
	}
	auto data_chunk = (duckdb::DataChunk *)chunk;
	return duckdb_appender_run_function(appender, [&](Appender &appender) { appender.AppendDataChunk(*data_chunk); });
}



using duckdb::ArrowConverter;
using duckdb::ArrowResultWrapper;
using duckdb::Connection;
using duckdb::DataChunk;
using duckdb::LogicalType;
using duckdb::MaterializedQueryResult;
using duckdb::PreparedStatementWrapper;
using duckdb::QueryResult;
using duckdb::QueryResultType;

duckdb_state duckdb_query_arrow(duckdb_connection connection, const char *query, duckdb_arrow *out_result) {
	Connection *conn = (Connection *)connection;
	auto wrapper = new ArrowResultWrapper();
	wrapper->result = conn->Query(query);
	*out_result = (duckdb_arrow)wrapper;
	return !wrapper->result->HasError() ? DuckDBSuccess : DuckDBError;
}

duckdb_state duckdb_query_arrow_schema(duckdb_arrow result, duckdb_arrow_schema *out_schema) {
	if (!out_schema) {
		return DuckDBSuccess;
	}
	auto wrapper = (ArrowResultWrapper *)result;
	ArrowConverter::ToArrowSchema((ArrowSchema *)*out_schema, wrapper->result->types, wrapper->result->names,
	                              wrapper->timezone_config);
	return DuckDBSuccess;
}

duckdb_state duckdb_query_arrow_array(duckdb_arrow result, duckdb_arrow_array *out_array) {
	if (!out_array) {
		return DuckDBSuccess;
	}
	auto wrapper = (ArrowResultWrapper *)result;
	auto success = wrapper->result->TryFetch(wrapper->current_chunk, wrapper->result->GetErrorObject());
	if (!success) { // LCOV_EXCL_START
		return DuckDBError;
	} // LCOV_EXCL_STOP
	if (!wrapper->current_chunk || wrapper->current_chunk->size() == 0) {
		return DuckDBSuccess;
	}
	ArrowConverter::ToArrowArray(*wrapper->current_chunk, (ArrowArray *)*out_array);
	return DuckDBSuccess;
}

idx_t duckdb_arrow_row_count(duckdb_arrow result) {
	auto wrapper = (ArrowResultWrapper *)result;
	if (wrapper->result->HasError()) {
		return 0;
	}
	return wrapper->result->RowCount();
}

idx_t duckdb_arrow_column_count(duckdb_arrow result) {
	auto wrapper = (ArrowResultWrapper *)result;
	return wrapper->result->ColumnCount();
}

idx_t duckdb_arrow_rows_changed(duckdb_arrow result) {
	auto wrapper = (ArrowResultWrapper *)result;
	if (wrapper->result->HasError()) {
		return 0;
	}
	idx_t rows_changed = 0;
	auto &collection = wrapper->result->Collection();
	idx_t row_count = collection.Count();
	if (row_count > 0 && wrapper->result->properties.return_type == duckdb::StatementReturnType::CHANGED_ROWS) {
		auto rows = collection.GetRows();
		D_ASSERT(row_count == 1);
		D_ASSERT(rows.size() == 1);
		rows_changed = rows[0].GetValue(0).GetValue<int64_t>();
	}
	return rows_changed;
}

const char *duckdb_query_arrow_error(duckdb_arrow result) {
	auto wrapper = (ArrowResultWrapper *)result;
	return wrapper->result->GetError().c_str();
}

void duckdb_destroy_arrow(duckdb_arrow *result) {
	if (*result) {
		auto wrapper = (ArrowResultWrapper *)*result;
		delete wrapper;
		*result = nullptr;
	}
}

duckdb_state duckdb_execute_prepared_arrow(duckdb_prepared_statement prepared_statement, duckdb_arrow *out_result) {
	auto wrapper = (PreparedStatementWrapper *)prepared_statement;
	if (!wrapper || !wrapper->statement || wrapper->statement->HasError() || !out_result) {
		return DuckDBError;
	}
	auto arrow_wrapper = new ArrowResultWrapper();
	if (wrapper->statement->context->config.set_variables.find("TimeZone") ==
	    wrapper->statement->context->config.set_variables.end()) {
		arrow_wrapper->timezone_config = "UTC";
	} else {
		arrow_wrapper->timezone_config =
		    wrapper->statement->context->config.set_variables["TimeZone"].GetValue<std::string>();
	}

	auto result = wrapper->statement->Execute(wrapper->values, false);
	D_ASSERT(result->type == QueryResultType::MATERIALIZED_RESULT);
	arrow_wrapper->result = duckdb::unique_ptr_cast<QueryResult, MaterializedQueryResult>(std::move(result));
	*out_result = (duckdb_arrow)arrow_wrapper;
	return !arrow_wrapper->result->HasError() ? DuckDBSuccess : DuckDBError;
}



namespace duckdb {

//! DECIMAL -> VARCHAR
template <>
bool CastDecimalCInternal(duckdb_result *source, duckdb_string &result, idx_t col, idx_t row) {
	auto result_data = (duckdb::DuckDBResultData *)source->internal_data;
	auto &query_result = result_data->result;
	auto &source_type = query_result->types[col];
	auto width = duckdb::DecimalType::GetWidth(source_type);
	auto scale = duckdb::DecimalType::GetScale(source_type);
	duckdb::Vector result_vec(duckdb::LogicalType::VARCHAR, false, false);
	duckdb::string_t result_string;
	void *source_address = UnsafeFetchPtr<hugeint_t>(source, col, row);
	switch (source_type.InternalType()) {
	case duckdb::PhysicalType::INT16:
		result_string = duckdb::StringCastFromDecimal::Operation<int16_t>(UnsafeFetchFromPtr<int16_t>(source_address),
		                                                                  width, scale, result_vec);
		break;
	case duckdb::PhysicalType::INT32:
		result_string = duckdb::StringCastFromDecimal::Operation<int32_t>(UnsafeFetchFromPtr<int32_t>(source_address),
		                                                                  width, scale, result_vec);
		break;
	case duckdb::PhysicalType::INT64:
		result_string = duckdb::StringCastFromDecimal::Operation<int64_t>(UnsafeFetchFromPtr<int64_t>(source_address),
		                                                                  width, scale, result_vec);
		break;
	case duckdb::PhysicalType::INT128:
		result_string = duckdb::StringCastFromDecimal::Operation<hugeint_t>(
		    UnsafeFetchFromPtr<hugeint_t>(source_address), width, scale, result_vec);
		break;
	default:
		throw duckdb::InternalException("Unimplemented internal type for decimal");
	}
	result.data = (char *)duckdb_malloc(sizeof(char) * (result_string.GetSize() + 1));
	memcpy(result.data, result_string.GetData(), result_string.GetSize());
	result.data[result_string.GetSize()] = '\0';
	result.size = result_string.GetSize();
	return true;
}

template <class INTERNAL_TYPE>
duckdb_hugeint FetchInternals(void *source_address) {
	throw duckdb::NotImplementedException("FetchInternals not implemented for internal type");
}

template <>
duckdb_hugeint FetchInternals<int16_t>(void *source_address) {
	duckdb_hugeint result;
	int16_t intermediate_result;

	if (!TryCast::Operation<int16_t, int16_t>(UnsafeFetchFromPtr<int16_t>(source_address), intermediate_result)) {
		intermediate_result = FetchDefaultValue::Operation<int16_t>();
	}
	hugeint_t hugeint_result = Hugeint::Cast<int16_t>(intermediate_result);
	result.lower = hugeint_result.lower;
	result.upper = hugeint_result.upper;
	return result;
}
template <>
duckdb_hugeint FetchInternals<int32_t>(void *source_address) {
	duckdb_hugeint result;
	int32_t intermediate_result;

	if (!TryCast::Operation<int32_t, int32_t>(UnsafeFetchFromPtr<int32_t>(source_address), intermediate_result)) {
		intermediate_result = FetchDefaultValue::Operation<int32_t>();
	}
	hugeint_t hugeint_result = Hugeint::Cast<int32_t>(intermediate_result);
	result.lower = hugeint_result.lower;
	result.upper = hugeint_result.upper;
	return result;
}
template <>
duckdb_hugeint FetchInternals<int64_t>(void *source_address) {
	duckdb_hugeint result;
	int64_t intermediate_result;

	if (!TryCast::Operation<int64_t, int64_t>(UnsafeFetchFromPtr<int64_t>(source_address), intermediate_result)) {
		intermediate_result = FetchDefaultValue::Operation<int64_t>();
	}
	hugeint_t hugeint_result = Hugeint::Cast<int64_t>(intermediate_result);
	result.lower = hugeint_result.lower;
	result.upper = hugeint_result.upper;
	return result;
}
template <>
duckdb_hugeint FetchInternals<hugeint_t>(void *source_address) {
	duckdb_hugeint result;
	hugeint_t intermediate_result;

	if (!TryCast::Operation<hugeint_t, hugeint_t>(UnsafeFetchFromPtr<hugeint_t>(source_address), intermediate_result)) {
		intermediate_result = FetchDefaultValue::Operation<hugeint_t>();
	}
	result.lower = intermediate_result.lower;
	result.upper = intermediate_result.upper;
	return result;
}

//! DECIMAL -> DECIMAL (internal fetch)
template <>
bool CastDecimalCInternal(duckdb_result *source, duckdb_decimal &result, idx_t col, idx_t row) {
	auto result_data = (duckdb::DuckDBResultData *)source->internal_data;
	result_data->result->types[col].GetDecimalProperties(result.width, result.scale);
	auto source_address = UnsafeFetchPtr<hugeint_t>(source, col, row);

	if (result.width > duckdb::Decimal::MAX_WIDTH_INT64) {
		result.value = FetchInternals<hugeint_t>(source_address);
	} else if (result.width > duckdb::Decimal::MAX_WIDTH_INT32) {
		result.value = FetchInternals<int64_t>(source_address);
	} else if (result.width > duckdb::Decimal::MAX_WIDTH_INT16) {
		result.value = FetchInternals<int32_t>(source_address);
	} else {
		result.value = FetchInternals<int16_t>(source_address);
	}
	return true;
}

} // namespace duckdb


namespace duckdb {

template <>
duckdb_decimal FetchDefaultValue::Operation() {
	duckdb_decimal result;
	result.scale = 0;
	result.width = 0;
	result.value = {0, 0};
	return result;
}

template <>
date_t FetchDefaultValue::Operation() {
	date_t result;
	result.days = 0;
	return result;
}

template <>
dtime_t FetchDefaultValue::Operation() {
	dtime_t result;
	result.micros = 0;
	return result;
}

template <>
timestamp_t FetchDefaultValue::Operation() {
	timestamp_t result;
	result.value = 0;
	return result;
}

template <>
interval_t FetchDefaultValue::Operation() {
	interval_t result;
	result.months = 0;
	result.days = 0;
	result.micros = 0;
	return result;
}

template <>
char *FetchDefaultValue::Operation() {
	return nullptr;
}

template <>
duckdb_string FetchDefaultValue::Operation() {
	duckdb_string result;
	result.data = nullptr;
	result.size = 0;
	return result;
}

template <>
duckdb_blob FetchDefaultValue::Operation() {
	duckdb_blob result;
	result.data = nullptr;
	result.size = 0;
	return result;
}

//===--------------------------------------------------------------------===//
// Blob Casts
//===--------------------------------------------------------------------===//

template <>
bool FromCBlobCastWrapper::Operation(duckdb_blob input, duckdb_string &result) {
	string_t input_str((const char *)input.data, input.size);
	return ToCStringCastWrapper<duckdb::CastFromBlob>::template Operation<string_t, duckdb_string>(input_str, result);
}

} // namespace duckdb

bool CanUseDeprecatedFetch(duckdb_result *result, idx_t col, idx_t row) {
	if (!result) {
		return false;
	}
	if (!duckdb::deprecated_materialize_result(result)) {
		return false;
	}
	if (col >= result->__deprecated_column_count || row >= result->__deprecated_row_count) {
		return false;
	}
	return true;
}

bool CanFetchValue(duckdb_result *result, idx_t col, idx_t row) {
	if (!CanUseDeprecatedFetch(result, col, row)) {
		return false;
	}
	if (result->__deprecated_columns[col].__deprecated_nullmask[row]) {
		return false;
	}
	return true;
}




using duckdb::DBConfig;
using duckdb::Value;

// config
duckdb_state duckdb_create_config(duckdb_config *out_config) {
	if (!out_config) {
		return DuckDBError;
	}
	DBConfig *config;
	try {
		config = new DBConfig();
	} catch (...) { // LCOV_EXCL_START
		return DuckDBError;
	} // LCOV_EXCL_STOP
	*out_config = (duckdb_config)config;
	return DuckDBSuccess;
}

size_t duckdb_config_count() {
	return DBConfig::GetOptionCount();
}

duckdb_state duckdb_get_config_flag(size_t index, const char **out_name, const char **out_description) {
	auto option = DBConfig::GetOptionByIndex(index);
	if (!option) {
		return DuckDBError;
	}
	if (out_name) {
		*out_name = option->name;
	}
	if (out_description) {
		*out_description = option->description;
	}
	return DuckDBSuccess;
}

duckdb_state duckdb_set_config(duckdb_config config, const char *name, const char *option) {
	if (!config || !name || !option) {
		return DuckDBError;
	}
	auto config_option = DBConfig::GetOptionByName(name);
	if (!config_option) {
		return DuckDBError;
	}
	try {
		auto db_config = (DBConfig *)config;
		db_config->SetOption(*config_option, Value(option));
	} catch (...) {
		return DuckDBError;
	}
	return DuckDBSuccess;
}

void duckdb_destroy_config(duckdb_config *config) {
	if (!config) {
		return;
	}
	if (*config) {
		auto db_config = (DBConfig *)*config;
		delete db_config;
		*config = nullptr;
	}
}




#include <string.h>

duckdb_data_chunk duckdb_create_data_chunk(duckdb_logical_type *ctypes, idx_t column_count) {
	if (!ctypes) {
		return nullptr;
	}
	duckdb::vector<duckdb::LogicalType> types;
	for (idx_t i = 0; i < column_count; i++) {
		auto ltype = (duckdb::LogicalType *)ctypes[i];
		types.push_back(*ltype);
	}

	auto result = new duckdb::DataChunk();
	result->Initialize(duckdb::Allocator::DefaultAllocator(), types);
	return reinterpret_cast<duckdb_data_chunk>(result);
}

void duckdb_destroy_data_chunk(duckdb_data_chunk *chunk) {
	if (chunk && *chunk) {
		auto dchunk = (duckdb::DataChunk *)*chunk;
		delete dchunk;
		*chunk = nullptr;
	}
}

void duckdb_data_chunk_reset(duckdb_data_chunk chunk) {
	if (!chunk) {
		return;
	}
	auto dchunk = (duckdb::DataChunk *)chunk;
	dchunk->Reset();
}

idx_t duckdb_data_chunk_get_column_count(duckdb_data_chunk chunk) {
	if (!chunk) {
		return 0;
	}
	auto dchunk = (duckdb::DataChunk *)chunk;
	return dchunk->ColumnCount();
}

duckdb_vector duckdb_data_chunk_get_vector(duckdb_data_chunk chunk, idx_t col_idx) {
	if (!chunk || col_idx >= duckdb_data_chunk_get_column_count(chunk)) {
		return nullptr;
	}
	auto dchunk = (duckdb::DataChunk *)chunk;
	return reinterpret_cast<duckdb_vector>(&dchunk->data[col_idx]);
}

idx_t duckdb_data_chunk_get_size(duckdb_data_chunk chunk) {
	if (!chunk) {
		return 0;
	}
	auto dchunk = (duckdb::DataChunk *)chunk;
	return dchunk->size();
}

void duckdb_data_chunk_set_size(duckdb_data_chunk chunk, idx_t size) {
	if (!chunk) {
		return;
	}
	auto dchunk = (duckdb::DataChunk *)chunk;
	dchunk->SetCardinality(size);
}

duckdb_logical_type duckdb_vector_get_column_type(duckdb_vector vector) {
	if (!vector) {
		return nullptr;
	}
	auto v = (duckdb::Vector *)vector;
	return reinterpret_cast<duckdb_logical_type>(new duckdb::LogicalType(v->GetType()));
}

void *duckdb_vector_get_data(duckdb_vector vector) {
	if (!vector) {
		return nullptr;
	}
	auto v = (duckdb::Vector *)vector;
	return duckdb::FlatVector::GetData(*v);
}

uint64_t *duckdb_vector_get_validity(duckdb_vector vector) {
	if (!vector) {
		return nullptr;
	}
	auto v = (duckdb::Vector *)vector;
	return duckdb::FlatVector::Validity(*v).GetData();
}

void duckdb_vector_ensure_validity_writable(duckdb_vector vector) {
	if (!vector) {
		return;
	}
	auto v = (duckdb::Vector *)vector;
	auto &validity = duckdb::FlatVector::Validity(*v);
	validity.EnsureWritable();
}

void duckdb_vector_assign_string_element(duckdb_vector vector, idx_t index, const char *str) {
	duckdb_vector_assign_string_element_len(vector, index, str, strlen(str));
}

void duckdb_vector_assign_string_element_len(duckdb_vector vector, idx_t index, const char *str, idx_t str_len) {
	if (!vector) {
		return;
	}
	auto v = (duckdb::Vector *)vector;
	auto data = duckdb::FlatVector::GetData<duckdb::string_t>(*v);
	data[index] = duckdb::StringVector::AddString(*v, str, str_len);
}

duckdb_vector duckdb_list_vector_get_child(duckdb_vector vector) {
	if (!vector) {
		return nullptr;
	}
	auto v = (duckdb::Vector *)vector;
	return reinterpret_cast<duckdb_vector>(&duckdb::ListVector::GetEntry(*v));
}

idx_t duckdb_list_vector_get_size(duckdb_vector vector) {
	if (!vector) {
		return 0;
	}
	auto v = (duckdb::Vector *)vector;
	return duckdb::ListVector::GetListSize(*v);
}

duckdb_state duckdb_list_vector_set_size(duckdb_vector vector, idx_t size) {
	if (!vector) {
		return duckdb_state::DuckDBError;
	}
	auto v = (duckdb::Vector *)vector;
	duckdb::ListVector::SetListSize(*v, size);
	return duckdb_state::DuckDBSuccess;
}

duckdb_state duckdb_list_vector_reserve(duckdb_vector vector, idx_t required_capacity) {
	if (!vector) {
		return duckdb_state::DuckDBError;
	}
	auto v = (duckdb::Vector *)vector;
	duckdb::ListVector::Reserve(*v, required_capacity);
	return duckdb_state::DuckDBSuccess;
}

duckdb_vector duckdb_struct_vector_get_child(duckdb_vector vector, idx_t index) {
	if (!vector) {
		return nullptr;
	}
	auto v = (duckdb::Vector *)vector;
	return reinterpret_cast<duckdb_vector>(duckdb::StructVector::GetEntries(*v)[index].get());
}

bool duckdb_validity_row_is_valid(uint64_t *validity, idx_t row) {
	if (!validity) {
		return true;
	}
	idx_t entry_idx = row / 64;
	idx_t idx_in_entry = row % 64;
	return validity[entry_idx] & ((idx_t)1 << idx_in_entry);
}

void duckdb_validity_set_row_validity(uint64_t *validity, idx_t row, bool valid) {
	if (valid) {
		duckdb_validity_set_row_valid(validity, row);
	} else {
		duckdb_validity_set_row_invalid(validity, row);
	}
}

void duckdb_validity_set_row_invalid(uint64_t *validity, idx_t row) {
	if (!validity) {
		return;
	}
	idx_t entry_idx = row / 64;
	idx_t idx_in_entry = row % 64;
	validity[entry_idx] &= ~((uint64_t)1 << idx_in_entry);
}

void duckdb_validity_set_row_valid(uint64_t *validity, idx_t row) {
	if (!validity) {
		return;
	}
	idx_t entry_idx = row / 64;
	idx_t idx_in_entry = row % 64;
	validity[entry_idx] |= (uint64_t)1 << idx_in_entry;
}





using duckdb::Date;
using duckdb::Time;
using duckdb::Timestamp;

using duckdb::date_t;
using duckdb::dtime_t;
using duckdb::timestamp_t;

duckdb_date_struct duckdb_from_date(duckdb_date date) {
	int32_t year, month, day;
	Date::Convert(date_t(date.days), year, month, day);

	duckdb_date_struct result;
	result.year = year;
	result.month = month;
	result.day = day;
	return result;
}

duckdb_date duckdb_to_date(duckdb_date_struct date) {
	duckdb_date result;
	result.days = Date::FromDate(date.year, date.month, date.day).days;
	return result;
}

duckdb_time_struct duckdb_from_time(duckdb_time time) {
	int32_t hour, minute, second, micros;
	Time::Convert(dtime_t(time.micros), hour, minute, second, micros);

	duckdb_time_struct result;
	result.hour = hour;
	result.min = minute;
	result.sec = second;
	result.micros = micros;
	return result;
}

duckdb_time duckdb_to_time(duckdb_time_struct time) {
	duckdb_time result;
	result.micros = Time::FromTime(time.hour, time.min, time.sec, time.micros).micros;
	return result;
}

duckdb_timestamp_struct duckdb_from_timestamp(duckdb_timestamp ts) {
	date_t date;
	dtime_t time;
	Timestamp::Convert(timestamp_t(ts.micros), date, time);

	duckdb_date ddate;
	ddate.days = date.days;

	duckdb_time dtime;
	dtime.micros = time.micros;

	duckdb_timestamp_struct result;
	result.date = duckdb_from_date(ddate);
	result.time = duckdb_from_time(dtime);
	return result;
}

duckdb_timestamp duckdb_to_timestamp(duckdb_timestamp_struct ts) {
	date_t date = date_t(duckdb_to_date(ts.date).days);
	dtime_t time = dtime_t(duckdb_to_time(ts.time).micros);

	duckdb_timestamp result;
	result.micros = Timestamp::FromDatetime(date, time).value;
	return result;
}


using duckdb::Connection;
using duckdb::DatabaseData;
using duckdb::DBConfig;
using duckdb::DuckDB;

duckdb_state duckdb_open_ext(const char *path, duckdb_database *out, duckdb_config config, char **error) {
	auto wrapper = new DatabaseData();
	try {
		auto db_config = (DBConfig *)config;
		wrapper->database = duckdb::make_uniq<DuckDB>(path, db_config);
	} catch (std::exception &ex) {
		if (error) {
			*error = strdup(ex.what());
		}
		delete wrapper;
		return DuckDBError;
	} catch (...) { // LCOV_EXCL_START
		if (error) {
			*error = strdup("Unknown error");
		}
		delete wrapper;
		return DuckDBError;
	} // LCOV_EXCL_STOP
	*out = (duckdb_database)wrapper;
	return DuckDBSuccess;
}

duckdb_state duckdb_open(const char *path, duckdb_database *out) {
	return duckdb_open_ext(path, out, nullptr, nullptr);
}

void duckdb_close(duckdb_database *database) {
	if (database && *database) {
		auto wrapper = (DatabaseData *)*database;
		delete wrapper;
		*database = nullptr;
	}
}

duckdb_state duckdb_connect(duckdb_database database, duckdb_connection *out) {
	if (!database || !out) {
		return DuckDBError;
	}
	auto wrapper = (DatabaseData *)database;
	Connection *connection;
	try {
		connection = new Connection(*wrapper->database);
	} catch (...) { // LCOV_EXCL_START
		return DuckDBError;
	} // LCOV_EXCL_STOP
	*out = (duckdb_connection)connection;
	return DuckDBSuccess;
}

void duckdb_disconnect(duckdb_connection *connection) {
	if (connection && *connection) {
		Connection *conn = (Connection *)*connection;
		delete conn;
		*connection = nullptr;
	}
}

duckdb_state duckdb_query(duckdb_connection connection, const char *query, duckdb_result *out) {
	Connection *conn = (Connection *)connection;
	auto result = conn->Query(query);
	return duckdb_translate_result(std::move(result), out);
}

const char *duckdb_library_version() {
	return DuckDB::LibraryVersion();
}


void duckdb_destroy_value(duckdb_value *value) {
	if (value && *value) {
		auto val = (duckdb::Value *)*value;
		delete val;
		*value = nullptr;
	}
}

duckdb_value duckdb_create_varchar_length(const char *text, idx_t length) {
	return reinterpret_cast<duckdb_value>(new duckdb::Value(std::string(text, length)));
}

duckdb_value duckdb_create_varchar(const char *text) {
	return duckdb_create_varchar_length(text, strlen(text));
}

duckdb_value duckdb_create_int64(int64_t input) {
	auto val = duckdb::Value::BIGINT(input);
	return reinterpret_cast<duckdb_value>(new duckdb::Value(val));
}

char *duckdb_get_varchar(duckdb_value value) {
	auto val = (duckdb::Value *)value;
	auto str_val = val->DefaultCastAs(duckdb::LogicalType::VARCHAR);
	auto &str = duckdb::StringValue::Get(str_val);

	auto result = (char *)malloc(sizeof(char *) * (str.size() + 1));
	memcpy(result, str.c_str(), str.size());
	result[str.size()] = '\0';
	return result;
}

int64_t duckdb_get_int64(duckdb_value value) {
	auto val = (duckdb::Value *)value;
	if (!val->DefaultTryCastAs(duckdb::LogicalType::BIGINT)) {
		return 0;
	}
	return duckdb::BigIntValue::Get(*val);
}


namespace duckdb {

LogicalTypeId ConvertCTypeToCPP(duckdb_type c_type) {
	switch (c_type) {
	case DUCKDB_TYPE_BOOLEAN:
		return LogicalTypeId::BOOLEAN;
	case DUCKDB_TYPE_TINYINT:
		return LogicalTypeId::TINYINT;
	case DUCKDB_TYPE_SMALLINT:
		return LogicalTypeId::SMALLINT;
	case DUCKDB_TYPE_INTEGER:
		return LogicalTypeId::INTEGER;
	case DUCKDB_TYPE_BIGINT:
		return LogicalTypeId::BIGINT;
	case DUCKDB_TYPE_UTINYINT:
		return LogicalTypeId::UTINYINT;
	case DUCKDB_TYPE_USMALLINT:
		return LogicalTypeId::USMALLINT;
	case DUCKDB_TYPE_UINTEGER:
		return LogicalTypeId::UINTEGER;
	case DUCKDB_TYPE_UBIGINT:
		return LogicalTypeId::UBIGINT;
	case DUCKDB_TYPE_HUGEINT:
		return LogicalTypeId::HUGEINT;
	case DUCKDB_TYPE_FLOAT:
		return LogicalTypeId::FLOAT;
	case DUCKDB_TYPE_DOUBLE:
		return LogicalTypeId::DOUBLE;
	case DUCKDB_TYPE_TIMESTAMP:
		return LogicalTypeId::TIMESTAMP;
	case DUCKDB_TYPE_DATE:
		return LogicalTypeId::DATE;
	case DUCKDB_TYPE_TIME:
		return LogicalTypeId::TIME;
	case DUCKDB_TYPE_VARCHAR:
		return LogicalTypeId::VARCHAR;
	case DUCKDB_TYPE_BLOB:
		return LogicalTypeId::BLOB;
	case DUCKDB_TYPE_INTERVAL:
		return LogicalTypeId::INTERVAL;
	case DUCKDB_TYPE_TIMESTAMP_S:
		return LogicalTypeId::TIMESTAMP_SEC;
	case DUCKDB_TYPE_TIMESTAMP_MS:
		return LogicalTypeId::TIMESTAMP_MS;
	case DUCKDB_TYPE_TIMESTAMP_NS:
		return LogicalTypeId::TIMESTAMP_NS;
	case DUCKDB_TYPE_UUID:
		return LogicalTypeId::UUID;
	default: // LCOV_EXCL_START
		D_ASSERT(0);
		return LogicalTypeId::INVALID;
	} // LCOV_EXCL_STOP
}

duckdb_type ConvertCPPTypeToC(const LogicalType &sql_type) {
	switch (sql_type.id()) {
	case LogicalTypeId::BOOLEAN:
		return DUCKDB_TYPE_BOOLEAN;
	case LogicalTypeId::TINYINT:
		return DUCKDB_TYPE_TINYINT;
	case LogicalTypeId::SMALLINT:
		return DUCKDB_TYPE_SMALLINT;
	case LogicalTypeId::INTEGER:
		return DUCKDB_TYPE_INTEGER;
	case LogicalTypeId::BIGINT:
		return DUCKDB_TYPE_BIGINT;
	case LogicalTypeId::UTINYINT:
		return DUCKDB_TYPE_UTINYINT;
	case LogicalTypeId::USMALLINT:
		return DUCKDB_TYPE_USMALLINT;
	case LogicalTypeId::UINTEGER:
		return DUCKDB_TYPE_UINTEGER;
	case LogicalTypeId::UBIGINT:
		return DUCKDB_TYPE_UBIGINT;
	case LogicalTypeId::HUGEINT:
		return DUCKDB_TYPE_HUGEINT;
	case LogicalTypeId::FLOAT:
		return DUCKDB_TYPE_FLOAT;
	case LogicalTypeId::DOUBLE:
		return DUCKDB_TYPE_DOUBLE;
	case LogicalTypeId::TIMESTAMP:
	case LogicalTypeId::TIMESTAMP_TZ:
		return DUCKDB_TYPE_TIMESTAMP;
	case LogicalTypeId::TIMESTAMP_SEC:
		return DUCKDB_TYPE_TIMESTAMP_S;
	case LogicalTypeId::TIMESTAMP_MS:
		return DUCKDB_TYPE_TIMESTAMP_MS;
	case LogicalTypeId::TIMESTAMP_NS:
		return DUCKDB_TYPE_TIMESTAMP_NS;
	case LogicalTypeId::DATE:
		return DUCKDB_TYPE_DATE;
	case LogicalTypeId::TIME:
	case LogicalTypeId::TIME_TZ:
		return DUCKDB_TYPE_TIME;
	case LogicalTypeId::VARCHAR:
		return DUCKDB_TYPE_VARCHAR;
	case LogicalTypeId::BLOB:
		return DUCKDB_TYPE_BLOB;
	case LogicalTypeId::BIT:
		return DUCKDB_TYPE_BIT;
	case LogicalTypeId::INTERVAL:
		return DUCKDB_TYPE_INTERVAL;
	case LogicalTypeId::DECIMAL:
		return DUCKDB_TYPE_DECIMAL;
	case LogicalTypeId::ENUM:
		return DUCKDB_TYPE_ENUM;
	case LogicalTypeId::LIST:
		return DUCKDB_TYPE_LIST;
	case LogicalTypeId::STRUCT:
		return DUCKDB_TYPE_STRUCT;
	case LogicalTypeId::MAP:
		return DUCKDB_TYPE_MAP;
	case LogicalTypeId::UNION:
		return DUCKDB_TYPE_UNION;
	case LogicalTypeId::UUID:
		return DUCKDB_TYPE_UUID;
	default: // LCOV_EXCL_START
		D_ASSERT(0);
		return DUCKDB_TYPE_INVALID;
	} // LCOV_EXCL_STOP
}

idx_t GetCTypeSize(duckdb_type type) {
	switch (type) {
	case DUCKDB_TYPE_BOOLEAN:
		return sizeof(bool);
	case DUCKDB_TYPE_TINYINT:
		return sizeof(int8_t);
	case DUCKDB_TYPE_SMALLINT:
		return sizeof(int16_t);
	case DUCKDB_TYPE_INTEGER:
		return sizeof(int32_t);
	case DUCKDB_TYPE_BIGINT:
		return sizeof(int64_t);
	case DUCKDB_TYPE_UTINYINT:
		return sizeof(uint8_t);
	case DUCKDB_TYPE_USMALLINT:
		return sizeof(uint16_t);
	case DUCKDB_TYPE_UINTEGER:
		return sizeof(uint32_t);
	case DUCKDB_TYPE_UBIGINT:
		return sizeof(uint64_t);
	case DUCKDB_TYPE_HUGEINT:
	case DUCKDB_TYPE_UUID:
		return sizeof(duckdb_hugeint);
	case DUCKDB_TYPE_FLOAT:
		return sizeof(float);
	case DUCKDB_TYPE_DOUBLE:
		return sizeof(double);
	case DUCKDB_TYPE_DATE:
		return sizeof(duckdb_date);
	case DUCKDB_TYPE_TIME:
		return sizeof(duckdb_time);
	case DUCKDB_TYPE_TIMESTAMP:
	case DUCKDB_TYPE_TIMESTAMP_S:
	case DUCKDB_TYPE_TIMESTAMP_MS:
	case DUCKDB_TYPE_TIMESTAMP_NS:
		return sizeof(duckdb_timestamp);
	case DUCKDB_TYPE_VARCHAR:
		return sizeof(const char *);
	case DUCKDB_TYPE_BLOB:
		return sizeof(duckdb_blob);
	case DUCKDB_TYPE_INTERVAL:
		return sizeof(duckdb_interval);
	case DUCKDB_TYPE_DECIMAL:
		return sizeof(duckdb_hugeint);
	default: // LCOV_EXCL_START
		// unsupported type
		D_ASSERT(0);
		return sizeof(const char *);
	} // LCOV_EXCL_STOP
}

} // namespace duckdb

void *duckdb_malloc(size_t size) {
	return malloc(size);
}

void duckdb_free(void *ptr) {
	free(ptr);
}

idx_t duckdb_vector_size() {
	return STANDARD_VECTOR_SIZE;
}

bool duckdb_string_is_inlined(duckdb_string_t string_p) {
	static_assert(sizeof(duckdb_string_t) == sizeof(duckdb::string_t),
	              "duckdb_string_t should have the same memory layout as duckdb::string_t");
	auto &string = *(duckdb::string_t *)(&string_p);
	return string.IsInlined();
}







using duckdb::Hugeint;
using duckdb::hugeint_t;
using duckdb::Value;

double duckdb_hugeint_to_double(duckdb_hugeint val) {
	hugeint_t internal;
	internal.lower = val.lower;
	internal.upper = val.upper;
	return Hugeint::Cast<double>(internal);
}

static duckdb_decimal to_decimal_cast(double val, uint8_t width, uint8_t scale) {
	if (width > duckdb::Decimal::MAX_WIDTH_INT64) {
		return duckdb::TryCastToDecimalCInternal<double, duckdb::ToCDecimalCastWrapper<hugeint_t>>(val, width, scale);
	}
	if (width > duckdb::Decimal::MAX_WIDTH_INT32) {
		return duckdb::TryCastToDecimalCInternal<double, duckdb::ToCDecimalCastWrapper<int64_t>>(val, width, scale);
	}
	if (width > duckdb::Decimal::MAX_WIDTH_INT16) {
		return duckdb::TryCastToDecimalCInternal<double, duckdb::ToCDecimalCastWrapper<int32_t>>(val, width, scale);
	}
	return duckdb::TryCastToDecimalCInternal<double, duckdb::ToCDecimalCastWrapper<int16_t>>(val, width, scale);
}

duckdb_decimal duckdb_double_to_decimal(double val, uint8_t width, uint8_t scale) {
	if (scale > width || width > duckdb::Decimal::MAX_WIDTH_INT128) {
		return duckdb::FetchDefaultValue::Operation<duckdb_decimal>();
	}
	return to_decimal_cast(val, width, scale);
}

duckdb_hugeint duckdb_double_to_hugeint(double val) {
	hugeint_t internal_result;
	if (!Value::DoubleIsFinite(val) || !Hugeint::TryConvert<double>(val, internal_result)) {
		internal_result.lower = 0;
		internal_result.upper = 0;
	}

	duckdb_hugeint result;
	result.lower = internal_result.lower;
	result.upper = internal_result.upper;
	return result;
}

double duckdb_decimal_to_double(duckdb_decimal val) {
	double result;
	hugeint_t value;
	value.lower = val.value.lower;
	value.upper = val.value.upper;
	duckdb::TryCastFromDecimal::Operation<hugeint_t, double>(value, result, nullptr, val.width, val.scale);
	return result;
}


static bool AssertLogicalTypeId(duckdb_logical_type type, duckdb::LogicalTypeId type_id) {
	if (!type) {
		return false;
	}
	auto &ltype = *((duckdb::LogicalType *)type);
	if (ltype.id() != type_id) {
		return false;
	}
	return true;
}

static bool AssertInternalType(duckdb_logical_type type, duckdb::PhysicalType physical_type) {
	if (!type) {
		return false;
	}
	auto &ltype = *((duckdb::LogicalType *)type);
	if (ltype.InternalType() != physical_type) {
		return false;
	}
	return true;
}

duckdb_logical_type duckdb_create_logical_type(duckdb_type type) {
	return reinterpret_cast<duckdb_logical_type>(new duckdb::LogicalType(duckdb::ConvertCTypeToCPP(type)));
}

duckdb_logical_type duckdb_create_list_type(duckdb_logical_type type) {
	if (!type) {
		return nullptr;
	}
	duckdb::LogicalType *ltype = new duckdb::LogicalType;
	*ltype = duckdb::LogicalType::LIST(*(duckdb::LogicalType *)type);
	return reinterpret_cast<duckdb_logical_type>(ltype);
}

duckdb_logical_type duckdb_create_union_type(duckdb_logical_type member_types_p, const char **member_names,
                                             idx_t member_count) {
	if (!member_types_p || !member_names) {
		return nullptr;
	}
	duckdb::LogicalType *member_types = (duckdb::LogicalType *)member_types_p;
	duckdb::LogicalType *mtype = new duckdb::LogicalType;
	duckdb::child_list_t<duckdb::LogicalType> members;

	for (idx_t i = 0; i < member_count; i++) {
		members.push_back(make_pair(member_names[i], member_types[i]));
	}
	*mtype = duckdb::LogicalType::UNION(members);
	return reinterpret_cast<duckdb_logical_type>(mtype);
}

duckdb_logical_type duckdb_create_map_type(duckdb_logical_type key_type, duckdb_logical_type value_type) {
	if (!key_type || !value_type) {
		return nullptr;
	}
	duckdb::LogicalType *mtype = new duckdb::LogicalType;
	*mtype = duckdb::LogicalType::MAP(*(duckdb::LogicalType *)key_type, *(duckdb::LogicalType *)value_type);
	return reinterpret_cast<duckdb_logical_type>(mtype);
}

duckdb_logical_type duckdb_create_decimal_type(uint8_t width, uint8_t scale) {
	return reinterpret_cast<duckdb_logical_type>(new duckdb::LogicalType(duckdb::LogicalType::DECIMAL(width, scale)));
}

duckdb_type duckdb_get_type_id(duckdb_logical_type type) {
	if (!type) {
		return DUCKDB_TYPE_INVALID;
	}
	auto ltype = (duckdb::LogicalType *)type;
	return duckdb::ConvertCPPTypeToC(*ltype);
}

void duckdb_destroy_logical_type(duckdb_logical_type *type) {
	if (type && *type) {
		auto ltype = (duckdb::LogicalType *)*type;
		delete ltype;
		*type = nullptr;
	}
}

uint8_t duckdb_decimal_width(duckdb_logical_type type) {
	if (!AssertLogicalTypeId(type, duckdb::LogicalTypeId::DECIMAL)) {
		return 0;
	}
	auto &ltype = *((duckdb::LogicalType *)type);
	return duckdb::DecimalType::GetWidth(ltype);
}

uint8_t duckdb_decimal_scale(duckdb_logical_type type) {
	if (!AssertLogicalTypeId(type, duckdb::LogicalTypeId::DECIMAL)) {
		return 0;
	}
	auto &ltype = *((duckdb::LogicalType *)type);
	return duckdb::DecimalType::GetScale(ltype);
}

duckdb_type duckdb_decimal_internal_type(duckdb_logical_type type) {
	if (!AssertLogicalTypeId(type, duckdb::LogicalTypeId::DECIMAL)) {
		return DUCKDB_TYPE_INVALID;
	}
	auto &ltype = *((duckdb::LogicalType *)type);
	switch (ltype.InternalType()) {
	case duckdb::PhysicalType::INT16:
		return DUCKDB_TYPE_SMALLINT;
	case duckdb::PhysicalType::INT32:
		return DUCKDB_TYPE_INTEGER;
	case duckdb::PhysicalType::INT64:
		return DUCKDB_TYPE_BIGINT;
	case duckdb::PhysicalType::INT128:
		return DUCKDB_TYPE_HUGEINT;
	default:
		return DUCKDB_TYPE_INVALID;
	}
}

duckdb_type duckdb_enum_internal_type(duckdb_logical_type type) {
	if (!AssertLogicalTypeId(type, duckdb::LogicalTypeId::ENUM)) {
		return DUCKDB_TYPE_INVALID;
	}
	auto &ltype = *((duckdb::LogicalType *)type);
	switch (ltype.InternalType()) {
	case duckdb::PhysicalType::UINT8:
		return DUCKDB_TYPE_UTINYINT;
	case duckdb::PhysicalType::UINT16:
		return DUCKDB_TYPE_USMALLINT;
	case duckdb::PhysicalType::UINT32:
		return DUCKDB_TYPE_UINTEGER;
	default:
		return DUCKDB_TYPE_INVALID;
	}
}

uint32_t duckdb_enum_dictionary_size(duckdb_logical_type type) {
	if (!AssertLogicalTypeId(type, duckdb::LogicalTypeId::ENUM)) {
		return 0;
	}
	auto &ltype = *((duckdb::LogicalType *)type);
	return duckdb::EnumType::GetSize(ltype);
}

char *duckdb_enum_dictionary_value(duckdb_logical_type type, idx_t index) {
	if (!AssertLogicalTypeId(type, duckdb::LogicalTypeId::ENUM)) {
		return nullptr;
	}
	auto &ltype = *((duckdb::LogicalType *)type);
	auto &vector = duckdb::EnumType::GetValuesInsertOrder(ltype);
	auto value = vector.GetValue(index);
	return strdup(duckdb::StringValue::Get(value).c_str());
}

duckdb_logical_type duckdb_list_type_child_type(duckdb_logical_type type) {
	if (!AssertLogicalTypeId(type, duckdb::LogicalTypeId::LIST) &&
	    !AssertLogicalTypeId(type, duckdb::LogicalTypeId::MAP)) {
		return nullptr;
	}
	auto &ltype = *((duckdb::LogicalType *)type);
	if (ltype.id() != duckdb::LogicalTypeId::LIST && ltype.id() != duckdb::LogicalTypeId::MAP) {
		return nullptr;
	}
	return reinterpret_cast<duckdb_logical_type>(new duckdb::LogicalType(duckdb::ListType::GetChildType(ltype)));
}

duckdb_logical_type duckdb_map_type_key_type(duckdb_logical_type type) {
	if (!AssertLogicalTypeId(type, duckdb::LogicalTypeId::MAP)) {
		return nullptr;
	}
	auto &mtype = *((duckdb::LogicalType *)type);
	if (mtype.id() != duckdb::LogicalTypeId::MAP) {
		return nullptr;
	}
	return reinterpret_cast<duckdb_logical_type>(new duckdb::LogicalType(duckdb::MapType::KeyType(mtype)));
}

duckdb_logical_type duckdb_map_type_value_type(duckdb_logical_type type) {
	if (!AssertLogicalTypeId(type, duckdb::LogicalTypeId::MAP)) {
		return nullptr;
	}
	auto &mtype = *((duckdb::LogicalType *)type);
	if (mtype.id() != duckdb::LogicalTypeId::MAP) {
		return nullptr;
	}
	return reinterpret_cast<duckdb_logical_type>(new duckdb::LogicalType(duckdb::MapType::ValueType(mtype)));
}

idx_t duckdb_struct_type_child_count(duckdb_logical_type type) {
	if (!AssertInternalType(type, duckdb::PhysicalType::STRUCT)) {
		return 0;
	}
	auto &ltype = *((duckdb::LogicalType *)type);
	return duckdb::StructType::GetChildCount(ltype);
}

idx_t duckdb_union_type_member_count(duckdb_logical_type type) {
	if (!AssertLogicalTypeId(type, duckdb::LogicalTypeId::UNION)) {
		return 0;
	}
	idx_t member_count = duckdb_struct_type_child_count(type);
	if (member_count != 0) {
		member_count--;
	}
	return member_count;
}

char *duckdb_union_type_member_name(duckdb_logical_type type, idx_t index) {
	if (!AssertInternalType(type, duckdb::PhysicalType::STRUCT)) {
		return nullptr;
	}
	if (!AssertLogicalTypeId(type, duckdb::LogicalTypeId::UNION)) {
		return nullptr;
	}
	auto &ltype = *((duckdb::LogicalType *)type);
	return strdup(duckdb::UnionType::GetMemberName(ltype, index).c_str());
}

duckdb_logical_type duckdb_union_type_member_type(duckdb_logical_type type, idx_t index) {
	if (!AssertInternalType(type, duckdb::PhysicalType::STRUCT)) {
		return nullptr;
	}
	if (!AssertLogicalTypeId(type, duckdb::LogicalTypeId::UNION)) {
		return nullptr;
	}
	auto &ltype = *((duckdb::LogicalType *)type);
	return reinterpret_cast<duckdb_logical_type>(
	    new duckdb::LogicalType(duckdb::UnionType::GetMemberType(ltype, index)));
}

char *duckdb_struct_type_child_name(duckdb_logical_type type, idx_t index) {
	if (!AssertInternalType(type, duckdb::PhysicalType::STRUCT)) {
		return nullptr;
	}
	auto &ltype = *((duckdb::LogicalType *)type);
	return strdup(duckdb::StructType::GetChildName(ltype, index).c_str());
}

duckdb_logical_type duckdb_struct_type_child_type(duckdb_logical_type type, idx_t index) {
	if (!AssertInternalType(type, duckdb::PhysicalType::STRUCT)) {
		return nullptr;
	}
	auto &ltype = *((duckdb::LogicalType *)type);
	if (ltype.InternalType() != duckdb::PhysicalType::STRUCT) {
		return nullptr;
	}
	return reinterpret_cast<duckdb_logical_type>(
	    new duckdb::LogicalType(duckdb::StructType::GetChildType(ltype, index)));
}





using duckdb::make_uniq;
using duckdb::PendingExecutionResult;
using duckdb::PendingQueryResult;
using duckdb::PendingStatementWrapper;
using duckdb::PreparedStatementWrapper;

duckdb_state duckdb_pending_prepared_internal(duckdb_prepared_statement prepared_statement,
                                              duckdb_pending_result *out_result, bool allow_streaming) {
	if (!prepared_statement || !out_result) {
		return DuckDBError;
	}
	auto wrapper = (PreparedStatementWrapper *)prepared_statement;
	auto result = new PendingStatementWrapper();
	result->allow_streaming = allow_streaming;
	try {
		result->statement = wrapper->statement->PendingQuery(wrapper->values, allow_streaming);
	} catch (const duckdb::Exception &ex) {
		result->statement = make_uniq<PendingQueryResult>(duckdb::PreservedError(ex));
	} catch (std::exception &ex) {
		result->statement = make_uniq<PendingQueryResult>(duckdb::PreservedError(ex));
	}
	duckdb_state return_value = !result->statement->HasError() ? DuckDBSuccess : DuckDBError;
	*out_result = (duckdb_pending_result)result;

	return return_value;
}

duckdb_state duckdb_pending_prepared(duckdb_prepared_statement prepared_statement, duckdb_pending_result *out_result) {
	return duckdb_pending_prepared_internal(prepared_statement, out_result, false);
}

duckdb_state duckdb_pending_prepared_streaming(duckdb_prepared_statement prepared_statement,
                                               duckdb_pending_result *out_result) {
	return duckdb_pending_prepared_internal(prepared_statement, out_result, true);
}

void duckdb_destroy_pending(duckdb_pending_result *pending_result) {
	if (!pending_result || !*pending_result) {
		return;
	}
	auto wrapper = (PendingStatementWrapper *)*pending_result;
	if (wrapper->statement) {
		wrapper->statement->Close();
	}
	delete wrapper;
	*pending_result = nullptr;
}

const char *duckdb_pending_error(duckdb_pending_result pending_result) {
	if (!pending_result) {
		return nullptr;
	}
	auto wrapper = (PendingStatementWrapper *)pending_result;
	if (!wrapper->statement) {
		return nullptr;
	}
	return wrapper->statement->GetError().c_str();
}

duckdb_pending_state duckdb_pending_execute_task(duckdb_pending_result pending_result) {
	if (!pending_result) {
		return DUCKDB_PENDING_ERROR;
	}
	auto wrapper = (PendingStatementWrapper *)pending_result;
	if (!wrapper->statement) {
		return DUCKDB_PENDING_ERROR;
	}
	if (wrapper->statement->HasError()) {
		return DUCKDB_PENDING_ERROR;
	}
	PendingExecutionResult return_value;
	try {
		return_value = wrapper->statement->ExecuteTask();
	} catch (const duckdb::Exception &ex) {
		wrapper->statement->SetError(duckdb::PreservedError(ex));
		return DUCKDB_PENDING_ERROR;
	} catch (std::exception &ex) {
		wrapper->statement->SetError(duckdb::PreservedError(ex));
		return DUCKDB_PENDING_ERROR;
	}
	switch (return_value) {
	case PendingExecutionResult::RESULT_READY:
		return DUCKDB_PENDING_RESULT_READY;
	case PendingExecutionResult::RESULT_NOT_READY:
		return DUCKDB_PENDING_RESULT_NOT_READY;
	default:
		return DUCKDB_PENDING_ERROR;
	}
}

duckdb_state duckdb_execute_pending(duckdb_pending_result pending_result, duckdb_result *out_result) {
	if (!pending_result || !out_result) {
		return DuckDBError;
	}
	auto wrapper = (PendingStatementWrapper *)pending_result;
	if (!wrapper->statement) {
		return DuckDBError;
	}

	duckdb::unique_ptr<duckdb::QueryResult> result;
	result = wrapper->statement->Execute();
	wrapper->statement.reset();
	return duckdb_translate_result(std::move(result), out_result);
}





using duckdb::Connection;
using duckdb::date_t;
using duckdb::dtime_t;
using duckdb::ExtractStatementsWrapper;
using duckdb::hugeint_t;
using duckdb::LogicalType;
using duckdb::MaterializedQueryResult;
using duckdb::PreparedStatementWrapper;
using duckdb::QueryResultType;
using duckdb::timestamp_t;
using duckdb::Value;

idx_t duckdb_extract_statements(duckdb_connection connection, const char *query,
                                duckdb_extracted_statements *out_extracted_statements) {
	if (!connection || !query || !out_extracted_statements) {
		return 0;
	}
	auto wrapper = new ExtractStatementsWrapper();
	Connection *conn = (Connection *)connection;
	try {
		wrapper->statements = conn->ExtractStatements(query);
	} catch (const duckdb::ParserException &e) {
		wrapper->error = e.what();
	}

	*out_extracted_statements = (duckdb_extracted_statements)wrapper;
	return wrapper->statements.size();
}

duckdb_state duckdb_prepare_extracted_statement(duckdb_connection connection,
                                                duckdb_extracted_statements extracted_statements, idx_t index,
                                                duckdb_prepared_statement *out_prepared_statement) {
	Connection *conn = (Connection *)connection;
	auto source_wrapper = (ExtractStatementsWrapper *)extracted_statements;

	if (!connection || !out_prepared_statement || index >= source_wrapper->statements.size()) {
		return DuckDBError;
	}
	auto wrapper = new PreparedStatementWrapper();
	wrapper->statement = conn->Prepare(std::move(source_wrapper->statements[index]));

	*out_prepared_statement = (duckdb_prepared_statement)wrapper;
	return wrapper->statement->HasError() ? DuckDBError : DuckDBSuccess;
}

const char *duckdb_extract_statements_error(duckdb_extracted_statements extracted_statements) {
	auto wrapper = (ExtractStatementsWrapper *)extracted_statements;
	if (!wrapper || wrapper->error.empty()) {
		return nullptr;
	}
	return wrapper->error.c_str();
}

duckdb_state duckdb_prepare(duckdb_connection connection, const char *query,
                            duckdb_prepared_statement *out_prepared_statement) {
	if (!connection || !query || !out_prepared_statement) {
		return DuckDBError;
	}
	auto wrapper = new PreparedStatementWrapper();
	Connection *conn = (Connection *)connection;
	wrapper->statement = conn->Prepare(query);
	*out_prepared_statement = (duckdb_prepared_statement)wrapper;
	return !wrapper->statement->HasError() ? DuckDBSuccess : DuckDBError;
}

const char *duckdb_prepare_error(duckdb_prepared_statement prepared_statement) {
	auto wrapper = (PreparedStatementWrapper *)prepared_statement;
	if (!wrapper || !wrapper->statement || !wrapper->statement->HasError()) {
		return nullptr;
	}
	return wrapper->statement->error.Message().c_str();
}

idx_t duckdb_nparams(duckdb_prepared_statement prepared_statement) {
	auto wrapper = (PreparedStatementWrapper *)prepared_statement;
	if (!wrapper || !wrapper->statement || wrapper->statement->HasError()) {
		return 0;
	}
	return wrapper->statement->n_param;
}

duckdb_type duckdb_param_type(duckdb_prepared_statement prepared_statement, idx_t param_idx) {
	auto wrapper = (PreparedStatementWrapper *)prepared_statement;
	if (!wrapper || !wrapper->statement || wrapper->statement->HasError()) {
		return DUCKDB_TYPE_INVALID;
	}
	LogicalType param_type;
	if (!wrapper->statement->data->TryGetType(param_idx, param_type)) {
		return DUCKDB_TYPE_INVALID;
	}
	return ConvertCPPTypeToC(param_type);
}

duckdb_state duckdb_clear_bindings(duckdb_prepared_statement prepared_statement) {
	auto wrapper = (PreparedStatementWrapper *)prepared_statement;
	if (!wrapper || !wrapper->statement || wrapper->statement->HasError()) {
		return DuckDBError;
	}
	wrapper->values.clear();
	return DuckDBSuccess;
}

static duckdb_state duckdb_bind_value(duckdb_prepared_statement prepared_statement, idx_t param_idx, Value val) {
	auto wrapper = (PreparedStatementWrapper *)prepared_statement;
	if (!wrapper || !wrapper->statement || wrapper->statement->HasError()) {
		return DuckDBError;
	}
	if (param_idx <= 0 || param_idx > wrapper->statement->n_param) {
		return DuckDBError;
	}
	if (param_idx > wrapper->values.size()) {
		wrapper->values.resize(param_idx);
	}
	wrapper->values[param_idx - 1] = val;
	return DuckDBSuccess;
}

duckdb_state duckdb_bind_boolean(duckdb_prepared_statement prepared_statement, idx_t param_idx, bool val) {
	return duckdb_bind_value(prepared_statement, param_idx, Value::BOOLEAN(val));
}

duckdb_state duckdb_bind_int8(duckdb_prepared_statement prepared_statement, idx_t param_idx, int8_t val) {
	return duckdb_bind_value(prepared_statement, param_idx, Value::TINYINT(val));
}

duckdb_state duckdb_bind_int16(duckdb_prepared_statement prepared_statement, idx_t param_idx, int16_t val) {
	return duckdb_bind_value(prepared_statement, param_idx, Value::SMALLINT(val));
}

duckdb_state duckdb_bind_int32(duckdb_prepared_statement prepared_statement, idx_t param_idx, int32_t val) {
	return duckdb_bind_value(prepared_statement, param_idx, Value::INTEGER(val));
}

duckdb_state duckdb_bind_int64(duckdb_prepared_statement prepared_statement, idx_t param_idx, int64_t val) {
	return duckdb_bind_value(prepared_statement, param_idx, Value::BIGINT(val));
}

static hugeint_t duckdb_internal_hugeint(duckdb_hugeint val) {
	hugeint_t internal;
	internal.lower = val.lower;
	internal.upper = val.upper;
	return internal;
}

duckdb_state duckdb_bind_hugeint(duckdb_prepared_statement prepared_statement, idx_t param_idx, duckdb_hugeint val) {
	return duckdb_bind_value(prepared_statement, param_idx, Value::HUGEINT(duckdb_internal_hugeint(val)));
}

duckdb_state duckdb_bind_uint8(duckdb_prepared_statement prepared_statement, idx_t param_idx, uint8_t val) {
	return duckdb_bind_value(prepared_statement, param_idx, Value::UTINYINT(val));
}

duckdb_state duckdb_bind_uint16(duckdb_prepared_statement prepared_statement, idx_t param_idx, uint16_t val) {
	return duckdb_bind_value(prepared_statement, param_idx, Value::USMALLINT(val));
}

duckdb_state duckdb_bind_uint32(duckdb_prepared_statement prepared_statement, idx_t param_idx, uint32_t val) {
	return duckdb_bind_value(prepared_statement, param_idx, Value::UINTEGER(val));
}

duckdb_state duckdb_bind_uint64(duckdb_prepared_statement prepared_statement, idx_t param_idx, uint64_t val) {
	return duckdb_bind_value(prepared_statement, param_idx, Value::UBIGINT(val));
}

duckdb_state duckdb_bind_float(duckdb_prepared_statement prepared_statement, idx_t param_idx, float val) {
	return duckdb_bind_value(prepared_statement, param_idx, Value::FLOAT(val));
}

duckdb_state duckdb_bind_double(duckdb_prepared_statement prepared_statement, idx_t param_idx, double val) {
	return duckdb_bind_value(prepared_statement, param_idx, Value::DOUBLE(val));
}

duckdb_state duckdb_bind_date(duckdb_prepared_statement prepared_statement, idx_t param_idx, duckdb_date val) {
	return duckdb_bind_value(prepared_statement, param_idx, Value::DATE(date_t(val.days)));
}

duckdb_state duckdb_bind_time(duckdb_prepared_statement prepared_statement, idx_t param_idx, duckdb_time val) {
	return duckdb_bind_value(prepared_statement, param_idx, Value::TIME(dtime_t(val.micros)));
}

duckdb_state duckdb_bind_timestamp(duckdb_prepared_statement prepared_statement, idx_t param_idx,
                                   duckdb_timestamp val) {
	return duckdb_bind_value(prepared_statement, param_idx, Value::TIMESTAMP(timestamp_t(val.micros)));
}

duckdb_state duckdb_bind_interval(duckdb_prepared_statement prepared_statement, idx_t param_idx, duckdb_interval val) {
	return duckdb_bind_value(prepared_statement, param_idx, Value::INTERVAL(val.months, val.days, val.micros));
}

duckdb_state duckdb_bind_varchar(duckdb_prepared_statement prepared_statement, idx_t param_idx, const char *val) {
	try {
		return duckdb_bind_value(prepared_statement, param_idx, Value(val));
	} catch (...) {
		return DuckDBError;
	}
}

duckdb_state duckdb_bind_varchar_length(duckdb_prepared_statement prepared_statement, idx_t param_idx, const char *val,
                                        idx_t length) {
	try {
		return duckdb_bind_value(prepared_statement, param_idx, Value(std::string(val, length)));
	} catch (...) {
		return DuckDBError;
	}
}

duckdb_state duckdb_bind_decimal(duckdb_prepared_statement prepared_statement, idx_t param_idx, duckdb_decimal val) {
	auto hugeint_val = duckdb_internal_hugeint(val.value);
	if (val.width > duckdb::Decimal::MAX_WIDTH_INT64) {
		return duckdb_bind_value(prepared_statement, param_idx, Value::DECIMAL(hugeint_val, val.width, val.scale));
	}
	auto value = hugeint_val.lower;
	return duckdb_bind_value(prepared_statement, param_idx, Value::DECIMAL((int64_t)value, val.width, val.scale));
}

duckdb_state duckdb_bind_blob(duckdb_prepared_statement prepared_statement, idx_t param_idx, const void *data,
                              idx_t length) {
	return duckdb_bind_value(prepared_statement, param_idx, Value::BLOB((duckdb::const_data_ptr_t)data, length));
}

duckdb_state duckdb_bind_null(duckdb_prepared_statement prepared_statement, idx_t param_idx) {
	return duckdb_bind_value(prepared_statement, param_idx, Value());
}

duckdb_state duckdb_execute_prepared(duckdb_prepared_statement prepared_statement, duckdb_result *out_result) {
	auto wrapper = (PreparedStatementWrapper *)prepared_statement;
	if (!wrapper || !wrapper->statement || wrapper->statement->HasError()) {
		return DuckDBError;
	}
	auto result = wrapper->statement->Execute(wrapper->values, false);
	return duckdb_translate_result(std::move(result), out_result);
}

template <class T>
void duckdb_destroy(void **wrapper) {
	if (!wrapper) {
		return;
	}

	auto casted = (T *)*wrapper;
	if (casted) {
		delete casted;
	}
	*wrapper = nullptr;
}

void duckdb_destroy_extracted(duckdb_extracted_statements *extracted_statements) {
	duckdb_destroy<ExtractStatementsWrapper>(reinterpret_cast<void **>(extracted_statements));
}

void duckdb_destroy_prepare(duckdb_prepared_statement *prepared_statement) {
	duckdb_destroy<PreparedStatementWrapper>(reinterpret_cast<void **>(prepared_statement));
}






namespace duckdb {

struct CAPIReplacementScanData : public ReplacementScanData {
	~CAPIReplacementScanData() {
		if (delete_callback) {
			delete_callback(extra_data);
		}
	}

	duckdb_replacement_callback_t callback;
	void *extra_data;
	duckdb_delete_callback_t delete_callback;
};

struct CAPIReplacementScanInfo {
	CAPIReplacementScanInfo(CAPIReplacementScanData *data) : data(data) {
	}

	CAPIReplacementScanData *data;
	string function_name;
	vector<Value> parameters;
	string error;
};

unique_ptr<TableRef> duckdb_capi_replacement_callback(ClientContext &context, const string &table_name,
                                                      ReplacementScanData *data) {
	auto &scan_data = (CAPIReplacementScanData &)*data;

	CAPIReplacementScanInfo info(&scan_data);
	scan_data.callback((duckdb_replacement_scan_info)&info, table_name.c_str(), scan_data.extra_data);
	if (!info.error.empty()) {
		throw BinderException("Error in replacement scan: %s\n", info.error);
	}
	if (info.function_name.empty()) {
		// no function provided: bail-out
		return nullptr;
	}
	auto table_function = make_uniq<TableFunctionRef>();
	vector<unique_ptr<ParsedExpression>> children;
	for (auto &param : info.parameters) {
		children.push_back(make_uniq<ConstantExpression>(std::move(param)));
	}
	table_function->function = make_uniq<FunctionExpression>(info.function_name, std::move(children));
	return std::move(table_function);
}

} // namespace duckdb

void duckdb_add_replacement_scan(duckdb_database db, duckdb_replacement_callback_t replacement, void *extra_data,
                                 duckdb_delete_callback_t delete_callback) {
	if (!db || !replacement) {
		return;
	}
	auto wrapper = (duckdb::DatabaseData *)db;
	auto scan_info = duckdb::make_uniq<duckdb::CAPIReplacementScanData>();
	scan_info->callback = replacement;
	scan_info->extra_data = extra_data;
	scan_info->delete_callback = delete_callback;

	auto &config = duckdb::DBConfig::GetConfig(*wrapper->database->instance);
	config.replacement_scans.push_back(
	    duckdb::ReplacementScan(duckdb::duckdb_capi_replacement_callback, std::move(scan_info)));
}

void duckdb_replacement_scan_set_function_name(duckdb_replacement_scan_info info_p, const char *function_name) {
	if (!info_p || !function_name) {
		return;
	}
	auto info = (duckdb::CAPIReplacementScanInfo *)info_p;
	info->function_name = function_name;
}

void duckdb_replacement_scan_add_parameter(duckdb_replacement_scan_info info_p, duckdb_value parameter) {
	if (!info_p || !parameter) {
		return;
	}
	auto info = (duckdb::CAPIReplacementScanInfo *)info_p;
	auto val = (duckdb::Value *)parameter;
	info->parameters.push_back(*val);
}

void duckdb_replacement_scan_set_error(duckdb_replacement_scan_info info_p, const char *error) {
	if (!info_p || !error) {
		return;
	}
	auto info = (duckdb::CAPIReplacementScanInfo *)info_p;
	info->error = error;
}




namespace duckdb {

struct CBaseConverter {
	template <class DST>
	static void NullConvert(DST &target) {
	}
};
struct CStandardConverter : public CBaseConverter {
	template <class SRC, class DST>
	static DST Convert(SRC input) {
		return input;
	}
};

struct CStringConverter {
	template <class SRC, class DST>
	static DST Convert(SRC input) {
		auto result = (char *)duckdb_malloc(input.GetSize() + 1);
		assert(result);
		memcpy((void *)result, input.GetData(), input.GetSize());
		auto write_arr = (char *)result;
		write_arr[input.GetSize()] = '\0';
		return result;
	}

	template <class DST>
	static void NullConvert(DST &target) {
		target = nullptr;
	}
};

struct CBlobConverter {
	template <class SRC, class DST>
	static DST Convert(SRC input) {
		duckdb_blob result;
		result.data = (char *)duckdb_malloc(input.GetSize());
		result.size = input.GetSize();
		assert(result.data);
		memcpy((void *)result.data, input.GetData(), input.GetSize());
		return result;
	}

	template <class DST>
	static void NullConvert(DST &target) {
		target.data = nullptr;
		target.size = 0;
	}
};

struct CTimestampMsConverter : public CBaseConverter {
	template <class SRC, class DST>
	static DST Convert(SRC input) {
		return Timestamp::FromEpochMs(input.value);
	}
};

struct CTimestampNsConverter : public CBaseConverter {
	template <class SRC, class DST>
	static DST Convert(SRC input) {
		return Timestamp::FromEpochNanoSeconds(input.value);
	}
};

struct CTimestampSecConverter : public CBaseConverter {
	template <class SRC, class DST>
	static DST Convert(SRC input) {
		return Timestamp::FromEpochSeconds(input.value);
	}
};

struct CHugeintConverter : public CBaseConverter {
	template <class SRC, class DST>
	static DST Convert(SRC input) {
		duckdb_hugeint result;
		result.lower = input.lower;
		result.upper = input.upper;
		return result;
	}
};

struct CIntervalConverter : public CBaseConverter {
	template <class SRC, class DST>
	static DST Convert(SRC input) {
		duckdb_interval result;
		result.days = input.days;
		result.months = input.months;
		result.micros = input.micros;
		return result;
	}
};

template <class T>
struct CDecimalConverter : public CBaseConverter {
	template <class SRC, class DST>
	static DST Convert(SRC input) {
		duckdb_hugeint result;
		result.lower = input;
		result.upper = 0;
		return result;
	}
};

template <class SRC, class DST = SRC, class OP = CStandardConverter>
void WriteData(duckdb_column *column, ColumnDataCollection &source, const vector<column_t> &column_ids) {
	idx_t row = 0;
	auto target = (DST *)column->__deprecated_data;
	for (auto &input : source.Chunks(column_ids)) {
		auto source = FlatVector::GetData<SRC>(input.data[0]);
		auto &mask = FlatVector::Validity(input.data[0]);

		for (idx_t k = 0; k < input.size(); k++, row++) {
			if (!mask.RowIsValid(k)) {
				OP::template NullConvert<DST>(target[row]);
			} else {
				target[row] = OP::template Convert<SRC, DST>(source[k]);
			}
		}
	}
}

duckdb_state deprecated_duckdb_translate_column(MaterializedQueryResult &result, duckdb_column *column, idx_t col) {
	D_ASSERT(!result.HasError());
	auto &collection = result.Collection();
	idx_t row_count = collection.Count();
	column->__deprecated_nullmask = (bool *)duckdb_malloc(sizeof(bool) * collection.Count());
	column->__deprecated_data = duckdb_malloc(GetCTypeSize(column->__deprecated_type) * row_count);
	if (!column->__deprecated_nullmask || !column->__deprecated_data) { // LCOV_EXCL_START
		// malloc failure
		return DuckDBError;
	} // LCOV_EXCL_STOP

	vector<column_t> column_ids {col};
	// first convert the nullmask
	{
		idx_t row = 0;
		for (auto &input : collection.Chunks(column_ids)) {
			for (idx_t k = 0; k < input.size(); k++) {
				column->__deprecated_nullmask[row++] = FlatVector::IsNull(input.data[0], k);
			}
		}
	}
	// then write the data
	switch (result.types[col].id()) {
	case LogicalTypeId::BOOLEAN:
		WriteData<bool>(column, collection, column_ids);
		break;
	case LogicalTypeId::TINYINT:
		WriteData<int8_t>(column, collection, column_ids);
		break;
	case LogicalTypeId::SMALLINT:
		WriteData<int16_t>(column, collection, column_ids);
		break;
	case LogicalTypeId::INTEGER:
		WriteData<int32_t>(column, collection, column_ids);
		break;
	case LogicalTypeId::BIGINT:
		WriteData<int64_t>(column, collection, column_ids);
		break;
	case LogicalTypeId::UTINYINT:
		WriteData<uint8_t>(column, collection, column_ids);
		break;
	case LogicalTypeId::USMALLINT:
		WriteData<uint16_t>(column, collection, column_ids);
		break;
	case LogicalTypeId::UINTEGER:
		WriteData<uint32_t>(column, collection, column_ids);
		break;
	case LogicalTypeId::UBIGINT:
		WriteData<uint64_t>(column, collection, column_ids);
		break;
	case LogicalTypeId::FLOAT:
		WriteData<float>(column, collection, column_ids);
		break;
	case LogicalTypeId::DOUBLE:
		WriteData<double>(column, collection, column_ids);
		break;
	case LogicalTypeId::DATE:
		WriteData<date_t>(column, collection, column_ids);
		break;
	case LogicalTypeId::TIME:
	case LogicalTypeId::TIME_TZ:
		WriteData<dtime_t>(column, collection, column_ids);
		break;
	case LogicalTypeId::TIMESTAMP:
	case LogicalTypeId::TIMESTAMP_TZ:
		WriteData<timestamp_t>(column, collection, column_ids);
		break;
	case LogicalTypeId::VARCHAR: {
		WriteData<string_t, const char *, CStringConverter>(column, collection, column_ids);
		break;
	}
	case LogicalTypeId::BLOB: {
		WriteData<string_t, duckdb_blob, CBlobConverter>(column, collection, column_ids);
		break;
	}
	case LogicalTypeId::TIMESTAMP_NS: {
		WriteData<timestamp_t, timestamp_t, CTimestampNsConverter>(column, collection, column_ids);
		break;
	}
	case LogicalTypeId::TIMESTAMP_MS: {
		WriteData<timestamp_t, timestamp_t, CTimestampMsConverter>(column, collection, column_ids);
		break;
	}
	case LogicalTypeId::TIMESTAMP_SEC: {
		WriteData<timestamp_t, timestamp_t, CTimestampSecConverter>(column, collection, column_ids);
		break;
	}
	case LogicalTypeId::HUGEINT: {
		WriteData<hugeint_t, duckdb_hugeint, CHugeintConverter>(column, collection, column_ids);
		break;
	}
	case LogicalTypeId::INTERVAL: {
		WriteData<interval_t, duckdb_interval, CIntervalConverter>(column, collection, column_ids);
		break;
	}
	case LogicalTypeId::DECIMAL: {
		// get data
		switch (result.types[col].InternalType()) {
		case PhysicalType::INT16: {
			WriteData<int16_t, duckdb_hugeint, CDecimalConverter<int16_t>>(column, collection, column_ids);
			break;
		}
		case PhysicalType::INT32: {
			WriteData<int32_t, duckdb_hugeint, CDecimalConverter<int32_t>>(column, collection, column_ids);
			break;
		}
		case PhysicalType::INT64: {
			WriteData<int64_t, duckdb_hugeint, CDecimalConverter<int64_t>>(column, collection, column_ids);
			break;
		}
		case PhysicalType::INT128: {
			WriteData<hugeint_t, duckdb_hugeint, CHugeintConverter>(column, collection, column_ids);
			break;
		}
		default:
			throw std::runtime_error("Unsupported physical type for Decimal" +
			                         TypeIdToString(result.types[col].InternalType()));
		}
		break;
	}
	default: // LCOV_EXCL_START
		return DuckDBError;
	} // LCOV_EXCL_STOP
	return DuckDBSuccess;
}

duckdb_state duckdb_translate_result(unique_ptr<QueryResult> result_p, duckdb_result *out) {
	auto &result = *result_p;
	D_ASSERT(result_p);
	if (!out) {
		// no result to write to, only return the status
		return !result.HasError() ? DuckDBSuccess : DuckDBError;
	}

	memset(out, 0, sizeof(duckdb_result));

	// initialize the result_data object
	auto result_data = new DuckDBResultData();
	result_data->result = std::move(result_p);
	result_data->result_set_type = CAPIResultSetType::CAPI_RESULT_TYPE_NONE;
	out->internal_data = result_data;

	if (result.HasError()) {
		// write the error message
		out->__deprecated_error_message = (char *)result.GetError().c_str();
		return DuckDBError;
	}
	// copy the data
	// first write the meta data
	out->__deprecated_column_count = result.ColumnCount();
	out->__deprecated_rows_changed = 0;
	return DuckDBSuccess;
}

bool deprecated_materialize_result(duckdb_result *result) {
	if (!result) {
		return false;
	}
	auto result_data = (duckdb::DuckDBResultData *)result->internal_data;
	if (result_data->result->HasError()) {
		return false;
	}
	if (result_data->result_set_type == CAPIResultSetType::CAPI_RESULT_TYPE_DEPRECATED) {
		// already materialized into deprecated result format
		return true;
	}
	if (result_data->result_set_type == CAPIResultSetType::CAPI_RESULT_TYPE_MATERIALIZED) {
		// already used as a new result set
		return false;
	}
	if (result_data->result_set_type == CAPIResultSetType::CAPI_RESULT_TYPE_STREAMING) {
		// already used as a streaming result
		return false;
	}
	// materialize as deprecated result set
	result_data->result_set_type = CAPIResultSetType::CAPI_RESULT_TYPE_DEPRECATED;
	auto column_count = result_data->result->ColumnCount();
	result->__deprecated_columns = (duckdb_column *)duckdb_malloc(sizeof(duckdb_column) * column_count);
	if (!result->__deprecated_columns) { // LCOV_EXCL_START
		// malloc failure
		return DuckDBError;
	} // LCOV_EXCL_STOP
	if (result_data->result->type == QueryResultType::STREAM_RESULT) {
		// if we are dealing with a stream result, convert it to a materialized result first
		auto &stream_result = (StreamQueryResult &)*result_data->result;
		result_data->result = stream_result.Materialize();
	}
	D_ASSERT(result_data->result->type == QueryResultType::MATERIALIZED_RESULT);
	auto &materialized = (MaterializedQueryResult &)*result_data->result;

	// convert the result to a materialized result
	// zero initialize the columns (so we can cleanly delete it in case a malloc fails)
	memset(result->__deprecated_columns, 0, sizeof(duckdb_column) * column_count);
	for (idx_t i = 0; i < column_count; i++) {
		result->__deprecated_columns[i].__deprecated_type = ConvertCPPTypeToC(result_data->result->types[i]);
		result->__deprecated_columns[i].__deprecated_name = (char *)result_data->result->names[i].c_str();
	}
	result->__deprecated_row_count = materialized.RowCount();
	if (result->__deprecated_row_count > 0 &&
	    materialized.properties.return_type == StatementReturnType::CHANGED_ROWS) {
		// update total changes
		auto row_changes = materialized.GetValue(0, 0);
		if (!row_changes.IsNull() && row_changes.DefaultTryCastAs(LogicalType::BIGINT)) {
			result->__deprecated_rows_changed = row_changes.GetValue<int64_t>();
		}
	}
	// now write the data
	for (idx_t col = 0; col < column_count; col++) {
		auto state = deprecated_duckdb_translate_column(materialized, &result->__deprecated_columns[col], col);
		if (state != DuckDBSuccess) {
			return false;
		}
	}
	return true;
}

} // namespace duckdb

static void DuckdbDestroyColumn(duckdb_column column, idx_t count) {
	if (column.__deprecated_data) {
		if (column.__deprecated_type == DUCKDB_TYPE_VARCHAR) {
			// varchar, delete individual strings
			auto data = (char **)column.__deprecated_data;
			for (idx_t i = 0; i < count; i++) {
				if (data[i]) {
					duckdb_free(data[i]);
				}
			}
		} else if (column.__deprecated_type == DUCKDB_TYPE_BLOB) {
			// blob, delete individual blobs
			auto data = (duckdb_blob *)column.__deprecated_data;
			for (idx_t i = 0; i < count; i++) {
				if (data[i].data) {
					duckdb_free((void *)data[i].data);
				}
			}
		}
		duckdb_free(column.__deprecated_data);
	}
	if (column.__deprecated_nullmask) {
		duckdb_free(column.__deprecated_nullmask);
	}
}

void duckdb_destroy_result(duckdb_result *result) {
	if (result->__deprecated_columns) {
		for (idx_t i = 0; i < result->__deprecated_column_count; i++) {
			DuckdbDestroyColumn(result->__deprecated_columns[i], result->__deprecated_row_count);
		}
		duckdb_free(result->__deprecated_columns);
	}
	if (result->internal_data) {
		auto result_data = (duckdb::DuckDBResultData *)result->internal_data;
		delete result_data;
	}
	memset(result, 0, sizeof(duckdb_result));
}

const char *duckdb_column_name(duckdb_result *result, idx_t col) {
	if (!result || col >= duckdb_column_count(result)) {
		return nullptr;
	}
	auto &result_data = *((duckdb::DuckDBResultData *)result->internal_data);
	return result_data.result->names[col].c_str();
}

duckdb_type duckdb_column_type(duckdb_result *result, idx_t col) {
	if (!result || col >= duckdb_column_count(result)) {
		return DUCKDB_TYPE_INVALID;
	}
	auto &result_data = *((duckdb::DuckDBResultData *)result->internal_data);
	return duckdb::ConvertCPPTypeToC(result_data.result->types[col]);
}

duckdb_logical_type duckdb_column_logical_type(duckdb_result *result, idx_t col) {
	if (!result || col >= duckdb_column_count(result)) {
		return nullptr;
	}
	auto &result_data = *((duckdb::DuckDBResultData *)result->internal_data);
	return reinterpret_cast<duckdb_logical_type>(new duckdb::LogicalType(result_data.result->types[col]));
}

idx_t duckdb_column_count(duckdb_result *result) {
	if (!result) {
		return 0;
	}
	auto &result_data = *((duckdb::DuckDBResultData *)result->internal_data);
	return result_data.result->ColumnCount();
}

idx_t duckdb_row_count(duckdb_result *result) {
	if (!result) {
		return 0;
	}
	auto &result_data = *((duckdb::DuckDBResultData *)result->internal_data);
	if (result_data.result->type == duckdb::QueryResultType::STREAM_RESULT) {
		// We can't know the row count beforehand
		return 0;
	}
	auto &materialized = (duckdb::MaterializedQueryResult &)*result_data.result;
	return materialized.RowCount();
}

idx_t duckdb_rows_changed(duckdb_result *result) {
	if (!result) {
		return 0;
	}
	if (!duckdb::deprecated_materialize_result(result)) {
		return 0;
	}
	return result->__deprecated_rows_changed;
}

void *duckdb_column_data(duckdb_result *result, idx_t col) {
	if (!result || col >= result->__deprecated_column_count) {
		return nullptr;
	}
	if (!duckdb::deprecated_materialize_result(result)) {
		return nullptr;
	}
	return result->__deprecated_columns[col].__deprecated_data;
}

bool *duckdb_nullmask_data(duckdb_result *result, idx_t col) {
	if (!result || col >= result->__deprecated_column_count) {
		return nullptr;
	}
	if (!duckdb::deprecated_materialize_result(result)) {
		return nullptr;
	}
	return result->__deprecated_columns[col].__deprecated_nullmask;
}

const char *duckdb_result_error(duckdb_result *result) {
	if (!result) {
		return nullptr;
	}
	auto &result_data = *((duckdb::DuckDBResultData *)result->internal_data);
	return !result_data.result->HasError() ? nullptr : result_data.result->GetError().c_str();
}

idx_t duckdb_result_chunk_count(duckdb_result result) {
	if (!result.internal_data) {
		return 0;
	}
	auto &result_data = *((duckdb::DuckDBResultData *)result.internal_data);
	if (result_data.result_set_type == duckdb::CAPIResultSetType::CAPI_RESULT_TYPE_DEPRECATED) {
		return 0;
	}
	if (result_data.result->type != duckdb::QueryResultType::MATERIALIZED_RESULT) {
		// Can't know beforehand how many chunks are returned.
		return 0;
	}
	auto &materialized = (duckdb::MaterializedQueryResult &)*result_data.result;
	return materialized.Collection().ChunkCount();
}

duckdb_data_chunk duckdb_result_get_chunk(duckdb_result result, idx_t chunk_idx) {
	if (!result.internal_data) {
		return nullptr;
	}
	auto &result_data = *((duckdb::DuckDBResultData *)result.internal_data);
	if (result_data.result_set_type == duckdb::CAPIResultSetType::CAPI_RESULT_TYPE_DEPRECATED) {
		return nullptr;
	}
	if (result_data.result->type != duckdb::QueryResultType::MATERIALIZED_RESULT) {
		// This API is only supported for materialized query results
		return nullptr;
	}
	result_data.result_set_type = duckdb::CAPIResultSetType::CAPI_RESULT_TYPE_MATERIALIZED;
	auto &materialized = (duckdb::MaterializedQueryResult &)*result_data.result;
	auto &collection = materialized.Collection();
	if (chunk_idx >= collection.ChunkCount()) {
		return nullptr;
	}
	auto chunk = duckdb::make_uniq<duckdb::DataChunk>();
	chunk->Initialize(duckdb::Allocator::DefaultAllocator(), collection.Types());
	collection.FetchChunk(chunk_idx, *chunk);
	return reinterpret_cast<duckdb_data_chunk>(chunk.release());
}

bool duckdb_result_is_streaming(duckdb_result result) {
	if (!result.internal_data) {
		return false;
	}
	if (duckdb_result_error(&result) != nullptr) {
		return false;
	}
	auto &result_data = *((duckdb::DuckDBResultData *)result.internal_data);
	return result_data.result->type == duckdb::QueryResultType::STREAM_RESULT;
}




duckdb_data_chunk duckdb_stream_fetch_chunk(duckdb_result result) {
	if (!result.internal_data) {
		return nullptr;
	}
	auto &result_data = *((duckdb::DuckDBResultData *)result.internal_data);
	if (result_data.result_set_type == duckdb::CAPIResultSetType::CAPI_RESULT_TYPE_DEPRECATED) {
		return nullptr;
	}
	if (result_data.result->type != duckdb::QueryResultType::STREAM_RESULT) {
		// We can only fetch from a StreamQueryResult
		return nullptr;
	}
	result_data.result_set_type = duckdb::CAPIResultSetType::CAPI_RESULT_TYPE_STREAMING;
	auto &streaming = (duckdb::StreamQueryResult &)*result_data.result;
	if (!streaming.IsOpen()) {
		return nullptr;
	}
	// FetchRaw ? Do we care about flattening them?
	auto chunk = streaming.Fetch();
	return reinterpret_cast<duckdb_data_chunk>(chunk.release());
}







namespace duckdb {

struct CTableFunctionInfo : public TableFunctionInfo {
	~CTableFunctionInfo() {
		if (extra_info && delete_callback) {
			delete_callback(extra_info);
		}
		extra_info = nullptr;
		delete_callback = nullptr;
	}

	duckdb_table_function_bind_t bind = nullptr;
	duckdb_table_function_init_t init = nullptr;
	duckdb_table_function_init_t local_init = nullptr;
	duckdb_table_function_t function = nullptr;
	void *extra_info = nullptr;
	duckdb_delete_callback_t delete_callback = nullptr;
};

struct CTableBindData : public TableFunctionData {
	CTableBindData(CTableFunctionInfo &info) : info(info) {
	}
	~CTableBindData() {
		if (bind_data && delete_callback) {
			delete_callback(bind_data);
		}
		bind_data = nullptr;
		delete_callback = nullptr;
	}

	CTableFunctionInfo &info;
	void *bind_data = nullptr;
	duckdb_delete_callback_t delete_callback = nullptr;
	unique_ptr<NodeStatistics> stats;
};

struct CTableInternalBindInfo {
	CTableInternalBindInfo(ClientContext &context, TableFunctionBindInput &input, vector<LogicalType> &return_types,
	                       vector<string> &names, CTableBindData &bind_data, CTableFunctionInfo &function_info)
	    : context(context), input(input), return_types(return_types), names(names), bind_data(bind_data),
	      function_info(function_info), success(true) {
	}

	ClientContext &context;
	TableFunctionBindInput &input;
	vector<LogicalType> &return_types;
	vector<string> &names;
	CTableBindData &bind_data;
	CTableFunctionInfo &function_info;
	bool success;
	string error;
};

struct CTableInitData {
	~CTableInitData() {
		if (init_data && delete_callback) {
			delete_callback(init_data);
		}
		init_data = nullptr;
		delete_callback = nullptr;
	}

	void *init_data = nullptr;
	duckdb_delete_callback_t delete_callback = nullptr;
	idx_t max_threads = 1;
};

struct CTableGlobalInitData : public GlobalTableFunctionState {
	CTableInitData init_data;

	idx_t MaxThreads() const override {
		return init_data.max_threads;
	}
};

struct CTableLocalInitData : public LocalTableFunctionState {
	CTableInitData init_data;
};

struct CTableInternalInitInfo {
	CTableInternalInitInfo(const CTableBindData &bind_data, CTableInitData &init_data,
	                       const vector<column_t> &column_ids, optional_ptr<TableFilterSet> filters)
	    : bind_data(bind_data), init_data(init_data), column_ids(column_ids), filters(filters), success(true) {
	}

	const CTableBindData &bind_data;
	CTableInitData &init_data;
	const vector<column_t> &column_ids;
	optional_ptr<TableFilterSet> filters;
	bool success;
	string error;
};

struct CTableInternalFunctionInfo {
	CTableInternalFunctionInfo(const CTableBindData &bind_data, CTableInitData &init_data, CTableInitData &local_data)
	    : bind_data(bind_data), init_data(init_data), local_data(local_data), success(true) {
	}

	const CTableBindData &bind_data;
	CTableInitData &init_data;
	CTableInitData &local_data;
	bool success;
	string error;
};

unique_ptr<FunctionData> CTableFunctionBind(ClientContext &context, TableFunctionBindInput &input,
                                            vector<LogicalType> &return_types, vector<string> &names) {
	auto &info = input.info->Cast<CTableFunctionInfo>();
	D_ASSERT(info.bind && info.function && info.init);
	auto result = make_uniq<CTableBindData>(info);
	CTableInternalBindInfo bind_info(context, input, return_types, names, *result, info);
	info.bind(&bind_info);
	if (!bind_info.success) {
		throw Exception(bind_info.error);
	}

	return std::move(result);
}

unique_ptr<GlobalTableFunctionState> CTableFunctionInit(ClientContext &context, TableFunctionInitInput &data_p) {
	auto &bind_data = data_p.bind_data->Cast<CTableBindData>();
	auto result = make_uniq<CTableGlobalInitData>();

	CTableInternalInitInfo init_info(bind_data, result->init_data, data_p.column_ids, data_p.filters);
	bind_data.info.init(&init_info);
	if (!init_info.success) {
		throw Exception(init_info.error);
	}
	return std::move(result);
}

unique_ptr<LocalTableFunctionState> CTableFunctionLocalInit(ExecutionContext &context, TableFunctionInitInput &data_p,
                                                            GlobalTableFunctionState *gstate) {
	auto &bind_data = data_p.bind_data->Cast<CTableBindData>();
	auto result = make_uniq<CTableLocalInitData>();
	if (!bind_data.info.local_init) {
		return std::move(result);
	}

	CTableInternalInitInfo init_info(bind_data, result->init_data, data_p.column_ids, data_p.filters);
	bind_data.info.local_init(&init_info);
	if (!init_info.success) {
		throw Exception(init_info.error);
	}
	return std::move(result);
}

unique_ptr<NodeStatistics> CTableFunctionCardinality(ClientContext &context, const FunctionData *bind_data_p) {
	auto &bind_data = bind_data_p->Cast<CTableBindData>();
	if (!bind_data.stats) {
		return nullptr;
	}
	return make_uniq<NodeStatistics>(*bind_data.stats);
}

void CTableFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind_data = data_p.bind_data->Cast<CTableBindData>();
	auto &global_data = (CTableGlobalInitData &)*data_p.global_state;
	auto &local_data = (CTableLocalInitData &)*data_p.local_state;
	CTableInternalFunctionInfo function_info(bind_data, global_data.init_data, local_data.init_data);
	bind_data.info.function(&function_info, reinterpret_cast<duckdb_data_chunk>(&output));
	if (!function_info.success) {
		throw Exception(function_info.error);
	}
}

} // namespace duckdb

//===--------------------------------------------------------------------===//
// Table Function
//===--------------------------------------------------------------------===//
duckdb_table_function duckdb_create_table_function() {
	auto function = new duckdb::TableFunction("", {}, duckdb::CTableFunction, duckdb::CTableFunctionBind,
	                                          duckdb::CTableFunctionInit, duckdb::CTableFunctionLocalInit);
	function->function_info = duckdb::make_shared<duckdb::CTableFunctionInfo>();
	function->cardinality = duckdb::CTableFunctionCardinality;
	return function;
}

void duckdb_destroy_table_function(duckdb_table_function *function) {
	if (function && *function) {
		auto tf = (duckdb::TableFunction *)*function;
		delete tf;
		*function = nullptr;
	}
}

void duckdb_table_function_set_name(duckdb_table_function function, const char *name) {
	if (!function || !name) {
		return;
	}
	auto tf = (duckdb::TableFunction *)function;
	tf->name = name;
}

void duckdb_table_function_add_parameter(duckdb_table_function function, duckdb_logical_type type) {
	if (!function || !type) {
		return;
	}
	auto tf = (duckdb::TableFunction *)function;
	auto logical_type = (duckdb::LogicalType *)type;
	tf->arguments.push_back(*logical_type);
}

void duckdb_table_function_add_named_parameter(duckdb_table_function function, const char *name,
                                               duckdb_logical_type type) {
	if (!function || !type) {
		return;
	}
	auto tf = (duckdb::TableFunction *)function;
	auto logical_type = (duckdb::LogicalType *)type;
	tf->named_parameters.insert({name, *logical_type});
}

void duckdb_table_function_set_extra_info(duckdb_table_function function, void *extra_info,
                                          duckdb_delete_callback_t destroy) {
	if (!function) {
		return;
	}
	auto tf = (duckdb::TableFunction *)function;
	auto info = (duckdb::CTableFunctionInfo *)tf->function_info.get();
	info->extra_info = extra_info;
	info->delete_callback = destroy;
}

void duckdb_table_function_set_bind(duckdb_table_function function, duckdb_table_function_bind_t bind) {
	if (!function || !bind) {
		return;
	}
	auto tf = (duckdb::TableFunction *)function;
	auto info = (duckdb::CTableFunctionInfo *)tf->function_info.get();
	info->bind = bind;
}

void duckdb_table_function_set_init(duckdb_table_function function, duckdb_table_function_init_t init) {
	if (!function || !init) {
		return;
	}
	auto tf = (duckdb::TableFunction *)function;
	auto info = (duckdb::CTableFunctionInfo *)tf->function_info.get();
	info->init = init;
}

void duckdb_table_function_set_local_init(duckdb_table_function function, duckdb_table_function_init_t init) {
	if (!function || !init) {
		return;
	}
	auto tf = (duckdb::TableFunction *)function;
	auto info = (duckdb::CTableFunctionInfo *)tf->function_info.get();
	info->local_init = init;
}

void duckdb_table_function_set_function(duckdb_table_function table_function, duckdb_table_function_t function) {
	if (!table_function || !function) {
		return;
	}
	auto tf = (duckdb::TableFunction *)table_function;
	auto info = (duckdb::CTableFunctionInfo *)tf->function_info.get();
	info->function = function;
}

void duckdb_table_function_supports_projection_pushdown(duckdb_table_function table_function, bool pushdown) {
	if (!table_function) {
		return;
	}
	auto tf = (duckdb::TableFunction *)table_function;
	tf->projection_pushdown = pushdown;
}

duckdb_state duckdb_register_table_function(duckdb_connection connection, duckdb_table_function function) {
	if (!connection || !function) {
		return DuckDBError;
	}
	auto con = (duckdb::Connection *)connection;
	auto tf = (duckdb::TableFunction *)function;
	auto info = (duckdb::CTableFunctionInfo *)tf->function_info.get();
	if (tf->name.empty() || !info->bind || !info->init || !info->function) {
		return DuckDBError;
	}
	con->context->RunFunctionInTransaction([&]() {
		auto &catalog = duckdb::Catalog::GetSystemCatalog(*con->context);
		duckdb::CreateTableFunctionInfo tf_info(*tf);

		// create the function in the catalog
		catalog.CreateTableFunction(*con->context, tf_info);
	});
	return DuckDBSuccess;
}

//===--------------------------------------------------------------------===//
// Bind Interface
//===--------------------------------------------------------------------===//
void *duckdb_bind_get_extra_info(duckdb_bind_info info) {
	if (!info) {
		return nullptr;
	}
	auto bind_info = (duckdb::CTableInternalBindInfo *)info;
	return bind_info->function_info.extra_info;
}

void duckdb_bind_add_result_column(duckdb_bind_info info, const char *name, duckdb_logical_type type) {
	if (!info || !name || !type) {
		return;
	}
	auto bind_info = (duckdb::CTableInternalBindInfo *)info;
	bind_info->names.push_back(name);
	bind_info->return_types.push_back(*((duckdb::LogicalType *)type));
}

idx_t duckdb_bind_get_parameter_count(duckdb_bind_info info) {
	if (!info) {
		return 0;
	}
	auto bind_info = (duckdb::CTableInternalBindInfo *)info;
	return bind_info->input.inputs.size();
}

duckdb_value duckdb_bind_get_parameter(duckdb_bind_info info, idx_t index) {
	if (!info || index >= duckdb_bind_get_parameter_count(info)) {
		return nullptr;
	}
	auto bind_info = (duckdb::CTableInternalBindInfo *)info;
	return reinterpret_cast<duckdb_value>(new duckdb::Value(bind_info->input.inputs[index]));
}

duckdb_value duckdb_bind_get_named_parameter(duckdb_bind_info info, const char *name) {
	if (!info || !name) {
		return nullptr;
	}
	auto bind_info = (duckdb::CTableInternalBindInfo *)info;
	auto t = bind_info->input.named_parameters.find(name);
	if (t == bind_info->input.named_parameters.end()) {
		return nullptr;
	} else {
		return reinterpret_cast<duckdb_value>(new duckdb::Value(t->second));
	}
}

void duckdb_bind_set_bind_data(duckdb_bind_info info, void *bind_data, duckdb_delete_callback_t destroy) {
	if (!info) {
		return;
	}
	auto bind_info = (duckdb::CTableInternalBindInfo *)info;
	bind_info->bind_data.bind_data = bind_data;
	bind_info->bind_data.delete_callback = destroy;
}

void duckdb_bind_set_cardinality(duckdb_bind_info info, idx_t cardinality, bool is_exact) {
	if (!info) {
		return;
	}
	auto bind_info = (duckdb::CTableInternalBindInfo *)info;
	if (is_exact) {
		bind_info->bind_data.stats = duckdb::make_uniq<duckdb::NodeStatistics>(cardinality);
	} else {
		bind_info->bind_data.stats = duckdb::make_uniq<duckdb::NodeStatistics>(cardinality, cardinality);
	}
}

void duckdb_bind_set_error(duckdb_bind_info info, const char *error) {
	if (!info || !error) {
		return;
	}
	auto function_info = (duckdb::CTableInternalBindInfo *)info;
	function_info->error = error;
	function_info->success = false;
}

//===--------------------------------------------------------------------===//
// Init Interface
//===--------------------------------------------------------------------===//
void *duckdb_init_get_extra_info(duckdb_init_info info) {
	if (!info) {
		return nullptr;
	}
	auto init_info = (duckdb::CTableInternalInitInfo *)info;
	return init_info->bind_data.info.extra_info;
}

void *duckdb_init_get_bind_data(duckdb_init_info info) {
	if (!info) {
		return nullptr;
	}
	auto init_info = (duckdb::CTableInternalInitInfo *)info;
	return init_info->bind_data.bind_data;
}

void duckdb_init_set_init_data(duckdb_init_info info, void *init_data, duckdb_delete_callback_t destroy) {
	if (!info) {
		return;
	}
	auto init_info = (duckdb::CTableInternalInitInfo *)info;
	init_info->init_data.init_data = init_data;
	init_info->init_data.delete_callback = destroy;
}

void duckdb_init_set_error(duckdb_init_info info, const char *error) {
	if (!info || !error) {
		return;
	}
	auto function_info = (duckdb::CTableInternalInitInfo *)info;
	function_info->error = error;
	function_info->success = false;
}

idx_t duckdb_init_get_column_count(duckdb_init_info info) {
	if (!info) {
		return 0;
	}
	auto function_info = (duckdb::CTableInternalInitInfo *)info;
	return function_info->column_ids.size();
}

idx_t duckdb_init_get_column_index(duckdb_init_info info, idx_t column_index) {
	if (!info) {
		return 0;
	}
	auto function_info = (duckdb::CTableInternalInitInfo *)info;
	if (column_index >= function_info->column_ids.size()) {
		return 0;
	}
	return function_info->column_ids[column_index];
}

void duckdb_init_set_max_threads(duckdb_init_info info, idx_t max_threads) {
	if (!info) {
		return;
	}
	auto function_info = (duckdb::CTableInternalInitInfo *)info;
	function_info->init_data.max_threads = max_threads;
}

//===--------------------------------------------------------------------===//
// Function Interface
//===--------------------------------------------------------------------===//
void *duckdb_function_get_extra_info(duckdb_function_info info) {
	if (!info) {
		return nullptr;
	}
	auto function_info = (duckdb::CTableInternalFunctionInfo *)info;
	return function_info->bind_data.info.extra_info;
}

void *duckdb_function_get_bind_data(duckdb_function_info info) {
	if (!info) {
		return nullptr;
	}
	auto function_info = (duckdb::CTableInternalFunctionInfo *)info;
	return function_info->bind_data.bind_data;
}

void *duckdb_function_get_init_data(duckdb_function_info info) {
	if (!info) {
		return nullptr;
	}
	auto function_info = (duckdb::CTableInternalFunctionInfo *)info;
	return function_info->init_data.init_data;
}

void *duckdb_function_get_local_init_data(duckdb_function_info info) {
	if (!info) {
		return nullptr;
	}
	auto function_info = (duckdb::CTableInternalFunctionInfo *)info;
	return function_info->local_data.init_data;
}

void duckdb_function_set_error(duckdb_function_info info, const char *error) {
	if (!info || !error) {
		return;
	}
	auto function_info = (duckdb::CTableInternalFunctionInfo *)info;
	function_info->error = error;
	function_info->success = false;
}



using duckdb::DatabaseData;

struct CAPITaskState {
	CAPITaskState(duckdb::DatabaseInstance &db)
	    : db(db), marker(duckdb::make_uniq<duckdb::atomic<bool>>(true)), execute_count(0) {
	}

	duckdb::DatabaseInstance &db;
	duckdb::unique_ptr<duckdb::atomic<bool>> marker;
	duckdb::atomic<idx_t> execute_count;
};

void duckdb_execute_tasks(duckdb_database database, idx_t max_tasks) {
	if (!database) {
		return;
	}
	auto wrapper = (DatabaseData *)database;
	auto &scheduler = duckdb::TaskScheduler::GetScheduler(*wrapper->database->instance);
	scheduler.ExecuteTasks(max_tasks);
}

duckdb_task_state duckdb_create_task_state(duckdb_database database) {
	if (!database) {
		return nullptr;
	}
	auto wrapper = (DatabaseData *)database;
	auto state = new CAPITaskState(*wrapper->database->instance);
	return state;
}

void duckdb_execute_tasks_state(duckdb_task_state state_p) {
	if (!state_p) {
		return;
	}
	auto state = (CAPITaskState *)state_p;
	auto &scheduler = duckdb::TaskScheduler::GetScheduler(state->db);
	state->execute_count++;
	scheduler.ExecuteForever(state->marker.get());
}

idx_t duckdb_execute_n_tasks_state(duckdb_task_state state_p, idx_t max_tasks) {
	if (!state_p) {
		return 0;
	}
	auto state = (CAPITaskState *)state_p;
	auto &scheduler = duckdb::TaskScheduler::GetScheduler(state->db);
	return scheduler.ExecuteTasks(state->marker.get(), max_tasks);
}

void duckdb_finish_execution(duckdb_task_state state_p) {
	if (!state_p) {
		return;
	}
	auto state = (CAPITaskState *)state_p;
	*state->marker = false;
	if (state->execute_count > 0) {
		// signal to the threads to wake up
		auto &scheduler = duckdb::TaskScheduler::GetScheduler(state->db);
		scheduler.Signal(state->execute_count);
	}
}

bool duckdb_task_state_is_finished(duckdb_task_state state_p) {
	if (!state_p) {
		return false;
	}
	auto state = (CAPITaskState *)state_p;
	return !(*state->marker);
}

void duckdb_destroy_task_state(duckdb_task_state state_p) {
	if (!state_p) {
		return;
	}
	auto state = (CAPITaskState *)state_p;
	delete state;
}

bool duckdb_execution_is_finished(duckdb_connection con) {
	if (!con) {
		return false;
	}
	duckdb::Connection *conn = (duckdb::Connection *)con;
	return conn->context->ExecutionIsFinished();
}








#include <cstring>

using duckdb::date_t;
using duckdb::dtime_t;
using duckdb::FetchDefaultValue;
using duckdb::GetInternalCValue;
using duckdb::hugeint_t;
using duckdb::interval_t;
using duckdb::StringCast;
using duckdb::timestamp_t;
using duckdb::ToCStringCastWrapper;
using duckdb::UnsafeFetch;

bool duckdb_value_boolean(duckdb_result *result, idx_t col, idx_t row) {
	return GetInternalCValue<bool>(result, col, row);
}

int8_t duckdb_value_int8(duckdb_result *result, idx_t col, idx_t row) {
	return GetInternalCValue<int8_t>(result, col, row);
}

int16_t duckdb_value_int16(duckdb_result *result, idx_t col, idx_t row) {
	return GetInternalCValue<int16_t>(result, col, row);
}

int32_t duckdb_value_int32(duckdb_result *result, idx_t col, idx_t row) {
	return GetInternalCValue<int32_t>(result, col, row);
}

int64_t duckdb_value_int64(duckdb_result *result, idx_t col, idx_t row) {
	return GetInternalCValue<int64_t>(result, col, row);
}

static bool ResultIsDecimal(duckdb_result *result, idx_t col) {
	if (!result) {
		return false;
	}
	if (!result->internal_data) {
		return false;
	}
	auto result_data = (duckdb::DuckDBResultData *)result->internal_data;
	auto &query_result = result_data->result;
	auto &source_type = query_result->types[col];
	return source_type.id() == duckdb::LogicalTypeId::DECIMAL;
}

duckdb_decimal duckdb_value_decimal(duckdb_result *result, idx_t col, idx_t row) {
	if (!CanFetchValue(result, col, row) || !ResultIsDecimal(result, col)) {
		return FetchDefaultValue::Operation<duckdb_decimal>();
	}

	return GetInternalCValue<duckdb_decimal>(result, col, row);
}

duckdb_hugeint duckdb_value_hugeint(duckdb_result *result, idx_t col, idx_t row) {
	duckdb_hugeint result_value;
	auto internal_value = GetInternalCValue<hugeint_t>(result, col, row);
	result_value.lower = internal_value.lower;
	result_value.upper = internal_value.upper;
	return result_value;
}

uint8_t duckdb_value_uint8(duckdb_result *result, idx_t col, idx_t row) {
	return GetInternalCValue<uint8_t>(result, col, row);
}

uint16_t duckdb_value_uint16(duckdb_result *result, idx_t col, idx_t row) {
	return GetInternalCValue<uint16_t>(result, col, row);
}

uint32_t duckdb_value_uint32(duckdb_result *result, idx_t col, idx_t row) {
	return GetInternalCValue<uint32_t>(result, col, row);
}

uint64_t duckdb_value_uint64(duckdb_result *result, idx_t col, idx_t row) {
	return GetInternalCValue<uint64_t>(result, col, row);
}

float duckdb_value_float(duckdb_result *result, idx_t col, idx_t row) {
	return GetInternalCValue<float>(result, col, row);
}

double duckdb_value_double(duckdb_result *result, idx_t col, idx_t row) {
	return GetInternalCValue<double>(result, col, row);
}

duckdb_date duckdb_value_date(duckdb_result *result, idx_t col, idx_t row) {
	duckdb_date result_value;
	result_value.days = GetInternalCValue<date_t>(result, col, row).days;
	return result_value;
}

duckdb_time duckdb_value_time(duckdb_result *result, idx_t col, idx_t row) {
	duckdb_time result_value;
	result_value.micros = GetInternalCValue<dtime_t>(result, col, row).micros;
	return result_value;
}

duckdb_timestamp duckdb_value_timestamp(duckdb_result *result, idx_t col, idx_t row) {
	duckdb_timestamp result_value;
	result_value.micros = GetInternalCValue<timestamp_t>(result, col, row).value;
	return result_value;
}

duckdb_interval duckdb_value_interval(duckdb_result *result, idx_t col, idx_t row) {
	duckdb_interval result_value;
	auto ival = GetInternalCValue<interval_t>(result, col, row);
	result_value.months = ival.months;
	result_value.days = ival.days;
	result_value.micros = ival.micros;
	return result_value;
}

char *duckdb_value_varchar(duckdb_result *result, idx_t col, idx_t row) {
	return duckdb_value_string(result, col, row).data;
}

duckdb_string duckdb_value_string(duckdb_result *result, idx_t col, idx_t row) {
	return GetInternalCValue<duckdb_string, ToCStringCastWrapper<StringCast>>(result, col, row);
}

char *duckdb_value_varchar_internal(duckdb_result *result, idx_t col, idx_t row) {
	return duckdb_value_string_internal(result, col, row).data;
}

duckdb_string duckdb_value_string_internal(duckdb_result *result, idx_t col, idx_t row) {
	if (!CanFetchValue(result, col, row)) {
		return FetchDefaultValue::Operation<duckdb_string>();
	}
	if (duckdb_column_type(result, col) != DUCKDB_TYPE_VARCHAR) {
		return FetchDefaultValue::Operation<duckdb_string>();
	}
	// FIXME: this obviously does not work when there are null bytes in the string
	// we need to remove the deprecated C result materialization to get that to work correctly
	// since the deprecated C result materialization stores strings as null-terminated
	duckdb_string res;
	res.data = UnsafeFetch<char *>(result, col, row);
	res.size = strlen(res.data);
	return res;
}

duckdb_blob duckdb_value_blob(duckdb_result *result, idx_t col, idx_t row) {
	if (CanFetchValue(result, col, row) && result->__deprecated_columns[col].__deprecated_type == DUCKDB_TYPE_BLOB) {
		auto internal_result = UnsafeFetch<duckdb_blob>(result, col, row);

		duckdb_blob result_blob;
		result_blob.data = malloc(internal_result.size);
		result_blob.size = internal_result.size;
		memcpy(result_blob.data, internal_result.data, internal_result.size);
		return result_blob;
	}
	return FetchDefaultValue::Operation<duckdb_blob>();
}

bool duckdb_value_is_null(duckdb_result *result, idx_t col, idx_t row) {
	if (!CanUseDeprecatedFetch(result, col, row)) {
		return false;
	}
	return result->__deprecated_columns[col].__deprecated_nullmask[row];
}
