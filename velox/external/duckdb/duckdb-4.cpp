// See https://raw.githubusercontent.com/duckdb/duckdb/master/LICENSE for licensing information

#include "duckdb.hpp"
#include "duckdb-internal.hpp"
#ifndef DUCKDB_AMALGAMATION
#error header mismatch
#endif



namespace duckdb {

template <class OP>
struct ComparisonOperationWrapper {
	template <class T>
	static inline bool Operation(T left, T right, bool left_is_null, bool right_is_null) {
		if (left_is_null || right_is_null) {
			return false;
		}
		return OP::Operation(left, right);
	}
};

struct InitialNestedLoopJoin {
	template <class T, class OP>
	static idx_t Operation(Vector &left, Vector &right, idx_t left_size, idx_t right_size, idx_t &lpos, idx_t &rpos,
	                       SelectionVector &lvector, SelectionVector &rvector, idx_t current_match_count) {
		// initialize phase of nested loop join
		// fill lvector and rvector with matches from the base vectors
		UnifiedVectorFormat left_data, right_data;
		left.ToUnifiedFormat(left_size, left_data);
		right.ToUnifiedFormat(right_size, right_data);

		auto ldata = (T *)left_data.data;
		auto rdata = (T *)right_data.data;
		idx_t result_count = 0;
		for (; rpos < right_size; rpos++) {
			idx_t right_position = right_data.sel->get_index(rpos);
			bool right_is_valid = right_data.validity.RowIsValid(right_position);
			for (; lpos < left_size; lpos++) {
				if (result_count == STANDARD_VECTOR_SIZE) {
					// out of space!
					return result_count;
				}
				idx_t left_position = left_data.sel->get_index(lpos);
				bool left_is_valid = left_data.validity.RowIsValid(left_position);
				if (OP::Operation(ldata[left_position], rdata[right_position], !left_is_valid, !right_is_valid)) {
					// emit tuple
					lvector.set_index(result_count, lpos);
					rvector.set_index(result_count, rpos);
					result_count++;
				}
			}
			lpos = 0;
		}
		return result_count;
	}
};

struct RefineNestedLoopJoin {
	template <class T, class OP>
	static idx_t Operation(Vector &left, Vector &right, idx_t left_size, idx_t right_size, idx_t &lpos, idx_t &rpos,
	                       SelectionVector &lvector, SelectionVector &rvector, idx_t current_match_count) {
		UnifiedVectorFormat left_data, right_data;
		left.ToUnifiedFormat(left_size, left_data);
		right.ToUnifiedFormat(right_size, right_data);

		// refine phase of the nested loop join
		// refine lvector and rvector based on matches of subsequent conditions (in case there are multiple conditions
		// in the join)
		D_ASSERT(current_match_count > 0);
		auto ldata = (T *)left_data.data;
		auto rdata = (T *)right_data.data;
		idx_t result_count = 0;
		for (idx_t i = 0; i < current_match_count; i++) {
			auto lidx = lvector.get_index(i);
			auto ridx = rvector.get_index(i);
			auto left_idx = left_data.sel->get_index(lidx);
			auto right_idx = right_data.sel->get_index(ridx);
			bool left_is_valid = left_data.validity.RowIsValid(left_idx);
			bool right_is_valid = right_data.validity.RowIsValid(right_idx);
			if (OP::Operation(ldata[left_idx], rdata[right_idx], !left_is_valid, !right_is_valid)) {
				lvector.set_index(result_count, lidx);
				rvector.set_index(result_count, ridx);
				result_count++;
			}
		}
		return result_count;
	}
};

template <class NLTYPE, class OP>
static idx_t NestedLoopJoinTypeSwitch(Vector &left, Vector &right, idx_t left_size, idx_t right_size, idx_t &lpos,
                                      idx_t &rpos, SelectionVector &lvector, SelectionVector &rvector,
                                      idx_t current_match_count) {
	switch (left.GetType().InternalType()) {
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
		return NLTYPE::template Operation<int8_t, OP>(left, right, left_size, right_size, lpos, rpos, lvector, rvector,
		                                              current_match_count);
	case PhysicalType::INT16:
		return NLTYPE::template Operation<int16_t, OP>(left, right, left_size, right_size, lpos, rpos, lvector, rvector,
		                                               current_match_count);
	case PhysicalType::INT32:
		return NLTYPE::template Operation<int32_t, OP>(left, right, left_size, right_size, lpos, rpos, lvector, rvector,
		                                               current_match_count);
	case PhysicalType::INT64:
		return NLTYPE::template Operation<int64_t, OP>(left, right, left_size, right_size, lpos, rpos, lvector, rvector,
		                                               current_match_count);
	case PhysicalType::UINT8:
		return NLTYPE::template Operation<uint8_t, OP>(left, right, left_size, right_size, lpos, rpos, lvector, rvector,
		                                               current_match_count);
	case PhysicalType::UINT16:
		return NLTYPE::template Operation<uint16_t, OP>(left, right, left_size, right_size, lpos, rpos, lvector,
		                                                rvector, current_match_count);
	case PhysicalType::UINT32:
		return NLTYPE::template Operation<uint32_t, OP>(left, right, left_size, right_size, lpos, rpos, lvector,
		                                                rvector, current_match_count);
	case PhysicalType::UINT64:
		return NLTYPE::template Operation<uint64_t, OP>(left, right, left_size, right_size, lpos, rpos, lvector,
		                                                rvector, current_match_count);
	case PhysicalType::INT128:
		return NLTYPE::template Operation<hugeint_t, OP>(left, right, left_size, right_size, lpos, rpos, lvector,
		                                                 rvector, current_match_count);
	case PhysicalType::FLOAT:
		return NLTYPE::template Operation<float, OP>(left, right, left_size, right_size, lpos, rpos, lvector, rvector,
		                                             current_match_count);
	case PhysicalType::DOUBLE:
		return NLTYPE::template Operation<double, OP>(left, right, left_size, right_size, lpos, rpos, lvector, rvector,
		                                              current_match_count);
	case PhysicalType::INTERVAL:
		return NLTYPE::template Operation<interval_t, OP>(left, right, left_size, right_size, lpos, rpos, lvector,
		                                                  rvector, current_match_count);
	case PhysicalType::VARCHAR:
		return NLTYPE::template Operation<string_t, OP>(left, right, left_size, right_size, lpos, rpos, lvector,
		                                                rvector, current_match_count);
	default:
		throw InternalException("Unimplemented type for join!");
	}
}

template <class NLTYPE>
idx_t NestedLoopJoinComparisonSwitch(Vector &left, Vector &right, idx_t left_size, idx_t right_size, idx_t &lpos,
                                     idx_t &rpos, SelectionVector &lvector, SelectionVector &rvector,
                                     idx_t current_match_count, ExpressionType comparison_type) {
	D_ASSERT(left.GetType() == right.GetType());
	switch (comparison_type) {
	case ExpressionType::COMPARE_EQUAL:
		return NestedLoopJoinTypeSwitch<NLTYPE, ComparisonOperationWrapper<duckdb::Equals>>(
		    left, right, left_size, right_size, lpos, rpos, lvector, rvector, current_match_count);
	case ExpressionType::COMPARE_NOTEQUAL:
		return NestedLoopJoinTypeSwitch<NLTYPE, ComparisonOperationWrapper<duckdb::NotEquals>>(
		    left, right, left_size, right_size, lpos, rpos, lvector, rvector, current_match_count);
	case ExpressionType::COMPARE_LESSTHAN:
		return NestedLoopJoinTypeSwitch<NLTYPE, ComparisonOperationWrapper<duckdb::LessThan>>(
		    left, right, left_size, right_size, lpos, rpos, lvector, rvector, current_match_count);
	case ExpressionType::COMPARE_GREATERTHAN:
		return NestedLoopJoinTypeSwitch<NLTYPE, ComparisonOperationWrapper<duckdb::GreaterThan>>(
		    left, right, left_size, right_size, lpos, rpos, lvector, rvector, current_match_count);
	case ExpressionType::COMPARE_LESSTHANOREQUALTO:
		return NestedLoopJoinTypeSwitch<NLTYPE, ComparisonOperationWrapper<duckdb::LessThanEquals>>(
		    left, right, left_size, right_size, lpos, rpos, lvector, rvector, current_match_count);
	case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
		return NestedLoopJoinTypeSwitch<NLTYPE, ComparisonOperationWrapper<duckdb::GreaterThanEquals>>(
		    left, right, left_size, right_size, lpos, rpos, lvector, rvector, current_match_count);
	case ExpressionType::COMPARE_DISTINCT_FROM:
		return NestedLoopJoinTypeSwitch<NLTYPE, duckdb::DistinctFrom>(left, right, left_size, right_size, lpos, rpos,
		                                                              lvector, rvector, current_match_count);
	default:
		throw NotImplementedException("Unimplemented comparison type for join!");
	}
}

idx_t NestedLoopJoinInner::Perform(idx_t &lpos, idx_t &rpos, DataChunk &left_conditions, DataChunk &right_conditions,
                                   SelectionVector &lvector, SelectionVector &rvector,
                                   const vector<JoinCondition> &conditions) {
	D_ASSERT(left_conditions.ColumnCount() == right_conditions.ColumnCount());
	if (lpos >= left_conditions.size() || rpos >= right_conditions.size()) {
		return 0;
	}
	// for the first condition, lvector and rvector are not set yet
	// we initialize them using the InitialNestedLoopJoin
	idx_t match_count = NestedLoopJoinComparisonSwitch<InitialNestedLoopJoin>(
	    left_conditions.data[0], right_conditions.data[0], left_conditions.size(), right_conditions.size(), lpos, rpos,
	    lvector, rvector, 0, conditions[0].comparison);
	// now resolve the rest of the conditions
	for (idx_t i = 1; i < conditions.size(); i++) {
		// check if we have run out of tuples to compare
		if (match_count == 0) {
			return 0;
		}
		// if not, get the vectors to compare
		Vector &l = left_conditions.data[i];
		Vector &r = right_conditions.data[i];
		// then we refine the currently obtained results using the RefineNestedLoopJoin
		match_count = NestedLoopJoinComparisonSwitch<RefineNestedLoopJoin>(
		    l, r, left_conditions.size(), right_conditions.size(), lpos, rpos, lvector, rvector, match_count,
		    conditions[i].comparison);
	}
	return match_count;
}

} // namespace duckdb




namespace duckdb {

template <class T, class OP>
static void TemplatedMarkJoin(Vector &left, Vector &right, idx_t lcount, idx_t rcount, bool found_match[]) {
	UnifiedVectorFormat left_data, right_data;
	left.ToUnifiedFormat(lcount, left_data);
	right.ToUnifiedFormat(rcount, right_data);

	auto ldata = (T *)left_data.data;
	auto rdata = (T *)right_data.data;
	for (idx_t i = 0; i < lcount; i++) {
		if (found_match[i]) {
			continue;
		}
		auto lidx = left_data.sel->get_index(i);
		if (!left_data.validity.RowIsValid(lidx)) {
			continue;
		}
		for (idx_t j = 0; j < rcount; j++) {
			auto ridx = right_data.sel->get_index(j);
			if (!right_data.validity.RowIsValid(ridx)) {
				continue;
			}
			if (OP::Operation(ldata[lidx], rdata[ridx])) {
				found_match[i] = true;
				break;
			}
		}
	}
}

static void MarkJoinNested(Vector &left, Vector &right, idx_t lcount, idx_t rcount, bool found_match[],
                           ExpressionType comparison_type) {
	Vector left_reference(left.GetType());
	SelectionVector true_sel(rcount);
	for (idx_t i = 0; i < lcount; i++) {
		if (found_match[i]) {
			continue;
		}
		ConstantVector::Reference(left_reference, left, i, rcount);
		idx_t count;
		switch (comparison_type) {
		case ExpressionType::COMPARE_EQUAL:
			count = VectorOperations::Equals(left_reference, right, nullptr, rcount, nullptr, nullptr);
			break;
		case ExpressionType::COMPARE_NOTEQUAL:
			count = VectorOperations::NotEquals(left_reference, right, nullptr, rcount, nullptr, nullptr);
			break;
		case ExpressionType::COMPARE_LESSTHAN:
			count = VectorOperations::LessThan(left_reference, right, nullptr, rcount, nullptr, nullptr);
			break;
		case ExpressionType::COMPARE_GREATERTHAN:
			count = VectorOperations::GreaterThan(left_reference, right, nullptr, rcount, nullptr, nullptr);
			break;
		case ExpressionType::COMPARE_LESSTHANOREQUALTO:
			count = VectorOperations::LessThanEquals(left_reference, right, nullptr, rcount, nullptr, nullptr);
			break;
		case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
			count = VectorOperations::GreaterThanEquals(left_reference, right, nullptr, rcount, nullptr, nullptr);
			break;
		default:
			throw InternalException("Unsupported comparison type for MarkJoinNested");
		}
		if (count > 0) {
			found_match[i] = true;
		}
	}
}

template <class OP>
static void MarkJoinSwitch(Vector &left, Vector &right, idx_t lcount, idx_t rcount, bool found_match[]) {
	switch (left.GetType().InternalType()) {
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
		return TemplatedMarkJoin<int8_t, OP>(left, right, lcount, rcount, found_match);
	case PhysicalType::INT16:
		return TemplatedMarkJoin<int16_t, OP>(left, right, lcount, rcount, found_match);
	case PhysicalType::INT32:
		return TemplatedMarkJoin<int32_t, OP>(left, right, lcount, rcount, found_match);
	case PhysicalType::INT64:
		return TemplatedMarkJoin<int64_t, OP>(left, right, lcount, rcount, found_match);
	case PhysicalType::INT128:
		return TemplatedMarkJoin<hugeint_t, OP>(left, right, lcount, rcount, found_match);
	case PhysicalType::UINT8:
		return TemplatedMarkJoin<uint8_t, OP>(left, right, lcount, rcount, found_match);
	case PhysicalType::UINT16:
		return TemplatedMarkJoin<uint16_t, OP>(left, right, lcount, rcount, found_match);
	case PhysicalType::UINT32:
		return TemplatedMarkJoin<uint32_t, OP>(left, right, lcount, rcount, found_match);
	case PhysicalType::UINT64:
		return TemplatedMarkJoin<uint64_t, OP>(left, right, lcount, rcount, found_match);
	case PhysicalType::FLOAT:
		return TemplatedMarkJoin<float, OP>(left, right, lcount, rcount, found_match);
	case PhysicalType::DOUBLE:
		return TemplatedMarkJoin<double, OP>(left, right, lcount, rcount, found_match);
	case PhysicalType::VARCHAR:
		return TemplatedMarkJoin<string_t, OP>(left, right, lcount, rcount, found_match);
	default:
		throw NotImplementedException("Unimplemented type for mark join!");
	}
}

static void MarkJoinComparisonSwitch(Vector &left, Vector &right, idx_t lcount, idx_t rcount, bool found_match[],
                                     ExpressionType comparison_type) {
	switch (left.GetType().InternalType()) {
	case PhysicalType::STRUCT:
	case PhysicalType::LIST:
		return MarkJoinNested(left, right, lcount, rcount, found_match, comparison_type);
	default:
		break;
	}
	D_ASSERT(left.GetType() == right.GetType());
	switch (comparison_type) {
	case ExpressionType::COMPARE_EQUAL:
		return MarkJoinSwitch<duckdb::Equals>(left, right, lcount, rcount, found_match);
	case ExpressionType::COMPARE_NOTEQUAL:
		return MarkJoinSwitch<duckdb::NotEquals>(left, right, lcount, rcount, found_match);
	case ExpressionType::COMPARE_LESSTHAN:
		return MarkJoinSwitch<duckdb::LessThan>(left, right, lcount, rcount, found_match);
	case ExpressionType::COMPARE_GREATERTHAN:
		return MarkJoinSwitch<duckdb::GreaterThan>(left, right, lcount, rcount, found_match);
	case ExpressionType::COMPARE_LESSTHANOREQUALTO:
		return MarkJoinSwitch<duckdb::LessThanEquals>(left, right, lcount, rcount, found_match);
	case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
		return MarkJoinSwitch<duckdb::GreaterThanEquals>(left, right, lcount, rcount, found_match);
	default:
		throw NotImplementedException("Unimplemented comparison type for join!");
	}
}

void NestedLoopJoinMark::Perform(DataChunk &left, ColumnDataCollection &right, bool found_match[],
                                 const vector<JoinCondition> &conditions) {
	// initialize a new temporary selection vector for the left chunk
	// loop over all chunks in the RHS
	ColumnDataScanState scan_state;
	right.InitializeScan(scan_state);

	DataChunk scan_chunk;
	right.InitializeScanChunk(scan_chunk);

	while (right.Scan(scan_state, scan_chunk)) {
		for (idx_t i = 0; i < conditions.size(); i++) {
			MarkJoinComparisonSwitch(left.data[i], scan_chunk.data[i], left.size(), scan_chunk.size(), found_match,
			                         conditions[i].comparison);
		}
	}
}

} // namespace duckdb




namespace duckdb {

AggregateObject::AggregateObject(AggregateFunction function, FunctionData *bind_data, idx_t child_count,
                                 idx_t payload_size, AggregateType aggr_type, PhysicalType return_type,
                                 Expression *filter)
    : function(std::move(function)),
      bind_data_wrapper(bind_data ? make_shared<FunctionDataWrapper>(bind_data->Copy()) : nullptr),
      child_count(child_count), payload_size(payload_size), aggr_type(aggr_type), return_type(return_type),
      filter(filter) {
}

AggregateObject::AggregateObject(BoundAggregateExpression *aggr)
    : AggregateObject(aggr->function, aggr->bind_info.get(), aggr->children.size(),
                      AlignValue(aggr->function.state_size()), aggr->aggr_type, aggr->return_type.InternalType(),
                      aggr->filter.get()) {
}

AggregateObject::AggregateObject(BoundWindowExpression &window)
    : AggregateObject(*window.aggregate, window.bind_info.get(), window.children.size(),
                      AlignValue(window.aggregate->state_size()), AggregateType::NON_DISTINCT,
                      window.return_type.InternalType(), window.filter_expr.get()) {
}

vector<AggregateObject> AggregateObject::CreateAggregateObjects(const vector<BoundAggregateExpression *> &bindings) {
	vector<AggregateObject> aggregates;
	aggregates.reserve(aggregates.size());
	for (auto &binding : bindings) {
		aggregates.emplace_back(binding);
	}
	return aggregates;
}

AggregateFilterData::AggregateFilterData(ClientContext &context, Expression &filter_expr,
                                         const vector<LogicalType> &payload_types)
    : filter_executor(context, &filter_expr), true_sel(STANDARD_VECTOR_SIZE) {
	if (payload_types.empty()) {
		return;
	}
	filtered_payload.Initialize(Allocator::Get(context), payload_types);
}

idx_t AggregateFilterData::ApplyFilter(DataChunk &payload) {
	filtered_payload.Reset();

	auto count = filter_executor.SelectExpression(payload, true_sel);
	filtered_payload.Slice(payload, true_sel, count);
	return count;
}

AggregateFilterDataSet::AggregateFilterDataSet() {
}

void AggregateFilterDataSet::Initialize(ClientContext &context, const vector<AggregateObject> &aggregates,
                                        const vector<LogicalType> &payload_types) {
	bool has_filters = false;
	for (auto &aggregate : aggregates) {
		if (aggregate.filter) {
			has_filters = true;
			break;
		}
	}
	if (!has_filters) {
		// no filters: nothing to do
		return;
	}
	filter_data.resize(aggregates.size());
	for (idx_t aggr_idx = 0; aggr_idx < aggregates.size(); aggr_idx++) {
		auto &aggr = aggregates[aggr_idx];
		if (aggr.filter) {
			filter_data[aggr_idx] = make_uniq<AggregateFilterData>(context, *aggr.filter, payload_types);
		}
	}
}

AggregateFilterData &AggregateFilterDataSet::GetFilterData(idx_t aggr_idx) {
	D_ASSERT(aggr_idx < filter_data.size());
	D_ASSERT(filter_data[aggr_idx]);
	return *filter_data[aggr_idx];
}
} // namespace duckdb






namespace duckdb {

//! Shared information about a collection of distinct aggregates
DistinctAggregateCollectionInfo::DistinctAggregateCollectionInfo(const vector<unique_ptr<Expression>> &aggregates,
                                                                 vector<idx_t> indices)
    : indices(std::move(indices)), aggregates(aggregates) {
	table_count = CreateTableIndexMap();

	const idx_t aggregate_count = aggregates.size();

	total_child_count = 0;
	for (idx_t i = 0; i < aggregate_count; i++) {
		auto &aggregate = aggregates[i]->Cast<BoundAggregateExpression>();

		if (!aggregate.IsDistinct()) {
			continue;
		}
		total_child_count += aggregate.children.size();
	}
}

//! Stateful data for the distinct aggregates

DistinctAggregateState::DistinctAggregateState(const DistinctAggregateData &data, ClientContext &client)
    : child_executor(client) {

	radix_states.resize(data.info.table_count);
	distinct_output_chunks.resize(data.info.table_count);

	idx_t aggregate_count = data.info.aggregates.size();
	for (idx_t i = 0; i < aggregate_count; i++) {
		auto &aggregate = data.info.aggregates[i]->Cast<BoundAggregateExpression>();

		// Initialize the child executor and get the payload types for every aggregate
		for (auto &child : aggregate.children) {
			child_executor.AddExpression(*child);
		}
		if (!aggregate.IsDistinct()) {
			continue;
		}
		D_ASSERT(data.info.table_map.count(i));
		idx_t table_idx = data.info.table_map.at(i);
		if (data.radix_tables[table_idx] == nullptr) {
			//! This table is unused because the aggregate shares its data with another
			continue;
		}

		// Get the global sinkstate for the aggregate
		auto &radix_table = *data.radix_tables[table_idx];
		radix_states[table_idx] = radix_table.GetGlobalSinkState(client);

		// Fill the chunk_types (group_by + children)
		vector<LogicalType> chunk_types;
		for (auto &group_type : data.grouped_aggregate_data[table_idx]->group_types) {
			chunk_types.push_back(group_type);
		}

		// This is used in Finalize to get the data from the radix table
		distinct_output_chunks[table_idx] = make_uniq<DataChunk>();
		distinct_output_chunks[table_idx]->Initialize(client, chunk_types);
	}
}

//! Persistent + shared (read-only) data for the distinct aggregates
DistinctAggregateData::DistinctAggregateData(const DistinctAggregateCollectionInfo &info)
    : DistinctAggregateData(info, {}, nullptr) {
}

DistinctAggregateData::DistinctAggregateData(const DistinctAggregateCollectionInfo &info, const GroupingSet &groups,
                                             const vector<unique_ptr<Expression>> *group_expressions)
    : info(info) {
	grouped_aggregate_data.resize(info.table_count);
	radix_tables.resize(info.table_count);
	grouping_sets.resize(info.table_count);

	for (auto &i : info.indices) {
		auto &aggregate = info.aggregates[i]->Cast<BoundAggregateExpression>();

		D_ASSERT(info.table_map.count(i));
		idx_t table_idx = info.table_map.at(i);
		if (radix_tables[table_idx] != nullptr) {
			//! This aggregate shares a table with another aggregate, and the table is already initialized
			continue;
		}
		// The grouping set contains the indices of the chunk that correspond to the data vector
		// that will be used to figure out in which bucket the payload should be put
		auto &grouping_set = grouping_sets[table_idx];
		//! Populate the group with the children of the aggregate
		for (auto &group : groups) {
			grouping_set.insert(group);
		}
		idx_t group_by_size = group_expressions ? group_expressions->size() : 0;
		for (idx_t set_idx = 0; set_idx < aggregate.children.size(); set_idx++) {
			grouping_set.insert(set_idx + group_by_size);
		}
		// Create the hashtable for the aggregate
		grouped_aggregate_data[table_idx] = make_uniq<GroupedAggregateData>();
		grouped_aggregate_data[table_idx]->InitializeDistinct(info.aggregates[i], group_expressions);
		radix_tables[table_idx] =
		    make_uniq<RadixPartitionedHashTable>(grouping_set, *grouped_aggregate_data[table_idx]);

		// Fill the chunk_types (only contains the payload of the distinct aggregates)
		vector<LogicalType> chunk_types;
		for (auto &child_p : aggregate.children) {
			chunk_types.push_back(child_p->return_type);
		}
	}
}

using aggr_ref_t = reference<BoundAggregateExpression>;

struct FindMatchingAggregate {
	explicit FindMatchingAggregate(const aggr_ref_t &aggr) : aggr_r(aggr) {
	}
	bool operator()(const aggr_ref_t other_r) {
		auto &other = other_r.get();
		auto &aggr = aggr_r.get();
		if (other.children.size() != aggr.children.size()) {
			return false;
		}
		if (!Expression::Equals(aggr.filter.get(), other.filter.get())) {
			return false;
		}
		for (idx_t i = 0; i < aggr.children.size(); i++) {
			auto &other_child = other.children[i]->Cast<BoundReferenceExpression>();
			auto &aggr_child = aggr.children[i]->Cast<BoundReferenceExpression>();
			if (other_child.index != aggr_child.index) {
				return false;
			}
		}
		return true;
	}
	const aggr_ref_t aggr_r;
};

idx_t DistinctAggregateCollectionInfo::CreateTableIndexMap() {
	vector<aggr_ref_t> table_inputs;

	D_ASSERT(table_map.empty());
	for (auto &agg_idx : indices) {
		D_ASSERT(agg_idx < aggregates.size());
		auto &aggregate = aggregates[agg_idx]->Cast<BoundAggregateExpression>();

		auto matching_inputs =
		    std::find_if(table_inputs.begin(), table_inputs.end(), FindMatchingAggregate(std::ref(aggregate)));
		if (matching_inputs != table_inputs.end()) {
			//! Assign the existing table to the aggregate
			idx_t found_idx = std::distance(table_inputs.begin(), matching_inputs);
			table_map[agg_idx] = found_idx;
			continue;
		}
		//! Create a new table and assign its index to the aggregate
		table_map[agg_idx] = table_inputs.size();
		table_inputs.push_back(std::ref(aggregate));
	}
	//! Every distinct aggregate needs to be assigned an index
	D_ASSERT(table_map.size() == indices.size());
	//! There can not be more tables than there are distinct aggregates
	D_ASSERT(table_inputs.size() <= indices.size());

	return table_inputs.size();
}

bool DistinctAggregateCollectionInfo::AnyDistinct() const {
	return !indices.empty();
}

const unsafe_vector<idx_t> &DistinctAggregateCollectionInfo::Indices() const {
	return this->indices;
}

static vector<idx_t> GetDistinctIndices(vector<unique_ptr<Expression>> &aggregates) {
	vector<idx_t> distinct_indices;
	for (idx_t i = 0; i < aggregates.size(); i++) {
		auto &aggregate = aggregates[i];
		auto &aggr = aggregate->Cast<BoundAggregateExpression>();
		if (aggr.IsDistinct()) {
			distinct_indices.push_back(i);
		}
	}
	return distinct_indices;
}

unique_ptr<DistinctAggregateCollectionInfo>
DistinctAggregateCollectionInfo::Create(vector<unique_ptr<Expression>> &aggregates) {
	vector<idx_t> indices = GetDistinctIndices(aggregates);
	if (indices.empty()) {
		return nullptr;
	}
	return make_uniq<DistinctAggregateCollectionInfo>(aggregates, std::move(indices));
}

bool DistinctAggregateData::IsDistinct(idx_t index) const {
	bool is_distinct = !radix_tables.empty() && info.table_map.count(index);
#ifdef DEBUG
	//! Make sure that if it is distinct, it's also in the indices
	//! And if it's not distinct, that it's also not in the indices
	bool found = false;
	for (auto &idx : info.indices) {
		if (idx == index) {
			found = true;
			break;
		}
	}
	D_ASSERT(found == is_distinct);
#endif
	return is_distinct;
}

} // namespace duckdb


namespace duckdb {

idx_t GroupedAggregateData::GroupCount() const {
	return groups.size();
}

const vector<vector<idx_t>> &GroupedAggregateData::GetGroupingFunctions() const {
	return grouping_functions;
}

void GroupedAggregateData::InitializeGroupby(vector<unique_ptr<Expression>> groups,
                                             vector<unique_ptr<Expression>> expressions,
                                             vector<unsafe_vector<idx_t>> grouping_functions) {
	InitializeGroupbyGroups(std::move(groups));
	vector<LogicalType> payload_types_filters;

	SetGroupingFunctions(grouping_functions);

	filter_count = 0;
	for (auto &expr : expressions) {
		D_ASSERT(expr->expression_class == ExpressionClass::BOUND_AGGREGATE);
		D_ASSERT(expr->IsAggregate());
		auto &aggr = expr->Cast<BoundAggregateExpression>();
		bindings.push_back(&aggr);

		aggregate_return_types.push_back(aggr.return_type);
		for (auto &child : aggr.children) {
			payload_types.push_back(child->return_type);
		}
		if (aggr.filter) {
			filter_count++;
			payload_types_filters.push_back(aggr.filter->return_type);
		}
		if (!aggr.function.combine) {
			throw InternalException("Aggregate function %s is missing a combine method", aggr.function.name);
		}
		aggregates.push_back(std::move(expr));
	}
	for (const auto &pay_filters : payload_types_filters) {
		payload_types.push_back(pay_filters);
	}
}

void GroupedAggregateData::InitializeDistinct(const unique_ptr<Expression> &aggregate,
                                              const vector<unique_ptr<Expression>> *groups_p) {
	auto &aggr = aggregate->Cast<BoundAggregateExpression>();
	D_ASSERT(aggr.IsDistinct());

	// Add the (empty in ungrouped case) groups of the aggregates
	InitializeDistinctGroups(groups_p);

	// bindings.push_back(&aggr);
	filter_count = 0;
	aggregate_return_types.push_back(aggr.return_type);
	for (idx_t i = 0; i < aggr.children.size(); i++) {
		auto &child = aggr.children[i];
		group_types.push_back(child->return_type);
		groups.push_back(child->Copy());
		payload_types.push_back(child->return_type);
		if (aggr.filter) {
			filter_count++;
		}
	}
	if (!aggr.function.combine) {
		throw InternalException("Aggregate function %s is missing a combine method", aggr.function.name);
	}
}

void GroupedAggregateData::InitializeDistinctGroups(const vector<unique_ptr<Expression>> *groups_p) {
	if (!groups_p) {
		return;
	}
	for (auto &expr : *groups_p) {
		group_types.push_back(expr->return_type);
		groups.push_back(expr->Copy());
	}
}

void GroupedAggregateData::InitializeGroupbyGroups(vector<unique_ptr<Expression>> groups) {
	// Add all the expressions of the group by clause
	for (auto &expr : groups) {
		group_types.push_back(expr->return_type);
	}
	this->groups = std::move(groups);
}

void GroupedAggregateData::SetGroupingFunctions(vector<unsafe_vector<idx_t>> &functions) {
	grouping_functions.reserve(functions.size());
	for (idx_t i = 0; i < functions.size(); i++) {
		grouping_functions.push_back(std::move(functions[i]));
	}
}

} // namespace duckdb

















namespace duckdb {

HashAggregateGroupingData::HashAggregateGroupingData(GroupingSet &grouping_set_p,
                                                     const GroupedAggregateData &grouped_aggregate_data,
                                                     unique_ptr<DistinctAggregateCollectionInfo> &info)
    : table_data(grouping_set_p, grouped_aggregate_data) {
	if (info) {
		distinct_data = make_uniq<DistinctAggregateData>(*info, grouping_set_p, &grouped_aggregate_data.groups);
	}
}

bool HashAggregateGroupingData::HasDistinct() const {
	return distinct_data != nullptr;
}

HashAggregateGroupingGlobalState::HashAggregateGroupingGlobalState(const HashAggregateGroupingData &data,
                                                                   ClientContext &context) {
	table_state = data.table_data.GetGlobalSinkState(context);
	if (data.HasDistinct()) {
		distinct_state = make_uniq<DistinctAggregateState>(*data.distinct_data, context);
	}
}

HashAggregateGroupingLocalState::HashAggregateGroupingLocalState(const PhysicalHashAggregate &op,
                                                                 const HashAggregateGroupingData &data,
                                                                 ExecutionContext &context) {
	table_state = data.table_data.GetLocalSinkState(context);
	if (!data.HasDistinct()) {
		return;
	}
	auto &distinct_data = *data.distinct_data;

	auto &distinct_indices = op.distinct_collection_info->Indices();
	D_ASSERT(!distinct_indices.empty());

	distinct_states.resize(op.distinct_collection_info->aggregates.size());
	auto &table_map = op.distinct_collection_info->table_map;

	for (auto &idx : distinct_indices) {
		idx_t table_idx = table_map[idx];
		auto &radix_table = distinct_data.radix_tables[table_idx];
		if (radix_table == nullptr) {
			// This aggregate has identical input as another aggregate, so no table is created for it
			continue;
		}
		// Initialize the states of the radix tables used for the distinct aggregates
		distinct_states[table_idx] = radix_table->GetLocalSinkState(context);
	}
}

static vector<LogicalType> CreateGroupChunkTypes(vector<unique_ptr<Expression>> &groups) {
	set<idx_t> group_indices;

	if (groups.empty()) {
		return {};
	}

	for (auto &group : groups) {
		D_ASSERT(group->type == ExpressionType::BOUND_REF);
		auto &bound_ref = group->Cast<BoundReferenceExpression>();
		group_indices.insert(bound_ref.index);
	}
	idx_t highest_index = *group_indices.rbegin();
	vector<LogicalType> types(highest_index + 1, LogicalType::SQLNULL);
	for (auto &group : groups) {
		auto &bound_ref = group->Cast<BoundReferenceExpression>();
		types[bound_ref.index] = bound_ref.return_type;
	}
	return types;
}

bool PhysicalHashAggregate::CanSkipRegularSink() const {
	if (!filter_indexes.empty()) {
		// If we have filters, we can't skip the regular sink, because we might lose groups otherwise.
		return false;
	}
	if (grouped_aggregate_data.aggregates.empty()) {
		// When there are no aggregates, we have to add to the main ht right away
		return false;
	}
	if (!non_distinct_filter.empty()) {
		return false;
	}
	return true;
}

PhysicalHashAggregate::PhysicalHashAggregate(ClientContext &context, vector<LogicalType> types,
                                             vector<unique_ptr<Expression>> expressions, idx_t estimated_cardinality)
    : PhysicalHashAggregate(context, std::move(types), std::move(expressions), {}, estimated_cardinality) {
}

PhysicalHashAggregate::PhysicalHashAggregate(ClientContext &context, vector<LogicalType> types,
                                             vector<unique_ptr<Expression>> expressions,
                                             vector<unique_ptr<Expression>> groups_p, idx_t estimated_cardinality)
    : PhysicalHashAggregate(context, std::move(types), std::move(expressions), std::move(groups_p), {}, {},
                            estimated_cardinality) {
}

PhysicalHashAggregate::PhysicalHashAggregate(ClientContext &context, vector<LogicalType> types,
                                             vector<unique_ptr<Expression>> expressions,
                                             vector<unique_ptr<Expression>> groups_p,
                                             vector<GroupingSet> grouping_sets_p,
                                             vector<unsafe_vector<idx_t>> grouping_functions_p,
                                             idx_t estimated_cardinality)
    : PhysicalOperator(PhysicalOperatorType::HASH_GROUP_BY, std::move(types), estimated_cardinality),
      grouping_sets(std::move(grouping_sets_p)) {
	// get a list of all aggregates to be computed
	const idx_t group_count = groups_p.size();
	if (grouping_sets.empty()) {
		GroupingSet set;
		for (idx_t i = 0; i < group_count; i++) {
			set.insert(i);
		}
		grouping_sets.push_back(std::move(set));
	}
	input_group_types = CreateGroupChunkTypes(groups_p);

	grouped_aggregate_data.InitializeGroupby(std::move(groups_p), std::move(expressions),
	                                         std::move(grouping_functions_p));

	auto &aggregates = grouped_aggregate_data.aggregates;
	// filter_indexes must be pre-built, not lazily instantiated in parallel...
	// Because everything that lives in this class should be read-only at execution time
	idx_t aggregate_input_idx = 0;
	for (idx_t i = 0; i < aggregates.size(); i++) {
		auto &aggregate = aggregates[i];
		auto &aggr = aggregate->Cast<BoundAggregateExpression>();
		aggregate_input_idx += aggr.children.size();
		if (aggr.aggr_type == AggregateType::DISTINCT) {
			distinct_filter.push_back(i);
		} else if (aggr.aggr_type == AggregateType::NON_DISTINCT) {
			non_distinct_filter.push_back(i);
		} else { // LCOV_EXCL_START
			throw NotImplementedException("AggregateType not implemented in PhysicalHashAggregate");
		} // LCOV_EXCL_STOP
	}

	for (idx_t i = 0; i < aggregates.size(); i++) {
		auto &aggregate = aggregates[i];
		auto &aggr = aggregate->Cast<BoundAggregateExpression>();
		if (aggr.filter) {
			auto &bound_ref_expr = aggr.filter->Cast<BoundReferenceExpression>();
			if (!filter_indexes.count(aggr.filter.get())) {
				// Replace the bound reference expression's index with the corresponding index of the payload chunk
				filter_indexes[aggr.filter.get()] = bound_ref_expr.index;
				bound_ref_expr.index = aggregate_input_idx;
			}
			aggregate_input_idx++;
		}
	}

	distinct_collection_info = DistinctAggregateCollectionInfo::Create(grouped_aggregate_data.aggregates);

	for (idx_t i = 0; i < grouping_sets.size(); i++) {
		groupings.emplace_back(grouping_sets[i], grouped_aggregate_data, distinct_collection_info);
	}
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
class HashAggregateGlobalState : public GlobalSinkState {
public:
	HashAggregateGlobalState(const PhysicalHashAggregate &op, ClientContext &context) {
		grouping_states.reserve(op.groupings.size());
		for (idx_t i = 0; i < op.groupings.size(); i++) {
			auto &grouping = op.groupings[i];
			grouping_states.emplace_back(grouping, context);
		}
		vector<LogicalType> filter_types;
		for (auto &aggr : op.grouped_aggregate_data.aggregates) {
			auto &aggregate = aggr->Cast<BoundAggregateExpression>();
			for (auto &child : aggregate.children) {
				payload_types.push_back(child->return_type);
			}
			if (aggregate.filter) {
				filter_types.push_back(aggregate.filter->return_type);
			}
		}
		payload_types.reserve(payload_types.size() + filter_types.size());
		payload_types.insert(payload_types.end(), filter_types.begin(), filter_types.end());
	}

	vector<HashAggregateGroupingGlobalState> grouping_states;
	vector<LogicalType> payload_types;
	//! Whether or not the aggregate is finished
	bool finished = false;
};

class HashAggregateLocalState : public LocalSinkState {
public:
	HashAggregateLocalState(const PhysicalHashAggregate &op, ExecutionContext &context) {

		auto &payload_types = op.grouped_aggregate_data.payload_types;
		if (!payload_types.empty()) {
			aggregate_input_chunk.InitializeEmpty(payload_types);
		}

		grouping_states.reserve(op.groupings.size());
		for (auto &grouping : op.groupings) {
			grouping_states.emplace_back(op, grouping, context);
		}
		// The filter set is only needed here for the distinct aggregates
		// the filtering of data for the regular aggregates is done within the hashtable
		vector<AggregateObject> aggregate_objects;
		for (auto &aggregate : op.grouped_aggregate_data.aggregates) {
			auto &aggr = aggregate->Cast<BoundAggregateExpression>();
			aggregate_objects.emplace_back(&aggr);
		}

		filter_set.Initialize(context.client, aggregate_objects, payload_types);
	}

	DataChunk aggregate_input_chunk;
	vector<HashAggregateGroupingLocalState> grouping_states;
	AggregateFilterDataSet filter_set;
};

void PhysicalHashAggregate::SetMultiScan(GlobalSinkState &state) {
	auto &gstate = state.Cast<HashAggregateGlobalState>();
	for (auto &grouping_state : gstate.grouping_states) {
		auto &radix_state = grouping_state.table_state;
		RadixPartitionedHashTable::SetMultiScan(*radix_state);
		if (!grouping_state.distinct_state) {
			continue;
		}
	}
}

unique_ptr<GlobalSinkState> PhysicalHashAggregate::GetGlobalSinkState(ClientContext &context) const {
	return make_uniq<HashAggregateGlobalState>(*this, context);
}

unique_ptr<LocalSinkState> PhysicalHashAggregate::GetLocalSinkState(ExecutionContext &context) const {
	return make_uniq<HashAggregateLocalState>(*this, context);
}

void PhysicalHashAggregate::SinkDistinctGrouping(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input,
                                                 idx_t grouping_idx) const {
	auto &sink = input.local_state.Cast<HashAggregateLocalState>();
	auto &global_sink = input.global_state.Cast<HashAggregateGlobalState>();

	auto &grouping_gstate = global_sink.grouping_states[grouping_idx];
	auto &grouping_lstate = sink.grouping_states[grouping_idx];
	auto &distinct_info = *distinct_collection_info;

	auto &distinct_state = grouping_gstate.distinct_state;
	auto &distinct_data = groupings[grouping_idx].distinct_data;

	DataChunk empty_chunk;

	// Create an empty filter for Sink, since we don't need to update any aggregate states here
	unsafe_vector<idx_t> empty_filter;

	for (idx_t &idx : distinct_info.indices) {
		auto &aggregate = grouped_aggregate_data.aggregates[idx]->Cast<BoundAggregateExpression>();

		D_ASSERT(distinct_info.table_map.count(idx));
		idx_t table_idx = distinct_info.table_map[idx];
		if (!distinct_data->radix_tables[table_idx]) {
			continue;
		}
		D_ASSERT(distinct_data->radix_tables[table_idx]);
		auto &radix_table = *distinct_data->radix_tables[table_idx];
		auto &radix_global_sink = *distinct_state->radix_states[table_idx];
		auto &radix_local_sink = *grouping_lstate.distinct_states[table_idx];

		InterruptState interrupt_state;
		OperatorSinkInput sink_input {radix_global_sink, radix_local_sink, interrupt_state};

		if (aggregate.filter) {
			DataChunk filter_chunk;
			auto &filtered_data = sink.filter_set.GetFilterData(idx);
			filter_chunk.InitializeEmpty(filtered_data.filtered_payload.GetTypes());

			// Add the filter Vector (BOOL)
			auto it = filter_indexes.find(aggregate.filter.get());
			D_ASSERT(it != filter_indexes.end());
			D_ASSERT(it->second < chunk.data.size());
			auto &filter_bound_ref = aggregate.filter->Cast<BoundReferenceExpression>();
			filter_chunk.data[filter_bound_ref.index].Reference(chunk.data[it->second]);
			filter_chunk.SetCardinality(chunk.size());

			// We cant use the AggregateFilterData::ApplyFilter method, because the chunk we need to
			// apply the filter to also has the groups, and the filtered_data.filtered_payload does not have those.
			SelectionVector sel_vec(STANDARD_VECTOR_SIZE);
			idx_t count = filtered_data.filter_executor.SelectExpression(filter_chunk, sel_vec);

			if (count == 0) {
				continue;
			}

			// Because the 'input' chunk needs to be re-used after this, we need to create
			// a duplicate of it, that we can apply the filter to
			DataChunk filtered_input;
			filtered_input.InitializeEmpty(chunk.GetTypes());

			for (idx_t group_idx = 0; group_idx < grouped_aggregate_data.groups.size(); group_idx++) {
				auto &group = grouped_aggregate_data.groups[group_idx];
				auto &bound_ref = group->Cast<BoundReferenceExpression>();
				filtered_input.data[bound_ref.index].Reference(chunk.data[bound_ref.index]);
			}
			for (idx_t child_idx = 0; child_idx < aggregate.children.size(); child_idx++) {
				auto &child = aggregate.children[child_idx];
				auto &bound_ref = child->Cast<BoundReferenceExpression>();

				filtered_input.data[bound_ref.index].Reference(chunk.data[bound_ref.index]);
			}
			filtered_input.Slice(sel_vec, count);
			filtered_input.SetCardinality(count);

			radix_table.Sink(context, filtered_input, sink_input, empty_chunk, empty_filter);
		} else {
			radix_table.Sink(context, chunk, sink_input, empty_chunk, empty_filter);
		}
	}
}

void PhysicalHashAggregate::SinkDistinct(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const {
	for (idx_t i = 0; i < groupings.size(); i++) {
		SinkDistinctGrouping(context, chunk, input, i);
	}
}

SinkResultType PhysicalHashAggregate::Sink(ExecutionContext &context, DataChunk &chunk,
                                           OperatorSinkInput &input) const {
	auto &llstate = input.local_state.Cast<HashAggregateLocalState>();
	auto &gstate = input.global_state.Cast<HashAggregateGlobalState>();

	if (distinct_collection_info) {
		SinkDistinct(context, chunk, input);
	}

	if (CanSkipRegularSink()) {
		return SinkResultType::NEED_MORE_INPUT;
	}

	DataChunk &aggregate_input_chunk = llstate.aggregate_input_chunk;

	auto &aggregates = grouped_aggregate_data.aggregates;
	idx_t aggregate_input_idx = 0;

	// Populate the aggregate child vectors
	for (auto &aggregate : aggregates) {
		auto &aggr = aggregate->Cast<BoundAggregateExpression>();
		for (auto &child_expr : aggr.children) {
			D_ASSERT(child_expr->type == ExpressionType::BOUND_REF);
			auto &bound_ref_expr = child_expr->Cast<BoundReferenceExpression>();
			D_ASSERT(bound_ref_expr.index < chunk.data.size());
			aggregate_input_chunk.data[aggregate_input_idx++].Reference(chunk.data[bound_ref_expr.index]);
		}
	}
	// Populate the filter vectors
	for (auto &aggregate : aggregates) {
		auto &aggr = aggregate->Cast<BoundAggregateExpression>();
		if (aggr.filter) {
			auto it = filter_indexes.find(aggr.filter.get());
			D_ASSERT(it != filter_indexes.end());
			D_ASSERT(it->second < chunk.data.size());
			aggregate_input_chunk.data[aggregate_input_idx++].Reference(chunk.data[it->second]);
		}
	}

	aggregate_input_chunk.SetCardinality(chunk.size());
	aggregate_input_chunk.Verify();

	// For every grouping set there is one radix_table
	for (idx_t i = 0; i < groupings.size(); i++) {
		auto &grouping_gstate = gstate.grouping_states[i];
		auto &grouping_lstate = llstate.grouping_states[i];
		InterruptState interrupt_state;
		OperatorSinkInput sink_input {*grouping_gstate.table_state, *grouping_lstate.table_state, interrupt_state};

		auto &grouping = groupings[i];
		auto &table = grouping.table_data;
		table.Sink(context, chunk, sink_input, aggregate_input_chunk, non_distinct_filter);
	}

	return SinkResultType::NEED_MORE_INPUT;
}

void PhysicalHashAggregate::CombineDistinct(ExecutionContext &context, GlobalSinkState &state,
                                            LocalSinkState &lstate) const {
	auto &global_sink = state.Cast<HashAggregateGlobalState>();
	auto &sink = lstate.Cast<HashAggregateLocalState>();

	if (!distinct_collection_info) {
		return;
	}
	for (idx_t i = 0; i < groupings.size(); i++) {
		auto &grouping_gstate = global_sink.grouping_states[i];
		auto &grouping_lstate = sink.grouping_states[i];

		auto &distinct_data = groupings[i].distinct_data;
		auto &distinct_state = grouping_gstate.distinct_state;

		const auto table_count = distinct_data->radix_tables.size();
		for (idx_t table_idx = 0; table_idx < table_count; table_idx++) {
			if (!distinct_data->radix_tables[table_idx]) {
				continue;
			}
			auto &radix_table = *distinct_data->radix_tables[table_idx];
			auto &radix_global_sink = *distinct_state->radix_states[table_idx];
			auto &radix_local_sink = *grouping_lstate.distinct_states[table_idx];

			radix_table.Combine(context, radix_global_sink, radix_local_sink);
		}
	}
}

void PhysicalHashAggregate::Combine(ExecutionContext &context, GlobalSinkState &state, LocalSinkState &lstate) const {
	auto &gstate = state.Cast<HashAggregateGlobalState>();
	auto &llstate = lstate.Cast<HashAggregateLocalState>();

	CombineDistinct(context, state, lstate);

	if (CanSkipRegularSink()) {
		return;
	}
	for (idx_t i = 0; i < groupings.size(); i++) {
		auto &grouping_gstate = gstate.grouping_states[i];
		auto &grouping_lstate = llstate.grouping_states[i];

		auto &grouping = groupings[i];
		auto &table = grouping.table_data;
		table.Combine(context, *grouping_gstate.table_state, *grouping_lstate.table_state);
	}
}

//! REGULAR FINALIZE EVENT

class HashAggregateMergeEvent : public BasePipelineEvent {
public:
	HashAggregateMergeEvent(const PhysicalHashAggregate &op_p, HashAggregateGlobalState &gstate_p, Pipeline *pipeline_p)
	    : BasePipelineEvent(*pipeline_p), op(op_p), gstate(gstate_p) {
	}

	const PhysicalHashAggregate &op;
	HashAggregateGlobalState &gstate;

public:
	void Schedule() override {
		vector<shared_ptr<Task>> tasks;
		for (idx_t i = 0; i < op.groupings.size(); i++) {
			auto &grouping_gstate = gstate.grouping_states[i];

			auto &grouping = op.groupings[i];
			auto &table = grouping.table_data;
			table.ScheduleTasks(pipeline->executor, shared_from_this(), *grouping_gstate.table_state, tasks);
		}
		D_ASSERT(!tasks.empty());
		SetTasks(std::move(tasks));
	}
};

//! REGULAR FINALIZE FROM DISTINCT FINALIZE

class HashAggregateFinalizeTask : public ExecutorTask {
public:
	HashAggregateFinalizeTask(Pipeline &pipeline, shared_ptr<Event> event_p, HashAggregateGlobalState &state_p,
	                          ClientContext &context, const PhysicalHashAggregate &op)
	    : ExecutorTask(pipeline.executor), pipeline(pipeline), event(std::move(event_p)), gstate(state_p),
	      context(context), op(op) {
	}

	TaskExecutionResult ExecuteTask(TaskExecutionMode mode) override {
		op.FinalizeInternal(pipeline, *event, context, gstate, false);
		D_ASSERT(!gstate.finished);
		gstate.finished = true;
		event->FinishTask();
		return TaskExecutionResult::TASK_FINISHED;
	}

private:
	Pipeline &pipeline;
	shared_ptr<Event> event;
	HashAggregateGlobalState &gstate;
	ClientContext &context;
	const PhysicalHashAggregate &op;
};

class HashAggregateFinalizeEvent : public BasePipelineEvent {
public:
	HashAggregateFinalizeEvent(const PhysicalHashAggregate &op_p, HashAggregateGlobalState &gstate_p,
	                           Pipeline *pipeline_p, ClientContext &context)
	    : BasePipelineEvent(*pipeline_p), op(op_p), gstate(gstate_p), context(context) {
	}

	const PhysicalHashAggregate &op;
	HashAggregateGlobalState &gstate;
	ClientContext &context;

public:
	void Schedule() override {
		vector<shared_ptr<Task>> tasks;
		tasks.push_back(make_uniq<HashAggregateFinalizeTask>(*pipeline, shared_from_this(), gstate, context, op));
		D_ASSERT(!tasks.empty());
		SetTasks(std::move(tasks));
	}
};

//! DISTINCT FINALIZE TASK

class HashDistinctAggregateFinalizeTask : public ExecutorTask {
public:
	HashDistinctAggregateFinalizeTask(Pipeline &pipeline, shared_ptr<Event> event_p, HashAggregateGlobalState &state_p,
	                                  ClientContext &context, const PhysicalHashAggregate &op,
	                                  vector<vector<unique_ptr<GlobalSourceState>>> &global_sources_p)
	    : ExecutorTask(pipeline.executor), pipeline(pipeline), event(std::move(event_p)), gstate(state_p),
	      context(context), op(op), global_sources(global_sources_p) {
	}

	void AggregateDistinctGrouping(DistinctAggregateCollectionInfo &info,
	                               const HashAggregateGroupingData &grouping_data,
	                               HashAggregateGroupingGlobalState &grouping_state, idx_t grouping_idx) {
		auto &aggregates = info.aggregates;
		auto &data = *grouping_data.distinct_data;
		auto &state = *grouping_state.distinct_state;
		auto &table_state = *grouping_state.table_state;

		ThreadContext temp_thread_context(context);
		ExecutionContext temp_exec_context(context, temp_thread_context, &pipeline);

		auto temp_local_state = grouping_data.table_data.GetLocalSinkState(temp_exec_context);

		// Create a chunk that mimics the 'input' chunk in Sink, for storing the group vectors
		DataChunk group_chunk;
		if (!op.input_group_types.empty()) {
			group_chunk.Initialize(context, op.input_group_types);
		}

		auto &groups = op.grouped_aggregate_data.groups;
		const idx_t group_by_size = groups.size();

		DataChunk aggregate_input_chunk;
		if (!gstate.payload_types.empty()) {
			aggregate_input_chunk.Initialize(context, gstate.payload_types);
		}

		idx_t payload_idx;
		idx_t next_payload_idx = 0;

		for (idx_t i = 0; i < op.grouped_aggregate_data.aggregates.size(); i++) {
			auto &aggregate = aggregates[i]->Cast<BoundAggregateExpression>();

			// Forward the payload idx
			payload_idx = next_payload_idx;
			next_payload_idx = payload_idx + aggregate.children.size();

			// If aggregate is not distinct, skip it
			if (!data.IsDistinct(i)) {
				continue;
			}
			D_ASSERT(data.info.table_map.count(i));
			auto table_idx = data.info.table_map.at(i);
			auto &radix_table_p = data.radix_tables[table_idx];

			// Create a duplicate of the output_chunk, because of multi-threading we cant alter the original
			DataChunk output_chunk;
			output_chunk.Initialize(context, state.distinct_output_chunks[table_idx]->GetTypes());

			auto &global_source = global_sources[grouping_idx][i];
			auto local_source = radix_table_p->GetLocalSourceState(temp_exec_context);

			// Fetch all the data from the aggregate ht, and Sink it into the main ht
			while (true) {
				output_chunk.Reset();
				group_chunk.Reset();
				aggregate_input_chunk.Reset();

				InterruptState interrupt_state;
				OperatorSourceInput source_input {*global_source, *local_source, interrupt_state};
				auto res = radix_table_p->GetData(temp_exec_context, output_chunk, *state.radix_states[table_idx],
				                                  source_input);

				if (res == SourceResultType::FINISHED) {
					D_ASSERT(output_chunk.size() == 0);
					break;
				} else if (res == SourceResultType::BLOCKED) {
					throw InternalException(
					    "Unexpected interrupt from radix table GetData in HashDistinctAggregateFinalizeTask");
				}

				auto &grouped_aggregate_data = *data.grouped_aggregate_data[table_idx];

				for (idx_t group_idx = 0; group_idx < group_by_size; group_idx++) {
					auto &group = grouped_aggregate_data.groups[group_idx];
					auto &bound_ref_expr = group->Cast<BoundReferenceExpression>();
					group_chunk.data[bound_ref_expr.index].Reference(output_chunk.data[group_idx]);
				}
				group_chunk.SetCardinality(output_chunk);

				for (idx_t child_idx = 0; child_idx < grouped_aggregate_data.groups.size() - group_by_size;
				     child_idx++) {
					aggregate_input_chunk.data[payload_idx + child_idx].Reference(
					    output_chunk.data[group_by_size + child_idx]);
				}
				aggregate_input_chunk.SetCardinality(output_chunk);

				// Sink it into the main ht
				OperatorSinkInput sink_input {table_state, *temp_local_state, interrupt_state};
				grouping_data.table_data.Sink(temp_exec_context, group_chunk, sink_input, aggregate_input_chunk, {i});
			}
		}
		grouping_data.table_data.Combine(temp_exec_context, table_state, *temp_local_state);
	}

	TaskExecutionResult ExecuteTask(TaskExecutionMode mode) override {
		D_ASSERT(op.distinct_collection_info);
		auto &info = *op.distinct_collection_info;
		for (idx_t i = 0; i < op.groupings.size(); i++) {
			auto &grouping = op.groupings[i];
			auto &grouping_state = gstate.grouping_states[i];
			AggregateDistinctGrouping(info, grouping, grouping_state, i);
		}
		event->FinishTask();
		return TaskExecutionResult::TASK_FINISHED;
	}

private:
	Pipeline &pipeline;
	shared_ptr<Event> event;
	HashAggregateGlobalState &gstate;
	ClientContext &context;
	const PhysicalHashAggregate &op;
	vector<vector<unique_ptr<GlobalSourceState>>> &global_sources;
};

//! DISTINCT FINALIZE EVENT

// TODO: Create tasks and run these in parallel instead of doing this all in Schedule, single threaded
class HashDistinctAggregateFinalizeEvent : public BasePipelineEvent {
public:
	HashDistinctAggregateFinalizeEvent(const PhysicalHashAggregate &op_p, HashAggregateGlobalState &gstate_p,
	                                   Pipeline &pipeline_p, ClientContext &context)
	    : BasePipelineEvent(pipeline_p), op(op_p), gstate(gstate_p), context(context) {
	}
	const PhysicalHashAggregate &op;
	HashAggregateGlobalState &gstate;
	ClientContext &context;
	//! The GlobalSourceStates for all the radix tables of the distinct aggregates
	vector<vector<unique_ptr<GlobalSourceState>>> global_sources;

public:
	void Schedule() override {
		global_sources = CreateGlobalSources();

		vector<shared_ptr<Task>> tasks;
		auto &scheduler = TaskScheduler::GetScheduler(context);
		auto number_of_threads = scheduler.NumberOfThreads();
		tasks.reserve(number_of_threads);
		for (int32_t i = 0; i < number_of_threads; i++) {
			tasks.push_back(make_uniq<HashDistinctAggregateFinalizeTask>(*pipeline, shared_from_this(), gstate, context,
			                                                             op, global_sources));
		}
		D_ASSERT(!tasks.empty());
		SetTasks(std::move(tasks));
	}

	void FinishEvent() override {
		//! Now that everything is added to the main ht, we can actually finalize
		auto new_event = make_shared<HashAggregateFinalizeEvent>(op, gstate, pipeline.get(), context);
		this->InsertEvent(std::move(new_event));
	}

private:
	vector<vector<unique_ptr<GlobalSourceState>>> CreateGlobalSources() {
		vector<vector<unique_ptr<GlobalSourceState>>> grouping_sources;
		grouping_sources.reserve(op.groupings.size());
		for (idx_t grouping_idx = 0; grouping_idx < op.groupings.size(); grouping_idx++) {
			auto &grouping = op.groupings[grouping_idx];
			auto &data = *grouping.distinct_data;

			vector<unique_ptr<GlobalSourceState>> aggregate_sources;
			aggregate_sources.reserve(op.grouped_aggregate_data.aggregates.size());

			for (idx_t i = 0; i < op.grouped_aggregate_data.aggregates.size(); i++) {
				auto &aggregate = op.grouped_aggregate_data.aggregates[i];
				auto &aggr = aggregate->Cast<BoundAggregateExpression>();

				if (!aggr.IsDistinct()) {
					aggregate_sources.push_back(nullptr);
					continue;
				}

				D_ASSERT(data.info.table_map.count(i));
				auto table_idx = data.info.table_map.at(i);
				auto &radix_table_p = data.radix_tables[table_idx];
				aggregate_sources.push_back(radix_table_p->GetGlobalSourceState(context));
			}
			grouping_sources.push_back(std::move(aggregate_sources));
		}
		return grouping_sources;
	}
};

//! DISTINCT COMBINE EVENT

class HashDistinctCombineFinalizeEvent : public BasePipelineEvent {
public:
	HashDistinctCombineFinalizeEvent(const PhysicalHashAggregate &op_p, HashAggregateGlobalState &gstate_p,
	                                 Pipeline &pipeline_p, ClientContext &client)
	    : BasePipelineEvent(pipeline_p), op(op_p), gstate(gstate_p), client(client) {
	}

	const PhysicalHashAggregate &op;
	HashAggregateGlobalState &gstate;
	ClientContext &client;

public:
	void Schedule() override {
		vector<shared_ptr<Task>> tasks;
		for (idx_t i = 0; i < op.groupings.size(); i++) {
			auto &grouping = op.groupings[i];
			auto &distinct_data = *grouping.distinct_data;
			auto &distinct_state = *gstate.grouping_states[i].distinct_state;
			for (idx_t table_idx = 0; table_idx < distinct_data.radix_tables.size(); table_idx++) {
				if (!distinct_data.radix_tables[table_idx]) {
					continue;
				}
				distinct_data.radix_tables[table_idx]->ScheduleTasks(pipeline->executor, shared_from_this(),
				                                                     *distinct_state.radix_states[table_idx], tasks);
			}
		}

		D_ASSERT(!tasks.empty());
		SetTasks(std::move(tasks));
	}

	void FinishEvent() override {
		//! Now that all tables are combined, it's time to do the distinct aggregations
		auto new_event = make_shared<HashDistinctAggregateFinalizeEvent>(op, gstate, *pipeline, client);
		this->InsertEvent(std::move(new_event));
	}
};

//! FINALIZE

SinkFinalizeType PhysicalHashAggregate::FinalizeDistinct(Pipeline &pipeline, Event &event, ClientContext &context,
                                                         GlobalSinkState &gstate_p) const {
	auto &gstate = gstate_p.Cast<HashAggregateGlobalState>();
	D_ASSERT(distinct_collection_info);

	bool any_partitioned = false;
	for (idx_t i = 0; i < groupings.size(); i++) {
		auto &grouping = groupings[i];
		auto &distinct_data = *grouping.distinct_data;
		auto &distinct_state = *gstate.grouping_states[i].distinct_state;

		for (idx_t table_idx = 0; table_idx < distinct_data.radix_tables.size(); table_idx++) {
			if (!distinct_data.radix_tables[table_idx]) {
				continue;
			}
			auto &radix_table = distinct_data.radix_tables[table_idx];
			auto &radix_state = *distinct_state.radix_states[table_idx];
			bool partitioned = radix_table->Finalize(context, radix_state);
			if (partitioned) {
				any_partitioned = true;
			}
		}
	}
	if (any_partitioned) {
		// If any of the groupings are partitioned then we first need to combine those, then aggregate
		auto new_event = make_shared<HashDistinctCombineFinalizeEvent>(*this, gstate, pipeline, context);
		event.InsertEvent(std::move(new_event));
	} else {
		// Hashtables aren't partitioned, they dont need to be joined first
		// so we can already compute the aggregate
		auto new_event = make_shared<HashDistinctAggregateFinalizeEvent>(*this, gstate, pipeline, context);
		event.InsertEvent(std::move(new_event));
	}
	return SinkFinalizeType::READY;
}

SinkFinalizeType PhysicalHashAggregate::FinalizeInternal(Pipeline &pipeline, Event &event, ClientContext &context,
                                                         GlobalSinkState &gstate_p, bool check_distinct) const {
	auto &gstate = gstate_p.Cast<HashAggregateGlobalState>();

	if (check_distinct && distinct_collection_info) {
		// There are distinct aggregates
		// If these are partitioned those need to be combined first
		// Then we Finalize again, skipping this step
		return FinalizeDistinct(pipeline, event, context, gstate_p);
	}

	bool any_partitioned = false;
	for (idx_t i = 0; i < groupings.size(); i++) {
		auto &grouping = groupings[i];
		auto &grouping_gstate = gstate.grouping_states[i];

		bool is_partitioned = grouping.table_data.Finalize(context, *grouping_gstate.table_state);
		if (is_partitioned) {
			any_partitioned = true;
		}
	}
	if (any_partitioned) {
		auto new_event = make_shared<HashAggregateMergeEvent>(*this, gstate, &pipeline);
		event.InsertEvent(std::move(new_event));
	}
	return SinkFinalizeType::READY;
}

SinkFinalizeType PhysicalHashAggregate::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                                 GlobalSinkState &gstate_p) const {
	return FinalizeInternal(pipeline, event, context, gstate_p, true);
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
class PhysicalHashAggregateGlobalSourceState : public GlobalSourceState {
public:
	PhysicalHashAggregateGlobalSourceState(ClientContext &context, const PhysicalHashAggregate &op)
	    : op(op), state_index(0) {
		for (auto &grouping : op.groupings) {
			auto &rt = grouping.table_data;
			radix_states.push_back(rt.GetGlobalSourceState(context));
		}
	}

	const PhysicalHashAggregate &op;
	mutex lock;
	atomic<idx_t> state_index;

	vector<unique_ptr<GlobalSourceState>> radix_states;

public:
	idx_t MaxThreads() override {
		// If there are no tables, we only need one thread.
		if (op.groupings.empty()) {
			return 1;
		}

		auto &ht_state = op.sink_state->Cast<HashAggregateGlobalState>();
		idx_t count = 0;
		for (size_t sidx = 0; sidx < op.groupings.size(); ++sidx) {
			auto &grouping = op.groupings[sidx];
			auto &grouping_gstate = ht_state.grouping_states[sidx];
			count += grouping.table_data.Size(*grouping_gstate.table_state);
		}
		return MaxValue<idx_t>(1, count / STANDARD_VECTOR_SIZE);
	}
};

unique_ptr<GlobalSourceState> PhysicalHashAggregate::GetGlobalSourceState(ClientContext &context) const {
	return make_uniq<PhysicalHashAggregateGlobalSourceState>(context, *this);
}

class PhysicalHashAggregateLocalSourceState : public LocalSourceState {
public:
	explicit PhysicalHashAggregateLocalSourceState(ExecutionContext &context, const PhysicalHashAggregate &op) {
		for (auto &grouping : op.groupings) {
			auto &rt = grouping.table_data;
			radix_states.push_back(rt.GetLocalSourceState(context));
		}
	}

	vector<unique_ptr<LocalSourceState>> radix_states;
};

unique_ptr<LocalSourceState> PhysicalHashAggregate::GetLocalSourceState(ExecutionContext &context,
                                                                        GlobalSourceState &gstate) const {
	return make_uniq<PhysicalHashAggregateLocalSourceState>(context, *this);
}

SourceResultType PhysicalHashAggregate::GetData(ExecutionContext &context, DataChunk &chunk,
                                                OperatorSourceInput &input) const {
	auto &sink_gstate = sink_state->Cast<HashAggregateGlobalState>();
	auto &gstate = input.global_state.Cast<PhysicalHashAggregateGlobalSourceState>();
	auto &lstate = input.local_state.Cast<PhysicalHashAggregateLocalSourceState>();
	while (true) {
		idx_t radix_idx = gstate.state_index;
		if (radix_idx >= groupings.size()) {
			break;
		}
		auto &grouping = groupings[radix_idx];
		auto &radix_table = grouping.table_data;
		auto &grouping_gstate = sink_gstate.grouping_states[radix_idx];

		InterruptState interrupt_state;
		OperatorSourceInput source_input {*gstate.radix_states[radix_idx], *lstate.radix_states[radix_idx],
		                                  interrupt_state};
		auto res = radix_table.GetData(context, chunk, *grouping_gstate.table_state, source_input);
		if (chunk.size() != 0) {
			return SourceResultType::HAVE_MORE_OUTPUT;
		} else if (res == SourceResultType::BLOCKED) {
			throw InternalException("Unexpectedly Blocked from radix_table");
		}

		// move to the next table
		lock_guard<mutex> l(gstate.lock);
		radix_idx++;
		if (radix_idx > gstate.state_index) {
			// we have not yet worked on the table
			// move the global index forwards
			gstate.state_index = radix_idx;
		}
	}

	return chunk.size() == 0 ? SourceResultType::FINISHED : SourceResultType::HAVE_MORE_OUTPUT;
}

string PhysicalHashAggregate::ParamsToString() const {
	string result;
	auto &groups = grouped_aggregate_data.groups;
	auto &aggregates = grouped_aggregate_data.aggregates;
	for (idx_t i = 0; i < groups.size(); i++) {
		if (i > 0) {
			result += "\n";
		}
		result += groups[i]->GetName();
	}
	for (idx_t i = 0; i < aggregates.size(); i++) {
		auto &aggregate = aggregates[i]->Cast<BoundAggregateExpression>();
		if (i > 0 || !groups.empty()) {
			result += "\n";
		}
		result += aggregates[i]->GetName();
		if (aggregate.filter) {
			result += " Filter: " + aggregate.filter->GetName();
		}
	}
	return result;
}

} // namespace duckdb







namespace duckdb {

PhysicalPerfectHashAggregate::PhysicalPerfectHashAggregate(ClientContext &context, vector<LogicalType> types_p,
                                                           vector<unique_ptr<Expression>> aggregates_p,
                                                           vector<unique_ptr<Expression>> groups_p,
                                                           const vector<unique_ptr<BaseStatistics>> &group_stats,
                                                           vector<idx_t> required_bits_p, idx_t estimated_cardinality)
    : PhysicalOperator(PhysicalOperatorType::PERFECT_HASH_GROUP_BY, std::move(types_p), estimated_cardinality),
      groups(std::move(groups_p)), aggregates(std::move(aggregates_p)), required_bits(std::move(required_bits_p)) {
	D_ASSERT(groups.size() == group_stats.size());
	group_minima.reserve(group_stats.size());
	for (auto &stats : group_stats) {
		D_ASSERT(stats);
		auto &nstats = *stats;
		D_ASSERT(NumericStats::HasMin(nstats));
		group_minima.push_back(NumericStats::Min(nstats));
	}
	for (auto &expr : groups) {
		group_types.push_back(expr->return_type);
	}

	vector<BoundAggregateExpression *> bindings;
	vector<LogicalType> payload_types_filters;
	for (auto &expr : aggregates) {
		D_ASSERT(expr->expression_class == ExpressionClass::BOUND_AGGREGATE);
		D_ASSERT(expr->IsAggregate());
		auto &aggr = expr->Cast<BoundAggregateExpression>();
		bindings.push_back(&aggr);

		D_ASSERT(!aggr.IsDistinct());
		D_ASSERT(aggr.function.combine);
		for (auto &child : aggr.children) {
			payload_types.push_back(child->return_type);
		}
		if (aggr.filter) {
			payload_types_filters.push_back(aggr.filter->return_type);
		}
	}
	for (const auto &pay_filters : payload_types_filters) {
		payload_types.push_back(pay_filters);
	}
	aggregate_objects = AggregateObject::CreateAggregateObjects(bindings);

	// filter_indexes must be pre-built, not lazily instantiated in parallel...
	idx_t aggregate_input_idx = 0;
	for (auto &aggregate : aggregates) {
		auto &aggr = aggregate->Cast<BoundAggregateExpression>();
		aggregate_input_idx += aggr.children.size();
	}
	for (auto &aggregate : aggregates) {
		auto &aggr = aggregate->Cast<BoundAggregateExpression>();
		if (aggr.filter) {
			auto &bound_ref_expr = aggr.filter->Cast<BoundReferenceExpression>();
			auto it = filter_indexes.find(aggr.filter.get());
			if (it == filter_indexes.end()) {
				filter_indexes[aggr.filter.get()] = bound_ref_expr.index;
				bound_ref_expr.index = aggregate_input_idx++;
			} else {
				++aggregate_input_idx;
			}
		}
	}
}

unique_ptr<PerfectAggregateHashTable> PhysicalPerfectHashAggregate::CreateHT(Allocator &allocator,
                                                                             ClientContext &context) const {
	return make_uniq<PerfectAggregateHashTable>(context, allocator, group_types, payload_types, aggregate_objects,
	                                            group_minima, required_bits);
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
class PerfectHashAggregateGlobalState : public GlobalSinkState {
public:
	PerfectHashAggregateGlobalState(const PhysicalPerfectHashAggregate &op, ClientContext &context)
	    : ht(op.CreateHT(Allocator::Get(context), context)) {
	}

	//! The lock for updating the global aggregate state
	mutex lock;
	//! The global aggregate hash table
	unique_ptr<PerfectAggregateHashTable> ht;
};

class PerfectHashAggregateLocalState : public LocalSinkState {
public:
	PerfectHashAggregateLocalState(const PhysicalPerfectHashAggregate &op, ExecutionContext &context)
	    : ht(op.CreateHT(Allocator::Get(context.client), context.client)) {
		group_chunk.InitializeEmpty(op.group_types);
		if (!op.payload_types.empty()) {
			aggregate_input_chunk.InitializeEmpty(op.payload_types);
		}
	}

	//! The local aggregate hash table
	unique_ptr<PerfectAggregateHashTable> ht;
	DataChunk group_chunk;
	DataChunk aggregate_input_chunk;
};

unique_ptr<GlobalSinkState> PhysicalPerfectHashAggregate::GetGlobalSinkState(ClientContext &context) const {
	return make_uniq<PerfectHashAggregateGlobalState>(*this, context);
}

unique_ptr<LocalSinkState> PhysicalPerfectHashAggregate::GetLocalSinkState(ExecutionContext &context) const {
	return make_uniq<PerfectHashAggregateLocalState>(*this, context);
}

SinkResultType PhysicalPerfectHashAggregate::Sink(ExecutionContext &context, DataChunk &chunk,
                                                  OperatorSinkInput &input) const {
	auto &lstate = input.local_state.Cast<PerfectHashAggregateLocalState>();
	DataChunk &group_chunk = lstate.group_chunk;
	DataChunk &aggregate_input_chunk = lstate.aggregate_input_chunk;

	for (idx_t group_idx = 0; group_idx < groups.size(); group_idx++) {
		auto &group = groups[group_idx];
		D_ASSERT(group->type == ExpressionType::BOUND_REF);
		auto &bound_ref_expr = group->Cast<BoundReferenceExpression>();
		group_chunk.data[group_idx].Reference(chunk.data[bound_ref_expr.index]);
	}
	idx_t aggregate_input_idx = 0;
	for (auto &aggregate : aggregates) {
		auto &aggr = aggregate->Cast<BoundAggregateExpression>();
		for (auto &child_expr : aggr.children) {
			D_ASSERT(child_expr->type == ExpressionType::BOUND_REF);
			auto &bound_ref_expr = child_expr->Cast<BoundReferenceExpression>();
			aggregate_input_chunk.data[aggregate_input_idx++].Reference(chunk.data[bound_ref_expr.index]);
		}
	}
	for (auto &aggregate : aggregates) {
		auto &aggr = aggregate->Cast<BoundAggregateExpression>();
		if (aggr.filter) {
			auto it = filter_indexes.find(aggr.filter.get());
			D_ASSERT(it != filter_indexes.end());
			aggregate_input_chunk.data[aggregate_input_idx++].Reference(chunk.data[it->second]);
		}
	}

	group_chunk.SetCardinality(chunk.size());

	aggregate_input_chunk.SetCardinality(chunk.size());

	group_chunk.Verify();
	aggregate_input_chunk.Verify();
	D_ASSERT(aggregate_input_chunk.ColumnCount() == 0 || group_chunk.size() == aggregate_input_chunk.size());

	lstate.ht->AddChunk(group_chunk, aggregate_input_chunk);
	return SinkResultType::NEED_MORE_INPUT;
}

//===--------------------------------------------------------------------===//
// Combine
//===--------------------------------------------------------------------===//
void PhysicalPerfectHashAggregate::Combine(ExecutionContext &context, GlobalSinkState &gstate_p,
                                           LocalSinkState &lstate_p) const {
	auto &lstate = lstate_p.Cast<PerfectHashAggregateLocalState>();
	auto &gstate = gstate_p.Cast<PerfectHashAggregateGlobalState>();

	lock_guard<mutex> l(gstate.lock);
	gstate.ht->Combine(*lstate.ht);
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
class PerfectHashAggregateState : public GlobalSourceState {
public:
	PerfectHashAggregateState() : ht_scan_position(0) {
	}

	//! The current position to scan the HT for output tuples
	idx_t ht_scan_position;
};

unique_ptr<GlobalSourceState> PhysicalPerfectHashAggregate::GetGlobalSourceState(ClientContext &context) const {
	return make_uniq<PerfectHashAggregateState>();
}

SourceResultType PhysicalPerfectHashAggregate::GetData(ExecutionContext &context, DataChunk &chunk,
                                                       OperatorSourceInput &input) const {
	auto &state = (PerfectHashAggregateState &)input.global_state;
	auto &gstate = sink_state->Cast<PerfectHashAggregateGlobalState>();

	gstate.ht->Scan(state.ht_scan_position, chunk);

	if (chunk.size() > 0) {
		return SourceResultType::HAVE_MORE_OUTPUT;
	} else {
		return SourceResultType::FINISHED;
	}
}

string PhysicalPerfectHashAggregate::ParamsToString() const {
	string result;
	for (idx_t i = 0; i < groups.size(); i++) {
		if (i > 0) {
			result += "\n";
		}
		result += groups[i]->GetName();
	}
	for (idx_t i = 0; i < aggregates.size(); i++) {
		if (i > 0 || !groups.empty()) {
			result += "\n";
		}
		result += aggregates[i]->GetName();
		auto &aggregate = aggregates[i]->Cast<BoundAggregateExpression>();
		if (aggregate.filter) {
			result += " Filter: " + aggregate.filter->GetName();
		}
	}
	return result;
}

} // namespace duckdb








namespace duckdb {

PhysicalStreamingWindow::PhysicalStreamingWindow(vector<LogicalType> types, vector<unique_ptr<Expression>> select_list,
                                                 idx_t estimated_cardinality, PhysicalOperatorType type)
    : PhysicalOperator(type, std::move(types), estimated_cardinality), select_list(std::move(select_list)) {
}

class StreamingWindowGlobalState : public GlobalOperatorState {
public:
	StreamingWindowGlobalState() : row_number(1) {
	}

	//! The next row number.
	std::atomic<int64_t> row_number;
};

class StreamingWindowState : public OperatorState {
public:
	using StateBuffer = vector<data_t>;

	StreamingWindowState() : initialized(false), statev(LogicalType::POINTER, (data_ptr_t)&state_ptr) {
	}

	~StreamingWindowState() override {
		for (size_t i = 0; i < aggregate_dtors.size(); ++i) {
			auto dtor = aggregate_dtors[i];
			if (dtor) {
				AggregateInputData aggr_input_data(aggregate_bind_data[i], Allocator::DefaultAllocator());
				state_ptr = aggregate_states[i].data();
				dtor(statev, aggr_input_data, 1);
			}
		}
	}

	void Initialize(ClientContext &context, DataChunk &input, const vector<unique_ptr<Expression>> &expressions) {
		const_vectors.resize(expressions.size());
		aggregate_states.resize(expressions.size());
		aggregate_bind_data.resize(expressions.size(), nullptr);
		aggregate_dtors.resize(expressions.size(), nullptr);

		for (idx_t expr_idx = 0; expr_idx < expressions.size(); expr_idx++) {
			auto &expr = *expressions[expr_idx];
			auto &wexpr = expr.Cast<BoundWindowExpression>();
			switch (expr.GetExpressionType()) {
			case ExpressionType::WINDOW_AGGREGATE: {
				auto &aggregate = *wexpr.aggregate;
				auto &state = aggregate_states[expr_idx];
				aggregate_bind_data[expr_idx] = wexpr.bind_info.get();
				aggregate_dtors[expr_idx] = aggregate.destructor;
				state.resize(aggregate.state_size());
				aggregate.initialize(state.data());
				break;
			}
			case ExpressionType::WINDOW_FIRST_VALUE: {
				// Just execute the expression once
				ExpressionExecutor executor(context);
				executor.AddExpression(*wexpr.children[0]);
				DataChunk result;
				result.Initialize(Allocator::Get(context), {wexpr.children[0]->return_type});
				executor.Execute(input, result);

				const_vectors[expr_idx] = make_uniq<Vector>(result.GetValue(0, 0));
				break;
			}
			case ExpressionType::WINDOW_PERCENT_RANK: {
				const_vectors[expr_idx] = make_uniq<Vector>(Value((double)0));
				break;
			}
			case ExpressionType::WINDOW_RANK:
			case ExpressionType::WINDOW_RANK_DENSE: {
				const_vectors[expr_idx] = make_uniq<Vector>(Value((int64_t)1));
				break;
			}
			default:
				break;
			}
		}
		initialized = true;
	}

public:
	bool initialized;
	vector<unique_ptr<Vector>> const_vectors;

	// Aggregation
	vector<StateBuffer> aggregate_states;
	vector<FunctionData *> aggregate_bind_data;
	vector<aggregate_destructor_t> aggregate_dtors;
	data_ptr_t state_ptr;
	Vector statev;
};

unique_ptr<GlobalOperatorState> PhysicalStreamingWindow::GetGlobalOperatorState(ClientContext &context) const {
	return make_uniq<StreamingWindowGlobalState>();
}

unique_ptr<OperatorState> PhysicalStreamingWindow::GetOperatorState(ExecutionContext &context) const {
	return make_uniq<StreamingWindowState>();
}

OperatorResultType PhysicalStreamingWindow::Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                                    GlobalOperatorState &gstate_p, OperatorState &state_p) const {
	auto &gstate = gstate_p.Cast<StreamingWindowGlobalState>();
	auto &state = (StreamingWindowState &)state_p;
	if (!state.initialized) {
		state.Initialize(context.client, input, select_list);
	}
	// Put payload columns in place
	for (idx_t col_idx = 0; col_idx < input.data.size(); col_idx++) {
		chunk.data[col_idx].Reference(input.data[col_idx]);
	}
	// Compute window function
	const idx_t count = input.size();
	for (idx_t expr_idx = 0; expr_idx < select_list.size(); expr_idx++) {
		idx_t col_idx = input.data.size() + expr_idx;
		auto &expr = *select_list[expr_idx];
		auto &result = chunk.data[col_idx];
		switch (expr.GetExpressionType()) {
		case ExpressionType::WINDOW_AGGREGATE: {
			//	Establish the aggregation environment
			auto &wexpr = expr.Cast<BoundWindowExpression>();
			auto &aggregate = *wexpr.aggregate;
			auto &statev = state.statev;
			state.state_ptr = state.aggregate_states[expr_idx].data();
			AggregateInputData aggr_input_data(wexpr.bind_info.get(), Allocator::DefaultAllocator());

			// Check for COUNT(*)
			if (wexpr.children.empty()) {
				D_ASSERT(GetTypeIdSize(result.GetType().InternalType()) == sizeof(int64_t));
				auto data = FlatVector::GetData<int64_t>(result);
				int64_t start_row = gstate.row_number;
				for (idx_t i = 0; i < input.size(); ++i) {
					data[i] = start_row + i;
				}
				break;
			}

			// Compute the arguments
			auto &allocator = Allocator::Get(context.client);
			ExpressionExecutor executor(context.client);
			vector<LogicalType> payload_types;
			for (auto &child : wexpr.children) {
				payload_types.push_back(child->return_type);
				executor.AddExpression(*child);
			}

			DataChunk payload;
			payload.Initialize(allocator, payload_types);
			executor.Execute(input, payload);

			// Iterate through them using a single SV
			payload.Flatten();
			DataChunk row;
			row.Initialize(allocator, payload_types);
			sel_t s = 0;
			SelectionVector sel(&s);
			row.Slice(sel, 1);
			for (size_t col_idx = 0; col_idx < payload.ColumnCount(); ++col_idx) {
				DictionaryVector::Child(row.data[col_idx]).Reference(payload.data[col_idx]);
			}

			// Update the state and finalize it one row at a time.
			for (idx_t i = 0; i < input.size(); ++i) {
				sel.set_index(0, i);
				aggregate.update(row.data.data(), aggr_input_data, row.ColumnCount(), statev, 1);
				aggregate.finalize(statev, aggr_input_data, result, 1, i);
			}
			break;
		}
		case ExpressionType::WINDOW_FIRST_VALUE:
		case ExpressionType::WINDOW_PERCENT_RANK:
		case ExpressionType::WINDOW_RANK:
		case ExpressionType::WINDOW_RANK_DENSE: {
			// Reference constant vector
			chunk.data[col_idx].Reference(*state.const_vectors[expr_idx]);
			break;
		}
		case ExpressionType::WINDOW_ROW_NUMBER: {
			// Set row numbers
			int64_t start_row = gstate.row_number;
			auto rdata = FlatVector::GetData<int64_t>(chunk.data[col_idx]);
			for (idx_t i = 0; i < count; i++) {
				rdata[i] = start_row + i;
			}
			break;
		}
		default:
			throw NotImplementedException("%s for StreamingWindow", ExpressionTypeToString(expr.GetExpressionType()));
		}
	}
	gstate.row_number += count;
	chunk.SetCardinality(count);
	return OperatorResultType::NEED_MORE_INPUT;
}

string PhysicalStreamingWindow::ParamsToString() const {
	string result;
	for (idx_t i = 0; i < select_list.size(); i++) {
		if (i > 0) {
			result += "\n";
		}
		result += select_list[i]->GetName();
	}
	return result;
}

} // namespace duckdb















#include <functional>


namespace duckdb {

PhysicalUngroupedAggregate::PhysicalUngroupedAggregate(vector<LogicalType> types,
                                                       vector<unique_ptr<Expression>> expressions,
                                                       idx_t estimated_cardinality)
    : PhysicalOperator(PhysicalOperatorType::UNGROUPED_AGGREGATE, std::move(types), estimated_cardinality),
      aggregates(std::move(expressions)) {

	distinct_collection_info = DistinctAggregateCollectionInfo::Create(aggregates);
	if (!distinct_collection_info) {
		return;
	}
	distinct_data = make_uniq<DistinctAggregateData>(*distinct_collection_info);
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
struct AggregateState {
	explicit AggregateState(const vector<unique_ptr<Expression>> &aggregate_expressions) {
		for (auto &aggregate : aggregate_expressions) {
			D_ASSERT(aggregate->GetExpressionClass() == ExpressionClass::BOUND_AGGREGATE);
			auto &aggr = aggregate->Cast<BoundAggregateExpression>();
			auto state = make_unsafe_uniq_array<data_t>(aggr.function.state_size());
			aggr.function.initialize(state.get());
			aggregates.push_back(std::move(state));
			bind_data.push_back(aggr.bind_info.get());
			destructors.push_back(aggr.function.destructor);
#ifdef DEBUG
			counts.push_back(0);
#endif
		}
	}
	~AggregateState() {
		D_ASSERT(destructors.size() == aggregates.size());
		for (idx_t i = 0; i < destructors.size(); i++) {
			if (!destructors[i]) {
				continue;
			}
			Vector state_vector(Value::POINTER((uintptr_t)aggregates[i].get()));
			state_vector.SetVectorType(VectorType::FLAT_VECTOR);

			AggregateInputData aggr_input_data(bind_data[i], Allocator::DefaultAllocator());
			destructors[i](state_vector, aggr_input_data, 1);
		}
	}

	void Move(AggregateState &other) {
		other.aggregates = std::move(aggregates);
		other.destructors = std::move(destructors);
	}

	//! The aggregate values
	vector<unsafe_unique_array<data_t>> aggregates;
	//! The bind data
	vector<FunctionData *> bind_data;
	//! The destructors
	vector<aggregate_destructor_t> destructors;
	//! Counts (used for verification)
	vector<idx_t> counts;
};

class UngroupedAggregateGlobalState : public GlobalSinkState {
public:
	UngroupedAggregateGlobalState(const PhysicalUngroupedAggregate &op, ClientContext &client)
	    : state(op.aggregates), finished(false) {
		if (op.distinct_data) {
			distinct_state = make_uniq<DistinctAggregateState>(*op.distinct_data, client);
		}
	}

	//! The lock for updating the global aggregate state
	mutex lock;
	//! The global aggregate state
	AggregateState state;
	//! Whether or not the aggregate is finished
	bool finished;
	//! The data related to the distinct aggregates (if there are any)
	unique_ptr<DistinctAggregateState> distinct_state;
};

class UngroupedAggregateLocalState : public LocalSinkState {
public:
	UngroupedAggregateLocalState(const PhysicalUngroupedAggregate &op, const vector<LogicalType> &child_types,
	                             GlobalSinkState &gstate_p, ExecutionContext &context)
	    : state(op.aggregates), child_executor(context.client), aggregate_input_chunk(), filter_set() {
		auto &gstate = gstate_p.Cast<UngroupedAggregateGlobalState>();

		auto &allocator = Allocator::Get(context.client);
		InitializeDistinctAggregates(op, gstate, context);

		vector<LogicalType> payload_types;
		vector<AggregateObject> aggregate_objects;
		for (auto &aggregate : op.aggregates) {
			D_ASSERT(aggregate->GetExpressionClass() == ExpressionClass::BOUND_AGGREGATE);
			auto &aggr = aggregate->Cast<BoundAggregateExpression>();
			// initialize the payload chunk
			for (auto &child : aggr.children) {
				payload_types.push_back(child->return_type);
				child_executor.AddExpression(*child);
			}
			aggregate_objects.emplace_back(&aggr);
		}
		if (!payload_types.empty()) { // for select count(*) from t; there is no payload at all
			aggregate_input_chunk.Initialize(allocator, payload_types);
		}
		filter_set.Initialize(context.client, aggregate_objects, child_types);
	}

	//! The local aggregate state
	AggregateState state;
	//! The executor
	ExpressionExecutor child_executor;
	//! The payload chunk, containing all the Vectors for the aggregates
	DataChunk aggregate_input_chunk;
	//! Aggregate filter data set
	AggregateFilterDataSet filter_set;
	//! The local sink states of the distinct aggregates hash tables
	vector<unique_ptr<LocalSinkState>> radix_states;

public:
	void Reset() {
		aggregate_input_chunk.Reset();
	}
	void InitializeDistinctAggregates(const PhysicalUngroupedAggregate &op, const UngroupedAggregateGlobalState &gstate,
	                                  ExecutionContext &context) {

		if (!op.distinct_data) {
			return;
		}
		auto &data = *op.distinct_data;
		auto &state = *gstate.distinct_state;
		D_ASSERT(!data.radix_tables.empty());

		const idx_t aggregate_count = state.radix_states.size();
		radix_states.resize(aggregate_count);

		auto &distinct_info = *op.distinct_collection_info;

		for (auto &idx : distinct_info.indices) {
			idx_t table_idx = distinct_info.table_map[idx];
			if (data.radix_tables[table_idx] == nullptr) {
				// This aggregate has identical input as another aggregate, so no table is created for it
				continue;
			}
			auto &radix_table = *data.radix_tables[table_idx];
			radix_states[table_idx] = radix_table.GetLocalSinkState(context);
		}
	}
};

bool PhysicalUngroupedAggregate::SinkOrderDependent() const {
	for (auto &expr : aggregates) {
		auto &aggr = expr->Cast<BoundAggregateExpression>();
		if (aggr.function.order_dependent == AggregateOrderDependent::ORDER_DEPENDENT) {
			return true;
		}
	}
	return false;
}

unique_ptr<GlobalSinkState> PhysicalUngroupedAggregate::GetGlobalSinkState(ClientContext &context) const {
	return make_uniq<UngroupedAggregateGlobalState>(*this, context);
}

unique_ptr<LocalSinkState> PhysicalUngroupedAggregate::GetLocalSinkState(ExecutionContext &context) const {
	D_ASSERT(sink_state);
	auto &gstate = *sink_state;
	return make_uniq<UngroupedAggregateLocalState>(*this, children[0]->GetTypes(), gstate, context);
}

void PhysicalUngroupedAggregate::SinkDistinct(ExecutionContext &context, DataChunk &chunk,
                                              OperatorSinkInput &input) const {
	auto &sink = input.local_state.Cast<UngroupedAggregateLocalState>();
	auto &global_sink = input.global_state.Cast<UngroupedAggregateGlobalState>();
	D_ASSERT(distinct_data);
	auto &distinct_state = *global_sink.distinct_state;
	auto &distinct_info = *distinct_collection_info;
	auto &distinct_indices = distinct_info.Indices();

	DataChunk empty_chunk;

	auto &distinct_filter = distinct_info.Indices();

	for (auto &idx : distinct_indices) {
		auto &aggregate = aggregates[idx]->Cast<BoundAggregateExpression>();

		idx_t table_idx = distinct_info.table_map[idx];
		if (!distinct_data->radix_tables[table_idx]) {
			// This distinct aggregate shares its data with another
			continue;
		}
		D_ASSERT(distinct_data->radix_tables[table_idx]);
		auto &radix_table = *distinct_data->radix_tables[table_idx];
		auto &radix_global_sink = *distinct_state.radix_states[table_idx];
		auto &radix_local_sink = *sink.radix_states[table_idx];
		OperatorSinkInput sink_input {radix_global_sink, radix_local_sink, input.interrupt_state};

		if (aggregate.filter) {
			// The hashtable can apply a filter, but only on the payload
			// And in our case, we need to filter the groups (the distinct aggr children)

			// Apply the filter before inserting into the hashtable
			auto &filtered_data = sink.filter_set.GetFilterData(idx);
			idx_t count = filtered_data.ApplyFilter(chunk);
			filtered_data.filtered_payload.SetCardinality(count);

			radix_table.Sink(context, filtered_data.filtered_payload, sink_input, empty_chunk, distinct_filter);
		} else {
			radix_table.Sink(context, chunk, sink_input, empty_chunk, distinct_filter);
		}
	}
}

SinkResultType PhysicalUngroupedAggregate::Sink(ExecutionContext &context, DataChunk &chunk,
                                                OperatorSinkInput &input) const {
	auto &sink = input.local_state.Cast<UngroupedAggregateLocalState>();

	// perform the aggregation inside the local state
	sink.Reset();

	if (distinct_data) {
		SinkDistinct(context, chunk, input);
	}

	DataChunk &payload_chunk = sink.aggregate_input_chunk;

	idx_t payload_idx = 0;
	idx_t next_payload_idx = 0;

	for (idx_t aggr_idx = 0; aggr_idx < aggregates.size(); aggr_idx++) {
		auto &aggregate = aggregates[aggr_idx]->Cast<BoundAggregateExpression>();

		payload_idx = next_payload_idx;
		next_payload_idx = payload_idx + aggregate.children.size();

		if (aggregate.IsDistinct()) {
			continue;
		}

		idx_t payload_cnt = 0;
		// resolve the filter (if any)
		if (aggregate.filter) {
			auto &filtered_data = sink.filter_set.GetFilterData(aggr_idx);
			auto count = filtered_data.ApplyFilter(chunk);

			sink.child_executor.SetChunk(filtered_data.filtered_payload);
			payload_chunk.SetCardinality(count);
		} else {
			sink.child_executor.SetChunk(chunk);
			payload_chunk.SetCardinality(chunk);
		}

#ifdef DEBUG
		sink.state.counts[aggr_idx] += payload_chunk.size();
#endif

		// resolve the child expressions of the aggregate (if any)
		for (idx_t i = 0; i < aggregate.children.size(); ++i) {
			sink.child_executor.ExecuteExpression(payload_idx + payload_cnt,
			                                      payload_chunk.data[payload_idx + payload_cnt]);
			payload_cnt++;
		}

		auto start_of_input = payload_cnt == 0 ? nullptr : &payload_chunk.data[payload_idx];
		AggregateInputData aggr_input_data(aggregate.bind_info.get(), Allocator::DefaultAllocator());
		aggregate.function.simple_update(start_of_input, aggr_input_data, payload_cnt,
		                                 sink.state.aggregates[aggr_idx].get(), payload_chunk.size());
	}
	return SinkResultType::NEED_MORE_INPUT;
}

//===--------------------------------------------------------------------===//
// Finalize
//===--------------------------------------------------------------------===//

void PhysicalUngroupedAggregate::CombineDistinct(ExecutionContext &context, GlobalSinkState &state,
                                                 LocalSinkState &lstate) const {
	auto &global_sink = state.Cast<UngroupedAggregateGlobalState>();
	auto &source = lstate.Cast<UngroupedAggregateLocalState>();

	if (!distinct_data) {
		return;
	}
	auto &distinct_state = global_sink.distinct_state;
	auto table_count = distinct_data->radix_tables.size();
	for (idx_t table_idx = 0; table_idx < table_count; table_idx++) {
		D_ASSERT(distinct_data->radix_tables[table_idx]);
		auto &radix_table = *distinct_data->radix_tables[table_idx];
		auto &radix_global_sink = *distinct_state->radix_states[table_idx];
		auto &radix_local_sink = *source.radix_states[table_idx];

		radix_table.Combine(context, radix_global_sink, radix_local_sink);
	}
}

void PhysicalUngroupedAggregate::Combine(ExecutionContext &context, GlobalSinkState &state,
                                         LocalSinkState &lstate) const {
	auto &gstate = state.Cast<UngroupedAggregateGlobalState>();
	auto &source = lstate.Cast<UngroupedAggregateLocalState>();
	D_ASSERT(!gstate.finished);

	// finalize: combine the local state into the global state
	// all aggregates are combinable: we might be doing a parallel aggregate
	// use the combine method to combine the partial aggregates
	lock_guard<mutex> glock(gstate.lock);

	CombineDistinct(context, state, lstate);

	for (idx_t aggr_idx = 0; aggr_idx < aggregates.size(); aggr_idx++) {
		auto &aggregate = aggregates[aggr_idx]->Cast<BoundAggregateExpression>();

		if (aggregate.IsDistinct()) {
			continue;
		}

		Vector source_state(Value::POINTER((uintptr_t)source.state.aggregates[aggr_idx].get()));
		Vector dest_state(Value::POINTER((uintptr_t)gstate.state.aggregates[aggr_idx].get()));

		AggregateInputData aggr_input_data(aggregate.bind_info.get(), Allocator::DefaultAllocator());
		aggregate.function.combine(source_state, dest_state, aggr_input_data, 1);
#ifdef DEBUG
		gstate.state.counts[aggr_idx] += source.state.counts[aggr_idx];
#endif
	}

	auto &client_profiler = QueryProfiler::Get(context.client);
	context.thread.profiler.Flush(*this, source.child_executor, "child_executor", 0);
	client_profiler.Flush(context.thread.profiler);
}

class UngroupedDistinctAggregateFinalizeTask : public ExecutorTask {
public:
	UngroupedDistinctAggregateFinalizeTask(Executor &executor, shared_ptr<Event> event_p,
	                                       UngroupedAggregateGlobalState &state_p, ClientContext &context,
	                                       const PhysicalUngroupedAggregate &op)
	    : ExecutorTask(executor), event(std::move(event_p)), gstate(state_p), context(context), op(op) {
	}

	void AggregateDistinct() {
		D_ASSERT(gstate.distinct_state);
		auto &aggregates = op.aggregates;
		auto &distinct_state = *gstate.distinct_state;
		auto &distinct_data = *op.distinct_data;

		ThreadContext temp_thread_context(context);
		ExecutionContext temp_exec_context(context, temp_thread_context, nullptr);

		idx_t payload_idx = 0;
		idx_t next_payload_idx = 0;

		for (idx_t i = 0; i < aggregates.size(); i++) {
			auto &aggregate = aggregates[i]->Cast<BoundAggregateExpression>();

			// Forward the payload idx
			payload_idx = next_payload_idx;
			next_payload_idx = payload_idx + aggregate.children.size();

			// If aggregate is not distinct, skip it
			if (!distinct_data.IsDistinct(i)) {
				continue;
			}

			DataChunk payload_chunk;

			D_ASSERT(distinct_data.info.table_map.count(i));
			auto table_idx = distinct_data.info.table_map.at(i);
			auto &radix_table_p = distinct_data.radix_tables[table_idx];
			auto &output_chunk = *distinct_state.distinct_output_chunks[table_idx];
			auto &grouped_aggregate_data = *distinct_data.grouped_aggregate_data[table_idx];

			payload_chunk.InitializeEmpty(grouped_aggregate_data.group_types);
			payload_chunk.SetCardinality(0);

			//! Create global and local state for the hashtable
			auto global_source_state = radix_table_p->GetGlobalSourceState(context);
			auto local_source_state = radix_table_p->GetLocalSourceState(temp_exec_context);

			//! Retrieve the stored data from the hashtable
			while (true) {
				output_chunk.Reset();

				InterruptState interrupt_state;
				OperatorSourceInput source_input {*global_source_state, *local_source_state, interrupt_state};
				auto res = radix_table_p->GetData(temp_exec_context, output_chunk,
				                                  *distinct_state.radix_states[table_idx], source_input);
				if (res == SourceResultType::FINISHED) {
					D_ASSERT(output_chunk.size() == 0);
					break;
				} else if (res == SourceResultType::BLOCKED) {
					throw InternalException(
					    "Unexpected interrupt from radix table GetData in UngroupedDistinctAggregateFinalizeTask");
				}

				// We dont need to resolve the filter, we already did this in Sink
				idx_t payload_cnt = aggregate.children.size();
				for (idx_t i = 0; i < payload_cnt; i++) {
					payload_chunk.data[i].Reference(output_chunk.data[i]);
				}
				payload_chunk.SetCardinality(output_chunk);
#ifdef DEBUG
				gstate.state.counts[i] += payload_chunk.size();
#endif

				auto start_of_input = payload_cnt ? &payload_chunk.data[0] : nullptr;
				//! Update the aggregate state
				AggregateInputData aggr_input_data(aggregate.bind_info.get(), Allocator::DefaultAllocator());
				aggregate.function.simple_update(start_of_input, aggr_input_data, payload_cnt,
				                                 gstate.state.aggregates[i].get(), payload_chunk.size());
			}
		}
		D_ASSERT(!gstate.finished);
		gstate.finished = true;
	}

	TaskExecutionResult ExecuteTask(TaskExecutionMode mode) override {
		AggregateDistinct();
		event->FinishTask();
		return TaskExecutionResult::TASK_FINISHED;
	}

private:
	shared_ptr<Event> event;
	UngroupedAggregateGlobalState &gstate;
	ClientContext &context;
	const PhysicalUngroupedAggregate &op;
};

// TODO: Create tasks and run these in parallel instead of doing this all in Schedule, single threaded
class UngroupedDistinctAggregateFinalizeEvent : public BasePipelineEvent {
public:
	UngroupedDistinctAggregateFinalizeEvent(const PhysicalUngroupedAggregate &op_p,
	                                        UngroupedAggregateGlobalState &gstate_p, Pipeline &pipeline_p,
	                                        ClientContext &context)
	    : BasePipelineEvent(pipeline_p), op(op_p), gstate(gstate_p), context(context) {
	}
	const PhysicalUngroupedAggregate &op;
	UngroupedAggregateGlobalState &gstate;
	ClientContext &context;

public:
	void Schedule() override {
		vector<shared_ptr<Task>> tasks;
		tasks.push_back(make_uniq<UngroupedDistinctAggregateFinalizeTask>(pipeline->executor, shared_from_this(),
		                                                                  gstate, context, op));
		D_ASSERT(!tasks.empty());
		SetTasks(std::move(tasks));
	}
};

class UngroupedDistinctCombineFinalizeEvent : public BasePipelineEvent {
public:
	UngroupedDistinctCombineFinalizeEvent(const PhysicalUngroupedAggregate &op_p,
	                                      UngroupedAggregateGlobalState &gstate_p, Pipeline &pipeline_p,
	                                      ClientContext &client)
	    : BasePipelineEvent(pipeline_p), op(op_p), gstate(gstate_p), client(client) {
	}

	const PhysicalUngroupedAggregate &op;
	UngroupedAggregateGlobalState &gstate;
	ClientContext &client;

public:
	void Schedule() override {
		auto &distinct_state = *gstate.distinct_state;
		auto &distinct_data = *op.distinct_data;
		vector<shared_ptr<Task>> tasks;
		for (idx_t table_idx = 0; table_idx < distinct_data.radix_tables.size(); table_idx++) {
			distinct_data.radix_tables[table_idx]->ScheduleTasks(pipeline->executor, shared_from_this(),
			                                                     *distinct_state.radix_states[table_idx], tasks);
		}
		D_ASSERT(!tasks.empty());
		SetTasks(std::move(tasks));
	}

	void FinishEvent() override {
		//! Now that all tables are combined, it's time to do the distinct aggregations
		auto new_event = make_shared<UngroupedDistinctAggregateFinalizeEvent>(op, gstate, *pipeline, client);
		this->InsertEvent(std::move(new_event));
	}
};

SinkFinalizeType PhysicalUngroupedAggregate::FinalizeDistinct(Pipeline &pipeline, Event &event, ClientContext &context,
                                                              GlobalSinkState &gstate_p) const {
	auto &gstate = gstate_p.Cast<UngroupedAggregateGlobalState>();
	D_ASSERT(distinct_data);
	auto &distinct_state = *gstate.distinct_state;

	bool any_partitioned = false;
	for (idx_t table_idx = 0; table_idx < distinct_data->radix_tables.size(); table_idx++) {
		auto &radix_table_p = distinct_data->radix_tables[table_idx];
		auto &radix_state = *distinct_state.radix_states[table_idx];
		bool partitioned = radix_table_p->Finalize(context, radix_state);
		if (partitioned) {
			any_partitioned = true;
		}
	}
	if (any_partitioned) {
		auto new_event = make_shared<UngroupedDistinctCombineFinalizeEvent>(*this, gstate, pipeline, context);
		event.InsertEvent(std::move(new_event));
	} else {
		//! Hashtables aren't partitioned, they dont need to be joined first
		//! So we can compute the aggregate already
		auto new_event = make_shared<UngroupedDistinctAggregateFinalizeEvent>(*this, gstate, pipeline, context);
		event.InsertEvent(std::move(new_event));
	}
	return SinkFinalizeType::READY;
}

SinkFinalizeType PhysicalUngroupedAggregate::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                                      GlobalSinkState &gstate_p) const {
	auto &gstate = gstate_p.Cast<UngroupedAggregateGlobalState>();

	if (distinct_data) {
		return FinalizeDistinct(pipeline, event, context, gstate_p);
	}

	D_ASSERT(!gstate.finished);
	gstate.finished = true;
	return SinkFinalizeType::READY;
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
void VerifyNullHandling(DataChunk &chunk, AggregateState &state, const vector<unique_ptr<Expression>> &aggregates) {
#ifdef DEBUG
	for (idx_t aggr_idx = 0; aggr_idx < aggregates.size(); aggr_idx++) {
		auto &aggr = aggregates[aggr_idx]->Cast<BoundAggregateExpression>();
		if (state.counts[aggr_idx] == 0 && aggr.function.null_handling == FunctionNullHandling::DEFAULT_NULL_HANDLING) {
			// Default is when 0 values go in, NULL comes out
			UnifiedVectorFormat vdata;
			chunk.data[aggr_idx].ToUnifiedFormat(1, vdata);
			D_ASSERT(!vdata.validity.RowIsValid(vdata.sel->get_index(0)));
		}
	}
#endif
}

SourceResultType PhysicalUngroupedAggregate::GetData(ExecutionContext &context, DataChunk &chunk,
                                                     OperatorSourceInput &input) const {
	auto &gstate = sink_state->Cast<UngroupedAggregateGlobalState>();
	D_ASSERT(gstate.finished);

	// initialize the result chunk with the aggregate values
	chunk.SetCardinality(1);
	for (idx_t aggr_idx = 0; aggr_idx < aggregates.size(); aggr_idx++) {
		auto &aggregate = aggregates[aggr_idx]->Cast<BoundAggregateExpression>();

		Vector state_vector(Value::POINTER((uintptr_t)gstate.state.aggregates[aggr_idx].get()));
		AggregateInputData aggr_input_data(aggregate.bind_info.get(), Allocator::DefaultAllocator());
		aggregate.function.finalize(state_vector, aggr_input_data, chunk.data[aggr_idx], 1, 0);
	}
	VerifyNullHandling(chunk, gstate.state, aggregates);

	return SourceResultType::FINISHED;
}

string PhysicalUngroupedAggregate::ParamsToString() const {
	string result;
	for (idx_t i = 0; i < aggregates.size(); i++) {
		auto &aggregate = aggregates[i]->Cast<BoundAggregateExpression>();
		if (i > 0) {
			result += "\n";
		}
		result += aggregates[i]->GetName();
		if (aggregate.filter) {
			result += " Filter: " + aggregate.filter->GetName();
		}
	}
	return result;
}

} // namespace duckdb






















#include <algorithm>
#include <cmath>
#include <numeric>

namespace duckdb {

//	Global sink state
class WindowGlobalSinkState : public GlobalSinkState {
public:
	WindowGlobalSinkState(const PhysicalWindow &op, ClientContext &context)
	    : mode(DBConfig::GetConfig(context).options.window_mode) {

		D_ASSERT(op.select_list[0]->GetExpressionClass() == ExpressionClass::BOUND_WINDOW);
		auto &wexpr = op.select_list[0]->Cast<BoundWindowExpression>();

		global_partition =
		    make_uniq<PartitionGlobalSinkState>(context, wexpr.partitions, wexpr.orders, op.children[0]->types,
		                                        wexpr.partitions_stats, op.estimated_cardinality);
	}

	unique_ptr<PartitionGlobalSinkState> global_partition;
	WindowAggregationMode mode;
};

//	Per-thread sink state
class WindowLocalSinkState : public LocalSinkState {
public:
	WindowLocalSinkState(ClientContext &context, const WindowGlobalSinkState &gstate)
	    : local_partition(context, *gstate.global_partition) {
	}

	void Sink(DataChunk &input_chunk) {
		local_partition.Sink(input_chunk);
	}

	void Combine() {
		local_partition.Combine();
	}

	PartitionLocalSinkState local_partition;
};

// this implements a sorted window functions variant
PhysicalWindow::PhysicalWindow(vector<LogicalType> types, vector<unique_ptr<Expression>> select_list_p,
                               idx_t estimated_cardinality, PhysicalOperatorType type)
    : PhysicalOperator(type, std::move(types), estimated_cardinality), select_list(std::move(select_list_p)) {
	is_order_dependent = false;
	for (auto &expr : select_list) {
		D_ASSERT(expr->expression_class == ExpressionClass::BOUND_WINDOW);
		auto &bound_window = expr->Cast<BoundWindowExpression>();
		if (bound_window.partitions.empty() && bound_window.orders.empty()) {
			is_order_dependent = true;
		}
	}
}

static idx_t FindNextStart(const ValidityMask &mask, idx_t l, const idx_t r, idx_t &n) {
	if (mask.AllValid()) {
		auto start = MinValue(l + n - 1, r);
		n -= MinValue(n, r - l);
		return start;
	}

	while (l < r) {
		//	If l is aligned with the start of a block, and the block is blank, then skip forward one block.
		idx_t entry_idx;
		idx_t shift;
		mask.GetEntryIndex(l, entry_idx, shift);

		const auto block = mask.GetValidityEntry(entry_idx);
		if (mask.NoneValid(block) && !shift) {
			l += ValidityMask::BITS_PER_VALUE;
			continue;
		}

		// Loop over the block
		for (; shift < ValidityMask::BITS_PER_VALUE && l < r; ++shift, ++l) {
			if (mask.RowIsValid(block, shift) && --n == 0) {
				return MinValue(l, r);
			}
		}
	}

	//	Didn't find a start so return the end of the range
	return r;
}

static idx_t FindPrevStart(const ValidityMask &mask, const idx_t l, idx_t r, idx_t &n) {
	if (mask.AllValid()) {
		auto start = (r <= l + n) ? l : r - n;
		n -= r - start;
		return start;
	}

	while (l < r) {
		// If r is aligned with the start of a block, and the previous block is blank,
		// then skip backwards one block.
		idx_t entry_idx;
		idx_t shift;
		mask.GetEntryIndex(r - 1, entry_idx, shift);

		const auto block = mask.GetValidityEntry(entry_idx);
		if (mask.NoneValid(block) && (shift + 1 == ValidityMask::BITS_PER_VALUE)) {
			// r is nonzero (> l) and word aligned, so this will not underflow.
			r -= ValidityMask::BITS_PER_VALUE;
			continue;
		}

		// Loop backwards over the block
		// shift is probing r-1 >= l >= 0
		for (++shift; shift-- > 0; --r) {
			if (mask.RowIsValid(block, shift) && --n == 0) {
				return MaxValue(l, r - 1);
			}
		}
	}

	//	Didn't find a start so return the start of the range
	return l;
}

static void PrepareInputExpressions(vector<unique_ptr<Expression>> &exprs, ExpressionExecutor &executor,
                                    DataChunk &chunk) {
	if (exprs.empty()) {
		return;
	}

	vector<LogicalType> types;
	for (idx_t expr_idx = 0; expr_idx < exprs.size(); ++expr_idx) {
		types.push_back(exprs[expr_idx]->return_type);
		executor.AddExpression(*exprs[expr_idx]);
	}

	if (!types.empty()) {
		auto &allocator = executor.GetAllocator();
		chunk.Initialize(allocator, types);
	}
}

static void PrepareInputExpression(Expression &expr, ExpressionExecutor &executor, DataChunk &chunk) {
	vector<LogicalType> types;
	types.push_back(expr.return_type);
	executor.AddExpression(expr);

	auto &allocator = executor.GetAllocator();
	chunk.Initialize(allocator, types);
}

struct WindowInputExpression {
	WindowInputExpression(optional_ptr<Expression> expr_p, ClientContext &context)
	    : expr(expr_p), ptype(PhysicalType::INVALID), scalar(true), executor(context) {
		if (expr) {
			PrepareInputExpression(*expr, executor, chunk);
			ptype = expr->return_type.InternalType();
			scalar = expr->IsScalar();
		}
	}

	void Execute(DataChunk &input_chunk) {
		if (expr) {
			chunk.Reset();
			executor.Execute(input_chunk, chunk);
			chunk.Verify();
		}
	}

	template <typename T>
	inline T GetCell(idx_t i) const {
		D_ASSERT(!chunk.data.empty());
		const auto data = FlatVector::GetData<T>(chunk.data[0]);
		return data[scalar ? 0 : i];
	}

	inline bool CellIsNull(idx_t i) const {
		D_ASSERT(!chunk.data.empty());
		if (chunk.data[0].GetVectorType() == VectorType::CONSTANT_VECTOR) {
			return ConstantVector::IsNull(chunk.data[0]);
		}
		return FlatVector::IsNull(chunk.data[0], i);
	}

	inline void CopyCell(Vector &target, idx_t target_offset) const {
		D_ASSERT(!chunk.data.empty());
		auto &source = chunk.data[0];
		auto source_offset = scalar ? 0 : target_offset;
		VectorOperations::Copy(source, target, source_offset + 1, source_offset, target_offset);
	}

	optional_ptr<Expression> expr;
	PhysicalType ptype;
	bool scalar;
	ExpressionExecutor executor;
	DataChunk chunk;
};

struct WindowInputColumn {
	WindowInputColumn(Expression *expr_p, ClientContext &context, idx_t capacity_p)
	    : input_expr(expr_p, context), count(0), capacity(capacity_p) {
		if (input_expr.expr) {
			target = make_uniq<Vector>(input_expr.chunk.data[0].GetType(), capacity);
		}
	}

	void Append(DataChunk &input_chunk) {
		if (input_expr.expr && (!input_expr.scalar || !count)) {
			input_expr.Execute(input_chunk);
			auto &source = input_expr.chunk.data[0];
			const auto source_count = input_expr.chunk.size();
			D_ASSERT(count + source_count <= capacity);
			VectorOperations::Copy(source, *target, source_count, 0, count);
			count += source_count;
		}
	}

	inline bool CellIsNull(idx_t i) {
		D_ASSERT(target);
		D_ASSERT(i < count);
		return FlatVector::IsNull(*target, input_expr.scalar ? 0 : i);
	}

	template <typename T>
	inline T GetCell(idx_t i) const {
		D_ASSERT(target);
		D_ASSERT(i < count);
		const auto data = FlatVector::GetData<T>(*target);
		return data[input_expr.scalar ? 0 : i];
	}

	WindowInputExpression input_expr;

private:
	unique_ptr<Vector> target;
	idx_t count;
	idx_t capacity;
};

static inline bool BoundaryNeedsPeer(const WindowBoundary &boundary) {
	switch (boundary) {
	case WindowBoundary::CURRENT_ROW_RANGE:
	case WindowBoundary::EXPR_PRECEDING_RANGE:
	case WindowBoundary::EXPR_FOLLOWING_RANGE:
		return true;
	default:
		return false;
	}
}

struct WindowBoundariesState {
	static inline bool IsScalar(const unique_ptr<Expression> &expr) {
		return expr ? expr->IsScalar() : true;
	}

	WindowBoundariesState(BoundWindowExpression &wexpr, const idx_t input_size)
	    : type(wexpr.type), input_size(input_size), start_boundary(wexpr.start), end_boundary(wexpr.end),
	      partition_count(wexpr.partitions.size()), order_count(wexpr.orders.size()),
	      range_sense(wexpr.orders.empty() ? OrderType::INVALID : wexpr.orders[0].type),
	      has_preceding_range(wexpr.start == WindowBoundary::EXPR_PRECEDING_RANGE ||
	                          wexpr.end == WindowBoundary::EXPR_PRECEDING_RANGE),
	      has_following_range(wexpr.start == WindowBoundary::EXPR_FOLLOWING_RANGE ||
	                          wexpr.end == WindowBoundary::EXPR_FOLLOWING_RANGE),
	      needs_peer(BoundaryNeedsPeer(wexpr.end) || wexpr.type == ExpressionType::WINDOW_CUME_DIST) {
	}

	void Update(const idx_t row_idx, WindowInputColumn &range_collection, const idx_t source_offset,
	            WindowInputExpression &boundary_start, WindowInputExpression &boundary_end,
	            const ValidityMask &partition_mask, const ValidityMask &order_mask);

	// Cached lookups
	const ExpressionType type;
	const idx_t input_size;
	const WindowBoundary start_boundary;
	const WindowBoundary end_boundary;
	const size_t partition_count;
	const size_t order_count;
	const OrderType range_sense;
	const bool has_preceding_range;
	const bool has_following_range;
	const bool needs_peer;

	idx_t partition_start = 0;
	idx_t partition_end = 0;
	idx_t peer_start = 0;
	idx_t peer_end = 0;
	idx_t valid_start = 0;
	idx_t valid_end = 0;
	int64_t window_start = -1;
	int64_t window_end = -1;
	bool is_same_partition = false;
	bool is_peer = false;
};

static bool WindowNeedsRank(const BoundWindowExpression &wexpr) {
	return wexpr.type == ExpressionType::WINDOW_PERCENT_RANK || wexpr.type == ExpressionType::WINDOW_RANK ||
	       wexpr.type == ExpressionType::WINDOW_RANK_DENSE || wexpr.type == ExpressionType::WINDOW_CUME_DIST;
}

template <typename T>
static T GetCell(DataChunk &chunk, idx_t column, idx_t index) {
	D_ASSERT(chunk.ColumnCount() > column);
	auto &source = chunk.data[column];
	const auto data = FlatVector::GetData<T>(source);
	return data[index];
}

static bool CellIsNull(DataChunk &chunk, idx_t column, idx_t index) {
	D_ASSERT(chunk.ColumnCount() > column);
	auto &source = chunk.data[column];
	return FlatVector::IsNull(source, index);
}

static void CopyCell(DataChunk &chunk, idx_t column, idx_t index, Vector &target, idx_t target_offset) {
	D_ASSERT(chunk.ColumnCount() > column);
	auto &source = chunk.data[column];
	VectorOperations::Copy(source, target, index + 1, index, target_offset);
}

template <typename T>
struct WindowColumnIterator {
	using iterator = WindowColumnIterator<T>;
	using iterator_category = std::forward_iterator_tag;
	using difference_type = std::ptrdiff_t;
	using value_type = T;
	using reference = T;
	using pointer = idx_t;

	explicit WindowColumnIterator(WindowInputColumn &coll_p, pointer pos_p = 0) : coll(&coll_p), pos(pos_p) {
	}

	inline reference operator*() const {
		return coll->GetCell<T>(pos);
	}
	inline explicit operator pointer() const {
		return pos;
	}

	inline iterator &operator++() {
		++pos;
		return *this;
	}
	inline iterator operator++(int) {
		auto result = *this;
		++(*this);
		return result;
	}

	friend inline bool operator==(const iterator &a, const iterator &b) {
		return a.pos == b.pos;
	}
	friend inline bool operator!=(const iterator &a, const iterator &b) {
		return a.pos != b.pos;
	}

private:
	optional_ptr<WindowInputColumn> coll;
	pointer pos;
};

template <typename T, typename OP>
struct OperationCompare : public std::function<bool(T, T)> {
	inline bool operator()(const T &lhs, const T &val) const {
		return OP::template Operation(lhs, val);
	}
};

template <typename T, typename OP, bool FROM>
static idx_t FindTypedRangeBound(WindowInputColumn &over, const idx_t order_begin, const idx_t order_end,
                                 WindowInputExpression &boundary, const idx_t boundary_row) {
	D_ASSERT(!boundary.CellIsNull(boundary_row));
	const auto val = boundary.GetCell<T>(boundary_row);

	OperationCompare<T, OP> comp;
	WindowColumnIterator<T> begin(over, order_begin);
	WindowColumnIterator<T> end(over, order_end);
	if (FROM) {
		return idx_t(std::lower_bound(begin, end, val, comp));
	} else {
		return idx_t(std::upper_bound(begin, end, val, comp));
	}
}

template <typename OP, bool FROM>
static idx_t FindRangeBound(WindowInputColumn &over, const idx_t order_begin, const idx_t order_end,
                            WindowInputExpression &boundary, const idx_t expr_idx) {
	D_ASSERT(boundary.chunk.ColumnCount() == 1);
	D_ASSERT(boundary.chunk.data[0].GetType().InternalType() == over.input_expr.ptype);

	switch (over.input_expr.ptype) {
	case PhysicalType::INT8:
		return FindTypedRangeBound<int8_t, OP, FROM>(over, order_begin, order_end, boundary, expr_idx);
	case PhysicalType::INT16:
		return FindTypedRangeBound<int16_t, OP, FROM>(over, order_begin, order_end, boundary, expr_idx);
	case PhysicalType::INT32:
		return FindTypedRangeBound<int32_t, OP, FROM>(over, order_begin, order_end, boundary, expr_idx);
	case PhysicalType::INT64:
		return FindTypedRangeBound<int64_t, OP, FROM>(over, order_begin, order_end, boundary, expr_idx);
	case PhysicalType::UINT8:
		return FindTypedRangeBound<uint8_t, OP, FROM>(over, order_begin, order_end, boundary, expr_idx);
	case PhysicalType::UINT16:
		return FindTypedRangeBound<uint16_t, OP, FROM>(over, order_begin, order_end, boundary, expr_idx);
	case PhysicalType::UINT32:
		return FindTypedRangeBound<uint32_t, OP, FROM>(over, order_begin, order_end, boundary, expr_idx);
	case PhysicalType::UINT64:
		return FindTypedRangeBound<uint64_t, OP, FROM>(over, order_begin, order_end, boundary, expr_idx);
	case PhysicalType::INT128:
		return FindTypedRangeBound<hugeint_t, OP, FROM>(over, order_begin, order_end, boundary, expr_idx);
	case PhysicalType::FLOAT:
		return FindTypedRangeBound<float, OP, FROM>(over, order_begin, order_end, boundary, expr_idx);
	case PhysicalType::DOUBLE:
		return FindTypedRangeBound<double, OP, FROM>(over, order_begin, order_end, boundary, expr_idx);
	case PhysicalType::INTERVAL:
		return FindTypedRangeBound<interval_t, OP, FROM>(over, order_begin, order_end, boundary, expr_idx);
	default:
		throw InternalException("Unsupported column type for RANGE");
	}
}

template <bool FROM>
static idx_t FindOrderedRangeBound(WindowInputColumn &over, const OrderType range_sense, const idx_t order_begin,
                                   const idx_t order_end, WindowInputExpression &boundary, const idx_t expr_idx) {
	switch (range_sense) {
	case OrderType::ASCENDING:
		return FindRangeBound<LessThan, FROM>(over, order_begin, order_end, boundary, expr_idx);
	case OrderType::DESCENDING:
		return FindRangeBound<GreaterThan, FROM>(over, order_begin, order_end, boundary, expr_idx);
	default:
		throw InternalException("Unsupported ORDER BY sense for RANGE");
	}
}

void WindowBoundariesState::Update(const idx_t row_idx, WindowInputColumn &range_collection, const idx_t expr_idx,
                                   WindowInputExpression &boundary_start, WindowInputExpression &boundary_end,
                                   const ValidityMask &partition_mask, const ValidityMask &order_mask) {

	auto &bounds = *this;
	if (bounds.partition_count + bounds.order_count > 0) {

		// determine partition and peer group boundaries to ultimately figure out window size
		bounds.is_same_partition = !partition_mask.RowIsValidUnsafe(row_idx);
		bounds.is_peer = !order_mask.RowIsValidUnsafe(row_idx);

		// when the partition changes, recompute the boundaries
		if (!bounds.is_same_partition) {
			bounds.partition_start = row_idx;
			bounds.peer_start = row_idx;

			// find end of partition
			bounds.partition_end = bounds.input_size;
			if (bounds.partition_count) {
				idx_t n = 1;
				bounds.partition_end = FindNextStart(partition_mask, bounds.partition_start + 1, bounds.input_size, n);
			}

			// Find valid ordering values for the new partition
			// so we can exclude NULLs from RANGE expression computations
			bounds.valid_start = bounds.partition_start;
			bounds.valid_end = bounds.partition_end;

			if ((bounds.valid_start < bounds.valid_end) && bounds.has_preceding_range) {
				// Exclude any leading NULLs
				if (range_collection.CellIsNull(bounds.valid_start)) {
					idx_t n = 1;
					bounds.valid_start = FindNextStart(order_mask, bounds.valid_start + 1, bounds.valid_end, n);
				}
			}

			if ((bounds.valid_start < bounds.valid_end) && bounds.has_following_range) {
				// Exclude any trailing NULLs
				if (range_collection.CellIsNull(bounds.valid_end - 1)) {
					idx_t n = 1;
					bounds.valid_end = FindPrevStart(order_mask, bounds.valid_start, bounds.valid_end, n);
				}
			}

		} else if (!bounds.is_peer) {
			bounds.peer_start = row_idx;
		}

		if (bounds.needs_peer) {
			bounds.peer_end = bounds.partition_end;
			if (bounds.order_count) {
				idx_t n = 1;
				bounds.peer_end = FindNextStart(order_mask, bounds.peer_start + 1, bounds.partition_end, n);
			}
		}

	} else {
		bounds.is_same_partition = false;
		bounds.is_peer = true;
		bounds.partition_end = bounds.input_size;
		bounds.peer_end = bounds.partition_end;
	}

	// determine window boundaries depending on the type of expression
	bounds.window_start = -1;
	bounds.window_end = -1;

	switch (bounds.start_boundary) {
	case WindowBoundary::UNBOUNDED_PRECEDING:
		bounds.window_start = bounds.partition_start;
		break;
	case WindowBoundary::CURRENT_ROW_ROWS:
		bounds.window_start = row_idx;
		break;
	case WindowBoundary::CURRENT_ROW_RANGE:
		bounds.window_start = bounds.peer_start;
		break;
	case WindowBoundary::EXPR_PRECEDING_ROWS: {
		bounds.window_start = (int64_t)row_idx - boundary_start.GetCell<int64_t>(expr_idx);
		break;
	}
	case WindowBoundary::EXPR_FOLLOWING_ROWS: {
		bounds.window_start = row_idx + boundary_start.GetCell<int64_t>(expr_idx);
		break;
	}
	case WindowBoundary::EXPR_PRECEDING_RANGE: {
		if (boundary_start.CellIsNull(expr_idx)) {
			bounds.window_start = bounds.peer_start;
		} else {
			bounds.window_start = FindOrderedRangeBound<true>(range_collection, bounds.range_sense, bounds.valid_start,
			                                                  row_idx, boundary_start, expr_idx);
		}
		break;
	}
	case WindowBoundary::EXPR_FOLLOWING_RANGE: {
		if (boundary_start.CellIsNull(expr_idx)) {
			bounds.window_start = bounds.peer_start;
		} else {
			bounds.window_start = FindOrderedRangeBound<true>(range_collection, bounds.range_sense, row_idx,
			                                                  bounds.valid_end, boundary_start, expr_idx);
		}
		break;
	}
	default:
		throw InternalException("Unsupported window start boundary");
	}

	switch (bounds.end_boundary) {
	case WindowBoundary::CURRENT_ROW_ROWS:
		bounds.window_end = row_idx + 1;
		break;
	case WindowBoundary::CURRENT_ROW_RANGE:
		bounds.window_end = bounds.peer_end;
		break;
	case WindowBoundary::UNBOUNDED_FOLLOWING:
		bounds.window_end = bounds.partition_end;
		break;
	case WindowBoundary::EXPR_PRECEDING_ROWS:
		bounds.window_end = (int64_t)row_idx - boundary_end.GetCell<int64_t>(expr_idx) + 1;
		break;
	case WindowBoundary::EXPR_FOLLOWING_ROWS:
		bounds.window_end = row_idx + boundary_end.GetCell<int64_t>(expr_idx) + 1;
		break;
	case WindowBoundary::EXPR_PRECEDING_RANGE: {
		if (boundary_end.CellIsNull(expr_idx)) {
			bounds.window_end = bounds.peer_end;
		} else {
			bounds.window_end = FindOrderedRangeBound<false>(range_collection, bounds.range_sense, bounds.valid_start,
			                                                 row_idx, boundary_end, expr_idx);
		}
		break;
	}
	case WindowBoundary::EXPR_FOLLOWING_RANGE: {
		if (boundary_end.CellIsNull(expr_idx)) {
			bounds.window_end = bounds.peer_end;
		} else {
			bounds.window_end = FindOrderedRangeBound<false>(range_collection, bounds.range_sense, row_idx,
			                                                 bounds.valid_end, boundary_end, expr_idx);
		}
		break;
	}
	default:
		throw InternalException("Unsupported window end boundary");
	}

	// clamp windows to partitions if they should exceed
	if (bounds.window_start < (int64_t)bounds.partition_start) {
		bounds.window_start = bounds.partition_start;
	}
	if (bounds.window_start > (int64_t)bounds.partition_end) {
		bounds.window_start = bounds.partition_end;
	}
	if (bounds.window_end < (int64_t)bounds.partition_start) {
		bounds.window_end = bounds.partition_start;
	}
	if (bounds.window_end > (int64_t)bounds.partition_end) {
		bounds.window_end = bounds.partition_end;
	}

	if (bounds.window_start < 0 || bounds.window_end < 0) {
		throw InternalException("Failed to compute window boundaries");
	}
}

struct WindowExecutor {
	static bool IsConstantAggregate(const BoundWindowExpression &wexpr);

	WindowExecutor(BoundWindowExpression &wexpr, ClientContext &context, const ValidityMask &partition_mask,
	               const idx_t count);

	void Sink(DataChunk &input_chunk, const idx_t input_idx, const idx_t total_count);
	void Finalize(WindowAggregationMode mode);

	void Evaluate(idx_t row_idx, DataChunk &input_chunk, Vector &result, const ValidityMask &partition_mask,
	              const ValidityMask &order_mask);

	// The function
	BoundWindowExpression &wexpr;

	// Frame management
	WindowBoundariesState bounds;
	uint64_t dense_rank = 1;
	uint64_t rank_equal = 0;
	uint64_t rank = 1;

	// Expression collections
	DataChunk payload_collection;
	ExpressionExecutor payload_executor;
	DataChunk payload_chunk;

	ExpressionExecutor filter_executor;
	ValidityMask filter_mask;
	vector<validity_t> filter_bits;
	SelectionVector filter_sel;

	// LEAD/LAG Evaluation
	WindowInputExpression leadlag_offset;
	WindowInputExpression leadlag_default;

	// evaluate boundaries if present. Parser has checked boundary types.
	WindowInputExpression boundary_start;
	WindowInputExpression boundary_end;

	// evaluate RANGE expressions, if needed
	WindowInputColumn range;

	// IGNORE NULLS
	ValidityMask ignore_nulls;

	// build a segment tree for frame-adhering aggregates
	// see http://www.vldb.org/pvldb/vol8/p1058-leis.pdf
	unique_ptr<WindowSegmentTree> segment_tree = nullptr;

	// all aggregate values are the same for each partition
	unique_ptr<WindowConstantAggregate> constant_aggregate = nullptr;
};

bool WindowExecutor::IsConstantAggregate(const BoundWindowExpression &wexpr) {
	if (!wexpr.aggregate) {
		return false;
	}

	//	COUNT(*) is already handled efficiently by segment trees.
	if (wexpr.children.empty()) {
		return false;
	}

	/*
	    The default framing option is RANGE UNBOUNDED PRECEDING, which
	    is the same as RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT
	    ROW; it sets the frame to be all rows from the partition start
	    up through the current row's last peer (a row that the window's
	    ORDER BY clause considers equivalent to the current row; all
	    rows are peers if there is no ORDER BY). In general, UNBOUNDED
	    PRECEDING means that the frame starts with the first row of the
	    partition, and similarly UNBOUNDED FOLLOWING means that the
	    frame ends with the last row of the partition, regardless of
	    RANGE, ROWS or GROUPS mode. In ROWS mode, CURRENT ROW means that
	    the frame starts or ends with the current row; but in RANGE or
	    GROUPS mode it means that the frame starts or ends with the
	    current row's first or last peer in the ORDER BY ordering. The
	    offset PRECEDING and offset FOLLOWING options vary in meaning
	    depending on the frame mode.
	*/
	switch (wexpr.start) {
	case WindowBoundary::UNBOUNDED_PRECEDING:
		break;
	case WindowBoundary::CURRENT_ROW_RANGE:
		if (!wexpr.orders.empty()) {
			return false;
		}
		break;
	default:
		return false;
	}

	switch (wexpr.end) {
	case WindowBoundary::UNBOUNDED_FOLLOWING:
		break;
	case WindowBoundary::CURRENT_ROW_RANGE:
		if (!wexpr.orders.empty()) {
			return false;
		}
		break;
	default:
		return false;
	}

	return true;
}

WindowExecutor::WindowExecutor(BoundWindowExpression &wexpr, ClientContext &context, const ValidityMask &partition_mask,
                               const idx_t count)
    : wexpr(wexpr), bounds(wexpr, count), payload_collection(), payload_executor(context), filter_executor(context),
      leadlag_offset(wexpr.offset_expr.get(), context), leadlag_default(wexpr.default_expr.get(), context),
      boundary_start(wexpr.start_expr.get(), context), boundary_end(wexpr.end_expr.get(), context),
      range((bounds.has_preceding_range || bounds.has_following_range) ? wexpr.orders[0].expression.get() : nullptr,
            context, count)

{
	// TODO we could evaluate those expressions in parallel

	//	Check for constant aggregate
	if (IsConstantAggregate(wexpr)) {
		constant_aggregate =
		    make_uniq<WindowConstantAggregate>(AggregateObject(wexpr), wexpr.return_type, partition_mask, count);
	}

	// evaluate the FILTER clause and stuff it into a large mask for compactness and reuse
	if (wexpr.filter_expr) {
		// 	Start with all invalid and set the ones that pass
		filter_bits.resize(ValidityMask::ValidityMaskSize(count), 0);
		filter_mask.Initialize(filter_bits.data());
		filter_executor.AddExpression(*wexpr.filter_expr);
		filter_sel.Initialize(STANDARD_VECTOR_SIZE);
	}

	// TODO: child may be a scalar, don't need to materialize the whole collection then

	// evaluate inner expressions of window functions, could be more complex
	PrepareInputExpressions(wexpr.children, payload_executor, payload_chunk);

	auto types = payload_chunk.GetTypes();
	if (!types.empty()) {
		payload_collection.Initialize(Allocator::Get(context), types);
	}
}

void WindowExecutor::Sink(DataChunk &input_chunk, const idx_t input_idx, const idx_t total_count) {
	// Single pass over the input to produce the global data.
	// Vectorisation for the win...

	// Set up a validity mask for IGNORE NULLS
	bool check_nulls = false;
	if (wexpr.ignore_nulls) {
		switch (wexpr.type) {
		case ExpressionType::WINDOW_LEAD:
		case ExpressionType::WINDOW_LAG:
		case ExpressionType::WINDOW_FIRST_VALUE:
		case ExpressionType::WINDOW_LAST_VALUE:
		case ExpressionType::WINDOW_NTH_VALUE:
			check_nulls = true;
			break;
		default:
			break;
		}
	}

	const auto count = input_chunk.size();

	idx_t filtered = 0;
	SelectionVector *filtering = nullptr;
	if (wexpr.filter_expr) {
		filtering = &filter_sel;
		filtered = filter_executor.SelectExpression(input_chunk, filter_sel);
		for (idx_t f = 0; f < filtered; ++f) {
			filter_mask.SetValid(input_idx + filter_sel[f]);
		}
	}

	if (!wexpr.children.empty()) {
		payload_chunk.Reset();
		payload_executor.Execute(input_chunk, payload_chunk);
		payload_chunk.Verify();
		if (constant_aggregate) {
			constant_aggregate->Sink(payload_chunk, filtering, filtered);
		} else {
			payload_collection.Append(payload_chunk, true);
		}

		// process payload chunks while they are still piping hot
		if (check_nulls) {
			UnifiedVectorFormat vdata;
			payload_chunk.data[0].ToUnifiedFormat(count, vdata);
			if (!vdata.validity.AllValid()) {
				//	Lazily materialise the contents when we find the first NULL
				if (ignore_nulls.AllValid()) {
					ignore_nulls.Initialize(total_count);
				}
				// Write to the current position
				if (input_idx % ValidityMask::BITS_PER_VALUE == 0) {
					// If we are at the edge of an output entry, just copy the entries
					auto dst = ignore_nulls.GetData() + ignore_nulls.EntryCount(input_idx);
					auto src = vdata.validity.GetData();
					for (auto entry_count = vdata.validity.EntryCount(count); entry_count-- > 0;) {
						*dst++ = *src++;
					}
				} else {
					// If not, we have ragged data and need to copy one bit at a time.
					for (idx_t i = 0; i < count; ++i) {
						ignore_nulls.Set(input_idx + i, vdata.validity.RowIsValid(i));
					}
				}
			}
		}
	}

	range.Append(input_chunk);
}

void WindowExecutor::Finalize(WindowAggregationMode mode) {
	// build a segment tree for frame-adhering aggregates
	// see http://www.vldb.org/pvldb/vol8/p1058-leis.pdf
	if (constant_aggregate) {
		constant_aggregate->Finalize();
	} else if (wexpr.aggregate) {
		segment_tree = make_uniq<WindowSegmentTree>(AggregateObject(wexpr), wexpr.return_type, &payload_collection,
		                                            filter_mask, mode);
	}
}

void WindowExecutor::Evaluate(idx_t row_idx, DataChunk &input_chunk, Vector &result, const ValidityMask &partition_mask,
                              const ValidityMask &order_mask) {
	// Evaluate the row-level arguments
	boundary_start.Execute(input_chunk);
	boundary_end.Execute(input_chunk);

	leadlag_offset.Execute(input_chunk);
	leadlag_default.Execute(input_chunk);

	// this is the main loop, go through all sorted rows and compute window function result
	for (idx_t output_offset = 0; output_offset < input_chunk.size(); ++output_offset, ++row_idx) {
		// special case, OVER (), aggregate over everything
		bounds.Update(row_idx, range, output_offset, boundary_start, boundary_end, partition_mask, order_mask);
		if (WindowNeedsRank(wexpr)) {
			if (!bounds.is_same_partition || row_idx == 0) { // special case for first row, need to init
				dense_rank = 1;
				rank = 1;
				rank_equal = 0;
			} else if (!bounds.is_peer) {
				dense_rank++;
				rank += rank_equal;
				rank_equal = 0;
			}
			rank_equal++;
		}

		// if no values are read for window, result is NULL
		if (bounds.window_start >= bounds.window_end) {
			FlatVector::SetNull(result, output_offset, true);
			continue;
		}

		switch (wexpr.type) {
		case ExpressionType::WINDOW_AGGREGATE: {
			if (constant_aggregate) {
				constant_aggregate->Compute(result, output_offset, bounds.window_start, bounds.window_end);
			} else {
				segment_tree->Compute(result, output_offset, bounds.window_start, bounds.window_end);
			}
			break;
		}
		case ExpressionType::WINDOW_ROW_NUMBER: {
			auto rdata = FlatVector::GetData<int64_t>(result);
			rdata[output_offset] = row_idx - bounds.partition_start + 1;
			break;
		}
		case ExpressionType::WINDOW_RANK_DENSE: {
			auto rdata = FlatVector::GetData<int64_t>(result);
			rdata[output_offset] = dense_rank;
			break;
		}
		case ExpressionType::WINDOW_RANK: {
			auto rdata = FlatVector::GetData<int64_t>(result);
			rdata[output_offset] = rank;
			break;
		}
		case ExpressionType::WINDOW_PERCENT_RANK: {
			int64_t denom = (int64_t)bounds.partition_end - bounds.partition_start - 1;
			double percent_rank = denom > 0 ? ((double)rank - 1) / denom : 0;
			auto rdata = FlatVector::GetData<double>(result);
			rdata[output_offset] = percent_rank;
			break;
		}
		case ExpressionType::WINDOW_CUME_DIST: {
			int64_t denom = (int64_t)bounds.partition_end - bounds.partition_start;
			double cume_dist = denom > 0 ? ((double)(bounds.peer_end - bounds.partition_start)) / denom : 0;
			auto rdata = FlatVector::GetData<double>(result);
			rdata[output_offset] = cume_dist;
			break;
		}
		case ExpressionType::WINDOW_NTILE: {
			D_ASSERT(payload_collection.ColumnCount() == 1);
			if (CellIsNull(payload_collection, 0, row_idx)) {
				FlatVector::SetNull(result, output_offset, true);
			} else {
				auto n_param = GetCell<int64_t>(payload_collection, 0, row_idx);
				if (n_param < 1) {
					throw InvalidInputException("Argument for ntile must be greater than zero");
				}
				// With thanks from SQLite's ntileValueFunc()
				int64_t n_total = bounds.partition_end - bounds.partition_start;
				if (n_param > n_total) {
					// more groups allowed than we have values
					// map every entry to a unique group
					n_param = n_total;
				}
				int64_t n_size = (n_total / n_param);
				// find the row idx within the group
				D_ASSERT(row_idx >= bounds.partition_start);
				int64_t adjusted_row_idx = row_idx - bounds.partition_start;
				// now compute the ntile
				int64_t n_large = n_total - n_param * n_size;
				int64_t i_small = n_large * (n_size + 1);
				int64_t result_ntile;

				D_ASSERT((n_large * (n_size + 1) + (n_param - n_large) * n_size) == n_total);

				if (adjusted_row_idx < i_small) {
					result_ntile = 1 + adjusted_row_idx / (n_size + 1);
				} else {
					result_ntile = 1 + n_large + (adjusted_row_idx - i_small) / n_size;
				}
				// result has to be between [1, NTILE]
				D_ASSERT(result_ntile >= 1 && result_ntile <= n_param);
				auto rdata = FlatVector::GetData<int64_t>(result);
				rdata[output_offset] = result_ntile;
			}
			break;
		}
		case ExpressionType::WINDOW_LEAD:
		case ExpressionType::WINDOW_LAG: {
			int64_t offset = 1;
			if (wexpr.offset_expr) {
				offset = leadlag_offset.GetCell<int64_t>(output_offset);
			}
			int64_t val_idx = (int64_t)row_idx;
			if (wexpr.type == ExpressionType::WINDOW_LEAD) {
				val_idx += offset;
			} else {
				val_idx -= offset;
			}

			idx_t delta = 0;
			if (val_idx < (int64_t)row_idx) {
				// Count backwards
				delta = idx_t(row_idx - val_idx);
				val_idx = FindPrevStart(ignore_nulls, bounds.partition_start, row_idx, delta);
			} else if (val_idx > (int64_t)row_idx) {
				delta = idx_t(val_idx - row_idx);
				val_idx = FindNextStart(ignore_nulls, row_idx + 1, bounds.partition_end, delta);
			}
			// else offset is zero, so don't move.

			if (!delta) {
				CopyCell(payload_collection, 0, val_idx, result, output_offset);
			} else if (wexpr.default_expr) {
				leadlag_default.CopyCell(result, output_offset);
			} else {
				FlatVector::SetNull(result, output_offset, true);
			}
			break;
		}
		case ExpressionType::WINDOW_FIRST_VALUE: {
			//	Same as NTH_VALUE(..., 1)
			idx_t n = 1;
			const auto first_idx = FindNextStart(ignore_nulls, bounds.window_start, bounds.window_end, n);
			if (!n) {
				CopyCell(payload_collection, 0, first_idx, result, output_offset);
			} else {
				FlatVector::SetNull(result, output_offset, true);
			}
			break;
		}
		case ExpressionType::WINDOW_LAST_VALUE: {
			idx_t n = 1;
			const auto last_idx = FindPrevStart(ignore_nulls, bounds.window_start, bounds.window_end, n);
			if (!n) {
				CopyCell(payload_collection, 0, last_idx, result, output_offset);
			} else {
				FlatVector::SetNull(result, output_offset, true);
			}
			break;
		}
		case ExpressionType::WINDOW_NTH_VALUE: {
			D_ASSERT(payload_collection.ColumnCount() == 2);
			// Returns value evaluated at the row that is the n'th row of the window frame (counting from 1);
			// returns NULL if there is no such row.
			if (CellIsNull(payload_collection, 1, row_idx)) {
				FlatVector::SetNull(result, output_offset, true);
			} else {
				auto n_param = GetCell<int64_t>(payload_collection, 1, row_idx);
				if (n_param < 1) {
					FlatVector::SetNull(result, output_offset, true);
				} else {
					auto n = idx_t(n_param);
					const auto nth_index = FindNextStart(ignore_nulls, bounds.window_start, bounds.window_end, n);
					if (!n) {
						CopyCell(payload_collection, 0, nth_index, result, output_offset);
					} else {
						FlatVector::SetNull(result, output_offset, true);
					}
				}
			}
			break;
		}
		default:
			throw InternalException("Window aggregate type %s", ExpressionTypeToString(wexpr.type));
		}
	}

	result.Verify(input_chunk.size());
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
SinkResultType PhysicalWindow::Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const {
	auto &lstate = input.local_state.Cast<WindowLocalSinkState>();

	lstate.Sink(chunk);

	return SinkResultType::NEED_MORE_INPUT;
}

void PhysicalWindow::Combine(ExecutionContext &context, GlobalSinkState &gstate_p, LocalSinkState &lstate_p) const {
	auto &lstate = lstate_p.Cast<WindowLocalSinkState>();
	lstate.Combine();
}

unique_ptr<LocalSinkState> PhysicalWindow::GetLocalSinkState(ExecutionContext &context) const {
	auto &gstate = sink_state->Cast<WindowGlobalSinkState>();
	return make_uniq<WindowLocalSinkState>(context.client, gstate);
}

unique_ptr<GlobalSinkState> PhysicalWindow::GetGlobalSinkState(ClientContext &context) const {
	return make_uniq<WindowGlobalSinkState>(*this, context);
}

SinkFinalizeType PhysicalWindow::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                          GlobalSinkState &gstate_p) const {
	auto &state = gstate_p.Cast<WindowGlobalSinkState>();

	//	Did we get any data?
	if (!state.global_partition->count) {
		return SinkFinalizeType::NO_OUTPUT_POSSIBLE;
	}

	// Do we have any sorting to schedule?
	if (state.global_partition->rows) {
		D_ASSERT(!state.global_partition->grouping_data);
		return state.global_partition->rows->count ? SinkFinalizeType::READY : SinkFinalizeType::NO_OUTPUT_POSSIBLE;
	}

	// Find the first group to sort
	auto &groups = state.global_partition->grouping_data->GetPartitions();
	if (groups.empty()) {
		// Empty input!
		return SinkFinalizeType::NO_OUTPUT_POSSIBLE;
	}

	// Schedule all the sorts for maximum thread utilisation
	auto new_event = make_shared<PartitionMergeEvent>(*state.global_partition, pipeline);
	event.InsertEvent(std::move(new_event));

	return SinkFinalizeType::READY;
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
class WindowGlobalSourceState : public GlobalSourceState {
public:
	explicit WindowGlobalSourceState(WindowGlobalSinkState &gsink) : gsink(*gsink.global_partition), next_bin(0) {
	}

	PartitionGlobalSinkState &gsink;
	//! The output read position.
	atomic<idx_t> next_bin;

public:
	idx_t MaxThreads() override {
		// If there is only one partition, we have to process it on one thread.
		if (!gsink.grouping_data) {
			return 1;
		}

		// If there is not a lot of data, process serially.
		if (gsink.count < STANDARD_ROW_GROUPS_SIZE) {
			return 1;
		}

		return gsink.hash_groups.size();
	}
};

// Per-thread read state
class WindowLocalSourceState : public LocalSourceState {
public:
	using HashGroupPtr = unique_ptr<PartitionGlobalHashGroup>;
	using WindowExecutorPtr = unique_ptr<WindowExecutor>;
	using WindowExecutors = vector<WindowExecutorPtr>;

	WindowLocalSourceState(const PhysicalWindow &op_p, ExecutionContext &context, WindowGlobalSourceState &gsource)
	    : context(context.client), op(op_p), gsink(gsource.gsink) {

		vector<LogicalType> output_types;
		for (idx_t expr_idx = 0; expr_idx < op.select_list.size(); ++expr_idx) {
			D_ASSERT(op.select_list[expr_idx]->GetExpressionClass() == ExpressionClass::BOUND_WINDOW);
			auto &wexpr = op.select_list[expr_idx]->Cast<BoundWindowExpression>();
			output_types.emplace_back(wexpr.return_type);
		}
		output_chunk.Initialize(Allocator::Get(context.client), output_types);

		const auto &input_types = gsink.payload_types;
		layout.Initialize(input_types);
		input_chunk.Initialize(gsink.allocator, input_types);
	}

	void MaterializeSortedData();
	void GeneratePartition(WindowGlobalSinkState &gstate, const idx_t hash_bin);
	void Scan(DataChunk &chunk);

	HashGroupPtr hash_group;
	ClientContext &context;
	const PhysicalWindow &op;

	PartitionGlobalSinkState &gsink;

	//! The generated input chunks
	unique_ptr<RowDataCollection> rows;
	unique_ptr<RowDataCollection> heap;
	RowLayout layout;
	//! The partition boundary mask
	vector<validity_t> partition_bits;
	ValidityMask partition_mask;
	//! The order boundary mask
	vector<validity_t> order_bits;
	ValidityMask order_mask;
	//! The current execution functions
	WindowExecutors window_execs;

	//! The read partition
	idx_t hash_bin;
	//! The read cursor
	unique_ptr<RowDataCollectionScanner> scanner;
	//! Buffer for the inputs
	DataChunk input_chunk;
	//! Buffer for window results
	DataChunk output_chunk;
};

void WindowLocalSourceState::MaterializeSortedData() {
	auto &global_sort_state = *hash_group->global_sort;
	if (global_sort_state.sorted_blocks.empty()) {
		return;
	}

	// scan the sorted row data
	D_ASSERT(global_sort_state.sorted_blocks.size() == 1);
	auto &sb = *global_sort_state.sorted_blocks[0];

	// Free up some memory before allocating more
	sb.radix_sorting_data.clear();
	sb.blob_sorting_data = nullptr;

	// Move the sorting row blocks into our RDCs
	auto &buffer_manager = global_sort_state.buffer_manager;
	auto &sd = *sb.payload_data;

	// Data blocks are required
	D_ASSERT(!sd.data_blocks.empty());
	auto &block = sd.data_blocks[0];
	rows = make_uniq<RowDataCollection>(buffer_manager, block->capacity, block->entry_size);
	rows->blocks = std::move(sd.data_blocks);
	rows->count = std::accumulate(rows->blocks.begin(), rows->blocks.end(), idx_t(0),
	                              [&](idx_t c, const unique_ptr<RowDataBlock> &b) { return c + b->count; });

	// Heap blocks are optional, but we want both for iteration.
	if (!sd.heap_blocks.empty()) {
		auto &block = sd.heap_blocks[0];
		heap = make_uniq<RowDataCollection>(buffer_manager, block->capacity, block->entry_size);
		heap->blocks = std::move(sd.heap_blocks);
		hash_group.reset();
	} else {
		heap = make_uniq<RowDataCollection>(buffer_manager, (idx_t)Storage::BLOCK_SIZE, 1, true);
	}
	heap->count = std::accumulate(heap->blocks.begin(), heap->blocks.end(), idx_t(0),
	                              [&](idx_t c, const unique_ptr<RowDataBlock> &b) { return c + b->count; });
}

void WindowLocalSourceState::GeneratePartition(WindowGlobalSinkState &gstate, const idx_t hash_bin_p) {
	//	Get rid of any stale data
	hash_bin = hash_bin_p;

	// There are three types of partitions:
	// 1. No partition (no sorting)
	// 2. One partition (sorting, but no hashing)
	// 3. Multiple partitions (sorting and hashing)

	//	How big is the partition?
	idx_t count = 0;
	if (hash_bin < gsink.hash_groups.size() && gsink.hash_groups[hash_bin]) {
		count = gsink.hash_groups[hash_bin]->count;
	} else if (gsink.rows && !hash_bin) {
		count = gsink.count;
	} else {
		return;
	}

	//	Initialise masks to false
	const auto bit_count = ValidityMask::ValidityMaskSize(count);
	partition_bits.clear();
	partition_bits.resize(bit_count, 0);
	partition_mask.Initialize(partition_bits.data());

	order_bits.clear();
	order_bits.resize(bit_count, 0);
	order_mask.Initialize(order_bits.data());

	// Scan the sorted data into new Collections
	auto external = gsink.external;
	if (gsink.rows && !hash_bin) {
		// Simple mask
		partition_mask.SetValidUnsafe(0);
		order_mask.SetValidUnsafe(0);
		//	No partition - align the heap blocks with the row blocks
		rows = gsink.rows->CloneEmpty(gsink.rows->keep_pinned);
		heap = gsink.strings->CloneEmpty(gsink.strings->keep_pinned);
		RowDataCollectionScanner::AlignHeapBlocks(*rows, *heap, *gsink.rows, *gsink.strings, layout);
		external = true;
	} else if (hash_bin < gsink.hash_groups.size() && gsink.hash_groups[hash_bin]) {
		// Overwrite the collections with the sorted data
		hash_group = std::move(gsink.hash_groups[hash_bin]);
		hash_group->ComputeMasks(partition_mask, order_mask);
		MaterializeSortedData();
	} else {
		return;
	}

	// Create the executors for each function
	window_execs.clear();
	for (idx_t expr_idx = 0; expr_idx < op.select_list.size(); ++expr_idx) {
		D_ASSERT(op.select_list[expr_idx]->GetExpressionClass() == ExpressionClass::BOUND_WINDOW);
		auto &wexpr = op.select_list[expr_idx]->Cast<BoundWindowExpression>();
		auto wexec = make_uniq<WindowExecutor>(wexpr, context, partition_mask, count);
		window_execs.emplace_back(std::move(wexec));
	}

	//	First pass over the input without flushing
	//	TODO: Factor out the constructor data as global state
	scanner = make_uniq<RowDataCollectionScanner>(*rows, *heap, layout, external, false);
	idx_t input_idx = 0;
	while (true) {
		input_chunk.Reset();
		scanner->Scan(input_chunk);
		if (input_chunk.size() == 0) {
			break;
		}

		//	TODO: Parallelization opportunity
		for (auto &wexec : window_execs) {
			wexec->Sink(input_chunk, input_idx, scanner->Count());
		}
		input_idx += input_chunk.size();
	}

	//	TODO: Parallelization opportunity
	for (auto &wexec : window_execs) {
		wexec->Finalize(gstate.mode);
	}

	// External scanning assumes all blocks are swizzled.
	scanner->ReSwizzle();

	//	Second pass can flush
	scanner->Reset(true);
}

void WindowLocalSourceState::Scan(DataChunk &result) {
	D_ASSERT(scanner);
	if (!scanner->Remaining()) {
		return;
	}

	const auto position = scanner->Scanned();
	input_chunk.Reset();
	scanner->Scan(input_chunk);

	output_chunk.Reset();
	for (idx_t expr_idx = 0; expr_idx < window_execs.size(); ++expr_idx) {
		auto &executor = *window_execs[expr_idx];
		executor.Evaluate(position, input_chunk, output_chunk.data[expr_idx], partition_mask, order_mask);
	}
	output_chunk.SetCardinality(input_chunk);
	output_chunk.Verify();

	idx_t out_idx = 0;
	result.SetCardinality(input_chunk);
	for (idx_t col_idx = 0; col_idx < input_chunk.ColumnCount(); col_idx++) {
		result.data[out_idx++].Reference(input_chunk.data[col_idx]);
	}
	for (idx_t col_idx = 0; col_idx < output_chunk.ColumnCount(); col_idx++) {
		result.data[out_idx++].Reference(output_chunk.data[col_idx]);
	}
	result.Verify();
}

unique_ptr<LocalSourceState> PhysicalWindow::GetLocalSourceState(ExecutionContext &context,
                                                                 GlobalSourceState &gstate_p) const {
	auto &gstate = gstate_p.Cast<WindowGlobalSourceState>();
	return make_uniq<WindowLocalSourceState>(*this, context, gstate);
}

unique_ptr<GlobalSourceState> PhysicalWindow::GetGlobalSourceState(ClientContext &context) const {
	auto &gsink = sink_state->Cast<WindowGlobalSinkState>();
	return make_uniq<WindowGlobalSourceState>(gsink);
}

SourceResultType PhysicalWindow::GetData(ExecutionContext &context, DataChunk &chunk,
                                         OperatorSourceInput &input) const {
	auto &lsource = input.local_state.Cast<WindowLocalSourceState>();
	auto &gsource = input.global_state.Cast<WindowGlobalSourceState>();
	auto &gsink = sink_state->Cast<WindowGlobalSinkState>();

	auto &hash_groups = gsink.global_partition->hash_groups;
	const auto bin_count = hash_groups.empty() ? 1 : hash_groups.size();

	while (chunk.size() == 0) {
		//	Move to the next bin if we are done.
		while (!lsource.scanner || !lsource.scanner->Remaining()) {
			lsource.scanner.reset();
			lsource.rows.reset();
			lsource.heap.reset();
			lsource.hash_group.reset();
			auto hash_bin = gsource.next_bin++;
			if (hash_bin >= bin_count) {
				return chunk.size() > 0 ? SourceResultType::HAVE_MORE_OUTPUT : SourceResultType::FINISHED;
			}

			for (; hash_bin < hash_groups.size(); hash_bin = gsource.next_bin++) {
				if (hash_groups[hash_bin]) {
					break;
				}
			}
			lsource.GeneratePartition(gsink, hash_bin);
		}

		lsource.Scan(chunk);
	}

	return chunk.size() == 0 ? SourceResultType::FINISHED : SourceResultType::HAVE_MORE_OUTPUT;
}

string PhysicalWindow::ParamsToString() const {
	string result;
	for (idx_t i = 0; i < select_list.size(); i++) {
		if (i > 0) {
			result += "\n";
		}
		result += select_list[i]->GetName();
	}
	return result;
}

} // namespace duckdb




namespace duckdb {

PhysicalFilter::PhysicalFilter(vector<LogicalType> types, vector<unique_ptr<Expression>> select_list,
                               idx_t estimated_cardinality)
    : CachingPhysicalOperator(PhysicalOperatorType::FILTER, std::move(types), estimated_cardinality) {
	D_ASSERT(select_list.size() > 0);
	if (select_list.size() > 1) {
		// create a big AND out of the expressions
		auto conjunction = make_uniq<BoundConjunctionExpression>(ExpressionType::CONJUNCTION_AND);
		for (auto &expr : select_list) {
			conjunction->children.push_back(std::move(expr));
		}
		expression = std::move(conjunction);
	} else {
		expression = std::move(select_list[0]);
	}
}

class FilterState : public CachingOperatorState {
public:
	explicit FilterState(ExecutionContext &context, Expression &expr)
	    : executor(context.client, expr), sel(STANDARD_VECTOR_SIZE) {
	}

	ExpressionExecutor executor;
	SelectionVector sel;

public:
	void Finalize(const PhysicalOperator &op, ExecutionContext &context) override {
		context.thread.profiler.Flush(op, executor, "filter", 0);
	}
};

unique_ptr<OperatorState> PhysicalFilter::GetOperatorState(ExecutionContext &context) const {
	return make_uniq<FilterState>(context, *expression);
}

OperatorResultType PhysicalFilter::ExecuteInternal(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                                   GlobalOperatorState &gstate, OperatorState &state_p) const {
	auto &state = (FilterState &)state_p;
	idx_t result_count = state.executor.SelectExpression(input, state.sel);
	if (result_count == input.size()) {
		// nothing was filtered: skip adding any selection vectors
		chunk.Reference(input);
	} else {
		chunk.Slice(input, state.sel, result_count);
	}
	return OperatorResultType::NEED_MORE_INPUT;
}

string PhysicalFilter::ParamsToString() const {
	auto result = expression->GetName();
	result += "\n[INFOSEPARATOR]\n";
	result += StringUtil::Format("EC: %llu", estimated_props->GetCardinality<idx_t>());
	return result;
}

} // namespace duckdb





namespace duckdb {

PhysicalBatchCollector::PhysicalBatchCollector(PreparedStatementData &data) : PhysicalResultCollector(data) {
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
class BatchCollectorGlobalState : public GlobalSinkState {
public:
	BatchCollectorGlobalState(ClientContext &context, const PhysicalBatchCollector &op) : data(op.types) {
	}

	mutex glock;
	BatchedDataCollection data;
	unique_ptr<MaterializedQueryResult> result;
};

class BatchCollectorLocalState : public LocalSinkState {
public:
	BatchCollectorLocalState(ClientContext &context, const PhysicalBatchCollector &op) : data(op.types) {
	}

	BatchedDataCollection data;
};

SinkResultType PhysicalBatchCollector::Sink(ExecutionContext &context, DataChunk &chunk,
                                            OperatorSinkInput &input) const {
	auto &state = input.local_state.Cast<BatchCollectorLocalState>();
	state.data.Append(chunk, state.partition_info.batch_index.GetIndex());
	return SinkResultType::NEED_MORE_INPUT;
}

void PhysicalBatchCollector::Combine(ExecutionContext &context, GlobalSinkState &gstate_p,
                                     LocalSinkState &lstate_p) const {
	auto &gstate = gstate_p.Cast<BatchCollectorGlobalState>();
	auto &state = lstate_p.Cast<BatchCollectorLocalState>();

	lock_guard<mutex> lock(gstate.glock);
	gstate.data.Merge(state.data);
}

SinkFinalizeType PhysicalBatchCollector::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                                  GlobalSinkState &gstate_p) const {
	auto &gstate = gstate_p.Cast<BatchCollectorGlobalState>();
	auto collection = gstate.data.FetchCollection();
	D_ASSERT(collection);
	auto result = make_uniq<MaterializedQueryResult>(statement_type, properties, names, std::move(collection),
	                                                 context.GetClientProperties());
	gstate.result = std::move(result);
	return SinkFinalizeType::READY;
}

unique_ptr<LocalSinkState> PhysicalBatchCollector::GetLocalSinkState(ExecutionContext &context) const {
	return make_uniq<BatchCollectorLocalState>(context.client, *this);
}

unique_ptr<GlobalSinkState> PhysicalBatchCollector::GetGlobalSinkState(ClientContext &context) const {
	return make_uniq<BatchCollectorGlobalState>(context, *this);
}

unique_ptr<QueryResult> PhysicalBatchCollector::GetResult(GlobalSinkState &state) {
	auto &gstate = state.Cast<BatchCollectorGlobalState>();
	D_ASSERT(gstate.result);
	return std::move(gstate.result);
}

} // namespace duckdb




namespace duckdb {

PhysicalExecute::PhysicalExecute(PhysicalOperator &plan)
    : PhysicalOperator(PhysicalOperatorType::EXECUTE, plan.types, -1), plan(plan) {
}

vector<const_reference<PhysicalOperator>> PhysicalExecute::GetChildren() const {
	return {plan};
}

void PhysicalExecute::BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) {
	// EXECUTE statement: build pipeline on child
	meta_pipeline.Build(plan);
}

} // namespace duckdb




namespace duckdb {

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
class ExplainAnalyzeStateGlobalState : public GlobalSinkState {
public:
	string analyzed_plan;
};

SinkResultType PhysicalExplainAnalyze::Sink(ExecutionContext &context, DataChunk &chunk,
                                            OperatorSinkInput &input) const {
	return SinkResultType::NEED_MORE_INPUT;
}

SinkFinalizeType PhysicalExplainAnalyze::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                                  GlobalSinkState &gstate_p) const {
	auto &gstate = gstate_p.Cast<ExplainAnalyzeStateGlobalState>();
	auto &profiler = QueryProfiler::Get(context);
	gstate.analyzed_plan = profiler.ToString();
	return SinkFinalizeType::READY;
}

unique_ptr<GlobalSinkState> PhysicalExplainAnalyze::GetGlobalSinkState(ClientContext &context) const {
	return make_uniq<ExplainAnalyzeStateGlobalState>();
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
SourceResultType PhysicalExplainAnalyze::GetData(ExecutionContext &context, DataChunk &chunk,
                                                 OperatorSourceInput &input) const {
	auto &gstate = sink_state->Cast<ExplainAnalyzeStateGlobalState>();

	chunk.SetValue(0, 0, Value("analyzed_plan"));
	chunk.SetValue(1, 0, Value(gstate.analyzed_plan));
	chunk.SetCardinality(1);

	return SourceResultType::FINISHED;
}

} // namespace duckdb









namespace duckdb {

PhysicalLimit::PhysicalLimit(vector<LogicalType> types, idx_t limit, idx_t offset,
                             unique_ptr<Expression> limit_expression, unique_ptr<Expression> offset_expression,
                             idx_t estimated_cardinality)
    : PhysicalOperator(PhysicalOperatorType::LIMIT, std::move(types), estimated_cardinality), limit_value(limit),
      offset_value(offset), limit_expression(std::move(limit_expression)),
      offset_expression(std::move(offset_expression)) {
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
class LimitGlobalState : public GlobalSinkState {
public:
	explicit LimitGlobalState(ClientContext &context, const PhysicalLimit &op) : data(op.types) {
		limit = 0;
		offset = 0;
	}

	mutex glock;
	idx_t limit;
	idx_t offset;
	BatchedDataCollection data;
};

class LimitLocalState : public LocalSinkState {
public:
	explicit LimitLocalState(ClientContext &context, const PhysicalLimit &op) : current_offset(0), data(op.types) {
		this->limit = op.limit_expression ? DConstants::INVALID_INDEX : op.limit_value;
		this->offset = op.offset_expression ? DConstants::INVALID_INDEX : op.offset_value;
	}

	idx_t current_offset;
	idx_t limit;
	idx_t offset;
	BatchedDataCollection data;
};

unique_ptr<GlobalSinkState> PhysicalLimit::GetGlobalSinkState(ClientContext &context) const {
	return make_uniq<LimitGlobalState>(context, *this);
}

unique_ptr<LocalSinkState> PhysicalLimit::GetLocalSinkState(ExecutionContext &context) const {
	return make_uniq<LimitLocalState>(context.client, *this);
}

bool PhysicalLimit::ComputeOffset(ExecutionContext &context, DataChunk &input, idx_t &limit, idx_t &offset,
                                  idx_t current_offset, idx_t &max_element, Expression *limit_expression,
                                  Expression *offset_expression) {
	if (limit != DConstants::INVALID_INDEX && offset != DConstants::INVALID_INDEX) {
		max_element = limit + offset;
		if ((limit == 0 || current_offset >= max_element) && !(limit_expression || offset_expression)) {
			return false;
		}
	}

	// get the next chunk from the child
	if (limit == DConstants::INVALID_INDEX) {
		limit = 1ULL << 62ULL;
		Value val = GetDelimiter(context, input, limit_expression);
		if (!val.IsNull()) {
			limit = val.GetValue<idx_t>();
		}
		if (limit > 1ULL << 62ULL) {
			throw BinderException("Max value %lld for LIMIT/OFFSET is %lld", limit, 1ULL << 62ULL);
		}
	}
	if (offset == DConstants::INVALID_INDEX) {
		offset = 0;
		Value val = GetDelimiter(context, input, offset_expression);
		if (!val.IsNull()) {
			offset = val.GetValue<idx_t>();
		}
		if (offset > 1ULL << 62ULL) {
			throw BinderException("Max value %lld for LIMIT/OFFSET is %lld", offset, 1ULL << 62ULL);
		}
	}
	max_element = limit + offset;
	if (limit == 0 || current_offset >= max_element) {
		return false;
	}
	return true;
}

SinkResultType PhysicalLimit::Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const {

	D_ASSERT(chunk.size() > 0);
	auto &state = input.local_state.Cast<LimitLocalState>();
	auto &limit = state.limit;
	auto &offset = state.offset;

	idx_t max_element;
	if (!ComputeOffset(context, chunk, limit, offset, state.current_offset, max_element, limit_expression.get(),
	                   offset_expression.get())) {
		return SinkResultType::FINISHED;
	}
	auto max_cardinality = max_element - state.current_offset;
	if (max_cardinality < chunk.size()) {
		chunk.SetCardinality(max_cardinality);
	}
	state.data.Append(chunk, state.partition_info.batch_index.GetIndex());
	state.current_offset += chunk.size();
	if (state.current_offset == max_element) {
		return SinkResultType::FINISHED;
	}
	return SinkResultType::NEED_MORE_INPUT;
}

void PhysicalLimit::Combine(ExecutionContext &context, GlobalSinkState &gstate_p, LocalSinkState &lstate_p) const {
	auto &gstate = gstate_p.Cast<LimitGlobalState>();
	auto &state = lstate_p.Cast<LimitLocalState>();

	lock_guard<mutex> lock(gstate.glock);
	gstate.limit = state.limit;
	gstate.offset = state.offset;
	gstate.data.Merge(state.data);
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
class LimitSourceState : public GlobalSourceState {
public:
	LimitSourceState() {
		initialized = false;
		current_offset = 0;
	}

	bool initialized;
	idx_t current_offset;
	BatchedChunkScanState scan_state;
};

unique_ptr<GlobalSourceState> PhysicalLimit::GetGlobalSourceState(ClientContext &context) const {
	return make_uniq<LimitSourceState>();
}

SourceResultType PhysicalLimit::GetData(ExecutionContext &context, DataChunk &chunk, OperatorSourceInput &input) const {
	auto &gstate = sink_state->Cast<LimitGlobalState>();
	auto &state = input.global_state.Cast<LimitSourceState>();
	while (state.current_offset < gstate.limit + gstate.offset) {
		if (!state.initialized) {
			gstate.data.InitializeScan(state.scan_state);
			state.initialized = true;
		}
		gstate.data.Scan(state.scan_state, chunk);
		if (chunk.size() == 0) {
			return SourceResultType::FINISHED;
		}
		if (HandleOffset(chunk, state.current_offset, gstate.offset, gstate.limit)) {
			break;
		}
	}

	return chunk.size() > 0 ? SourceResultType::HAVE_MORE_OUTPUT : SourceResultType::FINISHED;
}

bool PhysicalLimit::HandleOffset(DataChunk &input, idx_t &current_offset, idx_t offset, idx_t limit) {
	idx_t max_element = limit + offset;
	if (limit == DConstants::INVALID_INDEX) {
		max_element = DConstants::INVALID_INDEX;
	}
	idx_t input_size = input.size();
	if (current_offset < offset) {
		// we are not yet at the offset point
		if (current_offset + input.size() > offset) {
			// however we will reach it in this chunk
			// we have to copy part of the chunk with an offset
			idx_t start_position = offset - current_offset;
			auto chunk_count = MinValue<idx_t>(limit, input.size() - start_position);
			SelectionVector sel(STANDARD_VECTOR_SIZE);
			for (idx_t i = 0; i < chunk_count; i++) {
				sel.set_index(i, start_position + i);
			}
			// set up a slice of the input chunks
			input.Slice(input, sel, chunk_count);
		} else {
			current_offset += input_size;
			return false;
		}
	} else {
		// have to copy either the entire chunk or part of it
		idx_t chunk_count;
		if (current_offset + input.size() >= max_element) {
			// have to limit the count of the chunk
			chunk_count = max_element - current_offset;
		} else {
			// we copy the entire chunk
			chunk_count = input.size();
		}
		// instead of copying we just change the pointer in the current chunk
		input.Reference(input);
		input.SetCardinality(chunk_count);
	}

	current_offset += input_size;
	return true;
}

Value PhysicalLimit::GetDelimiter(ExecutionContext &context, DataChunk &input, Expression *expr) {
	DataChunk limit_chunk;
	vector<LogicalType> types {expr->return_type};
	auto &allocator = Allocator::Get(context.client);
	limit_chunk.Initialize(allocator, types);
	ExpressionExecutor limit_executor(context.client, expr);
	auto input_size = input.size();
	input.SetCardinality(1);
	limit_executor.Execute(input, limit_chunk);
	input.SetCardinality(input_size);
	auto limit_value = limit_chunk.GetValue(0, 0);
	return limit_value;
}

} // namespace duckdb







namespace duckdb {

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
class LimitPercentGlobalState : public GlobalSinkState {
public:
	explicit LimitPercentGlobalState(ClientContext &context, const PhysicalLimitPercent &op)
	    : current_offset(0), data(context, op.GetTypes()) {
		if (!op.limit_expression) {
			this->limit_percent = op.limit_percent;
			is_limit_percent_delimited = true;
		} else {
			this->limit_percent = 100.0;
		}

		if (!op.offset_expression) {
			this->offset = op.offset_value;
			is_offset_delimited = true;
		} else {
			this->offset = 0;
		}
	}

	idx_t current_offset;
	double limit_percent;
	idx_t offset;
	ColumnDataCollection data;

	bool is_limit_percent_delimited = false;
	bool is_offset_delimited = false;
};

unique_ptr<GlobalSinkState> PhysicalLimitPercent::GetGlobalSinkState(ClientContext &context) const {
	return make_uniq<LimitPercentGlobalState>(context, *this);
}

SinkResultType PhysicalLimitPercent::Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const {
	D_ASSERT(chunk.size() > 0);
	auto &state = input.global_state.Cast<LimitPercentGlobalState>();
	auto &limit_percent = state.limit_percent;
	auto &offset = state.offset;

	// get the next chunk from the child
	if (!state.is_limit_percent_delimited) {
		Value val = PhysicalLimit::GetDelimiter(context, chunk, limit_expression.get());
		if (!val.IsNull()) {
			limit_percent = val.GetValue<double>();
		}
		if (limit_percent < 0.0) {
			throw BinderException("Percentage value(%f) can't be negative", limit_percent);
		}
		state.is_limit_percent_delimited = true;
	}
	if (!state.is_offset_delimited) {
		Value val = PhysicalLimit::GetDelimiter(context, chunk, offset_expression.get());
		if (!val.IsNull()) {
			offset = val.GetValue<idx_t>();
		}
		if (offset > 1ULL << 62ULL) {
			throw BinderException("Max value %lld for LIMIT/OFFSET is %lld", offset, 1ULL << 62ULL);
		}
		state.is_offset_delimited = true;
	}

	if (!PhysicalLimit::HandleOffset(chunk, state.current_offset, offset, DConstants::INVALID_INDEX)) {
		return SinkResultType::NEED_MORE_INPUT;
	}

	state.data.Append(chunk);
	return SinkResultType::NEED_MORE_INPUT;
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
class LimitPercentOperatorState : public GlobalSourceState {
public:
	explicit LimitPercentOperatorState(const PhysicalLimitPercent &op)
	    : limit(DConstants::INVALID_INDEX), current_offset(0) {
		D_ASSERT(op.sink_state);
		auto &gstate = op.sink_state->Cast<LimitPercentGlobalState>();
		gstate.data.InitializeScan(scan_state);
	}

	ColumnDataScanState scan_state;
	idx_t limit;
	idx_t current_offset;
};

unique_ptr<GlobalSourceState> PhysicalLimitPercent::GetGlobalSourceState(ClientContext &context) const {
	return make_uniq<LimitPercentOperatorState>(*this);
}

SourceResultType PhysicalLimitPercent::GetData(ExecutionContext &context, DataChunk &chunk,
                                               OperatorSourceInput &input) const {
	auto &gstate = sink_state->Cast<LimitPercentGlobalState>();
	auto &state = input.global_state.Cast<LimitPercentOperatorState>();
	auto &percent_limit = gstate.limit_percent;
	auto &offset = gstate.offset;
	auto &limit = state.limit;
	auto &current_offset = state.current_offset;

	if (gstate.is_limit_percent_delimited && limit == DConstants::INVALID_INDEX) {
		idx_t count = gstate.data.Count();
		if (count > 0) {
			count += offset;
		}
		if (Value::IsNan(percent_limit) || percent_limit < 0 || percent_limit > 100) {
			throw OutOfRangeException("Limit percent out of range, should be between 0% and 100%");
		}
		double limit_dbl = percent_limit / 100 * count;
		if (limit_dbl > count) {
			limit = count;
		} else {
			limit = idx_t(limit_dbl);
		}
		if (limit == 0) {
			return SourceResultType::FINISHED;
		}
	}

	if (current_offset >= limit) {
		return SourceResultType::FINISHED;
	}
	if (!gstate.data.Scan(state.scan_state, chunk)) {
		return SourceResultType::FINISHED;
	}

	PhysicalLimit::HandleOffset(chunk, current_offset, 0, limit);

	return SourceResultType::HAVE_MORE_OUTPUT;
}

} // namespace duckdb



namespace duckdb {

SourceResultType PhysicalLoad::GetData(ExecutionContext &context, DataChunk &chunk, OperatorSourceInput &input) const {
	if (info->load_type == LoadType::INSTALL || info->load_type == LoadType::FORCE_INSTALL) {
		ExtensionHelper::InstallExtension(context.client, info->filename, info->load_type == LoadType::FORCE_INSTALL);
	} else {
		ExtensionHelper::LoadExternalExtension(context.client, info->filename);
	}

	return SourceResultType::FINISHED;
}

} // namespace duckdb





namespace duckdb {

PhysicalMaterializedCollector::PhysicalMaterializedCollector(PreparedStatementData &data, bool parallel)
    : PhysicalResultCollector(data), parallel(parallel) {
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
class MaterializedCollectorGlobalState : public GlobalSinkState {
public:
	mutex glock;
	unique_ptr<ColumnDataCollection> collection;
	shared_ptr<ClientContext> context;
};

class MaterializedCollectorLocalState : public LocalSinkState {
public:
	unique_ptr<ColumnDataCollection> collection;
	ColumnDataAppendState append_state;
};

SinkResultType PhysicalMaterializedCollector::Sink(ExecutionContext &context, DataChunk &chunk,
                                                   OperatorSinkInput &input) const {
	auto &lstate = input.local_state.Cast<MaterializedCollectorLocalState>();
	lstate.collection->Append(lstate.append_state, chunk);
	return SinkResultType::NEED_MORE_INPUT;
}

void PhysicalMaterializedCollector::Combine(ExecutionContext &context, GlobalSinkState &gstate_p,
                                            LocalSinkState &lstate_p) const {
	auto &gstate = gstate_p.Cast<MaterializedCollectorGlobalState>();
	auto &lstate = lstate_p.Cast<MaterializedCollectorLocalState>();
	if (lstate.collection->Count() == 0) {
		return;
	}

	lock_guard<mutex> l(gstate.glock);
	if (!gstate.collection) {
		gstate.collection = std::move(lstate.collection);
	} else {
		gstate.collection->Combine(*lstate.collection);
	}
}

unique_ptr<GlobalSinkState> PhysicalMaterializedCollector::GetGlobalSinkState(ClientContext &context) const {
	auto state = make_uniq<MaterializedCollectorGlobalState>();
	state->context = context.shared_from_this();
	return std::move(state);
}

unique_ptr<LocalSinkState> PhysicalMaterializedCollector::GetLocalSinkState(ExecutionContext &context) const {
	auto state = make_uniq<MaterializedCollectorLocalState>();
	state->collection = make_uniq<ColumnDataCollection>(Allocator::DefaultAllocator(), types);
	state->collection->InitializeAppend(state->append_state);
	return std::move(state);
}

unique_ptr<QueryResult> PhysicalMaterializedCollector::GetResult(GlobalSinkState &state) {
	auto &gstate = state.Cast<MaterializedCollectorGlobalState>();
	if (!gstate.collection) {
		gstate.collection = make_uniq<ColumnDataCollection>(Allocator::DefaultAllocator(), types);
	}
	auto result = make_uniq<MaterializedQueryResult>(statement_type, properties, names, std::move(gstate.collection),
	                                                 gstate.context->GetClientProperties());
	return std::move(result);
}

bool PhysicalMaterializedCollector::ParallelSink() const {
	return parallel;
}

bool PhysicalMaterializedCollector::SinkOrderDependent() const {
	return true;
}

} // namespace duckdb


namespace duckdb {

SourceResultType PhysicalPragma::GetData(ExecutionContext &context, DataChunk &chunk,
                                         OperatorSourceInput &input) const {
	auto &client = context.client;
	FunctionParameters parameters {info.parameters, info.named_parameters};
	function.function(client, parameters);

	return SourceResultType::FINISHED;
}

} // namespace duckdb



namespace duckdb {

SourceResultType PhysicalPrepare::GetData(ExecutionContext &context, DataChunk &chunk,
                                          OperatorSourceInput &input) const {
	auto &client = context.client;

	// store the prepared statement in the context
	ClientData::Get(client).prepared_statements[name] = prepared;

	return SourceResultType::FINISHED;
}

} // namespace duckdb



namespace duckdb {

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
class SampleGlobalSinkState : public GlobalSinkState {
public:
	explicit SampleGlobalSinkState(Allocator &allocator, SampleOptions &options) {
		if (options.is_percentage) {
			auto percentage = options.sample_size.GetValue<double>();
			if (percentage == 0) {
				return;
			}
			sample = make_uniq<ReservoirSamplePercentage>(allocator, percentage, options.seed);
		} else {
			auto size = options.sample_size.GetValue<int64_t>();
			if (size == 0) {
				return;
			}
			sample = make_uniq<ReservoirSample>(allocator, size, options.seed);
		}
	}

	//! The lock for updating the global aggregate state
	mutex lock;
	//! The reservoir sample
	unique_ptr<BlockingSample> sample;
};

unique_ptr<GlobalSinkState> PhysicalReservoirSample::GetGlobalSinkState(ClientContext &context) const {
	return make_uniq<SampleGlobalSinkState>(Allocator::Get(context), *options);
}

SinkResultType PhysicalReservoirSample::Sink(ExecutionContext &context, DataChunk &chunk,
                                             OperatorSinkInput &input) const {
	auto &gstate = input.global_state.Cast<SampleGlobalSinkState>();
	if (!gstate.sample) {
		return SinkResultType::FINISHED;
	}
	// we implement reservoir sampling without replacement and exponential jumps here
	// the algorithm is adopted from the paper Weighted random sampling with a reservoir by Pavlos S. Efraimidis et al.
	// note that the original algorithm is about weighted sampling; this is a simplified approach for uniform sampling
	lock_guard<mutex> glock(gstate.lock);
	gstate.sample->AddToReservoir(chunk);
	return SinkResultType::NEED_MORE_INPUT;
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
SourceResultType PhysicalReservoirSample::GetData(ExecutionContext &context, DataChunk &chunk,
                                                  OperatorSourceInput &input) const {
	auto &sink = this->sink_state->Cast<SampleGlobalSinkState>();
	if (!sink.sample) {
		return SourceResultType::FINISHED;
	}
	auto sample_chunk = sink.sample->GetChunk();
	if (!sample_chunk) {
		return SourceResultType::FINISHED;
	}
	chunk.Move(*sample_chunk);

	return SourceResultType::HAVE_MORE_OUTPUT;
}

string PhysicalReservoirSample::ParamsToString() const {
	return options->sample_size.ToString() + (options->is_percentage ? "%" : " rows");
}

} // namespace duckdb






namespace duckdb {

void PhysicalReset::ResetExtensionVariable(ExecutionContext &context, DBConfig &config,
                                           ExtensionOption &extension_option) const {
	if (extension_option.set_function) {
		extension_option.set_function(context.client, scope, extension_option.default_value);
	}
	if (scope == SetScope::GLOBAL) {
		config.ResetOption(name);
	} else {
		auto &client_config = ClientConfig::GetConfig(context.client);
		client_config.set_variables[name] = extension_option.default_value;
	}
}

SourceResultType PhysicalReset::GetData(ExecutionContext &context, DataChunk &chunk, OperatorSourceInput &input) const {
	auto option = DBConfig::GetOptionByName(name);
	if (!option) {
		// check if this is an extra extension variable
		auto &config = DBConfig::GetConfig(context.client);
		auto entry = config.extension_parameters.find(name);
		if (entry == config.extension_parameters.end()) {
			throw Catalog::UnrecognizedConfigurationError(context.client, name);
		}
		ResetExtensionVariable(context, config, entry->second);
		return SourceResultType::FINISHED;
	}

	// Transform scope
	SetScope variable_scope = scope;
	if (variable_scope == SetScope::AUTOMATIC) {
		if (option->set_local) {
			variable_scope = SetScope::SESSION;
		} else {
			D_ASSERT(option->set_global);
			variable_scope = SetScope::GLOBAL;
		}
	}

	switch (variable_scope) {
	case SetScope::GLOBAL: {
		if (!option->set_global) {
			throw CatalogException("option \"%s\" cannot be reset globally", name);
		}
		auto &db = DatabaseInstance::GetDatabase(context.client);
		auto &config = DBConfig::GetConfig(context.client);
		config.ResetOption(&db, *option);
		break;
	}
	case SetScope::SESSION:
		if (!option->reset_local) {
			throw CatalogException("option \"%s\" cannot be reset locally", name);
		}
		option->reset_local(context.client);
		break;
	default:
		throw InternalException("Unsupported SetScope for variable");
	}

	return SourceResultType::FINISHED;
}

} // namespace duckdb










namespace duckdb {

PhysicalResultCollector::PhysicalResultCollector(PreparedStatementData &data)
    : PhysicalOperator(PhysicalOperatorType::RESULT_COLLECTOR, {LogicalType::BOOLEAN}, 0),
      statement_type(data.statement_type), properties(data.properties), plan(*data.plan), names(data.names) {
	this->types = data.types;
}

unique_ptr<PhysicalResultCollector> PhysicalResultCollector::GetResultCollector(ClientContext &context,
                                                                                PreparedStatementData &data) {
	if (!PhysicalPlanGenerator::PreserveInsertionOrder(context, *data.plan)) {
		// the plan is not order preserving, so we just use the parallel materialized collector
		return make_uniq_base<PhysicalResultCollector, PhysicalMaterializedCollector>(data, true);
	} else if (!PhysicalPlanGenerator::UseBatchIndex(context, *data.plan)) {
		// the plan is order preserving, but we cannot use the batch index: use a single-threaded result collector
		return make_uniq_base<PhysicalResultCollector, PhysicalMaterializedCollector>(data, false);
	} else {
		// we care about maintaining insertion order and the sources all support batch indexes
		// use a batch collector
		return make_uniq_base<PhysicalResultCollector, PhysicalBatchCollector>(data);
	}
}

vector<const_reference<PhysicalOperator>> PhysicalResultCollector::GetChildren() const {
	return {plan};
}

void PhysicalResultCollector::BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) {
	// operator is a sink, build a pipeline
	sink_state.reset();

	D_ASSERT(children.empty());

	// single operator: the operator becomes the data source of the current pipeline
	auto &state = meta_pipeline.GetState();
	state.SetPipelineSource(current, *this);

	// we create a new pipeline starting from the child
	auto &child_meta_pipeline = meta_pipeline.CreateChildMetaPipeline(current, *this);
	child_meta_pipeline.Build(plan);
}

} // namespace duckdb






namespace duckdb {

void PhysicalSet::SetExtensionVariable(ClientContext &context, ExtensionOption &extension_option, const string &name,
                                       SetScope scope, const Value &value) {
	auto &config = DBConfig::GetConfig(context);
	auto &target_type = extension_option.type;
	Value target_value = value.CastAs(context, target_type);
	if (extension_option.set_function) {
		extension_option.set_function(context, scope, target_value);
	}
	if (scope == SetScope::GLOBAL) {
		config.SetOption(name, std::move(target_value));
	} else {
		auto &client_config = ClientConfig::GetConfig(context);
		client_config.set_variables[name] = std::move(target_value);
	}
}

SourceResultType PhysicalSet::GetData(ExecutionContext &context, DataChunk &chunk, OperatorSourceInput &input) const {
	auto option = DBConfig::GetOptionByName(name);
	if (!option) {
		// check if this is an extra extension variable
		auto &config = DBConfig::GetConfig(context.client);
		auto entry = config.extension_parameters.find(name);
		if (entry == config.extension_parameters.end()) {
			throw Catalog::UnrecognizedConfigurationError(context.client, name);
		}
		SetExtensionVariable(context.client, entry->second, name, scope, value);
		return SourceResultType::FINISHED;
	}
	SetScope variable_scope = scope;
	if (variable_scope == SetScope::AUTOMATIC) {
		if (option->set_local) {
			variable_scope = SetScope::SESSION;
		} else {
			D_ASSERT(option->set_global);
			variable_scope = SetScope::GLOBAL;
		}
	}

	Value input_val = value.CastAs(context.client, option->parameter_type);
	switch (variable_scope) {
	case SetScope::GLOBAL: {
		if (!option->set_global) {
			throw CatalogException("option \"%s\" cannot be set globally", name);
		}
		auto &db = DatabaseInstance::GetDatabase(context.client);
		auto &config = DBConfig::GetConfig(context.client);
		config.SetOption(&db, *option, input_val);
		break;
	}
	case SetScope::SESSION:
		if (!option->set_local) {
			throw CatalogException("option \"%s\" cannot be set locally", name);
		}
		option->set_local(context.client, input_val);
		break;
	default:
		throw InternalException("Unsupported SetScope for variable");
	}

	return SourceResultType::FINISHED;
}

} // namespace duckdb



namespace duckdb {

PhysicalStreamingLimit::PhysicalStreamingLimit(vector<LogicalType> types, idx_t limit, idx_t offset,
                                               unique_ptr<Expression> limit_expression,
                                               unique_ptr<Expression> offset_expression, idx_t estimated_cardinality,
                                               bool parallel)
    : PhysicalOperator(PhysicalOperatorType::STREAMING_LIMIT, std::move(types), estimated_cardinality),
      limit_value(limit), offset_value(offset), limit_expression(std::move(limit_expression)),
      offset_expression(std::move(offset_expression)), parallel(parallel) {
}

//===--------------------------------------------------------------------===//
// Operator
//===--------------------------------------------------------------------===//
class StreamingLimitOperatorState : public OperatorState {
public:
	explicit StreamingLimitOperatorState(const PhysicalStreamingLimit &op) {
		this->limit = op.limit_expression ? DConstants::INVALID_INDEX : op.limit_value;
		this->offset = op.offset_expression ? DConstants::INVALID_INDEX : op.offset_value;
	}

	idx_t limit;
	idx_t offset;
};

class StreamingLimitGlobalState : public GlobalOperatorState {
public:
	StreamingLimitGlobalState() : current_offset(0) {
	}

	std::atomic<idx_t> current_offset;
};

unique_ptr<OperatorState> PhysicalStreamingLimit::GetOperatorState(ExecutionContext &context) const {
	return make_uniq<StreamingLimitOperatorState>(*this);
}

unique_ptr<GlobalOperatorState> PhysicalStreamingLimit::GetGlobalOperatorState(ClientContext &context) const {
	return make_uniq<StreamingLimitGlobalState>();
}

OperatorResultType PhysicalStreamingLimit::Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                                   GlobalOperatorState &gstate_p, OperatorState &state_p) const {
	auto &gstate = gstate_p.Cast<StreamingLimitGlobalState>();
	auto &state = state_p.Cast<StreamingLimitOperatorState>();
	auto &limit = state.limit;
	auto &offset = state.offset;
	idx_t current_offset = gstate.current_offset.fetch_add(input.size());
	idx_t max_element;
	if (!PhysicalLimit::ComputeOffset(context, input, limit, offset, current_offset, max_element,
	                                  limit_expression.get(), offset_expression.get())) {
		return OperatorResultType::FINISHED;
	}
	if (PhysicalLimit::HandleOffset(input, current_offset, offset, limit)) {
		chunk.Reference(input);
	}
	return OperatorResultType::NEED_MORE_INPUT;
}

OrderPreservationType PhysicalStreamingLimit::OperatorOrder() const {
	return OrderPreservationType::FIXED_ORDER;
}

bool PhysicalStreamingLimit::ParallelOperator() const {
	return parallel;
}

} // namespace duckdb





namespace duckdb {

PhysicalStreamingSample::PhysicalStreamingSample(vector<LogicalType> types, SampleMethod method, double percentage,
                                                 int64_t seed, idx_t estimated_cardinality)
    : PhysicalOperator(PhysicalOperatorType::STREAMING_SAMPLE, std::move(types), estimated_cardinality), method(method),
      percentage(percentage / 100), seed(seed) {
}

//===--------------------------------------------------------------------===//
// Operator
//===--------------------------------------------------------------------===//
class StreamingSampleOperatorState : public OperatorState {
public:
	explicit StreamingSampleOperatorState(int64_t seed) : random(seed) {
	}

	RandomEngine random;
};

void PhysicalStreamingSample::SystemSample(DataChunk &input, DataChunk &result, OperatorState &state_p) const {
	// system sampling: we throw one dice per chunk
	auto &state = state_p.Cast<StreamingSampleOperatorState>();
	double rand = state.random.NextRandom();
	if (rand <= percentage) {
		// rand is smaller than sample_size: output chunk
		result.Reference(input);
	}
}

void PhysicalStreamingSample::BernoulliSample(DataChunk &input, DataChunk &result, OperatorState &state_p) const {
	// bernoulli sampling: we throw one dice per tuple
	// then slice the result chunk
	auto &state = state_p.Cast<StreamingSampleOperatorState>();
	idx_t result_count = 0;
	SelectionVector sel(STANDARD_VECTOR_SIZE);
	for (idx_t i = 0; i < input.size(); i++) {
		double rand = state.random.NextRandom();
		if (rand <= percentage) {
			sel.set_index(result_count++, i);
		}
	}
	if (result_count > 0) {
		result.Slice(input, sel, result_count);
	}
}

unique_ptr<OperatorState> PhysicalStreamingSample::GetOperatorState(ExecutionContext &context) const {
	return make_uniq<StreamingSampleOperatorState>(seed);
}

OperatorResultType PhysicalStreamingSample::Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                                    GlobalOperatorState &gstate, OperatorState &state) const {
	switch (method) {
	case SampleMethod::BERNOULLI_SAMPLE:
		BernoulliSample(input, chunk, state);
		break;
	case SampleMethod::SYSTEM_SAMPLE:
		SystemSample(input, chunk, state);
		break;
	default:
		throw InternalException("Unsupported sample method for streaming sample");
	}
	return OperatorResultType::NEED_MORE_INPUT;
}

string PhysicalStreamingSample::ParamsToString() const {
	return EnumUtil::ToString(method) + ": " + to_string(100 * percentage) + "%";
}

} // namespace duckdb




namespace duckdb {

SourceResultType PhysicalTransaction::GetData(ExecutionContext &context, DataChunk &chunk,
                                              OperatorSourceInput &input) const {
	auto &client = context.client;

	auto type = info->type;
	if (type == TransactionType::COMMIT && ValidChecker::IsInvalidated(client.ActiveTransaction())) {
		// transaction is invalidated - turn COMMIT into ROLLBACK
		type = TransactionType::ROLLBACK;
	}
	switch (type) {
	case TransactionType::BEGIN_TRANSACTION: {
		if (client.transaction.IsAutoCommit()) {
			// start the active transaction
			// if autocommit is active, we have already called
			// BeginTransaction by setting autocommit to false we
			// prevent it from being closed after this query, hence
			// preserving the transaction context for the next query
			client.transaction.SetAutoCommit(false);
		} else {
			throw TransactionException("cannot start a transaction within a transaction");
		}
		break;
	}
	case TransactionType::COMMIT: {
		if (client.transaction.IsAutoCommit()) {
			throw TransactionException("cannot commit - no transaction is active");
		} else {
			// explicitly commit the current transaction
			client.transaction.Commit();
		}
		break;
	}
	case TransactionType::ROLLBACK: {
		if (client.transaction.IsAutoCommit()) {
			throw TransactionException("cannot rollback - no transaction is active");
		} else {
			// explicitly rollback the current transaction
			client.transaction.Rollback();
		}
		break;
	}
	default:
		throw NotImplementedException("Unrecognized transaction type!");
	}

	return SourceResultType::FINISHED;
}

} // namespace duckdb







namespace duckdb {

PhysicalVacuum::PhysicalVacuum(unique_ptr<VacuumInfo> info_p, idx_t estimated_cardinality)
    : PhysicalOperator(PhysicalOperatorType::VACUUM, {LogicalType::BOOLEAN}, estimated_cardinality),
      info(std::move(info_p)) {
}

class VacuumLocalSinkState : public LocalSinkState {
public:
	explicit VacuumLocalSinkState(VacuumInfo &info) {
		for (idx_t col_idx = 0; col_idx < info.columns.size(); col_idx++) {
			column_distinct_stats.push_back(make_uniq<DistinctStatistics>());
		}
	};

	vector<unique_ptr<DistinctStatistics>> column_distinct_stats;
};

unique_ptr<LocalSinkState> PhysicalVacuum::GetLocalSinkState(ExecutionContext &context) const {
	return make_uniq<VacuumLocalSinkState>(*info);
}

class VacuumGlobalSinkState : public GlobalSinkState {
public:
	explicit VacuumGlobalSinkState(VacuumInfo &info) {
		for (idx_t col_idx = 0; col_idx < info.columns.size(); col_idx++) {
			column_distinct_stats.push_back(make_uniq<DistinctStatistics>());
		}
	};

	mutex stats_lock;
	vector<unique_ptr<DistinctStatistics>> column_distinct_stats;
};

unique_ptr<GlobalSinkState> PhysicalVacuum::GetGlobalSinkState(ClientContext &context) const {
	return make_uniq<VacuumGlobalSinkState>(*info);
}

SinkResultType PhysicalVacuum::Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const {
	auto &lstate = input.local_state.Cast<VacuumLocalSinkState>();
	D_ASSERT(lstate.column_distinct_stats.size() == info->column_id_map.size());

	for (idx_t col_idx = 0; col_idx < chunk.data.size(); col_idx++) {
		if (!DistinctStatistics::TypeIsSupported(chunk.data[col_idx].GetType())) {
			continue;
		}
		lstate.column_distinct_stats[col_idx]->Update(chunk.data[col_idx], chunk.size(), false);
	}

	return SinkResultType::NEED_MORE_INPUT;
}

void PhysicalVacuum::Combine(ExecutionContext &context, GlobalSinkState &gstate_p, LocalSinkState &lstate_p) const {
	auto &gstate = gstate_p.Cast<VacuumGlobalSinkState>();
	auto &lstate = lstate_p.Cast<VacuumLocalSinkState>();

	lock_guard<mutex> lock(gstate.stats_lock);
	D_ASSERT(gstate.column_distinct_stats.size() == lstate.column_distinct_stats.size());
	for (idx_t col_idx = 0; col_idx < gstate.column_distinct_stats.size(); col_idx++) {
		gstate.column_distinct_stats[col_idx]->Merge(*lstate.column_distinct_stats[col_idx]);
	}
}

SinkFinalizeType PhysicalVacuum::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                          GlobalSinkState &gstate) const {
	auto &sink = gstate.Cast<VacuumGlobalSinkState>();

	auto table = info->table;
	for (idx_t col_idx = 0; col_idx < sink.column_distinct_stats.size(); col_idx++) {
		table->GetStorage().SetDistinct(info->column_id_map.at(col_idx),
		                                std::move(sink.column_distinct_stats[col_idx]));
	}

	return SinkFinalizeType::READY;
}

SourceResultType PhysicalVacuum::GetData(ExecutionContext &context, DataChunk &chunk,
                                         OperatorSourceInput &input) const {
	// NOP
	return SourceResultType::FINISHED;
}

} // namespace duckdb


namespace duckdb {

OuterJoinMarker::OuterJoinMarker(bool enabled_p) : enabled(enabled_p), count(0) {
}

void OuterJoinMarker::Initialize(idx_t count_p) {
	if (!enabled) {
		return;
	}
	this->count = count_p;
	found_match = make_unsafe_uniq_array<bool>(count);
	Reset();
}

void OuterJoinMarker::Reset() {
	if (!enabled) {
		return;
	}
	memset(found_match.get(), 0, sizeof(bool) * count);
}

void OuterJoinMarker::SetMatch(idx_t position) {
	if (!enabled) {
		return;
	}
	D_ASSERT(position < count);
	found_match[position] = true;
}

void OuterJoinMarker::SetMatches(const SelectionVector &sel, idx_t count, idx_t base_idx) {
	if (!enabled) {
		return;
	}
	for (idx_t i = 0; i < count; i++) {
		auto idx = sel.get_index(i);
		auto pos = base_idx + idx;
		D_ASSERT(pos < this->count);
		found_match[pos] = true;
	}
}

void OuterJoinMarker::ConstructLeftJoinResult(DataChunk &left, DataChunk &result) {
	if (!enabled) {
		return;
	}
	D_ASSERT(count == STANDARD_VECTOR_SIZE);
	SelectionVector remaining_sel(STANDARD_VECTOR_SIZE);
	idx_t remaining_count = 0;
	for (idx_t i = 0; i < left.size(); i++) {
		if (!found_match[i]) {
			remaining_sel.set_index(remaining_count++, i);
		}
	}
	if (remaining_count > 0) {
		result.Slice(left, remaining_sel, remaining_count);
		for (idx_t idx = left.ColumnCount(); idx < result.ColumnCount(); idx++) {
			result.data[idx].SetVectorType(VectorType::CONSTANT_VECTOR);
			ConstantVector::SetNull(result.data[idx], true);
		}
	}
}

idx_t OuterJoinMarker::MaxThreads() const {
	return count / (STANDARD_VECTOR_SIZE * 10);
}

void OuterJoinMarker::InitializeScan(ColumnDataCollection &data, OuterJoinGlobalScanState &gstate) {
	gstate.data = &data;
	data.InitializeScan(gstate.global_scan);
}

void OuterJoinMarker::InitializeScan(OuterJoinGlobalScanState &gstate, OuterJoinLocalScanState &lstate) {
	D_ASSERT(gstate.data);
	lstate.match_sel.Initialize(STANDARD_VECTOR_SIZE);
	gstate.data->InitializeScanChunk(lstate.scan_chunk);
}

void OuterJoinMarker::Scan(OuterJoinGlobalScanState &gstate, OuterJoinLocalScanState &lstate, DataChunk &result) {
	D_ASSERT(gstate.data);
	// fill in NULL values for the LHS
	while (gstate.data->Scan(gstate.global_scan, lstate.local_scan, lstate.scan_chunk)) {
		idx_t result_count = 0;
		// figure out which tuples didn't find a match in the RHS
		for (idx_t i = 0; i < lstate.scan_chunk.size(); i++) {
			if (!found_match[lstate.local_scan.current_row_index + i]) {
				lstate.match_sel.set_index(result_count++, i);
			}
		}
		if (result_count > 0) {
			// if there were any tuples that didn't find a match, output them
			idx_t left_column_count = result.ColumnCount() - lstate.scan_chunk.ColumnCount();
			for (idx_t i = 0; i < left_column_count; i++) {
				result.data[i].SetVectorType(VectorType::CONSTANT_VECTOR);
				ConstantVector::SetNull(result.data[i], true);
			}
			for (idx_t col_idx = left_column_count; col_idx < result.ColumnCount(); col_idx++) {
				result.data[col_idx].Slice(lstate.scan_chunk.data[col_idx - left_column_count], lstate.match_sel,
				                           result_count);
			}
			result.SetCardinality(result_count);
			return;
		}
	}
}

} // namespace duckdb





namespace duckdb {

PerfectHashJoinExecutor::PerfectHashJoinExecutor(const PhysicalHashJoin &join_p, JoinHashTable &ht_p,
                                                 PerfectHashJoinStats perfect_join_stats)
    : join(join_p), ht(ht_p), perfect_join_statistics(std::move(perfect_join_stats)) {
}

bool PerfectHashJoinExecutor::CanDoPerfectHashJoin() {
	return perfect_join_statistics.is_build_small;
}

//===--------------------------------------------------------------------===//
// Build
//===--------------------------------------------------------------------===//
bool PerfectHashJoinExecutor::BuildPerfectHashTable(LogicalType &key_type) {
	// First, allocate memory for each build column
	auto build_size = perfect_join_statistics.build_range + 1;
	for (const auto &type : ht.build_types) {
		perfect_hash_table.emplace_back(type, build_size);
	}

	// and for duplicate_checking
	bitmap_build_idx = make_unsafe_uniq_array<bool>(build_size);
	memset(bitmap_build_idx.get(), 0, sizeof(bool) * build_size); // set false

	// Now fill columns with build data

	return FullScanHashTable(key_type);
}

bool PerfectHashJoinExecutor::FullScanHashTable(LogicalType &key_type) {
	auto &data_collection = ht.GetDataCollection();

	// TODO: In a parallel finalize: One should exclusively lock and each thread should do one part of the code below.
	Vector tuples_addresses(LogicalType::POINTER, ht.Count()); // allocate space for all the tuples

	idx_t key_count = 0;
	if (data_collection.ChunkCount() > 0) {
		JoinHTScanState join_ht_state(data_collection, 0, data_collection.ChunkCount(),
		                              TupleDataPinProperties::KEEP_EVERYTHING_PINNED);

		// Go through all the blocks and fill the keys addresses
		key_count = ht.FillWithHTOffsets(join_ht_state, tuples_addresses);
	}

	// Scan the build keys in the hash table
	Vector build_vector(key_type, key_count);
	RowOperations::FullScanColumn(ht.layout, tuples_addresses, build_vector, key_count, 0);

	// Now fill the selection vector using the build keys and create a sequential vector
	// TODO: add check for fast pass when probe is part of build domain
	SelectionVector sel_build(key_count + 1);
	SelectionVector sel_tuples(key_count + 1);
	bool success = FillSelectionVectorSwitchBuild(build_vector, sel_build, sel_tuples, key_count);

	// early out
	if (!success) {
		return false;
	}
	if (unique_keys == perfect_join_statistics.build_range + 1 && !ht.has_null) {
		perfect_join_statistics.is_build_dense = true;
	}
	key_count = unique_keys; // do not consider keys out of the range

	// Full scan the remaining build columns and fill the perfect hash table
	const auto build_size = perfect_join_statistics.build_range + 1;
	for (idx_t i = 0; i < ht.build_types.size(); i++) {
		auto &vector = perfect_hash_table[i];
		D_ASSERT(vector.GetType() == ht.build_types[i]);
		if (build_size > STANDARD_VECTOR_SIZE) {
			auto &col_mask = FlatVector::Validity(vector);
			col_mask.Initialize(build_size);
		}

		const auto col_no = ht.condition_types.size() + i;
		data_collection.Gather(tuples_addresses, sel_tuples, key_count, col_no, vector, sel_build);
	}

	return true;
}

bool PerfectHashJoinExecutor::FillSelectionVectorSwitchBuild(Vector &source, SelectionVector &sel_vec,
                                                             SelectionVector &seq_sel_vec, idx_t count) {
	switch (source.GetType().InternalType()) {
	case PhysicalType::INT8:
		return TemplatedFillSelectionVectorBuild<int8_t>(source, sel_vec, seq_sel_vec, count);
	case PhysicalType::INT16:
		return TemplatedFillSelectionVectorBuild<int16_t>(source, sel_vec, seq_sel_vec, count);
	case PhysicalType::INT32:
		return TemplatedFillSelectionVectorBuild<int32_t>(source, sel_vec, seq_sel_vec, count);
	case PhysicalType::INT64:
		return TemplatedFillSelectionVectorBuild<int64_t>(source, sel_vec, seq_sel_vec, count);
	case PhysicalType::UINT8:
		return TemplatedFillSelectionVectorBuild<uint8_t>(source, sel_vec, seq_sel_vec, count);
	case PhysicalType::UINT16:
		return TemplatedFillSelectionVectorBuild<uint16_t>(source, sel_vec, seq_sel_vec, count);
	case PhysicalType::UINT32:
		return TemplatedFillSelectionVectorBuild<uint32_t>(source, sel_vec, seq_sel_vec, count);
	case PhysicalType::UINT64:
		return TemplatedFillSelectionVectorBuild<uint64_t>(source, sel_vec, seq_sel_vec, count);
	default:
		throw NotImplementedException("Type not supported for perfect hash join");
	}
}

template <typename T>
bool PerfectHashJoinExecutor::TemplatedFillSelectionVectorBuild(Vector &source, SelectionVector &sel_vec,
                                                                SelectionVector &seq_sel_vec, idx_t count) {
	if (perfect_join_statistics.build_min.IsNull() || perfect_join_statistics.build_max.IsNull()) {
		return false;
	}
	auto min_value = perfect_join_statistics.build_min.GetValueUnsafe<T>();
	auto max_value = perfect_join_statistics.build_max.GetValueUnsafe<T>();
	UnifiedVectorFormat vector_data;
	source.ToUnifiedFormat(count, vector_data);
	auto data = reinterpret_cast<T *>(vector_data.data);
	// generate the selection vector
	for (idx_t i = 0, sel_idx = 0; i < count; ++i) {
		auto data_idx = vector_data.sel->get_index(i);
		auto input_value = data[data_idx];
		// add index to selection vector if value in the range
		if (min_value <= input_value && input_value <= max_value) {
			auto idx = (idx_t)(input_value - min_value); // subtract min value to get the idx position
			sel_vec.set_index(sel_idx, idx);
			if (bitmap_build_idx[idx]) {
				return false;
			} else {
				bitmap_build_idx[idx] = true;
				unique_keys++;
			}
			seq_sel_vec.set_index(sel_idx++, i);
		}
	}
	return true;
}

//===--------------------------------------------------------------------===//
// Probe
//===--------------------------------------------------------------------===//
class PerfectHashJoinState : public OperatorState {
public:
	PerfectHashJoinState(ClientContext &context, const PhysicalHashJoin &join) : probe_executor(context) {
		join_keys.Initialize(Allocator::Get(context), join.condition_types);
		for (auto &cond : join.conditions) {
			probe_executor.AddExpression(*cond.left);
		}
		build_sel_vec.Initialize(STANDARD_VECTOR_SIZE);
		probe_sel_vec.Initialize(STANDARD_VECTOR_SIZE);
		seq_sel_vec.Initialize(STANDARD_VECTOR_SIZE);
	}

	DataChunk join_keys;
	ExpressionExecutor probe_executor;
	SelectionVector build_sel_vec;
	SelectionVector probe_sel_vec;
	SelectionVector seq_sel_vec;
};

unique_ptr<OperatorState> PerfectHashJoinExecutor::GetOperatorState(ExecutionContext &context) {
	auto state = make_uniq<PerfectHashJoinState>(context.client, join);
	return std::move(state);
}

OperatorResultType PerfectHashJoinExecutor::ProbePerfectHashTable(ExecutionContext &context, DataChunk &input,
                                                                  DataChunk &result, OperatorState &state_p) {
	auto &state = (PerfectHashJoinState &)state_p;
	// keeps track of how many probe keys have a match
	idx_t probe_sel_count = 0;

	// fetch the join keys from the chunk
	state.join_keys.Reset();
	state.probe_executor.Execute(input, state.join_keys);
	// select the keys that are in the min-max range
	auto &keys_vec = state.join_keys.data[0];
	auto keys_count = state.join_keys.size();
	// todo: add check for fast pass when probe is part of build domain
	FillSelectionVectorSwitchProbe(keys_vec, state.build_sel_vec, state.probe_sel_vec, keys_count, probe_sel_count);

	// If build is dense and probe is in build's domain, just reference probe
	if (perfect_join_statistics.is_build_dense && keys_count == probe_sel_count) {
		result.Reference(input);
	} else {
		// otherwise, filter it out the values that do not match
		result.Slice(input, state.probe_sel_vec, probe_sel_count, 0);
	}
	// on the build side, we need to fetch the data and build dictionary vectors with the sel_vec
	for (idx_t i = 0; i < ht.build_types.size(); i++) {
		auto &result_vector = result.data[input.ColumnCount() + i];
		D_ASSERT(result_vector.GetType() == ht.build_types[i]);
		auto &build_vec = perfect_hash_table[i];
		result_vector.Reference(build_vec);
		result_vector.Slice(state.build_sel_vec, probe_sel_count);
	}
	return OperatorResultType::NEED_MORE_INPUT;
}

void PerfectHashJoinExecutor::FillSelectionVectorSwitchProbe(Vector &source, SelectionVector &build_sel_vec,
                                                             SelectionVector &probe_sel_vec, idx_t count,
                                                             idx_t &probe_sel_count) {
	switch (source.GetType().InternalType()) {
	case PhysicalType::INT8:
		TemplatedFillSelectionVectorProbe<int8_t>(source, build_sel_vec, probe_sel_vec, count, probe_sel_count);
		break;
	case PhysicalType::INT16:
		TemplatedFillSelectionVectorProbe<int16_t>(source, build_sel_vec, probe_sel_vec, count, probe_sel_count);
		break;
	case PhysicalType::INT32:
		TemplatedFillSelectionVectorProbe<int32_t>(source, build_sel_vec, probe_sel_vec, count, probe_sel_count);
		break;
	case PhysicalType::INT64:
		TemplatedFillSelectionVectorProbe<int64_t>(source, build_sel_vec, probe_sel_vec, count, probe_sel_count);
		break;
	case PhysicalType::UINT8:
		TemplatedFillSelectionVectorProbe<uint8_t>(source, build_sel_vec, probe_sel_vec, count, probe_sel_count);
		break;
	case PhysicalType::UINT16:
		TemplatedFillSelectionVectorProbe<uint16_t>(source, build_sel_vec, probe_sel_vec, count, probe_sel_count);
		break;
	case PhysicalType::UINT32:
		TemplatedFillSelectionVectorProbe<uint32_t>(source, build_sel_vec, probe_sel_vec, count, probe_sel_count);
		break;
	case PhysicalType::UINT64:
		TemplatedFillSelectionVectorProbe<uint64_t>(source, build_sel_vec, probe_sel_vec, count, probe_sel_count);
		break;
	default:
		throw NotImplementedException("Type not supported");
	}
}

template <typename T>
void PerfectHashJoinExecutor::TemplatedFillSelectionVectorProbe(Vector &source, SelectionVector &build_sel_vec,
                                                                SelectionVector &probe_sel_vec, idx_t count,
                                                                idx_t &probe_sel_count) {
	auto min_value = perfect_join_statistics.build_min.GetValueUnsafe<T>();
	auto max_value = perfect_join_statistics.build_max.GetValueUnsafe<T>();

	UnifiedVectorFormat vector_data;
	source.ToUnifiedFormat(count, vector_data);
	auto data = reinterpret_cast<T *>(vector_data.data);
	auto validity_mask = &vector_data.validity;
	// build selection vector for non-dense build
	if (validity_mask->AllValid()) {
		for (idx_t i = 0, sel_idx = 0; i < count; ++i) {
			// retrieve value from vector
			auto data_idx = vector_data.sel->get_index(i);
			auto input_value = data[data_idx];
			// add index to selection vector if value in the range
			if (min_value <= input_value && input_value <= max_value) {
				auto idx = (idx_t)(input_value - min_value); // subtract min value to get the idx position
				                                             // check for matches in the build
				if (bitmap_build_idx[idx]) {
					build_sel_vec.set_index(sel_idx, idx);
					probe_sel_vec.set_index(sel_idx++, i);
					probe_sel_count++;
				}
			}
		}
	} else {
		for (idx_t i = 0, sel_idx = 0; i < count; ++i) {
			// retrieve value from vector
			auto data_idx = vector_data.sel->get_index(i);
			if (!validity_mask->RowIsValid(data_idx)) {
				continue;
			}
			auto input_value = data[data_idx];
			// add index to selection vector if value in the range
			if (min_value <= input_value && input_value <= max_value) {
				auto idx = (idx_t)(input_value - min_value); // subtract min value to get the idx position
				                                             // check for matches in the build
				if (bitmap_build_idx[idx]) {
					build_sel_vec.set_index(sel_idx, idx);
					probe_sel_vec.set_index(sel_idx++, i);
					probe_sel_count++;
				}
			}
		}
	}
}

} // namespace duckdb















namespace duckdb {

PhysicalAsOfJoin::PhysicalAsOfJoin(LogicalComparisonJoin &op, unique_ptr<PhysicalOperator> left,
                                   unique_ptr<PhysicalOperator> right)
    : PhysicalComparisonJoin(op, PhysicalOperatorType::ASOF_JOIN, std::move(op.conditions), op.join_type,
                             op.estimated_cardinality) {

	// Convert the conditions partitions and sorts
	for (auto &cond : conditions) {
		D_ASSERT(cond.left->return_type == cond.right->return_type);
		join_key_types.push_back(cond.left->return_type);

		auto left = cond.left->Copy();
		auto right = cond.right->Copy();
		switch (cond.comparison) {
		case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
			null_sensitive.emplace_back(lhs_orders.size());
			lhs_orders.emplace_back(OrderType::ASCENDING, OrderByNullType::NULLS_LAST, std::move(left));
			rhs_orders.emplace_back(OrderType::ASCENDING, OrderByNullType::NULLS_LAST, std::move(right));
			break;
		case ExpressionType::COMPARE_EQUAL:
			null_sensitive.emplace_back(lhs_orders.size());
			// Fall through
		case ExpressionType::COMPARE_NOT_DISTINCT_FROM:
			lhs_partitions.emplace_back(std::move(left));
			rhs_partitions.emplace_back(std::move(right));
			break;
		default:
			throw NotImplementedException("Unsupported join condition for ASOF join");
		}
	}
	D_ASSERT(!lhs_orders.empty());
	D_ASSERT(!rhs_orders.empty());

	children.push_back(std::move(left));
	children.push_back(std::move(right));

	//	Fill out the right projection map.
	right_projection_map = op.right_projection_map;
	if (right_projection_map.empty()) {
		const auto right_count = children[1]->types.size();
		right_projection_map.reserve(right_count);
		for (column_t i = 0; i < right_count; ++i) {
			right_projection_map.emplace_back(i);
		}
	}
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
class AsOfGlobalSinkState : public GlobalSinkState {
public:
	AsOfGlobalSinkState(ClientContext &context, const PhysicalAsOfJoin &op)
	    : global_partition(context, op.rhs_partitions, op.rhs_orders, op.children[1]->types, {},
	                       op.estimated_cardinality),
	      is_outer(IsRightOuterJoin(op.join_type)), has_null(false) {
	}

	idx_t Count() const {
		return global_partition.count;
	}

	PartitionGlobalSinkState global_partition;

	//	One per partition
	const bool is_outer;
	vector<OuterJoinMarker> right_outers;
	bool has_null;
};

class AsOfLocalSinkState : public LocalSinkState {
public:
	explicit AsOfLocalSinkState(ClientContext &context, PartitionGlobalSinkState &gstate_p)
	    : local_partition(context, gstate_p) {
	}

	void Sink(DataChunk &input_chunk) {
		local_partition.Sink(input_chunk);
	}

	void Combine() {
		local_partition.Combine();
	}

	PartitionLocalSinkState local_partition;
};

unique_ptr<GlobalSinkState> PhysicalAsOfJoin::GetGlobalSinkState(ClientContext &context) const {
	return make_uniq<AsOfGlobalSinkState>(context, *this);
}

unique_ptr<LocalSinkState> PhysicalAsOfJoin::GetLocalSinkState(ExecutionContext &context) const {
	// We only sink the RHS
	auto &gsink = sink_state->Cast<AsOfGlobalSinkState>();
	return make_uniq<AsOfLocalSinkState>(context.client, gsink.global_partition);
}

SinkResultType PhysicalAsOfJoin::Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const {
	auto &lstate = input.local_state.Cast<AsOfLocalSinkState>();

	lstate.Sink(chunk);

	return SinkResultType::NEED_MORE_INPUT;
}

void PhysicalAsOfJoin::Combine(ExecutionContext &context, GlobalSinkState &gstate_p, LocalSinkState &lstate_p) const {
	auto &lstate = lstate_p.Cast<AsOfLocalSinkState>();
	lstate.Combine();
}

//===--------------------------------------------------------------------===//
// Finalize
//===--------------------------------------------------------------------===//
SinkFinalizeType PhysicalAsOfJoin::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                            GlobalSinkState &gstate_p) const {
	auto &gstate = gstate_p.Cast<AsOfGlobalSinkState>();

	// Find the first group to sort
	auto &groups = gstate.global_partition.grouping_data->GetPartitions();
	if (groups.empty() && EmptyResultIfRHSIsEmpty()) {
		// Empty input!
		return SinkFinalizeType::NO_OUTPUT_POSSIBLE;
	}

	// Schedule all the sorts for maximum thread utilisation
	auto new_event = make_shared<PartitionMergeEvent>(gstate.global_partition, pipeline);
	event.InsertEvent(std::move(new_event));

	return SinkFinalizeType::READY;
}

//===--------------------------------------------------------------------===//
// Operator
//===--------------------------------------------------------------------===//
class AsOfGlobalState : public GlobalOperatorState {
public:
	explicit AsOfGlobalState(AsOfGlobalSinkState &gsink) {
		// for FULL/RIGHT OUTER JOIN, initialize right_outers to false for every tuple
		auto &global_partition = gsink.global_partition;
		auto &right_outers = gsink.right_outers;
		right_outers.reserve(global_partition.hash_groups.size());
		for (const auto &hash_group : global_partition.hash_groups) {
			right_outers.emplace_back(OuterJoinMarker(gsink.is_outer));
			right_outers.back().Initialize(hash_group->count);
		}
	}
};

unique_ptr<GlobalOperatorState> PhysicalAsOfJoin::GetGlobalOperatorState(ClientContext &context) const {
	auto &gsink = sink_state->Cast<AsOfGlobalSinkState>();
	return make_uniq<AsOfGlobalState>(gsink);
}

class AsOfLocalState : public CachingOperatorState {
public:
	using Orders = vector<BoundOrderByNode>;
	using Match = std::pair<hash_t, idx_t>;

	AsOfLocalState(ClientContext &context, const PhysicalAsOfJoin &op, bool force_external);

public:
	void ResolveJoin(DataChunk &input, bool *found_matches, Match *matches = nullptr);

	void ResolveJoinKeys(DataChunk &input);

	ClientContext &context;
	Allocator &allocator;
	const PhysicalAsOfJoin &op;
	BufferManager &buffer_manager;
	const bool force_external;
	Orders lhs_orders;

	//	LHS sorting
	ExpressionExecutor lhs_executor;
	DataChunk lhs_keys;
	ValidityMask lhs_valid_mask;
	SelectionVector lhs_sel;
	idx_t lhs_valid;
	RowLayout lhs_layout;
	unique_ptr<GlobalSortState> lhs_global_state;
	DataChunk lhs_sorted;

	// LHS binning
	Vector hash_vector;
	Vector bin_vector;

	//	Output
	idx_t lhs_match_count;
	SelectionVector lhs_matched;
	OuterJoinMarker left_outer;
	bool fetch_next_left;
	DataChunk group_payload;
	DataChunk rhs_payload;
};

AsOfLocalState::AsOfLocalState(ClientContext &context, const PhysicalAsOfJoin &op, bool force_external)
    : context(context), allocator(Allocator::Get(context)), op(op),
      buffer_manager(BufferManager::GetBufferManager(context)), force_external(force_external), lhs_executor(context),
      hash_vector(LogicalType::HASH), bin_vector(LogicalType::HASH), left_outer(IsLeftOuterJoin(op.join_type)),
      fetch_next_left(true) {
	vector<unique_ptr<BaseStatistics>> partition_stats;
	Orders partitions; // Not used.
	PartitionGlobalSinkState::GenerateOrderings(partitions, lhs_orders, op.lhs_partitions, op.lhs_orders,
	                                            partition_stats);

	//	We sort the row numbers of the incoming block, not the rows
	lhs_layout.Initialize({LogicalType::UINTEGER});
	lhs_sorted.Initialize(allocator, lhs_layout.GetTypes());

	lhs_keys.Initialize(allocator, op.join_key_types);
	for (const auto &cond : op.conditions) {
		lhs_executor.AddExpression(*cond.left);
	}

	group_payload.Initialize(allocator, op.children[1]->types);
	rhs_payload.Initialize(allocator, op.children[1]->types);

	lhs_matched.Initialize();
	lhs_sel.Initialize();
	left_outer.Initialize(STANDARD_VECTOR_SIZE);
}

void AsOfLocalState::ResolveJoinKeys(DataChunk &input) {
	//	Compute the join keys
	lhs_keys.Reset();
	lhs_executor.Execute(input, lhs_keys);

	//	Extract the NULLs
	const auto count = input.size();
	lhs_valid_mask.Reset();
	for (auto col_idx : op.null_sensitive) {
		auto &col = lhs_keys.data[col_idx];
		UnifiedVectorFormat unified;
		col.ToUnifiedFormat(count, unified);
		lhs_valid_mask.Combine(unified.validity, count);
	}

	//	Convert the mask to a selection vector.
	//	We need this anyway for sorting
	lhs_valid = 0;
	const auto entry_count = lhs_valid_mask.EntryCount(count);
	idx_t base_idx = 0;
	for (idx_t entry_idx = 0; entry_idx < entry_count;) {
		const auto validity_entry = lhs_valid_mask.GetValidityEntry(entry_idx++);
		const auto next = MinValue<idx_t>(base_idx + ValidityMask::BITS_PER_VALUE, count);
		if (ValidityMask::AllValid(validity_entry)) {
			for (; base_idx < next; ++base_idx) {
				lhs_sel.set_index(lhs_valid++, base_idx);
			}
		} else if (ValidityMask::NoneValid(validity_entry)) {
			base_idx = next;
		} else {
			const auto start = base_idx;
			for (; base_idx < next; ++base_idx) {
				if (ValidityMask::RowIsValid(validity_entry, base_idx - start)) {
					lhs_sel.set_index(lhs_valid++, base_idx);
				}
			}
		}
	}

	//	Slice the keys to the ones we can match
	if (lhs_valid < count) {
		lhs_keys.Slice(lhs_sel, lhs_valid);
	}

	//	Hash to assign the partitions
	auto &global_partition = op.sink_state->Cast<AsOfGlobalSinkState>().global_partition;
	if (op.lhs_partitions.empty()) {
		// Only one hash group
		bin_vector.Reference(Value::HASH(0));
	} else {
		//	Hash to determine the partitions.
		VectorOperations::Hash(lhs_keys.data[0], hash_vector, lhs_sel, lhs_valid);
		for (size_t prt_idx = 1; prt_idx < op.lhs_partitions.size(); ++prt_idx) {
			VectorOperations::CombineHash(hash_vector, lhs_keys.data[prt_idx], lhs_sel, lhs_valid);
		}

		// Convert hashes to hash groups
		const auto radix_bits = global_partition.grouping_data->GetRadixBits();
		RadixPartitioning::HashesToBins(hash_vector, radix_bits, bin_vector, count);
	}

	// 	Sort the selection vector on the valid keys
	lhs_global_state = make_uniq<GlobalSortState>(buffer_manager, lhs_orders, lhs_layout);
	auto &global_state = *lhs_global_state;
	LocalSortState local_sort;
	local_sort.Initialize(*lhs_global_state, buffer_manager);

	DataChunk payload_chunk;
	payload_chunk.InitializeEmpty({LogicalType::UINTEGER});
	FlatVector::SetData(payload_chunk.data[0], (data_ptr_t)lhs_sel.data());
	payload_chunk.SetCardinality(lhs_valid);
	local_sort.SinkChunk(lhs_keys, payload_chunk);

	// Set external (can be forced with the PRAGMA)
	global_state.external = force_external;
	global_state.AddLocalState(local_sort);
	global_state.PrepareMergePhase();
	while (global_state.sorted_blocks.size() > 1) {
		MergeSorter merge_sorter(*lhs_global_state, buffer_manager);
		merge_sorter.PerformInMergeRound();
		global_state.CompleteMergeRound();
	}

	// Scan the sorted selection
	D_ASSERT(global_state.sorted_blocks.size() == 1);

	auto scanner = make_uniq<PayloadScanner>(*global_state.sorted_blocks[0]->payload_data, global_state, false);
	lhs_sorted.Reset();
	scanner->Scan(lhs_sorted);
}

void AsOfLocalState::ResolveJoin(DataChunk &input, bool *found_match, std::pair<hash_t, idx_t> *matches) {
	//	Sort the input into lhs_payload, radix keys in lhs_global_state
	ResolveJoinKeys(input);

	auto &gsink = op.sink_state->Cast<AsOfGlobalSinkState>();
	auto &global_partition = gsink.global_partition;

	// The bins are contiguous from sorting, so load them one at a time
	// But they may be constant, so unify.
	UnifiedVectorFormat bin_unified;
	bin_vector.ToUnifiedFormat(lhs_valid, bin_unified);
	const auto bins = (hash_t *)bin_unified.data;

	hash_t prev_bin = global_partition.bin_groups.size();
	optional_ptr<PartitionGlobalHashGroup> hash_group;
	optional_ptr<OuterJoinMarker> right_outer;
	//	Searching for right <= left
	SBIterator left(*lhs_global_state, ExpressionType::COMPARE_LESSTHANOREQUALTO);
	unique_ptr<SBIterator> right;
	lhs_match_count = 0;
	const auto sorted_sel = FlatVector::GetData<sel_t>(lhs_sorted.data[0]);
	for (idx_t i = 0; i < lhs_valid; ++i) {
		//	idx is the index in the input; i is the index in the sorted keys
		const auto idx = sorted_sel[i];
		const auto curr_bin = bins[bin_unified.sel->get_index(idx)];
		if (!hash_group || curr_bin != prev_bin) {
			//	Grab the next group
			prev_bin = curr_bin;
			const auto group_idx = global_partition.bin_groups[curr_bin];
			if (group_idx >= global_partition.hash_groups.size()) {
				//	No matching partition
				hash_group = nullptr;
				right_outer = nullptr;
				right.reset();
				continue;
			}
			hash_group = global_partition.hash_groups[group_idx].get();
			right_outer = gsink.right_outers.data() + group_idx;
			right = make_uniq<SBIterator>(*(hash_group->global_sort), ExpressionType::COMPARE_LESSTHANOREQUALTO);
		}
		left.SetIndex(i);

		//	If right > left, then there is no match
		if (!right->Compare(left)) {
			continue;
		}

		// Exponential search forward for a non-matching value using radix iterators
		// (We use exponential search to avoid thrashing the block manager on large probes)
		idx_t bound = 1;
		idx_t begin = right->GetIndex();
		right->SetIndex(begin + bound);
		while (right->GetIndex() < hash_group->count) {
			if (right->Compare(left)) {
				//	If right <= left, jump ahead
				bound *= 2;
				right->SetIndex(begin + bound);
			} else {
				break;
			}
		}

		//	Binary search for the first non-matching value using radix iterators
		//	The previous value (which we know exists) is the match
		auto first = begin + bound / 2;
		auto last = MinValue<idx_t>(begin + bound, hash_group->count);
		while (first < last) {
			const auto mid = first + (last - first) / 2;
			right->SetIndex(mid);
			if (right->Compare(left)) {
				//	If right <= left, new lower bound
				first = mid + 1;
			} else {
				last = mid;
			}
		}
		right->SetIndex(--first);

		//	Check partitions for strict equality
		if (!op.lhs_partitions.empty() && hash_group->ComparePartitions(left, *right)) {
			continue;
		}

		// Emit match data
		right_outer->SetMatch(first);
		left_outer.SetMatch(idx);
		if (found_match) {
			found_match[idx] = true;
		}
		if (matches) {
			matches[idx] = Match(curr_bin, first);
		}
		lhs_matched.set_index(lhs_match_count++, idx);
	}
}

unique_ptr<OperatorState> PhysicalAsOfJoin::GetOperatorState(ExecutionContext &context) const {
	auto &config = ClientConfig::GetConfig(context.client);
	return make_uniq<AsOfLocalState>(context.client, *this, config.force_external);
}

void PhysicalAsOfJoin::ResolveSimpleJoin(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                         OperatorState &lstate_p) const {
	auto &lstate = lstate_p.Cast<AsOfLocalState>();
	auto &gsink = sink_state->Cast<AsOfGlobalSinkState>();

	// perform the actual join
	bool found_match[STANDARD_VECTOR_SIZE] = {false};
	lstate.ResolveJoin(input, found_match);

	// now construct the result based on the join result
	switch (join_type) {
	case JoinType::MARK: {
		PhysicalJoin::ConstructMarkJoinResult(lstate.lhs_keys, input, chunk, found_match, gsink.has_null);
		break;
	}
	case JoinType::SEMI:
		PhysicalJoin::ConstructSemiJoinResult(input, chunk, found_match);
		break;
	case JoinType::ANTI:
		PhysicalJoin::ConstructAntiJoinResult(input, chunk, found_match);
		break;
	default:
		throw NotImplementedException("Unimplemented join type for AsOf join");
	}
}

OperatorResultType PhysicalAsOfJoin::ResolveComplexJoin(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                                        OperatorState &lstate_p) const {
	auto &lstate = lstate_p.Cast<AsOfLocalState>();
	auto &gsink = sink_state->Cast<AsOfGlobalSinkState>();

	if (!lstate.fetch_next_left) {
		lstate.fetch_next_left = true;
		if (lstate.left_outer.Enabled()) {
			// left join: before we move to the next chunk, see if we need to output any vectors that didn't
			// have a match found
			lstate.left_outer.ConstructLeftJoinResult(input, chunk);
			lstate.left_outer.Reset();
		}
		return OperatorResultType::NEED_MORE_INPUT;
	}

	// perform the actual join
	AsOfLocalState::Match matches[STANDARD_VECTOR_SIZE];
	lstate.ResolveJoin(input, nullptr, matches);
	lstate.group_payload.Reset();
	lstate.rhs_payload.Reset();

	auto &global_partition = gsink.global_partition;
	hash_t scan_bin = global_partition.bin_groups.size();
	optional_ptr<PartitionGlobalHashGroup> hash_group;
	unique_ptr<PayloadScanner> scanner;
	for (idx_t i = 0; i < lstate.lhs_match_count; ++i) {
		const auto idx = lstate.lhs_matched[i];
		const auto match_bin = matches[idx].first;
		const auto match_pos = matches[idx].second;
		if (match_bin != scan_bin) {
			//	Grab the next group
			const auto group_idx = global_partition.bin_groups[match_bin];
			hash_group = global_partition.hash_groups[group_idx].get();
			scan_bin = match_bin;
			scanner = make_uniq<PayloadScanner>(*hash_group->global_sort, false);
			lstate.group_payload.Reset();
		}
		// Skip to the range containing the match
		while (match_pos >= scanner->Scanned()) {
			lstate.group_payload.Reset();
			scanner->Scan(lstate.group_payload);
		}
		// Append the individual values
		// TODO: Batch the copies
		const auto source_offset = match_pos - (scanner->Scanned() - lstate.group_payload.size());
		for (idx_t col_idx = 0; col_idx < right_projection_map.size(); ++col_idx) {
			const auto rhs_idx = right_projection_map[col_idx];
			auto &source = lstate.group_payload.data[rhs_idx];
			auto &target = chunk.data[input.ColumnCount() + col_idx];
			VectorOperations::Copy(source, target, source_offset + 1, source_offset, i);
		}
	}

	//	Slice the input into the left side
	chunk.Slice(input, lstate.lhs_matched, lstate.lhs_match_count);

	//	If we are doing a left join, come back for the NULLs
	if (lstate.left_outer.Enabled()) {
		lstate.fetch_next_left = false;
		return OperatorResultType::HAVE_MORE_OUTPUT;
	}

	return OperatorResultType::NEED_MORE_INPUT;
}

OperatorResultType PhysicalAsOfJoin::ExecuteInternal(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                                     GlobalOperatorState &gstate, OperatorState &lstate) const {
	auto &gsink = sink_state->Cast<AsOfGlobalSinkState>();

	if (gsink.global_partition.count == 0) {
		// empty RHS
		if (!EmptyResultIfRHSIsEmpty()) {
			ConstructEmptyJoinResult(join_type, gsink.has_null, input, chunk);
			return OperatorResultType::NEED_MORE_INPUT;
		} else {
			return OperatorResultType::FINISHED;
		}
	}

	input.Verify();
	switch (join_type) {
	case JoinType::SEMI:
	case JoinType::ANTI:
	case JoinType::MARK:
		// simple joins can have max STANDARD_VECTOR_SIZE matches per chunk
		ResolveSimpleJoin(context, input, chunk, lstate);
		return OperatorResultType::NEED_MORE_INPUT;
	case JoinType::LEFT:
	case JoinType::INNER:
	case JoinType::RIGHT:
	case JoinType::OUTER:
		return ResolveComplexJoin(context, input, chunk, lstate);
	default:
		throw NotImplementedException("Unimplemented type for as-of join!");
	}
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
class AsOfGlobalSourceState : public GlobalSourceState {
public:
	explicit AsOfGlobalSourceState(PartitionGlobalSinkState &gsink_p) : gsink(gsink_p), next_bin(0) {
	}

	PartitionGlobalSinkState &gsink;
	//! The output read position.
	atomic<idx_t> next_bin;

public:
	idx_t MaxThreads() override {
		// If there is only one partition, we have to process it on one thread.
		if (!gsink.grouping_data) {
			return 1;
		}

		// If there is not a lot of data, process serially.
		if (gsink.count < STANDARD_ROW_GROUPS_SIZE) {
			return 1;
		}

		return gsink.hash_groups.size();
	}
};

unique_ptr<GlobalSourceState> PhysicalAsOfJoin::GetGlobalSourceState(ClientContext &context) const {
	auto &gsink = sink_state->Cast<AsOfGlobalSinkState>();
	return make_uniq<AsOfGlobalSourceState>(gsink.global_partition);
}

class AsOfLocalSourceState : public LocalSourceState {
public:
	using HashGroupPtr = unique_ptr<PartitionGlobalHashGroup>;

	explicit AsOfLocalSourceState(AsOfGlobalSinkState &gstate_p);

	idx_t GeneratePartition(const idx_t hash_bin);

	AsOfGlobalSinkState &gstate;

	//! The read partition
	idx_t hash_bin;
	HashGroupPtr hash_group;

	//! The read cursor
	unique_ptr<PayloadScanner> scanner;
	//! Buffer for the inputs
	DataChunk input_chunk;
	//! Pointer to the matches
	const bool *found_match;
};

AsOfLocalSourceState::AsOfLocalSourceState(AsOfGlobalSinkState &gstate_p) : gstate(gstate_p) {
	input_chunk.Initialize(gstate.global_partition.allocator, gstate.global_partition.payload_types);
}

idx_t AsOfLocalSourceState::GeneratePartition(const idx_t hash_bin_p) {
	//	Get rid of any stale data
	hash_bin = hash_bin_p;

	hash_group = std::move(gstate.global_partition.hash_groups[hash_bin]);
	scanner = make_uniq<PayloadScanner>(*hash_group->global_sort);
	found_match = gstate.right_outers[hash_bin].GetMatches();

	return scanner->Remaining();
}

unique_ptr<LocalSourceState> PhysicalAsOfJoin::GetLocalSourceState(ExecutionContext &context,
                                                                   GlobalSourceState &gstate) const {
	auto &gsink = sink_state->Cast<AsOfGlobalSinkState>();
	return make_uniq<AsOfLocalSourceState>(gsink);
}

SourceResultType PhysicalAsOfJoin::GetData(ExecutionContext &context, DataChunk &chunk,
                                           OperatorSourceInput &input) const {
	D_ASSERT(IsRightOuterJoin(join_type));

	auto &gsource = input.global_state.Cast<AsOfGlobalSourceState>();
	auto &lsource = input.local_state.Cast<AsOfLocalSourceState>();
	auto &gsink = gsource.gsink;

	auto &hash_groups = gsink.hash_groups;
	const auto bin_count = hash_groups.size();

	DataChunk rhs_chunk;
	rhs_chunk.Initialize(Allocator::Get(context.client), gsink.payload_types);
	SelectionVector rsel(STANDARD_VECTOR_SIZE);

	while (chunk.size() == 0) {
		//	Move to the next bin if we are done.
		while (!lsource.scanner || !lsource.scanner->Remaining()) {
			lsource.scanner.reset();
			lsource.hash_group.reset();
			auto hash_bin = gsource.next_bin++;
			if (hash_bin >= bin_count) {
				return SourceResultType::FINISHED;
			}

			for (; hash_bin < hash_groups.size(); hash_bin = gsource.next_bin++) {
				if (hash_groups[hash_bin]) {
					break;
				}
			}
			lsource.GeneratePartition(hash_bin);
		}
		const auto rhs_position = lsource.scanner->Scanned();
		lsource.scanner->Scan(rhs_chunk);

		const auto count = rhs_chunk.size();
		if (count == 0) {
			return SourceResultType::FINISHED;
		}

		// figure out which tuples didn't find a match in the RHS
		auto found_match = lsource.found_match;
		idx_t result_count = 0;
		for (idx_t i = 0; i < count; i++) {
			if (!found_match[rhs_position + i]) {
				rsel.set_index(result_count++, i);
			}
		}

		if (result_count > 0) {
			// if there were any tuples that didn't find a match, output them
			const idx_t left_column_count = children[0]->types.size();
			for (idx_t col_idx = 0; col_idx < left_column_count; ++col_idx) {
				chunk.data[col_idx].SetVectorType(VectorType::CONSTANT_VECTOR);
				ConstantVector::SetNull(chunk.data[col_idx], true);
			}
			for (idx_t col_idx = 0; col_idx < right_projection_map.size(); ++col_idx) {
				const auto rhs_idx = right_projection_map[col_idx];
				chunk.data[left_column_count + col_idx].Slice(rhs_chunk.data[rhs_idx], rsel, result_count);
			}
			chunk.SetCardinality(result_count);
			break;
		}
	}

	return chunk.size() > 0 ? SourceResultType::HAVE_MORE_OUTPUT : SourceResultType::FINISHED;
}

} // namespace duckdb










namespace duckdb {

PhysicalBlockwiseNLJoin::PhysicalBlockwiseNLJoin(LogicalOperator &op, unique_ptr<PhysicalOperator> left,
                                                 unique_ptr<PhysicalOperator> right, unique_ptr<Expression> condition,
                                                 JoinType join_type, idx_t estimated_cardinality)
    : PhysicalJoin(op, PhysicalOperatorType::BLOCKWISE_NL_JOIN, join_type, estimated_cardinality),
      condition(std::move(condition)) {
	children.push_back(std::move(left));
	children.push_back(std::move(right));
	// MARK and SINGLE joins not handled
	D_ASSERT(join_type != JoinType::MARK);
	D_ASSERT(join_type != JoinType::SINGLE);
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
class BlockwiseNLJoinLocalState : public LocalSinkState {
public:
	BlockwiseNLJoinLocalState() {
	}
};

class BlockwiseNLJoinGlobalState : public GlobalSinkState {
public:
	explicit BlockwiseNLJoinGlobalState(ClientContext &context, const PhysicalBlockwiseNLJoin &op)
	    : right_chunks(context, op.children[1]->GetTypes()), right_outer(IsRightOuterJoin(op.join_type)) {
	}

	mutex lock;
	ColumnDataCollection right_chunks;
	OuterJoinMarker right_outer;
};

unique_ptr<GlobalSinkState> PhysicalBlockwiseNLJoin::GetGlobalSinkState(ClientContext &context) const {
	return make_uniq<BlockwiseNLJoinGlobalState>(context, *this);
}

unique_ptr<LocalSinkState> PhysicalBlockwiseNLJoin::GetLocalSinkState(ExecutionContext &context) const {
	return make_uniq<BlockwiseNLJoinLocalState>();
}

SinkResultType PhysicalBlockwiseNLJoin::Sink(ExecutionContext &context, DataChunk &chunk,
                                             OperatorSinkInput &input) const {
	auto &gstate = input.global_state.Cast<BlockwiseNLJoinGlobalState>();
	lock_guard<mutex> nl_lock(gstate.lock);
	gstate.right_chunks.Append(chunk);
	return SinkResultType::NEED_MORE_INPUT;
}

//===--------------------------------------------------------------------===//
// Finalize
//===--------------------------------------------------------------------===//
SinkFinalizeType PhysicalBlockwiseNLJoin::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                                   GlobalSinkState &gstate_p) const {
	auto &gstate = gstate_p.Cast<BlockwiseNLJoinGlobalState>();
	gstate.right_outer.Initialize(gstate.right_chunks.Count());

	if (gstate.right_chunks.Count() == 0 && EmptyResultIfRHSIsEmpty()) {
		return SinkFinalizeType::NO_OUTPUT_POSSIBLE;
	}
	return SinkFinalizeType::READY;
}

//===--------------------------------------------------------------------===//
// Operator
//===--------------------------------------------------------------------===//
class BlockwiseNLJoinState : public CachingOperatorState {
public:
	explicit BlockwiseNLJoinState(ExecutionContext &context, ColumnDataCollection &rhs,
	                              const PhysicalBlockwiseNLJoin &op)
	    : cross_product(rhs), left_outer(IsLeftOuterJoin(op.join_type)), match_sel(STANDARD_VECTOR_SIZE),
	      executor(context.client, *op.condition) {
		left_outer.Initialize(STANDARD_VECTOR_SIZE);
	}

	CrossProductExecutor cross_product;
	OuterJoinMarker left_outer;
	SelectionVector match_sel;
	ExpressionExecutor executor;
	DataChunk intermediate_chunk;
};

unique_ptr<OperatorState> PhysicalBlockwiseNLJoin::GetOperatorState(ExecutionContext &context) const {
	auto &gstate = sink_state->Cast<BlockwiseNLJoinGlobalState>();
	auto result = make_uniq<BlockwiseNLJoinState>(context, gstate.right_chunks, *this);
	if (join_type == JoinType::SEMI || join_type == JoinType::ANTI) {
		vector<LogicalType> intermediate_types;
		for (auto &type : children[0]->types) {
			intermediate_types.emplace_back(type);
		}
		for (auto &type : children[1]->types) {
			intermediate_types.emplace_back(type);
		}
		result->intermediate_chunk.Initialize(Allocator::DefaultAllocator(), intermediate_types);
	}
	return std::move(result);
}

OperatorResultType PhysicalBlockwiseNLJoin::ExecuteInternal(ExecutionContext &context, DataChunk &input,
                                                            DataChunk &chunk, GlobalOperatorState &gstate_p,
                                                            OperatorState &state_p) const {
	D_ASSERT(input.size() > 0);
	auto &state = (BlockwiseNLJoinState &)state_p;
	auto &gstate = sink_state->Cast<BlockwiseNLJoinGlobalState>();

	if (gstate.right_chunks.Count() == 0) {
		// empty RHS
		if (!EmptyResultIfRHSIsEmpty()) {
			PhysicalComparisonJoin::ConstructEmptyJoinResult(join_type, false, input, chunk);
			return OperatorResultType::NEED_MORE_INPUT;
		} else {
			return OperatorResultType::FINISHED;
		}
	}

	DataChunk *intermediate_chunk = &chunk;
	if (join_type == JoinType::SEMI || join_type == JoinType::ANTI) {
		intermediate_chunk = &state.intermediate_chunk;
		intermediate_chunk->Reset();
	}

	// now perform the actual join
	// we perform a cross product, then execute the expression directly on the cross product result
	idx_t result_count = 0;
	bool found_match[STANDARD_VECTOR_SIZE] = {false};

	do {
		auto result = state.cross_product.Execute(input, *intermediate_chunk);
		if (result == OperatorResultType::NEED_MORE_INPUT) {
			// exhausted input, have to pull new LHS chunk
			if (state.left_outer.Enabled()) {
				// left join: before we move to the next chunk, see if we need to output any vectors that didn't
				// have a match found
				state.left_outer.ConstructLeftJoinResult(input, *intermediate_chunk);
				state.left_outer.Reset();
			}

			if (join_type == JoinType::SEMI) {
				PhysicalJoin::ConstructSemiJoinResult(input, chunk, found_match);
			}
			if (join_type == JoinType::ANTI) {
				PhysicalJoin::ConstructAntiJoinResult(input, chunk, found_match);
			}

			return OperatorResultType::NEED_MORE_INPUT;
		}

		// now perform the computation
		result_count = state.executor.SelectExpression(*intermediate_chunk, state.match_sel);

		// handle anti and semi joins with different logic
		if (result_count > 0) {
			// found a match!
			// handle anti semi join conditions first
			if (join_type == JoinType::ANTI || join_type == JoinType::SEMI) {
				if (state.cross_product.ScanLHS()) {
					found_match[state.cross_product.PositionInChunk()] = true;
				} else {
					for (idx_t i = 0; i < result_count; i++) {
						found_match[state.match_sel.get_index(i)] = true;
					}
				}
				intermediate_chunk->Reset();
				// trick the loop to continue as semi and anti joins will never produce more output than
				// the LHS cardinality
				result_count = 0;
			} else {
				// check if the cross product is scanning the LHS or the RHS in its entirety
				if (!state.cross_product.ScanLHS()) {
					// set the match flags in the LHS
					state.left_outer.SetMatches(state.match_sel, result_count);
					// set the match flag in the RHS
					gstate.right_outer.SetMatch(state.cross_product.ScanPosition() +
					                            state.cross_product.PositionInChunk());
				} else {
					// set the match flag in the LHS
					state.left_outer.SetMatch(state.cross_product.PositionInChunk());
					// set the match flags in the RHS
					gstate.right_outer.SetMatches(state.match_sel, result_count, state.cross_product.ScanPosition());
				}
				intermediate_chunk->Slice(state.match_sel, result_count);
			}
		} else {
			// no result: reset the chunk
			intermediate_chunk->Reset();
		}
	} while (result_count == 0);

	return OperatorResultType::HAVE_MORE_OUTPUT;
}

string PhysicalBlockwiseNLJoin::ParamsToString() const {
	string extra_info = EnumUtil::ToString(join_type) + "\n";
	extra_info += condition->GetName();
	return extra_info;
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
class BlockwiseNLJoinGlobalScanState : public GlobalSourceState {
public:
	explicit BlockwiseNLJoinGlobalScanState(const PhysicalBlockwiseNLJoin &op) : op(op) {
		D_ASSERT(op.sink_state);
		auto &sink = op.sink_state->Cast<BlockwiseNLJoinGlobalState>();
		sink.right_outer.InitializeScan(sink.right_chunks, scan_state);
	}

	const PhysicalBlockwiseNLJoin &op;
	OuterJoinGlobalScanState scan_state;

public:
	idx_t MaxThreads() override {
		auto &sink = op.sink_state->Cast<BlockwiseNLJoinGlobalState>();
		return sink.right_outer.MaxThreads();
	}
};

class BlockwiseNLJoinLocalScanState : public LocalSourceState {
public:
	explicit BlockwiseNLJoinLocalScanState(const PhysicalBlockwiseNLJoin &op, BlockwiseNLJoinGlobalScanState &gstate) {
		D_ASSERT(op.sink_state);
		auto &sink = op.sink_state->Cast<BlockwiseNLJoinGlobalState>();
		sink.right_outer.InitializeScan(gstate.scan_state, scan_state);
	}

	OuterJoinLocalScanState scan_state;
};

unique_ptr<GlobalSourceState> PhysicalBlockwiseNLJoin::GetGlobalSourceState(ClientContext &context) const {
	return make_uniq<BlockwiseNLJoinGlobalScanState>(*this);
}

unique_ptr<LocalSourceState> PhysicalBlockwiseNLJoin::GetLocalSourceState(ExecutionContext &context,
                                                                          GlobalSourceState &gstate) const {
	return make_uniq<BlockwiseNLJoinLocalScanState>(*this, (BlockwiseNLJoinGlobalScanState &)gstate);
}

SourceResultType PhysicalBlockwiseNLJoin::GetData(ExecutionContext &context, DataChunk &chunk,
                                                  OperatorSourceInput &input) const {
	D_ASSERT(IsRightOuterJoin(join_type));
	// check if we need to scan any unmatched tuples from the RHS for the full/right outer join
	auto &sink = sink_state->Cast<BlockwiseNLJoinGlobalState>();
	auto &gstate = (BlockwiseNLJoinGlobalScanState &)input.global_state;
	auto &lstate = (BlockwiseNLJoinLocalScanState &)input.local_state;

	// if the LHS is exhausted in a FULL/RIGHT OUTER JOIN, we scan chunks we still need to output
	sink.right_outer.Scan(gstate.scan_state, lstate.scan_state, chunk);

	return chunk.size() == 0 ? SourceResultType::FINISHED : SourceResultType::HAVE_MORE_OUTPUT;
}

} // namespace duckdb




namespace duckdb {

PhysicalComparisonJoin::PhysicalComparisonJoin(LogicalOperator &op, PhysicalOperatorType type,
                                               vector<JoinCondition> conditions_p, JoinType join_type,
                                               idx_t estimated_cardinality)
    : PhysicalJoin(op, type, join_type, estimated_cardinality) {
	conditions.resize(conditions_p.size());
	// we reorder conditions so the ones with COMPARE_EQUAL occur first
	idx_t equal_position = 0;
	idx_t other_position = conditions_p.size() - 1;
	for (idx_t i = 0; i < conditions_p.size(); i++) {
		if (conditions_p[i].comparison == ExpressionType::COMPARE_EQUAL ||
		    conditions_p[i].comparison == ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
			// COMPARE_EQUAL and COMPARE_NOT_DISTINCT_FROM, move to the start
			conditions[equal_position++] = std::move(conditions_p[i]);
		} else {
			// other expression, move to the end
			conditions[other_position--] = std::move(conditions_p[i]);
		}
	}
}

string PhysicalComparisonJoin::ParamsToString() const {
	string extra_info = EnumUtil::ToString(join_type) + "\n";
	for (auto &it : conditions) {
		string op = ExpressionTypeToOperator(it.comparison);
		extra_info += it.left->GetName() + " " + op + " " + it.right->GetName() + "\n";
	}
	extra_info += "\n[INFOSEPARATOR]\n";
	extra_info += StringUtil::Format("EC: %llu\n", estimated_props->GetCardinality<idx_t>());
	extra_info += StringUtil::Format("Cost: %llu", (idx_t)estimated_props->GetCost());
	return extra_info;
}

void PhysicalComparisonJoin::ConstructEmptyJoinResult(JoinType join_type, bool has_null, DataChunk &input,
                                                      DataChunk &result) {
	// empty hash table, special case
	if (join_type == JoinType::ANTI) {
		// anti join with empty hash table, NOP join
		// return the input
		D_ASSERT(input.ColumnCount() == result.ColumnCount());
		result.Reference(input);
	} else if (join_type == JoinType::MARK) {
		// MARK join with empty hash table
		D_ASSERT(join_type == JoinType::MARK);
		D_ASSERT(result.ColumnCount() == input.ColumnCount() + 1);
		auto &result_vector = result.data.back();
		D_ASSERT(result_vector.GetType() == LogicalType::BOOLEAN);
		// for every data vector, we just reference the child chunk
		result.SetCardinality(input);
		for (idx_t i = 0; i < input.ColumnCount(); i++) {
			result.data[i].Reference(input.data[i]);
		}
		// for the MARK vector:
		// if the HT has no NULL values (i.e. empty result set), return a vector that has false for every input
		// entry if the HT has NULL values (i.e. result set had values, but all were NULL), return a vector that
		// has NULL for every input entry
		if (!has_null) {
			auto bool_result = FlatVector::GetData<bool>(result_vector);
			for (idx_t i = 0; i < result.size(); i++) {
				bool_result[i] = false;
			}
		} else {
			FlatVector::Validity(result_vector).SetAllInvalid(result.size());
		}
	} else if (join_type == JoinType::LEFT || join_type == JoinType::OUTER || join_type == JoinType::SINGLE) {
		// LEFT/FULL OUTER/SINGLE join and build side is empty
		// for the LHS we reference the data
		result.SetCardinality(input.size());
		for (idx_t i = 0; i < input.ColumnCount(); i++) {
			result.data[i].Reference(input.data[i]);
		}
		// for the RHS
		for (idx_t k = input.ColumnCount(); k < result.ColumnCount(); k++) {
			result.data[k].SetVectorType(VectorType::CONSTANT_VECTOR);
			ConstantVector::SetNull(result.data[k], true);
		}
	}
}
} // namespace duckdb






namespace duckdb {

PhysicalCrossProduct::PhysicalCrossProduct(vector<LogicalType> types, unique_ptr<PhysicalOperator> left,
                                           unique_ptr<PhysicalOperator> right, idx_t estimated_cardinality)
    : CachingPhysicalOperator(PhysicalOperatorType::CROSS_PRODUCT, std::move(types), estimated_cardinality) {
	children.push_back(std::move(left));
	children.push_back(std::move(right));
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
class CrossProductGlobalState : public GlobalSinkState {
public:
	explicit CrossProductGlobalState(ClientContext &context, const PhysicalCrossProduct &op)
	    : rhs_materialized(context, op.children[1]->GetTypes()) {
		rhs_materialized.InitializeAppend(append_state);
	}

	ColumnDataCollection rhs_materialized;
	ColumnDataAppendState append_state;
	mutex rhs_lock;
};

unique_ptr<GlobalSinkState> PhysicalCrossProduct::GetGlobalSinkState(ClientContext &context) const {
	return make_uniq<CrossProductGlobalState>(context, *this);
}

SinkResultType PhysicalCrossProduct::Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const {
	auto &sink = input.global_state.Cast<CrossProductGlobalState>();
	lock_guard<mutex> client_guard(sink.rhs_lock);
	sink.rhs_materialized.Append(sink.append_state, chunk);
	return SinkResultType::NEED_MORE_INPUT;
}

//===--------------------------------------------------------------------===//
// Operator
//===--------------------------------------------------------------------===//
CrossProductExecutor::CrossProductExecutor(ColumnDataCollection &rhs)
    : rhs(rhs), position_in_chunk(0), initialized(false), finished(false) {
	rhs.InitializeScanChunk(scan_chunk);
}

void CrossProductExecutor::Reset(DataChunk &input, DataChunk &output) {
	initialized = true;
	finished = false;
	scan_input_chunk = false;
	rhs.InitializeScan(scan_state);
	position_in_chunk = 0;
	scan_chunk.Reset();
}

bool CrossProductExecutor::NextValue(DataChunk &input, DataChunk &output) {
	if (!initialized) {
		// not initialized yet: initialize the scan
		Reset(input, output);
	}
	position_in_chunk++;
	idx_t chunk_size = scan_input_chunk ? input.size() : scan_chunk.size();
	if (position_in_chunk < chunk_size) {
		return true;
	}
	// fetch the next chunk
	rhs.Scan(scan_state, scan_chunk);
	position_in_chunk = 0;
	if (scan_chunk.size() == 0) {
		return false;
	}
	// the way the cross product works is that we keep one chunk constantly referenced
	// while iterating over the other chunk one value at a time
	// the second one is the chunk we are "scanning"

	// for the engine, it is better if we emit larger chunks
	// hence the chunk that we keep constantly referenced should be the larger of the two
	scan_input_chunk = input.size() < scan_chunk.size();
	return true;
}

OperatorResultType CrossProductExecutor::Execute(DataChunk &input, DataChunk &output) {
	if (rhs.Count() == 0) {
		// no RHS: empty result
		return OperatorResultType::FINISHED;
	}
	if (!NextValue(input, output)) {
		// ran out of entries on the RHS
		// reset the RHS and move to the next chunk on the LHS
		initialized = false;
		return OperatorResultType::NEED_MORE_INPUT;
	}

	// set up the constant chunk
	auto &constant_chunk = scan_input_chunk ? scan_chunk : input;
	auto col_count = constant_chunk.ColumnCount();
	auto col_offset = scan_input_chunk ? input.ColumnCount() : 0;
	output.SetCardinality(constant_chunk.size());
	for (idx_t i = 0; i < col_count; i++) {
		output.data[col_offset + i].Reference(constant_chunk.data[i]);
	}

	// for the chunk that we are scanning, scan a single value from that chunk
	auto &scan = scan_input_chunk ? input : scan_chunk;
	col_count = scan.ColumnCount();
	col_offset = scan_input_chunk ? 0 : input.ColumnCount();
	for (idx_t i = 0; i < col_count; i++) {
		ConstantVector::Reference(output.data[col_offset + i], scan.data[i], position_in_chunk, scan.size());
	}
	return OperatorResultType::HAVE_MORE_OUTPUT;
}

class CrossProductOperatorState : public CachingOperatorState {
public:
	explicit CrossProductOperatorState(ColumnDataCollection &rhs) : executor(rhs) {
	}

	CrossProductExecutor executor;
};

unique_ptr<OperatorState> PhysicalCrossProduct::GetOperatorState(ExecutionContext &context) const {
	auto &sink = sink_state->Cast<CrossProductGlobalState>();
	return make_uniq<CrossProductOperatorState>(sink.rhs_materialized);
}

OperatorResultType PhysicalCrossProduct::ExecuteInternal(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                                         GlobalOperatorState &gstate, OperatorState &state_p) const {
	auto &state = state_p.Cast<CrossProductOperatorState>();
	return state.executor.Execute(input, chunk);
}

//===--------------------------------------------------------------------===//
// Pipeline Construction
//===--------------------------------------------------------------------===//
void PhysicalCrossProduct::BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) {
	PhysicalJoin::BuildJoinPipelines(current, meta_pipeline, *this);
}

vector<const_reference<PhysicalOperator>> PhysicalCrossProduct::GetSources() const {
	return children[0]->GetSources();
}

} // namespace duckdb











namespace duckdb {

PhysicalDelimJoin::PhysicalDelimJoin(vector<LogicalType> types, unique_ptr<PhysicalOperator> original_join,
                                     vector<const_reference<PhysicalOperator>> delim_scans, idx_t estimated_cardinality)
    : PhysicalOperator(PhysicalOperatorType::DELIM_JOIN, std::move(types), estimated_cardinality),
      join(std::move(original_join)), delim_scans(std::move(delim_scans)) {
	D_ASSERT(join->children.size() == 2);
	// now for the original join
	// we take its left child, this is the side that we will duplicate eliminate
	children.push_back(std::move(join->children[0]));

	// we replace it with a PhysicalColumnDataScan, that scans the ColumnDataCollection that we keep cached
	// the actual chunk collection to scan will be created in the DelimJoinGlobalState
	auto cached_chunk_scan = make_uniq<PhysicalColumnDataScan>(
	    children[0]->GetTypes(), PhysicalOperatorType::COLUMN_DATA_SCAN, estimated_cardinality);
	join->children[0] = std::move(cached_chunk_scan);
}

vector<const_reference<PhysicalOperator>> PhysicalDelimJoin::GetChildren() const {
	vector<const_reference<PhysicalOperator>> result;
	for (auto &child : children) {
		result.push_back(*child);
	}
	result.push_back(*join);
	result.push_back(*distinct);
	return result;
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
class DelimJoinGlobalState : public GlobalSinkState {
public:
	explicit DelimJoinGlobalState(ClientContext &context, const PhysicalDelimJoin &delim_join)
	    : lhs_data(context, delim_join.children[0]->GetTypes()) {
		D_ASSERT(delim_join.delim_scans.size() > 0);
		// set up the delim join chunk to scan in the original join
		auto &cached_chunk_scan = delim_join.join->children[0]->Cast<PhysicalColumnDataScan>();
		cached_chunk_scan.collection = &lhs_data;
	}

	ColumnDataCollection lhs_data;
	mutex lhs_lock;

	void Merge(ColumnDataCollection &input) {
		lock_guard<mutex> guard(lhs_lock);
		lhs_data.Combine(input);
	}
};

class DelimJoinLocalState : public LocalSinkState {
public:
	explicit DelimJoinLocalState(ClientContext &context, const PhysicalDelimJoin &delim_join)
	    : lhs_data(context, delim_join.children[0]->GetTypes()) {
		lhs_data.InitializeAppend(append_state);
	}

	unique_ptr<LocalSinkState> distinct_state;
	ColumnDataCollection lhs_data;
	ColumnDataAppendState append_state;

	void Append(DataChunk &input) {
		lhs_data.Append(input);
	}
};

unique_ptr<GlobalSinkState> PhysicalDelimJoin::GetGlobalSinkState(ClientContext &context) const {
	auto state = make_uniq<DelimJoinGlobalState>(context, *this);
	distinct->sink_state = distinct->GetGlobalSinkState(context);
	if (delim_scans.size() > 1) {
		PhysicalHashAggregate::SetMultiScan(*distinct->sink_state);
	}
	return std::move(state);
}

unique_ptr<LocalSinkState> PhysicalDelimJoin::GetLocalSinkState(ExecutionContext &context) const {
	auto state = make_uniq<DelimJoinLocalState>(context.client, *this);
	state->distinct_state = distinct->GetLocalSinkState(context);
	return std::move(state);
}

SinkResultType PhysicalDelimJoin::Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const {
	auto &lstate = input.local_state.Cast<DelimJoinLocalState>();
	lstate.lhs_data.Append(lstate.append_state, chunk);
	OperatorSinkInput distinct_sink_input {*distinct->sink_state, *lstate.distinct_state, input.interrupt_state};
	distinct->Sink(context, chunk, distinct_sink_input);
	return SinkResultType::NEED_MORE_INPUT;
}

void PhysicalDelimJoin::Combine(ExecutionContext &context, GlobalSinkState &state, LocalSinkState &lstate_p) const {
	auto &lstate = lstate_p.Cast<DelimJoinLocalState>();
	auto &gstate = state.Cast<DelimJoinGlobalState>();
	gstate.Merge(lstate.lhs_data);
	distinct->Combine(context, *distinct->sink_state, *lstate.distinct_state);
}

SinkFinalizeType PhysicalDelimJoin::Finalize(Pipeline &pipeline, Event &event, ClientContext &client,
                                             GlobalSinkState &gstate) const {
	// finalize the distinct HT
	D_ASSERT(distinct);
	distinct->Finalize(pipeline, event, client, *distinct->sink_state);
	return SinkFinalizeType::READY;
}

string PhysicalDelimJoin::ParamsToString() const {
	return join->ParamsToString();
}

//===--------------------------------------------------------------------===//
// Pipeline Construction
//===--------------------------------------------------------------------===//
void PhysicalDelimJoin::BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) {
	op_state.reset();
	sink_state.reset();

	auto &child_meta_pipeline = meta_pipeline.CreateChildMetaPipeline(current, *this);
	child_meta_pipeline.Build(*children[0]);

	if (type == PhysicalOperatorType::DELIM_JOIN) {
		// recurse into the actual join
		// any pipelines in there depend on the main pipeline
		// any scan of the duplicate eliminated data on the RHS depends on this pipeline
		// we add an entry to the mapping of (PhysicalOperator*) -> (Pipeline*)
		auto &state = meta_pipeline.GetState();
		for (auto &delim_scan : delim_scans) {
			state.delim_join_dependencies.insert(
			    make_pair(delim_scan, reference<Pipeline>(*child_meta_pipeline.GetBasePipeline())));
		}
		join->BuildPipelines(current, meta_pipeline);
	}
}

} // namespace duckdb
















namespace duckdb {

PhysicalHashJoin::PhysicalHashJoin(LogicalOperator &op, unique_ptr<PhysicalOperator> left,
                                   unique_ptr<PhysicalOperator> right, vector<JoinCondition> cond, JoinType join_type,
                                   const vector<idx_t> &left_projection_map,
                                   const vector<idx_t> &right_projection_map_p, vector<LogicalType> delim_types,
                                   idx_t estimated_cardinality, PerfectHashJoinStats perfect_join_stats)
    : PhysicalComparisonJoin(op, PhysicalOperatorType::HASH_JOIN, std::move(cond), join_type, estimated_cardinality),
      right_projection_map(right_projection_map_p), delim_types(std::move(delim_types)),
      perfect_join_statistics(std::move(perfect_join_stats)) {

	children.push_back(std::move(left));
	children.push_back(std::move(right));

	D_ASSERT(left_projection_map.empty());
	for (auto &condition : conditions) {
		condition_types.push_back(condition.left->return_type);
	}

	// for ANTI, SEMI and MARK join, we only need to store the keys, so for these the build types are empty
	if (join_type != JoinType::ANTI && join_type != JoinType::SEMI && join_type != JoinType::MARK) {
		build_types = LogicalOperator::MapTypes(children[1]->GetTypes(), right_projection_map);
	}
}

PhysicalHashJoin::PhysicalHashJoin(LogicalOperator &op, unique_ptr<PhysicalOperator> left,
                                   unique_ptr<PhysicalOperator> right, vector<JoinCondition> cond, JoinType join_type,
                                   idx_t estimated_cardinality, PerfectHashJoinStats perfect_join_state)
    : PhysicalHashJoin(op, std::move(left), std::move(right), std::move(cond), join_type, {}, {}, {},
                       estimated_cardinality, std::move(perfect_join_state)) {
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
class HashJoinGlobalSinkState : public GlobalSinkState {
public:
	HashJoinGlobalSinkState(const PhysicalHashJoin &op, ClientContext &context_p)
	    : context(context_p), finalized(false), scanned_data(false) {
		hash_table = op.InitializeHashTable(context);

		// for perfect hash join
		perfect_join_executor = make_uniq<PerfectHashJoinExecutor>(op, *hash_table, op.perfect_join_statistics);
		// for external hash join
		external = ClientConfig::GetConfig(context).force_external;
		// Set probe types
		const auto &payload_types = op.children[0]->types;
		probe_types.insert(probe_types.end(), op.condition_types.begin(), op.condition_types.end());
		probe_types.insert(probe_types.end(), payload_types.begin(), payload_types.end());
		probe_types.emplace_back(LogicalType::HASH);
	}

	void ScheduleFinalize(Pipeline &pipeline, Event &event);
	void InitializeProbeSpill();

public:
	ClientContext &context;
	//! Global HT used by the join
	unique_ptr<JoinHashTable> hash_table;
	//! The perfect hash join executor (if any)
	unique_ptr<PerfectHashJoinExecutor> perfect_join_executor;
	//! Whether or not the hash table has been finalized
	bool finalized = false;

	//! Whether we are doing an external join
	bool external;

	//! Hash tables built by each thread
	mutex lock;
	vector<unique_ptr<JoinHashTable>> local_hash_tables;

	//! Excess probe data gathered during Sink
	vector<LogicalType> probe_types;
	unique_ptr<JoinHashTable::ProbeSpill> probe_spill;

	//! Whether or not we have started scanning data using GetData
	atomic<bool> scanned_data;
};

class HashJoinLocalSinkState : public LocalSinkState {
public:
	HashJoinLocalSinkState(const PhysicalHashJoin &op, ClientContext &context) : build_executor(context) {
		auto &allocator = Allocator::Get(context);
		if (!op.right_projection_map.empty()) {
			build_chunk.Initialize(allocator, op.build_types);
		}
		for (auto &cond : op.conditions) {
			build_executor.AddExpression(*cond.right);
		}
		join_keys.Initialize(allocator, op.condition_types);

		hash_table = op.InitializeHashTable(context);

		hash_table->GetSinkCollection().InitializeAppendState(append_state);
	}

public:
	PartitionedTupleDataAppendState append_state;

	DataChunk build_chunk;
	DataChunk join_keys;
	ExpressionExecutor build_executor;

	//! Thread-local HT
	unique_ptr<JoinHashTable> hash_table;
};

unique_ptr<JoinHashTable> PhysicalHashJoin::InitializeHashTable(ClientContext &context) const {
	auto result =
	    make_uniq<JoinHashTable>(BufferManager::GetBufferManager(context), conditions, build_types, join_type);
	result->max_ht_size = double(BufferManager::GetBufferManager(context).GetMaxMemory()) * 0.6;
	if (!delim_types.empty() && join_type == JoinType::MARK) {
		// correlated MARK join
		if (delim_types.size() + 1 == conditions.size()) {
			// the correlated MARK join has one more condition than the amount of correlated columns
			// this is the case in a correlated ANY() expression
			// in this case we need to keep track of additional entries, namely:
			// - (1) the total amount of elements per group
			// - (2) the amount of non-null elements per group
			// we need these to correctly deal with the cases of either:
			// - (1) the group being empty [in which case the result is always false, even if the comparison is NULL]
			// - (2) the group containing a NULL value [in which case FALSE becomes NULL]
			auto &info = result->correlated_mark_join_info;

			vector<LogicalType> payload_types;
			vector<BoundAggregateExpression *> correlated_aggregates;
			unique_ptr<BoundAggregateExpression> aggr;

			// jury-rigging the GroupedAggregateHashTable
			// we need a count_star and a count to get counts with and without NULLs

			FunctionBinder function_binder(context);
			aggr = function_binder.BindAggregateFunction(CountStarFun::GetFunction(), {}, nullptr,
			                                             AggregateType::NON_DISTINCT);
			correlated_aggregates.push_back(&*aggr);
			payload_types.push_back(aggr->return_type);
			info.correlated_aggregates.push_back(std::move(aggr));

			auto count_fun = CountFun::GetFunction();
			vector<unique_ptr<Expression>> children;
			// this is a dummy but we need it to make the hash table understand whats going on
			children.push_back(make_uniq_base<Expression, BoundReferenceExpression>(count_fun.return_type, 0));
			aggr = function_binder.BindAggregateFunction(count_fun, std::move(children), nullptr,
			                                             AggregateType::NON_DISTINCT);
			correlated_aggregates.push_back(&*aggr);
			payload_types.push_back(aggr->return_type);
			info.correlated_aggregates.push_back(std::move(aggr));

			auto &allocator = Allocator::Get(context);
			info.correlated_counts = make_uniq<GroupedAggregateHashTable>(context, allocator, delim_types,
			                                                              payload_types, correlated_aggregates);
			info.correlated_types = delim_types;
			info.group_chunk.Initialize(allocator, delim_types);
			info.result_chunk.Initialize(allocator, payload_types);
		}
	}
	return result;
}

unique_ptr<GlobalSinkState> PhysicalHashJoin::GetGlobalSinkState(ClientContext &context) const {
	return make_uniq<HashJoinGlobalSinkState>(*this, context);
}

unique_ptr<LocalSinkState> PhysicalHashJoin::GetLocalSinkState(ExecutionContext &context) const {
	return make_uniq<HashJoinLocalSinkState>(*this, context.client);
}

SinkResultType PhysicalHashJoin::Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const {
	auto &lstate = input.local_state.Cast<HashJoinLocalSinkState>();

	// resolve the join keys for the right chunk
	lstate.join_keys.Reset();
	lstate.build_executor.Execute(chunk, lstate.join_keys);

	// build the HT
	auto &ht = *lstate.hash_table;
	if (!right_projection_map.empty()) {
		// there is a projection map: fill the build chunk with the projected columns
		lstate.build_chunk.Reset();
		lstate.build_chunk.SetCardinality(chunk);
		for (idx_t i = 0; i < right_projection_map.size(); i++) {
			lstate.build_chunk.data[i].Reference(chunk.data[right_projection_map[i]]);
		}
		ht.Build(lstate.append_state, lstate.join_keys, lstate.build_chunk);
	} else if (!build_types.empty()) {
		// there is not a projected map: place the entire right chunk in the HT
		ht.Build(lstate.append_state, lstate.join_keys, chunk);
	} else {
		// there are only keys: place an empty chunk in the payload
		lstate.build_chunk.SetCardinality(chunk.size());
		ht.Build(lstate.append_state, lstate.join_keys, lstate.build_chunk);
	}

	return SinkResultType::NEED_MORE_INPUT;
}

void PhysicalHashJoin::Combine(ExecutionContext &context, GlobalSinkState &gstate_p, LocalSinkState &lstate_p) const {
	auto &gstate = gstate_p.Cast<HashJoinGlobalSinkState>();
	auto &lstate = lstate_p.Cast<HashJoinLocalSinkState>();
	if (lstate.hash_table) {
		lstate.hash_table->GetSinkCollection().FlushAppendState(lstate.append_state);
		lock_guard<mutex> local_ht_lock(gstate.lock);
		gstate.local_hash_tables.push_back(std::move(lstate.hash_table));
	}
	auto &client_profiler = QueryProfiler::Get(context.client);
	context.thread.profiler.Flush(*this, lstate.build_executor, "build_executor", 1);
	client_profiler.Flush(context.thread.profiler);
}

//===--------------------------------------------------------------------===//
// Finalize
//===--------------------------------------------------------------------===//
class HashJoinFinalizeTask : public ExecutorTask {
public:
	HashJoinFinalizeTask(shared_ptr<Event> event_p, ClientContext &context, HashJoinGlobalSinkState &sink_p,
	                     idx_t chunk_idx_from_p, idx_t chunk_idx_to_p, bool parallel_p)
	    : ExecutorTask(context), event(std::move(event_p)), sink(sink_p), chunk_idx_from(chunk_idx_from_p),
	      chunk_idx_to(chunk_idx_to_p), parallel(parallel_p) {
	}

	TaskExecutionResult ExecuteTask(TaskExecutionMode mode) override {
		sink.hash_table->Finalize(chunk_idx_from, chunk_idx_to, parallel);
		event->FinishTask();
		return TaskExecutionResult::TASK_FINISHED;
	}

private:
	shared_ptr<Event> event;
	HashJoinGlobalSinkState &sink;
	idx_t chunk_idx_from;
	idx_t chunk_idx_to;
	bool parallel;
};

class HashJoinFinalizeEvent : public BasePipelineEvent {
public:
	HashJoinFinalizeEvent(Pipeline &pipeline_p, HashJoinGlobalSinkState &sink)
	    : BasePipelineEvent(pipeline_p), sink(sink) {
	}

	HashJoinGlobalSinkState &sink;

public:
	void Schedule() override {
		auto &context = pipeline->GetClientContext();

		vector<shared_ptr<Task>> finalize_tasks;
		auto &ht = *sink.hash_table;
		const auto chunk_count = ht.GetDataCollection().ChunkCount();
		const idx_t num_threads = TaskScheduler::GetScheduler(context).NumberOfThreads();
		if (num_threads == 1 || (ht.Count() < PARALLEL_CONSTRUCT_THRESHOLD && !context.config.verify_parallelism)) {
			// Single-threaded finalize
			finalize_tasks.push_back(
			    make_uniq<HashJoinFinalizeTask>(shared_from_this(), context, sink, 0, chunk_count, false));
		} else {
			// Parallel finalize
			auto chunks_per_thread = MaxValue<idx_t>((chunk_count + num_threads - 1) / num_threads, 1);

			idx_t chunk_idx = 0;
			for (idx_t thread_idx = 0; thread_idx < num_threads; thread_idx++) {
				auto chunk_idx_from = chunk_idx;
				auto chunk_idx_to = MinValue<idx_t>(chunk_idx_from + chunks_per_thread, chunk_count);
				finalize_tasks.push_back(make_uniq<HashJoinFinalizeTask>(shared_from_this(), context, sink,
				                                                         chunk_idx_from, chunk_idx_to, true));
				chunk_idx = chunk_idx_to;
				if (chunk_idx == chunk_count) {
					break;
				}
			}
		}
		SetTasks(std::move(finalize_tasks));
	}

	void FinishEvent() override {
		sink.hash_table->GetDataCollection().VerifyEverythingPinned();
		sink.hash_table->finalized = true;
	}

	static constexpr const idx_t PARALLEL_CONSTRUCT_THRESHOLD = 1048576;
};

void HashJoinGlobalSinkState::ScheduleFinalize(Pipeline &pipeline, Event &event) {
	if (hash_table->Count() == 0) {
		hash_table->finalized = true;
		return;
	}
	hash_table->InitializePointerTable();
	auto new_event = make_shared<HashJoinFinalizeEvent>(pipeline, *this);
	event.InsertEvent(std::move(new_event));
}

void HashJoinGlobalSinkState::InitializeProbeSpill() {
	lock_guard<mutex> guard(lock);
	if (!probe_spill) {
		probe_spill = make_uniq<JoinHashTable::ProbeSpill>(*hash_table, context, probe_types);
	}
}

class HashJoinPartitionTask : public ExecutorTask {
public:
	HashJoinPartitionTask(shared_ptr<Event> event_p, ClientContext &context, JoinHashTable &global_ht,
	                      JoinHashTable &local_ht)
	    : ExecutorTask(context), event(std::move(event_p)), global_ht(global_ht), local_ht(local_ht) {
	}

	TaskExecutionResult ExecuteTask(TaskExecutionMode mode) override {
		local_ht.Partition(global_ht);
		event->FinishTask();
		return TaskExecutionResult::TASK_FINISHED;
	}

private:
	shared_ptr<Event> event;

	JoinHashTable &global_ht;
	JoinHashTable &local_ht;
};

class HashJoinPartitionEvent : public BasePipelineEvent {
public:
	HashJoinPartitionEvent(Pipeline &pipeline_p, HashJoinGlobalSinkState &sink,
	                       vector<unique_ptr<JoinHashTable>> &local_hts)
	    : BasePipelineEvent(pipeline_p), sink(sink), local_hts(local_hts) {
	}

	HashJoinGlobalSinkState &sink;
	vector<unique_ptr<JoinHashTable>> &local_hts;

public:
	void Schedule() override {
		auto &context = pipeline->GetClientContext();
		vector<shared_ptr<Task>> partition_tasks;
		partition_tasks.reserve(local_hts.size());
		for (auto &local_ht : local_hts) {
			partition_tasks.push_back(
			    make_uniq<HashJoinPartitionTask>(shared_from_this(), context, *sink.hash_table, *local_ht));
		}
		SetTasks(std::move(partition_tasks));
	}

	void FinishEvent() override {
		local_hts.clear();
		sink.hash_table->PrepareExternalFinalize();
		sink.ScheduleFinalize(*pipeline, *this);
	}
};

SinkFinalizeType PhysicalHashJoin::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                            GlobalSinkState &gstate) const {
	auto &sink = gstate.Cast<HashJoinGlobalSinkState>();
	auto &ht = *sink.hash_table;

	sink.external = ht.RequiresExternalJoin(context.config, sink.local_hash_tables);
	if (sink.external) {
		sink.perfect_join_executor.reset();
		if (ht.RequiresPartitioning(context.config, sink.local_hash_tables)) {
			auto new_event = make_shared<HashJoinPartitionEvent>(pipeline, sink, sink.local_hash_tables);
			event.InsertEvent(std::move(new_event));
		} else {
			for (auto &local_ht : sink.local_hash_tables) {
				ht.Merge(*local_ht);
			}
			sink.local_hash_tables.clear();
			sink.hash_table->PrepareExternalFinalize();
			sink.ScheduleFinalize(pipeline, event);
		}
		sink.finalized = true;
		return SinkFinalizeType::READY;
	} else {
		for (auto &local_ht : sink.local_hash_tables) {
			ht.Merge(*local_ht);
		}
		sink.local_hash_tables.clear();
		ht.Unpartition();
	}

	// check for possible perfect hash table
	auto use_perfect_hash = sink.perfect_join_executor->CanDoPerfectHashJoin();
	if (use_perfect_hash) {
		D_ASSERT(ht.equality_types.size() == 1);
		auto key_type = ht.equality_types[0];
		use_perfect_hash = sink.perfect_join_executor->BuildPerfectHashTable(key_type);
	}
	// In case of a large build side or duplicates, use regular hash join
	if (!use_perfect_hash) {
		sink.perfect_join_executor.reset();
		sink.ScheduleFinalize(pipeline, event);
	}
	sink.finalized = true;
	if (ht.Count() == 0 && EmptyResultIfRHSIsEmpty()) {
		return SinkFinalizeType::NO_OUTPUT_POSSIBLE;
	}
	return SinkFinalizeType::READY;
}

//===--------------------------------------------------------------------===//
// Operator
//===--------------------------------------------------------------------===//
class HashJoinOperatorState : public CachingOperatorState {
public:
	explicit HashJoinOperatorState(ClientContext &context) : probe_executor(context), initialized(false) {
	}

	DataChunk join_keys;
	ExpressionExecutor probe_executor;
	unique_ptr<JoinHashTable::ScanStructure> scan_structure;
	unique_ptr<OperatorState> perfect_hash_join_state;

	bool initialized;
	JoinHashTable::ProbeSpillLocalAppendState spill_state;
	//! Chunk to sink data into for external join
	DataChunk spill_chunk;

public:
	void Finalize(const PhysicalOperator &op, ExecutionContext &context) override {
		context.thread.profiler.Flush(op, probe_executor, "probe_executor", 0);
	}
};

unique_ptr<OperatorState> PhysicalHashJoin::GetOperatorState(ExecutionContext &context) const {
	auto &allocator = Allocator::Get(context.client);
	auto &sink = sink_state->Cast<HashJoinGlobalSinkState>();
	auto state = make_uniq<HashJoinOperatorState>(context.client);
	if (sink.perfect_join_executor) {
		state->perfect_hash_join_state = sink.perfect_join_executor->GetOperatorState(context);
	} else {
		state->join_keys.Initialize(allocator, condition_types);
		for (auto &cond : conditions) {
			state->probe_executor.AddExpression(*cond.left);
		}
	}
	if (sink.external) {
		state->spill_chunk.Initialize(allocator, sink.probe_types);
		sink.InitializeProbeSpill();
	}

	return std::move(state);
}

OperatorResultType PhysicalHashJoin::ExecuteInternal(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                                     GlobalOperatorState &gstate, OperatorState &state_p) const {
	auto &state = state_p.Cast<HashJoinOperatorState>();
	auto &sink = sink_state->Cast<HashJoinGlobalSinkState>();
	D_ASSERT(sink.finalized);
	D_ASSERT(!sink.scanned_data);

	// some initialization for external hash join
	if (sink.external && !state.initialized) {
		if (!sink.probe_spill) {
			sink.InitializeProbeSpill();
		}
		state.spill_state = sink.probe_spill->RegisterThread();
		state.initialized = true;
	}

	if (sink.hash_table->Count() == 0 && EmptyResultIfRHSIsEmpty()) {
		return OperatorResultType::FINISHED;
	}

	if (sink.perfect_join_executor) {
		D_ASSERT(!sink.external);
		return sink.perfect_join_executor->ProbePerfectHashTable(context, input, chunk, *state.perfect_hash_join_state);
	}

	if (state.scan_structure) {
		// still have elements remaining (i.e. we got >STANDARD_VECTOR_SIZE elements in the previous probe)
		state.scan_structure->Next(state.join_keys, input, chunk);
		if (chunk.size() > 0) {
			return OperatorResultType::HAVE_MORE_OUTPUT;
		}
		state.scan_structure = nullptr;
		return OperatorResultType::NEED_MORE_INPUT;
	}

	// probe the HT
	if (sink.hash_table->Count() == 0) {
		ConstructEmptyJoinResult(sink.hash_table->join_type, sink.hash_table->has_null, input, chunk);
		return OperatorResultType::NEED_MORE_INPUT;
	}

	// resolve the join keys for the left chunk
	state.join_keys.Reset();
	state.probe_executor.Execute(input, state.join_keys);

	// perform the actual probe
	if (sink.external) {
		state.scan_structure = sink.hash_table->ProbeAndSpill(state.join_keys, input, *sink.probe_spill,
		                                                      state.spill_state, state.spill_chunk);
	} else {
		state.scan_structure = sink.hash_table->Probe(state.join_keys);
	}
	state.scan_structure->Next(state.join_keys, input, chunk);
	return OperatorResultType::HAVE_MORE_OUTPUT;
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
enum class HashJoinSourceStage : uint8_t { INIT, BUILD, PROBE, SCAN_HT, DONE };

class HashJoinLocalSourceState;

class HashJoinGlobalSourceState : public GlobalSourceState {
public:
	HashJoinGlobalSourceState(const PhysicalHashJoin &op, ClientContext &context);

	//! Initialize this source state using the info in the sink
	void Initialize(HashJoinGlobalSinkState &sink);
	//! Try to prepare the next stage
	void TryPrepareNextStage(HashJoinGlobalSinkState &sink);
	//! Prepare the next build/probe/scan_ht stage for external hash join (must hold lock)
	void PrepareBuild(HashJoinGlobalSinkState &sink);
	void PrepareProbe(HashJoinGlobalSinkState &sink);
	void PrepareScanHT(HashJoinGlobalSinkState &sink);
	//! Assigns a task to a local source state
	bool AssignTask(HashJoinGlobalSinkState &sink, HashJoinLocalSourceState &lstate);

	idx_t MaxThreads() override {
		return probe_count / ((idx_t)STANDARD_VECTOR_SIZE * parallel_scan_chunk_count);
	}

public:
	const PhysicalHashJoin &op;

	//! For synchronizing the external hash join
	atomic<HashJoinSourceStage> global_stage;
	mutex lock;

	//! For HT build synchronization
	idx_t build_chunk_idx;
	idx_t build_chunk_count;
	idx_t build_chunk_done;
	idx_t build_chunks_per_thread;

	//! For probe synchronization
	idx_t probe_chunk_count;
	idx_t probe_chunk_done;

	//! To determine the number of threads
	idx_t probe_count;
	idx_t parallel_scan_chunk_count;

	//! For full/outer synchronization
	idx_t full_outer_chunk_idx;
	idx_t full_outer_chunk_count;
	idx_t full_outer_chunk_done;
	idx_t full_outer_chunks_per_thread;
};

class HashJoinLocalSourceState : public LocalSourceState {
public:
	HashJoinLocalSourceState(const PhysicalHashJoin &op, Allocator &allocator);

	//! Do the work this thread has been assigned
	void ExecuteTask(HashJoinGlobalSinkState &sink, HashJoinGlobalSourceState &gstate, DataChunk &chunk);
	//! Whether this thread has finished the work it has been assigned
	bool TaskFinished();
	//! Build, probe and scan for external hash join
	void ExternalBuild(HashJoinGlobalSinkState &sink, HashJoinGlobalSourceState &gstate);
	void ExternalProbe(HashJoinGlobalSinkState &sink, HashJoinGlobalSourceState &gstate, DataChunk &chunk);
	void ExternalScanHT(HashJoinGlobalSinkState &sink, HashJoinGlobalSourceState &gstate, DataChunk &chunk);

public:
	//! The stage that this thread was assigned work for
	HashJoinSourceStage local_stage;
	//! Vector with pointers here so we don't have to re-initialize
	Vector addresses;

	//! Chunks assigned to this thread for building the pointer table
	idx_t build_chunk_idx_from;
	idx_t build_chunk_idx_to;

	//! Local scan state for probe spill
	ColumnDataConsumerScanState probe_local_scan;
	//! Chunks for holding the scanned probe collection
	DataChunk probe_chunk;
	DataChunk join_keys;
	DataChunk payload;
	//! Column indices to easily reference the join keys/payload columns in probe_chunk
	vector<idx_t> join_key_indices;
	vector<idx_t> payload_indices;
	//! Scan structure for the external probe
	unique_ptr<JoinHashTable::ScanStructure> scan_structure;
	bool empty_ht_probe_in_progress;

	//! Chunks assigned to this thread for a full/outer scan
	idx_t full_outer_chunk_idx_from;
	idx_t full_outer_chunk_idx_to;
	unique_ptr<JoinHTScanState> full_outer_scan_state;
};

unique_ptr<GlobalSourceState> PhysicalHashJoin::GetGlobalSourceState(ClientContext &context) const {
	return make_uniq<HashJoinGlobalSourceState>(*this, context);
}

unique_ptr<LocalSourceState> PhysicalHashJoin::GetLocalSourceState(ExecutionContext &context,
                                                                   GlobalSourceState &gstate) const {
	return make_uniq<HashJoinLocalSourceState>(*this, Allocator::Get(context.client));
}

HashJoinGlobalSourceState::HashJoinGlobalSourceState(const PhysicalHashJoin &op, ClientContext &context)
    : op(op), global_stage(HashJoinSourceStage::INIT), build_chunk_count(0), build_chunk_done(0), probe_chunk_count(0),
      probe_chunk_done(0), probe_count(op.children[0]->estimated_cardinality),
      parallel_scan_chunk_count(context.config.verify_parallelism ? 1 : 120) {
}

void HashJoinGlobalSourceState::Initialize(HashJoinGlobalSinkState &sink) {
	lock_guard<mutex> init_lock(lock);
	if (global_stage != HashJoinSourceStage::INIT) {
		// Another thread initialized
		return;
	}

	// Finalize the probe spill
	if (sink.probe_spill) {
		sink.probe_spill->Finalize();
	}

	global_stage = HashJoinSourceStage::PROBE;
	TryPrepareNextStage(sink);
}

void HashJoinGlobalSourceState::TryPrepareNextStage(HashJoinGlobalSinkState &sink) {
	switch (global_stage.load()) {
	case HashJoinSourceStage::BUILD:
		if (build_chunk_done == build_chunk_count) {
			sink.hash_table->GetDataCollection().VerifyEverythingPinned();
			sink.hash_table->finalized = true;
			PrepareProbe(sink);
		}
		break;
	case HashJoinSourceStage::PROBE:
		if (probe_chunk_done == probe_chunk_count) {
			if (IsRightOuterJoin(op.join_type)) {
				PrepareScanHT(sink);
			} else {
				PrepareBuild(sink);
			}
		}
		break;
	case HashJoinSourceStage::SCAN_HT:
		if (full_outer_chunk_done == full_outer_chunk_count) {
			PrepareBuild(sink);
		}
		break;
	default:
		break;
	}
}

void HashJoinGlobalSourceState::PrepareBuild(HashJoinGlobalSinkState &sink) {
	D_ASSERT(global_stage != HashJoinSourceStage::BUILD);
	auto &ht = *sink.hash_table;

	// Try to put the next partitions in the block collection of the HT
	if (!sink.external || !ht.PrepareExternalFinalize()) {
		global_stage = HashJoinSourceStage::DONE;
		return;
	}

	auto &data_collection = ht.GetDataCollection();
	if (data_collection.Count() == 0 && op.EmptyResultIfRHSIsEmpty()) {
		PrepareBuild(sink);
		return;
	}

	build_chunk_idx = 0;
	build_chunk_count = data_collection.ChunkCount();
	build_chunk_done = 0;

	auto num_threads = TaskScheduler::GetScheduler(sink.context).NumberOfThreads();
	build_chunks_per_thread = MaxValue<idx_t>((build_chunk_count + num_threads - 1) / num_threads, 1);

	ht.InitializePointerTable();

	global_stage = HashJoinSourceStage::BUILD;
}

void HashJoinGlobalSourceState::PrepareProbe(HashJoinGlobalSinkState &sink) {
	sink.probe_spill->PrepareNextProbe();
	const auto &consumer = *sink.probe_spill->consumer;

	probe_chunk_count = consumer.Count() == 0 ? 0 : consumer.ChunkCount();
	probe_chunk_done = 0;

	global_stage = HashJoinSourceStage::PROBE;
	if (probe_chunk_count == 0) {
		TryPrepareNextStage(sink);
		return;
	}
}

void HashJoinGlobalSourceState::PrepareScanHT(HashJoinGlobalSinkState &sink) {
	D_ASSERT(global_stage != HashJoinSourceStage::SCAN_HT);
	auto &ht = *sink.hash_table;

	auto &data_collection = ht.GetDataCollection();
	full_outer_chunk_idx = 0;
	full_outer_chunk_count = data_collection.ChunkCount();
	full_outer_chunk_done = 0;

	auto num_threads = TaskScheduler::GetScheduler(sink.context).NumberOfThreads();
	full_outer_chunks_per_thread = MaxValue<idx_t>((full_outer_chunk_count + num_threads - 1) / num_threads, 1);

	global_stage = HashJoinSourceStage::SCAN_HT;
}

bool HashJoinGlobalSourceState::AssignTask(HashJoinGlobalSinkState &sink, HashJoinLocalSourceState &lstate) {
	D_ASSERT(lstate.TaskFinished());

	lock_guard<mutex> guard(lock);
	switch (global_stage.load()) {
	case HashJoinSourceStage::BUILD:
		if (build_chunk_idx != build_chunk_count) {
			lstate.local_stage = global_stage;
			lstate.build_chunk_idx_from = build_chunk_idx;
			build_chunk_idx = MinValue<idx_t>(build_chunk_count, build_chunk_idx + build_chunks_per_thread);
			lstate.build_chunk_idx_to = build_chunk_idx;
			return true;
		}
		break;
	case HashJoinSourceStage::PROBE:
		if (sink.probe_spill->consumer && sink.probe_spill->consumer->AssignChunk(lstate.probe_local_scan)) {
			lstate.local_stage = global_stage;
			lstate.empty_ht_probe_in_progress = false;
			return true;
		}
		break;
	case HashJoinSourceStage::SCAN_HT:
		if (full_outer_chunk_idx != full_outer_chunk_count) {
			lstate.local_stage = global_stage;
			lstate.full_outer_chunk_idx_from = full_outer_chunk_idx;
			full_outer_chunk_idx =
			    MinValue<idx_t>(full_outer_chunk_count, full_outer_chunk_idx + full_outer_chunks_per_thread);
			lstate.full_outer_chunk_idx_to = full_outer_chunk_idx;
			return true;
		}
		break;
	case HashJoinSourceStage::DONE:
		break;
	default:
		throw InternalException("Unexpected HashJoinSourceStage in AssignTask!");
	}
	return false;
}

HashJoinLocalSourceState::HashJoinLocalSourceState(const PhysicalHashJoin &op, Allocator &allocator)
    : local_stage(HashJoinSourceStage::INIT), addresses(LogicalType::POINTER) {
	auto &chunk_state = probe_local_scan.current_chunk_state;
	chunk_state.properties = ColumnDataScanProperties::ALLOW_ZERO_COPY;

	auto &sink = op.sink_state->Cast<HashJoinGlobalSinkState>();
	probe_chunk.Initialize(allocator, sink.probe_types);
	join_keys.Initialize(allocator, op.condition_types);
	payload.Initialize(allocator, op.children[0]->types);

	// Store the indices of the columns to reference them easily
	idx_t col_idx = 0;
	for (; col_idx < op.condition_types.size(); col_idx++) {
		join_key_indices.push_back(col_idx);
	}
	for (; col_idx < sink.probe_types.size() - 1; col_idx++) {
		payload_indices.push_back(col_idx);
	}
}

void HashJoinLocalSourceState::ExecuteTask(HashJoinGlobalSinkState &sink, HashJoinGlobalSourceState &gstate,
                                           DataChunk &chunk) {
	switch (local_stage) {
	case HashJoinSourceStage::BUILD:
		ExternalBuild(sink, gstate);
		break;
	case HashJoinSourceStage::PROBE:
		ExternalProbe(sink, gstate, chunk);
		break;
	case HashJoinSourceStage::SCAN_HT:
		ExternalScanHT(sink, gstate, chunk);
		break;
	default:
		throw InternalException("Unexpected HashJoinSourceStage in ExecuteTask!");
	}
}

bool HashJoinLocalSourceState::TaskFinished() {
	switch (local_stage) {
	case HashJoinSourceStage::INIT:
	case HashJoinSourceStage::BUILD:
		return true;
	case HashJoinSourceStage::PROBE:
		return scan_structure == nullptr && !empty_ht_probe_in_progress;
	case HashJoinSourceStage::SCAN_HT:
		return full_outer_scan_state == nullptr;
	default:
		throw InternalException("Unexpected HashJoinSourceStage in TaskFinished!");
	}
}

void HashJoinLocalSourceState::ExternalBuild(HashJoinGlobalSinkState &sink, HashJoinGlobalSourceState &gstate) {
	D_ASSERT(local_stage == HashJoinSourceStage::BUILD);

	auto &ht = *sink.hash_table;
	ht.Finalize(build_chunk_idx_from, build_chunk_idx_to, true);

	lock_guard<mutex> guard(gstate.lock);
	gstate.build_chunk_done += build_chunk_idx_to - build_chunk_idx_from;
}

void HashJoinLocalSourceState::ExternalProbe(HashJoinGlobalSinkState &sink, HashJoinGlobalSourceState &gstate,
                                             DataChunk &chunk) {
	D_ASSERT(local_stage == HashJoinSourceStage::PROBE && sink.hash_table->finalized);

	if (scan_structure) {
		// Still have elements remaining (i.e. we got >STANDARD_VECTOR_SIZE elements in the previous probe)
		scan_structure->Next(join_keys, payload, chunk);
		if (chunk.size() != 0) {
			return;
		}
	}

	if (scan_structure || empty_ht_probe_in_progress) {
		// Previous probe is done
		scan_structure = nullptr;
		empty_ht_probe_in_progress = false;
		sink.probe_spill->consumer->FinishChunk(probe_local_scan);
		lock_guard<mutex> lock(gstate.lock);
		gstate.probe_chunk_done++;
		return;
	}

	// Scan input chunk for next probe
	sink.probe_spill->consumer->ScanChunk(probe_local_scan, probe_chunk);

	// Get the probe chunk columns/hashes
	join_keys.ReferenceColumns(probe_chunk, join_key_indices);
	payload.ReferenceColumns(probe_chunk, payload_indices);
	auto precomputed_hashes = &probe_chunk.data.back();

	if (sink.hash_table->Count() == 0 && !gstate.op.EmptyResultIfRHSIsEmpty()) {
		gstate.op.ConstructEmptyJoinResult(sink.hash_table->join_type, sink.hash_table->has_null, payload, chunk);
		empty_ht_probe_in_progress = true;
		return;
	}

	// Perform the probe
	scan_structure = sink.hash_table->Probe(join_keys, precomputed_hashes);
	scan_structure->Next(join_keys, payload, chunk);
}

void HashJoinLocalSourceState::ExternalScanHT(HashJoinGlobalSinkState &sink, HashJoinGlobalSourceState &gstate,
                                              DataChunk &chunk) {
	D_ASSERT(local_stage == HashJoinSourceStage::SCAN_HT);

	if (!full_outer_scan_state) {
		full_outer_scan_state = make_uniq<JoinHTScanState>(sink.hash_table->GetDataCollection(),
		                                                   full_outer_chunk_idx_from, full_outer_chunk_idx_to);
	}
	sink.hash_table->ScanFullOuter(*full_outer_scan_state, addresses, chunk);

	if (chunk.size() == 0) {
		full_outer_scan_state = nullptr;
		lock_guard<mutex> guard(gstate.lock);
		gstate.full_outer_chunk_done += full_outer_chunk_idx_to - full_outer_chunk_idx_from;
	}
}

SourceResultType PhysicalHashJoin::GetData(ExecutionContext &context, DataChunk &chunk,
                                           OperatorSourceInput &input) const {
	auto &sink = sink_state->Cast<HashJoinGlobalSinkState>();
	auto &gstate = input.global_state.Cast<HashJoinGlobalSourceState>();
	auto &lstate = input.local_state.Cast<HashJoinLocalSourceState>();
	sink.scanned_data = true;

	if (!sink.external && !IsRightOuterJoin(join_type)) {
		return SourceResultType::FINISHED;
	}

	if (gstate.global_stage == HashJoinSourceStage::INIT) {
		gstate.Initialize(sink);
	}

	// Any call to GetData must produce tuples, otherwise the pipeline executor thinks that we're done
	// Therefore, we loop until we've produced tuples, or until the operator is actually done
	while (gstate.global_stage != HashJoinSourceStage::DONE && chunk.size() == 0) {
		if (!lstate.TaskFinished() || gstate.AssignTask(sink, lstate)) {
			lstate.ExecuteTask(sink, gstate, chunk);
		} else {
			lock_guard<mutex> guard(gstate.lock);
			gstate.TryPrepareNextStage(sink);
		}
	}

	return chunk.size() == 0 ? SourceResultType::FINISHED : SourceResultType::HAVE_MORE_OUTPUT;
}

} // namespace duckdb














#include <thread>

namespace duckdb {

PhysicalIEJoin::PhysicalIEJoin(LogicalOperator &op, unique_ptr<PhysicalOperator> left,
                               unique_ptr<PhysicalOperator> right, vector<JoinCondition> cond, JoinType join_type,
                               idx_t estimated_cardinality)
    : PhysicalRangeJoin(op, PhysicalOperatorType::IE_JOIN, std::move(left), std::move(right), std::move(cond),
                        join_type, estimated_cardinality) {

	// 1. let L1 (resp. L2) be the array of column X (resp. Y)
	D_ASSERT(conditions.size() >= 2);
	lhs_orders.resize(2);
	rhs_orders.resize(2);
	for (idx_t i = 0; i < 2; ++i) {
		auto &cond = conditions[i];
		D_ASSERT(cond.left->return_type == cond.right->return_type);
		join_key_types.push_back(cond.left->return_type);

		// Convert the conditions to sort orders
		auto left = cond.left->Copy();
		auto right = cond.right->Copy();
		auto sense = OrderType::INVALID;

		// 2. if (op1 ∈ {>, ≥}) sort L1 in descending order
		// 3. else if (op1 ∈ {<, ≤}) sort L1 in ascending order
		// 4. if (op2 ∈ {>, ≥}) sort L2 in ascending order
		// 5. else if (op2 ∈ {<, ≤}) sort L2 in descending order
		switch (cond.comparison) {
		case ExpressionType::COMPARE_GREATERTHAN:
		case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
			sense = i ? OrderType::ASCENDING : OrderType::DESCENDING;
			break;
		case ExpressionType::COMPARE_LESSTHAN:
		case ExpressionType::COMPARE_LESSTHANOREQUALTO:
			sense = i ? OrderType::DESCENDING : OrderType::ASCENDING;
			break;
		default:
			throw NotImplementedException("Unimplemented join type for IEJoin");
		}
		lhs_orders[i].emplace_back(BoundOrderByNode(sense, OrderByNullType::NULLS_LAST, std::move(left)));
		rhs_orders[i].emplace_back(BoundOrderByNode(sense, OrderByNullType::NULLS_LAST, std::move(right)));
	}

	for (idx_t i = 2; i < conditions.size(); ++i) {
		auto &cond = conditions[i];
		D_ASSERT(cond.left->return_type == cond.right->return_type);
		join_key_types.push_back(cond.left->return_type);
	}
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
class IEJoinLocalState : public LocalSinkState {
public:
	using LocalSortedTable = PhysicalRangeJoin::LocalSortedTable;

	IEJoinLocalState(ClientContext &context, const PhysicalRangeJoin &op, const idx_t child)
	    : table(context, op, child) {
	}

	//! The local sort state
	LocalSortedTable table;
};

class IEJoinGlobalState : public GlobalSinkState {
public:
	using GlobalSortedTable = PhysicalRangeJoin::GlobalSortedTable;

public:
	IEJoinGlobalState(ClientContext &context, const PhysicalIEJoin &op) : child(0) {
		tables.resize(2);
		RowLayout lhs_layout;
		lhs_layout.Initialize(op.children[0]->types);
		vector<BoundOrderByNode> lhs_order;
		lhs_order.emplace_back(op.lhs_orders[0][0].Copy());
		tables[0] = make_uniq<GlobalSortedTable>(context, lhs_order, lhs_layout);

		RowLayout rhs_layout;
		rhs_layout.Initialize(op.children[1]->types);
		vector<BoundOrderByNode> rhs_order;
		rhs_order.emplace_back(op.rhs_orders[0][0].Copy());
		tables[1] = make_uniq<GlobalSortedTable>(context, rhs_order, rhs_layout);
	}

	IEJoinGlobalState(IEJoinGlobalState &prev)
	    : GlobalSinkState(prev), tables(std::move(prev.tables)), child(prev.child + 1) {
	}

	void Sink(DataChunk &input, IEJoinLocalState &lstate) {
		auto &table = *tables[child];
		auto &global_sort_state = table.global_sort_state;
		auto &local_sort_state = lstate.table.local_sort_state;

		// Sink the data into the local sort state
		lstate.table.Sink(input, global_sort_state);

		// When sorting data reaches a certain size, we sort it
		if (local_sort_state.SizeInBytes() >= table.memory_per_thread) {
			local_sort_state.Sort(global_sort_state, true);
		}
	}

	vector<unique_ptr<GlobalSortedTable>> tables;
	size_t child;
};

unique_ptr<GlobalSinkState> PhysicalIEJoin::GetGlobalSinkState(ClientContext &context) const {
	D_ASSERT(!sink_state);
	return make_uniq<IEJoinGlobalState>(context, *this);
}

unique_ptr<LocalSinkState> PhysicalIEJoin::GetLocalSinkState(ExecutionContext &context) const {
	idx_t sink_child = 0;
	if (sink_state) {
		const auto &ie_sink = sink_state->Cast<IEJoinGlobalState>();
		sink_child = ie_sink.child;
	}
	return make_uniq<IEJoinLocalState>(context.client, *this, sink_child);
}

SinkResultType PhysicalIEJoin::Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const {
	auto &gstate = input.global_state.Cast<IEJoinGlobalState>();
	auto &lstate = input.local_state.Cast<IEJoinLocalState>();

	gstate.Sink(chunk, lstate);

	return SinkResultType::NEED_MORE_INPUT;
}

void PhysicalIEJoin::Combine(ExecutionContext &context, GlobalSinkState &gstate_p, LocalSinkState &lstate_p) const {
	auto &gstate = gstate_p.Cast<IEJoinGlobalState>();
	auto &lstate = lstate_p.Cast<IEJoinLocalState>();
	gstate.tables[gstate.child]->Combine(lstate.table);
	auto &client_profiler = QueryProfiler::Get(context.client);

	context.thread.profiler.Flush(*this, lstate.table.executor, gstate.child ? "rhs_executor" : "lhs_executor", 1);
	client_profiler.Flush(context.thread.profiler);
}

//===--------------------------------------------------------------------===//
// Finalize
//===--------------------------------------------------------------------===//
SinkFinalizeType PhysicalIEJoin::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                          GlobalSinkState &gstate_p) const {
	auto &gstate = gstate_p.Cast<IEJoinGlobalState>();
	auto &table = *gstate.tables[gstate.child];
	auto &global_sort_state = table.global_sort_state;

	if ((gstate.child == 1 && IsRightOuterJoin(join_type)) || (gstate.child == 0 && IsLeftOuterJoin(join_type))) {
		// for FULL/LEFT/RIGHT OUTER JOIN, initialize found_match to false for every tuple
		table.IntializeMatches();
	}
	if (gstate.child == 1 && global_sort_state.sorted_blocks.empty() && EmptyResultIfRHSIsEmpty()) {
		// Empty input!
		return SinkFinalizeType::NO_OUTPUT_POSSIBLE;
	}

	// Sort the current input child
	table.Finalize(pipeline, event);

	// Move to the next input child
	++gstate.child;

	return SinkFinalizeType::READY;
}

//===--------------------------------------------------------------------===//
// Operator
//===--------------------------------------------------------------------===//
OperatorResultType PhysicalIEJoin::ExecuteInternal(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                                   GlobalOperatorState &gstate, OperatorState &state) const {
	return OperatorResultType::FINISHED;
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
struct IEJoinUnion {
	using SortedTable = PhysicalRangeJoin::GlobalSortedTable;

	static idx_t AppendKey(SortedTable &table, ExpressionExecutor &executor, SortedTable &marked, int64_t increment,
	                       int64_t base, const idx_t block_idx);

	static void Sort(SortedTable &table) {
		auto &global_sort_state = table.global_sort_state;
		global_sort_state.PrepareMergePhase();
		while (global_sort_state.sorted_blocks.size() > 1) {
			global_sort_state.InitializeMergeRound();
			MergeSorter merge_sorter(global_sort_state, global_sort_state.buffer_manager);
			merge_sorter.PerformInMergeRound();
			global_sort_state.CompleteMergeRound(true);
		}
	}

	template <typename T>
	static vector<T> ExtractColumn(SortedTable &table, idx_t col_idx) {
		vector<T> result;
		result.reserve(table.count);

		auto &gstate = table.global_sort_state;
		auto &blocks = *gstate.sorted_blocks[0]->payload_data;
		PayloadScanner scanner(blocks, gstate, false);

		DataChunk payload;
		payload.Initialize(Allocator::DefaultAllocator(), gstate.payload_layout.GetTypes());
		for (;;) {
			scanner.Scan(payload);
			const auto count = payload.size();
			if (!count) {
				break;
			}

			const auto data_ptr = FlatVector::GetData<T>(payload.data[col_idx]);
			result.insert(result.end(), data_ptr, data_ptr + count);
		}

		return result;
	}

	IEJoinUnion(ClientContext &context, const PhysicalIEJoin &op, SortedTable &t1, const idx_t b1, SortedTable &t2,
	            const idx_t b2);

	idx_t SearchL1(idx_t pos);
	bool NextRow();

	//! Inverted loop
	idx_t JoinComplexBlocks(SelectionVector &lsel, SelectionVector &rsel);

	//! L1
	unique_ptr<SortedTable> l1;
	//! L2
	unique_ptr<SortedTable> l2;

	//! Li
	vector<int64_t> li;
	//! P
	vector<idx_t> p;

	//! B
	vector<validity_t> bit_array;
	ValidityMask bit_mask;

	//! Bloom Filter
	static constexpr idx_t BLOOM_CHUNK_BITS = 1024;
	idx_t bloom_count;
	vector<validity_t> bloom_array;
	ValidityMask bloom_filter;

	//! Iteration state
	idx_t n;
	idx_t i;
	idx_t j;
	unique_ptr<SBIterator> op1;
	unique_ptr<SBIterator> off1;
	unique_ptr<SBIterator> op2;
	unique_ptr<SBIterator> off2;
	int64_t lrid;
};

idx_t IEJoinUnion::AppendKey(SortedTable &table, ExpressionExecutor &executor, SortedTable &marked, int64_t increment,
                             int64_t base, const idx_t block_idx) {
	LocalSortState local_sort_state;
	local_sort_state.Initialize(marked.global_sort_state, marked.global_sort_state.buffer_manager);

	// Reading
	const auto valid = table.count - table.has_null;
	auto &gstate = table.global_sort_state;
	PayloadScanner scanner(gstate, block_idx);
	auto table_idx = block_idx * gstate.block_capacity;

	DataChunk scanned;
	scanned.Initialize(Allocator::DefaultAllocator(), scanner.GetPayloadTypes());

	// Writing
	auto types = local_sort_state.sort_layout->logical_types;
	const idx_t payload_idx = types.size();

	const auto &payload_types = local_sort_state.payload_layout->GetTypes();
	types.insert(types.end(), payload_types.begin(), payload_types.end());
	const idx_t rid_idx = types.size() - 1;

	DataChunk keys;
	DataChunk payload;
	keys.Initialize(Allocator::DefaultAllocator(), types);

	idx_t inserted = 0;
	for (auto rid = base; table_idx < valid;) {
		scanner.Scan(scanned);

		// NULLs are at the end, so stop when we reach them
		auto scan_count = scanned.size();
		if (table_idx + scan_count > valid) {
			scan_count = valid - table_idx;
			scanned.SetCardinality(scan_count);
		}
		if (scan_count == 0) {
			break;
		}
		table_idx += scan_count;

		// Compute the input columns from the payload
		keys.Reset();
		keys.Split(payload, rid_idx);
		executor.Execute(scanned, keys);

		// Mark the rid column
		payload.data[0].Sequence(rid, increment, scan_count);
		payload.SetCardinality(scan_count);
		keys.Fuse(payload);
		rid += increment * scan_count;

		// Sort on the sort columns (which will no longer be needed)
		keys.Split(payload, payload_idx);
		local_sort_state.SinkChunk(keys, payload);
		inserted += scan_count;
		keys.Fuse(payload);

		// Flush when we have enough data
		if (local_sort_state.SizeInBytes() >= marked.memory_per_thread) {
			local_sort_state.Sort(marked.global_sort_state, true);
		}
	}
	marked.global_sort_state.AddLocalState(local_sort_state);
	marked.count += inserted;

	return inserted;
}

IEJoinUnion::IEJoinUnion(ClientContext &context, const PhysicalIEJoin &op, SortedTable &t1, const idx_t b1,
                         SortedTable &t2, const idx_t b2)
    : n(0), i(0) {
	// input : query Q with 2 join predicates t1.X op1 t2.X' and t1.Y op2 t2.Y', tables T, T' of sizes m and n resp.
	// output: a list of tuple pairs (ti , tj)
	// Note that T/T' are already sorted on X/X' and contain the payload data
	// We only join the two block numbers and use the sizes of the blocks as the counts

	// 0. Filter out tables with no overlap
	if (!t1.BlockSize(b1) || !t2.BlockSize(b2)) {
		return;
	}

	const auto &cmp1 = op.conditions[0].comparison;
	SBIterator bounds1(t1.global_sort_state, cmp1);
	SBIterator bounds2(t2.global_sort_state, cmp1);

	// t1.X[0] op1 t2.X'[-1]
	bounds1.SetIndex(bounds1.block_capacity * b1);
	bounds2.SetIndex(bounds2.block_capacity * b2 + t2.BlockSize(b2) - 1);
	if (!bounds1.Compare(bounds2)) {
		return;
	}

	// 1. let L1 (resp. L2) be the array of column X (resp. Y )
	const auto &order1 = op.lhs_orders[0][0];
	const auto &order2 = op.lhs_orders[1][0];

	// 2. if (op1 ∈ {>, ≥}) sort L1 in descending order
	// 3. else if (op1 ∈ {<, ≤}) sort L1 in ascending order

	// For the union algorithm, we make a unified table with the keys and the rids as the payload:
	//		X/X', Y/Y', R/R'/Li
	// The first position is the sort key.
	vector<LogicalType> types;
	types.emplace_back(order2.expression->return_type);
	types.emplace_back(LogicalType::BIGINT);
	RowLayout payload_layout;
	payload_layout.Initialize(types);

	// Sort on the first expression
	auto ref = make_uniq<BoundReferenceExpression>(order1.expression->return_type, 0);
	vector<BoundOrderByNode> orders;
	orders.emplace_back(order1.type, order1.null_order, std::move(ref));

	l1 = make_uniq<SortedTable>(context, orders, payload_layout);

	// LHS has positive rids
	ExpressionExecutor l_executor(context);
	l_executor.AddExpression(*order1.expression);
	l_executor.AddExpression(*order2.expression);
	AppendKey(t1, l_executor, *l1, 1, 1, b1);

	// RHS has negative rids
	ExpressionExecutor r_executor(context);
	r_executor.AddExpression(*op.rhs_orders[0][0].expression);
	r_executor.AddExpression(*op.rhs_orders[1][0].expression);
	AppendKey(t2, r_executor, *l1, -1, -1, b2);

	if (l1->global_sort_state.sorted_blocks.empty()) {
		return;
	}

	Sort(*l1);

	op1 = make_uniq<SBIterator>(l1->global_sort_state, cmp1);
	off1 = make_uniq<SBIterator>(l1->global_sort_state, cmp1);

	// We don't actually need the L1 column, just its sort key, which is in the sort blocks
	li = ExtractColumn<int64_t>(*l1, types.size() - 1);

	// 4. if (op2 ∈ {>, ≥}) sort L2 in ascending order
	// 5. else if (op2 ∈ {<, ≤}) sort L2 in descending order

	// We sort on Y/Y' to obtain the sort keys and the permutation array.
	// For this we just need a two-column table of Y, P
	types.clear();
	types.emplace_back(LogicalType::BIGINT);
	payload_layout.Initialize(types);

	// Sort on the first expression
	orders.clear();
	ref = make_uniq<BoundReferenceExpression>(order2.expression->return_type, 0);
	orders.emplace_back(order2.type, order2.null_order, std::move(ref));

	ExpressionExecutor executor(context);
	executor.AddExpression(*orders[0].expression);

	l2 = make_uniq<SortedTable>(context, orders, payload_layout);
	for (idx_t base = 0, block_idx = 0; block_idx < l1->BlockCount(); ++block_idx) {
		base += AppendKey(*l1, executor, *l2, 1, base, block_idx);
	}

	Sort(*l2);

	// We don't actually need the L2 column, just its sort key, which is in the sort blocks

	// 6. compute the permutation array P of L2 w.r.t. L1
	p = ExtractColumn<idx_t>(*l2, types.size() - 1);

	// 7. initialize bit-array B (|B| = n), and set all bits to 0
	n = l2->count.load();
	bit_array.resize(ValidityMask::EntryCount(n), 0);
	bit_mask.Initialize(bit_array.data());

	// Bloom filter
	bloom_count = (n + (BLOOM_CHUNK_BITS - 1)) / BLOOM_CHUNK_BITS;
	bloom_array.resize(ValidityMask::EntryCount(bloom_count), 0);
	bloom_filter.Initialize(bloom_array.data());

	// 11. for(i←1 to n) do
	const auto &cmp2 = op.conditions[1].comparison;
	op2 = make_uniq<SBIterator>(l2->global_sort_state, cmp2);
	off2 = make_uniq<SBIterator>(l2->global_sort_state, cmp2);
	i = 0;
	j = 0;
	(void)NextRow();
}

idx_t IEJoinUnion::SearchL1(idx_t pos) {
	// Perform an exponential search in the appropriate direction
	op1->SetIndex(pos);

	idx_t step = 1;
	auto hi = pos;
	auto lo = pos;
	if (!op1->cmp) {
		// Scan left for loose inequality
		lo -= MinValue(step, lo);
		step *= 2;
		off1->SetIndex(lo);
		while (lo > 0 && op1->Compare(*off1)) {
			hi = lo;
			lo -= MinValue(step, lo);
			step *= 2;
			off1->SetIndex(lo);
		}
	} else {
		// Scan right for strict inequality
		hi += MinValue(step, n - hi);
		step *= 2;
		off1->SetIndex(hi);
		while (hi < n && !op1->Compare(*off1)) {
			lo = hi;
			hi += MinValue(step, n - hi);
			step *= 2;
			off1->SetIndex(hi);
		}
	}

	// Binary search the target area
	while (lo < hi) {
		const auto mid = lo + (hi - lo) / 2;
		off1->SetIndex(mid);
		if (op1->Compare(*off1)) {
			hi = mid;
		} else {
			lo = mid + 1;
		}
	}

	off1->SetIndex(lo);

	return lo;
}

bool IEJoinUnion::NextRow() {
	for (; i < n; ++i) {
		// 12. pos ← P[i]
		auto pos = p[i];
		lrid = li[pos];
		if (lrid < 0) {
			continue;
		}

		// 16. B[pos] ← 1
		op2->SetIndex(i);
		for (; off2->GetIndex() < n; ++(*off2)) {
			if (!off2->Compare(*op2)) {
				break;
			}
			const auto p2 = p[off2->GetIndex()];
			if (li[p2] < 0) {
				// Only mark rhs matches.
				bit_mask.SetValid(p2);
				bloom_filter.SetValid(p2 / BLOOM_CHUNK_BITS);
			}
		}

		// 9.  if (op1 ∈ {≤,≥} and op2 ∈ {≤,≥}) eqOff = 0
		// 10. else eqOff = 1
		// No, because there could be more than one equal value.
		// Find the leftmost off1 where L1[pos] op1 L1[off1..n]
		// These are the rows that satisfy the op1 condition
		// and that is where we should start scanning B from
		j = SearchL1(pos);

		return true;
	}
	return false;
}

static idx_t NextValid(const ValidityMask &bits, idx_t j, const idx_t n) {
	if (j >= n) {
		return n;
	}

	// We can do a first approximation by checking entries one at a time
	// which gives 64:1.
	idx_t entry_idx, idx_in_entry;
	bits.GetEntryIndex(j, entry_idx, idx_in_entry);
	auto entry = bits.GetValidityEntry(entry_idx++);

	// Trim the bits before the start position
	entry &= (ValidityMask::ValidityBuffer::MAX_ENTRY << idx_in_entry);

	// Check the non-ragged entries
	for (const auto entry_count = bits.EntryCount(n); entry_idx < entry_count; ++entry_idx) {
		if (entry) {
			for (; idx_in_entry < bits.BITS_PER_VALUE; ++idx_in_entry, ++j) {
				if (bits.RowIsValid(entry, idx_in_entry)) {
					return j;
				}
			}
		} else {
			j += bits.BITS_PER_VALUE - idx_in_entry;
		}

		entry = bits.GetValidityEntry(entry_idx);
		idx_in_entry = 0;
	}

	// Check the final entry
	for (; j < n; ++idx_in_entry, ++j) {
		if (bits.RowIsValid(entry, idx_in_entry)) {
			return j;
		}
	}

	return j;
}

idx_t IEJoinUnion::JoinComplexBlocks(SelectionVector &lsel, SelectionVector &rsel) {
	// 8. initialize join result as an empty list for tuple pairs
	idx_t result_count = 0;

	// 11. for(i←1 to n) do
	while (i < n) {
		// 13. for (j ← pos+eqOff to n) do
		for (;;) {
			// 14. if B[j] = 1 then

			//	Use the Bloom filter to find candidate blocks
			while (j < n) {
				auto bloom_begin = NextValid(bloom_filter, j / BLOOM_CHUNK_BITS, bloom_count) * BLOOM_CHUNK_BITS;
				auto bloom_end = MinValue<idx_t>(n, bloom_begin + BLOOM_CHUNK_BITS);

				j = MaxValue<idx_t>(j, bloom_begin);
				j = NextValid(bit_mask, j, bloom_end);
				if (j < bloom_end) {
					break;
				}
			}

			if (j >= n) {
				break;
			}

			// Filter out tuples with the same sign (they come from the same table)
			const auto rrid = li[j];
			++j;

			// 15. add tuples w.r.t. (L1[j], L1[i]) to join result
			if (lrid > 0 && rrid < 0) {
				lsel.set_index(result_count, sel_t(+lrid - 1));
				rsel.set_index(result_count, sel_t(-rrid - 1));
				++result_count;
				if (result_count == STANDARD_VECTOR_SIZE) {
					// out of space!
					return result_count;
				}
			}
		}
		++i;

		if (!NextRow()) {
			break;
		}
	}

	return result_count;
}

class IEJoinLocalSourceState : public LocalSourceState {
public:
	explicit IEJoinLocalSourceState(ClientContext &context, const PhysicalIEJoin &op)
	    : op(op), true_sel(STANDARD_VECTOR_SIZE), left_executor(context), right_executor(context),
	      left_matches(nullptr), right_matches(nullptr) {
		auto &allocator = Allocator::Get(context);
		if (op.conditions.size() < 3) {
			return;
		}

		vector<LogicalType> left_types;
		vector<LogicalType> right_types;
		for (idx_t i = 2; i < op.conditions.size(); ++i) {
			const auto &cond = op.conditions[i];

			left_types.push_back(cond.left->return_type);
			left_executor.AddExpression(*cond.left);

			right_types.push_back(cond.left->return_type);
			right_executor.AddExpression(*cond.right);
		}

		left_keys.Initialize(allocator, left_types);
		right_keys.Initialize(allocator, right_types);
	}

	idx_t SelectOuterRows(bool *matches) {
		idx_t count = 0;
		for (; outer_idx < outer_count; ++outer_idx) {
			if (!matches[outer_idx]) {
				true_sel.set_index(count++, outer_idx);
				if (count >= STANDARD_VECTOR_SIZE) {
					outer_idx++;
					break;
				}
			}
		}

		return count;
	}

	const PhysicalIEJoin &op;

	// Joining
	unique_ptr<IEJoinUnion> joiner;

	idx_t left_base;
	idx_t left_block_index;

	idx_t right_base;
	idx_t right_block_index;

	// Trailing predicates
	SelectionVector true_sel;

	ExpressionExecutor left_executor;
	DataChunk left_keys;

	ExpressionExecutor right_executor;
	DataChunk right_keys;

	// Outer joins
	idx_t outer_idx;
	idx_t outer_count;
	bool *left_matches;
	bool *right_matches;
};

void PhysicalIEJoin::ResolveComplexJoin(ExecutionContext &context, DataChunk &chunk, LocalSourceState &state_p) const {
	auto &state = state_p.Cast<IEJoinLocalSourceState>();
	auto &ie_sink = sink_state->Cast<IEJoinGlobalState>();
	auto &left_table = *ie_sink.tables[0];
	auto &right_table = *ie_sink.tables[1];

	const auto left_cols = children[0]->GetTypes().size();
	do {
		SelectionVector lsel(STANDARD_VECTOR_SIZE);
		SelectionVector rsel(STANDARD_VECTOR_SIZE);
		auto result_count = state.joiner->JoinComplexBlocks(lsel, rsel);
		if (result_count == 0) {
			// exhausted this pair
			return;
		}

		// found matches: extract them
		chunk.Reset();
		SliceSortedPayload(chunk, left_table.global_sort_state, state.left_block_index, lsel, result_count, 0);
		SliceSortedPayload(chunk, right_table.global_sort_state, state.right_block_index, rsel, result_count,
		                   left_cols);
		chunk.SetCardinality(result_count);

		auto sel = FlatVector::IncrementalSelectionVector();
		if (conditions.size() > 2) {
			// If there are more expressions to compute,
			// split the result chunk into the left and right halves
			// so we can compute the values for comparison.
			const auto tail_cols = conditions.size() - 2;

			DataChunk right_chunk;
			chunk.Split(right_chunk, left_cols);
			state.left_executor.SetChunk(chunk);
			state.right_executor.SetChunk(right_chunk);

			auto tail_count = result_count;
			auto true_sel = &state.true_sel;
			for (size_t cmp_idx = 0; cmp_idx < tail_cols; ++cmp_idx) {
				auto &left = state.left_keys.data[cmp_idx];
				state.left_executor.ExecuteExpression(cmp_idx, left);

				auto &right = state.right_keys.data[cmp_idx];
				state.right_executor.ExecuteExpression(cmp_idx, right);

				if (tail_count < result_count) {
					left.Slice(*sel, tail_count);
					right.Slice(*sel, tail_count);
				}
				tail_count = SelectJoinTail(conditions[cmp_idx + 2].comparison, left, right, sel, tail_count, true_sel);
				sel = true_sel;
			}
			chunk.Fuse(right_chunk);

			if (tail_count < result_count) {
				result_count = tail_count;
				chunk.Slice(*sel, result_count);
			}
		}

		// found matches: mark the found matches if required
		if (left_table.found_match) {
			for (idx_t i = 0; i < result_count; i++) {
				left_table.found_match[state.left_base + lsel[sel->get_index(i)]] = true;
			}
		}
		if (right_table.found_match) {
			for (idx_t i = 0; i < result_count; i++) {
				right_table.found_match[state.right_base + rsel[sel->get_index(i)]] = true;
			}
		}
		chunk.Verify();
	} while (chunk.size() == 0);
}

class IEJoinGlobalSourceState : public GlobalSourceState {
public:
	explicit IEJoinGlobalSourceState(const PhysicalIEJoin &op)
	    : op(op), initialized(false), next_pair(0), completed(0), left_outers(0), next_left(0), right_outers(0),
	      next_right(0) {
	}

	void Initialize(IEJoinGlobalState &sink_state) {
		lock_guard<mutex> initializing(lock);
		if (initialized) {
			return;
		}

		// Compute the starting row for reach block
		// (In theory these are all the same size, but you never know...)
		auto &left_table = *sink_state.tables[0];
		const auto left_blocks = left_table.BlockCount();
		idx_t left_base = 0;

		for (size_t lhs = 0; lhs < left_blocks; ++lhs) {
			left_bases.emplace_back(left_base);
			left_base += left_table.BlockSize(lhs);
		}

		auto &right_table = *sink_state.tables[1];
		const auto right_blocks = right_table.BlockCount();
		idx_t right_base = 0;
		for (size_t rhs = 0; rhs < right_blocks; ++rhs) {
			right_bases.emplace_back(right_base);
			right_base += right_table.BlockSize(rhs);
		}

		// Outer join block counts
		if (left_table.found_match) {
			left_outers = left_blocks;
		}

		if (right_table.found_match) {
			right_outers = right_blocks;
		}

		// Ready for action
		initialized = true;
	}

public:
	idx_t MaxThreads() override {
		// We can't leverage any more threads than block pairs.
		const auto &sink_state = (op.sink_state->Cast<IEJoinGlobalState>());
		return sink_state.tables[0]->BlockCount() * sink_state.tables[1]->BlockCount();
	}

	void GetNextPair(ClientContext &client, IEJoinGlobalState &gstate, IEJoinLocalSourceState &lstate) {
		auto &left_table = *gstate.tables[0];
		auto &right_table = *gstate.tables[1];

		const auto left_blocks = left_table.BlockCount();
		const auto right_blocks = right_table.BlockCount();
		const auto pair_count = left_blocks * right_blocks;

		// Regular block
		const auto i = next_pair++;
		if (i < pair_count) {
			const auto b1 = i / right_blocks;
			const auto b2 = i % right_blocks;

			lstate.left_block_index = b1;
			lstate.left_base = left_bases[b1];

			lstate.right_block_index = b2;
			lstate.right_base = right_bases[b2];

			lstate.joiner = make_uniq<IEJoinUnion>(client, op, left_table, b1, right_table, b2);
			return;
		}

		// Outer joins
		if (!left_outers && !right_outers) {
			return;
		}

		// Spin wait for regular blocks to finish(!)
		while (completed < pair_count) {
			std::this_thread::yield();
		}

		// Left outer blocks
		const auto l = next_left++;
		if (l < left_outers) {
			lstate.joiner = nullptr;
			lstate.left_block_index = l;
			lstate.left_base = left_bases[l];

			lstate.left_matches = left_table.found_match.get() + lstate.left_base;
			lstate.outer_idx = 0;
			lstate.outer_count = left_table.BlockSize(l);
			return;
		} else {
			lstate.left_matches = nullptr;
		}

		// Right outer block
		const auto r = next_right++;
		if (r < right_outers) {
			lstate.joiner = nullptr;
			lstate.right_block_index = r;
			lstate.right_base = right_bases[r];

			lstate.right_matches = right_table.found_match.get() + lstate.right_base;
			lstate.outer_idx = 0;
			lstate.outer_count = right_table.BlockSize(r);
			return;
		} else {
			lstate.right_matches = nullptr;
		}
	}

	void PairCompleted(ClientContext &client, IEJoinGlobalState &gstate, IEJoinLocalSourceState &lstate) {
		lstate.joiner.reset();
		++completed;
		GetNextPair(client, gstate, lstate);
	}

	const PhysicalIEJoin &op;

	mutex lock;
	bool initialized;

	// Join queue state
	std::atomic<size_t> next_pair;
	std::atomic<size_t> completed;

	// Block base row number
	vector<idx_t> left_bases;
	vector<idx_t> right_bases;

	// Outer joins
	idx_t left_outers;
	std::atomic<idx_t> next_left;

	idx_t right_outers;
	std::atomic<idx_t> next_right;
};

unique_ptr<GlobalSourceState> PhysicalIEJoin::GetGlobalSourceState(ClientContext &context) const {
	return make_uniq<IEJoinGlobalSourceState>(*this);
}

unique_ptr<LocalSourceState> PhysicalIEJoin::GetLocalSourceState(ExecutionContext &context,
                                                                 GlobalSourceState &gstate) const {
	return make_uniq<IEJoinLocalSourceState>(context.client, *this);
}

SourceResultType PhysicalIEJoin::GetData(ExecutionContext &context, DataChunk &result,
                                         OperatorSourceInput &input) const {
	auto &ie_sink = sink_state->Cast<IEJoinGlobalState>();
	auto &ie_gstate = input.global_state.Cast<IEJoinGlobalSourceState>();
	auto &ie_lstate = input.local_state.Cast<IEJoinLocalSourceState>();

	ie_gstate.Initialize(ie_sink);

	if (!ie_lstate.joiner && !ie_lstate.left_matches && !ie_lstate.right_matches) {
		ie_gstate.GetNextPair(context.client, ie_sink, ie_lstate);
	}

	// Process INNER results
	while (ie_lstate.joiner) {
		ResolveComplexJoin(context, result, ie_lstate);

		if (result.size()) {
			return SourceResultType::HAVE_MORE_OUTPUT;
		}

		ie_gstate.PairCompleted(context.client, ie_sink, ie_lstate);
	}

	// Process LEFT OUTER results
	const auto left_cols = children[0]->GetTypes().size();
	while (ie_lstate.left_matches) {
		const idx_t count = ie_lstate.SelectOuterRows(ie_lstate.left_matches);
		if (!count) {
			ie_gstate.GetNextPair(context.client, ie_sink, ie_lstate);
			continue;
		}
		SliceSortedPayload(result, ie_sink.tables[0]->global_sort_state, ie_lstate.left_block_index, ie_lstate.true_sel,
		                   count);

		// Fill in NULLs to the right
		for (auto col_idx = left_cols; col_idx < result.ColumnCount(); ++col_idx) {
			result.data[col_idx].SetVectorType(VectorType::CONSTANT_VECTOR);
			ConstantVector::SetNull(result.data[col_idx], true);
		}

		result.SetCardinality(count);
		result.Verify();

		return result.size() == 0 ? SourceResultType::FINISHED : SourceResultType::HAVE_MORE_OUTPUT;
	}

	// Process RIGHT OUTER results
	while (ie_lstate.right_matches) {
		const idx_t count = ie_lstate.SelectOuterRows(ie_lstate.right_matches);
		if (!count) {
			ie_gstate.GetNextPair(context.client, ie_sink, ie_lstate);
			continue;
		}

		SliceSortedPayload(result, ie_sink.tables[1]->global_sort_state, ie_lstate.right_block_index,
		                   ie_lstate.true_sel, count, left_cols);

		// Fill in NULLs to the left
		for (idx_t col_idx = 0; col_idx < left_cols; ++col_idx) {
			result.data[col_idx].SetVectorType(VectorType::CONSTANT_VECTOR);
			ConstantVector::SetNull(result.data[col_idx], true);
		}

		result.SetCardinality(count);
		result.Verify();

		break;
	}

	return result.size() == 0 ? SourceResultType::FINISHED : SourceResultType::HAVE_MORE_OUTPUT;
}

//===--------------------------------------------------------------------===//
// Pipeline Construction
//===--------------------------------------------------------------------===//
void PhysicalIEJoin::BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) {
	D_ASSERT(children.size() == 2);
	if (meta_pipeline.HasRecursiveCTE()) {
		throw NotImplementedException("IEJoins are not supported in recursive CTEs yet");
	}

	// becomes a source after both children fully sink their data
	meta_pipeline.GetState().SetPipelineSource(current, *this);

	// Create one child meta pipeline that will hold the LHS and RHS pipelines
	auto &child_meta_pipeline = meta_pipeline.CreateChildMetaPipeline(current, *this);
	auto lhs_pipeline = child_meta_pipeline.GetBasePipeline();
	auto rhs_pipeline = child_meta_pipeline.CreatePipeline();

	// Build out LHS
	children[0]->BuildPipelines(*lhs_pipeline, child_meta_pipeline);

	// RHS depends on everything in LHS
	child_meta_pipeline.AddDependenciesFrom(rhs_pipeline, lhs_pipeline.get(), true);

	// Build out RHS
	children[1]->BuildPipelines(*rhs_pipeline, child_meta_pipeline);

	// Despite having the same sink, RHS needs its own PipelineFinishEvent
	child_meta_pipeline.AddFinishEvent(rhs_pipeline);
}

} // namespace duckdb

















namespace duckdb {

class IndexJoinOperatorState : public CachingOperatorState {
public:
	IndexJoinOperatorState(ClientContext &context, const PhysicalIndexJoin &op)
	    : probe_executor(context), arena_allocator(BufferAllocator::Get(context)), keys(STANDARD_VECTOR_SIZE) {
		auto &allocator = Allocator::Get(context);
		rhs_rows.resize(STANDARD_VECTOR_SIZE);
		result_sizes.resize(STANDARD_VECTOR_SIZE);

		join_keys.Initialize(allocator, op.condition_types);
		for (auto &cond : op.conditions) {
			probe_executor.AddExpression(*cond.left);
		}
		if (!op.fetch_types.empty()) {
			rhs_chunk.Initialize(allocator, op.fetch_types);
		}
		rhs_sel.Initialize(STANDARD_VECTOR_SIZE);
	}

	bool first_fetch = true;
	idx_t lhs_idx = 0;
	idx_t rhs_idx = 0;
	idx_t result_size = 0;
	vector<idx_t> result_sizes;
	DataChunk join_keys;
	DataChunk rhs_chunk;
	SelectionVector rhs_sel;

	//! Vector of rows that mush be fetched for every LHS key
	vector<vector<row_t>> rhs_rows;
	ExpressionExecutor probe_executor;

	ArenaAllocator arena_allocator;
	vector<ARTKey> keys;
	unique_ptr<ColumnFetchState> fetch_state;

public:
	void Finalize(const PhysicalOperator &op, ExecutionContext &context) override {
		context.thread.profiler.Flush(op, probe_executor, "probe_executor", 0);
	}
};

PhysicalIndexJoin::PhysicalIndexJoin(LogicalOperator &op, unique_ptr<PhysicalOperator> left,
                                     unique_ptr<PhysicalOperator> right, vector<JoinCondition> cond, JoinType join_type,
                                     const vector<idx_t> &left_projection_map_p, vector<idx_t> right_projection_map_p,
                                     vector<column_t> column_ids_p, Index &index_p, bool lhs_first,
                                     idx_t estimated_cardinality)
    : CachingPhysicalOperator(PhysicalOperatorType::INDEX_JOIN, std::move(op.types), estimated_cardinality),
      left_projection_map(left_projection_map_p), right_projection_map(std::move(right_projection_map_p)),
      index(index_p), conditions(std::move(cond)), join_type(join_type), lhs_first(lhs_first) {
	D_ASSERT(right->type == PhysicalOperatorType::TABLE_SCAN);
	auto &tbl_scan = (PhysicalTableScan &)*right;
	column_ids = std::move(column_ids_p);
	children.push_back(std::move(left));
	children.push_back(std::move(right));
	for (auto &condition : conditions) {
		condition_types.push_back(condition.left->return_type);
	}
	//! Only add to fetch_ids columns that are not indexed
	for (auto &index_id : index.column_ids) {
		index_ids.insert(index_id);
	}

	for (idx_t i = 0; i < column_ids.size(); i++) {
		auto column_id = column_ids[i];
		auto it = index_ids.find(column_id);
		if (it == index_ids.end()) {
			fetch_ids.push_back(column_id);
			if (column_id == COLUMN_IDENTIFIER_ROW_ID) {
				fetch_types.emplace_back(LogicalType::ROW_TYPE);
			} else {
				fetch_types.push_back(tbl_scan.returned_types[column_id]);
			}
		}
	}
	if (right_projection_map.empty()) {
		for (column_t i = 0; i < column_ids.size(); i++) {
			right_projection_map.push_back(i);
		}
	}
	if (left_projection_map.empty()) {
		for (column_t i = 0; i < children[0]->types.size(); i++) {
			left_projection_map.push_back(i);
		}
	}
}

unique_ptr<OperatorState> PhysicalIndexJoin::GetOperatorState(ExecutionContext &context) const {
	return make_uniq<IndexJoinOperatorState>(context.client, *this);
}

void PhysicalIndexJoin::Output(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                               OperatorState &state_p) const {
	auto &phy_tbl_scan = (PhysicalTableScan &)*children[1];
	auto &bind_tbl = phy_tbl_scan.bind_data->Cast<TableScanBindData>();
	auto &transaction = DuckTransaction::Get(context.client, bind_tbl.table.catalog);
	auto &state = state_p.Cast<IndexJoinOperatorState>();

	auto &tbl = bind_tbl.table.GetStorage();
	idx_t output_sel_idx = 0;
	vector<row_t> fetch_rows;

	while (output_sel_idx < STANDARD_VECTOR_SIZE && state.lhs_idx < input.size()) {
		if (state.rhs_idx < state.result_sizes[state.lhs_idx]) {
			state.rhs_sel.set_index(output_sel_idx++, state.lhs_idx);
			if (!fetch_types.empty()) {
				//! We need to collect the rows we want to fetch
				fetch_rows.push_back(state.rhs_rows[state.lhs_idx][state.rhs_idx]);
			}
			state.rhs_idx++;
		} else {
			//! We are done with the matches from this LHS Key
			state.rhs_idx = 0;
			state.lhs_idx++;
		}
	}
	//! Now we fetch the RHS data
	if (!fetch_types.empty()) {
		if (fetch_rows.empty()) {
			return;
		}
		state.rhs_chunk.Reset();
		state.fetch_state = make_uniq<ColumnFetchState>();
		Vector row_ids(LogicalType::ROW_TYPE, (data_ptr_t)&fetch_rows[0]);
		tbl.Fetch(transaction, state.rhs_chunk, fetch_ids, row_ids, output_sel_idx, *state.fetch_state);
	}

	//! Now we actually produce our result chunk
	idx_t left_offset = lhs_first ? 0 : right_projection_map.size();
	idx_t right_offset = lhs_first ? left_projection_map.size() : 0;
	idx_t rhs_column_idx = 0;
	for (idx_t i = 0; i < right_projection_map.size(); i++) {
		auto it = index_ids.find(column_ids[right_projection_map[i]]);
		if (it == index_ids.end()) {
			chunk.data[right_offset + i].Reference(state.rhs_chunk.data[rhs_column_idx++]);
		} else {
			chunk.data[right_offset + i].Slice(state.join_keys.data[0], state.rhs_sel, output_sel_idx);
		}
	}
	for (idx_t i = 0; i < left_projection_map.size(); i++) {
		chunk.data[left_offset + i].Slice(input.data[left_projection_map[i]], state.rhs_sel, output_sel_idx);
	}

	state.result_size = output_sel_idx;
	chunk.SetCardinality(state.result_size);
}

void PhysicalIndexJoin::GetRHSMatches(ExecutionContext &context, DataChunk &input, OperatorState &state_p) const {

	auto &state = state_p.Cast<IndexJoinOperatorState>();
	auto &art = index.Cast<ART>();

	// generate the keys for this chunk
	state.arena_allocator.Reset();
	ART::GenerateKeys(state.arena_allocator, state.join_keys, state.keys);

	for (idx_t i = 0; i < input.size(); i++) {
		state.rhs_rows[i].clear();
		if (!state.keys[i].Empty()) {
			if (fetch_types.empty()) {
				IndexLock lock;
				index.InitializeLock(lock);
				art.SearchEqualJoinNoFetch(state.keys[i], state.result_sizes[i]);
			} else {
				IndexLock lock;
				index.InitializeLock(lock);
				art.SearchEqual(state.keys[i], (idx_t)-1, state.rhs_rows[i]);
				state.result_sizes[i] = state.rhs_rows[i].size();
			}
		} else {
			//! This is null so no matches
			state.result_sizes[i] = 0;
		}
	}
	for (idx_t i = input.size(); i < STANDARD_VECTOR_SIZE; i++) {
		//! No LHS chunk value so result size is empty
		state.result_sizes[i] = 0;
	}
}

OperatorResultType PhysicalIndexJoin::ExecuteInternal(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                                      GlobalOperatorState &gstate, OperatorState &state_p) const {
	auto &state = state_p.Cast<IndexJoinOperatorState>();

	state.result_size = 0;
	if (state.first_fetch) {
		state.probe_executor.Execute(input, state.join_keys);

		//! Fill Matches for the current LHS chunk
		GetRHSMatches(context, input, state_p);
		state.first_fetch = false;
	}
	//! Check if we need to get a new LHS chunk
	if (state.lhs_idx >= input.size()) {
		state.lhs_idx = 0;
		state.rhs_idx = 0;
		state.first_fetch = true;
		// reset the LHS chunk to reset the validity masks
		state.join_keys.Reset();
		return OperatorResultType::NEED_MORE_INPUT;
	}
	//! Output vectors
	if (state.lhs_idx < input.size()) {
		Output(context, input, chunk, state_p);
	}
	return OperatorResultType::HAVE_MORE_OUTPUT;
}

//===--------------------------------------------------------------------===//
// Pipeline Construction
//===--------------------------------------------------------------------===//
void PhysicalIndexJoin::BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) {
	// index join: we only continue into the LHS
	// the right side is probed by the index join
	// so we don't need to do anything in the pipeline with this child
	meta_pipeline.GetState().AddPipelineOperator(current, *this);
	children[0]->BuildPipelines(current, meta_pipeline);
}

vector<const_reference<PhysicalOperator>> PhysicalIndexJoin::GetSources() const {
	return children[0]->GetSources();
}

} // namespace duckdb






namespace duckdb {

PhysicalJoin::PhysicalJoin(LogicalOperator &op, PhysicalOperatorType type, JoinType join_type,
                           idx_t estimated_cardinality)
    : CachingPhysicalOperator(type, op.types, estimated_cardinality), join_type(join_type) {
}

bool PhysicalJoin::EmptyResultIfRHSIsEmpty() const {
	// empty RHS with INNER, RIGHT or SEMI join means empty result set
	switch (join_type) {
	case JoinType::INNER:
	case JoinType::RIGHT:
	case JoinType::SEMI:
		return true;
	default:
		return false;
	}
}

//===--------------------------------------------------------------------===//
// Pipeline Construction
//===--------------------------------------------------------------------===//
void PhysicalJoin::BuildJoinPipelines(Pipeline &current, MetaPipeline &meta_pipeline, PhysicalOperator &op) {
	op.op_state.reset();
	op.sink_state.reset();

	// 'current' is the probe pipeline: add this operator
	auto &state = meta_pipeline.GetState();
	state.AddPipelineOperator(current, op);

	// save the last added pipeline to set up dependencies later (in case we need to add a child pipeline)
	vector<shared_ptr<Pipeline>> pipelines_so_far;
	meta_pipeline.GetPipelines(pipelines_so_far, false);
	auto last_pipeline = pipelines_so_far.back().get();

	// on the RHS (build side), we construct a child MetaPipeline with this operator as its sink
	auto &child_meta_pipeline = meta_pipeline.CreateChildMetaPipeline(current, op);
	child_meta_pipeline.Build(*op.children[1]);

	// continue building the current pipeline on the LHS (probe side)
	op.children[0]->BuildPipelines(current, meta_pipeline);

	switch (op.type) {
	case PhysicalOperatorType::POSITIONAL_JOIN:
		// Positional joins are always outer
		meta_pipeline.CreateChildPipeline(current, op, last_pipeline);
		return;
	case PhysicalOperatorType::CROSS_PRODUCT:
		return;
	default:
		break;
	}

	// Join can become a source operator if it's RIGHT/OUTER, or if the hash join goes out-of-core
	bool add_child_pipeline = false;
	auto &join_op = op.Cast<PhysicalJoin>();
	if (IsRightOuterJoin(join_op.join_type) || join_op.type == PhysicalOperatorType::HASH_JOIN) {
		add_child_pipeline = true;
	}

	if (add_child_pipeline) {
		meta_pipeline.CreateChildPipeline(current, op, last_pipeline);
	}
}

void PhysicalJoin::BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) {
	PhysicalJoin::BuildJoinPipelines(current, meta_pipeline, *this);
}

vector<const_reference<PhysicalOperator>> PhysicalJoin::GetSources() const {
	auto result = children[0]->GetSources();
	if (IsSource()) {
		result.push_back(*this);
	}
	return result;
}

} // namespace duckdb









namespace duckdb {

PhysicalNestedLoopJoin::PhysicalNestedLoopJoin(LogicalOperator &op, unique_ptr<PhysicalOperator> left,
                                               unique_ptr<PhysicalOperator> right, vector<JoinCondition> cond,
                                               JoinType join_type, idx_t estimated_cardinality)
    : PhysicalComparisonJoin(op, PhysicalOperatorType::NESTED_LOOP_JOIN, std::move(cond), join_type,
                             estimated_cardinality) {
	children.push_back(std::move(left));
	children.push_back(std::move(right));
}

bool PhysicalJoin::HasNullValues(DataChunk &chunk) {
	for (idx_t col_idx = 0; col_idx < chunk.ColumnCount(); col_idx++) {
		UnifiedVectorFormat vdata;
		chunk.data[col_idx].ToUnifiedFormat(chunk.size(), vdata);

		if (vdata.validity.AllValid()) {
			continue;
		}
		for (idx_t i = 0; i < chunk.size(); i++) {
			auto idx = vdata.sel->get_index(i);
			if (!vdata.validity.RowIsValid(idx)) {
				return true;
			}
		}
	}
	return false;
}

template <bool MATCH>
static void ConstructSemiOrAntiJoinResult(DataChunk &left, DataChunk &result, bool found_match[]) {
	D_ASSERT(left.ColumnCount() == result.ColumnCount());
	// create the selection vector from the matches that were found
	idx_t result_count = 0;
	SelectionVector sel(STANDARD_VECTOR_SIZE);
	for (idx_t i = 0; i < left.size(); i++) {
		if (found_match[i] == MATCH) {
			sel.set_index(result_count++, i);
		}
	}
	// construct the final result
	if (result_count > 0) {
		// we only return the columns on the left side
		// project them using the result selection vector
		// reference the columns of the left side from the result
		result.Slice(left, sel, result_count);
	} else {
		result.SetCardinality(0);
	}
}

void PhysicalJoin::ConstructSemiJoinResult(DataChunk &left, DataChunk &result, bool found_match[]) {
	ConstructSemiOrAntiJoinResult<true>(left, result, found_match);
}

void PhysicalJoin::ConstructAntiJoinResult(DataChunk &left, DataChunk &result, bool found_match[]) {
	ConstructSemiOrAntiJoinResult<false>(left, result, found_match);
}

void PhysicalJoin::ConstructMarkJoinResult(DataChunk &join_keys, DataChunk &left, DataChunk &result, bool found_match[],
                                           bool has_null) {
	// for the initial set of columns we just reference the left side
	result.SetCardinality(left);
	for (idx_t i = 0; i < left.ColumnCount(); i++) {
		result.data[i].Reference(left.data[i]);
	}
	auto &mark_vector = result.data.back();
	mark_vector.SetVectorType(VectorType::FLAT_VECTOR);
	// first we set the NULL values from the join keys
	// if there is any NULL in the keys, the result is NULL
	auto bool_result = FlatVector::GetData<bool>(mark_vector);
	auto &mask = FlatVector::Validity(mark_vector);
	for (idx_t col_idx = 0; col_idx < join_keys.ColumnCount(); col_idx++) {
		UnifiedVectorFormat jdata;
		join_keys.data[col_idx].ToUnifiedFormat(join_keys.size(), jdata);
		if (!jdata.validity.AllValid()) {
			for (idx_t i = 0; i < join_keys.size(); i++) {
				auto jidx = jdata.sel->get_index(i);
				mask.Set(i, jdata.validity.RowIsValid(jidx));
			}
		}
	}
	// now set the remaining entries to either true or false based on whether a match was found
	if (found_match) {
		for (idx_t i = 0; i < left.size(); i++) {
			bool_result[i] = found_match[i];
		}
	} else {
		memset(bool_result, 0, sizeof(bool) * left.size());
	}
	// if the right side contains NULL values, the result of any FALSE becomes NULL
	if (has_null) {
		for (idx_t i = 0; i < left.size(); i++) {
			if (!bool_result[i]) {
				mask.SetInvalid(i);
			}
		}
	}
}

bool PhysicalNestedLoopJoin::IsSupported(const vector<JoinCondition> &conditions, JoinType join_type) {
	if (join_type == JoinType::MARK) {
		return true;
	}
	for (auto &cond : conditions) {
		if (cond.left->return_type.InternalType() == PhysicalType::STRUCT ||
		    cond.left->return_type.InternalType() == PhysicalType::LIST) {
			return false;
		}
	}
	return true;
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
class NestedLoopJoinLocalState : public LocalSinkState {
public:
	explicit NestedLoopJoinLocalState(ClientContext &context, const vector<JoinCondition> &conditions)
	    : rhs_executor(context) {
		vector<LogicalType> condition_types;
		for (auto &cond : conditions) {
			rhs_executor.AddExpression(*cond.right);
			condition_types.push_back(cond.right->return_type);
		}
		right_condition.Initialize(Allocator::Get(context), condition_types);
	}

	//! The chunk holding the right condition
	DataChunk right_condition;
	//! The executor of the RHS condition
	ExpressionExecutor rhs_executor;
};

class NestedLoopJoinGlobalState : public GlobalSinkState {
public:
	explicit NestedLoopJoinGlobalState(ClientContext &context, const PhysicalNestedLoopJoin &op)
	    : right_payload_data(context, op.children[1]->types), right_condition_data(context, op.GetJoinTypes()),
	      has_null(false), right_outer(IsRightOuterJoin(op.join_type)) {
	}

	mutex nj_lock;
	//! Materialized data of the RHS
	ColumnDataCollection right_payload_data;
	//! Materialized join condition of the RHS
	ColumnDataCollection right_condition_data;
	//! Whether or not the RHS of the nested loop join has NULL values
	atomic<bool> has_null;
	//! A bool indicating for each tuple in the RHS if they found a match (only used in FULL OUTER JOIN)
	OuterJoinMarker right_outer;
};

vector<LogicalType> PhysicalNestedLoopJoin::GetJoinTypes() const {
	vector<LogicalType> result;
	for (auto &op : conditions) {
		result.push_back(op.right->return_type);
	}
	return result;
}

SinkResultType PhysicalNestedLoopJoin::Sink(ExecutionContext &context, DataChunk &chunk,
                                            OperatorSinkInput &input) const {
	auto &gstate = input.global_state.Cast<NestedLoopJoinGlobalState>();
	auto &nlj_state = input.local_state.Cast<NestedLoopJoinLocalState>();

	// resolve the join expression of the right side
	nlj_state.right_condition.Reset();
	nlj_state.rhs_executor.Execute(chunk, nlj_state.right_condition);

	// if we have not seen any NULL values yet, and we are performing a MARK join, check if there are NULL values in
	// this chunk
	if (join_type == JoinType::MARK && !gstate.has_null) {
		if (HasNullValues(nlj_state.right_condition)) {
			gstate.has_null = true;
		}
	}

	// append the payload data and the conditions
	lock_guard<mutex> nj_guard(gstate.nj_lock);
	gstate.right_payload_data.Append(chunk);
	gstate.right_condition_data.Append(nlj_state.right_condition);
	return SinkResultType::NEED_MORE_INPUT;
}

void PhysicalNestedLoopJoin::Combine(ExecutionContext &context, GlobalSinkState &gstate, LocalSinkState &lstate) const {
	auto &state = lstate.Cast<NestedLoopJoinLocalState>();
	auto &client_profiler = QueryProfiler::Get(context.client);

	context.thread.profiler.Flush(*this, state.rhs_executor, "rhs_executor", 1);
	client_profiler.Flush(context.thread.profiler);
}

SinkFinalizeType PhysicalNestedLoopJoin::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                                  GlobalSinkState &gstate_p) const {
	auto &gstate = gstate_p.Cast<NestedLoopJoinGlobalState>();
	gstate.right_outer.Initialize(gstate.right_payload_data.Count());
	if (gstate.right_payload_data.Count() == 0 && EmptyResultIfRHSIsEmpty()) {
		return SinkFinalizeType::NO_OUTPUT_POSSIBLE;
	}
	return SinkFinalizeType::READY;
}

unique_ptr<GlobalSinkState> PhysicalNestedLoopJoin::GetGlobalSinkState(ClientContext &context) const {
	return make_uniq<NestedLoopJoinGlobalState>(context, *this);
}

unique_ptr<LocalSinkState> PhysicalNestedLoopJoin::GetLocalSinkState(ExecutionContext &context) const {
	return make_uniq<NestedLoopJoinLocalState>(context.client, conditions);
}

//===--------------------------------------------------------------------===//
// Operator
//===--------------------------------------------------------------------===//
class PhysicalNestedLoopJoinState : public CachingOperatorState {
public:
	PhysicalNestedLoopJoinState(ClientContext &context, const PhysicalNestedLoopJoin &op,
	                            const vector<JoinCondition> &conditions)
	    : fetch_next_left(true), fetch_next_right(false), lhs_executor(context), left_tuple(0), right_tuple(0),
	      left_outer(IsLeftOuterJoin(op.join_type)) {
		vector<LogicalType> condition_types;
		for (auto &cond : conditions) {
			lhs_executor.AddExpression(*cond.left);
			condition_types.push_back(cond.left->return_type);
		}
		auto &allocator = Allocator::Get(context);
		left_condition.Initialize(allocator, condition_types);
		right_condition.Initialize(allocator, condition_types);
		right_payload.Initialize(allocator, op.children[1]->GetTypes());
		left_outer.Initialize(STANDARD_VECTOR_SIZE);
	}

	bool fetch_next_left;
	bool fetch_next_right;
	DataChunk left_condition;
	//! The executor of the LHS condition
	ExpressionExecutor lhs_executor;

	ColumnDataScanState condition_scan_state;
	ColumnDataScanState payload_scan_state;
	DataChunk right_condition;
	DataChunk right_payload;

	idx_t left_tuple;
	idx_t right_tuple;

	OuterJoinMarker left_outer;

public:
	void Finalize(const PhysicalOperator &op, ExecutionContext &context) override {
		context.thread.profiler.Flush(op, lhs_executor, "lhs_executor", 0);
	}
};

unique_ptr<OperatorState> PhysicalNestedLoopJoin::GetOperatorState(ExecutionContext &context) const {
	return make_uniq<PhysicalNestedLoopJoinState>(context.client, *this, conditions);
}

OperatorResultType PhysicalNestedLoopJoin::ExecuteInternal(ExecutionContext &context, DataChunk &input,
                                                           DataChunk &chunk, GlobalOperatorState &gstate_p,
                                                           OperatorState &state_p) const {
	auto &gstate = sink_state->Cast<NestedLoopJoinGlobalState>();

	if (gstate.right_payload_data.Count() == 0) {
		// empty RHS
		if (!EmptyResultIfRHSIsEmpty()) {
			ConstructEmptyJoinResult(join_type, gstate.has_null, input, chunk);
			return OperatorResultType::NEED_MORE_INPUT;
		} else {
			return OperatorResultType::FINISHED;
		}
	}

	switch (join_type) {
	case JoinType::SEMI:
	case JoinType::ANTI:
	case JoinType::MARK:
		// simple joins can have max STANDARD_VECTOR_SIZE matches per chunk
		ResolveSimpleJoin(context, input, chunk, state_p);
		return OperatorResultType::NEED_MORE_INPUT;
	case JoinType::LEFT:
	case JoinType::INNER:
	case JoinType::OUTER:
	case JoinType::RIGHT:
		return ResolveComplexJoin(context, input, chunk, state_p);
	default:
		throw NotImplementedException("Unimplemented type for nested loop join!");
	}
}

void PhysicalNestedLoopJoin::ResolveSimpleJoin(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                               OperatorState &state_p) const {
	auto &state = state_p.Cast<PhysicalNestedLoopJoinState>();
	auto &gstate = sink_state->Cast<NestedLoopJoinGlobalState>();

	// resolve the left join condition for the current chunk
	state.left_condition.Reset();
	state.lhs_executor.Execute(input, state.left_condition);

	bool found_match[STANDARD_VECTOR_SIZE] = {false};
	NestedLoopJoinMark::Perform(state.left_condition, gstate.right_condition_data, found_match, conditions);
	switch (join_type) {
	case JoinType::MARK:
		// now construct the mark join result from the found matches
		PhysicalJoin::ConstructMarkJoinResult(state.left_condition, input, chunk, found_match, gstate.has_null);
		break;
	case JoinType::SEMI:
		// construct the semi join result from the found matches
		PhysicalJoin::ConstructSemiJoinResult(input, chunk, found_match);
		break;
	case JoinType::ANTI:
		// construct the anti join result from the found matches
		PhysicalJoin::ConstructAntiJoinResult(input, chunk, found_match);
		break;
	default:
		throw NotImplementedException("Unimplemented type for simple nested loop join!");
	}
}

OperatorResultType PhysicalNestedLoopJoin::ResolveComplexJoin(ExecutionContext &context, DataChunk &input,
                                                              DataChunk &chunk, OperatorState &state_p) const {
	auto &state = state_p.Cast<PhysicalNestedLoopJoinState>();
	auto &gstate = sink_state->Cast<NestedLoopJoinGlobalState>();

	idx_t match_count;
	do {
		if (state.fetch_next_right) {
			// we exhausted the chunk on the right: move to the next chunk on the right
			state.left_tuple = 0;
			state.right_tuple = 0;
			state.fetch_next_right = false;
			// check if we exhausted all chunks on the RHS
			if (gstate.right_condition_data.Scan(state.condition_scan_state, state.right_condition)) {
				if (!gstate.right_payload_data.Scan(state.payload_scan_state, state.right_payload)) {
					throw InternalException("Nested loop join: payload and conditions are unaligned!?");
				}
				if (state.right_condition.size() != state.right_payload.size()) {
					throw InternalException("Nested loop join: payload and conditions are unaligned!?");
				}
			} else {
				// we exhausted all chunks on the right: move to the next chunk on the left
				state.fetch_next_left = true;
				if (state.left_outer.Enabled()) {
					// left join: before we move to the next chunk, see if we need to output any vectors that didn't
					// have a match found
					state.left_outer.ConstructLeftJoinResult(input, chunk);
					state.left_outer.Reset();
				}
				return OperatorResultType::NEED_MORE_INPUT;
			}
		}
		if (state.fetch_next_left) {
			// resolve the left join condition for the current chunk
			state.left_condition.Reset();
			state.lhs_executor.Execute(input, state.left_condition);

			state.left_tuple = 0;
			state.right_tuple = 0;
			gstate.right_condition_data.InitializeScan(state.condition_scan_state);
			gstate.right_condition_data.Scan(state.condition_scan_state, state.right_condition);

			gstate.right_payload_data.InitializeScan(state.payload_scan_state);
			gstate.right_payload_data.Scan(state.payload_scan_state, state.right_payload);
			state.fetch_next_left = false;
		}
		// now we have a left and a right chunk that we can join together
		// note that we only get here in the case of a LEFT, INNER or FULL join
		auto &left_chunk = input;
		auto &right_condition = state.right_condition;
		auto &right_payload = state.right_payload;

		// sanity check
		left_chunk.Verify();
		right_condition.Verify();
		right_payload.Verify();

		// now perform the join
		SelectionVector lvector(STANDARD_VECTOR_SIZE), rvector(STANDARD_VECTOR_SIZE);
		match_count = NestedLoopJoinInner::Perform(state.left_tuple, state.right_tuple, state.left_condition,
		                                           right_condition, lvector, rvector, conditions);
		// we have finished resolving the join conditions
		if (match_count > 0) {
			// we have matching tuples!
			// construct the result
			state.left_outer.SetMatches(lvector, match_count);
			gstate.right_outer.SetMatches(rvector, match_count, state.condition_scan_state.current_row_index);

			chunk.Slice(input, lvector, match_count);
			chunk.Slice(right_payload, rvector, match_count, input.ColumnCount());
		}

		// check if we exhausted the RHS, if we did we need to move to the next right chunk in the next iteration
		if (state.right_tuple >= right_condition.size()) {
			state.fetch_next_right = true;
		}
	} while (match_count == 0);
	return OperatorResultType::HAVE_MORE_OUTPUT;
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
class NestedLoopJoinGlobalScanState : public GlobalSourceState {
public:
	explicit NestedLoopJoinGlobalScanState(const PhysicalNestedLoopJoin &op) : op(op) {
		D_ASSERT(op.sink_state);
		auto &sink = op.sink_state->Cast<NestedLoopJoinGlobalState>();
		sink.right_outer.InitializeScan(sink.right_payload_data, scan_state);
	}

	const PhysicalNestedLoopJoin &op;
	OuterJoinGlobalScanState scan_state;

public:
	idx_t MaxThreads() override {
		auto &sink = op.sink_state->Cast<NestedLoopJoinGlobalState>();
		return sink.right_outer.MaxThreads();
	}
};

class NestedLoopJoinLocalScanState : public LocalSourceState {
public:
	explicit NestedLoopJoinLocalScanState(const PhysicalNestedLoopJoin &op, NestedLoopJoinGlobalScanState &gstate) {
		D_ASSERT(op.sink_state);
		auto &sink = op.sink_state->Cast<NestedLoopJoinGlobalState>();
		sink.right_outer.InitializeScan(gstate.scan_state, scan_state);
	}

	OuterJoinLocalScanState scan_state;
};

unique_ptr<GlobalSourceState> PhysicalNestedLoopJoin::GetGlobalSourceState(ClientContext &context) const {
	return make_uniq<NestedLoopJoinGlobalScanState>(*this);
}

unique_ptr<LocalSourceState> PhysicalNestedLoopJoin::GetLocalSourceState(ExecutionContext &context,
                                                                         GlobalSourceState &gstate) const {
	return make_uniq<NestedLoopJoinLocalScanState>(*this, (NestedLoopJoinGlobalScanState &)gstate);
}

SourceResultType PhysicalNestedLoopJoin::GetData(ExecutionContext &context, DataChunk &chunk,
                                                 OperatorSourceInput &input) const {
	D_ASSERT(IsRightOuterJoin(join_type));
	// check if we need to scan any unmatched tuples from the RHS for the full/right outer join
	auto &sink = sink_state->Cast<NestedLoopJoinGlobalState>();
	auto &gstate = (NestedLoopJoinGlobalScanState &)input.global_state;
	auto &lstate = (NestedLoopJoinLocalScanState &)input.local_state;

	// if the LHS is exhausted in a FULL/RIGHT OUTER JOIN, we scan chunks we still need to output
	sink.right_outer.Scan(gstate.scan_state, lstate.scan_state, chunk);

	return chunk.size() == 0 ? SourceResultType::FINISHED : SourceResultType::HAVE_MORE_OUTPUT;
}

} // namespace duckdb














namespace duckdb {

PhysicalPiecewiseMergeJoin::PhysicalPiecewiseMergeJoin(LogicalOperator &op, unique_ptr<PhysicalOperator> left,
                                                       unique_ptr<PhysicalOperator> right, vector<JoinCondition> cond,
                                                       JoinType join_type, idx_t estimated_cardinality)
    : PhysicalRangeJoin(op, PhysicalOperatorType::PIECEWISE_MERGE_JOIN, std::move(left), std::move(right),
                        std::move(cond), join_type, estimated_cardinality) {

	for (auto &cond : conditions) {
		D_ASSERT(cond.left->return_type == cond.right->return_type);
		join_key_types.push_back(cond.left->return_type);

		// Convert the conditions to sort orders
		auto left = cond.left->Copy();
		auto right = cond.right->Copy();
		switch (cond.comparison) {
		case ExpressionType::COMPARE_LESSTHAN:
		case ExpressionType::COMPARE_LESSTHANOREQUALTO:
			lhs_orders.emplace_back(OrderType::ASCENDING, OrderByNullType::NULLS_LAST, std::move(left));
			rhs_orders.emplace_back(OrderType::ASCENDING, OrderByNullType::NULLS_LAST, std::move(right));
			break;
		case ExpressionType::COMPARE_GREATERTHAN:
		case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
			lhs_orders.emplace_back(OrderType::DESCENDING, OrderByNullType::NULLS_LAST, std::move(left));
			rhs_orders.emplace_back(OrderType::DESCENDING, OrderByNullType::NULLS_LAST, std::move(right));
			break;
		case ExpressionType::COMPARE_NOTEQUAL:
		case ExpressionType::COMPARE_DISTINCT_FROM:
			// Allowed in multi-predicate joins, but can't be first/sort.
			D_ASSERT(!lhs_orders.empty());
			lhs_orders.emplace_back(OrderType::INVALID, OrderByNullType::NULLS_LAST, std::move(left));
			rhs_orders.emplace_back(OrderType::INVALID, OrderByNullType::NULLS_LAST, std::move(right));
			break;

		default:
			// COMPARE EQUAL not supported with merge join
			throw NotImplementedException("Unimplemented join type for merge join");
		}
	}
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
class MergeJoinLocalState : public LocalSinkState {
public:
	explicit MergeJoinLocalState(ClientContext &context, const PhysicalRangeJoin &op, const idx_t child)
	    : table(context, op, child) {
	}

	//! The local sort state
	PhysicalRangeJoin::LocalSortedTable table;
};

class MergeJoinGlobalState : public GlobalSinkState {
public:
	using GlobalSortedTable = PhysicalRangeJoin::GlobalSortedTable;

public:
	MergeJoinGlobalState(ClientContext &context, const PhysicalPiecewiseMergeJoin &op) {
		RowLayout rhs_layout;
		rhs_layout.Initialize(op.children[1]->types);
		vector<BoundOrderByNode> rhs_order;
		rhs_order.emplace_back(op.rhs_orders[0].Copy());
		table = make_uniq<GlobalSortedTable>(context, rhs_order, rhs_layout);
	}

	inline idx_t Count() const {
		return table->count;
	}

	void Sink(DataChunk &input, MergeJoinLocalState &lstate) {
		auto &global_sort_state = table->global_sort_state;
		auto &local_sort_state = lstate.table.local_sort_state;

		// Sink the data into the local sort state
		lstate.table.Sink(input, global_sort_state);

		// When sorting data reaches a certain size, we sort it
		if (local_sort_state.SizeInBytes() >= table->memory_per_thread) {
			local_sort_state.Sort(global_sort_state, true);
		}
	}

	unique_ptr<GlobalSortedTable> table;
};

unique_ptr<GlobalSinkState> PhysicalPiecewiseMergeJoin::GetGlobalSinkState(ClientContext &context) const {
	return make_uniq<MergeJoinGlobalState>(context, *this);
}

unique_ptr<LocalSinkState> PhysicalPiecewiseMergeJoin::GetLocalSinkState(ExecutionContext &context) const {
	// We only sink the RHS
	return make_uniq<MergeJoinLocalState>(context.client, *this, 1);
}

SinkResultType PhysicalPiecewiseMergeJoin::Sink(ExecutionContext &context, DataChunk &chunk,
                                                OperatorSinkInput &input) const {
	auto &gstate = input.global_state.Cast<MergeJoinGlobalState>();
	auto &lstate = input.local_state.Cast<MergeJoinLocalState>();

	gstate.Sink(chunk, lstate);

	return SinkResultType::NEED_MORE_INPUT;
}

void PhysicalPiecewiseMergeJoin::Combine(ExecutionContext &context, GlobalSinkState &gstate_p,
                                         LocalSinkState &lstate_p) const {
	auto &gstate = gstate_p.Cast<MergeJoinGlobalState>();
	auto &lstate = lstate_p.Cast<MergeJoinLocalState>();
	gstate.table->Combine(lstate.table);
	auto &client_profiler = QueryProfiler::Get(context.client);

	context.thread.profiler.Flush(*this, lstate.table.executor, "rhs_executor", 1);
	client_profiler.Flush(context.thread.profiler);
}

//===--------------------------------------------------------------------===//
// Finalize
//===--------------------------------------------------------------------===//
SinkFinalizeType PhysicalPiecewiseMergeJoin::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                                      GlobalSinkState &gstate_p) const {
	auto &gstate = gstate_p.Cast<MergeJoinGlobalState>();
	auto &global_sort_state = gstate.table->global_sort_state;

	if (IsRightOuterJoin(join_type)) {
		// for FULL/RIGHT OUTER JOIN, initialize found_match to false for every tuple
		gstate.table->IntializeMatches();
	}
	if (global_sort_state.sorted_blocks.empty() && EmptyResultIfRHSIsEmpty()) {
		// Empty input!
		return SinkFinalizeType::NO_OUTPUT_POSSIBLE;
	}

	// Sort the current input child
	gstate.table->Finalize(pipeline, event);

	return SinkFinalizeType::READY;
}

//===--------------------------------------------------------------------===//
// Operator
//===--------------------------------------------------------------------===//
class PiecewiseMergeJoinState : public CachingOperatorState {
public:
	using LocalSortedTable = PhysicalRangeJoin::LocalSortedTable;

	PiecewiseMergeJoinState(ClientContext &context, const PhysicalPiecewiseMergeJoin &op, bool force_external)
	    : context(context), allocator(Allocator::Get(context)), op(op),
	      buffer_manager(BufferManager::GetBufferManager(context)), force_external(force_external),
	      left_outer(IsLeftOuterJoin(op.join_type)), left_position(0), first_fetch(true), finished(true),
	      right_position(0), right_chunk_index(0), rhs_executor(context) {
		vector<LogicalType> condition_types;
		for (auto &order : op.lhs_orders) {
			condition_types.push_back(order.expression->return_type);
		}
		left_outer.Initialize(STANDARD_VECTOR_SIZE);
		lhs_layout.Initialize(op.children[0]->types);
		lhs_payload.Initialize(allocator, op.children[0]->types);

		lhs_order.emplace_back(op.lhs_orders[0].Copy());

		// Set up shared data for multiple predicates
		sel.Initialize(STANDARD_VECTOR_SIZE);
		condition_types.clear();
		for (auto &order : op.rhs_orders) {
			rhs_executor.AddExpression(*order.expression);
			condition_types.push_back(order.expression->return_type);
		}
		rhs_keys.Initialize(allocator, condition_types);
	}

	ClientContext &context;
	Allocator &allocator;
	const PhysicalPiecewiseMergeJoin &op;
	BufferManager &buffer_manager;
	bool force_external;

	// Block sorting
	DataChunk lhs_payload;
	OuterJoinMarker left_outer;
	vector<BoundOrderByNode> lhs_order;
	RowLayout lhs_layout;
	unique_ptr<LocalSortedTable> lhs_local_table;
	unique_ptr<GlobalSortState> lhs_global_state;
	unique_ptr<PayloadScanner> scanner;

	// Simple scans
	idx_t left_position;

	// Complex scans
	bool first_fetch;
	bool finished;
	idx_t right_position;
	idx_t right_chunk_index;
	idx_t right_base;

	// Secondary predicate shared data
	SelectionVector sel;
	DataChunk rhs_keys;
	DataChunk rhs_input;
	ExpressionExecutor rhs_executor;
	vector<BufferHandle> payload_heap_handles;

public:
	void ResolveJoinKeys(DataChunk &input) {
		// sort by join key
		lhs_global_state = make_uniq<GlobalSortState>(buffer_manager, lhs_order, lhs_layout);
		lhs_local_table = make_uniq<LocalSortedTable>(context, op, 0);
		lhs_local_table->Sink(input, *lhs_global_state);

		// Set external (can be forced with the PRAGMA)
		lhs_global_state->external = force_external;
		lhs_global_state->AddLocalState(lhs_local_table->local_sort_state);
		lhs_global_state->PrepareMergePhase();
		while (lhs_global_state->sorted_blocks.size() > 1) {
			MergeSorter merge_sorter(*lhs_global_state, buffer_manager);
			merge_sorter.PerformInMergeRound();
			lhs_global_state->CompleteMergeRound();
		}

		// Scan the sorted payload
		D_ASSERT(lhs_global_state->sorted_blocks.size() == 1);

		scanner = make_uniq<PayloadScanner>(*lhs_global_state->sorted_blocks[0]->payload_data, *lhs_global_state);
		lhs_payload.Reset();
		scanner->Scan(lhs_payload);

		// Recompute the sorted keys from the sorted input
		lhs_local_table->keys.Reset();
		lhs_local_table->executor.Execute(lhs_payload, lhs_local_table->keys);
	}

	void Finalize(const PhysicalOperator &op, ExecutionContext &context) override {
		if (lhs_local_table) {
			context.thread.profiler.Flush(op, lhs_local_table->executor, "lhs_executor", 0);
		}
	}
};

unique_ptr<OperatorState> PhysicalPiecewiseMergeJoin::GetOperatorState(ExecutionContext &context) const {
	auto &config = ClientConfig::GetConfig(context.client);
	return make_uniq<PiecewiseMergeJoinState>(context.client, *this, config.force_external);
}

static inline idx_t SortedBlockNotNull(const idx_t base, const idx_t count, const idx_t not_null) {
	return MinValue(base + count, MaxValue(base, not_null)) - base;
}

static int MergeJoinComparisonValue(ExpressionType comparison) {
	switch (comparison) {
	case ExpressionType::COMPARE_LESSTHAN:
	case ExpressionType::COMPARE_GREATERTHAN:
		return -1;
	case ExpressionType::COMPARE_LESSTHANOREQUALTO:
	case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
		return 0;
	default:
		throw InternalException("Unimplemented comparison type for merge join!");
	}
}

struct BlockMergeInfo {
	GlobalSortState &state;
	//! The block being scanned
	const idx_t block_idx;
	//! The number of not-NULL values in the block (they are at the end)
	const idx_t not_null;
	//! The current offset in the block
	idx_t &entry_idx;
	SelectionVector result;

	BlockMergeInfo(GlobalSortState &state, idx_t block_idx, idx_t &entry_idx, idx_t not_null)
	    : state(state), block_idx(block_idx), not_null(not_null), entry_idx(entry_idx), result(STANDARD_VECTOR_SIZE) {
	}
};

static void MergeJoinPinSortingBlock(SBScanState &scan, const idx_t block_idx) {
	scan.SetIndices(block_idx, 0);
	scan.PinRadix(block_idx);

	auto &sd = *scan.sb->blob_sorting_data;
	if (block_idx < sd.data_blocks.size()) {
		scan.PinData(sd);
	}
}

static data_ptr_t MergeJoinRadixPtr(SBScanState &scan, const idx_t entry_idx) {
	scan.entry_idx = entry_idx;
	return scan.RadixPtr();
}

static idx_t MergeJoinSimpleBlocks(PiecewiseMergeJoinState &lstate, MergeJoinGlobalState &rstate, bool *found_match,
                                   const ExpressionType comparison) {
	const auto cmp = MergeJoinComparisonValue(comparison);

	// The sort parameters should all be the same
	auto &lsort = *lstate.lhs_global_state;
	auto &rsort = rstate.table->global_sort_state;
	D_ASSERT(lsort.sort_layout.all_constant == rsort.sort_layout.all_constant);
	const auto all_constant = lsort.sort_layout.all_constant;
	D_ASSERT(lsort.external == rsort.external);
	const auto external = lsort.external;

	// There should only be one sorted block if they have been sorted
	D_ASSERT(lsort.sorted_blocks.size() == 1);
	SBScanState lread(lsort.buffer_manager, lsort);
	lread.sb = lsort.sorted_blocks[0].get();

	const idx_t l_block_idx = 0;
	idx_t l_entry_idx = 0;
	const auto lhs_not_null = lstate.lhs_local_table->count - lstate.lhs_local_table->has_null;
	MergeJoinPinSortingBlock(lread, l_block_idx);
	auto l_ptr = MergeJoinRadixPtr(lread, l_entry_idx);

	D_ASSERT(rsort.sorted_blocks.size() == 1);
	SBScanState rread(rsort.buffer_manager, rsort);
	rread.sb = rsort.sorted_blocks[0].get();

	const auto cmp_size = lsort.sort_layout.comparison_size;
	const auto entry_size = lsort.sort_layout.entry_size;

	idx_t right_base = 0;
	for (idx_t r_block_idx = 0; r_block_idx < rread.sb->radix_sorting_data.size(); r_block_idx++) {
		// we only care about the BIGGEST value in each of the RHS data blocks
		// because we want to figure out if the LHS values are less than [or equal] to ANY value
		// get the biggest value from the RHS chunk
		MergeJoinPinSortingBlock(rread, r_block_idx);

		auto &rblock = *rread.sb->radix_sorting_data[r_block_idx];
		const auto r_not_null =
		    SortedBlockNotNull(right_base, rblock.count, rstate.table->count - rstate.table->has_null);
		if (r_not_null == 0) {
			break;
		}
		const auto r_entry_idx = r_not_null - 1;
		right_base += rblock.count;

		auto r_ptr = MergeJoinRadixPtr(rread, r_entry_idx);

		// now we start from the current lpos value and check if we found a new value that is [<= OR <] the max RHS
		// value
		while (true) {
			int comp_res;
			if (all_constant) {
				comp_res = FastMemcmp(l_ptr, r_ptr, cmp_size);
			} else {
				lread.entry_idx = l_entry_idx;
				rread.entry_idx = r_entry_idx;
				comp_res = Comparators::CompareTuple(lread, rread, l_ptr, r_ptr, lsort.sort_layout, external);
			}

			if (comp_res <= cmp) {
				// found a match for lpos, set it in the found_match vector
				found_match[l_entry_idx] = true;
				l_entry_idx++;
				l_ptr += entry_size;
				if (l_entry_idx >= lhs_not_null) {
					// early out: we exhausted the entire LHS and they all match
					return 0;
				}
			} else {
				// we found no match: any subsequent value from the LHS we scan now will be bigger and thus also not
				// match move to the next RHS chunk
				break;
			}
		}
	}
	return 0;
}

void PhysicalPiecewiseMergeJoin::ResolveSimpleJoin(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                                   OperatorState &state_p) const {
	auto &state = (PiecewiseMergeJoinState &)state_p;
	auto &gstate = sink_state->Cast<MergeJoinGlobalState>();

	state.ResolveJoinKeys(input);
	auto &lhs_table = *state.lhs_local_table;

	// perform the actual join
	bool found_match[STANDARD_VECTOR_SIZE];
	memset(found_match, 0, sizeof(found_match));
	MergeJoinSimpleBlocks(state, gstate, found_match, conditions[0].comparison);

	// use the sorted payload
	const auto lhs_not_null = lhs_table.count - lhs_table.has_null;
	auto &payload = state.lhs_payload;

	// now construct the result based on the join result
	switch (join_type) {
	case JoinType::MARK: {
		// The only part of the join keys that is actually used is the validity mask.
		// Since the payload is sorted, we can just set the tail end of the validity masks to invalid.
		for (auto &key : lhs_table.keys.data) {
			key.Flatten(lhs_table.keys.size());
			auto &mask = FlatVector::Validity(key);
			if (mask.AllValid()) {
				continue;
			}
			mask.SetAllValid(lhs_not_null);
			for (idx_t i = lhs_not_null; i < lhs_table.count; ++i) {
				mask.SetInvalid(i);
			}
		}
		// So we make a set of keys that have the validity mask set for the
		PhysicalJoin::ConstructMarkJoinResult(lhs_table.keys, payload, chunk, found_match, gstate.table->has_null);
		break;
	}
	case JoinType::SEMI:
		PhysicalJoin::ConstructSemiJoinResult(payload, chunk, found_match);
		break;
	case JoinType::ANTI:
		PhysicalJoin::ConstructAntiJoinResult(payload, chunk, found_match);
		break;
	default:
		throw NotImplementedException("Unimplemented join type for merge join");
	}
}

static idx_t MergeJoinComplexBlocks(BlockMergeInfo &l, BlockMergeInfo &r, const ExpressionType comparison) {
	const auto cmp = MergeJoinComparisonValue(comparison);

	// The sort parameters should all be the same
	D_ASSERT(l.state.sort_layout.all_constant == r.state.sort_layout.all_constant);
	const auto all_constant = r.state.sort_layout.all_constant;
	D_ASSERT(l.state.external == r.state.external);
	const auto external = l.state.external;

	// There should only be one sorted block if they have been sorted
	D_ASSERT(l.state.sorted_blocks.size() == 1);
	SBScanState lread(l.state.buffer_manager, l.state);
	lread.sb = l.state.sorted_blocks[0].get();
	D_ASSERT(lread.sb->radix_sorting_data.size() == 1);
	MergeJoinPinSortingBlock(lread, l.block_idx);
	auto l_start = MergeJoinRadixPtr(lread, 0);
	auto l_ptr = MergeJoinRadixPtr(lread, l.entry_idx);

	D_ASSERT(r.state.sorted_blocks.size() == 1);
	SBScanState rread(r.state.buffer_manager, r.state);
	rread.sb = r.state.sorted_blocks[0].get();

	if (r.entry_idx >= r.not_null) {
		return 0;
	}

	MergeJoinPinSortingBlock(rread, r.block_idx);
	auto r_ptr = MergeJoinRadixPtr(rread, r.entry_idx);

	const auto cmp_size = l.state.sort_layout.comparison_size;
	const auto entry_size = l.state.sort_layout.entry_size;

	idx_t result_count = 0;
	while (true) {
		if (l.entry_idx < l.not_null) {
			int comp_res;
			if (all_constant) {
				comp_res = FastMemcmp(l_ptr, r_ptr, cmp_size);
			} else {
				lread.entry_idx = l.entry_idx;
				rread.entry_idx = r.entry_idx;
				comp_res = Comparators::CompareTuple(lread, rread, l_ptr, r_ptr, l.state.sort_layout, external);
			}

			if (comp_res <= cmp) {
				// left side smaller: found match
				l.result.set_index(result_count, sel_t(l.entry_idx));
				r.result.set_index(result_count, sel_t(r.entry_idx));
				result_count++;
				// move left side forward
				l.entry_idx++;
				l_ptr += entry_size;
				if (result_count == STANDARD_VECTOR_SIZE) {
					// out of space!
					break;
				}
				continue;
			}
		}
		// right side smaller or equal, or left side exhausted: move
		// right pointer forward reset left side to start
		r.entry_idx++;
		if (r.entry_idx >= r.not_null) {
			break;
		}
		r_ptr += entry_size;

		l_ptr = l_start;
		l.entry_idx = 0;
	}

	return result_count;
}

OperatorResultType PhysicalPiecewiseMergeJoin::ResolveComplexJoin(ExecutionContext &context, DataChunk &input,
                                                                  DataChunk &chunk, OperatorState &state_p) const {
	auto &state = (PiecewiseMergeJoinState &)state_p;
	auto &gstate = sink_state->Cast<MergeJoinGlobalState>();
	auto &rsorted = *gstate.table->global_sort_state.sorted_blocks[0];
	const auto left_cols = input.ColumnCount();
	const auto tail_cols = conditions.size() - 1;

	state.payload_heap_handles.clear();
	do {
		if (state.first_fetch) {
			state.ResolveJoinKeys(input);

			state.right_chunk_index = 0;
			state.right_base = 0;
			state.left_position = 0;
			state.right_position = 0;
			state.first_fetch = false;
			state.finished = false;
		}
		if (state.finished) {
			if (state.left_outer.Enabled()) {
				// left join: before we move to the next chunk, see if we need to output any vectors that didn't
				// have a match found
				state.left_outer.ConstructLeftJoinResult(state.lhs_payload, chunk);
				state.left_outer.Reset();
			}
			state.first_fetch = true;
			state.finished = false;
			return OperatorResultType::NEED_MORE_INPUT;
		}

		auto &lhs_table = *state.lhs_local_table;
		const auto lhs_not_null = lhs_table.count - lhs_table.has_null;
		BlockMergeInfo left_info(*state.lhs_global_state, 0, state.left_position, lhs_not_null);

		const auto &rblock = *rsorted.radix_sorting_data[state.right_chunk_index];
		const auto rhs_not_null =
		    SortedBlockNotNull(state.right_base, rblock.count, gstate.table->count - gstate.table->has_null);
		BlockMergeInfo right_info(gstate.table->global_sort_state, state.right_chunk_index, state.right_position,
		                          rhs_not_null);

		idx_t result_count = MergeJoinComplexBlocks(left_info, right_info, conditions[0].comparison);
		if (result_count == 0) {
			// exhausted this chunk on the right side
			// move to the next right chunk
			state.left_position = 0;
			state.right_position = 0;
			state.right_base += rsorted.radix_sorting_data[state.right_chunk_index]->count;
			state.right_chunk_index++;
			if (state.right_chunk_index >= rsorted.radix_sorting_data.size()) {
				state.finished = true;
			}
		} else {
			// found matches: extract them
			chunk.Reset();
			for (idx_t c = 0; c < state.lhs_payload.ColumnCount(); ++c) {
				chunk.data[c].Slice(state.lhs_payload.data[c], left_info.result, result_count);
			}
			state.payload_heap_handles.push_back(SliceSortedPayload(chunk, right_info.state, right_info.block_idx,
			                                                        right_info.result, result_count, left_cols));
			chunk.SetCardinality(result_count);

			auto sel = FlatVector::IncrementalSelectionVector();
			if (tail_cols) {
				// If there are more expressions to compute,
				// split the result chunk into the left and right halves
				// so we can compute the values for comparison.
				chunk.Split(state.rhs_input, left_cols);
				state.rhs_executor.SetChunk(state.rhs_input);
				state.rhs_keys.Reset();

				auto tail_count = result_count;
				for (size_t cmp_idx = 1; cmp_idx < conditions.size(); ++cmp_idx) {
					Vector left(lhs_table.keys.data[cmp_idx]);
					left.Slice(left_info.result, result_count);

					auto &right = state.rhs_keys.data[cmp_idx];
					state.rhs_executor.ExecuteExpression(cmp_idx, right);

					if (tail_count < result_count) {
						left.Slice(*sel, tail_count);
						right.Slice(*sel, tail_count);
					}
					tail_count =
					    SelectJoinTail(conditions[cmp_idx].comparison, left, right, sel, tail_count, &state.sel);
					sel = &state.sel;
				}
				chunk.Fuse(state.rhs_input);

				if (tail_count < result_count) {
					result_count = tail_count;
					chunk.Slice(*sel, result_count);
				}
			}

			// found matches: mark the found matches if required
			if (state.left_outer.Enabled()) {
				for (idx_t i = 0; i < result_count; i++) {
					state.left_outer.SetMatch(left_info.result[sel->get_index(i)]);
				}
			}
			if (gstate.table->found_match) {
				//	Absolute position of the block + start position inside that block
				for (idx_t i = 0; i < result_count; i++) {
					gstate.table->found_match[state.right_base + right_info.result[sel->get_index(i)]] = true;
				}
			}
			chunk.SetCardinality(result_count);
			chunk.Verify();
		}
	} while (chunk.size() == 0);
	return OperatorResultType::HAVE_MORE_OUTPUT;
}

OperatorResultType PhysicalPiecewiseMergeJoin::ExecuteInternal(ExecutionContext &context, DataChunk &input,
                                                               DataChunk &chunk, GlobalOperatorState &gstate_p,
                                                               OperatorState &state) const {
	auto &gstate = sink_state->Cast<MergeJoinGlobalState>();

	if (gstate.Count() == 0) {
		// empty RHS
		if (!EmptyResultIfRHSIsEmpty()) {
			ConstructEmptyJoinResult(join_type, gstate.table->has_null, input, chunk);
			return OperatorResultType::NEED_MORE_INPUT;
		} else {
			return OperatorResultType::FINISHED;
		}
	}

	input.Verify();
	switch (join_type) {
	case JoinType::SEMI:
	case JoinType::ANTI:
	case JoinType::MARK:
		// simple joins can have max STANDARD_VECTOR_SIZE matches per chunk
		ResolveSimpleJoin(context, input, chunk, state);
		return OperatorResultType::NEED_MORE_INPUT;
	case JoinType::LEFT:
	case JoinType::INNER:
	case JoinType::RIGHT:
	case JoinType::OUTER:
		return ResolveComplexJoin(context, input, chunk, state);
	default:
		throw NotImplementedException("Unimplemented type for piecewise merge loop join!");
	}
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
class PiecewiseJoinScanState : public GlobalSourceState {
public:
	explicit PiecewiseJoinScanState(const PhysicalPiecewiseMergeJoin &op) : op(op), right_outer_position(0) {
	}

	mutex lock;
	const PhysicalPiecewiseMergeJoin &op;
	unique_ptr<PayloadScanner> scanner;
	idx_t right_outer_position;

public:
	idx_t MaxThreads() override {
		auto &sink = op.sink_state->Cast<MergeJoinGlobalState>();
		return sink.Count() / (STANDARD_VECTOR_SIZE * idx_t(10));
	}
};

unique_ptr<GlobalSourceState> PhysicalPiecewiseMergeJoin::GetGlobalSourceState(ClientContext &context) const {
	return make_uniq<PiecewiseJoinScanState>(*this);
}

SourceResultType PhysicalPiecewiseMergeJoin::GetData(ExecutionContext &context, DataChunk &result,
                                                     OperatorSourceInput &input) const {
	D_ASSERT(IsRightOuterJoin(join_type));
	// check if we need to scan any unmatched tuples from the RHS for the full/right outer join
	auto &sink = sink_state->Cast<MergeJoinGlobalState>();
	auto &state = (PiecewiseJoinScanState &)input.global_state;

	lock_guard<mutex> l(state.lock);
	if (!state.scanner) {
		// Initialize scanner (if not yet initialized)
		auto &sort_state = sink.table->global_sort_state;
		if (sort_state.sorted_blocks.empty()) {
			return SourceResultType::FINISHED;
		}
		state.scanner = make_uniq<PayloadScanner>(*sort_state.sorted_blocks[0]->payload_data, sort_state);
	}

	// if the LHS is exhausted in a FULL/RIGHT OUTER JOIN, we scan the found_match for any chunks we
	// still need to output
	const auto found_match = sink.table->found_match.get();

	DataChunk rhs_chunk;
	rhs_chunk.Initialize(Allocator::Get(context.client), sink.table->global_sort_state.payload_layout.GetTypes());
	SelectionVector rsel(STANDARD_VECTOR_SIZE);
	for (;;) {
		// Read the next sorted chunk
		state.scanner->Scan(rhs_chunk);

		const auto count = rhs_chunk.size();
		if (count == 0) {
			return result.size() == 0 ? SourceResultType::FINISHED : SourceResultType::HAVE_MORE_OUTPUT;
		}

		idx_t result_count = 0;
		// figure out which tuples didn't find a match in the RHS
		for (idx_t i = 0; i < count; i++) {
			if (!found_match[state.right_outer_position + i]) {
				rsel.set_index(result_count++, i);
			}
		}
		state.right_outer_position += count;

		if (result_count > 0) {
			// if there were any tuples that didn't find a match, output them
			const idx_t left_column_count = children[0]->types.size();
			for (idx_t col_idx = 0; col_idx < left_column_count; ++col_idx) {
				result.data[col_idx].SetVectorType(VectorType::CONSTANT_VECTOR);
				ConstantVector::SetNull(result.data[col_idx], true);
			}
			const idx_t right_column_count = children[1]->types.size();
			;
			for (idx_t col_idx = 0; col_idx < right_column_count; ++col_idx) {
				result.data[left_column_count + col_idx].Slice(rhs_chunk.data[col_idx], rsel, result_count);
			}
			result.SetCardinality(result_count);
			break;
		}
	}

	return result.size() == 0 ? SourceResultType::FINISHED : SourceResultType::HAVE_MORE_OUTPUT;
}

} // namespace duckdb





namespace duckdb {

PhysicalPositionalJoin::PhysicalPositionalJoin(vector<LogicalType> types, unique_ptr<PhysicalOperator> left,
                                               unique_ptr<PhysicalOperator> right, idx_t estimated_cardinality)
    : PhysicalOperator(PhysicalOperatorType::POSITIONAL_JOIN, std::move(types), estimated_cardinality) {
	children.push_back(std::move(left));
	children.push_back(std::move(right));
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
class PositionalJoinGlobalState : public GlobalSinkState {
public:
	explicit PositionalJoinGlobalState(ClientContext &context, const PhysicalPositionalJoin &op)
	    : rhs(context, op.children[1]->GetTypes()), initialized(false), source_offset(0), exhausted(false) {
		rhs.InitializeAppend(append_state);
	}

	ColumnDataCollection rhs;
	ColumnDataAppendState append_state;
	mutex rhs_lock;

	bool initialized;
	ColumnDataScanState scan_state;
	DataChunk source;
	idx_t source_offset;
	bool exhausted;

	void InitializeScan();
	idx_t Refill();
	idx_t CopyData(DataChunk &output, const idx_t count, const idx_t col_offset);
	void Execute(DataChunk &input, DataChunk &output);
	void GetData(DataChunk &output);
};

unique_ptr<GlobalSinkState> PhysicalPositionalJoin::GetGlobalSinkState(ClientContext &context) const {
	return make_uniq<PositionalJoinGlobalState>(context, *this);
}

SinkResultType PhysicalPositionalJoin::Sink(ExecutionContext &context, DataChunk &chunk,
                                            OperatorSinkInput &input) const {
	auto &sink = input.global_state.Cast<PositionalJoinGlobalState>();
	lock_guard<mutex> client_guard(sink.rhs_lock);
	sink.rhs.Append(sink.append_state, chunk);
	return SinkResultType::NEED_MORE_INPUT;
}

//===--------------------------------------------------------------------===//
// Operator
//===--------------------------------------------------------------------===//
void PositionalJoinGlobalState::InitializeScan() {
	if (!initialized) {
		// not initialized yet: initialize the scan
		initialized = true;
		rhs.InitializeScanChunk(source);
		rhs.InitializeScan(scan_state);
	}
}

idx_t PositionalJoinGlobalState::Refill() {
	if (source_offset >= source.size()) {
		if (!exhausted) {
			source.Reset();
			rhs.Scan(scan_state, source);
		}
		source_offset = 0;
	}

	const auto available = source.size() - source_offset;
	if (!available) {
		if (!exhausted) {
			source.Reset();
			for (idx_t i = 0; i < source.ColumnCount(); ++i) {
				auto &vec = source.data[i];
				vec.SetVectorType(VectorType::CONSTANT_VECTOR);
				ConstantVector::SetNull(vec, true);
			}
			exhausted = true;
		}
	}

	return available;
}

idx_t PositionalJoinGlobalState::CopyData(DataChunk &output, const idx_t count, const idx_t col_offset) {
	if (!source_offset && (source.size() >= count || exhausted)) {
		//	Fast track: aligned and has enough data
		for (idx_t i = 0; i < source.ColumnCount(); ++i) {
			output.data[col_offset + i].Reference(source.data[i]);
		}
		source_offset += count;
	} else {
		// Copy data
		for (idx_t target_offset = 0; target_offset < count;) {
			const auto needed = count - target_offset;
			const auto available = exhausted ? needed : (source.size() - source_offset);
			const auto copy_size = MinValue(needed, available);
			const auto source_count = source_offset + copy_size;
			for (idx_t i = 0; i < source.ColumnCount(); ++i) {
				VectorOperations::Copy(source.data[i], output.data[col_offset + i], source_count, source_offset,
				                       target_offset);
			}
			target_offset += copy_size;
			source_offset += copy_size;
			Refill();
		}
	}

	return source.ColumnCount();
}

void PositionalJoinGlobalState::Execute(DataChunk &input, DataChunk &output) {
	lock_guard<mutex> client_guard(rhs_lock);

	// Reference the input and assume it will be full
	const auto col_offset = input.ColumnCount();
	for (idx_t i = 0; i < col_offset; ++i) {
		output.data[i].Reference(input.data[i]);
	}

	// Copy or reference the RHS columns
	const auto count = input.size();
	InitializeScan();
	Refill();
	CopyData(output, count, col_offset);

	output.SetCardinality(count);
}

OperatorResultType PhysicalPositionalJoin::Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                                   GlobalOperatorState &gstate, OperatorState &state_p) const {
	auto &sink = sink_state->Cast<PositionalJoinGlobalState>();
	sink.Execute(input, chunk);
	return OperatorResultType::NEED_MORE_INPUT;
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
void PositionalJoinGlobalState::GetData(DataChunk &output) {
	lock_guard<mutex> client_guard(rhs_lock);

	InitializeScan();
	Refill();

	//	LHS exhausted
	if (exhausted) {
		//	RHS exhausted too, so we are done
		output.SetCardinality(0);
		return;
	}

	//	LHS is all NULL
	const auto col_offset = output.ColumnCount() - source.ColumnCount();
	for (idx_t i = 0; i < col_offset; ++i) {
		auto &vec = output.data[i];
		vec.SetVectorType(VectorType::CONSTANT_VECTOR);
		ConstantVector::SetNull(vec, true);
	}

	//	RHS still has data, so copy it
	const auto count = MinValue<idx_t>(STANDARD_VECTOR_SIZE, source.size() - source_offset);
	CopyData(output, count, col_offset);
	output.SetCardinality(count);
}

SourceResultType PhysicalPositionalJoin::GetData(ExecutionContext &context, DataChunk &result,
                                                 OperatorSourceInput &input) const {
	auto &sink = sink_state->Cast<PositionalJoinGlobalState>();
	sink.GetData(result);

	return result.size() == 0 ? SourceResultType::FINISHED : SourceResultType::HAVE_MORE_OUTPUT;
}

//===--------------------------------------------------------------------===//
// Pipeline Construction
//===--------------------------------------------------------------------===//
void PhysicalPositionalJoin::BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) {
	PhysicalJoin::BuildJoinPipelines(current, meta_pipeline, *this);
}

vector<const_reference<PhysicalOperator>> PhysicalPositionalJoin::GetSources() const {
	auto result = children[0]->GetSources();
	if (IsSource()) {
		result.push_back(*this);
	}
	return result;
}

} // namespace duckdb













#include <thread>

namespace duckdb {

PhysicalRangeJoin::LocalSortedTable::LocalSortedTable(ClientContext &context, const PhysicalRangeJoin &op,
                                                      const idx_t child)
    : op(op), executor(context), has_null(0), count(0) {
	// Initialize order clause expression executor and key DataChunk
	vector<LogicalType> types;
	for (const auto &cond : op.conditions) {
		const auto &expr = child ? cond.right : cond.left;
		executor.AddExpression(*expr);

		types.push_back(expr->return_type);
	}
	auto &allocator = Allocator::Get(context);
	keys.Initialize(allocator, types);
}

void PhysicalRangeJoin::LocalSortedTable::Sink(DataChunk &input, GlobalSortState &global_sort_state) {
	// Initialize local state (if necessary)
	if (!local_sort_state.initialized) {
		local_sort_state.Initialize(global_sort_state, global_sort_state.buffer_manager);
	}

	// Obtain sorting columns
	keys.Reset();
	executor.Execute(input, keys);

	// Count the NULLs so we can exclude them later
	has_null += MergeNulls(op.conditions);
	count += keys.size();

	//	Only sort the primary key
	DataChunk join_head;
	join_head.data.emplace_back(keys.data[0]);
	join_head.SetCardinality(keys.size());

	// Sink the data into the local sort state
	local_sort_state.SinkChunk(join_head, input);
}

PhysicalRangeJoin::GlobalSortedTable::GlobalSortedTable(ClientContext &context, const vector<BoundOrderByNode> &orders,
                                                        RowLayout &payload_layout)
    : global_sort_state(BufferManager::GetBufferManager(context), orders, payload_layout), has_null(0), count(0),
      memory_per_thread(0) {
	D_ASSERT(orders.size() == 1);

	// Set external (can be forced with the PRAGMA)
	auto &config = ClientConfig::GetConfig(context);
	global_sort_state.external = config.force_external;
	memory_per_thread = PhysicalRangeJoin::GetMaxThreadMemory(context);
}

void PhysicalRangeJoin::GlobalSortedTable::Combine(LocalSortedTable &ltable) {
	global_sort_state.AddLocalState(ltable.local_sort_state);
	has_null += ltable.has_null;
	count += ltable.count;
}

void PhysicalRangeJoin::GlobalSortedTable::IntializeMatches() {
	found_match = make_unsafe_uniq_array<bool>(Count());
	memset(found_match.get(), 0, sizeof(bool) * Count());
}

void PhysicalRangeJoin::GlobalSortedTable::Print() {
	global_sort_state.Print();
}

class RangeJoinMergeTask : public ExecutorTask {
public:
	using GlobalSortedTable = PhysicalRangeJoin::GlobalSortedTable;

public:
	RangeJoinMergeTask(shared_ptr<Event> event_p, ClientContext &context, GlobalSortedTable &table)
	    : ExecutorTask(context), event(std::move(event_p)), context(context), table(table) {
	}

	TaskExecutionResult ExecuteTask(TaskExecutionMode mode) override {
		// Initialize iejoin sorted and iterate until done
		auto &global_sort_state = table.global_sort_state;
		MergeSorter merge_sorter(global_sort_state, BufferManager::GetBufferManager(context));
		merge_sorter.PerformInMergeRound();
		event->FinishTask();

		return TaskExecutionResult::TASK_FINISHED;
	}

private:
	shared_ptr<Event> event;
	ClientContext &context;
	GlobalSortedTable &table;
};

class RangeJoinMergeEvent : public BasePipelineEvent {
public:
	using GlobalSortedTable = PhysicalRangeJoin::GlobalSortedTable;

public:
	RangeJoinMergeEvent(GlobalSortedTable &table_p, Pipeline &pipeline_p)
	    : BasePipelineEvent(pipeline_p), table(table_p) {
	}

	GlobalSortedTable &table;

public:
	void Schedule() override {
		auto &context = pipeline->GetClientContext();

		// Schedule tasks equal to the number of threads, which will each merge multiple partitions
		auto &ts = TaskScheduler::GetScheduler(context);
		idx_t num_threads = ts.NumberOfThreads();

		vector<shared_ptr<Task>> iejoin_tasks;
		for (idx_t tnum = 0; tnum < num_threads; tnum++) {
			iejoin_tasks.push_back(make_uniq<RangeJoinMergeTask>(shared_from_this(), context, table));
		}
		SetTasks(std::move(iejoin_tasks));
	}

	void FinishEvent() override {
		auto &global_sort_state = table.global_sort_state;

		global_sort_state.CompleteMergeRound(true);
		if (global_sort_state.sorted_blocks.size() > 1) {
			// Multiple blocks remaining: Schedule the next round
			table.ScheduleMergeTasks(*pipeline, *this);
		}
	}
};

void PhysicalRangeJoin::GlobalSortedTable::ScheduleMergeTasks(Pipeline &pipeline, Event &event) {
	// Initialize global sort state for a round of merging
	global_sort_state.InitializeMergeRound();
	auto new_event = make_shared<RangeJoinMergeEvent>(*this, pipeline);
	event.InsertEvent(std::move(new_event));
}

void PhysicalRangeJoin::GlobalSortedTable::Finalize(Pipeline &pipeline, Event &event) {
	// Prepare for merge sort phase
	global_sort_state.PrepareMergePhase();

	// Start the merge phase or finish if a merge is not necessary
	if (global_sort_state.sorted_blocks.size() > 1) {
		ScheduleMergeTasks(pipeline, event);
	}
}

PhysicalRangeJoin::PhysicalRangeJoin(LogicalOperator &op, PhysicalOperatorType type, unique_ptr<PhysicalOperator> left,
                                     unique_ptr<PhysicalOperator> right, vector<JoinCondition> cond, JoinType join_type,
                                     idx_t estimated_cardinality)
    : PhysicalComparisonJoin(op, type, std::move(cond), join_type, estimated_cardinality) {
	// Reorder the conditions so that ranges are at the front.
	// TODO: use stats to improve the choice?
	// TODO: Prefer fixed length types?
	if (conditions.size() > 1) {
		auto conditions_p = std::move(conditions);
		conditions.resize(conditions_p.size());
		idx_t range_position = 0;
		idx_t other_position = conditions_p.size();
		for (idx_t i = 0; i < conditions_p.size(); ++i) {
			switch (conditions_p[i].comparison) {
			case ExpressionType::COMPARE_LESSTHAN:
			case ExpressionType::COMPARE_LESSTHANOREQUALTO:
			case ExpressionType::COMPARE_GREATERTHAN:
			case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
				conditions[range_position++] = std::move(conditions_p[i]);
				break;
			default:
				conditions[--other_position] = std::move(conditions_p[i]);
				break;
			}
		}
	}

	children.push_back(std::move(left));
	children.push_back(std::move(right));
}

idx_t PhysicalRangeJoin::LocalSortedTable::MergeNulls(const vector<JoinCondition> &conditions) {
	// Merge the validity masks of the comparison keys into the primary
	// Return the number of NULLs in the resulting chunk
	D_ASSERT(keys.ColumnCount() > 0);
	const auto count = keys.size();

	size_t all_constant = 0;
	for (auto &v : keys.data) {
		if (v.GetVectorType() == VectorType::CONSTANT_VECTOR) {
			++all_constant;
		}
	}

	auto &primary = keys.data[0];
	if (all_constant == keys.data.size()) {
		//	Either all NULL or no NULLs
		for (auto &v : keys.data) {
			if (ConstantVector::IsNull(v)) {
				ConstantVector::SetNull(primary, true);
				return count;
			}
		}
		return 0;
	} else if (keys.ColumnCount() > 1) {
		//	Flatten the primary, as it will need to merge arbitrary validity masks
		primary.Flatten(count);
		auto &pvalidity = FlatVector::Validity(primary);
		D_ASSERT(keys.ColumnCount() == conditions.size());
		for (size_t c = 1; c < keys.data.size(); ++c) {
			// Skip comparisons that accept NULLs
			if (conditions[c].comparison == ExpressionType::COMPARE_DISTINCT_FROM) {
				continue;
			}
			//	ToUnifiedFormat the rest, as the sort code will do this anyway.
			auto &v = keys.data[c];
			UnifiedVectorFormat vdata;
			v.ToUnifiedFormat(count, vdata);
			auto &vvalidity = vdata.validity;
			if (vvalidity.AllValid()) {
				continue;
			}
			pvalidity.EnsureWritable();
			switch (v.GetVectorType()) {
			case VectorType::FLAT_VECTOR: {
				// Merge entire entries
				auto pmask = pvalidity.GetData();
				const auto entry_count = pvalidity.EntryCount(count);
				for (idx_t entry_idx = 0; entry_idx < entry_count; ++entry_idx) {
					pmask[entry_idx] &= vvalidity.GetValidityEntry(entry_idx);
				}
				break;
			}
			case VectorType::CONSTANT_VECTOR:
				// All or nothing
				if (ConstantVector::IsNull(v)) {
					pvalidity.SetAllInvalid(count);
					return count;
				}
				break;
			default:
				// One by one
				for (idx_t i = 0; i < count; ++i) {
					const auto idx = vdata.sel->get_index(i);
					if (!vvalidity.RowIsValidUnsafe(idx)) {
						pvalidity.SetInvalidUnsafe(i);
					}
				}
				break;
			}
		}
		return count - pvalidity.CountValid(count);
	} else {
		return count - VectorOperations::CountNotNull(primary, count);
	}
}

BufferHandle PhysicalRangeJoin::SliceSortedPayload(DataChunk &payload, GlobalSortState &state, const idx_t block_idx,
                                                   const SelectionVector &result, const idx_t result_count,
                                                   const idx_t left_cols) {
	// There should only be one sorted block if they have been sorted
	D_ASSERT(state.sorted_blocks.size() == 1);
	SBScanState read_state(state.buffer_manager, state);
	read_state.sb = state.sorted_blocks[0].get();
	auto &sorted_data = *read_state.sb->payload_data;

	read_state.SetIndices(block_idx, 0);
	read_state.PinData(sorted_data);
	const auto data_ptr = read_state.DataPtr(sorted_data);
	data_ptr_t heap_ptr = nullptr;

	// Set up a batch of pointers to scan data from
	Vector addresses(LogicalType::POINTER, result_count);
	auto data_pointers = FlatVector::GetData<data_ptr_t>(addresses);

	// Set up the data pointers for the values that are actually referenced
	const idx_t &row_width = sorted_data.layout.GetRowWidth();

	auto prev_idx = result.get_index(0);
	SelectionVector gsel(result_count);
	idx_t addr_count = 0;
	gsel.set_index(0, addr_count);
	data_pointers[addr_count] = data_ptr + prev_idx * row_width;
	for (idx_t i = 1; i < result_count; ++i) {
		const auto row_idx = result.get_index(i);
		if (row_idx != prev_idx) {
			data_pointers[++addr_count] = data_ptr + row_idx * row_width;
			prev_idx = row_idx;
		}
		gsel.set_index(i, addr_count);
	}
	++addr_count;

	// Unswizzle the offsets back to pointers (if needed)
	if (!sorted_data.layout.AllConstant() && state.external) {
		heap_ptr = read_state.payload_heap_handle.Ptr();
	}

	// Deserialize the payload data
	auto sel = FlatVector::IncrementalSelectionVector();
	for (idx_t col_no = 0; col_no < sorted_data.layout.ColumnCount(); col_no++) {
		auto &col = payload.data[left_cols + col_no];
		RowOperations::Gather(addresses, *sel, col, *sel, addr_count, sorted_data.layout, col_no, 0, heap_ptr);
		col.Slice(gsel, result_count);
	}

	return std::move(read_state.payload_heap_handle);
}

idx_t PhysicalRangeJoin::SelectJoinTail(const ExpressionType &condition, Vector &left, Vector &right,
                                        const SelectionVector *sel, idx_t count, SelectionVector *true_sel) {
	switch (condition) {
	case ExpressionType::COMPARE_NOTEQUAL:
		return VectorOperations::NotEquals(left, right, sel, count, true_sel, nullptr);
	case ExpressionType::COMPARE_LESSTHAN:
		return VectorOperations::LessThan(left, right, sel, count, true_sel, nullptr);
	case ExpressionType::COMPARE_GREATERTHAN:
		return VectorOperations::GreaterThan(left, right, sel, count, true_sel, nullptr);
	case ExpressionType::COMPARE_LESSTHANOREQUALTO:
		return VectorOperations::LessThanEquals(left, right, sel, count, true_sel, nullptr);
	case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
		return VectorOperations::GreaterThanEquals(left, right, sel, count, true_sel, nullptr);
	case ExpressionType::COMPARE_DISTINCT_FROM:
		return VectorOperations::DistinctFrom(left, right, sel, count, true_sel, nullptr);
	case ExpressionType::COMPARE_NOT_DISTINCT_FROM:
		return VectorOperations::NotDistinctFrom(left, right, sel, count, true_sel, nullptr);
	case ExpressionType::COMPARE_EQUAL:
		return VectorOperations::Equals(left, right, sel, count, true_sel, nullptr);
	default:
		throw InternalException("Unsupported comparison type for PhysicalRangeJoin");
	}

	return count;
}

} // namespace duckdb









namespace duckdb {

PhysicalOrder::PhysicalOrder(vector<LogicalType> types, vector<BoundOrderByNode> orders, vector<idx_t> projections,
                             idx_t estimated_cardinality)
    : PhysicalOperator(PhysicalOperatorType::ORDER_BY, std::move(types), estimated_cardinality),
      orders(std::move(orders)), projections(std::move(projections)) {
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
class OrderGlobalSinkState : public GlobalSinkState {
public:
	OrderGlobalSinkState(BufferManager &buffer_manager, const PhysicalOrder &order, RowLayout &payload_layout)
	    : global_sort_state(buffer_manager, order.orders, payload_layout) {
	}

	//! Global sort state
	GlobalSortState global_sort_state;
	//! Memory usage per thread
	idx_t memory_per_thread;
};

class OrderLocalSinkState : public LocalSinkState {
public:
	OrderLocalSinkState(ClientContext &context, const PhysicalOrder &op) : key_executor(context) {
		// Initialize order clause expression executor and DataChunk
		vector<LogicalType> key_types;
		for (auto &order : op.orders) {
			key_types.push_back(order.expression->return_type);
			key_executor.AddExpression(*order.expression);
		}
		auto &allocator = Allocator::Get(context);
		keys.Initialize(allocator, key_types);
		payload.Initialize(allocator, op.types);
	}

public:
	//! The local sort state
	LocalSortState local_sort_state;
	//! Key expression executor, and chunk to hold the vectors
	ExpressionExecutor key_executor;
	DataChunk keys;
	//! Payload chunk to hold the vectors
	DataChunk payload;
};

unique_ptr<GlobalSinkState> PhysicalOrder::GetGlobalSinkState(ClientContext &context) const {
	// Get the payload layout from the return types
	RowLayout payload_layout;
	payload_layout.Initialize(types);
	auto state = make_uniq<OrderGlobalSinkState>(BufferManager::GetBufferManager(context), *this, payload_layout);
	// Set external (can be force with the PRAGMA)
	state->global_sort_state.external = ClientConfig::GetConfig(context).force_external;
	state->memory_per_thread = GetMaxThreadMemory(context);
	return std::move(state);
}

unique_ptr<LocalSinkState> PhysicalOrder::GetLocalSinkState(ExecutionContext &context) const {
	return make_uniq<OrderLocalSinkState>(context.client, *this);
}

SinkResultType PhysicalOrder::Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const {
	auto &gstate = input.global_state.Cast<OrderGlobalSinkState>();
	auto &lstate = input.local_state.Cast<OrderLocalSinkState>();

	auto &global_sort_state = gstate.global_sort_state;
	auto &local_sort_state = lstate.local_sort_state;

	// Initialize local state (if necessary)
	if (!local_sort_state.initialized) {
		local_sort_state.Initialize(global_sort_state, BufferManager::GetBufferManager(context.client));
	}

	// Obtain sorting columns
	auto &keys = lstate.keys;
	keys.Reset();
	lstate.key_executor.Execute(chunk, keys);

	auto &payload = lstate.payload;
	payload.ReferenceColumns(chunk, projections);

	// Sink the data into the local sort state
	keys.Verify();
	chunk.Verify();
	local_sort_state.SinkChunk(keys, payload);

	// When sorting data reaches a certain size, we sort it
	if (local_sort_state.SizeInBytes() >= gstate.memory_per_thread) {
		local_sort_state.Sort(global_sort_state, true);
	}
	return SinkResultType::NEED_MORE_INPUT;
}

void PhysicalOrder::Combine(ExecutionContext &context, GlobalSinkState &gstate_p, LocalSinkState &lstate_p) const {
	auto &gstate = gstate_p.Cast<OrderGlobalSinkState>();
	auto &lstate = lstate_p.Cast<OrderLocalSinkState>();
	gstate.global_sort_state.AddLocalState(lstate.local_sort_state);
}

class PhysicalOrderMergeTask : public ExecutorTask {
public:
	PhysicalOrderMergeTask(shared_ptr<Event> event_p, ClientContext &context, OrderGlobalSinkState &state)
	    : ExecutorTask(context), event(std::move(event_p)), context(context), state(state) {
	}

	TaskExecutionResult ExecuteTask(TaskExecutionMode mode) override {
		// Initialize merge sorted and iterate until done
		auto &global_sort_state = state.global_sort_state;
		MergeSorter merge_sorter(global_sort_state, BufferManager::GetBufferManager(context));
		merge_sorter.PerformInMergeRound();
		event->FinishTask();
		return TaskExecutionResult::TASK_FINISHED;
	}

private:
	shared_ptr<Event> event;
	ClientContext &context;
	OrderGlobalSinkState &state;
};

class OrderMergeEvent : public BasePipelineEvent {
public:
	OrderMergeEvent(OrderGlobalSinkState &gstate_p, Pipeline &pipeline_p)
	    : BasePipelineEvent(pipeline_p), gstate(gstate_p) {
	}

	OrderGlobalSinkState &gstate;

public:
	void Schedule() override {
		auto &context = pipeline->GetClientContext();

		// Schedule tasks equal to the number of threads, which will each merge multiple partitions
		auto &ts = TaskScheduler::GetScheduler(context);
		idx_t num_threads = ts.NumberOfThreads();

		vector<shared_ptr<Task>> merge_tasks;
		for (idx_t tnum = 0; tnum < num_threads; tnum++) {
			merge_tasks.push_back(make_uniq<PhysicalOrderMergeTask>(shared_from_this(), context, gstate));
		}
		SetTasks(std::move(merge_tasks));
	}

	void FinishEvent() override {
		auto &global_sort_state = gstate.global_sort_state;

		global_sort_state.CompleteMergeRound();
		if (global_sort_state.sorted_blocks.size() > 1) {
			// Multiple blocks remaining: Schedule the next round
			PhysicalOrder::ScheduleMergeTasks(*pipeline, *this, gstate);
		}
	}
};

SinkFinalizeType PhysicalOrder::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                         GlobalSinkState &gstate_p) const {
	auto &state = gstate_p.Cast<OrderGlobalSinkState>();
	auto &global_sort_state = state.global_sort_state;

	if (global_sort_state.sorted_blocks.empty()) {
		// Empty input!
		return SinkFinalizeType::NO_OUTPUT_POSSIBLE;
	}

	// Prepare for merge sort phase
	global_sort_state.PrepareMergePhase();

	// Start the merge phase or finish if a merge is not necessary
	if (global_sort_state.sorted_blocks.size() > 1) {
		PhysicalOrder::ScheduleMergeTasks(pipeline, event, state);
	}
	return SinkFinalizeType::READY;
}

void PhysicalOrder::ScheduleMergeTasks(Pipeline &pipeline, Event &event, OrderGlobalSinkState &state) {
	// Initialize global sort state for a round of merging
	state.global_sort_state.InitializeMergeRound();
	auto new_event = make_shared<OrderMergeEvent>(state, pipeline);
	event.InsertEvent(std::move(new_event));
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
class PhysicalOrderGlobalSourceState : public GlobalSourceState {
public:
	explicit PhysicalOrderGlobalSourceState(OrderGlobalSinkState &sink) : next_batch_index(0) {
		auto &global_sort_state = sink.global_sort_state;
		if (global_sort_state.sorted_blocks.empty()) {
			total_batches = 0;
		} else {
			D_ASSERT(global_sort_state.sorted_blocks.size() == 1);
			total_batches = global_sort_state.sorted_blocks[0]->payload_data->data_blocks.size();
		}
	}

	idx_t MaxThreads() override {
		return total_batches;
	}

public:
	atomic<idx_t> next_batch_index;
	idx_t total_batches;
};

unique_ptr<GlobalSourceState> PhysicalOrder::GetGlobalSourceState(ClientContext &context) const {
	auto &sink = this->sink_state->Cast<OrderGlobalSinkState>();
	return make_uniq<PhysicalOrderGlobalSourceState>(sink);
}

class PhysicalOrderLocalSourceState : public LocalSourceState {
public:
	explicit PhysicalOrderLocalSourceState(PhysicalOrderGlobalSourceState &gstate)
	    : batch_index(gstate.next_batch_index++) {
	}

public:
	idx_t batch_index;
	unique_ptr<PayloadScanner> scanner;
};

unique_ptr<LocalSourceState> PhysicalOrder::GetLocalSourceState(ExecutionContext &context,
                                                                GlobalSourceState &gstate_p) const {
	auto &gstate = gstate_p.Cast<PhysicalOrderGlobalSourceState>();
	return make_uniq<PhysicalOrderLocalSourceState>(gstate);
}

SourceResultType PhysicalOrder::GetData(ExecutionContext &context, DataChunk &chunk, OperatorSourceInput &input) const {
	auto &gstate = input.global_state.Cast<PhysicalOrderGlobalSourceState>();
	auto &lstate = input.local_state.Cast<PhysicalOrderLocalSourceState>();

	if (lstate.scanner && lstate.scanner->Remaining() == 0) {
		lstate.batch_index = gstate.next_batch_index++;
		lstate.scanner = nullptr;
	}

	if (lstate.batch_index >= gstate.total_batches) {
		return SourceResultType::FINISHED;
	}

	if (!lstate.scanner) {
		auto &sink = this->sink_state->Cast<OrderGlobalSinkState>();
		auto &global_sort_state = sink.global_sort_state;
		lstate.scanner = make_uniq<PayloadScanner>(global_sort_state, lstate.batch_index, true);
	}

	lstate.scanner->Scan(chunk);

	return chunk.size() == 0 ? SourceResultType::FINISHED : SourceResultType::HAVE_MORE_OUTPUT;
}

idx_t PhysicalOrder::GetBatchIndex(ExecutionContext &context, DataChunk &chunk, GlobalSourceState &gstate_p,
                                   LocalSourceState &lstate_p) const {
	auto &lstate = lstate_p.Cast<PhysicalOrderLocalSourceState>();
	return lstate.batch_index;
}

string PhysicalOrder::ParamsToString() const {
	string result = "ORDERS:\n";
	for (idx_t i = 0; i < orders.size(); i++) {
		if (i > 0) {
			result += "\n";
		}
		result += orders[i].expression->ToString() + " ";
		result += orders[i].type == OrderType::DESCENDING ? "DESC" : "ASC";
	}
	return result;
}

} // namespace duckdb










namespace duckdb {

PhysicalTopN::PhysicalTopN(vector<LogicalType> types, vector<BoundOrderByNode> orders, idx_t limit, idx_t offset,
                           idx_t estimated_cardinality)
    : PhysicalOperator(PhysicalOperatorType::TOP_N, std::move(types), estimated_cardinality), orders(std::move(orders)),
      limit(limit), offset(offset) {
}

//===--------------------------------------------------------------------===//
// Heaps
//===--------------------------------------------------------------------===//
class TopNHeap;

struct TopNScanState {
	unique_ptr<PayloadScanner> scanner;
	idx_t pos;
	bool exclude_offset;
};

class TopNSortState {
public:
	explicit TopNSortState(TopNHeap &heap);

	TopNHeap &heap;
	unique_ptr<LocalSortState> local_state;
	unique_ptr<GlobalSortState> global_state;
	idx_t count;
	bool is_sorted;

public:
	void Initialize();
	void Append(DataChunk &sort_chunk, DataChunk &payload);

	void Sink(DataChunk &input);
	void Finalize();

	void Move(TopNSortState &other);

	void InitializeScan(TopNScanState &state, bool exclude_offset);
	void Scan(TopNScanState &state, DataChunk &chunk);
};

class TopNHeap {
public:
	TopNHeap(ClientContext &context, const vector<LogicalType> &payload_types, const vector<BoundOrderByNode> &orders,
	         idx_t limit, idx_t offset);
	TopNHeap(ExecutionContext &context, const vector<LogicalType> &payload_types,
	         const vector<BoundOrderByNode> &orders, idx_t limit, idx_t offset);
	TopNHeap(ClientContext &context, Allocator &allocator, const vector<LogicalType> &payload_types,
	         const vector<BoundOrderByNode> &orders, idx_t limit, idx_t offset);

	Allocator &allocator;
	BufferManager &buffer_manager;
	const vector<LogicalType> &payload_types;
	const vector<BoundOrderByNode> &orders;
	idx_t limit;
	idx_t offset;
	TopNSortState sort_state;
	ExpressionExecutor executor;
	DataChunk sort_chunk;
	DataChunk compare_chunk;
	DataChunk payload_chunk;
	//! A set of boundary values that determine either the minimum or the maximum value we have to consider for our
	//! top-n
	DataChunk boundary_values;
	//! Whether or not the boundary_values has been set. The boundary_values are only set after a reduce step
	bool has_boundary_values;

	SelectionVector final_sel;
	SelectionVector true_sel;
	SelectionVector false_sel;
	SelectionVector new_remaining_sel;

public:
	void Sink(DataChunk &input);
	void Combine(TopNHeap &other);
	void Reduce();
	void Finalize();

	void ExtractBoundaryValues(DataChunk &current_chunk, DataChunk &prev_chunk);

	void InitializeScan(TopNScanState &state, bool exclude_offset);
	void Scan(TopNScanState &state, DataChunk &chunk);

	bool CheckBoundaryValues(DataChunk &sort_chunk, DataChunk &payload);
};

//===--------------------------------------------------------------------===//
// TopNSortState
//===--------------------------------------------------------------------===//
TopNSortState::TopNSortState(TopNHeap &heap) : heap(heap), count(0), is_sorted(false) {
}

void TopNSortState::Initialize() {
	RowLayout layout;
	layout.Initialize(heap.payload_types);
	auto &buffer_manager = heap.buffer_manager;
	global_state = make_uniq<GlobalSortState>(buffer_manager, heap.orders, layout);
	local_state = make_uniq<LocalSortState>();
	local_state->Initialize(*global_state, buffer_manager);
}

void TopNSortState::Append(DataChunk &sort_chunk, DataChunk &payload) {
	D_ASSERT(!is_sorted);
	if (heap.has_boundary_values) {
		if (!heap.CheckBoundaryValues(sort_chunk, payload)) {
			return;
		}
	}

	local_state->SinkChunk(sort_chunk, payload);
	count += payload.size();
}

void TopNSortState::Sink(DataChunk &input) {
	// compute the ordering values for the new chunk
	heap.sort_chunk.Reset();
	heap.executor.Execute(input, heap.sort_chunk);

	// append the new chunk to what we have already
	Append(heap.sort_chunk, input);
}

void TopNSortState::Move(TopNSortState &other) {
	local_state = std::move(other.local_state);
	global_state = std::move(other.global_state);
	count = other.count;
	is_sorted = other.is_sorted;
}

void TopNSortState::Finalize() {
	D_ASSERT(!is_sorted);
	global_state->AddLocalState(*local_state);

	global_state->PrepareMergePhase();
	while (global_state->sorted_blocks.size() > 1) {
		MergeSorter merge_sorter(*global_state, heap.buffer_manager);
		merge_sorter.PerformInMergeRound();
		global_state->CompleteMergeRound();
	}
	is_sorted = true;
}

void TopNSortState::InitializeScan(TopNScanState &state, bool exclude_offset) {
	D_ASSERT(is_sorted);
	if (global_state->sorted_blocks.empty()) {
		state.scanner = nullptr;
	} else {
		D_ASSERT(global_state->sorted_blocks.size() == 1);
		state.scanner = make_uniq<PayloadScanner>(*global_state->sorted_blocks[0]->payload_data, *global_state);
	}
	state.pos = 0;
	state.exclude_offset = exclude_offset && heap.offset > 0;
}

void TopNSortState::Scan(TopNScanState &state, DataChunk &chunk) {
	if (!state.scanner) {
		return;
	}
	auto offset = heap.offset;
	auto limit = heap.limit;
	D_ASSERT(is_sorted);
	while (chunk.size() == 0) {
		state.scanner->Scan(chunk);
		if (chunk.size() == 0) {
			break;
		}
		idx_t start = state.pos;
		idx_t end = state.pos + chunk.size();
		state.pos = end;

		idx_t chunk_start = 0;
		idx_t chunk_end = chunk.size();
		if (state.exclude_offset) {
			// we need to exclude all tuples before the OFFSET
			// check if we should include anything
			if (end <= offset) {
				// end is smaller than offset: include nothing!
				chunk.Reset();
				continue;
			} else if (start < offset) {
				// we need to slice
				chunk_start = offset - start;
			}
		}
		// check if we need to truncate at the offset + limit mark
		if (start >= offset + limit) {
			// we are finished
			chunk_end = 0;
		} else if (end > offset + limit) {
			// the end extends past the offset + limit
			// truncate the current chunk
			chunk_end = offset + limit - start;
		}
		D_ASSERT(chunk_end - chunk_start <= STANDARD_VECTOR_SIZE);
		if (chunk_end == chunk_start) {
			chunk.Reset();
			break;
		} else if (chunk_start > 0) {
			SelectionVector sel(STANDARD_VECTOR_SIZE);
			for (idx_t i = chunk_start; i < chunk_end; i++) {
				sel.set_index(i - chunk_start, i);
			}
			chunk.Slice(sel, chunk_end - chunk_start);
		} else if (chunk_end != chunk.size()) {
			chunk.SetCardinality(chunk_end);
		}
	}
}

//===--------------------------------------------------------------------===//
// TopNHeap
//===--------------------------------------------------------------------===//
TopNHeap::TopNHeap(ClientContext &context, Allocator &allocator, const vector<LogicalType> &payload_types_p,
                   const vector<BoundOrderByNode> &orders_p, idx_t limit, idx_t offset)
    : allocator(allocator), buffer_manager(BufferManager::GetBufferManager(context)), payload_types(payload_types_p),
      orders(orders_p), limit(limit), offset(offset), sort_state(*this), executor(context), has_boundary_values(false),
      final_sel(STANDARD_VECTOR_SIZE), true_sel(STANDARD_VECTOR_SIZE), false_sel(STANDARD_VECTOR_SIZE),
      new_remaining_sel(STANDARD_VECTOR_SIZE) {
	// initialize the executor and the sort_chunk
	vector<LogicalType> sort_types;
	for (auto &order : orders) {
		auto &expr = order.expression;
		sort_types.push_back(expr->return_type);
		executor.AddExpression(*expr);
	}
	payload_chunk.Initialize(allocator, payload_types);
	sort_chunk.Initialize(allocator, sort_types);
	compare_chunk.Initialize(allocator, sort_types);
	boundary_values.Initialize(allocator, sort_types);
	sort_state.Initialize();
}

TopNHeap::TopNHeap(ClientContext &context, const vector<LogicalType> &payload_types,
                   const vector<BoundOrderByNode> &orders, idx_t limit, idx_t offset)
    : TopNHeap(context, BufferAllocator::Get(context), payload_types, orders, limit, offset) {
}

TopNHeap::TopNHeap(ExecutionContext &context, const vector<LogicalType> &payload_types,
                   const vector<BoundOrderByNode> &orders, idx_t limit, idx_t offset)
    : TopNHeap(context.client, Allocator::Get(context.client), payload_types, orders, limit, offset) {
}

void TopNHeap::Sink(DataChunk &input) {
	sort_state.Sink(input);
}

void TopNHeap::Combine(TopNHeap &other) {
	other.Finalize();

	TopNScanState state;
	other.InitializeScan(state, false);
	while (true) {
		payload_chunk.Reset();
		other.Scan(state, payload_chunk);
		if (payload_chunk.size() == 0) {
			break;
		}
		Sink(payload_chunk);
	}
	Reduce();
}

void TopNHeap::Finalize() {
	sort_state.Finalize();
}

void TopNHeap::Reduce() {
	idx_t min_sort_threshold = MaxValue<idx_t>(STANDARD_VECTOR_SIZE * 5, 2 * (limit + offset));
	if (sort_state.count < min_sort_threshold) {
		// only reduce when we pass two times the limit + offset, or 5 vectors (whichever comes first)
		return;
	}
	sort_state.Finalize();
	TopNSortState new_state(*this);
	new_state.Initialize();

	TopNScanState state;
	sort_state.InitializeScan(state, false);

	DataChunk new_chunk;
	new_chunk.Initialize(allocator, payload_types);

	DataChunk *current_chunk = &new_chunk;
	DataChunk *prev_chunk = &payload_chunk;
	has_boundary_values = false;
	while (true) {
		current_chunk->Reset();
		Scan(state, *current_chunk);
		if (current_chunk->size() == 0) {
			ExtractBoundaryValues(*current_chunk, *prev_chunk);
			break;
		}
		new_state.Sink(*current_chunk);
		std::swap(current_chunk, prev_chunk);
	}

	sort_state.Move(new_state);
}

void TopNHeap::ExtractBoundaryValues(DataChunk &current_chunk, DataChunk &prev_chunk) {
	// extract the last entry of the prev_chunk and set as minimum value
	D_ASSERT(prev_chunk.size() > 0);
	for (idx_t col_idx = 0; col_idx < current_chunk.ColumnCount(); col_idx++) {
		ConstantVector::Reference(current_chunk.data[col_idx], prev_chunk.data[col_idx], prev_chunk.size() - 1,
		                          prev_chunk.size());
	}
	current_chunk.SetCardinality(1);
	sort_chunk.Reset();
	executor.Execute(&current_chunk, sort_chunk);

	boundary_values.Reset();
	boundary_values.Append(sort_chunk);
	boundary_values.SetCardinality(1);
	for (idx_t i = 0; i < boundary_values.ColumnCount(); i++) {
		boundary_values.data[i].SetVectorType(VectorType::CONSTANT_VECTOR);
	}
	has_boundary_values = true;
}

bool TopNHeap::CheckBoundaryValues(DataChunk &sort_chunk, DataChunk &payload) {
	// we have boundary values
	// from these boundary values, determine which values we should insert (if any)
	idx_t final_count = 0;

	SelectionVector remaining_sel(nullptr);
	idx_t remaining_count = sort_chunk.size();
	for (idx_t i = 0; i < orders.size(); i++) {
		if (remaining_sel.data()) {
			compare_chunk.data[i].Slice(sort_chunk.data[i], remaining_sel, remaining_count);
		} else {
			compare_chunk.data[i].Reference(sort_chunk.data[i]);
		}
		bool is_last = i + 1 == orders.size();
		idx_t true_count;
		if (orders[i].null_order == OrderByNullType::NULLS_LAST) {
			if (orders[i].type == OrderType::ASCENDING) {
				true_count = VectorOperations::DistinctLessThan(compare_chunk.data[i], boundary_values.data[i],
				                                                &remaining_sel, remaining_count, &true_sel, &false_sel);
			} else {
				true_count = VectorOperations::DistinctGreaterThanNullsFirst(compare_chunk.data[i],
				                                                             boundary_values.data[i], &remaining_sel,
				                                                             remaining_count, &true_sel, &false_sel);
			}
		} else {
			D_ASSERT(orders[i].null_order == OrderByNullType::NULLS_FIRST);
			if (orders[i].type == OrderType::ASCENDING) {
				true_count = VectorOperations::DistinctLessThanNullsFirst(compare_chunk.data[i],
				                                                          boundary_values.data[i], &remaining_sel,
				                                                          remaining_count, &true_sel, &false_sel);
			} else {
				true_count =
				    VectorOperations::DistinctGreaterThan(compare_chunk.data[i], boundary_values.data[i],
				                                          &remaining_sel, remaining_count, &true_sel, &false_sel);
			}
		}

		if (true_count > 0) {
			memcpy(final_sel.data() + final_count, true_sel.data(), true_count * sizeof(sel_t));
			final_count += true_count;
		}
		idx_t false_count = remaining_count - true_count;
		if (false_count > 0) {
			// check what we should continue to check
			compare_chunk.data[i].Slice(sort_chunk.data[i], false_sel, false_count);
			remaining_count = VectorOperations::NotDistinctFrom(compare_chunk.data[i], boundary_values.data[i],
			                                                    &false_sel, false_count, &new_remaining_sel, nullptr);
			if (is_last) {
				memcpy(final_sel.data() + final_count, new_remaining_sel.data(), remaining_count * sizeof(sel_t));
				final_count += remaining_count;
			} else {
				remaining_sel.Initialize(new_remaining_sel);
			}
		} else {
			break;
		}
	}
	if (final_count == 0) {
		return false;
	}
	if (final_count < sort_chunk.size()) {
		sort_chunk.Slice(final_sel, final_count);
		payload.Slice(final_sel, final_count);
	}
	return true;
}

void TopNHeap::InitializeScan(TopNScanState &state, bool exclude_offset) {
	sort_state.InitializeScan(state, exclude_offset);
}

void TopNHeap::Scan(TopNScanState &state, DataChunk &chunk) {
	sort_state.Scan(state, chunk);
}

class TopNGlobalState : public GlobalSinkState {
public:
	TopNGlobalState(ClientContext &context, const vector<LogicalType> &payload_types,
	                const vector<BoundOrderByNode> &orders, idx_t limit, idx_t offset)
	    : heap(context, payload_types, orders, limit, offset) {
	}

	mutex lock;
	TopNHeap heap;
};

class TopNLocalState : public LocalSinkState {
public:
	TopNLocalState(ExecutionContext &context, const vector<LogicalType> &payload_types,
	               const vector<BoundOrderByNode> &orders, idx_t limit, idx_t offset)
	    : heap(context, payload_types, orders, limit, offset) {
	}

	TopNHeap heap;
};

unique_ptr<LocalSinkState> PhysicalTopN::GetLocalSinkState(ExecutionContext &context) const {
	return make_uniq<TopNLocalState>(context, types, orders, limit, offset);
}

unique_ptr<GlobalSinkState> PhysicalTopN::GetGlobalSinkState(ClientContext &context) const {
	return make_uniq<TopNGlobalState>(context, types, orders, limit, offset);
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
SinkResultType PhysicalTopN::Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const {
	// append to the local sink state
	auto &sink = input.local_state.Cast<TopNLocalState>();
	sink.heap.Sink(chunk);
	sink.heap.Reduce();
	return SinkResultType::NEED_MORE_INPUT;
}

//===--------------------------------------------------------------------===//
// Combine
//===--------------------------------------------------------------------===//
void PhysicalTopN::Combine(ExecutionContext &context, GlobalSinkState &state, LocalSinkState &lstate_p) const {
	auto &gstate = state.Cast<TopNGlobalState>();
	auto &lstate = lstate_p.Cast<TopNLocalState>();

	// scan the local top N and append it to the global heap
	lock_guard<mutex> glock(gstate.lock);
	gstate.heap.Combine(lstate.heap);
}

//===--------------------------------------------------------------------===//
// Finalize
//===--------------------------------------------------------------------===//
SinkFinalizeType PhysicalTopN::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                        GlobalSinkState &gstate_p) const {
	auto &gstate = gstate_p.Cast<TopNGlobalState>();
	// global finalize: compute the final top N
	gstate.heap.Finalize();
	return SinkFinalizeType::READY;
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
class TopNOperatorState : public GlobalSourceState {
public:
	TopNScanState state;
	bool initialized = false;
};

unique_ptr<GlobalSourceState> PhysicalTopN::GetGlobalSourceState(ClientContext &context) const {
	return make_uniq<TopNOperatorState>();
}

SourceResultType PhysicalTopN::GetData(ExecutionContext &context, DataChunk &chunk, OperatorSourceInput &input) const {
	if (limit == 0) {
		return SourceResultType::FINISHED;
	}
	auto &state = input.global_state.Cast<TopNOperatorState>();
	auto &gstate = sink_state->Cast<TopNGlobalState>();

	if (!state.initialized) {
		gstate.heap.InitializeScan(state.state, true);
		state.initialized = true;
	}
	gstate.heap.Scan(state.state, chunk);

	return chunk.size() == 0 ? SourceResultType::FINISHED : SourceResultType::HAVE_MORE_OUTPUT;
}

string PhysicalTopN::ParamsToString() const {
	string result;
	result += "Top " + to_string(limit);
	if (offset > 0) {
		result += "\n";
		result += "Offset " + to_string(offset);
	}
	result += "\n[INFOSEPARATOR]";
	for (idx_t i = 0; i < orders.size(); i++) {
		result += "\n";
		result += orders[i].expression->ToString() + " ";
		result += orders[i].type == OrderType::DESCENDING ? "DESC" : "ASC";
	}
	return result;
}

} // namespace duckdb





















#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>

namespace duckdb {

string BaseCSVReader::GetLineNumberStr(idx_t line_error, bool is_line_estimated, idx_t buffer_idx) {
	// If an error happens during auto-detect it is an estimated line
	string estimated = (is_line_estimated ? string(" (estimated)") : string(""));
	return to_string(GetLineError(line_error, buffer_idx)) + estimated;
}

BaseCSVReader::BaseCSVReader(ClientContext &context_p, BufferedCSVReaderOptions options_p,
                             const vector<LogicalType> &requested_types)
    : context(context_p), fs(FileSystem::GetFileSystem(context)), allocator(BufferAllocator::Get(context)),
      options(std::move(options_p)) {
}

BaseCSVReader::~BaseCSVReader() {
}

unique_ptr<CSVFileHandle> BaseCSVReader::OpenCSV(const BufferedCSVReaderOptions &options_p) {
	return CSVFileHandle::OpenFile(fs, allocator, options_p.file_path, options_p.compression, true);
}

void BaseCSVReader::InitParseChunk(idx_t num_cols) {
	// adapt not null info
	if (options.force_not_null.size() != num_cols) {
		options.force_not_null.resize(num_cols, false);
	}
	if (num_cols == parse_chunk.ColumnCount()) {
		parse_chunk.Reset();
	} else {
		parse_chunk.Destroy();

		// initialize the parse_chunk with a set of VARCHAR types
		vector<LogicalType> varchar_types(num_cols, LogicalType::VARCHAR);
		parse_chunk.Initialize(allocator, varchar_types);
	}
}

void BaseCSVReader::InitializeProjection() {
	for (idx_t i = 0; i < GetTypes().size(); i++) {
		reader_data.column_ids.push_back(i);
		reader_data.column_mapping.push_back(i);
	}
}

void BaseCSVReader::SetDateFormat(const string &format_specifier, const LogicalTypeId &sql_type) {
	options.has_format[sql_type] = true;
	auto &date_format = options.date_format[sql_type];
	date_format.format_specifier = format_specifier;
	StrTimeFormat::ParseFormatSpecifier(date_format.format_specifier, date_format);
}

struct TryCastDecimalOperator {
	template <class OP, class T>
	static bool Operation(string_t input, uint8_t width, uint8_t scale) {
		T result;
		string error_message;
		return OP::Operation(input, result, &error_message, width, scale);
	}
};

struct TryCastFloatingOperator {
	template <class OP, class T>
	static bool Operation(string_t input) {
		T result;
		string error_message;
		return OP::Operation(input, result, &error_message);
	}
};

bool TryCastDecimalValueCommaSeparated(const string_t &value_str, const LogicalType &sql_type) {
	auto width = DecimalType::GetWidth(sql_type);
	auto scale = DecimalType::GetScale(sql_type);
	switch (sql_type.InternalType()) {
	case PhysicalType::INT16:
		return TryCastDecimalOperator::Operation<TryCastToDecimalCommaSeparated, int16_t>(value_str, width, scale);
	case PhysicalType::INT32:
		return TryCastDecimalOperator::Operation<TryCastToDecimalCommaSeparated, int32_t>(value_str, width, scale);
	case PhysicalType::INT64:
		return TryCastDecimalOperator::Operation<TryCastToDecimalCommaSeparated, int64_t>(value_str, width, scale);
	case PhysicalType::INT128:
		return TryCastDecimalOperator::Operation<TryCastToDecimalCommaSeparated, hugeint_t>(value_str, width, scale);
	default:
		throw InternalException("Unimplemented physical type for decimal");
	}
}

bool TryCastFloatingValueCommaSeparated(const string_t &value_str, const LogicalType &sql_type) {
	switch (sql_type.InternalType()) {
	case PhysicalType::DOUBLE:
		return TryCastFloatingOperator::Operation<TryCastErrorMessageCommaSeparated, double>(value_str);
	case PhysicalType::FLOAT:
		return TryCastFloatingOperator::Operation<TryCastErrorMessageCommaSeparated, float>(value_str);
	default:
		throw InternalException("Unimplemented physical type for floating");
	}
}

bool BaseCSVReader::TryCastValue(const Value &value, const LogicalType &sql_type) {
	if (value.IsNull()) {
		return true;
	}
	if (options.has_format[LogicalTypeId::DATE] && sql_type.id() == LogicalTypeId::DATE) {
		date_t result;
		string error_message;
		return options.date_format[LogicalTypeId::DATE].TryParseDate(string_t(StringValue::Get(value)), result,
		                                                             error_message);
	} else if (options.has_format[LogicalTypeId::TIMESTAMP] && sql_type.id() == LogicalTypeId::TIMESTAMP) {
		timestamp_t result;
		string error_message;
		return options.date_format[LogicalTypeId::TIMESTAMP].TryParseTimestamp(string_t(StringValue::Get(value)),
		                                                                       result, error_message);
	} else if (options.decimal_separator != "." && sql_type.id() == LogicalTypeId::DECIMAL) {
		return TryCastDecimalValueCommaSeparated(string_t(StringValue::Get(value)), sql_type);
	} else if (options.decimal_separator != "." &&
	           ((sql_type.id() == LogicalTypeId::FLOAT) || (sql_type.id() == LogicalTypeId::DOUBLE))) {
		return TryCastFloatingValueCommaSeparated(string_t(StringValue::Get(value)), sql_type);
	} else {
		Value new_value;
		string error_message;
		return value.TryCastAs(context, sql_type, new_value, &error_message, true);
	}
}

struct TryCastDateOperator {
	static bool Operation(BufferedCSVReaderOptions &options, string_t input, date_t &result, string &error_message) {
		return options.date_format[LogicalTypeId::DATE].TryParseDate(input, result, error_message);
	}
};

struct TryCastTimestampOperator {
	static bool Operation(BufferedCSVReaderOptions &options, string_t input, timestamp_t &result,
	                      string &error_message) {
		return options.date_format[LogicalTypeId::TIMESTAMP].TryParseTimestamp(input, result, error_message);
	}
};

template <class OP, class T>
static bool TemplatedTryCastDateVector(BufferedCSVReaderOptions &options, Vector &input_vector, Vector &result_vector,
                                       idx_t count, string &error_message, idx_t &line_error) {
	D_ASSERT(input_vector.GetType().id() == LogicalTypeId::VARCHAR);
	bool all_converted = true;
	idx_t cur_line = 0;
	UnaryExecutor::Execute<string_t, T>(input_vector, result_vector, count, [&](string_t input) {
		T result;
		if (!OP::Operation(options, input, result, error_message)) {
			line_error = cur_line;
			all_converted = false;
		}
		cur_line++;
		return result;
	});
	return all_converted;
}

bool TryCastDateVector(BufferedCSVReaderOptions &options, Vector &input_vector, Vector &result_vector, idx_t count,
                       string &error_message, idx_t &line_error) {
	return TemplatedTryCastDateVector<TryCastDateOperator, date_t>(options, input_vector, result_vector, count,
	                                                               error_message, line_error);
}

bool TryCastTimestampVector(BufferedCSVReaderOptions &options, Vector &input_vector, Vector &result_vector, idx_t count,
                            string &error_message) {
	idx_t line_error;
	return TemplatedTryCastDateVector<TryCastTimestampOperator, timestamp_t>(options, input_vector, result_vector,
	                                                                         count, error_message, line_error);
}

template <class OP, class T>
bool TemplatedTryCastFloatingVector(BufferedCSVReaderOptions &options, Vector &input_vector, Vector &result_vector,
                                    idx_t count, string &error_message, idx_t &line_error) {
	D_ASSERT(input_vector.GetType().id() == LogicalTypeId::VARCHAR);
	bool all_converted = true;
	idx_t row = 0;
	UnaryExecutor::Execute<string_t, T>(input_vector, result_vector, count, [&](string_t input) {
		T result;
		if (!OP::Operation(input, result, &error_message)) {
			line_error = row;
			all_converted = false;
		} else {
			row++;
		}
		return result;
	});
	return all_converted;
}

template <class OP, class T>
bool TemplatedTryCastDecimalVector(BufferedCSVReaderOptions &options, Vector &input_vector, Vector &result_vector,
                                   idx_t count, string &error_message, uint8_t width, uint8_t scale) {
	D_ASSERT(input_vector.GetType().id() == LogicalTypeId::VARCHAR);
	bool all_converted = true;
	UnaryExecutor::Execute<string_t, T>(input_vector, result_vector, count, [&](string_t input) {
		T result;
		if (!OP::Operation(input, result, &error_message, width, scale)) {
			all_converted = false;
		}
		return result;
	});
	return all_converted;
}

bool BaseCSVReader::TryCastVector(Vector &parse_chunk_col, idx_t size, const LogicalType &sql_type) {
	// try vector-cast from string to sql_type
	Vector dummy_result(sql_type);
	if (options.has_format[LogicalTypeId::DATE] && sql_type == LogicalTypeId::DATE) {
		// use the date format to cast the chunk
		string error_message;
		idx_t line_error;
		return TryCastDateVector(options, parse_chunk_col, dummy_result, size, error_message, line_error);
	} else if (options.has_format[LogicalTypeId::TIMESTAMP] && sql_type == LogicalTypeId::TIMESTAMP) {
		// use the timestamp format to cast the chunk
		string error_message;
		return TryCastTimestampVector(options, parse_chunk_col, dummy_result, size, error_message);
	} else {
		// target type is not varchar: perform a cast
		string error_message;
		return VectorOperations::DefaultTryCast(parse_chunk_col, dummy_result, size, &error_message, true);
	}
}

void BaseCSVReader::AddValue(string_t str_val, idx_t &column, vector<idx_t> &escape_positions, bool has_quotes,
                             idx_t buffer_idx) {
	auto length = str_val.GetSize();
	if (length == 0 && column == 0) {
		row_empty = true;
	} else {
		row_empty = false;
	}
	if (!return_types.empty() && column == return_types.size() && length == 0) {
		// skip a single trailing delimiter in last column
		return;
	}
	if (mode == ParserMode::SNIFFING_DIALECT) {
		column++;
		return;
	}
	if (column >= return_types.size()) {
		if (options.ignore_errors) {
			error_column_overflow = true;
			return;
		} else {
			throw InvalidInputException(
			    "Error in file \"%s\", on line %s: expected %lld values per row, but got more. (%s)", options.file_path,
			    GetLineNumberStr(linenr, linenr_estimated, buffer_idx).c_str(), return_types.size(),
			    options.ToString());
		}
	}

	// insert the line number into the chunk
	idx_t row_entry = parse_chunk.size();

	// test against null string, but only if the value was not quoted
	if ((!(has_quotes && !options.allow_quoted_nulls) || return_types[column].id() != LogicalTypeId::VARCHAR) &&
	    !options.force_not_null[column] && Equals::Operation(str_val, string_t(options.null_str))) {
		FlatVector::SetNull(parse_chunk.data[column], row_entry, true);
	} else {
		auto &v = parse_chunk.data[column];
		auto parse_data = FlatVector::GetData<string_t>(v);
		if (!escape_positions.empty()) {
			// remove escape characters (if any)
			string old_val = str_val.GetString();
			string new_val = "";
			idx_t prev_pos = 0;
			for (idx_t i = 0; i < escape_positions.size(); i++) {
				idx_t next_pos = escape_positions[i];
				new_val += old_val.substr(prev_pos, next_pos - prev_pos);

				if (options.escape.empty() || options.escape == options.quote) {
					prev_pos = next_pos + options.quote.size();
				} else {
					prev_pos = next_pos + options.escape.size();
				}
			}
			new_val += old_val.substr(prev_pos, old_val.size() - prev_pos);
			escape_positions.clear();
			parse_data[row_entry] = StringVector::AddStringOrBlob(v, string_t(new_val));
		} else {
			parse_data[row_entry] = str_val;
		}
	}

	// move to the next column
	column++;
}

bool BaseCSVReader::AddRow(DataChunk &insert_chunk, idx_t &column, string &error_message, idx_t buffer_idx) {
	linenr++;

	if (row_empty) {
		row_empty = false;
		if (return_types.size() != 1) {
			if (mode == ParserMode::PARSING) {
				FlatVector::SetNull(parse_chunk.data[0], parse_chunk.size(), false);
			}
			column = 0;
			return false;
		}
	}

	// Error forwarded by 'ignore_errors' - originally encountered in 'AddValue'
	if (error_column_overflow) {
		D_ASSERT(options.ignore_errors);
		error_column_overflow = false;
		column = 0;
		return false;
	}

	if (column < return_types.size() && mode != ParserMode::SNIFFING_DIALECT) {
		if (options.null_padding) {
			for (; column < return_types.size(); column++) {
				FlatVector::SetNull(parse_chunk.data[column], parse_chunk.size(), true);
			}
		} else if (options.ignore_errors) {
			column = 0;
			return false;
		} else {
			if (mode == ParserMode::SNIFFING_DATATYPES) {
				error_message = "Error when adding line";
				return false;
			} else {
				throw InvalidInputException(
				    "Error in file \"%s\" on line %s: expected %lld values per row, but got %d.\nParser options:\n%s",
				    options.file_path, GetLineNumberStr(linenr, linenr_estimated, buffer_idx).c_str(),
				    return_types.size(), column, options.ToString());
			}
		}
	}

	if (mode == ParserMode::SNIFFING_DIALECT) {
		sniffed_column_counts.push_back(column);

		if (sniffed_column_counts.size() == options.sample_chunk_size) {
			return true;
		}
	} else {
		parse_chunk.SetCardinality(parse_chunk.size() + 1);
	}

	if (mode == ParserMode::PARSING_HEADER) {
		return true;
	}

	if (mode == ParserMode::SNIFFING_DATATYPES && parse_chunk.size() == options.sample_chunk_size) {
		return true;
	}

	if (mode == ParserMode::PARSING && parse_chunk.size() == STANDARD_VECTOR_SIZE) {
		Flush(insert_chunk, buffer_idx);
		return true;
	}

	column = 0;
	return false;
}

void BaseCSVReader::VerifyUTF8(idx_t col_idx, idx_t row_idx, DataChunk &chunk, int64_t offset) {
	D_ASSERT(col_idx < chunk.data.size());
	D_ASSERT(row_idx < chunk.size());
	auto &v = chunk.data[col_idx];
	if (FlatVector::IsNull(v, row_idx)) {
		return;
	}

	auto parse_data = FlatVector::GetData<string_t>(chunk.data[col_idx]);
	auto s = parse_data[row_idx];
	auto utf_type = Utf8Proc::Analyze(s.GetData(), s.GetSize());
	if (utf_type == UnicodeType::INVALID) {
		string col_name = to_string(col_idx);
		if (col_idx < names.size()) {
			col_name = "\"" + names[col_idx] + "\"";
		}
		int64_t error_line = linenr - (chunk.size() - row_idx) + 1 + offset;
		D_ASSERT(error_line >= 0);
		throw InvalidInputException("Error in file \"%s\" at line %llu in column \"%s\": "
		                            "%s. Parser options:\n%s",
		                            options.file_path, error_line, col_name,
		                            ErrorManager::InvalidUnicodeError(s.GetString(), "CSV file"), options.ToString());
	}
}

void BaseCSVReader::VerifyUTF8(idx_t col_idx) {
	D_ASSERT(col_idx < parse_chunk.data.size());
	for (idx_t i = 0; i < parse_chunk.size(); i++) {
		VerifyUTF8(col_idx, i, parse_chunk);
	}
}

bool TryCastDecimalVectorCommaSeparated(BufferedCSVReaderOptions &options, Vector &input_vector, Vector &result_vector,
                                        idx_t count, string &error_message, const LogicalType &result_type) {
	auto width = DecimalType::GetWidth(result_type);
	auto scale = DecimalType::GetScale(result_type);
	switch (result_type.InternalType()) {
	case PhysicalType::INT16:
		return TemplatedTryCastDecimalVector<TryCastToDecimalCommaSeparated, int16_t>(
		    options, input_vector, result_vector, count, error_message, width, scale);
	case PhysicalType::INT32:
		return TemplatedTryCastDecimalVector<TryCastToDecimalCommaSeparated, int32_t>(
		    options, input_vector, result_vector, count, error_message, width, scale);
	case PhysicalType::INT64:
		return TemplatedTryCastDecimalVector<TryCastToDecimalCommaSeparated, int64_t>(
		    options, input_vector, result_vector, count, error_message, width, scale);
	case PhysicalType::INT128:
		return TemplatedTryCastDecimalVector<TryCastToDecimalCommaSeparated, hugeint_t>(
		    options, input_vector, result_vector, count, error_message, width, scale);
	default:
		throw InternalException("Unimplemented physical type for decimal");
	}
}

bool TryCastFloatingVectorCommaSeparated(BufferedCSVReaderOptions &options, Vector &input_vector, Vector &result_vector,
                                         idx_t count, string &error_message, const LogicalType &result_type,
                                         idx_t &line_error) {
	switch (result_type.InternalType()) {
	case PhysicalType::DOUBLE:
		return TemplatedTryCastFloatingVector<TryCastErrorMessageCommaSeparated, double>(
		    options, input_vector, result_vector, count, error_message, line_error);
	case PhysicalType::FLOAT:
		return TemplatedTryCastFloatingVector<TryCastErrorMessageCommaSeparated, float>(
		    options, input_vector, result_vector, count, error_message, line_error);
	default:
		throw InternalException("Unimplemented physical type for floating");
	}
}

bool BaseCSVReader::Flush(DataChunk &insert_chunk, idx_t buffer_idx, bool try_add_line) {
	if (parse_chunk.size() == 0) {
		return true;
	}

	bool conversion_error_ignored = false;

	// convert the columns in the parsed chunk to the types of the table
	insert_chunk.SetCardinality(parse_chunk);
	if (reader_data.column_ids.empty() && !reader_data.empty_columns) {
		throw InternalException("BaseCSVReader::Flush called on a CSV reader that was not correctly initialized. Call "
		                        "MultiFileReader::InitializeReader or InitializeProjection");
	}
	D_ASSERT(reader_data.column_ids.size() == reader_data.column_mapping.size());
	for (idx_t c = 0; c < reader_data.column_ids.size(); c++) {
		auto col_idx = reader_data.column_ids[c];
		auto result_idx = reader_data.column_mapping[c];
		auto &parse_vector = parse_chunk.data[col_idx];
		auto &result_vector = insert_chunk.data[result_idx];
		auto &type = result_vector.GetType();
		if (type.id() == LogicalTypeId::VARCHAR) {
			// target type is varchar: no need to convert
			// just test that all strings are valid utf-8 strings
			VerifyUTF8(col_idx);
			// reinterpret rather than reference so we can deal with user-defined types
			result_vector.Reinterpret(parse_vector);
		} else {
			string error_message;
			bool success;
			idx_t line_error = 0;
			bool target_type_not_varchar = false;
			if (options.has_format[LogicalTypeId::DATE] && type.id() == LogicalTypeId::DATE) {
				// use the date format to cast the chunk
				success = TryCastDateVector(options, parse_vector, result_vector, parse_chunk.size(), error_message,
				                            line_error);
			} else if (options.has_format[LogicalTypeId::TIMESTAMP] && type.id() == LogicalTypeId::TIMESTAMP) {
				// use the date format to cast the chunk
				success =
				    TryCastTimestampVector(options, parse_vector, result_vector, parse_chunk.size(), error_message);
			} else if (options.decimal_separator != "." &&
			           (type.id() == LogicalTypeId::FLOAT || type.id() == LogicalTypeId::DOUBLE)) {
				success = TryCastFloatingVectorCommaSeparated(options, parse_vector, result_vector, parse_chunk.size(),
				                                              error_message, type, line_error);
			} else if (options.decimal_separator != "." && type.id() == LogicalTypeId::DECIMAL) {
				success = TryCastDecimalVectorCommaSeparated(options, parse_vector, result_vector, parse_chunk.size(),
				                                             error_message, type);
			} else {
				// target type is not varchar: perform a cast
				target_type_not_varchar = true;
				success =
				    VectorOperations::TryCast(context, parse_vector, result_vector, parse_chunk.size(), &error_message);
			}
			if (success) {
				continue;
			}
			if (try_add_line) {
				return false;
			}
			if (options.ignore_errors) {
				conversion_error_ignored = true;
				continue;
			}
			string col_name = to_string(col_idx);
			if (col_idx < names.size()) {
				col_name = "\"" + names[col_idx] + "\"";
			}

			// figure out the exact line number
			if (target_type_not_varchar) {
				UnifiedVectorFormat inserted_column_data;
				result_vector.ToUnifiedFormat(parse_chunk.size(), inserted_column_data);
				for (; line_error < parse_chunk.size(); line_error++) {
					if (!inserted_column_data.validity.RowIsValid(line_error) &&
					    !FlatVector::IsNull(parse_vector, line_error)) {
						break;
					}
				}
			}

			idx_t error_line;
			// The line_error must be summed with linenr (All lines emmited from this batch)
			// But subtracted from the parse_chunk
			D_ASSERT(line_error + linenr >= parse_chunk.size());
			line_error += linenr;
			line_error -= parse_chunk.size();

			error_line = GetLineError(line_error, buffer_idx);

			if (options.auto_detect) {
				throw InvalidInputException("%s in column %s, at line %llu.\n\nParser "
				                            "options:\n%s.\n\nConsider either increasing the sample size "
				                            "(SAMPLE_SIZE=X [X rows] or SAMPLE_SIZE=-1 [all rows]), "
				                            "or skipping column conversion (ALL_VARCHAR=1)",
				                            error_message, col_name, error_line, options.ToString());
			} else {
				throw InvalidInputException("%s at line %llu in column %s. Parser options:\n%s ", error_message,
				                            error_line, col_name, options.ToString());
			}
		}
	}
	if (conversion_error_ignored) {
		D_ASSERT(options.ignore_errors);
		SelectionVector succesful_rows(parse_chunk.size());
		idx_t sel_size = 0;

		for (idx_t row_idx = 0; row_idx < parse_chunk.size(); row_idx++) {
			bool failed = false;
			for (idx_t c = 0; c < reader_data.column_ids.size(); c++) {
				auto col_idx = reader_data.column_ids[c];
				auto result_idx = reader_data.column_mapping[c];

				auto &parse_vector = parse_chunk.data[col_idx];
				auto &result_vector = insert_chunk.data[result_idx];

				bool was_already_null = FlatVector::IsNull(parse_vector, row_idx);
				if (!was_already_null && FlatVector::IsNull(result_vector, row_idx)) {
					failed = true;
					break;
				}
			}
			if (!failed) {
				succesful_rows.set_index(sel_size++, row_idx);
			}
		}
		insert_chunk.Slice(succesful_rows, sel_size);
	}
	parse_chunk.Reset();
	return true;
}

void BaseCSVReader::SetNewLineDelimiter(bool carry, bool carry_followed_by_nl) {
	if ((mode == ParserMode::SNIFFING_DIALECT && !options.has_newline) ||
	    options.new_line == NewLineIdentifier::NOT_SET) {
		if (options.new_line == NewLineIdentifier::MIX) {
			return;
		}
		NewLineIdentifier this_line_identifier;
		if (carry) {
			if (carry_followed_by_nl) {
				this_line_identifier = NewLineIdentifier::CARRY_ON;
			} else {
				this_line_identifier = NewLineIdentifier::SINGLE;
			}
		} else {
			this_line_identifier = NewLineIdentifier::SINGLE;
		}
		if (options.new_line == NewLineIdentifier::NOT_SET) {
			options.new_line = this_line_identifier;
			return;
		}
		if (options.new_line != this_line_identifier) {
			options.new_line = NewLineIdentifier::MIX;
			return;
		}
		options.new_line = this_line_identifier;
	}
}
} // namespace duckdb



















#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>

namespace duckdb {

BufferedCSVReader::BufferedCSVReader(ClientContext &context, BufferedCSVReaderOptions options_p,
                                     const vector<LogicalType> &requested_types)
    : BaseCSVReader(context, std::move(options_p), requested_types), buffer_size(0), position(0), start(0) {
	file_handle = OpenCSV(options);
	Initialize(requested_types);
}

BufferedCSVReader::BufferedCSVReader(ClientContext &context, string filename, BufferedCSVReaderOptions options_p,
                                     const vector<LogicalType> &requested_types)
    : BaseCSVReader(context, std::move(options_p), requested_types), buffer_size(0), position(0), start(0) {
	options.file_path = std::move(filename);
	file_handle = OpenCSV(options);
	Initialize(requested_types);
}

enum class QuoteRule : uint8_t { QUOTES_RFC = 0, QUOTES_OTHER = 1, NO_QUOTES = 2 };

static bool StartsWithNumericDate(string &separator, const string &value) {
	auto begin = value.c_str();
	auto end = begin + value.size();

	//	StrpTimeFormat::Parse will skip whitespace, so we can too
	auto field1 = std::find_if_not(begin, end, StringUtil::CharacterIsSpace);
	if (field1 == end) {
		return false;
	}

	//	first numeric field must start immediately
	if (!StringUtil::CharacterIsDigit(*field1)) {
		return false;
	}
	auto literal1 = std::find_if_not(field1, end, StringUtil::CharacterIsDigit);
	if (literal1 == end) {
		return false;
	}

	//	second numeric field must exist
	auto field2 = std::find_if(literal1, end, StringUtil::CharacterIsDigit);
	if (field2 == end) {
		return false;
	}
	auto literal2 = std::find_if_not(field2, end, StringUtil::CharacterIsDigit);
	if (literal2 == end) {
		return false;
	}

	//	third numeric field must exist
	auto field3 = std::find_if(literal2, end, StringUtil::CharacterIsDigit);
	if (field3 == end) {
		return false;
	}

	//	second literal must match first
	if (((field3 - literal2) != (field2 - literal1)) || strncmp(literal1, literal2, (field2 - literal1)) != 0) {
		return false;
	}

	//	copy the literal as the separator, escaping percent signs
	separator.clear();
	while (literal1 < field2) {
		const auto literal_char = *literal1++;
		if (literal_char == '%') {
			separator.push_back(literal_char);
		}
		separator.push_back(literal_char);
	}

	return true;
}

string GenerateDateFormat(const string &separator, const char *format_template) {
	string format_specifier = format_template;
	auto amount_of_dashes = std::count(format_specifier.begin(), format_specifier.end(), '-');
	if (!amount_of_dashes) {
		return format_specifier;
	}
	string result;
	result.reserve(format_specifier.size() - amount_of_dashes + (amount_of_dashes * separator.size()));
	for (auto &character : format_specifier) {
		if (character == '-') {
			result += separator;
		} else {
			result += character;
		}
	}
	return result;
}

TextSearchShiftArray::TextSearchShiftArray() {
}

TextSearchShiftArray::TextSearchShiftArray(string search_term) : length(search_term.size()) {
	if (length > 255) {
		throw InvalidInputException("Size of delimiter/quote/escape in CSV reader is limited to 255 bytes");
	}
	// initialize the shifts array
	shifts = unique_ptr<uint8_t[]>(new uint8_t[length * 255]);
	memset(shifts.get(), 0, length * 255 * sizeof(uint8_t));
	// iterate over each of the characters in the array
	for (idx_t main_idx = 0; main_idx < length; main_idx++) {
		uint8_t current_char = (uint8_t)search_term[main_idx];
		// now move over all the remaining positions
		for (idx_t i = main_idx; i < length; i++) {
			bool is_match = true;
			// check if the prefix matches at this position
			// if it does, we move to this position after encountering the current character
			for (idx_t j = 0; j < main_idx; j++) {
				if (search_term[i - main_idx + j] != search_term[j]) {
					is_match = false;
				}
			}
			if (!is_match) {
				continue;
			}
			shifts[i * 255 + current_char] = main_idx + 1;
		}
	}
}

// Helper function to generate column names
static string GenerateColumnName(const idx_t total_cols, const idx_t col_number, const string &prefix = "column") {
	int max_digits = NumericHelper::UnsignedLength(total_cols - 1);
	int digits = NumericHelper::UnsignedLength(col_number);
	string leading_zeros = string(max_digits - digits, '0');
	string value = to_string(col_number);
	return string(prefix + leading_zeros + value);
}

// Helper function for UTF-8 aware space trimming
static string TrimWhitespace(const string &col_name) {
	utf8proc_int32_t codepoint;
	auto str = reinterpret_cast<const utf8proc_uint8_t *>(col_name.c_str());
	idx_t size = col_name.size();
	// Find the first character that is not left trimmed
	idx_t begin = 0;
	while (begin < size) {
		auto bytes = utf8proc_iterate(str + begin, size - begin, &codepoint);
		D_ASSERT(bytes > 0);
		if (utf8proc_category(codepoint) != UTF8PROC_CATEGORY_ZS) {
			break;
		}
		begin += bytes;
	}

	// Find the last character that is not right trimmed
	idx_t end;
	end = begin;
	for (auto next = begin; next < col_name.size();) {
		auto bytes = utf8proc_iterate(str + next, size - next, &codepoint);
		D_ASSERT(bytes > 0);
		next += bytes;
		if (utf8proc_category(codepoint) != UTF8PROC_CATEGORY_ZS) {
			end = next;
		}
	}

	// return the trimmed string
	return col_name.substr(begin, end - begin);
}

static string NormalizeColumnName(const string &col_name) {
	// normalize UTF8 characters to NFKD
	auto nfkd = utf8proc_NFKD((const utf8proc_uint8_t *)col_name.c_str(), col_name.size());
	const string col_name_nfkd = string((const char *)nfkd, strlen((const char *)nfkd));
	free(nfkd);

	// only keep ASCII characters 0-9 a-z A-Z and replace spaces with regular whitespace
	string col_name_ascii = "";
	for (idx_t i = 0; i < col_name_nfkd.size(); i++) {
		if (col_name_nfkd[i] == '_' || (col_name_nfkd[i] >= '0' && col_name_nfkd[i] <= '9') ||
		    (col_name_nfkd[i] >= 'A' && col_name_nfkd[i] <= 'Z') ||
		    (col_name_nfkd[i] >= 'a' && col_name_nfkd[i] <= 'z')) {
			col_name_ascii += col_name_nfkd[i];
		} else if (StringUtil::CharacterIsSpace(col_name_nfkd[i])) {
			col_name_ascii += " ";
		}
	}

	// trim whitespace and replace remaining whitespace by _
	string col_name_trimmed = TrimWhitespace(col_name_ascii);
	string col_name_cleaned = "";
	bool in_whitespace = false;
	for (idx_t i = 0; i < col_name_trimmed.size(); i++) {
		if (col_name_trimmed[i] == ' ') {
			if (!in_whitespace) {
				col_name_cleaned += "_";
				in_whitespace = true;
			}
		} else {
			col_name_cleaned += col_name_trimmed[i];
			in_whitespace = false;
		}
	}

	// don't leave string empty; if not empty, make lowercase
	if (col_name_cleaned.empty()) {
		col_name_cleaned = "_";
	} else {
		col_name_cleaned = StringUtil::Lower(col_name_cleaned);
	}

	// prepend _ if name starts with a digit or is a reserved keyword
	if (KeywordHelper::IsKeyword(col_name_cleaned) || (col_name_cleaned[0] >= '0' && col_name_cleaned[0] <= '9')) {
		col_name_cleaned = "_" + col_name_cleaned;
	}
	return col_name_cleaned;
}

void BufferedCSVReader::Initialize(const vector<LogicalType> &requested_types) {
	PrepareComplexParser();
	if (options.auto_detect) {
		return_types = SniffCSV(requested_types);
		if (return_types.empty()) {
			throw InvalidInputException("Failed to detect column types from CSV: is the file a valid CSV file?");
		}
		JumpToBeginning(options.skip_rows, options.header);
	} else {
		return_types = requested_types;
		ResetBuffer();
		SkipRowsAndReadHeader(options.skip_rows, options.header);
	}
	InitParseChunk(return_types.size());
}

void BufferedCSVReader::ResetBuffer() {
	buffer.reset();
	buffer_size = 0;
	position = 0;
	start = 0;
	cached_buffers.clear();
}

void BufferedCSVReader::ResetStream() {
	file_handle->Reset();
	linenr = 0;
	linenr_estimated = false;
	bytes_per_line_avg = 0;
	sample_chunk_idx = 0;
	jumping_samples = false;
}

void BufferedCSVReader::JumpToBeginning(idx_t skip_rows = 0, bool skip_header = false) {
	ResetBuffer();
	ResetStream();
	sample_chunk_idx = 0;
	bytes_in_chunk = 0;
	end_of_file_reached = false;
	bom_checked = false;
	SkipRowsAndReadHeader(skip_rows, skip_header);
}

void BufferedCSVReader::SkipRowsAndReadHeader(idx_t skip_rows, bool skip_header) {
	for (idx_t i = 0; i < skip_rows; i++) {
		// ignore skip rows
		string read_line = file_handle->ReadLine();
		linenr++;
	}

	if (skip_header) {
		// ignore the first line as a header line
		InitParseChunk(return_types.size());
		ParseCSV(ParserMode::PARSING_HEADER);
	}
}

void BufferedCSVReader::PrepareComplexParser() {
	delimiter_search = TextSearchShiftArray(options.delimiter);
	escape_search = TextSearchShiftArray(options.escape);
	quote_search = TextSearchShiftArray(options.quote);
}

bool BufferedCSVReader::JumpToNextSample() {
	// get bytes contained in the previously read chunk
	idx_t remaining_bytes_in_buffer = buffer_size - start;
	bytes_in_chunk -= remaining_bytes_in_buffer;
	if (remaining_bytes_in_buffer == 0) {
		return false;
	}

	// assess if it makes sense to jump, based on size of the first chunk relative to size of the entire file
	if (sample_chunk_idx == 0) {
		idx_t bytes_first_chunk = bytes_in_chunk;
		double chunks_fit = (file_handle->FileSize() / (double)bytes_first_chunk);
		jumping_samples = chunks_fit >= options.sample_chunks;

		// jump back to the beginning
		JumpToBeginning(options.skip_rows, options.header);
		sample_chunk_idx++;
		return true;
	}

	if (end_of_file_reached || sample_chunk_idx >= options.sample_chunks) {
		return false;
	}

	// if we deal with any other sources than plaintext files, jumping_samples can be tricky. In that case
	// we just read x continuous chunks from the stream TODO: make jumps possible for zipfiles.
	if (!file_handle->OnDiskFile() || !jumping_samples) {
		sample_chunk_idx++;
		return true;
	}

	// update average bytes per line
	double bytes_per_line = bytes_in_chunk / (double)options.sample_chunk_size;
	bytes_per_line_avg = ((bytes_per_line_avg * (sample_chunk_idx)) + bytes_per_line) / (sample_chunk_idx + 1);

	// if none of the previous conditions were met, we can jump
	idx_t partition_size = (idx_t)round(file_handle->FileSize() / (double)options.sample_chunks);

	// calculate offset to end of the current partition
	int64_t offset = partition_size - bytes_in_chunk - remaining_bytes_in_buffer;
	auto current_pos = file_handle->SeekPosition();

	if (current_pos + offset < file_handle->FileSize()) {
		// set position in stream and clear failure bits
		file_handle->Seek(current_pos + offset);

		// estimate linenr
		linenr += (idx_t)round((offset + remaining_bytes_in_buffer) / bytes_per_line_avg);
		linenr_estimated = true;
	} else {
		// seek backwards from the end in last chunk and hope to catch the end of the file
		// TODO: actually it would be good to make sure that the end of file is being reached, because
		// messy end-lines are quite common. For this case, however, we first need a skip_end detection anyways.
		file_handle->Seek(file_handle->FileSize() - bytes_in_chunk);

		// estimate linenr
		linenr = (idx_t)round((file_handle->FileSize() - bytes_in_chunk) / bytes_per_line_avg);
		linenr_estimated = true;
	}

	// reset buffers and parse chunk
	ResetBuffer();

	// seek beginning of next line
	// FIXME: if this jump ends up in a quoted linebreak, we will have a problem
	string read_line = file_handle->ReadLine();
	linenr++;

	sample_chunk_idx++;

	return true;
}

void BufferedCSVReader::DetectDialect(const vector<LogicalType> &requested_types,
                                      BufferedCSVReaderOptions &original_options,
                                      vector<BufferedCSVReaderOptions> &info_candidates, idx_t &best_num_cols) {
	// set up the candidates we consider for delimiter and quote rules based on user input
	vector<string> delim_candidates;
	vector<QuoteRule> quoterule_candidates;
	vector<vector<string>> quote_candidates_map;
	vector<vector<string>> escape_candidates_map = {{""}, {"\\"}, {""}};

	if (options.has_delimiter) {
		// user provided a delimiter: use that delimiter
		delim_candidates = {options.delimiter};
	} else {
		// no delimiter provided: try standard/common delimiters
		delim_candidates = {",", "|", ";", "\t"};
	}
	if (options.has_quote) {
		// user provided quote: use that quote rule
		quote_candidates_map = {{options.quote}, {options.quote}, {options.quote}};
	} else {
		// no quote rule provided: use standard/common quotes
		quote_candidates_map = {{"\""}, {"\"", "'"}, {""}};
	}
	if (options.has_escape) {
		// user provided escape: use that escape rule
		if (options.escape.empty()) {
			quoterule_candidates = {QuoteRule::QUOTES_RFC};
		} else {
			quoterule_candidates = {QuoteRule::QUOTES_OTHER};
		}
		escape_candidates_map[static_cast<uint8_t>(quoterule_candidates[0])] = {options.escape};
	} else {
		// no escape provided: try standard/common escapes
		quoterule_candidates = {QuoteRule::QUOTES_RFC, QuoteRule::QUOTES_OTHER, QuoteRule::NO_QUOTES};
	}

	idx_t best_consistent_rows = 0;
	idx_t prev_padding_count = 0;
	for (auto quoterule : quoterule_candidates) {
		const auto &quote_candidates = quote_candidates_map[static_cast<uint8_t>(quoterule)];
		for (const auto &quote : quote_candidates) {
			for (const auto &delim : delim_candidates) {
				const auto &escape_candidates = escape_candidates_map[static_cast<uint8_t>(quoterule)];
				for (const auto &escape : escape_candidates) {
					BufferedCSVReaderOptions sniff_info = original_options;
					sniff_info.delimiter = delim;
					sniff_info.quote = quote;
					sniff_info.escape = escape;

					options = sniff_info;
					PrepareComplexParser();

					JumpToBeginning(original_options.skip_rows);
					sniffed_column_counts.clear();
					if (!TryParseCSV(ParserMode::SNIFFING_DIALECT)) {
						continue;
					}

					idx_t start_row = original_options.skip_rows;
					idx_t consistent_rows = 0;
					idx_t num_cols = sniffed_column_counts.empty() ? 0 : sniffed_column_counts[0];
					idx_t padding_count = 0;
					bool allow_padding = original_options.null_padding;
					for (idx_t row = 0; row < sniffed_column_counts.size(); row++) {
						if (sniffed_column_counts[row] == num_cols) {
							consistent_rows++;
						} else if (num_cols < sniffed_column_counts[row] && !original_options.skip_rows_set) {
							// we use the maximum amount of num_cols that we find
							num_cols = sniffed_column_counts[row];
							start_row = row + original_options.skip_rows;
							consistent_rows = 1;
							padding_count = 0;
						} else if (num_cols >= sniffed_column_counts[row] && allow_padding) {
							// we are missing some columns, we can parse this as long as we add padding
							padding_count++;
						}
					}

					// some logic
					consistent_rows += padding_count;
					bool more_values = (consistent_rows > best_consistent_rows && num_cols >= best_num_cols);
					bool require_more_padding = padding_count > prev_padding_count;
					bool require_less_padding = padding_count < prev_padding_count;
					bool single_column_before = best_num_cols < 2 && num_cols > best_num_cols;
					bool rows_consistent =
					    start_row + consistent_rows - original_options.skip_rows == sniffed_column_counts.size();
					bool more_than_one_row = (consistent_rows > 1);
					bool more_than_one_column = (num_cols > 1);
					bool start_good = !info_candidates.empty() && (start_row <= info_candidates.front().skip_rows);

					if (!requested_types.empty() && requested_types.size() != num_cols) {
						continue;
					} else if (rows_consistent && (single_column_before || (more_values && !require_more_padding) ||
					                               (more_than_one_column && require_less_padding))) {
						sniff_info.skip_rows = start_row;
						sniff_info.num_cols = num_cols;
						sniff_info.new_line = options.new_line;
						best_consistent_rows = consistent_rows;
						best_num_cols = num_cols;
						prev_padding_count = padding_count;

						info_candidates.clear();
						info_candidates.push_back(sniff_info);
					} else if (more_than_one_row && more_than_one_column && start_good && rows_consistent &&
					           !require_more_padding) {
						bool same_quote_is_candidate = false;
						for (auto &info_candidate : info_candidates) {
							if (quote.compare(info_candidate.quote) == 0) {
								same_quote_is_candidate = true;
							}
						}
						if (!same_quote_is_candidate) {
							sniff_info.skip_rows = start_row;
							sniff_info.num_cols = num_cols;
							sniff_info.new_line = options.new_line;
							info_candidates.push_back(sniff_info);
						}
					}
				}
			}
		}
	}
}

void BufferedCSVReader::DetectCandidateTypes(const vector<LogicalType> &type_candidates,
                                             const map<LogicalTypeId, vector<const char *>> &format_template_candidates,
                                             const vector<BufferedCSVReaderOptions> &info_candidates,
                                             BufferedCSVReaderOptions &original_options, idx_t best_num_cols,
                                             vector<vector<LogicalType>> &best_sql_types_candidates,
                                             std::map<LogicalTypeId, vector<string>> &best_format_candidates,
                                             DataChunk &best_header_row) {
	BufferedCSVReaderOptions best_options;
	idx_t min_varchar_cols = best_num_cols + 1;

	// check which info candidate leads to minimum amount of non-varchar columns...
	for (const auto &t : format_template_candidates) {
		best_format_candidates[t.first].clear();
	}
	for (auto &info_candidate : info_candidates) {
		options = info_candidate;
		vector<vector<LogicalType>> info_sql_types_candidates(options.num_cols, type_candidates);
		std::map<LogicalTypeId, bool> has_format_candidates;
		std::map<LogicalTypeId, vector<string>> format_candidates;
		for (const auto &t : format_template_candidates) {
			has_format_candidates[t.first] = false;
			format_candidates[t.first].clear();
		}

		// set all return_types to VARCHAR so we can do datatype detection based on VARCHAR values
		return_types.clear();
		return_types.assign(options.num_cols, LogicalType::VARCHAR);

		// jump to beginning and skip potential header
		JumpToBeginning(options.skip_rows, true);
		DataChunk header_row;
		header_row.Initialize(allocator, return_types);
		parse_chunk.Copy(header_row);

		if (header_row.size() == 0) {
			continue;
		}

		// init parse chunk and read csv with info candidate
		InitParseChunk(return_types.size());
		if (!TryParseCSV(ParserMode::SNIFFING_DATATYPES)) {
			continue;
		}
		for (idx_t row_idx = 0; row_idx <= parse_chunk.size(); row_idx++) {
			bool is_header_row = row_idx == 0;
			idx_t row = row_idx - 1;
			for (idx_t col = 0; col < parse_chunk.ColumnCount(); col++) {
				auto &col_type_candidates = info_sql_types_candidates[col];
				while (col_type_candidates.size() > 1) {
					const auto &sql_type = col_type_candidates.back();
					// try cast from string to sql_type
					Value dummy_val;
					if (is_header_row) {
						VerifyUTF8(col, 0, header_row, -int64_t(parse_chunk.size()));
						dummy_val = header_row.GetValue(col, 0);
					} else {
						VerifyUTF8(col, row, parse_chunk);
						dummy_val = parse_chunk.GetValue(col, row);
					}
					// try formatting for date types if the user did not specify one and it starts with numeric values.
					string separator;
					if (has_format_candidates.count(sql_type.id()) && !original_options.has_format[sql_type.id()] &&
					    !dummy_val.IsNull() && StartsWithNumericDate(separator, StringValue::Get(dummy_val))) {
						// generate date format candidates the first time through
						auto &type_format_candidates = format_candidates[sql_type.id()];
						const auto had_format_candidates = has_format_candidates[sql_type.id()];
						if (!has_format_candidates[sql_type.id()]) {
							has_format_candidates[sql_type.id()] = true;
							// order by preference
							auto entry = format_template_candidates.find(sql_type.id());
							if (entry != format_template_candidates.end()) {
								const auto &format_template_list = entry->second;
								for (const auto &t : format_template_list) {
									const auto format_string = GenerateDateFormat(separator, t);
									// don't parse ISO 8601
									if (format_string.find("%Y-%m-%d") == string::npos) {
										type_format_candidates.emplace_back(format_string);
									}
								}
							}
							//	initialise the first candidate
							options.has_format[sql_type.id()] = true;
							//	all formats are constructed to be valid
							SetDateFormat(type_format_candidates.back(), sql_type.id());
						}
						// check all formats and keep the first one that works
						StrpTimeFormat::ParseResult result;
						auto save_format_candidates = type_format_candidates;
						while (!type_format_candidates.empty()) {
							//	avoid using exceptions for flow control...
							auto &current_format = options.date_format[sql_type.id()];
							if (current_format.Parse(StringValue::Get(dummy_val), result)) {
								break;
							}
							//	doesn't work - move to the next one
							type_format_candidates.pop_back();
							options.has_format[sql_type.id()] = (!type_format_candidates.empty());
							if (!type_format_candidates.empty()) {
								SetDateFormat(type_format_candidates.back(), sql_type.id());
							}
						}
						//	if none match, then this is not a value of type sql_type,
						if (type_format_candidates.empty()) {
							//	so restore the candidates that did work.
							//	or throw them out if they were generated by this value.
							if (had_format_candidates) {
								type_format_candidates.swap(save_format_candidates);
								if (!type_format_candidates.empty()) {
									SetDateFormat(type_format_candidates.back(), sql_type.id());
								}
							} else {
								has_format_candidates[sql_type.id()] = false;
							}
						}
					}
					// try cast from string to sql_type
					if (TryCastValue(dummy_val, sql_type)) {
						break;
					} else {
						col_type_candidates.pop_back();
					}
				}
			}
			// reset type detection, because first row could be header,
			// but only do it if csv has more than one line (including header)
			if (parse_chunk.size() > 0 && is_header_row) {
				info_sql_types_candidates = vector<vector<LogicalType>>(options.num_cols, type_candidates);
				for (auto &f : format_candidates) {
					f.second.clear();
				}
				for (auto &h : has_format_candidates) {
					h.second = false;
				}
			}
		}

		idx_t varchar_cols = 0;
		for (idx_t col = 0; col < parse_chunk.ColumnCount(); col++) {
			auto &col_type_candidates = info_sql_types_candidates[col];
			// check number of varchar columns
			const auto &col_type = col_type_candidates.back();
			if (col_type == LogicalType::VARCHAR) {
				varchar_cols++;
			}
		}

		// it's good if the dialect creates more non-varchar columns, but only if we sacrifice < 30% of best_num_cols.
		if (varchar_cols < min_varchar_cols && parse_chunk.ColumnCount() > (best_num_cols * 0.7)) {
			// we have a new best_options candidate
			best_options = info_candidate;
			min_varchar_cols = varchar_cols;
			best_sql_types_candidates = info_sql_types_candidates;
			best_format_candidates = format_candidates;
			best_header_row.Destroy();
			auto header_row_types = header_row.GetTypes();
			best_header_row.Initialize(allocator, header_row_types);
			header_row.Copy(best_header_row);
		}
	}

	options = best_options;
	for (const auto &best : best_format_candidates) {
		if (!best.second.empty()) {
			SetDateFormat(best.second.back(), best.first);
		}
	}
}

void BufferedCSVReader::DetectHeader(const vector<vector<LogicalType>> &best_sql_types_candidates,
                                     const DataChunk &best_header_row) {
	// information for header detection
	bool first_row_consistent = true;
	bool first_row_nulls = false;

	// check if header row is all null and/or consistent with detected column data types
	first_row_nulls = true;
	for (idx_t col = 0; col < best_sql_types_candidates.size(); col++) {
		auto dummy_val = best_header_row.GetValue(col, 0);
		if (!dummy_val.IsNull()) {
			first_row_nulls = false;
		}

		// try cast to sql_type of column
		const auto &sql_type = best_sql_types_candidates[col].back();
		if (!TryCastValue(dummy_val, sql_type)) {
			first_row_consistent = false;
		}
	}

	// update parser info, and read, generate & set col_names based on previous findings
	if (((!first_row_consistent || first_row_nulls) && !options.has_header) || (options.has_header && options.header)) {
		options.header = true;
		case_insensitive_map_t<idx_t> name_collision_count;
		// get header names from CSV
		for (idx_t col = 0; col < options.num_cols; col++) {
			const auto &val = best_header_row.GetValue(col, 0);
			string col_name = val.ToString();

			// generate name if field is empty
			if (col_name.empty() || val.IsNull()) {
				col_name = GenerateColumnName(options.num_cols, col);
			}

			// normalize names or at least trim whitespace
			if (options.normalize_names) {
				col_name = NormalizeColumnName(col_name);
			} else {
				col_name = TrimWhitespace(col_name);
			}

			// avoid duplicate header names
			const string col_name_raw = col_name;
			while (name_collision_count.find(col_name) != name_collision_count.end()) {
				name_collision_count[col_name] += 1;
				col_name = col_name + "_" + to_string(name_collision_count[col_name]);
			}

			names.push_back(col_name);
			name_collision_count[col_name] = 0;
		}

	} else {
		options.header = false;
		for (idx_t col = 0; col < options.num_cols; col++) {
			string column_name = GenerateColumnName(options.num_cols, col);
			names.push_back(column_name);
		}
	}
	for (idx_t i = 0; i < MinValue<idx_t>(names.size(), options.name_list.size()); i++) {
		names[i] = options.name_list[i];
	}
}

vector<LogicalType> BufferedCSVReader::RefineTypeDetection(const vector<LogicalType> &type_candidates,
                                                           const vector<LogicalType> &requested_types,
                                                           vector<vector<LogicalType>> &best_sql_types_candidates,
                                                           map<LogicalTypeId, vector<string>> &best_format_candidates) {
	// for the type refine we set the SQL types to VARCHAR for all columns
	return_types.clear();
	return_types.assign(options.num_cols, LogicalType::VARCHAR);

	vector<LogicalType> detected_types;

	// if data types were provided, exit here if number of columns does not match
	if (!requested_types.empty()) {
		if (requested_types.size() != options.num_cols) {
			throw InvalidInputException(
			    "Error while determining column types: found %lld columns but expected %d. (%s)", options.num_cols,
			    requested_types.size(), options.ToString());
		} else {
			detected_types = requested_types;
		}
	} else if (options.all_varchar) {
		// return all types varchar
		detected_types = return_types;
	} else {
		// jump through the rest of the file and continue to refine the sql type guess
		while (JumpToNextSample()) {
			InitParseChunk(return_types.size());
			// if jump ends up a bad line, we just skip this chunk
			if (!TryParseCSV(ParserMode::SNIFFING_DATATYPES)) {
				continue;
			}
			for (idx_t col = 0; col < parse_chunk.ColumnCount(); col++) {
				vector<LogicalType> &col_type_candidates = best_sql_types_candidates[col];
				while (col_type_candidates.size() > 1) {
					const auto &sql_type = col_type_candidates.back();
					//	narrow down the date formats
					if (best_format_candidates.count(sql_type.id())) {
						auto &best_type_format_candidates = best_format_candidates[sql_type.id()];
						auto save_format_candidates = best_type_format_candidates;
						while (!best_type_format_candidates.empty()) {
							if (TryCastVector(parse_chunk.data[col], parse_chunk.size(), sql_type)) {
								break;
							}
							//	doesn't work - move to the next one
							best_type_format_candidates.pop_back();
							options.has_format[sql_type.id()] = (!best_type_format_candidates.empty());
							if (!best_type_format_candidates.empty()) {
								SetDateFormat(best_type_format_candidates.back(), sql_type.id());
							}
						}
						//	if none match, then this is not a column of type sql_type,
						if (best_type_format_candidates.empty()) {
							//	so restore the candidates that did work.
							best_type_format_candidates.swap(save_format_candidates);
							if (!best_type_format_candidates.empty()) {
								SetDateFormat(best_type_format_candidates.back(), sql_type.id());
							}
						}
					}

					if (TryCastVector(parse_chunk.data[col], parse_chunk.size(), sql_type)) {
						break;
					} else {
						col_type_candidates.pop_back();
					}
				}
			}
		}

		// set sql types
		for (auto &best_sql_types_candidate : best_sql_types_candidates) {
			LogicalType d_type = best_sql_types_candidate.back();
			if (best_sql_types_candidate.size() == type_candidates.size()) {
				d_type = LogicalType::VARCHAR;
			}
			detected_types.push_back(d_type);
		}
	}

	return detected_types;
}

string BufferedCSVReader::ColumnTypesError(case_insensitive_map_t<idx_t> sql_types_per_column,
                                           const vector<string> &names) {
	for (idx_t i = 0; i < names.size(); i++) {
		auto it = sql_types_per_column.find(names[i]);
		if (it != sql_types_per_column.end()) {
			sql_types_per_column.erase(names[i]);
			continue;
		}
	}
	if (sql_types_per_column.empty()) {
		return string();
	}
	string exception = "COLUMN_TYPES error: Columns with names: ";
	for (auto &col : sql_types_per_column) {
		exception += "\"" + col.first + "\",";
	}
	exception.pop_back();
	exception += " do not exist in the CSV File";
	return exception;
}

vector<LogicalType> BufferedCSVReader::SniffCSV(const vector<LogicalType> &requested_types) {
	for (auto &type : requested_types) {
		// auto detect for blobs not supported: there may be invalid UTF-8 in the file
		if (type.id() == LogicalTypeId::BLOB) {
			return requested_types;
		}
	}

	// #######
	// ### dialect detection
	// #######
	BufferedCSVReaderOptions original_options = options;
	vector<BufferedCSVReaderOptions> info_candidates;
	idx_t best_num_cols = 0;

	DetectDialect(requested_types, original_options, info_candidates, best_num_cols);

	// if no dialect candidate was found, then file was most likely empty and we throw an exception
	if (info_candidates.empty()) {
		throw InvalidInputException(
		    "Error in file \"%s\": CSV options could not be auto-detected. Consider setting parser options manually.",
		    options.file_path);
	}

	// #######
	// ### type detection (initial)
	// #######

	// format template candidates, ordered by descending specificity (~ from high to low)
	std::map<LogicalTypeId, vector<const char *>> format_template_candidates = {
	    {LogicalTypeId::DATE, {"%m-%d-%Y", "%m-%d-%y", "%d-%m-%Y", "%d-%m-%y", "%Y-%m-%d", "%y-%m-%d"}},
	    {LogicalTypeId::TIMESTAMP,
	     {"%Y-%m-%d %H:%M:%S.%f", "%m-%d-%Y %I:%M:%S %p", "%m-%d-%y %I:%M:%S %p", "%d-%m-%Y %H:%M:%S",
	      "%d-%m-%y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%y-%m-%d %H:%M:%S"}},
	};
	vector<vector<LogicalType>> best_sql_types_candidates;
	map<LogicalTypeId, vector<string>> best_format_candidates;
	DataChunk best_header_row;
	DetectCandidateTypes(options.auto_type_candidates, format_template_candidates, info_candidates, original_options,
	                     best_num_cols, best_sql_types_candidates, best_format_candidates, best_header_row);

	if (best_format_candidates.empty() || best_header_row.size() == 0) {
		throw InvalidInputException(
		    "Error in file \"%s\": CSV options could not be auto-detected. Consider setting parser options manually.",
		    original_options.file_path);
	}

	// #######
	// ### header detection
	// #######
	options.num_cols = best_num_cols;
	DetectHeader(best_sql_types_candidates, best_header_row);
	if (!options.sql_type_list.empty()) {
		// user-defined types were supplied for certain columns
		// override the types
		if (!options.sql_types_per_column.empty()) {
			// types supplied as name -> value map
			idx_t found = 0;
			for (idx_t i = 0; i < names.size(); i++) {
				auto it = options.sql_types_per_column.find(names[i]);
				if (it != options.sql_types_per_column.end()) {
					best_sql_types_candidates[i] = {options.sql_type_list[it->second]};
					found++;
					continue;
				}
			}
			if (!options.file_options.union_by_name && found < options.sql_types_per_column.size()) {
				string exception = ColumnTypesError(options.sql_types_per_column, names);
				if (!exception.empty()) {
					throw BinderException(exception);
				}
			}
		} else {
			// types supplied as list
			if (names.size() < options.sql_type_list.size()) {
				throw BinderException("read_csv: %d types were provided, but CSV file only has %d columns",
				                      options.sql_type_list.size(), names.size());
			}
			for (idx_t i = 0; i < options.sql_type_list.size(); i++) {
				best_sql_types_candidates[i] = {options.sql_type_list[i]};
			}
		}
	}

	// #######
	// ### type detection (refining)
	// #######
	return RefineTypeDetection(options.auto_type_candidates, requested_types, best_sql_types_candidates,
	                           best_format_candidates);
}

bool BufferedCSVReader::TryParseComplexCSV(DataChunk &insert_chunk, string &error_message) {
	// used for parsing algorithm
	bool finished_chunk = false;
	idx_t column = 0;
	vector<idx_t> escape_positions;
	bool has_quotes = false;
	uint8_t delimiter_pos = 0, escape_pos = 0, quote_pos = 0;
	idx_t offset = 0;
	idx_t line_start = 0;
	// read values into the buffer (if any)
	if (position >= buffer_size) {
		if (!ReadBuffer(start, line_start)) {
			return true;
		}
	}
	// start parsing the first value
	start = position;
	goto value_start;
value_start:
	/* state: value_start */
	// this state parses the first characters of a value
	offset = 0;
	delimiter_pos = 0;
	quote_pos = 0;
	do {
		idx_t count = 0;
		for (; position < buffer_size; position++) {
			quote_search.Match(quote_pos, buffer[position]);
			delimiter_search.Match(delimiter_pos, buffer[position]);
			count++;
			if (delimiter_pos == options.delimiter.size()) {
				// found a delimiter, add the value
				offset = options.delimiter.size() - 1;
				goto add_value;
			} else if (StringUtil::CharacterIsNewline(buffer[position])) {
				// found a newline, add the row
				goto add_row;
			}
			if (count > quote_pos) {
				// did not find a quote directly at the start of the value, stop looking for the quote now
				goto normal;
			}
			if (quote_pos == options.quote.size()) {
				// found a quote, go to quoted loop and skip the initial quote
				start += options.quote.size();
				goto in_quotes;
			}
		}
	} while (ReadBuffer(start, line_start));
	// file ends while scanning for quote/delimiter, go to final state
	goto final_state;
normal:
	/* state: normal parsing state */
	// this state parses the remainder of a non-quoted value until we reach a delimiter or newline
	position++;
	do {
		for (; position < buffer_size; position++) {
			delimiter_search.Match(delimiter_pos, buffer[position]);
			if (delimiter_pos == options.delimiter.size()) {
				offset = options.delimiter.size() - 1;
				goto add_value;
			} else if (StringUtil::CharacterIsNewline(buffer[position])) {
				goto add_row;
			}
		}
	} while (ReadBuffer(start, line_start));
	goto final_state;
add_value:
	AddValue(string_t(buffer.get() + start, position - start - offset), column, escape_positions, has_quotes);
	// increase position by 1 and move start to the new position
	offset = 0;
	has_quotes = false;
	start = ++position;
	if (position >= buffer_size && !ReadBuffer(start, line_start)) {
		// file ends right after delimiter, go to final state
		goto final_state;
	}
	goto value_start;
add_row : {
	// check type of newline (\r or \n)
	bool carriage_return = buffer[position] == '\r';
	AddValue(string_t(buffer.get() + start, position - start - offset), column, escape_positions, has_quotes);
	finished_chunk = AddRow(insert_chunk, column, error_message);

	if (!error_message.empty()) {
		return false;
	}
	// increase position by 1 and move start to the new position
	offset = 0;
	has_quotes = false;
	position++;
	SkipEmptyLines();
	start = position;
	if (position >= buffer_size && !ReadBuffer(start, line_start)) {
		// file ends right after newline, go to final state
		goto final_state;
	}
	if (carriage_return) {
		// \r newline, go to special state that parses an optional \n afterwards
		goto carriage_return;
	} else {
		// \n newline, move to value start
		if (finished_chunk) {
			return true;
		}
		goto value_start;
	}
}
in_quotes:
	/* state: in_quotes */
	// this state parses the remainder of a quoted value
	quote_pos = 0;
	escape_pos = 0;
	has_quotes = true;
	position++;
	do {
		for (; position < buffer_size; position++) {
			quote_search.Match(quote_pos, buffer[position]);
			escape_search.Match(escape_pos, buffer[position]);
			if (quote_pos == options.quote.size()) {
				goto unquote;
			} else if (escape_pos == options.escape.size()) {
				escape_positions.push_back(position - start - (options.escape.size() - 1));
				goto handle_escape;
			}
		}
	} while (ReadBuffer(start, line_start));
	// still in quoted state at the end of the file, error:
	error_message = StringUtil::Format("Error in file \"%s\" on line %s: unterminated quotes. (%s)", options.file_path,
	                                   GetLineNumberStr(linenr, linenr_estimated).c_str(), options.ToString());
	return false;
unquote:
	/* state: unquote */
	// this state handles the state directly after we unquote
	// in this state we expect either another quote (entering the quoted state again, and escaping the quote)
	// or a delimiter/newline, ending the current value and moving on to the next value
	delimiter_pos = 0;
	quote_pos = 0;
	position++;
	if (position >= buffer_size && !ReadBuffer(start, line_start)) {
		// file ends right after unquote, go to final state
		offset = options.quote.size();
		goto final_state;
	}
	if (StringUtil::CharacterIsNewline(buffer[position])) {
		// quote followed by newline, add row
		offset = options.quote.size();
		goto add_row;
	}
	do {
		idx_t count = 0;
		for (; position < buffer_size; position++) {
			quote_search.Match(quote_pos, buffer[position]);
			delimiter_search.Match(delimiter_pos, buffer[position]);
			count++;
			if (count > delimiter_pos && count > quote_pos) {
				error_message = StringUtil::Format(
				    "Error in file \"%s\" on line %s: quote should be followed by end of value, end "
				    "of row or another quote. (%s)",
				    options.file_path, GetLineNumberStr(linenr, linenr_estimated).c_str(), options.ToString());
				return false;
			}
			if (delimiter_pos == options.delimiter.size()) {
				// quote followed by delimiter, add value
				offset = options.quote.size() + options.delimiter.size() - 1;
				goto add_value;
			} else if (quote_pos == options.quote.size() &&
			           (options.escape.empty() || options.escape == options.quote)) {
				// quote followed by quote, go back to quoted state and add to escape
				escape_positions.push_back(position - start - (options.quote.size() - 1));
				goto in_quotes;
			}
		}
	} while (ReadBuffer(start, line_start));
	error_message = StringUtil::Format(
	    "Error in file \"%s\" on line %s: quote should be followed by end of value, end of row or another quote. (%s)",
	    options.file_path, GetLineNumberStr(linenr, linenr_estimated).c_str(), options.ToString());
	return false;
handle_escape:
	escape_pos = 0;
	quote_pos = 0;
	position++;
	do {
		idx_t count = 0;
		for (; position < buffer_size; position++) {
			quote_search.Match(quote_pos, buffer[position]);
			escape_search.Match(escape_pos, buffer[position]);
			count++;
			if (count > escape_pos && count > quote_pos) {
				error_message = StringUtil::Format(
				    "Error in file \"%s\" on line %s: neither QUOTE nor ESCAPE is proceeded by ESCAPE. (%s)",
				    options.file_path, GetLineNumberStr(linenr, linenr_estimated).c_str(), options.ToString());
				return false;
			}
			if (quote_pos == options.quote.size() || escape_pos == options.escape.size()) {
				// found quote or escape: move back to quoted state
				goto in_quotes;
			}
		}
	} while (ReadBuffer(start, line_start));
	error_message =
	    StringUtil::Format("Error in file \"%s\" on line %s: neither QUOTE nor ESCAPE is proceeded by ESCAPE. (%s)",
	                       options.file_path, GetLineNumberStr(linenr, linenr_estimated).c_str(), options.ToString());
	return false;
carriage_return:
	/* state: carriage_return */
	// this stage optionally skips a newline (\n) character, which allows \r\n to be interpreted as a single line
	if (buffer[position] == '\n') {
		// newline after carriage return: skip
		start = ++position;
		if (position >= buffer_size && !ReadBuffer(start, line_start)) {
			// file ends right after newline, go to final state
			goto final_state;
		}
	}
	if (finished_chunk) {
		return true;
	}
	goto value_start;
final_state:
	if (finished_chunk) {
		return true;
	}
	if (column > 0 || position > start) {
		// remaining values to be added to the chunk
		AddValue(string_t(buffer.get() + start, position - start - offset), column, escape_positions, has_quotes);
		finished_chunk = AddRow(insert_chunk, column, error_message);
		SkipEmptyLines();
		if (!error_message.empty()) {
			return false;
		}
	}
	// final stage, only reached after parsing the file is finished
	// flush the parsed chunk and finalize parsing
	if (mode == ParserMode::PARSING) {
		Flush(insert_chunk);
	}

	end_of_file_reached = true;
	return true;
}

void BufferedCSVReader::SkipEmptyLines() {
	if (parse_chunk.data.size() == 1) {
		// Empty lines are null data.
		return;
	}
	for (; position < buffer_size; position++) {
		if (!StringUtil::CharacterIsNewline(buffer[position])) {
			return;
		}
	}
}

void UpdateMaxLineLength(ClientContext &context, idx_t line_length) {
	if (!context.client_data->debug_set_max_line_length) {
		return;
	}
	if (line_length < context.client_data->debug_max_line_length) {
		return;
	}
	context.client_data->debug_max_line_length = line_length;
}

bool BufferedCSVReader::TryParseSimpleCSV(DataChunk &insert_chunk, string &error_message) {
	// used for parsing algorithm
	bool finished_chunk = false;
	idx_t column = 0;
	idx_t offset = 0;
	bool has_quotes = false;
	vector<idx_t> escape_positions;

	idx_t line_start = position;
	// read values into the buffer (if any)
	if (position >= buffer_size) {
		if (!ReadBuffer(start, line_start)) {
			return true;
		}
	}

	// start parsing the first value
	goto value_start;
value_start:
	offset = 0;
	/* state: value_start */
	// this state parses the first character of a value
	if (buffer[position] == options.quote[0]) {
		// quote: actual value starts in the next position
		// move to in_quotes state
		start = position + 1;
		goto in_quotes;
	} else {
		// no quote, move to normal parsing state
		start = position;
		goto normal;
	}
normal:
	/* state: normal parsing state */
	// this state parses the remainder of a non-quoted value until we reach a delimiter or newline
	do {
		for (; position < buffer_size; position++) {
			if (buffer[position] == options.delimiter[0]) {
				// delimiter: end the value and add it to the chunk
				goto add_value;
			} else if (StringUtil::CharacterIsNewline(buffer[position])) {
				// newline: add row
				goto add_row;
			}
		}
	} while (ReadBuffer(start, line_start));
	// file ends during normal scan: go to end state
	goto final_state;
add_value:
	AddValue(string_t(buffer.get() + start, position - start - offset), column, escape_positions, has_quotes);
	// increase position by 1 and move start to the new position
	offset = 0;
	has_quotes = false;
	start = ++position;
	if (position >= buffer_size && !ReadBuffer(start, line_start)) {
		// file ends right after delimiter, go to final state
		goto final_state;
	}
	goto value_start;
add_row : {
	// check type of newline (\r or \n)
	bool carriage_return = buffer[position] == '\r';
	AddValue(string_t(buffer.get() + start, position - start - offset), column, escape_positions, has_quotes);
	if (!error_message.empty()) {
		return false;
	}
	finished_chunk = AddRow(insert_chunk, column, error_message);
	UpdateMaxLineLength(context, position - line_start);
	if (!error_message.empty()) {
		return false;
	}
	// increase position by 1 and move start to the new position
	offset = 0;
	has_quotes = false;
	position++;
	start = position;
	line_start = position;
	if (position >= buffer_size && !ReadBuffer(start, line_start)) {
		// file ends right after delimiter, go to final state
		goto final_state;
	}
	if (carriage_return) {
		// \r newline, go to special state that parses an optional \n afterwards
		goto carriage_return;
	} else {
		SetNewLineDelimiter();
		SkipEmptyLines();

		start = position;
		line_start = position;
		if (position >= buffer_size && !ReadBuffer(start, line_start)) {
			// file ends right after delimiter, go to final state
			goto final_state;
		}
		// \n newline, move to value start
		if (finished_chunk) {
			return true;
		}
		goto value_start;
	}
}
in_quotes:
	/* state: in_quotes */
	// this state parses the remainder of a quoted value
	has_quotes = true;
	position++;
	do {
		for (; position < buffer_size; position++) {
			if (buffer[position] == options.quote[0]) {
				// quote: move to unquoted state
				goto unquote;
			} else if (buffer[position] == options.escape[0]) {
				// escape: store the escaped position and move to handle_escape state
				escape_positions.push_back(position - start);
				goto handle_escape;
			}
		}
	} while (ReadBuffer(start, line_start));
	// still in quoted state at the end of the file, error:
	throw InvalidInputException("Error in file \"%s\" on line %s: unterminated quotes. (%s)", options.file_path,
	                            GetLineNumberStr(linenr, linenr_estimated).c_str(), options.ToString());
unquote:
	/* state: unquote */
	// this state handles the state directly after we unquote
	// in this state we expect either another quote (entering the quoted state again, and escaping the quote)
	// or a delimiter/newline, ending the current value and moving on to the next value
	position++;
	if (position >= buffer_size && !ReadBuffer(start, line_start)) {
		// file ends right after unquote, go to final state
		offset = 1;
		goto final_state;
	}
	if (buffer[position] == options.quote[0] && (options.escape.empty() || options.escape[0] == options.quote[0])) {
		// escaped quote, return to quoted state and store escape position
		escape_positions.push_back(position - start);
		goto in_quotes;
	} else if (buffer[position] == options.delimiter[0]) {
		// delimiter, add value
		offset = 1;
		goto add_value;
	} else if (StringUtil::CharacterIsNewline(buffer[position])) {
		offset = 1;
		goto add_row;
	} else {
		error_message = StringUtil::Format(
		    "Error in file \"%s\" on line %s: quote should be followed by end of value, end of "
		    "row or another quote. (%s)",
		    options.file_path, GetLineNumberStr(linenr, linenr_estimated).c_str(), options.ToString());
		return false;
	}
handle_escape:
	/* state: handle_escape */
	// escape should be followed by a quote or another escape character
	position++;
	if (position >= buffer_size && !ReadBuffer(start, line_start)) {
		error_message = StringUtil::Format(
		    "Error in file \"%s\" on line %s: neither QUOTE nor ESCAPE is proceeded by ESCAPE. (%s)", options.file_path,
		    GetLineNumberStr(linenr, linenr_estimated).c_str(), options.ToString());
		return false;
	}
	if (buffer[position] != options.quote[0] && buffer[position] != options.escape[0]) {
		error_message = StringUtil::Format(
		    "Error in file \"%s\" on line %s: neither QUOTE nor ESCAPE is proceeded by ESCAPE. (%s)", options.file_path,
		    GetLineNumberStr(linenr, linenr_estimated).c_str(), options.ToString());
		return false;
	}
	// escape was followed by quote or escape, go back to quoted state
	goto in_quotes;
carriage_return:
	/* state: carriage_return */
	// this stage optionally skips a newline (\n) character, which allows \r\n to be interpreted as a single line
	if (buffer[position] == '\n') {
		SetNewLineDelimiter(true, true);
		// newline after carriage return: skip
		// increase position by 1 and move start to the new position
		start = ++position;
		if (position >= buffer_size && !ReadBuffer(start, line_start)) {
			// file ends right after delimiter, go to final state
			goto final_state;
		}
	} else {
		SetNewLineDelimiter(true, false);
	}
	if (finished_chunk) {
		return true;
	}
	SkipEmptyLines();
	start = position;
	line_start = position;
	if (position >= buffer_size && !ReadBuffer(start, line_start)) {
		// file ends right after delimiter, go to final state
		goto final_state;
	}

	goto value_start;
final_state:
	if (finished_chunk) {
		return true;
	}

	if (column > 0 || position > start) {
		// remaining values to be added to the chunk
		AddValue(string_t(buffer.get() + start, position - start - offset), column, escape_positions, has_quotes);
		finished_chunk = AddRow(insert_chunk, column, error_message);
		SkipEmptyLines();
		UpdateMaxLineLength(context, position - line_start);
		if (!error_message.empty()) {
			return false;
		}
	}

	// final stage, only reached after parsing the file is finished
	// flush the parsed chunk and finalize parsing
	if (mode == ParserMode::PARSING) {
		Flush(insert_chunk);
	}

	end_of_file_reached = true;
	return true;
}

bool BufferedCSVReader::ReadBuffer(idx_t &start, idx_t &line_start) {
	if (start > buffer_size) {
		return false;
	}
	auto old_buffer = std::move(buffer);

	// the remaining part of the last buffer
	idx_t remaining = buffer_size - start;

	bool large_buffers = mode == ParserMode::PARSING && !file_handle->OnDiskFile() && file_handle->CanSeek();
	idx_t buffer_read_size = large_buffers ? INITIAL_BUFFER_SIZE_LARGE : INITIAL_BUFFER_SIZE;

	while (remaining > buffer_read_size) {
		buffer_read_size *= 2;
	}

	// Check line length
	if (remaining > options.maximum_line_size) {
		throw InvalidInputException("Maximum line size of %llu bytes exceeded on line %s!", options.maximum_line_size,
		                            GetLineNumberStr(linenr, linenr_estimated));
	}

	buffer = make_unsafe_uniq_array<char>(buffer_read_size + remaining + 1);
	buffer_size = remaining + buffer_read_size;
	if (remaining > 0) {
		// remaining from last buffer: copy it here
		memcpy(buffer.get(), old_buffer.get() + start, remaining);
	}
	idx_t read_count = file_handle->Read(buffer.get() + remaining, buffer_read_size);

	bytes_in_chunk += read_count;
	buffer_size = remaining + read_count;
	buffer[buffer_size] = '\0';
	if (old_buffer) {
		cached_buffers.push_back(std::move(old_buffer));
	}
	start = 0;
	position = remaining;
	if (!bom_checked) {
		bom_checked = true;
		if (read_count >= 3 && buffer[0] == '\xEF' && buffer[1] == '\xBB' && buffer[2] == '\xBF') {
			start += 3;
			position += 3;
		}
	}
	line_start = start;

	return read_count > 0;
}

void BufferedCSVReader::ParseCSV(DataChunk &insert_chunk) {
	string error_message;
	if (!TryParseCSV(ParserMode::PARSING, insert_chunk, error_message)) {
		throw InvalidInputException(error_message);
	}
}

bool BufferedCSVReader::TryParseCSV(ParserMode mode) {
	DataChunk dummy_chunk;
	string error_message;
	return TryParseCSV(mode, dummy_chunk, error_message);
}

void BufferedCSVReader::ParseCSV(ParserMode mode) {
	DataChunk dummy_chunk;
	string error_message;
	if (!TryParseCSV(mode, dummy_chunk, error_message)) {
		throw InvalidInputException(error_message);
	}
}

bool BufferedCSVReader::TryParseCSV(ParserMode parser_mode, DataChunk &insert_chunk, string &error_message) {
	mode = parser_mode;

	if (options.quote.size() <= 1 && options.escape.size() <= 1 && options.delimiter.size() == 1) {
		return TryParseSimpleCSV(insert_chunk, error_message);
	} else {
		return TryParseComplexCSV(insert_chunk, error_message);
	}
}

} // namespace duckdb



namespace duckdb {

CSVBuffer::CSVBuffer(ClientContext &context, idx_t buffer_size_p, CSVFileHandle &file_handle,
                     idx_t &global_csv_current_position, idx_t file_number_p)
    : context(context), first_buffer(true), file_number(file_number_p) {
	this->handle = AllocateBuffer(buffer_size_p);

	auto buffer = Ptr();
	actual_size = file_handle.Read(buffer, buffer_size_p);
	global_csv_start = global_csv_current_position;
	global_csv_current_position += actual_size;
	if (actual_size >= 3 && buffer[0] == '\xEF' && buffer[1] == '\xBB' && buffer[2] == '\xBF') {
		start_position += 3;
	}
	last_buffer = file_handle.FinishedReading();
}

CSVBuffer::CSVBuffer(ClientContext &context, BufferHandle buffer_p, idx_t buffer_size_p, idx_t actual_size_p,
                     bool final_buffer, idx_t global_csv_current_position, idx_t file_number_p)
    : context(context), handle(std::move(buffer_p)), actual_size(actual_size_p), last_buffer(final_buffer),
      global_csv_start(global_csv_current_position), file_number(file_number_p) {
}

unique_ptr<CSVBuffer> CSVBuffer::Next(CSVFileHandle &file_handle, idx_t buffer_size, idx_t &global_csv_current_position,
                                      idx_t file_number_p) {
	auto next_buffer = AllocateBuffer(buffer_size);
	idx_t next_buffer_actual_size = file_handle.Read(next_buffer.Ptr(), buffer_size);
	if (next_buffer_actual_size == 0) {
		// We are done reading
		return nullptr;
	}

	auto next_csv_buffer =
	    make_uniq<CSVBuffer>(context, std::move(next_buffer), buffer_size, next_buffer_actual_size,
	                         file_handle.FinishedReading(), global_csv_current_position, file_number_p);
	global_csv_current_position += next_buffer_actual_size;
	return next_csv_buffer;
}

BufferHandle CSVBuffer::AllocateBuffer(idx_t buffer_size) {
	auto &buffer_manager = BufferManager::GetBufferManager(context);
	return buffer_manager.Allocate(MaxValue<idx_t>(Storage::BLOCK_SIZE, buffer_size));
}

idx_t CSVBuffer::GetBufferSize() {
	return actual_size;
}

idx_t CSVBuffer::GetStart() {
	return start_position;
}

bool CSVBuffer::IsCSVFileLastBuffer() {
	return last_buffer;
}

bool CSVBuffer::IsCSVFileFirstBuffer() {
	return first_buffer;
}

idx_t CSVBuffer::GetCSVGlobalStart() {
	return global_csv_start;
}

idx_t CSVBuffer::GetFileNumber() {
	return file_number;
}

} // namespace duckdb


namespace duckdb {

CSVFileHandle::CSVFileHandle(FileSystem &fs, Allocator &allocator, unique_ptr<FileHandle> file_handle_p,
                             const string &path_p, FileCompressionType compression, bool enable_reset)
    : fs(fs), allocator(allocator), file_handle(std::move(file_handle_p)), path(path_p), compression(compression),
      reset_enabled(enable_reset) {
	can_seek = file_handle->CanSeek();
	on_disk_file = file_handle->OnDiskFile();
	file_size = file_handle->GetFileSize();
}

unique_ptr<FileHandle> CSVFileHandle::OpenFileHandle(FileSystem &fs, Allocator &allocator, const string &path,
                                                     FileCompressionType compression) {
	auto file_handle = fs.OpenFile(path.c_str(), FileFlags::FILE_FLAGS_READ, FileLockType::NO_LOCK, compression);
	if (file_handle->CanSeek()) {
		file_handle->Reset();
	}
	return file_handle;
}

unique_ptr<CSVFileHandle> CSVFileHandle::OpenFile(FileSystem &fs, Allocator &allocator, const string &path,
                                                  FileCompressionType compression, bool enable_reset) {
	auto file_handle = CSVFileHandle::OpenFileHandle(fs, allocator, path, compression);
	return make_uniq<CSVFileHandle>(fs, allocator, std::move(file_handle), path, compression, enable_reset);
}

bool CSVFileHandle::CanSeek() {
	return can_seek;
}

void CSVFileHandle::Seek(idx_t position) {
	if (!can_seek) {
		throw InternalException("Cannot seek in this file");
	}
	file_handle->Seek(position);
}

idx_t CSVFileHandle::SeekPosition() {
	if (!can_seek) {
		throw InternalException("Cannot seek in this file");
	}
	return file_handle->SeekPosition();
}

void CSVFileHandle::Reset() {
	requested_bytes = 0;
	read_position = 0;
	if (can_seek) {
		// we can seek - reset the file handle
		file_handle->Reset();
	} else if (on_disk_file) {
		// we cannot seek but it is an on-disk file - re-open the file
		file_handle = CSVFileHandle::OpenFileHandle(fs, allocator, path, compression);
	} else {
		if (!reset_enabled) {
			throw InternalException("Reset called but reset is not enabled for this CSV Handle");
		}
		read_position = 0;
	}
}
bool CSVFileHandle::OnDiskFile() {
	return on_disk_file;
}

idx_t CSVFileHandle::FileSize() {
	return file_size;
}

bool CSVFileHandle::FinishedReading() {
	return requested_bytes >= file_size;
}

idx_t CSVFileHandle::Read(void *buffer, idx_t nr_bytes) {
	requested_bytes += nr_bytes;
	if (on_disk_file || can_seek) {
		// if this is a plain file source OR we can seek we are not caching anything
		return file_handle->Read(buffer, nr_bytes);
	}
	// not a plain file source: we need to do some bookkeeping around the reset functionality
	idx_t result_offset = 0;
	if (read_position < buffer_size) {
		// we need to read from our cached buffer
		auto buffer_read_count = MinValue<idx_t>(nr_bytes, buffer_size - read_position);
		memcpy(buffer, cached_buffer.get() + read_position, buffer_read_count);
		result_offset += buffer_read_count;
		read_position += buffer_read_count;
		if (result_offset == nr_bytes) {
			return nr_bytes;
		}
	} else if (!reset_enabled && cached_buffer.IsSet()) {
		// reset is disabled, but we still have cached data
		// we can remove any cached data
		cached_buffer.Reset();
		buffer_size = 0;
		buffer_capacity = 0;
		read_position = 0;
	}
	// we have data left to read from the file
	// read directly into the buffer
	auto bytes_read = file_handle->Read((char *)buffer + result_offset, nr_bytes - result_offset);
	file_size = file_handle->GetFileSize();
	read_position += bytes_read;
	if (reset_enabled) {
		// if reset caching is enabled, we need to cache the bytes that we have read
		if (buffer_size + bytes_read >= buffer_capacity) {
			// no space; first enlarge the buffer
			buffer_capacity = MaxValue<idx_t>(NextPowerOfTwo(buffer_size + bytes_read), buffer_capacity * 2);

			auto new_buffer = allocator.Allocate(buffer_capacity);
			if (buffer_size > 0) {
				memcpy(new_buffer.get(), cached_buffer.get(), buffer_size);
			}
			cached_buffer = std::move(new_buffer);
		}
		memcpy(cached_buffer.get() + buffer_size, (char *)buffer + result_offset, bytes_read);
		buffer_size += bytes_read;
	}

	return result_offset + bytes_read;
}

string CSVFileHandle::ReadLine() {
	bool carriage_return = false;
	string result;
	char buffer[1];
	while (true) {
		idx_t bytes_read = Read(buffer, 1);
		if (bytes_read == 0) {
			return result;
		}
		if (carriage_return) {
			if (buffer[0] != '\n') {
				if (!file_handle->CanSeek()) {
					throw BinderException(
					    "Carriage return newlines not supported when reading CSV files in which we cannot seek");
				}
				file_handle->Seek(file_handle->SeekPosition() - 1);
				return result;
			}
		}
		if (buffer[0] == '\n') {
			return result;
		}
		if (buffer[0] != '\r') {
			result += buffer[0];
		} else {
			carriage_return = true;
		}
	}
}

void CSVFileHandle::DisableReset() {
	this->reset_enabled = false;
}

} // namespace duckdb





namespace duckdb {

static bool ParseBoolean(const Value &value, const string &loption);

static bool ParseBoolean(const vector<Value> &set, const string &loption) {
	if (set.empty()) {
		// no option specified: default to true
		return true;
	}
	if (set.size() > 1) {
		throw BinderException("\"%s\" expects a single argument as a boolean value (e.g. TRUE or 1)", loption);
	}
	return ParseBoolean(set[0], loption);
}

static bool ParseBoolean(const Value &value, const string &loption) {

	if (value.type().id() == LogicalTypeId::LIST) {
		auto &children = ListValue::GetChildren(value);
		return ParseBoolean(children, loption);
	}
	if (value.type() == LogicalType::FLOAT || value.type() == LogicalType::DOUBLE ||
	    value.type().id() == LogicalTypeId::DECIMAL) {
		throw BinderException("\"%s\" expects a boolean value (e.g. TRUE or 1)", loption);
	}
	return BooleanValue::Get(value.DefaultCastAs(LogicalType::BOOLEAN));
}

static string ParseString(const Value &value, const string &loption) {
	if (value.IsNull()) {
		return string();
	}
	if (value.type().id() == LogicalTypeId::LIST) {
		auto &children = ListValue::GetChildren(value);
		if (children.size() != 1) {
			throw BinderException("\"%s\" expects a single argument as a string value", loption);
		}
		return ParseString(children[0], loption);
	}
	if (value.type().id() != LogicalTypeId::VARCHAR) {
		throw BinderException("\"%s\" expects a string argument!", loption);
	}
	return value.GetValue<string>();
}

static int64_t ParseInteger(const Value &value, const string &loption) {
	if (value.type().id() == LogicalTypeId::LIST) {
		auto &children = ListValue::GetChildren(value);
		if (children.size() != 1) {
			// no option specified or multiple options specified
			throw BinderException("\"%s\" expects a single argument as an integer value", loption);
		}
		return ParseInteger(children[0], loption);
	}
	return value.GetValue<int64_t>();
}

void BufferedCSVReaderOptions::SetHeader(bool input) {
	this->header = input;
	this->has_header = true;
}

void BufferedCSVReaderOptions::SetCompression(const string &compression_p) {
	this->compression = FileCompressionTypeFromString(compression_p);
}

void BufferedCSVReaderOptions::SetEscape(const string &input) {
	this->escape = input;
	this->has_escape = true;
}

void BufferedCSVReaderOptions::SetDelimiter(const string &input) {
	this->delimiter = StringUtil::Replace(input, "\\t", "\t");
	this->has_delimiter = true;
	if (input.empty()) {
		this->delimiter = string("\0", 1);
	}
}

void BufferedCSVReaderOptions::SetQuote(const string &quote_p) {
	this->quote = quote_p;
	this->has_quote = true;
}

void BufferedCSVReaderOptions::SetNewline(const string &input) {
	if (input == "\\n" || input == "\\r") {
		new_line = NewLineIdentifier::SINGLE;
	} else if (input == "\\r\\n") {
		new_line = NewLineIdentifier::CARRY_ON;
	} else {
		throw InvalidInputException("This is not accepted as a newline: " + input);
	}
	has_newline = true;
}

void BufferedCSVReaderOptions::SetDateFormat(LogicalTypeId type, const string &format, bool read_format) {
	string error;
	if (read_format) {
		error = StrTimeFormat::ParseFormatSpecifier(format, date_format[type]);
		date_format[type].format_specifier = format;
	} else {
		error = StrTimeFormat::ParseFormatSpecifier(format, write_date_format[type]);
	}
	if (!error.empty()) {
		throw InvalidInputException("Could not parse DATEFORMAT: %s", error.c_str());
	}
	has_format[type] = true;
}

void BufferedCSVReaderOptions::SetReadOption(const string &loption, const Value &value,
                                             vector<string> &expected_names) {
	if (SetBaseOption(loption, value)) {
		return;
	}
	if (loption == "auto_detect") {
		auto_detect = ParseBoolean(value, loption);
	} else if (loption == "sample_size") {
		int64_t sample_size = ParseInteger(value, loption);
		if (sample_size < 1 && sample_size != -1) {
			throw BinderException("Unsupported parameter for SAMPLE_SIZE: cannot be smaller than 1");
		}
		if (sample_size == -1) {
			sample_chunks = std::numeric_limits<uint64_t>::max();
			sample_chunk_size = STANDARD_VECTOR_SIZE;
		} else if (sample_size <= STANDARD_VECTOR_SIZE) {
			sample_chunk_size = sample_size;
			sample_chunks = 1;
		} else {
			sample_chunk_size = STANDARD_VECTOR_SIZE;
			sample_chunks = sample_size / STANDARD_VECTOR_SIZE + 1;
		}
	} else if (loption == "skip") {
		skip_rows = ParseInteger(value, loption);
		skip_rows_set = true;
	} else if (loption == "max_line_size" || loption == "maximum_line_size") {
		maximum_line_size = ParseInteger(value, loption);
	} else if (loption == "sample_chunk_size") {
		sample_chunk_size = ParseInteger(value, loption);
		if (sample_chunk_size > STANDARD_VECTOR_SIZE) {
			throw BinderException(
			    "Unsupported parameter for SAMPLE_CHUNK_SIZE: cannot be bigger than STANDARD_VECTOR_SIZE %d",
			    STANDARD_VECTOR_SIZE);
		} else if (sample_chunk_size < 1) {
			throw BinderException("Unsupported parameter for SAMPLE_CHUNK_SIZE: cannot be smaller than 1");
		}
	} else if (loption == "sample_chunks") {
		sample_chunks = ParseInteger(value, loption);
		if (sample_chunks < 1) {
			throw BinderException("Unsupported parameter for SAMPLE_CHUNKS: cannot be smaller than 1");
		}
	} else if (loption == "force_not_null") {
		force_not_null = ParseColumnList(value, expected_names, loption);
	} else if (loption == "date_format" || loption == "dateformat") {
		string format = ParseString(value, loption);
		SetDateFormat(LogicalTypeId::DATE, format, true);
	} else if (loption == "timestamp_format" || loption == "timestampformat") {
		string format = ParseString(value, loption);
		SetDateFormat(LogicalTypeId::TIMESTAMP, format, true);
	} else if (loption == "ignore_errors") {
		ignore_errors = ParseBoolean(value, loption);
	} else if (loption == "buffer_size") {
		buffer_size = ParseInteger(value, loption);
		if (buffer_size == 0) {
			throw InvalidInputException("Buffer Size option must be higher than 0");
		}
	} else if (loption == "decimal_separator") {
		decimal_separator = ParseString(value, loption);
		if (decimal_separator != "." && decimal_separator != ",") {
			throw BinderException("Unsupported parameter for DECIMAL_SEPARATOR: should be '.' or ','");
		}
	} else if (loption == "null_padding") {
		null_padding = ParseBoolean(value, loption);
	} else if (loption == "allow_quoted_nulls") {
		allow_quoted_nulls = ParseBoolean(value, loption);
	} else {
		throw BinderException("Unrecognized option for CSV reader \"%s\"", loption);
	}
}

void BufferedCSVReaderOptions::SetWriteOption(const string &loption, const Value &value) {
	if (loption == "new_line") {
		// Steal this from SetBaseOption so we can write different newlines (e.g., format JSON ARRAY)
		write_newline = ParseString(value, loption);
		return;
	}

	if (SetBaseOption(loption, value)) {
		return;
	}

	if (loption == "force_quote") {
		force_quote = ParseColumnList(value, name_list, loption);
	} else if (loption == "date_format" || loption == "dateformat") {
		string format = ParseString(value, loption);
		SetDateFormat(LogicalTypeId::DATE, format, false);
	} else if (loption == "timestamp_format" || loption == "timestampformat") {
		string format = ParseString(value, loption);
		if (StringUtil::Lower(format) == "iso") {
			format = "%Y-%m-%dT%H:%M:%S.%fZ";
		}
		SetDateFormat(LogicalTypeId::TIMESTAMP, format, false);
		SetDateFormat(LogicalTypeId::TIMESTAMP_TZ, format, false);
	} else if (loption == "prefix") {
		prefix = ParseString(value, loption);
	} else if (loption == "suffix") {
		suffix = ParseString(value, loption);
	} else {
		throw BinderException("Unrecognized option CSV writer \"%s\"", loption);
	}
}

bool BufferedCSVReaderOptions::SetBaseOption(const string &loption, const Value &value) {
	// Make sure this function was only called after the option was turned into lowercase
	D_ASSERT(!std::any_of(loption.begin(), loption.end(), ::isupper));

	if (StringUtil::StartsWith(loption, "delim") || StringUtil::StartsWith(loption, "sep")) {
		SetDelimiter(ParseString(value, loption));
	} else if (loption == "quote") {
		SetQuote(ParseString(value, loption));
	} else if (loption == "new_line") {
		SetNewline(ParseString(value, loption));
	} else if (loption == "escape") {
		SetEscape(ParseString(value, loption));
	} else if (loption == "header") {
		SetHeader(ParseBoolean(value, loption));
	} else if (loption == "null" || loption == "nullstr") {
		null_str = ParseString(value, loption);
	} else if (loption == "encoding") {
		auto encoding = StringUtil::Lower(ParseString(value, loption));
		if (encoding != "utf8" && encoding != "utf-8") {
			throw BinderException("Copy is only supported for UTF-8 encoded files, ENCODING 'UTF-8'");
		}
	} else if (loption == "compression") {
		SetCompression(ParseString(value, loption));
	} else {
		// unrecognized option in base CSV
		return false;
	}
	return true;
}

std::string BufferedCSVReaderOptions::ToString() const {
	return "  file=" + file_path + "\n  delimiter='" + delimiter +
	       (has_delimiter ? "'" : (auto_detect ? "' (auto detected)" : "' (default)")) + "\n  quote='" + quote +
	       (has_quote ? "'" : (auto_detect ? "' (auto detected)" : "' (default)")) + "\n  escape='" + escape +
	       (has_escape ? "'" : (auto_detect ? "' (auto detected)" : "' (default)")) +
	       "\n  header=" + std::to_string(header) +
	       (has_header ? "" : (auto_detect ? " (auto detected)" : "' (default)")) +
	       "\n  sample_size=" + std::to_string(sample_chunk_size * sample_chunks) +
	       "\n  ignore_errors=" + std::to_string(ignore_errors) + "\n  all_varchar=" + std::to_string(all_varchar);
}

} // namespace duckdb



















#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>

namespace duckdb {

ParallelCSVReader::ParallelCSVReader(ClientContext &context, BufferedCSVReaderOptions options_p,
                                     unique_ptr<CSVBufferRead> buffer_p, idx_t first_pos_first_buffer_p,
                                     const vector<LogicalType> &requested_types, idx_t file_idx_p)
    : BaseCSVReader(context, std::move(options_p), requested_types), file_idx(file_idx_p),
      first_pos_first_buffer(first_pos_first_buffer_p) {
	Initialize(requested_types);
	SetBufferRead(std::move(buffer_p));
	if (options.delimiter.size() > 1 || options.escape.size() > 1 || options.quote.size() > 1) {
		throw InternalException("Parallel CSV reader cannot handle CSVs with multi-byte delimiters/escapes/quotes");
	}
}

void ParallelCSVReader::Initialize(const vector<LogicalType> &requested_types) {
	return_types = requested_types;
	InitParseChunk(return_types.size());
}

bool ParallelCSVReader::NewLineDelimiter(bool carry, bool carry_followed_by_nl, bool first_char) {
	// Set the delimiter if not set yet.
	SetNewLineDelimiter(carry, carry_followed_by_nl);
	D_ASSERT(options.new_line == NewLineIdentifier::SINGLE || options.new_line == NewLineIdentifier::CARRY_ON);
	if (options.new_line == NewLineIdentifier::SINGLE) {
		return (!carry) || (carry && !carry_followed_by_nl);
	}
	return (carry && carry_followed_by_nl) || (!carry && first_char);
}

void ParallelCSVReader::SkipEmptyLines() {
	idx_t new_pos_buffer = position_buffer;
	if (parse_chunk.data.size() == 1) {
		// Empty lines are null data.
		return;
	}
	for (; new_pos_buffer < end_buffer; new_pos_buffer++) {
		if (StringUtil::CharacterIsNewline((*buffer)[new_pos_buffer])) {
			bool carrier_return = (*buffer)[new_pos_buffer] == '\r';
			new_pos_buffer++;
			if (carrier_return && new_pos_buffer < buffer_size && (*buffer)[new_pos_buffer] == '\n') {
				position_buffer++;
			}
			if (new_pos_buffer > end_buffer) {
				return;
			}
			position_buffer = new_pos_buffer;
		} else if ((*buffer)[new_pos_buffer] != ' ') {
			return;
		}
	}
}

bool ParallelCSVReader::SetPosition() {
	if (buffer->buffer->IsCSVFileFirstBuffer() && start_buffer == position_buffer &&
	    start_buffer == first_pos_first_buffer) {
		start_buffer = buffer->buffer->GetStart();
		position_buffer = start_buffer;
		verification_positions.beginning_of_first_line = position_buffer;
		verification_positions.end_of_last_line = position_buffer;
		// First buffer doesn't need any setting

		if (options.header) {
			for (; position_buffer < end_buffer; position_buffer++) {
				if (StringUtil::CharacterIsNewline((*buffer)[position_buffer])) {
					bool carrier_return = (*buffer)[position_buffer] == '\r';
					position_buffer++;
					if (carrier_return && position_buffer < buffer_size && (*buffer)[position_buffer] == '\n') {
						position_buffer++;
					}
					if (position_buffer > end_buffer) {
						return false;
					}
					SkipEmptyLines();
					if (verification_positions.beginning_of_first_line == 0) {
						verification_positions.beginning_of_first_line = position_buffer;
					}

					verification_positions.end_of_last_line = position_buffer;
					return true;
				}
			}
			return false;
		}
		SkipEmptyLines();
		if (verification_positions.beginning_of_first_line == 0) {
			verification_positions.beginning_of_first_line = position_buffer;
		}

		verification_positions.end_of_last_line = position_buffer;
		return true;
	}

	// We have to move position up to next new line
	idx_t end_buffer_real = end_buffer;
	// Check if we already start in a valid line
	string error_message;
	bool successfully_read_first_line = false;
	while (!successfully_read_first_line) {
		DataChunk first_line_chunk;
		first_line_chunk.Initialize(allocator, return_types);
		// Ensure that parse_chunk has no gunk when trying to figure new line
		parse_chunk.Reset();
		for (; position_buffer < end_buffer; position_buffer++) {
			if (StringUtil::CharacterIsNewline((*buffer)[position_buffer])) {
				bool carriage_return = (*buffer)[position_buffer] == '\r';
				bool carriage_return_followed = false;
				position_buffer++;
				if (position_buffer < end_buffer) {
					if (carriage_return && (*buffer)[position_buffer] == '\n') {
						carriage_return_followed = true;
						position_buffer++;
					}
				}
				if (NewLineDelimiter(carriage_return, carriage_return_followed, position_buffer - 1 == start_buffer)) {
					break;
				}
			}
		}
		SkipEmptyLines();

		if (position_buffer > buffer_size) {
			break;
		}

		if (position_buffer >= end_buffer && !StringUtil::CharacterIsNewline((*buffer)[position_buffer - 1])) {
			break;
		}

		if (position_buffer > end_buffer && options.new_line == NewLineIdentifier::CARRY_ON &&
		    (*buffer)[position_buffer - 1] == '\n') {
			break;
		}
		idx_t position_set = position_buffer;
		start_buffer = position_buffer;
		// We check if we can add this line
		// disable the projection pushdown while reading the first line
		// otherwise the first line parsing can be influenced by which columns we are reading
		auto column_ids = std::move(reader_data.column_ids);
		auto column_mapping = std::move(reader_data.column_mapping);
		InitializeProjection();
		try {
			successfully_read_first_line = TryParseSimpleCSV(first_line_chunk, error_message, true);
		} catch (...) {
			successfully_read_first_line = false;
		}
		// restore the projection pushdown
		reader_data.column_ids = std::move(column_ids);
		reader_data.column_mapping = std::move(column_mapping);
		end_buffer = end_buffer_real;
		start_buffer = position_set;
		if (position_buffer >= end_buffer) {
			if (successfully_read_first_line) {
				position_buffer = position_set;
			}
			break;
		}
		position_buffer = position_set;
	}
	if (verification_positions.beginning_of_first_line == 0) {
		verification_positions.beginning_of_first_line = position_buffer;
	}
	// Ensure that parse_chunk has no gunk when trying to figure new line
	parse_chunk.Reset();

	verification_positions.end_of_last_line = position_buffer;
	finished = false;
	return successfully_read_first_line;
}

void ParallelCSVReader::SetBufferRead(unique_ptr<CSVBufferRead> buffer_read_p) {
	if (!buffer_read_p->buffer) {
		throw InternalException("ParallelCSVReader::SetBufferRead - CSVBufferRead does not have a buffer to read");
	}
	position_buffer = buffer_read_p->buffer_start;
	start_buffer = buffer_read_p->buffer_start;
	end_buffer = buffer_read_p->buffer_end;
	if (buffer_read_p->next_buffer) {
		buffer_size = buffer_read_p->buffer->GetBufferSize() + buffer_read_p->next_buffer->GetBufferSize();
	} else {
		buffer_size = buffer_read_p->buffer->GetBufferSize();
	}
	buffer = std::move(buffer_read_p);

	reached_remainder_state = false;
	verification_positions.beginning_of_first_line = 0;
	verification_positions.end_of_last_line = 0;
	finished = false;
	D_ASSERT(end_buffer <= buffer_size);
}

VerificationPositions ParallelCSVReader::GetVerificationPositions() {
	verification_positions.beginning_of_first_line += buffer->buffer->GetCSVGlobalStart();
	verification_positions.end_of_last_line += buffer->buffer->GetCSVGlobalStart();
	return verification_positions;
}

// If BufferRemainder returns false, it means we are done scanning this buffer and should go to the end_state
bool ParallelCSVReader::BufferRemainder() {
	if (position_buffer >= end_buffer && !reached_remainder_state) {
		// First time we finish the buffer piece we should scan here, we set the variables
		// to allow this piece to be scanned up to the end of the buffer or the next new line
		reached_remainder_state = true;
		// end_buffer is allowed to go to buffer size to finish its last line
		end_buffer = buffer_size;
	}
	if (position_buffer >= end_buffer) {
		// buffer ends, return false
		return false;
	}
	// we can still scan stuff, return true
	return true;
}

void ParallelCSVReader::VerifyLineLength(idx_t line_size) {
	if (line_size > options.maximum_line_size) {
		throw InvalidInputException("Error in file \"%s\" on line %s: Maximum line size of %llu bytes exceeded!",
		                            options.file_path,
		                            GetLineNumberStr(parse_chunk.size(), linenr_estimated, buffer->batch_index).c_str(),
		                            options.maximum_line_size);
	}
}

bool AllNewLine(string_t value, idx_t column_amount) {
	auto value_str = value.GetString();
	if (value_str.empty() && column_amount == 1) {
		// This is a one column (empty)
		return false;
	}
	for (idx_t i = 0; i < value.GetSize(); i++) {
		if (!StringUtil::CharacterIsNewline(value_str[i])) {
			return false;
		}
	}
	return true;
}

bool ParallelCSVReader::TryParseSimpleCSV(DataChunk &insert_chunk, string &error_message, bool try_add_line) {
	// If line is not set, we have to figure it out, we assume whatever is in the first line
	if (options.new_line == NewLineIdentifier::NOT_SET) {
		idx_t cur_pos = position_buffer;
		// we can start in the middle of a new line, so move a bit forward.
		while (cur_pos < end_buffer) {
			if (StringUtil::CharacterIsNewline((*buffer)[cur_pos])) {
				cur_pos++;
			} else {
				break;
			}
		}
		for (; cur_pos < end_buffer; cur_pos++) {
			if (StringUtil::CharacterIsNewline((*buffer)[cur_pos])) {
				bool carriage_return = (*buffer)[cur_pos] == '\r';
				bool carriage_return_followed = false;
				cur_pos++;
				if (cur_pos < end_buffer) {
					if (carriage_return && (*buffer)[cur_pos] == '\n') {
						carriage_return_followed = true;
						cur_pos++;
					}
				}
				SetNewLineDelimiter(carriage_return, carriage_return_followed);
				break;
			}
		}
	}
	// used for parsing algorithm
	if (start_buffer == buffer_size) {
		// Nothing to read
		finished = true;
		return true;
	}
	D_ASSERT(end_buffer <= buffer_size);
	bool finished_chunk = false;
	idx_t column = 0;
	idx_t offset = 0;
	bool has_quotes = false;

	vector<idx_t> escape_positions;
	if ((start_buffer == buffer->buffer_start || start_buffer == buffer->buffer_end) && !try_add_line) {
		// First time reading this buffer piece
		if (!SetPosition()) {
			finished = true;
			return true;
		}
	}
	if (position_buffer == buffer_size) {
		// Nothing to read
		finished = true;
		return true;
	}
	// Keep track of line size
	idx_t line_start = position_buffer;
	// start parsing the first value
	goto value_start;

value_start : {
	/* state: value_start */
	if (!BufferRemainder()) {
		goto final_state;
	}
	offset = 0;

	// this state parses the first character of a value
	if ((*buffer)[position_buffer] == options.quote[0]) {
		// quote: actual value starts in the next position
		// move to in_quotes state
		start_buffer = position_buffer + 1;
		goto in_quotes;
	} else {
		// no quote, move to normal parsing state
		start_buffer = position_buffer;
		goto normal;
	}
};

normal : {
	/* state: normal parsing state */
	// this state parses the remainder of a non-quoted value until we reach a delimiter or newline
	for (; position_buffer < end_buffer; position_buffer++) {
		auto c = (*buffer)[position_buffer];
		if (c == options.delimiter[0]) {
			// delimiter: end the value and add it to the chunk
			goto add_value;
		} else if (c == options.quote[0] && try_add_line) {
			return false;
		} else if (StringUtil::CharacterIsNewline(c)) {
			// newline: add row
			if (column > 0 || try_add_line || parse_chunk.data.size() == 1) {
				goto add_row;
			}
			if (column == 0 && position_buffer == start_buffer) {
				start_buffer++;
			}
		}
	}
	if (!BufferRemainder()) {
		goto final_state;
	} else {
		goto normal;
	}
};

add_value : {
	/* state: Add value to string vector */
	AddValue(buffer->GetValue(start_buffer, position_buffer, offset), column, escape_positions, has_quotes,
	         buffer->local_batch_index);
	// increase position by 1 and move start to the new position
	offset = 0;
	has_quotes = false;
	start_buffer = ++position_buffer;
	if (!BufferRemainder()) {
		goto final_state;
	}
	goto value_start;
};

add_row : {
	/* state: Add Row to Parse chunk */
	// check type of newline (\r or \n)
	bool carriage_return = (*buffer)[position_buffer] == '\r';

	AddValue(buffer->GetValue(start_buffer, position_buffer, offset), column, escape_positions, has_quotes,
	         buffer->local_batch_index);
	if (try_add_line) {
		bool success = column == insert_chunk.ColumnCount();
		if (success) {
			idx_t cur_linenr = linenr;
			AddRow(insert_chunk, column, error_message, buffer->local_batch_index);
			success = Flush(insert_chunk, buffer->local_batch_index, true);
			linenr = cur_linenr;
		}
		reached_remainder_state = false;
		parse_chunk.Reset();
		return success;
	} else {
		VerifyLineLength(position_buffer - line_start);
		line_start = position_buffer;
		finished_chunk = AddRow(insert_chunk, column, error_message, buffer->local_batch_index);
	}
	// increase position by 1 and move start to the new position
	offset = 0;
	has_quotes = false;
	position_buffer++;
	start_buffer = position_buffer;
	verification_positions.end_of_last_line = position_buffer;
	if (carriage_return) {
		// \r newline, go to special state that parses an optional \n afterwards
		// optionally skips a newline (\n) character, which allows \r\n to be interpreted as a single line
		if (!BufferRemainder()) {
			goto final_state;
		}
		if ((*buffer)[position_buffer] == '\n') {
			if (options.new_line == NewLineIdentifier::SINGLE) {
				error_message = "Wrong NewLine Identifier. Expecting \\r\\n";
				return false;
			}
			// newline after carriage return: skip
			// increase position by 1 and move start to the new position
			start_buffer = ++position_buffer;

			SkipEmptyLines();
			verification_positions.end_of_last_line = position_buffer;
			start_buffer = position_buffer;
			if (reached_remainder_state) {
				goto final_state;
			}
		} else {
			if (options.new_line == NewLineIdentifier::CARRY_ON) {
				error_message = "Wrong NewLine Identifier. Expecting \\r or \\n";
				return false;
			}
		}
		if (!BufferRemainder()) {
			goto final_state;
		}
		if (reached_remainder_state || finished_chunk) {
			goto final_state;
		}
		goto value_start;
	} else {
		if (options.new_line == NewLineIdentifier::CARRY_ON) {
			error_message = "Wrong NewLine Identifier. Expecting \\r or \\n";
			return false;
		}
		if (reached_remainder_state) {
			goto final_state;
		}
		if (!BufferRemainder()) {
			goto final_state;
		}
		SkipEmptyLines();
		verification_positions.end_of_last_line = position_buffer;
		start_buffer = position_buffer;
		// \n newline, move to value start
		if (finished_chunk) {
			goto final_state;
		}
		goto value_start;
	}
}
in_quotes:
	/* state: in_quotes this state parses the remainder of a quoted value*/
	has_quotes = true;
	position_buffer++;
	for (; position_buffer < end_buffer; position_buffer++) {
		auto c = (*buffer)[position_buffer];
		if (c == options.quote[0]) {
			// quote: move to unquoted state
			goto unquote;
		} else if (c == options.escape[0]) {
			// escape: store the escaped position and move to handle_escape state
			escape_positions.push_back(position_buffer - start_buffer);
			goto handle_escape;
		}
	}
	if (!BufferRemainder()) {
		if (buffer->buffer->IsCSVFileLastBuffer()) {
			if (try_add_line) {
				return false;
			}
			// still in quoted state at the end of the file or at the end of a buffer when running multithreaded, error:
			throw InvalidInputException("Error in file \"%s\" on line %s: unterminated quotes. (%s)", options.file_path,
			                            GetLineNumberStr(linenr, linenr_estimated, buffer->local_batch_index).c_str(),
			                            options.ToString());
		} else {
			goto final_state;
		}
	} else {
		position_buffer--;
		goto in_quotes;
	}

unquote : {
	/* state: unquote: this state handles the state directly after we unquote*/
	//
	// in this state we expect either another quote (entering the quoted state again, and escaping the quote)
	// or a delimiter/newline, ending the current value and moving on to the next value
	position_buffer++;
	if (!BufferRemainder()) {
		offset = 1;
		goto final_state;
	}
	auto c = (*buffer)[position_buffer];
	if (c == options.quote[0] && (options.escape.empty() || options.escape[0] == options.quote[0])) {
		// escaped quote, return to quoted state and store escape position
		escape_positions.push_back(position_buffer - start_buffer);
		goto in_quotes;
	} else if (c == options.delimiter[0]) {
		// delimiter, add value
		offset = 1;
		goto add_value;
	} else if (StringUtil::CharacterIsNewline(c)) {
		offset = 1;
		// FIXME: should this be an assertion?
		D_ASSERT(try_add_line || (!try_add_line && column == parse_chunk.ColumnCount() - 1));
		goto add_row;
	} else if (position_buffer >= end_buffer) {
		// reached end of buffer
		offset = 1;
		goto final_state;
	} else {
		error_message = StringUtil::Format(
		    "Error in file \"%s\" on line %s: quote should be followed by end of value, end of "
		    "row or another quote. (%s). ",
		    options.file_path, GetLineNumberStr(linenr, linenr_estimated, buffer->local_batch_index).c_str(),
		    options.ToString());
		return false;
	}
}
handle_escape : {
	/* state: handle_escape */
	// escape should be followed by a quote or another escape character
	position_buffer++;
	if (!BufferRemainder()) {
		goto final_state;
	}
	if (position_buffer >= buffer_size && buffer->buffer->IsCSVFileLastBuffer()) {
		error_message = StringUtil::Format(
		    "Error in file \"%s\" on line %s: neither QUOTE nor ESCAPE is proceeded by ESCAPE. (%s)", options.file_path,
		    GetLineNumberStr(linenr, linenr_estimated, buffer->local_batch_index).c_str(), options.ToString());
		return false;
	}
	if ((*buffer)[position_buffer] != options.quote[0] && (*buffer)[position_buffer] != options.escape[0]) {
		error_message = StringUtil::Format(
		    "Error in file \"%s\" on line %s: neither QUOTE nor ESCAPE is proceeded by ESCAPE. (%s)", options.file_path,
		    GetLineNumberStr(linenr, linenr_estimated, buffer->local_batch_index).c_str(), options.ToString());
		return false;
	}
	// escape was followed by quote or escape, go back to quoted state
	goto in_quotes;
}
final_state : {
	/* state: final_stage reached after we finished reading the end_buffer of the csv buffer */
	// reset end buffer
	end_buffer = buffer->buffer_end;
	if (position_buffer == end_buffer) {
		reached_remainder_state = false;
	}
	if (finished_chunk) {
		if (position_buffer >= end_buffer) {
			if (position_buffer == end_buffer && StringUtil::CharacterIsNewline((*buffer)[position_buffer - 1]) &&
			    position_buffer < buffer_size) {
				// last position is a new line, we still have to go through one more line of this buffer
				finished = false;
			} else {
				finished = true;
			}
		}
		buffer->lines_read += insert_chunk.size();
		return true;
	}
	// If this is the last buffer, we have to read the last value
	if (buffer->buffer->IsCSVFileLastBuffer() || (buffer->next_buffer && buffer->next_buffer->IsCSVFileLastBuffer())) {
		if (column > 0 || start_buffer != position_buffer || try_add_line ||
		    (insert_chunk.data.size() == 1 && start_buffer != position_buffer)) {
			// remaining values to be added to the chunk
			auto str_value = buffer->GetValue(start_buffer, position_buffer, offset);
			if (!AllNewLine(str_value, insert_chunk.data.size()) || offset == 0) {
				AddValue(str_value, column, escape_positions, has_quotes, buffer->local_batch_index);
				if (try_add_line) {
					bool success = column == return_types.size();
					if (success) {
						auto cur_linenr = linenr;
						AddRow(insert_chunk, column, error_message, buffer->local_batch_index);
						success = Flush(insert_chunk, buffer->local_batch_index);
						linenr = cur_linenr;
					}
					parse_chunk.Reset();
					reached_remainder_state = false;
					return success;
				} else {
					VerifyLineLength(position_buffer - line_start);
					line_start = position_buffer;
					AddRow(insert_chunk, column, error_message, buffer->local_batch_index);
					verification_positions.end_of_last_line = position_buffer;
				}
			}
		}
	}
	// flush the parsed chunk and finalize parsing
	if (mode == ParserMode::PARSING) {
		Flush(insert_chunk, buffer->local_batch_index);
		buffer->lines_read += insert_chunk.size();
	}
	if (position_buffer - verification_positions.end_of_last_line > options.buffer_size) {
		error_message = "Line does not fit in one buffer. Increase the buffer size.";
		return false;
	}
	end_buffer = buffer_size;
	SkipEmptyLines();
	end_buffer = buffer->buffer_end;
	verification_positions.end_of_last_line = position_buffer;
	if (position_buffer >= end_buffer) {
		if (position_buffer >= end_buffer) {
			if (position_buffer == end_buffer && StringUtil::CharacterIsNewline((*buffer)[position_buffer - 1]) &&
			    position_buffer < buffer_size) {
				// last position is a new line, we still have to go through one more line of this buffer
				finished = false;
			} else {
				finished = true;
			}
		}
	}
	return true;
};
}

void ParallelCSVReader::ParseCSV(DataChunk &insert_chunk) {
	string error_message;
	if (!TryParseCSV(ParserMode::PARSING, insert_chunk, error_message)) {
		throw InvalidInputException(error_message);
	}
}

idx_t ParallelCSVReader::GetLineError(idx_t line_error, idx_t buffer_idx) {
	while (true) {
		if (buffer->line_info->CanItGetLine(file_idx, buffer_idx)) {
			auto cur_start = verification_positions.beginning_of_first_line + buffer->buffer->GetCSVGlobalStart();
			// line errors are 1-indexed
			return buffer->line_info->GetLine(buffer_idx, line_error, file_idx, cur_start, false);
		}
	}
}

bool ParallelCSVReader::TryParseCSV(ParserMode mode) {
	DataChunk dummy_chunk;
	string error_message;
	return TryParseCSV(mode, dummy_chunk, error_message);
}

void ParallelCSVReader::ParseCSV(ParserMode mode) {
	DataChunk dummy_chunk;
	string error_message;
	if (!TryParseCSV(mode, dummy_chunk, error_message)) {
		throw InvalidInputException(error_message);
	}
}

bool ParallelCSVReader::TryParseCSV(ParserMode parser_mode, DataChunk &insert_chunk, string &error_message) {
	mode = parser_mode;
	return TryParseSimpleCSV(insert_chunk, error_message);
}

} // namespace duckdb






#include <algorithm>

namespace duckdb {

PhysicalBatchCopyToFile::PhysicalBatchCopyToFile(vector<LogicalType> types, CopyFunction function_p,
                                                 unique_ptr<FunctionData> bind_data_p, idx_t estimated_cardinality)
    : PhysicalOperator(PhysicalOperatorType::BATCH_COPY_TO_FILE, std::move(types), estimated_cardinality),
      function(std::move(function_p)), bind_data(std::move(bind_data_p)) {
	if (!function.flush_batch || !function.prepare_batch) {
		throw InternalException(
		    "PhysicalBatchCopyToFile created for copy function that does not have prepare_batch/flush_batch defined");
	}
}

//===--------------------------------------------------------------------===//
// States
//===--------------------------------------------------------------------===//
class BatchCopyToGlobalState : public GlobalSinkState {
public:
	explicit BatchCopyToGlobalState(unique_ptr<GlobalFunctionData> global_state)
	    : rows_copied(0), global_state(std::move(global_state)), any_flushing(false) {
	}

	mutex lock;
	//! The total number of rows copied to the file
	atomic<idx_t> rows_copied;
	//! Global copy state
	unique_ptr<GlobalFunctionData> global_state;
	//! The prepared batch data by batch index - ready to flush
	map<idx_t, unique_ptr<PreparedBatchData>> batch_data;
	//! Lock for flushing to disk
	mutex flush_lock;
	//! Whether or not any threads are flushing (only one thread can flush at a time)
	atomic<bool> any_flushing;

	void AddBatchData(idx_t batch_index, unique_ptr<PreparedBatchData> new_batch) {
		// move the batch data to the set of prepared batch data
		lock_guard<mutex> l(lock);
		auto entry = batch_data.insert(make_pair(batch_index, std::move(new_batch)));
		if (!entry.second) {
			throw InternalException("Duplicate batch index %llu encountered in PhysicalBatchCopyToFile", batch_index);
		}
	}
};

class BatchCopyToLocalState : public LocalSinkState {
public:
	explicit BatchCopyToLocalState(unique_ptr<LocalFunctionData> local_state_p)
	    : local_state(std::move(local_state_p)), rows_copied(0) {
	}

	//! Local copy state
	unique_ptr<LocalFunctionData> local_state;
	//! The current collection we are appending to
	unique_ptr<ColumnDataCollection> collection;
	//! The append state of the collection
	ColumnDataAppendState append_state;
	//! How many rows have been copied in total
	idx_t rows_copied;
	//! The current batch index
	optional_idx batch_index;

	void InitializeCollection(ClientContext &context, const PhysicalOperator &op) {
		collection = make_uniq<ColumnDataCollection>(Allocator::Get(context), op.children[0]->types);
		collection->InitializeAppend(append_state);
	}
};

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
SinkResultType PhysicalBatchCopyToFile::Sink(ExecutionContext &context, DataChunk &chunk,
                                             OperatorSinkInput &input) const {
	auto &state = input.local_state.Cast<BatchCopyToLocalState>();
	if (!state.collection) {
		state.InitializeCollection(context.client, *this);
		state.batch_index = state.partition_info.batch_index.GetIndex();
	}
	state.rows_copied += chunk.size();
	state.collection->Append(state.append_state, chunk);
	return SinkResultType::NEED_MORE_INPUT;
}

void PhysicalBatchCopyToFile::Combine(ExecutionContext &context, GlobalSinkState &gstate_p,
                                      LocalSinkState &lstate) const {
	auto &state = lstate.Cast<BatchCopyToLocalState>();
	auto &gstate = gstate_p.Cast<BatchCopyToGlobalState>();
	gstate.rows_copied += state.rows_copied;
}

//===--------------------------------------------------------------------===//
// Finalize
//===--------------------------------------------------------------------===//
SinkFinalizeType PhysicalBatchCopyToFile::FinalFlush(ClientContext &context, GlobalSinkState &gstate_p) const {
	auto &gstate = gstate_p.Cast<BatchCopyToGlobalState>();
	idx_t min_batch_index = idx_t(NumericLimits<int64_t>::Maximum());
	FlushBatchData(context, gstate_p, min_batch_index);
	if (function.copy_to_finalize) {
		function.copy_to_finalize(context, *bind_data, *gstate.global_state);

		if (use_tmp_file) {
			PhysicalCopyToFile::MoveTmpFile(context, file_path);
		}
	}
	return SinkFinalizeType::READY;
}

SinkFinalizeType PhysicalBatchCopyToFile::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                                   GlobalSinkState &gstate_p) const {
	FinalFlush(context, gstate_p);
	return SinkFinalizeType::READY;
}

//===--------------------------------------------------------------------===//
// Batch Data Handling
//===--------------------------------------------------------------------===//
void PhysicalBatchCopyToFile::PrepareBatchData(ClientContext &context, GlobalSinkState &gstate_p, idx_t batch_index,
                                               unique_ptr<ColumnDataCollection> collection) const {
	auto &gstate = gstate_p.Cast<BatchCopyToGlobalState>();

	// prepare the batch
	auto batch_data = function.prepare_batch(context, *bind_data, *gstate.global_state, std::move(collection));
	gstate.AddBatchData(batch_index, std::move(batch_data));
}

void PhysicalBatchCopyToFile::FlushBatchData(ClientContext &context, GlobalSinkState &gstate_p, idx_t min_index) const {
	auto &gstate = gstate_p.Cast<BatchCopyToGlobalState>();

	// flush batch data to disk (if there are any to flush)
	// grab the flush lock - we can only call flush_batch with this lock
	// otherwise the data might end up in the wrong order
	{
		lock_guard<mutex> l(gstate.flush_lock);
		if (gstate.any_flushing) {
			return;
		}
		gstate.any_flushing = true;
	}
	ActiveFlushGuard active_flush(gstate.any_flushing);
	while (true) {
		unique_ptr<PreparedBatchData> batch_data;
		{
			// fetch the next batch to flush (if any)
			lock_guard<mutex> l(gstate.lock);
			if (gstate.batch_data.empty()) {
				// no batch data left to flush
				break;
			}
			auto entry = gstate.batch_data.begin();
			if (entry->first >= min_index) {
				// this data is past the min_index - we cannot write it yet
				break;
			}
			if (!entry->second) {
				// this batch is in process of being prepared but is not ready yet
				break;
			}
			batch_data = std::move(entry->second);
			gstate.batch_data.erase(entry);
		}
		function.flush_batch(context, *bind_data, *gstate.global_state, *batch_data);
	}
}

//===--------------------------------------------------------------------===//
// Next Batch
//===--------------------------------------------------------------------===//
void PhysicalBatchCopyToFile::NextBatch(ExecutionContext &context, GlobalSinkState &gstate_p,
                                        LocalSinkState &lstate) const {
	auto &state = lstate.Cast<BatchCopyToLocalState>();
	if (state.collection && state.collection->Count() > 0) {
		// we finished processing this batch
		// start flushing data
		auto min_batch_index = lstate.partition_info.min_batch_index.GetIndex();
		PrepareBatchData(context.client, gstate_p, state.batch_index.GetIndex(), std::move(state.collection));
		FlushBatchData(context.client, gstate_p, min_batch_index);
	}
	state.batch_index = lstate.partition_info.batch_index.GetIndex();

	state.InitializeCollection(context.client, *this);
}

unique_ptr<LocalSinkState> PhysicalBatchCopyToFile::GetLocalSinkState(ExecutionContext &context) const {
	return make_uniq<BatchCopyToLocalState>(function.copy_to_initialize_local(context, *bind_data));
}

unique_ptr<GlobalSinkState> PhysicalBatchCopyToFile::GetGlobalSinkState(ClientContext &context) const {
	return make_uniq<BatchCopyToGlobalState>(function.copy_to_initialize_global(context, *bind_data, file_path));
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
SourceResultType PhysicalBatchCopyToFile::GetData(ExecutionContext &context, DataChunk &chunk,
                                                  OperatorSourceInput &input) const {
	auto &g = sink_state->Cast<BatchCopyToGlobalState>();

	chunk.SetCardinality(1);
	chunk.SetValue(0, 0, Value::BIGINT(g.rows_copied));
	return SourceResultType::FINISHED;
}

} // namespace duckdb











namespace duckdb {

PhysicalBatchInsert::PhysicalBatchInsert(vector<LogicalType> types, TableCatalogEntry &table,
                                         physical_index_vector_t<idx_t> column_index_map,
                                         vector<unique_ptr<Expression>> bound_defaults, idx_t estimated_cardinality)
    : PhysicalOperator(PhysicalOperatorType::BATCH_INSERT, std::move(types), estimated_cardinality),
      column_index_map(std::move(column_index_map)), insert_table(&table), insert_types(table.GetTypes()),
      bound_defaults(std::move(bound_defaults)) {
}

PhysicalBatchInsert::PhysicalBatchInsert(LogicalOperator &op, SchemaCatalogEntry &schema,
                                         unique_ptr<BoundCreateTableInfo> info_p, idx_t estimated_cardinality)
    : PhysicalOperator(PhysicalOperatorType::BATCH_CREATE_TABLE_AS, op.types, estimated_cardinality),
      insert_table(nullptr), schema(&schema), info(std::move(info_p)) {
	PhysicalInsert::GetInsertInfo(*info, insert_types, bound_defaults);
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//

class CollectionMerger {
public:
	explicit CollectionMerger(ClientContext &context) : context(context) {
	}

	ClientContext &context;
	vector<unique_ptr<RowGroupCollection>> current_collections;

public:
	void AddCollection(unique_ptr<RowGroupCollection> collection) {
		current_collections.push_back(std::move(collection));
	}

	bool Empty() {
		return current_collections.empty();
	}

	unique_ptr<RowGroupCollection> Flush(OptimisticDataWriter &writer) {
		if (Empty()) {
			return nullptr;
		}
		unique_ptr<RowGroupCollection> new_collection = std::move(current_collections[0]);
		if (current_collections.size() > 1) {
			// we have gathered multiple collections: create one big collection and merge that
			auto &types = new_collection->GetTypes();
			TableAppendState append_state;
			new_collection->InitializeAppend(append_state);

			DataChunk scan_chunk;
			scan_chunk.Initialize(context, types);

			vector<column_t> column_ids;
			for (idx_t i = 0; i < types.size(); i++) {
				column_ids.push_back(i);
			}
			for (auto &collection : current_collections) {
				if (!collection) {
					continue;
				}
				TableScanState scan_state;
				scan_state.Initialize(column_ids);
				collection->InitializeScan(scan_state.local_state, column_ids, nullptr);

				while (true) {
					scan_chunk.Reset();
					scan_state.local_state.ScanCommitted(scan_chunk, TableScanType::TABLE_SCAN_COMMITTED_ROWS);
					if (scan_chunk.size() == 0) {
						break;
					}
					auto new_row_group = new_collection->Append(scan_chunk, append_state);
					if (new_row_group) {
						writer.WriteNewRowGroup(*new_collection);
					}
				}
			}
			new_collection->FinalizeAppend(TransactionData(0, 0), append_state);
			writer.WriteLastRowGroup(*new_collection);
		}
		current_collections.clear();
		return new_collection;
	}
};

enum class RowGroupBatchType : uint8_t { FLUSHED, NOT_FLUSHED };
struct RowGroupBatchEntry {
	RowGroupBatchEntry(idx_t batch_idx, unique_ptr<RowGroupCollection> collection_p, RowGroupBatchType type)
	    : batch_idx(batch_idx), total_rows(collection_p->GetTotalRows()), collection(std::move(collection_p)),
	      type(type) {
	}

	idx_t batch_idx;
	idx_t total_rows;
	unique_ptr<RowGroupCollection> collection;
	RowGroupBatchType type;
};

class BatchInsertGlobalState : public GlobalSinkState {
public:
	static constexpr const idx_t BATCH_FLUSH_THRESHOLD = LocalStorage::MERGE_THRESHOLD * 3;

public:
	explicit BatchInsertGlobalState(DuckTableEntry &table) : table(table), insert_count(0) {
	}

	mutex lock;
	DuckTableEntry &table;
	idx_t insert_count;
	vector<RowGroupBatchEntry> collections;
	idx_t next_start = 0;

	void FindMergeCollections(idx_t min_batch_index, optional_idx &merged_batch_index,
	                          vector<unique_ptr<RowGroupCollection>> &result) {
		bool merge = false;
		idx_t start_index = next_start;
		idx_t current_idx;
		idx_t total_count = 0;
		for (current_idx = start_index; current_idx < collections.size(); current_idx++) {
			auto &entry = collections[current_idx];
			if (entry.batch_idx >= min_batch_index) {
				// this entry is AFTER the min_batch_index
				// we might still find new entries!
				break;
			}
			if (entry.type == RowGroupBatchType::FLUSHED) {
				// already flushed: cannot flush anything here
				if (total_count > 0) {
					merge = true;
					break;
				}
				start_index = current_idx + 1;
				if (start_index > next_start) {
					// avoid checking this segment again in the future
					next_start = start_index;
				}
				total_count = 0;
				continue;
			}
			// not flushed - add to set of indexes to flush
			total_count += entry.total_rows;
			if (total_count >= BATCH_FLUSH_THRESHOLD) {
				merge = true;
				break;
			}
		}
		if (merge && total_count > 0) {
			D_ASSERT(current_idx > start_index);
			merged_batch_index = collections[start_index].batch_idx;
			for (idx_t idx = start_index; idx < current_idx; idx++) {
				auto &entry = collections[idx];
				if (!entry.collection || entry.type == RowGroupBatchType::FLUSHED) {
					throw InternalException("Adding a row group collection that should not be flushed");
				}
				result.push_back(std::move(entry.collection));
				entry.total_rows = total_count;
				entry.type = RowGroupBatchType::FLUSHED;
			}
			if (start_index + 1 < current_idx) {
				// erase all entries except the first one
				collections.erase(collections.begin() + start_index + 1, collections.begin() + current_idx);
			}
		}
	}

	unique_ptr<RowGroupCollection> MergeCollections(ClientContext &context,
	                                                vector<unique_ptr<RowGroupCollection>> merge_collections,
	                                                OptimisticDataWriter &writer) {
		CollectionMerger merger(context);
		for (auto &collection : merge_collections) {
			merger.AddCollection(std::move(collection));
		}
		return merger.Flush(writer);
	}

	void AddCollection(ClientContext &context, idx_t batch_index, idx_t min_batch_index,
	                   unique_ptr<RowGroupCollection> current_collection,
	                   optional_ptr<OptimisticDataWriter> writer = nullptr,
	                   optional_ptr<bool> written_to_disk = nullptr) {
		if (batch_index < min_batch_index) {
			throw InternalException(
			    "Batch index of the added collection (%llu) is smaller than the min batch index (%llu)", batch_index,
			    min_batch_index);
		}
		auto new_count = current_collection->GetTotalRows();
		auto batch_type =
		    new_count < RowGroup::ROW_GROUP_SIZE ? RowGroupBatchType::NOT_FLUSHED : RowGroupBatchType::FLUSHED;
		if (batch_type == RowGroupBatchType::FLUSHED && writer) {
			writer->WriteLastRowGroup(*current_collection);
		}
		optional_idx merged_batch_index;
		vector<unique_ptr<RowGroupCollection>> merge_collections;
		{
			lock_guard<mutex> l(lock);
			insert_count += new_count;

			// add the collection to the batch index
			RowGroupBatchEntry new_entry(batch_index, std::move(current_collection), batch_type);

			auto it = std::lower_bound(
			    collections.begin(), collections.end(), new_entry,
			    [&](const RowGroupBatchEntry &a, const RowGroupBatchEntry &b) { return a.batch_idx < b.batch_idx; });
			if (it != collections.end() && it->batch_idx == new_entry.batch_idx) {
				throw InternalException(
				    "PhysicalBatchInsert::AddCollection error: batch index %d is present in multiple "
				    "collections. This occurs when "
				    "batch indexes are not uniquely distributed over threads",
				    batch_index);
			}
			collections.insert(it, std::move(new_entry));
			if (writer) {
				FindMergeCollections(min_batch_index, merged_batch_index, merge_collections);
			}
		}
		if (!merge_collections.empty()) {
			// merge together the collections
			D_ASSERT(writer);
			auto final_collection = MergeCollections(context, std::move(merge_collections), *writer);
			if (written_to_disk) {
				*written_to_disk = true;
			}
			// add the merged-together collection to the set of batch indexes
			{
				lock_guard<mutex> l(lock);
				RowGroupBatchEntry new_entry(merged_batch_index.GetIndex(), std::move(final_collection),
				                             RowGroupBatchType::FLUSHED);
				auto it = std::lower_bound(collections.begin(), collections.end(), new_entry,
				                           [&](const RowGroupBatchEntry &a, const RowGroupBatchEntry &b) {
					                           return a.batch_idx < b.batch_idx;
				                           });
				if (it->batch_idx != merged_batch_index.GetIndex()) {
					throw InternalException("Merged batch index was no longer present in collection");
				}
				it->collection = std::move(new_entry.collection);
			}
		}
	}
};

class BatchInsertLocalState : public LocalSinkState {
public:
	BatchInsertLocalState(ClientContext &context, const vector<LogicalType> &types,
	                      const vector<unique_ptr<Expression>> &bound_defaults)
	    : default_executor(context, bound_defaults), written_to_disk(false) {
		insert_chunk.Initialize(Allocator::Get(context), types);
	}

	DataChunk insert_chunk;
	ExpressionExecutor default_executor;
	idx_t current_index;
	TableAppendState current_append_state;
	unique_ptr<RowGroupCollection> current_collection;
	optional_ptr<OptimisticDataWriter> writer;
	bool written_to_disk;

	void CreateNewCollection(DuckTableEntry &table, const vector<LogicalType> &insert_types) {
		auto &table_info = table.GetStorage().info;
		auto &block_manager = TableIOManager::Get(table.GetStorage()).GetBlockManagerForRowData();
		current_collection = make_uniq<RowGroupCollection>(table_info, block_manager, insert_types, MAX_ROW_ID);
		current_collection->InitializeEmpty();
		current_collection->InitializeAppend(current_append_state);
		written_to_disk = false;
	}
};

unique_ptr<GlobalSinkState> PhysicalBatchInsert::GetGlobalSinkState(ClientContext &context) const {
	optional_ptr<TableCatalogEntry> table;
	if (info) {
		// CREATE TABLE AS
		D_ASSERT(!insert_table);
		auto &catalog = schema->catalog;
		table = (TableCatalogEntry *)catalog
		            .CreateTable(catalog.GetCatalogTransaction(context), *schema.get_mutable(), *info)
		            .get();
	} else {
		D_ASSERT(insert_table);
		D_ASSERT(insert_table->IsDuckTable());
		table = insert_table.get_mutable();
	}
	auto result = make_uniq<BatchInsertGlobalState>(table->Cast<DuckTableEntry>());
	return std::move(result);
}

unique_ptr<LocalSinkState> PhysicalBatchInsert::GetLocalSinkState(ExecutionContext &context) const {
	return make_uniq<BatchInsertLocalState>(context.client, insert_types, bound_defaults);
}

void PhysicalBatchInsert::NextBatch(ExecutionContext &context, GlobalSinkState &state, LocalSinkState &lstate_p) const {
	auto &gstate = state.Cast<BatchInsertGlobalState>();
	auto &lstate = lstate_p.Cast<BatchInsertLocalState>();

	auto &table = gstate.table;
	auto batch_index = lstate.partition_info.batch_index.GetIndex();
	if (lstate.current_collection) {
		if (lstate.current_index == batch_index) {
			throw InternalException("NextBatch called with the same batch index?");
		}
		// batch index has changed: move the old collection to the global state and create a new collection
		TransactionData tdata(0, 0);
		lstate.current_collection->FinalizeAppend(tdata, lstate.current_append_state);
		gstate.AddCollection(context.client, lstate.current_index, lstate.partition_info.min_batch_index.GetIndex(),
		                     std::move(lstate.current_collection), lstate.writer, &lstate.written_to_disk);
		lstate.CreateNewCollection(table, insert_types);
	}
	lstate.current_index = batch_index;
}

SinkResultType PhysicalBatchInsert::Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const {
	auto &gstate = input.global_state.Cast<BatchInsertGlobalState>();
	auto &lstate = input.local_state.Cast<BatchInsertLocalState>();

	auto &table = gstate.table;
	PhysicalInsert::ResolveDefaults(table, chunk, column_index_map, lstate.default_executor, lstate.insert_chunk);

	auto batch_index = lstate.partition_info.batch_index.GetIndex();
	if (!lstate.current_collection) {
		lock_guard<mutex> l(gstate.lock);
		// no collection yet: create a new one
		lstate.CreateNewCollection(table, insert_types);
		lstate.writer = &table.GetStorage().CreateOptimisticWriter(context.client);
	} else if (lstate.current_index != batch_index) {
		throw InternalException("Current batch differs from batch - but NextBatch was not called!?");
	}
	lstate.current_index = batch_index;

	table.GetStorage().VerifyAppendConstraints(table, context.client, lstate.insert_chunk);

	auto new_row_group = lstate.current_collection->Append(lstate.insert_chunk, lstate.current_append_state);
	if (new_row_group) {
		// we have already written to disk - flush the next row group as well
		lstate.writer->WriteNewRowGroup(*lstate.current_collection);
		lstate.written_to_disk = true;
	}
	return SinkResultType::NEED_MORE_INPUT;
}

void PhysicalBatchInsert::Combine(ExecutionContext &context, GlobalSinkState &gstate_p,
                                  LocalSinkState &lstate_p) const {
	auto &gstate = gstate_p.Cast<BatchInsertGlobalState>();
	auto &lstate = lstate_p.Cast<BatchInsertLocalState>();
	auto &client_profiler = QueryProfiler::Get(context.client);
	context.thread.profiler.Flush(*this, lstate.default_executor, "default_executor", 1);
	client_profiler.Flush(context.thread.profiler);

	if (!lstate.current_collection) {
		return;
	}

	if (lstate.current_collection->GetTotalRows() > 0) {
		TransactionData tdata(0, 0);
		lstate.current_collection->FinalizeAppend(tdata, lstate.current_append_state);
		gstate.AddCollection(context.client, lstate.current_index, lstate.partition_info.min_batch_index.GetIndex(),
		                     std::move(lstate.current_collection));
	}
	{
		lock_guard<mutex> l(gstate.lock);
		gstate.table.GetStorage().FinalizeOptimisticWriter(context.client, *lstate.writer);
	}
}

SinkFinalizeType PhysicalBatchInsert::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                               GlobalSinkState &gstate_p) const {
	auto &gstate = gstate_p.Cast<BatchInsertGlobalState>();

	// in the finalize, do a final pass over all of the collections we created and try to merge smaller collections
	// together
	vector<unique_ptr<CollectionMerger>> mergers;
	unique_ptr<CollectionMerger> current_merger;

	auto &storage = gstate.table.GetStorage();
	for (auto &entry : gstate.collections) {
		if (entry.type == RowGroupBatchType::NOT_FLUSHED) {
			// this collection has not been flushed: add it to the merge set
			if (!current_merger) {
				current_merger = make_uniq<CollectionMerger>(context);
			}
			current_merger->AddCollection(std::move(entry.collection));
		} else {
			// this collection has been flushed: it does not need to be merged
			// create a separate collection merger only for this entry
			if (current_merger) {
				// we have small collections remaining: flush them
				mergers.push_back(std::move(current_merger));
				current_merger.reset();
			}
			auto larger_merger = make_uniq<CollectionMerger>(context);
			larger_merger->AddCollection(std::move(entry.collection));
			mergers.push_back(std::move(larger_merger));
		}
	}
	if (current_merger) {
		mergers.push_back(std::move(current_merger));
	}

	// now that we have created all of the mergers, perform the actual merging
	vector<unique_ptr<RowGroupCollection>> final_collections;
	final_collections.reserve(mergers.size());
	auto &writer = storage.CreateOptimisticWriter(context);
	for (auto &merger : mergers) {
		final_collections.push_back(merger->Flush(writer));
	}
	storage.FinalizeOptimisticWriter(context, writer);

	// finally, merge the row groups into the local storage
	for (auto &collection : final_collections) {
		storage.LocalMerge(context, *collection);
	}
	return SinkFinalizeType::READY;
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//

SourceResultType PhysicalBatchInsert::GetData(ExecutionContext &context, DataChunk &chunk,
                                              OperatorSourceInput &input) const {
	auto &insert_gstate = sink_state->Cast<BatchInsertGlobalState>();

	chunk.SetCardinality(1);
	chunk.SetValue(0, 0, Value::BIGINT(insert_gstate.insert_count));

	return SourceResultType::FINISHED;
}

} // namespace duckdb








#include <algorithm>

namespace duckdb {

class CopyToFunctionGlobalState : public GlobalSinkState {
public:
	explicit CopyToFunctionGlobalState(unique_ptr<GlobalFunctionData> global_state)
	    : rows_copied(0), last_file_offset(0), global_state(std::move(global_state)) {
	}
	mutex lock;
	idx_t rows_copied;
	idx_t last_file_offset;
	unique_ptr<GlobalFunctionData> global_state;

	//! shared state for HivePartitionedColumnData
	shared_ptr<GlobalHivePartitionState> partition_state;
};

class CopyToFunctionLocalState : public LocalSinkState {
public:
	explicit CopyToFunctionLocalState(unique_ptr<LocalFunctionData> local_state)
	    : local_state(std::move(local_state)), writer_offset(0) {
	}
	unique_ptr<GlobalFunctionData> global_state;
	unique_ptr<LocalFunctionData> local_state;

	//! Buffers the tuples in partitions before writing
	unique_ptr<HivePartitionedColumnData> part_buffer;
	unique_ptr<PartitionedColumnDataAppendState> part_buffer_append_state;

	idx_t writer_offset;
};

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//

void PhysicalCopyToFile::MoveTmpFile(ClientContext &context, const string &tmp_file_path) {
	auto &fs = FileSystem::GetFileSystem(context);
	auto file_path = tmp_file_path.substr(0, tmp_file_path.length() - 4);
	if (fs.FileExists(file_path)) {
		fs.RemoveFile(file_path);
	}
	fs.MoveFile(tmp_file_path, file_path);
}

PhysicalCopyToFile::PhysicalCopyToFile(vector<LogicalType> types, CopyFunction function_p,
                                       unique_ptr<FunctionData> bind_data, idx_t estimated_cardinality)
    : PhysicalOperator(PhysicalOperatorType::COPY_TO_FILE, std::move(types), estimated_cardinality),
      function(std::move(function_p)), bind_data(std::move(bind_data)), parallel(false) {
}

SinkResultType PhysicalCopyToFile::Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const {
	auto &g = input.global_state.Cast<CopyToFunctionGlobalState>();
	auto &l = input.local_state.Cast<CopyToFunctionLocalState>();

	if (partition_output) {
		l.part_buffer->Append(*l.part_buffer_append_state, chunk);
		return SinkResultType::NEED_MORE_INPUT;
	}

	{
		lock_guard<mutex> glock(g.lock);
		g.rows_copied += chunk.size();
	}
	function.copy_to_sink(context, *bind_data, per_thread_output ? *l.global_state : *g.global_state, *l.local_state,
	                      chunk);
	return SinkResultType::NEED_MORE_INPUT;
}

static void CreateDir(const string &dir_path, FileSystem &fs) {
	if (!fs.DirectoryExists(dir_path)) {
		fs.CreateDirectory(dir_path);
	}
}

static string CreateDirRecursive(const vector<idx_t> &cols, const vector<string> &names, const vector<Value> &values,
                                 string path, FileSystem &fs) {
	CreateDir(path, fs);

	for (idx_t i = 0; i < cols.size(); i++) {
		const auto &partition_col_name = names[cols[i]];
		const auto &partition_value = values[i];
		string p_dir = partition_col_name + "=" + partition_value.ToString();
		path = fs.JoinPath(path, p_dir);
		CreateDir(path, fs);
	}

	return path;
}

void PhysicalCopyToFile::Combine(ExecutionContext &context, GlobalSinkState &gstate, LocalSinkState &lstate) const {
	auto &g = gstate.Cast<CopyToFunctionGlobalState>();
	auto &l = lstate.Cast<CopyToFunctionLocalState>();

	if (partition_output) {
		auto &fs = FileSystem::GetFileSystem(context.client);
		l.part_buffer->FlushAppendState(*l.part_buffer_append_state);
		auto &partitions = l.part_buffer->GetPartitions();
		auto partition_key_map = l.part_buffer->GetReverseMap();

		string trimmed_path = file_path;
		StringUtil::RTrim(trimmed_path, fs.PathSeparator());

		for (idx_t i = 0; i < partitions.size(); i++) {
			string hive_path =
			    CreateDirRecursive(partition_columns, names, partition_key_map[i]->values, trimmed_path, fs);
			string full_path(filename_pattern.CreateFilename(fs, hive_path, function.extension, l.writer_offset));
			if (fs.FileExists(full_path) && !overwrite_or_ignore) {
				throw IOException("failed to create " + full_path +
				                  ", file exists! Enable OVERWRITE_OR_IGNORE option to force writing");
			}
			// Create a writer for the current file
			auto fun_data_global = function.copy_to_initialize_global(context.client, *bind_data, full_path);
			auto fun_data_local = function.copy_to_initialize_local(context, *bind_data);

			for (auto &chunk : partitions[i]->Chunks()) {
				function.copy_to_sink(context, *bind_data, *fun_data_global, *fun_data_local, chunk);
			}

			function.copy_to_combine(context, *bind_data, *fun_data_global, *fun_data_local);
			function.copy_to_finalize(context.client, *bind_data, *fun_data_global);
		}

		return;
	}

	if (function.copy_to_combine) {
		function.copy_to_combine(context, *bind_data, per_thread_output ? *l.global_state : *g.global_state,
		                         *l.local_state);

		if (per_thread_output) {
			function.copy_to_finalize(context.client, *bind_data, *l.global_state);
		}
	}
}

SinkFinalizeType PhysicalCopyToFile::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                              GlobalSinkState &gstate_p) const {
	auto &gstate = gstate_p.Cast<CopyToFunctionGlobalState>();
	if (per_thread_output || partition_output) {
		// already happened in combine
		return SinkFinalizeType::READY;
	}
	if (function.copy_to_finalize) {
		function.copy_to_finalize(context, *bind_data, *gstate.global_state);

		if (use_tmp_file) {
			D_ASSERT(!per_thread_output); // FIXME
			D_ASSERT(!partition_output);  // FIXME
			MoveTmpFile(context, file_path);
		}
	}
	return SinkFinalizeType::READY;
}

unique_ptr<LocalSinkState> PhysicalCopyToFile::GetLocalSinkState(ExecutionContext &context) const {
	if (partition_output) {
		auto state = make_uniq<CopyToFunctionLocalState>(nullptr);
		{
			auto &g = sink_state->Cast<CopyToFunctionGlobalState>();
			lock_guard<mutex> glock(g.lock);
			state->writer_offset = g.last_file_offset++;

			state->part_buffer = make_uniq<HivePartitionedColumnData>(context.client, expected_types, partition_columns,
			                                                          g.partition_state);
			state->part_buffer_append_state = make_uniq<PartitionedColumnDataAppendState>();
			state->part_buffer->InitializeAppendState(*state->part_buffer_append_state);
		}
		return std::move(state);
	}
	auto res = make_uniq<CopyToFunctionLocalState>(function.copy_to_initialize_local(context, *bind_data));
	if (per_thread_output) {
		idx_t this_file_offset;
		{
			auto &g = sink_state->Cast<CopyToFunctionGlobalState>();
			lock_guard<mutex> glock(g.lock);
			this_file_offset = g.last_file_offset++;
		}
		auto &fs = FileSystem::GetFileSystem(context.client);
		string output_path(filename_pattern.CreateFilename(fs, file_path, function.extension, this_file_offset));
		if (fs.FileExists(output_path) && !overwrite_or_ignore) {
			throw IOException("%s exists! Enable OVERWRITE_OR_IGNORE option to force writing", output_path);
		}
		res->global_state = function.copy_to_initialize_global(context.client, *bind_data, output_path);
	}
	return std::move(res);
}

unique_ptr<GlobalSinkState> PhysicalCopyToFile::GetGlobalSinkState(ClientContext &context) const {

	if (partition_output || per_thread_output) {
		auto &fs = FileSystem::GetFileSystem(context);

		if (fs.FileExists(file_path) && !overwrite_or_ignore) {
			throw IOException("%s exists! Enable OVERWRITE_OR_IGNORE option to force writing", file_path);
		}
		if (!fs.DirectoryExists(file_path)) {
			fs.CreateDirectory(file_path);
		} else if (!overwrite_or_ignore) {
			idx_t n_files = 0;
			fs.ListFiles(file_path, [&n_files](const string &path, bool) { n_files++; });
			if (n_files > 0) {
				throw IOException("Directory %s is not empty! Enable OVERWRITE_OR_IGNORE option to force writing",
				                  file_path);
			}
		}

		auto state = make_uniq<CopyToFunctionGlobalState>(nullptr);

		if (partition_output) {
			state->partition_state = make_shared<GlobalHivePartitionState>();
		}

		return std::move(state);
	}

	return make_uniq<CopyToFunctionGlobalState>(function.copy_to_initialize_global(context, *bind_data, file_path));
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//

SourceResultType PhysicalCopyToFile::GetData(ExecutionContext &context, DataChunk &chunk,
                                             OperatorSourceInput &input) const {
	auto &g = sink_state->Cast<CopyToFunctionGlobalState>();

	chunk.SetCardinality(1);
	chunk.SetValue(0, 0, Value::BIGINT(g.rows_copied));

	return SourceResultType::FINISHED;
}

} // namespace duckdb









namespace duckdb {

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
class DeleteGlobalState : public GlobalSinkState {
public:
	explicit DeleteGlobalState(ClientContext &context, const vector<LogicalType> &return_types)
	    : deleted_count(0), return_collection(context, return_types) {
	}

	mutex delete_lock;
	idx_t deleted_count;
	ColumnDataCollection return_collection;
};

class DeleteLocalState : public LocalSinkState {
public:
	DeleteLocalState(Allocator &allocator, const vector<LogicalType> &table_types) {
		delete_chunk.Initialize(allocator, table_types);
	}
	DataChunk delete_chunk;
};

SinkResultType PhysicalDelete::Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const {
	auto &gstate = input.global_state.Cast<DeleteGlobalState>();
	auto &ustate = input.local_state.Cast<DeleteLocalState>();

	// get rows and
	auto &transaction = DuckTransaction::Get(context.client, table.db);
	auto &row_identifiers = chunk.data[row_id_index];

	vector<column_t> column_ids;
	for (idx_t i = 0; i < table.column_definitions.size(); i++) {
		column_ids.emplace_back(i);
	};
	auto cfs = ColumnFetchState();

	lock_guard<mutex> delete_guard(gstate.delete_lock);
	if (return_chunk) {
		row_identifiers.Flatten(chunk.size());
		table.Fetch(transaction, ustate.delete_chunk, column_ids, row_identifiers, chunk.size(), cfs);
		gstate.return_collection.Append(ustate.delete_chunk);
	}
	gstate.deleted_count += table.Delete(tableref, context.client, row_identifiers, chunk.size());

	return SinkResultType::NEED_MORE_INPUT;
}

unique_ptr<GlobalSinkState> PhysicalDelete::GetGlobalSinkState(ClientContext &context) const {
	return make_uniq<DeleteGlobalState>(context, GetTypes());
}

unique_ptr<LocalSinkState> PhysicalDelete::GetLocalSinkState(ExecutionContext &context) const {
	return make_uniq<DeleteLocalState>(Allocator::Get(context.client), table.GetTypes());
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
class DeleteSourceState : public GlobalSourceState {
public:
	explicit DeleteSourceState(const PhysicalDelete &op) {
		if (op.return_chunk) {
			D_ASSERT(op.sink_state);
			auto &g = op.sink_state->Cast<DeleteGlobalState>();
			g.return_collection.InitializeScan(scan_state);
		}
	}

	ColumnDataScanState scan_state;
};

unique_ptr<GlobalSourceState> PhysicalDelete::GetGlobalSourceState(ClientContext &context) const {
	return make_uniq<DeleteSourceState>(*this);
}

SourceResultType PhysicalDelete::GetData(ExecutionContext &context, DataChunk &chunk,
                                         OperatorSourceInput &input) const {
	auto &state = input.global_state.Cast<DeleteSourceState>();
	auto &g = sink_state->Cast<DeleteGlobalState>();
	if (!return_chunk) {
		chunk.SetCardinality(1);
		chunk.SetValue(0, 0, Value::BIGINT(g.deleted_count));
		return SourceResultType::FINISHED;
	}

	g.return_collection.Scan(state.scan_state, chunk);

	return chunk.size() == 0 ? SourceResultType::FINISHED : SourceResultType::HAVE_MORE_OUTPUT;
}

} // namespace duckdb












#include <algorithm>
#include <sstream>

namespace duckdb {

using std::stringstream;

static void WriteCatalogEntries(stringstream &ss, vector<reference<CatalogEntry>> &entries) {
	for (auto &entry : entries) {
		if (entry.get().internal) {
			continue;
		}
		ss << entry.get().ToSQL() << std::endl;
	}
	ss << std::endl;
}

static void WriteStringStreamToFile(FileSystem &fs, stringstream &ss, const string &path) {
	auto ss_string = ss.str();
	auto handle = fs.OpenFile(path, FileFlags::FILE_FLAGS_WRITE | FileFlags::FILE_FLAGS_FILE_CREATE_NEW,
	                          FileLockType::WRITE_LOCK);
	fs.Write(*handle, (void *)ss_string.c_str(), ss_string.size());
	handle.reset();
}

static void WriteValueAsSQL(stringstream &ss, Value &val) {
	if (val.type().IsNumeric()) {
		ss << val.ToString();
	} else {
		ss << "'" << val.ToString() << "'";
	}
}

static void WriteCopyStatement(FileSystem &fs, stringstream &ss, CopyInfo &info, ExportedTableData &exported_table,
                               CopyFunction const &function) {
	ss << "COPY ";

	if (exported_table.schema_name != DEFAULT_SCHEMA) {
		ss << KeywordHelper::WriteOptionallyQuoted(exported_table.schema_name) << ".";
	}

	ss << StringUtil::Format("%s FROM %s (", SQLIdentifier(exported_table.table_name),
	                         SQLString(exported_table.file_path));

	// write the copy options
	ss << "FORMAT '" << info.format << "'";
	if (info.format == "csv") {
		// insert default csv options, if not specified
		if (info.options.find("header") == info.options.end()) {
			info.options["header"].push_back(Value::INTEGER(0));
		}
		if (info.options.find("delimiter") == info.options.end() && info.options.find("sep") == info.options.end() &&
		    info.options.find("delim") == info.options.end()) {
			info.options["delimiter"].push_back(Value(","));
		}
		if (info.options.find("quote") == info.options.end()) {
			info.options["quote"].push_back(Value("\""));
		}
	}
	for (auto &copy_option : info.options) {
		if (copy_option.first == "force_quote") {
			continue;
		}
		ss << ", " << copy_option.first << " ";
		if (copy_option.second.size() == 1) {
			WriteValueAsSQL(ss, copy_option.second[0]);
		} else {
			// FIXME handle multiple options
			throw NotImplementedException("FIXME: serialize list of options");
		}
	}
	ss << ");" << std::endl;
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
class ExportSourceState : public GlobalSourceState {
public:
	ExportSourceState() : finished(false) {
	}

	bool finished;
};

unique_ptr<GlobalSourceState> PhysicalExport::GetGlobalSourceState(ClientContext &context) const {
	return make_uniq<ExportSourceState>();
}

SourceResultType PhysicalExport::GetData(ExecutionContext &context, DataChunk &chunk,
                                         OperatorSourceInput &input) const {
	auto &state = input.global_state.Cast<ExportSourceState>();
	if (state.finished) {
		return SourceResultType::FINISHED;
	}

	auto &ccontext = context.client;
	auto &fs = FileSystem::GetFileSystem(ccontext);

	// gather all catalog types to export
	vector<reference<CatalogEntry>> schemas;
	vector<reference<CatalogEntry>> custom_types;
	vector<reference<CatalogEntry>> sequences;
	vector<reference<CatalogEntry>> tables;
	vector<reference<CatalogEntry>> views;
	vector<reference<CatalogEntry>> indexes;
	vector<reference<CatalogEntry>> macros;

	auto schema_list = Catalog::GetSchemas(ccontext, info->catalog);
	for (auto &schema_p : schema_list) {
		auto &schema = schema_p.get();
		if (!schema.internal) {
			schemas.push_back(schema);
		}
		schema.Scan(context.client, CatalogType::TABLE_ENTRY, [&](CatalogEntry &entry) {
			if (entry.internal) {
				return;
			}
			if (entry.type != CatalogType::TABLE_ENTRY) {
				views.push_back(entry);
			}
		});
		schema.Scan(context.client, CatalogType::SEQUENCE_ENTRY,
		            [&](CatalogEntry &entry) { sequences.push_back(entry); });
		schema.Scan(context.client, CatalogType::TYPE_ENTRY,
		            [&](CatalogEntry &entry) { custom_types.push_back(entry); });
		schema.Scan(context.client, CatalogType::INDEX_ENTRY, [&](CatalogEntry &entry) { indexes.push_back(entry); });
		schema.Scan(context.client, CatalogType::MACRO_ENTRY, [&](CatalogEntry &entry) {
			if (!entry.internal && entry.type == CatalogType::MACRO_ENTRY) {
				macros.push_back(entry);
			}
		});
		schema.Scan(context.client, CatalogType::TABLE_MACRO_ENTRY, [&](CatalogEntry &entry) {
			if (!entry.internal && entry.type == CatalogType::TABLE_MACRO_ENTRY) {
				macros.push_back(entry);
			}
		});
	}

	// consider the order of tables because of foreign key constraint
	for (idx_t i = 0; i < exported_tables.data.size(); i++) {
		tables.push_back(exported_tables.data[i].entry);
	}

	// order macro's by timestamp so nested macro's are imported nicely
	sort(macros.begin(), macros.end(), [](const reference<CatalogEntry> &lhs, const reference<CatalogEntry> &rhs) {
		return lhs.get().oid < rhs.get().oid;
	});

	// write the schema.sql file
	// export order is SCHEMA -> SEQUENCE -> TABLE -> VIEW -> INDEX

	stringstream ss;
	WriteCatalogEntries(ss, schemas);
	WriteCatalogEntries(ss, custom_types);
	WriteCatalogEntries(ss, sequences);
	WriteCatalogEntries(ss, tables);
	WriteCatalogEntries(ss, views);
	WriteCatalogEntries(ss, indexes);
	WriteCatalogEntries(ss, macros);

	WriteStringStreamToFile(fs, ss, fs.JoinPath(info->file_path, "schema.sql"));

	// write the load.sql file
	// for every table, we write COPY INTO statement with the specified options
	stringstream load_ss;
	for (idx_t i = 0; i < exported_tables.data.size(); i++) {
		auto exported_table_info = exported_tables.data[i].table_data;
		WriteCopyStatement(fs, load_ss, *info, exported_table_info, function);
	}
	WriteStringStreamToFile(fs, load_ss, fs.JoinPath(info->file_path, "load.sql"));
	state.finished = true;

	return SourceResultType::FINISHED;
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
SinkResultType PhysicalExport::Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const {
	// nop
	return SinkResultType::NEED_MORE_INPUT;
}

//===--------------------------------------------------------------------===//
// Pipeline Construction
//===--------------------------------------------------------------------===//
void PhysicalExport::BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) {
	// EXPORT has an optional child
	// we only need to schedule child pipelines if there is a child
	auto &state = meta_pipeline.GetState();
	state.SetPipelineSource(current, *this);
	if (children.empty()) {
		return;
	}
	PhysicalOperator::BuildPipelines(current, meta_pipeline);
}

vector<const_reference<PhysicalOperator>> PhysicalExport::GetSources() const {
	return {*this};
}

} // namespace duckdb









#include <algorithm>

namespace duckdb {

PhysicalFixedBatchCopy::PhysicalFixedBatchCopy(vector<LogicalType> types, CopyFunction function_p,
                                               unique_ptr<FunctionData> bind_data_p, idx_t estimated_cardinality)
    : PhysicalOperator(PhysicalOperatorType::BATCH_COPY_TO_FILE, std::move(types), estimated_cardinality),
      function(std::move(function_p)), bind_data(std::move(bind_data_p)) {
	if (!function.flush_batch || !function.prepare_batch || !function.desired_batch_size) {
		throw InternalException("PhysicalFixedBatchCopy created for copy function that does not have "
		                        "prepare_batch/flush_batch/desired_batch_size defined");
	}
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
class BatchCopyTask {
public:
	virtual ~BatchCopyTask() {
	}

	virtual void Execute(const PhysicalFixedBatchCopy &op, ClientContext &context, GlobalSinkState &gstate_p) = 0;
};

//===--------------------------------------------------------------------===//
// States
//===--------------------------------------------------------------------===//
class FixedBatchCopyGlobalState : public GlobalSinkState {
public:
	explicit FixedBatchCopyGlobalState(unique_ptr<GlobalFunctionData> global_state)
	    : rows_copied(0), global_state(std::move(global_state)), batch_size(0), scheduled_batch_index(0),
	      flushed_batch_index(0), any_flushing(false), any_finished(false) {
	}

	mutex lock;
	mutex flush_lock;
	//! The total number of rows copied to the file
	atomic<idx_t> rows_copied;
	//! Global copy state
	unique_ptr<GlobalFunctionData> global_state;
	//! The desired batch size (if any)
	idx_t batch_size;
	//! Unpartitioned batches - only used in case batch_size is required
	map<idx_t, unique_ptr<ColumnDataCollection>> raw_batches;
	//! The prepared batch data by batch index - ready to flush
	map<idx_t, unique_ptr<PreparedBatchData>> batch_data;
	//! The index of the latest batch index that has been scheduled
	atomic<idx_t> scheduled_batch_index;
	//! The index of the latest batch index that has been flushed
	atomic<idx_t> flushed_batch_index;
	//! Whether or not any thread is flushing
	atomic<bool> any_flushing;
	//! Whether or not any threads are finished
	atomic<bool> any_finished;

	void AddTask(unique_ptr<BatchCopyTask> task) {
		lock_guard<mutex> l(task_lock);
		task_queue.push(std::move(task));
	}

	unique_ptr<BatchCopyTask> GetTask() {
		lock_guard<mutex> l(task_lock);
		if (task_queue.empty()) {
			return nullptr;
		}
		auto entry = std::move(task_queue.front());
		task_queue.pop();
		return entry;
	}

	idx_t TaskCount() {
		lock_guard<mutex> l(task_lock);
		return task_queue.size();
	}

	void AddBatchData(idx_t batch_index, unique_ptr<PreparedBatchData> new_batch) {
		// move the batch data to the set of prepared batch data
		lock_guard<mutex> l(lock);
		auto entry = batch_data.insert(make_pair(batch_index, std::move(new_batch)));
		if (!entry.second) {
			throw InternalException("Duplicate batch index %llu encountered in PhysicalFixedBatchCopy", batch_index);
		}
	}

private:
	mutex task_lock;
	//! The task queue for the batch copy to file
	queue<unique_ptr<BatchCopyTask>> task_queue;
};

class FixedBatchCopyLocalState : public LocalSinkState {
public:
	explicit FixedBatchCopyLocalState(unique_ptr<LocalFunctionData> local_state_p)
	    : local_state(std::move(local_state_p)), rows_copied(0) {
	}

	//! Local copy state
	unique_ptr<LocalFunctionData> local_state;
	//! The current collection we are appending to
	unique_ptr<ColumnDataCollection> collection;
	//! The append state of the collection
	ColumnDataAppendState append_state;
	//! How many rows have been copied in total
	idx_t rows_copied;
	//! The current batch index
	optional_idx batch_index;

	void InitializeCollection(ClientContext &context, const PhysicalOperator &op) {
		collection = make_uniq<ColumnDataCollection>(Allocator::Get(context), op.children[0]->types);
		collection->InitializeAppend(append_state);
	}
};

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
SinkResultType PhysicalFixedBatchCopy::Sink(ExecutionContext &context, DataChunk &chunk,
                                            OperatorSinkInput &input) const {
	auto &state = input.local_state.Cast<FixedBatchCopyLocalState>();
	if (!state.collection) {
		state.InitializeCollection(context.client, *this);
		state.batch_index = state.partition_info.batch_index.GetIndex();
	}
	state.rows_copied += chunk.size();
	state.collection->Append(state.append_state, chunk);
	return SinkResultType::NEED_MORE_INPUT;
}

void PhysicalFixedBatchCopy::Combine(ExecutionContext &context, GlobalSinkState &gstate_p,
                                     LocalSinkState &lstate) const {
	auto &state = lstate.Cast<FixedBatchCopyLocalState>();
	auto &gstate = gstate_p.Cast<FixedBatchCopyGlobalState>();
	gstate.rows_copied += state.rows_copied;
	if (!gstate.any_finished) {
		// signal that this thread is finished processing batches and that we should move on to Finalize
		lock_guard<mutex> l(gstate.lock);
		gstate.any_finished = true;
	}
	ExecuteTasks(context.client, gstate);
}

//===--------------------------------------------------------------------===//
// ProcessRemainingBatchesEvent
//===--------------------------------------------------------------------===//
class ProcessRemainingBatchesTask : public ExecutorTask {
public:
	ProcessRemainingBatchesTask(Executor &executor, shared_ptr<Event> event_p, FixedBatchCopyGlobalState &state_p,
	                            ClientContext &context, const PhysicalFixedBatchCopy &op)
	    : ExecutorTask(executor), event(std::move(event_p)), op(op), gstate(state_p), context(context) {
	}

	TaskExecutionResult ExecuteTask(TaskExecutionMode mode) override {
		while (op.ExecuteTask(context, gstate)) {
			op.FlushBatchData(context, gstate, 0);
		}
		event->FinishTask();
		return TaskExecutionResult::TASK_FINISHED;
	}

private:
	shared_ptr<Event> event;
	const PhysicalFixedBatchCopy &op;
	FixedBatchCopyGlobalState &gstate;
	ClientContext &context;
};

class ProcessRemainingBatchesEvent : public BasePipelineEvent {
public:
	ProcessRemainingBatchesEvent(const PhysicalFixedBatchCopy &op_p, FixedBatchCopyGlobalState &gstate_p,
	                             Pipeline &pipeline_p, ClientContext &context)
	    : BasePipelineEvent(pipeline_p), op(op_p), gstate(gstate_p), context(context) {
	}
	const PhysicalFixedBatchCopy &op;
	FixedBatchCopyGlobalState &gstate;
	ClientContext &context;

public:
	void Schedule() override {
		vector<shared_ptr<Task>> tasks;
		for (idx_t i = 0; i < idx_t(TaskScheduler::GetScheduler(context).NumberOfThreads()); i++) {
			auto process_task =
			    make_uniq<ProcessRemainingBatchesTask>(pipeline->executor, shared_from_this(), gstate, context, op);
			tasks.push_back(std::move(process_task));
		}
		D_ASSERT(!tasks.empty());
		SetTasks(std::move(tasks));
	}

	void FinishEvent() override {
		//! Now that all batches are processed we finish flushing the file to disk
		op.FinalFlush(context, gstate);
	}
};
//===--------------------------------------------------------------------===//
// Finalize
//===--------------------------------------------------------------------===//
SinkFinalizeType PhysicalFixedBatchCopy::FinalFlush(ClientContext &context, GlobalSinkState &gstate_p) const {
	auto &gstate = gstate_p.Cast<FixedBatchCopyGlobalState>();
	if (gstate.TaskCount() != 0) {
		throw InternalException("Unexecuted tasks are remaining in PhysicalFixedBatchCopy::FinalFlush!?");
	}
	idx_t min_batch_index = idx_t(NumericLimits<int64_t>::Maximum());
	FlushBatchData(context, gstate_p, min_batch_index);
	if (gstate.scheduled_batch_index != gstate.flushed_batch_index) {
		throw InternalException("Not all batches were flushed to disk - incomplete file?");
	}
	if (function.copy_to_finalize) {
		function.copy_to_finalize(context, *bind_data, *gstate.global_state);

		if (use_tmp_file) {
			PhysicalCopyToFile::MoveTmpFile(context, file_path);
		}
	}
	return SinkFinalizeType::READY;
}

SinkFinalizeType PhysicalFixedBatchCopy::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                                  GlobalSinkState &gstate_p) const {
	auto &gstate = gstate_p.Cast<FixedBatchCopyGlobalState>();
	idx_t min_batch_index = idx_t(NumericLimits<int64_t>::Maximum());
	// repartition any remaining batches
	RepartitionBatches(context, gstate_p, min_batch_index, true);
	// check if we have multiple tasks to execute
	if (gstate.TaskCount() <= 1) {
		// we don't - just execute the remaining task and finish flushing to disk
		ExecuteTasks(context, gstate_p);
		FinalFlush(context, gstate_p);
		return SinkFinalizeType::READY;
	}
	// we have multiple tasks remaining - launch an event to execute the tasks in parallel
	auto new_event = make_shared<ProcessRemainingBatchesEvent>(*this, gstate, pipeline, context);
	event.InsertEvent(std::move(new_event));
	return SinkFinalizeType::READY;
}

//===--------------------------------------------------------------------===//
// Tasks
//===--------------------------------------------------------------------===//
class RepartitionedFlushTask : public BatchCopyTask {
public:
	RepartitionedFlushTask() {
	}

	void Execute(const PhysicalFixedBatchCopy &op, ClientContext &context, GlobalSinkState &gstate_p) override {
		op.FlushBatchData(context, gstate_p, 0);
	}
};

class PrepareBatchTask : public BatchCopyTask {
public:
	PrepareBatchTask(idx_t batch_index, unique_ptr<ColumnDataCollection> collection_p)
	    : batch_index(batch_index), collection(std::move(collection_p)) {
	}

	idx_t batch_index;
	unique_ptr<ColumnDataCollection> collection;

	void Execute(const PhysicalFixedBatchCopy &op, ClientContext &context, GlobalSinkState &gstate_p) override {
		auto &gstate = gstate_p.Cast<FixedBatchCopyGlobalState>();
		auto batch_data =
		    op.function.prepare_batch(context, *op.bind_data, *gstate.global_state, std::move(collection));
		gstate.AddBatchData(batch_index, std::move(batch_data));
		if (batch_index == gstate.flushed_batch_index) {
			gstate.AddTask(make_uniq<RepartitionedFlushTask>());
		}
	}
};

//===--------------------------------------------------------------------===//
// Batch Data Handling
//===--------------------------------------------------------------------===//
void PhysicalFixedBatchCopy::AddRawBatchData(ClientContext &context, GlobalSinkState &gstate_p, idx_t batch_index,
                                             unique_ptr<ColumnDataCollection> collection) const {
	auto &gstate = gstate_p.Cast<FixedBatchCopyGlobalState>();

	// add the batch index to the set of raw batches
	lock_guard<mutex> l(gstate.lock);
	auto entry = gstate.raw_batches.insert(make_pair(batch_index, std::move(collection)));
	if (!entry.second) {
		throw InternalException("Duplicate batch index %llu encountered in PhysicalFixedBatchCopy", batch_index);
	}
}

static bool CorrectSizeForBatch(idx_t collection_size, idx_t desired_size) {
	return idx_t(AbsValue<int64_t>(int64_t(collection_size) - int64_t(desired_size))) < STANDARD_VECTOR_SIZE;
}

void PhysicalFixedBatchCopy::RepartitionBatches(ClientContext &context, GlobalSinkState &gstate_p, idx_t min_index,
                                                bool final) const {
	auto &gstate = gstate_p.Cast<FixedBatchCopyGlobalState>();

	// repartition batches until the min index is reached
	lock_guard<mutex> l(gstate.lock);
	if (gstate.raw_batches.empty()) {
		return;
	}
	if (!final) {
		if (gstate.any_finished) {
			// we only repartition in ::NextBatch if all threads are still busy processing batches
			// otherwise we might end up repartitioning a lot of data with only a few threads remaining
			// which causes erratic performance
			return;
		}
		// if this is not the final flush we first check if we have enough data to merge past the batch threshold
		idx_t candidate_rows = 0;
		for (auto entry = gstate.raw_batches.begin(); entry != gstate.raw_batches.end(); entry++) {
			if (entry->first >= min_index) {
				// we have exceeded the minimum batch
				break;
			}
			candidate_rows += entry->second->Count();
		}
		if (candidate_rows < gstate.batch_size) {
			// not enough rows - cancel!
			return;
		}
	}
	// gather all collections we can repartition
	idx_t max_batch_index = 0;
	vector<unique_ptr<ColumnDataCollection>> collections;
	for (auto entry = gstate.raw_batches.begin(); entry != gstate.raw_batches.end();) {
		if (entry->first >= min_index) {
			break;
		}
		max_batch_index = entry->first;
		collections.push_back(std::move(entry->second));
		entry = gstate.raw_batches.erase(entry);
	}
	unique_ptr<ColumnDataCollection> current_collection;
	ColumnDataAppendState append_state;
	// now perform the actual repartitioning
	for (auto &collection : collections) {
		if (!current_collection) {
			if (CorrectSizeForBatch(collection->Count(), gstate.batch_size)) {
				// the collection is ~approximately equal to the batch size (off by at most one vector)
				// use it directly
				gstate.AddTask(make_uniq<PrepareBatchTask>(gstate.scheduled_batch_index++, std::move(collection)));
				collection.reset();
			} else if (collection->Count() < gstate.batch_size) {
				// the collection is smaller than the batch size - use it as a starting point
				current_collection = std::move(collection);
				collection.reset();
			} else {
				// the collection is too large for a batch - we need to repartition
				// create an empty collection
				current_collection = make_uniq<ColumnDataCollection>(Allocator::Get(context), children[0]->types);
			}
			if (current_collection) {
				current_collection->InitializeAppend(append_state);
			}
		}
		if (!collection) {
			// we have consumed the collection already - no need to append
			continue;
		}
		// iterate the collection while appending
		for (auto &chunk : collection->Chunks()) {
			// append the chunk to the collection
			current_collection->Append(append_state, chunk);
			if (current_collection->Count() < gstate.batch_size) {
				// the collection is still under the batch size - continue
				continue;
			}
			// the collection is full - move it to the result and create a new one
			gstate.AddTask(make_uniq<PrepareBatchTask>(gstate.scheduled_batch_index++, std::move(current_collection)));
			current_collection = make_uniq<ColumnDataCollection>(Allocator::Get(context), children[0]->types);
			current_collection->InitializeAppend(append_state);
		}
	}
	if (current_collection && current_collection->Count() > 0) {
		// if there are any remaining batches that are not filled up to the batch size
		// AND this is not the final collection
		// re-add it to the set of raw (to-be-merged) batches
		if (final || CorrectSizeForBatch(current_collection->Count(), gstate.batch_size)) {
			gstate.AddTask(make_uniq<PrepareBatchTask>(gstate.scheduled_batch_index++, std::move(current_collection)));
		} else {
			gstate.raw_batches[max_batch_index] = std::move(current_collection);
		}
	}
}

void PhysicalFixedBatchCopy::FlushBatchData(ClientContext &context, GlobalSinkState &gstate_p, idx_t min_index) const {
	auto &gstate = gstate_p.Cast<FixedBatchCopyGlobalState>();

	// flush batch data to disk (if there are any to flush)
	// grab the flush lock - we can only call flush_batch with this lock
	// otherwise the data might end up in the wrong order
	{
		lock_guard<mutex> l(gstate.flush_lock);
		if (gstate.any_flushing) {
			return;
		}
		gstate.any_flushing = true;
	}
	ActiveFlushGuard active_flush(gstate.any_flushing);
	while (true) {
		unique_ptr<PreparedBatchData> batch_data;
		{
			lock_guard<mutex> l(gstate.lock);
			if (gstate.batch_data.empty()) {
				// no batch data left to flush
				break;
			}
			auto entry = gstate.batch_data.begin();
			if (entry->first != gstate.flushed_batch_index) {
				// this entry is not yet ready to be flushed
				break;
			}
			if (entry->first < gstate.flushed_batch_index) {
				throw InternalException("Batch index was out of order!?");
			}
			batch_data = std::move(entry->second);
			gstate.batch_data.erase(entry);
		}
		function.flush_batch(context, *bind_data, *gstate.global_state, *batch_data);
		gstate.flushed_batch_index++;
	}
}

//===--------------------------------------------------------------------===//
// Tasks
//===--------------------------------------------------------------------===//
bool PhysicalFixedBatchCopy::ExecuteTask(ClientContext &context, GlobalSinkState &gstate_p) const {
	auto &gstate = gstate_p.Cast<FixedBatchCopyGlobalState>();
	auto task = gstate.GetTask();
	if (!task) {
		return false;
	}
	task->Execute(*this, context, gstate_p);
	return true;
}

void PhysicalFixedBatchCopy::ExecuteTasks(ClientContext &context, GlobalSinkState &gstate_p) const {
	while (ExecuteTask(context, gstate_p)) {
	}
}

//===--------------------------------------------------------------------===//
// Next Batch
//===--------------------------------------------------------------------===//
void PhysicalFixedBatchCopy::NextBatch(ExecutionContext &context, GlobalSinkState &gstate_p,
                                       LocalSinkState &lstate) const {
	auto &state = lstate.Cast<FixedBatchCopyLocalState>();
	if (state.collection && state.collection->Count() > 0) {
		// we finished processing this batch
		// start flushing data
		auto min_batch_index = lstate.partition_info.min_batch_index.GetIndex();
		// push the raw batch data into the set of unprocessed batches
		AddRawBatchData(context.client, gstate_p, state.batch_index.GetIndex(), std::move(state.collection));
		// attempt to repartition to our desired batch size
		RepartitionBatches(context.client, gstate_p, min_batch_index);
		// execute a single batch task
		ExecuteTask(context.client, gstate_p);
		FlushBatchData(context.client, gstate_p, min_batch_index);
	}
	state.batch_index = lstate.partition_info.batch_index.GetIndex();

	state.InitializeCollection(context.client, *this);
}

unique_ptr<LocalSinkState> PhysicalFixedBatchCopy::GetLocalSinkState(ExecutionContext &context) const {
	return make_uniq<FixedBatchCopyLocalState>(function.copy_to_initialize_local(context, *bind_data));
}

unique_ptr<GlobalSinkState> PhysicalFixedBatchCopy::GetGlobalSinkState(ClientContext &context) const {
	auto result =
	    make_uniq<FixedBatchCopyGlobalState>(function.copy_to_initialize_global(context, *bind_data, file_path));
	result->batch_size = function.desired_batch_size(context, *bind_data);
	return std::move(result);
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
SourceResultType PhysicalFixedBatchCopy::GetData(ExecutionContext &context, DataChunk &chunk,
                                                 OperatorSourceInput &input) const {
	auto &g = sink_state->Cast<FixedBatchCopyGlobalState>();

	chunk.SetCardinality(1);
	chunk.SetValue(0, 0, Value::BIGINT(g.rows_copied));
	return SourceResultType::FINISHED;
}

} // namespace duckdb




















namespace duckdb {

PhysicalInsert::PhysicalInsert(vector<LogicalType> types_p, TableCatalogEntry &table,
                               physical_index_vector_t<idx_t> column_index_map,
                               vector<unique_ptr<Expression>> bound_defaults,
                               vector<unique_ptr<Expression>> set_expressions, vector<PhysicalIndex> set_columns,
                               vector<LogicalType> set_types, idx_t estimated_cardinality, bool return_chunk,
                               bool parallel, OnConflictAction action_type,
                               unique_ptr<Expression> on_conflict_condition_p,
                               unique_ptr<Expression> do_update_condition_p, unordered_set<column_t> conflict_target_p,
                               vector<column_t> columns_to_fetch_p)
    : PhysicalOperator(PhysicalOperatorType::INSERT, std::move(types_p), estimated_cardinality),
      column_index_map(std::move(column_index_map)), insert_table(&table), insert_types(table.GetTypes()),
      bound_defaults(std::move(bound_defaults)), return_chunk(return_chunk), parallel(parallel),
      action_type(action_type), set_expressions(std::move(set_expressions)), set_columns(std::move(set_columns)),
      set_types(std::move(set_types)), on_conflict_condition(std::move(on_conflict_condition_p)),
      do_update_condition(std::move(do_update_condition_p)), conflict_target(std::move(conflict_target_p)),
      columns_to_fetch(std::move(columns_to_fetch_p)) {

	if (action_type == OnConflictAction::THROW) {
		return;
	}

	D_ASSERT(set_expressions.size() == set_columns.size());

	// One or more columns are referenced from the existing table,
	// we use the 'insert_types' to figure out which types these columns have
	types_to_fetch = vector<LogicalType>(columns_to_fetch.size(), LogicalType::SQLNULL);
	for (idx_t i = 0; i < columns_to_fetch.size(); i++) {
		auto &id = columns_to_fetch[i];
		D_ASSERT(id < insert_types.size());
		types_to_fetch[i] = insert_types[id];
	}
}

PhysicalInsert::PhysicalInsert(LogicalOperator &op, SchemaCatalogEntry &schema, unique_ptr<BoundCreateTableInfo> info_p,
                               idx_t estimated_cardinality, bool parallel)
    : PhysicalOperator(PhysicalOperatorType::CREATE_TABLE_AS, op.types, estimated_cardinality), insert_table(nullptr),
      return_chunk(false), schema(&schema), info(std::move(info_p)), parallel(parallel),
      action_type(OnConflictAction::THROW) {
	GetInsertInfo(*info, insert_types, bound_defaults);
}

void PhysicalInsert::GetInsertInfo(const BoundCreateTableInfo &info, vector<LogicalType> &insert_types,
                                   vector<unique_ptr<Expression>> &bound_defaults) {
	auto &create_info = (CreateTableInfo &)*info.base;
	for (auto &col : create_info.columns.Physical()) {
		insert_types.push_back(col.GetType());
		bound_defaults.push_back(make_uniq<BoundConstantExpression>(Value(col.GetType())));
	}
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
class InsertGlobalState : public GlobalSinkState {
public:
	explicit InsertGlobalState(ClientContext &context, const vector<LogicalType> &return_types, DuckTableEntry &table)
	    : table(table), insert_count(0), initialized(false), return_collection(context, return_types) {
	}

	mutex lock;
	DuckTableEntry &table;
	idx_t insert_count;
	bool initialized;
	LocalAppendState append_state;
	ColumnDataCollection return_collection;
};

class InsertLocalState : public LocalSinkState {
public:
	InsertLocalState(ClientContext &context, const vector<LogicalType> &types,
	                 const vector<unique_ptr<Expression>> &bound_defaults)
	    : default_executor(context, bound_defaults) {
		insert_chunk.Initialize(Allocator::Get(context), types);
	}

	DataChunk insert_chunk;
	ExpressionExecutor default_executor;
	TableAppendState local_append_state;
	unique_ptr<RowGroupCollection> local_collection;
	optional_ptr<OptimisticDataWriter> writer;
	// Rows that have been updated by a DO UPDATE conflict
	unordered_set<row_t> updated_global_rows;
	// Rows in the transaction-local storage that have been updated by a DO UPDATE conflict
	unordered_set<row_t> updated_local_rows;
	idx_t update_count = 0;
};

unique_ptr<GlobalSinkState> PhysicalInsert::GetGlobalSinkState(ClientContext &context) const {
	optional_ptr<TableCatalogEntry> table;
	if (info) {
		// CREATE TABLE AS
		D_ASSERT(!insert_table);
		auto &catalog = schema->catalog;
		table = &catalog.CreateTable(catalog.GetCatalogTransaction(context), *schema.get_mutable(), *info)
		             ->Cast<TableCatalogEntry>();
	} else {
		D_ASSERT(insert_table);
		D_ASSERT(insert_table->IsDuckTable());
		table = insert_table.get_mutable();
	}
	auto result = make_uniq<InsertGlobalState>(context, GetTypes(), table->Cast<DuckTableEntry>());
	return std::move(result);
}

unique_ptr<LocalSinkState> PhysicalInsert::GetLocalSinkState(ExecutionContext &context) const {
	return make_uniq<InsertLocalState>(context.client, insert_types, bound_defaults);
}

void PhysicalInsert::ResolveDefaults(const TableCatalogEntry &table, DataChunk &chunk,
                                     const physical_index_vector_t<idx_t> &column_index_map,
                                     ExpressionExecutor &default_executor, DataChunk &result) {
	chunk.Flatten();
	default_executor.SetChunk(chunk);

	result.Reset();
	result.SetCardinality(chunk);

	if (!column_index_map.empty()) {
		// columns specified by the user, use column_index_map
		for (auto &col : table.GetColumns().Physical()) {
			auto storage_idx = col.StorageOid();
			auto mapped_index = column_index_map[col.Physical()];
			if (mapped_index == DConstants::INVALID_INDEX) {
				// insert default value
				default_executor.ExecuteExpression(storage_idx, result.data[storage_idx]);
			} else {
				// get value from child chunk
				D_ASSERT((idx_t)mapped_index < chunk.ColumnCount());
				D_ASSERT(result.data[storage_idx].GetType() == chunk.data[mapped_index].GetType());
				result.data[storage_idx].Reference(chunk.data[mapped_index]);
			}
		}
	} else {
		// no columns specified, just append directly
		for (idx_t i = 0; i < result.ColumnCount(); i++) {
			D_ASSERT(result.data[i].GetType() == chunk.data[i].GetType());
			result.data[i].Reference(chunk.data[i]);
		}
	}
}

bool AllConflictsMeetCondition(DataChunk &result) {
	auto data = FlatVector::GetData<bool>(result.data[0]);
	for (idx_t i = 0; i < result.size(); i++) {
		if (!data[i]) {
			return false;
		}
	}
	return true;
}

void CheckOnConflictCondition(ExecutionContext &context, DataChunk &conflicts, const unique_ptr<Expression> &condition,
                              DataChunk &result) {
	ExpressionExecutor executor(context.client, *condition);
	result.Initialize(context.client, {LogicalType::BOOLEAN});
	executor.Execute(conflicts, result);
	result.SetCardinality(conflicts.size());
}

static void CombineExistingAndInsertTuples(DataChunk &result, DataChunk &scan_chunk, DataChunk &input_chunk,
                                           ClientContext &client, const PhysicalInsert &op) {
	auto &types_to_fetch = op.types_to_fetch;
	auto &insert_types = op.insert_types;

	if (types_to_fetch.empty()) {
		// We have not scanned the initial table, so we can just duplicate the initial chunk
		result.Initialize(client, input_chunk.GetTypes());
		result.Reference(input_chunk);
		result.SetCardinality(input_chunk);
		return;
	}
	vector<LogicalType> combined_types;
	combined_types.reserve(insert_types.size() + types_to_fetch.size());
	combined_types.insert(combined_types.end(), insert_types.begin(), insert_types.end());
	combined_types.insert(combined_types.end(), types_to_fetch.begin(), types_to_fetch.end());

	result.Initialize(client, combined_types);
	result.Reset();
	// Add the VALUES list
	for (idx_t i = 0; i < insert_types.size(); i++) {
		idx_t col_idx = i;
		auto &other_col = input_chunk.data[i];
		auto &this_col = result.data[col_idx];
		D_ASSERT(other_col.GetType() == this_col.GetType());
		this_col.Reference(other_col);
	}
	// Add the columns from the original conflicting tuples
	for (idx_t i = 0; i < types_to_fetch.size(); i++) {
		idx_t col_idx = i + insert_types.size();
		auto &other_col = scan_chunk.data[i];
		auto &this_col = result.data[col_idx];
		D_ASSERT(other_col.GetType() == this_col.GetType());
		this_col.Reference(other_col);
	}
	// This is guaranteed by the requirement of a conflict target to have a condition or set expressions
	// Only when we have any sort of condition or SET expression that references the existing table is this possible
	// to not be true.
	// We can have a SET expression without a conflict target ONLY if there is only 1 Index on the table
	// In which case this also can't cause a discrepancy between existing tuple count and insert tuple count
	D_ASSERT(input_chunk.size() == scan_chunk.size());
	result.SetCardinality(input_chunk.size());
}

static void CreateUpdateChunk(ExecutionContext &context, DataChunk &chunk, TableCatalogEntry &table, Vector &row_ids,
                              DataChunk &update_chunk, const PhysicalInsert &op) {

	auto &do_update_condition = op.do_update_condition;
	auto &set_types = op.set_types;
	auto &set_expressions = op.set_expressions;
	// Check the optional condition for the DO UPDATE clause, to filter which rows will be updated
	if (do_update_condition) {
		DataChunk do_update_filter_result;
		do_update_filter_result.Initialize(context.client, {LogicalType::BOOLEAN});
		ExpressionExecutor where_executor(context.client, *do_update_condition);
		where_executor.Execute(chunk, do_update_filter_result);
		do_update_filter_result.SetCardinality(chunk.size());

		ManagedSelection selection(chunk.size());

		auto where_data = FlatVector::GetData<bool>(do_update_filter_result.data[0]);
		for (idx_t i = 0; i < chunk.size(); i++) {
			if (where_data[i]) {
				selection.Append(i);
			}
		}
		if (selection.Count() != selection.Size()) {
			// Not all conflicts met the condition, need to filter out the ones that don't
			chunk.Slice(selection.Selection(), selection.Count());
			chunk.SetCardinality(selection.Count());
			// Also apply this Slice to the to-update row_ids
			row_ids.Slice(selection.Selection(), selection.Count());
		}
	}

	// Execute the SET expressions
	update_chunk.Initialize(context.client, set_types);
	ExpressionExecutor executor(context.client, set_expressions);
	executor.Execute(chunk, update_chunk);
	update_chunk.SetCardinality(chunk);
}

template <bool GLOBAL>
static idx_t PerformOnConflictAction(ExecutionContext &context, DataChunk &chunk, TableCatalogEntry &table,
                                     Vector &row_ids, const PhysicalInsert &op) {

	if (op.action_type == OnConflictAction::NOTHING) {
		return 0;
	}
	auto &set_columns = op.set_columns;

	DataChunk update_chunk;
	CreateUpdateChunk(context, chunk, table, row_ids, update_chunk, op);

	auto &data_table = table.GetStorage();
	// Perform the update, using the results of the SET expressions
	if (GLOBAL) {
		data_table.Update(table, context.client, row_ids, set_columns, update_chunk);
	} else {
		auto &local_storage = LocalStorage::Get(context.client, data_table.db);
		// Perform the update, using the results of the SET expressions
		local_storage.Update(data_table, row_ids, set_columns, update_chunk);
	}
	return update_chunk.size();
}

// TODO: should we use a hash table to keep track of this instead?
template <bool GLOBAL>
static void RegisterUpdatedRows(InsertLocalState &lstate, const Vector &row_ids, idx_t count) {
	// Insert all rows, if any of the rows has already been updated before, we throw an error
	auto data = FlatVector::GetData<row_t>(row_ids);

	// The rowids in the transaction-local ART aren't final yet so we have to separately keep track of the two sets of
	// rowids
	unordered_set<row_t> &updated_rows = GLOBAL ? lstate.updated_global_rows : lstate.updated_local_rows;
	for (idx_t i = 0; i < count; i++) {
		auto result = updated_rows.insert(data[i]);
		if (result.second == false) {
			throw InvalidInputException(
			    "ON CONFLICT DO UPDATE can not update the same row twice in the same command, Ensure that no rows "
			    "proposed for insertion within the same command have duplicate constrained values");
		}
	}
}

template <bool GLOBAL>
static idx_t HandleInsertConflicts(TableCatalogEntry &table, ExecutionContext &context, InsertLocalState &lstate,
                                   DataTable &data_table, const PhysicalInsert &op) {
	auto &types_to_fetch = op.types_to_fetch;
	auto &on_conflict_condition = op.on_conflict_condition;
	auto &conflict_target = op.conflict_target;
	auto &columns_to_fetch = op.columns_to_fetch;

	auto &local_storage = LocalStorage::Get(context.client, data_table.db);

	// We either want to do nothing, or perform an update when conflicts arise
	ConflictInfo conflict_info(conflict_target);
	ConflictManager conflict_manager(VerifyExistenceType::APPEND, lstate.insert_chunk.size(), &conflict_info);
	if (GLOBAL) {
		data_table.VerifyAppendConstraints(table, context.client, lstate.insert_chunk, &conflict_manager);
	} else {
		DataTable::VerifyUniqueIndexes(local_storage.GetIndexes(data_table), context.client, lstate.insert_chunk,
		                               &conflict_manager);
	}
	conflict_manager.Finalize();
	if (conflict_manager.ConflictCount() == 0) {
		// No conflicts found, 0 updates performed
		return 0;
	}
	auto &conflicts = conflict_manager.Conflicts();
	auto &row_ids = conflict_manager.RowIds();

	DataChunk conflict_chunk; // contains only the conflicting values
	DataChunk scan_chunk;     // contains the original values, that caused the conflict
	DataChunk combined_chunk; // contains conflict_chunk + scan_chunk (wide)

	// Filter out everything but the conflicting rows
	conflict_chunk.Initialize(context.client, lstate.insert_chunk.GetTypes());
	conflict_chunk.Reference(lstate.insert_chunk);
	conflict_chunk.Slice(conflicts.Selection(), conflicts.Count());
	conflict_chunk.SetCardinality(conflicts.Count());

	// Holds the pins for the fetched rows
	unique_ptr<ColumnFetchState> fetch_state;
	if (!types_to_fetch.empty()) {
		D_ASSERT(scan_chunk.size() == 0);
		// When these values are required for the conditions or the SET expressions,
		// then we scan the existing table for the conflicting tuples, using the rowids
		scan_chunk.Initialize(context.client, types_to_fetch);
		fetch_state = make_uniq<ColumnFetchState>();
		if (GLOBAL) {
			auto &transaction = DuckTransaction::Get(context.client, table.catalog);
			data_table.Fetch(transaction, scan_chunk, columns_to_fetch, row_ids, conflicts.Count(), *fetch_state);
		} else {
			local_storage.FetchChunk(data_table, row_ids, conflicts.Count(), columns_to_fetch, scan_chunk,
			                         *fetch_state);
		}
	}

	// Splice the Input chunk and the fetched chunk together
	CombineExistingAndInsertTuples(combined_chunk, scan_chunk, conflict_chunk, context.client, op);

	if (on_conflict_condition) {
		DataChunk conflict_condition_result;
		CheckOnConflictCondition(context, combined_chunk, on_conflict_condition, conflict_condition_result);
		bool conditions_met = AllConflictsMeetCondition(conflict_condition_result);
		if (!conditions_met) {
			// Filter out the tuples that did pass the filter, then run the verify again
			ManagedSelection sel(combined_chunk.size());
			auto data = FlatVector::GetData<bool>(conflict_condition_result.data[0]);
			for (idx_t i = 0; i < combined_chunk.size(); i++) {
				if (!data[i]) {
					// Only populate the selection vector with the tuples that did not meet the condition
					sel.Append(i);
				}
			}
			combined_chunk.Slice(sel.Selection(), sel.Count());
			row_ids.Slice(sel.Selection(), sel.Count());
			if (GLOBAL) {
				data_table.VerifyAppendConstraints(table, context.client, combined_chunk, nullptr);
			} else {
				DataTable::VerifyUniqueIndexes(local_storage.GetIndexes(data_table), context.client,
				                               lstate.insert_chunk, nullptr);
			}
			throw InternalException("The previous operation was expected to throw but didn't");
		}
	}

	RegisterUpdatedRows<GLOBAL>(lstate, row_ids, combined_chunk.size());

	idx_t updated_tuples = PerformOnConflictAction<GLOBAL>(context, combined_chunk, table, row_ids, op);

	// Remove the conflicting tuples from the insert chunk
	SelectionVector sel_vec(lstate.insert_chunk.size());
	idx_t new_size =
	    SelectionVector::Inverted(conflicts.Selection(), sel_vec, conflicts.Count(), lstate.insert_chunk.size());
	lstate.insert_chunk.Slice(sel_vec, new_size);
	lstate.insert_chunk.SetCardinality(new_size);
	return updated_tuples;
}

idx_t PhysicalInsert::OnConflictHandling(TableCatalogEntry &table, ExecutionContext &context,
                                         InsertLocalState &lstate) const {
	auto &data_table = table.GetStorage();
	if (action_type == OnConflictAction::THROW) {
		data_table.VerifyAppendConstraints(table, context.client, lstate.insert_chunk, nullptr);
		return 0;
	}
	// Check whether any conflicts arise, and if they all meet the conflict_target + condition
	// If that's not the case - We throw the first error
	idx_t updated_tuples = 0;
	updated_tuples += HandleInsertConflicts<true>(table, context, lstate, data_table, *this);
	// Also check the transaction-local storage+ART so we can detect conflicts within this transaction
	updated_tuples += HandleInsertConflicts<false>(table, context, lstate, data_table, *this);

	return updated_tuples;
}

SinkResultType PhysicalInsert::Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const {
	auto &gstate = input.global_state.Cast<InsertGlobalState>();
	auto &lstate = input.local_state.Cast<InsertLocalState>();

	auto &table = gstate.table;
	auto &storage = table.GetStorage();
	PhysicalInsert::ResolveDefaults(table, chunk, column_index_map, lstate.default_executor, lstate.insert_chunk);

	if (!parallel) {
		if (!gstate.initialized) {
			storage.InitializeLocalAppend(gstate.append_state, context.client);
			gstate.initialized = true;
		}

		idx_t updated_tuples = OnConflictHandling(table, context, lstate);
		gstate.insert_count += lstate.insert_chunk.size();
		gstate.insert_count += updated_tuples;
		storage.LocalAppend(gstate.append_state, table, context.client, lstate.insert_chunk, true);

		if (return_chunk) {
			gstate.return_collection.Append(lstate.insert_chunk);
		}
	} else {
		D_ASSERT(!return_chunk);
		// parallel append
		if (!lstate.local_collection) {
			lock_guard<mutex> l(gstate.lock);
			auto &table_info = storage.info;
			auto &block_manager = TableIOManager::Get(storage).GetBlockManagerForRowData();
			lstate.local_collection =
			    make_uniq<RowGroupCollection>(table_info, block_manager, insert_types, MAX_ROW_ID);
			lstate.local_collection->InitializeEmpty();
			lstate.local_collection->InitializeAppend(lstate.local_append_state);
			lstate.writer = &gstate.table.GetStorage().CreateOptimisticWriter(context.client);
		}
		OnConflictHandling(table, context, lstate);

		auto new_row_group = lstate.local_collection->Append(lstate.insert_chunk, lstate.local_append_state);
		if (new_row_group) {
			lstate.writer->WriteNewRowGroup(*lstate.local_collection);
		}
	}

	return SinkResultType::NEED_MORE_INPUT;
}

void PhysicalInsert::Combine(ExecutionContext &context, GlobalSinkState &gstate_p, LocalSinkState &lstate_p) const {
	auto &gstate = gstate_p.Cast<InsertGlobalState>();
	auto &lstate = lstate_p.Cast<InsertLocalState>();
	auto &client_profiler = QueryProfiler::Get(context.client);
	context.thread.profiler.Flush(*this, lstate.default_executor, "default_executor", 1);
	client_profiler.Flush(context.thread.profiler);

	if (!parallel) {
		return;
	}
	if (!lstate.local_collection) {
		return;
	}
	// parallel append: finalize the append
	TransactionData tdata(0, 0);
	lstate.local_collection->FinalizeAppend(tdata, lstate.local_append_state);

	auto append_count = lstate.local_collection->GetTotalRows();

	lock_guard<mutex> lock(gstate.lock);
	gstate.insert_count += append_count;
	if (append_count < RowGroup::ROW_GROUP_SIZE) {
		// we have few rows - append to the local storage directly
		auto &table = gstate.table;
		auto &storage = table.GetStorage();
		storage.InitializeLocalAppend(gstate.append_state, context.client);
		auto &transaction = DuckTransaction::Get(context.client, table.catalog);
		lstate.local_collection->Scan(transaction, [&](DataChunk &insert_chunk) {
			storage.LocalAppend(gstate.append_state, table, context.client, insert_chunk);
			return true;
		});
		storage.FinalizeLocalAppend(gstate.append_state);
	} else {
		// we have written rows to disk optimistically - merge directly into the transaction-local storage
		gstate.table.GetStorage().FinalizeOptimisticWriter(context.client, *lstate.writer);
		gstate.table.GetStorage().LocalMerge(context.client, *lstate.local_collection);
	}
}

SinkFinalizeType PhysicalInsert::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                          GlobalSinkState &state) const {
	auto &gstate = state.Cast<InsertGlobalState>();
	if (!parallel && gstate.initialized) {
		auto &table = gstate.table;
		auto &storage = table.GetStorage();
		storage.FinalizeLocalAppend(gstate.append_state);
	}
	return SinkFinalizeType::READY;
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
class InsertSourceState : public GlobalSourceState {
public:
	explicit InsertSourceState(const PhysicalInsert &op) {
		if (op.return_chunk) {
			D_ASSERT(op.sink_state);
			auto &g = op.sink_state->Cast<InsertGlobalState>();
			g.return_collection.InitializeScan(scan_state);
		}
	}

	ColumnDataScanState scan_state;
};

unique_ptr<GlobalSourceState> PhysicalInsert::GetGlobalSourceState(ClientContext &context) const {
	return make_uniq<InsertSourceState>(*this);
}

SourceResultType PhysicalInsert::GetData(ExecutionContext &context, DataChunk &chunk,
                                         OperatorSourceInput &input) const {
	auto &state = input.global_state.Cast<InsertSourceState>();
	auto &insert_gstate = sink_state->Cast<InsertGlobalState>();
	if (!return_chunk) {
		chunk.SetCardinality(1);
		chunk.SetValue(0, 0, Value::BIGINT(insert_gstate.insert_count));
		return SourceResultType::FINISHED;
	}

	insert_gstate.return_collection.Scan(state.scan_state, chunk);
	return chunk.size() == 0 ? SourceResultType::FINISHED : SourceResultType::HAVE_MORE_OUTPUT;
}

} // namespace duckdb











namespace duckdb {

PhysicalUpdate::PhysicalUpdate(vector<LogicalType> types, TableCatalogEntry &tableref, DataTable &table,
                               vector<PhysicalIndex> columns, vector<unique_ptr<Expression>> expressions,
                               vector<unique_ptr<Expression>> bound_defaults, idx_t estimated_cardinality,
                               bool return_chunk)
    : PhysicalOperator(PhysicalOperatorType::UPDATE, std::move(types), estimated_cardinality), tableref(tableref),
      table(table), columns(std::move(columns)), expressions(std::move(expressions)),
      bound_defaults(std::move(bound_defaults)), return_chunk(return_chunk) {
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
class UpdateGlobalState : public GlobalSinkState {
public:
	explicit UpdateGlobalState(ClientContext &context, const vector<LogicalType> &return_types)
	    : updated_count(0), return_collection(context, return_types) {
	}

	mutex lock;
	idx_t updated_count;
	unordered_set<row_t> updated_columns;
	ColumnDataCollection return_collection;
};

class UpdateLocalState : public LocalSinkState {
public:
	UpdateLocalState(ClientContext &context, const vector<unique_ptr<Expression>> &expressions,
	                 const vector<LogicalType> &table_types, const vector<unique_ptr<Expression>> &bound_defaults)
	    : default_executor(context, bound_defaults) {
		// initialize the update chunk
		auto &allocator = Allocator::Get(context);
		vector<LogicalType> update_types;
		update_types.reserve(expressions.size());
		for (auto &expr : expressions) {
			update_types.push_back(expr->return_type);
		}
		update_chunk.Initialize(allocator, update_types);
		// initialize the mock chunk
		mock_chunk.Initialize(allocator, table_types);
	}

	DataChunk update_chunk;
	DataChunk mock_chunk;
	ExpressionExecutor default_executor;
};

SinkResultType PhysicalUpdate::Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const {
	auto &gstate = input.global_state.Cast<UpdateGlobalState>();
	auto &lstate = input.local_state.Cast<UpdateLocalState>();

	DataChunk &update_chunk = lstate.update_chunk;
	DataChunk &mock_chunk = lstate.mock_chunk;

	chunk.Flatten();
	lstate.default_executor.SetChunk(chunk);

	// update data in the base table
	// the row ids are given to us as the last column of the child chunk
	auto &row_ids = chunk.data[chunk.ColumnCount() - 1];
	update_chunk.Reset();
	update_chunk.SetCardinality(chunk);

	for (idx_t i = 0; i < expressions.size(); i++) {
		if (expressions[i]->type == ExpressionType::VALUE_DEFAULT) {
			// default expression, set to the default value of the column
			lstate.default_executor.ExecuteExpression(columns[i].index, update_chunk.data[i]);
		} else {
			D_ASSERT(expressions[i]->type == ExpressionType::BOUND_REF);
			// index into child chunk
			auto &binding = expressions[i]->Cast<BoundReferenceExpression>();
			update_chunk.data[i].Reference(chunk.data[binding.index]);
		}
	}

	lock_guard<mutex> glock(gstate.lock);
	if (update_is_del_and_insert) {
		// index update or update on complex type, perform a delete and an append instead

		// figure out which rows have not yet been deleted in this update
		// this is required since we might see the same row_id multiple times
		// in the case of an UPDATE query that e.g. has joins
		auto row_id_data = FlatVector::GetData<row_t>(row_ids);
		SelectionVector sel(STANDARD_VECTOR_SIZE);
		idx_t update_count = 0;
		for (idx_t i = 0; i < update_chunk.size(); i++) {
			auto row_id = row_id_data[i];
			if (gstate.updated_columns.find(row_id) == gstate.updated_columns.end()) {
				gstate.updated_columns.insert(row_id);
				sel.set_index(update_count++, i);
			}
		}
		if (update_count != update_chunk.size()) {
			// we need to slice here
			update_chunk.Slice(sel, update_count);
		}
		table.Delete(tableref, context.client, row_ids, update_chunk.size());
		// for the append we need to arrange the columns in a specific manner (namely the "standard table order")
		mock_chunk.SetCardinality(update_chunk);
		for (idx_t i = 0; i < columns.size(); i++) {
			mock_chunk.data[columns[i].index].Reference(update_chunk.data[i]);
		}
		table.LocalAppend(tableref, context.client, mock_chunk);
	} else {
		if (return_chunk) {
			mock_chunk.SetCardinality(update_chunk);
			for (idx_t i = 0; i < columns.size(); i++) {
				mock_chunk.data[columns[i].index].Reference(update_chunk.data[i]);
			}
		}
		table.Update(tableref, context.client, row_ids, columns, update_chunk);
	}

	if (return_chunk) {
		gstate.return_collection.Append(mock_chunk);
	}

	gstate.updated_count += chunk.size();

	return SinkResultType::NEED_MORE_INPUT;
}

unique_ptr<GlobalSinkState> PhysicalUpdate::GetGlobalSinkState(ClientContext &context) const {
	return make_uniq<UpdateGlobalState>(context, GetTypes());
}

unique_ptr<LocalSinkState> PhysicalUpdate::GetLocalSinkState(ExecutionContext &context) const {
	return make_uniq<UpdateLocalState>(context.client, expressions, table.GetTypes(), bound_defaults);
}

void PhysicalUpdate::Combine(ExecutionContext &context, GlobalSinkState &gstate, LocalSinkState &lstate) const {
	auto &state = lstate.Cast<UpdateLocalState>();
	auto &client_profiler = QueryProfiler::Get(context.client);
	context.thread.profiler.Flush(*this, state.default_executor, "default_executor", 1);
	client_profiler.Flush(context.thread.profiler);
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
class UpdateSourceState : public GlobalSourceState {
public:
	explicit UpdateSourceState(const PhysicalUpdate &op) {
		if (op.return_chunk) {
			D_ASSERT(op.sink_state);
			auto &g = op.sink_state->Cast<UpdateGlobalState>();
			g.return_collection.InitializeScan(scan_state);
		}
	}

	ColumnDataScanState scan_state;
};

unique_ptr<GlobalSourceState> PhysicalUpdate::GetGlobalSourceState(ClientContext &context) const {
	return make_uniq<UpdateSourceState>(*this);
}

SourceResultType PhysicalUpdate::GetData(ExecutionContext &context, DataChunk &chunk,
                                         OperatorSourceInput &input) const {
	auto &state = input.global_state.Cast<UpdateSourceState>();
	auto &g = sink_state->Cast<UpdateGlobalState>();
	if (!return_chunk) {
		chunk.SetCardinality(1);
		chunk.SetValue(0, 0, Value::BIGINT(g.updated_count));
		return SourceResultType::FINISHED;
	}

	g.return_collection.Scan(state.scan_state, chunk);

	return chunk.size() == 0 ? SourceResultType::FINISHED : SourceResultType::HAVE_MORE_OUTPUT;
}

} // namespace duckdb



namespace duckdb {

PhysicalPivot::PhysicalPivot(vector<LogicalType> types_p, unique_ptr<PhysicalOperator> child,
                             BoundPivotInfo bound_pivot_p)
    : PhysicalOperator(PhysicalOperatorType::PIVOT, std::move(types_p), child->estimated_cardinality),
      bound_pivot(std::move(bound_pivot_p)) {
	children.push_back(std::move(child));
	for (idx_t p = 0; p < bound_pivot.pivot_values.size(); p++) {
		auto entry = pivot_map.find(bound_pivot.pivot_values[p]);
		if (entry != pivot_map.end()) {
			continue;
		}
		pivot_map[bound_pivot.pivot_values[p]] = bound_pivot.group_count + p;
	}
	// extract the empty aggregate expressions
	for (auto &aggr_expr : bound_pivot.aggregates) {
		auto &aggr = (BoundAggregateExpression &)*aggr_expr;
		// for each aggregate, initialize an empty aggregate state and finalize it immediately
		auto state = make_unsafe_uniq_array<data_t>(aggr.function.state_size());
		aggr.function.initialize(state.get());
		Vector state_vector(Value::POINTER((uintptr_t)state.get()));
		Vector result_vector(aggr_expr->return_type);
		AggregateInputData aggr_input_data(aggr.bind_info.get(), Allocator::DefaultAllocator());
		aggr.function.finalize(state_vector, aggr_input_data, result_vector, 1, 0);
		empty_aggregates.push_back(result_vector.GetValue(0));
	}
}

OperatorResultType PhysicalPivot::Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                          GlobalOperatorState &gstate, OperatorState &state) const {
	// copy the groups as-is
	for (idx_t i = 0; i < bound_pivot.group_count; i++) {
		chunk.data[i].Reference(input.data[i]);
	}
	auto pivot_column_lists = FlatVector::GetData<list_entry_t>(input.data.back());
	auto &pivot_column_values = ListVector::GetEntry(input.data.back());
	auto pivot_columns = FlatVector::GetData<string_t>(pivot_column_values);

	// initialize all aggregate columns with the empty aggregate value
	// if there are multiple aggregates the columns are in order of [AGGR1][AGGR2][AGGR1][AGGR2]
	// so we need to alternate the empty_aggregate that we use
	idx_t aggregate = 0;
	for (idx_t c = bound_pivot.group_count; c < chunk.ColumnCount(); c++) {
		chunk.data[c].Reference(empty_aggregates[aggregate]);
		chunk.data[c].Flatten(input.size());
		aggregate++;
		if (aggregate >= empty_aggregates.size()) {
			aggregate = 0;
		}
	}

	// move the pivots to the given columns
	for (idx_t r = 0; r < input.size(); r++) {
		auto list = pivot_column_lists[r];
		for (idx_t l = 0; l < list.length; l++) {
			// figure out the column value number of this list
			auto &column_name = pivot_columns[list.offset + l];
			auto entry = pivot_map.find(column_name);
			if (entry == pivot_map.end()) {
				// column entry not found in map - that means this element is explicitly excluded from the pivot list
				continue;
			}
			auto column_idx = entry->second;
			for (idx_t aggr = 0; aggr < empty_aggregates.size(); aggr++) {
				auto pivot_value_lists = FlatVector::GetData<list_entry_t>(input.data[bound_pivot.group_count + aggr]);
				auto &pivot_value_child = ListVector::GetEntry(input.data[bound_pivot.group_count + aggr]);
				if (list.offset != pivot_value_lists[r].offset || list.length != pivot_value_lists[r].length) {
					throw InternalException("Pivot - unaligned lists between values and columns!?");
				}
				chunk.data[column_idx + aggr].SetValue(r, pivot_value_child.GetValue(list.offset + l));
			}
		}
	}
	chunk.SetCardinality(input.size());
	return OperatorResultType::NEED_MORE_INPUT;
}

} // namespace duckdb





namespace duckdb {

class ProjectionState : public OperatorState {
public:
	explicit ProjectionState(ExecutionContext &context, const vector<unique_ptr<Expression>> &expressions)
	    : executor(context.client, expressions) {
	}

	ExpressionExecutor executor;

public:
	void Finalize(const PhysicalOperator &op, ExecutionContext &context) override {
		context.thread.profiler.Flush(op, executor, "projection", 0);
	}
};

PhysicalProjection::PhysicalProjection(vector<LogicalType> types, vector<unique_ptr<Expression>> select_list,
                                       idx_t estimated_cardinality)
    : PhysicalOperator(PhysicalOperatorType::PROJECTION, std::move(types), estimated_cardinality),
      select_list(std::move(select_list)) {
}

OperatorResultType PhysicalProjection::Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                               GlobalOperatorState &gstate, OperatorState &state_p) const {
	auto &state = (ProjectionState &)state_p;
	state.executor.Execute(input, chunk);
	return OperatorResultType::NEED_MORE_INPUT;
}

unique_ptr<OperatorState> PhysicalProjection::GetOperatorState(ExecutionContext &context) const {
	return make_uniq<ProjectionState>(context, select_list);
}

unique_ptr<PhysicalOperator>
PhysicalProjection::CreateJoinProjection(vector<LogicalType> proj_types, const vector<LogicalType> &lhs_types,
                                         const vector<LogicalType> &rhs_types, const vector<idx_t> &left_projection_map,
                                         const vector<idx_t> &right_projection_map, const idx_t estimated_cardinality) {

	vector<unique_ptr<Expression>> proj_selects;
	proj_selects.reserve(proj_types.size());

	if (left_projection_map.empty()) {
		for (storage_t i = 0; i < lhs_types.size(); ++i) {
			proj_selects.emplace_back(make_uniq<BoundReferenceExpression>(lhs_types[i], i));
		}
	} else {
		for (auto i : left_projection_map) {
			proj_selects.emplace_back(make_uniq<BoundReferenceExpression>(lhs_types[i], i));
		}
	}
	const auto left_cols = lhs_types.size();

	if (right_projection_map.empty()) {
		for (storage_t i = 0; i < rhs_types.size(); ++i) {
			proj_selects.emplace_back(make_uniq<BoundReferenceExpression>(rhs_types[i], left_cols + i));
		}

	} else {
		for (auto i : right_projection_map) {
			proj_selects.emplace_back(make_uniq<BoundReferenceExpression>(rhs_types[i], left_cols + i));
		}
	}

	return make_uniq<PhysicalProjection>(std::move(proj_types), std::move(proj_selects), estimated_cardinality);
}

string PhysicalProjection::ParamsToString() const {
	string extra_info;
	for (auto &expr : select_list) {
		extra_info += expr->GetName() + "\n";
	}
	return extra_info;
}

} // namespace duckdb


namespace duckdb {

class TableInOutLocalState : public OperatorState {
public:
	TableInOutLocalState() : row_index(0), new_row(true) {
	}

	unique_ptr<LocalTableFunctionState> local_state;
	idx_t row_index;
	bool new_row;
	DataChunk input_chunk;
};

class TableInOutGlobalState : public GlobalOperatorState {
public:
	TableInOutGlobalState() {
	}

	unique_ptr<GlobalTableFunctionState> global_state;
};

PhysicalTableInOutFunction::PhysicalTableInOutFunction(vector<LogicalType> types, TableFunction function_p,
                                                       unique_ptr<FunctionData> bind_data_p,
                                                       vector<column_t> column_ids_p, idx_t estimated_cardinality,
                                                       vector<column_t> project_input_p)
    : PhysicalOperator(PhysicalOperatorType::INOUT_FUNCTION, std::move(types), estimated_cardinality),
      function(std::move(function_p)), bind_data(std::move(bind_data_p)), column_ids(std::move(column_ids_p)),
      projected_input(std::move(project_input_p)) {
}

unique_ptr<OperatorState> PhysicalTableInOutFunction::GetOperatorState(ExecutionContext &context) const {
	auto &gstate = op_state->Cast<TableInOutGlobalState>();
	auto result = make_uniq<TableInOutLocalState>();
	if (function.init_local) {
		TableFunctionInitInput input(bind_data.get(), column_ids, vector<idx_t>(), nullptr);
		result->local_state = function.init_local(context, input, gstate.global_state.get());
	}
	if (!projected_input.empty()) {
		result->input_chunk.Initialize(context.client, children[0]->types);
	}
	return std::move(result);
}

unique_ptr<GlobalOperatorState> PhysicalTableInOutFunction::GetGlobalOperatorState(ClientContext &context) const {
	auto result = make_uniq<TableInOutGlobalState>();
	if (function.init_global) {
		TableFunctionInitInput input(bind_data.get(), column_ids, vector<idx_t>(), nullptr);
		result->global_state = function.init_global(context, input);
	}
	return std::move(result);
}

OperatorResultType PhysicalTableInOutFunction::Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                                       GlobalOperatorState &gstate_p, OperatorState &state_p) const {
	auto &gstate = gstate_p.Cast<TableInOutGlobalState>();
	auto &state = state_p.Cast<TableInOutLocalState>();
	TableFunctionInput data(bind_data.get(), state.local_state.get(), gstate.global_state.get());
	if (projected_input.empty()) {
		// straightforward case - no need to project input
		return function.in_out_function(context, data, input, chunk);
	}
	// when project_input is set we execute the input function row-by-row
	if (state.new_row) {
		if (state.row_index >= input.size()) {
			// finished processing this chunk
			state.new_row = true;
			state.row_index = 0;
			return OperatorResultType::NEED_MORE_INPUT;
		}
		// we are processing a new row: fetch the data for the current row
		D_ASSERT(input.ColumnCount() == state.input_chunk.ColumnCount());
		// set up the input data to the table in-out function
		for (idx_t col_idx = 0; col_idx < input.ColumnCount(); col_idx++) {
			ConstantVector::Reference(state.input_chunk.data[col_idx], input.data[col_idx], state.row_index, 1);
		}
		state.input_chunk.SetCardinality(1);
		state.row_index++;
		state.new_row = false;
	}
	// set up the output data in "chunk"
	D_ASSERT(chunk.ColumnCount() > projected_input.size());
	D_ASSERT(state.row_index > 0);
	idx_t base_idx = chunk.ColumnCount() - projected_input.size();
	for (idx_t project_idx = 0; project_idx < projected_input.size(); project_idx++) {
		auto source_idx = projected_input[project_idx];
		auto target_idx = base_idx + project_idx;
		ConstantVector::Reference(chunk.data[target_idx], input.data[source_idx], state.row_index - 1, 1);
	}
	auto result = function.in_out_function(context, data, state.input_chunk, chunk);
	if (result == OperatorResultType::FINISHED) {
		return result;
	}
	if (result == OperatorResultType::NEED_MORE_INPUT) {
		// we finished processing this row: move to the next row
		state.new_row = true;
	}
	return OperatorResultType::HAVE_MORE_OUTPUT;
}

OperatorFinalizeResultType PhysicalTableInOutFunction::FinalExecute(ExecutionContext &context, DataChunk &chunk,
                                                                    GlobalOperatorState &gstate_p,
                                                                    OperatorState &state_p) const {
	auto &gstate = gstate_p.Cast<TableInOutGlobalState>();
	auto &state = state_p.Cast<TableInOutLocalState>();
	if (!projected_input.empty()) {
		throw InternalException("FinalExecute not supported for project_input");
	}
	TableFunctionInput data(bind_data.get(), state.local_state.get(), gstate.global_state.get());
	return function.in_out_function_final(context, data, chunk);
}

} // namespace duckdb








namespace duckdb {

class UnnestOperatorState : public OperatorState {
public:
	UnnestOperatorState(ClientContext &context, const vector<unique_ptr<Expression>> &select_list)
	    : current_row(0), list_position(0), longest_list_length(DConstants::INVALID_INDEX), first_fetch(true),
	      executor(context) {

		// for each UNNEST in the select_list, we add the child expression to the expression executor
		// and set the return type in the list_data chunk, which will contain the evaluated expression results
		vector<LogicalType> list_data_types;
		for (auto &exp : select_list) {
			D_ASSERT(exp->type == ExpressionType::BOUND_UNNEST);
			auto &bue = exp->Cast<BoundUnnestExpression>();
			list_data_types.push_back(bue.child->return_type);
			executor.AddExpression(*bue.child.get());
		}

		auto &allocator = Allocator::Get(context);
		list_data.Initialize(allocator, list_data_types);

		list_vector_data.resize(list_data.ColumnCount());
		list_child_data.resize(list_data.ColumnCount());
	}

	idx_t current_row;
	idx_t list_position;
	idx_t longest_list_length;
	bool first_fetch;

	ExpressionExecutor executor;
	DataChunk list_data;
	vector<UnifiedVectorFormat> list_vector_data;
	vector<UnifiedVectorFormat> list_child_data;

public:
	//! Reset the fields of the unnest operator state
	void Reset();
	//! Set the longest list's length for the current row
	void SetLongestListLength();
};

void UnnestOperatorState::Reset() {
	current_row = 0;
	list_position = 0;
	longest_list_length = DConstants::INVALID_INDEX;
	first_fetch = true;
}

void UnnestOperatorState::SetLongestListLength() {
	longest_list_length = 0;
	for (idx_t col_idx = 0; col_idx < list_data.ColumnCount(); col_idx++) {

		auto &vector_data = list_vector_data[col_idx];
		auto current_idx = vector_data.sel->get_index(current_row);

		if (vector_data.validity.RowIsValid(current_idx)) {

			// check if this list is longer
			auto list_data = (list_entry_t *)vector_data.data;
			auto list_entry = list_data[current_idx];
			if (list_entry.length > longest_list_length) {
				longest_list_length = list_entry.length;
			}
		}
	}
}

PhysicalUnnest::PhysicalUnnest(vector<LogicalType> types, vector<unique_ptr<Expression>> select_list,
                               idx_t estimated_cardinality, PhysicalOperatorType type)
    : PhysicalOperator(type, std::move(types), estimated_cardinality), select_list(std::move(select_list)) {
	D_ASSERT(!this->select_list.empty());
}

static void UnnestNull(idx_t start, idx_t end, Vector &result) {

	D_ASSERT(result.GetVectorType() == VectorType::FLAT_VECTOR);
	auto &validity = FlatVector::Validity(result);
	for (idx_t i = start; i < end; i++) {
		validity.SetInvalid(i);
	}
	if (result.GetType().InternalType() == PhysicalType::STRUCT) {
		auto &struct_children = StructVector::GetEntries(result);
		for (auto &child : struct_children) {
			UnnestNull(start, end, *child);
		}
	}
}

template <class T>
static void TemplatedUnnest(UnifiedVectorFormat &vector_data, idx_t start, idx_t end, Vector &result) {

	auto source_data = (T *)vector_data.data;
	auto &source_mask = vector_data.validity;

	D_ASSERT(result.GetVectorType() == VectorType::FLAT_VECTOR);
	auto result_data = FlatVector::GetData<T>(result);
	auto &result_mask = FlatVector::Validity(result);

	for (idx_t i = start; i < end; i++) {
		auto source_idx = vector_data.sel->get_index(i);
		auto target_idx = i - start;
		if (source_mask.RowIsValid(source_idx)) {
			result_data[target_idx] = source_data[source_idx];
			result_mask.SetValid(target_idx);
		} else {
			result_mask.SetInvalid(target_idx);
		}
	}
}

static void UnnestValidity(UnifiedVectorFormat &vector_data, idx_t start, idx_t end, Vector &result) {

	auto &source_mask = vector_data.validity;
	D_ASSERT(result.GetVectorType() == VectorType::FLAT_VECTOR);
	auto &result_mask = FlatVector::Validity(result);

	for (idx_t i = start; i < end; i++) {
		auto source_idx = vector_data.sel->get_index(i);
		auto target_idx = i - start;
		result_mask.Set(target_idx, source_mask.RowIsValid(source_idx));
	}
}

static void UnnestVector(UnifiedVectorFormat &child_vector_data, Vector &child_vector, idx_t list_size, idx_t start,
                         idx_t end, Vector &result) {

	D_ASSERT(child_vector.GetType() == result.GetType());
	switch (result.GetType().InternalType()) {
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
		TemplatedUnnest<int8_t>(child_vector_data, start, end, result);
		break;
	case PhysicalType::INT16:
		TemplatedUnnest<int16_t>(child_vector_data, start, end, result);
		break;
	case PhysicalType::INT32:
		TemplatedUnnest<int32_t>(child_vector_data, start, end, result);
		break;
	case PhysicalType::INT64:
		TemplatedUnnest<int64_t>(child_vector_data, start, end, result);
		break;
	case PhysicalType::INT128:
		TemplatedUnnest<hugeint_t>(child_vector_data, start, end, result);
		break;
	case PhysicalType::UINT8:
		TemplatedUnnest<uint8_t>(child_vector_data, start, end, result);
		break;
	case PhysicalType::UINT16:
		TemplatedUnnest<uint16_t>(child_vector_data, start, end, result);
		break;
	case PhysicalType::UINT32:
		TemplatedUnnest<uint32_t>(child_vector_data, start, end, result);
		break;
	case PhysicalType::UINT64:
		TemplatedUnnest<uint64_t>(child_vector_data, start, end, result);
		break;
	case PhysicalType::FLOAT:
		TemplatedUnnest<float>(child_vector_data, start, end, result);
		break;
	case PhysicalType::DOUBLE:
		TemplatedUnnest<double>(child_vector_data, start, end, result);
		break;
	case PhysicalType::INTERVAL:
		TemplatedUnnest<interval_t>(child_vector_data, start, end, result);
		break;
	case PhysicalType::VARCHAR:
		TemplatedUnnest<string_t>(child_vector_data, start, end, result);
		break;
	case PhysicalType::LIST: {
		// the child vector of result now references the child vector source
		// FIXME: only reference relevant children (start - end) instead of all
		auto &target = ListVector::GetEntry(result);
		target.Reference(ListVector::GetEntry(child_vector));
		ListVector::SetListSize(result, ListVector::GetListSize(child_vector));
		// unnest
		TemplatedUnnest<list_entry_t>(child_vector_data, start, end, result);
		break;
	}
	case PhysicalType::STRUCT: {
		auto &child_vector_entries = StructVector::GetEntries(child_vector);
		auto &result_entries = StructVector::GetEntries(result);

		// set the validity mask for the 'outer' struct vector before unnesting its children
		UnnestValidity(child_vector_data, start, end, result);

		for (idx_t i = 0; i < child_vector_entries.size(); i++) {
			UnifiedVectorFormat child_vector_entries_data;
			child_vector_entries[i]->ToUnifiedFormat(list_size, child_vector_entries_data);
			UnnestVector(child_vector_entries_data, *child_vector_entries[i], list_size, start, end,
			             *result_entries[i]);
		}
		break;
	}
	default:
		throw InternalException("Unimplemented type for UNNEST.");
	}
}

static void PrepareInput(UnnestOperatorState &state, DataChunk &input,
                         const vector<unique_ptr<Expression>> &select_list) {

	state.list_data.Reset();
	// execute the expressions inside each UNNEST in the select_list to get the list data
	// execution results (lists) are kept in state.list_data chunk
	state.executor.Execute(input, state.list_data);

	// verify incoming lists
	state.list_data.Verify();
	D_ASSERT(input.size() == state.list_data.size());
	D_ASSERT(state.list_data.ColumnCount() == select_list.size());
	D_ASSERT(state.list_vector_data.size() == state.list_data.ColumnCount());
	D_ASSERT(state.list_child_data.size() == state.list_data.ColumnCount());

	// get the UnifiedVectorFormat of each list_data vector (LIST vectors for the different UNNESTs)
	// both for the vector itself and its child vector
	for (idx_t col_idx = 0; col_idx < state.list_data.ColumnCount(); col_idx++) {

		auto &list_vector = state.list_data.data[col_idx];
		list_vector.ToUnifiedFormat(state.list_data.size(), state.list_vector_data[col_idx]);

		if (list_vector.GetType() == LogicalType::SQLNULL) {
			// UNNEST(NULL): SQLNULL vectors don't have child vectors, but we need to point to the child vector of
			// each vector, so we just get the UnifiedVectorFormat of the vector itself
			auto &child_vector = list_vector;
			child_vector.ToUnifiedFormat(0, state.list_child_data[col_idx]);
		} else {
			auto list_size = ListVector::GetListSize(list_vector);
			auto &child_vector = ListVector::GetEntry(list_vector);
			child_vector.ToUnifiedFormat(list_size, state.list_child_data[col_idx]);
		}
	}

	state.first_fetch = false;
}

unique_ptr<OperatorState> PhysicalUnnest::GetOperatorState(ExecutionContext &context) const {
	return PhysicalUnnest::GetState(context, select_list);
}

unique_ptr<OperatorState> PhysicalUnnest::GetState(ExecutionContext &context,
                                                   const vector<unique_ptr<Expression>> &select_list) {
	return make_uniq<UnnestOperatorState>(context.client, select_list);
}

OperatorResultType PhysicalUnnest::ExecuteInternal(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                                   OperatorState &state_p,
                                                   const vector<unique_ptr<Expression>> &select_list,
                                                   bool include_input) {

	auto &state = state_p.Cast<UnnestOperatorState>();

	do {
		// prepare the input data by executing any expressions and getting the
		// UnifiedVectorFormat of each LIST vector (list_vector_data) and its child vector (list_child_data)
		if (state.first_fetch) {
			PrepareInput(state, input, select_list);
		}

		// finished with all rows of this input chunk, reset
		if (state.current_row >= input.size()) {
			state.Reset();
			return OperatorResultType::NEED_MORE_INPUT;
		}

		// each UNNEST in the select_list contains a list (or NULL) for this row, find longest list
		// because this length determines how many times we need to repeat for the current row
		if (state.longest_list_length == DConstants::INVALID_INDEX) {
			state.SetLongestListLength();
		}
		D_ASSERT(state.longest_list_length != DConstants::INVALID_INDEX);

		// we emit chunks of either STANDARD_VECTOR_SIZE or smaller
		auto this_chunk_len = MinValue<idx_t>(STANDARD_VECTOR_SIZE, state.longest_list_length - state.list_position);
		chunk.SetCardinality(this_chunk_len);

		// if we include other projection input columns, e.g. SELECT 1, UNNEST([1, 2]);, then
		// we need to add them as a constant vector to the resulting chunk
		// FIXME: emit multiple unnested rows. Currently, we never emit a chunk containing multiple unnested input rows,
		//  so setting a constant vector for the value at state.current_row is fine
		idx_t col_offset = 0;
		if (include_input) {
			for (idx_t col_idx = 0; col_idx < input.ColumnCount(); col_idx++) {
				ConstantVector::Reference(chunk.data[col_idx], input.data[col_idx], state.current_row, input.size());
			}
			col_offset = input.ColumnCount();
		}

		// unnest the lists
		for (idx_t col_idx = 0; col_idx < state.list_data.ColumnCount(); col_idx++) {

			auto &result_vector = chunk.data[col_idx + col_offset];

			if (state.list_data.data[col_idx].GetType() == LogicalType::SQLNULL) {
				// UNNEST(NULL)
				chunk.SetCardinality(0);
				break;

			} else {

				auto &vector_data = state.list_vector_data[col_idx];
				auto current_idx = vector_data.sel->get_index(state.current_row);

				if (!vector_data.validity.RowIsValid(current_idx)) {
					UnnestNull(0, this_chunk_len, result_vector);

				} else {

					auto list_data = (list_entry_t *)vector_data.data;
					auto list_entry = list_data[current_idx];

					idx_t list_count = 0;
					if (state.list_position < list_entry.length) {
						// there are still list_count elements to unnest
						list_count = MinValue<idx_t>(this_chunk_len, list_entry.length - state.list_position);

						auto &list_vector = state.list_data.data[col_idx];
						auto &child_vector = ListVector::GetEntry(list_vector);
						auto list_size = ListVector::GetListSize(list_vector);
						auto &child_vector_data = state.list_child_data[col_idx];

						auto base_offset = list_entry.offset + state.list_position;
						UnnestVector(child_vector_data, child_vector, list_size, base_offset, base_offset + list_count,
						             result_vector);
					}

					// fill the rest with NULLs
					if (list_count != this_chunk_len) {
						UnnestNull(list_count, this_chunk_len, result_vector);
					}
				}
			}
		}

		chunk.Verify();

		state.list_position += this_chunk_len;
		if (state.list_position == state.longest_list_length) {
			state.current_row++;
			state.longest_list_length = DConstants::INVALID_INDEX;
			state.list_position = 0;
		}

		// we only emit one unnested row (that contains data) at a time
	} while (chunk.size() == 0);
	return OperatorResultType::HAVE_MORE_OUTPUT;
}

OperatorResultType PhysicalUnnest::Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                           GlobalOperatorState &, OperatorState &state) const {
	return ExecuteInternal(context, input, chunk, state, select_list);
}

} // namespace duckdb






namespace duckdb {

PhysicalColumnDataScan::PhysicalColumnDataScan(vector<LogicalType> types, PhysicalOperatorType op_type,
                                               idx_t estimated_cardinality,
                                               unique_ptr<ColumnDataCollection> owned_collection_p)
    : PhysicalOperator(op_type, std::move(types), estimated_cardinality), collection(owned_collection_p.get()),
      owned_collection(std::move(owned_collection_p)) {
}

class PhysicalColumnDataScanState : public GlobalSourceState {
public:
	explicit PhysicalColumnDataScanState() : initialized(false) {
	}

	//! The current position in the scan
	ColumnDataScanState scan_state;
	bool initialized;
};

unique_ptr<GlobalSourceState> PhysicalColumnDataScan::GetGlobalSourceState(ClientContext &context) const {
	return make_uniq<PhysicalColumnDataScanState>();
}

SourceResultType PhysicalColumnDataScan::GetData(ExecutionContext &context, DataChunk &chunk,
                                                 OperatorSourceInput &input) const {
	auto &state = input.global_state.Cast<PhysicalColumnDataScanState>();
	if (collection->Count() == 0) {
		return SourceResultType::FINISHED;
	}
	if (!state.initialized) {
		collection->InitializeScan(state.scan_state);
		state.initialized = true;
	}
	collection->Scan(state.scan_state, chunk);

	return chunk.size() == 0 ? SourceResultType::FINISHED : SourceResultType::HAVE_MORE_OUTPUT;
}

//===--------------------------------------------------------------------===//
// Pipeline Construction
//===--------------------------------------------------------------------===//
void PhysicalColumnDataScan::BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) {
	// check if there is any additional action we need to do depending on the type
	auto &state = meta_pipeline.GetState();
	switch (type) {
	case PhysicalOperatorType::DELIM_SCAN: {
		auto entry = state.delim_join_dependencies.find(*this);
		D_ASSERT(entry != state.delim_join_dependencies.end());
		// this chunk scan introduces a dependency to the current pipeline
		// namely a dependency on the duplicate elimination pipeline to finish
		auto delim_dependency = entry->second.get().shared_from_this();
		auto delim_sink = state.GetPipelineSink(*delim_dependency);
		D_ASSERT(delim_sink);
		D_ASSERT(delim_sink->type == PhysicalOperatorType::DELIM_JOIN);
		auto &delim_join = delim_sink->Cast<PhysicalDelimJoin>();
		current.AddDependency(delim_dependency);
		state.SetPipelineSource(current, (PhysicalOperator &)*delim_join.distinct);
		return;
	}
	case PhysicalOperatorType::RECURSIVE_CTE_SCAN:
		if (!meta_pipeline.HasRecursiveCTE()) {
			throw InternalException("Recursive CTE scan found without recursive CTE node");
		}
		break;
	default:
		break;
	}
	D_ASSERT(children.empty());
	state.SetPipelineSource(current, *this);
}

} // namespace duckdb


namespace duckdb {

SourceResultType PhysicalDummyScan::GetData(ExecutionContext &context, DataChunk &chunk,
                                            OperatorSourceInput &input) const {
	// return a single row on the first call to the dummy scan
	chunk.SetCardinality(1);

	return SourceResultType::FINISHED;
}

} // namespace duckdb


namespace duckdb {

SourceResultType PhysicalEmptyResult::GetData(ExecutionContext &context, DataChunk &chunk,
                                              OperatorSourceInput &input) const {
	return SourceResultType::FINISHED;
}

} // namespace duckdb




namespace duckdb {

class ExpressionScanState : public OperatorState {
public:
	explicit ExpressionScanState(Allocator &allocator, const PhysicalExpressionScan &op) : expression_index(0) {
		temp_chunk.Initialize(allocator, op.GetTypes());
	}

	//! The current position in the scan
	idx_t expression_index;
	//! Temporary chunk for evaluating expressions
	DataChunk temp_chunk;
};

unique_ptr<OperatorState> PhysicalExpressionScan::GetOperatorState(ExecutionContext &context) const {
	return make_uniq<ExpressionScanState>(Allocator::Get(context.client), *this);
}

OperatorResultType PhysicalExpressionScan::Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                                   GlobalOperatorState &gstate, OperatorState &state_p) const {
	auto &state = (ExpressionScanState &)state_p;

	for (; chunk.size() + input.size() <= STANDARD_VECTOR_SIZE && state.expression_index < expressions.size();
	     state.expression_index++) {
		state.temp_chunk.Reset();
		EvaluateExpression(context.client, state.expression_index, &input, state.temp_chunk);
		chunk.Append(state.temp_chunk);
	}
	if (state.expression_index < expressions.size()) {
		return OperatorResultType::HAVE_MORE_OUTPUT;
	} else {
		state.expression_index = 0;
		return OperatorResultType::NEED_MORE_INPUT;
	}
}

void PhysicalExpressionScan::EvaluateExpression(ClientContext &context, idx_t expression_idx, DataChunk *child_chunk,
                                                DataChunk &result) const {
	ExpressionExecutor executor(context, expressions[expression_idx]);
	if (child_chunk) {
		child_chunk->Verify();
		executor.Execute(*child_chunk, result);
	} else {
		executor.Execute(result);
	}
}

bool PhysicalExpressionScan::IsFoldable() const {
	for (auto &expr_list : expressions) {
		for (auto &expr : expr_list) {
			if (!expr->IsFoldable()) {
				return false;
			}
		}
	}
	return true;
}

} // namespace duckdb








#include <utility>

namespace duckdb {

PhysicalPositionalScan::PhysicalPositionalScan(vector<LogicalType> types, unique_ptr<PhysicalOperator> left,
                                               unique_ptr<PhysicalOperator> right)
    : PhysicalOperator(PhysicalOperatorType::POSITIONAL_SCAN, std::move(types),
                       MaxValue(left->estimated_cardinality, right->estimated_cardinality)) {

	// Manage the children ourselves
	if (left->type == PhysicalOperatorType::TABLE_SCAN) {
		child_tables.emplace_back(std::move(left));
	} else if (left->type == PhysicalOperatorType::POSITIONAL_SCAN) {
		auto &left_scan = (PhysicalPositionalScan &)*left;
		child_tables = std::move(left_scan.child_tables);
	} else {
		throw InternalException("Invalid left input for PhysicalPositionalScan");
	}

	if (right->type == PhysicalOperatorType::TABLE_SCAN) {
		child_tables.emplace_back(std::move(right));
	} else if (right->type == PhysicalOperatorType::POSITIONAL_SCAN) {
		auto &right_scan = (PhysicalPositionalScan &)*right;
		auto &right_tables = right_scan.child_tables;
		child_tables.reserve(child_tables.size() + right_tables.size());
		std::move(right_tables.begin(), right_tables.end(), std::back_inserter(child_tables));
	} else {
		throw InternalException("Invalid right input for PhysicalPositionalScan");
	}
}

class PositionalScanGlobalSourceState : public GlobalSourceState {
public:
	PositionalScanGlobalSourceState(ClientContext &context, const PhysicalPositionalScan &op) {
		for (const auto &table : op.child_tables) {
			global_states.emplace_back(table->GetGlobalSourceState(context));
		}
	}

	vector<unique_ptr<GlobalSourceState>> global_states;

	idx_t MaxThreads() override {
		return 1;
	}
};

class PositionalTableScanner {
public:
	PositionalTableScanner(ExecutionContext &context, PhysicalOperator &table_p, GlobalSourceState &gstate_p)
	    : table(table_p), global_state(gstate_p), source_offset(0), exhausted(false) {
		local_state = table.GetLocalSourceState(context, gstate_p);
		source.Initialize(Allocator::Get(context.client), table.types);
	}

	idx_t Refill(ExecutionContext &context) {
		if (source_offset >= source.size()) {
			if (!exhausted) {
				source.Reset();

				InterruptState interrupt_state;
				OperatorSourceInput source_input {global_state, *local_state, interrupt_state};
				auto source_result = table.GetData(context, source, source_input);
				if (source_result == SourceResultType::BLOCKED) {
					throw NotImplementedException(
					    "Unexpected interrupt from table Source in PositionalTableScanner refill");
				}
			}
			source_offset = 0;
		}

		const auto available = source.size() - source_offset;
		if (!available) {
			if (!exhausted) {
				source.Reset();
				for (idx_t i = 0; i < source.ColumnCount(); ++i) {
					auto &vec = source.data[i];
					vec.SetVectorType(VectorType::CONSTANT_VECTOR);
					ConstantVector::SetNull(vec, true);
				}
				exhausted = true;
			}
		}

		return available;
	}

	idx_t CopyData(ExecutionContext &context, DataChunk &output, const idx_t count, const idx_t col_offset) {
		if (!source_offset && (source.size() >= count || exhausted)) {
			//	Fast track: aligned and has enough data
			for (idx_t i = 0; i < source.ColumnCount(); ++i) {
				output.data[col_offset + i].Reference(source.data[i]);
			}
			source_offset += count;
		} else {
			// Copy data
			for (idx_t target_offset = 0; target_offset < count;) {
				const auto needed = count - target_offset;
				const auto available = exhausted ? needed : (source.size() - source_offset);
				const auto copy_size = MinValue(needed, available);
				const auto source_count = source_offset + copy_size;
				for (idx_t i = 0; i < source.ColumnCount(); ++i) {
					VectorOperations::Copy(source.data[i], output.data[col_offset + i], source_count, source_offset,
					                       target_offset);
				}
				target_offset += copy_size;
				source_offset += copy_size;
				Refill(context);
			}
		}

		return source.ColumnCount();
	}

	double GetProgress(ClientContext &context) {
		return table.GetProgress(context, global_state);
	}

	PhysicalOperator &table;
	GlobalSourceState &global_state;
	unique_ptr<LocalSourceState> local_state;
	DataChunk source;
	idx_t source_offset;
	bool exhausted;
};

class PositionalScanLocalSourceState : public LocalSourceState {
public:
	PositionalScanLocalSourceState(ExecutionContext &context, PositionalScanGlobalSourceState &gstate,
	                               const PhysicalPositionalScan &op) {
		for (size_t i = 0; i < op.child_tables.size(); ++i) {
			auto &child = *op.child_tables[i];
			auto &global_state = *gstate.global_states[i];
			scanners.emplace_back(make_uniq<PositionalTableScanner>(context, child, global_state));
		}
	}

	vector<unique_ptr<PositionalTableScanner>> scanners;
};

unique_ptr<LocalSourceState> PhysicalPositionalScan::GetLocalSourceState(ExecutionContext &context,
                                                                         GlobalSourceState &gstate) const {
	return make_uniq<PositionalScanLocalSourceState>(context, gstate.Cast<PositionalScanGlobalSourceState>(), *this);
}

unique_ptr<GlobalSourceState> PhysicalPositionalScan::GetGlobalSourceState(ClientContext &context) const {
	return make_uniq<PositionalScanGlobalSourceState>(context, *this);
}

SourceResultType PhysicalPositionalScan::GetData(ExecutionContext &context, DataChunk &output,
                                                 OperatorSourceInput &input) const {
	auto &lstate = input.local_state.Cast<PositionalScanLocalSourceState>();

	// Find the longest source block
	idx_t count = 0;
	for (auto &scanner : lstate.scanners) {
		count = MaxValue(count, scanner->Refill(context));
	}

	//	All done?
	if (!count) {
		return SourceResultType::FINISHED;
	}

	// Copy or reference the source columns
	idx_t col_offset = 0;
	for (auto &scanner : lstate.scanners) {
		col_offset += scanner->CopyData(context, output, count, col_offset);
	}

	output.SetCardinality(count);
	return SourceResultType::HAVE_MORE_OUTPUT;
}

double PhysicalPositionalScan::GetProgress(ClientContext &context, GlobalSourceState &gstate_p) const {
	auto &gstate = gstate_p.Cast<PositionalScanGlobalSourceState>();

	double result = child_tables[0]->GetProgress(context, *gstate.global_states[0]);
	for (size_t t = 1; t < child_tables.size(); ++t) {
		result = MinValue(result, child_tables[t]->GetProgress(context, *gstate.global_states[t]));
	}

	return result;
}

bool PhysicalPositionalScan::Equals(const PhysicalOperator &other_p) const {
	if (type != other_p.type) {
		return false;
	}

	auto &other = other_p.Cast<PhysicalPositionalScan>();
	if (child_tables.size() != other.child_tables.size()) {
		return false;
	}
	for (size_t i = 0; i < child_tables.size(); ++i) {
		if (!child_tables[i]->Equals(*other.child_tables[i])) {
			return false;
		}
	}

	return true;
}

} // namespace duckdb







#include <utility>

namespace duckdb {

PhysicalTableScan::PhysicalTableScan(vector<LogicalType> types, TableFunction function_p,
                                     unique_ptr<FunctionData> bind_data_p, vector<column_t> column_ids_p,
                                     vector<string> names_p, unique_ptr<TableFilterSet> table_filters_p,
                                     idx_t estimated_cardinality)
    : PhysicalOperator(PhysicalOperatorType::TABLE_SCAN, std::move(types), estimated_cardinality),
      function(std::move(function_p)), bind_data(std::move(bind_data_p)), column_ids(std::move(column_ids_p)),
      names(std::move(names_p)), table_filters(std::move(table_filters_p)) {
}

PhysicalTableScan::PhysicalTableScan(vector<LogicalType> types, TableFunction function_p,
                                     unique_ptr<FunctionData> bind_data_p, vector<LogicalType> returned_types_p,
                                     vector<column_t> column_ids_p, vector<idx_t> projection_ids_p,
                                     vector<string> names_p, unique_ptr<TableFilterSet> table_filters_p,
                                     idx_t estimated_cardinality)
    : PhysicalOperator(PhysicalOperatorType::TABLE_SCAN, std::move(types), estimated_cardinality),
      function(std::move(function_p)), bind_data(std::move(bind_data_p)), returned_types(std::move(returned_types_p)),
      column_ids(std::move(column_ids_p)), projection_ids(std::move(projection_ids_p)), names(std::move(names_p)),
      table_filters(std::move(table_filters_p)) {
}

class TableScanGlobalSourceState : public GlobalSourceState {
public:
	TableScanGlobalSourceState(ClientContext &context, const PhysicalTableScan &op) {
		if (op.function.init_global) {
			TableFunctionInitInput input(op.bind_data.get(), op.column_ids, op.projection_ids, op.table_filters.get());
			global_state = op.function.init_global(context, input);
			if (global_state) {
				max_threads = global_state->MaxThreads();
			}
		} else {
			max_threads = 1;
		}
	}

	idx_t max_threads = 0;
	unique_ptr<GlobalTableFunctionState> global_state;

	idx_t MaxThreads() override {
		return max_threads;
	}
};

class TableScanLocalSourceState : public LocalSourceState {
public:
	TableScanLocalSourceState(ExecutionContext &context, TableScanGlobalSourceState &gstate,
	                          const PhysicalTableScan &op) {
		if (op.function.init_local) {
			TableFunctionInitInput input(op.bind_data.get(), op.column_ids, op.projection_ids, op.table_filters.get());
			local_state = op.function.init_local(context, input, gstate.global_state.get());
		}
	}

	unique_ptr<LocalTableFunctionState> local_state;
};

unique_ptr<LocalSourceState> PhysicalTableScan::GetLocalSourceState(ExecutionContext &context,
                                                                    GlobalSourceState &gstate) const {
	return make_uniq<TableScanLocalSourceState>(context, gstate.Cast<TableScanGlobalSourceState>(), *this);
}

unique_ptr<GlobalSourceState> PhysicalTableScan::GetGlobalSourceState(ClientContext &context) const {
	return make_uniq<TableScanGlobalSourceState>(context, *this);
}

SourceResultType PhysicalTableScan::GetData(ExecutionContext &context, DataChunk &chunk,
                                            OperatorSourceInput &input) const {
	D_ASSERT(!column_ids.empty());
	auto &gstate = input.global_state.Cast<TableScanGlobalSourceState>();
	auto &state = input.local_state.Cast<TableScanLocalSourceState>();

	TableFunctionInput data(bind_data.get(), state.local_state.get(), gstate.global_state.get());
	function.function(context.client, data, chunk);

	return chunk.size() == 0 ? SourceResultType::FINISHED : SourceResultType::HAVE_MORE_OUTPUT;
}

double PhysicalTableScan::GetProgress(ClientContext &context, GlobalSourceState &gstate_p) const {
	auto &gstate = gstate_p.Cast<TableScanGlobalSourceState>();
	if (function.table_scan_progress) {
		return function.table_scan_progress(context, bind_data.get(), gstate.global_state.get());
	}
	// if table_scan_progress is not implemented we don't support this function yet in the progress bar
	return -1;
}

idx_t PhysicalTableScan::GetBatchIndex(ExecutionContext &context, DataChunk &chunk, GlobalSourceState &gstate_p,
                                       LocalSourceState &lstate) const {
	D_ASSERT(SupportsBatchIndex());
	D_ASSERT(function.get_batch_index);
	auto &gstate = gstate_p.Cast<TableScanGlobalSourceState>();
	auto &state = lstate.Cast<TableScanLocalSourceState>();
	return function.get_batch_index(context.client, bind_data.get(), state.local_state.get(),
	                                gstate.global_state.get());
}

string PhysicalTableScan::GetName() const {
	return StringUtil::Upper(function.name + " " + function.extra_info);
}

string PhysicalTableScan::ParamsToString() const {
	string result;
	if (function.to_string) {
		result = function.to_string(bind_data.get());
		result += "\n[INFOSEPARATOR]\n";
	}
	if (function.projection_pushdown) {
		if (function.filter_prune) {
			for (idx_t i = 0; i < projection_ids.size(); i++) {
				const auto &column_id = column_ids[projection_ids[i]];
				if (column_id < names.size()) {
					if (i > 0) {
						result += "\n";
					}
					result += names[column_id];
				}
			}
		} else {
			for (idx_t i = 0; i < column_ids.size(); i++) {
				const auto &column_id = column_ids[i];
				if (column_id < names.size()) {
					if (i > 0) {
						result += "\n";
					}
					result += names[column_id];
				}
			}
		}
	}
	if (function.filter_pushdown && table_filters) {
		result += "\n[INFOSEPARATOR]\n";
		result += "Filters: ";
		for (auto &f : table_filters->filters) {
			auto &column_index = f.first;
			auto &filter = f.second;
			if (column_index < names.size()) {
				result += filter->ToString(names[column_ids[column_index]]);
				result += "\n";
			}
		}
	}
	result += "\n[INFOSEPARATOR]\n";
	result += StringUtil::Format("EC: %llu", estimated_props->GetCardinality<idx_t>());
	return result;
}

bool PhysicalTableScan::Equals(const PhysicalOperator &other_p) const {
	if (type != other_p.type) {
		return false;
	}
	auto &other = other_p.Cast<PhysicalTableScan>();
	if (function.function != other.function.function) {
		return false;
	}
	if (column_ids != other.column_ids) {
		return false;
	}
	if (!FunctionData::Equals(bind_data.get(), other.bind_data.get())) {
		return false;
	}
	return true;
}

} // namespace duckdb




namespace duckdb {

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
SourceResultType PhysicalAlter::GetData(ExecutionContext &context, DataChunk &chunk, OperatorSourceInput &input) const {
	auto &catalog = Catalog::GetCatalog(context.client, info->catalog);
	catalog.Alter(context.client, *info);

	return SourceResultType::FINISHED;
}

} // namespace duckdb











namespace duckdb {

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//

SourceResultType PhysicalAttach::GetData(ExecutionContext &context, DataChunk &chunk,
                                         OperatorSourceInput &input) const {
	// parse the options
	auto &config = DBConfig::GetConfig(context.client);
	AccessMode access_mode = config.options.access_mode;
	string type;
	string unrecognized_option;
	for (auto &entry : info->options) {
		if (entry.first == "readonly" || entry.first == "read_only") {
			auto read_only = BooleanValue::Get(entry.second.DefaultCastAs(LogicalType::BOOLEAN));
			if (read_only) {
				access_mode = AccessMode::READ_ONLY;
			} else {
				access_mode = AccessMode::READ_WRITE;
			}
		} else if (entry.first == "readwrite" || entry.first == "read_write") {
			auto read_only = !BooleanValue::Get(entry.second.DefaultCastAs(LogicalType::BOOLEAN));
			if (read_only) {
				access_mode = AccessMode::READ_ONLY;
			} else {
				access_mode = AccessMode::READ_WRITE;
			}
		} else if (entry.first == "type") {
			type = StringValue::Get(entry.second.DefaultCastAs(LogicalType::VARCHAR));
		} else if (unrecognized_option.empty()) {
			unrecognized_option = entry.first;
		}
	}
	auto &db = DatabaseInstance::GetDatabase(context.client);
	if (type.empty()) {
		// try to extract type from path
		auto path_and_type = DBPathAndType::Parse(info->path, config);
		type = path_and_type.type;
		info->path = path_and_type.path;
	}

	if (type.empty() && !unrecognized_option.empty()) {
		throw BinderException("Unrecognized option for attach \"%s\"", unrecognized_option);
	}

	// if we are loading a database type from an extension - check if that extension is loaded
	if (!type.empty()) {
		if (!db.ExtensionIsLoaded(type)) {
			ExtensionHelper::LoadExternalExtension(context.client, type);
		}
	}

	// attach the database
	auto &name = info->name;
	const auto &path = info->path;

	if (name.empty()) {
		name = AttachedDatabase::ExtractDatabaseName(path);
	}
	auto &db_manager = DatabaseManager::Get(context.client);
	auto existing_db = db_manager.GetDatabaseFromPath(context.client, path);
	if (existing_db) {
		throw BinderException("Database \"%s\" is already attached with alias \"%s\"", path, existing_db->GetName());
	}
	auto new_db = db.CreateAttachedDatabase(*info, type, access_mode);
	new_db->Initialize();

	db_manager.AddDatabase(context.client, std::move(new_db));

	return SourceResultType::FINISHED;
}

} // namespace duckdb





namespace duckdb {

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
SourceResultType PhysicalCreateFunction::GetData(ExecutionContext &context, DataChunk &chunk,
                                                 OperatorSourceInput &input) const {
	auto &catalog = Catalog::GetCatalog(context.client, info->catalog);
	catalog.CreateFunction(context.client, *info);

	return SourceResultType::FINISHED;
}

} // namespace duckdb










namespace duckdb {

PhysicalCreateIndex::PhysicalCreateIndex(LogicalOperator &op, TableCatalogEntry &table_p,
                                         const vector<column_t> &column_ids, unique_ptr<CreateIndexInfo> info,
                                         vector<unique_ptr<Expression>> unbound_expressions,
                                         idx_t estimated_cardinality)
    : PhysicalOperator(PhysicalOperatorType::CREATE_INDEX, op.types, estimated_cardinality),
      table(table_p.Cast<DuckTableEntry>()), info(std::move(info)),
      unbound_expressions(std::move(unbound_expressions)) {
	// convert virtual column ids to storage column ids
	for (auto &column_id : column_ids) {
		storage_ids.push_back(table.GetColumns().LogicalToPhysical(LogicalIndex(column_id)).index);
	}
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//

class CreateIndexGlobalSinkState : public GlobalSinkState {
public:
	//! Global index to be added to the table
	unique_ptr<Index> global_index;
};

class CreateIndexLocalSinkState : public LocalSinkState {
public:
	explicit CreateIndexLocalSinkState(ClientContext &context) : arena_allocator(Allocator::Get(context)) {};

	unique_ptr<Index> local_index;
	ArenaAllocator arena_allocator;
	vector<ARTKey> keys;
	DataChunk key_chunk;
	vector<column_t> key_column_ids;
};

unique_ptr<GlobalSinkState> PhysicalCreateIndex::GetGlobalSinkState(ClientContext &context) const {
	auto state = make_uniq<CreateIndexGlobalSinkState>();

	// create the global index
	switch (info->index_type) {
	case IndexType::ART: {
		auto &storage = table.GetStorage();
		state->global_index = make_uniq<ART>(storage_ids, TableIOManager::Get(storage), unbound_expressions,
		                                     info->constraint_type, storage.db);
		break;
	}
	default:
		throw InternalException("Unimplemented index type");
	}
	return (std::move(state));
}

unique_ptr<LocalSinkState> PhysicalCreateIndex::GetLocalSinkState(ExecutionContext &context) const {
	auto state = make_uniq<CreateIndexLocalSinkState>(context.client);

	// create the local index
	switch (info->index_type) {
	case IndexType::ART: {
		auto &storage = table.GetStorage();
		state->local_index = make_uniq<ART>(storage_ids, TableIOManager::Get(storage), unbound_expressions,
		                                    info->constraint_type, storage.db);
		break;
	}
	default:
		throw InternalException("Unimplemented index type");
	}
	state->keys = vector<ARTKey>(STANDARD_VECTOR_SIZE);
	state->key_chunk.Initialize(Allocator::Get(context.client), state->local_index->logical_types);

	for (idx_t i = 0; i < state->key_chunk.ColumnCount(); i++) {
		state->key_column_ids.push_back(i);
	}
	return std::move(state);
}

SinkResultType PhysicalCreateIndex::Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const {

	D_ASSERT(chunk.ColumnCount() >= 2);
	auto &lstate = input.local_state.Cast<CreateIndexLocalSinkState>();
	auto &row_identifiers = chunk.data[chunk.ColumnCount() - 1];

	// generate the keys for the given input
	lstate.key_chunk.ReferenceColumns(chunk, lstate.key_column_ids);
	lstate.arena_allocator.Reset();
	ART::GenerateKeys(lstate.arena_allocator, lstate.key_chunk, lstate.keys);

	auto &storage = table.GetStorage();
	auto art = make_uniq<ART>(lstate.local_index->column_ids, lstate.local_index->table_io_manager,
	                          lstate.local_index->unbound_expressions, lstate.local_index->constraint_type, storage.db);
	if (!art->ConstructFromSorted(lstate.key_chunk.size(), lstate.keys, row_identifiers)) {
		throw ConstraintException("Data contains duplicates on indexed column(s)");
	}

	// merge into the local ART
	if (!lstate.local_index->MergeIndexes(*art)) {
		throw ConstraintException("Data contains duplicates on indexed column(s)");
	}
	return SinkResultType::NEED_MORE_INPUT;
}

void PhysicalCreateIndex::Combine(ExecutionContext &context, GlobalSinkState &gstate_p,
                                  LocalSinkState &lstate_p) const {

	auto &gstate = gstate_p.Cast<CreateIndexGlobalSinkState>();
	auto &lstate = lstate_p.Cast<CreateIndexLocalSinkState>();

	// merge the local index into the global index
	if (!gstate.global_index->MergeIndexes(*lstate.local_index)) {
		throw ConstraintException("Data contains duplicates on indexed column(s)");
	}
}

SinkFinalizeType PhysicalCreateIndex::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                               GlobalSinkState &gstate_p) const {

	// here, we just set the resulting global index as the newly created index of the table

	auto &state = gstate_p.Cast<CreateIndexGlobalSinkState>();
	auto &storage = table.GetStorage();
	if (!storage.IsRoot()) {
		throw TransactionException("Transaction conflict: cannot add an index to a table that has been altered!");
	}

	auto &schema = table.schema;
	auto index_entry = schema.CreateIndex(context, *info, table).get();
	if (!index_entry) {
		D_ASSERT(info->on_conflict == OnCreateConflict::IGNORE_ON_CONFLICT);
		// index already exists, but error ignored because of IF NOT EXISTS
		return SinkFinalizeType::READY;
	}
	auto &index = index_entry->Cast<DuckIndexEntry>();

	index.index = state.global_index.get();
	index.info = storage.info;
	for (auto &parsed_expr : info->parsed_expressions) {
		index.parsed_expressions.push_back(parsed_expr->Copy());
	}

	// vacuum excess memory
	state.global_index->Vacuum();

	// add index to storage
	storage.info->indexes.AddIndex(std::move(state.global_index));
	return SinkFinalizeType::READY;
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//

SourceResultType PhysicalCreateIndex::GetData(ExecutionContext &context, DataChunk &chunk,
                                              OperatorSourceInput &input) const {
	return SourceResultType::FINISHED;
}

} // namespace duckdb



namespace duckdb {

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
SourceResultType PhysicalCreateSchema::GetData(ExecutionContext &context, DataChunk &chunk,
                                               OperatorSourceInput &input) const {
	auto &catalog = Catalog::GetCatalog(context.client, info->catalog);
	if (catalog.IsSystemCatalog()) {
		throw BinderException("Cannot create schema in system catalog");
	}
	catalog.CreateSchema(context.client, *info);

	return SourceResultType::FINISHED;
}

} // namespace duckdb



namespace duckdb {

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
SourceResultType PhysicalCreateSequence::GetData(ExecutionContext &context, DataChunk &chunk,
                                                 OperatorSourceInput &input) const {
	auto &catalog = Catalog::GetCatalog(context.client, info->catalog);
	catalog.CreateSequence(context.client, *info);

	return SourceResultType::FINISHED;
}

} // namespace duckdb







namespace duckdb {

PhysicalCreateTable::PhysicalCreateTable(LogicalOperator &op, SchemaCatalogEntry &schema,
                                         unique_ptr<BoundCreateTableInfo> info, idx_t estimated_cardinality)
    : PhysicalOperator(PhysicalOperatorType::CREATE_TABLE, op.types, estimated_cardinality), schema(schema),
      info(std::move(info)) {
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
SourceResultType PhysicalCreateTable::GetData(ExecutionContext &context, DataChunk &chunk,
                                              OperatorSourceInput &input) const {
	auto &catalog = schema.catalog;
	catalog.CreateTable(catalog.GetCatalogTransaction(context.client), schema, *info);

	return SourceResultType::FINISHED;
}

} // namespace duckdb







namespace duckdb {

PhysicalCreateType::PhysicalCreateType(unique_ptr<CreateTypeInfo> info_p, idx_t estimated_cardinality)
    : PhysicalOperator(PhysicalOperatorType::CREATE_TYPE, {LogicalType::BIGINT}, estimated_cardinality),
      info(std::move(info_p)) {
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
class CreateTypeGlobalState : public GlobalSinkState {
public:
	explicit CreateTypeGlobalState(ClientContext &context) : result(LogicalType::VARCHAR) {
	}
	Vector result;
	idx_t size = 0;
	idx_t capacity = STANDARD_VECTOR_SIZE;
	string_set_t found_strings;
};

unique_ptr<GlobalSinkState> PhysicalCreateType::GetGlobalSinkState(ClientContext &context) const {
	return make_uniq<CreateTypeGlobalState>(context);
}

SinkResultType PhysicalCreateType::Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const {
	auto &gstate = input.global_state.Cast<CreateTypeGlobalState>();
	idx_t total_row_count = gstate.size + chunk.size();
	if (total_row_count > NumericLimits<uint32_t>::Maximum()) {
		throw InvalidInputException("Attempted to create ENUM of size %llu, which exceeds the maximum size of %llu",
		                            total_row_count, NumericLimits<uint32_t>::Maximum());
	}
	UnifiedVectorFormat sdata;
	chunk.data[0].ToUnifiedFormat(chunk.size(), sdata);

	if (total_row_count > gstate.capacity) {
		// We must resize our result vector
		gstate.result.Resize(gstate.capacity, gstate.capacity * 2);
		gstate.capacity *= 2;
	}

	auto src_ptr = (string_t *)sdata.data;
	auto result_ptr = FlatVector::GetData<string_t>(gstate.result);
	// Input vector has NULL value, we just throw an exception
	for (idx_t i = 0; i < chunk.size(); i++) {
		idx_t idx = sdata.sel->get_index(i);
		if (!sdata.validity.RowIsValid(idx)) {
			throw InvalidInputException("Attempted to create ENUM type with NULL value!");
		}
		auto str = src_ptr[idx];
		auto entry = gstate.found_strings.find(src_ptr[idx]);
		if (entry != gstate.found_strings.end()) {
			// entry was already found - skip
			continue;
		}
		auto owned_string = StringVector::AddStringOrBlob(gstate.result, str.GetData(), str.GetSize());
		gstate.found_strings.insert(owned_string);
		result_ptr[gstate.size++] = owned_string;
	}
	return SinkResultType::NEED_MORE_INPUT;
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
SourceResultType PhysicalCreateType::GetData(ExecutionContext &context, DataChunk &chunk,
                                             OperatorSourceInput &input) const {
	if (IsSink()) {
		D_ASSERT(info->type == LogicalType::INVALID);
		auto &g_sink_state = sink_state->Cast<CreateTypeGlobalState>();
		info->type = LogicalType::ENUM(info->name, g_sink_state.result, g_sink_state.size);
	}

	auto &catalog = Catalog::GetCatalog(context.client, info->catalog);
	auto catalog_entry = catalog.CreateType(context.client, *info);
	D_ASSERT(catalog_entry->type == CatalogType::TYPE_ENTRY);
	auto &catalog_type = catalog_entry->Cast<TypeCatalogEntry>();
	EnumType::SetCatalog(info->type, &catalog_type);

	return SourceResultType::FINISHED;
}

} // namespace duckdb



namespace duckdb {

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
SourceResultType PhysicalCreateView::GetData(ExecutionContext &context, DataChunk &chunk,
                                             OperatorSourceInput &input) const {
	auto &catalog = Catalog::GetCatalog(context.client, info->catalog);
	catalog.CreateView(context.client, *info);

	return SourceResultType::FINISHED;
}

} // namespace duckdb








namespace duckdb {

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
SourceResultType PhysicalDetach::GetData(ExecutionContext &context, DataChunk &chunk,
                                         OperatorSourceInput &input) const {
	auto &db_manager = DatabaseManager::Get(context.client);
	db_manager.DetachDatabase(context.client, info->name, info->if_not_found);

	return SourceResultType::FINISHED;
}

} // namespace duckdb








namespace duckdb {

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
SourceResultType PhysicalDrop::GetData(ExecutionContext &context, DataChunk &chunk, OperatorSourceInput &input) const {
	switch (info->type) {
	case CatalogType::PREPARED_STATEMENT: {
		// DEALLOCATE silently ignores errors
		auto &statements = ClientData::Get(context.client).prepared_statements;
		if (statements.find(info->name) != statements.end()) {
			statements.erase(info->name);
		}
		break;
	}
	case CatalogType::SCHEMA_ENTRY: {
		auto &catalog = Catalog::GetCatalog(context.client, info->catalog);
		catalog.DropEntry(context.client, *info);
		auto qualified_name = QualifiedName::Parse(info->name);

		// Check if the dropped schema was set as the current schema
		auto &client_data = ClientData::Get(context.client);
		auto &default_entry = client_data.catalog_search_path->GetDefault();
		auto &current_catalog = default_entry.catalog;
		auto &current_schema = default_entry.schema;
		D_ASSERT(info->name != DEFAULT_SCHEMA);

		if (info->catalog == current_catalog && current_schema == info->name) {
			// Reset the schema to default
			SchemaSetting::SetLocal(context.client, DEFAULT_SCHEMA);
		}
		break;
	}
	default: {
		auto &catalog = Catalog::GetCatalog(context.client, info->catalog);
		catalog.DropEntry(context.client, *info);
		break;
	}
	}

	return SourceResultType::FINISHED;
}

} // namespace duckdb












namespace duckdb {

PhysicalRecursiveCTE::PhysicalRecursiveCTE(vector<LogicalType> types, bool union_all, unique_ptr<PhysicalOperator> top,
                                           unique_ptr<PhysicalOperator> bottom, idx_t estimated_cardinality)
    : PhysicalOperator(PhysicalOperatorType::RECURSIVE_CTE, std::move(types), estimated_cardinality),
      union_all(union_all) {
	children.push_back(std::move(top));
	children.push_back(std::move(bottom));
}

PhysicalRecursiveCTE::~PhysicalRecursiveCTE() {
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
class RecursiveCTEState : public GlobalSinkState {
public:
	explicit RecursiveCTEState(ClientContext &context, const PhysicalRecursiveCTE &op)
	    : intermediate_table(context, op.GetTypes()), new_groups(STANDARD_VECTOR_SIZE) {
		ht = make_uniq<GroupedAggregateHashTable>(context, Allocator::Get(context), op.types, vector<LogicalType>(),
		                                          vector<BoundAggregateExpression *>());
	}

	unique_ptr<GroupedAggregateHashTable> ht;

	bool intermediate_empty = true;
	ColumnDataCollection intermediate_table;
	ColumnDataScanState scan_state;
	bool initialized = false;
	bool finished_scan = false;
	SelectionVector new_groups;
	AggregateHTAppendState append_state;
};

unique_ptr<GlobalSinkState> PhysicalRecursiveCTE::GetGlobalSinkState(ClientContext &context) const {
	return make_uniq<RecursiveCTEState>(context, *this);
}

idx_t PhysicalRecursiveCTE::ProbeHT(DataChunk &chunk, RecursiveCTEState &state) const {
	Vector dummy_addresses(LogicalType::POINTER);

	// Use the HT to eliminate duplicate rows
	idx_t new_group_count = state.ht->FindOrCreateGroups(state.append_state, chunk, dummy_addresses, state.new_groups);

	// we only return entries we have not seen before (i.e. new groups)
	chunk.Slice(state.new_groups, new_group_count);

	return new_group_count;
}

SinkResultType PhysicalRecursiveCTE::Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const {
	auto &gstate = input.global_state.Cast<RecursiveCTEState>();
	if (!union_all) {
		idx_t match_count = ProbeHT(chunk, gstate);
		if (match_count > 0) {
			gstate.intermediate_table.Append(chunk);
		}
	} else {
		gstate.intermediate_table.Append(chunk);
	}
	return SinkResultType::NEED_MORE_INPUT;
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
SourceResultType PhysicalRecursiveCTE::GetData(ExecutionContext &context, DataChunk &chunk,
                                               OperatorSourceInput &input) const {
	auto &gstate = sink_state->Cast<RecursiveCTEState>();
	if (!gstate.initialized) {
		gstate.intermediate_table.InitializeScan(gstate.scan_state);
		gstate.finished_scan = false;
		gstate.initialized = true;
	}
	while (chunk.size() == 0) {
		if (!gstate.finished_scan) {
			// scan any chunks we have collected so far
			gstate.intermediate_table.Scan(gstate.scan_state, chunk);
			if (chunk.size() == 0) {
				gstate.finished_scan = true;
			} else {
				break;
			}
		} else {
			// we have run out of chunks
			// now we need to recurse
			// we set up the working table as the data we gathered in this iteration of the recursion
			working_table->Reset();
			working_table->Combine(gstate.intermediate_table);
			// and we clear the intermediate table
			gstate.finished_scan = false;
			gstate.intermediate_table.Reset();
			// now we need to re-execute all of the pipelines that depend on the recursion
			ExecuteRecursivePipelines(context);

			// check if we obtained any results
			// if not, we are done
			if (gstate.intermediate_table.Count() == 0) {
				gstate.finished_scan = true;
				break;
			}
			// set up the scan again
			gstate.intermediate_table.InitializeScan(gstate.scan_state);
		}
	}

	return chunk.size() == 0 ? SourceResultType::FINISHED : SourceResultType::HAVE_MORE_OUTPUT;
}

void PhysicalRecursiveCTE::ExecuteRecursivePipelines(ExecutionContext &context) const {
	if (!recursive_meta_pipeline) {
		throw InternalException("Missing meta pipeline for recursive CTE");
	}
	D_ASSERT(recursive_meta_pipeline->HasRecursiveCTE());

	// get and reset pipelines
	vector<shared_ptr<Pipeline>> pipelines;
	recursive_meta_pipeline->GetPipelines(pipelines, true);
	for (auto &pipeline : pipelines) {
		auto sink = pipeline->GetSink();
		if (sink.get() != this) {
			sink->sink_state.reset();
		}
		for (auto &op_ref : pipeline->GetOperators()) {
			auto &op = op_ref.get();
			op.op_state.reset();
		}
		pipeline->ClearSource();
	}

	// get the MetaPipelines in the recursive_meta_pipeline and reschedule them
	vector<shared_ptr<MetaPipeline>> meta_pipelines;
	recursive_meta_pipeline->GetMetaPipelines(meta_pipelines, true, false);
	auto &executor = recursive_meta_pipeline->GetExecutor();
	vector<shared_ptr<Event>> events;
	executor.ReschedulePipelines(meta_pipelines, events);

	while (true) {
		executor.WorkOnTasks();
		if (executor.HasError()) {
			executor.ThrowException();
		}
		bool finished = true;
		for (auto &event : events) {
			if (!event->IsFinished()) {
				finished = false;
				break;
			}
		}
		if (finished) {
			// all pipelines finished: done!
			break;
		}
	}
}

//===--------------------------------------------------------------------===//
// Pipeline Construction
//===--------------------------------------------------------------------===//
void PhysicalRecursiveCTE::BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) {
	op_state.reset();
	sink_state.reset();
	recursive_meta_pipeline.reset();

	auto &state = meta_pipeline.GetState();
	state.SetPipelineSource(current, *this);

	auto &executor = meta_pipeline.GetExecutor();
	executor.AddRecursiveCTE(*this);

	// the LHS of the recursive CTE is our initial state
	auto &initial_state_pipeline = meta_pipeline.CreateChildMetaPipeline(current, *this);
	initial_state_pipeline.Build(*children[0]);

	// the RHS is the recursive pipeline
	recursive_meta_pipeline = make_shared<MetaPipeline>(executor, state, this);
	recursive_meta_pipeline->SetRecursiveCTE();
	recursive_meta_pipeline->Build(*children[1]);
}

vector<const_reference<PhysicalOperator>> PhysicalRecursiveCTE::GetSources() const {
	return {*this};
}

} // namespace duckdb






namespace duckdb {

PhysicalUnion::PhysicalUnion(vector<LogicalType> types, unique_ptr<PhysicalOperator> top,
                             unique_ptr<PhysicalOperator> bottom, idx_t estimated_cardinality)
    : PhysicalOperator(PhysicalOperatorType::UNION, std::move(types), estimated_cardinality) {
	children.push_back(std::move(top));
	children.push_back(std::move(bottom));
}

//===--------------------------------------------------------------------===//
// Pipeline Construction
//===--------------------------------------------------------------------===//
void PhysicalUnion::BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) {
	op_state.reset();
	sink_state.reset();

	// order matters if any of the downstream operators are order dependent,
	// or if the sink preserves order, but does not support batch indices to do so
	auto sink = meta_pipeline.GetSink();
	bool order_matters = false;
	if (current.IsOrderDependent()) {
		order_matters = true;
	}
	if (sink) {
		if (sink->SinkOrderDependent() || sink->RequiresBatchIndex()) {
			order_matters = true;
		}
		if (!sink->ParallelSink()) {
			order_matters = true;
		}
	}

	// create a union pipeline that is identical to 'current'
	auto union_pipeline = meta_pipeline.CreateUnionPipeline(current, order_matters);

	// continue with the current pipeline
	children[0]->BuildPipelines(current, meta_pipeline);

	if (order_matters) {
		// order matters, so 'union_pipeline' must come after all pipelines created by building out 'current'
		meta_pipeline.AddDependenciesFrom(union_pipeline, union_pipeline, false);
	}

	// build the union pipeline
	children[1]->BuildPipelines(*union_pipeline, meta_pipeline);

	// Assign proper batch index to the union pipeline
	// This needs to happen after the pipelines have been built because unions can be nested
	meta_pipeline.AssignNextBatchIndex(union_pipeline);
}

vector<const_reference<PhysicalOperator>> PhysicalUnion::GetSources() const {
	vector<const_reference<PhysicalOperator>> result;
	for (auto &child : children) {
		auto child_sources = child->GetSources();
		result.insert(result.end(), child_sources.begin(), child_sources.end());
	}
	return result;
}

} // namespace duckdb




namespace duckdb {

RadixPartitionInfo::RadixPartitionInfo(const idx_t n_partitions_upper_bound)
    : n_partitions(PreviousPowerOfTwo(n_partitions_upper_bound)),
      radix_bits(RadixPartitioning::RadixBits(n_partitions)), radix_mask(RadixPartitioning::Mask(radix_bits)),
      radix_shift(RadixPartitioning::Shift(radix_bits)) {

	D_ASSERT(n_partitions > 0);
	D_ASSERT(n_partitions <= 256);
	D_ASSERT(IsPowerOfTwo(n_partitions));
	D_ASSERT(radix_bits <= 8);
}

PartitionableHashTable::PartitionableHashTable(ClientContext &context, Allocator &allocator,
                                               RadixPartitionInfo &partition_info_p, vector<LogicalType> group_types_p,
                                               vector<LogicalType> payload_types_p,
                                               vector<BoundAggregateExpression *> bindings_p)
    : context(context), allocator(allocator), group_types(std::move(group_types_p)),
      payload_types(std::move(payload_types_p)), bindings(std::move(bindings_p)), is_partitioned(false),
      partition_info(partition_info_p), hashes(LogicalType::HASH), hashes_subset(LogicalType::HASH) {

	sel_vectors.resize(partition_info.n_partitions);
	sel_vector_sizes.resize(partition_info.n_partitions);
	group_subset.Initialize(allocator, group_types);
	if (!payload_types.empty()) {
		payload_subset.Initialize(allocator, payload_types);
	}

	for (hash_t r = 0; r < partition_info.n_partitions; r++) {
		sel_vectors[r].Initialize();
	}

	RowLayout layout;
	layout.Initialize(group_types, AggregateObject::CreateAggregateObjects(bindings));
	tuple_size = layout.GetRowWidth();
}

HtEntryType PartitionableHashTable::GetHTEntrySize() {
	// we need at least STANDARD_VECTOR_SIZE entries to fit in the hash table
	if (GroupedAggregateHashTable::GetMaxCapacity(HtEntryType::HT_WIDTH_32, tuple_size) < STANDARD_VECTOR_SIZE) {
		return HtEntryType::HT_WIDTH_64;
	}
	return HtEntryType::HT_WIDTH_32;
}

idx_t PartitionableHashTable::ListAddChunk(HashTableList &list, DataChunk &groups, Vector &group_hashes,
                                           DataChunk &payload, const unsafe_vector<idx_t> &filter) {
	// If this is false, a single AddChunk would overflow the max capacity
	D_ASSERT(list.empty() || groups.size() <= list.back()->MaxCapacity());
	if (list.empty() || list.back()->Count() + groups.size() >= list.back()->MaxCapacity()) {
		idx_t new_capacity = GroupedAggregateHashTable::InitialCapacity();
		if (!list.empty()) {
			new_capacity = list.back()->Capacity();
			// early release first part of ht and prevent adding of more data
			list.back()->Finalize();
		}
		list.push_back(make_uniq<GroupedAggregateHashTable>(context, allocator, group_types, payload_types, bindings,
		                                                    GetHTEntrySize(), new_capacity));
	}
	return list.back()->AddChunk(append_state, groups, group_hashes, payload, filter);
}

idx_t PartitionableHashTable::AddChunk(DataChunk &groups, DataChunk &payload, bool do_partition,
                                       const unsafe_vector<idx_t> &filter) {
	groups.Hash(hashes);

	// we partition when we are asked to or when the unpartitioned ht runs out of space
	if (!IsPartitioned() && do_partition) {
		Partition();
	}

	if (!IsPartitioned()) {
		return ListAddChunk(unpartitioned_hts, groups, hashes, payload, filter);
	}

	// makes no sense to do this with 1 partition
	D_ASSERT(partition_info.n_partitions > 0);

	for (hash_t r = 0; r < partition_info.n_partitions; r++) {
		sel_vector_sizes[r] = 0;
	}

	hashes.Flatten(groups.size());
	auto hashes_ptr = FlatVector::GetData<hash_t>(hashes);

	// Determine for every partition how much data will be sinked into it
	for (idx_t i = 0; i < groups.size(); i++) {
		auto partition = partition_info.GetHashPartition(hashes_ptr[i]);
		D_ASSERT(partition < partition_info.n_partitions);
		sel_vectors[partition].set_index(sel_vector_sizes[partition]++, i);
	}

#ifdef DEBUG
	// make sure we have lost no rows
	idx_t total_count = 0;
	for (idx_t r = 0; r < partition_info.n_partitions; r++) {
		total_count += sel_vector_sizes[r];
	}
	D_ASSERT(total_count == groups.size());
#endif
	idx_t group_count = 0;
	for (hash_t r = 0; r < partition_info.n_partitions; r++) {
		group_subset.Slice(groups, sel_vectors[r], sel_vector_sizes[r]);
		if (!payload_types.empty()) {
			payload_subset.Slice(payload, sel_vectors[r], sel_vector_sizes[r]);
		} else {
			payload_subset.SetCardinality(sel_vector_sizes[r]);
		}
		hashes_subset.Slice(hashes, sel_vectors[r], sel_vector_sizes[r]);

		group_count += ListAddChunk(radix_partitioned_hts[r], group_subset, hashes_subset, payload_subset, filter);
	}
	return group_count;
}

void PartitionableHashTable::Partition() {
	D_ASSERT(!IsPartitioned());
	D_ASSERT(radix_partitioned_hts.empty());
	D_ASSERT(partition_info.n_partitions > 1);

	vector<GroupedAggregateHashTable *> partition_hts(partition_info.n_partitions);
	radix_partitioned_hts.resize(partition_info.n_partitions);
	for (auto &unpartitioned_ht : unpartitioned_hts) {
		for (idx_t r = 0; r < partition_info.n_partitions; r++) {
			radix_partitioned_hts[r].push_back(make_uniq<GroupedAggregateHashTable>(
			    context, allocator, group_types, payload_types, bindings, GetHTEntrySize()));
			partition_hts[r] = radix_partitioned_hts[r].back().get();
		}
		unpartitioned_ht->Partition(partition_hts, partition_info.radix_bits);
		unpartitioned_ht.reset();
	}
	unpartitioned_hts.clear();
	is_partitioned = true;
}

bool PartitionableHashTable::IsPartitioned() {
	return is_partitioned;
}

HashTableList PartitionableHashTable::GetPartition(idx_t partition) {
	D_ASSERT(IsPartitioned());
	D_ASSERT(partition < partition_info.n_partitions);
	D_ASSERT(radix_partitioned_hts.size() > partition);
	return std::move(radix_partitioned_hts[partition]);
}

HashTableList PartitionableHashTable::GetUnpartitioned() {
	D_ASSERT(!IsPartitioned());
	return std::move(unpartitioned_hts);
}

void PartitionableHashTable::Finalize() {
	if (IsPartitioned()) {
		for (auto &ht_list : radix_partitioned_hts) {
			for (auto &ht : ht_list) {
				D_ASSERT(ht);
				ht->Finalize();
			}
		}
	} else {
		for (auto &ht : unpartitioned_hts) {
			D_ASSERT(ht);
			ht->Finalize();
		}
	}
}

} // namespace duckdb




namespace duckdb {

PerfectAggregateHashTable::PerfectAggregateHashTable(ClientContext &context, Allocator &allocator,
                                                     const vector<LogicalType> &group_types_p,
                                                     vector<LogicalType> payload_types_p,
                                                     vector<AggregateObject> aggregate_objects_p,
                                                     vector<Value> group_minima_p, vector<idx_t> required_bits_p)
    : BaseAggregateHashTable(context, allocator, aggregate_objects_p, std::move(payload_types_p)),
      addresses(LogicalType::POINTER), required_bits(std::move(required_bits_p)), total_required_bits(0),
      group_minima(std::move(group_minima_p)), sel(STANDARD_VECTOR_SIZE), aggregate_allocator(allocator) {
	for (auto &group_bits : required_bits) {
		total_required_bits += group_bits;
	}
	// the total amount of groups we allocate space for is 2^required_bits
	total_groups = (uint64_t)1 << total_required_bits;
	// we don't need to store the groups in a perfect hash table, since the group keys can be deduced by their location
	grouping_columns = group_types_p.size();
	layout.Initialize(std::move(aggregate_objects_p));
	tuple_size = layout.GetRowWidth();

	// allocate and null initialize the data
	owned_data = make_unsafe_uniq_array<data_t>(tuple_size * total_groups);
	data = owned_data.get();

	// set up the empty payloads for every tuple, and initialize the "occupied" flag to false
	group_is_set = make_unsafe_uniq_array<bool>(total_groups);
	memset(group_is_set.get(), 0, total_groups * sizeof(bool));

	// initialize the hash table for each entry
	auto address_data = FlatVector::GetData<uintptr_t>(addresses);
	idx_t init_count = 0;
	for (idx_t i = 0; i < total_groups; i++) {
		address_data[init_count] = uintptr_t(data) + (tuple_size * i);
		init_count++;
		if (init_count == STANDARD_VECTOR_SIZE) {
			RowOperations::InitializeStates(layout, addresses, *FlatVector::IncrementalSelectionVector(), init_count);
			init_count = 0;
		}
	}
	RowOperations::InitializeStates(layout, addresses, *FlatVector::IncrementalSelectionVector(), init_count);
}

PerfectAggregateHashTable::~PerfectAggregateHashTable() {
	Destroy();
}

template <class T>
static void ComputeGroupLocationTemplated(UnifiedVectorFormat &group_data, Value &min, uintptr_t *address_data,
                                          idx_t current_shift, idx_t count) {
	auto data = (T *)group_data.data;
	auto min_val = min.GetValueUnsafe<T>();
	if (!group_data.validity.AllValid()) {
		for (idx_t i = 0; i < count; i++) {
			auto index = group_data.sel->get_index(i);
			// check if the value is NULL
			// NULL groups are considered as "0" in the hash table
			// that is to say, they have no effect on the position of the element (because 0 << shift is 0)
			// we only need to handle non-null values here
			if (group_data.validity.RowIsValid(index)) {
				D_ASSERT(data[index] >= min_val);
				uintptr_t adjusted_value = (data[index] - min_val) + 1;
				address_data[i] += adjusted_value << current_shift;
			}
		}
	} else {
		// no null values: we can directly compute the addresses
		for (idx_t i = 0; i < count; i++) {
			auto index = group_data.sel->get_index(i);
			uintptr_t adjusted_value = (data[index] - min_val) + 1;
			address_data[i] += adjusted_value << current_shift;
		}
	}
}

static void ComputeGroupLocation(Vector &group, Value &min, uintptr_t *address_data, idx_t current_shift, idx_t count) {
	UnifiedVectorFormat vdata;
	group.ToUnifiedFormat(count, vdata);

	switch (group.GetType().InternalType()) {
	case PhysicalType::INT8:
		ComputeGroupLocationTemplated<int8_t>(vdata, min, address_data, current_shift, count);
		break;
	case PhysicalType::INT16:
		ComputeGroupLocationTemplated<int16_t>(vdata, min, address_data, current_shift, count);
		break;
	case PhysicalType::INT32:
		ComputeGroupLocationTemplated<int32_t>(vdata, min, address_data, current_shift, count);
		break;
	case PhysicalType::INT64:
		ComputeGroupLocationTemplated<int64_t>(vdata, min, address_data, current_shift, count);
		break;
	default:
		throw InternalException("Unsupported group type for perfect aggregate hash table");
	}
}

void PerfectAggregateHashTable::AddChunk(DataChunk &groups, DataChunk &payload) {
	// first we need to find the location in the HT of each of the groups
	auto address_data = FlatVector::GetData<uintptr_t>(addresses);
	// zero-initialize the address data
	memset(address_data, 0, groups.size() * sizeof(uintptr_t));
	D_ASSERT(groups.ColumnCount() == group_minima.size());

	// then compute the actual group location by iterating over each of the groups
	idx_t current_shift = total_required_bits;
	for (idx_t i = 0; i < groups.ColumnCount(); i++) {
		current_shift -= required_bits[i];
		ComputeGroupLocation(groups.data[i], group_minima[i], address_data, current_shift, groups.size());
	}
	// now we have the HT entry number for every tuple
	// compute the actual pointer to the data by adding it to the base HT pointer and multiplying by the tuple size
	for (idx_t i = 0; i < groups.size(); i++) {
		const auto group = address_data[i];
		D_ASSERT(group < total_groups);
		group_is_set[group] = true;
		address_data[i] = uintptr_t(data) + group * tuple_size;
	}

	// after finding the group location we update the aggregates
	idx_t payload_idx = 0;
	auto &aggregates = layout.GetAggregates();
	RowOperationsState row_state(aggregate_allocator.GetAllocator());
	for (idx_t aggr_idx = 0; aggr_idx < aggregates.size(); aggr_idx++) {
		auto &aggregate = aggregates[aggr_idx];
		auto input_count = (idx_t)aggregate.child_count;
		if (aggregate.filter) {
			RowOperations::UpdateFilteredStates(row_state, filter_set.GetFilterData(aggr_idx), aggregate, addresses,
			                                    payload, payload_idx);
		} else {
			RowOperations::UpdateStates(row_state, aggregate, addresses, payload, payload_idx, payload.size());
		}
		// move to the next aggregate
		payload_idx += input_count;
		VectorOperations::AddInPlace(addresses, aggregate.payload_size, payload.size());
	}
}

void PerfectAggregateHashTable::Combine(PerfectAggregateHashTable &other) {
	D_ASSERT(total_groups == other.total_groups);
	D_ASSERT(tuple_size == other.tuple_size);

	Vector source_addresses(LogicalType::POINTER);
	Vector target_addresses(LogicalType::POINTER);
	auto source_addresses_ptr = FlatVector::GetData<data_ptr_t>(source_addresses);
	auto target_addresses_ptr = FlatVector::GetData<data_ptr_t>(target_addresses);

	// iterate over all entries of both hash tables and call combine for all entries that can be combined
	data_ptr_t source_ptr = other.data;
	data_ptr_t target_ptr = data;
	idx_t combine_count = 0;
	RowOperationsState row_state(aggregate_allocator.GetAllocator());
	for (idx_t i = 0; i < total_groups; i++) {
		auto has_entry_source = other.group_is_set[i];
		// we only have any work to do if the source has an entry for this group
		if (has_entry_source) {
			group_is_set[i] = true;
			source_addresses_ptr[combine_count] = source_ptr;
			target_addresses_ptr[combine_count] = target_ptr;
			combine_count++;
			if (combine_count == STANDARD_VECTOR_SIZE) {
				RowOperations::CombineStates(row_state, layout, source_addresses, target_addresses, combine_count);
				combine_count = 0;
			}
		}
		source_ptr += tuple_size;
		target_ptr += tuple_size;
	}
	RowOperations::CombineStates(row_state, layout, source_addresses, target_addresses, combine_count);
}

template <class T>
static void ReconstructGroupVectorTemplated(uint32_t group_values[], Value &min, idx_t mask, idx_t shift,
                                            idx_t entry_count, Vector &result) {
	auto data = FlatVector::GetData<T>(result);
	auto &validity_mask = FlatVector::Validity(result);
	auto min_data = min.GetValueUnsafe<T>();
	for (idx_t i = 0; i < entry_count; i++) {
		// extract the value of this group from the total group index
		auto group_index = (group_values[i] >> shift) & mask;
		if (group_index == 0) {
			// if it is 0, the value is NULL
			validity_mask.SetInvalid(i);
		} else {
			// otherwise we add the value (minus 1) to the min value
			data[i] = min_data + group_index - 1;
		}
	}
}

static void ReconstructGroupVector(uint32_t group_values[], Value &min, idx_t required_bits, idx_t shift,
                                   idx_t entry_count, Vector &result) {
	// construct the mask for this entry
	idx_t mask = ((uint64_t)1 << required_bits) - 1;
	switch (result.GetType().InternalType()) {
	case PhysicalType::INT8:
		ReconstructGroupVectorTemplated<int8_t>(group_values, min, mask, shift, entry_count, result);
		break;
	case PhysicalType::INT16:
		ReconstructGroupVectorTemplated<int16_t>(group_values, min, mask, shift, entry_count, result);
		break;
	case PhysicalType::INT32:
		ReconstructGroupVectorTemplated<int32_t>(group_values, min, mask, shift, entry_count, result);
		break;
	case PhysicalType::INT64:
		ReconstructGroupVectorTemplated<int64_t>(group_values, min, mask, shift, entry_count, result);
		break;
	default:
		throw InternalException("Invalid type for perfect aggregate HT group");
	}
}

void PerfectAggregateHashTable::Scan(idx_t &scan_position, DataChunk &result) {
	auto data_pointers = FlatVector::GetData<data_ptr_t>(addresses);
	uint32_t group_values[STANDARD_VECTOR_SIZE];

	// iterate over the HT until we either have exhausted the entire HT, or
	idx_t entry_count = 0;
	for (; scan_position < total_groups; scan_position++) {
		if (group_is_set[scan_position]) {
			// this group is set: add it to the set of groups to extract
			data_pointers[entry_count] = data + tuple_size * scan_position;
			group_values[entry_count] = scan_position;
			entry_count++;
			if (entry_count == STANDARD_VECTOR_SIZE) {
				scan_position++;
				break;
			}
		}
	}
	if (entry_count == 0) {
		// no entries found
		return;
	}
	// first reconstruct the groups from the group index
	idx_t shift = total_required_bits;
	for (idx_t i = 0; i < grouping_columns; i++) {
		shift -= required_bits[i];
		ReconstructGroupVector(group_values, group_minima[i], required_bits[i], shift, entry_count, result.data[i]);
	}
	// then construct the payloads
	result.SetCardinality(entry_count);
	RowOperationsState row_state(aggregate_allocator.GetAllocator());
	RowOperations::FinalizeStates(row_state, layout, addresses, result, grouping_columns);
}

void PerfectAggregateHashTable::Destroy() {
	// check if there is any destructor to call
	bool has_destructor = false;
	for (auto &aggr : layout.GetAggregates()) {
		if (aggr.function.destructor) {
			has_destructor = true;
		}
	}
	if (!has_destructor) {
		return;
	}
	// there are aggregates with destructors: loop over the hash table
	// and call the destructor method for each of the aggregates
	auto data_pointers = FlatVector::GetData<data_ptr_t>(addresses);
	idx_t count = 0;

	// iterate over all initialised slots of the hash table
	RowOperationsState row_state(aggregate_allocator.GetAllocator());
	data_ptr_t payload_ptr = data;
	for (idx_t i = 0; i < total_groups; i++) {
		if (group_is_set[i]) {
			data_pointers[count++] = payload_ptr;
			if (count == STANDARD_VECTOR_SIZE) {
				RowOperations::DestroyStates(row_state, layout, addresses, count);
				count = 0;
			}
		}
		payload_ptr += tuple_size;
	}
	RowOperations::DestroyStates(row_state, layout, addresses, count);
}

} // namespace duckdb













namespace duckdb {

string PhysicalOperator::GetName() const {
	return PhysicalOperatorToString(type);
}

string PhysicalOperator::ToString() const {
	TreeRenderer renderer;
	return renderer.ToString(*this);
}

// LCOV_EXCL_START
void PhysicalOperator::Print() const {
	Printer::Print(ToString());
}
// LCOV_EXCL_STOP

vector<const_reference<PhysicalOperator>> PhysicalOperator::GetChildren() const {
	vector<const_reference<PhysicalOperator>> result;
	for (auto &child : children) {
		result.push_back(*child);
	}
	return result;
}

//===--------------------------------------------------------------------===//
// Operator
//===--------------------------------------------------------------------===//
// LCOV_EXCL_START
unique_ptr<OperatorState> PhysicalOperator::GetOperatorState(ExecutionContext &context) const {
	return make_uniq<OperatorState>();
}

unique_ptr<GlobalOperatorState> PhysicalOperator::GetGlobalOperatorState(ClientContext &context) const {
	return make_uniq<GlobalOperatorState>();
}

OperatorResultType PhysicalOperator::Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                             GlobalOperatorState &gstate, OperatorState &state) const {
	throw InternalException("Calling Execute on a node that is not an operator!");
}

OperatorFinalizeResultType PhysicalOperator::FinalExecute(ExecutionContext &context, DataChunk &chunk,
                                                          GlobalOperatorState &gstate, OperatorState &state) const {
	throw InternalException("Calling FinalExecute on a node that is not an operator!");
}
// LCOV_EXCL_STOP

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
unique_ptr<LocalSourceState> PhysicalOperator::GetLocalSourceState(ExecutionContext &context,
                                                                   GlobalSourceState &gstate) const {
	return make_uniq<LocalSourceState>();
}

unique_ptr<GlobalSourceState> PhysicalOperator::GetGlobalSourceState(ClientContext &context) const {
	return make_uniq<GlobalSourceState>();
}

// LCOV_EXCL_START
SourceResultType PhysicalOperator::GetData(ExecutionContext &context, DataChunk &chunk,
                                           OperatorSourceInput &input) const {
	throw InternalException("Calling GetData on a node that is not a source!");
}

idx_t PhysicalOperator::GetBatchIndex(ExecutionContext &context, DataChunk &chunk, GlobalSourceState &gstate,
                                      LocalSourceState &lstate) const {
	throw InternalException("Calling GetBatchIndex on a node that does not support it");
}

double PhysicalOperator::GetProgress(ClientContext &context, GlobalSourceState &gstate) const {
	return -1;
}
// LCOV_EXCL_STOP

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
// LCOV_EXCL_START
SinkResultType PhysicalOperator::Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const {
	throw InternalException("Calling Sink on a node that is not a sink!");
}

// LCOV_EXCL_STOP

void PhysicalOperator::Combine(ExecutionContext &context, GlobalSinkState &gstate, LocalSinkState &lstate) const {
}

SinkFinalizeType PhysicalOperator::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                            GlobalSinkState &gstate) const {
	return SinkFinalizeType::READY;
}

void PhysicalOperator::NextBatch(ExecutionContext &context, GlobalSinkState &state, LocalSinkState &lstate_p) const {
}

unique_ptr<LocalSinkState> PhysicalOperator::GetLocalSinkState(ExecutionContext &context) const {
	return make_uniq<LocalSinkState>();
}

unique_ptr<GlobalSinkState> PhysicalOperator::GetGlobalSinkState(ClientContext &context) const {
	return make_uniq<GlobalSinkState>();
}

idx_t PhysicalOperator::GetMaxThreadMemory(ClientContext &context) {
	// Memory usage per thread should scale with max mem / num threads
	// We take 1/4th of this, to be conservative
	idx_t max_memory = BufferManager::GetBufferManager(context).GetMaxMemory();
	idx_t num_threads = TaskScheduler::GetScheduler(context).NumberOfThreads();
	return (max_memory / num_threads) / 4;
}

//===--------------------------------------------------------------------===//
// Pipeline Construction
//===--------------------------------------------------------------------===//
void PhysicalOperator::BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) {
	op_state.reset();

	auto &state = meta_pipeline.GetState();
	if (IsSink()) {
		// operator is a sink, build a pipeline
		sink_state.reset();
		D_ASSERT(children.size() == 1);

		// single operator: the operator becomes the data source of the current pipeline
		state.SetPipelineSource(current, *this);

		// we create a new pipeline starting from the child
		auto &child_meta_pipeline = meta_pipeline.CreateChildMetaPipeline(current, *this);
		child_meta_pipeline.Build(*children[0]);
	} else {
		// operator is not a sink! recurse in children
		if (children.empty()) {
			// source
			state.SetPipelineSource(current, *this);
		} else {
			if (children.size() != 1) {
				throw InternalException("Operator not supported in BuildPipelines");
			}
			state.AddPipelineOperator(current, *this);
			children[0]->BuildPipelines(current, meta_pipeline);
		}
	}
}

vector<const_reference<PhysicalOperator>> PhysicalOperator::GetSources() const {
	vector<const_reference<PhysicalOperator>> result;
	if (IsSink()) {
		D_ASSERT(children.size() == 1);
		result.push_back(*this);
		return result;
	} else {
		if (children.empty()) {
			// source
			result.push_back(*this);
			return result;
		} else {
			if (children.size() != 1) {
				throw InternalException("Operator not supported in GetSource");
			}
			return children[0]->GetSources();
		}
	}
}

bool PhysicalOperator::AllSourcesSupportBatchIndex() const {
	auto sources = GetSources();
	for (auto &source : sources) {
		if (!source.get().SupportsBatchIndex()) {
			return false;
		}
	}
	return true;
}

void PhysicalOperator::Verify() {
#ifdef DEBUG
	auto sources = GetSources();
	D_ASSERT(!sources.empty());
	for (auto &child : children) {
		child->Verify();
	}
#endif
}

bool CachingPhysicalOperator::CanCacheType(const LogicalType &type) {
	switch (type.id()) {
	case LogicalTypeId::LIST:
	case LogicalTypeId::MAP:
		return false;
	case LogicalTypeId::STRUCT: {
		auto &entries = StructType::GetChildTypes(type);
		for (auto &entry : entries) {
			if (!CanCacheType(entry.second)) {
				return false;
			}
		}
		return true;
	}
	default:
		return true;
	}
}

CachingPhysicalOperator::CachingPhysicalOperator(PhysicalOperatorType type, vector<LogicalType> types_p,
                                                 idx_t estimated_cardinality)
    : PhysicalOperator(type, std::move(types_p), estimated_cardinality) {

	caching_supported = true;
	for (auto &col_type : types) {
		if (!CanCacheType(col_type)) {
			caching_supported = false;
			break;
		}
	}
}

OperatorResultType CachingPhysicalOperator::Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                                    GlobalOperatorState &gstate, OperatorState &state_p) const {
	auto &state = state_p.Cast<CachingOperatorState>();

	// Execute child operator
	auto child_result = ExecuteInternal(context, input, chunk, gstate, state);

#if STANDARD_VECTOR_SIZE >= 128
	if (!state.initialized) {
		state.initialized = true;
		state.can_cache_chunk = true;

		if (!context.client.config.enable_caching_operators) {
			state.can_cache_chunk = false;
		} else if (!context.pipeline || !caching_supported) {
			state.can_cache_chunk = false;
		} else if (!context.pipeline->GetSink()) {
			// Disabling for pipelines without Sink, i.e. when pulling
			state.can_cache_chunk = false;
		} else if (context.pipeline->GetSink()->RequiresBatchIndex()) {
			state.can_cache_chunk = false;
		} else if (context.pipeline->IsOrderDependent()) {
			state.can_cache_chunk = false;
		}
	}
	if (!state.can_cache_chunk) {
		return child_result;
	}
	// TODO chunk size of 0 should not result in a cache being created!
	if (chunk.size() < CACHE_THRESHOLD) {
		// we have filtered out a significant amount of tuples
		// add this chunk to the cache and continue

		if (!state.cached_chunk) {
			state.cached_chunk = make_uniq<DataChunk>();
			state.cached_chunk->Initialize(Allocator::Get(context.client), chunk.GetTypes());
		}

		state.cached_chunk->Append(chunk);

		if (state.cached_chunk->size() >= (STANDARD_VECTOR_SIZE - CACHE_THRESHOLD) ||
		    child_result == OperatorResultType::FINISHED) {
			// chunk cache full: return it
			chunk.Move(*state.cached_chunk);
			state.cached_chunk->Initialize(Allocator::Get(context.client), chunk.GetTypes());
			return child_result;
		} else {
			// chunk cache not full return empty result
			chunk.Reset();
		}
	}
#endif

	return child_result;
}

OperatorFinalizeResultType CachingPhysicalOperator::FinalExecute(ExecutionContext &context, DataChunk &chunk,
                                                                 GlobalOperatorState &gstate,
                                                                 OperatorState &state_p) const {
	auto &state = state_p.Cast<CachingOperatorState>();
	if (state.cached_chunk) {
		chunk.Move(*state.cached_chunk);
		state.cached_chunk.reset();
	} else {
		chunk.SetCardinality(0);
	}
	return OperatorFinalizeResultType::FINISHED;
}

} // namespace duckdb














namespace duckdb {

static uint32_t RequiredBitsForValue(uint32_t n) {
	idx_t required_bits = 0;
	while (n > 0) {
		n >>= 1;
		required_bits++;
	}
	return required_bits;
}

static bool CanUsePerfectHashAggregate(ClientContext &context, LogicalAggregate &op, vector<idx_t> &bits_per_group) {
	if (op.grouping_sets.size() > 1 || !op.grouping_functions.empty()) {
		return false;
	}
	idx_t perfect_hash_bits = 0;
	if (op.group_stats.empty()) {
		op.group_stats.resize(op.groups.size());
	}
	for (idx_t group_idx = 0; group_idx < op.groups.size(); group_idx++) {
		auto &group = op.groups[group_idx];
		auto &stats = op.group_stats[group_idx];

		switch (group->return_type.InternalType()) {
		case PhysicalType::INT8:
		case PhysicalType::INT16:
		case PhysicalType::INT32:
		case PhysicalType::INT64:
			break;
		default:
			// we only support simple integer types for perfect hashing
			return false;
		}
		// check if the group has stats available
		auto &group_type = group->return_type;
		if (!stats) {
			// no stats, but we might still be able to use perfect hashing if the type is small enough
			// for small types we can just set the stats to [type_min, type_max]
			switch (group_type.InternalType()) {
			case PhysicalType::INT8:
			case PhysicalType::INT16:
				break;
			default:
				// type is too large and there are no stats: skip perfect hashing
				return false;
			}
			// construct stats with the min and max value of the type
			stats = NumericStats::CreateUnknown(group_type).ToUnique();
			NumericStats::SetMin(*stats, Value::MinimumValue(group_type));
			NumericStats::SetMax(*stats, Value::MaximumValue(group_type));
		}
		auto &nstats = *stats;

		if (!NumericStats::HasMinMax(nstats)) {
			return false;
		}
		// we have a min and a max value for the stats: use that to figure out how many bits we have
		// we add two here, one for the NULL value, and one to make the computation one-indexed
		// (e.g. if min and max are the same, we still need one entry in total)
		int64_t range;
		switch (group_type.InternalType()) {
		case PhysicalType::INT8:
			range = int64_t(NumericStats::GetMax<int8_t>(nstats)) - int64_t(NumericStats::GetMin<int8_t>(nstats));
			break;
		case PhysicalType::INT16:
			range = int64_t(NumericStats::GetMax<int16_t>(nstats)) - int64_t(NumericStats::GetMin<int16_t>(nstats));
			break;
		case PhysicalType::INT32:
			range = int64_t(NumericStats::GetMax<int32_t>(nstats)) - int64_t(NumericStats::GetMin<int32_t>(nstats));
			break;
		case PhysicalType::INT64:
			if (!TrySubtractOperator::Operation(NumericStats::GetMax<int64_t>(nstats),
			                                    NumericStats::GetMin<int64_t>(nstats), range)) {
				return false;
			}
			break;
		default:
			throw InternalException("Unsupported type for perfect hash (should be caught before)");
		}
		// bail out on any range bigger than 2^32
		if (range >= NumericLimits<int32_t>::Maximum()) {
			return false;
		}
		range += 2;
		// figure out how many bits we need
		idx_t required_bits = RequiredBitsForValue(range);
		bits_per_group.push_back(required_bits);
		perfect_hash_bits += required_bits;
		// check if we have exceeded the bits for the hash
		if (perfect_hash_bits > ClientConfig::GetConfig(context).perfect_ht_threshold) {
			// too many bits for perfect hash
			return false;
		}
	}
	for (auto &expression : op.expressions) {
		auto &aggregate = expression->Cast<BoundAggregateExpression>();
		if (aggregate.IsDistinct() || !aggregate.function.combine) {
			// distinct aggregates are not supported in perfect hash aggregates
			return false;
		}
	}
	return true;
}

unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(LogicalAggregate &op) {
	unique_ptr<PhysicalOperator> groupby;
	D_ASSERT(op.children.size() == 1);

	auto plan = CreatePlan(*op.children[0]);

	plan = ExtractAggregateExpressions(std::move(plan), op.expressions, op.groups);

	if (op.groups.empty() && op.grouping_sets.size() <= 1) {
		// no groups, check if we can use a simple aggregation
		// special case: aggregate entire columns together
		bool use_simple_aggregation = true;
		for (auto &expression : op.expressions) {
			auto &aggregate = expression->Cast<BoundAggregateExpression>();
			if (!aggregate.function.simple_update) {
				// unsupported aggregate for simple aggregation: use hash aggregation
				use_simple_aggregation = false;
				break;
			}
		}
		if (use_simple_aggregation) {
			groupby = make_uniq_base<PhysicalOperator, PhysicalUngroupedAggregate>(op.types, std::move(op.expressions),
			                                                                       op.estimated_cardinality);
		} else {
			groupby = make_uniq_base<PhysicalOperator, PhysicalHashAggregate>(
			    context, op.types, std::move(op.expressions), op.estimated_cardinality);
		}
	} else {
		// groups! create a GROUP BY aggregator
		// use a perfect hash aggregate if possible
		vector<idx_t> required_bits;
		if (CanUsePerfectHashAggregate(context, op, required_bits)) {
			groupby = make_uniq_base<PhysicalOperator, PhysicalPerfectHashAggregate>(
			    context, op.types, std::move(op.expressions), std::move(op.groups), std::move(op.group_stats),
			    std::move(required_bits), op.estimated_cardinality);
		} else {
			groupby = make_uniq_base<PhysicalOperator, PhysicalHashAggregate>(
			    context, op.types, std::move(op.expressions), std::move(op.groups), std::move(op.grouping_sets),
			    std::move(op.grouping_functions), op.estimated_cardinality);
		}
	}
	groupby->children.push_back(std::move(plan));
	return groupby;
}

unique_ptr<PhysicalOperator>
PhysicalPlanGenerator::ExtractAggregateExpressions(unique_ptr<PhysicalOperator> child,
                                                   vector<unique_ptr<Expression>> &aggregates,
                                                   vector<unique_ptr<Expression>> &groups) {
	vector<unique_ptr<Expression>> expressions;
	vector<LogicalType> types;

	// bind sorted aggregates
	for (auto &aggr : aggregates) {
		auto &bound_aggr = aggr->Cast<BoundAggregateExpression>();
		if (bound_aggr.order_bys) {
			// sorted aggregate!
			FunctionBinder::BindSortedAggregate(context, bound_aggr, groups);
		}
	}
	for (auto &group : groups) {
		auto ref = make_uniq<BoundReferenceExpression>(group->return_type, expressions.size());
		types.push_back(group->return_type);
		expressions.push_back(std::move(group));
		group = std::move(ref);
	}
	for (auto &aggr : aggregates) {
		auto &bound_aggr = aggr->Cast<BoundAggregateExpression>();
		for (auto &child : bound_aggr.children) {
			auto ref = make_uniq<BoundReferenceExpression>(child->return_type, expressions.size());
			types.push_back(child->return_type);
			expressions.push_back(std::move(child));
			child = std::move(ref);
		}
		if (bound_aggr.filter) {
			auto &filter = bound_aggr.filter;
			auto ref = make_uniq<BoundReferenceExpression>(filter->return_type, expressions.size());
			types.push_back(filter->return_type);
			expressions.push_back(std::move(filter));
			bound_aggr.filter = std::move(ref);
		}
	}
	if (expressions.empty()) {
		return child;
	}
	auto projection =
	    make_uniq<PhysicalProjection>(std::move(types), std::move(expressions), child->estimated_cardinality);
	projection->children.push_back(std::move(child));
	return std::move(projection);
}

} // namespace duckdb




namespace duckdb {

unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(LogicalAnyJoin &op) {
	// first visit the child nodes
	D_ASSERT(op.children.size() == 2);
	D_ASSERT(op.condition);

	auto left = CreatePlan(*op.children[0]);
	auto right = CreatePlan(*op.children[1]);

	// create the blockwise NL join
	return make_uniq<PhysicalBlockwiseNLJoin>(op, std::move(left), std::move(right), std::move(op.condition),
	                                          op.join_type, op.estimated_cardinality);
}

} // namespace duckdb











namespace duckdb {

unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(LogicalAsOfJoin &op) {
	// now visit the children
	D_ASSERT(op.children.size() == 2);
	idx_t lhs_cardinality = op.children[0]->EstimateCardinality(context);
	idx_t rhs_cardinality = op.children[1]->EstimateCardinality(context);
	auto left = CreatePlan(*op.children[0]);
	auto right = CreatePlan(*op.children[1]);
	D_ASSERT(left && right);

	//	Validate
	vector<idx_t> equi_indexes;
	auto asof_idx = op.conditions.size();
	for (size_t c = 0; c < op.conditions.size(); ++c) {
		auto &cond = op.conditions[c];
		switch (cond.comparison) {
		case ExpressionType::COMPARE_EQUAL:
		case ExpressionType::COMPARE_NOT_DISTINCT_FROM:
			equi_indexes.emplace_back(c);
			break;
		case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
			D_ASSERT(asof_idx == op.conditions.size());
			asof_idx = c;
			break;
		default:
			throw InternalException("Invalid ASOF JOIN comparison");
		}
	}
	D_ASSERT(asof_idx < op.conditions.size());

	if (!ClientConfig::GetConfig(context).force_asof_iejoin) {
		return make_uniq<PhysicalAsOfJoin>(op, std::move(left), std::move(right));
	}

	//	Debug implementation: IEJoin of Window
	//	LEAD(asof_column, 1, infinity) OVER (PARTITION BY equi_column... ORDER BY asof_column) AS asof_temp
	auto &asof_comp = op.conditions[asof_idx];
	auto &asof_column = asof_comp.right;
	auto asof_type = asof_column->return_type;
	auto asof_temp = make_uniq<BoundWindowExpression>(ExpressionType::WINDOW_LEAD, asof_type, nullptr, nullptr);
	asof_temp->children.emplace_back(asof_column->Copy());
	asof_temp->offset_expr = make_uniq<BoundConstantExpression>(Value::BIGINT(1));
	// TODO: If infinities are not supported for a type, fake them by looking at LHS statistics?
	asof_temp->default_expr = make_uniq<BoundConstantExpression>(Value::Infinity(asof_type));
	for (auto equi_idx : equi_indexes) {
		asof_temp->partitions.emplace_back(op.conditions[equi_idx].right->Copy());
	}
	asof_temp->orders.emplace_back(OrderType::ASCENDING, OrderByNullType::NULLS_FIRST, asof_column->Copy());
	asof_temp->start = WindowBoundary::UNBOUNDED_PRECEDING;
	asof_temp->end = WindowBoundary::CURRENT_ROW_ROWS;

	vector<unique_ptr<Expression>> window_select;
	window_select.emplace_back(std::move(asof_temp));

	auto window_types = right->types;
	window_types.emplace_back(asof_type);

	auto window = make_uniq<PhysicalWindow>(window_types, std::move(window_select), rhs_cardinality);
	window->children.emplace_back(std::move(right));

	// IEJoin(left, window, conditions || asof_column < asof_temp)
	JoinCondition asof_upper;
	asof_upper.left = asof_comp.left->Copy();
	asof_upper.right = make_uniq<BoundReferenceExpression>(asof_type, window_types.size() - 1);
	asof_upper.comparison = ExpressionType::COMPARE_LESSTHAN;

	// We have an equality condition, so we may have to deal with projection maps.
	// IEJoin does not (currently) support them, so we have to do it manually
	auto proj_types = op.types;
	op.types.clear();

	auto lhs_types = op.children[0]->types;
	op.types = lhs_types;

	auto rhs_types = op.children[1]->types;
	op.types.insert(op.types.end(), rhs_types.begin(), rhs_types.end());

	op.types.emplace_back(asof_type);
	op.conditions.emplace_back(std::move(asof_upper));
	auto iejoin = make_uniq<PhysicalIEJoin>(op, std::move(left), std::move(window), std::move(op.conditions),
	                                        op.join_type, op.estimated_cardinality);

	//	Project away asof_temp and anything from the projection maps
	auto proj = PhysicalProjection::CreateJoinProjection(proj_types, lhs_types, rhs_types, op.left_projection_map,
	                                                     op.right_projection_map, lhs_cardinality);
	proj->children.push_back(std::move(iejoin));

	return proj;
}

} // namespace duckdb




namespace duckdb {

unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(LogicalColumnDataGet &op) {
	D_ASSERT(op.children.size() == 0);
	D_ASSERT(op.collection);

	// create a PhysicalChunkScan pointing towards the owned collection
	auto chunk_scan = make_uniq<PhysicalColumnDataScan>(op.types, PhysicalOperatorType::COLUMN_DATA_SCAN,
	                                                    op.estimated_cardinality, std::move(op.collection));
	return std::move(chunk_scan);
}

} // namespace duckdb



















namespace duckdb {

static bool CanPlanIndexJoin(ClientContext &context, TableScanBindData &bind_data, PhysicalTableScan &scan) {
	auto &table = bind_data.table;
	auto &transaction = DuckTransaction::Get(context, table.catalog);
	auto &local_storage = LocalStorage::Get(transaction);
	if (local_storage.Find(table.GetStorage())) {
		// transaction local appends: skip index join
		return false;
	}
	if (scan.table_filters && !scan.table_filters->filters.empty()) {
		// table scan filters
		return false;
	}
	return true;
}

bool ExtractNumericValue(Value val, int64_t &result) {
	if (!val.type().IsIntegral()) {
		switch (val.type().InternalType()) {
		case PhysicalType::INT16:
			result = val.GetValueUnsafe<int16_t>();
			break;
		case PhysicalType::INT32:
			result = val.GetValueUnsafe<int32_t>();
			break;
		case PhysicalType::INT64:
			result = val.GetValueUnsafe<int64_t>();
			break;
		default:
			return false;
		}
	} else {
		if (!val.DefaultTryCastAs(LogicalType::BIGINT)) {
			return false;
		}
		result = val.GetValue<int64_t>();
	}
	return true;
}

void CheckForPerfectJoinOpt(LogicalComparisonJoin &op, PerfectHashJoinStats &join_state) {
	// we only do this optimization for inner joins
	if (op.join_type != JoinType::INNER) {
		return;
	}
	// with one condition
	if (op.conditions.size() != 1) {
		return;
	}
	// with propagated statistics
	if (op.join_stats.empty()) {
		return;
	}
	for (auto &type : op.children[1]->types) {
		switch (type.InternalType()) {
		case PhysicalType::STRUCT:
		case PhysicalType::LIST:
			return;
		default:
			break;
		}
	}
	// with equality condition and null values not equal
	for (auto &&condition : op.conditions) {
		if (condition.comparison != ExpressionType::COMPARE_EQUAL) {
			return;
		}
	}
	// with integral internal types
	for (auto &&join_stat : op.join_stats) {
		if (!TypeIsInteger(join_stat->GetType().InternalType()) ||
		    join_stat->GetType().InternalType() == PhysicalType::INT128) {
			// perfect join not possible for non-integral types or hugeint
			return;
		}
	}

	// and when the build range is smaller than the threshold
	auto &stats_build = *op.join_stats[0].get(); // lhs stats
	if (!NumericStats::HasMinMax(stats_build)) {
		return;
	}
	int64_t min_value, max_value;
	if (!ExtractNumericValue(NumericStats::Min(stats_build), min_value) ||
	    !ExtractNumericValue(NumericStats::Max(stats_build), max_value)) {
		return;
	}
	int64_t build_range;
	if (!TrySubtractOperator::Operation(max_value, min_value, build_range)) {
		return;
	}

	// Fill join_stats for invisible join
	auto &stats_probe = *op.join_stats[1].get(); // rhs stats
	if (!NumericStats::HasMinMax(stats_probe)) {
		return;
	}

	// The max size our build must have to run the perfect HJ
	const idx_t MAX_BUILD_SIZE = 1000000;
	join_state.probe_min = NumericStats::Min(stats_probe);
	join_state.probe_max = NumericStats::Max(stats_probe);
	join_state.build_min = NumericStats::Min(stats_build);
	join_state.build_max = NumericStats::Max(stats_build);
	join_state.estimated_cardinality = op.estimated_cardinality;
	join_state.build_range = build_range;
	if (join_state.build_range > MAX_BUILD_SIZE) {
		return;
	}
	if (NumericStats::Min(stats_build) <= NumericStats::Min(stats_probe) &&
	    NumericStats::Max(stats_probe) <= NumericStats::Max(stats_build)) {
		join_state.is_probe_in_domain = true;
	}
	join_state.is_build_small = true;
	return;
}

static optional_ptr<Index> CanUseIndexJoin(TableScanBindData &tbl, Expression &expr) {
	optional_ptr<Index> result;
	tbl.table.GetStorage().info->indexes.Scan([&](Index &index) {
		if (index.unbound_expressions.size() != 1) {
			return false;
		}
		if (expr.alias == index.unbound_expressions[0]->alias) {
			result = &index;
			return true;
		}
		return false;
	});
	return result;
}

optional_ptr<Index> CheckIndexJoin(ClientContext &context, LogicalComparisonJoin &op, PhysicalOperator &plan,
                                   Expression &condition) {
	if (op.type == LogicalOperatorType::LOGICAL_DELIM_JOIN) {
		return nullptr;
	}
	// check if one of the tables has an index on column
	if (op.join_type != JoinType::INNER) {
		return nullptr;
	}
	if (op.conditions.size() != 1) {
		return nullptr;
	}
	// check if the child is (1) a table scan, and (2) has an index on the join condition
	if (plan.type != PhysicalOperatorType::TABLE_SCAN) {
		return nullptr;
	}
	auto &tbl_scan = plan.Cast<PhysicalTableScan>();
	auto tbl_data = dynamic_cast<TableScanBindData *>(tbl_scan.bind_data.get());
	if (!tbl_data) {
		return nullptr;
	}
	optional_ptr<Index> result;
	if (CanPlanIndexJoin(context, *tbl_data, tbl_scan)) {
		result = CanUseIndexJoin(*tbl_data, condition);
	}
	return result;
}

static bool PlanIndexJoin(ClientContext &context, LogicalComparisonJoin &op, unique_ptr<PhysicalOperator> &plan,
                          unique_ptr<PhysicalOperator> &left, unique_ptr<PhysicalOperator> &right,
                          optional_ptr<Index> index, bool swap_condition = false) {
	if (!index) {
		return false;
	}
	// index joins are not supported if there are pushed down table filters
	D_ASSERT(right->type == PhysicalOperatorType::TABLE_SCAN);
	auto &tbl_scan = (PhysicalTableScan &)*right;
	//	if (tbl_scan.table_filters && !tbl_scan.table_filters->filters.empty()) {
	//		return false;
	//	}
	// index joins are disabled if enable_optimizer is false
	if (!ClientConfig::GetConfig(context).enable_optimizer) {
		return false;
	}
	// check if the cardinality difference justifies an index join
	if (!((ClientConfig::GetConfig(context).force_index_join ||
	       left->estimated_cardinality < 0.01 * right->estimated_cardinality))) {
		return false;
	}

	// plan the index join
	if (swap_condition) {
		swap(op.conditions[0].left, op.conditions[0].right);
		swap(op.left_projection_map, op.right_projection_map);
	}
	plan = make_uniq<PhysicalIndexJoin>(op, std::move(left), std::move(right), std::move(op.conditions), op.join_type,
	                                    op.left_projection_map, op.right_projection_map, tbl_scan.column_ids, *index,
	                                    !swap_condition, op.estimated_cardinality);
	return true;
}

static bool PlanIndexJoin(ClientContext &context, LogicalComparisonJoin &op, unique_ptr<PhysicalOperator> &plan,
                          unique_ptr<PhysicalOperator> &left, unique_ptr<PhysicalOperator> &right) {
	if (op.conditions.empty()) {
		return false;
	}
	// check if we can plan an index join on the RHS
	auto right_index = CheckIndexJoin(context, op, *right, *op.conditions[0].right);
	if (PlanIndexJoin(context, op, plan, left, right, right_index)) {
		return true;
	}
	// else check if we can plan an index join on the left side
	auto left_index = CheckIndexJoin(context, op, *left, *op.conditions[0].left);
	if (PlanIndexJoin(context, op, plan, right, left, left_index, true)) {
		return true;
	}
	return false;
}

static void RewriteJoinCondition(Expression &expr, idx_t offset) {
	if (expr.type == ExpressionType::BOUND_REF) {
		auto &ref = expr.Cast<BoundReferenceExpression>();
		ref.index += offset;
	}
	ExpressionIterator::EnumerateChildren(expr, [&](Expression &child) { RewriteJoinCondition(child, offset); });
}

unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(LogicalComparisonJoin &op) {
	// now visit the children
	D_ASSERT(op.children.size() == 2);
	idx_t lhs_cardinality = op.children[0]->EstimateCardinality(context);
	idx_t rhs_cardinality = op.children[1]->EstimateCardinality(context);
	auto left = CreatePlan(*op.children[0]);
	auto right = CreatePlan(*op.children[1]);
	left->estimated_cardinality = lhs_cardinality;
	right->estimated_cardinality = rhs_cardinality;
	D_ASSERT(left && right);

	if (op.conditions.empty()) {
		// no conditions: insert a cross product
		return make_uniq<PhysicalCrossProduct>(op.types, std::move(left), std::move(right), op.estimated_cardinality);
	}

	bool has_equality = false;
	// bool has_inequality = false;
	size_t has_range = 0;
	for (size_t c = 0; c < op.conditions.size(); ++c) {
		auto &cond = op.conditions[c];
		switch (cond.comparison) {
		case ExpressionType::COMPARE_EQUAL:
		case ExpressionType::COMPARE_NOT_DISTINCT_FROM:
			has_equality = true;
			break;
		case ExpressionType::COMPARE_LESSTHAN:
		case ExpressionType::COMPARE_GREATERTHAN:
		case ExpressionType::COMPARE_LESSTHANOREQUALTO:
		case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
			++has_range;
			break;
		case ExpressionType::COMPARE_NOTEQUAL:
		case ExpressionType::COMPARE_DISTINCT_FROM:
			// has_inequality = true;
			break;
		default:
			throw NotImplementedException("Unimplemented comparison join");
		}
	}

	unique_ptr<PhysicalOperator> plan;
	if (has_equality) {
		// check if we can use an index join
		if (PlanIndexJoin(context, op, plan, left, right)) {
			return plan;
		}
		// Equality join with small number of keys : possible perfect join optimization
		PerfectHashJoinStats perfect_join_stats;
		CheckForPerfectJoinOpt(op, perfect_join_stats);
		plan = make_uniq<PhysicalHashJoin>(op, std::move(left), std::move(right), std::move(op.conditions),
		                                   op.join_type, op.left_projection_map, op.right_projection_map,
		                                   std::move(op.delim_types), op.estimated_cardinality, perfect_join_stats);

	} else {
		static constexpr const idx_t NESTED_LOOP_JOIN_THRESHOLD = 5;
		bool can_merge = has_range > 0;
		bool can_iejoin = has_range >= 2 && recursive_cte_tables.empty();
		switch (op.join_type) {
		case JoinType::SEMI:
		case JoinType::ANTI:
		case JoinType::MARK:
			can_merge = can_merge && op.conditions.size() == 1;
			can_iejoin = false;
			break;
		default:
			break;
		}
		if (left->estimated_cardinality <= NESTED_LOOP_JOIN_THRESHOLD ||
		    right->estimated_cardinality <= NESTED_LOOP_JOIN_THRESHOLD) {
			can_iejoin = false;
			can_merge = false;
		}
		if (can_iejoin) {
			plan = make_uniq<PhysicalIEJoin>(op, std::move(left), std::move(right), std::move(op.conditions),
			                                 op.join_type, op.estimated_cardinality);
		} else if (can_merge) {
			// range join: use piecewise merge join
			plan =
			    make_uniq<PhysicalPiecewiseMergeJoin>(op, std::move(left), std::move(right), std::move(op.conditions),
			                                          op.join_type, op.estimated_cardinality);
		} else if (PhysicalNestedLoopJoin::IsSupported(op.conditions, op.join_type)) {
			// inequality join: use nested loop
			plan = make_uniq<PhysicalNestedLoopJoin>(op, std::move(left), std::move(right), std::move(op.conditions),
			                                         op.join_type, op.estimated_cardinality);
		} else {
			for (auto &cond : op.conditions) {
				RewriteJoinCondition(*cond.right, left->types.size());
			}
			auto condition = JoinCondition::CreateExpression(std::move(op.conditions));
			plan = make_uniq<PhysicalBlockwiseNLJoin>(op, std::move(left), std::move(right), std::move(condition),
			                                          op.join_type, op.estimated_cardinality);
		}
	}
	return plan;
}

} // namespace duckdb






namespace duckdb {

unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(LogicalCopyToFile &op) {
	auto plan = CreatePlan(*op.children[0]);
	bool preserve_insertion_order = PhysicalPlanGenerator::PreserveInsertionOrder(context, *plan);
	bool supports_batch_index = PhysicalPlanGenerator::UseBatchIndex(context, *plan);
	auto &fs = FileSystem::GetFileSystem(context);
	op.file_path = fs.ExpandPath(op.file_path);
	if (op.use_tmp_file) {
		op.file_path += ".tmp";
	}
	if (op.per_thread_output || op.partition_output || !op.partition_columns.empty() || op.overwrite_or_ignore) {
		// hive-partitioning/per-thread output does not care about insertion order, and does not support batch indexes
		preserve_insertion_order = false;
		supports_batch_index = false;
	}
	auto mode = CopyFunctionExecutionMode::REGULAR_COPY_TO_FILE;
	if (op.function.execution_mode) {
		mode = op.function.execution_mode(preserve_insertion_order, supports_batch_index);
	}
	if (mode == CopyFunctionExecutionMode::BATCH_COPY_TO_FILE) {
		if (!supports_batch_index) {
			throw InternalException("BATCH_COPY_TO_FILE can only be used if batch indexes are supported");
		}
		// batched copy to file
		if (op.function.desired_batch_size) {
			auto copy = make_uniq<PhysicalFixedBatchCopy>(op.types, op.function, std::move(op.bind_data),
			                                              op.estimated_cardinality);
			copy->file_path = op.file_path;
			copy->use_tmp_file = op.use_tmp_file;
			copy->children.push_back(std::move(plan));
			return std::move(copy);
		} else {
			auto copy = make_uniq<PhysicalBatchCopyToFile>(op.types, op.function, std::move(op.bind_data),
			                                               op.estimated_cardinality);
			copy->file_path = op.file_path;
			copy->use_tmp_file = op.use_tmp_file;
			copy->children.push_back(std::move(plan));
			return std::move(copy);
		}
	}
	// COPY from select statement to file
	auto copy = make_uniq<PhysicalCopyToFile>(op.types, op.function, std::move(op.bind_data), op.estimated_cardinality);
	copy->file_path = op.file_path;
	copy->use_tmp_file = op.use_tmp_file;
	copy->overwrite_or_ignore = op.overwrite_or_ignore;
	copy->filename_pattern = op.filename_pattern;
	copy->per_thread_output = op.per_thread_output;
	copy->partition_output = op.partition_output;
	copy->partition_columns = op.partition_columns;
	copy->names = op.names;
	copy->expected_types = op.expected_types;
	copy->parallel = mode == CopyFunctionExecutionMode::PARALLEL_COPY_TO_FILE;

	copy->children.push_back(std::move(plan));
	return std::move(copy);
}

} // namespace duckdb










namespace duckdb {

unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(LogicalCreate &op) {
	switch (op.type) {
	case LogicalOperatorType::LOGICAL_CREATE_SEQUENCE:
		return make_uniq<PhysicalCreateSequence>(unique_ptr_cast<CreateInfo, CreateSequenceInfo>(std::move(op.info)),
		                                         op.estimated_cardinality);
	case LogicalOperatorType::LOGICAL_CREATE_VIEW:
		return make_uniq<PhysicalCreateView>(unique_ptr_cast<CreateInfo, CreateViewInfo>(std::move(op.info)),
		                                     op.estimated_cardinality);
	case LogicalOperatorType::LOGICAL_CREATE_SCHEMA:
		return make_uniq<PhysicalCreateSchema>(unique_ptr_cast<CreateInfo, CreateSchemaInfo>(std::move(op.info)),
		                                       op.estimated_cardinality);
	case LogicalOperatorType::LOGICAL_CREATE_MACRO:
		return make_uniq<PhysicalCreateFunction>(unique_ptr_cast<CreateInfo, CreateMacroInfo>(std::move(op.info)),
		                                         op.estimated_cardinality);
	case LogicalOperatorType::LOGICAL_CREATE_TYPE: {
		unique_ptr<PhysicalOperator> create = make_uniq<PhysicalCreateType>(
		    unique_ptr_cast<CreateInfo, CreateTypeInfo>(std::move(op.info)), op.estimated_cardinality);
		if (!op.children.empty()) {
			D_ASSERT(op.children.size() == 1);
			auto plan = CreatePlan(*op.children[0]);
			create->children.push_back(std::move(plan));
		}
		return create;
	}
	default:
		throw NotImplementedException("Unimplemented type for logical simple create");
	}
}

} // namespace duckdb













namespace duckdb {

unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(LogicalCreateIndex &op) {

	// generate a physical plan for the parallel index creation which consists of the following operators
	// table scan - projection (for expression execution) - filter (NOT NULL) - order - create index

	D_ASSERT(op.children.empty());

	// validate that all expressions contain valid scalar functions
	// e.g. get_current_timestamp(), random(), and sequence values are not allowed as ART keys
	// because they make deletions and lookups unfeasible
	for (idx_t i = 0; i < op.unbound_expressions.size(); i++) {
		auto &expr = op.unbound_expressions[i];
		if (expr->HasSideEffects()) {
			throw BinderException("Index keys cannot contain expressions with side "
			                      "effects.");
		}
	}

	// table scan operator for index key columns and row IDs

	unique_ptr<TableFilterSet> table_filters;
	op.info->column_ids.emplace_back(COLUMN_IDENTIFIER_ROW_ID);

	auto &bind_data = op.bind_data->Cast<TableScanBindData>();
	bind_data.is_create_index = true;

	auto table_scan =
	    make_uniq<PhysicalTableScan>(op.info->scan_types, op.function, std::move(op.bind_data), op.info->column_ids,
	                                 op.info->names, std::move(table_filters), op.estimated_cardinality);

	dependencies.AddDependency(op.table);
	op.info->column_ids.pop_back();

	D_ASSERT(op.info->scan_types.size() - 1 <= op.info->names.size());
	D_ASSERT(op.info->scan_types.size() - 1 <= op.info->column_ids.size());

	// projection to execute expressions on the key columns

	vector<LogicalType> new_column_types;
	vector<unique_ptr<Expression>> select_list;
	for (idx_t i = 0; i < op.expressions.size(); i++) {
		new_column_types.push_back(op.expressions[i]->return_type);
		select_list.push_back(std::move(op.expressions[i]));
	}
	new_column_types.emplace_back(LogicalType::ROW_TYPE);
	select_list.push_back(make_uniq<BoundReferenceExpression>(LogicalType::ROW_TYPE, op.info->scan_types.size() - 1));

	auto projection = make_uniq<PhysicalProjection>(new_column_types, std::move(select_list), op.estimated_cardinality);
	projection->children.push_back(std::move(table_scan));

	// filter operator for IS_NOT_NULL on each key column

	vector<LogicalType> filter_types;
	vector<unique_ptr<Expression>> filter_select_list;

	for (idx_t i = 0; i < new_column_types.size() - 1; i++) {
		filter_types.push_back(new_column_types[i]);
		auto is_not_null_expr =
		    make_uniq<BoundOperatorExpression>(ExpressionType::OPERATOR_IS_NOT_NULL, LogicalType::BOOLEAN);
		auto bound_ref = make_uniq<BoundReferenceExpression>(new_column_types[i], i);
		is_not_null_expr->children.push_back(std::move(bound_ref));
		filter_select_list.push_back(std::move(is_not_null_expr));
	}

	auto null_filter =
	    make_uniq<PhysicalFilter>(std::move(filter_types), std::move(filter_select_list), op.estimated_cardinality);
	null_filter->types.emplace_back(LogicalType::ROW_TYPE);
	null_filter->children.push_back(std::move(projection));

	// order operator

	vector<BoundOrderByNode> orders;
	vector<idx_t> projections;
	for (idx_t i = 0; i < new_column_types.size() - 1; i++) {
		auto col_expr = make_uniq_base<Expression, BoundReferenceExpression>(new_column_types[i], i);
		orders.emplace_back(OrderType::ASCENDING, OrderByNullType::NULLS_FIRST, std::move(col_expr));
		projections.emplace_back(i);
	}
	projections.emplace_back(new_column_types.size() - 1);

	auto physical_order =
	    make_uniq<PhysicalOrder>(new_column_types, std::move(orders), std::move(projections), op.estimated_cardinality);
	physical_order->children.push_back(std::move(null_filter));

	// actual physical create index operator

	auto physical_create_index =
	    make_uniq<PhysicalCreateIndex>(op, op.table, op.info->column_ids, std::move(op.info),
	                                   std::move(op.unbound_expressions), op.estimated_cardinality);
	physical_create_index->children.push_back(std::move(physical_order));
	return std::move(physical_create_index);
}

} // namespace duckdb














namespace duckdb {

unique_ptr<PhysicalOperator> DuckCatalog::PlanCreateTableAs(ClientContext &context, LogicalCreateTable &op,
                                                            unique_ptr<PhysicalOperator> plan) {
	bool parallel_streaming_insert = !PhysicalPlanGenerator::PreserveInsertionOrder(context, *plan);
	bool use_batch_index = PhysicalPlanGenerator::UseBatchIndex(context, *plan);
	auto num_threads = TaskScheduler::GetScheduler(context).NumberOfThreads();
	unique_ptr<PhysicalOperator> create;
	if (!parallel_streaming_insert && use_batch_index) {
		create = make_uniq<PhysicalBatchInsert>(op, op.schema, std::move(op.info), op.estimated_cardinality);

	} else {
		create = make_uniq<PhysicalInsert>(op, op.schema, std::move(op.info), op.estimated_cardinality,
		                                   parallel_streaming_insert && num_threads > 1);
	}

	D_ASSERT(op.children.size() == 1);
	create->children.push_back(std::move(plan));
	return create;
}

unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(LogicalCreateTable &op) {
	const auto &create_info = (CreateTableInfo &)*op.info->base;
	auto &catalog = op.info->schema.catalog;
	auto existing_entry = catalog.GetEntry<TableCatalogEntry>(context, create_info.schema, create_info.table,
	                                                          OnEntryNotFound::RETURN_NULL);
	bool replace = op.info->Base().on_conflict == OnCreateConflict::REPLACE_ON_CONFLICT;
	if ((!existing_entry || replace) && !op.children.empty()) {
		auto plan = CreatePlan(*op.children[0]);
		return op.schema.catalog.PlanCreateTableAs(context, op, std::move(plan));
	} else {
		return make_uniq<PhysicalCreateTable>(op, op.schema, std::move(op.info), op.estimated_cardinality);
	}
}

} // namespace duckdb




namespace duckdb {

unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(LogicalCrossProduct &op) {
	D_ASSERT(op.children.size() == 2);

	auto left = CreatePlan(*op.children[0]);
	auto right = CreatePlan(*op.children[1]);
	return make_uniq<PhysicalCrossProduct>(op.types, std::move(left), std::move(right), op.estimated_cardinality);
}

} // namespace duckdb







namespace duckdb {

unique_ptr<PhysicalOperator> DuckCatalog::PlanDelete(ClientContext &context, LogicalDelete &op,
                                                     unique_ptr<PhysicalOperator> plan) {
	// get the index of the row_id column
	auto &bound_ref = op.expressions[0]->Cast<BoundReferenceExpression>();

	auto del = make_uniq<PhysicalDelete>(op.types, op.table, op.table.GetStorage(), bound_ref.index,
	                                     op.estimated_cardinality, op.return_chunk);
	del->children.push_back(std::move(plan));
	return std::move(del);
}

unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(LogicalDelete &op) {
	D_ASSERT(op.children.size() == 1);
	D_ASSERT(op.expressions.size() == 1);
	D_ASSERT(op.expressions[0]->type == ExpressionType::BOUND_REF);

	auto plan = CreatePlan(*op.children[0]);

	dependencies.AddDependency(op.table);
	return op.table.catalog.PlanDelete(context, op, std::move(plan));
}

} // namespace duckdb




namespace duckdb {

unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(LogicalDelimGet &op) {
	D_ASSERT(op.children.empty());

	// create a PhysicalChunkScan without an owned_collection, the collection will be added later
	auto chunk_scan =
	    make_uniq<PhysicalColumnDataScan>(op.types, PhysicalOperatorType::DELIM_SCAN, op.estimated_cardinality);
	return std::move(chunk_scan);
}

} // namespace duckdb










namespace duckdb {

static void GatherDelimScans(const PhysicalOperator &op, vector<const_reference<PhysicalOperator>> &delim_scans) {
	if (op.type == PhysicalOperatorType::DELIM_SCAN) {
		delim_scans.push_back(op);
	}
	for (auto &child : op.children) {
		GatherDelimScans(*child, delim_scans);
	}
}

unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(LogicalDelimJoin &op) {
	// first create the underlying join
	auto plan = CreatePlan(op.Cast<LogicalComparisonJoin>());
	// this should create a join, not a cross product
	D_ASSERT(plan && plan->type != PhysicalOperatorType::CROSS_PRODUCT);
	// duplicate eliminated join
	// first gather the scans on the duplicate eliminated data set from the RHS
	vector<const_reference<PhysicalOperator>> delim_scans;
	GatherDelimScans(*plan->children[1], delim_scans);
	if (delim_scans.empty()) {
		// no duplicate eliminated scans in the RHS!
		// in this case we don't need to create a delim join
		// just push the normal join
		return plan;
	}
	vector<LogicalType> delim_types;
	vector<unique_ptr<Expression>> distinct_groups, distinct_expressions;
	for (auto &delim_expr : op.duplicate_eliminated_columns) {
		D_ASSERT(delim_expr->type == ExpressionType::BOUND_REF);
		auto &bound_ref = delim_expr->Cast<BoundReferenceExpression>();
		delim_types.push_back(bound_ref.return_type);
		distinct_groups.push_back(make_uniq<BoundReferenceExpression>(bound_ref.return_type, bound_ref.index));
	}
	// now create the duplicate eliminated join
	auto delim_join = make_uniq<PhysicalDelimJoin>(op.types, std::move(plan), delim_scans, op.estimated_cardinality);
	// we still have to create the DISTINCT clause that is used to generate the duplicate eliminated chunk
	delim_join->distinct = make_uniq<PhysicalHashAggregate>(context, delim_types, std::move(distinct_expressions),
	                                                        std::move(distinct_groups), op.estimated_cardinality);
	return std::move(delim_join);
}

} // namespace duckdb









namespace duckdb {

unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(LogicalDistinct &op) {
	D_ASSERT(op.children.size() == 1);
	auto child = CreatePlan(*op.children[0]);
	auto &distinct_targets = op.distinct_targets;
	D_ASSERT(child);
	D_ASSERT(!distinct_targets.empty());

	auto &types = child->GetTypes();
	vector<unique_ptr<Expression>> groups, aggregates, projections;
	idx_t group_count = distinct_targets.size();
	unordered_map<idx_t, idx_t> group_by_references;
	vector<LogicalType> aggregate_types;
	// creates one group per distinct_target
	for (idx_t i = 0; i < distinct_targets.size(); i++) {
		auto &target = distinct_targets[i];
		if (target->type == ExpressionType::BOUND_REF) {
			auto &bound_ref = target->Cast<BoundReferenceExpression>();
			group_by_references[bound_ref.index] = i;
		}
		aggregate_types.push_back(target->return_type);
		groups.push_back(std::move(target));
	}
	bool requires_projection = false;
	if (types.size() != group_count) {
		requires_projection = true;
	}
	// we need to create one aggregate per column in the select_list
	for (idx_t i = 0; i < types.size(); ++i) {
		auto logical_type = types[i];
		// check if we can directly refer to a group, or if we need to push an aggregate with FIRST
		auto entry = group_by_references.find(i);
		if (entry != group_by_references.end()) {
			auto group_index = entry->second;
			// entry is found: can directly refer to a group
			projections.push_back(make_uniq<BoundReferenceExpression>(logical_type, group_index));
			if (group_index != i) {
				// we require a projection only if this group element is out of order
				requires_projection = true;
			}
		} else {
			if (op.distinct_type == DistinctType::DISTINCT && op.order_by) {
				throw InternalException("Entry that is not a group, but not a DISTINCT ON aggregate");
			}
			// entry is not one of the groups: need to push a FIRST aggregate
			auto bound = make_uniq<BoundReferenceExpression>(logical_type, i);
			vector<unique_ptr<Expression>> first_children;
			first_children.push_back(std::move(bound));

			FunctionBinder function_binder(context);
			auto first_aggregate = function_binder.BindAggregateFunction(
			    FirstFun::GetFunction(logical_type), std::move(first_children), nullptr, AggregateType::NON_DISTINCT);
			first_aggregate->order_bys = op.order_by ? op.order_by->Copy() : nullptr;
			// add the projection
			projections.push_back(make_uniq<BoundReferenceExpression>(logical_type, group_count + aggregates.size()));
			// push it to the list of aggregates
			aggregate_types.push_back(logical_type);
			aggregates.push_back(std::move(first_aggregate));
			requires_projection = true;
		}
	}

	child = ExtractAggregateExpressions(std::move(child), aggregates, groups);

	// we add a physical hash aggregation in the plan to select the distinct groups
	auto groupby = make_uniq<PhysicalHashAggregate>(context, aggregate_types, std::move(aggregates), std::move(groups),
	                                                child->estimated_cardinality);
	groupby->children.push_back(std::move(child));
	if (!requires_projection) {
		return std::move(groupby);
	}

	// we add a physical projection on top of the aggregation to project all members in the select list
	auto aggr_projection = make_uniq<PhysicalProjection>(types, std::move(projections), groupby->estimated_cardinality);
	aggr_projection->children.push_back(std::move(groupby));
	return std::move(aggr_projection);
}

} // namespace duckdb




namespace duckdb {

unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(LogicalDummyScan &op) {
	D_ASSERT(op.children.size() == 0);
	return make_uniq<PhysicalDummyScan>(op.types, op.estimated_cardinality);
}

} // namespace duckdb




namespace duckdb {

unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(LogicalEmptyResult &op) {
	D_ASSERT(op.children.size() == 0);
	return make_uniq<PhysicalEmptyResult>(op.types, op.estimated_cardinality);
}

} // namespace duckdb




namespace duckdb {

unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(LogicalExecute &op) {
	if (!op.prepared->plan) {
		D_ASSERT(op.children.size() == 1);
		auto owned_plan = CreatePlan(*op.children[0]);
		auto execute = make_uniq<PhysicalExecute>(*owned_plan);
		execute->owned_plan = std::move(owned_plan);
		execute->prepared = std::move(op.prepared);
		return std::move(execute);
	} else {
		D_ASSERT(op.children.size() == 0);
		return make_uniq<PhysicalExecute>(*op.prepared->plan);
	}
}

} // namespace duckdb








namespace duckdb {

unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(LogicalExplain &op) {
	D_ASSERT(op.children.size() == 1);
	auto logical_plan_opt = op.children[0]->ToString();
	auto plan = CreatePlan(*op.children[0]);
	if (op.explain_type == ExplainType::EXPLAIN_ANALYZE) {
		auto result = make_uniq<PhysicalExplainAnalyze>(op.types);
		result->children.push_back(std::move(plan));
		return std::move(result);
	}

	op.physical_plan = plan->ToString();
	// the output of the explain
	vector<string> keys, values;
	switch (ClientConfig::GetConfig(context).explain_output_type) {
	case ExplainOutputType::OPTIMIZED_ONLY:
		keys = {"logical_opt"};
		values = {logical_plan_opt};
		break;
	case ExplainOutputType::PHYSICAL_ONLY:
		keys = {"physical_plan"};
		values = {op.physical_plan};
		break;
	default:
		keys = {"logical_plan", "logical_opt", "physical_plan"};
		values = {op.logical_plan_unopt, logical_plan_opt, op.physical_plan};
	}

	// create a ColumnDataCollection from the output
	auto &allocator = Allocator::Get(context);
	vector<LogicalType> plan_types {LogicalType::VARCHAR, LogicalType::VARCHAR};
	auto collection =
	    make_uniq<ColumnDataCollection>(context, plan_types, ColumnDataAllocatorType::IN_MEMORY_ALLOCATOR);

	DataChunk chunk;
	chunk.Initialize(allocator, op.types);
	for (idx_t i = 0; i < keys.size(); i++) {
		chunk.SetValue(0, chunk.size(), Value(keys[i]));
		chunk.SetValue(1, chunk.size(), Value(values[i]));
		chunk.SetCardinality(chunk.size() + 1);
		if (chunk.size() == STANDARD_VECTOR_SIZE) {
			collection->Append(chunk);
			chunk.Reset();
		}
	}
	collection->Append(chunk);

	// create a chunk scan to output the result
	auto chunk_scan = make_uniq<PhysicalColumnDataScan>(op.types, PhysicalOperatorType::COLUMN_DATA_SCAN,
	                                                    op.estimated_cardinality, std::move(collection));
	return std::move(chunk_scan);
}

} // namespace duckdb





namespace duckdb {

unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(LogicalExport &op) {
	auto &config = DBConfig::GetConfig(context);
	if (!config.options.enable_external_access) {
		throw PermissionException("Export is disabled through configuration");
	}
	auto export_node = make_uniq<PhysicalExport>(op.types, op.function, std::move(op.copy_info),
	                                             op.estimated_cardinality, op.exported_tables);
	// plan the underlying copy statements, if any
	if (!op.children.empty()) {
		auto plan = CreatePlan(*op.children[0]);
		export_node->children.push_back(std::move(plan));
	}
	return std::move(export_node);
}

} // namespace duckdb






namespace duckdb {

unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(LogicalExpressionGet &op) {
	D_ASSERT(op.children.size() == 1);
	auto plan = CreatePlan(*op.children[0]);

	auto expr_scan = make_uniq<PhysicalExpressionScan>(op.types, std::move(op.expressions), op.estimated_cardinality);
	expr_scan->children.push_back(std::move(plan));
	if (!expr_scan->IsFoldable()) {
		return std::move(expr_scan);
	}
	auto &allocator = Allocator::Get(context);
	// simple expression scan (i.e. no subqueries to evaluate and no prepared statement parameters)
	// we can evaluate all the expressions right now and turn this into a chunk collection scan
	auto chunk_scan = make_uniq<PhysicalColumnDataScan>(op.types, PhysicalOperatorType::COLUMN_DATA_SCAN,
	                                                    expr_scan->expressions.size());
	chunk_scan->owned_collection = make_uniq<ColumnDataCollection>(context, op.types);
	chunk_scan->collection = chunk_scan->owned_collection.get();

	DataChunk chunk;
	chunk.Initialize(allocator, op.types);

	ColumnDataAppendState append_state;
	chunk_scan->owned_collection->InitializeAppend(append_state);
	for (idx_t expression_idx = 0; expression_idx < expr_scan->expressions.size(); expression_idx++) {
		chunk.Reset();
		expr_scan->EvaluateExpression(context, expression_idx, nullptr, chunk);
		chunk_scan->owned_collection->Append(append_state, chunk);
	}
	return std::move(chunk_scan);
}

} // namespace duckdb










namespace duckdb {

unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(LogicalFilter &op) {
	D_ASSERT(op.children.size() == 1);
	unique_ptr<PhysicalOperator> plan = CreatePlan(*op.children[0]);
	if (!op.expressions.empty()) {
		D_ASSERT(plan->types.size() > 0);
		// create a filter if there is anything to filter
		auto filter = make_uniq<PhysicalFilter>(plan->types, std::move(op.expressions), op.estimated_cardinality);
		filter->children.push_back(std::move(plan));
		plan = std::move(filter);
	}
	if (!op.projection_map.empty()) {
		// there is a projection map, generate a physical projection
		vector<unique_ptr<Expression>> select_list;
		for (idx_t i = 0; i < op.projection_map.size(); i++) {
			select_list.push_back(make_uniq<BoundReferenceExpression>(op.types[i], op.projection_map[i]));
		}
		auto proj = make_uniq<PhysicalProjection>(op.types, std::move(select_list), op.estimated_cardinality);
		proj->children.push_back(std::move(plan));
		plan = std::move(proj);
	}
	return plan;
}

} // namespace duckdb









namespace duckdb {

unique_ptr<TableFilterSet> CreateTableFilterSet(TableFilterSet &table_filters, vector<column_t> &column_ids) {
	// create the table filter map
	auto table_filter_set = make_uniq<TableFilterSet>();
	for (auto &table_filter : table_filters.filters) {
		// find the relative column index from the absolute column index into the table
		idx_t column_index = DConstants::INVALID_INDEX;
		for (idx_t i = 0; i < column_ids.size(); i++) {
			if (table_filter.first == column_ids[i]) {
				column_index = i;
				break;
			}
		}
		if (column_index == DConstants::INVALID_INDEX) {
			throw InternalException("Could not find column index for table filter");
		}
		table_filter_set->filters[column_index] = std::move(table_filter.second);
	}
	return table_filter_set;
}

unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(LogicalGet &op) {
	if (!op.children.empty()) {
		// this is for table producing functions that consume subquery results
		D_ASSERT(op.children.size() == 1);
		auto node = make_uniq<PhysicalTableInOutFunction>(op.types, op.function, std::move(op.bind_data), op.column_ids,
		                                                  op.estimated_cardinality, std::move(op.projected_input));
		node->children.push_back(CreatePlan(std::move(op.children[0])));
		return std::move(node);
	}
	if (!op.projected_input.empty()) {
		throw InternalException("LogicalGet::project_input can only be set for table-in-out functions");
	}

	unique_ptr<TableFilterSet> table_filters;
	if (!op.table_filters.filters.empty()) {
		table_filters = CreateTableFilterSet(op.table_filters, op.column_ids);
	}

	if (op.function.dependency) {
		op.function.dependency(dependencies, op.bind_data.get());
	}
	// create the table scan node
	if (!op.function.projection_pushdown) {
		// function does not support projection pushdown
		auto node = make_uniq<PhysicalTableScan>(op.returned_types, op.function, std::move(op.bind_data),
		                                         op.returned_types, op.column_ids, vector<column_t>(), op.names,
		                                         std::move(table_filters), op.estimated_cardinality);
		// first check if an additional projection is necessary
		if (op.column_ids.size() == op.returned_types.size()) {
			bool projection_necessary = false;
			for (idx_t i = 0; i < op.column_ids.size(); i++) {
				if (op.column_ids[i] != i) {
					projection_necessary = true;
					break;
				}
			}
			if (!projection_necessary) {
				// a projection is not necessary if all columns have been requested in-order
				// in that case we just return the node

				return std::move(node);
			}
		}
		// push a projection on top that does the projection
		vector<LogicalType> types;
		vector<unique_ptr<Expression>> expressions;
		for (auto &column_id : op.column_ids) {
			if (column_id == COLUMN_IDENTIFIER_ROW_ID) {
				types.emplace_back(LogicalType::BIGINT);
				expressions.push_back(make_uniq<BoundConstantExpression>(Value::BIGINT(0)));
			} else {
				auto type = op.returned_types[column_id];
				types.push_back(type);
				expressions.push_back(make_uniq<BoundReferenceExpression>(type, column_id));
			}
		}

		auto projection =
		    make_uniq<PhysicalProjection>(std::move(types), std::move(expressions), op.estimated_cardinality);
		projection->children.push_back(std::move(node));
		return std::move(projection);
	} else {
		return make_uniq<PhysicalTableScan>(op.types, op.function, std::move(op.bind_data), op.returned_types,
		                                    op.column_ids, op.projection_ids, op.names, std::move(table_filters),
		                                    op.estimated_cardinality);
	}
}

} // namespace duckdb










namespace duckdb {

static OrderPreservationType OrderPreservationRecursive(PhysicalOperator &op) {
	if (op.IsSource()) {
		return op.SourceOrder();
	}
	for (auto &child : op.children) {
		auto child_preservation = OrderPreservationRecursive(*child);
		if (child_preservation != OrderPreservationType::INSERTION_ORDER) {
			return child_preservation;
		}
	}
	return OrderPreservationType::INSERTION_ORDER;
}

bool PhysicalPlanGenerator::PreserveInsertionOrder(ClientContext &context, PhysicalOperator &plan) {
	auto &config = DBConfig::GetConfig(context);

	auto preservation_type = OrderPreservationRecursive(plan);
	if (preservation_type == OrderPreservationType::FIXED_ORDER) {
		// always need to maintain preservation order
		return true;
	}
	if (preservation_type == OrderPreservationType::NO_ORDER) {
		// never need to preserve order
		return false;
	}
	// preserve insertion order - check flags
	if (!config.options.preserve_insertion_order) {
		// preserving insertion order is disabled by config
		return false;
	}
	return true;
}

bool PhysicalPlanGenerator::PreserveInsertionOrder(PhysicalOperator &plan) {
	return PreserveInsertionOrder(context, plan);
}

bool PhysicalPlanGenerator::UseBatchIndex(ClientContext &context, PhysicalOperator &plan) {
	// TODO: always preserve order if query contains ORDER BY
	auto &scheduler = TaskScheduler::GetScheduler(context);
	if (scheduler.NumberOfThreads() == 1) {
		// batch index usage only makes sense if we are using multiple threads
		return false;
	}
	if (!plan.AllSourcesSupportBatchIndex()) {
		// batch index is not supported
		return false;
	}
	return true;
}

bool PhysicalPlanGenerator::UseBatchIndex(PhysicalOperator &plan) {
	return UseBatchIndex(context, plan);
}

unique_ptr<PhysicalOperator> DuckCatalog::PlanInsert(ClientContext &context, LogicalInsert &op,
                                                     unique_ptr<PhysicalOperator> plan) {
	bool parallel_streaming_insert = !PhysicalPlanGenerator::PreserveInsertionOrder(context, *plan);
	bool use_batch_index = PhysicalPlanGenerator::UseBatchIndex(context, *plan);
	auto num_threads = TaskScheduler::GetScheduler(context).NumberOfThreads();
	if (op.return_chunk) {
		// not supported for RETURNING (yet?)
		parallel_streaming_insert = false;
		use_batch_index = false;
	}
	if (op.action_type != OnConflictAction::THROW) {
		// We don't support ON CONFLICT clause in batch insertion operation currently
		use_batch_index = false;
	}
	if (op.action_type == OnConflictAction::UPDATE) {
		// When we potentially need to perform updates, we have to check that row is not updated twice
		// that currently needs to be done for every chunk, which would add a huge bottleneck to parallelized insertion
		parallel_streaming_insert = false;
	}
	unique_ptr<PhysicalOperator> insert;
	if (use_batch_index && !parallel_streaming_insert) {
		insert = make_uniq<PhysicalBatchInsert>(op.types, op.table, op.column_index_map, std::move(op.bound_defaults),
		                                        op.estimated_cardinality);
	} else {
		insert = make_uniq<PhysicalInsert>(
		    op.types, op.table, op.column_index_map, std::move(op.bound_defaults), std::move(op.expressions),
		    std::move(op.set_columns), std::move(op.set_types), op.estimated_cardinality, op.return_chunk,
		    parallel_streaming_insert && num_threads > 1, op.action_type, std::move(op.on_conflict_condition),
		    std::move(op.do_update_condition), std::move(op.on_conflict_filter), std::move(op.columns_to_fetch));
	}
	D_ASSERT(plan);
	insert->children.push_back(std::move(plan));
	return insert;
}

unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(LogicalInsert &op) {
	unique_ptr<PhysicalOperator> plan;
	if (!op.children.empty()) {
		D_ASSERT(op.children.size() == 1);
		plan = CreatePlan(*op.children[0]);
	}
	dependencies.AddDependency(op.table);
	return op.table.catalog.PlanInsert(context, op, std::move(plan));
}

} // namespace duckdb






namespace duckdb {

unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(LogicalLimit &op) {
	D_ASSERT(op.children.size() == 1);

	auto plan = CreatePlan(*op.children[0]);

	unique_ptr<PhysicalOperator> limit;
	if (!PreserveInsertionOrder(*plan)) {
		// use parallel streaming limit if insertion order is not important
		limit = make_uniq<PhysicalStreamingLimit>(op.types, (idx_t)op.limit_val, op.offset_val, std::move(op.limit),
		                                          std::move(op.offset), op.estimated_cardinality, true);
	} else {
		// maintaining insertion order is important
		if (UseBatchIndex(*plan)) {
			// source supports batch index: use parallel batch limit
			limit = make_uniq<PhysicalLimit>(op.types, (idx_t)op.limit_val, op.offset_val, std::move(op.limit),
			                                 std::move(op.offset), op.estimated_cardinality);
		} else {
			// source does not support batch index: use a non-parallel streaming limit
			limit = make_uniq<PhysicalStreamingLimit>(op.types, (idx_t)op.limit_val, op.offset_val, std::move(op.limit),
			                                          std::move(op.offset), op.estimated_cardinality, false);
		}
	}

	limit->children.push_back(std::move(plan));
	return limit;
}

} // namespace duckdb




namespace duckdb {

unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(LogicalLimitPercent &op) {
	D_ASSERT(op.children.size() == 1);

	auto plan = CreatePlan(*op.children[0]);

	auto limit = make_uniq<PhysicalLimitPercent>(op.types, op.limit_percent, op.offset_val, std::move(op.limit),
	                                             std::move(op.offset), op.estimated_cardinality);
	limit->children.push_back(std::move(plan));
	return std::move(limit);
}

} // namespace duckdb




namespace duckdb {

unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(LogicalOrder &op) {
	D_ASSERT(op.children.size() == 1);

	auto plan = CreatePlan(*op.children[0]);
	if (!op.orders.empty()) {
		vector<idx_t> projections;
		if (op.projections.empty()) {
			for (idx_t i = 0; i < plan->types.size(); i++) {
				projections.push_back(i);
			}
		} else {
			projections = std::move(op.projections);
		}
		auto order =
		    make_uniq<PhysicalOrder>(op.types, std::move(op.orders), std::move(projections), op.estimated_cardinality);
		order->children.push_back(std::move(plan));
		plan = std::move(order);
	}
	return plan;
}

} // namespace duckdb




namespace duckdb {

unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(LogicalPivot &op) {
	D_ASSERT(op.children.size() == 1);
	auto child_plan = CreatePlan(*op.children[0]);
	auto pivot = make_uniq<PhysicalPivot>(std::move(op.types), std::move(child_plan), std::move(op.bound_pivot));
	return std::move(pivot);
}

} // namespace duckdb





namespace duckdb {

unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(LogicalPositionalJoin &op) {
	D_ASSERT(op.children.size() == 2);

	auto left = CreatePlan(*op.children[0]);
	auto right = CreatePlan(*op.children[1]);
	switch (left->type) {
	case PhysicalOperatorType::TABLE_SCAN:
	case PhysicalOperatorType::POSITIONAL_SCAN:
		switch (right->type) {
		case PhysicalOperatorType::TABLE_SCAN:
		case PhysicalOperatorType::POSITIONAL_SCAN:
			return make_uniq<PhysicalPositionalScan>(op.types, std::move(left), std::move(right));
		default:
			break;
		}
	default:
		break;
	}

	return make_uniq<PhysicalPositionalJoin>(op.types, std::move(left), std::move(right), op.estimated_cardinality);
}

} // namespace duckdb




namespace duckdb {

unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(LogicalPragma &op) {
	return make_uniq<PhysicalPragma>(op.function, op.info, op.estimated_cardinality);
}

} // namespace duckdb





namespace duckdb {

unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(LogicalPrepare &op) {
	D_ASSERT(op.children.size() <= 1);

	// generate physical plan
	if (!op.children.empty()) {
		auto plan = CreatePlan(*op.children[0]);
		op.prepared->types = plan->types;
		op.prepared->plan = std::move(plan);
	}

	return make_uniq<PhysicalPrepare>(op.name, std::move(op.prepared), op.estimated_cardinality);
}

} // namespace duckdb





namespace duckdb {

unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(LogicalProjection &op) {
	D_ASSERT(op.children.size() == 1);
	auto plan = CreatePlan(*op.children[0]);

#ifdef DEBUG
	for (auto &expr : op.expressions) {
		D_ASSERT(!expr->IsWindow());
		D_ASSERT(!expr->IsAggregate());
	}
#endif
	if (plan->types.size() == op.types.size()) {
		// check if this projection can be omitted entirely
		// this happens if a projection simply emits the columns in the same order
		// e.g. PROJECTION(#0, #1, #2, #3, ...)
		bool omit_projection = true;
		for (idx_t i = 0; i < op.types.size(); i++) {
			if (op.expressions[i]->type == ExpressionType::BOUND_REF) {
				auto &bound_ref = op.expressions[i]->Cast<BoundReferenceExpression>();
				if (bound_ref.index == i) {
					continue;
				}
			}
			omit_projection = false;
			break;
		}
		if (omit_projection) {
			// the projection only directly projects the child' columns: omit it entirely
			return plan;
		}
	}

	auto projection = make_uniq<PhysicalProjection>(op.types, std::move(op.expressions), op.estimated_cardinality);
	projection->children.push_back(std::move(plan));
	return std::move(projection);
}

} // namespace duckdb








namespace duckdb {

unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(LogicalRecursiveCTE &op) {
	D_ASSERT(op.children.size() == 2);

	// Create the working_table that the PhysicalRecursiveCTE will use for evaluation.
	auto working_table = std::make_shared<ColumnDataCollection>(context, op.types);

	// Add the ColumnDataCollection to the context of this PhysicalPlanGenerator
	recursive_cte_tables[op.table_index] = working_table;

	auto left = CreatePlan(*op.children[0]);
	auto right = CreatePlan(*op.children[1]);

	auto cte = make_uniq<PhysicalRecursiveCTE>(op.types, op.union_all, std::move(left), std::move(right),
	                                           op.estimated_cardinality);
	cte->working_table = working_table;

	return std::move(cte);
}

unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(LogicalCTERef &op) {
	D_ASSERT(op.children.empty());

	auto chunk_scan =
	    make_uniq<PhysicalColumnDataScan>(op.types, PhysicalOperatorType::RECURSIVE_CTE_SCAN, op.estimated_cardinality);

	// CreatePlan of a LogicalRecursiveCTE must have happened before.
	auto cte = recursive_cte_tables.find(op.cte_index);
	if (cte == recursive_cte_tables.end()) {
		throw InvalidInputException("Referenced recursive CTE does not exist.");
	}
	chunk_scan->collection = cte->second.get();
	return std::move(chunk_scan);
}

} // namespace duckdb




namespace duckdb {

unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(LogicalReset &op) {
	return make_uniq<PhysicalReset>(op.name, op.scope, op.estimated_cardinality);
}

} // namespace duckdb






namespace duckdb {

unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(LogicalSample &op) {
	D_ASSERT(op.children.size() == 1);

	auto plan = CreatePlan(*op.children[0]);

	unique_ptr<PhysicalOperator> sample;
	switch (op.sample_options->method) {
	case SampleMethod::RESERVOIR_SAMPLE:
		sample = make_uniq<PhysicalReservoirSample>(op.types, std::move(op.sample_options), op.estimated_cardinality);
		break;
	case SampleMethod::SYSTEM_SAMPLE:
	case SampleMethod::BERNOULLI_SAMPLE:
		if (!op.sample_options->is_percentage) {
			throw ParserException("Sample method %s cannot be used with a discrete sample count, either switch to "
			                      "reservoir sampling or use a sample_size",
			                      EnumUtil::ToString(op.sample_options->method));
		}
		sample = make_uniq<PhysicalStreamingSample>(op.types, op.sample_options->method,
		                                            op.sample_options->sample_size.GetValue<double>(),
		                                            op.sample_options->seed, op.estimated_cardinality);
		break;
	default:
		throw InternalException("Unimplemented sample method");
	}
	sample->children.push_back(std::move(plan));
	return sample;
}

} // namespace duckdb




namespace duckdb {

unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(LogicalSet &op) {
	return make_uniq<PhysicalSet>(op.name, op.value, op.scope, op.estimated_cardinality);
}

} // namespace duckdb






namespace duckdb {

unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(LogicalSetOperation &op) {
	D_ASSERT(op.children.size() == 2);

	auto left = CreatePlan(*op.children[0]);
	auto right = CreatePlan(*op.children[1]);

	if (left->GetTypes() != right->GetTypes()) {
		throw InvalidInputException("Type mismatch for SET OPERATION");
	}

	switch (op.type) {
	case LogicalOperatorType::LOGICAL_UNION:
		// UNION
		return make_uniq<PhysicalUnion>(op.types, std::move(left), std::move(right), op.estimated_cardinality);
	default: {
		// EXCEPT/INTERSECT
		D_ASSERT(op.type == LogicalOperatorType::LOGICAL_EXCEPT || op.type == LogicalOperatorType::LOGICAL_INTERSECT);
		auto &types = left->GetTypes();
		vector<JoinCondition> conditions;
		// create equality condition for all columns
		for (idx_t i = 0; i < types.size(); i++) {
			JoinCondition cond;
			cond.left = make_uniq<BoundReferenceExpression>(types[i], i);
			cond.right = make_uniq<BoundReferenceExpression>(types[i], i);
			cond.comparison = ExpressionType::COMPARE_NOT_DISTINCT_FROM;
			conditions.push_back(std::move(cond));
		}
		// EXCEPT is ANTI join
		// INTERSECT is SEMI join
		PerfectHashJoinStats join_stats; // used in inner joins only
		JoinType join_type = op.type == LogicalOperatorType::LOGICAL_EXCEPT ? JoinType::ANTI : JoinType::SEMI;
		return make_uniq<PhysicalHashJoin>(op, std::move(left), std::move(right), std::move(conditions), join_type,
		                                   op.estimated_cardinality, join_stats);
	}
	}
}

} // namespace duckdb





namespace duckdb {

unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(LogicalShow &op) {
	DataChunk output;
	output.Initialize(Allocator::Get(context), op.types);

	auto collection = make_uniq<ColumnDataCollection>(context, op.types);
	ColumnDataAppendState append_state;
	collection->InitializeAppend(append_state);
	for (idx_t column_idx = 0; column_idx < op.types_select.size(); column_idx++) {
		auto type = op.types_select[column_idx];
		auto &name = op.aliases[column_idx];

		// "name", TypeId::VARCHAR
		output.SetValue(0, output.size(), Value(name));
		// "type", TypeId::VARCHAR
		output.SetValue(1, output.size(), Value(type.ToString()));
		// "null", TypeId::VARCHAR
		output.SetValue(2, output.size(), Value("YES"));
		// "pk", TypeId::BOOL
		output.SetValue(3, output.size(), Value());
		// "dflt_value", TypeId::VARCHAR
		output.SetValue(4, output.size(), Value());
		// "extra", TypeId::VARCHAR
		output.SetValue(5, output.size(), Value());

		output.SetCardinality(output.size() + 1);
		if (output.size() == STANDARD_VECTOR_SIZE) {
			collection->Append(append_state, output);
			output.Reset();
		}
	}

	collection->Append(append_state, output);

	// create a chunk scan to output the result
	auto chunk_scan = make_uniq<PhysicalColumnDataScan>(op.types, PhysicalOperatorType::COLUMN_DATA_SCAN,
	                                                    op.estimated_cardinality, std::move(collection));
	return std::move(chunk_scan);
}

} // namespace duckdb














namespace duckdb {

unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(LogicalSimple &op) {
	switch (op.type) {
	case LogicalOperatorType::LOGICAL_ALTER:
		return make_uniq<PhysicalAlter>(unique_ptr_cast<ParseInfo, AlterInfo>(std::move(op.info)),
		                                op.estimated_cardinality);
	case LogicalOperatorType::LOGICAL_DROP:
		return make_uniq<PhysicalDrop>(unique_ptr_cast<ParseInfo, DropInfo>(std::move(op.info)),
		                               op.estimated_cardinality);
	case LogicalOperatorType::LOGICAL_TRANSACTION:
		return make_uniq<PhysicalTransaction>(unique_ptr_cast<ParseInfo, TransactionInfo>(std::move(op.info)),
		                                      op.estimated_cardinality);
	case LogicalOperatorType::LOGICAL_VACUUM: {
		auto result = make_uniq<PhysicalVacuum>(unique_ptr_cast<ParseInfo, VacuumInfo>(std::move(op.info)),
		                                        op.estimated_cardinality);
		if (!op.children.empty()) {
			auto child = CreatePlan(*op.children[0]);
			result->children.push_back(std::move(child));
		}
		return std::move(result);
	}
	case LogicalOperatorType::LOGICAL_LOAD:
		return make_uniq<PhysicalLoad>(unique_ptr_cast<ParseInfo, LoadInfo>(std::move(op.info)),
		                               op.estimated_cardinality);
	case LogicalOperatorType::LOGICAL_ATTACH:
		return make_uniq<PhysicalAttach>(unique_ptr_cast<ParseInfo, AttachInfo>(std::move(op.info)),
		                                 op.estimated_cardinality);
	case LogicalOperatorType::LOGICAL_DETACH:
		return make_uniq<PhysicalDetach>(unique_ptr_cast<ParseInfo, DetachInfo>(std::move(op.info)),
		                                 op.estimated_cardinality);
	default:
		throw NotImplementedException("Unimplemented type for logical simple operator");
	}
}

} // namespace duckdb




namespace duckdb {

unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(LogicalTopN &op) {
	D_ASSERT(op.children.size() == 1);

	auto plan = CreatePlan(*op.children[0]);

	auto top_n =
	    make_uniq<PhysicalTopN>(op.types, std::move(op.orders), (idx_t)op.limit, op.offset, op.estimated_cardinality);
	top_n->children.push_back(std::move(plan));
	return std::move(top_n);
}

} // namespace duckdb




namespace duckdb {

unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(LogicalUnnest &op) {
	D_ASSERT(op.children.size() == 1);
	auto plan = CreatePlan(*op.children[0]);
	auto unnest = make_uniq<PhysicalUnnest>(op.types, std::move(op.expressions), op.estimated_cardinality);
	unnest->children.push_back(std::move(plan));
	return std::move(unnest);
}

} // namespace duckdb






namespace duckdb {

unique_ptr<PhysicalOperator> DuckCatalog::PlanUpdate(ClientContext &context, LogicalUpdate &op,
                                                     unique_ptr<PhysicalOperator> plan) {
	auto update =
	    make_uniq<PhysicalUpdate>(op.types, op.table, op.table.GetStorage(), op.columns, std::move(op.expressions),
	                              std::move(op.bound_defaults), op.estimated_cardinality, op.return_chunk);

	update->update_is_del_and_insert = op.update_is_del_and_insert;
	update->children.push_back(std::move(plan));
	return std::move(update);
}

unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(LogicalUpdate &op) {
	D_ASSERT(op.children.size() == 1);

	auto plan = CreatePlan(*op.children[0]);

	dependencies.AddDependency(op.table);
	return op.table.catalog.PlanUpdate(context, op, std::move(plan));
}

} // namespace duckdb








#include <numeric>

namespace duckdb {

static bool IsStreamingWindow(unique_ptr<Expression> &expr) {
	auto &wexpr = expr->Cast<BoundWindowExpression>();
	if (!wexpr.partitions.empty() || !wexpr.orders.empty() || wexpr.ignore_nulls) {
		return false;
	}
	switch (wexpr.type) {
	// TODO: add more expression types here?
	case ExpressionType::WINDOW_AGGREGATE:
		// We can stream aggregates if they are "running totals" and don't use filters
		return wexpr.start == WindowBoundary::UNBOUNDED_PRECEDING && wexpr.end == WindowBoundary::CURRENT_ROW_ROWS &&
		       !wexpr.filter_expr;
	case ExpressionType::WINDOW_FIRST_VALUE:
	case ExpressionType::WINDOW_PERCENT_RANK:
	case ExpressionType::WINDOW_RANK:
	case ExpressionType::WINDOW_RANK_DENSE:
	case ExpressionType::WINDOW_ROW_NUMBER:
		return true;
	default:
		return false;
	}
}

unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(LogicalWindow &op) {
	D_ASSERT(op.children.size() == 1);

	auto plan = CreatePlan(*op.children[0]);
#ifdef DEBUG
	for (auto &expr : op.expressions) {
		D_ASSERT(expr->IsWindow());
	}
#endif

	op.estimated_cardinality = op.EstimateCardinality(context);

	// Slice types
	auto types = op.types;
	const auto output_idx = types.size() - op.expressions.size();
	types.resize(output_idx);

	// Identify streaming windows
	vector<idx_t> blocking_windows;
	vector<idx_t> streaming_windows;
	for (idx_t expr_idx = 0; expr_idx < op.expressions.size(); expr_idx++) {
		if (IsStreamingWindow(op.expressions[expr_idx])) {
			streaming_windows.push_back(expr_idx);
		} else {
			blocking_windows.push_back(expr_idx);
		}
	}

	// Process the window functions by sharing the partition/order definitions
	vector<idx_t> evaluation_order;
	while (!blocking_windows.empty() || !streaming_windows.empty()) {
		const bool process_streaming = blocking_windows.empty();
		auto &remaining = process_streaming ? streaming_windows : blocking_windows;

		// Find all functions that share the partitioning of the first remaining expression
		const auto over_idx = remaining[0];
		auto &over_expr = op.expressions[over_idx]->Cast<BoundWindowExpression>();

		vector<idx_t> matching;
		vector<idx_t> unprocessed;
		for (const auto &expr_idx : remaining) {
			D_ASSERT(op.expressions[expr_idx]->GetExpressionClass() == ExpressionClass::BOUND_WINDOW);
			auto &wexpr = op.expressions[expr_idx]->Cast<BoundWindowExpression>();
			if (over_expr.KeysAreCompatible(wexpr)) {
				matching.emplace_back(expr_idx);
			} else {
				unprocessed.emplace_back(expr_idx);
			}
		}
		remaining.swap(unprocessed);

		// Extract the matching expressions
		vector<unique_ptr<Expression>> select_list;
		for (const auto &expr_idx : matching) {
			select_list.emplace_back(std::move(op.expressions[expr_idx]));
			types.emplace_back(op.types[output_idx + expr_idx]);
		}

		// Chain the new window operator on top of the plan
		unique_ptr<PhysicalOperator> window;
		if (process_streaming) {
			window = make_uniq<PhysicalStreamingWindow>(types, std::move(select_list), op.estimated_cardinality);
		} else {
			window = make_uniq<PhysicalWindow>(types, std::move(select_list), op.estimated_cardinality);
		}
		window->children.push_back(std::move(plan));
		plan = std::move(window);

		// Remember the projection order if we changed it
		if (!streaming_windows.empty() || !blocking_windows.empty() || !evaluation_order.empty()) {
			evaluation_order.insert(evaluation_order.end(), matching.begin(), matching.end());
		}
	}

	// Put everything back into place if it moved
	if (!evaluation_order.empty()) {
		vector<unique_ptr<Expression>> select_list(op.types.size());
		// The inputs don't move
		for (idx_t i = 0; i < output_idx; ++i) {
			select_list[i] = make_uniq<BoundReferenceExpression>(op.types[i], i);
		}
		// The outputs have been rearranged
		for (idx_t i = 0; i < evaluation_order.size(); ++i) {
			const auto expr_idx = evaluation_order[i] + output_idx;
			select_list[expr_idx] = make_uniq<BoundReferenceExpression>(op.types[expr_idx], i + output_idx);
		}
		auto proj = make_uniq<PhysicalProjection>(op.types, std::move(select_list), op.estimated_cardinality);
		proj->children.push_back(std::move(plan));
		plan = std::move(proj);
	}

	return plan;
}

} // namespace duckdb












namespace duckdb {

class DependencyExtractor : public LogicalOperatorVisitor {
public:
	explicit DependencyExtractor(DependencyList &dependencies) : dependencies(dependencies) {
	}

protected:
	unique_ptr<Expression> VisitReplace(BoundFunctionExpression &expr, unique_ptr<Expression> *expr_ptr) override {
		// extract dependencies from the bound function expression
		if (expr.function.dependency) {
			expr.function.dependency(expr, dependencies);
		}
		return nullptr;
	}

private:
	DependencyList &dependencies;
};

PhysicalPlanGenerator::PhysicalPlanGenerator(ClientContext &context) : context(context) {
}

PhysicalPlanGenerator::~PhysicalPlanGenerator() {
}

unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(unique_ptr<LogicalOperator> op) {
	auto &profiler = QueryProfiler::Get(context);

	// first resolve column references
	profiler.StartPhase("column_binding");
	ColumnBindingResolver resolver;
	resolver.VisitOperator(*op);
	profiler.EndPhase();

	// now resolve types of all the operators
	profiler.StartPhase("resolve_types");
	op->ResolveOperatorTypes();
	profiler.EndPhase();

	// extract dependencies from the logical plan
	DependencyExtractor extractor(dependencies);
	extractor.VisitOperator(*op);

	// then create the main physical plan
	profiler.StartPhase("create_plan");
	auto plan = CreatePlan(*op);
	profiler.EndPhase();

	plan->Verify();
	return plan;
}

unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(LogicalOperator &op) {
	op.estimated_cardinality = op.EstimateCardinality(context);
	unique_ptr<PhysicalOperator> plan = nullptr;

	switch (op.type) {
	case LogicalOperatorType::LOGICAL_GET:
		plan = CreatePlan(op.Cast<LogicalGet>());
		break;
	case LogicalOperatorType::LOGICAL_PROJECTION:
		plan = CreatePlan(op.Cast<LogicalProjection>());
		break;
	case LogicalOperatorType::LOGICAL_EMPTY_RESULT:
		plan = CreatePlan(op.Cast<LogicalEmptyResult>());
		break;
	case LogicalOperatorType::LOGICAL_FILTER:
		plan = CreatePlan(op.Cast<LogicalFilter>());
		break;
	case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
		plan = CreatePlan(op.Cast<LogicalAggregate>());
		break;
	case LogicalOperatorType::LOGICAL_WINDOW:
		plan = CreatePlan(op.Cast<LogicalWindow>());
		break;
	case LogicalOperatorType::LOGICAL_UNNEST:
		plan = CreatePlan(op.Cast<LogicalUnnest>());
		break;
	case LogicalOperatorType::LOGICAL_LIMIT:
		plan = CreatePlan(op.Cast<LogicalLimit>());
		break;
	case LogicalOperatorType::LOGICAL_LIMIT_PERCENT:
		plan = CreatePlan(op.Cast<LogicalLimitPercent>());
		break;
	case LogicalOperatorType::LOGICAL_SAMPLE:
		plan = CreatePlan(op.Cast<LogicalSample>());
		break;
	case LogicalOperatorType::LOGICAL_ORDER_BY:
		plan = CreatePlan(op.Cast<LogicalOrder>());
		break;
	case LogicalOperatorType::LOGICAL_TOP_N:
		plan = CreatePlan(op.Cast<LogicalTopN>());
		break;
	case LogicalOperatorType::LOGICAL_COPY_TO_FILE:
		plan = CreatePlan(op.Cast<LogicalCopyToFile>());
		break;
	case LogicalOperatorType::LOGICAL_DUMMY_SCAN:
		plan = CreatePlan(op.Cast<LogicalDummyScan>());
		break;
	case LogicalOperatorType::LOGICAL_ANY_JOIN:
		plan = CreatePlan(op.Cast<LogicalAnyJoin>());
		break;
	case LogicalOperatorType::LOGICAL_DELIM_JOIN:
		plan = CreatePlan(op.Cast<LogicalDelimJoin>());
		break;
	case LogicalOperatorType::LOGICAL_ASOF_JOIN:
		plan = CreatePlan(op.Cast<LogicalAsOfJoin>());
		break;
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
		plan = CreatePlan(op.Cast<LogicalComparisonJoin>());
		break;
	case LogicalOperatorType::LOGICAL_CROSS_PRODUCT:
		plan = CreatePlan(op.Cast<LogicalCrossProduct>());
		break;
	case LogicalOperatorType::LOGICAL_POSITIONAL_JOIN:
		plan = CreatePlan(op.Cast<LogicalPositionalJoin>());
		break;
	case LogicalOperatorType::LOGICAL_UNION:
	case LogicalOperatorType::LOGICAL_EXCEPT:
	case LogicalOperatorType::LOGICAL_INTERSECT:
		plan = CreatePlan(op.Cast<LogicalSetOperation>());
		break;
	case LogicalOperatorType::LOGICAL_INSERT:
		plan = CreatePlan(op.Cast<LogicalInsert>());
		break;
	case LogicalOperatorType::LOGICAL_DELETE:
		plan = CreatePlan(op.Cast<LogicalDelete>());
		break;
	case LogicalOperatorType::LOGICAL_CHUNK_GET:
		plan = CreatePlan(op.Cast<LogicalColumnDataGet>());
		break;
	case LogicalOperatorType::LOGICAL_DELIM_GET:
		plan = CreatePlan(op.Cast<LogicalDelimGet>());
		break;
	case LogicalOperatorType::LOGICAL_EXPRESSION_GET:
		plan = CreatePlan(op.Cast<LogicalExpressionGet>());
		break;
	case LogicalOperatorType::LOGICAL_UPDATE:
		plan = CreatePlan(op.Cast<LogicalUpdate>());
		break;
	case LogicalOperatorType::LOGICAL_CREATE_TABLE:
		plan = CreatePlan(op.Cast<LogicalCreateTable>());
		break;
	case LogicalOperatorType::LOGICAL_CREATE_INDEX:
		plan = CreatePlan(op.Cast<LogicalCreateIndex>());
		break;
	case LogicalOperatorType::LOGICAL_EXPLAIN:
		plan = CreatePlan(op.Cast<LogicalExplain>());
		break;
	case LogicalOperatorType::LOGICAL_SHOW:
		plan = CreatePlan(op.Cast<LogicalShow>());
		break;
	case LogicalOperatorType::LOGICAL_DISTINCT:
		plan = CreatePlan(op.Cast<LogicalDistinct>());
		break;
	case LogicalOperatorType::LOGICAL_PREPARE:
		plan = CreatePlan(op.Cast<LogicalPrepare>());
		break;
	case LogicalOperatorType::LOGICAL_EXECUTE:
		plan = CreatePlan(op.Cast<LogicalExecute>());
		break;
	case LogicalOperatorType::LOGICAL_CREATE_VIEW:
	case LogicalOperatorType::LOGICAL_CREATE_SEQUENCE:
	case LogicalOperatorType::LOGICAL_CREATE_SCHEMA:
	case LogicalOperatorType::LOGICAL_CREATE_MACRO:
	case LogicalOperatorType::LOGICAL_CREATE_TYPE:
		plan = CreatePlan(op.Cast<LogicalCreate>());
		break;
	case LogicalOperatorType::LOGICAL_PRAGMA:
		plan = CreatePlan(op.Cast<LogicalPragma>());
		break;
	case LogicalOperatorType::LOGICAL_TRANSACTION:
	case LogicalOperatorType::LOGICAL_ALTER:
	case LogicalOperatorType::LOGICAL_DROP:
	case LogicalOperatorType::LOGICAL_VACUUM:
	case LogicalOperatorType::LOGICAL_LOAD:
	case LogicalOperatorType::LOGICAL_ATTACH:
	case LogicalOperatorType::LOGICAL_DETACH:
		plan = CreatePlan(op.Cast<LogicalSimple>());
		break;
	case LogicalOperatorType::LOGICAL_RECURSIVE_CTE:
		plan = CreatePlan(op.Cast<LogicalRecursiveCTE>());
		break;
	case LogicalOperatorType::LOGICAL_CTE_REF:
		plan = CreatePlan(op.Cast<LogicalCTERef>());
		break;
	case LogicalOperatorType::LOGICAL_EXPORT:
		plan = CreatePlan(op.Cast<LogicalExport>());
		break;
	case LogicalOperatorType::LOGICAL_SET:
		plan = CreatePlan(op.Cast<LogicalSet>());
		break;
	case LogicalOperatorType::LOGICAL_RESET:
		plan = CreatePlan(op.Cast<LogicalReset>());
		break;
	case LogicalOperatorType::LOGICAL_PIVOT:
		plan = CreatePlan((LogicalPivot &)op);
		break;
	case LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR:
		plan = ((LogicalExtensionOperator &)op).CreatePlan(context, *this);

		if (!plan) {
			throw InternalException("Missing PhysicalOperator for Extension Operator");
		}
		break;
	case LogicalOperatorType::LOGICAL_JOIN:
	case LogicalOperatorType::LOGICAL_INVALID: {
		throw NotImplementedException("Unimplemented logical operator type!");
	}
	}
	if (!plan) {
		throw InternalException("Physical plan generator - no plan generated");
	}

	if (op.estimated_props) {
		plan->estimated_cardinality = op.estimated_props->GetCardinality<idx_t>();
		plan->estimated_props = op.estimated_props->Copy();
	} else {
		plan->estimated_props = make_uniq<EstimatedProperties>();
	}

	return plan;
}

} // namespace duckdb







namespace duckdb {

// compute the GROUPING values
// for each parameter to the GROUPING clause, we check if the hash table groups on this particular group
// if it does, we return 0, otherwise we return 1
// we then use bitshifts to combine these values
void RadixPartitionedHashTable::SetGroupingValues() {
	auto &grouping_functions = op.GetGroupingFunctions();
	for (auto &grouping : grouping_functions) {
		int64_t grouping_value = 0;
		D_ASSERT(grouping.size() < sizeof(int64_t) * 8);
		for (idx_t i = 0; i < grouping.size(); i++) {
			if (grouping_set.find(grouping[i]) == grouping_set.end()) {
				// we don't group on this value!
				grouping_value += (int64_t)1 << (grouping.size() - (i + 1));
			}
		}
		grouping_values.push_back(Value::BIGINT(grouping_value));
	}
}

RadixPartitionedHashTable::RadixPartitionedHashTable(GroupingSet &grouping_set_p, const GroupedAggregateData &op_p)
    : grouping_set(grouping_set_p), op(op_p) {

	auto groups_count = op.GroupCount();
	for (idx_t i = 0; i < groups_count; i++) {
		if (grouping_set.find(i) == grouping_set.end()) {
			null_groups.push_back(i);
		}
	}

	// 10000 seems like a good compromise here
	radix_limit = 10000;

	if (grouping_set.empty()) {
		// fake a single group with a constant value for aggregation without groups
		group_types.emplace_back(LogicalType::TINYINT);
	}
	for (auto &entry : grouping_set) {
		D_ASSERT(entry < op.group_types.size());
		group_types.push_back(op.group_types[entry]);
	}
	SetGroupingValues();
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
class RadixHTGlobalState : public GlobalSinkState {
	constexpr const static idx_t MAX_RADIX_PARTITIONS = 32;

public:
	explicit RadixHTGlobalState(ClientContext &context)
	    : is_empty(true), multi_scan(true), partitioned(false),
	      partition_info(
	          MinValue<idx_t>(MAX_RADIX_PARTITIONS, TaskScheduler::GetScheduler(context).NumberOfThreads())) {
	}

	vector<unique_ptr<PartitionableHashTable>> intermediate_hts;
	vector<shared_ptr<GroupedAggregateHashTable>> finalized_hts;

	//! Whether or not any tuples were added to the HT
	bool is_empty;
	//! Whether or not the hash table should be scannable multiple times
	bool multi_scan;
	//! The lock for updating the global aggregate state
	mutex lock;
	//! Whether or not any thread has crossed the partitioning threshold
	atomic<bool> partitioned;

	bool is_finalized = false;
	bool is_partitioned = false;

	RadixPartitionInfo partition_info;
	AggregateHTAppendState append_state;
};

class RadixHTLocalState : public LocalSinkState {
public:
	explicit RadixHTLocalState(const RadixPartitionedHashTable &ht) : total_groups(0), is_empty(true) {
		// if there are no groups we create a fake group so everything has the same group
		group_chunk.InitializeEmpty(ht.group_types);
		if (ht.grouping_set.empty()) {
			group_chunk.data[0].Reference(Value::TINYINT(42));
		}
	}

	DataChunk group_chunk;
	//! The aggregate HT
	unique_ptr<PartitionableHashTable> ht;
	//! The total number of groups found by this thread
	idx_t total_groups;

	//! Whether or not any tuples were added to the HT
	bool is_empty;
};

void RadixPartitionedHashTable::SetMultiScan(GlobalSinkState &state) {
	auto &gstate = state.Cast<RadixHTGlobalState>();
	gstate.multi_scan = true;
}

unique_ptr<GlobalSinkState> RadixPartitionedHashTable::GetGlobalSinkState(ClientContext &context) const {
	return make_uniq<RadixHTGlobalState>(context);
}

unique_ptr<LocalSinkState> RadixPartitionedHashTable::GetLocalSinkState(ExecutionContext &context) const {
	return make_uniq<RadixHTLocalState>(*this);
}

void RadixPartitionedHashTable::PopulateGroupChunk(DataChunk &group_chunk, DataChunk &input_chunk) const {
	idx_t chunk_index = 0;
	// Populate the group_chunk
	for (auto &group_idx : grouping_set) {
		// Retrieve the expression containing the index in the input chunk
		auto &group = op.groups[group_idx];
		D_ASSERT(group->type == ExpressionType::BOUND_REF);
		auto &bound_ref_expr = group->Cast<BoundReferenceExpression>();
		// Reference from input_chunk[group.index] -> group_chunk[chunk_index]
		group_chunk.data[chunk_index++].Reference(input_chunk.data[bound_ref_expr.index]);
	}
	group_chunk.SetCardinality(input_chunk.size());
	group_chunk.Verify();
}

void RadixPartitionedHashTable::Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input,
                                     DataChunk &payload_input, const unsafe_vector<idx_t> &filter) const {
	auto &llstate = input.local_state.Cast<RadixHTLocalState>();
	auto &gstate = input.global_state.Cast<RadixHTGlobalState>();
	D_ASSERT(!gstate.is_finalized);

	DataChunk &group_chunk = llstate.group_chunk;
	PopulateGroupChunk(group_chunk, chunk);

	// if we have non-combinable aggregates (e.g. string_agg) we cannot keep parallel hash
	// tables
	if (ForceSingleHT(input.global_state)) {
		lock_guard<mutex> glock(gstate.lock);
		gstate.is_empty = gstate.is_empty && group_chunk.size() == 0;
		if (gstate.finalized_hts.empty()) {
			// Create a finalized ht in the global state, that we can populate
			gstate.finalized_hts.push_back(
			    make_shared<GroupedAggregateHashTable>(context.client, Allocator::Get(context.client), group_types,
			                                           op.payload_types, op.bindings, HtEntryType::HT_WIDTH_64));
		}
		D_ASSERT(gstate.finalized_hts.size() == 1);
		D_ASSERT(gstate.finalized_hts[0]);
		llstate.total_groups +=
		    gstate.finalized_hts[0]->AddChunk(gstate.append_state, group_chunk, payload_input, filter);
		return;
	}

	if (group_chunk.size() > 0) {
		llstate.is_empty = false;
	}

	if (!llstate.ht) {
		llstate.ht =
		    make_uniq<PartitionableHashTable>(context.client, Allocator::Get(context.client), gstate.partition_info,
		                                      group_types, op.payload_types, op.bindings);
	}

	llstate.total_groups += llstate.ht->AddChunk(group_chunk, payload_input,
	                                             gstate.partitioned && gstate.partition_info.n_partitions > 1, filter);
	if (llstate.total_groups >= radix_limit) {
		gstate.partitioned = true;
	}
}

void RadixPartitionedHashTable::Combine(ExecutionContext &context, GlobalSinkState &state,
                                        LocalSinkState &lstate) const {
	auto &llstate = lstate.Cast<RadixHTLocalState>();
	auto &gstate = state.Cast<RadixHTGlobalState>();
	D_ASSERT(!gstate.is_finalized);

	// this actually does not do a lot but just pushes the local HTs into the global state so we can later combine them
	// in parallel

	if (ForceSingleHT(state)) {
		D_ASSERT(gstate.finalized_hts.size() <= 1);
		return;
	}

	if (!llstate.ht) {
		return; // no data
	}

	if (!llstate.ht->IsPartitioned() && gstate.partition_info.n_partitions > 1 && gstate.partitioned) {
		llstate.ht->Partition();
	}

	// we will never add new values to these HTs so we can drop the first part of the HT
	llstate.ht->Finalize();

	lock_guard<mutex> glock(gstate.lock);
	if (!llstate.is_empty) {
		gstate.is_empty = false;
	}
	// at this point we just collect them the PhysicalHashAggregateFinalizeTask (below) will merge them in parallel
	gstate.intermediate_hts.push_back(std::move(llstate.ht));
}

bool RadixPartitionedHashTable::Finalize(ClientContext &context, GlobalSinkState &gstate_p) const {
	auto &gstate = gstate_p.Cast<RadixHTGlobalState>();
	D_ASSERT(!gstate.is_finalized);
	gstate.is_finalized = true;

	// special case if we have non-combinable aggregates
	// we have already aggreagted into a global shared HT that does not require any additional finalization steps
	if (ForceSingleHT(gstate)) {
		D_ASSERT(gstate.finalized_hts.size() <= 1);
		D_ASSERT(gstate.finalized_hts.empty() || gstate.finalized_hts[0]);
		return false;
	}

	// we can have two cases now, non-partitioned for few groups and radix-partitioned for very many groups.
	// go through all of the child hts and see if we ever called partition() on any of them
	// if we did, its the latter case.
	bool any_partitioned = false;
	for (auto &pht : gstate.intermediate_hts) {
		if (pht->IsPartitioned()) {
			any_partitioned = true;
			break;
		}
	}

	auto &allocator = Allocator::Get(context);
	if (any_partitioned) {
		// if one is partitioned, all have to be
		// this should mostly have already happened in Combine, but if not we do it here
		for (auto &pht : gstate.intermediate_hts) {
			if (!pht->IsPartitioned()) {
				pht->Partition();
			}
		}
		// schedule additional tasks to combine the partial HTs
		gstate.finalized_hts.resize(gstate.partition_info.n_partitions);
		for (idx_t r = 0; r < gstate.partition_info.n_partitions; r++) {
			gstate.finalized_hts[r] = make_shared<GroupedAggregateHashTable>(
			    context, allocator, group_types, op.payload_types, op.bindings, HtEntryType::HT_WIDTH_64);
		}
		gstate.is_partitioned = true;
		return true;
	} else { // in the non-partitioned case we immediately combine all the unpartitioned hts created by the threads.
		     // TODO possible optimization, if total count < limit for 32 bit ht, use that one
		     // create this ht here so finalize needs no lock on gstate

		gstate.finalized_hts.push_back(make_shared<GroupedAggregateHashTable>(
		    context, allocator, group_types, op.payload_types, op.bindings, HtEntryType::HT_WIDTH_64));
		for (auto &pht : gstate.intermediate_hts) {
			auto unpartitioned = pht->GetUnpartitioned();
			for (auto &unpartitioned_ht : unpartitioned) {
				D_ASSERT(unpartitioned_ht);
				gstate.finalized_hts[0]->Combine(*unpartitioned_ht);
				unpartitioned_ht.reset();
			}
			unpartitioned.clear();
		}
		D_ASSERT(gstate.finalized_hts[0]);
		gstate.finalized_hts[0]->Finalize();
		return false;
	}
}

// this task is run in multiple threads and combines the radix-partitioned hash tables into a single onen and then
// folds them into the global ht finally.
class RadixAggregateFinalizeTask : public ExecutorTask {
public:
	RadixAggregateFinalizeTask(Executor &executor, shared_ptr<Event> event_p, RadixHTGlobalState &state_p,
	                           idx_t radix_p)
	    : ExecutorTask(executor), event(std::move(event_p)), state(state_p), radix(radix_p) {
	}

	static void FinalizeHT(RadixHTGlobalState &gstate, idx_t radix) {
		D_ASSERT(gstate.partition_info.n_partitions <= gstate.finalized_hts.size());
		D_ASSERT(gstate.finalized_hts[radix]);
		for (auto &pht : gstate.intermediate_hts) {
			for (auto &ht : pht->GetPartition(radix)) {
				gstate.finalized_hts[radix]->Combine(*ht);
				ht.reset();
			}
		}
		gstate.finalized_hts[radix]->Finalize();
	}

	TaskExecutionResult ExecuteTask(TaskExecutionMode mode) override {
		FinalizeHT(state, radix);
		event->FinishTask();
		return TaskExecutionResult::TASK_FINISHED;
	}

private:
	shared_ptr<Event> event;
	RadixHTGlobalState &state;
	idx_t radix;
};

void RadixPartitionedHashTable::ScheduleTasks(Executor &executor, const shared_ptr<Event> &event,
                                              GlobalSinkState &state, vector<shared_ptr<Task>> &tasks) const {
	auto &gstate = state.Cast<RadixHTGlobalState>();
	if (!gstate.is_partitioned) {
		return;
	}
	for (idx_t r = 0; r < gstate.partition_info.n_partitions; r++) {
		D_ASSERT(gstate.partition_info.n_partitions <= gstate.finalized_hts.size());
		D_ASSERT(gstate.finalized_hts[r]);
		tasks.push_back(make_uniq<RadixAggregateFinalizeTask>(executor, event, gstate, r));
	}
}

bool RadixPartitionedHashTable::ForceSingleHT(GlobalSinkState &state) const {
	auto &gstate = state.Cast<RadixHTGlobalState>();
	return gstate.partition_info.n_partitions < 2;
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
class RadixHTGlobalSourceState : public GlobalSourceState {
public:
	explicit RadixHTGlobalSourceState(Allocator &allocator, const RadixPartitionedHashTable &ht)
	    : ht_index(0), initialized(false), finished(false) {
	}

	//! Heavy handed for now.
	mutex lock;
	//! The current position to scan the HT for output tuples
	idx_t ht_index;
	//! The set of aggregate scan states
	unsafe_unique_array<TupleDataParallelScanState> ht_scan_states;
	atomic<bool> initialized;
	atomic<bool> finished;
};

class RadixHTLocalSourceState : public LocalSourceState {
public:
	explicit RadixHTLocalSourceState(ExecutionContext &context, const RadixPartitionedHashTable &ht) {
		auto &allocator = Allocator::Get(context.client);
		auto scan_chunk_types = ht.group_types;
		for (auto &aggr_type : ht.op.aggregate_return_types) {
			scan_chunk_types.push_back(aggr_type);
		}
		scan_chunk.Initialize(allocator, scan_chunk_types);
	}

	//! Materialized GROUP BY expressions & aggregates
	DataChunk scan_chunk;
	//! HT index
	idx_t ht_index = DConstants::INVALID_INDEX;
	//! A reference to the current HT that we are scanning
	shared_ptr<GroupedAggregateHashTable> ht;
	//! Scan state for the current HT
	TupleDataLocalScanState scan_state;
};

unique_ptr<GlobalSourceState> RadixPartitionedHashTable::GetGlobalSourceState(ClientContext &context) const {
	return make_uniq<RadixHTGlobalSourceState>(Allocator::Get(context), *this);
}

unique_ptr<LocalSourceState> RadixPartitionedHashTable::GetLocalSourceState(ExecutionContext &context) const {
	return make_uniq<RadixHTLocalSourceState>(context, *this);
}

idx_t RadixPartitionedHashTable::Size(GlobalSinkState &sink_state) const {
	auto &gstate = sink_state.Cast<RadixHTGlobalState>();
	if (gstate.is_empty && grouping_set.empty()) {
		return 1;
	}

	idx_t count = 0;
	for (const auto &ht : gstate.finalized_hts) {
		count += ht->Count();
	}
	return count;
}

SourceResultType RadixPartitionedHashTable::GetData(ExecutionContext &context, DataChunk &chunk,
                                                    GlobalSinkState &sink_state, OperatorSourceInput &input) const {
	auto &gstate = sink_state.Cast<RadixHTGlobalState>();
	auto &state = input.global_state.Cast<RadixHTGlobalSourceState>();
	auto &lstate = input.local_state.Cast<RadixHTLocalSourceState>();
	D_ASSERT(gstate.is_finalized);
	if (state.finished) {
		return SourceResultType::FINISHED;
	}

	// special case hack to sort out aggregating from empty intermediates
	// for aggregations without groups
	if (gstate.is_empty && grouping_set.empty()) {
		D_ASSERT(chunk.ColumnCount() == null_groups.size() + op.aggregates.size() + op.grouping_functions.size());
		// for each column in the aggregates, set to initial state
		chunk.SetCardinality(1);
		for (auto null_group : null_groups) {
			chunk.data[null_group].SetVectorType(VectorType::CONSTANT_VECTOR);
			ConstantVector::SetNull(chunk.data[null_group], true);
		}
		for (idx_t i = 0; i < op.aggregates.size(); i++) {
			D_ASSERT(op.aggregates[i]->GetExpressionClass() == ExpressionClass::BOUND_AGGREGATE);
			auto &aggr = op.aggregates[i]->Cast<BoundAggregateExpression>();
			auto aggr_state = make_unsafe_uniq_array<data_t>(aggr.function.state_size());
			aggr.function.initialize(aggr_state.get());

			AggregateInputData aggr_input_data(aggr.bind_info.get(), Allocator::DefaultAllocator());
			Vector state_vector(Value::POINTER((uintptr_t)aggr_state.get()));
			aggr.function.finalize(state_vector, aggr_input_data, chunk.data[null_groups.size() + i], 1, 0);
			if (aggr.function.destructor) {
				aggr.function.destructor(state_vector, aggr_input_data, 1);
			}
		}
		// Place the grouping values (all the groups of the grouping_set condensed into a single value)
		// Behind the null groups + aggregates
		for (idx_t i = 0; i < op.grouping_functions.size(); i++) {
			chunk.data[null_groups.size() + op.aggregates.size() + i].Reference(grouping_values[i]);
		}
		state.finished = true;
		return chunk.size() == 0 ? SourceResultType::FINISHED : SourceResultType::HAVE_MORE_OUTPUT;
	}
	if (gstate.is_empty) {
		state.finished = true;
		return chunk.size() == 0 ? SourceResultType::FINISHED : SourceResultType::HAVE_MORE_OUTPUT;
	}
	idx_t elements_found = 0;

	lstate.scan_chunk.Reset();
	if (!state.initialized) {
		lock_guard<mutex> l(state.lock);
		if (!state.initialized) {
			auto &finalized_hts = gstate.finalized_hts;
			state.ht_scan_states = make_unsafe_uniq_array<TupleDataParallelScanState>(finalized_hts.size());

			const auto &layout = gstate.finalized_hts[0]->GetDataCollection().GetLayout();
			vector<column_t> column_ids;
			column_ids.reserve(layout.ColumnCount() - 1);
			for (idx_t col_idx = 0; col_idx < layout.ColumnCount() - 1; col_idx++) {
				column_ids.emplace_back(col_idx);
			}

			for (idx_t ht_idx = 0; ht_idx < finalized_hts.size(); ht_idx++) {
				gstate.finalized_hts[ht_idx]->GetDataCollection().InitializeScan(
				    state.ht_scan_states.get()[ht_idx].scan_state, column_ids);
			}
			state.initialized = true;
		}
	}

	auto &local_scan_state = lstate.scan_state;
	while (true) {
		D_ASSERT(state.ht_scan_states);
		idx_t ht_index;
		{
			lock_guard<mutex> l(state.lock);
			ht_index = state.ht_index;
			if (ht_index >= gstate.finalized_hts.size()) {
				state.finished = true;
				return chunk.size() == 0 ? SourceResultType::FINISHED : SourceResultType::HAVE_MORE_OUTPUT;
			}
		}
		D_ASSERT(ht_index < gstate.finalized_hts.size());
		if (lstate.ht_index != DConstants::INVALID_INDEX && ht_index != lstate.ht_index) {
			lstate.ht->GetDataCollection().FinalizePinState(local_scan_state.pin_state);
		}
		lstate.ht_index = ht_index;
		lstate.ht = gstate.finalized_hts[ht_index];
		D_ASSERT(lstate.ht);

		auto &global_scan_state = state.ht_scan_states[ht_index];
		elements_found = lstate.ht->Scan(global_scan_state, local_scan_state, lstate.scan_chunk);
		if (elements_found > 0) {
			break;
		}
		lstate.ht->GetDataCollection().FinalizePinState(local_scan_state.pin_state);

		// move to the next hash table
		lock_guard<mutex> l(state.lock);
		ht_index++;
		if (ht_index > state.ht_index) {
			// we have not yet worked on the table
			// move the global index forwards
			if (!gstate.multi_scan) {
				gstate.finalized_hts[state.ht_index].reset();
			}
			state.ht_index = ht_index;
		}
	}

	// compute the final projection list
	chunk.SetCardinality(elements_found);

	idx_t chunk_index = 0;
	for (auto &entry : grouping_set) {
		chunk.data[entry].Reference(lstate.scan_chunk.data[chunk_index++]);
	}
	for (auto null_group : null_groups) {
		chunk.data[null_group].SetVectorType(VectorType::CONSTANT_VECTOR);
		ConstantVector::SetNull(chunk.data[null_group], true);
	}
	D_ASSERT(grouping_set.size() + null_groups.size() == op.GroupCount());
	for (idx_t col_idx = 0; col_idx < op.aggregates.size(); col_idx++) {
		chunk.data[op.GroupCount() + col_idx].Reference(lstate.scan_chunk.data[group_types.size() + col_idx]);
	}
	D_ASSERT(op.grouping_functions.size() == grouping_values.size());
	for (idx_t i = 0; i < op.grouping_functions.size(); i++) {
		chunk.data[op.GroupCount() + op.aggregates.size() + i].Reference(grouping_values[i]);
	}
	return chunk.size() == 0 ? SourceResultType::FINISHED : SourceResultType::HAVE_MORE_OUTPUT;
}

} // namespace duckdb



namespace duckdb {

ReservoirSample::ReservoirSample(Allocator &allocator, idx_t sample_count, int64_t seed)
    : BlockingSample(seed), sample_count(sample_count), reservoir(allocator) {
}

void ReservoirSample::AddToReservoir(DataChunk &input) {
	if (sample_count == 0) {
		return;
	}
	// Input: A population V of n weighted items
	// Output: A reservoir R with a size m
	// 1: The first m items of V are inserted into R
	// first we need to check if the reservoir already has "m" elements
	if (reservoir.Count() < sample_count) {
		if (FillReservoir(input) == 0) {
			// entire chunk was consumed by reservoir
			return;
		}
	}
	// find the position of next_index relative to current_count
	idx_t remaining = input.size();
	idx_t base_offset = 0;
	while (true) {
		idx_t offset = base_reservoir_sample.next_index - base_reservoir_sample.current_count;
		if (offset >= remaining) {
			// not in this chunk! increment current count and go to the next chunk
			base_reservoir_sample.current_count += remaining;
			return;
		}
		// in this chunk! replace the element
		ReplaceElement(input, base_offset + offset);
		// shift the chunk forward
		remaining -= offset;
		base_offset += offset;
	}
}

unique_ptr<DataChunk> ReservoirSample::GetChunk() {
	return reservoir.Fetch();
}

void ReservoirSample::ReplaceElement(DataChunk &input, idx_t index_in_chunk) {
	// replace the entry in the reservoir
	// 8. The item in R with the minimum key is replaced by item vi
	for (idx_t col_idx = 0; col_idx < input.ColumnCount(); col_idx++) {
		reservoir.SetValue(col_idx, base_reservoir_sample.min_entry, input.GetValue(col_idx, index_in_chunk));
	}
	base_reservoir_sample.ReplaceElement();
}

idx_t ReservoirSample::FillReservoir(DataChunk &input) {
	idx_t chunk_count = input.size();
	input.Flatten();

	// we have not: append to the reservoir
	idx_t required_count;
	if (reservoir.Count() + chunk_count >= sample_count) {
		// have to limit the count of the chunk
		required_count = sample_count - reservoir.Count();
	} else {
		// we copy the entire chunk
		required_count = chunk_count;
	}
	// instead of copying we just change the pointer in the current chunk
	input.SetCardinality(required_count);
	reservoir.Append(input);

	base_reservoir_sample.InitializeReservoir(reservoir.Count(), sample_count);

	// check if there are still elements remaining
	// this happens if we are on a boundary
	// for example, input.size() is 1024, but our sample size is 10
	if (required_count == chunk_count) {
		// we are done here
		return 0;
	}
	// we still need to process a part of the chunk
	// create a selection vector of the remaining elements
	SelectionVector sel(STANDARD_VECTOR_SIZE);
	for (idx_t i = required_count; i < chunk_count; i++) {
		sel.set_index(i - required_count, i);
	}
	// slice the input vector and continue
	input.Slice(sel, chunk_count - required_count);
	return input.size();
}

ReservoirSamplePercentage::ReservoirSamplePercentage(Allocator &allocator, double percentage, int64_t seed)
    : BlockingSample(seed), allocator(allocator), sample_percentage(percentage / 100.0), current_count(0),
      is_finalized(false) {
	reservoir_sample_size = idx_t(sample_percentage * RESERVOIR_THRESHOLD);
	current_sample = make_uniq<ReservoirSample>(allocator, reservoir_sample_size, random.NextRandomInteger());
}

void ReservoirSamplePercentage::AddToReservoir(DataChunk &input) {
	if (current_count + input.size() > RESERVOIR_THRESHOLD) {
		// we don't have enough space in our current reservoir
		// first check what we still need to append to the current sample
		idx_t append_to_current_sample_count = RESERVOIR_THRESHOLD - current_count;
		idx_t append_to_next_sample = input.size() - append_to_current_sample_count;
		if (append_to_current_sample_count > 0) {
			// we have elements remaining, first add them to the current sample
			input.Flatten();

			input.SetCardinality(append_to_current_sample_count);
			current_sample->AddToReservoir(input);
		}
		if (append_to_next_sample > 0) {
			// slice the input for the remainder
			SelectionVector sel(STANDARD_VECTOR_SIZE);
			for (idx_t i = 0; i < append_to_next_sample; i++) {
				sel.set_index(i, append_to_current_sample_count + i);
			}
			input.Slice(sel, append_to_next_sample);
		}
		// now our first sample is filled: append it to the set of finished samples
		finished_samples.push_back(std::move(current_sample));

		// allocate a new sample, and potentially add the remainder of the current input to that sample
		current_sample = make_uniq<ReservoirSample>(allocator, reservoir_sample_size, random.NextRandomInteger());
		if (append_to_next_sample > 0) {
			current_sample->AddToReservoir(input);
		}
		current_count = append_to_next_sample;
	} else {
		// we can just append to the current sample
		current_count += input.size();
		current_sample->AddToReservoir(input);
	}
}

unique_ptr<DataChunk> ReservoirSamplePercentage::GetChunk() {
	if (!is_finalized) {
		Finalize();
	}
	while (!finished_samples.empty()) {
		auto &front = finished_samples.front();
		auto chunk = front->GetChunk();
		if (chunk && chunk->size() > 0) {
			return chunk;
		}
		// move to the next sample
		finished_samples.erase(finished_samples.begin());
	}
	return nullptr;
}

void ReservoirSamplePercentage::Finalize() {
	// need to finalize the current sample, if any
	if (current_count > 0) {
		// create a new sample
		auto new_sample_size = idx_t(round(sample_percentage * current_count));
		auto new_sample = make_uniq<ReservoirSample>(allocator, new_sample_size, random.NextRandomInteger());
		while (true) {
			auto chunk = current_sample->GetChunk();
			if (!chunk || chunk->size() == 0) {
				break;
			}
			new_sample->AddToReservoir(*chunk);
		}
		finished_samples.push_back(std::move(new_sample));
	}
	is_finalized = true;
}

BaseReservoirSampling::BaseReservoirSampling(int64_t seed) : random(seed) {
	next_index = 0;
	min_threshold = 0;
	min_entry = 0;
	current_count = 0;
}

BaseReservoirSampling::BaseReservoirSampling() : BaseReservoirSampling(-1) {
}

void BaseReservoirSampling::InitializeReservoir(idx_t cur_size, idx_t sample_size) {
	//! 1: The first m items of V are inserted into R
	//! first we need to check if the reservoir already has "m" elements
	if (cur_size == sample_size) {
		//! 2. For each item vi ∈ R: Calculate a key ki = random(0, 1)
		//! we then define the threshold to enter the reservoir T_w as the minimum key of R
		//! we use a priority queue to extract the minimum key in O(1) time
		for (idx_t i = 0; i < sample_size; i++) {
			double k_i = random.NextRandom();
			reservoir_weights.push(std::make_pair(-k_i, i));
		}
		SetNextEntry();
	}
}

void BaseReservoirSampling::SetNextEntry() {
	//! 4. Let r = random(0, 1) and Xw = log(r) / log(T_w)
	auto &min_key = reservoir_weights.top();
	double t_w = -min_key.first;
	double r = random.NextRandom();
	double x_w = log(r) / log(t_w);
	//! 5. From the current item vc skip items until item vi , such that:
	//! 6. wc +wc+1 +···+wi−1 < Xw <= wc +wc+1 +···+wi−1 +wi
	//! since all our weights are 1 (uniform sampling), we can just determine the amount of elements to skip
	min_threshold = t_w;
	min_entry = min_key.second;
	next_index = MaxValue<idx_t>(1, idx_t(round(x_w)));
	current_count = 0;
}

void BaseReservoirSampling::ReplaceElement() {
	//! replace the entry in the reservoir
	//! pop the minimum entry
	reservoir_weights.pop();
	//! now update the reservoir
	//! 8. Let tw = Tw i , r2 = random(tw,1) and vi’s key: ki = (r2)1/wi
	//! 9. The new threshold Tw is the new minimum key of R
	//! we generate a random number between (min_threshold, 1)
	double r2 = random.NextRandom(min_threshold, 1);
	//! now we insert the new weight into the reservoir
	reservoir_weights.push(std::make_pair(-r2, min_entry));
	//! we update the min entry with the new min entry in the reservoir
	SetNextEntry();
}

} // namespace duckdb






namespace duckdb {

//===--------------------------------------------------------------------===//
// WindowAggregateState
//===--------------------------------------------------------------------===//

WindowAggregateState::WindowAggregateState(AggregateObject aggr, const LogicalType &result_type_p)
    : aggr(std::move(aggr)), result_type(result_type_p), state(aggr.function.state_size()),
      statev(Value::POINTER((idx_t)state.data())), statep(Value::POINTER((idx_t)state.data())) {
	statev.SetVectorType(VectorType::FLAT_VECTOR); // Prevent conversion of results to constants
}

WindowAggregateState::~WindowAggregateState() {
}

void WindowAggregateState::AggregateInit() {
	aggr.function.initialize(state.data());
}

void WindowAggregateState::AggegateFinal(Vector &result, idx_t rid) {
	AggregateInputData aggr_input_data(aggr.GetFunctionData(), Allocator::DefaultAllocator());
	aggr.function.finalize(statev, aggr_input_data, result, 1, rid);

	if (aggr.function.destructor) {
		aggr.function.destructor(statev, aggr_input_data, 1);
	}
}

void WindowAggregateState::Sink(DataChunk &payload_chunk, SelectionVector *filter_sel, idx_t filtered) {
}

void WindowAggregateState::Finalize() {
}

void WindowAggregateState::Compute(Vector &result, idx_t rid, idx_t start, idx_t end) {
}

//===--------------------------------------------------------------------===//
// WindowConstantAggregate
//===--------------------------------------------------------------------===//

WindowConstantAggregate::WindowConstantAggregate(AggregateObject aggr, const LogicalType &result_type,
                                                 const ValidityMask &partition_mask, const idx_t count)
    : WindowAggregateState(std::move(aggr), result_type), partition(0), row(0) {

	// Locate the partition boundaries
	idx_t start = 0;
	if (partition_mask.AllValid()) {
		partition_offsets.emplace_back(0);
	} else {
		idx_t entry_idx;
		idx_t shift;
		while (start < count) {
			partition_mask.GetEntryIndex(start, entry_idx, shift);

			//	If start is aligned with the start of a block,
			//	and the block is blank, then skip forward one block.
			const auto block = partition_mask.GetValidityEntry(entry_idx);
			if (partition_mask.NoneValid(block) && !shift) {
				start += ValidityMask::BITS_PER_VALUE;
				continue;
			}

			// Loop over the block
			for (; shift < ValidityMask::BITS_PER_VALUE && start < count; ++shift, ++start) {
				if (partition_mask.RowIsValid(block, shift)) {
					partition_offsets.emplace_back(start);
				}
			}
		}
	}

	//	Initialise the vector for caching the results
	results = make_uniq<Vector>(result_type, partition_offsets.size());
	partition_offsets.emplace_back(count);

	//	Start the first aggregate
	AggregateInit();
}

void WindowConstantAggregate::Sink(DataChunk &payload_chunk, SelectionVector *filter_sel, idx_t filtered) {
	const auto chunk_begin = row;
	const auto chunk_end = chunk_begin + payload_chunk.size();

	if (!inputs.ColumnCount() && payload_chunk.ColumnCount()) {
		inputs.Initialize(Allocator::DefaultAllocator(), payload_chunk.GetTypes());
	}

	AggregateInputData aggr_input_data(aggr.GetFunctionData(), Allocator::DefaultAllocator());
	idx_t begin = 0;
	idx_t filter_idx = 0;
	auto partition_end = partition_offsets[partition + 1];
	while (row < chunk_end) {
		if (row == partition_end) {
			AggegateFinal(*results, partition++);
			AggregateInit();
			partition_end = partition_offsets[partition + 1];
		}
		partition_end = MinValue(partition_end, chunk_end);
		auto end = partition_end - chunk_begin;

		inputs.Reset();
		if (filter_sel) {
			// 	Slice to any filtered rows in [begin, end)
			SelectionVector sel;

			//	Find the first value in [begin, end)
			for (; filter_idx < filtered; ++filter_idx) {
				auto idx = filter_sel->get_index(filter_idx);
				if (idx >= begin) {
					break;
				}
			}

			//	Find the first value in [end, filtered)
			sel.Initialize(filter_sel->data() + filter_idx);
			idx_t nsel = 0;
			for (; filter_idx < filtered; ++filter_idx, ++nsel) {
				auto idx = filter_sel->get_index(filter_idx);
				if (idx >= end) {
					break;
				}
			}

			if (nsel != inputs.size()) {
				inputs.Slice(payload_chunk, sel, nsel);
			}
		} else {
			//	Slice to [begin, end)
			if (begin) {
				for (idx_t c = 0; c < payload_chunk.ColumnCount(); ++c) {
					inputs.data[c].Slice(payload_chunk.data[c], begin, end);
				}
			} else {
				inputs.Reference(payload_chunk);
			}
			inputs.SetCardinality(end - begin);
		}

		//	Aggregate the filtered rows into a single state
		const auto count = inputs.size();
		if (aggr.function.simple_update) {
			aggr.function.simple_update(inputs.data.data(), aggr_input_data, inputs.ColumnCount(), state.data(), count);
		} else {
			aggr.function.update(inputs.data.data(), aggr_input_data, inputs.ColumnCount(), statep, count);
		}

		//	Skip filtered rows too!
		row += end - begin;
		begin = end;
	}
}

void WindowConstantAggregate::Finalize() {
	AggegateFinal(*results, partition++);

	partition = 0;
	row = 0;
}

void WindowConstantAggregate::Compute(Vector &target, idx_t rid, idx_t start, idx_t end) {
	//	Find the partition containing [start, end)
	while (start < partition_offsets[partition] || partition_offsets[partition + 1] <= start) {
		++partition;
	}
	D_ASSERT(partition_offsets[partition] <= start);
	D_ASSERT(partition + 1 < partition_offsets.size());
	D_ASSERT(end <= partition_offsets[partition + 1]);

	// Copy the value
	VectorOperations::Copy(*results, target, partition + 1, partition, rid);
}

//===--------------------------------------------------------------------===//
// WindowSegmentTree
//===--------------------------------------------------------------------===//
WindowSegmentTree::WindowSegmentTree(AggregateObject aggr, const LogicalType &result_type_p, DataChunk *input,
                                     const ValidityMask &filter_mask_p, WindowAggregationMode mode_p)
    : aggr(std::move(aggr)), result_type(result_type_p), state(aggr.function.state_size()),
      statep(Value::POINTER((idx_t)state.data())), frame(0, 0), statev(Value::POINTER((idx_t)state.data())),
      internal_nodes(0), input_ref(input), filter_mask(filter_mask_p), mode(mode_p) {
	statep.Flatten(input->size());
	statev.SetVectorType(VectorType::FLAT_VECTOR); // Prevent conversion of results to constants

	if (input_ref && input_ref->ColumnCount() > 0) {
		filter_sel.Initialize(input->size());
		inputs.Initialize(Allocator::DefaultAllocator(), input_ref->GetTypes());
		// if we have a frame-by-frame method, share the single state
		if (aggr.function.window && UseWindowAPI()) {
			AggregateInit();
			inputs.Reference(*input_ref);
		} else {
			inputs.SetCapacity(*input_ref);
			if (aggr.function.combine && UseCombineAPI()) {
				ConstructTree();
			}
		}
	}
}

WindowSegmentTree::~WindowSegmentTree() {
	if (!aggr.function.destructor) {
		// nothing to destroy
		return;
	}
	AggregateInputData aggr_input_data(aggr.GetFunctionData(), Allocator::DefaultAllocator());
	// call the destructor for all the intermediate states
	data_ptr_t address_data[STANDARD_VECTOR_SIZE];
	Vector addresses(LogicalType::POINTER, (data_ptr_t)address_data);
	idx_t count = 0;
	for (idx_t i = 0; i < internal_nodes; i++) {
		address_data[count++] = data_ptr_t(levels_flat_native.get() + i * state.size());
		if (count == STANDARD_VECTOR_SIZE) {
			aggr.function.destructor(addresses, aggr_input_data, count);
			count = 0;
		}
	}
	if (count > 0) {
		aggr.function.destructor(addresses, aggr_input_data, count);
	}

	if (aggr.function.window && UseWindowAPI()) {
		aggr.function.destructor(statev, aggr_input_data, 1);
	}
}

void WindowSegmentTree::AggregateInit() {
	aggr.function.initialize(state.data());
}

void WindowSegmentTree::AggegateFinal(Vector &result, idx_t rid) {
	AggregateInputData aggr_input_data(aggr.GetFunctionData(), Allocator::DefaultAllocator());
	aggr.function.finalize(statev, aggr_input_data, result, 1, rid);

	if (aggr.function.destructor) {
		aggr.function.destructor(statev, aggr_input_data, 1);
	}
}

void WindowSegmentTree::ExtractFrame(idx_t begin, idx_t end) {
	const auto size = end - begin;

	auto &chunk = *input_ref;
	const auto input_count = input_ref->ColumnCount();
	inputs.SetCardinality(size);
	for (idx_t i = 0; i < input_count; ++i) {
		auto &v = inputs.data[i];
		auto &vec = chunk.data[i];
		v.Slice(vec, begin, end);
		v.Verify(size);
	}

	// Slice to any filtered rows
	if (!filter_mask.AllValid()) {
		idx_t filtered = 0;
		for (idx_t i = begin; i < end; ++i) {
			if (filter_mask.RowIsValid(i)) {
				filter_sel.set_index(filtered++, i - begin);
			}
		}
		if (filtered != inputs.size()) {
			inputs.Slice(filter_sel, filtered);
		}
	}
}

void WindowSegmentTree::WindowSegmentValue(idx_t l_idx, idx_t begin, idx_t end) {
	D_ASSERT(begin <= end);
	if (begin == end || inputs.ColumnCount() == 0) {
		return;
	}

	const auto count = end - begin;
	Vector s(statep, 0, count);
	if (l_idx == 0) {
		ExtractFrame(begin, end);
		AggregateInputData aggr_input_data(aggr.GetFunctionData(), Allocator::DefaultAllocator());
		D_ASSERT(!inputs.data.empty());
		aggr.function.update(&inputs.data[0], aggr_input_data, input_ref->ColumnCount(), s, inputs.size());
	} else {
		// find out where the states begin
		data_ptr_t begin_ptr = levels_flat_native.get() + state.size() * (begin + levels_flat_start[l_idx - 1]);
		// set up a vector of pointers that point towards the set of states
		Vector v(LogicalType::POINTER, count);
		auto pdata = FlatVector::GetData<data_ptr_t>(v);
		for (idx_t i = 0; i < count; i++) {
			pdata[i] = begin_ptr + i * state.size();
		}
		v.Verify(count);
		AggregateInputData aggr_input_data(aggr.GetFunctionData(), Allocator::DefaultAllocator());
		aggr.function.combine(v, s, aggr_input_data, count);
	}
}

void WindowSegmentTree::ConstructTree() {
	D_ASSERT(input_ref);
	D_ASSERT(inputs.ColumnCount() > 0);

	// compute space required to store internal nodes of segment tree
	internal_nodes = 0;
	idx_t level_nodes = input_ref->size();
	do {
		level_nodes = (level_nodes + (TREE_FANOUT - 1)) / TREE_FANOUT;
		internal_nodes += level_nodes;
	} while (level_nodes > 1);
	levels_flat_native = make_unsafe_uniq_array<data_t>(internal_nodes * state.size());
	levels_flat_start.push_back(0);

	idx_t levels_flat_offset = 0;
	idx_t level_current = 0;
	// level 0 is data itself
	idx_t level_size;
	// iterate over the levels of the segment tree
	while ((level_size = (level_current == 0 ? input_ref->size()
	                                         : levels_flat_offset - levels_flat_start[level_current - 1])) > 1) {
		for (idx_t pos = 0; pos < level_size; pos += TREE_FANOUT) {
			// compute the aggregate for this entry in the segment tree
			AggregateInit();
			WindowSegmentValue(level_current, pos, MinValue(level_size, pos + TREE_FANOUT));

			memcpy(levels_flat_native.get() + (levels_flat_offset * state.size()), state.data(), state.size());

			levels_flat_offset++;
		}

		levels_flat_start.push_back(levels_flat_offset);
		level_current++;
	}

	// Corner case: single element in the window
	if (levels_flat_offset == 0) {
		aggr.function.initialize(levels_flat_native.get());
	}
}

void WindowSegmentTree::Compute(Vector &result, idx_t rid, idx_t begin, idx_t end) {
	D_ASSERT(input_ref);

	// If we have a window function, use that
	if (aggr.function.window && UseWindowAPI()) {
		// Frame boundaries
		auto prev = frame;
		frame = FrameBounds(begin, end);

		// Extract the range
		AggregateInputData aggr_input_data(aggr.GetFunctionData(), Allocator::DefaultAllocator());
		aggr.function.window(input_ref->data.data(), filter_mask, aggr_input_data, inputs.ColumnCount(), state.data(),
		                     frame, prev, result, rid, 0);
		return;
	}

	AggregateInit();

	// Aggregate everything at once if we can't combine states
	if (!aggr.function.combine || !UseCombineAPI()) {
		WindowSegmentValue(0, begin, end);
		AggegateFinal(result, rid);
		return;
	}

	for (idx_t l_idx = 0; l_idx < levels_flat_start.size() + 1; l_idx++) {
		idx_t parent_begin = begin / TREE_FANOUT;
		idx_t parent_end = end / TREE_FANOUT;
		if (parent_begin == parent_end) {
			WindowSegmentValue(l_idx, begin, end);
			break;
		}
		idx_t group_begin = parent_begin * TREE_FANOUT;
		if (begin != group_begin) {
			WindowSegmentValue(l_idx, begin, group_begin + TREE_FANOUT);
			parent_begin++;
		}
		idx_t group_end = parent_end * TREE_FANOUT;
		if (end != group_end) {
			WindowSegmentValue(l_idx, group_end, end);
		}
		begin = parent_begin;
		end = parent_end;
	}

	AggegateFinal(result, rid);
}

} // namespace duckdb





namespace duckdb {

struct BaseCountFunction {
	template <class STATE>
	static void Initialize(STATE *state) {
		*state = 0;
	}

	template <class STATE, class OP>
	static void Combine(const STATE &source, STATE *target, AggregateInputData &) {
		*target += source;
	}

	template <class T, class STATE>
	static void Finalize(Vector &result, AggregateInputData &, STATE *state, T *target, ValidityMask &mask, idx_t idx) {
		target[idx] = *state;
	}
};

struct CountStarFunction : public BaseCountFunction {
	template <class STATE, class OP>
	static void Operation(STATE *state, AggregateInputData &, idx_t idx) {
		*state += 1;
	}

	template <class STATE, class OP>
	static void ConstantOperation(STATE *state, AggregateInputData &, idx_t count) {
		*state += count;
	}

	template <typename RESULT_TYPE>
	static void Window(Vector inputs[], const ValidityMask &filter_mask, AggregateInputData &aggr_input_data,
	                   idx_t input_count, data_ptr_t state, const FrameBounds &frame, const FrameBounds &prev,
	                   Vector &result, idx_t rid, idx_t bias) {
		D_ASSERT(input_count == 0);
		auto data = FlatVector::GetData<RESULT_TYPE>(result);
		const auto begin = frame.first;
		const auto end = frame.second;
		// Slice to any filtered rows
		if (!filter_mask.AllValid()) {
			RESULT_TYPE filtered = 0;
			for (auto i = begin; i < end; ++i) {
				filtered += filter_mask.RowIsValid(i);
			}
			data[rid] = filtered;
		} else {
			data[rid] = end - begin;
		}
	}
};

struct CountFunction : public BaseCountFunction {
	template <class INPUT_TYPE, class STATE, class OP>
	static void Operation(STATE *state, AggregateInputData &, INPUT_TYPE *input, ValidityMask &mask, idx_t idx) {
		*state += 1;
	}

	template <class INPUT_TYPE, class STATE, class OP>
	static void ConstantOperation(STATE *state, AggregateInputData &, INPUT_TYPE *input, ValidityMask &mask,
	                              idx_t count) {
		*state += count;
	}

	static bool IgnoreNull() {
		return true;
	}
};

AggregateFunction CountFun::GetFunction() {
	auto fun = AggregateFunction::UnaryAggregate<int64_t, int64_t, int64_t, CountFunction>(
	    LogicalType(LogicalTypeId::ANY), LogicalType::BIGINT);
	fun.name = "count";
	fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	fun.order_dependent = AggregateOrderDependent::NOT_ORDER_DEPENDENT;
	return fun;
}

static void CountStarSerialize(FieldWriter &writer, const FunctionData *bind_data, const AggregateFunction &function) {
}

static unique_ptr<FunctionData> CountStarDeserialize(ClientContext &context, FieldReader &reader,
                                                     AggregateFunction &function) {
	return nullptr;
}

AggregateFunction CountStarFun::GetFunction() {
	auto fun = AggregateFunction::NullaryAggregate<int64_t, int64_t, CountStarFunction>(LogicalType::BIGINT);
	fun.name = "count_star";
	fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	fun.window = CountStarFunction::Window<int64_t>;
	// TODO is there a better way to set those?
	fun.serialize = CountStarSerialize;
	fun.deserialize = CountStarDeserialize;
	return fun;
}

unique_ptr<BaseStatistics> CountPropagateStats(ClientContext &context, BoundAggregateExpression &expr,
                                               AggregateStatisticsInput &input) {
	if (!expr.IsDistinct() && !input.child_stats[0].CanHaveNull()) {
		// count on a column without null values: use count star
		expr.function = CountStarFun::GetFunction();
		expr.function.name = "count_star";
		expr.children.clear();
	}
	return nullptr;
}

void CountFun::RegisterFunction(BuiltinFunctions &set) {
	AggregateFunction count_function = CountFun::GetFunction();
	count_function.statistics = CountPropagateStats;
	AggregateFunctionSet count("count");
	count.AddFunction(count_function);
	// the count function can also be called without arguments
	count_function.arguments.clear();
	count_function.statistics = nullptr;
	count_function.window = CountStarFunction::Window<int64_t>;
	count.AddFunction(count_function);
	set.AddFunction(count);
}

void CountStarFun::RegisterFunction(BuiltinFunctions &set) {
	AggregateFunctionSet count("count_star");
	count.AddFunction(CountStarFun::GetFunction());
	set.AddFunction(count);
}

} // namespace duckdb
