// See https://raw.githubusercontent.com/duckdb/duckdb/master/LICENSE for licensing information

#include "duckdb.hpp"
#include "duckdb-internal.hpp"
#ifndef DUCKDB_AMALGAMATION
#error header mismatch
#endif







namespace duckdb {

string QueryErrorContext::Format(const string &query, const string &error_message, int error_loc) {
	if (error_loc < 0 || size_t(error_loc) >= query.size()) {
		// no location in query provided
		return error_message;
	}
	idx_t error_location = idx_t(error_loc);
	// count the line numbers until the error location
	// and set the start position as the first character of that line
	idx_t start_pos = 0;
	idx_t line_number = 1;
	for (idx_t i = 0; i < error_location; i++) {
		if (StringUtil::CharacterIsNewline(query[i])) {
			line_number++;
			start_pos = i + 1;
		}
	}
	// now find either the next newline token after the query, or find the end of string
	// this is the initial end position
	idx_t end_pos = query.size();
	for (idx_t i = error_location; i < query.size(); i++) {
		if (StringUtil::CharacterIsNewline(query[i])) {
			end_pos = i;
			break;
		}
	}
	// now start scanning from the start pos
	// we want to figure out the start and end pos of what we are going to render
	// we want to render at most 80 characters in total, with the error_location located in the middle
	const char *buf = query.c_str() + start_pos;
	idx_t len = end_pos - start_pos;
	vector<idx_t> render_widths;
	vector<idx_t> positions;
	if (Utf8Proc::IsValid(buf, len)) {
		// for unicode awareness, we traverse the graphemes of the current line and keep track of their render widths
		// and of their position in the string
		for (idx_t cpos = 0; cpos < len;) {
			auto char_render_width = Utf8Proc::RenderWidth(buf, len, cpos);
			positions.push_back(cpos);
			render_widths.push_back(char_render_width);
			cpos = Utf8Proc::NextGraphemeCluster(buf, len, cpos);
		}
	} else { // LCOV_EXCL_START
		// invalid utf-8, we can't do much at this point
		// we just assume every character is a character, and every character has a render width of 1
		for (idx_t cpos = 0; cpos < len; cpos++) {
			positions.push_back(cpos);
			render_widths.push_back(1);
		}
	} // LCOV_EXCL_STOP
	// now we want to find the (unicode aware) start and end position
	idx_t epos = 0;
	// start by finding the error location inside the array
	for (idx_t i = 0; i < positions.size(); i++) {
		if (positions[i] >= (error_location - start_pos)) {
			epos = i;
			break;
		}
	}
	bool truncate_beginning = false;
	bool truncate_end = false;
	idx_t spos = 0;
	// now we iterate backwards from the error location
	// we show max 40 render width before the error location
	idx_t current_render_width = 0;
	for (idx_t i = epos; i > 0; i--) {
		current_render_width += render_widths[i];
		if (current_render_width >= 40) {
			truncate_beginning = true;
			start_pos = positions[i];
			spos = i;
			break;
		}
	}
	// now do the same, but going forward
	current_render_width = 0;
	for (idx_t i = epos; i < positions.size(); i++) {
		current_render_width += render_widths[i];
		if (current_render_width >= 40) {
			truncate_end = true;
			end_pos = positions[i];
			break;
		}
	}
	string line_indicator = "LINE " + to_string(line_number) + ": ";
	string begin_trunc = truncate_beginning ? "..." : "";
	string end_trunc = truncate_end ? "..." : "";

	// get the render width of the error indicator (i.e. how many spaces we need to insert before the ^)
	idx_t error_render_width = 0;
	for (idx_t i = spos; i < epos; i++) {
		error_render_width += render_widths[i];
	}
	error_render_width += line_indicator.size() + begin_trunc.size();

	// now first print the error message plus the current line (or a subset of the line)
	string result = error_message;
	result += "\n" + line_indicator + begin_trunc + query.substr(start_pos, end_pos - start_pos) + end_trunc;
	// print an arrow pointing at the error location
	result += "\n" + string(error_render_width, ' ') + "^";
	return result;
}

string QueryErrorContext::FormatErrorRecursive(const string &msg, vector<ExceptionFormatValue> &values) {
	string error_message = values.empty() ? msg : ExceptionFormatValue::Format(msg, values);
	if (!statement || query_location >= statement->query.size()) {
		// no statement provided or query location out of range
		return error_message;
	}
	return Format(statement->query, error_message, query_location);
}

} // namespace duckdb





namespace duckdb {

string RecursiveCTENode::ToString() const {
	string result;
	result += "(" + left->ToString() + ")";
	result += " UNION ";
	if (union_all) {
		result += " ALL ";
	}
	result += "(" + right->ToString() + ")";
	return result;
}

bool RecursiveCTENode::Equals(const QueryNode *other_p) const {
	if (!QueryNode::Equals(other_p)) {
		return false;
	}
	if (this == other_p) {
		return true;
	}
	auto &other = other_p->Cast<RecursiveCTENode>();

	if (other.union_all != union_all) {
		return false;
	}
	if (!left->Equals(other.left.get())) {
		return false;
	}
	if (!right->Equals(other.right.get())) {
		return false;
	}
	return true;
}

unique_ptr<QueryNode> RecursiveCTENode::Copy() const {
	auto result = make_uniq<RecursiveCTENode>();
	result->ctename = ctename;
	result->union_all = union_all;
	result->left = left->Copy();
	result->right = right->Copy();
	result->aliases = aliases;
	this->CopyProperties(*result);
	return std::move(result);
}

void RecursiveCTENode::Serialize(FieldWriter &writer) const {
	writer.WriteString(ctename);
	writer.WriteField<bool>(union_all);
	writer.WriteSerializable(*left);
	writer.WriteSerializable(*right);
	writer.WriteList<string>(aliases);
}

unique_ptr<QueryNode> RecursiveCTENode::Deserialize(FieldReader &reader) {
	auto result = make_uniq<RecursiveCTENode>();
	result->ctename = reader.ReadRequired<string>();
	result->union_all = reader.ReadRequired<bool>();
	result->left = reader.ReadRequiredSerializable<QueryNode>();
	result->right = reader.ReadRequiredSerializable<QueryNode>();
	result->aliases = reader.ReadRequiredList<string>();
	return std::move(result);
}

void RecursiveCTENode::FormatSerialize(FormatSerializer &serializer) const {
	QueryNode::FormatSerialize(serializer);
	serializer.WriteProperty("cte_name", ctename);
	serializer.WriteProperty("union_all", union_all);
	serializer.WriteProperty("left", *left);
	serializer.WriteProperty("right", *right);
	serializer.WriteProperty("aliases", aliases);
}

unique_ptr<QueryNode> RecursiveCTENode::FormatDeserialize(FormatDeserializer &deserializer) {
	auto result = make_uniq<RecursiveCTENode>();
	deserializer.ReadProperty("cte_name", result->ctename);
	deserializer.ReadProperty("union_all", result->union_all);
	deserializer.ReadProperty("left", result->left);
	deserializer.ReadProperty("right", result->right);
	deserializer.ReadProperty("aliases", result->aliases);
	return std::move(result);
}

} // namespace duckdb







namespace duckdb {

SelectNode::SelectNode()
    : QueryNode(QueryNodeType::SELECT_NODE), aggregate_handling(AggregateHandling::STANDARD_HANDLING) {
}

string SelectNode::ToString() const {
	string result;
	result = cte_map.ToString();
	result += "SELECT ";

	// search for a distinct modifier
	for (idx_t modifier_idx = 0; modifier_idx < modifiers.size(); modifier_idx++) {
		if (modifiers[modifier_idx]->type == ResultModifierType::DISTINCT_MODIFIER) {
			auto &distinct_modifier = (DistinctModifier &)*modifiers[modifier_idx];
			result += "DISTINCT ";
			if (!distinct_modifier.distinct_on_targets.empty()) {
				result += "ON (";
				for (idx_t k = 0; k < distinct_modifier.distinct_on_targets.size(); k++) {
					if (k > 0) {
						result += ", ";
					}
					result += distinct_modifier.distinct_on_targets[k]->ToString();
				}
				result += ") ";
			}
		}
	}
	for (idx_t i = 0; i < select_list.size(); i++) {
		if (i > 0) {
			result += ", ";
		}
		result += select_list[i]->ToString();
		if (!select_list[i]->alias.empty()) {
			result += StringUtil::Format(" AS %s", SQLIdentifier(select_list[i]->alias));
		}
	}
	if (from_table && from_table->type != TableReferenceType::EMPTY) {
		result += " FROM " + from_table->ToString();
	}
	if (where_clause) {
		result += " WHERE " + where_clause->ToString();
	}
	if (!groups.grouping_sets.empty()) {
		result += " GROUP BY ";
		// if we are dealing with multiple grouping sets, we have to add a few additional brackets
		bool grouping_sets = groups.grouping_sets.size() > 1;
		if (grouping_sets) {
			result += "GROUPING SETS (";
		}
		for (idx_t i = 0; i < groups.grouping_sets.size(); i++) {
			auto &grouping_set = groups.grouping_sets[i];
			if (i > 0) {
				result += ",";
			}
			if (grouping_set.empty()) {
				result += "()";
				continue;
			}
			if (grouping_sets) {
				result += "(";
			}
			bool first = true;
			for (auto &grp : grouping_set) {
				if (!first) {
					result += ", ";
				}
				result += groups.group_expressions[grp]->ToString();
				first = false;
			}
			if (grouping_sets) {
				result += ")";
			}
		}
		if (grouping_sets) {
			result += ")";
		}
	} else if (aggregate_handling == AggregateHandling::FORCE_AGGREGATES) {
		result += " GROUP BY ALL";
	}
	if (having) {
		result += " HAVING " + having->ToString();
	}
	if (qualify) {
		result += " QUALIFY " + qualify->ToString();
	}
	if (sample) {
		result += " USING SAMPLE ";
		result += sample->sample_size.ToString();
		if (sample->is_percentage) {
			result += "%";
		}
		result += " (" + EnumUtil::ToString(sample->method);
		if (sample->seed >= 0) {
			result += ", " + std::to_string(sample->seed);
		}
		result += ")";
	}
	return result + ResultModifiersToString();
}

bool SelectNode::Equals(const QueryNode *other_p) const {
	if (!QueryNode::Equals(other_p)) {
		return false;
	}
	if (this == other_p) {
		return true;
	}
	auto &other = other_p->Cast<SelectNode>();

	// SELECT
	if (!ExpressionUtil::ListEquals(select_list, other.select_list)) {
		return false;
	}
	// FROM
	if (from_table) {
		// we have a FROM clause, compare to the other one
		if (!from_table->Equals(other.from_table.get())) {
			return false;
		}
	} else if (other.from_table) {
		// we don't have a FROM clause, if the other statement has one they are
		// not equal
		return false;
	}
	// WHERE
	if (!BaseExpression::Equals(where_clause.get(), other.where_clause.get())) {
		return false;
	}
	// GROUP BY
	if (!ExpressionUtil::ListEquals(groups.group_expressions, other.groups.group_expressions)) {
		return false;
	}
	if (groups.grouping_sets != other.groups.grouping_sets) {
		return false;
	}
	if (!SampleOptions::Equals(sample.get(), other.sample.get())) {
		return false;
	}
	// HAVING
	if (!BaseExpression::Equals(having.get(), other.having.get())) {
		return false;
	}
	// QUALIFY
	if (!BaseExpression::Equals(qualify.get(), other.qualify.get())) {
		return false;
	}
	return true;
}

unique_ptr<QueryNode> SelectNode::Copy() const {
	auto result = make_uniq<SelectNode>();
	for (auto &child : select_list) {
		result->select_list.push_back(child->Copy());
	}
	result->from_table = from_table ? from_table->Copy() : nullptr;
	result->where_clause = where_clause ? where_clause->Copy() : nullptr;
	// groups
	for (auto &group : groups.group_expressions) {
		result->groups.group_expressions.push_back(group->Copy());
	}
	result->groups.grouping_sets = groups.grouping_sets;
	result->aggregate_handling = aggregate_handling;
	result->having = having ? having->Copy() : nullptr;
	result->qualify = qualify ? qualify->Copy() : nullptr;
	result->sample = sample ? sample->Copy() : nullptr;
	this->CopyProperties(*result);
	return std::move(result);
}

void SelectNode::Serialize(FieldWriter &writer) const {
	writer.WriteSerializableList(select_list);
	writer.WriteOptional(from_table);
	writer.WriteOptional(where_clause);
	writer.WriteSerializableList(groups.group_expressions);
	writer.WriteField<uint32_t>(groups.grouping_sets.size());
	auto &serializer = writer.GetSerializer();
	for (auto &grouping_set : groups.grouping_sets) {
		serializer.Write<idx_t>(grouping_set.size());
		for (auto &idx : grouping_set) {
			serializer.Write<idx_t>(idx);
		}
	}
	writer.WriteField<AggregateHandling>(aggregate_handling);
	writer.WriteOptional(having);
	writer.WriteOptional(sample);
	writer.WriteOptional(qualify);
}

void SelectNode::FormatSerialize(FormatSerializer &serializer) const {
	QueryNode::FormatSerialize(serializer);
	serializer.WriteProperty("select_list", select_list);
	serializer.WriteOptionalProperty("from_table", from_table);
	serializer.WriteOptionalProperty("where_clause", where_clause);
	serializer.WriteProperty("group_expressions", groups.group_expressions);
	serializer.WriteProperty("group_sets", groups.grouping_sets);
	serializer.WriteProperty("aggregate_handling", aggregate_handling);
	serializer.WriteOptionalProperty("having", having);
	serializer.WriteOptionalProperty("sample", sample);
	serializer.WriteOptionalProperty("qualify", qualify);
}

unique_ptr<QueryNode> SelectNode::FormatDeserialize(FormatDeserializer &deserializer) {
	auto result = make_uniq<SelectNode>();

	deserializer.ReadProperty("select_list", result->select_list);
	deserializer.ReadOptionalProperty("from_table", result->from_table);
	deserializer.ReadOptionalProperty("where_clause", result->where_clause);
	deserializer.ReadProperty("group_expressions", result->groups.group_expressions);
	deserializer.ReadProperty("group_sets", result->groups.grouping_sets);
	deserializer.ReadProperty("aggregate_handling", result->aggregate_handling);
	deserializer.ReadOptionalProperty("having", result->having);
	deserializer.ReadOptionalProperty("sample", result->sample);
	deserializer.ReadOptionalProperty("qualify", result->qualify);

	return std::move(result);
}

unique_ptr<QueryNode> SelectNode::Deserialize(FieldReader &reader) {
	auto result = make_uniq<SelectNode>();
	result->select_list = reader.ReadRequiredSerializableList<ParsedExpression>();
	result->from_table = reader.ReadOptional<TableRef>(nullptr);
	result->where_clause = reader.ReadOptional<ParsedExpression>(nullptr);
	result->groups.group_expressions = reader.ReadRequiredSerializableList<ParsedExpression>();

	auto grouping_set_count = reader.ReadRequired<uint32_t>();
	auto &source = reader.GetSource();
	for (idx_t set_idx = 0; set_idx < grouping_set_count; set_idx++) {
		auto set_entries = source.Read<idx_t>();
		GroupingSet grouping_set;
		for (idx_t i = 0; i < set_entries; i++) {
			grouping_set.insert(source.Read<idx_t>());
		}
		result->groups.grouping_sets.push_back(grouping_set);
	}

	result->aggregate_handling = reader.ReadRequired<AggregateHandling>();
	result->having = reader.ReadOptional<ParsedExpression>(nullptr);
	result->sample = reader.ReadOptional<SampleOptions>(nullptr);
	result->qualify = reader.ReadOptional<ParsedExpression>(nullptr);
	return std::move(result);
}

} // namespace duckdb






namespace duckdb {

string SetOperationNode::ToString() const {
	string result;
	result = cte_map.ToString();
	result += "(" + left->ToString() + ") ";
	bool is_distinct = false;
	for (idx_t modifier_idx = 0; modifier_idx < modifiers.size(); modifier_idx++) {
		if (modifiers[modifier_idx]->type == ResultModifierType::DISTINCT_MODIFIER) {
			is_distinct = true;
			break;
		}
	}

	switch (setop_type) {
	case SetOperationType::UNION:
		result += is_distinct ? "UNION" : "UNION ALL";
		break;
	case SetOperationType::UNION_BY_NAME:
		result += is_distinct ? "UNION BY NAME" : "UNION ALL BY NAME";
		break;
	case SetOperationType::EXCEPT:
		D_ASSERT(is_distinct);
		result += "EXCEPT";
		break;
	case SetOperationType::INTERSECT:
		D_ASSERT(is_distinct);
		result += "INTERSECT";
		break;
	default:
		throw InternalException("Unsupported set operation type");
	}
	result += " (" + right->ToString() + ")";
	return result + ResultModifiersToString();
}

bool SetOperationNode::Equals(const QueryNode *other_p) const {
	if (!QueryNode::Equals(other_p)) {
		return false;
	}
	if (this == other_p) {
		return true;
	}
	auto &other = other_p->Cast<SetOperationNode>();
	if (setop_type != other.setop_type) {
		return false;
	}
	if (!left->Equals(other.left.get())) {
		return false;
	}
	if (!right->Equals(other.right.get())) {
		return false;
	}
	return true;
}

unique_ptr<QueryNode> SetOperationNode::Copy() const {
	auto result = make_uniq<SetOperationNode>();
	result->setop_type = setop_type;
	result->left = left->Copy();
	result->right = right->Copy();
	this->CopyProperties(*result);
	return std::move(result);
}

void SetOperationNode::Serialize(FieldWriter &writer) const {
	writer.WriteField<SetOperationType>(setop_type);
	writer.WriteSerializable(*left);
	writer.WriteSerializable(*right);
}

unique_ptr<QueryNode> SetOperationNode::Deserialize(FieldReader &reader) {
	auto result = make_uniq<SetOperationNode>();
	result->setop_type = reader.ReadRequired<SetOperationType>();
	result->left = reader.ReadRequiredSerializable<QueryNode>();
	result->right = reader.ReadRequiredSerializable<QueryNode>();
	return std::move(result);
}

void SetOperationNode::FormatSerialize(duckdb::FormatSerializer &serializer) const {
	QueryNode::FormatSerialize(serializer);
	serializer.WriteProperty("set_op_type", setop_type);
	serializer.WriteProperty("left", *left);
	serializer.WriteProperty("right", *right);
}

unique_ptr<QueryNode> SetOperationNode::FormatDeserialize(duckdb::FormatDeserializer &deserializer) {
	auto result = make_uniq<SetOperationNode>();
	deserializer.ReadProperty("set_op_type", result->setop_type);
	deserializer.ReadProperty("left", result->left);
	deserializer.ReadProperty("right", result->right);
	return std::move(result);
}

} // namespace duckdb










namespace duckdb {

CommonTableExpressionMap::CommonTableExpressionMap() {
}

CommonTableExpressionMap CommonTableExpressionMap::Copy() const {
	CommonTableExpressionMap res;
	for (auto &kv : this->map) {
		auto kv_info = make_uniq<CommonTableExpressionInfo>();
		for (auto &al : kv.second->aliases) {
			kv_info->aliases.push_back(al);
		}
		kv_info->query = unique_ptr_cast<SQLStatement, SelectStatement>(kv.second->query->Copy());
		res.map[kv.first] = std::move(kv_info);
	}
	return res;
}

string CommonTableExpressionMap::ToString() const {
	if (map.empty()) {
		return string();
	}
	// check if there are any recursive CTEs
	bool has_recursive = false;
	for (auto &kv : map) {
		if (kv.second->query->node->type == QueryNodeType::RECURSIVE_CTE_NODE) {
			has_recursive = true;
			break;
		}
	}
	string result = "WITH ";
	if (has_recursive) {
		result += "RECURSIVE ";
	}
	bool first_cte = true;
	for (auto &kv : map) {
		if (!first_cte) {
			result += ", ";
		}
		auto &cte = *kv.second;
		result += KeywordHelper::WriteOptionallyQuoted(kv.first);
		if (!cte.aliases.empty()) {
			result += " (";
			for (idx_t k = 0; k < cte.aliases.size(); k++) {
				if (k > 0) {
					result += ", ";
				}
				result += KeywordHelper::WriteOptionallyQuoted(cte.aliases[k]);
			}
			result += ")";
		}
		result += " AS (";
		result += cte.query->ToString();
		result += ")";
		first_cte = false;
	}
	return result;
}

void CommonTableExpressionMap::FormatSerialize(FormatSerializer &serializer) const {
	serializer.WriteProperty("map", map);
}

CommonTableExpressionMap CommonTableExpressionMap::FormatDeserialize(FormatDeserializer &deserializer) {
	auto result = CommonTableExpressionMap();
	deserializer.ReadProperty("map", result.map);
	return result;
}

string QueryNode::ResultModifiersToString() const {
	string result;
	for (idx_t modifier_idx = 0; modifier_idx < modifiers.size(); modifier_idx++) {
		auto &modifier = *modifiers[modifier_idx];
		if (modifier.type == ResultModifierType::ORDER_MODIFIER) {
			auto &order_modifier = (OrderModifier &)modifier;
			result += " ORDER BY ";
			for (idx_t k = 0; k < order_modifier.orders.size(); k++) {
				if (k > 0) {
					result += ", ";
				}
				result += order_modifier.orders[k].ToString();
			}
		} else if (modifier.type == ResultModifierType::LIMIT_MODIFIER) {
			auto &limit_modifier = (LimitModifier &)modifier;
			if (limit_modifier.limit) {
				result += " LIMIT " + limit_modifier.limit->ToString();
			}
			if (limit_modifier.offset) {
				result += " OFFSET " + limit_modifier.offset->ToString();
			}
		} else if (modifier.type == ResultModifierType::LIMIT_PERCENT_MODIFIER) {
			auto &limit_p_modifier = (LimitPercentModifier &)modifier;
			if (limit_p_modifier.limit) {
				result += " LIMIT (" + limit_p_modifier.limit->ToString() + ") %";
			}
			if (limit_p_modifier.offset) {
				result += " OFFSET " + limit_p_modifier.offset->ToString();
			}
		}
	}
	return result;
}

bool QueryNode::Equals(const QueryNode *other) const {
	if (!other) {
		return false;
	}
	if (this == other) {
		return true;
	}
	if (other->type != this->type) {
		return false;
	}
	if (modifiers.size() != other->modifiers.size()) {
		return false;
	}
	for (idx_t i = 0; i < modifiers.size(); i++) {
		if (!modifiers[i]->Equals(other->modifiers[i].get())) {
			return false;
		}
	}
	// WITH clauses (CTEs)
	if (cte_map.map.size() != other->cte_map.map.size()) {
		return false;
	}
	for (auto &entry : cte_map.map) {
		auto other_entry = other->cte_map.map.find(entry.first);
		if (other_entry == other->cte_map.map.end()) {
			return false;
		}
		if (entry.second->aliases != other_entry->second->aliases) {
			return false;
		}
		if (!entry.second->query->Equals(other_entry->second->query.get())) {
			return false;
		}
	}
	return other->type == type;
}

void QueryNode::CopyProperties(QueryNode &other) const {
	for (auto &modifier : modifiers) {
		other.modifiers.push_back(modifier->Copy());
	}
	for (auto &kv : cte_map.map) {
		auto kv_info = make_uniq<CommonTableExpressionInfo>();
		for (auto &al : kv.second->aliases) {
			kv_info->aliases.push_back(al);
		}
		kv_info->query = unique_ptr_cast<SQLStatement, SelectStatement>(kv.second->query->Copy());
		other.cte_map.map[kv.first] = std::move(kv_info);
	}
}

void QueryNode::Serialize(Serializer &main_serializer) const {
	FieldWriter writer(main_serializer);
	writer.WriteField<QueryNodeType>(type);
	writer.WriteSerializableList(modifiers);
	// cte_map

	writer.WriteField<uint32_t>((uint32_t)cte_map.map.size());
	auto &serializer = writer.GetSerializer();
	for (auto &cte : cte_map.map) {
		serializer.WriteString(cte.first);
		serializer.WriteStringVector(cte.second->aliases);
		cte.second->query->Serialize(serializer);
	}
	Serialize(writer);

	writer.Finalize();
}

// Children should call the base method before their own.
void QueryNode::FormatSerialize(FormatSerializer &serializer) const {
	serializer.WriteProperty("type", type);
	serializer.WriteProperty("modifiers", modifiers);
	serializer.WriteProperty("cte_map", cte_map);
}

unique_ptr<QueryNode> QueryNode::FormatDeserialize(FormatDeserializer &deserializer) {

	auto type = deserializer.ReadProperty<QueryNodeType>("type");

	auto modifiers = deserializer.ReadProperty<vector<unique_ptr<ResultModifier>>>("modifiers");
	auto cte_map = deserializer.ReadProperty<CommonTableExpressionMap>("cte_map");

	unique_ptr<QueryNode> result;

	switch (type) {
	case QueryNodeType::SELECT_NODE:
		result = SelectNode::FormatDeserialize(deserializer);
		break;
	case QueryNodeType::SET_OPERATION_NODE:
		result = SetOperationNode::FormatDeserialize(deserializer);
		break;
	case QueryNodeType::RECURSIVE_CTE_NODE:
		result = RecursiveCTENode::FormatDeserialize(deserializer);
		break;
	default:
		throw SerializationException("Could not deserialize Query Node: unknown type!");
	}

	result->type = type;
	result->modifiers = std::move(modifiers);
	result->cte_map = std::move(cte_map);
	return result;
}

unique_ptr<QueryNode> QueryNode::Deserialize(Deserializer &main_source) {
	FieldReader reader(main_source);

	auto type = reader.ReadRequired<QueryNodeType>();
	auto modifiers = reader.ReadRequiredSerializableList<ResultModifier>();
	// cte_map
	auto cte_count = reader.ReadRequired<uint32_t>();
	auto &source = reader.GetSource();
	case_insensitive_map_t<unique_ptr<CommonTableExpressionInfo>> new_map;
	for (idx_t i = 0; i < cte_count; i++) {
		auto name = source.Read<string>();
		auto info = make_uniq<CommonTableExpressionInfo>();
		source.ReadStringVector(info->aliases);
		info->query = SelectStatement::Deserialize(source);
		new_map[name] = std::move(info);
	}
	unique_ptr<QueryNode> result;
	switch (type) {
	case QueryNodeType::SELECT_NODE:
		result = SelectNode::Deserialize(reader);
		break;
	case QueryNodeType::SET_OPERATION_NODE:
		result = SetOperationNode::Deserialize(reader);
		break;
	case QueryNodeType::RECURSIVE_CTE_NODE:
		result = RecursiveCTENode::Deserialize(reader);
		break;
	default:
		throw SerializationException("Could not deserialize Query Node: unknown type!");
	}
	result->modifiers = std::move(modifiers);
	result->cte_map.map = std::move(new_map);
	reader.Finalize();
	return result;
}

void QueryNode::AddDistinct() {
	// check if we already have a DISTINCT modifier
	for (idx_t modifier_idx = modifiers.size(); modifier_idx > 0; modifier_idx--) {
		auto &modifier = *modifiers[modifier_idx - 1];
		if (modifier.type == ResultModifierType::DISTINCT_MODIFIER) {
			auto &distinct_modifier = (DistinctModifier &)modifier;
			if (distinct_modifier.distinct_on_targets.empty()) {
				// we have a DISTINCT without an ON clause - this distinct does not need to be added
				return;
			}
		} else if (modifier.type == ResultModifierType::LIMIT_MODIFIER ||
		           modifier.type == ResultModifierType::LIMIT_PERCENT_MODIFIER) {
			// we encountered a LIMIT or LIMIT PERCENT - these change the result of DISTINCT, so we do need to push a
			// DISTINCT relation
			break;
		}
	}
	modifiers.push_back(make_uniq<DistinctModifier>());
}

} // namespace duckdb






namespace duckdb {

bool ResultModifier::Equals(const ResultModifier *other) const {
	if (!other) {
		return false;
	}
	return type == other->type;
}

void ResultModifier::Serialize(Serializer &serializer) const {
	FieldWriter writer(serializer);
	writer.WriteField<ResultModifierType>(type);
	Serialize(writer);
	writer.Finalize();
}

void ResultModifier::FormatSerialize(FormatSerializer &serializer) const {
	serializer.WriteProperty("type", type);
}

unique_ptr<ResultModifier> ResultModifier::FormatDeserialize(FormatDeserializer &deserializer) {
	auto type = deserializer.ReadProperty<ResultModifierType>("type");

	unique_ptr<ResultModifier> result;
	switch (type) {
	case ResultModifierType::LIMIT_MODIFIER:
		result = LimitModifier::FormatDeserialize(deserializer);
		break;
	case ResultModifierType::ORDER_MODIFIER:
		result = OrderModifier::FormatDeserialize(deserializer);
		break;
	case ResultModifierType::DISTINCT_MODIFIER:
		result = DistinctModifier::FormatDeserialize(deserializer);
		break;
	case ResultModifierType::LIMIT_PERCENT_MODIFIER:
		result = LimitPercentModifier::FormatDeserialize(deserializer);
		break;
	default:
		throw InternalException("Unrecognized ResultModifierType for Deserialization");
	}
	return result;
}

unique_ptr<ResultModifier> ResultModifier::Deserialize(Deserializer &source) {
	FieldReader reader(source);
	auto type = reader.ReadRequired<ResultModifierType>();

	unique_ptr<ResultModifier> result;
	switch (type) {
	case ResultModifierType::LIMIT_MODIFIER:
		result = LimitModifier::Deserialize(reader);
		break;
	case ResultModifierType::ORDER_MODIFIER:
		result = OrderModifier::Deserialize(reader);
		break;
	case ResultModifierType::DISTINCT_MODIFIER:
		result = DistinctModifier::Deserialize(reader);
		break;
	case ResultModifierType::LIMIT_PERCENT_MODIFIER:
		result = LimitPercentModifier::Deserialize(reader);
		break;
	default:
		throw InternalException("Unrecognized ResultModifierType for Deserialization");
	}
	reader.Finalize();
	return result;
}

bool LimitModifier::Equals(const ResultModifier *other_p) const {
	if (!ResultModifier::Equals(other_p)) {
		return false;
	}
	auto &other = (LimitModifier &)*other_p;
	if (!BaseExpression::Equals(limit.get(), other.limit.get())) {
		return false;
	}
	if (!BaseExpression::Equals(offset.get(), other.offset.get())) {
		return false;
	}
	return true;
}

unique_ptr<ResultModifier> LimitModifier::Copy() const {
	auto copy = make_uniq<LimitModifier>();
	if (limit) {
		copy->limit = limit->Copy();
	}
	if (offset) {
		copy->offset = offset->Copy();
	}
	return std::move(copy);
}

void LimitModifier::Serialize(FieldWriter &writer) const {
	writer.WriteOptional(limit);
	writer.WriteOptional(offset);
}

void LimitModifier::FormatSerialize(FormatSerializer &serializer) const {
	ResultModifier::FormatSerialize(serializer);
	serializer.WriteOptionalProperty("limit", limit);
	serializer.WriteOptionalProperty("offset", offset);
}

unique_ptr<ResultModifier> LimitModifier::FormatDeserialize(FormatDeserializer &deserializer) {
	auto mod = make_uniq<LimitModifier>();
	deserializer.ReadOptionalProperty("limit", mod->limit);
	deserializer.ReadOptionalProperty("offset", mod->offset);
	return std::move(mod);
}

unique_ptr<ResultModifier> LimitModifier::Deserialize(FieldReader &reader) {
	auto mod = make_uniq<LimitModifier>();
	mod->limit = reader.ReadOptional<ParsedExpression>(nullptr);
	mod->offset = reader.ReadOptional<ParsedExpression>(nullptr);
	return std::move(mod);
}

bool DistinctModifier::Equals(const ResultModifier *other_p) const {
	if (!ResultModifier::Equals(other_p)) {
		return false;
	}
	auto &other = (DistinctModifier &)*other_p;
	if (!ExpressionUtil::ListEquals(distinct_on_targets, other.distinct_on_targets)) {
		return false;
	}
	return true;
}

unique_ptr<ResultModifier> DistinctModifier::Copy() const {
	auto copy = make_uniq<DistinctModifier>();
	for (auto &expr : distinct_on_targets) {
		copy->distinct_on_targets.push_back(expr->Copy());
	}
	return std::move(copy);
}

void DistinctModifier::Serialize(FieldWriter &writer) const {
	writer.WriteSerializableList(distinct_on_targets);
}

void DistinctModifier::FormatSerialize(duckdb::FormatSerializer &serializer) const {
	ResultModifier::FormatSerialize(serializer);
	serializer.WriteProperty("distinct_on_targets", distinct_on_targets);
}

unique_ptr<ResultModifier> DistinctModifier::FormatDeserialize(FormatDeserializer &deserializer) {
	auto mod = make_uniq<DistinctModifier>();
	deserializer.ReadProperty("distinct_on_targets", mod->distinct_on_targets);
	return std::move(mod);
}

unique_ptr<ResultModifier> DistinctModifier::Deserialize(FieldReader &reader) {
	auto mod = make_uniq<DistinctModifier>();
	mod->distinct_on_targets = reader.ReadRequiredSerializableList<ParsedExpression>();
	return std::move(mod);
}

bool OrderModifier::Equals(const ResultModifier *other_p) const {
	if (!ResultModifier::Equals(other_p)) {
		return false;
	}
	auto &other = (OrderModifier &)*other_p;
	if (orders.size() != other.orders.size()) {
		return false;
	}
	for (idx_t i = 0; i < orders.size(); i++) {
		if (orders[i].type != other.orders[i].type) {
			return false;
		}
		if (!BaseExpression::Equals(orders[i].expression.get(), other.orders[i].expression.get())) {
			return false;
		}
	}
	return true;
}

unique_ptr<ResultModifier> OrderModifier::Copy() const {
	auto copy = make_uniq<OrderModifier>();
	for (auto &order : orders) {
		copy->orders.emplace_back(order.type, order.null_order, order.expression->Copy());
	}
	return std::move(copy);
}

string OrderByNode::ToString() const {
	auto str = expression->ToString();
	switch (type) {
	case OrderType::ASCENDING:
		str += " ASC";
		break;
	case OrderType::DESCENDING:
		str += " DESC";
		break;
	default:
		break;
	}

	switch (null_order) {
	case OrderByNullType::NULLS_FIRST:
		str += " NULLS FIRST";
		break;
	case OrderByNullType::NULLS_LAST:
		str += " NULLS LAST";
		break;
	default:
		break;
	}
	return str;
}

void OrderByNode::Serialize(Serializer &serializer) const {
	FieldWriter writer(serializer);
	writer.WriteField<OrderType>(type);
	writer.WriteField<OrderByNullType>(null_order);
	writer.WriteSerializable(*expression);
	writer.Finalize();
}

void OrderByNode::FormatSerialize(FormatSerializer &serializer) const {
	serializer.WriteProperty("type", type);
	serializer.WriteProperty("null_order", null_order);
	serializer.WriteProperty("expression", expression);
}

OrderByNode OrderByNode::FormatDeserialize(FormatDeserializer &deserializer) {
	auto type = deserializer.ReadProperty<OrderType>("type");
	auto null_order = deserializer.ReadProperty<OrderByNullType>("null_order");
	auto expression = deserializer.ReadProperty<unique_ptr<ParsedExpression>>("expression");
	return OrderByNode(type, null_order, std::move(expression));
}

OrderByNode OrderByNode::Deserialize(Deserializer &source) {
	FieldReader reader(source);
	auto type = reader.ReadRequired<OrderType>();
	auto null_order = reader.ReadRequired<OrderByNullType>();
	auto expression = reader.ReadRequiredSerializable<ParsedExpression>();
	reader.Finalize();
	return OrderByNode(type, null_order, std::move(expression));
}

void OrderModifier::Serialize(FieldWriter &writer) const {
	writer.WriteRegularSerializableList(orders);
}

void OrderModifier::FormatSerialize(FormatSerializer &serializer) const {
	ResultModifier::FormatSerialize(serializer);
	serializer.WriteProperty("orders", orders);
}
unique_ptr<ResultModifier> OrderModifier::FormatDeserialize(FormatDeserializer &deserializer) {
	auto mod = make_uniq<OrderModifier>();
	deserializer.ReadProperty("orders", mod->orders);
	return std::move(mod);
}

unique_ptr<ResultModifier> OrderModifier::Deserialize(FieldReader &reader) {
	auto mod = make_uniq<OrderModifier>();
	mod->orders = reader.ReadRequiredSerializableList<OrderByNode, OrderByNode>();
	return std::move(mod);
}

bool LimitPercentModifier::Equals(const ResultModifier *other_p) const {
	if (!ResultModifier::Equals(other_p)) {
		return false;
	}
	auto &other = (LimitPercentModifier &)*other_p;
	if (!BaseExpression::Equals(limit.get(), other.limit.get())) {
		return false;
	}
	if (!BaseExpression::Equals(offset.get(), other.offset.get())) {
		return false;
	}
	return true;
}

unique_ptr<ResultModifier> LimitPercentModifier::Copy() const {
	auto copy = make_uniq<LimitPercentModifier>();
	if (limit) {
		copy->limit = limit->Copy();
	}
	if (offset) {
		copy->offset = offset->Copy();
	}
	return std::move(copy);
}

void LimitPercentModifier::Serialize(FieldWriter &writer) const {
	writer.WriteOptional(limit);
	writer.WriteOptional(offset);
}

void LimitPercentModifier::FormatSerialize(FormatSerializer &serializer) const {
	ResultModifier::FormatSerialize(serializer);
	serializer.WriteOptionalProperty("limit", limit);
	serializer.WriteOptionalProperty("offset", offset);
}

unique_ptr<ResultModifier> LimitPercentModifier::Deserialize(FieldReader &reader) {
	auto mod = make_uniq<LimitPercentModifier>();
	mod->limit = reader.ReadOptional<ParsedExpression>(nullptr);
	mod->offset = reader.ReadOptional<ParsedExpression>(nullptr);
	return std::move(mod);
}

unique_ptr<ResultModifier> LimitPercentModifier::FormatDeserialize(FormatDeserializer &deserializer) {
	auto mod = make_uniq<LimitPercentModifier>();
	deserializer.ReadOptionalProperty("limit", mod->limit);
	deserializer.ReadOptionalProperty("offset", mod->offset);
	return std::move(mod);
}

} // namespace duckdb


namespace duckdb {

AlterStatement::AlterStatement() : SQLStatement(StatementType::ALTER_STATEMENT) {
}

AlterStatement::AlterStatement(const AlterStatement &other) : SQLStatement(other), info(other.info->Copy()) {
}

unique_ptr<SQLStatement> AlterStatement::Copy() const {
	return unique_ptr<AlterStatement>(new AlterStatement(*this));
}

} // namespace duckdb


namespace duckdb {

AttachStatement::AttachStatement() : SQLStatement(StatementType::ATTACH_STATEMENT) {
}

AttachStatement::AttachStatement(const AttachStatement &other) : SQLStatement(other), info(other.info->Copy()) {
}

unique_ptr<SQLStatement> AttachStatement::Copy() const {
	return unique_ptr<AttachStatement>(new AttachStatement(*this));
}

} // namespace duckdb


namespace duckdb {

CallStatement::CallStatement() : SQLStatement(StatementType::CALL_STATEMENT) {
}

CallStatement::CallStatement(const CallStatement &other) : SQLStatement(other), function(other.function->Copy()) {
}

unique_ptr<SQLStatement> CallStatement::Copy() const {
	return unique_ptr<CallStatement>(new CallStatement(*this));
}

} // namespace duckdb


namespace duckdb {

CopyStatement::CopyStatement() : SQLStatement(StatementType::COPY_STATEMENT), info(make_uniq<CopyInfo>()) {
}

CopyStatement::CopyStatement(const CopyStatement &other) : SQLStatement(other), info(other.info->Copy()) {
	if (other.select_statement) {
		select_statement = other.select_statement->Copy();
	}
}

string CopyStatement::CopyOptionsToString(const string &format,
                                          const case_insensitive_map_t<vector<Value>> &options) const {
	if (format.empty() && options.empty()) {
		return string();
	}
	string result;

	result += " (";
	if (!format.empty()) {
		result += " FORMAT ";
		result += format;
	}
	for (auto it = options.begin(); it != options.end(); it++) {
		if (!format.empty() || it != options.begin()) {
			result += ", ";
		}
		auto &name = it->first;
		auto &values = it->second;

		result += name + " ";
		if (values.empty()) {
			// Options like HEADER don't need an explicit value
			// just providing the name already sets it to true
		} else if (values.size() == 1) {
			result += values[0].ToSQLString();
		} else {
			result += "( ";
			for (idx_t i = 0; i < values.size(); i++) {
				if (i) {
					result += ", ";
				}
				result += values[i].ToSQLString();
			}
			result += " )";
		}
	}
	result += " )";
	return result;
}

// COPY table-name (c1, c2, ..)
string TablePart(const CopyInfo &info) {
	string result;

	if (!info.catalog.empty()) {
		result += KeywordHelper::WriteOptionallyQuoted(info.catalog) + ".";
	}
	if (!info.schema.empty()) {
		result += KeywordHelper::WriteOptionallyQuoted(info.schema) + ".";
	}
	D_ASSERT(!info.table.empty());
	result += KeywordHelper::WriteOptionallyQuoted(info.table);

	// (c1, c2, ..)
	if (!info.select_list.empty()) {
		result += " (";
		for (idx_t i = 0; i < info.select_list.size(); i++) {
			if (i > 0) {
				result += ", ";
			}
			result += KeywordHelper::WriteOptionallyQuoted(info.select_list[i]);
		}
		result += " )";
	}
	return result;
}

string CopyStatement::ToString() const {
	string result;

	result += "COPY ";
	if (info->is_from) {
		D_ASSERT(!select_statement);
		result += TablePart(*info);
		result += " FROM";
		result += StringUtil::Format(" %s", SQLString(info->file_path));
		result += CopyOptionsToString(info->format, info->options);
	} else {
		if (select_statement) {
			// COPY (select-node) TO ...
			result += "(" + select_statement->ToString() + ")";
		} else {
			result += TablePart(*info);
		}
		result += " TO ";
		result += StringUtil::Format("%s", SQLString(info->file_path));
		result += CopyOptionsToString(info->format, info->options);
	}
	return result;
}

unique_ptr<SQLStatement> CopyStatement::Copy() const {
	return unique_ptr<CopyStatement>(new CopyStatement(*this));
}

} // namespace duckdb


namespace duckdb {

CreateStatement::CreateStatement() : SQLStatement(StatementType::CREATE_STATEMENT) {
}

CreateStatement::CreateStatement(const CreateStatement &other) : SQLStatement(other), info(other.info->Copy()) {
}

unique_ptr<SQLStatement> CreateStatement::Copy() const {
	return unique_ptr<CreateStatement>(new CreateStatement(*this));
}

} // namespace duckdb



namespace duckdb {

DeleteStatement::DeleteStatement() : SQLStatement(StatementType::DELETE_STATEMENT) {
}

DeleteStatement::DeleteStatement(const DeleteStatement &other) : SQLStatement(other), table(other.table->Copy()) {
	if (other.condition) {
		condition = other.condition->Copy();
	}
	for (const auto &using_clause : other.using_clauses) {
		using_clauses.push_back(using_clause->Copy());
	}
	for (auto &expr : other.returning_list) {
		returning_list.emplace_back(expr->Copy());
	}
	cte_map = other.cte_map.Copy();
}

string DeleteStatement::ToString() const {
	string result;
	result = cte_map.ToString();
	result += "DELETE FROM ";
	result += table->ToString();
	if (!using_clauses.empty()) {
		result += " USING ";
		for (idx_t i = 0; i < using_clauses.size(); i++) {
			if (i > 0) {
				result += ", ";
			}
			result += using_clauses[i]->ToString();
		}
	}
	if (condition) {
		result += " WHERE " + condition->ToString();
	}

	if (!returning_list.empty()) {
		result += " RETURNING ";
		for (idx_t i = 0; i < returning_list.size(); i++) {
			if (i > 0) {
				result += ", ";
			}
			result += returning_list[i]->ToString();
		}
	}
	return result;
}

unique_ptr<SQLStatement> DeleteStatement::Copy() const {
	return unique_ptr<DeleteStatement>(new DeleteStatement(*this));
}

} // namespace duckdb


namespace duckdb {

DetachStatement::DetachStatement() : SQLStatement(StatementType::DETACH_STATEMENT) {
}

DetachStatement::DetachStatement(const DetachStatement &other) : SQLStatement(other), info(other.info->Copy()) {
}

unique_ptr<SQLStatement> DetachStatement::Copy() const {
	return unique_ptr<DetachStatement>(new DetachStatement(*this));
}

} // namespace duckdb


namespace duckdb {

DropStatement::DropStatement() : SQLStatement(StatementType::DROP_STATEMENT), info(make_uniq<DropInfo>()) {
}

DropStatement::DropStatement(const DropStatement &other) : SQLStatement(other), info(other.info->Copy()) {
}

unique_ptr<SQLStatement> DropStatement::Copy() const {
	return unique_ptr<DropStatement>(new DropStatement(*this));
}

} // namespace duckdb


namespace duckdb {

ExecuteStatement::ExecuteStatement() : SQLStatement(StatementType::EXECUTE_STATEMENT) {
}

ExecuteStatement::ExecuteStatement(const ExecuteStatement &other) : SQLStatement(other), name(other.name) {
	for (const auto &value : other.values) {
		values.push_back(value->Copy());
	}
}

unique_ptr<SQLStatement> ExecuteStatement::Copy() const {
	return unique_ptr<ExecuteStatement>(new ExecuteStatement(*this));
}

} // namespace duckdb


namespace duckdb {

ExplainStatement::ExplainStatement(unique_ptr<SQLStatement> stmt, ExplainType explain_type)
    : SQLStatement(StatementType::EXPLAIN_STATEMENT), stmt(std::move(stmt)), explain_type(explain_type) {
}

ExplainStatement::ExplainStatement(const ExplainStatement &other)
    : SQLStatement(other), stmt(other.stmt->Copy()), explain_type(other.explain_type) {
}

unique_ptr<SQLStatement> ExplainStatement::Copy() const {
	return unique_ptr<ExplainStatement>(new ExplainStatement(*this));
}

} // namespace duckdb


namespace duckdb {

ExportStatement::ExportStatement(unique_ptr<CopyInfo> info)
    : SQLStatement(StatementType::EXPORT_STATEMENT), info(std::move(info)) {
}

ExportStatement::ExportStatement(const ExportStatement &other) : SQLStatement(other), info(other.info->Copy()) {
}

unique_ptr<SQLStatement> ExportStatement::Copy() const {
	return unique_ptr<ExportStatement>(new ExportStatement(*this));
}

} // namespace duckdb


namespace duckdb {

ExtensionStatement::ExtensionStatement(ParserExtension extension_p, unique_ptr<ParserExtensionParseData> parse_data_p)
    : SQLStatement(StatementType::EXTENSION_STATEMENT), extension(std::move(extension_p)),
      parse_data(std::move(parse_data_p)) {
}

unique_ptr<SQLStatement> ExtensionStatement::Copy() const {
	return make_uniq<ExtensionStatement>(extension, parse_data->Copy());
}

} // namespace duckdb





namespace duckdb {

OnConflictInfo::OnConflictInfo() : action_type(OnConflictAction::THROW) {
}

OnConflictInfo::OnConflictInfo(const OnConflictInfo &other)
    : action_type(other.action_type), indexed_columns(other.indexed_columns) {
	if (other.set_info) {
		set_info = other.set_info->Copy();
	}
	if (other.condition) {
		condition = other.condition->Copy();
	}
}

unique_ptr<OnConflictInfo> OnConflictInfo::Copy() const {
	return unique_ptr<OnConflictInfo>(new OnConflictInfo(*this));
}

InsertStatement::InsertStatement()
    : SQLStatement(StatementType::INSERT_STATEMENT), schema(DEFAULT_SCHEMA), catalog(INVALID_CATALOG) {
}

InsertStatement::InsertStatement(const InsertStatement &other)
    : SQLStatement(other), select_statement(unique_ptr_cast<SQLStatement, SelectStatement>(
                               other.select_statement ? other.select_statement->Copy() : nullptr)),
      columns(other.columns), table(other.table), schema(other.schema), catalog(other.catalog),
      default_values(other.default_values), column_order(other.column_order) {
	cte_map = other.cte_map.Copy();
	for (auto &expr : other.returning_list) {
		returning_list.emplace_back(expr->Copy());
	}
	if (other.table_ref) {
		table_ref = other.table_ref->Copy();
	}
	if (other.on_conflict_info) {
		on_conflict_info = other.on_conflict_info->Copy();
	}
}

string InsertStatement::OnConflictActionToString(OnConflictAction action) {
	switch (action) {
	case OnConflictAction::NOTHING:
		return "DO NOTHING";
	case OnConflictAction::REPLACE:
	case OnConflictAction::UPDATE:
		return "DO UPDATE";
	case OnConflictAction::THROW:
		// Explicitly left empty, for ToString purposes
		return "";
	default: {
		throw NotImplementedException("type not implemented for OnConflictActionType");
	}
	}
}

string InsertStatement::ToString() const {
	bool or_replace_shorthand_set = false;
	string result;

	result = cte_map.ToString();
	result += "INSERT";
	if (on_conflict_info && on_conflict_info->action_type == OnConflictAction::REPLACE) {
		or_replace_shorthand_set = true;
		result += " OR REPLACE";
	}
	result += " INTO ";
	if (!catalog.empty()) {
		result += KeywordHelper::WriteOptionallyQuoted(catalog) + ".";
	}
	if (!schema.empty()) {
		result += KeywordHelper::WriteOptionallyQuoted(schema) + ".";
	}
	result += KeywordHelper::WriteOptionallyQuoted(table);
	// Write the (optional) alias of the insert target
	if (table_ref && !table_ref->alias.empty()) {
		result += StringUtil::Format(" AS %s", KeywordHelper::WriteOptionallyQuoted(table_ref->alias));
	}
	if (column_order == InsertColumnOrder::INSERT_BY_NAME) {
		result += " BY NAME";
	}
	if (!columns.empty()) {
		result += " (";
		for (idx_t i = 0; i < columns.size(); i++) {
			if (i > 0) {
				result += ", ";
			}
			result += KeywordHelper::WriteOptionallyQuoted(columns[i]);
		}
		result += " )";
	}
	result += " ";
	auto values_list = GetValuesList();
	if (values_list) {
		D_ASSERT(!default_values);
		values_list->alias = string();
		result += values_list->ToString();
	} else if (select_statement) {
		D_ASSERT(!default_values);
		result += select_statement->ToString();
	} else {
		D_ASSERT(default_values);
		result += "DEFAULT VALUES";
	}
	if (!or_replace_shorthand_set && on_conflict_info) {
		auto &conflict_info = *on_conflict_info;
		result += " ON CONFLICT ";
		// (optional) conflict target
		if (!conflict_info.indexed_columns.empty()) {
			result += "(";
			auto &columns = conflict_info.indexed_columns;
			for (auto it = columns.begin(); it != columns.end();) {
				result += StringUtil::Lower(*it);
				if (++it != columns.end()) {
					result += ", ";
				}
			}
			result += " )";
		}

		// (optional) where clause
		if (conflict_info.condition) {
			result += " WHERE " + conflict_info.condition->ToString();
		}
		result += " " + OnConflictActionToString(conflict_info.action_type);
		if (conflict_info.set_info) {
			D_ASSERT(conflict_info.action_type == OnConflictAction::UPDATE);
			result += " SET ";
			auto &set_info = *conflict_info.set_info;
			D_ASSERT(set_info.columns.size() == set_info.expressions.size());
			// SET <column_name> = <expression>
			for (idx_t i = 0; i < set_info.columns.size(); i++) {
				auto &column = set_info.columns[i];
				auto &expr = set_info.expressions[i];
				if (i) {
					result += ", ";
				}
				result += StringUtil::Lower(column) + " = " + expr->ToString();
			}
			// (optional) where clause
			if (set_info.condition) {
				result += " WHERE " + set_info.condition->ToString();
			}
		}
	}
	if (!returning_list.empty()) {
		result += " RETURNING ";
		for (idx_t i = 0; i < returning_list.size(); i++) {
			if (i > 0) {
				result += ", ";
			}
			result += returning_list[i]->ToString();
		}
	}
	return result;
}

unique_ptr<SQLStatement> InsertStatement::Copy() const {
	return unique_ptr<InsertStatement>(new InsertStatement(*this));
}

optional_ptr<ExpressionListRef> InsertStatement::GetValuesList() const {
	if (!select_statement) {
		return nullptr;
	}
	if (select_statement->node->type != QueryNodeType::SELECT_NODE) {
		return nullptr;
	}
	auto &node = select_statement->node->Cast<SelectNode>();
	if (node.where_clause || node.qualify || node.having) {
		return nullptr;
	}
	if (!node.cte_map.map.empty()) {
		return nullptr;
	}
	if (!node.groups.grouping_sets.empty()) {
		return nullptr;
	}
	if (node.aggregate_handling != AggregateHandling::STANDARD_HANDLING) {
		return nullptr;
	}
	if (node.select_list.size() != 1 || node.select_list[0]->type != ExpressionType::STAR) {
		return nullptr;
	}
	if (!node.from_table || node.from_table->type != TableReferenceType::EXPRESSION_LIST) {
		return nullptr;
	}
	return &node.from_table->Cast<ExpressionListRef>();
}

} // namespace duckdb


namespace duckdb {

LoadStatement::LoadStatement() : SQLStatement(StatementType::LOAD_STATEMENT) {
}

LoadStatement::LoadStatement(const LoadStatement &other) : SQLStatement(other), info(other.info->Copy()) {
}

unique_ptr<SQLStatement> LoadStatement::Copy() const {
	return unique_ptr<LoadStatement>(new LoadStatement(*this));
}

} // namespace duckdb


namespace duckdb {

MultiStatement::MultiStatement() : SQLStatement(StatementType::MULTI_STATEMENT) {
}

MultiStatement::MultiStatement(const MultiStatement &other) : SQLStatement(other) {
	for (auto &stmt : other.statements) {
		statements.push_back(stmt->Copy());
	}
}

unique_ptr<SQLStatement> MultiStatement::Copy() const {
	return unique_ptr<MultiStatement>(new MultiStatement(*this));
}

} // namespace duckdb


namespace duckdb {

PragmaStatement::PragmaStatement() : SQLStatement(StatementType::PRAGMA_STATEMENT), info(make_uniq<PragmaInfo>()) {
}

PragmaStatement::PragmaStatement(const PragmaStatement &other) : SQLStatement(other), info(other.info->Copy()) {
}

unique_ptr<SQLStatement> PragmaStatement::Copy() const {
	return unique_ptr<PragmaStatement>(new PragmaStatement(*this));
}

} // namespace duckdb


namespace duckdb {

PrepareStatement::PrepareStatement() : SQLStatement(StatementType::PREPARE_STATEMENT), statement(nullptr), name("") {
}

PrepareStatement::PrepareStatement(const PrepareStatement &other)
    : SQLStatement(other), statement(other.statement->Copy()), name(other.name) {
}

unique_ptr<SQLStatement> PrepareStatement::Copy() const {
	return unique_ptr<PrepareStatement>(new PrepareStatement(*this));
}

} // namespace duckdb


namespace duckdb {

RelationStatement::RelationStatement(shared_ptr<Relation> relation)
    : SQLStatement(StatementType::RELATION_STATEMENT), relation(std::move(relation)) {
}

unique_ptr<SQLStatement> RelationStatement::Copy() const {
	return unique_ptr<RelationStatement>(new RelationStatement(*this));
}

} // namespace duckdb






namespace duckdb {

SelectStatement::SelectStatement(const SelectStatement &other) : SQLStatement(other), node(other.node->Copy()) {
}

unique_ptr<SQLStatement> SelectStatement::Copy() const {
	return unique_ptr<SelectStatement>(new SelectStatement(*this));
}

void SelectStatement::Serialize(Serializer &serializer) const {
	node->Serialize(serializer);
}

void SelectStatement::FormatSerialize(FormatSerializer &serializer) const {
	serializer.WriteProperty("node", node);
}

unique_ptr<SelectStatement> SelectStatement::Deserialize(Deserializer &source) {
	auto result = make_uniq<SelectStatement>();
	result->node = QueryNode::Deserialize(source);
	return result;
}

unique_ptr<SelectStatement> SelectStatement::FormatDeserialize(FormatDeserializer &deserializer) {
	auto result = make_uniq<SelectStatement>();
	deserializer.ReadProperty("node", result->node);
	return result;
}

bool SelectStatement::Equals(const SQLStatement *other_p) const {
	if (type != other_p->type) {
		return false;
	}
	auto &other = other_p->Cast<SelectStatement>();
	return node->Equals(other.node.get());
}

string SelectStatement::ToString() const {
	return node->ToString();
}

} // namespace duckdb


namespace duckdb {

SetStatement::SetStatement(std::string name_p, SetScope scope_p, SetType type_p)
    : SQLStatement(StatementType::SET_STATEMENT), name(std::move(name_p)), scope(scope_p), set_type(type_p) {
}

unique_ptr<SQLStatement> SetStatement::Copy() const {
	return unique_ptr<SetStatement>(new SetStatement(*this));
}

// Set Variable

SetVariableStatement::SetVariableStatement(std::string name_p, Value value_p, SetScope scope_p)
    : SetStatement(std::move(name_p), scope_p, SetType::SET), value(std::move(value_p)) {
}

unique_ptr<SQLStatement> SetVariableStatement::Copy() const {
	return unique_ptr<SetVariableStatement>(new SetVariableStatement(*this));
}

// Reset Variable

ResetVariableStatement::ResetVariableStatement(std::string name_p, SetScope scope_p)
    : SetStatement(std::move(name_p), scope_p, SetType::RESET) {
}

} // namespace duckdb


namespace duckdb {

ShowStatement::ShowStatement() : SQLStatement(StatementType::SHOW_STATEMENT), info(make_uniq<ShowSelectInfo>()) {
}

ShowStatement::ShowStatement(const ShowStatement &other) : SQLStatement(other), info(other.info->Copy()) {
}

unique_ptr<SQLStatement> ShowStatement::Copy() const {
	return unique_ptr<ShowStatement>(new ShowStatement(*this));
}

} // namespace duckdb


namespace duckdb {

TransactionStatement::TransactionStatement(TransactionType type)
    : SQLStatement(StatementType::TRANSACTION_STATEMENT), info(make_uniq<TransactionInfo>(type)) {
}

TransactionStatement::TransactionStatement(const TransactionStatement &other)
    : SQLStatement(other), info(make_uniq<TransactionInfo>(other.info->type)) {
}

unique_ptr<SQLStatement> TransactionStatement::Copy() const {
	return unique_ptr<TransactionStatement>(new TransactionStatement(*this));
}

} // namespace duckdb



namespace duckdb {

UpdateSetInfo::UpdateSetInfo() {
}

UpdateSetInfo::UpdateSetInfo(const UpdateSetInfo &other) : columns(other.columns) {
	if (other.condition) {
		condition = other.condition->Copy();
	}
	for (auto &expr : other.expressions) {
		expressions.emplace_back(expr->Copy());
	}
}

unique_ptr<UpdateSetInfo> UpdateSetInfo::Copy() const {
	return unique_ptr<UpdateSetInfo>(new UpdateSetInfo(*this));
}

UpdateStatement::UpdateStatement() : SQLStatement(StatementType::UPDATE_STATEMENT) {
}

UpdateStatement::UpdateStatement(const UpdateStatement &other)
    : SQLStatement(other), table(other.table->Copy()), set_info(other.set_info->Copy()) {
	if (other.from_table) {
		from_table = other.from_table->Copy();
	}
	for (auto &expr : other.returning_list) {
		returning_list.emplace_back(expr->Copy());
	}
	cte_map = other.cte_map.Copy();
}

string UpdateStatement::ToString() const {
	D_ASSERT(set_info);
	auto &condition = set_info->condition;
	auto &columns = set_info->columns;
	auto &expressions = set_info->expressions;

	string result;
	result = cte_map.ToString();
	result += "UPDATE ";
	result += table->ToString();
	result += " SET ";
	D_ASSERT(columns.size() == expressions.size());
	for (idx_t i = 0; i < columns.size(); i++) {
		if (i > 0) {
			result += ", ";
		}
		result += KeywordHelper::WriteOptionallyQuoted(columns[i]);
		result += " = ";
		result += expressions[i]->ToString();
	}
	if (from_table) {
		result += " FROM " + from_table->ToString();
	}
	if (condition) {
		result += " WHERE " + condition->ToString();
	}
	if (!returning_list.empty()) {
		result += " RETURNING ";
		for (idx_t i = 0; i < returning_list.size(); i++) {
			if (i > 0) {
				result += ", ";
			}
			result += returning_list[i]->ToString();
		}
	}
	return result;
}

unique_ptr<SQLStatement> UpdateStatement::Copy() const {
	return unique_ptr<UpdateStatement>(new UpdateStatement(*this));
}

} // namespace duckdb


namespace duckdb {

VacuumStatement::VacuumStatement(const VacuumOptions &options)
    : SQLStatement(StatementType::VACUUM_STATEMENT), info(make_uniq<VacuumInfo>(options)) {
}

VacuumStatement::VacuumStatement(const VacuumStatement &other) : SQLStatement(other), info(other.info->Copy()) {
}

unique_ptr<SQLStatement> VacuumStatement::Copy() const {
	return unique_ptr<VacuumStatement>(new VacuumStatement(*this));
}

} // namespace duckdb







namespace duckdb {

string BaseTableRef::ToString() const {
	string result;
	result += catalog_name.empty() ? "" : (KeywordHelper::WriteOptionallyQuoted(catalog_name) + ".");
	result += schema_name.empty() ? "" : (KeywordHelper::WriteOptionallyQuoted(schema_name) + ".");
	result += KeywordHelper::WriteOptionallyQuoted(table_name);
	return BaseToString(result, column_name_alias);
}

bool BaseTableRef::Equals(const TableRef *other_p) const {
	if (!TableRef::Equals(other_p)) {
		return false;
	}
	auto other = (BaseTableRef *)other_p;
	return other->catalog_name == catalog_name && other->schema_name == schema_name &&
	       other->table_name == table_name && column_name_alias == other->column_name_alias;
}

void BaseTableRef::Serialize(FieldWriter &writer) const {
	writer.WriteString(schema_name);
	writer.WriteString(table_name);
	writer.WriteList<string>(column_name_alias);
	writer.WriteString(catalog_name);
}

void BaseTableRef::FormatSerialize(FormatSerializer &serializer) const {
	TableRef::FormatSerialize(serializer);
	serializer.WriteProperty("schema_name", schema_name);
	serializer.WriteProperty("table_name", table_name);
	serializer.WriteProperty("column_name_alias", column_name_alias);
	serializer.WriteProperty("catalog_name", catalog_name);
}

unique_ptr<TableRef> BaseTableRef::FormatDeserialize(FormatDeserializer &deserializer) {
	auto result = make_uniq<BaseTableRef>();

	deserializer.ReadProperty("schema_name", result->schema_name);
	deserializer.ReadProperty("table_name", result->table_name);
	deserializer.ReadProperty("column_name_alias", result->column_name_alias);
	deserializer.ReadProperty("catalog_name", result->catalog_name);

	return std::move(result);
}

unique_ptr<TableRef> BaseTableRef::Deserialize(FieldReader &reader) {
	auto result = make_uniq<BaseTableRef>();

	result->schema_name = reader.ReadRequired<string>();
	result->table_name = reader.ReadRequired<string>();
	result->column_name_alias = reader.ReadRequiredList<string>();
	result->catalog_name = reader.ReadField<string>(INVALID_CATALOG);

	return std::move(result);
}

unique_ptr<TableRef> BaseTableRef::Copy() {
	auto copy = make_uniq<BaseTableRef>();

	copy->catalog_name = catalog_name;
	copy->schema_name = schema_name;
	copy->table_name = table_name;
	copy->column_name_alias = column_name_alias;
	CopyProperties(*copy);

	return std::move(copy);
}
} // namespace duckdb




namespace duckdb {

string EmptyTableRef::ToString() const {
	return "";
}

bool EmptyTableRef::Equals(const TableRef *other) const {
	return TableRef::Equals(other);
}

unique_ptr<TableRef> EmptyTableRef::Copy() {
	return make_uniq<EmptyTableRef>();
}

void EmptyTableRef::Serialize(FieldWriter &writer) const {
}

unique_ptr<TableRef> EmptyTableRef::Deserialize(FieldReader &reader) {
	return make_uniq<EmptyTableRef>();
}

unique_ptr<TableRef> EmptyTableRef::FormatDeserialize(FormatDeserializer &source) {
	return make_uniq<EmptyTableRef>();
}

} // namespace duckdb






namespace duckdb {

string ExpressionListRef::ToString() const {
	D_ASSERT(!values.empty());
	string result = "(VALUES ";
	for (idx_t row_idx = 0; row_idx < values.size(); row_idx++) {
		if (row_idx > 0) {
			result += ", ";
		}
		auto &row = values[row_idx];
		result += "(";
		for (idx_t col_idx = 0; col_idx < row.size(); col_idx++) {
			if (col_idx > 0) {
				result += ", ";
			}
			result += row[col_idx]->ToString();
		}
		result += ")";
	}
	result += ")";
	return BaseToString(result, expected_names);
}

bool ExpressionListRef::Equals(const TableRef *other_p) const {
	if (!TableRef::Equals(other_p)) {
		return false;
	}
	auto other = (ExpressionListRef *)other_p;
	if (values.size() != other->values.size()) {
		return false;
	}
	for (idx_t i = 0; i < values.size(); i++) {
		if (values[i].size() != other->values[i].size()) {
			return false;
		}
		for (idx_t j = 0; j < values[i].size(); j++) {
			if (!values[i][j]->Equals(other->values[i][j].get())) {
				return false;
			}
		}
	}
	return true;
}

unique_ptr<TableRef> ExpressionListRef::Copy() {
	// value list
	auto result = make_uniq<ExpressionListRef>();
	for (auto &val_list : values) {
		vector<unique_ptr<ParsedExpression>> new_val_list;
		new_val_list.reserve(val_list.size());
		for (auto &val : val_list) {
			new_val_list.push_back(val->Copy());
		}
		result->values.push_back(std::move(new_val_list));
	}
	result->expected_names = expected_names;
	result->expected_types = expected_types;
	CopyProperties(*result);
	return std::move(result);
}

void ExpressionListRef::Serialize(FieldWriter &writer) const {
	writer.WriteList<string>(expected_names);
	writer.WriteRegularSerializableList<LogicalType>(expected_types);
	auto &serializer = writer.GetSerializer();
	writer.WriteField<uint32_t>(values.size());
	for (idx_t i = 0; i < values.size(); i++) {
		serializer.WriteList(values[i]);
	}
}

void ExpressionListRef::FormatSerialize(FormatSerializer &serializer) const {
	TableRef::FormatSerialize(serializer);
	serializer.WriteProperty("expected_names", expected_names);
	serializer.WriteProperty("expected_types", expected_types);
	serializer.WriteProperty("values", values);
}

unique_ptr<TableRef> ExpressionListRef::FormatDeserialize(FormatDeserializer &source) {
	auto result = make_uniq<ExpressionListRef>();
	source.ReadProperty("expected_names", result->expected_names);
	source.ReadProperty("expected_types", result->expected_types);
	source.ReadProperty("values", result->values);
	return std::move(result);
}

unique_ptr<TableRef> ExpressionListRef::Deserialize(FieldReader &reader) {
	auto result = make_uniq<ExpressionListRef>();
	// value list
	result->expected_names = reader.ReadRequiredList<string>();
	result->expected_types = reader.ReadRequiredSerializableList<LogicalType, LogicalType>();
	idx_t value_list_size = reader.ReadRequired<uint32_t>();
	auto &source = reader.GetSource();
	for (idx_t i = 0; i < value_list_size; i++) {
		vector<unique_ptr<ParsedExpression>> value_list;
		source.ReadList<ParsedExpression>(value_list);
		result->values.push_back(std::move(value_list));
	}
	return std::move(result);
}

} // namespace duckdb







namespace duckdb {

string JoinRef::ToString() const {
	string result;
	result = left->ToString() + " ";
	switch (ref_type) {
	case JoinRefType::REGULAR:
		result += EnumUtil::ToString(type) + " JOIN ";
		break;
	case JoinRefType::NATURAL:
		result += "NATURAL ";
		result += EnumUtil::ToString(type) + " JOIN ";
		break;
	case JoinRefType::ASOF:
		result += "ASOF ";
		result += EnumUtil::ToString(type) + " JOIN ";
		break;
	case JoinRefType::CROSS:
		result += ", ";
		break;
	case JoinRefType::POSITIONAL:
		result += "POSITIONAL JOIN ";
		break;
	}
	result += right->ToString();
	if (condition) {
		D_ASSERT(using_columns.empty());
		result += " ON (";
		result += condition->ToString();
		result += ")";
	} else if (!using_columns.empty()) {
		result += " USING (";
		for (idx_t i = 0; i < using_columns.size(); i++) {
			if (i > 0) {
				result += ", ";
			}
			result += using_columns[i];
		}
		result += ")";
	}
	return result;
}

bool JoinRef::Equals(const TableRef *other_p) const {
	if (!TableRef::Equals(other_p)) {
		return false;
	}
	auto other = (JoinRef *)other_p;
	if (using_columns.size() != other->using_columns.size()) {
		return false;
	}
	for (idx_t i = 0; i < using_columns.size(); i++) {
		if (using_columns[i] != other->using_columns[i]) {
			return false;
		}
	}
	return left->Equals(other->left.get()) && right->Equals(other->right.get()) &&
	       BaseExpression::Equals(condition.get(), other->condition.get()) && type == other->type;
}

unique_ptr<TableRef> JoinRef::Copy() {
	auto copy = make_uniq<JoinRef>(ref_type);
	copy->left = left->Copy();
	copy->right = right->Copy();
	if (condition) {
		copy->condition = condition->Copy();
	}
	copy->type = type;
	copy->ref_type = ref_type;
	copy->alias = alias;
	copy->using_columns = using_columns;
	return std::move(copy);
}

void JoinRef::Serialize(FieldWriter &writer) const {
	writer.WriteSerializable(*left);
	writer.WriteSerializable(*right);
	writer.WriteOptional(condition);
	writer.WriteField<JoinType>(type);
	writer.WriteField<JoinRefType>(ref_type);
	writer.WriteList<string>(using_columns);
}

void JoinRef::FormatSerialize(FormatSerializer &serializer) const {
	TableRef::FormatSerialize(serializer);
	serializer.WriteProperty("left", *left);
	serializer.WriteProperty("right", *right);
	serializer.WriteOptionalProperty("condition", condition);
	serializer.WriteProperty("join_type", type);
	serializer.WriteProperty("ref_type", ref_type);
	serializer.WriteProperty("using_columns", using_columns);
}

unique_ptr<TableRef> JoinRef::FormatDeserialize(FormatDeserializer &source) {
	auto result = make_uniq<JoinRef>(JoinRefType::REGULAR);

	source.ReadProperty("left", result->left);
	source.ReadProperty("right", result->right);
	source.ReadOptionalProperty("condition", result->condition);
	source.ReadProperty("join_type", result->type);
	source.ReadProperty("ref_type", result->ref_type);
	source.ReadProperty("using_columns", result->using_columns);

	return std::move(result);
}

unique_ptr<TableRef> JoinRef::Deserialize(FieldReader &reader) {
	auto result = make_uniq<JoinRef>(JoinRefType::REGULAR);
	result->left = reader.ReadRequiredSerializable<TableRef>();
	result->right = reader.ReadRequiredSerializable<TableRef>();
	result->condition = reader.ReadOptional<ParsedExpression>(nullptr);
	result->type = reader.ReadRequired<JoinType>();
	result->ref_type = reader.ReadRequired<JoinRefType>();
	result->using_columns = reader.ReadRequiredList<string>();
	return std::move(result);
}

} // namespace duckdb







namespace duckdb {

//===--------------------------------------------------------------------===//
// PivotColumn
//===--------------------------------------------------------------------===//
string PivotColumn::ToString() const {
	string result;
	if (!unpivot_names.empty()) {
		D_ASSERT(pivot_expressions.empty());
		// unpivot
		if (unpivot_names.size() == 1) {
			result += KeywordHelper::WriteOptionallyQuoted(unpivot_names[0]);
		} else {
			result += "(";
			for (idx_t n = 0; n < unpivot_names.size(); n++) {
				if (n > 0) {
					result += ", ";
				}
				result += KeywordHelper::WriteOptionallyQuoted(unpivot_names[n]);
			}
			result += ")";
		}
	} else if (!pivot_expressions.empty()) {
		// pivot
		result += "(";
		for (idx_t n = 0; n < pivot_expressions.size(); n++) {
			if (n > 0) {
				result += ", ";
			}
			result += pivot_expressions[n]->ToString();
		}
		result += ")";
	}
	result += " IN ";
	if (pivot_enum.empty()) {
		result += "(";
		for (idx_t e = 0; e < entries.size(); e++) {
			auto &entry = entries[e];
			if (e > 0) {
				result += ", ";
			}
			if (entry.star_expr) {
				D_ASSERT(entry.values.empty());
				result += entry.star_expr->ToString();
			} else if (entry.values.size() == 1) {
				result += entry.values[0].ToSQLString();
			} else {
				result += "(";
				for (idx_t v = 0; v < entry.values.size(); v++) {
					if (v > 0) {
						result += ", ";
					}
					result += entry.values[v].ToSQLString();
				}
				result += ")";
			}
			if (!entry.alias.empty()) {
				result += " AS " + KeywordHelper::WriteOptionallyQuoted(entry.alias);
			}
		}
		result += ")";
	} else {
		result += KeywordHelper::WriteOptionallyQuoted(pivot_enum);
	}
	return result;
}

bool PivotColumnEntry::Equals(const PivotColumnEntry &other) const {
	if (alias != other.alias) {
		return false;
	}
	if (values.size() != other.values.size()) {
		return false;
	}
	for (idx_t i = 0; i < values.size(); i++) {
		if (!Value::NotDistinctFrom(values[i], other.values[i])) {
			return false;
		}
	}
	return true;
}

bool PivotColumn::Equals(const PivotColumn &other) const {
	if (!ExpressionUtil::ListEquals(pivot_expressions, other.pivot_expressions)) {
		return false;
	}
	if (other.unpivot_names != unpivot_names) {
		return false;
	}
	if (other.pivot_enum != pivot_enum) {
		return false;
	}
	if (other.entries.size() != entries.size()) {
		return false;
	}
	for (idx_t i = 0; i < entries.size(); i++) {
		if (!entries[i].Equals(other.entries[i])) {
			return false;
		}
	}
	return true;
}

PivotColumn PivotColumn::Copy() const {
	PivotColumn result;
	for (auto &expr : pivot_expressions) {
		result.pivot_expressions.push_back(expr->Copy());
	}
	result.unpivot_names = unpivot_names;
	for (auto &entry : entries) {
		result.entries.push_back(entry.Copy());
	}
	result.pivot_enum = pivot_enum;
	return result;
}

void PivotColumn::Serialize(Serializer &serializer) const {
	FieldWriter writer(serializer);
	writer.WriteSerializableList(pivot_expressions);
	writer.WriteList<string>(unpivot_names);
	writer.WriteRegularSerializableList(entries);
	writer.WriteString(pivot_enum);
	writer.Finalize();
}

void PivotColumn::FormatSerialize(FormatSerializer &serializer) const {
	serializer.WriteProperty("pivot_expressions", pivot_expressions);
	serializer.WriteProperty("unpivot_names", unpivot_names);
	serializer.WriteProperty("entries", entries);
	serializer.WriteProperty("pivot_enum", pivot_enum);
}

PivotColumn PivotColumn::Deserialize(Deserializer &source) {
	PivotColumn result;
	FieldReader reader(source);
	result.pivot_expressions = reader.ReadRequiredSerializableList<ParsedExpression>();
	result.unpivot_names = reader.ReadRequiredList<string>();
	result.entries = reader.ReadRequiredSerializableList<PivotColumnEntry, PivotColumnEntry>();
	result.pivot_enum = reader.ReadRequired<string>();
	reader.Finalize();
	return result;
}

PivotColumn PivotColumn::FormatDeserialize(FormatDeserializer &source) {
	PivotColumn result;
	source.ReadProperty("pivot_expressions", result.pivot_expressions);
	source.ReadProperty("unpivot_names", result.unpivot_names);
	source.ReadProperty("entries", result.entries);
	source.ReadProperty("pivot_enum", result.pivot_enum);
	return result;
}

//===--------------------------------------------------------------------===//
// PivotColumnEntry
//===--------------------------------------------------------------------===//
PivotColumnEntry PivotColumnEntry::Copy() const {
	PivotColumnEntry result;
	result.values = values;
	result.star_expr = star_expr ? star_expr->Copy() : nullptr;
	result.alias = alias;
	return result;
}

void PivotColumnEntry::Serialize(Serializer &serializer) const {
	FieldWriter writer(serializer);
	writer.WriteRegularSerializableList(values);
	writer.WriteOptional(star_expr);
	writer.WriteString(alias);
	writer.Finalize();
}

void PivotColumnEntry::FormatSerialize(FormatSerializer &serializer) const {
	serializer.WriteProperty("values", values);
	serializer.WriteOptionalProperty("star_expr", star_expr);
	serializer.WriteProperty("alias", alias);
}

PivotColumnEntry PivotColumnEntry::Deserialize(Deserializer &source) {
	PivotColumnEntry result;
	FieldReader reader(source);
	result.values = reader.ReadRequiredSerializableList<Value, Value>();
	result.star_expr = reader.ReadOptional<ParsedExpression>(nullptr);
	result.alias = reader.ReadRequired<string>();
	reader.Finalize();
	return result;
}

PivotColumnEntry PivotColumnEntry::FormatDeserialize(FormatDeserializer &source) {
	PivotColumnEntry result;
	source.ReadProperty("values", result.values);
	source.ReadOptionalProperty("star_expr", result.star_expr);
	source.ReadProperty("alias", result.alias);
	return result;
}

//===--------------------------------------------------------------------===//
// PivotRef
//===--------------------------------------------------------------------===//
string PivotRef::ToString() const {
	string result;
	result = source->ToString();
	if (!aggregates.empty()) {
		// pivot
		result += " PIVOT (";
		for (idx_t aggr_idx = 0; aggr_idx < aggregates.size(); aggr_idx++) {
			if (aggr_idx > 0) {
				result += ", ";
			}
			result += aggregates[aggr_idx]->ToString();
			if (!aggregates[aggr_idx]->alias.empty()) {
				result += " AS " + KeywordHelper::WriteOptionallyQuoted(aggregates[aggr_idx]->alias);
			}
		}
	} else {
		// unpivot
		result += " UNPIVOT ";
		if (include_nulls) {
			result += "INCLUDE NULLS ";
		}
		result += "(";
		if (unpivot_names.size() == 1) {
			result += KeywordHelper::WriteOptionallyQuoted(unpivot_names[0]);
		} else {
			result += "(";
			for (idx_t n = 0; n < unpivot_names.size(); n++) {
				if (n > 0) {
					result += ", ";
				}
				result += KeywordHelper::WriteOptionallyQuoted(unpivot_names[n]);
			}
			result += ")";
		}
	}
	result += " FOR";
	for (auto &pivot : pivots) {
		result += " ";
		result += pivot.ToString();
	}
	if (!groups.empty()) {
		result += " GROUP BY ";
		for (idx_t i = 0; i < groups.size(); i++) {
			if (i > 0) {
				result += ", ";
			}
			result += groups[i];
		}
	}
	result += ")";
	if (!alias.empty()) {
		result += " AS " + KeywordHelper::WriteOptionallyQuoted(alias);
		if (!column_name_alias.empty()) {
			result += "(";
			for (idx_t i = 0; i < column_name_alias.size(); i++) {
				if (i > 0) {
					result += ", ";
				}
				result += KeywordHelper::WriteOptionallyQuoted(column_name_alias[i]);
			}
			result += ")";
		}
	}
	return result;
}

bool PivotRef::Equals(const TableRef *other_p) const {
	if (!TableRef::Equals(other_p)) {
		return false;
	}
	auto other = (PivotRef *)other_p;
	if (!source->Equals(other->source.get())) {
		return false;
	}
	if (aggregates.size() != other->aggregates.size()) {
		return false;
	}
	for (idx_t i = 0; i < aggregates.size(); i++) {
		if (!BaseExpression::Equals(aggregates[i].get(), other->aggregates[i].get())) {
			return false;
		}
	}
	if (pivots.size() != other->pivots.size()) {
		return false;
	}
	for (idx_t i = 0; i < pivots.size(); i++) {
		if (!pivots[i].Equals(other->pivots[i])) {
			return false;
		}
	}
	if (unpivot_names != other->unpivot_names) {
		return false;
	}
	if (alias != other->alias) {
		return false;
	}
	if (groups != other->groups) {
		return false;
	}
	if (include_nulls != other->include_nulls) {
		return false;
	}
	return true;
}

unique_ptr<TableRef> PivotRef::Copy() {
	auto copy = make_uniq<PivotRef>();
	copy->source = source->Copy();
	for (auto &aggr : aggregates) {
		copy->aggregates.push_back(aggr->Copy());
	}
	copy->unpivot_names = unpivot_names;
	for (auto &entry : pivots) {
		copy->pivots.push_back(entry.Copy());
	}
	copy->groups = groups;
	copy->column_name_alias = column_name_alias;
	copy->include_nulls = include_nulls;
	copy->alias = alias;
	return std::move(copy);
}

void PivotRef::Serialize(FieldWriter &writer) const {
	writer.WriteSerializable(*source);
	writer.WriteSerializableList(aggregates);
	writer.WriteList<string>(unpivot_names);
	writer.WriteRegularSerializableList(pivots);
	writer.WriteList<string>(groups);
	writer.WriteList<string>(column_name_alias);
	writer.WriteField<bool>(include_nulls);
}

void PivotRef::FormatSerialize(FormatSerializer &serializer) const {
	TableRef::FormatSerialize(serializer);
	serializer.WriteProperty("source", source);
	serializer.WriteProperty("aggregates", aggregates);
	serializer.WriteProperty("unpivot_names", unpivot_names);
	serializer.WriteProperty("pivots", pivots);
	serializer.WriteProperty("groups", groups);
	serializer.WriteProperty("column_name_alias", column_name_alias);
	serializer.WriteProperty("include_nulls", include_nulls);
}

unique_ptr<TableRef> PivotRef::Deserialize(FieldReader &reader) {
	auto result = make_uniq<PivotRef>();
	result->source = reader.ReadRequiredSerializable<TableRef>();
	result->aggregates = reader.ReadRequiredSerializableList<ParsedExpression>();
	result->unpivot_names = reader.ReadRequiredList<string>();
	result->pivots = reader.ReadRequiredSerializableList<PivotColumn, PivotColumn>();
	result->groups = reader.ReadRequiredList<string>();
	result->column_name_alias = reader.ReadRequiredList<string>();
	result->include_nulls = reader.ReadRequired<bool>();
	return std::move(result);
}

unique_ptr<TableRef> PivotRef::FormatDeserialize(FormatDeserializer &source) {
	auto result = make_uniq<PivotRef>();
	source.ReadProperty("source", result->source);
	source.ReadProperty("aggregates", result->aggregates);
	source.ReadProperty("unpivot_names", result->unpivot_names);
	source.ReadProperty("pivots", result->pivots);
	source.ReadProperty("groups", result->groups);
	source.ReadProperty("column_name_alias", result->column_name_alias);
	source.ReadProperty("include_nulls", result->include_nulls);
	return std::move(result);
}

} // namespace duckdb







namespace duckdb {

string SubqueryRef::ToString() const {
	string result = "(" + subquery->ToString() + ")";
	return BaseToString(result, column_name_alias);
}

SubqueryRef::SubqueryRef(unique_ptr<SelectStatement> subquery_p, string alias_p)
    : TableRef(TableReferenceType::SUBQUERY), subquery(std::move(subquery_p)) {
	this->alias = std::move(alias_p);
}

bool SubqueryRef::Equals(const TableRef *other_p) const {
	if (!TableRef::Equals(other_p)) {
		return false;
	}
	auto other = (SubqueryRef *)other_p;
	return subquery->Equals(other->subquery.get());
}

unique_ptr<TableRef> SubqueryRef::Copy() {
	auto copy = make_uniq<SubqueryRef>(unique_ptr_cast<SQLStatement, SelectStatement>(subquery->Copy()), alias);
	copy->column_name_alias = column_name_alias;
	CopyProperties(*copy);
	return std::move(copy);
}

void SubqueryRef::Serialize(FieldWriter &writer) const {
	writer.WriteSerializable(*subquery);
	writer.WriteList<string>(column_name_alias);
}

void SubqueryRef::FormatSerialize(FormatSerializer &serializer) const {
	TableRef::FormatSerialize(serializer);
	serializer.WriteProperty("subquery", subquery);
	serializer.WriteProperty("column_name_alias", column_name_alias);
}

unique_ptr<TableRef> SubqueryRef::FormatDeserialize(FormatDeserializer &deserializer) {
	auto subquery = deserializer.ReadProperty<unique_ptr<SelectStatement>>("subquery");
	auto result = make_uniq<SubqueryRef>(std::move(subquery));
	deserializer.ReadProperty("column_name_alias", result->column_name_alias);
	return std::move(result);
}

unique_ptr<TableRef> SubqueryRef::Deserialize(FieldReader &reader) {
	auto subquery = reader.ReadRequiredSerializable<SelectStatement>();
	auto result = make_uniq<SubqueryRef>(std::move(subquery));
	result->column_name_alias = reader.ReadRequiredList<string>();
	return std::move(result);
}

} // namespace duckdb






namespace duckdb {

TableFunctionRef::TableFunctionRef() : TableRef(TableReferenceType::TABLE_FUNCTION) {
}

string TableFunctionRef::ToString() const {
	return BaseToString(function->ToString(), column_name_alias);
}

bool TableFunctionRef::Equals(const TableRef *other_p) const {
	if (!TableRef::Equals(other_p)) {
		return false;
	}
	auto other = (TableFunctionRef *)other_p;
	return function->Equals(other->function.get());
}

void TableFunctionRef::Serialize(FieldWriter &writer) const {
	writer.WriteSerializable(*function);
	writer.WriteString(alias);
	writer.WriteList<string>(column_name_alias);
}

void TableFunctionRef::FormatSerialize(FormatSerializer &serializer) const {
	TableRef::FormatSerialize(serializer);
	serializer.WriteProperty("function", function);
	serializer.WriteProperty("alias", alias);
	serializer.WriteProperty("column_name_alias", column_name_alias);
}

unique_ptr<TableRef> TableFunctionRef::FormatDeserialize(FormatDeserializer &deserializer) {
	auto result = make_uniq<TableFunctionRef>();
	deserializer.ReadProperty("function", result->function);
	deserializer.ReadProperty("alias", result->alias);
	deserializer.ReadProperty("column_name_alias", result->column_name_alias);
	return std::move(result);
}

unique_ptr<TableRef> TableFunctionRef::Deserialize(FieldReader &reader) {
	auto result = make_uniq<TableFunctionRef>();
	result->function = reader.ReadRequiredSerializable<ParsedExpression>();
	result->alias = reader.ReadRequired<string>();
	result->column_name_alias = reader.ReadRequiredList<string>();
	return std::move(result);
}

unique_ptr<TableRef> TableFunctionRef::Copy() {
	auto copy = make_uniq<TableFunctionRef>();

	copy->function = function->Copy();
	copy->column_name_alias = column_name_alias;
	CopyProperties(*copy);

	return std::move(copy);
}

} // namespace duckdb









namespace duckdb {

string TableRef::BaseToString(string result) const {
	vector<string> column_name_alias;
	return BaseToString(std::move(result), column_name_alias);
}

string TableRef::BaseToString(string result, const vector<string> &column_name_alias) const {
	if (!alias.empty()) {
		result += StringUtil::Format(" AS %s", SQLIdentifier(alias));
	}
	if (!column_name_alias.empty()) {
		D_ASSERT(!alias.empty());
		result += "(";
		for (idx_t i = 0; i < column_name_alias.size(); i++) {
			if (i > 0) {
				result += ", ";
			}
			result += KeywordHelper::WriteOptionallyQuoted(column_name_alias[i]);
		}
		result += ")";
	}
	if (sample) {
		result += " TABLESAMPLE " + EnumUtil::ToString(sample->method);
		result += "(" + sample->sample_size.ToString() + " " + string(sample->is_percentage ? "PERCENT" : "ROWS") + ")";
		if (sample->seed >= 0) {
			result += "REPEATABLE (" + to_string(sample->seed) + ")";
		}
	}

	return result;
}

bool TableRef::Equals(const TableRef *other) const {
	return other && type == other->type && alias == other->alias &&
	       SampleOptions::Equals(sample.get(), other->sample.get());
}

void TableRef::Serialize(Serializer &serializer) const {
	FieldWriter writer(serializer);
	writer.WriteField<TableReferenceType>(type);
	writer.WriteString(alias);
	writer.WriteOptional(sample);
	Serialize(writer);
	writer.Finalize();
}

void TableRef::FormatSerialize(FormatSerializer &serializer) const {
	serializer.WriteProperty("type", type);
	serializer.WriteProperty("alias", alias);
	serializer.WriteOptionalProperty("sample", sample);
}

unique_ptr<TableRef> TableRef::FormatDeserialize(FormatDeserializer &deserializer) {
	auto type = deserializer.ReadProperty<TableReferenceType>("type");
	auto alias = deserializer.ReadProperty<string>("alias");
	auto sample = deserializer.ReadOptionalProperty<unique_ptr<SampleOptions>>("sample");

	unique_ptr<TableRef> result;

	switch (type) {
	case TableReferenceType::BASE_TABLE:
		result = BaseTableRef::FormatDeserialize(deserializer);
		break;
	case TableReferenceType::JOIN:
		result = JoinRef::FormatDeserialize(deserializer);
		break;
	case TableReferenceType::SUBQUERY:
		result = SubqueryRef::FormatDeserialize(deserializer);
		break;
	case TableReferenceType::TABLE_FUNCTION:
		result = TableFunctionRef::FormatDeserialize(deserializer);
		break;
	case TableReferenceType::EMPTY:
		result = EmptyTableRef::FormatDeserialize(deserializer);
		break;
	case TableReferenceType::EXPRESSION_LIST:
		result = ExpressionListRef::FormatDeserialize(deserializer);
		break;
	case TableReferenceType::PIVOT:
		result = PivotRef::FormatDeserialize(deserializer);
		break;
	case TableReferenceType::CTE:
	case TableReferenceType::INVALID:
		throw InternalException("Unsupported type for TableRef::FormatDeserialize");
	}
	result->alias = alias;
	result->sample = std::move(sample);
	return result;
}

unique_ptr<TableRef> TableRef::Deserialize(Deserializer &source) {
	FieldReader reader(source);

	auto type = reader.ReadRequired<TableReferenceType>();
	auto alias = reader.ReadRequired<string>();
	auto sample = reader.ReadOptional<SampleOptions>(nullptr);
	unique_ptr<TableRef> result;
	switch (type) {
	case TableReferenceType::BASE_TABLE:
		result = BaseTableRef::Deserialize(reader);
		break;
	case TableReferenceType::JOIN:
		result = JoinRef::Deserialize(reader);
		break;
	case TableReferenceType::SUBQUERY:
		result = SubqueryRef::Deserialize(reader);
		break;
	case TableReferenceType::TABLE_FUNCTION:
		result = TableFunctionRef::Deserialize(reader);
		break;
	case TableReferenceType::EMPTY:
		result = EmptyTableRef::Deserialize(reader);
		break;
	case TableReferenceType::EXPRESSION_LIST:
		result = ExpressionListRef::Deserialize(reader);
		break;
	case TableReferenceType::PIVOT:
		result = PivotRef::Deserialize(reader);
		break;
	case TableReferenceType::CTE:
	case TableReferenceType::INVALID:
		throw InternalException("Unsupported type for TableRef::Deserialize");
	}
	reader.Finalize();

	result->alias = alias;
	result->sample = std::move(sample);
	return result;
}

void TableRef::CopyProperties(TableRef &target) const {
	D_ASSERT(type == target.type);
	target.alias = alias;
	target.query_location = query_location;
	target.sample = sample ? sample->Copy() : nullptr;
}

void TableRef::Print() {
	Printer::Print(ToString());
}

} // namespace duckdb





namespace duckdb {

static void ParseSchemaTableNameFK(duckdb_libpgquery::PGRangeVar *input, ForeignKeyInfo &fk_info) {
	if (input->catalogname) {
		throw ParserException("FOREIGN KEY constraints cannot be defined cross-database");
	}
	if (input->schemaname) {
		fk_info.schema = input->schemaname;
	} else {
		fk_info.schema = "";
	};
	fk_info.table = input->relname;
}

unique_ptr<Constraint> Transformer::TransformConstraint(duckdb_libpgquery::PGListCell *cell) {
	auto constraint = reinterpret_cast<duckdb_libpgquery::PGConstraint *>(cell->data.ptr_value);
	switch (constraint->contype) {
	case duckdb_libpgquery::PG_CONSTR_UNIQUE:
	case duckdb_libpgquery::PG_CONSTR_PRIMARY: {
		bool is_primary_key = constraint->contype == duckdb_libpgquery::PG_CONSTR_PRIMARY;
		vector<string> columns;
		for (auto kc = constraint->keys->head; kc; kc = kc->next) {
			columns.emplace_back(reinterpret_cast<duckdb_libpgquery::PGValue *>(kc->data.ptr_value)->val.str);
		}
		return make_uniq<UniqueConstraint>(columns, is_primary_key);
	}
	case duckdb_libpgquery::PG_CONSTR_CHECK: {
		auto expression = TransformExpression(constraint->raw_expr);
		if (expression->HasSubquery()) {
			throw ParserException("subqueries prohibited in CHECK constraints");
		}
		return make_uniq<CheckConstraint>(TransformExpression(constraint->raw_expr));
	}
	case duckdb_libpgquery::PG_CONSTR_FOREIGN: {
		ForeignKeyInfo fk_info;
		fk_info.type = ForeignKeyType::FK_TYPE_FOREIGN_KEY_TABLE;
		ParseSchemaTableNameFK(constraint->pktable, fk_info);
		vector<string> pk_columns, fk_columns;
		for (auto kc = constraint->fk_attrs->head; kc; kc = kc->next) {
			fk_columns.emplace_back(reinterpret_cast<duckdb_libpgquery::PGValue *>(kc->data.ptr_value)->val.str);
		}
		if (constraint->pk_attrs) {
			for (auto kc = constraint->pk_attrs->head; kc; kc = kc->next) {
				pk_columns.emplace_back(reinterpret_cast<duckdb_libpgquery::PGValue *>(kc->data.ptr_value)->val.str);
			}
		}
		if (!pk_columns.empty() && pk_columns.size() != fk_columns.size()) {
			throw ParserException("The number of referencing and referenced columns for foreign keys must be the same");
		}
		if (fk_columns.empty()) {
			throw ParserException("The set of referencing and referenced columns for foreign keys must be not empty");
		}
		return make_uniq<ForeignKeyConstraint>(pk_columns, fk_columns, std::move(fk_info));
	}
	default:
		throw NotImplementedException("Constraint type not handled yet!");
	}
}

unique_ptr<Constraint> Transformer::TransformConstraint(duckdb_libpgquery::PGListCell *cell, ColumnDefinition &column,
                                                        idx_t index) {
	auto constraint = reinterpret_cast<duckdb_libpgquery::PGConstraint *>(cell->data.ptr_value);
	D_ASSERT(constraint);
	switch (constraint->contype) {
	case duckdb_libpgquery::PG_CONSTR_NOTNULL:
		return make_uniq<NotNullConstraint>(LogicalIndex(index));
	case duckdb_libpgquery::PG_CONSTR_CHECK:
		return TransformConstraint(cell);
	case duckdb_libpgquery::PG_CONSTR_PRIMARY:
		return make_uniq<UniqueConstraint>(LogicalIndex(index), true);
	case duckdb_libpgquery::PG_CONSTR_UNIQUE:
		return make_uniq<UniqueConstraint>(LogicalIndex(index), false);
	case duckdb_libpgquery::PG_CONSTR_NULL:
		return nullptr;
	case duckdb_libpgquery::PG_CONSTR_GENERATED_VIRTUAL: {
		if (column.DefaultValue()) {
			throw InvalidInputException("DEFAULT constraint on GENERATED column \"%s\" is not allowed", column.Name());
		}
		column.SetGeneratedExpression(TransformExpression(constraint->raw_expr));
		return nullptr;
	}
	case duckdb_libpgquery::PG_CONSTR_GENERATED_STORED:
		throw InvalidInputException("Can not create a STORED generated column!");
	case duckdb_libpgquery::PG_CONSTR_DEFAULT:
		column.SetDefaultValue(TransformExpression(constraint->raw_expr));
		return nullptr;
	case duckdb_libpgquery::PG_CONSTR_COMPRESSION:
		column.SetCompressionType(CompressionTypeFromString(constraint->compression_name));
		if (column.CompressionType() == CompressionType::COMPRESSION_AUTO) {
			throw ParserException("Unrecognized option for column compression, expected none, uncompressed, rle, "
			                      "dictionary, pfor, bitpacking or fsst");
		}
		return nullptr;
	case duckdb_libpgquery::PG_CONSTR_FOREIGN: {
		ForeignKeyInfo fk_info;
		fk_info.type = ForeignKeyType::FK_TYPE_FOREIGN_KEY_TABLE;
		ParseSchemaTableNameFK(constraint->pktable, fk_info);

		vector<string> pk_columns, fk_columns;
		fk_columns.emplace_back(column.Name().c_str());
		if (constraint->pk_attrs) {
			for (auto kc = constraint->pk_attrs->head; kc; kc = kc->next) {
				pk_columns.emplace_back(reinterpret_cast<duckdb_libpgquery::PGValue *>(kc->data.ptr_value)->val.str);
			}
		}
		if (!pk_columns.empty() && pk_columns.size() != fk_columns.size()) {
			throw ParserException("The number of referencing and referenced columns for foreign keys must be the same");
		}
		return make_uniq<ForeignKeyConstraint>(pk_columns, fk_columns, std::move(fk_info));
	}
	default:
		throw NotImplementedException("Constraint not implemented!");
	}
}

} // namespace duckdb






namespace duckdb {

unique_ptr<ParsedExpression> Transformer::TransformArrayAccess(duckdb_libpgquery::PGAIndirection *indirection_node) {
	// transform the source expression
	unique_ptr<ParsedExpression> result;
	result = TransformExpression(indirection_node->arg);

	// now go over the indices
	// note that a single indirection node can contain multiple indices
	// this happens for e.g. more complex accesses (e.g. (foo).field1[42])
	idx_t list_size = 0;
	for (auto node = indirection_node->indirection->head; node != nullptr; node = node->next) {
		auto target = reinterpret_cast<duckdb_libpgquery::PGNode *>(node->data.ptr_value);
		D_ASSERT(target);

		switch (target->type) {
		case duckdb_libpgquery::T_PGAIndices: {
			// index access (either slice or extract)
			auto index = (duckdb_libpgquery::PGAIndices *)target;
			vector<unique_ptr<ParsedExpression>> children;
			children.push_back(std::move(result));
			if (index->is_slice) {
				// slice
				children.push_back(!index->lidx ? make_uniq<ConstantExpression>(Value())
				                                : TransformExpression(index->lidx));
				children.push_back(!index->uidx ? make_uniq<ConstantExpression>(Value())
				                                : TransformExpression(index->uidx));
				result = make_uniq<OperatorExpression>(ExpressionType::ARRAY_SLICE, std::move(children));
			} else {
				// array access
				D_ASSERT(!index->lidx);
				D_ASSERT(index->uidx);
				children.push_back(TransformExpression(index->uidx));
				result = make_uniq<OperatorExpression>(ExpressionType::ARRAY_EXTRACT, std::move(children));
			}
			break;
		}
		case duckdb_libpgquery::T_PGString: {
			auto val = (duckdb_libpgquery::PGValue *)target;
			vector<unique_ptr<ParsedExpression>> children;
			children.push_back(std::move(result));
			children.push_back(TransformValue(*val));
			result = make_uniq<OperatorExpression>(ExpressionType::STRUCT_EXTRACT, std::move(children));
			break;
		}
		case duckdb_libpgquery::T_PGFuncCall: {
			auto func = (duckdb_libpgquery::PGFuncCall *)target;
			auto function = TransformFuncCall(func);
			if (function->type != ExpressionType::FUNCTION) {
				throw ParserException("%s.%s() call must be a function", result->ToString(), function->ToString());
			}
			auto &f = function->Cast<FunctionExpression>();
			f.children.insert(f.children.begin(), std::move(result));
			result = std::move(function);
			break;
		}
		default:
			throw NotImplementedException("Unimplemented subscript type");
		}
		list_size++;
		StackCheck(list_size);
	}
	return result;
}

} // namespace duckdb




namespace duckdb {

unique_ptr<ParsedExpression> Transformer::TransformBoolExpr(duckdb_libpgquery::PGBoolExpr *root) {
	unique_ptr<ParsedExpression> result;
	for (auto node = root->args->head; node != nullptr; node = node->next) {
		auto next = TransformExpression(reinterpret_cast<duckdb_libpgquery::PGNode *>(node->data.ptr_value));

		switch (root->boolop) {
		case duckdb_libpgquery::PG_AND_EXPR: {
			if (!result) {
				result = std::move(next);
			} else {
				result = make_uniq<ConjunctionExpression>(ExpressionType::CONJUNCTION_AND, std::move(result),
				                                          std::move(next));
			}
			break;
		}
		case duckdb_libpgquery::PG_OR_EXPR: {
			if (!result) {
				result = std::move(next);
			} else {
				result = make_uniq<ConjunctionExpression>(ExpressionType::CONJUNCTION_OR, std::move(result),
				                                          std::move(next));
			}
			break;
		}
		case duckdb_libpgquery::PG_NOT_EXPR: {
			if (next->type == ExpressionType::COMPARE_IN) {
				// convert COMPARE_IN to COMPARE_NOT_IN
				next->type = ExpressionType::COMPARE_NOT_IN;
				result = std::move(next);
			} else if (next->type >= ExpressionType::COMPARE_EQUAL &&
			           next->type <= ExpressionType::COMPARE_GREATERTHANOREQUALTO) {
				// NOT on a comparison: we can negate the comparison
				// e.g. NOT(x > y) is equivalent to x <= y
				next->type = NegateComparisonExpression(next->type);
				result = std::move(next);
			} else {
				result = make_uniq<OperatorExpression>(ExpressionType::OPERATOR_NOT, std::move(next));
			}
			break;
		}
		}
	}
	return result;
}

} // namespace duckdb






namespace duckdb {

unique_ptr<ParsedExpression> Transformer::TransformBooleanTest(duckdb_libpgquery::PGBooleanTest *node) {
	auto argument = TransformExpression(reinterpret_cast<duckdb_libpgquery::PGNode *>(node->arg));

	auto expr_true = make_uniq<ConstantExpression>(Value::BOOLEAN(true));
	auto expr_false = make_uniq<ConstantExpression>(Value::BOOLEAN(false));
	// we cast the argument to bool to remove ambiguity wrt function binding on the comparision
	auto cast_argument = make_uniq<CastExpression>(LogicalType::BOOLEAN, argument->Copy());

	switch (node->booltesttype) {
	case duckdb_libpgquery::PGBoolTestType::PG_IS_TRUE:
		return make_uniq<ComparisonExpression>(ExpressionType::COMPARE_NOT_DISTINCT_FROM, std::move(cast_argument),
		                                       std::move(expr_true));
	case duckdb_libpgquery::PGBoolTestType::IS_NOT_TRUE:
		return make_uniq<ComparisonExpression>(ExpressionType::COMPARE_DISTINCT_FROM, std::move(cast_argument),
		                                       std::move(expr_true));
	case duckdb_libpgquery::PGBoolTestType::IS_FALSE:
		return make_uniq<ComparisonExpression>(ExpressionType::COMPARE_NOT_DISTINCT_FROM, std::move(cast_argument),
		                                       std::move(expr_false));
	case duckdb_libpgquery::PGBoolTestType::IS_NOT_FALSE:
		return make_uniq<ComparisonExpression>(ExpressionType::COMPARE_DISTINCT_FROM, std::move(cast_argument),
		                                       std::move(expr_false));
	case duckdb_libpgquery::PGBoolTestType::IS_UNKNOWN: // IS NULL
		return make_uniq<OperatorExpression>(ExpressionType::OPERATOR_IS_NULL, std::move(argument));
	case duckdb_libpgquery::PGBoolTestType::IS_NOT_UNKNOWN: // IS NOT NULL
		return make_uniq<OperatorExpression>(ExpressionType::OPERATOR_IS_NOT_NULL, std::move(argument));
	default:
		throw NotImplementedException("Unknown boolean test type %d", node->booltesttype);
	}
}

} // namespace duckdb





namespace duckdb {

unique_ptr<ParsedExpression> Transformer::TransformCase(duckdb_libpgquery::PGCaseExpr *root) {
	D_ASSERT(root);

	auto case_node = make_uniq<CaseExpression>();
	auto root_arg = TransformExpression(reinterpret_cast<duckdb_libpgquery::PGNode *>(root->arg));
	for (auto cell = root->args->head; cell != nullptr; cell = cell->next) {
		CaseCheck case_check;

		auto w = reinterpret_cast<duckdb_libpgquery::PGCaseWhen *>(cell->data.ptr_value);
		auto test_raw = TransformExpression(reinterpret_cast<duckdb_libpgquery::PGNode *>(w->expr));
		unique_ptr<ParsedExpression> test;
		if (root_arg) {
			case_check.when_expr =
			    make_uniq<ComparisonExpression>(ExpressionType::COMPARE_EQUAL, root_arg->Copy(), std::move(test_raw));
		} else {
			case_check.when_expr = std::move(test_raw);
		}
		case_check.then_expr = TransformExpression(reinterpret_cast<duckdb_libpgquery::PGNode *>(w->result));
		case_node->case_checks.push_back(std::move(case_check));
	}

	if (root->defresult) {
		case_node->else_expr = TransformExpression(reinterpret_cast<duckdb_libpgquery::PGNode *>(root->defresult));
	} else {
		case_node->else_expr = make_uniq<ConstantExpression>(Value(LogicalType::SQLNULL));
	}
	return std::move(case_node);
}

} // namespace duckdb






namespace duckdb {

unique_ptr<ParsedExpression> Transformer::TransformTypeCast(duckdb_libpgquery::PGTypeCast *root) {
	D_ASSERT(root);

	// get the type to cast to
	auto type_name = root->typeName;
	LogicalType target_type = TransformTypeName(type_name);

	// check for a constant BLOB value, then return ConstantExpression with BLOB
	if (!root->tryCast && target_type == LogicalType::BLOB && root->arg->type == duckdb_libpgquery::T_PGAConst) {
		auto c = reinterpret_cast<duckdb_libpgquery::PGAConst *>(root->arg);
		if (c->val.type == duckdb_libpgquery::T_PGString) {
			return make_uniq<ConstantExpression>(Value::BLOB(string(c->val.val.str)));
		}
	}
	// transform the expression node
	auto expression = TransformExpression(root->arg);
	bool try_cast = root->tryCast;

	// now create a cast operation
	return make_uniq<CastExpression>(target_type, std::move(expression), try_cast);
}

} // namespace duckdb



namespace duckdb {

// COALESCE(a,b,c) returns the first argument that is NOT NULL, so
// rewrite into CASE(a IS NOT NULL, a, CASE(b IS NOT NULL, b, c))
unique_ptr<ParsedExpression> Transformer::TransformCoalesce(duckdb_libpgquery::PGAExpr *root) {
	D_ASSERT(root);

	auto coalesce_args = reinterpret_cast<duckdb_libpgquery::PGList *>(root->lexpr);
	D_ASSERT(coalesce_args->length > 0); // parser ensures this already

	auto coalesce_op = make_uniq<OperatorExpression>(ExpressionType::OPERATOR_COALESCE);
	for (auto cell = coalesce_args->head; cell; cell = cell->next) {
		// get the value of the COALESCE
		auto value_expr = TransformExpression(reinterpret_cast<duckdb_libpgquery::PGNode *>(cell->data.ptr_value));
		coalesce_op->children.push_back(std::move(value_expr));
	}
	return std::move(coalesce_op);
}

} // namespace duckdb






namespace duckdb {

unique_ptr<ParsedExpression> Transformer::TransformStarExpression(duckdb_libpgquery::PGNode *node) {
	auto star = (duckdb_libpgquery::PGAStar *)node;
	auto result = make_uniq<StarExpression>(star->relation ? star->relation : string());
	if (star->except_list) {
		for (auto head = star->except_list->head; head; head = head->next) {
			auto value = (duckdb_libpgquery::PGValue *)head->data.ptr_value;
			D_ASSERT(value->type == duckdb_libpgquery::T_PGString);
			string exclude_entry = value->val.str;
			if (result->exclude_list.find(exclude_entry) != result->exclude_list.end()) {
				throw ParserException("Duplicate entry \"%s\" in EXCLUDE list", exclude_entry);
			}
			result->exclude_list.insert(std::move(exclude_entry));
		}
	}
	if (star->replace_list) {
		for (auto head = star->replace_list->head; head; head = head->next) {
			auto list = (duckdb_libpgquery::PGList *)head->data.ptr_value;
			D_ASSERT(list->length == 2);
			auto replace_expression = TransformExpression((duckdb_libpgquery::PGNode *)list->head->data.ptr_value);
			auto value = (duckdb_libpgquery::PGValue *)list->tail->data.ptr_value;
			D_ASSERT(value->type == duckdb_libpgquery::T_PGString);
			string exclude_entry = value->val.str;
			if (result->replace_list.find(exclude_entry) != result->replace_list.end()) {
				throw ParserException("Duplicate entry \"%s\" in REPLACE list", exclude_entry);
			}
			if (result->exclude_list.find(exclude_entry) != result->exclude_list.end()) {
				throw ParserException("Column \"%s\" cannot occur in both EXCEPT and REPLACE list", exclude_entry);
			}
			result->replace_list.insert(make_pair(std::move(exclude_entry), std::move(replace_expression)));
		}
	}
	if (star->expr) {
		D_ASSERT(star->columns);
		D_ASSERT(result->relation_name.empty());
		D_ASSERT(result->exclude_list.empty());
		D_ASSERT(result->replace_list.empty());
		result->expr = TransformExpression(star->expr);
		if (result->expr->type == ExpressionType::STAR) {
			auto &child_star = result->expr->Cast<StarExpression>();
			result->exclude_list = std::move(child_star.exclude_list);
			result->replace_list = std::move(child_star.replace_list);
			result->expr.reset();
		} else if (result->expr->type == ExpressionType::LAMBDA) {
			vector<unique_ptr<ParsedExpression>> children;
			children.push_back(make_uniq<StarExpression>());
			children.push_back(std::move(result->expr));
			auto list_filter = make_uniq<FunctionExpression>("list_filter", std::move(children));
			result->expr = std::move(list_filter);
		}
	}
	result->columns = star->columns;
	result->query_location = star->location;
	return std::move(result);
}

unique_ptr<ParsedExpression> Transformer::TransformColumnRef(duckdb_libpgquery::PGColumnRef *root) {
	auto fields = root->fields;
	auto head_node = (duckdb_libpgquery::PGNode *)fields->head->data.ptr_value;
	switch (head_node->type) {
	case duckdb_libpgquery::T_PGString: {
		if (fields->length < 1) {
			throw InternalException("Unexpected field length");
		}
		vector<string> column_names;
		for (auto node = fields->head; node; node = node->next) {
			column_names.emplace_back(reinterpret_cast<duckdb_libpgquery::PGValue *>(node->data.ptr_value)->val.str);
		}
		auto colref = make_uniq<ColumnRefExpression>(std::move(column_names));
		colref->query_location = root->location;
		return std::move(colref);
	}
	case duckdb_libpgquery::T_PGAStar: {
		return TransformStarExpression(head_node);
	}
	default:
		throw NotImplementedException("ColumnRef not implemented!");
	}
}

} // namespace duckdb






namespace duckdb {

unique_ptr<ConstantExpression> Transformer::TransformValue(duckdb_libpgquery::PGValue val) {
	switch (val.type) {
	case duckdb_libpgquery::T_PGInteger:
		D_ASSERT(val.val.ival <= NumericLimits<int32_t>::Maximum());
		return make_uniq<ConstantExpression>(Value::INTEGER((int32_t)val.val.ival));
	case duckdb_libpgquery::T_PGBitString: // FIXME: this should actually convert to BLOB
	case duckdb_libpgquery::T_PGString:
		return make_uniq<ConstantExpression>(Value(string(val.val.str)));
	case duckdb_libpgquery::T_PGFloat: {
		string_t str_val(val.val.str);
		bool try_cast_as_integer = true;
		bool try_cast_as_decimal = true;
		int decimal_position = -1;
		for (idx_t i = 0; i < str_val.GetSize(); i++) {
			if (val.val.str[i] == '.') {
				// decimal point: cast as either decimal or double
				try_cast_as_integer = false;
				decimal_position = i;
			}
			if (val.val.str[i] == 'e' || val.val.str[i] == 'E') {
				// found exponent, cast as double
				try_cast_as_integer = false;
				try_cast_as_decimal = false;
			}
		}
		if (try_cast_as_integer) {
			int64_t bigint_value;
			// try to cast as bigint first
			if (TryCast::Operation<string_t, int64_t>(str_val, bigint_value)) {
				// successfully cast to bigint: bigint value
				return make_uniq<ConstantExpression>(Value::BIGINT(bigint_value));
			}
			hugeint_t hugeint_value;
			// if that is not successful; try to cast as hugeint
			if (TryCast::Operation<string_t, hugeint_t>(str_val, hugeint_value)) {
				// successfully cast to bigint: bigint value
				return make_uniq<ConstantExpression>(Value::HUGEINT(hugeint_value));
			}
		}
		idx_t decimal_offset = val.val.str[0] == '-' ? 3 : 2;
		if (try_cast_as_decimal && decimal_position >= 0 &&
		    str_val.GetSize() < Decimal::MAX_WIDTH_DECIMAL + decimal_offset) {
			// figure out the width/scale based on the decimal position
			auto width = uint8_t(str_val.GetSize() - 1);
			auto scale = uint8_t(width - decimal_position);
			if (val.val.str[0] == '-') {
				width--;
			}
			if (width <= Decimal::MAX_WIDTH_DECIMAL) {
				// we can cast the value as a decimal
				Value val = Value(str_val);
				val = val.DefaultCastAs(LogicalType::DECIMAL(width, scale));
				return make_uniq<ConstantExpression>(std::move(val));
			}
		}
		// if there is a decimal or the value is too big to cast as either hugeint or bigint
		double dbl_value = Cast::Operation<string_t, double>(str_val);
		return make_uniq<ConstantExpression>(Value::DOUBLE(dbl_value));
	}
	case duckdb_libpgquery::T_PGNull:
		return make_uniq<ConstantExpression>(Value(LogicalType::SQLNULL));
	default:
		throw NotImplementedException("Value not implemented!");
	}
}

unique_ptr<ParsedExpression> Transformer::TransformConstant(duckdb_libpgquery::PGAConst *c) {
	return TransformValue(c->val);
}

} // namespace duckdb




namespace duckdb {

unique_ptr<ParsedExpression> Transformer::TransformResTarget(duckdb_libpgquery::PGResTarget *root) {
	D_ASSERT(root);

	auto expr = TransformExpression(root->val);
	if (!expr) {
		return nullptr;
	}
	if (root->name) {
		expr->alias = string(root->name);
	}
	return expr;
}

unique_ptr<ParsedExpression> Transformer::TransformNamedArg(duckdb_libpgquery::PGNamedArgExpr *root) {
	D_ASSERT(root);

	auto expr = TransformExpression((duckdb_libpgquery::PGNode *)root->arg);
	if (root->name) {
		expr->alias = string(root->name);
	}
	return expr;
}

unique_ptr<ParsedExpression> Transformer::TransformExpression(duckdb_libpgquery::PGNode *node) {
	if (!node) {
		return nullptr;
	}

	auto stack_checker = StackCheck();

	switch (node->type) {
	case duckdb_libpgquery::T_PGColumnRef:
		return TransformColumnRef(reinterpret_cast<duckdb_libpgquery::PGColumnRef *>(node));
	case duckdb_libpgquery::T_PGAConst:
		return TransformConstant(reinterpret_cast<duckdb_libpgquery::PGAConst *>(node));
	case duckdb_libpgquery::T_PGAExpr:
		return TransformAExpr(reinterpret_cast<duckdb_libpgquery::PGAExpr *>(node));
	case duckdb_libpgquery::T_PGFuncCall:
		return TransformFuncCall(reinterpret_cast<duckdb_libpgquery::PGFuncCall *>(node));
	case duckdb_libpgquery::T_PGBoolExpr:
		return TransformBoolExpr(reinterpret_cast<duckdb_libpgquery::PGBoolExpr *>(node));
	case duckdb_libpgquery::T_PGTypeCast:
		return TransformTypeCast(reinterpret_cast<duckdb_libpgquery::PGTypeCast *>(node));
	case duckdb_libpgquery::T_PGCaseExpr:
		return TransformCase(reinterpret_cast<duckdb_libpgquery::PGCaseExpr *>(node));
	case duckdb_libpgquery::T_PGSubLink:
		return TransformSubquery(reinterpret_cast<duckdb_libpgquery::PGSubLink *>(node));
	case duckdb_libpgquery::T_PGCoalesceExpr:
		return TransformCoalesce(reinterpret_cast<duckdb_libpgquery::PGAExpr *>(node));
	case duckdb_libpgquery::T_PGNullTest:
		return TransformNullTest(reinterpret_cast<duckdb_libpgquery::PGNullTest *>(node));
	case duckdb_libpgquery::T_PGResTarget:
		return TransformResTarget(reinterpret_cast<duckdb_libpgquery::PGResTarget *>(node));
	case duckdb_libpgquery::T_PGParamRef:
		return TransformParamRef(reinterpret_cast<duckdb_libpgquery::PGParamRef *>(node));
	case duckdb_libpgquery::T_PGNamedArgExpr:
		return TransformNamedArg(reinterpret_cast<duckdb_libpgquery::PGNamedArgExpr *>(node));
	case duckdb_libpgquery::T_PGSQLValueFunction:
		return TransformSQLValueFunction(reinterpret_cast<duckdb_libpgquery::PGSQLValueFunction *>(node));
	case duckdb_libpgquery::T_PGSetToDefault:
		return make_uniq<DefaultExpression>();
	case duckdb_libpgquery::T_PGCollateClause:
		return TransformCollateExpr(reinterpret_cast<duckdb_libpgquery::PGCollateClause *>(node));
	case duckdb_libpgquery::T_PGIntervalConstant:
		return TransformInterval(reinterpret_cast<duckdb_libpgquery::PGIntervalConstant *>(node));
	case duckdb_libpgquery::T_PGLambdaFunction:
		return TransformLambda(reinterpret_cast<duckdb_libpgquery::PGLambdaFunction *>(node));
	case duckdb_libpgquery::T_PGAIndirection:
		return TransformArrayAccess(reinterpret_cast<duckdb_libpgquery::PGAIndirection *>(node));
	case duckdb_libpgquery::T_PGPositionalReference:
		return TransformPositionalReference(reinterpret_cast<duckdb_libpgquery::PGPositionalReference *>(node));
	case duckdb_libpgquery::T_PGGroupingFunc:
		return TransformGroupingFunction(reinterpret_cast<duckdb_libpgquery::PGGroupingFunc *>(node));
	case duckdb_libpgquery::T_PGAStar:
		return TransformStarExpression(node);
	case duckdb_libpgquery::T_PGBooleanTest:
		return TransformBooleanTest(reinterpret_cast<duckdb_libpgquery::PGBooleanTest *>(node));

	default:
		throw NotImplementedException("Expression type %s (%d)", NodetypeToString(node->type), (int)node->type);
	}
}

void Transformer::TransformExpressionList(duckdb_libpgquery::PGList &list,
                                          vector<unique_ptr<ParsedExpression>> &result) {
	for (auto node = list.head; node != nullptr; node = node->next) {
		auto target = reinterpret_cast<duckdb_libpgquery::PGNode *>(node->data.ptr_value);
		D_ASSERT(target);

		auto expr = TransformExpression(target);
		D_ASSERT(expr);

		result.push_back(std::move(expr));
	}
}

} // namespace duckdb













namespace duckdb {

void Transformer::TransformWindowDef(duckdb_libpgquery::PGWindowDef *window_spec, WindowExpression *expr,
                                     const char *window_name) {
	D_ASSERT(window_spec);
	D_ASSERT(expr);

	// next: partitioning/ordering expressions
	if (window_spec->partitionClause) {
		if (window_name && !expr->partitions.empty()) {
			throw ParserException("Cannot override PARTITION BY clause of window \"%s\"", window_name);
		}
		TransformExpressionList(*window_spec->partitionClause, expr->partitions);
	}
	if (window_spec->orderClause) {
		if (window_name && !expr->orders.empty()) {
			throw ParserException("Cannot override ORDER BY clause of window \"%s\"", window_name);
		}
		TransformOrderBy(window_spec->orderClause, expr->orders);
	}
}

void Transformer::TransformWindowFrame(duckdb_libpgquery::PGWindowDef *window_spec, WindowExpression *expr) {
	D_ASSERT(window_spec);
	D_ASSERT(expr);

	// finally: specifics of bounds
	expr->start_expr = TransformExpression(window_spec->startOffset);
	expr->end_expr = TransformExpression(window_spec->endOffset);

	if ((window_spec->frameOptions & FRAMEOPTION_END_UNBOUNDED_PRECEDING) ||
	    (window_spec->frameOptions & FRAMEOPTION_START_UNBOUNDED_FOLLOWING)) {
		throw InternalException(
		    "Window frames starting with unbounded following or ending in unbounded preceding make no sense");
	}

	const bool rangeMode = (window_spec->frameOptions & FRAMEOPTION_RANGE) != 0;
	if (window_spec->frameOptions & FRAMEOPTION_START_UNBOUNDED_PRECEDING) {
		expr->start = WindowBoundary::UNBOUNDED_PRECEDING;
	} else if (window_spec->frameOptions & FRAMEOPTION_START_VALUE_PRECEDING) {
		expr->start = rangeMode ? WindowBoundary::EXPR_PRECEDING_RANGE : WindowBoundary::EXPR_PRECEDING_ROWS;
	} else if (window_spec->frameOptions & FRAMEOPTION_START_VALUE_FOLLOWING) {
		expr->start = rangeMode ? WindowBoundary::EXPR_FOLLOWING_RANGE : WindowBoundary::EXPR_FOLLOWING_ROWS;
	} else if (window_spec->frameOptions & FRAMEOPTION_START_CURRENT_ROW) {
		expr->start = rangeMode ? WindowBoundary::CURRENT_ROW_RANGE : WindowBoundary::CURRENT_ROW_ROWS;
	}

	if (window_spec->frameOptions & FRAMEOPTION_END_UNBOUNDED_FOLLOWING) {
		expr->end = WindowBoundary::UNBOUNDED_FOLLOWING;
	} else if (window_spec->frameOptions & FRAMEOPTION_END_VALUE_PRECEDING) {
		expr->end = rangeMode ? WindowBoundary::EXPR_PRECEDING_RANGE : WindowBoundary::EXPR_PRECEDING_ROWS;
	} else if (window_spec->frameOptions & FRAMEOPTION_END_VALUE_FOLLOWING) {
		expr->end = rangeMode ? WindowBoundary::EXPR_FOLLOWING_RANGE : WindowBoundary::EXPR_FOLLOWING_ROWS;
	} else if (window_spec->frameOptions & FRAMEOPTION_END_CURRENT_ROW) {
		expr->end = rangeMode ? WindowBoundary::CURRENT_ROW_RANGE : WindowBoundary::CURRENT_ROW_ROWS;
	}

	D_ASSERT(expr->start != WindowBoundary::INVALID && expr->end != WindowBoundary::INVALID);
	if (((window_spec->frameOptions & (FRAMEOPTION_START_VALUE_PRECEDING | FRAMEOPTION_START_VALUE_FOLLOWING)) &&
	     !expr->start_expr) ||
	    ((window_spec->frameOptions & (FRAMEOPTION_END_VALUE_PRECEDING | FRAMEOPTION_END_VALUE_FOLLOWING)) &&
	     !expr->end_expr)) {
		throw InternalException("Failed to transform window boundary expression");
	}
}

bool Transformer::ExpressionIsEmptyStar(ParsedExpression &expr) {
	if (expr.expression_class != ExpressionClass::STAR) {
		return false;
	}
	auto &star = expr.Cast<StarExpression>();
	if (!star.columns && star.exclude_list.empty() && star.replace_list.empty()) {
		return true;
	}
	return false;
}

bool Transformer::InWindowDefinition() {
	if (in_window_definition) {
		return true;
	}
	if (parent) {
		return parent->InWindowDefinition();
	}
	return false;
}

unique_ptr<ParsedExpression> Transformer::TransformFuncCall(duckdb_libpgquery::PGFuncCall *root) {
	auto name = root->funcname;
	string catalog, schema, function_name;
	if (name->length == 3) {
		// catalog + schema + name
		catalog = reinterpret_cast<duckdb_libpgquery::PGValue *>(name->head->data.ptr_value)->val.str;
		schema = reinterpret_cast<duckdb_libpgquery::PGValue *>(name->head->next->data.ptr_value)->val.str;
		function_name = reinterpret_cast<duckdb_libpgquery::PGValue *>(name->head->next->next->data.ptr_value)->val.str;
	} else if (name->length == 2) {
		// schema + name
		catalog = INVALID_CATALOG;
		schema = reinterpret_cast<duckdb_libpgquery::PGValue *>(name->head->data.ptr_value)->val.str;
		function_name = reinterpret_cast<duckdb_libpgquery::PGValue *>(name->head->next->data.ptr_value)->val.str;
	} else if (name->length == 1) {
		// unqualified name
		catalog = INVALID_CATALOG;
		schema = INVALID_SCHEMA;
		function_name = reinterpret_cast<duckdb_libpgquery::PGValue *>(name->head->data.ptr_value)->val.str;
	} else {
		throw ParserException("TransformFuncCall - Expected 1, 2 or 3 qualifications");
	}

	//  transform children
	vector<unique_ptr<ParsedExpression>> children;
	if (root->args) {
		TransformExpressionList(*root->args, children);
	}
	if (children.size() == 1 && ExpressionIsEmptyStar(*children[0]) && !root->agg_distinct && !root->agg_order) {
		// COUNT(*) gets translated into COUNT()
		children.clear();
	}

	auto lowercase_name = StringUtil::Lower(function_name);
	if (root->over) {
		if (InWindowDefinition()) {
			throw ParserException("window functions are not allowed in window definitions");
		}

		const auto win_fun_type = WindowExpression::WindowToExpressionType(lowercase_name);
		if (win_fun_type == ExpressionType::INVALID) {
			throw InternalException("Unknown/unsupported window function");
		}

		if (root->agg_distinct) {
			throw ParserException("DISTINCT is not implemented for window functions!");
		}

		if (root->agg_order) {
			throw ParserException("ORDER BY is not implemented for window functions!");
		}

		if (win_fun_type != ExpressionType::WINDOW_AGGREGATE && root->agg_filter) {
			throw ParserException("FILTER is not implemented for non-aggregate window functions!");
		}
		if (root->export_state) {
			throw ParserException("EXPORT_STATE is not supported for window functions!");
		}

		if (win_fun_type == ExpressionType::WINDOW_AGGREGATE && root->agg_ignore_nulls) {
			throw ParserException("IGNORE NULLS is not supported for windowed aggregates");
		}

		auto expr = make_uniq<WindowExpression>(win_fun_type, std::move(catalog), std::move(schema), lowercase_name);
		expr->ignore_nulls = root->agg_ignore_nulls;

		if (root->agg_filter) {
			auto filter_expr = TransformExpression(root->agg_filter);
			expr->filter_expr = std::move(filter_expr);
		}

		if (win_fun_type == ExpressionType::WINDOW_AGGREGATE) {
			expr->children = std::move(children);
		} else {
			if (!children.empty()) {
				expr->children.push_back(std::move(children[0]));
			}
			if (win_fun_type == ExpressionType::WINDOW_LEAD || win_fun_type == ExpressionType::WINDOW_LAG) {
				if (children.size() > 1) {
					expr->offset_expr = std::move(children[1]);
				}
				if (children.size() > 2) {
					expr->default_expr = std::move(children[2]);
				}
				if (children.size() > 3) {
					throw ParserException("Incorrect number of parameters for function %s", lowercase_name);
				}
			} else if (win_fun_type == ExpressionType::WINDOW_NTH_VALUE) {
				if (children.size() > 1) {
					expr->children.push_back(std::move(children[1]));
				}
				if (children.size() > 2) {
					throw ParserException("Incorrect number of parameters for function %s", lowercase_name);
				}
			} else {
				if (children.size() > 1) {
					throw ParserException("Incorrect number of parameters for function %s", lowercase_name);
				}
			}
		}
		auto window_spec = reinterpret_cast<duckdb_libpgquery::PGWindowDef *>(root->over);
		if (window_spec->name) {
			auto it = window_clauses.find(StringUtil::Lower(string(window_spec->name)));
			if (it == window_clauses.end()) {
				throw ParserException("window \"%s\" does not exist", window_spec->name);
			}
			window_spec = it->second;
			D_ASSERT(window_spec);
		}
		auto window_ref = window_spec;
		auto window_name = window_ref->refname;
		if (window_ref->refname) {
			auto it = window_clauses.find(StringUtil::Lower(string(window_spec->refname)));
			if (it == window_clauses.end()) {
				throw ParserException("window \"%s\" does not exist", window_spec->refname);
			}
			window_ref = it->second;
			D_ASSERT(window_ref);
		}
		in_window_definition = true;
		TransformWindowDef(window_ref, expr.get());
		if (window_ref != window_spec) {
			TransformWindowDef(window_spec, expr.get(), window_name);
		}
		TransformWindowFrame(window_spec, expr.get());
		in_window_definition = false;
		expr->query_location = root->location;
		return std::move(expr);
	}

	if (root->agg_ignore_nulls) {
		throw ParserException("IGNORE NULLS is not supported for non-window functions");
	}

	unique_ptr<ParsedExpression> filter_expr;
	if (root->agg_filter) {
		filter_expr = TransformExpression(root->agg_filter);
	}

	auto order_bys = make_uniq<OrderModifier>();
	TransformOrderBy(root->agg_order, order_bys->orders);

	// Ordered aggregates can be either WITHIN GROUP or after the function arguments
	if (root->agg_within_group) {
		//	https://www.postgresql.org/docs/current/functions-aggregate.html#FUNCTIONS-ORDEREDSET-TABLE
		//  Since we implement "ordered aggregates" without sorting,
		//  we map all the ones we support to the corresponding aggregate function.
		if (order_bys->orders.size() != 1) {
			throw ParserException("Cannot use multiple ORDER BY clauses with WITHIN GROUP");
		}
		if (lowercase_name == "percentile_cont") {
			if (children.size() != 1) {
				throw ParserException("Wrong number of arguments for PERCENTILE_CONT");
			}
			lowercase_name = "quantile_cont";
		} else if (lowercase_name == "percentile_disc") {
			if (children.size() != 1) {
				throw ParserException("Wrong number of arguments for PERCENTILE_DISC");
			}
			lowercase_name = "quantile_disc";
		} else if (lowercase_name == "mode") {
			if (!children.empty()) {
				throw ParserException("Wrong number of arguments for MODE");
			}
			lowercase_name = "mode";
		} else {
			throw ParserException("Unknown ordered aggregate \"%s\".", function_name);
		}
	}

	// star gets eaten in the parser
	if (lowercase_name == "count" && children.empty()) {
		lowercase_name = "count_star";
	}

	if (lowercase_name == "if") {
		if (children.size() != 3) {
			throw ParserException("Wrong number of arguments to IF.");
		}
		auto expr = make_uniq<CaseExpression>();
		CaseCheck check;
		check.when_expr = std::move(children[0]);
		check.then_expr = std::move(children[1]);
		expr->case_checks.push_back(std::move(check));
		expr->else_expr = std::move(children[2]);
		return std::move(expr);
	} else if (lowercase_name == "construct_array") {
		auto construct_array = make_uniq<OperatorExpression>(ExpressionType::ARRAY_CONSTRUCTOR);
		construct_array->children = std::move(children);
		return std::move(construct_array);
	} else if (lowercase_name == "ifnull") {
		if (children.size() != 2) {
			throw ParserException("Wrong number of arguments to IFNULL.");
		}

		//  Two-argument COALESCE
		auto coalesce_op = make_uniq<OperatorExpression>(ExpressionType::OPERATOR_COALESCE);
		coalesce_op->children.push_back(std::move(children[0]));
		coalesce_op->children.push_back(std::move(children[1]));
		return std::move(coalesce_op);
	} else if (lowercase_name == "list" && order_bys->orders.size() == 1) {
		// list(expr ORDER BY expr <sense> <nulls>) => list_sort(list(expr), <sense>, <nulls>)
		if (children.size() != 1) {
			throw ParserException("Wrong number of arguments to LIST.");
		}
		auto arg_expr = children[0].get();
		auto &order_by = order_bys->orders[0];
		if (arg_expr->Equals(order_by.expression.get())) {
			auto sense = make_uniq<ConstantExpression>(EnumUtil::ToChars(order_by.type));
			auto nulls = make_uniq<ConstantExpression>(EnumUtil::ToChars(order_by.null_order));
			order_bys = nullptr;
			auto unordered = make_uniq<FunctionExpression>(catalog, schema, lowercase_name.c_str(), std::move(children),
			                                               std::move(filter_expr), std::move(order_bys),
			                                               root->agg_distinct, false, root->export_state);
			lowercase_name = "list_sort";
			order_bys.reset();   // NOLINT
			filter_expr.reset(); // NOLINT
			children.clear();    // NOLINT
			children.emplace_back(std::move(unordered));
			children.emplace_back(std::move(sense));
			children.emplace_back(std::move(nulls));
		}
	}

	auto function = make_uniq<FunctionExpression>(std::move(catalog), std::move(schema), lowercase_name.c_str(),
	                                              std::move(children), std::move(filter_expr), std::move(order_bys),
	                                              root->agg_distinct, false, root->export_state);
	function->query_location = root->location;

	return std::move(function);
}

unique_ptr<ParsedExpression> Transformer::TransformSQLValueFunction(duckdb_libpgquery::PGSQLValueFunction *node) {
	throw InternalException("SQL value functions should not be emitted by the parser");
}

} // namespace duckdb



namespace duckdb {

unique_ptr<ParsedExpression> Transformer::TransformGroupingFunction(duckdb_libpgquery::PGGroupingFunc *n) {
	auto op = make_uniq<OperatorExpression>(ExpressionType::GROUPING_FUNCTION);
	for (auto node = n->args->head; node; node = node->next) {
		auto n = (duckdb_libpgquery::PGNode *)node->data.ptr_value;
		op->children.push_back(TransformExpression(n));
	}
	op->query_location = n->location;
	return std::move(op);
}

} // namespace duckdb






namespace duckdb {

unique_ptr<ParsedExpression> Transformer::TransformInterval(duckdb_libpgquery::PGIntervalConstant *node) {
	// handle post-fix notation of INTERVAL

	// three scenarios
	// interval (expr) year
	// interval 'string' year
	// interval int year
	unique_ptr<ParsedExpression> expr;
	switch (node->val_type) {
	case duckdb_libpgquery::T_PGAExpr:
		expr = TransformExpression(node->eval);
		break;
	case duckdb_libpgquery::T_PGString:
		expr = make_uniq<ConstantExpression>(Value(node->sval));
		break;
	case duckdb_libpgquery::T_PGInteger:
		expr = make_uniq<ConstantExpression>(Value(node->ival));
		break;
	default:
		throw InternalException("Unsupported interval transformation");
	}

	if (!node->typmods) {
		return make_uniq<CastExpression>(LogicalType::INTERVAL, std::move(expr));
	}

	int32_t mask = ((duckdb_libpgquery::PGAConst *)node->typmods->head->data.ptr_value)->val.val.ival;
	// these seemingly random constants are from datetime.hpp
	// they are copied here to avoid having to include this header
	// the bitshift is from the function INTERVAL_MASK in the parser
	constexpr int32_t MONTH_MASK = 1 << 1;
	constexpr int32_t YEAR_MASK = 1 << 2;
	constexpr int32_t DAY_MASK = 1 << 3;
	constexpr int32_t HOUR_MASK = 1 << 10;
	constexpr int32_t MINUTE_MASK = 1 << 11;
	constexpr int32_t SECOND_MASK = 1 << 12;
	constexpr int32_t MILLISECOND_MASK = 1 << 13;
	constexpr int32_t MICROSECOND_MASK = 1 << 14;

	// we need to check certain combinations
	// because certain interval masks (e.g. INTERVAL '10' HOURS TO DAYS) set multiple bits
	// for now we don't support all of the combined ones
	// (we might add support if someone complains about it)

	string fname;
	LogicalType target_type;
	if (mask & YEAR_MASK && mask & MONTH_MASK) {
		// DAY TO HOUR
		throw ParserException("YEAR TO MONTH is not supported");
	} else if (mask & DAY_MASK && mask & HOUR_MASK) {
		// DAY TO HOUR
		throw ParserException("DAY TO HOUR is not supported");
	} else if (mask & DAY_MASK && mask & MINUTE_MASK) {
		// DAY TO MINUTE
		throw ParserException("DAY TO MINUTE is not supported");
	} else if (mask & DAY_MASK && mask & SECOND_MASK) {
		// DAY TO SECOND
		throw ParserException("DAY TO SECOND is not supported");
	} else if (mask & HOUR_MASK && mask & MINUTE_MASK) {
		// DAY TO SECOND
		throw ParserException("HOUR TO MINUTE is not supported");
	} else if (mask & HOUR_MASK && mask & SECOND_MASK) {
		// DAY TO SECOND
		throw ParserException("HOUR TO SECOND is not supported");
	} else if (mask & MINUTE_MASK && mask & SECOND_MASK) {
		// DAY TO SECOND
		throw ParserException("MINUTE TO SECOND is not supported");
	} else if (mask & YEAR_MASK) {
		// YEAR
		fname = "to_years";
		target_type = LogicalType::INTEGER;
	} else if (mask & MONTH_MASK) {
		// MONTH
		fname = "to_months";
		target_type = LogicalType::INTEGER;
	} else if (mask & DAY_MASK) {
		// DAY
		fname = "to_days";
		target_type = LogicalType::INTEGER;
	} else if (mask & HOUR_MASK) {
		// HOUR
		fname = "to_hours";
		target_type = LogicalType::BIGINT;
	} else if (mask & MINUTE_MASK) {
		// MINUTE
		fname = "to_minutes";
		target_type = LogicalType::BIGINT;
	} else if (mask & SECOND_MASK) {
		// SECOND
		fname = "to_seconds";
		target_type = LogicalType::BIGINT;
	} else if (mask & MILLISECOND_MASK) {
		// MILLISECOND
		fname = "to_milliseconds";
		target_type = LogicalType::BIGINT;
	} else if (mask & MICROSECOND_MASK) {
		// SECOND
		fname = "to_microseconds";
		target_type = LogicalType::BIGINT;
	} else {
		throw InternalException("Unsupported interval post-fix");
	}
	// first push a cast to the target type
	expr = make_uniq<CastExpression>(target_type, std::move(expr));
	// now push the operation
	vector<unique_ptr<ParsedExpression>> children;
	children.push_back(std::move(expr));
	return make_uniq<FunctionExpression>(fname, std::move(children));
}

} // namespace duckdb




namespace duckdb {

unique_ptr<ParsedExpression> Transformer::TransformNullTest(duckdb_libpgquery::PGNullTest *root) {
	D_ASSERT(root);
	auto arg = TransformExpression(reinterpret_cast<duckdb_libpgquery::PGNode *>(root->arg));
	if (root->argisrow) {
		throw NotImplementedException("IS NULL argisrow");
	}
	ExpressionType expr_type = (root->nulltesttype == duckdb_libpgquery::PG_IS_NULL)
	                               ? ExpressionType::OPERATOR_IS_NULL
	                               : ExpressionType::OPERATOR_IS_NOT_NULL;

	return unique_ptr<ParsedExpression>(new OperatorExpression(expr_type, std::move(arg)));
}

} // namespace duckdb




namespace duckdb {

unique_ptr<ParsedExpression> Transformer::TransformLambda(duckdb_libpgquery::PGLambdaFunction *node) {

	D_ASSERT(node->lhs);
	D_ASSERT(node->rhs);

	auto lhs = TransformExpression(node->lhs);
	auto rhs = TransformExpression(node->rhs);
	D_ASSERT(lhs);
	D_ASSERT(rhs);
	return make_uniq<LambdaExpression>(std::move(lhs), std::move(rhs));
}

} // namespace duckdb















namespace duckdb {

unique_ptr<ParsedExpression> Transformer::TransformUnaryOperator(const string &op, unique_ptr<ParsedExpression> child) {
	vector<unique_ptr<ParsedExpression>> children;
	children.push_back(std::move(child));

	// built-in operator function
	auto result = make_uniq<FunctionExpression>(op, std::move(children));
	result->is_operator = true;
	return std::move(result);
}

unique_ptr<ParsedExpression> Transformer::TransformBinaryOperator(string op, unique_ptr<ParsedExpression> left,
                                                                  unique_ptr<ParsedExpression> right) {
	vector<unique_ptr<ParsedExpression>> children;
	children.push_back(std::move(left));
	children.push_back(std::move(right));

	if (options.integer_division && op == "/") {
		op = "//";
	}
	if (op == "~" || op == "!~") {
		// rewrite 'asdf' SIMILAR TO '.*sd.*' into regexp_full_match('asdf', '.*sd.*')
		bool invert_similar = op == "!~";

		auto result = make_uniq<FunctionExpression>("regexp_full_match", std::move(children));
		if (invert_similar) {
			return make_uniq<OperatorExpression>(ExpressionType::OPERATOR_NOT, std::move(result));
		} else {
			return std::move(result);
		}
	} else {
		auto target_type = OperatorToExpressionType(op);
		if (target_type != ExpressionType::INVALID) {
			// built-in comparison operator
			return make_uniq<ComparisonExpression>(target_type, std::move(children[0]), std::move(children[1]));
		}
		// not a special operator: convert to a function expression
		auto result = make_uniq<FunctionExpression>(std::move(op), std::move(children));
		result->is_operator = true;
		return std::move(result);
	}
}

unique_ptr<ParsedExpression> Transformer::TransformAExprInternal(duckdb_libpgquery::PGAExpr *root) {
	D_ASSERT(root);
	auto name = string((reinterpret_cast<duckdb_libpgquery::PGValue *>(root->name->head->data.ptr_value))->val.str);

	switch (root->kind) {
	case duckdb_libpgquery::PG_AEXPR_OP_ALL:
	case duckdb_libpgquery::PG_AEXPR_OP_ANY: {
		// left=ANY(right)
		// we turn this into left=ANY((SELECT UNNEST(right)))
		auto left_expr = TransformExpression(root->lexpr);
		auto right_expr = TransformExpression(root->rexpr);

		auto subquery_expr = make_uniq<SubqueryExpression>();
		auto select_statement = make_uniq<SelectStatement>();
		auto select_node = make_uniq<SelectNode>();
		vector<unique_ptr<ParsedExpression>> children;
		children.push_back(std::move(right_expr));

		select_node->select_list.push_back(make_uniq<FunctionExpression>("UNNEST", std::move(children)));
		select_node->from_table = make_uniq<EmptyTableRef>();
		select_statement->node = std::move(select_node);
		subquery_expr->subquery = std::move(select_statement);
		subquery_expr->subquery_type = SubqueryType::ANY;
		subquery_expr->child = std::move(left_expr);
		subquery_expr->comparison_type = OperatorToExpressionType(name);
		subquery_expr->query_location = root->location;

		if (root->kind == duckdb_libpgquery::PG_AEXPR_OP_ALL) {
			// ALL sublink is equivalent to NOT(ANY) with inverted comparison
			// e.g. [= ALL()] is equivalent to [NOT(<> ANY())]
			// first invert the comparison type
			subquery_expr->comparison_type = NegateComparisonExpression(subquery_expr->comparison_type);
			return make_uniq<OperatorExpression>(ExpressionType::OPERATOR_NOT, std::move(subquery_expr));
		}
		return std::move(subquery_expr);
	}
	case duckdb_libpgquery::PG_AEXPR_IN: {
		auto left_expr = TransformExpression(root->lexpr);
		ExpressionType operator_type;
		// this looks very odd, but seems to be the way to find out its NOT IN
		if (name == "<>") {
			// NOT IN
			operator_type = ExpressionType::COMPARE_NOT_IN;
		} else {
			// IN
			operator_type = ExpressionType::COMPARE_IN;
		}
		auto result = make_uniq<OperatorExpression>(operator_type, std::move(left_expr));
		result->query_location = root->location;
		TransformExpressionList(*((duckdb_libpgquery::PGList *)root->rexpr), result->children);
		return std::move(result);
	}
	// rewrite NULLIF(a, b) into CASE WHEN a=b THEN NULL ELSE a END
	case duckdb_libpgquery::PG_AEXPR_NULLIF: {
		vector<unique_ptr<ParsedExpression>> children;
		children.push_back(TransformExpression(root->lexpr));
		children.push_back(TransformExpression(root->rexpr));
		return make_uniq<FunctionExpression>("nullif", std::move(children));
	}
	// rewrite (NOT) X BETWEEN A AND B into (NOT) AND(GREATERTHANOREQUALTO(X,
	// A), LESSTHANOREQUALTO(X, B))
	case duckdb_libpgquery::PG_AEXPR_BETWEEN:
	case duckdb_libpgquery::PG_AEXPR_NOT_BETWEEN: {
		auto between_args = reinterpret_cast<duckdb_libpgquery::PGList *>(root->rexpr);
		if (between_args->length != 2 || !between_args->head->data.ptr_value || !between_args->tail->data.ptr_value) {
			throw InternalException("(NOT) BETWEEN needs two args");
		}

		auto input = TransformExpression(root->lexpr);
		auto between_left =
		    TransformExpression(reinterpret_cast<duckdb_libpgquery::PGNode *>(between_args->head->data.ptr_value));
		auto between_right =
		    TransformExpression(reinterpret_cast<duckdb_libpgquery::PGNode *>(between_args->tail->data.ptr_value));

		auto compare_between =
		    make_uniq<BetweenExpression>(std::move(input), std::move(between_left), std::move(between_right));
		if (root->kind == duckdb_libpgquery::PG_AEXPR_BETWEEN) {
			return std::move(compare_between);
		} else {
			return make_uniq<OperatorExpression>(ExpressionType::OPERATOR_NOT, std::move(compare_between));
		}
	}
	// rewrite SIMILAR TO into regexp_full_match('asdf', '.*sd.*')
	case duckdb_libpgquery::PG_AEXPR_SIMILAR: {
		auto left_expr = TransformExpression(root->lexpr);
		auto right_expr = TransformExpression(root->rexpr);

		vector<unique_ptr<ParsedExpression>> children;
		children.push_back(std::move(left_expr));

		auto &similar_func = reinterpret_cast<FunctionExpression &>(*right_expr);
		D_ASSERT(similar_func.function_name == "similar_escape");
		D_ASSERT(similar_func.children.size() == 2);
		if (similar_func.children[1]->type != ExpressionType::VALUE_CONSTANT) {
			throw NotImplementedException("Custom escape in SIMILAR TO");
		}
		auto &constant = similar_func.children[1]->Cast<ConstantExpression>();
		if (!constant.value.IsNull()) {
			throw NotImplementedException("Custom escape in SIMILAR TO");
		}
		// take the child of the similar_func
		children.push_back(std::move(similar_func.children[0]));

		// this looks very odd, but seems to be the way to find out its NOT IN
		bool invert_similar = false;
		if (name == "!~") {
			// NOT SIMILAR TO
			invert_similar = true;
		}
		const auto regex_function = "regexp_full_match";
		auto result = make_uniq<FunctionExpression>(regex_function, std::move(children));

		if (invert_similar) {
			return make_uniq<OperatorExpression>(ExpressionType::OPERATOR_NOT, std::move(result));
		} else {
			return std::move(result);
		}
	}
	case duckdb_libpgquery::PG_AEXPR_NOT_DISTINCT: {
		auto left_expr = TransformExpression(root->lexpr);
		auto right_expr = TransformExpression(root->rexpr);
		return make_uniq<ComparisonExpression>(ExpressionType::COMPARE_NOT_DISTINCT_FROM, std::move(left_expr),
		                                       std::move(right_expr));
	}
	case duckdb_libpgquery::PG_AEXPR_DISTINCT: {
		auto left_expr = TransformExpression(root->lexpr);
		auto right_expr = TransformExpression(root->rexpr);
		return make_uniq<ComparisonExpression>(ExpressionType::COMPARE_DISTINCT_FROM, std::move(left_expr),
		                                       std::move(right_expr));
	}

	default:
		break;
	}
	auto left_expr = TransformExpression(root->lexpr);
	auto right_expr = TransformExpression(root->rexpr);

	if (!left_expr) {
		// prefix operator
		return TransformUnaryOperator(name, std::move(right_expr));
	} else if (!right_expr) {
		// postfix operator, only ! is currently supported
		return TransformUnaryOperator(name + "__postfix", std::move(left_expr));
	} else {
		return TransformBinaryOperator(std::move(name), std::move(left_expr), std::move(right_expr));
	}
}

unique_ptr<ParsedExpression> Transformer::TransformAExpr(duckdb_libpgquery::PGAExpr *root) {
	auto result = TransformAExprInternal(root);
	if (result) {
		result->query_location = root->location;
	}
	return result;
}

} // namespace duckdb




namespace duckdb {

unique_ptr<ParsedExpression> Transformer::TransformParamRef(duckdb_libpgquery::PGParamRef *node) {
	D_ASSERT(node);
	auto expr = make_uniq<ParameterExpression>();
	if (node->number < 0) {
		throw ParserException("Parameter numbers cannot be negative");
	}

	if (node->name) {
		// This is a named parameter, try to find an entry for it
		D_ASSERT(node->number == 0);
		int32_t index;
		if (GetNamedParam(node->name, index)) {
			// We've seen this named parameter before and assigned it an index!
			node->number = index;
		}
	}
	if (node->number == 0) {
		expr->parameter_nr = ParamCount() + 1;
		if (node->name && !HasNamedParameters() && ParamCount() != 0) {
			// This parameter is named, but there were other parameter before it, and they were not named
			throw NotImplementedException("Mixing positional and named parameters is not supported yet");
		}
		if (node->name) {
			D_ASSERT(!named_param_map.count(node->name));
			// Add it to the named parameter map so we can find it next time it's referenced
			SetNamedParam(node->name, expr->parameter_nr);
		}
	} else {
		if (!node->name && HasNamedParameters()) {
			// This parameter does not have a name, but the named param map is not empty
			throw NotImplementedException("Mixing positional and named parameters is not supported yet");
		}
		expr->parameter_nr = node->number;
	}
	SetParamCount(MaxValue<idx_t>(ParamCount(), expr->parameter_nr));
	return std::move(expr);
}

} // namespace duckdb




namespace duckdb {

unique_ptr<ParsedExpression> Transformer::TransformPositionalReference(duckdb_libpgquery::PGPositionalReference *node) {
	if (node->position <= 0) {
		throw ParserException("Positional reference node needs to be >= 1");
	}
	auto result = make_uniq<PositionalReferenceExpression>(node->position);
	result->query_location = node->location;
	return std::move(result);
}

} // namespace duckdb





namespace duckdb {

unique_ptr<ParsedExpression> Transformer::TransformSubquery(duckdb_libpgquery::PGSubLink *root) {
	D_ASSERT(root);
	auto subquery_expr = make_uniq<SubqueryExpression>();

	subquery_expr->subquery = TransformSelect(root->subselect);
	D_ASSERT(subquery_expr->subquery);
	D_ASSERT(subquery_expr->subquery->node->GetSelectList().size() > 0);

	switch (root->subLinkType) {
	case duckdb_libpgquery::PG_EXISTS_SUBLINK: {
		subquery_expr->subquery_type = SubqueryType::EXISTS;
		break;
	}
	case duckdb_libpgquery::PG_ANY_SUBLINK:
	case duckdb_libpgquery::PG_ALL_SUBLINK: {
		// comparison with ANY() or ALL()
		subquery_expr->subquery_type = SubqueryType::ANY;
		subquery_expr->child = TransformExpression(root->testexpr);
		// get the operator name
		if (!root->operName) {
			// simple IN
			subquery_expr->comparison_type = ExpressionType::COMPARE_EQUAL;
		} else {
			auto operator_name =
			    string((reinterpret_cast<duckdb_libpgquery::PGValue *>(root->operName->head->data.ptr_value))->val.str);
			subquery_expr->comparison_type = OperatorToExpressionType(operator_name);
		}
		if (subquery_expr->comparison_type != ExpressionType::COMPARE_EQUAL &&
		    subquery_expr->comparison_type != ExpressionType::COMPARE_NOTEQUAL &&
		    subquery_expr->comparison_type != ExpressionType::COMPARE_GREATERTHAN &&
		    subquery_expr->comparison_type != ExpressionType::COMPARE_GREATERTHANOREQUALTO &&
		    subquery_expr->comparison_type != ExpressionType::COMPARE_LESSTHAN &&
		    subquery_expr->comparison_type != ExpressionType::COMPARE_LESSTHANOREQUALTO) {
			throw ParserException("ANY and ALL operators require one of =,<>,>,<,>=,<= comparisons!");
		}
		if (root->subLinkType == duckdb_libpgquery::PG_ALL_SUBLINK) {
			// ALL sublink is equivalent to NOT(ANY) with inverted comparison
			// e.g. [= ALL()] is equivalent to [NOT(<> ANY())]
			// first invert the comparison type
			subquery_expr->comparison_type = NegateComparisonExpression(subquery_expr->comparison_type);
			return make_uniq<OperatorExpression>(ExpressionType::OPERATOR_NOT, std::move(subquery_expr));
		}
		break;
	}
	case duckdb_libpgquery::PG_EXPR_SUBLINK: {
		// return a single scalar value from the subquery
		// no child expression to compare to
		subquery_expr->subquery_type = SubqueryType::SCALAR;
		break;
	}
	case duckdb_libpgquery::PG_ARRAY_SUBLINK: {
		auto subquery_table_alias = "__subquery";
		auto subquery_column_alias = "__arr_element";

		// ARRAY expression
		// wrap subquery into "SELECT CASE WHEN ARRAY_AGG(i) IS NULL THEN [] ELSE ARRAY_AGG(i) END FROM (...) tbl(i)"
		auto select_node = make_uniq<SelectNode>();

		// ARRAY_AGG(i)
		vector<unique_ptr<ParsedExpression>> children;
		children.push_back(
		    make_uniq_base<ParsedExpression, ColumnRefExpression>(subquery_column_alias, subquery_table_alias));
		auto aggr = make_uniq<FunctionExpression>("array_agg", std::move(children));
		// ARRAY_AGG(i) IS NULL
		auto agg_is_null = make_uniq<OperatorExpression>(ExpressionType::OPERATOR_IS_NULL, aggr->Copy());
		// empty list
		vector<unique_ptr<ParsedExpression>> list_children;
		auto empty_list = make_uniq<FunctionExpression>("list_value", std::move(list_children));
		// CASE
		auto case_expr = make_uniq<CaseExpression>();
		CaseCheck check;
		check.when_expr = std::move(agg_is_null);
		check.then_expr = std::move(empty_list);
		case_expr->case_checks.push_back(std::move(check));
		case_expr->else_expr = std::move(aggr);

		select_node->select_list.push_back(std::move(case_expr));

		// FROM (...) tbl(i)
		auto child_subquery = make_uniq<SubqueryRef>(std::move(subquery_expr->subquery), subquery_table_alias);
		child_subquery->column_name_alias.emplace_back(subquery_column_alias);
		select_node->from_table = std::move(child_subquery);

		auto new_subquery = make_uniq<SelectStatement>();
		new_subquery->node = std::move(select_node);
		subquery_expr->subquery = std::move(new_subquery);

		subquery_expr->subquery_type = SubqueryType::SCALAR;
		break;
	}
	default:
		throw NotImplementedException("Subquery of type %d not implemented\n", (int)root->subLinkType);
	}
	subquery_expr->query_location = root->location;
	return std::move(subquery_expr);
}

} // namespace duckdb


namespace duckdb {

std::string Transformer::NodetypeToString(duckdb_libpgquery::PGNodeTag type) { // LCOV_EXCL_START
	switch (type) {
	case duckdb_libpgquery::T_PGInvalid:
		return "T_Invalid";
	case duckdb_libpgquery::T_PGIndexInfo:
		return "T_IndexInfo";
	case duckdb_libpgquery::T_PGExprContext:
		return "T_ExprContext";
	case duckdb_libpgquery::T_PGProjectionInfo:
		return "T_ProjectionInfo";
	case duckdb_libpgquery::T_PGJunkFilter:
		return "T_JunkFilter";
	case duckdb_libpgquery::T_PGResultRelInfo:
		return "T_ResultRelInfo";
	case duckdb_libpgquery::T_PGEState:
		return "T_EState";
	case duckdb_libpgquery::T_PGTupleTableSlot:
		return "T_TupleTableSlot";
	case duckdb_libpgquery::T_PGPlan:
		return "T_Plan";
	case duckdb_libpgquery::T_PGResult:
		return "T_Result";
	case duckdb_libpgquery::T_PGProjectSet:
		return "T_ProjectSet";
	case duckdb_libpgquery::T_PGModifyTable:
		return "T_ModifyTable";
	case duckdb_libpgquery::T_PGAppend:
		return "T_Append";
	case duckdb_libpgquery::T_PGMergeAppend:
		return "T_MergeAppend";
	case duckdb_libpgquery::T_PGRecursiveUnion:
		return "T_RecursiveUnion";
	case duckdb_libpgquery::T_PGBitmapAnd:
		return "T_BitmapAnd";
	case duckdb_libpgquery::T_PGBitmapOr:
		return "T_BitmapOr";
	case duckdb_libpgquery::T_PGScan:
		return "T_Scan";
	case duckdb_libpgquery::T_PGSeqScan:
		return "T_SeqScan";
	case duckdb_libpgquery::T_PGSampleScan:
		return "T_SampleScan";
	case duckdb_libpgquery::T_PGIndexScan:
		return "T_IndexScan";
	case duckdb_libpgquery::T_PGIndexOnlyScan:
		return "T_IndexOnlyScan";
	case duckdb_libpgquery::T_PGBitmapIndexScan:
		return "T_BitmapIndexScan";
	case duckdb_libpgquery::T_PGBitmapHeapScan:
		return "T_BitmapHeapScan";
	case duckdb_libpgquery::T_PGTidScan:
		return "T_TidScan";
	case duckdb_libpgquery::T_PGSubqueryScan:
		return "T_SubqueryScan";
	case duckdb_libpgquery::T_PGFunctionScan:
		return "T_FunctionScan";
	case duckdb_libpgquery::T_PGValuesScan:
		return "T_ValuesScan";
	case duckdb_libpgquery::T_PGTableFuncScan:
		return "T_TableFuncScan";
	case duckdb_libpgquery::T_PGCteScan:
		return "T_CteScan";
	case duckdb_libpgquery::T_PGNamedTuplestoreScan:
		return "T_NamedTuplestoreScan";
	case duckdb_libpgquery::T_PGWorkTableScan:
		return "T_WorkTableScan";
	case duckdb_libpgquery::T_PGForeignScan:
		return "T_ForeignScan";
	case duckdb_libpgquery::T_PGCustomScan:
		return "T_CustomScan";
	case duckdb_libpgquery::T_PGJoin:
		return "T_Join";
	case duckdb_libpgquery::T_PGNestLoop:
		return "T_NestLoop";
	case duckdb_libpgquery::T_PGMergeJoin:
		return "T_MergeJoin";
	case duckdb_libpgquery::T_PGHashJoin:
		return "T_HashJoin";
	case duckdb_libpgquery::T_PGMaterial:
		return "T_Material";
	case duckdb_libpgquery::T_PGSort:
		return "T_Sort";
	case duckdb_libpgquery::T_PGGroup:
		return "T_Group";
	case duckdb_libpgquery::T_PGAgg:
		return "T_Agg";
	case duckdb_libpgquery::T_PGWindowAgg:
		return "T_WindowAgg";
	case duckdb_libpgquery::T_PGUnique:
		return "T_Unique";
	case duckdb_libpgquery::T_PGGather:
		return "T_Gather";
	case duckdb_libpgquery::T_PGGatherMerge:
		return "T_GatherMerge";
	case duckdb_libpgquery::T_PGHash:
		return "T_Hash";
	case duckdb_libpgquery::T_PGSetOp:
		return "T_SetOp";
	case duckdb_libpgquery::T_PGLockRows:
		return "T_LockRows";
	case duckdb_libpgquery::T_PGLimit:
		return "T_Limit";
	case duckdb_libpgquery::T_PGNestLoopParam:
		return "T_NestLoopParam";
	case duckdb_libpgquery::T_PGPlanRowMark:
		return "T_PlanRowMark";
	case duckdb_libpgquery::T_PGPlanInvalItem:
		return "T_PlanInvalItem";
	case duckdb_libpgquery::T_PGPlanState:
		return "T_PlanState";
	case duckdb_libpgquery::T_PGResultState:
		return "T_ResultState";
	case duckdb_libpgquery::T_PGProjectSetState:
		return "T_ProjectSetState";
	case duckdb_libpgquery::T_PGModifyTableState:
		return "T_ModifyTableState";
	case duckdb_libpgquery::T_PGAppendState:
		return "T_AppendState";
	case duckdb_libpgquery::T_PGMergeAppendState:
		return "T_MergeAppendState";
	case duckdb_libpgquery::T_PGRecursiveUnionState:
		return "T_RecursiveUnionState";
	case duckdb_libpgquery::T_PGBitmapAndState:
		return "T_BitmapAndState";
	case duckdb_libpgquery::T_PGBitmapOrState:
		return "T_BitmapOrState";
	case duckdb_libpgquery::T_PGScanState:
		return "T_ScanState";
	case duckdb_libpgquery::T_PGSeqScanState:
		return "T_SeqScanState";
	case duckdb_libpgquery::T_PGSampleScanState:
		return "T_SampleScanState";
	case duckdb_libpgquery::T_PGIndexScanState:
		return "T_IndexScanState";
	case duckdb_libpgquery::T_PGIndexOnlyScanState:
		return "T_IndexOnlyScanState";
	case duckdb_libpgquery::T_PGBitmapIndexScanState:
		return "T_BitmapIndexScanState";
	case duckdb_libpgquery::T_PGBitmapHeapScanState:
		return "T_BitmapHeapScanState";
	case duckdb_libpgquery::T_PGTidScanState:
		return "T_TidScanState";
	case duckdb_libpgquery::T_PGSubqueryScanState:
		return "T_SubqueryScanState";
	case duckdb_libpgquery::T_PGFunctionScanState:
		return "T_FunctionScanState";
	case duckdb_libpgquery::T_PGTableFuncScanState:
		return "T_TableFuncScanState";
	case duckdb_libpgquery::T_PGValuesScanState:
		return "T_ValuesScanState";
	case duckdb_libpgquery::T_PGCteScanState:
		return "T_CteScanState";
	case duckdb_libpgquery::T_PGNamedTuplestoreScanState:
		return "T_NamedTuplestoreScanState";
	case duckdb_libpgquery::T_PGWorkTableScanState:
		return "T_WorkTableScanState";
	case duckdb_libpgquery::T_PGForeignScanState:
		return "T_ForeignScanState";
	case duckdb_libpgquery::T_PGCustomScanState:
		return "T_CustomScanState";
	case duckdb_libpgquery::T_PGJoinState:
		return "T_JoinState";
	case duckdb_libpgquery::T_PGNestLoopState:
		return "T_NestLoopState";
	case duckdb_libpgquery::T_PGMergeJoinState:
		return "T_MergeJoinState";
	case duckdb_libpgquery::T_PGHashJoinState:
		return "T_HashJoinState";
	case duckdb_libpgquery::T_PGMaterialState:
		return "T_MaterialState";
	case duckdb_libpgquery::T_PGSortState:
		return "T_SortState";
	case duckdb_libpgquery::T_PGGroupState:
		return "T_GroupState";
	case duckdb_libpgquery::T_PGAggState:
		return "T_AggState";
	case duckdb_libpgquery::T_PGWindowAggState:
		return "T_WindowAggState";
	case duckdb_libpgquery::T_PGUniqueState:
		return "T_UniqueState";
	case duckdb_libpgquery::T_PGGatherState:
		return "T_GatherState";
	case duckdb_libpgquery::T_PGGatherMergeState:
		return "T_GatherMergeState";
	case duckdb_libpgquery::T_PGHashState:
		return "T_HashState";
	case duckdb_libpgquery::T_PGSetOpState:
		return "T_SetOpState";
	case duckdb_libpgquery::T_PGLockRowsState:
		return "T_LockRowsState";
	case duckdb_libpgquery::T_PGLimitState:
		return "T_LimitState";
	case duckdb_libpgquery::T_PGAlias:
		return "T_Alias";
	case duckdb_libpgquery::T_PGRangeVar:
		return "T_RangeVar";
	case duckdb_libpgquery::T_PGTableFunc:
		return "T_TableFunc";
	case duckdb_libpgquery::T_PGExpr:
		return "T_Expr";
	case duckdb_libpgquery::T_PGVar:
		return "T_Var";
	case duckdb_libpgquery::T_PGConst:
		return "T_Const";
	case duckdb_libpgquery::T_PGParam:
		return "T_Param";
	case duckdb_libpgquery::T_PGAggref:
		return "T_Aggref";
	case duckdb_libpgquery::T_PGGroupingFunc:
		return "T_GroupingFunc";
	case duckdb_libpgquery::T_PGWindowFunc:
		return "T_WindowFunc";
	case duckdb_libpgquery::T_PGArrayRef:
		return "T_ArrayRef";
	case duckdb_libpgquery::T_PGFuncExpr:
		return "T_FuncExpr";
	case duckdb_libpgquery::T_PGNamedArgExpr:
		return "T_NamedArgExpr";
	case duckdb_libpgquery::T_PGOpExpr:
		return "T_OpExpr";
	case duckdb_libpgquery::T_PGDistinctExpr:
		return "T_DistinctExpr";
	case duckdb_libpgquery::T_PGNullIfExpr:
		return "T_NullIfExpr";
	case duckdb_libpgquery::T_PGScalarArrayOpExpr:
		return "T_ScalarArrayOpExpr";
	case duckdb_libpgquery::T_PGBoolExpr:
		return "T_BoolExpr";
	case duckdb_libpgquery::T_PGSubLink:
		return "T_SubLink";
	case duckdb_libpgquery::T_PGSubPlan:
		return "T_SubPlan";
	case duckdb_libpgquery::T_PGAlternativeSubPlan:
		return "T_AlternativeSubPlan";
	case duckdb_libpgquery::T_PGFieldSelect:
		return "T_FieldSelect";
	case duckdb_libpgquery::T_PGFieldStore:
		return "T_FieldStore";
	case duckdb_libpgquery::T_PGRelabelType:
		return "T_RelabelType";
	case duckdb_libpgquery::T_PGCoerceViaIO:
		return "T_CoerceViaIO";
	case duckdb_libpgquery::T_PGArrayCoerceExpr:
		return "T_ArrayCoerceExpr";
	case duckdb_libpgquery::T_PGConvertRowtypeExpr:
		return "T_ConvertRowtypeExpr";
	case duckdb_libpgquery::T_PGCollateExpr:
		return "T_CollateExpr";
	case duckdb_libpgquery::T_PGCaseExpr:
		return "T_CaseExpr";
	case duckdb_libpgquery::T_PGCaseWhen:
		return "T_CaseWhen";
	case duckdb_libpgquery::T_PGCaseTestExpr:
		return "T_CaseTestExpr";
	case duckdb_libpgquery::T_PGArrayExpr:
		return "T_ArrayExpr";
	case duckdb_libpgquery::T_PGRowExpr:
		return "T_RowExpr";
	case duckdb_libpgquery::T_PGRowCompareExpr:
		return "T_RowCompareExpr";
	case duckdb_libpgquery::T_PGCoalesceExpr:
		return "T_CoalesceExpr";
	case duckdb_libpgquery::T_PGMinMaxExpr:
		return "T_MinMaxExpr";
	case duckdb_libpgquery::T_PGSQLValueFunction:
		return "T_SQLValueFunction";
	case duckdb_libpgquery::T_PGXmlExpr:
		return "T_XmlExpr";
	case duckdb_libpgquery::T_PGNullTest:
		return "T_NullTest";
	case duckdb_libpgquery::T_PGBooleanTest:
		return "T_BooleanTest";
	case duckdb_libpgquery::T_PGCoerceToDomain:
		return "T_CoerceToDomain";
	case duckdb_libpgquery::T_PGCoerceToDomainValue:
		return "T_CoerceToDomainValue";
	case duckdb_libpgquery::T_PGSetToDefault:
		return "T_SetToDefault";
	case duckdb_libpgquery::T_PGCurrentOfExpr:
		return "T_CurrentOfExpr";
	case duckdb_libpgquery::T_PGNextValueExpr:
		return "T_NextValueExpr";
	case duckdb_libpgquery::T_PGInferenceElem:
		return "T_InferenceElem";
	case duckdb_libpgquery::T_PGTargetEntry:
		return "T_TargetEntry";
	case duckdb_libpgquery::T_PGRangeTblRef:
		return "T_RangeTblRef";
	case duckdb_libpgquery::T_PGJoinExpr:
		return "T_JoinExpr";
	case duckdb_libpgquery::T_PGFromExpr:
		return "T_FromExpr";
	case duckdb_libpgquery::T_PGOnConflictExpr:
		return "T_OnConflictExpr";
	case duckdb_libpgquery::T_PGIntoClause:
		return "T_IntoClause";
	case duckdb_libpgquery::T_PGExprState:
		return "T_ExprState";
	case duckdb_libpgquery::T_PGAggrefExprState:
		return "T_AggrefExprState";
	case duckdb_libpgquery::T_PGWindowFuncExprState:
		return "T_WindowFuncExprState";
	case duckdb_libpgquery::T_PGSetExprState:
		return "T_SetExprState";
	case duckdb_libpgquery::T_PGSubPlanState:
		return "T_SubPlanState";
	case duckdb_libpgquery::T_PGAlternativeSubPlanState:
		return "T_AlternativeSubPlanState";
	case duckdb_libpgquery::T_PGDomainConstraintState:
		return "T_DomainConstraintState";
	case duckdb_libpgquery::T_PGPlannerInfo:
		return "T_PlannerInfo";
	case duckdb_libpgquery::T_PGPlannerGlobal:
		return "T_PlannerGlobal";
	case duckdb_libpgquery::T_PGRelOptInfo:
		return "T_RelOptInfo";
	case duckdb_libpgquery::T_PGIndexOptInfo:
		return "T_IndexOptInfo";
	case duckdb_libpgquery::T_PGForeignKeyOptInfo:
		return "T_ForeignKeyOptInfo";
	case duckdb_libpgquery::T_PGParamPathInfo:
		return "T_ParamPathInfo";
	case duckdb_libpgquery::T_PGPath:
		return "T_Path";
	case duckdb_libpgquery::T_PGIndexPath:
		return "T_IndexPath";
	case duckdb_libpgquery::T_PGBitmapHeapPath:
		return "T_BitmapHeapPath";
	case duckdb_libpgquery::T_PGBitmapAndPath:
		return "T_BitmapAndPath";
	case duckdb_libpgquery::T_PGBitmapOrPath:
		return "T_BitmapOrPath";
	case duckdb_libpgquery::T_PGTidPath:
		return "T_TidPath";
	case duckdb_libpgquery::T_PGSubqueryScanPath:
		return "T_SubqueryScanPath";
	case duckdb_libpgquery::T_PGForeignPath:
		return "T_ForeignPath";
	case duckdb_libpgquery::T_PGCustomPath:
		return "T_CustomPath";
	case duckdb_libpgquery::T_PGNestPath:
		return "T_NestPath";
	case duckdb_libpgquery::T_PGMergePath:
		return "T_MergePath";
	case duckdb_libpgquery::T_PGHashPath:
		return "T_HashPath";
	case duckdb_libpgquery::T_PGAppendPath:
		return "T_AppendPath";
	case duckdb_libpgquery::T_PGMergeAppendPath:
		return "T_MergeAppendPath";
	case duckdb_libpgquery::T_PGResultPath:
		return "T_ResultPath";
	case duckdb_libpgquery::T_PGMaterialPath:
		return "T_MaterialPath";
	case duckdb_libpgquery::T_PGUniquePath:
		return "T_UniquePath";
	case duckdb_libpgquery::T_PGGatherPath:
		return "T_GatherPath";
	case duckdb_libpgquery::T_PGGatherMergePath:
		return "T_GatherMergePath";
	case duckdb_libpgquery::T_PGProjectionPath:
		return "T_ProjectionPath";
	case duckdb_libpgquery::T_PGProjectSetPath:
		return "T_ProjectSetPath";
	case duckdb_libpgquery::T_PGSortPath:
		return "T_SortPath";
	case duckdb_libpgquery::T_PGGroupPath:
		return "T_GroupPath";
	case duckdb_libpgquery::T_PGUpperUniquePath:
		return "T_UpperUniquePath";
	case duckdb_libpgquery::T_PGAggPath:
		return "T_AggPath";
	case duckdb_libpgquery::T_PGGroupingSetsPath:
		return "T_GroupingSetsPath";
	case duckdb_libpgquery::T_PGMinMaxAggPath:
		return "T_MinMaxAggPath";
	case duckdb_libpgquery::T_PGWindowAggPath:
		return "T_WindowAggPath";
	case duckdb_libpgquery::T_PGSetOpPath:
		return "T_SetOpPath";
	case duckdb_libpgquery::T_PGRecursiveUnionPath:
		return "T_RecursiveUnionPath";
	case duckdb_libpgquery::T_PGLockRowsPath:
		return "T_LockRowsPath";
	case duckdb_libpgquery::T_PGModifyTablePath:
		return "T_ModifyTablePath";
	case duckdb_libpgquery::T_PGLimitPath:
		return "T_LimitPath";
	case duckdb_libpgquery::T_PGEquivalenceClass:
		return "T_EquivalenceClass";
	case duckdb_libpgquery::T_PGEquivalenceMember:
		return "T_EquivalenceMember";
	case duckdb_libpgquery::T_PGPathKey:
		return "T_PathKey";
	case duckdb_libpgquery::T_PGPathTarget:
		return "T_PathTarget";
	case duckdb_libpgquery::T_PGRestrictInfo:
		return "T_RestrictInfo";
	case duckdb_libpgquery::T_PGPlaceHolderVar:
		return "T_PlaceHolderVar";
	case duckdb_libpgquery::T_PGSpecialJoinInfo:
		return "T_SpecialJoinInfo";
	case duckdb_libpgquery::T_PGAppendRelInfo:
		return "T_AppendRelInfo";
	case duckdb_libpgquery::T_PGPartitionedChildRelInfo:
		return "T_PartitionedChildRelInfo";
	case duckdb_libpgquery::T_PGPlaceHolderInfo:
		return "T_PlaceHolderInfo";
	case duckdb_libpgquery::T_PGMinMaxAggInfo:
		return "T_MinMaxAggInfo";
	case duckdb_libpgquery::T_PGPlannerParamItem:
		return "T_PlannerParamItem";
	case duckdb_libpgquery::T_PGRollupData:
		return "T_RollupData";
	case duckdb_libpgquery::T_PGGroupingSetData:
		return "T_GroupingSetData";
	case duckdb_libpgquery::T_PGStatisticExtInfo:
		return "T_StatisticExtInfo";
	case duckdb_libpgquery::T_PGMemoryContext:
		return "T_MemoryContext";
	case duckdb_libpgquery::T_PGAllocSetContext:
		return "T_AllocSetContext";
	case duckdb_libpgquery::T_PGSlabContext:
		return "T_SlabContext";
	case duckdb_libpgquery::T_PGValue:
		return "T_Value";
	case duckdb_libpgquery::T_PGInteger:
		return "T_Integer";
	case duckdb_libpgquery::T_PGFloat:
		return "T_Float";
	case duckdb_libpgquery::T_PGString:
		return "T_String";
	case duckdb_libpgquery::T_PGBitString:
		return "T_BitString";
	case duckdb_libpgquery::T_PGNull:
		return "T_Null";
	case duckdb_libpgquery::T_PGList:
		return "T_List";
	case duckdb_libpgquery::T_PGIntList:
		return "T_IntList";
	case duckdb_libpgquery::T_PGOidList:
		return "T_OidList";
	case duckdb_libpgquery::T_PGExtensibleNode:
		return "T_ExtensibleNode";
	case duckdb_libpgquery::T_PGRawStmt:
		return "T_RawStmt";
	case duckdb_libpgquery::T_PGQuery:
		return "T_Query";
	case duckdb_libpgquery::T_PGPlannedStmt:
		return "T_PlannedStmt";
	case duckdb_libpgquery::T_PGInsertStmt:
		return "T_InsertStmt";
	case duckdb_libpgquery::T_PGDeleteStmt:
		return "T_DeleteStmt";
	case duckdb_libpgquery::T_PGUpdateStmt:
		return "T_UpdateStmt";
	case duckdb_libpgquery::T_PGSelectStmt:
		return "T_SelectStmt";
	case duckdb_libpgquery::T_PGAlterTableStmt:
		return "T_AlterTableStmt";
	case duckdb_libpgquery::T_PGAlterTableCmd:
		return "T_AlterTableCmd";
	case duckdb_libpgquery::T_PGAlterDomainStmt:
		return "T_AlterDomainStmt";
	case duckdb_libpgquery::T_PGSetOperationStmt:
		return "T_SetOperationStmt";
	case duckdb_libpgquery::T_PGGrantStmt:
		return "T_GrantStmt";
	case duckdb_libpgquery::T_PGGrantRoleStmt:
		return "T_GrantRoleStmt";
	case duckdb_libpgquery::T_PGAlterDefaultPrivilegesStmt:
		return "T_AlterDefaultPrivilegesStmt";
	case duckdb_libpgquery::T_PGClosePortalStmt:
		return "T_ClosePortalStmt";
	case duckdb_libpgquery::T_PGClusterStmt:
		return "T_ClusterStmt";
	case duckdb_libpgquery::T_PGCopyStmt:
		return "T_CopyStmt";
	case duckdb_libpgquery::T_PGCreateStmt:
		return "T_CreateStmt";
	case duckdb_libpgquery::T_PGDefineStmt:
		return "T_DefineStmt";
	case duckdb_libpgquery::T_PGDropStmt:
		return "T_DropStmt";
	case duckdb_libpgquery::T_PGTruncateStmt:
		return "T_TruncateStmt";
	case duckdb_libpgquery::T_PGCommentStmt:
		return "T_CommentStmt";
	case duckdb_libpgquery::T_PGFetchStmt:
		return "T_FetchStmt";
	case duckdb_libpgquery::T_PGIndexStmt:
		return "T_IndexStmt";
	case duckdb_libpgquery::T_PGCreateFunctionStmt:
		return "T_CreateFunctionStmt";
	case duckdb_libpgquery::T_PGAlterFunctionStmt:
		return "T_AlterFunctionStmt";
	case duckdb_libpgquery::T_PGDoStmt:
		return "T_DoStmt";
	case duckdb_libpgquery::T_PGRenameStmt:
		return "T_RenameStmt";
	case duckdb_libpgquery::T_PGRuleStmt:
		return "T_RuleStmt";
	case duckdb_libpgquery::T_PGNotifyStmt:
		return "T_NotifyStmt";
	case duckdb_libpgquery::T_PGListenStmt:
		return "T_ListenStmt";
	case duckdb_libpgquery::T_PGUnlistenStmt:
		return "T_UnlistenStmt";
	case duckdb_libpgquery::T_PGTransactionStmt:
		return "T_TransactionStmt";
	case duckdb_libpgquery::T_PGViewStmt:
		return "T_ViewStmt";
	case duckdb_libpgquery::T_PGLoadStmt:
		return "T_LoadStmt";
	case duckdb_libpgquery::T_PGCreateDomainStmt:
		return "T_CreateDomainStmt";
	case duckdb_libpgquery::T_PGCreatedbStmt:
		return "T_CreatedbStmt";
	case duckdb_libpgquery::T_PGDropdbStmt:
		return "T_DropdbStmt";
	case duckdb_libpgquery::T_PGVacuumStmt:
		return "T_VacuumStmt";
	case duckdb_libpgquery::T_PGExplainStmt:
		return "T_ExplainStmt";
	case duckdb_libpgquery::T_PGCreateTableAsStmt:
		return "T_CreateTableAsStmt";
	case duckdb_libpgquery::T_PGCreateSeqStmt:
		return "T_CreateSeqStmt";
	case duckdb_libpgquery::T_PGAlterSeqStmt:
		return "T_AlterSeqStmt";
	case duckdb_libpgquery::T_PGVariableSetStmt:
		return "T_VariableSetStmt";
	case duckdb_libpgquery::T_PGVariableShowStmt:
		return "T_VariableShowStmt";
	case duckdb_libpgquery::T_PGVariableShowSelectStmt:
		return "T_VariableShowSelectStmt";
	case duckdb_libpgquery::T_PGDiscardStmt:
		return "T_DiscardStmt";
	case duckdb_libpgquery::T_PGCreateTrigStmt:
		return "T_CreateTrigStmt";
	case duckdb_libpgquery::T_PGCreatePLangStmt:
		return "T_CreatePLangStmt";
	case duckdb_libpgquery::T_PGCreateRoleStmt:
		return "T_CreateRoleStmt";
	case duckdb_libpgquery::T_PGAlterRoleStmt:
		return "T_AlterRoleStmt";
	case duckdb_libpgquery::T_PGDropRoleStmt:
		return "T_DropRoleStmt";
	case duckdb_libpgquery::T_PGLockStmt:
		return "T_LockStmt";
	case duckdb_libpgquery::T_PGConstraintsSetStmt:
		return "T_ConstraintsSetStmt";
	case duckdb_libpgquery::T_PGReindexStmt:
		return "T_ReindexStmt";
	case duckdb_libpgquery::T_PGCheckPointStmt:
		return "T_CheckPointStmt";
	case duckdb_libpgquery::T_PGCreateSchemaStmt:
		return "T_CreateSchemaStmt";
	case duckdb_libpgquery::T_PGAlterDatabaseStmt:
		return "T_AlterDatabaseStmt";
	case duckdb_libpgquery::T_PGAlterDatabaseSetStmt:
		return "T_AlterDatabaseSetStmt";
	case duckdb_libpgquery::T_PGAlterRoleSetStmt:
		return "T_AlterRoleSetStmt";
	case duckdb_libpgquery::T_PGCreateConversionStmt:
		return "T_CreateConversionStmt";
	case duckdb_libpgquery::T_PGCreateCastStmt:
		return "T_CreateCastStmt";
	case duckdb_libpgquery::T_PGCreateOpClassStmt:
		return "T_CreateOpClassStmt";
	case duckdb_libpgquery::T_PGCreateOpFamilyStmt:
		return "T_CreateOpFamilyStmt";
	case duckdb_libpgquery::T_PGAlterOpFamilyStmt:
		return "T_AlterOpFamilyStmt";
	case duckdb_libpgquery::T_PGPrepareStmt:
		return "T_PrepareStmt";
	case duckdb_libpgquery::T_PGExecuteStmt:
		return "T_ExecuteStmt";
	case duckdb_libpgquery::T_PGCallStmt:
		return "T_CallStmt";
	case duckdb_libpgquery::T_PGDeallocateStmt:
		return "T_DeallocateStmt";
	case duckdb_libpgquery::T_PGDeclareCursorStmt:
		return "T_DeclareCursorStmt";
	case duckdb_libpgquery::T_PGCreateTableSpaceStmt:
		return "T_CreateTableSpaceStmt";
	case duckdb_libpgquery::T_PGDropTableSpaceStmt:
		return "T_DropTableSpaceStmt";
	case duckdb_libpgquery::T_PGAlterObjectDependsStmt:
		return "T_AlterObjectDependsStmt";
	case duckdb_libpgquery::T_PGAlterObjectSchemaStmt:
		return "T_AlterObjectSchemaStmt";
	case duckdb_libpgquery::T_PGAlterOwnerStmt:
		return "T_AlterOwnerStmt";
	case duckdb_libpgquery::T_PGAlterOperatorStmt:
		return "T_AlterOperatorStmt";
	case duckdb_libpgquery::T_PGDropOwnedStmt:
		return "T_DropOwnedStmt";
	case duckdb_libpgquery::T_PGReassignOwnedStmt:
		return "T_ReassignOwnedStmt";
	case duckdb_libpgquery::T_PGCompositeTypeStmt:
		return "T_CompositeTypeStmt";
	case duckdb_libpgquery::T_PGCreateTypeStmt:
		return "T_CreateTypeStmt";
	case duckdb_libpgquery::T_PGCreateRangeStmt:
		return "T_CreateRangeStmt";
	case duckdb_libpgquery::T_PGAlterEnumStmt:
		return "T_AlterEnumStmt";
	case duckdb_libpgquery::T_PGAlterTSDictionaryStmt:
		return "T_AlterTSDictionaryStmt";
	case duckdb_libpgquery::T_PGAlterTSConfigurationStmt:
		return "T_AlterTSConfigurationStmt";
	case duckdb_libpgquery::T_PGCreateFdwStmt:
		return "T_CreateFdwStmt";
	case duckdb_libpgquery::T_PGAlterFdwStmt:
		return "T_AlterFdwStmt";
	case duckdb_libpgquery::T_PGCreateForeignServerStmt:
		return "T_CreateForeignServerStmt";
	case duckdb_libpgquery::T_PGAlterForeignServerStmt:
		return "T_AlterForeignServerStmt";
	case duckdb_libpgquery::T_PGCreateUserMappingStmt:
		return "T_CreateUserMappingStmt";
	case duckdb_libpgquery::T_PGAlterUserMappingStmt:
		return "T_AlterUserMappingStmt";
	case duckdb_libpgquery::T_PGDropUserMappingStmt:
		return "T_DropUserMappingStmt";
	case duckdb_libpgquery::T_PGAlterTableSpaceOptionsStmt:
		return "T_AlterTableSpaceOptionsStmt";
	case duckdb_libpgquery::T_PGAlterTableMoveAllStmt:
		return "T_AlterTableMoveAllStmt";
	case duckdb_libpgquery::T_PGSecLabelStmt:
		return "T_SecLabelStmt";
	case duckdb_libpgquery::T_PGCreateForeignTableStmt:
		return "T_CreateForeignTableStmt";
	case duckdb_libpgquery::T_PGImportForeignSchemaStmt:
		return "T_ImportForeignSchemaStmt";
	case duckdb_libpgquery::T_PGCreateExtensionStmt:
		return "T_CreateExtensionStmt";
	case duckdb_libpgquery::T_PGAlterExtensionStmt:
		return "T_AlterExtensionStmt";
	case duckdb_libpgquery::T_PGAlterExtensionContentsStmt:
		return "T_AlterExtensionContentsStmt";
	case duckdb_libpgquery::T_PGCreateEventTrigStmt:
		return "T_CreateEventTrigStmt";
	case duckdb_libpgquery::T_PGAlterEventTrigStmt:
		return "T_AlterEventTrigStmt";
	case duckdb_libpgquery::T_PGRefreshMatViewStmt:
		return "T_RefreshMatViewStmt";
	case duckdb_libpgquery::T_PGReplicaIdentityStmt:
		return "T_ReplicaIdentityStmt";
	case duckdb_libpgquery::T_PGAlterSystemStmt:
		return "T_AlterSystemStmt";
	case duckdb_libpgquery::T_PGCreatePolicyStmt:
		return "T_CreatePolicyStmt";
	case duckdb_libpgquery::T_PGAlterPolicyStmt:
		return "T_AlterPolicyStmt";
	case duckdb_libpgquery::T_PGCreateTransformStmt:
		return "T_CreateTransformStmt";
	case duckdb_libpgquery::T_PGCreateAmStmt:
		return "T_CreateAmStmt";
	case duckdb_libpgquery::T_PGCreatePublicationStmt:
		return "T_CreatePublicationStmt";
	case duckdb_libpgquery::T_PGAlterPublicationStmt:
		return "T_AlterPublicationStmt";
	case duckdb_libpgquery::T_PGCreateSubscriptionStmt:
		return "T_CreateSubscriptionStmt";
	case duckdb_libpgquery::T_PGAlterSubscriptionStmt:
		return "T_AlterSubscriptionStmt";
	case duckdb_libpgquery::T_PGDropSubscriptionStmt:
		return "T_DropSubscriptionStmt";
	case duckdb_libpgquery::T_PGCreateStatsStmt:
		return "T_CreateStatsStmt";
	case duckdb_libpgquery::T_PGAlterCollationStmt:
		return "T_AlterCollationStmt";
	case duckdb_libpgquery::T_PGAExpr:
		return "TAExpr";
	case duckdb_libpgquery::T_PGColumnRef:
		return "T_ColumnRef";
	case duckdb_libpgquery::T_PGParamRef:
		return "T_ParamRef";
	case duckdb_libpgquery::T_PGAConst:
		return "TAConst";
	case duckdb_libpgquery::T_PGFuncCall:
		return "T_FuncCall";
	case duckdb_libpgquery::T_PGAStar:
		return "TAStar";
	case duckdb_libpgquery::T_PGAIndices:
		return "TAIndices";
	case duckdb_libpgquery::T_PGAIndirection:
		return "TAIndirection";
	case duckdb_libpgquery::T_PGAArrayExpr:
		return "TAArrayExpr";
	case duckdb_libpgquery::T_PGResTarget:
		return "T_ResTarget";
	case duckdb_libpgquery::T_PGMultiAssignRef:
		return "T_MultiAssignRef";
	case duckdb_libpgquery::T_PGTypeCast:
		return "T_TypeCast";
	case duckdb_libpgquery::T_PGCollateClause:
		return "T_CollateClause";
	case duckdb_libpgquery::T_PGSortBy:
		return "T_SortBy";
	case duckdb_libpgquery::T_PGWindowDef:
		return "T_WindowDef";
	case duckdb_libpgquery::T_PGRangeSubselect:
		return "T_RangeSubselect";
	case duckdb_libpgquery::T_PGRangeFunction:
		return "T_RangeFunction";
	case duckdb_libpgquery::T_PGRangeTableSample:
		return "T_RangeTableSample";
	case duckdb_libpgquery::T_PGRangeTableFunc:
		return "T_RangeTableFunc";
	case duckdb_libpgquery::T_PGRangeTableFuncCol:
		return "T_RangeTableFuncCol";
	case duckdb_libpgquery::T_PGTypeName:
		return "T_TypeName";
	case duckdb_libpgquery::T_PGColumnDef:
		return "T_ColumnDef";
	case duckdb_libpgquery::T_PGIndexElem:
		return "T_IndexElem";
	case duckdb_libpgquery::T_PGConstraint:
		return "T_Constraint";
	case duckdb_libpgquery::T_PGDefElem:
		return "T_DefElem";
	case duckdb_libpgquery::T_PGRangeTblEntry:
		return "T_RangeTblEntry";
	case duckdb_libpgquery::T_PGRangeTblFunction:
		return "T_RangeTblFunction";
	case duckdb_libpgquery::T_PGTableSampleClause:
		return "T_TableSampleClause";
	case duckdb_libpgquery::T_PGWithCheckOption:
		return "T_WithCheckOption";
	case duckdb_libpgquery::T_PGSortGroupClause:
		return "T_SortGroupClause";
	case duckdb_libpgquery::T_PGGroupingSet:
		return "T_GroupingSet";
	case duckdb_libpgquery::T_PGWindowClause:
		return "T_WindowClause";
	case duckdb_libpgquery::T_PGObjectWithArgs:
		return "T_ObjectWithArgs";
	case duckdb_libpgquery::T_PGAccessPriv:
		return "T_AccessPriv";
	case duckdb_libpgquery::T_PGCreateOpClassItem:
		return "T_CreateOpClassItem";
	case duckdb_libpgquery::T_PGTableLikeClause:
		return "T_TableLikeClause";
	case duckdb_libpgquery::T_PGFunctionParameter:
		return "T_FunctionParameter";
	case duckdb_libpgquery::T_PGLockingClause:
		return "T_LockingClause";
	case duckdb_libpgquery::T_PGRowMarkClause:
		return "T_RowMarkClause";
	case duckdb_libpgquery::T_PGXmlSerialize:
		return "T_XmlSerialize";
	case duckdb_libpgquery::T_PGWithClause:
		return "T_WithClause";
	case duckdb_libpgquery::T_PGInferClause:
		return "T_InferClause";
	case duckdb_libpgquery::T_PGOnConflictClause:
		return "T_OnConflictClause";
	case duckdb_libpgquery::T_PGCommonTableExpr:
		return "T_CommonTableExpr";
	case duckdb_libpgquery::T_PGRoleSpec:
		return "T_RoleSpec";
	case duckdb_libpgquery::T_PGTriggerTransition:
		return "T_TriggerTransition";
	case duckdb_libpgquery::T_PGPartitionElem:
		return "T_PartitionElem";
	case duckdb_libpgquery::T_PGPartitionSpec:
		return "T_PartitionSpec";
	case duckdb_libpgquery::T_PGPartitionBoundSpec:
		return "T_PartitionBoundSpec";
	case duckdb_libpgquery::T_PGPartitionRangeDatum:
		return "T_PartitionRangeDatum";
	case duckdb_libpgquery::T_PGPartitionCmd:
		return "T_PartitionCmd";
	case duckdb_libpgquery::T_PGIdentifySystemCmd:
		return "T_IdentifySystemCmd";
	case duckdb_libpgquery::T_PGBaseBackupCmd:
		return "T_BaseBackupCmd";
	case duckdb_libpgquery::T_PGCreateReplicationSlotCmd:
		return "T_CreateReplicationSlotCmd";
	case duckdb_libpgquery::T_PGDropReplicationSlotCmd:
		return "T_DropReplicationSlotCmd";
	case duckdb_libpgquery::T_PGStartReplicationCmd:
		return "T_StartReplicationCmd";
	case duckdb_libpgquery::T_PGTimeLineHistoryCmd:
		return "T_TimeLineHistoryCmd";
	case duckdb_libpgquery::T_PGSQLCmd:
		return "T_SQLCmd";
	case duckdb_libpgquery::T_PGTriggerData:
		return "T_TriggerData";
	case duckdb_libpgquery::T_PGEventTriggerData:
		return "T_EventTriggerData";
	case duckdb_libpgquery::T_PGReturnSetInfo:
		return "T_ReturnSetInfo";
	case duckdb_libpgquery::T_PGWindowObjectData:
		return "T_WindowObjectData";
	case duckdb_libpgquery::T_PGTIDBitmap:
		return "T_TIDBitmap";
	case duckdb_libpgquery::T_PGInlineCodeBlock:
		return "T_InlineCodeBlock";
	case duckdb_libpgquery::T_PGFdwRoutine:
		return "T_FdwRoutine";
	case duckdb_libpgquery::T_PGIndexAmRoutine:
		return "T_IndexAmRoutine";
	case duckdb_libpgquery::T_PGTsmRoutine:
		return "T_TsmRoutine";
	case duckdb_libpgquery::T_PGForeignKeyCacheInfo:
		return "T_ForeignKeyCacheInfo";
	case duckdb_libpgquery::T_PGAttachStmt:
		return "T_PGAttachStmt";
	case duckdb_libpgquery::T_PGUseStmt:
		return "T_PGUseStmt";
	case duckdb_libpgquery::T_PGCreateDatabaseStmt:
		return "T_PGCreateDatabaseStmt";
	default:
		return "(UNKNOWN)";
	}
} // LCOV_EXCL_STOP

} // namespace duckdb


namespace duckdb {

vector<string> Transformer::TransformStringList(duckdb_libpgquery::PGList *list) {
	vector<string> result;
	if (!list) {
		return result;
	}
	for (auto node = list->head; node != nullptr; node = node->next) {
		result.emplace_back(reinterpret_cast<duckdb_libpgquery::PGValue *>(node->data.ptr_value)->val.str);
	}
	return result;
}

string Transformer::TransformAlias(duckdb_libpgquery::PGAlias *root, vector<string> &column_name_alias) {
	if (!root) {
		return "";
	}
	column_name_alias = TransformStringList(root->colnames);
	return root->aliasname;
}

} // namespace duckdb






namespace duckdb {

unique_ptr<CommonTableExpressionInfo> CommonTableExpressionInfo::Copy() {
	auto result = make_uniq<CommonTableExpressionInfo>();
	result->aliases = aliases;
	result->query = unique_ptr_cast<SQLStatement, SelectStatement>(query->Copy());
	return result;
}

void Transformer::ExtractCTEsRecursive(CommonTableExpressionMap &cte_map) {
	for (auto &cte_entry : stored_cte_map) {
		for (auto &entry : cte_entry->map) {
			auto found_entry = cte_map.map.find(entry.first);
			if (found_entry != cte_map.map.end()) {
				// entry already present - use top-most entry
				continue;
			}
			cte_map.map[entry.first] = entry.second->Copy();
		}
	}
	if (parent) {
		parent->ExtractCTEsRecursive(cte_map);
	}
}

void Transformer::TransformCTE(duckdb_libpgquery::PGWithClause *de_with_clause, CommonTableExpressionMap &cte_map) {
	// TODO: might need to update in case of future lawsuit
	D_ASSERT(de_with_clause);
	stored_cte_map.push_back(&cte_map);

	D_ASSERT(de_with_clause->ctes);
	for (auto cte_ele = de_with_clause->ctes->head; cte_ele != nullptr; cte_ele = cte_ele->next) {
		auto info = make_uniq<CommonTableExpressionInfo>();

		auto cte = reinterpret_cast<duckdb_libpgquery::PGCommonTableExpr *>(cte_ele->data.ptr_value);
		if (cte->aliascolnames) {
			for (auto node = cte->aliascolnames->head; node != nullptr; node = node->next) {
				info->aliases.emplace_back(
				    reinterpret_cast<duckdb_libpgquery::PGValue *>(node->data.ptr_value)->val.str);
			}
		}
		// lets throw some errors on unsupported features early
		if (cte->ctecolnames) {
			throw NotImplementedException("Column name setting not supported in CTEs");
		}
		if (cte->ctecoltypes) {
			throw NotImplementedException("Column type setting not supported in CTEs");
		}
		if (cte->ctecoltypmods) {
			throw NotImplementedException("Column type modification not supported in CTEs");
		}
		if (cte->ctecolcollations) {
			throw NotImplementedException("CTE collations not supported");
		}
		// we need a query
		if (!cte->ctequery || cte->ctequery->type != duckdb_libpgquery::T_PGSelectStmt) {
			throw NotImplementedException("A CTE needs a SELECT");
		}

		// CTE transformation can either result in inlining for non recursive CTEs, or in recursive CTE bindings
		// otherwise.
		if (cte->cterecursive || de_with_clause->recursive) {
			info->query = TransformRecursiveCTE(cte, *info);
		} else {
			Transformer cte_transformer(*this);
			info->query = cte_transformer.TransformSelect(cte->ctequery);
		}
		D_ASSERT(info->query);
		auto cte_name = string(cte->ctename);

		auto it = cte_map.map.find(cte_name);
		if (it != cte_map.map.end()) {
			// can't have two CTEs with same name
			throw ParserException("Duplicate CTE name \"%s\"", cte_name);
		}
		cte_map.map[cte_name] = std::move(info);
	}
}

unique_ptr<SelectStatement> Transformer::TransformRecursiveCTE(duckdb_libpgquery::PGCommonTableExpr *cte,
                                                               CommonTableExpressionInfo &info) {
	auto stmt = (duckdb_libpgquery::PGSelectStmt *)cte->ctequery;

	unique_ptr<SelectStatement> select;
	switch (stmt->op) {
	case duckdb_libpgquery::PG_SETOP_UNION:
	case duckdb_libpgquery::PG_SETOP_EXCEPT:
	case duckdb_libpgquery::PG_SETOP_INTERSECT: {
		select = make_uniq<SelectStatement>();
		select->node = make_uniq_base<QueryNode, RecursiveCTENode>();
		auto result = (RecursiveCTENode *)select->node.get();
		result->ctename = string(cte->ctename);
		result->union_all = stmt->all;
		result->left = TransformSelectNode(stmt->larg);
		result->right = TransformSelectNode(stmt->rarg);
		result->aliases = info.aliases;

		D_ASSERT(result->left);
		D_ASSERT(result->right);

		if (stmt->op != duckdb_libpgquery::PG_SETOP_UNION) {
			throw ParserException("Unsupported setop type for recursive CTE: only UNION or UNION ALL are supported");
		}
		break;
	}
	default:
		// This CTE is not recursive. Fallback to regular query transformation.
		return TransformSelect(cte->ctequery);
	}

	if (stmt->limitCount || stmt->limitOffset) {
		throw ParserException("LIMIT or OFFSET in a recursive query is not allowed");
	}
	if (stmt->sortClause) {
		throw ParserException("ORDER BY in a recursive query is not allowed");
	}
	return select;
}

} // namespace duckdb






namespace duckdb {

static void CheckGroupingSetMax(idx_t count) {
	static constexpr const idx_t MAX_GROUPING_SETS = 65535;
	if (count > MAX_GROUPING_SETS) {
		throw ParserException("Maximum grouping set count of %d exceeded", MAX_GROUPING_SETS);
	}
}

static void CheckGroupingSetCubes(idx_t current_count, idx_t cube_count) {
	idx_t combinations = 1;
	for (idx_t i = 0; i < cube_count; i++) {
		combinations *= 2;
		CheckGroupingSetMax(current_count + combinations);
	}
}

struct GroupingExpressionMap {
	parsed_expression_map_t<idx_t> map;
};

static GroupingSet VectorToGroupingSet(vector<idx_t> &indexes) {
	GroupingSet result;
	for (idx_t i = 0; i < indexes.size(); i++) {
		result.insert(indexes[i]);
	}
	return result;
}

static void MergeGroupingSet(GroupingSet &result, GroupingSet &other) {
	CheckGroupingSetMax(result.size() + other.size());
	result.insert(other.begin(), other.end());
}

void Transformer::AddGroupByExpression(unique_ptr<ParsedExpression> expression, GroupingExpressionMap &map,
                                       GroupByNode &result, vector<idx_t> &result_set) {
	if (expression->type == ExpressionType::FUNCTION) {
		auto &func = expression->Cast<FunctionExpression>();
		if (func.function_name == "row") {
			for (auto &child : func.children) {
				AddGroupByExpression(std::move(child), map, result, result_set);
			}
			return;
		}
	}
	auto entry = map.map.find(*expression);
	idx_t result_idx;
	if (entry == map.map.end()) {
		result_idx = result.group_expressions.size();
		map.map[*expression] = result_idx;
		result.group_expressions.push_back(std::move(expression));
	} else {
		result_idx = entry->second;
	}
	result_set.push_back(result_idx);
}

static void AddCubeSets(const GroupingSet &current_set, vector<GroupingSet> &result_set,
                        vector<GroupingSet> &result_sets, idx_t start_idx = 0) {
	CheckGroupingSetMax(result_sets.size());
	result_sets.push_back(current_set);
	for (idx_t k = start_idx; k < result_set.size(); k++) {
		auto child_set = current_set;
		MergeGroupingSet(child_set, result_set[k]);
		AddCubeSets(child_set, result_set, result_sets, k + 1);
	}
}

void Transformer::TransformGroupByExpression(duckdb_libpgquery::PGNode *n, GroupingExpressionMap &map,
                                             GroupByNode &result, vector<idx_t> &indexes) {
	auto expression = TransformExpression(n);
	AddGroupByExpression(std::move(expression), map, result, indexes);
}

// If one GROUPING SETS clause is nested inside another,
// the effect is the same as if all the elements of the inner clause had been written directly in the outer clause.
void Transformer::TransformGroupByNode(duckdb_libpgquery::PGNode *n, GroupingExpressionMap &map, SelectNode &result,
                                       vector<GroupingSet> &result_sets) {
	if (n->type == duckdb_libpgquery::T_PGGroupingSet) {
		auto grouping_set = (duckdb_libpgquery::PGGroupingSet *)n;
		switch (grouping_set->kind) {
		case duckdb_libpgquery::GROUPING_SET_EMPTY:
			result_sets.emplace_back();
			break;
		case duckdb_libpgquery::GROUPING_SET_ALL: {
			result.aggregate_handling = AggregateHandling::FORCE_AGGREGATES;
			break;
		}
		case duckdb_libpgquery::GROUPING_SET_SETS: {
			for (auto node = grouping_set->content->head; node; node = node->next) {
				auto pg_node = (duckdb_libpgquery::PGNode *)node->data.ptr_value;
				TransformGroupByNode(pg_node, map, result, result_sets);
			}
			break;
		}
		case duckdb_libpgquery::GROUPING_SET_ROLLUP: {
			vector<GroupingSet> rollup_sets;
			for (auto node = grouping_set->content->head; node; node = node->next) {
				auto pg_node = (duckdb_libpgquery::PGNode *)node->data.ptr_value;
				vector<idx_t> rollup_set;
				TransformGroupByExpression(pg_node, map, result.groups, rollup_set);
				rollup_sets.push_back(VectorToGroupingSet(rollup_set));
			}
			// generate the subsets of the rollup set and add them to the grouping sets
			GroupingSet current_set;
			result_sets.push_back(current_set);
			for (idx_t i = 0; i < rollup_sets.size(); i++) {
				MergeGroupingSet(current_set, rollup_sets[i]);
				result_sets.push_back(current_set);
			}
			break;
		}
		case duckdb_libpgquery::GROUPING_SET_CUBE: {
			vector<GroupingSet> cube_sets;
			for (auto node = grouping_set->content->head; node; node = node->next) {
				auto pg_node = (duckdb_libpgquery::PGNode *)node->data.ptr_value;
				vector<idx_t> cube_set;
				TransformGroupByExpression(pg_node, map, result.groups, cube_set);
				cube_sets.push_back(VectorToGroupingSet(cube_set));
			}
			// generate the subsets of the rollup set and add them to the grouping sets
			CheckGroupingSetCubes(result_sets.size(), cube_sets.size());

			GroupingSet current_set;
			AddCubeSets(current_set, cube_sets, result_sets, 0);
			break;
		}
		default:
			throw InternalException("Unsupported GROUPING SET type %d", grouping_set->kind);
		}
	} else {
		vector<idx_t> indexes;
		TransformGroupByExpression(n, map, result.groups, indexes);
		result_sets.push_back(VectorToGroupingSet(indexes));
	}
}

// If multiple grouping items are specified in a single GROUP BY clause,
// then the final list of grouping sets is the cross product of the individual items.
bool Transformer::TransformGroupBy(duckdb_libpgquery::PGList *group, SelectNode &select_node) {
	if (!group) {
		return false;
	}
	auto &result = select_node.groups;
	GroupingExpressionMap map;
	for (auto node = group->head; node != nullptr; node = node->next) {
		auto n = reinterpret_cast<duckdb_libpgquery::PGNode *>(node->data.ptr_value);
		vector<GroupingSet> result_sets;
		TransformGroupByNode(n, map, select_node, result_sets);
		CheckGroupingSetMax(result_sets.size());
		if (result.grouping_sets.empty()) {
			// no grouping sets yet: use the current set of grouping sets
			result.grouping_sets = std::move(result_sets);
		} else {
			// compute the cross product
			vector<GroupingSet> new_sets;
			idx_t grouping_set_count = result.grouping_sets.size() * result_sets.size();
			CheckGroupingSetMax(grouping_set_count);
			new_sets.reserve(grouping_set_count);
			for (idx_t current_idx = 0; current_idx < result.grouping_sets.size(); current_idx++) {
				auto &current_set = result.grouping_sets[current_idx];
				for (idx_t new_idx = 0; new_idx < result_sets.size(); new_idx++) {
					auto &new_set = result_sets[new_idx];
					GroupingSet set;
					set.insert(current_set.begin(), current_set.end());
					set.insert(new_set.begin(), new_set.end());
					new_sets.push_back(std::move(set));
				}
			}
			result.grouping_sets = std::move(new_sets);
		}
	}
	if (result.group_expressions.size() == 1 && result.grouping_sets.size() == 1 &&
	    ExpressionIsEmptyStar(*result.group_expressions[0])) {
		// GROUP BY *
		result.group_expressions.clear();
		result.grouping_sets.clear();
		select_node.aggregate_handling = AggregateHandling::FORCE_AGGREGATES;
	}
	return true;
}

} // namespace duckdb





namespace duckdb {

bool Transformer::TransformOrderBy(duckdb_libpgquery::PGList *order, vector<OrderByNode> &result) {
	if (!order) {
		return false;
	}

	for (auto node = order->head; node != nullptr; node = node->next) {
		auto temp = reinterpret_cast<duckdb_libpgquery::PGNode *>(node->data.ptr_value);
		if (temp->type == duckdb_libpgquery::T_PGSortBy) {
			OrderType type;
			OrderByNullType null_order;
			auto sort = reinterpret_cast<duckdb_libpgquery::PGSortBy *>(temp);
			auto target = sort->node;
			if (sort->sortby_dir == duckdb_libpgquery::PG_SORTBY_DEFAULT) {
				type = OrderType::ORDER_DEFAULT;
			} else if (sort->sortby_dir == duckdb_libpgquery::PG_SORTBY_ASC) {
				type = OrderType::ASCENDING;
			} else if (sort->sortby_dir == duckdb_libpgquery::PG_SORTBY_DESC) {
				type = OrderType::DESCENDING;
			} else {
				throw NotImplementedException("Unimplemented order by type");
			}
			if (sort->sortby_nulls == duckdb_libpgquery::PG_SORTBY_NULLS_DEFAULT) {
				null_order = OrderByNullType::ORDER_DEFAULT;
			} else if (sort->sortby_nulls == duckdb_libpgquery::PG_SORTBY_NULLS_FIRST) {
				null_order = OrderByNullType::NULLS_FIRST;
			} else if (sort->sortby_nulls == duckdb_libpgquery::PG_SORTBY_NULLS_LAST) {
				null_order = OrderByNullType::NULLS_LAST;
			} else {
				throw NotImplementedException("Unimplemented order by type");
			}
			auto order_expression = TransformExpression(target);
			result.emplace_back(type, null_order, std::move(order_expression));
		} else {
			throw NotImplementedException("ORDER BY list member type %d\n", temp->type);
		}
	}
	return true;
}

} // namespace duckdb





namespace duckdb {

static SampleMethod GetSampleMethod(const string &method) {
	auto lmethod = StringUtil::Lower(method);
	if (lmethod == "system") {
		return SampleMethod::SYSTEM_SAMPLE;
	} else if (lmethod == "bernoulli") {
		return SampleMethod::BERNOULLI_SAMPLE;
	} else if (lmethod == "reservoir") {
		return SampleMethod::RESERVOIR_SAMPLE;
	} else {
		throw ParserException("Unrecognized sampling method %s, expected system, bernoulli or reservoir", method);
	}
}

unique_ptr<SampleOptions> Transformer::TransformSampleOptions(duckdb_libpgquery::PGNode *options) {
	if (!options) {
		return nullptr;
	}
	auto result = make_uniq<SampleOptions>();
	auto &sample_options = (duckdb_libpgquery::PGSampleOptions &)*options;
	auto &sample_size = (duckdb_libpgquery::PGSampleSize &)*sample_options.sample_size;
	auto sample_value = TransformValue(sample_size.sample_size)->value;
	result->is_percentage = sample_size.is_percentage;
	if (sample_size.is_percentage) {
		// sample size is given in sample_size: use system sampling
		auto percentage = sample_value.GetValue<double>();
		if (percentage < 0 || percentage > 100) {
			throw ParserException("Sample sample_size %llf out of range, must be between 0 and 100", percentage);
		}
		result->sample_size = Value::DOUBLE(percentage);
		result->method = SampleMethod::SYSTEM_SAMPLE;
	} else {
		// sample size is given in rows: use reservoir sampling
		auto rows = sample_value.GetValue<int64_t>();
		if (rows < 0) {
			throw ParserException("Sample rows %lld out of range, must be bigger than or equal to 0", rows);
		}
		result->sample_size = Value::BIGINT(rows);
		result->method = SampleMethod::RESERVOIR_SAMPLE;
	}
	if (sample_options.method) {
		result->method = GetSampleMethod(sample_options.method);
	}
	if (sample_options.has_seed) {
		result->seed = sample_options.seed;
	}
	return result;
}

} // namespace duckdb







namespace duckdb {

LogicalType Transformer::TransformTypeName(duckdb_libpgquery::PGTypeName *type_name) {
	if (!type_name || type_name->type != duckdb_libpgquery::T_PGTypeName) {
		throw ParserException("Expected a type");
	}
	auto stack_checker = StackCheck();

	auto name = (reinterpret_cast<duckdb_libpgquery::PGValue *>(type_name->names->tail->data.ptr_value)->val.str);
	// transform it to the SQL type
	LogicalTypeId base_type = TransformStringToLogicalTypeId(name);

	LogicalType result_type;
	if (base_type == LogicalTypeId::LIST) {
		throw ParserException("LIST is not valid as a stand-alone type");
	} else if (base_type == LogicalTypeId::ENUM) {
		throw ParserException("ENUM is not valid as a stand-alone type");
	} else if (base_type == LogicalTypeId::STRUCT) {
		if (!type_name->typmods || type_name->typmods->length == 0) {
			throw ParserException("Struct needs a name and entries");
		}
		child_list_t<LogicalType> children;
		case_insensitive_set_t name_collision_set;

		for (auto node = type_name->typmods->head; node; node = node->next) {
			auto &type_val = *((duckdb_libpgquery::PGList *)node->data.ptr_value);
			if (type_val.length != 2) {
				throw ParserException("Struct entry needs an entry name and a type name");
			}

			auto entry_name_node = (duckdb_libpgquery::PGValue *)(type_val.head->data.ptr_value);
			D_ASSERT(entry_name_node->type == duckdb_libpgquery::T_PGString);
			auto entry_type_node = (duckdb_libpgquery::PGValue *)(type_val.tail->data.ptr_value);
			D_ASSERT(entry_type_node->type == duckdb_libpgquery::T_PGTypeName);

			auto entry_name = string(entry_name_node->val.str);
			D_ASSERT(!entry_name.empty());

			if (name_collision_set.find(entry_name) != name_collision_set.end()) {
				throw ParserException("Duplicate struct entry name \"%s\"", entry_name);
			}
			name_collision_set.insert(entry_name);

			auto entry_type = TransformTypeName((duckdb_libpgquery::PGTypeName *)entry_type_node);
			children.push_back(make_pair(entry_name, entry_type));
		}
		D_ASSERT(!children.empty());
		result_type = LogicalType::STRUCT(children);

	} else if (base_type == LogicalTypeId::MAP) {

		if (!type_name->typmods || type_name->typmods->length != 2) {
			throw ParserException("Map type needs exactly two entries, key and value type");
		}
		auto key_type = TransformTypeName((duckdb_libpgquery::PGTypeName *)type_name->typmods->head->data.ptr_value);
		auto value_type = TransformTypeName((duckdb_libpgquery::PGTypeName *)type_name->typmods->tail->data.ptr_value);

		result_type = LogicalType::MAP(std::move(key_type), std::move(value_type));
	} else if (base_type == LogicalTypeId::UNION) {
		if (!type_name->typmods || type_name->typmods->length == 0) {
			throw ParserException("Union type needs at least one member");
		}
		if (type_name->typmods->length > (int)UnionType::MAX_UNION_MEMBERS) {
			throw ParserException("Union types can have at most %d members", UnionType::MAX_UNION_MEMBERS);
		}

		child_list_t<LogicalType> children;
		case_insensitive_set_t name_collision_set;

		for (auto node = type_name->typmods->head; node; node = node->next) {
			auto &type_val = *((duckdb_libpgquery::PGList *)node->data.ptr_value);
			if (type_val.length != 2) {
				throw ParserException("Union type member needs a tag name and a type name");
			}

			auto entry_name_node = (duckdb_libpgquery::PGValue *)(type_val.head->data.ptr_value);
			D_ASSERT(entry_name_node->type == duckdb_libpgquery::T_PGString);
			auto entry_type_node = (duckdb_libpgquery::PGValue *)(type_val.tail->data.ptr_value);
			D_ASSERT(entry_type_node->type == duckdb_libpgquery::T_PGTypeName);

			auto entry_name = string(entry_name_node->val.str);
			D_ASSERT(!entry_name.empty());

			if (name_collision_set.find(entry_name) != name_collision_set.end()) {
				throw ParserException("Duplicate union type tag name \"%s\"", entry_name);
			}

			name_collision_set.insert(entry_name);

			auto entry_type = TransformTypeName((duckdb_libpgquery::PGTypeName *)entry_type_node);
			children.push_back(make_pair(entry_name, entry_type));
		}
		D_ASSERT(!children.empty());
		result_type = LogicalType::UNION(std::move(children));
	} else {
		int64_t width, scale;
		if (base_type == LogicalTypeId::DECIMAL) {
			// default decimal width/scale
			width = 18;
			scale = 3;
		} else {
			width = 0;
			scale = 0;
		}
		// check any modifiers
		int modifier_idx = 0;
		if (type_name->typmods) {
			for (auto node = type_name->typmods->head; node; node = node->next) {
				auto &const_val = *((duckdb_libpgquery::PGAConst *)node->data.ptr_value);
				if (const_val.type != duckdb_libpgquery::T_PGAConst ||
				    const_val.val.type != duckdb_libpgquery::T_PGInteger) {
					throw ParserException("Expected an integer constant as type modifier");
				}
				if (const_val.val.val.ival < 0) {
					throw ParserException("Negative modifier not supported");
				}
				if (modifier_idx == 0) {
					width = const_val.val.val.ival;
					if (base_type == LogicalTypeId::BIT && const_val.location != -1) {
						width = 0;
					}
				} else if (modifier_idx == 1) {
					scale = const_val.val.val.ival;
				} else {
					throw ParserException("A maximum of two modifiers is supported");
				}
				modifier_idx++;
			}
		}
		switch (base_type) {
		case LogicalTypeId::VARCHAR:
			if (modifier_idx > 1) {
				throw ParserException("VARCHAR only supports a single modifier");
			}
			// FIXME: create CHECK constraint based on varchar width
			width = 0;
			result_type = LogicalType::VARCHAR;
			break;
		case LogicalTypeId::DECIMAL:
			if (modifier_idx == 1) {
				// only width is provided: set scale to 0
				scale = 0;
			}
			if (width <= 0 || width > Decimal::MAX_WIDTH_DECIMAL) {
				throw ParserException("Width must be between 1 and %d!", (int)Decimal::MAX_WIDTH_DECIMAL);
			}
			if (scale > width) {
				throw ParserException("Scale cannot be bigger than width");
			}
			result_type = LogicalType::DECIMAL(width, scale);
			break;
		case LogicalTypeId::INTERVAL:
			if (modifier_idx > 1) {
				throw ParserException("INTERVAL only supports a single modifier");
			}
			width = 0;
			result_type = LogicalType::INTERVAL;
			break;
		case LogicalTypeId::USER: {
			string user_type_name {name};
			result_type = LogicalType::USER(user_type_name);
			break;
		}
		case LogicalTypeId::BIT: {
			if (!width && type_name->typmods) {
				throw ParserException("Type %s does not support any modifiers!", LogicalType(base_type).ToString());
			}
			result_type = LogicalType(base_type);
			break;
		}
		case LogicalTypeId::TIMESTAMP:
			if (modifier_idx == 0) {
				result_type = LogicalType::TIMESTAMP;
			} else {
				if (modifier_idx > 1) {
					throw ParserException("TIMESTAMP only supports a single modifier");
				}
				if (width > 10) {
					throw ParserException("TIMESTAMP only supports until nano-second precision (9)");
				}
				if (width == 0) {
					result_type = LogicalType::TIMESTAMP_S;
				} else if (width <= 3) {
					result_type = LogicalType::TIMESTAMP_MS;
				} else if (width <= 6) {
					result_type = LogicalType::TIMESTAMP;
				} else {
					result_type = LogicalType::TIMESTAMP_NS;
				}
			}
			break;
		default:
			if (modifier_idx > 0) {
				throw ParserException("Type %s does not support any modifiers!", LogicalType(base_type).ToString());
			}
			result_type = LogicalType(base_type);
			break;
		}
	}
	if (type_name->arrayBounds) {
		// array bounds: turn the type into a list
		idx_t extra_stack = 0;
		for (auto cell = type_name->arrayBounds->head; cell != nullptr; cell = cell->next) {
			result_type = LogicalType::LIST(result_type);
			StackCheck(extra_stack++);
		}
	}
	return result_type;
}

} // namespace duckdb







namespace duckdb {

unique_ptr<AlterStatement> Transformer::TransformAlterSequence(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGAlterSeqStmt *>(node);
	D_ASSERT(stmt);
	auto result = make_uniq<AlterStatement>();

	auto qname = TransformQualifiedName(stmt->sequence);
	auto sequence_catalog = qname.catalog;
	auto sequence_schema = qname.schema;
	auto sequence_name = qname.name;

	if (!stmt->options) {
		throw InternalException("Expected an argument for ALTER SEQUENCE.");
	}

	unordered_set<SequenceInfo, EnumClassHash> used;
	duckdb_libpgquery::PGListCell *cell = nullptr;
	for_each_cell(cell, stmt->options->head) {
		auto *def_elem = reinterpret_cast<duckdb_libpgquery::PGDefElem *>(cell->data.ptr_value);
		string opt_name = string(def_elem->defname);

		if (opt_name == "owned_by") {
			if (used.find(SequenceInfo::SEQ_OWN) != used.end()) {
				throw ParserException("Owned by value should be passed as most once");
			}
			used.insert(SequenceInfo::SEQ_OWN);

			auto val = (duckdb_libpgquery::PGValue *)def_elem->arg;
			if (!val) {
				throw InternalException("Expected an argument for option %s", opt_name);
			}
			D_ASSERT(val);
			if (val->type != duckdb_libpgquery::T_PGList) {
				throw InternalException("Expected a string argument for option %s", opt_name);
			}
			auto opt_values = vector<string>();

			auto opt_value_list = (duckdb_libpgquery::PGList *)(val);
			for (auto c = opt_value_list->head; c != nullptr; c = lnext(c)) {
				auto target = (duckdb_libpgquery::PGResTarget *)(c->data.ptr_value);
				opt_values.emplace_back(target->name);
			}
			D_ASSERT(!opt_values.empty());
			string owner_schema = INVALID_SCHEMA;
			string owner_name;
			if (opt_values.size() == 2) {
				owner_schema = opt_values[0];
				owner_name = opt_values[1];
			} else if (opt_values.size() == 1) {
				owner_schema = DEFAULT_SCHEMA;
				owner_name = opt_values[0];
			} else {
				throw InternalException("Wrong argument for %s. Expected either <schema>.<name> or <name>", opt_name);
			}
			auto info = make_uniq<ChangeOwnershipInfo>(CatalogType::SEQUENCE_ENTRY, sequence_catalog, sequence_schema,
			                                           sequence_name, owner_schema, owner_name,
			                                           TransformOnEntryNotFound(stmt->missing_ok));
			result->info = std::move(info);
		} else {
			throw NotImplementedException("ALTER SEQUENCE option not supported yet!");
		}
	}
	result->info->if_not_found = TransformOnEntryNotFound(stmt->missing_ok);
	return result;
}
} // namespace duckdb






namespace duckdb {

OnEntryNotFound Transformer::TransformOnEntryNotFound(bool missing_ok) {
	return missing_ok ? OnEntryNotFound::RETURN_NULL : OnEntryNotFound::THROW_EXCEPTION;
}

unique_ptr<AlterStatement> Transformer::TransformAlter(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGAlterTableStmt *>(node);
	D_ASSERT(stmt);
	D_ASSERT(stmt->relation);

	if (stmt->cmds->length != 1) {
		throw ParserException("Only one ALTER command per statement is supported");
	}

	auto result = make_uniq<AlterStatement>();
	auto qname = TransformQualifiedName(stmt->relation);

	// first we check the type of ALTER
	for (auto c = stmt->cmds->head; c != nullptr; c = c->next) {
		auto command = reinterpret_cast<duckdb_libpgquery::PGAlterTableCmd *>(lfirst(c));
		AlterEntryData data(qname.catalog, qname.schema, qname.name, TransformOnEntryNotFound(stmt->missing_ok));
		// TODO: Include more options for command->subtype
		switch (command->subtype) {
		case duckdb_libpgquery::PG_AT_AddColumn: {
			auto cdef = (duckdb_libpgquery::PGColumnDef *)command->def;

			if (stmt->relkind != duckdb_libpgquery::PG_OBJECT_TABLE) {
				throw ParserException("Adding columns is only supported for tables");
			}
			if (cdef->category == duckdb_libpgquery::COL_GENERATED) {
				throw ParserException("Adding generated columns after table creation is not supported yet");
			}
			auto centry = TransformColumnDefinition(cdef);

			if (cdef->constraints) {
				for (auto constr = cdef->constraints->head; constr != nullptr; constr = constr->next) {
					auto constraint = TransformConstraint(constr, centry, 0);
					if (!constraint) {
						continue;
					}
					throw ParserException("Adding columns with constraints not yet supported");
				}
			}
			result->info = make_uniq<AddColumnInfo>(std::move(data), std::move(centry), command->missing_ok);
			break;
		}
		case duckdb_libpgquery::PG_AT_DropColumn: {
			bool cascade = command->behavior == duckdb_libpgquery::PG_DROP_CASCADE;

			if (stmt->relkind != duckdb_libpgquery::PG_OBJECT_TABLE) {
				throw ParserException("Dropping columns is only supported for tables");
			}
			result->info = make_uniq<RemoveColumnInfo>(std::move(data), command->name, command->missing_ok, cascade);
			break;
		}
		case duckdb_libpgquery::PG_AT_ColumnDefault: {
			auto expr = TransformExpression(command->def);

			if (stmt->relkind != duckdb_libpgquery::PG_OBJECT_TABLE) {
				throw ParserException("Alter column's default is only supported for tables");
			}
			result->info = make_uniq<SetDefaultInfo>(std::move(data), command->name, std::move(expr));
			break;
		}
		case duckdb_libpgquery::PG_AT_AlterColumnType: {
			auto cdef = (duckdb_libpgquery::PGColumnDef *)command->def;
			auto column_definition = TransformColumnDefinition(cdef);
			unique_ptr<ParsedExpression> expr;

			if (stmt->relkind != duckdb_libpgquery::PG_OBJECT_TABLE) {
				throw ParserException("Alter column's type is only supported for tables");
			}
			if (cdef->raw_default) {
				expr = TransformExpression(cdef->raw_default);
			} else {
				auto colref = make_uniq<ColumnRefExpression>(command->name);
				expr = make_uniq<CastExpression>(column_definition.Type(), std::move(colref));
			}
			result->info = make_uniq<ChangeColumnTypeInfo>(std::move(data), command->name, column_definition.Type(),
			                                               std::move(expr));
			break;
		}
		case duckdb_libpgquery::PG_AT_SetNotNull: {
			result->info = make_uniq<SetNotNullInfo>(std::move(data), command->name);
			break;
		}
		case duckdb_libpgquery::PG_AT_DropNotNull: {
			result->info = make_uniq<DropNotNullInfo>(std::move(data), command->name);
			break;
		}
		case duckdb_libpgquery::PG_AT_DropConstraint:
		default:
			throw NotImplementedException("ALTER TABLE option not supported yet!");
		}
	}

	return result;
}

} // namespace duckdb





namespace duckdb {

unique_ptr<AttachStatement> Transformer::TransformAttach(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGAttachStmt *>(node);
	auto result = make_uniq<AttachStatement>();
	auto info = make_uniq<AttachInfo>();
	info->name = stmt->name ? stmt->name : string();
	info->path = stmt->path;

	if (stmt->options) {
		duckdb_libpgquery::PGListCell *cell = nullptr;
		for_each_cell(cell, stmt->options->head) {
			auto *def_elem = reinterpret_cast<duckdb_libpgquery::PGDefElem *>(cell->data.ptr_value);
			Value val;
			if (def_elem->arg) {
				val = TransformValue(*((duckdb_libpgquery::PGValue *)def_elem->arg))->value;
			} else {
				val = Value::BOOLEAN(true);
			}
			info->options[StringUtil::Lower(def_elem->defname)] = std::move(val);
		}
	}
	result->info = std::move(info);
	return result;
}

} // namespace duckdb



namespace duckdb {

unique_ptr<CallStatement> Transformer::TransformCall(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGCallStmt *>(node);
	D_ASSERT(stmt);

	auto result = make_uniq<CallStatement>();
	result->function = TransformFuncCall((duckdb_libpgquery::PGFuncCall *)stmt->func);
	return result;
}

} // namespace duckdb





namespace duckdb {

unique_ptr<SQLStatement> Transformer::TransformCheckpoint(duckdb_libpgquery::PGNode *node) {
	auto checkpoint = (duckdb_libpgquery::PGCheckPointStmt *)node;

	vector<unique_ptr<ParsedExpression>> children;
	// transform into "CALL checkpoint()" or "CALL force_checkpoint()"
	auto checkpoint_name = checkpoint->force ? "force_checkpoint" : "checkpoint";
	auto result = make_uniq<CallStatement>();
	auto function = make_uniq<FunctionExpression>(checkpoint_name, std::move(children));
	if (checkpoint->name) {
		function->children.push_back(make_uniq<ConstantExpression>(Value(checkpoint->name)));
	}
	result->function = std::move(function);
	return std::move(result);
}

} // namespace duckdb







#include <cstring>

namespace duckdb {

void Transformer::TransformCopyOptions(CopyInfo &info, duckdb_libpgquery::PGList *options) {
	if (!options) {
		return;
	}
	duckdb_libpgquery::PGListCell *cell = nullptr;

	// iterate over each option
	for_each_cell(cell, options->head) {
		auto *def_elem = reinterpret_cast<duckdb_libpgquery::PGDefElem *>(cell->data.ptr_value);
		if (StringUtil::Lower(def_elem->defname) == "format") {
			// format specifier: interpret this option
			auto *format_val = (duckdb_libpgquery::PGValue *)(def_elem->arg);
			if (!format_val || format_val->type != duckdb_libpgquery::T_PGString) {
				throw ParserException("Unsupported parameter type for FORMAT: expected e.g. FORMAT 'csv', 'parquet'");
			}
			info.format = StringUtil::Lower(format_val->val.str);
			continue;
		}
		// otherwise
		if (info.options.find(def_elem->defname) != info.options.end()) {
			throw ParserException("Unexpected duplicate option \"%s\"", def_elem->defname);
		}
		if (!def_elem->arg) {
			info.options[def_elem->defname] = vector<Value>();
			continue;
		}
		switch (def_elem->arg->type) {
		case duckdb_libpgquery::T_PGList: {
			auto column_list = (duckdb_libpgquery::PGList *)(def_elem->arg);
			for (auto c = column_list->head; c != nullptr; c = lnext(c)) {
				auto target = (duckdb_libpgquery::PGResTarget *)(c->data.ptr_value);
				info.options[def_elem->defname].push_back(Value(target->name));
			}
			break;
		}
		case duckdb_libpgquery::T_PGAStar:
			info.options[def_elem->defname].push_back(Value("*"));
			break;
		default:
			info.options[def_elem->defname].push_back(
			    TransformValue(*((duckdb_libpgquery::PGValue *)def_elem->arg))->value);
			break;
		}
	}
}

unique_ptr<CopyStatement> Transformer::TransformCopy(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGCopyStmt *>(node);
	D_ASSERT(stmt);
	auto result = make_uniq<CopyStatement>();
	auto &info = *result->info;

	// get file_path and is_from
	info.is_from = stmt->is_from;
	if (!stmt->filename) {
		// stdin/stdout
		info.file_path = info.is_from ? "/dev/stdin" : "/dev/stdout";
	} else {
		// copy to a file
		info.file_path = stmt->filename;
	}
	if (StringUtil::EndsWith(info.file_path, ".parquet")) {
		info.format = "parquet";
	} else if (StringUtil::EndsWith(info.file_path, ".json") || StringUtil::EndsWith(info.file_path, ".ndjson")) {
		info.format = "json";
	} else {
		info.format = "csv";
	}

	// get select_list
	if (stmt->attlist) {
		for (auto n = stmt->attlist->head; n != nullptr; n = n->next) {
			auto target = reinterpret_cast<duckdb_libpgquery::PGResTarget *>(n->data.ptr_value);
			if (target->name) {
				info.select_list.emplace_back(target->name);
			}
		}
	}

	if (stmt->relation) {
		auto ref = TransformRangeVar(stmt->relation);
		auto &table = *reinterpret_cast<BaseTableRef *>(ref.get());
		info.table = table.table_name;
		info.schema = table.schema_name;
		info.catalog = table.catalog_name;
	} else {
		result->select_statement = TransformSelectNode((duckdb_libpgquery::PGSelectStmt *)stmt->query);
	}

	// handle the different options of the COPY statement
	TransformCopyOptions(info, stmt->options);

	return result;
}

} // namespace duckdb






namespace duckdb {

unique_ptr<CreateStatement> Transformer::TransformCreateDatabase(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGCreateDatabaseStmt *>(node);
	auto result = make_uniq<CreateStatement>();
	auto info = make_uniq<CreateDatabaseInfo>();

	info->path = stmt->path ? stmt->path : string();

	auto qualified_name = TransformQualifiedName(stmt->name);
	if (!IsInvalidCatalog(qualified_name.catalog)) {
		throw ParserException("Expected \"CREATE DATABASE database\" ");
	}

	info->catalog = qualified_name.catalog;
	info->name = qualified_name.name;

	result->info = std::move(info);
	return result;
}

} // namespace duckdb








namespace duckdb {

unique_ptr<CreateStatement> Transformer::TransformCreateFunction(duckdb_libpgquery::PGNode *node) {
	D_ASSERT(node);
	D_ASSERT(node->type == duckdb_libpgquery::T_PGCreateFunctionStmt);

	auto stmt = reinterpret_cast<duckdb_libpgquery::PGCreateFunctionStmt *>(node);
	D_ASSERT(stmt);
	D_ASSERT(stmt->function || stmt->query);

	auto result = make_uniq<CreateStatement>();
	auto qname = TransformQualifiedName(stmt->name);

	unique_ptr<MacroFunction> macro_func;

	// function can be null here
	if (stmt->function) {
		auto expression = TransformExpression(stmt->function);
		macro_func = make_uniq<ScalarMacroFunction>(std::move(expression));
	} else if (stmt->query) {
		auto query_node = TransformSelect(stmt->query, true)->node->Copy();
		macro_func = make_uniq<TableMacroFunction>(std::move(query_node));
	}
	PivotEntryCheck("macro");

	auto info =
	    make_uniq<CreateMacroInfo>((stmt->function ? CatalogType::MACRO_ENTRY : CatalogType::TABLE_MACRO_ENTRY));
	info->catalog = qname.catalog;
	info->schema = qname.schema;
	info->name = qname.name;

	// temporary macro
	switch (stmt->name->relpersistence) {
	case duckdb_libpgquery::PG_RELPERSISTENCE_TEMP:
		info->temporary = true;
		break;
	case duckdb_libpgquery::PG_RELPERSISTENCE_UNLOGGED:
		throw ParserException("Unlogged flag not supported for macros: '%s'", qname.name);
		break;
	case duckdb_libpgquery::RELPERSISTENCE_PERMANENT:
		info->temporary = false;
		break;
	}

	// what to do on conflict
	info->on_conflict = TransformOnConflict(stmt->onconflict);

	if (stmt->params) {
		vector<unique_ptr<ParsedExpression>> parameters;
		TransformExpressionList(*stmt->params, parameters);
		for (auto &param : parameters) {
			if (param->type == ExpressionType::VALUE_CONSTANT) {
				// parameters with default value (must have an alias)
				if (param->alias.empty()) {
					throw ParserException("Invalid parameter: '%s'", param->ToString());
				}
				if (macro_func->default_parameters.find(param->alias) != macro_func->default_parameters.end()) {
					throw ParserException("Duplicate default parameter: '%s'", param->alias);
				}
				macro_func->default_parameters[param->alias] = std::move(param);
			} else if (param->GetExpressionClass() == ExpressionClass::COLUMN_REF) {
				// positional parameters
				if (!macro_func->default_parameters.empty()) {
					throw ParserException("Positional parameters cannot come after parameters with a default value!");
				}
				macro_func->parameters.push_back(std::move(param));
			} else {
				throw ParserException("Invalid parameter: '%s'", param->ToString());
			}
		}
	}

	info->function = std::move(macro_func);
	result->info = std::move(info);

	return result;
}

} // namespace duckdb







namespace duckdb {

static IndexType StringToIndexType(const string &str) {
	string upper_str = StringUtil::Upper(str);
	if (upper_str == "INVALID") {
		return IndexType::INVALID;
	} else if (upper_str == "ART") {
		return IndexType::ART;
	} else {
		throw ConversionException("No IndexType conversion from string '%s'", upper_str);
	}
	return IndexType::INVALID;
}

vector<unique_ptr<ParsedExpression>> Transformer::TransformIndexParameters(duckdb_libpgquery::PGList *list,
                                                                           const string &relation_name) {
	vector<unique_ptr<ParsedExpression>> expressions;
	for (auto cell = list->head; cell != nullptr; cell = cell->next) {
		auto index_element = (duckdb_libpgquery::PGIndexElem *)cell->data.ptr_value;
		if (index_element->collation) {
			throw NotImplementedException("Index with collation not supported yet!");
		}
		if (index_element->opclass) {
			throw NotImplementedException("Index with opclass not supported yet!");
		}

		if (index_element->name) {
			// create a column reference expression
			expressions.push_back(make_uniq<ColumnRefExpression>(index_element->name, relation_name));
		} else {
			// parse the index expression
			D_ASSERT(index_element->expr);
			expressions.push_back(TransformExpression(index_element->expr));
		}
	}
	return expressions;
}

unique_ptr<CreateStatement> Transformer::TransformCreateIndex(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGIndexStmt *>(node);
	D_ASSERT(stmt);
	auto result = make_uniq<CreateStatement>();
	auto info = make_uniq<CreateIndexInfo>();
	if (stmt->unique) {
		info->constraint_type = IndexConstraintType::UNIQUE;
	} else {
		info->constraint_type = IndexConstraintType::NONE;
	}

	info->on_conflict = TransformOnConflict(stmt->onconflict);

	info->expressions = TransformIndexParameters(stmt->indexParams, stmt->relation->relname);

	info->index_type = StringToIndexType(string(stmt->accessMethod));
	auto tableref = make_uniq<BaseTableRef>();
	tableref->table_name = stmt->relation->relname;
	if (stmt->relation->schemaname) {
		tableref->schema_name = stmt->relation->schemaname;
	}
	info->table = std::move(tableref);
	if (stmt->idxname) {
		info->index_name = stmt->idxname;
	} else {
		throw NotImplementedException("Index without a name not supported yet!");
	}
	for (auto &expr : info->expressions) {
		info->parsed_expressions.emplace_back(expr->Copy());
	}
	result->info = std::move(info);
	return result;
}

} // namespace duckdb




namespace duckdb {

unique_ptr<CreateStatement> Transformer::TransformCreateSchema(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGCreateSchemaStmt *>(node);
	D_ASSERT(stmt);
	auto result = make_uniq<CreateStatement>();
	auto info = make_uniq<CreateSchemaInfo>();

	D_ASSERT(stmt->schemaname);
	info->catalog = stmt->catalogname ? stmt->catalogname : INVALID_CATALOG;
	info->schema = stmt->schemaname;
	info->on_conflict = TransformOnConflict(stmt->onconflict);

	if (stmt->schemaElts) {
		// schema elements
		for (auto cell = stmt->schemaElts->head; cell != nullptr; cell = cell->next) {
			auto node = reinterpret_cast<duckdb_libpgquery::PGNode *>(cell->data.ptr_value);
			switch (node->type) {
			case duckdb_libpgquery::T_PGCreateStmt:
			case duckdb_libpgquery::T_PGViewStmt:
			default:
				throw NotImplementedException("Schema element not supported yet!");
			}
		}
	}
	result->info = std::move(info);
	return result;
}

} // namespace duckdb







namespace duckdb {

unique_ptr<CreateStatement> Transformer::TransformCreateSequence(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGCreateSeqStmt *>(node);

	auto result = make_uniq<CreateStatement>();
	auto info = make_uniq<CreateSequenceInfo>();

	auto qname = TransformQualifiedName(stmt->sequence);
	info->catalog = qname.catalog;
	info->schema = qname.schema;
	info->name = qname.name;

	if (stmt->options) {
		unordered_set<SequenceInfo, EnumClassHash> used;
		duckdb_libpgquery::PGListCell *cell = nullptr;
		for_each_cell(cell, stmt->options->head) {
			auto *def_elem = reinterpret_cast<duckdb_libpgquery::PGDefElem *>(cell->data.ptr_value);
			string opt_name = string(def_elem->defname);
			auto val = (duckdb_libpgquery::PGValue *)def_elem->arg;
			bool nodef = def_elem->defaction == duckdb_libpgquery::PG_DEFELEM_UNSPEC && !val; // e.g. NO MINVALUE
			int64_t opt_value = 0;

			if (val) {
				if (val->type == duckdb_libpgquery::T_PGInteger) {
					opt_value = val->val.ival;
				} else if (val->type == duckdb_libpgquery::T_PGFloat) {
					if (!TryCast::Operation<string_t, int64_t>(string_t(val->val.str), opt_value, true)) {
						throw ParserException("Expected an integer argument for option %s", opt_name);
					}
				} else {
					throw ParserException("Expected an integer argument for option %s", opt_name);
				}
			}
			if (opt_name == "increment") {
				if (used.find(SequenceInfo::SEQ_INC) != used.end()) {
					throw ParserException("Increment value should be passed as most once");
				}
				used.insert(SequenceInfo::SEQ_INC);
				if (nodef) {
					continue;
				}

				info->increment = opt_value;
				if (info->increment == 0) {
					throw ParserException("Increment must not be zero");
				}
				if (info->increment < 0) {
					info->start_value = info->max_value = -1;
					info->min_value = NumericLimits<int64_t>::Minimum();
				} else {
					info->start_value = info->min_value = 1;
					info->max_value = NumericLimits<int64_t>::Maximum();
				}
			} else if (opt_name == "minvalue") {
				if (used.find(SequenceInfo::SEQ_MIN) != used.end()) {
					throw ParserException("Minvalue should be passed as most once");
				}
				used.insert(SequenceInfo::SEQ_MIN);
				if (nodef) {
					continue;
				}

				info->min_value = opt_value;
				if (info->increment > 0) {
					info->start_value = info->min_value;
				}
			} else if (opt_name == "maxvalue") {
				if (used.find(SequenceInfo::SEQ_MAX) != used.end()) {
					throw ParserException("Maxvalue should be passed as most once");
				}
				used.insert(SequenceInfo::SEQ_MAX);
				if (nodef) {
					continue;
				}

				info->max_value = opt_value;
				if (info->increment < 0) {
					info->start_value = info->max_value;
				}
			} else if (opt_name == "start") {
				if (used.find(SequenceInfo::SEQ_START) != used.end()) {
					throw ParserException("Start value should be passed as most once");
				}
				used.insert(SequenceInfo::SEQ_START);
				if (nodef) {
					continue;
				}

				info->start_value = opt_value;
			} else if (opt_name == "cycle") {
				if (used.find(SequenceInfo::SEQ_CYCLE) != used.end()) {
					throw ParserException("Cycle value should be passed as most once");
				}
				used.insert(SequenceInfo::SEQ_CYCLE);
				if (nodef) {
					continue;
				}

				info->cycle = opt_value > 0;
			} else {
				throw ParserException("Unrecognized option \"%s\" for CREATE SEQUENCE", opt_name);
			}
		}
	}
	info->temporary = !stmt->sequence->relpersistence;
	info->on_conflict = TransformOnConflict(stmt->onconflict);
	if (info->max_value <= info->min_value) {
		throw ParserException("MINVALUE (%lld) must be less than MAXVALUE (%lld)", info->min_value, info->max_value);
	}
	if (info->start_value < info->min_value) {
		throw ParserException("START value (%lld) cannot be less than MINVALUE (%lld)", info->start_value,
		                      info->min_value);
	}
	if (info->start_value > info->max_value) {
		throw ParserException("START value (%lld) cannot be greater than MAXVALUE (%lld)", info->start_value,
		                      info->max_value);
	}
	result->info = std::move(info);
	return result;
}

} // namespace duckdb








namespace duckdb {

string Transformer::TransformCollation(duckdb_libpgquery::PGCollateClause *collate) {
	if (!collate) {
		return string();
	}
	string collation;
	for (auto c = collate->collname->head; c != nullptr; c = lnext(c)) {
		auto pgvalue = (duckdb_libpgquery::PGValue *)c->data.ptr_value;
		if (pgvalue->type != duckdb_libpgquery::T_PGString) {
			throw ParserException("Expected a string as collation type!");
		}
		auto collation_argument = string(pgvalue->val.str);
		if (collation.empty()) {
			collation = collation_argument;
		} else {
			collation += "." + collation_argument;
		}
	}
	return collation;
}

OnCreateConflict Transformer::TransformOnConflict(duckdb_libpgquery::PGOnCreateConflict conflict) {
	switch (conflict) {
	case duckdb_libpgquery::PG_ERROR_ON_CONFLICT:
		return OnCreateConflict::ERROR_ON_CONFLICT;
	case duckdb_libpgquery::PG_IGNORE_ON_CONFLICT:
		return OnCreateConflict::IGNORE_ON_CONFLICT;
	case duckdb_libpgquery::PG_REPLACE_ON_CONFLICT:
		return OnCreateConflict::REPLACE_ON_CONFLICT;
	default:
		throw InternalException("Unrecognized OnConflict type");
	}
}

unique_ptr<ParsedExpression> Transformer::TransformCollateExpr(duckdb_libpgquery::PGCollateClause *collate) {
	auto child = TransformExpression(collate->arg);
	auto collation = TransformCollation(collate);
	return make_uniq<CollateExpression>(collation, std::move(child));
}

ColumnDefinition Transformer::TransformColumnDefinition(duckdb_libpgquery::PGColumnDef *cdef) {
	string colname;
	if (cdef->colname) {
		colname = cdef->colname;
	}
	bool optional_type = cdef->category == duckdb_libpgquery::COL_GENERATED;
	LogicalType target_type = (optional_type && !cdef->typeName) ? LogicalType::ANY : TransformTypeName(cdef->typeName);
	if (cdef->collClause) {
		if (cdef->category == duckdb_libpgquery::COL_GENERATED) {
			throw ParserException("Collations are not supported on generated columns");
		}
		if (target_type.id() != LogicalTypeId::VARCHAR) {
			throw ParserException("Only VARCHAR columns can have collations!");
		}
		target_type = LogicalType::VARCHAR_COLLATION(TransformCollation(cdef->collClause));
	}

	return ColumnDefinition(colname, target_type);
}

unique_ptr<CreateStatement> Transformer::TransformCreateTable(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGCreateStmt *>(node);
	D_ASSERT(stmt);
	auto result = make_uniq<CreateStatement>();
	auto info = make_uniq<CreateTableInfo>();

	if (stmt->inhRelations) {
		throw NotImplementedException("inherited relations not implemented");
	}
	D_ASSERT(stmt->relation);

	info->catalog = INVALID_CATALOG;
	auto qname = TransformQualifiedName(stmt->relation);
	info->catalog = qname.catalog;
	info->schema = qname.schema;
	info->table = qname.name;
	info->on_conflict = TransformOnConflict(stmt->onconflict);
	info->temporary =
	    stmt->relation->relpersistence == duckdb_libpgquery::PGPostgresRelPersistence::PG_RELPERSISTENCE_TEMP;

	if (info->temporary && stmt->oncommit != duckdb_libpgquery::PGOnCommitAction::PG_ONCOMMIT_PRESERVE_ROWS &&
	    stmt->oncommit != duckdb_libpgquery::PGOnCommitAction::PG_ONCOMMIT_NOOP) {
		throw NotImplementedException("Only ON COMMIT PRESERVE ROWS is supported");
	}
	if (!stmt->tableElts) {
		throw ParserException("Table must have at least one column!");
	}

	idx_t column_count = 0;
	for (auto c = stmt->tableElts->head; c != nullptr; c = lnext(c)) {
		auto node = reinterpret_cast<duckdb_libpgquery::PGNode *>(c->data.ptr_value);
		switch (node->type) {
		case duckdb_libpgquery::T_PGColumnDef: {
			auto cdef = (duckdb_libpgquery::PGColumnDef *)c->data.ptr_value;
			auto centry = TransformColumnDefinition(cdef);
			if (cdef->constraints) {
				for (auto constr = cdef->constraints->head; constr != nullptr; constr = constr->next) {
					auto constraint = TransformConstraint(constr, centry, info->columns.LogicalColumnCount());
					if (constraint) {
						info->constraints.push_back(std::move(constraint));
					}
				}
			}
			info->columns.AddColumn(std::move(centry));
			column_count++;
			break;
		}
		case duckdb_libpgquery::T_PGConstraint: {
			info->constraints.push_back(TransformConstraint(c));
			break;
		}
		default:
			throw NotImplementedException("ColumnDef type not handled yet");
		}
	}

	if (!column_count) {
		throw ParserException("Table must have at least one column!");
	}

	result->info = std::move(info);
	return result;
}

} // namespace duckdb




namespace duckdb {

unique_ptr<CreateStatement> Transformer::TransformCreateTableAs(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGCreateTableAsStmt *>(node);
	D_ASSERT(stmt);
	if (stmt->relkind == duckdb_libpgquery::PG_OBJECT_MATVIEW) {
		throw NotImplementedException("Materialized view not implemented");
	}
	if (stmt->is_select_into || stmt->into->colNames || stmt->into->options) {
		throw NotImplementedException("Unimplemented features for CREATE TABLE as");
	}
	auto qname = TransformQualifiedName(stmt->into->rel);
	if (stmt->query->type != duckdb_libpgquery::T_PGSelectStmt) {
		throw ParserException("CREATE TABLE AS requires a SELECT clause");
	}
	auto query = TransformSelect(stmt->query, false);

	auto result = make_uniq<CreateStatement>();
	auto info = make_uniq<CreateTableInfo>();
	info->catalog = qname.catalog;
	info->schema = qname.schema;
	info->table = qname.name;
	info->on_conflict = TransformOnConflict(stmt->onconflict);
	info->temporary =
	    stmt->into->rel->relpersistence == duckdb_libpgquery::PGPostgresRelPersistence::PG_RELPERSISTENCE_TEMP;
	info->query = std::move(query);
	result->info = std::move(info);
	return result;
}

} // namespace duckdb






namespace duckdb {

Vector ReadPgListToVector(duckdb_libpgquery::PGList *column_list, idx_t &size) {
	if (!column_list) {
		Vector result(LogicalType::VARCHAR);
		return result;
	}
	// First we discover the size of this list
	for (auto c = column_list->head; c != nullptr; c = lnext(c)) {
		size++;
	}

	Vector result(LogicalType::VARCHAR, size);
	auto result_ptr = FlatVector::GetData<string_t>(result);

	size = 0;
	for (auto c = column_list->head; c != nullptr; c = lnext(c)) {
		auto &type_val = *((duckdb_libpgquery::PGAConst *)c->data.ptr_value);
		auto entry_value_node = (duckdb_libpgquery::PGValue)(type_val.val);
		if (entry_value_node.type != duckdb_libpgquery::T_PGString) {
			throw ParserException("Expected a string constant as value");
		}

		auto entry_value = string(entry_value_node.val.str);
		D_ASSERT(!entry_value.empty());
		result_ptr[size++] = StringVector::AddStringOrBlob(result, entry_value);
	}
	return result;
}

unique_ptr<CreateStatement> Transformer::TransformCreateType(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGCreateTypeStmt *>(node);
	D_ASSERT(stmt);
	auto result = make_uniq<CreateStatement>();
	auto info = make_uniq<CreateTypeInfo>();

	auto qualified_name = TransformQualifiedName(stmt->typeName);
	info->catalog = qualified_name.catalog;
	info->schema = qualified_name.schema;
	info->name = qualified_name.name;

	switch (stmt->kind) {
	case duckdb_libpgquery::PG_NEWTYPE_ENUM: {
		info->internal = false;
		if (stmt->query) {
			// CREATE TYPE mood AS ENUM (SELECT ...)
			D_ASSERT(stmt->vals == nullptr);
			auto query = TransformSelect(stmt->query, false);
			info->query = std::move(query);
			info->type = LogicalType::INVALID;
		} else {
			D_ASSERT(stmt->query == nullptr);
			idx_t size = 0;
			auto ordered_array = ReadPgListToVector(stmt->vals, size);
			info->type = LogicalType::ENUM(info->name, ordered_array, size);
		}
	} break;

	case duckdb_libpgquery::PG_NEWTYPE_ALIAS: {
		LogicalType target_type = TransformTypeName(stmt->ofType);
		info->type = target_type;
	} break;

	default:
		throw InternalException("Unknown kind of new type");
	}

	result->info = std::move(info);
	return result;
}
} // namespace duckdb




namespace duckdb {

unique_ptr<CreateStatement> Transformer::TransformCreateView(duckdb_libpgquery::PGNode *node) {
	D_ASSERT(node);
	D_ASSERT(node->type == duckdb_libpgquery::T_PGViewStmt);

	auto stmt = reinterpret_cast<duckdb_libpgquery::PGViewStmt *>(node);
	D_ASSERT(stmt);
	D_ASSERT(stmt->view);

	auto result = make_uniq<CreateStatement>();
	auto info = make_uniq<CreateViewInfo>();

	auto qname = TransformQualifiedName(stmt->view);
	info->catalog = qname.catalog;
	info->schema = qname.schema;
	info->view_name = qname.name;
	info->temporary = !stmt->view->relpersistence;
	if (info->temporary && IsInvalidCatalog(info->catalog)) {
		info->catalog = TEMP_CATALOG;
	}
	info->on_conflict = TransformOnConflict(stmt->onconflict);

	info->query = TransformSelect(stmt->query, false);

	PivotEntryCheck("view");

	if (stmt->aliases && stmt->aliases->length > 0) {
		for (auto c = stmt->aliases->head; c != nullptr; c = lnext(c)) {
			auto node = reinterpret_cast<duckdb_libpgquery::PGNode *>(c->data.ptr_value);
			switch (node->type) {
			case duckdb_libpgquery::T_PGString: {
				auto val = (duckdb_libpgquery::PGValue *)node;
				info->aliases.emplace_back(val->val.str);
				break;
			}
			default:
				throw NotImplementedException("View projection type");
			}
		}
		if (info->aliases.empty()) {
			throw ParserException("Need at least one column name in CREATE VIEW projection list");
		}
	}

	if (stmt->options && stmt->options->length > 0) {
		throw NotImplementedException("VIEW options");
	}

	if (stmt->withCheckOption != duckdb_libpgquery::PGViewCheckOption::PG_NO_CHECK_OPTION) {
		throw NotImplementedException("VIEW CHECK options");
	}
	result->info = std::move(info);
	return result;
}

} // namespace duckdb



namespace duckdb {

unique_ptr<DeleteStatement> Transformer::TransformDelete(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGDeleteStmt *>(node);
	D_ASSERT(stmt);
	auto result = make_uniq<DeleteStatement>();
	if (stmt->withClause) {
		TransformCTE(reinterpret_cast<duckdb_libpgquery::PGWithClause *>(stmt->withClause), result->cte_map);
	}

	result->condition = TransformExpression(stmt->whereClause);
	result->table = TransformRangeVar(stmt->relation);
	if (result->table->type != TableReferenceType::BASE_TABLE) {
		throw Exception("Can only delete from base tables!");
	}
	if (stmt->usingClause) {
		for (auto n = stmt->usingClause->head; n != nullptr; n = n->next) {
			auto target = reinterpret_cast<duckdb_libpgquery::PGNode *>(n->data.ptr_value);
			auto using_entry = TransformTableRefNode(target);
			result->using_clauses.push_back(std::move(using_entry));
		}
	}

	if (stmt->returningList) {
		Transformer::TransformExpressionList(*(stmt->returningList), result->returning_list);
	}
	return result;
}

} // namespace duckdb





namespace duckdb {

unique_ptr<DetachStatement> Transformer::TransformDetach(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGDetachStmt *>(node);
	auto result = make_uniq<DetachStatement>();
	auto info = make_uniq<DetachInfo>();
	info->name = stmt->db_name;
	info->if_not_found = TransformOnEntryNotFound(stmt->missing_ok);

	result->info = std::move(info);
	return result;
}

} // namespace duckdb



namespace duckdb {

unique_ptr<SQLStatement> Transformer::TransformDrop(duckdb_libpgquery::PGNode *node) {
	auto stmt = (duckdb_libpgquery::PGDropStmt *)(node);
	auto result = make_uniq<DropStatement>();
	auto &info = *result->info.get();
	D_ASSERT(stmt);
	if (stmt->objects->length != 1) {
		throw NotImplementedException("Can only drop one object at a time");
	}
	switch (stmt->removeType) {
	case duckdb_libpgquery::PG_OBJECT_TABLE:
		info.type = CatalogType::TABLE_ENTRY;
		break;
	case duckdb_libpgquery::PG_OBJECT_SCHEMA:
		info.type = CatalogType::SCHEMA_ENTRY;
		break;
	case duckdb_libpgquery::PG_OBJECT_INDEX:
		info.type = CatalogType::INDEX_ENTRY;
		break;
	case duckdb_libpgquery::PG_OBJECT_VIEW:
		info.type = CatalogType::VIEW_ENTRY;
		break;
	case duckdb_libpgquery::PG_OBJECT_SEQUENCE:
		info.type = CatalogType::SEQUENCE_ENTRY;
		break;
	case duckdb_libpgquery::PG_OBJECT_FUNCTION:
		info.type = CatalogType::MACRO_ENTRY;
		break;
	case duckdb_libpgquery::PG_OBJECT_TABLE_MACRO:
		info.type = CatalogType::TABLE_MACRO_ENTRY;
		break;
	case duckdb_libpgquery::PG_OBJECT_TYPE:
		info.type = CatalogType::TYPE_ENTRY;
		break;
	case duckdb_libpgquery::PG_OBJECT_DATABASE:
		info.type = CatalogType::DATABASE_ENTRY;
		break;
	default:
		throw NotImplementedException("Cannot drop this type yet");
	}

	switch (stmt->removeType) {
	case duckdb_libpgquery::PG_OBJECT_TYPE: {
		auto view_list = (duckdb_libpgquery::PGList *)stmt->objects;
		auto target = (duckdb_libpgquery::PGTypeName *)(view_list->head->data.ptr_value);
		info.name = (reinterpret_cast<duckdb_libpgquery::PGValue *>(target->names->tail->data.ptr_value)->val.str);
		break;
	}
	case duckdb_libpgquery::PG_OBJECT_SCHEMA: {
		auto view_list = (duckdb_libpgquery::PGList *)stmt->objects->head->data.ptr_value;
		if (view_list->length == 2) {
			info.catalog = ((duckdb_libpgquery::PGValue *)view_list->head->data.ptr_value)->val.str;
			info.name = ((duckdb_libpgquery::PGValue *)view_list->head->next->data.ptr_value)->val.str;
		} else if (view_list->length == 1) {
			info.name = ((duckdb_libpgquery::PGValue *)view_list->head->data.ptr_value)->val.str;
		} else {
			throw ParserException("Expected \"catalog.schema\" or \"schema\"");
		}
		break;
	}
	default: {
		auto view_list = (duckdb_libpgquery::PGList *)stmt->objects->head->data.ptr_value;
		if (view_list->length == 3) {
			info.catalog = ((duckdb_libpgquery::PGValue *)view_list->head->data.ptr_value)->val.str;
			info.schema = ((duckdb_libpgquery::PGValue *)view_list->head->next->data.ptr_value)->val.str;
			info.name = ((duckdb_libpgquery::PGValue *)view_list->head->next->next->data.ptr_value)->val.str;
		} else if (view_list->length == 2) {
			info.schema = ((duckdb_libpgquery::PGValue *)view_list->head->data.ptr_value)->val.str;
			info.name = ((duckdb_libpgquery::PGValue *)view_list->head->next->data.ptr_value)->val.str;
		} else if (view_list->length == 1) {
			info.name = ((duckdb_libpgquery::PGValue *)view_list->head->data.ptr_value)->val.str;
		} else {
			throw ParserException("Expected \"catalog.schema.name\", \"schema.name\"or \"name\"");
		}
		break;
	}
	}
	info.cascade = stmt->behavior == duckdb_libpgquery::PGDropBehavior::PG_DROP_CASCADE;
	info.if_not_found = TransformOnEntryNotFound(stmt->missing_ok);
	return std::move(result);
}

} // namespace duckdb



namespace duckdb {

unique_ptr<ExplainStatement> Transformer::TransformExplain(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGExplainStmt *>(node);
	D_ASSERT(stmt);
	auto explain_type = ExplainType::EXPLAIN_STANDARD;
	if (stmt->options) {
		for (auto n = stmt->options->head; n; n = n->next) {
			auto def_elem = ((duckdb_libpgquery::PGDefElem *)n->data.ptr_value)->defname;
			string elem(def_elem);
			if (elem == "analyze") {
				explain_type = ExplainType::EXPLAIN_ANALYZE;
			} else {
				throw NotImplementedException("Unimplemented explain type: %s", elem);
			}
		}
	}
	return make_uniq<ExplainStatement>(TransformStatement(stmt->query), explain_type);
}

} // namespace duckdb



namespace duckdb {

unique_ptr<ExportStatement> Transformer::TransformExport(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGExportStmt *>(node);
	auto info = make_uniq<CopyInfo>();
	info->file_path = stmt->filename;
	info->format = "csv";
	info->is_from = false;
	// handle export options
	TransformCopyOptions(*info, stmt->options);

	auto result = make_uniq<ExportStatement>(std::move(info));
	if (stmt->database) {
		result->database = stmt->database;
	}
	return result;
}

} // namespace duckdb



namespace duckdb {

unique_ptr<PragmaStatement> Transformer::TransformImport(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGImportStmt *>(node);
	auto result = make_uniq<PragmaStatement>();
	result->info->name = "import_database";
	result->info->parameters.emplace_back(stmt->filename);
	return result;
}

} // namespace duckdb




namespace duckdb {

unique_ptr<TableRef> Transformer::TransformValuesList(duckdb_libpgquery::PGList *list) {
	auto result = make_uniq<ExpressionListRef>();
	for (auto value_list = list->head; value_list != nullptr; value_list = value_list->next) {
		auto target = (duckdb_libpgquery::PGList *)(value_list->data.ptr_value);

		vector<unique_ptr<ParsedExpression>> insert_values;
		TransformExpressionList(*target, insert_values);
		if (!result->values.empty()) {
			if (result->values[0].size() != insert_values.size()) {
				throw ParserException("VALUES lists must all be the same length");
			}
		}
		result->values.push_back(std::move(insert_values));
	}
	result->alias = "valueslist";
	return std::move(result);
}

unique_ptr<InsertStatement> Transformer::TransformInsert(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGInsertStmt *>(node);
	D_ASSERT(stmt);

	auto result = make_uniq<InsertStatement>();
	if (stmt->withClause) {
		TransformCTE(reinterpret_cast<duckdb_libpgquery::PGWithClause *>(stmt->withClause), result->cte_map);
	}

	// first check if there are any columns specified
	if (stmt->cols) {
		for (auto c = stmt->cols->head; c != nullptr; c = lnext(c)) {
			auto target = (duckdb_libpgquery::PGResTarget *)(c->data.ptr_value);
			result->columns.emplace_back(target->name);
		}
	}

	// Grab and transform the returning columns from the parser.
	if (stmt->returningList) {
		Transformer::TransformExpressionList(*(stmt->returningList), result->returning_list);
	}
	if (stmt->selectStmt) {
		result->select_statement = TransformSelect(stmt->selectStmt, false);
	} else {
		result->default_values = true;
	}

	auto qname = TransformQualifiedName(stmt->relation);
	result->table = qname.name;
	result->schema = qname.schema;

	if (stmt->onConflictClause) {
		if (stmt->onConflictAlias != duckdb_libpgquery::PG_ONCONFLICT_ALIAS_NONE) {
			// OR REPLACE | OR IGNORE are shorthands for the ON CONFLICT clause
			throw ParserException("You can not provide both OR REPLACE|IGNORE and an ON CONFLICT clause, please remove "
			                      "the first if you want to have more granual control");
		}
		result->on_conflict_info = TransformOnConflictClause(stmt->onConflictClause, result->schema);
		result->table_ref = TransformRangeVar(stmt->relation);
	}
	if (stmt->onConflictAlias != duckdb_libpgquery::PG_ONCONFLICT_ALIAS_NONE) {
		D_ASSERT(!stmt->onConflictClause);
		result->on_conflict_info = DummyOnConflictClause(stmt->onConflictAlias, result->schema);
		result->table_ref = TransformRangeVar(stmt->relation);
	}
	switch (stmt->insert_column_order) {
	case duckdb_libpgquery::PG_INSERT_BY_POSITION:
		result->column_order = InsertColumnOrder::INSERT_BY_POSITION;
		break;
	case duckdb_libpgquery::PG_INSERT_BY_NAME:
		result->column_order = InsertColumnOrder::INSERT_BY_NAME;
		break;
	default:
		throw InternalException("Unrecognized insert column order in TransformInsert");
	}
	result->catalog = qname.catalog;
	return result;
}

} // namespace duckdb



namespace duckdb {

unique_ptr<LoadStatement> Transformer::TransformLoad(duckdb_libpgquery::PGNode *node) {
	D_ASSERT(node->type == duckdb_libpgquery::T_PGLoadStmt);
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGLoadStmt *>(node);

	auto load_stmt = make_uniq<LoadStatement>();
	auto load_info = make_uniq<LoadInfo>();
	load_info->filename = std::string(stmt->filename);
	switch (stmt->load_type) {
	case duckdb_libpgquery::PG_LOAD_TYPE_LOAD:
		load_info->load_type = LoadType::LOAD;
		break;
	case duckdb_libpgquery::PG_LOAD_TYPE_INSTALL:
		load_info->load_type = LoadType::INSTALL;
		break;
	case duckdb_libpgquery::PG_LOAD_TYPE_FORCE_INSTALL:
		load_info->load_type = LoadType::FORCE_INSTALL;
		break;
	}
	load_stmt->info = std::move(load_info);
	return load_stmt;
}

} // namespace duckdb


















namespace duckdb {

void Transformer::AddPivotEntry(string enum_name, unique_ptr<SelectNode> base, unique_ptr<ParsedExpression> column,
                                unique_ptr<QueryNode> subquery) {
	if (parent) {
		parent->AddPivotEntry(std::move(enum_name), std::move(base), std::move(column), std::move(subquery));
		return;
	}
	auto result = make_uniq<CreatePivotEntry>();
	result->enum_name = std::move(enum_name);
	result->base = std::move(base);
	result->column = std::move(column);
	result->subquery = std::move(subquery);

	pivot_entries.push_back(std::move(result));
}

bool Transformer::HasPivotEntries() {
	return !GetPivotEntries().empty();
}

idx_t Transformer::PivotEntryCount() {
	return GetPivotEntries().size();
}

vector<unique_ptr<Transformer::CreatePivotEntry>> &Transformer::GetPivotEntries() {
	if (parent) {
		return parent->GetPivotEntries();
	}
	return pivot_entries;
}

void Transformer::PivotEntryCheck(const string &type) {
	auto &entries = GetPivotEntries();
	if (!entries.empty()) {
		throw ParserException(
		    "PIVOT statements with pivot elements extracted from the data cannot be used in %ss.\nIn order to use "
		    "PIVOT in a %s the PIVOT values must be manually specified, e.g.:\nPIVOT ... ON %s IN (val1, val2, ...)",
		    type, type, entries[0]->column->ToString());
	}
}
unique_ptr<SQLStatement> Transformer::GenerateCreateEnumStmt(unique_ptr<CreatePivotEntry> entry) {
	auto result = make_uniq<CreateStatement>();
	auto info = make_uniq<CreateTypeInfo>();

	info->temporary = true;
	info->internal = false;
	info->catalog = INVALID_CATALOG;
	info->schema = INVALID_SCHEMA;
	info->name = std::move(entry->enum_name);
	info->on_conflict = OnCreateConflict::REPLACE_ON_CONFLICT;

	// generate the query that will result in the enum creation
	unique_ptr<QueryNode> subselect;
	if (!entry->subquery) {
		auto select_node = std::move(entry->base);
		auto columnref = entry->column->Copy();
		auto cast = make_uniq<CastExpression>(LogicalType::VARCHAR, std::move(columnref));
		select_node->select_list.push_back(std::move(cast));

		auto is_not_null =
		    make_uniq<OperatorExpression>(ExpressionType::OPERATOR_IS_NOT_NULL, std::move(entry->column));
		select_node->where_clause = std::move(is_not_null);

		// order by the column
		select_node->modifiers.push_back(make_uniq<DistinctModifier>());
		auto modifier = make_uniq<OrderModifier>();
		modifier->orders.emplace_back(OrderType::ASCENDING, OrderByNullType::ORDER_DEFAULT,
		                              make_uniq<ConstantExpression>(Value::INTEGER(1)));
		select_node->modifiers.push_back(std::move(modifier));
		subselect = std::move(select_node);
	} else {
		subselect = std::move(entry->subquery);
	}

	auto select = make_uniq<SelectStatement>();
	select->node = std::move(subselect);
	info->query = std::move(select);
	info->type = LogicalType::INVALID;

	result->info = std::move(info);
	return std::move(result);
}

// unique_ptr<SQLStatement> GenerateDropEnumStmt(string enum_name) {
//	auto result = make_uniq<DropStatement>();
//	result->info->if_exists = true;
//	result->info->schema = INVALID_SCHEMA;
//	result->info->catalog = INVALID_CATALOG;
//	result->info->name = std::move(enum_name);
//	result->info->type = CatalogType::TYPE_ENTRY;
//	return std::move(result);
//}

unique_ptr<SQLStatement> Transformer::CreatePivotStatement(unique_ptr<SQLStatement> statement) {
	auto result = make_uniq<MultiStatement>();
	for (auto &pivot : pivot_entries) {
		result->statements.push_back(GenerateCreateEnumStmt(std::move(pivot)));
	}
	result->statements.push_back(std::move(statement));
	// FIXME: drop the types again!?
	//	for(auto &pivot : pivot_entries) {
	//		result->statements.push_back(GenerateDropEnumStmt(std::move(pivot->enum_name)));
	//	}
	return std::move(result);
}

unique_ptr<QueryNode> Transformer::TransformPivotStatement(duckdb_libpgquery::PGSelectStmt *stmt) {
	auto pivot = stmt->pivot;
	auto source = TransformTableRefNode(pivot->source);

	auto select_node = make_uniq<SelectNode>();
	// handle the CTEs
	if (stmt->withClause) {
		TransformCTE(reinterpret_cast<duckdb_libpgquery::PGWithClause *>(stmt->withClause), select_node->cte_map);
	}
	if (!pivot->columns) {
		// no pivot columns - not actually a pivot
		select_node->from_table = std::move(source);
		if (pivot->groups) {
			auto groups = TransformStringList(pivot->groups);
			GroupingSet set;
			for (idx_t gr = 0; gr < groups.size(); gr++) {
				auto &group = groups[gr];
				auto colref = make_uniq<ColumnRefExpression>(group);
				select_node->select_list.push_back(colref->Copy());
				select_node->groups.group_expressions.push_back(std::move(colref));
				set.insert(gr);
			}
			select_node->groups.grouping_sets.push_back(std::move(set));
		}
		if (pivot->aggrs) {
			TransformExpressionList(*pivot->aggrs, select_node->select_list);
		}
		return std::move(select_node);
	}

	// generate CREATE TYPE statements for each of the columns that do not have an IN list
	auto columns = TransformPivotList(pivot->columns);
	auto pivot_idx = PivotEntryCount();
	for (idx_t c = 0; c < columns.size(); c++) {
		auto &col = columns[c];
		if (!col.pivot_enum.empty() || !col.entries.empty()) {
			continue;
		}
		if (col.pivot_expressions.size() != 1) {
			throw InternalException("PIVOT statement with multiple names in pivot entry!?");
		}
		auto enum_name = "__pivot_enum_" + std::to_string(pivot_idx) + "_" + std::to_string(c);

		auto new_select = make_uniq<SelectNode>();
		ExtractCTEsRecursive(new_select->cte_map);
		new_select->from_table = source->Copy();
		AddPivotEntry(enum_name, std::move(new_select), col.pivot_expressions[0]->Copy(), std::move(col.subquery));
		col.pivot_enum = enum_name;
	}

	// generate the actual query, including the pivot
	select_node->select_list.push_back(make_uniq<StarExpression>());

	auto pivot_ref = make_uniq<PivotRef>();
	pivot_ref->source = std::move(source);
	if (pivot->unpivots) {
		pivot_ref->unpivot_names = TransformStringList(pivot->unpivots);
	} else {
		if (pivot->aggrs) {
			TransformExpressionList(*pivot->aggrs, pivot_ref->aggregates);
		} else {
			// pivot but no aggregates specified - push a count star
			vector<unique_ptr<ParsedExpression>> children;
			auto function = make_uniq<FunctionExpression>("count_star", std::move(children));
			pivot_ref->aggregates.push_back(std::move(function));
		}
	}
	if (pivot->groups) {
		pivot_ref->groups = TransformStringList(pivot->groups);
	}
	pivot_ref->pivots = std::move(columns);
	select_node->from_table = std::move(pivot_ref);
	// transform order by/limit modifiers
	TransformModifiers(*stmt, *select_node);
	return std::move(select_node);
}

} // namespace duckdb









namespace duckdb {

unique_ptr<SQLStatement> Transformer::TransformPragma(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGPragmaStmt *>(node);

	auto result = make_uniq<PragmaStatement>();
	auto &info = *result->info;

	info.name = stmt->name;
	// parse the arguments, if any
	if (stmt->args) {
		for (auto cell = stmt->args->head; cell != nullptr; cell = cell->next) {
			auto node = reinterpret_cast<duckdb_libpgquery::PGNode *>(cell->data.ptr_value);
			auto expr = TransformExpression(node);

			if (expr->type == ExpressionType::COMPARE_EQUAL) {
				auto &comp = expr->Cast<ComparisonExpression>();
				if (comp.right->type != ExpressionType::VALUE_CONSTANT) {
					throw ParserException("Named parameter requires a constant on the RHS");
				}
				if (comp.left->type != ExpressionType::COLUMN_REF) {
					throw ParserException("Named parameter requires a column reference on the LHS");
				}
				auto &columnref = comp.left->Cast<ColumnRefExpression>();
				auto &constant = comp.right->Cast<ConstantExpression>();
				info.named_parameters[columnref.GetName()] = constant.value;
			} else if (node->type == duckdb_libpgquery::T_PGAConst) {
				auto constant = TransformConstant((duckdb_libpgquery::PGAConst *)node);
				info.parameters.push_back((constant->Cast<ConstantExpression>()).value);
			} else if (expr->type == ExpressionType::COLUMN_REF) {
				auto &colref = expr->Cast<ColumnRefExpression>();
				if (!colref.IsQualified()) {
					info.parameters.emplace_back(colref.GetColumnName());
				} else {
					info.parameters.emplace_back(expr->ToString());
				}
			} else {
				info.parameters.emplace_back(expr->ToString());
			}
		}
	}
	// now parse the pragma type
	switch (stmt->kind) {
	case duckdb_libpgquery::PG_PRAGMA_TYPE_NOTHING: {
		if (!info.parameters.empty() || !info.named_parameters.empty()) {
			throw InternalException("PRAGMA statement that is not a call or assignment cannot contain parameters");
		}
		break;
	case duckdb_libpgquery::PG_PRAGMA_TYPE_ASSIGNMENT:
		if (info.parameters.size() != 1) {
			throw InternalException("PRAGMA statement with assignment should contain exactly one parameter");
		}
		if (!info.named_parameters.empty()) {
			throw InternalException("PRAGMA statement with assignment cannot have named parameters");
		}
		// SQLite does not distinguish between:
		// "PRAGMA table_info='integers'"
		// "PRAGMA table_info('integers')"
		// for compatibility, any pragmas that match the SQLite ones are parsed as calls
		case_insensitive_set_t sqlite_compat_pragmas {"table_info"};
		if (sqlite_compat_pragmas.find(info.name) != sqlite_compat_pragmas.end()) {
			break;
		}
		auto set_statement = make_uniq<SetVariableStatement>(info.name, info.parameters[0], SetScope::AUTOMATIC);
		return std::move(set_statement);
	}
	case duckdb_libpgquery::PG_PRAGMA_TYPE_CALL:
		break;
	default:
		throw InternalException("Unknown pragma type");
	}

	return std::move(result);
}

} // namespace duckdb





namespace duckdb {

unique_ptr<PrepareStatement> Transformer::TransformPrepare(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGPrepareStmt *>(node);
	D_ASSERT(stmt);

	if (stmt->argtypes && stmt->argtypes->length > 0) {
		throw NotImplementedException("Prepared statement argument types are not supported, use CAST");
	}

	auto result = make_uniq<PrepareStatement>();
	result->name = string(stmt->name);
	result->statement = TransformStatement(stmt->query);
	if (!result->statement->named_param_map.empty()) {
		throw NotImplementedException("Named parameters are not supported in this client yet");
	}
	SetParamCount(0);

	return result;
}

unique_ptr<ExecuteStatement> Transformer::TransformExecute(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGExecuteStmt *>(node);
	D_ASSERT(stmt);

	auto result = make_uniq<ExecuteStatement>();
	result->name = string(stmt->name);

	if (stmt->params) {
		TransformExpressionList(*stmt->params, result->values);
	}
	for (auto &expr : result->values) {
		if (!expr->IsScalar()) {
			throw Exception("Only scalar parameters or NULL supported for EXECUTE");
		}
	}
	return result;
}

unique_ptr<DropStatement> Transformer::TransformDeallocate(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGDeallocateStmt *>(node);
	D_ASSERT(stmt);
	if (!stmt->name) {
		throw ParserException("DEALLOCATE requires a name");
	}

	auto result = make_uniq<DropStatement>();
	result->info->type = CatalogType::PREPARED_STATEMENT;
	result->info->name = string(stmt->name);
	return result;
}

} // namespace duckdb



namespace duckdb {

unique_ptr<AlterStatement> Transformer::TransformRename(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGRenameStmt *>(node);
	D_ASSERT(stmt);
	if (!stmt->relation) {
		throw NotImplementedException("Altering schemas is not yet supported");
	}

	unique_ptr<AlterInfo> info;

	AlterEntryData data;
	data.if_not_found = TransformOnEntryNotFound(stmt->missing_ok);
	data.catalog = stmt->relation->catalogname ? stmt->relation->catalogname : INVALID_CATALOG;
	data.schema = stmt->relation->schemaname ? stmt->relation->schemaname : INVALID_SCHEMA;
	if (stmt->relation->relname) {
		data.name = stmt->relation->relname;
	}
	// first we check the type of ALTER
	switch (stmt->renameType) {
	case duckdb_libpgquery::PG_OBJECT_COLUMN: {
		// change column name

		// get the old name and the new name
		string old_name = stmt->subname;
		string new_name = stmt->newname;
		info = make_uniq<RenameColumnInfo>(std::move(data), old_name, new_name);
		break;
	}
	case duckdb_libpgquery::PG_OBJECT_TABLE: {
		// change table name
		string new_name = stmt->newname;
		info = make_uniq<RenameTableInfo>(std::move(data), new_name);
		break;
	}
	case duckdb_libpgquery::PG_OBJECT_VIEW: {
		// change view name
		string new_name = stmt->newname;
		info = make_uniq<RenameViewInfo>(std::move(data), new_name);
		break;
	}
	case duckdb_libpgquery::PG_OBJECT_DATABASE:
	default:
		throw NotImplementedException("Schema element not supported yet!");
	}
	D_ASSERT(info);

	auto result = make_uniq<AlterStatement>();
	result->info = std::move(info);
	return result;
}

} // namespace duckdb




namespace duckdb {

unique_ptr<QueryNode> Transformer::TransformSelectNode(duckdb_libpgquery::PGSelectStmt *stmt) {
	if (stmt->pivot) {
		return TransformPivotStatement(stmt);
	} else {
		return TransformSelectInternal(stmt);
	}
}

unique_ptr<SelectStatement> Transformer::TransformSelect(duckdb_libpgquery::PGNode *node, bool is_select) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGSelectStmt *>(node);
	auto result = make_uniq<SelectStatement>();

	// Both Insert/Create Table As uses this.
	if (is_select) {
		if (stmt->intoClause) {
			throw ParserException("SELECT INTO not supported!");
		}
		if (stmt->lockingClause) {
			throw ParserException("SELECT locking clause is not supported!");
		}
	}

	result->node = TransformSelectNode(stmt);
	return result;
}

} // namespace duckdb








namespace duckdb {

void Transformer::TransformModifiers(duckdb_libpgquery::PGSelectStmt &stmt, QueryNode &node) {
	// transform the common properties
	// both the set operations and the regular select can have an ORDER BY/LIMIT attached to them
	vector<OrderByNode> orders;
	TransformOrderBy(stmt.sortClause, orders);
	if (!orders.empty()) {
		auto order_modifier = make_uniq<OrderModifier>();
		order_modifier->orders = std::move(orders);
		node.modifiers.push_back(std::move(order_modifier));
	}
	if (stmt.limitCount || stmt.limitOffset) {
		if (stmt.limitCount && stmt.limitCount->type == duckdb_libpgquery::T_PGLimitPercent) {
			auto limit_percent_modifier = make_uniq<LimitPercentModifier>();
			auto expr_node = reinterpret_cast<duckdb_libpgquery::PGLimitPercent *>(stmt.limitCount)->limit_percent;
			limit_percent_modifier->limit = TransformExpression(expr_node);
			if (stmt.limitOffset) {
				limit_percent_modifier->offset = TransformExpression(stmt.limitOffset);
			}
			node.modifiers.push_back(std::move(limit_percent_modifier));
		} else {
			auto limit_modifier = make_uniq<LimitModifier>();
			if (stmt.limitCount) {
				limit_modifier->limit = TransformExpression(stmt.limitCount);
			}
			if (stmt.limitOffset) {
				limit_modifier->offset = TransformExpression(stmt.limitOffset);
			}
			node.modifiers.push_back(std::move(limit_modifier));
		}
	}
}

unique_ptr<QueryNode> Transformer::TransformSelectInternal(duckdb_libpgquery::PGSelectStmt *stmt) {
	D_ASSERT(stmt->type == duckdb_libpgquery::T_PGSelectStmt);
	auto stack_checker = StackCheck();

	unique_ptr<QueryNode> node;

	switch (stmt->op) {
	case duckdb_libpgquery::PG_SETOP_NONE: {
		node = make_uniq<SelectNode>();
		auto &result = node->Cast<SelectNode>();
		if (stmt->withClause) {
			TransformCTE(reinterpret_cast<duckdb_libpgquery::PGWithClause *>(stmt->withClause), node->cte_map);
		}
		if (stmt->windowClause) {
			for (auto window_ele = stmt->windowClause->head; window_ele != nullptr; window_ele = window_ele->next) {
				auto window_def = reinterpret_cast<duckdb_libpgquery::PGWindowDef *>(window_ele->data.ptr_value);
				D_ASSERT(window_def);
				D_ASSERT(window_def->name);
				string window_name(window_def->name);
				auto it = window_clauses.find(window_name);
				if (it != window_clauses.end()) {
					throw ParserException("window \"%s\" is already defined", window_name);
				}
				window_clauses[window_name] = window_def;
			}
		}

		// checks distinct clause
		if (stmt->distinctClause != nullptr) {
			auto modifier = make_uniq<DistinctModifier>();
			// checks distinct on clause
			auto target = reinterpret_cast<duckdb_libpgquery::PGNode *>(stmt->distinctClause->head->data.ptr_value);
			if (target) {
				//  add the columns defined in the ON clause to the select list
				TransformExpressionList(*stmt->distinctClause, modifier->distinct_on_targets);
			}
			result.modifiers.push_back(std::move(modifier));
		}

		// do this early so the value lists also have a `FROM`
		if (stmt->valuesLists) {
			// VALUES list, create an ExpressionList
			D_ASSERT(!stmt->fromClause);
			result.from_table = TransformValuesList(stmt->valuesLists);
			result.select_list.push_back(make_uniq<StarExpression>());
		} else {
			if (!stmt->targetList) {
				throw ParserException("SELECT clause without selection list");
			}
			// select list
			TransformExpressionList(*stmt->targetList, result.select_list);
			result.from_table = TransformFrom(stmt->fromClause);
		}

		// where
		result.where_clause = TransformExpression(stmt->whereClause);
		// group by
		TransformGroupBy(stmt->groupClause, result);
		// having
		result.having = TransformExpression(stmt->havingClause);
		// qualify
		result.qualify = TransformExpression(stmt->qualifyClause);
		// sample
		result.sample = TransformSampleOptions(stmt->sampleOptions);
		break;
	}
	case duckdb_libpgquery::PG_SETOP_UNION:
	case duckdb_libpgquery::PG_SETOP_EXCEPT:
	case duckdb_libpgquery::PG_SETOP_INTERSECT:
	case duckdb_libpgquery::PG_SETOP_UNION_BY_NAME: {
		node = make_uniq<SetOperationNode>();
		auto &result = node->Cast<SetOperationNode>();
		if (stmt->withClause) {
			TransformCTE(reinterpret_cast<duckdb_libpgquery::PGWithClause *>(stmt->withClause), node->cte_map);
		}
		result.left = TransformSelectNode(stmt->larg);
		result.right = TransformSelectNode(stmt->rarg);
		if (!result.left || !result.right) {
			throw Exception("Failed to transform setop children.");
		}

		bool select_distinct = true;
		switch (stmt->op) {
		case duckdb_libpgquery::PG_SETOP_UNION:
			select_distinct = !stmt->all;
			result.setop_type = SetOperationType::UNION;
			break;
		case duckdb_libpgquery::PG_SETOP_EXCEPT:
			result.setop_type = SetOperationType::EXCEPT;
			break;
		case duckdb_libpgquery::PG_SETOP_INTERSECT:
			result.setop_type = SetOperationType::INTERSECT;
			break;
		case duckdb_libpgquery::PG_SETOP_UNION_BY_NAME:
			select_distinct = !stmt->all;
			result.setop_type = SetOperationType::UNION_BY_NAME;
			break;
		default:
			throw Exception("Unexpected setop type");
		}
		if (select_distinct) {
			result.modifiers.push_back(make_uniq<DistinctModifier>());
		}
		if (stmt->sampleOptions) {
			throw ParserException("SAMPLE clause is only allowed in regular SELECT statements");
		}
		break;
	}
	default:
		throw NotImplementedException("Statement type %d not implemented!", stmt->op);
	}
	TransformModifiers(*stmt, *node);
	return node;
}

} // namespace duckdb





namespace duckdb {

namespace {

SetScope ToSetScope(duckdb_libpgquery::VariableSetScope pg_scope) {
	switch (pg_scope) {
	case duckdb_libpgquery::VariableSetScope::VAR_SET_SCOPE_LOCAL:
		return SetScope::LOCAL;
	case duckdb_libpgquery::VariableSetScope::VAR_SET_SCOPE_SESSION:
		return SetScope::SESSION;
	case duckdb_libpgquery::VariableSetScope::VAR_SET_SCOPE_GLOBAL:
		return SetScope::GLOBAL;
	case duckdb_libpgquery::VariableSetScope::VAR_SET_SCOPE_DEFAULT:
		return SetScope::AUTOMATIC;
	default:
		throw InternalException("Unexpected pg_scope: %d", pg_scope);
	}
}

SetType ToSetType(duckdb_libpgquery::VariableSetKind pg_kind) {
	switch (pg_kind) {
	case duckdb_libpgquery::VariableSetKind::VAR_SET_VALUE:
		return SetType::SET;
	case duckdb_libpgquery::VariableSetKind::VAR_RESET:
		return SetType::RESET;
	default:
		throw NotImplementedException("Can only SET or RESET a variable");
	}
}

} // namespace

unique_ptr<SetStatement> Transformer::TransformSetVariable(duckdb_libpgquery::PGVariableSetStmt *stmt) {
	D_ASSERT(stmt->kind == duckdb_libpgquery::VariableSetKind::VAR_SET_VALUE);

	if (stmt->scope == duckdb_libpgquery::VariableSetScope::VAR_SET_SCOPE_LOCAL) {
		throw NotImplementedException("SET LOCAL is not implemented.");
	}

	auto name = std::string(stmt->name);
	D_ASSERT(!name.empty()); // parser protect us!
	if (stmt->args->length != 1) {
		throw ParserException("SET needs a single scalar value parameter");
	}
	D_ASSERT(stmt->args->head && stmt->args->head->data.ptr_value);
	D_ASSERT(((duckdb_libpgquery::PGNode *)stmt->args->head->data.ptr_value)->type == duckdb_libpgquery::T_PGAConst);

	auto value = TransformValue(((duckdb_libpgquery::PGAConst *)stmt->args->head->data.ptr_value)->val)->value;

	return make_uniq<SetVariableStatement>(name, value, ToSetScope(stmt->scope));
}

unique_ptr<SetStatement> Transformer::TransformResetVariable(duckdb_libpgquery::PGVariableSetStmt *stmt) {
	D_ASSERT(stmt->kind == duckdb_libpgquery::VariableSetKind::VAR_RESET);

	if (stmt->scope == duckdb_libpgquery::VariableSetScope::VAR_SET_SCOPE_LOCAL) {
		throw NotImplementedException("RESET LOCAL is not implemented.");
	}

	auto name = std::string(stmt->name);
	D_ASSERT(!name.empty()); // parser protect us!

	return make_uniq<ResetVariableStatement>(name, ToSetScope(stmt->scope));
}

unique_ptr<SetStatement> Transformer::TransformSet(duckdb_libpgquery::PGNode *node) {
	D_ASSERT(node->type == duckdb_libpgquery::T_PGVariableSetStmt);
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGVariableSetStmt *>(node);

	SetType set_type = ToSetType(stmt->kind);

	switch (set_type) {
	case SetType::SET:
		return TransformSetVariable(stmt);
	case SetType::RESET:
		return TransformResetVariable(stmt);
	default:
		throw NotImplementedException("Type not implemented for SetType");
	}
}

} // namespace duckdb







namespace duckdb {

static void TransformShowName(unique_ptr<PragmaStatement> &result, const string &name) {
	auto &info = *result->info;
	auto lname = StringUtil::Lower(name);

	if (lname == "\"databases\"") {
		info.name = "show_databases";
	} else if (lname == "\"tables\"") {
		// show all tables
		info.name = "show_tables";
	} else if (lname == "__show_tables_expanded") {
		info.name = "show_tables_expanded";
	} else {
		// show one specific table
		info.name = "show";
		info.parameters.emplace_back(name);
	}
}

unique_ptr<SQLStatement> Transformer::TransformShow(duckdb_libpgquery::PGNode *node) {
	// we transform SHOW x into PRAGMA SHOW('x')

	auto stmt = reinterpret_cast<duckdb_libpgquery::PGVariableShowStmt *>(node);
	if (stmt->is_summary) {
		auto result = make_uniq<ShowStatement>();
		auto &info = *result->info;
		info.is_summary = stmt->is_summary;

		auto select = make_uniq<SelectNode>();
		select->select_list.push_back(make_uniq<StarExpression>());
		auto basetable = make_uniq<BaseTableRef>();
		auto qualified_name = QualifiedName::Parse(stmt->name);
		basetable->schema_name = qualified_name.schema;
		basetable->table_name = qualified_name.name;
		select->from_table = std::move(basetable);

		info.query = std::move(select);
		return std::move(result);
	}

	auto result = make_uniq<PragmaStatement>();

	auto show_name = stmt->name;
	TransformShowName(result, show_name);
	return std::move(result);
}

} // namespace duckdb




namespace duckdb {

unique_ptr<ShowStatement> Transformer::TransformShowSelect(duckdb_libpgquery::PGNode *node) {
	// we capture the select statement of SHOW
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGVariableShowSelectStmt *>(node);
	auto select_stmt = reinterpret_cast<duckdb_libpgquery::PGSelectStmt *>(stmt->stmt);

	auto result = make_uniq<ShowStatement>();
	auto &info = *result->info;
	info.is_summary = stmt->is_summary;

	info.query = TransformSelectNode(select_stmt);

	return result;
}

} // namespace duckdb



namespace duckdb {

unique_ptr<TransactionStatement> Transformer::TransformTransaction(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGTransactionStmt *>(node);
	D_ASSERT(stmt);
	switch (stmt->kind) {
	case duckdb_libpgquery::PG_TRANS_STMT_BEGIN:
	case duckdb_libpgquery::PG_TRANS_STMT_START:
		return make_uniq<TransactionStatement>(TransactionType::BEGIN_TRANSACTION);
	case duckdb_libpgquery::PG_TRANS_STMT_COMMIT:
		return make_uniq<TransactionStatement>(TransactionType::COMMIT);
	case duckdb_libpgquery::PG_TRANS_STMT_ROLLBACK:
		return make_uniq<TransactionStatement>(TransactionType::ROLLBACK);
	default:
		throw NotImplementedException("Transaction type %d not implemented yet", stmt->kind);
	}
}

} // namespace duckdb



namespace duckdb {

unique_ptr<UpdateSetInfo> Transformer::TransformUpdateSetInfo(duckdb_libpgquery::PGList *target_list,
                                                              duckdb_libpgquery::PGNode *where_clause) {
	auto result = make_uniq<UpdateSetInfo>();

	auto root = target_list;
	for (auto cell = root->head; cell != nullptr; cell = cell->next) {
		auto target = (duckdb_libpgquery::PGResTarget *)(cell->data.ptr_value);
		result->columns.emplace_back(target->name);
		result->expressions.push_back(TransformExpression(target->val));
	}
	result->condition = TransformExpression(where_clause);
	return result;
}

unique_ptr<UpdateStatement> Transformer::TransformUpdate(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGUpdateStmt *>(node);
	D_ASSERT(stmt);

	auto result = make_uniq<UpdateStatement>();
	if (stmt->withClause) {
		TransformCTE(reinterpret_cast<duckdb_libpgquery::PGWithClause *>(stmt->withClause), result->cte_map);
	}

	result->table = TransformRangeVar(stmt->relation);
	if (stmt->fromClause) {
		result->from_table = TransformFrom(stmt->fromClause);
	}

	result->set_info = TransformUpdateSetInfo(stmt->targetList, stmt->whereClause);

	// Grab and transform the returning columns from the parser.
	if (stmt->returningList) {
		Transformer::TransformExpressionList(*(stmt->returningList), result->returning_list);
	}

	return result;
}

} // namespace duckdb





namespace duckdb {

OnConflictAction TransformOnConflictAction(duckdb_libpgquery::PGOnConflictClause *on_conflict) {
	if (!on_conflict) {
		return OnConflictAction::THROW;
	}
	switch (on_conflict->action) {
	case duckdb_libpgquery::PG_ONCONFLICT_NONE:
		return OnConflictAction::THROW;
	case duckdb_libpgquery::PG_ONCONFLICT_NOTHING:
		return OnConflictAction::NOTHING;
	case duckdb_libpgquery::PG_ONCONFLICT_UPDATE:
		return OnConflictAction::UPDATE;
	default:
		throw InternalException("Type not implemented for OnConflictAction");
	}
}

vector<string> TransformConflictTarget(duckdb_libpgquery::PGList *list) {
	vector<string> columns;
	for (auto cell = list->head; cell != nullptr; cell = cell->next) {
		auto index_element = (duckdb_libpgquery::PGIndexElem *)cell->data.ptr_value;
		if (index_element->collation) {
			throw NotImplementedException("Index with collation not supported yet!");
		}
		if (index_element->opclass) {
			throw NotImplementedException("Index with opclass not supported yet!");
		}
		if (!index_element->name) {
			throw NotImplementedException("Non-column index element not supported yet!");
		}
		if (index_element->nulls_ordering) {
			throw NotImplementedException("Index with null_ordering not supported yet!");
		}
		if (index_element->ordering) {
			throw NotImplementedException("Index with ordering not supported yet!");
		}
		columns.emplace_back(index_element->name);
	}
	return columns;
}

unique_ptr<OnConflictInfo> Transformer::DummyOnConflictClause(duckdb_libpgquery::PGOnConflictActionAlias type,
                                                              const string &relname) {
	switch (type) {
	case duckdb_libpgquery::PGOnConflictActionAlias::PG_ONCONFLICT_ALIAS_REPLACE: {
		// This can not be fully resolved yet until the bind stage
		auto result = make_uniq<OnConflictInfo>();
		result->action_type = OnConflictAction::REPLACE;
		return result;
	}
	case duckdb_libpgquery::PGOnConflictActionAlias::PG_ONCONFLICT_ALIAS_IGNORE: {
		// We can just fully replace this with DO NOTHING, and be done with it
		auto result = make_uniq<OnConflictInfo>();
		result->action_type = OnConflictAction::NOTHING;
		return result;
	}
	default: {
		throw InternalException("Type not implemented for PGOnConflictActionAlias");
	}
	}
}

unique_ptr<OnConflictInfo> Transformer::TransformOnConflictClause(duckdb_libpgquery::PGOnConflictClause *node,
                                                                  const string &relname) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGOnConflictClause *>(node);
	D_ASSERT(stmt);

	auto result = make_uniq<OnConflictInfo>();
	result->action_type = TransformOnConflictAction(stmt);
	if (stmt->infer) {
		// A filter for the ON CONFLICT ... is specified
		if (stmt->infer->indexElems) {
			// Columns are specified
			result->indexed_columns = TransformConflictTarget(stmt->infer->indexElems);
			if (stmt->infer->whereClause) {
				result->condition = TransformExpression(stmt->infer->whereClause);
			}
		} else {
			throw NotImplementedException("ON CONSTRAINT conflict target is not supported yet");
		}
	}

	if (result->action_type == OnConflictAction::UPDATE) {
		result->set_info = TransformUpdateSetInfo(stmt->targetList, stmt->whereClause);
	}
	return result;
}

} // namespace duckdb



namespace duckdb {

unique_ptr<SetStatement> Transformer::TransformUse(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGUseStmt *>(node);
	auto qualified_name = TransformQualifiedName(stmt->name);
	if (!IsInvalidCatalog(qualified_name.catalog)) {
		throw ParserException("Expected \"USE database\" or \"USE database.schema\"");
	}
	string name;
	if (IsInvalidSchema(qualified_name.schema)) {
		name = qualified_name.name;
	} else {
		name = qualified_name.schema + "." + qualified_name.name;
	}
	return make_uniq<SetVariableStatement>("schema", std::move(name), SetScope::AUTOMATIC);
}

} // namespace duckdb



namespace duckdb {

VacuumOptions ParseOptions(int options) {
	VacuumOptions result;
	if (options & duckdb_libpgquery::PGVacuumOption::PG_VACOPT_VACUUM) {
		result.vacuum = true;
	}
	if (options & duckdb_libpgquery::PGVacuumOption::PG_VACOPT_ANALYZE) {
		result.analyze = true;
	}
	if (options & duckdb_libpgquery::PGVacuumOption::PG_VACOPT_VERBOSE) {
		throw NotImplementedException("Verbose vacuum option");
	}
	if (options & duckdb_libpgquery::PGVacuumOption::PG_VACOPT_FREEZE) {
		throw NotImplementedException("Freeze vacuum option");
	}
	if (options & duckdb_libpgquery::PGVacuumOption::PG_VACOPT_FULL) {
		throw NotImplementedException("Full vacuum option");
	}
	if (options & duckdb_libpgquery::PGVacuumOption::PG_VACOPT_NOWAIT) {
		throw NotImplementedException("No Wait vacuum option");
	}
	if (options & duckdb_libpgquery::PGVacuumOption::PG_VACOPT_SKIPTOAST) {
		throw NotImplementedException("Skip Toast vacuum option");
	}
	if (options & duckdb_libpgquery::PGVacuumOption::PG_VACOPT_DISABLE_PAGE_SKIPPING) {
		throw NotImplementedException("Disable Page Skipping vacuum option");
	}
	return result;
}

unique_ptr<SQLStatement> Transformer::TransformVacuum(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGVacuumStmt *>(node);
	D_ASSERT(stmt);

	auto result = make_uniq<VacuumStatement>(ParseOptions(stmt->options));

	if (stmt->relation) {
		result->info->ref = TransformRangeVar(stmt->relation);
		result->info->has_table = true;
	}

	if (stmt->va_cols) {
		D_ASSERT(result->info->has_table);
		for (auto col_node = stmt->va_cols->head; col_node != nullptr; col_node = col_node->next) {
			result->info->columns.emplace_back(
			    reinterpret_cast<duckdb_libpgquery::PGValue *>(col_node->data.ptr_value)->val.str);
		}
	}

	return std::move(result);
}

} // namespace duckdb



namespace duckdb {

unique_ptr<TableRef> Transformer::TransformRangeVar(duckdb_libpgquery::PGRangeVar *root) {
	auto result = make_uniq<BaseTableRef>();

	result->alias = TransformAlias(root->alias, result->column_name_alias);
	if (root->relname) {
		result->table_name = root->relname;
	}
	if (root->catalogname) {
		result->catalog_name = root->catalogname;
	}
	if (root->schemaname) {
		result->schema_name = root->schemaname;
	}
	if (root->sample) {
		result->sample = TransformSampleOptions(root->sample);
	}
	result->query_location = root->location;
	return std::move(result);
}

QualifiedName Transformer::TransformQualifiedName(duckdb_libpgquery::PGRangeVar *root) {
	QualifiedName qname;
	if (root->catalogname) {
		qname.catalog = root->catalogname;
	} else {
		qname.catalog = INVALID_CATALOG;
	}
	if (root->schemaname) {
		qname.schema = root->schemaname;
	} else {
		qname.schema = INVALID_SCHEMA;
	}
	if (root->relname) {
		qname.name = root->relname;
	} else {
		qname.name = string();
	}
	return qname;
}

} // namespace duckdb




namespace duckdb {

unique_ptr<TableRef> Transformer::TransformFrom(duckdb_libpgquery::PGList *root) {
	if (!root) {
		return make_uniq<EmptyTableRef>();
	}

	if (root->length > 1) {
		// Cross Product
		auto result = make_uniq<JoinRef>(JoinRefType::CROSS);
		JoinRef *cur_root = result.get();
		idx_t list_size = 0;
		for (auto node = root->head; node != nullptr; node = node->next) {
			auto n = reinterpret_cast<duckdb_libpgquery::PGNode *>(node->data.ptr_value);
			unique_ptr<TableRef> next = TransformTableRefNode(n);
			if (!cur_root->left) {
				cur_root->left = std::move(next);
			} else if (!cur_root->right) {
				cur_root->right = std::move(next);
			} else {
				auto old_res = std::move(result);
				result = make_uniq<JoinRef>(JoinRefType::CROSS);
				result->left = std::move(old_res);
				result->right = std::move(next);
				cur_root = result.get();
			}
			list_size++;
			StackCheck(list_size);
		}
		return std::move(result);
	}

	auto n = reinterpret_cast<duckdb_libpgquery::PGNode *>(root->head->data.ptr_value);
	return TransformTableRefNode(n);
}

} // namespace duckdb





namespace duckdb {

unique_ptr<TableRef> Transformer::TransformJoin(duckdb_libpgquery::PGJoinExpr *root) {
	auto result = make_uniq<JoinRef>(JoinRefType::REGULAR);
	switch (root->jointype) {
	case duckdb_libpgquery::PG_JOIN_INNER: {
		result->type = JoinType::INNER;
		break;
	}
	case duckdb_libpgquery::PG_JOIN_LEFT: {
		result->type = JoinType::LEFT;
		break;
	}
	case duckdb_libpgquery::PG_JOIN_FULL: {
		result->type = JoinType::OUTER;
		break;
	}
	case duckdb_libpgquery::PG_JOIN_RIGHT: {
		result->type = JoinType::RIGHT;
		break;
	}
	case duckdb_libpgquery::PG_JOIN_SEMI: {
		result->type = JoinType::SEMI;
		break;
	}
	case duckdb_libpgquery::PG_JOIN_ANTI: {
		result->type = JoinType::ANTI;
		break;
	}
	case duckdb_libpgquery::PG_JOIN_POSITION: {
		result->ref_type = JoinRefType::POSITIONAL;
		break;
	}
	default: {
		throw NotImplementedException("Join type %d not supported\n", root->jointype);
	}
	}

	// Check the type of left arg and right arg before transform
	result->left = TransformTableRefNode(root->larg);
	result->right = TransformTableRefNode(root->rarg);
	switch (root->joinreftype) {
	case duckdb_libpgquery::PG_JOIN_NATURAL:
		result->ref_type = JoinRefType::NATURAL;
		break;
	case duckdb_libpgquery::PG_JOIN_ASOF:
		result->ref_type = JoinRefType::ASOF;
		break;
	default:
		break;
	}
	result->query_location = root->location;

	if (root->usingClause && root->usingClause->length > 0) {
		// usingClause is a list of strings
		for (auto node = root->usingClause->head; node != nullptr; node = node->next) {
			auto target = reinterpret_cast<duckdb_libpgquery::PGNode *>(node->data.ptr_value);
			D_ASSERT(target->type == duckdb_libpgquery::T_PGString);
			auto column_name = string(reinterpret_cast<duckdb_libpgquery::PGValue *>(target)->val.str);
			result->using_columns.push_back(column_name);
		}
		return std::move(result);
	}

	if (!root->quals && result->using_columns.empty() && result->ref_type == JoinRefType::REGULAR) { // CROSS PRODUCT
		result->ref_type = JoinRefType::CROSS;
	}
	result->condition = TransformExpression(root->quals);
	return std::move(result);
}

} // namespace duckdb







namespace duckdb {

static void TransformPivotInList(unique_ptr<ParsedExpression> &expr, PivotColumnEntry &entry, bool root_entry = true) {
	if (expr->type == ExpressionType::COLUMN_REF) {
		auto &colref = expr->Cast<ColumnRefExpression>();
		if (colref.IsQualified()) {
			throw ParserException("PIVOT IN list cannot contain qualified column references");
		}
		entry.values.emplace_back(colref.GetColumnName());
	} else if (expr->type == ExpressionType::VALUE_CONSTANT) {
		auto &constant_expr = expr->Cast<ConstantExpression>();
		entry.values.push_back(std::move(constant_expr.value));
	} else if (root_entry && expr->type == ExpressionType::FUNCTION) {
		auto &function = expr->Cast<FunctionExpression>();
		if (function.function_name != "row") {
			throw ParserException("PIVOT IN list must contain columns or lists of columns");
		}
		for (auto &child : function.children) {
			TransformPivotInList(child, entry, false);
		}
	} else if (root_entry && expr->type == ExpressionType::STAR) {
		entry.star_expr = std::move(expr);
	} else {
		throw ParserException("PIVOT IN list must contain columns or lists of columns");
	}
}

PivotColumn Transformer::TransformPivotColumn(duckdb_libpgquery::PGPivot *pivot) {
	PivotColumn col;
	if (pivot->pivot_columns) {
		TransformExpressionList(*pivot->pivot_columns, col.pivot_expressions);
		for (auto &expr : col.pivot_expressions) {
			if (expr->IsScalar()) {
				throw ParserException("Cannot pivot on constant value \"%s\"", expr->ToString());
			}
			if (expr->HasSubquery()) {
				throw ParserException("Cannot pivot on subquery \"%s\"", expr->ToString());
			}
		}
	} else if (pivot->unpivot_columns) {
		col.unpivot_names = TransformStringList(pivot->unpivot_columns);
	} else {
		throw InternalException("Either pivot_columns or unpivot_columns must be defined");
	}
	if (pivot->pivot_value) {
		for (auto node = pivot->pivot_value->head; node != nullptr; node = node->next) {
			auto n = (duckdb_libpgquery::PGNode *)node->data.ptr_value;
			auto expr = TransformExpression(n);
			PivotColumnEntry entry;
			entry.alias = expr->alias;
			TransformPivotInList(expr, entry);
			col.entries.push_back(std::move(entry));
		}
	}
	if (pivot->subquery) {
		col.subquery = TransformSelectNode(reinterpret_cast<duckdb_libpgquery::PGSelectStmt *>(pivot->subquery));
	}
	if (pivot->pivot_enum) {
		col.pivot_enum = pivot->pivot_enum;
	}
	return col;
}

vector<PivotColumn> Transformer::TransformPivotList(duckdb_libpgquery::PGList *list) {
	vector<PivotColumn> result;
	for (auto node = list->head; node != nullptr; node = node->next) {
		auto pivot = (duckdb_libpgquery::PGPivot *)node->data.ptr_value;
		result.push_back(TransformPivotColumn(pivot));
	}
	return result;
}

unique_ptr<TableRef> Transformer::TransformPivot(duckdb_libpgquery::PGPivotExpr *root) {
	auto result = make_uniq<PivotRef>();
	result->source = TransformTableRefNode(root->source);
	if (root->aggrs) {
		TransformExpressionList(*root->aggrs, result->aggregates);
	}
	if (root->unpivots) {
		result->unpivot_names = TransformStringList(root->unpivots);
	}
	result->pivots = TransformPivotList(root->pivots);
	if (root->groups) {
		result->groups = TransformStringList(root->groups);
	}
	for (auto &pivot : result->pivots) {
		idx_t expected_size;
		bool is_pivot = result->unpivot_names.empty();
		if (!result->unpivot_names.empty()) {
			// unpivot
			if (pivot.unpivot_names.size() != 1) {
				throw ParserException("UNPIVOT requires a single column name for the PIVOT IN clause");
			}
			D_ASSERT(pivot.pivot_expressions.empty());
			expected_size = pivot.entries[0].values.size();
		} else {
			// pivot
			expected_size = pivot.pivot_expressions.size();
			D_ASSERT(pivot.unpivot_names.empty());
		}
		for (auto &entry : pivot.entries) {
			if (entry.star_expr && is_pivot) {
				throw ParserException("PIVOT IN list must contain columns or lists of columns - star expressions are "
				                      "only supported for UNPIVOT");
			}
			if (entry.values.size() != expected_size) {
				throw ParserException("PIVOT IN list - inconsistent amount of rows - expected %d but got %d",
				                      expected_size, entry.values.size());
			}
		}
	}
	result->include_nulls = root->include_nulls;
	result->alias = TransformAlias(root->alias, result->column_name_alias);
	return std::move(result);
}

} // namespace duckdb



namespace duckdb {

unique_ptr<TableRef> Transformer::TransformRangeSubselect(duckdb_libpgquery::PGRangeSubselect *root) {
	Transformer subquery_transformer(*this);
	auto subquery = subquery_transformer.TransformSelect(root->subquery);
	if (!subquery) {
		return nullptr;
	}
	auto result = make_uniq<SubqueryRef>(std::move(subquery));
	result->alias = TransformAlias(root->alias, result->column_name_alias);
	if (root->sample) {
		result->sample = TransformSampleOptions(root->sample);
	}
	return std::move(result);
}

} // namespace duckdb




namespace duckdb {

unique_ptr<TableRef> Transformer::TransformRangeFunction(duckdb_libpgquery::PGRangeFunction *root) {
	if (root->ordinality) {
		throw NotImplementedException("WITH ORDINALITY not implemented");
	}
	if (root->is_rowsfrom) {
		throw NotImplementedException("ROWS FROM() not implemented");
	}
	if (root->functions->length != 1) {
		throw NotImplementedException("Need exactly one function");
	}
	auto function_sublist = (duckdb_libpgquery::PGList *)root->functions->head->data.ptr_value;
	D_ASSERT(function_sublist->length == 2);
	auto call_tree = (duckdb_libpgquery::PGNode *)function_sublist->head->data.ptr_value;
	auto coldef = function_sublist->head->next->data.ptr_value;

	if (coldef) {
		throw NotImplementedException("Explicit column definition not supported yet");
	}
	// transform the function call
	auto result = make_uniq<TableFunctionRef>();
	switch (call_tree->type) {
	case duckdb_libpgquery::T_PGFuncCall: {
		auto func_call = (duckdb_libpgquery::PGFuncCall *)call_tree;
		result->function = TransformFuncCall(func_call);
		result->query_location = func_call->location;
		break;
	}
	case duckdb_libpgquery::T_PGSQLValueFunction:
		result->function = TransformSQLValueFunction((duckdb_libpgquery::PGSQLValueFunction *)call_tree);
		break;
	default:
		throw ParserException("Not a function call or value function");
	}
	result->alias = TransformAlias(root->alias, result->column_name_alias);
	if (root->sample) {
		result->sample = TransformSampleOptions(root->sample);
	}
	return std::move(result);
}

} // namespace duckdb




namespace duckdb {

unique_ptr<TableRef> Transformer::TransformTableRefNode(duckdb_libpgquery::PGNode *n) {
	auto stack_checker = StackCheck();

	switch (n->type) {
	case duckdb_libpgquery::T_PGRangeVar:
		return TransformRangeVar(reinterpret_cast<duckdb_libpgquery::PGRangeVar *>(n));
	case duckdb_libpgquery::T_PGJoinExpr:
		return TransformJoin(reinterpret_cast<duckdb_libpgquery::PGJoinExpr *>(n));
	case duckdb_libpgquery::T_PGRangeSubselect:
		return TransformRangeSubselect(reinterpret_cast<duckdb_libpgquery::PGRangeSubselect *>(n));
	case duckdb_libpgquery::T_PGRangeFunction:
		return TransformRangeFunction(reinterpret_cast<duckdb_libpgquery::PGRangeFunction *>(n));
	case duckdb_libpgquery::T_PGPivotExpr:
		return TransformPivot(reinterpret_cast<duckdb_libpgquery::PGPivotExpr *>(n));
	default:
		throw NotImplementedException("From Type %d not supported", n->type);
	}
}

} // namespace duckdb








namespace duckdb {

StackChecker::StackChecker(Transformer &transformer_p, idx_t stack_usage_p)
    : transformer(transformer_p), stack_usage(stack_usage_p) {
	transformer.stack_depth += stack_usage;
}

StackChecker::~StackChecker() {
	transformer.stack_depth -= stack_usage;
}

StackChecker::StackChecker(StackChecker &&other) noexcept
    : transformer(other.transformer), stack_usage(other.stack_usage) {
	other.stack_usage = 0;
}

Transformer::Transformer(ParserOptions &options)
    : parent(nullptr), options(options), stack_depth(DConstants::INVALID_INDEX) {
}

Transformer::Transformer(Transformer &parent)
    : parent(&parent), options(parent.options), stack_depth(DConstants::INVALID_INDEX) {
}

Transformer::~Transformer() {
}

void Transformer::Clear() {
	SetParamCount(0);
	pivot_entries.clear();
}

bool Transformer::TransformParseTree(duckdb_libpgquery::PGList *tree, vector<unique_ptr<SQLStatement>> &statements) {
	InitializeStackCheck();
	for (auto entry = tree->head; entry != nullptr; entry = entry->next) {
		Clear();
		auto n = (duckdb_libpgquery::PGNode *)entry->data.ptr_value;
		auto stmt = TransformStatement(n);
		D_ASSERT(stmt);
		if (HasPivotEntries()) {
			stmt = CreatePivotStatement(std::move(stmt));
		}
		stmt->n_param = ParamCount();
		statements.push_back(std::move(stmt));
	}
	return true;
}

void Transformer::InitializeStackCheck() {
	stack_depth = 0;
}

StackChecker Transformer::StackCheck(idx_t extra_stack) {
	auto &root = RootTransformer();
	D_ASSERT(root.stack_depth != DConstants::INVALID_INDEX);
	if (root.stack_depth + extra_stack >= options.max_expression_depth) {
		throw ParserException("Max expression depth limit of %lld exceeded. Use \"SET max_expression_depth TO x\" to "
		                      "increase the maximum expression depth.",
		                      options.max_expression_depth);
	}
	return StackChecker(root, extra_stack);
}

unique_ptr<SQLStatement> Transformer::TransformStatement(duckdb_libpgquery::PGNode *stmt) {
	auto result = TransformStatementInternal(stmt);
	result->n_param = ParamCount();
	if (!named_param_map.empty()) {
		// Avoid overriding a previous move with nothing
		result->named_param_map = std::move(named_param_map);
	}
	return result;
}

Transformer &Transformer::RootTransformer() {
	reference<Transformer> node = *this;
	while (node.get().parent) {
		node = *node.get().parent;
	}
	return node.get();
}

const Transformer &Transformer::RootTransformer() const {
	reference<const Transformer> node = *this;
	while (node.get().parent) {
		node = *node.get().parent;
	}
	return node.get();
}

idx_t Transformer::ParamCount() const {
	auto &root = RootTransformer();
	return root.prepared_statement_parameter_index;
}

void Transformer::SetParamCount(idx_t new_count) {
	auto &root = RootTransformer();
	root.prepared_statement_parameter_index = new_count;
}
void Transformer::SetNamedParam(const string &name, int32_t index) {
	auto &root = RootTransformer();
	D_ASSERT(!root.named_param_map.count(name));
	root.named_param_map[name] = index;
}
bool Transformer::GetNamedParam(const string &name, int32_t &index) {
	auto &root = RootTransformer();
	auto entry = root.named_param_map.find(name);
	if (entry == root.named_param_map.end()) {
		return false;
	}
	index = entry->second;
	return true;
}
bool Transformer::HasNamedParameters() const {
	auto &root = RootTransformer();
	return !root.named_param_map.empty();
}

unique_ptr<SQLStatement> Transformer::TransformStatementInternal(duckdb_libpgquery::PGNode *stmt) {
	switch (stmt->type) {
	case duckdb_libpgquery::T_PGRawStmt: {
		auto raw_stmt = (duckdb_libpgquery::PGRawStmt *)stmt;
		auto result = TransformStatement(raw_stmt->stmt);
		if (result) {
			result->stmt_location = raw_stmt->stmt_location;
			result->stmt_length = raw_stmt->stmt_len;
		}
		return result;
	}
	case duckdb_libpgquery::T_PGSelectStmt:
		return TransformSelect(stmt);
	case duckdb_libpgquery::T_PGCreateStmt:
		return TransformCreateTable(stmt);
	case duckdb_libpgquery::T_PGCreateSchemaStmt:
		return TransformCreateSchema(stmt);
	case duckdb_libpgquery::T_PGViewStmt:
		return TransformCreateView(stmt);
	case duckdb_libpgquery::T_PGCreateSeqStmt:
		return TransformCreateSequence(stmt);
	case duckdb_libpgquery::T_PGCreateFunctionStmt:
		return TransformCreateFunction(stmt);
	case duckdb_libpgquery::T_PGDropStmt:
		return TransformDrop(stmt);
	case duckdb_libpgquery::T_PGInsertStmt:
		return TransformInsert(stmt);
	case duckdb_libpgquery::T_PGCopyStmt:
		return TransformCopy(stmt);
	case duckdb_libpgquery::T_PGTransactionStmt:
		return TransformTransaction(stmt);
	case duckdb_libpgquery::T_PGDeleteStmt:
		return TransformDelete(stmt);
	case duckdb_libpgquery::T_PGUpdateStmt:
		return TransformUpdate(stmt);
	case duckdb_libpgquery::T_PGIndexStmt:
		return TransformCreateIndex(stmt);
	case duckdb_libpgquery::T_PGAlterTableStmt:
		return TransformAlter(stmt);
	case duckdb_libpgquery::T_PGRenameStmt:
		return TransformRename(stmt);
	case duckdb_libpgquery::T_PGPrepareStmt:
		return TransformPrepare(stmt);
	case duckdb_libpgquery::T_PGExecuteStmt:
		return TransformExecute(stmt);
	case duckdb_libpgquery::T_PGDeallocateStmt:
		return TransformDeallocate(stmt);
	case duckdb_libpgquery::T_PGCreateTableAsStmt:
		return TransformCreateTableAs(stmt);
	case duckdb_libpgquery::T_PGPragmaStmt:
		return TransformPragma(stmt);
	case duckdb_libpgquery::T_PGExportStmt:
		return TransformExport(stmt);
	case duckdb_libpgquery::T_PGImportStmt:
		return TransformImport(stmt);
	case duckdb_libpgquery::T_PGExplainStmt:
		return TransformExplain(stmt);
	case duckdb_libpgquery::T_PGVacuumStmt:
		return TransformVacuum(stmt);
	case duckdb_libpgquery::T_PGVariableShowStmt:
		return TransformShow(stmt);
	case duckdb_libpgquery::T_PGVariableShowSelectStmt:
		return TransformShowSelect(stmt);
	case duckdb_libpgquery::T_PGCallStmt:
		return TransformCall(stmt);
	case duckdb_libpgquery::T_PGVariableSetStmt:
		return TransformSet(stmt);
	case duckdb_libpgquery::T_PGCheckPointStmt:
		return TransformCheckpoint(stmt);
	case duckdb_libpgquery::T_PGLoadStmt:
		return TransformLoad(stmt);
	case duckdb_libpgquery::T_PGCreateTypeStmt:
		return TransformCreateType(stmt);
	case duckdb_libpgquery::T_PGAlterSeqStmt:
		return TransformAlterSequence(stmt);
	case duckdb_libpgquery::T_PGAttachStmt:
		return TransformAttach(stmt);
	case duckdb_libpgquery::T_PGDetachStmt:
		return TransformDetach(stmt);
	case duckdb_libpgquery::T_PGUseStmt:
		return TransformUse(stmt);
	case duckdb_libpgquery::T_PGCreateDatabaseStmt:
		return TransformCreateDatabase(stmt);
	default:
		throw NotImplementedException(NodetypeToString(stmt->type));
	}
}

} // namespace duckdb

















#include <algorithm>

namespace duckdb {

string BindContext::GetMatchingBinding(const string &column_name) {
	string result;
	for (auto &kv : bindings) {
		auto binding = kv.second.get();
		auto is_using_binding = GetUsingBinding(column_name, kv.first);
		if (is_using_binding) {
			continue;
		}
		if (binding->HasMatchingBinding(column_name)) {
			if (!result.empty() || is_using_binding) {
				throw BinderException("Ambiguous reference to column name \"%s\" (use: \"%s.%s\" "
				                      "or \"%s.%s\")",
				                      column_name, result, column_name, kv.first, column_name);
			}
			result = kv.first;
		}
	}
	return result;
}

vector<string> BindContext::GetSimilarBindings(const string &column_name) {
	vector<pair<string, idx_t>> scores;
	for (auto &kv : bindings) {
		auto binding = kv.second.get();
		for (auto &name : binding->names) {
			idx_t distance = StringUtil::SimilarityScore(name, column_name);
			scores.emplace_back(binding->alias + "." + name, distance);
		}
	}
	return StringUtil::TopNStrings(scores);
}

void BindContext::AddUsingBinding(const string &column_name, UsingColumnSet &set) {
	using_columns[column_name].insert(set);
}

void BindContext::AddUsingBindingSet(unique_ptr<UsingColumnSet> set) {
	using_column_sets.push_back(std::move(set));
}

optional_ptr<UsingColumnSet> BindContext::GetUsingBinding(const string &column_name) {
	auto entry = using_columns.find(column_name);
	if (entry == using_columns.end()) {
		return nullptr;
	}
	auto &using_bindings = entry->second;
	if (using_bindings.size() > 1) {
		string error = "Ambiguous column reference: column \"" + column_name + "\" can refer to either:\n";
		for (auto &using_set_ref : using_bindings) {
			auto &using_set = using_set_ref.get();
			string result_bindings;
			for (auto &binding : using_set.bindings) {
				if (result_bindings.empty()) {
					result_bindings = "[";
				} else {
					result_bindings += ", ";
				}
				result_bindings += binding;
				result_bindings += ".";
				result_bindings += GetActualColumnName(binding, column_name);
			}
			error += result_bindings + "]";
		}
		throw BinderException(error);
	}
	for (auto &using_set : using_bindings) {
		return &using_set.get();
	}
	throw InternalException("Using binding found but no entries");
}

optional_ptr<UsingColumnSet> BindContext::GetUsingBinding(const string &column_name, const string &binding_name) {
	if (binding_name.empty()) {
		throw InternalException("GetUsingBinding: expected non-empty binding_name");
	}
	auto entry = using_columns.find(column_name);
	if (entry == using_columns.end()) {
		return nullptr;
	}
	auto &using_bindings = entry->second;
	for (auto &using_set_ref : using_bindings) {
		auto &using_set = using_set_ref.get();
		auto &bindings = using_set.bindings;
		if (bindings.find(binding_name) != bindings.end()) {
			return &using_set;
		}
	}
	return nullptr;
}

void BindContext::RemoveUsingBinding(const string &column_name, UsingColumnSet &set) {
	auto entry = using_columns.find(column_name);
	if (entry == using_columns.end()) {
		throw InternalException("Attempting to remove using binding that is not there");
	}
	auto &bindings = entry->second;
	if (bindings.find(set) != bindings.end()) {
		bindings.erase(set);
	}
	if (bindings.empty()) {
		using_columns.erase(column_name);
	}
}

void BindContext::TransferUsingBinding(BindContext &current_context, optional_ptr<UsingColumnSet> current_set,
                                       UsingColumnSet &new_set, const string &binding, const string &using_column) {
	AddUsingBinding(using_column, new_set);
	if (current_set) {
		current_context.RemoveUsingBinding(using_column, *current_set);
	}
}

string BindContext::GetActualColumnName(const string &binding_name, const string &column_name) {
	string error;
	auto binding = GetBinding(binding_name, error);
	if (!binding) {
		throw InternalException("No binding with name \"%s\"", binding_name);
	}
	column_t binding_index;
	if (!binding->TryGetBindingIndex(column_name, binding_index)) { // LCOV_EXCL_START
		throw InternalException("Binding with name \"%s\" does not have a column named \"%s\"", binding_name,
		                        column_name);
	} // LCOV_EXCL_STOP
	return binding->names[binding_index];
}

unordered_set<string> BindContext::GetMatchingBindings(const string &column_name) {
	unordered_set<string> result;
	for (auto &kv : bindings) {
		auto binding = kv.second.get();
		if (binding->HasMatchingBinding(column_name)) {
			result.insert(kv.first);
		}
	}
	return result;
}

unique_ptr<ParsedExpression> BindContext::ExpandGeneratedColumn(const string &table_name, const string &column_name) {
	string error_message;

	auto binding = GetBinding(table_name, error_message);
	D_ASSERT(binding);
	auto &table_binding = (TableBinding &)*binding;
	auto result = table_binding.ExpandGeneratedColumn(column_name);
	result->alias = column_name;
	return result;
}

unique_ptr<ParsedExpression> BindContext::CreateColumnReference(const string &table_name, const string &column_name) {
	string schema_name;
	return CreateColumnReference(schema_name, table_name, column_name);
}

static bool ColumnIsGenerated(Binding &binding, column_t index) {
	if (binding.binding_type != BindingType::TABLE) {
		return false;
	}
	auto &table_binding = (TableBinding &)binding;
	auto catalog_entry = table_binding.GetStandardEntry();
	if (!catalog_entry) {
		return false;
	}
	if (index == COLUMN_IDENTIFIER_ROW_ID) {
		return false;
	}
	D_ASSERT(catalog_entry->type == CatalogType::TABLE_ENTRY);
	auto &table_entry = catalog_entry->Cast<TableCatalogEntry>();
	return table_entry.GetColumn(LogicalIndex(index)).Generated();
}

unique_ptr<ParsedExpression> BindContext::CreateColumnReference(const string &catalog_name, const string &schema_name,
                                                                const string &table_name, const string &column_name) {
	string error_message;
	vector<string> names;
	if (!catalog_name.empty()) {
		names.push_back(catalog_name);
	}
	if (!schema_name.empty()) {
		names.push_back(schema_name);
	}
	names.push_back(table_name);
	names.push_back(column_name);

	auto result = make_uniq<ColumnRefExpression>(std::move(names));
	auto binding = GetBinding(table_name, error_message);
	if (!binding) {
		return std::move(result);
	}
	auto column_index = binding->GetBindingIndex(column_name);
	if (ColumnIsGenerated(*binding, column_index)) {
		return ExpandGeneratedColumn(table_name, column_name);
	} else if (column_index < binding->names.size() && binding->names[column_index] != column_name) {
		// because of case insensitivity in the binder we rename the column to the original name
		// as it appears in the binding itself
		result->alias = binding->names[column_index];
	}
	return std::move(result);
}

unique_ptr<ParsedExpression> BindContext::CreateColumnReference(const string &schema_name, const string &table_name,
                                                                const string &column_name) {
	string catalog_name;
	return CreateColumnReference(catalog_name, schema_name, table_name, column_name);
}

optional_ptr<Binding> BindContext::GetCTEBinding(const string &ctename) {
	auto match = cte_bindings.find(ctename);
	if (match == cte_bindings.end()) {
		return nullptr;
	}
	return match->second.get();
}

optional_ptr<Binding> BindContext::GetBinding(const string &name, string &out_error) {
	auto match = bindings.find(name);
	if (match == bindings.end()) {
		// alias not found in this BindContext
		vector<string> candidates;
		for (auto &kv : bindings) {
			candidates.push_back(kv.first);
		}
		string candidate_str =
		    StringUtil::CandidatesMessage(StringUtil::TopNLevenshtein(candidates, name), "Candidate tables");
		out_error = StringUtil::Format("Referenced table \"%s\" not found!%s", name, candidate_str);
		return nullptr;
	}
	return match->second.get();
}

BindResult BindContext::BindColumn(ColumnRefExpression &colref, idx_t depth) {
	if (!colref.IsQualified()) {
		throw InternalException("Could not bind alias \"%s\"!", colref.GetColumnName());
	}

	string error;
	auto binding = GetBinding(colref.GetTableName(), error);
	if (!binding) {
		return BindResult(error);
	}
	return binding->Bind(colref, depth);
}

string BindContext::BindColumn(PositionalReferenceExpression &ref, string &table_name, string &column_name) {
	idx_t total_columns = 0;
	idx_t current_position = ref.index - 1;
	for (auto &entry : bindings_list) {
		auto &binding = entry.get();
		idx_t entry_column_count = binding.names.size();
		if (ref.index == 0) {
			// this is a row id
			table_name = binding.alias;
			column_name = "rowid";
			return string();
		}
		if (current_position < entry_column_count) {
			table_name = binding.alias;
			column_name = binding.names[current_position];
			return string();
		} else {
			total_columns += entry_column_count;
			current_position -= entry_column_count;
		}
	}
	return StringUtil::Format("Positional reference %d out of range (total %d columns)", ref.index, total_columns);
}

BindResult BindContext::BindColumn(PositionalReferenceExpression &ref, idx_t depth) {
	string table_name, column_name;

	string error = BindColumn(ref, table_name, column_name);
	if (!error.empty()) {
		return BindResult(error);
	}
	auto column_ref = make_uniq<ColumnRefExpression>(column_name, table_name);
	return BindColumn(*column_ref, depth);
}

bool BindContext::CheckExclusionList(StarExpression &expr, const string &column_name,
                                     vector<unique_ptr<ParsedExpression>> &new_select_list,
                                     case_insensitive_set_t &excluded_columns) {
	if (expr.exclude_list.find(column_name) != expr.exclude_list.end()) {
		excluded_columns.insert(column_name);
		return true;
	}
	auto entry = expr.replace_list.find(column_name);
	if (entry != expr.replace_list.end()) {
		auto new_entry = entry->second->Copy();
		new_entry->alias = entry->first;
		excluded_columns.insert(entry->first);
		new_select_list.push_back(std::move(new_entry));
		return true;
	}
	return false;
}

void BindContext::GenerateAllColumnExpressions(StarExpression &expr,
                                               vector<unique_ptr<ParsedExpression>> &new_select_list) {
	if (bindings_list.empty()) {
		throw BinderException("* expression without FROM clause!");
	}
	case_insensitive_set_t excluded_columns;
	if (expr.relation_name.empty()) {
		// SELECT * case
		// bind all expressions of each table in-order
		reference_set_t<UsingColumnSet> handled_using_columns;
		for (auto &entry : bindings_list) {
			auto &binding = entry.get();
			for (auto &column_name : binding.names) {
				if (CheckExclusionList(expr, column_name, new_select_list, excluded_columns)) {
					continue;
				}
				// check if this column is a USING column
				auto using_binding_ptr = GetUsingBinding(column_name, binding.alias);
				if (using_binding_ptr) {
					auto &using_binding = *using_binding_ptr;
					// it is!
					// check if we have already emitted the using column
					if (handled_using_columns.find(using_binding) != handled_using_columns.end()) {
						// we have! bail out
						continue;
					}
					// we have not! output the using column
					if (using_binding.primary_binding.empty()) {
						// no primary binding: output a coalesce
						auto coalesce = make_uniq<OperatorExpression>(ExpressionType::OPERATOR_COALESCE);
						for (auto &child_binding : using_binding.bindings) {
							coalesce->children.push_back(make_uniq<ColumnRefExpression>(column_name, child_binding));
						}
						coalesce->alias = column_name;
						new_select_list.push_back(std::move(coalesce));
					} else {
						// primary binding: output the qualified column ref
						new_select_list.push_back(
						    make_uniq<ColumnRefExpression>(column_name, using_binding.primary_binding));
					}
					handled_using_columns.insert(using_binding);
					continue;
				}
				new_select_list.push_back(make_uniq<ColumnRefExpression>(column_name, binding.alias));
			}
		}
	} else {
		// SELECT tbl.* case
		// SELECT struct.* case
		string error;
		auto binding = GetBinding(expr.relation_name, error);
		bool is_struct_ref = false;
		if (!binding) {
			auto binding_name = GetMatchingBinding(expr.relation_name);
			if (binding_name.empty()) {
				throw BinderException(error);
			}
			binding = bindings[binding_name].get();
			is_struct_ref = true;
		}

		if (is_struct_ref) {
			auto col_idx = binding->GetBindingIndex(expr.relation_name);
			auto col_type = binding->types[col_idx];
			if (col_type.id() != LogicalTypeId::STRUCT) {
				throw BinderException(StringUtil::Format(
				    "Cannot extract field from expression \"%s\" because it is not a struct", expr.ToString()));
			}
			auto &struct_children = StructType::GetChildTypes(col_type);
			vector<string> column_names(3);
			column_names[0] = binding->alias;
			column_names[1] = expr.relation_name;
			for (auto &child : struct_children) {
				if (CheckExclusionList(expr, child.first, new_select_list, excluded_columns)) {
					continue;
				}
				column_names[2] = child.first;
				new_select_list.push_back(make_uniq<ColumnRefExpression>(column_names));
			}
		} else {
			for (auto &column_name : binding->names) {
				if (CheckExclusionList(expr, column_name, new_select_list, excluded_columns)) {
					continue;
				}

				new_select_list.push_back(make_uniq<ColumnRefExpression>(column_name, binding->alias));
			}
		}
	}
	for (auto &excluded : expr.exclude_list) {
		if (excluded_columns.find(excluded) == excluded_columns.end()) {
			throw BinderException("Column \"%s\" in EXCLUDE list not found in %s", excluded,
			                      expr.relation_name.empty() ? "FROM clause" : expr.relation_name.c_str());
		}
	}
	for (auto &entry : expr.replace_list) {
		if (excluded_columns.find(entry.first) == excluded_columns.end()) {
			throw BinderException("Column \"%s\" in REPLACE list not found in %s", entry.first,
			                      expr.relation_name.empty() ? "FROM clause" : expr.relation_name.c_str());
		}
	}
}

void BindContext::GetTypesAndNames(vector<string> &result_names, vector<LogicalType> &result_types) {
	for (auto &binding_entry : bindings_list) {
		auto &binding = binding_entry.get();
		D_ASSERT(binding.names.size() == binding.types.size());
		for (idx_t i = 0; i < binding.names.size(); i++) {
			result_names.push_back(binding.names[i]);
			result_types.push_back(binding.types[i]);
		}
	}
}

void BindContext::AddBinding(const string &alias, unique_ptr<Binding> binding) {
	if (bindings.find(alias) != bindings.end()) {
		throw BinderException("Duplicate alias \"%s\" in query!", alias);
	}
	bindings_list.push_back(*binding);
	bindings[alias] = std::move(binding);
}

void BindContext::AddBaseTable(idx_t index, const string &alias, const vector<string> &names,
                               const vector<LogicalType> &types, vector<column_t> &bound_column_ids,
                               StandardEntry *entry, bool add_row_id) {
	AddBinding(alias, make_uniq<TableBinding>(alias, types, names, bound_column_ids, entry, index, add_row_id));
}

void BindContext::AddTableFunction(idx_t index, const string &alias, const vector<string> &names,
                                   const vector<LogicalType> &types, vector<column_t> &bound_column_ids,
                                   StandardEntry *entry) {
	AddBinding(alias, make_uniq<TableBinding>(alias, types, names, bound_column_ids, entry, index));
}

static string AddColumnNameToBinding(const string &base_name, case_insensitive_set_t &current_names) {
	idx_t index = 1;
	string name = base_name;
	while (current_names.find(name) != current_names.end()) {
		name = base_name + ":" + std::to_string(index++);
	}
	current_names.insert(name);
	return name;
}

vector<string> BindContext::AliasColumnNames(const string &table_name, const vector<string> &names,
                                             const vector<string> &column_aliases) {
	vector<string> result;
	if (column_aliases.size() > names.size()) {
		throw BinderException("table \"%s\" has %lld columns available but %lld columns specified", table_name,
		                      names.size(), column_aliases.size());
	}
	case_insensitive_set_t current_names;
	// use any provided column aliases first
	for (idx_t i = 0; i < column_aliases.size(); i++) {
		result.push_back(AddColumnNameToBinding(column_aliases[i], current_names));
	}
	// if not enough aliases were provided, use the default names for remaining columns
	for (idx_t i = column_aliases.size(); i < names.size(); i++) {
		result.push_back(AddColumnNameToBinding(names[i], current_names));
	}
	return result;
}

void BindContext::AddSubquery(idx_t index, const string &alias, SubqueryRef &ref, BoundQueryNode &subquery) {
	auto names = AliasColumnNames(alias, subquery.names, ref.column_name_alias);
	AddGenericBinding(index, alias, names, subquery.types);
}

void BindContext::AddEntryBinding(idx_t index, const string &alias, const vector<string> &names,
                                  const vector<LogicalType> &types, StandardEntry *entry) {
	D_ASSERT(entry);
	AddBinding(alias, make_uniq<EntryBinding>(alias, types, names, index, *entry));
}

void BindContext::AddView(idx_t index, const string &alias, SubqueryRef &ref, BoundQueryNode &subquery,
                          ViewCatalogEntry *view) {
	auto names = AliasColumnNames(alias, subquery.names, ref.column_name_alias);
	AddEntryBinding(index, alias, names, subquery.types, (StandardEntry *)view);
}

void BindContext::AddSubquery(idx_t index, const string &alias, TableFunctionRef &ref, BoundQueryNode &subquery) {
	auto names = AliasColumnNames(alias, subquery.names, ref.column_name_alias);
	AddGenericBinding(index, alias, names, subquery.types);
}

void BindContext::AddGenericBinding(idx_t index, const string &alias, const vector<string> &names,
                                    const vector<LogicalType> &types) {
	AddBinding(alias, make_uniq<Binding>(BindingType::BASE, alias, types, names, index));
}

void BindContext::AddCTEBinding(idx_t index, const string &alias, const vector<string> &names,
                                const vector<LogicalType> &types) {
	auto binding = make_shared<Binding>(BindingType::BASE, alias, types, names, index);

	if (cte_bindings.find(alias) != cte_bindings.end()) {
		throw BinderException("Duplicate alias \"%s\" in query!", alias);
	}
	cte_bindings[alias] = std::move(binding);
	cte_references[alias] = std::make_shared<idx_t>(0);
}

void BindContext::AddContext(BindContext other) {
	for (auto &binding : other.bindings) {
		if (bindings.find(binding.first) != bindings.end()) {
			throw BinderException("Duplicate alias \"%s\" in query!", binding.first);
		}
		bindings[binding.first] = std::move(binding.second);
	}
	for (auto &binding : other.bindings_list) {
		bindings_list.push_back(binding);
	}
	for (auto &entry : other.using_columns) {
		for (auto &alias : entry.second) {
#ifdef DEBUG
			for (auto &other_alias : using_columns[entry.first]) {
				for (auto &col : alias.get().bindings) {
					D_ASSERT(other_alias.get().bindings.find(col) == other_alias.get().bindings.end());
				}
			}
#endif
			using_columns[entry.first].insert(alias);
		}
	}
}

void BindContext::RemoveContext(vector<reference<Binding>> &other_bindings_list) {
	for (auto &other_binding : other_bindings_list) {
		auto it = std::remove_if(bindings_list.begin(), bindings_list.end(), [other_binding](reference<Binding> x) {
			return x.get().alias == other_binding.get().alias;
		});
		bindings_list.erase(it, bindings_list.end());
	}

	for (auto &other_binding : other_bindings_list) {
		auto &alias = other_binding.get().alias;
		if (bindings.find(alias) != bindings.end()) {
			bindings.erase(alias);
		}
	}
}

} // namespace duckdb

















namespace duckdb {

static Value NegatePercentileValue(const Value &v, const bool desc) {
	if (v.IsNull()) {
		return v;
	}

	const auto frac = v.GetValue<double>();
	if (frac < 0 || frac > 1) {
		throw BinderException("PERCENTILEs can only take parameters in the range [0, 1]");
	}

	if (!desc) {
		return v;
	}

	const auto &type = v.type();
	switch (type.id()) {
	case LogicalTypeId::DECIMAL: {
		// Negate DECIMALs as DECIMAL.
		const auto integral = IntegralValue::Get(v);
		const auto width = DecimalType::GetWidth(type);
		const auto scale = DecimalType::GetScale(type);
		switch (type.InternalType()) {
		case PhysicalType::INT16:
			return Value::DECIMAL(Cast::Operation<hugeint_t, int16_t>(-integral), width, scale);
		case PhysicalType::INT32:
			return Value::DECIMAL(Cast::Operation<hugeint_t, int32_t>(-integral), width, scale);
		case PhysicalType::INT64:
			return Value::DECIMAL(Cast::Operation<hugeint_t, int64_t>(-integral), width, scale);
		case PhysicalType::INT128:
			return Value::DECIMAL(-integral, width, scale);
		default:
			throw InternalException("Unknown DECIMAL type");
		}
	}
	default:
		// Everything else can just be a DOUBLE
		return Value::DOUBLE(-v.GetValue<double>());
	}
}

static void NegatePercentileFractions(ClientContext &context, unique_ptr<ParsedExpression> &fractions, bool desc) {
	D_ASSERT(fractions.get());
	D_ASSERT(fractions->expression_class == ExpressionClass::BOUND_EXPRESSION);
	auto &bound = BoundExpression::GetExpression(*fractions);

	if (!bound->IsFoldable()) {
		return;
	}

	Value value = ExpressionExecutor::EvaluateScalar(context, *bound);
	if (value.type().id() == LogicalTypeId::LIST) {
		vector<Value> values;
		for (const auto &element_val : ListValue::GetChildren(value)) {
			values.push_back(NegatePercentileValue(element_val, desc));
		}
		if (values.empty()) {
			throw BinderException("Empty list in percentile not allowed");
		}
		bound = make_uniq<BoundConstantExpression>(Value::LIST(values));
	} else {
		bound = make_uniq<BoundConstantExpression>(NegatePercentileValue(value, desc));
	}
}

BindResult BaseSelectBinder::BindAggregate(FunctionExpression &aggr, AggregateFunctionCatalogEntry &func, idx_t depth) {
	// first bind the child of the aggregate expression (if any)
	this->bound_aggregate = true;
	unique_ptr<Expression> bound_filter;
	AggregateBinder aggregate_binder(binder, context);
	string error, filter_error;

	// Now we bind the filter (if any)
	if (aggr.filter) {
		aggregate_binder.BindChild(aggr.filter, 0, error);
	}

	// Handle ordered-set aggregates by moving the single ORDER BY expression to the front of the children.
	//	https://www.postgresql.org/docs/current/functions-aggregate.html#FUNCTIONS-ORDEREDSET-TABLE
	bool ordered_set_agg = false;
	bool negate_fractions = false;
	if (aggr.order_bys && aggr.order_bys->orders.size() == 1) {
		const auto &func_name = aggr.function_name;
		ordered_set_agg = (func_name == "quantile_cont" || func_name == "quantile_disc" ||
		                   (func_name == "mode" && aggr.children.empty()));

		if (ordered_set_agg) {
			auto &config = DBConfig::GetConfig(context);
			const auto &order = aggr.order_bys->orders[0];
			const auto sense =
			    (order.type == OrderType::ORDER_DEFAULT) ? config.options.default_order_type : order.type;
			negate_fractions = (sense == OrderType::DESCENDING);
		}
	}

	for (auto &child : aggr.children) {
		aggregate_binder.BindChild(child, 0, error);
		// We have to negate the fractions for PERCENTILE_XXXX DESC
		if (error.empty() && ordered_set_agg) {
			NegatePercentileFractions(context, child, negate_fractions);
		}
	}

	// Bind the ORDER BYs, if any
	if (aggr.order_bys && !aggr.order_bys->orders.empty()) {
		for (auto &order : aggr.order_bys->orders) {
			aggregate_binder.BindChild(order.expression, 0, error);
		}
	}

	if (!error.empty()) {
		// failed to bind child
		if (aggregate_binder.HasBoundColumns()) {
			for (idx_t i = 0; i < aggr.children.size(); i++) {
				// however, we bound columns!
				// that means this aggregation belongs to this node
				// check if we have to resolve any errors by binding with parent binders
				bool success = aggregate_binder.BindCorrelatedColumns(aggr.children[i]);
				// if there is still an error after this, we could not successfully bind the aggregate
				if (!success) {
					throw BinderException(error);
				}
				auto &bound_expr = BoundExpression::GetExpression(*aggr.children[i]);
				ExtractCorrelatedExpressions(binder, *bound_expr);
			}
			if (aggr.filter) {
				bool success = aggregate_binder.BindCorrelatedColumns(aggr.filter);
				// if there is still an error after this, we could not successfully bind the aggregate
				if (!success) {
					throw BinderException(error);
				}
				auto &bound_expr = BoundExpression::GetExpression(*aggr.filter);
				ExtractCorrelatedExpressions(binder, *bound_expr);
			}
			if (aggr.order_bys && !aggr.order_bys->orders.empty()) {
				for (auto &order : aggr.order_bys->orders) {
					bool success = aggregate_binder.BindCorrelatedColumns(order.expression);
					if (!success) {
						throw BinderException(error);
					}
					auto &bound_expr = BoundExpression::GetExpression(*order.expression);
					ExtractCorrelatedExpressions(binder, *bound_expr);
				}
			}
		} else {
			// we didn't bind columns, try again in children
			return BindResult(error);
		}
	} else if (depth > 0 && !aggregate_binder.HasBoundColumns()) {
		return BindResult("Aggregate with only constant parameters has to be bound in the root subquery");
	}
	if (!filter_error.empty()) {
		return BindResult(filter_error);
	}

	if (aggr.filter) {
		auto &child = BoundExpression::GetExpression(*aggr.filter);
		bound_filter = BoundCastExpression::AddCastToType(context, std::move(child), LogicalType::BOOLEAN);
	}

	// all children bound successfully
	// extract the children and types
	vector<LogicalType> types;
	vector<LogicalType> arguments;
	vector<unique_ptr<Expression>> children;

	if (ordered_set_agg) {
		const bool order_sensitive = (aggr.function_name == "mode");
		for (auto &order : aggr.order_bys->orders) {
			auto &child = BoundExpression::GetExpression(*order.expression);
			types.push_back(child->return_type);
			arguments.push_back(child->return_type);
			if (order_sensitive) {
				children.push_back(child->Copy());
			} else {
				children.push_back(std::move(child));
			}
		}
		if (!order_sensitive) {
			aggr.order_bys->orders.clear();
		}
	}

	for (idx_t i = 0; i < aggr.children.size(); i++) {
		auto &child = BoundExpression::GetExpression(*aggr.children[i]);
		types.push_back(child->return_type);
		arguments.push_back(child->return_type);
		children.push_back(std::move(child));
	}

	// bind the aggregate
	FunctionBinder function_binder(context);
	idx_t best_function = function_binder.BindFunction(func.name, func.functions, types, error);
	if (best_function == DConstants::INVALID_INDEX) {
		throw BinderException(binder.FormatError(aggr, error));
	}
	// found a matching function!
	auto bound_function = func.functions.GetFunctionByOffset(best_function);

	// Bind any sort columns, unless the aggregate is order-insensitive
	unique_ptr<BoundOrderModifier> order_bys;
	if (!aggr.order_bys->orders.empty()) {
		order_bys = make_uniq<BoundOrderModifier>();
		auto &config = DBConfig::GetConfig(context);
		for (auto &order : aggr.order_bys->orders) {
			auto &order_expr = BoundExpression::GetExpression(*order.expression);
			const auto sense = config.ResolveOrder(order.type);
			const auto null_order = config.ResolveNullOrder(sense, order.null_order);
			order_bys->orders.emplace_back(sense, null_order, std::move(order_expr));
		}
	}

	auto aggregate =
	    function_binder.BindAggregateFunction(bound_function, std::move(children), std::move(bound_filter),
	                                          aggr.distinct ? AggregateType::DISTINCT : AggregateType::NON_DISTINCT);
	if (aggr.export_state) {
		aggregate = ExportAggregateFunction::Bind(std::move(aggregate));
	}
	aggregate->order_bys = std::move(order_bys);

	// check for all the aggregates if this aggregate already exists
	idx_t aggr_index;
	auto entry = node.aggregate_map.find(*aggregate);
	if (entry == node.aggregate_map.end()) {
		// new aggregate: insert into aggregate list
		aggr_index = node.aggregates.size();
		node.aggregate_map[*aggregate] = aggr_index;
		node.aggregates.push_back(std::move(aggregate));
	} else {
		// duplicate aggregate: simplify refer to this aggregate
		aggr_index = entry->second;
	}

	// now create a column reference referring to the aggregate
	auto colref = make_uniq<BoundColumnRefExpression>(
	    aggr.alias.empty() ? node.aggregates[aggr_index]->ToString() : aggr.alias,
	    node.aggregates[aggr_index]->return_type, ColumnBinding(node.aggregate_index, aggr_index), depth);
	// move the aggregate expression into the set of bound aggregates
	return BindResult(std::move(colref));
}
} // namespace duckdb








namespace duckdb {

BindResult ExpressionBinder::BindExpression(BetweenExpression &expr, idx_t depth) {
	// first try to bind the children of the case expression
	string error;
	BindChild(expr.input, depth, error);
	BindChild(expr.lower, depth, error);
	BindChild(expr.upper, depth, error);
	if (!error.empty()) {
		return BindResult(error);
	}
	// the children have been successfully resolved
	auto &input = BoundExpression::GetExpression(*expr.input);
	auto &lower = BoundExpression::GetExpression(*expr.lower);
	auto &upper = BoundExpression::GetExpression(*expr.upper);

	auto input_sql_type = input->return_type;
	auto lower_sql_type = lower->return_type;
	auto upper_sql_type = upper->return_type;

	// cast the input types to the same type
	// now obtain the result type of the input types
	auto input_type = BoundComparisonExpression::BindComparison(input_sql_type, lower_sql_type);
	input_type = BoundComparisonExpression::BindComparison(input_type, upper_sql_type);
	// add casts (if necessary)
	input = BoundCastExpression::AddCastToType(context, std::move(input), input_type);
	lower = BoundCastExpression::AddCastToType(context, std::move(lower), input_type);
	upper = BoundCastExpression::AddCastToType(context, std::move(upper), input_type);
	if (input_type.id() == LogicalTypeId::VARCHAR) {
		// handle collation
		auto collation = StringType::GetCollation(input_type);
		input = PushCollation(context, std::move(input), collation, false);
		lower = PushCollation(context, std::move(lower), collation, false);
		upper = PushCollation(context, std::move(upper), collation, false);
	}
	if (!input->HasSideEffects() && !input->HasParameter() && !input->HasSubquery()) {
		// the expression does not have side effects and can be copied: create two comparisons
		// the reason we do this is that individual comparisons are easier to handle in optimizers
		// if both comparisons remain they will be folded together again into a single BETWEEN in the optimizer
		auto left_compare = make_uniq<BoundComparisonExpression>(ExpressionType::COMPARE_GREATERTHANOREQUALTO,
		                                                         input->Copy(), std::move(lower));
		auto right_compare = make_uniq<BoundComparisonExpression>(ExpressionType::COMPARE_LESSTHANOREQUALTO,
		                                                          std::move(input), std::move(upper));
		return BindResult(make_uniq<BoundConjunctionExpression>(ExpressionType::CONJUNCTION_AND,
		                                                        std::move(left_compare), std::move(right_compare)));
	} else {
		// expression has side effects: we cannot duplicate it
		// create a bound_between directly
		return BindResult(
		    make_uniq<BoundBetweenExpression>(std::move(input), std::move(lower), std::move(upper), true, true));
	}
}

} // namespace duckdb





namespace duckdb {

BindResult ExpressionBinder::BindExpression(CaseExpression &expr, idx_t depth) {
	// first try to bind the children of the case expression
	string error;
	for (auto &check : expr.case_checks) {
		BindChild(check.when_expr, depth, error);
		BindChild(check.then_expr, depth, error);
	}
	BindChild(expr.else_expr, depth, error);
	if (!error.empty()) {
		return BindResult(error);
	}
	// the children have been successfully resolved
	// figure out the result type of the CASE expression
	auto &else_expr = BoundExpression::GetExpression(*expr.else_expr);
	auto return_type = else_expr->return_type;
	for (auto &check : expr.case_checks) {
		auto &then_expr = BoundExpression::GetExpression(*check.then_expr);
		return_type = LogicalType::MaxLogicalType(return_type, then_expr->return_type);
	}

	// bind all the individual components of the CASE statement
	auto result = make_uniq<BoundCaseExpression>(return_type);
	for (idx_t i = 0; i < expr.case_checks.size(); i++) {
		auto &check = expr.case_checks[i];
		auto &when_expr = BoundExpression::GetExpression(*check.when_expr);
		auto &then_expr = BoundExpression::GetExpression(*check.then_expr);
		BoundCaseCheck result_check;
		result_check.when_expr =
		    BoundCastExpression::AddCastToType(context, std::move(when_expr), LogicalType::BOOLEAN);
		result_check.then_expr = BoundCastExpression::AddCastToType(context, std::move(then_expr), return_type);
		result->case_checks.push_back(std::move(result_check));
	}
	result->else_expr = BoundCastExpression::AddCastToType(context, std::move(else_expr), return_type);
	return BindResult(std::move(result));
}
} // namespace duckdb






namespace duckdb {

BindResult ExpressionBinder::BindExpression(CastExpression &expr, idx_t depth) {
	// first try to bind the child of the cast expression
	string error = Bind(expr.child, depth);
	if (!error.empty()) {
		return BindResult(error);
	}
	// FIXME: We can also implement 'hello'::schema.custom_type; and pass by the schema down here.
	// Right now just considering its DEFAULT_SCHEMA always
	Binder::BindLogicalType(context, expr.cast_type);
	// the children have been successfully resolved
	auto &child = BoundExpression::GetExpression(*expr.child);
	if (expr.try_cast) {
		if (child->return_type == expr.cast_type) {
			// no cast required: type matches
			return BindResult(std::move(child));
		}
		child = BoundCastExpression::AddCastToType(context, std::move(child), expr.cast_type, true);
	} else {
		// otherwise add a cast to the target type
		child = BoundCastExpression::AddCastToType(context, std::move(child), expr.cast_type);
	}
	return BindResult(std::move(child));
}
} // namespace duckdb




namespace duckdb {

BindResult ExpressionBinder::BindExpression(CollateExpression &expr, idx_t depth) {
	// first try to bind the child of the cast expression
	string error = Bind(expr.child, depth);
	if (!error.empty()) {
		return BindResult(error);
	}
	auto &child = BoundExpression::GetExpression(*expr.child);
	if (child->HasParameter()) {
		throw ParameterNotResolvedException();
	}
	if (child->return_type.id() != LogicalTypeId::VARCHAR) {
		throw BinderException("collations are only supported for type varchar");
	}
	// Validate the collation, but don't use it
	PushCollation(context, child->Copy(), expr.collation, false);
	child->return_type = LogicalType::VARCHAR_COLLATION(expr.collation);
	return BindResult(std::move(child));
}

} // namespace duckdb

















namespace duckdb {

string GetSQLValueFunctionName(const string &column_name) {
	auto lcase = StringUtil::Lower(column_name);
	if (lcase == "current_catalog") {
		return "current_catalog";
	} else if (lcase == "current_date") {
		return "current_date";
	} else if (lcase == "current_schema") {
		return "current_schema";
	} else if (lcase == "current_role") {
		return "current_role";
	} else if (lcase == "current_time") {
		return "get_current_time";
	} else if (lcase == "current_timestamp") {
		return "get_current_timestamp";
	} else if (lcase == "current_user") {
		return "current_user";
	} else if (lcase == "localtime") {
		return "current_localtime";
	} else if (lcase == "localtimestamp") {
		return "current_localtimestamp";
	} else if (lcase == "session_user") {
		return "session_user";
	} else if (lcase == "user") {
		return "user";
	}
	return string();
}

unique_ptr<ParsedExpression> ExpressionBinder::GetSQLValueFunction(const string &column_name) {
	auto value_function = GetSQLValueFunctionName(column_name);
	if (value_function.empty()) {
		return nullptr;
	}

	vector<unique_ptr<ParsedExpression>> children;
	return make_uniq<FunctionExpression>(value_function, std::move(children));
}

unique_ptr<ParsedExpression> ExpressionBinder::QualifyColumnName(const string &column_name, string &error_message) {
	auto using_binding = binder.bind_context.GetUsingBinding(column_name);
	if (using_binding) {
		// we are referencing a USING column
		// check if we can refer to one of the base columns directly
		unique_ptr<Expression> expression;
		if (!using_binding->primary_binding.empty()) {
			// we can! just assign the table name and re-bind
			return binder.bind_context.CreateColumnReference(using_binding->primary_binding, column_name);
		} else {
			// // we cannot! we need to bind this as a coalesce between all the relevant columns
			auto coalesce = make_uniq<OperatorExpression>(ExpressionType::OPERATOR_COALESCE);
			coalesce->children.reserve(using_binding->bindings.size());
			for (auto &entry : using_binding->bindings) {
				coalesce->children.push_back(make_uniq<ColumnRefExpression>(column_name, entry));
			}
			return std::move(coalesce);
		}
	}

	// find a binding that contains this
	string table_name = binder.bind_context.GetMatchingBinding(column_name);

	// throw an error if a macro conflicts with a column name
	auto is_macro_column = false;
	if (binder.macro_binding != nullptr && binder.macro_binding->HasMatchingBinding(column_name)) {
		is_macro_column = true;
		if (!table_name.empty()) {
			throw BinderException("Conflicting column names for column " + column_name + "!");
		}
	}

	if (lambda_bindings) {
		for (idx_t i = 0; i < lambda_bindings->size(); i++) {
			if ((*lambda_bindings)[i].HasMatchingBinding(column_name)) {

				// throw an error if a lambda conflicts with a column name or a macro
				if (!table_name.empty() || is_macro_column) {
					throw BinderException("Conflicting column names for column " + column_name + "!");
				}

				D_ASSERT(!(*lambda_bindings)[i].alias.empty());
				return make_uniq<ColumnRefExpression>(column_name, (*lambda_bindings)[i].alias);
			}
		}
	}

	if (is_macro_column) {
		D_ASSERT(!binder.macro_binding->alias.empty());
		return make_uniq<ColumnRefExpression>(column_name, binder.macro_binding->alias);
	}
	// see if it's a column
	if (table_name.empty()) {
		// column was not found - check if it is a SQL value function
		auto value_function = GetSQLValueFunction(column_name);
		if (value_function) {
			return value_function;
		}
		// it's not, find candidates and error
		auto similar_bindings = binder.bind_context.GetSimilarBindings(column_name);
		string candidate_str = StringUtil::CandidatesMessage(similar_bindings, "Candidate bindings");
		error_message =
		    StringUtil::Format("Referenced column \"%s\" not found in FROM clause!%s", column_name, candidate_str);
		return nullptr;
	}
	return binder.bind_context.CreateColumnReference(table_name, column_name);
}

void ExpressionBinder::QualifyColumnNames(unique_ptr<ParsedExpression> &expr) {
	switch (expr->type) {
	case ExpressionType::COLUMN_REF: {
		auto &colref = expr->Cast<ColumnRefExpression>();
		string error_message;
		auto new_expr = QualifyColumnName(colref, error_message);
		if (new_expr) {
			if (!expr->alias.empty()) {
				new_expr->alias = expr->alias;
			}
			new_expr->query_location = colref.query_location;
			expr = std::move(new_expr);
		}
		break;
	}
	case ExpressionType::POSITIONAL_REFERENCE: {
		auto &ref = expr->Cast<PositionalReferenceExpression>();
		if (ref.alias.empty()) {
			string table_name, column_name;
			auto error = binder.bind_context.BindColumn(ref, table_name, column_name);
			if (error.empty()) {
				ref.alias = column_name;
			}
		}
		break;
	}
	default:
		break;
	}
	ParsedExpressionIterator::EnumerateChildren(
	    *expr, [&](unique_ptr<ParsedExpression> &child) { QualifyColumnNames(child); });
}

void ExpressionBinder::QualifyColumnNames(Binder &binder, unique_ptr<ParsedExpression> &expr) {
	WhereBinder where_binder(binder, binder.context);
	where_binder.QualifyColumnNames(expr);
}

unique_ptr<ParsedExpression> ExpressionBinder::CreateStructExtract(unique_ptr<ParsedExpression> base,
                                                                   string field_name) {

	// we need to transform the struct extract if it is inside a lambda expression
	// because we cannot bind to an existing table, so we remove the dummy table also
	if (lambda_bindings && base->type == ExpressionType::COLUMN_REF) {
		auto &lambda_column_ref = base->Cast<ColumnRefExpression>();
		D_ASSERT(!lambda_column_ref.column_names.empty());

		if (lambda_column_ref.column_names[0].find(DummyBinding::DUMMY_NAME) != string::npos) {
			D_ASSERT(lambda_column_ref.column_names.size() == 2);
			auto lambda_param_name = lambda_column_ref.column_names.back();
			lambda_column_ref.column_names.clear();
			lambda_column_ref.column_names.push_back(lambda_param_name);
		}
	}

	vector<unique_ptr<ParsedExpression>> children;
	children.push_back(std::move(base));
	children.push_back(make_uniq_base<ParsedExpression, ConstantExpression>(Value(std::move(field_name))));
	auto extract_fun = make_uniq<OperatorExpression>(ExpressionType::STRUCT_EXTRACT, std::move(children));
	return std::move(extract_fun);
}

unique_ptr<ParsedExpression> ExpressionBinder::CreateStructPack(ColumnRefExpression &colref) {
	D_ASSERT(colref.column_names.size() <= 3);
	string error_message;
	auto &table_name = colref.column_names.back();
	auto binding = binder.bind_context.GetBinding(table_name, error_message);
	if (!binding) {
		return nullptr;
	}
	if (colref.column_names.size() >= 2) {
		// "schema_name.table_name"
		auto catalog_entry = binding->GetStandardEntry();
		if (!catalog_entry) {
			return nullptr;
		}
		if (catalog_entry->name != table_name) {
			return nullptr;
		}
		if (colref.column_names.size() == 2) {
			auto &qualifier = colref.column_names[0];
			if (catalog_entry->catalog.GetName() != qualifier && catalog_entry->schema.name != qualifier) {
				return nullptr;
			}
		} else if (colref.column_names.size() == 3) {
			auto &catalog_name = colref.column_names[0];
			auto &schema_name = colref.column_names[1];
			if (catalog_entry->catalog.GetName() != catalog_name || catalog_entry->schema.name != schema_name) {
				return nullptr;
			}
		} else {
			throw InternalException("Expected 2 or 3 column names for CreateStructPack");
		}
	}
	// We found the table, now create the struct_pack expression
	vector<unique_ptr<ParsedExpression>> child_expressions;
	child_expressions.reserve(binding->names.size());
	for (const auto &column_name : binding->names) {
		child_expressions.push_back(make_uniq<ColumnRefExpression>(column_name, table_name));
	}
	return make_uniq<FunctionExpression>("struct_pack", std::move(child_expressions));
}

unique_ptr<ParsedExpression> ExpressionBinder::QualifyColumnName(ColumnRefExpression &colref, string &error_message) {
	idx_t column_parts = colref.column_names.size();
	// column names can have an arbitrary amount of dots
	// here is how the resolution works:
	if (column_parts == 1) {
		// no dots (i.e. "part1")
		// -> part1 refers to a column
		// check if we can qualify the column name with the table name
		auto qualified_colref = QualifyColumnName(colref.GetColumnName(), error_message);
		if (qualified_colref) {
			// we could: return it
			return qualified_colref;
		}
		// we could not! Try creating an implicit struct_pack
		return CreateStructPack(colref);
	} else if (column_parts == 2) {
		// one dot (i.e. "part1.part2")
		// EITHER:
		// -> part1 is a table, part2 is a column
		// -> part1 is a column, part2 is a property of that column (i.e. struct_extract)

		// first check if part1 is a table, and part2 is a standard column
		if (binder.HasMatchingBinding(colref.column_names[0], colref.column_names[1], error_message)) {
			// it is! return the colref directly
			return binder.bind_context.CreateColumnReference(colref.column_names[0], colref.column_names[1]);
		} else {
			// otherwise check if we can turn this into a struct extract
			auto new_colref = make_uniq<ColumnRefExpression>(colref.column_names[0]);
			string other_error;
			auto qualified_colref = QualifyColumnName(colref.column_names[0], other_error);
			if (qualified_colref) {
				// we could: create a struct extract
				return CreateStructExtract(std::move(qualified_colref), colref.column_names[1]);
			}
			// we could not! Try creating an implicit struct_pack
			return CreateStructPack(colref);
		}
	} else {
		// two or more dots (i.e. "part1.part2.part3.part4...")
		// -> part1 is a catalog, part2 is a schema, part3 is a table, part4 is a column name, part 5 and beyond are
		// struct fields
		// -> part1 is a catalog, part2 is a table, part3 is a column name, part4 and beyond are struct fields
		// -> part1 is a schema, part2 is a table, part3 is a column name, part4 and beyond are struct fields
		// -> part1 is a table, part2 is a column name, part3 and beyond are struct fields
		// -> part1 is a column, part2 and beyond are struct fields

		// we always prefer the most top-level view
		// i.e. in case of multiple resolution options, we resolve in order:
		// -> 1. resolve "part1" as a catalog
		// -> 2. resolve "part1" as a schema
		// -> 3. resolve "part1" as a table
		// -> 4. resolve "part1" as a column

		unique_ptr<ParsedExpression> result_expr;
		idx_t struct_extract_start;
		// first check if part1 is a catalog
		if (colref.column_names.size() > 3 &&
		    binder.HasMatchingBinding(colref.column_names[0], colref.column_names[1], colref.column_names[2],
		                              colref.column_names[3], error_message)) {
			// part1 is a catalog - the column reference is "catalog.schema.table.column"
			result_expr = binder.bind_context.CreateColumnReference(colref.column_names[0], colref.column_names[1],
			                                                        colref.column_names[2], colref.column_names[3]);
			struct_extract_start = 4;
		} else if (binder.HasMatchingBinding(colref.column_names[0], INVALID_SCHEMA, colref.column_names[1],
		                                     colref.column_names[2], error_message)) {
			// part1 is a catalog - the column reference is "catalog.table.column"
			result_expr = binder.bind_context.CreateColumnReference(colref.column_names[0], INVALID_SCHEMA,
			                                                        colref.column_names[1], colref.column_names[2]);
			struct_extract_start = 3;
		} else if (binder.HasMatchingBinding(colref.column_names[0], colref.column_names[1], colref.column_names[2],
		                                     error_message)) {
			// part1 is a schema - the column reference is "schema.table.column"
			// any additional fields are turned into struct_extract calls
			result_expr = binder.bind_context.CreateColumnReference(colref.column_names[0], colref.column_names[1],
			                                                        colref.column_names[2]);
			struct_extract_start = 3;
		} else if (binder.HasMatchingBinding(colref.column_names[0], colref.column_names[1], error_message)) {
			// part1 is a table
			// the column reference is "table.column"
			// any additional fields are turned into struct_extract calls
			result_expr = binder.bind_context.CreateColumnReference(colref.column_names[0], colref.column_names[1]);
			struct_extract_start = 2;
		} else {
			// part1 could be a column
			string col_error;
			result_expr = QualifyColumnName(colref.column_names[0], col_error);
			if (!result_expr) {
				// it is not! Try creating an implicit struct_pack
				return CreateStructPack(colref);
			}
			// it is! add the struct extract calls
			struct_extract_start = 1;
		}
		for (idx_t i = struct_extract_start; i < colref.column_names.size(); i++) {
			result_expr = CreateStructExtract(std::move(result_expr), colref.column_names[i]);
		}
		return result_expr;
	}
}

BindResult ExpressionBinder::BindExpression(ColumnRefExpression &colref_p, idx_t depth) {
	if (binder.GetBindingMode() == BindingMode::EXTRACT_NAMES) {
		return BindResult(make_uniq<BoundConstantExpression>(Value(LogicalType::SQLNULL)));
	}
	string error_message;
	auto expr = QualifyColumnName(colref_p, error_message);
	if (!expr) {
		return BindResult(binder.FormatError(colref_p, error_message));
	}
	expr->query_location = colref_p.query_location;

	// a generated column returns a generated expression, a struct on a column returns a struct extract
	if (expr->type != ExpressionType::COLUMN_REF) {
		auto alias = expr->alias;
		auto result = BindExpression(expr, depth);
		if (result.expression) {
			result.expression->alias = std::move(alias);
		}
		return result;
	}

	auto &colref = expr->Cast<ColumnRefExpression>();
	D_ASSERT(colref.IsQualified());
	auto &table_name = colref.GetTableName();

	// individual column reference
	// resolve to either a base table or a subquery expression
	// if it was a macro parameter, let macro_binding bind it to the argument
	// if it was a lambda parameter, let lambda_bindings bind it to the argument

	BindResult result;

	auto found_lambda_binding = false;
	if (lambda_bindings) {
		for (idx_t i = 0; i < lambda_bindings->size(); i++) {
			if (table_name == (*lambda_bindings)[i].alias) {
				result = (*lambda_bindings)[i].Bind(colref, i, depth);
				found_lambda_binding = true;
				break;
			}
		}
	}

	if (!found_lambda_binding) {
		if (binder.macro_binding && table_name == binder.macro_binding->alias) {
			result = binder.macro_binding->Bind(colref, depth);
		} else {
			result = binder.bind_context.BindColumn(colref, depth);
		}
	}

	if (!result.HasError()) {
		BoundColumnReferenceInfo ref;
		ref.name = colref.column_names.back();
		ref.query_location = colref.query_location;
		bound_columns.push_back(std::move(ref));
	} else {
		result.error = binder.FormatError(colref_p, result.error);
	}
	return result;
}

bool ExpressionBinder::QualifyColumnAlias(const ColumnRefExpression &colref) {
	// Only BaseSelectBinder will have a valid col alias map,
	// otherwise just return false
	return false;
}

} // namespace duckdb


















namespace duckdb {

unique_ptr<Expression> ExpressionBinder::PushCollation(ClientContext &context, unique_ptr<Expression> source,
                                                       const string &collation_p, bool equality_only) {
	// replace default collation with system collation
	string collation;
	if (collation_p.empty()) {
		collation = DBConfig::GetConfig(context).options.collation;
	} else {
		collation = collation_p;
	}
	collation = StringUtil::Lower(collation);
	// bind the collation
	if (collation.empty() || collation == "binary" || collation == "c" || collation == "posix") {
		// binary collation: just skip
		return source;
	}
	auto &catalog = Catalog::GetSystemCatalog(context);
	auto splits = StringUtil::Split(StringUtil::Lower(collation), ".");
	vector<reference<CollateCatalogEntry>> entries;
	for (auto &collation_argument : splits) {
		auto &collation_entry = catalog.GetEntry<CollateCatalogEntry>(context, DEFAULT_SCHEMA, collation_argument);
		if (collation_entry.combinable) {
			entries.insert(entries.begin(), collation_entry);
		} else {
			if (!entries.empty() && !entries.back().get().combinable) {
				throw BinderException("Cannot combine collation types \"%s\" and \"%s\"", entries.back().get().name,
				                      collation_entry.name);
			}
			entries.push_back(collation_entry);
		}
	}
	for (auto &entry : entries) {
		auto &collation_entry = entry.get();
		if (equality_only && collation_entry.not_required_for_equality) {
			continue;
		}
		vector<unique_ptr<Expression>> children;
		children.push_back(std::move(source));

		FunctionBinder function_binder(context);
		auto function = function_binder.BindScalarFunction(collation_entry.function, std::move(children));
		source = std::move(function);
	}
	return source;
}

void ExpressionBinder::TestCollation(ClientContext &context, const string &collation) {
	PushCollation(context, make_uniq<BoundConstantExpression>(Value("")), collation);
}

LogicalType BoundComparisonExpression::BindComparison(LogicalType left_type, LogicalType right_type) {
	auto result_type = LogicalType::MaxLogicalType(left_type, right_type);
	switch (result_type.id()) {
	case LogicalTypeId::DECIMAL: {
		// result is a decimal: we need the maximum width and the maximum scale over width
		vector<LogicalType> argument_types = {left_type, right_type};
		uint8_t max_width = 0, max_scale = 0, max_width_over_scale = 0;
		for (idx_t i = 0; i < argument_types.size(); i++) {
			uint8_t width, scale;
			auto can_convert = argument_types[i].GetDecimalProperties(width, scale);
			if (!can_convert) {
				return result_type;
			}
			max_width = MaxValue<uint8_t>(width, max_width);
			max_scale = MaxValue<uint8_t>(scale, max_scale);
			max_width_over_scale = MaxValue<uint8_t>(width - scale, max_width_over_scale);
		}
		max_width = MaxValue<uint8_t>(max_scale + max_width_over_scale, max_width);
		if (max_width > Decimal::MAX_WIDTH_DECIMAL) {
			// target width does not fit in decimal: truncate the scale (if possible) to try and make it fit
			max_width = Decimal::MAX_WIDTH_DECIMAL;
		}
		return LogicalType::DECIMAL(max_width, max_scale);
	}
	case LogicalTypeId::VARCHAR:
		// for comparison with strings, we prefer to bind to the numeric types
		if (left_type.IsNumeric() || left_type.id() == LogicalTypeId::BOOLEAN) {
			return left_type;
		} else if (right_type.IsNumeric() || right_type.id() == LogicalTypeId::BOOLEAN) {
			return right_type;
		} else {
			// else: check if collations are compatible
			auto left_collation = StringType::GetCollation(left_type);
			auto right_collation = StringType::GetCollation(right_type);
			if (!left_collation.empty() && !right_collation.empty() && left_collation != right_collation) {
				throw BinderException("Cannot combine types with different collation!");
			}
		}
		return result_type;
	default:
		return result_type;
	}
}

BindResult ExpressionBinder::BindExpression(ComparisonExpression &expr, idx_t depth) {
	// first try to bind the children of the case expression
	string error;
	BindChild(expr.left, depth, error);
	BindChild(expr.right, depth, error);
	if (!error.empty()) {
		return BindResult(error);
	}
	// the children have been successfully resolved
	auto &left = BoundExpression::GetExpression(*expr.left);
	auto &right = BoundExpression::GetExpression(*expr.right);
	auto left_sql_type = left->return_type;
	auto right_sql_type = right->return_type;
	// cast the input types to the same type
	// now obtain the result type of the input types
	auto input_type = BoundComparisonExpression::BindComparison(left_sql_type, right_sql_type);
	// add casts (if necessary)
	left = BoundCastExpression::AddCastToType(context, std::move(left), input_type,
	                                          input_type.id() == LogicalTypeId::ENUM);
	right = BoundCastExpression::AddCastToType(context, std::move(right), input_type,
	                                           input_type.id() == LogicalTypeId::ENUM);

	if (input_type.id() == LogicalTypeId::VARCHAR) {
		// handle collation
		auto collation = StringType::GetCollation(input_type);
		left = PushCollation(context, std::move(left), collation, expr.type == ExpressionType::COMPARE_EQUAL);
		right = PushCollation(context, std::move(right), collation, expr.type == ExpressionType::COMPARE_EQUAL);
	}
	// now create the bound comparison expression
	return BindResult(make_uniq<BoundComparisonExpression>(expr.type, std::move(left), std::move(right)));
}

} // namespace duckdb





namespace duckdb {

BindResult ExpressionBinder::BindExpression(ConjunctionExpression &expr, idx_t depth) {
	// first try to bind the children of the case expression
	string error;
	for (idx_t i = 0; i < expr.children.size(); i++) {
		BindChild(expr.children[i], depth, error);
	}
	if (!error.empty()) {
		return BindResult(error);
	}
	// the children have been successfully resolved
	// cast the input types to boolean (if necessary)
	// and construct the bound conjunction expression
	auto result = make_uniq<BoundConjunctionExpression>(expr.type);
	for (auto &child_expr : expr.children) {
		auto &child = BoundExpression::GetExpression(*child_expr);
		result->children.push_back(BoundCastExpression::AddCastToType(context, std::move(child), LogicalType::BOOLEAN));
	}
	// now create the bound conjunction expression
	return BindResult(std::move(result));
}

} // namespace duckdb




namespace duckdb {

BindResult ExpressionBinder::BindExpression(ConstantExpression &expr, idx_t depth) {
	return BindResult(make_uniq<BoundConstantExpression>(expr.value));
}

} // namespace duckdb
















namespace duckdb {

BindResult ExpressionBinder::BindExpression(FunctionExpression &function, idx_t depth,
                                            unique_ptr<ParsedExpression> &expr_ptr) {
	// lookup the function in the catalog
	QueryErrorContext error_context(binder.root_statement, function.query_location);
	auto func = Catalog::GetEntry(context, CatalogType::SCALAR_FUNCTION_ENTRY, function.catalog, function.schema,
	                              function.function_name, OnEntryNotFound::RETURN_NULL, error_context);
	if (!func) {
		// function was not found - check if we this is a table function
		auto table_func =
		    Catalog::GetEntry(context, CatalogType::TABLE_FUNCTION_ENTRY, function.catalog, function.schema,
		                      function.function_name, OnEntryNotFound::RETURN_NULL, error_context);
		if (table_func) {
			throw BinderException(binder.FormatError(
			    function,
			    StringUtil::Format("Function \"%s\" is a table function but it was used as a scalar function. This "
			                       "function has to be called in a FROM clause (similar to a table).",
			                       function.function_name)));
		}
		// not a table function - check if the schema is set
		if (!function.schema.empty()) {
			// the schema is set - check if we can turn this the schema into a column ref
			string error;
			unique_ptr<ColumnRefExpression> colref;
			if (function.catalog.empty()) {
				colref = make_uniq<ColumnRefExpression>(function.schema);
			} else {
				colref = make_uniq<ColumnRefExpression>(function.schema, function.catalog);
			}
			auto new_colref = QualifyColumnName(*colref, error);
			bool is_col = error.empty() ? true : false;
			bool is_col_alias = QualifyColumnAlias(*colref);

			if (is_col || is_col_alias) {
				// we can! transform this into a function call on the column
				// i.e. "x.lower()" becomes "lower(x)"
				function.children.insert(function.children.begin(), std::move(colref));
				function.catalog = INVALID_CATALOG;
				function.schema = INVALID_SCHEMA;
			}
		}
		// rebind the function
		func = Catalog::GetEntry(context, CatalogType::SCALAR_FUNCTION_ENTRY, function.catalog, function.schema,
		                         function.function_name, OnEntryNotFound::THROW_EXCEPTION, error_context);
	}

	if (func->type != CatalogType::AGGREGATE_FUNCTION_ENTRY &&
	    (function.distinct || function.filter || !function.order_bys->orders.empty())) {
		throw InvalidInputException("Function \"%s\" is a %s. \"DISTINCT\", \"FILTER\", and \"ORDER BY\" are only "
		                            "applicable to aggregate functions.",
		                            function.function_name, CatalogTypeToString(func->type));
	}

	switch (func->type) {
	case CatalogType::SCALAR_FUNCTION_ENTRY:
		// scalar function

		// check for lambda parameters, ignore ->> operator (JSON extension)
		if (function.function_name != "->>") {
			for (auto &child : function.children) {
				if (child->expression_class == ExpressionClass::LAMBDA) {
					return BindLambdaFunction(function, func->Cast<ScalarFunctionCatalogEntry>(), depth);
				}
			}
		}

		// other scalar function
		return BindFunction(function, func->Cast<ScalarFunctionCatalogEntry>(), depth);

	case CatalogType::MACRO_ENTRY:
		// macro function
		return BindMacro(function, func->Cast<ScalarMacroCatalogEntry>(), depth, expr_ptr);
	default:
		// aggregate function
		return BindAggregate(function, func->Cast<AggregateFunctionCatalogEntry>(), depth);
	}
}

BindResult ExpressionBinder::BindFunction(FunctionExpression &function, ScalarFunctionCatalogEntry &func, idx_t depth) {

	// bind the children of the function expression
	string error;

	// bind of each child
	for (idx_t i = 0; i < function.children.size(); i++) {
		BindChild(function.children[i], depth, error);
	}

	if (!error.empty()) {
		return BindResult(error);
	}
	if (binder.GetBindingMode() == BindingMode::EXTRACT_NAMES) {
		return BindResult(make_uniq<BoundConstantExpression>(Value(LogicalType::SQLNULL)));
	}

	// all children bound successfully
	// extract the children and types
	vector<unique_ptr<Expression>> children;
	for (idx_t i = 0; i < function.children.size(); i++) {
		auto &child = BoundExpression::GetExpression(*function.children[i]);
		children.push_back(std::move(child));
	}

	FunctionBinder function_binder(context);
	unique_ptr<Expression> result =
	    function_binder.BindScalarFunction(func, std::move(children), error, function.is_operator, &binder);
	if (!result) {
		throw BinderException(binder.FormatError(function, error));
	}
	return BindResult(std::move(result));
}

BindResult ExpressionBinder::BindLambdaFunction(FunctionExpression &function, ScalarFunctionCatalogEntry &func,
                                                idx_t depth) {

	// bind the children of the function expression
	string error;

	if (function.children.size() != 2) {
		throw BinderException("Invalid function arguments!");
	}
	D_ASSERT(function.children[1]->GetExpressionClass() == ExpressionClass::LAMBDA);

	// bind the list parameter
	BindChild(function.children[0], depth, error);
	if (!error.empty()) {
		return BindResult(error);
	}

	// get the logical type of the children of the list
	auto &list_child = BoundExpression::GetExpression(*function.children[0]);
	if (list_child->return_type.id() != LogicalTypeId::LIST && list_child->return_type.id() != LogicalTypeId::SQLNULL &&
	    list_child->return_type.id() != LogicalTypeId::UNKNOWN) {
		throw BinderException(" Invalid LIST argument to " + function.function_name + "!");
	}

	LogicalType list_child_type = list_child->return_type.id();
	if (list_child->return_type.id() != LogicalTypeId::SQLNULL &&
	    list_child->return_type.id() != LogicalTypeId::UNKNOWN) {
		list_child_type = ListType::GetChildType(list_child->return_type);
	}

	// bind the lambda parameter
	auto &lambda_expr = function.children[1]->Cast<LambdaExpression>();
	BindResult bind_lambda_result = BindExpression(lambda_expr, depth, true, list_child_type);

	if (bind_lambda_result.HasError()) {
		error = bind_lambda_result.error;
	} else {
		// successfully bound: replace the node with a BoundExpression
		auto alias = function.children[1]->alias;
		bind_lambda_result.expression->alias = alias;
		if (!alias.empty()) {
			bind_lambda_result.expression->alias = alias;
		}
		function.children[1] = make_uniq<BoundExpression>(std::move(bind_lambda_result.expression));
	}

	if (!error.empty()) {
		return BindResult(error);
	}
	if (binder.GetBindingMode() == BindingMode::EXTRACT_NAMES) {
		return BindResult(make_uniq<BoundConstantExpression>(Value(LogicalType::SQLNULL)));
	}

	// all children bound successfully
	// extract the children and types
	vector<unique_ptr<Expression>> children;
	for (idx_t i = 0; i < function.children.size(); i++) {
		auto &child = BoundExpression::GetExpression(*function.children[i]);
		children.push_back(std::move(child));
	}

	// capture the (lambda) columns
	auto &bound_lambda_expr = (BoundLambdaExpression &)*children.back();
	CaptureLambdaColumns(bound_lambda_expr.captures, list_child_type, bound_lambda_expr.lambda_expr);

	FunctionBinder function_binder(context);
	unique_ptr<Expression> result =
	    function_binder.BindScalarFunction(func, std::move(children), error, function.is_operator, &binder);
	if (!result) {
		throw BinderException(binder.FormatError(function, error));
	}

	auto &bound_function_expr = result->Cast<BoundFunctionExpression>();
	D_ASSERT(bound_function_expr.children.size() == 2);

	// remove the lambda expression from the children
	auto lambda = std::move(bound_function_expr.children.back());
	bound_function_expr.children.pop_back();
	auto &bound_lambda = lambda->Cast<BoundLambdaExpression>();

	// push back (in reverse order) any nested lambda parameters so that we can later use them in the lambda expression
	// (rhs)
	if (lambda_bindings) {
		for (idx_t i = lambda_bindings->size(); i > 0; i--) {

			idx_t lambda_index = lambda_bindings->size() - i + 1;
			auto &binding = (*lambda_bindings)[i - 1];

			D_ASSERT(binding.names.size() == 1);
			D_ASSERT(binding.types.size() == 1);

			bound_function_expr.function.arguments.push_back(binding.types[0]);
			auto bound_lambda_param =
			    make_uniq<BoundReferenceExpression>(binding.names[0], binding.types[0], lambda_index);
			bound_function_expr.children.push_back(std::move(bound_lambda_param));
		}
	}

	// push back the captures into the children vector and the correct return types into the bound_function arguments
	for (auto &capture : bound_lambda.captures) {
		bound_function_expr.function.arguments.push_back(capture->return_type);
		bound_function_expr.children.push_back(std::move(capture));
	}

	return BindResult(std::move(result));
}

BindResult ExpressionBinder::BindAggregate(FunctionExpression &expr, AggregateFunctionCatalogEntry &function,
                                           idx_t depth) {
	return BindResult(binder.FormatError(expr, UnsupportedAggregateMessage()));
}

BindResult ExpressionBinder::BindUnnest(FunctionExpression &expr, idx_t depth, bool root_expression) {
	return BindResult(binder.FormatError(expr, UnsupportedUnnestMessage()));
}

string ExpressionBinder::UnsupportedAggregateMessage() {
	return "Aggregate functions are not supported here";
}

string ExpressionBinder::UnsupportedUnnestMessage() {
	return "UNNEST not supported here";
}

} // namespace duckdb












namespace duckdb {

BindResult ExpressionBinder::BindExpression(LambdaExpression &expr, idx_t depth, const bool is_lambda,
                                            const LogicalType &list_child_type) {

	if (!is_lambda) {
		// this is for binding JSON
		auto lhs_expr = expr.lhs->Copy();
		OperatorExpression arrow_expr(ExpressionType::ARROW, std::move(lhs_expr), expr.expr->Copy());
		return BindExpression(arrow_expr, depth);
	}

	// binding the lambda expression
	D_ASSERT(expr.lhs);
	if (expr.lhs->expression_class != ExpressionClass::FUNCTION &&
	    expr.lhs->expression_class != ExpressionClass::COLUMN_REF) {
		throw BinderException(
		    "Invalid parameter list! Parameters must be comma-separated column names, e.g. x or (x, y).");
	}

	// move the lambda parameters to the params vector
	if (expr.lhs->expression_class == ExpressionClass::COLUMN_REF) {
		expr.params.push_back(std::move(expr.lhs));
	} else {
		auto &func_expr = expr.lhs->Cast<FunctionExpression>();
		for (idx_t i = 0; i < func_expr.children.size(); i++) {
			expr.params.push_back(std::move(func_expr.children[i]));
		}
	}
	D_ASSERT(!expr.params.empty());

	// create dummy columns for the lambda parameters (lhs)
	vector<LogicalType> column_types;
	vector<string> column_names;
	vector<string> params_strings;

	// positional parameters as column references
	for (idx_t i = 0; i < expr.params.size(); i++) {
		if (expr.params[i]->GetExpressionClass() != ExpressionClass::COLUMN_REF) {
			throw BinderException("Parameter must be a column name.");
		}

		auto column_ref = expr.params[i]->Cast<ColumnRefExpression>();
		if (column_ref.IsQualified()) {
			throw BinderException("Invalid parameter name '%s': must be unqualified", column_ref.ToString());
		}

		column_types.emplace_back(list_child_type);
		column_names.push_back(column_ref.GetColumnName());
		params_strings.push_back(expr.params[i]->ToString());
	}

	// base table alias
	auto params_alias = StringUtil::Join(params_strings, ", ");
	if (params_strings.size() > 1) {
		params_alias = "(" + params_alias + ")";
	}

	// create a lambda binding and push it to the lambda bindings vector
	vector<DummyBinding> local_bindings;
	if (!lambda_bindings) {
		lambda_bindings = &local_bindings;
	}
	DummyBinding new_lambda_binding(column_types, column_names, params_alias);
	lambda_bindings->push_back(new_lambda_binding);

	// bind the parameter expressions
	for (idx_t i = 0; i < expr.params.size(); i++) {
		auto result = BindExpression(expr.params[i], depth, false);
		if (result.HasError()) {
			throw InternalException("Error during lambda binding: %s", result.error);
		}
	}

	auto result = BindExpression(expr.expr, depth, false);
	lambda_bindings->pop_back();

	// successfully bound a subtree of nested lambdas, set this to nullptr in case other parts of the
	// query also contain lambdas
	if (lambda_bindings->empty()) {
		lambda_bindings = nullptr;
	}

	if (result.HasError()) {
		throw BinderException(result.error);
	}

	return BindResult(make_uniq<BoundLambdaExpression>(ExpressionType::LAMBDA, LogicalType::LAMBDA,
	                                                   std::move(result.expression), params_strings.size()));
}

void ExpressionBinder::TransformCapturedLambdaColumn(unique_ptr<Expression> &original,
                                                     unique_ptr<Expression> &replacement,
                                                     vector<unique_ptr<Expression>> &captures,
                                                     LogicalType &list_child_type) {

	// check if the original expression is a lambda parameter
	if (original->expression_class == ExpressionClass::BOUND_LAMBDA_REF) {

		// determine if this is the lambda parameter
		auto &bound_lambda_ref = (BoundLambdaRefExpression &)*original;
		auto alias = bound_lambda_ref.alias;

		if (lambda_bindings && bound_lambda_ref.lambda_index != lambda_bindings->size()) {

			D_ASSERT(bound_lambda_ref.lambda_index < lambda_bindings->size());
			auto &lambda_binding = (*lambda_bindings)[bound_lambda_ref.lambda_index];

			D_ASSERT(lambda_binding.names.size() == 1);
			D_ASSERT(lambda_binding.types.size() == 1);
			// refers to a lambda parameter outside of the current lambda function
			replacement =
			    make_uniq<BoundReferenceExpression>(lambda_binding.names[0], lambda_binding.types[0],
			                                        lambda_bindings->size() - bound_lambda_ref.lambda_index + 1);

		} else {
			// refers to current lambda parameter
			replacement = make_uniq<BoundReferenceExpression>(alias, list_child_type, 0);
		}

	} else {
		// always at least the current lambda parameter
		idx_t index_offset = 1;
		if (lambda_bindings) {
			index_offset += lambda_bindings->size();
		}

		// this is not a lambda parameter, so we need to create a new argument for the arguments vector
		replacement = make_uniq<BoundReferenceExpression>(original->alias, original->return_type,
		                                                  captures.size() + index_offset + 1);
		captures.push_back(std::move(original));
	}
}

void ExpressionBinder::CaptureLambdaColumns(vector<unique_ptr<Expression>> &captures, LogicalType &list_child_type,
                                            unique_ptr<Expression> &expr) {

	if (expr->expression_class == ExpressionClass::BOUND_SUBQUERY) {
		throw InvalidInputException("Subqueries are not supported in lambda expressions!");
	}

	// these expression classes do not have children, transform them
	if (expr->expression_class == ExpressionClass::BOUND_CONSTANT ||
	    expr->expression_class == ExpressionClass::BOUND_COLUMN_REF ||
	    expr->expression_class == ExpressionClass::BOUND_PARAMETER ||
	    expr->expression_class == ExpressionClass::BOUND_LAMBDA_REF) {

		// move the expr because we are going to replace it
		auto original = std::move(expr);
		unique_ptr<Expression> replacement;

		TransformCapturedLambdaColumn(original, replacement, captures, list_child_type);

		// replace the expression
		expr = std::move(replacement);

	} else {
		// recursively enumerate the children of the expression
		ExpressionIterator::EnumerateChildren(
		    *expr, [&](unique_ptr<Expression> &child) { CaptureLambdaColumns(captures, list_child_type, child); });
	}

	expr->Verify();
}

} // namespace duckdb









namespace duckdb {

void ExpressionBinder::ReplaceMacroParametersRecursive(unique_ptr<ParsedExpression> &expr) {
	switch (expr->GetExpressionClass()) {
	case ExpressionClass::COLUMN_REF: {
		// if expr is a parameter, replace it with its argument
		auto &colref = expr->Cast<ColumnRefExpression>();
		bool bind_macro_parameter = false;
		if (colref.IsQualified()) {
			bind_macro_parameter = false;
			if (colref.GetTableName().find(DummyBinding::DUMMY_NAME) != string::npos) {
				bind_macro_parameter = true;
			}
		} else {
			bind_macro_parameter = macro_binding->HasMatchingBinding(colref.GetColumnName());
		}
		if (bind_macro_parameter) {
			D_ASSERT(macro_binding->HasMatchingBinding(colref.GetColumnName()));
			expr = macro_binding->ParamToArg(colref);
		}
		return;
	}
	case ExpressionClass::SUBQUERY: {
		// replacing parameters within a subquery is slightly different
		auto &sq = (expr->Cast<SubqueryExpression>()).subquery;
		ParsedExpressionIterator::EnumerateQueryNodeChildren(
		    *sq->node, [&](unique_ptr<ParsedExpression> &child) { ReplaceMacroParametersRecursive(child); });
		break;
	}
	default: // fall through
		break;
	}
	// unfold child expressions
	ParsedExpressionIterator::EnumerateChildren(
	    *expr, [&](unique_ptr<ParsedExpression> &child) { ReplaceMacroParametersRecursive(child); });
}

BindResult ExpressionBinder::BindMacro(FunctionExpression &function, ScalarMacroCatalogEntry &macro_func, idx_t depth,
                                       unique_ptr<ParsedExpression> &expr) {
	// recast function so we can access the scalar member function->expression
	auto &macro_def = macro_func.function->Cast<ScalarMacroFunction>();

	// validate the arguments and separate positional and default arguments
	vector<unique_ptr<ParsedExpression>> positionals;
	unordered_map<string, unique_ptr<ParsedExpression>> defaults;

	string error =
	    MacroFunction::ValidateArguments(*macro_func.function, macro_func.name, function, positionals, defaults);
	if (!error.empty()) {
		throw BinderException(binder.FormatError(*expr, error));
	}

	// create a MacroBinding to bind this macro's parameters to its arguments
	vector<LogicalType> types;
	vector<string> names;
	// positional parameters
	for (idx_t i = 0; i < macro_def.parameters.size(); i++) {
		types.emplace_back(LogicalType::SQLNULL);
		auto &param = macro_def.parameters[i]->Cast<ColumnRefExpression>();
		names.push_back(param.GetColumnName());
	}
	// default parameters
	for (auto it = macro_def.default_parameters.begin(); it != macro_def.default_parameters.end(); it++) {
		types.emplace_back(LogicalType::SQLNULL);
		names.push_back(it->first);
		// now push the defaults into the positionals
		positionals.push_back(std::move(defaults[it->first]));
	}
	auto new_macro_binding = make_uniq<DummyBinding>(types, names, macro_func.name);
	new_macro_binding->arguments = &positionals;
	macro_binding = new_macro_binding.get();

	// replace current expression with stored macro expression, and replace params
	expr = macro_def.expression->Copy();
	ReplaceMacroParametersRecursive(expr);

	// bind the unfolded macro
	return BindExpression(expr, depth);
}

} // namespace duckdb








namespace duckdb {

static LogicalType ResolveNotType(OperatorExpression &op, vector<unique_ptr<Expression>> &children) {
	// NOT expression, cast child to BOOLEAN
	D_ASSERT(children.size() == 1);
	children[0] = BoundCastExpression::AddDefaultCastToType(std::move(children[0]), LogicalType::BOOLEAN);
	return LogicalType(LogicalTypeId::BOOLEAN);
}

static LogicalType ResolveInType(OperatorExpression &op, vector<unique_ptr<Expression>> &children) {
	if (children.empty()) {
		throw InternalException("IN requires at least a single child node");
	}
	// get the maximum type from the children
	LogicalType max_type = children[0]->return_type;
	bool any_varchar = children[0]->return_type == LogicalType::VARCHAR;
	bool any_enum = children[0]->return_type.id() == LogicalTypeId::ENUM;
	for (idx_t i = 1; i < children.size(); i++) {
		max_type = LogicalType::MaxLogicalType(max_type, children[i]->return_type);
		if (children[i]->return_type == LogicalType::VARCHAR) {
			any_varchar = true;
		}
		if (children[i]->return_type.id() == LogicalTypeId::ENUM) {
			any_enum = true;
		}
	}
	if (any_varchar && any_enum) {
		// For the coalesce function, we must be sure we always upcast the parameters to VARCHAR, if there are at least
		// one enum and one varchar
		max_type = LogicalType::VARCHAR;
	}

	// cast all children to the same type
	for (idx_t i = 0; i < children.size(); i++) {
		children[i] = BoundCastExpression::AddDefaultCastToType(std::move(children[i]), max_type);
	}
	// (NOT) IN always returns a boolean
	return LogicalType::BOOLEAN;
}

static LogicalType ResolveOperatorType(OperatorExpression &op, vector<unique_ptr<Expression>> &children) {
	switch (op.type) {
	case ExpressionType::OPERATOR_IS_NULL:
	case ExpressionType::OPERATOR_IS_NOT_NULL:
		// IS (NOT) NULL always returns a boolean, and does not cast its children
		if (!children[0]->return_type.IsValid()) {
			throw ParameterNotResolvedException();
		}
		return LogicalType::BOOLEAN;
	case ExpressionType::COMPARE_IN:
	case ExpressionType::COMPARE_NOT_IN:
		return ResolveInType(op, children);
	case ExpressionType::OPERATOR_COALESCE: {
		ResolveInType(op, children);
		return children[0]->return_type;
	}
	case ExpressionType::OPERATOR_NOT:
		return ResolveNotType(op, children);
	default:
		throw InternalException("Unrecognized expression type for ResolveOperatorType");
	}
}

BindResult ExpressionBinder::BindGroupingFunction(OperatorExpression &op, idx_t depth) {
	return BindResult("GROUPING function is not supported here");
}

BindResult ExpressionBinder::BindExpression(OperatorExpression &op, idx_t depth) {
	if (op.type == ExpressionType::GROUPING_FUNCTION) {
		return BindGroupingFunction(op, depth);
	}
	// bind the children of the operator expression
	string error;
	for (idx_t i = 0; i < op.children.size(); i++) {
		BindChild(op.children[i], depth, error);
	}
	if (!error.empty()) {
		return BindResult(error);
	}
	// all children bound successfully
	string function_name;
	switch (op.type) {
	case ExpressionType::ARRAY_EXTRACT: {
		D_ASSERT(op.children[0]->expression_class == ExpressionClass::BOUND_EXPRESSION);
		auto &b_exp = BoundExpression::GetExpression(*op.children[0]);
		if (b_exp->return_type.id() == LogicalTypeId::MAP) {
			function_name = "map_extract";
		} else {
			function_name = "array_extract";
		}
		break;
	}
	case ExpressionType::ARRAY_SLICE:
		function_name = "array_slice";
		break;
	case ExpressionType::STRUCT_EXTRACT: {
		D_ASSERT(op.children.size() == 2);
		D_ASSERT(op.children[0]->expression_class == ExpressionClass::BOUND_EXPRESSION);
		D_ASSERT(op.children[1]->expression_class == ExpressionClass::BOUND_EXPRESSION);
		auto &extract_exp = BoundExpression::GetExpression(*op.children[0]);
		auto &name_exp = BoundExpression::GetExpression(*op.children[1]);
		auto extract_expr_type = extract_exp->return_type.id();
		if (extract_expr_type != LogicalTypeId::STRUCT && extract_expr_type != LogicalTypeId::UNION &&
		    extract_expr_type != LogicalTypeId::SQLNULL) {
			return BindResult(StringUtil::Format(
			    "Cannot extract field %s from expression \"%s\" because it is not a struct or a union",
			    name_exp->ToString(), extract_exp->ToString()));
		}
		function_name = extract_expr_type == LogicalTypeId::UNION ? "union_extract" : "struct_extract";
		break;
	}
	case ExpressionType::ARRAY_CONSTRUCTOR:
		function_name = "list_value";
		break;
	case ExpressionType::ARROW:
		function_name = "json_extract";
		break;
	default:
		break;
	}
	if (!function_name.empty()) {
		auto function = make_uniq_base<ParsedExpression, FunctionExpression>(function_name, std::move(op.children));
		return BindExpression(function, depth, false);
	}

	vector<unique_ptr<Expression>> children;
	for (idx_t i = 0; i < op.children.size(); i++) {
		D_ASSERT(op.children[i]->expression_class == ExpressionClass::BOUND_EXPRESSION);
		children.push_back(std::move(BoundExpression::GetExpression(*op.children[i])));
	}
	// now resolve the types
	LogicalType result_type = ResolveOperatorType(op, children);
	if (op.type == ExpressionType::OPERATOR_COALESCE) {
		if (children.empty()) {
			throw BinderException("COALESCE needs at least one child");
		}
		if (children.size() == 1) {
			return BindResult(std::move(children[0]));
		}
	}

	auto result = make_uniq<BoundOperatorExpression>(op.type, result_type);
	for (auto &child : children) {
		result->children.push_back(std::move(child));
	}
	return BindResult(std::move(result));
}

} // namespace duckdb






namespace duckdb {

BindResult ExpressionBinder::BindExpression(ParameterExpression &expr, idx_t depth) {
	D_ASSERT(expr.parameter_nr > 0);
	auto bound_parameter = make_uniq<BoundParameterExpression>(expr.parameter_nr);
	bound_parameter->alias = expr.alias;
	if (!binder.parameters) {
		throw BinderException("Unexpected prepared parameter. This type of statement can't be prepared!");
	}
	auto parameter_idx = expr.parameter_nr;
	// check if a parameter value has already been supplied
	if (parameter_idx <= binder.parameters->parameter_data.size()) {
		// it has! emit a constant directly
		auto &data = binder.parameters->parameter_data[parameter_idx - 1];
		auto constant = make_uniq<BoundConstantExpression>(data.value);
		constant->alias = expr.alias;
		return BindResult(std::move(constant));
	}
	auto entry = binder.parameters->parameters.find(parameter_idx);
	if (entry == binder.parameters->parameters.end()) {
		// no entry yet: create a new one
		auto data = make_shared<BoundParameterData>();
		data->return_type = binder.parameters->GetReturnType(parameter_idx - 1);
		bound_parameter->return_type = data->return_type;
		bound_parameter->parameter_data = data;
		binder.parameters->parameters[parameter_idx] = std::move(data);
	} else {
		// a prepared statement with this parameter index was already there: use it
		auto &data = entry->second;
		bound_parameter->parameter_data = data;
		bound_parameter->return_type = binder.parameters->GetReturnType(parameter_idx - 1);
	}
	return BindResult(std::move(bound_parameter));
}

} // namespace duckdb




namespace duckdb {

BindResult ExpressionBinder::BindExpression(PositionalReferenceExpression &ref, idx_t depth) {
	if (depth != 0) {
		return BindResult("Positional reference expression could not be bound");
	}
	return binder.bind_context.BindColumn(ref, depth);
}

} // namespace duckdb








namespace duckdb {

string GetColumnsStringValue(ParsedExpression &expr) {
	if (expr.type == ExpressionType::COLUMN_REF) {
		auto &colref = expr.Cast<ColumnRefExpression>();
		return colref.GetColumnName();
	} else {
		return expr.ToString();
	}
}

bool Binder::FindStarExpression(unique_ptr<ParsedExpression> &expr, StarExpression **star, bool is_root,
                                bool in_columns) {
	bool has_star = false;
	if (expr->GetExpressionClass() == ExpressionClass::STAR) {
		auto &current_star = expr->Cast<StarExpression>();
		if (!current_star.columns) {
			if (is_root) {
				*star = &current_star;
				return true;
			}
			if (!in_columns) {
				throw BinderException(
				    "STAR expression is only allowed as the root element of an expression. Use COLUMNS(*) instead.");
			}
			// star expression inside a COLUMNS - convert to a constant list
			if (!current_star.replace_list.empty()) {
				throw BinderException(
				    "STAR expression with REPLACE list is only allowed as the root element of COLUMNS");
			}
			vector<unique_ptr<ParsedExpression>> star_list;
			bind_context.GenerateAllColumnExpressions(current_star, star_list);

			vector<Value> values;
			values.reserve(star_list.size());
			for (auto &expr : star_list) {
				values.emplace_back(GetColumnsStringValue(*expr));
			}
			D_ASSERT(!values.empty());

			expr = make_uniq<ConstantExpression>(Value::LIST(LogicalType::VARCHAR, values));
			return true;
		}
		if (in_columns) {
			throw BinderException("COLUMNS expression is not allowed inside another COLUMNS expression");
		}
		in_columns = true;
		if (*star) {
			// we can have multiple
			if (!StarExpression::Equal(*star, &current_star)) {
				throw BinderException(
				    FormatError(*expr, "Multiple different STAR/COLUMNS in the same expression are not supported"));
			}
			return true;
		}
		*star = &current_star;
		has_star = true;
	}
	ParsedExpressionIterator::EnumerateChildren(*expr, [&](unique_ptr<ParsedExpression> &child_expr) {
		if (FindStarExpression(child_expr, star, false, in_columns)) {
			has_star = true;
		}
	});
	return has_star;
}

void Binder::ReplaceStarExpression(unique_ptr<ParsedExpression> &expr, unique_ptr<ParsedExpression> &replacement) {
	D_ASSERT(expr);
	if (expr->GetExpressionClass() == ExpressionClass::STAR) {
		D_ASSERT(replacement);
		expr = replacement->Copy();
		return;
	}
	ParsedExpressionIterator::EnumerateChildren(
	    *expr, [&](unique_ptr<ParsedExpression> &child_expr) { ReplaceStarExpression(child_expr, replacement); });
}

void Binder::ExpandStarExpression(unique_ptr<ParsedExpression> expr,
                                  vector<unique_ptr<ParsedExpression>> &new_select_list) {
	StarExpression *star = nullptr;
	if (!FindStarExpression(expr, &star, true, false)) {
		// no star expression: add it as-is
		D_ASSERT(!star);
		new_select_list.push_back(std::move(expr));
		return;
	}
	D_ASSERT(star);
	vector<unique_ptr<ParsedExpression>> star_list;
	// we have star expressions! expand the list of star expressions
	bind_context.GenerateAllColumnExpressions(*star, star_list);

	if (star->expr) {
		// COLUMNS with an expression
		// two options:
		// VARCHAR parameter <- this is a regular expression
		// LIST of VARCHAR parameters <- this is a set of columns
		TableFunctionBinder binder(*this, context);
		auto child = star->expr->Copy();
		auto result = binder.Bind(child);
		if (!result->IsFoldable()) {
			// cannot resolve parameters here
			if (star->expr->HasParameter()) {
				throw ParameterNotResolvedException();
			} else {
				throw BinderException("Unsupported expression in COLUMNS");
			}
		}
		auto val = ExpressionExecutor::EvaluateScalar(context, *result);
		if (val.type().id() == LogicalTypeId::VARCHAR) {
			// regex
			if (val.IsNull()) {
				throw BinderException("COLUMNS does not support NULL as regex argument");
			}
			auto &regex_str = StringValue::Get(val);
			duckdb_re2::RE2 regex(regex_str);
			if (!regex.error().empty()) {
				auto err = StringUtil::Format("Failed to compile regex \"%s\": %s", regex_str, regex.error());
				throw BinderException(FormatError(*star, err));
			}
			vector<unique_ptr<ParsedExpression>> new_list;
			for (idx_t i = 0; i < star_list.size(); i++) {
				auto &colref = star_list[i]->Cast<ColumnRefExpression>();
				if (!RE2::PartialMatch(colref.GetColumnName(), regex)) {
					continue;
				}
				new_list.push_back(std::move(star_list[i]));
			}
			if (new_list.empty()) {
				auto err = StringUtil::Format("No matching columns found that match regex \"%s\"", regex_str);
				throw BinderException(FormatError(*star, err));
			}
			star_list = std::move(new_list);
		} else if (val.type().id() == LogicalTypeId::LIST &&
		           ListType::GetChildType(val.type()).id() == LogicalTypeId::VARCHAR) {
			// list of varchar columns
			if (val.IsNull() || ListValue::GetChildren(val).empty()) {
				auto err =
				    StringUtil::Format("Star expression \"%s\" resulted in an empty set of columns", star->ToString());
				throw BinderException(FormatError(*star, err));
			}
			auto &children = ListValue::GetChildren(val);
			vector<unique_ptr<ParsedExpression>> new_list;
			// scan the list for all selected columns and construct a lookup table
			case_insensitive_map_t<bool> selected_set;
			for (auto &child : children) {
				selected_set.insert(make_pair(StringValue::Get(child), false));
			}
			// now check the list of all possible expressions and select which ones make it in
			for (auto &expr : star_list) {
				auto str = GetColumnsStringValue(*expr);
				auto entry = selected_set.find(str);
				if (entry != selected_set.end()) {
					new_list.push_back(std::move(expr));
					entry->second = true;
				}
			}
			// check if all expressions found a match
			for (auto &entry : selected_set) {
				if (!entry.second) {
					throw BinderException("Column \"%s\" was selected but was not found in the FROM clause",
					                      entry.first);
				}
			}
			star_list = std::move(new_list);
		} else {
			throw BinderException(FormatError(
			    *star, "COLUMNS expects either a VARCHAR argument (regex) or a LIST of VARCHAR (list of columns)"));
		}
	}

	// now perform the replacement
	for (idx_t i = 0; i < star_list.size(); i++) {
		auto new_expr = expr->Copy();
		ReplaceStarExpression(new_expr, star_list[i]);
		new_select_list.push_back(std::move(new_expr));
	}
}

void Binder::ExpandStarExpressions(vector<unique_ptr<ParsedExpression>> &select_list,
                                   vector<unique_ptr<ParsedExpression>> &new_select_list) {
	for (auto &select_element : select_list) {
		ExpandStarExpression(std::move(select_element), new_select_list);
	}
}

} // namespace duckdb







namespace duckdb {

class BoundSubqueryNode : public QueryNode {
public:
	BoundSubqueryNode(shared_ptr<Binder> subquery_binder, unique_ptr<BoundQueryNode> bound_node,
	                  unique_ptr<SelectStatement> subquery)
	    : QueryNode(QueryNodeType::BOUND_SUBQUERY_NODE), subquery_binder(std::move(subquery_binder)),
	      bound_node(std::move(bound_node)), subquery(std::move(subquery)) {
	}

	shared_ptr<Binder> subquery_binder;
	unique_ptr<BoundQueryNode> bound_node;
	unique_ptr<SelectStatement> subquery;

	const vector<unique_ptr<ParsedExpression>> &GetSelectList() const override {
		throw InternalException("Cannot get select list of bound subquery node");
	}

	string ToString() const override {
		throw InternalException("Cannot ToString bound subquery node");
	}
	unique_ptr<QueryNode> Copy() const override {
		throw InternalException("Cannot copy bound subquery node");
	}
	void Serialize(FieldWriter &writer) const override {
		throw InternalException("Cannot serialize bound subquery node");
	}

	void FormatSerialize(FormatSerializer &serializer) const override {
		throw InternalException("Cannot serialize bound subquery node");
	}
};

BindResult ExpressionBinder::BindExpression(SubqueryExpression &expr, idx_t depth) {
	if (expr.subquery->node->type != QueryNodeType::BOUND_SUBQUERY_NODE) {
		D_ASSERT(depth == 0);
		// first bind the actual subquery in a new binder
		auto subquery_binder = Binder::CreateBinder(context, &binder);
		subquery_binder->can_contain_nulls = true;
		auto bound_node = subquery_binder->BindNode(*expr.subquery->node);
		// check the correlated columns of the subquery for correlated columns with depth > 1
		for (idx_t i = 0; i < subquery_binder->correlated_columns.size(); i++) {
			CorrelatedColumnInfo corr = subquery_binder->correlated_columns[i];
			if (corr.depth > 1) {
				// depth > 1, the column references the query ABOVE the current one
				// add to the set of correlated columns for THIS query
				corr.depth -= 1;
				binder.AddCorrelatedColumn(corr);
			}
		}
		if (expr.subquery_type != SubqueryType::EXISTS && bound_node->types.size() > 1) {
			throw BinderException(binder.FormatError(
			    expr, StringUtil::Format("Subquery returns %zu columns - expected 1", bound_node->types.size())));
		}
		auto prior_subquery = std::move(expr.subquery);
		expr.subquery = make_uniq<SelectStatement>();
		expr.subquery->node =
		    make_uniq<BoundSubqueryNode>(std::move(subquery_binder), std::move(bound_node), std::move(prior_subquery));
	}
	// now bind the child node of the subquery
	if (expr.child) {
		// first bind the children of the subquery, if any
		string error = Bind(expr.child, depth);
		if (!error.empty()) {
			return BindResult(error);
		}
	}
	// both binding the child and binding the subquery was successful
	D_ASSERT(expr.subquery->node->type == QueryNodeType::BOUND_SUBQUERY_NODE);
	auto bound_subquery = (BoundSubqueryNode *)expr.subquery->node.get();
	auto subquery_binder = std::move(bound_subquery->subquery_binder);
	auto bound_node = std::move(bound_subquery->bound_node);
	LogicalType return_type =
	    expr.subquery_type == SubqueryType::SCALAR ? bound_node->types[0] : LogicalType(LogicalTypeId::BOOLEAN);
	if (return_type.id() == LogicalTypeId::UNKNOWN) {
		return_type = LogicalType::SQLNULL;
	}

	auto result = make_uniq<BoundSubqueryExpression>(return_type);
	if (expr.subquery_type == SubqueryType::ANY) {
		// ANY comparison
		// cast child and subquery child to equivalent types
		D_ASSERT(bound_node->types.size() == 1);
		auto &child = BoundExpression::GetExpression(*expr.child);
		auto compare_type = LogicalType::MaxLogicalType(child->return_type, bound_node->types[0]);
		child = BoundCastExpression::AddCastToType(context, std::move(child), compare_type);
		result->child_type = bound_node->types[0];
		result->child_target = compare_type;
		result->child = std::move(child);
	}
	result->binder = std::move(subquery_binder);
	result->subquery = std::move(bound_node);
	result->subquery_type = expr.subquery_type;
	result->comparison_type = expr.comparison_type;

	return BindResult(std::move(result));
}

} // namespace duckdb
















namespace duckdb {

unique_ptr<Expression> CreateBoundStructExtract(ClientContext &context, unique_ptr<Expression> expr, string key) {
	vector<unique_ptr<Expression>> arguments;
	arguments.push_back(std::move(expr));
	arguments.push_back(make_uniq<BoundConstantExpression>(Value(key)));
	auto extract_function = StructExtractFun::GetFunction();
	auto bind_info = extract_function.bind(context, extract_function, arguments);
	auto return_type = extract_function.return_type;
	auto result = make_uniq<BoundFunctionExpression>(return_type, std::move(extract_function), std::move(arguments),
	                                                 std::move(bind_info));
	result->alias = std::move(key);
	return std::move(result);
}

BindResult SelectBinder::BindUnnest(FunctionExpression &function, idx_t depth, bool root_expression) {
	// bind the children of the function expression
	if (depth > 0) {
		return BindResult(binder.FormatError(function, "UNNEST() for correlated expressions is not supported yet"));
	}
	string error;
	if (function.children.empty()) {
		return BindResult(binder.FormatError(function, "UNNEST() requires a single argument"));
	}
	idx_t max_depth = 1;
	if (function.children.size() != 1) {
		bool has_parameter = false;
		bool supported_argument = false;
		for (idx_t i = 1; i < function.children.size(); i++) {
			if (has_parameter) {
				return BindResult(binder.FormatError(function, "UNNEST() only supports a single additional argument"));
			}
			if (function.children[i]->HasParameter()) {
				throw ParameterNotAllowedException("Parameter not allowed in unnest parameter");
			}
			if (!function.children[i]->IsScalar()) {
				break;
			}
			auto alias = function.children[i]->alias;
			BindChild(function.children[i], depth, error);
			if (!error.empty()) {
				return BindResult(error);
			}
			auto &const_child = BoundExpression::GetExpression(*function.children[i]);
			auto value = ExpressionExecutor::EvaluateScalar(context, *const_child, true);
			if (alias == "recursive") {
				auto recursive = value.GetValue<bool>();
				if (recursive) {
					max_depth = NumericLimits<idx_t>::Maximum();
				}
			} else if (alias == "max_depth") {
				max_depth = value.GetValue<uint32_t>();
				if (max_depth == 0) {
					throw BinderException("UNNEST cannot have a max depth of 0");
				}
			} else if (!alias.empty()) {
				throw BinderException("Unsupported parameter \"%s\" for unnest", alias);
			} else {
				break;
			}
			has_parameter = true;
			supported_argument = true;
		}
		if (!supported_argument) {
			return BindResult(binder.FormatError(function, "UNNEST - unsupported extra argument, unnest only supports "
			                                               "recursive := [true/false] or max_depth := #"));
		}
	}
	unnest_level++;
	BindChild(function.children[0], depth, error);
	if (!error.empty()) {
		// failed to bind
		// try to bind correlated columns manually
		if (!BindCorrelatedColumns(function.children[0])) {
			return BindResult(error);
		}
		auto &bound_expr = BoundExpression::GetExpression(*function.children[0]);
		ExtractCorrelatedExpressions(binder, *bound_expr);
	}
	auto &child = BoundExpression::GetExpression(*function.children[0]);
	auto &child_type = child->return_type;
	unnest_level--;

	if (unnest_level > 0) {
		throw BinderException(
		    "Nested UNNEST calls are not supported - use UNNEST(x, recursive := true) to unnest multiple levels");
	}

	switch (child_type.id()) {
	case LogicalTypeId::UNKNOWN:
		throw ParameterNotResolvedException();
	case LogicalTypeId::LIST:
	case LogicalTypeId::STRUCT:
	case LogicalTypeId::SQLNULL:
		break;
	default:
		return BindResult(binder.FormatError(function, "UNNEST() can only be applied to lists, structs and NULL"));
	}

	idx_t list_unnests;
	idx_t struct_unnests = 0;

	auto unnest_expr = std::move(child);
	if (child_type.id() == LogicalTypeId::SQLNULL) {
		list_unnests = 1;
	} else {
		// first do all of the list unnests
		auto type = child_type;
		list_unnests = 0;
		while (type.id() == LogicalTypeId::LIST) {
			type = ListType::GetChildType(type);
			list_unnests++;
			if (list_unnests >= max_depth) {
				break;
			}
		}
		// unnest structs all the way afterwards, if there are any
		if (type.id() == LogicalTypeId::STRUCT) {
			struct_unnests = max_depth - list_unnests;
		}
	}
	if (struct_unnests > 0 && !root_expression) {
		return BindResult(binder.FormatError(
		    function, "UNNEST() on a struct column can only be applied as the root element of a SELECT expression"));
	}
	// perform all of the list unnests first
	auto return_type = child_type;
	for (idx_t current_depth = 0; current_depth < list_unnests; current_depth++) {
		if (return_type.id() == LogicalTypeId::LIST) {
			return_type = ListType::GetChildType(return_type);
		}
		auto result = make_uniq<BoundUnnestExpression>(return_type);
		result->child = std::move(unnest_expr);
		auto alias = function.alias.empty() ? result->ToString() : function.alias;

		auto current_level = unnest_level + list_unnests - current_depth - 1;
		auto entry = node.unnests.find(current_level);
		idx_t unnest_table_index;
		idx_t unnest_column_index;
		if (entry == node.unnests.end()) {
			BoundUnnestNode unnest_node;
			unnest_node.index = binder.GenerateTableIndex();
			unnest_node.expressions.push_back(std::move(result));
			unnest_table_index = unnest_node.index;
			unnest_column_index = 0;
			node.unnests.insert(make_pair(current_level, std::move(unnest_node)));
		} else {
			unnest_table_index = entry->second.index;
			unnest_column_index = entry->second.expressions.size();
			entry->second.expressions.push_back(std::move(result));
		}
		// now create a column reference referring to the unnest
		unnest_expr = make_uniq<BoundColumnRefExpression>(
		    std::move(alias), return_type, ColumnBinding(unnest_table_index, unnest_column_index), depth);
	}
	// now perform struct unnests, if any
	if (struct_unnests > 0) {
		vector<unique_ptr<Expression>> struct_expressions;
		struct_expressions.push_back(std::move(unnest_expr));

		for (idx_t i = 0; i < struct_unnests; i++) {
			vector<unique_ptr<Expression>> new_expressions;
			// check if there are any structs left
			bool has_structs = false;
			for (auto &expr : struct_expressions) {
				if (expr->return_type.id() == LogicalTypeId::STRUCT) {
					// struct! push a struct_extract
					auto &child_types = StructType::GetChildTypes(expr->return_type);
					for (auto &entry : child_types) {
						new_expressions.push_back(CreateBoundStructExtract(context, expr->Copy(), entry.first));
					}
					has_structs = true;
				} else {
					// not a struct - push as-is
					new_expressions.push_back(std::move(expr));
				}
			}
			struct_expressions = std::move(new_expressions);
			if (!has_structs) {
				break;
			}
		}
		expanded_expressions = std::move(struct_expressions);
		unnest_expr = make_uniq<BoundConstantExpression>(Value(42));
	}
	return BindResult(std::move(unnest_expr));
}

} // namespace duckdb
















namespace duckdb {

static LogicalType ResolveWindowExpressionType(ExpressionType window_type, const vector<LogicalType> &child_types) {

	idx_t param_count;
	switch (window_type) {
	case ExpressionType::WINDOW_RANK:
	case ExpressionType::WINDOW_RANK_DENSE:
	case ExpressionType::WINDOW_ROW_NUMBER:
	case ExpressionType::WINDOW_PERCENT_RANK:
	case ExpressionType::WINDOW_CUME_DIST:
		param_count = 0;
		break;
	case ExpressionType::WINDOW_NTILE:
	case ExpressionType::WINDOW_FIRST_VALUE:
	case ExpressionType::WINDOW_LAST_VALUE:
	case ExpressionType::WINDOW_LEAD:
	case ExpressionType::WINDOW_LAG:
		param_count = 1;
		break;
	case ExpressionType::WINDOW_NTH_VALUE:
		param_count = 2;
		break;
	default:
		throw InternalException("Unrecognized window expression type " + ExpressionTypeToString(window_type));
	}
	if (child_types.size() != param_count) {
		throw BinderException("%s needs %d parameter%s, got %d", ExpressionTypeToString(window_type), param_count,
		                      param_count == 1 ? "" : "s", child_types.size());
	}
	switch (window_type) {
	case ExpressionType::WINDOW_PERCENT_RANK:
	case ExpressionType::WINDOW_CUME_DIST:
		return LogicalType(LogicalTypeId::DOUBLE);
	case ExpressionType::WINDOW_ROW_NUMBER:
	case ExpressionType::WINDOW_RANK:
	case ExpressionType::WINDOW_RANK_DENSE:
	case ExpressionType::WINDOW_NTILE:
		return LogicalType::BIGINT;
	case ExpressionType::WINDOW_NTH_VALUE:
	case ExpressionType::WINDOW_FIRST_VALUE:
	case ExpressionType::WINDOW_LAST_VALUE:
	case ExpressionType::WINDOW_LEAD:
	case ExpressionType::WINDOW_LAG:
		return child_types[0];
	default:
		throw InternalException("Unrecognized window expression type " + ExpressionTypeToString(window_type));
	}
}

static unique_ptr<Expression> GetExpression(unique_ptr<ParsedExpression> &expr) {
	if (!expr) {
		return nullptr;
	}
	D_ASSERT(expr.get());
	D_ASSERT(expr->expression_class == ExpressionClass::BOUND_EXPRESSION);
	return std::move(BoundExpression::GetExpression(*expr));
}

static unique_ptr<Expression> CastWindowExpression(unique_ptr<ParsedExpression> &expr, const LogicalType &type) {
	if (!expr) {
		return nullptr;
	}
	D_ASSERT(expr.get());
	D_ASSERT(expr->expression_class == ExpressionClass::BOUND_EXPRESSION);

	auto &bound = BoundExpression::GetExpression(*expr);
	bound = BoundCastExpression::AddDefaultCastToType(std::move(bound), type);

	return std::move(bound);
}

static LogicalType BindRangeExpression(ClientContext &context, const string &name, unique_ptr<ParsedExpression> &expr,
                                       unique_ptr<ParsedExpression> &order_expr) {

	vector<unique_ptr<Expression>> children;

	D_ASSERT(order_expr.get());
	D_ASSERT(order_expr->expression_class == ExpressionClass::BOUND_EXPRESSION);
	auto &bound_order = BoundExpression::GetExpression(*order_expr);
	children.emplace_back(bound_order->Copy());

	D_ASSERT(expr.get());
	D_ASSERT(expr->expression_class == ExpressionClass::BOUND_EXPRESSION);
	auto &bound = BoundExpression::GetExpression(*expr);
	children.emplace_back(std::move(bound));

	string error;
	FunctionBinder function_binder(context);
	auto function = function_binder.BindScalarFunction(DEFAULT_SCHEMA, name, std::move(children), error, true);
	if (!function) {
		throw BinderException(error);
	}
	bound = std::move(function);
	return bound->return_type;
}

BindResult BaseSelectBinder::BindWindow(WindowExpression &window, idx_t depth) {
	auto name = window.GetName();

	QueryErrorContext error_context(binder.GetRootStatement(), window.query_location);
	if (inside_window) {
		throw BinderException(error_context.FormatError("window function calls cannot be nested"));
	}
	if (depth > 0) {
		throw BinderException(error_context.FormatError("correlated columns in window functions not supported"));
	}
	// If we have range expressions, then only one order by clause is allowed.
	if ((window.start == WindowBoundary::EXPR_PRECEDING_RANGE || window.start == WindowBoundary::EXPR_FOLLOWING_RANGE ||
	     window.end == WindowBoundary::EXPR_PRECEDING_RANGE || window.end == WindowBoundary::EXPR_FOLLOWING_RANGE) &&
	    window.orders.size() != 1) {
		throw BinderException(error_context.FormatError("RANGE frames must have only one ORDER BY expression"));
	}
	// bind inside the children of the window function
	// we set the inside_window flag to true to prevent binding nested window functions
	this->inside_window = true;
	string error;
	for (auto &child : window.children) {
		BindChild(child, depth, error);
	}
	for (auto &child : window.partitions) {
		BindChild(child, depth, error);
	}
	for (auto &order : window.orders) {
		BindChild(order.expression, depth, error);
	}
	BindChild(window.filter_expr, depth, error);
	BindChild(window.start_expr, depth, error);
	BindChild(window.end_expr, depth, error);
	BindChild(window.offset_expr, depth, error);
	BindChild(window.default_expr, depth, error);

	this->inside_window = false;
	if (!error.empty()) {
		// failed to bind children of window function
		return BindResult(error);
	}
	// successfully bound all children: create bound window function
	vector<LogicalType> types;
	vector<unique_ptr<Expression>> children;
	for (auto &child : window.children) {
		D_ASSERT(child.get());
		D_ASSERT(child->expression_class == ExpressionClass::BOUND_EXPRESSION);
		auto &bound = BoundExpression::GetExpression(*child);
		// Add casts for positional arguments
		const auto argno = children.size();
		switch (window.type) {
		case ExpressionType::WINDOW_NTILE:
			// ntile(bigint)
			if (argno == 0) {
				bound = BoundCastExpression::AddCastToType(context, std::move(bound), LogicalType::BIGINT);
			}
			break;
		case ExpressionType::WINDOW_NTH_VALUE:
			// nth_value(<expr>, index)
			if (argno == 1) {
				bound = BoundCastExpression::AddCastToType(context, std::move(bound), LogicalType::BIGINT);
			}
		default:
			break;
		}
		types.push_back(bound->return_type);
		children.push_back(std::move(bound));
	}
	//  Determine the function type.
	LogicalType sql_type;
	unique_ptr<AggregateFunction> aggregate;
	unique_ptr<FunctionData> bind_info;
	if (window.type == ExpressionType::WINDOW_AGGREGATE) {
		//  Look up the aggregate function in the catalog
		auto &func = Catalog::GetEntry<AggregateFunctionCatalogEntry>(context, window.catalog, window.schema,
		                                                              window.function_name, error_context);
		D_ASSERT(func.type == CatalogType::AGGREGATE_FUNCTION_ENTRY);

		// bind the aggregate
		string error;
		FunctionBinder function_binder(context);
		auto best_function = function_binder.BindFunction(func.name, func.functions, types, error);
		if (best_function == DConstants::INVALID_INDEX) {
			throw BinderException(binder.FormatError(window, error));
		}
		// found a matching function! bind it as an aggregate
		auto bound_function = func.functions.GetFunctionByOffset(best_function);
		auto bound_aggregate = function_binder.BindAggregateFunction(bound_function, std::move(children));
		// create the aggregate
		aggregate = make_uniq<AggregateFunction>(bound_aggregate->function);
		bind_info = std::move(bound_aggregate->bind_info);
		children = std::move(bound_aggregate->children);
		sql_type = bound_aggregate->return_type;
	} else {
		// fetch the child of the non-aggregate window function (if any)
		sql_type = ResolveWindowExpressionType(window.type, types);
	}
	auto result = make_uniq<BoundWindowExpression>(window.type, sql_type, std::move(aggregate), std::move(bind_info));
	result->children = std::move(children);
	for (auto &child : window.partitions) {
		result->partitions.push_back(GetExpression(child));
	}
	result->ignore_nulls = window.ignore_nulls;

	// Convert RANGE boundary expressions to ORDER +/- expressions.
	// Note that PRECEEDING and FOLLOWING refer to the sequential order in the frame,
	// not the natural ordering of the type. This means that the offset arithmetic must be reversed
	// for ORDER BY DESC.
	auto &config = DBConfig::GetConfig(context);
	auto range_sense = OrderType::INVALID;
	LogicalType start_type = LogicalType::BIGINT;
	if (window.start == WindowBoundary::EXPR_PRECEDING_RANGE) {
		D_ASSERT(window.orders.size() == 1);
		range_sense = config.ResolveOrder(window.orders[0].type);
		const auto name = (range_sense == OrderType::ASCENDING) ? "-" : "+";
		start_type = BindRangeExpression(context, name, window.start_expr, window.orders[0].expression);
	} else if (window.start == WindowBoundary::EXPR_FOLLOWING_RANGE) {
		D_ASSERT(window.orders.size() == 1);
		range_sense = config.ResolveOrder(window.orders[0].type);
		const auto name = (range_sense == OrderType::ASCENDING) ? "+" : "-";
		start_type = BindRangeExpression(context, name, window.start_expr, window.orders[0].expression);
	}

	LogicalType end_type = LogicalType::BIGINT;
	if (window.end == WindowBoundary::EXPR_PRECEDING_RANGE) {
		D_ASSERT(window.orders.size() == 1);
		range_sense = config.ResolveOrder(window.orders[0].type);
		const auto name = (range_sense == OrderType::ASCENDING) ? "-" : "+";
		end_type = BindRangeExpression(context, name, window.end_expr, window.orders[0].expression);
	} else if (window.end == WindowBoundary::EXPR_FOLLOWING_RANGE) {
		D_ASSERT(window.orders.size() == 1);
		range_sense = config.ResolveOrder(window.orders[0].type);
		const auto name = (range_sense == OrderType::ASCENDING) ? "+" : "-";
		end_type = BindRangeExpression(context, name, window.end_expr, window.orders[0].expression);
	}

	// Cast ORDER and boundary expressions to the same type
	if (range_sense != OrderType::INVALID) {
		D_ASSERT(window.orders.size() == 1);

		auto &order_expr = window.orders[0].expression;
		D_ASSERT(order_expr.get());
		D_ASSERT(order_expr->expression_class == ExpressionClass::BOUND_EXPRESSION);
		auto &bound_order = BoundExpression::GetExpression(*order_expr);
		auto order_type = bound_order->return_type;
		if (window.start_expr) {
			order_type = LogicalType::MaxLogicalType(order_type, start_type);
		}
		if (window.end_expr) {
			order_type = LogicalType::MaxLogicalType(order_type, end_type);
		}

		// Cast all three to match
		bound_order = BoundCastExpression::AddCastToType(context, std::move(bound_order), order_type);
		start_type = end_type = order_type;
	}

	for (auto &order : window.orders) {
		auto type = config.ResolveOrder(order.type);
		auto null_order = config.ResolveNullOrder(type, order.null_order);
		auto expression = GetExpression(order.expression);
		result->orders.emplace_back(type, null_order, std::move(expression));
	}

	result->filter_expr = CastWindowExpression(window.filter_expr, LogicalType::BOOLEAN);

	result->start_expr = CastWindowExpression(window.start_expr, start_type);
	result->end_expr = CastWindowExpression(window.end_expr, end_type);
	result->offset_expr = CastWindowExpression(window.offset_expr, LogicalType::BIGINT);
	result->default_expr = CastWindowExpression(window.default_expr, result->return_type);
	result->start = window.start;
	result->end = window.end;

	// create a BoundColumnRef that references this entry
	auto colref = make_uniq<BoundColumnRefExpression>(std::move(name), result->return_type,
	                                                  ColumnBinding(node.window_index, node.windows.size()), depth);
	// move the WINDOW expression into the set of bound windows
	node.windows.push_back(std::move(result));
	return BindResult(std::move(colref));
}

} // namespace duckdb








namespace duckdb {

unique_ptr<BoundQueryNode> Binder::BindNode(RecursiveCTENode &statement) {
	auto result = make_uniq<BoundRecursiveCTENode>();

	// first recursively visit the recursive CTE operations
	// the left side is visited first and is added to the BindContext of the right side
	D_ASSERT(statement.left);
	D_ASSERT(statement.right);

	result->ctename = statement.ctename;
	result->union_all = statement.union_all;
	result->setop_index = GenerateTableIndex();

	result->left_binder = Binder::CreateBinder(context, this);
	result->left = result->left_binder->BindNode(*statement.left);

	// the result types of the CTE are the types of the LHS
	result->types = result->left->types;
	// names are picked from the LHS, unless aliases are explicitly specified
	result->names = result->left->names;
	for (idx_t i = 0; i < statement.aliases.size() && i < result->names.size(); i++) {
		result->names[i] = statement.aliases[i];
	}

	// This allows the right side to reference the CTE recursively
	bind_context.AddGenericBinding(result->setop_index, statement.ctename, result->names, result->types);

	result->right_binder = Binder::CreateBinder(context, this);

	// Add bindings of left side to temporary CTE bindings context
	result->right_binder->bind_context.AddCTEBinding(result->setop_index, statement.ctename, result->names,
	                                                 result->types);
	result->right = result->right_binder->BindNode(*statement.right);

	// move the correlated expressions from the child binders to this binder
	MoveCorrelatedExpressions(*result->left_binder);
	MoveCorrelatedExpressions(*result->right_binder);

	// now both sides have been bound we can resolve types
	if (result->left->types.size() != result->right->types.size()) {
		throw BinderException("Set operations can only apply to expressions with the "
		                      "same number of result columns");
	}

	if (!statement.modifiers.empty()) {
		throw NotImplementedException("FIXME: bind modifiers in recursive CTE");
	}

	return std::move(result);
}

} // namespace duckdb























namespace duckdb {

unique_ptr<Expression> Binder::BindOrderExpression(OrderBinder &order_binder, unique_ptr<ParsedExpression> expr) {
	// we treat the Distinct list as a order by
	auto bound_expr = order_binder.Bind(std::move(expr));
	if (!bound_expr) {
		// DISTINCT ON non-integer constant
		// remove the expression from the DISTINCT ON list
		return nullptr;
	}
	D_ASSERT(bound_expr->type == ExpressionType::BOUND_COLUMN_REF);
	return bound_expr;
}

unique_ptr<Expression> Binder::BindDelimiter(ClientContext &context, OrderBinder &order_binder,
                                             unique_ptr<ParsedExpression> delimiter, const LogicalType &type,
                                             Value &delimiter_value) {
	auto new_binder = Binder::CreateBinder(context, this, true);
	if (delimiter->HasSubquery()) {
		if (!order_binder.HasExtraList()) {
			throw BinderException("Subquery in LIMIT/OFFSET not supported in set operation");
		}
		return order_binder.CreateExtraReference(std::move(delimiter));
	}
	ExpressionBinder expr_binder(*new_binder, context);
	expr_binder.target_type = type;
	auto expr = expr_binder.Bind(delimiter);
	if (expr->IsFoldable()) {
		//! this is a constant
		delimiter_value = ExpressionExecutor::EvaluateScalar(context, *expr).CastAs(context, type);
		return nullptr;
	}
	if (!new_binder->correlated_columns.empty()) {
		throw BinderException("Correlated columns not supported in LIMIT/OFFSET");
	}
	// move any correlated columns to this binder
	MoveCorrelatedExpressions(*new_binder);
	return expr;
}

duckdb::unique_ptr<BoundResultModifier> Binder::BindLimit(OrderBinder &order_binder, LimitModifier &limit_mod) {
	auto result = make_uniq<BoundLimitModifier>();
	if (limit_mod.limit) {
		Value val;
		result->limit = BindDelimiter(context, order_binder, std::move(limit_mod.limit), LogicalType::BIGINT, val);
		if (!result->limit) {
			result->limit_val = val.IsNull() ? NumericLimits<int64_t>::Maximum() : val.GetValue<int64_t>();
			if (result->limit_val < 0) {
				throw BinderException("LIMIT cannot be negative");
			}
		}
	}
	if (limit_mod.offset) {
		Value val;
		result->offset = BindDelimiter(context, order_binder, std::move(limit_mod.offset), LogicalType::BIGINT, val);
		if (!result->offset) {
			result->offset_val = val.IsNull() ? 0 : val.GetValue<int64_t>();
			if (result->offset_val < 0) {
				throw BinderException("OFFSET cannot be negative");
			}
		}
	}
	return std::move(result);
}

unique_ptr<BoundResultModifier> Binder::BindLimitPercent(OrderBinder &order_binder, LimitPercentModifier &limit_mod) {
	auto result = make_uniq<BoundLimitPercentModifier>();
	if (limit_mod.limit) {
		Value val;
		result->limit = BindDelimiter(context, order_binder, std::move(limit_mod.limit), LogicalType::DOUBLE, val);
		if (!result->limit) {
			result->limit_percent = val.IsNull() ? 100 : val.GetValue<double>();
			if (result->limit_percent < 0.0) {
				throw Exception("Limit percentage can't be negative value");
			}
		}
	}
	if (limit_mod.offset) {
		Value val;
		result->offset = BindDelimiter(context, order_binder, std::move(limit_mod.offset), LogicalType::BIGINT, val);
		if (!result->offset) {
			result->offset_val = val.IsNull() ? 0 : val.GetValue<int64_t>();
		}
	}
	return std::move(result);
}

void Binder::BindModifiers(OrderBinder &order_binder, QueryNode &statement, BoundQueryNode &result) {
	for (auto &mod : statement.modifiers) {
		unique_ptr<BoundResultModifier> bound_modifier;
		switch (mod->type) {
		case ResultModifierType::DISTINCT_MODIFIER: {
			auto &distinct = (DistinctModifier &)*mod;
			auto bound_distinct = make_uniq<BoundDistinctModifier>();
			bound_distinct->distinct_type =
			    distinct.distinct_on_targets.empty() ? DistinctType::DISTINCT : DistinctType::DISTINCT_ON;
			if (distinct.distinct_on_targets.empty()) {
				for (idx_t i = 0; i < result.names.size(); i++) {
					distinct.distinct_on_targets.push_back(make_uniq<ConstantExpression>(Value::INTEGER(1 + i)));
				}
			}
			for (auto &distinct_on_target : distinct.distinct_on_targets) {
				auto expr = BindOrderExpression(order_binder, std::move(distinct_on_target));
				if (!expr) {
					continue;
				}
				bound_distinct->target_distincts.push_back(std::move(expr));
			}
			bound_modifier = std::move(bound_distinct);
			break;
		}
		case ResultModifierType::ORDER_MODIFIER: {
			auto &order = (OrderModifier &)*mod;
			auto bound_order = make_uniq<BoundOrderModifier>();
			auto &config = DBConfig::GetConfig(context);
			D_ASSERT(!order.orders.empty());
			auto &order_binders = order_binder.GetBinders();
			if (order.orders.size() == 1 && order.orders[0].expression->type == ExpressionType::STAR) {
				auto &star = order.orders[0].expression->Cast<StarExpression>();
				if (star.exclude_list.empty() && star.replace_list.empty() && !star.expr) {
					// ORDER BY ALL
					// replace the order list with the all elements in the SELECT list
					auto order_type = order.orders[0].type;
					auto null_order = order.orders[0].null_order;

					vector<OrderByNode> new_orders;
					for (idx_t i = 0; i < order_binder.MaxCount(); i++) {
						new_orders.emplace_back(order_type, null_order,
						                        make_uniq<ConstantExpression>(Value::INTEGER(i + 1)));
					}
					order.orders = std::move(new_orders);
				}
			}
			for (auto &order_node : order.orders) {
				vector<unique_ptr<ParsedExpression>> order_list;
				order_binders[0]->ExpandStarExpression(std::move(order_node.expression), order_list);

				auto type = config.ResolveOrder(order_node.type);
				auto null_order = config.ResolveNullOrder(type, order_node.null_order);
				for (auto &order_expr : order_list) {
					auto bound_expr = BindOrderExpression(order_binder, std::move(order_expr));
					if (!bound_expr) {
						continue;
					}
					bound_order->orders.emplace_back(type, null_order, std::move(bound_expr));
				}
			}
			if (!bound_order->orders.empty()) {
				bound_modifier = std::move(bound_order);
			}
			break;
		}
		case ResultModifierType::LIMIT_MODIFIER:
			bound_modifier = BindLimit(order_binder, (LimitModifier &)*mod);
			break;
		case ResultModifierType::LIMIT_PERCENT_MODIFIER:
			bound_modifier = BindLimitPercent(order_binder, (LimitPercentModifier &)*mod);
			break;
		default:
			throw Exception("Unsupported result modifier");
		}
		if (bound_modifier) {
			result.modifiers.push_back(std::move(bound_modifier));
		}
	}
}

static void AssignReturnType(unique_ptr<Expression> &expr, const vector<LogicalType> &sql_types) {
	if (!expr) {
		return;
	}
	if (expr->type != ExpressionType::BOUND_COLUMN_REF) {
		return;
	}
	auto &bound_colref = expr->Cast<BoundColumnRefExpression>();
	bound_colref.return_type = sql_types[bound_colref.binding.column_index];
}

void Binder::BindModifierTypes(BoundQueryNode &result, const vector<LogicalType> &sql_types, idx_t projection_index) {
	for (auto &bound_mod : result.modifiers) {
		switch (bound_mod->type) {
		case ResultModifierType::DISTINCT_MODIFIER: {
			auto &distinct = (BoundDistinctModifier &)*bound_mod;
			D_ASSERT(!distinct.target_distincts.empty());
			// set types of distinct targets
			for (auto &expr : distinct.target_distincts) {
				D_ASSERT(expr->type == ExpressionType::BOUND_COLUMN_REF);
				auto &bound_colref = expr->Cast<BoundColumnRefExpression>();
				if (bound_colref.binding.column_index == DConstants::INVALID_INDEX) {
					throw BinderException("Ambiguous name in DISTINCT ON!");
				}
				D_ASSERT(bound_colref.binding.column_index < sql_types.size());
				bound_colref.return_type = sql_types[bound_colref.binding.column_index];
			}
			for (auto &target_distinct : distinct.target_distincts) {
				auto &bound_colref = target_distinct->Cast<BoundColumnRefExpression>();
				const auto &sql_type = sql_types[bound_colref.binding.column_index];
				if (sql_type.id() == LogicalTypeId::VARCHAR) {
					target_distinct = ExpressionBinder::PushCollation(context, std::move(target_distinct),
					                                                  StringType::GetCollation(sql_type), true);
				}
			}
			break;
		}
		case ResultModifierType::LIMIT_MODIFIER: {
			auto &limit = (BoundLimitModifier &)*bound_mod;
			AssignReturnType(limit.limit, sql_types);
			AssignReturnType(limit.offset, sql_types);
			break;
		}
		case ResultModifierType::LIMIT_PERCENT_MODIFIER: {
			auto &limit = (BoundLimitPercentModifier &)*bound_mod;
			AssignReturnType(limit.limit, sql_types);
			AssignReturnType(limit.offset, sql_types);
			break;
		}
		case ResultModifierType::ORDER_MODIFIER: {
			auto &order = (BoundOrderModifier &)*bound_mod;
			for (auto &order_node : order.orders) {
				auto &expr = order_node.expression;
				D_ASSERT(expr->type == ExpressionType::BOUND_COLUMN_REF);
				auto &bound_colref = expr->Cast<BoundColumnRefExpression>();
				if (bound_colref.binding.column_index == DConstants::INVALID_INDEX) {
					throw BinderException("Ambiguous name in ORDER BY!");
				}
				D_ASSERT(bound_colref.binding.column_index < sql_types.size());
				const auto &sql_type = sql_types[bound_colref.binding.column_index];
				bound_colref.return_type = sql_types[bound_colref.binding.column_index];
				if (sql_type.id() == LogicalTypeId::VARCHAR) {
					order_node.expression = ExpressionBinder::PushCollation(context, std::move(order_node.expression),
					                                                        StringType::GetCollation(sql_type));
				}
			}
			break;
		}
		default:
			break;
		}
	}
}

unique_ptr<BoundQueryNode> Binder::BindNode(SelectNode &statement) {
	D_ASSERT(statement.from_table);
	// first bind the FROM table statement
	auto from = std::move(statement.from_table);
	auto from_table = Bind(*from);
	return BindSelectNode(statement, std::move(from_table));
}

void Binder::BindWhereStarExpression(unique_ptr<ParsedExpression> &expr) {
	// expand any expressions in the upper AND recursively
	if (expr->type == ExpressionType::CONJUNCTION_AND) {
		auto &conj = expr->Cast<ConjunctionExpression>();
		for (auto &child : conj.children) {
			BindWhereStarExpression(child);
		}
		return;
	}
	if (expr->type == ExpressionType::STAR) {
		auto &star = expr->Cast<StarExpression>();
		if (!star.columns) {
			throw ParserException("STAR expression is not allowed in the WHERE clause. Use COLUMNS(*) instead.");
		}
	}
	// expand the stars for this expression
	vector<unique_ptr<ParsedExpression>> new_conditions;
	ExpandStarExpression(std::move(expr), new_conditions);

	// set up an AND conjunction between the expanded conditions
	expr = std::move(new_conditions[0]);
	for (idx_t i = 1; i < new_conditions.size(); i++) {
		auto and_conj = make_uniq<ConjunctionExpression>(ExpressionType::CONJUNCTION_AND, std::move(expr),
		                                                 std::move(new_conditions[i]));
		expr = std::move(and_conj);
	}
}

unique_ptr<BoundQueryNode> Binder::BindSelectNode(SelectNode &statement, unique_ptr<BoundTableRef> from_table) {
	D_ASSERT(from_table);
	D_ASSERT(!statement.from_table);
	auto result = make_uniq<BoundSelectNode>();
	result->projection_index = GenerateTableIndex();
	result->group_index = GenerateTableIndex();
	result->aggregate_index = GenerateTableIndex();
	result->groupings_index = GenerateTableIndex();
	result->window_index = GenerateTableIndex();
	result->prune_index = GenerateTableIndex();

	result->from_table = std::move(from_table);
	// bind the sample clause
	if (statement.sample) {
		result->sample_options = std::move(statement.sample);
	}

	// visit the select list and expand any "*" statements
	vector<unique_ptr<ParsedExpression>> new_select_list;
	ExpandStarExpressions(statement.select_list, new_select_list);

	if (new_select_list.empty()) {
		throw BinderException("SELECT list is empty after resolving * expressions!");
	}
	statement.select_list = std::move(new_select_list);

	// create a mapping of (alias -> index) and a mapping of (Expression -> index) for the SELECT list
	case_insensitive_map_t<idx_t> alias_map;
	parsed_expression_map_t<idx_t> projection_map;
	for (idx_t i = 0; i < statement.select_list.size(); i++) {
		auto &expr = statement.select_list[i];
		result->names.push_back(expr->GetName());
		ExpressionBinder::QualifyColumnNames(*this, expr);
		if (!expr->alias.empty()) {
			alias_map[expr->alias] = i;
			result->names[i] = expr->alias;
		}
		projection_map[*expr] = i;
		result->original_expressions.push_back(expr->Copy());
	}
	result->column_count = statement.select_list.size();

	// first visit the WHERE clause
	// the WHERE clause happens before the GROUP BY, PROJECTION or HAVING clauses
	if (statement.where_clause) {
		// bind any star expressions in the WHERE clause
		BindWhereStarExpression(statement.where_clause);

		ColumnAliasBinder alias_binder(*result, alias_map);
		WhereBinder where_binder(*this, context, &alias_binder);
		unique_ptr<ParsedExpression> condition = std::move(statement.where_clause);
		result->where_clause = where_binder.Bind(condition);
	}

	// now bind all the result modifiers; including DISTINCT and ORDER BY targets
	OrderBinder order_binder({this}, result->projection_index, statement, alias_map, projection_map);
	BindModifiers(order_binder, statement, *result);

	vector<unique_ptr<ParsedExpression>> unbound_groups;
	BoundGroupInformation info;
	auto &group_expressions = statement.groups.group_expressions;
	if (!group_expressions.empty()) {
		// the statement has a GROUP BY clause, bind it
		unbound_groups.resize(group_expressions.size());
		GroupBinder group_binder(*this, context, statement, result->group_index, alias_map, info.alias_map);
		for (idx_t i = 0; i < group_expressions.size(); i++) {

			// we keep a copy of the unbound expression;
			// we keep the unbound copy around to check for group references in the SELECT and HAVING clause
			// the reason we want the unbound copy is because we want to figure out whether an expression
			// is a group reference BEFORE binding in the SELECT/HAVING binder
			group_binder.unbound_expression = group_expressions[i]->Copy();
			group_binder.bind_index = i;

			// bind the groups
			LogicalType group_type;
			auto bound_expr = group_binder.Bind(group_expressions[i], &group_type);
			D_ASSERT(bound_expr->return_type.id() != LogicalTypeId::INVALID);

			// push a potential collation, if necessary
			bound_expr = ExpressionBinder::PushCollation(context, std::move(bound_expr),
			                                             StringType::GetCollation(group_type), true);
			result->groups.group_expressions.push_back(std::move(bound_expr));

			// in the unbound expression we DO bind the table names of any ColumnRefs
			// we do this to make sure that "table.a" and "a" are treated the same
			// if we wouldn't do this then (SELECT test.a FROM test GROUP BY a) would not work because "test.a" <> "a"
			// hence we convert "a" -> "test.a" in the unbound expression
			unbound_groups[i] = std::move(group_binder.unbound_expression);
			ExpressionBinder::QualifyColumnNames(*this, unbound_groups[i]);
			info.map[*unbound_groups[i]] = i;
		}
	}
	result->groups.grouping_sets = std::move(statement.groups.grouping_sets);

	// bind the HAVING clause, if any
	if (statement.having) {
		HavingBinder having_binder(*this, context, *result, info, alias_map, statement.aggregate_handling);
		ExpressionBinder::QualifyColumnNames(*this, statement.having);
		result->having = having_binder.Bind(statement.having);
	}

	// bind the QUALIFY clause, if any
	if (statement.qualify) {
		if (statement.aggregate_handling == AggregateHandling::FORCE_AGGREGATES) {
			throw BinderException("Combining QUALIFY with GROUP BY ALL is not supported yet");
		}
		QualifyBinder qualify_binder(*this, context, *result, info, alias_map);
		ExpressionBinder::QualifyColumnNames(*this, statement.qualify);
		result->qualify = qualify_binder.Bind(statement.qualify);
		if (qualify_binder.HasBoundColumns() && qualify_binder.BoundAggregates()) {
			throw BinderException("Cannot mix aggregates with non-aggregated columns!");
		}
	}

	// after that, we bind to the SELECT list
	SelectBinder select_binder(*this, context, *result, info, alias_map);
	vector<LogicalType> internal_sql_types;
	vector<idx_t> group_by_all_indexes;
	vector<string> new_names;
	for (idx_t i = 0; i < statement.select_list.size(); i++) {
		bool is_window = statement.select_list[i]->IsWindow();
		idx_t unnest_count = result->unnests.size();
		LogicalType result_type;
		auto expr = select_binder.Bind(statement.select_list[i], &result_type, true);
		bool is_original_column = i < result->column_count;
		bool can_group_by_all =
		    statement.aggregate_handling == AggregateHandling::FORCE_AGGREGATES && is_original_column;
		if (select_binder.HasExpandedExpressions()) {
			if (!is_original_column) {
				throw InternalException("Only original columns can have expanded expressions");
			}
			if (statement.aggregate_handling == AggregateHandling::FORCE_AGGREGATES) {
				throw BinderException("UNNEST of struct cannot be combined with GROUP BY ALL");
			}
			auto &struct_expressions = select_binder.ExpandedExpressions();
			D_ASSERT(!struct_expressions.empty());
			for (auto &struct_expr : struct_expressions) {
				new_names.push_back(struct_expr->GetName());
				result->types.push_back(struct_expr->return_type);
				result->select_list.push_back(std::move(struct_expr));
			}
			struct_expressions.clear();
			continue;
		}
		if (can_group_by_all && select_binder.HasBoundColumns()) {
			if (select_binder.BoundAggregates()) {
				throw BinderException("Cannot mix aggregates with non-aggregated columns!");
			}
			if (is_window) {
				throw BinderException("Cannot group on a window clause");
			}
			if (result->unnests.size() > unnest_count) {
				throw BinderException("Cannot group on an UNNEST or UNLIST clause");
			}
			// we are forcing aggregates, and the node has columns bound
			// this entry becomes a group
			group_by_all_indexes.push_back(i);
		}
		result->select_list.push_back(std::move(expr));
		if (is_original_column) {
			new_names.push_back(std::move(result->names[i]));
			result->types.push_back(result_type);
		}
		internal_sql_types.push_back(result_type);
		if (can_group_by_all) {
			select_binder.ResetBindings();
		}
	}
	// push the GROUP BY ALL expressions into the group set
	for (auto &group_by_all_index : group_by_all_indexes) {
		auto &expr = result->select_list[group_by_all_index];
		auto group_ref = make_uniq<BoundColumnRefExpression>(
		    expr->return_type, ColumnBinding(result->group_index, result->groups.group_expressions.size()));
		result->groups.group_expressions.push_back(std::move(expr));
		expr = std::move(group_ref);
	}
	result->column_count = new_names.size();
	result->names = std::move(new_names);
	result->need_prune = result->select_list.size() > result->column_count;

	// in the normal select binder, we bind columns as if there is no aggregation
	// i.e. in the query [SELECT i, SUM(i) FROM integers;] the "i" will be bound as a normal column
	// since we have an aggregation, we need to either (1) throw an error, or (2) wrap the column in a FIRST() aggregate
	// we choose the former one [CONTROVERSIAL: this is the PostgreSQL behavior]
	if (!result->groups.group_expressions.empty() || !result->aggregates.empty() || statement.having ||
	    !result->groups.grouping_sets.empty()) {
		if (statement.aggregate_handling == AggregateHandling::NO_AGGREGATES_ALLOWED) {
			throw BinderException("Aggregates cannot be present in a Project relation!");
		} else if (select_binder.HasBoundColumns()) {
			auto &bound_columns = select_binder.GetBoundColumns();
			string error;
			error = "column \"%s\" must appear in the GROUP BY clause or must be part of an aggregate function.";
			if (statement.aggregate_handling == AggregateHandling::FORCE_AGGREGATES) {
				error += "\nGROUP BY ALL will only group entries in the SELECT list. Add it to the SELECT list or "
				         "GROUP BY this entry explicitly.";
			} else {
				error += "\nEither add it to the GROUP BY list, or use \"ANY_VALUE(%s)\" if the exact value of \"%s\" "
				         "is not important.";
			}
			throw BinderException(FormatError(bound_columns[0].query_location, error, bound_columns[0].name,
			                                  bound_columns[0].name, bound_columns[0].name));
		}
	}

	// QUALIFY clause requires at least one window function to be specified in at least one of the SELECT column list or
	// the filter predicate of the QUALIFY clause
	if (statement.qualify && result->windows.empty()) {
		throw BinderException("at least one window function must appear in the SELECT column or QUALIFY clause");
	}

	// now that the SELECT list is bound, we set the types of DISTINCT/ORDER BY expressions
	BindModifierTypes(*result, internal_sql_types, result->projection_index);
	return std::move(result);
}

} // namespace duckdb












namespace duckdb {

static void GatherAliases(BoundQueryNode &node, case_insensitive_map_t<idx_t> &aliases,
                          parsed_expression_map_t<idx_t> &expressions, const vector<idx_t> &reorder_idx) {
	if (node.type == QueryNodeType::SET_OPERATION_NODE) {
		// setop, recurse
		auto &setop = node.Cast<BoundSetOperationNode>();

		// create new reorder index
		if (setop.setop_type == SetOperationType::UNION_BY_NAME) {
			vector<idx_t> new_left_reorder_idx(setop.left_reorder_idx.size());
			vector<idx_t> new_right_reorder_idx(setop.right_reorder_idx.size());
			for (idx_t i = 0; i < setop.left_reorder_idx.size(); ++i) {
				new_left_reorder_idx[i] = reorder_idx[setop.left_reorder_idx[i]];
			}

			for (idx_t i = 0; i < setop.right_reorder_idx.size(); ++i) {
				new_right_reorder_idx[i] = reorder_idx[setop.right_reorder_idx[i]];
			}

			// use new reorder index
			GatherAliases(*setop.left, aliases, expressions, new_left_reorder_idx);
			GatherAliases(*setop.right, aliases, expressions, new_right_reorder_idx);
			return;
		}

		GatherAliases(*setop.left, aliases, expressions, reorder_idx);
		GatherAliases(*setop.right, aliases, expressions, reorder_idx);
	} else {
		// query node
		D_ASSERT(node.type == QueryNodeType::SELECT_NODE);
		auto &select = node.Cast<BoundSelectNode>();
		// fill the alias lists
		for (idx_t i = 0; i < select.names.size(); i++) {
			auto &name = select.names[i];
			auto &expr = select.original_expressions[i];
			// first check if the alias is already in there
			auto entry = aliases.find(name);

			idx_t index = reorder_idx[i];

			if (entry != aliases.end()) {
				// the alias already exists
				// check if there is a conflict

				if (entry->second != index) {
					// there is a conflict
					// we place "-1" in the aliases map at this location
					// "-1" signifies that there is an ambiguous reference
					aliases[name] = DConstants::INVALID_INDEX;
				}
			} else {
				// the alias is not in there yet, just assign it
				aliases[name] = index;
			}
			// now check if the node is already in the set of expressions
			auto expr_entry = expressions.find(*expr);
			if (expr_entry != expressions.end()) {
				// the node is in there
				// repeat the same as with the alias: if there is an ambiguity we insert "-1"
				if (expr_entry->second != index) {
					expressions[*expr] = DConstants::INVALID_INDEX;
				}
			} else {
				// not in there yet, just place it in there
				expressions[*expr] = index;
			}
		}
	}
}

static void BuildUnionByNameInfo(BoundSetOperationNode &result, bool can_contain_nulls) {
	D_ASSERT(result.setop_type == SetOperationType::UNION_BY_NAME);
	case_insensitive_map_t<idx_t> left_names_map;
	case_insensitive_map_t<idx_t> right_names_map;

	BoundQueryNode *left_node = result.left.get();
	BoundQueryNode *right_node = result.right.get();

	// Build a name_map to use to check if a name exists
	// We throw a binder exception if two same name in the SELECT list
	for (idx_t i = 0; i < left_node->names.size(); ++i) {
		if (left_names_map.find(left_node->names[i]) != left_names_map.end()) {
			throw BinderException("UNION(ALL) BY NAME operation doesn't support same name in SELECT list");
		}
		left_names_map[left_node->names[i]] = i;
	}

	for (idx_t i = 0; i < right_node->names.size(); ++i) {
		if (right_names_map.find(right_node->names[i]) != right_names_map.end()) {
			throw BinderException("UNION(ALL) BY NAME operation doesn't support same name in SELECT list");
		}
		if (left_names_map.find(right_node->names[i]) == left_names_map.end()) {
			result.names.push_back(right_node->names[i]);
		}
		right_names_map[right_node->names[i]] = i;
	}

	idx_t new_size = result.names.size();
	bool need_reorder = false;
	vector<idx_t> left_reorder_idx(left_node->names.size());
	vector<idx_t> right_reorder_idx(right_node->names.size());

	// Construct return type and reorder_idxs
	// reorder_idxs is used to gather correct alias_map
	// and expression_map in GatherAlias(...)
	for (idx_t i = 0; i < new_size; ++i) {
		auto left_index = left_names_map.find(result.names[i]);
		auto right_index = right_names_map.find(result.names[i]);
		bool left_exist = left_index != left_names_map.end();
		bool right_exist = right_index != right_names_map.end();
		LogicalType result_type;
		if (left_exist && right_exist) {
			result_type = LogicalType::MaxLogicalType(left_node->types[left_index->second],
			                                          right_node->types[right_index->second]);
			if (left_index->second != i || right_index->second != i) {
				need_reorder = true;
			}
			left_reorder_idx[left_index->second] = i;
			right_reorder_idx[right_index->second] = i;
		} else if (left_exist) {
			result_type = left_node->types[left_index->second];
			need_reorder = true;
			left_reorder_idx[left_index->second] = i;
		} else {
			D_ASSERT(right_exist);
			result_type = right_node->types[right_index->second];
			need_reorder = true;
			right_reorder_idx[right_index->second] = i;
		}

		if (!can_contain_nulls) {
			if (ExpressionBinder::ContainsNullType(result_type)) {
				result_type = ExpressionBinder::ExchangeNullType(result_type);
			}
		}

		result.types.push_back(result_type);
	}

	result.left_reorder_idx = std::move(left_reorder_idx);
	result.right_reorder_idx = std::move(right_reorder_idx);

	// If reorder is required, collect reorder expressions for push projection
	// into the two child nodes of union node
	if (need_reorder) {
		for (idx_t i = 0; i < new_size; ++i) {
			auto left_index = left_names_map.find(result.names[i]);
			auto right_index = right_names_map.find(result.names[i]);
			bool left_exist = left_index != left_names_map.end();
			bool right_exist = right_index != right_names_map.end();
			unique_ptr<Expression> left_reorder_expr;
			unique_ptr<Expression> right_reorder_expr;
			if (left_exist && right_exist) {
				left_reorder_expr = make_uniq<BoundColumnRefExpression>(
				    left_node->types[left_index->second], ColumnBinding(left_node->GetRootIndex(), left_index->second));
				right_reorder_expr =
				    make_uniq<BoundColumnRefExpression>(right_node->types[right_index->second],
				                                        ColumnBinding(right_node->GetRootIndex(), right_index->second));
			} else if (left_exist) {
				left_reorder_expr = make_uniq<BoundColumnRefExpression>(
				    left_node->types[left_index->second], ColumnBinding(left_node->GetRootIndex(), left_index->second));
				// create null value here
				right_reorder_expr = make_uniq<BoundConstantExpression>(Value(result.types[i]));
			} else {
				D_ASSERT(right_exist);
				left_reorder_expr = make_uniq<BoundConstantExpression>(Value(result.types[i]));
				right_reorder_expr =
				    make_uniq<BoundColumnRefExpression>(right_node->types[right_index->second],
				                                        ColumnBinding(right_node->GetRootIndex(), right_index->second));
			}
			result.left_reorder_exprs.push_back(std::move(left_reorder_expr));
			result.right_reorder_exprs.push_back(std::move(right_reorder_expr));
		}
	}
}

unique_ptr<BoundQueryNode> Binder::BindNode(SetOperationNode &statement) {
	auto result = make_uniq<BoundSetOperationNode>();
	result->setop_type = statement.setop_type;

	// first recursively visit the set operations
	// both the left and right sides have an independent BindContext and Binder
	D_ASSERT(statement.left);
	D_ASSERT(statement.right);

	result->setop_index = GenerateTableIndex();

	result->left_binder = Binder::CreateBinder(context, this);
	result->left_binder->can_contain_nulls = true;
	result->left = result->left_binder->BindNode(*statement.left);
	result->right_binder = Binder::CreateBinder(context, this);
	result->right_binder->can_contain_nulls = true;
	result->right = result->right_binder->BindNode(*statement.right);

	result->names = result->left->names;

	// move the correlated expressions from the child binders to this binder
	MoveCorrelatedExpressions(*result->left_binder);
	MoveCorrelatedExpressions(*result->right_binder);

	// now both sides have been bound we can resolve types
	if (result->setop_type != SetOperationType::UNION_BY_NAME &&
	    result->left->types.size() != result->right->types.size()) {
		throw BinderException("Set operations can only apply to expressions with the "
		                      "same number of result columns");
	}

	if (result->setop_type == SetOperationType::UNION_BY_NAME) {
		BuildUnionByNameInfo(*result, can_contain_nulls);

	} else {
		// figure out the types of the setop result by picking the max of both
		for (idx_t i = 0; i < result->left->types.size(); i++) {
			auto result_type = LogicalType::MaxLogicalType(result->left->types[i], result->right->types[i]);
			if (!can_contain_nulls) {
				if (ExpressionBinder::ContainsNullType(result_type)) {
					result_type = ExpressionBinder::ExchangeNullType(result_type);
				}
			}
			result->types.push_back(result_type);
		}
	}

	if (!statement.modifiers.empty()) {
		// handle the ORDER BY/DISTINCT clauses

		// we recursively visit the children of this node to extract aliases and expressions that can be referenced
		// in the ORDER BY
		case_insensitive_map_t<idx_t> alias_map;
		parsed_expression_map_t<idx_t> expression_map;

		if (result->setop_type == SetOperationType::UNION_BY_NAME) {
			GatherAliases(*result->left, alias_map, expression_map, result->left_reorder_idx);
			GatherAliases(*result->right, alias_map, expression_map, result->right_reorder_idx);
		} else {
			vector<idx_t> reorder_idx;
			for (idx_t i = 0; i < result->names.size(); i++) {
				reorder_idx.push_back(i);
			}
			GatherAliases(*result, alias_map, expression_map, reorder_idx);
		}
		// now we perform the actual resolution of the ORDER BY/DISTINCT expressions
		OrderBinder order_binder({result->left_binder.get(), result->right_binder.get()}, result->setop_index,
		                         alias_map, expression_map, result->names.size());
		BindModifiers(order_binder, statement, *result);
	}

	// finally bind the types of the ORDER/DISTINCT clause expressions
	BindModifierTypes(*result, result->types, result->setop_index);
	return std::move(result);
}

} // namespace duckdb
















namespace duckdb {

unique_ptr<QueryNode> Binder::BindTableMacro(FunctionExpression &function, TableMacroCatalogEntry &macro_func,
                                             idx_t depth) {

	auto &macro_def = macro_func.function->Cast<TableMacroFunction>();
	auto node = macro_def.query_node->Copy();

	// auto &macro_def = *macro_func->function;

	// validate the arguments and separate positional and default arguments
	vector<unique_ptr<ParsedExpression>> positionals;
	unordered_map<string, unique_ptr<ParsedExpression>> defaults;
	string error =
	    MacroFunction::ValidateArguments(*macro_func.function, macro_func.name, function, positionals, defaults);
	if (!error.empty()) {
		// cannot use error below as binder rnot in scope
		// return BindResult(binder. FormatError(*expr->get(), error));
		throw BinderException(FormatError(function, error));
	}

	// create a MacroBinding to bind this macro's parameters to its arguments
	vector<LogicalType> types;
	vector<string> names;
	// positional parameters
	for (idx_t i = 0; i < macro_def.parameters.size(); i++) {
		types.emplace_back(LogicalType::SQLNULL);
		auto &param = macro_def.parameters[i]->Cast<ColumnRefExpression>();
		names.push_back(param.GetColumnName());
	}
	// default parameters
	for (auto it = macro_def.default_parameters.begin(); it != macro_def.default_parameters.end(); it++) {
		types.emplace_back(LogicalType::SQLNULL);
		names.push_back(it->first);
		// now push the defaults into the positionals
		positionals.push_back(std::move(defaults[it->first]));
	}
	auto new_macro_binding = make_uniq<DummyBinding>(types, names, macro_func.name);
	new_macro_binding->arguments = &positionals;

	// We need an ExpressionBinder so that we can call ExpressionBinder::ReplaceMacroParametersRecursive()
	auto eb = ExpressionBinder(*this, this->context);

	eb.macro_binding = new_macro_binding.get();

	/* Does it all goes throu every expression in a selectstmt  */
	ParsedExpressionIterator::EnumerateQueryNodeChildren(
	    *node, [&](unique_ptr<ParsedExpression> &child) { eb.ReplaceMacroParametersRecursive(child); });

	return node;
}

} // namespace duckdb








namespace duckdb {

unique_ptr<LogicalOperator> Binder::VisitQueryNode(BoundQueryNode &node, unique_ptr<LogicalOperator> root) {
	D_ASSERT(root);
	for (auto &mod : node.modifiers) {
		switch (mod->type) {
		case ResultModifierType::DISTINCT_MODIFIER: {
			auto &bound = (BoundDistinctModifier &)*mod;
			auto distinct = make_uniq<LogicalDistinct>(std::move(bound.target_distincts), bound.distinct_type);
			distinct->AddChild(std::move(root));
			root = std::move(distinct);
			break;
		}
		case ResultModifierType::ORDER_MODIFIER: {
			auto &bound = (BoundOrderModifier &)*mod;
			if (root->type == LogicalOperatorType::LOGICAL_DISTINCT) {
				auto &distinct = root->Cast<LogicalDistinct>();
				if (distinct.distinct_type == DistinctType::DISTINCT_ON) {
					auto order_by = make_uniq<BoundOrderModifier>();
					for (auto &order_node : bound.orders) {
						order_by->orders.push_back(order_node.Copy());
					}
					distinct.order_by = std::move(order_by);
				}
			}
			auto order = make_uniq<LogicalOrder>(std::move(bound.orders));
			order->AddChild(std::move(root));
			root = std::move(order);
			break;
		}
		case ResultModifierType::LIMIT_MODIFIER: {
			auto &bound = (BoundLimitModifier &)*mod;
			auto limit = make_uniq<LogicalLimit>(bound.limit_val, bound.offset_val, std::move(bound.limit),
			                                     std::move(bound.offset));
			limit->AddChild(std::move(root));
			root = std::move(limit);
			break;
		}
		case ResultModifierType::LIMIT_PERCENT_MODIFIER: {
			auto &bound = (BoundLimitPercentModifier &)*mod;
			auto limit = make_uniq<LogicalLimitPercent>(bound.limit_percent, bound.offset_val, std::move(bound.limit),
			                                            std::move(bound.offset));
			limit->AddChild(std::move(root));
			root = std::move(limit);
			break;
		}
		default:
			throw BinderException("Unimplemented modifier type!");
		}
	}
	return root;
}

} // namespace duckdb








namespace duckdb {

unique_ptr<LogicalOperator> Binder::CreatePlan(BoundRecursiveCTENode &node) {
	// Generate the logical plan for the left and right sides of the set operation
	node.left_binder->plan_subquery = plan_subquery;
	node.right_binder->plan_subquery = plan_subquery;

	auto left_node = node.left_binder->CreatePlan(*node.left);
	auto right_node = node.right_binder->CreatePlan(*node.right);

	// check if there are any unplanned subqueries left in either child
	has_unplanned_subqueries =
	    node.left_binder->has_unplanned_subqueries || node.right_binder->has_unplanned_subqueries;

	// for both the left and right sides, cast them to the same types
	left_node = CastLogicalOperatorToTypes(node.left->types, node.types, std::move(left_node));
	right_node = CastLogicalOperatorToTypes(node.right->types, node.types, std::move(right_node));

	if (!node.right_binder->bind_context.cte_references[node.ctename] ||
	    *node.right_binder->bind_context.cte_references[node.ctename] == 0) {
		auto root = make_uniq<LogicalSetOperation>(node.setop_index, node.types.size(), std::move(left_node),
		                                           std::move(right_node), LogicalOperatorType::LOGICAL_UNION);
		return VisitQueryNode(node, std::move(root));
	}
	auto root = make_uniq<LogicalRecursiveCTE>(node.setop_index, node.types.size(), node.union_all,
	                                           std::move(left_node), std::move(right_node));

	return VisitQueryNode(node, std::move(root));
}

} // namespace duckdb








namespace duckdb {

unique_ptr<LogicalOperator> Binder::PlanFilter(unique_ptr<Expression> condition, unique_ptr<LogicalOperator> root) {
	PlanSubqueries(condition, root);
	auto filter = make_uniq<LogicalFilter>(std::move(condition));
	filter->AddChild(std::move(root));
	return std::move(filter);
}

unique_ptr<LogicalOperator> Binder::CreatePlan(BoundSelectNode &statement) {
	unique_ptr<LogicalOperator> root;
	D_ASSERT(statement.from_table);
	root = CreatePlan(*statement.from_table);
	D_ASSERT(root);

	// plan the sample clause
	if (statement.sample_options) {
		root = make_uniq<LogicalSample>(std::move(statement.sample_options), std::move(root));
	}

	if (statement.where_clause) {
		root = PlanFilter(std::move(statement.where_clause), std::move(root));
	}

	if (!statement.aggregates.empty() || !statement.groups.group_expressions.empty()) {
		if (!statement.groups.group_expressions.empty()) {
			// visit the groups
			for (auto &group : statement.groups.group_expressions) {
				PlanSubqueries(group, root);
			}
		}
		// now visit all aggregate expressions
		for (auto &expr : statement.aggregates) {
			PlanSubqueries(expr, root);
		}
		// finally create the aggregate node with the group_index and aggregate_index as obtained from the binder
		auto aggregate = make_uniq<LogicalAggregate>(statement.group_index, statement.aggregate_index,
		                                             std::move(statement.aggregates));
		aggregate->groups = std::move(statement.groups.group_expressions);
		aggregate->groupings_index = statement.groupings_index;
		aggregate->grouping_sets = std::move(statement.groups.grouping_sets);
		aggregate->grouping_functions = std::move(statement.grouping_functions);

		aggregate->AddChild(std::move(root));
		root = std::move(aggregate);
	} else if (!statement.groups.grouping_sets.empty()) {
		// edge case: we have grouping sets but no groups or aggregates
		// this can only happen if we have e.g. select 1 from tbl group by ();
		// just output a dummy scan
		root = make_uniq_base<LogicalOperator, LogicalDummyScan>(statement.group_index);
	}

	if (statement.having) {
		PlanSubqueries(statement.having, root);
		auto having = make_uniq<LogicalFilter>(std::move(statement.having));

		having->AddChild(std::move(root));
		root = std::move(having);
	}

	if (!statement.windows.empty()) {
		auto win = make_uniq<LogicalWindow>(statement.window_index);
		win->expressions = std::move(statement.windows);
		// visit the window expressions
		for (auto &expr : win->expressions) {
			PlanSubqueries(expr, root);
		}
		D_ASSERT(!win->expressions.empty());
		win->AddChild(std::move(root));
		root = std::move(win);
	}

	if (statement.qualify) {
		PlanSubqueries(statement.qualify, root);
		auto qualify = make_uniq<LogicalFilter>(std::move(statement.qualify));

		qualify->AddChild(std::move(root));
		root = std::move(qualify);
	}

	for (idx_t i = statement.unnests.size(); i > 0; i--) {
		auto unnest_level = i - 1;
		auto entry = statement.unnests.find(unnest_level);
		if (entry == statement.unnests.end()) {
			throw InternalException("unnests specified at level %d but none were found", unnest_level);
		}
		auto &unnest_node = entry->second;
		auto unnest = make_uniq<LogicalUnnest>(unnest_node.index);
		unnest->expressions = std::move(unnest_node.expressions);
		// visit the unnest expressions
		for (auto &expr : unnest->expressions) {
			PlanSubqueries(expr, root);
		}
		D_ASSERT(!unnest->expressions.empty());
		unnest->AddChild(std::move(root));
		root = std::move(unnest);
	}

	for (auto &expr : statement.select_list) {
		PlanSubqueries(expr, root);
	}

	auto proj = make_uniq<LogicalProjection>(statement.projection_index, std::move(statement.select_list));
	auto &projection = *proj;
	proj->AddChild(std::move(root));
	root = std::move(proj);

	// finish the plan by handling the elements of the QueryNode
	root = VisitQueryNode(statement, std::move(root));

	// add a prune node if necessary
	if (statement.need_prune) {
		D_ASSERT(root);
		vector<unique_ptr<Expression>> prune_expressions;
		for (idx_t i = 0; i < statement.column_count; i++) {
			prune_expressions.push_back(make_uniq<BoundColumnRefExpression>(
			    projection.expressions[i]->return_type, ColumnBinding(statement.projection_index, i)));
		}
		auto prune = make_uniq<LogicalProjection>(statement.prune_index, std::move(prune_expressions));
		prune->AddChild(std::move(root));
		root = std::move(prune);
	}
	return root;
}

} // namespace duckdb







namespace duckdb {

// Optionally push a PROJECTION operator
unique_ptr<LogicalOperator> Binder::CastLogicalOperatorToTypes(vector<LogicalType> &source_types,
                                                               vector<LogicalType> &target_types,
                                                               unique_ptr<LogicalOperator> op) {
	D_ASSERT(op);
	// first check if we even need to cast
	D_ASSERT(source_types.size() == target_types.size());
	if (source_types == target_types) {
		// source and target types are equal: don't need to cast
		return op;
	}
	// otherwise add casts
	auto node = op.get();
	if (node->type == LogicalOperatorType::LOGICAL_PROJECTION) {
		// "node" is a projection; we can just do the casts in there
		D_ASSERT(node->expressions.size() == source_types.size());
		// add the casts to the selection list
		for (idx_t i = 0; i < target_types.size(); i++) {
			if (source_types[i] != target_types[i]) {
				// differing types, have to add a cast
				string alias = node->expressions[i]->alias;
				node->expressions[i] =
				    BoundCastExpression::AddCastToType(context, std::move(node->expressions[i]), target_types[i]);
				node->expressions[i]->alias = alias;
			}
		}
		return op;
	} else {
		// found a non-projection operator
		// push a new projection containing the casts

		// fetch the set of column bindings
		auto setop_columns = op->GetColumnBindings();
		D_ASSERT(setop_columns.size() == source_types.size());

		// now generate the expression list
		vector<unique_ptr<Expression>> select_list;
		for (idx_t i = 0; i < target_types.size(); i++) {
			unique_ptr<Expression> result = make_uniq<BoundColumnRefExpression>(source_types[i], setop_columns[i]);
			if (source_types[i] != target_types[i]) {
				// add a cast only if the source and target types are not equivalent
				result = BoundCastExpression::AddCastToType(context, std::move(result), target_types[i]);
			}
			select_list.push_back(std::move(result));
		}
		auto projection = make_uniq<LogicalProjection>(GenerateTableIndex(), std::move(select_list));
		projection->children.push_back(std::move(op));
		return std::move(projection);
	}
}

unique_ptr<LogicalOperator> Binder::CreatePlan(BoundSetOperationNode &node) {
	// Generate the logical plan for the left and right sides of the set operation
	node.left_binder->plan_subquery = plan_subquery;
	node.right_binder->plan_subquery = plan_subquery;

	auto left_node = node.left_binder->CreatePlan(*node.left);
	auto right_node = node.right_binder->CreatePlan(*node.right);

	// Add a new projection to child node
	D_ASSERT(node.left_reorder_exprs.size() == node.right_reorder_exprs.size());
	if (!node.left_reorder_exprs.empty()) {
		D_ASSERT(node.setop_type == SetOperationType::UNION_BY_NAME);
		vector<LogicalType> left_types;
		vector<LogicalType> right_types;
		// We are going to add a new projection operator, so collect the type
		// of reorder exprs in order to call CastLogicalOperatorToTypes()
		for (idx_t i = 0; i < node.left_reorder_exprs.size(); ++i) {
			left_types.push_back(node.left_reorder_exprs[i]->return_type);
			right_types.push_back(node.right_reorder_exprs[i]->return_type);
		}

		auto left_projection = make_uniq<LogicalProjection>(GenerateTableIndex(), std::move(node.left_reorder_exprs));
		left_projection->children.push_back(std::move(left_node));
		left_node = std::move(left_projection);

		auto right_projection = make_uniq<LogicalProjection>(GenerateTableIndex(), std::move(node.right_reorder_exprs));
		right_projection->children.push_back(std::move(right_node));
		right_node = std::move(right_projection);

		left_node = CastLogicalOperatorToTypes(left_types, node.types, std::move(left_node));
		right_node = CastLogicalOperatorToTypes(right_types, node.types, std::move(right_node));
	} else {
		left_node = CastLogicalOperatorToTypes(node.left->types, node.types, std::move(left_node));
		right_node = CastLogicalOperatorToTypes(node.right->types, node.types, std::move(right_node));
	}

	// check if there are any unplanned subqueries left in either child
	has_unplanned_subqueries =
	    node.left_binder->has_unplanned_subqueries || node.right_binder->has_unplanned_subqueries;

	// create actual logical ops for setops
	LogicalOperatorType logical_type;
	switch (node.setop_type) {
	case SetOperationType::UNION:
	case SetOperationType::UNION_BY_NAME:
		logical_type = LogicalOperatorType::LOGICAL_UNION;
		break;
	case SetOperationType::EXCEPT:
		logical_type = LogicalOperatorType::LOGICAL_EXCEPT;
		break;
	default:
		D_ASSERT(node.setop_type == SetOperationType::INTERSECT);
		logical_type = LogicalOperatorType::LOGICAL_INTERSECT;
		break;
	}

	auto root = make_uniq<LogicalSetOperation>(node.setop_index, node.types.size(), std::move(left_node),
	                                           std::move(right_node), logical_type);

	return VisitQueryNode(node, std::move(root));
}

} // namespace duckdb

















namespace duckdb {

static unique_ptr<Expression> PlanUncorrelatedSubquery(Binder &binder, BoundSubqueryExpression &expr,
                                                       unique_ptr<LogicalOperator> &root,
                                                       unique_ptr<LogicalOperator> plan) {
	D_ASSERT(!expr.IsCorrelated());
	switch (expr.subquery_type) {
	case SubqueryType::EXISTS: {
		// uncorrelated EXISTS
		// we only care about existence, hence we push a LIMIT 1 operator
		auto limit = make_uniq<LogicalLimit>(1, 0, nullptr, nullptr);
		limit->AddChild(std::move(plan));
		plan = std::move(limit);

		// now we push a COUNT(*) aggregate onto the limit, this will be either 0 or 1 (EXISTS or NOT EXISTS)
		auto count_star_fun = CountStarFun::GetFunction();

		FunctionBinder function_binder(binder.context);
		auto count_star =
		    function_binder.BindAggregateFunction(count_star_fun, {}, nullptr, AggregateType::NON_DISTINCT);
		auto idx_type = count_star->return_type;
		vector<unique_ptr<Expression>> aggregate_list;
		aggregate_list.push_back(std::move(count_star));
		auto aggregate_index = binder.GenerateTableIndex();
		auto aggregate =
		    make_uniq<LogicalAggregate>(binder.GenerateTableIndex(), aggregate_index, std::move(aggregate_list));
		aggregate->AddChild(std::move(plan));
		plan = std::move(aggregate);

		// now we push a projection with a comparison to 1
		auto left_child = make_uniq<BoundColumnRefExpression>(idx_type, ColumnBinding(aggregate_index, 0));
		auto right_child = make_uniq<BoundConstantExpression>(Value::Numeric(idx_type, 1));
		auto comparison = make_uniq<BoundComparisonExpression>(ExpressionType::COMPARE_EQUAL, std::move(left_child),
		                                                       std::move(right_child));

		vector<unique_ptr<Expression>> projection_list;
		projection_list.push_back(std::move(comparison));
		auto projection_index = binder.GenerateTableIndex();
		auto projection = make_uniq<LogicalProjection>(projection_index, std::move(projection_list));
		projection->AddChild(std::move(plan));
		plan = std::move(projection);

		// we add it to the main query by adding a cross product
		// FIXME: should use something else besides cross product as we always add only one scalar constant
		root = LogicalCrossProduct::Create(std::move(root), std::move(plan));

		// we replace the original subquery with a ColumnRefExpression referring to the result of the projection (either
		// TRUE or FALSE)
		return make_uniq<BoundColumnRefExpression>(expr.GetName(), LogicalType::BOOLEAN,
		                                           ColumnBinding(projection_index, 0));
	}
	case SubqueryType::SCALAR: {
		// uncorrelated scalar, we want to return the first entry
		// figure out the table index of the bound table of the entry which we want to return
		auto bindings = plan->GetColumnBindings();
		D_ASSERT(bindings.size() == 1);
		idx_t table_idx = bindings[0].table_index;

		// in the uncorrelated case we are only interested in the first result of the query
		// hence we simply push a LIMIT 1 to get the first row of the subquery
		auto limit = make_uniq<LogicalLimit>(1, 0, nullptr, nullptr);
		limit->AddChild(std::move(plan));
		plan = std::move(limit);

		// we push an aggregate that returns the FIRST element
		vector<unique_ptr<Expression>> expressions;
		auto bound = make_uniq<BoundColumnRefExpression>(expr.return_type, ColumnBinding(table_idx, 0));
		vector<unique_ptr<Expression>> first_children;
		first_children.push_back(std::move(bound));

		FunctionBinder function_binder(binder.context);
		auto first_agg = function_binder.BindAggregateFunction(
		    FirstFun::GetFunction(expr.return_type), std::move(first_children), nullptr, AggregateType::NON_DISTINCT);

		expressions.push_back(std::move(first_agg));
		auto aggr_index = binder.GenerateTableIndex();
		auto aggr = make_uniq<LogicalAggregate>(binder.GenerateTableIndex(), aggr_index, std::move(expressions));
		aggr->AddChild(std::move(plan));
		plan = std::move(aggr);

		// in the uncorrelated case, we add the value to the main query through a cross product
		// FIXME: should use something else besides cross product as we always add only one scalar constant and cross
		// product is not optimized for this.
		D_ASSERT(root);
		root = LogicalCrossProduct::Create(std::move(root), std::move(plan));

		// we replace the original subquery with a BoundColumnRefExpression referring to the first result of the
		// aggregation
		return make_uniq<BoundColumnRefExpression>(expr.GetName(), expr.return_type, ColumnBinding(aggr_index, 0));
	}
	default: {
		D_ASSERT(expr.subquery_type == SubqueryType::ANY);
		// we generate a MARK join that results in either (TRUE, FALSE or NULL)
		// subquery has NULL values -> result is (TRUE or NULL)
		// subquery has no NULL values -> result is (TRUE, FALSE or NULL [if input is NULL])
		// fetch the column bindings
		auto plan_columns = plan->GetColumnBindings();

		// then we generate the MARK join with the subquery
		idx_t mark_index = binder.GenerateTableIndex();
		auto join = make_uniq<LogicalComparisonJoin>(JoinType::MARK);
		join->mark_index = mark_index;
		join->AddChild(std::move(root));
		join->AddChild(std::move(plan));
		// create the JOIN condition
		JoinCondition cond;
		cond.left = std::move(expr.child);
		cond.right = BoundCastExpression::AddDefaultCastToType(
		    make_uniq<BoundColumnRefExpression>(expr.child_type, plan_columns[0]), expr.child_target);
		cond.comparison = expr.comparison_type;
		join->conditions.push_back(std::move(cond));
		root = std::move(join);

		// we replace the original subquery with a BoundColumnRefExpression referring to the mark column
		return make_uniq<BoundColumnRefExpression>(expr.GetName(), expr.return_type, ColumnBinding(mark_index, 0));
	}
	}
}

static unique_ptr<LogicalDelimJoin>
CreateDuplicateEliminatedJoin(const vector<CorrelatedColumnInfo> &correlated_columns, JoinType join_type,
                              unique_ptr<LogicalOperator> original_plan, bool perform_delim) {
	auto delim_join = make_uniq<LogicalDelimJoin>(join_type);
	if (!perform_delim) {
		// if we are not performing a delim join, we push a row_number() OVER() window operator on the LHS
		// and perform all duplicate elimination on that row number instead
		D_ASSERT(correlated_columns[0].type.id() == LogicalTypeId::BIGINT);
		auto window = make_uniq<LogicalWindow>(correlated_columns[0].binding.table_index);
		auto row_number =
		    make_uniq<BoundWindowExpression>(ExpressionType::WINDOW_ROW_NUMBER, LogicalType::BIGINT, nullptr, nullptr);
		row_number->start = WindowBoundary::UNBOUNDED_PRECEDING;
		row_number->end = WindowBoundary::CURRENT_ROW_ROWS;
		row_number->alias = "delim_index";
		window->expressions.push_back(std::move(row_number));
		window->AddChild(std::move(original_plan));
		original_plan = std::move(window);
	}
	delim_join->AddChild(std::move(original_plan));
	for (idx_t i = 0; i < correlated_columns.size(); i++) {
		auto &col = correlated_columns[i];
		delim_join->duplicate_eliminated_columns.push_back(make_uniq<BoundColumnRefExpression>(col.type, col.binding));
		delim_join->delim_types.push_back(col.type);
	}
	return delim_join;
}

static void CreateDelimJoinConditions(LogicalDelimJoin &delim_join,
                                      const vector<CorrelatedColumnInfo> &correlated_columns,
                                      vector<ColumnBinding> bindings, idx_t base_offset, bool perform_delim) {
	auto col_count = perform_delim ? correlated_columns.size() : 1;
	for (idx_t i = 0; i < col_count; i++) {
		auto &col = correlated_columns[i];
		auto binding_idx = base_offset + i;
		if (binding_idx >= bindings.size()) {
			throw InternalException("Delim join - binding index out of range");
		}
		JoinCondition cond;
		cond.left = make_uniq<BoundColumnRefExpression>(col.name, col.type, col.binding);
		cond.right = make_uniq<BoundColumnRefExpression>(col.name, col.type, bindings[binding_idx]);
		cond.comparison = ExpressionType::COMPARE_NOT_DISTINCT_FROM;
		delim_join.conditions.push_back(std::move(cond));
	}
}

static bool PerformDelimOnType(const LogicalType &type) {
	if (type.InternalType() == PhysicalType::LIST) {
		return false;
	}
	if (type.InternalType() == PhysicalType::STRUCT) {
		for (auto &entry : StructType::GetChildTypes(type)) {
			if (!PerformDelimOnType(entry.second)) {
				return false;
			}
		}
	}
	return true;
}

static bool PerformDuplicateElimination(Binder &binder, vector<CorrelatedColumnInfo> &correlated_columns) {
	if (!ClientConfig::GetConfig(binder.context).enable_optimizer) {
		// if optimizations are disabled we always do a delim join
		return true;
	}
	bool perform_delim = true;
	for (auto &col : correlated_columns) {
		if (!PerformDelimOnType(col.type)) {
			perform_delim = false;
			break;
		}
	}
	if (perform_delim) {
		return true;
	}
	auto binding = ColumnBinding(binder.GenerateTableIndex(), 0);
	auto type = LogicalType::BIGINT;
	auto name = "delim_index";
	CorrelatedColumnInfo info(binding, type, name, 0);
	correlated_columns.insert(correlated_columns.begin(), std::move(info));
	return false;
}

static unique_ptr<Expression> PlanCorrelatedSubquery(Binder &binder, BoundSubqueryExpression &expr,
                                                     unique_ptr<LogicalOperator> &root,
                                                     unique_ptr<LogicalOperator> plan) {
	auto &correlated_columns = expr.binder->correlated_columns;
	// FIXME: there should be a way of disabling decorrelation for ANY queries as well, but not for now...
	bool perform_delim =
	    expr.subquery_type == SubqueryType::ANY ? true : PerformDuplicateElimination(binder, correlated_columns);
	D_ASSERT(expr.IsCorrelated());
	// correlated subquery
	// for a more in-depth explanation of this code, read the paper "Unnesting Arbitrary Subqueries"
	// we handle three types of correlated subqueries: Scalar, EXISTS and ANY
	// all three cases are very similar with some minor changes (mainly the type of join performed at the end)
	switch (expr.subquery_type) {
	case SubqueryType::SCALAR: {
		// correlated SCALAR query
		// first push a DUPLICATE ELIMINATED join
		// a duplicate eliminated join creates a duplicate eliminated copy of the LHS
		// and pushes it into any DUPLICATE_ELIMINATED SCAN operators on the RHS

		// in the SCALAR case, we create a SINGLE join (because we are only interested in obtaining the value)
		// NULL values are equal in this join because we join on the correlated columns ONLY
		// and e.g. in the query: SELECT (SELECT 42 FROM integers WHERE i1.i IS NULL LIMIT 1) FROM integers i1;
		// the input value NULL will generate the value 42, and we need to join NULL on the LHS with NULL on the RHS
		// the left side is the original plan
		// this is the side that will be duplicate eliminated and pushed into the RHS
		auto delim_join =
		    CreateDuplicateEliminatedJoin(correlated_columns, JoinType::SINGLE, std::move(root), perform_delim);

		// the right side initially is a DEPENDENT join between the duplicate eliminated scan and the subquery
		// HOWEVER: we do not explicitly create the dependent join
		// instead, we eliminate the dependent join by pushing it down into the right side of the plan
		FlattenDependentJoins flatten(binder, correlated_columns, perform_delim);

		// first we check which logical operators have correlated expressions in the first place
		flatten.DetectCorrelatedExpressions(plan.get());
		// now we push the dependent join down
		auto dependent_join = flatten.PushDownDependentJoin(std::move(plan));

		// now the dependent join is fully eliminated
		// we only need to create the join conditions between the LHS and the RHS
		// fetch the set of columns
		auto plan_columns = dependent_join->GetColumnBindings();

		// now create the join conditions
		CreateDelimJoinConditions(*delim_join, correlated_columns, plan_columns, flatten.delim_offset, perform_delim);
		delim_join->AddChild(std::move(dependent_join));
		root = std::move(delim_join);
		// finally push the BoundColumnRefExpression referring to the data element returned by the join
		return make_uniq<BoundColumnRefExpression>(expr.GetName(), expr.return_type, plan_columns[flatten.data_offset]);
	}
	case SubqueryType::EXISTS: {
		// correlated EXISTS query
		// this query is similar to the correlated SCALAR query, except we use a MARK join here
		idx_t mark_index = binder.GenerateTableIndex();
		auto delim_join =
		    CreateDuplicateEliminatedJoin(correlated_columns, JoinType::MARK, std::move(root), perform_delim);
		delim_join->mark_index = mark_index;
		// RHS
		FlattenDependentJoins flatten(binder, correlated_columns, perform_delim, true);
		flatten.DetectCorrelatedExpressions(plan.get());
		auto dependent_join = flatten.PushDownDependentJoin(std::move(plan));

		// fetch the set of columns
		auto plan_columns = dependent_join->GetColumnBindings();

		// now we create the join conditions between the dependent join and the original table
		CreateDelimJoinConditions(*delim_join, correlated_columns, plan_columns, flatten.delim_offset, perform_delim);
		delim_join->AddChild(std::move(dependent_join));
		root = std::move(delim_join);
		// finally push the BoundColumnRefExpression referring to the marker
		return make_uniq<BoundColumnRefExpression>(expr.GetName(), expr.return_type, ColumnBinding(mark_index, 0));
	}
	default: {
		D_ASSERT(expr.subquery_type == SubqueryType::ANY);
		// correlated ANY query
		// this query is similar to the correlated SCALAR query
		// however, in this case we push a correlated MARK join
		// note that in this join null values are NOT equal for ALL columns, but ONLY for the correlated columns
		// the correlated mark join handles this case by itself
		// as the MARK join has one extra join condition (the original condition, of the ANY expression, e.g.
		// [i=ANY(...)])
		idx_t mark_index = binder.GenerateTableIndex();
		auto delim_join =
		    CreateDuplicateEliminatedJoin(correlated_columns, JoinType::MARK, std::move(root), perform_delim);
		delim_join->mark_index = mark_index;
		// RHS
		FlattenDependentJoins flatten(binder, correlated_columns, true, true);
		flatten.DetectCorrelatedExpressions(plan.get());
		auto dependent_join = flatten.PushDownDependentJoin(std::move(plan));

		// fetch the columns
		auto plan_columns = dependent_join->GetColumnBindings();

		// now we create the join conditions between the dependent join and the original table
		CreateDelimJoinConditions(*delim_join, correlated_columns, plan_columns, flatten.delim_offset, perform_delim);
		// add the actual condition based on the ANY/ALL predicate
		JoinCondition compare_cond;
		compare_cond.left = std::move(expr.child);
		compare_cond.right = BoundCastExpression::AddDefaultCastToType(
		    make_uniq<BoundColumnRefExpression>(expr.child_type, plan_columns[0]), expr.child_target);
		compare_cond.comparison = expr.comparison_type;
		delim_join->conditions.push_back(std::move(compare_cond));

		delim_join->AddChild(std::move(dependent_join));
		root = std::move(delim_join);
		// finally push the BoundColumnRefExpression referring to the marker
		return make_uniq<BoundColumnRefExpression>(expr.GetName(), expr.return_type, ColumnBinding(mark_index, 0));
	}
	}
}

class RecursiveSubqueryPlanner : public LogicalOperatorVisitor {
public:
	explicit RecursiveSubqueryPlanner(Binder &binder) : binder(binder) {
	}
	void VisitOperator(LogicalOperator &op) override {
		if (!op.children.empty()) {
			root = std::move(op.children[0]);
			D_ASSERT(root);
			VisitOperatorExpressions(op);
			op.children[0] = std::move(root);
			for (idx_t i = 0; i < op.children.size(); i++) {
				D_ASSERT(op.children[i]);
				VisitOperator(*op.children[i]);
			}
		}
	}

	unique_ptr<Expression> VisitReplace(BoundSubqueryExpression &expr, unique_ptr<Expression> *expr_ptr) override {
		return binder.PlanSubquery(expr, root);
	}

private:
	unique_ptr<LogicalOperator> root;
	Binder &binder;
};

unique_ptr<Expression> Binder::PlanSubquery(BoundSubqueryExpression &expr, unique_ptr<LogicalOperator> &root) {
	D_ASSERT(root);
	// first we translate the QueryNode of the subquery into a logical plan
	// note that we do not plan nested subqueries yet
	auto sub_binder = Binder::CreateBinder(context, this);
	sub_binder->plan_subquery = false;
	auto subquery_root = sub_binder->CreatePlan(*expr.subquery);
	D_ASSERT(subquery_root);

	// now we actually flatten the subquery
	auto plan = std::move(subquery_root);
	unique_ptr<Expression> result_expression;
	if (!expr.IsCorrelated()) {
		result_expression = PlanUncorrelatedSubquery(*this, expr, root, std::move(plan));
	} else {
		result_expression = PlanCorrelatedSubquery(*this, expr, root, std::move(plan));
	}
	// finally, we recursively plan the nested subqueries (if there are any)
	if (sub_binder->has_unplanned_subqueries) {
		RecursiveSubqueryPlanner plan(*this);
		plan.VisitOperator(*root);
	}
	return result_expression;
}

void Binder::PlanSubqueries(unique_ptr<Expression> &expr_ptr, unique_ptr<LogicalOperator> &root) {
	if (!expr_ptr) {
		return;
	}
	auto &expr = *expr_ptr;

	// first visit the children of the node, if any
	ExpressionIterator::EnumerateChildren(expr, [&](unique_ptr<Expression> &expr) { PlanSubqueries(expr, root); });

	// check if this is a subquery node
	if (expr.expression_class == ExpressionClass::BOUND_SUBQUERY) {
		auto &subquery = expr.Cast<BoundSubqueryExpression>();
		// subquery node! plan it
		if (subquery.IsCorrelated() && !plan_subquery) {
			// detected a nested correlated subquery
			// we don't plan it yet here, we are currently planning a subquery
			// nested subqueries will only be planned AFTER the current subquery has been flattened entirely
			has_unplanned_subqueries = true;
			return;
		}
		expr_ptr = PlanSubquery(subquery, root);
	}
}

unique_ptr<LogicalOperator> Binder::PlanLateralJoin(unique_ptr<LogicalOperator> left, unique_ptr<LogicalOperator> right,
                                                    vector<CorrelatedColumnInfo> &correlated_columns,
                                                    JoinType join_type, unique_ptr<Expression> condition) {
	// scan the right operator for correlated columns
	// correlated LATERAL JOIN
	vector<JoinCondition> conditions;
	vector<unique_ptr<Expression>> arbitrary_expressions;
	if (condition) {
		// extract join conditions, if there are any
		LogicalComparisonJoin::ExtractJoinConditions(join_type, left, right, std::move(condition), conditions,
		                                             arbitrary_expressions);
	}

	auto perform_delim = PerformDuplicateElimination(*this, correlated_columns);
	auto delim_join = CreateDuplicateEliminatedJoin(correlated_columns, join_type, std::move(left), perform_delim);

	FlattenDependentJoins flatten(*this, correlated_columns, perform_delim);

	// first we check which logical operators have correlated expressions in the first place
	flatten.DetectCorrelatedExpressions(right.get(), true);
	// now we push the dependent join down
	auto dependent_join = flatten.PushDownDependentJoin(std::move(right));

	// now the dependent join is fully eliminated
	// we only need to create the join conditions between the LHS and the RHS
	// fetch the set of columns
	auto plan_columns = dependent_join->GetColumnBindings();

	// now create the join conditions
	// start off with the conditions that were passed in (if any)
	D_ASSERT(delim_join->conditions.empty());
	delim_join->conditions = std::move(conditions);
	// then add the delim join conditions
	CreateDelimJoinConditions(*delim_join, correlated_columns, plan_columns, flatten.delim_offset, perform_delim);
	delim_join->AddChild(std::move(dependent_join));

	// check if there are any arbitrary expressions left
	if (!arbitrary_expressions.empty()) {
		// we can only evaluate scalar arbitrary expressions for inner joins
		if (join_type != JoinType::INNER) {
			throw BinderException(
			    "Join condition for non-inner LATERAL JOIN must be a comparison between the left and right side");
		}
		auto filter = make_uniq<LogicalFilter>();
		filter->expressions = std::move(arbitrary_expressions);
		filter->AddChild(std::move(delim_join));
		return std::move(filter);
	}
	return std::move(delim_join);
}

} // namespace duckdb






namespace duckdb {

BoundStatement Binder::Bind(AttachStatement &stmt) {
	BoundStatement result;
	result.types = {LogicalType::BOOLEAN};
	result.names = {"Success"};

	result.plan = make_uniq<LogicalSimple>(LogicalOperatorType::LOGICAL_ATTACH, std::move(stmt.info));
	properties.allow_stream_result = false;
	properties.return_type = StatementReturnType::NOTHING;
	return result;
}

} // namespace duckdb






namespace duckdb {

BoundStatement Binder::Bind(CallStatement &stmt) {
	BoundStatement result;

	TableFunctionRef ref;
	ref.function = std::move(stmt.function);

	auto bound_func = Bind(ref);
	auto &bound_table_func = bound_func->Cast<BoundTableFunction>();
	;
	auto &get = bound_table_func.get->Cast<LogicalGet>();
	D_ASSERT(get.returned_types.size() > 0);
	for (idx_t i = 0; i < get.returned_types.size(); i++) {
		get.column_ids.push_back(i);
	}

	result.types = get.returned_types;
	result.names = get.names;
	result.plan = CreatePlan(*bound_func);
	properties.return_type = StatementReturnType::QUERY_RESULT;
	return result;
}

} // namespace duckdb






















#include <algorithm>

namespace duckdb {

vector<string> GetUniqueNames(const vector<string> &original_names) {
	unordered_set<string> name_set;
	vector<string> unique_names;
	unique_names.reserve(original_names.size());

	for (auto &name : original_names) {
		auto insert_result = name_set.insert(name);
		if (insert_result.second == false) {
			// Could not be inserted, name already exists
			idx_t index = 1;
			string postfixed_name;
			while (true) {
				postfixed_name = StringUtil::Format("%s:%d", name, index);
				auto res = name_set.insert(postfixed_name);
				if (!res.second) {
					index++;
					continue;
				}
				break;
			}
			unique_names.push_back(postfixed_name);
		} else {
			unique_names.push_back(name);
		}
	}
	return unique_names;
}

BoundStatement Binder::BindCopyTo(CopyStatement &stmt) {
	// COPY TO a file
	auto &config = DBConfig::GetConfig(context);
	if (!config.options.enable_external_access) {
		throw PermissionException("COPY TO is disabled by configuration");
	}
	BoundStatement result;
	result.types = {LogicalType::BIGINT};
	result.names = {"Count"};

	// lookup the format in the catalog
	auto &copy_function =
	    Catalog::GetEntry<CopyFunctionCatalogEntry>(context, INVALID_CATALOG, DEFAULT_SCHEMA, stmt.info->format);
	if (copy_function.function.plan) {
		// plan rewrite COPY TO
		return copy_function.function.plan(*this, stmt);
	}

	// bind the select statement
	auto select_node = Bind(*stmt.select_statement);

	if (!copy_function.function.copy_to_bind) {
		throw NotImplementedException("COPY TO is not supported for FORMAT \"%s\"", stmt.info->format);
	}
	bool use_tmp_file = true;
	bool overwrite_or_ignore = false;
	FilenamePattern filename_pattern;
	bool user_set_use_tmp_file = false;
	bool per_thread_output = false;
	vector<idx_t> partition_cols;

	auto original_options = stmt.info->options;
	stmt.info->options.clear();

	for (auto &option : original_options) {
		auto loption = StringUtil::Lower(option.first);
		if (loption == "use_tmp_file") {
			use_tmp_file =
			    option.second.empty() || option.second[0].CastAs(context, LogicalType::BOOLEAN).GetValue<bool>();
			user_set_use_tmp_file = true;
			continue;
		}
		if (loption == "overwrite_or_ignore") {
			overwrite_or_ignore =
			    option.second.empty() || option.second[0].CastAs(context, LogicalType::BOOLEAN).GetValue<bool>();
			continue;
		}
		if (loption == "filename_pattern") {
			if (option.second.empty()) {
				throw IOException("FILENAME_PATTERN cannot be empty");
			}
			filename_pattern.SetFilenamePattern(
			    option.second[0].CastAs(context, LogicalType::VARCHAR).GetValue<string>());
			continue;
		}

		if (loption == "per_thread_output") {
			per_thread_output =
			    option.second.empty() || option.second[0].CastAs(context, LogicalType::BOOLEAN).GetValue<bool>();
			continue;
		}
		if (loption == "partition_by") {
			auto converted = ConvertVectorToValue(std::move(option.second));
			partition_cols = ParseColumnsOrdered(converted, select_node.names, loption);
			continue;
		}
		stmt.info->options[option.first] = option.second;
	}
	if (user_set_use_tmp_file && per_thread_output) {
		throw NotImplementedException("Can't combine USE_TMP_FILE and PER_THREAD_OUTPUT for COPY");
	}
	if (user_set_use_tmp_file && !partition_cols.empty()) {
		throw NotImplementedException("Can't combine USE_TMP_FILE and PARTITION_BY for COPY");
	}
	if (per_thread_output && !partition_cols.empty()) {
		throw NotImplementedException("Can't combine PER_THREAD_OUTPUT and PARTITION_BY for COPY");
	}
	bool is_remote_file = config.file_system->IsRemoteFile(stmt.info->file_path);
	if (is_remote_file) {
		use_tmp_file = false;
	} else {
		bool is_file_and_exists = config.file_system->FileExists(stmt.info->file_path);
		bool is_stdout = stmt.info->file_path == "/dev/stdout";
		if (!user_set_use_tmp_file) {
			use_tmp_file = is_file_and_exists && !per_thread_output && partition_cols.empty() && !is_stdout;
		}
	}

	auto unique_column_names = GetUniqueNames(select_node.names);

	auto function_data =
	    copy_function.function.copy_to_bind(context, *stmt.info, unique_column_names, select_node.types);
	// now create the copy information
	auto copy = make_uniq<LogicalCopyToFile>(copy_function.function, std::move(function_data));
	copy->file_path = stmt.info->file_path;
	copy->use_tmp_file = use_tmp_file;
	copy->overwrite_or_ignore = overwrite_or_ignore;
	copy->filename_pattern = filename_pattern;
	copy->per_thread_output = per_thread_output;
	copy->partition_output = !partition_cols.empty();
	copy->partition_columns = std::move(partition_cols);

	copy->names = unique_column_names;
	copy->expected_types = select_node.types;

	copy->AddChild(std::move(select_node.plan));

	result.plan = std::move(copy);

	return result;
}

BoundStatement Binder::BindCopyFrom(CopyStatement &stmt) {
	auto &config = DBConfig::GetConfig(context);
	if (!config.options.enable_external_access) {
		throw PermissionException("COPY FROM is disabled by configuration");
	}
	BoundStatement result;
	result.types = {LogicalType::BIGINT};
	result.names = {"Count"};

	if (stmt.info->table.empty()) {
		throw ParserException("COPY FROM requires a table name to be specified");
	}
	// COPY FROM a file
	// generate an insert statement for the the to-be-inserted table
	InsertStatement insert;
	insert.table = stmt.info->table;
	insert.schema = stmt.info->schema;
	insert.catalog = stmt.info->catalog;
	insert.columns = stmt.info->select_list;

	// bind the insert statement to the base table
	auto insert_statement = Bind(insert);
	D_ASSERT(insert_statement.plan->type == LogicalOperatorType::LOGICAL_INSERT);

	auto &bound_insert = insert_statement.plan->Cast<LogicalInsert>();

	// lookup the format in the catalog
	auto &catalog = Catalog::GetSystemCatalog(context);
	auto &copy_function = catalog.GetEntry<CopyFunctionCatalogEntry>(context, DEFAULT_SCHEMA, stmt.info->format);
	if (!copy_function.function.copy_from_bind) {
		throw NotImplementedException("COPY FROM is not supported for FORMAT \"%s\"", stmt.info->format);
	}
	// lookup the table to copy into
	BindSchemaOrCatalog(stmt.info->catalog, stmt.info->schema);
	auto &table =
	    Catalog::GetEntry<TableCatalogEntry>(context, stmt.info->catalog, stmt.info->schema, stmt.info->table);
	vector<string> expected_names;
	if (!bound_insert.column_index_map.empty()) {
		expected_names.resize(bound_insert.expected_types.size());
		for (auto &col : table.GetColumns().Physical()) {
			auto i = col.Physical();
			if (bound_insert.column_index_map[i] != DConstants::INVALID_INDEX) {
				expected_names[bound_insert.column_index_map[i]] = col.Name();
			}
		}
	} else {
		expected_names.reserve(bound_insert.expected_types.size());
		for (auto &col : table.GetColumns().Physical()) {
			expected_names.push_back(col.Name());
		}
	}

	auto function_data =
	    copy_function.function.copy_from_bind(context, *stmt.info, expected_names, bound_insert.expected_types);
	auto get = make_uniq<LogicalGet>(GenerateTableIndex(), copy_function.function.copy_from_function,
	                                 std::move(function_data), bound_insert.expected_types, expected_names);
	for (idx_t i = 0; i < bound_insert.expected_types.size(); i++) {
		get->column_ids.push_back(i);
	}
	insert_statement.plan->children.push_back(std::move(get));
	result.plan = std::move(insert_statement.plan);
	return result;
}

BoundStatement Binder::Bind(CopyStatement &stmt) {
	if (!stmt.info->is_from && !stmt.select_statement) {
		// copy table into file without a query
		// generate SELECT * FROM table;
		auto ref = make_uniq<BaseTableRef>();
		ref->catalog_name = stmt.info->catalog;
		ref->schema_name = stmt.info->schema;
		ref->table_name = stmt.info->table;

		auto statement = make_uniq<SelectNode>();
		statement->from_table = std::move(ref);
		if (!stmt.info->select_list.empty()) {
			for (auto &name : stmt.info->select_list) {
				statement->select_list.push_back(make_uniq<ColumnRefExpression>(name));
			}
		} else {
			statement->select_list.push_back(make_uniq<StarExpression>());
		}
		stmt.select_statement = std::move(statement);
	}
	properties.allow_stream_result = false;
	properties.return_type = StatementReturnType::CHANGED_ROWS;
	if (stmt.info->is_from) {
		return BindCopyFrom(stmt);
	} else {
		return BindCopyTo(stmt);
	}
}

} // namespace duckdb











































namespace duckdb {

void Binder::BindSchemaOrCatalog(ClientContext &context, string &catalog, string &schema) {
	if (catalog.empty() && !schema.empty()) {
		// schema is specified - but catalog is not
		// try searching for the catalog instead
		auto &db_manager = DatabaseManager::Get(context);
		auto database = db_manager.GetDatabase(context, schema);
		if (database) {
			// we have a database with this name
			// check if there is a schema
			auto schema_obj = Catalog::GetSchema(context, INVALID_CATALOG, schema, OnEntryNotFound::RETURN_NULL);
			if (schema_obj) {
				auto &attached = schema_obj->catalog.GetAttached();
				throw BinderException(
				    "Ambiguous reference to catalog or schema \"%s\" - use a fully qualified path like \"%s.%s\"",
				    schema, attached.GetName(), schema);
			}
			catalog = schema;
			schema = string();
		}
	}
}

void Binder::BindSchemaOrCatalog(string &catalog, string &schema) {
	BindSchemaOrCatalog(context, catalog, schema);
}

SchemaCatalogEntry &Binder::BindSchema(CreateInfo &info) {
	BindSchemaOrCatalog(info.catalog, info.schema);
	if (IsInvalidCatalog(info.catalog) && info.temporary) {
		info.catalog = TEMP_CATALOG;
	}
	auto &search_path = ClientData::Get(context).catalog_search_path;
	if (IsInvalidCatalog(info.catalog) && IsInvalidSchema(info.schema)) {
		auto &default_entry = search_path->GetDefault();
		info.catalog = default_entry.catalog;
		info.schema = default_entry.schema;
	} else if (IsInvalidSchema(info.schema)) {
		info.schema = search_path->GetDefaultSchema(info.catalog);
	} else if (IsInvalidCatalog(info.catalog)) {
		info.catalog = search_path->GetDefaultCatalog(info.schema);
	}
	if (IsInvalidCatalog(info.catalog)) {
		info.catalog = DatabaseManager::GetDefaultDatabase(context);
	}
	if (!info.temporary) {
		// non-temporary create: not read only
		if (info.catalog == TEMP_CATALOG) {
			throw ParserException("Only TEMPORARY table names can use the \"%s\" catalog", TEMP_CATALOG);
		}
	} else {
		if (info.catalog != TEMP_CATALOG) {
			throw ParserException("TEMPORARY table names can *only* use the \"%s\" catalog", TEMP_CATALOG);
		}
	}
	// fetch the schema in which we want to create the object
	auto &schema_obj = Catalog::GetSchema(context, info.catalog, info.schema);
	D_ASSERT(schema_obj.type == CatalogType::SCHEMA_ENTRY);
	info.schema = schema_obj.name;
	if (!info.temporary) {
		properties.modified_databases.insert(schema_obj.catalog.GetName());
	}
	return schema_obj;
}

SchemaCatalogEntry &Binder::BindCreateSchema(CreateInfo &info) {
	auto &schema = BindSchema(info);
	if (schema.catalog.IsSystemCatalog()) {
		throw BinderException("Cannot create entry in system catalog");
	}
	return schema;
}

void Binder::BindCreateViewInfo(CreateViewInfo &base) {
	// bind the view as if it were a query so we can catch errors
	// note that we bind the original, and replace the original with a copy
	auto view_binder = Binder::CreateBinder(context);
	view_binder->can_contain_nulls = true;

	auto copy = base.query->Copy();
	auto query_node = view_binder->Bind(*base.query);
	base.query = unique_ptr_cast<SQLStatement, SelectStatement>(std::move(copy));
	if (base.aliases.size() > query_node.names.size()) {
		throw BinderException("More VIEW aliases than columns in query result");
	}
	// fill up the aliases with the remaining names of the bound query
	base.aliases.reserve(query_node.names.size());
	for (idx_t i = base.aliases.size(); i < query_node.names.size(); i++) {
		base.aliases.push_back(query_node.names[i]);
	}
	base.types = query_node.types;
}

static void QualifyFunctionNames(ClientContext &context, unique_ptr<ParsedExpression> &expr) {
	switch (expr->GetExpressionClass()) {
	case ExpressionClass::FUNCTION: {
		auto &func = expr->Cast<FunctionExpression>();
		auto function = Catalog::GetEntry(context, CatalogType::SCALAR_FUNCTION_ENTRY, func.catalog, func.schema,
		                                  func.function_name, OnEntryNotFound::RETURN_NULL);
		if (function) {
			func.catalog = function->ParentCatalog().GetName();
			func.schema = function->ParentSchema().name;
		}
		break;
	}
	case ExpressionClass::SUBQUERY: {
		// replacing parameters within a subquery is slightly different
		auto &sq = (expr->Cast<SubqueryExpression>()).subquery;
		ParsedExpressionIterator::EnumerateQueryNodeChildren(
		    *sq->node, [&](unique_ptr<ParsedExpression> &child) { QualifyFunctionNames(context, child); });
		break;
	}
	default: // fall through
		break;
	}
	// unfold child expressions
	ParsedExpressionIterator::EnumerateChildren(
	    *expr, [&](unique_ptr<ParsedExpression> &child) { QualifyFunctionNames(context, child); });
}

SchemaCatalogEntry &Binder::BindCreateFunctionInfo(CreateInfo &info) {
	auto &base = info.Cast<CreateMacroInfo>();
	auto &scalar_function = base.function->Cast<ScalarMacroFunction>();

	if (scalar_function.expression->HasParameter()) {
		throw BinderException("Parameter expressions within macro's are not supported!");
	}

	// create macro binding in order to bind the function
	vector<LogicalType> dummy_types;
	vector<string> dummy_names;
	// positional parameters
	for (idx_t i = 0; i < base.function->parameters.size(); i++) {
		auto param = base.function->parameters[i]->Cast<ColumnRefExpression>();
		if (param.IsQualified()) {
			throw BinderException("Invalid parameter name '%s': must be unqualified", param.ToString());
		}
		dummy_types.emplace_back(LogicalType::SQLNULL);
		dummy_names.push_back(param.GetColumnName());
	}
	// default parameters
	for (auto it = base.function->default_parameters.begin(); it != base.function->default_parameters.end(); it++) {
		auto &val = it->second->Cast<ConstantExpression>();
		dummy_types.push_back(val.value.type());
		dummy_names.push_back(it->first);
	}
	auto this_macro_binding = make_uniq<DummyBinding>(dummy_types, dummy_names, base.name);
	macro_binding = this_macro_binding.get();
	ExpressionBinder::QualifyColumnNames(*this, scalar_function.expression);
	QualifyFunctionNames(context, scalar_function.expression);

	// create a copy of the expression because we do not want to alter the original
	auto expression = scalar_function.expression->Copy();

	// bind it to verify the function was defined correctly
	string error;
	auto sel_node = make_uniq<BoundSelectNode>();
	auto group_info = make_uniq<BoundGroupInformation>();
	SelectBinder binder(*this, context, *sel_node, *group_info);
	error = binder.Bind(expression, 0, false);

	if (!error.empty()) {
		throw BinderException(error);
	}

	return BindCreateSchema(info);
}

void Binder::BindLogicalType(ClientContext &context, LogicalType &type, optional_ptr<Catalog> catalog,
                             const string &schema) {
	if (type.id() == LogicalTypeId::LIST || type.id() == LogicalTypeId::MAP) {
		auto child_type = ListType::GetChildType(type);
		BindLogicalType(context, child_type, catalog, schema);
		auto alias = type.GetAlias();
		if (type.id() == LogicalTypeId::LIST) {
			type = LogicalType::LIST(child_type);
		} else {
			D_ASSERT(child_type.id() == LogicalTypeId::STRUCT); // map must be list of structs
			type = LogicalType::MAP(child_type);
		}

		type.SetAlias(alias);
	} else if (type.id() == LogicalTypeId::STRUCT) {
		auto child_types = StructType::GetChildTypes(type);
		for (auto &child_type : child_types) {
			BindLogicalType(context, child_type.second, catalog, schema);
		}
		// Generate new Struct Type
		auto alias = type.GetAlias();
		type = LogicalType::STRUCT(child_types);
		type.SetAlias(alias);
	} else if (type.id() == LogicalTypeId::UNION) {
		auto member_types = UnionType::CopyMemberTypes(type);
		for (auto &member_type : member_types) {
			BindLogicalType(context, member_type.second, catalog, schema);
		}
		// Generate new Union Type
		auto alias = type.GetAlias();
		type = LogicalType::UNION(member_types);
		type.SetAlias(alias);
	} else if (type.id() == LogicalTypeId::USER) {
		auto &user_type_name = UserType::GetTypeName(type);
		if (catalog) {
			type = catalog->GetType(context, schema, user_type_name, OnEntryNotFound::RETURN_NULL);
			if (type.id() == LogicalTypeId::INVALID) {
				// look in the system catalog if the type was not found
				type = Catalog::GetType(context, SYSTEM_CATALOG, schema, user_type_name);
			}
		} else {
			type = Catalog::GetType(context, INVALID_CATALOG, schema, user_type_name);
		}
	} else if (type.id() == LogicalTypeId::ENUM) {
		auto &enum_type_name = EnumType::GetTypeName(type);
		optional_ptr<TypeCatalogEntry> enum_type_catalog;
		if (catalog) {
			enum_type_catalog =
			    catalog->GetEntry<TypeCatalogEntry>(context, schema, enum_type_name, OnEntryNotFound::RETURN_NULL);
			if (!enum_type_catalog) {
				// look in the system catalog if the type was not found
				enum_type_catalog = Catalog::GetEntry<TypeCatalogEntry>(context, SYSTEM_CATALOG, schema, enum_type_name,
				                                                        OnEntryNotFound::RETURN_NULL);
			}
		} else {
			enum_type_catalog = Catalog::GetEntry<TypeCatalogEntry>(context, INVALID_CATALOG, schema, enum_type_name,
			                                                        OnEntryNotFound::RETURN_NULL);
		}

		EnumType::SetCatalog(type, enum_type_catalog.get());
	}
}

static void FindMatchingPrimaryKeyColumns(const ColumnList &columns, const vector<unique_ptr<Constraint>> &constraints,
                                          ForeignKeyConstraint &fk) {
	// find the matching primary key constraint
	bool found_constraint = false;
	// if no columns are defined, we will automatically try to bind to the primary key
	bool find_primary_key = fk.pk_columns.empty();
	for (auto &constr : constraints) {
		if (constr->type != ConstraintType::UNIQUE) {
			continue;
		}
		auto &unique = constr->Cast<UniqueConstraint>();
		if (find_primary_key && !unique.is_primary_key) {
			continue;
		}
		found_constraint = true;

		vector<string> pk_names;
		if (unique.index.index != DConstants::INVALID_INDEX) {
			pk_names.push_back(columns.GetColumn(LogicalIndex(unique.index)).Name());
		} else {
			pk_names = unique.columns;
		}
		if (find_primary_key) {
			// found matching primary key
			if (pk_names.size() != fk.fk_columns.size()) {
				auto pk_name_str = StringUtil::Join(pk_names, ",");
				auto fk_name_str = StringUtil::Join(fk.fk_columns, ",");
				throw BinderException(
				    "Failed to create foreign key: number of referencing (%s) and referenced columns (%s) differ",
				    fk_name_str, pk_name_str);
			}
			fk.pk_columns = pk_names;
			return;
		}
		if (pk_names.size() != fk.fk_columns.size()) {
			// the number of referencing and referenced columns for foreign keys must be the same
			continue;
		}
		bool equals = true;
		for (idx_t i = 0; i < fk.pk_columns.size(); i++) {
			if (!StringUtil::CIEquals(fk.pk_columns[i], pk_names[i])) {
				equals = false;
				break;
			}
		}
		if (!equals) {
			continue;
		}
		// found match
		return;
	}
	// no match found! examine why
	if (!found_constraint) {
		// no unique constraint or primary key
		string search_term = find_primary_key ? "primary key" : "primary key or unique constraint";
		throw BinderException("Failed to create foreign key: there is no %s for referenced table \"%s\"", search_term,
		                      fk.info.table);
	}
	// check if all the columns exist
	for (auto &name : fk.pk_columns) {
		bool found = columns.ColumnExists(name);
		if (!found) {
			throw BinderException(
			    "Failed to create foreign key: referenced table \"%s\" does not have a column named \"%s\"",
			    fk.info.table, name);
		}
	}
	auto fk_names = StringUtil::Join(fk.pk_columns, ",");
	throw BinderException("Failed to create foreign key: referenced table \"%s\" does not have a primary key or unique "
	                      "constraint on the columns %s",
	                      fk.info.table, fk_names);
}

static void FindForeignKeyIndexes(const ColumnList &columns, const vector<string> &names,
                                  vector<PhysicalIndex> &indexes) {
	D_ASSERT(indexes.empty());
	D_ASSERT(!names.empty());
	for (auto &name : names) {
		if (!columns.ColumnExists(name)) {
			throw BinderException("column \"%s\" named in key does not exist", name);
		}
		auto &column = columns.GetColumn(name);
		if (column.Generated()) {
			throw BinderException("Failed to create foreign key: referenced column \"%s\" is a generated column",
			                      column.Name());
		}
		indexes.push_back(column.Physical());
	}
}

static void CheckForeignKeyTypes(const ColumnList &pk_columns, const ColumnList &fk_columns, ForeignKeyConstraint &fk) {
	D_ASSERT(fk.info.pk_keys.size() == fk.info.fk_keys.size());
	for (idx_t c_idx = 0; c_idx < fk.info.pk_keys.size(); c_idx++) {
		auto &pk_col = pk_columns.GetColumn(fk.info.pk_keys[c_idx]);
		auto &fk_col = fk_columns.GetColumn(fk.info.fk_keys[c_idx]);
		if (pk_col.Type() != fk_col.Type()) {
			throw BinderException("Failed to create foreign key: incompatible types between column \"%s\" (\"%s\") and "
			                      "column \"%s\" (\"%s\")",
			                      pk_col.Name(), pk_col.Type().ToString(), fk_col.Name(), fk_col.Type().ToString());
		}
	}
}

void ExpressionContainsGeneratedColumn(const ParsedExpression &expr, const unordered_set<string> &gcols,
                                       bool &contains_gcol) {
	if (contains_gcol) {
		return;
	}
	if (expr.type == ExpressionType::COLUMN_REF) {
		auto &column_ref = expr.Cast<ColumnRefExpression>();
		auto &name = column_ref.GetColumnName();
		if (gcols.count(name)) {
			contains_gcol = true;
			return;
		}
	}
	ParsedExpressionIterator::EnumerateChildren(
	    expr, [&](const ParsedExpression &child) { ExpressionContainsGeneratedColumn(child, gcols, contains_gcol); });
}

static bool AnyConstraintReferencesGeneratedColumn(CreateTableInfo &table_info) {
	unordered_set<string> generated_columns;
	for (auto &col : table_info.columns.Logical()) {
		if (!col.Generated()) {
			continue;
		}
		generated_columns.insert(col.Name());
	}
	if (generated_columns.empty()) {
		return false;
	}

	for (auto &constr : table_info.constraints) {
		switch (constr->type) {
		case ConstraintType::CHECK: {
			auto &constraint = constr->Cast<CheckConstraint>();
			auto &expr = constraint.expression;
			bool contains_generated_column = false;
			ExpressionContainsGeneratedColumn(*expr, generated_columns, contains_generated_column);
			if (contains_generated_column) {
				return true;
			}
			break;
		}
		case ConstraintType::NOT_NULL: {
			auto &constraint = constr->Cast<NotNullConstraint>();
			if (table_info.columns.GetColumn(constraint.index).Generated()) {
				return true;
			}
			break;
		}
		case ConstraintType::UNIQUE: {
			auto &constraint = constr->Cast<UniqueConstraint>();
			auto index = constraint.index;
			if (index.index == DConstants::INVALID_INDEX) {
				for (auto &col : constraint.columns) {
					if (generated_columns.count(col)) {
						return true;
					}
				}
			} else {
				if (table_info.columns.GetColumn(index).Generated()) {
					return true;
				}
			}
			break;
		}
		case ConstraintType::FOREIGN_KEY: {
			// If it contained a generated column, an exception would have been thrown inside AddDataTableIndex earlier
			break;
		}
		default: {
			throw NotImplementedException("ConstraintType not implemented");
		}
		}
	}
	return false;
}

unique_ptr<LogicalOperator> DuckCatalog::BindCreateIndex(Binder &binder, CreateStatement &stmt,
                                                         TableCatalogEntry &table, unique_ptr<LogicalOperator> plan) {
	D_ASSERT(plan->type == LogicalOperatorType::LOGICAL_GET);
	auto &base = stmt.info->Cast<CreateIndexInfo>();

	auto &get = plan->Cast<LogicalGet>();
	// bind the index expressions
	IndexBinder index_binder(binder, binder.context);
	vector<unique_ptr<Expression>> expressions;
	expressions.reserve(base.expressions.size());
	for (auto &expr : base.expressions) {
		expressions.push_back(index_binder.Bind(expr));
	}

	auto create_index_info = unique_ptr_cast<CreateInfo, CreateIndexInfo>(std::move(stmt.info));
	for (auto &column_id : get.column_ids) {
		if (column_id == COLUMN_IDENTIFIER_ROW_ID) {
			throw BinderException("Cannot create an index on the rowid!");
		}
		create_index_info->scan_types.push_back(get.returned_types[column_id]);
	}
	create_index_info->scan_types.emplace_back(LogicalType::ROW_TYPE);
	create_index_info->names = get.names;
	create_index_info->column_ids = get.column_ids;

	// the logical CREATE INDEX also needs all fields to scan the referenced table
	return make_uniq<LogicalCreateIndex>(std::move(get.bind_data), std::move(create_index_info), std::move(expressions),
	                                     table, std::move(get.function));
}

BoundStatement Binder::Bind(CreateStatement &stmt) {
	BoundStatement result;
	result.names = {"Count"};
	result.types = {LogicalType::BIGINT};

	auto catalog_type = stmt.info->type;
	switch (catalog_type) {
	case CatalogType::SCHEMA_ENTRY:
		result.plan = make_uniq<LogicalCreate>(LogicalOperatorType::LOGICAL_CREATE_SCHEMA, std::move(stmt.info));
		break;
	case CatalogType::VIEW_ENTRY: {
		auto &base = (CreateViewInfo &)*stmt.info;
		// bind the schema
		auto &schema = BindCreateSchema(*stmt.info);
		BindCreateViewInfo(base);
		result.plan = make_uniq<LogicalCreate>(LogicalOperatorType::LOGICAL_CREATE_VIEW, std::move(stmt.info), &schema);
		break;
	}
	case CatalogType::SEQUENCE_ENTRY: {
		auto &schema = BindCreateSchema(*stmt.info);
		result.plan =
		    make_uniq<LogicalCreate>(LogicalOperatorType::LOGICAL_CREATE_SEQUENCE, std::move(stmt.info), &schema);
		break;
	}
	case CatalogType::TABLE_MACRO_ENTRY: {
		auto &schema = BindCreateSchema(*stmt.info);
		result.plan =
		    make_uniq<LogicalCreate>(LogicalOperatorType::LOGICAL_CREATE_MACRO, std::move(stmt.info), &schema);
		break;
	}
	case CatalogType::MACRO_ENTRY: {
		auto &schema = BindCreateFunctionInfo(*stmt.info);
		result.plan =
		    make_uniq<LogicalCreate>(LogicalOperatorType::LOGICAL_CREATE_MACRO, std::move(stmt.info), &schema);
		break;
	}
	case CatalogType::INDEX_ENTRY: {
		auto &base = stmt.info->Cast<CreateIndexInfo>();

		// visit the table reference
		auto bound_table = Bind(*base.table);
		if (bound_table->type != TableReferenceType::BASE_TABLE) {
			throw BinderException("Can only create an index over a base table!");
		}
		auto &table_binding = bound_table->Cast<BoundBaseTableRef>();
		auto &table = table_binding.table;
		if (table.temporary) {
			stmt.info->temporary = true;
		}
		// create a plan over the bound table
		auto plan = CreatePlan(*bound_table);
		if (plan->type != LogicalOperatorType::LOGICAL_GET) {
			throw BinderException("Cannot create index on a view!");
		}

		result.plan = table.catalog.BindCreateIndex(*this, stmt, table, std::move(plan));
		break;
	}
	case CatalogType::TABLE_ENTRY: {
		auto &create_info = stmt.info->Cast<CreateTableInfo>();
		// If there is a foreign key constraint, resolve primary key column's index from primary key column's name
		reference_set_t<SchemaCatalogEntry> fk_schemas;
		for (idx_t i = 0; i < create_info.constraints.size(); i++) {
			auto &cond = create_info.constraints[i];
			if (cond->type != ConstraintType::FOREIGN_KEY) {
				continue;
			}
			auto &fk = cond->Cast<ForeignKeyConstraint>();
			if (fk.info.type != ForeignKeyType::FK_TYPE_FOREIGN_KEY_TABLE) {
				continue;
			}
			D_ASSERT(fk.info.pk_keys.empty());
			D_ASSERT(fk.info.fk_keys.empty());
			FindForeignKeyIndexes(create_info.columns, fk.fk_columns, fk.info.fk_keys);
			if (StringUtil::CIEquals(create_info.table, fk.info.table)) {
				// self-referential foreign key constraint
				fk.info.type = ForeignKeyType::FK_TYPE_SELF_REFERENCE_TABLE;
				FindMatchingPrimaryKeyColumns(create_info.columns, create_info.constraints, fk);
				FindForeignKeyIndexes(create_info.columns, fk.pk_columns, fk.info.pk_keys);
				CheckForeignKeyTypes(create_info.columns, create_info.columns, fk);
			} else {
				// have to resolve referenced table
				auto &pk_table_entry_ptr =
				    Catalog::GetEntry<TableCatalogEntry>(context, INVALID_CATALOG, fk.info.schema, fk.info.table);
				fk_schemas.insert(pk_table_entry_ptr.schema);
				FindMatchingPrimaryKeyColumns(pk_table_entry_ptr.GetColumns(), pk_table_entry_ptr.GetConstraints(), fk);
				FindForeignKeyIndexes(pk_table_entry_ptr.GetColumns(), fk.pk_columns, fk.info.pk_keys);
				CheckForeignKeyTypes(pk_table_entry_ptr.GetColumns(), create_info.columns, fk);
				auto &storage = pk_table_entry_ptr.GetStorage();
				auto index = storage.info->indexes.FindForeignKeyIndex(fk.info.pk_keys,
				                                                       ForeignKeyType::FK_TYPE_PRIMARY_KEY_TABLE);
				if (!index) {
					auto fk_column_names = StringUtil::Join(fk.pk_columns, ",");
					throw BinderException("Failed to create foreign key on %s(%s): no UNIQUE or PRIMARY KEY constraint "
					                      "present on these columns",
					                      pk_table_entry_ptr.name, fk_column_names);
				}
			}
			D_ASSERT(fk.info.pk_keys.size() == fk.info.fk_keys.size());
			D_ASSERT(fk.info.pk_keys.size() == fk.pk_columns.size());
			D_ASSERT(fk.info.fk_keys.size() == fk.fk_columns.size());
		}
		if (AnyConstraintReferencesGeneratedColumn(create_info)) {
			throw BinderException("Constraints on generated columns are not supported yet");
		}
		auto bound_info = BindCreateTableInfo(std::move(stmt.info));
		auto root = std::move(bound_info->query);
		for (auto &fk_schema : fk_schemas) {
			if (&fk_schema.get() != &bound_info->schema) {
				throw BinderException("Creating foreign keys across different schemas or catalogs is not supported");
			}
		}

		// create the logical operator
		auto &schema = bound_info->schema;
		auto create_table = make_uniq<LogicalCreateTable>(schema, std::move(bound_info));
		if (root) {
			// CREATE TABLE AS
			properties.return_type = StatementReturnType::CHANGED_ROWS;
			create_table->children.push_back(std::move(root));
		}
		result.plan = std::move(create_table);
		break;
	}
	case CatalogType::TYPE_ENTRY: {
		auto &schema = BindCreateSchema(*stmt.info);
		auto &create_type_info = stmt.info->Cast<CreateTypeInfo>();
		result.plan = make_uniq<LogicalCreate>(LogicalOperatorType::LOGICAL_CREATE_TYPE, std::move(stmt.info), &schema);
		if (create_type_info.query) {
			// CREATE TYPE mood AS ENUM (SELECT 'happy')
			auto query_obj = Bind(*create_type_info.query);
			auto query = std::move(query_obj.plan);
			create_type_info.query.reset();

			auto &sql_types = query_obj.types;
			if (sql_types.size() != 1) {
				// add cast expression?
				throw BinderException("The query must return a single column");
			}
			if (sql_types[0].id() != LogicalType::VARCHAR) {
				// push a projection casting to varchar
				vector<unique_ptr<Expression>> select_list;
				auto ref = make_uniq<BoundColumnRefExpression>(sql_types[0], query->GetColumnBindings()[0]);
				auto cast_expr = BoundCastExpression::AddCastToType(context, std::move(ref), LogicalType::VARCHAR);
				select_list.push_back(std::move(cast_expr));
				auto proj = make_uniq<LogicalProjection>(GenerateTableIndex(), std::move(select_list));
				proj->AddChild(std::move(query));
				query = std::move(proj);
			}

			result.plan->AddChild(std::move(query));
		} else if (create_type_info.type.id() == LogicalTypeId::USER) {
			// two cases:
			// 1: create a type with a non-existant type as source, catalog.GetType(...) will throw exception.
			// 2: create a type alias with a custom type.
			// eg. CREATE TYPE a AS INT; CREATE TYPE b AS a;
			// We set b to be an alias for the underlying type of a
			auto inner_type = Catalog::GetType(context, schema.catalog.GetName(), schema.name,
			                                   UserType::GetTypeName(create_type_info.type));
			// clear to nullptr, we don't need this
			EnumType::SetCatalog(inner_type, nullptr);
			inner_type.SetAlias(create_type_info.name);
			create_type_info.type = inner_type;
		}
		break;
	}
	case CatalogType::DATABASE_ENTRY: {
		// not supported in DuckDB yet but allow extensions to intercept and implement this functionality
		auto &base = stmt.info->Cast<CreateDatabaseInfo>();
		string database_name = base.name;
		string source_path = base.path;

		auto &config = DBConfig::GetConfig(context);

		if (config.storage_extensions.empty()) {
			throw NotImplementedException("CREATE DATABASE not supported in DuckDB yet");
		}
		// for now assume only one storage extension provides the custom create_database impl
		for (auto &extension_entry : config.storage_extensions) {
			if (extension_entry.second->create_database != nullptr) {
				auto &storage_extension = extension_entry.second;
				auto create_database_function_ref = storage_extension->create_database(
				    storage_extension->storage_info.get(), context, database_name, source_path);
				if (create_database_function_ref) {
					auto bound_create_database_func = Bind(*create_database_function_ref);
					result.plan = CreatePlan(*bound_create_database_func);
					break;
				}
			}
		}
		break;
	}
	default:
		throw Exception("Unrecognized type!");
	}
	properties.return_type = StatementReturnType::NOTHING;
	properties.allow_stream_result = false;
	return result;
}

} // namespace duckdb






















#include <algorithm>

namespace duckdb {

static void CreateColumnDependencyManager(BoundCreateTableInfo &info) {
	auto &base = info.base->Cast<CreateTableInfo>();
	for (auto &col : base.columns.Logical()) {
		if (!col.Generated()) {
			continue;
		}
		info.column_dependency_manager.AddGeneratedColumn(col, base.columns);
	}
}

static void BindCheckConstraint(Binder &binder, BoundCreateTableInfo &info, const unique_ptr<Constraint> &cond) {
	auto &base = info.base->Cast<CreateTableInfo>();

	auto bound_constraint = make_uniq<BoundCheckConstraint>();
	// check constraint: bind the expression
	CheckBinder check_binder(binder, binder.context, base.table, base.columns, bound_constraint->bound_columns);
	auto &check = cond->Cast<CheckConstraint>();
	// create a copy of the unbound expression because the binding destroys the constraint
	auto unbound_expression = check.expression->Copy();
	// now bind the constraint and create a new BoundCheckConstraint
	bound_constraint->expression = check_binder.Bind(check.expression);
	info.bound_constraints.push_back(std::move(bound_constraint));
	// move the unbound constraint back into the original check expression
	check.expression = std::move(unbound_expression);
}

static void BindConstraints(Binder &binder, BoundCreateTableInfo &info) {
	auto &base = info.base->Cast<CreateTableInfo>();

	bool has_primary_key = false;
	logical_index_set_t not_null_columns;
	vector<LogicalIndex> primary_keys;
	for (idx_t i = 0; i < base.constraints.size(); i++) {
		auto &cond = base.constraints[i];
		switch (cond->type) {
		case ConstraintType::CHECK: {
			BindCheckConstraint(binder, info, cond);
			break;
		}
		case ConstraintType::NOT_NULL: {
			auto &not_null = cond->Cast<NotNullConstraint>();
			auto &col = base.columns.GetColumn(LogicalIndex(not_null.index));
			info.bound_constraints.push_back(make_uniq<BoundNotNullConstraint>(PhysicalIndex(col.StorageOid())));
			not_null_columns.insert(not_null.index);
			break;
		}
		case ConstraintType::UNIQUE: {
			auto &unique = cond->Cast<UniqueConstraint>();
			// have to resolve columns of the unique constraint
			vector<LogicalIndex> keys;
			logical_index_set_t key_set;
			if (unique.index.index != DConstants::INVALID_INDEX) {
				D_ASSERT(unique.index.index < base.columns.LogicalColumnCount());
				// unique constraint is given by single index
				unique.columns.push_back(base.columns.GetColumn(unique.index).Name());
				keys.push_back(unique.index);
				key_set.insert(unique.index);
			} else {
				// unique constraint is given by list of names
				// have to resolve names
				D_ASSERT(!unique.columns.empty());
				for (auto &keyname : unique.columns) {
					if (!base.columns.ColumnExists(keyname)) {
						throw ParserException("column \"%s\" named in key does not exist", keyname);
					}
					auto &column = base.columns.GetColumn(keyname);
					auto column_index = column.Logical();
					if (key_set.find(column_index) != key_set.end()) {
						throw ParserException("column \"%s\" appears twice in "
						                      "primary key constraint",
						                      keyname);
					}
					keys.push_back(column_index);
					key_set.insert(column_index);
				}
			}

			if (unique.is_primary_key) {
				// we can only have one primary key per table
				if (has_primary_key) {
					throw ParserException("table \"%s\" has more than one primary key", base.table);
				}
				has_primary_key = true;
				primary_keys = keys;
			}
			info.bound_constraints.push_back(
			    make_uniq<BoundUniqueConstraint>(std::move(keys), std::move(key_set), unique.is_primary_key));
			break;
		}
		case ConstraintType::FOREIGN_KEY: {
			auto &fk = cond->Cast<ForeignKeyConstraint>();
			D_ASSERT((fk.info.type == ForeignKeyType::FK_TYPE_FOREIGN_KEY_TABLE && !fk.info.pk_keys.empty()) ||
			         (fk.info.type == ForeignKeyType::FK_TYPE_PRIMARY_KEY_TABLE && !fk.info.pk_keys.empty()) ||
			         fk.info.type == ForeignKeyType::FK_TYPE_SELF_REFERENCE_TABLE);
			physical_index_set_t fk_key_set, pk_key_set;
			for (idx_t i = 0; i < fk.info.pk_keys.size(); i++) {
				if (pk_key_set.find(fk.info.pk_keys[i]) != pk_key_set.end()) {
					throw BinderException("Duplicate primary key referenced in FOREIGN KEY constraint");
				}
				pk_key_set.insert(fk.info.pk_keys[i]);
			}
			for (idx_t i = 0; i < fk.info.fk_keys.size(); i++) {
				if (fk_key_set.find(fk.info.fk_keys[i]) != fk_key_set.end()) {
					throw BinderException("Duplicate key specified in FOREIGN KEY constraint");
				}
				fk_key_set.insert(fk.info.fk_keys[i]);
			}
			info.bound_constraints.push_back(
			    make_uniq<BoundForeignKeyConstraint>(fk.info, std::move(pk_key_set), std::move(fk_key_set)));
			break;
		}
		default:
			throw NotImplementedException("unrecognized constraint type in bind");
		}
	}
	if (has_primary_key) {
		// if there is a primary key index, also create a NOT NULL constraint for each of the columns
		for (auto &column_index : primary_keys) {
			if (not_null_columns.count(column_index)) {
				//! No need to create a NotNullConstraint, it's already present
				continue;
			}
			auto physical_index = base.columns.LogicalToPhysical(column_index);
			base.constraints.push_back(make_uniq<NotNullConstraint>(column_index));
			info.bound_constraints.push_back(make_uniq<BoundNotNullConstraint>(physical_index));
		}
	}
}

void Binder::BindGeneratedColumns(BoundCreateTableInfo &info) {
	auto &base = info.base->Cast<CreateTableInfo>();

	vector<string> names;
	vector<LogicalType> types;

	D_ASSERT(base.type == CatalogType::TABLE_ENTRY);
	for (auto &col : base.columns.Logical()) {
		names.push_back(col.Name());
		types.push_back(col.Type());
	}
	auto table_index = GenerateTableIndex();

	// Create a new binder because we dont need (or want) these bindings in this scope
	auto binder = Binder::CreateBinder(context);
	binder->bind_context.AddGenericBinding(table_index, base.table, names, types);
	auto expr_binder = ExpressionBinder(*binder, context);
	string ignore;
	auto table_binding = binder->bind_context.GetBinding(base.table, ignore);
	D_ASSERT(table_binding && ignore.empty());

	auto bind_order = info.column_dependency_manager.GetBindOrder(base.columns);
	logical_index_set_t bound_indices;

	while (!bind_order.empty()) {
		auto i = bind_order.top();
		bind_order.pop();
		auto &col = base.columns.GetColumnMutable(i);

		//! Already bound this previously
		//! This can not be optimized out of the GetBindOrder function
		//! These occurrences happen because we need to make sure that ALL dependencies of a column are resolved before
		//! it gets resolved
		if (bound_indices.count(i)) {
			continue;
		}
		D_ASSERT(col.Generated());
		auto expression = col.GeneratedExpression().Copy();

		auto bound_expression = expr_binder.Bind(expression);
		D_ASSERT(bound_expression);
		D_ASSERT(!bound_expression->HasSubquery());
		if (col.Type().id() == LogicalTypeId::ANY) {
			// Do this before changing the type, so we know it's the first time the type is set
			col.ChangeGeneratedExpressionType(bound_expression->return_type);
			col.SetType(bound_expression->return_type);

			// Update the type in the binding, for future expansions
			string ignore;
			table_binding->types[i.index] = col.Type();
		}
		bound_indices.insert(i);
	}
}

void Binder::BindDefaultValues(const ColumnList &columns, vector<unique_ptr<Expression>> &bound_defaults) {
	for (auto &column : columns.Physical()) {
		unique_ptr<Expression> bound_default;
		if (column.DefaultValue()) {
			// we bind a copy of the DEFAULT value because binding is destructive
			// and we want to keep the original expression around for serialization
			auto default_copy = column.DefaultValue()->Copy();
			ConstantBinder default_binder(*this, context, "DEFAULT value");
			default_binder.target_type = column.Type();
			bound_default = default_binder.Bind(default_copy);
		} else {
			// no default value specified: push a default value of constant null
			bound_default = make_uniq<BoundConstantExpression>(Value(column.Type()));
		}
		bound_defaults.push_back(std::move(bound_default));
	}
}

static void ExtractExpressionDependencies(Expression &expr, DependencyList &dependencies) {
	if (expr.type == ExpressionType::BOUND_FUNCTION) {
		auto &function = expr.Cast<BoundFunctionExpression>();
		if (function.function.dependency) {
			function.function.dependency(function, dependencies);
		}
	}
	ExpressionIterator::EnumerateChildren(
	    expr, [&](Expression &child) { ExtractExpressionDependencies(child, dependencies); });
}

static void ExtractDependencies(BoundCreateTableInfo &info) {
	for (auto &default_value : info.bound_defaults) {
		if (default_value) {
			ExtractExpressionDependencies(*default_value, info.dependencies);
		}
	}
	for (auto &constraint : info.bound_constraints) {
		if (constraint->type == ConstraintType::CHECK) {
			auto &bound_check = constraint->Cast<BoundCheckConstraint>();
			ExtractExpressionDependencies(*bound_check.expression, info.dependencies);
		}
	}
}
unique_ptr<BoundCreateTableInfo> Binder::BindCreateTableInfo(unique_ptr<CreateInfo> info, SchemaCatalogEntry &schema) {
	auto &base = info->Cast<CreateTableInfo>();
	auto result = make_uniq<BoundCreateTableInfo>(schema, std::move(info));
	if (base.query) {
		// construct the result object
		auto query_obj = Bind(*base.query);
		base.query.reset();
		result->query = std::move(query_obj.plan);

		// construct the set of columns based on the names and types of the query
		auto &names = query_obj.names;
		auto &sql_types = query_obj.types;
		D_ASSERT(names.size() == sql_types.size());
		base.columns.SetAllowDuplicates(true);
		for (idx_t i = 0; i < names.size(); i++) {
			base.columns.AddColumn(ColumnDefinition(names[i], sql_types[i]));
		}
		CreateColumnDependencyManager(*result);
		// bind the generated column expressions
		BindGeneratedColumns(*result);
	} else {
		CreateColumnDependencyManager(*result);
		// bind the generated column expressions
		BindGeneratedColumns(*result);
		// bind any constraints
		BindConstraints(*this, *result);
		// bind the default values
		BindDefaultValues(base.columns, result->bound_defaults);
	}
	// extract dependencies from any default values or CHECK constraints
	ExtractDependencies(*result);

	if (base.columns.PhysicalColumnCount() == 0) {
		throw BinderException("Creating a table without physical (non-generated) columns is not supported");
	}
	// bind collations to detect any unsupported collation errors
	for (idx_t i = 0; i < base.columns.PhysicalColumnCount(); i++) {
		auto &column = base.columns.GetColumnMutable(PhysicalIndex(i));
		if (column.Type().id() == LogicalTypeId::VARCHAR) {
			ExpressionBinder::TestCollation(context, StringType::GetCollation(column.Type()));
		}
		BindLogicalType(context, column.TypeMutable(), &result->schema.catalog);
		// We add a catalog dependency
		auto type_dependency = EnumType::GetCatalog(column.Type());
		if (type_dependency) {
			// Only if the USER comes from a create type
			result->dependencies.AddDependency(*type_dependency);
		}
	}
	result->dependencies.VerifyDependencies(schema.catalog, result->Base().table);
	properties.allow_stream_result = false;
	return result;
}

unique_ptr<BoundCreateTableInfo> Binder::BindCreateTableInfo(unique_ptr<CreateInfo> info) {
	auto &base = info->Cast<CreateTableInfo>();
	auto &schema = BindCreateSchema(base);
	return BindCreateTableInfo(std::move(info), schema);
}

vector<unique_ptr<Expression>> Binder::BindCreateIndexExpressions(TableCatalogEntry &table, CreateIndexInfo &info) {
	auto index_binder = IndexBinder(*this, this->context, &table, &info);
	vector<unique_ptr<Expression>> expressions;
	expressions.reserve(info.expressions.size());
	for (auto &expr : info.expressions) {
		expressions.push_back(index_binder.Bind(expr));
	}

	return expressions;
}

} // namespace duckdb












namespace duckdb {

BoundStatement Binder::Bind(DeleteStatement &stmt) {
	BoundStatement result;

	// visit the table reference
	auto bound_table = Bind(*stmt.table);
	if (bound_table->type != TableReferenceType::BASE_TABLE) {
		throw BinderException("Can only delete from base table!");
	}
	auto &table_binding = bound_table->Cast<BoundBaseTableRef>();
	auto &table = table_binding.table;

	auto root = CreatePlan(*bound_table);
	auto &get = root->Cast<LogicalGet>();
	D_ASSERT(root->type == LogicalOperatorType::LOGICAL_GET);

	if (!table.temporary) {
		// delete from persistent table: not read only!
		properties.modified_databases.insert(table.catalog.GetName());
	}

	// Add CTEs as bindable
	AddCTEMap(stmt.cte_map);

	// plan any tables from the various using clauses
	if (!stmt.using_clauses.empty()) {
		unique_ptr<LogicalOperator> child_operator;
		for (auto &using_clause : stmt.using_clauses) {
			// bind the using clause
			auto using_binder = Binder::CreateBinder(context, this);
			auto bound_node = using_binder->Bind(*using_clause);
			auto op = CreatePlan(*bound_node);
			if (child_operator) {
				// already bound a child: create a cross product to unify the two
				child_operator = LogicalCrossProduct::Create(std::move(child_operator), std::move(op));
			} else {
				child_operator = std::move(op);
			}
			bind_context.AddContext(std::move(using_binder->bind_context));
		}
		if (child_operator) {
			root = LogicalCrossProduct::Create(std::move(root), std::move(child_operator));
		}
	}

	// project any additional columns required for the condition
	unique_ptr<Expression> condition;
	if (stmt.condition) {
		WhereBinder binder(*this, context);
		condition = binder.Bind(stmt.condition);

		PlanSubqueries(condition, root);
		auto filter = make_uniq<LogicalFilter>(std::move(condition));
		filter->AddChild(std::move(root));
		root = std::move(filter);
	}
	// create the delete node
	auto del = make_uniq<LogicalDelete>(table, GenerateTableIndex());
	del->AddChild(std::move(root));

	// set up the delete expression
	del->expressions.push_back(make_uniq<BoundColumnRefExpression>(
	    LogicalType::ROW_TYPE, ColumnBinding(get.table_index, get.column_ids.size())));
	get.column_ids.push_back(COLUMN_IDENTIFIER_ROW_ID);

	if (!stmt.returning_list.empty()) {
		del->return_chunk = true;

		auto update_table_index = GenerateTableIndex();
		del->table_index = update_table_index;

		unique_ptr<LogicalOperator> del_as_logicaloperator = std::move(del);
		return BindReturning(std::move(stmt.returning_list), table, stmt.table->alias, update_table_index,
		                     std::move(del_as_logicaloperator), std::move(result));
	}
	result.plan = std::move(del);
	result.names = {"Count"};
	result.types = {LogicalType::BIGINT};
	properties.allow_stream_result = false;
	properties.return_type = StatementReturnType::CHANGED_ROWS;

	return result;
}

} // namespace duckdb





namespace duckdb {

BoundStatement Binder::Bind(DetachStatement &stmt) {
	BoundStatement result;

	result.plan = make_uniq<LogicalSimple>(LogicalOperatorType::LOGICAL_DETACH, std::move(stmt.info));
	result.names = {"Success"};
	result.types = {LogicalType::BOOLEAN};
	properties.allow_stream_result = false;
	properties.return_type = StatementReturnType::NOTHING;
	return result;
}

} // namespace duckdb











namespace duckdb {

BoundStatement Binder::Bind(DropStatement &stmt) {
	BoundStatement result;

	switch (stmt.info->type) {
	case CatalogType::PREPARED_STATEMENT:
		// dropping prepared statements is always possible
		// it also does not require a valid transaction
		properties.requires_valid_transaction = false;
		break;
	case CatalogType::SCHEMA_ENTRY: {
		// dropping a schema is never read-only because there are no temporary schemas
		auto &catalog = Catalog::GetCatalog(context, stmt.info->catalog);
		properties.modified_databases.insert(catalog.GetName());
		break;
	}
	case CatalogType::VIEW_ENTRY:
	case CatalogType::SEQUENCE_ENTRY:
	case CatalogType::MACRO_ENTRY:
	case CatalogType::TABLE_MACRO_ENTRY:
	case CatalogType::INDEX_ENTRY:
	case CatalogType::TABLE_ENTRY:
	case CatalogType::TYPE_ENTRY: {
		BindSchemaOrCatalog(stmt.info->catalog, stmt.info->schema);
		auto entry = Catalog::GetEntry(context, stmt.info->type, stmt.info->catalog, stmt.info->schema, stmt.info->name,
		                               OnEntryNotFound::RETURN_NULL);
		if (!entry) {
			break;
		}
		stmt.info->catalog = entry->ParentCatalog().GetName();
		if (!entry->temporary) {
			// we can only drop temporary tables in read-only mode
			properties.modified_databases.insert(stmt.info->catalog);
		}
		stmt.info->schema = entry->ParentSchema().name;
		break;
	}
	case CatalogType::DATABASE_ENTRY: {
		auto &base = (DropInfo &)*stmt.info;
		string database_name = base.name;

		auto &config = DBConfig::GetConfig(context);
		// for now assume only one storage extension provides the custom drop_database impl
		for (auto &extension_entry : config.storage_extensions) {
			if (extension_entry.second->drop_database == nullptr) {
				continue;
			}
			auto &storage_extension = extension_entry.second;
			auto drop_database_function_ref =
			    storage_extension->drop_database(storage_extension->storage_info.get(), context, database_name);
			if (drop_database_function_ref) {
				auto bound_drop_database_func = Bind(*drop_database_function_ref);
				result.plan = CreatePlan(*bound_drop_database_func);
				result.names = {"Success"};
				result.types = {LogicalType::BIGINT};
				properties.allow_stream_result = false;
				properties.return_type = StatementReturnType::NOTHING;
				return result;
			}
		}
		throw BinderException("Drop is not supported for this database!");
	}
	default:
		throw BinderException("Unknown catalog type for drop statement!");
	}
	result.plan = make_uniq<LogicalSimple>(LogicalOperatorType::LOGICAL_DROP, std::move(stmt.info));
	result.names = {"Success"};
	result.types = {LogicalType::BOOLEAN};
	properties.allow_stream_result = false;
	properties.return_type = StatementReturnType::NOTHING;
	return result;
}

} // namespace duckdb









namespace duckdb {

BoundStatement Binder::Bind(ExecuteStatement &stmt) {
	auto parameter_count = stmt.n_param;

	// bind the prepared statement
	auto &client_data = ClientData::Get(context);

	auto entry = client_data.prepared_statements.find(stmt.name);
	if (entry == client_data.prepared_statements.end()) {
		throw BinderException("Prepared statement \"%s\" does not exist", stmt.name);
	}

	// check if we need to rebind the prepared statement
	// this happens if the catalog changes, since in this case e.g. tables we relied on may have been deleted
	auto prepared = entry->second;

	// bind any supplied parameters
	vector<Value> bind_values;
	auto constant_binder = Binder::CreateBinder(context);
	constant_binder->SetCanContainNulls(true);
	for (idx_t i = 0; i < stmt.values.size(); i++) {
		ConstantBinder cbinder(*constant_binder, context, "EXECUTE statement");
		auto bound_expr = cbinder.Bind(stmt.values[i]);

		Value value = ExpressionExecutor::EvaluateScalar(context, *bound_expr, true);
		bind_values.push_back(std::move(value));
	}
	unique_ptr<LogicalOperator> rebound_plan;
	if (prepared->RequireRebind(context, bind_values)) {
		// catalog was modified or statement does not have clear types: rebind the statement before running the execute
		Planner prepared_planner(context);
		for (idx_t i = 0; i < bind_values.size(); i++) {
			prepared_planner.parameter_data.emplace_back(bind_values[i]);
		}
		prepared = prepared_planner.PrepareSQLStatement(entry->second->unbound_statement->Copy());
		rebound_plan = std::move(prepared_planner.plan);
		D_ASSERT(prepared->properties.bound_all_parameters);
		this->bound_tables = prepared_planner.binder->bound_tables;
	}
	// copy the properties of the prepared statement into the planner
	this->properties = prepared->properties;
	this->properties.parameter_count = parameter_count;
	BoundStatement result;
	result.names = prepared->names;
	result.types = prepared->types;

	prepared->Bind(std::move(bind_values));
	if (rebound_plan) {
		auto execute_plan = make_uniq<LogicalExecute>(std::move(prepared));
		execute_plan->children.push_back(std::move(rebound_plan));
		result.plan = std::move(execute_plan);
	} else {
		result.plan = make_uniq<LogicalExecute>(std::move(prepared));
	}
	return result;
}

} // namespace duckdb




namespace duckdb {

BoundStatement Binder::Bind(ExplainStatement &stmt) {
	BoundStatement result;

	// bind the underlying statement
	auto plan = Bind(*stmt.stmt);
	// get the unoptimized logical plan, and create the explain statement
	auto logical_plan_unopt = plan.plan->ToString();
	auto explain = make_uniq<LogicalExplain>(std::move(plan.plan), stmt.explain_type);
	explain->logical_plan_unopt = logical_plan_unopt;

	result.plan = std::move(explain);
	result.names = {"explain_key", "explain_value"};
	result.types = {LogicalType::VARCHAR, LogicalType::VARCHAR};
	properties.return_type = StatementReturnType::QUERY_RESULT;
	return result;
}

} // namespace duckdb















#include <algorithm>

namespace duckdb {

//! Sanitizes a string to have only low case chars and underscores
string SanitizeExportIdentifier(const string &str) {
	// Copy the original string to result
	string result(str);

	for (idx_t i = 0; i < str.length(); ++i) {
		auto c = str[i];
		if (c >= 'a' && c <= 'z') {
			// If it is lower case just continue
			continue;
		}

		if (c >= 'A' && c <= 'Z') {
			// To lowercase
			result[i] = tolower(c);
		} else {
			// Substitute to underscore
			result[i] = '_';
		}
	}

	return result;
}

bool IsExistMainKeyTable(string &table_name, vector<reference<TableCatalogEntry>> &unordered) {
	for (idx_t i = 0; i < unordered.size(); i++) {
		if (unordered[i].get().name == table_name) {
			return true;
		}
	}
	return false;
}

void ScanForeignKeyTable(vector<reference<TableCatalogEntry>> &ordered, vector<reference<TableCatalogEntry>> &unordered,
                         bool move_only_pk_table) {
	for (auto i = unordered.begin(); i != unordered.end();) {
		auto table_entry = *i;
		bool move_to_ordered = true;
		auto &constraints = table_entry.get().GetConstraints();
		for (idx_t j = 0; j < constraints.size(); j++) {
			auto &cond = constraints[j];
			if (cond->type == ConstraintType::FOREIGN_KEY) {
				auto &fk = cond->Cast<ForeignKeyConstraint>();
				if ((move_only_pk_table && fk.info.type == ForeignKeyType::FK_TYPE_FOREIGN_KEY_TABLE) ||
				    (!move_only_pk_table && fk.info.type == ForeignKeyType::FK_TYPE_FOREIGN_KEY_TABLE &&
				     IsExistMainKeyTable(fk.info.table, unordered))) {
					move_to_ordered = false;
					break;
				}
			}
		}
		if (move_to_ordered) {
			ordered.push_back(table_entry);
			i = unordered.erase(i);
		} else {
			i++;
		}
	}
}

void ReorderTableEntries(vector<reference<TableCatalogEntry>> &tables) {
	vector<reference<TableCatalogEntry>> ordered;
	vector<reference<TableCatalogEntry>> unordered = tables;
	ScanForeignKeyTable(ordered, unordered, true);
	while (!unordered.empty()) {
		ScanForeignKeyTable(ordered, unordered, false);
	}
	tables = ordered;
}

string CreateFileName(const string &id_suffix, TableCatalogEntry &table, const string &extension) {
	auto name = SanitizeExportIdentifier(table.name);
	if (table.schema.name == DEFAULT_SCHEMA) {
		return StringUtil::Format("%s%s.%s", name, id_suffix, extension);
	}
	auto schema = SanitizeExportIdentifier(table.schema.name);
	return StringUtil::Format("%s_%s%s.%s", schema, name, id_suffix, extension);
}

BoundStatement Binder::Bind(ExportStatement &stmt) {
	// COPY TO a file
	auto &config = DBConfig::GetConfig(context);
	if (!config.options.enable_external_access) {
		throw PermissionException("COPY TO is disabled through configuration");
	}
	BoundStatement result;
	result.types = {LogicalType::BOOLEAN};
	result.names = {"Success"};

	// lookup the format in the catalog
	auto &copy_function =
	    Catalog::GetEntry<CopyFunctionCatalogEntry>(context, INVALID_CATALOG, DEFAULT_SCHEMA, stmt.info->format);
	if (!copy_function.function.copy_to_bind && !copy_function.function.plan) {
		throw NotImplementedException("COPY TO is not supported for FORMAT \"%s\"", stmt.info->format);
	}

	// gather a list of all the tables
	string catalog = stmt.database.empty() ? INVALID_CATALOG : stmt.database;
	vector<reference<TableCatalogEntry>> tables;
	auto schemas = Catalog::GetSchemas(context, catalog);
	for (auto &schema : schemas) {
		schema.get().Scan(context, CatalogType::TABLE_ENTRY, [&](CatalogEntry &entry) {
			if (entry.type == CatalogType::TABLE_ENTRY) {
				tables.push_back(entry.Cast<TableCatalogEntry>());
			}
		});
	}

	// reorder tables because of foreign key constraint
	ReorderTableEntries(tables);

	// now generate the COPY statements for each of the tables
	auto &fs = FileSystem::GetFileSystem(context);
	unique_ptr<LogicalOperator> child_operator;

	BoundExportData exported_tables;

	unordered_set<string> table_name_index;
	for (auto &t : tables) {
		auto &table = t.get();
		auto info = make_uniq<CopyInfo>();
		// we copy the options supplied to the EXPORT
		info->format = stmt.info->format;
		info->options = stmt.info->options;
		// set up the file name for the COPY TO

		idx_t id = 0;
		while (true) {
			string id_suffix = id == 0 ? string() : "_" + to_string(id);
			auto name = CreateFileName(id_suffix, table, copy_function.function.extension);
			auto directory = stmt.info->file_path;
			auto full_path = fs.JoinPath(directory, name);
			info->file_path = full_path;
			auto insert_result = table_name_index.insert(info->file_path);
			if (insert_result.second == true) {
				// this name was not yet taken: take it
				break;
			}
			id++;
		}
		info->is_from = false;
		info->catalog = catalog;
		info->schema = table.schema.name;
		info->table = table.name;

		// We can not export generated columns
		for (auto &col : table.GetColumns().Physical()) {
			info->select_list.push_back(col.GetName());
		}

		ExportedTableData exported_data;
		exported_data.database_name = catalog;
		exported_data.table_name = info->table;
		exported_data.schema_name = info->schema;

		exported_data.file_path = info->file_path;

		ExportedTableInfo table_info(table, std::move(exported_data));
		exported_tables.data.push_back(table_info);
		id++;

		// generate the copy statement and bind it
		CopyStatement copy_stmt;
		copy_stmt.info = std::move(info);

		auto copy_binder = Binder::CreateBinder(context, this);
		auto bound_statement = copy_binder->Bind(copy_stmt);
		if (child_operator) {
			// use UNION ALL to combine the individual copy statements into a single node
			auto copy_union =
			    make_uniq<LogicalSetOperation>(GenerateTableIndex(), 1, std::move(child_operator),
			                                   std::move(bound_statement.plan), LogicalOperatorType::LOGICAL_UNION);
			child_operator = std::move(copy_union);
		} else {
			child_operator = std::move(bound_statement.plan);
		}
	}

	// try to create the directory, if it doesn't exist yet
	// a bit hacky to do it here, but we need to create the directory BEFORE the copy statements run
	if (!fs.DirectoryExists(stmt.info->file_path)) {
		fs.CreateDirectory(stmt.info->file_path);
	}

	// create the export node
	auto export_node = make_uniq<LogicalExport>(copy_function.function, std::move(stmt.info), exported_tables);

	if (child_operator) {
		export_node->children.push_back(std::move(child_operator));
	}

	result.plan = std::move(export_node);
	properties.allow_stream_result = false;
	properties.return_type = StatementReturnType::NOTHING;
	return result;
}

} // namespace duckdb




namespace duckdb {

BoundStatement Binder::Bind(ExtensionStatement &stmt) {
	BoundStatement result;

	// perform the planning of the function
	D_ASSERT(stmt.extension.plan_function);
	auto parse_result =
	    stmt.extension.plan_function(stmt.extension.parser_info.get(), context, std::move(stmt.parse_data));

	properties.modified_databases = parse_result.modified_databases;
	properties.requires_valid_transaction = parse_result.requires_valid_transaction;
	properties.return_type = parse_result.return_type;

	// create the plan as a scan of the given table function
	result.plan = BindTableFunction(parse_result.function, std::move(parse_result.parameters));
	D_ASSERT(result.plan->type == LogicalOperatorType::LOGICAL_GET);
	auto &get = result.plan->Cast<LogicalGet>();
	result.names = get.names;
	result.types = get.returned_types;
	get.column_ids.clear();
	for (idx_t i = 0; i < get.returned_types.size(); i++) {
		get.column_ids.push_back(i);
	}
	return result;
}

} // namespace duckdb





























namespace duckdb {

static void CheckInsertColumnCountMismatch(int64_t expected_columns, int64_t result_columns, bool columns_provided,
                                           const char *tname) {
	if (result_columns != expected_columns) {
		string msg = StringUtil::Format(!columns_provided ? "table %s has %lld columns but %lld values were supplied"
		                                                  : "Column name/value mismatch for insert on %s: "
		                                                    "expected %lld columns but %lld values were supplied",
		                                tname, expected_columns, result_columns);
		throw BinderException(msg);
	}
}

unique_ptr<ParsedExpression> ExpandDefaultExpression(const ColumnDefinition &column) {
	if (column.DefaultValue()) {
		return column.DefaultValue()->Copy();
	} else {
		return make_uniq<ConstantExpression>(Value(column.Type()));
	}
}

void ReplaceDefaultExpression(unique_ptr<ParsedExpression> &expr, const ColumnDefinition &column) {
	D_ASSERT(expr->type == ExpressionType::VALUE_DEFAULT);
	expr = ExpandDefaultExpression(column);
}

void QualifyColumnReferences(unique_ptr<ParsedExpression> &expr, const string &table_name) {
	// To avoid ambiguity with 'excluded', we explicitly qualify all column references
	if (expr->type == ExpressionType::COLUMN_REF) {
		auto &column_ref = expr->Cast<ColumnRefExpression>();
		if (column_ref.IsQualified()) {
			return;
		}
		auto column_name = column_ref.GetColumnName();
		expr = make_uniq<ColumnRefExpression>(column_name, table_name);
	}
	ParsedExpressionIterator::EnumerateChildren(
	    *expr, [&](unique_ptr<ParsedExpression> &child) { QualifyColumnReferences(child, table_name); });
}

// Replace binding.table_index with 'dest' if it's 'source'
void ReplaceColumnBindings(Expression &expr, idx_t source, idx_t dest) {
	if (expr.type == ExpressionType::BOUND_COLUMN_REF) {
		auto &bound_columnref = expr.Cast<BoundColumnRefExpression>();
		if (bound_columnref.binding.table_index == source) {
			bound_columnref.binding.table_index = dest;
		}
	}
	ExpressionIterator::EnumerateChildren(
	    expr, [&](unique_ptr<Expression> &child) { ReplaceColumnBindings(*child, source, dest); });
}

void Binder::BindDoUpdateSetExpressions(const string &table_alias, LogicalInsert &insert, UpdateSetInfo &set_info,
                                        TableCatalogEntry &table, TableStorageInfo &storage_info) {
	D_ASSERT(insert.children.size() == 1);
	D_ASSERT(insert.children[0]->type == LogicalOperatorType::LOGICAL_PROJECTION);

	vector<column_t> logical_column_ids;
	vector<string> column_names;
	D_ASSERT(set_info.columns.size() == set_info.expressions.size());

	for (idx_t i = 0; i < set_info.columns.size(); i++) {
		auto &colname = set_info.columns[i];
		auto &expr = set_info.expressions[i];
		if (!table.ColumnExists(colname)) {
			throw BinderException("Referenced update column %s not found in table!", colname);
		}
		auto &column = table.GetColumn(colname);
		if (column.Generated()) {
			throw BinderException("Cant update column \"%s\" because it is a generated column!", column.Name());
		}
		if (std::find(insert.set_columns.begin(), insert.set_columns.end(), column.Physical()) !=
		    insert.set_columns.end()) {
			throw BinderException("Multiple assignments to same column \"%s\"", colname);
		}
		insert.set_columns.push_back(column.Physical());
		logical_column_ids.push_back(column.Oid());
		insert.set_types.push_back(column.Type());
		column_names.push_back(colname);
		if (expr->type == ExpressionType::VALUE_DEFAULT) {
			expr = ExpandDefaultExpression(column);
		}
		UpdateBinder binder(*this, context);
		binder.target_type = column.Type();

		// Avoid ambiguity issues
		QualifyColumnReferences(expr, table_alias);

		auto bound_expr = binder.Bind(expr);
		D_ASSERT(bound_expr);
		if (bound_expr->expression_class == ExpressionClass::BOUND_SUBQUERY) {
			throw BinderException("Expression in the DO UPDATE SET clause can not be a subquery");
		}

		insert.expressions.push_back(std::move(bound_expr));
	}

	// Figure out which columns are indexed on
	unordered_set<column_t> indexed_columns;
	for (auto &index : storage_info.index_info) {
		for (auto &column_id : index.column_set) {
			indexed_columns.insert(column_id);
		}
	}

	// Verify that none of the columns that are targeted with a SET expression are indexed on
	for (idx_t i = 0; i < logical_column_ids.size(); i++) {
		auto &column = logical_column_ids[i];
		if (indexed_columns.count(column)) {
			throw BinderException("Can not assign to column '%s' because it has a UNIQUE/PRIMARY KEY constraint",
			                      column_names[i]);
		}
	}
}

unique_ptr<UpdateSetInfo> CreateSetInfoForReplace(TableCatalogEntry &table, InsertStatement &insert,
                                                  TableStorageInfo &storage_info) {
	auto set_info = make_uniq<UpdateSetInfo>();

	auto &columns = set_info->columns;
	// Figure out which columns are indexed on

	unordered_set<column_t> indexed_columns;
	for (auto &index : storage_info.index_info) {
		for (auto &column_id : index.column_set) {
			indexed_columns.insert(column_id);
		}
	}

	auto &column_list = table.GetColumns();
	if (insert.columns.empty()) {
		for (auto &column : column_list.Physical()) {
			auto &name = column.Name();
			// FIXME: can these column names be aliased somehow?
			if (indexed_columns.count(column.Oid())) {
				continue;
			}
			columns.push_back(name);
		}
	} else {
		// a list of columns was explicitly supplied, only update those
		for (auto &name : insert.columns) {
			auto &column = column_list.GetColumn(name);
			if (indexed_columns.count(column.Oid())) {
				continue;
			}
			columns.push_back(name);
		}
	}

	// Create 'excluded' qualified column references of these columns
	for (auto &column : columns) {
		set_info->expressions.push_back(make_uniq<ColumnRefExpression>(column, "excluded"));
	}

	return set_info;
}

void Binder::BindOnConflictClause(LogicalInsert &insert, TableCatalogEntry &table, InsertStatement &stmt) {
	if (!stmt.on_conflict_info) {
		insert.action_type = OnConflictAction::THROW;
		return;
	}
	D_ASSERT(stmt.table_ref->type == TableReferenceType::BASE_TABLE);

	// visit the table reference
	auto bound_table = Bind(*stmt.table_ref);
	if (bound_table->type != TableReferenceType::BASE_TABLE) {
		throw BinderException("Can only update base table!");
	}

	auto &table_ref = stmt.table_ref->Cast<BaseTableRef>();
	const string &table_alias = !table_ref.alias.empty() ? table_ref.alias : table_ref.table_name;

	auto &on_conflict = *stmt.on_conflict_info;
	D_ASSERT(on_conflict.action_type != OnConflictAction::THROW);
	insert.action_type = on_conflict.action_type;

	// obtain the table storage info
	auto storage_info = table.GetStorageInfo(context);

	auto &columns = table.GetColumns();
	if (!on_conflict.indexed_columns.empty()) {
		// Bind the ON CONFLICT (<columns>)

		// create a mapping of (list index) -> (column index)
		case_insensitive_map_t<idx_t> specified_columns;
		for (idx_t i = 0; i < on_conflict.indexed_columns.size(); i++) {
			specified_columns[on_conflict.indexed_columns[i]] = i;
			auto column_index = table.GetColumnIndex(on_conflict.indexed_columns[i]);
			if (column_index.index == COLUMN_IDENTIFIER_ROW_ID) {
				throw BinderException("Cannot specify ROWID as ON CONFLICT target");
			}
			auto &col = columns.GetColumn(column_index);
			if (col.Generated()) {
				throw BinderException("Cannot specify a generated column as ON CONFLICT target");
			}
		}
		for (auto &col : columns.Physical()) {
			auto entry = specified_columns.find(col.Name());
			if (entry != specified_columns.end()) {
				// column was specified, set to the index
				insert.on_conflict_filter.insert(col.Oid());
			}
		}
		bool index_references_columns = false;
		for (auto &index : storage_info.index_info) {
			if (!index.is_unique) {
				continue;
			}
			bool index_matches = insert.on_conflict_filter == index.column_set;
			if (index_matches) {
				index_references_columns = true;
				break;
			}
		}
		if (!index_references_columns) {
			// Same as before, this is essentially a no-op, turning this into a DO THROW instead
			// But since this makes no logical sense, it's probably better to throw an error
			throw BinderException(
			    "The specified columns as conflict target are not referenced by a UNIQUE/PRIMARY KEY CONSTRAINT");
		}
	} else {
		// When omitting the conflict target, the ON CONFLICT applies to every UNIQUE/PRIMARY KEY on the table

		// We check if there are any constraints on the table, if there aren't we throw an error.
		idx_t found_matching_indexes = 0;
		for (auto &index : storage_info.index_info) {
			if (!index.is_unique) {
				continue;
			}
			// does this work with multi-column indexes?
			auto &indexed_columns = index.column_set;
			for (auto &column : table.GetColumns().Physical()) {
				if (indexed_columns.count(column.Physical().index)) {
					found_matching_indexes++;
				}
			}
		}
		if (!found_matching_indexes) {
			throw BinderException(
			    "There are no UNIQUE/PRIMARY KEY Indexes that refer to this table, ON CONFLICT is a no-op");
		}
		if (insert.action_type != OnConflictAction::NOTHING && found_matching_indexes != 1) {
			// When no conflict target is provided, and the action type is UPDATE,
			// we only allow the operation when only a single Index exists
			throw BinderException("Conflict target has to be provided for a DO UPDATE operation when the table has "
			                      "multiple UNIQUE/PRIMARY KEY constraints");
		}
	}

	// add the 'excluded' dummy table binding
	AddTableName("excluded");
	// add a bind context entry for it
	auto excluded_index = GenerateTableIndex();
	insert.excluded_table_index = excluded_index;
	auto table_column_names = columns.GetColumnNames();
	auto table_column_types = columns.GetColumnTypes();
	bind_context.AddGenericBinding(excluded_index, "excluded", table_column_names, table_column_types);

	if (on_conflict.condition) {
		// Avoid ambiguity between <table_name> binding and 'excluded'
		QualifyColumnReferences(on_conflict.condition, table_alias);
		// Bind the ON CONFLICT ... WHERE clause
		WhereBinder where_binder(*this, context);
		auto condition = where_binder.Bind(on_conflict.condition);
		if (condition && condition->expression_class == ExpressionClass::BOUND_SUBQUERY) {
			throw BinderException("conflict_target WHERE clause can not be a subquery");
		}
		insert.on_conflict_condition = std::move(condition);
	}

	auto bindings = insert.children[0]->GetColumnBindings();
	idx_t projection_index = DConstants::INVALID_INDEX;
	vector<unique_ptr<LogicalOperator>> *insert_child_operators;
	insert_child_operators = &insert.children;
	while (projection_index == DConstants::INVALID_INDEX) {
		if (insert_child_operators->empty()) {
			// No further children to visit
			break;
		}
		D_ASSERT(insert_child_operators->size() >= 1);
		auto &current_child = (*insert_child_operators)[0];
		auto table_indices = current_child->GetTableIndex();
		if (table_indices.empty()) {
			// This operator does not have a table index to refer to, we have to visit its children
			insert_child_operators = &current_child->children;
			continue;
		}
		projection_index = table_indices[0];
	}
	if (projection_index == DConstants::INVALID_INDEX) {
		throw InternalException("Could not locate a table_index from the children of the insert");
	}

	string unused;
	auto original_binding = bind_context.GetBinding(table_alias, unused);
	D_ASSERT(original_binding);

	auto table_index = original_binding->index;

	// Replace any column bindings to refer to the projection table_index, rather than the source table
	if (insert.on_conflict_condition) {
		ReplaceColumnBindings(*insert.on_conflict_condition, table_index, projection_index);
	}

	if (insert.action_type == OnConflictAction::REPLACE) {
		D_ASSERT(on_conflict.set_info == nullptr);
		on_conflict.set_info = CreateSetInfoForReplace(table, stmt, storage_info);
		insert.action_type = OnConflictAction::UPDATE;
	}
	if (on_conflict.set_info && on_conflict.set_info->columns.empty()) {
		// if we are doing INSERT OR REPLACE on a table with no columns outside of the primary key column
		// convert to INSERT OR IGNORE
		insert.action_type = OnConflictAction::NOTHING;
	}
	if (insert.action_type == OnConflictAction::NOTHING) {
		if (!insert.on_conflict_condition) {
			return;
		}
		// Get the column_ids we need to fetch later on from the conflicting tuples
		// of the original table, to execute the expressions
		D_ASSERT(original_binding->binding_type == BindingType::TABLE);
		auto &table_binding = (TableBinding &)*original_binding;
		insert.columns_to_fetch = table_binding.GetBoundColumnIds();
		return;
	}

	D_ASSERT(on_conflict.set_info);
	auto &set_info = *on_conflict.set_info;
	D_ASSERT(set_info.columns.size() == set_info.expressions.size());

	if (set_info.condition) {
		// Avoid ambiguity between <table_name> binding and 'excluded'
		QualifyColumnReferences(set_info.condition, table_alias);
		// Bind the SET ... WHERE clause
		WhereBinder where_binder(*this, context);
		auto condition = where_binder.Bind(set_info.condition);
		if (condition && condition->expression_class == ExpressionClass::BOUND_SUBQUERY) {
			throw BinderException("conflict_target WHERE clause can not be a subquery");
		}
		insert.do_update_condition = std::move(condition);
	}

	BindDoUpdateSetExpressions(table_alias, insert, set_info, table, storage_info);

	// Get the column_ids we need to fetch later on from the conflicting tuples
	// of the original table, to execute the expressions
	D_ASSERT(original_binding->binding_type == BindingType::TABLE);
	auto &table_binding = (TableBinding &)*original_binding;
	insert.columns_to_fetch = table_binding.GetBoundColumnIds();

	// Replace the column bindings to refer to the child operator
	for (auto &expr : insert.expressions) {
		// Change the non-excluded column references to refer to the projection index
		ReplaceColumnBindings(*expr, table_index, projection_index);
	}
	// Do the same for the (optional) DO UPDATE condition
	if (insert.do_update_condition) {
		ReplaceColumnBindings(*insert.do_update_condition, table_index, projection_index);
	}
}

BoundStatement Binder::Bind(InsertStatement &stmt) {
	BoundStatement result;
	result.names = {"Count"};
	result.types = {LogicalType::BIGINT};

	BindSchemaOrCatalog(stmt.catalog, stmt.schema);
	auto &table = Catalog::GetEntry<TableCatalogEntry>(context, stmt.catalog, stmt.schema, stmt.table);
	if (!table.temporary) {
		// inserting into a non-temporary table: alters underlying database
		properties.modified_databases.insert(table.catalog.GetName());
	}

	auto insert = make_uniq<LogicalInsert>(table, GenerateTableIndex());
	// Add CTEs as bindable
	AddCTEMap(stmt.cte_map);

	auto values_list = stmt.GetValuesList();

	// bind the root select node (if any)
	BoundStatement root_select;
	if (stmt.column_order == InsertColumnOrder::INSERT_BY_NAME) {
		if (values_list) {
			throw BinderException("INSERT BY NAME can only be used when inserting from a SELECT statement");
		}
		if (!stmt.columns.empty()) {
			throw BinderException("INSERT BY NAME cannot be combined with an explicit column list");
		}
		D_ASSERT(stmt.select_statement);
		// INSERT BY NAME - generate the columns from the names of the SELECT statement
		auto select_binder = Binder::CreateBinder(context, this);
		root_select = select_binder->Bind(*stmt.select_statement);
		MoveCorrelatedExpressions(*select_binder);

		stmt.columns = root_select.names;
	}

	vector<LogicalIndex> named_column_map;
	if (!stmt.columns.empty() || stmt.default_values) {
		// insertion statement specifies column list

		// create a mapping of (list index) -> (column index)
		case_insensitive_map_t<idx_t> column_name_map;
		for (idx_t i = 0; i < stmt.columns.size(); i++) {
			auto entry = column_name_map.insert(make_pair(stmt.columns[i], i));
			if (!entry.second) {
				throw BinderException("Duplicate column name \"%s\" in INSERT", stmt.columns[i]);
			}
			column_name_map[stmt.columns[i]] = i;
			auto column_index = table.GetColumnIndex(stmt.columns[i]);
			if (column_index.index == COLUMN_IDENTIFIER_ROW_ID) {
				throw BinderException("Cannot explicitly insert values into rowid column");
			}
			auto &col = table.GetColumn(column_index);
			if (col.Generated()) {
				throw BinderException("Cannot insert into a generated column");
			}
			insert->expected_types.push_back(col.Type());
			named_column_map.push_back(column_index);
		}
		for (auto &col : table.GetColumns().Physical()) {
			auto entry = column_name_map.find(col.Name());
			if (entry == column_name_map.end()) {
				// column not specified, set index to DConstants::INVALID_INDEX
				insert->column_index_map.push_back(DConstants::INVALID_INDEX);
			} else {
				// column was specified, set to the index
				insert->column_index_map.push_back(entry->second);
			}
		}
	} else {
		// insert by position and no columns specified - insertion into all columns of the table
		// intentionally don't populate 'column_index_map' as an indication of this
		for (auto &col : table.GetColumns().Physical()) {
			named_column_map.push_back(col.Logical());
			insert->expected_types.push_back(col.Type());
		}
	}

	// bind the default values
	BindDefaultValues(table.GetColumns(), insert->bound_defaults);
	if (!stmt.select_statement && !stmt.default_values) {
		result.plan = std::move(insert);
		return result;
	}
	// Exclude the generated columns from this amount
	idx_t expected_columns = stmt.columns.empty() ? table.GetColumns().PhysicalColumnCount() : stmt.columns.size();

	// special case: check if we are inserting from a VALUES statement
	if (values_list) {
		auto &expr_list = values_list->Cast<ExpressionListRef>();
		expr_list.expected_types.resize(expected_columns);
		expr_list.expected_names.resize(expected_columns);

		D_ASSERT(expr_list.values.size() > 0);
		CheckInsertColumnCountMismatch(expected_columns, expr_list.values[0].size(), !stmt.columns.empty(),
		                               table.name.c_str());

		// VALUES list!
		for (idx_t col_idx = 0; col_idx < expected_columns; col_idx++) {
			D_ASSERT(named_column_map.size() >= col_idx);
			auto &table_col_idx = named_column_map[col_idx];

			// set the expected types as the types for the INSERT statement
			auto &column = table.GetColumn(table_col_idx);
			expr_list.expected_types[col_idx] = column.Type();
			expr_list.expected_names[col_idx] = column.Name();

			// now replace any DEFAULT values with the corresponding default expression
			for (idx_t list_idx = 0; list_idx < expr_list.values.size(); list_idx++) {
				if (expr_list.values[list_idx][col_idx]->type == ExpressionType::VALUE_DEFAULT) {
					// DEFAULT value! replace the entry
					ReplaceDefaultExpression(expr_list.values[list_idx][col_idx], column);
				}
			}
		}
	}

	// parse select statement and add to logical plan
	unique_ptr<LogicalOperator> root;
	if (stmt.select_statement) {
		if (stmt.column_order == InsertColumnOrder::INSERT_BY_POSITION) {
			auto select_binder = Binder::CreateBinder(context, this);
			root_select = select_binder->Bind(*stmt.select_statement);
			MoveCorrelatedExpressions(*select_binder);
		}
		// inserting from a select - check if the column count matches
		CheckInsertColumnCountMismatch(expected_columns, root_select.types.size(), !stmt.columns.empty(),
		                               table.name.c_str());

		root = CastLogicalOperatorToTypes(root_select.types, insert->expected_types, std::move(root_select.plan));
	} else {
		root = make_uniq<LogicalDummyScan>(GenerateTableIndex());
	}
	insert->AddChild(std::move(root));

	BindOnConflictClause(*insert, table, stmt);

	if (!stmt.returning_list.empty()) {
		insert->return_chunk = true;
		result.types.clear();
		result.names.clear();
		auto insert_table_index = GenerateTableIndex();
		insert->table_index = insert_table_index;
		unique_ptr<LogicalOperator> index_as_logicaloperator = std::move(insert);

		return BindReturning(std::move(stmt.returning_list), table, stmt.table_ref ? stmt.table_ref->alias : string(),
		                     insert_table_index, std::move(index_as_logicaloperator), std::move(result));
	}

	D_ASSERT(result.types.size() == result.names.size());
	result.plan = std::move(insert);
	properties.allow_stream_result = false;
	properties.return_type = StatementReturnType::CHANGED_ROWS;
	return result;
}

} // namespace duckdb



#include <algorithm>

namespace duckdb {

BoundStatement Binder::Bind(LoadStatement &stmt) {
	BoundStatement result;
	result.types = {LogicalType::BOOLEAN};
	result.names = {"Success"};

	result.plan = make_uniq<LogicalSimple>(LogicalOperatorType::LOGICAL_LOAD, std::move(stmt.info));
	properties.allow_stream_result = false;
	properties.return_type = StatementReturnType::NOTHING;
	return result;
}

} // namespace duckdb


#include <algorithm>

namespace duckdb {

idx_t GetMaxTableIndex(LogicalOperator &op) {
	idx_t result = 0;
	for (auto &child : op.children) {
		auto max_child_index = GetMaxTableIndex(*child);
		result = MaxValue<idx_t>(result, max_child_index);
	}
	auto indexes = op.GetTableIndex();
	for (auto &index : indexes) {
		result = MaxValue<idx_t>(result, index);
	}
	return result;
}

BoundStatement Binder::Bind(LogicalPlanStatement &stmt) {
	BoundStatement result;
	result.types = stmt.plan->types;
	for (idx_t i = 0; i < result.types.size(); i++) {
		result.names.push_back(StringUtil::Format("col%d", i));
	}
	result.plan = std::move(stmt.plan);
	properties.allow_stream_result = true;
	properties.return_type = StatementReturnType::QUERY_RESULT; // TODO could also be something else

	if (parent) {
		throw InternalException("LogicalPlanStatement should be bound in root binder");
	}
	bound_tables = GetMaxTableIndex(*result.plan) + 1;
	return result;
}

} // namespace duckdb







namespace duckdb {

BoundStatement Binder::Bind(PragmaStatement &stmt) {
	// bind the pragma function
	auto &entry =
	    Catalog::GetEntry<PragmaFunctionCatalogEntry>(context, INVALID_CATALOG, DEFAULT_SCHEMA, stmt.info->name);
	string error;
	FunctionBinder function_binder(context);
	idx_t bound_idx = function_binder.BindFunction(entry.name, entry.functions, *stmt.info, error);
	if (bound_idx == DConstants::INVALID_INDEX) {
		throw BinderException(FormatError(stmt.stmt_location, error));
	}
	auto bound_function = entry.functions.GetFunctionByOffset(bound_idx);
	if (!bound_function.function) {
		throw BinderException("PRAGMA function does not have a function specified");
	}

	// bind and check named params
	QueryErrorContext error_context(root_statement, stmt.stmt_location);
	BindNamedParameters(bound_function.named_parameters, stmt.info->named_parameters, error_context,
	                    bound_function.name);

	BoundStatement result;
	result.names = {"Success"};
	result.types = {LogicalType::BOOLEAN};
	result.plan = make_uniq<LogicalPragma>(bound_function, *stmt.info);
	properties.return_type = StatementReturnType::QUERY_RESULT;
	return result;
}

} // namespace duckdb





namespace duckdb {

BoundStatement Binder::Bind(PrepareStatement &stmt) {
	Planner prepared_planner(context);
	auto prepared_data = prepared_planner.PrepareSQLStatement(std::move(stmt.statement));
	this->bound_tables = prepared_planner.binder->bound_tables;

	auto prepare = make_uniq<LogicalPrepare>(stmt.name, std::move(prepared_data), std::move(prepared_planner.plan));
	// we can always prepare, even if the transaction has been invalidated
	// this is required because most clients ALWAYS invoke prepared statements
	properties.requires_valid_transaction = false;
	properties.allow_stream_result = false;
	properties.bound_all_parameters = true;
	properties.parameter_count = 0;
	properties.return_type = StatementReturnType::NOTHING;

	BoundStatement result;
	result.names = {"Success"};
	result.types = {LogicalType::BOOLEAN};
	result.plan = std::move(prepare);
	return result;
}

} // namespace duckdb







namespace duckdb {

BoundStatement Binder::Bind(RelationStatement &stmt) {
	return stmt.relation->Bind(*this);
}

} // namespace duckdb




namespace duckdb {

BoundStatement Binder::Bind(SelectStatement &stmt) {
	properties.allow_stream_result = true;
	properties.return_type = StatementReturnType::QUERY_RESULT;
	return Bind(*stmt.node);
}

} // namespace duckdb




#include <algorithm>

namespace duckdb {

BoundStatement Binder::Bind(SetVariableStatement &stmt) {
	BoundStatement result;
	result.types = {LogicalType::BOOLEAN};
	result.names = {"Success"};

	result.plan = make_uniq<LogicalSet>(stmt.name, stmt.value, stmt.scope);
	properties.return_type = StatementReturnType::NOTHING;
	return result;
}

BoundStatement Binder::Bind(ResetVariableStatement &stmt) {
	BoundStatement result;
	result.types = {LogicalType::BOOLEAN};
	result.names = {"Success"};

	result.plan = make_uniq<LogicalReset>(stmt.name, stmt.scope);
	properties.return_type = StatementReturnType::NOTHING;
	return result;
}

BoundStatement Binder::Bind(SetStatement &stmt) {
	switch (stmt.set_type) {
	case SetType::SET: {
		auto &set_stmt = stmt.Cast<SetVariableStatement>();
		return Bind(set_stmt);
	}
	case SetType::RESET: {
		auto &set_stmt = stmt.Cast<ResetVariableStatement>();
		return Bind(set_stmt);
	}
	default:
		throw NotImplementedException("Type not implemented for SetType");
	}
}

} // namespace duckdb




namespace duckdb {

BoundStatement Binder::Bind(ShowStatement &stmt) {
	BoundStatement result;

	if (stmt.info->is_summary) {
		return BindSummarize(stmt);
	}
	auto plan = Bind(*stmt.info->query);
	stmt.info->types = plan.types;
	stmt.info->aliases = plan.names;

	auto show = make_uniq<LogicalShow>(std::move(plan.plan));
	show->types_select = plan.types;
	show->aliases = plan.names;

	result.plan = std::move(show);

	result.names = {"column_name", "column_type", "null", "key", "default", "extra"};
	result.types = {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
	                LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR};
	properties.return_type = StatementReturnType::QUERY_RESULT;
	return result;
}

} // namespace duckdb









//! This file contains the binder definitions for statements that do not need to be bound at all and only require a
//! straightforward conversion

namespace duckdb {

BoundStatement Binder::Bind(AlterStatement &stmt) {
	BoundStatement result;
	result.names = {"Success"};
	result.types = {LogicalType::BOOLEAN};
	BindSchemaOrCatalog(stmt.info->catalog, stmt.info->schema);
	auto entry = Catalog::GetEntry(context, stmt.info->GetCatalogType(), stmt.info->catalog, stmt.info->schema,
	                               stmt.info->name, stmt.info->if_not_found);
	if (entry) {
		auto &catalog = entry->ParentCatalog();
		if (!entry->temporary) {
			// we can only alter temporary tables/views in read-only mode
			properties.modified_databases.insert(catalog.GetName());
		}
		stmt.info->catalog = catalog.GetName();
		stmt.info->schema = entry->ParentSchema().name;
	}
	result.plan = make_uniq<LogicalSimple>(LogicalOperatorType::LOGICAL_ALTER, std::move(stmt.info));
	properties.return_type = StatementReturnType::NOTHING;
	return result;
}

BoundStatement Binder::Bind(TransactionStatement &stmt) {
	// transaction statements do not require a valid transaction
	properties.requires_valid_transaction = stmt.info->type == TransactionType::BEGIN_TRANSACTION;

	BoundStatement result;
	result.names = {"Success"};
	result.types = {LogicalType::BOOLEAN};
	result.plan = make_uniq<LogicalSimple>(LogicalOperatorType::LOGICAL_TRANSACTION, std::move(stmt.info));
	properties.return_type = StatementReturnType::NOTHING;
	return result;
}

} // namespace duckdb









namespace duckdb {

static unique_ptr<ParsedExpression> SummarizeWrapUnnest(vector<unique_ptr<ParsedExpression>> &children,
                                                        const string &alias) {
	auto list_function = make_uniq<FunctionExpression>("list_value", std::move(children));
	vector<unique_ptr<ParsedExpression>> unnest_children;
	unnest_children.push_back(std::move(list_function));
	auto unnest_function = make_uniq<FunctionExpression>("unnest", std::move(unnest_children));
	unnest_function->alias = alias;
	return std::move(unnest_function);
}

static unique_ptr<ParsedExpression> SummarizeCreateAggregate(const string &aggregate, string column_name) {
	vector<unique_ptr<ParsedExpression>> children;
	children.push_back(make_uniq<ColumnRefExpression>(std::move(column_name)));
	auto aggregate_function = make_uniq<FunctionExpression>(aggregate, std::move(children));
	auto cast_function = make_uniq<CastExpression>(LogicalType::VARCHAR, std::move(aggregate_function));
	return std::move(cast_function);
}

static unique_ptr<ParsedExpression> SummarizeCreateAggregate(const string &aggregate, string column_name,
                                                             const Value &modifier) {
	vector<unique_ptr<ParsedExpression>> children;
	children.push_back(make_uniq<ColumnRefExpression>(std::move(column_name)));
	children.push_back(make_uniq<ConstantExpression>(modifier));
	auto aggregate_function = make_uniq<FunctionExpression>(aggregate, std::move(children));
	auto cast_function = make_uniq<CastExpression>(LogicalType::VARCHAR, std::move(aggregate_function));
	return std::move(cast_function);
}

static unique_ptr<ParsedExpression> SummarizeCreateCountStar() {
	vector<unique_ptr<ParsedExpression>> children;
	auto aggregate_function = make_uniq<FunctionExpression>("count_star", std::move(children));
	return std::move(aggregate_function);
}

static unique_ptr<ParsedExpression> SummarizeCreateBinaryFunction(const string &op, unique_ptr<ParsedExpression> left,
                                                                  unique_ptr<ParsedExpression> right) {
	vector<unique_ptr<ParsedExpression>> children;
	children.push_back(std::move(left));
	children.push_back(std::move(right));
	auto binary_function = make_uniq<FunctionExpression>(op, std::move(children));
	return std::move(binary_function);
}

static unique_ptr<ParsedExpression> SummarizeCreateNullPercentage(string column_name) {
	auto count_star = make_uniq<CastExpression>(LogicalType::DOUBLE, SummarizeCreateCountStar());
	auto count =
	    make_uniq<CastExpression>(LogicalType::DOUBLE, SummarizeCreateAggregate("count", std::move(column_name)));
	auto null_percentage = SummarizeCreateBinaryFunction("/", std::move(count), std::move(count_star));
	auto negate_x =
	    SummarizeCreateBinaryFunction("-", make_uniq<ConstantExpression>(Value::DOUBLE(1)), std::move(null_percentage));
	auto percentage_x =
	    SummarizeCreateBinaryFunction("*", std::move(negate_x), make_uniq<ConstantExpression>(Value::DOUBLE(100)));
	auto round_x = SummarizeCreateBinaryFunction("round", std::move(percentage_x),
	                                             make_uniq<ConstantExpression>(Value::INTEGER(2)));
	auto concat_x =
	    SummarizeCreateBinaryFunction("concat", std::move(round_x), make_uniq<ConstantExpression>(Value("%")));

	return concat_x;
}

BoundStatement Binder::BindSummarize(ShowStatement &stmt) {
	auto query_copy = stmt.info->query->Copy();

	// we bind the plan once in a child-node to figure out the column names and column types
	auto child_binder = Binder::CreateBinder(context);
	auto plan = child_binder->Bind(*stmt.info->query);
	D_ASSERT(plan.types.size() == plan.names.size());
	vector<unique_ptr<ParsedExpression>> name_children;
	vector<unique_ptr<ParsedExpression>> type_children;
	vector<unique_ptr<ParsedExpression>> min_children;
	vector<unique_ptr<ParsedExpression>> max_children;
	vector<unique_ptr<ParsedExpression>> unique_children;
	vector<unique_ptr<ParsedExpression>> avg_children;
	vector<unique_ptr<ParsedExpression>> std_children;
	vector<unique_ptr<ParsedExpression>> q25_children;
	vector<unique_ptr<ParsedExpression>> q50_children;
	vector<unique_ptr<ParsedExpression>> q75_children;
	vector<unique_ptr<ParsedExpression>> count_children;
	vector<unique_ptr<ParsedExpression>> null_percentage_children;
	auto select = make_uniq<SelectStatement>();
	select->node = std::move(query_copy);
	for (idx_t i = 0; i < plan.names.size(); i++) {
		name_children.push_back(make_uniq<ConstantExpression>(Value(plan.names[i])));
		type_children.push_back(make_uniq<ConstantExpression>(Value(plan.types[i].ToString())));
		min_children.push_back(SummarizeCreateAggregate("min", plan.names[i]));
		max_children.push_back(SummarizeCreateAggregate("max", plan.names[i]));
		unique_children.push_back(SummarizeCreateAggregate("approx_count_distinct", plan.names[i]));
		if (plan.types[i].IsNumeric()) {
			avg_children.push_back(SummarizeCreateAggregate("avg", plan.names[i]));
			std_children.push_back(SummarizeCreateAggregate("stddev", plan.names[i]));
			q25_children.push_back(SummarizeCreateAggregate("approx_quantile", plan.names[i], Value::FLOAT(0.25)));
			q50_children.push_back(SummarizeCreateAggregate("approx_quantile", plan.names[i], Value::FLOAT(0.50)));
			q75_children.push_back(SummarizeCreateAggregate("approx_quantile", plan.names[i], Value::FLOAT(0.75)));
		} else {
			avg_children.push_back(make_uniq<ConstantExpression>(Value()));
			std_children.push_back(make_uniq<ConstantExpression>(Value()));
			q25_children.push_back(make_uniq<ConstantExpression>(Value()));
			q50_children.push_back(make_uniq<ConstantExpression>(Value()));
			q75_children.push_back(make_uniq<ConstantExpression>(Value()));
		}
		count_children.push_back(SummarizeCreateCountStar());
		null_percentage_children.push_back(SummarizeCreateNullPercentage(plan.names[i]));
	}
	auto subquery_ref = make_uniq<SubqueryRef>(std::move(select), "summarize_tbl");
	subquery_ref->column_name_alias = plan.names;

	auto select_node = make_uniq<SelectNode>();
	select_node->select_list.push_back(SummarizeWrapUnnest(name_children, "column_name"));
	select_node->select_list.push_back(SummarizeWrapUnnest(type_children, "column_type"));
	select_node->select_list.push_back(SummarizeWrapUnnest(min_children, "min"));
	select_node->select_list.push_back(SummarizeWrapUnnest(max_children, "max"));
	select_node->select_list.push_back(SummarizeWrapUnnest(unique_children, "approx_unique"));
	select_node->select_list.push_back(SummarizeWrapUnnest(avg_children, "avg"));
	select_node->select_list.push_back(SummarizeWrapUnnest(std_children, "std"));
	select_node->select_list.push_back(SummarizeWrapUnnest(q25_children, "q25"));
	select_node->select_list.push_back(SummarizeWrapUnnest(q50_children, "q50"));
	select_node->select_list.push_back(SummarizeWrapUnnest(q75_children, "q75"));
	select_node->select_list.push_back(SummarizeWrapUnnest(count_children, "count"));
	select_node->select_list.push_back(SummarizeWrapUnnest(null_percentage_children, "null_percentage"));
	select_node->from_table = std::move(subquery_ref);

	properties.return_type = StatementReturnType::QUERY_RESULT;
	return Bind(*select_node);
}

} // namespace duckdb


















#include <algorithm>

namespace duckdb {

static void BindExtraColumns(TableCatalogEntry &table, LogicalGet &get, LogicalProjection &proj, LogicalUpdate &update,
                             physical_index_set_t &bound_columns) {
	if (bound_columns.size() <= 1) {
		return;
	}
	idx_t found_column_count = 0;
	physical_index_set_t found_columns;
	for (idx_t i = 0; i < update.columns.size(); i++) {
		if (bound_columns.find(update.columns[i]) != bound_columns.end()) {
			// this column is referenced in the CHECK constraint
			found_column_count++;
			found_columns.insert(update.columns[i]);
		}
	}
	if (found_column_count > 0 && found_column_count != bound_columns.size()) {
		// columns in this CHECK constraint were referenced, but not all were part of the UPDATE
		// add them to the scan and update set
		for (auto &check_column_id : bound_columns) {
			if (found_columns.find(check_column_id) != found_columns.end()) {
				// column is already projected
				continue;
			}
			// column is not projected yet: project it by adding the clause "i=i" to the set of updated columns
			auto &column = table.GetColumns().GetColumn(check_column_id);
			update.expressions.push_back(make_uniq<BoundColumnRefExpression>(
			    column.Type(), ColumnBinding(proj.table_index, proj.expressions.size())));
			proj.expressions.push_back(make_uniq<BoundColumnRefExpression>(
			    column.Type(), ColumnBinding(get.table_index, get.column_ids.size())));
			get.column_ids.push_back(check_column_id.index);
			update.columns.push_back(check_column_id);
		}
	}
}

static bool TypeSupportsRegularUpdate(const LogicalType &type) {
	switch (type.id()) {
	case LogicalTypeId::LIST:
	case LogicalTypeId::MAP:
	case LogicalTypeId::UNION:
		// lists and maps and unions don't support updates directly
		return false;
	case LogicalTypeId::STRUCT: {
		auto &child_types = StructType::GetChildTypes(type);
		for (auto &entry : child_types) {
			if (!TypeSupportsRegularUpdate(entry.second)) {
				return false;
			}
		}
		return true;
	}
	default:
		return true;
	}
}

static void BindUpdateConstraints(TableCatalogEntry &table, LogicalGet &get, LogicalProjection &proj,
                                  LogicalUpdate &update) {
	if (!table.IsDuckTable()) {
		return;
	}
	// check the constraints and indexes of the table to see if we need to project any additional columns
	// we do this for indexes with multiple columns and CHECK constraints in the UPDATE clause
	// suppose we have a constraint CHECK(i + j < 10); now we need both i and j to check the constraint
	// if we are only updating one of the two columns we add the other one to the UPDATE set
	// with a "useless" update (i.e. i=i) so we can verify that the CHECK constraint is not violated
	for (auto &constraint : table.GetBoundConstraints()) {
		if (constraint->type == ConstraintType::CHECK) {
			auto &check = *reinterpret_cast<BoundCheckConstraint *>(constraint.get());
			// check constraint! check if we need to add any extra columns to the UPDATE clause
			BindExtraColumns(table, get, proj, update, check.bound_columns);
		}
	}
	auto &storage = table.GetStorage();
	if (update.return_chunk) {
		physical_index_set_t all_columns;
		for (idx_t i = 0; i < storage.column_definitions.size(); i++) {
			all_columns.insert(PhysicalIndex(i));
		}
		BindExtraColumns(table, get, proj, update, all_columns);
	}
	// for index updates we always turn any update into an insert and a delete
	// we thus need all the columns to be available, hence we check if the update touches any index columns
	// If the returning keyword is used, we need access to the whole row in case the user requests it.
	// Therefore switch the update to a delete and insert.
	update.update_is_del_and_insert = false;
	storage.info->indexes.Scan([&](Index &index) {
		if (index.IndexIsUpdated(update.columns)) {
			update.update_is_del_and_insert = true;
			return true;
		}
		return false;
	});

	// we also convert any updates on LIST columns into delete + insert
	for (auto &col_index : update.columns) {
		auto &column = table.GetColumns().GetColumn(col_index);
		if (!TypeSupportsRegularUpdate(column.Type())) {
			update.update_is_del_and_insert = true;
			break;
		}
	}

	if (update.update_is_del_and_insert) {
		// the update updates a column required by an index or requires returning the updated rows,
		// push projections for all columns
		physical_index_set_t all_columns;
		for (idx_t i = 0; i < storage.column_definitions.size(); i++) {
			all_columns.insert(PhysicalIndex(i));
		}
		BindExtraColumns(table, get, proj, update, all_columns);
	}
}

// This creates a LogicalProjection and moves 'root' into it as a child
// unless there are no expressions to project, in which case it just returns 'root'
unique_ptr<LogicalOperator> Binder::BindUpdateSet(LogicalOperator &op, unique_ptr<LogicalOperator> root,
                                                  UpdateSetInfo &set_info, TableCatalogEntry &table,
                                                  vector<PhysicalIndex> &columns) {
	auto proj_index = GenerateTableIndex();

	vector<unique_ptr<Expression>> projection_expressions;
	D_ASSERT(set_info.columns.size() == set_info.expressions.size());
	for (idx_t i = 0; i < set_info.columns.size(); i++) {
		auto &colname = set_info.columns[i];
		auto &expr = set_info.expressions[i];
		if (!table.ColumnExists(colname)) {
			throw BinderException("Referenced update column %s not found in table!", colname);
		}
		auto &column = table.GetColumn(colname);
		if (column.Generated()) {
			throw BinderException("Cant update column \"%s\" because it is a generated column!", column.Name());
		}
		if (std::find(columns.begin(), columns.end(), column.Physical()) != columns.end()) {
			throw BinderException("Multiple assignments to same column \"%s\"", colname);
		}
		columns.push_back(column.Physical());
		if (expr->type == ExpressionType::VALUE_DEFAULT) {
			op.expressions.push_back(make_uniq<BoundDefaultExpression>(column.Type()));
		} else {
			UpdateBinder binder(*this, context);
			binder.target_type = column.Type();
			auto bound_expr = binder.Bind(expr);
			PlanSubqueries(bound_expr, root);

			op.expressions.push_back(make_uniq<BoundColumnRefExpression>(
			    bound_expr->return_type, ColumnBinding(proj_index, projection_expressions.size())));
			projection_expressions.push_back(std::move(bound_expr));
		}
	}
	if (op.type != LogicalOperatorType::LOGICAL_UPDATE && projection_expressions.empty()) {
		return root;
	}
	// now create the projection
	auto proj = make_uniq<LogicalProjection>(proj_index, std::move(projection_expressions));
	proj->AddChild(std::move(root));
	return unique_ptr_cast<LogicalProjection, LogicalOperator>(std::move(proj));
}

BoundStatement Binder::Bind(UpdateStatement &stmt) {
	BoundStatement result;
	unique_ptr<LogicalOperator> root;

	// visit the table reference
	auto bound_table = Bind(*stmt.table);
	if (bound_table->type != TableReferenceType::BASE_TABLE) {
		throw BinderException("Can only update base table!");
	}
	auto &table_binding = bound_table->Cast<BoundBaseTableRef>();
	auto &table = table_binding.table;

	// Add CTEs as bindable
	AddCTEMap(stmt.cte_map);

	optional_ptr<LogicalGet> get;
	if (stmt.from_table) {
		auto from_binder = Binder::CreateBinder(context, this);
		BoundJoinRef bound_crossproduct(JoinRefType::CROSS);
		bound_crossproduct.left = std::move(bound_table);
		bound_crossproduct.right = from_binder->Bind(*stmt.from_table);
		root = CreatePlan(bound_crossproduct);
		get = &root->children[0]->Cast<LogicalGet>();
		bind_context.AddContext(std::move(from_binder->bind_context));
	} else {
		root = CreatePlan(*bound_table);
		get = &root->Cast<LogicalGet>();
	}

	if (!table.temporary) {
		// update of persistent table: not read only!
		properties.modified_databases.insert(table.catalog.GetName());
	}
	auto update = make_uniq<LogicalUpdate>(table);

	// set return_chunk boolean early because it needs uses update_is_del_and_insert logic
	if (!stmt.returning_list.empty()) {
		update->return_chunk = true;
	}
	// bind the default values
	BindDefaultValues(table.GetColumns(), update->bound_defaults);

	// project any additional columns required for the condition/expressions
	if (stmt.set_info->condition) {
		WhereBinder binder(*this, context);
		auto condition = binder.Bind(stmt.set_info->condition);

		PlanSubqueries(condition, root);
		auto filter = make_uniq<LogicalFilter>(std::move(condition));
		filter->AddChild(std::move(root));
		root = std::move(filter);
	}

	D_ASSERT(stmt.set_info);
	D_ASSERT(stmt.set_info->columns.size() == stmt.set_info->expressions.size());

	auto proj_tmp = BindUpdateSet(*update, std::move(root), *stmt.set_info, table, update->columns);
	D_ASSERT(proj_tmp->type == LogicalOperatorType::LOGICAL_PROJECTION);
	auto proj = unique_ptr_cast<LogicalOperator, LogicalProjection>(std::move(proj_tmp));

	// bind any extra columns necessary for CHECK constraints or indexes
	BindUpdateConstraints(table, *get, *proj, *update);
	// finally add the row id column to the projection list
	proj->expressions.push_back(make_uniq<BoundColumnRefExpression>(
	    LogicalType::ROW_TYPE, ColumnBinding(get->table_index, get->column_ids.size())));
	get->column_ids.push_back(COLUMN_IDENTIFIER_ROW_ID);

	// set the projection as child of the update node and finalize the result
	update->AddChild(std::move(proj));

	auto update_table_index = GenerateTableIndex();
	update->table_index = update_table_index;
	if (!stmt.returning_list.empty()) {
		unique_ptr<LogicalOperator> update_as_logicaloperator = std::move(update);

		return BindReturning(std::move(stmt.returning_list), table, stmt.table->alias, update_table_index,
		                     std::move(update_as_logicaloperator), std::move(result));
	}

	result.names = {"Count"};
	result.types = {LogicalType::BIGINT};
	result.plan = std::move(update);
	properties.allow_stream_result = false;
	properties.return_type = StatementReturnType::CHANGED_ROWS;
	return result;
}

} // namespace duckdb








namespace duckdb {

BoundStatement Binder::Bind(VacuumStatement &stmt) {
	BoundStatement result;

	unique_ptr<LogicalOperator> root;

	if (stmt.info->has_table) {
		D_ASSERT(!stmt.info->table);
		D_ASSERT(stmt.info->column_id_map.empty());
		auto bound_table = Bind(*stmt.info->ref);
		if (bound_table->type != TableReferenceType::BASE_TABLE) {
			throw InvalidInputException("Can only vacuum/analyze base tables!");
		}
		auto ref = unique_ptr_cast<BoundTableRef, BoundBaseTableRef>(std::move(bound_table));
		auto &table = ref->table;
		stmt.info->table = &table;

		auto &columns = stmt.info->columns;
		vector<unique_ptr<Expression>> select_list;
		if (columns.empty()) {
			// Empty means ALL columns should be vacuumed/analyzed
			auto &get = ref->get->Cast<LogicalGet>();
			columns.insert(columns.end(), get.names.begin(), get.names.end());
		}

		case_insensitive_set_t column_name_set;
		vector<string> non_generated_column_names;
		for (auto &col_name : columns) {
			if (column_name_set.count(col_name) > 0) {
				throw BinderException("Vacuum the same column twice(same name in column name list)");
			}
			column_name_set.insert(col_name);
			if (!table.ColumnExists(col_name)) {
				throw BinderException("Column with name \"%s\" does not exist", col_name);
			}
			auto &col = table.GetColumn(col_name);
			// ignore generated column
			if (col.Generated()) {
				continue;
			}
			non_generated_column_names.push_back(col_name);
			ColumnRefExpression colref(col_name, table.name);
			auto result = bind_context.BindColumn(colref, 0);
			if (result.HasError()) {
				throw BinderException(result.error);
			}
			select_list.push_back(std::move(result.expression));
		}
		stmt.info->columns = std::move(non_generated_column_names);
		if (!select_list.empty()) {
			auto table_scan = CreatePlan(*ref);
			D_ASSERT(table_scan->type == LogicalOperatorType::LOGICAL_GET);

			auto &get = table_scan->Cast<LogicalGet>();

			D_ASSERT(select_list.size() == get.column_ids.size());
			D_ASSERT(stmt.info->columns.size() == get.column_ids.size());
			for (idx_t i = 0; i < get.column_ids.size(); i++) {
				stmt.info->column_id_map[i] =
				    table.GetColumns().LogicalToPhysical(LogicalIndex(get.column_ids[i])).index;
			}

			auto projection = make_uniq<LogicalProjection>(GenerateTableIndex(), std::move(select_list));
			projection->children.push_back(std::move(table_scan));

			root = std::move(projection);
		} else {
			// eg. CREATE TABLE test (x AS (1));
			//     ANALYZE test;
			// Make it not a SINK so it doesn't have to do anything
			stmt.info->has_table = false;
		}
	}
	auto vacuum = make_uniq<LogicalSimple>(LogicalOperatorType::LOGICAL_VACUUM, std::move(stmt.info));
	if (root) {
		vacuum->children.push_back(std::move(root));
	}

	result.names = {"Success"};
	result.types = {LogicalType::BOOLEAN};
	result.plan = std::move(vacuum);
	properties.return_type = StatementReturnType::NOTHING;
	return result;
}

} // namespace duckdb

















namespace duckdb {

unique_ptr<BoundTableRef> Binder::Bind(BaseTableRef &ref) {
	QueryErrorContext error_context(root_statement, ref.query_location);
	// CTEs and views are also referred to using BaseTableRefs, hence need to distinguish here
	// check if the table name refers to a CTE
	auto found_cte = FindCTE(ref.table_name, ref.table_name == alias);
	if (found_cte) {
		// Check if there is a CTE binding in the BindContext
		auto &cte = *found_cte;
		auto ctebinding = bind_context.GetCTEBinding(ref.table_name);
		if (!ctebinding) {
			if (CTEIsAlreadyBound(cte)) {
				throw BinderException("Circular reference to CTE \"%s\", use WITH RECURSIVE to use recursive CTEs",
				                      ref.table_name);
			}
			// Move CTE to subquery and bind recursively
			SubqueryRef subquery(unique_ptr_cast<SQLStatement, SelectStatement>(cte.query->Copy()));
			subquery.alias = ref.alias.empty() ? ref.table_name : ref.alias;
			subquery.column_name_alias = cte.aliases;
			for (idx_t i = 0; i < ref.column_name_alias.size(); i++) {
				if (i < subquery.column_name_alias.size()) {
					subquery.column_name_alias[i] = ref.column_name_alias[i];
				} else {
					subquery.column_name_alias.push_back(ref.column_name_alias[i]);
				}
			}
			return Bind(subquery, found_cte);
		} else {
			// There is a CTE binding in the BindContext.
			// This can only be the case if there is a recursive CTE present.
			auto index = GenerateTableIndex();
			auto result = make_uniq<BoundCTERef>(index, ctebinding->index);
			auto b = ctebinding;
			auto alias = ref.alias.empty() ? ref.table_name : ref.alias;
			auto names = BindContext::AliasColumnNames(alias, b->names, ref.column_name_alias);

			bind_context.AddGenericBinding(index, alias, names, b->types);
			// Update references to CTE
			auto cteref = bind_context.cte_references[ref.table_name];
			(*cteref)++;

			result->types = b->types;
			result->bound_columns = std::move(names);
			return std::move(result);
		}
	}
	// not a CTE
	// extract a table or view from the catalog
	BindSchemaOrCatalog(ref.catalog_name, ref.schema_name);
	auto table_or_view = Catalog::GetEntry(context, CatalogType::TABLE_ENTRY, ref.catalog_name, ref.schema_name,
	                                       ref.table_name, OnEntryNotFound::RETURN_NULL, error_context);
	// we still didn't find the table
	if (GetBindingMode() == BindingMode::EXTRACT_NAMES) {
		if (!table_or_view || table_or_view->type == CatalogType::TABLE_ENTRY) {
			// if we are in EXTRACT_NAMES, we create a dummy table ref
			AddTableName(ref.table_name);

			// add a bind context entry
			auto table_index = GenerateTableIndex();
			auto alias = ref.alias.empty() ? ref.table_name : ref.alias;
			vector<LogicalType> types {LogicalType::INTEGER};
			vector<string> names {"__dummy_col" + to_string(table_index)};
			bind_context.AddGenericBinding(table_index, alias, names, types);
			return make_uniq_base<BoundTableRef, BoundEmptyTableRef>(table_index);
		}
	}
	if (!table_or_view) {
		string table_name = ref.catalog_name;
		if (!ref.schema_name.empty()) {
			table_name += (!table_name.empty() ? "." : "") + ref.schema_name;
		}
		table_name += (!table_name.empty() ? "." : "") + ref.table_name;
		// table could not be found: try to bind a replacement scan
		auto &config = DBConfig::GetConfig(context);
		if (context.config.use_replacement_scans) {
			for (auto &scan : config.replacement_scans) {
				auto replacement_function = scan.function(context, table_name, scan.data.get());
				if (replacement_function) {
					if (!ref.alias.empty()) {
						// user-provided alias overrides the default alias
						replacement_function->alias = ref.alias;
					} else if (replacement_function->alias.empty()) {
						// if the replacement scan itself did not provide an alias we use the table name
						replacement_function->alias = ref.table_name;
					}
					if (replacement_function->type == TableReferenceType::TABLE_FUNCTION) {
						auto &table_function = replacement_function->Cast<TableFunctionRef>();
						table_function.column_name_alias = ref.column_name_alias;
					} else if (replacement_function->type == TableReferenceType::SUBQUERY) {
						auto &subquery = replacement_function->Cast<SubqueryRef>();
						subquery.column_name_alias = ref.column_name_alias;
					} else {
						throw InternalException("Replacement scan should return either a table function or a subquery");
					}
					return Bind(*replacement_function);
				}
			}
		}

		// could not find an alternative: bind again to get the error
		table_or_view = Catalog::GetEntry(context, CatalogType::TABLE_ENTRY, ref.catalog_name, ref.schema_name,
		                                  ref.table_name, OnEntryNotFound::THROW_EXCEPTION, error_context);
	}
	switch (table_or_view->type) {
	case CatalogType::TABLE_ENTRY: {
		// base table: create the BoundBaseTableRef node
		auto table_index = GenerateTableIndex();
		auto &table = table_or_view->Cast<TableCatalogEntry>();

		unique_ptr<FunctionData> bind_data;
		auto scan_function = table.GetScanFunction(context, bind_data);
		auto alias = ref.alias.empty() ? ref.table_name : ref.alias;
		// TODO: bundle the type and name vector in a struct (e.g PackedColumnMetadata)
		vector<LogicalType> table_types;
		vector<string> table_names;
		vector<TableColumnType> table_categories;

		vector<LogicalType> return_types;
		vector<string> return_names;
		for (auto &col : table.GetColumns().Logical()) {
			table_types.push_back(col.Type());
			table_names.push_back(col.Name());
			return_types.push_back(col.Type());
			return_names.push_back(col.Name());
		}
		table_names = BindContext::AliasColumnNames(alias, table_names, ref.column_name_alias);

		auto logical_get = make_uniq<LogicalGet>(table_index, scan_function, std::move(bind_data),
		                                         std::move(return_types), std::move(return_names));
		bind_context.AddBaseTable(table_index, alias, table_names, table_types, logical_get->column_ids,
		                          logical_get->GetTable().get());
		return make_uniq_base<BoundTableRef, BoundBaseTableRef>(table, std::move(logical_get));
	}
	case CatalogType::VIEW_ENTRY: {
		// the node is a view: get the query that the view represents
		auto &view_catalog_entry = table_or_view->Cast<ViewCatalogEntry>();
		// We need to use a new binder for the view that doesn't reference any CTEs
		// defined for this binder so there are no collisions between the CTEs defined
		// for the view and for the current query
		bool inherit_ctes = false;
		auto view_binder = Binder::CreateBinder(context, this, inherit_ctes);
		view_binder->can_contain_nulls = true;
		SubqueryRef subquery(unique_ptr_cast<SQLStatement, SelectStatement>(view_catalog_entry.query->Copy()));
		subquery.alias = ref.alias.empty() ? ref.table_name : ref.alias;
		subquery.column_name_alias =
		    BindContext::AliasColumnNames(subquery.alias, view_catalog_entry.aliases, ref.column_name_alias);
		// bind the child subquery
		view_binder->AddBoundView(view_catalog_entry);
		auto bound_child = view_binder->Bind(subquery);
		if (!view_binder->correlated_columns.empty()) {
			throw BinderException("Contents of view were altered - view bound correlated columns");
		}

		D_ASSERT(bound_child->type == TableReferenceType::SUBQUERY);
		// verify that the types and names match up with the expected types and names
		auto &bound_subquery = bound_child->Cast<BoundSubqueryRef>();
		if (GetBindingMode() != BindingMode::EXTRACT_NAMES &&
		    bound_subquery.subquery->types != view_catalog_entry.types) {
			throw BinderException("Contents of view were altered: types don't match!");
		}
		bind_context.AddView(bound_subquery.subquery->GetRootIndex(), subquery.alias, subquery,
		                     *bound_subquery.subquery, &view_catalog_entry);
		return bound_child;
	}
	default:
		throw InternalException("Catalog entry type");
	}
}
} // namespace duckdb




namespace duckdb {

unique_ptr<BoundTableRef> Binder::Bind(EmptyTableRef &ref) {
	return make_uniq<BoundEmptyTableRef>(GenerateTableIndex());
}

} // namespace duckdb







namespace duckdb {

unique_ptr<BoundTableRef> Binder::Bind(ExpressionListRef &expr) {
	auto result = make_uniq<BoundExpressionListRef>();
	result->types = expr.expected_types;
	result->names = expr.expected_names;
	// bind value list
	InsertBinder binder(*this, context);
	binder.target_type = LogicalType(LogicalTypeId::INVALID);
	for (idx_t list_idx = 0; list_idx < expr.values.size(); list_idx++) {
		auto &expression_list = expr.values[list_idx];
		if (result->names.empty()) {
			// no names provided, generate them
			for (idx_t val_idx = 0; val_idx < expression_list.size(); val_idx++) {
				result->names.push_back("col" + to_string(val_idx));
			}
		}

		vector<unique_ptr<Expression>> list;
		for (idx_t val_idx = 0; val_idx < expression_list.size(); val_idx++) {
			if (!result->types.empty()) {
				D_ASSERT(result->types.size() == expression_list.size());
				binder.target_type = result->types[val_idx];
			}
			auto expr = binder.Bind(expression_list[val_idx]);
			list.push_back(std::move(expr));
		}
		result->values.push_back(std::move(list));
	}
	if (result->types.empty() && !expr.values.empty()) {
		// there are no types specified
		// we have to figure out the result types
		// for each column, we iterate over all of the expressions and select the max logical type
		// we initialize all types to SQLNULL
		result->types.resize(expr.values[0].size(), LogicalType::SQLNULL);
		// now loop over the lists and select the max logical type
		for (idx_t list_idx = 0; list_idx < result->values.size(); list_idx++) {
			auto &list = result->values[list_idx];
			for (idx_t val_idx = 0; val_idx < list.size(); val_idx++) {
				result->types[val_idx] =
				    LogicalType::MaxLogicalType(result->types[val_idx], list[val_idx]->return_type);
			}
		}
		// finally do another loop over the expressions and add casts where required
		for (idx_t list_idx = 0; list_idx < result->values.size(); list_idx++) {
			auto &list = result->values[list_idx];
			for (idx_t val_idx = 0; val_idx < list.size(); val_idx++) {
				list[val_idx] =
				    BoundCastExpression::AddCastToType(context, std::move(list[val_idx]), result->types[val_idx]);
			}
		}
	}
	result->bind_index = GenerateTableIndex();
	bind_context.AddGenericBinding(result->bind_index, expr.alias, result->names, result->types);
	return std::move(result);
}

} // namespace duckdb















namespace duckdb {

static unique_ptr<ParsedExpression> BindColumn(Binder &binder, ClientContext &context, const string &alias,
                                               const string &column_name) {
	auto expr = make_uniq_base<ParsedExpression, ColumnRefExpression>(column_name, alias);
	ExpressionBinder expr_binder(binder, context);
	auto result = expr_binder.Bind(expr);
	return make_uniq<BoundExpression>(std::move(result));
}

static unique_ptr<ParsedExpression> AddCondition(ClientContext &context, Binder &left_binder, Binder &right_binder,
                                                 const string &left_alias, const string &right_alias,
                                                 const string &column_name, ExpressionType type) {
	ExpressionBinder expr_binder(left_binder, context);
	auto left = BindColumn(left_binder, context, left_alias, column_name);
	auto right = BindColumn(right_binder, context, right_alias, column_name);
	return make_uniq<ComparisonExpression>(type, std::move(left), std::move(right));
}

bool Binder::TryFindBinding(const string &using_column, const string &join_side, string &result) {
	// for each using column, get the matching binding
	auto bindings = bind_context.GetMatchingBindings(using_column);
	if (bindings.empty()) {
		return false;
	}
	// find the join binding
	for (auto &binding : bindings) {
		if (!result.empty()) {
			string error = "Column name \"";
			error += using_column;
			error += "\" is ambiguous: it exists more than once on ";
			error += join_side;
			error += " side of join.\nCandidates:";
			for (auto &binding : bindings) {
				error += "\n\t";
				error += binding;
				error += ".";
				error += bind_context.GetActualColumnName(binding, using_column);
			}
			throw BinderException(error);
		} else {
			result = binding;
		}
	}
	return true;
}

string Binder::FindBinding(const string &using_column, const string &join_side) {
	string result;
	if (!TryFindBinding(using_column, join_side, result)) {
		throw BinderException("Column \"%s\" does not exist on %s side of join!", using_column, join_side);
	}
	return result;
}

static void AddUsingBindings(UsingColumnSet &set, optional_ptr<UsingColumnSet> input_set, const string &input_binding) {
	if (input_set) {
		for (auto &entry : input_set->bindings) {
			set.bindings.insert(entry);
		}
	} else {
		set.bindings.insert(input_binding);
	}
}

static void SetPrimaryBinding(UsingColumnSet &set, JoinType join_type, const string &left_binding,
                              const string &right_binding) {
	switch (join_type) {
	case JoinType::LEFT:
	case JoinType::INNER:
	case JoinType::SEMI:
	case JoinType::ANTI:
		set.primary_binding = left_binding;
		break;
	case JoinType::RIGHT:
		set.primary_binding = right_binding;
		break;
	default:
		break;
	}
}

string Binder::RetrieveUsingBinding(Binder &current_binder, optional_ptr<UsingColumnSet> current_set,
                                    const string &using_column, const string &join_side) {
	string binding;
	if (!current_set) {
		binding = current_binder.FindBinding(using_column, join_side);
	} else {
		binding = current_set->primary_binding;
	}
	return binding;
}

static vector<string> RemoveDuplicateUsingColumns(const vector<string> &using_columns) {
	vector<string> result;
	case_insensitive_set_t handled_columns;
	for (auto &using_column : using_columns) {
		if (handled_columns.find(using_column) == handled_columns.end()) {
			handled_columns.insert(using_column);
			result.push_back(using_column);
		}
	}
	return result;
}

unique_ptr<BoundTableRef> Binder::Bind(JoinRef &ref) {
	auto result = make_uniq<BoundJoinRef>(ref.ref_type);
	result->left_binder = Binder::CreateBinder(context, this);
	result->right_binder = Binder::CreateBinder(context, this);
	auto &left_binder = *result->left_binder;
	auto &right_binder = *result->right_binder;

	result->type = ref.type;
	result->left = left_binder.Bind(*ref.left);
	{
		LateralBinder binder(left_binder, context);
		result->right = right_binder.Bind(*ref.right);
		result->correlated_columns = binder.ExtractCorrelatedColumns(right_binder);

		result->lateral = binder.HasCorrelatedColumns();
		if (result->lateral) {
			// lateral join: can only be an INNER or LEFT join
			if (ref.type != JoinType::INNER && ref.type != JoinType::LEFT) {
				throw BinderException("The combining JOIN type must be INNER or LEFT for a LATERAL reference");
			}
		}
	}

	vector<unique_ptr<ParsedExpression>> extra_conditions;
	vector<string> extra_using_columns;
	switch (ref.ref_type) {
	case JoinRefType::NATURAL: {
		// natural join, figure out which column names are present in both sides of the join
		// first bind the left hand side and get a list of all the tables and column names
		case_insensitive_set_t lhs_columns;
		auto &lhs_binding_list = left_binder.bind_context.GetBindingsList();
		for (auto &binding : lhs_binding_list) {
			for (auto &column_name : binding.get().names) {
				lhs_columns.insert(column_name);
			}
		}
		// now bind the rhs
		for (auto &column_name : lhs_columns) {
			auto right_using_binding = right_binder.bind_context.GetUsingBinding(column_name);

			string right_binding;
			// loop over the set of lhs columns, and figure out if there is a table in the rhs with the same name
			if (!right_using_binding) {
				if (!right_binder.TryFindBinding(column_name, "right", right_binding)) {
					// no match found for this column on the rhs: skip
					continue;
				}
			}
			extra_using_columns.push_back(column_name);
		}
		if (extra_using_columns.empty()) {
			// no matching bindings found in natural join: throw an exception
			string error_msg = "No columns found to join on in NATURAL JOIN.\n";
			error_msg += "Use CROSS JOIN if you intended for this to be a cross-product.";
			// gather all left/right candidates
			string left_candidates, right_candidates;
			auto &rhs_binding_list = right_binder.bind_context.GetBindingsList();
			for (auto &binding_ref : lhs_binding_list) {
				auto &binding = binding_ref.get();
				for (auto &column_name : binding.names) {
					if (!left_candidates.empty()) {
						left_candidates += ", ";
					}
					left_candidates += binding.alias + "." + column_name;
				}
			}
			for (auto &binding_ref : rhs_binding_list) {
				auto &binding = binding_ref.get();
				for (auto &column_name : binding.names) {
					if (!right_candidates.empty()) {
						right_candidates += ", ";
					}
					right_candidates += binding.alias + "." + column_name;
				}
			}
			error_msg += "\n   Left candidates: " + left_candidates;
			error_msg += "\n   Right candidates: " + right_candidates;
			throw BinderException(FormatError(ref, error_msg));
		}
		break;
	}
	case JoinRefType::REGULAR:
	case JoinRefType::ASOF:
		if (!ref.using_columns.empty()) {
			// USING columns
			D_ASSERT(!result->condition);
			extra_using_columns = ref.using_columns;
		}
		break;

	case JoinRefType::CROSS:
	case JoinRefType::POSITIONAL:
		break;
	}
	extra_using_columns = RemoveDuplicateUsingColumns(extra_using_columns);

	if (!extra_using_columns.empty()) {
		vector<optional_ptr<UsingColumnSet>> left_using_bindings;
		vector<optional_ptr<UsingColumnSet>> right_using_bindings;
		for (idx_t i = 0; i < extra_using_columns.size(); i++) {
			auto &using_column = extra_using_columns[i];
			// we check if there is ALREADY a using column of the same name in the left and right set
			// this can happen if we chain USING clauses
			// e.g. x JOIN y USING (c) JOIN z USING (c)
			auto left_using_binding = left_binder.bind_context.GetUsingBinding(using_column);
			auto right_using_binding = right_binder.bind_context.GetUsingBinding(using_column);
			if (!left_using_binding) {
				left_binder.bind_context.GetMatchingBinding(using_column);
			}
			if (!right_using_binding) {
				right_binder.bind_context.GetMatchingBinding(using_column);
			}
			left_using_bindings.push_back(left_using_binding);
			right_using_bindings.push_back(right_using_binding);
		}

		for (idx_t i = 0; i < extra_using_columns.size(); i++) {
			auto &using_column = extra_using_columns[i];
			string left_binding;
			string right_binding;

			auto set = make_uniq<UsingColumnSet>();
			auto &left_using_binding = left_using_bindings[i];
			auto &right_using_binding = right_using_bindings[i];
			left_binding = RetrieveUsingBinding(left_binder, left_using_binding, using_column, "left");
			right_binding = RetrieveUsingBinding(right_binder, right_using_binding, using_column, "right");

			// Last column of ASOF JOIN ... USING is >=
			const auto type = (ref.ref_type == JoinRefType::ASOF && i == extra_using_columns.size() - 1)
			                      ? ExpressionType::COMPARE_GREATERTHANOREQUALTO
			                      : ExpressionType::COMPARE_EQUAL;

			extra_conditions.push_back(
			    AddCondition(context, left_binder, right_binder, left_binding, right_binding, using_column, type));

			AddUsingBindings(*set, left_using_binding, left_binding);
			AddUsingBindings(*set, right_using_binding, right_binding);
			SetPrimaryBinding(*set, ref.type, left_binding, right_binding);
			bind_context.TransferUsingBinding(left_binder.bind_context, left_using_binding, *set, left_binding,
			                                  using_column);
			bind_context.TransferUsingBinding(right_binder.bind_context, right_using_binding, *set, right_binding,
			                                  using_column);
			AddUsingBindingSet(std::move(set));
		}
	}

	auto right_bindings_list_copy = right_binder.bind_context.GetBindingsList();

	bind_context.AddContext(std::move(left_binder.bind_context));
	bind_context.AddContext(std::move(right_binder.bind_context));
	MoveCorrelatedExpressions(left_binder);
	MoveCorrelatedExpressions(right_binder);
	for (auto &condition : extra_conditions) {
		if (ref.condition) {
			ref.condition = make_uniq<ConjunctionExpression>(ExpressionType::CONJUNCTION_AND, std::move(ref.condition),
			                                                 std::move(condition));
		} else {
			ref.condition = std::move(condition);
		}
	}
	if (ref.condition) {
		WhereBinder binder(*this, context);
		result->condition = binder.Bind(ref.condition);
	}

	if (result->type == JoinType::SEMI || result->type == JoinType::ANTI) {
		bind_context.RemoveContext(right_bindings_list_copy);
	}

	return std::move(result);
}

} // namespace duckdb


namespace duckdb {

void Binder::BindNamedParameters(named_parameter_type_map_t &types, named_parameter_map_t &values,
                                 QueryErrorContext &error_context, string &func_name) {
	for (auto &kv : values) {
		auto entry = types.find(kv.first);
		if (entry == types.end()) {
			// create a list of named parameters for the error
			string named_params;
			for (auto &kv : types) {
				named_params += "    ";
				named_params += kv.first;
				named_params += " ";
				named_params += kv.second.ToString();
				named_params += "\n";
			}
			string error_msg;
			if (named_params.empty()) {
				error_msg = "Function does not accept any named parameters.";
			} else {
				error_msg = "Candidates:\n" + named_params;
			}
			throw BinderException(error_context.FormatError("Invalid named parameter \"%s\" for function %s\n%s",
			                                                kv.first, func_name, error_msg));
		}
		if (entry->second.id() != LogicalTypeId::ANY) {
			kv.second = kv.second.DefaultCastAs(entry->second);
		}
	}
}

} // namespace duckdb





















namespace duckdb {

static void ConstructPivots(PivotRef &ref, vector<PivotValueElement> &pivot_values, idx_t pivot_idx = 0,
                            const PivotValueElement &current_value = PivotValueElement()) {
	auto &pivot = ref.pivots[pivot_idx];
	bool last_pivot = pivot_idx + 1 == ref.pivots.size();
	for (auto &entry : pivot.entries) {
		PivotValueElement new_value = current_value;
		string name = entry.alias;
		D_ASSERT(entry.values.size() == pivot.pivot_expressions.size());
		for (idx_t v = 0; v < entry.values.size(); v++) {
			auto &value = entry.values[v];
			new_value.values.push_back(value);
			if (entry.alias.empty()) {
				if (name.empty()) {
					name = value.ToString();
				} else {
					name += "_" + value.ToString();
				}
			}
		}
		if (!current_value.name.empty()) {
			new_value.name = current_value.name + "_" + name;
		} else {
			new_value.name = std::move(name);
		}
		if (last_pivot) {
			pivot_values.push_back(std::move(new_value));
		} else {
			// need to recurse
			ConstructPivots(ref, pivot_values, pivot_idx + 1, new_value);
		}
	}
}

static void ExtractPivotExpressions(ParsedExpression &expr, case_insensitive_set_t &handled_columns) {
	if (expr.type == ExpressionType::COLUMN_REF) {
		auto &child_colref = expr.Cast<ColumnRefExpression>();
		if (child_colref.IsQualified()) {
			throw BinderException("PIVOT expression cannot contain qualified columns");
		}
		handled_columns.insert(child_colref.GetColumnName());
	}
	ParsedExpressionIterator::EnumerateChildren(
	    expr, [&](ParsedExpression &child) { ExtractPivotExpressions(child, handled_columns); });
}

struct PivotBindState {
	vector<string> internal_group_names;
	vector<string> group_names;
	vector<string> aggregate_names;
	vector<string> internal_aggregate_names;
};

static unique_ptr<SelectNode> PivotInitialAggregate(PivotBindState &bind_state, PivotRef &ref,
                                                    vector<unique_ptr<ParsedExpression>> all_columns,
                                                    const case_insensitive_set_t &handled_columns) {
	auto subquery_stage1 = make_uniq<SelectNode>();
	subquery_stage1->from_table = std::move(ref.source);
	if (ref.groups.empty()) {
		// if rows are not specified any columns that are not pivoted/aggregated on are added to the GROUP BY clause
		for (auto &entry : all_columns) {
			if (entry->type != ExpressionType::COLUMN_REF) {
				throw InternalException("Unexpected child of pivot source - not a ColumnRef");
			}
			auto &columnref = (ColumnRefExpression &)*entry;
			if (handled_columns.find(columnref.GetColumnName()) == handled_columns.end()) {
				// not handled - add to grouping set
				subquery_stage1->groups.group_expressions.push_back(
				    make_uniq<ConstantExpression>(Value::INTEGER(subquery_stage1->select_list.size() + 1)));
				subquery_stage1->select_list.push_back(make_uniq<ColumnRefExpression>(columnref.GetColumnName()));
			}
		}
	} else {
		// if rows are specified only the columns mentioned in rows are added as groups
		for (auto &row : ref.groups) {
			subquery_stage1->groups.group_expressions.push_back(
			    make_uniq<ConstantExpression>(Value::INTEGER(subquery_stage1->select_list.size() + 1)));
			subquery_stage1->select_list.push_back(make_uniq<ColumnRefExpression>(row));
		}
	}
	idx_t group_count = 0;
	for (auto &expr : subquery_stage1->select_list) {
		bind_state.group_names.push_back(expr->GetName());
		if (expr->alias.empty()) {
			expr->alias = "__internal_pivot_group" + std::to_string(++group_count);
		}
		bind_state.internal_group_names.push_back(expr->alias);
	}
	// group by all of the pivot values
	idx_t pivot_count = 0;
	for (auto &pivot_column : ref.pivots) {
		for (auto &pivot_expr : pivot_column.pivot_expressions) {
			if (pivot_expr->alias.empty()) {
				pivot_expr->alias = "__internal_pivot_ref" + std::to_string(++pivot_count);
			}
			auto pivot_alias = pivot_expr->alias;
			subquery_stage1->groups.group_expressions.push_back(
			    make_uniq<ConstantExpression>(Value::INTEGER(subquery_stage1->select_list.size() + 1)));
			subquery_stage1->select_list.push_back(std::move(pivot_expr));
			pivot_expr = make_uniq<ColumnRefExpression>(std::move(pivot_alias));
		}
	}
	idx_t aggregate_count = 0;
	// finally add the aggregates
	for (auto &aggregate : ref.aggregates) {
		auto aggregate_alias = "__internal_pivot_aggregate" + std::to_string(++aggregate_count);
		bind_state.aggregate_names.push_back(aggregate->alias);
		bind_state.internal_aggregate_names.push_back(aggregate_alias);
		aggregate->alias = std::move(aggregate_alias);
		subquery_stage1->select_list.push_back(std::move(aggregate));
	}
	return subquery_stage1;
}

static unique_ptr<SelectNode> PivotListAggregate(PivotBindState &bind_state, PivotRef &ref,
                                                 unique_ptr<SelectNode> subquery_stage1) {
	auto subquery_stage2 = make_uniq<SelectNode>();
	// wrap the subquery of stage 1
	auto subquery_select = make_uniq<SelectStatement>();
	subquery_select->node = std::move(subquery_stage1);
	auto subquery_ref = make_uniq<SubqueryRef>(std::move(subquery_select));

	// add all of the groups
	for (idx_t gr = 0; gr < bind_state.internal_group_names.size(); gr++) {
		subquery_stage2->groups.group_expressions.push_back(
		    make_uniq<ConstantExpression>(Value::INTEGER(subquery_stage2->select_list.size() + 1)));
		auto group_reference = make_uniq<ColumnRefExpression>(bind_state.internal_group_names[gr]);
		group_reference->alias = bind_state.internal_group_names[gr];
		subquery_stage2->select_list.push_back(std::move(group_reference));
	}

	// construct the list aggregates
	for (idx_t aggr = 0; aggr < bind_state.internal_aggregate_names.size(); aggr++) {
		auto colref = make_uniq<ColumnRefExpression>(bind_state.internal_aggregate_names[aggr]);
		vector<unique_ptr<ParsedExpression>> list_children;
		list_children.push_back(std::move(colref));
		auto aggregate = make_uniq<FunctionExpression>("list", std::move(list_children));
		aggregate->alias = bind_state.internal_aggregate_names[aggr];
		subquery_stage2->select_list.push_back(std::move(aggregate));
	}
	// construct the pivot list
	auto pivot_name = "__internal_pivot_name";
	unique_ptr<ParsedExpression> expr;
	for (auto &pivot : ref.pivots) {
		for (auto &pivot_expr : pivot.pivot_expressions) {
			// coalesce(pivot::VARCHAR, 'NULL')
			auto cast = make_uniq<CastExpression>(LogicalType::VARCHAR, std::move(pivot_expr));
			vector<unique_ptr<ParsedExpression>> coalesce_children;
			coalesce_children.push_back(std::move(cast));
			coalesce_children.push_back(make_uniq<ConstantExpression>(Value("NULL")));
			auto coalesce =
			    make_uniq<OperatorExpression>(ExpressionType::OPERATOR_COALESCE, std::move(coalesce_children));

			if (!expr) {
				expr = std::move(coalesce);
			} else {
				// string concat
				vector<unique_ptr<ParsedExpression>> concat_children;
				concat_children.push_back(std::move(expr));
				concat_children.push_back(make_uniq<ConstantExpression>(Value("_")));
				concat_children.push_back(std::move(coalesce));
				auto concat = make_uniq<FunctionExpression>("concat", std::move(concat_children));
				expr = std::move(concat);
			}
		}
	}
	// list(coalesce)
	vector<unique_ptr<ParsedExpression>> list_children;
	list_children.push_back(std::move(expr));
	auto aggregate = make_uniq<FunctionExpression>("list", std::move(list_children));

	aggregate->alias = pivot_name;
	subquery_stage2->select_list.push_back(std::move(aggregate));

	subquery_stage2->from_table = std::move(subquery_ref);
	return subquery_stage2;
}

static unique_ptr<SelectNode> PivotFinalOperator(PivotBindState &bind_state, PivotRef &ref,
                                                 unique_ptr<SelectNode> subquery,
                                                 vector<PivotValueElement> pivot_values) {
	auto final_pivot_operator = make_uniq<SelectNode>();
	// wrap the subquery of stage 1
	auto subquery_select = make_uniq<SelectStatement>();
	subquery_select->node = std::move(subquery);
	auto subquery_ref = make_uniq<SubqueryRef>(std::move(subquery_select));

	auto bound_pivot = make_uniq<PivotRef>();
	bound_pivot->bound_pivot_values = std::move(pivot_values);
	bound_pivot->bound_group_names = std::move(bind_state.group_names);
	bound_pivot->bound_aggregate_names = std::move(bind_state.aggregate_names);
	bound_pivot->source = std::move(subquery_ref);

	final_pivot_operator->select_list.push_back(make_uniq<StarExpression>());
	final_pivot_operator->from_table = std::move(bound_pivot);
	return final_pivot_operator;
}

void ExtractPivotAggregates(BoundTableRef &node, vector<unique_ptr<Expression>> &aggregates) {
	if (node.type != TableReferenceType::SUBQUERY) {
		throw InternalException("Pivot - Expected a subquery");
	}
	auto &subq = node.Cast<BoundSubqueryRef>();
	if (subq.subquery->type != QueryNodeType::SELECT_NODE) {
		throw InternalException("Pivot - Expected a select node");
	}
	auto &select = subq.subquery->Cast<BoundSelectNode>();
	if (select.from_table->type != TableReferenceType::SUBQUERY) {
		throw InternalException("Pivot - Expected another subquery");
	}
	auto &subq2 = select.from_table->Cast<BoundSubqueryRef>();
	if (subq2.subquery->type != QueryNodeType::SELECT_NODE) {
		throw InternalException("Pivot - Expected another select node");
	}
	auto &select2 = subq2.subquery->Cast<BoundSelectNode>();
	for (auto &aggr : select2.aggregates) {
		aggregates.push_back(aggr->Copy());
	}
}

unique_ptr<BoundTableRef> Binder::BindBoundPivot(PivotRef &ref) {
	// bind the child table in a child binder
	auto result = make_uniq<BoundPivotRef>();
	result->bind_index = GenerateTableIndex();
	result->child_binder = Binder::CreateBinder(context, this);
	result->child = result->child_binder->Bind(*ref.source);

	auto &aggregates = result->bound_pivot.aggregates;
	ExtractPivotAggregates(*result->child, aggregates);
	if (aggregates.size() != ref.bound_aggregate_names.size()) {
		throw InternalException("Pivot - aggregate count mismatch");
	}

	vector<string> child_names;
	vector<LogicalType> child_types;
	result->child_binder->bind_context.GetTypesAndNames(child_names, child_types);

	vector<string> names;
	vector<LogicalType> types;
	// emit the groups
	for (idx_t i = 0; i < ref.bound_group_names.size(); i++) {
		names.push_back(ref.bound_group_names[i]);
		types.push_back(child_types[i]);
	}
	// emit the pivot columns
	for (auto &pivot_value : ref.bound_pivot_values) {
		for (idx_t aggr_idx = 0; aggr_idx < ref.bound_aggregate_names.size(); aggr_idx++) {
			auto &aggr = aggregates[aggr_idx];
			auto &aggr_name = ref.bound_aggregate_names[aggr_idx];
			auto name = pivot_value.name;
			if (aggregates.size() > 1 || !aggr_name.empty()) {
				// if there are multiple aggregates specified we add the name of the aggregate as well
				name += "_" + (aggr_name.empty() ? aggr->GetName() : aggr_name);
			}
			string pivot_str;
			for (auto &value : pivot_value.values) {
				auto str = value.ToString();
				if (pivot_str.empty()) {
					pivot_str = std::move(str);
				} else {
					pivot_str += "_" + str;
				}
			}
			result->bound_pivot.pivot_values.push_back(std::move(pivot_str));
			names.push_back(std::move(name));
			types.push_back(aggr->return_type);
		}
	}
	result->bound_pivot.group_count = ref.bound_group_names.size();
	result->bound_pivot.types = types;
	auto subquery_alias = ref.alias.empty() ? "__unnamed_pivot" : ref.alias;
	bind_context.AddGenericBinding(result->bind_index, subquery_alias, names, types);
	MoveCorrelatedExpressions(*result->child_binder);
	return std::move(result);
}

unique_ptr<SelectNode> Binder::BindPivot(PivotRef &ref, vector<unique_ptr<ParsedExpression>> all_columns) {
	// keep track of the columns by which we pivot/aggregate
	// any columns which are not pivoted/aggregated on are added to the GROUP BY clause
	case_insensitive_set_t handled_columns;
	// parse the aggregate, and extract the referenced columns from the aggregate
	for (auto &aggr : ref.aggregates) {
		if (aggr->type != ExpressionType::FUNCTION) {
			throw BinderException(FormatError(*aggr, "Pivot expression must be an aggregate"));
		}
		if (aggr->HasSubquery()) {
			throw BinderException(FormatError(*aggr, "Pivot expression cannot contain subqueries"));
		}
		if (aggr->IsWindow()) {
			throw BinderException(FormatError(*aggr, "Pivot expression cannot contain window functions"));
		}
		ExtractPivotExpressions(*aggr, handled_columns);
	}
	value_set_t pivots;

	// first add all pivots to the set of handled columns, and check for duplicates
	idx_t total_pivots = 1;
	for (auto &pivot : ref.pivots) {
		if (!pivot.pivot_enum.empty()) {
			auto type = Catalog::GetType(context, INVALID_CATALOG, INVALID_SCHEMA, pivot.pivot_enum);
			if (type.id() != LogicalTypeId::ENUM) {
				throw BinderException(
				    FormatError(ref, StringUtil::Format("Pivot must reference an ENUM type: \"%s\" is of type \"%s\"",
				                                        pivot.pivot_enum, type.ToString())));
			}
			auto enum_size = EnumType::GetSize(type);
			for (idx_t i = 0; i < enum_size; i++) {
				auto enum_value = EnumType::GetValue(Value::ENUM(i, type));
				PivotColumnEntry entry;
				entry.values.emplace_back(enum_value);
				entry.alias = std::move(enum_value);
				pivot.entries.push_back(std::move(entry));
			}
		}
		total_pivots *= pivot.entries.size();
		// add the pivoted column to the columns that have been handled
		for (auto &pivot_name : pivot.pivot_expressions) {
			ExtractPivotExpressions(*pivot_name, handled_columns);
		}
		value_set_t pivots;
		for (auto &entry : pivot.entries) {
			D_ASSERT(!entry.star_expr);
			Value val;
			if (entry.values.size() == 1) {
				val = entry.values[0];
			} else {
				val = Value::LIST(LogicalType::VARCHAR, entry.values);
			}
			if (pivots.find(val) != pivots.end()) {
				throw BinderException(FormatError(
				    ref, StringUtil::Format("The value \"%s\" was specified multiple times in the IN clause",
				                            val.ToString())));
			}
			if (entry.values.size() != pivot.pivot_expressions.size()) {
				throw ParserException("PIVOT IN list - inconsistent amount of rows - expected %d but got %d",
				                      pivot.pivot_expressions.size(), entry.values.size());
			}
			pivots.insert(val);
		}
	}
	auto pivot_limit = ClientConfig::GetConfig(context).pivot_limit;
	if (total_pivots >= pivot_limit) {
		throw BinderException("Pivot column limit of %llu exceeded. Use SET pivot_limit=X to increase the limit.",
		                      ClientConfig::GetConfig(context).pivot_limit);
	}

	// construct the required pivot values recursively
	vector<PivotValueElement> pivot_values;
	ConstructPivots(ref, pivot_values);

	// pivots have three components
	// - the pivots (i.e. future column names)
	// - the groups (i.e. the future row names
	// - the aggregates (i.e. the values of the pivot columns)

	// executing a pivot statement happens in three stages
	// 1) execute the query "SELECT {groups}, {pivots}, {aggregates} FROM {from_clause} GROUP BY {groups}, {pivots}
	// this computes all values that are required in the final result, but not yet in the correct orientation
	// 2) execute the query "SELECT {groups}, LIST({pivots}), LIST({aggregates}) FROM [Q1] GROUP BY {groups}
	// this pushes all pivots and aggregates that belong to a specific group together in an aligned manner
	// 3) push a PIVOT operator, that performs the actual pivoting of the values into the different columns

	PivotBindState bind_state;
	// Pivot Stage 1
	// SELECT {groups}, {pivots}, {aggregates} FROM {from_clause} GROUP BY {groups}, {pivots}
	auto subquery_stage1 = PivotInitialAggregate(bind_state, ref, std::move(all_columns), handled_columns);

	// Pivot stage 2
	// SELECT {groups}, LIST({pivots}), LIST({aggregates}) FROM [Q1] GROUP BY {groups}
	auto subquery_stage2 = PivotListAggregate(bind_state, ref, std::move(subquery_stage1));

	// Pivot stage 3
	// construct the final pivot operator
	auto pivot_node = PivotFinalOperator(bind_state, ref, std::move(subquery_stage2), std::move(pivot_values));
	return pivot_node;
}

unique_ptr<SelectNode> Binder::BindUnpivot(Binder &child_binder, PivotRef &ref,
                                           vector<unique_ptr<ParsedExpression>> all_columns,
                                           unique_ptr<ParsedExpression> &where_clause) {
	D_ASSERT(ref.groups.empty());
	D_ASSERT(ref.pivots.size() == 1);

	unique_ptr<ParsedExpression> expr;
	auto select_node = make_uniq<SelectNode>();
	select_node->from_table = std::move(ref.source);

	// handle the pivot
	auto &unpivot = ref.pivots[0];

	// handle star expressions in any entries
	vector<PivotColumnEntry> new_entries;
	for (auto &entry : unpivot.entries) {
		if (entry.star_expr) {
			D_ASSERT(entry.values.empty());
			vector<unique_ptr<ParsedExpression>> star_columns;
			child_binder.ExpandStarExpression(std::move(entry.star_expr), star_columns);

			for (auto &col : star_columns) {
				if (col->type != ExpressionType::COLUMN_REF) {
					throw InternalException("Unexpected child of unpivot star - not a ColumnRef");
				}
				auto &columnref = col->Cast<ColumnRefExpression>();
				PivotColumnEntry new_entry;
				new_entry.values.emplace_back(columnref.GetColumnName());
				new_entry.alias = columnref.GetColumnName();
				new_entries.push_back(std::move(new_entry));
			}
		} else {
			new_entries.push_back(std::move(entry));
		}
	}
	unpivot.entries = std::move(new_entries);

	case_insensitive_set_t handled_columns;
	case_insensitive_map_t<string> name_map;
	for (auto &entry : unpivot.entries) {
		for (auto &value : entry.values) {
			handled_columns.insert(value.ToString());
		}
	}

	for (auto &col_expr : all_columns) {
		if (col_expr->type != ExpressionType::COLUMN_REF) {
			throw InternalException("Unexpected child of pivot source - not a ColumnRef");
		}
		auto &columnref = col_expr->Cast<ColumnRefExpression>();
		auto &column_name = columnref.GetColumnName();
		auto entry = handled_columns.find(column_name);
		if (entry == handled_columns.end()) {
			// not handled - add to the set of regularly selected columns
			select_node->select_list.push_back(std::move(col_expr));
		} else {
			name_map[column_name] = column_name;
			handled_columns.erase(entry);
		}
	}
	if (!handled_columns.empty()) {
		for (auto &entry : handled_columns) {
			throw BinderException("Column \"%s\" referenced in UNPIVOT but no matching entry was found in the table",
			                      entry);
		}
	}
	vector<Value> unpivot_names;
	for (auto &entry : unpivot.entries) {
		string generated_name;
		for (auto &val : entry.values) {
			auto name_entry = name_map.find(val.ToString());
			if (name_entry == name_map.end()) {
				throw InternalException("Unpivot - could not find column name in name map");
			}
			if (!generated_name.empty()) {
				generated_name += "_";
			}
			generated_name += name_entry->second;
		}
		unpivot_names.emplace_back(!entry.alias.empty() ? entry.alias : generated_name);
	}
	vector<vector<unique_ptr<ParsedExpression>>> unpivot_expressions;
	for (idx_t v_idx = 1; v_idx < unpivot.entries.size(); v_idx++) {
		if (unpivot.entries[v_idx].values.size() != unpivot.entries[0].values.size()) {
			throw BinderException(
			    "UNPIVOT value count mismatch - entry has %llu values, but expected all entries to have %llu values",
			    unpivot.entries[v_idx].values.size(), unpivot.entries[0].values.size());
		}
	}

	for (idx_t v_idx = 0; v_idx < unpivot.entries[0].values.size(); v_idx++) {
		vector<unique_ptr<ParsedExpression>> expressions;
		expressions.reserve(unpivot.entries.size());
		for (auto &entry : unpivot.entries) {
			expressions.push_back(make_uniq<ColumnRefExpression>(entry.values[v_idx].ToString()));
		}
		unpivot_expressions.push_back(std::move(expressions));
	}

	// construct the UNNEST expression for the set of names (constant)
	auto unpivot_list = Value::LIST(LogicalType::VARCHAR, std::move(unpivot_names));
	auto unpivot_name_expr = make_uniq<ConstantExpression>(std::move(unpivot_list));
	vector<unique_ptr<ParsedExpression>> unnest_name_children;
	unnest_name_children.push_back(std::move(unpivot_name_expr));
	auto unnest_name_expr = make_uniq<FunctionExpression>("unnest", std::move(unnest_name_children));
	unnest_name_expr->alias = unpivot.unpivot_names[0];
	select_node->select_list.push_back(std::move(unnest_name_expr));

	// construct the UNNEST expression for the set of unpivoted columns
	if (ref.unpivot_names.size() != unpivot_expressions.size()) {
		throw BinderException("UNPIVOT name count mismatch - got %d names but %d expressions", ref.unpivot_names.size(),
		                      unpivot_expressions.size());
	}
	for (idx_t i = 0; i < unpivot_expressions.size(); i++) {
		auto list_expr = make_uniq<FunctionExpression>("list_value", std::move(unpivot_expressions[i]));
		vector<unique_ptr<ParsedExpression>> unnest_val_children;
		unnest_val_children.push_back(std::move(list_expr));
		auto unnest_val_expr = make_uniq<FunctionExpression>("unnest", std::move(unnest_val_children));
		auto unnest_name = i < ref.column_name_alias.size() ? ref.column_name_alias[i] : ref.unpivot_names[i];
		unnest_val_expr->alias = unnest_name;
		select_node->select_list.push_back(std::move(unnest_val_expr));
		if (!ref.include_nulls) {
			// if we are running with EXCLUDE NULLS we need to add an IS NOT NULL filter
			auto colref = make_uniq<ColumnRefExpression>(unnest_name);
			auto filter = make_uniq<OperatorExpression>(ExpressionType::OPERATOR_IS_NOT_NULL, std::move(colref));
			if (where_clause) {
				where_clause = make_uniq<ConjunctionExpression>(ExpressionType::CONJUNCTION_AND,
				                                                std::move(where_clause), std::move(filter));
			} else {
				where_clause = std::move(filter);
			}
		}
	}
	return select_node;
}

unique_ptr<BoundTableRef> Binder::Bind(PivotRef &ref) {
	if (!ref.source) {
		throw InternalException("Pivot without a source!?");
	}
	if (!ref.bound_pivot_values.empty() || !ref.bound_group_names.empty() || !ref.bound_aggregate_names.empty()) {
		// bound pivot
		return BindBoundPivot(ref);
	}

	// bind the source of the pivot
	// we need to do this to be able to expand star expressions
	auto copied_source = ref.source->Copy();
	auto star_binder = Binder::CreateBinder(context, this);
	star_binder->Bind(*copied_source);

	// figure out the set of column names that are in the source of the pivot
	vector<unique_ptr<ParsedExpression>> all_columns;
	star_binder->ExpandStarExpression(make_uniq<StarExpression>(), all_columns);

	unique_ptr<SelectNode> select_node;
	unique_ptr<ParsedExpression> where_clause;
	if (!ref.aggregates.empty()) {
		select_node = BindPivot(ref, std::move(all_columns));
	} else {
		select_node = BindUnpivot(*star_binder, ref, std::move(all_columns), where_clause);
	}
	// bind the generated select node
	auto child_binder = Binder::CreateBinder(context, this);
	auto bound_select_node = child_binder->BindNode(*select_node);
	auto root_index = bound_select_node->GetRootIndex();
	BoundQueryNode *bound_select_ptr = bound_select_node.get();

	unique_ptr<BoundTableRef> result;
	MoveCorrelatedExpressions(*child_binder);
	result = make_uniq<BoundSubqueryRef>(std::move(child_binder), std::move(bound_select_node));
	auto subquery_alias = ref.alias.empty() ? "__unnamed_pivot" : ref.alias;
	SubqueryRef subquery_ref(nullptr, subquery_alias);
	subquery_ref.column_name_alias = std::move(ref.column_name_alias);
	if (where_clause) {
		// if a WHERE clause was provided - bind a subquery holding the WHERE clause
		// we need to bind a new subquery here because the WHERE clause has to be applied AFTER the unnest
		child_binder = Binder::CreateBinder(context, this);
		child_binder->bind_context.AddSubquery(root_index, subquery_ref.alias, subquery_ref, *bound_select_ptr);
		auto where_query = make_uniq<SelectNode>();
		where_query->select_list.push_back(make_uniq<StarExpression>());
		where_query->where_clause = std::move(where_clause);
		bound_select_node = child_binder->BindSelectNode(*where_query, std::move(result));
		bound_select_ptr = bound_select_node.get();
		root_index = bound_select_node->GetRootIndex();
		result = make_uniq<BoundSubqueryRef>(std::move(child_binder), std::move(bound_select_node));
	}
	bind_context.AddSubquery(root_index, subquery_ref.alias, subquery_ref, *bound_select_ptr);
	return result;
}

} // namespace duckdb




namespace duckdb {

unique_ptr<BoundTableRef> Binder::Bind(SubqueryRef &ref, optional_ptr<CommonTableExpressionInfo> cte) {
	auto binder = Binder::CreateBinder(context, this);
	binder->can_contain_nulls = true;
	if (cte) {
		binder->bound_ctes.insert(*cte);
	}
	binder->alias = ref.alias.empty() ? "unnamed_subquery" : ref.alias;
	auto subquery = binder->BindNode(*ref.subquery->node);
	idx_t bind_index = subquery->GetRootIndex();
	string subquery_alias;
	if (ref.alias.empty()) {
		subquery_alias = "unnamed_subquery" + to_string(bind_index);
	} else {
		subquery_alias = ref.alias;
	}
	auto result = make_uniq<BoundSubqueryRef>(std::move(binder), std::move(subquery));
	bind_context.AddSubquery(bind_index, subquery_alias, ref, *result->subquery);
	MoveCorrelatedExpressions(*result->binder);
	return std::move(result);
}

} // namespace duckdb























namespace duckdb {

static bool IsTableInTableOutFunction(TableFunctionCatalogEntry &table_function) {
	auto fun = table_function.functions.GetFunctionByOffset(0);
	return table_function.functions.Size() == 1 && fun.arguments.size() == 1 &&
	       fun.arguments[0].id() == LogicalTypeId::TABLE;
}

bool Binder::BindTableInTableOutFunction(vector<unique_ptr<ParsedExpression>> &expressions,
                                         unique_ptr<BoundSubqueryRef> &subquery, string &error) {
	auto binder = Binder::CreateBinder(this->context, this, true);
	unique_ptr<QueryNode> subquery_node;
	if (expressions.size() == 1 && expressions[0]->type == ExpressionType::SUBQUERY) {
		// general case: argument is a subquery, bind it as part of the node
		auto &se = expressions[0]->Cast<SubqueryExpression>();
		subquery_node = std::move(se.subquery->node);
	} else {
		// special case: non-subquery parameter to table-in table-out function
		// generate a subquery and bind that (i.e. UNNEST([1,2,3]) becomes UNNEST((SELECT [1,2,3]))
		auto select_node = make_uniq<SelectNode>();
		select_node->select_list = std::move(expressions);
		select_node->from_table = make_uniq<EmptyTableRef>();
		subquery_node = std::move(select_node);
	}
	auto node = binder->BindNode(*subquery_node);
	subquery = make_uniq<BoundSubqueryRef>(std::move(binder), std::move(node));
	MoveCorrelatedExpressions(*subquery->binder);
	return true;
}

bool Binder::BindTableFunctionParameters(TableFunctionCatalogEntry &table_function,
                                         vector<unique_ptr<ParsedExpression>> &expressions,
                                         vector<LogicalType> &arguments, vector<Value> &parameters,
                                         named_parameter_map_t &named_parameters,
                                         unique_ptr<BoundSubqueryRef> &subquery, string &error) {
	if (IsTableInTableOutFunction(table_function)) {
		// special case binding for table-in table-out function
		arguments.emplace_back(LogicalTypeId::TABLE);
		return BindTableInTableOutFunction(expressions, subquery, error);
	}
	bool seen_subquery = false;
	for (auto &child : expressions) {
		string parameter_name;

		// hack to make named parameters work
		if (child->type == ExpressionType::COMPARE_EQUAL) {
			// comparison, check if the LHS is a columnref
			auto &comp = child->Cast<ComparisonExpression>();
			if (comp.left->type == ExpressionType::COLUMN_REF) {
				auto &colref = comp.left->Cast<ColumnRefExpression>();
				if (!colref.IsQualified()) {
					parameter_name = colref.GetColumnName();
					child = std::move(comp.right);
				}
			}
		}
		if (child->type == ExpressionType::SUBQUERY) {
			if (seen_subquery) {
				error = "Table function can have at most one subquery parameter ";
				return false;
			}
			auto binder = Binder::CreateBinder(this->context, this, true);
			auto &se = child->Cast<SubqueryExpression>();
			auto node = binder->BindNode(*se.subquery->node);
			subquery = make_uniq<BoundSubqueryRef>(std::move(binder), std::move(node));
			seen_subquery = true;
			arguments.emplace_back(LogicalTypeId::TABLE);
			continue;
		}

		TableFunctionBinder binder(*this, context);
		LogicalType sql_type;
		auto expr = binder.Bind(child, &sql_type);
		if (expr->HasParameter()) {
			throw ParameterNotResolvedException();
		}
		if (!expr->IsScalar()) {
			error = "Table function requires a constant parameter";
			return false;
		}
		auto constant = ExpressionExecutor::EvaluateScalar(context, *expr, true);
		if (parameter_name.empty()) {
			// unnamed parameter
			if (!named_parameters.empty()) {
				error = "Unnamed parameters cannot come after named parameters";
				return false;
			}
			arguments.emplace_back(sql_type);
			parameters.emplace_back(std::move(constant));
		} else {
			named_parameters[parameter_name] = std::move(constant);
		}
	}
	return true;
}

unique_ptr<LogicalOperator>
Binder::BindTableFunctionInternal(TableFunction &table_function, const string &function_name, vector<Value> parameters,
                                  named_parameter_map_t named_parameters, vector<LogicalType> input_table_types,
                                  vector<string> input_table_names, const vector<string> &column_name_alias,
                                  unique_ptr<ExternalDependency> external_dependency) {
	auto bind_index = GenerateTableIndex();
	// perform the binding
	unique_ptr<FunctionData> bind_data;
	vector<LogicalType> return_types;
	vector<string> return_names;
	if (table_function.bind || table_function.bind_replace) {
		TableFunctionBindInput bind_input(parameters, named_parameters, input_table_types, input_table_names,
		                                  table_function.function_info.get());
		if (table_function.bind_replace) {
			auto new_plan = table_function.bind_replace(context, bind_input);
			if (new_plan != nullptr) {
				return CreatePlan(*Bind(*new_plan));
			} else if (!table_function.bind) {
				throw BinderException("Failed to bind \"%s\": nullptr returned from bind_replace without bind function",
				                      table_function.name);
			}
		}
		bind_data = table_function.bind(context, bind_input, return_types, return_names);
		if (table_function.name == "pandas_scan" || table_function.name == "arrow_scan") {
			auto arrow_bind = (PyTableFunctionData *)bind_data.get();
			arrow_bind->external_dependency = std::move(external_dependency);
		}
		if (table_function.name == "read_csv" || table_function.name == "read_csv_auto") {
			auto &csv_bind = bind_data->Cast<ReadCSVData>();
			if (csv_bind.single_threaded) {
				table_function.extra_info = "(Single-Threaded)";
			} else {
				table_function.extra_info = "(Multi-Threaded)";
			}
		}
	}
	if (return_types.size() != return_names.size()) {
		throw InternalException(
		    "Failed to bind \"%s\": Table function return_types and return_names must be of the same size",
		    table_function.name);
	}
	if (return_types.empty()) {
		throw InternalException("Failed to bind \"%s\": Table function must return at least one column",
		                        table_function.name);
	}
	// overwrite the names with any supplied aliases
	for (idx_t i = 0; i < column_name_alias.size() && i < return_names.size(); i++) {
		return_names[i] = column_name_alias[i];
	}
	for (idx_t i = 0; i < return_names.size(); i++) {
		if (return_names[i].empty()) {
			return_names[i] = "C" + to_string(i);
		}
	}

	auto get = make_uniq<LogicalGet>(bind_index, table_function, std::move(bind_data), return_types, return_names);
	get->parameters = parameters;
	get->named_parameters = named_parameters;
	get->input_table_types = input_table_types;
	get->input_table_names = input_table_names;
	if (table_function.in_out_function && !table_function.projection_pushdown) {
		get->column_ids.reserve(return_types.size());
		for (idx_t i = 0; i < return_types.size(); i++) {
			get->column_ids.push_back(i);
		}
	}
	// now add the table function to the bind context so its columns can be bound
	bind_context.AddTableFunction(bind_index, function_name, return_names, return_types, get->column_ids,
	                              get->GetTable().get());
	return std::move(get);
}

unique_ptr<LogicalOperator> Binder::BindTableFunction(TableFunction &function, vector<Value> parameters) {
	named_parameter_map_t named_parameters;
	vector<LogicalType> input_table_types;
	vector<string> input_table_names;
	vector<string> column_name_aliases;
	return BindTableFunctionInternal(function, function.name, std::move(parameters), std::move(named_parameters),
	                                 std::move(input_table_types), std::move(input_table_names), column_name_aliases,
	                                 nullptr);
}

unique_ptr<BoundTableRef> Binder::Bind(TableFunctionRef &ref) {
	QueryErrorContext error_context(root_statement, ref.query_location);

	D_ASSERT(ref.function->type == ExpressionType::FUNCTION);
	auto &fexpr = ref.function->Cast<FunctionExpression>();

	// fetch the function from the catalog
	auto &func_catalog = Catalog::GetEntry(context, CatalogType::TABLE_FUNCTION_ENTRY, fexpr.catalog, fexpr.schema,
	                                       fexpr.function_name, error_context);

	if (func_catalog.type == CatalogType::TABLE_MACRO_ENTRY) {
		auto &macro_func = func_catalog.Cast<TableMacroCatalogEntry>();
		auto query_node = BindTableMacro(fexpr, macro_func, 0);
		D_ASSERT(query_node);

		auto binder = Binder::CreateBinder(context, this);
		binder->can_contain_nulls = true;

		binder->alias = ref.alias.empty() ? "unnamed_query" : ref.alias;
		auto query = binder->BindNode(*query_node);

		idx_t bind_index = query->GetRootIndex();
		// string alias;
		string alias = (ref.alias.empty() ? "unnamed_query" + to_string(bind_index) : ref.alias);

		auto result = make_uniq<BoundSubqueryRef>(std::move(binder), std::move(query));
		// remember ref here is TableFunctionRef and NOT base class
		bind_context.AddSubquery(bind_index, alias, ref, *result->subquery);
		MoveCorrelatedExpressions(*result->binder);
		return std::move(result);
	}
	D_ASSERT(func_catalog.type == CatalogType::TABLE_FUNCTION_ENTRY);
	auto &function = func_catalog.Cast<TableFunctionCatalogEntry>();

	// evaluate the input parameters to the function
	vector<LogicalType> arguments;
	vector<Value> parameters;
	named_parameter_map_t named_parameters;
	unique_ptr<BoundSubqueryRef> subquery;
	string error;
	if (!BindTableFunctionParameters(function, fexpr.children, arguments, parameters, named_parameters, subquery,
	                                 error)) {
		throw BinderException(FormatError(ref, error));
	}

	// select the function based on the input parameters
	FunctionBinder function_binder(context);
	idx_t best_function_idx = function_binder.BindFunction(function.name, function.functions, arguments, error);
	if (best_function_idx == DConstants::INVALID_INDEX) {
		throw BinderException(FormatError(ref, error));
	}
	auto table_function = function.functions.GetFunctionByOffset(best_function_idx);

	// now check the named parameters
	BindNamedParameters(table_function.named_parameters, named_parameters, error_context, table_function.name);

	// cast the parameters to the type of the function
	for (idx_t i = 0; i < arguments.size(); i++) {
		auto target_type = i < table_function.arguments.size() ? table_function.arguments[i] : table_function.varargs;

		if (target_type != LogicalType::ANY && target_type != LogicalType::TABLE &&
		    target_type != LogicalType::POINTER && target_type.id() != LogicalTypeId::LIST) {
			parameters[i] = parameters[i].CastAs(context, target_type);
		}
	}

	vector<LogicalType> input_table_types;
	vector<string> input_table_names;

	if (subquery) {
		input_table_types = subquery->subquery->types;
		input_table_names = subquery->subquery->names;
	}
	auto get = BindTableFunctionInternal(table_function, ref.alias.empty() ? fexpr.function_name : ref.alias,
	                                     std::move(parameters), std::move(named_parameters),
	                                     std::move(input_table_types), std::move(input_table_names),
	                                     ref.column_name_alias, std::move(ref.external_dependency));
	if (subquery) {
		get->children.push_back(Binder::CreatePlan(*subquery));
	}

	return make_uniq_base<BoundTableRef, BoundTableFunction>(std::move(get));
}

} // namespace duckdb




namespace duckdb {

unique_ptr<LogicalOperator> Binder::CreatePlan(BoundBaseTableRef &ref) {
	return std::move(ref.get);
}

} // namespace duckdb




namespace duckdb {

unique_ptr<LogicalOperator> Binder::CreatePlan(BoundCTERef &ref) {
	auto index = ref.bind_index;

	vector<LogicalType> types;
	types.reserve(ref.types.size());
	for (auto &type : ref.types) {
		types.push_back(type);
	}

	return make_uniq<LogicalCTERef>(index, ref.cte_index, types, ref.bound_columns);
}

} // namespace duckdb




namespace duckdb {

unique_ptr<LogicalOperator> Binder::CreatePlan(BoundEmptyTableRef &ref) {
	return make_uniq<LogicalDummyScan>(ref.bind_index);
}

} // namespace duckdb





namespace duckdb {

unique_ptr<LogicalOperator> Binder::CreatePlan(BoundExpressionListRef &ref) {
	auto root = make_uniq_base<LogicalOperator, LogicalDummyScan>(GenerateTableIndex());
	// values list, first plan any subqueries in the list
	for (auto &expr_list : ref.values) {
		for (auto &expr : expr_list) {
			PlanSubqueries(expr, root);
		}
	}
	// now create a LogicalExpressionGet from the set of expressions
	// fetch the types
	vector<LogicalType> types;
	for (auto &expr : ref.values[0]) {
		types.push_back(expr->return_type);
	}
	auto expr_get = make_uniq<LogicalExpressionGet>(ref.bind_index, types, std::move(ref.values));
	expr_get->AddChild(std::move(root));
	return std::move(expr_get);
}

} // namespace duckdb


















namespace duckdb {

//! Create a JoinCondition from a comparison
static bool CreateJoinCondition(Expression &expr, const unordered_set<idx_t> &left_bindings,
                                const unordered_set<idx_t> &right_bindings, vector<JoinCondition> &conditions) {
	// comparison
	auto &comparison = expr.Cast<BoundComparisonExpression>();
	auto left_side = JoinSide::GetJoinSide(*comparison.left, left_bindings, right_bindings);
	auto right_side = JoinSide::GetJoinSide(*comparison.right, left_bindings, right_bindings);
	if (left_side != JoinSide::BOTH && right_side != JoinSide::BOTH) {
		// join condition can be divided in a left/right side
		JoinCondition condition;
		condition.comparison = expr.type;
		auto left = std::move(comparison.left);
		auto right = std::move(comparison.right);
		if (left_side == JoinSide::RIGHT) {
			// left = right, right = left, flip the comparison symbol and reverse sides
			swap(left, right);
			condition.comparison = FlipComparisonExpression(expr.type);
		}
		condition.left = std::move(left);
		condition.right = std::move(right);
		conditions.push_back(std::move(condition));
		return true;
	}
	return false;
}

void LogicalComparisonJoin::ExtractJoinConditions(JoinType type, unique_ptr<LogicalOperator> &left_child,
                                                  unique_ptr<LogicalOperator> &right_child,
                                                  const unordered_set<idx_t> &left_bindings,
                                                  const unordered_set<idx_t> &right_bindings,
                                                  vector<unique_ptr<Expression>> &expressions,
                                                  vector<JoinCondition> &conditions,
                                                  vector<unique_ptr<Expression>> &arbitrary_expressions) {
	for (auto &expr : expressions) {
		auto total_side = JoinSide::GetJoinSide(*expr, left_bindings, right_bindings);
		if (total_side != JoinSide::BOTH) {
			// join condition does not reference both sides, add it as filter under the join
			if (type == JoinType::LEFT && total_side == JoinSide::RIGHT) {
				// filter is on RHS and the join is a LEFT OUTER join, we can push it in the right child
				if (right_child->type != LogicalOperatorType::LOGICAL_FILTER) {
					// not a filter yet, push a new empty filter
					auto filter = make_uniq<LogicalFilter>();
					filter->AddChild(std::move(right_child));
					right_child = std::move(filter);
				}
				// push the expression into the filter
				auto &filter = right_child->Cast<LogicalFilter>();
				filter.expressions.push_back(std::move(expr));
				continue;
			}
		} else if ((expr->type >= ExpressionType::COMPARE_EQUAL &&
		            expr->type <= ExpressionType::COMPARE_GREATERTHANOREQUALTO) ||
		           expr->type == ExpressionType::COMPARE_DISTINCT_FROM ||
		           expr->type == ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
			// comparison, check if we can create a comparison JoinCondition
			if (CreateJoinCondition(*expr, left_bindings, right_bindings, conditions)) {
				// successfully created the join condition
				continue;
			}
		}
		arbitrary_expressions.push_back(std::move(expr));
	}
}

void LogicalComparisonJoin::ExtractJoinConditions(JoinType type, unique_ptr<LogicalOperator> &left_child,
                                                  unique_ptr<LogicalOperator> &right_child,
                                                  vector<unique_ptr<Expression>> &expressions,
                                                  vector<JoinCondition> &conditions,
                                                  vector<unique_ptr<Expression>> &arbitrary_expressions) {
	unordered_set<idx_t> left_bindings, right_bindings;
	LogicalJoin::GetTableReferences(*left_child, left_bindings);
	LogicalJoin::GetTableReferences(*right_child, right_bindings);
	return ExtractJoinConditions(type, left_child, right_child, left_bindings, right_bindings, expressions, conditions,
	                             arbitrary_expressions);
}

void LogicalComparisonJoin::ExtractJoinConditions(JoinType type, unique_ptr<LogicalOperator> &left_child,
                                                  unique_ptr<LogicalOperator> &right_child,
                                                  unique_ptr<Expression> condition, vector<JoinCondition> &conditions,
                                                  vector<unique_ptr<Expression>> &arbitrary_expressions) {
	// split the expressions by the AND clause
	vector<unique_ptr<Expression>> expressions;
	expressions.push_back(std::move(condition));
	LogicalFilter::SplitPredicates(expressions);
	return ExtractJoinConditions(type, left_child, right_child, expressions, conditions, arbitrary_expressions);
}

unique_ptr<LogicalOperator> LogicalComparisonJoin::CreateJoin(JoinType type, JoinRefType reftype,
                                                              unique_ptr<LogicalOperator> left_child,
                                                              unique_ptr<LogicalOperator> right_child,
                                                              vector<JoinCondition> conditions,
                                                              vector<unique_ptr<Expression>> arbitrary_expressions) {
	// Validate the conditions
	bool need_to_consider_arbitrary_expressions = true;
	switch (reftype) {
	case JoinRefType::ASOF: {
		need_to_consider_arbitrary_expressions = false;
		auto asof_idx = conditions.size();
		for (size_t c = 0; c < conditions.size(); ++c) {
			auto &cond = conditions[c];
			switch (cond.comparison) {
			case ExpressionType::COMPARE_EQUAL:
			case ExpressionType::COMPARE_NOT_DISTINCT_FROM:
				break;
			case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
				if (asof_idx < conditions.size()) {
					throw BinderException("Multiple ASOF JOIN inequalities");
				}
				asof_idx = c;
				break;
			default:
				throw BinderException("Invalid ASOF JOIN comparison");
			}
		}
		if (asof_idx == conditions.size()) {
			throw BinderException("Missing ASOF JOIN inequality");
		}
		break;
	}
	default:
		break;
	}

	if (type == JoinType::INNER && reftype == JoinRefType::REGULAR) {
		// for inner joins we can push arbitrary expressions as a filter
		// here we prefer to create a comparison join if possible
		// that way we can use the much faster hash join to process the main join
		// rather than doing a nested loop join to handle arbitrary expressions

		// for left and full outer joins we HAVE to process all join conditions
		// because pushing a filter will lead to an incorrect result, as non-matching tuples cannot be filtered out
		need_to_consider_arbitrary_expressions = false;
	}
	if ((need_to_consider_arbitrary_expressions && !arbitrary_expressions.empty()) || conditions.empty()) {
		if (arbitrary_expressions.empty()) {
			// all conditions were pushed down, add TRUE predicate
			arbitrary_expressions.push_back(make_uniq<BoundConstantExpression>(Value::BOOLEAN(true)));
		}
		for (auto &condition : conditions) {
			arbitrary_expressions.push_back(JoinCondition::CreateExpression(std::move(condition)));
		}
		// if we get here we could not create any JoinConditions
		// turn this into an arbitrary expression join
		auto any_join = make_uniq<LogicalAnyJoin>(type);
		// create the condition
		any_join->children.push_back(std::move(left_child));
		any_join->children.push_back(std::move(right_child));
		// AND all the arbitrary expressions together
		// do the same with any remaining conditions
		any_join->condition = std::move(arbitrary_expressions[0]);
		for (idx_t i = 1; i < arbitrary_expressions.size(); i++) {
			any_join->condition = make_uniq<BoundConjunctionExpression>(
			    ExpressionType::CONJUNCTION_AND, std::move(any_join->condition), std::move(arbitrary_expressions[i]));
		}
		return std::move(any_join);
	} else {
		// we successfully converted expressions into JoinConditions
		// create a LogicalComparisonJoin
		unique_ptr<LogicalComparisonJoin> comp_join;
		if (reftype == JoinRefType::ASOF) {
			comp_join = make_uniq<LogicalAsOfJoin>(type);
		} else {
			comp_join = make_uniq<LogicalComparisonJoin>(type);
		}
		comp_join->conditions = std::move(conditions);
		comp_join->children.push_back(std::move(left_child));
		comp_join->children.push_back(std::move(right_child));
		if (!arbitrary_expressions.empty()) {
			// we have some arbitrary expressions as well
			// add them to a filter
			auto filter = make_uniq<LogicalFilter>();
			for (auto &expr : arbitrary_expressions) {
				filter->expressions.push_back(std::move(expr));
			}
			LogicalFilter::SplitPredicates(filter->expressions);
			filter->children.push_back(std::move(comp_join));
			return std::move(filter);
		}
		return std::move(comp_join);
	}
}

static bool HasCorrelatedColumns(Expression &expression) {
	if (expression.type == ExpressionType::BOUND_COLUMN_REF) {
		auto &colref = expression.Cast<BoundColumnRefExpression>();
		if (colref.depth > 0) {
			return true;
		}
	}
	bool has_correlated_columns = false;
	ExpressionIterator::EnumerateChildren(expression, [&](Expression &child) {
		if (HasCorrelatedColumns(child)) {
			has_correlated_columns = true;
		}
	});
	return has_correlated_columns;
}

unique_ptr<LogicalOperator> LogicalComparisonJoin::CreateJoin(JoinType type, JoinRefType reftype,
                                                              unique_ptr<LogicalOperator> left_child,
                                                              unique_ptr<LogicalOperator> right_child,
                                                              unique_ptr<Expression> condition) {
	vector<JoinCondition> conditions;
	vector<unique_ptr<Expression>> arbitrary_expressions;
	LogicalComparisonJoin::ExtractJoinConditions(type, left_child, right_child, std::move(condition), conditions,
	                                             arbitrary_expressions);
	return LogicalComparisonJoin::CreateJoin(type, reftype, std::move(left_child), std::move(right_child),
	                                         std::move(conditions), std::move(arbitrary_expressions));
}

unique_ptr<LogicalOperator> Binder::CreatePlan(BoundJoinRef &ref) {
	auto left = CreatePlan(*ref.left);
	auto right = CreatePlan(*ref.right);
	if (!ref.lateral && !ref.correlated_columns.empty()) {
		// non-lateral join with correlated columns
		// this happens if there is a join (or cross product) in a correlated subquery
		// due to the lateral binder the expression depth of all correlated columns in the "ref.correlated_columns" set
		// is 1 too high
		// we reduce expression depth of all columns in the "ref.correlated_columns" set by 1
		LateralBinder::ReduceExpressionDepth(*right, ref.correlated_columns);
	}
	if (ref.type == JoinType::RIGHT && ref.ref_type != JoinRefType::ASOF &&
	    ClientConfig::GetConfig(context).enable_optimizer) {
		// we turn any right outer joins into left outer joins for optimization purposes
		// they are the same but with sides flipped, so treating them the same simplifies life
		ref.type = JoinType::LEFT;
		std::swap(left, right);
	}
	if (ref.lateral) {
		// lateral join
		return PlanLateralJoin(std::move(left), std::move(right), ref.correlated_columns, ref.type,
		                       std::move(ref.condition));
	}
	switch (ref.ref_type) {
	case JoinRefType::CROSS:
		return LogicalCrossProduct::Create(std::move(left), std::move(right));
	case JoinRefType::POSITIONAL:
		return LogicalPositionalJoin::Create(std::move(left), std::move(right));
	default:
		break;
	}
	if (ref.type == JoinType::INNER && (ref.condition->HasSubquery() || HasCorrelatedColumns(*ref.condition)) &&
	    ref.ref_type == JoinRefType::REGULAR) {
		// inner join, generate a cross product + filter
		// this will be later turned into a proper join by the join order optimizer
		auto root = LogicalCrossProduct::Create(std::move(left), std::move(right));

		auto filter = make_uniq<LogicalFilter>(std::move(ref.condition));
		// visit the expressions in the filter
		for (auto &expression : filter->expressions) {
			PlanSubqueries(expression, root);
		}
		filter->AddChild(std::move(root));
		return std::move(filter);
	}

	// now create the join operator from the join condition
	auto result = LogicalComparisonJoin::CreateJoin(ref.type, ref.ref_type, std::move(left), std::move(right),
	                                                std::move(ref.condition));

	optional_ptr<LogicalOperator> join;
	if (result->type == LogicalOperatorType::LOGICAL_FILTER) {
		join = result->children[0].get();
	} else {
		join = result.get();
	}
	for (auto &child : join->children) {
		if (child->type == LogicalOperatorType::LOGICAL_FILTER) {
			auto &filter = child->Cast<LogicalFilter>();
			for (auto &expr : filter.expressions) {
				PlanSubqueries(expr, filter.children[0]);
			}
		}
	}

	// we visit the expressions depending on the type of join
	switch (join->type) {
	case LogicalOperatorType::LOGICAL_ASOF_JOIN:
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN: {
		// comparison join
		// in this join we visit the expressions on the LHS with the LHS as root node
		// and the expressions on the RHS with the RHS as root node
		auto &comp_join = join->Cast<LogicalComparisonJoin>();
		for (idx_t i = 0; i < comp_join.conditions.size(); i++) {
			PlanSubqueries(comp_join.conditions[i].left, comp_join.children[0]);
			PlanSubqueries(comp_join.conditions[i].right, comp_join.children[1]);
		}
		break;
	}
	case LogicalOperatorType::LOGICAL_ANY_JOIN: {
		auto &any_join = join->Cast<LogicalAnyJoin>();
		// for the any join we just visit the condition
		if (any_join.condition->HasSubquery()) {
			throw NotImplementedException("Cannot perform non-inner join on subquery!");
		}
		break;
	}
	default:
		break;
	}
	return result;
}

} // namespace duckdb



namespace duckdb {

unique_ptr<LogicalOperator> Binder::CreatePlan(BoundPivotRef &ref) {
	auto subquery = ref.child_binder->CreatePlan(*ref.child);

	auto result = make_uniq<LogicalPivot>(ref.bind_index, std::move(subquery), std::move(ref.bound_pivot));
	return std::move(result);
}

} // namespace duckdb



namespace duckdb {

unique_ptr<LogicalOperator> Binder::CreatePlan(BoundSubqueryRef &ref) {
	// generate the logical plan for the subquery
	// this happens separately from the current LogicalPlan generation
	ref.binder->plan_subquery = plan_subquery;
	auto subquery = ref.binder->CreatePlan(*ref.subquery);
	if (ref.binder->has_unplanned_subqueries) {
		has_unplanned_subqueries = true;
	}
	return subquery;
}

} // namespace duckdb



namespace duckdb {

unique_ptr<LogicalOperator> Binder::CreatePlan(BoundTableFunction &ref) {
	return std::move(ref.get);
}

} // namespace duckdb



















#include <algorithm>

namespace duckdb {

shared_ptr<Binder> Binder::CreateBinder(ClientContext &context, optional_ptr<Binder> parent, bool inherit_ctes) {
	return make_shared<Binder>(true, context, parent ? parent->shared_from_this() : nullptr, inherit_ctes);
}

Binder::Binder(bool, ClientContext &context, shared_ptr<Binder> parent_p, bool inherit_ctes_p)
    : context(context), parent(std::move(parent_p)), bound_tables(0), inherit_ctes(inherit_ctes_p) {
	if (parent) {

		// We have to inherit macro and lambda parameter bindings and from the parent binder, if there is a parent.
		macro_binding = parent->macro_binding;
		lambda_bindings = parent->lambda_bindings;

		if (inherit_ctes) {
			// We have to inherit CTE bindings from the parent bind_context, if there is a parent.
			bind_context.SetCTEBindings(parent->bind_context.GetCTEBindings());
			bind_context.cte_references = parent->bind_context.cte_references;
			parameters = parent->parameters;
		}
	}
}

BoundStatement Binder::Bind(SQLStatement &statement) {
	root_statement = &statement;
	switch (statement.type) {
	case StatementType::SELECT_STATEMENT:
		return Bind(statement.Cast<SelectStatement>());
	case StatementType::INSERT_STATEMENT:
		return Bind(statement.Cast<InsertStatement>());
	case StatementType::COPY_STATEMENT:
		return Bind(statement.Cast<CopyStatement>());
	case StatementType::DELETE_STATEMENT:
		return Bind(statement.Cast<DeleteStatement>());
	case StatementType::UPDATE_STATEMENT:
		return Bind(statement.Cast<UpdateStatement>());
	case StatementType::RELATION_STATEMENT:
		return Bind(statement.Cast<RelationStatement>());
	case StatementType::CREATE_STATEMENT:
		return Bind(statement.Cast<CreateStatement>());
	case StatementType::DROP_STATEMENT:
		return Bind(statement.Cast<DropStatement>());
	case StatementType::ALTER_STATEMENT:
		return Bind(statement.Cast<AlterStatement>());
	case StatementType::TRANSACTION_STATEMENT:
		return Bind(statement.Cast<TransactionStatement>());
	case StatementType::PRAGMA_STATEMENT:
		return Bind(statement.Cast<PragmaStatement>());
	case StatementType::EXPLAIN_STATEMENT:
		return Bind(statement.Cast<ExplainStatement>());
	case StatementType::VACUUM_STATEMENT:
		return Bind(statement.Cast<VacuumStatement>());
	case StatementType::SHOW_STATEMENT:
		return Bind(statement.Cast<ShowStatement>());
	case StatementType::CALL_STATEMENT:
		return Bind(statement.Cast<CallStatement>());
	case StatementType::EXPORT_STATEMENT:
		return Bind(statement.Cast<ExportStatement>());
	case StatementType::SET_STATEMENT:
		return Bind(statement.Cast<SetStatement>());
	case StatementType::LOAD_STATEMENT:
		return Bind(statement.Cast<LoadStatement>());
	case StatementType::EXTENSION_STATEMENT:
		return Bind(statement.Cast<ExtensionStatement>());
	case StatementType::PREPARE_STATEMENT:
		return Bind(statement.Cast<PrepareStatement>());
	case StatementType::EXECUTE_STATEMENT:
		return Bind(statement.Cast<ExecuteStatement>());
	case StatementType::LOGICAL_PLAN_STATEMENT:
		return Bind(statement.Cast<LogicalPlanStatement>());
	case StatementType::ATTACH_STATEMENT:
		return Bind(statement.Cast<AttachStatement>());
	case StatementType::DETACH_STATEMENT:
		return Bind(statement.Cast<DetachStatement>());
	default: // LCOV_EXCL_START
		throw NotImplementedException("Unimplemented statement type \"%s\" for Bind",
		                              StatementTypeToString(statement.type));
	} // LCOV_EXCL_STOP
}

void Binder::AddCTEMap(CommonTableExpressionMap &cte_map) {
	for (auto &cte_it : cte_map.map) {
		AddCTE(cte_it.first, *cte_it.second);
	}
}

unique_ptr<BoundQueryNode> Binder::BindNode(QueryNode &node) {
	// first we visit the set of CTEs and add them to the bind context
	AddCTEMap(node.cte_map);
	// now we bind the node
	unique_ptr<BoundQueryNode> result;
	switch (node.type) {
	case QueryNodeType::SELECT_NODE:
		result = BindNode(node.Cast<SelectNode>());
		break;
	case QueryNodeType::RECURSIVE_CTE_NODE:
		result = BindNode(node.Cast<RecursiveCTENode>());
		break;
	default:
		D_ASSERT(node.type == QueryNodeType::SET_OPERATION_NODE);
		result = BindNode(node.Cast<SetOperationNode>());
		break;
	}
	return result;
}

BoundStatement Binder::Bind(QueryNode &node) {
	auto bound_node = BindNode(node);

	BoundStatement result;
	result.names = bound_node->names;
	result.types = bound_node->types;

	// and plan it
	result.plan = CreatePlan(*bound_node);
	return result;
}

unique_ptr<LogicalOperator> Binder::CreatePlan(BoundQueryNode &node) {
	switch (node.type) {
	case QueryNodeType::SELECT_NODE:
		return CreatePlan(node.Cast<BoundSelectNode>());
	case QueryNodeType::SET_OPERATION_NODE:
		return CreatePlan(node.Cast<BoundSetOperationNode>());
	case QueryNodeType::RECURSIVE_CTE_NODE:
		return CreatePlan(node.Cast<BoundRecursiveCTENode>());
	default:
		throw InternalException("Unsupported bound query node type");
	}
}

unique_ptr<BoundTableRef> Binder::Bind(TableRef &ref) {
	unique_ptr<BoundTableRef> result;
	switch (ref.type) {
	case TableReferenceType::BASE_TABLE:
		result = Bind(ref.Cast<BaseTableRef>());
		break;
	case TableReferenceType::JOIN:
		result = Bind(ref.Cast<JoinRef>());
		break;
	case TableReferenceType::SUBQUERY:
		result = Bind(ref.Cast<SubqueryRef>());
		break;
	case TableReferenceType::EMPTY:
		result = Bind(ref.Cast<EmptyTableRef>());
		break;
	case TableReferenceType::TABLE_FUNCTION:
		result = Bind(ref.Cast<TableFunctionRef>());
		break;
	case TableReferenceType::EXPRESSION_LIST:
		result = Bind(ref.Cast<ExpressionListRef>());
		break;
	case TableReferenceType::PIVOT:
		result = Bind(ref.Cast<PivotRef>());
		break;
	case TableReferenceType::CTE:
	case TableReferenceType::INVALID:
	default:
		throw InternalException("Unknown table ref type");
	}
	result->sample = std::move(ref.sample);
	return result;
}

unique_ptr<LogicalOperator> Binder::CreatePlan(BoundTableRef &ref) {
	unique_ptr<LogicalOperator> root;
	switch (ref.type) {
	case TableReferenceType::BASE_TABLE:
		root = CreatePlan(ref.Cast<BoundBaseTableRef>());
		break;
	case TableReferenceType::SUBQUERY:
		root = CreatePlan(ref.Cast<BoundSubqueryRef>());
		break;
	case TableReferenceType::JOIN:
		root = CreatePlan(ref.Cast<BoundJoinRef>());
		break;
	case TableReferenceType::TABLE_FUNCTION:
		root = CreatePlan(ref.Cast<BoundTableFunction>());
		break;
	case TableReferenceType::EMPTY:
		root = CreatePlan(ref.Cast<BoundEmptyTableRef>());
		break;
	case TableReferenceType::EXPRESSION_LIST:
		root = CreatePlan(ref.Cast<BoundExpressionListRef>());
		break;
	case TableReferenceType::CTE:
		root = CreatePlan(ref.Cast<BoundCTERef>());
		break;
	case TableReferenceType::PIVOT:
		root = CreatePlan((BoundPivotRef &)ref);
		break;
	case TableReferenceType::INVALID:
	default:
		throw InternalException("Unsupported bound table ref type");
	}
	// plan the sample clause
	if (ref.sample) {
		root = make_uniq<LogicalSample>(std::move(ref.sample), std::move(root));
	}
	return root;
}

void Binder::AddCTE(const string &name, CommonTableExpressionInfo &info) {
	D_ASSERT(!name.empty());
	auto entry = CTE_bindings.find(name);
	if (entry != CTE_bindings.end()) {
		throw InternalException("Duplicate CTE \"%s\" in query!", name);
	}
	CTE_bindings.insert(make_pair(name, reference<CommonTableExpressionInfo>(info)));
}

optional_ptr<CommonTableExpressionInfo> Binder::FindCTE(const string &name, bool skip) {
	auto entry = CTE_bindings.find(name);
	if (entry != CTE_bindings.end()) {
		if (!skip || entry->second.get().query->node->type == QueryNodeType::RECURSIVE_CTE_NODE) {
			return &entry->second.get();
		}
	}
	if (parent && inherit_ctes) {
		return parent->FindCTE(name, name == alias);
	}
	return nullptr;
}

bool Binder::CTEIsAlreadyBound(CommonTableExpressionInfo &cte) {
	if (bound_ctes.find(cte) != bound_ctes.end()) {
		return true;
	}
	if (parent && inherit_ctes) {
		return parent->CTEIsAlreadyBound(cte);
	}
	return false;
}

void Binder::AddBoundView(ViewCatalogEntry &view) {
	// check if the view is already bound
	auto current = this;
	while (current) {
		if (current->bound_views.find(view) != current->bound_views.end()) {
			throw BinderException("infinite recursion detected: attempting to recursively bind view \"%s\"", view.name);
		}
		current = current->parent.get();
	}
	bound_views.insert(view);
}

idx_t Binder::GenerateTableIndex() {
	D_ASSERT(parent.get() != this);
	if (parent) {
		return parent->GenerateTableIndex();
	}
	return bound_tables++;
}

void Binder::PushExpressionBinder(ExpressionBinder &binder) {
	GetActiveBinders().push_back(binder);
}

void Binder::PopExpressionBinder() {
	D_ASSERT(HasActiveBinder());
	GetActiveBinders().pop_back();
}

void Binder::SetActiveBinder(ExpressionBinder &binder) {
	D_ASSERT(HasActiveBinder());
	GetActiveBinders().back() = binder;
}

ExpressionBinder &Binder::GetActiveBinder() {
	return GetActiveBinders().back();
}

bool Binder::HasActiveBinder() {
	return !GetActiveBinders().empty();
}

vector<reference<ExpressionBinder>> &Binder::GetActiveBinders() {
	if (parent) {
		return parent->GetActiveBinders();
	}
	return active_binders;
}

void Binder::AddUsingBindingSet(unique_ptr<UsingColumnSet> set) {
	if (parent) {
		parent->AddUsingBindingSet(std::move(set));
		return;
	}
	bind_context.AddUsingBindingSet(std::move(set));
}

void Binder::MoveCorrelatedExpressions(Binder &other) {
	MergeCorrelatedColumns(other.correlated_columns);
	other.correlated_columns.clear();
}

void Binder::MergeCorrelatedColumns(vector<CorrelatedColumnInfo> &other) {
	for (idx_t i = 0; i < other.size(); i++) {
		AddCorrelatedColumn(other[i]);
	}
}

void Binder::AddCorrelatedColumn(const CorrelatedColumnInfo &info) {
	// we only add correlated columns to the list if they are not already there
	if (std::find(correlated_columns.begin(), correlated_columns.end(), info) == correlated_columns.end()) {
		correlated_columns.push_back(info);
	}
}

bool Binder::HasMatchingBinding(const string &table_name, const string &column_name, string &error_message) {
	string empty_schema;
	return HasMatchingBinding(empty_schema, table_name, column_name, error_message);
}

bool Binder::HasMatchingBinding(const string &schema_name, const string &table_name, const string &column_name,
                                string &error_message) {
	string empty_catalog;
	return HasMatchingBinding(empty_catalog, schema_name, table_name, column_name, error_message);
}

bool Binder::HasMatchingBinding(const string &catalog_name, const string &schema_name, const string &table_name,
                                const string &column_name, string &error_message) {
	optional_ptr<Binding> binding;
	D_ASSERT(!lambda_bindings);
	if (macro_binding && table_name == macro_binding->alias) {
		binding = optional_ptr<Binding>(macro_binding.get());
	} else {
		binding = bind_context.GetBinding(table_name, error_message);
	}

	if (!binding) {
		return false;
	}
	if (!catalog_name.empty() || !schema_name.empty()) {
		auto catalog_entry = binding->GetStandardEntry();
		if (!catalog_entry) {
			return false;
		}
		if (!catalog_name.empty() && catalog_entry->catalog.GetName() != catalog_name) {
			return false;
		}
		if (!schema_name.empty() && catalog_entry->schema.name != schema_name) {
			return false;
		}
		if (catalog_entry->name != table_name) {
			return false;
		}
	}
	bool binding_found;
	binding_found = binding->HasMatchingBinding(column_name);
	if (!binding_found) {
		error_message = binding->ColumnNotFoundError(column_name);
	}
	return binding_found;
}

void Binder::SetBindingMode(BindingMode mode) {
	if (parent) {
		parent->SetBindingMode(mode);
	}
	this->mode = mode;
}

BindingMode Binder::GetBindingMode() {
	if (parent) {
		return parent->GetBindingMode();
	}
	return mode;
}

void Binder::SetCanContainNulls(bool can_contain_nulls_p) {
	can_contain_nulls = can_contain_nulls_p;
}

void Binder::AddTableName(string table_name) {
	if (parent) {
		parent->AddTableName(std::move(table_name));
		return;
	}
	table_names.insert(std::move(table_name));
}

const unordered_set<string> &Binder::GetTableNames() {
	if (parent) {
		return parent->GetTableNames();
	}
	return table_names;
}

string Binder::FormatError(ParsedExpression &expr_context, const string &message) {
	return FormatError(expr_context.query_location, message);
}

string Binder::FormatError(TableRef &ref_context, const string &message) {
	return FormatError(ref_context.query_location, message);
}

string Binder::FormatErrorRecursive(idx_t query_location, const string &message, vector<ExceptionFormatValue> &values) {
	QueryErrorContext context(root_statement, query_location);
	return context.FormatErrorRecursive(message, values);
}

// FIXME: this is extremely naive
void VerifyNotExcluded(ParsedExpression &expr) {
	if (expr.type == ExpressionType::COLUMN_REF) {
		auto &column_ref = expr.Cast<ColumnRefExpression>();
		if (!column_ref.IsQualified()) {
			return;
		}
		auto &table_name = column_ref.GetTableName();
		if (table_name == "excluded") {
			throw NotImplementedException("'excluded' qualified columns are not supported in the RETURNING clause yet");
		}
		return;
	}
	ParsedExpressionIterator::EnumerateChildren(
	    expr, [&](const ParsedExpression &child) { VerifyNotExcluded((ParsedExpression &)child); });
}

BoundStatement Binder::BindReturning(vector<unique_ptr<ParsedExpression>> returning_list, TableCatalogEntry &table,
                                     const string &alias, idx_t update_table_index,
                                     unique_ptr<LogicalOperator> child_operator, BoundStatement result) {

	vector<LogicalType> types;
	vector<std::string> names;

	auto binder = Binder::CreateBinder(context);

	vector<column_t> bound_columns;
	idx_t column_count = 0;
	for (auto &col : table.GetColumns().Logical()) {
		names.push_back(col.Name());
		types.push_back(col.Type());
		if (!col.Generated()) {
			bound_columns.push_back(column_count);
		}
		column_count++;
	}

	binder->bind_context.AddBaseTable(update_table_index, alias.empty() ? table.name : alias, names, types,
	                                  bound_columns, &table, false);
	ReturningBinder returning_binder(*binder, context);

	vector<unique_ptr<Expression>> projection_expressions;
	LogicalType result_type;
	vector<unique_ptr<ParsedExpression>> new_returning_list;
	binder->ExpandStarExpressions(returning_list, new_returning_list);
	for (auto &returning_expr : new_returning_list) {
		VerifyNotExcluded(*returning_expr);
		auto expr = returning_binder.Bind(returning_expr, &result_type);
		result.names.push_back(expr->GetName());
		result.types.push_back(result_type);
		projection_expressions.push_back(std::move(expr));
	}

	auto projection = make_uniq<LogicalProjection>(GenerateTableIndex(), std::move(projection_expressions));
	projection->AddChild(std::move(child_operator));
	D_ASSERT(result.types.size() == result.names.size());
	result.plan = std::move(projection);
	properties.allow_stream_result = true;
	properties.return_type = StatementReturnType::QUERY_RESULT;
	return result;
}

} // namespace duckdb



namespace duckdb {

BoundResultModifier::BoundResultModifier(ResultModifierType type) : type(type) {
}

BoundResultModifier::~BoundResultModifier() {
}

BoundOrderByNode::BoundOrderByNode(OrderType type, OrderByNullType null_order, unique_ptr<Expression> expression)
    : type(type), null_order(null_order), expression(std::move(expression)) {
}
BoundOrderByNode::BoundOrderByNode(OrderType type, OrderByNullType null_order, unique_ptr<Expression> expression,
                                   unique_ptr<BaseStatistics> stats)
    : type(type), null_order(null_order), expression(std::move(expression)), stats(std::move(stats)) {
}

BoundOrderByNode BoundOrderByNode::Copy() const {
	if (stats) {
		return BoundOrderByNode(type, null_order, expression->Copy(), stats->ToUnique());
	} else {
		return BoundOrderByNode(type, null_order, expression->Copy());
	}
}

bool BoundOrderByNode::Equals(const BoundOrderByNode &other) const {
	if (type != other.type || null_order != other.null_order) {
		return false;
	}
	if (!expression->Equals(other.expression.get())) {
		return false;
	}

	return true;
}

string BoundOrderByNode::ToString() const {
	auto str = expression->ToString();
	switch (type) {
	case OrderType::ASCENDING:
		str += " ASC";
		break;
	case OrderType::DESCENDING:
		str += " DESC";
		break;
	default:
		break;
	}

	switch (null_order) {
	case OrderByNullType::NULLS_FIRST:
		str += " NULLS FIRST";
		break;
	case OrderByNullType::NULLS_LAST:
		str += " NULLS LAST";
		break;
	default:
		break;
	}
	return str;
}

void BoundOrderByNode::Serialize(Serializer &serializer) const {
	FieldWriter writer(serializer);
	writer.WriteField(type);
	writer.WriteField(null_order);
	writer.WriteSerializable(*expression);
	// TODO statistics
	writer.Finalize();
}

BoundOrderByNode BoundOrderByNode::Deserialize(Deserializer &source, PlanDeserializationState &state) {
	FieldReader reader(source);
	auto type = reader.ReadRequired<OrderType>();
	auto null_order = reader.ReadRequired<OrderByNullType>();
	auto expression = reader.ReadRequiredSerializable<Expression>(state);
	reader.Finalize();
	return BoundOrderByNode(type, null_order, std::move(expression));
}

unique_ptr<BoundOrderModifier> BoundOrderModifier::Copy() const {
	auto result = make_uniq<BoundOrderModifier>();
	for (auto &order : orders) {
		result->orders.push_back(order.Copy());
	}
	return result;
}

bool BoundOrderModifier::Equals(const BoundOrderModifier *left, const BoundOrderModifier *right) {
	if (left == right) {
		return true;
	}
	if (!left || !right) {
		return false;
	}
	if (left->orders.size() != right->orders.size()) {
		return false;
	}
	for (idx_t i = 0; i < left->orders.size(); i++) {
		if (!left->orders[i].Equals(right->orders[i])) {
			return false;
		}
	}
	return true;
}

BoundLimitModifier::BoundLimitModifier() : BoundResultModifier(ResultModifierType::LIMIT_MODIFIER) {
}

BoundOrderModifier::BoundOrderModifier() : BoundResultModifier(ResultModifierType::ORDER_MODIFIER) {
}

BoundDistinctModifier::BoundDistinctModifier() : BoundResultModifier(ResultModifierType::DISTINCT_MODIFIER) {
}

BoundLimitPercentModifier::BoundLimitPercentModifier()
    : BoundResultModifier(ResultModifierType::LIMIT_PERCENT_MODIFIER) {
}

} // namespace duckdb









namespace duckdb {

BoundAggregateExpression::BoundAggregateExpression(AggregateFunction function, vector<unique_ptr<Expression>> children,
                                                   unique_ptr<Expression> filter, unique_ptr<FunctionData> bind_info,
                                                   AggregateType aggr_type)
    : Expression(ExpressionType::BOUND_AGGREGATE, ExpressionClass::BOUND_AGGREGATE, function.return_type),
      function(std::move(function)), children(std::move(children)), bind_info(std::move(bind_info)),
      aggr_type(aggr_type), filter(std::move(filter)) {
	D_ASSERT(!function.name.empty());
}

string BoundAggregateExpression::ToString() const {
	return FunctionExpression::ToString<BoundAggregateExpression, Expression, BoundOrderModifier>(
	    *this, string(), function.name, false, IsDistinct(), filter.get(), order_bys.get());
}

hash_t BoundAggregateExpression::Hash() const {
	hash_t result = Expression::Hash();
	result = CombineHash(result, function.Hash());
	result = CombineHash(result, duckdb::Hash(IsDistinct()));
	return result;
}

bool BoundAggregateExpression::Equals(const BaseExpression *other_p) const {
	if (!Expression::Equals(other_p)) {
		return false;
	}
	auto &other = other_p->Cast<BoundAggregateExpression>();
	if (other.aggr_type != aggr_type) {
		return false;
	}
	if (other.function != function) {
		return false;
	}
	if (children.size() != other.children.size()) {
		return false;
	}
	if (!Expression::Equals(other.filter.get(), filter.get())) {
		return false;
	}
	for (idx_t i = 0; i < children.size(); i++) {
		if (!Expression::Equals(children[i].get(), other.children[i].get())) {
			return false;
		}
	}
	if (!FunctionData::Equals(bind_info.get(), other.bind_info.get())) {
		return false;
	}
	if (!BoundOrderModifier::Equals(order_bys.get(), other.order_bys.get())) {
		return false;
	}
	return true;
}

bool BoundAggregateExpression::PropagatesNullValues() const {
	return function.null_handling == FunctionNullHandling::SPECIAL_HANDLING ? false
	                                                                        : Expression::PropagatesNullValues();
}

unique_ptr<Expression> BoundAggregateExpression::Copy() {
	vector<unique_ptr<Expression>> new_children;
	new_children.reserve(children.size());
	for (auto &child : children) {
		new_children.push_back(child->Copy());
	}
	auto new_bind_info = bind_info ? bind_info->Copy() : nullptr;
	auto new_filter = filter ? filter->Copy() : nullptr;
	auto copy = make_uniq<BoundAggregateExpression>(function, std::move(new_children), std::move(new_filter),
	                                                std::move(new_bind_info), aggr_type);
	copy->CopyProperties(*this);
	copy->order_bys = order_bys ? order_bys->Copy() : nullptr;
	return std::move(copy);
}

void BoundAggregateExpression::Serialize(FieldWriter &writer) const {
	writer.WriteField(IsDistinct());
	writer.WriteOptional(filter);
	if (order_bys) {
		throw NotImplementedException("Serialization of ORDER BY aggregate not yet supported");
	}
	FunctionSerializer::Serialize<AggregateFunction>(writer, function, return_type, children, bind_info.get());
}

unique_ptr<Expression> BoundAggregateExpression::Deserialize(ExpressionDeserializationState &state,
                                                             FieldReader &reader) {
	auto distinct = reader.ReadRequired<bool>();
	auto filter = reader.ReadOptional<Expression>(nullptr, state.gstate);
	vector<unique_ptr<Expression>> children;
	unique_ptr<FunctionData> bind_info;
	auto function = FunctionSerializer::Deserialize<AggregateFunction, AggregateFunctionCatalogEntry>(
	    reader, state, CatalogType::AGGREGATE_FUNCTION_ENTRY, children, bind_info);

	return make_uniq<BoundAggregateExpression>(function, std::move(children), std::move(filter), std::move(bind_info),
	                                           distinct ? AggregateType::DISTINCT : AggregateType::NON_DISTINCT);
}

} // namespace duckdb




namespace duckdb {

BoundBetweenExpression::BoundBetweenExpression(unique_ptr<Expression> input, unique_ptr<Expression> lower,
                                               unique_ptr<Expression> upper, bool lower_inclusive, bool upper_inclusive)
    : Expression(ExpressionType::COMPARE_BETWEEN, ExpressionClass::BOUND_BETWEEN, LogicalType::BOOLEAN),
      input(std::move(input)), lower(std::move(lower)), upper(std::move(upper)), lower_inclusive(lower_inclusive),
      upper_inclusive(upper_inclusive) {
}

string BoundBetweenExpression::ToString() const {
	return BetweenExpression::ToString<BoundBetweenExpression, Expression>(*this);
}

bool BoundBetweenExpression::Equals(const BaseExpression *other_p) const {
	if (!Expression::Equals(other_p)) {
		return false;
	}
	auto &other = other_p->Cast<BoundBetweenExpression>();
	if (!Expression::Equals(input.get(), other.input.get())) {
		return false;
	}
	if (!Expression::Equals(lower.get(), other.lower.get())) {
		return false;
	}
	if (!Expression::Equals(upper.get(), other.upper.get())) {
		return false;
	}
	return lower_inclusive == other.lower_inclusive && upper_inclusive == other.upper_inclusive;
}

unique_ptr<Expression> BoundBetweenExpression::Copy() {
	auto copy = make_uniq<BoundBetweenExpression>(input->Copy(), lower->Copy(), upper->Copy(), lower_inclusive,
	                                              upper_inclusive);
	copy->CopyProperties(*this);
	return std::move(copy);
}

void BoundBetweenExpression::Serialize(FieldWriter &writer) const {
	writer.WriteOptional(input);
	writer.WriteOptional(lower);
	writer.WriteOptional(upper);
	writer.WriteField(lower_inclusive);
	writer.WriteField(upper_inclusive);
}

unique_ptr<Expression> BoundBetweenExpression::Deserialize(ExpressionDeserializationState &state, FieldReader &reader) {
	auto input = reader.ReadOptional<Expression>(nullptr, state.gstate);
	auto lower = reader.ReadOptional<Expression>(nullptr, state.gstate);
	auto upper = reader.ReadOptional<Expression>(nullptr, state.gstate);
	auto lower_inclusive = reader.ReadRequired<bool>();
	auto upper_inclusive = reader.ReadRequired<bool>();
	return make_uniq<BoundBetweenExpression>(std::move(input), std::move(lower), std::move(upper), lower_inclusive,
	                                         upper_inclusive);
}

} // namespace duckdb




namespace duckdb {

BoundCaseExpression::BoundCaseExpression(LogicalType type)
    : Expression(ExpressionType::CASE_EXPR, ExpressionClass::BOUND_CASE, std::move(type)) {
}

BoundCaseExpression::BoundCaseExpression(unique_ptr<Expression> when_expr, unique_ptr<Expression> then_expr,
                                         unique_ptr<Expression> else_expr_p)
    : Expression(ExpressionType::CASE_EXPR, ExpressionClass::BOUND_CASE, then_expr->return_type),
      else_expr(std::move(else_expr_p)) {
	BoundCaseCheck check;
	check.when_expr = std::move(when_expr);
	check.then_expr = std::move(then_expr);
	case_checks.push_back(std::move(check));
}

string BoundCaseExpression::ToString() const {
	return CaseExpression::ToString<BoundCaseExpression, Expression>(*this);
}

bool BoundCaseExpression::Equals(const BaseExpression *other_p) const {
	if (!Expression::Equals(other_p)) {
		return false;
	}
	auto &other = other_p->Cast<BoundCaseExpression>();
	if (case_checks.size() != other.case_checks.size()) {
		return false;
	}
	for (idx_t i = 0; i < case_checks.size(); i++) {
		if (!Expression::Equals(case_checks[i].when_expr.get(), other.case_checks[i].when_expr.get())) {
			return false;
		}
		if (!Expression::Equals(case_checks[i].then_expr.get(), other.case_checks[i].then_expr.get())) {
			return false;
		}
	}
	if (!Expression::Equals(else_expr.get(), other.else_expr.get())) {
		return false;
	}
	return true;
}

unique_ptr<Expression> BoundCaseExpression::Copy() {
	auto new_case = make_uniq<BoundCaseExpression>(return_type);
	for (auto &check : case_checks) {
		BoundCaseCheck new_check;
		new_check.when_expr = check.when_expr->Copy();
		new_check.then_expr = check.then_expr->Copy();
		new_case->case_checks.push_back(std::move(new_check));
	}
	new_case->else_expr = else_expr->Copy();

	new_case->CopyProperties(*this);
	return std::move(new_case);
}

void BoundCaseCheck::Serialize(Serializer &serializer) const {
	FieldWriter writer(serializer);
	writer.WriteSerializable(*when_expr);
	writer.WriteSerializable(*then_expr);
	writer.Finalize();
}

BoundCaseCheck BoundCaseCheck::Deserialize(Deserializer &source, PlanDeserializationState &state) {
	FieldReader reader(source);
	auto when_expr = reader.ReadRequiredSerializable<Expression>(state);
	auto then_expr = reader.ReadRequiredSerializable<Expression>(state);
	reader.Finalize();
	BoundCaseCheck result;
	result.when_expr = std::move(when_expr);
	result.then_expr = std::move(then_expr);
	return result;
}

void BoundCaseExpression::Serialize(FieldWriter &writer) const {
	writer.WriteSerializable(return_type);
	writer.WriteRegularSerializableList(case_checks);
	writer.WriteSerializable(*else_expr);
}

unique_ptr<Expression> BoundCaseExpression::Deserialize(ExpressionDeserializationState &state, FieldReader &reader) {
	auto return_type = reader.ReadRequiredSerializable<LogicalType, LogicalType>();
	auto case_checks = reader.ReadRequiredSerializableList<BoundCaseCheck, BoundCaseCheck>(state.gstate);
	auto else_expr = reader.ReadRequiredSerializable<Expression>(state.gstate);

	auto result = make_uniq<BoundCaseExpression>(return_type);
	result->else_expr = std::move(else_expr);
	result->case_checks = std::move(case_checks);
	return std::move(result);
}

} // namespace duckdb








namespace duckdb {

BoundCastExpression::BoundCastExpression(unique_ptr<Expression> child_p, LogicalType target_type_p,
                                         BoundCastInfo bound_cast_p, bool try_cast_p)
    : Expression(ExpressionType::OPERATOR_CAST, ExpressionClass::BOUND_CAST, std::move(target_type_p)),
      child(std::move(child_p)), try_cast(try_cast_p), bound_cast(std::move(bound_cast_p)) {
}

unique_ptr<Expression> AddCastExpressionInternal(unique_ptr<Expression> expr, const LogicalType &target_type,
                                                 BoundCastInfo bound_cast, bool try_cast) {
	if (expr->return_type == target_type) {
		return expr;
	}
	auto &expr_type = expr->return_type;
	if (target_type.id() == LogicalTypeId::LIST && expr_type.id() == LogicalTypeId::LIST) {
		auto &target_list = ListType::GetChildType(target_type);
		auto &expr_list = ListType::GetChildType(expr_type);
		if (target_list.id() == LogicalTypeId::ANY || expr_list == target_list) {
			return expr;
		}
	}
	return make_uniq<BoundCastExpression>(std::move(expr), target_type, std::move(bound_cast), try_cast);
}

static BoundCastInfo BindCastFunction(ClientContext &context, const LogicalType &source, const LogicalType &target) {
	auto &cast_functions = DBConfig::GetConfig(context).GetCastFunctions();
	GetCastFunctionInput input(context);
	return cast_functions.GetCastFunction(source, target, input);
}

unique_ptr<Expression> AddCastToTypeInternal(unique_ptr<Expression> expr, const LogicalType &target_type,
                                             CastFunctionSet &cast_functions, GetCastFunctionInput &get_input,
                                             bool try_cast) {
	D_ASSERT(expr);
	if (expr->expression_class == ExpressionClass::BOUND_PARAMETER) {
		auto &parameter = expr->Cast<BoundParameterExpression>();
		if (!target_type.IsValid()) {
			// invalidate the parameter
			parameter.parameter_data->return_type = LogicalType::INVALID;
			parameter.return_type = target_type;
			return expr;
		}
		if (parameter.parameter_data->return_type.id() == LogicalTypeId::INVALID) {
			// we don't know the type of this parameter
			parameter.return_type = target_type;
			return expr;
		}
		if (parameter.parameter_data->return_type.id() == LogicalTypeId::UNKNOWN) {
			// prepared statement parameter cast - but there is no type, convert the type
			parameter.parameter_data->return_type = target_type;
			parameter.return_type = target_type;
			return expr;
		}
		// prepared statement parameter already has a type
		if (parameter.parameter_data->return_type == target_type) {
			// this type! we are done
			parameter.return_type = parameter.parameter_data->return_type;
			return expr;
		}
		// invalidate the type
		parameter.parameter_data->return_type = LogicalType::INVALID;
		parameter.return_type = target_type;
		return expr;
	} else if (expr->expression_class == ExpressionClass::BOUND_DEFAULT) {
		D_ASSERT(target_type.IsValid());
		auto &def = expr->Cast<BoundDefaultExpression>();
		def.return_type = target_type;
	}
	if (!target_type.IsValid()) {
		return expr;
	}

	auto cast_function = cast_functions.GetCastFunction(expr->return_type, target_type, get_input);
	return AddCastExpressionInternal(std::move(expr), target_type, std::move(cast_function), try_cast);
}

unique_ptr<Expression> BoundCastExpression::AddDefaultCastToType(unique_ptr<Expression> expr,
                                                                 const LogicalType &target_type, bool try_cast) {
	CastFunctionSet default_set;
	GetCastFunctionInput get_input;
	return AddCastToTypeInternal(std::move(expr), target_type, default_set, get_input, try_cast);
}

unique_ptr<Expression> BoundCastExpression::AddCastToType(ClientContext &context, unique_ptr<Expression> expr,
                                                          const LogicalType &target_type, bool try_cast) {
	auto &cast_functions = DBConfig::GetConfig(context).GetCastFunctions();
	GetCastFunctionInput get_input(context);
	return AddCastToTypeInternal(std::move(expr), target_type, cast_functions, get_input, try_cast);
}

bool BoundCastExpression::CastIsInvertible(const LogicalType &source_type, const LogicalType &target_type) {
	D_ASSERT(source_type.IsValid() && target_type.IsValid());
	if (source_type.id() == LogicalTypeId::BOOLEAN || target_type.id() == LogicalTypeId::BOOLEAN) {
		return false;
	}
	if (source_type.id() == LogicalTypeId::FLOAT || target_type.id() == LogicalTypeId::FLOAT) {
		return false;
	}
	if (source_type.id() == LogicalTypeId::DOUBLE || target_type.id() == LogicalTypeId::DOUBLE) {
		return false;
	}
	if (source_type.id() == LogicalTypeId::DECIMAL || target_type.id() == LogicalTypeId::DECIMAL) {
		uint8_t source_width, target_width;
		uint8_t source_scale, target_scale;
		// cast to or from decimal
		// cast is only invertible if the cast is strictly widening
		if (!source_type.GetDecimalProperties(source_width, source_scale)) {
			return false;
		}
		if (!target_type.GetDecimalProperties(target_width, target_scale)) {
			return false;
		}
		if (target_scale < source_scale) {
			return false;
		}
		return true;
	}
	if (source_type.id() == LogicalTypeId::TIMESTAMP || source_type.id() == LogicalTypeId::TIMESTAMP_TZ) {
		switch (target_type.id()) {
		case LogicalTypeId::DATE:
		case LogicalTypeId::TIME:
		case LogicalTypeId::TIME_TZ:
			return false;
		default:
			break;
		}
	}
	if (source_type.id() == LogicalTypeId::VARCHAR) {
		switch (target_type.id()) {
		case LogicalTypeId::TIME:
		case LogicalTypeId::TIMESTAMP:
		case LogicalTypeId::TIMESTAMP_NS:
		case LogicalTypeId::TIMESTAMP_MS:
		case LogicalTypeId::TIMESTAMP_SEC:
		case LogicalTypeId::TIME_TZ:
		case LogicalTypeId::TIMESTAMP_TZ:
			return true;
		default:
			return false;
		}
	}
	if (target_type.id() == LogicalTypeId::VARCHAR) {
		switch (source_type.id()) {
		case LogicalTypeId::DATE:
		case LogicalTypeId::TIME:
		case LogicalTypeId::TIMESTAMP:
		case LogicalTypeId::TIMESTAMP_NS:
		case LogicalTypeId::TIMESTAMP_MS:
		case LogicalTypeId::TIMESTAMP_SEC:
		case LogicalTypeId::TIME_TZ:
		case LogicalTypeId::TIMESTAMP_TZ:
			return true;
		default:
			return false;
		}
	}
	return true;
}

string BoundCastExpression::ToString() const {
	return (try_cast ? "TRY_CAST(" : "CAST(") + child->GetName() + " AS " + return_type.ToString() + ")";
}

bool BoundCastExpression::Equals(const BaseExpression *other_p) const {
	if (!Expression::Equals(other_p)) {
		return false;
	}
	auto &other = other_p->Cast<BoundCastExpression>();
	if (!Expression::Equals(child.get(), other.child.get())) {
		return false;
	}
	if (try_cast != other.try_cast) {
		return false;
	}
	return true;
}

unique_ptr<Expression> BoundCastExpression::Copy() {
	auto copy = make_uniq<BoundCastExpression>(child->Copy(), return_type, bound_cast.Copy(), try_cast);
	copy->CopyProperties(*this);
	return std::move(copy);
}

void BoundCastExpression::Serialize(FieldWriter &writer) const {
	writer.WriteSerializable(*child);
	writer.WriteSerializable(return_type);
	writer.WriteField(try_cast);
}

unique_ptr<Expression> BoundCastExpression::Deserialize(ExpressionDeserializationState &state, FieldReader &reader) {
	auto child = reader.ReadRequiredSerializable<Expression>(state.gstate);
	auto target_type = reader.ReadRequiredSerializable<LogicalType, LogicalType>();
	auto try_cast = reader.ReadRequired<bool>();
	auto cast_function = BindCastFunction(state.gstate.context, child->return_type, target_type);
	return make_uniq<BoundCastExpression>(std::move(child), std::move(target_type), std::move(cast_function), try_cast);
}

} // namespace duckdb






namespace duckdb {

BoundColumnRefExpression::BoundColumnRefExpression(string alias_p, LogicalType type, ColumnBinding binding, idx_t depth)
    : Expression(ExpressionType::BOUND_COLUMN_REF, ExpressionClass::BOUND_COLUMN_REF, std::move(type)),
      binding(binding), depth(depth) {
	this->alias = std::move(alias_p);
}

BoundColumnRefExpression::BoundColumnRefExpression(LogicalType type, ColumnBinding binding, idx_t depth)
    : BoundColumnRefExpression(string(), std::move(type), binding, depth) {
}

unique_ptr<Expression> BoundColumnRefExpression::Copy() {
	return make_uniq<BoundColumnRefExpression>(alias, return_type, binding, depth);
}

hash_t BoundColumnRefExpression::Hash() const {
	auto result = Expression::Hash();
	result = CombineHash(result, duckdb::Hash<uint64_t>(binding.column_index));
	result = CombineHash(result, duckdb::Hash<uint64_t>(binding.table_index));
	return CombineHash(result, duckdb::Hash<uint64_t>(depth));
}

bool BoundColumnRefExpression::Equals(const BaseExpression *other_p) const {
	if (!Expression::Equals(other_p)) {
		return false;
	}
	auto &other = other_p->Cast<BoundColumnRefExpression>();
	return other.binding == binding && other.depth == depth;
}

string BoundColumnRefExpression::GetName() const {
#ifdef DEBUG
	if (DBConfigOptions::debug_print_bindings) {
		return binding.ToString();
	}
#endif
	return Expression::GetName();
}

string BoundColumnRefExpression::ToString() const {
#ifdef DEBUG
	if (DBConfigOptions::debug_print_bindings) {
		return binding.ToString();
	}
#endif
	if (!alias.empty()) {
		return alias;
	}
	return binding.ToString();
}

void BoundColumnRefExpression::Serialize(FieldWriter &writer) const {
	writer.WriteString(alias);
	writer.WriteSerializable(return_type);
	writer.WriteField(binding.table_index);
	writer.WriteField(binding.column_index);
	writer.WriteField(depth);
}

unique_ptr<Expression> BoundColumnRefExpression::Deserialize(ExpressionDeserializationState &state,
                                                             FieldReader &reader) {
	auto alias = reader.ReadRequired<string>();
	auto return_type = reader.ReadRequiredSerializable<LogicalType, LogicalType>();
	auto table_index = reader.ReadRequired<idx_t>();
	auto column_index = reader.ReadRequired<idx_t>();
	auto depth = reader.ReadRequired<idx_t>();

	return make_uniq<BoundColumnRefExpression>(alias, return_type, ColumnBinding(table_index, column_index), depth);
}

} // namespace duckdb




namespace duckdb {

BoundComparisonExpression::BoundComparisonExpression(ExpressionType type, unique_ptr<Expression> left,
                                                     unique_ptr<Expression> right)
    : Expression(type, ExpressionClass::BOUND_COMPARISON, LogicalType::BOOLEAN), left(std::move(left)),
      right(std::move(right)) {
}

string BoundComparisonExpression::ToString() const {
	return ComparisonExpression::ToString<BoundComparisonExpression, Expression>(*this);
}

bool BoundComparisonExpression::Equals(const BaseExpression *other_p) const {
	if (!Expression::Equals(other_p)) {
		return false;
	}
	auto &other = other_p->Cast<BoundComparisonExpression>();
	if (!Expression::Equals(left.get(), other.left.get())) {
		return false;
	}
	if (!Expression::Equals(right.get(), other.right.get())) {
		return false;
	}
	return true;
}

unique_ptr<Expression> BoundComparisonExpression::Copy() {
	auto copy = make_uniq<BoundComparisonExpression>(type, left->Copy(), right->Copy());
	copy->CopyProperties(*this);
	return std::move(copy);
}

void BoundComparisonExpression::Serialize(FieldWriter &writer) const {
	writer.WriteOptional(left);
	writer.WriteOptional(right);
}

unique_ptr<Expression> BoundComparisonExpression::Deserialize(ExpressionDeserializationState &state,
                                                              FieldReader &reader) {
	auto left = reader.ReadOptional<Expression>(nullptr, state.gstate);
	auto right = reader.ReadOptional<Expression>(nullptr, state.gstate);
	return make_uniq<BoundComparisonExpression>(state.type, std::move(left), std::move(right));
}

} // namespace duckdb





namespace duckdb {

BoundConjunctionExpression::BoundConjunctionExpression(ExpressionType type)
    : Expression(type, ExpressionClass::BOUND_CONJUNCTION, LogicalType::BOOLEAN) {
}

BoundConjunctionExpression::BoundConjunctionExpression(ExpressionType type, unique_ptr<Expression> left,
                                                       unique_ptr<Expression> right)
    : BoundConjunctionExpression(type) {
	children.push_back(std::move(left));
	children.push_back(std::move(right));
}

string BoundConjunctionExpression::ToString() const {
	return ConjunctionExpression::ToString<BoundConjunctionExpression, Expression>(*this);
}

bool BoundConjunctionExpression::Equals(const BaseExpression *other_p) const {
	if (!Expression::Equals(other_p)) {
		return false;
	}
	auto &other = other_p->Cast<BoundConjunctionExpression>();
	return ExpressionUtil::SetEquals(children, other.children);
}

bool BoundConjunctionExpression::PropagatesNullValues() const {
	return false;
}

unique_ptr<Expression> BoundConjunctionExpression::Copy() {
	auto copy = make_uniq<BoundConjunctionExpression>(type);
	for (auto &expr : children) {
		copy->children.push_back(expr->Copy());
	}
	copy->CopyProperties(*this);
	return std::move(copy);
}

void BoundConjunctionExpression::Serialize(FieldWriter &writer) const {
	writer.WriteSerializableList(children);
}

unique_ptr<Expression> BoundConjunctionExpression::Deserialize(ExpressionDeserializationState &state,
                                                               FieldReader &reader) {
	auto children = reader.ReadRequiredSerializableList<Expression>(state.gstate);
	auto res = make_uniq<BoundConjunctionExpression>(state.type);
	res->children = std::move(children);
	return std::move(res);
}

} // namespace duckdb





namespace duckdb {

BoundConstantExpression::BoundConstantExpression(Value value_p)
    : Expression(ExpressionType::VALUE_CONSTANT, ExpressionClass::BOUND_CONSTANT, value_p.type()),
      value(std::move(value_p)) {
}

string BoundConstantExpression::ToString() const {
	return value.ToSQLString();
}

bool BoundConstantExpression::Equals(const BaseExpression *other_p) const {
	if (!Expression::Equals(other_p)) {
		return false;
	}
	auto &other = other_p->Cast<BoundConstantExpression>();
	return value.type() == other.value.type() && !ValueOperations::DistinctFrom(value, other.value);
}

hash_t BoundConstantExpression::Hash() const {
	hash_t result = Expression::Hash();
	return CombineHash(value.Hash(), result);
}

unique_ptr<Expression> BoundConstantExpression::Copy() {
	auto copy = make_uniq<BoundConstantExpression>(value);
	copy->CopyProperties(*this);
	return std::move(copy);
}

void BoundConstantExpression::Serialize(FieldWriter &writer) const {
	value.Serialize(writer.GetSerializer());
}

unique_ptr<Expression> BoundConstantExpression::Deserialize(ExpressionDeserializationState &state,
                                                            FieldReader &reader) {
	auto value = Value::Deserialize(reader.GetSource());
	return make_uniq<BoundConstantExpression>(value);
}

} // namespace duckdb



namespace duckdb {

void BoundDefaultExpression::Serialize(FieldWriter &writer) const {
	throw NotImplementedException(ExpressionTypeToString(type));
}

} // namespace duckdb


namespace duckdb {

BoundExpression::BoundExpression(unique_ptr<Expression> expr_p)
    : ParsedExpression(ExpressionType::INVALID, ExpressionClass::BOUND_EXPRESSION), expr(std::move(expr_p)) {
	this->alias = expr->alias;
}

unique_ptr<Expression> &BoundExpression::GetExpression(ParsedExpression &expr) {
	auto &bound_expr = expr.Cast<BoundExpression>();
	if (!bound_expr.expr) {
		throw InternalException("BoundExpression::GetExpression called on empty bound expression");
	}
	return bound_expr.expr;
}

string BoundExpression::ToString() const {
	if (!expr) {
		throw InternalException("ToString(): BoundExpression does not have a child");
	}
	return expr->ToString();
}

bool BoundExpression::Equals(const BaseExpression *other) const {
	return false;
}
hash_t BoundExpression::Hash() const {
	return 0;
}

unique_ptr<ParsedExpression> BoundExpression::Copy() const {
	throw SerializationException("Cannot copy or serialize bound expression");
}

void BoundExpression::Serialize(FieldWriter &writer) const {
	throw SerializationException("Cannot copy or serialize bound expression");
}

void BoundExpression::FormatSerialize(FormatSerializer &serializer) const {
	throw SerializationException("Cannot copy or serialize bound expression");
}

} // namespace duckdb







namespace duckdb {

BoundFunctionExpression::BoundFunctionExpression(LogicalType return_type, ScalarFunction bound_function,
                                                 vector<unique_ptr<Expression>> arguments,
                                                 unique_ptr<FunctionData> bind_info, bool is_operator)
    : Expression(ExpressionType::BOUND_FUNCTION, ExpressionClass::BOUND_FUNCTION, std::move(return_type)),
      function(std::move(bound_function)), children(std::move(arguments)), bind_info(std::move(bind_info)),
      is_operator(is_operator) {
	D_ASSERT(!function.name.empty());
}

bool BoundFunctionExpression::HasSideEffects() const {
	return function.side_effects == FunctionSideEffects::HAS_SIDE_EFFECTS ? true : Expression::HasSideEffects();
}

bool BoundFunctionExpression::IsFoldable() const {
	// functions with side effects cannot be folded: they have to be executed once for every row
	return function.side_effects == FunctionSideEffects::HAS_SIDE_EFFECTS ? false : Expression::IsFoldable();
}

string BoundFunctionExpression::ToString() const {
	return FunctionExpression::ToString<BoundFunctionExpression, Expression>(*this, string(), function.name,
	                                                                         is_operator);
}
bool BoundFunctionExpression::PropagatesNullValues() const {
	return function.null_handling == FunctionNullHandling::SPECIAL_HANDLING ? false
	                                                                        : Expression::PropagatesNullValues();
}

hash_t BoundFunctionExpression::Hash() const {
	hash_t result = Expression::Hash();
	return CombineHash(result, function.Hash());
}

bool BoundFunctionExpression::Equals(const BaseExpression *other_p) const {
	if (!Expression::Equals(other_p)) {
		return false;
	}
	auto &other = other_p->Cast<BoundFunctionExpression>();
	if (other.function != function) {
		return false;
	}
	if (!ExpressionUtil::ListEquals(children, other.children)) {
		return false;
	}
	if (!FunctionData::Equals(bind_info.get(), other.bind_info.get())) {
		return false;
	}
	return true;
}

unique_ptr<Expression> BoundFunctionExpression::Copy() {
	vector<unique_ptr<Expression>> new_children;
	new_children.reserve(children.size());
	for (auto &child : children) {
		new_children.push_back(child->Copy());
	}
	unique_ptr<FunctionData> new_bind_info = bind_info ? bind_info->Copy() : nullptr;

	auto copy = make_uniq<BoundFunctionExpression>(return_type, function, std::move(new_children),
	                                               std::move(new_bind_info), is_operator);
	copy->CopyProperties(*this);
	return std::move(copy);
}

void BoundFunctionExpression::Verify() const {
	D_ASSERT(!function.name.empty());
}

void BoundFunctionExpression::Serialize(FieldWriter &writer) const {
	D_ASSERT(!function.name.empty());
	D_ASSERT(return_type == function.return_type);
	writer.WriteField(is_operator);
	FunctionSerializer::Serialize<ScalarFunction>(writer, function, return_type, children, bind_info.get());
}

unique_ptr<Expression> BoundFunctionExpression::Deserialize(ExpressionDeserializationState &state,
                                                            FieldReader &reader) {
	auto is_operator = reader.ReadRequired<bool>();
	vector<unique_ptr<Expression>> children;
	unique_ptr<FunctionData> bind_info;
	auto function = FunctionSerializer::Deserialize<ScalarFunction, ScalarFunctionCatalogEntry>(
	    reader, state, CatalogType::SCALAR_FUNCTION_ENTRY, children, bind_info);

	auto return_type = function.return_type;
	return make_uniq<BoundFunctionExpression>(std::move(return_type), std::move(function), std::move(children),
	                                          std::move(bind_info), is_operator);
}
} // namespace duckdb




namespace duckdb {

BoundLambdaExpression::BoundLambdaExpression(ExpressionType type_p, LogicalType return_type_p,
                                             unique_ptr<Expression> lambda_expr_p, idx_t parameter_count_p)
    : Expression(type_p, ExpressionClass::BOUND_LAMBDA, std::move(return_type_p)),
      lambda_expr(std::move(lambda_expr_p)), parameter_count(parameter_count_p) {
}

string BoundLambdaExpression::ToString() const {
	return lambda_expr->ToString();
}

bool BoundLambdaExpression::Equals(const BaseExpression *other_p) const {
	if (!Expression::Equals(other_p)) {
		return false;
	}
	auto &other = other_p->Cast<BoundLambdaExpression>();
	if (!Expression::Equals(lambda_expr.get(), other.lambda_expr.get())) {
		return false;
	}
	if (!ExpressionUtil::ListEquals(captures, other.captures)) {
		return false;
	}
	if (parameter_count != other.parameter_count) {
		return false;
	}
	return true;
}

unique_ptr<Expression> BoundLambdaExpression::Copy() {
	auto copy = make_uniq<BoundLambdaExpression>(type, return_type, lambda_expr->Copy(), parameter_count);
	for (auto &capture : captures) {
		copy->captures.push_back(capture->Copy());
	}
	return std::move(copy);
}

void BoundLambdaExpression::Serialize(FieldWriter &writer) const {
	throw NotImplementedException(ExpressionTypeToString(type));
}

} // namespace duckdb






namespace duckdb {

BoundLambdaRefExpression::BoundLambdaRefExpression(string alias_p, LogicalType type, ColumnBinding binding,
                                                   idx_t lambda_index, idx_t depth)
    : Expression(ExpressionType::BOUND_LAMBDA_REF, ExpressionClass::BOUND_LAMBDA_REF, std::move(type)),
      binding(binding), lambda_index(lambda_index), depth(depth) {
	this->alias = std::move(alias_p);
}

BoundLambdaRefExpression::BoundLambdaRefExpression(LogicalType type, ColumnBinding binding, idx_t lambda_index,
                                                   idx_t depth)
    : BoundLambdaRefExpression(string(), std::move(type), binding, lambda_index, depth) {
}

unique_ptr<Expression> BoundLambdaRefExpression::Copy() {
	return make_uniq<BoundLambdaRefExpression>(alias, return_type, binding, lambda_index, depth);
}

hash_t BoundLambdaRefExpression::Hash() const {
	auto result = Expression::Hash();
	result = CombineHash(result, duckdb::Hash<uint64_t>(lambda_index));
	result = CombineHash(result, duckdb::Hash<uint64_t>(binding.column_index));
	result = CombineHash(result, duckdb::Hash<uint64_t>(binding.table_index));
	return CombineHash(result, duckdb::Hash<uint64_t>(depth));
}

bool BoundLambdaRefExpression::Equals(const BaseExpression *other_p) const {
	if (!Expression::Equals(other_p)) {
		return false;
	}
	auto &other = other_p->Cast<BoundLambdaRefExpression>();
	return other.binding == binding && other.lambda_index == lambda_index && other.depth == depth;
}

string BoundLambdaRefExpression::ToString() const {
	if (!alias.empty()) {
		return alias;
	}
	return "#[" + to_string(binding.table_index) + "." + to_string(binding.column_index) + "." +
	       to_string(lambda_index) + "]";
}

void BoundLambdaRefExpression::Serialize(FieldWriter &writer) const {
	writer.WriteString(alias);
	writer.WriteSerializable(return_type);
	writer.WriteField(lambda_index);
	writer.WriteField(binding.table_index);
	writer.WriteField(binding.column_index);
	writer.WriteField(depth);
}

unique_ptr<Expression> BoundLambdaRefExpression::Deserialize(ExpressionDeserializationState &state,
                                                             FieldReader &reader) {
	auto alias = reader.ReadRequired<string>();
	auto return_type = reader.ReadRequiredSerializable<LogicalType, LogicalType>();
	auto lambda_index = reader.ReadRequired<idx_t>();
	auto table_index = reader.ReadRequired<idx_t>();
	auto column_index = reader.ReadRequired<idx_t>();
	auto depth = reader.ReadRequired<idx_t>();

	return make_uniq<BoundLambdaRefExpression>(alias, return_type, ColumnBinding(table_index, column_index),
	                                           lambda_index, depth);
}

} // namespace duckdb






namespace duckdb {

BoundOperatorExpression::BoundOperatorExpression(ExpressionType type, LogicalType return_type)
    : Expression(type, ExpressionClass::BOUND_OPERATOR, std::move(return_type)) {
}

string BoundOperatorExpression::ToString() const {
	return OperatorExpression::ToString<BoundOperatorExpression, Expression>(*this);
}

bool BoundOperatorExpression::Equals(const BaseExpression *other_p) const {
	if (!Expression::Equals(other_p)) {
		return false;
	}
	auto &other = other_p->Cast<BoundOperatorExpression>();
	if (!ExpressionUtil::ListEquals(children, other.children)) {
		return false;
	}
	return true;
}

unique_ptr<Expression> BoundOperatorExpression::Copy() {
	auto copy = make_uniq<BoundOperatorExpression>(type, return_type);
	copy->CopyProperties(*this);
	for (auto &child : children) {
		copy->children.push_back(child->Copy());
	}
	return std::move(copy);
}

void BoundOperatorExpression::Serialize(FieldWriter &writer) const {
	writer.WriteSerializable(return_type);
	writer.WriteSerializableList(children);
}

unique_ptr<Expression> BoundOperatorExpression::Deserialize(ExpressionDeserializationState &state,
                                                            FieldReader &reader) {
	auto return_type = reader.ReadRequiredSerializable<LogicalType, LogicalType>();
	auto children = reader.ReadRequiredSerializableList<Expression>(state.gstate);

	auto result = make_uniq<BoundOperatorExpression>(state.type, return_type);
	result->children = std::move(children);
	return std::move(result);
}

} // namespace duckdb






namespace duckdb {

BoundParameterExpression::BoundParameterExpression(idx_t parameter_nr)
    : Expression(ExpressionType::VALUE_PARAMETER, ExpressionClass::BOUND_PARAMETER,
                 LogicalType(LogicalTypeId::UNKNOWN)),
      parameter_nr(parameter_nr) {
}

void BoundParameterExpression::Invalidate(Expression &expr) {
	if (expr.type != ExpressionType::VALUE_PARAMETER) {
		throw InternalException("BoundParameterExpression::Invalidate requires a parameter as input");
	}
	auto &bound_parameter = expr.Cast<BoundParameterExpression>();
	bound_parameter.return_type = LogicalTypeId::SQLNULL;
	bound_parameter.parameter_data->return_type = LogicalTypeId::INVALID;
}

void BoundParameterExpression::InvalidateRecursive(Expression &expr) {
	if (expr.type == ExpressionType::VALUE_PARAMETER) {
		Invalidate(expr);
		return;
	}
	ExpressionIterator::EnumerateChildren(expr, [&](Expression &child) { InvalidateRecursive(child); });
}

bool BoundParameterExpression::IsScalar() const {
	return true;
}
bool BoundParameterExpression::HasParameter() const {
	return true;
}
bool BoundParameterExpression::IsFoldable() const {
	return false;
}

string BoundParameterExpression::ToString() const {
	return "$" + to_string(parameter_nr);
}

bool BoundParameterExpression::Equals(const BaseExpression *other_p) const {
	if (!Expression::Equals(other_p)) {
		return false;
	}
	auto &other = other_p->Cast<BoundParameterExpression>();
	return parameter_nr == other.parameter_nr;
}

hash_t BoundParameterExpression::Hash() const {
	hash_t result = Expression::Hash();
	result = CombineHash(duckdb::Hash(parameter_nr), result);
	return result;
}

unique_ptr<Expression> BoundParameterExpression::Copy() {
	auto result = make_uniq<BoundParameterExpression>(parameter_nr);
	result->parameter_data = parameter_data;
	result->return_type = return_type;
	result->CopyProperties(*this);
	return std::move(result);
}

void BoundParameterExpression::Serialize(FieldWriter &writer) const {
	writer.WriteField(parameter_nr);
	writer.WriteSerializable(return_type);
	writer.WriteSerializable(*parameter_data);
}

unique_ptr<Expression> BoundParameterExpression::Deserialize(ExpressionDeserializationState &state,
                                                             FieldReader &reader) {
	auto &global_parameter_set = state.gstate.parameter_data;
	auto parameter_nr = reader.ReadRequired<idx_t>();
	auto result = make_uniq<BoundParameterExpression>(parameter_nr);
	result->return_type = reader.ReadRequiredSerializable<LogicalType, LogicalType>();
	auto parameter_data = reader.ReadRequiredSerializable<BoundParameterData, shared_ptr<BoundParameterData>>();
	// check if we have already deserialized a parameter with this number
	auto entry = global_parameter_set.find(parameter_nr);
	if (entry == global_parameter_set.end()) {
		// we have not - store the entry we deserialized from this parameter expression
		global_parameter_set[parameter_nr] = parameter_data;
	} else {
		// we have! use the previously deserialized entry
		parameter_data = entry->second;
	}
	result->parameter_data = std::move(parameter_data);
	return std::move(result);
}

} // namespace duckdb








namespace duckdb {

BoundReferenceExpression::BoundReferenceExpression(string alias, LogicalType type, idx_t index)
    : Expression(ExpressionType::BOUND_REF, ExpressionClass::BOUND_REF, std::move(type)), index(index) {
	this->alias = std::move(alias);
}
BoundReferenceExpression::BoundReferenceExpression(LogicalType type, idx_t index)
    : BoundReferenceExpression(string(), std::move(type), index) {
}

string BoundReferenceExpression::ToString() const {
#ifdef DEBUG
	if (DBConfigOptions::debug_print_bindings) {
		return "#" + to_string(index);
	}
#endif
	if (!alias.empty()) {
		return alias;
	}
	return "#" + to_string(index);
}

bool BoundReferenceExpression::Equals(const BaseExpression *other_p) const {
	if (!Expression::Equals(other_p)) {
		return false;
	}
	auto &other = other_p->Cast<BoundReferenceExpression>();
	return other.index == index;
}

hash_t BoundReferenceExpression::Hash() const {
	return CombineHash(Expression::Hash(), duckdb::Hash<idx_t>(index));
}

unique_ptr<Expression> BoundReferenceExpression::Copy() {
	return make_uniq<BoundReferenceExpression>(alias, return_type, index);
}

void BoundReferenceExpression::Serialize(FieldWriter &writer) const {
	writer.WriteString(alias);
	writer.WriteSerializable(return_type);
	writer.WriteField(index);
}

unique_ptr<Expression> BoundReferenceExpression::Deserialize(ExpressionDeserializationState &state,
                                                             FieldReader &reader) {
	auto alias = reader.ReadRequired<string>();
	auto return_type = reader.ReadRequiredSerializable<LogicalType, LogicalType>();
	auto index = reader.ReadRequired<idx_t>();
	return make_uniq<BoundReferenceExpression>(alias, return_type, index);
}

} // namespace duckdb




namespace duckdb {

BoundSubqueryExpression::BoundSubqueryExpression(LogicalType return_type)
    : Expression(ExpressionType::SUBQUERY, ExpressionClass::BOUND_SUBQUERY, std::move(return_type)) {
}

string BoundSubqueryExpression::ToString() const {
	return "SUBQUERY";
}

bool BoundSubqueryExpression::Equals(const BaseExpression *other_p) const {
	// equality between bound subqueries not implemented currently
	return false;
}

unique_ptr<Expression> BoundSubqueryExpression::Copy() {
	throw SerializationException("Cannot copy BoundSubqueryExpression");
}

bool BoundSubqueryExpression::PropagatesNullValues() const {
	// TODO this can be optimized further by checking the actual subquery node
	return false;
}

void BoundSubqueryExpression::Serialize(FieldWriter &writer) const {
	throw NotImplementedException(ExpressionTypeToString(type));
}

} // namespace duckdb






namespace duckdb {

BoundUnnestExpression::BoundUnnestExpression(LogicalType return_type)
    : Expression(ExpressionType::BOUND_UNNEST, ExpressionClass::BOUND_UNNEST, std::move(return_type)) {
}

bool BoundUnnestExpression::IsFoldable() const {
	return false;
}

string BoundUnnestExpression::ToString() const {
	return "UNNEST(" + child->ToString() + ")";
}

hash_t BoundUnnestExpression::Hash() const {
	hash_t result = Expression::Hash();
	return CombineHash(result, duckdb::Hash("unnest"));
}

bool BoundUnnestExpression::Equals(const BaseExpression *other_p) const {
	if (!Expression::Equals(other_p)) {
		return false;
	}
	auto &other = other_p->Cast<BoundUnnestExpression>();
	if (!Expression::Equals(child.get(), other.child.get())) {
		return false;
	}
	return true;
}

unique_ptr<Expression> BoundUnnestExpression::Copy() {
	auto copy = make_uniq<BoundUnnestExpression>(return_type);
	copy->child = child->Copy();
	return std::move(copy);
}

void BoundUnnestExpression::Serialize(FieldWriter &writer) const {
	writer.WriteSerializable(return_type);
	writer.WriteSerializable(*child);
}

unique_ptr<Expression> BoundUnnestExpression::Deserialize(ExpressionDeserializationState &state, FieldReader &reader) {
	auto return_type = reader.ReadRequiredSerializable<LogicalType, LogicalType>();
	auto child = reader.ReadRequiredSerializable<Expression>(state.gstate);

	auto result = make_uniq<BoundUnnestExpression>(return_type);
	result->child = std::move(child);
	return std::move(result);
}

} // namespace duckdb








namespace duckdb {

BoundWindowExpression::BoundWindowExpression(ExpressionType type, LogicalType return_type,
                                             unique_ptr<AggregateFunction> aggregate,
                                             unique_ptr<FunctionData> bind_info)
    : Expression(type, ExpressionClass::BOUND_WINDOW, std::move(return_type)), aggregate(std::move(aggregate)),
      bind_info(std::move(bind_info)), ignore_nulls(false) {
}

string BoundWindowExpression::ToString() const {
	string function_name = aggregate.get() ? aggregate->name : ExpressionTypeToString(type);
	return WindowExpression::ToString<BoundWindowExpression, Expression, BoundOrderByNode>(*this, string(),
	                                                                                       function_name);
}

bool BoundWindowExpression::Equals(const BaseExpression *other_p) const {
	if (!Expression::Equals(other_p)) {
		return false;
	}
	auto &other = other_p->Cast<BoundWindowExpression>();

	if (ignore_nulls != other.ignore_nulls) {
		return false;
	}
	if (start != other.start || end != other.end) {
		return false;
	}
	// check if the child expressions are equivalent
	if (other.children.size() != children.size()) {
		return false;
	}
	for (idx_t i = 0; i < children.size(); i++) {
		if (!Expression::Equals(children[i].get(), other.children[i].get())) {
			return false;
		}
	}
	// check if the filter expressions are equivalent
	if (!Expression::Equals(filter_expr.get(), other.filter_expr.get())) {
		return false;
	}

	// check if the framing expressions are equivalent
	if (!Expression::Equals(start_expr.get(), other.start_expr.get()) ||
	    !Expression::Equals(end_expr.get(), other.end_expr.get()) ||
	    !Expression::Equals(offset_expr.get(), other.offset_expr.get()) ||
	    !Expression::Equals(default_expr.get(), other.default_expr.get())) {
		return false;
	}

	return KeysAreCompatible(other);
}

bool BoundWindowExpression::KeysAreCompatible(const BoundWindowExpression &other) const {
	// check if the partitions are equivalent
	if (partitions.size() != other.partitions.size()) {
		return false;
	}
	for (idx_t i = 0; i < partitions.size(); i++) {
		if (!Expression::Equals(partitions[i].get(), other.partitions[i].get())) {
			return false;
		}
	}
	// check if the orderings are equivalent
	if (orders.size() != other.orders.size()) {
		return false;
	}
	for (idx_t i = 0; i < orders.size(); i++) {
		if (orders[i].type != other.orders[i].type) {
			return false;
		}
		if (!BaseExpression::Equals((BaseExpression *)orders[i].expression.get(),
		                            (BaseExpression *)other.orders[i].expression.get())) {
			return false;
		}
	}
	return true;
}

unique_ptr<Expression> BoundWindowExpression::Copy() {
	auto new_window = make_uniq<BoundWindowExpression>(type, return_type, nullptr, nullptr);
	new_window->CopyProperties(*this);

	if (aggregate) {
		new_window->aggregate = make_uniq<AggregateFunction>(*aggregate);
	}
	if (bind_info) {
		new_window->bind_info = bind_info->Copy();
	}
	for (auto &child : children) {
		new_window->children.push_back(child->Copy());
	}
	for (auto &e : partitions) {
		new_window->partitions.push_back(e->Copy());
	}
	for (auto &ps : partitions_stats) {
		if (ps) {
			new_window->partitions_stats.push_back(ps->ToUnique());
		} else {
			new_window->partitions_stats.push_back(nullptr);
		}
	}
	for (auto &o : orders) {
		new_window->orders.emplace_back(o.type, o.null_order, o.expression->Copy());
	}

	new_window->filter_expr = filter_expr ? filter_expr->Copy() : nullptr;

	new_window->start = start;
	new_window->end = end;
	new_window->start_expr = start_expr ? start_expr->Copy() : nullptr;
	new_window->end_expr = end_expr ? end_expr->Copy() : nullptr;
	new_window->offset_expr = offset_expr ? offset_expr->Copy() : nullptr;
	new_window->default_expr = default_expr ? default_expr->Copy() : nullptr;
	new_window->ignore_nulls = ignore_nulls;

	return std::move(new_window);
}

void BoundWindowExpression::Serialize(FieldWriter &writer) const {
	writer.WriteField<bool>(aggregate.get());
	if (aggregate) {
		D_ASSERT(return_type == aggregate->return_type);
		FunctionSerializer::Serialize<AggregateFunction>(writer, *aggregate, return_type, children, bind_info.get());
	} else {
		// children and return_type are written as part of the aggregate function otherwise
		writer.WriteSerializableList(children);
		writer.WriteSerializable(return_type);
	}
	writer.WriteSerializableList(partitions);
	writer.WriteRegularSerializableList(orders);
	// FIXME: partitions_stats
	writer.WriteOptional(filter_expr);
	writer.WriteField<bool>(ignore_nulls);
	writer.WriteField<WindowBoundary>(start);
	writer.WriteField<WindowBoundary>(end);
	writer.WriteOptional(start_expr);
	writer.WriteOptional(end_expr);
	writer.WriteOptional(offset_expr);
	writer.WriteOptional(default_expr);
}

unique_ptr<Expression> BoundWindowExpression::Deserialize(ExpressionDeserializationState &state, FieldReader &reader) {
	auto has_aggregate = reader.ReadRequired<bool>();
	unique_ptr<AggregateFunction> aggregate;
	unique_ptr<FunctionData> bind_info;
	vector<unique_ptr<Expression>> children;
	LogicalType return_type;
	if (has_aggregate) {
		auto aggr_function = FunctionSerializer::Deserialize<AggregateFunction, AggregateFunctionCatalogEntry>(
		    reader, state, CatalogType::AGGREGATE_FUNCTION_ENTRY, children, bind_info);
		aggregate = make_uniq<AggregateFunction>(std::move(aggr_function));
		return_type = aggregate->return_type;
	} else {
		children = reader.ReadRequiredSerializableList<Expression>(state.gstate);
		return_type = reader.ReadRequiredSerializable<LogicalType, LogicalType>();
	}
	auto result = make_uniq<BoundWindowExpression>(state.type, return_type, std::move(aggregate), std::move(bind_info));

	result->partitions = reader.ReadRequiredSerializableList<Expression>(state.gstate);
	result->orders = reader.ReadRequiredSerializableList<BoundOrderByNode, BoundOrderByNode>(state.gstate);
	result->filter_expr = reader.ReadOptional<Expression>(nullptr, state.gstate);
	result->ignore_nulls = reader.ReadRequired<bool>();
	result->start = reader.ReadRequired<WindowBoundary>();
	result->end = reader.ReadRequired<WindowBoundary>();
	result->start_expr = reader.ReadOptional<Expression>(nullptr, state.gstate);
	result->end_expr = reader.ReadOptional<Expression>(nullptr, state.gstate);
	result->offset_expr = reader.ReadOptional<Expression>(nullptr, state.gstate);
	result->default_expr = reader.ReadOptional<Expression>(nullptr, state.gstate);
	result->children = std::move(children);
	return std::move(result);
}

} // namespace duckdb









namespace duckdb {

Expression::Expression(ExpressionType type, ExpressionClass expression_class, LogicalType return_type)
    : BaseExpression(type, expression_class), return_type(std::move(return_type)) {
}

Expression::~Expression() {
}

bool Expression::IsAggregate() const {
	bool is_aggregate = false;
	ExpressionIterator::EnumerateChildren(*this, [&](const Expression &child) { is_aggregate |= child.IsAggregate(); });
	return is_aggregate;
}

bool Expression::IsWindow() const {
	bool is_window = false;
	ExpressionIterator::EnumerateChildren(*this, [&](const Expression &child) { is_window |= child.IsWindow(); });
	return is_window;
}

bool Expression::IsScalar() const {
	bool is_scalar = true;
	ExpressionIterator::EnumerateChildren(*this, [&](const Expression &child) {
		if (!child.IsScalar()) {
			is_scalar = false;
		}
	});
	return is_scalar;
}

bool Expression::HasSideEffects() const {
	bool has_side_effects = false;
	ExpressionIterator::EnumerateChildren(*this, [&](const Expression &child) {
		if (child.HasSideEffects()) {
			has_side_effects = true;
		}
	});
	return has_side_effects;
}

bool Expression::PropagatesNullValues() const {
	if (type == ExpressionType::OPERATOR_IS_NULL || type == ExpressionType::OPERATOR_IS_NOT_NULL ||
	    type == ExpressionType::COMPARE_NOT_DISTINCT_FROM || type == ExpressionType::COMPARE_DISTINCT_FROM ||
	    type == ExpressionType::CONJUNCTION_OR || type == ExpressionType::CONJUNCTION_AND ||
	    type == ExpressionType::OPERATOR_COALESCE) {
		return false;
	}
	bool propagate_null_values = true;
	ExpressionIterator::EnumerateChildren(*this, [&](const Expression &child) {
		if (!child.PropagatesNullValues()) {
			propagate_null_values = false;
		}
	});
	return propagate_null_values;
}

bool Expression::IsFoldable() const {
	bool is_foldable = true;
	ExpressionIterator::EnumerateChildren(*this, [&](const Expression &child) {
		if (!child.IsFoldable()) {
			is_foldable = false;
		}
	});
	return is_foldable;
}

bool Expression::HasParameter() const {
	bool has_parameter = false;
	ExpressionIterator::EnumerateChildren(*this,
	                                      [&](const Expression &child) { has_parameter |= child.HasParameter(); });
	return has_parameter;
}

bool Expression::HasSubquery() const {
	bool has_subquery = false;
	ExpressionIterator::EnumerateChildren(*this, [&](const Expression &child) { has_subquery |= child.HasSubquery(); });
	return has_subquery;
}

hash_t Expression::Hash() const {
	hash_t hash = duckdb::Hash<uint32_t>((uint32_t)type);
	hash = CombineHash(hash, return_type.Hash());
	ExpressionIterator::EnumerateChildren(*this,
	                                      [&](const Expression &child) { hash = CombineHash(child.Hash(), hash); });
	return hash;
}

void Expression::Serialize(Serializer &serializer) const {
	FieldWriter writer(serializer);
	writer.WriteField<ExpressionClass>(expression_class);
	writer.WriteField<ExpressionType>(type);
	writer.WriteString(alias);
	Serialize(writer);
	writer.Finalize();
}

unique_ptr<Expression> Expression::Deserialize(Deserializer &source, PlanDeserializationState &gstate) {
	FieldReader reader(source);
	auto expression_class = reader.ReadRequired<ExpressionClass>();
	auto type = reader.ReadRequired<ExpressionType>();
	auto alias = reader.ReadRequired<string>();

	ExpressionDeserializationState state(gstate, type);

	unique_ptr<Expression> result;
	switch (expression_class) {
	case ExpressionClass::BOUND_REF:
		result = BoundReferenceExpression::Deserialize(state, reader);
		break;
	case ExpressionClass::BOUND_COLUMN_REF:
		result = BoundColumnRefExpression::Deserialize(state, reader);
		break;
	case ExpressionClass::BOUND_AGGREGATE:
		result = BoundAggregateExpression::Deserialize(state, reader);
		break;
	case ExpressionClass::BOUND_BETWEEN:
		result = BoundBetweenExpression::Deserialize(state, reader);
		break;
	case ExpressionClass::BOUND_CONSTANT:
		result = BoundConstantExpression::Deserialize(state, reader);
		break;
	case ExpressionClass::BOUND_FUNCTION:
		result = BoundFunctionExpression::Deserialize(state, reader);
		break;
	case ExpressionClass::BOUND_CAST:
		result = BoundCastExpression::Deserialize(state, reader);
		break;
	case ExpressionClass::BOUND_CASE:
		result = BoundCaseExpression::Deserialize(state, reader);
		break;
	case ExpressionClass::BOUND_CONJUNCTION:
		result = BoundConjunctionExpression::Deserialize(state, reader);
		break;
	case ExpressionClass::BOUND_COMPARISON:
		result = BoundComparisonExpression::Deserialize(state, reader);
		break;
	case ExpressionClass::BOUND_OPERATOR:
		result = BoundOperatorExpression::Deserialize(state, reader);
		break;
	case ExpressionClass::BOUND_WINDOW:
		result = BoundWindowExpression::Deserialize(state, reader);
		break;
	case ExpressionClass::BOUND_UNNEST:
		result = BoundUnnestExpression::Deserialize(state, reader);
		break;
	case ExpressionClass::BOUND_PARAMETER:
		result = BoundParameterExpression::Deserialize(state, reader);
		break;
	default:
		throw SerializationException("Unsupported type for expression deserialization %s",
		                             ExpressionTypeToString(type));
	}
	result->alias = alias;
	reader.Finalize();
	return result;
}

} // namespace duckdb




namespace duckdb {

AggregateBinder::AggregateBinder(Binder &binder, ClientContext &context) : ExpressionBinder(binder, context, true) {
}

BindResult AggregateBinder::BindExpression(unique_ptr<ParsedExpression> &expr_ptr, idx_t depth, bool root_expression) {
	auto &expr = *expr_ptr;
	switch (expr.expression_class) {
	case ExpressionClass::WINDOW:
		throw ParserException("aggregate function calls cannot contain window function calls");
	default:
		return ExpressionBinder::BindExpression(expr_ptr, depth);
	}
}

string AggregateBinder::UnsupportedAggregateMessage() {
	return "aggregate function calls cannot be nested";
}
} // namespace duckdb






namespace duckdb {

AlterBinder::AlterBinder(Binder &binder, ClientContext &context, TableCatalogEntry &table,
                         vector<LogicalIndex> &bound_columns, LogicalType target_type)
    : ExpressionBinder(binder, context), table(table), bound_columns(bound_columns) {
	this->target_type = std::move(target_type);
}

BindResult AlterBinder::BindExpression(unique_ptr<ParsedExpression> &expr_ptr, idx_t depth, bool root_expression) {
	auto &expr = *expr_ptr;
	switch (expr.GetExpressionClass()) {
	case ExpressionClass::WINDOW:
		return BindResult("window functions are not allowed in alter statement");
	case ExpressionClass::SUBQUERY:
		return BindResult("cannot use subquery in alter statement");
	case ExpressionClass::COLUMN_REF:
		return BindColumn(expr.Cast<ColumnRefExpression>());
	default:
		return ExpressionBinder::BindExpression(expr_ptr, depth);
	}
}

string AlterBinder::UnsupportedAggregateMessage() {
	return "aggregate functions are not allowed in alter statement";
}

BindResult AlterBinder::BindColumn(ColumnRefExpression &colref) {
	if (colref.column_names.size() > 1) {
		return BindQualifiedColumnName(colref, table.name);
	}
	auto idx = table.GetColumnIndex(colref.column_names[0], true);
	if (!idx.IsValid()) {
		throw BinderException("Table does not contain column %s referenced in alter statement!",
		                      colref.column_names[0]);
	}
	if (table.GetColumn(idx).Generated()) {
		throw BinderException("Using generated columns in alter statement not supported");
	}
	bound_columns.push_back(idx);
	return BindResult(make_uniq<BoundReferenceExpression>(table.GetColumn(idx).Type(), bound_columns.size() - 1));
}

} // namespace duckdb













namespace duckdb {

BaseSelectBinder::BaseSelectBinder(Binder &binder, ClientContext &context, BoundSelectNode &node,
                                   BoundGroupInformation &info, case_insensitive_map_t<idx_t> alias_map)
    : ExpressionBinder(binder, context), inside_window(false), node(node), info(info), alias_map(std::move(alias_map)) {
}

BaseSelectBinder::BaseSelectBinder(Binder &binder, ClientContext &context, BoundSelectNode &node,
                                   BoundGroupInformation &info)
    : BaseSelectBinder(binder, context, node, info, case_insensitive_map_t<idx_t>()) {
}

BindResult BaseSelectBinder::BindExpression(unique_ptr<ParsedExpression> &expr_ptr, idx_t depth, bool root_expression) {
	auto &expr = *expr_ptr;
	// check if the expression binds to one of the groups
	auto group_index = TryBindGroup(expr, depth);
	if (group_index != DConstants::INVALID_INDEX) {
		return BindGroup(expr, depth, group_index);
	}
	switch (expr.expression_class) {
	case ExpressionClass::COLUMN_REF:
		return BindColumnRef(expr_ptr, depth);
	case ExpressionClass::DEFAULT:
		return BindResult("SELECT clause cannot contain DEFAULT clause");
	case ExpressionClass::WINDOW:
		return BindWindow(expr.Cast<WindowExpression>(), depth);
	default:
		return ExpressionBinder::BindExpression(expr_ptr, depth, root_expression);
	}
}

idx_t BaseSelectBinder::TryBindGroup(ParsedExpression &expr, idx_t depth) {
	// first check the group alias map, if expr is a ColumnRefExpression
	if (expr.type == ExpressionType::COLUMN_REF) {
		auto &colref = expr.Cast<ColumnRefExpression>();
		if (!colref.IsQualified()) {
			auto alias_entry = info.alias_map.find(colref.column_names[0]);
			if (alias_entry != info.alias_map.end()) {
				// found entry!
				return alias_entry->second;
			}
		}
	}
	// no alias reference found
	// check the list of group columns for a match
	auto entry = info.map.find(expr);
	if (entry != info.map.end()) {
		return entry->second;
	}
#ifdef DEBUG
	for (auto entry : info.map) {
		D_ASSERT(!entry.first.get().Equals(&expr));
		D_ASSERT(!expr.Equals(&entry.first.get()));
	}
#endif
	return DConstants::INVALID_INDEX;
}

BindResult BaseSelectBinder::BindColumnRef(unique_ptr<ParsedExpression> &expr_ptr, idx_t depth) {
	// first try to bind the column reference regularly
	auto result = ExpressionBinder::BindExpression(expr_ptr, depth);
	if (!result.HasError()) {
		return result;
	}
	// binding failed
	// check in the alias map
	auto &colref = (expr_ptr.get())->Cast<ColumnRefExpression>();
	if (!colref.IsQualified()) {
		auto alias_entry = alias_map.find(colref.column_names[0]);
		if (alias_entry != alias_map.end()) {
			// found entry!
			auto index = alias_entry->second;
			if (index >= node.select_list.size()) {
				throw BinderException("Column \"%s\" referenced that exists in the SELECT clause - but this column "
				                      "cannot be referenced before it is defined",
				                      colref.column_names[0]);
			}
			if (node.select_list[index]->HasSideEffects()) {
				throw BinderException("Alias \"%s\" referenced in a SELECT clause - but the expression has side "
				                      "effects. This is not yet supported.",
				                      colref.column_names[0]);
			}
			if (node.select_list[index]->HasSubquery()) {
				throw BinderException("Alias \"%s\" referenced in a SELECT clause - but the expression has a subquery."
				                      " This is not yet supported.",
				                      colref.column_names[0]);
			}
			auto result = BindResult(node.select_list[index]->Copy());
			if (result.expression->type == ExpressionType::BOUND_COLUMN_REF) {
				auto &result_expr = result.expression->Cast<BoundColumnRefExpression>();
				result_expr.depth = depth;
			}
			return result;
		}
	}
	// entry was not found in the alias map: return the original error
	return result;
}

BindResult BaseSelectBinder::BindGroupingFunction(OperatorExpression &op, idx_t depth) {
	if (op.children.empty()) {
		throw InternalException("GROUPING requires at least one child");
	}
	if (node.groups.group_expressions.empty()) {
		return BindResult(binder.FormatError(op, "GROUPING statement cannot be used without groups"));
	}
	if (op.children.size() >= 64) {
		return BindResult(binder.FormatError(op, "GROUPING statement cannot have more than 64 groups"));
	}
	vector<idx_t> group_indexes;
	group_indexes.reserve(op.children.size());
	for (auto &child : op.children) {
		ExpressionBinder::QualifyColumnNames(binder, child);
		auto idx = TryBindGroup(*child, depth);
		if (idx == DConstants::INVALID_INDEX) {
			return BindResult(binder.FormatError(
			    op, StringUtil::Format("GROUPING child \"%s\" must be a grouping column", child->GetName())));
		}
		group_indexes.push_back(idx);
	}
	auto col_idx = node.grouping_functions.size();
	node.grouping_functions.push_back(std::move(group_indexes));
	return BindResult(make_uniq<BoundColumnRefExpression>(op.GetName(), LogicalType::BIGINT,
	                                                      ColumnBinding(node.groupings_index, col_idx), depth));
}

BindResult BaseSelectBinder::BindGroup(ParsedExpression &expr, idx_t depth, idx_t group_index) {
	auto &group = node.groups.group_expressions[group_index];
	return BindResult(make_uniq<BoundColumnRefExpression>(expr.GetName(), group->return_type,
	                                                      ColumnBinding(node.group_index, group_index), depth));
}

bool BaseSelectBinder::QualifyColumnAlias(const ColumnRefExpression &colref) {
	if (!colref.IsQualified()) {
		return alias_map.find(colref.column_names[0]) != alias_map.end() ? true : false;
	}
	return false;
}

} // namespace duckdb






namespace duckdb {

CheckBinder::CheckBinder(Binder &binder, ClientContext &context, string table_p, const ColumnList &columns,
                         physical_index_set_t &bound_columns)
    : ExpressionBinder(binder, context), table(std::move(table_p)), columns(columns), bound_columns(bound_columns) {
	target_type = LogicalType::INTEGER;
}

BindResult CheckBinder::BindExpression(unique_ptr<ParsedExpression> &expr_ptr, idx_t depth, bool root_expression) {
	auto &expr = *expr_ptr;
	switch (expr.GetExpressionClass()) {
	case ExpressionClass::WINDOW:
		return BindResult("window functions are not allowed in check constraints");
	case ExpressionClass::SUBQUERY:
		return BindResult("cannot use subquery in check constraint");
	case ExpressionClass::COLUMN_REF:
		return BindCheckColumn(expr.Cast<ColumnRefExpression>());
	default:
		return ExpressionBinder::BindExpression(expr_ptr, depth);
	}
}

string CheckBinder::UnsupportedAggregateMessage() {
	return "aggregate functions are not allowed in check constraints";
}

BindResult ExpressionBinder::BindQualifiedColumnName(ColumnRefExpression &colref, const string &table_name) {
	idx_t struct_start = 0;
	if (colref.column_names[0] == table_name) {
		struct_start++;
	}
	auto result = make_uniq_base<ParsedExpression, ColumnRefExpression>(colref.column_names.back());
	for (idx_t i = struct_start; i + 1 < colref.column_names.size(); i++) {
		result = CreateStructExtract(std::move(result), colref.column_names[i]);
	}
	return BindExpression(result, 0);
}

BindResult CheckBinder::BindCheckColumn(ColumnRefExpression &colref) {

	// if this is a lambda parameters, then we temporarily add a BoundLambdaRef,
	// which we capture and remove later
	if (lambda_bindings) {
		for (idx_t i = 0; i < lambda_bindings->size(); i++) {
			if (colref.GetColumnName() == (*lambda_bindings)[i].dummy_name) {
				// FIXME: support lambdas in CHECK constraints
				// FIXME: like so: return (*lambda_bindings)[i].Bind(colref, i, depth);
				throw NotImplementedException("Lambda functions are currently not supported in CHECK constraints.");
			}
		}
	}

	if (colref.column_names.size() > 1) {
		return BindQualifiedColumnName(colref, table);
	}
	if (!columns.ColumnExists(colref.column_names[0])) {
		throw BinderException("Table does not contain column %s referenced in check constraint!",
		                      colref.column_names[0]);
	}
	auto &col = columns.GetColumn(colref.column_names[0]);
	if (col.Generated()) {
		auto bound_expression = col.GeneratedExpression().Copy();
		return BindExpression(bound_expression, 0, false);
	}
	bound_columns.insert(col.Physical());
	D_ASSERT(col.StorageOid() != DConstants::INVALID_INDEX);
	return BindResult(make_uniq<BoundReferenceExpression>(col.Type(), col.StorageOid()));
}

} // namespace duckdb








namespace duckdb {

ColumnAliasBinder::ColumnAliasBinder(BoundSelectNode &node, const case_insensitive_map_t<idx_t> &alias_map)
    : node(node), alias_map(alias_map), visited_select_indexes() {
}

BindResult ColumnAliasBinder::BindAlias(ExpressionBinder &enclosing_binder, ColumnRefExpression &expr, idx_t depth,
                                        bool root_expression) {
	if (expr.IsQualified()) {
		return BindResult(StringUtil::Format("Alias %s cannot be qualified.", expr.ToString()));
	}

	auto alias_entry = alias_map.find(expr.column_names[0]);
	if (alias_entry == alias_map.end()) {
		return BindResult(StringUtil::Format("Alias %s is not found.", expr.ToString()));
	}

	if (visited_select_indexes.find(alias_entry->second) != visited_select_indexes.end()) {
		return BindResult("Cannot resolve self-referential alias");
	}

	// found an alias: bind the alias expression
	auto expression = node.original_expressions[alias_entry->second]->Copy();
	visited_select_indexes.insert(alias_entry->second);

	// since the alias has been found, pass a depth of 0. See Issue 4978 (#16)
	// ColumnAliasBinders are only in Having, Qualify and Where Binders
	auto result = enclosing_binder.BindExpression(expression, 0, root_expression);
	visited_select_indexes.erase(alias_entry->second);
	return result;
}

} // namespace duckdb



namespace duckdb {

ConstantBinder::ConstantBinder(Binder &binder, ClientContext &context, string clause)
    : ExpressionBinder(binder, context), clause(std::move(clause)) {
}

BindResult ConstantBinder::BindExpression(unique_ptr<ParsedExpression> &expr_ptr, idx_t depth, bool root_expression) {
	auto &expr = *expr_ptr;
	switch (expr.GetExpressionClass()) {
	case ExpressionClass::COLUMN_REF: {
		auto &colref = expr.Cast<ColumnRefExpression>();
		if (!colref.IsQualified()) {
			auto value_function = GetSQLValueFunction(colref.GetColumnName());
			if (value_function) {
				expr_ptr = std::move(value_function);
				return BindExpression(expr_ptr, depth, root_expression);
			}
		}
		return BindResult(clause + " cannot contain column names");
	}
	case ExpressionClass::SUBQUERY:
		throw BinderException(clause + " cannot contain subqueries");
	case ExpressionClass::DEFAULT:
		return BindResult(clause + " cannot contain DEFAULT clause");
	case ExpressionClass::WINDOW:
		return BindResult(clause + " cannot contain window functions!");
	default:
		return ExpressionBinder::BindExpression(expr_ptr, depth);
	}
}

string ConstantBinder::UnsupportedAggregateMessage() {
	return clause + " cannot contain aggregates!";
}

} // namespace duckdb








namespace duckdb {

GroupBinder::GroupBinder(Binder &binder, ClientContext &context, SelectNode &node, idx_t group_index,
                         case_insensitive_map_t<idx_t> &alias_map, case_insensitive_map_t<idx_t> &group_alias_map)
    : ExpressionBinder(binder, context), node(node), alias_map(alias_map), group_alias_map(group_alias_map),
      group_index(group_index) {
}

BindResult GroupBinder::BindExpression(unique_ptr<ParsedExpression> &expr_ptr, idx_t depth, bool root_expression) {
	auto &expr = *expr_ptr;
	if (root_expression && depth == 0) {
		switch (expr.expression_class) {
		case ExpressionClass::COLUMN_REF:
			return BindColumnRef(expr.Cast<ColumnRefExpression>());
		case ExpressionClass::CONSTANT:
			return BindConstant(expr.Cast<ConstantExpression>());
		case ExpressionClass::PARAMETER:
			throw ParameterNotAllowedException("Parameter not supported in GROUP BY clause");
		default:
			break;
		}
	}
	switch (expr.expression_class) {
	case ExpressionClass::DEFAULT:
		return BindResult("GROUP BY clause cannot contain DEFAULT clause");
	case ExpressionClass::WINDOW:
		return BindResult("GROUP BY clause cannot contain window functions!");
	default:
		return ExpressionBinder::BindExpression(expr_ptr, depth);
	}
}

string GroupBinder::UnsupportedAggregateMessage() {
	return "GROUP BY clause cannot contain aggregates!";
}

BindResult GroupBinder::BindSelectRef(idx_t entry) {
	if (used_aliases.find(entry) != used_aliases.end()) {
		// the alias has already been bound to before!
		// this happens if we group on the same alias twice
		// e.g. GROUP BY k, k or GROUP BY 1, 1
		// in this case, we can just replace the grouping with a constant since the second grouping has no effect
		// (the constant grouping will be optimized out later)
		return BindResult(make_uniq<BoundConstantExpression>(Value::INTEGER(42)));
	}
	if (entry >= node.select_list.size()) {
		throw BinderException("GROUP BY term out of range - should be between 1 and %d", (int)node.select_list.size());
	}
	// we replace the root expression, also replace the unbound expression
	unbound_expression = node.select_list[entry]->Copy();
	// move the expression that this refers to here and bind it
	auto select_entry = std::move(node.select_list[entry]);
	auto binding = Bind(select_entry, nullptr, false);
	// now replace the original expression in the select list with a reference to this group
	group_alias_map[to_string(entry)] = bind_index;
	node.select_list[entry] = make_uniq<ColumnRefExpression>(to_string(entry));
	// insert into the set of used aliases
	used_aliases.insert(entry);
	return BindResult(std::move(binding));
}

BindResult GroupBinder::BindConstant(ConstantExpression &constant) {
	// constant as root expression
	if (!constant.value.type().IsIntegral()) {
		// non-integral expression, we just leave the constant here.
		return ExpressionBinder::BindExpression(constant, 0);
	}
	// INTEGER constant: we use the integer as an index into the select list (e.g. GROUP BY 1)
	auto index = (idx_t)constant.value.GetValue<int64_t>();
	return BindSelectRef(index - 1);
}

BindResult GroupBinder::BindColumnRef(ColumnRefExpression &colref) {
	// columns in GROUP BY clauses:
	// FIRST refer to the original tables, and
	// THEN if no match is found refer to aliases in the SELECT list
	// THEN if no match is found, refer to outer queries

	// first try to bind to the base columns (original tables)
	auto result = ExpressionBinder::BindExpression(colref, 0);
	if (result.HasError()) {
		if (colref.IsQualified()) {
			// explicit table name: not an alias reference
			return result;
		}
		// failed to bind the column and the node is the root expression with depth = 0
		// check if refers to an alias in the select clause
		auto alias_name = colref.column_names[0];
		auto entry = alias_map.find(alias_name);
		if (entry == alias_map.end()) {
			// no matching alias found
			return result;
		}
		result = BindResult(BindSelectRef(entry->second));
		if (!result.HasError()) {
			group_alias_map[alias_name] = bind_index;
		}
	}
	return result;
}

} // namespace duckdb








namespace duckdb {

HavingBinder::HavingBinder(Binder &binder, ClientContext &context, BoundSelectNode &node, BoundGroupInformation &info,
                           case_insensitive_map_t<idx_t> &alias_map, AggregateHandling aggregate_handling)
    : BaseSelectBinder(binder, context, node, info), column_alias_binder(node, alias_map),
      aggregate_handling(aggregate_handling) {
	target_type = LogicalType(LogicalTypeId::BOOLEAN);
}

BindResult HavingBinder::BindColumnRef(unique_ptr<ParsedExpression> &expr_ptr, idx_t depth, bool root_expression) {
	auto &expr = expr_ptr->Cast<ColumnRefExpression>();
	auto alias_result = column_alias_binder.BindAlias(*this, expr, depth, root_expression);
	if (!alias_result.HasError()) {
		if (depth > 0) {
			throw BinderException("Having clause cannot reference alias in correlated subquery");
		}
		return alias_result;
	}
	if (aggregate_handling == AggregateHandling::FORCE_AGGREGATES) {
		if (depth > 0) {
			throw BinderException("Having clause cannot reference column in correlated subquery and group by all");
		}
		auto expr = duckdb::BaseSelectBinder::BindExpression(expr_ptr, depth);
		if (expr.HasError()) {
			return expr;
		}
		auto group_ref = make_uniq<BoundColumnRefExpression>(
		    expr.expression->return_type, ColumnBinding(node.group_index, node.groups.group_expressions.size()));
		node.groups.group_expressions.push_back(std::move(expr.expression));
		return BindResult(std::move(group_ref));
	}
	return BindResult(StringUtil::Format(
	    "column %s must appear in the GROUP BY clause or be used in an aggregate function", expr.ToString()));
}

BindResult HavingBinder::BindExpression(unique_ptr<ParsedExpression> &expr_ptr, idx_t depth, bool root_expression) {
	auto &expr = *expr_ptr;
	// check if the expression binds to one of the groups
	auto group_index = TryBindGroup(expr, depth);
	if (group_index != DConstants::INVALID_INDEX) {
		return BindGroup(expr, depth, group_index);
	}
	switch (expr.expression_class) {
	case ExpressionClass::WINDOW:
		return BindResult("HAVING clause cannot contain window functions!");
	case ExpressionClass::COLUMN_REF:
		return BindColumnRef(expr_ptr, depth, root_expression);
	default:
		return duckdb::BaseSelectBinder::BindExpression(expr_ptr, depth);
	}
}

} // namespace duckdb







namespace duckdb {

IndexBinder::IndexBinder(Binder &binder, ClientContext &context, optional_ptr<TableCatalogEntry> table,
                         optional_ptr<CreateIndexInfo> info)
    : ExpressionBinder(binder, context), table(table), info(info) {
}

BindResult IndexBinder::BindExpression(unique_ptr<ParsedExpression> &expr_ptr, idx_t depth, bool root_expression) {
	auto &expr = *expr_ptr;
	switch (expr.expression_class) {
	case ExpressionClass::WINDOW:
		return BindResult("window functions are not allowed in index expressions");
	case ExpressionClass::SUBQUERY:
		return BindResult("cannot use subquery in index expressions");
	case ExpressionClass::COLUMN_REF: {
		if (table) {
			// WAL replay
			// we assume that the parsed expressions have qualified column names
			// and that the columns exist in the table
			auto &col_ref = expr.Cast<ColumnRefExpression>();
			auto col_idx = table->GetColumnIndex(col_ref.column_names.back());
			auto col_type = table->GetColumn(col_idx).GetType();

			// find the col_idx in the index.column_ids
			auto col_id_idx = DConstants::INVALID_INDEX;
			for (idx_t i = 0; i < info->column_ids.size(); i++) {
				if (col_idx.index == info->column_ids[i]) {
					col_id_idx = i;
				}
			}

			if (col_id_idx == DConstants::INVALID_INDEX) {
				throw InternalException("failed to replay CREATE INDEX statement - column id not found");
			}
			return BindResult(
			    make_uniq<BoundColumnRefExpression>(col_ref.GetColumnName(), col_type, ColumnBinding(0, col_id_idx)));
		}
		return ExpressionBinder::BindExpression(expr_ptr, depth);
	}
	default:
		return ExpressionBinder::BindExpression(expr_ptr, depth);
	}
}

string IndexBinder::UnsupportedAggregateMessage() {
	return "aggregate functions are not allowed in index expressions";
}

} // namespace duckdb




namespace duckdb {

InsertBinder::InsertBinder(Binder &binder, ClientContext &context) : ExpressionBinder(binder, context) {
}

BindResult InsertBinder::BindExpression(unique_ptr<ParsedExpression> &expr_ptr, idx_t depth, bool root_expression) {
	auto &expr = *expr_ptr;
	switch (expr.GetExpressionClass()) {
	case ExpressionClass::DEFAULT:
		return BindResult("DEFAULT is not allowed here!");
	case ExpressionClass::WINDOW:
		return BindResult("INSERT statement cannot contain window functions!");
	default:
		return ExpressionBinder::BindExpression(expr_ptr, depth);
	}
}

string InsertBinder::UnsupportedAggregateMessage() {
	return "INSERT statement cannot contain aggregates!";
}

} // namespace duckdb






namespace duckdb {

LateralBinder::LateralBinder(Binder &binder, ClientContext &context) : ExpressionBinder(binder, context) {
}

void LateralBinder::ExtractCorrelatedColumns(Expression &expr) {
	if (expr.type == ExpressionType::BOUND_COLUMN_REF) {
		auto &bound_colref = expr.Cast<BoundColumnRefExpression>();
		if (bound_colref.depth > 0) {
			// add the correlated column info
			CorrelatedColumnInfo info(bound_colref);
			if (std::find(correlated_columns.begin(), correlated_columns.end(), info) == correlated_columns.end()) {
				correlated_columns.push_back(std::move(info));
			}
		}
	}
	ExpressionIterator::EnumerateChildren(expr, [&](Expression &child) { ExtractCorrelatedColumns(child); });
}

BindResult LateralBinder::BindColumnRef(unique_ptr<ParsedExpression> &expr_ptr, idx_t depth, bool root_expression) {
	if (depth == 0) {
		throw InternalException("Lateral binder can only bind correlated columns");
	}
	auto result = ExpressionBinder::BindExpression(expr_ptr, depth);
	if (result.HasError()) {
		return result;
	}
	if (depth > 1) {
		throw BinderException("Nested lateral joins are not supported yet");
	}
	ExtractCorrelatedColumns(*result.expression);
	return result;
}

vector<CorrelatedColumnInfo> LateralBinder::ExtractCorrelatedColumns(Binder &binder) {

	if (correlated_columns.empty()) {
		return binder.correlated_columns;
	}

	// clear outer
	correlated_columns.clear();
	auto all_correlated_columns = binder.correlated_columns;

	// remove outer from inner
	for (auto &corr_column : correlated_columns) {
		auto entry = std::find(binder.correlated_columns.begin(), binder.correlated_columns.end(), corr_column);
		if (entry != binder.correlated_columns.end()) {
			binder.correlated_columns.erase(entry);
		}
	}

	// add inner to outer
	for (auto &corr_column : binder.correlated_columns) {
		correlated_columns.push_back(corr_column);
	}

	// clear inner
	binder.correlated_columns.clear();
	return all_correlated_columns;
}

BindResult LateralBinder::BindExpression(unique_ptr<ParsedExpression> &expr_ptr, idx_t depth, bool root_expression) {
	auto &expr = *expr_ptr;
	switch (expr.GetExpressionClass()) {
	case ExpressionClass::DEFAULT:
		return BindResult("LATERAL join cannot contain DEFAULT clause");
	case ExpressionClass::WINDOW:
		return BindResult("LATERAL join cannot contain window functions!");
	case ExpressionClass::COLUMN_REF:
		return BindColumnRef(expr_ptr, depth, root_expression);
	default:
		return ExpressionBinder::BindExpression(expr_ptr, depth);
	}
}

string LateralBinder::UnsupportedAggregateMessage() {
	return "LATERAL join cannot contain aggregates!";
}

class ExpressionDepthReducer : public LogicalOperatorVisitor {
public:
	explicit ExpressionDepthReducer(const vector<CorrelatedColumnInfo> &correlated) : correlated_columns(correlated) {
	}

protected:
	void ReduceColumnRefDepth(BoundColumnRefExpression &expr) {
		// don't need to reduce this
		if (expr.depth == 0) {
			return;
		}
		for (auto &correlated : correlated_columns) {
			if (correlated.binding == expr.binding) {
				D_ASSERT(expr.depth > 1);
				expr.depth--;
				break;
			}
		}
	}

	unique_ptr<Expression> VisitReplace(BoundColumnRefExpression &expr, unique_ptr<Expression> *expr_ptr) override {
		ReduceColumnRefDepth(expr);
		return nullptr;
	}

	void ReduceExpressionSubquery(BoundSubqueryExpression &expr) {
		for (auto &s_correlated : expr.binder->correlated_columns) {
			for (auto &correlated : correlated_columns) {
				if (correlated == s_correlated) {
					s_correlated.depth--;
					break;
				}
			}
		}
	}

	void ReduceExpressionDepth(Expression &expr) {
		if (expr.GetExpressionType() == ExpressionType::BOUND_COLUMN_REF) {
			ReduceColumnRefDepth(expr.Cast<BoundColumnRefExpression>());
		}
		if (expr.GetExpressionClass() == ExpressionClass::BOUND_SUBQUERY) {
			ReduceExpressionSubquery(expr.Cast<BoundSubqueryExpression>());
		}
	}

	unique_ptr<Expression> VisitReplace(BoundSubqueryExpression &expr, unique_ptr<Expression> *expr_ptr) override {
		ReduceExpressionSubquery(expr);
		ExpressionIterator::EnumerateQueryNodeChildren(
		    *expr.subquery, [&](Expression &child_expr) { ReduceExpressionDepth(child_expr); });
		return nullptr;
	}

	const vector<CorrelatedColumnInfo> &correlated_columns;
};

void LateralBinder::ReduceExpressionDepth(LogicalOperator &op, const vector<CorrelatedColumnInfo> &correlated) {
	ExpressionDepthReducer depth_reducer(correlated);
	depth_reducer.VisitOperator(op);
}

} // namespace duckdb












namespace duckdb {

OrderBinder::OrderBinder(vector<Binder *> binders, idx_t projection_index, case_insensitive_map_t<idx_t> &alias_map,
                         parsed_expression_map_t<idx_t> &projection_map, idx_t max_count)
    : binders(std::move(binders)), projection_index(projection_index), max_count(max_count), extra_list(nullptr),
      alias_map(alias_map), projection_map(projection_map) {
}
OrderBinder::OrderBinder(vector<Binder *> binders, idx_t projection_index, SelectNode &node,
                         case_insensitive_map_t<idx_t> &alias_map, parsed_expression_map_t<idx_t> &projection_map)
    : binders(std::move(binders)), projection_index(projection_index), alias_map(alias_map),
      projection_map(projection_map) {
	this->max_count = node.select_list.size();
	this->extra_list = &node.select_list;
}

unique_ptr<Expression> OrderBinder::CreateProjectionReference(ParsedExpression &expr, idx_t index) {
	string alias;
	if (extra_list && index < extra_list->size()) {
		alias = extra_list->at(index)->ToString();
	} else {
		if (!expr.alias.empty()) {
			alias = expr.alias;
		}
	}
	return make_uniq<BoundColumnRefExpression>(std::move(alias), LogicalType::INVALID,
	                                           ColumnBinding(projection_index, index));
}

unique_ptr<Expression> OrderBinder::CreateExtraReference(unique_ptr<ParsedExpression> expr) {
	if (!extra_list) {
		throw InternalException("CreateExtraReference called without extra_list");
	}
	auto result = CreateProjectionReference(*expr, extra_list->size());
	extra_list->push_back(std::move(expr));
	return result;
}

unique_ptr<Expression> OrderBinder::BindConstant(ParsedExpression &expr, const Value &val) {
	// ORDER BY a constant
	if (!val.type().IsIntegral()) {
		// non-integral expression, we just leave the constant here.
		// ORDER BY <constant> has no effect
		// CONTROVERSIAL: maybe we should throw an error
		return nullptr;
	}
	// INTEGER constant: we use the integer as an index into the select list (e.g. ORDER BY 1)
	auto index = (idx_t)val.GetValue<int64_t>();
	if (index < 1 || index > max_count) {
		throw BinderException("ORDER term out of range - should be between 1 and %lld", (idx_t)max_count);
	}
	return CreateProjectionReference(expr, index - 1);
}

unique_ptr<Expression> OrderBinder::Bind(unique_ptr<ParsedExpression> expr) {
	// in the ORDER BY clause we do not bind children
	// we bind ONLY to the select list
	// if there is no matching entry in the SELECT list already, we add the expression to the SELECT list and refer the
	// new expression the new entry will then be bound later during the binding of the SELECT list we also don't do type
	// resolution here: this only happens after the SELECT list has been bound
	switch (expr->expression_class) {
	case ExpressionClass::CONSTANT: {
		// ORDER BY constant
		// is the ORDER BY expression a constant integer? (e.g. ORDER BY 1)
		auto &constant = expr->Cast<ConstantExpression>();
		return BindConstant(*expr, constant.value);
	}
	case ExpressionClass::COLUMN_REF: {
		// COLUMN REF expression
		// check if we can bind it to an alias in the select list
		auto &colref = expr->Cast<ColumnRefExpression>();
		// if there is an explicit table name we can't bind to an alias
		if (colref.IsQualified()) {
			break;
		}
		// check the alias list
		auto entry = alias_map.find(colref.column_names[0]);
		if (entry != alias_map.end()) {
			// it does! point it to that entry
			return CreateProjectionReference(*expr, entry->second);
		}
		break;
	}
	case ExpressionClass::POSITIONAL_REFERENCE: {
		auto &posref = expr->Cast<PositionalReferenceExpression>();
		if (posref.index < 1 || posref.index > max_count) {
			throw BinderException("ORDER term out of range - should be between 1 and %lld", (idx_t)max_count);
		}
		return CreateProjectionReference(*expr, posref.index - 1);
	}
	case ExpressionClass::PARAMETER: {
		throw ParameterNotAllowedException("Parameter not supported in ORDER BY clause");
	}
	default:
		break;
	}
	// general case
	// first bind the table names of this entry
	for (auto &binder : binders) {
		ExpressionBinder::QualifyColumnNames(*binder, expr);
	}
	// first check if the ORDER BY clause already points to an entry in the projection list
	auto entry = projection_map.find(*expr);
	if (entry != projection_map.end()) {
		if (entry->second == DConstants::INVALID_INDEX) {
			throw BinderException("Ambiguous reference to column");
		}
		// there is a matching entry in the projection list
		// just point to that entry
		return CreateProjectionReference(*expr, entry->second);
	}
	if (!extra_list) {
		// no extra list specified: we cannot push an extra ORDER BY clause
		throw BinderException("Could not ORDER BY column \"%s\": add the expression/function to every SELECT, or move "
		                      "the UNION into a FROM clause.",
		                      expr->ToString());
	}
	// otherwise we need to push the ORDER BY entry into the select list
	return CreateExtraReference(std::move(expr));
}

} // namespace duckdb









namespace duckdb {

QualifyBinder::QualifyBinder(Binder &binder, ClientContext &context, BoundSelectNode &node, BoundGroupInformation &info,
                             case_insensitive_map_t<idx_t> &alias_map)
    : BaseSelectBinder(binder, context, node, info), column_alias_binder(node, alias_map) {
	target_type = LogicalType(LogicalTypeId::BOOLEAN);
}

BindResult QualifyBinder::BindColumnRef(unique_ptr<ParsedExpression> &expr_ptr, idx_t depth, bool root_expression) {
	auto &expr = expr_ptr->Cast<ColumnRefExpression>();
	auto result = duckdb::BaseSelectBinder::BindExpression(expr_ptr, depth);
	if (!result.HasError()) {
		return result;
	}

	auto alias_result = column_alias_binder.BindAlias(*this, expr, depth, root_expression);
	if (!alias_result.HasError()) {
		return alias_result;
	}

	return BindResult(StringUtil::Format("Referenced column %s not found in FROM clause and can't find in alias map.",
	                                     expr.ToString()));
}

BindResult QualifyBinder::BindExpression(unique_ptr<ParsedExpression> &expr_ptr, idx_t depth, bool root_expression) {
	auto &expr = *expr_ptr;
	// check if the expression binds to one of the groups
	auto group_index = TryBindGroup(expr, depth);
	if (group_index != DConstants::INVALID_INDEX) {
		return BindGroup(expr, depth, group_index);
	}
	switch (expr.expression_class) {
	case ExpressionClass::WINDOW:
		return BindWindow(expr.Cast<WindowExpression>(), depth);
	case ExpressionClass::COLUMN_REF:
		return BindColumnRef(expr_ptr, depth, root_expression);
	default:
		return duckdb::BaseSelectBinder::BindExpression(expr_ptr, depth);
	}
}

} // namespace duckdb


namespace duckdb {

RelationBinder::RelationBinder(Binder &binder, ClientContext &context, string op)
    : ExpressionBinder(binder, context), op(std::move(op)) {
}

BindResult RelationBinder::BindExpression(unique_ptr<ParsedExpression> &expr_ptr, idx_t depth, bool root_expression) {
	auto &expr = *expr_ptr;
	switch (expr.expression_class) {
	case ExpressionClass::AGGREGATE:
		return BindResult("aggregate functions are not allowed in " + op);
	case ExpressionClass::DEFAULT:
		return BindResult(op + " cannot contain DEFAULT clause");
	case ExpressionClass::SUBQUERY:
		return BindResult("subqueries are not allowed in " + op);
	case ExpressionClass::WINDOW:
		return BindResult("window functions are not allowed in " + op);
	default:
		return ExpressionBinder::BindExpression(expr_ptr, depth);
	}
}

string RelationBinder::UnsupportedAggregateMessage() {
	return "aggregate functions are not allowed in " + op;
}

} // namespace duckdb




namespace duckdb {

ReturningBinder::ReturningBinder(Binder &binder, ClientContext &context) : ExpressionBinder(binder, context) {
}

BindResult ReturningBinder::BindExpression(unique_ptr<ParsedExpression> &expr_ptr, idx_t depth, bool root_expression) {
	auto &expr = *expr_ptr;
	switch (expr.GetExpressionClass()) {
	case ExpressionClass::SUBQUERY:
		return BindResult("SUBQUERY is not supported in returning statements");
	case ExpressionClass::BOUND_SUBQUERY:
		return BindResult("BOUND SUBQUERY is not supported in returning statements");
	case ExpressionClass::COLUMN_REF:
		return ExpressionBinder::BindExpression(expr_ptr, depth);
	default:
		return ExpressionBinder::BindExpression(expr_ptr, depth);
	}
}

} // namespace duckdb


namespace duckdb {

SelectBinder::SelectBinder(Binder &binder, ClientContext &context, BoundSelectNode &node, BoundGroupInformation &info,
                           case_insensitive_map_t<idx_t> alias_map)
    : BaseSelectBinder(binder, context, node, info, std::move(alias_map)) {
}

SelectBinder::SelectBinder(Binder &binder, ClientContext &context, BoundSelectNode &node, BoundGroupInformation &info)
    : SelectBinder(binder, context, node, info, case_insensitive_map_t<idx_t>()) {
}

} // namespace duckdb





namespace duckdb {

TableFunctionBinder::TableFunctionBinder(Binder &binder, ClientContext &context) : ExpressionBinder(binder, context) {
}

BindResult TableFunctionBinder::BindColumnReference(ColumnRefExpression &expr, idx_t depth, bool root_expression) {

	// if this is a lambda parameters, then we temporarily add a BoundLambdaRef,
	// which we capture and remove later
	if (lambda_bindings) {
		auto &colref = expr.Cast<ColumnRefExpression>();
		for (idx_t i = 0; i < lambda_bindings->size(); i++) {
			if (colref.GetColumnName() == (*lambda_bindings)[i].dummy_name) {
				return (*lambda_bindings)[i].Bind(colref, i, depth);
			}
		}
	}
	auto value_function = ExpressionBinder::GetSQLValueFunction(expr.GetColumnName());
	if (value_function) {
		return BindExpression(value_function, depth, root_expression);
	}

	auto result_name = StringUtil::Join(expr.column_names, ".");
	return BindResult(make_uniq<BoundConstantExpression>(Value(result_name)));
}

BindResult TableFunctionBinder::BindExpression(unique_ptr<ParsedExpression> &expr_ptr, idx_t depth,
                                               bool root_expression) {
	auto &expr = *expr_ptr;
	switch (expr.GetExpressionClass()) {
	case ExpressionClass::COLUMN_REF:
		return BindColumnReference(expr.Cast<ColumnRefExpression>(), depth, root_expression);
	case ExpressionClass::SUBQUERY:
		throw BinderException("Table function cannot contain subqueries");
	case ExpressionClass::DEFAULT:
		return BindResult("Table function cannot contain DEFAULT clause");
	case ExpressionClass::WINDOW:
		return BindResult("Table function cannot contain window functions!");
	default:
		return ExpressionBinder::BindExpression(expr_ptr, depth);
	}
}

string TableFunctionBinder::UnsupportedAggregateMessage() {
	return "Table function cannot contain aggregates!";
}

} // namespace duckdb


namespace duckdb {

UpdateBinder::UpdateBinder(Binder &binder, ClientContext &context) : ExpressionBinder(binder, context) {
}

BindResult UpdateBinder::BindExpression(unique_ptr<ParsedExpression> &expr_ptr, idx_t depth, bool root_expression) {
	auto &expr = *expr_ptr;
	switch (expr.expression_class) {
	case ExpressionClass::WINDOW:
		return BindResult("window functions are not allowed in UPDATE");
	default:
		return ExpressionBinder::BindExpression(expr_ptr, depth);
	}
}

string UpdateBinder::UnsupportedAggregateMessage() {
	return "aggregate functions are not allowed in UPDATE";
}

} // namespace duckdb




namespace duckdb {

WhereBinder::WhereBinder(Binder &binder, ClientContext &context, optional_ptr<ColumnAliasBinder> column_alias_binder)
    : ExpressionBinder(binder, context), column_alias_binder(column_alias_binder) {
	target_type = LogicalType(LogicalTypeId::BOOLEAN);
}

BindResult WhereBinder::BindColumnRef(unique_ptr<ParsedExpression> &expr_ptr, idx_t depth, bool root_expression) {
	auto &expr = expr_ptr->Cast<ColumnRefExpression>();
	auto result = ExpressionBinder::BindExpression(expr_ptr, depth);
	if (!result.HasError() || !column_alias_binder) {
		return result;
	}

	BindResult alias_result = column_alias_binder->BindAlias(*this, expr, depth, root_expression);
	// This code path cannot be exercised at thispoint. #1547 might change that.
	if (!alias_result.HasError()) {
		return alias_result;
	}

	return result;
}

BindResult WhereBinder::BindExpression(unique_ptr<ParsedExpression> &expr_ptr, idx_t depth, bool root_expression) {
	auto &expr = *expr_ptr;
	switch (expr.GetExpressionClass()) {
	case ExpressionClass::DEFAULT:
		return BindResult("WHERE clause cannot contain DEFAULT clause");
	case ExpressionClass::WINDOW:
		return BindResult("WHERE clause cannot contain window functions!");
	case ExpressionClass::COLUMN_REF:
		return BindColumnRef(expr_ptr, depth, root_expression);
	default:
		return ExpressionBinder::BindExpression(expr_ptr, depth);
	}
}

string WhereBinder::UnsupportedAggregateMessage() {
	return "WHERE clause cannot contain aggregates!";
}

} // namespace duckdb








namespace duckdb {

ExpressionBinder::ExpressionBinder(Binder &binder, ClientContext &context, bool replace_binder)
    : binder(binder), context(context) {
	if (replace_binder) {
		stored_binder = &binder.GetActiveBinder();
		binder.SetActiveBinder(*this);
	} else {
		binder.PushExpressionBinder(*this);
	}
}

ExpressionBinder::~ExpressionBinder() {
	if (binder.HasActiveBinder()) {
		if (stored_binder) {
			binder.SetActiveBinder(*stored_binder);
		} else {
			binder.PopExpressionBinder();
		}
	}
}

BindResult ExpressionBinder::BindExpression(unique_ptr<ParsedExpression> &expr, idx_t depth, bool root_expression) {
	auto &expr_ref = *expr;
	switch (expr_ref.expression_class) {
	case ExpressionClass::BETWEEN:
		return BindExpression(expr_ref.Cast<BetweenExpression>(), depth);
	case ExpressionClass::CASE:
		return BindExpression(expr_ref.Cast<CaseExpression>(), depth);
	case ExpressionClass::CAST:
		return BindExpression(expr_ref.Cast<CastExpression>(), depth);
	case ExpressionClass::COLLATE:
		return BindExpression(expr_ref.Cast<CollateExpression>(), depth);
	case ExpressionClass::COLUMN_REF:
		return BindExpression(expr_ref.Cast<ColumnRefExpression>(), depth);
	case ExpressionClass::COMPARISON:
		return BindExpression(expr_ref.Cast<ComparisonExpression>(), depth);
	case ExpressionClass::CONJUNCTION:
		return BindExpression(expr_ref.Cast<ConjunctionExpression>(), depth);
	case ExpressionClass::CONSTANT:
		return BindExpression(expr_ref.Cast<ConstantExpression>(), depth);
	case ExpressionClass::FUNCTION: {
		auto &function = expr_ref.Cast<FunctionExpression>();
		if (function.function_name == "unnest" || function.function_name == "unlist") {
			// special case, not in catalog
			return BindUnnest(function, depth, root_expression);
		}
		// binding function expression has extra parameter needed for macro's
		return BindExpression(function, depth, expr);
	}
	case ExpressionClass::LAMBDA:
		return BindExpression(expr_ref.Cast<LambdaExpression>(), depth, false, LogicalTypeId::INVALID);
	case ExpressionClass::OPERATOR:
		return BindExpression(expr_ref.Cast<OperatorExpression>(), depth);
	case ExpressionClass::SUBQUERY:
		return BindExpression(expr_ref.Cast<SubqueryExpression>(), depth);
	case ExpressionClass::PARAMETER:
		return BindExpression(expr_ref.Cast<ParameterExpression>(), depth);
	case ExpressionClass::POSITIONAL_REFERENCE:
		return BindExpression(expr_ref.Cast<PositionalReferenceExpression>(), depth);
	case ExpressionClass::STAR:
		return BindResult(binder.FormatError(expr_ref, "STAR expression is not supported here"));
	default:
		throw NotImplementedException("Unimplemented expression class");
	}
}

bool ExpressionBinder::BindCorrelatedColumns(unique_ptr<ParsedExpression> &expr) {
	// try to bind in one of the outer queries, if the binding error occurred in a subquery
	auto &active_binders = binder.GetActiveBinders();
	// make a copy of the set of binders, so we can restore it later
	auto binders = active_binders;
	active_binders.pop_back();
	idx_t depth = 1;
	bool success = false;
	while (!active_binders.empty()) {
		auto &next_binder = active_binders.back().get();
		ExpressionBinder::QualifyColumnNames(next_binder.binder, expr);
		auto bind_result = next_binder.Bind(expr, depth);
		if (bind_result.empty()) {
			success = true;
			break;
		}
		depth++;
		active_binders.pop_back();
	}
	active_binders = binders;
	return success;
}

void ExpressionBinder::BindChild(unique_ptr<ParsedExpression> &expr, idx_t depth, string &error) {
	if (expr) {
		string bind_error = Bind(expr, depth);
		if (error.empty()) {
			error = bind_error;
		}
	}
}

void ExpressionBinder::ExtractCorrelatedExpressions(Binder &binder, Expression &expr) {
	if (expr.type == ExpressionType::BOUND_COLUMN_REF) {
		auto &bound_colref = expr.Cast<BoundColumnRefExpression>();
		if (bound_colref.depth > 0) {
			binder.AddCorrelatedColumn(CorrelatedColumnInfo(bound_colref));
		}
	}
	ExpressionIterator::EnumerateChildren(expr,
	                                      [&](Expression &child) { ExtractCorrelatedExpressions(binder, child); });
}

bool ExpressionBinder::ContainsType(const LogicalType &type, LogicalTypeId target) {
	if (type.id() == target) {
		return true;
	}
	switch (type.id()) {
	case LogicalTypeId::STRUCT: {
		auto child_count = StructType::GetChildCount(type);
		for (idx_t i = 0; i < child_count; i++) {
			if (ContainsType(StructType::GetChildType(type, i), target)) {
				return true;
			}
		}
		return false;
	}
	case LogicalTypeId::UNION: {
		auto member_count = UnionType::GetMemberCount(type);
		for (idx_t i = 0; i < member_count; i++) {
			if (ContainsType(UnionType::GetMemberType(type, i), target)) {
				return true;
			}
		}
		return false;
	}
	case LogicalTypeId::LIST:
	case LogicalTypeId::MAP:
		return ContainsType(ListType::GetChildType(type), target);
	default:
		return false;
	}
}

LogicalType ExpressionBinder::ExchangeType(const LogicalType &type, LogicalTypeId target, LogicalType new_type) {
	if (type.id() == target) {
		return new_type;
	}
	switch (type.id()) {
	case LogicalTypeId::STRUCT: {
		// we make a copy of the child types of the struct here
		auto child_types = StructType::GetChildTypes(type);
		for (auto &child_type : child_types) {
			child_type.second = ExchangeType(child_type.second, target, new_type);
		}
		return LogicalType::STRUCT(child_types);
	}
	case LogicalTypeId::UNION: {
		auto member_types = UnionType::CopyMemberTypes(type);
		for (auto &member_type : member_types) {
			member_type.second = ExchangeType(member_type.second, target, new_type);
		}
		return LogicalType::UNION(std::move(member_types));
	}
	case LogicalTypeId::LIST:
		return LogicalType::LIST(ExchangeType(ListType::GetChildType(type), target, new_type));
	case LogicalTypeId::MAP:
		return LogicalType::MAP(ExchangeType(ListType::GetChildType(type), target, new_type));
	default:
		return type;
	}
}

bool ExpressionBinder::ContainsNullType(const LogicalType &type) {
	return ContainsType(type, LogicalTypeId::SQLNULL);
}

LogicalType ExpressionBinder::ExchangeNullType(const LogicalType &type) {
	return ExchangeType(type, LogicalTypeId::SQLNULL, LogicalType::INTEGER);
}

unique_ptr<Expression> ExpressionBinder::Bind(unique_ptr<ParsedExpression> &expr, optional_ptr<LogicalType> result_type,
                                              bool root_expression) {
	// bind the main expression
	auto error_msg = Bind(expr, 0, root_expression);
	if (!error_msg.empty()) {
		// failed to bind: try to bind correlated columns in the expression (if any)
		bool success = BindCorrelatedColumns(expr);
		if (!success) {
			throw BinderException(error_msg);
		}
		auto &bound_expr = expr->Cast<BoundExpression>();
		ExtractCorrelatedExpressions(binder, *bound_expr.expr);
	}
	auto &bound_expr = expr->Cast<BoundExpression>();
	unique_ptr<Expression> result = std::move(bound_expr.expr);
	if (target_type.id() != LogicalTypeId::INVALID) {
		// the binder has a specific target type: add a cast to that type
		result = BoundCastExpression::AddCastToType(context, std::move(result), target_type);
	} else {
		if (!binder.can_contain_nulls) {
			// SQL NULL type is only used internally in the binder
			// cast to INTEGER if we encounter it outside of the binder
			if (ContainsNullType(result->return_type)) {
				auto exchanged_type = ExchangeNullType(result->return_type);
				result = BoundCastExpression::AddCastToType(context, std::move(result), exchanged_type);
			}
		}
		if (result->return_type.id() == LogicalTypeId::UNKNOWN) {
			throw ParameterNotResolvedException();
		}
	}
	if (result_type) {
		*result_type = result->return_type;
	}
	return result;
}

string ExpressionBinder::Bind(unique_ptr<ParsedExpression> &expr, idx_t depth, bool root_expression) {
	// bind the node, but only if it has not been bound yet
	auto &expression = *expr;
	auto alias = expression.alias;
	if (expression.GetExpressionClass() == ExpressionClass::BOUND_EXPRESSION) {
		// already bound, don't bind it again
		return string();
	}
	// bind the expression
	BindResult result = BindExpression(expr, depth, root_expression);
	if (result.HasError()) {
		return result.error;
	}
	// successfully bound: replace the node with a BoundExpression
	expr = make_uniq<BoundExpression>(std::move(result.expression));
	auto &be = expr->Cast<BoundExpression>();
	be.alias = alias;
	if (!alias.empty()) {
		be.expr->alias = alias;
	}
	return string();
}

} // namespace duckdb









namespace duckdb {

void ExpressionIterator::EnumerateChildren(const Expression &expr,
                                           const std::function<void(const Expression &child)> &callback) {
	EnumerateChildren((Expression &)expr, [&](unique_ptr<Expression> &child) { callback(*child); });
}

void ExpressionIterator::EnumerateChildren(Expression &expr, const std::function<void(Expression &child)> &callback) {
	EnumerateChildren(expr, [&](unique_ptr<Expression> &child) { callback(*child); });
}

void ExpressionIterator::EnumerateChildren(Expression &expr,
                                           const std::function<void(unique_ptr<Expression> &child)> &callback) {
	switch (expr.expression_class) {
	case ExpressionClass::BOUND_AGGREGATE: {
		auto &aggr_expr = expr.Cast<BoundAggregateExpression>();
		for (auto &child : aggr_expr.children) {
			callback(child);
		}
		if (aggr_expr.filter) {
			callback(aggr_expr.filter);
		}
		if (aggr_expr.order_bys) {
			for (auto &order : aggr_expr.order_bys->orders) {
				callback(order.expression);
			}
		}
		break;
	}
	case ExpressionClass::BOUND_BETWEEN: {
		auto &between_expr = expr.Cast<BoundBetweenExpression>();
		callback(between_expr.input);
		callback(between_expr.lower);
		callback(between_expr.upper);
		break;
	}
	case ExpressionClass::BOUND_CASE: {
		auto &case_expr = expr.Cast<BoundCaseExpression>();
		for (auto &case_check : case_expr.case_checks) {
			callback(case_check.when_expr);
			callback(case_check.then_expr);
		}
		callback(case_expr.else_expr);
		break;
	}
	case ExpressionClass::BOUND_CAST: {
		auto &cast_expr = expr.Cast<BoundCastExpression>();
		callback(cast_expr.child);
		break;
	}
	case ExpressionClass::BOUND_COMPARISON: {
		auto &comp_expr = expr.Cast<BoundComparisonExpression>();
		callback(comp_expr.left);
		callback(comp_expr.right);
		break;
	}
	case ExpressionClass::BOUND_CONJUNCTION: {
		auto &conj_expr = expr.Cast<BoundConjunctionExpression>();
		for (auto &child : conj_expr.children) {
			callback(child);
		}
		break;
	}
	case ExpressionClass::BOUND_FUNCTION: {
		auto &func_expr = expr.Cast<BoundFunctionExpression>();
		for (auto &child : func_expr.children) {
			callback(child);
		}
		break;
	}
	case ExpressionClass::BOUND_OPERATOR: {
		auto &op_expr = expr.Cast<BoundOperatorExpression>();
		for (auto &child : op_expr.children) {
			callback(child);
		}
		break;
	}
	case ExpressionClass::BOUND_SUBQUERY: {
		auto &subquery_expr = expr.Cast<BoundSubqueryExpression>();
		if (subquery_expr.child) {
			callback(subquery_expr.child);
		}
		break;
	}
	case ExpressionClass::BOUND_WINDOW: {
		auto &window_expr = expr.Cast<BoundWindowExpression>();
		for (auto &partition : window_expr.partitions) {
			callback(partition);
		}
		for (auto &order : window_expr.orders) {
			callback(order.expression);
		}
		for (auto &child : window_expr.children) {
			callback(child);
		}
		if (window_expr.filter_expr) {
			callback(window_expr.filter_expr);
		}
		if (window_expr.start_expr) {
			callback(window_expr.start_expr);
		}
		if (window_expr.end_expr) {
			callback(window_expr.end_expr);
		}
		if (window_expr.offset_expr) {
			callback(window_expr.offset_expr);
		}
		if (window_expr.default_expr) {
			callback(window_expr.default_expr);
		}
		break;
	}
	case ExpressionClass::BOUND_UNNEST: {
		auto &unnest_expr = expr.Cast<BoundUnnestExpression>();
		callback(unnest_expr.child);
		break;
	}
	case ExpressionClass::BOUND_COLUMN_REF:
	case ExpressionClass::BOUND_LAMBDA_REF:
	case ExpressionClass::BOUND_CONSTANT:
	case ExpressionClass::BOUND_DEFAULT:
	case ExpressionClass::BOUND_PARAMETER:
	case ExpressionClass::BOUND_REF:
		// these node types have no children
		break;
	default:
		throw InternalException("ExpressionIterator used on unbound expression");
	}
}

void ExpressionIterator::EnumerateExpression(unique_ptr<Expression> &expr,
                                             const std::function<void(Expression &child)> &callback) {
	if (!expr) {
		return;
	}
	callback(*expr);
	ExpressionIterator::EnumerateChildren(*expr,
	                                      [&](unique_ptr<Expression> &child) { EnumerateExpression(child, callback); });
}

void ExpressionIterator::EnumerateTableRefChildren(BoundTableRef &ref,
                                                   const std::function<void(Expression &child)> &callback) {
	switch (ref.type) {
	case TableReferenceType::EXPRESSION_LIST: {
		auto &bound_expr_list = ref.Cast<BoundExpressionListRef>();
		for (auto &expr_list : bound_expr_list.values) {
			for (auto &expr : expr_list) {
				EnumerateExpression(expr, callback);
			}
		}
		break;
	}
	case TableReferenceType::JOIN: {
		auto &bound_join = ref.Cast<BoundJoinRef>();
		if (bound_join.condition) {
			EnumerateExpression(bound_join.condition, callback);
		}
		EnumerateTableRefChildren(*bound_join.left, callback);
		EnumerateTableRefChildren(*bound_join.right, callback);
		break;
	}
	case TableReferenceType::SUBQUERY: {
		auto &bound_subquery = ref.Cast<BoundSubqueryRef>();
		EnumerateQueryNodeChildren(*bound_subquery.subquery, callback);
		break;
	}
	case TableReferenceType::TABLE_FUNCTION:
	case TableReferenceType::EMPTY:
	case TableReferenceType::BASE_TABLE:
	case TableReferenceType::CTE:
		break;
	default:
		throw NotImplementedException("Unimplemented table reference type in ExpressionIterator");
	}
}

void ExpressionIterator::EnumerateQueryNodeChildren(BoundQueryNode &node,
                                                    const std::function<void(Expression &child)> &callback) {
	switch (node.type) {
	case QueryNodeType::SET_OPERATION_NODE: {
		auto &bound_setop = node.Cast<BoundSetOperationNode>();
		EnumerateQueryNodeChildren(*bound_setop.left, callback);
		EnumerateQueryNodeChildren(*bound_setop.right, callback);
		break;
	}
	case QueryNodeType::RECURSIVE_CTE_NODE: {
		auto &cte_node = node.Cast<BoundRecursiveCTENode>();
		EnumerateQueryNodeChildren(*cte_node.left, callback);
		EnumerateQueryNodeChildren(*cte_node.right, callback);
		break;
	}
	case QueryNodeType::SELECT_NODE: {
		auto &bound_select = node.Cast<BoundSelectNode>();
		for (auto &expr : bound_select.select_list) {
			EnumerateExpression(expr, callback);
		}
		EnumerateExpression(bound_select.where_clause, callback);
		for (auto &expr : bound_select.groups.group_expressions) {
			EnumerateExpression(expr, callback);
		}
		EnumerateExpression(bound_select.having, callback);
		for (auto &expr : bound_select.aggregates) {
			EnumerateExpression(expr, callback);
		}
		for (auto &entry : bound_select.unnests) {
			for (auto &expr : entry.second.expressions) {
				EnumerateExpression(expr, callback);
			}
		}
		for (auto &expr : bound_select.windows) {
			EnumerateExpression(expr, callback);
		}
		if (bound_select.from_table) {
			EnumerateTableRefChildren(*bound_select.from_table, callback);
		}
		break;
	}
	default:
		throw NotImplementedException("Unimplemented query node in ExpressionIterator");
	}
	for (idx_t i = 0; i < node.modifiers.size(); i++) {
		switch (node.modifiers[i]->type) {
		case ResultModifierType::DISTINCT_MODIFIER:
			for (auto &expr : ((BoundDistinctModifier &)*node.modifiers[i]).target_distincts) {
				EnumerateExpression(expr, callback);
			}
			break;
		case ResultModifierType::ORDER_MODIFIER:
			for (auto &order : ((BoundOrderModifier &)*node.modifiers[i]).orders) {
				EnumerateExpression(order.expression, callback);
			}
			break;
		default:
			break;
		}
	}
}

} // namespace duckdb



namespace duckdb {

ConjunctionOrFilter::ConjunctionOrFilter() : ConjunctionFilter(TableFilterType::CONJUNCTION_OR) {
}

FilterPropagateResult ConjunctionOrFilter::CheckStatistics(BaseStatistics &stats) {
	// the OR filter is true if ANY of the children is true
	D_ASSERT(!child_filters.empty());
	for (auto &filter : child_filters) {
		auto prune_result = filter->CheckStatistics(stats);
		if (prune_result == FilterPropagateResult::NO_PRUNING_POSSIBLE) {
			return FilterPropagateResult::NO_PRUNING_POSSIBLE;
		} else if (prune_result == FilterPropagateResult::FILTER_ALWAYS_TRUE) {
			return FilterPropagateResult::FILTER_ALWAYS_TRUE;
		}
	}
	return FilterPropagateResult::FILTER_ALWAYS_FALSE;
}

string ConjunctionOrFilter::ToString(const string &column_name) {
	string result;
	for (idx_t i = 0; i < child_filters.size(); i++) {
		if (i > 0) {
			result += " OR ";
		}
		result += child_filters[i]->ToString(column_name);
	}
	return result;
}

bool ConjunctionOrFilter::Equals(const TableFilter &other_p) const {
	if (!ConjunctionFilter::Equals(other_p)) {
		return false;
	}
	auto &other = (ConjunctionOrFilter &)other_p;
	if (other.child_filters.size() != child_filters.size()) {
		return false;
	}
	for (idx_t i = 0; i < other.child_filters.size(); i++) {
		if (!child_filters[i]->Equals(*other.child_filters[i])) {
			return false;
		}
	}
	return true;
}

void ConjunctionOrFilter::Serialize(FieldWriter &writer) const {
	writer.WriteSerializableList(child_filters);
}

unique_ptr<TableFilter> ConjunctionOrFilter::Deserialize(FieldReader &source) {
	auto res = make_uniq<ConjunctionOrFilter>();
	res->child_filters = source.ReadRequiredSerializableList<TableFilter>();
	return std::move(res);
}

ConjunctionAndFilter::ConjunctionAndFilter() : ConjunctionFilter(TableFilterType::CONJUNCTION_AND) {
}

FilterPropagateResult ConjunctionAndFilter::CheckStatistics(BaseStatistics &stats) {
	// the AND filter is true if ALL of the children is true
	D_ASSERT(!child_filters.empty());
	auto result = FilterPropagateResult::FILTER_ALWAYS_TRUE;
	for (auto &filter : child_filters) {
		auto prune_result = filter->CheckStatistics(stats);
		if (prune_result == FilterPropagateResult::FILTER_ALWAYS_FALSE) {
			return FilterPropagateResult::FILTER_ALWAYS_FALSE;
		} else if (prune_result != result) {
			result = FilterPropagateResult::NO_PRUNING_POSSIBLE;
		}
	}
	return result;
}

string ConjunctionAndFilter::ToString(const string &column_name) {
	string result;
	for (idx_t i = 0; i < child_filters.size(); i++) {
		if (i > 0) {
			result += " AND ";
		}
		result += child_filters[i]->ToString(column_name);
	}
	return result;
}

bool ConjunctionAndFilter::Equals(const TableFilter &other_p) const {
	if (!ConjunctionFilter::Equals(other_p)) {
		return false;
	}
	auto &other = (ConjunctionAndFilter &)other_p;
	if (other.child_filters.size() != child_filters.size()) {
		return false;
	}
	for (idx_t i = 0; i < other.child_filters.size(); i++) {
		if (!child_filters[i]->Equals(*other.child_filters[i])) {
			return false;
		}
	}
	return true;
}

void ConjunctionAndFilter::Serialize(FieldWriter &writer) const {
	writer.WriteSerializableList(child_filters);
}

unique_ptr<TableFilter> ConjunctionAndFilter::Deserialize(FieldReader &source) {
	auto res = make_uniq<ConjunctionAndFilter>();
	res->child_filters = source.ReadRequiredSerializableList<TableFilter>();
	return std::move(res);
}

} // namespace duckdb




namespace duckdb {

ConstantFilter::ConstantFilter(ExpressionType comparison_type_p, Value constant_p)
    : TableFilter(TableFilterType::CONSTANT_COMPARISON), comparison_type(comparison_type_p),
      constant(std::move(constant_p)) {
}

FilterPropagateResult ConstantFilter::CheckStatistics(BaseStatistics &stats) {
	D_ASSERT(constant.type().id() == stats.GetType().id());
	switch (constant.type().InternalType()) {
	case PhysicalType::UINT8:
	case PhysicalType::UINT16:
	case PhysicalType::UINT32:
	case PhysicalType::UINT64:
	case PhysicalType::INT8:
	case PhysicalType::INT16:
	case PhysicalType::INT32:
	case PhysicalType::INT64:
	case PhysicalType::INT128:
	case PhysicalType::FLOAT:
	case PhysicalType::DOUBLE:
		return NumericStats::CheckZonemap(stats, comparison_type, constant);
	case PhysicalType::VARCHAR:
		return StringStats::CheckZonemap(stats, comparison_type, StringValue::Get(constant));
	default:
		return FilterPropagateResult::NO_PRUNING_POSSIBLE;
	}
}

string ConstantFilter::ToString(const string &column_name) {
	return column_name + ExpressionTypeToOperator(comparison_type) + constant.ToString();
}

bool ConstantFilter::Equals(const TableFilter &other_p) const {
	if (!TableFilter::Equals(other_p)) {
		return false;
	}
	auto &other = (ConstantFilter &)other_p;
	return other.comparison_type == comparison_type && other.constant == constant;
}

void ConstantFilter::Serialize(FieldWriter &writer) const {
	writer.WriteField(comparison_type);
	writer.WriteSerializable(constant);
}

unique_ptr<TableFilter> ConstantFilter::Deserialize(FieldReader &source) {
	auto comparision_type = source.ReadRequired<ExpressionType>();
	auto constant = source.ReadRequiredSerializable<Value, Value>();
	return make_uniq<ConstantFilter>(comparision_type, constant);
}

} // namespace duckdb



namespace duckdb {

IsNullFilter::IsNullFilter() : TableFilter(TableFilterType::IS_NULL) {
}

FilterPropagateResult IsNullFilter::CheckStatistics(BaseStatistics &stats) {
	if (!stats.CanHaveNull()) {
		// no null values are possible: always false
		return FilterPropagateResult::FILTER_ALWAYS_FALSE;
	}
	if (!stats.CanHaveNoNull()) {
		// no non-null values are possible: always true
		return FilterPropagateResult::FILTER_ALWAYS_TRUE;
	}
	return FilterPropagateResult::NO_PRUNING_POSSIBLE;
}

string IsNullFilter::ToString(const string &column_name) {
	return column_name + "IS NULL";
}

IsNotNullFilter::IsNotNullFilter() : TableFilter(TableFilterType::IS_NOT_NULL) {
}

FilterPropagateResult IsNotNullFilter::CheckStatistics(BaseStatistics &stats) {
	if (!stats.CanHaveNoNull()) {
		// no non-null values are possible: always false
		return FilterPropagateResult::FILTER_ALWAYS_FALSE;
	}
	if (!stats.CanHaveNull()) {
		// no null values are possible: always true
		return FilterPropagateResult::FILTER_ALWAYS_TRUE;
	}
	return FilterPropagateResult::NO_PRUNING_POSSIBLE;
}

string IsNotNullFilter::ToString(const string &column_name) {
	return column_name + " IS NOT NULL";
}

void IsNotNullFilter::Serialize(FieldWriter &writer) const {
}

unique_ptr<TableFilter> IsNotNullFilter::Deserialize(FieldReader &source) {
	return make_uniq<IsNotNullFilter>();
}

void IsNullFilter::Serialize(FieldWriter &writer) const {
}

unique_ptr<TableFilter> IsNullFilter::Deserialize(FieldReader &source) {
	return make_uniq<IsNullFilter>();
}

} // namespace duckdb









namespace duckdb {

unique_ptr<Expression> JoinCondition::CreateExpression(JoinCondition cond) {
	auto bound_comparison =
	    make_uniq<BoundComparisonExpression>(cond.comparison, std::move(cond.left), std::move(cond.right));
	return std::move(bound_comparison);
}

unique_ptr<Expression> JoinCondition::CreateExpression(vector<JoinCondition> conditions) {
	unique_ptr<Expression> result;
	for (auto &cond : conditions) {
		auto expr = CreateExpression(std::move(cond));
		if (!result) {
			result = std::move(expr);
		} else {
			auto conj = make_uniq<BoundConjunctionExpression>(ExpressionType::CONJUNCTION_AND, std::move(expr),
			                                                  std::move(result));
			result = std::move(conj);
		}
	}
	return result;
}

//! Serializes a JoinCondition to a stand-alone binary blob
void JoinCondition::Serialize(Serializer &serializer) const {
	FieldWriter writer(serializer);
	writer.WriteOptional(left);
	writer.WriteOptional(right);
	writer.WriteField<ExpressionType>(comparison);
	writer.Finalize();
}

//! Deserializes a blob back into a JoinCondition
JoinCondition JoinCondition::Deserialize(Deserializer &source, PlanDeserializationState &state) {
	auto result = JoinCondition();

	FieldReader reader(source);
	auto left = reader.ReadOptional<Expression>(nullptr, state);
	auto right = reader.ReadOptional<Expression>(nullptr, state);
	result.left = std::move(left);
	result.right = std::move(right);
	result.comparison = reader.ReadRequired<ExpressionType>();
	reader.Finalize();
	return result;
}

JoinSide JoinSide::CombineJoinSide(JoinSide left, JoinSide right) {
	if (left == JoinSide::NONE) {
		return right;
	}
	if (right == JoinSide::NONE) {
		return left;
	}
	if (left != right) {
		return JoinSide::BOTH;
	}
	return left;
}

JoinSide JoinSide::GetJoinSide(idx_t table_binding, const unordered_set<idx_t> &left_bindings,
                               const unordered_set<idx_t> &right_bindings) {
	if (left_bindings.find(table_binding) != left_bindings.end()) {
		// column references table on left side
		D_ASSERT(right_bindings.find(table_binding) == right_bindings.end());
		return JoinSide::LEFT;
	} else {
		// column references table on right side
		D_ASSERT(right_bindings.find(table_binding) != right_bindings.end());
		return JoinSide::RIGHT;
	}
}

JoinSide JoinSide::GetJoinSide(Expression &expression, const unordered_set<idx_t> &left_bindings,
                               const unordered_set<idx_t> &right_bindings) {
	if (expression.type == ExpressionType::BOUND_COLUMN_REF) {
		auto &colref = expression.Cast<BoundColumnRefExpression>();
		if (colref.depth > 0) {
			throw Exception("Non-inner join on correlated columns not supported");
		}
		return GetJoinSide(colref.binding.table_index, left_bindings, right_bindings);
	}
	D_ASSERT(expression.type != ExpressionType::BOUND_REF);
	if (expression.type == ExpressionType::SUBQUERY) {
		D_ASSERT(expression.GetExpressionClass() == ExpressionClass::BOUND_SUBQUERY);
		auto &subquery = expression.Cast<BoundSubqueryExpression>();
		JoinSide side = JoinSide::NONE;
		if (subquery.child) {
			side = GetJoinSide(*subquery.child, left_bindings, right_bindings);
		}
		// correlated subquery, check the side of each of correlated columns in the subquery
		for (auto &corr : subquery.binder->correlated_columns) {
			if (corr.depth > 1) {
				// correlated column has depth > 1
				// it does not refer to any table in the current set of bindings
				return JoinSide::BOTH;
			}
			auto correlated_side = GetJoinSide(corr.binding.table_index, left_bindings, right_bindings);
			side = CombineJoinSide(side, correlated_side);
		}
		return side;
	}
	JoinSide join_side = JoinSide::NONE;
	ExpressionIterator::EnumerateChildren(expression, [&](Expression &child) {
		auto child_side = GetJoinSide(child, left_bindings, right_bindings);
		join_side = CombineJoinSide(child_side, join_side);
	});
	return join_side;
}

JoinSide JoinSide::GetJoinSide(const unordered_set<idx_t> &bindings, const unordered_set<idx_t> &left_bindings,
                               const unordered_set<idx_t> &right_bindings) {
	JoinSide side = JoinSide::NONE;
	for (auto binding : bindings) {
		side = CombineJoinSide(side, GetJoinSide(binding, left_bindings, right_bindings));
	}
	return side;
}

} // namespace duckdb











namespace duckdb {

const uint64_t PLAN_SERIALIZATION_VERSION = 1;

LogicalOperator::LogicalOperator(LogicalOperatorType type)
    : type(type), estimated_cardinality(0), has_estimated_cardinality(false) {
}

LogicalOperator::LogicalOperator(LogicalOperatorType type, vector<unique_ptr<Expression>> expressions)
    : type(type), expressions(std::move(expressions)), estimated_cardinality(0), has_estimated_cardinality(false) {
}

LogicalOperator::~LogicalOperator() {
}

vector<ColumnBinding> LogicalOperator::GetColumnBindings() {
	return {ColumnBinding(0, 0)};
}

string LogicalOperator::GetName() const {
	return LogicalOperatorToString(type);
}

string LogicalOperator::ParamsToString() const {
	string result;
	for (idx_t i = 0; i < expressions.size(); i++) {
		if (i > 0) {
			result += "\n";
		}
		result += expressions[i]->GetName();
	}
	return result;
}

void LogicalOperator::ResolveOperatorTypes() {

	types.clear();
	// first resolve child types
	for (auto &child : children) {
		child->ResolveOperatorTypes();
	}
	// now resolve the types for this operator
	ResolveTypes();
	D_ASSERT(types.size() == GetColumnBindings().size());
}

vector<ColumnBinding> LogicalOperator::GenerateColumnBindings(idx_t table_idx, idx_t column_count) {
	vector<ColumnBinding> result;
	result.reserve(column_count);
	for (idx_t i = 0; i < column_count; i++) {
		result.emplace_back(table_idx, i);
	}
	return result;
}

vector<LogicalType> LogicalOperator::MapTypes(const vector<LogicalType> &types, const vector<idx_t> &projection_map) {
	if (projection_map.empty()) {
		return types;
	} else {
		vector<LogicalType> result_types;
		result_types.reserve(projection_map.size());
		for (auto index : projection_map) {
			result_types.push_back(types[index]);
		}
		return result_types;
	}
}

vector<ColumnBinding> LogicalOperator::MapBindings(const vector<ColumnBinding> &bindings,
                                                   const vector<idx_t> &projection_map) {
	if (projection_map.empty()) {
		return bindings;
	} else {
		vector<ColumnBinding> result_bindings;
		result_bindings.reserve(projection_map.size());
		for (auto index : projection_map) {
			D_ASSERT(index < bindings.size());
			result_bindings.push_back(bindings[index]);
		}
		return result_bindings;
	}
}

string LogicalOperator::ToString() const {
	TreeRenderer renderer;
	return renderer.ToString(*this);
}

void LogicalOperator::Verify(ClientContext &context) {
#ifdef DEBUG
	// verify expressions
	for (idx_t expr_idx = 0; expr_idx < expressions.size(); expr_idx++) {
		auto str = expressions[expr_idx]->ToString();
		// verify that we can (correctly) copy this expression
		auto copy = expressions[expr_idx]->Copy();
		auto original_hash = expressions[expr_idx]->Hash();
		auto copy_hash = copy->Hash();
		// copy should be identical to original
		D_ASSERT(expressions[expr_idx]->ToString() == copy->ToString());
		D_ASSERT(original_hash == copy_hash);
		D_ASSERT(Expression::Equals(expressions[expr_idx].get(), copy.get()));

		D_ASSERT(!Expression::Equals(expressions[expr_idx].get(), nullptr));
		for (idx_t other_idx = 0; other_idx < expr_idx; other_idx++) {
			// comparison with other expressions
			auto other_hash = expressions[other_idx]->Hash();
			bool expr_equal = Expression::Equals(expressions[expr_idx].get(), expressions[other_idx].get());
			if (original_hash != other_hash) {
				// if the hashes are not equal the expressions should not be equal either
				D_ASSERT(!expr_equal);
			}
		}
		D_ASSERT(!str.empty());

		// verify that serialization + deserialization round-trips correctly
		if (expressions[expr_idx]->HasParameter()) {
			continue;
		}
		BufferedSerializer serializer;
		// We are serializing a query plan
		serializer.is_query_plan = true;
		try {
			expressions[expr_idx]->Serialize(serializer);
		} catch (NotImplementedException &ex) {
			// ignore for now (FIXME)
			return;
		}

		auto data = serializer.GetData();
		auto deserializer = BufferedContextDeserializer(context, data.data.get(), data.size);

		PlanDeserializationState state(context);
		auto deserialized_expression = Expression::Deserialize(deserializer, state);
		// FIXME: expressions might not be equal yet because of statistics propagation
		continue;
		D_ASSERT(Expression::Equals(expressions[expr_idx].get(), deserialized_expression.get()));
		D_ASSERT(expressions[expr_idx]->Hash() == deserialized_expression->Hash());
	}
	D_ASSERT(!ToString().empty());
	for (auto &child : children) {
		child->Verify(context);
	}
#endif
}

void LogicalOperator::AddChild(unique_ptr<LogicalOperator> child) {
	D_ASSERT(child);
	children.push_back(std::move(child));
}

idx_t LogicalOperator::EstimateCardinality(ClientContext &context) {
	// simple estimator, just take the max of the children
	if (has_estimated_cardinality) {
		return estimated_cardinality;
	}
	idx_t max_cardinality = 0;
	for (auto &child : children) {
		max_cardinality = MaxValue(child->EstimateCardinality(context), max_cardinality);
	}
	has_estimated_cardinality = true;
	estimated_cardinality = max_cardinality;
	return estimated_cardinality;
}

void LogicalOperator::Print() {
	Printer::Print(ToString());
}

void LogicalOperator::Serialize(Serializer &serializer) const {
	FieldWriter writer(serializer);
	writer.WriteField<LogicalOperatorType>(type);
	writer.WriteSerializableList(children);

	Serialize(writer);
	writer.Finalize();
}

unique_ptr<LogicalOperator> LogicalOperator::Deserialize(Deserializer &deserializer, PlanDeserializationState &gstate) {
	unique_ptr<LogicalOperator> result;

	FieldReader reader(deserializer);
	auto type = reader.ReadRequired<LogicalOperatorType>();
	auto children = reader.ReadRequiredSerializableList<LogicalOperator>(gstate);

	LogicalDeserializationState state(gstate, type, children);
	switch (type) {
	case LogicalOperatorType::LOGICAL_PROJECTION:
		result = LogicalProjection::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_FILTER:
		result = LogicalFilter::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
		result = LogicalAggregate::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_WINDOW:
		result = LogicalWindow::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_UNNEST:
		result = LogicalUnnest::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_LIMIT:
		result = LogicalLimit::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_ORDER_BY:
		result = LogicalOrder::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_TOP_N:
		result = LogicalTopN::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_COPY_TO_FILE:
		result = LogicalCopyToFile::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_DISTINCT:
		result = LogicalDistinct::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_SAMPLE:
		result = LogicalSample::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_LIMIT_PERCENT:
		result = LogicalLimitPercent::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_GET:
		result = LogicalGet::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_CHUNK_GET:
		result = LogicalColumnDataGet::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_DELIM_GET:
		result = LogicalDelimGet::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_EXPRESSION_GET:
		result = LogicalExpressionGet::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_DUMMY_SCAN:
		result = LogicalDummyScan::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_EMPTY_RESULT:
		result = LogicalEmptyResult::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_CTE_REF:
		result = LogicalCTERef::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_JOIN:
		throw InternalException("LogicalJoin deserialize not supported");
	case LogicalOperatorType::LOGICAL_DELIM_JOIN:
		result = LogicalDelimJoin::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_ASOF_JOIN:
		result = LogicalAsOfJoin::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
		result = LogicalComparisonJoin::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_ANY_JOIN:
		result = LogicalAnyJoin::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_CROSS_PRODUCT:
		result = LogicalCrossProduct::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_POSITIONAL_JOIN:
		result = LogicalPositionalJoin::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_UNION:
		result = LogicalSetOperation::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_EXCEPT:
		result = LogicalSetOperation::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_INTERSECT:
		result = LogicalSetOperation::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_RECURSIVE_CTE:
		result = LogicalRecursiveCTE::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_INSERT:
		result = LogicalInsert::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_DELETE:
		result = LogicalDelete::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_UPDATE:
		result = LogicalUpdate::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_CREATE_TABLE:
		result = LogicalCreateTable::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_CREATE_INDEX:
		result = LogicalCreateIndex::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_CREATE_SEQUENCE:
		result = LogicalCreate::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_CREATE_VIEW:
		result = LogicalCreate::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_CREATE_SCHEMA:
		result = LogicalCreate::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_CREATE_MACRO:
		result = LogicalCreate::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_PRAGMA:
		result = LogicalPragma::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_CREATE_TYPE:
		result = LogicalCreate::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_EXPLAIN:
		result = LogicalExplain::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_SHOW:
		result = LogicalShow::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_PREPARE:
		result = LogicalPrepare::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_EXECUTE:
		result = LogicalExecute::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_EXPORT:
		result = LogicalExport::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_SET:
		result = LogicalSet::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_RESET:
		result = LogicalReset::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_ALTER:
	case LogicalOperatorType::LOGICAL_VACUUM:
	case LogicalOperatorType::LOGICAL_LOAD:
	case LogicalOperatorType::LOGICAL_ATTACH:
	case LogicalOperatorType::LOGICAL_TRANSACTION:
	case LogicalOperatorType::LOGICAL_DROP:
	case LogicalOperatorType::LOGICAL_DETACH:
		result = LogicalSimple::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR:
		result = LogicalExtensionOperator::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_PIVOT:
		result = LogicalPivot::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_INVALID:
		/* no default here to trigger a warning if we forget to implement deserialize for a new operator */
		throw SerializationException("Invalid type for operator deserialization");
	}

	reader.Finalize();
	result->children = std::move(children);

	return result;
}

vector<idx_t> LogicalOperator::GetTableIndex() const {
	return vector<idx_t> {};
}

unique_ptr<LogicalOperator> LogicalOperator::Copy(ClientContext &context) const {
	BufferedSerializer logical_op_serializer;
	try {
		this->Serialize(logical_op_serializer);
	} catch (NotImplementedException &ex) {
		throw NotImplementedException("Logical Operator Copy requires the logical operator and all of its children to "
		                              "be serializable: " +
		                              std::string(ex.what()));
	}
	auto data = logical_op_serializer.GetData();
	auto logical_op_deserializer = BufferedContextDeserializer(context, data.data.get(), data.size);
	PlanDeserializationState state(context);
	auto op_copy = LogicalOperator::Deserialize(logical_op_deserializer, state);
	return op_copy;
}

} // namespace duckdb






namespace duckdb {

void LogicalOperatorVisitor::VisitOperator(LogicalOperator &op) {
	VisitOperatorChildren(op);
	VisitOperatorExpressions(op);
}

void LogicalOperatorVisitor::VisitOperatorChildren(LogicalOperator &op) {
	for (auto &child : op.children) {
		VisitOperator(*child);
	}
}

void LogicalOperatorVisitor::EnumerateExpressions(LogicalOperator &op,
                                                  const std::function<void(unique_ptr<Expression> *child)> &callback) {
	switch (op.type) {
	case LogicalOperatorType::LOGICAL_EXPRESSION_GET: {
		auto &get = op.Cast<LogicalExpressionGet>();
		for (auto &expr_list : get.expressions) {
			for (auto &expr : expr_list) {
				callback(&expr);
			}
		}
		break;
	}
	case LogicalOperatorType::LOGICAL_ORDER_BY: {
		auto &order = op.Cast<LogicalOrder>();
		for (auto &node : order.orders) {
			callback(&node.expression);
		}
		break;
	}
	case LogicalOperatorType::LOGICAL_TOP_N: {
		auto &order = op.Cast<LogicalTopN>();
		for (auto &node : order.orders) {
			callback(&node.expression);
		}
		break;
	}
	case LogicalOperatorType::LOGICAL_DISTINCT: {
		auto &distinct = op.Cast<LogicalDistinct>();
		for (auto &target : distinct.distinct_targets) {
			callback(&target);
		}
		if (distinct.order_by) {
			for (auto &order : distinct.order_by->orders) {
				callback(&order.expression);
			}
		}
		break;
	}
	case LogicalOperatorType::LOGICAL_INSERT: {
		auto &insert = op.Cast<LogicalInsert>();
		if (insert.on_conflict_condition) {
			callback(&insert.on_conflict_condition);
		}
		if (insert.do_update_condition) {
			callback(&insert.do_update_condition);
		}
		break;
	}
	case LogicalOperatorType::LOGICAL_ASOF_JOIN:
	case LogicalOperatorType::LOGICAL_DELIM_JOIN:
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN: {
		if (op.type == LogicalOperatorType::LOGICAL_DELIM_JOIN) {
			auto &delim_join = op.Cast<LogicalDelimJoin>();
			for (auto &expr : delim_join.duplicate_eliminated_columns) {
				callback(&expr);
			}
		}
		auto &join = op.Cast<LogicalComparisonJoin>();
		for (auto &cond : join.conditions) {
			callback(&cond.left);
			callback(&cond.right);
		}
		break;
	}
	case LogicalOperatorType::LOGICAL_ANY_JOIN: {
		auto &join = op.Cast<LogicalAnyJoin>();
		callback(&join.condition);
		break;
	}
	case LogicalOperatorType::LOGICAL_LIMIT: {
		auto &limit = op.Cast<LogicalLimit>();
		if (limit.limit) {
			callback(&limit.limit);
		}
		if (limit.offset) {
			callback(&limit.offset);
		}
		break;
	}
	case LogicalOperatorType::LOGICAL_LIMIT_PERCENT: {
		auto &limit = (LogicalLimitPercent &)op;
		if (limit.limit) {
			callback(&limit.limit);
		}
		if (limit.offset) {
			callback(&limit.offset);
		}
		break;
	}
	case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY: {
		auto &aggr = op.Cast<LogicalAggregate>();
		for (auto &group : aggr.groups) {
			callback(&group);
		}
		break;
	}
	default:
		break;
	}
	for (auto &expression : op.expressions) {
		callback(&expression);
	}
}

void LogicalOperatorVisitor::VisitOperatorExpressions(LogicalOperator &op) {
	LogicalOperatorVisitor::EnumerateExpressions(op, [&](unique_ptr<Expression> *child) { VisitExpression(child); });
}

void LogicalOperatorVisitor::VisitExpression(unique_ptr<Expression> *expression) {
	auto &expr = **expression;
	unique_ptr<Expression> result;
	switch (expr.GetExpressionClass()) {
	case ExpressionClass::BOUND_AGGREGATE:
		result = VisitReplace(expr.Cast<BoundAggregateExpression>(), expression);
		break;
	case ExpressionClass::BOUND_BETWEEN:
		result = VisitReplace(expr.Cast<BoundBetweenExpression>(), expression);
		break;
	case ExpressionClass::BOUND_CASE:
		result = VisitReplace(expr.Cast<BoundCaseExpression>(), expression);
		break;
	case ExpressionClass::BOUND_CAST:
		result = VisitReplace(expr.Cast<BoundCastExpression>(), expression);
		break;
	case ExpressionClass::BOUND_COLUMN_REF:
		result = VisitReplace(expr.Cast<BoundColumnRefExpression>(), expression);
		break;
	case ExpressionClass::BOUND_COMPARISON:
		result = VisitReplace(expr.Cast<BoundComparisonExpression>(), expression);
		break;
	case ExpressionClass::BOUND_CONJUNCTION:
		result = VisitReplace(expr.Cast<BoundConjunctionExpression>(), expression);
		break;
	case ExpressionClass::BOUND_CONSTANT:
		result = VisitReplace(expr.Cast<BoundConstantExpression>(), expression);
		break;
	case ExpressionClass::BOUND_FUNCTION:
		result = VisitReplace(expr.Cast<BoundFunctionExpression>(), expression);
		break;
	case ExpressionClass::BOUND_SUBQUERY:
		result = VisitReplace(expr.Cast<BoundSubqueryExpression>(), expression);
		break;
	case ExpressionClass::BOUND_OPERATOR:
		result = VisitReplace(expr.Cast<BoundOperatorExpression>(), expression);
		break;
	case ExpressionClass::BOUND_PARAMETER:
		result = VisitReplace(expr.Cast<BoundParameterExpression>(), expression);
		break;
	case ExpressionClass::BOUND_REF:
		result = VisitReplace(expr.Cast<BoundReferenceExpression>(), expression);
		break;
	case ExpressionClass::BOUND_DEFAULT:
		result = VisitReplace(expr.Cast<BoundDefaultExpression>(), expression);
		break;
	case ExpressionClass::BOUND_WINDOW:
		result = VisitReplace(expr.Cast<BoundWindowExpression>(), expression);
		break;
	case ExpressionClass::BOUND_UNNEST:
		result = VisitReplace(expr.Cast<BoundUnnestExpression>(), expression);
		break;
	default:
		throw InternalException("Unrecognized expression type in logical operator visitor");
	}
	if (result) {
		*expression = std::move(result);
	} else {
		// visit the children of this node
		VisitExpressionChildren(expr);
	}
}

void LogicalOperatorVisitor::VisitExpressionChildren(Expression &expr) {
	ExpressionIterator::EnumerateChildren(expr, [&](unique_ptr<Expression> &expr) { VisitExpression(&expr); });
}

// these are all default methods that can be overriden
// we don't care about coverage here
// LCOV_EXCL_START
unique_ptr<Expression> LogicalOperatorVisitor::VisitReplace(BoundAggregateExpression &expr,
                                                            unique_ptr<Expression> *expr_ptr) {
	return nullptr;
}

unique_ptr<Expression> LogicalOperatorVisitor::VisitReplace(BoundBetweenExpression &expr,
                                                            unique_ptr<Expression> *expr_ptr) {
	return nullptr;
}

unique_ptr<Expression> LogicalOperatorVisitor::VisitReplace(BoundCaseExpression &expr,
                                                            unique_ptr<Expression> *expr_ptr) {
	return nullptr;
}

unique_ptr<Expression> LogicalOperatorVisitor::VisitReplace(BoundCastExpression &expr,
                                                            unique_ptr<Expression> *expr_ptr) {
	return nullptr;
}

unique_ptr<Expression> LogicalOperatorVisitor::VisitReplace(BoundColumnRefExpression &expr,
                                                            unique_ptr<Expression> *expr_ptr) {
	return nullptr;
}

unique_ptr<Expression> LogicalOperatorVisitor::VisitReplace(BoundComparisonExpression &expr,
                                                            unique_ptr<Expression> *expr_ptr) {
	return nullptr;
}

unique_ptr<Expression> LogicalOperatorVisitor::VisitReplace(BoundConjunctionExpression &expr,
                                                            unique_ptr<Expression> *expr_ptr) {
	return nullptr;
}

unique_ptr<Expression> LogicalOperatorVisitor::VisitReplace(BoundConstantExpression &expr,
                                                            unique_ptr<Expression> *expr_ptr) {
	return nullptr;
}

unique_ptr<Expression> LogicalOperatorVisitor::VisitReplace(BoundDefaultExpression &expr,
                                                            unique_ptr<Expression> *expr_ptr) {
	return nullptr;
}

unique_ptr<Expression> LogicalOperatorVisitor::VisitReplace(BoundFunctionExpression &expr,
                                                            unique_ptr<Expression> *expr_ptr) {
	return nullptr;
}

unique_ptr<Expression> LogicalOperatorVisitor::VisitReplace(BoundOperatorExpression &expr,
                                                            unique_ptr<Expression> *expr_ptr) {
	return nullptr;
}

unique_ptr<Expression> LogicalOperatorVisitor::VisitReplace(BoundParameterExpression &expr,
                                                            unique_ptr<Expression> *expr_ptr) {
	return nullptr;
}

unique_ptr<Expression> LogicalOperatorVisitor::VisitReplace(BoundReferenceExpression &expr,
                                                            unique_ptr<Expression> *expr_ptr) {
	return nullptr;
}

unique_ptr<Expression> LogicalOperatorVisitor::VisitReplace(BoundSubqueryExpression &expr,
                                                            unique_ptr<Expression> *expr_ptr) {
	return nullptr;
}

unique_ptr<Expression> LogicalOperatorVisitor::VisitReplace(BoundWindowExpression &expr,
                                                            unique_ptr<Expression> *expr_ptr) {
	return nullptr;
}

unique_ptr<Expression> LogicalOperatorVisitor::VisitReplace(BoundUnnestExpression &expr,
                                                            unique_ptr<Expression> *expr_ptr) {
	return nullptr;
}

// LCOV_EXCL_STOP

} // namespace duckdb






namespace duckdb {

LogicalAggregate::LogicalAggregate(idx_t group_index, idx_t aggregate_index, vector<unique_ptr<Expression>> select_list)
    : LogicalOperator(LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY, std::move(select_list)),
      group_index(group_index), aggregate_index(aggregate_index), groupings_index(DConstants::INVALID_INDEX) {
}

void LogicalAggregate::ResolveTypes() {
	D_ASSERT(groupings_index != DConstants::INVALID_INDEX || grouping_functions.empty());
	for (auto &expr : groups) {
		types.push_back(expr->return_type);
	}
	// get the chunk types from the projection list
	for (auto &expr : expressions) {
		types.push_back(expr->return_type);
	}
	for (idx_t i = 0; i < grouping_functions.size(); i++) {
		types.emplace_back(LogicalType::BIGINT);
	}
}

vector<ColumnBinding> LogicalAggregate::GetColumnBindings() {
	D_ASSERT(groupings_index != DConstants::INVALID_INDEX || grouping_functions.empty());
	vector<ColumnBinding> result;
	result.reserve(groups.size() + expressions.size() + grouping_functions.size());
	for (idx_t i = 0; i < groups.size(); i++) {
		result.emplace_back(group_index, i);
	}
	for (idx_t i = 0; i < expressions.size(); i++) {
		result.emplace_back(aggregate_index, i);
	}
	for (idx_t i = 0; i < grouping_functions.size(); i++) {
		result.emplace_back(groupings_index, i);
	}
	return result;
}

string LogicalAggregate::ParamsToString() const {
	string result;
	for (idx_t i = 0; i < groups.size(); i++) {
		if (i > 0) {
			result += "\n";
		}
		result += groups[i]->GetName();
	}
	for (idx_t i = 0; i < expressions.size(); i++) {
		if (i > 0 || !groups.empty()) {
			result += "\n";
		}
		result += expressions[i]->GetName();
	}
	return result;
}

void LogicalAggregate::Serialize(FieldWriter &writer) const {
	writer.WriteSerializableList(expressions);

	writer.WriteField(group_index);
	writer.WriteField(aggregate_index);
	writer.WriteField(groupings_index);
	writer.WriteSerializableList(groups);
	writer.WriteField<idx_t>(grouping_sets.size());
	for (auto &entry : grouping_sets) {
		writer.WriteList<idx_t>(entry);
	}
	writer.WriteField<idx_t>(grouping_functions.size());
	for (auto &entry : grouping_functions) {
		writer.WriteList<idx_t>(entry);
	}

	// TODO statistics
}

unique_ptr<LogicalOperator> LogicalAggregate::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto expressions = reader.ReadRequiredSerializableList<Expression>(state.gstate);

	auto group_index = reader.ReadRequired<idx_t>();
	auto aggregate_index = reader.ReadRequired<idx_t>();
	auto groupings_index = reader.ReadRequired<idx_t>();
	auto groups = reader.ReadRequiredSerializableList<Expression>(state.gstate);
	auto grouping_sets_size = reader.ReadRequired<idx_t>();
	vector<GroupingSet> grouping_sets;
	for (idx_t i = 0; i < grouping_sets_size; i++) {
		grouping_sets.push_back(reader.ReadRequiredSet<idx_t>());
	}
	vector<unsafe_vector<idx_t>> grouping_functions;
	auto grouping_functions_size = reader.ReadRequired<idx_t>();
	for (idx_t i = 0; i < grouping_functions_size; i++) {
		grouping_functions.push_back(reader.ReadRequiredList<idx_t>());
	}
	auto result = make_uniq<LogicalAggregate>(group_index, aggregate_index, std::move(expressions));
	result->groupings_index = groupings_index;
	result->groups = std::move(groups);
	result->grouping_functions = std::move(grouping_functions);
	result->grouping_sets = std::move(grouping_sets);

	return std::move(result);
}

idx_t LogicalAggregate::EstimateCardinality(ClientContext &context) {
	if (groups.empty()) {
		// ungrouped aggregate
		return 1;
	}
	return LogicalOperator::EstimateCardinality(context);
}

vector<idx_t> LogicalAggregate::GetTableIndex() const {
	vector<idx_t> result {group_index, aggregate_index};
	if (groupings_index != DConstants::INVALID_INDEX) {
		result.push_back(groupings_index);
	}
	return result;
}

string LogicalAggregate::GetName() const {
#ifdef DEBUG
	if (DBConfigOptions::debug_print_bindings) {
		return LogicalOperator::GetName() +
		       StringUtil::Format(" #%llu, #%llu, #%llu", group_index, aggregate_index, groupings_index);
	}
#endif
	return LogicalOperator::GetName();
}

} // namespace duckdb



namespace duckdb {

LogicalAnyJoin::LogicalAnyJoin(JoinType type) : LogicalJoin(type, LogicalOperatorType::LOGICAL_ANY_JOIN) {
}

string LogicalAnyJoin::ParamsToString() const {
	return condition->ToString();
}

void LogicalAnyJoin::Serialize(FieldWriter &writer) const {
	writer.WriteField(join_type);
	writer.WriteOptional(condition);
}

unique_ptr<LogicalOperator> LogicalAnyJoin::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto join_type = reader.ReadRequired<JoinType>();
	auto condition = reader.ReadOptional<Expression>(nullptr, state.gstate);
	auto result = make_uniq<LogicalAnyJoin>(join_type);
	result->condition = std::move(condition);
	return std::move(result);
}

} // namespace duckdb


namespace duckdb {

LogicalAsOfJoin::LogicalAsOfJoin(JoinType type) : LogicalComparisonJoin(type, LogicalOperatorType::LOGICAL_ASOF_JOIN) {
}

unique_ptr<LogicalOperator> LogicalAsOfJoin::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto result = make_uniq<LogicalAsOfJoin>(JoinType::INVALID);
	LogicalComparisonJoin::Deserialize(*result, state, reader);
	return std::move(result);
}

} // namespace duckdb






namespace duckdb {

LogicalColumnDataGet::LogicalColumnDataGet(idx_t table_index, vector<LogicalType> types,
                                           unique_ptr<ColumnDataCollection> collection)
    : LogicalOperator(LogicalOperatorType::LOGICAL_CHUNK_GET), table_index(table_index),
      collection(std::move(collection)) {
	D_ASSERT(types.size() > 0);
	chunk_types = std::move(types);
}

vector<ColumnBinding> LogicalColumnDataGet::GetColumnBindings() {
	return GenerateColumnBindings(table_index, chunk_types.size());
}

void LogicalColumnDataGet::Serialize(FieldWriter &writer) const {
	writer.WriteField(table_index);
	writer.WriteRegularSerializableList(chunk_types);
	writer.WriteField(collection->ChunkCount());
	for (auto &chunk : collection->Chunks()) {
		chunk.Serialize(writer.GetSerializer());
	}
}

unique_ptr<LogicalOperator> LogicalColumnDataGet::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto table_index = reader.ReadRequired<idx_t>();
	auto chunk_types = reader.ReadRequiredSerializableList<LogicalType, LogicalType>();
	auto chunk_count = reader.ReadRequired<idx_t>();
	auto collection = make_uniq<ColumnDataCollection>(state.gstate.context, chunk_types);
	for (idx_t i = 0; i < chunk_count; i++) {
		DataChunk chunk;
		chunk.Deserialize(reader.GetSource());
		collection->Append(chunk);
	}
	return make_uniq<LogicalColumnDataGet>(table_index, std::move(chunk_types), std::move(collection));
}

vector<idx_t> LogicalColumnDataGet::GetTableIndex() const {
	return vector<idx_t> {table_index};
}

string LogicalColumnDataGet::GetName() const {
#ifdef DEBUG
	if (DBConfigOptions::debug_print_bindings) {
		return LogicalOperator::GetName() + StringUtil::Format(" #%llu", table_index);
	}
#endif
	return LogicalOperator::GetName();
}

} // namespace duckdb





namespace duckdb {

LogicalComparisonJoin::LogicalComparisonJoin(JoinType join_type, LogicalOperatorType logical_type)
    : LogicalJoin(join_type, logical_type) {
}

string LogicalComparisonJoin::ParamsToString() const {
	string result = EnumUtil::ToChars(join_type);
	for (auto &condition : conditions) {
		result += "\n";
		auto expr =
		    make_uniq<BoundComparisonExpression>(condition.comparison, condition.left->Copy(), condition.right->Copy());
		result += expr->ToString();
	}

	return result;
}

void LogicalComparisonJoin::Serialize(FieldWriter &writer) const {
	LogicalJoin::Serialize(writer);
	writer.WriteRegularSerializableList(conditions);
	writer.WriteRegularSerializableList(delim_types);
}

void LogicalComparisonJoin::Deserialize(LogicalComparisonJoin &comparison_join, LogicalDeserializationState &state,
                                        FieldReader &reader) {
	LogicalJoin::Deserialize(comparison_join, state, reader);
	comparison_join.conditions = reader.ReadRequiredSerializableList<JoinCondition, JoinCondition>(state.gstate);
	comparison_join.delim_types = reader.ReadRequiredSerializableList<LogicalType, LogicalType>();
}

unique_ptr<LogicalOperator> LogicalComparisonJoin::Deserialize(LogicalDeserializationState &state,
                                                               FieldReader &reader) {
	auto result = make_uniq<LogicalComparisonJoin>(JoinType::INVALID, state.type);
	LogicalComparisonJoin::Deserialize(*result, state, reader);
	return std::move(result);
}

} // namespace duckdb






namespace duckdb {

// Warning: some fields here were added while this code appears untested
// -> copy test in test/api/test_plan_serialization.cpp was commented out as WIP
void LogicalCopyToFile::Serialize(FieldWriter &writer) const {
	writer.WriteString(file_path);
	writer.WriteField(use_tmp_file);
	writer.WriteField(overwrite_or_ignore);
	writer.WriteField(per_thread_output);
	writer.WriteList<idx_t>(partition_columns);

	D_ASSERT(!function.name.empty());
	writer.WriteString(function.name);

	writer.WriteField(bind_data != nullptr);
	if (bind_data && !function.serialize) {
		throw InvalidInputException("Can't serialize copy function %s", function.name);
	}

	function.serialize(writer, *bind_data, function);
}

unique_ptr<LogicalOperator> LogicalCopyToFile::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto file_path = reader.ReadRequired<string>();
	auto use_tmp_file = reader.ReadRequired<bool>();
	auto overwrite_or_ignore = reader.ReadRequired<bool>();
	auto per_thread_output = reader.ReadRequired<bool>();
	auto partition_columns = reader.ReadRequiredList<idx_t>();

	auto copy_func_name = reader.ReadRequired<string>();

	auto has_bind_data = reader.ReadRequired<bool>();

	auto &context = state.gstate.context;
	auto &copy_func_catalog_entry =
	    Catalog::GetEntry<CopyFunctionCatalogEntry>(context, INVALID_CATALOG, DEFAULT_SCHEMA, copy_func_name);
	auto &copy_func = copy_func_catalog_entry.function;

	unique_ptr<FunctionData> bind_data;
	if (has_bind_data) {
		if (!copy_func.deserialize) {
			throw SerializationException("Have bind info but no deserialization function for %s", copy_func.name);
		}
		bind_data = copy_func.deserialize(context, reader, copy_func);
	}

	auto result = make_uniq<LogicalCopyToFile>(copy_func, std::move(bind_data));
	result->file_path = file_path;
	result->use_tmp_file = use_tmp_file;
	result->overwrite_or_ignore = overwrite_or_ignore;
	result->per_thread_output = per_thread_output;
	result->partition_columns = std::move(partition_columns);
	return std::move(result);
}

idx_t LogicalCopyToFile::EstimateCardinality(ClientContext &context) {
	return 1;
}

} // namespace duckdb


namespace duckdb {

void LogicalCreate::Serialize(FieldWriter &writer) const {
	info->Serialize(writer.GetSerializer());
}

unique_ptr<LogicalOperator> LogicalCreate::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto &context = state.gstate.context;
	auto info = CreateInfo::Deserialize(reader.GetSource());

	auto schema_catalog_entry = Catalog::GetSchema(context, info->catalog, info->schema, OnEntryNotFound::RETURN_NULL);
	return make_uniq<LogicalCreate>(state.type, std::move(info), schema_catalog_entry);
}

idx_t LogicalCreate::EstimateCardinality(ClientContext &context) {
	return 1;
}

} // namespace duckdb






namespace duckdb {

void LogicalCreateIndex::Serialize(FieldWriter &writer) const {
	writer.WriteOptional(info);
	writer.WriteString(table.catalog.GetName());
	writer.WriteString(table.schema.name);
	writer.WriteString(table.name);
	FunctionSerializer::SerializeBase<TableFunction>(writer, function, bind_data.get());
	writer.WriteSerializableList(unbound_expressions);
}

unique_ptr<LogicalOperator> LogicalCreateIndex::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto &context = state.gstate.context;

	auto info = reader.ReadOptional<CreateInfo>(nullptr);
	auto catalog = reader.ReadRequired<string>();
	auto schema = reader.ReadRequired<string>();
	auto table_name = reader.ReadRequired<string>();
	unique_ptr<FunctionData> bind_data;
	bool has_deserialize;
	auto function = FunctionSerializer::DeserializeBaseInternal<TableFunction, TableFunctionCatalogEntry>(
	    reader, state.gstate, CatalogType::TABLE_FUNCTION_ENTRY, bind_data, has_deserialize);
	auto unbound_expressions = reader.ReadRequiredSerializableList<Expression>(state.gstate);
	if (info->type != CatalogType::INDEX_ENTRY) {
		throw InternalException("Unexpected type: '%s', expected '%s'", CatalogTypeToString(info->type),
		                        CatalogTypeToString(CatalogType::INDEX_ENTRY));
	}
	auto index_info = unique_ptr_cast<CreateInfo, CreateIndexInfo>(std::move(info));
	auto &table = Catalog::GetEntry<TableCatalogEntry>(context, catalog, schema, table_name);
	return make_uniq<LogicalCreateIndex>(std::move(bind_data), std::move(index_info), std::move(unbound_expressions),
	                                     table, std::move(function));
}

} // namespace duckdb


namespace duckdb {

void LogicalCreateTable::Serialize(FieldWriter &writer) const {
	writer.WriteSerializable(*info);
}

unique_ptr<LogicalOperator> LogicalCreateTable::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto info = reader.ReadRequiredSerializable<BoundCreateTableInfo>(state.gstate);
	auto &schema = info->schema;
	return make_uniq<LogicalCreateTable>(schema, std::move(info));
}

idx_t LogicalCreateTable::EstimateCardinality(ClientContext &context) {
	return 1;
}

} // namespace duckdb


namespace duckdb {

LogicalCrossProduct::LogicalCrossProduct(unique_ptr<LogicalOperator> left, unique_ptr<LogicalOperator> right)
    : LogicalUnconditionalJoin(LogicalOperatorType::LOGICAL_CROSS_PRODUCT, std::move(left), std::move(right)) {
}

unique_ptr<LogicalOperator> LogicalCrossProduct::Create(unique_ptr<LogicalOperator> left,
                                                        unique_ptr<LogicalOperator> right) {
	if (left->type == LogicalOperatorType::LOGICAL_DUMMY_SCAN) {
		return right;
	}
	if (right->type == LogicalOperatorType::LOGICAL_DUMMY_SCAN) {
		return left;
	}
	return make_uniq<LogicalCrossProduct>(std::move(left), std::move(right));
}

void LogicalCrossProduct::Serialize(FieldWriter &writer) const {
}

unique_ptr<LogicalOperator> LogicalCrossProduct::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	// TODO(stephwang): review if unique_ptr<LogicalOperator> plan is needed
	auto result = unique_ptr<LogicalCrossProduct>(new LogicalCrossProduct());
	return std::move(result);
}

} // namespace duckdb





namespace duckdb {

void LogicalCTERef::Serialize(FieldWriter &writer) const {
	writer.WriteField(table_index);
	writer.WriteField(cte_index);
	writer.WriteRegularSerializableList(chunk_types);
	writer.WriteList<string>(bound_columns);
}

unique_ptr<LogicalOperator> LogicalCTERef::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto table_index = reader.ReadRequired<idx_t>();
	auto cte_index = reader.ReadRequired<idx_t>();
	auto chunk_types = reader.ReadRequiredSerializableList<LogicalType, LogicalType>();
	auto bound_columns = reader.ReadRequiredList<string>();
	return make_uniq<LogicalCTERef>(table_index, cte_index, chunk_types, bound_columns);
}

vector<idx_t> LogicalCTERef::GetTableIndex() const {
	return vector<idx_t> {table_index};
}

string LogicalCTERef::GetName() const {
#ifdef DEBUG
	if (DBConfigOptions::debug_print_bindings) {
		return LogicalOperator::GetName() + StringUtil::Format(" #%llu", table_index);
	}
#endif
	return LogicalOperator::GetName();
}

} // namespace duckdb






namespace duckdb {

LogicalDelete::LogicalDelete(TableCatalogEntry &table, idx_t table_index)
    : LogicalOperator(LogicalOperatorType::LOGICAL_DELETE), table(table), table_index(table_index),
      return_chunk(false) {
}

void LogicalDelete::Serialize(FieldWriter &writer) const {
	table.Serialize(writer.GetSerializer());
	writer.WriteField(table_index);
	writer.WriteField(return_chunk);
	writer.WriteSerializableList(this->expressions);
}

unique_ptr<LogicalOperator> LogicalDelete::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto &context = state.gstate.context;
	auto info = TableCatalogEntry::Deserialize(reader.GetSource(), context);

	auto &table_catalog_entry = Catalog::GetEntry<TableCatalogEntry>(context, info->catalog, info->schema, info->table);

	auto table_index = reader.ReadRequired<idx_t>();
	auto result = make_uniq<LogicalDelete>(table_catalog_entry, table_index);
	result->return_chunk = reader.ReadRequired<bool>();
	result->expressions = reader.ReadRequiredSerializableList<duckdb::Expression>(state.gstate);
	return std::move(result);
}

idx_t LogicalDelete::EstimateCardinality(ClientContext &context) {
	return return_chunk ? LogicalOperator::EstimateCardinality(context) : 1;
}

vector<idx_t> LogicalDelete::GetTableIndex() const {
	return vector<idx_t> {table_index};
}

vector<ColumnBinding> LogicalDelete::GetColumnBindings() {
	if (return_chunk) {
		return GenerateColumnBindings(table_index, table.GetTypes().size());
	}
	return {ColumnBinding(0, 0)};
}

void LogicalDelete::ResolveTypes() {
	if (return_chunk) {
		types = table.GetTypes();
	} else {
		types.emplace_back(LogicalType::BIGINT);
	}
}

string LogicalDelete::GetName() const {
#ifdef DEBUG
	if (DBConfigOptions::debug_print_bindings) {
		return LogicalOperator::GetName() + StringUtil::Format(" #%llu", table_index);
	}
#endif
	return LogicalOperator::GetName();
}

} // namespace duckdb





namespace duckdb {

void LogicalDelimGet::Serialize(FieldWriter &writer) const {
	writer.WriteField(table_index);
	writer.WriteRegularSerializableList(chunk_types);
}

unique_ptr<LogicalOperator> LogicalDelimGet::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto table_index = reader.ReadRequired<idx_t>();
	auto chunk_types = reader.ReadRequiredSerializableList<LogicalType, LogicalType>();
	return make_uniq<LogicalDelimGet>(table_index, chunk_types);
}

vector<idx_t> LogicalDelimGet::GetTableIndex() const {
	return vector<idx_t> {table_index};
}

string LogicalDelimGet::GetName() const {
#ifdef DEBUG
	if (DBConfigOptions::debug_print_bindings) {
		return LogicalOperator::GetName() + StringUtil::Format(" #%llu", table_index);
	}
#endif
	return LogicalOperator::GetName();
}

} // namespace duckdb



namespace duckdb {

LogicalDelimJoin::LogicalDelimJoin(JoinType type)
    : LogicalComparisonJoin(type, LogicalOperatorType::LOGICAL_DELIM_JOIN) {
}

void LogicalDelimJoin::Serialize(FieldWriter &writer) const {
	LogicalComparisonJoin::Serialize(writer);
	if (type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
		D_ASSERT(duplicate_eliminated_columns.empty());
		// if the delim join has no delim columns anymore it is turned into a regular comparison join
		return;
	}
	writer.WriteSerializableList(duplicate_eliminated_columns);
}

unique_ptr<LogicalOperator> LogicalDelimJoin::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto result = make_uniq<LogicalDelimJoin>(JoinType::INVALID);
	LogicalComparisonJoin::Deserialize(*result, state, reader);
	result->duplicate_eliminated_columns = reader.ReadRequiredSerializableList<Expression>(state.gstate);
	return std::move(result);
}

} // namespace duckdb




namespace duckdb {

string LogicalDistinct::ParamsToString() const {
	string result = LogicalOperator::ParamsToString();
	if (!distinct_targets.empty()) {
		result += StringUtil::Join(distinct_targets, distinct_targets.size(), "\n",
		                           [](const unique_ptr<Expression> &child) { return child->GetName(); });
	}

	return result;
}
void LogicalDistinct::Serialize(FieldWriter &writer) const {
	writer.WriteField<DistinctType>(distinct_type);
	writer.WriteSerializableList(distinct_targets);
	if (order_by) {
		throw NotImplementedException("Serializing ORDER BY not yet supported");
	}
}

unique_ptr<LogicalOperator> LogicalDistinct::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto distinct_type = reader.ReadRequired<DistinctType>();
	auto distinct_targets = reader.ReadRequiredSerializableList<Expression>(state.gstate);
	return make_uniq<LogicalDistinct>(std::move(distinct_targets), distinct_type);
}

} // namespace duckdb





namespace duckdb {

void LogicalDummyScan::Serialize(FieldWriter &writer) const {
	writer.WriteField(table_index);
}

unique_ptr<LogicalOperator> LogicalDummyScan::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto table_index = reader.ReadRequired<idx_t>();
	return make_uniq<LogicalDummyScan>(table_index);
}

vector<idx_t> LogicalDummyScan::GetTableIndex() const {
	return vector<idx_t> {table_index};
}

string LogicalDummyScan::GetName() const {
#ifdef DEBUG
	if (DBConfigOptions::debug_print_bindings) {
		return LogicalOperator::GetName() + StringUtil::Format(" #%llu", table_index);
	}
#endif
	return LogicalOperator::GetName();
}

} // namespace duckdb



namespace duckdb {

LogicalEmptyResult::LogicalEmptyResult(unique_ptr<LogicalOperator> op)
    : LogicalOperator(LogicalOperatorType::LOGICAL_EMPTY_RESULT) {

	this->bindings = op->GetColumnBindings();

	op->ResolveOperatorTypes();
	this->return_types = op->types;
}

LogicalEmptyResult::LogicalEmptyResult() : LogicalOperator(LogicalOperatorType::LOGICAL_EMPTY_RESULT) {
}

void LogicalEmptyResult::Serialize(FieldWriter &writer) const {
	writer.WriteRegularSerializableList(return_types);
	writer.WriteList<ColumnBinding>(bindings);
}

unique_ptr<LogicalOperator> LogicalEmptyResult::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto return_types = reader.ReadRequiredSerializableList<LogicalType, LogicalType>();
	auto bindings = reader.ReadRequiredList<ColumnBinding>();
	auto result = unique_ptr<LogicalEmptyResult>(new LogicalEmptyResult());
	result->return_types = return_types;
	result->bindings = bindings;
	return std::move(result);
}

} // namespace duckdb


namespace duckdb {

void LogicalExecute::Serialize(FieldWriter &writer) const {
	throw NotImplementedException(LogicalOperatorToString(type));
}

unique_ptr<LogicalOperator> LogicalExecute::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	throw NotImplementedException(LogicalOperatorToString(state.type));
}
} // namespace duckdb



namespace duckdb {

void LogicalExplain::Serialize(FieldWriter &writer) const {
	writer.WriteField(explain_type);
	writer.WriteString(physical_plan);
	writer.WriteString(logical_plan_unopt);
	writer.WriteString(logical_plan_opt);
}

unique_ptr<LogicalOperator> LogicalExplain::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto explain_type = reader.ReadRequired<ExplainType>();
	// TODO(stephwang) review if unique_ptr<LogicalOperator> plan is needed
	auto result = unique_ptr<LogicalExplain>(new LogicalExplain(explain_type));
	result->physical_plan = reader.ReadRequired<string>();
	result->logical_plan_unopt = reader.ReadRequired<string>();
	result->logical_plan_opt = reader.ReadRequired<string>();
	return std::move(result);
}
} // namespace duckdb


namespace duckdb {

void LogicalExport::Serialize(FieldWriter &writer) const {
	throw NotImplementedException(LogicalOperatorToString(type));
}

unique_ptr<LogicalOperator> LogicalExport::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	throw NotImplementedException(LogicalOperatorToString(state.type));
}

} // namespace duckdb





namespace duckdb {

void LogicalExpressionGet::Serialize(FieldWriter &writer) const {
	writer.WriteField(table_index);
	writer.WriteRegularSerializableList(expr_types);

	writer.WriteField<idx_t>(expressions.size());
	for (auto &entry : expressions) {
		writer.WriteSerializableList(entry);
	}
}

unique_ptr<LogicalOperator> LogicalExpressionGet::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto table_index = reader.ReadRequired<idx_t>();
	auto expr_types = reader.ReadRequiredSerializableList<LogicalType, LogicalType>();

	auto expressions_size = reader.ReadRequired<idx_t>();
	vector<vector<unique_ptr<Expression>>> expressions;
	for (idx_t i = 0; i < expressions_size; i++) {
		expressions.push_back(reader.ReadRequiredSerializableList<Expression>(state.gstate));
	}

	return make_uniq<LogicalExpressionGet>(table_index, expr_types, std::move(expressions));
}

vector<idx_t> LogicalExpressionGet::GetTableIndex() const {
	return vector<idx_t> {table_index};
}

string LogicalExpressionGet::GetName() const {
#ifdef DEBUG
	if (DBConfigOptions::debug_print_bindings) {
		return LogicalOperator::GetName() + StringUtil::Format(" #%llu", table_index);
	}
#endif
	return LogicalOperator::GetName();
}

} // namespace duckdb



namespace duckdb {
unique_ptr<LogicalExtensionOperator> LogicalExtensionOperator::Deserialize(LogicalDeserializationState &state,
                                                                           FieldReader &reader) {
	auto &config = DBConfig::GetConfig(state.gstate.context);

	auto extension_name = reader.ReadRequired<std::string>();
	for (auto &extension : config.operator_extensions) {
		if (extension->GetName() == extension_name) {
			return extension->Deserialize(state, reader);
		}
	}

	throw SerializationException("No serialization method exists for extension: " + extension_name);
}
} // namespace duckdb




namespace duckdb {

LogicalFilter::LogicalFilter(unique_ptr<Expression> expression) : LogicalOperator(LogicalOperatorType::LOGICAL_FILTER) {
	expressions.push_back(std::move(expression));
	SplitPredicates(expressions);
}

LogicalFilter::LogicalFilter() : LogicalOperator(LogicalOperatorType::LOGICAL_FILTER) {
}

void LogicalFilter::ResolveTypes() {
	types = MapTypes(children[0]->types, projection_map);
}

vector<ColumnBinding> LogicalFilter::GetColumnBindings() {
	return MapBindings(children[0]->GetColumnBindings(), projection_map);
}

// Split the predicates separated by AND statements
// These are the predicates that are safe to push down because all of them MUST
// be true
bool LogicalFilter::SplitPredicates(vector<unique_ptr<Expression>> &expressions) {
	bool found_conjunction = false;
	for (idx_t i = 0; i < expressions.size(); i++) {
		if (expressions[i]->type == ExpressionType::CONJUNCTION_AND) {
			auto &conjunction = expressions[i]->Cast<BoundConjunctionExpression>();
			found_conjunction = true;
			// AND expression, append the other children
			for (idx_t k = 1; k < conjunction.children.size(); k++) {
				expressions.push_back(std::move(conjunction.children[k]));
			}
			// replace this expression with the first child of the conjunction
			expressions[i] = std::move(conjunction.children[0]);
			// we move back by one so the right child is checked again
			// in case it is an AND expression as well
			i--;
		}
	}
	return found_conjunction;
}

void LogicalFilter::Serialize(FieldWriter &writer) const {
	writer.WriteSerializableList<Expression>(expressions);
	writer.WriteList<idx_t>(projection_map);
}

unique_ptr<LogicalOperator> LogicalFilter::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto expressions = reader.ReadRequiredSerializableList<Expression>(state.gstate);
	auto projection_map = reader.ReadRequiredList<idx_t>();
	auto result = make_uniq<LogicalFilter>();
	result->expressions = std::move(expressions);
	result->projection_map = std::move(projection_map);
	return std::move(result);
}

} // namespace duckdb
