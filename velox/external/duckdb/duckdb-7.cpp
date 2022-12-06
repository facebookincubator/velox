// See https://raw.githubusercontent.com/duckdb/duckdb/master/LICENSE for licensing information

#include "duckdb.hpp"
#include "duckdb-internal.hpp"
#ifndef DUCKDB_AMALGAMATION
#error header mismatch
#endif




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
	auto copy = make_unique<LimitModifier>();
	if (limit) {
		copy->limit = limit->Copy();
	}
	if (offset) {
		copy->offset = offset->Copy();
	}
	return move(copy);
}

void LimitModifier::Serialize(FieldWriter &writer) const {
	writer.WriteOptional(limit);
	writer.WriteOptional(offset);
}

unique_ptr<ResultModifier> LimitModifier::Deserialize(FieldReader &reader) {
	auto mod = make_unique<LimitModifier>();
	mod->limit = reader.ReadOptional<ParsedExpression>(nullptr);
	mod->offset = reader.ReadOptional<ParsedExpression>(nullptr);
	return move(mod);
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
	auto copy = make_unique<DistinctModifier>();
	for (auto &expr : distinct_on_targets) {
		copy->distinct_on_targets.push_back(expr->Copy());
	}
	return move(copy);
}

void DistinctModifier::Serialize(FieldWriter &writer) const {
	writer.WriteSerializableList(distinct_on_targets);
}

unique_ptr<ResultModifier> DistinctModifier::Deserialize(FieldReader &reader) {
	auto mod = make_unique<DistinctModifier>();
	mod->distinct_on_targets = reader.ReadRequiredSerializableList<ParsedExpression>();
	return move(mod);
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
	auto copy = make_unique<OrderModifier>();
	for (auto &order : orders) {
		copy->orders.emplace_back(order.type, order.null_order, order.expression->Copy());
	}
	return move(copy);
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

OrderByNode OrderByNode::Deserialize(Deserializer &source) {
	FieldReader reader(source);
	auto type = reader.ReadRequired<OrderType>();
	auto null_order = reader.ReadRequired<OrderByNullType>();
	auto expression = reader.ReadRequiredSerializable<ParsedExpression>();
	reader.Finalize();
	return OrderByNode(type, null_order, move(expression));
}

void OrderModifier::Serialize(FieldWriter &writer) const {
	writer.WriteRegularSerializableList(orders);
}

unique_ptr<ResultModifier> OrderModifier::Deserialize(FieldReader &reader) {
	auto mod = make_unique<OrderModifier>();
	mod->orders = reader.ReadRequiredSerializableList<OrderByNode, OrderByNode>();
	return move(mod);
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
	auto copy = make_unique<LimitPercentModifier>();
	if (limit) {
		copy->limit = limit->Copy();
	}
	if (offset) {
		copy->offset = offset->Copy();
	}
	return move(copy);
}

void LimitPercentModifier::Serialize(FieldWriter &writer) const {
	writer.WriteOptional(limit);
	writer.WriteOptional(offset);
}

unique_ptr<ResultModifier> LimitPercentModifier::Deserialize(FieldReader &reader) {
	auto mod = make_unique<LimitPercentModifier>();
	mod->limit = reader.ReadOptional<ParsedExpression>(nullptr);
	mod->offset = reader.ReadOptional<ParsedExpression>(nullptr);
	return move(mod);
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

CallStatement::CallStatement() : SQLStatement(StatementType::CALL_STATEMENT) {
}

CallStatement::CallStatement(const CallStatement &other) : SQLStatement(other), function(other.function->Copy()) {
}

unique_ptr<SQLStatement> CallStatement::Copy() const {
	return unique_ptr<CallStatement>(new CallStatement(*this));
}

} // namespace duckdb


namespace duckdb {

CopyStatement::CopyStatement() : SQLStatement(StatementType::COPY_STATEMENT), info(make_unique<CopyInfo>()) {
}

CopyStatement::CopyStatement(const CopyStatement &other) : SQLStatement(other), info(other.info->Copy()) {
	if (other.select_statement) {
		select_statement = other.select_statement->Copy();
	}
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

DropStatement::DropStatement() : SQLStatement(StatementType::DROP_STATEMENT), info(make_unique<DropInfo>()) {
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
    : SQLStatement(StatementType::EXPLAIN_STATEMENT), stmt(move(stmt)), explain_type(explain_type) {
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
    : SQLStatement(StatementType::EXPORT_STATEMENT), info(move(info)) {
}

ExportStatement::ExportStatement(const ExportStatement &other) : SQLStatement(other), info(other.info->Copy()) {
}

unique_ptr<SQLStatement> ExportStatement::Copy() const {
	return unique_ptr<ExportStatement>(new ExportStatement(*this));
}

} // namespace duckdb


namespace duckdb {

ExtensionStatement::ExtensionStatement(ParserExtension extension_p, unique_ptr<ParserExtensionParseData> parse_data_p)
    : SQLStatement(StatementType::EXTENSION_STATEMENT), extension(move(extension_p)), parse_data(move(parse_data_p)) {
}

unique_ptr<SQLStatement> ExtensionStatement::Copy() const {
	return make_unique<ExtensionStatement>(extension, parse_data->Copy());
}

} // namespace duckdb




namespace duckdb {

InsertStatement::InsertStatement() : SQLStatement(StatementType::INSERT_STATEMENT), schema(DEFAULT_SCHEMA) {
}

InsertStatement::InsertStatement(const InsertStatement &other)
    : SQLStatement(other),
      select_statement(unique_ptr_cast<SQLStatement, SelectStatement>(other.select_statement->Copy())),
      columns(other.columns), table(other.table), schema(other.schema) {
	cte_map = other.cte_map.Copy();
}

string InsertStatement::ToString() const {
	string result;
	result = cte_map.ToString();
	result += "INSERT INTO ";
	if (!schema.empty()) {
		result += KeywordHelper::WriteOptionallyQuoted(schema) + ".";
	}
	result += KeywordHelper::WriteOptionallyQuoted(table);
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
		values_list->alias = string();
		result += values_list->ToString();
	} else {
		result += select_statement->ToString();
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

ExpressionListRef *InsertStatement::GetValuesList() const {
	if (select_statement->node->type != QueryNodeType::SELECT_NODE) {
		return nullptr;
	}
	auto &node = (SelectNode &)*select_statement->node;
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
	return (ExpressionListRef *)node.from_table.get();
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

PragmaStatement::PragmaStatement() : SQLStatement(StatementType::PRAGMA_STATEMENT), info(make_unique<PragmaInfo>()) {
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
    : SQLStatement(StatementType::RELATION_STATEMENT), relation(move(relation)) {
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

unique_ptr<SelectStatement> SelectStatement::Deserialize(Deserializer &source) {
	auto result = make_unique<SelectStatement>();
	result->node = QueryNode::Deserialize(source);
	return result;
}

bool SelectStatement::Equals(const SQLStatement *other_p) const {
	if (type != other_p->type) {
		return false;
	}
	auto other = (SelectStatement *)other_p;
	return node->Equals(other->node.get());
}

string SelectStatement::ToString() const {
	return node->ToString();
}

} // namespace duckdb


namespace duckdb {

SetStatement::SetStatement(std::string name_p, Value value_p, SetScope scope_p)
    : SQLStatement(StatementType::SET_STATEMENT), name(move(name_p)), value(move(value_p)), scope(scope_p) {
}

unique_ptr<SQLStatement> SetStatement::Copy() const {
	return unique_ptr<SetStatement>(new SetStatement(*this));
}

} // namespace duckdb


namespace duckdb {

ShowStatement::ShowStatement() : SQLStatement(StatementType::SHOW_STATEMENT), info(make_unique<ShowSelectInfo>()) {
}

ShowStatement::ShowStatement(const ShowStatement &other) : SQLStatement(other), info(other.info->Copy()) {
}

unique_ptr<SQLStatement> ShowStatement::Copy() const {
	return unique_ptr<ShowStatement>(new ShowStatement(*this));
}

} // namespace duckdb


namespace duckdb {

TransactionStatement::TransactionStatement(TransactionType type)
    : SQLStatement(StatementType::TRANSACTION_STATEMENT), info(make_unique<TransactionInfo>(type)) {
}

TransactionStatement::TransactionStatement(const TransactionStatement &other)
    : SQLStatement(other), info(make_unique<TransactionInfo>(other.info->type)) {
}

unique_ptr<SQLStatement> TransactionStatement::Copy() const {
	return unique_ptr<TransactionStatement>(new TransactionStatement(*this));
}

} // namespace duckdb



namespace duckdb {

UpdateStatement::UpdateStatement() : SQLStatement(StatementType::UPDATE_STATEMENT) {
}

UpdateStatement::UpdateStatement(const UpdateStatement &other)
    : SQLStatement(other), table(other.table->Copy()), columns(other.columns) {
	if (other.condition) {
		condition = other.condition->Copy();
	}
	if (other.from_table) {
		from_table = other.from_table->Copy();
	}
	for (auto &expr : other.expressions) {
		expressions.emplace_back(expr->Copy());
	}
	cte_map = other.cte_map.Copy();
}

string UpdateStatement::ToString() const {
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
    : SQLStatement(StatementType::VACUUM_STATEMENT), info(make_unique<VacuumInfo>(options)) {
}

VacuumStatement::VacuumStatement(const VacuumStatement &other) : SQLStatement(other), info(other.info->Copy()) {
}

unique_ptr<SQLStatement> VacuumStatement::Copy() const {
	return unique_ptr<VacuumStatement>(new VacuumStatement(*this));
}

} // namespace duckdb





namespace duckdb {

string BaseTableRef::ToString() const {
	string schema = schema_name.empty() ? "" : KeywordHelper::WriteOptionallyQuoted(schema_name) + ".";
	string result = schema + KeywordHelper::WriteOptionallyQuoted(table_name);
	return BaseToString(result, column_name_alias);
}

bool BaseTableRef::Equals(const TableRef *other_p) const {
	if (!TableRef::Equals(other_p)) {
		return false;
	}
	auto other = (BaseTableRef *)other_p;
	return other->schema_name == schema_name && other->table_name == table_name &&
	       column_name_alias == other->column_name_alias;
}

void BaseTableRef::Serialize(FieldWriter &writer) const {
	writer.WriteString(schema_name);
	writer.WriteString(table_name);
	writer.WriteList<string>(column_name_alias);
}

unique_ptr<TableRef> BaseTableRef::Deserialize(FieldReader &reader) {
	auto result = make_unique<BaseTableRef>();

	result->schema_name = reader.ReadRequired<string>();
	result->table_name = reader.ReadRequired<string>();
	result->column_name_alias = reader.ReadRequiredList<string>();

	return move(result);
}

unique_ptr<TableRef> BaseTableRef::Copy() {
	auto copy = make_unique<BaseTableRef>();

	copy->schema_name = schema_name;
	copy->table_name = table_name;
	copy->column_name_alias = column_name_alias;
	CopyProperties(*copy);

	return move(copy);
}
} // namespace duckdb




namespace duckdb {

string CrossProductRef::ToString() const {
	return left->ToString() + ", " + right->ToString();
}

bool CrossProductRef::Equals(const TableRef *other_p) const {
	if (!TableRef::Equals(other_p)) {
		return false;
	}
	auto other = (CrossProductRef *)other_p;
	return left->Equals(other->left.get()) && right->Equals(other->right.get());
}

unique_ptr<TableRef> CrossProductRef::Copy() {
	auto copy = make_unique<CrossProductRef>();
	copy->left = left->Copy();
	copy->right = right->Copy();
	copy->alias = alias;
	return move(copy);
}

void CrossProductRef::Serialize(FieldWriter &writer) const {
	writer.WriteSerializable(*left);
	writer.WriteSerializable(*right);
}

unique_ptr<TableRef> CrossProductRef::Deserialize(FieldReader &reader) {
	auto result = make_unique<CrossProductRef>();

	result->left = reader.ReadRequiredSerializable<TableRef>();
	result->right = reader.ReadRequiredSerializable<TableRef>();
	D_ASSERT(result->left);
	D_ASSERT(result->right);

	return move(result);
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
	return make_unique<EmptyTableRef>();
}

void EmptyTableRef::Serialize(FieldWriter &writer) const {
}

unique_ptr<TableRef> EmptyTableRef::Deserialize(FieldReader &reader) {
	return make_unique<EmptyTableRef>();
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
	auto result = make_unique<ExpressionListRef>();
	for (auto &val_list : values) {
		vector<unique_ptr<ParsedExpression>> new_val_list;
		new_val_list.reserve(val_list.size());
		for (auto &val : val_list) {
			new_val_list.push_back(val->Copy());
		}
		result->values.push_back(move(new_val_list));
	}
	result->expected_names = expected_names;
	result->expected_types = expected_types;
	CopyProperties(*result);
	return move(result);
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

unique_ptr<TableRef> ExpressionListRef::Deserialize(FieldReader &reader) {
	auto result = make_unique<ExpressionListRef>();
	// value list
	result->expected_names = reader.ReadRequiredList<string>();
	result->expected_types = reader.ReadRequiredSerializableList<LogicalType, LogicalType>();
	idx_t value_list_size = reader.ReadRequired<uint32_t>();
	auto &source = reader.GetSource();
	for (idx_t i = 0; i < value_list_size; i++) {
		vector<unique_ptr<ParsedExpression>> value_list;
		source.ReadList<ParsedExpression>(value_list);
		result->values.push_back(move(value_list));
	}
	return move(result);
}

} // namespace duckdb





namespace duckdb {

string JoinRef::ToString() const {
	string result;
	result = left->ToString() + " ";
	if (is_natural) {
		result += "NATURAL ";
	}
	result += JoinTypeToString(type) + " JOIN ";
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
	auto copy = make_unique<JoinRef>();
	copy->left = left->Copy();
	copy->right = right->Copy();
	if (condition) {
		copy->condition = condition->Copy();
	}
	copy->type = type;
	copy->is_natural = is_natural;
	copy->alias = alias;
	copy->using_columns = using_columns;
	return move(copy);
}

void JoinRef::Serialize(FieldWriter &writer) const {
	writer.WriteSerializable(*left);
	writer.WriteSerializable(*right);
	writer.WriteOptional(condition);
	writer.WriteField<JoinType>(type);
	writer.WriteField<bool>(is_natural);
	writer.WriteList<string>(using_columns);
}

unique_ptr<TableRef> JoinRef::Deserialize(FieldReader &reader) {
	auto result = make_unique<JoinRef>();
	result->left = reader.ReadRequiredSerializable<TableRef>();
	result->right = reader.ReadRequiredSerializable<TableRef>();
	result->condition = reader.ReadOptional<ParsedExpression>(nullptr);
	result->type = reader.ReadRequired<JoinType>();
	result->is_natural = reader.ReadRequired<bool>();
	result->using_columns = reader.ReadRequiredList<string>();
	return move(result);
}

} // namespace duckdb





namespace duckdb {

string SubqueryRef::ToString() const {
	string result = "(" + subquery->ToString() + ")";
	return BaseToString(result, column_name_alias);
}

SubqueryRef::SubqueryRef(unique_ptr<SelectStatement> subquery_p, string alias_p)
    : TableRef(TableReferenceType::SUBQUERY), subquery(move(subquery_p)) {
	this->alias = move(alias_p);
}

bool SubqueryRef::Equals(const TableRef *other_p) const {
	if (!TableRef::Equals(other_p)) {
		return false;
	}
	auto other = (SubqueryRef *)other_p;
	return subquery->Equals(other->subquery.get());
}

unique_ptr<TableRef> SubqueryRef::Copy() {
	auto copy = make_unique<SubqueryRef>(unique_ptr_cast<SQLStatement, SelectStatement>(subquery->Copy()), alias);
	copy->column_name_alias = column_name_alias;
	CopyProperties(*copy);
	return move(copy);
}

void SubqueryRef::Serialize(FieldWriter &writer) const {
	writer.WriteSerializable(*subquery);
	writer.WriteList<string>(column_name_alias);
}

unique_ptr<TableRef> SubqueryRef::Deserialize(FieldReader &reader) {
	auto subquery = reader.ReadRequiredSerializable<SelectStatement>();
	auto result = make_unique<SubqueryRef>(move(subquery));
	result->column_name_alias = reader.ReadRequiredList<string>();
	return move(result);
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

unique_ptr<TableRef> TableFunctionRef::Deserialize(FieldReader &reader) {
	auto result = make_unique<TableFunctionRef>();
	result->function = reader.ReadRequiredSerializable<ParsedExpression>();
	result->alias = reader.ReadRequired<string>();
	result->column_name_alias = reader.ReadRequiredList<string>();
	return move(result);
}

unique_ptr<TableRef> TableFunctionRef::Copy() {
	auto copy = make_unique<TableFunctionRef>();

	copy->function = function->Copy();
	copy->column_name_alias = column_name_alias;
	CopyProperties(*copy);

	return move(copy);
}

} // namespace duckdb







namespace duckdb {

string TableRef::BaseToString(string result) const {
	vector<string> column_name_alias;
	return BaseToString(move(result), column_name_alias);
}

string TableRef::BaseToString(string result, const vector<string> &column_name_alias) const {
	if (!alias.empty()) {
		result += " AS " + KeywordHelper::WriteOptionallyQuoted(alias);
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
		result += " TABLESAMPLE " + SampleMethodToString(sample->method);
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
	case TableReferenceType::CROSS_PRODUCT:
		result = CrossProductRef::Deserialize(reader);
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
	case TableReferenceType::CTE:
	case TableReferenceType::INVALID:
		throw InternalException("Unsupported type for TableRef::Deserialize");
	}
	reader.Finalize();

	result->alias = alias;
	result->sample = move(sample);
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
		return make_unique<UniqueConstraint>(columns, is_primary_key);
	}
	case duckdb_libpgquery::PG_CONSTR_CHECK: {
		auto expression = TransformExpression(constraint->raw_expr);
		if (expression->HasSubquery()) {
			throw ParserException("subqueries prohibited in CHECK constraints");
		}
		return make_unique<CheckConstraint>(TransformExpression(constraint->raw_expr));
	}
	case duckdb_libpgquery::PG_CONSTR_FOREIGN: {
		ForeignKeyInfo fk_info;
		fk_info.type = ForeignKeyType::FK_TYPE_FOREIGN_KEY_TABLE;
		if (constraint->pktable->schemaname) {
			fk_info.schema = constraint->pktable->schemaname;
		} else {
			fk_info.schema = "";
		}
		fk_info.table = constraint->pktable->relname;
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
		return make_unique<ForeignKeyConstraint>(pk_columns, fk_columns, move(fk_info));
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
		return make_unique<NotNullConstraint>(LogicalIndex(index));
	case duckdb_libpgquery::PG_CONSTR_CHECK:
		return TransformConstraint(cell);
	case duckdb_libpgquery::PG_CONSTR_PRIMARY:
		return make_unique<UniqueConstraint>(LogicalIndex(index), true);
	case duckdb_libpgquery::PG_CONSTR_UNIQUE:
		return make_unique<UniqueConstraint>(LogicalIndex(index), false);
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
		if (constraint->pktable->schemaname) {
			fk_info.schema = constraint->pktable->schemaname;
		} else {
			fk_info.schema = "";
		}
		fk_info.table = constraint->pktable->relname;
		vector<string> pk_columns, fk_columns;

		fk_columns.emplace_back(column.Name().c_str());
		if (constraint->pk_attrs) {
			for (auto kc = constraint->pk_attrs->head; kc; kc = kc->next) {
				pk_columns.emplace_back(reinterpret_cast<duckdb_libpgquery::PGValue *>(kc->data.ptr_value)->val.str);
			}
		}
		if (pk_columns.size() != fk_columns.size()) {
			throw ParserException("The number of referencing and referenced columns for foreign keys must be the same");
		}
		return make_unique<ForeignKeyConstraint>(pk_columns, fk_columns, move(fk_info));
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
			children.push_back(move(result));
			if (index->is_slice) {
				// slice
				children.push_back(!index->lidx ? make_unique<ConstantExpression>(Value())
				                                : TransformExpression(index->lidx));
				children.push_back(!index->uidx ? make_unique<ConstantExpression>(Value())
				                                : TransformExpression(index->uidx));
				result = make_unique<OperatorExpression>(ExpressionType::ARRAY_SLICE, move(children));
			} else {
				// array access
				D_ASSERT(!index->lidx);
				D_ASSERT(index->uidx);
				children.push_back(TransformExpression(index->uidx));
				result = make_unique<OperatorExpression>(ExpressionType::ARRAY_EXTRACT, move(children));
			}
			break;
		}
		case duckdb_libpgquery::T_PGString: {
			auto val = (duckdb_libpgquery::PGValue *)target;
			vector<unique_ptr<ParsedExpression>> children;
			children.push_back(move(result));
			children.push_back(TransformValue(*val));
			result = make_unique<OperatorExpression>(ExpressionType::STRUCT_EXTRACT, move(children));
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
				result = move(next);
			} else {
				result = make_unique<ConjunctionExpression>(ExpressionType::CONJUNCTION_AND, move(result), move(next));
			}
			break;
		}
		case duckdb_libpgquery::PG_OR_EXPR: {
			if (!result) {
				result = move(next);
			} else {
				result = make_unique<ConjunctionExpression>(ExpressionType::CONJUNCTION_OR, move(result), move(next));
			}
			break;
		}
		case duckdb_libpgquery::PG_NOT_EXPR: {
			if (next->type == ExpressionType::COMPARE_IN) {
				// convert COMPARE_IN to COMPARE_NOT_IN
				next->type = ExpressionType::COMPARE_NOT_IN;
				result = move(next);
			} else if (next->type >= ExpressionType::COMPARE_EQUAL &&
			           next->type <= ExpressionType::COMPARE_GREATERTHANOREQUALTO) {
				// NOT on a comparison: we can negate the comparison
				// e.g. NOT(x > y) is equivalent to x <= y
				next->type = NegateComparisionExpression(next->type);
				result = move(next);
			} else {
				result = make_unique<OperatorExpression>(ExpressionType::OPERATOR_NOT, move(next));
			}
			break;
		}
		}
	}
	return result;
}

} // namespace duckdb





namespace duckdb {

unique_ptr<ParsedExpression> Transformer::TransformCase(duckdb_libpgquery::PGCaseExpr *root) {
	D_ASSERT(root);

	auto case_node = make_unique<CaseExpression>();
	for (auto cell = root->args->head; cell != nullptr; cell = cell->next) {
		CaseCheck case_check;

		auto w = reinterpret_cast<duckdb_libpgquery::PGCaseWhen *>(cell->data.ptr_value);
		auto test_raw = TransformExpression(reinterpret_cast<duckdb_libpgquery::PGNode *>(w->expr));
		unique_ptr<ParsedExpression> test;
		auto arg = TransformExpression(reinterpret_cast<duckdb_libpgquery::PGNode *>(root->arg));
		if (arg) {
			case_check.when_expr =
			    make_unique<ComparisonExpression>(ExpressionType::COMPARE_EQUAL, move(arg), move(test_raw));
		} else {
			case_check.when_expr = move(test_raw);
		}
		case_check.then_expr = TransformExpression(reinterpret_cast<duckdb_libpgquery::PGNode *>(w->result));
		case_node->case_checks.push_back(move(case_check));
	}

	if (root->defresult) {
		case_node->else_expr = TransformExpression(reinterpret_cast<duckdb_libpgquery::PGNode *>(root->defresult));
	} else {
		case_node->else_expr = make_unique<ConstantExpression>(Value(LogicalType::SQLNULL));
	}
	return move(case_node);
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
			return make_unique<ConstantExpression>(Value::BLOB(string(c->val.val.str)));
		}
	}
	// transform the expression node
	auto expression = TransformExpression(root->arg);
	bool try_cast = root->tryCast;

	// now create a cast operation
	return make_unique<CastExpression>(target_type, move(expression), try_cast);
}

} // namespace duckdb



namespace duckdb {

// COALESCE(a,b,c) returns the first argument that is NOT NULL, so
// rewrite into CASE(a IS NOT NULL, a, CASE(b IS NOT NULL, b, c))
unique_ptr<ParsedExpression> Transformer::TransformCoalesce(duckdb_libpgquery::PGAExpr *root) {
	D_ASSERT(root);

	auto coalesce_args = reinterpret_cast<duckdb_libpgquery::PGList *>(root->lexpr);
	D_ASSERT(coalesce_args->length > 0); // parser ensures this already

	auto coalesce_op = make_unique<OperatorExpression>(ExpressionType::OPERATOR_COALESCE);
	for (auto cell = coalesce_args->head; cell; cell = cell->next) {
		// get the value of the COALESCE
		auto value_expr = TransformExpression(reinterpret_cast<duckdb_libpgquery::PGNode *>(cell->data.ptr_value));
		coalesce_op->children.push_back(move(value_expr));
	}
	return move(coalesce_op);
}

} // namespace duckdb





namespace duckdb {

unique_ptr<ParsedExpression> Transformer::TransformStarExpression(duckdb_libpgquery::PGNode *node) {
	auto star = (duckdb_libpgquery::PGAStar *)node;
	auto result = make_unique<StarExpression>(star->relation ? star->relation : string());
	if (star->except_list) {
		for (auto head = star->except_list->head; head; head = head->next) {
			auto value = (duckdb_libpgquery::PGValue *)head->data.ptr_value;
			D_ASSERT(value->type == duckdb_libpgquery::T_PGString);
			string exclude_entry = value->val.str;
			if (result->exclude_list.find(exclude_entry) != result->exclude_list.end()) {
				throw ParserException("Duplicate entry \"%s\" in EXCLUDE list", exclude_entry);
			}
			result->exclude_list.insert(move(exclude_entry));
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
			result->replace_list.insert(make_pair(move(exclude_entry), move(replace_expression)));
		}
	}
	if (star->regex) {
		D_ASSERT(result->relation_name.empty());
		D_ASSERT(result->exclude_list.empty());
		D_ASSERT(result->replace_list.empty());
		result->regex = star->regex;
	}
	result->columns = star->columns;
	return move(result);
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
		auto colref = make_unique<ColumnRefExpression>(move(column_names));
		colref->query_location = root->location;
		return move(colref);
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
		return make_unique<ConstantExpression>(Value::INTEGER((int32_t)val.val.ival));
	case duckdb_libpgquery::T_PGBitString: // FIXME: this should actually convert to BLOB
	case duckdb_libpgquery::T_PGString:
		return make_unique<ConstantExpression>(Value(string(val.val.str)));
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
				return make_unique<ConstantExpression>(Value::BIGINT(bigint_value));
			}
			hugeint_t hugeint_value;
			// if that is not successful; try to cast as hugeint
			if (TryCast::Operation<string_t, hugeint_t>(str_val, hugeint_value)) {
				// successfully cast to bigint: bigint value
				return make_unique<ConstantExpression>(Value::HUGEINT(hugeint_value));
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
				return make_unique<ConstantExpression>(move(val));
			}
		}
		// if there is a decimal or the value is too big to cast as either hugeint or bigint
		double dbl_value = Cast::Operation<string_t, double>(str_val);
		return make_unique<ConstantExpression>(Value::DOUBLE(dbl_value));
	}
	case duckdb_libpgquery::T_PGNull:
		return make_unique<ConstantExpression>(Value(LogicalType::SQLNULL));
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
		return make_unique<DefaultExpression>();
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
	default:
		throw NotImplementedException("Expr of type %d not implemented\n", (int)node->type);
	}
}

void Transformer::TransformExpressionList(duckdb_libpgquery::PGList &list,
                                          vector<unique_ptr<ParsedExpression>> &result) {
	for (auto node = list.head; node != nullptr; node = node->next) {
		auto target = reinterpret_cast<duckdb_libpgquery::PGNode *>(node->data.ptr_value);
		D_ASSERT(target);

		auto expr = TransformExpression(target);
		D_ASSERT(expr);

		result.push_back(move(expr));
	}
}

} // namespace duckdb











namespace duckdb {

static ExpressionType WindowToExpressionType(string &fun_name) {
	if (fun_name == "rank") {
		return ExpressionType::WINDOW_RANK;
	} else if (fun_name == "rank_dense" || fun_name == "dense_rank") {
		return ExpressionType::WINDOW_RANK_DENSE;
	} else if (fun_name == "percent_rank") {
		return ExpressionType::WINDOW_PERCENT_RANK;
	} else if (fun_name == "row_number") {
		return ExpressionType::WINDOW_ROW_NUMBER;
	} else if (fun_name == "first_value" || fun_name == "first") {
		return ExpressionType::WINDOW_FIRST_VALUE;
	} else if (fun_name == "last_value" || fun_name == "last") {
		return ExpressionType::WINDOW_LAST_VALUE;
	} else if (fun_name == "nth_value" || fun_name == "last") {
		return ExpressionType::WINDOW_NTH_VALUE;
	} else if (fun_name == "cume_dist") {
		return ExpressionType::WINDOW_CUME_DIST;
	} else if (fun_name == "lead") {
		return ExpressionType::WINDOW_LEAD;
	} else if (fun_name == "lag") {
		return ExpressionType::WINDOW_LAG;
	} else if (fun_name == "ntile") {
		return ExpressionType::WINDOW_NTILE;
	}

	return ExpressionType::WINDOW_AGGREGATE;
}

void Transformer::TransformWindowDef(duckdb_libpgquery::PGWindowDef *window_spec, WindowExpression *expr) {
	D_ASSERT(window_spec);
	D_ASSERT(expr);

	// next: partitioning/ordering expressions
	if (window_spec->partitionClause) {
		TransformExpressionList(*window_spec->partitionClause, expr->partitions);
	}
	TransformOrderBy(window_spec->orderClause, expr->orders);
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

unique_ptr<ParsedExpression> Transformer::TransformFuncCall(duckdb_libpgquery::PGFuncCall *root) {
	auto name = root->funcname;
	string schema, function_name;
	if (name->length == 2) {
		// schema + name
		schema = reinterpret_cast<duckdb_libpgquery::PGValue *>(name->head->data.ptr_value)->val.str;
		function_name = reinterpret_cast<duckdb_libpgquery::PGValue *>(name->head->next->data.ptr_value)->val.str;
	} else {
		// unqualified name
		schema = INVALID_SCHEMA;
		function_name = reinterpret_cast<duckdb_libpgquery::PGValue *>(name->head->data.ptr_value)->val.str;
	}

	auto lowercase_name = StringUtil::Lower(function_name);

	if (root->over) {
		const auto win_fun_type = WindowToExpressionType(lowercase_name);
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

		auto expr = make_unique<WindowExpression>(win_fun_type, schema, lowercase_name);
		expr->ignore_nulls = root->agg_ignore_nulls;

		if (root->agg_filter) {
			auto filter_expr = TransformExpression(root->agg_filter);
			expr->filter_expr = move(filter_expr);
		}

		if (root->args) {
			vector<unique_ptr<ParsedExpression>> function_list;
			TransformExpressionList(*root->args, function_list);

			if (win_fun_type == ExpressionType::WINDOW_AGGREGATE) {
				for (auto &child : function_list) {
					expr->children.push_back(move(child));
				}
			} else {
				if (!function_list.empty()) {
					expr->children.push_back(move(function_list[0]));
				}
				if (win_fun_type == ExpressionType::WINDOW_LEAD || win_fun_type == ExpressionType::WINDOW_LAG) {
					if (function_list.size() > 1) {
						expr->offset_expr = move(function_list[1]);
					}
					if (function_list.size() > 2) {
						expr->default_expr = move(function_list[2]);
					}
					if (function_list.size() > 3) {
						throw ParserException("Incorrect number of parameters for function %s", lowercase_name);
					}
				} else if (win_fun_type == ExpressionType::WINDOW_NTH_VALUE) {
					if (function_list.size() > 1) {
						expr->children.push_back(move(function_list[1]));
					}
					if (function_list.size() > 2) {
						throw ParserException("Incorrect number of parameters for function %s", lowercase_name);
					}
				} else {
					if (function_list.size() > 1) {
						throw ParserException("Incorrect number of parameters for function %s", lowercase_name);
					}
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
		if (window_ref->refname) {
			auto it = window_clauses.find(StringUtil::Lower(string(window_spec->refname)));
			if (it == window_clauses.end()) {
				throw ParserException("window \"%s\" does not exist", window_spec->refname);
			}
			window_ref = it->second;
			D_ASSERT(window_ref);
		}
		TransformWindowDef(window_ref, expr.get());
		TransformWindowFrame(window_spec, expr.get());
		expr->query_location = root->location;
		return move(expr);
	}

	if (root->agg_ignore_nulls) {
		throw ParserException("IGNORE NULLS is not supported for non-window functions");
	}

	//  TransformExpressionList??
	vector<unique_ptr<ParsedExpression>> children;
	if (root->args != nullptr) {
		for (auto node = root->args->head; node != nullptr; node = node->next) {
			auto child_expr = TransformExpression((duckdb_libpgquery::PGNode *)node->data.ptr_value);
			children.push_back(move(child_expr));
		}
	}
	unique_ptr<ParsedExpression> filter_expr;
	if (root->agg_filter) {
		filter_expr = TransformExpression(root->agg_filter);
	}

	auto order_bys = make_unique<OrderModifier>();
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
		auto expr = make_unique<CaseExpression>();
		CaseCheck check;
		check.when_expr = move(children[0]);
		check.then_expr = move(children[1]);
		expr->case_checks.push_back(move(check));
		expr->else_expr = move(children[2]);
		return move(expr);
	} else if (lowercase_name == "construct_array") {
		auto construct_array = make_unique<OperatorExpression>(ExpressionType::ARRAY_CONSTRUCTOR);
		construct_array->children = move(children);
		return move(construct_array);
	} else if (lowercase_name == "ifnull") {
		if (children.size() != 2) {
			throw ParserException("Wrong number of arguments to IFNULL.");
		}

		//  Two-argument COALESCE
		auto coalesce_op = make_unique<OperatorExpression>(ExpressionType::OPERATOR_COALESCE);
		coalesce_op->children.push_back(move(children[0]));
		coalesce_op->children.push_back(move(children[1]));
		return move(coalesce_op);
	}

	auto function = make_unique<FunctionExpression>(schema, lowercase_name.c_str(), move(children), move(filter_expr),
	                                                move(order_bys), root->agg_distinct, false, root->export_state);
	function->query_location = root->location;

	return move(function);
}

static string SQLValueOpToString(duckdb_libpgquery::PGSQLValueFunctionOp op) {
	switch (op) {
	case duckdb_libpgquery::PG_SVFOP_CURRENT_DATE:
		return "current_date";
	case duckdb_libpgquery::PG_SVFOP_CURRENT_TIME:
		return "get_current_time";
	case duckdb_libpgquery::PG_SVFOP_CURRENT_TIME_N:
		return "current_time_n";
	case duckdb_libpgquery::PG_SVFOP_CURRENT_TIMESTAMP:
		return "get_current_timestamp";
	case duckdb_libpgquery::PG_SVFOP_CURRENT_TIMESTAMP_N:
		return "current_timestamp_n";
	case duckdb_libpgquery::PG_SVFOP_LOCALTIME:
		return "current_localtime";
	case duckdb_libpgquery::PG_SVFOP_LOCALTIME_N:
		return "current_localtime_n";
	case duckdb_libpgquery::PG_SVFOP_LOCALTIMESTAMP:
		return "current_localtimestamp";
	case duckdb_libpgquery::PG_SVFOP_LOCALTIMESTAMP_N:
		return "current_localtimestamp_n";
	case duckdb_libpgquery::PG_SVFOP_CURRENT_ROLE:
		return "current_role";
	case duckdb_libpgquery::PG_SVFOP_CURRENT_USER:
		return "current_user";
	case duckdb_libpgquery::PG_SVFOP_USER:
		return "user";
	case duckdb_libpgquery::PG_SVFOP_SESSION_USER:
		return "session_user";
	case duckdb_libpgquery::PG_SVFOP_CURRENT_CATALOG:
		return "current_catalog";
	case duckdb_libpgquery::PG_SVFOP_CURRENT_SCHEMA:
		return "current_schema";
	default:
		throw InternalException("Could not find named SQL value function specification " + to_string((int)op));
	}
}

unique_ptr<ParsedExpression> Transformer::TransformSQLValueFunction(duckdb_libpgquery::PGSQLValueFunction *node) {
	D_ASSERT(node);
	vector<unique_ptr<ParsedExpression>> children;
	auto fname = SQLValueOpToString(node->op);
	return make_unique<FunctionExpression>(DEFAULT_SCHEMA, fname, move(children));
}

} // namespace duckdb



namespace duckdb {

unique_ptr<ParsedExpression> Transformer::TransformGroupingFunction(duckdb_libpgquery::PGGroupingFunc *n) {
	auto op = make_unique<OperatorExpression>(ExpressionType::GROUPING_FUNCTION);
	for (auto node = n->args->head; node; node = node->next) {
		auto n = (duckdb_libpgquery::PGNode *)node->data.ptr_value;
		op->children.push_back(TransformExpression(n));
	}
	op->query_location = n->location;
	return move(op);
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
		expr = make_unique<ConstantExpression>(Value(node->sval));
		break;
	case duckdb_libpgquery::T_PGInteger:
		expr = make_unique<ConstantExpression>(Value(node->ival));
		break;
	default:
		throw InternalException("Unsupported interval transformation");
	}

	if (!node->typmods) {
		return make_unique<CastExpression>(LogicalType::INTERVAL, move(expr));
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
	expr = make_unique<CastExpression>(target_type, move(expr));
	// now push the operation
	vector<unique_ptr<ParsedExpression>> children;
	children.push_back(move(expr));
	return make_unique<FunctionExpression>(fname, move(children));
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

	return unique_ptr<ParsedExpression>(new OperatorExpression(expr_type, move(arg)));
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
	return make_unique<LambdaExpression>(move(lhs), move(rhs));
}

} // namespace duckdb














namespace duckdb {

unique_ptr<ParsedExpression> Transformer::TransformUnaryOperator(const string &op, unique_ptr<ParsedExpression> child) {
	const auto schema = DEFAULT_SCHEMA;

	vector<unique_ptr<ParsedExpression>> children;
	children.push_back(move(child));

	// built-in operator function
	auto result = make_unique<FunctionExpression>(schema, op, move(children));
	result->is_operator = true;
	return move(result);
}

unique_ptr<ParsedExpression> Transformer::TransformBinaryOperator(const string &op, unique_ptr<ParsedExpression> left,
                                                                  unique_ptr<ParsedExpression> right) {
	const auto schema = DEFAULT_SCHEMA;

	vector<unique_ptr<ParsedExpression>> children;
	children.push_back(move(left));
	children.push_back(move(right));

	if (op == "~" || op == "!~") {
		// rewrite 'asdf' SIMILAR TO '.*sd.*' into regexp_full_match('asdf', '.*sd.*')
		bool invert_similar = op == "!~";

		auto result = make_unique<FunctionExpression>(schema, "regexp_full_match", move(children));
		if (invert_similar) {
			return make_unique<OperatorExpression>(ExpressionType::OPERATOR_NOT, move(result));
		} else {
			return move(result);
		}
	} else {
		auto target_type = OperatorToExpressionType(op);
		if (target_type != ExpressionType::INVALID) {
			// built-in comparison operator
			return make_unique<ComparisonExpression>(target_type, move(children[0]), move(children[1]));
		}
		// not a special operator: convert to a function expression
		auto result = make_unique<FunctionExpression>(schema, op, move(children));
		result->is_operator = true;
		return move(result);
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

		auto subquery_expr = make_unique<SubqueryExpression>();
		auto select_statement = make_unique<SelectStatement>();
		auto select_node = make_unique<SelectNode>();
		vector<unique_ptr<ParsedExpression>> children;
		children.push_back(move(right_expr));

		select_node->select_list.push_back(make_unique<FunctionExpression>("UNNEST", move(children)));
		select_node->from_table = make_unique<EmptyTableRef>();
		select_statement->node = move(select_node);
		subquery_expr->subquery = move(select_statement);
		subquery_expr->subquery_type = SubqueryType::ANY;
		subquery_expr->child = move(left_expr);
		subquery_expr->comparison_type = OperatorToExpressionType(name);
		subquery_expr->query_location = root->location;

		if (root->kind == duckdb_libpgquery::PG_AEXPR_OP_ALL) {
			// ALL sublink is equivalent to NOT(ANY) with inverted comparison
			// e.g. [= ALL()] is equivalent to [NOT(<> ANY())]
			// first invert the comparison type
			subquery_expr->comparison_type = NegateComparisionExpression(subquery_expr->comparison_type);
			return make_unique<OperatorExpression>(ExpressionType::OPERATOR_NOT, move(subquery_expr));
		}
		return move(subquery_expr);
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
		auto result = make_unique<OperatorExpression>(operator_type, move(left_expr));
		result->query_location = root->location;
		TransformExpressionList(*((duckdb_libpgquery::PGList *)root->rexpr), result->children);
		return move(result);
	}
	// rewrite NULLIF(a, b) into CASE WHEN a=b THEN NULL ELSE a END
	case duckdb_libpgquery::PG_AEXPR_NULLIF: {
		vector<unique_ptr<ParsedExpression>> children;
		children.push_back(TransformExpression(root->lexpr));
		children.push_back(TransformExpression(root->rexpr));
		return make_unique<FunctionExpression>("nullif", move(children));
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

		auto compare_between = make_unique<BetweenExpression>(move(input), move(between_left), move(between_right));
		if (root->kind == duckdb_libpgquery::PG_AEXPR_BETWEEN) {
			return move(compare_between);
		} else {
			return make_unique<OperatorExpression>(ExpressionType::OPERATOR_NOT, move(compare_between));
		}
	}
	// rewrite SIMILAR TO into regexp_full_match('asdf', '.*sd.*')
	case duckdb_libpgquery::PG_AEXPR_SIMILAR: {
		auto left_expr = TransformExpression(root->lexpr);
		auto right_expr = TransformExpression(root->rexpr);

		vector<unique_ptr<ParsedExpression>> children;
		children.push_back(move(left_expr));

		auto &similar_func = reinterpret_cast<FunctionExpression &>(*right_expr);
		D_ASSERT(similar_func.function_name == "similar_escape");
		D_ASSERT(similar_func.children.size() == 2);
		if (similar_func.children[1]->type != ExpressionType::VALUE_CONSTANT) {
			throw NotImplementedException("Custom escape in SIMILAR TO");
		}
		auto &constant = (ConstantExpression &)*similar_func.children[1];
		if (!constant.value.IsNull()) {
			throw NotImplementedException("Custom escape in SIMILAR TO");
		}
		// take the child of the similar_func
		children.push_back(move(similar_func.children[0]));

		// this looks very odd, but seems to be the way to find out its NOT IN
		bool invert_similar = false;
		if (name == "!~") {
			// NOT SIMILAR TO
			invert_similar = true;
		}
		const auto schema = DEFAULT_SCHEMA;
		const auto regex_function = "regexp_full_match";
		auto result = make_unique<FunctionExpression>(schema, regex_function, move(children));

		if (invert_similar) {
			return make_unique<OperatorExpression>(ExpressionType::OPERATOR_NOT, move(result));
		} else {
			return move(result);
		}
	}
	case duckdb_libpgquery::PG_AEXPR_NOT_DISTINCT: {
		auto left_expr = TransformExpression(root->lexpr);
		auto right_expr = TransformExpression(root->rexpr);
		return make_unique<ComparisonExpression>(ExpressionType::COMPARE_NOT_DISTINCT_FROM, move(left_expr),
		                                         move(right_expr));
	}
	case duckdb_libpgquery::PG_AEXPR_DISTINCT: {
		auto left_expr = TransformExpression(root->lexpr);
		auto right_expr = TransformExpression(root->rexpr);
		return make_unique<ComparisonExpression>(ExpressionType::COMPARE_DISTINCT_FROM, move(left_expr),
		                                         move(right_expr));
	}

	default:
		break;
	}
	auto left_expr = TransformExpression(root->lexpr);
	auto right_expr = TransformExpression(root->rexpr);

	if (!left_expr) {
		// prefix operator
		return TransformUnaryOperator(name, move(right_expr));
	} else if (!right_expr) {
		// postfix operator, only ! is currently supported
		return TransformUnaryOperator(name + "__postfix", move(left_expr));
	} else {
		return TransformBinaryOperator(name, move(left_expr), move(right_expr));
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
	auto expr = make_unique<ParameterExpression>();
	if (node->number < 0) {
		throw ParserException("Parameter numbers cannot be negative");
	}
	if (node->number == 0) {
		expr->parameter_nr = ParamCount() + 1;
	} else {
		expr->parameter_nr = node->number;
	}
	SetParamCount(MaxValue<idx_t>(ParamCount(), expr->parameter_nr));
	return move(expr);
}

} // namespace duckdb




namespace duckdb {

unique_ptr<ParsedExpression> Transformer::TransformPositionalReference(duckdb_libpgquery::PGPositionalReference *node) {
	if (node->position <= 0) {
		throw ParserException("Positional reference node needs to be >= 1");
	}
	auto result = make_unique<PositionalReferenceExpression>(node->position);
	result->query_location = node->location;
	return move(result);
}

} // namespace duckdb





namespace duckdb {

unique_ptr<ParsedExpression> Transformer::TransformSubquery(duckdb_libpgquery::PGSubLink *root) {
	D_ASSERT(root);
	auto subquery_expr = make_unique<SubqueryExpression>();

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
			subquery_expr->comparison_type = NegateComparisionExpression(subquery_expr->comparison_type);
			return make_unique<OperatorExpression>(ExpressionType::OPERATOR_NOT, move(subquery_expr));
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
		auto select_node = make_unique<SelectNode>();

		// ARRAY_AGG(i)
		vector<unique_ptr<ParsedExpression>> children;
		children.push_back(
		    make_unique_base<ParsedExpression, ColumnRefExpression>(subquery_column_alias, subquery_table_alias));
		auto aggr = make_unique<FunctionExpression>("array_agg", move(children));
		// ARRAY_AGG(i) IS NULL
		auto agg_is_null = make_unique<OperatorExpression>(ExpressionType::OPERATOR_IS_NULL, aggr->Copy());
		// empty list
		vector<unique_ptr<ParsedExpression>> list_children;
		auto empty_list = make_unique<FunctionExpression>("list_value", move(list_children));
		// CASE
		auto case_expr = make_unique<CaseExpression>();
		CaseCheck check;
		check.when_expr = move(agg_is_null);
		check.then_expr = move(empty_list);
		case_expr->case_checks.push_back(move(check));
		case_expr->else_expr = move(aggr);

		select_node->select_list.push_back(move(case_expr));

		// FROM (...) tbl(i)
		auto child_subquery = make_unique<SubqueryRef>(move(subquery_expr->subquery), subquery_table_alias);
		child_subquery->column_name_alias.emplace_back(subquery_column_alias);
		select_node->from_table = move(child_subquery);

		auto new_subquery = make_unique<SelectStatement>();
		new_subquery->node = move(select_node);
		subquery_expr->subquery = move(new_subquery);

		subquery_expr->subquery_type = SubqueryType::SCALAR;
		break;
	}
	default:
		throw NotImplementedException("Subquery of type %d not implemented\n", (int)root->subLinkType);
	}
	subquery_expr->query_location = root->location;
	return move(subquery_expr);
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
	default:
		return "(UNKNOWN)";
	}
} // LCOV_EXCL_STOP

} // namespace duckdb


namespace duckdb {

string Transformer::TransformAlias(duckdb_libpgquery::PGAlias *root, vector<string> &column_name_alias) {
	if (!root) {
		return "";
	}
	if (root->colnames) {
		for (auto node = root->colnames->head; node != nullptr; node = node->next) {
			column_name_alias.emplace_back(
			    reinterpret_cast<duckdb_libpgquery::PGValue *>(node->data.ptr_value)->val.str);
		}
	}
	return root->aliasname;
}

} // namespace duckdb






namespace duckdb {

void Transformer::TransformCTE(duckdb_libpgquery::PGWithClause *de_with_clause, CommonTableExpressionMap &cte_map) {
	// TODO: might need to update in case of future lawsuit
	D_ASSERT(de_with_clause);

	D_ASSERT(de_with_clause->ctes);
	for (auto cte_ele = de_with_clause->ctes->head; cte_ele != nullptr; cte_ele = cte_ele->next) {
		auto info = make_unique<CommonTableExpressionInfo>();

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
			info->query = TransformSelect(cte->ctequery);
		}
		D_ASSERT(info->query);
		auto cte_name = string(cte->ctename);

		auto it = cte_map.map.find(cte_name);
		if (it != cte_map.map.end()) {
			// can't have two CTEs with same name
			throw ParserException("Duplicate CTE name \"%s\"", cte_name);
		}
		cte_map.map[cte_name] = move(info);
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
		select = make_unique<SelectStatement>();
		select->node = make_unique_base<QueryNode, RecursiveCTENode>();
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
	expression_map_t<idx_t> map;
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
		auto &func = (FunctionExpression &)*expression;
		if (func.function_name == "row") {
			for (auto &child : func.children) {
				AddGroupByExpression(move(child), map, result, result_set);
			}
			return;
		}
	}
	auto entry = map.map.find(expression.get());
	idx_t result_idx;
	if (entry == map.map.end()) {
		result_idx = result.group_expressions.size();
		map.map[expression.get()] = result_idx;
		result.group_expressions.push_back(move(expression));
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
	AddGroupByExpression(move(expression), map, result, indexes);
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
			result.grouping_sets = move(result_sets);
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
					new_sets.push_back(move(set));
				}
			}
			result.grouping_sets = move(new_sets);
		}
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
			if (order_expression->GetExpressionClass() == ExpressionClass::STAR) {
				auto &star_expr = (StarExpression &)*order_expression;
				D_ASSERT(star_expr.relation_name.empty());
				if (star_expr.columns) {
					throw ParserException("COLUMNS expr is not supported in ORDER BY");
				}
			}
			result.emplace_back(type, null_order, move(order_expression));
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
	auto result = make_unique<SampleOptions>();
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
		result_type = LogicalType::STRUCT(move(children));
	} else if (base_type == LogicalTypeId::MAP) {
		//! We transform MAP<TYPE_KEY, TYPE_VALUE> to STRUCT<LIST<key: TYPE_KEY>, LIST<value: TYPE_VALUE>>

		if (!type_name->typmods || type_name->typmods->length != 2) {
			throw ParserException("Map type needs exactly two entries, key and value type");
		}
		child_list_t<LogicalType> children;
		auto key_type = TransformTypeName((duckdb_libpgquery::PGTypeName *)type_name->typmods->head->data.ptr_value);
		auto value_type = TransformTypeName((duckdb_libpgquery::PGTypeName *)type_name->typmods->tail->data.ptr_value);

		children.push_back({"key", LogicalType::LIST(key_type)});
		children.push_back({"value", LogicalType::LIST(value_type)});

		D_ASSERT(children.size() == 2);

		result_type = LogicalType::MAP(move(children));
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
		result_type = LogicalType::UNION(move(children));
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
			result_type = LogicalType::LIST(move(result_type));
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
	auto result = make_unique<AlterStatement>();

	auto qname = TransformQualifiedName(stmt->sequence);
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
			string owner_schema = "";
			string owner_name = "";
			if (opt_values.size() == 2) {
				owner_schema = opt_values[0];
				owner_name = opt_values[1];
			} else if (opt_values.size() == 1) {
				owner_schema = DEFAULT_SCHEMA;
				owner_name = opt_values[0];
			} else {
				throw InternalException("Wrong argument for %s. Expected either <schema>.<name> or <name>", opt_name);
			}
			auto info = make_unique<ChangeOwnershipInfo>(CatalogType::SEQUENCE_ENTRY, sequence_schema, sequence_name,
			                                             owner_schema, owner_name, stmt->missing_ok);
			result->info = move(info);
		} else {
			throw NotImplementedException("ALTER SEQUENCE option not supported yet!");
		}
	}
	result->info->if_exists = stmt->missing_ok;
	return result;
}
} // namespace duckdb






namespace duckdb {

unique_ptr<AlterStatement> Transformer::TransformAlter(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGAlterTableStmt *>(node);
	D_ASSERT(stmt);
	D_ASSERT(stmt->relation);

	if (stmt->cmds->length != 1) {
		throw ParserException("Only one ALTER command per statement is supported");
	}

	auto result = make_unique<AlterStatement>();
	auto qname = TransformQualifiedName(stmt->relation);

	// first we check the type of ALTER
	for (auto c = stmt->cmds->head; c != nullptr; c = c->next) {
		auto command = reinterpret_cast<duckdb_libpgquery::PGAlterTableCmd *>(lfirst(c));
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
			result->info = make_unique<AddColumnInfo>(qname.schema, qname.name, stmt->missing_ok, move(centry),
			                                          command->missing_ok);
			break;
		}
		case duckdb_libpgquery::PG_AT_DropColumn: {
			bool cascade = command->behavior == duckdb_libpgquery::PG_DROP_CASCADE;

			if (stmt->relkind != duckdb_libpgquery::PG_OBJECT_TABLE) {
				throw ParserException("Dropping columns is only supported for tables");
			}
			result->info = make_unique<RemoveColumnInfo>(qname.schema, qname.name, stmt->missing_ok, command->name,
			                                             command->missing_ok, cascade);
			break;
		}
		case duckdb_libpgquery::PG_AT_ColumnDefault: {
			auto expr = TransformExpression(command->def);

			if (stmt->relkind != duckdb_libpgquery::PG_OBJECT_TABLE) {
				throw ParserException("Alter column's default is only supported for tables");
			}
			result->info =
			    make_unique<SetDefaultInfo>(qname.schema, qname.name, stmt->missing_ok, command->name, move(expr));
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
				auto colref = make_unique<ColumnRefExpression>(command->name);
				expr = make_unique<CastExpression>(column_definition.Type(), move(colref));
			}
			result->info = make_unique<ChangeColumnTypeInfo>(qname.schema, qname.name, stmt->missing_ok, command->name,
			                                                 column_definition.Type(), move(expr));
			break;
		}
		case duckdb_libpgquery::PG_AT_SetNotNull: {
			result->info = make_unique<SetNotNullInfo>(qname.schema, qname.name, stmt->missing_ok, command->name);
			break;
		}
		case duckdb_libpgquery::PG_AT_DropNotNull: {
			result->info = make_unique<DropNotNullInfo>(qname.schema, qname.name, stmt->missing_ok, command->name);
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

unique_ptr<CallStatement> Transformer::TransformCall(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGCallStmt *>(node);
	D_ASSERT(stmt);

	auto result = make_unique<CallStatement>();
	result->function = TransformFuncCall((duckdb_libpgquery::PGFuncCall *)stmt->func);
	return result;
}

} // namespace duckdb




namespace duckdb {

unique_ptr<SQLStatement> Transformer::TransformCheckpoint(duckdb_libpgquery::PGNode *node) {
	auto checkpoint = (duckdb_libpgquery::PGCheckPointStmt *)node;

	vector<unique_ptr<ParsedExpression>> children;
	// transform into "CALL checkpoint()" or "CALL force_checkpoint()"
	auto result = make_unique<CallStatement>();
	result->function =
	    make_unique<FunctionExpression>(checkpoint->force ? "force_checkpoint" : "checkpoint", move(children));
	return move(result);
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
	auto result = make_unique<CopyStatement>();
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
	} else {
		result->select_statement = TransformSelectNode((duckdb_libpgquery::PGSelectStmt *)stmt->query);
	}

	// handle the different options of the COPY statement
	TransformCopyOptions(info, stmt->options);

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

	auto result = make_unique<CreateStatement>();
	auto qname = TransformQualifiedName(stmt->name);

	unique_ptr<MacroFunction> macro_func;

	// function can be null here
	if (stmt->function) {
		auto expression = TransformExpression(stmt->function);
		macro_func = make_unique<ScalarMacroFunction>(move(expression));
	} else if (stmt->query) {
		auto query_node = TransformSelect(stmt->query, true)->node->Copy();
		macro_func = make_unique<TableMacroFunction>(move(query_node));
	}

	auto info =
	    make_unique<CreateMacroInfo>((stmt->function ? CatalogType::MACRO_ENTRY : CatalogType::TABLE_MACRO_ENTRY));
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
				macro_func->default_parameters[param->alias] = move(param);
			} else if (param->GetExpressionClass() == ExpressionClass::COLUMN_REF) {
				// positional parameters
				if (!macro_func->default_parameters.empty()) {
					throw ParserException("Positional parameters cannot come after parameters with a default value!");
				}
				macro_func->parameters.push_back(move(param));
			} else {
				throw ParserException("Invalid parameter: '%s'", param->ToString());
			}
		}
	}

	info->function = move(macro_func);
	result->info = move(info);

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

unique_ptr<CreateStatement> Transformer::TransformCreateIndex(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGIndexStmt *>(node);
	D_ASSERT(stmt);
	auto result = make_unique<CreateStatement>();
	auto info = make_unique<CreateIndexInfo>();
	if (stmt->unique) {
		info->constraint_type = IndexConstraintType::UNIQUE;
	} else {
		info->constraint_type = IndexConstraintType::NONE;
	}

	info->on_conflict = TransformOnConflict(stmt->onconflict);

	for (auto cell = stmt->indexParams->head; cell != nullptr; cell = cell->next) {
		auto index_element = (duckdb_libpgquery::PGIndexElem *)cell->data.ptr_value;
		if (index_element->collation) {
			throw NotImplementedException("Index with collation not supported yet!");
		}
		if (index_element->opclass) {
			throw NotImplementedException("Index with opclass not supported yet!");
		}

		if (index_element->name) {
			// create a column reference expression
			info->expressions.push_back(make_unique<ColumnRefExpression>(index_element->name, stmt->relation->relname));
		} else {
			// parse the index expression
			D_ASSERT(index_element->expr);
			info->expressions.push_back(TransformExpression(index_element->expr));
		}
	}

	info->index_type = StringToIndexType(string(stmt->accessMethod));
	auto tableref = make_unique<BaseTableRef>();
	tableref->table_name = stmt->relation->relname;
	if (stmt->relation->schemaname) {
		tableref->schema_name = stmt->relation->schemaname;
	}
	info->table = move(tableref);
	if (stmt->idxname) {
		info->index_name = stmt->idxname;
	} else {
		throw NotImplementedException("Index without a name not supported yet!");
	}
	for (auto &expr : info->expressions) {
		info->parsed_expressions.emplace_back(expr->Copy());
	}
	result->info = move(info);
	return result;
}

} // namespace duckdb




namespace duckdb {

unique_ptr<CreateStatement> Transformer::TransformCreateSchema(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGCreateSchemaStmt *>(node);
	D_ASSERT(stmt);
	auto result = make_unique<CreateStatement>();
	auto info = make_unique<CreateSchemaInfo>();

	D_ASSERT(stmt->schemaname);
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
	result->info = move(info);
	return result;
}

} // namespace duckdb







namespace duckdb {

unique_ptr<CreateStatement> Transformer::TransformCreateSequence(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGCreateSeqStmt *>(node);

	auto result = make_unique<CreateStatement>();
	auto info = make_unique<CreateSequenceInfo>();

	auto qname = TransformQualifiedName(stmt->sequence);
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
	result->info = move(info);
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
	return make_unique<CollateExpression>(collation, move(child));
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
	auto result = make_unique<CreateStatement>();
	auto info = make_unique<CreateTableInfo>();

	if (stmt->inhRelations) {
		throw NotImplementedException("inherited relations not implemented");
	}
	D_ASSERT(stmt->relation);

	info->schema = INVALID_SCHEMA;
	if (stmt->relation->schemaname) {
		info->schema = stmt->relation->schemaname;
	}
	info->table = stmt->relation->relname;
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
						info->constraints.push_back(move(constraint));
					}
				}
			}
			info->columns.AddColumn(move(centry));
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

	result->info = move(info);
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

	auto result = make_unique<CreateStatement>();
	auto info = make_unique<CreateTableInfo>();
	info->schema = qname.schema;
	info->table = qname.name;
	info->on_conflict = TransformOnConflict(stmt->onconflict);
	info->temporary =
	    stmt->into->rel->relpersistence == duckdb_libpgquery::PGPostgresRelPersistence::PG_RELPERSISTENCE_TEMP;
	info->query = move(query);
	result->info = move(info);
	return result;
}

} // namespace duckdb






namespace duckdb {

vector<string> ReadPgListToString(duckdb_libpgquery::PGList *column_list) {
	vector<string> result;
	if (!column_list) {
		return result;
	}
	for (auto c = column_list->head; c != nullptr; c = lnext(c)) {
		auto target = (duckdb_libpgquery::PGResTarget *)(c->data.ptr_value);
		result.emplace_back(target->name);
	}
	return result;
}

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
	auto result = make_unique<CreateStatement>();
	auto info = make_unique<CreateTypeInfo>();
	info->name = ReadPgListToString(stmt->typeName)[0];
	switch (stmt->kind) {
	case duckdb_libpgquery::PG_NEWTYPE_ENUM: {
		info->internal = false;
		if (stmt->query) {
			// CREATE TYPE mood AS ENUM (SELECT ...)
			D_ASSERT(stmt->vals == nullptr);
			auto query = TransformSelect(stmt->query, false);
			info->query = move(query);
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
		target_type.SetAlias(info->name);
		info->type = target_type;
	} break;

	default:
		throw InternalException("Unknown kind of new type");
	}

	result->info = move(info);
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

	auto result = make_unique<CreateStatement>();
	auto info = make_unique<CreateViewInfo>();

	if (stmt->view->schemaname) {
		info->schema = stmt->view->schemaname;
	}
	info->view_name = stmt->view->relname;
	info->temporary = !stmt->view->relpersistence;
	if (info->temporary) {
		info->schema = TEMP_SCHEMA;
	}
	info->on_conflict = TransformOnConflict(stmt->onconflict);

	info->query = TransformSelect(stmt->query, false);

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
	result->info = move(info);
	return result;
}

} // namespace duckdb




namespace duckdb {

unique_ptr<DeleteStatement> Transformer::TransformDelete(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGDeleteStmt *>(node);
	D_ASSERT(stmt);
	auto result = make_unique<DeleteStatement>();
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
			result->using_clauses.push_back(move(using_entry));
		}
	}

	if (stmt->returningList) {
		Transformer::TransformExpressionList(*(stmt->returningList), result->returning_list);
	}
	return result;
}

} // namespace duckdb



namespace duckdb {

unique_ptr<SQLStatement> Transformer::TransformDrop(duckdb_libpgquery::PGNode *node) {
	auto stmt = (duckdb_libpgquery::PGDropStmt *)(node);
	auto result = make_unique<DropStatement>();
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
	default:
		throw NotImplementedException("Cannot drop this type yet");
	}

	switch (stmt->removeType) {
	case duckdb_libpgquery::PG_OBJECT_SCHEMA:
		info.name = ((duckdb_libpgquery::PGValue *)stmt->objects->head->data.ptr_value)->val.str;
		break;
	case duckdb_libpgquery::PG_OBJECT_TYPE: {
		auto view_list = (duckdb_libpgquery::PGList *)stmt->objects;
		auto target = (duckdb_libpgquery::PGTypeName *)(view_list->head->data.ptr_value);
		info.name = (reinterpret_cast<duckdb_libpgquery::PGValue *>(target->names->tail->data.ptr_value)->val.str);
		break;
	}
	default: {
		auto view_list = (duckdb_libpgquery::PGList *)stmt->objects->head->data.ptr_value;
		if (view_list->length == 2) {
			info.schema = ((duckdb_libpgquery::PGValue *)view_list->head->data.ptr_value)->val.str;
			info.name = ((duckdb_libpgquery::PGValue *)view_list->head->next->data.ptr_value)->val.str;
		} else {
			info.name = ((duckdb_libpgquery::PGValue *)view_list->head->data.ptr_value)->val.str;
		}
		break;
	}
	}
	info.cascade = stmt->behavior == duckdb_libpgquery::PGDropBehavior::PG_DROP_CASCADE;
	info.if_exists = stmt->missing_ok;
	return move(result);
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
	return make_unique<ExplainStatement>(TransformStatement(stmt->query), explain_type);
}

} // namespace duckdb



namespace duckdb {

unique_ptr<ExportStatement> Transformer::TransformExport(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGExportStmt *>(node);
	auto info = make_unique<CopyInfo>();
	info->file_path = stmt->filename;
	info->format = "csv";
	info->is_from = false;
	// handle export options
	TransformCopyOptions(*info, stmt->options);

	return make_unique<ExportStatement>(move(info));
}

} // namespace duckdb



namespace duckdb {

unique_ptr<PragmaStatement> Transformer::TransformImport(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGImportStmt *>(node);
	auto result = make_unique<PragmaStatement>();
	result->info->name = "import_database";
	result->info->parameters.emplace_back(stmt->filename);
	return result;
}

} // namespace duckdb




namespace duckdb {

unique_ptr<TableRef> Transformer::TransformValuesList(duckdb_libpgquery::PGList *list) {
	auto result = make_unique<ExpressionListRef>();
	for (auto value_list = list->head; value_list != nullptr; value_list = value_list->next) {
		auto target = (duckdb_libpgquery::PGList *)(value_list->data.ptr_value);

		vector<unique_ptr<ParsedExpression>> insert_values;
		TransformExpressionList(*target, insert_values);
		if (!result->values.empty()) {
			if (result->values[0].size() != insert_values.size()) {
				throw ParserException("VALUES lists must all be the same length");
			}
		}
		result->values.push_back(move(insert_values));
	}
	result->alias = "valueslist";
	return move(result);
}

unique_ptr<InsertStatement> Transformer::TransformInsert(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGInsertStmt *>(node);
	D_ASSERT(stmt);
	if (stmt->onConflictClause && stmt->onConflictClause->action != duckdb_libpgquery::PG_ONCONFLICT_NONE) {
		throw ParserException("ON CONFLICT IGNORE/UPDATE clauses are not supported");
	}
	if (!stmt->selectStmt) {
		throw ParserException("DEFAULT VALUES clause is not supported!");
	}

	auto result = make_unique<InsertStatement>();
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
	result->select_statement = TransformSelect(stmt->selectStmt, false);

	auto qname = TransformQualifiedName(stmt->relation);
	result->table = qname.name;
	result->schema = qname.schema;
	return result;
}

} // namespace duckdb



namespace duckdb {

unique_ptr<LoadStatement> Transformer::TransformLoad(duckdb_libpgquery::PGNode *node) {
	D_ASSERT(node->type == duckdb_libpgquery::T_PGLoadStmt);
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGLoadStmt *>(node);

	auto load_stmt = make_unique<LoadStatement>();
	auto load_info = make_unique<LoadInfo>();
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
	load_stmt->info = move(load_info);
	return load_stmt;
}

} // namespace duckdb









namespace duckdb {

unique_ptr<SQLStatement> Transformer::TransformPragma(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGPragmaStmt *>(node);

	auto result = make_unique<PragmaStatement>();
	auto &info = *result->info;

	info.name = stmt->name;
	// parse the arguments, if any
	if (stmt->args) {
		for (auto cell = stmt->args->head; cell != nullptr; cell = cell->next) {
			auto node = reinterpret_cast<duckdb_libpgquery::PGNode *>(cell->data.ptr_value);
			auto expr = TransformExpression(node);

			if (expr->type == ExpressionType::COMPARE_EQUAL) {
				auto &comp = (ComparisonExpression &)*expr;
				if (comp.right->type != ExpressionType::VALUE_CONSTANT) {
					throw ParserException("Named parameter requires a constant on the RHS");
				}
				if (comp.left->type != ExpressionType::COLUMN_REF) {
					throw ParserException("Named parameter requires a column reference on the LHS");
				}
				auto &columnref = (ColumnRefExpression &)*comp.left;
				auto &constant = (ConstantExpression &)*comp.right;
				info.named_parameters[columnref.GetName()] = constant.value;
			} else if (node->type == duckdb_libpgquery::T_PGAConst) {
				auto constant = TransformConstant((duckdb_libpgquery::PGAConst *)node);
				info.parameters.push_back(((ConstantExpression &)*constant).value);
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
		auto set_statement = make_unique<SetStatement>(info.name, info.parameters[0], SetScope::AUTOMATIC);
		return move(set_statement);
	}
	case duckdb_libpgquery::PG_PRAGMA_TYPE_CALL:
		break;
	default:
		throw InternalException("Unknown pragma type");
	}

	return move(result);
}

} // namespace duckdb





namespace duckdb {

unique_ptr<PrepareStatement> Transformer::TransformPrepare(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGPrepareStmt *>(node);
	D_ASSERT(stmt);

	if (stmt->argtypes && stmt->argtypes->length > 0) {
		throw NotImplementedException("Prepared statement argument types are not supported, use CAST");
	}

	auto result = make_unique<PrepareStatement>();
	result->name = string(stmt->name);
	result->statement = TransformStatement(stmt->query);
	SetParamCount(0);

	return result;
}

unique_ptr<ExecuteStatement> Transformer::TransformExecute(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGExecuteStmt *>(node);
	D_ASSERT(stmt);

	auto result = make_unique<ExecuteStatement>();
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

	auto result = make_unique<DropStatement>();
	result->info->type = CatalogType::PREPARED_STATEMENT;
	result->info->name = string(stmt->name);
	return result;
}

} // namespace duckdb



namespace duckdb {

unique_ptr<AlterStatement> Transformer::TransformRename(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGRenameStmt *>(node);
	D_ASSERT(stmt);
	D_ASSERT(stmt->relation);

	unique_ptr<AlterInfo> info;

	// first we check the type of ALTER
	switch (stmt->renameType) {
	case duckdb_libpgquery::PG_OBJECT_COLUMN: {
		// change column name

		// get the table and schema
		string schema = INVALID_SCHEMA;
		string table;
		D_ASSERT(stmt->relation->relname);
		if (stmt->relation->relname) {
			table = stmt->relation->relname;
		}
		if (stmt->relation->schemaname) {
			schema = stmt->relation->schemaname;
		}
		// get the old name and the new name
		string old_name = stmt->subname;
		string new_name = stmt->newname;
		info = make_unique<RenameColumnInfo>(schema, table, stmt->missing_ok, old_name, new_name);
		break;
	}
	case duckdb_libpgquery::PG_OBJECT_TABLE: {
		// change table name

		// get the table and schema
		string schema = INVALID_SCHEMA;
		string table;
		D_ASSERT(stmt->relation->relname);
		if (stmt->relation->relname) {
			table = stmt->relation->relname;
		}
		if (stmt->relation->schemaname) {
			schema = stmt->relation->schemaname;
		}
		string new_name = stmt->newname;
		info = make_unique<RenameTableInfo>(schema, table, stmt->missing_ok, new_name);
		break;
	}

	case duckdb_libpgquery::PG_OBJECT_VIEW: {
		// change view name

		// get the view and schema
		string schema = INVALID_SCHEMA;
		string view;
		D_ASSERT(stmt->relation->relname);
		if (stmt->relation->relname) {
			view = stmt->relation->relname;
		}
		if (stmt->relation->schemaname) {
			schema = stmt->relation->schemaname;
		}
		string new_name = stmt->newname;
		info = make_unique<RenameViewInfo>(schema, view, stmt->missing_ok, new_name);
		break;
	}
	case duckdb_libpgquery::PG_OBJECT_DATABASE:
	default:
		throw NotImplementedException("Schema element not supported yet!");
	}
	D_ASSERT(info);
	info->if_exists = stmt->missing_ok;

	auto result = make_unique<AlterStatement>();
	result->info = move(info);
	return result;
}

} // namespace duckdb




namespace duckdb {

unique_ptr<SelectStatement> Transformer::TransformSelect(duckdb_libpgquery::PGNode *node, bool is_select) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGSelectStmt *>(node);
	auto result = make_unique<SelectStatement>();

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

unique_ptr<QueryNode> Transformer::TransformSelectNode(duckdb_libpgquery::PGSelectStmt *stmt) {
	D_ASSERT(stmt->type == duckdb_libpgquery::T_PGSelectStmt);
	auto stack_checker = StackCheck();

	unique_ptr<QueryNode> node;

	switch (stmt->op) {
	case duckdb_libpgquery::PG_SETOP_NONE: {
		node = make_unique<SelectNode>();
		auto result = (SelectNode *)node.get();
		if (stmt->withClause) {
			TransformCTE(reinterpret_cast<duckdb_libpgquery::PGWithClause *>(stmt->withClause), node->cte_map);
		}
		if (stmt->windowClause) {
			for (auto window_ele = stmt->windowClause->head; window_ele != nullptr; window_ele = window_ele->next) {
				auto window_def = reinterpret_cast<duckdb_libpgquery::PGWindowDef *>(window_ele->data.ptr_value);
				D_ASSERT(window_def);
				D_ASSERT(window_def->name);
				auto window_name = StringUtil::Lower(string(window_def->name));

				auto it = window_clauses.find(window_name);
				if (it != window_clauses.end()) {
					throw ParserException("window \"%s\" is already defined", window_name);
				}
				window_clauses[window_name] = window_def;
			}
		}

		// checks distinct clause
		if (stmt->distinctClause != nullptr) {
			auto modifier = make_unique<DistinctModifier>();
			// checks distinct on clause
			auto target = reinterpret_cast<duckdb_libpgquery::PGNode *>(stmt->distinctClause->head->data.ptr_value);
			if (target) {
				//  add the columns defined in the ON clause to the select list
				TransformExpressionList(*stmt->distinctClause, modifier->distinct_on_targets);
			}
			result->modifiers.push_back(move(modifier));
		}

		// do this early so the value lists also have a `FROM`
		if (stmt->valuesLists) {
			// VALUES list, create an ExpressionList
			D_ASSERT(!stmt->fromClause);
			result->from_table = TransformValuesList(stmt->valuesLists);
			result->select_list.push_back(make_unique<StarExpression>());
		} else {
			if (!stmt->targetList) {
				throw ParserException("SELECT clause without selection list");
			}
			// select list
			TransformExpressionList(*stmt->targetList, result->select_list);
			result->from_table = TransformFrom(stmt->fromClause);
		}

		// where
		result->where_clause = TransformExpression(stmt->whereClause);
		// group by
		TransformGroupBy(stmt->groupClause, *result);
		// having
		result->having = TransformExpression(stmt->havingClause);
		// qualify
		result->qualify = TransformExpression(stmt->qualifyClause);
		// sample
		result->sample = TransformSampleOptions(stmt->sampleOptions);
		break;
	}
	case duckdb_libpgquery::PG_SETOP_UNION:
	case duckdb_libpgquery::PG_SETOP_EXCEPT:
	case duckdb_libpgquery::PG_SETOP_INTERSECT:
	case duckdb_libpgquery::PG_SETOP_UNION_BY_NAME: {
		node = make_unique<SetOperationNode>();
		auto result = (SetOperationNode *)node.get();
		if (stmt->withClause) {
			TransformCTE(reinterpret_cast<duckdb_libpgquery::PGWithClause *>(stmt->withClause), node->cte_map);
		}
		result->left = TransformSelectNode(stmt->larg);
		result->right = TransformSelectNode(stmt->rarg);
		if (!result->left || !result->right) {
			throw Exception("Failed to transform setop children.");
		}

		bool select_distinct = true;
		switch (stmt->op) {
		case duckdb_libpgquery::PG_SETOP_UNION:
			select_distinct = !stmt->all;
			result->setop_type = SetOperationType::UNION;
			break;
		case duckdb_libpgquery::PG_SETOP_EXCEPT:
			result->setop_type = SetOperationType::EXCEPT;
			break;
		case duckdb_libpgquery::PG_SETOP_INTERSECT:
			result->setop_type = SetOperationType::INTERSECT;
			break;
		case duckdb_libpgquery::PG_SETOP_UNION_BY_NAME:
			select_distinct = !stmt->all;
			result->setop_type = SetOperationType::UNION_BY_NAME;
			break;
		default:
			throw Exception("Unexpected setop type");
		}
		if (select_distinct) {
			result->modifiers.push_back(make_unique<DistinctModifier>());
		}
		if (stmt->sampleOptions) {
			throw ParserException("SAMPLE clause is only allowed in regular SELECT statements");
		}
		break;
	}
	default:
		throw NotImplementedException("Statement type %d not implemented!", stmt->op);
	}
	// transform the common properties
	// both the set operations and the regular select can have an ORDER BY/LIMIT attached to them
	vector<OrderByNode> orders;
	TransformOrderBy(stmt->sortClause, orders);
	if (!orders.empty()) {
		auto order_modifier = make_unique<OrderModifier>();
		order_modifier->orders = move(orders);
		node->modifiers.push_back(move(order_modifier));
	}
	if (stmt->limitCount || stmt->limitOffset) {
		if (stmt->limitCount && stmt->limitCount->type == duckdb_libpgquery::T_PGLimitPercent) {
			auto limit_percent_modifier = make_unique<LimitPercentModifier>();
			auto expr_node = reinterpret_cast<duckdb_libpgquery::PGLimitPercent *>(stmt->limitCount)->limit_percent;
			limit_percent_modifier->limit = TransformExpression(expr_node);
			if (stmt->limitOffset) {
				limit_percent_modifier->offset = TransformExpression(stmt->limitOffset);
			}
			node->modifiers.push_back(move(limit_percent_modifier));
		} else {
			auto limit_modifier = make_unique<LimitModifier>();
			if (stmt->limitCount) {
				limit_modifier->limit = TransformExpression(stmt->limitCount);
			}
			if (stmt->limitOffset) {
				limit_modifier->offset = TransformExpression(stmt->limitOffset);
			}
			node->modifiers.push_back(move(limit_modifier));
		}
	}
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

} // namespace

unique_ptr<SetStatement> Transformer::TransformSet(duckdb_libpgquery::PGNode *node) {
	D_ASSERT(node->type == duckdb_libpgquery::T_PGVariableSetStmt);
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGVariableSetStmt *>(node);

	if (stmt->kind != duckdb_libpgquery::VariableSetKind::VAR_SET_VALUE) {
		throw ParserException("Can only SET a variable to a value");
	}

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

	return make_unique<SetStatement>(name, value, ToSetScope(stmt->scope));
}

} // namespace duckdb







namespace duckdb {

static void TransformShowName(unique_ptr<PragmaStatement> &result, const string &name) {
	auto &info = *result->info;

	if (name == "\"tables\"") {
		// show all tables
		info.name = "show_tables";
	} else if (name == "__show_tables_expanded") {
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
		auto result = make_unique<ShowStatement>();
		auto &info = *result->info;
		info.is_summary = stmt->is_summary;

		auto select = make_unique<SelectNode>();
		select->select_list.push_back(make_unique<StarExpression>());
		auto basetable = make_unique<BaseTableRef>();
		auto qualified_name = QualifiedName::Parse(stmt->name);
		basetable->schema_name = qualified_name.schema;
		basetable->table_name = qualified_name.name;
		select->from_table = move(basetable);

		info.query = move(select);
		return move(result);
	}

	auto result = make_unique<PragmaStatement>();

	auto show_name = stmt->name;
	TransformShowName(result, show_name);
	return move(result);
}

} // namespace duckdb




namespace duckdb {

unique_ptr<ShowStatement> Transformer::TransformShowSelect(duckdb_libpgquery::PGNode *node) {
	// we capture the select statement of SHOW
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGVariableShowSelectStmt *>(node);
	auto select_stmt = reinterpret_cast<duckdb_libpgquery::PGSelectStmt *>(stmt->stmt);

	auto result = make_unique<ShowStatement>();
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
		return make_unique<TransactionStatement>(TransactionType::BEGIN_TRANSACTION);
	case duckdb_libpgquery::PG_TRANS_STMT_COMMIT:
		return make_unique<TransactionStatement>(TransactionType::COMMIT);
	case duckdb_libpgquery::PG_TRANS_STMT_ROLLBACK:
		return make_unique<TransactionStatement>(TransactionType::ROLLBACK);
	default:
		throw NotImplementedException("Transaction type %d not implemented yet", stmt->kind);
	}
}

} // namespace duckdb



namespace duckdb {

unique_ptr<UpdateStatement> Transformer::TransformUpdate(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGUpdateStmt *>(node);
	D_ASSERT(stmt);

	auto result = make_unique<UpdateStatement>();
	if (stmt->withClause) {
		TransformCTE(reinterpret_cast<duckdb_libpgquery::PGWithClause *>(stmt->withClause), result->cte_map);
	}

	result->table = TransformRangeVar(stmt->relation);
	if (stmt->fromClause) {
		result->from_table = TransformFrom(stmt->fromClause);
	}

	auto root = stmt->targetList;
	for (auto cell = root->head; cell != nullptr; cell = cell->next) {
		auto target = (duckdb_libpgquery::PGResTarget *)(cell->data.ptr_value);
		result->columns.emplace_back(target->name);
		result->expressions.push_back(TransformExpression(target->val));
	}

	// Grab and transform the returning columns from the parser.
	if (stmt->returningList) {
		Transformer::TransformExpressionList(*(stmt->returningList), result->returning_list);
	}

	result->condition = TransformExpression(stmt->whereClause);
	return result;
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

	auto result = make_unique<VacuumStatement>(ParseOptions(stmt->options));

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

	return move(result);
}

} // namespace duckdb



namespace duckdb {

unique_ptr<TableRef> Transformer::TransformRangeVar(duckdb_libpgquery::PGRangeVar *root) {
	auto result = make_unique<BaseTableRef>();

	result->alias = TransformAlias(root->alias, result->column_name_alias);
	if (root->relname) {
		result->table_name = root->relname;
	}
	if (root->schemaname) {
		result->schema_name = root->schemaname;
	}
	if (root->sample) {
		result->sample = TransformSampleOptions(root->sample);
	}
	result->query_location = root->location;
	return move(result);
}

QualifiedName Transformer::TransformQualifiedName(duckdb_libpgquery::PGRangeVar *root) {
	QualifiedName qname;
	if (root->relname) {
		qname.name = root->relname;
	} else {
		qname.name = string();
	}
	if (root->schemaname) {
		qname.schema = root->schemaname;
	} else {
		qname.schema = INVALID_SCHEMA;
	}
	return qname;
}

} // namespace duckdb




namespace duckdb {

unique_ptr<TableRef> Transformer::TransformFrom(duckdb_libpgquery::PGList *root) {
	if (!root) {
		return make_unique<EmptyTableRef>();
	}

	if (root->length > 1) {
		// Cross Product
		auto result = make_unique<CrossProductRef>();
		CrossProductRef *cur_root = result.get();
		idx_t list_size = 0;
		for (auto node = root->head; node != nullptr; node = node->next) {
			auto n = reinterpret_cast<duckdb_libpgquery::PGNode *>(node->data.ptr_value);
			unique_ptr<TableRef> next = TransformTableRefNode(n);
			if (!cur_root->left) {
				cur_root->left = move(next);
			} else if (!cur_root->right) {
				cur_root->right = move(next);
			} else {
				auto old_res = move(result);
				result = make_unique<CrossProductRef>();
				result->left = move(old_res);
				result->right = move(next);
				cur_root = result.get();
			}
			list_size++;
			StackCheck(list_size);
		}
		return move(result);
	}

	auto n = reinterpret_cast<duckdb_libpgquery::PGNode *>(root->head->data.ptr_value);
	return TransformTableRefNode(n);
}

} // namespace duckdb






namespace duckdb {

unique_ptr<TableRef> Transformer::TransformJoin(duckdb_libpgquery::PGJoinExpr *root) {
	auto result = make_unique<JoinRef>();
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
	default: {
		throw NotImplementedException("Join type %d not supported\n", root->jointype);
	}
	}

	// Check the type of left arg and right arg before transform
	result->left = TransformTableRefNode(root->larg);
	result->right = TransformTableRefNode(root->rarg);
	result->is_natural = root->isNatural;
	result->query_location = root->location;

	if (root->usingClause && root->usingClause->length > 0) {
		// usingClause is a list of strings
		for (auto node = root->usingClause->head; node != nullptr; node = node->next) {
			auto target = reinterpret_cast<duckdb_libpgquery::PGNode *>(node->data.ptr_value);
			D_ASSERT(target->type == duckdb_libpgquery::T_PGString);
			auto column_name = string(reinterpret_cast<duckdb_libpgquery::PGValue *>(target)->val.str);
			result->using_columns.push_back(column_name);
		}
		return move(result);
	}

	if (!root->quals && result->using_columns.empty() && !result->is_natural) { // CROSS PRODUCT
		auto cross = make_unique<CrossProductRef>();
		cross->left = move(result->left);
		cross->right = move(result->right);
		return move(cross);
	}
	result->condition = TransformExpression(root->quals);
	return move(result);
}

} // namespace duckdb



namespace duckdb {

unique_ptr<TableRef> Transformer::TransformRangeSubselect(duckdb_libpgquery::PGRangeSubselect *root) {
	Transformer subquery_transformer(this);
	auto subquery = subquery_transformer.TransformSelect(root->subquery);
	if (!subquery) {
		return nullptr;
	}
	if (root->lateral) {
		throw NotImplementedException("LATERAL not implemented");
	}
	auto result = make_unique<SubqueryRef>(move(subquery));
	result->alias = TransformAlias(root->alias, result->column_name_alias);
	if (root->sample) {
		result->sample = TransformSampleOptions(root->sample);
	}
	return move(result);
}

} // namespace duckdb




namespace duckdb {

unique_ptr<TableRef> Transformer::TransformRangeFunction(duckdb_libpgquery::PGRangeFunction *root) {
	if (root->lateral) {
		throw NotImplementedException("LATERAL not implemented");
	}
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
	auto result = make_unique<TableFunctionRef>();
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
	return move(result);
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

Transformer::Transformer(idx_t max_expression_depth_p)
    : parent(nullptr), max_expression_depth(max_expression_depth_p), stack_depth(DConstants::INVALID_INDEX) {
}

Transformer::Transformer(Transformer *parent)
    : parent(parent), max_expression_depth(parent->max_expression_depth), stack_depth(DConstants::INVALID_INDEX) {
}

bool Transformer::TransformParseTree(duckdb_libpgquery::PGList *tree, vector<unique_ptr<SQLStatement>> &statements) {
	InitializeStackCheck();
	for (auto entry = tree->head; entry != nullptr; entry = entry->next) {
		SetParamCount(0);
		auto stmt = TransformStatement((duckdb_libpgquery::PGNode *)entry->data.ptr_value);
		D_ASSERT(stmt);
		stmt->n_param = ParamCount();
		statements.push_back(move(stmt));
	}
	return true;
}

void Transformer::InitializeStackCheck() {
	stack_depth = 0;
}

StackChecker Transformer::StackCheck(idx_t extra_stack) {
	auto node = this;
	while (node->parent) {
		node = node->parent;
	}
	D_ASSERT(node->stack_depth != DConstants::INVALID_INDEX);
	if (node->stack_depth + extra_stack >= max_expression_depth) {
		throw ParserException("Max expression depth limit of %lld exceeded. Use \"SET max_expression_depth TO x\" to "
		                      "increase the maximum expression depth.",
		                      max_expression_depth);
	}
	return StackChecker(*node, extra_stack);
}

unique_ptr<SQLStatement> Transformer::TransformStatement(duckdb_libpgquery::PGNode *stmt) {
	auto result = TransformStatementInternal(stmt);
	result->n_param = ParamCount();
	return result;
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
	default:
		throw NotImplementedException(NodetypeToString(stmt->type));
	}
	return nullptr;
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
			idx_t distance = StringUtil::LevenshteinDistance(name, column_name);
			scores.emplace_back(binding->alias + "." + name, distance);
		}
	}
	return StringUtil::TopNStrings(scores);
}

void BindContext::AddUsingBinding(const string &column_name, UsingColumnSet *set) {
	using_columns[column_name].insert(set);
}

void BindContext::AddUsingBindingSet(unique_ptr<UsingColumnSet> set) {
	using_column_sets.push_back(move(set));
}

bool BindContext::FindUsingBinding(const string &column_name, unordered_set<UsingColumnSet *> **out) {
	auto entry = using_columns.find(column_name);
	if (entry != using_columns.end()) {
		*out = &entry->second;
		return true;
	}
	return false;
}

UsingColumnSet *BindContext::GetUsingBinding(const string &column_name) {
	unordered_set<UsingColumnSet *> *using_bindings;
	if (!FindUsingBinding(column_name, &using_bindings)) {
		return nullptr;
	}
	if (using_bindings->size() > 1) {
		string error = "Ambiguous column reference: column \"" + column_name + "\" can refer to either:\n";
		for (auto &using_set : *using_bindings) {
			string result_bindings;
			for (auto &binding : using_set->bindings) {
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
	for (auto &using_set : *using_bindings) {
		return using_set;
	}
	throw InternalException("Using binding found but no entries");
}

UsingColumnSet *BindContext::GetUsingBinding(const string &column_name, const string &binding_name) {
	if (binding_name.empty()) {
		throw InternalException("GetUsingBinding: expected non-empty binding_name");
	}
	unordered_set<UsingColumnSet *> *using_bindings;
	if (!FindUsingBinding(column_name, &using_bindings)) {
		return nullptr;
	}
	for (auto &using_set : *using_bindings) {
		auto &bindings = using_set->bindings;
		if (bindings.find(binding_name) != bindings.end()) {
			return using_set;
		}
	}
	return nullptr;
}

void BindContext::RemoveUsingBinding(const string &column_name, UsingColumnSet *set) {
	if (!set) {
		return;
	}
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

void BindContext::TransferUsingBinding(BindContext &current_context, UsingColumnSet *current_set,
                                       UsingColumnSet *new_set, const string &binding, const string &using_column) {
	AddUsingBinding(using_column, new_set);
	current_context.RemoveUsingBinding(using_column, current_set);
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
	auto &table_binding = *(TableBinding *)binding;
	auto result = table_binding.ExpandGeneratedColumn(column_name);
	result->alias = column_name;
	return result;
}

unique_ptr<ParsedExpression> BindContext::CreateColumnReference(const string &table_name, const string &column_name) {
	string schema_name;
	return CreateColumnReference(schema_name, table_name, column_name);
}

static bool ColumnIsGenerated(Binding *binding, column_t index) {
	if (binding->binding_type != BindingType::TABLE) {
		return false;
	}
	auto table_binding = (TableBinding *)binding;
	auto catalog_entry = table_binding->GetStandardEntry();
	if (!catalog_entry) {
		return false;
	}
	if (index == COLUMN_IDENTIFIER_ROW_ID) {
		return false;
	}
	D_ASSERT(catalog_entry->type == CatalogType::TABLE_ENTRY);
	auto table_entry = (TableCatalogEntry *)catalog_entry;
	return table_entry->columns.GetColumn(LogicalIndex(index)).Generated();
}

unique_ptr<ParsedExpression> BindContext::CreateColumnReference(const string &schema_name, const string &table_name,
                                                                const string &column_name) {
	string error_message;
	vector<string> names;
	if (!schema_name.empty()) {
		names.push_back(schema_name);
	}
	names.push_back(table_name);
	names.push_back(column_name);

	auto result = make_unique<ColumnRefExpression>(move(names));
	auto binding = GetBinding(table_name, error_message);
	if (!binding) {
		return move(result);
	}
	auto column_index = binding->GetBindingIndex(column_name);
	if (ColumnIsGenerated(binding, column_index)) {
		return ExpandGeneratedColumn(table_name, column_name);
	} else if (column_index < binding->names.size() && binding->names[column_index] != column_name) {
		// because of case insensitivity in the binder we rename the column to the original name
		// as it appears in the binding itself
		result->alias = binding->names[column_index];
	}
	return move(result);
}

Binding *BindContext::GetCTEBinding(const string &ctename) {
	auto match = cte_bindings.find(ctename);
	if (match == cte_bindings.end()) {
		return nullptr;
	}
	return match->second.get();
}

Binding *BindContext::GetBinding(const string &name, string &out_error) {
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
		idx_t entry_column_count = entry.second->names.size();
		if (ref.index == 0) {
			// this is a row id
			table_name = entry.first;
			column_name = "rowid";
			return string();
		}
		if (current_position < entry_column_count) {
			table_name = entry.first;
			column_name = entry.second->names[current_position];
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
	auto column_ref = make_unique<ColumnRefExpression>(column_name, table_name);
	return BindColumn(*column_ref, depth);
}

bool BindContext::CheckExclusionList(StarExpression &expr, Binding *binding, const string &column_name,
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
		new_select_list.push_back(move(new_entry));
		return true;
	}
	return false;
}

bool CheckRegex(const string &column_name, duckdb_re2::RE2 *regex) {
	if (!regex) {
		return true;
	}
	return RE2::PartialMatch(column_name, *regex);
}

void BindContext::GenerateAllColumnExpressions(StarExpression &expr,
                                               vector<unique_ptr<ParsedExpression>> &new_select_list) {
	if (bindings_list.empty()) {
		throw BinderException("SELECT * expression without FROM clause!");
	}
	case_insensitive_set_t excluded_columns;
	if (expr.relation_name.empty()) {
		// SELECT * case
		// bind all expressions of each table in-order
		unique_ptr<duckdb_re2::RE2> regex;
		bool found_match = true;
		if (!expr.regex.empty()) {
			regex = make_unique<duckdb_re2::RE2>(expr.regex);
			if (!regex->error().empty()) {
				throw BinderException("Failed to compile regex \"%s\": %s", expr.regex, regex->error());
			}
			found_match = false;
		}
		unordered_set<UsingColumnSet *> handled_using_columns;
		for (auto &entry : bindings_list) {
			auto binding = entry.second;
			for (auto &column_name : binding->names) {
				if (CheckExclusionList(expr, binding, column_name, new_select_list, excluded_columns)) {
					continue;
				}
				if (!CheckRegex(column_name, regex.get())) {
					continue;
				}
				found_match = true;
				// check if this column is a USING column
				auto using_binding = GetUsingBinding(column_name, binding->alias);
				if (using_binding) {
					// it is!
					// check if we have already emitted the using column
					if (handled_using_columns.find(using_binding) != handled_using_columns.end()) {
						// we have! bail out
						continue;
					}
					// we have not! output the using column
					if (using_binding->primary_binding.empty()) {
						// no primary binding: output a coalesce
						auto coalesce = make_unique<OperatorExpression>(ExpressionType::OPERATOR_COALESCE);
						for (auto &child_binding : using_binding->bindings) {
							coalesce->children.push_back(make_unique<ColumnRefExpression>(column_name, child_binding));
						}
						coalesce->alias = column_name;
						new_select_list.push_back(move(coalesce));
					} else {
						// primary binding: output the qualified column ref
						new_select_list.push_back(
						    make_unique<ColumnRefExpression>(column_name, using_binding->primary_binding));
					}
					handled_using_columns.insert(using_binding);
					continue;
				}
				new_select_list.push_back(make_unique<ColumnRefExpression>(column_name, binding->alias));
			}
		}
		if (!found_match) {
			throw BinderException("No matching columns found that match regex \"%s\"", expr.regex);
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
				if (CheckExclusionList(expr, binding, child.first, new_select_list, excluded_columns)) {
					continue;
				}
				column_names[2] = child.first;
				new_select_list.push_back(make_unique<ColumnRefExpression>(column_names));
			}
		} else {
			for (auto &column_name : binding->names) {
				if (CheckExclusionList(expr, binding, column_name, new_select_list, excluded_columns)) {
					continue;
				}

				new_select_list.push_back(make_unique<ColumnRefExpression>(column_name, binding->alias));
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

void BindContext::AddBinding(const string &alias, unique_ptr<Binding> binding) {
	if (bindings.find(alias) != bindings.end()) {
		throw BinderException("Duplicate alias \"%s\" in query!", alias);
	}
	bindings_list.emplace_back(alias, binding.get());
	bindings[alias] = move(binding);
}

void BindContext::AddBaseTable(idx_t index, const string &alias, const vector<string> &names,
                               const vector<LogicalType> &types, vector<column_t> &bound_column_ids,
                               StandardEntry *entry) {
	AddBinding(alias, make_unique<TableBinding>(alias, types, names, bound_column_ids, entry, index, true));
}

void BindContext::AddTableFunction(idx_t index, const string &alias, const vector<string> &names,
                                   const vector<LogicalType> &types, vector<column_t> &bound_column_ids,
                                   StandardEntry *entry) {
	AddBinding(alias, make_unique<TableBinding>(alias, types, names, bound_column_ids, entry, index));
}

static string AddColumnNameToBinding(const string &base_name, case_insensitive_set_t &current_names) {
	idx_t index = 1;
	string name = base_name;
	while (current_names.find(name) != current_names.end()) {
		name = base_name + ":" + to_string(index++);
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
	AddBinding(alias, make_unique<EntryBinding>(alias, types, names, index, *entry));
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
	AddBinding(alias, make_unique<Binding>(BindingType::BASE, alias, types, names, index));
}

void BindContext::AddCTEBinding(idx_t index, const string &alias, const vector<string> &names,
                                const vector<LogicalType> &types) {
	auto binding = make_shared<Binding>(BindingType::BASE, alias, types, names, index);

	if (cte_bindings.find(alias) != cte_bindings.end()) {
		throw BinderException("Duplicate alias \"%s\" in query!", alias);
	}
	cte_bindings[alias] = move(binding);
	cte_references[alias] = std::make_shared<idx_t>(0);
}

void BindContext::AddContext(BindContext other) {
	for (auto &binding : other.bindings) {
		if (bindings.find(binding.first) != bindings.end()) {
			throw BinderException("Duplicate alias \"%s\" in query!", binding.first);
		}
		bindings[binding.first] = move(binding.second);
	}
	for (auto &binding : other.bindings_list) {
		bindings_list.push_back(move(binding));
	}
	for (auto &entry : other.using_columns) {
		for (auto &alias : entry.second) {
#ifdef DEBUG
			for (auto &other_alias : using_columns[entry.first]) {
				for (auto &col : alias->bindings) {
					D_ASSERT(other_alias->bindings.find(col) == other_alias->bindings.end());
				}
			}
#endif
			using_columns[entry.first].insert(alias);
		}
	}
}

} // namespace duckdb
















namespace duckdb {

static void InvertPercentileFractions(ClientContext &context, unique_ptr<ParsedExpression> &fractions) {
	D_ASSERT(fractions.get());
	D_ASSERT(fractions->expression_class == ExpressionClass::BOUND_EXPRESSION);
	auto &bound = (BoundExpression &)*fractions;

	if (!bound.expr->IsFoldable()) {
		return;
	}

	Value value = ExpressionExecutor::EvaluateScalar(context, *bound.expr);
	if (value.type().id() == LogicalTypeId::LIST) {
		vector<Value> values;
		for (const auto &element_val : ListValue::GetChildren(value)) {
			if (element_val.IsNull()) {
				values.push_back(element_val);
			} else {
				values.push_back(Value::DOUBLE(1 - element_val.GetValue<double>()));
			}
		}
		bound.expr = make_unique<BoundConstantExpression>(Value::LIST(values));
	} else if (value.IsNull()) {
		bound.expr = make_unique<BoundConstantExpression>(value);
	} else {
		bound.expr = make_unique<BoundConstantExpression>(Value::DOUBLE(1 - value.GetValue<double>()));
	}
}

BindResult SelectBinder::BindAggregate(FunctionExpression &aggr, AggregateFunctionCatalogEntry *func, idx_t depth) {
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
	bool invert_fractions = false;
	if (aggr.order_bys && aggr.order_bys->orders.size() == 1) {
		const auto &func_name = aggr.function_name;
		ordered_set_agg = (func_name == "quantile_cont" || func_name == "quantile_disc" || func_name == "mode");

		if (ordered_set_agg) {
			auto &config = DBConfig::GetConfig(context);
			const auto &order = aggr.order_bys->orders[0];
			const auto sense =
			    (order.type == OrderType::ORDER_DEFAULT) ? config.options.default_order_type : order.type;
			invert_fractions = (sense == OrderType::DESCENDING);
		}
	}

	for (auto &child : aggr.children) {
		aggregate_binder.BindChild(child, 0, error);
		// We have to invert the fractions for PERCENTILE_XXXX DESC
		if (error.empty() && invert_fractions) {
			InvertPercentileFractions(context, child);
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
				auto &bound_expr = (BoundExpression &)*aggr.children[i];
				ExtractCorrelatedExpressions(binder, *bound_expr.expr);
			}
			if (aggr.filter) {
				bool success = aggregate_binder.BindCorrelatedColumns(aggr.filter);
				// if there is still an error after this, we could not successfully bind the aggregate
				if (!success) {
					throw BinderException(error);
				}
				auto &bound_expr = (BoundExpression &)*aggr.filter;
				ExtractCorrelatedExpressions(binder, *bound_expr.expr);
			}
			if (aggr.order_bys && !aggr.order_bys->orders.empty()) {
				for (auto &order : aggr.order_bys->orders) {
					bool success = aggregate_binder.BindCorrelatedColumns(order.expression);
					if (!success) {
						throw BinderException(error);
					}
					auto &bound_expr = (BoundExpression &)*order.expression;
					ExtractCorrelatedExpressions(binder, *bound_expr.expr);
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
		auto &child = (BoundExpression &)*aggr.filter;
		bound_filter = BoundCastExpression::AddCastToType(context, move(child.expr), LogicalType::BOOLEAN);
	}

	// all children bound successfully
	// extract the children and types
	vector<LogicalType> types;
	vector<LogicalType> arguments;
	vector<unique_ptr<Expression>> children;

	if (ordered_set_agg) {
		for (auto &order : aggr.order_bys->orders) {
			auto &child = (BoundExpression &)*order.expression;
			types.push_back(child.expr->return_type);
			arguments.push_back(child.expr->return_type);
			children.push_back(move(child.expr));
		}
		aggr.order_bys->orders.clear();
	}

	for (idx_t i = 0; i < aggr.children.size(); i++) {
		auto &child = (BoundExpression &)*aggr.children[i];
		types.push_back(child.expr->return_type);
		arguments.push_back(child.expr->return_type);
		children.push_back(move(child.expr));
	}

	// bind the aggregate
	FunctionBinder function_binder(context);
	idx_t best_function = function_binder.BindFunction(func->name, func->functions, types, error);
	if (best_function == DConstants::INVALID_INDEX) {
		throw BinderException(binder.FormatError(aggr, error));
	}
	// found a matching function!
	auto bound_function = func->functions.GetFunctionByOffset(best_function);

	// Bind any sort columns, unless the aggregate is order-insensitive
	auto order_bys = make_unique<BoundOrderModifier>();
	if (!aggr.order_bys->orders.empty()) {
		auto &config = DBConfig::GetConfig(context);
		for (auto &order : aggr.order_bys->orders) {
			auto &order_expr = (BoundExpression &)*order.expression;
			const auto sense =
			    (order.type == OrderType::ORDER_DEFAULT) ? config.options.default_order_type : order.type;
			const auto null_order = (order.null_order == OrderByNullType::ORDER_DEFAULT)
			                            ? config.options.default_null_order
			                            : order.null_order;
			order_bys->orders.emplace_back(BoundOrderByNode(sense, null_order, move(order_expr.expr)));
		}
	}

	auto aggregate = function_binder.BindAggregateFunction(
	    bound_function, move(children), move(bound_filter),
	    aggr.distinct ? AggregateType::DISTINCT : AggregateType::NON_DISTINCT, move(order_bys));
	if (aggr.export_state) {
		aggregate = ExportAggregateFunction::Bind(move(aggregate));
	}

	// check for all the aggregates if this aggregate already exists
	idx_t aggr_index;
	auto entry = node.aggregate_map.find(aggregate.get());
	if (entry == node.aggregate_map.end()) {
		// new aggregate: insert into aggregate list
		aggr_index = node.aggregates.size();
		node.aggregate_map.insert(make_pair(aggregate.get(), aggr_index));
		node.aggregates.push_back(move(aggregate));
	} else {
		// duplicate aggregate: simplify refer to this aggregate
		aggr_index = entry->second;
	}

	// now create a column reference referring to the aggregate
	auto colref = make_unique<BoundColumnRefExpression>(
	    aggr.alias.empty() ? node.aggregates[aggr_index]->ToString() : aggr.alias,
	    node.aggregates[aggr_index]->return_type, ColumnBinding(node.aggregate_index, aggr_index), depth);
	// move the aggregate expression into the set of bound aggregates
	return BindResult(move(colref));
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
	auto &input = (BoundExpression &)*expr.input;
	auto &lower = (BoundExpression &)*expr.lower;
	auto &upper = (BoundExpression &)*expr.upper;

	auto input_sql_type = input.expr->return_type;
	auto lower_sql_type = lower.expr->return_type;
	auto upper_sql_type = upper.expr->return_type;

	// cast the input types to the same type
	// now obtain the result type of the input types
	auto input_type = BoundComparisonExpression::BindComparison(input_sql_type, lower_sql_type);
	input_type = BoundComparisonExpression::BindComparison(input_type, upper_sql_type);
	// add casts (if necessary)
	input.expr = BoundCastExpression::AddCastToType(context, move(input.expr), input_type);
	lower.expr = BoundCastExpression::AddCastToType(context, move(lower.expr), input_type);
	upper.expr = BoundCastExpression::AddCastToType(context, move(upper.expr), input_type);
	if (input_type.id() == LogicalTypeId::VARCHAR) {
		// handle collation
		auto collation = StringType::GetCollation(input_type);
		input.expr = PushCollation(context, move(input.expr), collation, false);
		lower.expr = PushCollation(context, move(lower.expr), collation, false);
		upper.expr = PushCollation(context, move(upper.expr), collation, false);
	}
	if (!input.expr->HasSideEffects() && !input.expr->HasParameter() && !input.expr->HasSubquery()) {
		// the expression does not have side effects and can be copied: create two comparisons
		// the reason we do this is that individual comparisons are easier to handle in optimizers
		// if both comparisons remain they will be folded together again into a single BETWEEN in the optimizer
		auto left_compare = make_unique<BoundComparisonExpression>(ExpressionType::COMPARE_GREATERTHANOREQUALTO,
		                                                           input.expr->Copy(), move(lower.expr));
		auto right_compare = make_unique<BoundComparisonExpression>(ExpressionType::COMPARE_LESSTHANOREQUALTO,
		                                                            move(input.expr), move(upper.expr));
		return BindResult(make_unique<BoundConjunctionExpression>(ExpressionType::CONJUNCTION_AND, move(left_compare),
		                                                          move(right_compare)));
	} else {
		// expression has side effects: we cannot duplicate it
		// create a bound_between directly
		return BindResult(
		    make_unique<BoundBetweenExpression>(move(input.expr), move(lower.expr), move(upper.expr), true, true));
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
	auto return_type = ((BoundExpression &)*expr.else_expr).expr->return_type;
	for (auto &check : expr.case_checks) {
		auto &then_expr = (BoundExpression &)*check.then_expr;
		return_type = LogicalType::MaxLogicalType(return_type, then_expr.expr->return_type);
	}

	// bind all the individual components of the CASE statement
	auto result = make_unique<BoundCaseExpression>(return_type);
	for (idx_t i = 0; i < expr.case_checks.size(); i++) {
		auto &check = expr.case_checks[i];
		auto &when_expr = (BoundExpression &)*check.when_expr;
		auto &then_expr = (BoundExpression &)*check.then_expr;
		BoundCaseCheck result_check;
		result_check.when_expr =
		    BoundCastExpression::AddCastToType(context, move(when_expr.expr), LogicalType::BOOLEAN);
		result_check.then_expr = BoundCastExpression::AddCastToType(context, move(then_expr.expr), return_type);
		result->case_checks.push_back(move(result_check));
	}
	auto &else_expr = (BoundExpression &)*expr.else_expr;
	result->else_expr = BoundCastExpression::AddCastToType(context, move(else_expr.expr), return_type);
	return BindResult(move(result));
}
} // namespace duckdb






namespace duckdb {

BindResult ExpressionBinder::BindExpression(CastExpression &expr, idx_t depth) {
	// first try to bind the child of the cast expression
	string error = Bind(&expr.child, depth);
	if (!error.empty()) {
		return BindResult(error);
	}
	// FIXME: We can also implement 'hello'::schema.custom_type; and pass by the schema down here.
	// Right now just considering its DEFAULT_SCHEMA always
	Binder::BindLogicalType(context, expr.cast_type, DEFAULT_SCHEMA);
	// the children have been successfully resolved
	auto &child = (BoundExpression &)*expr.child;
	if (expr.try_cast) {
		if (child.expr->return_type == expr.cast_type) {
			// no cast required: type matches
			return BindResult(move(child.expr));
		}
		child.expr = BoundCastExpression::AddCastToType(context, move(child.expr), expr.cast_type, true);
	} else {
		// otherwise add a cast to the target type
		child.expr = BoundCastExpression::AddCastToType(context, move(child.expr), expr.cast_type);
	}
	return BindResult(move(child.expr));
}
} // namespace duckdb




namespace duckdb {

BindResult ExpressionBinder::BindExpression(CollateExpression &expr, idx_t depth) {
	// first try to bind the child of the cast expression
	string error = Bind(&expr.child, depth);
	if (!error.empty()) {
		return BindResult(error);
	}
	auto &child = (BoundExpression &)*expr.child;
	if (child.expr->HasParameter()) {
		throw ParameterNotResolvedException();
	}
	if (child.expr->return_type.id() != LogicalTypeId::VARCHAR) {
		throw BinderException("collations are only supported for type varchar");
	}
	// Validate the collation, but don't use it
	PushCollation(context, child.expr->Copy(), expr.collation, false);
	child.expr->return_type = LogicalType::VARCHAR_COLLATION(expr.collation);
	return BindResult(move(child.expr));
}

} // namespace duckdb
















namespace duckdb {

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
			auto coalesce = make_unique<OperatorExpression>(ExpressionType::OPERATOR_COALESCE);
			for (auto &entry : using_binding->bindings) {
				coalesce->children.push_back(make_unique<ColumnRefExpression>(column_name, entry));
			}
			return move(coalesce);
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
				return make_unique<ColumnRefExpression>(column_name, (*lambda_bindings)[i].alias);
			}
		}
	}

	if (is_macro_column) {
		D_ASSERT(!binder.macro_binding->alias.empty());
		return make_unique<ColumnRefExpression>(column_name, binder.macro_binding->alias);
	}
	// see if it's a column
	if (table_name.empty()) {
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
		auto &colref = (ColumnRefExpression &)*expr;
		string error_message;
		auto new_expr = QualifyColumnName(colref, error_message);
		if (new_expr) {
			if (!expr->alias.empty()) {
				new_expr->alias = expr->alias;
			}
			new_expr->query_location = colref.query_location;
			expr = move(new_expr);
		}
		break;
	}
	case ExpressionType::POSITIONAL_REFERENCE: {
		auto &ref = (PositionalReferenceExpression &)*expr;
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
		auto &lambda_column_ref = (ColumnRefExpression &)*base;
		D_ASSERT(!lambda_column_ref.column_names.empty());

		if (lambda_column_ref.column_names[0].find(DummyBinding::DUMMY_NAME) != string::npos) {
			D_ASSERT(lambda_column_ref.column_names.size() == 2);
			auto lambda_param_name = lambda_column_ref.column_names.back();
			lambda_column_ref.column_names.clear();
			lambda_column_ref.column_names.push_back(lambda_param_name);
		}
	}

	vector<unique_ptr<ParsedExpression>> children;
	children.push_back(move(base));
	children.push_back(make_unique_base<ParsedExpression, ConstantExpression>(Value(move(field_name))));
	auto extract_fun = make_unique<OperatorExpression>(ExpressionType::STRUCT_EXTRACT, move(children));
	return move(extract_fun);
}

unique_ptr<ParsedExpression> ExpressionBinder::CreateStructPack(ColumnRefExpression &colref) {
	D_ASSERT(colref.column_names.size() <= 2);
	string error_message;
	auto &table_name = colref.column_names.back();
	auto binding = binder.bind_context.GetBinding(table_name, error_message);
	if (!binding) {
		return nullptr;
	}
	if (colref.column_names.size() == 2) {
		// "schema_name.table_name"
		auto catalog_entry = binding->GetStandardEntry();
		if (!catalog_entry) {
			return nullptr;
		}
		auto &schema_name = colref.column_names[0];
		if (catalog_entry->schema->name != schema_name || catalog_entry->name != table_name) {
			return nullptr;
		}
	}
	// We found the table, now create the struct_pack expression
	vector<unique_ptr<ParsedExpression>> child_exprs;
	for (const auto &column_name : binding->names) {
		child_exprs.push_back(make_unique<ColumnRefExpression>(column_name, table_name));
	}
	return make_unique<FunctionExpression>("struct_pack", move(child_exprs));
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
			auto new_colref = make_unique<ColumnRefExpression>(colref.column_names[0]);
			string other_error;
			auto qualified_colref = QualifyColumnName(colref.column_names[0], other_error);
			if (qualified_colref) {
				// we could: create a struct extract
				return CreateStructExtract(move(qualified_colref), colref.column_names[1]);
			}
			// we could not! Try creating an implicit struct_pack
			return CreateStructPack(colref);
		}
	} else {
		// two or more dots (i.e. "part1.part2.part3.part4...")
		// -> part1 is a schema, part2 is a table, part3 is a column name, part4 and beyond are struct fields
		// -> part1 is a table, part2 is a column name, part3 and beyond are struct fields
		// -> part1 is a column, part2 and beyond are struct fields

		// we always prefer the most top-level view
		// i.e. in case of multiple resolution options, we resolve in order:
		// -> 1. resolve "part1" as a schema
		// -> 2. resolve "part1" as a table
		// -> 3. resolve "part1" as a column

		unique_ptr<ParsedExpression> result_expr;
		idx_t struct_extract_start;
		// first check if part1 is a schema
		if (binder.HasMatchingBinding(colref.column_names[0], colref.column_names[1], colref.column_names[2],
		                              error_message)) {
			// it is! the column reference is "schema.table.column"
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
				// it is not! return the error
				return nullptr;
			}
			// it is! add the struct extract calls
			struct_extract_start = 1;
		}
		for (idx_t i = struct_extract_start; i < colref.column_names.size(); i++) {
			result_expr = CreateStructExtract(move(result_expr), colref.column_names[i]);
		}
		return result_expr;
	}
}

BindResult ExpressionBinder::BindExpression(ColumnRefExpression &colref_p, idx_t depth) {
	if (binder.GetBindingMode() == BindingMode::EXTRACT_NAMES) {
		return BindResult(make_unique<BoundConstantExpression>(Value(LogicalType::SQLNULL)));
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
		auto result = BindExpression(&expr, depth);
		if (result.expression) {
			result.expression->alias = move(alias);
		}
		return result;
	}

	auto &colref = (ColumnRefExpression &)*expr;
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
				result = (*lambda_bindings)[i].Bind(colref, depth);
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
		bound_columns.push_back(move(ref));
	} else {
		result.error = binder.FormatError(colref_p, result.error);
	}
	return result;
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
	auto &catalog = Catalog::GetCatalog(context);
	auto splits = StringUtil::Split(StringUtil::Lower(collation), ".");
	vector<CollateCatalogEntry *> entries;
	for (auto &collation_argument : splits) {
		auto collation_entry = catalog.GetEntry<CollateCatalogEntry>(context, DEFAULT_SCHEMA, collation_argument);
		if (collation_entry->combinable) {
			entries.insert(entries.begin(), collation_entry);
		} else {
			if (!entries.empty() && !entries.back()->combinable) {
				throw BinderException("Cannot combine collation types \"%s\" and \"%s\"", entries.back()->name,
				                      collation_entry->name);
			}
			entries.push_back(collation_entry);
		}
	}
	for (auto &collation_entry : entries) {
		if (equality_only && collation_entry->not_required_for_equality) {
			continue;
		}
		vector<unique_ptr<Expression>> children;
		children.push_back(move(source));

		FunctionBinder function_binder(context);
		auto function = function_binder.BindScalarFunction(collation_entry->function, move(children));
		source = move(function);
	}
	return source;
}

void ExpressionBinder::TestCollation(ClientContext &context, const string &collation) {
	PushCollation(context, make_unique<BoundConstantExpression>(Value("")), collation);
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
	auto &left = (BoundExpression &)*expr.left;
	auto &right = (BoundExpression &)*expr.right;
	auto left_sql_type = left.expr->return_type;
	auto right_sql_type = right.expr->return_type;
	// cast the input types to the same type
	// now obtain the result type of the input types
	auto input_type = BoundComparisonExpression::BindComparison(left_sql_type, right_sql_type);
	// add casts (if necessary)
	left.expr = BoundCastExpression::AddCastToType(context, move(left.expr), input_type,
	                                               input_type.id() == LogicalTypeId::ENUM);
	right.expr = BoundCastExpression::AddCastToType(context, move(right.expr), input_type,
	                                                input_type.id() == LogicalTypeId::ENUM);

	if (input_type.id() == LogicalTypeId::VARCHAR) {
		// handle collation
		auto collation = StringType::GetCollation(input_type);
		left.expr = PushCollation(context, move(left.expr), collation, expr.type == ExpressionType::COMPARE_EQUAL);
		right.expr = PushCollation(context, move(right.expr), collation, expr.type == ExpressionType::COMPARE_EQUAL);
	}
	// now create the bound comparison expression
	return BindResult(make_unique<BoundComparisonExpression>(expr.type, move(left.expr), move(right.expr)));
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
	auto result = make_unique<BoundConjunctionExpression>(expr.type);
	for (auto &child_expr : expr.children) {
		auto &child = (BoundExpression &)*child_expr;
		result->children.push_back(BoundCastExpression::AddCastToType(context, move(child.expr), LogicalType::BOOLEAN));
	}
	// now create the bound conjunction expression
	return BindResult(move(result));
}

} // namespace duckdb




namespace duckdb {

BindResult ExpressionBinder::BindExpression(ConstantExpression &expr, idx_t depth) {
	return BindResult(make_unique<BoundConstantExpression>(expr.value));
}

} // namespace duckdb













namespace duckdb {

BindResult ExpressionBinder::BindExpression(FunctionExpression &function, idx_t depth,
                                            unique_ptr<ParsedExpression> *expr_ptr) {
	// lookup the function in the catalog
	QueryErrorContext error_context(binder.root_statement, function.query_location);

	if (function.function_name == "unnest" || function.function_name == "unlist") {
		// special case, not in catalog
		// TODO make sure someone does not create such a function OR
		// have unnest live in catalog, too
		return BindUnnest(function, depth);
	}
	auto &catalog = Catalog::GetCatalog(context);
	auto func = catalog.GetEntry(context, CatalogType::SCALAR_FUNCTION_ENTRY, function.schema, function.function_name,
	                             false, error_context);

	switch (func->type) {
	case CatalogType::SCALAR_FUNCTION_ENTRY:
		// scalar function

		// check for lambda parameters, ignore ->> operator (JSON extension)
		if (function.function_name != "->>") {
			for (auto &child : function.children) {
				if (child->expression_class == ExpressionClass::LAMBDA) {
					return BindLambdaFunction(function, (ScalarFunctionCatalogEntry *)func, depth);
				}
			}
		}

		// other scalar function
		return BindFunction(function, (ScalarFunctionCatalogEntry *)func, depth);

	case CatalogType::MACRO_ENTRY:
		// macro function
		return BindMacro(function, (ScalarMacroCatalogEntry *)func, depth, expr_ptr);
	default:
		// aggregate function
		return BindAggregate(function, (AggregateFunctionCatalogEntry *)func, depth);
	}
}

BindResult ExpressionBinder::BindFunction(FunctionExpression &function, ScalarFunctionCatalogEntry *func, idx_t depth) {

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
		return BindResult(make_unique<BoundConstantExpression>(Value(LogicalType::SQLNULL)));
	}

	// all children bound successfully
	// extract the children and types
	vector<unique_ptr<Expression>> children;
	for (idx_t i = 0; i < function.children.size(); i++) {
		auto &child = (BoundExpression &)*function.children[i];
		D_ASSERT(child.expr);
		children.push_back(move(child.expr));
	}

	FunctionBinder function_binder(context);
	unique_ptr<Expression> result =
	    function_binder.BindScalarFunction(*func, move(children), error, function.is_operator, &binder);
	if (!result) {
		throw BinderException(binder.FormatError(function, error));
	}
	return BindResult(move(result));
}

BindResult ExpressionBinder::BindLambdaFunction(FunctionExpression &function, ScalarFunctionCatalogEntry *func,
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
	auto &list_child = (BoundExpression &)*function.children[0];

	if (list_child.expr->return_type.id() != LogicalTypeId::LIST &&
	    list_child.expr->return_type.id() != LogicalTypeId::SQLNULL &&
	    list_child.expr->return_type.id() != LogicalTypeId::UNKNOWN) {
		throw BinderException(" Invalid LIST argument to " + function.function_name + "!");
	}

	LogicalType list_child_type = list_child.expr->return_type.id();
	if (list_child.expr->return_type.id() != LogicalTypeId::SQLNULL &&
	    list_child.expr->return_type.id() != LogicalTypeId::UNKNOWN) {
		list_child_type = ListType::GetChildType(list_child.expr->return_type);
	}

	// bind the lambda parameter
	auto &lambda_expr = (LambdaExpression &)*function.children[1];
	BindResult bind_lambda_result = BindExpression(lambda_expr, depth, true, list_child_type);

	if (bind_lambda_result.HasError()) {
		error = bind_lambda_result.error;
	} else {
		// successfully bound: replace the node with a BoundExpression
		auto alias = function.children[1]->alias;
		function.children[1] = make_unique<BoundExpression>(move(bind_lambda_result.expression));
		auto be = (BoundExpression *)function.children[1].get();
		D_ASSERT(be);
		be->alias = alias;
		if (!alias.empty()) {
			be->expr->alias = alias;
		}
	}

	if (!error.empty()) {
		return BindResult(error);
	}
	if (binder.GetBindingMode() == BindingMode::EXTRACT_NAMES) {
		return BindResult(make_unique<BoundConstantExpression>(Value(LogicalType::SQLNULL)));
	}

	// all children bound successfully
	// extract the children and types
	vector<unique_ptr<Expression>> children;
	for (idx_t i = 0; i < function.children.size(); i++) {
		auto &child = (BoundExpression &)*function.children[i];
		D_ASSERT(child.expr);
		children.push_back(move(child.expr));
	}

	// capture the (lambda) columns
	auto &bound_lambda_expr = (BoundLambdaExpression &)*children.back();
	CaptureLambdaColumns(bound_lambda_expr.captures, list_child_type, bound_lambda_expr.lambda_expr,
	                     children[0]->alias);

	FunctionBinder function_binder(context);
	unique_ptr<Expression> result =
	    function_binder.BindScalarFunction(*func, move(children), error, function.is_operator, &binder);
	if (!result) {
		throw BinderException(binder.FormatError(function, error));
	}

	// remove the lambda expression from the children
	auto &bound_function_expr = (BoundFunctionExpression &)*result;
	auto lambda = move(bound_function_expr.children.back());
	bound_function_expr.children.pop_back();
	auto &bound_lambda = (BoundLambdaExpression &)*lambda;

	// push back the captures into the children vector and the correct return types into the bound_function arguments
	for (auto &capture : bound_lambda.captures) {
		bound_function_expr.function.arguments.push_back(capture->return_type);
		bound_function_expr.children.push_back(move(capture));
	}

	return BindResult(move(result));
}

BindResult ExpressionBinder::BindAggregate(FunctionExpression &expr, AggregateFunctionCatalogEntry *function,
                                           idx_t depth) {
	return BindResult(binder.FormatError(expr, UnsupportedAggregateMessage()));
}

BindResult ExpressionBinder::BindUnnest(FunctionExpression &expr, idx_t depth) {
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
		OperatorExpression arrow_expr(ExpressionType::ARROW, move(lhs_expr), expr.expr->Copy());
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
		expr.params.push_back(move(expr.lhs));
	} else {
		auto &func_expr = (FunctionExpression &)*expr.lhs;
		for (idx_t i = 0; i < func_expr.children.size(); i++) {
			expr.params.push_back(move(func_expr.children[i]));
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

		auto column_ref = (ColumnRefExpression &)*expr.params[i];
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
		auto result = BindExpression(&expr.params[i], depth, false);
		D_ASSERT(!result.HasError());
	}

	auto result = BindExpression(&expr.expr, depth, false);
	lambda_bindings->pop_back();

	// successfully bound a subtree of nested lambdas, set this to nullptr in case other parts of the
	// query also contain lambdas
	if (lambda_bindings->empty()) {
		lambda_bindings = nullptr;
	}

	if (result.HasError()) {
		throw BinderException(result.error);
	}

	return BindResult(make_unique<BoundLambdaExpression>(ExpressionType::LAMBDA, LogicalType::LAMBDA,
	                                                     move(result.expression), params_strings.size()));
}

void ExpressionBinder::TransformCapturedLambdaColumn(unique_ptr<Expression> &original,
                                                     unique_ptr<Expression> &replacement,
                                                     vector<unique_ptr<Expression>> &captures,
                                                     LogicalType &list_child_type, string &alias) {

	// check if the original expression is a lambda parameter
	bool is_lambda_parameter = false;
	if (original->expression_class == ExpressionClass::BOUND_COLUMN_REF) {

		// determine if this is the lambda parameter
		auto &bound_col_ref = (BoundColumnRefExpression &)*original;
		if (bound_col_ref.binding.table_index == DConstants::INVALID_INDEX) {
			is_lambda_parameter = true;
		}
	}

	if (is_lambda_parameter) {
		// this is a lambda parameter, so the replacement refers to the first argument, which is the list
		replacement = make_unique<BoundReferenceExpression>(alias, list_child_type, 0);

	} else {
		// this is not a lambda parameter, so we need to create a new argument for the arguments vector
		replacement =
		    make_unique<BoundReferenceExpression>(original->alias, original->return_type, captures.size() + 1);
		captures.push_back(move(original));
	}
}

void ExpressionBinder::CaptureLambdaColumns(vector<unique_ptr<Expression>> &captures, LogicalType &list_child_type,
                                            unique_ptr<Expression> &expr, string &alias) {

	if (expr->expression_class == ExpressionClass::BOUND_SUBQUERY) {
		throw InvalidInputException("Subqueries are not supported in lambda expressions!");
	}

	// these expression classes do not have children, transform them
	if (expr->expression_class == ExpressionClass::BOUND_CONSTANT ||
	    expr->expression_class == ExpressionClass::BOUND_COLUMN_REF ||
	    expr->expression_class == ExpressionClass::BOUND_PARAMETER) {

		// move the expr because we are going to replace it
		auto original = move(expr);
		unique_ptr<Expression> replacement;

		TransformCapturedLambdaColumn(original, replacement, captures, list_child_type, alias);

		// replace the expression
		expr = move(replacement);

	} else {
		// recursively enumerate the children of the expression
		ExpressionIterator::EnumerateChildren(*expr, [&](unique_ptr<Expression> &child) {
			CaptureLambdaColumns(captures, list_child_type, child, alias);
		});
	}
}

} // namespace duckdb









namespace duckdb {

void ExpressionBinder::ReplaceMacroParametersRecursive(unique_ptr<ParsedExpression> &expr) {
	switch (expr->GetExpressionClass()) {
	case ExpressionClass::COLUMN_REF: {
		// if expr is a parameter, replace it with its argument
		auto &colref = (ColumnRefExpression &)*expr;
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
		auto &sq = ((SubqueryExpression &)*expr).subquery;
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

BindResult ExpressionBinder::BindMacro(FunctionExpression &function, ScalarMacroCatalogEntry *macro_func, idx_t depth,
                                       unique_ptr<ParsedExpression> *expr) {
	// recast function so we can access the scalar member function->expression
	auto &macro_def = (ScalarMacroFunction &)*macro_func->function;

	// validate the arguments and separate positional and default arguments
	vector<unique_ptr<ParsedExpression>> positionals;
	unordered_map<string, unique_ptr<ParsedExpression>> defaults;

	string error =
	    MacroFunction::ValidateArguments(*macro_func->function, macro_func->name, function, positionals, defaults);
	if (!error.empty()) {
		throw BinderException(binder.FormatError(*expr->get(), error));
	}

	// create a MacroBinding to bind this macro's parameters to its arguments
	vector<LogicalType> types;
	vector<string> names;
	// positional parameters
	for (idx_t i = 0; i < macro_def.parameters.size(); i++) {
		types.emplace_back(LogicalType::SQLNULL);
		auto &param = (ColumnRefExpression &)*macro_def.parameters[i];
		names.push_back(param.GetColumnName());
	}
	// default parameters
	for (auto it = macro_def.default_parameters.begin(); it != macro_def.default_parameters.end(); it++) {
		types.emplace_back(LogicalType::SQLNULL);
		names.push_back(it->first);
		// now push the defaults into the positionals
		positionals.push_back(move(defaults[it->first]));
	}
	auto new_macro_binding = make_unique<DummyBinding>(types, names, macro_func->name);
	new_macro_binding->arguments = &positionals;
	macro_binding = new_macro_binding.get();

	// replace current expression with stored macro expression, and replace params
	*expr = macro_def.expression->Copy();
	ReplaceMacroParametersRecursive(*expr);

	// bind the unfolded macro
	return BindExpression(expr, depth);
}

} // namespace duckdb








namespace duckdb {

static LogicalType ResolveNotType(OperatorExpression &op, vector<BoundExpression *> &children) {
	// NOT expression, cast child to BOOLEAN
	D_ASSERT(children.size() == 1);
	children[0]->expr = BoundCastExpression::AddDefaultCastToType(move(children[0]->expr), LogicalType::BOOLEAN);
	return LogicalType(LogicalTypeId::BOOLEAN);
}

static LogicalType ResolveInType(OperatorExpression &op, vector<BoundExpression *> &children) {
	if (children.empty()) {
		throw InternalException("IN requires at least a single child node");
	}
	// get the maximum type from the children
	LogicalType max_type = children[0]->expr->return_type;
	for (idx_t i = 1; i < children.size(); i++) {
		max_type = LogicalType::MaxLogicalType(max_type, children[i]->expr->return_type);
	}

	// cast all children to the same type
	for (idx_t i = 0; i < children.size(); i++) {
		children[i]->expr = BoundCastExpression::AddDefaultCastToType(move(children[i]->expr), max_type);
	}
	// (NOT) IN always returns a boolean
	return LogicalType::BOOLEAN;
}

static LogicalType ResolveOperatorType(OperatorExpression &op, vector<BoundExpression *> &children) {
	switch (op.type) {
	case ExpressionType::OPERATOR_IS_NULL:
	case ExpressionType::OPERATOR_IS_NOT_NULL:
		// IS (NOT) NULL always returns a boolean, and does not cast its children
		if (!children[0]->expr->return_type.IsValid()) {
			throw ParameterNotResolvedException();
		}
		return LogicalType::BOOLEAN;
	case ExpressionType::COMPARE_IN:
	case ExpressionType::COMPARE_NOT_IN:
		return ResolveInType(op, children);
	case ExpressionType::OPERATOR_COALESCE: {
		ResolveInType(op, children);
		return children[0]->expr->return_type;
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
		auto &b_exp = (BoundExpression &)*op.children[0];
		if (b_exp.expr->return_type.id() == LogicalTypeId::MAP) {
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
		auto &extract_exp = (BoundExpression &)*op.children[0];
		auto &name_exp = (BoundExpression &)*op.children[1];
		auto extract_expr_type = extract_exp.expr->return_type.id();
		if (extract_expr_type != LogicalTypeId::STRUCT && extract_expr_type != LogicalTypeId::UNION &&
		    extract_expr_type != LogicalTypeId::SQLNULL) {
			return BindResult(StringUtil::Format(
			    "Cannot extract field %s from expression \"%s\" because it is not a struct or a union",
			    name_exp.ToString(), extract_exp.ToString()));
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
		auto function = make_unique<FunctionExpression>(function_name, move(op.children));
		return BindExpression(*function, depth, nullptr);
	}

	vector<BoundExpression *> children;
	for (idx_t i = 0; i < op.children.size(); i++) {
		D_ASSERT(op.children[i]->expression_class == ExpressionClass::BOUND_EXPRESSION);
		children.push_back((BoundExpression *)op.children[i].get());
	}
	// now resolve the types
	LogicalType result_type = ResolveOperatorType(op, children);
	if (op.type == ExpressionType::OPERATOR_COALESCE) {
		if (children.empty()) {
			throw BinderException("COALESCE needs at least one child");
		}
		if (children.size() == 1) {
			return BindResult(move(children[0]->expr));
		}
	}

	auto result = make_unique<BoundOperatorExpression>(op.type, result_type);
	for (auto &child : children) {
		result->children.push_back(move(child->expr));
	}
	return BindResult(move(result));
}

} // namespace duckdb






namespace duckdb {

BindResult ExpressionBinder::BindExpression(ParameterExpression &expr, idx_t depth) {
	D_ASSERT(expr.parameter_nr > 0);
	auto bound_parameter = make_unique<BoundParameterExpression>(expr.parameter_nr);
	bound_parameter->alias = expr.alias;
	if (!binder.parameters) {
		throw BinderException("Unexpected prepared parameter. This type of statement can't be prepared!");
	}
	auto parameter_idx = expr.parameter_nr;
	// check if a parameter value has already been supplied
	if (parameter_idx <= binder.parameters->parameter_data.size()) {
		// it has! emit a constant directly
		auto &data = binder.parameters->parameter_data[parameter_idx - 1];
		auto constant = make_unique<BoundConstantExpression>(data.value);
		constant->alias = expr.alias;
		return BindResult(move(constant));
	}
	auto entry = binder.parameters->parameters.find(parameter_idx);
	if (entry == binder.parameters->parameters.end()) {
		// no entry yet: create a new one
		auto data = make_shared<BoundParameterData>();
		data->return_type = binder.parameters->GetReturnType(parameter_idx - 1);
		bound_parameter->return_type = data->return_type;
		bound_parameter->parameter_data = data;
		binder.parameters->parameters[parameter_idx] = move(data);
	} else {
		// a prepared statement with this parameter index was already there: use it
		auto &data = entry->second;
		bound_parameter->parameter_data = data;
		bound_parameter->return_type = binder.parameters->GetReturnType(parameter_idx - 1);
	}
	return BindResult(move(bound_parameter));
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

class BoundSubqueryNode : public QueryNode {
public:
	BoundSubqueryNode(shared_ptr<Binder> subquery_binder, unique_ptr<BoundQueryNode> bound_node,
	                  unique_ptr<SelectStatement> subquery)
	    : QueryNode(QueryNodeType::BOUND_SUBQUERY_NODE), subquery_binder(move(subquery_binder)),
	      bound_node(move(bound_node)), subquery(move(subquery)) {
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
		auto prior_subquery = move(expr.subquery);
		expr.subquery = make_unique<SelectStatement>();
		expr.subquery->node =
		    make_unique<BoundSubqueryNode>(move(subquery_binder), move(bound_node), move(prior_subquery));
	}
	// now bind the child node of the subquery
	if (expr.child) {
		// first bind the children of the subquery, if any
		string error = Bind(&expr.child, depth);
		if (!error.empty()) {
			return BindResult(error);
		}
	}
	// both binding the child and binding the subquery was successful
	D_ASSERT(expr.subquery->node->type == QueryNodeType::BOUND_SUBQUERY_NODE);
	auto bound_subquery = (BoundSubqueryNode *)expr.subquery->node.get();
	auto child = (BoundExpression *)expr.child.get();
	auto subquery_binder = move(bound_subquery->subquery_binder);
	auto bound_node = move(bound_subquery->bound_node);
	LogicalType return_type =
	    expr.subquery_type == SubqueryType::SCALAR ? bound_node->types[0] : LogicalType(LogicalTypeId::BOOLEAN);
	if (return_type.id() == LogicalTypeId::UNKNOWN) {
		return_type = LogicalType::SQLNULL;
	}

	auto result = make_unique<BoundSubqueryExpression>(return_type);
	if (expr.subquery_type == SubqueryType::ANY) {
		// ANY comparison
		// cast child and subquery child to equivalent types
		D_ASSERT(bound_node->types.size() == 1);
		auto compare_type = LogicalType::MaxLogicalType(child->expr->return_type, bound_node->types[0]);
		child->expr = BoundCastExpression::AddCastToType(context, move(child->expr), compare_type);
		result->child_type = bound_node->types[0];
		result->child_target = compare_type;
	}
	result->binder = move(subquery_binder);
	result->subquery = move(bound_node);
	result->subquery_type = expr.subquery_type;
	result->child = child ? move(child->expr) : nullptr;
	result->comparison_type = expr.comparison_type;

	return BindResult(move(result));
}

} // namespace duckdb












namespace duckdb {

BindResult SelectBinder::BindUnnest(FunctionExpression &function, idx_t depth) {
	// bind the children of the function expression
	string error;
	if (function.children.size() != 1) {
		return BindResult(binder.FormatError(function, "Unnest() needs exactly one child expressions"));
	}
	BindChild(function.children[0], depth, error);
	if (!error.empty()) {
		// failed to bind
		// try to bind correlated columns manually
		if (!BindCorrelatedColumns(function.children[0])) {
			return BindResult(error);
		}
		auto bound_expr = (BoundExpression *)function.children[0].get();
		ExtractCorrelatedExpressions(binder, *bound_expr->expr);
	}
	auto &child = (BoundExpression &)*function.children[0];
	auto &child_type = child.expr->return_type;

	if (child_type.id() != LogicalTypeId::LIST && child_type.id() != LogicalTypeId::SQLNULL &&
	    child_type.id() != LogicalTypeId::UNKNOWN) {
		return BindResult(binder.FormatError(function, "Unnest() can only be applied to lists and NULL"));
	}

	if (depth > 0) {
		throw BinderException(binder.FormatError(function, "Unnest() for correlated expressions is not supported yet"));
	}

	auto return_type = LogicalType(LogicalTypeId::SQLNULL);
	if (child_type.id() == LogicalTypeId::LIST) {
		return_type = ListType::GetChildType(child_type);
	} else if (child_type.id() == LogicalTypeId::UNKNOWN) {
		throw ParameterNotResolvedException();
	}

	auto result = make_unique<BoundUnnestExpression>(return_type);
	result->child = move(child.expr);

	auto unnest_index = node.unnests.size();
	node.unnests.push_back(move(result));

	// TODO what if we have multiple unnests in the same projection list? ignore for now

	// now create a column reference referring to the unnest
	auto colref = make_unique<BoundColumnRefExpression>(
	    function.alias.empty() ? node.unnests[unnest_index]->ToString() : function.alias, return_type,
	    ColumnBinding(node.unnest_index, unnest_index), depth);

	return BindResult(move(colref));
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

static inline OrderType ResolveOrderType(const DBConfig &config, OrderType type) {
	return (type == OrderType::ORDER_DEFAULT) ? config.options.default_order_type : type;
}

static inline OrderByNullType ResolveNullOrder(const DBConfig &config, OrderByNullType null_order) {
	return (null_order == OrderByNullType::ORDER_DEFAULT) ? config.options.default_null_order : null_order;
}

static unique_ptr<Expression> GetExpression(unique_ptr<ParsedExpression> &expr) {
	if (!expr) {
		return nullptr;
	}
	D_ASSERT(expr.get());
	D_ASSERT(expr->expression_class == ExpressionClass::BOUND_EXPRESSION);
	return move(((BoundExpression &)*expr).expr);
}

static unique_ptr<Expression> CastWindowExpression(unique_ptr<ParsedExpression> &expr, const LogicalType &type) {
	if (!expr) {
		return nullptr;
	}
	D_ASSERT(expr.get());
	D_ASSERT(expr->expression_class == ExpressionClass::BOUND_EXPRESSION);

	auto &bound = (BoundExpression &)*expr;
	bound.expr = BoundCastExpression::AddDefaultCastToType(move(bound.expr), type);

	return move(bound.expr);
}

static LogicalType BindRangeExpression(ClientContext &context, const string &name, unique_ptr<ParsedExpression> &expr,
                                       unique_ptr<ParsedExpression> &order_expr) {

	vector<unique_ptr<Expression>> children;

	D_ASSERT(order_expr.get());
	D_ASSERT(order_expr->expression_class == ExpressionClass::BOUND_EXPRESSION);
	auto &bound_order = (BoundExpression &)*order_expr;
	children.emplace_back(bound_order.expr->Copy());

	D_ASSERT(expr.get());
	D_ASSERT(expr->expression_class == ExpressionClass::BOUND_EXPRESSION);
	auto &bound = (BoundExpression &)*expr;
	children.emplace_back(move(bound.expr));

	string error;
	FunctionBinder function_binder(context);
	auto function = function_binder.BindScalarFunction(DEFAULT_SCHEMA, name, move(children), error, true);
	if (!function) {
		throw BinderException(error);
	}
	bound.expr = move(function);
	return bound.expr->return_type;
}

BindResult SelectBinder::BindWindow(WindowExpression &window, idx_t depth) {
	auto name = window.GetName();

	QueryErrorContext error_context(binder.root_statement, window.query_location);
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
		auto &bound = (BoundExpression &)*child;
		// Add casts for positional arguments
		const auto argno = children.size();
		switch (window.type) {
		case ExpressionType::WINDOW_NTILE:
			// ntile(bigint)
			if (argno == 0) {
				bound.expr = BoundCastExpression::AddCastToType(context, move(bound.expr), LogicalType::BIGINT);
			}
			break;
		case ExpressionType::WINDOW_NTH_VALUE:
			// nth_value(<expr>, index)
			if (argno == 1) {
				bound.expr = BoundCastExpression::AddCastToType(context, move(bound.expr), LogicalType::BIGINT);
			}
		default:
			break;
		}
		types.push_back(bound.expr->return_type);
		children.push_back(move(bound.expr));
	}
	//  Determine the function type.
	LogicalType sql_type;
	unique_ptr<AggregateFunction> aggregate;
	unique_ptr<FunctionData> bind_info;
	if (window.type == ExpressionType::WINDOW_AGGREGATE) {
		//  Look up the aggregate function in the catalog
		auto func =
		    (AggregateFunctionCatalogEntry *)Catalog::GetCatalog(context).GetEntry<AggregateFunctionCatalogEntry>(
		        context, window.schema, window.function_name, false, error_context);
		D_ASSERT(func->type == CatalogType::AGGREGATE_FUNCTION_ENTRY);

		// bind the aggregate
		string error;
		FunctionBinder function_binder(context);
		auto best_function = function_binder.BindFunction(func->name, func->functions, types, error);
		if (best_function == DConstants::INVALID_INDEX) {
			throw BinderException(binder.FormatError(window, error));
		}
		// found a matching function! bind it as an aggregate
		auto bound_function = func->functions.GetFunctionByOffset(best_function);
		auto bound_aggregate = function_binder.BindAggregateFunction(bound_function, move(children));
		// create the aggregate
		aggregate = make_unique<AggregateFunction>(bound_aggregate->function);
		bind_info = move(bound_aggregate->bind_info);
		children = move(bound_aggregate->children);
		sql_type = bound_aggregate->return_type;
	} else {
		// fetch the child of the non-aggregate window function (if any)
		sql_type = ResolveWindowExpressionType(window.type, types);
	}
	auto result = make_unique<BoundWindowExpression>(window.type, sql_type, move(aggregate), move(bind_info));
	result->children = move(children);
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
		range_sense = ResolveOrderType(config, window.orders[0].type);
		const auto name = (range_sense == OrderType::ASCENDING) ? "-" : "+";
		start_type = BindRangeExpression(context, name, window.start_expr, window.orders[0].expression);
	} else if (window.start == WindowBoundary::EXPR_FOLLOWING_RANGE) {
		D_ASSERT(window.orders.size() == 1);
		range_sense = ResolveOrderType(config, window.orders[0].type);
		const auto name = (range_sense == OrderType::ASCENDING) ? "+" : "-";
		start_type = BindRangeExpression(context, name, window.start_expr, window.orders[0].expression);
	}

	LogicalType end_type = LogicalType::BIGINT;
	if (window.end == WindowBoundary::EXPR_PRECEDING_RANGE) {
		D_ASSERT(window.orders.size() == 1);
		range_sense = ResolveOrderType(config, window.orders[0].type);
		const auto name = (range_sense == OrderType::ASCENDING) ? "-" : "+";
		end_type = BindRangeExpression(context, name, window.end_expr, window.orders[0].expression);
	} else if (window.end == WindowBoundary::EXPR_FOLLOWING_RANGE) {
		D_ASSERT(window.orders.size() == 1);
		range_sense = ResolveOrderType(config, window.orders[0].type);
		const auto name = (range_sense == OrderType::ASCENDING) ? "+" : "-";
		end_type = BindRangeExpression(context, name, window.end_expr, window.orders[0].expression);
	}

	// Cast ORDER and boundary expressions to the same type
	if (range_sense != OrderType::INVALID) {
		D_ASSERT(window.orders.size() == 1);

		auto &order_expr = window.orders[0].expression;
		D_ASSERT(order_expr.get());
		D_ASSERT(order_expr->expression_class == ExpressionClass::BOUND_EXPRESSION);
		auto &bound_order = (BoundExpression &)*order_expr;
		auto order_type = bound_order.expr->return_type;
		if (window.start_expr) {
			order_type = LogicalType::MaxLogicalType(order_type, start_type);
		}
		if (window.end_expr) {
			order_type = LogicalType::MaxLogicalType(order_type, end_type);
		}

		// Cast all three to match
		bound_order.expr = BoundCastExpression::AddCastToType(context, move(bound_order.expr), order_type);
		start_type = end_type = order_type;
	}

	for (auto &order : window.orders) {
		auto type = ResolveOrderType(config, order.type);
		auto null_order = ResolveNullOrder(config, order.null_order);
		auto expression = GetExpression(order.expression);
		result->orders.emplace_back(type, null_order, move(expression));
	}

	result->filter_expr = CastWindowExpression(window.filter_expr, LogicalType::BOOLEAN);

	result->start_expr = CastWindowExpression(window.start_expr, start_type);
	result->end_expr = CastWindowExpression(window.end_expr, end_type);
	result->offset_expr = CastWindowExpression(window.offset_expr, LogicalType::BIGINT);
	result->default_expr = CastWindowExpression(window.default_expr, result->return_type);
	result->start = window.start;
	result->end = window.end;

	// create a BoundColumnRef that references this entry
	auto colref = make_unique<BoundColumnRefExpression>(move(name), result->return_type,
	                                                    ColumnBinding(node.window_index, node.windows.size()), depth);
	// move the WINDOW expression into the set of bound windows
	node.windows.push_back(move(result));
	return BindResult(move(colref));
}

} // namespace duckdb








namespace duckdb {

unique_ptr<BoundQueryNode> Binder::BindNode(RecursiveCTENode &statement) {
	auto result = make_unique<BoundRecursiveCTENode>();

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

	return move(result);
}

} // namespace duckdb
























namespace duckdb {

unique_ptr<Expression> Binder::BindOrderExpression(OrderBinder &order_binder, unique_ptr<ParsedExpression> expr) {
	// we treat the Distinct list as a order by
	auto bound_expr = order_binder.Bind(move(expr));
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
		return order_binder.CreateExtraReference(move(delimiter));
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

unique_ptr<BoundResultModifier> Binder::BindLimit(OrderBinder &order_binder, LimitModifier &limit_mod) {
	auto result = make_unique<BoundLimitModifier>();
	if (limit_mod.limit) {
		Value val;
		result->limit = BindDelimiter(context, order_binder, move(limit_mod.limit), LogicalType::BIGINT, val);
		if (!result->limit) {
			result->limit_val = val.IsNull() ? NumericLimits<int64_t>::Maximum() : val.GetValue<int64_t>();
			if (result->limit_val < 0) {
				throw BinderException("LIMIT cannot be negative");
			}
		}
	}
	if (limit_mod.offset) {
		Value val;
		result->offset = BindDelimiter(context, order_binder, move(limit_mod.offset), LogicalType::BIGINT, val);
		if (!result->offset) {
			result->offset_val = val.IsNull() ? 0 : val.GetValue<int64_t>();
			if (result->offset_val < 0) {
				throw BinderException("OFFSET cannot be negative");
			}
		}
	}
	return move(result);
}

unique_ptr<BoundResultModifier> Binder::BindLimitPercent(OrderBinder &order_binder, LimitPercentModifier &limit_mod) {
	auto result = make_unique<BoundLimitPercentModifier>();
	if (limit_mod.limit) {
		Value val;
		result->limit = BindDelimiter(context, order_binder, move(limit_mod.limit), LogicalType::DOUBLE, val);
		if (!result->limit) {
			result->limit_percent = val.IsNull() ? 100 : val.GetValue<double>();
			if (result->limit_percent < 0.0) {
				throw Exception("Limit percentage can't be negative value");
			}
		}
	}
	if (limit_mod.offset) {
		Value val;
		result->offset = BindDelimiter(context, order_binder, move(limit_mod.offset), LogicalType::BIGINT, val);
		if (!result->offset) {
			result->offset_val = val.IsNull() ? 0 : val.GetValue<int64_t>();
		}
	}
	return move(result);
}

void Binder::BindModifiers(OrderBinder &order_binder, QueryNode &statement, BoundQueryNode &result) {
	for (auto &mod : statement.modifiers) {
		unique_ptr<BoundResultModifier> bound_modifier;
		switch (mod->type) {
		case ResultModifierType::DISTINCT_MODIFIER: {
			auto &distinct = (DistinctModifier &)*mod;
			auto bound_distinct = make_unique<BoundDistinctModifier>();
			if (distinct.distinct_on_targets.empty()) {
				for (idx_t i = 0; i < result.names.size(); i++) {
					distinct.distinct_on_targets.push_back(make_unique<ConstantExpression>(Value::INTEGER(1 + i)));
				}
			}
			for (auto &distinct_on_target : distinct.distinct_on_targets) {
				auto expr = BindOrderExpression(order_binder, move(distinct_on_target));
				if (!expr) {
					continue;
				}
				bound_distinct->target_distincts.push_back(move(expr));
			}
			bound_modifier = move(bound_distinct);
			break;
		}
		case ResultModifierType::ORDER_MODIFIER: {
			auto &order = (OrderModifier &)*mod;
			auto bound_order = make_unique<BoundOrderModifier>();
			auto &config = DBConfig::GetConfig(context);
			D_ASSERT(!order.orders.empty());
			if (order.orders[0].expression->type == ExpressionType::STAR) {
				// ORDER BY ALL
				// replace the order list with the maximum order by count
				D_ASSERT(order.orders.size() == 1);
				auto order_type = order.orders[0].type;
				auto null_order = order.orders[0].null_order;

				vector<OrderByNode> new_orders;
				for (idx_t i = 0; i < order_binder.MaxCount(); i++) {
					new_orders.emplace_back(order_type, null_order,
					                        make_unique<ConstantExpression>(Value::INTEGER(i + 1)));
				}
				order.orders = move(new_orders);
			}
			for (auto &order_node : order.orders) {
				auto order_expression = BindOrderExpression(order_binder, move(order_node.expression));
				if (!order_expression) {
					continue;
				}
				auto type =
				    order_node.type == OrderType::ORDER_DEFAULT ? config.options.default_order_type : order_node.type;
				auto null_order = order_node.null_order == OrderByNullType::ORDER_DEFAULT
				                      ? config.options.default_null_order
				                      : order_node.null_order;
				bound_order->orders.emplace_back(type, null_order, move(order_expression));
			}
			if (!bound_order->orders.empty()) {
				bound_modifier = move(bound_order);
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
			result.modifiers.push_back(move(bound_modifier));
		}
	}
}

static void AssignReturnType(unique_ptr<Expression> &expr, const vector<LogicalType> &sql_types,
                             idx_t projection_index) {
	if (!expr) {
		return;
	}
	if (expr->type != ExpressionType::BOUND_COLUMN_REF) {
		return;
	}
	auto &bound_colref = (BoundColumnRefExpression &)*expr;
	bound_colref.return_type = sql_types[bound_colref.binding.column_index];
}

void Binder::BindModifierTypes(BoundQueryNode &result, const vector<LogicalType> &sql_types, idx_t projection_index) {
	for (auto &bound_mod : result.modifiers) {
		switch (bound_mod->type) {
		case ResultModifierType::DISTINCT_MODIFIER: {
			auto &distinct = (BoundDistinctModifier &)*bound_mod;
			if (distinct.target_distincts.empty()) {
				// DISTINCT without a target: push references to the standard select list
				for (idx_t i = 0; i < sql_types.size(); i++) {
					distinct.target_distincts.push_back(
					    make_unique<BoundColumnRefExpression>(sql_types[i], ColumnBinding(projection_index, i)));
				}
			} else {
				// DISTINCT with target list: set types
				for (auto &expr : distinct.target_distincts) {
					D_ASSERT(expr->type == ExpressionType::BOUND_COLUMN_REF);
					auto &bound_colref = (BoundColumnRefExpression &)*expr;
					if (bound_colref.binding.column_index == DConstants::INVALID_INDEX) {
						throw BinderException("Ambiguous name in DISTINCT ON!");
					}
					D_ASSERT(bound_colref.binding.column_index < sql_types.size());
					bound_colref.return_type = sql_types[bound_colref.binding.column_index];
				}
			}
			for (auto &target_distinct : distinct.target_distincts) {
				auto &bound_colref = (BoundColumnRefExpression &)*target_distinct;
				const auto &sql_type = sql_types[bound_colref.binding.column_index];
				if (sql_type.id() == LogicalTypeId::VARCHAR) {
					target_distinct = ExpressionBinder::PushCollation(context, move(target_distinct),
					                                                  StringType::GetCollation(sql_type), true);
				}
			}
			break;
		}
		case ResultModifierType::LIMIT_MODIFIER: {
			auto &limit = (BoundLimitModifier &)*bound_mod;
			AssignReturnType(limit.limit, sql_types, projection_index);
			AssignReturnType(limit.offset, sql_types, projection_index);
			break;
		}
		case ResultModifierType::LIMIT_PERCENT_MODIFIER: {
			auto &limit = (BoundLimitPercentModifier &)*bound_mod;
			AssignReturnType(limit.limit, sql_types, projection_index);
			AssignReturnType(limit.offset, sql_types, projection_index);
			break;
		}
		case ResultModifierType::ORDER_MODIFIER: {
			auto &order = (BoundOrderModifier &)*bound_mod;
			for (auto &order_node : order.orders) {
				auto &expr = order_node.expression;
				D_ASSERT(expr->type == ExpressionType::BOUND_COLUMN_REF);
				auto &bound_colref = (BoundColumnRefExpression &)*expr;
				if (bound_colref.binding.column_index == DConstants::INVALID_INDEX) {
					throw BinderException("Ambiguous name in ORDER BY!");
				}
				D_ASSERT(bound_colref.binding.column_index < sql_types.size());
				const auto &sql_type = sql_types[bound_colref.binding.column_index];
				bound_colref.return_type = sql_types[bound_colref.binding.column_index];
				if (sql_type.id() == LogicalTypeId::VARCHAR) {
					order_node.expression = ExpressionBinder::PushCollation(context, move(order_node.expression),
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

bool Binder::FindStarExpression(ParsedExpression &expr, StarExpression **star) {
	if (expr.GetExpressionClass() == ExpressionClass::STAR) {
		auto current_star = (StarExpression *)&expr;
		if (*star) {
			// we can have multiple
			if (!StarExpression::Equals(*star, current_star)) {
				throw BinderException(
				    FormatError(expr, "Multiple different STAR/COLUMNS in the same expression are not supported"));
			}
			return true;
		}
		*star = current_star;
		return true;
	}
	bool has_star = false;
	ParsedExpressionIterator::EnumerateChildren(expr, [&](ParsedExpression &child_expr) {
		if (FindStarExpression(child_expr, star)) {
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
	if (!FindStarExpression(*expr, &star)) {
		// no star expression: add it as-is
		D_ASSERT(!star);
		new_select_list.push_back(move(expr));
		return;
	}
	D_ASSERT(star);
	vector<unique_ptr<ParsedExpression>> star_list;
	// we have star expressions! expand the list of star expressions
	bind_context.GenerateAllColumnExpressions(*star, star_list);

	// now perform the replacement
	for (idx_t i = 0; i < star_list.size(); i++) {
		auto new_expr = expr->Copy();
		ReplaceStarExpression(new_expr, star_list[i]);
		new_select_list.push_back(move(new_expr));
	}
}

void Binder::ExpandStarExpressions(vector<unique_ptr<ParsedExpression>> &select_list,
                                   vector<unique_ptr<ParsedExpression>> &new_select_list) {
	for (auto &select_element : select_list) {
		ExpandStarExpression(move(select_element), new_select_list);
	}
}

unique_ptr<BoundQueryNode> Binder::BindNode(SelectNode &statement) {
	auto result = make_unique<BoundSelectNode>();
	result->projection_index = GenerateTableIndex();
	result->group_index = GenerateTableIndex();
	result->aggregate_index = GenerateTableIndex();
	result->groupings_index = GenerateTableIndex();
	result->window_index = GenerateTableIndex();
	result->unnest_index = GenerateTableIndex();
	result->prune_index = GenerateTableIndex();

	// first bind the FROM table statement
	result->from_table = Bind(*statement.from_table);

	// bind the sample clause
	if (statement.sample) {
		result->sample_options = move(statement.sample);
	}

	// visit the select list and expand any "*" statements
	vector<unique_ptr<ParsedExpression>> new_select_list;
	ExpandStarExpressions(statement.select_list, new_select_list);

	if (new_select_list.empty()) {
		throw BinderException("SELECT list is empty after resolving * expressions!");
	}
	statement.select_list = move(new_select_list);

	// create a mapping of (alias -> index) and a mapping of (Expression -> index) for the SELECT list
	case_insensitive_map_t<idx_t> alias_map;
	expression_map_t<idx_t> projection_map;
	for (idx_t i = 0; i < statement.select_list.size(); i++) {
		auto &expr = statement.select_list[i];
		result->names.push_back(expr->GetName());
		ExpressionBinder::QualifyColumnNames(*this, expr);
		if (!expr->alias.empty()) {
			alias_map[expr->alias] = i;
			result->names[i] = expr->alias;
		}
		projection_map[expr.get()] = i;
		result->original_expressions.push_back(expr->Copy());
	}
	result->column_count = statement.select_list.size();

	// first visit the WHERE clause
	// the WHERE clause happens before the GROUP BY, PROJECTION or HAVING clauses
	if (statement.where_clause) {
		ColumnAliasBinder alias_binder(*result, alias_map);
		WhereBinder where_binder(*this, context, &alias_binder);
		unique_ptr<ParsedExpression> condition = move(statement.where_clause);
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
			bound_expr =
			    ExpressionBinder::PushCollation(context, move(bound_expr), StringType::GetCollation(group_type), true);
			result->groups.group_expressions.push_back(move(bound_expr));

			// in the unbound expression we DO bind the table names of any ColumnRefs
			// we do this to make sure that "table.a" and "a" are treated the same
			// if we wouldn't do this then (SELECT test.a FROM test GROUP BY a) would not work because "test.a" <> "a"
			// hence we convert "a" -> "test.a" in the unbound expression
			unbound_groups[i] = move(group_binder.unbound_expression);
			ExpressionBinder::QualifyColumnNames(*this, unbound_groups[i]);
			info.map[unbound_groups[i].get()] = i;
		}
	}
	result->groups.grouping_sets = move(statement.groups.grouping_sets);

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
	for (idx_t i = 0; i < statement.select_list.size(); i++) {
		bool is_window = statement.select_list[i]->IsWindow();
		LogicalType result_type;
		auto expr = select_binder.Bind(statement.select_list[i], &result_type);
		if (statement.aggregate_handling == AggregateHandling::FORCE_AGGREGATES && select_binder.HasBoundColumns()) {
			if (select_binder.BoundAggregates()) {
				throw BinderException("Cannot mix aggregates with non-aggregated columns!");
			}
			if (is_window) {
				throw BinderException("Cannot group on a window clause");
			}
			// we are forcing aggregates, and the node has columns bound
			// this entry becomes a group
			auto group_ref = make_unique<BoundColumnRefExpression>(
			    expr->return_type, ColumnBinding(result->group_index, result->groups.group_expressions.size()));
			result->groups.group_expressions.push_back(move(expr));
			expr = move(group_ref);
		}
		result->select_list.push_back(move(expr));
		if (i < result->column_count) {
			result->types.push_back(result_type);
		}
		internal_sql_types.push_back(result_type);
		if (statement.aggregate_handling == AggregateHandling::FORCE_AGGREGATES) {
			select_binder.ResetBindings();
		}
	}
	result->need_prune = result->select_list.size() > result->column_count;

	// in the normal select binder, we bind columns as if there is no aggregation
	// i.e. in the query [SELECT i, SUM(i) FROM integers;] the "i" will be bound as a normal column
	// since we have an aggregation, we need to either (1) throw an error, or (2) wrap the column in a FIRST() aggregate
	// we choose the former one [CONTROVERSIAL: this is the PostgreSQL behavior]
	if (!result->groups.group_expressions.empty() || !result->aggregates.empty() || statement.having ||
	    !result->groups.grouping_sets.empty()) {
		if (statement.aggregate_handling == AggregateHandling::NO_AGGREGATES_ALLOWED) {
			throw BinderException("Aggregates cannot be present in a Project relation!");
		} else if (statement.aggregate_handling == AggregateHandling::STANDARD_HANDLING) {
			if (select_binder.HasBoundColumns()) {
				auto &bound_columns = select_binder.GetBoundColumns();
				string error;
				error = "column \"%s\" must appear in the GROUP BY clause or must be part of an aggregate function.";
				error += "\nEither add it to the GROUP BY list, or use \"ANY_VALUE(%s)\" if the exact value of \"%s\" "
				         "is not important.";
				throw BinderException(FormatError(bound_columns[0].query_location, error, bound_columns[0].name,
				                                  bound_columns[0].name, bound_columns[0].name));
			}
		}
	}

	// QUALIFY clause requires at least one window function to be specified in at least one of the SELECT column list or
	// the filter predicate of the QUALIFY clause
	if (statement.qualify && result->windows.empty()) {
		throw BinderException("at least one window function must appear in the SELECT column or QUALIFY clause");
	}

	// now that the SELECT list is bound, we set the types of DISTINCT/ORDER BY expressions
	BindModifierTypes(*result, internal_sql_types, result->projection_index);
	return move(result);
}

} // namespace duckdb












namespace duckdb {

static void GatherAliases(BoundQueryNode &node, case_insensitive_map_t<idx_t> &aliases,
                          expression_map_t<idx_t> &expressions, const vector<idx_t> &reorder_idx) {
	if (node.type == QueryNodeType::SET_OPERATION_NODE) {
		// setop, recurse
		auto &setop = (BoundSetOperationNode &)node;

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
		auto &select = (BoundSelectNode &)node;
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
			auto expr_entry = expressions.find(expr.get());
			if (expr_entry != expressions.end()) {
				// the node is in there
				// repeat the same as with the alias: if there is an ambiguity we insert "-1"
				if (expr_entry->second != index) {
					expressions[expr.get()] = DConstants::INVALID_INDEX;
				}
			} else {
				// not in there yet, just place it in there
				expressions[expr.get()] = index;
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

	result.left_reorder_idx = move(left_reorder_idx);
	result.right_reorder_idx = move(right_reorder_idx);

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
				left_reorder_expr = make_unique<BoundColumnRefExpression>(
				    left_node->types[left_index->second], ColumnBinding(left_node->GetRootIndex(), left_index->second));
				right_reorder_expr = make_unique<BoundColumnRefExpression>(
				    right_node->types[right_index->second],
				    ColumnBinding(right_node->GetRootIndex(), right_index->second));
			} else if (left_exist) {
				left_reorder_expr = make_unique<BoundColumnRefExpression>(
				    left_node->types[left_index->second], ColumnBinding(left_node->GetRootIndex(), left_index->second));
				// create null value here
				right_reorder_expr = make_unique<BoundConstantExpression>(Value(result.types[i]));
			} else {
				D_ASSERT(right_exist);
				left_reorder_expr = make_unique<BoundConstantExpression>(Value(result.types[i]));
				right_reorder_expr = make_unique<BoundColumnRefExpression>(
				    right_node->types[right_index->second],
				    ColumnBinding(right_node->GetRootIndex(), right_index->second));
			}
			result.left_reorder_exprs.push_back(move(left_reorder_expr));
			result.right_reorder_exprs.push_back(move(right_reorder_expr));
		}
	}
}

unique_ptr<BoundQueryNode> Binder::BindNode(SetOperationNode &statement) {
	auto result = make_unique<BoundSetOperationNode>();
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
		expression_map_t<idx_t> expression_map;

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
	return move(result);
}

} // namespace duckdb
















namespace duckdb {

unique_ptr<QueryNode> Binder::BindTableMacro(FunctionExpression &function, TableMacroCatalogEntry *macro_func,
                                             idx_t depth) {

	auto &macro_def = (TableMacroFunction &)*macro_func->function;
	auto node = macro_def.query_node->Copy();

	// auto &macro_def = *macro_func->function;

	// validate the arguments and separate positional and default arguments
	vector<unique_ptr<ParsedExpression>> positionals;
	unordered_map<string, unique_ptr<ParsedExpression>> defaults;
	string error =
	    MacroFunction::ValidateArguments(*macro_func->function, macro_func->name, function, positionals, defaults);
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
		auto &param = (ColumnRefExpression &)*macro_def.parameters[i];
		names.push_back(param.GetColumnName());
	}
	// default parameters
	for (auto it = macro_def.default_parameters.begin(); it != macro_def.default_parameters.end(); it++) {
		types.emplace_back(LogicalType::SQLNULL);
		names.push_back(it->first);
		// now push the defaults into the positionals
		positionals.push_back(move(defaults[it->first]));
	}
	auto new_macro_binding = make_unique<DummyBinding>(types, names, macro_func->name);
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
			auto distinct = make_unique<LogicalDistinct>(move(bound.target_distincts));
			distinct->AddChild(move(root));
			root = move(distinct);
			break;
		}
		case ResultModifierType::ORDER_MODIFIER: {
			auto &bound = (BoundOrderModifier &)*mod;
			auto order = make_unique<LogicalOrder>(move(bound.orders));
			order->AddChild(move(root));
			root = move(order);
			break;
		}
		case ResultModifierType::LIMIT_MODIFIER: {
			auto &bound = (BoundLimitModifier &)*mod;
			auto limit =
			    make_unique<LogicalLimit>(bound.limit_val, bound.offset_val, move(bound.limit), move(bound.offset));
			limit->AddChild(move(root));
			root = move(limit);
			break;
		}
		case ResultModifierType::LIMIT_PERCENT_MODIFIER: {
			auto &bound = (BoundLimitPercentModifier &)*mod;
			auto limit = make_unique<LogicalLimitPercent>(bound.limit_percent, bound.offset_val, move(bound.limit),
			                                              move(bound.offset));
			limit->AddChild(move(root));
			root = move(limit);
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
	left_node = CastLogicalOperatorToTypes(node.left->types, node.types, move(left_node));
	right_node = CastLogicalOperatorToTypes(node.right->types, node.types, move(right_node));

	if (!node.right_binder->bind_context.cte_references[node.ctename] ||
	    *node.right_binder->bind_context.cte_references[node.ctename] == 0) {
		auto root = make_unique<LogicalSetOperation>(node.setop_index, node.types.size(), move(left_node),
		                                             move(right_node), LogicalOperatorType::LOGICAL_UNION);
		return VisitQueryNode(node, move(root));
	}
	auto root = make_unique<LogicalRecursiveCTE>(node.setop_index, node.types.size(), node.union_all, move(left_node),
	                                             move(right_node), LogicalOperatorType::LOGICAL_RECURSIVE_CTE);

	return VisitQueryNode(node, move(root));
}

} // namespace duckdb










namespace duckdb {

unique_ptr<LogicalOperator> Binder::PlanFilter(unique_ptr<Expression> condition, unique_ptr<LogicalOperator> root) {
	PlanSubqueries(&condition, &root);
	auto filter = make_unique<LogicalFilter>(move(condition));
	filter->AddChild(move(root));
	return move(filter);
}

unique_ptr<LogicalOperator> Binder::CreatePlan(BoundSelectNode &statement) {
	unique_ptr<LogicalOperator> root;
	D_ASSERT(statement.from_table);
	root = CreatePlan(*statement.from_table);
	D_ASSERT(root);

	// plan the sample clause
	if (statement.sample_options) {
		root = make_unique<LogicalSample>(move(statement.sample_options), move(root));
	}

	if (statement.where_clause) {
		root = PlanFilter(move(statement.where_clause), move(root));
	}

	if (!statement.aggregates.empty() || !statement.groups.group_expressions.empty()) {
		if (!statement.groups.group_expressions.empty()) {
			// visit the groups
			for (auto &group : statement.groups.group_expressions) {
				PlanSubqueries(&group, &root);
			}
		}
		// now visit all aggregate expressions
		for (auto &expr : statement.aggregates) {
			PlanSubqueries(&expr, &root);
		}
		// finally create the aggregate node with the group_index and aggregate_index as obtained from the binder
		auto aggregate =
		    make_unique<LogicalAggregate>(statement.group_index, statement.aggregate_index, move(statement.aggregates));
		aggregate->groups = move(statement.groups.group_expressions);
		aggregate->groupings_index = statement.groupings_index;
		aggregate->grouping_sets = move(statement.groups.grouping_sets);
		aggregate->grouping_functions = move(statement.grouping_functions);

		aggregate->AddChild(move(root));
		root = move(aggregate);
	} else if (!statement.groups.grouping_sets.empty()) {
		// edge case: we have grouping sets but no groups or aggregates
		// this can only happen if we have e.g. select 1 from tbl group by ();
		// just output a dummy scan
		root = make_unique_base<LogicalOperator, LogicalDummyScan>(statement.group_index);
	}

	if (statement.having) {
		PlanSubqueries(&statement.having, &root);
		auto having = make_unique<LogicalFilter>(move(statement.having));

		having->AddChild(move(root));
		root = move(having);
	}

	if (!statement.windows.empty()) {
		auto win = make_unique<LogicalWindow>(statement.window_index);
		win->expressions = move(statement.windows);
		// visit the window expressions
		for (auto &expr : win->expressions) {
			PlanSubqueries(&expr, &root);
		}
		D_ASSERT(!win->expressions.empty());
		win->AddChild(move(root));
		root = move(win);
	}

	if (statement.qualify) {
		PlanSubqueries(&statement.qualify, &root);
		auto qualify = make_unique<LogicalFilter>(move(statement.qualify));

		qualify->AddChild(move(root));
		root = move(qualify);
	}

	if (!statement.unnests.empty()) {
		auto unnest = make_unique<LogicalUnnest>(statement.unnest_index);
		unnest->expressions = move(statement.unnests);
		// visit the unnest expressions
		for (auto &expr : unnest->expressions) {
			PlanSubqueries(&expr, &root);
		}
		D_ASSERT(!unnest->expressions.empty());
		unnest->AddChild(move(root));
		root = move(unnest);
	}

	for (auto &expr : statement.select_list) {
		PlanSubqueries(&expr, &root);
	}

	// create the projection
	auto proj = make_unique<LogicalProjection>(statement.projection_index, move(statement.select_list));
	auto &projection = *proj;
	proj->AddChild(move(root));
	root = move(proj);

	// finish the plan by handling the elements of the QueryNode
	root = VisitQueryNode(statement, move(root));

	// add a prune node if necessary
	if (statement.need_prune) {
		D_ASSERT(root);
		vector<unique_ptr<Expression>> prune_expressions;
		for (idx_t i = 0; i < statement.column_count; i++) {
			prune_expressions.push_back(make_unique<BoundColumnRefExpression>(
			    projection.expressions[i]->return_type, ColumnBinding(statement.projection_index, i)));
		}
		auto prune = make_unique<LogicalProjection>(statement.prune_index, move(prune_expressions));
		prune->AddChild(move(root));
		root = move(prune);
	}
	return root;
}

} // namespace duckdb







namespace duckdb {

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
				    BoundCastExpression::AddCastToType(context, move(node->expressions[i]), target_types[i]);
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
			unique_ptr<Expression> result = make_unique<BoundColumnRefExpression>(source_types[i], setop_columns[i]);
			if (source_types[i] != target_types[i]) {
				// add a cast only if the source and target types are not equivalent
				result = BoundCastExpression::AddCastToType(context, move(result), target_types[i]);
			}
			select_list.push_back(move(result));
		}
		auto projection = make_unique<LogicalProjection>(GenerateTableIndex(), move(select_list));
		projection->children.push_back(move(op));
		return move(projection);
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

		auto left_projection = make_unique<LogicalProjection>(GenerateTableIndex(), move(node.left_reorder_exprs));
		left_projection->children.push_back(move(left_node));
		left_node = move(left_projection);

		auto right_projection = make_unique<LogicalProjection>(GenerateTableIndex(), move(node.right_reorder_exprs));
		right_projection->children.push_back(move(right_node));
		right_node = move(right_projection);

		left_node = CastLogicalOperatorToTypes(left_types, node.types, move(left_node));
		right_node = CastLogicalOperatorToTypes(right_types, node.types, move(right_node));
	} else {
		left_node = CastLogicalOperatorToTypes(node.left->types, node.types, move(left_node));
		right_node = CastLogicalOperatorToTypes(node.right->types, node.types, move(right_node));
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

	auto root = make_unique<LogicalSetOperation>(node.setop_index, node.types.size(), move(left_node), move(right_node),
	                                             logical_type);

	return VisitQueryNode(node, move(root));
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
		auto limit = make_unique<LogicalLimit>(1, 0, nullptr, nullptr);
		limit->AddChild(move(plan));
		plan = move(limit);

		// now we push a COUNT(*) aggregate onto the limit, this will be either 0 or 1 (EXISTS or NOT EXISTS)
		auto count_star_fun = CountStarFun::GetFunction();

		FunctionBinder function_binder(binder.context);
		auto count_star =
		    function_binder.BindAggregateFunction(count_star_fun, {}, nullptr, AggregateType::NON_DISTINCT);
		auto idx_type = count_star->return_type;
		vector<unique_ptr<Expression>> aggregate_list;
		aggregate_list.push_back(move(count_star));
		auto aggregate_index = binder.GenerateTableIndex();
		auto aggregate =
		    make_unique<LogicalAggregate>(binder.GenerateTableIndex(), aggregate_index, move(aggregate_list));
		aggregate->AddChild(move(plan));
		plan = move(aggregate);

		// now we push a projection with a comparison to 1
		auto left_child = make_unique<BoundColumnRefExpression>(idx_type, ColumnBinding(aggregate_index, 0));
		auto right_child = make_unique<BoundConstantExpression>(Value::Numeric(idx_type, 1));
		auto comparison =
		    make_unique<BoundComparisonExpression>(ExpressionType::COMPARE_EQUAL, move(left_child), move(right_child));

		vector<unique_ptr<Expression>> projection_list;
		projection_list.push_back(move(comparison));
		auto projection_index = binder.GenerateTableIndex();
		auto projection = make_unique<LogicalProjection>(projection_index, move(projection_list));
		projection->AddChild(move(plan));
		plan = move(projection);

		// we add it to the main query by adding a cross product
		// FIXME: should use something else besides cross product as we always add only one scalar constant
		root = LogicalCrossProduct::Create(move(root), move(plan));

		// we replace the original subquery with a ColumnRefExpression referring to the result of the projection (either
		// TRUE or FALSE)
		return make_unique<BoundColumnRefExpression>(expr.GetName(), LogicalType::BOOLEAN,
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
		auto limit = make_unique<LogicalLimit>(1, 0, nullptr, nullptr);
		limit->AddChild(move(plan));
		plan = move(limit);

		// we push an aggregate that returns the FIRST element
		vector<unique_ptr<Expression>> expressions;
		auto bound = make_unique<BoundColumnRefExpression>(expr.return_type, ColumnBinding(table_idx, 0));
		vector<unique_ptr<Expression>> first_children;
		first_children.push_back(move(bound));

		FunctionBinder function_binder(binder.context);
		auto first_agg = function_binder.BindAggregateFunction(
		    FirstFun::GetFunction(expr.return_type), move(first_children), nullptr, AggregateType::NON_DISTINCT);

		expressions.push_back(move(first_agg));
		auto aggr_index = binder.GenerateTableIndex();
		auto aggr = make_unique<LogicalAggregate>(binder.GenerateTableIndex(), aggr_index, move(expressions));
		aggr->AddChild(move(plan));
		plan = move(aggr);

		// in the uncorrelated case, we add the value to the main query through a cross product
		// FIXME: should use something else besides cross product as we always add only one scalar constant and cross
		// product is not optimized for this.
		D_ASSERT(root);
		root = LogicalCrossProduct::Create(move(root), move(plan));

		// we replace the original subquery with a BoundColumnRefExpression referring to the first result of the
		// aggregation
		return make_unique<BoundColumnRefExpression>(expr.GetName(), expr.return_type, ColumnBinding(aggr_index, 0));
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
		auto join = make_unique<LogicalComparisonJoin>(JoinType::MARK);
		join->mark_index = mark_index;
		join->AddChild(move(root));
		join->AddChild(move(plan));
		// create the JOIN condition
		JoinCondition cond;
		cond.left = move(expr.child);
		cond.right = BoundCastExpression::AddDefaultCastToType(
		    make_unique<BoundColumnRefExpression>(expr.child_type, plan_columns[0]), expr.child_target);
		cond.comparison = expr.comparison_type;
		join->conditions.push_back(move(cond));
		root = move(join);

		// we replace the original subquery with a BoundColumnRefExpression referring to the mark column
		return make_unique<BoundColumnRefExpression>(expr.GetName(), expr.return_type, ColumnBinding(mark_index, 0));
	}
	}
}

static unique_ptr<LogicalDelimJoin> CreateDuplicateEliminatedJoin(vector<CorrelatedColumnInfo> &correlated_columns,
                                                                  JoinType join_type,
                                                                  unique_ptr<LogicalOperator> original_plan,
                                                                  bool perform_delim) {
	auto delim_join = make_unique<LogicalDelimJoin>(join_type);
	if (!perform_delim) {
		// if we are not performing a delim join, we push a row_number() OVER() window operator on the LHS
		// and perform all duplicate elimination on that row number instead
		D_ASSERT(correlated_columns[0].type.id() == LogicalTypeId::BIGINT);
		auto window = make_unique<LogicalWindow>(correlated_columns[0].binding.table_index);
		auto row_number = make_unique<BoundWindowExpression>(ExpressionType::WINDOW_ROW_NUMBER, LogicalType::BIGINT,
		                                                     nullptr, nullptr);
		row_number->start = WindowBoundary::UNBOUNDED_PRECEDING;
		row_number->end = WindowBoundary::CURRENT_ROW_ROWS;
		row_number->alias = "delim_index";
		window->expressions.push_back(move(row_number));
		window->AddChild(move(original_plan));
		original_plan = move(window);
	}
	delim_join->AddChild(move(original_plan));
	for (idx_t i = 0; i < correlated_columns.size(); i++) {
		auto &col = correlated_columns[i];
		delim_join->duplicate_eliminated_columns.push_back(
		    make_unique<BoundColumnRefExpression>(col.type, col.binding));
		delim_join->delim_types.push_back(col.type);
	}
	return delim_join;
}

static void CreateDelimJoinConditions(LogicalDelimJoin &delim_join, vector<CorrelatedColumnInfo> &correlated_columns,
                                      vector<ColumnBinding> bindings, idx_t base_offset, bool perform_delim) {
	auto col_count = perform_delim ? correlated_columns.size() : 1;
	for (idx_t i = 0; i < col_count; i++) {
		auto &col = correlated_columns[i];
		JoinCondition cond;
		cond.left = make_unique<BoundColumnRefExpression>(col.name, col.type, col.binding);
		cond.right = make_unique<BoundColumnRefExpression>(col.name, col.type, bindings[base_offset + i]);
		cond.comparison = ExpressionType::COMPARE_NOT_DISTINCT_FROM;
		delim_join.conditions.push_back(move(cond));
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
	correlated_columns.insert(correlated_columns.begin(), move(info));
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
		    CreateDuplicateEliminatedJoin(correlated_columns, JoinType::SINGLE, move(root), perform_delim);

		// the right side initially is a DEPENDENT join between the duplicate eliminated scan and the subquery
		// HOWEVER: we do not explicitly create the dependent join
		// instead, we eliminate the dependent join by pushing it down into the right side of the plan
		FlattenDependentJoins flatten(binder, correlated_columns, perform_delim);

		// first we check which logical operators have correlated expressions in the first place
		flatten.DetectCorrelatedExpressions(plan.get());
		// now we push the dependent join down
		auto dependent_join = flatten.PushDownDependentJoin(move(plan));

		// now the dependent join is fully eliminated
		// we only need to create the join conditions between the LHS and the RHS
		// fetch the set of columns
		auto plan_columns = dependent_join->GetColumnBindings();

		// now create the join conditions
		CreateDelimJoinConditions(*delim_join, correlated_columns, plan_columns, flatten.delim_offset, perform_delim);
		delim_join->AddChild(move(dependent_join));
		root = move(delim_join);
		// finally push the BoundColumnRefExpression referring to the data element returned by the join
		return make_unique<BoundColumnRefExpression>(expr.GetName(), expr.return_type,
		                                             plan_columns[flatten.data_offset]);
	}
	case SubqueryType::EXISTS: {
		// correlated EXISTS query
		// this query is similar to the correlated SCALAR query, except we use a MARK join here
		idx_t mark_index = binder.GenerateTableIndex();
		auto delim_join = CreateDuplicateEliminatedJoin(correlated_columns, JoinType::MARK, move(root), perform_delim);
		delim_join->mark_index = mark_index;
		// RHS
		FlattenDependentJoins flatten(binder, correlated_columns, perform_delim, true);
		flatten.DetectCorrelatedExpressions(plan.get());
		auto dependent_join = flatten.PushDownDependentJoin(move(plan));

		// fetch the set of columns
		auto plan_columns = dependent_join->GetColumnBindings();

		// now we create the join conditions between the dependent join and the original table
		CreateDelimJoinConditions(*delim_join, correlated_columns, plan_columns, flatten.delim_offset, perform_delim);
		delim_join->AddChild(move(dependent_join));
		root = move(delim_join);
		// finally push the BoundColumnRefExpression referring to the marker
		return make_unique<BoundColumnRefExpression>(expr.GetName(), expr.return_type, ColumnBinding(mark_index, 0));
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
		auto delim_join = CreateDuplicateEliminatedJoin(correlated_columns, JoinType::MARK, move(root), perform_delim);
		delim_join->mark_index = mark_index;
		// RHS
		FlattenDependentJoins flatten(binder, correlated_columns, true, true);
		flatten.DetectCorrelatedExpressions(plan.get());
		auto dependent_join = flatten.PushDownDependentJoin(move(plan));

		// fetch the columns
		auto plan_columns = dependent_join->GetColumnBindings();

		// now we create the join conditions between the dependent join and the original table
		CreateDelimJoinConditions(*delim_join, correlated_columns, plan_columns, flatten.delim_offset, perform_delim);
		// add the actual condition based on the ANY/ALL predicate
		JoinCondition compare_cond;
		compare_cond.left = move(expr.child);
		compare_cond.right = BoundCastExpression::AddDefaultCastToType(
		    make_unique<BoundColumnRefExpression>(expr.child_type, plan_columns[0]), expr.child_target);
		compare_cond.comparison = expr.comparison_type;
		delim_join->conditions.push_back(move(compare_cond));

		delim_join->AddChild(move(dependent_join));
		root = move(delim_join);
		// finally push the BoundColumnRefExpression referring to the marker
		return make_unique<BoundColumnRefExpression>(expr.GetName(), expr.return_type, ColumnBinding(mark_index, 0));
	}
	}
}

class RecursiveSubqueryPlanner : public LogicalOperatorVisitor {
public:
	explicit RecursiveSubqueryPlanner(Binder &binder) : binder(binder) {
	}
	void VisitOperator(LogicalOperator &op) override {
		if (!op.children.empty()) {
			root = move(op.children[0]);
			D_ASSERT(root);
			VisitOperatorExpressions(op);
			op.children[0] = move(root);
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
	auto plan = move(subquery_root);
	unique_ptr<Expression> result_expression;
	if (!expr.IsCorrelated()) {
		result_expression = PlanUncorrelatedSubquery(*this, expr, root, move(plan));
	} else {
		result_expression = PlanCorrelatedSubquery(*this, expr, root, move(plan));
	}
	// finally, we recursively plan the nested subqueries (if there are any)
	if (sub_binder->has_unplanned_subqueries) {
		RecursiveSubqueryPlanner plan(*this);
		plan.VisitOperator(*root);
	}
	return result_expression;
}

void Binder::PlanSubqueries(unique_ptr<Expression> *expr_ptr, unique_ptr<LogicalOperator> *root) {
	if (!*expr_ptr) {
		return;
	}
	auto &expr = **expr_ptr;

	// first visit the children of the node, if any
	ExpressionIterator::EnumerateChildren(expr, [&](unique_ptr<Expression> &expr) { PlanSubqueries(&expr, root); });

	// check if this is a subquery node
	if (expr.expression_class == ExpressionClass::BOUND_SUBQUERY) {
		auto &subquery = (BoundSubqueryExpression &)expr;
		// subquery node! plan it
		if (subquery.IsCorrelated() && !plan_subquery) {
			// detected a nested correlated subquery
			// we don't plan it yet here, we are currently planning a subquery
			// nested subqueries will only be planned AFTER the current subquery has been flattened entirely
			has_unplanned_subqueries = true;
			return;
		}
		*expr_ptr = PlanSubquery(subquery, *root);
	}
}

} // namespace duckdb






namespace duckdb {

BoundStatement Binder::Bind(CallStatement &stmt) {
	BoundStatement result;

	TableFunctionRef ref;
	ref.function = move(stmt.function);

	auto bound_func = Bind(ref);
	auto &bound_table_func = (BoundTableFunction &)*bound_func;
	auto &get = (LogicalGet &)*bound_table_func.get;
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

BoundStatement Binder::BindCopyTo(CopyStatement &stmt) {
	// COPY TO a file
	auto &config = DBConfig::GetConfig(context);
	if (!config.options.enable_external_access) {
		throw PermissionException("COPY TO is disabled by configuration");
	}
	BoundStatement result;
	result.types = {LogicalType::BIGINT};
	result.names = {"Count"};

	// bind the select statement
	auto select_node = Bind(*stmt.select_statement);

	// lookup the format in the catalog
	auto &catalog = Catalog::GetCatalog(context);
	auto copy_function = catalog.GetEntry<CopyFunctionCatalogEntry>(context, DEFAULT_SCHEMA, stmt.info->format);
	if (!copy_function->function.copy_to_bind) {
		throw NotImplementedException("COPY TO is not supported for FORMAT \"%s\"", stmt.info->format);
	}
	bool use_tmp_file = true;
	for (auto &option : stmt.info->options) {
		auto loption = StringUtil::Lower(option.first);
		if (loption == "use_tmp_file") {
			use_tmp_file = option.second[0].CastAs(context, LogicalType::BOOLEAN).GetValue<bool>();
			stmt.info->options.erase(option.first);
			break;
		}
	}
	auto function_data =
	    copy_function->function.copy_to_bind(context, *stmt.info, select_node.names, select_node.types);
	// now create the copy information
	auto copy = make_unique<LogicalCopyToFile>(copy_function->function, move(function_data));
	copy->file_path = stmt.info->file_path;
	copy->use_tmp_file = use_tmp_file;
	copy->is_file_and_exists = config.file_system->FileExists(copy->file_path);

	copy->AddChild(move(select_node.plan));

	result.plan = move(copy);

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

	D_ASSERT(!stmt.info->table.empty());
	// COPY FROM a file
	// generate an insert statement for the the to-be-inserted table
	InsertStatement insert;
	insert.table = stmt.info->table;
	insert.schema = stmt.info->schema;
	insert.columns = stmt.info->select_list;

	// bind the insert statement to the base table
	auto insert_statement = Bind(insert);
	D_ASSERT(insert_statement.plan->type == LogicalOperatorType::LOGICAL_INSERT);

	auto &bound_insert = (LogicalInsert &)*insert_statement.plan;

	// lookup the format in the catalog
	auto &catalog = Catalog::GetCatalog(context);
	auto copy_function = catalog.GetEntry<CopyFunctionCatalogEntry>(context, DEFAULT_SCHEMA, stmt.info->format);
	if (!copy_function->function.copy_from_bind) {
		throw NotImplementedException("COPY FROM is not supported for FORMAT \"%s\"", stmt.info->format);
	}
	// lookup the table to copy into
	auto table = Catalog::GetCatalog(context).GetEntry<TableCatalogEntry>(context, stmt.info->schema, stmt.info->table);
	vector<string> expected_names;
	if (!bound_insert.column_index_map.empty()) {
		expected_names.resize(bound_insert.expected_types.size());
		for (auto &col : table->columns.Logical()) {
			auto i = col.Physical();
			if (bound_insert.column_index_map[i] != DConstants::INVALID_INDEX) {
				expected_names[bound_insert.column_index_map[i]] = col.Name();
			}
		}
	} else {
		expected_names.reserve(bound_insert.expected_types.size());
		for (auto &col : table->columns.Logical()) {
			expected_names.push_back(col.Name());
		}
	}

	auto function_data =
	    copy_function->function.copy_from_bind(context, *stmt.info, expected_names, bound_insert.expected_types);
	auto get = make_unique<LogicalGet>(GenerateTableIndex(), copy_function->function.copy_from_function,
	                                   move(function_data), bound_insert.expected_types, expected_names);
	for (idx_t i = 0; i < bound_insert.expected_types.size(); i++) {
		get->column_ids.push_back(i);
	}
	insert_statement.plan->children.push_back(move(get));
	result.plan = move(insert_statement.plan);
	return result;
}

BoundStatement Binder::Bind(CopyStatement &stmt) {
	if (!stmt.info->is_from && !stmt.select_statement) {
		// copy table into file without a query
		// generate SELECT * FROM table;
		auto ref = make_unique<BaseTableRef>();
		ref->schema_name = stmt.info->schema;
		ref->table_name = stmt.info->table;

		auto statement = make_unique<SelectNode>();
		statement->from_table = move(ref);
		if (!stmt.info->select_list.empty()) {
			for (auto &name : stmt.info->select_list) {
				statement->select_list.push_back(make_unique<ColumnRefExpression>(name));
			}
		} else {
			statement->select_list.push_back(make_unique<StarExpression>());
		}
		stmt.select_statement = move(statement);
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

SchemaCatalogEntry *Binder::BindSchema(CreateInfo &info) {
	if (info.schema.empty()) {
		info.schema = info.temporary ? TEMP_SCHEMA : ClientData::Get(context).catalog_search_path->GetDefault();
	}

	if (!info.temporary) {
		// non-temporary create: not read only
		if (info.schema == TEMP_SCHEMA) {
			throw ParserException("Only TEMPORARY table names can use the \"temp\" schema");
		}
		properties.read_only = false;
	} else {
		if (info.schema != TEMP_SCHEMA) {
			throw ParserException("TEMPORARY table names can *only* use the \"%s\" schema", TEMP_SCHEMA);
		}
	}
	// fetch the schema in which we want to create the object
	auto schema_obj = Catalog::GetCatalog(context).GetSchema(context, info.schema);
	D_ASSERT(schema_obj->type == CatalogType::SCHEMA_ENTRY);
	info.schema = schema_obj->name;
	return schema_obj;
}

void Binder::BindCreateViewInfo(CreateViewInfo &base) {
	// bind the view as if it were a query so we can catch errors
	// note that we bind the original, and replace the original with a copy
	auto view_binder = Binder::CreateBinder(context);
	view_binder->can_contain_nulls = true;

	auto copy = base.query->Copy();
	auto query_node = view_binder->Bind(*base.query);
	base.query = unique_ptr_cast<SQLStatement, SelectStatement>(move(copy));
	if (base.aliases.size() > query_node.names.size()) {
		throw BinderException("More VIEW aliases than columns in query result");
	}
	// fill up the aliases with the remaining names of the bound query
	for (idx_t i = base.aliases.size(); i < query_node.names.size(); i++) {
		base.aliases.push_back(query_node.names[i]);
	}
	base.types = query_node.types;
}

SchemaCatalogEntry *Binder::BindCreateFunctionInfo(CreateInfo &info) {
	auto &base = (CreateMacroInfo &)info;
	auto &scalar_function = (ScalarMacroFunction &)*base.function;

	if (scalar_function.expression->HasParameter()) {
		throw BinderException("Parameter expressions within macro's are not supported!");
	}

	// create macro binding in order to bind the function
	vector<LogicalType> dummy_types;
	vector<string> dummy_names;
	// positional parameters
	for (idx_t i = 0; i < base.function->parameters.size(); i++) {
		auto param = (ColumnRefExpression &)*base.function->parameters[i];
		if (param.IsQualified()) {
			throw BinderException("Invalid parameter name '%s': must be unqualified", param.ToString());
		}
		dummy_types.emplace_back(LogicalType::SQLNULL);
		dummy_names.push_back(param.GetColumnName());
	}
	// default parameters
	for (auto it = base.function->default_parameters.begin(); it != base.function->default_parameters.end(); it++) {
		auto &val = (ConstantExpression &)*it->second;
		dummy_types.push_back(val.value.type());
		dummy_names.push_back(it->first);
	}
	auto this_macro_binding = make_unique<DummyBinding>(dummy_types, dummy_names, base.name);
	macro_binding = this_macro_binding.get();
	ExpressionBinder::QualifyColumnNames(*this, scalar_function.expression);

	// create a copy of the expression because we do not want to alter the original
	auto expression = scalar_function.expression->Copy();

	// bind it to verify the function was defined correctly
	string error;
	auto sel_node = make_unique<BoundSelectNode>();
	auto group_info = make_unique<BoundGroupInformation>();
	SelectBinder binder(*this, context, *sel_node, *group_info);
	error = binder.Bind(&expression, 0, false);

	if (!error.empty()) {
		throw BinderException(error);
	}

	return BindSchema(info);
}

void Binder::BindLogicalType(ClientContext &context, LogicalType &type, const string &schema) {
	if (type.id() == LogicalTypeId::LIST) {
		auto child_type = ListType::GetChildType(type);
		BindLogicalType(context, child_type, schema);
		auto alias = type.GetAlias();
		type = LogicalType::LIST(child_type);
		type.SetAlias(alias);
	} else if (type.id() == LogicalTypeId::STRUCT || type.id() == LogicalTypeId::MAP) {
		auto child_types = StructType::GetChildTypes(type);
		for (auto &child_type : child_types) {
			BindLogicalType(context, child_type.second, schema);
		}
		// Generate new Struct/Map Type
		auto alias = type.GetAlias();
		if (type.id() == LogicalTypeId::STRUCT) {
			type = LogicalType::STRUCT(child_types);
		} else {
			type = LogicalType::MAP(child_types);
		}
		type.SetAlias(alias);
	} else if (type.id() == LogicalTypeId::UNION) {
		auto member_types = UnionType::CopyMemberTypes(type);
		for (auto &member_type : member_types) {
			BindLogicalType(context, member_type.second, schema);
		}
		// Generate new Union Type
		auto alias = type.GetAlias();
		type = LogicalType::UNION(member_types);
		type.SetAlias(alias);
	} else if (type.id() == LogicalTypeId::USER) {
		auto &catalog = Catalog::GetCatalog(context);
		type = catalog.GetType(context, schema, UserType::GetTypeName(type));
	} else if (type.id() == LogicalTypeId::ENUM) {
		auto &enum_type_name = EnumType::GetTypeName(type);
		auto enum_type_catalog = (TypeCatalogEntry *)context.db->GetCatalog().GetEntry(context, CatalogType::TYPE_ENTRY,
		                                                                               schema, enum_type_name, true);
		LogicalType::SetCatalog(type, enum_type_catalog);
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
		auto &unique = (UniqueConstraint &)*constr;
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
		if (pk_names.size() != fk.fk_columns.size()) {
			// the number of referencing and referenced columns for foreign keys must be the same
			continue;
		}
		if (find_primary_key) {
			// found matching primary key
			fk.pk_columns = pk_names;
			return;
		}
		if (fk.pk_columns != pk_names) {
			// Name mismatch
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
		auto &column_ref = (ColumnRefExpression &)expr;
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
			auto &constraint = (CheckConstraint &)*constr;
			auto &expr = constraint.expression;
			bool contains_generated_column = false;
			ExpressionContainsGeneratedColumn(*expr, generated_columns, contains_generated_column);
			if (contains_generated_column) {
				return true;
			}
			break;
		}
		case ConstraintType::NOT_NULL: {
			auto &constraint = (NotNullConstraint &)*constr;
			if (table_info.columns.GetColumn(constraint.index).Generated()) {
				return true;
			}
			break;
		}
		case ConstraintType::UNIQUE: {
			auto &constraint = (UniqueConstraint &)*constr;
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

BoundStatement Binder::Bind(CreateStatement &stmt) {
	BoundStatement result;
	result.names = {"Count"};
	result.types = {LogicalType::BIGINT};
	properties.return_type = StatementReturnType::NOTHING;

	auto catalog_type = stmt.info->type;
	switch (catalog_type) {
	case CatalogType::SCHEMA_ENTRY:
		result.plan = make_unique<LogicalCreate>(LogicalOperatorType::LOGICAL_CREATE_SCHEMA, move(stmt.info));
		break;
	case CatalogType::VIEW_ENTRY: {
		auto &base = (CreateViewInfo &)*stmt.info;
		// bind the schema
		auto schema = BindSchema(*stmt.info);
		BindCreateViewInfo(base);
		result.plan = make_unique<LogicalCreate>(LogicalOperatorType::LOGICAL_CREATE_VIEW, move(stmt.info), schema);
		break;
	}
	case CatalogType::SEQUENCE_ENTRY: {
		auto schema = BindSchema(*stmt.info);
		result.plan = make_unique<LogicalCreate>(LogicalOperatorType::LOGICAL_CREATE_SEQUENCE, move(stmt.info), schema);
		break;
	}
	case CatalogType::TABLE_MACRO_ENTRY: {
		auto schema = BindSchema(*stmt.info);
		result.plan = make_unique<LogicalCreate>(LogicalOperatorType::LOGICAL_CREATE_MACRO, move(stmt.info), schema);
		break;
	}
	case CatalogType::MACRO_ENTRY: {
		auto schema = BindCreateFunctionInfo(*stmt.info);
		result.plan = make_unique<LogicalCreate>(LogicalOperatorType::LOGICAL_CREATE_MACRO, move(stmt.info), schema);
		break;
	}
	case CatalogType::INDEX_ENTRY: {
		auto &base = (CreateIndexInfo &)*stmt.info;

		// visit the table reference
		auto bound_table = Bind(*base.table);
		if (bound_table->type != TableReferenceType::BASE_TABLE) {
			throw BinderException("Can only delete from base table!");
		}
		auto &table_binding = (BoundBaseTableRef &)*bound_table;
		auto table = table_binding.table;

		// bind the index expressions
		vector<unique_ptr<Expression>> expressions;
		IndexBinder binder(*this, context);
		for (auto &expr : base.expressions) {
			expressions.push_back(binder.Bind(expr));
		}

		auto plan = CreatePlan(*bound_table);
		if (plan->type != LogicalOperatorType::LOGICAL_GET) {
			throw BinderException("Cannot create index on a view!");
		}

		auto &get = (LogicalGet &)*plan;
		for (auto &column_id : get.column_ids) {
			if (column_id == COLUMN_IDENTIFIER_ROW_ID) {
				throw BinderException("Cannot create an index on the rowid!");
			}
		}

		auto create_index_info = unique_ptr_cast<CreateInfo, CreateIndexInfo>(move(stmt.info));
		for (auto &index : get.column_ids) {
			create_index_info->scan_types.push_back(get.returned_types[index]);
		}
		create_index_info->scan_types.emplace_back(LogicalType::ROW_TYPE);
		create_index_info->names = get.names;
		create_index_info->column_ids = get.column_ids;

		// the logical CREATE INDEX also needs all fields to scan the referenced table
		result.plan = make_unique<LogicalCreateIndex>(move(get.bind_data), move(create_index_info), move(expressions),
		                                              *table, move(get.function));
		break;
	}
	case CatalogType::TABLE_ENTRY: {
		auto &create_info = (CreateTableInfo &)*stmt.info;
		auto &catalog = Catalog::GetCatalog(context);
		// If there is a foreign key constraint, resolve primary key column's index from primary key column's name
		unordered_set<SchemaCatalogEntry *> fk_schemas;
		for (idx_t i = 0; i < create_info.constraints.size(); i++) {
			auto &cond = create_info.constraints[i];
			if (cond->type != ConstraintType::FOREIGN_KEY) {
				continue;
			}
			auto &fk = (ForeignKeyConstraint &)*cond;
			if (fk.info.type != ForeignKeyType::FK_TYPE_FOREIGN_KEY_TABLE) {
				continue;
			}
			D_ASSERT(fk.info.pk_keys.empty());
			D_ASSERT(fk.info.fk_keys.empty());
			FindForeignKeyIndexes(create_info.columns, fk.fk_columns, fk.info.fk_keys);
			if (create_info.table == fk.info.table) {
				// self-referential foreign key constraint
				fk.info.type = ForeignKeyType::FK_TYPE_SELF_REFERENCE_TABLE;
				FindMatchingPrimaryKeyColumns(create_info.columns, create_info.constraints, fk);
				FindForeignKeyIndexes(create_info.columns, fk.pk_columns, fk.info.pk_keys);
				CheckForeignKeyTypes(create_info.columns, create_info.columns, fk);
			} else {
				// have to resolve referenced table
				auto pk_table_entry_ptr = catalog.GetEntry<TableCatalogEntry>(context, fk.info.schema, fk.info.table);
				fk_schemas.insert(pk_table_entry_ptr->schema);
				FindMatchingPrimaryKeyColumns(pk_table_entry_ptr->columns, pk_table_entry_ptr->constraints, fk);
				FindForeignKeyIndexes(pk_table_entry_ptr->columns, fk.pk_columns, fk.info.pk_keys);
				CheckForeignKeyTypes(pk_table_entry_ptr->columns, create_info.columns, fk);
				auto index = pk_table_entry_ptr->storage->info->indexes.FindForeignKeyIndex(
				    fk.info.pk_keys, ForeignKeyType::FK_TYPE_PRIMARY_KEY_TABLE);
				if (!index) {
					auto fk_column_names = StringUtil::Join(fk.pk_columns, ",");
					throw BinderException("Failed to create foreign key on %s(%s): no UNIQUE or PRIMARY KEY constraint "
					                      "present on these columns",
					                      pk_table_entry_ptr->name, fk_column_names);
				}
			}
			D_ASSERT(fk.info.pk_keys.size() == fk.info.fk_keys.size());
			D_ASSERT(fk.info.pk_keys.size() == fk.pk_columns.size());
			D_ASSERT(fk.info.fk_keys.size() == fk.fk_columns.size());
		}
		if (AnyConstraintReferencesGeneratedColumn(create_info)) {
			throw BinderException("Constraints on generated columns are not supported yet");
		}
		auto bound_info = BindCreateTableInfo(move(stmt.info));
		auto root = move(bound_info->query);
		for (auto &fk_schema : fk_schemas) {
			if (fk_schema != bound_info->schema) {
				throw BinderException("Creating foreign keys across different schemas is not supported");
			}
		}

		// create the logical operator
		auto &schema = bound_info->schema;
		auto create_table = make_unique<LogicalCreateTable>(schema, move(bound_info));
		if (root) {
			// CREATE TABLE AS
			properties.return_type = StatementReturnType::CHANGED_ROWS;
			create_table->children.push_back(move(root));
		}
		result.plan = move(create_table);
		break;
	}
	case CatalogType::TYPE_ENTRY: {
		auto schema = BindSchema(*stmt.info);
		auto &create_type_info = (CreateTypeInfo &)(*stmt.info);
		result.plan = make_unique<LogicalCreate>(LogicalOperatorType::LOGICAL_CREATE_TYPE, move(stmt.info), schema);
		if (create_type_info.query) {
			// CREATE TYPE mood AS ENUM (SELECT 'happy')
			auto &select_stmt = (SelectStatement &)*create_type_info.query;
			auto &query_node = *select_stmt.node;

			// We always add distinct modifier implicitly
			bool need_to_add = true;
			if (!query_node.modifiers.empty()) {
				if (query_node.modifiers[0]->type == ResultModifierType::DISTINCT_MODIFIER) {
					// There are cases where the same column is grouped repeatedly
					// CREATE TYPE mood AS ENUM (SELECT DISTINCT ON(x) x FROM test);
					// When we push into a constant expression
					// => CREATE TYPE mood AS ENUM (SELECT DISTINCT ON(x, x) x FROM test);
					auto &distinct_modifier = (DistinctModifier &)*query_node.modifiers[0];
					distinct_modifier.distinct_on_targets.push_back(make_unique<ConstantExpression>(Value::INTEGER(1)));
					need_to_add = false;
				}
			}

			// Add distinct modifier
			if (need_to_add) {
				auto distinct_modifier = make_unique<DistinctModifier>();
				distinct_modifier->distinct_on_targets.push_back(make_unique<ConstantExpression>(Value::INTEGER(1)));
				query_node.modifiers.emplace(query_node.modifiers.begin(), move(distinct_modifier));
			}

			auto query_obj = Bind(*create_type_info.query);
			auto query = move(query_obj.plan);

			auto &sql_types = query_obj.types;
			if (sql_types.size() != 1 || sql_types[0].id() != LogicalType::VARCHAR) {
				// add cast expression?
				throw BinderException("The query must return one varchar column");
			}

			result.plan->AddChild(move(query));
		}
		break;
	}
	default:
		throw Exception("Unrecognized type!");
	}
	properties.allow_stream_result = false;
	return result;
}

} // namespace duckdb



















#include <algorithm>

namespace duckdb {

static void CreateColumnDependencyManager(BoundCreateTableInfo &info) {
	auto &base = (CreateTableInfo &)*info.base;
	for (auto &col : base.columns.Logical()) {
		if (!col.Generated()) {
			continue;
		}
		info.column_dependency_manager.AddGeneratedColumn(col, base.columns);
	}
}

static void BindCheckConstraint(Binder &binder, BoundCreateTableInfo &info, const unique_ptr<Constraint> &cond) {
	auto &base = (CreateTableInfo &)*info.base;

	auto bound_constraint = make_unique<BoundCheckConstraint>();
	// check constraint: bind the expression
	CheckBinder check_binder(binder, binder.context, base.table, base.columns, bound_constraint->bound_columns);
	auto &check = (CheckConstraint &)*cond;
	// create a copy of the unbound expression because the binding destroys the constraint
	auto unbound_expression = check.expression->Copy();
	// now bind the constraint and create a new BoundCheckConstraint
	bound_constraint->expression = check_binder.Bind(check.expression);
	info.bound_constraints.push_back(move(bound_constraint));
	// move the unbound constraint back into the original check expression
	check.expression = move(unbound_expression);
}

static void BindConstraints(Binder &binder, BoundCreateTableInfo &info) {
	auto &base = (CreateTableInfo &)*info.base;

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
			auto &not_null = (NotNullConstraint &)*cond;
			auto &col = base.columns.GetColumn(LogicalIndex(not_null.index));
			info.bound_constraints.push_back(make_unique<BoundNotNullConstraint>(PhysicalIndex(col.StorageOid())));
			not_null_columns.insert(not_null.index);
			break;
		}
		case ConstraintType::UNIQUE: {
			auto &unique = (UniqueConstraint &)*cond;
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
			    make_unique<BoundUniqueConstraint>(move(keys), move(key_set), unique.is_primary_key));
			break;
		}
		case ConstraintType::FOREIGN_KEY: {
			auto &fk = (ForeignKeyConstraint &)*cond;
			D_ASSERT((fk.info.type == ForeignKeyType::FK_TYPE_FOREIGN_KEY_TABLE && !fk.info.pk_keys.empty()) ||
			         (fk.info.type == ForeignKeyType::FK_TYPE_PRIMARY_KEY_TABLE && !fk.info.pk_keys.empty()) ||
			         fk.info.type == ForeignKeyType::FK_TYPE_SELF_REFERENCE_TABLE);
			physical_index_set_t fk_key_set, pk_key_set;
			for (idx_t i = 0; i < fk.info.pk_keys.size(); i++) {
				pk_key_set.insert(fk.info.pk_keys[i]);
			}
			for (idx_t i = 0; i < fk.info.fk_keys.size(); i++) {
				fk_key_set.insert(fk.info.fk_keys[i]);
			}
			info.bound_constraints.push_back(
			    make_unique<BoundForeignKeyConstraint>(fk.info, move(pk_key_set), move(fk_key_set)));
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
			base.constraints.push_back(make_unique<NotNullConstraint>(column_index));
			info.bound_constraints.push_back(make_unique<BoundNotNullConstraint>(physical_index));
		}
	}
}

void Binder::BindGeneratedColumns(BoundCreateTableInfo &info) {
	auto &base = (CreateTableInfo &)*info.base;

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

void Binder::BindDefaultValues(ColumnList &columns, vector<unique_ptr<Expression>> &bound_defaults) {
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
			bound_default = make_unique<BoundConstantExpression>(Value(column.Type()));
		}
		bound_defaults.push_back(move(bound_default));
	}
}

static void ExtractExpressionDependencies(Expression &expr, unordered_set<CatalogEntry *> &dependencies) {
	if (expr.type == ExpressionType::BOUND_FUNCTION) {
		auto &function = (BoundFunctionExpression &)expr;
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
			auto &bound_check = (BoundCheckConstraint &)*constraint;
			ExtractExpressionDependencies(*bound_check.expression, info.dependencies);
		}
	}
}

unique_ptr<BoundCreateTableInfo> Binder::BindCreateTableInfo(unique_ptr<CreateInfo> info) {
	auto &base = (CreateTableInfo &)*info;

	auto result = make_unique<BoundCreateTableInfo>(move(info));
	result->schema = BindSchema(*result->base);
	if (base.query) {
		// construct the result object
		auto query_obj = Bind(*base.query);
		result->query = move(query_obj.plan);

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
		BindLogicalType(context, column.TypeMutable());
		// We add a catalog dependency
		auto type_dependency = LogicalType::GetCatalog(column.Type());
		if (type_dependency) {
			// Only if the USER comes from a create type
			result->dependencies.insert(type_dependency);
		}
	}
	properties.allow_stream_result = false;
	return result;
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
	auto &table_binding = (BoundBaseTableRef &)*bound_table;
	auto table = table_binding.table;

	auto root = CreatePlan(*bound_table);
	auto &get = (LogicalGet &)*root;
	D_ASSERT(root->type == LogicalOperatorType::LOGICAL_GET);

	if (!table->temporary) {
		// delete from persistent table: not read only!
		properties.read_only = false;
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
				child_operator = LogicalCrossProduct::Create(move(child_operator), move(op));
			} else {
				child_operator = move(op);
			}
			bind_context.AddContext(move(using_binder->bind_context));
		}
		if (child_operator) {
			root = LogicalCrossProduct::Create(move(root), move(child_operator));
		}
	}

	// project any additional columns required for the condition
	unique_ptr<Expression> condition;
	if (stmt.condition) {
		WhereBinder binder(*this, context);
		condition = binder.Bind(stmt.condition);

		PlanSubqueries(&condition, &root);
		auto filter = make_unique<LogicalFilter>(move(condition));
		filter->AddChild(move(root));
		root = move(filter);
	}
	// create the delete node
	auto del = make_unique<LogicalDelete>(table, GenerateTableIndex());
	del->AddChild(move(root));

	// set up the delete expression
	del->expressions.push_back(make_unique<BoundColumnRefExpression>(
	    LogicalType::ROW_TYPE, ColumnBinding(get.table_index, get.column_ids.size())));
	get.column_ids.push_back(COLUMN_IDENTIFIER_ROW_ID);

	if (!stmt.returning_list.empty()) {
		del->return_chunk = true;

		auto update_table_index = GenerateTableIndex();
		del->table_index = update_table_index;

		unique_ptr<LogicalOperator> del_as_logicaloperator = move(del);
		return BindReturning(move(stmt.returning_list), table, update_table_index, move(del_as_logicaloperator),
		                     move(result));
	}
	result.plan = move(del);
	result.names = {"Count"};
	result.types = {LogicalType::BIGINT};
	properties.allow_stream_result = false;
	properties.return_type = StatementReturnType::CHANGED_ROWS;

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
	case CatalogType::SCHEMA_ENTRY:
		// dropping a schema is never read-only because there are no temporary schemas
		properties.read_only = false;
		break;
	case CatalogType::VIEW_ENTRY:
	case CatalogType::SEQUENCE_ENTRY:
	case CatalogType::MACRO_ENTRY:
	case CatalogType::TABLE_MACRO_ENTRY:
	case CatalogType::INDEX_ENTRY:
	case CatalogType::TABLE_ENTRY:
	case CatalogType::TYPE_ENTRY: {
		auto entry = (StandardEntry *)Catalog::GetCatalog(context).GetEntry(context, stmt.info->type, stmt.info->schema,
		                                                                    stmt.info->name, true);
		if (!entry) {
			break;
		}
		if (!entry->temporary) {
			// we can only drop temporary tables in read-only mode
			properties.read_only = false;
		}
		stmt.info->schema = entry->schema->name;
		break;
	}
	default:
		throw BinderException("Unknown catalog type for drop statement!");
	}
	result.plan = make_unique<LogicalSimple>(LogicalOperatorType::LOGICAL_DROP, move(stmt.info));
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
		bind_values.push_back(move(value));
	}
	unique_ptr<LogicalOperator> rebound_plan;
	if (prepared->RequireRebind(context, bind_values)) {
		// catalog was modified or statement does not have clear types: rebind the statement before running the execute
		Planner prepared_planner(context);
		for (idx_t i = 0; i < bind_values.size(); i++) {
			prepared_planner.parameter_data.emplace_back(bind_values[i]);
		}
		prepared = prepared_planner.PrepareSQLStatement(entry->second->unbound_statement->Copy());
		rebound_plan = move(prepared_planner.plan);
		D_ASSERT(prepared->properties.bound_all_parameters);
		this->bound_tables = prepared_planner.binder->bound_tables;
	}
	// copy the properties of the prepared statement into the planner
	this->properties = prepared->properties;
	this->properties.parameter_count = parameter_count;
	BoundStatement result;
	result.names = prepared->names;
	result.types = prepared->types;

	prepared->Bind(move(bind_values));
	if (rebound_plan) {
		auto execute_plan = make_unique<LogicalExecute>(move(prepared));
		execute_plan->children.push_back(move(rebound_plan));
		result.plan = move(execute_plan);
	} else {
		result.plan = make_unique<LogicalExecute>(move(prepared));
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
	auto explain = make_unique<LogicalExplain>(move(plan.plan), stmt.explain_type);
	explain->logical_plan_unopt = logical_plan_unopt;

	result.plan = move(explain);
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

bool IsExistMainKeyTable(string &table_name, vector<TableCatalogEntry *> &unordered) {
	for (idx_t i = 0; i < unordered.size(); i++) {
		if (unordered[i]->name == table_name) {
			return true;
		}
	}
	return false;
}

void ScanForeignKeyTable(vector<TableCatalogEntry *> &ordered, vector<TableCatalogEntry *> &unordered,
                         bool move_only_pk_table) {
	for (auto i = unordered.begin(); i != unordered.end();) {
		auto table_entry = *i;
		bool move_to_ordered = true;
		for (idx_t j = 0; j < table_entry->constraints.size(); j++) {
			auto &cond = table_entry->constraints[j];
			if (cond->type == ConstraintType::FOREIGN_KEY) {
				auto &fk = (ForeignKeyConstraint &)*cond;
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

void ReorderTableEntries(vector<TableCatalogEntry *> &tables) {
	vector<TableCatalogEntry *> ordered;
	vector<TableCatalogEntry *> unordered = tables;
	ScanForeignKeyTable(ordered, unordered, true);
	while (!unordered.empty()) {
		ScanForeignKeyTable(ordered, unordered, false);
	}
	tables = ordered;
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
	auto &catalog = Catalog::GetCatalog(context);
	auto copy_function = catalog.GetEntry<CopyFunctionCatalogEntry>(context, DEFAULT_SCHEMA, stmt.info->format);
	if (!copy_function->function.copy_to_bind) {
		throw NotImplementedException("COPY TO is not supported for FORMAT \"%s\"", stmt.info->format);
	}

	// gather a list of all the tables
	vector<TableCatalogEntry *> tables;
	auto schemas = catalog.schemas->GetEntries<SchemaCatalogEntry>(context);
	for (auto &schema : schemas) {
		schema->Scan(context, CatalogType::TABLE_ENTRY, [&](CatalogEntry *entry) {
			if (entry->type == CatalogType::TABLE_ENTRY) {
				tables.push_back((TableCatalogEntry *)entry);
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
	for (auto &table : tables) {
		auto info = make_unique<CopyInfo>();
		// we copy the options supplied to the EXPORT
		info->format = stmt.info->format;
		info->options = stmt.info->options;
		// set up the file name for the COPY TO

		auto exported_data = ExportedTableData();
		idx_t id = 0;
		while (true) {
			string id_suffix = id == 0 ? string() : "_" + to_string(id);
			if (table->schema->name == DEFAULT_SCHEMA) {
				info->file_path = fs.JoinPath(stmt.info->file_path,
				                              StringUtil::Format("%s%s.%s", SanitizeExportIdentifier(table->name),
				                                                 id_suffix, copy_function->function.extension));
			} else {
				info->file_path =
				    fs.JoinPath(stmt.info->file_path,
				                StringUtil::Format("%s_%s%s.%s", SanitizeExportIdentifier(table->schema->name),
				                                   SanitizeExportIdentifier(table->name), id_suffix,
				                                   copy_function->function.extension));
			}
			if (table_name_index.find(info->file_path) == table_name_index.end()) {
				// this name was not yet taken: take it
				table_name_index.insert(info->file_path);
				break;
			}
			id++;
		}
		info->is_from = false;
		info->schema = table->schema->name;
		info->table = table->name;

		// We can not export generated columns
		for (auto &col : table->columns.Physical()) {
			info->select_list.push_back(col.GetName());
		}

		exported_data.table_name = info->table;
		exported_data.schema_name = info->schema;
		exported_data.file_path = info->file_path;

		ExportedTableInfo table_info;
		table_info.entry = table;
		table_info.table_data = exported_data;
		exported_tables.data.push_back(table_info);
		id++;

		// generate the copy statement and bind it
		CopyStatement copy_stmt;
		copy_stmt.info = move(info);

		auto copy_binder = Binder::CreateBinder(context, this);
		auto bound_statement = copy_binder->Bind(copy_stmt);
		if (child_operator) {
			// use UNION ALL to combine the individual copy statements into a single node
			auto copy_union =
			    make_unique<LogicalSetOperation>(GenerateTableIndex(), 1, move(child_operator),
			                                     move(bound_statement.plan), LogicalOperatorType::LOGICAL_UNION);
			child_operator = move(copy_union);
		} else {
			child_operator = move(bound_statement.plan);
		}
	}

	// try to create the directory, if it doesn't exist yet
	// a bit hacky to do it here, but we need to create the directory BEFORE the copy statements run
	if (!fs.DirectoryExists(stmt.info->file_path)) {
		fs.CreateDirectory(stmt.info->file_path);
	}

	// create the export node
	auto export_node = make_unique<LogicalExport>(copy_function->function, move(stmt.info), exported_tables);

	if (child_operator) {
		export_node->children.push_back(move(child_operator));
	}

	result.plan = move(export_node);
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
	auto parse_result = stmt.extension.plan_function(stmt.extension.parser_info.get(), context, move(stmt.parse_data));

	properties.read_only = parse_result.read_only;
	properties.requires_valid_transaction = parse_result.requires_valid_transaction;
	properties.return_type = parse_result.return_type;

	// create the plan as a scan of the given table function
	result.plan = BindTableFunction(parse_result.function, move(parse_result.parameters));
	D_ASSERT(result.plan->type == LogicalOperatorType::LOGICAL_GET);
	auto &get = (LogicalGet &)*result.plan;
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

BoundStatement Binder::Bind(InsertStatement &stmt) {
	BoundStatement result;
	result.names = {"Count"};
	result.types = {LogicalType::BIGINT};

	auto table = Catalog::GetCatalog(context).GetEntry<TableCatalogEntry>(context, stmt.schema, stmt.table);
	D_ASSERT(table);
	if (!table->temporary) {
		// inserting into a non-temporary table: alters underlying database
		properties.read_only = false;
	}

	auto insert = make_unique<LogicalInsert>(table, GenerateTableIndex());

	// Add CTEs as bindable
	AddCTEMap(stmt.cte_map);

	vector<LogicalIndex> named_column_map;
	if (!stmt.columns.empty()) {
		// insertion statement specifies column list

		// create a mapping of (list index) -> (column index)
		case_insensitive_map_t<idx_t> column_name_map;
		for (idx_t i = 0; i < stmt.columns.size(); i++) {
			column_name_map[stmt.columns[i]] = i;
			auto column_index = table->GetColumnIndex(stmt.columns[i]);
			if (column_index.index == COLUMN_IDENTIFIER_ROW_ID) {
				throw BinderException("Cannot explicitly insert values into rowid column");
			}
			auto &col = table->columns.GetColumn(column_index);
			if (col.Generated()) {
				throw BinderException("Cannot insert into a generated column");
			}
			insert->expected_types.push_back(col.Type());
			named_column_map.push_back(column_index);
		}
		for (auto &col : table->columns.Physical()) {
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
		for (auto &col : table->columns.Physical()) {
			named_column_map.push_back(col.Logical());
			insert->expected_types.push_back(col.Type());
		}
	}

	// bind the default values
	BindDefaultValues(table->columns, insert->bound_defaults);
	if (!stmt.select_statement) {
		result.plan = move(insert);
		return result;
	}

	// Exclude the generated columns from this amount
	idx_t expected_columns = stmt.columns.empty() ? table->columns.PhysicalColumnCount() : stmt.columns.size();

	// special case: check if we are inserting from a VALUES statement
	auto values_list = stmt.GetValuesList();
	if (values_list) {
		auto &expr_list = (ExpressionListRef &)*values_list;
		expr_list.expected_types.resize(expected_columns);
		expr_list.expected_names.resize(expected_columns);

		D_ASSERT(expr_list.values.size() > 0);
		CheckInsertColumnCountMismatch(expected_columns, expr_list.values[0].size(), !stmt.columns.empty(),
		                               table->name.c_str());

		// VALUES list!
		for (idx_t col_idx = 0; col_idx < expected_columns; col_idx++) {
			auto table_col_idx = named_column_map[col_idx];

			// set the expected types as the types for the INSERT statement
			auto &column = table->columns.GetColumn(table_col_idx);
			expr_list.expected_types[col_idx] = column.Type();
			expr_list.expected_names[col_idx] = column.Name();

			// now replace any DEFAULT values with the corresponding default expression
			for (idx_t list_idx = 0; list_idx < expr_list.values.size(); list_idx++) {
				if (expr_list.values[list_idx][col_idx]->type == ExpressionType::VALUE_DEFAULT) {
					// DEFAULT value! replace the entry
					if (column.DefaultValue()) {
						expr_list.values[list_idx][col_idx] = column.DefaultValue()->Copy();
					} else {
						expr_list.values[list_idx][col_idx] = make_unique<ConstantExpression>(Value(column.Type()));
					}
				}
			}
		}
	}

	// parse select statement and add to logical plan
	auto select_binder = Binder::CreateBinder(context, this);
	auto root_select = select_binder->Bind(*stmt.select_statement);
	MoveCorrelatedExpressions(*select_binder);

	CheckInsertColumnCountMismatch(expected_columns, root_select.types.size(), !stmt.columns.empty(),
	                               table->name.c_str());

	auto root = CastLogicalOperatorToTypes(root_select.types, insert->expected_types, move(root_select.plan));

	insert->AddChild(move(root));

	if (!stmt.returning_list.empty()) {
		insert->return_chunk = true;
		result.types.clear();
		result.names.clear();
		auto insert_table_index = GenerateTableIndex();
		insert->table_index = insert_table_index;
		unique_ptr<LogicalOperator> index_as_logicaloperator = move(insert);

		return BindReturning(move(stmt.returning_list), table, insert_table_index, move(index_as_logicaloperator),
		                     move(result));
	}

	D_ASSERT(result.types.size() == result.names.size());
	result.plan = move(insert);
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

	result.plan = make_unique<LogicalSimple>(LogicalOperatorType::LOGICAL_LOAD, move(stmt.info));
	properties.allow_stream_result = false;
	properties.return_type = StatementReturnType::NOTHING;
	return result;
}

} // namespace duckdb


#include <algorithm>

namespace duckdb {

BoundStatement Binder::Bind(LogicalPlanStatement &stmt) {
	BoundStatement result;
	result.types = stmt.plan->types;
	for (idx_t i = 0; i < result.types.size(); i++) {
		result.names.push_back(StringUtil::Format("col%d", i));
	}
	result.plan = move(stmt.plan);
	properties.allow_stream_result = true;
	properties.return_type = StatementReturnType::QUERY_RESULT; // TODO could also be something else

	return result;
}

} // namespace duckdb







namespace duckdb {

BoundStatement Binder::Bind(PragmaStatement &stmt) {
	auto &catalog = Catalog::GetCatalog(context);
	// bind the pragma function
	auto entry = catalog.GetEntry<PragmaFunctionCatalogEntry>(context, DEFAULT_SCHEMA, stmt.info->name, false);
	string error;
	FunctionBinder function_binder(context);
	idx_t bound_idx = function_binder.BindFunction(entry->name, entry->functions, *stmt.info, error);
	if (bound_idx == DConstants::INVALID_INDEX) {
		throw BinderException(FormatError(stmt.stmt_location, error));
	}
	auto bound_function = entry->functions.GetFunctionByOffset(bound_idx);
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
	result.plan = make_unique<LogicalPragma>(bound_function, *stmt.info);
	properties.return_type = StatementReturnType::QUERY_RESULT;
	return result;
}

} // namespace duckdb





namespace duckdb {

BoundStatement Binder::Bind(PrepareStatement &stmt) {
	Planner prepared_planner(context);
	auto prepared_data = prepared_planner.PrepareSQLStatement(move(stmt.statement));
	this->bound_tables = prepared_planner.binder->bound_tables;

	auto prepare = make_unique<LogicalPrepare>(stmt.name, move(prepared_data), move(prepared_planner.plan));
	// we can prepare in read-only mode: prepared statements are not written to the catalog
	properties.read_only = true;
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
	result.plan = move(prepare);
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

BoundStatement Binder::Bind(SetStatement &stmt) {
	BoundStatement result;
	result.types = {LogicalType::BOOLEAN};
	result.names = {"Success"};

	result.plan = make_unique<LogicalSet>(stmt.name, stmt.value, stmt.scope);
	properties.return_type = StatementReturnType::NOTHING;
	return result;
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

	auto show = make_unique<LogicalShow>(move(plan.plan));
	show->types_select = plan.types;
	show->aliases = plan.names;

	result.plan = move(show);

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
	Catalog &catalog = Catalog::GetCatalog(context);
	auto entry = catalog.GetEntry(context, stmt.info->GetCatalogType(), stmt.info->schema, stmt.info->name,
	                              stmt.info->if_exists);
	if (entry && !entry->temporary) {
		// we can only alter temporary tables/views in read-only mode
		properties.read_only = false;
	}
	result.plan = make_unique<LogicalSimple>(LogicalOperatorType::LOGICAL_ALTER, move(stmt.info));
	properties.return_type = StatementReturnType::NOTHING;
	return result;
}

BoundStatement Binder::Bind(TransactionStatement &stmt) {
	// transaction statements do not require a valid transaction
	properties.requires_valid_transaction = stmt.info->type == TransactionType::BEGIN_TRANSACTION;

	BoundStatement result;
	result.names = {"Success"};
	result.types = {LogicalType::BOOLEAN};
	result.plan = make_unique<LogicalSimple>(LogicalOperatorType::LOGICAL_TRANSACTION, move(stmt.info));
	properties.return_type = StatementReturnType::NOTHING;
	return result;
}

} // namespace duckdb









namespace duckdb {

static unique_ptr<ParsedExpression> SummarizeWrapUnnest(vector<unique_ptr<ParsedExpression>> &children,
                                                        const string &alias) {
	auto list_function = make_unique<FunctionExpression>("list_value", move(children));
	vector<unique_ptr<ParsedExpression>> unnest_children;
	unnest_children.push_back(move(list_function));
	auto unnest_function = make_unique<FunctionExpression>("unnest", move(unnest_children));
	unnest_function->alias = alias;
	return move(unnest_function);
}

static unique_ptr<ParsedExpression> SummarizeCreateAggregate(const string &aggregate, string column_name) {
	vector<unique_ptr<ParsedExpression>> children;
	children.push_back(make_unique<ColumnRefExpression>(move(column_name)));
	auto aggregate_function = make_unique<FunctionExpression>(aggregate, move(children));
	auto cast_function = make_unique<CastExpression>(LogicalType::VARCHAR, move(aggregate_function));
	return move(cast_function);
}

static unique_ptr<ParsedExpression> SummarizeCreateAggregate(const string &aggregate, string column_name,
                                                             const Value &modifier) {
	vector<unique_ptr<ParsedExpression>> children;
	children.push_back(make_unique<ColumnRefExpression>(move(column_name)));
	children.push_back(make_unique<ConstantExpression>(modifier));
	auto aggregate_function = make_unique<FunctionExpression>(aggregate, move(children));
	auto cast_function = make_unique<CastExpression>(LogicalType::VARCHAR, move(aggregate_function));
	return move(cast_function);
}

static unique_ptr<ParsedExpression> SummarizeCreateCountStar() {
	vector<unique_ptr<ParsedExpression>> children;
	auto aggregate_function = make_unique<FunctionExpression>("count_star", move(children));
	return move(aggregate_function);
}

static unique_ptr<ParsedExpression> SummarizeCreateBinaryFunction(const string &op, unique_ptr<ParsedExpression> left,
                                                                  unique_ptr<ParsedExpression> right) {
	vector<unique_ptr<ParsedExpression>> children;
	children.push_back(move(left));
	children.push_back(move(right));
	auto binary_function = make_unique<FunctionExpression>(op, move(children));
	return move(binary_function);
}

static unique_ptr<ParsedExpression> SummarizeCreateNullPercentage(string column_name) {
	auto count_star = make_unique<CastExpression>(LogicalType::DOUBLE, SummarizeCreateCountStar());
	auto count = make_unique<CastExpression>(LogicalType::DOUBLE, SummarizeCreateAggregate("count", move(column_name)));
	auto null_percentage = SummarizeCreateBinaryFunction("/", move(count), move(count_star));
	auto negate_x =
	    SummarizeCreateBinaryFunction("-", make_unique<ConstantExpression>(Value::DOUBLE(1)), move(null_percentage));
	auto percentage_x =
	    SummarizeCreateBinaryFunction("*", move(negate_x), make_unique<ConstantExpression>(Value::DOUBLE(100)));
	auto round_x =
	    SummarizeCreateBinaryFunction("round", move(percentage_x), make_unique<ConstantExpression>(Value::INTEGER(2)));
	auto concat_x = SummarizeCreateBinaryFunction("concat", move(round_x), make_unique<ConstantExpression>(Value("%")));

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
	auto select = make_unique<SelectStatement>();
	select->node = move(query_copy);
	for (idx_t i = 0; i < plan.names.size(); i++) {
		name_children.push_back(make_unique<ConstantExpression>(Value(plan.names[i])));
		type_children.push_back(make_unique<ConstantExpression>(Value(plan.types[i].ToString())));
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
			avg_children.push_back(make_unique<ConstantExpression>(Value()));
			std_children.push_back(make_unique<ConstantExpression>(Value()));
			q25_children.push_back(make_unique<ConstantExpression>(Value()));
			q50_children.push_back(make_unique<ConstantExpression>(Value()));
			q75_children.push_back(make_unique<ConstantExpression>(Value()));
		}
		count_children.push_back(SummarizeCreateCountStar());
		null_percentage_children.push_back(SummarizeCreateNullPercentage(plan.names[i]));
	}
	auto subquery_ref = make_unique<SubqueryRef>(move(select), "summarize_tbl");
	subquery_ref->column_name_alias = plan.names;

	auto select_node = make_unique<SelectNode>();
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
	select_node->from_table = move(subquery_ref);

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
			auto &column = table.columns.GetColumn(check_column_id);
			update.expressions.push_back(make_unique<BoundColumnRefExpression>(
			    column.Type(), ColumnBinding(proj.table_index, proj.expressions.size())));
			proj.expressions.push_back(make_unique<BoundColumnRefExpression>(
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
	// check the constraints and indexes of the table to see if we need to project any additional columns
	// we do this for indexes with multiple columns and CHECK constraints in the UPDATE clause
	// suppose we have a constraint CHECK(i + j < 10); now we need both i and j to check the constraint
	// if we are only updating one of the two columns we add the other one to the UPDATE set
	// with a "useless" update (i.e. i=i) so we can verify that the CHECK constraint is not violated
	for (auto &constraint : table.bound_constraints) {
		if (constraint->type == ConstraintType::CHECK) {
			auto &check = *reinterpret_cast<BoundCheckConstraint *>(constraint.get());
			// check constraint! check if we need to add any extra columns to the UPDATE clause
			BindExtraColumns(table, get, proj, update, check.bound_columns);
		}
	}
	if (update.return_chunk) {
		physical_index_set_t all_columns;
		for (idx_t i = 0; i < table.storage->column_definitions.size(); i++) {
			all_columns.insert(PhysicalIndex(i));
		}
		BindExtraColumns(table, get, proj, update, all_columns);
	}
	// for index updates we always turn any update into an insert and a delete
	// we thus need all the columns to be available, hence we check if the update touches any index columns
	// If the returning keyword is used, we need access to the whole row in case the user requests it.
	// Therefore switch the update to a delete and insert.
	update.update_is_del_and_insert = false;
	table.storage->info->indexes.Scan([&](Index &index) {
		if (index.IndexIsUpdated(update.columns)) {
			update.update_is_del_and_insert = true;
			return true;
		}
		return false;
	});

	// we also convert any updates on LIST columns into delete + insert
	for (auto &col_index : update.columns) {
		auto &column = table.columns.GetColumn(col_index);
		if (!TypeSupportsRegularUpdate(column.Type())) {
			update.update_is_del_and_insert = true;
			break;
		}
	}

	if (update.update_is_del_and_insert) {
		// the update updates a column required by an index or requires returning the updated rows,
		// push projections for all columns
		physical_index_set_t all_columns;
		for (idx_t i = 0; i < table.storage->column_definitions.size(); i++) {
			all_columns.insert(PhysicalIndex(i));
		}
		BindExtraColumns(table, get, proj, update, all_columns);
	}
}

BoundStatement Binder::Bind(UpdateStatement &stmt) {
	BoundStatement result;
	unique_ptr<LogicalOperator> root;
	LogicalGet *get;

	// visit the table reference
	auto bound_table = Bind(*stmt.table);
	if (bound_table->type != TableReferenceType::BASE_TABLE) {
		throw BinderException("Can only update base table!");
	}
	auto &table_binding = (BoundBaseTableRef &)*bound_table;
	auto table = table_binding.table;

	// Add CTEs as bindable
	AddCTEMap(stmt.cte_map);

	if (stmt.from_table) {
		BoundCrossProductRef bound_crossproduct;
		bound_crossproduct.left = move(bound_table);
		bound_crossproduct.right = Bind(*stmt.from_table);
		root = CreatePlan(bound_crossproduct);
		get = (LogicalGet *)root->children[0].get();
	} else {
		root = CreatePlan(*bound_table);
		get = (LogicalGet *)root.get();
	}

	if (!table->temporary) {
		// update of persistent table: not read only!
		properties.read_only = false;
	}
	auto update = make_unique<LogicalUpdate>(table);

	// set return_chunk boolean early because it needs uses update_is_del_and_insert logic
	if (!stmt.returning_list.empty()) {
		update->return_chunk = true;
	}
	// bind the default values
	BindDefaultValues(table->columns, update->bound_defaults);

	// project any additional columns required for the condition/expressions
	if (stmt.condition) {
		WhereBinder binder(*this, context);
		auto condition = binder.Bind(stmt.condition);

		PlanSubqueries(&condition, &root);
		auto filter = make_unique<LogicalFilter>(move(condition));
		filter->AddChild(move(root));
		root = move(filter);
	}

	D_ASSERT(stmt.columns.size() == stmt.expressions.size());

	auto proj_index = GenerateTableIndex();
	vector<unique_ptr<Expression>> projection_expressions;

	for (idx_t i = 0; i < stmt.columns.size(); i++) {
		auto &colname = stmt.columns[i];
		auto &expr = stmt.expressions[i];
		if (!table->ColumnExists(colname)) {
			throw BinderException("Referenced update column %s not found in table!", colname);
		}
		auto &column = table->GetColumn(colname);
		if (column.Generated()) {
			throw BinderException("Cant update column \"%s\" because it is a generated column!", column.Name());
		}
		if (std::find(update->columns.begin(), update->columns.end(), column.Physical()) != update->columns.end()) {
			throw BinderException("Multiple assignments to same column \"%s\"", colname);
		}
		update->columns.push_back(column.Physical());

		if (expr->type == ExpressionType::VALUE_DEFAULT) {
			update->expressions.push_back(make_unique<BoundDefaultExpression>(column.Type()));
		} else {
			UpdateBinder binder(*this, context);
			binder.target_type = column.Type();
			auto bound_expr = binder.Bind(expr);
			PlanSubqueries(&bound_expr, &root);

			update->expressions.push_back(make_unique<BoundColumnRefExpression>(
			    bound_expr->return_type, ColumnBinding(proj_index, projection_expressions.size())));
			projection_expressions.push_back(move(bound_expr));
		}
	}

	// now create the projection
	auto proj = make_unique<LogicalProjection>(proj_index, move(projection_expressions));
	proj->AddChild(move(root));

	// bind any extra columns necessary for CHECK constraints or indexes
	BindUpdateConstraints(*table, *get, *proj, *update);

	// finally add the row id column to the projection list
	proj->expressions.push_back(make_unique<BoundColumnRefExpression>(
	    LogicalType::ROW_TYPE, ColumnBinding(get->table_index, get->column_ids.size())));
	get->column_ids.push_back(COLUMN_IDENTIFIER_ROW_ID);

	// set the projection as child of the update node and finalize the result
	update->AddChild(move(proj));

	auto update_table_index = GenerateTableIndex();
	update->table_index = update_table_index;
	if (!stmt.returning_list.empty()) {
		unique_ptr<LogicalOperator> update_as_logicaloperator = move(update);

		return BindReturning(move(stmt.returning_list), table, update_table_index, move(update_as_logicaloperator),
		                     move(result));
	}

	result.names = {"Count"};
	result.types = {LogicalType::BIGINT};
	result.plan = move(update);
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
		auto ref = unique_ptr_cast<BoundTableRef, BoundBaseTableRef>(move(bound_table));
		stmt.info->table = ref->table;

		auto &columns = stmt.info->columns;
		vector<unique_ptr<Expression>> select_list;
		if (columns.empty()) {
			// Empty means ALL columns should be vacuumed/analyzed
			auto &get = (LogicalGet &)*ref->get;
			columns.insert(columns.end(), get.names.begin(), get.names.end());
		}

		case_insensitive_set_t column_name_set;
		vector<string> non_generated_column_names;
		for (auto &col_name : columns) {
			if (column_name_set.count(col_name) > 0) {
				throw BinderException("Vacuum the same column twice(same name in column name list)");
			}
			column_name_set.insert(col_name);
			if (!ref->table->ColumnExists(col_name)) {
				throw BinderException("Column with name \"%s\" does not exist", col_name);
			}
			auto &col = ref->table->GetColumn(col_name);
			// ignore generated column
			if (col.Generated()) {
				continue;
			}
			non_generated_column_names.push_back(col_name);
			ColumnRefExpression colref(col_name, ref->table->name);
			auto result = bind_context.BindColumn(colref, 0);
			if (result.HasError()) {
				throw BinderException(result.error);
			}
			select_list.push_back(move(result.expression));
		}
		stmt.info->columns = move(non_generated_column_names);
		if (!select_list.empty()) {
			auto table_scan = CreatePlan(*ref);
			D_ASSERT(table_scan->type == LogicalOperatorType::LOGICAL_GET);

			auto &get = (LogicalGet &)*table_scan;

			D_ASSERT(select_list.size() == get.column_ids.size());
			D_ASSERT(stmt.info->columns.size() == get.column_ids.size());
			for (idx_t i = 0; i < get.column_ids.size(); i++) {
				stmt.info->column_id_map[i] =
				    ref->table->columns.LogicalToPhysical(LogicalIndex(get.column_ids[i])).index;
			}

			auto projection = make_unique<LogicalProjection>(GenerateTableIndex(), move(select_list));
			projection->children.push_back(move(table_scan));

			root = move(projection);
		} else {
			// eg. CREATE TABLE test (x AS (1));
			//     ANALYZE test;
			// Make it not a SINK so it doesn't have to do anything
			stmt.info->has_table = false;
		}
	}
	auto vacuum = make_unique<LogicalSimple>(LogicalOperatorType::LOGICAL_VACUUM, move(stmt.info));
	if (root) {
		vacuum->children.push_back(move(root));
	}

	result.names = {"Success"};
	result.types = {LogicalType::BOOLEAN};
	result.plan = move(vacuum);
	properties.return_type = StatementReturnType::NOTHING;
	return result;
}

} // namespace duckdb
















namespace duckdb {

unique_ptr<BoundTableRef> Binder::Bind(BaseTableRef &ref) {
	QueryErrorContext error_context(root_statement, ref.query_location);
	// CTEs and views are also referred to using BaseTableRefs, hence need to distinguish here
	// check if the table name refers to a CTE
	auto cte = FindCTE(ref.table_name, ref.table_name == alias);
	if (cte) {
		// Check if there is a CTE binding in the BindContext
		auto ctebinding = bind_context.GetCTEBinding(ref.table_name);
		if (!ctebinding) {
			if (CTEIsAlreadyBound(cte)) {
				throw BinderException("Circular reference to CTE \"%s\", use WITH RECURSIVE to use recursive CTEs",
				                      ref.table_name);
			}
			// Move CTE to subquery and bind recursively
			SubqueryRef subquery(unique_ptr_cast<SQLStatement, SelectStatement>(cte->query->Copy()));
			subquery.alias = ref.alias.empty() ? ref.table_name : ref.alias;
			subquery.column_name_alias = cte->aliases;
			for (idx_t i = 0; i < ref.column_name_alias.size(); i++) {
				if (i < subquery.column_name_alias.size()) {
					subquery.column_name_alias[i] = ref.column_name_alias[i];
				} else {
					subquery.column_name_alias.push_back(ref.column_name_alias[i]);
				}
			}
			return Bind(subquery, cte);
		} else {
			// There is a CTE binding in the BindContext.
			// This can only be the case if there is a recursive CTE present.
			auto index = GenerateTableIndex();
			auto result = make_unique<BoundCTERef>(index, ctebinding->index);
			auto b = ctebinding;
			auto alias = ref.alias.empty() ? ref.table_name : ref.alias;
			auto names = BindContext::AliasColumnNames(alias, b->names, ref.column_name_alias);

			bind_context.AddGenericBinding(index, alias, names, b->types);
			// Update references to CTE
			auto cteref = bind_context.cte_references[ref.table_name];
			(*cteref)++;

			result->types = b->types;
			result->bound_columns = move(names);
			return move(result);
		}
	}
	// not a CTE
	// extract a table or view from the catalog
	auto &catalog = Catalog::GetCatalog(context);
	auto table_or_view =
	    catalog.GetEntry(context, CatalogType::TABLE_ENTRY, ref.schema_name, ref.table_name, true, error_context);
	if (!table_or_view) {
		auto table_name = ref.schema_name.empty() ? ref.table_name : (ref.schema_name + "." + ref.table_name);
		// table could not be found: try to bind a replacement scan
		auto &config = DBConfig::GetConfig(context);
		for (auto &scan : config.replacement_scans) {
			auto replacement_function = scan.function(context, table_name, scan.data.get());
			if (replacement_function) {
				replacement_function->alias = ref.alias.empty() ? ref.table_name : ref.alias;
				replacement_function->column_name_alias = ref.column_name_alias;
				return Bind(*replacement_function);
			}
		}
		// we still didn't find the table
		if (GetBindingMode() == BindingMode::EXTRACT_NAMES) {
			// if we are in EXTRACT_NAMES, we create a dummy table ref
			AddTableName(table_name);

			// add a bind context entry
			auto table_index = GenerateTableIndex();
			auto alias = ref.alias.empty() ? table_name : ref.alias;
			vector<LogicalType> types {LogicalType::INTEGER};
			vector<string> names {"__dummy_col" + to_string(table_index)};
			bind_context.AddGenericBinding(table_index, alias, names, types);
			return make_unique_base<BoundTableRef, BoundEmptyTableRef>(table_index);
		}
		// could not find an alternative: bind again to get the error
		table_or_view = Catalog::GetCatalog(context).GetEntry(context, CatalogType::TABLE_ENTRY, ref.schema_name,
		                                                      ref.table_name, false, error_context);
	}
	switch (table_or_view->type) {
	case CatalogType::TABLE_ENTRY: {
		// base table: create the BoundBaseTableRef node
		auto table_index = GenerateTableIndex();
		auto table = (TableCatalogEntry *)table_or_view;

		auto scan_function = TableScanFunction::GetFunction();
		auto bind_data = make_unique<TableScanBindData>(table);
		auto alias = ref.alias.empty() ? ref.table_name : ref.alias;
		// TODO: bundle the type and name vector in a struct (e.g PackedColumnMetadata)
		vector<LogicalType> table_types;
		vector<string> table_names;
		vector<TableColumnType> table_categories;

		vector<LogicalType> return_types;
		vector<string> return_names;
		for (auto &col : table->columns.Logical()) {
			table_types.push_back(col.Type());
			table_names.push_back(col.Name());
			return_types.push_back(col.Type());
			return_names.push_back(col.Name());
		}
		table_names = BindContext::AliasColumnNames(alias, table_names, ref.column_name_alias);

		auto logical_get = make_unique<LogicalGet>(table_index, scan_function, move(bind_data), move(return_types),
		                                           move(return_names));
		bind_context.AddBaseTable(table_index, alias, table_names, table_types, logical_get->column_ids,
		                          logical_get->GetTable());
		return make_unique_base<BoundTableRef, BoundBaseTableRef>(table, move(logical_get));
	}
	case CatalogType::VIEW_ENTRY: {
		// the node is a view: get the query that the view represents
		auto view_catalog_entry = (ViewCatalogEntry *)table_or_view;
		// We need to use a new binder for the view that doesn't reference any CTEs
		// defined for this binder so there are no collisions between the CTEs defined
		// for the view and for the current query
		bool inherit_ctes = false;
		auto view_binder = Binder::CreateBinder(context, this, inherit_ctes);
		view_binder->can_contain_nulls = true;
		SubqueryRef subquery(unique_ptr_cast<SQLStatement, SelectStatement>(view_catalog_entry->query->Copy()));
		subquery.alias = ref.alias.empty() ? ref.table_name : ref.alias;
		subquery.column_name_alias =
		    BindContext::AliasColumnNames(subquery.alias, view_catalog_entry->aliases, ref.column_name_alias);
		// bind the child subquery
		view_binder->AddBoundView(view_catalog_entry);
		auto bound_child = view_binder->Bind(subquery);
		if (!view_binder->correlated_columns.empty()) {
			throw BinderException("Contents of view were altered - view bound correlated columns");
		}

		D_ASSERT(bound_child->type == TableReferenceType::SUBQUERY);
		// verify that the types and names match up with the expected types and names
		auto &bound_subquery = (BoundSubqueryRef &)*bound_child;
		if (bound_subquery.subquery->types != view_catalog_entry->types) {
			throw BinderException("Contents of view were altered: types don't match!");
		}
		bind_context.AddView(bound_subquery.subquery->GetRootIndex(), subquery.alias, subquery,
		                     *bound_subquery.subquery, view_catalog_entry);
		return bound_child;
	}
	default:
		throw InternalException("Catalog entry type");
	}
}
} // namespace duckdb




namespace duckdb {

unique_ptr<BoundTableRef> Binder::Bind(CrossProductRef &ref) {
	auto result = make_unique<BoundCrossProductRef>();
	result->left_binder = Binder::CreateBinder(context, this);
	result->right_binder = Binder::CreateBinder(context, this);
	auto &left_binder = *result->left_binder;
	auto &right_binder = *result->right_binder;

	result->left = left_binder.Bind(*ref.left);
	result->right = right_binder.Bind(*ref.right);

	bind_context.AddContext(move(left_binder.bind_context));
	bind_context.AddContext(move(right_binder.bind_context));
	MoveCorrelatedExpressions(left_binder);
	MoveCorrelatedExpressions(right_binder);
	return move(result);
}

} // namespace duckdb




namespace duckdb {

unique_ptr<BoundTableRef> Binder::Bind(EmptyTableRef &ref) {
	return make_unique<BoundEmptyTableRef>(GenerateTableIndex());
}

} // namespace duckdb







namespace duckdb {

unique_ptr<BoundTableRef> Binder::Bind(ExpressionListRef &expr) {
	auto result = make_unique<BoundExpressionListRef>();
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
			list.push_back(move(expr));
		}
		result->values.push_back(move(list));
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
				    BoundCastExpression::AddCastToType(context, move(list[val_idx]), result->types[val_idx]);
			}
		}
	}
	result->bind_index = GenerateTableIndex();
	bind_context.AddGenericBinding(result->bind_index, expr.alias, result->names, result->types);
	return move(result);
}

} // namespace duckdb












namespace duckdb {

static unique_ptr<ParsedExpression> BindColumn(Binder &binder, ClientContext &context, const string &alias,
                                               const string &column_name) {
	auto expr = make_unique_base<ParsedExpression, ColumnRefExpression>(column_name, alias);
	ExpressionBinder expr_binder(binder, context);
	auto result = expr_binder.Bind(expr);
	return make_unique<BoundExpression>(move(result));
}

static unique_ptr<ParsedExpression> AddCondition(ClientContext &context, Binder &left_binder, Binder &right_binder,
                                                 const string &left_alias, const string &right_alias,
                                                 const string &column_name) {
	ExpressionBinder expr_binder(left_binder, context);
	auto left = BindColumn(left_binder, context, left_alias, column_name);
	auto right = BindColumn(right_binder, context, right_alias, column_name);
	return make_unique<ComparisonExpression>(ExpressionType::COMPARE_EQUAL, move(left), move(right));
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

static void AddUsingBindings(UsingColumnSet &set, UsingColumnSet *input_set, const string &input_binding) {
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

string Binder::RetrieveUsingBinding(Binder &current_binder, UsingColumnSet *current_set, const string &using_column,
                                    const string &join_side, UsingColumnSet *new_set) {
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
	auto result = make_unique<BoundJoinRef>();
	result->left_binder = Binder::CreateBinder(context, this);
	result->right_binder = Binder::CreateBinder(context, this);
	auto &left_binder = *result->left_binder;
	auto &right_binder = *result->right_binder;

	result->type = ref.type;
	result->left = left_binder.Bind(*ref.left);
	result->right = right_binder.Bind(*ref.right);

	vector<unique_ptr<ParsedExpression>> extra_conditions;
	vector<string> extra_using_columns;
	if (ref.is_natural) {
		// natural join, figure out which column names are present in both sides of the join
		// first bind the left hand side and get a list of all the tables and column names
		case_insensitive_set_t lhs_columns;
		auto &lhs_binding_list = left_binder.bind_context.GetBindingsList();
		for (auto &binding : lhs_binding_list) {
			for (auto &column_name : binding.second->names) {
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
			for (auto &binding : lhs_binding_list) {
				for (auto &column_name : binding.second->names) {
					if (!left_candidates.empty()) {
						left_candidates += ", ";
					}
					left_candidates += binding.first + "." + column_name;
				}
			}
			for (auto &binding : rhs_binding_list) {
				for (auto &column_name : binding.second->names) {
					if (!right_candidates.empty()) {
						right_candidates += ", ";
					}
					right_candidates += binding.first + "." + column_name;
				}
			}
			error_msg += "\n   Left candidates: " + left_candidates;
			error_msg += "\n   Right candidates: " + right_candidates;
			throw BinderException(FormatError(ref, error_msg));
		}
	} else if (!ref.using_columns.empty()) {
		// USING columns
		D_ASSERT(!result->condition);
		extra_using_columns = ref.using_columns;
	}
	extra_using_columns = RemoveDuplicateUsingColumns(extra_using_columns);

	if (!extra_using_columns.empty()) {
		vector<UsingColumnSet *> left_using_bindings;
		vector<UsingColumnSet *> right_using_bindings;
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

			auto set = make_unique<UsingColumnSet>();
			auto left_using_binding = left_using_bindings[i];
			auto right_using_binding = right_using_bindings[i];
			left_binding = RetrieveUsingBinding(left_binder, left_using_binding, using_column, "left", set.get());
			right_binding = RetrieveUsingBinding(right_binder, right_using_binding, using_column, "right", set.get());

			extra_conditions.push_back(
			    AddCondition(context, left_binder, right_binder, left_binding, right_binding, using_column));

			AddUsingBindings(*set, left_using_binding, left_binding);
			AddUsingBindings(*set, right_using_binding, right_binding);
			SetPrimaryBinding(*set, ref.type, left_binding, right_binding);
			bind_context.TransferUsingBinding(left_binder.bind_context, left_using_binding, set.get(), left_binding,
			                                  using_column);
			bind_context.TransferUsingBinding(right_binder.bind_context, right_using_binding, set.get(), right_binding,
			                                  using_column);
			AddUsingBindingSet(move(set));
		}
	}

	bind_context.AddContext(move(left_binder.bind_context));
	bind_context.AddContext(move(right_binder.bind_context));
	MoveCorrelatedExpressions(left_binder);
	MoveCorrelatedExpressions(right_binder);
	for (auto &condition : extra_conditions) {
		if (ref.condition) {
			ref.condition = make_unique<ConjunctionExpression>(ExpressionType::CONJUNCTION_AND, move(ref.condition),
			                                                   move(condition));
		} else {
			ref.condition = move(condition);
		}
	}
	if (ref.condition) {
		WhereBinder binder(*this, context);
		result->condition = binder.Bind(ref.condition);
	}
	D_ASSERT(result->condition);
	return move(result);
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
				error_msg = "Candidates: " + named_params;
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

unique_ptr<BoundTableRef> Binder::Bind(SubqueryRef &ref, CommonTableExpressionInfo *cte) {
	auto binder = Binder::CreateBinder(context, this);
	binder->can_contain_nulls = true;
	if (cte) {
		binder->bound_ctes.insert(cte);
	}
	binder->alias = ref.alias.empty() ? "unnamed_subquery" : ref.alias;
	auto subquery = binder->BindNode(*ref.subquery->node);
	idx_t bind_index = subquery->GetRootIndex();
	string alias;
	if (ref.alias.empty()) {
		alias = "unnamed_subquery" + to_string(bind_index);
	} else {
		alias = ref.alias;
	}
	auto result = make_unique<BoundSubqueryRef>(move(binder), move(subquery));
	bind_context.AddSubquery(bind_index, alias, ref, *result->subquery);
	MoveCorrelatedExpressions(*result->binder);
	return move(result);
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
	if (expressions.size() == 1 && expressions[0]->type == ExpressionType::SUBQUERY) {
		// general case: argument is a subquery, bind it as part of the node
		auto &se = (SubqueryExpression &)*expressions[0];
		auto node = binder->BindNode(*se.subquery->node);
		subquery = make_unique<BoundSubqueryRef>(move(binder), move(node));
		return true;
	}
	// special case: non-subquery parameter to table-in table-out function
	// generate a subquery and bind that (i.e. UNNEST([1,2,3]) becomes UNNEST((SELECT [1,2,3]))
	auto select_node = make_unique<SelectNode>();
	select_node->select_list = move(expressions);
	select_node->from_table = make_unique<EmptyTableRef>();
	auto node = binder->BindNode(*select_node);
	subquery = make_unique<BoundSubqueryRef>(move(binder), move(node));
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
			auto &comp = (ComparisonExpression &)*child;
			if (comp.left->type == ExpressionType::COLUMN_REF) {
				auto &colref = (ColumnRefExpression &)*comp.left;
				if (!colref.IsQualified()) {
					parameter_name = colref.GetColumnName();
					child = move(comp.right);
				}
			}
		}
		if (child->type == ExpressionType::SUBQUERY) {
			if (seen_subquery) {
				error = "Table function can have at most one subquery parameter ";
				return false;
			}
			auto binder = Binder::CreateBinder(this->context, this, true);
			auto &se = (SubqueryExpression &)*child;
			auto node = binder->BindNode(*se.subquery->node);
			subquery = make_unique<BoundSubqueryRef>(move(binder), move(node));
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
			parameters.emplace_back(move(constant));
		} else {
			named_parameters[parameter_name] = move(constant);
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
	if (table_function.bind) {
		TableFunctionBindInput bind_input(parameters, named_parameters, input_table_types, input_table_names,
		                                  table_function.function_info.get());
		bind_data = table_function.bind(context, bind_input, return_types, return_names);
		if (table_function.name == "pandas_scan" || table_function.name == "arrow_scan") {
			auto arrow_bind = (PyTableFunctionData *)bind_data.get();
			arrow_bind->external_dependency = move(external_dependency);
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
	auto get = make_unique<LogicalGet>(bind_index, table_function, move(bind_data), return_types, return_names);
	get->parameters = parameters;
	get->named_parameters = named_parameters;
	get->input_table_types = input_table_types;
	get->input_table_names = input_table_names;
	// now add the table function to the bind context so its columns can be bound
	bind_context.AddTableFunction(bind_index, function_name, return_names, return_types, get->column_ids,
	                              get->GetTable());
	return move(get);
}

unique_ptr<LogicalOperator> Binder::BindTableFunction(TableFunction &function, vector<Value> parameters) {
	named_parameter_map_t named_parameters;
	vector<LogicalType> input_table_types;
	vector<string> input_table_names;
	vector<string> column_name_aliases;
	return BindTableFunctionInternal(function, function.name, move(parameters), move(named_parameters),
	                                 move(input_table_types), move(input_table_names), column_name_aliases, nullptr);
}

unique_ptr<BoundTableRef> Binder::Bind(TableFunctionRef &ref) {
	QueryErrorContext error_context(root_statement, ref.query_location);

	D_ASSERT(ref.function->type == ExpressionType::FUNCTION);
	auto fexpr = (FunctionExpression *)ref.function.get();

	TableFunctionCatalogEntry *function = nullptr;

	// fetch the function from the catalog
	auto &catalog = Catalog::GetCatalog(context);

	auto func_catalog = catalog.GetEntry(context, CatalogType::TABLE_FUNCTION_ENTRY, fexpr->schema,
	                                     fexpr->function_name, false, error_context);

	if (func_catalog->type == CatalogType::TABLE_FUNCTION_ENTRY) {
		function = (TableFunctionCatalogEntry *)func_catalog;
	} else if (func_catalog->type == CatalogType::TABLE_MACRO_ENTRY) {
		auto macro_func = (TableMacroCatalogEntry *)func_catalog;
		auto query_node = BindTableMacro(*fexpr, macro_func, 0);
		D_ASSERT(query_node);

		auto binder = Binder::CreateBinder(context, this);
		binder->can_contain_nulls = true;

		binder->alias = ref.alias.empty() ? "unnamed_query" : ref.alias;
		auto query = binder->BindNode(*query_node);

		idx_t bind_index = query->GetRootIndex();
		// string alias;
		string alias = (ref.alias.empty() ? "unnamed_query" + to_string(bind_index) : ref.alias);

		auto result = make_unique<BoundSubqueryRef>(move(binder), move(query));
		// remember ref here is TableFunctionRef and NOT base class
		bind_context.AddSubquery(bind_index, alias, ref, *result->subquery);
		MoveCorrelatedExpressions(*result->binder);
		return move(result);
	}

	// evaluate the input parameters to the function
	vector<LogicalType> arguments;
	vector<Value> parameters;
	named_parameter_map_t named_parameters;
	unique_ptr<BoundSubqueryRef> subquery;
	string error;
	if (!BindTableFunctionParameters(*function, fexpr->children, arguments, parameters, named_parameters, subquery,
	                                 error)) {
		throw BinderException(FormatError(ref, error));
	}

	// select the function based on the input parameters
	FunctionBinder function_binder(context);
	idx_t best_function_idx = function_binder.BindFunction(function->name, function->functions, arguments, error);
	if (best_function_idx == DConstants::INVALID_INDEX) {
		throw BinderException(FormatError(ref, error));
	}
	auto table_function = function->functions.GetFunctionByOffset(best_function_idx);

	// now check the named parameters
	BindNamedParameters(table_function.named_parameters, named_parameters, error_context, table_function.name);

	// cast the parameters to the type of the function
	for (idx_t i = 0; i < arguments.size(); i++) {
		if (table_function.arguments[i] != LogicalType::ANY && table_function.arguments[i] != LogicalType::TABLE &&
		    table_function.arguments[i] != LogicalType::POINTER &&
		    table_function.arguments[i].id() != LogicalTypeId::LIST) {
			parameters[i] = parameters[i].CastAs(context, table_function.arguments[i]);
		}
	}

	vector<LogicalType> input_table_types;
	vector<string> input_table_names;

	if (subquery) {
		input_table_types = subquery->subquery->types;
		input_table_names = subquery->subquery->names;
	}
	auto get = BindTableFunctionInternal(table_function, ref.alias.empty() ? fexpr->function_name : ref.alias,
	                                     move(parameters), move(named_parameters), move(input_table_types),
	                                     move(input_table_names), ref.column_name_alias, move(ref.external_dependency));
	if (subquery) {
		get->children.push_back(Binder::CreatePlan(*subquery));
	}

	return make_unique_base<BoundTableRef, BoundTableFunction>(move(get));
}

} // namespace duckdb




namespace duckdb {

unique_ptr<LogicalOperator> Binder::CreatePlan(BoundBaseTableRef &ref) {
	return move(ref.get);
}

} // namespace duckdb




namespace duckdb {

unique_ptr<LogicalOperator> Binder::CreatePlan(BoundCrossProductRef &expr) {
	auto left = CreatePlan(*expr.left);
	auto right = CreatePlan(*expr.right);

	return LogicalCrossProduct::Create(move(left), move(right));
}

} // namespace duckdb




namespace duckdb {

unique_ptr<LogicalOperator> Binder::CreatePlan(BoundCTERef &ref) {
	auto index = ref.bind_index;

	vector<LogicalType> types;
	for (auto &type : ref.types) {
		types.push_back(type);
	}

	return make_unique<LogicalCTERef>(index, ref.cte_index, types, ref.bound_columns);
}

} // namespace duckdb




namespace duckdb {

unique_ptr<LogicalOperator> Binder::CreatePlan(BoundEmptyTableRef &ref) {
	return make_unique<LogicalDummyScan>(ref.bind_index);
}

} // namespace duckdb





namespace duckdb {

unique_ptr<LogicalOperator> Binder::CreatePlan(BoundExpressionListRef &ref) {
	auto root = make_unique_base<LogicalOperator, LogicalDummyScan>(GenerateTableIndex());
	// values list, first plan any subqueries in the list
	for (auto &expr_list : ref.values) {
		for (auto &expr : expr_list) {
			PlanSubqueries(&expr, &root);
		}
	}
	// now create a LogicalExpressionGet from the set of expressions
	// fetch the types
	vector<LogicalType> types;
	for (auto &expr : ref.values[0]) {
		types.push_back(expr->return_type);
	}
	auto expr_get = make_unique<LogicalExpressionGet>(ref.bind_index, types, move(ref.values));
	expr_get->AddChild(move(root));
	return move(expr_get);
}

} // namespace duckdb















namespace duckdb {

//! Create a JoinCondition from a comparison
static bool CreateJoinCondition(Expression &expr, unordered_set<idx_t> &left_bindings,
                                unordered_set<idx_t> &right_bindings, vector<JoinCondition> &conditions) {
	// comparison
	auto &comparison = (BoundComparisonExpression &)expr;
	auto left_side = JoinSide::GetJoinSide(*comparison.left, left_bindings, right_bindings);
	auto right_side = JoinSide::GetJoinSide(*comparison.right, left_bindings, right_bindings);
	if (left_side != JoinSide::BOTH && right_side != JoinSide::BOTH) {
		// join condition can be divided in a left/right side
		JoinCondition condition;
		condition.comparison = expr.type;
		auto left = move(comparison.left);
		auto right = move(comparison.right);
		if (left_side == JoinSide::RIGHT) {
			// left = right, right = left, flip the comparison symbol and reverse sides
			swap(left, right);
			condition.comparison = FlipComparisionExpression(expr.type);
		}
		condition.left = move(left);
		condition.right = move(right);
		conditions.push_back(move(condition));
		return true;
	}
	return false;
}

unique_ptr<LogicalOperator> LogicalComparisonJoin::CreateJoin(JoinType type, unique_ptr<LogicalOperator> left_child,
                                                              unique_ptr<LogicalOperator> right_child,
                                                              unordered_set<idx_t> &left_bindings,
                                                              unordered_set<idx_t> &right_bindings,
                                                              vector<unique_ptr<Expression>> &expressions) {
	vector<JoinCondition> conditions;
	vector<unique_ptr<Expression>> arbitrary_expressions;
	// first check if we can create
	for (auto &expr : expressions) {
		auto total_side = JoinSide::GetJoinSide(*expr, left_bindings, right_bindings);
		if (total_side != JoinSide::BOTH) {
			// join condition does not reference both sides, add it as filter under the join
			if (type == JoinType::LEFT && total_side == JoinSide::RIGHT) {
				// filter is on RHS and the join is a LEFT OUTER join, we can push it in the right child
				if (right_child->type != LogicalOperatorType::LOGICAL_FILTER) {
					// not a filter yet, push a new empty filter
					auto filter = make_unique<LogicalFilter>();
					filter->AddChild(move(right_child));
					right_child = move(filter);
				}
				// push the expression into the filter
				auto &filter = (LogicalFilter &)*right_child;
				filter.expressions.push_back(move(expr));
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
		arbitrary_expressions.push_back(move(expr));
	}
	bool need_to_consider_arbitrary_expressions = true;
	if (type == JoinType::INNER) {
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
			arbitrary_expressions.push_back(make_unique<BoundConstantExpression>(Value::BOOLEAN(true)));
		}
		for (auto &condition : conditions) {
			arbitrary_expressions.push_back(JoinCondition::CreateExpression(move(condition)));
		}
		// if we get here we could not create any JoinConditions
		// turn this into an arbitrary expression join
		auto any_join = make_unique<LogicalAnyJoin>(type);
		// create the condition
		any_join->children.push_back(move(left_child));
		any_join->children.push_back(move(right_child));
		// AND all the arbitrary expressions together
		// do the same with any remaining conditions
		any_join->condition = move(arbitrary_expressions[0]);
		for (idx_t i = 1; i < arbitrary_expressions.size(); i++) {
			any_join->condition = make_unique<BoundConjunctionExpression>(
			    ExpressionType::CONJUNCTION_AND, move(any_join->condition), move(arbitrary_expressions[i]));
		}
		return move(any_join);
	} else {
		// we successfully converted expressions into JoinConditions
		// create a LogicalComparisonJoin
		auto comp_join = make_unique<LogicalComparisonJoin>(type);
		comp_join->conditions = move(conditions);
		comp_join->children.push_back(move(left_child));
		comp_join->children.push_back(move(right_child));
		if (!arbitrary_expressions.empty()) {
			// we have some arbitrary expressions as well
			// add them to a filter
			auto filter = make_unique<LogicalFilter>();
			for (auto &expr : arbitrary_expressions) {
				filter->expressions.push_back(move(expr));
			}
			LogicalFilter::SplitPredicates(filter->expressions);
			filter->children.push_back(move(comp_join));
			return move(filter);
		}
		return move(comp_join);
	}
}

static bool HasCorrelatedColumns(Expression &expression) {
	if (expression.type == ExpressionType::BOUND_COLUMN_REF) {
		auto &colref = (BoundColumnRefExpression &)expression;
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

unique_ptr<LogicalOperator> Binder::CreatePlan(BoundJoinRef &ref) {
	auto left = CreatePlan(*ref.left);
	auto right = CreatePlan(*ref.right);
	if (ref.type == JoinType::RIGHT && ClientConfig::GetConfig(context).enable_optimizer) {
		// we turn any right outer joins into left outer joins for optimization purposes
		// they are the same but with sides flipped, so treating them the same simplifies life
		ref.type = JoinType::LEFT;
		std::swap(left, right);
	}
	if (ref.type == JoinType::INNER && (ref.condition->HasSubquery() || HasCorrelatedColumns(*ref.condition))) {
		// inner join, generate a cross product + filter
		// this will be later turned into a proper join by the join order optimizer
		auto root = LogicalCrossProduct::Create(move(left), move(right));

		auto filter = make_unique<LogicalFilter>(move(ref.condition));
		// visit the expressions in the filter
		for (auto &expression : filter->expressions) {
			PlanSubqueries(&expression, &root);
		}
		filter->AddChild(move(root));
		return move(filter);
	}

	// split the expressions by the AND clause
	vector<unique_ptr<Expression>> expressions;
	expressions.push_back(move(ref.condition));
	LogicalFilter::SplitPredicates(expressions);

	// find the table bindings on the LHS and RHS of the join
	unordered_set<idx_t> left_bindings, right_bindings;
	LogicalJoin::GetTableReferences(*left, left_bindings);
	LogicalJoin::GetTableReferences(*right, right_bindings);
	// now create the join operator from the set of join conditions
	auto result = LogicalComparisonJoin::CreateJoin(ref.type, move(left), move(right), left_bindings, right_bindings,
	                                                expressions);

	LogicalOperator *join;
	if (result->type == LogicalOperatorType::LOGICAL_FILTER) {
		join = result->children[0].get();
	} else {
		join = result.get();
	}
	for (auto &child : join->children) {
		if (child->type == LogicalOperatorType::LOGICAL_FILTER) {
			auto &filter = (LogicalFilter &)*child;
			for (auto &expr : filter.expressions) {
				PlanSubqueries(&expr, &filter.children[0]);
			}
		}
	}

	// we visit the expressions depending on the type of join
	if (join->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
		// comparison join
		// in this join we visit the expressions on the LHS with the LHS as root node
		// and the expressions on the RHS with the RHS as root node
		auto &comp_join = (LogicalComparisonJoin &)*join;
		for (idx_t i = 0; i < comp_join.conditions.size(); i++) {
			PlanSubqueries(&comp_join.conditions[i].left, &comp_join.children[0]);
			PlanSubqueries(&comp_join.conditions[i].right, &comp_join.children[1]);
		}
	} else if (join->type == LogicalOperatorType::LOGICAL_ANY_JOIN) {
		auto &any_join = (LogicalAnyJoin &)*join;
		// for the any join we just visit the condition
		if (any_join.condition->HasSubquery()) {
			throw NotImplementedException("Cannot perform non-inner join on subquery!");
		}
	}
	return result;
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
	return move(ref.get);
}

} // namespace duckdb
















#include <algorithm>

namespace duckdb {

shared_ptr<Binder> Binder::CreateBinder(ClientContext &context, Binder *parent, bool inherit_ctes) {
	return make_shared<Binder>(true, context, parent ? parent->shared_from_this() : nullptr, inherit_ctes);
}

Binder::Binder(bool, ClientContext &context, shared_ptr<Binder> parent_p, bool inherit_ctes_p)
    : context(context), parent(move(parent_p)), bound_tables(0), inherit_ctes(inherit_ctes_p) {
	parameters = nullptr;
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
		return Bind((SelectStatement &)statement);
	case StatementType::INSERT_STATEMENT:
		return Bind((InsertStatement &)statement);
	case StatementType::COPY_STATEMENT:
		return Bind((CopyStatement &)statement);
	case StatementType::DELETE_STATEMENT:
		return Bind((DeleteStatement &)statement);
	case StatementType::UPDATE_STATEMENT:
		return Bind((UpdateStatement &)statement);
	case StatementType::RELATION_STATEMENT:
		return Bind((RelationStatement &)statement);
	case StatementType::CREATE_STATEMENT:
		return Bind((CreateStatement &)statement);
	case StatementType::DROP_STATEMENT:
		return Bind((DropStatement &)statement);
	case StatementType::ALTER_STATEMENT:
		return Bind((AlterStatement &)statement);
	case StatementType::TRANSACTION_STATEMENT:
		return Bind((TransactionStatement &)statement);
	case StatementType::PRAGMA_STATEMENT:
		return Bind((PragmaStatement &)statement);
	case StatementType::EXPLAIN_STATEMENT:
		return Bind((ExplainStatement &)statement);
	case StatementType::VACUUM_STATEMENT:
		return Bind((VacuumStatement &)statement);
	case StatementType::SHOW_STATEMENT:
		return Bind((ShowStatement &)statement);
	case StatementType::CALL_STATEMENT:
		return Bind((CallStatement &)statement);
	case StatementType::EXPORT_STATEMENT:
		return Bind((ExportStatement &)statement);
	case StatementType::SET_STATEMENT:
		return Bind((SetStatement &)statement);
	case StatementType::LOAD_STATEMENT:
		return Bind((LoadStatement &)statement);
	case StatementType::EXTENSION_STATEMENT:
		return Bind((ExtensionStatement &)statement);
	case StatementType::PREPARE_STATEMENT:
		return Bind((PrepareStatement &)statement);
	case StatementType::EXECUTE_STATEMENT:
		return Bind((ExecuteStatement &)statement);
	case StatementType::LOGICAL_PLAN_STATEMENT:
		return Bind((LogicalPlanStatement &)statement);
	default: // LCOV_EXCL_START
		throw NotImplementedException("Unimplemented statement type \"%s\" for Bind",
		                              StatementTypeToString(statement.type));
	} // LCOV_EXCL_STOP
}

void Binder::AddCTEMap(CommonTableExpressionMap &cte_map) {
	for (auto &cte_it : cte_map.map) {
		AddCTE(cte_it.first, cte_it.second.get());
	}
}

unique_ptr<BoundQueryNode> Binder::BindNode(QueryNode &node) {
	// first we visit the set of CTEs and add them to the bind context
	AddCTEMap(node.cte_map);
	// now we bind the node
	unique_ptr<BoundQueryNode> result;
	switch (node.type) {
	case QueryNodeType::SELECT_NODE:
		result = BindNode((SelectNode &)node);
		break;
	case QueryNodeType::RECURSIVE_CTE_NODE:
		result = BindNode((RecursiveCTENode &)node);
		break;
	default:
		D_ASSERT(node.type == QueryNodeType::SET_OPERATION_NODE);
		result = BindNode((SetOperationNode &)node);
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
		return CreatePlan((BoundSelectNode &)node);
	case QueryNodeType::SET_OPERATION_NODE:
		return CreatePlan((BoundSetOperationNode &)node);
	case QueryNodeType::RECURSIVE_CTE_NODE:
		return CreatePlan((BoundRecursiveCTENode &)node);
	default:
		throw InternalException("Unsupported bound query node type");
	}
}

unique_ptr<BoundTableRef> Binder::Bind(TableRef &ref) {
	unique_ptr<BoundTableRef> result;
	switch (ref.type) {
	case TableReferenceType::BASE_TABLE:
		result = Bind((BaseTableRef &)ref);
		break;
	case TableReferenceType::CROSS_PRODUCT:
		result = Bind((CrossProductRef &)ref);
		break;
	case TableReferenceType::JOIN:
		result = Bind((JoinRef &)ref);
		break;
	case TableReferenceType::SUBQUERY:
		result = Bind((SubqueryRef &)ref);
		break;
	case TableReferenceType::EMPTY:
		result = Bind((EmptyTableRef &)ref);
		break;
	case TableReferenceType::TABLE_FUNCTION:
		result = Bind((TableFunctionRef &)ref);
		break;
	case TableReferenceType::EXPRESSION_LIST:
		result = Bind((ExpressionListRef &)ref);
		break;
	default:
		throw InternalException("Unknown table ref type");
	}
	result->sample = move(ref.sample);
	return result;
}

unique_ptr<LogicalOperator> Binder::CreatePlan(BoundTableRef &ref) {
	unique_ptr<LogicalOperator> root;
	switch (ref.type) {
	case TableReferenceType::BASE_TABLE:
		root = CreatePlan((BoundBaseTableRef &)ref);
		break;
	case TableReferenceType::SUBQUERY:
		root = CreatePlan((BoundSubqueryRef &)ref);
		break;
	case TableReferenceType::JOIN:
		root = CreatePlan((BoundJoinRef &)ref);
		break;
	case TableReferenceType::CROSS_PRODUCT:
		root = CreatePlan((BoundCrossProductRef &)ref);
		break;
	case TableReferenceType::TABLE_FUNCTION:
		root = CreatePlan((BoundTableFunction &)ref);
		break;
	case TableReferenceType::EMPTY:
		root = CreatePlan((BoundEmptyTableRef &)ref);
		break;
	case TableReferenceType::EXPRESSION_LIST:
		root = CreatePlan((BoundExpressionListRef &)ref);
		break;
	case TableReferenceType::CTE:
		root = CreatePlan((BoundCTERef &)ref);
		break;
	default:
		throw InternalException("Unsupported bound table ref type type");
	}
	// plan the sample clause
	if (ref.sample) {
		root = make_unique<LogicalSample>(move(ref.sample), move(root));
	}
	return root;
}

void Binder::AddCTE(const string &name, CommonTableExpressionInfo *info) {
	D_ASSERT(info);
	D_ASSERT(!name.empty());
	auto entry = CTE_bindings.find(name);
	if (entry != CTE_bindings.end()) {
		throw InternalException("Duplicate CTE \"%s\" in query!", name);
	}
	CTE_bindings[name] = info;
}

CommonTableExpressionInfo *Binder::FindCTE(const string &name, bool skip) {
	auto entry = CTE_bindings.find(name);
	if (entry != CTE_bindings.end()) {
		if (!skip || entry->second->query->node->type == QueryNodeType::RECURSIVE_CTE_NODE) {
			return entry->second;
		}
	}
	if (parent && inherit_ctes) {
		return parent->FindCTE(name, name == alias);
	}
	return nullptr;
}

bool Binder::CTEIsAlreadyBound(CommonTableExpressionInfo *cte) {
	if (bound_ctes.find(cte) != bound_ctes.end()) {
		return true;
	}
	if (parent && inherit_ctes) {
		return parent->CTEIsAlreadyBound(cte);
	}
	return false;
}

void Binder::AddBoundView(ViewCatalogEntry *view) {
	// check if the view is already bound
	auto current = this;
	while (current) {
		if (current->bound_views.find(view) != current->bound_views.end()) {
			throw BinderException("infinite recursion detected: attempting to recursively bind view \"%s\"",
			                      view->name);
		}
		current = current->parent.get();
	}
	bound_views.insert(view);
}

idx_t Binder::GenerateTableIndex() {
	if (parent) {
		return parent->GenerateTableIndex();
	}
	return bound_tables++;
}

void Binder::PushExpressionBinder(ExpressionBinder *binder) {
	GetActiveBinders().push_back(binder);
}

void Binder::PopExpressionBinder() {
	D_ASSERT(HasActiveBinder());
	GetActiveBinders().pop_back();
}

void Binder::SetActiveBinder(ExpressionBinder *binder) {
	D_ASSERT(HasActiveBinder());
	GetActiveBinders().back() = binder;
}

ExpressionBinder *Binder::GetActiveBinder() {
	return GetActiveBinders().back();
}

bool Binder::HasActiveBinder() {
	return !GetActiveBinders().empty();
}

vector<ExpressionBinder *> &Binder::GetActiveBinders() {
	if (parent) {
		return parent->GetActiveBinders();
	}
	return active_binders;
}

void Binder::AddUsingBindingSet(unique_ptr<UsingColumnSet> set) {
	if (parent) {
		parent->AddUsingBindingSet(move(set));
		return;
	}
	bind_context.AddUsingBindingSet(move(set));
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
	Binding *binding = nullptr;
	D_ASSERT(!lambda_bindings);
	if (macro_binding && table_name == macro_binding->alias) {
		binding = macro_binding;
	} else {
		binding = bind_context.GetBinding(table_name, error_message);
	}

	if (!binding) {
		return false;
	}
	if (!schema_name.empty()) {
		auto catalog_entry = binding->GetStandardEntry();
		if (!catalog_entry) {
			return false;
		}
		if (catalog_entry->schema->name != schema_name || catalog_entry->name != table_name) {
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
		parent->AddTableName(move(table_name));
		return;
	}
	table_names.insert(move(table_name));
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

BoundStatement Binder::BindReturning(vector<unique_ptr<ParsedExpression>> returning_list, TableCatalogEntry *table,
                                     idx_t update_table_index, unique_ptr<LogicalOperator> child_operator,
                                     BoundStatement result) {

	vector<LogicalType> types;
	vector<std::string> names;

	auto binder = Binder::CreateBinder(context);

	vector<column_t> bound_columns;
	idx_t column_count = 0;
	for (auto &col : table->columns.Logical()) {
		names.push_back(col.Name());
		types.push_back(col.Type());
		if (!col.Generated()) {
			bound_columns.push_back(column_count);
		}
		column_count++;
	}

	binder->bind_context.AddBaseTable(update_table_index, table->name, names, types, bound_columns, table);
	ReturningBinder returning_binder(*binder, context);

	vector<unique_ptr<Expression>> projection_expressions;
	LogicalType result_type;
	for (auto &returning_expr : returning_list) {
		auto expr_type = returning_expr->GetExpressionType();
		if (expr_type == ExpressionType::STAR) {
			auto generated_star_list = vector<unique_ptr<ParsedExpression>>();
			binder->bind_context.GenerateAllColumnExpressions((StarExpression &)*returning_expr, generated_star_list);

			for (auto &star_column : generated_star_list) {
				auto star_expr = returning_binder.Bind(star_column, &result_type);
				result.types.push_back(result_type);
				result.names.push_back(star_expr->GetName());
				projection_expressions.push_back(move(star_expr));
			}
		} else {
			auto expr = returning_binder.Bind(returning_expr, &result_type);
			result.names.push_back(expr->GetName());
			result.types.push_back(result_type);
			projection_expressions.push_back(move(expr));
		}
	}

	auto projection = make_unique<LogicalProjection>(GenerateTableIndex(), move(projection_expressions));
	projection->AddChild(move(child_operator));
	D_ASSERT(result.types.size() == result.names.size());
	result.plan = move(projection);
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
    : type(type), null_order(null_order), expression(move(expression)) {
}
BoundOrderByNode::BoundOrderByNode(OrderType type, OrderByNullType null_order, unique_ptr<Expression> expression,
                                   unique_ptr<BaseStatistics> stats)
    : type(type), null_order(null_order), expression(move(expression)), stats(move(stats)) {
}

BoundOrderByNode BoundOrderByNode::Copy() const {
	if (stats) {
		return BoundOrderByNode(type, null_order, expression->Copy(), stats->Copy());
	} else {
		return BoundOrderByNode(type, null_order, expression->Copy());
	}
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
	return BoundOrderByNode(type, null_order, move(expression));
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
      function(move(function)), children(move(children)), bind_info(move(bind_info)), aggr_type(aggr_type),
      filter(move(filter)) {
	D_ASSERT(!function.name.empty());
}

string BoundAggregateExpression::ToString() const {
	return FunctionExpression::ToString<BoundAggregateExpression, Expression>(*this, string(), function.name, false,
	                                                                          IsDistinct(), filter.get());
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
	auto other = (BoundAggregateExpression *)other_p;
	if (other->aggr_type != aggr_type) {
		return false;
	}
	if (other->function != function) {
		return false;
	}
	if (children.size() != other->children.size()) {
		return false;
	}
	if (!Expression::Equals(other->filter.get(), filter.get())) {
		return false;
	}
	for (idx_t i = 0; i < children.size(); i++) {
		if (!Expression::Equals(children[i].get(), other->children[i].get())) {
			return false;
		}
	}
	if (!FunctionData::Equals(bind_info.get(), other->bind_info.get())) {
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
	for (auto &child : children) {
		new_children.push_back(child->Copy());
	}
	auto new_bind_info = bind_info ? bind_info->Copy() : nullptr;
	auto new_filter = filter ? filter->Copy() : nullptr;
	auto copy = make_unique<BoundAggregateExpression>(function, move(new_children), move(new_filter),
	                                                  move(new_bind_info), aggr_type);
	copy->CopyProperties(*this);
	return move(copy);
}

void BoundAggregateExpression::Serialize(FieldWriter &writer) const {
	writer.WriteField(IsDistinct());
	writer.WriteOptional(filter);
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

	return make_unique<BoundAggregateExpression>(function, move(children), move(filter), move(bind_info),
	                                             distinct ? AggregateType::DISTINCT : AggregateType::NON_DISTINCT);
}

} // namespace duckdb




namespace duckdb {

BoundBetweenExpression::BoundBetweenExpression(unique_ptr<Expression> input, unique_ptr<Expression> lower,
                                               unique_ptr<Expression> upper, bool lower_inclusive, bool upper_inclusive)
    : Expression(ExpressionType::COMPARE_BETWEEN, ExpressionClass::BOUND_BETWEEN, LogicalType::BOOLEAN),
      input(move(input)), lower(move(lower)), upper(move(upper)), lower_inclusive(lower_inclusive),
      upper_inclusive(upper_inclusive) {
}

string BoundBetweenExpression::ToString() const {
	return BetweenExpression::ToString<BoundBetweenExpression, Expression>(*this);
}

bool BoundBetweenExpression::Equals(const BaseExpression *other_p) const {
	if (!Expression::Equals(other_p)) {
		return false;
	}
	auto other = (BoundBetweenExpression *)other_p;
	if (!Expression::Equals(input.get(), other->input.get())) {
		return false;
	}
	if (!Expression::Equals(lower.get(), other->lower.get())) {
		return false;
	}
	if (!Expression::Equals(upper.get(), other->upper.get())) {
		return false;
	}
	return lower_inclusive == other->lower_inclusive && upper_inclusive == other->upper_inclusive;
}

unique_ptr<Expression> BoundBetweenExpression::Copy() {
	auto copy = make_unique<BoundBetweenExpression>(input->Copy(), lower->Copy(), upper->Copy(), lower_inclusive,
	                                                upper_inclusive);
	copy->CopyProperties(*this);
	return move(copy);
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
	return make_unique<BoundBetweenExpression>(move(input), move(lower), move(upper), lower_inclusive, upper_inclusive);
}

} // namespace duckdb




namespace duckdb {

BoundCaseExpression::BoundCaseExpression(LogicalType type)
    : Expression(ExpressionType::CASE_EXPR, ExpressionClass::BOUND_CASE, move(type)) {
}

BoundCaseExpression::BoundCaseExpression(unique_ptr<Expression> when_expr, unique_ptr<Expression> then_expr,
                                         unique_ptr<Expression> else_expr_p)
    : Expression(ExpressionType::CASE_EXPR, ExpressionClass::BOUND_CASE, then_expr->return_type),
      else_expr(move(else_expr_p)) {
	BoundCaseCheck check;
	check.when_expr = move(when_expr);
	check.then_expr = move(then_expr);
	case_checks.push_back(move(check));
}

string BoundCaseExpression::ToString() const {
	return CaseExpression::ToString<BoundCaseExpression, Expression>(*this);
}

bool BoundCaseExpression::Equals(const BaseExpression *other_p) const {
	if (!Expression::Equals(other_p)) {
		return false;
	}
	auto &other = (BoundCaseExpression &)*other_p;
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
	auto new_case = make_unique<BoundCaseExpression>(return_type);
	for (auto &check : case_checks) {
		BoundCaseCheck new_check;
		new_check.when_expr = check.when_expr->Copy();
		new_check.then_expr = check.then_expr->Copy();
		new_case->case_checks.push_back(move(new_check));
	}
	new_case->else_expr = else_expr->Copy();

	new_case->CopyProperties(*this);
	return move(new_case);
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
	result.when_expr = move(when_expr);
	result.then_expr = move(then_expr);
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

	auto result = make_unique<BoundCaseExpression>(return_type);
	result->else_expr = move(else_expr);
	result->case_checks = move(case_checks);
	return move(result);
}

} // namespace duckdb








namespace duckdb {

BoundCastExpression::BoundCastExpression(unique_ptr<Expression> child_p, LogicalType target_type_p,
                                         BoundCastInfo bound_cast_p, bool try_cast_p)
    : Expression(ExpressionType::OPERATOR_CAST, ExpressionClass::BOUND_CAST, move(target_type_p)), child(move(child_p)),
      try_cast(try_cast_p), bound_cast(move(bound_cast_p)) {
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
	return make_unique<BoundCastExpression>(move(expr), target_type, move(bound_cast), try_cast);
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
		auto &parameter = (BoundParameterExpression &)*expr;
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
		auto &def = (BoundDefaultExpression &)*expr;
		def.return_type = target_type;
	}
	if (!target_type.IsValid()) {
		return expr;
	}

	auto cast_function = cast_functions.GetCastFunction(expr->return_type, target_type, get_input);
	return AddCastExpressionInternal(move(expr), target_type, move(cast_function), try_cast);
}

unique_ptr<Expression> BoundCastExpression::AddDefaultCastToType(unique_ptr<Expression> expr,
                                                                 const LogicalType &target_type, bool try_cast) {
	CastFunctionSet default_set;
	GetCastFunctionInput get_input;
	return AddCastToTypeInternal(move(expr), target_type, default_set, get_input, try_cast);
}

unique_ptr<Expression> BoundCastExpression::AddCastToType(ClientContext &context, unique_ptr<Expression> expr,
                                                          const LogicalType &target_type, bool try_cast) {
	auto &cast_functions = DBConfig::GetConfig(context).GetCastFunctions();
	GetCastFunctionInput get_input(context);
	return AddCastToTypeInternal(move(expr), target_type, cast_functions, get_input, try_cast);
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
	auto other = (BoundCastExpression *)other_p;
	if (!Expression::Equals(child.get(), other->child.get())) {
		return false;
	}
	if (try_cast != other->try_cast) {
		return false;
	}
	return true;
}

unique_ptr<Expression> BoundCastExpression::Copy() {
	auto copy = make_unique<BoundCastExpression>(child->Copy(), return_type, bound_cast.Copy(), try_cast);
	copy->CopyProperties(*this);
	return move(copy);
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
	return make_unique<BoundCastExpression>(move(child), move(target_type), move(cast_function), try_cast);
}

} // namespace duckdb






namespace duckdb {

BoundColumnRefExpression::BoundColumnRefExpression(string alias_p, LogicalType type, ColumnBinding binding, idx_t depth)
    : Expression(ExpressionType::BOUND_COLUMN_REF, ExpressionClass::BOUND_COLUMN_REF, move(type)), binding(binding),
      depth(depth) {
	this->alias = move(alias_p);
}

BoundColumnRefExpression::BoundColumnRefExpression(LogicalType type, ColumnBinding binding, idx_t depth)
    : BoundColumnRefExpression(string(), move(type), binding, depth) {
}

unique_ptr<Expression> BoundColumnRefExpression::Copy() {
	return make_unique<BoundColumnRefExpression>(alias, return_type, binding, depth);
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
	auto other = (BoundColumnRefExpression *)other_p;
	return other->binding == binding && other->depth == depth;
}

string BoundColumnRefExpression::ToString() const {
	if (!alias.empty()) {
		return alias;
	}
	return "#[" + to_string(binding.table_index) + "." + to_string(binding.column_index) + "]";
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

	return make_unique<BoundColumnRefExpression>(alias, return_type, ColumnBinding(table_index, column_index), depth);
}

} // namespace duckdb




namespace duckdb {

BoundComparisonExpression::BoundComparisonExpression(ExpressionType type, unique_ptr<Expression> left,
                                                     unique_ptr<Expression> right)
    : Expression(type, ExpressionClass::BOUND_COMPARISON, LogicalType::BOOLEAN), left(move(left)), right(move(right)) {
}

string BoundComparisonExpression::ToString() const {
	return ComparisonExpression::ToString<BoundComparisonExpression, Expression>(*this);
}

bool BoundComparisonExpression::Equals(const BaseExpression *other_p) const {
	if (!Expression::Equals(other_p)) {
		return false;
	}
	auto other = (BoundComparisonExpression *)other_p;
	if (!Expression::Equals(left.get(), other->left.get())) {
		return false;
	}
	if (!Expression::Equals(right.get(), other->right.get())) {
		return false;
	}

	return true;
}

unique_ptr<Expression> BoundComparisonExpression::Copy() {
	auto copy = make_unique<BoundComparisonExpression>(type, left->Copy(), right->Copy());
	copy->CopyProperties(*this);
	return move(copy);
}

void BoundComparisonExpression::Serialize(FieldWriter &writer) const {
	writer.WriteOptional(left);
	writer.WriteOptional(right);
}

unique_ptr<Expression> BoundComparisonExpression::Deserialize(ExpressionDeserializationState &state,
                                                              FieldReader &reader) {
	auto left = reader.ReadOptional<Expression>(nullptr, state.gstate);
	auto right = reader.ReadOptional<Expression>(nullptr, state.gstate);
	return make_unique<BoundComparisonExpression>(state.type, move(left), move(right));
}

} // namespace duckdb





namespace duckdb {

BoundConjunctionExpression::BoundConjunctionExpression(ExpressionType type)
    : Expression(type, ExpressionClass::BOUND_CONJUNCTION, LogicalType::BOOLEAN) {
}

BoundConjunctionExpression::BoundConjunctionExpression(ExpressionType type, unique_ptr<Expression> left,
                                                       unique_ptr<Expression> right)
    : BoundConjunctionExpression(type) {
	children.push_back(move(left));
	children.push_back(move(right));
}

string BoundConjunctionExpression::ToString() const {
	return ConjunctionExpression::ToString<BoundConjunctionExpression, Expression>(*this);
}

bool BoundConjunctionExpression::Equals(const BaseExpression *other_p) const {
	if (!Expression::Equals(other_p)) {
		return false;
	}
	auto other = (BoundConjunctionExpression *)other_p;
	return ExpressionUtil::SetEquals(children, other->children);
}

bool BoundConjunctionExpression::PropagatesNullValues() const {
	return false;
}

unique_ptr<Expression> BoundConjunctionExpression::Copy() {
	auto copy = make_unique<BoundConjunctionExpression>(type);
	for (auto &expr : children) {
		copy->children.push_back(expr->Copy());
	}
	copy->CopyProperties(*this);
	return move(copy);
}

void BoundConjunctionExpression::Serialize(FieldWriter &writer) const {
	writer.WriteSerializableList(children);
}

unique_ptr<Expression> BoundConjunctionExpression::Deserialize(ExpressionDeserializationState &state,
                                                               FieldReader &reader) {
	auto children = reader.ReadRequiredSerializableList<Expression>(state.gstate);
	auto res = make_unique<BoundConjunctionExpression>(state.type);
	res->children = move(children);
	return move(res);
}

} // namespace duckdb





namespace duckdb {

BoundConstantExpression::BoundConstantExpression(Value value_p)
    : Expression(ExpressionType::VALUE_CONSTANT, ExpressionClass::BOUND_CONSTANT, value_p.type()),
      value(move(value_p)) {
}

string BoundConstantExpression::ToString() const {
	return value.ToSQLString();
}

bool BoundConstantExpression::Equals(const BaseExpression *other_p) const {
	if (!Expression::Equals(other_p)) {
		return false;
	}
	auto other = (BoundConstantExpression *)other_p;
	return value.type() == other->value.type() && !ValueOperations::DistinctFrom(value, other->value);
}

hash_t BoundConstantExpression::Hash() const {
	hash_t result = Expression::Hash();
	return CombineHash(value.Hash(), result);
}

unique_ptr<Expression> BoundConstantExpression::Copy() {
	auto copy = make_unique<BoundConstantExpression>(value);
	copy->CopyProperties(*this);
	return move(copy);
}

void BoundConstantExpression::Serialize(FieldWriter &writer) const {
	value.Serialize(writer.GetSerializer());
}

unique_ptr<Expression> BoundConstantExpression::Deserialize(ExpressionDeserializationState &state,
                                                            FieldReader &reader) {
	auto value = Value::Deserialize(reader.GetSource());
	return make_unique<BoundConstantExpression>(value);
}

} // namespace duckdb



namespace duckdb {

void BoundDefaultExpression::Serialize(FieldWriter &writer) const {
	throw NotImplementedException(ExpressionTypeToString(type));
}

} // namespace duckdb


namespace duckdb {

BoundExpression::BoundExpression(unique_ptr<Expression> expr)
    : ParsedExpression(ExpressionType::INVALID, ExpressionClass::BOUND_EXPRESSION), expr(move(expr)) {
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

} // namespace duckdb







namespace duckdb {

BoundFunctionExpression::BoundFunctionExpression(LogicalType return_type, ScalarFunction bound_function,
                                                 vector<unique_ptr<Expression>> arguments,
                                                 unique_ptr<FunctionData> bind_info, bool is_operator)
    : Expression(ExpressionType::BOUND_FUNCTION, ExpressionClass::BOUND_FUNCTION, move(return_type)),
      function(move(bound_function)), children(move(arguments)), bind_info(move(bind_info)), is_operator(is_operator) {
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
	auto other = (BoundFunctionExpression *)other_p;
	if (other->function != function) {
		return false;
	}
	if (!ExpressionUtil::ListEquals(children, other->children)) {
		return false;
	}
	if (!FunctionData::Equals(bind_info.get(), other->bind_info.get())) {
		return false;
	}
	return true;
}

unique_ptr<Expression> BoundFunctionExpression::Copy() {
	vector<unique_ptr<Expression>> new_children;
	for (auto &child : children) {
		new_children.push_back(child->Copy());
	}
	unique_ptr<FunctionData> new_bind_info = bind_info ? bind_info->Copy() : nullptr;

	auto copy = make_unique<BoundFunctionExpression>(return_type, function, move(new_children), move(new_bind_info),
	                                                 is_operator);
	copy->CopyProperties(*this);
	return move(copy);
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
	return make_unique<BoundFunctionExpression>(move(return_type), move(function), move(children), move(bind_info),
	                                            is_operator);
}
} // namespace duckdb




namespace duckdb {

BoundLambdaExpression::BoundLambdaExpression(ExpressionType type_p, LogicalType return_type_p,
                                             unique_ptr<Expression> lambda_expr_p, idx_t parameter_count_p)
    : Expression(type_p, ExpressionClass::BOUND_LAMBDA, move(return_type_p)), lambda_expr(move(lambda_expr_p)),
      parameter_count(parameter_count_p) {
}

string BoundLambdaExpression::ToString() const {
	return lambda_expr->ToString();
}

bool BoundLambdaExpression::Equals(const BaseExpression *other_p) const {
	if (!Expression::Equals(other_p)) {
		return false;
	}
	auto other = (BoundLambdaExpression *)other_p;
	if (!Expression::Equals(lambda_expr.get(), other->lambda_expr.get())) {
		return false;
	}
	if (!ExpressionUtil::ListEquals(captures, other->captures)) {
		return false;
	}
	if (parameter_count != other->parameter_count) {
		return false;
	}
	return true;
}

unique_ptr<Expression> BoundLambdaExpression::Copy() {
	auto copy = make_unique<BoundLambdaExpression>(type, return_type, lambda_expr->Copy(), parameter_count);
	for (auto &capture : captures) {
		copy->captures.push_back(capture->Copy());
	}
	return move(copy);
}

void BoundLambdaExpression::Serialize(FieldWriter &writer) const {
	throw NotImplementedException(ExpressionTypeToString(type));
}

} // namespace duckdb






namespace duckdb {

BoundOperatorExpression::BoundOperatorExpression(ExpressionType type, LogicalType return_type)
    : Expression(type, ExpressionClass::BOUND_OPERATOR, move(return_type)) {
}

string BoundOperatorExpression::ToString() const {
	return OperatorExpression::ToString<BoundOperatorExpression, Expression>(*this);
}

bool BoundOperatorExpression::Equals(const BaseExpression *other_p) const {
	if (!Expression::Equals(other_p)) {
		return false;
	}
	auto other = (BoundOperatorExpression *)other_p;
	if (!ExpressionUtil::ListEquals(children, other->children)) {
		return false;
	}
	return true;
}

unique_ptr<Expression> BoundOperatorExpression::Copy() {
	auto copy = make_unique<BoundOperatorExpression>(type, return_type);
	copy->CopyProperties(*this);
	for (auto &child : children) {
		copy->children.push_back(child->Copy());
	}
	return move(copy);
}

void BoundOperatorExpression::Serialize(FieldWriter &writer) const {
	writer.WriteSerializable(return_type);
	writer.WriteSerializableList(children);
}

unique_ptr<Expression> BoundOperatorExpression::Deserialize(ExpressionDeserializationState &state,
                                                            FieldReader &reader) {
	auto return_type = reader.ReadRequiredSerializable<LogicalType, LogicalType>();
	auto children = reader.ReadRequiredSerializableList<Expression>(state.gstate);

	auto result = make_unique<BoundOperatorExpression>(state.type, return_type);
	result->children = move(children);
	return move(result);
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
	auto &bound_parameter = (BoundParameterExpression &)expr;
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
	auto other = (BoundParameterExpression *)other_p;
	return parameter_nr == other->parameter_nr;
}

hash_t BoundParameterExpression::Hash() const {
	hash_t result = Expression::Hash();
	result = CombineHash(duckdb::Hash(parameter_nr), result);
	return result;
}

unique_ptr<Expression> BoundParameterExpression::Copy() {
	auto result = make_unique<BoundParameterExpression>(parameter_nr);
	result->parameter_data = parameter_data;
	result->return_type = return_type;
	result->CopyProperties(*this);
	return move(result);
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
	auto result = make_unique<BoundParameterExpression>(parameter_nr);
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
	result->parameter_data = move(parameter_data);
	return move(result);
}

} // namespace duckdb







namespace duckdb {

BoundReferenceExpression::BoundReferenceExpression(string alias, LogicalType type, idx_t index)
    : Expression(ExpressionType::BOUND_REF, ExpressionClass::BOUND_REF, move(type)), index(index) {
	this->alias = move(alias);
}
BoundReferenceExpression::BoundReferenceExpression(LogicalType type, idx_t index)
    : BoundReferenceExpression(string(), move(type), index) {
}

string BoundReferenceExpression::ToString() const {
	if (!alias.empty()) {
		return alias;
	}
	return "#" + to_string(index);
}

bool BoundReferenceExpression::Equals(const BaseExpression *other_p) const {
	if (!Expression::Equals(other_p)) {
		return false;
	}
	auto other = (BoundReferenceExpression *)other_p;
	return other->index == index;
}

hash_t BoundReferenceExpression::Hash() const {
	return CombineHash(Expression::Hash(), duckdb::Hash<idx_t>(index));
}

unique_ptr<Expression> BoundReferenceExpression::Copy() {
	return make_unique<BoundReferenceExpression>(alias, return_type, index);
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
	return make_unique<BoundReferenceExpression>(alias, return_type, index);
}

} // namespace duckdb




namespace duckdb {

BoundSubqueryExpression::BoundSubqueryExpression(LogicalType return_type)
    : Expression(ExpressionType::SUBQUERY, ExpressionClass::BOUND_SUBQUERY, move(return_type)) {
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
    : Expression(ExpressionType::BOUND_UNNEST, ExpressionClass::BOUND_UNNEST, move(return_type)) {
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
	auto other = (BoundUnnestExpression *)other_p;
	if (!Expression::Equals(child.get(), other->child.get())) {
		return false;
	}
	return true;
}

unique_ptr<Expression> BoundUnnestExpression::Copy() {
	auto copy = make_unique<BoundUnnestExpression>(return_type);
	copy->child = child->Copy();
	return move(copy);
}

void BoundUnnestExpression::Serialize(FieldWriter &writer) const {
	writer.WriteSerializable(return_type);
	writer.WriteSerializable(*child);
}

unique_ptr<Expression> BoundUnnestExpression::Deserialize(ExpressionDeserializationState &state, FieldReader &reader) {
	auto return_type = reader.ReadRequiredSerializable<LogicalType, LogicalType>();
	auto child = reader.ReadRequiredSerializable<Expression>(state.gstate);

	auto result = make_unique<BoundUnnestExpression>(return_type);
	result->child = move(child);
	return move(result);
}

} // namespace duckdb








namespace duckdb {

BoundWindowExpression::BoundWindowExpression(ExpressionType type, LogicalType return_type,
                                             unique_ptr<AggregateFunction> aggregate,
                                             unique_ptr<FunctionData> bind_info)
    : Expression(type, ExpressionClass::BOUND_WINDOW, move(return_type)), aggregate(move(aggregate)),
      bind_info(move(bind_info)), ignore_nulls(false) {
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
	auto other = (BoundWindowExpression *)other_p;

	if (ignore_nulls != other->ignore_nulls) {
		return false;
	}
	if (start != other->start || end != other->end) {
		return false;
	}
	// check if the child expressions are equivalent
	if (other->children.size() != children.size()) {
		return false;
	}
	for (idx_t i = 0; i < children.size(); i++) {
		if (!Expression::Equals(children[i].get(), other->children[i].get())) {
			return false;
		}
	}
	// check if the filter expressions are equivalent
	if (!Expression::Equals(filter_expr.get(), other->filter_expr.get())) {
		return false;
	}

	// check if the framing expressions are equivalent
	if (!Expression::Equals(start_expr.get(), other->start_expr.get()) ||
	    !Expression::Equals(end_expr.get(), other->end_expr.get()) ||
	    !Expression::Equals(offset_expr.get(), other->offset_expr.get()) ||
	    !Expression::Equals(default_expr.get(), other->default_expr.get())) {
		return false;
	}

	return KeysAreCompatible(other);
}

bool BoundWindowExpression::KeysAreCompatible(const BoundWindowExpression *other) const {
	// check if the partitions are equivalent
	if (partitions.size() != other->partitions.size()) {
		return false;
	}
	for (idx_t i = 0; i < partitions.size(); i++) {
		if (!Expression::Equals(partitions[i].get(), other->partitions[i].get())) {
			return false;
		}
	}
	// check if the orderings are equivalent
	if (orders.size() != other->orders.size()) {
		return false;
	}
	for (idx_t i = 0; i < orders.size(); i++) {
		if (orders[i].type != other->orders[i].type) {
			return false;
		}
		if (!BaseExpression::Equals((BaseExpression *)orders[i].expression.get(),
		                            (BaseExpression *)other->orders[i].expression.get())) {
			return false;
		}
	}
	return true;
}

unique_ptr<Expression> BoundWindowExpression::Copy() {
	auto new_window = make_unique<BoundWindowExpression>(type, return_type, nullptr, nullptr);
	new_window->CopyProperties(*this);

	if (aggregate) {
		new_window->aggregate = make_unique<AggregateFunction>(*aggregate);
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
			new_window->partitions_stats.push_back(ps->Copy());
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

	return move(new_window);
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
		aggregate = make_unique<AggregateFunction>(move(aggr_function));
		return_type = aggregate->return_type;
	} else {
		children = reader.ReadRequiredSerializableList<Expression>(state.gstate);
		return_type = reader.ReadRequiredSerializable<LogicalType, LogicalType>();
	}
	auto result = make_unique<BoundWindowExpression>(state.type, return_type, move(aggregate), move(bind_info));

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
	result->children = move(children);
	return move(result);
}

} // namespace duckdb









namespace duckdb {

Expression::Expression(ExpressionType type, ExpressionClass expression_class, LogicalType return_type)
    : BaseExpression(type, expression_class), return_type(move(return_type)) {
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
	    type == ExpressionType::CONJUNCTION_OR || type == ExpressionType::CONJUNCTION_AND) {
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

BindResult AggregateBinder::BindExpression(unique_ptr<ParsedExpression> *expr_ptr, idx_t depth, bool root_expression) {
	auto &expr = **expr_ptr;
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
	this->target_type = move(target_type);
}

BindResult AlterBinder::BindExpression(unique_ptr<ParsedExpression> *expr_ptr, idx_t depth, bool root_expression) {
	auto &expr = **expr_ptr;
	switch (expr.GetExpressionClass()) {
	case ExpressionClass::WINDOW:
		return BindResult("window functions are not allowed in alter statement");
	case ExpressionClass::SUBQUERY:
		return BindResult("cannot use subquery in alter statement");
	case ExpressionClass::COLUMN_REF:
		return BindColumn((ColumnRefExpression &)expr);
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
	if (table.columns.GetColumn(LogicalIndex(idx)).Generated()) {
		throw BinderException("Using generated columns in alter statement not supported");
	}
	bound_columns.push_back(idx);
	return BindResult(
	    make_unique<BoundReferenceExpression>(table.columns.GetColumn(idx).Type(), bound_columns.size() - 1));
}

} // namespace duckdb





namespace duckdb {

CheckBinder::CheckBinder(Binder &binder, ClientContext &context, string table_p, const ColumnList &columns,
                         physical_index_set_t &bound_columns)
    : ExpressionBinder(binder, context), table(move(table_p)), columns(columns), bound_columns(bound_columns) {
	target_type = LogicalType::INTEGER;
}

BindResult CheckBinder::BindExpression(unique_ptr<ParsedExpression> *expr_ptr, idx_t depth, bool root_expression) {
	auto &expr = **expr_ptr;
	switch (expr.GetExpressionClass()) {
	case ExpressionClass::WINDOW:
		return BindResult("window functions are not allowed in check constraints");
	case ExpressionClass::SUBQUERY:
		return BindResult("cannot use subquery in check constraint");
	case ExpressionClass::COLUMN_REF:
		return BindCheckColumn((ColumnRefExpression &)expr);
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
	auto result = make_unique_base<ParsedExpression, ColumnRefExpression>(colref.column_names.back());
	for (idx_t i = struct_start; i + 1 < colref.column_names.size(); i++) {
		result = CreateStructExtract(move(result), colref.column_names[i]);
	}
	return BindExpression(&result, 0);
}

BindResult CheckBinder::BindCheckColumn(ColumnRefExpression &colref) {
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
		return BindExpression(&bound_expression, 0, false);
	}
	bound_columns.insert(col.Physical());
	D_ASSERT(col.StorageOid() != DConstants::INVALID_INDEX);
	return BindResult(make_unique<BoundReferenceExpression>(col.Type(), col.StorageOid()));
}

} // namespace duckdb






namespace duckdb {

ColumnAliasBinder::ColumnAliasBinder(BoundSelectNode &node, const case_insensitive_map_t<idx_t> &alias_map)
    : node(node), alias_map(alias_map), in_alias(false) {
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

	if (in_alias) {
		return BindResult("Cannot resolve self-referential alias");
	}

	// found an alias: bind the alias expression
	auto expression = node.original_expressions[alias_entry->second]->Copy();
	in_alias = true;

	// since the alias has been found, pass a depth of 0. See Issue 4978 (#16)
	// ColumnAliasBinders are only in Having, Qualify and Where Binders
	auto result = enclosing_binder.BindExpression(&expression, 0, root_expression);
	in_alias = false;
	return result;
}

} // namespace duckdb


namespace duckdb {

ConstantBinder::ConstantBinder(Binder &binder, ClientContext &context, string clause)
    : ExpressionBinder(binder, context), clause(move(clause)) {
}

BindResult ConstantBinder::BindExpression(unique_ptr<ParsedExpression> *expr_ptr, idx_t depth, bool root_expression) {
	auto &expr = **expr_ptr;
	switch (expr.GetExpressionClass()) {
	case ExpressionClass::COLUMN_REF:
		return BindResult(clause + " cannot contain column names");
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

BindResult GroupBinder::BindExpression(unique_ptr<ParsedExpression> *expr_ptr, idx_t depth, bool root_expression) {
	auto &expr = **expr_ptr;
	if (root_expression && depth == 0) {
		switch (expr.expression_class) {
		case ExpressionClass::COLUMN_REF:
			return BindColumnRef((ColumnRefExpression &)expr);
		case ExpressionClass::CONSTANT:
			return BindConstant((ConstantExpression &)expr);
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
		return BindResult(make_unique<BoundConstantExpression>(Value::INTEGER(42)));
	}
	if (entry >= node.select_list.size()) {
		throw BinderException("GROUP BY term out of range - should be between 1 and %d", (int)node.select_list.size());
	}
	// we replace the root expression, also replace the unbound expression
	unbound_expression = node.select_list[entry]->Copy();
	// move the expression that this refers to here and bind it
	auto select_entry = move(node.select_list[entry]);
	auto binding = Bind(select_entry, nullptr, false);
	// now replace the original expression in the select list with a reference to this group
	group_alias_map[to_string(entry)] = bind_index;
	node.select_list[entry] = make_unique<ColumnRefExpression>(to_string(entry));
	// insert into the set of used aliases
	used_aliases.insert(entry);
	return BindResult(move(binding));
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
    : SelectBinder(binder, context, node, info), column_alias_binder(node, alias_map),
      aggregate_handling(aggregate_handling) {
	target_type = LogicalType(LogicalTypeId::BOOLEAN);
}

BindResult HavingBinder::BindColumnRef(unique_ptr<ParsedExpression> *expr_ptr, idx_t depth, bool root_expression) {
	auto &expr = (ColumnRefExpression &)**expr_ptr;
	auto alias_result = column_alias_binder.BindAlias(*this, expr, depth, root_expression);
	if (!alias_result.HasError()) {
		return alias_result;
	}
	if (aggregate_handling == AggregateHandling::FORCE_AGGREGATES) {
		auto expr = duckdb::SelectBinder::BindExpression(expr_ptr, depth);
		if (expr.HasError()) {
			return expr;
		}
		auto group_ref = make_unique<BoundColumnRefExpression>(
		    expr.expression->return_type, ColumnBinding(node.group_index, node.groups.group_expressions.size()));
		node.groups.group_expressions.push_back(move(expr.expression));
		return BindResult(move(group_ref));
	}
	return BindResult(StringUtil::Format(
	    "column %s must appear in the GROUP BY clause or be used in an aggregate function", expr.ToString()));
}

BindResult HavingBinder::BindExpression(unique_ptr<ParsedExpression> *expr_ptr, idx_t depth, bool root_expression) {
	auto &expr = **expr_ptr;
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
		return duckdb::SelectBinder::BindExpression(expr_ptr, depth);
	}
}

} // namespace duckdb


namespace duckdb {

IndexBinder::IndexBinder(Binder &binder, ClientContext &context) : ExpressionBinder(binder, context) {
}

BindResult IndexBinder::BindExpression(unique_ptr<ParsedExpression> *expr_ptr, idx_t depth, bool root_expression) {
	auto &expr = **expr_ptr;
	switch (expr.expression_class) {
	case ExpressionClass::WINDOW:
		return BindResult("window functions are not allowed in index expressions");
	case ExpressionClass::SUBQUERY:
		return BindResult("cannot use subquery in index expressions");
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

BindResult InsertBinder::BindExpression(unique_ptr<ParsedExpression> *expr_ptr, idx_t depth, bool root_expression) {
	auto &expr = **expr_ptr;
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

OrderBinder::OrderBinder(vector<Binder *> binders, idx_t projection_index, case_insensitive_map_t<idx_t> &alias_map,
                         expression_map_t<idx_t> &projection_map, idx_t max_count)
    : binders(move(binders)), projection_index(projection_index), max_count(max_count), extra_list(nullptr),
      alias_map(alias_map), projection_map(projection_map) {
}
OrderBinder::OrderBinder(vector<Binder *> binders, idx_t projection_index, SelectNode &node,
                         case_insensitive_map_t<idx_t> &alias_map, expression_map_t<idx_t> &projection_map)
    : binders(move(binders)), projection_index(projection_index), alias_map(alias_map), projection_map(projection_map) {
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
	return make_unique<BoundColumnRefExpression>(move(alias), LogicalType::INVALID,
	                                             ColumnBinding(projection_index, index));
}

unique_ptr<Expression> OrderBinder::CreateExtraReference(unique_ptr<ParsedExpression> expr) {
	if (!extra_list) {
		throw InternalException("CreateExtraReference called without extra_list");
	}
	auto result = CreateProjectionReference(*expr, extra_list->size());
	extra_list->push_back(move(expr));
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
		auto &constant = (ConstantExpression &)*expr;
		return BindConstant(*expr, constant.value);
	}
	case ExpressionClass::COLUMN_REF: {
		// COLUMN REF expression
		// check if we can bind it to an alias in the select list
		auto &colref = (ColumnRefExpression &)*expr;
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
		auto &posref = (PositionalReferenceExpression &)*expr;
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
	auto entry = projection_map.find(expr.get());
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
	return CreateExtraReference(move(expr));
}

} // namespace duckdb








namespace duckdb {

QualifyBinder::QualifyBinder(Binder &binder, ClientContext &context, BoundSelectNode &node, BoundGroupInformation &info,
                             case_insensitive_map_t<idx_t> &alias_map)
    : SelectBinder(binder, context, node, info), column_alias_binder(node, alias_map) {
	target_type = LogicalType(LogicalTypeId::BOOLEAN);
}

BindResult QualifyBinder::BindColumnRef(unique_ptr<ParsedExpression> *expr_ptr, idx_t depth, bool root_expression) {
	auto &expr = (ColumnRefExpression &)**expr_ptr;
	auto result = duckdb::SelectBinder::BindExpression(expr_ptr, depth);
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

BindResult QualifyBinder::BindExpression(unique_ptr<ParsedExpression> *expr_ptr, idx_t depth, bool root_expression) {
	auto &expr = **expr_ptr;
	// check if the expression binds to one of the groups
	auto group_index = TryBindGroup(expr, depth);
	if (group_index != DConstants::INVALID_INDEX) {
		return BindGroup(expr, depth, group_index);
	}
	switch (expr.expression_class) {
	case ExpressionClass::WINDOW:
		return BindWindow((WindowExpression &)expr, depth);
	case ExpressionClass::COLUMN_REF:
		return BindColumnRef(expr_ptr, depth, root_expression);
	default:
		return duckdb::SelectBinder::BindExpression(expr_ptr, depth);
	}
}

} // namespace duckdb


namespace duckdb {

RelationBinder::RelationBinder(Binder &binder, ClientContext &context, string op)
    : ExpressionBinder(binder, context), op(move(op)) {
}

BindResult RelationBinder::BindExpression(unique_ptr<ParsedExpression> *expr_ptr, idx_t depth, bool root_expression) {
	auto &expr = **expr_ptr;
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

BindResult ReturningBinder::BindColumnRef(unique_ptr<ParsedExpression> *expr_ptr, idx_t depth) {
	auto &expr = **expr_ptr;
	if (expr.GetName() == "rowid") {
		return BindResult("rowid is not supported in returning statements");
	}
	return ExpressionBinder::BindExpression(expr_ptr, depth);
}

BindResult ReturningBinder::BindExpression(unique_ptr<ParsedExpression> *expr_ptr, idx_t depth, bool root_expression) {
	auto &expr = **expr_ptr;
	switch (expr.GetExpressionClass()) {
	case ExpressionClass::SUBQUERY:
		return BindResult("SUBQUERY is not supported in returning statements");
	case ExpressionClass::BOUND_SUBQUERY:
		return BindResult("BOUND SUBQUERY is not supported in returning statements");
	case ExpressionClass::COLUMN_REF:
		return BindColumnRef(expr_ptr, depth);
	default:
		return ExpressionBinder::BindExpression(expr_ptr, depth);
	}
}

} // namespace duckdb













namespace duckdb {

SelectBinder::SelectBinder(Binder &binder, ClientContext &context, BoundSelectNode &node, BoundGroupInformation &info,
                           case_insensitive_map_t<idx_t> alias_map)
    : ExpressionBinder(binder, context), inside_window(false), node(node), info(info), alias_map(move(alias_map)) {
}

SelectBinder::SelectBinder(Binder &binder, ClientContext &context, BoundSelectNode &node, BoundGroupInformation &info)
    : SelectBinder(binder, context, node, info, case_insensitive_map_t<idx_t>()) {
}

BindResult SelectBinder::BindExpression(unique_ptr<ParsedExpression> *expr_ptr, idx_t depth, bool root_expression) {
	auto &expr = **expr_ptr;
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
		return BindWindow((WindowExpression &)expr, depth);
	default:
		return ExpressionBinder::BindExpression(expr_ptr, depth);
	}
}

idx_t SelectBinder::TryBindGroup(ParsedExpression &expr, idx_t depth) {
	// first check the group alias map, if expr is a ColumnRefExpression
	if (expr.type == ExpressionType::COLUMN_REF) {
		auto &colref = (ColumnRefExpression &)expr;
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
	auto entry = info.map.find(&expr);
	if (entry != info.map.end()) {
		return entry->second;
	}
#ifdef DEBUG
	for (auto entry : info.map) {
		D_ASSERT(!entry.first->Equals(&expr));
		D_ASSERT(!expr.Equals(entry.first));
	}
#endif
	return DConstants::INVALID_INDEX;
}

BindResult SelectBinder::BindColumnRef(unique_ptr<ParsedExpression> *expr_ptr, idx_t depth) {
	// first try to bind the column reference regularly
	auto result = ExpressionBinder::BindExpression(expr_ptr, depth);
	if (!result.HasError()) {
		return result;
	}
	// binding failed
	// check in the alias map
	auto &colref = (ColumnRefExpression &)**expr_ptr;
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
			return BindResult(node.select_list[index]->Copy());
		}
	}
	// entry was not found in the alias map: return the original error
	return result;
}

BindResult SelectBinder::BindGroupingFunction(OperatorExpression &op, idx_t depth) {
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
	node.grouping_functions.push_back(move(group_indexes));
	return BindResult(make_unique<BoundColumnRefExpression>(op.GetName(), LogicalType::BIGINT,
	                                                        ColumnBinding(node.groupings_index, col_idx), depth));
}

BindResult SelectBinder::BindGroup(ParsedExpression &expr, idx_t depth, idx_t group_index) {
	auto &group = node.groups.group_expressions[group_index];
	return BindResult(make_unique<BoundColumnRefExpression>(expr.GetName(), group->return_type,
	                                                        ColumnBinding(node.group_index, group_index), depth));
}

} // namespace duckdb




namespace duckdb {

TableFunctionBinder::TableFunctionBinder(Binder &binder, ClientContext &context) : ExpressionBinder(binder, context) {
}

BindResult TableFunctionBinder::BindColumnReference(ColumnRefExpression &expr) {
	auto result_name = StringUtil::Join(expr.column_names, ".");
	return BindResult(make_unique<BoundConstantExpression>(Value(result_name)));
}

BindResult TableFunctionBinder::BindExpression(unique_ptr<ParsedExpression> *expr_ptr, idx_t depth,
                                               bool root_expression) {
	auto &expr = **expr_ptr;
	switch (expr.GetExpressionClass()) {
	case ExpressionClass::COLUMN_REF:
		return BindColumnReference((ColumnRefExpression &)expr);
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

BindResult UpdateBinder::BindExpression(unique_ptr<ParsedExpression> *expr_ptr, idx_t depth, bool root_expression) {
	auto &expr = **expr_ptr;
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

WhereBinder::WhereBinder(Binder &binder, ClientContext &context, ColumnAliasBinder *column_alias_binder)
    : ExpressionBinder(binder, context), column_alias_binder(column_alias_binder) {
	target_type = LogicalType(LogicalTypeId::BOOLEAN);
}

BindResult WhereBinder::BindColumnRef(unique_ptr<ParsedExpression> *expr_ptr, idx_t depth, bool root_expression) {
	auto &expr = (ColumnRefExpression &)**expr_ptr;
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

BindResult WhereBinder::BindExpression(unique_ptr<ParsedExpression> *expr_ptr, idx_t depth, bool root_expression) {
	auto &expr = **expr_ptr;
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
    : binder(binder), context(context), stored_binder(nullptr) {
	if (replace_binder) {
		stored_binder = binder.GetActiveBinder();
		binder.SetActiveBinder(this);
	} else {
		binder.PushExpressionBinder(this);
	}
}

ExpressionBinder::~ExpressionBinder() {
	if (binder.HasActiveBinder()) {
		if (stored_binder) {
			binder.SetActiveBinder(stored_binder);
		} else {
			binder.PopExpressionBinder();
		}
	}
}

BindResult ExpressionBinder::BindExpression(unique_ptr<ParsedExpression> *expr, idx_t depth, bool root_expression) {
	auto &expr_ref = **expr;
	switch (expr_ref.expression_class) {
	case ExpressionClass::BETWEEN:
		return BindExpression((BetweenExpression &)expr_ref, depth);
	case ExpressionClass::CASE:
		return BindExpression((CaseExpression &)expr_ref, depth);
	case ExpressionClass::CAST:
		return BindExpression((CastExpression &)expr_ref, depth);
	case ExpressionClass::COLLATE:
		return BindExpression((CollateExpression &)expr_ref, depth);
	case ExpressionClass::COLUMN_REF:
		return BindExpression((ColumnRefExpression &)expr_ref, depth);
	case ExpressionClass::COMPARISON:
		return BindExpression((ComparisonExpression &)expr_ref, depth);
	case ExpressionClass::CONJUNCTION:
		return BindExpression((ConjunctionExpression &)expr_ref, depth);
	case ExpressionClass::CONSTANT:
		return BindExpression((ConstantExpression &)expr_ref, depth);
	case ExpressionClass::FUNCTION:
		// binding function expression has extra parameter needed for macro's
		return BindExpression((FunctionExpression &)expr_ref, depth, expr);
	case ExpressionClass::LAMBDA:
		return BindExpression((LambdaExpression &)expr_ref, depth, false, LogicalTypeId::INVALID);
	case ExpressionClass::OPERATOR:
		return BindExpression((OperatorExpression &)expr_ref, depth);
	case ExpressionClass::SUBQUERY:
		return BindExpression((SubqueryExpression &)expr_ref, depth);
	case ExpressionClass::PARAMETER:
		return BindExpression((ParameterExpression &)expr_ref, depth);
	case ExpressionClass::POSITIONAL_REFERENCE:
		return BindExpression((PositionalReferenceExpression &)expr_ref, depth);
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
		auto &next_binder = active_binders.back();
		ExpressionBinder::QualifyColumnNames(next_binder->binder, expr);
		auto bind_result = next_binder->Bind(&expr, depth);
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
		string bind_error = Bind(&expr, depth);
		if (error.empty()) {
			error = bind_error;
		}
	}
}

void ExpressionBinder::ExtractCorrelatedExpressions(Binder &binder, Expression &expr) {
	if (expr.type == ExpressionType::BOUND_COLUMN_REF) {
		auto &bound_colref = (BoundColumnRefExpression &)expr;
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
	case LogicalTypeId::STRUCT:
	case LogicalTypeId::MAP: {
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
	case LogicalTypeId::STRUCT:
	case LogicalTypeId::MAP: {
		// we make a copy of the child types of the struct here
		auto child_types = StructType::GetChildTypes(type);
		for (auto &child_type : child_types) {
			child_type.second = ExchangeType(child_type.second, target, new_type);
		}
		return type.id() == LogicalTypeId::MAP ? LogicalType::MAP(move(child_types))
		                                       : LogicalType::STRUCT(move(child_types));
	}
	case LogicalTypeId::UNION: {
		auto member_types = UnionType::CopyMemberTypes(type);
		for (auto &member_type : member_types) {
			member_type.second = ExchangeType(member_type.second, target, new_type);
		}
		return LogicalType::UNION(move(member_types));
	}
	case LogicalTypeId::LIST:
		return LogicalType::LIST(ExchangeType(ListType::GetChildType(type), target, new_type));
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

unique_ptr<Expression> ExpressionBinder::Bind(unique_ptr<ParsedExpression> &expr, LogicalType *result_type,
                                              bool root_expression) {
	// bind the main expression
	auto error_msg = Bind(&expr, 0, root_expression);
	if (!error_msg.empty()) {
		// failed to bind: try to bind correlated columns in the expression (if any)
		bool success = BindCorrelatedColumns(expr);
		if (!success) {
			throw BinderException(error_msg);
		}
		auto bound_expr = (BoundExpression *)expr.get();
		ExtractCorrelatedExpressions(binder, *bound_expr->expr);
	}
	D_ASSERT(expr->expression_class == ExpressionClass::BOUND_EXPRESSION);
	auto bound_expr = (BoundExpression *)expr.get();
	unique_ptr<Expression> result = move(bound_expr->expr);
	if (target_type.id() != LogicalTypeId::INVALID) {
		// the binder has a specific target type: add a cast to that type
		result = BoundCastExpression::AddCastToType(context, move(result), target_type);
	} else {
		if (!binder.can_contain_nulls) {
			// SQL NULL type is only used internally in the binder
			// cast to INTEGER if we encounter it outside of the binder
			if (ContainsNullType(result->return_type)) {
				auto result_type = ExchangeNullType(result->return_type);
				result = BoundCastExpression::AddCastToType(context, move(result), result_type);
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

string ExpressionBinder::Bind(unique_ptr<ParsedExpression> *expr, idx_t depth, bool root_expression) {
	// bind the node, but only if it has not been bound yet
	auto &expression = **expr;
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
	*expr = make_unique<BoundExpression>(move(result.expression));
	auto be = (BoundExpression *)expr->get();
	D_ASSERT(be);
	be->alias = alias;
	if (!alias.empty()) {
		be->expr->alias = alias;
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
		auto &aggr_expr = (BoundAggregateExpression &)expr;
		for (auto &child : aggr_expr.children) {
			callback(child);
		}
		if (aggr_expr.filter) {
			callback(aggr_expr.filter);
		}
		break;
	}
	case ExpressionClass::BOUND_BETWEEN: {
		auto &between_expr = (BoundBetweenExpression &)expr;
		callback(between_expr.input);
		callback(between_expr.lower);
		callback(between_expr.upper);
		break;
	}
	case ExpressionClass::BOUND_CASE: {
		auto &case_expr = (BoundCaseExpression &)expr;
		for (auto &case_check : case_expr.case_checks) {
			callback(case_check.when_expr);
			callback(case_check.then_expr);
		}
		callback(case_expr.else_expr);
		break;
	}
	case ExpressionClass::BOUND_CAST: {
		auto &cast_expr = (BoundCastExpression &)expr;
		callback(cast_expr.child);
		break;
	}
	case ExpressionClass::BOUND_COMPARISON: {
		auto &comp_expr = (BoundComparisonExpression &)expr;
		callback(comp_expr.left);
		callback(comp_expr.right);
		break;
	}
	case ExpressionClass::BOUND_CONJUNCTION: {
		auto &conj_expr = (BoundConjunctionExpression &)expr;
		for (auto &child : conj_expr.children) {
			callback(child);
		}
		break;
	}
	case ExpressionClass::BOUND_FUNCTION: {
		auto &func_expr = (BoundFunctionExpression &)expr;
		for (auto &child : func_expr.children) {
			callback(child);
		}
		break;
	}
	case ExpressionClass::BOUND_OPERATOR: {
		auto &op_expr = (BoundOperatorExpression &)expr;
		for (auto &child : op_expr.children) {
			callback(child);
		}
		break;
	}
	case ExpressionClass::BOUND_SUBQUERY: {
		auto &subquery_expr = (BoundSubqueryExpression &)expr;
		if (subquery_expr.child) {
			callback(subquery_expr.child);
		}
		break;
	}
	case ExpressionClass::BOUND_WINDOW: {
		auto &window_expr = (BoundWindowExpression &)expr;
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
		auto &unnest_expr = (BoundUnnestExpression &)expr;
		callback(unnest_expr.child);
		break;
	}
	case ExpressionClass::BOUND_COLUMN_REF:
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
	case TableReferenceType::CROSS_PRODUCT: {
		auto &bound_crossproduct = (BoundCrossProductRef &)ref;
		EnumerateTableRefChildren(*bound_crossproduct.left, callback);
		EnumerateTableRefChildren(*bound_crossproduct.right, callback);
		break;
	}
	case TableReferenceType::EXPRESSION_LIST: {
		auto &bound_expr_list = (BoundExpressionListRef &)ref;
		for (auto &expr_list : bound_expr_list.values) {
			for (auto &expr : expr_list) {
				EnumerateExpression(expr, callback);
			}
		}
		break;
	}
	case TableReferenceType::JOIN: {
		auto &bound_join = (BoundJoinRef &)ref;
		EnumerateExpression(bound_join.condition, callback);
		EnumerateTableRefChildren(*bound_join.left, callback);
		EnumerateTableRefChildren(*bound_join.right, callback);
		break;
	}
	case TableReferenceType::SUBQUERY: {
		auto &bound_subquery = (BoundSubqueryRef &)ref;
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
		auto &bound_setop = (BoundSetOperationNode &)node;
		EnumerateQueryNodeChildren(*bound_setop.left, callback);
		EnumerateQueryNodeChildren(*bound_setop.right, callback);
		break;
	}
	case QueryNodeType::RECURSIVE_CTE_NODE: {
		auto &cte_node = (BoundRecursiveCTENode &)node;
		EnumerateQueryNodeChildren(*cte_node.left, callback);
		EnumerateQueryNodeChildren(*cte_node.right, callback);
		break;
	}
	case QueryNodeType::SELECT_NODE: {
		auto &bound_select = (BoundSelectNode &)node;
		for (idx_t i = 0; i < bound_select.select_list.size(); i++) {
			EnumerateExpression(bound_select.select_list[i], callback);
		}
		EnumerateExpression(bound_select.where_clause, callback);
		for (idx_t i = 0; i < bound_select.groups.group_expressions.size(); i++) {
			EnumerateExpression(bound_select.groups.group_expressions[i], callback);
		}
		EnumerateExpression(bound_select.having, callback);
		for (idx_t i = 0; i < bound_select.aggregates.size(); i++) {
			EnumerateExpression(bound_select.aggregates[i], callback);
		}
		for (idx_t i = 0; i < bound_select.unnests.size(); i++) {
			EnumerateExpression(bound_select.unnests[i], callback);
		}
		for (idx_t i = 0; i < bound_select.windows.size(); i++) {
			EnumerateExpression(bound_select.windows[i], callback);
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
	auto res = make_unique<ConjunctionOrFilter>();
	res->child_filters = source.ReadRequiredSerializableList<TableFilter>();
	return move(res);
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
	auto res = make_unique<ConjunctionAndFilter>();
	res->child_filters = source.ReadRequiredSerializableList<TableFilter>();
	return move(res);
}

} // namespace duckdb






namespace duckdb {

ConstantFilter::ConstantFilter(ExpressionType comparison_type_p, Value constant_p)
    : TableFilter(TableFilterType::CONSTANT_COMPARISON), comparison_type(comparison_type_p),
      constant(move(constant_p)) {
}

FilterPropagateResult ConstantFilter::CheckStatistics(BaseStatistics &stats) {
	D_ASSERT(constant.type().id() == stats.type.id());
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
		return ((NumericStatistics &)stats).CheckZonemap(comparison_type, constant);
	case PhysicalType::VARCHAR:
		return ((StringStatistics &)stats).CheckZonemap(comparison_type, StringValue::Get(constant));
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
	return make_unique<ConstantFilter>(comparision_type, constant);
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
	return make_unique<IsNotNullFilter>();
}

void IsNullFilter::Serialize(FieldWriter &writer) const {
}

unique_ptr<TableFilter> IsNullFilter::Deserialize(FieldReader &source) {
	return make_unique<IsNullFilter>();
}

} // namespace duckdb









namespace duckdb {

unique_ptr<Expression> JoinCondition::CreateExpression(JoinCondition cond) {
	auto bound_comparison = make_unique<BoundComparisonExpression>(cond.comparison, move(cond.left), move(cond.right));
	return move(bound_comparison);
}

unique_ptr<Expression> JoinCondition::CreateExpression(vector<JoinCondition> conditions) {
	unique_ptr<Expression> result;
	for (auto &cond : conditions) {
		auto expr = CreateExpression(move(cond));
		if (!result) {
			result = move(expr);
		} else {
			auto conj =
			    make_unique<BoundConjunctionExpression>(ExpressionType::CONJUNCTION_AND, move(expr), move(result));
			result = move(conj);
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
	result.left = move(left);
	result.right = move(right);
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

JoinSide JoinSide::GetJoinSide(idx_t table_binding, unordered_set<idx_t> &left_bindings,
                               unordered_set<idx_t> &right_bindings) {
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

JoinSide JoinSide::GetJoinSide(Expression &expression, unordered_set<idx_t> &left_bindings,
                               unordered_set<idx_t> &right_bindings) {
	if (expression.type == ExpressionType::BOUND_COLUMN_REF) {
		auto &colref = (BoundColumnRefExpression &)expression;
		if (colref.depth > 0) {
			throw Exception("Non-inner join on correlated columns not supported");
		}
		return GetJoinSide(colref.binding.table_index, left_bindings, right_bindings);
	}
	D_ASSERT(expression.type != ExpressionType::BOUND_REF);
	if (expression.type == ExpressionType::SUBQUERY) {
		D_ASSERT(expression.GetExpressionClass() == ExpressionClass::BOUND_SUBQUERY);
		auto &subquery = (BoundSubqueryExpression &)expression;
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

JoinSide JoinSide::GetJoinSide(const unordered_set<idx_t> &bindings, unordered_set<idx_t> &left_bindings,
                               unordered_set<idx_t> &right_bindings) {
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
    : type(type), expressions(move(expressions)), estimated_cardinality(0), has_estimated_cardinality(false) {
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
		try {
			expressions[expr_idx]->Serialize(serializer);
		} catch (NotImplementedException &ex) {
			// ignore for now (FIXME)
			return;
		}

		auto data = serializer.GetData();
		auto deserializer = BufferedDeserializer(data.data.get(), data.size);

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
	children.push_back(move(child));
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
	return max_cardinality;
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
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
		result = LogicalComparisonJoin::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_ANY_JOIN:
		result = LogicalAnyJoin::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_CROSS_PRODUCT:
		result = LogicalCrossProduct::Deserialize(state, reader);
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
	case LogicalOperatorType::LOGICAL_ALTER:
		result = LogicalSimple::Deserialize(state, reader);
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
	case LogicalOperatorType::LOGICAL_DROP:
		result = LogicalSimple::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_PRAGMA:
		result = LogicalPragma::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_TRANSACTION:
		result = LogicalSimple::Deserialize(state, reader);
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
	case LogicalOperatorType::LOGICAL_VACUUM:
		result = LogicalSimple::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_SET:
		result = LogicalSet::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_LOAD:
		result = LogicalSimple::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR:
		result = LogicalExtensionOperator::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_INVALID:
		/* no default here to trigger a warning if we forget to implement deserialize for a new operator */
		throw SerializationException("Invalid type for operator deserialization");
	}

	reader.Finalize();
	result->children = move(children);

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
	auto logical_op_deserializer = BufferedDeserializer(data.data.get(), data.size);
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
		auto &get = (LogicalExpressionGet &)op;
		for (auto &expr_list : get.expressions) {
			for (auto &expr : expr_list) {
				callback(&expr);
			}
		}
		break;
	}
	case LogicalOperatorType::LOGICAL_ORDER_BY: {
		auto &order = (LogicalOrder &)op;
		for (auto &node : order.orders) {
			callback(&node.expression);
		}
		break;
	}
	case LogicalOperatorType::LOGICAL_TOP_N: {
		auto &order = (LogicalTopN &)op;
		for (auto &node : order.orders) {
			callback(&node.expression);
		}
		break;
	}
	case LogicalOperatorType::LOGICAL_DISTINCT: {
		auto &distinct = (LogicalDistinct &)op;
		for (auto &target : distinct.distinct_targets) {
			callback(&target);
		}
		break;
	}
	case LogicalOperatorType::LOGICAL_DELIM_JOIN:
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN: {
		if (op.type == LogicalOperatorType::LOGICAL_DELIM_JOIN) {
			auto &delim_join = (LogicalDelimJoin &)op;
			for (auto &expr : delim_join.duplicate_eliminated_columns) {
				callback(&expr);
			}
		}
		auto &join = (LogicalComparisonJoin &)op;
		for (auto &cond : join.conditions) {
			callback(&cond.left);
			callback(&cond.right);
		}
		break;
	}
	case LogicalOperatorType::LOGICAL_ANY_JOIN: {
		auto &join = (LogicalAnyJoin &)op;
		callback(&join.condition);
		break;
	}
	case LogicalOperatorType::LOGICAL_LIMIT: {
		auto &limit = (LogicalLimit &)op;
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
		auto &aggr = (LogicalAggregate &)op;
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
		result = VisitReplace((BoundAggregateExpression &)expr, expression);
		break;
	case ExpressionClass::BOUND_BETWEEN:
		result = VisitReplace((BoundBetweenExpression &)expr, expression);
		break;
	case ExpressionClass::BOUND_CASE:
		result = VisitReplace((BoundCaseExpression &)expr, expression);
		break;
	case ExpressionClass::BOUND_CAST:
		result = VisitReplace((BoundCastExpression &)expr, expression);
		break;
	case ExpressionClass::BOUND_COLUMN_REF:
		result = VisitReplace((BoundColumnRefExpression &)expr, expression);
		break;
	case ExpressionClass::BOUND_COMPARISON:
		result = VisitReplace((BoundComparisonExpression &)expr, expression);
		break;
	case ExpressionClass::BOUND_CONJUNCTION:
		result = VisitReplace((BoundConjunctionExpression &)expr, expression);
		break;
	case ExpressionClass::BOUND_CONSTANT:
		result = VisitReplace((BoundConstantExpression &)expr, expression);
		break;
	case ExpressionClass::BOUND_FUNCTION:
		result = VisitReplace((BoundFunctionExpression &)expr, expression);
		break;
	case ExpressionClass::BOUND_SUBQUERY:
		result = VisitReplace((BoundSubqueryExpression &)expr, expression);
		break;
	case ExpressionClass::BOUND_OPERATOR:
		result = VisitReplace((BoundOperatorExpression &)expr, expression);
		break;
	case ExpressionClass::BOUND_PARAMETER:
		result = VisitReplace((BoundParameterExpression &)expr, expression);
		break;
	case ExpressionClass::BOUND_REF:
		result = VisitReplace((BoundReferenceExpression &)expr, expression);
		break;
	case ExpressionClass::BOUND_DEFAULT:
		result = VisitReplace((BoundDefaultExpression &)expr, expression);
		break;
	case ExpressionClass::BOUND_WINDOW:
		result = VisitReplace((BoundWindowExpression &)expr, expression);
		break;
	case ExpressionClass::BOUND_UNNEST:
		result = VisitReplace((BoundUnnestExpression &)expr, expression);
		break;
	default:
		throw InternalException("Unrecognized expression type in logical operator visitor");
	}
	if (result) {
		*expression = move(result);
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
    : LogicalOperator(LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY, move(select_list)), group_index(group_index),
      aggregate_index(aggregate_index), groupings_index(DConstants::INVALID_INDEX) {
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
	vector<vector<idx_t>> grouping_functions;
	auto grouping_functions_size = reader.ReadRequired<idx_t>();
	for (idx_t i = 0; i < grouping_functions_size; i++) {
		grouping_functions.push_back(reader.ReadRequiredList<idx_t>());
	}
	auto result = make_unique<LogicalAggregate>(group_index, aggregate_index, move(expressions));
	result->groupings_index = groupings_index;
	result->groups = move(groups);
	result->grouping_functions = move(grouping_functions);
	result->grouping_sets = move(grouping_sets);

	return move(result);
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
	auto result = make_unique<LogicalAnyJoin>(join_type);
	result->condition = move(condition);
	return move(result);
}

} // namespace duckdb




namespace duckdb {

LogicalColumnDataGet::LogicalColumnDataGet(idx_t table_index, vector<LogicalType> types,
                                           unique_ptr<ColumnDataCollection> collection)
    : LogicalOperator(LogicalOperatorType::LOGICAL_CHUNK_GET), table_index(table_index), collection(move(collection)) {
	D_ASSERT(types.size() > 0);
	chunk_types = move(types);
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
	auto collection = make_unique<ColumnDataCollection>(state.gstate.context, chunk_types);
	for (idx_t i = 0; i < chunk_count; i++) {
		DataChunk chunk;
		chunk.Deserialize(reader.GetSource());
		collection->Append(chunk);
	}
	return make_unique<LogicalColumnDataGet>(table_index, move(chunk_types), move(collection));
}

vector<idx_t> LogicalColumnDataGet::GetTableIndex() const {
	return vector<idx_t> {table_index};
}

} // namespace duckdb





namespace duckdb {

LogicalComparisonJoin::LogicalComparisonJoin(JoinType join_type, LogicalOperatorType logical_type)
    : LogicalJoin(join_type, logical_type) {
}

string LogicalComparisonJoin::ParamsToString() const {
	string result = JoinTypeToString(join_type);
	for (auto &condition : conditions) {
		result += "\n";
		auto expr = make_unique<BoundComparisonExpression>(condition.comparison, condition.left->Copy(),
		                                                   condition.right->Copy());
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
	auto result = make_unique<LogicalComparisonJoin>(JoinType::INVALID, state.type);
	LogicalComparisonJoin::Deserialize(*result, state, reader);
	return move(result);
}

} // namespace duckdb






namespace duckdb {

void LogicalCopyToFile::Serialize(FieldWriter &writer) const {
	writer.WriteString(file_path);
	writer.WriteField(use_tmp_file);
	writer.WriteField(is_file_and_exists);

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
	auto is_file_and_exists = reader.ReadRequired<bool>();

	auto copy_func_name = reader.ReadRequired<string>();

	auto has_bind_data = reader.ReadRequired<bool>();

	auto &context = state.gstate.context;
	auto &catalog = Catalog::GetCatalog(context);
	auto func_catalog = catalog.GetEntry(context, CatalogType::COPY_FUNCTION_ENTRY, DEFAULT_SCHEMA, copy_func_name);
	if (!func_catalog || func_catalog->type != CatalogType::COPY_FUNCTION_ENTRY) {
		throw InternalException("Cant find catalog entry for function %s", copy_func_name);
	}
	auto copy_func_catalog_entry = (CopyFunctionCatalogEntry *)func_catalog;
	CopyFunction copy_func = copy_func_catalog_entry->function;

	unique_ptr<FunctionData> bind_data;
	if (has_bind_data) {
		if (!copy_func.deserialize) {
			throw SerializationException("Have bind info but no deserialization function for %s", copy_func.name);
		}
		bind_data = copy_func.deserialize(context, reader, copy_func);
	}

	auto result = make_unique<LogicalCopyToFile>(copy_func, move(bind_data));
	result->file_path = file_path;
	result->use_tmp_file = use_tmp_file;
	result->is_file_and_exists = is_file_and_exists;
	return move(result);
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

	auto &catalog = Catalog::GetCatalog(context);
	// TODO(stephwang): review if below is necessary or just not pass schema_catalog_entry
	SchemaCatalogEntry *schema_catalog_entry = catalog.GetSchema(context, info->schema, true);

	return make_unique<LogicalCreate>(state.type, move(info), schema_catalog_entry);
}

idx_t LogicalCreate::EstimateCardinality(ClientContext &context) {
	return 1;
}

} // namespace duckdb






namespace duckdb {

void LogicalCreateIndex::Serialize(FieldWriter &writer) const {

	writer.WriteOptional(info);
	table.Serialize(writer.GetSerializer());
	FunctionSerializer::SerializeBase<TableFunction>(writer, function, bind_data.get());
	writer.WriteSerializableList(unbound_expressions);

	writer.Finalize();
}

unique_ptr<LogicalOperator> LogicalCreateIndex::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {

	auto &context = state.gstate.context;
	auto catalog_info = TableCatalogEntry::Deserialize(reader.GetSource(), context);
	auto &catalog = Catalog::GetCatalog(context);
	TableCatalogEntry *table = catalog.GetEntry<TableCatalogEntry>(context, catalog_info->schema, catalog_info->table);

	auto unbound_expressions = reader.ReadRequiredSerializableList<Expression>(state.gstate);

	auto create_info = reader.ReadOptional<CreateInfo>(nullptr);
	if (create_info->type != CatalogType::INDEX_ENTRY) {
		throw InternalException("Unexpected type: '%s', expected '%s'", CatalogTypeToString(create_info->type),
		                        CatalogTypeToString(CatalogType::INDEX_ENTRY));
	}

	CreateInfo *raw_create_info_ptr = create_info.release();
	CreateIndexInfo *raw_create_index_info_ptr = static_cast<CreateIndexInfo *>(raw_create_info_ptr);
	unique_ptr<CreateIndexInfo> uptr_create_index_info = unique_ptr<CreateIndexInfo> {raw_create_index_info_ptr};
	auto info = unique_ptr<CreateIndexInfo> {static_cast<CreateIndexInfo *>(create_info.release())};

	unique_ptr<FunctionData> bind_data;
	bool has_deserialize;
	auto function = FunctionSerializer::DeserializeBaseInternal<TableFunction, TableFunctionCatalogEntry>(
	    reader, state.gstate, CatalogType::TABLE_FUNCTION_ENTRY, bind_data, has_deserialize);

	reader.Finalize();
	return make_unique<LogicalCreateIndex>(move(bind_data), move(info), move(unbound_expressions), *table,
	                                       move(function));
}

} // namespace duckdb


namespace duckdb {

void LogicalCreateTable::Serialize(FieldWriter &writer) const {
	writer.WriteSerializable(*info);
}

unique_ptr<LogicalOperator> LogicalCreateTable::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto info = reader.ReadRequiredSerializable<BoundCreateTableInfo>(state.gstate);
	auto schema = info->schema;
	return make_unique<LogicalCreateTable>(schema, move(info));
}

idx_t LogicalCreateTable::EstimateCardinality(ClientContext &context) {
	return 1;
}

} // namespace duckdb


namespace duckdb {

LogicalCrossProduct::LogicalCrossProduct(unique_ptr<LogicalOperator> left, unique_ptr<LogicalOperator> right)
    : LogicalOperator(LogicalOperatorType::LOGICAL_CROSS_PRODUCT) {
	D_ASSERT(left);
	D_ASSERT(right);
	children.push_back(move(left));
	children.push_back(move(right));
}

vector<ColumnBinding> LogicalCrossProduct::GetColumnBindings() {
	auto left_bindings = children[0]->GetColumnBindings();
	auto right_bindings = children[1]->GetColumnBindings();
	left_bindings.insert(left_bindings.end(), right_bindings.begin(), right_bindings.end());
	return left_bindings;
}

void LogicalCrossProduct::ResolveTypes() {
	types.insert(types.end(), children[0]->types.begin(), children[0]->types.end());
	types.insert(types.end(), children[1]->types.begin(), children[1]->types.end());
}

unique_ptr<LogicalOperator> LogicalCrossProduct::Create(unique_ptr<LogicalOperator> left,
                                                        unique_ptr<LogicalOperator> right) {
	if (left->type == LogicalOperatorType::LOGICAL_DUMMY_SCAN) {
		return right;
	}
	if (right->type == LogicalOperatorType::LOGICAL_DUMMY_SCAN) {
		return left;
	}
	return make_unique<LogicalCrossProduct>(move(left), move(right));
}

void LogicalCrossProduct::Serialize(FieldWriter &writer) const {
}

unique_ptr<LogicalOperator> LogicalCrossProduct::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	// TODO(stephwang): review if unique_ptr<LogicalOperator> plan is needed
	auto result = unique_ptr<LogicalCrossProduct>(new LogicalCrossProduct());
	return move(result);
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
	return make_unique<LogicalCTERef>(table_index, cte_index, chunk_types, bound_columns);
}

vector<idx_t> LogicalCTERef::GetTableIndex() const {
	return vector<idx_t> {table_index};
}

} // namespace duckdb



namespace duckdb {

void LogicalDelete::Serialize(FieldWriter &writer) const {
	table->Serialize(writer.GetSerializer());
	writer.WriteField(table_index);
	writer.WriteField(return_chunk);
}

unique_ptr<LogicalOperator> LogicalDelete::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto &context = state.gstate.context;
	auto info = TableCatalogEntry::Deserialize(reader.GetSource(), context);

	auto &catalog = Catalog::GetCatalog(context);

	TableCatalogEntry *table_catalog_entry = catalog.GetEntry<TableCatalogEntry>(context, info->schema, info->table);

	auto table_index = reader.ReadRequired<idx_t>();
	auto result = make_unique<LogicalDelete>(table_catalog_entry, table_index);
	result->return_chunk = reader.ReadRequired<bool>();
	return move(result);
}

idx_t LogicalDelete::EstimateCardinality(ClientContext &context) {
	return return_chunk ? LogicalOperator::EstimateCardinality(context) : 1;
}

vector<idx_t> LogicalDelete::GetTableIndex() const {
	return vector<idx_t> {table_index};
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
	return make_unique<LogicalDelimGet>(table_index, chunk_types);
}

vector<idx_t> LogicalDelimGet::GetTableIndex() const {
	return vector<idx_t> {table_index};
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
	auto result = make_unique<LogicalDelimJoin>(JoinType::INVALID);
	LogicalComparisonJoin::Deserialize(*result, state, reader);
	result->duplicate_eliminated_columns = reader.ReadRequiredSerializableList<Expression>(state.gstate);
	return move(result);
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
	writer.WriteSerializableList(distinct_targets);
}

unique_ptr<LogicalOperator> LogicalDistinct::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto distinct_targets = reader.ReadRequiredSerializableList<Expression>(state.gstate);
	return make_unique<LogicalDistinct>(move(distinct_targets));
}

} // namespace duckdb



namespace duckdb {

void LogicalDummyScan::Serialize(FieldWriter &writer) const {
	writer.WriteField(table_index);
}

unique_ptr<LogicalOperator> LogicalDummyScan::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto table_index = reader.ReadRequired<idx_t>();
	return make_unique<LogicalDummyScan>(table_index);
}

vector<idx_t> LogicalDummyScan::GetTableIndex() const {
	return vector<idx_t> {table_index};
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
	return move(result);
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
	return move(result);
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

	return make_unique<LogicalExpressionGet>(table_index, expr_types, move(expressions));
}

vector<idx_t> LogicalExpressionGet::GetTableIndex() const {
	return vector<idx_t> {table_index};
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
	expressions.push_back(move(expression));
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
			auto &conjunction = (BoundConjunctionExpression &)*expressions[i];
			found_conjunction = true;
			// AND expression, append the other children
			for (idx_t k = 1; k < conjunction.children.size(); k++) {
				expressions.push_back(move(conjunction.children[k]));
			}
			// replace this expression with the first child of the conjunction
			expressions[i] = move(conjunction.children[0]);
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
	auto result = make_unique<LogicalFilter>();
	result->expressions = move(expressions);
	result->projection_map = move(projection_map);
	return move(result);
}

} // namespace duckdb










namespace duckdb {

LogicalGet::LogicalGet(idx_t table_index, TableFunction function, unique_ptr<FunctionData> bind_data,
                       vector<LogicalType> returned_types, vector<string> returned_names)
    : LogicalOperator(LogicalOperatorType::LOGICAL_GET), table_index(table_index), function(move(function)),
      bind_data(move(bind_data)), returned_types(move(returned_types)), names(move(returned_names)) {
}

string LogicalGet::GetName() const {
	return StringUtil::Upper(function.name);
}

TableCatalogEntry *LogicalGet::GetTable() const {
	return TableScanFunction::GetTableEntry(function, bind_data.get());
}

string LogicalGet::ParamsToString() const {
	string result;
	for (auto &kv : table_filters.filters) {
		auto &column_index = kv.first;
		auto &filter = kv.second;
		if (column_index < names.size()) {
			result += filter->ToString(names[column_index]);
		}
		result += "\n";
	}
	if (!function.to_string) {
		return string();
	}
	return function.to_string(bind_data.get());
}

vector<ColumnBinding> LogicalGet::GetColumnBindings() {
	if (column_ids.empty()) {
		return {ColumnBinding(table_index, 0)};
	}
	vector<ColumnBinding> result;
	if (projection_ids.empty()) {
		for (idx_t col_idx = 0; col_idx < column_ids.size(); col_idx++) {
			result.emplace_back(table_index, col_idx);
		}
	} else {
		for (auto proj_id : projection_ids) {
			result.emplace_back(table_index, proj_id);
		}
	}
	return result;
}

void LogicalGet::ResolveTypes() {
	if (column_ids.empty()) {
		column_ids.push_back(COLUMN_IDENTIFIER_ROW_ID);
	}

	if (projection_ids.empty()) {
		for (auto &index : column_ids) {
			if (index == COLUMN_IDENTIFIER_ROW_ID) {
				types.emplace_back(LogicalType::ROW_TYPE);
			} else {
				types.push_back(returned_types[index]);
			}
		}
	} else {
		for (auto &proj_index : projection_ids) {
			auto &index = column_ids[proj_index];
			if (index == COLUMN_IDENTIFIER_ROW_ID) {
				types.emplace_back(LogicalType::ROW_TYPE);
			} else {
				types.push_back(returned_types[index]);
			}
		}
	}
}

idx_t LogicalGet::EstimateCardinality(ClientContext &context) {
	if (function.cardinality) {
		auto node_stats = function.cardinality(context, bind_data.get());
		if (node_stats && node_stats->has_estimated_cardinality) {
			return node_stats->estimated_cardinality;
		}
	}
	return 1;
}

void LogicalGet::Serialize(FieldWriter &writer) const {
	writer.WriteField(table_index);
	writer.WriteRegularSerializableList(returned_types);
	writer.WriteList<string>(names);
	writer.WriteList<column_t>(column_ids);
	writer.WriteList<column_t>(projection_ids);
	writer.WriteSerializable(table_filters);

	FunctionSerializer::SerializeBase<TableFunction>(writer, function, bind_data.get());
	if (!function.serialize) {
		D_ASSERT(!function.deserialize);
		// no serialize method: serialize input values and named_parameters for rebinding purposes
		writer.WriteRegularSerializableList(parameters);
		writer.WriteField<idx_t>(named_parameters.size());
		for (auto &pair : named_parameters) {
			writer.WriteString(pair.first);
			writer.WriteSerializable(pair.second);
		}
		writer.WriteRegularSerializableList(input_table_types);
		writer.WriteList<string>(input_table_names);
	}
}

unique_ptr<LogicalOperator> LogicalGet::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto table_index = reader.ReadRequired<idx_t>();
	auto returned_types = reader.ReadRequiredSerializableList<LogicalType, LogicalType>();
	auto returned_names = reader.ReadRequiredList<string>();
	auto column_ids = reader.ReadRequiredList<column_t>();
	auto projection_ids = reader.ReadRequiredList<column_t>();
	auto table_filters = reader.ReadRequiredSerializable<TableFilterSet>();

	unique_ptr<FunctionData> bind_data;
	bool has_deserialize;
	auto function = FunctionSerializer::DeserializeBaseInternal<TableFunction, TableFunctionCatalogEntry>(
	    reader, state.gstate, CatalogType::TABLE_FUNCTION_ENTRY, bind_data, has_deserialize);

	vector<Value> parameters;
	named_parameter_map_t named_parameters;
	vector<LogicalType> input_table_types;
	vector<string> input_table_names;
	if (!has_deserialize) {
		D_ASSERT(!bind_data);
		parameters = reader.ReadRequiredSerializableList<Value, Value>();

		auto named_parameters_size = reader.ReadRequired<idx_t>();
		for (idx_t i = 0; i < named_parameters_size; i++) {
			auto first = reader.ReadRequired<string>();
			auto second = reader.ReadRequiredSerializable<Value, Value>();
			auto pair = make_pair(first, second);
			named_parameters.insert(pair);
		}

		input_table_types = reader.ReadRequiredSerializableList<LogicalType, LogicalType>();
		input_table_names = reader.ReadRequiredList<string>();
		TableFunctionBindInput input(parameters, named_parameters, input_table_types, input_table_names,
		                             function.function_info.get());

		vector<LogicalType> bind_return_types;
		vector<string> bind_names;
		bind_data = function.bind(state.gstate.context, input, bind_return_types, bind_names);
		if (returned_types != bind_return_types) {
			throw SerializationException(
			    "Table function deserialization failure - bind returned different return types than were serialized");
		}
		// names can actually be different because of aliases - only the sizes cannot be different
		if (returned_names.size() != bind_names.size()) {
			throw SerializationException(
			    "Table function deserialization failure - bind returned different returned names than were serialized");
		}
	}

	auto result = make_unique<LogicalGet>(table_index, function, move(bind_data), returned_types, returned_names);
	result->column_ids = move(column_ids);
	result->projection_ids = move(projection_ids);
	result->table_filters = move(*table_filters);
	result->parameters = move(parameters);
	result->named_parameters = move(named_parameters);
	result->input_table_types = input_table_types;
	result->input_table_names = input_table_names;
	return move(result);
}

vector<idx_t> LogicalGet::GetTableIndex() const {
	return vector<idx_t> {table_index};
}

} // namespace duckdb





namespace duckdb {

void LogicalInsert::Serialize(FieldWriter &writer) const {
	writer.WriteField<idx_t>(insert_values.size());
	for (auto &entry : insert_values) {
		writer.WriteSerializableList(entry);
	}

	writer.WriteList<idx_t>(column_index_map);
	writer.WriteRegularSerializableList(expected_types);
	table->Serialize(writer.GetSerializer());
	writer.WriteField(table_index);
	writer.WriteField(return_chunk);
	writer.WriteSerializableList(bound_defaults);
}

unique_ptr<LogicalOperator> LogicalInsert::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto &context = state.gstate.context;
	auto insert_values_size = reader.ReadRequired<idx_t>();
	vector<vector<unique_ptr<Expression>>> insert_values;
	for (idx_t i = 0; i < insert_values_size; ++i) {
		insert_values.push_back(reader.ReadRequiredSerializableList<Expression>(state.gstate));
	}

	auto column_index_map = reader.ReadRequiredList<idx_t, physical_index_vector_t<idx_t>>();
	auto expected_types = reader.ReadRequiredSerializableList<LogicalType, LogicalType>();
	auto info = TableCatalogEntry::Deserialize(reader.GetSource(), context);
	auto table_index = reader.ReadRequired<idx_t>();
	auto return_chunk = reader.ReadRequired<bool>();
	auto bound_defaults = reader.ReadRequiredSerializableList<Expression>(state.gstate);

	auto &catalog = Catalog::GetCatalog(context);

	TableCatalogEntry *table_catalog_entry = catalog.GetEntry<TableCatalogEntry>(context, info->schema, info->table);

	if (!table_catalog_entry) {
		throw InternalException("Cant find catalog entry for table %s", info->table);
	}

	auto result = make_unique<LogicalInsert>(table_catalog_entry, table_index);
	result->type = state.type;
	result->table = table_catalog_entry;
	result->return_chunk = return_chunk;
	result->insert_values = move(insert_values);
	result->column_index_map = column_index_map;
	result->expected_types = expected_types;
	result->bound_defaults = move(bound_defaults);
	return move(result);
}

idx_t LogicalInsert::EstimateCardinality(ClientContext &context) {
	return return_chunk ? LogicalOperator::EstimateCardinality(context) : 1;
}

vector<idx_t> LogicalInsert::GetTableIndex() const {
	return vector<idx_t> {table_index};
}

} // namespace duckdb






namespace duckdb {

LogicalJoin::LogicalJoin(JoinType join_type, LogicalOperatorType logical_type)
    : LogicalOperator(logical_type), join_type(join_type) {
}

vector<ColumnBinding> LogicalJoin::GetColumnBindings() {
	auto left_bindings = MapBindings(children[0]->GetColumnBindings(), left_projection_map);
	if (join_type == JoinType::SEMI || join_type == JoinType::ANTI) {
		// for SEMI and ANTI join we only project the left hand side
		return left_bindings;
	}
	if (join_type == JoinType::MARK) {
		// for MARK join we project the left hand side plus the MARK column
		left_bindings.emplace_back(mark_index, 0);
		return left_bindings;
	}
	// for other join types we project both the LHS and the RHS
	auto right_bindings = MapBindings(children[1]->GetColumnBindings(), right_projection_map);
	left_bindings.insert(left_bindings.end(), right_bindings.begin(), right_bindings.end());
	return left_bindings;
}

void LogicalJoin::ResolveTypes() {
	types = MapTypes(children[0]->types, left_projection_map);
	if (join_type == JoinType::SEMI || join_type == JoinType::ANTI) {
		// for SEMI and ANTI join we only project the left hand side
		return;
	}
	if (join_type == JoinType::MARK) {
		// for MARK join we project the left hand side, plus a BOOLEAN column indicating the MARK
		types.emplace_back(LogicalType::BOOLEAN);
		return;
	}
	// for any other join we project both sides
	auto right_types = MapTypes(children[1]->types, right_projection_map);
	types.insert(types.end(), right_types.begin(), right_types.end());
}

void LogicalJoin::GetTableReferences(LogicalOperator &op, unordered_set<idx_t> &bindings) {
	auto column_bindings = op.GetColumnBindings();
	for (auto binding : column_bindings) {
		bindings.insert(binding.table_index);
	}
}

void LogicalJoin::GetExpressionBindings(Expression &expr, unordered_set<idx_t> &bindings) {
	if (expr.type == ExpressionType::BOUND_COLUMN_REF) {
		auto &colref = (BoundColumnRefExpression &)expr;
		D_ASSERT(colref.depth == 0);
		bindings.insert(colref.binding.table_index);
	}
	ExpressionIterator::EnumerateChildren(expr, [&](Expression &child) { GetExpressionBindings(child, bindings); });
}

void LogicalJoin::Serialize(FieldWriter &writer) const {
	writer.WriteField<JoinType>(join_type);
	writer.WriteField<idx_t>(mark_index);
	writer.WriteList<idx_t>(left_projection_map);
	writer.WriteList<idx_t>(right_projection_map);
	//	writer.WriteSerializableList(join_stats);
}

void LogicalJoin::Deserialize(LogicalJoin &join, LogicalDeserializationState &state, FieldReader &reader) {
	join.join_type = reader.ReadRequired<JoinType>();
	join.mark_index = reader.ReadRequired<idx_t>();
	join.left_projection_map = reader.ReadRequiredList<idx_t>();
	join.right_projection_map = reader.ReadRequiredList<idx_t>();
	//	join.join_stats = reader.ReadRequiredSerializableList<BaseStatistics>(reader.GetSource());
}

} // namespace duckdb



namespace duckdb {

LogicalLimit::LogicalLimit(int64_t limit_val, int64_t offset_val, unique_ptr<Expression> limit,
                           unique_ptr<Expression> offset)
    : LogicalOperator(LogicalOperatorType::LOGICAL_LIMIT), limit_val(limit_val), offset_val(offset_val),
      limit(move(limit)), offset(move(offset)) {
}

vector<ColumnBinding> LogicalLimit::GetColumnBindings() {
	return children[0]->GetColumnBindings();
}

idx_t LogicalLimit::EstimateCardinality(ClientContext &context) {
	auto child_cardinality = children[0]->EstimateCardinality(context);
	if (limit_val >= 0 && idx_t(limit_val) < child_cardinality) {
		child_cardinality = limit_val;
	}
	return child_cardinality;
}

void LogicalLimit::ResolveTypes() {
	types = children[0]->types;
}

void LogicalLimit::Serialize(FieldWriter &writer) const {
	writer.WriteField(limit_val);
	writer.WriteField(offset_val);
	writer.WriteOptional(limit);
	writer.WriteOptional(offset);
}

unique_ptr<LogicalOperator> LogicalLimit::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto limit_val = reader.ReadRequired<int64_t>();
	auto offset_val = reader.ReadRequired<int64_t>();
	auto limit = reader.ReadOptional<Expression>(nullptr, state.gstate);
	auto offset = reader.ReadOptional<Expression>(nullptr, state.gstate);
	return make_unique<LogicalLimit>(limit_val, offset_val, move(limit), move(offset));
}

} // namespace duckdb


#include <cmath>

namespace duckdb {

void LogicalLimitPercent::Serialize(FieldWriter &writer) const {
	writer.WriteField(limit_percent);
	writer.WriteField(offset_val);
	writer.WriteOptional(limit);
	writer.WriteOptional(offset);
}

unique_ptr<LogicalOperator> LogicalLimitPercent::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto limit_percent = reader.ReadRequired<double>();
	auto offset_val = reader.ReadRequired<int64_t>();
	auto limit = reader.ReadOptional<Expression>(nullptr, state.gstate);
	auto offset = reader.ReadOptional<Expression>(nullptr, state.gstate);
	return make_unique<LogicalLimitPercent>(limit_percent, offset_val, move(limit), move(offset));
}

idx_t LogicalLimitPercent::EstimateCardinality(ClientContext &context) {
	auto child_cardinality = LogicalOperator::EstimateCardinality(context);
	if ((limit_percent < 0 || limit_percent > 100) || std::isnan(limit_percent)) {
		return child_cardinality;
	}
	return idx_t(child_cardinality * (limit_percent / 100.0));
}

} // namespace duckdb




namespace duckdb {

void LogicalOrder::Serialize(FieldWriter &writer) const {
	writer.WriteRegularSerializableList(orders);
	writer.WriteList<idx_t>(projections);
}

unique_ptr<LogicalOperator> LogicalOrder::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto orders = reader.ReadRequiredSerializableList<BoundOrderByNode, BoundOrderByNode>(state.gstate);
	auto projections = reader.ReadRequiredList<idx_t>();
	auto result = make_unique<LogicalOrder>(move(orders));
	result->projections = move(projections);
	return move(result);
}

} // namespace duckdb


namespace duckdb {

void LogicalPragma::Serialize(FieldWriter &writer) const {
	throw NotImplementedException(LogicalOperatorToString(type));
}

unique_ptr<LogicalOperator> LogicalPragma::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	throw NotImplementedException(LogicalOperatorToString(state.type));
}

idx_t LogicalPragma::EstimateCardinality(ClientContext &context) {
	return 1;
}

} // namespace duckdb


namespace duckdb {

void LogicalPrepare::Serialize(FieldWriter &writer) const {
	throw NotImplementedException(LogicalOperatorToString(type));
}

unique_ptr<LogicalOperator> LogicalPrepare::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	throw NotImplementedException(LogicalOperatorToString(state.type));
}

idx_t LogicalPrepare::EstimateCardinality(ClientContext &context) {
	return 1;
}

} // namespace duckdb



namespace duckdb {

LogicalProjection::LogicalProjection(idx_t table_index, vector<unique_ptr<Expression>> select_list)
    : LogicalOperator(LogicalOperatorType::LOGICAL_PROJECTION, move(select_list)), table_index(table_index) {
}

vector<ColumnBinding> LogicalProjection::GetColumnBindings() {
	return GenerateColumnBindings(table_index, expressions.size());
}

void LogicalProjection::ResolveTypes() {
	for (auto &expr : expressions) {
		types.push_back(expr->return_type);
	}
}

void LogicalProjection::Serialize(FieldWriter &writer) const {
	writer.WriteField(table_index);
	writer.WriteSerializableList<Expression>(expressions);
}

unique_ptr<LogicalOperator> LogicalProjection::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto table_index = reader.ReadRequired<idx_t>();
	auto expressions = reader.ReadRequiredSerializableList<Expression>(state.gstate);
	return make_unique<LogicalProjection>(table_index, move(expressions));
}

vector<idx_t> LogicalProjection::GetTableIndex() const {
	return vector<idx_t> {table_index};
}

} // namespace duckdb



namespace duckdb {

void LogicalRecursiveCTE::Serialize(FieldWriter &writer) const {
	writer.WriteField(table_index);
	writer.WriteField(column_count);
	writer.WriteField(union_all);
}

unique_ptr<LogicalOperator> LogicalRecursiveCTE::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto table_index = reader.ReadRequired<idx_t>();
	auto column_count = reader.ReadRequired<idx_t>();
	auto union_all = reader.ReadRequired<bool>();
	// TODO(stephwang): review if unique_ptr<LogicalOperator> plan is needed
	return unique_ptr<LogicalRecursiveCTE>(new LogicalRecursiveCTE(table_index, column_count, union_all, state.type));
}

vector<idx_t> LogicalRecursiveCTE::GetTableIndex() const {
	return vector<idx_t> {table_index};
}

} // namespace duckdb



namespace duckdb {

LogicalSample::LogicalSample(unique_ptr<SampleOptions> sample_options_p, unique_ptr<LogicalOperator> child)
    : LogicalOperator(LogicalOperatorType::LOGICAL_SAMPLE), sample_options(move(sample_options_p)) {
	children.push_back(move(child));
}

vector<ColumnBinding> LogicalSample::GetColumnBindings() {
	return children[0]->GetColumnBindings();
}

idx_t LogicalSample::EstimateCardinality(ClientContext &context) {
	auto child_cardinality = children[0]->EstimateCardinality(context);
	if (sample_options->is_percentage) {
		double sample_cardinality =
		    double(child_cardinality) * (sample_options->sample_size.GetValue<double>() / 100.0);
		if (sample_cardinality > double(child_cardinality)) {
			return child_cardinality;
		}
		return idx_t(sample_cardinality);
	} else {
		auto sample_size = sample_options->sample_size.GetValue<uint64_t>();
		if (sample_size < child_cardinality) {
			return sample_size;
		}
	}
	return child_cardinality;
}

void LogicalSample::ResolveTypes() {
	types = children[0]->types;
}

void LogicalSample::Serialize(FieldWriter &writer) const {
	sample_options->Serialize(writer.GetSerializer());
}

unique_ptr<LogicalOperator> LogicalSample::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto sample_options = SampleOptions::Deserialize(reader.GetSource());
	// TODO(stephwang): review how to pass child LogicalOperator
	auto result = make_unique<LogicalSample>(move(sample_options), nullptr);
	return move(result);
}
} // namespace duckdb



namespace duckdb {

void LogicalSet::Serialize(FieldWriter &writer) const {
	writer.WriteString(name);
	value.Serialize(writer.GetSerializer());
	writer.WriteField(scope);
}

unique_ptr<LogicalOperator> LogicalSet::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto name = reader.ReadRequired<std::string>();
	auto value = Value::Deserialize(reader.GetSource());
	auto scope = reader.ReadRequired<SetScope>();
	return make_unique<LogicalSet>(name, value, scope);
}

idx_t LogicalSet::EstimateCardinality(ClientContext &context) {
	return 1;
}

} // namespace duckdb



namespace duckdb {

void LogicalSetOperation::Serialize(FieldWriter &writer) const {
	writer.WriteField(table_index);
	writer.WriteField(column_count);
}

unique_ptr<LogicalOperator> LogicalSetOperation::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto table_index = reader.ReadRequired<idx_t>();
	auto column_count = reader.ReadRequired<idx_t>();
	// TODO(stephwang): review if unique_ptr<LogicalOperator> plan is needed
	return unique_ptr<LogicalSetOperation>(new LogicalSetOperation(table_index, column_count, state.type));
}

vector<idx_t> LogicalSetOperation::GetTableIndex() const {
	return vector<idx_t> {table_index};
}

} // namespace duckdb



namespace duckdb {

void LogicalShow::Serialize(FieldWriter &writer) const {
	writer.WriteRegularSerializableList(types_select);
	writer.WriteList<string>(aliases);
}

unique_ptr<LogicalOperator> LogicalShow::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto types_select = reader.ReadRequiredSerializableList<LogicalType, LogicalType>();
	auto aliases = reader.ReadRequiredList<string>();

	// TODO(stephwang): review if we need to pass unique_ptr<LogicalOperator> plan
	auto result = unique_ptr<LogicalShow>(new LogicalShow());
	result->types_select = types_select;
	result->aliases = aliases;
	return move(result);
}
} // namespace duckdb


namespace duckdb {

void LogicalSimple::Serialize(FieldWriter &writer) const {
	throw NotImplementedException(LogicalOperatorToString(type));
}

unique_ptr<LogicalOperator> LogicalSimple::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	throw NotImplementedException(LogicalOperatorToString(state.type));
}

idx_t LogicalSimple::EstimateCardinality(ClientContext &context) {
	return 1;
}

} // namespace duckdb



namespace duckdb {

void LogicalTopN::Serialize(FieldWriter &writer) const {
	writer.WriteRegularSerializableList(orders);
	writer.WriteField(offset);
	writer.WriteField(limit);
}

unique_ptr<LogicalOperator> LogicalTopN::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto orders = reader.ReadRequiredSerializableList<BoundOrderByNode, BoundOrderByNode>(state.gstate);
	auto offset = reader.ReadRequired<idx_t>();
	auto limit = reader.ReadRequired<idx_t>();
	return make_unique<LogicalTopN>(move(orders), limit, offset);
}

idx_t LogicalTopN::EstimateCardinality(ClientContext &context) {
	auto child_cardinality = LogicalOperator::EstimateCardinality(context);
	if (limit >= 0 && child_cardinality < idx_t(limit)) {
		return limit;
	}
	return child_cardinality;
}

} // namespace duckdb



namespace duckdb {

vector<ColumnBinding> LogicalUnnest::GetColumnBindings() {
	auto child_bindings = children[0]->GetColumnBindings();
	for (idx_t i = 0; i < expressions.size(); i++) {
		child_bindings.emplace_back(unnest_index, i);
	}
	return child_bindings;
}

void LogicalUnnest::ResolveTypes() {
	types.insert(types.end(), children[0]->types.begin(), children[0]->types.end());
	for (auto &expr : expressions) {
		types.push_back(expr->return_type);
	}
}

void LogicalUnnest::Serialize(FieldWriter &writer) const {
	writer.WriteField(unnest_index);
	writer.WriteSerializableList<Expression>(expressions);
}

unique_ptr<LogicalOperator> LogicalUnnest::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto unnest_index = reader.ReadRequired<idx_t>();
	auto expressions = reader.ReadRequiredSerializableList<Expression>(state.gstate);
	auto result = make_unique<LogicalUnnest>(unnest_index);
	result->expressions = move(expressions);
	return move(result);
}

vector<idx_t> LogicalUnnest::GetTableIndex() const {
	return vector<idx_t> {unnest_index};
}

} // namespace duckdb





namespace duckdb {

void LogicalUpdate::Serialize(FieldWriter &writer) const {
	table->Serialize(writer.GetSerializer());
	writer.WriteField(table_index);
	writer.WriteField(return_chunk);
	writer.WriteIndexList<PhysicalIndex>(columns);
	writer.WriteSerializableList(bound_defaults);
	writer.WriteField(update_is_del_and_insert);
}

unique_ptr<LogicalOperator> LogicalUpdate::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto &context = state.gstate.context;
	auto info = TableCatalogEntry::Deserialize(reader.GetSource(), context);
	auto &catalog = Catalog::GetCatalog(context);

	TableCatalogEntry *table_catalog_entry = catalog.GetEntry<TableCatalogEntry>(context, info->schema, info->table);

	if (!table_catalog_entry) {
		throw InternalException("Cant find catalog entry for table %s", info->table);
	}

	auto result = make_unique<LogicalUpdate>(table_catalog_entry);
	result->table_index = reader.ReadRequired<idx_t>();
	result->return_chunk = reader.ReadRequired<bool>();
	result->columns = reader.ReadRequiredIndexList<PhysicalIndex>();
	result->bound_defaults = reader.ReadRequiredSerializableList<Expression>(state.gstate);
	result->update_is_del_and_insert = reader.ReadRequired<bool>();
	return move(result);
}

idx_t LogicalUpdate::EstimateCardinality(ClientContext &context) {
	return return_chunk ? LogicalOperator::EstimateCardinality(context) : 1;
}

} // namespace duckdb



namespace duckdb {

vector<ColumnBinding> LogicalWindow::GetColumnBindings() {
	auto child_bindings = children[0]->GetColumnBindings();
	for (idx_t i = 0; i < expressions.size(); i++) {
		child_bindings.emplace_back(window_index, i);
	}
	return child_bindings;
}

void LogicalWindow::ResolveTypes() {
	types.insert(types.end(), children[0]->types.begin(), children[0]->types.end());
	for (auto &expr : expressions) {
		types.push_back(expr->return_type);
	}
}

void LogicalWindow::Serialize(FieldWriter &writer) const {
	writer.WriteField(window_index);
	writer.WriteSerializableList<Expression>(expressions);
}

unique_ptr<LogicalOperator> LogicalWindow::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto window_index = reader.ReadRequired<idx_t>();
	auto result = make_unique<LogicalWindow>(window_index);
	result->expressions = reader.ReadRequiredSerializableList<Expression>(state.gstate);
	return move(result);
}

vector<idx_t> LogicalWindow::GetTableIndex() const {
	return vector<idx_t> {window_index};
}

} // namespace duckdb





namespace duckdb {
void BoundCreateTableInfo::Serialize(Serializer &serializer) const {
	D_ASSERT(schema);
	schema->Serialize(serializer);
	serializer.WriteOptional(base);

	// TODO[YLM]: Review if we want/need to serialize more of the fields.
	//! The map of column names -> column index, used during binding
	// case_insensitive_map_t<column_t> name_map;

	//! Column dependency manager of the table
	// ColumnDependencyManager column_dependency_manager;

	serializer.WriteList(constraints);
	serializer.WriteList(bound_constraints);
	serializer.WriteList(bound_defaults);

	//! Dependents of the table (in e.g. default values)
	// unordered_set<CatalogEntry *> dependencies;

	//! The existing table data on disk (if any)
	// unique_ptr<PersistentTableData> data;

	//! CREATE TABLE from QUERY
	serializer.WriteOptional(query);

	//! Indexes created by this table <Block_ID, Offset>
	// vector<BlockPointer> indexes;
}

unique_ptr<BoundCreateTableInfo> BoundCreateTableInfo::Deserialize(Deserializer &source,
                                                                   PlanDeserializationState &state) {
	auto create_info = SchemaCatalogEntry::Deserialize(source);
	auto schema_name = create_info->schema;
	auto result = make_unique<BoundCreateTableInfo>(move(create_info));
	auto &context = state.context;
	result->schema = Catalog::GetCatalog(context).GetSchema(context, schema_name);
	result->base = source.ReadOptional<CreateInfo>();

	source.ReadList<Constraint>(result->constraints);
	source.ReadList<BoundConstraint>(result->bound_constraints);
	source.ReadList<Expression>(result->bound_defaults, state);

	result->query = source.ReadOptional<LogicalOperator>(state);
	return result;
}
} // namespace duckdb



namespace duckdb {

PlanDeserializationState::PlanDeserializationState(ClientContext &context) : context(context) {
}
PlanDeserializationState::~PlanDeserializationState() {
}

LogicalDeserializationState::LogicalDeserializationState(PlanDeserializationState &gstate, LogicalOperatorType type,
                                                         vector<unique_ptr<LogicalOperator>> &children)
    : gstate(gstate), type(type), children(children) {
}

ExpressionDeserializationState::ExpressionDeserializationState(PlanDeserializationState &gstate, ExpressionType type)
    : gstate(gstate), type(type) {
}

} // namespace duckdb













namespace duckdb {

Planner::Planner(ClientContext &context) : binder(Binder::CreateBinder(context)), context(context) {
}

static void CheckTreeDepth(const LogicalOperator &op, idx_t max_depth, idx_t depth = 0) {
	if (depth >= max_depth) {
		throw ParserException("Maximum tree depth of %lld exceeded in logical planner", max_depth);
	}
	for (auto &child : op.children) {
		CheckTreeDepth(*child, max_depth, depth + 1);
	}
}

void Planner::CreatePlan(SQLStatement &statement) {
	auto &profiler = QueryProfiler::Get(context);
	auto parameter_count = statement.n_param;

	BoundParameterMap bound_parameters(parameter_data);

	// first bind the tables and columns to the catalog
	bool parameters_resolved = true;
	try {
		profiler.StartPhase("binder");
		binder->parameters = &bound_parameters;
		auto bound_statement = binder->Bind(statement);
		profiler.EndPhase();

		this->names = bound_statement.names;
		this->types = bound_statement.types;
		this->plan = move(bound_statement.plan);

		auto max_tree_depth = ClientConfig::GetConfig(context).max_expression_depth;
		CheckTreeDepth(*plan, max_tree_depth);
	} catch (const ParameterNotResolvedException &ex) {
		// parameter types could not be resolved
		this->names = {"unknown"};
		this->types = {LogicalTypeId::UNKNOWN};
		this->plan = nullptr;
		parameters_resolved = false;
	} catch (const Exception &ex) {
		auto &config = DBConfig::GetConfig(context);

		this->plan = nullptr;
		for (auto &extension_op : config.operator_extensions) {
			auto bound_statement =
			    extension_op->Bind(context, *this->binder, extension_op->operator_info.get(), statement);
			if (bound_statement.plan != nullptr) {
				this->names = bound_statement.names;
				this->types = bound_statement.types;
				this->plan = move(bound_statement.plan);
				break;
			}
		}

		if (!this->plan) {
			throw;
		}
	} catch (std::exception &ex) {
		throw;
	}
	this->properties = binder->properties;
	this->properties.parameter_count = parameter_count;
	properties.bound_all_parameters = parameters_resolved;

	Planner::VerifyPlan(context, plan, &bound_parameters.parameters);

	// set up a map of parameter number -> value entries
	for (auto &kv : bound_parameters.parameters) {
		auto parameter_index = kv.first;
		auto &parameter_data = kv.second;
		// check if the type of the parameter could be resolved
		if (!parameter_data->return_type.IsValid()) {
			properties.bound_all_parameters = false;
			continue;
		}
		parameter_data->value = Value(parameter_data->return_type);
		value_map[parameter_index] = parameter_data;
	}
}

shared_ptr<PreparedStatementData> Planner::PrepareSQLStatement(unique_ptr<SQLStatement> statement) {
	auto copied_statement = statement->Copy();
	// create a plan of the underlying statement
	CreatePlan(move(statement));
	// now create the logical prepare
	auto prepared_data = make_shared<PreparedStatementData>(copied_statement->type);
	prepared_data->unbound_statement = move(copied_statement);
	prepared_data->names = names;
	prepared_data->types = types;
	prepared_data->value_map = move(value_map);
	prepared_data->properties = properties;
	prepared_data->catalog_version = Transaction::GetTransaction(context).catalog_version;
	return prepared_data;
}

void Planner::CreatePlan(unique_ptr<SQLStatement> statement) {
	D_ASSERT(statement);
	switch (statement->type) {
	case StatementType::SELECT_STATEMENT:
	case StatementType::INSERT_STATEMENT:
	case StatementType::COPY_STATEMENT:
	case StatementType::DELETE_STATEMENT:
	case StatementType::UPDATE_STATEMENT:
	case StatementType::CREATE_STATEMENT:
	case StatementType::DROP_STATEMENT:
	case StatementType::ALTER_STATEMENT:
	case StatementType::TRANSACTION_STATEMENT:
	case StatementType::EXPLAIN_STATEMENT:
	case StatementType::VACUUM_STATEMENT:
	case StatementType::RELATION_STATEMENT:
	case StatementType::CALL_STATEMENT:
	case StatementType::EXPORT_STATEMENT:
	case StatementType::PRAGMA_STATEMENT:
	case StatementType::SHOW_STATEMENT:
	case StatementType::SET_STATEMENT:
	case StatementType::LOAD_STATEMENT:
	case StatementType::EXTENSION_STATEMENT:
	case StatementType::PREPARE_STATEMENT:
	case StatementType::EXECUTE_STATEMENT:
	case StatementType::LOGICAL_PLAN_STATEMENT:
		CreatePlan(*statement);
		break;
	default:
		throw NotImplementedException("Cannot plan statement of type %s!", StatementTypeToString(statement->type));
	}
}

static bool OperatorSupportsSerialization(LogicalOperator &op) {
	switch (op.type) {
	case LogicalOperatorType::LOGICAL_INSERT:
	case LogicalOperatorType::LOGICAL_UPDATE:
	case LogicalOperatorType::LOGICAL_DELETE:
	case LogicalOperatorType::LOGICAL_PREPARE:
	case LogicalOperatorType::LOGICAL_EXECUTE:
	case LogicalOperatorType::LOGICAL_ALTER:
	case LogicalOperatorType::LOGICAL_CREATE_TABLE:
	case LogicalOperatorType::LOGICAL_CREATE_INDEX:
	case LogicalOperatorType::LOGICAL_CREATE_SEQUENCE:
	case LogicalOperatorType::LOGICAL_CREATE_VIEW:
	case LogicalOperatorType::LOGICAL_CREATE_SCHEMA:
	case LogicalOperatorType::LOGICAL_CREATE_MACRO:
	case LogicalOperatorType::LOGICAL_DROP:
	case LogicalOperatorType::LOGICAL_PRAGMA:
	case LogicalOperatorType::LOGICAL_TRANSACTION:
	case LogicalOperatorType::LOGICAL_CREATE_TYPE:
	case LogicalOperatorType::LOGICAL_EXPLAIN:
	case LogicalOperatorType::LOGICAL_COPY_TO_FILE:
	case LogicalOperatorType::LOGICAL_LOAD:
	case LogicalOperatorType::LOGICAL_VACUUM:
		// unsupported (for now)
		return false;
	default:
		break;
	}
	for (auto &child : op.children) {
		if (!OperatorSupportsSerialization(*child)) {
			return false;
		}
	}
	return true;
}

void Planner::VerifyPlan(ClientContext &context, unique_ptr<LogicalOperator> &op, bound_parameter_map_t *map) {
	if (!op || !ClientConfig::GetConfig(context).verify_serializer) {
		return;
	}
	//! SELECT only for now
	if (!OperatorSupportsSerialization(*op)) {
		return;
	}

	BufferedSerializer serializer;
	try {
		op->Serialize(serializer);
	} catch (NotImplementedException &ex) {
		// ignore for now (FIXME)
		return;
	}
	auto data = serializer.GetData();
	auto deserializer = BufferedDeserializer(data.data.get(), data.size);

	PlanDeserializationState state(context);
	auto new_plan = LogicalOperator::Deserialize(deserializer, state);
	if (map) {
		*map = move(state.parameter_data);
	}
	op = move(new_plan);
}

} // namespace duckdb
















namespace duckdb {

PragmaHandler::PragmaHandler(ClientContext &context) : context(context) {
}

void PragmaHandler::HandlePragmaStatementsInternal(vector<unique_ptr<SQLStatement>> &statements) {
	vector<unique_ptr<SQLStatement>> new_statements;
	for (idx_t i = 0; i < statements.size(); i++) {
		if (statements[i]->type == StatementType::PRAGMA_STATEMENT) {
			// PRAGMA statement: check if we need to replace it by a new set of statements
			PragmaHandler handler(context);
			auto new_query = handler.HandlePragma(statements[i].get());
			if (!new_query.empty()) {
				// this PRAGMA statement gets replaced by a new query string
				// push the new query string through the parser again and add it to the transformer
				Parser parser(context.GetParserOptions());
				parser.ParseQuery(new_query);
				// insert the new statements and remove the old statement
				for (idx_t j = 0; j < parser.statements.size(); j++) {
					new_statements.push_back(move(parser.statements[j]));
				}
				continue;
			}
		}
		new_statements.push_back(move(statements[i]));
	}
	statements = move(new_statements);
}

void PragmaHandler::HandlePragmaStatements(ClientContextLock &lock, vector<unique_ptr<SQLStatement>> &statements) {
	// first check if there are any pragma statements
	bool found_pragma = false;
	for (idx_t i = 0; i < statements.size(); i++) {
		if (statements[i]->type == StatementType::PRAGMA_STATEMENT) {
			found_pragma = true;
			break;
		}
	}
	if (!found_pragma) {
		// no pragmas: skip this step
		return;
	}
	context.RunFunctionInTransactionInternal(lock, [&]() { HandlePragmaStatementsInternal(statements); });
}

string PragmaHandler::HandlePragma(SQLStatement *statement) { // PragmaInfo &info
	auto info = *((PragmaStatement &)*statement).info;
	auto entry =
	    Catalog::GetCatalog(context).GetEntry<PragmaFunctionCatalogEntry>(context, DEFAULT_SCHEMA, info.name, false);
	string error;

	FunctionBinder function_binder(context);
	idx_t bound_idx = function_binder.BindFunction(entry->name, entry->functions, info, error);
	if (bound_idx == DConstants::INVALID_INDEX) {
		throw BinderException(error);
	}
	auto bound_function = entry->functions.GetFunctionByOffset(bound_idx);
	if (bound_function.query) {
		QueryErrorContext error_context(statement, statement->stmt_location);
		Binder::BindNamedParameters(bound_function.named_parameters, info.named_parameters, error_context,
		                            bound_function.name);
		FunctionParameters parameters {info.parameters, info.named_parameters};
		return bound_function.query(context, parameters);
	}
	return string();
}

} // namespace duckdb













namespace duckdb {

FlattenDependentJoins::FlattenDependentJoins(Binder &binder, const vector<CorrelatedColumnInfo> &correlated,
                                             bool perform_delim, bool any_join)
    : binder(binder), correlated_columns(correlated), perform_delim(perform_delim), any_join(any_join) {
	for (idx_t i = 0; i < correlated_columns.size(); i++) {
		auto &col = correlated_columns[i];
		correlated_map[col.binding] = i;
		delim_types.push_back(col.type);
	}
}

bool FlattenDependentJoins::DetectCorrelatedExpressions(LogicalOperator *op) {
	D_ASSERT(op);
	// check if this entry has correlated expressions
	HasCorrelatedExpressions visitor(correlated_columns);
	visitor.VisitOperator(*op);
	bool has_correlation = visitor.has_correlated_expressions;
	// now visit the children of this entry and check if they have correlated expressions
	for (auto &child : op->children) {
		// we OR the property with its children such that has_correlation is true if either
		// (1) this node has a correlated expression or
		// (2) one of its children has a correlated expression
		if (DetectCorrelatedExpressions(child.get())) {
			has_correlation = true;
		}
	}
	// set the entry in the map
	has_correlated_expressions[op] = has_correlation;
	return has_correlation;
}

unique_ptr<LogicalOperator> FlattenDependentJoins::PushDownDependentJoin(unique_ptr<LogicalOperator> plan) {
	bool propagate_null_values = true;
	auto result = PushDownDependentJoinInternal(move(plan), propagate_null_values);
	if (!replacement_map.empty()) {
		// check if we have to replace any COUNT aggregates into "CASE WHEN X IS NULL THEN 0 ELSE COUNT END"
		RewriteCountAggregates aggr(replacement_map);
		aggr.VisitOperator(*result);
	}
	return result;
}

bool SubqueryDependentFilter(Expression *expr) {
	if (expr->expression_class == ExpressionClass::BOUND_CONJUNCTION &&
	    expr->GetExpressionType() == ExpressionType::CONJUNCTION_AND) {
		auto bound_conjuction = (BoundConjunctionExpression *)expr;
		for (auto &child : bound_conjuction->children) {
			if (SubqueryDependentFilter(child.get())) {
				return true;
			}
		}
	}
	if (expr->expression_class == ExpressionClass::BOUND_SUBQUERY) {
		return true;
	}
	return false;
}
unique_ptr<LogicalOperator> FlattenDependentJoins::PushDownDependentJoinInternal(unique_ptr<LogicalOperator> plan,
                                                                                 bool &parent_propagate_null_values) {
	// first check if the logical operator has correlated expressions
	auto entry = has_correlated_expressions.find(plan.get());
	D_ASSERT(entry != has_correlated_expressions.end());
	if (!entry->second) {
		// we reached a node without correlated expressions
		// we can eliminate the dependent join now and create a simple cross product
		// now create the duplicate eliminated scan for this node
		auto left_columns = plan->GetColumnBindings().size();
		auto delim_index = binder.GenerateTableIndex();
		this->base_binding = ColumnBinding(delim_index, 0);
		this->delim_offset = 0;
		this->data_offset = left_columns;
		auto delim_scan = make_unique<LogicalDelimGet>(delim_index, delim_types);
		return LogicalCrossProduct::Create(move(plan), move(delim_scan));
	}
	switch (plan->type) {
	case LogicalOperatorType::LOGICAL_UNNEST:
	case LogicalOperatorType::LOGICAL_FILTER: {
		// filter
		// first we flatten the dependent join in the child of the filter
		for (auto &expr : plan->expressions) {
			any_join |= SubqueryDependentFilter(expr.get());
		}
		plan->children[0] = PushDownDependentJoinInternal(move(plan->children[0]), parent_propagate_null_values);

		// then we replace any correlated expressions with the corresponding entry in the correlated_map
		RewriteCorrelatedExpressions rewriter(base_binding, correlated_map);
		rewriter.VisitOperator(*plan);
		return plan;
	}
	case LogicalOperatorType::LOGICAL_PROJECTION: {
		// projection
		// first we flatten the dependent join in the child of the projection
		for (auto &expr : plan->expressions) {
			parent_propagate_null_values &= expr->PropagatesNullValues();
		}
		plan->children[0] = PushDownDependentJoinInternal(move(plan->children[0]), parent_propagate_null_values);

		// then we replace any correlated expressions with the corresponding entry in the correlated_map
		RewriteCorrelatedExpressions rewriter(base_binding, correlated_map);
		rewriter.VisitOperator(*plan);
		// now we add all the columns of the delim_scan to the projection list
		auto proj = (LogicalProjection *)plan.get();
		for (idx_t i = 0; i < correlated_columns.size(); i++) {
			auto &col = correlated_columns[i];
			auto colref = make_unique<BoundColumnRefExpression>(
			    col.name, col.type, ColumnBinding(base_binding.table_index, base_binding.column_index + i));
			plan->expressions.push_back(move(colref));
		}

		base_binding.table_index = proj->table_index;
		this->delim_offset = base_binding.column_index = plan->expressions.size() - correlated_columns.size();
		this->data_offset = 0;
		return plan;
	}
	case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY: {
		auto &aggr = (LogicalAggregate &)*plan;
		// aggregate and group by
		// first we flatten the dependent join in the child of the projection
		for (auto &expr : plan->expressions) {
			parent_propagate_null_values &= expr->PropagatesNullValues();
		}
		plan->children[0] = PushDownDependentJoinInternal(move(plan->children[0]), parent_propagate_null_values);
		// then we replace any correlated expressions with the corresponding entry in the correlated_map
		RewriteCorrelatedExpressions rewriter(base_binding, correlated_map);
		rewriter.VisitOperator(*plan);
		// now we add all the columns of the delim_scan to the grouping operators AND the projection list
		idx_t delim_table_index;
		idx_t delim_column_offset;
		idx_t delim_data_offset;
		auto new_group_count = perform_delim ? correlated_columns.size() : 1;
		for (idx_t i = 0; i < new_group_count; i++) {
			auto &col = correlated_columns[i];
			auto colref = make_unique<BoundColumnRefExpression>(
			    col.name, col.type, ColumnBinding(base_binding.table_index, base_binding.column_index + i));
			for (auto &set : aggr.grouping_sets) {
				set.insert(aggr.groups.size());
			}
			aggr.groups.push_back(move(colref));
		}
		if (!perform_delim) {
			// if we are not performing the duplicate elimination, we have only added the row_id column to the grouping
			// operators in this case, we push a FIRST aggregate for each of the remaining expressions
			delim_table_index = aggr.aggregate_index;
			delim_column_offset = aggr.expressions.size();
			delim_data_offset = aggr.groups.size();
			for (idx_t i = 0; i < correlated_columns.size(); i++) {
				auto &col = correlated_columns[i];
				auto first_aggregate = FirstFun::GetFunction(col.type);
				auto colref = make_unique<BoundColumnRefExpression>(
				    col.name, col.type, ColumnBinding(base_binding.table_index, base_binding.column_index + i));
				vector<unique_ptr<Expression>> aggr_children;
				aggr_children.push_back(move(colref));
				auto first_fun = make_unique<BoundAggregateExpression>(move(first_aggregate), move(aggr_children),
				                                                       nullptr, nullptr, AggregateType::NON_DISTINCT);
				aggr.expressions.push_back(move(first_fun));
			}
		} else {
			delim_table_index = aggr.group_index;
			delim_column_offset = aggr.groups.size() - correlated_columns.size();
			delim_data_offset = aggr.groups.size();
		}
		if (aggr.groups.size() == new_group_count) {
			// we have to perform a LEFT OUTER JOIN between the result of this aggregate and the delim scan
			// FIXME: this does not always have to be a LEFT OUTER JOIN, depending on whether aggr.expressions return
			// NULL or a value
			unique_ptr<LogicalComparisonJoin> join = make_unique<LogicalComparisonJoin>(JoinType::INNER);
			for (auto &aggr_exp : aggr.expressions) {
				auto b_aggr_exp = (BoundAggregateExpression *)aggr_exp.get();
				if (!b_aggr_exp->PropagatesNullValues() || any_join || !parent_propagate_null_values) {
					join = make_unique<LogicalComparisonJoin>(JoinType::LEFT);
					break;
				}
			}
			auto left_index = binder.GenerateTableIndex();
			auto delim_scan = make_unique<LogicalDelimGet>(left_index, delim_types);
			join->children.push_back(move(delim_scan));
			join->children.push_back(move(plan));
			for (idx_t i = 0; i < new_group_count; i++) {
				auto &col = correlated_columns[i];
				JoinCondition cond;
				cond.left = make_unique<BoundColumnRefExpression>(col.name, col.type, ColumnBinding(left_index, i));
				cond.right = make_unique<BoundColumnRefExpression>(
				    correlated_columns[i].type, ColumnBinding(delim_table_index, delim_column_offset + i));
				cond.comparison = ExpressionType::COMPARE_NOT_DISTINCT_FROM;
				join->conditions.push_back(move(cond));
			}
			// for any COUNT aggregate we replace references to the column with: CASE WHEN COUNT(*) IS NULL THEN 0
			// ELSE COUNT(*) END
			for (idx_t i = 0; i < aggr.expressions.size(); i++) {
				D_ASSERT(aggr.expressions[i]->GetExpressionClass() == ExpressionClass::BOUND_AGGREGATE);
				auto bound = (BoundAggregateExpression *)&*aggr.expressions[i];
				vector<LogicalType> arguments;
				if (bound->function == CountFun::GetFunction() || bound->function == CountStarFun::GetFunction()) {
					// have to replace this ColumnBinding with the CASE expression
					replacement_map[ColumnBinding(aggr.aggregate_index, i)] = i;
				}
			}
			// now we update the delim_index
			base_binding.table_index = left_index;
			this->delim_offset = base_binding.column_index = 0;
			this->data_offset = 0;
			return move(join);
		} else {
			// update the delim_index
			base_binding.table_index = delim_table_index;
			this->delim_offset = base_binding.column_index = delim_column_offset;
			this->data_offset = delim_data_offset;
			return plan;
		}
	}
	case LogicalOperatorType::LOGICAL_CROSS_PRODUCT: {
		// cross product
		// push into both sides of the plan
		bool left_has_correlation = has_correlated_expressions.find(plan->children[0].get())->second;
		bool right_has_correlation = has_correlated_expressions.find(plan->children[1].get())->second;
		if (!right_has_correlation) {
			// only left has correlation: push into left
			plan->children[0] = PushDownDependentJoinInternal(move(plan->children[0]), parent_propagate_null_values);
			return plan;
		}
		if (!left_has_correlation) {
			// only right has correlation: push into right
			plan->children[1] = PushDownDependentJoinInternal(move(plan->children[1]), parent_propagate_null_values);
			return plan;
		}
		// both sides have correlation
		// turn into an inner join
		auto join = make_unique<LogicalComparisonJoin>(JoinType::INNER);
		plan->children[0] = PushDownDependentJoinInternal(move(plan->children[0]), parent_propagate_null_values);
		auto left_binding = this->base_binding;
		plan->children[1] = PushDownDependentJoinInternal(move(plan->children[1]), parent_propagate_null_values);
		// add the correlated columns to the join conditions
		for (idx_t i = 0; i < correlated_columns.size(); i++) {
			JoinCondition cond;
			cond.left = make_unique<BoundColumnRefExpression>(
			    correlated_columns[i].type, ColumnBinding(left_binding.table_index, left_binding.column_index + i));
			cond.right = make_unique<BoundColumnRefExpression>(
			    correlated_columns[i].type, ColumnBinding(base_binding.table_index, base_binding.column_index + i));
			cond.comparison = ExpressionType::COMPARE_NOT_DISTINCT_FROM;
			join->conditions.push_back(move(cond));
		}
		join->children.push_back(move(plan->children[0]));
		join->children.push_back(move(plan->children[1]));
		return move(join);
	}
	case LogicalOperatorType::LOGICAL_ANY_JOIN:
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN: {
		auto &join = (LogicalJoin &)*plan;
		D_ASSERT(plan->children.size() == 2);
		// check the correlated expressions in the children of the join
		bool left_has_correlation = has_correlated_expressions.find(plan->children[0].get())->second;
		bool right_has_correlation = has_correlated_expressions.find(plan->children[1].get())->second;

		if (join.join_type == JoinType::INNER) {
			// inner join
			if (!right_has_correlation) {
				// only left has correlation: push into left
				plan->children[0] =
				    PushDownDependentJoinInternal(move(plan->children[0]), parent_propagate_null_values);
				return plan;
			}
			if (!left_has_correlation) {
				// only right has correlation: push into right
				plan->children[1] =
				    PushDownDependentJoinInternal(move(plan->children[1]), parent_propagate_null_values);
				return plan;
			}
		} else if (join.join_type == JoinType::LEFT) {
			// left outer join
			if (!right_has_correlation) {
				// only left has correlation: push into left
				plan->children[0] =
				    PushDownDependentJoinInternal(move(plan->children[0]), parent_propagate_null_values);
				return plan;
			}
		} else if (join.join_type == JoinType::RIGHT) {
			// left outer join
			if (!left_has_correlation) {
				// only right has correlation: push into right
				plan->children[1] =
				    PushDownDependentJoinInternal(move(plan->children[1]), parent_propagate_null_values);
				return plan;
			}
		} else if (join.join_type == JoinType::MARK) {
			if (right_has_correlation) {
				throw Exception("MARK join with correlation in RHS not supported");
			}
			// push the child into the LHS
			plan->children[0] = PushDownDependentJoinInternal(move(plan->children[0]), parent_propagate_null_values);
			// rewrite expressions in the join conditions
			RewriteCorrelatedExpressions rewriter(base_binding, correlated_map);
			rewriter.VisitOperator(*plan);
			return plan;
		} else {
			throw Exception("Unsupported join type for flattening correlated subquery");
		}
		// both sides have correlation
		// push into both sides
		plan->children[0] = PushDownDependentJoinInternal(move(plan->children[0]), parent_propagate_null_values);
		auto left_binding = this->base_binding;
		plan->children[1] = PushDownDependentJoinInternal(move(plan->children[1]), parent_propagate_null_values);
		auto right_binding = this->base_binding;
		// NOTE: for OUTER JOINS it matters what the BASE BINDING is after the join
		// for the LEFT OUTER JOIN, we want the LEFT side to be the base binding after we push
		// because the RIGHT binding might contain NULL values
		if (join.join_type == JoinType::LEFT) {
			this->base_binding = left_binding;
		} else if (join.join_type == JoinType::RIGHT) {
			this->base_binding = right_binding;
		}
		// add the correlated columns to the join conditions
		for (idx_t i = 0; i < correlated_columns.size(); i++) {
			auto left = make_unique<BoundColumnRefExpression>(
			    correlated_columns[i].type, ColumnBinding(left_binding.table_index, left_binding.column_index + i));
			auto right = make_unique<BoundColumnRefExpression>(
			    correlated_columns[i].type, ColumnBinding(right_binding.table_index, right_binding.column_index + i));

			if (join.type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
				JoinCondition cond;
				cond.left = move(left);
				cond.right = move(right);
				cond.comparison = ExpressionType::COMPARE_NOT_DISTINCT_FROM;

				auto &comparison_join = (LogicalComparisonJoin &)join;
				comparison_join.conditions.push_back(move(cond));
			} else {
				auto &any_join = (LogicalAnyJoin &)join;
				auto comparison = make_unique<BoundComparisonExpression>(ExpressionType::COMPARE_NOT_DISTINCT_FROM,
				                                                         move(left), move(right));
				auto conjunction = make_unique<BoundConjunctionExpression>(ExpressionType::CONJUNCTION_AND,
				                                                           move(comparison), move(any_join.condition));
				any_join.condition = move(conjunction);
			}
		}
		// then we replace any correlated expressions with the corresponding entry in the correlated_map
		RewriteCorrelatedExpressions rewriter(right_binding, correlated_map);
		rewriter.VisitOperator(*plan);
		return plan;
	}
	case LogicalOperatorType::LOGICAL_LIMIT: {
		auto &limit = (LogicalLimit &)*plan;
		if (limit.limit || limit.offset) {
			throw ParserException("Non-constant limit or offset not supported in correlated subquery");
		}
		auto rownum_alias = "limit_rownum";
		unique_ptr<LogicalOperator> child;
		unique_ptr<LogicalOrder> order_by;

		// check if the direct child of this LIMIT node is an ORDER BY node, if so, keep it separate
		// this is done for an optimization to avoid having to compute the total order
		if (plan->children[0]->type == LogicalOperatorType::LOGICAL_ORDER_BY) {
			order_by = unique_ptr_cast<LogicalOperator, LogicalOrder>(move(plan->children[0]));
			child = PushDownDependentJoinInternal(move(order_by->children[0]), parent_propagate_null_values);
		} else {
			child = PushDownDependentJoinInternal(move(plan->children[0]), parent_propagate_null_values);
		}
		auto child_column_count = child->GetColumnBindings().size();
		// we push a row_number() OVER (PARTITION BY [correlated columns])
		auto window_index = binder.GenerateTableIndex();
		auto window = make_unique<LogicalWindow>(window_index);
		auto row_number = make_unique<BoundWindowExpression>(ExpressionType::WINDOW_ROW_NUMBER, LogicalType::BIGINT,
		                                                     nullptr, nullptr);
		auto partition_count = perform_delim ? correlated_columns.size() : 1;
		for (idx_t i = 0; i < partition_count; i++) {
			auto &col = correlated_columns[i];
			auto colref = make_unique<BoundColumnRefExpression>(
			    col.name, col.type, ColumnBinding(base_binding.table_index, base_binding.column_index + i));
			row_number->partitions.push_back(move(colref));
		}
		if (order_by) {
			// optimization: if there is an ORDER BY node followed by a LIMIT
			// rather than computing the entire order, we push the ORDER BY expressions into the row_num computation
			// this way, the order only needs to be computed per partition
			row_number->orders = move(order_by->orders);
		}
		row_number->start = WindowBoundary::UNBOUNDED_PRECEDING;
		row_number->end = WindowBoundary::CURRENT_ROW_ROWS;
		window->expressions.push_back(move(row_number));
		window->children.push_back(move(child));

		// add a filter based on the row_number
		// the filter we add is "row_number > offset AND row_number <= offset + limit"
		auto filter = make_unique<LogicalFilter>();
		unique_ptr<Expression> condition;
		auto row_num_ref =
		    make_unique<BoundColumnRefExpression>(rownum_alias, LogicalType::BIGINT, ColumnBinding(window_index, 0));

		int64_t upper_bound_limit = NumericLimits<int64_t>::Maximum();
		TryAddOperator::Operation(limit.offset_val, limit.limit_val, upper_bound_limit);
		auto upper_bound = make_unique<BoundConstantExpression>(Value::BIGINT(upper_bound_limit));
		condition = make_unique<BoundComparisonExpression>(ExpressionType::COMPARE_LESSTHANOREQUALTO,
		                                                   row_num_ref->Copy(), move(upper_bound));
		// we only need to add "row_number >= offset + 1" if offset is bigger than 0
		if (limit.offset_val > 0) {
			auto lower_bound = make_unique<BoundConstantExpression>(Value::BIGINT(limit.offset_val));
			auto lower_comp = make_unique<BoundComparisonExpression>(ExpressionType::COMPARE_GREATERTHAN,
			                                                         row_num_ref->Copy(), move(lower_bound));
			auto conj = make_unique<BoundConjunctionExpression>(ExpressionType::CONJUNCTION_AND, move(lower_comp),
			                                                    move(condition));
			condition = move(conj);
		}
		filter->expressions.push_back(move(condition));
		filter->children.push_back(move(window));
		// we prune away the row_number after the filter clause using the projection map
		for (idx_t i = 0; i < child_column_count; i++) {
			filter->projection_map.push_back(i);
		}
		return move(filter);
	}
	case LogicalOperatorType::LOGICAL_LIMIT_PERCENT: {
		// NOTE: limit percent could be supported in a manner similar to the LIMIT above
		// but instead of filtering by an exact number of rows, the limit should be expressed as
		// COUNT computed over the partition multiplied by the percentage
		throw ParserException("Limit percent operator not supported in correlated subquery");
	}
	case LogicalOperatorType::LOGICAL_WINDOW: {
		auto &window = (LogicalWindow &)*plan;
		// push into children
		plan->children[0] = PushDownDependentJoinInternal(move(plan->children[0]), parent_propagate_null_values);
		// add the correlated columns to the PARTITION BY clauses in the Window
		for (auto &expr : window.expressions) {
			D_ASSERT(expr->GetExpressionClass() == ExpressionClass::BOUND_WINDOW);
			auto &w = (BoundWindowExpression &)*expr;
			for (idx_t i = 0; i < correlated_columns.size(); i++) {
				w.partitions.push_back(make_unique<BoundColumnRefExpression>(
				    correlated_columns[i].type,
				    ColumnBinding(base_binding.table_index, base_binding.column_index + i)));
			}
		}
		return plan;
	}
	case LogicalOperatorType::LOGICAL_EXCEPT:
	case LogicalOperatorType::LOGICAL_INTERSECT:
	case LogicalOperatorType::LOGICAL_UNION: {
		auto &setop = (LogicalSetOperation &)*plan;
		// set operator, push into both children
#ifdef DEBUG
		plan->children[0]->ResolveOperatorTypes();
		plan->children[1]->ResolveOperatorTypes();
		D_ASSERT(plan->children[0]->types == plan->children[1]->types);
#endif
		plan->children[0] = PushDownDependentJoin(move(plan->children[0]));
		plan->children[1] = PushDownDependentJoin(move(plan->children[1]));
#ifdef DEBUG
		D_ASSERT(plan->children[0]->GetColumnBindings().size() == plan->children[1]->GetColumnBindings().size());
		plan->children[0]->ResolveOperatorTypes();
		plan->children[1]->ResolveOperatorTypes();
		D_ASSERT(plan->children[0]->types == plan->children[1]->types);
#endif
		// we have to refer to the setop index now
		base_binding.table_index = setop.table_index;
		base_binding.column_index = setop.column_count;
		setop.column_count += correlated_columns.size();
		return plan;
	}
	case LogicalOperatorType::LOGICAL_DISTINCT: {
		auto &distinct = (LogicalDistinct &)*plan;
		// push down into child
		distinct.children[0] = PushDownDependentJoin(move(distinct.children[0]));
		// add all correlated columns to the distinct targets
		for (idx_t i = 0; i < correlated_columns.size(); i++) {
			distinct.distinct_targets.push_back(make_unique<BoundColumnRefExpression>(
			    correlated_columns[i].type, ColumnBinding(base_binding.table_index, base_binding.column_index + i)));
		}
		return plan;
	}
	case LogicalOperatorType::LOGICAL_EXPRESSION_GET: {
		// expression get
		// first we flatten the dependent join in the child
		plan->children[0] = PushDownDependentJoinInternal(move(plan->children[0]), parent_propagate_null_values);
		// then we replace any correlated expressions with the corresponding entry in the correlated_map
		RewriteCorrelatedExpressions rewriter(base_binding, correlated_map);
		rewriter.VisitOperator(*plan);
		// now we add all the correlated columns to each of the expressions of the expression scan
		auto expr_get = (LogicalExpressionGet *)plan.get();
		for (idx_t i = 0; i < correlated_columns.size(); i++) {
			for (auto &expr_list : expr_get->expressions) {
				auto colref = make_unique<BoundColumnRefExpression>(
				    correlated_columns[i].type, ColumnBinding(base_binding.table_index, base_binding.column_index + i));
				expr_list.push_back(move(colref));
			}
			expr_get->expr_types.push_back(correlated_columns[i].type);
		}

		base_binding.table_index = expr_get->table_index;
		this->delim_offset = base_binding.column_index = expr_get->expr_types.size() - correlated_columns.size();
		this->data_offset = 0;
		return plan;
	}
	case LogicalOperatorType::LOGICAL_ORDER_BY:
		plan->children[0] = PushDownDependentJoin(move(plan->children[0]));
		return plan;
	case LogicalOperatorType::LOGICAL_RECURSIVE_CTE: {
		throw ParserException("Recursive CTEs not supported in correlated subquery");
	}
	default:
		throw InternalException("Logical operator type \"%s\" for dependent join", LogicalOperatorToString(plan->type));
	}
}

} // namespace duckdb
