// See https://raw.githubusercontent.com/duckdb/duckdb/master/LICENSE for licensing information

#include "duckdb.hpp"
#include "duckdb-internal.hpp"
#ifndef DUCKDB_AMALGAMATION
#error header mismatch
#endif




namespace duckdb {

bool ClientContextFileOpener::TryGetCurrentSetting(const string &key, Value &result) {
	return context.TryGetCurrentSetting(key, result);
}

} // namespace duckdb










namespace duckdb {

ClientData::ClientData(ClientContext &context) : catalog_search_path(make_unique<CatalogSearchPath>(context)) {
	profiler = make_shared<QueryProfiler>(context);
	query_profiler_history = make_unique<QueryProfilerHistory>();
	temporary_objects = make_shared<SchemaCatalogEntry>(&Catalog::GetCatalog(context), TEMP_SCHEMA, true);
	random_engine = make_unique<RandomEngine>();
	file_opener = make_unique<ClientContextFileOpener>(context);
}
ClientData::~ClientData() {
}

ClientData &ClientData::Get(ClientContext &context) {
	return *context.client_data;
}

RandomEngine &RandomEngine::Get(ClientContext &context) {
	return *ClientData::Get(context).random_engine;
}

} // namespace duckdb







namespace duckdb {

PreservedError ClientContext::VerifyQuery(ClientContextLock &lock, const string &query,
                                          unique_ptr<SQLStatement> statement) {
	D_ASSERT(statement->type == StatementType::SELECT_STATEMENT);
	// Aggressive query verification

	// The purpose of this function is to test correctness of otherwise hard to test features:
	// Copy() of statements and expressions
	// Serialize()/Deserialize() of expressions
	// Hash() of expressions
	// Equality() of statements and expressions
	// ToString() of statements and expressions
	// Correctness of plans both with and without optimizers

	const auto &stmt = *statement;
	vector<unique_ptr<StatementVerifier>> statement_verifiers;
	unique_ptr<StatementVerifier> prepared_statement_verifier;
	if (config.query_verification_enabled) {
		statement_verifiers.emplace_back(StatementVerifier::Create(VerificationType::COPIED, stmt));
		statement_verifiers.emplace_back(StatementVerifier::Create(VerificationType::DESERIALIZED, stmt));
		statement_verifiers.emplace_back(StatementVerifier::Create(VerificationType::UNOPTIMIZED, stmt));
		prepared_statement_verifier = StatementVerifier::Create(VerificationType::PREPARED, stmt);
	}
	if (config.verify_external) {
		statement_verifiers.emplace_back(StatementVerifier::Create(VerificationType::EXTERNAL, stmt));
	}

	auto original = make_unique<StatementVerifier>(move(statement));
	for (auto &verifier : statement_verifiers) {
		original->CheckExpressions(*verifier);
	}
	original->CheckExpressions();

	// See below
	auto statement_copy_for_explain = stmt.Copy();

	// Save settings
	bool optimizer_enabled = config.enable_optimizer;
	bool profiling_is_enabled = config.enable_profiler;
	bool force_external = config.force_external;

	// Disable profiling if it is enabled
	if (profiling_is_enabled) {
		config.enable_profiler = false;
	}

	// Execute the original statement
	bool any_failed = original->Run(*this, query, [&](const string &q, unique_ptr<SQLStatement> s) {
		return RunStatementInternal(lock, q, move(s), false, false);
	});
	if (!any_failed) {
		statement_verifiers.emplace_back(
		    StatementVerifier::Create(VerificationType::PARSED, *statement_copy_for_explain));
	}
	// Execute the verifiers
	for (auto &verifier : statement_verifiers) {
		bool failed = verifier->Run(*this, query, [&](const string &q, unique_ptr<SQLStatement> s) {
			return RunStatementInternal(lock, q, move(s), false, false);
		});
		any_failed = any_failed || failed;
	}

	if (!any_failed && prepared_statement_verifier) {
		// If none failed, we execute the prepared statement verifier
		bool failed = prepared_statement_verifier->Run(*this, query, [&](const string &q, unique_ptr<SQLStatement> s) {
			return RunStatementInternal(lock, q, move(s), false, false);
		});
		if (!failed) {
			// PreparedStatementVerifier fails if it runs into a ParameterNotAllowedException, which is OK
			statement_verifiers.push_back(move(prepared_statement_verifier));
		}
	} else {
		if (ValidChecker::IsInvalidated(*db)) {
			return original->materialized_result->GetErrorObject();
		}
	}

	// Restore config setting
	config.enable_optimizer = optimizer_enabled;
	config.force_external = force_external;

	// Check explain, only if q does not already contain EXPLAIN
	if (original->materialized_result->success) {
		auto explain_q = "EXPLAIN " + query;
		auto explain_stmt = make_unique<ExplainStatement>(move(statement_copy_for_explain));
		try {
			RunStatementInternal(lock, explain_q, move(explain_stmt), false, false);
		} catch (std::exception &ex) { // LCOV_EXCL_START
			interrupted = false;
			return PreservedError("EXPLAIN failed but query did not (" + string(ex.what()) + ")");
		} // LCOV_EXCL_STOP

#ifdef DUCKDB_VERIFY_BOX_RENDERER
		// this is pretty slow, so disabled by default
		// test the box renderer on the result
		// we mostly care that this does not crash
		RandomEngine random;
		BoxRendererConfig config;
		// test with a random width
		config.max_width = random.NextRandomInteger() % 500;
		BoxRenderer renderer(config);
		renderer.ToString(*this, original->materialized_result->names, original->materialized_result->Collection());
#endif
	}

	// Restore profiler setting
	if (profiling_is_enabled) {
		config.enable_profiler = true;
	}

	// Now compare the results
	// The results of all runs should be identical
	for (auto &verifier : statement_verifiers) {
		auto result = original->CompareResults(*verifier);
		if (!result.empty()) {
			return PreservedError(result);
		}
	}

	return PreservedError();
}

} // namespace duckdb






namespace duckdb {

#define DUCKDB_GLOBAL(_PARAM)                                                                                          \
	{ _PARAM::Name, _PARAM::Description, _PARAM::InputType, _PARAM::SetGlobal, nullptr, _PARAM::GetSetting }
#define DUCKDB_GLOBAL_ALIAS(_ALIAS, _PARAM)                                                                            \
	{ _ALIAS, _PARAM::Description, _PARAM::InputType, _PARAM::SetGlobal, nullptr, _PARAM::GetSetting }

#define DUCKDB_LOCAL(_PARAM)                                                                                           \
	{ _PARAM::Name, _PARAM::Description, _PARAM::InputType, nullptr, _PARAM::SetLocal, _PARAM::GetSetting }
#define DUCKDB_LOCAL_ALIAS(_ALIAS, _PARAM)                                                                             \
	{ _ALIAS, _PARAM::Description, _PARAM::InputType, nullptr, _PARAM::SetLocal, _PARAM::GetSetting }

#define DUCKDB_GLOBAL_LOCAL(_PARAM)                                                                                    \
	{ _PARAM::Name, _PARAM::Description, _PARAM::InputType, _PARAM::SetGlobal, _PARAM::SetLocal, _PARAM::GetSetting }
#define DUCKDB_GLOBAL_LOCAL_ALIAS(_ALIAS, _PARAM)                                                                      \
	{ _ALIAS, _PARAM::Description, _PARAM::InputType, _PARAM::SetGlobal, _PARAM::SetLocal, _PARAM::GetSetting }
#define FINAL_SETTING                                                                                                  \
	{ nullptr, nullptr, LogicalTypeId::INVALID, nullptr, nullptr, nullptr }

static ConfigurationOption internal_options[] = {DUCKDB_GLOBAL(AccessModeSetting),
                                                 DUCKDB_GLOBAL(CheckpointThresholdSetting),
                                                 DUCKDB_GLOBAL(DebugCheckpointAbort),
                                                 DUCKDB_LOCAL(DebugForceExternal),
                                                 DUCKDB_LOCAL(DebugForceNoCrossProduct),
                                                 DUCKDB_GLOBAL(DebugWindowMode),
                                                 DUCKDB_GLOBAL_LOCAL(DefaultCollationSetting),
                                                 DUCKDB_GLOBAL(DefaultOrderSetting),
                                                 DUCKDB_GLOBAL(DefaultNullOrderSetting),
                                                 DUCKDB_GLOBAL(DisabledOptimizersSetting),
                                                 DUCKDB_GLOBAL(EnableExternalAccessSetting),
                                                 DUCKDB_GLOBAL(EnableFSSTVectors),
                                                 DUCKDB_GLOBAL(AllowUnsignedExtensionsSetting),
                                                 DUCKDB_GLOBAL(EnableObjectCacheSetting),
                                                 DUCKDB_LOCAL(EnableProfilingSetting),
                                                 DUCKDB_LOCAL(EnableProgressBarSetting),
                                                 DUCKDB_GLOBAL(ExperimentalParallelCSVSetting),
                                                 DUCKDB_LOCAL(ExplainOutputSetting),
                                                 DUCKDB_GLOBAL(ExternalThreadsSetting),
                                                 DUCKDB_LOCAL(FileSearchPathSetting),
                                                 DUCKDB_GLOBAL(ForceCompressionSetting),
                                                 DUCKDB_LOCAL(HomeDirectorySetting),
                                                 DUCKDB_LOCAL(LogQueryPathSetting),
                                                 DUCKDB_LOCAL(MaximumExpressionDepthSetting),
                                                 DUCKDB_GLOBAL(MaximumMemorySetting),
                                                 DUCKDB_GLOBAL_ALIAS("memory_limit", MaximumMemorySetting),
                                                 DUCKDB_GLOBAL_ALIAS("null_order", DefaultNullOrderSetting),
                                                 DUCKDB_GLOBAL(PasswordSetting),
                                                 DUCKDB_LOCAL(PerfectHashThresholdSetting),
                                                 DUCKDB_LOCAL(PreserveIdentifierCase),
                                                 DUCKDB_GLOBAL(PreserveInsertionOrder),
                                                 DUCKDB_LOCAL(ProfilerHistorySize),
                                                 DUCKDB_LOCAL(ProfileOutputSetting),
                                                 DUCKDB_LOCAL(ProfilingModeSetting),
                                                 DUCKDB_LOCAL_ALIAS("profiling_output", ProfileOutputSetting),
                                                 DUCKDB_LOCAL(ProgressBarTimeSetting),
                                                 DUCKDB_LOCAL(SchemaSetting),
                                                 DUCKDB_LOCAL(SearchPathSetting),
                                                 DUCKDB_GLOBAL(TempDirectorySetting),
                                                 DUCKDB_GLOBAL(ThreadsSetting),
                                                 DUCKDB_GLOBAL(UsernameSetting),
                                                 DUCKDB_GLOBAL_ALIAS("user", UsernameSetting),
                                                 DUCKDB_GLOBAL_ALIAS("wal_autocheckpoint", CheckpointThresholdSetting),
                                                 DUCKDB_GLOBAL_ALIAS("worker_threads", ThreadsSetting),
                                                 FINAL_SETTING};

vector<ConfigurationOption> DBConfig::GetOptions() {
	vector<ConfigurationOption> options;
	for (idx_t index = 0; internal_options[index].name; index++) {
		options.push_back(internal_options[index]);
	}
	return options;
}

idx_t DBConfig::GetOptionCount() {
	idx_t count = 0;
	for (idx_t index = 0; internal_options[index].name; index++) {
		count++;
	}
	return count;
}

vector<std::string> DBConfig::GetOptionNames() {
	vector<string> names;
	for (idx_t i = 0, option_count = DBConfig::GetOptionCount(); i < option_count; i++) {
		names.emplace_back(DBConfig::GetOptionByIndex(i)->name);
	}
	return names;
}

ConfigurationOption *DBConfig::GetOptionByIndex(idx_t target_index) {
	for (idx_t index = 0; internal_options[index].name; index++) {
		if (index == target_index) {
			return internal_options + index;
		}
	}
	return nullptr;
}

ConfigurationOption *DBConfig::GetOptionByName(const string &name) {
	auto lname = StringUtil::Lower(name);
	for (idx_t index = 0; internal_options[index].name; index++) {
		D_ASSERT(StringUtil::Lower(internal_options[index].name) == string(internal_options[index].name));
		if (internal_options[index].name == lname) {
			return internal_options + index;
		}
	}
	return nullptr;
}

void DBConfig::SetOption(const ConfigurationOption &option, const Value &value) {
	SetOption(nullptr, option, value);
}

void DBConfig::SetOption(DatabaseInstance *db, const ConfigurationOption &option, const Value &value) {
	lock_guard<mutex> l(config_lock);
	if (!option.set_global) {
		throw InternalException("Could not set option \"%s\" as a global option", option.name);
	}
	Value input = value.DefaultCastAs(option.parameter_type);
	option.set_global(db, *this, input);
}

void DBConfig::SetOption(const string &name, Value value) {
	lock_guard<mutex> l(config_lock);
	options.set_variables[name] = move(value);
}

void DBConfig::AddExtensionOption(string name, string description, LogicalType parameter,
                                  set_option_callback_t function) {
	extension_parameters.insert(make_pair(move(name), ExtensionOption(move(description), move(parameter), function)));
}

CastFunctionSet &DBConfig::GetCastFunctions() {
	return *cast_functions;
}

idx_t DBConfig::ParseMemoryLimit(const string &arg) {
	if (arg[0] == '-' || arg == "null" || arg == "none") {
		return DConstants::INVALID_INDEX;
	}
	// split based on the number/non-number
	idx_t idx = 0;
	while (StringUtil::CharacterIsSpace(arg[idx])) {
		idx++;
	}
	idx_t num_start = idx;
	while ((arg[idx] >= '0' && arg[idx] <= '9') || arg[idx] == '.' || arg[idx] == 'e' || arg[idx] == 'E' ||
	       arg[idx] == '-') {
		idx++;
	}
	if (idx == num_start) {
		throw ParserException("Memory limit must have a number (e.g. SET memory_limit=1GB");
	}
	string number = arg.substr(num_start, idx - num_start);

	// try to parse the number
	double limit = Cast::Operation<string_t, double>(string_t(number));

	// now parse the memory limit unit (e.g. bytes, gb, etc)
	while (StringUtil::CharacterIsSpace(arg[idx])) {
		idx++;
	}
	idx_t start = idx;
	while (idx < arg.size() && !StringUtil::CharacterIsSpace(arg[idx])) {
		idx++;
	}
	if (limit < 0) {
		// limit < 0, set limit to infinite
		return (idx_t)-1;
	}
	string unit = StringUtil::Lower(arg.substr(start, idx - start));
	idx_t multiplier;
	if (unit == "byte" || unit == "bytes" || unit == "b") {
		multiplier = 1;
	} else if (unit == "kilobyte" || unit == "kilobytes" || unit == "kb" || unit == "k") {
		multiplier = 1000LL;
	} else if (unit == "megabyte" || unit == "megabytes" || unit == "mb" || unit == "m") {
		multiplier = 1000LL * 1000LL;
	} else if (unit == "gigabyte" || unit == "gigabytes" || unit == "gb" || unit == "g") {
		multiplier = 1000LL * 1000LL * 1000LL;
	} else if (unit == "terabyte" || unit == "terabytes" || unit == "tb" || unit == "t") {
		multiplier = 1000LL * 1000LL * 1000LL * 1000LL;
	} else {
		throw ParserException("Unknown unit for memory_limit: %s (expected: b, mb, gb or tb)", unit);
	}
	return (idx_t)multiplier * limit;
}

// Right now we only really care about access mode when comparing DBConfigs
bool DBConfigOptions::operator==(const DBConfigOptions &other) const {
	return other.access_mode == access_mode;
}

bool DBConfig::operator==(const DBConfig &other) {
	return other.options == options;
}

bool DBConfig::operator!=(const DBConfig &other) {
	return !(other.options == options);
}

} // namespace duckdb



















namespace duckdb {

Connection::Connection(DatabaseInstance &database) : context(make_shared<ClientContext>(database.shared_from_this())) {
	ConnectionManager::Get(database).AddConnection(*context);
#ifdef DEBUG
	EnableProfiling();
	context->config.emit_profiler_output = false;
#endif
}

Connection::Connection(DuckDB &database) : Connection(*database.instance) {
}

Connection::~Connection() {
	ConnectionManager::Get(*context->db).RemoveConnection(*context);
}

string Connection::GetProfilingInformation(ProfilerPrintFormat format) {
	auto &profiler = QueryProfiler::Get(*context);
	if (format == ProfilerPrintFormat::JSON) {
		return profiler.ToJSON();
	} else {
		return profiler.QueryTreeToString();
	}
}

void Connection::Interrupt() {
	context->Interrupt();
}

void Connection::EnableProfiling() {
	context->EnableProfiling();
}

void Connection::DisableProfiling() {
	context->DisableProfiling();
}

void Connection::EnableQueryVerification() {
	ClientConfig::GetConfig(*context).query_verification_enabled = true;
}

void Connection::DisableQueryVerification() {
	ClientConfig::GetConfig(*context).query_verification_enabled = false;
}

void Connection::ForceParallelism() {
	ClientConfig::GetConfig(*context).verify_parallelism = true;
}

unique_ptr<QueryResult> Connection::SendQuery(const string &query) {
	return context->Query(query, true);
}

unique_ptr<MaterializedQueryResult> Connection::Query(const string &query) {
	auto result = context->Query(query, false);
	D_ASSERT(result->type == QueryResultType::MATERIALIZED_RESULT);
	return unique_ptr_cast<QueryResult, MaterializedQueryResult>(move(result));
}

unique_ptr<MaterializedQueryResult> Connection::Query(unique_ptr<SQLStatement> statement) {
	auto result = context->Query(move(statement), false);
	D_ASSERT(result->type == QueryResultType::MATERIALIZED_RESULT);
	return unique_ptr_cast<QueryResult, MaterializedQueryResult>(move(result));
}

unique_ptr<PendingQueryResult> Connection::PendingQuery(const string &query, bool allow_stream_result) {
	return context->PendingQuery(query, allow_stream_result);
}

unique_ptr<PendingQueryResult> Connection::PendingQuery(unique_ptr<SQLStatement> statement, bool allow_stream_result) {
	return context->PendingQuery(move(statement), allow_stream_result);
}

unique_ptr<PreparedStatement> Connection::Prepare(const string &query) {
	return context->Prepare(query);
}

unique_ptr<PreparedStatement> Connection::Prepare(unique_ptr<SQLStatement> statement) {
	return context->Prepare(move(statement));
}

unique_ptr<QueryResult> Connection::QueryParamsRecursive(const string &query, vector<Value> &values) {
	auto statement = Prepare(query);
	if (statement->HasError()) {
		return make_unique<MaterializedQueryResult>(statement->error);
	}
	return statement->Execute(values, false);
}

unique_ptr<TableDescription> Connection::TableInfo(const string &table_name) {
	return TableInfo(DEFAULT_SCHEMA, table_name);
}

unique_ptr<TableDescription> Connection::TableInfo(const string &schema_name, const string &table_name) {
	return context->TableInfo(schema_name, table_name);
}

vector<unique_ptr<SQLStatement>> Connection::ExtractStatements(const string &query) {
	return context->ParseStatements(query);
}

unique_ptr<LogicalOperator> Connection::ExtractPlan(const string &query) {
	return context->ExtractPlan(query);
}

void Connection::Append(TableDescription &description, DataChunk &chunk) {
	if (chunk.size() == 0) {
		return;
	}
	ColumnDataCollection collection(Allocator::Get(*context), chunk.GetTypes());
	collection.Append(chunk);
	Append(description, collection);
}

void Connection::Append(TableDescription &description, ColumnDataCollection &collection) {
	context->Append(description, collection);
}

shared_ptr<Relation> Connection::Table(const string &table_name) {
	return Table(DEFAULT_SCHEMA, table_name);
}

shared_ptr<Relation> Connection::Table(const string &schema_name, const string &table_name) {
	auto table_info = TableInfo(schema_name, table_name);
	if (!table_info) {
		throw Exception("Table does not exist!");
	}
	return make_shared<TableRelation>(context, move(table_info));
}

shared_ptr<Relation> Connection::View(const string &tname) {
	return View(DEFAULT_SCHEMA, tname);
}

shared_ptr<Relation> Connection::View(const string &schema_name, const string &table_name) {
	return make_shared<ViewRelation>(context, schema_name, table_name);
}

shared_ptr<Relation> Connection::TableFunction(const string &fname) {
	vector<Value> values;
	named_parameter_map_t named_parameters;
	return TableFunction(fname, values, named_parameters);
}

shared_ptr<Relation> Connection::TableFunction(const string &fname, const vector<Value> &values,
                                               const named_parameter_map_t &named_parameters) {
	return make_shared<TableFunctionRelation>(context, fname, values, named_parameters);
}

shared_ptr<Relation> Connection::TableFunction(const string &fname, const vector<Value> &values) {
	return make_shared<TableFunctionRelation>(context, fname, values);
}

shared_ptr<Relation> Connection::Values(const vector<vector<Value>> &values) {
	vector<string> column_names;
	return Values(values, column_names);
}

shared_ptr<Relation> Connection::Values(const vector<vector<Value>> &values, const vector<string> &column_names,
                                        const string &alias) {
	return make_shared<ValueRelation>(context, values, column_names, alias);
}

shared_ptr<Relation> Connection::Values(const string &values) {
	vector<string> column_names;
	return Values(values, column_names);
}

shared_ptr<Relation> Connection::Values(const string &values, const vector<string> &column_names, const string &alias) {
	return make_shared<ValueRelation>(context, values, column_names, alias);
}

shared_ptr<Relation> Connection::ReadCSV(const string &csv_file) {
	BufferedCSVReaderOptions options;
	options.file_path = csv_file;
	options.auto_detect = true;
	BufferedCSVReader reader(*context, options);
	vector<ColumnDefinition> column_list;
	for (idx_t i = 0; i < reader.sql_types.size(); i++) {
		column_list.emplace_back(reader.col_names[i], reader.sql_types[i]);
	}
	return make_shared<ReadCSVRelation>(context, csv_file, move(column_list), true);
}

shared_ptr<Relation> Connection::ReadCSV(const string &csv_file, const vector<string> &columns) {
	// parse columns
	vector<ColumnDefinition> column_list;
	for (auto &column : columns) {
		auto col_list = Parser::ParseColumnList(column, context->GetParserOptions());
		if (col_list.LogicalColumnCount() != 1) {
			throw ParserException("Expected a single column definition");
		}
		column_list.push_back(move(col_list.GetColumnMutable(LogicalIndex(0))));
	}
	return make_shared<ReadCSVRelation>(context, csv_file, move(column_list));
}

unordered_set<string> Connection::GetTableNames(const string &query) {
	return context->GetTableNames(query);
}

shared_ptr<Relation> Connection::RelationFromQuery(const string &query, const string &alias, const string &error) {
	return RelationFromQuery(QueryRelation::ParseStatement(*context, query, error), alias);
}

shared_ptr<Relation> Connection::RelationFromQuery(unique_ptr<SelectStatement> select_stmt, const string &alias) {
	return make_shared<QueryRelation>(context, move(select_stmt), alias);
}

void Connection::BeginTransaction() {
	auto result = Query("BEGIN TRANSACTION");
	if (result->HasError()) {
		result->ThrowError();
	}
}

void Connection::Commit() {
	auto result = Query("COMMIT");
	if (result->HasError()) {
		result->ThrowError();
	}
}

void Connection::Rollback() {
	auto result = Query("ROLLBACK");
	if (result->HasError()) {
		result->ThrowError();
	}
}

void Connection::SetAutoCommit(bool auto_commit) {
	context->transaction.SetAutoCommit(auto_commit);
}

bool Connection::IsAutoCommit() {
	return context->transaction.IsAutoCommit();
}
bool Connection::HasActiveTransaction() {
	return context->transaction.HasActiveTransaction();
}

} // namespace duckdb
















#ifndef DUCKDB_NO_THREADS

#endif

namespace duckdb {

DBConfig::DBConfig() {
	compression_functions = make_unique<CompressionFunctionSet>();
	replacement_opens.push_back(ExtensionPrefixReplacementOpen());
	cast_functions = make_unique<CastFunctionSet>();
	error_manager = make_unique<ErrorManager>();
}

DBConfig::DBConfig(std::unordered_map<string, string> &config_dict, bool read_only) {
	compression_functions = make_unique<CompressionFunctionSet>();
	if (read_only) {
		options.access_mode = AccessMode::READ_ONLY;
	}
	for (auto &kv : config_dict) {
		string key = kv.first;
		string val = kv.second;
		auto config_property = DBConfig::GetOptionByName(key);
		if (!config_property) {
			throw InvalidInputException("Unrecognized configuration property \"%s\"", key);
		}
		auto opt_val = Value(val);
		DBConfig::SetOption(*config_property, opt_val);
	}
}

DBConfig::~DBConfig() {
}

DatabaseInstance::DatabaseInstance() {
}

DatabaseInstance::~DatabaseInstance() {
	if (Exception::UncaughtException()) {
		return;
	}

	// shutting down: attempt to checkpoint the database
	// but only if we are not cleaning up as part of an exception unwind
	try {
		auto &storage = StorageManager::GetStorageManager(*this);
		if (!storage.InMemory()) {
			auto &config = storage.db.config;
			if (!config.options.checkpoint_on_shutdown) {
				return;
			}
			storage.CreateCheckpoint(true);
		}
	} catch (...) {
	}
}

BufferManager &BufferManager::GetBufferManager(DatabaseInstance &db) {
	return *db.GetStorageManager().buffer_manager;
}

DatabaseInstance &DatabaseInstance::GetDatabase(ClientContext &context) {
	return *context.db;
}

StorageManager &StorageManager::GetStorageManager(DatabaseInstance &db) {
	return db.GetStorageManager();
}

Catalog &Catalog::GetCatalog(DatabaseInstance &db) {
	return db.GetCatalog();
}

FileSystem &FileSystem::GetFileSystem(DatabaseInstance &db) {
	return db.GetFileSystem();
}

DBConfig &DBConfig::GetConfig(DatabaseInstance &db) {
	return db.config;
}

ClientConfig &ClientConfig::GetConfig(ClientContext &context) {
	return context.config;
}

const DBConfig &DBConfig::GetConfig(const DatabaseInstance &db) {
	return db.config;
}

const ClientConfig &ClientConfig::GetConfig(const ClientContext &context) {
	return context.config;
}

TransactionManager &TransactionManager::Get(ClientContext &context) {
	return TransactionManager::Get(DatabaseInstance::GetDatabase(context));
}

TransactionManager &TransactionManager::Get(DatabaseInstance &db) {
	return db.GetTransactionManager();
}

ConnectionManager &ConnectionManager::Get(DatabaseInstance &db) {
	return db.GetConnectionManager();
}

ConnectionManager &ConnectionManager::Get(ClientContext &context) {
	return ConnectionManager::Get(DatabaseInstance::GetDatabase(context));
}

void DatabaseInstance::Initialize(const char *database_path, DBConfig *user_config) {
	DBConfig default_config;
	DBConfig *config_ptr = &default_config;
	if (user_config) {
		config_ptr = user_config;
	}

	if (config_ptr->options.temporary_directory.empty() && database_path) {
		// no directory specified: use default temp path
		config_ptr->options.temporary_directory = string(database_path) + ".tmp";

		// special treatment for in-memory mode
		if (strcmp(database_path, ":memory:") == 0) {
			config_ptr->options.temporary_directory = ".tmp";
		}
	}

	if (database_path) {
		config_ptr->options.database_path = database_path;
	} else {
		config_ptr->options.database_path.clear();
	}

	for (auto &open : config_ptr->replacement_opens) {
		if (open.pre_func) {
			open.data = open.pre_func(*config_ptr, open.static_data.get());
			if (open.data) {
				break;
			}
		}
	}
	Configure(*config_ptr);

	if (user_config && !user_config->options.use_temporary_directory) {
		// temporary directories explicitly disabled
		config.options.temporary_directory = string();
	}

	// TODO: Support an extension here, to generate different storage managers
	// depending on the DB path structure/prefix.
	const string dbPath = config.options.database_path;
	storage = make_unique<SingleFileStorageManager>(*this, dbPath, config.options.access_mode == AccessMode::READ_ONLY);

	catalog = make_unique<Catalog>(*this);
	transaction_manager = make_unique<TransactionManager>(*this);
	scheduler = make_unique<TaskScheduler>(*this);
	object_cache = make_unique<ObjectCache>();
	connection_manager = make_unique<ConnectionManager>();

	// initialize the database
	storage->Initialize();

	// only increase thread count after storage init because we get races on catalog otherwise
	scheduler->SetThreads(config.options.maximum_threads);

	for (auto &open : config.replacement_opens) {
		if (open.post_func && open.data) {
			open.post_func(*this, open.data.get());
			break;
		}
	}
}

DuckDB::DuckDB(const char *path, DBConfig *new_config) : instance(make_shared<DatabaseInstance>()) {
	instance->Initialize(path, new_config);
	if (instance->config.options.load_extensions) {
		ExtensionHelper::LoadAllExtensions(*this);
	}
}

DuckDB::DuckDB(const string &path, DBConfig *config) : DuckDB(path.c_str(), config) {
}

DuckDB::DuckDB(DatabaseInstance &instance_p) : instance(instance_p.shared_from_this()) {
}

DuckDB::~DuckDB() {
}

StorageManager &DatabaseInstance::GetStorageManager() {
	return *storage;
}

Catalog &DatabaseInstance::GetCatalog() {
	return *catalog;
}

TransactionManager &DatabaseInstance::GetTransactionManager() {
	return *transaction_manager;
}

TaskScheduler &DatabaseInstance::GetScheduler() {
	return *scheduler;
}

ObjectCache &DatabaseInstance::GetObjectCache() {
	return *object_cache;
}

FileSystem &DatabaseInstance::GetFileSystem() {
	return *config.file_system;
}

ConnectionManager &DatabaseInstance::GetConnectionManager() {
	return *connection_manager;
}

FileSystem &DuckDB::GetFileSystem() {
	return instance->GetFileSystem();
}

Allocator &Allocator::Get(ClientContext &context) {
	return Allocator::Get(*context.db);
}

Allocator &Allocator::Get(DatabaseInstance &db) {
	return *db.config.allocator;
}

void DatabaseInstance::Configure(DBConfig &new_config) {
	config.options = new_config.options;
	if (config.options.access_mode == AccessMode::UNDEFINED) {
		config.options.access_mode = AccessMode::READ_WRITE;
	}
	if (new_config.file_system) {
		config.file_system = move(new_config.file_system);
	} else {
		config.file_system = make_unique<VirtualFileSystem>();
	}
	if (config.options.maximum_memory == (idx_t)-1) {
		auto memory = FileSystem::GetAvailableMemory();
		if (memory != DConstants::INVALID_INDEX) {
			config.options.maximum_memory = memory * 8 / 10;
		}
	}
	if (new_config.options.maximum_threads == (idx_t)-1) {
#ifndef DUCKDB_NO_THREADS
		config.options.maximum_threads = std::thread::hardware_concurrency();
#else
		config.options.maximum_threads = 1;
#endif
	}
	config.allocator = move(new_config.allocator);
	if (!config.allocator) {
		config.allocator = make_unique<Allocator>();
	}
	config.replacement_scans = move(new_config.replacement_scans);
	config.replacement_opens = move(new_config.replacement_opens);
	config.parser_extensions = move(new_config.parser_extensions);
	config.error_manager = move(new_config.error_manager);
	if (!config.error_manager) {
		config.error_manager = make_unique<ErrorManager>();
	}
	if (!config.default_allocator) {
		config.default_allocator = Allocator::DefaultAllocatorReference();
	}
}

DBConfig &DBConfig::GetConfig(ClientContext &context) {
	return context.db->config;
}

const DBConfig &DBConfig::GetConfig(const ClientContext &context) {
	return context.db->config;
}

idx_t DatabaseInstance::NumberOfThreads() {
	return scheduler->NumberOfThreads();
}

const unordered_set<std::string> &DatabaseInstance::LoadedExtensions() {
	return loaded_extensions;
}

idx_t DuckDB::NumberOfThreads() {
	return instance->NumberOfThreads();
}

bool DuckDB::ExtensionIsLoaded(const std::string &name) {
	return instance->loaded_extensions.find(name) != instance->loaded_extensions.end();
}
void DatabaseInstance::SetExtensionLoaded(const std::string &name) {
	loaded_extensions.insert(name);
}

bool DatabaseInstance::TryGetCurrentSetting(const std::string &key, Value &result) {
	// check the session values
	auto &db_config = DBConfig::GetConfig(*this);
	const auto &global_config_map = db_config.options.set_variables;

	auto global_value = global_config_map.find(key);
	bool found_global_value = global_value != global_config_map.end();
	if (!found_global_value) {
		return false;
	}
	result = global_value->second;
	return true;
}

string ClientConfig::ExtractTimezone() const {
	auto entry = set_variables.find("TimeZone");
	if (entry == set_variables.end()) {
		return "UTC";
	} else {
		return entry->second.GetValue<std::string>();
	}
}

ValidChecker &DatabaseInstance::GetValidChecker() {
	return db_validity;
}

ValidChecker &ValidChecker::Get(DatabaseInstance &db) {
	return db.GetValidChecker();
}

} // namespace duckdb


namespace duckdb {

string GetDBAbsolutePath(const string &database) {
	if (database.empty()) {
		return ":memory:";
	}
	if (database.rfind(":memory:", 0) == 0) {
		// this is a memory db, just return it.
		return database;
	}
	if (FileSystem::IsPathAbsolute(database)) {
		return database;
	}
	return FileSystem::JoinPath(FileSystem::GetWorkingDirectory(), database);
}

shared_ptr<DuckDB> DBInstanceCache::GetInstanceInternal(const string &database, const DBConfig &config) {
	shared_ptr<DuckDB> db_instance;
	auto abs_database_path = GetDBAbsolutePath(database);
	if (db_instances.find(abs_database_path) != db_instances.end()) {
		db_instance = db_instances[abs_database_path].lock();
		if (db_instance) {
			if (db_instance->instance->config != config) {
				throw duckdb::Exception(ExceptionType::CONNECTION,
				                        "Can't open a connection to same database file with a different configuration "
				                        "than existing connections");
			}
		} else {
			// clean-up
			db_instances.erase(abs_database_path);
		}
	}
	return db_instance;
}

shared_ptr<DuckDB> DBInstanceCache::GetInstance(const string &database, const DBConfig &config) {
	lock_guard<mutex> l(cache_lock);
	return GetInstanceInternal(database, config);
}

shared_ptr<DuckDB> DBInstanceCache::CreateInstanceInternal(const string &database, DBConfig &config,
                                                           bool cache_instance) {
	auto abs_database_path = GetDBAbsolutePath(database);
	if (db_instances.find(abs_database_path) != db_instances.end()) {
		throw duckdb::Exception(ExceptionType::CONNECTION,
		                        "Instance with path: " + abs_database_path + " already exists.");
	}
	// Creates new instance
	string instance_path = abs_database_path;
	if (abs_database_path.rfind(":memory:", 0) == 0) {
		instance_path = ":memory:";
	}
	auto db_instance = make_shared<DuckDB>(instance_path, &config);
	if (cache_instance) {
		db_instances[abs_database_path] = db_instance;
	}
	return db_instance;
}

shared_ptr<DuckDB> DBInstanceCache::CreateInstance(const string &database, DBConfig &config, bool cache_instance) {
	lock_guard<mutex> l(cache_lock);
	return CreateInstanceInternal(database, config, cache_instance);
}

shared_ptr<DuckDB> DBInstanceCache::GetOrCreateInstance(const string &database, DBConfig &config_dict,
                                                        bool cache_instance) {
	lock_guard<mutex> l(cache_lock);
	if (cache_instance) {
		auto instance = GetInstanceInternal(database, config_dict);
		if (instance) {
			return instance;
		}
	}
	return CreateInstanceInternal(database, config_dict, cache_instance);
}

} // namespace duckdb




namespace duckdb {

struct DefaultError {
	ErrorType type;
	const char *error;
};

static DefaultError internal_errors[] = {
    {ErrorType::UNSIGNED_EXTENSION,
     "Extension \"%s\" could not be loaded because its signature is either missing or invalid and unsigned extensions "
     "are disabled by configuration (allow_unsigned_extensions)"},
    {ErrorType::INVALIDATED_TRANSACTION, "Current transaction is aborted (please ROLLBACK)"},
    {ErrorType::INVALIDATED_DATABASE, "Failed: database has been invalidated because of a previous fatal error. The "
                                      "database must be restarted prior to being used again.\nOriginal error: \"%s\""},
    {ErrorType::INVALID, nullptr}};

string ErrorManager::FormatExceptionRecursive(ErrorType error_type, vector<ExceptionFormatValue> &values) {
	if (error_type >= ErrorType::ERROR_COUNT) {
		throw InternalException("Invalid error type passed to ErrorManager::FormatError");
	}
	auto entry = custom_errors.find(error_type);
	string error;
	if (entry == custom_errors.end()) {
		// error was not overwritten
		error = internal_errors[int(error_type)].error;
	} else {
		// error was overwritten
		error = entry->second;
	}
	return ExceptionFormatValue::Format(error, values);
}

string ErrorManager::InvalidUnicodeError(const string &input, const string &context) {
	UnicodeInvalidReason reason;
	size_t pos;
	auto unicode = Utf8Proc::Analyze((const char *)input.c_str(), input.size(), &reason, &pos);
	if (unicode != UnicodeType::INVALID) {
		return "Invalid unicode error thrown but no invalid unicode detected in " + context;
	}
	string base_message;
	switch (reason) {
	case UnicodeInvalidReason::BYTE_MISMATCH:
		base_message = "Invalid unicode (byte sequence mismatch)";
		break;
	case UnicodeInvalidReason::INVALID_UNICODE:
		base_message = "Invalid unicode";
		break;
	default:
		break;
	}
	return base_message + " detected in " + context;
}

void ErrorManager::AddCustomError(ErrorType type, string new_error) {
	custom_errors.insert(make_pair(type, move(new_error)));
}

ErrorManager &ErrorManager::Get(ClientContext &context) {
	return *DBConfig::GetConfig(context).error_manager;
}

ErrorManager &ErrorManager::Get(DatabaseInstance &context) {
	return *DBConfig::GetConfig(context).error_manager;
}

} // namespace duckdb


namespace duckdb {

static ExtensionAlias internal_aliases[] = {{"http", "httpfs"}, // httpfs
                                            {"https", "httpfs"},
                                            {"s3", "httpfs"},
                                            {"postgres", "postgres_scanner"}, // postgres
                                            {"sqlite", "sqlite_scanner"},     // sqlite
                                            {"sqlite3", "sqlite_scanner"},
                                            {nullptr, nullptr}};

idx_t ExtensionHelper::ExtensionAliasCount() {
	idx_t index;
	for (index = 0; internal_aliases[index].alias != nullptr; index++) {
	}
	return index;
}

ExtensionAlias ExtensionHelper::GetExtensionAlias(idx_t index) {
	D_ASSERT(index < ExtensionAliasCount());
	return internal_aliases[index];
}

string ExtensionHelper::ApplyExtensionAlias(string extension_name) {
	for (idx_t index = 0; internal_aliases[index].alias; index++) {
		if (extension_name == internal_aliases[index].alias) {
			return internal_aliases[index].extension;
		}
	}
	return extension_name;
}

} // namespace duckdb








#if defined(BUILD_ICU_EXTENSION) && !defined(DISABLE_BUILTIN_EXTENSIONS)
#define ICU_STATICALLY_LOADED true
#include "icu-extension.hpp"
#else
#define ICU_STATICALLY_LOADED false
#endif

#if defined(BUILD_PARQUET_EXTENSION) && !defined(DISABLE_BUILTIN_EXTENSIONS)
#define PARQUET_STATICALLY_LOADED true
#include "parquet-extension.hpp"
#else
#define PARQUET_STATICALLY_LOADED false
#endif

#if defined(BUILD_TPCH_EXTENSION) && !defined(DISABLE_BUILTIN_EXTENSIONS)
#define TPCH_STATICALLY_LOADED true
#include "tpch-extension.hpp"
#else
#define TPCH_STATICALLY_LOADED false
#endif

#if defined(BUILD_TPCDS_EXTENSION) && !defined(DISABLE_BUILTIN_EXTENSIONS)
#define TPCDS_STATICALLY_LOADED true
#include "tpcds-extension.hpp"
#else
#define TPCDS_STATICALLY_LOADED false
#endif

#if defined(BUILD_FTS_EXTENSION) && !defined(DISABLE_BUILTIN_EXTENSIONS)
#define FTS_STATICALLY_LOADED true
#include "fts-extension.hpp"
#else
#define FTS_STATICALLY_LOADED false
#endif

#if defined(BUILD_HTTPFS_EXTENSION) && !defined(DISABLE_BUILTIN_EXTENSIONS)
#define HTTPFS_STATICALLY_LOADED true
#include "httpfs-extension.hpp"
#else
#define HTTPFS_STATICALLY_LOADED false
#endif

#if defined(BUILD_VISUALIZER_EXTENSION) && !defined(DISABLE_BUILTIN_EXTENSIONS)
#include "visualizer-extension.hpp"
#endif

#if defined(BUILD_JSON_EXTENSION) && !defined(DISABLE_BUILTIN_EXTENSIONS)
#define JSON_STATICALLY_LOADED true
#include "json-extension.hpp"
#else
#define JSON_STATICALLY_LOADED false
#endif

#if defined(BUILD_JEMALLOC_EXTENSION) && !defined(DISABLE_BUILTIN_EXTENSIONS)
#define JEMALLOC_STATICALLY_LOADED true
#include "jemalloc-extension.hpp"
#else
#define JEMALLOC_STATICALLY_LOADED false
#endif

#if defined(BUILD_EXCEL_EXTENSION) && !defined(DISABLE_BUILTIN_EXTENSIONS)
#include "excel-extension.hpp"
#endif

#if defined(BUILD_SQLSMITH_EXTENSION) && !defined(DISABLE_BUILTIN_EXTENSIONS)
#include "sqlsmith-extension.hpp"
#endif

#if defined(BUILD_INET_EXTENSION) && !defined(DISABLE_BUILTIN_EXTENSIONS)
#include "inet-extension.hpp"
#endif

namespace duckdb {

//===--------------------------------------------------------------------===//
// Default Extensions
//===--------------------------------------------------------------------===//
static DefaultExtension internal_extensions[] = {
    {"icu", "Adds support for time zones and collations using the ICU library", ICU_STATICALLY_LOADED},
    {"parquet", "Adds support for reading and writing parquet files", PARQUET_STATICALLY_LOADED},
    {"tpch", "Adds TPC-H data generation and query support", TPCH_STATICALLY_LOADED},
    {"tpcds", "Adds TPC-DS data generation and query support", TPCDS_STATICALLY_LOADED},
    {"fts", "Adds support for Full-Text Search Indexes", FTS_STATICALLY_LOADED},
    {"httpfs", "Adds support for reading and writing files over a HTTP(S) connection", HTTPFS_STATICALLY_LOADED},
    {"json", "Adds support for JSON operations", JSON_STATICALLY_LOADED},
    {"jemalloc", "Overwrites system allocator with JEMalloc", JEMALLOC_STATICALLY_LOADED},
    {"sqlite_scanner", "Adds support for reading SQLite database files", false},
    {"postgres_scanner", "Adds support for reading from a Postgres database", false},
    {"inet", "Adds support for IP-related data types and functions", false},
    {nullptr, nullptr, false}};

idx_t ExtensionHelper::DefaultExtensionCount() {
	idx_t index;
	for (index = 0; internal_extensions[index].name != nullptr; index++) {
	}
	return index;
}

DefaultExtension ExtensionHelper::GetDefaultExtension(idx_t index) {
	D_ASSERT(index < DefaultExtensionCount());
	return internal_extensions[index];
}

//===--------------------------------------------------------------------===//
// Load Statically Compiled Extension
//===--------------------------------------------------------------------===//
void ExtensionHelper::LoadAllExtensions(DuckDB &db) {
	unordered_set<string> extensions {"parquet",    "icu",  "tpch",  "tpcds",    "fts",  "httpfs",
	                                  "visualizer", "json", "excel", "sqlsmith", "inet", "jemalloc"};
	for (auto &ext : extensions) {
		LoadExtensionInternal(db, ext, true);
	}
}

ExtensionLoadResult ExtensionHelper::LoadExtension(DuckDB &db, const std::string &extension) {
	return LoadExtensionInternal(db, extension, false);
}

ExtensionLoadResult ExtensionHelper::LoadExtensionInternal(DuckDB &db, const std::string &extension,
                                                           bool initial_load) {
#ifdef DUCKDB_TEST_REMOTE_INSTALL
	if (!initial_load && StringUtil::Contains(DUCKDB_TEST_REMOTE_INSTALL, extension)) {
		Connection con(db);
		auto result = con.Query("INSTALL " + extension);
		if (result->HasError()) {
			result->Print();
			return ExtensionLoadResult::EXTENSION_UNKNOWN;
		}
		result = con.Query("LOAD " + extension);
		if (result->HasError()) {
			result->Print();
			return ExtensionLoadResult::EXTENSION_UNKNOWN;
		}
		return ExtensionLoadResult::LOADED_EXTENSION;
	}
#endif
	if (extension == "parquet") {
#if PARQUET_STATICALLY_LOADED
		db.LoadExtension<ParquetExtension>();
#else
		// parquet extension required but not build: skip this test
		return ExtensionLoadResult::NOT_LOADED;
#endif
	} else if (extension == "icu") {
#if ICU_STATICALLY_LOADED
		db.LoadExtension<ICUExtension>();
#else
		// icu extension required but not build: skip this test
		return ExtensionLoadResult::NOT_LOADED;
#endif
	} else if (extension == "tpch") {
#if TPCH_STATICALLY_LOADED
		db.LoadExtension<TPCHExtension>();
#else
		// icu extension required but not build: skip this test
		return ExtensionLoadResult::NOT_LOADED;
#endif
	} else if (extension == "tpcds") {
#if TPCDS_STATICALLY_LOADED
		db.LoadExtension<TPCDSExtension>();
#else
		// icu extension required but not build: skip this test
		return ExtensionLoadResult::NOT_LOADED;
#endif
	} else if (extension == "fts") {
#if FTS_STATICALLY_LOADED
		db.LoadExtension<FTSExtension>();
#else
		// fts extension required but not build: skip this test
		return ExtensionLoadResult::NOT_LOADED;
#endif
	} else if (extension == "httpfs") {
#if HTTPFS_STATICALLY_LOADED
		db.LoadExtension<HTTPFsExtension>();
#else
		return ExtensionLoadResult::NOT_LOADED;
#endif
	} else if (extension == "visualizer") {
#if defined(BUILD_VISUALIZER_EXTENSION) && !defined(DISABLE_BUILTIN_EXTENSIONS)
		db.LoadExtension<VisualizerExtension>();
#else
		// visualizer extension required but not build: skip this test
		return ExtensionLoadResult::NOT_LOADED;
#endif
	} else if (extension == "json") {
#if JSON_STATICALLY_LOADED
		db.LoadExtension<JSONExtension>();
#else
		// json extension required but not build: skip this test
		return ExtensionLoadResult::NOT_LOADED;
#endif
	} else if (extension == "excel") {
#if defined(BUILD_EXCEL_EXTENSION) && !defined(DISABLE_BUILTIN_EXTENSIONS)
		db.LoadExtension<EXCELExtension>();
#else
		// excel extension required but not build: skip this test
		return ExtensionLoadResult::NOT_LOADED;
#endif
	} else if (extension == "sqlsmith") {
#if defined(BUILD_SQLSMITH_EXTENSION) && !defined(DISABLE_BUILTIN_EXTENSIONS)
		db.LoadExtension<SQLSmithExtension>();
#else
		// excel extension required but not build: skip this test
		return ExtensionLoadResult::NOT_LOADED;
#endif
	} else if (extension == "jemalloc") {
#if defined(BUILD_JEMALLOC_EXTENSION) && !defined(DISABLE_BUILTIN_EXTENSIONS)
		db.LoadExtension<JEMallocExtension>();
#else
		// jemalloc extension required but not build: skip this test
		return ExtensionLoadResult::NOT_LOADED;
#endif
	} else if (extension == "inet") {
#if defined(BUILD_INET_EXTENSION) && !defined(DISABLE_BUILTIN_EXTENSIONS)
		db.LoadExtension<INETExtension>();
#else
		// inet extension required but not build: skip this test
		return ExtensionLoadResult::NOT_LOADED;
#endif
	} else {
		// unknown extension
		return ExtensionLoadResult::EXTENSION_UNKNOWN;
	}
	return ExtensionLoadResult::LOADED_EXTENSION;
}

static std::vector<std::string> public_keys = {
    R"(
-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA6aZuHUa1cLR9YDDYaEfi
UDbWY8m2t7b71S+k1ZkXfHqu+5drAxm+dIDzdOHOKZSIdwnJbT3sSqwFoG6PlXF3
g3dsJjax5qESIhbVvf98nyipwNINxoyHCkcCIPkX17QP2xpnT7V59+CqcfDJXLqB
ymjqoFSlaH8dUCHybM4OXlWnAtVHW/nmw0khF8CetcWn4LxaTUHptByaBz8CasSs
gWpXgSfaHc3R9eArsYhtsVFGyL/DEWgkEHWolxY3Llenhgm/zOf3s7PsAMe7EJX4
qlSgiXE6OVBXnqd85z4k20lCw/LAOe5hoTMmRWXIj74MudWe2U91J6GrrGEZa7zT
7QIDAQAB
-----END PUBLIC KEY-----
)",
    R"(
-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAq8Gg1S/LI6ApMAYsFc9m
PrkFIY+nc0LXSpxm77twU8D5M0Xkz/Av4f88DQmj1OE3164bEtR7sl7xDPZojFHj
YYyucJxEI97l5OU1d3Pc1BdKXL4+mnW5FlUGj218u8qD+G1hrkySXQkrUzIjPPNw
o6knF3G/xqQF+KI+tc7ajnTni8CAlnUSxfnstycqbVS86m238PLASVPK9/SmIRgO
XCEV+ZNMlerq8EwsW4cJPHH0oNVMcaG+QT4z79roW1rbJghn9ubAVdQU6VLUAikI
b8keUyY+D0XdY9DpDBeiorb1qPYt8BPLOAQrIUAw1CgpMM9KFp9TNvW47KcG4bcB
dQIDAQAB
-----END PUBLIC KEY-----
)",
    R"(
-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAyYATA9KOQ0Azf97QAPfY
Jc/WeZyE4E1qlRgKWKqNtYSXZqk5At0V7w2ntAWtYSpczFrVepCJ0oPMDpZTigEr
NgOgfo5LEhPx5XmtCf62xY/xL3kgtfz9Mm5TBkuQy4KwY4z1npGr4NYYDXtF7kkf
LQE+FnD8Yr4E0wHBib7ey7aeeKWmwqvUjzDqG+TzaqwzO/RCUsSctqSS0t1oo2hv
4q1ofanUXsV8MXk/ujtgxu7WkVvfiSpK1zRazgeZjcrQFO9qL/pla0vBUxa1U8He
GMLnL0oRfcMg7yKrbIMrvlEl2ZmiR9im44dXJWfY42quObwr1PuEkEoCMcMisSWl
jwIDAQAB
-----END PUBLIC KEY-----
)",
    R"(
-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA4RvbWx3zLblDHH/lGUF5
Q512MT+v3YPriuibROMllv8WiCLAMeJ0QXbVaIzBOeHDeLx8yvoZZN+TENKxtT6u
IfMMneUzxHBqy0AQNfIsSsOnG5nqoeE/AwbS6VqCdH1aLfoCoPffacHYa0XvTcsi
aVlZfr+UzJS+ty8pRmFVi1UKSOADDdK8XfIovJl/zMP2TxYX2Y3fnjeLtl8Sqs2e
P+eHDoy7Wi4EPTyY7tNTCfxwKNHn1HQ5yrv5dgvMxFWIWXGz24yikFvtwLGHe8uJ
Wi+fBX+0PF0diZ6pIthZ149VU8qCqYAXjgpxZ0EZdrsiF6Ewz0cfg20SYApFcmW4
pwIDAQAB
-----END PUBLIC KEY-----
)",
    R"(
-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAyhd5AfwrUohG3O4DE0K9
O3FmgB7zE4aDnkL8UUfGCh5kdP8q7ewMjekY+c6LwWOmpdJpSwqhfV1q5ZU1l6rk
3hlt03LO3sgs28kcfOVH15hqfxts6Sg5KcRjxStE50ORmXGwXDcS9vqkJ60J1EHA
lcZqbCRSO73ZPLhdepfd0/C6tM0L7Ge6cAE62/MTmYNGv8fDzwQr/kYIJMdoS8Zp
thRpctFZJtPs3b0fffZA/TCLVKMvEVgTWs48751qKid7N/Lm/iEGx/tOf4o23Nec
Pz1IQaGLP+UOLVQbqQBHJWNOqigm7kWhDgs3N4YagWgxPEQ0WVLtFji/ZjlKZc7h
dwIDAQAB
-----END PUBLIC KEY-----
)",
    R"(
-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAnFDg3LhyV6BVE2Z3zQvN
6urrKvPhygTa5+wIPGwYTzJ8DfGALqlsX3VOXMvcJTca6SbuwwkoXHuSU5wQxfcs
bt4jTXD3NIoRwQPl+D9IbgIMuX0ACl27rJmr/f9zkY7qui4k1X82pQkxBe+/qJ4r
TBwVNONVx1fekTMnSCEhwg5yU3TNbkObu0qlQeJfuMWLDQbW/8v/qfr/Nz0JqHDN
yYKfKvFMlORxyJYiOyeOsbzNGEhkGQGOmKhRUhS35kD+oA0jqwPwMCM9O4kFg/L8
iZbpBBX2By1K3msejWMRAewTOyPas6YMQOYq9BMmWQqzVtG5xcaSJwN/YnMpJyqb
sQIDAQAB
-----END PUBLIC KEY-----
)",
    R"(
-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA1z0RU8vGrfEkrscEoZKA
GiOcGh2EMcKwjQpl4nKuR9H4o/dg+CZregVSHg7MP2f8mhLZZyoFev49oWOV4Rmi
qs99UNxm7DyKW1fF1ovowsUW5lsDoKYLvpuzHo0s4laiV4AnIYP7tHGLdzsnK2Os
Cp5dSuMwKHPZ9N25hXxFB/dRrAdIiXHvbSqr4N29XzfQloQpL3bGHLKY6guFHluH
X5dJ9eirVakWWou7BR2rnD0k9vER6oRdVnJ6YKb5uhWEOQ3NmV961oyr+uiDTcep
qqtGHWuFhENixtiWGjFJJcACwqxEAW3bz9lyrfnPDsHSW/rlQVDIAkik+fOp+R7L
kQIDAQAB
-----END PUBLIC KEY-----
)",
    R"(
-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAxwO27e1vnbNcpiDg7Wwx
K/w5aEGukXotu3529ieq+O39H0+Bak4vIbzGhDUh3/ElmxaFMAs4PYrWe/hc2WFD
H4JCOoFIn4y9gQeE855DGGFgeIVd1BnSs5S+5wUEMxLNyHdHSmINN6FsoZ535iUg
KdYjRh1iZevezg7ln8o/O36uthu925ehFBXSy6jLJgQlwmq0KxZJE0OAZhuDBM60
MtIunNa/e5y+Gw3GknFwtRLmn/nEckZx1nEtepYvvUa7UGy+8KuGuhOerCZTutbG
k8liCVgGenRve8unA2LrBbpL+AUf3CrZU/uAxxTqWmw6Z/S6TeW5ozeeyOCh8ii6
TwIDAQAB
-----END PUBLIC KEY-----
)",
    R"(
-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAsGIFOfIQ4RI5qu4klOxf
ge6eXwBMAkuTXyhyIIJDtE8CurnwQvUXVlt+Kf0SfuIFW6MY5ErcWE/vMFbc81IR
9wByOAAV2CTyiLGZT63uE8pN6FSHd6yGYCLjXd3P3cnP3Qj5pBncpLuAUDfHG4wP
bs9jIADw3HysD+eCNja8p7ZC7CzWxTcO7HsEu9deAAU19YywdpagXvQ0pJ9zV5qU
jrHxBygl31t6TmmX+3d+azjGu9Hu36E+5wcSOOhuwAFXDejb40Ixv53ItJ3fZzzH
PF2nj9sQvQ8c5ptjyOvQCBRdqkEWXIVHClxqWb+o59pDIh1G0UGcmiDN7K9Gz5HA
ZQIDAQAB
-----END PUBLIC KEY-----
)",
    R"(
-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAt9uUnlW/CoUXT68yaZh9
SeXHzGRCPNEI98Tara+dgYxDX1z7nfOh8o15liT0QsAzx34EewZOxcKCNiV/dZX5
z4clCkD8uUbZut6IVx8Eu+7Qcd5jZthRc6hQrN9Ltv7ZQEh7KGXOHa53kT2K01ws
4jbVmd/7Nx7y0Yyqhja01pIu/CUaTkODfQxBXwriLdIzp7y/iJeF/TLqCwZWHKQx
QOZnsPEveB1F00Va9MeAtTlXFUJ/TQXquqTjeLj4HuIRtbyuNgWoc0JyF+mcafAl
bnrNEBIfxZhAT81aUCIAzRJp6AqfdeZxnZ/WwohtZQZLXAxFQPTWCcP+Z9M7OIQL
WwIDAQAB
-----END PUBLIC KEY-----
)",
    R"(
-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA56NhfACkeCyZM07l2wmd
iTp24E2tLLKU3iByKlIRWRAvXsOejRMJTHTNHWa3cQ7uLP++Tf2St7ksNsyPMNZy
9QRTLNCYr9rN9loLwdb2sMWxFBwwzCaAOTahGI7GJQy30UB7FEND0X/5U2rZvQij
Q6K+O4aa+K9M5qyOHNMmXywmTnAgWKNaNxQHPRtD2+dSj60T6zXdtIuCrPfcNGg5
gj07qWGEXX83V/L7nSqCiIVYg/wqds1x52Yjk1nhXYNBTqlnhmOd8LynGxz/sXC7
h2Q9XsHjXIChW4FHyLIOl6b4zPMBSxzCigYm3QZJWfAkZv5PBRtnq7vhYOLHzLQj
CwIDAQAB
-----END PUBLIC KEY-----
)",
    R"(
-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAmfPLe0IWGYC0MZC6YiM3
QGfhT6zSKB0I2DW44nlBlWUcF+32jW2bFJtgE76qGGKFeU4kJBWYr99ufHoAodNg
M1Ehl/JfQ5KmbC1WIqnFTrgbmqJde79jeCvCpbFLuqnzidwO1PbXDbfRFQcgWaXT
mDVLNNVmLxA0GkCv+kydE2gtcOD9BDceg7F/56TDvclyI5QqAnjE2XIRMPZlXQP4
oF2kgz4Cn7LxLHYmkU2sS9NYLzHoyUqFplWlxkQjA4eQ0neutV1Ydmc1IX8W7R38
A7nFtaT8iI8w6Vkv7ijYN6xf5cVBPKZ3Dv7AdwPet86JD5mf5v+r7iwg5xl3r77Z
iwIDAQAB
-----END PUBLIC KEY-----
)",
    R"(
-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAoB1kWsX8YmCcFOD9ilBY
xK076HmUAN026uJ8JpmU9Hz+QT1FNXOsnj1h2G6U6btYVIdHUTHy/BvAumrDKqRz
qcEAzCuhxUjPjss54a/Zqu6nQcoIPHuG/Er39oZHIVkPR1WCvWj8wmyYv6T//dPH
unO6tW29sXXxS+J1Gah6vpbtJw1pI/liah1DZzb13KWPDI6ZzviTNnW4S05r6js/
30He+Yud6aywrdaP/7G90qcrteEFcjFy4Xf+5vG960oKoGoDplwX5poay1oCP9tb
g8AC8VSRAGi3oviTeSWZcrLXS8AtJhGvF48cXQj2q+8YeVKVDpH6fPQxJ9Sh9aeU
awIDAQAB
-----END PUBLIC KEY-----
)",
    R"(
-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA4NTMAIYIlCMID00ufy/I
AZXc8pocDx9N1Q5x5/cL3aIpLmx02AKo9BvTJaJuHiTjlwYhPtlhIrHV4HUVTkOX
sISp8B8v9i2I1RIvCTAcvy3gcH6rdRWZ0cdTUiMEqnnxBX9zdzl8oMzZcyauv19D
BeqJvzflIT96b8g8K3mvgJHs9a1j9f0gN8FuTA0c52DouKnrh8UwH7mlrumYerJw
6goJGQuK1HEOt6bcQuvogkbgJWOoEYwjNrPwQvIcP4wyrgSnOHg1yXOFE84oVynJ
czQEOz9ke42I3h8wrnQxilEYBVo2uX8MenqTyfGnE32lPRt3Wv1iEVQls8Cxiuy2
CQIDAQAB
-----END PUBLIC KEY-----
)",
    R"(
-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA3bUtfp66OtRyvIF/oucn
id8mo7gvbNEH04QMLO3Ok43dlWgWI3hekJAqOYc0mvoI5anqr98h8FI7aCYZm/bY
vpz0I1aXBaEPh3aWh8f/w9HME7ykBvmhMe3J+VFGWWL4eswfRl//GCtnSMBzDFhM
SaQOTvADWHkC0njeI5yXjf/lNm6fMACP1cnhuvCtnx7VP/DAtvUk9usDKG56MJnZ
UoVM3HHjbJeRwxCdlSWe12ilCdwMRKSDY92Hk38/zBLenH04C3HRQLjBGewACUmx
uvNInehZ4kSYFGa+7UxBxFtzJhlKzGR73qUjpWzZivCe1K0WfRVP5IWsKNCCESJ/
nQIDAQAB
-----END PUBLIC KEY-----
)",
    R"(
-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAyV2dE/CRUAUE8ybq/DoS
Lc7QlYXh04K+McbhN724TbHahLTuDk5mR5TAunA8Nea4euRzknKdMFAz1eh9gyy3
5x4UfXQW1fIZqNo6WNrGxYJgWAXU+pov+OvxsMQWzqS4jrTHDHbblCCLKp1akwJk
aFNyqgjAL373PcqXC+XAn8vHx4xHFoFP5lq4lLcJCOW5ee9v9El3w0USLwS+t1cF
RY3kuV6Njlr4zsRH9iM6/zaSuCALYWJ/JrPEurSJXzFZnWsvn6aQdeNeAn08+z0F
k2NwaauEo0xmLqzqTRGzjHqKKmeefN3/+M/FN2FrApDlxWQfhD2Y3USdAiN547Nj
1wIDAQAB
-----END PUBLIC KEY-----
)",
    R"(
-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAvm2+kTrEQWZXuxhWzBdl
PCbQGqbrukbeS6JKSlQLJDC8ayZIxFxatqg1Q8UPyv89MVRsHOGlG1OqFaOEtPjQ
Oo6j/moFwB4GPyJhJHOGpCKa4CLB5clhfDCLJw6ty7PcDU3T6yW4X4Qc5k4LRRWy
yzC8lVHfBdarN+1iEe0ALMOGoeiJjVn6i/AFxktRwgd8njqv/oWQyfjJZXkNMsb6
7ZDxNVAUrp/WXpE4Kq694bB9xa/pWsqv7FjQJUgTnEzvbN+qXnVPtA7dHcOYYJ8Z
SbrJUfHrf8TS5B54AiopFpWG+hIbjqqdigqabBqFpmjiRDZgDy4zJJj52xJZMnrp
rwIDAQAB
-----END PUBLIC KEY-----
)",
    R"(
-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAwEAcVmY3589O02pLA22f
MlarLyJUgy0BeJDG5AUsi17ct8sHZzRiv9zKQVCBk1CtZY//jyqnrM7iCBLWsyby
TiTOtGYHHApaLnNjjtaHdQ6zplhbc3g2XLy+4ab8GNKG3zc8iXpsQM6r+JO5n9pm
V9vollz9dkFxS9l+1P17lZdIgCh9O3EIFJv5QCd5c9l2ezHAan2OhkWhiDtldnH/
MfRXbz7X5sqlwWLa/jhPtvY45x7dZaCHGqNzbupQZs0vHnAVdDu3vAWDmT/3sXHG
vmGxswKA9tPU0prSvQWLz4LUCnGi/cC5R+fiu+fovFM/BwvaGtqBFIF/1oWVq7bZ
4wIDAQAB
-----END PUBLIC KEY-----
)",
    R"(
-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA25qGwNO1+qHygC8mjm8L
3I66mV/IzslgBDHC91mE8YcI5Fq0sdrtsbUhK3z89wIN/zOhbHX0NEiXm2GxUnsI
vb5tDZXAh7AbTnXTMVbxO/e/8sPLUiObGjDvjVzyzrxOeG87yK/oIiilwk9wTsIb
wMn2Grj4ht9gVKx3oGHYV7STNdWBlzSaJj4Ou7+5M1InjPDRFZG1K31D2d3IHByX
lmcRPZtPFTa5C1uVJw00fI4F4uEFlPclZQlR5yA0G9v+0uDgLcjIUB4eqwMthUWc
dHhlmrPp04LI19eksWHCtG30RzmUaxDiIC7J2Ut0zHDqUe7aXn8tOVI7dE9tTKQD
KQIDAQAB
-----END PUBLIC KEY-----
)",
    R"(
-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA7EC2bx7aRnf3TcRg5gmw
QOKNCUheCelK8hoXLMsKSJqmufyJ+IHUejpXGOpvyYRbACiJ5GiNcww20MVpTBU7
YESWB2QSU2eEJJXMq84qsZSO8WGmAuKpUckI+hNHKQYJBEDOougV6/vVVEm5c5bc
SLWQo0+/ciQ21Zwz5SwimX8ep1YpqYirO04gcyGZzAfGboXRvdUwA+1bZvuUXdKC
4zsCw2QALlcVpzPwjB5mqA/3a+SPgdLAiLOwWXFDRMnQw44UjsnPJFoXgEZiUpZm
EMS5gLv50CzQqJXK9mNzPuYXNUIc4Pw4ssVWe0OfN3Od90gl5uFUwk/G9lWSYnBN
3wIDAQAB
-----END PUBLIC KEY-----
)"};

const vector<string> ExtensionHelper::GetPublicKeys() {
	return public_keys;
}

} // namespace duckdb





#ifndef DISABLE_DUCKDB_REMOTE_INSTALL

#endif


#include <fstream>

namespace duckdb {

//===--------------------------------------------------------------------===//
// Install Extension
//===--------------------------------------------------------------------===//
const string ExtensionHelper::NormalizeVersionTag(const string &version_tag) {
	if (version_tag.length() > 0 && version_tag[0] != 'v') {
		return "v" + version_tag;
	}
	return version_tag;
}

bool ExtensionHelper::IsRelease(const string &version_tag) {
	return !StringUtil::Contains(version_tag, "-dev");
}

const string ExtensionHelper::GetVersionDirectoryName() {
	if (IsRelease(DuckDB::LibraryVersion())) {
		return NormalizeVersionTag(DuckDB::LibraryVersion());
	} else {
		return DuckDB::SourceID();
	}
}

const vector<string> ExtensionHelper::PathComponents() {
	return vector<string> {".duckdb", "extensions", GetVersionDirectoryName(), DuckDB::Platform()};
}

string ExtensionHelper::ExtensionDirectory(ClientContext &context) {
	auto &fs = FileSystem::GetFileSystem(context);
	string local_path = fs.GetHomeDirectory(FileSystem::GetFileOpener(context));
	if (!fs.DirectoryExists(local_path)) {
		throw IOException("Can't find the home directory at '%s'\nSpecify a home directory using the SET "
		                  "home_directory='/path/to/dir' option.",
		                  local_path);
	}
	auto path_components = PathComponents();
	for (auto &path_ele : path_components) {
		local_path = fs.JoinPath(local_path, path_ele);
		if (!fs.DirectoryExists(local_path)) {
			fs.CreateDirectory(local_path);
		}
	}
	return local_path;
}

void ExtensionHelper::InstallExtension(ClientContext &context, const string &extension, bool force_install) {
	auto &config = DBConfig::GetConfig(context);
	if (!config.options.enable_external_access) {
		throw PermissionException("Installing extensions is disabled through configuration");
	}
	auto &fs = FileSystem::GetFileSystem(context);

	string local_path = ExtensionDirectory(context);

	auto extension_name = ApplyExtensionAlias(fs.ExtractBaseName(extension));

	string local_extension_path = fs.JoinPath(local_path, extension_name + ".duckdb_extension");
	if (fs.FileExists(local_extension_path) && !force_install) {
		return;
	}

	auto uuid = UUID::ToString(UUID::GenerateRandomUUID());
	string temp_path = local_extension_path + ".tmp-" + uuid;
	if (fs.FileExists(temp_path)) {
		fs.RemoveFile(temp_path);
	}
	auto is_http_url = StringUtil::Contains(extension, "http://");
	if (fs.FileExists(extension)) {

		std::ifstream in(extension, std::ios::binary);
		if (in.bad()) {
			throw IOException("Failed to read extension from \"%s\"", extension);
		}
		std::ofstream out(temp_path, std::ios::binary);
		out << in.rdbuf();
		if (out.bad()) {
			throw IOException("Failed to write extension to \"%s\"", temp_path);
		}
		in.close();
		out.close();

		fs.MoveFile(temp_path, local_extension_path);
		return;
	} else if (StringUtil::Contains(extension, "/") && !is_http_url) {
		throw IOException("Failed to read extension from \"%s\": no such file", extension);
	}

#ifdef DISABLE_DUCKDB_REMOTE_INSTALL
	throw BinderException("Remote extension installation is disabled through configuration");
#else
	string url_template = "http://extensions.duckdb.org/${REVISION}/${PLATFORM}/${NAME}.duckdb_extension.gz";

	if (is_http_url) {
		url_template = extension;
		extension_name = "";
	}

	auto url = StringUtil::Replace(url_template, "${REVISION}", GetVersionDirectoryName());
	url = StringUtil::Replace(url, "${PLATFORM}", DuckDB::Platform());
	url = StringUtil::Replace(url, "${NAME}", extension_name);

	string no_http = StringUtil::Replace(url, "http://", "");

	idx_t next = no_http.find('/', 0);
	if (next == string::npos) {
		throw IOException("No slash in URL template");
	}

	// Push the substring [last, next) on to splits
	auto hostname_without_http = no_http.substr(0, next);
	auto url_local_part = no_http.substr(next);

	auto url_base = "http://" + hostname_without_http;
	duckdb_httplib::Client cli(url_base.c_str());

	duckdb_httplib::Headers headers = {{"User-Agent", StringUtil::Format("DuckDB %s %s %s", DuckDB::LibraryVersion(),
	                                                                     DuckDB::SourceID(), DuckDB::Platform())}};

	auto res = cli.Get(url_local_part.c_str(), headers);

	if (!res || res->status != 200) {
		// create suggestions
		vector<string> candidates;
		for (idx_t ext_count = ExtensionHelper::DefaultExtensionCount(), i = 0; i < ext_count; i++) {
			candidates.emplace_back(ExtensionHelper::GetDefaultExtension(i).name);
		}
		for (idx_t ext_count = ExtensionHelper::ExtensionAliasCount(), i = 0; i < ext_count; i++) {
			candidates.emplace_back(ExtensionHelper::GetExtensionAlias(i).alias);
		}
		auto closest_extensions = StringUtil::TopNLevenshtein(candidates, extension_name);
		auto message = StringUtil::CandidatesMessage(closest_extensions, "Candidate extensions");
		for (auto &closest : closest_extensions) {
			if (closest == extension_name) {
				message = "Extension \"" + extension_name + "\" is an existing extension.\n";
				message += "Are you using a development build? In this case, extensions might not (yet) be uploaded.";
				break;
			}
		}
		throw IOException("Failed to download extension \"%s\" at URL \"%s%s\"\n%s", extension_name, url_base,
		                  url_local_part, message);
	}
	auto decompressed_body = GZipFileSystem::UncompressGZIPString(res->body);
	std::ofstream out(temp_path, std::ios::binary);
	out.write(decompressed_body.data(), decompressed_body.size());
	if (out.bad()) {
		throw IOException("Failed to write extension to %s", temp_path);
	}
	out.close();
	fs.MoveFile(temp_path, local_extension_path);
#endif
}

} // namespace duckdb







namespace duckdb {

//===--------------------------------------------------------------------===//
// Load External Extension
//===--------------------------------------------------------------------===//
typedef void (*ext_init_fun_t)(DatabaseInstance &);
typedef const char *(*ext_version_fun_t)(void);

template <class T>
static T LoadFunctionFromDLL(void *dll, const string &function_name, const string &filename) {
	auto function = dlsym(dll, function_name.c_str());
	if (!function) {
		throw IOException("File \"%s\" did not contain function \"%s\": %s", filename, function_name, GetDLError());
	}
	return (T)function;
}

ExtensionInitResult ExtensionHelper::InitialLoad(DBConfig &config, FileOpener *opener, const string &extension) {
	if (!config.options.enable_external_access) {
		throw PermissionException("Loading external extensions is disabled through configuration");
	}
	VirtualFileSystem fallback_file_system; // config may not contain one yet
	auto &fs = config.file_system ? *config.file_system : fallback_file_system;
	auto filename = fs.ConvertSeparators(extension);

	// shorthand case
	if (!StringUtil::Contains(extension, ".") && !StringUtil::Contains(extension, fs.PathSeparator())) {
		string local_path = fs.GetHomeDirectory(opener);
		auto path_components = PathComponents();
		for (auto &path_ele : path_components) {
			local_path = fs.JoinPath(local_path, path_ele);
		}
		string extension_name = ApplyExtensionAlias(extension);
		filename = fs.JoinPath(local_path, extension_name + ".duckdb_extension");
	}

	if (!fs.FileExists(filename)) {
		throw IOException("Extension \"%s\" not found", filename);
	}
	{
		auto handle = fs.OpenFile(filename, FileFlags::FILE_FLAGS_READ);

		// signature is the last 265 bytes of the file

		string signature;
		signature.resize(256);

		auto signature_offset = handle->GetFileSize() - signature.size();

		string file_content;
		file_content.resize(signature_offset);
		handle->Read((void *)file_content.data(), signature_offset, 0);

		// TODO maybe we should do a stream read / hash update here
		handle->Read((void *)signature.data(), signature.size(), signature_offset);

		auto hash = duckdb_mbedtls::MbedTlsWrapper::ComputeSha256Hash(file_content);

		bool any_valid = false;
		for (auto &key : ExtensionHelper::GetPublicKeys()) {
			if (duckdb_mbedtls::MbedTlsWrapper::IsValidSha256Signature(key, signature, hash)) {
				any_valid = true;
				break;
			}
		}
		if (!any_valid && !config.options.allow_unsigned_extensions) {
			throw IOException(config.error_manager->FormatException(ErrorType::UNSIGNED_EXTENSION, filename));
		}
	}
	auto lib_hdl = dlopen(filename.c_str(), RTLD_NOW | RTLD_LOCAL);
	if (!lib_hdl) {
		throw IOException("Extension \"%s\" could not be loaded: %s", filename, GetDLError());
	}

	auto basename = fs.ExtractBaseName(filename);

	ext_version_fun_t version_fun;
	auto version_fun_name = basename + "_version";

	version_fun = LoadFunctionFromDLL<ext_version_fun_t>(lib_hdl, version_fun_name, filename);

	std::string engine_version = std::string(DuckDB::LibraryVersion());

	auto version_fun_result = (*version_fun)();
	if (version_fun_result == nullptr) {
		throw InvalidInputException("Extension \"%s\" returned a nullptr", filename);
	}
	std::string extension_version = std::string(version_fun_result);

	// Trim v's if necessary
	std::string extension_version_trimmed = extension_version;
	std::string engine_version_trimmed = engine_version;
	if (extension_version.length() > 0 && extension_version[0] == 'v') {
		extension_version_trimmed = extension_version.substr(1);
	}
	if (engine_version.length() > 0 && engine_version[0] == 'v') {
		engine_version_trimmed = engine_version.substr(1);
	}

	if (extension_version_trimmed != engine_version_trimmed) {
		throw InvalidInputException("Extension \"%s\" version (%s) does not match DuckDB version (%s)", filename,
		                            extension_version, engine_version);
	}

	ExtensionInitResult res;
	res.basename = basename;
	res.filename = filename;
	res.lib_hdl = lib_hdl;
	return res;
}

void ExtensionHelper::LoadExternalExtension(ClientContext &context, const string &extension) {
	auto &db = DatabaseInstance::GetDatabase(context);

	auto res = InitialLoad(DBConfig::GetConfig(context), FileSystem::GetFileOpener(context), extension);
	auto init_fun_name = res.basename + "_init";

	ext_init_fun_t init_fun;
	init_fun = LoadFunctionFromDLL<ext_init_fun_t>(res.lib_hdl, init_fun_name, res.filename);

	try {
		(*init_fun)(db);
	} catch (std::exception &e) {
		throw InvalidInputException("Initialization function \"%s\" from file \"%s\" threw an exception: \"%s\"",
		                            init_fun_name, res.filename, e.what());
	}

	DatabaseInstance::GetDatabase(context).SetExtensionLoaded(extension);
}

unique_ptr<ReplacementOpenData> ExtensionHelper::ReplacementOpenPre(const string &extension, DBConfig &config) {

	auto res = InitialLoad(config, nullptr, extension); // TODO opener
	auto init_fun_name = res.basename + "_replacement_open_pre";

	replacement_open_pre_t open_pre_fun;
	open_pre_fun = LoadFunctionFromDLL<replacement_open_pre_t>(res.lib_hdl, init_fun_name, res.filename);

	try {
		return (*open_pre_fun)(config, nullptr);
	} catch (std::exception &e) {
		throw InvalidInputException("Initialization function \"%s\" from file \"%s\" threw an exception: \"%s\"",
		                            init_fun_name, res.filename, e.what());
	}
}

void ExtensionHelper::ReplacementOpenPost(ClientContext &context, const string &extension, DatabaseInstance &instance,
                                          ReplacementOpenData *open_data) {
	auto res = InitialLoad(DBConfig::GetConfig(context), FileSystem::GetFileOpener(context), extension);
	auto init_fun_name = res.basename + "_replacement_open_post";

	replacement_open_post_t open_post_fun;
	open_post_fun = LoadFunctionFromDLL<replacement_open_post_t>(res.lib_hdl, init_fun_name, res.filename);

	try {
		(*open_post_fun)(instance, open_data);
	} catch (std::exception &e) {
		throw InvalidInputException("Initialization function \"%s\" from file \"%s\" threw an exception: \"%s\"",
		                            init_fun_name, res.filename, e.what());
	}
}

} // namespace duckdb


namespace duckdb {

Extension::~Extension() {
}

} // namespace duckdb





namespace duckdb {

struct ExtensionPrefixOpenData : public ReplacementOpenData {
	ExtensionPrefixOpenData(string extension, string path, unique_ptr<ReplacementOpenData> data)
	    : extension(move(extension)), path(move(path)), data(move(data)) {
	}
	string extension;
	string path;
	unique_ptr<ReplacementOpenData> data;
};

static unique_ptr<ReplacementOpenData> ExtensionPrefixPreOpen(DBConfig &config, ReplacementOpenStaticData *) {
	auto path = config.options.database_path;
	auto first_colon = path.find(':');
	if (first_colon == string::npos || first_colon < 2) { // needs to be at least two characters because windows c: ...
		return nullptr;
	}
	auto extension = path.substr(0, first_colon);
	D_ASSERT(extension.size() > 1);
	// needs to be alphanumeric
	for (auto &ch : extension) {
		if (!isalnum(ch) && ch != '_') {
			return nullptr;
		}
	}
	auto extension_data = ExtensionHelper::ReplacementOpenPre(extension, config);
	if (extension_data) {
		return make_unique<ExtensionPrefixOpenData>(extension, path, move(extension_data));
	}
	return nullptr;
}

static void ExtensionPrefixPostOpen(DatabaseInstance &instance, ReplacementOpenData *open_data) {
	if (!open_data) {
		return;
	}
	D_ASSERT(open_data);
	auto prefix_open_data = (ExtensionPrefixOpenData *)open_data;

	Connection con(instance);
	ExtensionHelper::LoadExternalExtension(*con.context, prefix_open_data->extension);
	ExtensionHelper::ReplacementOpenPost(*con.context, prefix_open_data->extension, instance,
	                                     prefix_open_data->data.get());
}

ExtensionPrefixReplacementOpen::ExtensionPrefixReplacementOpen()
    : ReplacementOpen(ExtensionPrefixPreOpen, ExtensionPrefixPostOpen) {
}

} // namespace duckdb




namespace duckdb {

MaterializedQueryResult::MaterializedQueryResult(StatementType statement_type, StatementProperties properties,
                                                 vector<string> names_p, unique_ptr<ColumnDataCollection> collection_p,
                                                 ClientProperties client_properties)
    : QueryResult(QueryResultType::MATERIALIZED_RESULT, statement_type, properties, collection_p->Types(),
                  move(names_p), move(client_properties)),
      collection(move(collection_p)), scan_initialized(false) {
}

MaterializedQueryResult::MaterializedQueryResult(PreservedError error)
    : QueryResult(QueryResultType::MATERIALIZED_RESULT, move(error)), scan_initialized(false) {
}

string MaterializedQueryResult::ToString() {
	string result;
	if (success) {
		result = HeaderToString();
		result += "[ Rows: " + to_string(collection->Count()) + "]\n";
		auto &coll = Collection();
		for (auto &row : coll.Rows()) {
			for (idx_t col_idx = 0; col_idx < coll.ColumnCount(); col_idx++) {
				if (col_idx > 0) {
					result += "\t";
				}
				auto val = row.GetValue(col_idx);
				result += val.IsNull() ? "NULL" : StringUtil::Replace(val.ToString(), string("\0", 1), "\\0");
			}
			result += "\n";
		}
		result += "\n";
	} else {
		result = GetError() + "\n";
	}
	return result;
}

Value MaterializedQueryResult::GetValue(idx_t column, idx_t index) {
	if (!row_collection) {
		row_collection = make_unique<ColumnDataRowCollection>(collection->GetRows());
	}
	return row_collection->GetValue(column, index);
}

idx_t MaterializedQueryResult::RowCount() const {
	return collection ? collection->Count() : 0;
}

ColumnDataCollection &MaterializedQueryResult::Collection() {
	if (HasError()) {
		throw InvalidInputException("Attempting to get collection from an unsuccessful query result\n: Error %s",
		                            GetError());
	}
	if (!collection) {
		throw InternalException("Missing collection from materialized query result");
	}
	return *collection;
}

unique_ptr<DataChunk> MaterializedQueryResult::Fetch() {
	return FetchRaw();
}

unique_ptr<DataChunk> MaterializedQueryResult::FetchRaw() {
	if (HasError()) {
		throw InvalidInputException("Attempting to fetch from an unsuccessful query result\nError: %s", GetError());
	}
	auto result = make_unique<DataChunk>();
	collection->InitializeScanChunk(*result);
	if (!scan_initialized) {
		// we disallow zero copy so the chunk is independently usable even after the result is destroyed
		collection->InitializeScan(scan_state, ColumnDataScanProperties::DISALLOW_ZERO_COPY);
		scan_initialized = true;
	}
	collection->Scan(scan_state, *result);
	if (result->size() == 0) {
		return nullptr;
	}
	return result;
}

} // namespace duckdb




namespace duckdb {

PendingQueryResult::PendingQueryResult(shared_ptr<ClientContext> context_p, PreparedStatementData &statement,
                                       vector<LogicalType> types_p, bool allow_stream_result)
    : BaseQueryResult(QueryResultType::PENDING_RESULT, statement.statement_type, statement.properties, move(types_p),
                      statement.names),
      context(move(context_p)), allow_stream_result(allow_stream_result) {
}

PendingQueryResult::PendingQueryResult(PreservedError error)
    : BaseQueryResult(QueryResultType::PENDING_RESULT, move(error)) {
}

PendingQueryResult::~PendingQueryResult() {
}

unique_ptr<ClientContextLock> PendingQueryResult::LockContext() {
	if (!context) {
		if (HasError()) {
			throw InvalidInputException(
			    "Attempting to execute an unsuccessful or closed pending query result\nError: %s", GetError());
		}
		throw InvalidInputException("Attempting to execute an unsuccessful or closed pending query result");
	}
	return context->LockContext();
}

void PendingQueryResult::CheckExecutableInternal(ClientContextLock &lock) {
	bool invalidated = HasError() || !context;
	if (!invalidated) {
		invalidated = !context->IsActiveResult(lock, this);
	}
	if (invalidated) {
		if (HasError()) {
			throw InvalidInputException(
			    "Attempting to execute an unsuccessful or closed pending query result\nError: %s", GetError());
		}
		throw InvalidInputException("Attempting to execute an unsuccessful or closed pending query result");
	}
}

PendingExecutionResult PendingQueryResult::ExecuteTask() {
	auto lock = LockContext();
	return ExecuteTaskInternal(*lock);
}

PendingExecutionResult PendingQueryResult::ExecuteTaskInternal(ClientContextLock &lock) {
	CheckExecutableInternal(lock);
	return context->ExecuteTaskInternal(lock, *this);
}

unique_ptr<QueryResult> PendingQueryResult::ExecuteInternal(ClientContextLock &lock) {
	CheckExecutableInternal(lock);
	while (ExecuteTaskInternal(lock) == PendingExecutionResult::RESULT_NOT_READY) {
	}
	if (HasError()) {
		return make_unique<MaterializedQueryResult>(error);
	}
	auto result = context->FetchResultInternal(lock, *this);
	Close();
	return result;
}

unique_ptr<QueryResult> PendingQueryResult::Execute() {
	auto lock = LockContext();
	return ExecuteInternal(*lock);
}

void PendingQueryResult::Close() {
	context.reset();
}

} // namespace duckdb





namespace duckdb {

PreparedStatement::PreparedStatement(shared_ptr<ClientContext> context, shared_ptr<PreparedStatementData> data_p,
                                     string query, idx_t n_param)
    : context(move(context)), data(move(data_p)), query(move(query)), success(true), n_param(n_param) {
	D_ASSERT(data || !success);
}

PreparedStatement::PreparedStatement(PreservedError error) : context(nullptr), success(false), error(move(error)) {
}

PreparedStatement::~PreparedStatement() {
}

const string &PreparedStatement::GetError() {
	D_ASSERT(HasError());
	return error.Message();
}

bool PreparedStatement::HasError() const {
	return !success;
}

idx_t PreparedStatement::ColumnCount() {
	D_ASSERT(data);
	return data->types.size();
}

StatementType PreparedStatement::GetStatementType() {
	D_ASSERT(data);
	return data->statement_type;
}

StatementProperties PreparedStatement::GetStatementProperties() {
	D_ASSERT(data);
	return data->properties;
}

const vector<LogicalType> &PreparedStatement::GetTypes() {
	D_ASSERT(data);
	return data->types;
}

const vector<string> &PreparedStatement::GetNames() {
	D_ASSERT(data);
	return data->names;
}

unique_ptr<QueryResult> PreparedStatement::Execute(vector<Value> &values, bool allow_stream_result) {
	auto pending = PendingQuery(values, allow_stream_result);
	if (pending->HasError()) {
		return make_unique<MaterializedQueryResult>(pending->GetErrorObject());
	}
	return pending->Execute();
}

unique_ptr<PendingQueryResult> PreparedStatement::PendingQuery(vector<Value> &values, bool allow_stream_result) {
	if (!success) {
		throw InvalidInputException("Attempting to execute an unsuccessfully prepared statement!");
	}
	D_ASSERT(data);
	PendingQueryParameters parameters;
	parameters.parameters = &values;
	parameters.allow_stream_result = allow_stream_result && data->properties.allow_stream_result;
	auto result = context->PendingQuery(query, data, parameters);
	return result;
}

} // namespace duckdb




namespace duckdb {

PreparedStatementData::PreparedStatementData(StatementType type) : statement_type(type) {
}

PreparedStatementData::~PreparedStatementData() {
}

void PreparedStatementData::CheckParameterCount(idx_t parameter_count) {
	const auto required = properties.parameter_count;
	if (parameter_count != required) {
		throw BinderException("Parameter/argument count mismatch for prepared statement. Expected %llu, got %llu",
		                      required, parameter_count);
	}
}

bool PreparedStatementData::RequireRebind(ClientContext &context, const vector<Value> &values) {
	CheckParameterCount(values.size());
	if (!unbound_statement) {
		// no unbound statement!? cannot rebind?
		return false;
	}
	if (!properties.bound_all_parameters) {
		// parameters not yet bound: query always requires a rebind
		return true;
	}
	auto &catalog = Catalog::GetCatalog(context);
	if (catalog.GetCatalogVersion() != catalog_version) {
		//! context is out of bounds
		return true;
	}
	for (auto &it : value_map) {
		const idx_t i = it.first - 1;
		if (values[i].type() != it.second->return_type) {
			return true;
		}
	}
	return false;
}

void PreparedStatementData::Bind(vector<Value> values) {
	// set parameters
	D_ASSERT(!unbound_statement || unbound_statement->n_param == properties.parameter_count);
	CheckParameterCount(values.size());

	// bind the required values
	for (auto &it : value_map) {
		const idx_t i = it.first - 1;
		if (i >= values.size()) {
			throw BinderException("Could not find parameter with index %llu", i + 1);
		}
		D_ASSERT(it.second);
		if (!values[i].DefaultTryCastAs(it.second->return_type)) {
			throw BinderException(
			    "Type mismatch for binding parameter with index %llu, expected type %s but got type %s", i + 1,
			    it.second->return_type.ToString().c_str(), values[i].type().ToString().c_str());
		}
		it.second->value = values[i];
	}
}

bool PreparedStatementData::TryGetType(idx_t param_idx, LogicalType &result) {
	auto it = value_map.find(param_idx);
	if (it == value_map.end()) {
		return false;
	}
	if (it->second->return_type.id() != LogicalTypeId::INVALID) {
		result = it->second->return_type;
	} else {
		result = it->second->value.type();
	}
	return true;
}

LogicalType PreparedStatementData::GetType(idx_t param_idx) {
	LogicalType result;
	if (!TryGetType(param_idx, result)) {
		throw BinderException("Could not find parameter with index %llu", param_idx);
	}
	return result;
}

} // namespace duckdb















#include <utility>
#include <algorithm>

namespace duckdb {

QueryProfiler::QueryProfiler(ClientContext &context_p)
    : context(context_p), running(false), query_requires_profiling(false), is_explain_analyze(false) {
}

bool QueryProfiler::IsEnabled() const {
	return is_explain_analyze ? true : ClientConfig::GetConfig(context).enable_profiler;
}

bool QueryProfiler::IsDetailedEnabled() const {
	return is_explain_analyze ? false : ClientConfig::GetConfig(context).enable_detailed_profiling;
}

ProfilerPrintFormat QueryProfiler::GetPrintFormat() const {
	return ClientConfig::GetConfig(context).profiler_print_format;
}

bool QueryProfiler::PrintOptimizerOutput() const {
	return GetPrintFormat() == ProfilerPrintFormat::QUERY_TREE_OPTIMIZER || IsDetailedEnabled();
}

string QueryProfiler::GetSaveLocation() const {
	return is_explain_analyze ? string() : ClientConfig::GetConfig(context).profiler_save_location;
}

QueryProfiler &QueryProfiler::Get(ClientContext &context) {
	return *ClientData::Get(context).profiler;
}

void QueryProfiler::StartQuery(string query, bool is_explain_analyze, bool start_at_optimizer) {
	if (is_explain_analyze) {
		StartExplainAnalyze();
	}
	if (!IsEnabled()) {
		return;
	}
	if (start_at_optimizer && !PrintOptimizerOutput()) {
		// This is the StartQuery call before the optimizer, but we don't have to print optimizer output
		return;
	}
	if (running) {
		// Called while already running: this should only happen when we print optimizer output
		D_ASSERT(PrintOptimizerOutput());
		return;
	}
	this->running = true;
	this->query = move(query);
	tree_map.clear();
	root = nullptr;
	phase_timings.clear();
	phase_stack.clear();

	main_query.Start();
}

bool QueryProfiler::OperatorRequiresProfiling(PhysicalOperatorType op_type) {
	switch (op_type) {
	case PhysicalOperatorType::ORDER_BY:
	case PhysicalOperatorType::RESERVOIR_SAMPLE:
	case PhysicalOperatorType::STREAMING_SAMPLE:
	case PhysicalOperatorType::LIMIT:
	case PhysicalOperatorType::LIMIT_PERCENT:
	case PhysicalOperatorType::STREAMING_LIMIT:
	case PhysicalOperatorType::TOP_N:
	case PhysicalOperatorType::WINDOW:
	case PhysicalOperatorType::UNNEST:
	case PhysicalOperatorType::UNGROUPED_AGGREGATE:
	case PhysicalOperatorType::HASH_GROUP_BY:
	case PhysicalOperatorType::FILTER:
	case PhysicalOperatorType::PROJECTION:
	case PhysicalOperatorType::COPY_TO_FILE:
	case PhysicalOperatorType::TABLE_SCAN:
	case PhysicalOperatorType::CHUNK_SCAN:
	case PhysicalOperatorType::DELIM_SCAN:
	case PhysicalOperatorType::EXPRESSION_SCAN:
	case PhysicalOperatorType::BLOCKWISE_NL_JOIN:
	case PhysicalOperatorType::NESTED_LOOP_JOIN:
	case PhysicalOperatorType::HASH_JOIN:
	case PhysicalOperatorType::CROSS_PRODUCT:
	case PhysicalOperatorType::PIECEWISE_MERGE_JOIN:
	case PhysicalOperatorType::IE_JOIN:
	case PhysicalOperatorType::DELIM_JOIN:
	case PhysicalOperatorType::UNION:
	case PhysicalOperatorType::RECURSIVE_CTE:
	case PhysicalOperatorType::EMPTY_RESULT:
		return true;
	default:
		return false;
	}
}

void QueryProfiler::Finalize(TreeNode &node) {
	for (auto &child : node.children) {
		Finalize(*child);
		if (node.type == PhysicalOperatorType::UNION) {
			node.info.elements += child->info.elements;
		}
	}
}

void QueryProfiler::StartExplainAnalyze() {
	this->is_explain_analyze = true;
}

void QueryProfiler::EndQuery() {
	lock_guard<mutex> guard(flush_lock);
	if (!IsEnabled() || !running) {
		return;
	}

	main_query.End();
	if (root) {
		Finalize(*root);
	}
	this->running = false;
	// print or output the query profiling after termination
	// EXPLAIN ANALYSE should not be outputted by the profiler
	if (IsEnabled() && !is_explain_analyze) {
		string query_info = ToString();
		auto save_location = GetSaveLocation();
		if (!ClientConfig::GetConfig(context).emit_profiler_output) {
			// disable output
		} else if (save_location.empty()) {
			Printer::Print(query_info);
			Printer::Print("\n");
		} else {
			WriteToFile(save_location.c_str(), query_info);
		}
	}
	this->is_explain_analyze = false;
}
string QueryProfiler::ToString() const {
	const auto format = GetPrintFormat();
	switch (format) {
	case ProfilerPrintFormat::QUERY_TREE:
	case ProfilerPrintFormat::QUERY_TREE_OPTIMIZER:
		return QueryTreeToString();
	case ProfilerPrintFormat::JSON:
		return ToJSON();
	default:
		throw InternalException("Unknown ProfilerPrintFormat \"%s\"", format);
	}
}

void QueryProfiler::StartPhase(string new_phase) {
	if (!IsEnabled() || !running) {
		return;
	}

	if (!phase_stack.empty()) {
		// there are active phases
		phase_profiler.End();
		// add the timing to all phases prior to this one
		string prefix = "";
		for (auto &phase : phase_stack) {
			phase_timings[phase] += phase_profiler.Elapsed();
			prefix += phase + " > ";
		}
		// when there are previous phases, we prefix the current phase with those phases
		new_phase = prefix + new_phase;
	}

	// start a new phase
	phase_stack.push_back(new_phase);
	// restart the timer
	phase_profiler.Start();
}

void QueryProfiler::EndPhase() {
	if (!IsEnabled() || !running) {
		return;
	}
	D_ASSERT(phase_stack.size() > 0);

	// end the timer
	phase_profiler.End();
	// add the timing to all currently active phases
	for (auto &phase : phase_stack) {
		phase_timings[phase] += phase_profiler.Elapsed();
	}
	// now remove the last added phase
	phase_stack.pop_back();

	if (!phase_stack.empty()) {
		phase_profiler.Start();
	}
}

void QueryProfiler::Initialize(PhysicalOperator *root_op) {
	if (!IsEnabled() || !running) {
		return;
	}
	this->query_requires_profiling = false;
	this->root = CreateTree(root_op);
	if (!query_requires_profiling) {
		// query does not require profiling: disable profiling for this query
		this->running = false;
		tree_map.clear();
		root = nullptr;
		phase_timings.clear();
		phase_stack.clear();
	}
}

OperatorProfiler::OperatorProfiler(bool enabled_p) : enabled(enabled_p), active_operator(nullptr) {
}

void OperatorProfiler::StartOperator(const PhysicalOperator *phys_op) {
	if (!enabled) {
		return;
	}

	if (active_operator) {
		throw InternalException("OperatorProfiler: Attempting to call StartOperator while another operator is active");
	}

	active_operator = phys_op;

	// start timing for current element
	op.Start();
}

void OperatorProfiler::EndOperator(DataChunk *chunk) {
	if (!enabled) {
		return;
	}

	if (!active_operator) {
		throw InternalException("OperatorProfiler: Attempting to call EndOperator while another operator is active");
	}

	// finish timing for the current element
	op.End();

	AddTiming(active_operator, op.Elapsed(), chunk ? chunk->size() : 0);
	active_operator = nullptr;
}

void OperatorProfiler::AddTiming(const PhysicalOperator *op, double time, idx_t elements) {
	if (!enabled) {
		return;
	}
	if (!Value::DoubleIsFinite(time)) {
		return;
	}
	auto entry = timings.find(op);
	if (entry == timings.end()) {
		// add new entry
		timings[op] = OperatorInformation(time, elements);
	} else {
		// add to existing entry
		entry->second.time += time;
		entry->second.elements += elements;
	}
}
void OperatorProfiler::Flush(const PhysicalOperator *phys_op, ExpressionExecutor *expression_executor,
                             const string &name, int id) {
	auto entry = timings.find(phys_op);
	if (entry == timings.end()) {
		return;
	}
	auto &operator_timing = timings.find(phys_op)->second;
	if (int(operator_timing.executors_info.size()) <= id) {
		operator_timing.executors_info.resize(id + 1);
	}
	operator_timing.executors_info[id] = make_unique<ExpressionExecutorInfo>(*expression_executor, name, id);
	operator_timing.name = phys_op->GetName();
}

void QueryProfiler::Flush(OperatorProfiler &profiler) {
	lock_guard<mutex> guard(flush_lock);
	if (!IsEnabled() || !running) {
		return;
	}
	for (auto &node : profiler.timings) {
		auto entry = tree_map.find(node.first);
		D_ASSERT(entry != tree_map.end());

		entry->second->info.time += node.second.time;
		entry->second->info.elements += node.second.elements;
		if (!IsDetailedEnabled()) {
			continue;
		}
		for (auto &info : node.second.executors_info) {
			if (!info) {
				continue;
			}
			auto info_id = info->id;
			if (int(entry->second->info.executors_info.size()) <= info_id) {
				entry->second->info.executors_info.resize(info_id + 1);
			}
			entry->second->info.executors_info[info_id] = move(info);
		}
	}
	profiler.timings.clear();
}

static string DrawPadded(const string &str, idx_t width) {
	if (str.size() > width) {
		return str.substr(0, width);
	} else {
		width -= str.size();
		int half_spaces = width / 2;
		int extra_left_space = width % 2 != 0 ? 1 : 0;
		return string(half_spaces + extra_left_space, ' ') + str + string(half_spaces, ' ');
	}
}

static string RenderTitleCase(string str) {
	str = StringUtil::Lower(str);
	str[0] = toupper(str[0]);
	for (idx_t i = 0; i < str.size(); i++) {
		if (str[i] == '_') {
			str[i] = ' ';
			if (i + 1 < str.size()) {
				str[i + 1] = toupper(str[i + 1]);
			}
		}
	}
	return str;
}

static string RenderTiming(double timing) {
	string timing_s;
	if (timing >= 1) {
		timing_s = StringUtil::Format("%.2f", timing);
	} else if (timing >= 0.1) {
		timing_s = StringUtil::Format("%.3f", timing);
	} else {
		timing_s = StringUtil::Format("%.4f", timing);
	}
	return timing_s + "s";
}

string QueryProfiler::QueryTreeToString() const {
	std::stringstream str;
	QueryTreeToStream(str);
	return str.str();
}

void QueryProfiler::QueryTreeToStream(std::ostream &ss) const {
	if (!IsEnabled()) {
		ss << "Query profiling is disabled. Call "
		      "Connection::EnableProfiling() to enable profiling!";
		return;
	}
	ss << "┌─────────────────────────────────────┐\n";
	ss << "│┌───────────────────────────────────┐│\n";
	ss << "││    Query Profiling Information    ││\n";
	ss << "│└───────────────────────────────────┘│\n";
	ss << "└─────────────────────────────────────┘\n";
	ss << StringUtil::Replace(query, "\n", " ") + "\n";
	if (query.empty()) {
		return;
	}

	constexpr idx_t TOTAL_BOX_WIDTH = 39;
	ss << "┌─────────────────────────────────────┐\n";
	ss << "│┌───────────────────────────────────┐│\n";
	string total_time = "Total Time: " + RenderTiming(main_query.Elapsed());
	ss << "││" + DrawPadded(total_time, TOTAL_BOX_WIDTH - 4) + "││\n";
	ss << "│└───────────────────────────────────┘│\n";
	ss << "└─────────────────────────────────────┘\n";
	// print phase timings
	if (PrintOptimizerOutput()) {
		bool has_previous_phase = false;
		for (const auto &entry : GetOrderedPhaseTimings()) {
			if (!StringUtil::Contains(entry.first, " > ")) {
				// primary phase!
				if (has_previous_phase) {
					ss << "│└───────────────────────────────────┘│\n";
					ss << "└─────────────────────────────────────┘\n";
				}
				ss << "┌─────────────────────────────────────┐\n";
				ss << "│" +
				          DrawPadded(RenderTitleCase(entry.first) + ": " + RenderTiming(entry.second),
				                     TOTAL_BOX_WIDTH - 2) +
				          "│\n";
				ss << "│┌───────────────────────────────────┐│\n";
				has_previous_phase = true;
			} else {
				string entry_name = StringUtil::Split(entry.first, " > ")[1];
				ss << "││" +
				          DrawPadded(RenderTitleCase(entry_name) + ": " + RenderTiming(entry.second),
				                     TOTAL_BOX_WIDTH - 4) +
				          "││\n";
			}
		}
		if (has_previous_phase) {
			ss << "│└───────────────────────────────────┘│\n";
			ss << "└─────────────────────────────────────┘\n";
		}
	}
	// render the main operator tree
	if (root) {
		Render(*root, ss);
	}
}

static string JSONSanitize(const string &text) {
	string result;
	result.reserve(text.size());
	for (idx_t i = 0; i < text.size(); i++) {
		switch (text[i]) {
		case '\b':
			result += "\\b";
			break;
		case '\f':
			result += "\\f";
			break;
		case '\n':
			result += "\\n";
			break;
		case '\r':
			result += "\\r";
			break;
		case '\t':
			result += "\\t";
			break;
		case '"':
			result += "\\\"";
			break;
		case '\\':
			result += "\\\\";
			break;
		default:
			result += text[i];
			break;
		}
	}
	return result;
}

// Print a row
static void PrintRow(std::ostream &ss, const string &annotation, int id, const string &name, double time,
                     int sample_counter, int tuple_counter, const string &extra_info, int depth) {
	ss << string(depth * 3, ' ') << " {\n";
	ss << string(depth * 3, ' ') << "   \"annotation\": \"" + JSONSanitize(annotation) + "\",\n";
	ss << string(depth * 3, ' ') << "   \"id\": " + to_string(id) + ",\n";
	ss << string(depth * 3, ' ') << "   \"name\": \"" + JSONSanitize(name) + "\",\n";
#if defined(RDTSC)
	ss << string(depth * 3, ' ') << "   \"timing\": \"NULL\" ,\n";
	ss << string(depth * 3, ' ') << "   \"cycles_per_tuple\": " + StringUtil::Format("%.4f", time) + ",\n";
#else
	ss << string(depth * 3, ' ') << "   \"timing\":" + to_string(time) + ",\n";
	ss << string(depth * 3, ' ') << "   \"cycles_per_tuple\": \"NULL\" ,\n";
#endif
	ss << string(depth * 3, ' ') << "   \"sample_size\": " << to_string(sample_counter) + ",\n";
	ss << string(depth * 3, ' ') << "   \"input_size\": " << to_string(tuple_counter) + ",\n";
	ss << string(depth * 3, ' ') << "   \"extra_info\": \"" << JSONSanitize(extra_info) + "\"\n";
	ss << string(depth * 3, ' ') << " },\n";
}

static void ExtractFunctions(std::ostream &ss, ExpressionInfo &info, int &fun_id, int depth) {
	if (info.hasfunction) {
		double time = info.sample_tuples_count == 0 ? 0 : int(info.function_time) / double(info.sample_tuples_count);
		PrintRow(ss, "Function", fun_id++, info.function_name, time, info.sample_tuples_count, info.tuples_count, "",
		         depth);
	}
	if (info.children.empty()) {
		return;
	}
	// extract the children of this node
	for (auto &child : info.children) {
		ExtractFunctions(ss, *child, fun_id, depth);
	}
}

static void ToJSONRecursive(QueryProfiler::TreeNode &node, std::ostream &ss, int depth = 1) {
	ss << string(depth * 3, ' ') << " {\n";
	ss << string(depth * 3, ' ') << "   \"name\": \"" + JSONSanitize(node.name) + "\",\n";
	ss << string(depth * 3, ' ') << "   \"timing\":" + to_string(node.info.time) + ",\n";
	ss << string(depth * 3, ' ') << "   \"cardinality\":" + to_string(node.info.elements) + ",\n";
	ss << string(depth * 3, ' ') << "   \"extra_info\": \"" + JSONSanitize(node.extra_info) + "\",\n";
	ss << string(depth * 3, ' ') << "   \"timings\": [";
	int32_t function_counter = 1;
	int32_t expression_counter = 1;
	ss << "\n ";
	for (auto &expr_executor : node.info.executors_info) {
		// For each Expression tree
		if (!expr_executor) {
			continue;
		}
		for (auto &expr_timer : expr_executor->roots) {
			double time = expr_timer->sample_tuples_count == 0
			                  ? 0
			                  : double(expr_timer->time) / double(expr_timer->sample_tuples_count);
			PrintRow(ss, "ExpressionRoot", expression_counter++, expr_timer->name, time,
			         expr_timer->sample_tuples_count, expr_timer->tuples_count, expr_timer->extra_info, depth + 1);
			// Extract all functions inside the tree
			ExtractFunctions(ss, *expr_timer->root, function_counter, depth + 1);
		}
	}
	ss.seekp(-2, ss.cur);
	ss << "\n";
	ss << string(depth * 3, ' ') << "   ],\n";
	ss << string(depth * 3, ' ') << "   \"children\": [\n";
	if (node.children.empty()) {
		ss << string(depth * 3, ' ') << "   ]\n";
	} else {
		for (idx_t i = 0; i < node.children.size(); i++) {
			if (i > 0) {
				ss << ",\n";
			}
			ToJSONRecursive(*node.children[i], ss, depth + 1);
		}
		ss << string(depth * 3, ' ') << "   ]\n";
	}
	ss << string(depth * 3, ' ') << " }\n";
}

string QueryProfiler::ToJSON() const {
	if (!IsEnabled()) {
		return "{ \"result\": \"disabled\" }\n";
	}
	if (query.empty()) {
		return "{ \"result\": \"empty\" }\n";
	}
	if (!root) {
		return "{ \"result\": \"error\" }\n";
	}
	std::stringstream ss;
	ss << "{\n";
	ss << "   \"name\":  \"Query\", \n";
	ss << "   \"result\": " + to_string(main_query.Elapsed()) + ",\n";
	ss << "   \"timing\": " + to_string(main_query.Elapsed()) + ",\n";
	ss << "   \"cardinality\": " + to_string(root->info.elements) + ",\n";
	// JSON cannot have literal control characters in string literals
	string extra_info = JSONSanitize(query);
	ss << "   \"extra-info\": \"" + extra_info + "\", \n";
	// print the phase timings
	ss << "   \"timings\": [\n";
	const auto &ordered_phase_timings = GetOrderedPhaseTimings();
	for (idx_t i = 0; i < ordered_phase_timings.size(); i++) {
		if (i > 0) {
			ss << ",\n";
		}
		ss << "   {\n";
		ss << "   \"annotation\": \"" + ordered_phase_timings[i].first + "\", \n";
		ss << "   \"timing\": " + to_string(ordered_phase_timings[i].second) + "\n";
		ss << "   }";
	}
	ss << "\n";
	ss << "   ],\n";
	// recursively print the physical operator tree
	ss << "   \"children\": [\n";
	ToJSONRecursive(*root, ss);
	ss << "   ]\n";
	ss << "}";
	return ss.str();
}

void QueryProfiler::WriteToFile(const char *path, string &info) const {
	ofstream out(path);
	out << info;
	out.close();
	// throw an IO exception if it fails to write the file
	if (out.fail()) {
		throw IOException(strerror(errno));
	}
}

unique_ptr<QueryProfiler::TreeNode> QueryProfiler::CreateTree(PhysicalOperator *root, idx_t depth) {
	if (OperatorRequiresProfiling(root->type)) {
		this->query_requires_profiling = true;
	}
	auto node = make_unique<QueryProfiler::TreeNode>();
	node->type = root->type;
	node->name = root->GetName();
	node->extra_info = root->ParamsToString();
	node->depth = depth;
	tree_map[root] = node.get();
	auto children = root->GetChildren();
	for (auto &child : children) {
		auto child_node = CreateTree(child, depth + 1);
		node->children.push_back(move(child_node));
	}
	return node;
}

void QueryProfiler::Render(const QueryProfiler::TreeNode &node, std::ostream &ss) const {
	TreeRenderer renderer;
	if (IsDetailedEnabled()) {
		renderer.EnableDetailed();
	} else {
		renderer.EnableStandard();
	}
	renderer.Render(node, ss);
}

void QueryProfiler::Print() {
	Printer::Print(QueryTreeToString());
}

vector<QueryProfiler::PhaseTimingItem> QueryProfiler::GetOrderedPhaseTimings() const {
	vector<PhaseTimingItem> result;
	// first sort the phases alphabetically
	vector<string> phases;
	for (auto &entry : phase_timings) {
		phases.push_back(entry.first);
	}
	std::sort(phases.begin(), phases.end());
	for (const auto &phase : phases) {
		auto entry = phase_timings.find(phase);
		D_ASSERT(entry != phase_timings.end());
		result.emplace_back(entry->first, entry->second);
	}
	return result;
}
void QueryProfiler::Propagate(QueryProfiler &qp) {
}

void ExpressionInfo::ExtractExpressionsRecursive(unique_ptr<ExpressionState> &state) {
	if (state->child_states.empty()) {
		return;
	}
	// extract the children of this node
	for (auto &child : state->child_states) {
		auto expr_info = make_unique<ExpressionInfo>();
		if (child->expr.expression_class == ExpressionClass::BOUND_FUNCTION) {
			expr_info->hasfunction = true;
			expr_info->function_name = ((BoundFunctionExpression &)child->expr).function.ToString();
			expr_info->function_time = child->profiler.time;
			expr_info->sample_tuples_count = child->profiler.sample_tuples_count;
			expr_info->tuples_count = child->profiler.tuples_count;
		}
		expr_info->ExtractExpressionsRecursive(child);
		children.push_back(move(expr_info));
	}
	return;
}

ExpressionExecutorInfo::ExpressionExecutorInfo(ExpressionExecutor &executor, const string &name, int id) : id(id) {
	// Extract Expression Root Information from ExpressionExecutorStats
	for (auto &state : executor.GetStates()) {
		roots.push_back(make_unique<ExpressionRootInfo>(*state, name));
	}
}

ExpressionRootInfo::ExpressionRootInfo(ExpressionExecutorState &state, string name)
    : current_count(state.profiler.current_count), sample_count(state.profiler.sample_count),
      sample_tuples_count(state.profiler.sample_tuples_count), tuples_count(state.profiler.tuples_count),
      name(state.name), time(state.profiler.time) {
	// Use the name of expression-tree as extra-info
	extra_info = move(name);
	auto expression_info_p = make_unique<ExpressionInfo>();
	// Maybe root has a function
	if (state.root_state->expr.expression_class == ExpressionClass::BOUND_FUNCTION) {
		expression_info_p->hasfunction = true;
		expression_info_p->function_name = ((BoundFunctionExpression &)state.root_state->expr).function.name;
		expression_info_p->function_time = state.root_state->profiler.time;
		expression_info_p->sample_tuples_count = state.root_state->profiler.sample_tuples_count;
		expression_info_p->tuples_count = state.root_state->profiler.tuples_count;
	}
	expression_info_p->ExtractExpressionsRecursive(state.root_state);
	root = move(expression_info_p);
}
} // namespace duckdb





namespace duckdb {

BaseQueryResult::BaseQueryResult(QueryResultType type, StatementType statement_type, StatementProperties properties,
                                 vector<LogicalType> types_p, vector<string> names_p)
    : type(type), statement_type(statement_type), properties(properties), types(move(types_p)), names(move(names_p)),
      success(true) {
	D_ASSERT(types.size() == names.size());
}

BaseQueryResult::BaseQueryResult(QueryResultType type, PreservedError error)
    : type(type), success(false), error(move(error)) {
}

BaseQueryResult::~BaseQueryResult() {
}

void BaseQueryResult::ThrowError(const string &prepended_message) const {
	D_ASSERT(HasError());
	error.Throw(prepended_message);
}

void BaseQueryResult::SetError(PreservedError error) {
	success = !error;
	this->error = move(error);
}

bool BaseQueryResult::HasError() const {
	D_ASSERT((bool)error == !success);
	return !success;
}

const ExceptionType &BaseQueryResult::GetErrorType() const {
	return error.Type();
}

const std::string &BaseQueryResult::GetError() {
	D_ASSERT(HasError());
	return error.Message();
}

PreservedError &BaseQueryResult::GetErrorObject() {
	return error;
}

idx_t BaseQueryResult::ColumnCount() {
	return types.size();
}

QueryResult::QueryResult(QueryResultType type, StatementType statement_type, StatementProperties properties,
                         vector<LogicalType> types_p, vector<string> names_p, ClientProperties client_properties_p)
    : BaseQueryResult(type, statement_type, properties, move(types_p), move(names_p)),
      client_properties(move(client_properties_p)) {
}

QueryResult::QueryResult(QueryResultType type, PreservedError error) : BaseQueryResult(type, move(error)) {
}

QueryResult::~QueryResult() {
}

unique_ptr<DataChunk> QueryResult::Fetch() {
	auto chunk = FetchRaw();
	if (!chunk) {
		return nullptr;
	}
	chunk->Flatten();
	return chunk;
}

bool QueryResult::Equals(QueryResult &other) { // LCOV_EXCL_START
	// first compare the success state of the results
	if (success != other.success) {
		return false;
	}
	if (!success) {
		return error == other.error;
	}
	// compare names
	if (names != other.names) {
		return false;
	}
	// compare types
	if (types != other.types) {
		return false;
	}
	// now compare the actual values
	// fetch chunks
	while (true) {
		auto lchunk = Fetch();
		auto rchunk = other.Fetch();
		if (!lchunk && !rchunk) {
			return true;
		}
		if (!lchunk || !rchunk) {
			return false;
		}
		if (lchunk->size() == 0 && rchunk->size() == 0) {
			return true;
		}
		if (lchunk->size() != rchunk->size()) {
			return false;
		}
		D_ASSERT(lchunk->ColumnCount() == rchunk->ColumnCount());
		for (idx_t col = 0; col < rchunk->ColumnCount(); col++) {
			for (idx_t row = 0; row < rchunk->size(); row++) {
				auto lvalue = lchunk->GetValue(col, row);
				auto rvalue = rchunk->GetValue(col, row);
				if (lvalue.IsNull() && rvalue.IsNull()) {
					continue;
				}
				if (lvalue.IsNull() != rvalue.IsNull()) {
					return false;
				}
				if (lvalue != rvalue) {
					return false;
				}
			}
		}
	}
} // LCOV_EXCL_STOP

void QueryResult::Print() {
	Printer::Print(ToString());
}

string QueryResult::HeaderToString() {
	string result;
	for (auto &name : names) {
		result += name + "\t";
	}
	result += "\n";
	for (auto &type : types) {
		result += type.ToString() + "\t";
	}
	result += "\n";
	return result;
}

string QueryResult::GetConfigTimezone(QueryResult &query_result) {
	return query_result.client_properties.timezone;
}

} // namespace duckdb





namespace duckdb {

AggregateRelation::AggregateRelation(shared_ptr<Relation> child_p,
                                     vector<unique_ptr<ParsedExpression>> parsed_expressions)
    : Relation(child_p->context, RelationType::AGGREGATE_RELATION), expressions(move(parsed_expressions)),
      child(move(child_p)) {
	// bind the expressions
	context.GetContext()->TryBindRelation(*this, this->columns);
}

AggregateRelation::AggregateRelation(shared_ptr<Relation> child_p,
                                     vector<unique_ptr<ParsedExpression>> parsed_expressions,
                                     vector<unique_ptr<ParsedExpression>> groups_p)
    : Relation(child_p->context, RelationType::AGGREGATE_RELATION), expressions(move(parsed_expressions)),
      groups(move(groups_p)), child(move(child_p)) {
	// bind the expressions
	context.GetContext()->TryBindRelation(*this, this->columns);
}

unique_ptr<QueryNode> AggregateRelation::GetQueryNode() {
	auto child_ptr = child.get();
	while (child_ptr->InheritsColumnBindings()) {
		child_ptr = child_ptr->ChildRelation();
	}
	unique_ptr<QueryNode> result;
	if (child_ptr->type == RelationType::JOIN_RELATION) {
		// child node is a join: push projection into the child query node
		result = child->GetQueryNode();
	} else {
		// child node is not a join: create a new select node and push the child as a table reference
		auto select = make_unique<SelectNode>();
		select->from_table = child->GetTableRef();
		result = move(select);
	}
	D_ASSERT(result->type == QueryNodeType::SELECT_NODE);
	auto &select_node = (SelectNode &)*result;
	if (!groups.empty()) {
		// explicit groups provided: use standard handling
		select_node.aggregate_handling = AggregateHandling::STANDARD_HANDLING;
		select_node.groups.group_expressions.clear();
		GroupingSet grouping_set;
		for (idx_t i = 0; i < groups.size(); i++) {
			select_node.groups.group_expressions.push_back(groups[i]->Copy());
			grouping_set.insert(i);
		}
		select_node.groups.grouping_sets.push_back(move(grouping_set));
	} else {
		// no groups provided: automatically figure out groups (if any)
		select_node.aggregate_handling = AggregateHandling::FORCE_AGGREGATES;
	}
	select_node.select_list.clear();
	for (auto &expr : expressions) {
		select_node.select_list.push_back(expr->Copy());
	}
	return result;
}

string AggregateRelation::GetAlias() {
	return child->GetAlias();
}

const vector<ColumnDefinition> &AggregateRelation::Columns() {
	return columns;
}

string AggregateRelation::ToString(idx_t depth) {
	string str = RenderWhitespace(depth) + "Aggregate [";
	for (idx_t i = 0; i < expressions.size(); i++) {
		if (i != 0) {
			str += ", ";
		}
		str += expressions[i]->ToString();
	}
	str += "]\n";
	return str + child->ToString(depth + 1);
}

} // namespace duckdb






namespace duckdb {

CreateTableRelation::CreateTableRelation(shared_ptr<Relation> child_p, string schema_name, string table_name)
    : Relation(child_p->context, RelationType::CREATE_TABLE_RELATION), child(move(child_p)),
      schema_name(move(schema_name)), table_name(move(table_name)) {
	context.GetContext()->TryBindRelation(*this, this->columns);
}

BoundStatement CreateTableRelation::Bind(Binder &binder) {
	auto select = make_unique<SelectStatement>();
	select->node = child->GetQueryNode();

	CreateStatement stmt;
	auto info = make_unique<CreateTableInfo>();
	info->schema = schema_name;
	info->table = table_name;
	info->query = move(select);
	info->on_conflict = OnCreateConflict::ERROR_ON_CONFLICT;
	stmt.info = move(info);
	return binder.Bind((SQLStatement &)stmt);
}

const vector<ColumnDefinition> &CreateTableRelation::Columns() {
	return columns;
}

string CreateTableRelation::ToString(idx_t depth) {
	string str = RenderWhitespace(depth) + "Create Table\n";
	return str + child->ToString(depth + 1);
}

} // namespace duckdb





namespace duckdb {

CreateViewRelation::CreateViewRelation(shared_ptr<Relation> child_p, string view_name_p, bool replace_p,
                                       bool temporary_p)
    : Relation(child_p->context, RelationType::CREATE_VIEW_RELATION), child(move(child_p)),
      view_name(move(view_name_p)), replace(replace_p), temporary(temporary_p) {
	context.GetContext()->TryBindRelation(*this, this->columns);
}

CreateViewRelation::CreateViewRelation(shared_ptr<Relation> child_p, string schema_name_p, string view_name_p,
                                       bool replace_p, bool temporary_p)
    : Relation(child_p->context, RelationType::CREATE_VIEW_RELATION), child(move(child_p)),
      schema_name(move(schema_name_p)), view_name(move(view_name_p)), replace(replace_p), temporary(temporary_p) {
	context.GetContext()->TryBindRelation(*this, this->columns);
}

BoundStatement CreateViewRelation::Bind(Binder &binder) {
	auto select = make_unique<SelectStatement>();
	select->node = child->GetQueryNode();

	CreateStatement stmt;
	auto info = make_unique<CreateViewInfo>();
	info->query = move(select);
	info->view_name = view_name;
	info->temporary = temporary;
	info->schema = schema_name;
	info->on_conflict = replace ? OnCreateConflict::REPLACE_ON_CONFLICT : OnCreateConflict::ERROR_ON_CONFLICT;
	stmt.info = move(info);
	return binder.Bind((SQLStatement &)stmt);
}

const vector<ColumnDefinition> &CreateViewRelation::Columns() {
	return columns;
}

string CreateViewRelation::ToString(idx_t depth) {
	string str = RenderWhitespace(depth) + "Create View\n";
	return str + child->ToString(depth + 1);
}

} // namespace duckdb






namespace duckdb {

CrossProductRelation::CrossProductRelation(shared_ptr<Relation> left_p, shared_ptr<Relation> right_p)
    : Relation(left_p->context, RelationType::CROSS_PRODUCT_RELATION), left(move(left_p)), right(move(right_p)) {
	if (left->context.GetContext() != right->context.GetContext()) {
		throw Exception("Cannot combine LEFT and RIGHT relations of different connections!");
	}
	context.GetContext()->TryBindRelation(*this, this->columns);
}

unique_ptr<QueryNode> CrossProductRelation::GetQueryNode() {
	auto result = make_unique<SelectNode>();
	result->select_list.push_back(make_unique<StarExpression>());
	result->from_table = GetTableRef();
	return move(result);
}

unique_ptr<TableRef> CrossProductRelation::GetTableRef() {
	auto cross_product_ref = make_unique<CrossProductRef>();
	cross_product_ref->left = left->GetTableRef();
	cross_product_ref->right = right->GetTableRef();
	return move(cross_product_ref);
}

const vector<ColumnDefinition> &CrossProductRelation::Columns() {
	return this->columns;
}

string CrossProductRelation::ToString(idx_t depth) {
	string str = RenderWhitespace(depth);
	str = "Cross Product";
	return str + "\n" + left->ToString(depth + 1) + right->ToString(depth + 1);
}

} // namespace duckdb






namespace duckdb {

DeleteRelation::DeleteRelation(ClientContextWrapper &context, unique_ptr<ParsedExpression> condition_p,
                               string schema_name_p, string table_name_p)
    : Relation(context, RelationType::DELETE_RELATION), condition(move(condition_p)), schema_name(move(schema_name_p)),
      table_name(move(table_name_p)) {
	context.GetContext()->TryBindRelation(*this, this->columns);
}

BoundStatement DeleteRelation::Bind(Binder &binder) {
	auto basetable = make_unique<BaseTableRef>();
	basetable->schema_name = schema_name;
	basetable->table_name = table_name;

	DeleteStatement stmt;
	stmt.condition = condition ? condition->Copy() : nullptr;
	stmt.table = move(basetable);
	return binder.Bind((SQLStatement &)stmt);
}

const vector<ColumnDefinition> &DeleteRelation::Columns() {
	return columns;
}

string DeleteRelation::ToString(idx_t depth) {
	string str = RenderWhitespace(depth) + "DELETE FROM " + table_name;
	if (condition) {
		str += " WHERE " + condition->ToString();
	}
	return str;
}

} // namespace duckdb




namespace duckdb {

DistinctRelation::DistinctRelation(shared_ptr<Relation> child_p)
    : Relation(child_p->context, RelationType::DISTINCT_RELATION), child(move(child_p)) {
	D_ASSERT(child.get() != this);
	vector<ColumnDefinition> dummy_columns;
	context.GetContext()->TryBindRelation(*this, dummy_columns);
}

unique_ptr<QueryNode> DistinctRelation::GetQueryNode() {
	auto child_node = child->GetQueryNode();
	child_node->AddDistinct();
	return child_node;
}

string DistinctRelation::GetAlias() {
	return child->GetAlias();
}

const vector<ColumnDefinition> &DistinctRelation::Columns() {
	return child->Columns();
}

string DistinctRelation::ToString(idx_t depth) {
	string str = RenderWhitespace(depth) + "Distinct\n";
	return str + child->ToString(depth + 1);
	;
}

} // namespace duckdb







namespace duckdb {

ExplainRelation::ExplainRelation(shared_ptr<Relation> child_p)
    : Relation(child_p->context, RelationType::EXPLAIN_RELATION), child(move(child_p)) {
	context.GetContext()->TryBindRelation(*this, this->columns);
}

BoundStatement ExplainRelation::Bind(Binder &binder) {
	auto select = make_unique<SelectStatement>();
	select->node = child->GetQueryNode();
	ExplainStatement explain(move(select));
	return binder.Bind((SQLStatement &)explain);
}

const vector<ColumnDefinition> &ExplainRelation::Columns() {
	return columns;
}

string ExplainRelation::ToString(idx_t depth) {
	string str = RenderWhitespace(depth) + "Explain\n";
	return str + child->ToString(depth + 1);
}

} // namespace duckdb







namespace duckdb {

FilterRelation::FilterRelation(shared_ptr<Relation> child_p, unique_ptr<ParsedExpression> condition_p)
    : Relation(child_p->context, RelationType::FILTER_RELATION), condition(move(condition_p)), child(move(child_p)) {
	D_ASSERT(child.get() != this);
	vector<ColumnDefinition> dummy_columns;
	context.GetContext()->TryBindRelation(*this, dummy_columns);
}

unique_ptr<QueryNode> FilterRelation::GetQueryNode() {
	auto child_ptr = child.get();
	while (child_ptr->InheritsColumnBindings()) {
		child_ptr = child_ptr->ChildRelation();
	}
	if (child_ptr->type == RelationType::JOIN_RELATION) {
		// child node is a join: push filter into WHERE clause of select node
		auto child_node = child->GetQueryNode();
		D_ASSERT(child_node->type == QueryNodeType::SELECT_NODE);
		auto &select_node = (SelectNode &)*child_node;
		if (!select_node.where_clause) {
			select_node.where_clause = condition->Copy();
		} else {
			select_node.where_clause = make_unique<ConjunctionExpression>(
			    ExpressionType::CONJUNCTION_AND, move(select_node.where_clause), condition->Copy());
		}
		return child_node;
	} else {
		auto result = make_unique<SelectNode>();
		result->select_list.push_back(make_unique<StarExpression>());
		result->from_table = child->GetTableRef();
		result->where_clause = condition->Copy();
		return move(result);
	}
}

string FilterRelation::GetAlias() {
	return child->GetAlias();
}

const vector<ColumnDefinition> &FilterRelation::Columns() {
	return child->Columns();
}

string FilterRelation::ToString(idx_t depth) {
	string str = RenderWhitespace(depth) + "Filter [" + condition->ToString() + "]\n";
	return str + child->ToString(depth + 1);
}

} // namespace duckdb







namespace duckdb {

InsertRelation::InsertRelation(shared_ptr<Relation> child_p, string schema_name, string table_name)
    : Relation(child_p->context, RelationType::INSERT_RELATION), child(move(child_p)), schema_name(move(schema_name)),
      table_name(move(table_name)) {
	context.GetContext()->TryBindRelation(*this, this->columns);
}

BoundStatement InsertRelation::Bind(Binder &binder) {
	InsertStatement stmt;
	auto select = make_unique<SelectStatement>();
	select->node = child->GetQueryNode();

	stmt.schema = schema_name;
	stmt.table = table_name;
	stmt.select_statement = move(select);
	return binder.Bind((SQLStatement &)stmt);
}

const vector<ColumnDefinition> &InsertRelation::Columns() {
	return columns;
}

string InsertRelation::ToString(idx_t depth) {
	string str = RenderWhitespace(depth) + "Insert\n";
	return str + child->ToString(depth + 1);
}

} // namespace duckdb






namespace duckdb {

JoinRelation::JoinRelation(shared_ptr<Relation> left_p, shared_ptr<Relation> right_p,
                           unique_ptr<ParsedExpression> condition_p, JoinType type)
    : Relation(left_p->context, RelationType::JOIN_RELATION), left(move(left_p)), right(move(right_p)),
      condition(move(condition_p)), join_type(type) {
	if (left->context.GetContext() != right->context.GetContext()) {
		throw Exception("Cannot combine LEFT and RIGHT relations of different connections!");
	}
	context.GetContext()->TryBindRelation(*this, this->columns);
}

JoinRelation::JoinRelation(shared_ptr<Relation> left_p, shared_ptr<Relation> right_p, vector<string> using_columns_p,
                           JoinType type)
    : Relation(left_p->context, RelationType::JOIN_RELATION), left(move(left_p)), right(move(right_p)),
      using_columns(move(using_columns_p)), join_type(type) {
	if (left->context.GetContext() != right->context.GetContext()) {
		throw Exception("Cannot combine LEFT and RIGHT relations of different connections!");
	}
	context.GetContext()->TryBindRelation(*this, this->columns);
}

unique_ptr<QueryNode> JoinRelation::GetQueryNode() {
	auto result = make_unique<SelectNode>();
	result->select_list.push_back(make_unique<StarExpression>());
	result->from_table = GetTableRef();
	return move(result);
}

unique_ptr<TableRef> JoinRelation::GetTableRef() {
	auto join_ref = make_unique<JoinRef>();
	join_ref->left = left->GetTableRef();
	join_ref->right = right->GetTableRef();
	if (condition) {
		join_ref->condition = condition->Copy();
	}
	join_ref->using_columns = using_columns;
	join_ref->type = join_type;
	return move(join_ref);
}

const vector<ColumnDefinition> &JoinRelation::Columns() {
	return this->columns;
}

string JoinRelation::ToString(idx_t depth) {
	string str = RenderWhitespace(depth);
	str += "Join " + JoinTypeToString(join_type);
	if (condition) {
		str += " " + condition->GetName();
	}

	return str + "\n" + left->ToString(depth + 1) + "\n" + right->ToString(depth + 1);
}

} // namespace duckdb






namespace duckdb {

LimitRelation::LimitRelation(shared_ptr<Relation> child_p, int64_t limit, int64_t offset)
    : Relation(child_p->context, RelationType::PROJECTION_RELATION), limit(limit), offset(offset),
      child(move(child_p)) {
	D_ASSERT(child.get() != this);
}

unique_ptr<QueryNode> LimitRelation::GetQueryNode() {
	auto child_node = child->GetQueryNode();
	auto limit_node = make_unique<LimitModifier>();
	if (limit >= 0) {
		limit_node->limit = make_unique<ConstantExpression>(Value::BIGINT(limit));
	}
	if (offset > 0) {
		limit_node->offset = make_unique<ConstantExpression>(Value::BIGINT(offset));
	}

	child_node->modifiers.push_back(move(limit_node));
	return child_node;
}

string LimitRelation::GetAlias() {
	return child->GetAlias();
}

const vector<ColumnDefinition> &LimitRelation::Columns() {
	return child->Columns();
}

string LimitRelation::ToString(idx_t depth) {
	string str = RenderWhitespace(depth) + "Limit " + to_string(limit);
	if (offset > 0) {
		str += " Offset " + to_string(offset);
	}
	str += "\n";
	return str + child->ToString(depth + 1);
}

} // namespace duckdb






namespace duckdb {

OrderRelation::OrderRelation(shared_ptr<Relation> child_p, vector<OrderByNode> orders)
    : Relation(child_p->context, RelationType::ORDER_RELATION), orders(move(orders)), child(move(child_p)) {
	D_ASSERT(child.get() != this);
	// bind the expressions
	context.GetContext()->TryBindRelation(*this, this->columns);
}

unique_ptr<QueryNode> OrderRelation::GetQueryNode() {
	auto select = make_unique<SelectNode>();
	select->from_table = child->GetTableRef();
	select->select_list.push_back(make_unique<StarExpression>());
	auto order_node = make_unique<OrderModifier>();
	for (idx_t i = 0; i < orders.size(); i++) {
		order_node->orders.emplace_back(orders[i].type, orders[i].null_order, orders[i].expression->Copy());
	}
	select->modifiers.push_back(move(order_node));
	return move(select);
}

string OrderRelation::GetAlias() {
	return child->GetAlias();
}

const vector<ColumnDefinition> &OrderRelation::Columns() {
	return columns;
}

string OrderRelation::ToString(idx_t depth) {
	string str = RenderWhitespace(depth) + "Order [";
	for (idx_t i = 0; i < orders.size(); i++) {
		if (i != 0) {
			str += ", ";
		}
		str += orders[i].expression->ToString() + (orders[i].type == OrderType::ASCENDING ? " ASC" : " DESC");
	}
	str += "]\n";
	return str + child->ToString(depth + 1);
}

} // namespace duckdb





namespace duckdb {

ProjectionRelation::ProjectionRelation(shared_ptr<Relation> child_p,
                                       vector<unique_ptr<ParsedExpression>> parsed_expressions, vector<string> aliases)
    : Relation(child_p->context, RelationType::PROJECTION_RELATION), expressions(move(parsed_expressions)),
      child(move(child_p)) {
	if (!aliases.empty()) {
		if (aliases.size() != expressions.size()) {
			throw ParserException("Aliases list length must match expression list length!");
		}
		for (idx_t i = 0; i < aliases.size(); i++) {
			expressions[i]->alias = aliases[i];
		}
	}
	// bind the expressions
	context.GetContext()->TryBindRelation(*this, this->columns);
}

unique_ptr<QueryNode> ProjectionRelation::GetQueryNode() {
	auto child_ptr = child.get();
	while (child_ptr->InheritsColumnBindings()) {
		child_ptr = child_ptr->ChildRelation();
	}
	unique_ptr<QueryNode> result;
	if (child_ptr->type == RelationType::JOIN_RELATION) {
		// child node is a join: push projection into the child query node
		result = child->GetQueryNode();
	} else {
		// child node is not a join: create a new select node and push the child as a table reference
		auto select = make_unique<SelectNode>();
		select->from_table = child->GetTableRef();
		result = move(select);
	}
	D_ASSERT(result->type == QueryNodeType::SELECT_NODE);
	auto &select_node = (SelectNode &)*result;
	select_node.aggregate_handling = AggregateHandling::NO_AGGREGATES_ALLOWED;
	select_node.select_list.clear();
	for (auto &expr : expressions) {
		select_node.select_list.push_back(expr->Copy());
	}
	return result;
}

string ProjectionRelation::GetAlias() {
	return child->GetAlias();
}

const vector<ColumnDefinition> &ProjectionRelation::Columns() {
	return columns;
}

string ProjectionRelation::ToString(idx_t depth) {
	string str = RenderWhitespace(depth) + "Projection [";
	for (idx_t i = 0; i < expressions.size(); i++) {
		if (i != 0) {
			str += ", ";
		}
		str += expressions[i]->ToString() + " as " + expressions[i]->alias;
	}
	str += "]\n";
	return str + child->ToString(depth + 1);
}

} // namespace duckdb






namespace duckdb {

QueryRelation::QueryRelation(const std::shared_ptr<ClientContext> &context, unique_ptr<SelectStatement> select_stmt_p,
                             string alias_p)
    : Relation(context, RelationType::QUERY_RELATION), select_stmt(move(select_stmt_p)), alias(move(alias_p)) {
	context->TryBindRelation(*this, this->columns);
}

QueryRelation::~QueryRelation() {
}

unique_ptr<SelectStatement> QueryRelation::ParseStatement(ClientContext &context, const string &query,
                                                          const string &error) {
	Parser parser(context.GetParserOptions());
	parser.ParseQuery(query);
	if (parser.statements.size() != 1) {
		throw ParserException(error);
	}
	if (parser.statements[0]->type != StatementType::SELECT_STATEMENT) {
		throw ParserException(error);
	}
	return unique_ptr_cast<SQLStatement, SelectStatement>(move(parser.statements[0]));
}

unique_ptr<SelectStatement> QueryRelation::GetSelectStatement() {
	return unique_ptr_cast<SQLStatement, SelectStatement>(select_stmt->Copy());
}

unique_ptr<QueryNode> QueryRelation::GetQueryNode() {
	auto select = GetSelectStatement();
	return move(select->node);
}

unique_ptr<TableRef> QueryRelation::GetTableRef() {
	auto subquery_ref = make_unique<SubqueryRef>(GetSelectStatement(), GetAlias());
	return move(subquery_ref);
}

string QueryRelation::GetAlias() {
	return alias;
}

const vector<ColumnDefinition> &QueryRelation::Columns() {
	return columns;
}

string QueryRelation::ToString(idx_t depth) {
	return RenderWhitespace(depth) + "Subquery";
}

} // namespace duckdb











namespace duckdb {

ReadCSVRelation::ReadCSVRelation(const std::shared_ptr<ClientContext> &context, string csv_file_p,
                                 vector<ColumnDefinition> columns_p, bool auto_detect, string alias_p)
    : Relation(context, RelationType::READ_CSV_RELATION), csv_file(move(csv_file_p)), auto_detect(auto_detect),
      alias(move(alias_p)), columns(move(columns_p)) {
	if (alias.empty()) {
		alias = StringUtil::Split(csv_file, ".")[0];
	}
}

unique_ptr<QueryNode> ReadCSVRelation::GetQueryNode() {
	auto result = make_unique<SelectNode>();
	result->select_list.push_back(make_unique<StarExpression>());
	result->from_table = GetTableRef();
	return move(result);
}

unique_ptr<TableRef> ReadCSVRelation::GetTableRef() {
	auto table_ref = make_unique<TableFunctionRef>();
	table_ref->alias = alias;
	vector<unique_ptr<ParsedExpression>> children;
	// CSV file
	children.push_back(make_unique<ConstantExpression>(Value(csv_file)));
	if (!auto_detect) {
		// parameters
		child_list_t<Value> column_names;
		for (idx_t i = 0; i < columns.size(); i++) {
			column_names.push_back(make_pair(columns[i].Name(), Value(columns[i].Type().ToString())));
		}
		auto colnames = make_unique<ConstantExpression>(Value::STRUCT(move(column_names)));
		children.push_back(make_unique<ComparisonExpression>(
		    ExpressionType::COMPARE_EQUAL, make_unique<ColumnRefExpression>("columns"), move(colnames)));
	} else {
		children.push_back(make_unique<ComparisonExpression>(ExpressionType::COMPARE_EQUAL,
		                                                     make_unique<ColumnRefExpression>("auto_detect"),
		                                                     make_unique<ConstantExpression>(Value::BOOLEAN(true))));
	}
	table_ref->function = make_unique<FunctionExpression>("read_csv", move(children));
	return move(table_ref);
}

string ReadCSVRelation::GetAlias() {
	return alias;
}

const vector<ColumnDefinition> &ReadCSVRelation::Columns() {
	return columns;
}

string ReadCSVRelation::ToString(idx_t depth) {
	return RenderWhitespace(depth) + "Read CSV [" + csv_file + "]";
}

} // namespace duckdb





namespace duckdb {

SetOpRelation::SetOpRelation(shared_ptr<Relation> left_p, shared_ptr<Relation> right_p, SetOperationType setop_type_p)
    : Relation(left_p->context, RelationType::SET_OPERATION_RELATION), left(move(left_p)), right(move(right_p)),
      setop_type(setop_type_p) {
	if (left->context.GetContext() != right->context.GetContext()) {
		throw Exception("Cannot combine LEFT and RIGHT relations of different connections!");
	}
	vector<ColumnDefinition> dummy_columns;
	context.GetContext()->TryBindRelation(*this, dummy_columns);
}

unique_ptr<QueryNode> SetOpRelation::GetQueryNode() {
	auto result = make_unique<SetOperationNode>();
	if (setop_type == SetOperationType::EXCEPT || setop_type == SetOperationType::INTERSECT) {
		result->modifiers.push_back(make_unique<DistinctModifier>());
	}
	result->left = left->GetQueryNode();
	result->right = right->GetQueryNode();
	result->setop_type = setop_type;
	return move(result);
}

string SetOpRelation::GetAlias() {
	return left->GetAlias();
}

const vector<ColumnDefinition> &SetOpRelation::Columns() {
	return left->Columns();
}

string SetOpRelation::ToString(idx_t depth) {
	string str = RenderWhitespace(depth);
	switch (setop_type) {
	case SetOperationType::UNION:
		str += "Union";
		break;
	case SetOperationType::EXCEPT:
		str += "Except";
		break;
	case SetOperationType::INTERSECT:
		str += "Intersect";
		break;
	default:
		throw InternalException("Unknown setop type");
	}
	return str + "\n" + left->ToString(depth + 1) + right->ToString(depth + 1);
}

} // namespace duckdb




namespace duckdb {

SubqueryRelation::SubqueryRelation(shared_ptr<Relation> child_p, string alias_p)
    : Relation(child_p->context, RelationType::SUBQUERY_RELATION), child(move(child_p)), alias(move(alias_p)) {
	D_ASSERT(child.get() != this);
	vector<ColumnDefinition> dummy_columns;
	context.GetContext()->TryBindRelation(*this, dummy_columns);
}

unique_ptr<QueryNode> SubqueryRelation::GetQueryNode() {
	return child->GetQueryNode();
}

string SubqueryRelation::GetAlias() {
	return alias;
}

const vector<ColumnDefinition> &SubqueryRelation::Columns() {
	return child->Columns();
}

string SubqueryRelation::ToString(idx_t depth) {
	return child->ToString(depth);
}

} // namespace duckdb












namespace duckdb {

TableFunctionRelation::TableFunctionRelation(const std::shared_ptr<ClientContext> &context, string name_p,
                                             vector<Value> parameters_p, named_parameter_map_t named_parameters,
                                             shared_ptr<Relation> input_relation_p)
    : Relation(context, RelationType::TABLE_FUNCTION_RELATION), name(move(name_p)), parameters(move(parameters_p)),
      named_parameters(move(named_parameters)), input_relation(move(input_relation_p)) {
	context->TryBindRelation(*this, this->columns);
}
TableFunctionRelation::TableFunctionRelation(const std::shared_ptr<ClientContext> &context, string name_p,
                                             vector<Value> parameters_p,

                                             shared_ptr<Relation> input_relation_p)
    : Relation(context, RelationType::TABLE_FUNCTION_RELATION), name(move(name_p)), parameters(move(parameters_p)),
      input_relation(move(input_relation_p)) {
	context->TryBindRelation(*this, this->columns);
}

unique_ptr<QueryNode> TableFunctionRelation::GetQueryNode() {
	auto result = make_unique<SelectNode>();
	result->select_list.push_back(make_unique<StarExpression>());
	result->from_table = GetTableRef();
	return move(result);
}

unique_ptr<TableRef> TableFunctionRelation::GetTableRef() {
	vector<unique_ptr<ParsedExpression>> children;
	if (input_relation) { // input relation becomes first parameter if present, always
		auto subquery = make_unique<SubqueryExpression>();
		subquery->subquery = make_unique<SelectStatement>();
		subquery->subquery->node = input_relation->GetQueryNode();
		subquery->subquery_type = SubqueryType::SCALAR;
		children.push_back(move(subquery));
	}
	for (auto &parameter : parameters) {
		children.push_back(make_unique<ConstantExpression>(parameter));
	}

	for (auto &parameter : named_parameters) {
		// Hackity-hack some comparisons with column refs
		// This is all but pretty, basically the named parameter is the column, the table is empty because that's what
		// the function binder likes
		auto column_ref = make_unique<ColumnRefExpression>(parameter.first);
		auto constant_value = make_unique<ConstantExpression>(parameter.second);
		auto comparison =
		    make_unique<ComparisonExpression>(ExpressionType::COMPARE_EQUAL, move(column_ref), move(constant_value));
		children.push_back(move(comparison));
	}

	auto table_function = make_unique<TableFunctionRef>();
	auto function = make_unique<FunctionExpression>(name, move(children));
	table_function->function = move(function);
	return move(table_function);
}

string TableFunctionRelation::GetAlias() {
	return name;
}

const vector<ColumnDefinition> &TableFunctionRelation::Columns() {
	return columns;
}

string TableFunctionRelation::ToString(idx_t depth) {
	string function_call = name + "(";
	for (idx_t i = 0; i < parameters.size(); i++) {
		if (i > 0) {
			function_call += ", ";
		}
		function_call += parameters[i].ToString();
	}
	function_call += ")";
	return RenderWhitespace(depth) + function_call;
}

} // namespace duckdb









namespace duckdb {

TableRelation::TableRelation(const std::shared_ptr<ClientContext> &context, unique_ptr<TableDescription> description)
    : Relation(context, RelationType::TABLE_RELATION), description(move(description)) {
}

unique_ptr<QueryNode> TableRelation::GetQueryNode() {
	auto result = make_unique<SelectNode>();
	result->select_list.push_back(make_unique<StarExpression>());
	result->from_table = GetTableRef();
	return move(result);
}

unique_ptr<TableRef> TableRelation::GetTableRef() {
	auto table_ref = make_unique<BaseTableRef>();
	table_ref->schema_name = description->schema;
	table_ref->table_name = description->table;
	return move(table_ref);
}

string TableRelation::GetAlias() {
	return description->table;
}

const vector<ColumnDefinition> &TableRelation::Columns() {
	return description->columns;
}

string TableRelation::ToString(idx_t depth) {
	return RenderWhitespace(depth) + "Scan Table [" + description->table + "]";
}

static unique_ptr<ParsedExpression> ParseCondition(ClientContext &context, const string &condition) {
	if (!condition.empty()) {
		auto expression_list = Parser::ParseExpressionList(condition, context.GetParserOptions());
		if (expression_list.size() != 1) {
			throw ParserException("Expected a single expression as filter condition");
		}
		return move(expression_list[0]);
	} else {
		return nullptr;
	}
}

void TableRelation::Update(const string &update_list, const string &condition) {
	vector<string> update_columns;
	vector<unique_ptr<ParsedExpression>> expressions;
	auto cond = ParseCondition(*context.GetContext(), condition);
	Parser::ParseUpdateList(update_list, update_columns, expressions, context.GetContext()->GetParserOptions());
	auto update = make_shared<UpdateRelation>(context, move(cond), description->schema, description->table,
	                                          move(update_columns), move(expressions));
	update->Execute();
}

void TableRelation::Delete(const string &condition) {
	auto cond = ParseCondition(*context.GetContext(), condition);
	auto del = make_shared<DeleteRelation>(context, move(cond), description->schema, description->table);
	del->Execute();
}

} // namespace duckdb






namespace duckdb {

UpdateRelation::UpdateRelation(ClientContextWrapper &context, unique_ptr<ParsedExpression> condition_p,
                               string schema_name_p, string table_name_p, vector<string> update_columns_p,
                               vector<unique_ptr<ParsedExpression>> expressions_p)
    : Relation(context, RelationType::UPDATE_RELATION), condition(move(condition_p)), schema_name(move(schema_name_p)),
      table_name(move(table_name_p)), update_columns(move(update_columns_p)), expressions(move(expressions_p)) {
	D_ASSERT(update_columns.size() == expressions.size());
	context.GetContext()->TryBindRelation(*this, this->columns);
}

BoundStatement UpdateRelation::Bind(Binder &binder) {
	auto basetable = make_unique<BaseTableRef>();
	basetable->schema_name = schema_name;
	basetable->table_name = table_name;

	UpdateStatement stmt;
	stmt.condition = condition ? condition->Copy() : nullptr;
	stmt.table = move(basetable);
	stmt.columns = update_columns;
	for (auto &expr : expressions) {
		stmt.expressions.push_back(expr->Copy());
	}
	return binder.Bind((SQLStatement &)stmt);
}

const vector<ColumnDefinition> &UpdateRelation::Columns() {
	return columns;
}

string UpdateRelation::ToString(idx_t depth) {
	string str = RenderWhitespace(depth) + "UPDATE " + table_name + " SET\n";
	for (idx_t i = 0; i < expressions.size(); i++) {
		str += update_columns[i] + " = " + expressions[i]->ToString() + "\n";
	}
	if (condition) {
		str += "WHERE " + condition->ToString() + "\n";
	}
	return str;
}

} // namespace duckdb








namespace duckdb {

ValueRelation::ValueRelation(const std::shared_ptr<ClientContext> &context, const vector<vector<Value>> &values,
                             vector<string> names_p, string alias_p)
    : Relation(context, RelationType::VALUE_LIST_RELATION), names(move(names_p)), alias(move(alias_p)) {
	// create constant expressions for the values
	for (idx_t row_idx = 0; row_idx < values.size(); row_idx++) {
		auto &list = values[row_idx];
		vector<unique_ptr<ParsedExpression>> expressions;
		for (idx_t col_idx = 0; col_idx < list.size(); col_idx++) {
			expressions.push_back(make_unique<ConstantExpression>(list[col_idx]));
		}
		this->expressions.push_back(move(expressions));
	}
	context->TryBindRelation(*this, this->columns);
}

ValueRelation::ValueRelation(const std::shared_ptr<ClientContext> &context, const string &values_list,
                             vector<string> names_p, string alias_p)
    : Relation(context, RelationType::VALUE_LIST_RELATION), names(move(names_p)), alias(move(alias_p)) {
	this->expressions = Parser::ParseValuesList(values_list, context->GetParserOptions());
	context->TryBindRelation(*this, this->columns);
}

unique_ptr<QueryNode> ValueRelation::GetQueryNode() {
	auto result = make_unique<SelectNode>();
	result->select_list.push_back(make_unique<StarExpression>());
	result->from_table = GetTableRef();
	return move(result);
}

unique_ptr<TableRef> ValueRelation::GetTableRef() {
	auto table_ref = make_unique<ExpressionListRef>();
	// set the expected types/names
	if (columns.empty()) {
		// no columns yet: only set up names
		for (idx_t i = 0; i < names.size(); i++) {
			table_ref->expected_names.push_back(names[i]);
		}
	} else {
		for (idx_t i = 0; i < columns.size(); i++) {
			table_ref->expected_names.push_back(columns[i].Name());
			table_ref->expected_types.push_back(columns[i].Type());
			D_ASSERT(names.size() == 0 || columns[i].Name() == names[i]);
		}
	}
	// copy the expressions
	for (auto &expr_list : expressions) {
		vector<unique_ptr<ParsedExpression>> copied_list;
		copied_list.reserve(expr_list.size());
		for (auto &expr : expr_list) {
			copied_list.push_back(expr->Copy());
		}
		table_ref->values.push_back(move(copied_list));
	}
	table_ref->alias = GetAlias();
	return move(table_ref);
}

string ValueRelation::GetAlias() {
	return alias;
}

const vector<ColumnDefinition> &ValueRelation::Columns() {
	return columns;
}

string ValueRelation::ToString(idx_t depth) {
	string str = RenderWhitespace(depth) + "Values ";
	for (idx_t row_idx = 0; row_idx < expressions.size(); row_idx++) {
		auto &list = expressions[row_idx];
		str += row_idx > 0 ? ", (" : "(";
		for (idx_t col_idx = 0; col_idx < list.size(); col_idx++) {
			str += col_idx > 0 ? ", " : "";
			str += list[col_idx]->ToString();
		}
		str += ")";
	}
	str += "\n";
	return str;
}

} // namespace duckdb







namespace duckdb {

ViewRelation::ViewRelation(const std::shared_ptr<ClientContext> &context, string schema_name_p, string view_name_p)
    : Relation(context, RelationType::VIEW_RELATION), schema_name(move(schema_name_p)), view_name(move(view_name_p)) {
	context->TryBindRelation(*this, this->columns);
}

unique_ptr<QueryNode> ViewRelation::GetQueryNode() {
	auto result = make_unique<SelectNode>();
	result->select_list.push_back(make_unique<StarExpression>());
	result->from_table = GetTableRef();
	return move(result);
}

unique_ptr<TableRef> ViewRelation::GetTableRef() {
	auto table_ref = make_unique<BaseTableRef>();
	table_ref->schema_name = schema_name;
	table_ref->table_name = view_name;
	return move(table_ref);
}

string ViewRelation::GetAlias() {
	return view_name;
}

const vector<ColumnDefinition> &ViewRelation::Columns() {
	return columns;
}

string ViewRelation::ToString(idx_t depth) {
	return RenderWhitespace(depth) + "View [" + view_name + "]";
}

} // namespace duckdb






namespace duckdb {

WriteCSVRelation::WriteCSVRelation(shared_ptr<Relation> child_p, string csv_file_p)
    : Relation(child_p->context, RelationType::WRITE_CSV_RELATION), child(move(child_p)), csv_file(move(csv_file_p)) {
	context.GetContext()->TryBindRelation(*this, this->columns);
}

BoundStatement WriteCSVRelation::Bind(Binder &binder) {
	CopyStatement copy;
	copy.select_statement = child->GetQueryNode();
	auto info = make_unique<CopyInfo>();
	info->is_from = false;
	info->file_path = csv_file;
	info->format = "csv";
	copy.info = move(info);
	return binder.Bind((SQLStatement &)copy);
}

const vector<ColumnDefinition> &WriteCSVRelation::Columns() {
	return columns;
}

string WriteCSVRelation::ToString(idx_t depth) {
	string str = RenderWhitespace(depth) + "Write To CSV [" + csv_file + "]\n";
	return str + child->ToString(depth + 1);
}

} // namespace duckdb



























namespace duckdb {

shared_ptr<Relation> Relation::Project(const string &select_list) {
	return Project(select_list, vector<string>());
}

shared_ptr<Relation> Relation::Project(const string &expression, const string &alias) {
	return Project(expression, vector<string>({alias}));
}

shared_ptr<Relation> Relation::Project(const string &select_list, const vector<string> &aliases) {
	auto expressions = Parser::ParseExpressionList(select_list, context.GetContext()->GetParserOptions());
	return make_shared<ProjectionRelation>(shared_from_this(), move(expressions), aliases);
}

shared_ptr<Relation> Relation::Project(const vector<string> &expressions) {
	vector<string> aliases;
	return Project(expressions, aliases);
}

static vector<unique_ptr<ParsedExpression>> StringListToExpressionList(ClientContext &context,
                                                                       const vector<string> &expressions) {
	if (expressions.empty()) {
		throw ParserException("Zero expressions provided");
	}
	vector<unique_ptr<ParsedExpression>> result_list;
	for (auto &expr : expressions) {
		auto expression_list = Parser::ParseExpressionList(expr, context.GetParserOptions());
		if (expression_list.size() != 1) {
			throw ParserException("Expected a single expression in the expression list");
		}
		result_list.push_back(move(expression_list[0]));
	}
	return result_list;
}

shared_ptr<Relation> Relation::Project(const vector<string> &expressions, const vector<string> &aliases) {
	auto result_list = StringListToExpressionList(*context.GetContext(), expressions);
	return make_shared<ProjectionRelation>(shared_from_this(), move(result_list), aliases);
}

shared_ptr<Relation> Relation::Filter(const string &expression) {
	auto expression_list = Parser::ParseExpressionList(expression, context.GetContext()->GetParserOptions());
	if (expression_list.size() != 1) {
		throw ParserException("Expected a single expression as filter condition");
	}
	return make_shared<FilterRelation>(shared_from_this(), move(expression_list[0]));
}

shared_ptr<Relation> Relation::Filter(const vector<string> &expressions) {
	// if there are multiple expressions, we AND them together
	auto expression_list = StringListToExpressionList(*context.GetContext(), expressions);
	D_ASSERT(!expression_list.empty());

	auto expr = move(expression_list[0]);
	for (idx_t i = 1; i < expression_list.size(); i++) {
		expr =
		    make_unique<ConjunctionExpression>(ExpressionType::CONJUNCTION_AND, move(expr), move(expression_list[i]));
	}
	return make_shared<FilterRelation>(shared_from_this(), move(expr));
}

shared_ptr<Relation> Relation::Limit(int64_t limit, int64_t offset) {
	return make_shared<LimitRelation>(shared_from_this(), limit, offset);
}

shared_ptr<Relation> Relation::Order(const string &expression) {
	auto order_list = Parser::ParseOrderList(expression, context.GetContext()->GetParserOptions());
	return make_shared<OrderRelation>(shared_from_this(), move(order_list));
}

shared_ptr<Relation> Relation::Order(const vector<string> &expressions) {
	if (expressions.empty()) {
		throw ParserException("Zero ORDER BY expressions provided");
	}
	vector<OrderByNode> order_list;
	for (auto &expression : expressions) {
		auto inner_list = Parser::ParseOrderList(expression, context.GetContext()->GetParserOptions());
		if (inner_list.size() != 1) {
			throw ParserException("Expected a single ORDER BY expression in the expression list");
		}
		order_list.push_back(move(inner_list[0]));
	}
	return make_shared<OrderRelation>(shared_from_this(), move(order_list));
}

shared_ptr<Relation> Relation::Join(const shared_ptr<Relation> &other, const string &condition, JoinType type) {
	auto expression_list = Parser::ParseExpressionList(condition, context.GetContext()->GetParserOptions());
	D_ASSERT(!expression_list.empty());

	if (expression_list.size() > 1 || expression_list[0]->type == ExpressionType::COLUMN_REF) {
		// multiple columns or single column ref: the condition is a USING list
		vector<string> using_columns;
		for (auto &expr : expression_list) {
			if (expr->type != ExpressionType::COLUMN_REF) {
				throw ParserException("Expected a single expression as join condition");
			}
			auto &colref = (ColumnRefExpression &)*expr;
			if (colref.IsQualified()) {
				throw ParserException("Expected unqualified column for column in USING clause");
			}
			using_columns.push_back(colref.column_names[0]);
		}
		return make_shared<JoinRelation>(shared_from_this(), other, move(using_columns), type);
	} else {
		// single expression that is not a column reference: use the expression as a join condition
		return make_shared<JoinRelation>(shared_from_this(), other, move(expression_list[0]), type);
	}
}

shared_ptr<Relation> Relation::CrossProduct(const shared_ptr<Relation> &other) {
	return make_shared<CrossProductRelation>(shared_from_this(), other);
}

shared_ptr<Relation> Relation::Union(const shared_ptr<Relation> &other) {
	return make_shared<SetOpRelation>(shared_from_this(), other, SetOperationType::UNION);
}

shared_ptr<Relation> Relation::Except(const shared_ptr<Relation> &other) {
	return make_shared<SetOpRelation>(shared_from_this(), other, SetOperationType::EXCEPT);
}

shared_ptr<Relation> Relation::Intersect(const shared_ptr<Relation> &other) {
	return make_shared<SetOpRelation>(shared_from_this(), other, SetOperationType::INTERSECT);
}

shared_ptr<Relation> Relation::Distinct() {
	return make_shared<DistinctRelation>(shared_from_this());
}

shared_ptr<Relation> Relation::Alias(const string &alias) {
	return make_shared<SubqueryRelation>(shared_from_this(), alias);
}

shared_ptr<Relation> Relation::Aggregate(const string &aggregate_list) {
	auto expression_list = Parser::ParseExpressionList(aggregate_list, context.GetContext()->GetParserOptions());
	return make_shared<AggregateRelation>(shared_from_this(), move(expression_list));
}

shared_ptr<Relation> Relation::Aggregate(const string &aggregate_list, const string &group_list) {
	auto expression_list = Parser::ParseExpressionList(aggregate_list, context.GetContext()->GetParserOptions());
	auto groups = Parser::ParseExpressionList(group_list, context.GetContext()->GetParserOptions());
	return make_shared<AggregateRelation>(shared_from_this(), move(expression_list), move(groups));
}

shared_ptr<Relation> Relation::Aggregate(const vector<string> &aggregates) {
	auto aggregate_list = StringListToExpressionList(*context.GetContext(), aggregates);
	return make_shared<AggregateRelation>(shared_from_this(), move(aggregate_list));
}

shared_ptr<Relation> Relation::Aggregate(const vector<string> &aggregates, const vector<string> &groups) {
	auto aggregate_list = StringListToExpressionList(*context.GetContext(), aggregates);
	auto group_list = StringListToExpressionList(*context.GetContext(), groups);
	return make_shared<AggregateRelation>(shared_from_this(), move(aggregate_list), move(group_list));
}

string Relation::GetAlias() {
	return "relation";
}

unique_ptr<TableRef> Relation::GetTableRef() {
	auto select = make_unique<SelectStatement>();
	select->node = GetQueryNode();
	return make_unique<SubqueryRef>(move(select), GetAlias());
}

unique_ptr<QueryResult> Relation::Execute() {
	return context.GetContext()->Execute(shared_from_this());
}

BoundStatement Relation::Bind(Binder &binder) {
	SelectStatement stmt;
	stmt.node = GetQueryNode();
	return binder.Bind((SQLStatement &)stmt);
}

void Relation::Insert(const string &table_name) {
	Insert(DEFAULT_SCHEMA, table_name);
}

void Relation::Insert(const string &schema_name, const string &table_name) {
	auto insert = make_shared<InsertRelation>(shared_from_this(), schema_name, table_name);
	auto res = insert->Execute();
	if (res->HasError()) {
		const string prepended_message = "Failed to insert into table '" + table_name + "': ";
		res->ThrowError(prepended_message);
	}
}

void Relation::Insert(const vector<vector<Value>> &values) {
	vector<string> column_names;
	auto rel = make_shared<ValueRelation>(context.GetContext(), values, move(column_names), "values");
	rel->Insert(GetAlias());
}

void Relation::Create(const string &table_name) {
	Create(DEFAULT_SCHEMA, table_name);
}

void Relation::Create(const string &schema_name, const string &table_name) {
	auto create = make_shared<CreateTableRelation>(shared_from_this(), schema_name, table_name);
	auto res = create->Execute();
	if (res->HasError()) {
		const string prepended_message = "Failed to create table '" + table_name + "': ";
		res->ThrowError(prepended_message);
	}
}

void Relation::WriteCSV(const string &csv_file) {
	auto write_csv = make_shared<WriteCSVRelation>(shared_from_this(), csv_file);
	auto res = write_csv->Execute();
	if (res->HasError()) {
		const string prepended_message = "Failed to write '" + csv_file + "': ";
		res->ThrowError(prepended_message);
	}
}

shared_ptr<Relation> Relation::CreateView(const string &name, bool replace, bool temporary) {
	auto view = make_shared<CreateViewRelation>(shared_from_this(), name, replace, temporary);
	auto res = view->Execute();
	if (res->HasError()) {
		const string prepended_message = "Failed to create view '" + name + "': ";
		res->ThrowError(prepended_message);
	}
	return shared_from_this();
}

shared_ptr<Relation> Relation::CreateView(const string &schema_name, const string &name, bool replace, bool temporary) {
	auto view = make_shared<CreateViewRelation>(shared_from_this(), schema_name, name, replace, temporary);
	auto res = view->Execute();
	if (res->HasError()) {
		const string prepended_message = "Failed to create view '" + name + "': ";
		res->ThrowError(prepended_message);
	}
	return shared_from_this();
}

unique_ptr<QueryResult> Relation::Query(const string &sql) {
	return context.GetContext()->Query(sql, false);
}

unique_ptr<QueryResult> Relation::Query(const string &name, const string &sql) {
	CreateView(name);
	return Query(sql);
}

unique_ptr<QueryResult> Relation::Explain() {
	auto explain = make_shared<ExplainRelation>(shared_from_this());
	return explain->Execute();
}

void Relation::Update(const string &update, const string &condition) {
	throw Exception("UPDATE can only be used on base tables!");
}

void Relation::Delete(const string &condition) {
	throw Exception("DELETE can only be used on base tables!");
}

shared_ptr<Relation> Relation::TableFunction(const std::string &fname, const vector<Value> &values,
                                             const named_parameter_map_t &named_parameters) {
	return make_shared<TableFunctionRelation>(context.GetContext(), fname, values, named_parameters,
	                                          shared_from_this());
}

shared_ptr<Relation> Relation::TableFunction(const std::string &fname, const vector<Value> &values) {
	return make_shared<TableFunctionRelation>(context.GetContext(), fname, values, shared_from_this());
}

string Relation::ToString() {
	string str;
	str += "---------------------\n";
	str += "--- Relation Tree ---\n";
	str += "---------------------\n";
	str += ToString(0);
	str += "\n\n";
	str += "---------------------\n";
	str += "-- Result Columns  --\n";
	str += "---------------------\n";
	auto &cols = Columns();
	for (idx_t i = 0; i < cols.size(); i++) {
		str += "- " + cols[i].Name() + " (" + cols[i].Type().ToString() + ")\n";
	}
	return str;
}

// LCOV_EXCL_START
unique_ptr<QueryNode> Relation::GetQueryNode() {
	throw InternalException("Cannot create a query node from this node type");
}

void Relation::Head(idx_t limit) {
	auto limit_node = Limit(limit);
	limit_node->Execute()->Print();
}
// LCOV_EXCL_STOP

void Relation::Print() {
	Printer::Print(ToString());
}

string Relation::RenderWhitespace(idx_t depth) {
	return string(depth * 2, ' ');
}

vector<shared_ptr<ExternalDependency>> Relation::GetAllDependencies() {
	vector<shared_ptr<ExternalDependency>> all_dependencies;
	Relation *cur = this;
	while (cur) {
		if (cur->extra_dependencies) {
			all_dependencies.push_back(cur->extra_dependencies);
		}
		cur = cur->ChildRelation();
	}
	return all_dependencies;
}

} // namespace duckdb














namespace duckdb {

//===--------------------------------------------------------------------===//
// Access Mode
//===--------------------------------------------------------------------===//
void AccessModeSetting::SetGlobal(DatabaseInstance *db, DBConfig &config, const Value &input) {
	auto parameter = StringUtil::Lower(input.ToString());
	if (parameter == "automatic") {
		config.options.access_mode = AccessMode::AUTOMATIC;
	} else if (parameter == "read_only") {
		config.options.access_mode = AccessMode::READ_ONLY;
	} else if (parameter == "read_write") {
		config.options.access_mode = AccessMode::READ_WRITE;
	} else {
		throw InvalidInputException(
		    "Unrecognized parameter for option ACCESS_MODE \"%s\". Expected READ_ONLY or READ_WRITE.", parameter);
	}
}

Value AccessModeSetting::GetSetting(ClientContext &context) {
	auto &config = DBConfig::GetConfig(context);
	switch (config.options.access_mode) {
	case AccessMode::AUTOMATIC:
		return "automatic";
	case AccessMode::READ_ONLY:
		return "read_only";
	case AccessMode::READ_WRITE:
		return "read_write";
	default:
		throw InternalException("Unknown access mode setting");
	}
}

//===--------------------------------------------------------------------===//
// Checkpoint Threshold
//===--------------------------------------------------------------------===//
void CheckpointThresholdSetting::SetGlobal(DatabaseInstance *db, DBConfig &config, const Value &input) {
	idx_t new_limit = DBConfig::ParseMemoryLimit(input.ToString());
	config.options.checkpoint_wal_size = new_limit;
}

Value CheckpointThresholdSetting::GetSetting(ClientContext &context) {
	auto &config = DBConfig::GetConfig(context);
	return Value(StringUtil::BytesToHumanReadableString(config.options.checkpoint_wal_size));
}

//===--------------------------------------------------------------------===//
// Debug Checkpoint Abort
//===--------------------------------------------------------------------===//
void DebugCheckpointAbort::SetGlobal(DatabaseInstance *db, DBConfig &config, const Value &input) {
	auto checkpoint_abort = StringUtil::Lower(input.ToString());
	if (checkpoint_abort == "none") {
		config.options.checkpoint_abort = CheckpointAbort::NO_ABORT;
	} else if (checkpoint_abort == "before_truncate") {
		config.options.checkpoint_abort = CheckpointAbort::DEBUG_ABORT_BEFORE_TRUNCATE;
	} else if (checkpoint_abort == "before_header") {
		config.options.checkpoint_abort = CheckpointAbort::DEBUG_ABORT_BEFORE_HEADER;
	} else if (checkpoint_abort == "after_free_list_write") {
		config.options.checkpoint_abort = CheckpointAbort::DEBUG_ABORT_AFTER_FREE_LIST_WRITE;
	} else {
		throw ParserException(
		    "Unrecognized option for PRAGMA debug_checkpoint_abort, expected none, before_truncate or before_header");
	}
}

Value DebugCheckpointAbort::GetSetting(ClientContext &context) {
	return Value();
}

//===--------------------------------------------------------------------===//
// Debug Force External
//===--------------------------------------------------------------------===//
void DebugForceExternal::SetLocal(ClientContext &context, const Value &input) {
	ClientConfig::GetConfig(context).force_external = input.GetValue<bool>();
}

Value DebugForceExternal::GetSetting(ClientContext &context) {
	return Value::BOOLEAN(ClientConfig::GetConfig(context).force_external);
}

//===--------------------------------------------------------------------===//
// Debug Force NoCrossProduct
//===--------------------------------------------------------------------===//
void DebugForceNoCrossProduct::SetLocal(ClientContext &context, const Value &input) {
	ClientConfig::GetConfig(context).force_no_cross_product = input.GetValue<bool>();
}

Value DebugForceNoCrossProduct::GetSetting(ClientContext &context) {
	return Value::BOOLEAN(ClientConfig::GetConfig(context).force_no_cross_product);
}

//===--------------------------------------------------------------------===//
// Debug Window Mode
//===--------------------------------------------------------------------===//
void DebugWindowMode::SetGlobal(DatabaseInstance *db, DBConfig &config, const Value &input) {
	auto param = StringUtil::Lower(input.ToString());
	if (param == "window") {
		config.options.window_mode = WindowAggregationMode::WINDOW;
	} else if (param == "combine") {
		config.options.window_mode = WindowAggregationMode::COMBINE;
	} else if (param == "separate") {
		config.options.window_mode = WindowAggregationMode::SEPARATE;
	} else {
		throw ParserException("Unrecognized option for PRAGMA debug_window_mode, expected window, combine or separate");
	}
}

Value DebugWindowMode::GetSetting(ClientContext &context) {
	return Value();
}

//===--------------------------------------------------------------------===//
// Default Collation
//===--------------------------------------------------------------------===//
void DefaultCollationSetting::SetGlobal(DatabaseInstance *db, DBConfig &config, const Value &input) {
	auto parameter = StringUtil::Lower(input.ToString());
	config.options.collation = parameter;
}

void DefaultCollationSetting::SetLocal(ClientContext &context, const Value &input) {
	auto parameter = input.ToString();
	// bind the collation to verify that it exists
	ExpressionBinder::TestCollation(context, parameter);
	auto &config = DBConfig::GetConfig(context);
	config.options.collation = parameter;
}

Value DefaultCollationSetting::GetSetting(ClientContext &context) {
	auto &config = DBConfig::GetConfig(context);
	return Value(config.options.collation);
}

//===--------------------------------------------------------------------===//
// Default Order
//===--------------------------------------------------------------------===//
void DefaultOrderSetting::SetGlobal(DatabaseInstance *db, DBConfig &config, const Value &input) {
	auto parameter = StringUtil::Lower(input.ToString());
	if (parameter == "ascending" || parameter == "asc") {
		config.options.default_order_type = OrderType::ASCENDING;
	} else if (parameter == "descending" || parameter == "desc") {
		config.options.default_order_type = OrderType::DESCENDING;
	} else {
		throw InvalidInputException("Unrecognized parameter for option DEFAULT_ORDER \"%s\". Expected ASC or DESC.",
		                            parameter);
	}
}

Value DefaultOrderSetting::GetSetting(ClientContext &context) {
	auto &config = DBConfig::GetConfig(context);
	switch (config.options.default_order_type) {
	case OrderType::ASCENDING:
		return "asc";
	case OrderType::DESCENDING:
		return "desc";
	default:
		throw InternalException("Unknown order type setting");
	}
}

//===--------------------------------------------------------------------===//
// Default Null Order
//===--------------------------------------------------------------------===//
void DefaultNullOrderSetting::SetGlobal(DatabaseInstance *db, DBConfig &config, const Value &input) {
	auto parameter = StringUtil::Lower(input.ToString());

	if (parameter == "nulls_first" || parameter == "nulls first" || parameter == "null first" || parameter == "first") {
		config.options.default_null_order = OrderByNullType::NULLS_FIRST;
	} else if (parameter == "nulls_last" || parameter == "nulls last" || parameter == "null last" ||
	           parameter == "last") {
		config.options.default_null_order = OrderByNullType::NULLS_LAST;
	} else {
		throw ParserException(
		    "Unrecognized parameter for option NULL_ORDER \"%s\", expected either NULLS FIRST or NULLS LAST",
		    parameter);
	}
}

Value DefaultNullOrderSetting::GetSetting(ClientContext &context) {
	auto &config = DBConfig::GetConfig(context);
	switch (config.options.default_null_order) {
	case OrderByNullType::NULLS_FIRST:
		return "nulls_first";
	case OrderByNullType::NULLS_LAST:
		return "nulls_last";
	default:
		throw InternalException("Unknown null order setting");
	}
}

//===--------------------------------------------------------------------===//
// Disabled Optimizer
//===--------------------------------------------------------------------===//
void DisabledOptimizersSetting::SetGlobal(DatabaseInstance *db, DBConfig &config, const Value &input) {
	auto list = StringUtil::Split(input.ToString(), ",");
	set<OptimizerType> disabled_optimizers;
	for (auto &entry : list) {
		auto param = StringUtil::Lower(entry);
		StringUtil::Trim(param);
		if (param.empty()) {
			continue;
		}
		disabled_optimizers.insert(OptimizerTypeFromString(param));
	}
	config.options.disabled_optimizers = move(disabled_optimizers);
}

Value DisabledOptimizersSetting::GetSetting(ClientContext &context) {
	auto &config = DBConfig::GetConfig(context);
	string result;
	for (auto &optimizer : config.options.disabled_optimizers) {
		if (!result.empty()) {
			result += ",";
		}
		result += OptimizerTypeToString(optimizer);
	}
	return Value(result);
}

//===--------------------------------------------------------------------===//
// Enable External Access
//===--------------------------------------------------------------------===//
void EnableExternalAccessSetting::SetGlobal(DatabaseInstance *db, DBConfig &config, const Value &input) {
	auto new_value = input.GetValue<bool>();
	if (db && new_value) {
		throw InvalidInputException("Cannot change enable_external_access setting while database is running");
	}
	config.options.enable_external_access = new_value;
}

Value EnableExternalAccessSetting::GetSetting(ClientContext &context) {
	auto &config = DBConfig::GetConfig(context);
	return Value::BOOLEAN(config.options.enable_external_access);
}

//===--------------------------------------------------------------------===//
// Enable FSST Vectors
//===--------------------------------------------------------------------===//
void EnableFSSTVectors::SetGlobal(DatabaseInstance *db, DBConfig &config, const Value &input) {
	config.options.enable_fsst_vectors = input.GetValue<bool>();
}

Value EnableFSSTVectors::GetSetting(ClientContext &context) {
	auto &config = DBConfig::GetConfig(context);
	return Value::BOOLEAN(config.options.enable_fsst_vectors);
}

//===--------------------------------------------------------------------===//
// Allow Unsigned Extensions
//===--------------------------------------------------------------------===//
void AllowUnsignedExtensionsSetting::SetGlobal(DatabaseInstance *db, DBConfig &config, const Value &input) {
	auto new_value = input.GetValue<bool>();
	if (db && new_value) {
		throw InvalidInputException("Cannot change allow_unsigned_extensions setting while database is running");
	}
	config.options.allow_unsigned_extensions = new_value;
}

Value AllowUnsignedExtensionsSetting::GetSetting(ClientContext &context) {
	auto &config = DBConfig::GetConfig(context);
	return Value::BOOLEAN(config.options.allow_unsigned_extensions);
}

//===--------------------------------------------------------------------===//
// Enable Object Cache
//===--------------------------------------------------------------------===//
void EnableObjectCacheSetting::SetGlobal(DatabaseInstance *db, DBConfig &config, const Value &input) {
	config.options.object_cache_enable = input.GetValue<bool>();
}

Value EnableObjectCacheSetting::GetSetting(ClientContext &context) {
	auto &config = DBConfig::GetConfig(context);
	return Value::BOOLEAN(config.options.object_cache_enable);
}

//===--------------------------------------------------------------------===//
// Enable Profiling
//===--------------------------------------------------------------------===//
void EnableProfilingSetting::SetLocal(ClientContext &context, const Value &input) {
	auto parameter = StringUtil::Lower(input.ToString());

	auto &config = ClientConfig::GetConfig(context);
	if (parameter == "json") {
		config.profiler_print_format = ProfilerPrintFormat::JSON;
	} else if (parameter == "query_tree") {
		config.profiler_print_format = ProfilerPrintFormat::QUERY_TREE;
	} else if (parameter == "query_tree_optimizer") {
		config.profiler_print_format = ProfilerPrintFormat::QUERY_TREE_OPTIMIZER;
	} else {
		throw ParserException(
		    "Unrecognized print format %s, supported formats: [json, query_tree, query_tree_optimizer]", parameter);
	}
	config.enable_profiler = true;
	config.emit_profiler_output = true;
}

Value EnableProfilingSetting::GetSetting(ClientContext &context) {
	auto &config = ClientConfig::GetConfig(context);
	if (!config.enable_profiler) {
		return Value();
	}
	switch (config.profiler_print_format) {
	case ProfilerPrintFormat::JSON:
		return Value("json");
	case ProfilerPrintFormat::QUERY_TREE:
		return Value("query_tree");
	case ProfilerPrintFormat::QUERY_TREE_OPTIMIZER:
		return Value("query_tree_optimizer");
	default:
		throw InternalException("Unsupported profiler print format");
	}
}

//===--------------------------------------------------------------------===//
// Enable Progress Bar
//===--------------------------------------------------------------------===//
void EnableProgressBarSetting::SetLocal(ClientContext &context, const Value &input) {
	ClientConfig::GetConfig(context).enable_progress_bar = input.GetValue<bool>();
}

Value EnableProgressBarSetting::GetSetting(ClientContext &context) {
	return Value::BOOLEAN(ClientConfig::GetConfig(context).enable_progress_bar);
}

//===--------------------------------------------------------------------===//
// Experimental Parallel CSV
//===--------------------------------------------------------------------===//
void ExperimentalParallelCSVSetting::SetGlobal(DatabaseInstance *db, DBConfig &config, const Value &input) {
	config.options.experimental_parallel_csv_reader = input.GetValue<bool>();
}

Value ExperimentalParallelCSVSetting::GetSetting(ClientContext &context) {
	auto &config = DBConfig::GetConfig(context);
	return Value::BIGINT(config.options.experimental_parallel_csv_reader);
}

//===--------------------------------------------------------------------===//
// Explain Output
//===--------------------------------------------------------------------===//
void ExplainOutputSetting::SetLocal(ClientContext &context, const Value &input) {
	auto parameter = StringUtil::Lower(input.ToString());
	if (parameter == "all") {
		ClientConfig::GetConfig(context).explain_output_type = ExplainOutputType::ALL;
	} else if (parameter == "optimized_only") {
		ClientConfig::GetConfig(context).explain_output_type = ExplainOutputType::OPTIMIZED_ONLY;
	} else if (parameter == "physical_only") {
		ClientConfig::GetConfig(context).explain_output_type = ExplainOutputType::PHYSICAL_ONLY;
	} else {
		throw ParserException("Unrecognized output type \"%s\", expected either ALL, OPTIMIZED_ONLY or PHYSICAL_ONLY",
		                      parameter);
	}
}

Value ExplainOutputSetting::GetSetting(ClientContext &context) {
	switch (ClientConfig::GetConfig(context).explain_output_type) {
	case ExplainOutputType::ALL:
		return "all";
	case ExplainOutputType::OPTIMIZED_ONLY:
		return "optimized_only";
	case ExplainOutputType::PHYSICAL_ONLY:
		return "physical_only";
	default:
		throw InternalException("Unrecognized explain output type");
	}
}

//===--------------------------------------------------------------------===//
// External Threads Setting
//===--------------------------------------------------------------------===//
void ExternalThreadsSetting::SetGlobal(DatabaseInstance *db, DBConfig &config, const Value &input) {
	config.options.external_threads = input.GetValue<int64_t>();
}

Value ExternalThreadsSetting::GetSetting(ClientContext &context) {
	auto &config = DBConfig::GetConfig(context);
	return Value::BIGINT(config.options.external_threads);
}

//===--------------------------------------------------------------------===//
// File Search Path
//===--------------------------------------------------------------------===//
void FileSearchPathSetting::SetLocal(ClientContext &context, const Value &input) {
	auto parameter = input.ToString();
	auto &client_data = ClientData::Get(context);
	client_data.file_search_path = parameter;
}

Value FileSearchPathSetting::GetSetting(ClientContext &context) {
	auto &client_data = ClientData::Get(context);
	return Value(client_data.file_search_path);
}

//===--------------------------------------------------------------------===//
// Force Compression
//===--------------------------------------------------------------------===//
void ForceCompressionSetting::SetGlobal(DatabaseInstance *db, DBConfig &config, const Value &input) {
	auto compression = StringUtil::Lower(input.ToString());
	if (compression == "none") {
		config.options.force_compression = CompressionType::COMPRESSION_AUTO;
	} else {
		auto compression_type = CompressionTypeFromString(compression);
		if (compression_type == CompressionType::COMPRESSION_AUTO) {
			throw ParserException("Unrecognized option for PRAGMA force_compression, expected none, uncompressed, rle, "
			                      "dictionary, pfor, bitpacking or fsst");
		}
		config.options.force_compression = compression_type;
	}
}

Value ForceCompressionSetting::GetSetting(ClientContext &context) {
	return Value();
}

//===--------------------------------------------------------------------===//
// Home Directory
//===--------------------------------------------------------------------===//
void HomeDirectorySetting::SetLocal(ClientContext &context, const Value &input) {
	auto &config = ClientConfig::GetConfig(context);
	config.home_directory = input.IsNull() ? string() : input.ToString();
}

Value HomeDirectorySetting::GetSetting(ClientContext &context) {
	auto &config = ClientConfig::GetConfig(context);
	return Value(config.home_directory);
}

//===--------------------------------------------------------------------===//
// Log Query Path
//===--------------------------------------------------------------------===//
void LogQueryPathSetting::SetLocal(ClientContext &context, const Value &input) {
	auto &client_data = ClientData::Get(context);
	auto path = input.ToString();
	if (path.empty()) {
		// empty path: clean up query writer
		client_data.log_query_writer = nullptr;
	} else {
		client_data.log_query_writer =
		    make_unique<BufferedFileWriter>(FileSystem::GetFileSystem(context), path,
		                                    BufferedFileWriter::DEFAULT_OPEN_FLAGS, client_data.file_opener.get());
	}
}

Value LogQueryPathSetting::GetSetting(ClientContext &context) {
	auto &client_data = ClientData::Get(context);
	return client_data.log_query_writer ? Value(client_data.log_query_writer->path) : Value();
}

//===--------------------------------------------------------------------===//
// Maximum Expression Depth
//===--------------------------------------------------------------------===//
void MaximumExpressionDepthSetting::SetLocal(ClientContext &context, const Value &input) {
	ClientConfig::GetConfig(context).max_expression_depth = input.GetValue<uint64_t>();
}

Value MaximumExpressionDepthSetting::GetSetting(ClientContext &context) {
	return Value::UBIGINT(ClientConfig::GetConfig(context).max_expression_depth);
}

//===--------------------------------------------------------------------===//
// Maximum Memory
//===--------------------------------------------------------------------===//
void MaximumMemorySetting::SetGlobal(DatabaseInstance *db, DBConfig &config, const Value &input) {
	config.options.maximum_memory = DBConfig::ParseMemoryLimit(input.ToString());
	if (db) {
		BufferManager::GetBufferManager(*db).SetLimit(config.options.maximum_memory);
	}
}

Value MaximumMemorySetting::GetSetting(ClientContext &context) {
	auto &config = DBConfig::GetConfig(context);
	return Value(StringUtil::BytesToHumanReadableString(config.options.maximum_memory));
}

//===--------------------------------------------------------------------===//
// Password Setting
//===--------------------------------------------------------------------===//
void PasswordSetting::SetGlobal(DatabaseInstance *db, DBConfig &config, const Value &input) {
	// nop
}

Value PasswordSetting::GetSetting(ClientContext &context) {
	return Value();
}

//===--------------------------------------------------------------------===//
// Perfect Hash Threshold
//===--------------------------------------------------------------------===//
void PerfectHashThresholdSetting::SetLocal(ClientContext &context, const Value &input) {
	auto bits = input.GetValue<int32_t>();
	if (bits < 0 || bits > 32) {
		throw ParserException("Perfect HT threshold out of range: should be within range 0 - 32");
	}
	ClientConfig::GetConfig(context).perfect_ht_threshold = bits;
}

Value PerfectHashThresholdSetting::GetSetting(ClientContext &context) {
	return Value::BIGINT(ClientConfig::GetConfig(context).perfect_ht_threshold);
}

//===--------------------------------------------------------------------===//
// PreserveIdentifierCase
//===--------------------------------------------------------------------===//
void PreserveIdentifierCase::SetLocal(ClientContext &context, const Value &input) {
	ClientConfig::GetConfig(context).preserve_identifier_case = input.GetValue<bool>();
}

Value PreserveIdentifierCase::GetSetting(ClientContext &context) {
	return Value::BOOLEAN(ClientConfig::GetConfig(context).preserve_identifier_case);
}

//===--------------------------------------------------------------------===//
// PreserveInsertionOrder
//===--------------------------------------------------------------------===//
void PreserveInsertionOrder::SetGlobal(DatabaseInstance *db, DBConfig &config, const Value &input) {
	config.options.preserve_insertion_order = input.GetValue<bool>();
}

Value PreserveInsertionOrder::GetSetting(ClientContext &context) {
	auto &config = DBConfig::GetConfig(context);
	return Value::BOOLEAN(config.options.preserve_insertion_order);
}

//===--------------------------------------------------------------------===//
// Profiler History Size
//===--------------------------------------------------------------------===//
void ProfilerHistorySize::SetLocal(ClientContext &context, const Value &input) {
	auto size = input.GetValue<int64_t>();
	if (size <= 0) {
		throw ParserException("Size should be >= 0");
	}
	auto &client_data = ClientData::Get(context);
	client_data.query_profiler_history->SetProfilerHistorySize(size);
}

Value ProfilerHistorySize::GetSetting(ClientContext &context) {
	return Value();
}

//===--------------------------------------------------------------------===//
// Profile Output
//===--------------------------------------------------------------------===//
void ProfileOutputSetting::SetLocal(ClientContext &context, const Value &input) {
	auto &config = ClientConfig::GetConfig(context);
	auto parameter = input.ToString();
	config.profiler_save_location = parameter;
}

Value ProfileOutputSetting::GetSetting(ClientContext &context) {
	auto &config = ClientConfig::GetConfig(context);
	return Value(config.profiler_save_location);
}

//===--------------------------------------------------------------------===//
// Profiling Mode
//===--------------------------------------------------------------------===//
void ProfilingModeSetting::SetLocal(ClientContext &context, const Value &input) {
	auto parameter = StringUtil::Lower(input.ToString());
	auto &config = ClientConfig::GetConfig(context);
	if (parameter == "standard") {
		config.enable_profiler = true;
		config.enable_detailed_profiling = false;
		config.emit_profiler_output = true;
	} else if (parameter == "detailed") {
		config.enable_profiler = true;
		config.enable_detailed_profiling = true;
		config.emit_profiler_output = true;
	} else {
		throw ParserException("Unrecognized profiling mode \"%s\", supported formats: [standard, detailed]", parameter);
	}
}

Value ProfilingModeSetting::GetSetting(ClientContext &context) {
	auto &config = ClientConfig::GetConfig(context);
	if (!config.enable_profiler) {
		return Value();
	}
	return Value(config.enable_detailed_profiling ? "detailed" : "standard");
}

//===--------------------------------------------------------------------===//
// Progress Bar Time
//===--------------------------------------------------------------------===//
void ProgressBarTimeSetting::SetLocal(ClientContext &context, const Value &input) {
	ClientConfig::GetConfig(context).wait_time = input.GetValue<int32_t>();
	ClientConfig::GetConfig(context).enable_progress_bar = true;
}

Value ProgressBarTimeSetting::GetSetting(ClientContext &context) {
	return Value::BIGINT(ClientConfig::GetConfig(context).wait_time);
}

//===--------------------------------------------------------------------===//
// Schema
//===--------------------------------------------------------------------===//
void SchemaSetting::SetLocal(ClientContext &context, const Value &input) {
	auto parameter = input.ToString();
	auto &client_data = ClientData::Get(context);
	client_data.catalog_search_path->Set(parameter, true);
}

Value SchemaSetting::GetSetting(ClientContext &context) {
	return SearchPathSetting::GetSetting(context);
}

//===--------------------------------------------------------------------===//
// Search Path
//===--------------------------------------------------------------------===//
void SearchPathSetting::SetLocal(ClientContext &context, const Value &input) {
	auto parameter = input.ToString();
	auto &client_data = ClientData::Get(context);
	client_data.catalog_search_path->Set(parameter, false);
}

Value SearchPathSetting::GetSetting(ClientContext &context) {
	auto &client_data = ClientData::Get(context);
	return Value(StringUtil::Join(client_data.catalog_search_path->GetSetPaths(), ","));
}

//===--------------------------------------------------------------------===//
// Temp Directory
//===--------------------------------------------------------------------===//
void TempDirectorySetting::SetGlobal(DatabaseInstance *db, DBConfig &config, const Value &input) {
	config.options.temporary_directory = input.ToString();
	config.options.use_temporary_directory = !config.options.temporary_directory.empty();
	if (db) {
		auto &buffer_manager = BufferManager::GetBufferManager(*db);
		buffer_manager.SetTemporaryDirectory(config.options.temporary_directory);
	}
}

Value TempDirectorySetting::GetSetting(ClientContext &context) {
	auto &buffer_manager = BufferManager::GetBufferManager(context);
	return Value(buffer_manager.GetTemporaryDirectory());
}

//===--------------------------------------------------------------------===//
// Threads Setting
//===--------------------------------------------------------------------===//
void ThreadsSetting::SetGlobal(DatabaseInstance *db, DBConfig &config, const Value &input) {
	config.options.maximum_threads = input.GetValue<int64_t>();
	if (db) {
		TaskScheduler::GetScheduler(*db).SetThreads(config.options.maximum_threads);
	}
}

Value ThreadsSetting::GetSetting(ClientContext &context) {
	auto &config = DBConfig::GetConfig(context);
	return Value::BIGINT(config.options.maximum_threads);
}

//===--------------------------------------------------------------------===//
// Username Setting
//===--------------------------------------------------------------------===//
void UsernameSetting::SetGlobal(DatabaseInstance *db, DBConfig &config, const Value &input) {
	// nop
}

Value UsernameSetting::GetSetting(ClientContext &context) {
	return Value();
}

} // namespace duckdb





namespace duckdb {

StreamQueryResult::StreamQueryResult(StatementType statement_type, StatementProperties properties,
                                     shared_ptr<ClientContext> context_p, vector<LogicalType> types,
                                     vector<string> names)
    : QueryResult(QueryResultType::STREAM_RESULT, statement_type, properties, move(types), move(names),
                  context_p->GetClientProperties()),
      context(move(context_p)) {
	D_ASSERT(context);
}

StreamQueryResult::~StreamQueryResult() {
}

string StreamQueryResult::ToString() {
	string result;
	if (success) {
		result = HeaderToString();
		result += "[[STREAM RESULT]]";
	} else {
		result = GetError() + "\n";
	}
	return result;
}

unique_ptr<ClientContextLock> StreamQueryResult::LockContext() {
	if (!context) {
		string error_str = "Attempting to execute an unsuccessful or closed pending query result";
		if (HasError()) {
			error_str += StringUtil::Format("\nError: %s", GetError());
		}
		throw InvalidInputException(error_str);
	}
	return context->LockContext();
}

void StreamQueryResult::CheckExecutableInternal(ClientContextLock &lock) {
	if (!IsOpenInternal(lock)) {
		string error_str = "Attempting to execute an unsuccessful or closed pending query result";
		if (HasError()) {
			error_str += StringUtil::Format("\nError: %s", GetError());
		}
		throw InvalidInputException(error_str);
	}
}

unique_ptr<DataChunk> StreamQueryResult::FetchRaw() {
	unique_ptr<DataChunk> chunk;
	{
		auto lock = LockContext();
		CheckExecutableInternal(*lock);
		chunk = context->Fetch(*lock, *this);
	}
	if (!chunk || chunk->ColumnCount() == 0 || chunk->size() == 0) {
		Close();
		return nullptr;
	}
	return chunk;
}

unique_ptr<MaterializedQueryResult> StreamQueryResult::Materialize() {
	if (HasError() || !context) {
		return make_unique<MaterializedQueryResult>(GetErrorObject());
	}
	auto collection = make_unique<ColumnDataCollection>(Allocator::DefaultAllocator(), types);

	ColumnDataAppendState append_state;
	collection->InitializeAppend(append_state);
	while (true) {
		auto chunk = Fetch();
		if (!chunk || chunk->size() == 0) {
			break;
		}
		collection->Append(append_state, *chunk);
	}
	auto result =
	    make_unique<MaterializedQueryResult>(statement_type, properties, names, move(collection), client_properties);
	if (HasError()) {
		return make_unique<MaterializedQueryResult>(GetErrorObject());
	}
	return result;
}

bool StreamQueryResult::IsOpenInternal(ClientContextLock &lock) {
	bool invalidated = !success || !context;
	if (!invalidated) {
		invalidated = !context->IsActiveResult(lock, this);
	}
	return !invalidated;
}

bool StreamQueryResult::IsOpen() {
	if (!success || !context) {
		return false;
	}
	auto lock = LockContext();
	return IsOpenInternal(*lock);
}

void StreamQueryResult::Close() {
	context.reset();
}

} // namespace duckdb


namespace duckdb {

ValidChecker::ValidChecker() : is_invalidated(false) {
}

void ValidChecker::Invalidate(string error) {
	lock_guard<mutex> l(invalidate_lock);
	this->is_invalidated = true;
	this->invalidated_msg = move(error);
}

bool ValidChecker::IsInvalidated() {
	return this->is_invalidated;
}

string ValidChecker::InvalidatedMessage() {
	lock_guard<mutex> l(invalidate_lock);
	return invalidated_msg;
}
} // namespace duckdb










namespace duckdb {

static TableCatalogEntry *GetCatalogTableEntry(LogicalOperator *op) {
	if (op->type == LogicalOperatorType::LOGICAL_GET) {
		auto get = (LogicalGet *)op;
		TableCatalogEntry *entry = get->GetTable();
		return entry;
	}
	for (auto &child : op->children) {
		TableCatalogEntry *entry = GetCatalogTableEntry(child.get());
		if (entry != nullptr) {
			return entry;
		}
	}
	return nullptr;
}

// The filter was made on top of a logical sample or other projection,
// but no specific columns are referenced. See issue 4978 number 4.
bool CardinalityEstimator::EmptyFilter(FilterInfo *filter_info) {
	if (!filter_info->left_set && !filter_info->right_set) {
		return true;
	}
	return false;
}

void CardinalityEstimator::AddRelationTdom(FilterInfo *filter_info) {
	D_ASSERT(filter_info->set->count >= 1);
	for (const RelationsToTDom &r2tdom : relations_to_tdoms) {
		auto &i_set = r2tdom.equivalent_relations;
		if (i_set.find(filter_info->left_binding) != i_set.end()) {
			// found an equivalent filter
			return;
		}
	}
	auto key = ColumnBinding(filter_info->left_binding.table_index, filter_info->left_binding.column_index);
	column_binding_set_t tmp({key});
	relations_to_tdoms.emplace_back(RelationsToTDom(tmp));
}

bool CardinalityEstimator::SingleColumnFilter(FilterInfo *filter_info) {
	if (filter_info->left_set && filter_info->right_set) {
		// Both set
		return false;
	}
	if (EmptyFilter(filter_info)) {
		return false;
	}
	return true;
}

vector<idx_t> CardinalityEstimator::DetermineMatchingEquivalentSets(FilterInfo *filter_info) {
	vector<idx_t> matching_equivalent_sets;
	auto equivalent_relation_index = 0;

	for (const RelationsToTDom &r2tdom : relations_to_tdoms) {
		auto &i_set = r2tdom.equivalent_relations;
		if (i_set.find(filter_info->left_binding) != i_set.end()) {
			matching_equivalent_sets.push_back(equivalent_relation_index);
		} else if (i_set.find(filter_info->right_binding) != i_set.end()) {
			// don't add both left and right to the matching_equivalent_sets
			// since both left and right get added to that index anyway.
			matching_equivalent_sets.push_back(equivalent_relation_index);
		}
		equivalent_relation_index++;
	}
	return matching_equivalent_sets;
}

void CardinalityEstimator::AddToEquivalenceSets(FilterInfo *filter_info, vector<idx_t> matching_equivalent_sets) {
	D_ASSERT(matching_equivalent_sets.size() <= 2);
	if (matching_equivalent_sets.size() > 1) {
		// an equivalence relation is connecting to sets of equivalence relations
		// so push all relations from the second set into the first. Later we will delete
		// the second set.
		for (ColumnBinding i : relations_to_tdoms.at(matching_equivalent_sets[1]).equivalent_relations) {
			relations_to_tdoms.at(matching_equivalent_sets[0]).equivalent_relations.insert(i);
		}
		relations_to_tdoms.at(matching_equivalent_sets[1]).equivalent_relations.clear();
		relations_to_tdoms.at(matching_equivalent_sets[0]).filters.push_back(filter_info);
		// add all values of one set to the other, delete the empty one
	} else if (matching_equivalent_sets.size() == 1) {
		auto &tdom_i = relations_to_tdoms.at(matching_equivalent_sets.at(0));
		tdom_i.equivalent_relations.insert(filter_info->left_binding);
		tdom_i.equivalent_relations.insert(filter_info->right_binding);
		tdom_i.filters.push_back(filter_info);
	} else if (matching_equivalent_sets.empty()) {
		column_binding_set_t tmp;
		tmp.insert(filter_info->left_binding);
		tmp.insert(filter_info->right_binding);
		relations_to_tdoms.emplace_back(RelationsToTDom(tmp));
		relations_to_tdoms.back().filters.push_back(filter_info);
	}
}

void CardinalityEstimator::AddRelationToColumnMapping(ColumnBinding key, ColumnBinding value) {
	relation_column_to_original_column[key] = value;
}

void CardinalityEstimator::CopyRelationMap(column_binding_map_t<ColumnBinding> &child_binding_map) {
	for (auto &binding_map : relation_column_to_original_column) {
		child_binding_map[binding_map.first] = binding_map.second;
	}
}

void CardinalityEstimator::AddColumnToRelationMap(idx_t table_index, idx_t column_index) {
	relation_attributes[table_index].columns.insert(column_index);
}

void CardinalityEstimator::InitEquivalentRelations(vector<unique_ptr<FilterInfo>> *filter_infos) {
	// For each filter, we fill keep track of the index of the equivalent relation set
	// the left and right relation needs to be added to.
	for (auto &filter : *filter_infos) {
		if (SingleColumnFilter(filter.get())) {
			// Filter on one relation, (i.e string or range filter on a column).
			// Grab the first relation and add it to  the equivalence_relations
			AddRelationTdom(filter.get());
			continue;
		} else if (EmptyFilter(filter.get())) {
			continue;
		}
		D_ASSERT(filter->left_set->count >= 1);
		D_ASSERT(filter->right_set->count >= 1);

		auto matching_equivalent_sets = DetermineMatchingEquivalentSets(filter.get());
		AddToEquivalenceSets(filter.get(), matching_equivalent_sets);
	}
}

void CardinalityEstimator::VerifySymmetry(JoinNode *result, JoinNode *entry) {
	if (result->GetCardinality<double>() != entry->GetCardinality<double>()) {
		// Currently it's possible that some entries are cartesian joins.
		// When this is the case, you don't always have symmetry, but
		// if the cost of the result is less, then just assure the cardinality
		// is also less, then you have the same effect of symmetry.
		D_ASSERT(ceil(result->GetCardinality<double>()) <= ceil(entry->GetCardinality<double>()) ||
		         floor(result->GetCardinality<double>()) <= floor(entry->GetCardinality<double>()));
	}
}

void CardinalityEstimator::InitTotalDomains() {
	auto remove_start = std::remove_if(relations_to_tdoms.begin(), relations_to_tdoms.end(),
	                                   [](RelationsToTDom &r_2_tdom) { return r_2_tdom.equivalent_relations.empty(); });
	relations_to_tdoms.erase(remove_start, relations_to_tdoms.end());
}

double CardinalityEstimator::ComputeCost(JoinNode *left, JoinNode *right, double expected_cardinality) {
	return expected_cardinality + left->GetCost() + right->GetCost();
}

double CardinalityEstimator::EstimateCrossProduct(const JoinNode *left, const JoinNode *right) {
	// need to explicity use double here, otherwise auto converts it to an int, then
	// there is an autocast in the return.
	return left->GetCardinality<double>() >= (NumericLimits<double>::Maximum() / right->GetCardinality<double>())
	           ? NumericLimits<double>::Maximum()
	           : left->GetCardinality<double>() * right->GetCardinality<double>();
}

void CardinalityEstimator::AddRelationColumnMapping(LogicalGet *get, idx_t relation_id) {
	for (idx_t it = 0; it < get->column_ids.size(); it++) {
		auto key = ColumnBinding(relation_id, it);
		auto value = ColumnBinding(get->table_index, get->column_ids[it]);
		AddRelationToColumnMapping(key, value);
	}
}

void UpdateDenom(Subgraph2Denominator *relation_2_denom, RelationsToTDom *relation_to_tdom) {
	relation_2_denom->denom *=
	    relation_to_tdom->has_tdom_hll ? relation_to_tdom->tdom_hll : relation_to_tdom->tdom_no_hll;
}

void FindSubgraphMatchAndMerge(Subgraph2Denominator &merge_to, idx_t find_me,
                               vector<Subgraph2Denominator>::iterator subgraph,
                               vector<Subgraph2Denominator>::iterator end) {
	for (; subgraph != end; subgraph++) {
		if (subgraph->relations.count(find_me) >= 1) {
			for (auto &relation : subgraph->relations) {
				merge_to.relations.insert(relation);
			}
			subgraph->relations.clear();
			merge_to.denom *= subgraph->denom;
			return;
		}
	}
}

double CardinalityEstimator::EstimateCardinalityWithSet(JoinRelationSet *new_set) {
	double numerator = 1;
	unordered_set<idx_t> actual_set;
	for (idx_t i = 0; i < new_set->count; i++) {
		numerator *= relation_attributes[new_set->relations[i]].cardinality;
		actual_set.insert(new_set->relations[i]);
	}
	vector<Subgraph2Denominator> subgraphs;
	bool done = false;
	bool found_match = false;

	// Finding the denominator is tricky. You need to go through the tdoms in decreasing order
	// Then loop through all filters in the equivalence set of the tdom to see if both the
	// left and right relations are in the new set, if so you can use that filter.
	// You must also make sure that the filters all relations in the given set, so we use subgraphs
	// that should eventually merge into one connected graph that joins all the relations
	// TODO: Implement a method to cache subgraphs so you don't have to build them up every
	// time the cardinality of a new set is requested

	// relations_to_tdoms has already been sorted.
	for (auto &relation_2_tdom : relations_to_tdoms) {
		// loop through each filter in the tdom.
		if (done) {
			break;
		}
		for (auto &filter : relation_2_tdom.filters) {
			if (actual_set.count(filter->left_binding.table_index) == 0 ||
			    actual_set.count(filter->right_binding.table_index) == 0) {
				continue;
			}
			// the join filter is on relations in the new set.
			found_match = false;
			vector<Subgraph2Denominator>::iterator it;
			for (it = subgraphs.begin(); it != subgraphs.end(); it++) {
				auto left_in = it->relations.count(filter->left_binding.table_index);
				auto right_in = it->relations.count(filter->right_binding.table_index);
				if (left_in && right_in) {
					// if both left and right bindings are in the subgraph, continue.
					// This means another filter is connecting relations already in the
					// subgraph it, but it has a tdom that is less, and we don't care.
					found_match = true;
					continue;
				}
				if (!left_in && !right_in) {
					// if both left and right bindings are *not* in the subgraph, continue
					// without finding a match. This will trigger the process to add a new
					// subgraph
					continue;
				}
				idx_t find_table;
				if (left_in) {
					find_table = filter->right_binding.table_index;
				} else {
					D_ASSERT(right_in);
					find_table = filter->left_binding.table_index;
				}
				auto next_subgraph = it + 1;
				// iterate through other subgraphs and merge.
				FindSubgraphMatchAndMerge(*it, find_table, next_subgraph, subgraphs.end());
				// Now insert the right binding and update denominator with the
				// tdom of the filter
				it->relations.insert(find_table);
				UpdateDenom(&(*it), &relation_2_tdom);
				found_match = true;
				break;
			}
			// means that the filter joins relations in the given set, but there is no
			// connection to any subgraph in subgraphs. Add a new subgraph, and maybe later there will be
			// a connection.
			if (!found_match) {
				subgraphs.emplace_back(Subgraph2Denominator());
				auto subgraph = &subgraphs.back();
				subgraph->relations.insert(filter->left_binding.table_index);
				subgraph->relations.insert(filter->right_binding.table_index);
				UpdateDenom(subgraph, &relation_2_tdom);
			}
			auto remove_start = std::remove_if(subgraphs.begin(), subgraphs.end(),
			                                   [](Subgraph2Denominator &s) { return s.relations.empty(); });
			subgraphs.erase(remove_start, subgraphs.end());

			if (subgraphs.size() == 1 && subgraphs.at(0).relations.size() == new_set->count) {
				// You have found enough filters to connect the relations. These are guaranteed
				// to be the filters with the highest Tdoms.
				done = true;
				break;
			}
		}
	}
	double denom = 1;
	// TODO: It's possible cross-products were added and are not present in the filters in the relation_2_tdom
	//       structures. When that's the case, multiply the denom structures that have no intersection
	for (auto &match : subgraphs) {
		// It's possible that in production, one of the D_ASSERTS above will fail and not all subgraphs
		// were connected. When this happens, just use the largest denominator of all the subgraphs.
		if (match.denom > denom) {
			denom = match.denom;
		}
	}
	// can happen if a table has cardinality 0, or a tdom is set to 0
	if (denom == 0) {
		denom = 1;
	}
	return numerator / denom;
}

static bool IsLogicalFilter(LogicalOperator *op) {
	return op->type == LogicalOperatorType::LOGICAL_FILTER;
}

static LogicalGet *GetLogicalGet(LogicalOperator *op) {
	LogicalGet *get = nullptr;
	switch (op->type) {
	case LogicalOperatorType::LOGICAL_GET:
		get = (LogicalGet *)op;
		break;
	case LogicalOperatorType::LOGICAL_FILTER:
		get = GetLogicalGet(op->children.at(0).get());
		break;
	case LogicalOperatorType::LOGICAL_PROJECTION:
		get = GetLogicalGet(op->children.at(0).get());
		break;
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN: {
		LogicalComparisonJoin *join = (LogicalComparisonJoin *)op;
		if (join->join_type == JoinType::MARK || join->join_type == JoinType::LEFT) {
			auto child = join->children.at(0).get();
			get = GetLogicalGet(child);
		}
		break;
	}
	default:
		// return null pointer, maybe there is no logical get under this child
		break;
	}
	return get;
}

void CardinalityEstimator::MergeBindings(idx_t binding_index, idx_t relation_id,
                                         vector<column_binding_map_t<ColumnBinding>> &child_binding_maps) {
	for (auto &map_set : child_binding_maps) {
		for (auto &mapping : map_set) {
			ColumnBinding relation_bindings = mapping.first;
			ColumnBinding actual_bindings = mapping.second;

			if (actual_bindings.table_index == binding_index) {
				auto key = ColumnBinding(relation_id, relation_bindings.column_index);
				AddRelationToColumnMapping(key, actual_bindings);
			}
		}
	}
}

bool SortTdoms(const RelationsToTDom &a, const RelationsToTDom &b) {
	if (a.has_tdom_hll && b.has_tdom_hll) {
		return a.tdom_hll > b.tdom_hll;
	}
	if (a.has_tdom_hll) {
		return a.tdom_hll > b.tdom_no_hll;
	}
	if (b.has_tdom_hll) {
		return a.tdom_no_hll > b.tdom_hll;
	}
	return a.tdom_no_hll > b.tdom_no_hll;
}

void CardinalityEstimator::InitCardinalityEstimatorProps(vector<struct NodeOp> *node_ops,
                                                         vector<unique_ptr<FilterInfo>> *filter_infos) {
	InitEquivalentRelations(filter_infos);
	InitTotalDomains();
	for (idx_t i = 0; i < node_ops->size(); i++) {
		auto join_node = (*node_ops)[i].node.get();
		auto op = (*node_ops)[i].op;
		join_node->SetBaseTableCardinality(op->EstimateCardinality(context));
		if (op->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
			auto &join = (LogicalComparisonJoin &)*op;
			if (join.join_type == JoinType::LEFT) {
				// TODO: inspect child operators to get a more accurate cost
				// and cardinality estimation. If an base op is a Logical Comparison join
				// it is probably a left join, so cost of the larger table is a fine
				// estimate
				// No need to update a mark join cost because I say so.
				join_node->SetCost(join_node->GetBaseTableCardinality());
			}
		}
		// update cardinality with filters
		EstimateBaseTableCardinality(join_node, op);
		UpdateTotalDomains(join_node, op);
	}

	// sort relations from greatest tdom to lowest tdom.
	std::sort(relations_to_tdoms.begin(), relations_to_tdoms.end(), SortTdoms);
}

void CardinalityEstimator::UpdateTotalDomains(JoinNode *node, LogicalOperator *op) {
	auto relation_id = node->set->relations[0];
	relation_attributes[relation_id].cardinality = node->GetCardinality<double>();
	TableCatalogEntry *catalog_table = nullptr;
	auto get = GetLogicalGet(op);
	if (get) {
		catalog_table = GetCatalogTableEntry(get);
	}

	//! Initialize the tdoms for all columns the relation uses in join conditions.
	unordered_set<idx_t>::iterator ite;
	idx_t count = node->GetBaseTableCardinality();

	bool direct_filter = false;
	for (auto &column : relation_attributes[relation_id].columns) {
		//! for every column in the relation, get the count via either HLL, or assume it to be
		//! the cardinality
		ColumnBinding key = ColumnBinding(relation_id, column);

		if (catalog_table) {
			relation_attributes[relation_id].original_name = catalog_table->name;
			// Get HLL stats here
			auto actual_binding = relation_column_to_original_column[key];

			auto base_stats = catalog_table->GetStatistics(context, actual_binding.column_index);
			if (base_stats) {
				count = base_stats->GetDistinctCount();
			}

			// means you have a direct filter on a column. The count/total domain for the column
			// should be decreased to match the predicted total domain matching the filter.
			// We decrease the total domain for all columns in the equivalence set because filter pushdown
			// will mean all columns are affected.
			if (direct_filter) {
				count = node->GetCardinality<idx_t>();
			}

			// HLL has estimation error, count can't be greater than cardinality of the table before filters
			if (count > node->GetBaseTableCardinality()) {
				count = node->GetBaseTableCardinality();
			}
		} else {
			// No HLL. So if we know there is a direct filter, reduce count to cardinality with filter
			// otherwise assume the total domain is still the cardinality
			if (direct_filter) {
				count = node->GetCardinality<idx_t>();
			} else {
				count = node->GetBaseTableCardinality();
			}
		}

		for (auto &relation_to_tdom : relations_to_tdoms) {
			column_binding_set_t i_set = relation_to_tdom.equivalent_relations;
			if (i_set.count(key) != 1) {
				continue;
			}
			if (catalog_table) {
				if (relation_to_tdom.tdom_hll < count) {
					relation_to_tdom.tdom_hll = count;
					relation_to_tdom.has_tdom_hll = true;
				}
				if (relation_to_tdom.tdom_no_hll > count) {
					relation_to_tdom.tdom_no_hll = count;
				}
			} else {
				// Here we don't have catalog statistics, and the following is how we determine
				// the tdom
				// 1. If there is any hll data in the equivalence set, use that
				// 2. Otherwise, use the table with the smallest cardinality
				if (relation_to_tdom.tdom_no_hll > count && !relation_to_tdom.has_tdom_hll) {
					relation_to_tdom.tdom_no_hll = count;
				}
			}
			break;
		}
	}
}

TableFilterSet *CardinalityEstimator::GetTableFilters(LogicalOperator *op) {
	// First check table filters
	auto get = GetLogicalGet(op);
	return get ? &get->table_filters : nullptr;
}

idx_t CardinalityEstimator::InspectConjunctionAND(idx_t cardinality, idx_t column_index, ConjunctionAndFilter *filter,
                                                  unique_ptr<BaseStatistics> base_stats) {
	auto has_equality_filter = false;
	auto cardinality_after_filters = cardinality;
	for (auto &child_filter : filter->child_filters) {
		if (child_filter->filter_type != TableFilterType::CONSTANT_COMPARISON) {
			continue;
		}
		auto comparison_filter = (ConstantFilter &)*child_filter;
		if (comparison_filter.comparison_type != ExpressionType::COMPARE_EQUAL) {
			continue;
		}
		auto column_count = 0;
		if (base_stats) {
			column_count = base_stats->GetDistinctCount();
		}
		auto filtered_card = cardinality;
		// column_count = 0 when there is no column count (i.e parquet scans)
		if (column_count > 0) {
			// we want the ceil of cardinality/column_count. We also want to avoid compiler errors
			filtered_card = (cardinality + column_count - 1) / column_count;
			cardinality_after_filters = filtered_card;
		}
		if (has_equality_filter) {
			cardinality_after_filters = MinValue(filtered_card, cardinality_after_filters);
		}
		has_equality_filter = true;
	}
	return cardinality_after_filters;
}

idx_t CardinalityEstimator::InspectConjunctionOR(idx_t cardinality, idx_t column_index, ConjunctionOrFilter *filter,
                                                 unique_ptr<BaseStatistics> base_stats) {
	auto has_equality_filter = false;
	auto cardinality_after_filters = cardinality;
	for (auto &child_filter : filter->child_filters) {
		if (child_filter->filter_type != TableFilterType::CONSTANT_COMPARISON) {
			continue;
		}
		auto comparison_filter = (ConstantFilter &)*child_filter;
		if (comparison_filter.comparison_type == ExpressionType::COMPARE_EQUAL) {
			auto column_count = cardinality_after_filters;
			if (base_stats) {
				column_count = base_stats->GetDistinctCount();
			}
			auto increment = MaxValue<idx_t>(((cardinality + column_count - 1) / column_count), 1);
			if (has_equality_filter) {
				cardinality_after_filters += increment;
			} else {
				cardinality_after_filters = increment;
			}
			has_equality_filter = true;
		}
	}
	D_ASSERT(cardinality_after_filters > 0);
	return cardinality_after_filters;
}

idx_t CardinalityEstimator::InspectTableFilters(idx_t cardinality, LogicalOperator *op, TableFilterSet *table_filters) {
	idx_t cardinality_after_filters = cardinality;
	auto get = GetLogicalGet(op);
	unique_ptr<BaseStatistics> column_statistics;
	for (auto &it : table_filters->filters) {
		column_statistics = nullptr;
		if (get->bind_data && get->function.name.compare("seq_scan") == 0) {
			auto &table_scan_bind_data = (TableScanBindData &)*get->bind_data;
			column_statistics = get->function.statistics(context, &table_scan_bind_data, it.first);
		}
		if (it.second->filter_type == TableFilterType::CONJUNCTION_AND) {
			auto &filter = (ConjunctionAndFilter &)*it.second;
			idx_t cardinality_with_and_filter =
			    InspectConjunctionAND(cardinality, it.first, &filter, move(column_statistics));
			cardinality_after_filters = MinValue(cardinality_after_filters, cardinality_with_and_filter);
		} else if (it.second->filter_type == TableFilterType::CONJUNCTION_OR) {
			auto &filter = (ConjunctionOrFilter &)*it.second;
			idx_t cardinality_with_or_filter =
			    InspectConjunctionOR(cardinality, it.first, &filter, move(column_statistics));
			cardinality_after_filters = MinValue(cardinality_after_filters, cardinality_with_or_filter);
		}
	}
	// if the above code didn't find an equality filter (i.e country_code = "[us]")
	// and there are other table filters, use default selectivity.
	bool has_equality_filter = (cardinality_after_filters != cardinality);
	if (!has_equality_filter && !table_filters->filters.empty()) {
		cardinality_after_filters = MaxValue<idx_t>(cardinality * DEFAULT_SELECTIVITY, 1);
	}
	return cardinality_after_filters;
}

void CardinalityEstimator::EstimateBaseTableCardinality(JoinNode *node, LogicalOperator *op) {
	auto has_logical_filter = IsLogicalFilter(op);
	auto table_filters = GetTableFilters(op);

	auto card_after_filters = node->GetBaseTableCardinality();
	// Logical Filter on a seq scan
	if (has_logical_filter) {
		card_after_filters *= DEFAULT_SELECTIVITY;
	} else if (table_filters) {
		double inspect_result = (double)InspectTableFilters(card_after_filters, op, table_filters);
		card_after_filters = MinValue(inspect_result, (double)card_after_filters);
	}
	node->SetEstimatedCardinality(card_after_filters);
}

} // namespace duckdb








namespace duckdb {

void ColumnLifetimeAnalyzer::ExtractUnusedColumnBindings(vector<ColumnBinding> bindings,
                                                         column_binding_set_t &unused_bindings) {
	for (idx_t i = 0; i < bindings.size(); i++) {
		if (column_references.find(bindings[i]) == column_references.end()) {
			unused_bindings.insert(bindings[i]);
		}
	}
}

void ColumnLifetimeAnalyzer::GenerateProjectionMap(vector<ColumnBinding> bindings,
                                                   column_binding_set_t &unused_bindings,
                                                   vector<idx_t> &projection_map) {
	if (unused_bindings.empty()) {
		return;
	}
	// now iterate over the result bindings of the child
	for (idx_t i = 0; i < bindings.size(); i++) {
		// if this binding does not belong to the unused bindings, add it to the projection map
		if (unused_bindings.find(bindings[i]) == unused_bindings.end()) {
			projection_map.push_back(i);
		}
	}
	if (projection_map.size() == bindings.size()) {
		projection_map.clear();
	}
}

void ColumnLifetimeAnalyzer::StandardVisitOperator(LogicalOperator &op) {
	LogicalOperatorVisitor::VisitOperatorExpressions(op);
	if (op.type == LogicalOperatorType::LOGICAL_DELIM_JOIN) {
		// visit the duplicate eliminated columns on the LHS, if any
		auto &delim_join = (LogicalDelimJoin &)op;
		for (auto &expr : delim_join.duplicate_eliminated_columns) {
			VisitExpression(&expr);
		}
	}
	LogicalOperatorVisitor::VisitOperatorChildren(op);
}

void ColumnLifetimeAnalyzer::VisitOperator(LogicalOperator &op) {
	switch (op.type) {
	case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY: {
		// FIXME: groups that are not referenced can be removed from projection
		// recurse into the children of the aggregate
		ColumnLifetimeAnalyzer analyzer;
		analyzer.VisitOperatorExpressions(op);
		analyzer.VisitOperator(*op.children[0]);
		return;
	}
	case LogicalOperatorType::LOGICAL_DELIM_JOIN:
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN: {
		if (everything_referenced) {
			break;
		}
		auto &comp_join = (LogicalComparisonJoin &)op;
		if (comp_join.join_type == JoinType::MARK || comp_join.join_type == JoinType::SEMI ||
		    comp_join.join_type == JoinType::ANTI) {
			break;
		}
		// FIXME for now, we only push into the projection map for equality (hash) joins
		// FIXME: add projection to LHS as well
		bool has_equality = false;
		for (auto &cond : comp_join.conditions) {
			if (cond.comparison == ExpressionType::COMPARE_EQUAL) {
				has_equality = true;
			}
		}
		if (!has_equality) {
			break;
		}
		// now, for each of the columns of the RHS, check which columns need to be projected
		column_binding_set_t unused_bindings;
		ExtractUnusedColumnBindings(op.children[1]->GetColumnBindings(), unused_bindings);

		// now recurse into the filter and its children
		StandardVisitOperator(op);

		// then generate the projection map
		GenerateProjectionMap(op.children[1]->GetColumnBindings(), unused_bindings, comp_join.right_projection_map);
		return;
	}
	case LogicalOperatorType::LOGICAL_UNION:
	case LogicalOperatorType::LOGICAL_EXCEPT:
	case LogicalOperatorType::LOGICAL_INTERSECT:
		// for set operations we don't remove anything, just recursively visit the children
		// FIXME: for UNION we can remove unreferenced columns as long as everything_referenced is false (i.e. we
		// encounter a UNION node that is not preceded by a DISTINCT)
		for (auto &child : op.children) {
			ColumnLifetimeAnalyzer analyzer(true);
			analyzer.VisitOperator(*child);
		}
		return;
	case LogicalOperatorType::LOGICAL_PROJECTION: {
		// then recurse into the children of this projection
		ColumnLifetimeAnalyzer analyzer;
		analyzer.VisitOperatorExpressions(op);
		analyzer.VisitOperator(*op.children[0]);
		return;
	}
	case LogicalOperatorType::LOGICAL_DISTINCT: {
		// distinct, all projected columns are used for the DISTINCT computation
		// mark all columns as used and continue to the children
		// FIXME: DISTINCT with expression list does not implicitly reference everything
		everything_referenced = true;
		break;
	}
	case LogicalOperatorType::LOGICAL_FILTER: {
		auto &filter = (LogicalFilter &)op;
		if (everything_referenced) {
			break;
		}
		// filter, figure out which columns are not needed after the filter
		column_binding_set_t unused_bindings;
		ExtractUnusedColumnBindings(op.children[0]->GetColumnBindings(), unused_bindings);

		// now recurse into the filter and its children
		StandardVisitOperator(op);

		// then generate the projection map
		GenerateProjectionMap(op.children[0]->GetColumnBindings(), unused_bindings, filter.projection_map);
		return;
	}
	default:
		break;
	}
	StandardVisitOperator(op);
}

unique_ptr<Expression> ColumnLifetimeAnalyzer::VisitReplace(BoundColumnRefExpression &expr,
                                                            unique_ptr<Expression> *expr_ptr) {
	column_references.insert(expr.binding);
	return nullptr;
}

unique_ptr<Expression> ColumnLifetimeAnalyzer::VisitReplace(BoundReferenceExpression &expr,
                                                            unique_ptr<Expression> *expr_ptr) {
	// BoundReferenceExpression should not be used here yet, they only belong in the physical plan
	throw InternalException("BoundReferenceExpression should not be used here yet!");
}

} // namespace duckdb







namespace duckdb {

void CommonAggregateOptimizer::VisitOperator(LogicalOperator &op) {
	LogicalOperatorVisitor::VisitOperator(op);
	switch (op.type) {
	case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
		ExtractCommonAggregates((LogicalAggregate &)op);
		break;
	default:
		break;
	}
}

unique_ptr<Expression> CommonAggregateOptimizer::VisitReplace(BoundColumnRefExpression &expr,
                                                              unique_ptr<Expression> *expr_ptr) {
	// check if this column ref points to an aggregate that was remapped; if it does we remap it
	auto entry = aggregate_map.find(expr.binding);
	if (entry != aggregate_map.end()) {
		expr.binding = entry->second;
	}
	return nullptr;
}

void CommonAggregateOptimizer::ExtractCommonAggregates(LogicalAggregate &aggr) {
	expression_map_t<idx_t> aggregate_remap;
	idx_t total_erased = 0;
	for (idx_t i = 0; i < aggr.expressions.size(); i++) {
		idx_t original_index = i + total_erased;
		auto entry = aggregate_remap.find(aggr.expressions[i].get());
		if (entry == aggregate_remap.end()) {
			// aggregate does not exist yet: add it to the map
			aggregate_remap[aggr.expressions[i].get()] = i;
			if (i != original_index) {
				// this aggregate is not erased, however an agregate BEFORE it has been erased
				// so we need to remap this aggregaet
				ColumnBinding original_binding(aggr.aggregate_index, original_index);
				ColumnBinding new_binding(aggr.aggregate_index, i);
				aggregate_map[original_binding] = new_binding;
			}
		} else {
			// aggregate already exists! we can remove this entry
			total_erased++;
			aggr.expressions.erase(aggr.expressions.begin() + i);
			i--;
			// we need to remap any references to this aggregate so they point to the other aggregate
			ColumnBinding original_binding(aggr.aggregate_index, original_index);
			ColumnBinding new_binding(aggr.aggregate_index, entry->second);
			aggregate_map[original_binding] = new_binding;
		}
	}
}

} // namespace duckdb









namespace duckdb {

//! The CSENode contains information about a common subexpression; how many times it occurs, and the column index in the
//! underlying projection
struct CSENode {
	idx_t count;
	idx_t column_index;

	CSENode() : count(1), column_index(DConstants::INVALID_INDEX) {
	}
};

//! The CSEReplacementState
struct CSEReplacementState {
	//! The projection index of the new projection
	idx_t projection_index;
	//! Map of expression -> CSENode
	expression_map_t<CSENode> expression_count;
	//! Map of column bindings to column indexes in the projection expression list
	column_binding_map_t<idx_t> column_map;
	//! The set of expressions of the resulting projection
	vector<unique_ptr<Expression>> expressions;
	//! Cached expressions that are kept around so the expression_map always contains valid expressions
	vector<unique_ptr<Expression>> cached_expressions;
};

void CommonSubExpressionOptimizer::VisitOperator(LogicalOperator &op) {
	switch (op.type) {
	case LogicalOperatorType::LOGICAL_PROJECTION:
	case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
		ExtractCommonSubExpresions(op);
		break;
	default:
		break;
	}
	LogicalOperatorVisitor::VisitOperator(op);
}

void CommonSubExpressionOptimizer::CountExpressions(Expression &expr, CSEReplacementState &state) {
	// we only consider expressions with children for CSE elimination
	switch (expr.expression_class) {
	case ExpressionClass::BOUND_COLUMN_REF:
	case ExpressionClass::BOUND_CONSTANT:
	case ExpressionClass::BOUND_PARAMETER:
	// skip conjunctions and case, since short-circuiting might be incorrectly disabled otherwise
	case ExpressionClass::BOUND_CONJUNCTION:
	case ExpressionClass::BOUND_CASE:
		return;
	default:
		break;
	}
	if (expr.expression_class != ExpressionClass::BOUND_AGGREGATE && !expr.HasSideEffects()) {
		// we can't move aggregates to a projection, so we only consider the children of the aggregate
		auto node = state.expression_count.find(&expr);
		if (node == state.expression_count.end()) {
			// first time we encounter this expression, insert this node with [count = 1]
			state.expression_count[&expr] = CSENode();
		} else {
			// we encountered this expression before, increment the occurrence count
			node->second.count++;
		}
	}
	// recursively count the children
	ExpressionIterator::EnumerateChildren(expr, [&](Expression &child) { CountExpressions(child, state); });
}

void CommonSubExpressionOptimizer::PerformCSEReplacement(unique_ptr<Expression> *expr_ptr, CSEReplacementState &state) {
	Expression &expr = **expr_ptr;
	if (expr.expression_class == ExpressionClass::BOUND_COLUMN_REF) {
		auto &bound_column_ref = (BoundColumnRefExpression &)expr;
		// bound column ref, check if this one has already been recorded in the expression list
		auto column_entry = state.column_map.find(bound_column_ref.binding);
		if (column_entry == state.column_map.end()) {
			// not there yet: push the expression
			idx_t new_column_index = state.expressions.size();
			state.column_map[bound_column_ref.binding] = new_column_index;
			state.expressions.push_back(make_unique<BoundColumnRefExpression>(
			    bound_column_ref.alias, bound_column_ref.return_type, bound_column_ref.binding));
			bound_column_ref.binding = ColumnBinding(state.projection_index, new_column_index);
		} else {
			// else: just update the column binding!
			bound_column_ref.binding = ColumnBinding(state.projection_index, column_entry->second);
		}
		return;
	}
	// check if this child is eligible for CSE elimination
	bool can_cse = expr.expression_class != ExpressionClass::BOUND_CONJUNCTION &&
	               expr.expression_class != ExpressionClass::BOUND_CASE;
	if (can_cse && state.expression_count.find(&expr) != state.expression_count.end()) {
		auto &node = state.expression_count[&expr];
		if (node.count > 1) {
			// this expression occurs more than once! push it into the projection
			// check if it has already been pushed into the projection
			auto alias = expr.alias;
			auto type = expr.return_type;
			if (node.column_index == DConstants::INVALID_INDEX) {
				// has not been pushed yet: push it
				node.column_index = state.expressions.size();
				state.expressions.push_back(move(*expr_ptr));
			} else {
				state.cached_expressions.push_back(move(*expr_ptr));
			}
			// replace the original expression with a bound column ref
			*expr_ptr = make_unique<BoundColumnRefExpression>(alias, type,
			                                                  ColumnBinding(state.projection_index, node.column_index));
			return;
		}
	}
	// this expression only occurs once, we can't perform CSE elimination
	// look into the children to see if we can replace them
	ExpressionIterator::EnumerateChildren(expr,
	                                      [&](unique_ptr<Expression> &child) { PerformCSEReplacement(&child, state); });
}

void CommonSubExpressionOptimizer::ExtractCommonSubExpresions(LogicalOperator &op) {
	D_ASSERT(op.children.size() == 1);

	// first we count for each expression with children how many types it occurs
	CSEReplacementState state;
	LogicalOperatorVisitor::EnumerateExpressions(
	    op, [&](unique_ptr<Expression> *child) { CountExpressions(**child, state); });
	// check if there are any expressions to extract
	bool perform_replacement = false;
	for (auto &expr : state.expression_count) {
		if (expr.second.count > 1) {
			perform_replacement = true;
			break;
		}
	}
	if (!perform_replacement) {
		// no CSEs to extract
		return;
	}
	state.projection_index = binder.GenerateTableIndex();
	// we found common subexpressions to extract
	// now we iterate over all the expressions and perform the actual CSE elimination
	LogicalOperatorVisitor::EnumerateExpressions(
	    op, [&](unique_ptr<Expression> *child) { PerformCSEReplacement(child, state); });
	D_ASSERT(state.expressions.size() > 0);
	// create a projection node as the child of this node
	auto projection = make_unique<LogicalProjection>(state.projection_index, move(state.expressions));
	projection->children.push_back(move(op.children[0]));
	op.children[0] = move(projection);
}

} // namespace duckdb











namespace duckdb {

class DeliminatorPlanUpdater : LogicalOperatorVisitor {
public:
	DeliminatorPlanUpdater() {
	}
	//! Update the plan after a DelimGet has been removed
	void VisitOperator(LogicalOperator &op) override;
	void VisitExpression(unique_ptr<Expression> *expression) override;

public:
	expression_map_t<Expression *> expr_map;
	column_binding_map_t<bool> projection_map;
	column_binding_map_t<Expression *> reverse_proj_or_agg_map;
	unique_ptr<LogicalOperator> temp_ptr;
};

static idx_t DelimGetCount(LogicalOperator &op) {
	if (op.type == LogicalOperatorType::LOGICAL_DELIM_GET) {
		return 1;
	}
	idx_t child_count = 0;
	for (auto &child : op.children) {
		child_count += DelimGetCount(*child);
	}
	return child_count;
}

static bool IsEqualityJoinCondition(JoinCondition &cond) {
	switch (cond.comparison) {
	case ExpressionType::COMPARE_EQUAL:
	case ExpressionType::COMPARE_NOT_DISTINCT_FROM:
		return true;
	default:
		return false;
	}
}

static bool InequalityDelimJoinCanBeEliminated(JoinType &join_type) {
	switch (join_type) {
	case JoinType::ANTI:
	case JoinType::MARK:
	case JoinType::SEMI:
	case JoinType::SINGLE:
		return true;
	default:
		return false;
	}
}

void DeliminatorPlanUpdater::VisitOperator(LogicalOperator &op) {
	VisitOperatorChildren(op);
	VisitOperatorExpressions(op);
	if (op.type == LogicalOperatorType::LOGICAL_DELIM_JOIN && DelimGetCount(op) == 0) {
		auto &delim_join = (LogicalDelimJoin &)op;
		auto decs = &delim_join.duplicate_eliminated_columns;
		for (auto &cond : delim_join.conditions) {
			if (!IsEqualityJoinCondition(cond)) {
				continue;
			}
			auto rhs = cond.right.get();
			while (rhs->type == ExpressionType::OPERATOR_CAST) {
				auto &cast = (BoundCastExpression &)*rhs;
				rhs = cast.child.get();
			}
			if (rhs->type != ExpressionType::BOUND_COLUMN_REF) {
				throw InternalException("Error in Deliminator: expected a bound column reference");
			}
			auto &colref = (BoundColumnRefExpression &)*rhs;
			if (projection_map.find(colref.binding) != projection_map.end()) {
				// value on the right is a projection of removed DelimGet
				for (idx_t i = 0; i < decs->size(); i++) {
					if (decs->at(i)->Equals(cond.left.get())) {
						// the value on the left no longer needs to be a duplicate-eliminated column
						decs->erase(decs->begin() + i);
						break;
					}
				}
				// whether we applied an IS NOT NULL filter
				cond.comparison = ExpressionType::COMPARE_NOT_DISTINCT_FROM;
			}
		}
		// change type if there are no more duplicate-eliminated columns
		if (decs->empty()) {
			delim_join.type = LogicalOperatorType::LOGICAL_COMPARISON_JOIN;
		}
	}
}

void DeliminatorPlanUpdater::VisitExpression(unique_ptr<Expression> *expression) {
	if (expr_map.find(expression->get()) != expr_map.end()) {
		*expression = expr_map[expression->get()]->Copy();
	} else {
		VisitExpressionChildren(**expression);
	}
}

unique_ptr<LogicalOperator> Deliminator::Optimize(unique_ptr<LogicalOperator> op) {
	vector<unique_ptr<LogicalOperator> *> candidates;
	FindCandidates(&op, candidates);

	for (auto &candidate : candidates) {
		DeliminatorPlanUpdater updater;
		if (RemoveCandidate(&op, candidate, updater)) {
			updater.VisitOperator(*op);
		}
	}
	return op;
}

void Deliminator::FindCandidates(unique_ptr<LogicalOperator> *op_ptr,
                                 vector<unique_ptr<LogicalOperator> *> &candidates) {
	auto op = op_ptr->get();
	// search children before adding, so the deepest candidates get added first
	for (auto &child : op->children) {
		FindCandidates(&child, candidates);
	}
	// search for projection/aggregate
	if (op->type != LogicalOperatorType::LOGICAL_PROJECTION &&
	    op->type != LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		return;
	}
	// followed by a join
	if (op->children[0]->type != LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
		return;
	}
	auto &join = *op->children[0];
	// with a DelimGet as a direct child (left or right)
	if (join.children[0]->type == LogicalOperatorType::LOGICAL_DELIM_GET ||
	    join.children[1]->type == LogicalOperatorType::LOGICAL_DELIM_GET) {
		candidates.push_back(op_ptr);
		return;
	}
	// or a filter followed by a DelimGet (left)
	if (join.children[0]->type == LogicalOperatorType::LOGICAL_FILTER &&
	    join.children[0]->children[0]->type == LogicalOperatorType::LOGICAL_DELIM_GET) {
		candidates.push_back(op_ptr);
		return;
	}
	// filter followed by a DelimGet (right)
	if (join.children[1]->type == LogicalOperatorType::LOGICAL_FILTER &&
	    join.children[1]->children[0]->type == LogicalOperatorType::LOGICAL_DELIM_GET) {
		candidates.push_back(op_ptr);
		return;
	}
}

static bool OperatorIsDelimGet(LogicalOperator &op) {
	if (op.type == LogicalOperatorType::LOGICAL_DELIM_GET) {
		return true;
	}
	if (op.type == LogicalOperatorType::LOGICAL_FILTER &&
	    op.children[0]->type == LogicalOperatorType::LOGICAL_DELIM_GET) {
		return true;
	}
	return false;
}

static bool ChildJoinTypeCanBeDeliminated(JoinType &join_type) {
	switch (join_type) {
	case JoinType::INNER:
	case JoinType::SEMI:
		return true;
	default:
		return false;
	}
}

bool Deliminator::RemoveCandidate(unique_ptr<LogicalOperator> *plan, unique_ptr<LogicalOperator> *candidate,
                                  DeliminatorPlanUpdater &updater) {
	auto &proj_or_agg = **candidate;
	auto &join = (LogicalComparisonJoin &)*proj_or_agg.children[0];
	if (!ChildJoinTypeCanBeDeliminated(join.join_type)) {
		return false;
	}

	// get the index (left or right) of the DelimGet side of the join
	idx_t delim_idx = OperatorIsDelimGet(*join.children[0]) ? 0 : 1;
	D_ASSERT(OperatorIsDelimGet(*join.children[delim_idx]));
	// get the filter (if any)
	LogicalFilter *filter = nullptr;
	if (join.children[delim_idx]->type == LogicalOperatorType::LOGICAL_FILTER) {
		filter = (LogicalFilter *)join.children[delim_idx].get();
	}
	auto &delim_get = (LogicalDelimGet &)*(filter ? filter->children[0].get() : join.children[delim_idx].get());
	if (join.conditions.size() != delim_get.chunk_types.size()) {
		// joining with DelimGet adds new information
		return false;
	}
	// check if joining with the DelimGet is redundant, and collect relevant column information
	bool all_equality_conditions = true;
	vector<Expression *> nulls_are_not_equal_exprs;
	for (auto &cond : join.conditions) {
		all_equality_conditions = all_equality_conditions && IsEqualityJoinCondition(cond);
		auto delim_side = delim_idx == 0 ? cond.left.get() : cond.right.get();
		auto other_side = delim_idx == 0 ? cond.right.get() : cond.left.get();
		if (delim_side->type != ExpressionType::BOUND_COLUMN_REF) {
			// non-colref e.g. expression -(4, 1) in 4-i=j where i is from DelimGet
			// FIXME: might be possible to also eliminate these
			return false;
		}
		updater.expr_map[delim_side] = other_side;
		if (cond.comparison != ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
			nulls_are_not_equal_exprs.push_back(other_side);
		}
	}

	// removed DelimGet columns are assigned a new ColumnBinding by Projection/Aggregation, keep track here
	if (proj_or_agg.type == LogicalOperatorType::LOGICAL_PROJECTION) {
		for (auto &cb : proj_or_agg.GetColumnBindings()) {
			updater.projection_map[cb] = true;
			updater.reverse_proj_or_agg_map[cb] = proj_or_agg.expressions[cb.column_index].get();
			for (auto &expr : nulls_are_not_equal_exprs) {
				if (proj_or_agg.expressions[cb.column_index]->Equals(expr)) {
					updater.projection_map[cb] = false;
					break;
				}
			}
		}
	} else {
		auto &agg = (LogicalAggregate &)proj_or_agg;

		// Create a vector of all exprs in the agg
		vector<Expression *> all_agg_exprs;
		for (auto &expr : agg.groups) {
			all_agg_exprs.push_back(expr.get());
		}
		for (auto &expr : agg.expressions) {
			all_agg_exprs.push_back(expr.get());
		}

		for (auto &cb : agg.GetColumnBindings()) {
			updater.projection_map[cb] = true;
			updater.reverse_proj_or_agg_map[cb] = all_agg_exprs[cb.column_index];
			for (auto &expr : nulls_are_not_equal_exprs) {
				if ((cb.table_index == agg.group_index && agg.groups[cb.column_index]->Equals(expr)) ||
				    (cb.table_index == agg.aggregate_index && agg.expressions[cb.column_index]->Equals(expr))) {
					updater.projection_map[cb] = false;
					break;
				}
			}
		}
	}

	if (!all_equality_conditions) {
		// we can get rid of an inequality join with a DelimGet, but only under specific circumstances
		if (!RemoveInequalityCandidate(plan, candidate, updater)) {
			return false;
		}
	}

	// make a filter if needed
	if (!nulls_are_not_equal_exprs.empty() || filter != nullptr) {
		auto filter_op = make_unique<LogicalFilter>();
		if (!nulls_are_not_equal_exprs.empty()) {
			// add an IS NOT NULL filter that was implicitly in JoinCondition::null_values_are_equal
			for (auto &expr : nulls_are_not_equal_exprs) {
				auto is_not_null_expr =
				    make_unique<BoundOperatorExpression>(ExpressionType::OPERATOR_IS_NOT_NULL, LogicalType::BOOLEAN);
				is_not_null_expr->children.push_back(expr->Copy());
				filter_op->expressions.push_back(move(is_not_null_expr));
			}
		}
		if (filter != nullptr) {
			for (auto &expr : filter->expressions) {
				filter_op->expressions.push_back(move(expr));
			}
		}
		filter_op->children.push_back(move(join.children[1 - delim_idx]));
		join.children[1 - delim_idx] = move(filter_op);
	}
	// temporarily save deleted operator so its expressions are still available
	updater.temp_ptr = move(proj_or_agg.children[0]);
	// replace the redundant join
	proj_or_agg.children[0] = move(join.children[1 - delim_idx]);
	return true;
}

static void GetDelimJoins(LogicalOperator &op, vector<LogicalOperator *> &delim_joins) {
	for (auto &child : op.children) {
		GetDelimJoins(*child, delim_joins);
	}
	if (op.type == LogicalOperatorType::LOGICAL_DELIM_JOIN) {
		delim_joins.push_back(&op);
	}
}

static bool HasChild(LogicalOperator *haystack, LogicalOperator *needle, idx_t &side) {
	if (haystack == needle) {
		return true;
	}
	for (idx_t i = 0; i < haystack->children.size(); i++) {
		auto &child = haystack->children[i];
		idx_t dummy_side;
		if (HasChild(child.get(), needle, dummy_side)) {
			side = i;
			return true;
		}
	}
	return false;
}

bool Deliminator::RemoveInequalityCandidate(unique_ptr<LogicalOperator> *plan, unique_ptr<LogicalOperator> *candidate,
                                            DeliminatorPlanUpdater &updater) {
	auto &proj_or_agg = **candidate;
	// first, we find a DelimJoin in "plan" that has only one DelimGet as a child, which is in "candidate"
	if (DelimGetCount(proj_or_agg) != 1) {
		// the candidate therefore must have only a single DelimGet in its children
		return false;
	}

	vector<LogicalOperator *> delim_joins;
	GetDelimJoins(**plan, delim_joins);

	LogicalOperator *parent = nullptr;
	idx_t parent_delim_get_side;
	for (auto dj : delim_joins) {
		D_ASSERT(dj->type == LogicalOperatorType::LOGICAL_DELIM_JOIN);
		if (!HasChild(dj, &proj_or_agg, parent_delim_get_side)) {
			continue;
		}
		// we found a parent DelimJoin
		if (DelimGetCount(*dj) != 1) {
			// it has more than one DelimGet children
			continue;
		}

		// we can only remove inequality join with a DelimGet if the parent DelimJoin has one of these join types
		auto &delim_join = (LogicalDelimJoin &)*dj;
		if (!InequalityDelimJoinCanBeEliminated(delim_join.join_type)) {
			continue;
		}

		parent = dj;
		break;
	}
	if (!parent) {
		return false;
	}

	// we found the parent delim join, and we may be able to remove the child DelimGet join
	// but we need to make sure that their conditions refer to exactly the same columns
	auto &parent_delim_join = (LogicalDelimJoin &)*parent;
	auto &join = (LogicalComparisonJoin &)*proj_or_agg.children[0];
	if (parent_delim_join.conditions.size() != join.conditions.size()) {
		// different number of conditions, can't replace
		return false;
	}

	// we can only do this optimization under the following conditions:
	// 1. all join expressions coming from the DelimGet side are colrefs
	// 2. these expressions refer to colrefs coming from the proj/agg on top of the child DelimGet join
	// 3. the expression (before it was proj/agg) can be found in the conditions of the child DelimGet join
	for (auto &parent_cond : parent_delim_join.conditions) {
		auto &parent_expr = parent_delim_get_side == 0 ? parent_cond.left : parent_cond.right;
		if (parent_expr->type != ExpressionType::BOUND_COLUMN_REF) {
			// can only deal with colrefs
			return false;
		}
		auto &parent_colref = (BoundColumnRefExpression &)*parent_expr;
		auto it = updater.reverse_proj_or_agg_map.find(parent_colref.binding);
		if (it == updater.reverse_proj_or_agg_map.end()) {
			// refers to a column that was not in the child DelimGet join
			return false;
		}
		// try to find the corresponding child condition
		// TODO: can be more flexible - allow CAST
		auto child_expr = it->second;
		bool found = false;
		for (auto &child_cond : join.conditions) {
			if (child_cond.left->Equals(child_expr) || child_cond.right->Equals(child_expr)) {
				found = true;
				break;
			}
		}
		if (!found) {
			// could not find the mapped expression in the child condition expressions
			return false;
		}
	}

	// TODO: we cannot perform the optimization here because our pure inequality joins don't implement
	//  JoinType::SINGLE yet
	if (parent_delim_join.join_type == JoinType::SINGLE) {
		bool has_one_equality = false;
		for (auto &cond : join.conditions) {
			has_one_equality = has_one_equality || IsEqualityJoinCondition(cond);
		}
		if (!has_one_equality) {
			return false;
		}
	}

	// we are now sure that we can remove the child DelimGet join, so we basically do the same loop as above
	// this time without checks because we already did them, and replace the expressions
	for (auto &parent_cond : parent_delim_join.conditions) {
		auto &parent_expr = parent_delim_get_side == 0 ? parent_cond.left : parent_cond.right;
		auto &parent_colref = (BoundColumnRefExpression &)*parent_expr;
		auto it = updater.reverse_proj_or_agg_map.find(parent_colref.binding);
		auto child_expr = it->second;
		for (auto &child_cond : join.conditions) {
			if (!child_cond.left->Equals(child_expr) && !child_cond.right->Equals(child_expr)) {
				continue;
			}
			parent_expr =
			    make_unique<BoundColumnRefExpression>(parent_expr->alias, parent_expr->return_type, it->first);
			parent_cond.comparison =
			    parent_delim_get_side == 0 ? child_cond.comparison : FlipComparisionExpression(child_cond.comparison);
			break;
		}
	}

	// no longer needs to be a delim join
	parent_delim_join.duplicate_eliminated_columns.clear();
	parent_delim_join.type = LogicalOperatorType::LOGICAL_COMPARISON_JOIN;

	return true;
}

} // namespace duckdb



namespace duckdb {

template <>
double EstimatedProperties::GetCardinality() const {
	return cardinality;
}

template <>
idx_t EstimatedProperties::GetCardinality() const {
	auto max_idx_t = NumericLimits<idx_t>::Maximum() - 10000;
	return MinValue<double>(cardinality, max_idx_t);
}

double EstimatedProperties::GetCost() const {
	return cost;
}

void EstimatedProperties::SetCardinality(double new_card) {
	cardinality = new_card;
}

void EstimatedProperties::SetCost(double new_cost) {
	cost = new_cost;
}

} // namespace duckdb



namespace duckdb {

unique_ptr<LogicalOperator> ExpressionHeuristics::Rewrite(unique_ptr<LogicalOperator> op) {
	VisitOperator(*op);
	return op;
}

void ExpressionHeuristics::VisitOperator(LogicalOperator &op) {
	if (op.type == LogicalOperatorType::LOGICAL_FILTER) {
		// reorder all filter expressions
		if (op.expressions.size() > 1) {
			ReorderExpressions(op.expressions);
		}
	}

	// traverse recursively through the operator tree
	VisitOperatorChildren(op);
	VisitOperatorExpressions(op);
}

unique_ptr<Expression> ExpressionHeuristics::VisitReplace(BoundConjunctionExpression &expr,
                                                          unique_ptr<Expression> *expr_ptr) {
	ReorderExpressions(expr.children);
	return nullptr;
}

void ExpressionHeuristics::ReorderExpressions(vector<unique_ptr<Expression>> &expressions) {

	struct ExpressionCosts {
		unique_ptr<Expression> expr;
		idx_t cost;

		bool operator==(const ExpressionCosts &p) const {
			return cost == p.cost;
		}
		bool operator<(const ExpressionCosts &p) const {
			return cost < p.cost;
		}
	};

	vector<ExpressionCosts> expression_costs;
	// iterate expressions, get cost for each one
	for (idx_t i = 0; i < expressions.size(); i++) {
		idx_t cost = Cost(*expressions[i]);
		expression_costs.push_back({move(expressions[i]), cost});
	}

	// sort by cost and put back in place
	sort(expression_costs.begin(), expression_costs.end());
	for (idx_t i = 0; i < expression_costs.size(); i++) {
		expressions[i] = move(expression_costs[i].expr);
	}
}

idx_t ExpressionHeuristics::ExpressionCost(BoundBetweenExpression &expr) {
	return Cost(*expr.input) + Cost(*expr.lower) + Cost(*expr.upper) + 10;
}

idx_t ExpressionHeuristics::ExpressionCost(BoundCaseExpression &expr) {
	// CASE WHEN check THEN result_if_true ELSE result_if_false END
	idx_t case_cost = 0;
	for (auto &case_check : expr.case_checks) {
		case_cost += Cost(*case_check.then_expr);
		case_cost += Cost(*case_check.when_expr);
	}
	case_cost += Cost(*expr.else_expr);
	return case_cost;
}

idx_t ExpressionHeuristics::ExpressionCost(BoundCastExpression &expr) {
	// OPERATOR_CAST
	// determine cast cost by comparing cast_expr.source_type and cast_expr_target_type
	idx_t cast_cost = 0;
	if (expr.return_type != expr.source_type()) {
		// if cast from or to varchar
		// TODO: we might want to add more cases
		if (expr.return_type.id() == LogicalTypeId::VARCHAR || expr.source_type().id() == LogicalTypeId::VARCHAR ||
		    expr.return_type.id() == LogicalTypeId::BLOB || expr.source_type().id() == LogicalTypeId::BLOB) {
			cast_cost = 200;
		} else {
			cast_cost = 5;
		}
	}
	return Cost(*expr.child) + cast_cost;
}

idx_t ExpressionHeuristics::ExpressionCost(BoundComparisonExpression &expr) {
	// COMPARE_EQUAL, COMPARE_NOTEQUAL, COMPARE_GREATERTHAN, COMPARE_GREATERTHANOREQUALTO, COMPARE_LESSTHAN,
	// COMPARE_LESSTHANOREQUALTO
	return Cost(*expr.left) + 5 + Cost(*expr.right);
}

idx_t ExpressionHeuristics::ExpressionCost(BoundConjunctionExpression &expr) {
	// CONJUNCTION_AND, CONJUNCTION_OR
	idx_t cost = 5;
	for (auto &child : expr.children) {
		cost += Cost(*child);
	}
	return cost;
}

idx_t ExpressionHeuristics::ExpressionCost(BoundFunctionExpression &expr) {
	idx_t cost_children = 0;
	for (auto &child : expr.children) {
		cost_children += Cost(*child);
	}

	auto cost_function = function_costs.find(expr.function.name);
	if (cost_function != function_costs.end()) {
		return cost_children + cost_function->second;
	} else {
		return cost_children + 1000;
	}
}

idx_t ExpressionHeuristics::ExpressionCost(BoundOperatorExpression &expr, ExpressionType &expr_type) {
	idx_t sum = 0;
	for (auto &child : expr.children) {
		sum += Cost(*child);
	}

	// OPERATOR_IS_NULL, OPERATOR_IS_NOT_NULL
	if (expr_type == ExpressionType::OPERATOR_IS_NULL || expr_type == ExpressionType::OPERATOR_IS_NOT_NULL) {
		return sum + 5;
	} else if (expr_type == ExpressionType::COMPARE_IN || expr_type == ExpressionType::COMPARE_NOT_IN) {
		// COMPARE_IN, COMPARE_NOT_IN
		return sum + (expr.children.size() - 1) * 100;
	} else if (expr_type == ExpressionType::OPERATOR_NOT) {
		// OPERATOR_NOT
		return sum + 10; // TODO: evaluate via measured runtimes
	} else {
		return sum + 1000;
	}
}

idx_t ExpressionHeuristics::ExpressionCost(PhysicalType return_type, idx_t multiplier) {
	// TODO: ajust values according to benchmark results
	switch (return_type) {
	case PhysicalType::VARCHAR:
		return 5 * multiplier;
	case PhysicalType::FLOAT:
	case PhysicalType::DOUBLE:
		return 2 * multiplier;
	default:
		return 1 * multiplier;
	}
}

idx_t ExpressionHeuristics::Cost(Expression &expr) {
	switch (expr.expression_class) {
	case ExpressionClass::BOUND_CASE: {
		auto &case_expr = (BoundCaseExpression &)expr;
		return ExpressionCost(case_expr);
	}
	case ExpressionClass::BOUND_BETWEEN: {
		auto &between_expr = (BoundBetweenExpression &)expr;
		return ExpressionCost(between_expr);
	}
	case ExpressionClass::BOUND_CAST: {
		auto &cast_expr = (BoundCastExpression &)expr;
		return ExpressionCost(cast_expr);
	}
	case ExpressionClass::BOUND_COMPARISON: {
		auto &comp_expr = (BoundComparisonExpression &)expr;
		return ExpressionCost(comp_expr);
	}
	case ExpressionClass::BOUND_CONJUNCTION: {
		auto &conj_expr = (BoundConjunctionExpression &)expr;
		return ExpressionCost(conj_expr);
	}
	case ExpressionClass::BOUND_FUNCTION: {
		auto &func_expr = (BoundFunctionExpression &)expr;
		return ExpressionCost(func_expr);
	}
	case ExpressionClass::BOUND_OPERATOR: {
		auto &op_expr = (BoundOperatorExpression &)expr;
		return ExpressionCost(op_expr, expr.type);
	}
	case ExpressionClass::BOUND_COLUMN_REF: {
		auto &col_expr = (BoundColumnRefExpression &)expr;
		return ExpressionCost(col_expr.return_type.InternalType(), 8);
	}
	case ExpressionClass::BOUND_CONSTANT: {
		auto &const_expr = (BoundConstantExpression &)expr;
		return ExpressionCost(const_expr.return_type.InternalType(), 1);
	}
	case ExpressionClass::BOUND_PARAMETER: {
		auto &const_expr = (BoundParameterExpression &)expr;
		return ExpressionCost(const_expr.return_type.InternalType(), 1);
	}
	case ExpressionClass::BOUND_REF: {
		auto &col_expr = (BoundColumnRefExpression &)expr;
		return ExpressionCost(col_expr.return_type.InternalType(), 8);
	}
	default: {
		break;
	}
	}

	// return a very high value if nothing matches
	return 1000;
}

} // namespace duckdb









namespace duckdb {

unique_ptr<Expression> ExpressionRewriter::ApplyRules(LogicalOperator &op, const vector<Rule *> &rules,
                                                      unique_ptr<Expression> expr, bool &changes_made, bool is_root) {
	for (auto &rule : rules) {
		vector<Expression *> bindings;
		if (rule->root->Match(expr.get(), bindings)) {
			// the rule matches! try to apply it
			bool rule_made_change = false;
			auto result = rule->Apply(op, bindings, rule_made_change, is_root);
			if (result) {
				changes_made = true;
				// the base node changed: the rule applied changes
				// rerun on the new node
				return ExpressionRewriter::ApplyRules(op, rules, move(result), changes_made);
			} else if (rule_made_change) {
				changes_made = true;
				// the base node didn't change, but changes were made, rerun
				return expr;
			}
			// else nothing changed, continue to the next rule
			continue;
		}
	}
	// no changes could be made to this node
	// recursively run on the children of this node
	ExpressionIterator::EnumerateChildren(*expr, [&](unique_ptr<Expression> &child) {
		child = ExpressionRewriter::ApplyRules(op, rules, move(child), changes_made);
	});
	return expr;
}

unique_ptr<Expression> ExpressionRewriter::ConstantOrNull(unique_ptr<Expression> child, Value value) {
	vector<unique_ptr<Expression>> children;
	children.push_back(make_unique<BoundConstantExpression>(value));
	children.push_back(move(child));
	return ConstantOrNull(move(children), move(value));
}

unique_ptr<Expression> ExpressionRewriter::ConstantOrNull(vector<unique_ptr<Expression>> children, Value value) {
	auto type = value.type();
	children.insert(children.begin(), make_unique<BoundConstantExpression>(value));
	return make_unique<BoundFunctionExpression>(type, ConstantOrNull::GetFunction(type), move(children),
	                                            ConstantOrNull::Bind(move(value)));
}

void ExpressionRewriter::VisitOperator(LogicalOperator &op) {
	VisitOperatorChildren(op);
	this->op = &op;

	to_apply_rules.clear();
	for (auto &rule : rules) {
		if (rule->logical_root && !rule->logical_root->Match(op.type)) {
			// this rule does not apply to this type of LogicalOperator
			continue;
		}
		to_apply_rules.push_back(rule.get());
	}
	if (to_apply_rules.empty()) {
		// no rules to apply on this node
		return;
	}

	VisitOperatorExpressions(op);

	// if it is a LogicalFilter, we split up filter conjunctions again
	if (op.type == LogicalOperatorType::LOGICAL_FILTER) {
		auto &filter = (LogicalFilter &)op;
		filter.SplitPredicates();
	}
}

void ExpressionRewriter::VisitExpression(unique_ptr<Expression> *expression) {
	bool changes_made;
	do {
		changes_made = false;
		*expression = ExpressionRewriter::ApplyRules(*op, to_apply_rules, move(*expression), changes_made, true);
	} while (changes_made);
}

ClientContext &Rule::GetContext() const {
	return rewriter.context;
}

} // namespace duckdb




















namespace duckdb {

using ExpressionValueInformation = FilterCombiner::ExpressionValueInformation;

ValueComparisonResult CompareValueInformation(ExpressionValueInformation &left, ExpressionValueInformation &right);

FilterCombiner::FilterCombiner(ClientContext &context) : context(context) {
}

FilterCombiner::FilterCombiner(Optimizer &optimizer) : FilterCombiner(optimizer.context) {
}

Expression *FilterCombiner::GetNode(Expression *expr) {
	auto entry = stored_expressions.find(expr);
	if (entry != stored_expressions.end()) {
		// expression already exists: return a reference to the stored expression
		return entry->second.get();
	}
	// expression does not exist yet: create a copy and store it
	auto copy = expr->Copy();
	auto pointer_copy = copy.get();
	D_ASSERT(stored_expressions.find(pointer_copy) == stored_expressions.end());
	stored_expressions.insert(make_pair(pointer_copy, move(copy)));
	return pointer_copy;
}

idx_t FilterCombiner::GetEquivalenceSet(Expression *expr) {
	D_ASSERT(stored_expressions.find(expr) != stored_expressions.end());
	D_ASSERT(stored_expressions.find(expr)->second.get() == expr);
	auto entry = equivalence_set_map.find(expr);
	if (entry == equivalence_set_map.end()) {
		idx_t index = set_index++;
		equivalence_set_map[expr] = index;
		equivalence_map[index].push_back(expr);
		constant_values.insert(make_pair(index, vector<ExpressionValueInformation>()));
		return index;
	} else {
		return entry->second;
	}
}

FilterResult FilterCombiner::AddConstantComparison(vector<ExpressionValueInformation> &info_list,
                                                   ExpressionValueInformation info) {
	if (info.constant.IsNull()) {
		return FilterResult::UNSATISFIABLE;
	}
	for (idx_t i = 0; i < info_list.size(); i++) {
		auto comparison = CompareValueInformation(info_list[i], info);
		switch (comparison) {
		case ValueComparisonResult::PRUNE_LEFT:
			// prune the entry from the info list
			info_list.erase(info_list.begin() + i);
			i--;
			break;
		case ValueComparisonResult::PRUNE_RIGHT:
			// prune the current info
			return FilterResult::SUCCESS;
		case ValueComparisonResult::UNSATISFIABLE_CONDITION:
			// combination of filters is unsatisfiable: prune the entire branch
			return FilterResult::UNSATISFIABLE;
		default:
			// prune nothing, move to the next condition
			break;
		}
	}
	// finally add the entry to the list
	info_list.push_back(info);
	return FilterResult::SUCCESS;
}

FilterResult FilterCombiner::AddFilter(unique_ptr<Expression> expr) {
	//	LookUpConjunctions(expr.get());
	// try to push the filter into the combiner
	auto result = AddFilter(expr.get());
	if (result == FilterResult::UNSUPPORTED) {
		// unsupported filter, push into remaining filters
		remaining_filters.push_back(move(expr));
		return FilterResult::SUCCESS;
	}
	return result;
}

void FilterCombiner::GenerateFilters(const std::function<void(unique_ptr<Expression> filter)> &callback) {
	// first loop over the remaining filters
	for (auto &filter : remaining_filters) {
		callback(move(filter));
	}
	remaining_filters.clear();
	// now loop over the equivalence sets
	for (auto &entry : equivalence_map) {
		auto equivalence_set = entry.first;
		auto &entries = entry.second;
		auto &constant_list = constant_values.find(equivalence_set)->second;
		// for each entry generate an equality expression comparing to each other
		for (idx_t i = 0; i < entries.size(); i++) {
			for (idx_t k = i + 1; k < entries.size(); k++) {
				auto comparison = make_unique<BoundComparisonExpression>(ExpressionType::COMPARE_EQUAL,
				                                                         entries[i]->Copy(), entries[k]->Copy());
				callback(move(comparison));
			}
			// for each entry also create a comparison with each constant
			int lower_index = -1, upper_index = -1;
			bool lower_inclusive, upper_inclusive;
			for (idx_t k = 0; k < constant_list.size(); k++) {
				auto &info = constant_list[k];
				if (info.comparison_type == ExpressionType::COMPARE_GREATERTHAN ||
				    info.comparison_type == ExpressionType::COMPARE_GREATERTHANOREQUALTO) {
					lower_index = k;
					lower_inclusive = info.comparison_type == ExpressionType::COMPARE_GREATERTHANOREQUALTO;
				} else if (info.comparison_type == ExpressionType::COMPARE_LESSTHAN ||
				           info.comparison_type == ExpressionType::COMPARE_LESSTHANOREQUALTO) {
					upper_index = k;
					upper_inclusive = info.comparison_type == ExpressionType::COMPARE_LESSTHANOREQUALTO;
				} else {
					auto constant = make_unique<BoundConstantExpression>(info.constant);
					auto comparison = make_unique<BoundComparisonExpression>(info.comparison_type, entries[i]->Copy(),
					                                                         move(constant));
					callback(move(comparison));
				}
			}
			if (lower_index >= 0 && upper_index >= 0) {
				// found both lower and upper index, create a BETWEEN expression
				auto lower_constant = make_unique<BoundConstantExpression>(constant_list[lower_index].constant);
				auto upper_constant = make_unique<BoundConstantExpression>(constant_list[upper_index].constant);
				auto between = make_unique<BoundBetweenExpression>(
				    entries[i]->Copy(), move(lower_constant), move(upper_constant), lower_inclusive, upper_inclusive);
				callback(move(between));
			} else if (lower_index >= 0) {
				// only lower index found, create simple comparison expression
				auto constant = make_unique<BoundConstantExpression>(constant_list[lower_index].constant);
				auto comparison = make_unique<BoundComparisonExpression>(constant_list[lower_index].comparison_type,
				                                                         entries[i]->Copy(), move(constant));
				callback(move(comparison));
			} else if (upper_index >= 0) {
				// only upper index found, create simple comparison expression
				auto constant = make_unique<BoundConstantExpression>(constant_list[upper_index].constant);
				auto comparison = make_unique<BoundComparisonExpression>(constant_list[upper_index].comparison_type,
				                                                         entries[i]->Copy(), move(constant));
				callback(move(comparison));
			}
		}
	}
	stored_expressions.clear();
	equivalence_set_map.clear();
	constant_values.clear();
	equivalence_map.clear();
}

bool FilterCombiner::HasFilters() {
	bool has_filters = false;
	GenerateFilters([&](unique_ptr<Expression> child) { has_filters = true; });
	return has_filters;
}

// unordered_map<idx_t, std::pair<Value *, Value *>> MergeAnd(unordered_map<idx_t, std::pair<Value *, Value *>> &f_1,
//                                                            unordered_map<idx_t, std::pair<Value *, Value *>> &f_2) {
// 	unordered_map<idx_t, std::pair<Value *, Value *>> result;
// 	for (auto &f : f_1) {
// 		auto it = f_2.find(f.first);
// 		if (it == f_2.end()) {
// 			result[f.first] = f.second;
// 		} else {
// 			Value *min = nullptr, *max = nullptr;
// 			if (it->second.first && f.second.first) {
// 				if (*f.second.first > *it->second.first) {
// 					min = f.second.first;
// 				} else {
// 					min = it->second.first;
// 				}

// 			} else if (it->second.first) {
// 				min = it->second.first;
// 			} else if (f.second.first) {
// 				min = f.second.first;
// 			} else {
// 				min = nullptr;
// 			}
// 			if (it->second.second && f.second.second) {
// 				if (*f.second.second < *it->second.second) {
// 					max = f.second.second;
// 				} else {
// 					max = it->second.second;
// 				}
// 			} else if (it->second.second) {
// 				max = it->second.second;
// 			} else if (f.second.second) {
// 				max = f.second.second;
// 			} else {
// 				max = nullptr;
// 			}
// 			result[f.first] = {min, max};
// 			f_2.erase(f.first);
// 		}
// 	}
// 	for (auto &f : f_2) {
// 		result[f.first] = f.second;
// 	}
// 	return result;
// }

// unordered_map<idx_t, std::pair<Value *, Value *>> MergeOr(unordered_map<idx_t, std::pair<Value *, Value *>> &f_1,
//                                                           unordered_map<idx_t, std::pair<Value *, Value *>> &f_2) {
// 	unordered_map<idx_t, std::pair<Value *, Value *>> result;
// 	for (auto &f : f_1) {
// 		auto it = f_2.find(f.first);
// 		if (it != f_2.end()) {
// 			Value *min = nullptr, *max = nullptr;
// 			if (it->second.first && f.second.first) {
// 				if (*f.second.first < *it->second.first) {
// 					min = f.second.first;
// 				} else {
// 					min = it->second.first;
// 				}
// 			}
// 			if (it->second.second && f.second.second) {
// 				if (*f.second.second > *it->second.second) {
// 					max = f.second.second;
// 				} else {
// 					max = it->second.second;
// 				}
// 			}
// 			result[f.first] = {min, max};
// 			f_2.erase(f.first);
// 		}
// 	}
// 	return result;
// }

// unordered_map<idx_t, std::pair<Value *, Value *>>
// FilterCombiner::FindZonemapChecks(vector<idx_t> &column_ids, unordered_set<idx_t> &not_constants, Expression *filter)
// { 	unordered_map<idx_t, std::pair<Value *, Value *>> checks; 	switch (filter->type) { 	case
// ExpressionType::CONJUNCTION_OR: {
// 		//! For a filter to
// 		auto &or_exp = (BoundConjunctionExpression &)*filter;
// 		checks = FindZonemapChecks(column_ids, not_constants, or_exp.children[0].get());
// 		for (size_t i = 1; i < or_exp.children.size(); ++i) {
// 			auto child_check = FindZonemapChecks(column_ids, not_constants, or_exp.children[i].get());
// 			checks = MergeOr(checks, child_check);
// 		}
// 		return checks;
// 	}
// 	case ExpressionType::CONJUNCTION_AND: {
// 		auto &and_exp = (BoundConjunctionExpression &)*filter;
// 		checks = FindZonemapChecks(column_ids, not_constants, and_exp.children[0].get());
// 		for (size_t i = 1; i < and_exp.children.size(); ++i) {
// 			auto child_check = FindZonemapChecks(column_ids, not_constants, and_exp.children[i].get());
// 			checks = MergeAnd(checks, child_check);
// 		}
// 		return checks;
// 	}
// 	case ExpressionType::COMPARE_IN: {
// 		auto &comp_in_exp = (BoundOperatorExpression &)*filter;
// 		if (comp_in_exp.children[0]->type == ExpressionType::BOUND_COLUMN_REF) {
// 			Value *min = nullptr, *max = nullptr;
// 			auto &column_ref = (BoundColumnRefExpression &)*comp_in_exp.children[0].get();
// 			for (size_t i {1}; i < comp_in_exp.children.size(); i++) {
// 				if (comp_in_exp.children[i]->type != ExpressionType::VALUE_CONSTANT) {
// 					//! This indicates the column has a comparison that is not with a constant
// 					not_constants.insert(column_ids[column_ref.binding.column_index]);
// 					break;
// 				} else {
// 					auto &const_value_expr = (BoundConstantExpression &)*comp_in_exp.children[i].get();
// 					if (const_value_expr.value.IsNull()) {
// 						return checks;
// 					}
// 					if (!min && !max) {
// 						min = &const_value_expr.value;
// 						max = min;
// 					} else {
// 						if (*min > const_value_expr.value) {
// 							min = &const_value_expr.value;
// 						}
// 						if (*max < const_value_expr.value) {
// 							max = &const_value_expr.value;
// 						}
// 					}
// 				}
// 			}
// 			checks[column_ids[column_ref.binding.column_index]] = {min, max};
// 		}
// 		return checks;
// 	}
// 	case ExpressionType::COMPARE_EQUAL: {
// 		auto &comp_exp = (BoundComparisonExpression &)*filter;
// 		if ((comp_exp.left->expression_class == ExpressionClass::BOUND_COLUMN_REF &&
// 		     comp_exp.right->expression_class == ExpressionClass::BOUND_CONSTANT)) {
// 			auto &column_ref = (BoundColumnRefExpression &)*comp_exp.left;
// 			auto &constant_value_expr = (BoundConstantExpression &)*comp_exp.right;
// 			checks[column_ids[column_ref.binding.column_index]] = {&constant_value_expr.value,
// 			                                                       &constant_value_expr.value};
// 		}
// 		if ((comp_exp.left->expression_class == ExpressionClass::BOUND_CONSTANT &&
// 		     comp_exp.right->expression_class == ExpressionClass::BOUND_COLUMN_REF)) {
// 			auto &column_ref = (BoundColumnRefExpression &)*comp_exp.right;
// 			auto &constant_value_expr = (BoundConstantExpression &)*comp_exp.left;
// 			checks[column_ids[column_ref.binding.column_index]] = {&constant_value_expr.value,
// 			                                                       &constant_value_expr.value};
// 		}
// 		return checks;
// 	}
// 	case ExpressionType::COMPARE_LESSTHAN:
// 	case ExpressionType::COMPARE_LESSTHANOREQUALTO: {
// 		auto &comp_exp = (BoundComparisonExpression &)*filter;
// 		if ((comp_exp.left->expression_class == ExpressionClass::BOUND_COLUMN_REF &&
// 		     comp_exp.right->expression_class == ExpressionClass::BOUND_CONSTANT)) {
// 			auto &column_ref = (BoundColumnRefExpression &)*comp_exp.left;
// 			auto &constant_value_expr = (BoundConstantExpression &)*comp_exp.right;
// 			checks[column_ids[column_ref.binding.column_index]] = {nullptr, &constant_value_expr.value};
// 		}
// 		if ((comp_exp.left->expression_class == ExpressionClass::BOUND_CONSTANT &&
// 		     comp_exp.right->expression_class == ExpressionClass::BOUND_COLUMN_REF)) {
// 			auto &column_ref = (BoundColumnRefExpression &)*comp_exp.right;
// 			auto &constant_value_expr = (BoundConstantExpression &)*comp_exp.left;
// 			checks[column_ids[column_ref.binding.column_index]] = {&constant_value_expr.value, nullptr};
// 		}
// 		return checks;
// 	}
// 	case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
// 	case ExpressionType::COMPARE_GREATERTHAN: {
// 		auto &comp_exp = (BoundComparisonExpression &)*filter;
// 		if ((comp_exp.left->expression_class == ExpressionClass::BOUND_COLUMN_REF &&
// 		     comp_exp.right->expression_class == ExpressionClass::BOUND_CONSTANT)) {
// 			auto &column_ref = (BoundColumnRefExpression &)*comp_exp.left;
// 			auto &constant_value_expr = (BoundConstantExpression &)*comp_exp.right;
// 			checks[column_ids[column_ref.binding.column_index]] = {&constant_value_expr.value, nullptr};
// 		}
// 		if ((comp_exp.left->expression_class == ExpressionClass::BOUND_CONSTANT &&
// 		     comp_exp.right->expression_class == ExpressionClass::BOUND_COLUMN_REF)) {
// 			auto &column_ref = (BoundColumnRefExpression &)*comp_exp.right;
// 			auto &constant_value_expr = (BoundConstantExpression &)*comp_exp.left;
// 			checks[column_ids[column_ref.binding.column_index]] = {nullptr, &constant_value_expr.value};
// 		}
// 		return checks;
// 	}
// 	default:
// 		return checks;
// 	}
// }

// vector<TableFilter> FilterCombiner::GenerateZonemapChecks(vector<idx_t> &column_ids,
//                                                           vector<TableFilter> &pushed_filters) {
// 	vector<TableFilter> zonemap_checks;
// 	unordered_set<idx_t> not_constants;
// 	//! We go through the remaining filters and capture their min max
// 	if (remaining_filters.empty()) {
// 		return zonemap_checks;
// 	}

// 	auto checks = FindZonemapChecks(column_ids, not_constants, remaining_filters[0].get());
// 	for (size_t i = 1; i < remaining_filters.size(); ++i) {
// 		auto child_check = FindZonemapChecks(column_ids, not_constants, remaining_filters[i].get());
// 		checks = MergeAnd(checks, child_check);
// 	}
// 	//! We construct the equivalent filters
// 	for (auto not_constant : not_constants) {
// 		checks.erase(not_constant);
// 	}
// 	for (const auto &pushed_filter : pushed_filters) {
// 		checks.erase(column_ids[pushed_filter.column_index]);
// 	}
// 	for (const auto &check : checks) {
// 		if (check.second.first) {
// 			zonemap_checks.emplace_back(check.second.first->Copy(), ExpressionType::COMPARE_GREATERTHANOREQUALTO,
// 			                            check.first);
// 		}
// 		if (check.second.second) {
// 			zonemap_checks.emplace_back(check.second.second->Copy(), ExpressionType::COMPARE_LESSTHANOREQUALTO,
// 			                            check.first);
// 		}
// 	}
// 	return zonemap_checks;
// }

TableFilterSet FilterCombiner::GenerateTableScanFilters(vector<idx_t> &column_ids) {
	TableFilterSet table_filters;
	//! First, we figure the filters that have constant expressions that we can push down to the table scan
	for (auto &constant_value : constant_values) {
		if (!constant_value.second.empty()) {
			auto filter_exp = equivalence_map.end();
			if ((constant_value.second[0].comparison_type == ExpressionType::COMPARE_EQUAL ||
			     constant_value.second[0].comparison_type == ExpressionType::COMPARE_GREATERTHAN ||
			     constant_value.second[0].comparison_type == ExpressionType::COMPARE_GREATERTHANOREQUALTO ||
			     constant_value.second[0].comparison_type == ExpressionType::COMPARE_LESSTHAN ||
			     constant_value.second[0].comparison_type == ExpressionType::COMPARE_LESSTHANOREQUALTO) &&
			    (TypeIsNumeric(constant_value.second[0].constant.type().InternalType()) ||
			     constant_value.second[0].constant.type().InternalType() == PhysicalType::VARCHAR ||
			     constant_value.second[0].constant.type().InternalType() == PhysicalType::BOOL)) {
				//! Here we check if these filters are column references
				filter_exp = equivalence_map.find(constant_value.first);
				if (filter_exp->second.size() == 1 && filter_exp->second[0]->type == ExpressionType::BOUND_COLUMN_REF) {
					auto filter_col_exp = static_cast<BoundColumnRefExpression *>(filter_exp->second[0]);
					auto column_index = column_ids[filter_col_exp->binding.column_index];
					if (column_index == COLUMN_IDENTIFIER_ROW_ID) {
						break;
					}
					auto equivalence_set = filter_exp->first;
					auto &entries = filter_exp->second;
					auto &constant_list = constant_values.find(equivalence_set)->second;
					// for each entry generate an equality expression comparing to each other
					for (idx_t i = 0; i < entries.size(); i++) {
						// for each entry also create a comparison with each constant
						for (idx_t k = 0; k < constant_list.size(); k++) {
							auto constant_filter = make_unique<ConstantFilter>(constant_value.second[k].comparison_type,
							                                                   constant_value.second[k].constant);
							table_filters.PushFilter(column_index, move(constant_filter));
						}
						table_filters.PushFilter(column_index, make_unique<IsNotNullFilter>());
					}
					equivalence_map.erase(filter_exp);
				}
			}
		}
	}
	//! Here we look for LIKE or IN filters
	for (idx_t rem_fil_idx = 0; rem_fil_idx < remaining_filters.size(); rem_fil_idx++) {
		auto &remaining_filter = remaining_filters[rem_fil_idx];
		if (remaining_filter->expression_class == ExpressionClass::BOUND_FUNCTION) {
			auto &func = (BoundFunctionExpression &)*remaining_filter;
			if (func.function.name == "prefix" &&
			    func.children[0]->expression_class == ExpressionClass::BOUND_COLUMN_REF &&
			    func.children[1]->type == ExpressionType::VALUE_CONSTANT) {
				//! This is a like function.
				auto &column_ref = (BoundColumnRefExpression &)*func.children[0].get();
				auto &constant_value_expr = (BoundConstantExpression &)*func.children[1].get();
				auto like_string = StringValue::Get(constant_value_expr.value);
				if (like_string.empty()) {
					continue;
				}
				auto column_index = column_ids[column_ref.binding.column_index];
				//! Here the like must be transformed to a BOUND COMPARISON geq le
				auto lower_bound =
				    make_unique<ConstantFilter>(ExpressionType::COMPARE_GREATERTHANOREQUALTO, Value(like_string));
				like_string[like_string.size() - 1]++;
				auto upper_bound = make_unique<ConstantFilter>(ExpressionType::COMPARE_LESSTHAN, Value(like_string));
				table_filters.PushFilter(column_index, move(lower_bound));
				table_filters.PushFilter(column_index, move(upper_bound));
				table_filters.PushFilter(column_index, make_unique<IsNotNullFilter>());
			}
			if (func.function.name == "~~" && func.children[0]->expression_class == ExpressionClass::BOUND_COLUMN_REF &&
			    func.children[1]->type == ExpressionType::VALUE_CONSTANT) {
				//! This is a like function.
				auto &column_ref = (BoundColumnRefExpression &)*func.children[0].get();
				auto &constant_value_expr = (BoundConstantExpression &)*func.children[1].get();
				auto &like_string = StringValue::Get(constant_value_expr.value);
				if (like_string[0] == '%' || like_string[0] == '_') {
					//! We have no prefix so nothing to pushdown
					break;
				}
				string prefix;
				bool equality = true;
				for (char const &c : like_string) {
					if (c == '%' || c == '_') {
						equality = false;
						break;
					}
					prefix += c;
				}
				auto column_index = column_ids[column_ref.binding.column_index];
				if (equality) {
					//! Here the like can be transformed to an equality query
					auto equal_filter = make_unique<ConstantFilter>(ExpressionType::COMPARE_EQUAL, Value(prefix));
					table_filters.PushFilter(column_index, move(equal_filter));
					table_filters.PushFilter(column_index, make_unique<IsNotNullFilter>());
				} else {
					//! Here the like must be transformed to a BOUND COMPARISON geq le
					auto lower_bound =
					    make_unique<ConstantFilter>(ExpressionType::COMPARE_GREATERTHANOREQUALTO, Value(prefix));
					prefix[prefix.size() - 1]++;
					auto upper_bound = make_unique<ConstantFilter>(ExpressionType::COMPARE_LESSTHAN, Value(prefix));
					table_filters.PushFilter(column_index, move(lower_bound));
					table_filters.PushFilter(column_index, move(upper_bound));
					table_filters.PushFilter(column_index, make_unique<IsNotNullFilter>());
				}
			}
		} else if (remaining_filter->type == ExpressionType::COMPARE_IN) {
			auto &func = (BoundOperatorExpression &)*remaining_filter;
			vector<hugeint_t> in_values;
			D_ASSERT(func.children.size() > 1);
			if (func.children[0]->expression_class != ExpressionClass::BOUND_COLUMN_REF) {
				continue;
			}
			auto &column_ref = (BoundColumnRefExpression &)*func.children[0].get();
			auto column_index = column_ids[column_ref.binding.column_index];
			if (column_index == COLUMN_IDENTIFIER_ROW_ID) {
				break;
			}
			//! check if all children are const expr
			bool children_constant = true;
			for (size_t i {1}; i < func.children.size(); i++) {
				if (func.children[i]->type != ExpressionType::VALUE_CONSTANT) {
					children_constant = false;
				}
			}
			if (!children_constant) {
				continue;
			}
			auto &fst_const_value_expr = (BoundConstantExpression &)*func.children[1].get();
			auto &type = fst_const_value_expr.value.type();

			//! Check if values are consecutive, if yes transform them to >= <= (only for integers)
			// e.g. if we have x IN (1, 2, 3, 4, 5) we transform this into x >= 1 AND x <= 5
			if (!type.IsIntegral()) {
				continue;
			}

			bool can_simplify_in_clause = true;
			for (idx_t i = 1; i < func.children.size(); i++) {
				auto &const_value_expr = (BoundConstantExpression &)*func.children[i].get();
				if (const_value_expr.value.IsNull()) {
					can_simplify_in_clause = false;
					break;
				}
				in_values.push_back(const_value_expr.value.GetValue<hugeint_t>());
			}
			if (!can_simplify_in_clause || in_values.empty()) {
				continue;
			}

			sort(in_values.begin(), in_values.end());

			for (idx_t in_val_idx = 1; in_val_idx < in_values.size(); in_val_idx++) {
				if (in_values[in_val_idx] - in_values[in_val_idx - 1] > 1) {
					can_simplify_in_clause = false;
					break;
				}
			}
			if (!can_simplify_in_clause) {
				continue;
			}
			auto lower_bound = make_unique<ConstantFilter>(ExpressionType::COMPARE_GREATERTHANOREQUALTO,
			                                               Value::Numeric(type, in_values.front()));
			auto upper_bound = make_unique<ConstantFilter>(ExpressionType::COMPARE_LESSTHANOREQUALTO,
			                                               Value::Numeric(type, in_values.back()));
			table_filters.PushFilter(column_index, move(lower_bound));
			table_filters.PushFilter(column_index, move(upper_bound));
			table_filters.PushFilter(column_index, make_unique<IsNotNullFilter>());

			remaining_filters.erase(remaining_filters.begin() + rem_fil_idx);
		}
	}

	//	GenerateORFilters(table_filters, column_ids);

	return table_filters;
}

static bool IsGreaterThan(ExpressionType type) {
	return type == ExpressionType::COMPARE_GREATERTHAN || type == ExpressionType::COMPARE_GREATERTHANOREQUALTO;
}

static bool IsLessThan(ExpressionType type) {
	return type == ExpressionType::COMPARE_LESSTHAN || type == ExpressionType::COMPARE_LESSTHANOREQUALTO;
}

FilterResult FilterCombiner::AddBoundComparisonFilter(Expression *expr) {
	auto &comparison = (BoundComparisonExpression &)*expr;
	if (comparison.type != ExpressionType::COMPARE_LESSTHAN &&
	    comparison.type != ExpressionType::COMPARE_LESSTHANOREQUALTO &&
	    comparison.type != ExpressionType::COMPARE_GREATERTHAN &&
	    comparison.type != ExpressionType::COMPARE_GREATERTHANOREQUALTO &&
	    comparison.type != ExpressionType::COMPARE_EQUAL && comparison.type != ExpressionType::COMPARE_NOTEQUAL) {
		// only support [>, >=, <, <=, ==] expressions
		return FilterResult::UNSUPPORTED;
	}
	// check if one of the sides is a scalar value
	bool left_is_scalar = comparison.left->IsFoldable();
	bool right_is_scalar = comparison.right->IsFoldable();
	if (left_is_scalar || right_is_scalar) {
		// comparison with scalar
		auto node = GetNode(left_is_scalar ? comparison.right.get() : comparison.left.get());
		idx_t equivalence_set = GetEquivalenceSet(node);
		auto scalar = left_is_scalar ? comparison.left.get() : comparison.right.get();
		Value constant_value;
		if (!ExpressionExecutor::TryEvaluateScalar(context, *scalar, constant_value)) {
			return FilterResult::UNSATISFIABLE;
		}
		if (constant_value.IsNull()) {
			// comparisons with null are always null (i.e. will never result in rows)
			return FilterResult::UNSATISFIABLE;
		}

		// create the ExpressionValueInformation
		ExpressionValueInformation info;
		info.comparison_type = left_is_scalar ? FlipComparisionExpression(comparison.type) : comparison.type;
		info.constant = constant_value;

		// get the current bucket of constant values
		D_ASSERT(constant_values.find(equivalence_set) != constant_values.end());
		auto &info_list = constant_values.find(equivalence_set)->second;
		// check the existing constant comparisons to see if we can do any pruning
		auto ret = AddConstantComparison(info_list, info);

		auto non_scalar = left_is_scalar ? comparison.right.get() : comparison.left.get();
		auto transitive_filter = FindTransitiveFilter(non_scalar);
		if (transitive_filter != nullptr) {
			// try to add transitive filters
			if (AddTransitiveFilters((BoundComparisonExpression &)*transitive_filter) == FilterResult::UNSUPPORTED) {
				// in case of unsuccessful re-add filter into remaining ones
				remaining_filters.push_back(move(transitive_filter));
			}
		}
		return ret;
	} else {
		// comparison between two non-scalars
		// only handle comparisons for now
		if (expr->type != ExpressionType::COMPARE_EQUAL) {
			if (IsGreaterThan(expr->type) || IsLessThan(expr->type)) {
				return AddTransitiveFilters(comparison);
			}
			return FilterResult::UNSUPPORTED;
		}
		// get the LHS and RHS nodes
		auto left_node = GetNode(comparison.left.get());
		auto right_node = GetNode(comparison.right.get());
		if (BaseExpression::Equals(left_node, right_node)) {
			return FilterResult::UNSUPPORTED;
		}
		// get the equivalence sets of the LHS and RHS
		auto left_equivalence_set = GetEquivalenceSet(left_node);
		auto right_equivalence_set = GetEquivalenceSet(right_node);
		if (left_equivalence_set == right_equivalence_set) {
			// this equality filter already exists, prune it
			return FilterResult::SUCCESS;
		}
		// add the right bucket into the left bucket
		D_ASSERT(equivalence_map.find(left_equivalence_set) != equivalence_map.end());
		D_ASSERT(equivalence_map.find(right_equivalence_set) != equivalence_map.end());

		auto &left_bucket = equivalence_map.find(left_equivalence_set)->second;
		auto &right_bucket = equivalence_map.find(right_equivalence_set)->second;
		for (auto &i : right_bucket) {
			// rewrite the equivalence set mapping for this node
			equivalence_set_map[i] = left_equivalence_set;
			// add the node to the left bucket
			left_bucket.push_back(i);
		}
		// now add all constant values from the right bucket to the left bucket
		D_ASSERT(constant_values.find(left_equivalence_set) != constant_values.end());
		D_ASSERT(constant_values.find(right_equivalence_set) != constant_values.end());
		auto &left_constant_bucket = constant_values.find(left_equivalence_set)->second;
		auto &right_constant_bucket = constant_values.find(right_equivalence_set)->second;
		for (auto &i : right_constant_bucket) {
			if (AddConstantComparison(left_constant_bucket, i) == FilterResult::UNSATISFIABLE) {
				return FilterResult::UNSATISFIABLE;
			}
		}
	}
	return FilterResult::SUCCESS;
}

FilterResult FilterCombiner::AddFilter(Expression *expr) {
	if (expr->HasParameter()) {
		return FilterResult::UNSUPPORTED;
	}
	if (expr->IsFoldable()) {
		// scalar condition, evaluate it
		Value result;
		if (!ExpressionExecutor::TryEvaluateScalar(context, *expr, result)) {
			return FilterResult::UNSUPPORTED;
		}
		result = result.DefaultCastAs(LogicalType::BOOLEAN);
		// check if the filter passes
		if (result.IsNull() || !BooleanValue::Get(result)) {
			// the filter does not pass the scalar test, create an empty result
			return FilterResult::UNSATISFIABLE;
		} else {
			// the filter passes the scalar test, just remove the condition
			return FilterResult::SUCCESS;
		}
	}
	D_ASSERT(!expr->IsFoldable());
	if (expr->GetExpressionClass() == ExpressionClass::BOUND_BETWEEN) {
		auto &comparison = (BoundBetweenExpression &)*expr;
		//! check if one of the sides is a scalar value
		bool lower_is_scalar = comparison.lower->IsFoldable();
		bool upper_is_scalar = comparison.upper->IsFoldable();
		if (lower_is_scalar || upper_is_scalar) {
			//! comparison with scalar - break apart
			auto node = GetNode(comparison.input.get());
			idx_t equivalence_set = GetEquivalenceSet(node);
			auto result = FilterResult::UNSATISFIABLE;

			if (lower_is_scalar) {
				auto scalar = comparison.lower.get();
				Value constant_value;
				if (!ExpressionExecutor::TryEvaluateScalar(context, *scalar, constant_value)) {
					return FilterResult::UNSUPPORTED;
				}

				// create the ExpressionValueInformation
				ExpressionValueInformation info;
				if (comparison.lower_inclusive) {
					info.comparison_type = ExpressionType::COMPARE_GREATERTHANOREQUALTO;
				} else {
					info.comparison_type = ExpressionType::COMPARE_GREATERTHAN;
				}
				info.constant = constant_value;

				// get the current bucket of constant values
				D_ASSERT(constant_values.find(equivalence_set) != constant_values.end());
				auto &info_list = constant_values.find(equivalence_set)->second;
				// check the existing constant comparisons to see if we can do any pruning
				result = AddConstantComparison(info_list, info);
			} else {
				D_ASSERT(upper_is_scalar);
				const auto type = comparison.upper_inclusive ? ExpressionType::COMPARE_LESSTHANOREQUALTO
				                                             : ExpressionType::COMPARE_LESSTHAN;
				auto left = comparison.lower->Copy();
				auto right = comparison.input->Copy();
				auto lower_comp = make_unique<BoundComparisonExpression>(type, move(left), move(right));
				result = AddBoundComparisonFilter(lower_comp.get());
			}

			//	 Stop if we failed
			if (result != FilterResult::SUCCESS) {
				return result;
			}

			if (upper_is_scalar) {
				auto scalar = comparison.upper.get();
				Value constant_value;
				if (!ExpressionExecutor::TryEvaluateScalar(context, *scalar, constant_value)) {
					return FilterResult::UNSUPPORTED;
				}

				// create the ExpressionValueInformation
				ExpressionValueInformation info;
				if (comparison.upper_inclusive) {
					info.comparison_type = ExpressionType::COMPARE_LESSTHANOREQUALTO;
				} else {
					info.comparison_type = ExpressionType::COMPARE_LESSTHAN;
				}
				info.constant = constant_value;

				// get the current bucket of constant values
				D_ASSERT(constant_values.find(equivalence_set) != constant_values.end());
				// check the existing constant comparisons to see if we can do any pruning
				result = AddConstantComparison(constant_values.find(equivalence_set)->second, info);
			} else {
				D_ASSERT(lower_is_scalar);
				const auto type = comparison.upper_inclusive ? ExpressionType::COMPARE_LESSTHANOREQUALTO
				                                             : ExpressionType::COMPARE_LESSTHAN;
				auto left = comparison.input->Copy();
				auto right = comparison.upper->Copy();
				auto upper_comp = make_unique<BoundComparisonExpression>(type, move(left), move(right));
				result = AddBoundComparisonFilter(upper_comp.get());
			}

			return result;
		}
	} else if (expr->GetExpressionClass() == ExpressionClass::BOUND_COMPARISON) {
		return AddBoundComparisonFilter(expr);
	}
	// only comparisons supported for now
	return FilterResult::UNSUPPORTED;
}

/*
 * Create and add new transitive filters from a two non-scalar filter such as j > i, j >= i, j < i, and j <= i
 * It's missing to create another method to add transitive filters from scalar filters, e.g, i > 10
 */
FilterResult FilterCombiner::AddTransitiveFilters(BoundComparisonExpression &comparison) {
	D_ASSERT(IsGreaterThan(comparison.type) || IsLessThan(comparison.type));
	// get the LHS and RHS nodes
	Expression *left_node = GetNode(comparison.left.get());
	Expression *right_node = GetNode(comparison.right.get());
	// In case with filters like CAST(i) = j and i = 5 we replace the COLUMN_REF i with the constant 5
	if (right_node->type == ExpressionType::OPERATOR_CAST) {
		auto &bound_cast_expr = (BoundCastExpression &)*right_node;
		if (bound_cast_expr.child->type == ExpressionType::BOUND_COLUMN_REF) {
			auto &col_ref = (BoundColumnRefExpression &)*bound_cast_expr.child;
			for (auto &stored_exp : stored_expressions) {
				if (stored_exp.first->type == ExpressionType::BOUND_COLUMN_REF) {
					auto &st_col_ref = (BoundColumnRefExpression &)*stored_exp.second;
					if (st_col_ref.binding == col_ref.binding) {
						bound_cast_expr.child = stored_exp.second->Copy();
						right_node = GetNode(bound_cast_expr.child.get());
						break;
					}
				}
			}
		}
	}

	if (BaseExpression::Equals(left_node, right_node)) {
		return FilterResult::UNSUPPORTED;
	}
	// get the equivalence sets of the LHS and RHS
	idx_t left_equivalence_set = GetEquivalenceSet(left_node);
	idx_t right_equivalence_set = GetEquivalenceSet(right_node);
	if (left_equivalence_set == right_equivalence_set) {
		// this equality filter already exists, prune it
		return FilterResult::SUCCESS;
	}

	vector<ExpressionValueInformation> &left_constants = constant_values.find(left_equivalence_set)->second;
	vector<ExpressionValueInformation> &right_constants = constant_values.find(right_equivalence_set)->second;
	bool is_successful = false;
	bool is_inserted = false;
	// read every constant filters already inserted for the right scalar variable
	// and see if we can create new transitive filters, e.g., there is already a filter i > 10,
	// suppose that we have now the j >= i, then we can infer a new filter j > 10
	for (const auto &right_constant : right_constants) {
		ExpressionValueInformation info;
		info.constant = right_constant.constant;
		// there is already an equality filter, e.g., i = 10
		if (right_constant.comparison_type == ExpressionType::COMPARE_EQUAL) {
			// create filter j [>, >=, <, <=] 10
			// suppose the new comparison is j >= i and we have already a filter i = 10,
			// then we create a new filter j >= 10
			// and the filter j >= i can be pruned by not adding it into the remaining filters
			info.comparison_type = comparison.type;
		} else if ((comparison.type == ExpressionType::COMPARE_GREATERTHANOREQUALTO &&
		            IsGreaterThan(right_constant.comparison_type)) ||
		           (comparison.type == ExpressionType::COMPARE_LESSTHANOREQUALTO &&
		            IsLessThan(right_constant.comparison_type))) {
			// filters (j >= i AND i [>, >=] 10) OR (j <= i AND i [<, <=] 10)
			// create filter j [>, >=] 10 and add the filter j [>=, <=] i into the remaining filters
			info.comparison_type = right_constant.comparison_type; // create filter j [>, >=, <, <=] 10
			if (!is_inserted) {
				// Add the filter j >= i in the remaing filters
				auto filter = make_unique<BoundComparisonExpression>(comparison.type, comparison.left->Copy(),
				                                                     comparison.right->Copy());
				remaining_filters.push_back(move(filter));
				is_inserted = true;
			}
		} else if ((comparison.type == ExpressionType::COMPARE_GREATERTHAN &&
		            IsGreaterThan(right_constant.comparison_type)) ||
		           (comparison.type == ExpressionType::COMPARE_LESSTHAN &&
		            IsLessThan(right_constant.comparison_type))) {
			// filters (j > i AND i [>, >=] 10) OR j < i AND i [<, <=] 10
			// create filter j [>, <] 10 and add the filter j [>, <] i into the remaining filters
			// the comparisons j > i and j < i are more restrictive
			info.comparison_type = comparison.type;
			if (!is_inserted) {
				// Add the filter j [>, <] i
				auto filter = make_unique<BoundComparisonExpression>(comparison.type, comparison.left->Copy(),
				                                                     comparison.right->Copy());
				remaining_filters.push_back(move(filter));
				is_inserted = true;
			}
		} else {
			// we cannot add a new filter
			continue;
		}
		// Add the new filer into the left set
		if (AddConstantComparison(left_constants, info) == FilterResult::UNSATISFIABLE) {
			return FilterResult::UNSATISFIABLE;
		}
		is_successful = true;
	}
	if (is_successful) {
		// now check for remaining trasitive filters from the left column
		auto transitive_filter = FindTransitiveFilter(comparison.left.get());
		if (transitive_filter != nullptr) {
			// try to add transitive filters
			if (AddTransitiveFilters((BoundComparisonExpression &)*transitive_filter) == FilterResult::UNSUPPORTED) {
				// in case of unsuccessful re-add filter into remaining ones
				remaining_filters.push_back(move(transitive_filter));
			}
		}
		return FilterResult::SUCCESS;
	}

	return FilterResult::UNSUPPORTED;
}

/*
 * Find a transitive filter already inserted into the remaining filters
 * Check for a match between the right column of bound comparisons and the expression,
 * then removes the bound comparison from the remaining filters and returns it
 */
unique_ptr<Expression> FilterCombiner::FindTransitiveFilter(Expression *expr) {
	// We only check for bound column ref
	if (expr->type == ExpressionType::BOUND_COLUMN_REF) {
		for (idx_t i = 0; i < remaining_filters.size(); i++) {
			if (remaining_filters[i]->GetExpressionClass() == ExpressionClass::BOUND_COMPARISON) {
				auto comparison = (BoundComparisonExpression *)remaining_filters[i].get();
				if (expr->Equals(comparison->right.get()) && comparison->type != ExpressionType::COMPARE_NOTEQUAL) {
					auto filter = move(remaining_filters[i]);
					remaining_filters.erase(remaining_filters.begin() + i);
					return filter;
				}
			}
		}
	}
	return nullptr;
}

ValueComparisonResult InvertValueComparisonResult(ValueComparisonResult result) {
	if (result == ValueComparisonResult::PRUNE_RIGHT) {
		return ValueComparisonResult::PRUNE_LEFT;
	}
	if (result == ValueComparisonResult::PRUNE_LEFT) {
		return ValueComparisonResult::PRUNE_RIGHT;
	}
	return result;
}

ValueComparisonResult CompareValueInformation(ExpressionValueInformation &left, ExpressionValueInformation &right) {
	if (left.comparison_type == ExpressionType::COMPARE_EQUAL) {
		// left is COMPARE_EQUAL, we can either
		// (1) prune the right side or
		// (2) return UNSATISFIABLE
		bool prune_right_side = false;
		switch (right.comparison_type) {
		case ExpressionType::COMPARE_LESSTHAN:
			prune_right_side = left.constant < right.constant;
			break;
		case ExpressionType::COMPARE_LESSTHANOREQUALTO:
			prune_right_side = left.constant <= right.constant;
			break;
		case ExpressionType::COMPARE_GREATERTHAN:
			prune_right_side = left.constant > right.constant;
			break;
		case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
			prune_right_side = left.constant >= right.constant;
			break;
		case ExpressionType::COMPARE_NOTEQUAL:
			prune_right_side = left.constant != right.constant;
			break;
		default:
			D_ASSERT(right.comparison_type == ExpressionType::COMPARE_EQUAL);
			prune_right_side = left.constant == right.constant;
			break;
		}
		if (prune_right_side) {
			return ValueComparisonResult::PRUNE_RIGHT;
		} else {
			return ValueComparisonResult::UNSATISFIABLE_CONDITION;
		}
	} else if (right.comparison_type == ExpressionType::COMPARE_EQUAL) {
		// right is COMPARE_EQUAL
		return InvertValueComparisonResult(CompareValueInformation(right, left));
	} else if (left.comparison_type == ExpressionType::COMPARE_NOTEQUAL) {
		// left is COMPARE_NOTEQUAL, we can either
		// (1) prune the left side or
		// (2) not prune anything
		bool prune_left_side = false;
		switch (right.comparison_type) {
		case ExpressionType::COMPARE_LESSTHAN:
			prune_left_side = left.constant >= right.constant;
			break;
		case ExpressionType::COMPARE_LESSTHANOREQUALTO:
			prune_left_side = left.constant > right.constant;
			break;
		case ExpressionType::COMPARE_GREATERTHAN:
			prune_left_side = left.constant <= right.constant;
			break;
		case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
			prune_left_side = left.constant < right.constant;
			break;
		default:
			D_ASSERT(right.comparison_type == ExpressionType::COMPARE_NOTEQUAL);
			prune_left_side = left.constant == right.constant;
			break;
		}
		if (prune_left_side) {
			return ValueComparisonResult::PRUNE_LEFT;
		} else {
			return ValueComparisonResult::PRUNE_NOTHING;
		}
	} else if (right.comparison_type == ExpressionType::COMPARE_NOTEQUAL) {
		return InvertValueComparisonResult(CompareValueInformation(right, left));
	} else if (IsGreaterThan(left.comparison_type) && IsGreaterThan(right.comparison_type)) {
		// both comparisons are [>], we can either
		// (1) prune the left side or
		// (2) prune the right side
		if (left.constant > right.constant) {
			// left constant is more selective, prune right
			return ValueComparisonResult::PRUNE_RIGHT;
		} else if (left.constant < right.constant) {
			// right constant is more selective, prune left
			return ValueComparisonResult::PRUNE_LEFT;
		} else {
			// constants are equivalent
			// however we can still have the scenario where one is [>=] and the other is [>]
			// we want to prune the [>=] because [>] is more selective
			// if left is [>=] we prune the left, else we prune the right
			if (left.comparison_type == ExpressionType::COMPARE_GREATERTHANOREQUALTO) {
				return ValueComparisonResult::PRUNE_LEFT;
			} else {
				return ValueComparisonResult::PRUNE_RIGHT;
			}
		}
	} else if (IsLessThan(left.comparison_type) && IsLessThan(right.comparison_type)) {
		// both comparisons are [<], we can either
		// (1) prune the left side or
		// (2) prune the right side
		if (left.constant < right.constant) {
			// left constant is more selective, prune right
			return ValueComparisonResult::PRUNE_RIGHT;
		} else if (left.constant > right.constant) {
			// right constant is more selective, prune left
			return ValueComparisonResult::PRUNE_LEFT;
		} else {
			// constants are equivalent
			// however we can still have the scenario where one is [<=] and the other is [<]
			// we want to prune the [<=] because [<] is more selective
			// if left is [<=] we prune the left, else we prune the right
			if (left.comparison_type == ExpressionType::COMPARE_LESSTHANOREQUALTO) {
				return ValueComparisonResult::PRUNE_LEFT;
			} else {
				return ValueComparisonResult::PRUNE_RIGHT;
			}
		}
	} else if (IsLessThan(left.comparison_type)) {
		D_ASSERT(IsGreaterThan(right.comparison_type));
		// left is [<] and right is [>], in this case we can either
		// (1) prune nothing or
		// (2) return UNSATISFIABLE
		// the SMALLER THAN constant has to be greater than the BIGGER THAN constant
		if (left.constant >= right.constant) {
			return ValueComparisonResult::PRUNE_NOTHING;
		} else {
			return ValueComparisonResult::UNSATISFIABLE_CONDITION;
		}
	} else {
		// left is [>] and right is [<] or [!=]
		D_ASSERT(IsLessThan(right.comparison_type) && IsGreaterThan(left.comparison_type));
		return InvertValueComparisonResult(CompareValueInformation(right, left));
	}
}
//
// void FilterCombiner::LookUpConjunctions(Expression *expr) {
//	if (expr->GetExpressionType() == ExpressionType::CONJUNCTION_OR) {
//		auto root_or_expr = (BoundConjunctionExpression *)expr;
//		for (const auto &entry : map_col_conjunctions) {
//			for (const auto &conjs_to_push : entry.second) {
//				if (conjs_to_push->root_or->Equals(root_or_expr)) {
//					return;
//				}
//			}
//		}
//
//		cur_root_or = root_or_expr;
//		cur_conjunction = root_or_expr;
//		cur_colref_to_push = nullptr;
//		if (!BFSLookUpConjunctions(cur_root_or)) {
//			if (cur_colref_to_push) {
//				auto entry = map_col_conjunctions.find(cur_colref_to_push);
//				auto &vec_conjs_to_push = entry->second;
//				if (vec_conjs_to_push.size() == 1) {
//					map_col_conjunctions.erase(entry);
//					return;
//				}
//				vec_conjs_to_push.pop_back();
//			}
//		}
//		return;
//	}
//
//	// Verify if the expression has a column already pushed down by other OR expression
//	VerifyOrsToPush(*expr);
//}
//
// bool FilterCombiner::BFSLookUpConjunctions(BoundConjunctionExpression *conjunction) {
//	vector<BoundConjunctionExpression *> conjunctions_to_visit;
//
//	for (auto &child : conjunction->children) {
//		switch (child->GetExpressionClass()) {
//		case ExpressionClass::BOUND_CONJUNCTION: {
//			auto child_conjunction = (BoundConjunctionExpression *)child.get();
//			conjunctions_to_visit.emplace_back(child_conjunction);
//			break;
//		}
//		case ExpressionClass::BOUND_COMPARISON: {
//			if (!UpdateConjunctionFilter((BoundComparisonExpression *)child.get())) {
//				return false;
//			}
//			break;
//		}
//		default: {
//			return false;
//		}
//		}
//	}
//
//	for (auto child_conjunction : conjunctions_to_visit) {
//		cur_conjunction = child_conjunction;
//		// traverse child conjuction
//		if (!BFSLookUpConjunctions(child_conjunction)) {
//			return false;
//		}
//	}
//	return true;
//}
//
// void FilterCombiner::VerifyOrsToPush(Expression &expr) {
//	if (expr.type == ExpressionType::BOUND_COLUMN_REF) {
//		auto colref = (BoundColumnRefExpression *)&expr;
//		auto entry = map_col_conjunctions.find(colref);
//		if (entry == map_col_conjunctions.end()) {
//			return;
//		}
//		map_col_conjunctions.erase(entry);
//	}
//	ExpressionIterator::EnumerateChildren(expr, [&](Expression &child) { VerifyOrsToPush(child); });
//}
//
// bool FilterCombiner::UpdateConjunctionFilter(BoundComparisonExpression *comparison_expr) {
//	bool left_is_scalar = comparison_expr->left->IsFoldable();
//	bool right_is_scalar = comparison_expr->right->IsFoldable();
//
//	Expression *non_scalar_expr;
//	if (left_is_scalar || right_is_scalar) {
//		// only support comparison with scalar
//		non_scalar_expr = left_is_scalar ? comparison_expr->right.get() : comparison_expr->left.get();
//
//		if (non_scalar_expr->GetExpressionType() == ExpressionType::BOUND_COLUMN_REF) {
//			return UpdateFilterByColumn((BoundColumnRefExpression *)non_scalar_expr, comparison_expr);
//		}
//	}
//
//	return false;
//}
//
// bool FilterCombiner::UpdateFilterByColumn(BoundColumnRefExpression *column_ref,
//                                          BoundComparisonExpression *comparison_expr) {
//	if (cur_colref_to_push == nullptr) {
//		cur_colref_to_push = column_ref;
//
//		auto or_conjunction = make_unique<BoundConjunctionExpression>(ExpressionType::CONJUNCTION_OR);
//		or_conjunction->children.emplace_back(comparison_expr->Copy());
//
//		unique_ptr<ConjunctionsToPush> conjs_to_push = make_unique<ConjunctionsToPush>();
//		conjs_to_push->conjunctions.emplace_back(move(or_conjunction));
//		conjs_to_push->root_or = cur_root_or;
//
//		auto &&vec_col_conjs = map_col_conjunctions[column_ref];
//		vec_col_conjs.emplace_back(move(conjs_to_push));
//		vec_colref_insertion_order.emplace_back(column_ref);
//		return true;
//	}
//
//	auto entry = map_col_conjunctions.find(cur_colref_to_push);
//	D_ASSERT(entry != map_col_conjunctions.end());
//	auto &conjunctions_to_push = entry->second.back();
//
//	if (!cur_colref_to_push->Equals(column_ref)) {
//		// check for multiple colunms in the same root OR node
//		if (cur_root_or == cur_conjunction) {
//			return false;
//		}
//		// found an AND using a different column, we should stop the look up
//		if (cur_conjunction->GetExpressionType() == ExpressionType::CONJUNCTION_AND) {
//			return false;
//		}
//
//		// found a different column, AND conditions cannot be preserved anymore
//		conjunctions_to_push->preserve_and = false;
//		return true;
//	}
//
//	auto &last_conjunction = conjunctions_to_push->conjunctions.back();
//	if (cur_conjunction->GetExpressionType() == last_conjunction->GetExpressionType()) {
//		last_conjunction->children.emplace_back(comparison_expr->Copy());
//	} else {
//		auto new_conjunction = make_unique<BoundConjunctionExpression>(cur_conjunction->GetExpressionType());
//		new_conjunction->children.emplace_back(comparison_expr->Copy());
//		conjunctions_to_push->conjunctions.emplace_back(move(new_conjunction));
//	}
//	return true;
//}
//
// void FilterCombiner::GenerateORFilters(TableFilterSet &table_filter, vector<idx_t> &column_ids) {
//	for (const auto colref : vec_colref_insertion_order) {
//		auto column_index = column_ids[colref->binding.column_index];
//		if (column_index == COLUMN_IDENTIFIER_ROW_ID) {
//			break;
//		}
//
//		for (const auto &conjunctions_to_push : map_col_conjunctions[colref]) {
//			// root OR filter to push into the TableFilter
//			auto root_or_filter = make_unique<ConjunctionOrFilter>();
//			// variable to hold the last conjuntion filter pointer
//			// the next filter will be added into it, i.e., we create a chain of conjunction filters
//			ConjunctionFilter *last_conj_filter = root_or_filter.get();
//
//			for (auto &conjunction : conjunctions_to_push->conjunctions) {
//				if (conjunction->GetExpressionType() == ExpressionType::CONJUNCTION_AND &&
//				    conjunctions_to_push->preserve_and) {
//					GenerateConjunctionFilter<ConjunctionAndFilter>(conjunction.get(), last_conj_filter);
//				} else {
//					GenerateConjunctionFilter<ConjunctionOrFilter>(conjunction.get(), last_conj_filter);
//				}
//			}
//			table_filter.PushFilter(column_index, move(root_or_filter));
//		}
//	}
//	map_col_conjunctions.clear();
//	vec_colref_insertion_order.clear();
//}

} // namespace duckdb



namespace duckdb {

unique_ptr<LogicalOperator> FilterPullup::Rewrite(unique_ptr<LogicalOperator> op) {
	switch (op->type) {
	case LogicalOperatorType::LOGICAL_FILTER:
		return PullupFilter(move(op));
	case LogicalOperatorType::LOGICAL_PROJECTION:
		return PullupProjection(move(op));
	case LogicalOperatorType::LOGICAL_CROSS_PRODUCT:
		return PullupCrossProduct(move(op));
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
	case LogicalOperatorType::LOGICAL_ANY_JOIN:
	case LogicalOperatorType::LOGICAL_DELIM_JOIN:
		return PullupJoin(move(op));
	case LogicalOperatorType::LOGICAL_INTERSECT:
	case LogicalOperatorType::LOGICAL_EXCEPT:
		return PullupSetOperation(move(op));
	case LogicalOperatorType::LOGICAL_DISTINCT:
	case LogicalOperatorType::LOGICAL_ORDER_BY: {
		// we can just pull directly through these operations without any rewriting
		op->children[0] = Rewrite(move(op->children[0]));
		return op;
	}
	default:
		return FinishPullup(move(op));
	}
}

unique_ptr<LogicalOperator> FilterPullup::PullupJoin(unique_ptr<LogicalOperator> op) {
	D_ASSERT(op->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN ||
	         op->type == LogicalOperatorType::LOGICAL_ANY_JOIN || op->type == LogicalOperatorType::LOGICAL_DELIM_JOIN);
	auto &join = (LogicalJoin &)*op;

	switch (join.join_type) {
	case JoinType::INNER:
		return PullupInnerJoin(move(op));
	case JoinType::LEFT:
	case JoinType::ANTI:
	case JoinType::SEMI: {
		return PullupFromLeft(move(op));
	}
	default:
		// unsupported join type: call children pull up
		return FinishPullup(move(op));
	}
}

unique_ptr<LogicalOperator> FilterPullup::PullupInnerJoin(unique_ptr<LogicalOperator> op) {
	D_ASSERT(((LogicalJoin &)*op).join_type == JoinType::INNER);
	D_ASSERT(op->type != LogicalOperatorType::LOGICAL_DELIM_JOIN);
	return PullupBothSide(move(op));
}

unique_ptr<LogicalOperator> FilterPullup::PullupCrossProduct(unique_ptr<LogicalOperator> op) {
	D_ASSERT(op->type == LogicalOperatorType::LOGICAL_CROSS_PRODUCT);
	return PullupBothSide(move(op));
}

unique_ptr<LogicalOperator> FilterPullup::GeneratePullupFilter(unique_ptr<LogicalOperator> child,
                                                               vector<unique_ptr<Expression>> &expressions) {
	unique_ptr<LogicalFilter> filter = make_unique<LogicalFilter>();
	for (idx_t i = 0; i < expressions.size(); ++i) {
		filter->expressions.push_back(move(expressions[i]));
	}
	expressions.clear();
	filter->children.push_back(move(child));
	return move(filter);
}

unique_ptr<LogicalOperator> FilterPullup::FinishPullup(unique_ptr<LogicalOperator> op) {
	// unhandled type, first perform filter pushdown in its children
	for (idx_t i = 0; i < op->children.size(); i++) {
		FilterPullup pullup;
		op->children[i] = pullup.Rewrite(move(op->children[i]));
	}
	// now pull up any existing filters
	if (filters_expr_pullup.empty()) {
		// no filters to pull up
		return op;
	}
	return GeneratePullupFilter(move(op), filters_expr_pullup);
}

} // namespace duckdb







namespace duckdb {

using Filter = FilterPushdown::Filter;

FilterPushdown::FilterPushdown(Optimizer &optimizer) : optimizer(optimizer), combiner(optimizer.context) {
}

unique_ptr<LogicalOperator> FilterPushdown::Rewrite(unique_ptr<LogicalOperator> op) {
	D_ASSERT(!combiner.HasFilters());
	switch (op->type) {
	case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
		return PushdownAggregate(move(op));
	case LogicalOperatorType::LOGICAL_FILTER:
		return PushdownFilter(move(op));
	case LogicalOperatorType::LOGICAL_CROSS_PRODUCT:
		return PushdownCrossProduct(move(op));
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
	case LogicalOperatorType::LOGICAL_ANY_JOIN:
	case LogicalOperatorType::LOGICAL_DELIM_JOIN:
		return PushdownJoin(move(op));
	case LogicalOperatorType::LOGICAL_PROJECTION:
		return PushdownProjection(move(op));
	case LogicalOperatorType::LOGICAL_INTERSECT:
	case LogicalOperatorType::LOGICAL_EXCEPT:
	case LogicalOperatorType::LOGICAL_UNION:
		return PushdownSetOperation(move(op));
	case LogicalOperatorType::LOGICAL_DISTINCT:
	case LogicalOperatorType::LOGICAL_ORDER_BY: {
		// we can just push directly through these operations without any rewriting
		op->children[0] = Rewrite(move(op->children[0]));
		return op;
	}
	case LogicalOperatorType::LOGICAL_GET:
		return PushdownGet(move(op));
	case LogicalOperatorType::LOGICAL_LIMIT:
		return PushdownLimit(move(op));
	default:
		return FinishPushdown(move(op));
	}
}

unique_ptr<LogicalOperator> FilterPushdown::PushdownJoin(unique_ptr<LogicalOperator> op) {
	D_ASSERT(op->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN ||
	         op->type == LogicalOperatorType::LOGICAL_ANY_JOIN || op->type == LogicalOperatorType::LOGICAL_DELIM_JOIN);
	auto &join = (LogicalJoin &)*op;
	unordered_set<idx_t> left_bindings, right_bindings;
	LogicalJoin::GetTableReferences(*op->children[0], left_bindings);
	LogicalJoin::GetTableReferences(*op->children[1], right_bindings);

	switch (join.join_type) {
	case JoinType::INNER:
		return PushdownInnerJoin(move(op), left_bindings, right_bindings);
	case JoinType::LEFT:
		return PushdownLeftJoin(move(op), left_bindings, right_bindings);
	case JoinType::MARK:
		return PushdownMarkJoin(move(op), left_bindings, right_bindings);
	case JoinType::SINGLE:
		return PushdownSingleJoin(move(op), left_bindings, right_bindings);
	default:
		// unsupported join type: stop pushing down
		return FinishPushdown(move(op));
	}
}
void FilterPushdown::PushFilters() {
	for (auto &f : filters) {
		auto result = combiner.AddFilter(move(f->filter));
		D_ASSERT(result != FilterResult::UNSUPPORTED);
		(void)result;
	}
	filters.clear();
}

FilterResult FilterPushdown::AddFilter(unique_ptr<Expression> expr) {
	PushFilters();
	// split up the filters by AND predicate
	vector<unique_ptr<Expression>> expressions;
	expressions.push_back(move(expr));
	LogicalFilter::SplitPredicates(expressions);
	// push the filters into the combiner
	for (auto &child_expr : expressions) {
		if (combiner.AddFilter(move(child_expr)) == FilterResult::UNSATISFIABLE) {
			return FilterResult::UNSATISFIABLE;
		}
	}
	return FilterResult::SUCCESS;
}

void FilterPushdown::GenerateFilters() {
	if (!filters.empty()) {
		D_ASSERT(!combiner.HasFilters());
		return;
	}
	combiner.GenerateFilters([&](unique_ptr<Expression> filter) {
		auto f = make_unique<Filter>();
		f->filter = move(filter);
		f->ExtractBindings();
		filters.push_back(move(f));
	});
}

unique_ptr<LogicalOperator> FilterPushdown::FinishPushdown(unique_ptr<LogicalOperator> op) {
	// unhandled type, first perform filter pushdown in its children
	for (auto &child : op->children) {
		FilterPushdown pushdown(optimizer);
		child = pushdown.Rewrite(move(child));
	}
	// now push any existing filters
	if (filters.empty()) {
		// no filters to push
		return op;
	}
	auto filter = make_unique<LogicalFilter>();
	for (auto &f : filters) {
		filter->expressions.push_back(move(f->filter));
	}
	filter->children.push_back(move(op));
	return move(filter);
}

void FilterPushdown::Filter::ExtractBindings() {
	bindings.clear();
	LogicalJoin::GetExpressionBindings(*filter, bindings);
}

} // namespace duckdb










namespace duckdb {

unique_ptr<LogicalOperator> InClauseRewriter::Rewrite(unique_ptr<LogicalOperator> op) {
	if (op->children.size() == 1) {
		root = move(op->children[0]);
		VisitOperatorExpressions(*op);
		op->children[0] = move(root);
	}

	for (auto &child : op->children) {
		child = Rewrite(move(child));
	}
	return op;
}

unique_ptr<Expression> InClauseRewriter::VisitReplace(BoundOperatorExpression &expr, unique_ptr<Expression> *expr_ptr) {
	if (expr.type != ExpressionType::COMPARE_IN && expr.type != ExpressionType::COMPARE_NOT_IN) {
		return nullptr;
	}
	D_ASSERT(root);
	auto in_type = expr.children[0]->return_type;
	bool is_regular_in = expr.type == ExpressionType::COMPARE_IN;
	bool all_scalar = true;
	// IN clause with many children: try to generate a mark join that replaces this IN expression
	// we can only do this if the expressions in the expression list are scalar
	for (idx_t i = 1; i < expr.children.size(); i++) {
		if (!expr.children[i]->IsFoldable()) {
			// non-scalar expression
			all_scalar = false;
		}
	}
	if (expr.children.size() == 2) {
		// only one child
		// IN: turn into X = 1
		// NOT IN: turn into X <> 1
		return make_unique<BoundComparisonExpression>(is_regular_in ? ExpressionType::COMPARE_EQUAL
		                                                            : ExpressionType::COMPARE_NOTEQUAL,
		                                              move(expr.children[0]), move(expr.children[1]));
	}
	if (expr.children.size() < 6 || !all_scalar) {
		// low amount of children or not all scalar
		// IN: turn into (X = 1 OR X = 2 OR X = 3...)
		// NOT IN: turn into (X <> 1 AND X <> 2 AND X <> 3 ...)
		auto conjunction = make_unique<BoundConjunctionExpression>(is_regular_in ? ExpressionType::CONJUNCTION_OR
		                                                                         : ExpressionType::CONJUNCTION_AND);
		for (idx_t i = 1; i < expr.children.size(); i++) {
			conjunction->children.push_back(make_unique<BoundComparisonExpression>(
			    is_regular_in ? ExpressionType::COMPARE_EQUAL : ExpressionType::COMPARE_NOTEQUAL,
			    expr.children[0]->Copy(), move(expr.children[i])));
		}
		return move(conjunction);
	}
	// IN clause with many constant children
	// generate a mark join that replaces this IN expression
	// first generate a ColumnDataCollection from the set of expressions
	vector<LogicalType> types = {in_type};
	auto collection = make_unique<ColumnDataCollection>(context, types);
	ColumnDataAppendState append_state;
	collection->InitializeAppend(append_state);

	DataChunk chunk;
	chunk.Initialize(context, types);
	for (idx_t i = 1; i < expr.children.size(); i++) {
		// resolve this expression to a constant
		auto value = ExpressionExecutor::EvaluateScalar(context, *expr.children[i]);
		idx_t index = chunk.size();
		chunk.SetCardinality(chunk.size() + 1);
		chunk.SetValue(0, index, value);
		if (chunk.size() == STANDARD_VECTOR_SIZE || i + 1 == expr.children.size()) {
			// chunk full: append to chunk collection
			collection->Append(append_state, chunk);
			chunk.Reset();
		}
	}
	// now generate a ChunkGet that scans this collection
	auto chunk_index = optimizer.binder.GenerateTableIndex();
	auto chunk_scan = make_unique<LogicalColumnDataGet>(chunk_index, types, move(collection));

	// then we generate the MARK join with the chunk scan on the RHS
	auto join = make_unique<LogicalComparisonJoin>(JoinType::MARK);
	join->mark_index = chunk_index;
	join->AddChild(move(root));
	join->AddChild(move(chunk_scan));
	// create the JOIN condition
	JoinCondition cond;
	cond.left = move(expr.children[0]);

	cond.right = make_unique<BoundColumnRefExpression>(in_type, ColumnBinding(chunk_index, 0));
	cond.comparison = ExpressionType::COMPARE_EQUAL;
	join->conditions.push_back(move(cond));
	root = move(join);

	// we replace the original subquery with a BoundColumnRefExpression referring to the mark column
	unique_ptr<Expression> result =
	    make_unique<BoundColumnRefExpression>("IN (...)", LogicalType::BOOLEAN, ColumnBinding(chunk_index, 0));
	if (!is_regular_in) {
		// NOT IN: invert
		auto invert = make_unique<BoundOperatorExpression>(ExpressionType::OPERATOR_NOT, LogicalType::BOOLEAN);
		invert->children.push_back(move(result));
		result = move(invert);
	}
	return result;
}

} // namespace duckdb





namespace duckdb {

JoinNode::JoinNode(JoinRelationSet *set, const double base_cardinality)
    : set(set), info(nullptr), has_filter(false), left(nullptr), right(nullptr), base_cardinality(base_cardinality) {
	estimated_props = make_unique<EstimatedProperties>(base_cardinality, 0);
}

JoinNode::JoinNode(JoinRelationSet *set, NeighborInfo *info, JoinNode *left, JoinNode *right,
                   const double base_cardinality, double cost)
    : set(set), info(info), has_filter(false), left(left), right(right), base_cardinality(base_cardinality) {
	estimated_props = make_unique<EstimatedProperties>(base_cardinality, cost);
}

unique_ptr<EstimatedProperties> EstimatedProperties::Copy() {
	auto result = make_unique<EstimatedProperties>(cardinality, cost);
	return result;
}

double JoinNode::GetCost() {
	return estimated_props->GetCost();
}

void JoinNode::SetCost(double cost) {
	estimated_props->SetCost(cost);
}

double JoinNode::GetBaseTableCardinality() {
	if (set->count > 1) {
		throw InvalidInputException("Cannot call get base table cardinality on intermediate join node");
	}
	return base_cardinality;
}

void JoinNode::SetBaseTableCardinality(double base_card) {
	base_cardinality = base_card;
}

void JoinNode::SetEstimatedCardinality(double estimated_card) {
	estimated_props->SetCardinality(estimated_card);
}

string JoinNode::ToString() {
	if (!set) {
		return "";
	}
	string result = "-------------------------------\n";
	result += set->ToString() + "\n";
	result += "card = " + to_string(GetCardinality<double>()) + "\n";
	bool is_cartesian = false;
	if (left && right) {
		is_cartesian = (GetCardinality<double>() == left->GetCardinality<double>() * right->GetCardinality<double>());
	}
	result += "cartesian = " + to_string(is_cartesian) + "\n";
	result += "cost = " + to_string(estimated_props->GetCost()) + "\n";
	result += "left = \n";
	if (left) {
		result += left->ToString();
	}
	result += "right = \n";
	if (right) {
		result += right->ToString();
	}
	return result;
}
} // namespace duckdb




#include <algorithm>

namespace duckdb {

using JoinRelationTreeNode = JoinRelationSetManager::JoinRelationTreeNode;

// LCOV_EXCL_START
string JoinRelationSet::ToString() const {
	string result = "[";
	result += StringUtil::Join(relations, count, ", ", [](const idx_t &relation) { return to_string(relation); });
	result += "]";
	return result;
}
// LCOV_EXCL_STOP

//! Returns true if sub is a subset of super
bool JoinRelationSet::IsSubset(JoinRelationSet *super, JoinRelationSet *sub) {
	D_ASSERT(sub->count > 0);
	if (sub->count > super->count) {
		return false;
	}
	idx_t j = 0;
	for (idx_t i = 0; i < super->count; i++) {
		if (sub->relations[j] == super->relations[i]) {
			j++;
			if (j == sub->count) {
				return true;
			}
		}
	}
	return false;
}

JoinRelationSet *JoinRelationSetManager::GetJoinRelation(unique_ptr<idx_t[]> relations, idx_t count) {
	// now look it up in the tree
	JoinRelationTreeNode *info = &root;
	for (idx_t i = 0; i < count; i++) {
		auto entry = info->children.find(relations[i]);
		if (entry == info->children.end()) {
			// node not found, create it
			auto insert_it = info->children.insert(make_pair(relations[i], make_unique<JoinRelationTreeNode>()));
			entry = insert_it.first;
		}
		// move to the next node
		info = entry->second.get();
	}
	// now check if the JoinRelationSet has already been created
	if (!info->relation) {
		// if it hasn't we need to create it
		info->relation = make_unique<JoinRelationSet>(move(relations), count);
	}
	return info->relation.get();
}

//! Create or get a JoinRelationSet from a single node with the given index
JoinRelationSet *JoinRelationSetManager::GetJoinRelation(idx_t index) {
	// create a sorted vector of the relations
	auto relations = unique_ptr<idx_t[]>(new idx_t[1]);
	relations[0] = index;
	idx_t count = 1;
	return GetJoinRelation(move(relations), count);
}

JoinRelationSet *JoinRelationSetManager::GetJoinRelation(unordered_set<idx_t> &bindings) {
	// create a sorted vector of the relations
	unique_ptr<idx_t[]> relations = bindings.empty() ? nullptr : unique_ptr<idx_t[]>(new idx_t[bindings.size()]);
	idx_t count = 0;
	for (auto &entry : bindings) {
		relations[count++] = entry;
	}
	std::sort(relations.get(), relations.get() + count);
	return GetJoinRelation(move(relations), count);
}

JoinRelationSet *JoinRelationSetManager::Union(JoinRelationSet *left, JoinRelationSet *right) {
	auto relations = unique_ptr<idx_t[]>(new idx_t[left->count + right->count]);
	idx_t count = 0;
	// move through the left and right relations, eliminating duplicates
	idx_t i = 0, j = 0;
	while (true) {
		if (i == left->count) {
			// exhausted left relation, add remaining of right relation
			for (; j < right->count; j++) {
				relations[count++] = right->relations[j];
			}
			break;
		} else if (j == right->count) {
			// exhausted right relation, add remaining of left
			for (; i < left->count; i++) {
				relations[count++] = left->relations[i];
			}
			break;
		} else if (left->relations[i] == right->relations[j]) {
			// equivalent, add only one of the two pairs
			relations[count++] = left->relations[i];
			i++;
			j++;
		} else if (left->relations[i] < right->relations[j]) {
			// left is smaller, progress left and add it to the set
			relations[count++] = left->relations[i];
			i++;
		} else {
			// right is smaller, progress right and add it to the set
			relations[count++] = right->relations[j];
			j++;
		}
	}
	return GetJoinRelation(move(relations), count);
}

// JoinRelationSet *JoinRelationSetManager::Difference(JoinRelationSet *left, JoinRelationSet *right) {
// 	auto relations = unique_ptr<idx_t[]>(new idx_t[left->count]);
// 	idx_t count = 0;
// 	// move through the left and right relations
// 	idx_t i = 0, j = 0;
// 	while (true) {
// 		if (i == left->count) {
// 			// exhausted left relation, we are done
// 			break;
// 		} else if (j == right->count) {
// 			// exhausted right relation, add remaining of left
// 			for (; i < left->count; i++) {
// 				relations[count++] = left->relations[i];
// 			}
// 			break;
// 		} else if (left->relations[i] == right->relations[j]) {
// 			// equivalent, add nothing
// 			i++;
// 			j++;
// 		} else if (left->relations[i] < right->relations[j]) {
// 			// left is smaller, progress left and add it to the set
// 			relations[count++] = left->relations[i];
// 			i++;
// 		} else {
// 			// right is smaller, progress right
// 			j++;
// 		}
// 	}
// 	return GetJoinRelation(move(relations), count);
// }

} // namespace duckdb







namespace duckdb {

using QueryEdge = QueryGraph::QueryEdge;

// LCOV_EXCL_START
static string QueryEdgeToString(const QueryEdge *info, vector<idx_t> prefix) {
	string result = "";
	string source = "[";
	for (idx_t i = 0; i < prefix.size(); i++) {
		source += to_string(prefix[i]) + (i < prefix.size() - 1 ? ", " : "");
	}
	source += "]";
	for (auto &entry : info->neighbors) {
		result += StringUtil::Format("%s -> %s\n", source.c_str(), entry->neighbor->ToString().c_str());
	}
	for (auto &entry : info->children) {
		vector<idx_t> new_prefix = prefix;
		new_prefix.push_back(entry.first);
		result += QueryEdgeToString(entry.second.get(), new_prefix);
	}
	return result;
}

string QueryGraph::ToString() const {
	return QueryEdgeToString(&root, {});
}

void QueryGraph::Print() {
	Printer::Print(ToString());
}
// LCOV_EXCL_STOP

QueryEdge *QueryGraph::GetQueryEdge(JoinRelationSet *left) {
	D_ASSERT(left && left->count > 0);
	// find the EdgeInfo corresponding to the left set
	QueryEdge *info = &root;
	for (idx_t i = 0; i < left->count; i++) {
		auto entry = info->children.find(left->relations[i]);
		if (entry == info->children.end()) {
			// node not found, create it
			auto insert_it = info->children.insert(make_pair(left->relations[i], make_unique<QueryEdge>()));
			entry = insert_it.first;
		}
		// move to the next node
		info = entry->second.get();
	}
	return info;
}

void QueryGraph::CreateEdge(JoinRelationSet *left, JoinRelationSet *right, FilterInfo *filter_info) {
	D_ASSERT(left && right && left->count > 0 && right->count > 0);
	// find the EdgeInfo corresponding to the left set
	auto info = GetQueryEdge(left);
	// now insert the edge to the right relation, if it does not exist
	for (idx_t i = 0; i < info->neighbors.size(); i++) {
		if (info->neighbors[i]->neighbor == right) {
			if (filter_info) {
				// neighbor already exists just add the filter, if we have any
				info->neighbors[i]->filters.push_back(filter_info);
			}
			return;
		}
	}
	// neighbor does not exist, create it
	auto n = make_unique<NeighborInfo>();
	if (filter_info) {
		n->filters.push_back(filter_info);
	}
	n->neighbor = right;
	info->neighbors.push_back(move(n));
}

void QueryGraph::EnumerateNeighbors(JoinRelationSet *node, const std::function<bool(NeighborInfo *)> &callback) {
	for (idx_t j = 0; j < node->count; j++) {
		QueryEdge *info = &root;
		for (idx_t i = j; i < node->count; i++) {
			auto entry = info->children.find(node->relations[i]);
			if (entry == info->children.end()) {
				// node not found
				break;
			}
			// check if any subset of the other set is in this sets neighbors
			info = entry->second.get();
			for (auto &neighbor : info->neighbors) {
				if (callback(neighbor.get())) {
					return;
				}
			}
		}
	}
}

//! Returns true if a JoinRelationSet is banned by the list of exclusion_set, false otherwise
static bool JoinRelationSetIsExcluded(JoinRelationSet *node, unordered_set<idx_t> &exclusion_set) {
	return exclusion_set.find(node->relations[0]) != exclusion_set.end();
}

vector<idx_t> QueryGraph::GetNeighbors(JoinRelationSet *node, unordered_set<idx_t> &exclusion_set) {
	unordered_set<idx_t> result;
	EnumerateNeighbors(node, [&](NeighborInfo *info) -> bool {
		if (!JoinRelationSetIsExcluded(info->neighbor, exclusion_set)) {
			// add the smallest node of the neighbor to the set
			result.insert(info->neighbor->relations[0]);
		}
		return false;
	});
	vector<idx_t> neighbors;
	neighbors.insert(neighbors.end(), result.begin(), result.end());
	return neighbors;
}

vector<NeighborInfo *> QueryGraph::GetConnections(JoinRelationSet *node, JoinRelationSet *other) {
	vector<NeighborInfo *> connections;
	EnumerateNeighbors(node, [&](NeighborInfo *info) -> bool {
		if (JoinRelationSet::IsSubset(other, info->neighbor)) {
			connections.push_back(info);
		}
		return false;
	});
	return connections;
}

} // namespace duckdb








#include <algorithm>

namespace std {

//! A JoinNode is defined by the relations it joins.
template <>
struct hash<duckdb::JoinNode> {
	inline string operator()(const duckdb::JoinNode &join_node) const {
		return join_node.set->ToString();
	}
};
} // namespace std

namespace duckdb {

//! Returns true if A and B are disjoint, false otherwise
template <class T>
static bool Disjoint(unordered_set<T> &a, unordered_set<T> &b) {
	for (auto &entry : a) {
		if (b.find(entry) != b.end()) {
			return false;
		}
	}
	return true;
}

//! Extract the set of relations referred to inside an expression
bool JoinOrderOptimizer::ExtractBindings(Expression &expression, unordered_set<idx_t> &bindings) {
	if (expression.type == ExpressionType::BOUND_COLUMN_REF) {
		auto &colref = (BoundColumnRefExpression &)expression;
		D_ASSERT(colref.depth == 0);
		D_ASSERT(colref.binding.table_index != DConstants::INVALID_INDEX);
		// map the base table index to the relation index used by the JoinOrderOptimizer
		D_ASSERT(relation_mapping.find(colref.binding.table_index) != relation_mapping.end());
		auto catalog_table = relation_mapping[colref.binding.table_index];
		auto column_index = colref.binding.column_index;
		cardinality_estimator.AddColumnToRelationMap(catalog_table, column_index);
		bindings.insert(relation_mapping[colref.binding.table_index]);
	}
	if (expression.type == ExpressionType::BOUND_REF) {
		// bound expression
		bindings.clear();
		return false;
	}
	D_ASSERT(expression.type != ExpressionType::SUBQUERY);
	bool can_reorder = true;
	ExpressionIterator::EnumerateChildren(expression, [&](Expression &expr) {
		if (!ExtractBindings(expr, bindings)) {
			can_reorder = false;
			return;
		}
	});
	return can_reorder;
}

void JoinOrderOptimizer::GetColumnBinding(Expression &expression, ColumnBinding &binding) {
	if (expression.type == ExpressionType::BOUND_COLUMN_REF) {
		// Here you have a filter on a single column in a table. Return a binding for the column
		// being filtered on so the filter estimator knows what HLL count to pull
		auto &colref = (BoundColumnRefExpression &)expression;
		D_ASSERT(colref.depth == 0);
		D_ASSERT(colref.binding.table_index != DConstants::INVALID_INDEX);
		// map the base table index to the relation index used by the JoinOrderOptimizer
		D_ASSERT(relation_mapping.find(colref.binding.table_index) != relation_mapping.end());
		binding = ColumnBinding(relation_mapping[colref.binding.table_index], colref.binding.column_index);
	}
	// TODO: handle inequality filters with functions.
	ExpressionIterator::EnumerateChildren(expression, [&](Expression &expr) { GetColumnBinding(expr, binding); });
}

static unique_ptr<LogicalOperator> PushFilter(unique_ptr<LogicalOperator> node, unique_ptr<Expression> expr) {
	// push an expression into a filter
	// first check if we have any filter to push it into
	if (node->type != LogicalOperatorType::LOGICAL_FILTER) {
		// we don't, we need to create one
		auto filter = make_unique<LogicalFilter>();
		filter->children.push_back(move(node));
		node = move(filter);
	}
	// push the filter into the LogicalFilter
	D_ASSERT(node->type == LogicalOperatorType::LOGICAL_FILTER);
	auto filter = (LogicalFilter *)node.get();
	filter->expressions.push_back(move(expr));
	return node;
}

bool JoinOrderOptimizer::ExtractJoinRelations(LogicalOperator &input_op, vector<LogicalOperator *> &filter_operators,
                                              LogicalOperator *parent) {
	LogicalOperator *op = &input_op;
	while (op->children.size() == 1 && (op->type != LogicalOperatorType::LOGICAL_PROJECTION &&
	                                    op->type != LogicalOperatorType::LOGICAL_EXPRESSION_GET)) {
		if (op->type == LogicalOperatorType::LOGICAL_FILTER) {
			// extract join conditions from filter
			filter_operators.push_back(op);
		}
		if (op->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY ||
		    op->type == LogicalOperatorType::LOGICAL_WINDOW) {
			// don't push filters through projection or aggregate and group by
			JoinOrderOptimizer optimizer(context);
			op->children[0] = optimizer.Optimize(move(op->children[0]));
			return false;
		}
		op = op->children[0].get();
	}
	bool non_reorderable_operation = false;
	if (op->type == LogicalOperatorType::LOGICAL_UNION || op->type == LogicalOperatorType::LOGICAL_EXCEPT ||
	    op->type == LogicalOperatorType::LOGICAL_INTERSECT || op->type == LogicalOperatorType::LOGICAL_DELIM_JOIN ||
	    op->type == LogicalOperatorType::LOGICAL_ANY_JOIN) {
		// set operation, optimize separately in children
		non_reorderable_operation = true;
	}

	if (op->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
		auto &join = (LogicalComparisonJoin &)*op;
		if (join.join_type == JoinType::INNER) {
			// extract join conditions from inner join
			filter_operators.push_back(op);
		} else {
			// non-inner join, not reorderable yet
			non_reorderable_operation = true;
			if (join.join_type == JoinType::LEFT && join.right_projection_map.empty()) {
				// for left joins; if the RHS cardinality is significantly larger than the LHS (2x)
				// we convert to doing a RIGHT OUTER JOIN
				// FIXME: for now we don't swap if the right_projection_map is not empty
				// this can be fixed once we implement the left_projection_map properly...
				auto lhs_cardinality = join.children[0]->EstimateCardinality(context);
				auto rhs_cardinality = join.children[1]->EstimateCardinality(context);
				if (rhs_cardinality > lhs_cardinality * 2) {
					join.join_type = JoinType::RIGHT;
					std::swap(join.children[0], join.children[1]);
					for (auto &cond : join.conditions) {
						std::swap(cond.left, cond.right);
						cond.comparison = FlipComparisionExpression(cond.comparison);
					}
				}
			}
		}
	}
	if (non_reorderable_operation) {
		// we encountered a non-reordable operation (setop or non-inner join)
		// we do not reorder non-inner joins yet, however we do want to expand the potential join graph around them
		// non-inner joins are also tricky because we can't freely make conditions through them
		// e.g. suppose we have (left LEFT OUTER JOIN right WHERE right IS NOT NULL), the join can generate
		// new NULL values in the right side, so pushing this condition through the join leads to incorrect results
		// for this reason, we just start a new JoinOptimizer pass in each of the children of the join

		// Keep track of all of the filter bindings the new join order optimizer makes
		vector<column_binding_map_t<ColumnBinding>> child_binding_maps;
		idx_t child_bindings_it = 0;
		for (auto &child : op->children) {
			child_binding_maps.emplace_back(column_binding_map_t<ColumnBinding>());
			JoinOrderOptimizer optimizer(context);
			child = optimizer.Optimize(move(child));
			// save the relation bindings from the optimized child. These later all get added to the
			// parent cardinality_estimator relation column binding map.
			optimizer.cardinality_estimator.CopyRelationMap(child_binding_maps.at(child_bindings_it));
			child_bindings_it += 1;
		}
		// after this we want to treat this node as one  "end node" (like e.g. a base relation)
		// however the join refers to multiple base relations
		// enumerate all base relations obtained from this join and add them to the relation mapping
		// also, we have to resolve the join conditions for the joins here
		// get the left and right bindings
		unordered_set<idx_t> bindings;
		LogicalJoin::GetTableReferences(*op, bindings);
		// now create the relation that refers to all these bindings
		auto relation = make_unique<SingleJoinRelation>(&input_op, parent);
		auto relation_id = relations.size();
		// Add binding information from the nonreorderable join to this relation.
		for (idx_t it : bindings) {
			cardinality_estimator.MergeBindings(it, relation_id, child_binding_maps);
			relation_mapping[it] = relation_id;
		}
		relations.push_back(move(relation));
		return true;
	}
	if (op->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN ||
	    op->type == LogicalOperatorType::LOGICAL_CROSS_PRODUCT) {
		// inner join or cross product
		bool can_reorder_left = ExtractJoinRelations(*op->children[0], filter_operators, op);
		bool can_reorder_right = ExtractJoinRelations(*op->children[1], filter_operators, op);
		return can_reorder_left && can_reorder_right;
	} else if (op->type == LogicalOperatorType::LOGICAL_GET) {
		// base table scan, add to set of relations
		auto get = (LogicalGet *)op;
		auto relation = make_unique<SingleJoinRelation>(&input_op, parent);
		idx_t relation_id = relations.size();
		//! make sure the optimizer has knowledge of the exact column bindings as well.
		auto table_index = get->table_index;
		relation_mapping[table_index] = relation_id;
		cardinality_estimator.AddRelationColumnMapping(get, relation_id);
		relations.push_back(move(relation));
		return true;
	} else if (op->type == LogicalOperatorType::LOGICAL_EXPRESSION_GET) {
		// base table scan, add to set of relations
		auto get = (LogicalExpressionGet *)op;
		auto relation = make_unique<SingleJoinRelation>(&input_op, parent);
		idx_t relation_id = relations.size();
		//! make sure the optimizer has knowledge of the exact column bindings as well.
		auto table_index = get->table_index;
		relation_mapping[table_index] = relation_id;
		relations.push_back(move(relation));
		return true;
	} else if (op->type == LogicalOperatorType::LOGICAL_DUMMY_SCAN) {
		// table function call, add to set of relations
		auto dummy_scan = (LogicalDummyScan *)op;
		auto relation = make_unique<SingleJoinRelation>(&input_op, parent);
		relation_mapping[dummy_scan->table_index] = relations.size();
		relations.push_back(move(relation));
		return true;
	} else if (op->type == LogicalOperatorType::LOGICAL_PROJECTION) {
		auto proj = (LogicalProjection *)op;
		// we run the join order optimizer witin the subquery as well
		JoinOrderOptimizer optimizer(context);
		op->children[0] = optimizer.Optimize(move(op->children[0]));
		// projection, add to the set of relations
		auto relation = make_unique<SingleJoinRelation>(&input_op, parent);
		relation_mapping[proj->table_index] = relations.size();
		relations.push_back(move(relation));
		return true;
	}
	return false;
}

//! Update the exclusion set with all entries in the subgraph
static void UpdateExclusionSet(JoinRelationSet *node, unordered_set<idx_t> &exclusion_set) {
	for (idx_t i = 0; i < node->count; i++) {
		exclusion_set.insert(node->relations[i]);
	}
}

//! Create a new JoinTree node by joining together two previous JoinTree nodes
unique_ptr<JoinNode> JoinOrderOptimizer::CreateJoinTree(JoinRelationSet *set,
                                                        const vector<NeighborInfo *> &possible_connections,
                                                        JoinNode *left, JoinNode *right) {
	// for the hash join we want the right side (build side) to have the smallest cardinality
	// also just a heuristic but for now...
	// FIXME: we should probably actually benchmark that as well
	// FIXME: should consider different join algorithms, should we pick a join algorithm here as well? (probably)
	double expected_cardinality;
	NeighborInfo *best_connection = nullptr;
	auto plan = plans.find(set);
	// if we have already calculated an expected cardinality for this set,
	// just re-use that cardinality
	if (left->GetCardinality<double>() < right->GetCardinality<double>()) {
		return CreateJoinTree(set, possible_connections, right, left);
	}
	if (plan != plans.end()) {
		if (!plan->second) {
			throw InternalException("No plan: internal error in join order optimizer");
		}
		expected_cardinality = plan->second->GetCardinality<double>();
		best_connection = possible_connections.back();
	} else if (possible_connections.empty()) {
		// cross product
		expected_cardinality = cardinality_estimator.EstimateCrossProduct(left, right);
	} else {
		// normal join, expect foreign key join
		expected_cardinality = cardinality_estimator.EstimateCardinalityWithSet(set);
		best_connection = possible_connections.back();
	}

	auto cost = CardinalityEstimator::ComputeCost(left, right, expected_cardinality);
	auto result = make_unique<JoinNode>(set, best_connection, left, right, expected_cardinality, cost);
	D_ASSERT(cost >= expected_cardinality);
	return result;
}

void JoinOrderOptimizer::UpdateJoinNodesInFullPlan(JoinNode *node) {
	if (!node) {
		return;
	}
	if (node->set->count == relations.size()) {
		join_nodes_in_full_plan.clear();
	}
	if (node->set->count < relations.size()) {
		join_nodes_in_full_plan.insert(node->set->ToString());
	}
	UpdateJoinNodesInFullPlan(node->left);
	UpdateJoinNodesInFullPlan(node->right);
}

JoinNode *JoinOrderOptimizer::EmitPair(JoinRelationSet *left, JoinRelationSet *right,
                                       const vector<NeighborInfo *> &info) {
	// get the left and right join plans
	auto &left_plan = plans[left];
	auto &right_plan = plans[right];
	if (!left_plan || !right_plan) {
		throw InternalException("No left or right plan: internal error in join order optimizer");
	}
	auto new_set = set_manager.Union(left, right);
	// create the join tree based on combining the two plans
	auto new_plan = CreateJoinTree(new_set, info, left_plan.get(), right_plan.get());
	// check if this plan is the optimal plan we found for this set of relations
	auto entry = plans.find(new_set);
	if (entry == plans.end() || new_plan->GetCost() < entry->second->GetCost()) {
		// the plan is the optimal plan, move it into the dynamic programming tree
		auto result = new_plan.get();

		//! make sure plans are symmetric for cardinality estimation
		if (entry != plans.end()) {
			cardinality_estimator.VerifySymmetry(result, entry->second.get());
		}
		if (full_plan_found &&
		    join_nodes_in_full_plan.find(new_plan->set->ToString()) != join_nodes_in_full_plan.end()) {
			must_update_full_plan = true;
		}
		if (new_set->count == relations.size()) {
			full_plan_found = true;
			// If we find a full plan, we need to keep track of which nodes are in the full plan.
			// It's possible the DP algorithm updates one of these nodes, then goes on to solve
			// the order approximately. In the approximate algorithm, it's not guaranteed that the
			// node references are updated. If the original full plan is determined to still have
			// the lowest cost, it's possible to get use-after-free errors.
			// If we know a node in the full plan is updated, we can prevent ourselves from exiting the
			// DP algorithm until the last plan updated is a full plan
			UpdateJoinNodesInFullPlan(result);
			if (must_update_full_plan) {
				must_update_full_plan = false;
			}
		}

		D_ASSERT(new_plan);
		plans[new_set] = move(new_plan);
		return result;
	}
	return entry->second.get();
}

bool JoinOrderOptimizer::TryEmitPair(JoinRelationSet *left, JoinRelationSet *right,
                                     const vector<NeighborInfo *> &info) {
	pairs++;
	// If a full plan is created, it's possible a node in the plan gets updated. When this happens, make sure you keep
	// emitting pairs until you emit another final plan. Another final plan is guaranteed to be produced because of
	// our symmetry guarantees.
	if (pairs >= 10000 && !must_update_full_plan) {
		// when the amount of pairs gets too large we exit the dynamic programming and resort to a greedy algorithm
		// FIXME: simple heuristic currently
		// at 10K pairs stop searching exactly and switch to heuristic
		return false;
	}
	EmitPair(left, right, info);
	return true;
}

bool JoinOrderOptimizer::EmitCSG(JoinRelationSet *node) {
	if (node->count == relations.size()) {
		return true;
	}
	// create the exclusion set as everything inside the subgraph AND anything with members BELOW it
	unordered_set<idx_t> exclusion_set;
	for (idx_t i = 0; i < node->relations[0]; i++) {
		exclusion_set.insert(i);
	}
	UpdateExclusionSet(node, exclusion_set);
	// find the neighbors given this exclusion set
	auto neighbors = query_graph.GetNeighbors(node, exclusion_set);
	if (neighbors.empty()) {
		return true;
	}

	//! Neighbors should be reversed when iterating over them.
	std::sort(neighbors.begin(), neighbors.end(), std::greater_equal<idx_t>());
	for (idx_t i = 0; i < neighbors.size() - 1; i++) {
		D_ASSERT(neighbors[i] >= neighbors[i + 1]);
	}
	for (auto neighbor : neighbors) {
		// since the GetNeighbors only returns the smallest element in a list, the entry might not be connected to
		// (only!) this neighbor,  hence we have to do a connectedness check before we can emit it
		auto neighbor_relation = set_manager.GetJoinRelation(neighbor);
		auto connections = query_graph.GetConnections(node, neighbor_relation);
		if (!connections.empty()) {
			if (!TryEmitPair(node, neighbor_relation, connections)) {
				return false;
			}
		}
		if (!EnumerateCmpRecursive(node, neighbor_relation, exclusion_set)) {
			return false;
		}
	}
	return true;
}

bool JoinOrderOptimizer::EnumerateCmpRecursive(JoinRelationSet *left, JoinRelationSet *right,
                                               unordered_set<idx_t> exclusion_set) {
	// get the neighbors of the second relation under the exclusion set
	auto neighbors = query_graph.GetNeighbors(right, exclusion_set);
	if (neighbors.empty()) {
		return true;
	}
	vector<JoinRelationSet *> union_sets;
	union_sets.resize(neighbors.size());
	for (idx_t i = 0; i < neighbors.size(); i++) {
		auto neighbor = set_manager.GetJoinRelation(neighbors[i]);
		// emit the combinations of this node and its neighbors
		auto combined_set = set_manager.Union(right, neighbor);
		if (combined_set->count > right->count && plans.find(combined_set) != plans.end()) {
			auto connections = query_graph.GetConnections(left, combined_set);
			if (!connections.empty()) {
				if (!TryEmitPair(left, combined_set, connections)) {
					return false;
				}
			}
		}
		union_sets[i] = combined_set;
	}
	// recursively enumerate the sets
	unordered_set<idx_t> new_exclusion_set = exclusion_set;
	for (idx_t i = 0; i < neighbors.size(); i++) {
		// updated the set of excluded entries with this neighbor
		new_exclusion_set.insert(neighbors[i]);
		if (!EnumerateCmpRecursive(left, union_sets[i], new_exclusion_set)) {
			return false;
		}
	}
	return true;
}

bool JoinOrderOptimizer::EnumerateCSGRecursive(JoinRelationSet *node, unordered_set<idx_t> &exclusion_set) {
	// find neighbors of S under the exclusion set
	auto neighbors = query_graph.GetNeighbors(node, exclusion_set);
	if (neighbors.empty()) {
		return true;
	}
	vector<JoinRelationSet *> union_sets;
	union_sets.resize(neighbors.size());
	for (idx_t i = 0; i < neighbors.size(); i++) {
		auto neighbor = set_manager.GetJoinRelation(neighbors[i]);
		// emit the combinations of this node and its neighbors
		auto new_set = set_manager.Union(node, neighbor);
		if (new_set->count > node->count && plans.find(new_set) != plans.end()) {
			if (!EmitCSG(new_set)) {
				return false;
			}
		}
		union_sets[i] = new_set;
	}
	// recursively enumerate the sets
	unordered_set<idx_t> new_exclusion_set = exclusion_set;
	for (idx_t i = 0; i < neighbors.size(); i++) {
		// Reset the exclusion set so that the algorithm considers all combinations
		// of the exclusion_set with a subset of neighbors.
		new_exclusion_set = exclusion_set;
		new_exclusion_set.insert(neighbors[i]);
		// updated the set of excluded entries with this neighbor
		if (!EnumerateCSGRecursive(union_sets[i], new_exclusion_set)) {
			return false;
		}
	}
	return true;
}

bool JoinOrderOptimizer::SolveJoinOrderExactly() {
	// now we perform the actual dynamic programming to compute the final result
	// we enumerate over all the possible pairs in the neighborhood
	for (idx_t i = relations.size(); i > 0; i--) {
		// for every node in the set, we consider it as the start node once
		auto start_node = set_manager.GetJoinRelation(i - 1);
		// emit the start node
		if (!EmitCSG(start_node)) {
			return false;
		}
		// initialize the set of exclusion_set as all the nodes with a number below this
		unordered_set<idx_t> exclusion_set;
		for (idx_t j = 0; j < i - 1; j++) {
			exclusion_set.insert(j);
		}
		// then we recursively search for neighbors that do not belong to the banned entries
		if (!EnumerateCSGRecursive(start_node, exclusion_set)) {
			return false;
		}
	}
	return true;
}

static vector<unordered_set<idx_t>> AddSuperSets(vector<unordered_set<idx_t>> current,
                                                 const vector<idx_t> &all_neighbors) {
	vector<unordered_set<idx_t>> ret;
	for (auto &neighbor : all_neighbors) {
		for (auto &neighbor_set : current) {
			auto max_val = std::max_element(neighbor_set.begin(), neighbor_set.end());
			if (*max_val >= neighbor) {
				continue;
			}
			if (neighbor_set.count(neighbor) == 0) {
				unordered_set<idx_t> new_set;
				for (auto &n : neighbor_set) {
					new_set.insert(n);
				}
				new_set.insert(neighbor);
				ret.push_back(new_set);
			}
		}
	}
	return ret;
}

// works by first creating all sets with cardinality 1
// then iterates over each previously create group of subsets and will only add a neighbor if the neighbor
// is greater than all relations in the set.
static vector<unordered_set<idx_t>> GetAllNeighborSets(JoinRelationSet *new_set, unordered_set<idx_t> &exclusion_set,
                                                       vector<idx_t> neighbors) {
	vector<unordered_set<idx_t>> ret;
	sort(neighbors.begin(), neighbors.end());
	vector<unordered_set<idx_t>> added;
	for (auto &neighbor : neighbors) {
		added.push_back(unordered_set<idx_t>({neighbor}));
		ret.push_back(unordered_set<idx_t>({neighbor}));
	}
	do {
		added = AddSuperSets(added, neighbors);
		for (auto &d : added) {
			ret.push_back(d);
		}
	} while (!added.empty());
#if DEBUG
	// drive by test to make sure we have an accurate amount of
	// subsets, and that each neighbor is in a correct amount
	// of those subsets.
	D_ASSERT(ret.size() == pow(2, neighbors.size()) - 1);
	for (auto &n : neighbors) {
		idx_t count = 0;
		for (auto &set : ret) {
			if (set.count(n) >= 1) {
				count += 1;
			}
		}
		D_ASSERT(count == pow(2, neighbors.size() - 1));
	}
#endif
	return ret;
}

void JoinOrderOptimizer::UpdateDPTree(JoinNode *new_plan) {
	auto new_set = new_plan->set;
	// now update every plan that uses this plan
	unordered_set<idx_t> exclusion_set;
	for (idx_t i = 0; i < new_set->count; i++) {
		exclusion_set.insert(new_set->relations[i]);
	}
	auto neighbors = query_graph.GetNeighbors(new_set, exclusion_set);
	auto all_neighbors = GetAllNeighborSets(new_set, exclusion_set, neighbors);
	for (auto neighbor : all_neighbors) {
		auto neighbor_relation = set_manager.GetJoinRelation(neighbor);
		auto combined_set = set_manager.Union(new_set, neighbor_relation);

		auto combined_set_plan = plans.find(combined_set);
		if (combined_set_plan == plans.end()) {
			continue;
		}

		double combined_set_plan_cost = combined_set_plan->second->GetCost();
		auto connections = query_graph.GetConnections(new_set, neighbor_relation);
		// recurse and update up the tree if the combined set produces a plan with a lower cost
		// only recurse on neighbor relations that have plans.
		auto right_plan = plans.find(neighbor_relation);
		if (right_plan == plans.end()) {
			continue;
		}
		auto updated_plan = EmitPair(new_set, neighbor_relation, connections);
		// <= because the child node has already been replaced. You need to
		// replace the parent node as well in this case
		if (updated_plan->GetCost() < combined_set_plan_cost) {
			UpdateDPTree(updated_plan);
		}
	}
}

void JoinOrderOptimizer::SolveJoinOrderApproximately() {
	// at this point, we exited the dynamic programming but did not compute the final join order because it took too
	// long instead, we use a greedy heuristic to obtain a join ordering now we use Greedy Operator Ordering to
	// construct the result tree first we start out with all the base relations (the to-be-joined relations)
	vector<JoinRelationSet *> join_relations; // T in the paper
	for (idx_t i = 0; i < relations.size(); i++) {
		join_relations.push_back(set_manager.GetJoinRelation(i));
	}
	while (join_relations.size() > 1) {
		// now in every step of the algorithm, we greedily pick the join between the to-be-joined relations that has the
		// smallest cost. This is O(r^2) per step, and every step will reduce the total amount of relations to-be-joined
		// by 1, so the total cost is O(r^3) in the amount of relations
		idx_t best_left = 0, best_right = 0;
		JoinNode *best_connection = nullptr;
		for (idx_t i = 0; i < join_relations.size(); i++) {
			auto left = join_relations[i];
			for (idx_t j = i + 1; j < join_relations.size(); j++) {
				auto right = join_relations[j];
				// check if we can connect these two relations
				auto connection = query_graph.GetConnections(left, right);
				if (!connection.empty()) {
					// we can check the cost of this connection
					auto node = EmitPair(left, right, connection);

					// update the DP tree in case a plan created by the DP algorithm uses the node
					// that was potentially just updated by EmitPair. You will get a use-after-free
					// error if future plans rely on the old node that was just replaced.
					UpdateDPTree(node);

					if (!best_connection || node->GetCost() < best_connection->GetCost()) {
						// best pair found so far
						best_connection = node;
						best_left = i;
						best_right = j;
					}
				}
			}
		}
		if (!best_connection) {
			// could not find a connection, but we were not done with finding a completed plan
			// we have to add a cross product; we add it between the two smallest relations
			JoinNode *smallest_plans[2] = {nullptr};
			idx_t smallest_index[2];
			for (idx_t i = 0; i < join_relations.size(); i++) {
				// get the plan for this relation
				auto current_plan = plans[join_relations[i]].get();
				// check if the cardinality is smaller than the smallest two found so far
				for (idx_t j = 0; j < 2; j++) {
					if (!smallest_plans[j] ||
					    smallest_plans[j]->GetCardinality<double>() > current_plan->GetCardinality<double>()) {
						smallest_plans[j] = current_plan;
						smallest_index[j] = i;
						break;
					}
				}
			}
			if (!smallest_plans[0] || !smallest_plans[1]) {
				throw InternalException("Internal error in join order optimizer");
			}
			D_ASSERT(smallest_plans[0] && smallest_plans[1]);
			D_ASSERT(smallest_index[0] != smallest_index[1]);
			auto left = smallest_plans[0]->set;
			auto right = smallest_plans[1]->set;
			// create a cross product edge (i.e. edge with empty filter) between these two sets in the query graph
			query_graph.CreateEdge(left, right, nullptr);
			// now emit the pair and continue with the algorithm
			auto connections = query_graph.GetConnections(left, right);
			D_ASSERT(!connections.empty());

			best_connection = EmitPair(left, right, connections);
			best_left = smallest_index[0];
			best_right = smallest_index[1];

			UpdateDPTree(best_connection);
			// the code below assumes best_right > best_left
			if (best_left > best_right) {
				std::swap(best_left, best_right);
			}
		}
		// now update the to-be-checked pairs
		// remove left and right, and add the combination

		// important to erase the biggest element first
		// if we erase the smallest element first the index of the biggest element changes
		D_ASSERT(best_right > best_left);
		join_relations.erase(join_relations.begin() + best_right);
		join_relations.erase(join_relations.begin() + best_left);
		join_relations.push_back(best_connection->set);
	}
}

void JoinOrderOptimizer::SolveJoinOrder() {
	// first try to solve the join order exactly
	if (!SolveJoinOrderExactly()) {
		// otherwise, if that times out we resort to a greedy algorithm
		SolveJoinOrderApproximately();
	}
}

void JoinOrderOptimizer::GenerateCrossProducts() {
	// generate a set of cross products to combine the currently available plans into a full join plan
	// we create edges between every relation with a high cost
	for (idx_t i = 0; i < relations.size(); i++) {
		auto left = set_manager.GetJoinRelation(i);
		for (idx_t j = 0; j < relations.size(); j++) {
			if (i != j) {
				auto right = set_manager.GetJoinRelation(j);
				query_graph.CreateEdge(left, right, nullptr);
				query_graph.CreateEdge(right, left, nullptr);
			}
		}
	}
}

static unique_ptr<LogicalOperator> ExtractJoinRelation(SingleJoinRelation &rel) {
	auto &children = rel.parent->children;
	for (idx_t i = 0; i < children.size(); i++) {
		if (children[i].get() == rel.op) {
			// found it! take ownership of it from the parent
			auto result = move(children[i]);
			children.erase(children.begin() + i);
			return result;
		}
	}
	throw Exception("Could not find relation in parent node (?)");
}

pair<JoinRelationSet *, unique_ptr<LogicalOperator>>
JoinOrderOptimizer::GenerateJoins(vector<unique_ptr<LogicalOperator>> &extracted_relations, JoinNode *node) {
	JoinRelationSet *left_node = nullptr, *right_node = nullptr;
	JoinRelationSet *result_relation;
	unique_ptr<LogicalOperator> result_operator;
	if (node->left && node->right) {
		// generate the left and right children
		auto left = GenerateJoins(extracted_relations, node->left);
		auto right = GenerateJoins(extracted_relations, node->right);

		if (node->info->filters.empty()) {
			// no filters, create a cross product
			result_operator = LogicalCrossProduct::Create(move(left.second), move(right.second));
		} else {
			// we have filters, create a join node
			auto join = make_unique<LogicalComparisonJoin>(JoinType::INNER);
			join->children.push_back(move(left.second));
			join->children.push_back(move(right.second));
			// set the join conditions from the join node
			for (auto &f : node->info->filters) {
				// extract the filter from the operator it originally belonged to
				D_ASSERT(filters[f->filter_index]);
				auto condition = move(filters[f->filter_index]);
				// now create the actual join condition
				D_ASSERT((JoinRelationSet::IsSubset(left.first, f->left_set) &&
				          JoinRelationSet::IsSubset(right.first, f->right_set)) ||
				         (JoinRelationSet::IsSubset(left.first, f->right_set) &&
				          JoinRelationSet::IsSubset(right.first, f->left_set)));
				JoinCondition cond;
				D_ASSERT(condition->GetExpressionClass() == ExpressionClass::BOUND_COMPARISON);
				auto &comparison = (BoundComparisonExpression &)*condition;
				// we need to figure out which side is which by looking at the relations available to us
				bool invert = !JoinRelationSet::IsSubset(left.first, f->left_set);
				cond.left = !invert ? move(comparison.left) : move(comparison.right);
				cond.right = !invert ? move(comparison.right) : move(comparison.left);
				cond.comparison = condition->type;

				if (invert) {
					// reverse comparison expression if we reverse the order of the children
					cond.comparison = FlipComparisionExpression(cond.comparison);
				}
				join->conditions.push_back(move(cond));
			}
			D_ASSERT(!join->conditions.empty());
			result_operator = move(join);
		}
		left_node = left.first;
		right_node = right.first;
		right_node = right.first;
		result_relation = set_manager.Union(left_node, right_node);
	} else {
		// base node, get the entry from the list of extracted relations
		D_ASSERT(node->set->count == 1);
		D_ASSERT(extracted_relations[node->set->relations[0]]);
		result_relation = node->set;
		result_operator = move(extracted_relations[node->set->relations[0]]);
	}
	result_operator->estimated_cardinality = node->GetCardinality<idx_t>();
	result_operator->has_estimated_cardinality = true;
	result_operator->estimated_props = node->estimated_props->Copy();
	// check if we should do a pushdown on this node
	// basically, any remaining filter that is a subset of the current relation will no longer be used in joins
	// hence we should push it here
	for (auto &filter_info : filter_infos) {
		// check if the filter has already been extracted
		auto info = filter_info.get();
		if (filters[info->filter_index]) {
			// now check if the filter is a subset of the current relation
			// note that infos with an empty relation set are a special case and we do not push them down
			if (info->set->count > 0 && JoinRelationSet::IsSubset(result_relation, info->set)) {
				auto filter = move(filters[info->filter_index]);
				// if it is, we can push the filter
				// we can push it either into a join or as a filter
				// check if we are in a join or in a base table
				if (!left_node || !info->left_set) {
					// base table or non-comparison expression, push it as a filter
					result_operator = PushFilter(move(result_operator), move(filter));
					continue;
				}
				// the node below us is a join or cross product and the expression is a comparison
				// check if the nodes can be split up into left/right
				bool found_subset = false;
				bool invert = false;
				if (JoinRelationSet::IsSubset(left_node, info->left_set) &&
				    JoinRelationSet::IsSubset(right_node, info->right_set)) {
					found_subset = true;
				} else if (JoinRelationSet::IsSubset(right_node, info->left_set) &&
				           JoinRelationSet::IsSubset(left_node, info->right_set)) {
					invert = true;
					found_subset = true;
				}
				if (!found_subset) {
					// could not be split up into left/right
					result_operator = PushFilter(move(result_operator), move(filter));
					continue;
				}
				// create the join condition
				JoinCondition cond;
				D_ASSERT(filter->GetExpressionClass() == ExpressionClass::BOUND_COMPARISON);
				auto &comparison = (BoundComparisonExpression &)*filter;
				// we need to figure out which side is which by looking at the relations available to us
				cond.left = !invert ? move(comparison.left) : move(comparison.right);
				cond.right = !invert ? move(comparison.right) : move(comparison.left);
				cond.comparison = comparison.type;
				if (invert) {
					// reverse comparison expression if we reverse the order of the children
					cond.comparison = FlipComparisionExpression(comparison.type);
				}
				// now find the join to push it into
				auto node = result_operator.get();
				if (node->type == LogicalOperatorType::LOGICAL_FILTER) {
					node = node->children[0].get();
				}
				if (node->type == LogicalOperatorType::LOGICAL_CROSS_PRODUCT) {
					// turn into comparison join
					auto comp_join = make_unique<LogicalComparisonJoin>(JoinType::INNER);
					comp_join->children.push_back(move(node->children[0]));
					comp_join->children.push_back(move(node->children[1]));
					comp_join->conditions.push_back(move(cond));
					if (node == result_operator.get()) {
						result_operator = move(comp_join);
					} else {
						D_ASSERT(result_operator->type == LogicalOperatorType::LOGICAL_FILTER);
						result_operator->children[0] = move(comp_join);
					}
				} else {
					D_ASSERT(node->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN);
					auto &comp_join = (LogicalComparisonJoin &)*node;
					comp_join.conditions.push_back(move(cond));
				}
			}
		}
	}
	return make_pair(result_relation, move(result_operator));
}

unique_ptr<LogicalOperator> JoinOrderOptimizer::RewritePlan(unique_ptr<LogicalOperator> plan, JoinNode *node) {
	// now we have to rewrite the plan
	bool root_is_join = plan->children.size() > 1;

	// first we will extract all relations from the main plan
	vector<unique_ptr<LogicalOperator>> extracted_relations;
	for (auto &relation : relations) {
		extracted_relations.push_back(ExtractJoinRelation(*relation));
	}
	// now we generate the actual joins
	auto join_tree = GenerateJoins(extracted_relations, node);
	// perform the final pushdown of remaining filters
	for (auto &filter : filters) {
		// check if the filter has already been extracted
		if (filter) {
			// if not we need to push it
			join_tree.second = PushFilter(move(join_tree.second), move(filter));
		}
	}

	// find the first join in the relation to know where to place this node
	if (root_is_join) {
		// first node is the join, return it immediately
		return move(join_tree.second);
	}
	D_ASSERT(plan->children.size() == 1);
	// have to move up through the relations
	auto op = plan.get();
	auto parent = plan.get();
	while (op->type != LogicalOperatorType::LOGICAL_CROSS_PRODUCT &&
	       op->type != LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
		D_ASSERT(op->children.size() == 1);
		parent = op;
		op = op->children[0].get();
	}
	// have to replace at this node
	parent->children[0] = move(join_tree.second);
	return plan;
}

// the join ordering is pretty much a straight implementation of the paper "Dynamic Programming Strikes Back" by Guido
// Moerkotte and Thomas Neumannn, see that paper for additional info/documentation bonus slides:
// https://db.in.tum.de/teaching/ws1415/queryopt/chapter3.pdf?lang=de
// FIXME: incorporate cardinality estimation into the plans, possibly by pushing samples?
unique_ptr<LogicalOperator> JoinOrderOptimizer::Optimize(unique_ptr<LogicalOperator> plan) {
	D_ASSERT(filters.empty() && relations.empty()); // assert that the JoinOrderOptimizer has not been used before
	LogicalOperator *op = plan.get();
	// now we optimize the current plan
	// we skip past until we find the first projection, we do this because the HAVING clause inserts a Filter AFTER the
	// group by and this filter cannot be reordered
	// extract a list of all relations that have to be joined together
	// and a list of all conditions that is applied to them
	vector<LogicalOperator *> filter_operators;
	if (!ExtractJoinRelations(*op, filter_operators)) {
		// do not support reordering this type of plan
		return plan;
	}
	if (relations.size() <= 1) {
		// at most one relation, nothing to reorder
		return plan;
	}
	// now that we know we are going to perform join ordering we actually extract the filters, eliminating duplicate
	// filters in the process
	expression_set_t filter_set;
	for (auto &op : filter_operators) {
		if (op->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
			auto &join = (LogicalComparisonJoin &)*op;
			D_ASSERT(join.join_type == JoinType::INNER);
			D_ASSERT(join.expressions.empty());
			for (auto &cond : join.conditions) {
				auto comparison =
				    make_unique<BoundComparisonExpression>(cond.comparison, move(cond.left), move(cond.right));
				if (filter_set.find(comparison.get()) == filter_set.end()) {
					filter_set.insert(comparison.get());
					filters.push_back(move(comparison));
				}
			}
			join.conditions.clear();
		} else {
			for (auto &expression : op->expressions) {
				if (filter_set.find(expression.get()) == filter_set.end()) {
					filter_set.insert(expression.get());
					filters.push_back(move(expression));
				}
			}
			op->expressions.clear();
		}
	}
	// create potential edges from the comparisons
	for (idx_t i = 0; i < filters.size(); i++) {
		auto &filter = filters[i];
		auto info = make_unique<FilterInfo>();
		auto filter_info = info.get();
		filter_infos.push_back(move(info));
		// first extract the relation set for the entire filter
		unordered_set<idx_t> bindings;
		ExtractBindings(*filter, bindings);
		filter_info->set = set_manager.GetJoinRelation(bindings);
		filter_info->filter_index = i;
		// now check if it can be used as a join predicate
		if (filter->GetExpressionClass() == ExpressionClass::BOUND_COMPARISON) {
			auto comparison = (BoundComparisonExpression *)filter.get();
			// extract the bindings that are required for the left and right side of the comparison
			unordered_set<idx_t> left_bindings, right_bindings;
			ExtractBindings(*comparison->left, left_bindings);
			ExtractBindings(*comparison->right, right_bindings);
			GetColumnBinding(*comparison->left, filter_info->left_binding);
			GetColumnBinding(*comparison->right, filter_info->right_binding);
			if (!left_bindings.empty() && !right_bindings.empty()) {
				// both the left and the right side have bindings
				// first create the relation sets, if they do not exist
				filter_info->left_set = set_manager.GetJoinRelation(left_bindings);
				filter_info->right_set = set_manager.GetJoinRelation(right_bindings);
				// we can only create a meaningful edge if the sets are not exactly the same
				if (filter_info->left_set != filter_info->right_set) {
					// check if the sets are disjoint
					if (Disjoint(left_bindings, right_bindings)) {
						// they are disjoint, we only need to create one set of edges in the join graph
						query_graph.CreateEdge(filter_info->left_set, filter_info->right_set, filter_info);
						query_graph.CreateEdge(filter_info->right_set, filter_info->left_set, filter_info);
					} else {
						continue;
					}
					continue;
				}
			}
		}
	}
	// now use dynamic programming to figure out the optimal join order
	// First we initialize each of the single-node plans with themselves and with their cardinalities these are the leaf
	// nodes of the join tree NOTE: we can just use pointers to JoinRelationSet* here because the GetJoinRelation
	// function ensures that a unique combination of relations will have a unique JoinRelationSet object.
	vector<NodeOp> nodes_ops;
	for (idx_t i = 0; i < relations.size(); i++) {
		auto &rel = *relations[i];
		auto node = set_manager.GetJoinRelation(i);
		nodes_ops.emplace_back(NodeOp(make_unique<JoinNode>(node, 0), rel.op));
	}

	cardinality_estimator.InitCardinalityEstimatorProps(&nodes_ops, &filter_infos);

	for (auto &node_op : nodes_ops) {
		D_ASSERT(node_op.node);
		plans[node_op.node->set] = move(node_op.node);
	}
	// now we perform the actual dynamic programming to compute the final result
	SolveJoinOrder();
	// now the optimal join path should have been found
	// get it from the node
	unordered_set<idx_t> bindings;
	for (idx_t i = 0; i < relations.size(); i++) {
		bindings.insert(i);
	}
	auto total_relation = set_manager.GetJoinRelation(bindings);
	auto final_plan = plans.find(total_relation);
	if (final_plan == plans.end()) {
		// could not find the final plan
		// this should only happen in case the sets are actually disjunct
		// in this case we need to generate cross product to connect the disjoint sets
		if (context.config.force_no_cross_product) {
			throw InvalidInputException(
			    "Query requires a cross-product, but 'force_no_cross_product' PRAGMA is enabled");
		}
		GenerateCrossProducts();
		//! solve the join order again
		SolveJoinOrder();
		// now we can obtain the final plan!
		final_plan = plans.find(total_relation);
		D_ASSERT(final_plan != plans.end());
	}
	// now perform the actual reordering
	return RewritePlan(move(plan), final_plan->second.get());
}

} // namespace duckdb




namespace duckdb {

bool ExpressionMatcher::Match(Expression *expr, vector<Expression *> &bindings) {
	if (type && !type->Match(expr->return_type)) {
		return false;
	}
	if (expr_type && !expr_type->Match(expr->type)) {
		return false;
	}
	if (expr_class != ExpressionClass::INVALID && expr_class != expr->GetExpressionClass()) {
		return false;
	}
	bindings.push_back(expr);
	return true;
}

bool ExpressionEqualityMatcher::Match(Expression *expr, vector<Expression *> &bindings) {
	if (!Expression::Equals(expression, expr)) {
		return false;
	}
	bindings.push_back(expr);
	return true;
}

bool CaseExpressionMatcher::Match(Expression *expr_p, vector<Expression *> &bindings) {
	if (!ExpressionMatcher::Match(expr_p, bindings)) {
		return false;
	}
	return true;
}

bool ComparisonExpressionMatcher::Match(Expression *expr_p, vector<Expression *> &bindings) {
	if (!ExpressionMatcher::Match(expr_p, bindings)) {
		return false;
	}
	auto expr = (BoundComparisonExpression *)expr_p;
	vector<Expression *> expressions = {expr->left.get(), expr->right.get()};
	return SetMatcher::Match(matchers, expressions, bindings, policy);
}

bool CastExpressionMatcher::Match(Expression *expr_p, vector<Expression *> &bindings) {
	if (!ExpressionMatcher::Match(expr_p, bindings)) {
		return false;
	}
	if (!matcher) {
		return true;
	}
	auto expr = (BoundCastExpression *)expr_p;
	return matcher->Match(expr->child.get(), bindings);
}

bool InClauseExpressionMatcher::Match(Expression *expr_p, vector<Expression *> &bindings) {
	if (!ExpressionMatcher::Match(expr_p, bindings)) {
		return false;
	}
	auto expr = (BoundOperatorExpression *)expr_p;
	if (expr->type != ExpressionType::COMPARE_IN || expr->type == ExpressionType::COMPARE_NOT_IN) {
		return false;
	}
	return SetMatcher::Match(matchers, expr->children, bindings, policy);
}

bool ConjunctionExpressionMatcher::Match(Expression *expr_p, vector<Expression *> &bindings) {
	if (!ExpressionMatcher::Match(expr_p, bindings)) {
		return false;
	}
	auto expr = (BoundConjunctionExpression *)expr_p;
	if (!SetMatcher::Match(matchers, expr->children, bindings, policy)) {
		return false;
	}
	return true;
}

bool FunctionExpressionMatcher::Match(Expression *expr_p, vector<Expression *> &bindings) {
	if (!ExpressionMatcher::Match(expr_p, bindings)) {
		return false;
	}
	auto expr = (BoundFunctionExpression *)expr_p;
	if (!FunctionMatcher::Match(function, expr->function.name)) {
		return false;
	}
	if (!SetMatcher::Match(matchers, expr->children, bindings, policy)) {
		return false;
	}
	return true;
}

bool FoldableConstantMatcher::Match(Expression *expr, vector<Expression *> &bindings) {
	// we match on ANY expression that is a scalar expression
	if (!expr->IsFoldable()) {
		return false;
	}
	bindings.push_back(expr);
	return true;
}

} // namespace duckdb


























namespace duckdb {

Optimizer::Optimizer(Binder &binder, ClientContext &context) : context(context), binder(binder), rewriter(context) {
	rewriter.rules.push_back(make_unique<ConstantFoldingRule>(rewriter));
	rewriter.rules.push_back(make_unique<DistributivityRule>(rewriter));
	rewriter.rules.push_back(make_unique<ArithmeticSimplificationRule>(rewriter));
	rewriter.rules.push_back(make_unique<CaseSimplificationRule>(rewriter));
	rewriter.rules.push_back(make_unique<ConjunctionSimplificationRule>(rewriter));
	rewriter.rules.push_back(make_unique<DatePartSimplificationRule>(rewriter));
	rewriter.rules.push_back(make_unique<ComparisonSimplificationRule>(rewriter));
	rewriter.rules.push_back(make_unique<InClauseSimplificationRule>(rewriter));
	rewriter.rules.push_back(make_unique<EqualOrNullSimplification>(rewriter));
	rewriter.rules.push_back(make_unique<MoveConstantsRule>(rewriter));
	rewriter.rules.push_back(make_unique<LikeOptimizationRule>(rewriter));
	rewriter.rules.push_back(make_unique<RegexOptimizationRule>(rewriter));
	rewriter.rules.push_back(make_unique<EmptyNeedleRemovalRule>(rewriter));
	rewriter.rules.push_back(make_unique<EnumComparisonRule>(rewriter));

#ifdef DEBUG
	for (auto &rule : rewriter.rules) {
		// root not defined in rule
		D_ASSERT(rule->root);
	}
#endif
}

void Optimizer::RunOptimizer(OptimizerType type, const std::function<void()> &callback) {
	auto &config = DBConfig::GetConfig(context);
	if (config.options.disabled_optimizers.find(type) != config.options.disabled_optimizers.end()) {
		// optimizer is marked as disabled: skip
		return;
	}
	auto &profiler = QueryProfiler::Get(context);
	profiler.StartPhase(OptimizerTypeToString(type));
	callback();
	profiler.EndPhase();
	if (plan) {
		Verify(*plan);
	}
}

void Optimizer::Verify(LogicalOperator &op) {
	ColumnBindingResolver::Verify(op);
}

unique_ptr<LogicalOperator> Optimizer::Optimize(unique_ptr<LogicalOperator> plan_p) {
	Verify(*plan_p);
	this->plan = move(plan_p);
	// first we perform expression rewrites using the ExpressionRewriter
	// this does not change the logical plan structure, but only simplifies the expression trees
	RunOptimizer(OptimizerType::EXPRESSION_REWRITER, [&]() { rewriter.VisitOperator(*plan); });

	// perform filter pullup
	RunOptimizer(OptimizerType::FILTER_PULLUP, [&]() {
		FilterPullup filter_pullup;
		plan = filter_pullup.Rewrite(move(plan));
	});

	// perform filter pushdown
	RunOptimizer(OptimizerType::FILTER_PUSHDOWN, [&]() {
		FilterPushdown filter_pushdown(*this);
		plan = filter_pushdown.Rewrite(move(plan));
	});

	RunOptimizer(OptimizerType::REGEX_RANGE, [&]() {
		RegexRangeFilter regex_opt;
		plan = regex_opt.Rewrite(move(plan));
	});

	RunOptimizer(OptimizerType::IN_CLAUSE, [&]() {
		InClauseRewriter rewriter(context, *this);
		plan = rewriter.Rewrite(move(plan));
	});

	// then we perform the join ordering optimization
	// this also rewrites cross products + filters into joins and performs filter pushdowns
	RunOptimizer(OptimizerType::JOIN_ORDER, [&]() {
		JoinOrderOptimizer optimizer(context);
		plan = optimizer.Optimize(move(plan));
	});

	// removes any redundant DelimGets/DelimJoins
	RunOptimizer(OptimizerType::DELIMINATOR, [&]() {
		Deliminator deliminator;
		plan = deliminator.Optimize(move(plan));
	});

	RunOptimizer(OptimizerType::UNUSED_COLUMNS, [&]() {
		RemoveUnusedColumns unused(binder, context, true);
		unused.VisitOperator(*plan);
	});

	// perform statistics propagation
	RunOptimizer(OptimizerType::STATISTICS_PROPAGATION, [&]() {
		StatisticsPropagator propagator(context);
		propagator.PropagateStatistics(plan);
	});

	// then we extract common subexpressions inside the different operators
	RunOptimizer(OptimizerType::COMMON_SUBEXPRESSIONS, [&]() {
		CommonSubExpressionOptimizer cse_optimizer(binder);
		cse_optimizer.VisitOperator(*plan);
	});

	RunOptimizer(OptimizerType::COMMON_AGGREGATE, [&]() {
		CommonAggregateOptimizer common_aggregate;
		common_aggregate.VisitOperator(*plan);
	});

	RunOptimizer(OptimizerType::COLUMN_LIFETIME, [&]() {
		ColumnLifetimeAnalyzer column_lifetime(true);
		column_lifetime.VisitOperator(*plan);
	});

	// transform ORDER BY + LIMIT to TopN
	RunOptimizer(OptimizerType::TOP_N, [&]() {
		TopN topn;
		plan = topn.Optimize(move(plan));
	});

	// apply simple expression heuristics to get an initial reordering
	RunOptimizer(OptimizerType::REORDER_FILTER, [&]() {
		ExpressionHeuristics expression_heuristics(*this);
		plan = expression_heuristics.Rewrite(move(plan));
	});

	for (auto &optimizer_extension : DBConfig::GetConfig(context).optimizer_extensions) {
		RunOptimizer(OptimizerType::EXTENSION, [&]() {
			optimizer_extension.optimize_function(context, optimizer_extension.optimizer_info.get(), plan);
		});
	}

	Planner::VerifyPlan(context, plan);

	return move(plan);
}

} // namespace duckdb


namespace duckdb {

unique_ptr<LogicalOperator> FilterPullup::PullupBothSide(unique_ptr<LogicalOperator> op) {
	FilterPullup left_pullup(true, can_add_column);
	FilterPullup right_pullup(true, can_add_column);
	op->children[0] = left_pullup.Rewrite(move(op->children[0]));
	op->children[1] = right_pullup.Rewrite(move(op->children[1]));
	D_ASSERT(left_pullup.can_add_column == can_add_column);
	D_ASSERT(right_pullup.can_add_column == can_add_column);

	// merging filter expressions
	for (idx_t i = 0; i < right_pullup.filters_expr_pullup.size(); ++i) {
		left_pullup.filters_expr_pullup.push_back(move(right_pullup.filters_expr_pullup[i]));
	}

	if (!left_pullup.filters_expr_pullup.empty()) {
		return GeneratePullupFilter(move(op), left_pullup.filters_expr_pullup);
	}
	return op;
}

} // namespace duckdb






namespace duckdb {

unique_ptr<LogicalOperator> FilterPullup::PullupFilter(unique_ptr<LogicalOperator> op) {
	D_ASSERT(op->type == LogicalOperatorType::LOGICAL_FILTER);

	auto &filter = (LogicalFilter &)*op;
	if (can_pullup && filter.projection_map.empty()) {
		unique_ptr<LogicalOperator> child = move(op->children[0]);
		child = Rewrite(move(child));
		// moving filter's expressions
		for (idx_t i = 0; i < op->expressions.size(); ++i) {
			filters_expr_pullup.push_back(move(op->expressions[i]));
		}
		return child;
	}
	op->children[0] = Rewrite(move(op->children[0]));
	return op;
}

} // namespace duckdb




namespace duckdb {

unique_ptr<LogicalOperator> FilterPullup::PullupFromLeft(unique_ptr<LogicalOperator> op) {
	D_ASSERT(op->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN ||
	         op->type == LogicalOperatorType::LOGICAL_ANY_JOIN || op->type == LogicalOperatorType::LOGICAL_EXCEPT ||
	         op->type == LogicalOperatorType::LOGICAL_DELIM_JOIN);

	FilterPullup left_pullup(true, can_add_column);
	FilterPullup right_pullup(false, can_add_column);

	op->children[0] = left_pullup.Rewrite(move(op->children[0]));
	op->children[1] = right_pullup.Rewrite(move(op->children[1]));

	// check only for filters from the LHS
	if (!left_pullup.filters_expr_pullup.empty() && right_pullup.filters_expr_pullup.empty()) {
		return GeneratePullupFilter(move(op), left_pullup.filters_expr_pullup);
	}
	return op;
}

} // namespace duckdb







namespace duckdb {

static void RevertFilterPullup(LogicalProjection &proj, vector<unique_ptr<Expression>> &expressions) {
	unique_ptr<LogicalFilter> filter = make_unique<LogicalFilter>();
	for (idx_t i = 0; i < expressions.size(); ++i) {
		filter->expressions.push_back(move(expressions[i]));
	}
	expressions.clear();
	filter->children.push_back(move(proj.children[0]));
	proj.children[0] = move(filter);
}

static void ReplaceExpressionBinding(vector<unique_ptr<Expression>> &proj_expressions, Expression &expr,
                                     idx_t proj_table_idx) {
	if (expr.type == ExpressionType::BOUND_COLUMN_REF) {
		bool found_proj_col = false;
		BoundColumnRefExpression &colref = (BoundColumnRefExpression &)expr;
		// find the corresponding column index in the projection expressions
		for (idx_t proj_idx = 0; proj_idx < proj_expressions.size(); proj_idx++) {
			auto proj_expr = proj_expressions[proj_idx].get();
			if (proj_expr->type == ExpressionType::BOUND_COLUMN_REF) {
				if (colref.Equals(proj_expr)) {
					colref.binding.table_index = proj_table_idx;
					colref.binding.column_index = proj_idx;
					found_proj_col = true;
					break;
				}
			}
		}
		if (!found_proj_col) {
			// Project a new column
			auto new_colref = colref.Copy();
			colref.binding.table_index = proj_table_idx;
			colref.binding.column_index = proj_expressions.size();
			proj_expressions.push_back(move(new_colref));
		}
	}
	ExpressionIterator::EnumerateChildren(
	    expr, [&](Expression &child) { return ReplaceExpressionBinding(proj_expressions, child, proj_table_idx); });
}

void FilterPullup::ProjectSetOperation(LogicalProjection &proj) {
	vector<unique_ptr<Expression>> copy_proj_expressions;
	// copying the project expressions, it's useful whether we should revert the filter pullup
	for (idx_t i = 0; i < proj.expressions.size(); ++i) {
		copy_proj_expressions.push_back(proj.expressions[i]->Copy());
	}

	// Replace filter expression bindings, when need we add new columns into the copied projection expression
	vector<unique_ptr<Expression>> changed_filter_expressions;
	for (idx_t i = 0; i < filters_expr_pullup.size(); ++i) {
		auto copy_filter_expr = filters_expr_pullup[i]->Copy();
		ReplaceExpressionBinding(copy_proj_expressions, (Expression &)*copy_filter_expr, proj.table_index);
		changed_filter_expressions.push_back(move(copy_filter_expr));
	}

	/// Case new columns were added into the projection
	// we must skip filter pullup because adding new columns to these operators will change the result
	if (copy_proj_expressions.size() > proj.expressions.size()) {
		RevertFilterPullup(proj, filters_expr_pullup);
		return;
	}

	// now we must replace the filter bindings
	D_ASSERT(filters_expr_pullup.size() == changed_filter_expressions.size());
	for (idx_t i = 0; i < filters_expr_pullup.size(); ++i) {
		filters_expr_pullup[i] = move(changed_filter_expressions[i]);
	}
}

unique_ptr<LogicalOperator> FilterPullup::PullupProjection(unique_ptr<LogicalOperator> op) {
	D_ASSERT(op->type == LogicalOperatorType::LOGICAL_PROJECTION);
	op->children[0] = Rewrite(move(op->children[0]));
	if (!filters_expr_pullup.empty()) {
		auto &proj = (LogicalProjection &)*op;
		// INTERSECT, EXCEPT, and DISTINCT
		if (!can_add_column) {
			// special treatment for operators that cannot add columns, e.g., INTERSECT, EXCEPT, and DISTINCT
			ProjectSetOperation(proj);
			return op;
		}

		for (idx_t i = 0; i < filters_expr_pullup.size(); ++i) {
			auto &expr = (Expression &)*filters_expr_pullup[i];
			ReplaceExpressionBinding(proj.expressions, expr, proj.table_index);
		}
	}
	return op;
}

} // namespace duckdb





namespace duckdb {

static void ReplaceFilterTableIndex(Expression &expr, LogicalSetOperation &setop) {
	if (expr.type == ExpressionType::BOUND_COLUMN_REF) {
		auto &colref = (BoundColumnRefExpression &)expr;
		D_ASSERT(colref.depth == 0);

		colref.binding.table_index = setop.table_index;
		return;
	}
	ExpressionIterator::EnumerateChildren(expr, [&](Expression &child) { ReplaceFilterTableIndex(child, setop); });
}

unique_ptr<LogicalOperator> FilterPullup::PullupSetOperation(unique_ptr<LogicalOperator> op) {
	D_ASSERT(op->type == LogicalOperatorType::LOGICAL_INTERSECT || op->type == LogicalOperatorType::LOGICAL_EXCEPT);
	can_add_column = false;
	can_pullup = true;
	if (op->type == LogicalOperatorType::LOGICAL_INTERSECT) {
		op = PullupBothSide(move(op));
	} else {
		// EXCEPT only pull ups from LHS
		op = PullupFromLeft(move(op));
	}
	if (op->type == LogicalOperatorType::LOGICAL_FILTER) {
		auto &filter = (LogicalFilter &)*op;
		auto &setop = (LogicalSetOperation &)*filter.children[0];
		for (idx_t i = 0; i < filter.expressions.size(); ++i) {
			ReplaceFilterTableIndex(*filter.expressions[i], setop);
		}
	}
	return op;
}

} // namespace duckdb







namespace duckdb {

using Filter = FilterPushdown::Filter;

static unique_ptr<Expression> ReplaceGroupBindings(LogicalAggregate &proj, unique_ptr<Expression> expr) {
	if (expr->type == ExpressionType::BOUND_COLUMN_REF) {
		auto &colref = (BoundColumnRefExpression &)*expr;
		D_ASSERT(colref.binding.table_index == proj.group_index);
		D_ASSERT(colref.binding.column_index < proj.groups.size());
		D_ASSERT(colref.depth == 0);
		// replace the binding with a copy to the expression at the referenced index
		return proj.groups[colref.binding.column_index]->Copy();
	}
	ExpressionIterator::EnumerateChildren(
	    *expr, [&](unique_ptr<Expression> &child) { child = ReplaceGroupBindings(proj, move(child)); });
	return expr;
}

unique_ptr<LogicalOperator> FilterPushdown::PushdownAggregate(unique_ptr<LogicalOperator> op) {
	D_ASSERT(op->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY);
	auto &aggr = (LogicalAggregate &)*op;

	// pushdown into AGGREGATE and GROUP BY
	// we cannot push expressions that refer to the aggregate
	FilterPushdown child_pushdown(optimizer);
	for (idx_t i = 0; i < filters.size(); i++) {
		auto &f = *filters[i];
		if (f.bindings.find(aggr.aggregate_index) != f.bindings.end()) {
			// filter on aggregate: cannot pushdown
			continue;
		}
		if (f.bindings.find(aggr.groupings_index) != f.bindings.end()) {
			// filter on GROUPINGS function: cannot pushdown
			continue;
		}
		// if there are any empty grouping sets, we cannot push down filters
		bool has_empty_grouping_sets = false;
		for (auto &grp : aggr.grouping_sets) {
			if (grp.empty()) {
				has_empty_grouping_sets = true;
			}
		}
		if (has_empty_grouping_sets) {
			continue;
		}
		// no aggregate! we can push this down
		// rewrite any group bindings within the filter
		f.filter = ReplaceGroupBindings(aggr, move(f.filter));
		// add the filter to the child node
		if (child_pushdown.AddFilter(move(f.filter)) == FilterResult::UNSATISFIABLE) {
			// filter statically evaluates to false, strip tree
			return make_unique<LogicalEmptyResult>(move(op));
		}
		// erase the filter from here
		filters.erase(filters.begin() + i);
		i--;
	}
	child_pushdown.GenerateFilters();

	op->children[0] = child_pushdown.Rewrite(move(op->children[0]));
	return FinishPushdown(move(op));
}

} // namespace duckdb




namespace duckdb {

using Filter = FilterPushdown::Filter;

unique_ptr<LogicalOperator> FilterPushdown::PushdownCrossProduct(unique_ptr<LogicalOperator> op) {
	D_ASSERT(op->type == LogicalOperatorType::LOGICAL_CROSS_PRODUCT);
	FilterPushdown left_pushdown(optimizer), right_pushdown(optimizer);
	vector<unique_ptr<Expression>> join_conditions;
	unordered_set<idx_t> left_bindings, right_bindings;
	if (!filters.empty()) {
		// check to see into which side we should push the filters
		// first get the LHS and RHS bindings
		LogicalJoin::GetTableReferences(*op->children[0], left_bindings);
		LogicalJoin::GetTableReferences(*op->children[1], right_bindings);
		// now check the set of filters
		for (auto &f : filters) {
			auto side = JoinSide::GetJoinSide(f->bindings, left_bindings, right_bindings);
			if (side == JoinSide::LEFT) {
				// bindings match left side: push into left
				left_pushdown.filters.push_back(move(f));
			} else if (side == JoinSide::RIGHT) {
				// bindings match right side: push into right
				right_pushdown.filters.push_back(move(f));
			} else {
				D_ASSERT(side == JoinSide::BOTH || side == JoinSide::NONE);
				// bindings match both: turn into join condition
				join_conditions.push_back(move(f->filter));
			}
		}
	}

	op->children[0] = left_pushdown.Rewrite(move(op->children[0]));
	op->children[1] = right_pushdown.Rewrite(move(op->children[1]));

	if (!join_conditions.empty()) {
		// join conditions found: turn into inner join
		return LogicalComparisonJoin::CreateJoin(JoinType::INNER, move(op->children[0]), move(op->children[1]),
		                                         left_bindings, right_bindings, join_conditions);
	} else {
		// no join conditions found: keep as cross product
		return op;
	}
}

} // namespace duckdb




namespace duckdb {

using Filter = FilterPushdown::Filter;

unique_ptr<LogicalOperator> FilterPushdown::PushdownFilter(unique_ptr<LogicalOperator> op) {
	D_ASSERT(op->type == LogicalOperatorType::LOGICAL_FILTER);
	auto &filter = (LogicalFilter &)*op;
	if (!filter.projection_map.empty()) {
		return FinishPushdown(move(op));
	}
	// filter: gather the filters and remove the filter from the set of operations
	for (auto &expression : filter.expressions) {
		if (AddFilter(move(expression)) == FilterResult::UNSATISFIABLE) {
			// filter statically evaluates to false, strip tree
			return make_unique<LogicalEmptyResult>(move(op));
		}
	}
	GenerateFilters();
	return Rewrite(move(filter.children[0]));
}

} // namespace duckdb








namespace duckdb {

unique_ptr<LogicalOperator> FilterPushdown::PushdownGet(unique_ptr<LogicalOperator> op) {
	D_ASSERT(op->type == LogicalOperatorType::LOGICAL_GET);
	auto &get = (LogicalGet &)*op;

	if (get.function.pushdown_complex_filter || get.function.filter_pushdown) {
		// this scan supports some form of filter push-down
		// check if there are any parameters
		// if there are, invalidate them to force a re-bind on execution
		for (auto &filter : filters) {
			if (filter->filter->HasParameter()) {
				// there is a parameter in the filters! invalidate it
				BoundParameterExpression::InvalidateRecursive(*filter->filter);
			}
		}
	}
	if (get.function.pushdown_complex_filter) {
		// for the remaining filters, check if we can push any of them into the scan as well
		vector<unique_ptr<Expression>> expressions;
		for (auto &filter : filters) {
			expressions.push_back(move(filter->filter));
		}
		filters.clear();

		get.function.pushdown_complex_filter(optimizer.context, get, get.bind_data.get(), expressions);

		if (expressions.empty()) {
			return op;
		}
		// re-generate the filters
		for (auto &expr : expressions) {
			auto f = make_unique<Filter>();
			f->filter = move(expr);
			f->ExtractBindings();
			filters.push_back(move(f));
		}
	}

	if (!get.table_filters.filters.empty() || !get.function.filter_pushdown) {
		// the table function does not support filter pushdown: push a LogicalFilter on top
		return FinishPushdown(move(op));
	}
	PushFilters();

	//! We generate the table filters that will be executed during the table scan
	//! Right now this only executes simple AND filters
	get.table_filters = combiner.GenerateTableScanFilters(get.column_ids);

	// //! For more complex filters if all filters to a column are constants we generate a min max boundary used to
	// check
	// //! the zonemaps.
	// auto zonemap_checks = combiner.GenerateZonemapChecks(get.column_ids, get.table_filters);

	// for (auto &f : get.table_filters) {
	// 	f.column_index = get.column_ids[f.column_index];
	// }

	// //! Use zonemap checks as table filters for pre-processing
	// for (auto &zonemap_check : zonemap_checks) {
	// 	if (zonemap_check.column_index != COLUMN_IDENTIFIER_ROW_ID) {
	// 		get.table_filters.push_back(zonemap_check);
	// 	}
	// }

	GenerateFilters();

	//! Now we try to pushdown the remaining filters to perform zonemap checking
	return FinishPushdown(move(op));
}

} // namespace duckdb






namespace duckdb {

using Filter = FilterPushdown::Filter;

unique_ptr<LogicalOperator> FilterPushdown::PushdownInnerJoin(unique_ptr<LogicalOperator> op,
                                                              unordered_set<idx_t> &left_bindings,
                                                              unordered_set<idx_t> &right_bindings) {
	auto &join = (LogicalJoin &)*op;
	D_ASSERT(join.join_type == JoinType::INNER);
	D_ASSERT(op->type != LogicalOperatorType::LOGICAL_DELIM_JOIN);
	// inner join: gather all the conditions of the inner join and add to the filter list
	if (op->type == LogicalOperatorType::LOGICAL_ANY_JOIN) {
		auto &any_join = (LogicalAnyJoin &)join;
		// any join: only one filter to add
		if (AddFilter(move(any_join.condition)) == FilterResult::UNSATISFIABLE) {
			// filter statically evaluates to false, strip tree
			return make_unique<LogicalEmptyResult>(move(op));
		}
	} else {
		// comparison join
		D_ASSERT(op->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN);
		auto &comp_join = (LogicalComparisonJoin &)join;
		// turn the conditions into filters
		for (auto &i : comp_join.conditions) {
			auto condition = JoinCondition::CreateExpression(move(i));
			if (AddFilter(move(condition)) == FilterResult::UNSATISFIABLE) {
				// filter statically evaluates to false, strip tree
				return make_unique<LogicalEmptyResult>(move(op));
			}
		}
	}
	GenerateFilters();

	// turn the inner join into a cross product
	auto cross_product = make_unique<LogicalCrossProduct>(move(op->children[0]), move(op->children[1]));
	// then push down cross product
	return PushdownCrossProduct(move(cross_product));
}

} // namespace duckdb










namespace duckdb {

using Filter = FilterPushdown::Filter;

static unique_ptr<Expression> ReplaceColRefWithNull(unique_ptr<Expression> expr, unordered_set<idx_t> &right_bindings) {
	if (expr->type == ExpressionType::BOUND_COLUMN_REF) {
		auto &bound_colref = (BoundColumnRefExpression &)*expr;
		if (right_bindings.find(bound_colref.binding.table_index) != right_bindings.end()) {
			// bound colref belongs to RHS
			// replace it with a constant NULL
			return make_unique<BoundConstantExpression>(Value(expr->return_type));
		}
		return expr;
	}
	ExpressionIterator::EnumerateChildren(
	    *expr, [&](unique_ptr<Expression> &child) { child = ReplaceColRefWithNull(move(child), right_bindings); });
	return expr;
}

static bool FilterRemovesNull(ClientContext &context, ExpressionRewriter &rewriter, Expression *expr,
                              unordered_set<idx_t> &right_bindings) {
	// make a copy of the expression
	auto copy = expr->Copy();
	// replace all BoundColumnRef expressions frmo the RHS with NULL constants in the copied expression
	copy = ReplaceColRefWithNull(move(copy), right_bindings);

	// attempt to flatten the expression by running the expression rewriter on it
	auto filter = make_unique<LogicalFilter>();
	filter->expressions.push_back(move(copy));
	rewriter.VisitOperator(*filter);

	// check if all expressions are foldable
	for (idx_t i = 0; i < filter->expressions.size(); i++) {
		if (!filter->expressions[i]->IsFoldable()) {
			return false;
		}
		// we flattened the result into a scalar, check if it is FALSE or NULL
		auto val =
		    ExpressionExecutor::EvaluateScalar(context, *filter->expressions[i]).DefaultCastAs(LogicalType::BOOLEAN);
		// if the result of the expression with all expressions replaced with NULL is "NULL" or "false"
		// then any extra entries generated by the LEFT OUTER JOIN will be filtered out!
		// hence the LEFT OUTER JOIN is equivalent to an inner join
		if (val.IsNull() || !BooleanValue::Get(val)) {
			return true;
		}
	}
	return false;
}

unique_ptr<LogicalOperator> FilterPushdown::PushdownLeftJoin(unique_ptr<LogicalOperator> op,
                                                             unordered_set<idx_t> &left_bindings,
                                                             unordered_set<idx_t> &right_bindings) {
	auto &join = (LogicalJoin &)*op;
	D_ASSERT(join.join_type == JoinType::LEFT);
	D_ASSERT(op->type != LogicalOperatorType::LOGICAL_DELIM_JOIN);
	FilterPushdown left_pushdown(optimizer), right_pushdown(optimizer);
	// for a comparison join we create a FilterCombiner that checks if we can push conditions on LHS join conditions
	// into the RHS of the join
	FilterCombiner filter_combiner(optimizer);
	if (op->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
		// add all comparison conditions
		auto &comparison_join = (LogicalComparisonJoin &)*op;
		for (auto &cond : comparison_join.conditions) {
			filter_combiner.AddFilter(
			    make_unique<BoundComparisonExpression>(cond.comparison, cond.left->Copy(), cond.right->Copy()));
		}
	}
	// now check the set of filters
	for (idx_t i = 0; i < filters.size(); i++) {
		auto side = JoinSide::GetJoinSide(filters[i]->bindings, left_bindings, right_bindings);
		if (side == JoinSide::LEFT) {
			// bindings match left side
			// we can push the filter into the left side
			if (op->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
				// we MIGHT be able to push it down the RHS as well, but only if it is a comparison that matches the
				// join predicates we use the FilterCombiner to figure this out add the expression to the FilterCombiner
				filter_combiner.AddFilter(filters[i]->filter->Copy());
			}
			left_pushdown.filters.push_back(move(filters[i]));
			// erase the filter from the list of filters
			filters.erase(filters.begin() + i);
			i--;
		} else {
			// bindings match right side or both sides: we cannot directly push it into the right
			// however, if the filter removes rows with null values from the RHS we can turn the left outer join
			// in an inner join, and then push down as we would push down an inner join
			if (FilterRemovesNull(optimizer.context, optimizer.rewriter, filters[i]->filter.get(), right_bindings)) {
				// the filter removes NULL values, turn it into an inner join
				join.join_type = JoinType::INNER;
				// now we can do more pushdown
				// move all filters we added to the left_pushdown back into the filter list
				for (auto &left_filter : left_pushdown.filters) {
					filters.push_back(move(left_filter));
				}
				// now push down the inner join
				return PushdownInnerJoin(move(op), left_bindings, right_bindings);
			}
		}
	}
	// finally we check the FilterCombiner to see if there are any predicates we can push into the RHS
	// we only added (1) predicates that have JoinSide::BOTH from the conditions, and
	// (2) predicates that have JoinSide::LEFT from the filters
	// we check now if this combination generated any new filters that are only on JoinSide::RIGHT
	// this happens if, e.g. a join condition is (i=a) and there is a filter (i=500), we can then push the filter
	// (a=500) into the RHS
	filter_combiner.GenerateFilters([&](unique_ptr<Expression> filter) {
		if (JoinSide::GetJoinSide(*filter, left_bindings, right_bindings) == JoinSide::RIGHT) {
			right_pushdown.AddFilter(move(filter));
		}
	});
	right_pushdown.GenerateFilters();
	op->children[0] = left_pushdown.Rewrite(move(op->children[0]));
	op->children[1] = right_pushdown.Rewrite(move(op->children[1]));
	return FinishPushdown(move(op));
}

} // namespace duckdb






namespace duckdb {

unique_ptr<LogicalOperator> FilterPushdown::PushdownLimit(unique_ptr<LogicalOperator> op) {
	auto &limit = (LogicalLimit &)*op;

	if (!limit.limit && limit.limit_val == 0) {
		return make_unique<LogicalEmptyResult>(move(op));
	}

	return FinishPushdown(move(op));
}

} // namespace duckdb




namespace duckdb {

using Filter = FilterPushdown::Filter;

unique_ptr<LogicalOperator> FilterPushdown::PushdownMarkJoin(unique_ptr<LogicalOperator> op,
                                                             unordered_set<idx_t> &left_bindings,
                                                             unordered_set<idx_t> &right_bindings) {
	auto &join = (LogicalJoin &)*op;
	auto &comp_join = (LogicalComparisonJoin &)*op;
	D_ASSERT(join.join_type == JoinType::MARK);
	D_ASSERT(op->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN ||
	         op->type == LogicalOperatorType::LOGICAL_DELIM_JOIN);

	right_bindings.insert(comp_join.mark_index);
	FilterPushdown left_pushdown(optimizer), right_pushdown(optimizer);
#ifdef DEBUG
	bool simplified_mark_join = false;
#endif
	// now check the set of filters
	for (idx_t i = 0; i < filters.size(); i++) {
		auto side = JoinSide::GetJoinSide(filters[i]->bindings, left_bindings, right_bindings);
		if (side == JoinSide::LEFT) {
			// bindings match left side: push into left
			left_pushdown.filters.push_back(move(filters[i]));
			// erase the filter from the list of filters
			filters.erase(filters.begin() + i);
			i--;
		} else if (side == JoinSide::RIGHT) {
#ifdef DEBUG
			D_ASSERT(!simplified_mark_join);
#endif
			// this filter references the marker
			// we can turn this into a SEMI join if the filter is on only the marker
			if (filters[i]->filter->type == ExpressionType::BOUND_COLUMN_REF) {
				// filter just references the marker: turn into semi join
#ifdef DEBUG
				simplified_mark_join = true;
#endif
				join.join_type = JoinType::SEMI;
				filters.erase(filters.begin() + i);
				i--;
				continue;
			}
			// if the filter is on NOT(marker) AND the join conditions are all set to "null_values_are_equal" we can
			// turn this into an ANTI join if all join conditions have null_values_are_equal=true, then the result of
			// the MARK join is always TRUE or FALSE, and never NULL this happens in the case of a correlated EXISTS
			// clause
			if (filters[i]->filter->type == ExpressionType::OPERATOR_NOT) {
				auto &op_expr = (BoundOperatorExpression &)*filters[i]->filter;
				if (op_expr.children[0]->type == ExpressionType::BOUND_COLUMN_REF) {
					// the filter is NOT(marker), check the join conditions
					bool all_null_values_are_equal = true;
					for (auto &cond : comp_join.conditions) {
						if (cond.comparison != ExpressionType::COMPARE_DISTINCT_FROM &&
						    cond.comparison != ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
							all_null_values_are_equal = false;
							break;
						}
					}
					if (all_null_values_are_equal) {
#ifdef DEBUG
						simplified_mark_join = true;
#endif
						// all null values are equal, convert to ANTI join
						join.join_type = JoinType::ANTI;
						filters.erase(filters.begin() + i);
						i--;
						continue;
					}
				}
			}
		}
	}
	op->children[0] = left_pushdown.Rewrite(move(op->children[0]));
	op->children[1] = right_pushdown.Rewrite(move(op->children[1]));
	return FinishPushdown(move(op));
}

} // namespace duckdb






namespace duckdb {

static unique_ptr<Expression> ReplaceProjectionBindings(LogicalProjection &proj, unique_ptr<Expression> expr) {
	if (expr->type == ExpressionType::BOUND_COLUMN_REF) {
		auto &colref = (BoundColumnRefExpression &)*expr;
		D_ASSERT(colref.binding.table_index == proj.table_index);
		D_ASSERT(colref.binding.column_index < proj.expressions.size());
		D_ASSERT(colref.depth == 0);
		// replace the binding with a copy to the expression at the referenced index
		return proj.expressions[colref.binding.column_index]->Copy();
	}
	ExpressionIterator::EnumerateChildren(
	    *expr, [&](unique_ptr<Expression> &child) { child = ReplaceProjectionBindings(proj, move(child)); });
	return expr;
}

unique_ptr<LogicalOperator> FilterPushdown::PushdownProjection(unique_ptr<LogicalOperator> op) {
	D_ASSERT(op->type == LogicalOperatorType::LOGICAL_PROJECTION);
	auto &proj = (LogicalProjection &)*op;
	// push filter through logical projection
	// all the BoundColumnRefExpressions in the filter should refer to the LogicalProjection
	// we can rewrite them by replacing those references with the expression of the LogicalProjection node
	FilterPushdown child_pushdown(optimizer);
	for (auto &filter : filters) {
		auto &f = *filter;
		D_ASSERT(f.bindings.size() <= 1);
		// rewrite the bindings within this subquery
		f.filter = ReplaceProjectionBindings(proj, move(f.filter));
		// add the filter to the child pushdown
		if (child_pushdown.AddFilter(move(f.filter)) == FilterResult::UNSATISFIABLE) {
			// filter statically evaluates to false, strip tree
			return make_unique<LogicalEmptyResult>(move(op));
		}
	}
	child_pushdown.GenerateFilters();
	// now push into children
	op->children[0] = child_pushdown.Rewrite(move(op->children[0]));
	if (op->children[0]->type == LogicalOperatorType::LOGICAL_EMPTY_RESULT) {
		// child returns an empty result: generate an empty result here too
		return make_unique<LogicalEmptyResult>(move(op));
	}
	return op;
}

} // namespace duckdb









namespace duckdb {

using Filter = FilterPushdown::Filter;

static void ReplaceSetOpBindings(vector<ColumnBinding> &bindings, Filter &filter, Expression &expr,
                                 LogicalSetOperation &setop) {
	if (expr.type == ExpressionType::BOUND_COLUMN_REF) {
		auto &colref = (BoundColumnRefExpression &)expr;
		D_ASSERT(colref.binding.table_index == setop.table_index);
		D_ASSERT(colref.depth == 0);

		// rewrite the binding by looking into the bound_tables list of the subquery
		colref.binding = bindings[colref.binding.column_index];
		filter.bindings.insert(colref.binding.table_index);
		return;
	}
	ExpressionIterator::EnumerateChildren(
	    expr, [&](Expression &child) { ReplaceSetOpBindings(bindings, filter, child, setop); });
}

unique_ptr<LogicalOperator> FilterPushdown::PushdownSetOperation(unique_ptr<LogicalOperator> op) {
	D_ASSERT(op->type == LogicalOperatorType::LOGICAL_UNION || op->type == LogicalOperatorType::LOGICAL_EXCEPT ||
	         op->type == LogicalOperatorType::LOGICAL_INTERSECT);
	auto &setop = (LogicalSetOperation &)*op;

	D_ASSERT(op->children.size() == 2);
	auto left_bindings = op->children[0]->GetColumnBindings();
	auto right_bindings = op->children[1]->GetColumnBindings();
	if (left_bindings.size() != right_bindings.size()) {
		throw InternalException("Filter pushdown - set operation LHS and RHS have incompatible counts");
	}

	// pushdown into set operation, we can duplicate the condition and pushdown the expressions into both sides
	FilterPushdown left_pushdown(optimizer), right_pushdown(optimizer);
	for (idx_t i = 0; i < filters.size(); i++) {
		// first create a copy of the filter
		auto right_filter = make_unique<Filter>();
		right_filter->filter = filters[i]->filter->Copy();

		// in the original filter, rewrite references to the result of the union into references to the left_index
		ReplaceSetOpBindings(left_bindings, *filters[i], *filters[i]->filter, setop);
		// in the copied filter, rewrite references to the result of the union into references to the right_index
		ReplaceSetOpBindings(right_bindings, *right_filter, *right_filter->filter, setop);

		// extract bindings again
		filters[i]->ExtractBindings();
		right_filter->ExtractBindings();

		// move the filters into the child pushdown nodes
		left_pushdown.filters.push_back(move(filters[i]));
		right_pushdown.filters.push_back(move(right_filter));
	}

	op->children[0] = left_pushdown.Rewrite(move(op->children[0]));
	op->children[1] = right_pushdown.Rewrite(move(op->children[1]));

	bool left_empty = op->children[0]->type == LogicalOperatorType::LOGICAL_EMPTY_RESULT;
	bool right_empty = op->children[1]->type == LogicalOperatorType::LOGICAL_EMPTY_RESULT;
	if (left_empty && right_empty) {
		// both empty: return empty result
		return make_unique<LogicalEmptyResult>(move(op));
	}
	if (left_empty) {
		// left child is empty result
		switch (op->type) {
		case LogicalOperatorType::LOGICAL_UNION:
			if (op->children[1]->type == LogicalOperatorType::LOGICAL_PROJECTION) {
				// union with empty left side: return right child
				auto &projection = (LogicalProjection &)*op->children[1];
				projection.table_index = setop.table_index;
				return move(op->children[1]);
			}
			break;
		case LogicalOperatorType::LOGICAL_EXCEPT:
			// except: if left child is empty, return empty result
		case LogicalOperatorType::LOGICAL_INTERSECT:
			// intersect: if any child is empty, return empty result itself
			return make_unique<LogicalEmptyResult>(move(op));
		default:
			throw InternalException("Unsupported set operation");
		}
	} else if (right_empty) {
		// right child is empty result
		switch (op->type) {
		case LogicalOperatorType::LOGICAL_UNION:
		case LogicalOperatorType::LOGICAL_EXCEPT:
			if (op->children[0]->type == LogicalOperatorType::LOGICAL_PROJECTION) {
				// union or except with empty right child: return left child
				auto &projection = (LogicalProjection &)*op->children[0];
				projection.table_index = setop.table_index;
				return move(op->children[0]);
			}
			break;
		case LogicalOperatorType::LOGICAL_INTERSECT:
			// intersect: if any child is empty, return empty result itself
			return make_unique<LogicalEmptyResult>(move(op));
		default:
			throw InternalException("Unsupported set operation");
		}
	}
	return op;
}

} // namespace duckdb



namespace duckdb {

using Filter = FilterPushdown::Filter;

unique_ptr<LogicalOperator> FilterPushdown::PushdownSingleJoin(unique_ptr<LogicalOperator> op,
                                                               unordered_set<idx_t> &left_bindings,
                                                               unordered_set<idx_t> &right_bindings) {
	D_ASSERT(((LogicalJoin &)*op).join_type == JoinType::SINGLE);
	FilterPushdown left_pushdown(optimizer), right_pushdown(optimizer);
	// now check the set of filters
	for (idx_t i = 0; i < filters.size(); i++) {
		auto side = JoinSide::GetJoinSide(filters[i]->bindings, left_bindings, right_bindings);
		if (side == JoinSide::LEFT) {
			// bindings match left side: push into left
			left_pushdown.filters.push_back(move(filters[i]));
			// erase the filter from the list of filters
			filters.erase(filters.begin() + i);
			i--;
		}
	}
	op->children[0] = left_pushdown.Rewrite(move(op->children[0]));
	op->children[1] = right_pushdown.Rewrite(move(op->children[1]));
	return FinishPushdown(move(op));
}

} // namespace duckdb















namespace duckdb {

unique_ptr<LogicalOperator> RegexRangeFilter::Rewrite(unique_ptr<LogicalOperator> op) {

	for (idx_t child_idx = 0; child_idx < op->children.size(); child_idx++) {
		op->children[child_idx] = Rewrite(move(op->children[child_idx]));
	}

	if (op->type != LogicalOperatorType::LOGICAL_FILTER) {
		return op;
	}

	auto new_filter = make_unique<LogicalFilter>();

	for (auto &expr : op->expressions) {
		if (expr->type == ExpressionType::BOUND_FUNCTION) {
			auto &func = (BoundFunctionExpression &)*expr.get();
			if (func.function.name != "regexp_full_match" || func.children.size() != 2) {
				continue;
			}
			auto &info = (RegexpMatchesBindData &)*func.bind_info;
			if (!info.range_success) {
				continue;
			}
			auto filter_left = make_unique<BoundComparisonExpression>(
			    ExpressionType::COMPARE_GREATERTHANOREQUALTO, func.children[0]->Copy(),
			    make_unique<BoundConstantExpression>(
			        Value::BLOB((const_data_ptr_t)info.range_min.c_str(), info.range_min.size())));
			auto filter_right = make_unique<BoundComparisonExpression>(
			    ExpressionType::COMPARE_LESSTHANOREQUALTO, func.children[0]->Copy(),
			    make_unique<BoundConstantExpression>(
			        Value::BLOB((const_data_ptr_t)info.range_max.c_str(), info.range_max.size())));
			auto filter_expr = make_unique<BoundConjunctionExpression>(ExpressionType::CONJUNCTION_AND,
			                                                           move(filter_left), move(filter_right));

			new_filter->expressions.push_back(move(filter_expr));
		}
	}

	if (!new_filter->expressions.empty()) {
		new_filter->children = move(op->children);
		op->children.clear();
		op->children.push_back(move(new_filter));
	}

	return op;
}

} // namespace duckdb




















namespace duckdb {

void RemoveUnusedColumns::ReplaceBinding(ColumnBinding current_binding, ColumnBinding new_binding) {
	auto colrefs = column_references.find(current_binding);
	if (colrefs != column_references.end()) {
		for (auto &colref : colrefs->second) {
			D_ASSERT(colref->binding == current_binding);
			colref->binding = new_binding;
		}
	}
}

template <class T>
void RemoveUnusedColumns::ClearUnusedExpressions(vector<T> &list, idx_t table_idx, bool replace) {
	idx_t offset = 0;
	for (idx_t col_idx = 0; col_idx < list.size(); col_idx++) {
		auto current_binding = ColumnBinding(table_idx, col_idx + offset);
		auto entry = column_references.find(current_binding);
		if (entry == column_references.end()) {
			// this entry is not referred to, erase it from the set of expressions
			list.erase(list.begin() + col_idx);
			offset++;
			col_idx--;
		} else if (offset > 0 && replace) {
			// column is used but the ColumnBinding has changed because of removed columns
			ReplaceBinding(current_binding, ColumnBinding(table_idx, col_idx));
		}
	}
}

void RemoveUnusedColumns::VisitOperator(LogicalOperator &op) {
	switch (op.type) {
	case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY: {
		// aggregate
		if (!everything_referenced) {
			// FIXME: groups that are not referenced need to stay -> but they don't need to be scanned and output!
			auto &aggr = (LogicalAggregate &)op;
			ClearUnusedExpressions(aggr.expressions, aggr.aggregate_index);
			if (aggr.expressions.empty() && aggr.groups.empty()) {
				// removed all expressions from the aggregate: push a COUNT(*)
				auto count_star_fun = CountStarFun::GetFunction();
				FunctionBinder function_binder(context);
				aggr.expressions.push_back(
				    function_binder.BindAggregateFunction(count_star_fun, {}, nullptr, AggregateType::NON_DISTINCT));
			}
		}

		// then recurse into the children of the aggregate
		RemoveUnusedColumns remove(binder, context);
		remove.VisitOperatorExpressions(op);
		remove.VisitOperator(*op.children[0]);
		return;
	}
	case LogicalOperatorType::LOGICAL_DELIM_JOIN:
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN: {
		if (!everything_referenced) {
			auto &comp_join = (LogicalComparisonJoin &)op;

			if (comp_join.join_type != JoinType::INNER) {
				break;
			}
			// for inner joins with equality predicates in the form of (X=Y)
			// we can replace any references to the RHS (Y) to references to the LHS (X)
			// this reduces the amount of columns we need to extract from the join hash table
			for (auto &cond : comp_join.conditions) {
				if (cond.comparison == ExpressionType::COMPARE_EQUAL) {
					if (cond.left->expression_class == ExpressionClass::BOUND_COLUMN_REF &&
					    cond.right->expression_class == ExpressionClass::BOUND_COLUMN_REF) {
						// comparison join between two bound column refs
						// we can replace any reference to the RHS (build-side) with a reference to the LHS (probe-side)
						auto &lhs_col = (BoundColumnRefExpression &)*cond.left;
						auto &rhs_col = (BoundColumnRefExpression &)*cond.right;
						// if there are any columns that refer to the RHS,
						auto colrefs = column_references.find(rhs_col.binding);
						if (colrefs != column_references.end()) {
							for (auto &entry : colrefs->second) {
								entry->binding = lhs_col.binding;
								column_references[lhs_col.binding].push_back(entry);
							}
							column_references.erase(rhs_col.binding);
						}
					}
				}
			}
		}
		break;
	}
	case LogicalOperatorType::LOGICAL_ANY_JOIN:
		break;
	case LogicalOperatorType::LOGICAL_UNION:
		if (!everything_referenced) {
			// for UNION we can remove unreferenced columns as long as everything_referenced is false (i.e. we
			// encounter a UNION node that is not preceded by a DISTINCT)
			// this happens when UNION ALL is used
			auto &setop = (LogicalSetOperation &)op;
			vector<idx_t> entries;
			for (idx_t i = 0; i < setop.column_count; i++) {
				entries.push_back(i);
			}
			ClearUnusedExpressions(entries, setop.table_index);
			if (entries.size() < setop.column_count) {
				if (entries.empty()) {
					// no columns referenced: this happens in the case of a COUNT(*)
					// extract the first column
					entries.push_back(0);
				}
				// columns were cleared
				setop.column_count = entries.size();

				for (idx_t child_idx = 0; child_idx < op.children.size(); child_idx++) {
					RemoveUnusedColumns remove(binder, context, true);
					auto &child = op.children[child_idx];

					// we push a projection under this child that references the required columns of the union
					child->ResolveOperatorTypes();
					auto bindings = child->GetColumnBindings();
					vector<unique_ptr<Expression>> expressions;
					expressions.reserve(entries.size());
					for (auto &column_idx : entries) {
						expressions.push_back(
						    make_unique<BoundColumnRefExpression>(child->types[column_idx], bindings[column_idx]));
					}
					auto new_projection =
					    make_unique<LogicalProjection>(binder.GenerateTableIndex(), move(expressions));
					new_projection->children.push_back(move(child));
					op.children[child_idx] = move(new_projection);

					remove.VisitOperator(*op.children[child_idx]);
				}
				return;
			}
		}
		for (auto &child : op.children) {
			RemoveUnusedColumns remove(binder, context, true);
			remove.VisitOperator(*child);
		}
		return;
	case LogicalOperatorType::LOGICAL_EXCEPT:
	case LogicalOperatorType::LOGICAL_INTERSECT:
		// for INTERSECT/EXCEPT operations we can't remove anything, just recursively visit the children
		for (auto &child : op.children) {
			RemoveUnusedColumns remove(binder, context, true);
			remove.VisitOperator(*child);
		}
		return;
	case LogicalOperatorType::LOGICAL_ORDER_BY:
		if (!everything_referenced) {
			auto &order = (LogicalOrder &)op;
			D_ASSERT(order.projections.empty()); // should not yet be set
			const auto all_bindings = order.GetColumnBindings();

			for (idx_t col_idx = 0; col_idx < all_bindings.size(); col_idx++) {
				if (column_references.find(all_bindings[col_idx]) != column_references.end()) {
					order.projections.push_back(col_idx);
				}
			}
		}
		for (auto &child : op.children) {
			RemoveUnusedColumns remove(binder, context, true);
			remove.VisitOperator(*child);
		}
		return;
	case LogicalOperatorType::LOGICAL_PROJECTION: {
		if (!everything_referenced) {
			auto &proj = (LogicalProjection &)op;
			ClearUnusedExpressions(proj.expressions, proj.table_index);

			if (proj.expressions.empty()) {
				// nothing references the projected expressions
				// this happens in the case of e.g. EXISTS(SELECT * FROM ...)
				// in this case we only need to project a single constant
				proj.expressions.push_back(make_unique<BoundConstantExpression>(Value::INTEGER(42)));
			}
		}
		// then recurse into the children of this projection
		RemoveUnusedColumns remove(binder, context);
		remove.VisitOperatorExpressions(op);
		remove.VisitOperator(*op.children[0]);
		return;
	}
	case LogicalOperatorType::LOGICAL_INSERT:
	case LogicalOperatorType::LOGICAL_UPDATE:
	case LogicalOperatorType::LOGICAL_DELETE: {
		//! When RETURNING is used, a PROJECTION is the top level operator for INSERTS, UPDATES, and DELETES
		//! We still need to project all values from these operators so the projection
		//! on top of them can select from only the table values being inserted.
		//! TODO: Push down the projections from the returning statement
		//! TODO: Be careful because you might be adding expressions when a user returns *
		RemoveUnusedColumns remove(binder, context, true);
		remove.VisitOperatorExpressions(op);
		remove.VisitOperator(*op.children[0]);
		return;
	}
	case LogicalOperatorType::LOGICAL_GET:
		LogicalOperatorVisitor::VisitOperatorExpressions(op);
		if (!everything_referenced) {
			auto &get = (LogicalGet &)op;

			// Create "selection vector" of all column ids
			vector<idx_t> proj_sel;
			for (idx_t col_idx = 0; col_idx < get.column_ids.size(); col_idx++) {
				proj_sel.push_back(col_idx);
			}
			// Create a copy that we can use to match ids later
			auto col_sel = proj_sel;
			// Clear unused ids, exclude filter columns that are projected out immediately
			ClearUnusedExpressions(proj_sel, get.table_index, false);

			// for every table filter, push a column binding into the column references map to prevent the column from
			// being projected out
			for (auto &filter : get.table_filters.filters) {
				idx_t index = DConstants::INVALID_INDEX;
				for (idx_t i = 0; i < get.column_ids.size(); i++) {
					if (get.column_ids[i] == filter.first) {
						index = i;
						break;
					}
				}
				if (index == DConstants::INVALID_INDEX) {
					throw InternalException("Could not find column index for table filter");
				}
				ColumnBinding filter_binding(get.table_index, index);
				if (column_references.find(filter_binding) == column_references.end()) {
					column_references.insert(make_pair(filter_binding, vector<BoundColumnRefExpression *>()));
				}
			}

			// Clear unused ids, include filter columns that are projected out immediately
			ClearUnusedExpressions(col_sel, get.table_index);

			// Now set the column ids in the LogicalGet using the "selection vector"
			vector<column_t> column_ids;
			column_ids.reserve(col_sel.size());
			for (auto col_sel_idx : col_sel) {
				column_ids.push_back(get.column_ids[col_sel_idx]);
			}
			get.column_ids = move(column_ids);

			if (get.function.filter_prune) {
				// Now set the projection cols by matching the "selection vector" that excludes filter columns
				// with the "selection vector" that includes filter columns
				idx_t col_idx = 0;
				for (auto proj_sel_idx : proj_sel) {
					for (; col_idx < col_sel.size(); col_idx++) {
						if (proj_sel_idx == col_sel[col_idx]) {
							get.projection_ids.push_back(col_idx);
							break;
						}
					}
				}
			}

			if (get.column_ids.empty()) {
				// this generally means we are only interested in whether or not anything exists in the table (e.g.
				// EXISTS(SELECT * FROM tbl)) in this case, we just scan the row identifier column as it means we do not
				// need to read any of the columns
				get.column_ids.push_back(COLUMN_IDENTIFIER_ROW_ID);
			}
		}
		return;
	case LogicalOperatorType::LOGICAL_DISTINCT: {
		// distinct, all projected columns are used for the DISTINCT computation
		// mark all columns as used and continue to the children
		// FIXME: DISTINCT with expression list does not implicitly reference everything
		everything_referenced = true;
		break;
	}
	case LogicalOperatorType::LOGICAL_RECURSIVE_CTE: {
		everything_referenced = true;
		break;
	}
	case LogicalOperatorType::LOGICAL_CTE_REF: {
		everything_referenced = true;
		break;
	}
	default:
		break;
	}
	LogicalOperatorVisitor::VisitOperatorExpressions(op);
	LogicalOperatorVisitor::VisitOperatorChildren(op);
}

unique_ptr<Expression> RemoveUnusedColumns::VisitReplace(BoundColumnRefExpression &expr,
                                                         unique_ptr<Expression> *expr_ptr) {
	// add a column reference
	column_references[expr.binding].push_back(&expr);
	return nullptr;
}

unique_ptr<Expression> RemoveUnusedColumns::VisitReplace(BoundReferenceExpression &expr,
                                                         unique_ptr<Expression> *expr_ptr) {
	// BoundReferenceExpression should not be used here yet, they only belong in the physical plan
	throw InternalException("BoundReferenceExpression should not be used here yet!");
}

} // namespace duckdb







namespace duckdb {

ArithmeticSimplificationRule::ArithmeticSimplificationRule(ExpressionRewriter &rewriter) : Rule(rewriter) {
	// match on an OperatorExpression that has a ConstantExpression as child
	auto op = make_unique<FunctionExpressionMatcher>();
	op->matchers.push_back(make_unique<ConstantExpressionMatcher>());
	op->matchers.push_back(make_unique<ExpressionMatcher>());
	op->policy = SetMatcher::Policy::SOME;
	// we only match on simple arithmetic expressions (+, -, *, /)
	op->function = make_unique<ManyFunctionMatcher>(unordered_set<string> {"+", "-", "*", "/"});
	// and only with numeric results
	op->type = make_unique<IntegerTypeMatcher>();
	op->matchers[0]->type = make_unique<IntegerTypeMatcher>();
	op->matchers[1]->type = make_unique<IntegerTypeMatcher>();
	root = move(op);
}

unique_ptr<Expression> ArithmeticSimplificationRule::Apply(LogicalOperator &op, vector<Expression *> &bindings,
                                                           bool &changes_made, bool is_root) {
	auto root = (BoundFunctionExpression *)bindings[0];
	auto constant = (BoundConstantExpression *)bindings[1];
	int constant_child = root->children[0].get() == constant ? 0 : 1;
	D_ASSERT(root->children.size() == 2);
	(void)root;
	// any arithmetic operator involving NULL is always NULL
	if (constant->value.IsNull()) {
		return make_unique<BoundConstantExpression>(Value(root->return_type));
	}
	auto &func_name = root->function.name;
	if (func_name == "+") {
		if (constant->value == 0) {
			// addition with 0
			// we can remove the entire operator and replace it with the non-constant child
			return move(root->children[1 - constant_child]);
		}
	} else if (func_name == "-") {
		if (constant_child == 1 && constant->value == 0) {
			// subtraction by 0
			// we can remove the entire operator and replace it with the non-constant child
			return move(root->children[1 - constant_child]);
		}
	} else if (func_name == "*") {
		if (constant->value == 1) {
			// multiply with 1, replace with non-constant child
			return move(root->children[1 - constant_child]);
		} else if (constant->value == 0) {
			// multiply by zero: replace with constant or null
			return ExpressionRewriter::ConstantOrNull(move(root->children[1 - constant_child]),
			                                          Value::Numeric(root->return_type, 0));
		}
	} else {
		D_ASSERT(func_name == "/");
		if (constant_child == 1) {
			if (constant->value == 1) {
				// divide by 1, replace with non-constant child
				return move(root->children[1 - constant_child]);
			} else if (constant->value == 0) {
				// divide by 0, replace with NULL
				return make_unique<BoundConstantExpression>(Value(root->return_type));
			}
		}
	}
	return nullptr;
}
} // namespace duckdb





namespace duckdb {

CaseSimplificationRule::CaseSimplificationRule(ExpressionRewriter &rewriter) : Rule(rewriter) {
	// match on a CaseExpression that has a ConstantExpression as a check
	auto op = make_unique<CaseExpressionMatcher>();
	root = move(op);
}

unique_ptr<Expression> CaseSimplificationRule::Apply(LogicalOperator &op, vector<Expression *> &bindings,
                                                     bool &changes_made, bool is_root) {
	auto root = (BoundCaseExpression *)bindings[0];
	for (idx_t i = 0; i < root->case_checks.size(); i++) {
		auto &case_check = root->case_checks[i];
		if (case_check.when_expr->IsFoldable()) {
			// the WHEN check is a foldable expression
			// use an ExpressionExecutor to execute the expression
			auto constant_value = ExpressionExecutor::EvaluateScalar(GetContext(), *case_check.when_expr);

			// fold based on the constant condition
			auto condition = constant_value.DefaultCastAs(LogicalType::BOOLEAN);
			if (condition.IsNull() || !BooleanValue::Get(condition)) {
				// the condition is always false: remove this case check
				root->case_checks.erase(root->case_checks.begin() + i);
				i--;
			} else {
				// the condition is always true
				// move the THEN clause to the ELSE of the case
				root->else_expr = move(case_check.then_expr);
				// remove this case check and any case checks after this one
				root->case_checks.erase(root->case_checks.begin() + i, root->case_checks.end());
				break;
			}
		}
	}
	if (root->case_checks.empty()) {
		// no case checks left: return the ELSE expression
		return move(root->else_expr);
	}
	return nullptr;
}

} // namespace duckdb






namespace duckdb {

ComparisonSimplificationRule::ComparisonSimplificationRule(ExpressionRewriter &rewriter) : Rule(rewriter) {
	// match on a ComparisonExpression that has a ConstantExpression as a check
	auto op = make_unique<ComparisonExpressionMatcher>();
	op->matchers.push_back(make_unique<FoldableConstantMatcher>());
	op->policy = SetMatcher::Policy::SOME;
	root = move(op);
}

unique_ptr<Expression> ComparisonSimplificationRule::Apply(LogicalOperator &op, vector<Expression *> &bindings,
                                                           bool &changes_made, bool is_root) {
	D_ASSERT(bindings[0]->expression_class == ExpressionClass::BOUND_COMPARISON);
	auto expr = (BoundComparisonExpression *)bindings[0];
	auto constant_expr = bindings[1];
	bool column_ref_left = expr->left.get() != constant_expr;
	auto column_ref_expr = !column_ref_left ? expr->right.get() : expr->left.get();
	// the constant_expr is a scalar expression that we have to fold
	// use an ExpressionExecutor to execute the expression
	D_ASSERT(constant_expr->IsFoldable());
	Value constant_value;
	if (!ExpressionExecutor::TryEvaluateScalar(GetContext(), *constant_expr, constant_value)) {
		return nullptr;
	}
	if (constant_value.IsNull() && !(expr->type == ExpressionType::COMPARE_NOT_DISTINCT_FROM ||
	                                 expr->type == ExpressionType::COMPARE_DISTINCT_FROM)) {
		// comparison with constant NULL, return NULL
		return make_unique<BoundConstantExpression>(Value(LogicalType::BOOLEAN));
	}
	if (column_ref_expr->expression_class == ExpressionClass::BOUND_CAST) {
		//! Here we check if we can apply the expression on the constant side
		//! We can do this if the cast itself is invertible and casting the constant is
		//! invertible in practice.
		auto cast_expression = (BoundCastExpression *)column_ref_expr;
		auto target_type = cast_expression->source_type();
		if (!BoundCastExpression::CastIsInvertible(target_type, cast_expression->return_type)) {
			return nullptr;
		}

		// Can we cast the constant at all?
		string error_message;
		Value cast_constant;
		auto new_constant = constant_value.DefaultTryCastAs(target_type, cast_constant, &error_message, true);
		if (!new_constant) {
			return nullptr;
		}

		// Is the constant cast invertible?
		if (!cast_constant.IsNull() &&
		    !BoundCastExpression::CastIsInvertible(cast_expression->return_type, target_type)) {
			// Is it actually invertible?
			Value uncast_constant;
			if (!cast_constant.DefaultTryCastAs(constant_value.type(), uncast_constant, &error_message, true) ||
			    uncast_constant != constant_value) {
				return nullptr;
			}
		}

		//! We can cast, now we change our column_ref_expression from an operator cast to a column reference
		auto child_expression = move(cast_expression->child);
		auto new_constant_expr = make_unique<BoundConstantExpression>(cast_constant);
		if (column_ref_left) {
			expr->left = move(child_expression);
			expr->right = move(new_constant_expr);
		} else {
			expr->left = move(new_constant_expr);
			expr->right = move(child_expression);
		}
	}
	return nullptr;
}

} // namespace duckdb






namespace duckdb {

ConjunctionSimplificationRule::ConjunctionSimplificationRule(ExpressionRewriter &rewriter) : Rule(rewriter) {
	// match on a ComparisonExpression that has a ConstantExpression as a check
	auto op = make_unique<ConjunctionExpressionMatcher>();
	op->matchers.push_back(make_unique<FoldableConstantMatcher>());
	op->policy = SetMatcher::Policy::SOME;
	root = move(op);
}

unique_ptr<Expression> ConjunctionSimplificationRule::RemoveExpression(BoundConjunctionExpression &conj,
                                                                       Expression *expr) {
	for (idx_t i = 0; i < conj.children.size(); i++) {
		if (conj.children[i].get() == expr) {
			// erase the expression
			conj.children.erase(conj.children.begin() + i);
			break;
		}
	}
	if (conj.children.size() == 1) {
		// one expression remaining: simply return that expression and erase the conjunction
		return move(conj.children[0]);
	}
	return nullptr;
}

unique_ptr<Expression> ConjunctionSimplificationRule::Apply(LogicalOperator &op, vector<Expression *> &bindings,
                                                            bool &changes_made, bool is_root) {
	auto conjunction = (BoundConjunctionExpression *)bindings[0];
	auto constant_expr = bindings[1];
	// the constant_expr is a scalar expression that we have to fold
	// use an ExpressionExecutor to execute the expression
	D_ASSERT(constant_expr->IsFoldable());
	Value constant_value;
	if (!ExpressionExecutor::TryEvaluateScalar(GetContext(), *constant_expr, constant_value)) {
		return nullptr;
	}
	constant_value = constant_value.DefaultCastAs(LogicalType::BOOLEAN);
	if (constant_value.IsNull()) {
		// we can't simplify conjunctions with a constant NULL
		return nullptr;
	}
	if (conjunction->type == ExpressionType::CONJUNCTION_AND) {
		if (!BooleanValue::Get(constant_value)) {
			// FALSE in AND, result of expression is false
			return make_unique<BoundConstantExpression>(Value::BOOLEAN(false));
		} else {
			// TRUE in AND, remove the expression from the set
			return RemoveExpression(*conjunction, constant_expr);
		}
	} else {
		D_ASSERT(conjunction->type == ExpressionType::CONJUNCTION_OR);
		if (!BooleanValue::Get(constant_value)) {
			// FALSE in OR, remove the expression from the set
			return RemoveExpression(*conjunction, constant_expr);
		} else {
			// TRUE in OR, result of expression is true
			return make_unique<BoundConstantExpression>(Value::BOOLEAN(true));
		}
	}
}

} // namespace duckdb







namespace duckdb {

//! The ConstantFoldingExpressionMatcher matches on any scalar expression (i.e. Expression::IsFoldable is true)
class ConstantFoldingExpressionMatcher : public FoldableConstantMatcher {
public:
	bool Match(Expression *expr, vector<Expression *> &bindings) override {
		// we also do not match on ConstantExpressions, because we cannot fold those any further
		if (expr->type == ExpressionType::VALUE_CONSTANT) {
			return false;
		}
		return FoldableConstantMatcher::Match(expr, bindings);
	}
};

ConstantFoldingRule::ConstantFoldingRule(ExpressionRewriter &rewriter) : Rule(rewriter) {
	auto op = make_unique<ConstantFoldingExpressionMatcher>();
	root = move(op);
}

unique_ptr<Expression> ConstantFoldingRule::Apply(LogicalOperator &op, vector<Expression *> &bindings,
                                                  bool &changes_made, bool is_root) {
	auto root = bindings[0];
	// the root is a scalar expression that we have to fold
	D_ASSERT(root->IsFoldable() && root->type != ExpressionType::VALUE_CONSTANT);

	// use an ExpressionExecutor to execute the expression
	Value result_value;
	if (!ExpressionExecutor::TryEvaluateScalar(GetContext(), *root, result_value)) {
		return nullptr;
	}
	D_ASSERT(result_value.type().InternalType() == root->return_type.InternalType());
	// now get the value from the result vector and insert it back into the plan as a constant expression
	return make_unique<BoundConstantExpression>(result_value);
}

} // namespace duckdb











namespace duckdb {

DatePartSimplificationRule::DatePartSimplificationRule(ExpressionRewriter &rewriter) : Rule(rewriter) {
	auto func = make_unique<FunctionExpressionMatcher>();
	func->function = make_unique<SpecificFunctionMatcher>("date_part");
	func->matchers.push_back(make_unique<ConstantExpressionMatcher>());
	func->matchers.push_back(make_unique<ExpressionMatcher>());
	func->policy = SetMatcher::Policy::ORDERED;
	root = move(func);
}

unique_ptr<Expression> DatePartSimplificationRule::Apply(LogicalOperator &op, vector<Expression *> &bindings,
                                                         bool &changes_made, bool is_root) {
	auto &date_part = (BoundFunctionExpression &)*bindings[0];
	auto &constant_expr = (BoundConstantExpression &)*bindings[1];
	auto &constant = constant_expr.value;

	if (constant.IsNull()) {
		// NULL specifier: return constant NULL
		return make_unique<BoundConstantExpression>(Value(date_part.return_type));
	}
	// otherwise check the specifier
	auto specifier = GetDatePartSpecifier(StringValue::Get(constant));
	string new_function_name;
	switch (specifier) {
	case DatePartSpecifier::YEAR:
		new_function_name = "year";
		break;
	case DatePartSpecifier::MONTH:
		new_function_name = "month";
		break;
	case DatePartSpecifier::DAY:
		new_function_name = "day";
		break;
	case DatePartSpecifier::DECADE:
		new_function_name = "decade";
		break;
	case DatePartSpecifier::CENTURY:
		new_function_name = "century";
		break;
	case DatePartSpecifier::MILLENNIUM:
		new_function_name = "millennium";
		break;
	case DatePartSpecifier::QUARTER:
		new_function_name = "quarter";
		break;
	case DatePartSpecifier::WEEK:
		new_function_name = "week";
		break;
	case DatePartSpecifier::YEARWEEK:
		new_function_name = "yearweek";
		break;
	case DatePartSpecifier::DOW:
		new_function_name = "dayofweek";
		break;
	case DatePartSpecifier::ISODOW:
		new_function_name = "isodow";
		break;
	case DatePartSpecifier::DOY:
		new_function_name = "dayofyear";
		break;
	case DatePartSpecifier::EPOCH:
		new_function_name = "epoch";
		break;
	case DatePartSpecifier::MICROSECONDS:
		new_function_name = "microsecond";
		break;
	case DatePartSpecifier::MILLISECONDS:
		new_function_name = "millisecond";
		break;
	case DatePartSpecifier::SECOND:
		new_function_name = "second";
		break;
	case DatePartSpecifier::MINUTE:
		new_function_name = "minute";
		break;
	case DatePartSpecifier::HOUR:
		new_function_name = "hour";
		break;
	default:
		return nullptr;
	}
	// found a replacement function: bind it
	vector<unique_ptr<Expression>> children;
	children.push_back(move(date_part.children[1]));

	string error;
	FunctionBinder binder(rewriter.context);
	auto function = binder.BindScalarFunction(DEFAULT_SCHEMA, new_function_name, move(children), error, false);
	if (!function) {
		throw BinderException(error);
	}
	return function;
}

} // namespace duckdb








namespace duckdb {

DistributivityRule::DistributivityRule(ExpressionRewriter &rewriter) : Rule(rewriter) {
	// we match on an OR expression within a LogicalFilter node
	root = make_unique<ExpressionMatcher>();
	root->expr_type = make_unique<SpecificExpressionTypeMatcher>(ExpressionType::CONJUNCTION_OR);
}

void DistributivityRule::AddExpressionSet(Expression &expr, expression_set_t &set) {
	if (expr.type == ExpressionType::CONJUNCTION_AND) {
		auto &and_expr = (BoundConjunctionExpression &)expr;
		for (auto &child : and_expr.children) {
			set.insert(child.get());
		}
	} else {
		set.insert(&expr);
	}
}

unique_ptr<Expression> DistributivityRule::ExtractExpression(BoundConjunctionExpression &conj, idx_t idx,
                                                             Expression &expr) {
	auto &child = conj.children[idx];
	unique_ptr<Expression> result;
	if (child->type == ExpressionType::CONJUNCTION_AND) {
		// AND, remove expression from the list
		auto &and_expr = (BoundConjunctionExpression &)*child;
		for (idx_t i = 0; i < and_expr.children.size(); i++) {
			if (Expression::Equals(and_expr.children[i].get(), &expr)) {
				result = move(and_expr.children[i]);
				and_expr.children.erase(and_expr.children.begin() + i);
				break;
			}
		}
		if (and_expr.children.size() == 1) {
			conj.children[idx] = move(and_expr.children[0]);
		}
	} else {
		// not an AND node! remove the entire expression
		// this happens in the case of e.g. (X AND B) OR X
		D_ASSERT(Expression::Equals(child.get(), &expr));
		result = move(child);
		conj.children[idx] = nullptr;
	}
	D_ASSERT(result);
	return result;
}

unique_ptr<Expression> DistributivityRule::Apply(LogicalOperator &op, vector<Expression *> &bindings,
                                                 bool &changes_made, bool is_root) {
	auto initial_or = (BoundConjunctionExpression *)bindings[0];

	// we want to find expressions that occur in each of the children of the OR
	// i.e. (X AND A) OR (X AND B) => X occurs in all branches
	// first, for the initial child, we create an expression set of which expressions occur
	// this is our initial candidate set (in the example: [X, A])
	expression_set_t candidate_set;
	AddExpressionSet(*initial_or->children[0], candidate_set);
	// now for each of the remaining children, we create a set again and intersect them
	// in our example: the second set would be [X, B]
	// the intersection would leave [X]
	for (idx_t i = 1; i < initial_or->children.size(); i++) {
		expression_set_t next_set;
		AddExpressionSet(*initial_or->children[i], next_set);
		expression_set_t intersect_result;
		for (auto &expr : candidate_set) {
			if (next_set.find(expr) != next_set.end()) {
				intersect_result.insert(expr);
			}
		}
		candidate_set = intersect_result;
	}
	if (candidate_set.empty()) {
		// nothing found: abort
		return nullptr;
	}
	// now for each of the remaining expressions in the candidate set we know that it is contained in all branches of
	// the OR
	auto new_root = make_unique<BoundConjunctionExpression>(ExpressionType::CONJUNCTION_AND);
	for (auto &expr : candidate_set) {
		D_ASSERT(initial_or->children.size() > 0);

		// extract the expression from the first child of the OR
		auto result = ExtractExpression(*initial_or, 0, (Expression &)*expr);
		// now for the subsequent expressions, simply remove the expression
		for (idx_t i = 1; i < initial_or->children.size(); i++) {
			ExtractExpression(*initial_or, i, *result);
		}
		// now we add the expression to the new root
		new_root->children.push_back(move(result));
	}

	// check if we completely erased one of the children of the OR
	// this happens if we have an OR in the form of "X OR (X AND A)"
	// the left child will be completely empty, as it only contains common expressions
	// in this case, any other children are not useful:
	// X OR (X AND A) is the same as "X"
	// since (1) only tuples that do not qualify "X" will not pass this predicate
	//   and (2) all tuples that qualify "X" will pass this predicate
	for (idx_t i = 0; i < initial_or->children.size(); i++) {
		if (!initial_or->children[i]) {
			if (new_root->children.size() <= 1) {
				return move(new_root->children[0]);
			} else {
				return move(new_root);
			}
		}
	}
	// finally we need to add the remaining expressions in the OR to the new root
	if (initial_or->children.size() == 1) {
		// one child: skip the OR entirely and only add the single child
		new_root->children.push_back(move(initial_or->children[0]));
	} else if (initial_or->children.size() > 1) {
		// multiple children still remain: push them into a new OR and add that to the new root
		auto new_or = make_unique<BoundConjunctionExpression>(ExpressionType::CONJUNCTION_OR);
		for (auto &child : initial_or->children) {
			new_or->children.push_back(move(child));
		}
		new_root->children.push_back(move(new_or));
	}
	// finally return the new root
	if (new_root->children.size() == 1) {
		return move(new_root->children[0]);
	}
	return move(new_root);
}

} // namespace duckdb









namespace duckdb {

EmptyNeedleRemovalRule::EmptyNeedleRemovalRule(ExpressionRewriter &rewriter) : Rule(rewriter) {
	// match on a FunctionExpression that has a foldable ConstantExpression
	auto func = make_unique<FunctionExpressionMatcher>();
	func->matchers.push_back(make_unique<ExpressionMatcher>());
	func->matchers.push_back(make_unique<ExpressionMatcher>());
	func->policy = SetMatcher::Policy::SOME;

	unordered_set<string> functions = {"prefix", "contains", "suffix"};
	func->function = make_unique<ManyFunctionMatcher>(functions);
	root = move(func);
}

unique_ptr<Expression> EmptyNeedleRemovalRule::Apply(LogicalOperator &op, vector<Expression *> &bindings,
                                                     bool &changes_made, bool is_root) {
	auto root = (BoundFunctionExpression *)bindings[0];
	D_ASSERT(root->children.size() == 2);
	(void)root;
	auto prefix_expr = bindings[2];

	// the constant_expr is a scalar expression that we have to fold
	if (!prefix_expr->IsFoldable()) {
		return nullptr;
	}
	D_ASSERT(root->return_type.id() == LogicalTypeId::BOOLEAN);

	auto prefix_value = ExpressionExecutor::EvaluateScalar(GetContext(), *prefix_expr);

	if (prefix_value.IsNull()) {
		return make_unique<BoundConstantExpression>(Value(LogicalType::BOOLEAN));
	}

	D_ASSERT(prefix_value.type() == prefix_expr->return_type);
	auto &needle_string = StringValue::Get(prefix_value);

	// PREFIX('xyz', '') is TRUE
	// PREFIX(NULL, '') is NULL
	// so rewrite PREFIX(x, '') to TRUE_OR_NULL(x)
	if (needle_string.empty()) {
		return ExpressionRewriter::ConstantOrNull(move(root->children[0]), Value::BOOLEAN(true));
	}
	return nullptr;
}

} // namespace duckdb









namespace duckdb {

EnumComparisonRule::EnumComparisonRule(ExpressionRewriter &rewriter) : Rule(rewriter) {
	// match on a ComparisonExpression that is an Equality and has a VARCHAR and ENUM as its children
	auto op = make_unique<ComparisonExpressionMatcher>();
	// Enum requires expression to be root
	op->expr_type = make_unique<SpecificExpressionTypeMatcher>(ExpressionType::COMPARE_EQUAL);
	for (idx_t i = 0; i < 2; i++) {
		auto child = make_unique<CastExpressionMatcher>();
		child->type = make_unique<TypeMatcherId>(LogicalTypeId::VARCHAR);
		child->matcher = make_unique<ExpressionMatcher>();
		child->matcher->type = make_unique<TypeMatcherId>(LogicalTypeId::ENUM);
		op->matchers.push_back(move(child));
	}
	root = move(op);
}

bool AreMatchesPossible(LogicalType &left, LogicalType &right) {
	LogicalType *small_enum, *big_enum;
	if (EnumType::GetSize(left) < EnumType::GetSize(right)) {
		small_enum = &left;
		big_enum = &right;
	} else {
		small_enum = &right;
		big_enum = &left;
	}
	auto &string_vec = EnumType::GetValuesInsertOrder(*small_enum);
	auto string_vec_ptr = FlatVector::GetData<string_t>(string_vec);
	auto size = EnumType::GetSize(*small_enum);
	for (idx_t i = 0; i < size; i++) {
		auto key = string_vec_ptr[i].GetString();
		if (EnumType::GetPos(*big_enum, key) != -1) {
			return true;
		}
	}
	return false;
}
unique_ptr<Expression> EnumComparisonRule::Apply(LogicalOperator &op, vector<Expression *> &bindings,
                                                 bool &changes_made, bool is_root) {

	auto root = (BoundComparisonExpression *)bindings[0];
	auto left_child = (BoundCastExpression *)bindings[1];
	auto right_child = (BoundCastExpression *)bindings[3];

	if (!AreMatchesPossible(left_child->child->return_type, right_child->child->return_type)) {
		vector<unique_ptr<Expression>> children;
		children.push_back(move(root->left));
		children.push_back(move(root->right));
		return ExpressionRewriter::ConstantOrNull(move(children), Value::BOOLEAN(false));
	}

	if (!is_root || op.type != LogicalOperatorType::LOGICAL_FILTER) {
		return nullptr;
	}

	auto cast_left_to_right =
	    BoundCastExpression::AddDefaultCastToType(move(left_child->child), right_child->child->return_type, true);
	return make_unique<BoundComparisonExpression>(root->type, move(cast_left_to_right), move(right_child->child));
}

} // namespace duckdb






namespace duckdb {

EqualOrNullSimplification::EqualOrNullSimplification(ExpressionRewriter &rewriter) : Rule(rewriter) {
	// match on OR conjunction
	auto op = make_unique<ConjunctionExpressionMatcher>();
	op->expr_type = make_unique<SpecificExpressionTypeMatcher>(ExpressionType::CONJUNCTION_OR);
	op->policy = SetMatcher::Policy::SOME;

	// equi comparison on one side
	auto equal_child = make_unique<ComparisonExpressionMatcher>();
	equal_child->expr_type = make_unique<SpecificExpressionTypeMatcher>(ExpressionType::COMPARE_EQUAL);
	equal_child->policy = SetMatcher::Policy::SOME;
	op->matchers.push_back(move(equal_child));

	// AND conjuction on the other
	auto and_child = make_unique<ConjunctionExpressionMatcher>();
	and_child->expr_type = make_unique<SpecificExpressionTypeMatcher>(ExpressionType::CONJUNCTION_AND);
	and_child->policy = SetMatcher::Policy::SOME;

	// IS NULL tests inside AND
	auto isnull_child = make_unique<ExpressionMatcher>();
	isnull_child->expr_type = make_unique<SpecificExpressionTypeMatcher>(ExpressionType::OPERATOR_IS_NULL);
	// I could try to use std::make_unique for a copy, but it's available from C++14 only
	auto isnull_child2 = make_unique<ExpressionMatcher>();
	isnull_child2->expr_type = make_unique<SpecificExpressionTypeMatcher>(ExpressionType::OPERATOR_IS_NULL);
	and_child->matchers.push_back(move(isnull_child));
	and_child->matchers.push_back(move(isnull_child2));

	op->matchers.push_back(move(and_child));
	root = move(op);
}

// a=b OR (a IS NULL AND b IS NULL) to a IS NOT DISTINCT FROM b
static unique_ptr<Expression> TryRewriteEqualOrIsNull(const Expression *equal_expr, const Expression *and_expr) {
	if (equal_expr->type != ExpressionType::COMPARE_EQUAL || and_expr->type != ExpressionType::CONJUNCTION_AND) {
		return nullptr;
	}

	const auto equal_cast = (BoundComparisonExpression *)equal_expr;
	const auto and_cast = (BoundConjunctionExpression *)and_expr;

	if (and_cast->children.size() != 2) {
		return nullptr;
	}

	// Make sure on the AND conjuction the relevant conditions appear
	const auto a_exp = equal_cast->left.get();
	const auto b_exp = equal_cast->right.get();
	bool valid = true;
	bool a_is_null_found = false;
	bool b_is_null_found = false;

	for (const auto &item : and_cast->children) {
		const auto next_exp = item.get();

		if (next_exp->type == ExpressionType::OPERATOR_IS_NULL) {
			const auto next_exp_cast = (BoundOperatorExpression *)next_exp;
			const auto child = next_exp_cast->children[0].get();

			// Test for equality on both 'a' and 'b' expressions
			if (Expression::Equals(child, a_exp)) {
				a_is_null_found = true;
			} else if (Expression::Equals(child, b_exp)) {
				b_is_null_found = true;
			} else {
				valid = false;
				break;
			}
		} else {
			valid = false;
			break;
		}
	}
	if (valid && a_is_null_found && b_is_null_found) {
		return make_unique<BoundComparisonExpression>(ExpressionType::COMPARE_NOT_DISTINCT_FROM, move(equal_cast->left),
		                                              move(equal_cast->right));
	}
	return nullptr;
}

unique_ptr<Expression> EqualOrNullSimplification::Apply(LogicalOperator &op, vector<Expression *> &bindings,
                                                        bool &changes_made, bool is_root) {
	const Expression *or_exp = bindings[0];

	if (or_exp->type != ExpressionType::CONJUNCTION_OR) {
		return nullptr;
	}

	const auto or_exp_cast = (BoundConjunctionExpression *)or_exp;

	if (or_exp_cast->children.size() != 2) {
		return nullptr;
	}

	const auto left_exp = or_exp_cast->children[0].get();
	const auto right_exp = or_exp_cast->children[1].get();
	// Test for: a=b OR (a IS NULL AND b IS NULL)
	auto first_try = TryRewriteEqualOrIsNull(left_exp, right_exp);
	if (first_try) {
		return first_try;
	}
	// Test for: (a IS NULL AND b IS NULL) OR a=b
	return TryRewriteEqualOrIsNull(right_exp, left_exp);
}

} // namespace duckdb





namespace duckdb {

InClauseSimplificationRule::InClauseSimplificationRule(ExpressionRewriter &rewriter) : Rule(rewriter) {
	// match on InClauseExpression that has a ConstantExpression as a check
	auto op = make_unique<InClauseExpressionMatcher>();
	op->policy = SetMatcher::Policy::SOME;
	root = move(op);
}

unique_ptr<Expression> InClauseSimplificationRule::Apply(LogicalOperator &op, vector<Expression *> &bindings,
                                                         bool &changes_made, bool is_root) {
	D_ASSERT(bindings[0]->expression_class == ExpressionClass::BOUND_OPERATOR);
	auto expr = (BoundOperatorExpression *)bindings[0];
	if (expr->children[0]->expression_class != ExpressionClass::BOUND_CAST) {
		return nullptr;
	}
	auto cast_expression = (BoundCastExpression *)expr->children[0].get();
	if (cast_expression->child->expression_class != ExpressionClass::BOUND_COLUMN_REF) {
		return nullptr;
	}
	//! Here we check if we can apply the expression on the constant side
	auto target_type = cast_expression->source_type();
	if (!BoundCastExpression::CastIsInvertible(cast_expression->return_type, target_type)) {
		return nullptr;
	}
	vector<unique_ptr<BoundConstantExpression>> cast_list;
	//! First check if we can cast all children
	for (size_t i = 1; i < expr->children.size(); i++) {
		if (expr->children[i]->expression_class != ExpressionClass::BOUND_CONSTANT) {
			return nullptr;
		}
		D_ASSERT(expr->children[i]->IsFoldable());
		auto constant_value = ExpressionExecutor::EvaluateScalar(GetContext(), *expr->children[i]);
		auto new_constant = constant_value.DefaultTryCastAs(target_type);
		if (!new_constant) {
			return nullptr;
		} else {
			auto new_constant_expr = make_unique<BoundConstantExpression>(constant_value);
			cast_list.push_back(move(new_constant_expr));
		}
	}
	//! We can cast, so we move the new constant
	for (size_t i = 1; i < expr->children.size(); i++) {
		expr->children[i] = move(cast_list[i - 1]);

		//		expr->children[i] = move(new_constant_expr);
	}
	//! We can cast the full list, so we move the column
	expr->children[0] = move(cast_expression->child);
	return nullptr;
}

} // namespace duckdb








namespace duckdb {

LikeOptimizationRule::LikeOptimizationRule(ExpressionRewriter &rewriter) : Rule(rewriter) {
	// match on a FunctionExpression that has a foldable ConstantExpression
	auto func = make_unique<FunctionExpressionMatcher>();
	func->matchers.push_back(make_unique<ExpressionMatcher>());
	func->matchers.push_back(make_unique<ConstantExpressionMatcher>());
	func->policy = SetMatcher::Policy::ORDERED;
	// we match on LIKE ("~~") and NOT LIKE ("!~~")
	func->function = make_unique<ManyFunctionMatcher>(unordered_set<string> {"!~~", "~~"});
	root = move(func);
}

static bool PatternIsConstant(const string &pattern) {
	for (idx_t i = 0; i < pattern.size(); i++) {
		if (pattern[i] == '%' || pattern[i] == '_') {
			return false;
		}
	}
	return true;
}

static bool PatternIsPrefix(const string &pattern) {
	idx_t i;
	for (i = pattern.size(); i > 0; i--) {
		if (pattern[i - 1] != '%') {
			break;
		}
	}
	if (i == pattern.size()) {
		// no trailing %
		// cannot be a prefix
		return false;
	}
	// continue to look in the string
	// if there is a % or _ in the string (besides at the very end) this is not a prefix match
	for (; i > 0; i--) {
		if (pattern[i - 1] == '%' || pattern[i - 1] == '_') {
			return false;
		}
	}
	return true;
}

static bool PatternIsSuffix(const string &pattern) {
	idx_t i;
	for (i = 0; i < pattern.size(); i++) {
		if (pattern[i] != '%') {
			break;
		}
	}
	if (i == 0) {
		// no leading %
		// cannot be a suffix
		return false;
	}
	// continue to look in the string
	// if there is a % or _ in the string (besides at the beginning) this is not a suffix match
	for (; i < pattern.size(); i++) {
		if (pattern[i] == '%' || pattern[i] == '_') {
			return false;
		}
	}
	return true;
}

static bool PatternIsContains(const string &pattern) {
	idx_t start;
	idx_t end;
	for (start = 0; start < pattern.size(); start++) {
		if (pattern[start] != '%') {
			break;
		}
	}
	for (end = pattern.size(); end > 0; end--) {
		if (pattern[end - 1] != '%') {
			break;
		}
	}
	if (start == 0 || end == pattern.size()) {
		// contains requires both a leading AND a trailing %
		return false;
	}
	// check if there are any other special characters in the string
	// if there is a % or _ in the string (besides at the beginning/end) this is not a contains match
	for (idx_t i = start; i < end; i++) {
		if (pattern[i] == '%' || pattern[i] == '_') {
			return false;
		}
	}
	return true;
}

unique_ptr<Expression> LikeOptimizationRule::Apply(LogicalOperator &op, vector<Expression *> &bindings,
                                                   bool &changes_made, bool is_root) {
	auto root = (BoundFunctionExpression *)bindings[0];
	auto constant_expr = (BoundConstantExpression *)bindings[2];
	D_ASSERT(root->children.size() == 2);

	if (constant_expr->value.IsNull()) {
		return make_unique<BoundConstantExpression>(Value(root->return_type));
	}

	// the constant_expr is a scalar expression that we have to fold
	if (!constant_expr->IsFoldable()) {
		return nullptr;
	}

	auto constant_value = ExpressionExecutor::EvaluateScalar(GetContext(), *constant_expr);
	D_ASSERT(constant_value.type() == constant_expr->return_type);
	auto &patt_str = StringValue::Get(constant_value);

	bool is_not_like = root->function.name == "!~~";
	if (PatternIsConstant(patt_str)) {
		// Pattern is constant
		return make_unique<BoundComparisonExpression>(is_not_like ? ExpressionType::COMPARE_NOTEQUAL
		                                                          : ExpressionType::COMPARE_EQUAL,
		                                              move(root->children[0]), move(root->children[1]));
	} else if (PatternIsPrefix(patt_str)) {
		// Prefix LIKE pattern : [^%_]*[%]+, ignoring underscore
		return ApplyRule(root, PrefixFun::GetFunction(), patt_str, is_not_like);
	} else if (PatternIsSuffix(patt_str)) {
		// Suffix LIKE pattern: [%]+[^%_]*, ignoring underscore
		return ApplyRule(root, SuffixFun::GetFunction(), patt_str, is_not_like);
	} else if (PatternIsContains(patt_str)) {
		// Contains LIKE pattern: [%]+[^%_]*[%]+, ignoring underscore
		return ApplyRule(root, ContainsFun::GetFunction(), patt_str, is_not_like);
	}
	return nullptr;
}

unique_ptr<Expression> LikeOptimizationRule::ApplyRule(BoundFunctionExpression *expr, ScalarFunction function,
                                                       string pattern, bool is_not_like) {
	// replace LIKE by an optimized function
	unique_ptr<Expression> result;
	auto new_function =
	    make_unique<BoundFunctionExpression>(expr->return_type, move(function), move(expr->children), nullptr);

	// removing "%" from the pattern
	pattern.erase(std::remove(pattern.begin(), pattern.end(), '%'), pattern.end());

	new_function->children[1] = make_unique<BoundConstantExpression>(Value(move(pattern)));

	result = move(new_function);
	if (is_not_like) {
		auto negation = make_unique<BoundOperatorExpression>(ExpressionType::OPERATOR_NOT, LogicalType::BOOLEAN);
		negation->children.push_back(move(result));
		result = move(negation);
	}

	return result;
}

} // namespace duckdb









namespace duckdb {

MoveConstantsRule::MoveConstantsRule(ExpressionRewriter &rewriter) : Rule(rewriter) {
	auto op = make_unique<ComparisonExpressionMatcher>();
	op->matchers.push_back(make_unique<ConstantExpressionMatcher>());
	op->policy = SetMatcher::Policy::UNORDERED;

	auto arithmetic = make_unique<FunctionExpressionMatcher>();
	// we handle multiplication, addition and subtraction because those are "easy"
	// integer division makes the division case difficult
	// e.g. [x / 2 = 3] means [x = 6 OR x = 7] because of truncation -> no clean rewrite rules
	arithmetic->function = make_unique<ManyFunctionMatcher>(unordered_set<string> {"+", "-", "*"});
	// we match only on integral numeric types
	arithmetic->type = make_unique<IntegerTypeMatcher>();
	arithmetic->matchers.push_back(make_unique<ConstantExpressionMatcher>());
	arithmetic->matchers.push_back(make_unique<ExpressionMatcher>());
	arithmetic->policy = SetMatcher::Policy::SOME;
	op->matchers.push_back(move(arithmetic));
	root = move(op);
}

unique_ptr<Expression> MoveConstantsRule::Apply(LogicalOperator &op, vector<Expression *> &bindings, bool &changes_made,
                                                bool is_root) {
	auto comparison = (BoundComparisonExpression *)bindings[0];
	auto outer_constant = (BoundConstantExpression *)bindings[1];
	auto arithmetic = (BoundFunctionExpression *)bindings[2];
	auto inner_constant = (BoundConstantExpression *)bindings[3];
	if (!TypeIsIntegral(arithmetic->return_type.InternalType())) {
		return nullptr;
	}
	if (inner_constant->value.IsNull() || outer_constant->value.IsNull()) {
		return make_unique<BoundConstantExpression>(Value(comparison->return_type));
	}
	auto &constant_type = outer_constant->return_type;
	hugeint_t outer_value = IntegralValue::Get(outer_constant->value);
	hugeint_t inner_value = IntegralValue::Get(inner_constant->value);

	idx_t arithmetic_child_index = arithmetic->children[0].get() == inner_constant ? 1 : 0;
	auto &op_type = arithmetic->function.name;
	if (op_type == "+") {
		// [x + 1 COMP 10] OR [1 + x COMP 10]
		// order does not matter in addition:
		// simply change right side to 10-1 (outer_constant - inner_constant)
		if (!Hugeint::SubtractInPlace(outer_value, inner_value)) {
			return nullptr;
		}
		auto result_value = Value::HUGEINT(outer_value);
		if (!result_value.DefaultTryCastAs(constant_type)) {
			// if the cast is not possible then the comparison is not possible
			// for example, if we have x + 5 = 3, where x is an unsigned number, we will get x = -2
			// since this is not possible we can remove the entire branch here
			return ExpressionRewriter::ConstantOrNull(move(arithmetic->children[arithmetic_child_index]),
			                                          Value::BOOLEAN(false));
		}
		outer_constant->value = move(result_value);
	} else if (op_type == "-") {
		// [x - 1 COMP 10] O R [1 - x COMP 10]
		// order matters in subtraction:
		if (arithmetic_child_index == 0) {
			// [x - 1 COMP 10]
			// change right side to 10+1 (outer_constant + inner_constant)
			if (!Hugeint::AddInPlace(outer_value, inner_value)) {
				return nullptr;
			}
			auto result_value = Value::HUGEINT(outer_value);
			if (!result_value.DefaultTryCastAs(constant_type)) {
				// if the cast is not possible then the comparison is not possible
				return ExpressionRewriter::ConstantOrNull(move(arithmetic->children[arithmetic_child_index]),
				                                          Value::BOOLEAN(false));
			}
			outer_constant->value = move(result_value);
		} else {
			// [1 - x COMP 10]
			// change right side to 1-10=-9
			if (!Hugeint::SubtractInPlace(inner_value, outer_value)) {
				return nullptr;
			}
			auto result_value = Value::HUGEINT(inner_value);
			if (!result_value.DefaultTryCastAs(constant_type)) {
				// if the cast is not possible then the comparison is not possible
				return ExpressionRewriter::ConstantOrNull(move(arithmetic->children[arithmetic_child_index]),
				                                          Value::BOOLEAN(false));
			}
			outer_constant->value = move(result_value);
			// in this case, we should also flip the comparison
			// e.g. if we have [4 - x < 2] then we should have [x > 2]
			comparison->type = FlipComparisionExpression(comparison->type);
		}
	} else {
		D_ASSERT(op_type == "*");
		// [x * 2 COMP 10] OR [2 * x COMP 10]
		// order does not matter in multiplication:
		// change right side to 10/2 (outer_constant / inner_constant)
		// but ONLY if outer_constant is cleanly divisible by the inner_constant
		if (inner_value == 0) {
			// x * 0, the result is either 0 or NULL
			// we let the arithmetic_simplification rule take care of simplifying this first
			return nullptr;
		}
		if (outer_value % inner_value != 0) {
			// not cleanly divisible
			bool is_equality = comparison->type == ExpressionType::COMPARE_EQUAL;
			bool is_inequality = comparison->type == ExpressionType::COMPARE_NOTEQUAL;
			if (is_equality || is_inequality) {
				// we know the values are not equal
				// the result will be either FALSE or NULL (if COMPARE_EQUAL)
				// or TRUE or NULL (if COMPARE_NOTEQUAL)
				return ExpressionRewriter::ConstantOrNull(move(arithmetic->children[arithmetic_child_index]),
				                                          Value::BOOLEAN(is_inequality));
			} else {
				// not cleanly divisible and we are doing > >= < <=, skip the simplification for now
				return nullptr;
			}
		}
		if (inner_value < 0) {
			// multiply by negative value, need to flip expression
			comparison->type = FlipComparisionExpression(comparison->type);
		}
		// else divide the RHS by the LHS
		// we need to do a range check on the cast even though we do a division
		// because e.g. -128 / -1 = 128, which is out of range
		auto result_value = Value::HUGEINT(outer_value / inner_value);
		if (!result_value.DefaultTryCastAs(constant_type)) {
			return ExpressionRewriter::ConstantOrNull(move(arithmetic->children[arithmetic_child_index]),
			                                          Value::BOOLEAN(false));
		}
		outer_constant->value = move(result_value);
	}
	// replace left side with x
	// first extract x from the arithmetic expression
	auto arithmetic_child = move(arithmetic->children[arithmetic_child_index]);
	// then place in the comparison
	if (comparison->left.get() == outer_constant) {
		comparison->right = move(arithmetic_child);
	} else {
		comparison->left = move(arithmetic_child);
	}
	changes_made = true;
	return nullptr;
}

} // namespace duckdb









namespace duckdb {

RegexOptimizationRule::RegexOptimizationRule(ExpressionRewriter &rewriter) : Rule(rewriter) {
	auto func = make_unique<FunctionExpressionMatcher>();
	func->function = make_unique<SpecificFunctionMatcher>("regexp_matches");
	func->policy = SetMatcher::Policy::ORDERED;
	func->matchers.push_back(make_unique<ExpressionMatcher>());
	func->matchers.push_back(make_unique<ConstantExpressionMatcher>());
	root = move(func);
}

unique_ptr<Expression> RegexOptimizationRule::Apply(LogicalOperator &op, vector<Expression *> &bindings,
                                                    bool &changes_made, bool is_root) {
	auto root = (BoundFunctionExpression *)bindings[0];
	auto constant_expr = (BoundConstantExpression *)bindings[2];
	D_ASSERT(root->children.size() == 2);

	if (constant_expr->value.IsNull()) {
		return make_unique<BoundConstantExpression>(Value(root->return_type));
	}

	// the constant_expr is a scalar expression that we have to fold
	if (!constant_expr->IsFoldable()) {
		return nullptr;
	}

	auto constant_value = ExpressionExecutor::EvaluateScalar(GetContext(), *constant_expr);
	D_ASSERT(constant_value.type() == constant_expr->return_type);
	auto &patt_str = StringValue::Get(constant_value);

	duckdb_re2::RE2 pattern(patt_str);
	if (!pattern.ok()) {
		return nullptr; // this should fail somewhere else
	}

	if (pattern.Regexp()->op() == duckdb_re2::kRegexpLiteralString ||
	    pattern.Regexp()->op() == duckdb_re2::kRegexpLiteral) {
		auto contains = make_unique<BoundFunctionExpression>(root->return_type, ContainsFun::GetFunction(),
		                                                     move(root->children), nullptr);

		contains->children[1] = make_unique<BoundConstantExpression>(Value(patt_str));
		return move(contains);
	}
	return nullptr;
}

} // namespace duckdb



namespace duckdb {

unique_ptr<BaseStatistics> StatisticsPropagator::PropagateExpression(BoundAggregateExpression &aggr,
                                                                     unique_ptr<Expression> *expr_ptr) {
	vector<unique_ptr<BaseStatistics>> stats;
	stats.reserve(aggr.children.size());
	for (auto &child : aggr.children) {
		stats.push_back(PropagateExpression(child));
	}
	if (!aggr.function.statistics) {
		return nullptr;
	}
	return aggr.function.statistics(context, aggr, aggr.bind_info.get(), stats, node_stats.get());
}

} // namespace duckdb










namespace duckdb {

template <class T>
bool GetCastType(T signed_range, LogicalType &cast_type) {
	auto range = static_cast<typename std::make_unsigned<decltype(signed_range)>::type>(signed_range);

	// Check if this range fits in a smaller type
	if (range < NumericLimits<uint8_t>::Maximum()) {
		cast_type = LogicalType::UTINYINT;
	} else if (sizeof(T) > sizeof(uint16_t) && range < NumericLimits<uint16_t>::Maximum()) {
		cast_type = LogicalType::USMALLINT;
	} else if (sizeof(T) > sizeof(uint32_t) && range < NumericLimits<uint32_t>::Maximum()) {
		cast_type = LogicalType::UINTEGER;
	} else {
		return false;
	}
	return true;
}

template <>
bool GetCastType(hugeint_t range, LogicalType &cast_type) {
	if (range < NumericLimits<uint8_t>().Maximum()) {
		cast_type = LogicalType::UTINYINT;
	} else if (range < NumericLimits<uint16_t>().Maximum()) {
		cast_type = LogicalType::USMALLINT;
	} else if (range < NumericLimits<uint32_t>().Maximum()) {
		cast_type = LogicalType::UINTEGER;
	} else if (range < NumericLimits<uint64_t>().Maximum()) {
		cast_type = LogicalType::UBIGINT;
	} else {
		return false;
	}
	return true;
}

template <class T>
unique_ptr<Expression> TemplatedCastToSmallestType(unique_ptr<Expression> expr, NumericStatistics &num_stats) {
	// Compute range
	if (num_stats.min.IsNull() || num_stats.max.IsNull()) {
		return expr;
	}

	auto signed_min_val = num_stats.min.GetValue<T>();
	auto signed_max_val = num_stats.max.GetValue<T>();
	if (signed_max_val < signed_min_val) {
		return expr;
	}

	// Compute range, cast to unsigned to prevent comparing signed with unsigned
	T signed_range;
	if (!TrySubtractOperator::Operation(signed_max_val, signed_min_val, signed_range)) {
		// overflow in subtraction: cannot do any simplification
		return expr;
	}

	// Check if this range fits in a smaller type
	LogicalType cast_type;
	if (!GetCastType(signed_range, cast_type)) {
		return expr;
	}

	// Create expression to map to a smaller range
	auto input_type = expr->return_type;
	auto minimum_expr = make_unique<BoundConstantExpression>(Value::CreateValue(signed_min_val));
	vector<unique_ptr<Expression>> arguments;
	arguments.push_back(move(expr));
	arguments.push_back(move(minimum_expr));
	auto minus_expr = make_unique<BoundFunctionExpression>(input_type, SubtractFun::GetFunction(input_type, input_type),
	                                                       move(arguments), nullptr, true);

	// Cast to smaller type
	return BoundCastExpression::AddDefaultCastToType(move(minus_expr), cast_type);
}

unique_ptr<Expression> CastToSmallestType(unique_ptr<Expression> expr, NumericStatistics &num_stats) {
	auto physical_type = expr->return_type.InternalType();
	switch (physical_type) {
	case PhysicalType::UINT8:
	case PhysicalType::INT8:
		return expr;
	case PhysicalType::UINT16:
		return TemplatedCastToSmallestType<uint16_t>(move(expr), num_stats);
	case PhysicalType::INT16:
		return TemplatedCastToSmallestType<int16_t>(move(expr), num_stats);
	case PhysicalType::UINT32:
		return TemplatedCastToSmallestType<uint32_t>(move(expr), num_stats);
	case PhysicalType::INT32:
		return TemplatedCastToSmallestType<int32_t>(move(expr), num_stats);
	case PhysicalType::UINT64:
		return TemplatedCastToSmallestType<uint64_t>(move(expr), num_stats);
	case PhysicalType::INT64:
		return TemplatedCastToSmallestType<int64_t>(move(expr), num_stats);
	case PhysicalType::INT128:
		return TemplatedCastToSmallestType<hugeint_t>(move(expr), num_stats);
	default:
		throw NotImplementedException("Unknown integer type!");
	}
}

void StatisticsPropagator::PropagateAndCompress(unique_ptr<Expression> &expr, unique_ptr<BaseStatistics> &stats) {
	stats = PropagateExpression(expr);
	if (stats) {
		if (expr->return_type.IsIntegral()) {
			expr = CastToSmallestType(move(expr), (NumericStatistics &)*stats);
		}
	}
}

} // namespace duckdb






namespace duckdb {

unique_ptr<BaseStatistics> StatisticsPropagator::PropagateExpression(BoundBetweenExpression &between,
                                                                     unique_ptr<Expression> *expr_ptr) {
	// propagate in all the children
	auto input_stats = PropagateExpression(between.input);
	auto lower_stats = PropagateExpression(between.lower);
	auto upper_stats = PropagateExpression(between.upper);
	if (!input_stats) {
		return nullptr;
	}
	auto lower_comparison = between.LowerComparisonType();
	auto upper_comparison = between.UpperComparisonType();
	// propagate the comparisons
	auto lower_prune = FilterPropagateResult::NO_PRUNING_POSSIBLE;
	auto upper_prune = FilterPropagateResult::NO_PRUNING_POSSIBLE;
	if (lower_stats) {
		lower_prune = PropagateComparison(*input_stats, *lower_stats, lower_comparison);
	}
	if (upper_stats) {
		upper_prune = PropagateComparison(*input_stats, *upper_stats, upper_comparison);
	}
	if (lower_prune == FilterPropagateResult::FILTER_ALWAYS_TRUE &&
	    upper_prune == FilterPropagateResult::FILTER_ALWAYS_TRUE) {
		// both filters are always true: replace the between expression with a constant true
		*expr_ptr = make_unique<BoundConstantExpression>(Value::BOOLEAN(true));
	} else if (lower_prune == FilterPropagateResult::FILTER_ALWAYS_FALSE ||
	           upper_prune == FilterPropagateResult::FILTER_ALWAYS_FALSE) {
		// either one of the filters is always false: replace the between expression with a constant false
		*expr_ptr = make_unique<BoundConstantExpression>(Value::BOOLEAN(false));
	} else if (lower_prune == FilterPropagateResult::FILTER_FALSE_OR_NULL ||
	           upper_prune == FilterPropagateResult::FILTER_FALSE_OR_NULL) {
		// either one of the filters is false or null: replace with a constant or null (false)
		vector<unique_ptr<Expression>> children;
		children.push_back(move(between.input));
		children.push_back(move(between.lower));
		children.push_back(move(between.upper));
		*expr_ptr = ExpressionRewriter::ConstantOrNull(move(children), Value::BOOLEAN(false));
	} else if (lower_prune == FilterPropagateResult::FILTER_TRUE_OR_NULL &&
	           upper_prune == FilterPropagateResult::FILTER_TRUE_OR_NULL) {
		// both filters are true or null: replace with a true or null
		vector<unique_ptr<Expression>> children;
		children.push_back(move(between.input));
		children.push_back(move(between.lower));
		children.push_back(move(between.upper));
		*expr_ptr = ExpressionRewriter::ConstantOrNull(move(children), Value::BOOLEAN(true));
	} else if (lower_prune == FilterPropagateResult::FILTER_ALWAYS_TRUE) {
		// lower filter is always true: replace with upper comparison
		*expr_ptr = make_unique<BoundComparisonExpression>(upper_comparison, move(between.input), move(between.upper));
	} else if (upper_prune == FilterPropagateResult::FILTER_ALWAYS_TRUE) {
		// upper filter is always true: replace with lower comparison
		*expr_ptr = make_unique<BoundComparisonExpression>(lower_comparison, move(between.input), move(between.lower));
	}
	return nullptr;
}

} // namespace duckdb



namespace duckdb {

unique_ptr<BaseStatistics> StatisticsPropagator::PropagateExpression(BoundCaseExpression &bound_case,
                                                                     unique_ptr<Expression> *expr_ptr) {
	// propagate in all the children
	auto result_stats = PropagateExpression(bound_case.else_expr);
	for (auto &case_check : bound_case.case_checks) {
		PropagateExpression(case_check.when_expr);
		auto then_stats = PropagateExpression(case_check.then_expr);
		if (!then_stats) {
			result_stats.reset();
		} else if (result_stats) {
			result_stats->Merge(*then_stats);
		}
	}
	return result_stats;
}

} // namespace duckdb




namespace duckdb {

static unique_ptr<BaseStatistics> StatisticsOperationsNumericNumericCast(const BaseStatistics *input_p,
                                                                         const LogicalType &target) {
	auto &input = (NumericStatistics &)*input_p;

	Value min = input.min, max = input.max;
	if (!min.DefaultTryCastAs(target) || !max.DefaultTryCastAs(target)) {
		// overflow in cast: bailout
		return nullptr;
	}
	auto stats = make_unique<NumericStatistics>(target, move(min), move(max), input.stats_type);
	stats->CopyBase(*input_p);
	return move(stats);
}

static unique_ptr<BaseStatistics> StatisticsNumericCastSwitch(const BaseStatistics *input, const LogicalType &target) {
	switch (target.InternalType()) {
	case PhysicalType::INT8:
	case PhysicalType::INT16:
	case PhysicalType::INT32:
	case PhysicalType::INT64:
	case PhysicalType::INT128:
	case PhysicalType::FLOAT:
	case PhysicalType::DOUBLE:
		return StatisticsOperationsNumericNumericCast(input, target);
	default:
		return nullptr;
	}
}

unique_ptr<BaseStatistics> StatisticsPropagator::PropagateExpression(BoundCastExpression &cast,
                                                                     unique_ptr<Expression> *expr_ptr) {
	auto child_stats = PropagateExpression(cast.child);
	if (!child_stats) {
		return nullptr;
	}
	unique_ptr<BaseStatistics> result_stats;
	switch (cast.child->return_type.InternalType()) {
	case PhysicalType::INT8:
	case PhysicalType::INT16:
	case PhysicalType::INT32:
	case PhysicalType::INT64:
	case PhysicalType::INT128:
	case PhysicalType::FLOAT:
	case PhysicalType::DOUBLE:
		result_stats = StatisticsNumericCastSwitch(child_stats.get(), cast.return_type);
		break;
	default:
		return nullptr;
	}
	if (cast.try_cast && result_stats) {
		result_stats->validity_stats = make_unique<ValidityStatistics>(true, true);
	}
	return result_stats;
}

} // namespace duckdb



namespace duckdb {

unique_ptr<BaseStatistics> StatisticsPropagator::PropagateExpression(BoundColumnRefExpression &colref,
                                                                     unique_ptr<Expression> *expr_ptr) {
	auto stats = statistics_map.find(colref.binding);
	if (stats == statistics_map.end()) {
		return nullptr;
	}
	return stats->second->Copy();
}

} // namespace duckdb






namespace duckdb {

FilterPropagateResult StatisticsPropagator::PropagateComparison(BaseStatistics &left, BaseStatistics &right,
                                                                ExpressionType comparison) {
	// only handle numerics for now
	switch (left.type.InternalType()) {
	case PhysicalType::BOOL:
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
		break;
	default:
		return FilterPropagateResult::NO_PRUNING_POSSIBLE;
	}
	auto &lstats = (NumericStatistics &)left;
	auto &rstats = (NumericStatistics &)right;
	if (lstats.min.IsNull() || lstats.max.IsNull() || rstats.min.IsNull() || rstats.max.IsNull()) {
		// no stats available: nothing to prune
		return FilterPropagateResult::NO_PRUNING_POSSIBLE;
	}
	// the result of the propagation depend on whether or not either side has null values
	// if there are null values present, we cannot say whether or not
	bool has_null = lstats.CanHaveNull() || rstats.CanHaveNull();
	switch (comparison) {
	case ExpressionType::COMPARE_EQUAL:
		// l = r, if l.min > r.max or r.min > l.max equality is not possible
		if (lstats.min > rstats.max || rstats.min > lstats.max) {
			return has_null ? FilterPropagateResult::FILTER_FALSE_OR_NULL : FilterPropagateResult::FILTER_ALWAYS_FALSE;
		} else {
			return FilterPropagateResult::NO_PRUNING_POSSIBLE;
		}
	case ExpressionType::COMPARE_GREATERTHAN:
		// l > r
		if (lstats.min > rstats.max) {
			// if l.min > r.max, it is always true ONLY if neither side contains nulls
			return has_null ? FilterPropagateResult::FILTER_TRUE_OR_NULL : FilterPropagateResult::FILTER_ALWAYS_TRUE;
		}
		// if r.min is bigger or equal to l.max, the filter is always false
		if (rstats.min >= lstats.max) {
			return has_null ? FilterPropagateResult::FILTER_FALSE_OR_NULL : FilterPropagateResult::FILTER_ALWAYS_FALSE;
		}
		return FilterPropagateResult::NO_PRUNING_POSSIBLE;
	case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
		// l >= r
		if (lstats.min >= rstats.max) {
			// if l.min >= r.max, it is always true ONLY if neither side contains nulls
			return has_null ? FilterPropagateResult::FILTER_TRUE_OR_NULL : FilterPropagateResult::FILTER_ALWAYS_TRUE;
		}
		// if r.min > l.max, the filter is always false
		if (rstats.min > lstats.max) {
			return has_null ? FilterPropagateResult::FILTER_FALSE_OR_NULL : FilterPropagateResult::FILTER_ALWAYS_FALSE;
		}
		return FilterPropagateResult::NO_PRUNING_POSSIBLE;
	case ExpressionType::COMPARE_LESSTHAN:
		// l < r
		if (lstats.max < rstats.min) {
			// if l.max < r.min, it is always true ONLY if neither side contains nulls
			return has_null ? FilterPropagateResult::FILTER_TRUE_OR_NULL : FilterPropagateResult::FILTER_ALWAYS_TRUE;
		}
		// if l.min >= rstats.max, the filter is always false
		if (lstats.min >= rstats.max) {
			return has_null ? FilterPropagateResult::FILTER_FALSE_OR_NULL : FilterPropagateResult::FILTER_ALWAYS_FALSE;
		}
		return FilterPropagateResult::NO_PRUNING_POSSIBLE;
	case ExpressionType::COMPARE_LESSTHANOREQUALTO:
		// l <= r
		if (lstats.max <= rstats.min) {
			// if l.max <= r.min, it is always true ONLY if neither side contains nulls
			return has_null ? FilterPropagateResult::FILTER_TRUE_OR_NULL : FilterPropagateResult::FILTER_ALWAYS_TRUE;
		}
		// if l.min > rstats.max, the filter is always false
		if (lstats.min > rstats.max) {
			return has_null ? FilterPropagateResult::FILTER_FALSE_OR_NULL : FilterPropagateResult::FILTER_ALWAYS_FALSE;
		}
		return FilterPropagateResult::NO_PRUNING_POSSIBLE;
	default:
		return FilterPropagateResult::NO_PRUNING_POSSIBLE;
	}
}

unique_ptr<BaseStatistics> StatisticsPropagator::PropagateExpression(BoundComparisonExpression &expr,
                                                                     unique_ptr<Expression> *expr_ptr) {
	auto left_stats = PropagateExpression(expr.left);
	auto right_stats = PropagateExpression(expr.right);
	if (!left_stats || !right_stats) {
		return nullptr;
	}
	// propagate the statistics of the comparison operator
	auto propagate_result = PropagateComparison(*left_stats, *right_stats, expr.type);
	switch (propagate_result) {
	case FilterPropagateResult::FILTER_ALWAYS_TRUE:
		*expr_ptr = make_unique<BoundConstantExpression>(Value::BOOLEAN(true));
		return PropagateExpression(*expr_ptr);
	case FilterPropagateResult::FILTER_ALWAYS_FALSE:
		*expr_ptr = make_unique<BoundConstantExpression>(Value::BOOLEAN(false));
		return PropagateExpression(*expr_ptr);
	case FilterPropagateResult::FILTER_TRUE_OR_NULL: {
		vector<unique_ptr<Expression>> children;
		children.push_back(move(expr.left));
		children.push_back(move(expr.right));
		*expr_ptr = ExpressionRewriter::ConstantOrNull(move(children), Value::BOOLEAN(true));
		return nullptr;
	}
	case FilterPropagateResult::FILTER_FALSE_OR_NULL: {
		vector<unique_ptr<Expression>> children;
		children.push_back(move(expr.left));
		children.push_back(move(expr.right));
		*expr_ptr = ExpressionRewriter::ConstantOrNull(move(children), Value::BOOLEAN(false));
		return nullptr;
	}
	default:
		// FIXME: we can propagate nulls here, i.e. this expression will have nulls only if left and right has nulls
		return nullptr;
	}
}

} // namespace duckdb








namespace duckdb {

unique_ptr<BaseStatistics> StatisticsPropagator::PropagateExpression(BoundConjunctionExpression &expr,
                                                                     unique_ptr<Expression> *expr_ptr) {
	auto is_and = expr.type == ExpressionType::CONJUNCTION_AND;
	for (idx_t expr_idx = 0; expr_idx < expr.children.size(); expr_idx++) {
		auto &child = expr.children[expr_idx];
		auto stats = PropagateExpression(child);
		if (!child->IsFoldable()) {
			continue;
		}
		// we have a constant in a conjunction
		// we (1) either prune the child
		// or (2) replace the entire conjunction with a constant
		auto constant = ExpressionExecutor::EvaluateScalar(context, *child);
		if (constant.IsNull()) {
			continue;
		}
		auto b = BooleanValue::Get(constant);
		bool prune_child = false;
		bool constant_value = true;
		if (b) {
			// true
			if (is_and) {
				// true in and: prune child
				prune_child = true;
			} else {
				// true in OR: replace with TRUE
				constant_value = true;
			}
		} else {
			// false
			if (is_and) {
				// false in AND: replace with FALSE
				constant_value = false;
			} else {
				// false in OR: prune child
				prune_child = true;
			}
		}
		if (prune_child) {
			expr.children.erase(expr.children.begin() + expr_idx);
			expr_idx--;
			continue;
		}
		*expr_ptr = make_unique<BoundConstantExpression>(Value::BOOLEAN(constant_value));
		return PropagateExpression(*expr_ptr);
	}
	if (expr.children.empty()) {
		// if there are no children left, replace the conjunction with TRUE (for AND) or FALSE (for OR)
		*expr_ptr = make_unique<BoundConstantExpression>(Value::BOOLEAN(is_and));
		return PropagateExpression(*expr_ptr);
	} else if (expr.children.size() == 1) {
		// if there is one child left, replace the conjunction with that one child
		*expr_ptr = move(expr.children[0]);
	}
	return nullptr;
}

} // namespace duckdb








namespace duckdb {

void UpdateDistinctStats(BaseStatistics &distinct_stats, const Value &input) {
	Vector v(input);
	auto &d_stats = (DistinctStatistics &)distinct_stats;
	d_stats.Update(v, 1);
}

unique_ptr<BaseStatistics> StatisticsPropagator::StatisticsFromValue(const Value &input) {
	switch (input.type().InternalType()) {
	case PhysicalType::BOOL:
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
	case PhysicalType::DOUBLE: {
		auto result = make_unique<NumericStatistics>(input.type(), input, input, StatisticsType::GLOBAL_STATS);
		result->validity_stats = make_unique<ValidityStatistics>(input.IsNull(), !input.IsNull());
		UpdateDistinctStats(*result->distinct_stats, input);
		return move(result);
	}
	case PhysicalType::VARCHAR: {
		auto result = make_unique<StringStatistics>(input.type(), StatisticsType::GLOBAL_STATS);
		result->validity_stats = make_unique<ValidityStatistics>(input.IsNull(), !input.IsNull());
		UpdateDistinctStats(*result->distinct_stats, input);
		if (!input.IsNull()) {
			auto &string_value = StringValue::Get(input);
			result->Update(string_t(string_value));
		}
		return move(result);
	}
	case PhysicalType::STRUCT: {
		auto result = make_unique<StructStatistics>(input.type());
		result->validity_stats = make_unique<ValidityStatistics>(input.IsNull(), !input.IsNull());
		if (input.IsNull()) {
			for (auto &child_stat : result->child_stats) {
				child_stat.reset();
			}
		} else {
			auto &struct_children = StructValue::GetChildren(input);
			D_ASSERT(result->child_stats.size() == struct_children.size());
			for (idx_t i = 0; i < result->child_stats.size(); i++) {
				result->child_stats[i] = StatisticsFromValue(struct_children[i]);
			}
		}
		return move(result);
	}
	case PhysicalType::LIST: {
		auto result = make_unique<ListStatistics>(input.type());
		result->validity_stats = make_unique<ValidityStatistics>(input.IsNull(), !input.IsNull());
		if (input.IsNull()) {
			result->child_stats.reset();
		} else {
			auto &list_children = ListValue::GetChildren(input);
			for (auto &child_element : list_children) {
				auto child_element_stats = StatisticsFromValue(child_element);
				if (child_element_stats) {
					result->child_stats->Merge(*child_element_stats);
				} else {
					result->child_stats.reset();
				}
			}
		}
		return move(result);
	}
	default:
		return nullptr;
	}
}

unique_ptr<BaseStatistics> StatisticsPropagator::PropagateExpression(BoundConstantExpression &constant,
                                                                     unique_ptr<Expression> *expr_ptr) {
	return StatisticsFromValue(constant.value);
}

} // namespace duckdb



namespace duckdb {

unique_ptr<BaseStatistics> StatisticsPropagator::PropagateExpression(BoundFunctionExpression &func,
                                                                     unique_ptr<Expression> *expr_ptr) {
	vector<unique_ptr<BaseStatistics>> stats;
	stats.reserve(func.children.size());
	for (idx_t i = 0; i < func.children.size(); i++) {
		stats.push_back(PropagateExpression(func.children[i]));
	}
	if (!func.function.statistics) {
		return nullptr;
	}
	FunctionStatisticsInput input(func, func.bind_info.get(), stats, expr_ptr);
	return func.function.statistics(context, input);
}

} // namespace duckdb




namespace duckdb {

unique_ptr<BaseStatistics> StatisticsPropagator::PropagateExpression(BoundOperatorExpression &expr,
                                                                     unique_ptr<Expression> *expr_ptr) {
	bool all_have_stats = true;
	vector<unique_ptr<BaseStatistics>> child_stats;
	child_stats.reserve(expr.children.size());
	for (auto &child : expr.children) {
		auto stats = PropagateExpression(child);
		if (!stats) {
			all_have_stats = false;
		}
		child_stats.push_back(move(stats));
	}
	if (!all_have_stats) {
		return nullptr;
	}
	switch (expr.type) {
	case ExpressionType::OPERATOR_COALESCE:
		// COALESCE, merge stats of all children
		for (idx_t i = 0; i < expr.children.size(); i++) {
			D_ASSERT(child_stats[i]);
			if (!child_stats[i]->CanHaveNoNull()) {
				// this child is always NULL, we can remove it from the coalesce
				// UNLESS there is only one node remaining
				if (expr.children.size() > 1) {
					expr.children.erase(expr.children.begin() + i);
					child_stats.erase(child_stats.begin() + i);
					i--;
				}
			} else if (!child_stats[i]->CanHaveNull()) {
				// coalesce child cannot have NULL entries
				// this is the last coalesce node that influences the result
				// we can erase any children after this node
				if (i + 1 < expr.children.size()) {
					expr.children.erase(expr.children.begin() + i + 1, expr.children.end());
					child_stats.erase(child_stats.begin() + i + 1, child_stats.end());
				}
				break;
			}
		}
		D_ASSERT(!expr.children.empty());
		D_ASSERT(expr.children.size() == child_stats.size());
		if (expr.children.size() == 1) {
			// coalesce of one entry: simply return that entry
			*expr_ptr = move(expr.children[0]);
		} else {
			// coalesce of multiple entries
			// merge the stats
			for (idx_t i = 1; i < expr.children.size(); i++) {
				child_stats[0]->Merge(*child_stats[i]);
			}
		}
		return move(child_stats[0]);
	case ExpressionType::OPERATOR_IS_NULL:
		if (!child_stats[0]->CanHaveNull()) {
			// child has no null values: x IS NULL will always be false
			*expr_ptr = make_unique<BoundConstantExpression>(Value::BOOLEAN(false));
			return PropagateExpression(*expr_ptr);
		}
		return nullptr;
	case ExpressionType::OPERATOR_IS_NOT_NULL:
		if (!child_stats[0]->CanHaveNull()) {
			// child has no null values: x IS NOT NULL will always be true
			*expr_ptr = make_unique<BoundConstantExpression>(Value::BOOLEAN(true));
			return PropagateExpression(*expr_ptr);
		}
		return nullptr;
	default:
		return nullptr;
	}
}

} // namespace duckdb




namespace duckdb {

unique_ptr<NodeStatistics> StatisticsPropagator::PropagateStatistics(LogicalAggregate &aggr,
                                                                     unique_ptr<LogicalOperator> *node_ptr) {
	// first propagate statistics in the child node
	node_stats = PropagateStatistics(aggr.children[0]);

	// handle the groups: simply propagate statistics and assign the stats to the group binding
	aggr.group_stats.resize(aggr.groups.size());
	for (idx_t group_idx = 0; group_idx < aggr.groups.size(); group_idx++) {
		auto stats = PropagateExpression(aggr.groups[group_idx]);
		aggr.group_stats[group_idx] = stats ? stats->Copy() : nullptr;
		if (!stats) {
			continue;
		}
		if (aggr.grouping_sets.size() > 1) {
			// aggregates with multiple grouping sets can introduce NULL values to certain groups
			// FIXME: actually figure out WHICH groups can have null values introduced
			stats->validity_stats = make_unique<ValidityStatistics>(true, true);
			continue;
		}
		ColumnBinding group_binding(aggr.group_index, group_idx);
		statistics_map[group_binding] = move(stats);
	}
	// propagate statistics in the aggregates
	for (idx_t aggregate_idx = 0; aggregate_idx < aggr.expressions.size(); aggregate_idx++) {
		auto stats = PropagateExpression(aggr.expressions[aggregate_idx]);
		if (!stats) {
			continue;
		}
		ColumnBinding aggregate_binding(aggr.aggregate_index, aggregate_idx);
		statistics_map[aggregate_binding] = move(stats);
	}
	// the max cardinality of an aggregate is the max cardinality of the input (i.e. when every row is a unique group)
	return move(node_stats);
}

} // namespace duckdb



namespace duckdb {

unique_ptr<NodeStatistics> StatisticsPropagator::PropagateStatistics(LogicalCrossProduct &cp,
                                                                     unique_ptr<LogicalOperator> *node_ptr) {
	// first propagate statistics in the child node
	auto left_stats = PropagateStatistics(cp.children[0]);
	auto right_stats = PropagateStatistics(cp.children[1]);
	if (!left_stats || !right_stats) {
		return nullptr;
	}
	MultiplyCardinalities(left_stats, *right_stats);
	return left_stats;
}

} // namespace duckdb









namespace duckdb {

static bool IsCompareDistinct(ExpressionType type) {
	return type == ExpressionType::COMPARE_DISTINCT_FROM || type == ExpressionType::COMPARE_NOT_DISTINCT_FROM;
}

bool StatisticsPropagator::ExpressionIsConstant(Expression &expr, const Value &val) {
	if (expr.GetExpressionClass() != ExpressionClass::BOUND_CONSTANT) {
		return false;
	}
	auto &bound_constant = (BoundConstantExpression &)expr;
	D_ASSERT(bound_constant.value.type() == val.type());
	return Value::NotDistinctFrom(bound_constant.value, val);
}

bool StatisticsPropagator::ExpressionIsConstantOrNull(Expression &expr, const Value &val) {
	if (expr.GetExpressionClass() != ExpressionClass::BOUND_FUNCTION) {
		return false;
	}
	auto &bound_function = (BoundFunctionExpression &)expr;
	return ConstantOrNull::IsConstantOrNull(bound_function, val);
}

void StatisticsPropagator::SetStatisticsNotNull(ColumnBinding binding) {
	auto entry = statistics_map.find(binding);
	if (entry == statistics_map.end()) {
		return;
	}
	entry->second->validity_stats = make_unique<ValidityStatistics>(false);
}

void StatisticsPropagator::UpdateFilterStatistics(BaseStatistics &stats, ExpressionType comparison_type,
                                                  const Value &constant) {
	// regular comparisons removes all null values
	if (!IsCompareDistinct(comparison_type)) {
		stats.validity_stats = make_unique<ValidityStatistics>(false);
	}
	if (!stats.type.IsNumeric()) {
		// don't handle non-numeric columns here (yet)
		return;
	}
	auto &numeric_stats = (NumericStatistics &)stats;
	if (numeric_stats.min.IsNull() || numeric_stats.max.IsNull()) {
		// no stats available: skip this
		return;
	}
	switch (comparison_type) {
	case ExpressionType::COMPARE_LESSTHAN:
	case ExpressionType::COMPARE_LESSTHANOREQUALTO:
		// X < constant OR X <= constant
		// max becomes the constant
		numeric_stats.max = constant;
		break;
	case ExpressionType::COMPARE_GREATERTHAN:
	case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
		// X > constant OR X >= constant
		// min becomes the constant
		numeric_stats.min = constant;
		break;
	case ExpressionType::COMPARE_EQUAL:
		// X = constant
		// both min and max become the constant
		numeric_stats.min = constant;
		numeric_stats.max = constant;
		break;
	default:
		break;
	}
}

void StatisticsPropagator::UpdateFilterStatistics(BaseStatistics &lstats, BaseStatistics &rstats,
                                                  ExpressionType comparison_type) {
	// regular comparisons removes all null values
	if (!IsCompareDistinct(comparison_type)) {
		lstats.validity_stats = make_unique<ValidityStatistics>(false);
		rstats.validity_stats = make_unique<ValidityStatistics>(false);
	}
	D_ASSERT(lstats.type == rstats.type);
	if (!lstats.type.IsNumeric()) {
		// don't handle non-numeric columns here (yet)
		return;
	}
	auto &left_stats = (NumericStatistics &)lstats;
	auto &right_stats = (NumericStatistics &)rstats;
	if (left_stats.min.IsNull() || left_stats.max.IsNull() || right_stats.min.IsNull() || right_stats.max.IsNull()) {
		// no stats available: skip this
		return;
	}
	switch (comparison_type) {
	case ExpressionType::COMPARE_LESSTHAN:
	case ExpressionType::COMPARE_LESSTHANOREQUALTO:
		// LEFT < RIGHT OR LEFT <= RIGHT
		// we know that every value of left is smaller (or equal to) every value in right
		// i.e. if we have left = [-50, 250] and right = [-100, 100]

		// we know that left.max is AT MOST equal to right.max
		// because any value in left that is BIGGER than right.max will not pass the filter
		if (left_stats.max > right_stats.max) {
			left_stats.max = right_stats.max;
		}

		// we also know that right.min is AT MOST equal to left.min
		// because any value in right that is SMALLER than left.min will not pass the filter
		if (right_stats.min < left_stats.min) {
			right_stats.min = left_stats.min;
		}
		// so in our example, the bounds get updated as follows:
		// left: [-50, 100], right: [-50, 100]
		break;
	case ExpressionType::COMPARE_GREATERTHAN:
	case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
		// LEFT > RIGHT OR LEFT >= RIGHT
		// we know that every value of left is bigger (or equal to) every value in right
		// this is essentially the inverse of the less than (or equal to) scenario
		if (right_stats.max > left_stats.max) {
			right_stats.max = left_stats.max;
		}
		if (left_stats.min < right_stats.min) {
			left_stats.min = right_stats.min;
		}
		break;
	case ExpressionType::COMPARE_EQUAL:
	case ExpressionType::COMPARE_NOT_DISTINCT_FROM:
		// LEFT = RIGHT
		// only the tightest bounds pass
		// so if we have e.g. left = [-50, 250] and right = [-100, 100]
		// the tighest bounds are [-50, 100]
		// select the highest min
		if (left_stats.min > right_stats.min) {
			right_stats.min = left_stats.min;
		} else {
			left_stats.min = right_stats.min;
		}
		// select the lowest max
		if (left_stats.max < right_stats.max) {
			right_stats.max = left_stats.max;
		} else {
			left_stats.max = right_stats.max;
		}
		break;
	default:
		break;
	}
}

void StatisticsPropagator::UpdateFilterStatistics(Expression &left, Expression &right, ExpressionType comparison_type) {
	// first check if either side is a bound column ref
	// any column ref involved in a comparison will not be null after the comparison
	bool compare_distinct = IsCompareDistinct(comparison_type);
	if (!compare_distinct && left.type == ExpressionType::BOUND_COLUMN_REF) {
		SetStatisticsNotNull(((BoundColumnRefExpression &)left).binding);
	}
	if (!compare_distinct && right.type == ExpressionType::BOUND_COLUMN_REF) {
		SetStatisticsNotNull(((BoundColumnRefExpression &)right).binding);
	}
	// check if this is a comparison between a constant and a column ref
	BoundConstantExpression *constant = nullptr;
	BoundColumnRefExpression *columnref = nullptr;
	if (left.type == ExpressionType::VALUE_CONSTANT && right.type == ExpressionType::BOUND_COLUMN_REF) {
		constant = (BoundConstantExpression *)&left;
		columnref = (BoundColumnRefExpression *)&right;
		comparison_type = FlipComparisionExpression(comparison_type);
	} else if (left.type == ExpressionType::BOUND_COLUMN_REF && right.type == ExpressionType::VALUE_CONSTANT) {
		columnref = (BoundColumnRefExpression *)&left;
		constant = (BoundConstantExpression *)&right;
	} else if (left.type == ExpressionType::BOUND_COLUMN_REF && right.type == ExpressionType::BOUND_COLUMN_REF) {
		// comparison between two column refs
		auto &left_column_ref = (BoundColumnRefExpression &)left;
		auto &right_column_ref = (BoundColumnRefExpression &)right;
		auto lentry = statistics_map.find(left_column_ref.binding);
		auto rentry = statistics_map.find(right_column_ref.binding);
		if (lentry == statistics_map.end() || rentry == statistics_map.end()) {
			return;
		}
		UpdateFilterStatistics(*lentry->second, *rentry->second, comparison_type);
	} else {
		// unsupported filter
		return;
	}
	if (constant && columnref) {
		// comparison between columnref
		auto entry = statistics_map.find(columnref->binding);
		if (entry == statistics_map.end()) {
			return;
		}
		UpdateFilterStatistics(*entry->second, comparison_type, constant->value);
	}
}

void StatisticsPropagator::UpdateFilterStatistics(Expression &condition) {
	// in filters, we check for constant comparisons with bound columns
	// if we find a comparison in the form of e.g. "i=3", we can update our statistics for that column
	switch (condition.GetExpressionClass()) {
	case ExpressionClass::BOUND_BETWEEN: {
		auto &between = (BoundBetweenExpression &)condition;
		UpdateFilterStatistics(*between.input, *between.lower, between.LowerComparisonType());
		UpdateFilterStatistics(*between.input, *between.upper, between.UpperComparisonType());
		break;
	}
	case ExpressionClass::BOUND_COMPARISON: {
		auto &comparison = (BoundComparisonExpression &)condition;
		UpdateFilterStatistics(*comparison.left, *comparison.right, comparison.type);
		break;
	}
	default:
		break;
	}
}

unique_ptr<NodeStatistics> StatisticsPropagator::PropagateStatistics(LogicalFilter &filter,
                                                                     unique_ptr<LogicalOperator> *node_ptr) {
	// first propagate to the child
	node_stats = PropagateStatistics(filter.children[0]);
	if (filter.children[0]->type == LogicalOperatorType::LOGICAL_EMPTY_RESULT) {
		ReplaceWithEmptyResult(*node_ptr);
		return make_unique<NodeStatistics>(0, 0);
	}

	// then propagate to each of the expressions
	for (idx_t i = 0; i < filter.expressions.size(); i++) {
		auto &condition = filter.expressions[i];
		PropagateExpression(condition);

		if (ExpressionIsConstant(*condition, Value::BOOLEAN(true))) {
			// filter is always true; it is useless to execute it
			// erase this condition
			filter.expressions.erase(filter.expressions.begin() + i);
			i--;
			if (filter.expressions.empty()) {
				// all conditions have been erased: remove the entire filter
				*node_ptr = move(filter.children[0]);
				break;
			}
		} else if (ExpressionIsConstant(*condition, Value::BOOLEAN(false)) ||
		           ExpressionIsConstantOrNull(*condition, Value::BOOLEAN(false))) {
			// filter is always false or null; this entire filter should be replaced by an empty result block
			ReplaceWithEmptyResult(*node_ptr);
			return make_unique<NodeStatistics>(0, 0);
		} else {
			// cannot prune this filter: propagate statistics from the filter
			UpdateFilterStatistics(*condition);
		}
	}
	// the max cardinality of a filter is the cardinality of the input (i.e. no tuples get filtered)
	return move(node_stats);
}

} // namespace duckdb






namespace duckdb {

FilterPropagateResult StatisticsPropagator::PropagateTableFilter(BaseStatistics &stats, TableFilter &filter) {
	return filter.CheckStatistics(stats);
}

void StatisticsPropagator::UpdateFilterStatistics(BaseStatistics &input, TableFilter &filter) {
	// FIXME: update stats...
	switch (filter.filter_type) {
	case TableFilterType::CONJUNCTION_AND: {
		auto &conjunction_and = (ConjunctionAndFilter &)filter;
		for (auto &child_filter : conjunction_and.child_filters) {
			UpdateFilterStatistics(input, *child_filter);
		}
		break;
	}
	case TableFilterType::CONSTANT_COMPARISON: {
		auto &constant_filter = (ConstantFilter &)filter;
		UpdateFilterStatistics(input, constant_filter.comparison_type, constant_filter.constant);
		break;
	}
	default:
		break;
	}
}

unique_ptr<NodeStatistics> StatisticsPropagator::PropagateStatistics(LogicalGet &get,
                                                                     unique_ptr<LogicalOperator> *node_ptr) {
	if (get.function.cardinality) {
		node_stats = get.function.cardinality(context, get.bind_data.get());
	}
	if (!get.function.statistics) {
		// no column statistics to get
		return move(node_stats);
	}
	for (idx_t i = 0; i < get.column_ids.size(); i++) {
		auto stats = get.function.statistics(context, get.bind_data.get(), get.column_ids[i]);
		if (stats) {
			ColumnBinding binding(get.table_index, i);
			statistics_map.insert(make_pair(binding, move(stats)));
		}
	}
	// push table filters into the statistics
	vector<idx_t> column_indexes;
	column_indexes.reserve(get.table_filters.filters.size());
	for (auto &kv : get.table_filters.filters) {
		column_indexes.push_back(kv.first);
	}

	for (auto &table_filter_column : column_indexes) {
		idx_t column_index;
		for (column_index = 0; column_index < get.column_ids.size(); column_index++) {
			if (get.column_ids[column_index] == table_filter_column) {
				break;
			}
		}
		D_ASSERT(column_index < get.column_ids.size());
		D_ASSERT(get.column_ids[column_index] == table_filter_column);

		// find the stats
		ColumnBinding stats_binding(get.table_index, column_index);
		auto entry = statistics_map.find(stats_binding);
		if (entry == statistics_map.end()) {
			// no stats for this entry
			continue;
		}
		auto &stats = *entry->second;

		// fetch the table filter
		D_ASSERT(get.table_filters.filters.count(table_filter_column) > 0);
		auto &filter = get.table_filters.filters[table_filter_column];
		auto propagate_result = PropagateTableFilter(stats, *filter);
		switch (propagate_result) {
		case FilterPropagateResult::FILTER_ALWAYS_TRUE:
			// filter is always true; it is useless to execute it
			// erase this condition
			get.table_filters.filters.erase(table_filter_column);
			break;
		case FilterPropagateResult::FILTER_FALSE_OR_NULL:
		case FilterPropagateResult::FILTER_ALWAYS_FALSE:
			// filter is always false; this entire filter should be replaced by an empty result block
			ReplaceWithEmptyResult(*node_ptr);
			return make_unique<NodeStatistics>(0, 0);
		default:
			// general case: filter can be true or false, update this columns' statistics
			UpdateFilterStatistics(stats, *filter);
			break;
		}
	}
	return move(node_stats);
}

} // namespace duckdb










namespace duckdb {

void StatisticsPropagator::PropagateStatistics(LogicalComparisonJoin &join, unique_ptr<LogicalOperator> *node_ptr) {
	for (idx_t i = 0; i < join.conditions.size(); i++) {
		auto &condition = join.conditions[i];
		auto stats_left = PropagateExpression(condition.left);
		auto stats_right = PropagateExpression(condition.right);
		if (stats_left && stats_right) {
			if ((condition.comparison == ExpressionType::COMPARE_DISTINCT_FROM ||
			     condition.comparison == ExpressionType::COMPARE_NOT_DISTINCT_FROM) &&
			    stats_left->CanHaveNull() && stats_right->CanHaveNull()) {
				// null values are equal in this join, and both sides can have null values
				// nothing to do here
				continue;
			}
			auto prune_result = PropagateComparison(*stats_left, *stats_right, condition.comparison);
			// Add stats to logical_join for perfect hash join
			join.join_stats.push_back(move(stats_left));
			join.join_stats.push_back(move(stats_right));
			switch (prune_result) {
			case FilterPropagateResult::FILTER_FALSE_OR_NULL:
			case FilterPropagateResult::FILTER_ALWAYS_FALSE:
				// filter is always false or null, none of the join conditions matter
				switch (join.join_type) {
				case JoinType::SEMI:
				case JoinType::INNER:
					// semi or inner join on false; entire node can be pruned
					ReplaceWithEmptyResult(*node_ptr);
					return;
				case JoinType::ANTI: {
					// when the right child has data, return the left child
					// when the right child has no data, return an empty set
					auto limit = make_unique<LogicalLimit>(1, 0, nullptr, nullptr);
					limit->AddChild(move(join.children[1]));
					auto cross_product = LogicalCrossProduct::Create(move(join.children[0]), move(limit));
					*node_ptr = move(cross_product);
					return;
				}
				case JoinType::LEFT:
					// anti/left outer join: replace right side with empty node
					ReplaceWithEmptyResult(join.children[1]);
					return;
				case JoinType::RIGHT:
					// right outer join: replace left side with empty node
					ReplaceWithEmptyResult(join.children[0]);
					return;
				default:
					// other join types: can't do much meaningful with this information
					// full outer join requires both sides anyway; we can skip the execution of the actual join, but eh
					// mark/single join requires knowing if the rhs has null values or not
					break;
				}
				break;
			case FilterPropagateResult::FILTER_ALWAYS_TRUE:
				// filter is always true
				if (join.conditions.size() > 1) {
					// there are multiple conditions: erase this condition
					join.conditions.erase(join.conditions.begin() + i);
					// remove the corresponding statistics
					join.join_stats.clear();
					i--;
					continue;
				} else {
					// this is the only condition and it is always true: all conditions are true
					switch (join.join_type) {
					case JoinType::SEMI: {
						// when the right child has data, return the left child
						// when the right child has no data, return an empty set
						auto limit = make_unique<LogicalLimit>(1, 0, nullptr, nullptr);
						limit->AddChild(move(join.children[1]));
						auto cross_product = LogicalCrossProduct::Create(move(join.children[0]), move(limit));
						*node_ptr = move(cross_product);
						return;
					}
					case JoinType::INNER:
					case JoinType::LEFT:
					case JoinType::RIGHT:
					case JoinType::OUTER: {
						// inner/left/right/full outer join, replace with cross product
						// since the condition is always true, left/right/outer join are equivalent to inner join here
						auto cross_product =
						    LogicalCrossProduct::Create(move(join.children[0]), move(join.children[1]));
						*node_ptr = move(cross_product);
						return;
					}
					case JoinType::ANTI:
						// anti join on true: empty result
						ReplaceWithEmptyResult(*node_ptr);
						return;
					default:
						// we don't handle mark/single join here yet
						break;
					}
				}
				break;
			default:
				break;
			}
		}
		// after we have propagated, we can update the statistics on both sides
		// note that it is fine to do this now, even if the same column is used again later
		// e.g. if we have i=j AND i=k, and the stats for j and k are disjoint, we know there are no results
		// so if we have e.g. i: [0, 100], j: [0, 25], k: [75, 100]
		// we can set i: [0, 25] after the first comparison, and statically determine that the second comparison is fals

		// note that we can't update statistics the same for all join types
		// mark and single joins don't filter any tuples -> so there is no propagation possible
		// anti joins have inverse statistics propagation
		// (i.e. if we have an anti join on i: [0, 100] and j: [0, 25], the resulting stats are i:[25,100])
		// for now we don't handle anti joins
		if (condition.comparison == ExpressionType::COMPARE_DISTINCT_FROM ||
		    condition.comparison == ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
			// skip update when null values are equal (for now?)
			continue;
		}
		switch (join.join_type) {
		case JoinType::INNER:
		case JoinType::SEMI: {
			UpdateFilterStatistics(*condition.left, *condition.right, condition.comparison);
			auto stats_left = PropagateExpression(condition.left);
			auto stats_right = PropagateExpression(condition.right);
			// Update join_stats when is already part of the join
			if (join.join_stats.size() == 2) {
				join.join_stats[0] = move(stats_left);
				join.join_stats[1] = move(stats_right);
			}
			break;
		}
		default:
			break;
		}
	}
}

void StatisticsPropagator::PropagateStatistics(LogicalAnyJoin &join, unique_ptr<LogicalOperator> *node_ptr) {
	// propagate the expression into the join condition
	PropagateExpression(join.condition);
}

void StatisticsPropagator::MultiplyCardinalities(unique_ptr<NodeStatistics> &stats, NodeStatistics &new_stats) {
	if (!stats->has_estimated_cardinality || !new_stats.has_estimated_cardinality || !stats->has_max_cardinality ||
	    !new_stats.has_max_cardinality) {
		stats = nullptr;
		return;
	}
	stats->estimated_cardinality = MaxValue<idx_t>(stats->estimated_cardinality, new_stats.estimated_cardinality);
	auto new_max = Hugeint::Multiply(stats->max_cardinality, new_stats.max_cardinality);
	if (new_max < NumericLimits<int64_t>::Maximum()) {
		int64_t result;
		if (!Hugeint::TryCast<int64_t>(new_max, result)) {
			throw InternalException("Overflow in cast in statistics propagation");
		}
		D_ASSERT(result >= 0);
		stats->max_cardinality = idx_t(result);
	} else {
		stats = nullptr;
	}
}

unique_ptr<NodeStatistics> StatisticsPropagator::PropagateStatistics(LogicalJoin &join,
                                                                     unique_ptr<LogicalOperator> *node_ptr) {
	// first propagate through the children of the join
	node_stats = PropagateStatistics(join.children[0]);
	for (idx_t child_idx = 1; child_idx < join.children.size(); child_idx++) {
		auto child_stats = PropagateStatistics(join.children[child_idx]);
		if (!child_stats) {
			node_stats = nullptr;
		} else if (node_stats) {
			MultiplyCardinalities(node_stats, *child_stats);
		}
	}

	auto join_type = join.join_type;
	// depending on the join type, we might need to alter the statistics
	// LEFT, FULL, RIGHT OUTER and SINGLE joins can introduce null values
	// this requires us to alter the statistics after this point in the query plan
	bool adds_null_on_left = IsRightOuterJoin(join_type);
	bool adds_null_on_right = IsLeftOuterJoin(join_type) || join_type == JoinType::SINGLE;

	vector<ColumnBinding> left_bindings, right_bindings;
	if (adds_null_on_left) {
		left_bindings = join.children[0]->GetColumnBindings();
	}
	if (adds_null_on_right) {
		right_bindings = join.children[1]->GetColumnBindings();
	}

	// then propagate into the join conditions
	switch (join.type) {
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
	case LogicalOperatorType::LOGICAL_DELIM_JOIN:
		PropagateStatistics((LogicalComparisonJoin &)join, node_ptr);
		break;
	case LogicalOperatorType::LOGICAL_ANY_JOIN:
		PropagateStatistics((LogicalAnyJoin &)join, node_ptr);
		break;
	default:
		break;
	}

	if (adds_null_on_right) {
		// left or full outer join: set IsNull() to true for all rhs statistics
		for (auto &binding : right_bindings) {
			auto stats = statistics_map.find(binding);
			if (stats != statistics_map.end()) {
				stats->second->validity_stats = make_unique<ValidityStatistics>(true);
			}
		}
	}
	if (adds_null_on_left) {
		// right or full outer join: set IsNull() to true for all lhs statistics
		for (auto &binding : left_bindings) {
			auto stats = statistics_map.find(binding);
			if (stats != statistics_map.end()) {
				stats->second->validity_stats = make_unique<ValidityStatistics>(true);
			}
		}
	}
	return move(node_stats);
}

} // namespace duckdb



namespace duckdb {

unique_ptr<NodeStatistics> StatisticsPropagator::PropagateStatistics(LogicalLimit &limit,
                                                                     unique_ptr<LogicalOperator> *node_ptr) {
	// propagate statistics in the child node
	PropagateStatistics(limit.children[0]);
	// return the node stats, with as expected cardinality the amount specified in the limit
	return make_unique<NodeStatistics>(limit.limit_val, limit.limit_val);
}

} // namespace duckdb




namespace duckdb {

unique_ptr<NodeStatistics> StatisticsPropagator::PropagateStatistics(LogicalOrder &order,
                                                                     unique_ptr<LogicalOperator> *node_ptr) {
	// first propagate to the child
	node_stats = PropagateStatistics(order.children[0]);

	// then propagate to each of the order expressions
	for (auto &bound_order : order.orders) {
		PropagateAndCompress(bound_order.expression, bound_order.stats);
	}
	return move(node_stats);
}

} // namespace duckdb



namespace duckdb {

unique_ptr<NodeStatistics> StatisticsPropagator::PropagateStatistics(LogicalProjection &proj,
                                                                     unique_ptr<LogicalOperator> *node_ptr) {
	// first propagate to the child
	node_stats = PropagateStatistics(proj.children[0]);
	if (proj.children[0]->type == LogicalOperatorType::LOGICAL_EMPTY_RESULT) {
		ReplaceWithEmptyResult(*node_ptr);
		return move(node_stats);
	}

	// then propagate to each of the expressions
	for (idx_t i = 0; i < proj.expressions.size(); i++) {
		auto stats = PropagateExpression(proj.expressions[i]);
		if (stats) {
			ColumnBinding binding(proj.table_index, i);
			statistics_map.insert(make_pair(binding, move(stats)));
		}
	}
	return move(node_stats);
}

} // namespace duckdb



namespace duckdb {

void StatisticsPropagator::AddCardinalities(unique_ptr<NodeStatistics> &stats, NodeStatistics &new_stats) {
	if (!stats->has_estimated_cardinality || !new_stats.has_estimated_cardinality || !stats->has_max_cardinality ||
	    !new_stats.has_max_cardinality) {
		stats = nullptr;
		return;
	}
	stats->estimated_cardinality += new_stats.estimated_cardinality;
	auto new_max = Hugeint::Add(stats->max_cardinality, new_stats.max_cardinality);
	if (new_max < NumericLimits<int64_t>::Maximum()) {
		int64_t result;
		if (!Hugeint::TryCast<int64_t>(new_max, result)) {
			throw InternalException("Overflow in cast in statistics propagation");
		}
		D_ASSERT(result >= 0);
		stats->max_cardinality = idx_t(result);
	} else {
		stats = nullptr;
	}
}

unique_ptr<NodeStatistics> StatisticsPropagator::PropagateStatistics(LogicalSetOperation &setop,
                                                                     unique_ptr<LogicalOperator> *node_ptr) {
	// first propagate statistics in the child nodes
	auto left_stats = PropagateStatistics(setop.children[0]);
	auto right_stats = PropagateStatistics(setop.children[1]);

	// now fetch the column bindings on both sides
	auto left_bindings = setop.children[0]->GetColumnBindings();
	auto right_bindings = setop.children[1]->GetColumnBindings();

	D_ASSERT(left_bindings.size() == right_bindings.size());
	D_ASSERT(left_bindings.size() == setop.column_count);
	for (idx_t i = 0; i < setop.column_count; i++) {
		// for each column binding, we fetch the statistics from both the lhs and the rhs
		auto left_entry = statistics_map.find(left_bindings[i]);
		auto right_entry = statistics_map.find(right_bindings[i]);
		if (left_entry == statistics_map.end() || right_entry == statistics_map.end()) {
			// no statistics on one of the sides: can't propagate stats
			continue;
		}
		unique_ptr<BaseStatistics> new_stats;
		switch (setop.type) {
		case LogicalOperatorType::LOGICAL_UNION:
			// union: merge the stats of the LHS and RHS together
			new_stats = left_entry->second->Copy();
			new_stats->Merge(*right_entry->second);
			break;
		case LogicalOperatorType::LOGICAL_EXCEPT:
			// except: use the stats of the LHS
			new_stats = left_entry->second->Copy();
			break;
		case LogicalOperatorType::LOGICAL_INTERSECT:
			// intersect: intersect the two stats
			// FIXME: for now we just use the stats of the LHS, as this is correct
			// however, the stats can be further refined to the minimal subset of the LHS and RHS
			new_stats = left_entry->second->Copy();
			break;
		default:
			throw InternalException("Unsupported setop type");
		}
		ColumnBinding binding(setop.table_index, i);
		statistics_map[binding] = move(new_stats);
	}
	if (!left_stats || !right_stats) {
		return nullptr;
	}
	if (setop.type == LogicalOperatorType::LOGICAL_UNION) {
		AddCardinalities(left_stats, *right_stats);
	}
	return left_stats;
}

} // namespace duckdb




namespace duckdb {

unique_ptr<NodeStatistics> StatisticsPropagator::PropagateStatistics(LogicalWindow &window,
                                                                     unique_ptr<LogicalOperator> *node_ptr) {
	// first propagate to the child
	node_stats = PropagateStatistics(window.children[0]);

	// then propagate to each of the order expressions
	for (auto &window_expr : window.expressions) {
		auto over_expr = reinterpret_cast<BoundWindowExpression *>(window_expr.get());
		for (auto &expr : over_expr->partitions) {
			over_expr->partitions_stats.push_back(PropagateExpression(expr));
		}
		for (auto &bound_order : over_expr->orders) {
			bound_order.stats = PropagateExpression(bound_order.expression);
		}
	}
	return move(node_stats);
}

} // namespace duckdb







namespace duckdb {

StatisticsPropagator::StatisticsPropagator(ClientContext &context) : context(context) {
}

void StatisticsPropagator::ReplaceWithEmptyResult(unique_ptr<LogicalOperator> &node) {
	node = make_unique<LogicalEmptyResult>(move(node));
}

unique_ptr<NodeStatistics> StatisticsPropagator::PropagateChildren(LogicalOperator &node,
                                                                   unique_ptr<LogicalOperator> *node_ptr) {
	for (idx_t child_idx = 0; child_idx < node.children.size(); child_idx++) {
		PropagateStatistics(node.children[child_idx]);
	}
	return nullptr;
}

unique_ptr<NodeStatistics> StatisticsPropagator::PropagateStatistics(LogicalOperator &node,
                                                                     unique_ptr<LogicalOperator> *node_ptr) {
	switch (node.type) {
	case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
		return PropagateStatistics((LogicalAggregate &)node, node_ptr);
	case LogicalOperatorType::LOGICAL_CROSS_PRODUCT:
		return PropagateStatistics((LogicalCrossProduct &)node, node_ptr);
	case LogicalOperatorType::LOGICAL_FILTER:
		return PropagateStatistics((LogicalFilter &)node, node_ptr);
	case LogicalOperatorType::LOGICAL_GET:
		return PropagateStatistics((LogicalGet &)node, node_ptr);
	case LogicalOperatorType::LOGICAL_PROJECTION:
		return PropagateStatistics((LogicalProjection &)node, node_ptr);
	case LogicalOperatorType::LOGICAL_ANY_JOIN:
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
	case LogicalOperatorType::LOGICAL_JOIN:
	case LogicalOperatorType::LOGICAL_DELIM_JOIN:
		return PropagateStatistics((LogicalJoin &)node, node_ptr);
	case LogicalOperatorType::LOGICAL_UNION:
	case LogicalOperatorType::LOGICAL_EXCEPT:
	case LogicalOperatorType::LOGICAL_INTERSECT:
		return PropagateStatistics((LogicalSetOperation &)node, node_ptr);
	case LogicalOperatorType::LOGICAL_ORDER_BY:
		return PropagateStatistics((LogicalOrder &)node, node_ptr);
	case LogicalOperatorType::LOGICAL_WINDOW:
		return PropagateStatistics((LogicalWindow &)node, node_ptr);
	default:
		return PropagateChildren(node, node_ptr);
	}
}

unique_ptr<NodeStatistics> StatisticsPropagator::PropagateStatistics(unique_ptr<LogicalOperator> &node_ptr) {
	return PropagateStatistics(*node_ptr, &node_ptr);
}

unique_ptr<BaseStatistics> StatisticsPropagator::PropagateExpression(Expression &expr,
                                                                     unique_ptr<Expression> *expr_ptr) {
	switch (expr.GetExpressionClass()) {
	case ExpressionClass::BOUND_AGGREGATE:
		return PropagateExpression((BoundAggregateExpression &)expr, expr_ptr);
	case ExpressionClass::BOUND_BETWEEN:
		return PropagateExpression((BoundBetweenExpression &)expr, expr_ptr);
	case ExpressionClass::BOUND_CASE:
		return PropagateExpression((BoundCaseExpression &)expr, expr_ptr);
	case ExpressionClass::BOUND_CONJUNCTION:
		return PropagateExpression((BoundConjunctionExpression &)expr, expr_ptr);
	case ExpressionClass::BOUND_FUNCTION:
		return PropagateExpression((BoundFunctionExpression &)expr, expr_ptr);
	case ExpressionClass::BOUND_CAST:
		return PropagateExpression((BoundCastExpression &)expr, expr_ptr);
	case ExpressionClass::BOUND_COMPARISON:
		return PropagateExpression((BoundComparisonExpression &)expr, expr_ptr);
	case ExpressionClass::BOUND_CONSTANT:
		return PropagateExpression((BoundConstantExpression &)expr, expr_ptr);
	case ExpressionClass::BOUND_COLUMN_REF:
		return PropagateExpression((BoundColumnRefExpression &)expr, expr_ptr);
	case ExpressionClass::BOUND_OPERATOR:
		return PropagateExpression((BoundOperatorExpression &)expr, expr_ptr);
	default:
		break;
	}
	ExpressionIterator::EnumerateChildren(expr, [&](unique_ptr<Expression> &child) { PropagateExpression(child); });
	return nullptr;
}

unique_ptr<BaseStatistics> StatisticsPropagator::PropagateExpression(unique_ptr<Expression> &expr) {
	auto stats = PropagateExpression(*expr, &expr);
	if (ClientConfig::GetConfig(context).query_verification_enabled && stats) {
		expr->verification_stats = stats->Copy();
	}
	return stats;
}

} // namespace duckdb






namespace duckdb {

unique_ptr<LogicalOperator> TopN::Optimize(unique_ptr<LogicalOperator> op) {
	if (op->type == LogicalOperatorType::LOGICAL_LIMIT &&
	    op->children[0]->type == LogicalOperatorType::LOGICAL_ORDER_BY) {
		auto &limit = (LogicalLimit &)*op;
		auto &order_by = (LogicalOrder &)*(op->children[0]);

		// This optimization doesn't apply when OFFSET is present without LIMIT
		// Or if offset is not constant
		if (limit.limit_val != NumericLimits<int64_t>::Maximum() || limit.offset) {
			auto topn = make_unique<LogicalTopN>(move(order_by.orders), limit.limit_val, limit.offset_val);
			topn->AddChild(move(order_by.children[0]));
			op = move(topn);
		}
	} else {
		for (auto &child : op->children) {
			child = Optimize(move(child));
		}
	}
	return op;
}

} // namespace duckdb


namespace duckdb {

BasePipelineEvent::BasePipelineEvent(shared_ptr<Pipeline> pipeline_p)
    : Event(pipeline_p->executor), pipeline(move(pipeline_p)) {
}

BasePipelineEvent::BasePipelineEvent(Pipeline &pipeline_p)
    : Event(pipeline_p.executor), pipeline(pipeline_p.shared_from_this()) {
}

} // namespace duckdb






namespace duckdb {

Event::Event(Executor &executor_p)
    : executor(executor_p), finished_tasks(0), total_tasks(0), finished_dependencies(0), total_dependencies(0),
      finished(false) {
}

void Event::CompleteDependency() {
	idx_t current_finished = ++finished_dependencies;
	D_ASSERT(current_finished <= total_dependencies);
	if (current_finished == total_dependencies) {
		// all dependencies have been completed: schedule the event
		D_ASSERT(total_tasks == 0);
		Schedule();
		if (total_tasks == 0) {
			Finish();
		}
	}
}

void Event::Finish() {
	D_ASSERT(!finished);
	FinishEvent();
	finished = true;
	// finished processing the pipeline, now we can schedule pipelines that depend on this pipeline
	for (auto &parent_entry : parents) {
		auto parent = parent_entry.lock();
		if (!parent) { // LCOV_EXCL_START
			continue;
		} // LCOV_EXCL_STOP
		// mark a dependency as completed for each of the parents
		parent->CompleteDependency();
	}
	FinalizeFinish();
}

void Event::AddDependency(Event &event) {
	total_dependencies++;
	event.parents.push_back(weak_ptr<Event>(shared_from_this()));
#ifdef DEBUG
	event.parents_raw.push_back(this);
#endif
}

const vector<Event *> &Event::GetParentsVerification() const {
	D_ASSERT(parents.size() == parents_raw.size());
	return parents_raw;
}

void Event::FinishTask() {
	D_ASSERT(finished_tasks.load() < total_tasks.load());
	idx_t current_tasks = total_tasks;
	idx_t current_finished = ++finished_tasks;
	D_ASSERT(current_finished <= current_tasks);
	if (current_finished == current_tasks) {
		Finish();
	}
}

void Event::InsertEvent(shared_ptr<Event> replacement_event) {
	replacement_event->parents = move(parents);
#ifdef DEBUG
	replacement_event->parents_raw = move(parents_raw);
#endif
	replacement_event->AddDependency(*this);
	executor.AddEvent(move(replacement_event));
}

void Event::SetTasks(vector<unique_ptr<Task>> tasks) {
	auto &ts = TaskScheduler::GetScheduler(executor.context);
	D_ASSERT(total_tasks == 0);
	D_ASSERT(!tasks.empty());
	this->total_tasks = tasks.size();
	for (auto &task : tasks) {
		ts.ScheduleTask(executor.GetToken(), move(task));
	}
}

} // namespace duckdb

















#include <algorithm>

namespace duckdb {

Executor::Executor(ClientContext &context) : context(context) {
}

Executor::~Executor() {
}

Executor &Executor::Get(ClientContext &context) {
	return context.GetExecutor();
}

void Executor::AddEvent(shared_ptr<Event> event) {
	lock_guard<mutex> elock(executor_lock);
	if (cancelled) {
		return;
	}
	events.push_back(move(event));
}

struct PipelineEventStack {
	Event *pipeline_initialize_event;
	Event *pipeline_event;
	Event *pipeline_finish_event;
	Event *pipeline_complete_event;
};

using event_map_t = unordered_map<const Pipeline *, PipelineEventStack>;

struct ScheduleEventData {
	ScheduleEventData(const vector<shared_ptr<MetaPipeline>> &meta_pipelines, vector<shared_ptr<Event>> &events,
	                  bool initial_schedule)
	    : meta_pipelines(meta_pipelines), events(events), initial_schedule(initial_schedule) {
	}

	const vector<shared_ptr<MetaPipeline>> &meta_pipelines;
	vector<shared_ptr<Event>> &events;
	bool initial_schedule;
	event_map_t event_map;
};

void Executor::SchedulePipeline(const shared_ptr<MetaPipeline> &meta_pipeline, ScheduleEventData &event_data) {
	D_ASSERT(meta_pipeline);
	auto &events = event_data.events;
	auto &event_map = event_data.event_map;

	// create events/stack for the base pipeline
	auto base_pipeline = meta_pipeline->GetBasePipeline();
	auto base_initialize_event = make_shared<PipelineInitializeEvent>(base_pipeline);
	auto base_event = make_shared<PipelineEvent>(base_pipeline);
	auto base_finish_event = make_shared<PipelineFinishEvent>(base_pipeline);
	auto base_complete_event = make_shared<PipelineCompleteEvent>(base_pipeline->executor, event_data.initial_schedule);
	PipelineEventStack base_stack {base_initialize_event.get(), base_event.get(), base_finish_event.get(),
	                               base_complete_event.get()};
	events.push_back(move(base_initialize_event));
	events.push_back(move(base_event));
	events.push_back(move(base_finish_event));
	events.push_back(move(base_complete_event));

	// dependencies: initialize -> event -> finish -> complete
	base_stack.pipeline_event->AddDependency(*base_stack.pipeline_initialize_event);
	base_stack.pipeline_finish_event->AddDependency(*base_stack.pipeline_event);
	base_stack.pipeline_complete_event->AddDependency(*base_stack.pipeline_finish_event);

	// create an event and stack for all pipelines in the MetaPipeline
	vector<shared_ptr<Pipeline>> pipelines;
	meta_pipeline->GetPipelines(pipelines, false);
	for (idx_t i = 1; i < pipelines.size(); i++) { // loop starts at 1 because 0 is the base pipeline
		auto &pipeline = pipelines[i];
		D_ASSERT(pipeline);

		// create events/stack for this pipeline
		auto pipeline_event = make_shared<PipelineEvent>(pipeline);
		Event *pipeline_finish_event_ptr;
		if (meta_pipeline->HasFinishEvent(pipeline.get())) {
			// this pipeline has its own finish event (despite going into the same sink - Finalize twice!)
			auto pipeline_finish_event = make_unique<PipelineFinishEvent>(pipeline);
			pipeline_finish_event_ptr = pipeline_finish_event.get();
			events.push_back(move(pipeline_finish_event));
			base_stack.pipeline_complete_event->AddDependency(*pipeline_finish_event_ptr);
		} else {
			pipeline_finish_event_ptr = base_stack.pipeline_finish_event;
		}
		PipelineEventStack pipeline_stack {base_stack.pipeline_initialize_event, pipeline_event.get(),
		                                   pipeline_finish_event_ptr, base_stack.pipeline_complete_event};
		events.push_back(move(pipeline_event));

		// dependencies: base_initialize -> pipeline_event -> base_finish
		pipeline_stack.pipeline_event->AddDependency(*base_stack.pipeline_initialize_event);
		pipeline_stack.pipeline_finish_event->AddDependency(*pipeline_stack.pipeline_event);

		// add pipeline stack to event map
		event_map.insert(make_pair(pipeline.get(), pipeline_stack));
	}

	// add base stack to the event data too
	event_map.insert(make_pair(base_pipeline.get(), base_stack));

	// set up the dependencies within this MetaPipeline
	for (auto &pipeline : pipelines) {
		auto source = pipeline->GetSource();
		if (source->type == PhysicalOperatorType::TABLE_SCAN) {
			// we have to reset the source here (in the main thread), because some of our clients (looking at you, R)
			// do not like it when threads other than the main thread call into R, for e.g., arrow scans
			pipeline->ResetSource(true);
		}

		auto dependencies = meta_pipeline->GetDependencies(pipeline.get());
		if (!dependencies) {
			continue;
		}
		auto &pipeline_stack = event_map[pipeline.get()];
		for (auto &dependency : *dependencies) {
			auto &dependency_stack = event_map[dependency];
			pipeline_stack.pipeline_event->AddDependency(*dependency_stack.pipeline_event);
		}
	}
}

void Executor::ScheduleEventsInternal(ScheduleEventData &event_data) {
	auto &events = event_data.events;
	D_ASSERT(events.empty());

	// create all the required pipeline events
	for (auto &pipeline : event_data.meta_pipelines) {
		SchedulePipeline(pipeline, event_data);
	}

	// set up the dependencies across MetaPipelines
	auto &event_map = event_data.event_map;
	for (auto &entry : event_map) {
		auto pipeline = entry.first;
		for (auto &dependency : pipeline->dependencies) {
			auto dep = dependency.lock();
			D_ASSERT(dep);
			auto event_map_entry = event_map.find(dep.get());
			D_ASSERT(event_map_entry != event_map.end());
			auto &dep_entry = event_map_entry->second;
			D_ASSERT(dep_entry.pipeline_complete_event);
			entry.second.pipeline_event->AddDependency(*dep_entry.pipeline_complete_event);
		}
	}

	// verify that we have no cyclic dependencies
	VerifyScheduledEvents(event_data);

	// schedule the pipelines that do not have dependencies
	for (auto &event : events) {
		if (!event->HasDependencies()) {
			event->Schedule();
		}
	}
}

void Executor::ScheduleEvents(const vector<shared_ptr<MetaPipeline>> &meta_pipelines) {
	ScheduleEventData event_data(meta_pipelines, events, true);
	ScheduleEventsInternal(event_data);
}

void Executor::VerifyScheduledEvents(const ScheduleEventData &event_data) {
#ifdef DEBUG
	const idx_t count = event_data.events.size();
	vector<Event *> vertices;
	vertices.reserve(count);
	for (const auto &event : event_data.events) {
		vertices.push_back(event.get());
	}
	vector<bool> visited(count, false);
	vector<bool> recursion_stack(count, false);
	for (idx_t i = 0; i < count; i++) {
		VerifyScheduledEventsInternal(i, vertices, visited, recursion_stack);
	}
#endif
}

void Executor::VerifyScheduledEventsInternal(const idx_t vertex, const vector<Event *> &vertices, vector<bool> &visited,
                                             vector<bool> &recursion_stack) {
	D_ASSERT(!recursion_stack[vertex]); // this vertex is in the recursion stack: circular dependency!
	if (visited[vertex]) {
		return; // early out: we already visited this vertex
	}

	auto &parents = vertices[vertex]->GetParentsVerification();
	if (parents.empty()) {
		return; // early out: outgoing edges
	}

	// create a vector the indices of the adjacent events
	vector<idx_t> adjacent;
	const idx_t count = vertices.size();
	for (auto parent : parents) {
		idx_t i;
		for (i = 0; i < count; i++) {
			if (vertices[i] == parent) {
				adjacent.push_back(i);
				break;
			}
		}
		D_ASSERT(i != count); // dependency must be in there somewhere
	}

	// mark vertex as visited and add to recursion stack
	visited[vertex] = true;
	recursion_stack[vertex] = true;

	// recurse into adjacent vertices
	for (const auto &i : adjacent) {
		VerifyScheduledEventsInternal(i, vertices, visited, recursion_stack);
	}

	// remove vertex from recursion stack
	recursion_stack[vertex] = false;
}

void Executor::AddRecursiveCTE(PhysicalOperator *rec_cte) {
	recursive_ctes.push_back(rec_cte);
}

void Executor::ReschedulePipelines(const vector<shared_ptr<MetaPipeline>> &pipelines_p,
                                   vector<shared_ptr<Event>> &events_p) {
	ScheduleEventData event_data(pipelines_p, events_p, false);
	ScheduleEventsInternal(event_data);
}

bool Executor::NextExecutor() {
	if (root_pipeline_idx >= root_pipelines.size()) {
		return false;
	}
	root_pipelines[root_pipeline_idx]->Reset();
	root_executor = make_unique<PipelineExecutor>(context, *root_pipelines[root_pipeline_idx]);
	root_pipeline_idx++;
	return true;
}

void Executor::VerifyPipeline(Pipeline &pipeline) {
	D_ASSERT(!pipeline.ToString().empty());
	auto operators = pipeline.GetOperators();
	for (auto &other_pipeline : pipelines) {
		auto other_operators = other_pipeline->GetOperators();
		for (idx_t op_idx = 0; op_idx < operators.size(); op_idx++) {
			for (idx_t other_idx = 0; other_idx < other_operators.size(); other_idx++) {
				auto &left = *operators[op_idx];
				auto &right = *other_operators[other_idx];
				if (left.Equals(right)) {
					D_ASSERT(right.Equals(left));
				} else {
					D_ASSERT(!right.Equals(left));
				}
			}
		}
	}
}

void Executor::VerifyPipelines() {
#ifdef DEBUG
	for (auto &pipeline : pipelines) {
		VerifyPipeline(*pipeline);
	}
#endif
}

void Executor::Initialize(unique_ptr<PhysicalOperator> physical_plan) {
	Reset();
	owned_plan = move(physical_plan);
	InitializeInternal(owned_plan.get());
}

void Executor::Initialize(PhysicalOperator *plan) {
	Reset();
	InitializeInternal(plan);
}

void Executor::InitializeInternal(PhysicalOperator *plan) {

	auto &scheduler = TaskScheduler::GetScheduler(context);
	{
		lock_guard<mutex> elock(executor_lock);
		physical_plan = plan;

		this->profiler = ClientData::Get(context).profiler;
		profiler->Initialize(physical_plan);
		this->producer = scheduler.CreateProducer();

		// build and ready the pipelines
		PipelineBuildState state;
		auto root_pipeline = make_shared<MetaPipeline>(*this, state, nullptr);
		root_pipeline->Build(physical_plan);
		root_pipeline->Ready();

		// ready recursive cte pipelines too
		for (auto &rec_cte : recursive_ctes) {
			D_ASSERT(rec_cte->type == PhysicalOperatorType::RECURSIVE_CTE);
			auto &rec_cte_op = (PhysicalRecursiveCTE &)*rec_cte;
			rec_cte_op.recursive_meta_pipeline->Ready();
		}

		// set root pipelines, i.e., all pipelines that end in the final sink
		root_pipeline->GetPipelines(root_pipelines, false);
		root_pipeline_idx = 0;

		// collect all meta-pipelines from the root pipeline
		vector<shared_ptr<MetaPipeline>> to_schedule;
		root_pipeline->GetMetaPipelines(to_schedule, true, true);

		// number of 'PipelineCompleteEvent's is equal to the number of meta pipelines, so we have to set it here
		total_pipelines = to_schedule.size();

		// collect all pipelines from the root pipelines (recursively) for the progress bar and verify them
		root_pipeline->GetPipelines(pipelines, true);

		// finally, verify and schedule
		VerifyPipelines();
		ScheduleEvents(to_schedule);
	}
}

void Executor::CancelTasks() {
	task.reset();
	// we do this by creating weak pointers to all pipelines
	// then clearing our references to the pipelines
	// and waiting until all pipelines have been destroyed
	vector<weak_ptr<Pipeline>> weak_references;
	{
		lock_guard<mutex> elock(executor_lock);
		weak_references.reserve(pipelines.size());
		cancelled = true;
		for (auto &pipeline : pipelines) {
			weak_references.push_back(weak_ptr<Pipeline>(pipeline));
		}
		for (auto op : recursive_ctes) {
			D_ASSERT(op->type == PhysicalOperatorType::RECURSIVE_CTE);
			auto &rec_cte = (PhysicalRecursiveCTE &)*op;
			rec_cte.recursive_meta_pipeline.reset();
		}
		pipelines.clear();
		root_pipelines.clear();
		events.clear();
	}
	WorkOnTasks();
	for (auto &weak_ref : weak_references) {
		while (true) {
			auto weak = weak_ref.lock();
			if (!weak) {
				break;
			}
		}
	}
}

void Executor::WorkOnTasks() {
	auto &scheduler = TaskScheduler::GetScheduler(context);

	unique_ptr<Task> task;
	while (scheduler.GetTaskFromProducer(*producer, task)) {
		task->Execute(TaskExecutionMode::PROCESS_ALL);
		task.reset();
	}
}

PendingExecutionResult Executor::ExecuteTask() {
	if (execution_result != PendingExecutionResult::RESULT_NOT_READY) {
		return execution_result;
	}
	// check if there are any incomplete pipelines
	auto &scheduler = TaskScheduler::GetScheduler(context);
	while (completed_pipelines < total_pipelines) {
		// there are! if we don't already have a task, fetch one
		if (!task) {
			scheduler.GetTaskFromProducer(*producer, task);
		}
		if (task) {
			// if we have a task, partially process it
			auto result = task->Execute(TaskExecutionMode::PROCESS_PARTIAL);
			if (result != TaskExecutionResult::TASK_NOT_FINISHED) {
				// if the task is finished, clean it up
				task.reset();
			}
		}
		if (!HasError()) {
			// we (partially) processed a task and no exceptions were thrown
			// give back control to the caller
			return PendingExecutionResult::RESULT_NOT_READY;
		}
		execution_result = PendingExecutionResult::EXECUTION_ERROR;

		// an exception has occurred executing one of the pipelines
		// we need to cancel all tasks associated with this executor
		CancelTasks();
		ThrowException();
	}
	D_ASSERT(!task);

	lock_guard<mutex> elock(executor_lock);
	pipelines.clear();
	NextExecutor();
	if (HasError()) { // LCOV_EXCL_START
		// an exception has occurred executing one of the pipelines
		execution_result = PendingExecutionResult::EXECUTION_ERROR;
		ThrowException();
	} // LCOV_EXCL_STOP
	execution_result = PendingExecutionResult::RESULT_READY;
	return execution_result;
}

void Executor::Reset() {
	lock_guard<mutex> elock(executor_lock);
	physical_plan = nullptr;
	cancelled = false;
	owned_plan.reset();
	root_executor.reset();
	root_pipelines.clear();
	root_pipeline_idx = 0;
	completed_pipelines = 0;
	total_pipelines = 0;
	exceptions.clear();
	pipelines.clear();
	events.clear();
	execution_result = PendingExecutionResult::RESULT_NOT_READY;
}

shared_ptr<Pipeline> Executor::CreateChildPipeline(Pipeline *current, PhysicalOperator *op) {
	D_ASSERT(!current->operators.empty());
	D_ASSERT(op->IsSource());
	// found another operator that is a source, schedule a child pipeline
	// 'op' is the source, and the sink is the same
	auto child_pipeline = make_shared<Pipeline>(*this);
	child_pipeline->sink = current->sink;
	child_pipeline->source = op;

	// the child pipeline has the same operators up until 'op'
	for (auto current_op : current->operators) {
		if (current_op == op) {
			break;
		}
		child_pipeline->operators.push_back(current_op);
	}

	return child_pipeline;
}

vector<LogicalType> Executor::GetTypes() {
	D_ASSERT(physical_plan);
	return physical_plan->GetTypes();
}

void Executor::PushError(PreservedError exception) {
	lock_guard<mutex> elock(error_lock);
	// interrupt execution of any other pipelines that belong to this executor
	context.interrupted = true;
	// push the exception onto the stack
	exceptions.push_back(move(exception));
}

bool Executor::HasError() {
	lock_guard<mutex> elock(error_lock);
	return !exceptions.empty();
}

void Executor::ThrowException() {
	lock_guard<mutex> elock(error_lock);
	D_ASSERT(!exceptions.empty());
	auto &entry = exceptions[0];
	entry.Throw();
}

void Executor::Flush(ThreadContext &tcontext) {
	profiler->Flush(tcontext.profiler);
}

bool Executor::GetPipelinesProgress(double &current_progress) { // LCOV_EXCL_START
	lock_guard<mutex> elock(executor_lock);

	vector<double> progress;
	vector<idx_t> cardinality;
	idx_t total_cardinality = 0;
	for (auto &pipeline : pipelines) {
		double child_percentage;
		idx_t child_cardinality;

		if (!pipeline->GetProgress(child_percentage, child_cardinality)) {
			return false;
		}
		progress.push_back(child_percentage);
		cardinality.push_back(child_cardinality);
		total_cardinality += child_cardinality;
	}
	current_progress = 0;
	for (size_t i = 0; i < progress.size(); i++) {
		current_progress += progress[i] * double(cardinality[i]) / double(total_cardinality);
	}
	return true;
} // LCOV_EXCL_STOP

bool Executor::HasResultCollector() {
	return physical_plan->type == PhysicalOperatorType::RESULT_COLLECTOR;
}

unique_ptr<QueryResult> Executor::GetResult() {
	D_ASSERT(HasResultCollector());
	auto &result_collector = (PhysicalResultCollector &)*physical_plan;
	D_ASSERT(result_collector.sink_state);
	return result_collector.GetResult(*result_collector.sink_state);
}

unique_ptr<DataChunk> Executor::FetchChunk() {
	D_ASSERT(physical_plan);

	auto chunk = make_unique<DataChunk>();
	root_executor->InitializeChunk(*chunk);
	while (true) {
		root_executor->ExecutePull(*chunk);
		if (chunk->size() == 0) {
			root_executor->PullFinalize();
			if (NextExecutor()) {
				continue;
			}
			break;
		} else {
			break;
		}
	}
	return chunk;
}

} // namespace duckdb



namespace duckdb {

ExecutorTask::ExecutorTask(Executor &executor_p) : executor(executor_p) {
}

ExecutorTask::ExecutorTask(ClientContext &context) : ExecutorTask(Executor::Get(context)) {
}

ExecutorTask::~ExecutorTask() {
}

TaskExecutionResult ExecutorTask::Execute(TaskExecutionMode mode) {
	try {
		return ExecuteTask(mode);
	} catch (Exception &ex) {
		executor.PushError(PreservedError(ex));
	} catch (std::exception &ex) {
		executor.PushError(PreservedError(ex));
	} catch (...) { // LCOV_EXCL_START
		executor.PushError(PreservedError("Unknown exception in Finalize!"));
	} // LCOV_EXCL_STOP
	return TaskExecutionResult::TASK_ERROR;
}

} // namespace duckdb





namespace duckdb {

MetaPipeline::MetaPipeline(Executor &executor_p, PipelineBuildState &state_p, PhysicalOperator *sink_p)
    : executor(executor_p), state(state_p), sink(sink_p), recursive_cte(false), next_batch_index(0) {
	CreatePipeline();
}

Executor &MetaPipeline::GetExecutor() const {
	return executor;
}

PipelineBuildState &MetaPipeline::GetState() const {
	return state;
}

PhysicalOperator *MetaPipeline::GetSink() const {
	return sink;
}

shared_ptr<Pipeline> &MetaPipeline::GetBasePipeline() {
	return pipelines[0];
}

void MetaPipeline::GetPipelines(vector<shared_ptr<Pipeline>> &result, bool recursive) {
	result.insert(result.end(), pipelines.begin(), pipelines.end());
	if (recursive) {
		for (auto &child : children) {
			child->GetPipelines(result, true);
		}
	}
}

void MetaPipeline::GetMetaPipelines(vector<shared_ptr<MetaPipeline>> &result, bool recursive, bool skip) {
	if (!skip) {
		result.push_back(shared_from_this());
	}
	if (recursive) {
		for (auto &child : children) {
			child->GetMetaPipelines(result, true, false);
		}
	}
}

const vector<Pipeline *> *MetaPipeline::GetDependencies(Pipeline *dependant) const {
	auto it = dependencies.find(dependant);
	if (it == dependencies.end()) {
		return nullptr;
	} else {
		return &it->second;
	}
}

bool MetaPipeline::HasRecursiveCTE() const {
	return recursive_cte;
}

void MetaPipeline::SetRecursiveCTE() {
	recursive_cte = true;
}

void MetaPipeline::AssignNextBatchIndex(Pipeline *pipeline) {
	pipeline->base_batch_index = next_batch_index++ * PipelineBuildState::BATCH_INCREMENT;
}

void MetaPipeline::Build(PhysicalOperator *op) {
	D_ASSERT(pipelines.size() == 1);
	D_ASSERT(children.empty());
	D_ASSERT(final_pipelines.empty());
	op->BuildPipelines(*pipelines.back(), *this);
}

void MetaPipeline::Ready() {
	for (auto &pipeline : pipelines) {
		pipeline->Ready();
	}
	for (auto &child : children) {
		child->Ready();
	}
}

MetaPipeline *MetaPipeline::CreateChildMetaPipeline(Pipeline &current, PhysicalOperator *op) {
	children.push_back(make_unique<MetaPipeline>(executor, state, op));
	auto child_meta_pipeline = children.back().get();
	// child MetaPipeline must finish completely before this MetaPipeline can start
	current.AddDependency(child_meta_pipeline->GetBasePipeline());
	// child meta pipeline is part of the recursive CTE too
	child_meta_pipeline->recursive_cte = recursive_cte;
	return child_meta_pipeline;
}

Pipeline *MetaPipeline::CreatePipeline() {
	pipelines.emplace_back(make_unique<Pipeline>(executor));
	state.SetPipelineSink(*pipelines.back(), sink, next_batch_index++);
	return pipelines.back().get();
}

void MetaPipeline::AddDependenciesFrom(Pipeline *dependant, Pipeline *start, bool including) {
	// find 'start'
	auto it = pipelines.begin();
	for (; it->get() != start; it++) {
	}

	if (!including) {
		it++;
	}

	// collect pipelines that were created from then
	vector<Pipeline *> created_pipelines;
	for (; it != pipelines.end(); it++) {
		if (it->get() == dependant) {
			// cannot depend on itself
			continue;
		}
		created_pipelines.push_back(it->get());
	}

	// add them to the dependencies
	auto &deps = dependencies[dependant];
	deps.insert(deps.begin(), created_pipelines.begin(), created_pipelines.end());
}

void MetaPipeline::AddFinishEvent(Pipeline *pipeline) {
	finish_pipelines.insert(pipeline);
}

bool MetaPipeline::HasFinishEvent(Pipeline *pipeline) {
	return finish_pipelines.find(pipeline) != finish_pipelines.end();
}

Pipeline *MetaPipeline::CreateUnionPipeline(Pipeline &current, bool order_matters) {
	if (HasRecursiveCTE()) {
		throw NotImplementedException("UNIONS are not supported in recursive CTEs yet");
	}

	// create the union pipeline (batch index 0, should be set correctly afterwards)
	auto union_pipeline = CreatePipeline();
	state.SetPipelineOperators(*union_pipeline, state.GetPipelineOperators(current));
	state.SetPipelineSink(*union_pipeline, sink, 0);

	// 'union_pipeline' inherits ALL dependencies of 'current' (within this MetaPipeline, and across MetaPipelines)
	union_pipeline->dependencies = current.dependencies;
	auto current_deps = GetDependencies(&current);
	if (current_deps) {
		dependencies[union_pipeline] = *current_deps;
	}

	if (order_matters) {
		// if we need to preserve order, or if the sink is not parallel, we set a dependency
		dependencies[union_pipeline].push_back(&current);
	}

	return union_pipeline;
}

void MetaPipeline::CreateChildPipeline(Pipeline &current, PhysicalOperator *op, Pipeline *last_pipeline) {
	// rule 2: 'current' must be fully built (down to the source) before creating the child pipeline
	D_ASSERT(current.source);
	if (HasRecursiveCTE()) {
		throw NotImplementedException("Child pipelines are not supported in recursive CTEs yet");
	}

	// create the child pipeline (same batch index)
	pipelines.emplace_back(state.CreateChildPipeline(executor, current, op));
	auto child_pipeline = pipelines.back().get();
	child_pipeline->base_batch_index = current.base_batch_index;

	// child pipeline has a depency (within this MetaPipeline on all pipelines that were scheduled
	// between 'current' and now (including 'current') - set them up
	dependencies[child_pipeline].push_back(&current);
	AddDependenciesFrom(child_pipeline, last_pipeline, false);
	D_ASSERT(!GetDependencies(child_pipeline)->empty());
}

} // namespace duckdb
















namespace duckdb {

class PipelineTask : public ExecutorTask {
	static constexpr const idx_t PARTIAL_CHUNK_COUNT = 50;

public:
	explicit PipelineTask(Pipeline &pipeline_p, shared_ptr<Event> event_p)
	    : ExecutorTask(pipeline_p.executor), pipeline(pipeline_p), event(move(event_p)) {
	}

	Pipeline &pipeline;
	shared_ptr<Event> event;
	unique_ptr<PipelineExecutor> pipeline_executor;

public:
	TaskExecutionResult ExecuteTask(TaskExecutionMode mode) override {
		if (!pipeline_executor) {
			pipeline_executor = make_unique<PipelineExecutor>(pipeline.GetClientContext(), pipeline);
		}
		if (mode == TaskExecutionMode::PROCESS_PARTIAL) {
			bool finished = pipeline_executor->Execute(PARTIAL_CHUNK_COUNT);
			if (!finished) {
				return TaskExecutionResult::TASK_NOT_FINISHED;
			}
		} else {
			pipeline_executor->Execute();
		}
		event->FinishTask();
		pipeline_executor.reset();
		return TaskExecutionResult::TASK_FINISHED;
	}
};

Pipeline::Pipeline(Executor &executor_p)
    : executor(executor_p), ready(false), initialized(false), source(nullptr), sink(nullptr) {
}

ClientContext &Pipeline::GetClientContext() {
	return executor.context;
}

bool Pipeline::GetProgress(double &current_percentage, idx_t &source_cardinality) {
	D_ASSERT(source);
	source_cardinality = source->estimated_cardinality;
	if (!initialized) {
		current_percentage = 0;
		return true;
	}
	auto &client = executor.context;
	current_percentage = source->GetProgress(client, *source_state);
	return current_percentage >= 0;
}

void Pipeline::ScheduleSequentialTask(shared_ptr<Event> &event) {
	vector<unique_ptr<Task>> tasks;
	tasks.push_back(make_unique<PipelineTask>(*this, event));
	event->SetTasks(move(tasks));
}

bool Pipeline::ScheduleParallel(shared_ptr<Event> &event) {
	// check if the sink, source and all intermediate operators support parallelism
	if (!sink->ParallelSink()) {
		return false;
	}
	if (!source->ParallelSource()) {
		return false;
	}
	for (auto &op : operators) {
		if (!op->ParallelOperator()) {
			return false;
		}
	}
	if (sink->RequiresBatchIndex()) {
		if (!source->SupportsBatchIndex()) {
			throw InternalException(
			    "Attempting to schedule a pipeline where the sink requires batch index but source does not support it");
		}
	}
	idx_t max_threads = source_state->MaxThreads();
	return LaunchScanTasks(event, max_threads);
}

bool Pipeline::IsOrderDependent() const {
	auto &config = DBConfig::GetConfig(executor.context);
	if (!config.options.preserve_insertion_order) {
		return false;
	}
	if (sink && sink->IsOrderDependent()) {
		return true;
	}
	if (source && source->IsOrderDependent()) {
		return true;
	}
	for (auto &op : operators) {
		if (op->IsOrderDependent()) {
			return true;
		}
	}
	return false;
}

void Pipeline::Schedule(shared_ptr<Event> &event) {
	D_ASSERT(ready);
	D_ASSERT(sink);
	Reset();
	if (!ScheduleParallel(event)) {
		// could not parallelize this pipeline: push a sequential task instead
		ScheduleSequentialTask(event);
	}
}

bool Pipeline::LaunchScanTasks(shared_ptr<Event> &event, idx_t max_threads) {
	// split the scan up into parts and schedule the parts
	auto &scheduler = TaskScheduler::GetScheduler(executor.context);
	idx_t active_threads = scheduler.NumberOfThreads();
	if (max_threads > active_threads) {
		max_threads = active_threads;
	}
	if (max_threads <= 1) {
		// too small to parallelize
		return false;
	}

	// launch a task for every thread
	vector<unique_ptr<Task>> tasks;
	for (idx_t i = 0; i < max_threads; i++) {
		tasks.push_back(make_unique<PipelineTask>(*this, event));
	}
	event->SetTasks(move(tasks));
	return true;
}

void Pipeline::ResetSink() {
	if (sink) {
		lock_guard<mutex> guard(sink->lock);
		if (!sink->sink_state) {
			sink->sink_state = sink->GetGlobalSinkState(GetClientContext());
		}
	}
}

void Pipeline::Reset() {
	ResetSink();
	for (auto &op : operators) {
		if (op) {
			lock_guard<mutex> guard(op->lock);
			if (!op->op_state) {
				op->op_state = op->GetGlobalOperatorState(GetClientContext());
			}
		}
	}
	ResetSource(false);
	// we no longer reset source here because this function is no longer guaranteed to be called by the main thread
	// source reset needs to be called by the main thread because resetting a source may call into clients like R
	initialized = true;
}

void Pipeline::ResetSource(bool force) {
	if (force || !source_state) {
		source_state = source->GetGlobalSourceState(GetClientContext());
	}
}

void Pipeline::Ready() {
	if (ready) {
		return;
	}
	ready = true;
	std::reverse(operators.begin(), operators.end());
}

void Pipeline::Finalize(Event &event) {
	if (executor.HasError()) {
		return;
	}
	D_ASSERT(ready);
	try {
		auto sink_state = sink->Finalize(*this, event, executor.context, *sink->sink_state);
		sink->sink_state->state = sink_state;
	} catch (Exception &ex) { // LCOV_EXCL_START
		executor.PushError(PreservedError(ex));
	} catch (std::exception &ex) {
		executor.PushError(PreservedError(ex));
	} catch (...) {
		executor.PushError(PreservedError("Unknown exception in Finalize!"));
	} // LCOV_EXCL_STOP
}

void Pipeline::AddDependency(shared_ptr<Pipeline> &pipeline) {
	D_ASSERT(pipeline);
	dependencies.push_back(weak_ptr<Pipeline>(pipeline));
	pipeline->parents.push_back(weak_ptr<Pipeline>(shared_from_this()));
}

string Pipeline::ToString() const {
	TreeRenderer renderer;
	return renderer.ToString(*this);
}

void Pipeline::Print() const {
	Printer::Print(ToString());
}

void Pipeline::PrintDependencies() const {
	for (auto &dep : dependencies) {
		shared_ptr<Pipeline>(dep)->Print();
	}
}

vector<PhysicalOperator *> Pipeline::GetOperators() const {
	vector<PhysicalOperator *> result;
	D_ASSERT(source);
	result.push_back(source);
	result.insert(result.end(), operators.begin(), operators.end());
	if (sink) {
		result.push_back(sink);
	}
	return result;
}

//===--------------------------------------------------------------------===//
// Pipeline Build State
//===--------------------------------------------------------------------===//
void PipelineBuildState::SetPipelineSource(Pipeline &pipeline, PhysicalOperator *op) {
	pipeline.source = op;
}

void PipelineBuildState::SetPipelineSink(Pipeline &pipeline, PhysicalOperator *op, idx_t sink_pipeline_count) {
	pipeline.sink = op;
	// set the base batch index of this pipeline based on how many other pipelines have this node as their sink
	pipeline.base_batch_index = BATCH_INCREMENT * sink_pipeline_count;
}

void PipelineBuildState::AddPipelineOperator(Pipeline &pipeline, PhysicalOperator *op) {
	pipeline.operators.push_back(op);
}

PhysicalOperator *PipelineBuildState::GetPipelineSource(Pipeline &pipeline) {
	return pipeline.source;
}

PhysicalOperator *PipelineBuildState::GetPipelineSink(Pipeline &pipeline) {
	return pipeline.sink;
}

void PipelineBuildState::SetPipelineOperators(Pipeline &pipeline, vector<PhysicalOperator *> operators) {
	pipeline.operators = move(operators);
}

shared_ptr<Pipeline> PipelineBuildState::CreateChildPipeline(Executor &executor, Pipeline &pipeline,
                                                             PhysicalOperator *op) {
	return executor.CreateChildPipeline(&pipeline, op);
}

vector<PhysicalOperator *> PipelineBuildState::GetPipelineOperators(Pipeline &pipeline) {
	return pipeline.operators;
}

} // namespace duckdb



namespace duckdb {

PipelineCompleteEvent::PipelineCompleteEvent(Executor &executor, bool complete_pipeline_p)
    : Event(executor), complete_pipeline(complete_pipeline_p) {
}

void PipelineCompleteEvent::Schedule() {
}

void PipelineCompleteEvent::FinalizeFinish() {
	if (complete_pipeline) {
		executor.CompletePipeline();
	}
}

} // namespace duckdb



namespace duckdb {

PipelineEvent::PipelineEvent(shared_ptr<Pipeline> pipeline_p) : BasePipelineEvent(move(pipeline_p)) {
}

void PipelineEvent::Schedule() {
	auto event = shared_from_this();
	auto &executor = pipeline->executor;
	try {
		pipeline->Schedule(event);
		D_ASSERT(total_tasks > 0);
	} catch (Exception &ex) {
		executor.PushError(PreservedError(ex));
	} catch (std::exception &ex) {
		executor.PushError(PreservedError(ex));
	} catch (...) { // LCOV_EXCL_START
		executor.PushError(PreservedError("Unknown exception in Finalize!"));
	} // LCOV_EXCL_STOP
}

void PipelineEvent::FinishEvent() {
}

} // namespace duckdb




namespace duckdb {

PipelineExecutor::PipelineExecutor(ClientContext &context_p, Pipeline &pipeline_p)
    : pipeline(pipeline_p), thread(context_p), context(context_p, thread, &pipeline_p) {
	D_ASSERT(pipeline.source_state);
	local_source_state = pipeline.source->GetLocalSourceState(context, *pipeline.source_state);
	if (pipeline.sink) {
		local_sink_state = pipeline.sink->GetLocalSinkState(context);
		requires_batch_index = pipeline.sink->RequiresBatchIndex() && pipeline.source->SupportsBatchIndex();
	}

	intermediate_chunks.reserve(pipeline.operators.size());
	intermediate_states.reserve(pipeline.operators.size());
	for (idx_t i = 0; i < pipeline.operators.size(); i++) {
		auto prev_operator = i == 0 ? pipeline.source : pipeline.operators[i - 1];
		auto current_operator = pipeline.operators[i];

		auto chunk = make_unique<DataChunk>();
		chunk->Initialize(Allocator::Get(context.client), prev_operator->GetTypes());
		intermediate_chunks.push_back(move(chunk));

		auto op_state = current_operator->GetOperatorState(context);
		intermediate_states.push_back(std::move(op_state));

		if (current_operator->IsSink() && current_operator->sink_state->state == SinkFinalizeType::NO_OUTPUT_POSSIBLE) {
			// one of the operators has already figured out no output is possible
			// we can skip executing the pipeline
			FinishProcessing();
		}
	}
	InitializeChunk(final_chunk);
}

bool PipelineExecutor::Execute(idx_t max_chunks) {
	D_ASSERT(pipeline.sink);
	bool exhausted_source = false;
	auto &source_chunk = pipeline.operators.empty() ? final_chunk : *intermediate_chunks[0];
	for (idx_t i = 0; i < max_chunks; i++) {
		if (IsFinished()) {
			break;
		}
		source_chunk.Reset();
		FetchFromSource(source_chunk);
		if (source_chunk.size() == 0) {
			exhausted_source = true;
			break;
		}
		auto result = ExecutePushInternal(source_chunk);
		if (result == OperatorResultType::FINISHED) {
			D_ASSERT(IsFinished());
			break;
		}
	}
	if (!exhausted_source && !IsFinished()) {
		return false;
	}
	PushFinalize();
	return true;
}

void PipelineExecutor::Execute() {
	Execute(NumericLimits<idx_t>::Maximum());
}

OperatorResultType PipelineExecutor::ExecutePush(DataChunk &input) { // LCOV_EXCL_START
	return ExecutePushInternal(input);
} // LCOV_EXCL_STOP

void PipelineExecutor::FinishProcessing(int32_t operator_idx) {
	finished_processing_idx = operator_idx < 0 ? NumericLimits<int32_t>::Maximum() : operator_idx;
	in_process_operators = stack<idx_t>();
}

bool PipelineExecutor::IsFinished() {
	return finished_processing_idx >= 0;
}

OperatorResultType PipelineExecutor::ExecutePushInternal(DataChunk &input, idx_t initial_idx) {
	D_ASSERT(pipeline.sink);
	if (input.size() == 0) { // LCOV_EXCL_START
		return OperatorResultType::NEED_MORE_INPUT;
	} // LCOV_EXCL_STOP
	while (true) {
		OperatorResultType result;
		// Note: if input is the final_chunk, we don't do any executing, the chunk just needs to be sinked
		if (&input != &final_chunk) {
			final_chunk.Reset();
			result = Execute(input, final_chunk, initial_idx);
			if (result == OperatorResultType::FINISHED) {
				return OperatorResultType::FINISHED;
			}
		} else {
			result = OperatorResultType::NEED_MORE_INPUT;
		}
		auto &sink_chunk = final_chunk;
		if (sink_chunk.size() > 0) {
			StartOperator(pipeline.sink);
			D_ASSERT(pipeline.sink);
			D_ASSERT(pipeline.sink->sink_state);
			auto sink_result = pipeline.sink->Sink(context, *pipeline.sink->sink_state, *local_sink_state, sink_chunk);
			EndOperator(pipeline.sink, nullptr);
			if (sink_result == SinkResultType::FINISHED) {
				FinishProcessing();
				return OperatorResultType::FINISHED;
			}
		}
		if (result == OperatorResultType::NEED_MORE_INPUT) {
			return OperatorResultType::NEED_MORE_INPUT;
		}
	}
}

// Pull a single DataChunk from the pipeline by flushing any operators holding cached output
void PipelineExecutor::FlushCachingOperatorsPull(DataChunk &result) {
	idx_t start_idx = IsFinished() ? idx_t(finished_processing_idx) : 0;
	idx_t op_idx = start_idx;
	while (op_idx < pipeline.operators.size()) {
		if (!pipeline.operators[op_idx]->RequiresFinalExecute()) {
			op_idx++;
			continue;
		}

		OperatorFinalizeResultType finalize_result;
		DataChunk &curr_chunk =
		    op_idx + 1 >= intermediate_chunks.size() ? final_chunk : *intermediate_chunks[op_idx + 1];

		if (pending_final_execute) {
			// Still have a cached chunk from a last pull, reuse chunk
			finalize_result = cached_final_execute_result;
		} else {
			// Flush the current operator
			finalize_result = pipeline.operators[op_idx]->FinalExecute(
			    context, curr_chunk, *pipeline.operators[op_idx]->op_state, *intermediate_states[op_idx]);
		}

		auto execute_result = Execute(curr_chunk, result, op_idx + 1);

		if (execute_result == OperatorResultType::HAVE_MORE_OUTPUT) {
			pending_final_execute = true;
			cached_final_execute_result = finalize_result;
		} else {
			pending_final_execute = false;
			if (finalize_result == OperatorFinalizeResultType::FINISHED) {
				FinishProcessing(op_idx);
				op_idx++;
			}
		}

		// Some non-empty result was pulled from some caching operator, we're done for this pull
		if (result.size() > 0) {
			break;
		}
	}
}

// Push all remaining cached operator output through the pipeline
void PipelineExecutor::FlushCachingOperatorsPush() {
	idx_t start_idx = IsFinished() ? idx_t(finished_processing_idx) : 0;
	for (idx_t op_idx = start_idx; op_idx < pipeline.operators.size(); op_idx++) {
		if (!pipeline.operators[op_idx]->RequiresFinalExecute()) {
			continue;
		}

		OperatorFinalizeResultType finalize_result;
		OperatorResultType push_result;

		do {
			auto &curr_chunk =
			    op_idx + 1 >= intermediate_chunks.size() ? final_chunk : *intermediate_chunks[op_idx + 1];
			finalize_result = pipeline.operators[op_idx]->FinalExecute(
			    context, curr_chunk, *pipeline.operators[op_idx]->op_state, *intermediate_states[op_idx]);
			push_result = ExecutePushInternal(curr_chunk, op_idx + 1);
		} while (finalize_result != OperatorFinalizeResultType::FINISHED &&
		         push_result != OperatorResultType::FINISHED);

		if (push_result == OperatorResultType::FINISHED) {
			break;
		}
	}
}

void PipelineExecutor::PushFinalize() {
	if (finalized) {
		throw InternalException("Calling PushFinalize on a pipeline that has been finalized already");
	}
	finalized = true;
	// flush all caching operators
	// note that even if an operator has finished, we might still need to flush caches AFTER
	// that operator e.g. if we have SOURCE -> LIMIT -> CROSS_PRODUCT -> SINK, if the
	// LIMIT reports no more rows will be passed on we still need to flush caches from the CROSS_PRODUCT
	D_ASSERT(in_process_operators.empty());

	FlushCachingOperatorsPush();

	D_ASSERT(local_sink_state);
	// run the combine for the sink
	pipeline.sink->Combine(context, *pipeline.sink->sink_state, *local_sink_state);

	// flush all query profiler info
	for (idx_t i = 0; i < intermediate_states.size(); i++) {
		intermediate_states[i]->Finalize(pipeline.operators[i], context);
	}
	pipeline.executor.Flush(thread);
	local_sink_state.reset();
}

void PipelineExecutor::ExecutePull(DataChunk &result) {
	if (IsFinished()) {
		return;
	}
	auto &executor = pipeline.executor;
	try {
		D_ASSERT(!pipeline.sink);
		auto &source_chunk = pipeline.operators.empty() ? result : *intermediate_chunks[0];
		while (result.size() == 0) {
			if (source_empty) {
				FlushCachingOperatorsPull(result);
				break;
			}

			if (in_process_operators.empty()) {
				source_chunk.Reset();
				FetchFromSource(source_chunk);

				if (source_chunk.size() == 0) {
					source_empty = true;
					continue;
				}
			}

			if (!pipeline.operators.empty()) {
				auto state = Execute(source_chunk, result);
				if (state == OperatorResultType::FINISHED) {
					break;
				}
			}
		}
	} catch (const Exception &ex) { // LCOV_EXCL_START
		if (executor.HasError()) {
			executor.ThrowException();
		}
		throw;
	} catch (std::exception &ex) {
		if (executor.HasError()) {
			executor.ThrowException();
		}
		throw;
	} catch (...) {
		if (executor.HasError()) {
			executor.ThrowException();
		}
		throw;
	} // LCOV_EXCL_STOP
}

void PipelineExecutor::PullFinalize() {
	if (finalized) {
		throw InternalException("Calling PullFinalize on a pipeline that has been finalized already");
	}
	finalized = true;
	pipeline.executor.Flush(thread);
}

void PipelineExecutor::GoToSource(idx_t &current_idx, idx_t initial_idx) {
	// we go back to the first operator (the source)
	current_idx = initial_idx;
	if (!in_process_operators.empty()) {
		// ... UNLESS there is an in process operator
		// if there is an in-process operator, we start executing at the latest one
		// for example, if we have a join operator that has tuples left, we first need to emit those tuples
		current_idx = in_process_operators.top();
		in_process_operators.pop();
	}
	D_ASSERT(current_idx >= initial_idx);
}

OperatorResultType PipelineExecutor::Execute(DataChunk &input, DataChunk &result, idx_t initial_idx) {
	if (input.size() == 0) { // LCOV_EXCL_START
		return OperatorResultType::NEED_MORE_INPUT;
	} // LCOV_EXCL_STOP
	D_ASSERT(!pipeline.operators.empty());

	idx_t current_idx;
	GoToSource(current_idx, initial_idx);
	if (current_idx == initial_idx) {
		current_idx++;
	}
	if (current_idx > pipeline.operators.size()) {
		result.Reference(input);
		return OperatorResultType::NEED_MORE_INPUT;
	}
	while (true) {
		if (context.client.interrupted) {
			throw InterruptException();
		}
		// now figure out where to put the chunk
		// if current_idx is the last possible index (>= operators.size()) we write to the result
		// otherwise we write to an intermediate chunk
		auto current_intermediate = current_idx;
		auto &current_chunk =
		    current_intermediate >= intermediate_chunks.size() ? result : *intermediate_chunks[current_intermediate];
		current_chunk.Reset();
		if (current_idx == initial_idx) {
			// we went back to the source: we need more input
			return OperatorResultType::NEED_MORE_INPUT;
		} else {
			auto &prev_chunk =
			    current_intermediate == initial_idx + 1 ? input : *intermediate_chunks[current_intermediate - 1];
			auto operator_idx = current_idx - 1;
			auto current_operator = pipeline.operators[operator_idx];

			// if current_idx > source_idx, we pass the previous' operators output through the Execute of the current
			// operator
			StartOperator(current_operator);
			auto result = current_operator->Execute(context, prev_chunk, current_chunk, *current_operator->op_state,
			                                        *intermediate_states[current_intermediate - 1]);
			EndOperator(current_operator, &current_chunk);
			if (result == OperatorResultType::HAVE_MORE_OUTPUT) {
				// more data remains in this operator
				// push in-process marker
				in_process_operators.push(current_idx);
			} else if (result == OperatorResultType::FINISHED) {
				D_ASSERT(current_chunk.size() == 0);
				FinishProcessing(current_idx);
				return OperatorResultType::FINISHED;
			}
			current_chunk.Verify();
		}

		if (current_chunk.size() == 0) {
			// no output from this operator!
			if (current_idx == initial_idx) {
				// if we got no output from the scan, we are done
				break;
			} else {
				// if we got no output from an intermediate op
				// we go back and try to pull data from the source again
				GoToSource(current_idx, initial_idx);
				continue;
			}
		} else {
			// we got output! continue to the next operator
			current_idx++;
			if (current_idx > pipeline.operators.size()) {
				// if we got output and are at the last operator, we are finished executing for this output chunk
				// return the data and push it into the chunk
				break;
			}
		}
	}
	return in_process_operators.empty() ? OperatorResultType::NEED_MORE_INPUT : OperatorResultType::HAVE_MORE_OUTPUT;
}

void PipelineExecutor::FetchFromSource(DataChunk &result) {
	StartOperator(pipeline.source);
	pipeline.source->GetData(context, result, *pipeline.source_state, *local_source_state);
	if (result.size() != 0 && requires_batch_index) {
		auto next_batch_index =
		    pipeline.source->GetBatchIndex(context, result, *pipeline.source_state, *local_source_state);
		next_batch_index += pipeline.base_batch_index;
		D_ASSERT(local_sink_state->batch_index <= next_batch_index ||
		         local_sink_state->batch_index == DConstants::INVALID_INDEX);
		local_sink_state->batch_index = next_batch_index;
	}
	EndOperator(pipeline.source, &result);
}

void PipelineExecutor::InitializeChunk(DataChunk &chunk) {
	PhysicalOperator *last_op = pipeline.operators.empty() ? pipeline.source : pipeline.operators.back();
	chunk.Initialize(Allocator::DefaultAllocator(), last_op->GetTypes());
}

void PipelineExecutor::StartOperator(PhysicalOperator *op) {
	if (context.client.interrupted) {
		throw InterruptException();
	}
	context.thread.profiler.StartOperator(op);
}

void PipelineExecutor::EndOperator(PhysicalOperator *op, DataChunk *chunk) {
	context.thread.profiler.EndOperator(chunk);

	if (chunk) {
		chunk->Verify();
	}
}

} // namespace duckdb



namespace duckdb {

PipelineFinishEvent::PipelineFinishEvent(shared_ptr<Pipeline> pipeline_p) : BasePipelineEvent(move(pipeline_p)) {
}

void PipelineFinishEvent::Schedule() {
}

void PipelineFinishEvent::FinishEvent() {
	pipeline->Finalize(*this);
}

} // namespace duckdb




namespace duckdb {

PipelineInitializeEvent::PipelineInitializeEvent(shared_ptr<Pipeline> pipeline_p)
    : BasePipelineEvent(move(pipeline_p)) {
}

class PipelineInitializeTask : public ExecutorTask {
public:
	explicit PipelineInitializeTask(Pipeline &pipeline_p, shared_ptr<Event> event_p)
	    : ExecutorTask(pipeline_p.executor), pipeline(pipeline_p), event(move(event_p)) {
	}

	Pipeline &pipeline;
	shared_ptr<Event> event;

public:
	TaskExecutionResult ExecuteTask(TaskExecutionMode mode) override {
		pipeline.ResetSink();
		event->FinishTask();
		return TaskExecutionResult::TASK_FINISHED;
	}
};

void PipelineInitializeEvent::Schedule() {
	// needs to spawn a task to get the chain of tasks for the query plan going
	vector<unique_ptr<Task>> tasks;
	tasks.push_back(make_unique<PipelineInitializeTask>(*pipeline, shared_from_this()));
	SetTasks(move(tasks));
}

void PipelineInitializeEvent::FinishEvent() {
}

} // namespace duckdb






#ifndef DUCKDB_NO_THREADS



#else
#include <queue>
#endif

namespace duckdb {

struct SchedulerThread {
#ifndef DUCKDB_NO_THREADS
	explicit SchedulerThread(unique_ptr<thread> thread_p) : internal_thread(move(thread_p)) {
	}

	unique_ptr<thread> internal_thread;
#endif
};

#ifndef DUCKDB_NO_THREADS
typedef duckdb_moodycamel::ConcurrentQueue<unique_ptr<Task>> concurrent_queue_t;
typedef duckdb_moodycamel::LightweightSemaphore lightweight_semaphore_t;

struct ConcurrentQueue {
	concurrent_queue_t q;
	lightweight_semaphore_t semaphore;

	void Enqueue(ProducerToken &token, unique_ptr<Task> task);
	bool DequeueFromProducer(ProducerToken &token, unique_ptr<Task> &task);
};

struct QueueProducerToken {
	explicit QueueProducerToken(ConcurrentQueue &queue) : queue_token(queue.q) {
	}

	duckdb_moodycamel::ProducerToken queue_token;
};

void ConcurrentQueue::Enqueue(ProducerToken &token, unique_ptr<Task> task) {
	lock_guard<mutex> producer_lock(token.producer_lock);
	if (q.enqueue(token.token->queue_token, move(task))) {
		semaphore.signal();
	} else {
		throw InternalException("Could not schedule task!");
	}
}

bool ConcurrentQueue::DequeueFromProducer(ProducerToken &token, unique_ptr<Task> &task) {
	lock_guard<mutex> producer_lock(token.producer_lock);
	return q.try_dequeue_from_producer(token.token->queue_token, task);
}

#else
struct ConcurrentQueue {
	std::queue<std::unique_ptr<Task>> q;
	mutex qlock;

	void Enqueue(ProducerToken &token, unique_ptr<Task> task);
	bool DequeueFromProducer(ProducerToken &token, unique_ptr<Task> &task);
};

void ConcurrentQueue::Enqueue(ProducerToken &token, unique_ptr<Task> task) {
	lock_guard<mutex> lock(qlock);
	q.push(move(task));
}

bool ConcurrentQueue::DequeueFromProducer(ProducerToken &token, unique_ptr<Task> &task) {
	lock_guard<mutex> lock(qlock);
	if (q.empty()) {
		return false;
	}
	task = move(q.front());
	q.pop();
	return true;
}

struct QueueProducerToken {
	QueueProducerToken(ConcurrentQueue &queue) {
	}
};
#endif

ProducerToken::ProducerToken(TaskScheduler &scheduler, unique_ptr<QueueProducerToken> token)
    : scheduler(scheduler), token(move(token)) {
}

ProducerToken::~ProducerToken() {
}

TaskScheduler::TaskScheduler(DatabaseInstance &db) : db(db), queue(make_unique<ConcurrentQueue>()) {
}

TaskScheduler::~TaskScheduler() {
#ifndef DUCKDB_NO_THREADS
	SetThreadsInternal(1);
#endif
}

TaskScheduler &TaskScheduler::GetScheduler(ClientContext &context) {
	return TaskScheduler::GetScheduler(DatabaseInstance::GetDatabase(context));
}

TaskScheduler &TaskScheduler::GetScheduler(DatabaseInstance &db) {
	return db.GetScheduler();
}

unique_ptr<ProducerToken> TaskScheduler::CreateProducer() {
	auto token = make_unique<QueueProducerToken>(*queue);
	return make_unique<ProducerToken>(*this, move(token));
}

void TaskScheduler::ScheduleTask(ProducerToken &token, unique_ptr<Task> task) {
	// Enqueue a task for the given producer token and signal any sleeping threads
	queue->Enqueue(token, move(task));
}

bool TaskScheduler::GetTaskFromProducer(ProducerToken &token, unique_ptr<Task> &task) {
	return queue->DequeueFromProducer(token, task);
}

void TaskScheduler::ExecuteForever(atomic<bool> *marker) {
#ifndef DUCKDB_NO_THREADS
	unique_ptr<Task> task;
	// loop until the marker is set to false
	while (*marker) {
		// wait for a signal with a timeout
		queue->semaphore.wait();
		if (queue->q.try_dequeue(task)) {
			task->Execute(TaskExecutionMode::PROCESS_ALL);
			task.reset();
		}
	}
#else
	throw NotImplementedException("DuckDB was compiled without threads! Background thread loop is not allowed.");
#endif
}

idx_t TaskScheduler::ExecuteTasks(atomic<bool> *marker, idx_t max_tasks) {
#ifndef DUCKDB_NO_THREADS
	idx_t completed_tasks = 0;
	// loop until the marker is set to false
	while (*marker && completed_tasks < max_tasks) {
		unique_ptr<Task> task;
		if (!queue->q.try_dequeue(task)) {
			return completed_tasks;
		}
		task->Execute(TaskExecutionMode::PROCESS_ALL);
		task.reset();
		completed_tasks++;
	}
	return completed_tasks;
#else
	throw NotImplementedException("DuckDB was compiled without threads! Background thread loop is not allowed.");
#endif
}

void TaskScheduler::ExecuteTasks(idx_t max_tasks) {
#ifndef DUCKDB_NO_THREADS
	unique_ptr<Task> task;
	for (idx_t i = 0; i < max_tasks; i++) {
		queue->semaphore.wait(TASK_TIMEOUT_USECS);
		if (!queue->q.try_dequeue(task)) {
			return;
		}
		try {
			task->Execute(TaskExecutionMode::PROCESS_ALL);
			task.reset();
		} catch (...) {
			return;
		}
	}
#else
	throw NotImplementedException("DuckDB was compiled without threads! Background thread loop is not allowed.");
#endif
}

#ifndef DUCKDB_NO_THREADS
static void ThreadExecuteTasks(TaskScheduler *scheduler, atomic<bool> *marker) {
	scheduler->ExecuteForever(marker);
}
#endif

int32_t TaskScheduler::NumberOfThreads() {
	lock_guard<mutex> t(thread_lock);
	auto &config = DBConfig::GetConfig(db);
	return threads.size() + config.options.external_threads + 1;
}

void TaskScheduler::SetThreads(int32_t n) {
#ifndef DUCKDB_NO_THREADS
	lock_guard<mutex> t(thread_lock);
	if (n < 1) {
		throw SyntaxException("Must have at least 1 thread!");
	}
	SetThreadsInternal(n);
#else
	if (n != 1) {
		throw NotImplementedException("DuckDB was compiled without threads! Setting threads > 1 is not allowed.");
	}
#endif
}

void TaskScheduler::Signal(idx_t n) {
#ifndef DUCKDB_NO_THREADS
	queue->semaphore.signal(n);
#endif
}

void TaskScheduler::SetThreadsInternal(int32_t n) {
#ifndef DUCKDB_NO_THREADS
	if (threads.size() == idx_t(n - 1)) {
		return;
	}
	idx_t new_thread_count = n - 1;
	if (threads.size() > new_thread_count) {
		// we are reducing the number of threads: clear all threads first
		for (idx_t i = 0; i < threads.size(); i++) {
			*markers[i] = false;
		}
		Signal(threads.size());
		// now join the threads to ensure they are fully stopped before erasing them
		for (idx_t i = 0; i < threads.size(); i++) {
			threads[i]->internal_thread->join();
		}
		// erase the threads/markers
		threads.clear();
		markers.clear();
	}
	if (threads.size() < new_thread_count) {
		// we are increasing the number of threads: launch them and run tasks on them
		idx_t create_new_threads = new_thread_count - threads.size();
		for (idx_t i = 0; i < create_new_threads; i++) {
			// launch a thread and assign it a cancellation marker
			auto marker = unique_ptr<atomic<bool>>(new atomic<bool>(true));
			auto worker_thread = make_unique<thread>(ThreadExecuteTasks, this, marker.get());
			auto thread_wrapper = make_unique<SchedulerThread>(move(worker_thread));

			threads.push_back(move(thread_wrapper));
			markers.push_back(move(marker));
		}
	}
#endif
}

} // namespace duckdb




namespace duckdb {

ThreadContext::ThreadContext(ClientContext &context) : profiler(QueryProfiler::Get(context).IsEnabled()) {
}

} // namespace duckdb




namespace duckdb {

void BaseExpression::Print() const {
	Printer::Print(ToString());
}

string BaseExpression::GetName() const {
	return !alias.empty() ? alias : ToString();
}

bool BaseExpression::Equals(const BaseExpression *other) const {
	if (!other) {
		return false;
	}
	if (this->expression_class != other->expression_class || this->type != other->type) {
		return false;
	}
	return true;
}

void BaseExpression::Verify() const {
}

} // namespace duckdb







namespace duckdb {

ColumnDefinition::ColumnDefinition(string name_p, LogicalType type_p) : name(move(name_p)), type(move(type_p)) {
}

ColumnDefinition::ColumnDefinition(string name_p, LogicalType type_p, unique_ptr<ParsedExpression> expression,
                                   TableColumnType category)
    : name(move(name_p)), type(move(type_p)), category(category) {
	switch (category) {
	case TableColumnType::STANDARD: {
		default_value = move(expression);
		break;
	}
	case TableColumnType::GENERATED: {
		generated_expression = move(expression);
		break;
	}
	default: {
		throw InternalException("Type not implemented for TableColumnType");
	}
	}
}

ColumnDefinition ColumnDefinition::Copy() const {
	ColumnDefinition copy(name, type);
	copy.oid = oid;
	copy.storage_oid = storage_oid;
	copy.SetDefaultValue(default_value ? default_value->Copy() : nullptr);
	copy.generated_expression = generated_expression ? generated_expression->Copy() : nullptr;
	copy.compression_type = compression_type;
	copy.category = category;
	return copy;
}

void ColumnDefinition::Serialize(Serializer &serializer) const {
	FieldWriter writer(serializer);
	writer.WriteString(name);
	writer.WriteSerializable(type);
	if (Generated()) {
		writer.WriteOptional(generated_expression);
	} else {
		writer.WriteOptional(default_value);
	}
	writer.WriteField<TableColumnType>(category);
	writer.Finalize();
}

ColumnDefinition ColumnDefinition::Deserialize(Deserializer &source) {
	FieldReader reader(source);
	auto column_name = reader.ReadRequired<string>();
	auto column_type = reader.ReadRequiredSerializable<LogicalType, LogicalType>();
	auto expression = reader.ReadOptional<ParsedExpression>(nullptr);
	auto category = reader.ReadField<TableColumnType>(TableColumnType::STANDARD);
	reader.Finalize();

	switch (category) {
	case TableColumnType::STANDARD:
		return ColumnDefinition(column_name, column_type, move(expression), TableColumnType::STANDARD);
	case TableColumnType::GENERATED:
		return ColumnDefinition(column_name, column_type, move(expression), TableColumnType::GENERATED);
	default:
		throw NotImplementedException("Type not implemented for TableColumnType");
	}
}

const unique_ptr<ParsedExpression> &ColumnDefinition::DefaultValue() const {
	return default_value;
}

void ColumnDefinition::SetDefaultValue(unique_ptr<ParsedExpression> default_value) {
	this->default_value = move(default_value);
}

const LogicalType &ColumnDefinition::Type() const {
	return type;
}

LogicalType &ColumnDefinition::TypeMutable() {
	return type;
}

void ColumnDefinition::SetType(const LogicalType &type) {
	this->type = type;
}

const string &ColumnDefinition::Name() const {
	return name;
}

void ColumnDefinition::SetName(const string &name) {
	this->name = name;
}

const duckdb::CompressionType &ColumnDefinition::CompressionType() const {
	return compression_type;
}

void ColumnDefinition::SetCompressionType(duckdb::CompressionType compression_type) {
	this->compression_type = compression_type;
}

const storage_t &ColumnDefinition::StorageOid() const {
	return storage_oid;
}

LogicalIndex ColumnDefinition::Logical() const {
	return LogicalIndex(oid);
}

PhysicalIndex ColumnDefinition::Physical() const {
	return PhysicalIndex(storage_oid);
}

void ColumnDefinition::SetStorageOid(storage_t storage_oid) {
	this->storage_oid = storage_oid;
}

const column_t &ColumnDefinition::Oid() const {
	return oid;
}

void ColumnDefinition::SetOid(column_t oid) {
	this->oid = oid;
}

const TableColumnType &ColumnDefinition::Category() const {
	return category;
}

bool ColumnDefinition::Generated() const {
	return category == TableColumnType::GENERATED;
}

//===--------------------------------------------------------------------===//
// Generated Columns (VIRTUAL)
//===--------------------------------------------------------------------===//

static void VerifyColumnRefs(ParsedExpression &expr) {
	if (expr.type == ExpressionType::COLUMN_REF) {
		auto &column_ref = (ColumnRefExpression &)expr;
		if (column_ref.IsQualified()) {
			throw ParserException(
			    "Qualified (tbl.name) column references are not allowed inside of generated column expressions");
		}
	}
	ParsedExpressionIterator::EnumerateChildren(
	    expr, [&](const ParsedExpression &child) { VerifyColumnRefs((ParsedExpression &)child); });
}

static void InnerGetListOfDependencies(ParsedExpression &expr, vector<string> &dependencies) {
	if (expr.type == ExpressionType::COLUMN_REF) {
		auto columnref = (ColumnRefExpression &)expr;
		auto &name = columnref.GetColumnName();
		dependencies.push_back(name);
	}
	ParsedExpressionIterator::EnumerateChildren(expr, [&](const ParsedExpression &child) {
		InnerGetListOfDependencies((ParsedExpression &)child, dependencies);
	});
}

void ColumnDefinition::GetListOfDependencies(vector<string> &dependencies) const {
	D_ASSERT(Generated());
	InnerGetListOfDependencies(*generated_expression, dependencies);
}

string ColumnDefinition::GetName() const {
	return name;
}

LogicalType ColumnDefinition::GetType() const {
	return type;
}

void ColumnDefinition::SetGeneratedExpression(unique_ptr<ParsedExpression> expression) {
	category = TableColumnType::GENERATED;

	if (expression->HasSubquery()) {
		throw ParserException("Expression of generated column \"%s\" contains a subquery, which isn't allowed", name);
	}

	VerifyColumnRefs(*expression);
	if (type.id() == LogicalTypeId::ANY) {
		generated_expression = move(expression);
		return;
	}
	// Always wrap the expression in a cast, that way we can always update the cast when we change the type
	// Except if the type is LogicalType::ANY (no type specified)
	generated_expression = make_unique_base<ParsedExpression, CastExpression>(type, move(expression));
}

void ColumnDefinition::ChangeGeneratedExpressionType(const LogicalType &type) {
	D_ASSERT(Generated());
	// First time the type is set, add a cast around the expression
	D_ASSERT(this->type.id() == LogicalTypeId::ANY);
	generated_expression = make_unique_base<ParsedExpression, CastExpression>(type, move(generated_expression));
	// Every generated expression should be wrapped in a cast on creation
	// D_ASSERT(generated_expression->type == ExpressionType::OPERATOR_CAST);
	// auto &cast_expr = (CastExpression &)*generated_expression;
	// auto base_expr = move(cast_expr.child);
	// generated_expression = make_unique_base<ParsedExpression, CastExpression>(type, move(base_expr));
}

const ParsedExpression &ColumnDefinition::GeneratedExpression() const {
	D_ASSERT(Generated());
	return *generated_expression;
}

ParsedExpression &ColumnDefinition::GeneratedExpressionMutable() {
	D_ASSERT(Generated());
	return *generated_expression;
}

} // namespace duckdb




namespace duckdb {

ColumnList::ColumnList(bool allow_duplicate_names) : allow_duplicate_names(allow_duplicate_names) {
}
void ColumnList::AddColumn(ColumnDefinition column) {
	auto oid = columns.size();
	if (!column.Generated()) {
		column.SetStorageOid(physical_columns.size());
		physical_columns.push_back(oid);
	} else {
		column.SetStorageOid(DConstants::INVALID_INDEX);
	}
	column.SetOid(columns.size());
	AddToNameMap(column);
	columns.push_back(move(column));
}

void ColumnList::Finalize() {
	// add the "rowid" alias, if there is no rowid column specified in the table
	if (name_map.find("rowid") == name_map.end()) {
		name_map["rowid"] = COLUMN_IDENTIFIER_ROW_ID;
	}
}

void ColumnList::AddToNameMap(ColumnDefinition &col) {
	if (allow_duplicate_names) {
		idx_t index = 1;
		string base_name = col.Name();
		while (name_map.find(col.Name()) != name_map.end()) {
			col.SetName(base_name + ":" + to_string(index++));
		}
	} else {
		if (name_map.find(col.Name()) != name_map.end()) {
			throw CatalogException("Column with name %s already exists!", col.Name());
		}
	}
	name_map[col.Name()] = col.Oid();
}

ColumnDefinition &ColumnList::GetColumnMutable(LogicalIndex logical) {
	if (logical.index >= columns.size()) {
		throw InternalException("Logical column index %lld out of range", logical.index);
	}
	return columns[logical.index];
}

ColumnDefinition &ColumnList::GetColumnMutable(PhysicalIndex physical) {
	if (physical.index >= physical_columns.size()) {
		throw InternalException("Physical column index %lld out of range", physical.index);
	}
	auto logical_index = physical_columns[physical.index];
	D_ASSERT(logical_index < columns.size());
	return columns[logical_index];
}

ColumnDefinition &ColumnList::GetColumnMutable(const string &name) {
	auto entry = name_map.find(name);
	if (entry == name_map.end()) {
		throw InternalException("Column with name \"%s\" does not exist", name);
	}
	auto logical_index = entry->second;
	D_ASSERT(logical_index < columns.size());
	return columns[logical_index];
}

const ColumnDefinition &ColumnList::GetColumn(LogicalIndex logical) const {
	if (logical.index >= columns.size()) {
		throw InternalException("Logical column index %lld out of range", logical.index);
	}
	return columns[logical.index];
}

const ColumnDefinition &ColumnList::GetColumn(PhysicalIndex physical) const {
	if (physical.index >= physical_columns.size()) {
		throw InternalException("Physical column index %lld out of range", physical.index);
	}
	auto logical_index = physical_columns[physical.index];
	D_ASSERT(logical_index < columns.size());
	return columns[logical_index];
}

const ColumnDefinition &ColumnList::GetColumn(const string &name) const {
	auto entry = name_map.find(name);
	if (entry == name_map.end()) {
		throw InternalException("Column with name \"%s\" does not exist", name);
	}
	auto logical_index = entry->second;
	D_ASSERT(logical_index < columns.size());
	return columns[logical_index];
}

bool ColumnList::ColumnExists(const string &name) const {
	auto entry = name_map.find(name);
	return entry != name_map.end();
}

PhysicalIndex ColumnList::LogicalToPhysical(LogicalIndex logical) const {
	auto &column = GetColumn(logical);
	if (column.Generated()) {
		throw InternalException("Column at position %d is not a physical column", logical.index);
	}
	return column.Physical();
}

LogicalIndex ColumnList::PhysicalToLogical(PhysicalIndex index) const {
	auto &column = GetColumn(index);
	return column.Logical();
}

LogicalIndex ColumnList::GetColumnIndex(string &column_name) const {
	auto entry = name_map.find(column_name);
	if (entry == name_map.end()) {
		return LogicalIndex(DConstants::INVALID_INDEX);
	}
	if (entry->second == COLUMN_IDENTIFIER_ROW_ID) {
		column_name = "rowid";
		return LogicalIndex(COLUMN_IDENTIFIER_ROW_ID);
	}
	column_name = columns[entry->second].Name();
	return LogicalIndex(entry->second);
}

ColumnList ColumnList::Copy() const {
	ColumnList result(allow_duplicate_names);
	for (auto &col : columns) {
		result.AddColumn(col.Copy());
	}
	return result;
}

void ColumnList::Serialize(FieldWriter &writer) const {
	writer.WriteRegularSerializableList(columns);
}

ColumnList ColumnList::Deserialize(FieldReader &reader) {
	ColumnList result;
	auto columns = reader.ReadRequiredSerializableList<ColumnDefinition, ColumnDefinition>();
	for (auto &col : columns) {
		result.AddColumn(move(col));
	}
	return result;
}

ColumnList::ColumnListIterator ColumnList::Logical() const {
	return ColumnListIterator(*this, false);
}

ColumnList::ColumnListIterator ColumnList::Physical() const {
	return ColumnListIterator(*this, true);
}

} // namespace duckdb







namespace duckdb {

Constraint::Constraint(ConstraintType type) : type(type) {
}

Constraint::~Constraint() {
}

void Constraint::Serialize(Serializer &serializer) const {
	FieldWriter writer(serializer);
	writer.WriteField<ConstraintType>(type);
	Serialize(writer);
	writer.Finalize();
}

unique_ptr<Constraint> Constraint::Deserialize(Deserializer &source) {
	FieldReader reader(source);
	auto type = reader.ReadRequired<ConstraintType>();
	unique_ptr<Constraint> result;
	switch (type) {
	case ConstraintType::NOT_NULL:
		result = NotNullConstraint::Deserialize(reader);
		break;
	case ConstraintType::CHECK:
		result = CheckConstraint::Deserialize(reader);
		break;
	case ConstraintType::UNIQUE:
		result = UniqueConstraint::Deserialize(reader);
		break;
	case ConstraintType::FOREIGN_KEY:
		result = ForeignKeyConstraint::Deserialize(reader);
		break;
	default:
		throw InternalException("Unrecognized constraint type for serialization");
	}
	reader.Finalize();
	return result;
}

void Constraint::Print() const {
	Printer::Print(ToString());
}

} // namespace duckdb




namespace duckdb {

CheckConstraint::CheckConstraint(unique_ptr<ParsedExpression> expression)
    : Constraint(ConstraintType::CHECK), expression(move(expression)) {
}

string CheckConstraint::ToString() const {
	return "CHECK(" + expression->ToString() + ")";
}

unique_ptr<Constraint> CheckConstraint::Copy() const {
	return make_unique<CheckConstraint>(expression->Copy());
}

void CheckConstraint::Serialize(FieldWriter &writer) const {
	writer.WriteSerializable(*expression);
}

unique_ptr<Constraint> CheckConstraint::Deserialize(FieldReader &source) {
	auto expression = source.ReadRequiredSerializable<ParsedExpression>();
	return make_unique<CheckConstraint>(move(expression));
}

} // namespace duckdb






namespace duckdb {

ForeignKeyConstraint::ForeignKeyConstraint(vector<string> pk_columns, vector<string> fk_columns, ForeignKeyInfo info)
    : Constraint(ConstraintType::FOREIGN_KEY), pk_columns(move(pk_columns)), fk_columns(move(fk_columns)),
      info(move(info)) {
}

string ForeignKeyConstraint::ToString() const {
	if (info.type == ForeignKeyType::FK_TYPE_FOREIGN_KEY_TABLE) {
		string base = "FOREIGN KEY (";

		for (idx_t i = 0; i < fk_columns.size(); i++) {
			if (i > 0) {
				base += ", ";
			}
			base += KeywordHelper::WriteOptionallyQuoted(fk_columns[i]);
		}
		base += ") REFERENCES ";
		if (!info.schema.empty()) {
			base += info.schema;
			base += ".";
		}
		base += info.table;
		base += "(";

		for (idx_t i = 0; i < pk_columns.size(); i++) {
			if (i > 0) {
				base += ", ";
			}
			base += KeywordHelper::WriteOptionallyQuoted(pk_columns[i]);
		}
		base += ")";

		return base;
	}

	return "";
}

unique_ptr<Constraint> ForeignKeyConstraint::Copy() const {
	return make_unique<ForeignKeyConstraint>(pk_columns, fk_columns, info);
}

void ForeignKeyConstraint::Serialize(FieldWriter &writer) const {
	D_ASSERT(pk_columns.size() <= NumericLimits<uint32_t>::Maximum());
	writer.WriteList<string>(pk_columns);
	D_ASSERT(fk_columns.size() <= NumericLimits<uint32_t>::Maximum());
	writer.WriteList<string>(fk_columns);
	writer.WriteField<ForeignKeyType>(info.type);
	writer.WriteString(info.schema);
	writer.WriteString(info.table);
	writer.WriteIndexList<PhysicalIndex>(info.pk_keys);
	writer.WriteIndexList<PhysicalIndex>(info.fk_keys);
}

unique_ptr<Constraint> ForeignKeyConstraint::Deserialize(FieldReader &source) {
	ForeignKeyInfo read_info;
	auto pk_columns = source.ReadRequiredList<string>();
	auto fk_columns = source.ReadRequiredList<string>();
	read_info.type = source.ReadRequired<ForeignKeyType>();
	read_info.schema = source.ReadRequired<string>();
	read_info.table = source.ReadRequired<string>();
	read_info.pk_keys = source.ReadRequiredIndexList<PhysicalIndex>();
	read_info.fk_keys = source.ReadRequiredIndexList<PhysicalIndex>();

	// column list parsed constraint
	return make_unique<ForeignKeyConstraint>(pk_columns, fk_columns, move(read_info));
}

} // namespace duckdb




namespace duckdb {

NotNullConstraint::NotNullConstraint(LogicalIndex index) : Constraint(ConstraintType::NOT_NULL), index(index) {
}

NotNullConstraint::~NotNullConstraint() {
}

string NotNullConstraint::ToString() const {
	return "NOT NULL";
}

unique_ptr<Constraint> NotNullConstraint::Copy() const {
	return make_unique<NotNullConstraint>(index);
}

void NotNullConstraint::Serialize(FieldWriter &writer) const {
	writer.WriteField<idx_t>(index.index);
}

unique_ptr<Constraint> NotNullConstraint::Deserialize(FieldReader &source) {
	auto index = source.ReadRequired<idx_t>();
	return make_unique_base<Constraint, NotNullConstraint>(LogicalIndex(index));
}

} // namespace duckdb






namespace duckdb {

UniqueConstraint::UniqueConstraint(LogicalIndex index, bool is_primary_key)
    : Constraint(ConstraintType::UNIQUE), index(index), is_primary_key(is_primary_key) {
}
UniqueConstraint::UniqueConstraint(vector<string> columns, bool is_primary_key)
    : Constraint(ConstraintType::UNIQUE), index(DConstants::INVALID_INDEX), columns(move(columns)),
      is_primary_key(is_primary_key) {
}

string UniqueConstraint::ToString() const {
	string base = is_primary_key ? "PRIMARY KEY(" : "UNIQUE(";
	for (idx_t i = 0; i < columns.size(); i++) {
		if (i > 0) {
			base += ", ";
		}
		base += KeywordHelper::WriteOptionallyQuoted(columns[i]);
	}
	return base + ")";
}

unique_ptr<Constraint> UniqueConstraint::Copy() const {
	if (index.index == DConstants::INVALID_INDEX) {
		return make_unique<UniqueConstraint>(columns, is_primary_key);
	} else {
		auto result = make_unique<UniqueConstraint>(index, is_primary_key);
		result->columns = columns;
		return move(result);
	}
}

void UniqueConstraint::Serialize(FieldWriter &writer) const {
	writer.WriteField<bool>(is_primary_key);
	writer.WriteField<uint64_t>(index.index);
	D_ASSERT(columns.size() <= NumericLimits<uint32_t>::Maximum());
	writer.WriteList<string>(columns);
}

unique_ptr<Constraint> UniqueConstraint::Deserialize(FieldReader &source) {
	auto is_primary_key = source.ReadRequired<bool>();
	auto index = source.ReadRequired<uint64_t>();
	auto columns = source.ReadRequiredList<string>();

	if (index != DConstants::INVALID_INDEX) {
		// single column parsed constraint
		auto result = make_unique<UniqueConstraint>(LogicalIndex(index), is_primary_key);
		result->columns = move(columns);
		return move(result);
	} else {
		// column list parsed constraint
		return make_unique<UniqueConstraint>(move(columns), is_primary_key);
	}
}

} // namespace duckdb



namespace duckdb {

BetweenExpression::BetweenExpression(unique_ptr<ParsedExpression> input_p, unique_ptr<ParsedExpression> lower_p,
                                     unique_ptr<ParsedExpression> upper_p)
    : ParsedExpression(ExpressionType::COMPARE_BETWEEN, ExpressionClass::BETWEEN), input(move(input_p)),
      lower(move(lower_p)), upper(move(upper_p)) {
}

string BetweenExpression::ToString() const {
	return ToString<BetweenExpression, ParsedExpression>(*this);
}

bool BetweenExpression::Equals(const BetweenExpression *a, const BetweenExpression *b) {
	if (!a->input->Equals(b->input.get())) {
		return false;
	}
	if (!a->lower->Equals(b->lower.get())) {
		return false;
	}
	if (!a->upper->Equals(b->upper.get())) {
		return false;
	}
	return true;
}

unique_ptr<ParsedExpression> BetweenExpression::Copy() const {
	auto copy = make_unique<BetweenExpression>(input->Copy(), lower->Copy(), upper->Copy());
	copy->CopyProperties(*this);
	return move(copy);
}

void BetweenExpression::Serialize(FieldWriter &writer) const {
	writer.WriteSerializable(*input);
	writer.WriteSerializable(*lower);
	writer.WriteSerializable(*upper);
}

unique_ptr<ParsedExpression> BetweenExpression::Deserialize(ExpressionType type, FieldReader &source) {
	auto input = source.ReadRequiredSerializable<ParsedExpression>();
	auto lower = source.ReadRequiredSerializable<ParsedExpression>();
	auto upper = source.ReadRequiredSerializable<ParsedExpression>();
	return make_unique<BetweenExpression>(move(input), move(lower), move(upper));
}

} // namespace duckdb





namespace duckdb {

CaseExpression::CaseExpression() : ParsedExpression(ExpressionType::CASE_EXPR, ExpressionClass::CASE) {
}

string CaseExpression::ToString() const {
	return ToString<CaseExpression, ParsedExpression>(*this);
}

bool CaseExpression::Equals(const CaseExpression *a, const CaseExpression *b) {
	if (a->case_checks.size() != b->case_checks.size()) {
		return false;
	}
	for (idx_t i = 0; i < a->case_checks.size(); i++) {
		if (!a->case_checks[i].when_expr->Equals(b->case_checks[i].when_expr.get())) {
			return false;
		}
		if (!a->case_checks[i].then_expr->Equals(b->case_checks[i].then_expr.get())) {
			return false;
		}
	}
	if (!a->else_expr->Equals(b->else_expr.get())) {
		return false;
	}
	return true;
}

unique_ptr<ParsedExpression> CaseExpression::Copy() const {
	auto copy = make_unique<CaseExpression>();
	copy->CopyProperties(*this);
	for (auto &check : case_checks) {
		CaseCheck new_check;
		new_check.when_expr = check.when_expr->Copy();
		new_check.then_expr = check.then_expr->Copy();
		copy->case_checks.push_back(move(new_check));
	}
	copy->else_expr = else_expr->Copy();
	return move(copy);
}

void CaseExpression::Serialize(FieldWriter &writer) const {
	auto &serializer = writer.GetSerializer();
	// we write a list of multiple expressions here
	// in order to write this as a single field we directly use the field writers' internal serializer
	writer.WriteField<uint32_t>(case_checks.size());
	for (auto &check : case_checks) {
		check.when_expr->Serialize(serializer);
		check.then_expr->Serialize(serializer);
	}
	writer.WriteSerializable<ParsedExpression>(*else_expr);
}

unique_ptr<ParsedExpression> CaseExpression::Deserialize(ExpressionType type, FieldReader &reader) {
	auto result = make_unique<CaseExpression>();
	auto &source = reader.GetSource();
	auto count = reader.ReadRequired<uint32_t>();
	for (idx_t i = 0; i < count; i++) {
		CaseCheck new_check;
		new_check.when_expr = ParsedExpression::Deserialize(source);
		new_check.then_expr = ParsedExpression::Deserialize(source);
		result->case_checks.push_back(move(new_check));
	}
	result->else_expr = reader.ReadRequiredSerializable<ParsedExpression>();
	return move(result);
}

} // namespace duckdb





namespace duckdb {

CastExpression::CastExpression(LogicalType target, unique_ptr<ParsedExpression> child, bool try_cast_p)
    : ParsedExpression(ExpressionType::OPERATOR_CAST, ExpressionClass::CAST), cast_type(move(target)),
      try_cast(try_cast_p) {
	D_ASSERT(child);
	this->child = move(child);
}

string CastExpression::ToString() const {
	return ToString<CastExpression, ParsedExpression>(*this);
}

bool CastExpression::Equals(const CastExpression *a, const CastExpression *b) {
	if (!a->child->Equals(b->child.get())) {
		return false;
	}
	if (a->cast_type != b->cast_type) {
		return false;
	}
	if (a->try_cast != b->try_cast) {
		return false;
	}
	return true;
}

unique_ptr<ParsedExpression> CastExpression::Copy() const {
	auto copy = make_unique<CastExpression>(cast_type, child->Copy(), try_cast);
	copy->CopyProperties(*this);
	return move(copy);
}

void CastExpression::Serialize(FieldWriter &writer) const {
	writer.WriteSerializable(*child);
	writer.WriteSerializable(cast_type);
	writer.WriteField<bool>(try_cast);
}

unique_ptr<ParsedExpression> CastExpression::Deserialize(ExpressionType type, FieldReader &reader) {
	auto child = reader.ReadRequiredSerializable<ParsedExpression>();
	auto cast_type = reader.ReadRequiredSerializable<LogicalType, LogicalType>();
	auto try_cast = reader.ReadRequired<bool>();
	return make_unique_base<ParsedExpression, CastExpression>(cast_type, move(child), try_cast);
}

} // namespace duckdb





namespace duckdb {

CollateExpression::CollateExpression(string collation_p, unique_ptr<ParsedExpression> child)
    : ParsedExpression(ExpressionType::COLLATE, ExpressionClass::COLLATE), collation(move(collation_p)) {
	D_ASSERT(child);
	this->child = move(child);
}

string CollateExpression::ToString() const {
	return child->ToString() + " COLLATE " + KeywordHelper::WriteOptionallyQuoted(collation);
}

bool CollateExpression::Equals(const CollateExpression *a, const CollateExpression *b) {
	if (!a->child->Equals(b->child.get())) {
		return false;
	}
	if (a->collation != b->collation) {
		return false;
	}
	return true;
}

unique_ptr<ParsedExpression> CollateExpression::Copy() const {
	auto copy = make_unique<CollateExpression>(collation, child->Copy());
	copy->CopyProperties(*this);
	return move(copy);
}

void CollateExpression::Serialize(FieldWriter &writer) const {
	writer.WriteSerializable(*child);
	writer.WriteString(collation);
}

unique_ptr<ParsedExpression> CollateExpression::Deserialize(ExpressionType type, FieldReader &reader) {
	auto child = reader.ReadRequiredSerializable<ParsedExpression>();
	auto collation = reader.ReadRequired<string>();
	return make_unique_base<ParsedExpression, CollateExpression>(collation, move(child));
}

} // namespace duckdb







namespace duckdb {

ColumnRefExpression::ColumnRefExpression(string column_name, string table_name)
    : ColumnRefExpression(table_name.empty() ? vector<string> {move(column_name)}
                                             : vector<string> {move(table_name), move(column_name)}) {
}

ColumnRefExpression::ColumnRefExpression(string column_name) : ColumnRefExpression(vector<string> {move(column_name)}) {
}

ColumnRefExpression::ColumnRefExpression(vector<string> column_names_p)
    : ParsedExpression(ExpressionType::COLUMN_REF, ExpressionClass::COLUMN_REF), column_names(move(column_names_p)) {
#ifdef DEBUG
	for (auto &col_name : column_names) {
		D_ASSERT(!col_name.empty());
	}
#endif
}

bool ColumnRefExpression::IsQualified() const {
	return column_names.size() > 1;
}

const string &ColumnRefExpression::GetColumnName() const {
	D_ASSERT(column_names.size() <= 3);
	return column_names.back();
}

const string &ColumnRefExpression::GetTableName() const {
	D_ASSERT(column_names.size() >= 2 && column_names.size() <= 3);
	return column_names.size() == 3 ? column_names[1] : column_names[0];
}

string ColumnRefExpression::GetName() const {
	return !alias.empty() ? alias : column_names.back();
}

string ColumnRefExpression::ToString() const {
	string result;
	for (idx_t i = 0; i < column_names.size(); i++) {
		if (i > 0) {
			result += ".";
		}
		result += KeywordHelper::WriteOptionallyQuoted(column_names[i]);
	}
	return result;
}

bool ColumnRefExpression::Equals(const ColumnRefExpression *a, const ColumnRefExpression *b) {
	if (a->column_names.size() != b->column_names.size()) {
		return false;
	}
	for (idx_t i = 0; i < a->column_names.size(); i++) {
		auto lcase_a = StringUtil::Lower(a->column_names[i]);
		auto lcase_b = StringUtil::Lower(b->column_names[i]);
		if (lcase_a != lcase_b) {
			return false;
		}
	}
	return true;
}

hash_t ColumnRefExpression::Hash() const {
	hash_t result = ParsedExpression::Hash();
	for (auto &column_name : column_names) {
		auto lcase = StringUtil::Lower(column_name);
		result = CombineHash(result, duckdb::Hash<const char *>(lcase.c_str()));
	}
	return result;
}

unique_ptr<ParsedExpression> ColumnRefExpression::Copy() const {
	auto copy = make_unique<ColumnRefExpression>(column_names);
	copy->CopyProperties(*this);
	return move(copy);
}

void ColumnRefExpression::Serialize(FieldWriter &writer) const {
	writer.WriteList<string>(column_names);
}

unique_ptr<ParsedExpression> ColumnRefExpression::Deserialize(ExpressionType type, FieldReader &reader) {
	auto column_names = reader.ReadRequiredList<string>();
	auto expression = make_unique<ColumnRefExpression>(move(column_names));
	return move(expression);
}

} // namespace duckdb






namespace duckdb {

ComparisonExpression::ComparisonExpression(ExpressionType type, unique_ptr<ParsedExpression> left,
                                           unique_ptr<ParsedExpression> right)
    : ParsedExpression(type, ExpressionClass::COMPARISON), left(move(left)), right(move(right)) {
}

string ComparisonExpression::ToString() const {
	return ToString<ComparisonExpression, ParsedExpression>(*this);
}

bool ComparisonExpression::Equals(const ComparisonExpression *a, const ComparisonExpression *b) {
	if (!a->left->Equals(b->left.get())) {
		return false;
	}
	if (!a->right->Equals(b->right.get())) {
		return false;
	}
	return true;
}

unique_ptr<ParsedExpression> ComparisonExpression::Copy() const {
	auto copy = make_unique<ComparisonExpression>(type, left->Copy(), right->Copy());
	copy->CopyProperties(*this);
	return move(copy);
}

void ComparisonExpression::Serialize(FieldWriter &writer) const {
	writer.WriteSerializable(*left);
	writer.WriteSerializable(*right);
}

unique_ptr<ParsedExpression> ComparisonExpression::Deserialize(ExpressionType type, FieldReader &reader) {
	auto left_child = reader.ReadRequiredSerializable<ParsedExpression>();
	auto right_child = reader.ReadRequiredSerializable<ParsedExpression>();
	return make_unique<ComparisonExpression>(type, move(left_child), move(right_child));
}

} // namespace duckdb





namespace duckdb {

ConjunctionExpression::ConjunctionExpression(ExpressionType type)
    : ParsedExpression(type, ExpressionClass::CONJUNCTION) {
}

ConjunctionExpression::ConjunctionExpression(ExpressionType type, vector<unique_ptr<ParsedExpression>> children)
    : ParsedExpression(type, ExpressionClass::CONJUNCTION) {
	for (auto &child : children) {
		AddExpression(move(child));
	}
}

ConjunctionExpression::ConjunctionExpression(ExpressionType type, unique_ptr<ParsedExpression> left,
                                             unique_ptr<ParsedExpression> right)
    : ParsedExpression(type, ExpressionClass::CONJUNCTION) {
	AddExpression(move(left));
	AddExpression(move(right));
}

void ConjunctionExpression::AddExpression(unique_ptr<ParsedExpression> expr) {
	if (expr->type == type) {
		// expr is a conjunction of the same type: merge the expression lists together
		auto &other = (ConjunctionExpression &)*expr;
		for (auto &child : other.children) {
			children.push_back(move(child));
		}
	} else {
		children.push_back(move(expr));
	}
}

string ConjunctionExpression::ToString() const {
	return ToString<ConjunctionExpression, ParsedExpression>(*this);
}

bool ConjunctionExpression::Equals(const ConjunctionExpression *a, const ConjunctionExpression *b) {
	return ExpressionUtil::SetEquals(a->children, b->children);
}

unique_ptr<ParsedExpression> ConjunctionExpression::Copy() const {
	vector<unique_ptr<ParsedExpression>> copy_children;
	for (auto &expr : children) {
		copy_children.push_back(expr->Copy());
	}
	auto copy = make_unique<ConjunctionExpression>(type, move(copy_children));
	copy->CopyProperties(*this);
	return move(copy);
}

void ConjunctionExpression::Serialize(FieldWriter &writer) const {
	writer.WriteSerializableList(children);
}

unique_ptr<ParsedExpression> ConjunctionExpression::Deserialize(ExpressionType type, FieldReader &reader) {
	auto result = make_unique<ConjunctionExpression>(type);
	result->children = reader.ReadRequiredSerializableList<ParsedExpression>();
	return move(result);
}

} // namespace duckdb







namespace duckdb {

ConstantExpression::ConstantExpression(Value val)
    : ParsedExpression(ExpressionType::VALUE_CONSTANT, ExpressionClass::CONSTANT), value(move(val)) {
}

string ConstantExpression::ToString() const {
	return value.ToSQLString();
}

bool ConstantExpression::Equals(const ConstantExpression *a, const ConstantExpression *b) {
	return a->value.type() == b->value.type() && !ValueOperations::DistinctFrom(a->value, b->value);
}

hash_t ConstantExpression::Hash() const {
	return ParsedExpression::Hash();
}

unique_ptr<ParsedExpression> ConstantExpression::Copy() const {
	auto copy = make_unique<ConstantExpression>(value);
	copy->CopyProperties(*this);
	return move(copy);
}

void ConstantExpression::Serialize(FieldWriter &writer) const {
	writer.WriteSerializable(value);
}

unique_ptr<ParsedExpression> ConstantExpression::Deserialize(ExpressionType type, FieldReader &reader) {
	Value value = reader.ReadRequiredSerializable<Value, Value>();
	return make_unique<ConstantExpression>(move(value));
}

} // namespace duckdb




namespace duckdb {

DefaultExpression::DefaultExpression() : ParsedExpression(ExpressionType::VALUE_DEFAULT, ExpressionClass::DEFAULT) {
}

string DefaultExpression::ToString() const {
	return "DEFAULT";
}

unique_ptr<ParsedExpression> DefaultExpression::Copy() const {
	auto copy = make_unique<DefaultExpression>();
	copy->CopyProperties(*this);
	return move(copy);
}

void DefaultExpression::Serialize(FieldWriter &writer) const {
}

unique_ptr<ParsedExpression> DefaultExpression::Deserialize(ExpressionType type, FieldReader &source) {
	return make_unique<DefaultExpression>();
}

} // namespace duckdb


#include <utility>





namespace duckdb {

FunctionExpression::FunctionExpression(string schema, const string &function_name,
                                       vector<unique_ptr<ParsedExpression>> children_p,
                                       unique_ptr<ParsedExpression> filter, unique_ptr<OrderModifier> order_bys_p,
                                       bool distinct, bool is_operator, bool export_state_p)
    : ParsedExpression(ExpressionType::FUNCTION, ExpressionClass::FUNCTION), schema(std::move(schema)),
      function_name(StringUtil::Lower(function_name)), is_operator(is_operator), children(move(children_p)),
      distinct(distinct), filter(move(filter)), order_bys(move(order_bys_p)), export_state(export_state_p) {
	D_ASSERT(!function_name.empty());
	if (!order_bys) {
		order_bys = make_unique<OrderModifier>();
	}
}

FunctionExpression::FunctionExpression(const string &function_name, vector<unique_ptr<ParsedExpression>> children_p,
                                       unique_ptr<ParsedExpression> filter, unique_ptr<OrderModifier> order_bys,
                                       bool distinct, bool is_operator, bool export_state_p)
    : FunctionExpression(INVALID_SCHEMA, function_name, move(children_p), move(filter), move(order_bys), distinct,
                         is_operator, export_state_p) {
}

string FunctionExpression::ToString() const {
	return ToString<FunctionExpression, ParsedExpression>(*this, schema, function_name, is_operator, distinct,
	                                                      filter.get(), order_bys.get(), export_state, true);
}

bool FunctionExpression::Equals(const FunctionExpression *a, const FunctionExpression *b) {
	if (a->schema != b->schema || a->function_name != b->function_name || b->distinct != a->distinct) {
		return false;
	}
	if (b->children.size() != a->children.size()) {
		return false;
	}
	for (idx_t i = 0; i < a->children.size(); i++) {
		if (!a->children[i]->Equals(b->children[i].get())) {
			return false;
		}
	}
	if (!BaseExpression::Equals(a->filter.get(), b->filter.get())) {
		return false;
	}
	if (!a->order_bys->Equals(b->order_bys.get())) {
		return false;
	}
	if (a->export_state != b->export_state) {
		return false;
	}
	return true;
}

hash_t FunctionExpression::Hash() const {
	hash_t result = ParsedExpression::Hash();
	result = CombineHash(result, duckdb::Hash<const char *>(schema.c_str()));
	result = CombineHash(result, duckdb::Hash<const char *>(function_name.c_str()));
	result = CombineHash(result, duckdb::Hash<bool>(distinct));
	result = CombineHash(result, duckdb::Hash<bool>(export_state));
	return result;
}

unique_ptr<ParsedExpression> FunctionExpression::Copy() const {
	vector<unique_ptr<ParsedExpression>> copy_children;
	unique_ptr<ParsedExpression> filter_copy;
	for (auto &child : children) {
		copy_children.push_back(child->Copy());
	}
	if (filter) {
		filter_copy = filter->Copy();
	}
	unique_ptr<OrderModifier> order_copy;
	if (order_bys) {
		order_copy.reset(static_cast<OrderModifier *>(order_bys->Copy().release()));
	}

	auto copy = make_unique<FunctionExpression>(function_name, move(copy_children), move(filter_copy), move(order_copy),
	                                            distinct, is_operator, export_state);
	copy->schema = schema;
	copy->CopyProperties(*this);
	return move(copy);
}

void FunctionExpression::Serialize(FieldWriter &writer) const {
	writer.WriteString(function_name);
	writer.WriteString(schema);
	writer.WriteSerializableList(children);
	writer.WriteOptional(filter);
	writer.WriteSerializable((ResultModifier &)*order_bys);
	writer.WriteField<bool>(distinct);
	writer.WriteField<bool>(is_operator);
	writer.WriteField<bool>(export_state);
}

unique_ptr<ParsedExpression> FunctionExpression::Deserialize(ExpressionType type, FieldReader &reader) {
	auto function_name = reader.ReadRequired<string>();
	auto schema = reader.ReadRequired<string>();
	auto children = reader.ReadRequiredSerializableList<ParsedExpression>();
	auto filter = reader.ReadOptional<ParsedExpression>(nullptr);
	auto order_bys = unique_ptr_cast<ResultModifier, OrderModifier>(reader.ReadRequiredSerializable<ResultModifier>());
	auto distinct = reader.ReadRequired<bool>();
	auto is_operator = reader.ReadRequired<bool>();
	auto export_state = reader.ReadField<bool>(false);

	unique_ptr<FunctionExpression> function;
	function = make_unique<FunctionExpression>(function_name, move(children), move(filter), move(order_bys), distinct,
	                                           is_operator, export_state);
	function->schema = schema;
	return move(function);
}

void FunctionExpression::Verify() const {
	D_ASSERT(!function_name.empty());
}

} // namespace duckdb





namespace duckdb {

LambdaExpression::LambdaExpression(unique_ptr<ParsedExpression> lhs, unique_ptr<ParsedExpression> expr)
    : ParsedExpression(ExpressionType::LAMBDA, ExpressionClass::LAMBDA), lhs(move(lhs)), expr(move(expr)) {
}

string LambdaExpression::ToString() const {
	return lhs->ToString() + " -> " + expr->ToString();
}

bool LambdaExpression::Equals(const LambdaExpression *a, const LambdaExpression *b) {
	return a->lhs->Equals(b->lhs.get()) && a->expr->Equals(b->expr.get());
}

hash_t LambdaExpression::Hash() const {
	hash_t result = lhs->Hash();
	ParsedExpression::Hash();
	result = CombineHash(result, expr->Hash());
	return result;
}

unique_ptr<ParsedExpression> LambdaExpression::Copy() const {
	auto copy = make_unique<LambdaExpression>(lhs->Copy(), expr->Copy());
	copy->CopyProperties(*this);
	return move(copy);
}

void LambdaExpression::Serialize(FieldWriter &writer) const {
	writer.WriteSerializable(*lhs);
	writer.WriteSerializable(*expr);
}

unique_ptr<ParsedExpression> LambdaExpression::Deserialize(ExpressionType type, FieldReader &reader) {
	auto lhs = reader.ReadRequiredSerializable<ParsedExpression>();
	auto expr = reader.ReadRequiredSerializable<ParsedExpression>();
	return make_unique<LambdaExpression>(move(lhs), move(expr));
}

} // namespace duckdb





namespace duckdb {

OperatorExpression::OperatorExpression(ExpressionType type, unique_ptr<ParsedExpression> left,
                                       unique_ptr<ParsedExpression> right)
    : ParsedExpression(type, ExpressionClass::OPERATOR) {
	if (left) {
		children.push_back(move(left));
	}
	if (right) {
		children.push_back(move(right));
	}
}

OperatorExpression::OperatorExpression(ExpressionType type, vector<unique_ptr<ParsedExpression>> children)
    : ParsedExpression(type, ExpressionClass::OPERATOR), children(move(children)) {
}

string OperatorExpression::ToString() const {
	return ToString<OperatorExpression, ParsedExpression>(*this);
}

bool OperatorExpression::Equals(const OperatorExpression *a, const OperatorExpression *b) {
	if (a->children.size() != b->children.size()) {
		return false;
	}
	for (idx_t i = 0; i < a->children.size(); i++) {
		if (!a->children[i]->Equals(b->children[i].get())) {
			return false;
		}
	}
	return true;
}

unique_ptr<ParsedExpression> OperatorExpression::Copy() const {
	auto copy = make_unique<OperatorExpression>(type);
	copy->CopyProperties(*this);
	for (auto &it : children) {
		copy->children.push_back(it->Copy());
	}
	return move(copy);
}

void OperatorExpression::Serialize(FieldWriter &writer) const {
	writer.WriteSerializableList(children);
}

unique_ptr<ParsedExpression> OperatorExpression::Deserialize(ExpressionType type, FieldReader &reader) {
	auto expression = make_unique<OperatorExpression>(type);
	expression->children = reader.ReadRequiredSerializableList<ParsedExpression>();
	return move(expression);
}

} // namespace duckdb







namespace duckdb {

ParameterExpression::ParameterExpression()
    : ParsedExpression(ExpressionType::VALUE_PARAMETER, ExpressionClass::PARAMETER), parameter_nr(0) {
}

string ParameterExpression::ToString() const {
	return "$" + to_string(parameter_nr);
}

unique_ptr<ParsedExpression> ParameterExpression::Copy() const {
	auto copy = make_unique<ParameterExpression>();
	copy->parameter_nr = parameter_nr;
	copy->CopyProperties(*this);
	return move(copy);
}

bool ParameterExpression::Equals(const ParameterExpression *a, const ParameterExpression *b) {
	return a->parameter_nr == b->parameter_nr;
}

hash_t ParameterExpression::Hash() const {
	hash_t result = ParsedExpression::Hash();
	return CombineHash(duckdb::Hash(parameter_nr), result);
}

void ParameterExpression::Serialize(FieldWriter &writer) const {
	writer.WriteField<idx_t>(parameter_nr);
}

unique_ptr<ParsedExpression> ParameterExpression::Deserialize(ExpressionType type, FieldReader &reader) {
	auto expression = make_unique<ParameterExpression>();
	expression->parameter_nr = reader.ReadRequired<idx_t>();
	return move(expression);
}

} // namespace duckdb







namespace duckdb {

PositionalReferenceExpression::PositionalReferenceExpression(idx_t index)
    : ParsedExpression(ExpressionType::POSITIONAL_REFERENCE, ExpressionClass::POSITIONAL_REFERENCE), index(index) {
}

string PositionalReferenceExpression::ToString() const {
	return "#" + to_string(index);
}

bool PositionalReferenceExpression::Equals(const PositionalReferenceExpression *a,
                                           const PositionalReferenceExpression *b) {
	return a->index == b->index;
}

unique_ptr<ParsedExpression> PositionalReferenceExpression::Copy() const {
	auto copy = make_unique<PositionalReferenceExpression>(index);
	copy->CopyProperties(*this);
	return move(copy);
}

hash_t PositionalReferenceExpression::Hash() const {
	hash_t result = ParsedExpression::Hash();
	return CombineHash(duckdb::Hash(index), result);
}

void PositionalReferenceExpression::Serialize(FieldWriter &writer) const {
	writer.WriteField<idx_t>(index);
}

unique_ptr<ParsedExpression> PositionalReferenceExpression::Deserialize(ExpressionType type, FieldReader &reader) {
	auto expression = make_unique<PositionalReferenceExpression>(reader.ReadRequired<idx_t>());
	return move(expression);
}

} // namespace duckdb





namespace duckdb {

StarExpression::StarExpression(string relation_name_p)
    : ParsedExpression(ExpressionType::STAR, ExpressionClass::STAR), relation_name(move(relation_name_p)) {
}

string StarExpression::ToString() const {
	if (!regex.empty()) {
		D_ASSERT(columns);
		return "COLUMNS('" + regex + "')";
	}
	string result;
	if (columns) {
		result += "COLUMNS(";
	}
	result += relation_name.empty() ? "*" : relation_name + ".*";
	if (!exclude_list.empty()) {
		result += " EXCLUDE (";
		bool first_entry = true;
		for (auto &entry : exclude_list) {
			if (!first_entry) {
				result += ", ";
			}
			result += entry;
			first_entry = false;
		}
		result += ")";
	}
	if (!replace_list.empty()) {
		result += " REPLACE (";
		bool first_entry = true;
		for (auto &entry : replace_list) {
			if (!first_entry) {
				result += ", ";
			}
			result += entry.second->ToString();
			result += " AS ";
			result += entry.first;
			first_entry = false;
		}
		result += ")";
	}
	if (columns) {
		result += ")";
	}
	return result;
}

bool StarExpression::Equals(const StarExpression *a, const StarExpression *b) {
	if (a->relation_name != b->relation_name || a->exclude_list != b->exclude_list) {
		return false;
	}
	if (a->columns != b->columns) {
		return false;
	}
	if (a->replace_list.size() != b->replace_list.size()) {
		return false;
	}
	for (auto &entry : a->replace_list) {
		auto other_entry = b->replace_list.find(entry.first);
		if (other_entry == b->replace_list.end()) {
			return false;
		}
		if (!entry.second->Equals(other_entry->second.get())) {
			return false;
		}
	}
	if (a->regex != b->regex) {
		return false;
	}
	return true;
}

void StarExpression::Serialize(FieldWriter &writer) const {
	auto &serializer = writer.GetSerializer();

	writer.WriteString(relation_name);

	// in order to write the exclude_list/replace_list as single fields we directly use the field writers' internal
	// serializer
	writer.WriteField<uint32_t>(exclude_list.size());
	for (auto &exclusion : exclude_list) {
		serializer.WriteString(exclusion);
	}
	writer.WriteField<uint32_t>(replace_list.size());
	for (auto &entry : replace_list) {
		serializer.WriteString(entry.first);
		entry.second->Serialize(serializer);
	}
	writer.WriteField<bool>(columns);
	writer.WriteString(regex);
}

unique_ptr<ParsedExpression> StarExpression::Deserialize(ExpressionType type, FieldReader &reader) {
	auto &source = reader.GetSource();

	auto result = make_unique<StarExpression>();
	result->relation_name = reader.ReadRequired<string>();
	auto exclusion_count = reader.ReadRequired<uint32_t>();
	for (idx_t i = 0; i < exclusion_count; i++) {
		result->exclude_list.insert(source.Read<string>());
	}
	auto replace_count = reader.ReadRequired<uint32_t>();
	for (idx_t i = 0; i < replace_count; i++) {
		auto name = source.Read<string>();
		auto expr = ParsedExpression::Deserialize(source);
		result->replace_list.insert(make_pair(name, move(expr)));
	}
	result->columns = reader.ReadField<bool>(false);
	result->regex = reader.ReadField<string>(string());
	return move(result);
}

unique_ptr<ParsedExpression> StarExpression::Copy() const {
	auto copy = make_unique<StarExpression>(relation_name);
	copy->exclude_list = exclude_list;
	for (auto &entry : replace_list) {
		copy->replace_list[entry.first] = entry.second->Copy();
	}
	copy->columns = columns;
	copy->regex = regex;
	copy->CopyProperties(*this);
	return move(copy);
}

} // namespace duckdb





namespace duckdb {

SubqueryExpression::SubqueryExpression()
    : ParsedExpression(ExpressionType::SUBQUERY, ExpressionClass::SUBQUERY), subquery_type(SubqueryType::INVALID),
      comparison_type(ExpressionType::INVALID) {
}

string SubqueryExpression::ToString() const {
	switch (subquery_type) {
	case SubqueryType::ANY:
		return "(" + child->ToString() + " " + ExpressionTypeToOperator(comparison_type) + " ANY(" +
		       subquery->ToString() + "))";
	case SubqueryType::EXISTS:
		return "EXISTS(" + subquery->ToString() + ")";
	case SubqueryType::NOT_EXISTS:
		return "NOT EXISTS(" + subquery->ToString() + ")";
	case SubqueryType::SCALAR:
		return "(" + subquery->ToString() + ")";
	default:
		throw InternalException("Unrecognized type for subquery");
	}
}

bool SubqueryExpression::Equals(const SubqueryExpression *a, const SubqueryExpression *b) {
	if (!a->subquery || !b->subquery) {
		return false;
	}
	if (!BaseExpression::Equals(a->child.get(), b->child.get())) {
		return false;
	}
	return a->comparison_type == b->comparison_type && a->subquery_type == b->subquery_type &&
	       a->subquery->Equals(b->subquery.get());
}

unique_ptr<ParsedExpression> SubqueryExpression::Copy() const {
	auto copy = make_unique<SubqueryExpression>();
	copy->CopyProperties(*this);
	copy->subquery = unique_ptr_cast<SQLStatement, SelectStatement>(subquery->Copy());
	copy->subquery_type = subquery_type;
	copy->child = child ? child->Copy() : nullptr;
	copy->comparison_type = comparison_type;
	return move(copy);
}

void SubqueryExpression::Serialize(FieldWriter &writer) const {
	auto &serializer = writer.GetSerializer();

	writer.WriteField<SubqueryType>(subquery_type);
	// FIXME: this shouldn't use a serializer (probably)?
	subquery->Serialize(serializer);
	writer.WriteOptional(child);
	writer.WriteField<ExpressionType>(comparison_type);
}

unique_ptr<ParsedExpression> SubqueryExpression::Deserialize(ExpressionType type, FieldReader &reader) {
	// FIXME: this shouldn't use a source
	auto &source = reader.GetSource();

	auto subquery_type = reader.ReadRequired<SubqueryType>();
	auto subquery = SelectStatement::Deserialize(source);

	auto expression = make_unique<SubqueryExpression>();
	expression->subquery_type = subquery_type;
	expression->subquery = move(subquery);
	expression->child = reader.ReadOptional<ParsedExpression>(nullptr);
	expression->comparison_type = reader.ReadRequired<ExpressionType>();
	return move(expression);
}

} // namespace duckdb






namespace duckdb {

WindowExpression::WindowExpression(ExpressionType type, string schema, const string &function_name)
    : ParsedExpression(type, ExpressionClass::WINDOW), schema(move(schema)),
      function_name(StringUtil::Lower(function_name)), ignore_nulls(false) {
	switch (type) {
	case ExpressionType::WINDOW_AGGREGATE:
	case ExpressionType::WINDOW_ROW_NUMBER:
	case ExpressionType::WINDOW_FIRST_VALUE:
	case ExpressionType::WINDOW_LAST_VALUE:
	case ExpressionType::WINDOW_NTH_VALUE:
	case ExpressionType::WINDOW_RANK:
	case ExpressionType::WINDOW_RANK_DENSE:
	case ExpressionType::WINDOW_PERCENT_RANK:
	case ExpressionType::WINDOW_CUME_DIST:
	case ExpressionType::WINDOW_LEAD:
	case ExpressionType::WINDOW_LAG:
	case ExpressionType::WINDOW_NTILE:
		break;
	default:
		throw NotImplementedException("Window aggregate type %s not supported", ExpressionTypeToString(type).c_str());
	}
}

string WindowExpression::ToString() const {
	return ToString<WindowExpression, ParsedExpression, OrderByNode>(*this, schema, function_name);
}

bool WindowExpression::Equals(const WindowExpression *a, const WindowExpression *b) {
	// check if the child expressions are equivalent
	if (b->children.size() != a->children.size()) {
		return false;
	}
	if (a->ignore_nulls != b->ignore_nulls) {
		return false;
	}
	for (idx_t i = 0; i < a->children.size(); i++) {
		if (!a->children[i]->Equals(b->children[i].get())) {
			return false;
		}
	}
	if (a->start != b->start || a->end != b->end) {
		return false;
	}
	// check if the framing expressions are equivalent
	if (!BaseExpression::Equals(a->start_expr.get(), b->start_expr.get()) ||
	    !BaseExpression::Equals(a->end_expr.get(), b->end_expr.get()) ||
	    !BaseExpression::Equals(a->offset_expr.get(), b->offset_expr.get()) ||
	    !BaseExpression::Equals(a->default_expr.get(), b->default_expr.get())) {
		return false;
	}

	// check if the partitions are equivalent
	if (a->partitions.size() != b->partitions.size()) {
		return false;
	}
	for (idx_t i = 0; i < a->partitions.size(); i++) {
		if (!a->partitions[i]->Equals(b->partitions[i].get())) {
			return false;
		}
	}
	// check if the orderings are equivalent
	if (a->orders.size() != b->orders.size()) {
		return false;
	}
	for (idx_t i = 0; i < a->orders.size(); i++) {
		if (a->orders[i].type != b->orders[i].type) {
			return false;
		}
		if (!a->orders[i].expression->Equals(b->orders[i].expression.get())) {
			return false;
		}
	}
	// check if the filter clauses are equivalent
	if (!BaseExpression::Equals(a->filter_expr.get(), b->filter_expr.get())) {
		return false;
	}

	return true;
}

unique_ptr<ParsedExpression> WindowExpression::Copy() const {
	auto new_window = make_unique<WindowExpression>(type, schema, function_name);
	new_window->CopyProperties(*this);

	for (auto &child : children) {
		new_window->children.push_back(child->Copy());
	}

	for (auto &e : partitions) {
		new_window->partitions.push_back(e->Copy());
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

void WindowExpression::Serialize(FieldWriter &writer) const {
	auto &serializer = writer.GetSerializer();

	writer.WriteString(function_name);
	writer.WriteString(schema);
	writer.WriteSerializableList(children);
	writer.WriteSerializableList(partitions);
	// FIXME: should not use serializer here (probably)?
	D_ASSERT(orders.size() <= NumericLimits<uint32_t>::Maximum());
	writer.WriteField<uint32_t>((uint32_t)orders.size());
	for (auto &order : orders) {
		order.Serialize(serializer);
	}
	writer.WriteField<WindowBoundary>(start);
	writer.WriteField<WindowBoundary>(end);

	writer.WriteOptional(start_expr);
	writer.WriteOptional(end_expr);
	writer.WriteOptional(offset_expr);
	writer.WriteOptional(default_expr);
	writer.WriteField<bool>(ignore_nulls);
	writer.WriteOptional(filter_expr);
}

unique_ptr<ParsedExpression> WindowExpression::Deserialize(ExpressionType type, FieldReader &reader) {
	auto function_name = reader.ReadRequired<string>();
	auto schema = reader.ReadRequired<string>();
	auto expr = make_unique<WindowExpression>(type, schema, function_name);
	expr->children = reader.ReadRequiredSerializableList<ParsedExpression>();
	expr->partitions = reader.ReadRequiredSerializableList<ParsedExpression>();

	auto order_count = reader.ReadRequired<uint32_t>();
	auto &source = reader.GetSource();
	for (idx_t i = 0; i < order_count; i++) {
		expr->orders.push_back(OrderByNode::Deserialize(source));
	}
	expr->start = reader.ReadRequired<WindowBoundary>();
	expr->end = reader.ReadRequired<WindowBoundary>();

	expr->start_expr = reader.ReadOptional<ParsedExpression>(nullptr);
	expr->end_expr = reader.ReadOptional<ParsedExpression>(nullptr);
	expr->offset_expr = reader.ReadOptional<ParsedExpression>(nullptr);
	expr->default_expr = reader.ReadOptional<ParsedExpression>(nullptr);
	expr->ignore_nulls = reader.ReadRequired<bool>();
	expr->filter_expr = reader.ReadOptional<ParsedExpression>(nullptr);
	return move(expr);
}

} // namespace duckdb





namespace duckdb {

template <class T>
bool ExpressionUtil::ExpressionListEquals(const vector<unique_ptr<T>> &a, const vector<unique_ptr<T>> &b) {
	if (a.size() != b.size()) {
		return false;
	}
	for (idx_t i = 0; i < a.size(); i++) {
		if (!(*a[i] == *b[i])) {
			return false;
		}
	}
	return true;
}

template <class T>
bool ExpressionUtil::ExpressionSetEquals(const vector<unique_ptr<T>> &a, const vector<unique_ptr<T>> &b) {
	if (a.size() != b.size()) {
		return false;
	}
	// we create a map of expression -> count for the left side
	// we keep the count because the same expression can occur multiple times (e.g. "1 AND 1" is legal)
	// in this case we track the following value: map["Constant(1)"] = 2
	expression_map_t<idx_t> map;
	for (idx_t i = 0; i < a.size(); i++) {
		map[a[i].get()]++;
	}
	// now on the right side we reduce the counts again
	// if the conjunctions are identical, all the counts will be 0 after the
	for (auto &expr : b) {
		auto entry = map.find(expr.get());
		// first we check if we can find the expression in the map at all
		if (entry == map.end()) {
			return false;
		}
		// if we found it we check the count; if the count is already 0 we return false
		// this happens if e.g. the left side contains "1 AND X", and the right side contains "1 AND 1"
		// "1" is contained in the map, however, the right side contains the expression twice
		// hence we know the children are not identical in this case because the LHS and RHS have a different count for
		// the Constant(1) expression
		if (entry->second == 0) {
			return false;
		}
		entry->second--;
	}
	return true;
}

bool ExpressionUtil::ListEquals(const vector<unique_ptr<ParsedExpression>> &a,
                                const vector<unique_ptr<ParsedExpression>> &b) {
	return ExpressionListEquals<ParsedExpression>(a, b);
}

bool ExpressionUtil::ListEquals(const vector<unique_ptr<Expression>> &a, const vector<unique_ptr<Expression>> &b) {
	return ExpressionListEquals<Expression>(a, b);
}

bool ExpressionUtil::SetEquals(const vector<unique_ptr<ParsedExpression>> &a,
                               const vector<unique_ptr<ParsedExpression>> &b) {
	return ExpressionSetEquals<ParsedExpression>(a, b);
}

bool ExpressionUtil::SetEquals(const vector<unique_ptr<Expression>> &a, const vector<unique_ptr<Expression>> &b) {
	return ExpressionSetEquals<Expression>(a, b);
}

} // namespace duckdb




namespace duckdb {

bool KeywordHelper::IsKeyword(const string &text) {
	return Parser::IsKeyword(text);
}

bool KeywordHelper::RequiresQuotes(const string &text, bool allow_caps) {
	for (size_t i = 0; i < text.size(); i++) {
		if (i > 0 && (text[i] >= '0' && text[i] <= '9')) {
			continue;
		}
		if (text[i] >= 'a' && text[i] <= 'z') {
			continue;
		}
		if (allow_caps) {
			if (text[i] >= 'A' && text[i] <= 'Z') {
				continue;
			}
		}
		if (text[i] == '_') {
			continue;
		}
		return true;
	}
	return IsKeyword(text);
}

string KeywordHelper::WriteOptionallyQuoted(const string &text, char quote, bool allow_caps) {
	if (!RequiresQuotes(text, allow_caps)) {
		return text;
	}
	return string(1, quote) + StringUtil::Replace(text, string(1, quote), string(2, quote)) + string(1, quote);
}

} // namespace duckdb





namespace duckdb {

//===--------------------------------------------------------------------===//
// AlterFunctionInfo
//===--------------------------------------------------------------------===//
AlterFunctionInfo::AlterFunctionInfo(AlterFunctionType type, string schema_p, string table_p, bool if_exists)
    : AlterInfo(AlterType::ALTER_FUNCTION, move(move(schema_p)), move(table_p), if_exists), alter_function_type(type) {
}
AlterFunctionInfo::~AlterFunctionInfo() {
}

CatalogType AlterFunctionInfo::GetCatalogType() const {
	return CatalogType::SCALAR_FUNCTION_ENTRY;
}

void AlterFunctionInfo::Serialize(FieldWriter &writer) const {
	writer.WriteField<AlterFunctionType>(alter_function_type);
	writer.WriteString(schema);
	writer.WriteString(name);
	writer.WriteField(if_exists);
}

unique_ptr<AlterInfo> AlterFunctionInfo::Deserialize(FieldReader &reader) {
	//	auto type = reader.ReadRequired<AlterFunctionType>();
	//	auto schema = reader.ReadRequired<string>();
	//	auto table = reader.ReadRequired<string>();
	//	auto if_exists = reader.ReadRequired<bool>();

	throw NotImplementedException("AlterFunctionInfo cannot be deserialized");
}

//===--------------------------------------------------------------------===//
// AddFunctionOverloadInfo
//===--------------------------------------------------------------------===//
AddFunctionOverloadInfo::AddFunctionOverloadInfo(string schema_p, string name_p, bool if_exists_p,
                                                 ScalarFunctionSet new_overloads_p)
    : AlterFunctionInfo(AlterFunctionType::ADD_FUNCTION_OVERLOADS, move(schema_p), move(name_p), if_exists_p),
      new_overloads(move(new_overloads_p)) {
}
AddFunctionOverloadInfo::~AddFunctionOverloadInfo() {
}

unique_ptr<AlterInfo> AddFunctionOverloadInfo::Copy() const {
	return make_unique_base<AlterInfo, AddFunctionOverloadInfo>(schema, name, if_exists, new_overloads);
}

} // namespace duckdb






namespace duckdb {

AlterInfo::AlterInfo(AlterType type, string schema_p, string name_p, bool if_exists)
    : type(type), if_exists(if_exists), schema(move(schema_p)), name(move(name_p)) {
}

AlterInfo::~AlterInfo() {
}

void AlterInfo::Serialize(Serializer &serializer) const {
	FieldWriter writer(serializer);
	writer.WriteField<AlterType>(type);
	Serialize(writer);
	writer.Finalize();
}

unique_ptr<AlterInfo> AlterInfo::Deserialize(Deserializer &source) {
	FieldReader reader(source);
	auto type = reader.ReadRequired<AlterType>();

	unique_ptr<AlterInfo> result;
	switch (type) {
	case AlterType::ALTER_TABLE:
		result = AlterTableInfo::Deserialize(reader);
		break;
	case AlterType::ALTER_VIEW:
		result = AlterViewInfo::Deserialize(reader);
		break;
	case AlterType::ALTER_FUNCTION:
		result = AlterFunctionInfo::Deserialize(reader);
		break;
	default:
		throw SerializationException("Unknown alter type for deserialization!");
	}
	reader.Finalize();

	return result;
}

} // namespace duckdb





namespace duckdb {

//===--------------------------------------------------------------------===//
// ChangeOwnershipInfo
//===--------------------------------------------------------------------===//
ChangeOwnershipInfo::ChangeOwnershipInfo(CatalogType entry_catalog_type, string entry_schema_p, string entry_name_p,
                                         string owner_schema_p, string owner_name_p, bool if_exists)
    : AlterInfo(AlterType::CHANGE_OWNERSHIP, move(entry_schema_p), move(entry_name_p), if_exists),
      entry_catalog_type(entry_catalog_type), owner_schema(move(owner_schema_p)), owner_name(move(owner_name_p)) {
}

CatalogType ChangeOwnershipInfo::GetCatalogType() const {
	return entry_catalog_type;
}

unique_ptr<AlterInfo> ChangeOwnershipInfo::Copy() const {
	return make_unique_base<AlterInfo, ChangeOwnershipInfo>(entry_catalog_type, schema, name, owner_schema, owner_name,
	                                                        if_exists);
}

void ChangeOwnershipInfo::Serialize(FieldWriter &writer) const {
	throw InternalException("ChangeOwnershipInfo cannot be serialized");
}

//===--------------------------------------------------------------------===//
// AlterTableInfo
//===--------------------------------------------------------------------===//
AlterTableInfo::AlterTableInfo(AlterTableType type, string schema_p, string table_p, bool if_exists)
    : AlterInfo(AlterType::ALTER_TABLE, move(move(schema_p)), move(table_p), if_exists), alter_table_type(type) {
}
AlterTableInfo::~AlterTableInfo() {
}

CatalogType AlterTableInfo::GetCatalogType() const {
	return CatalogType::TABLE_ENTRY;
}

void AlterTableInfo::Serialize(FieldWriter &writer) const {
	writer.WriteField<AlterTableType>(alter_table_type);
	writer.WriteString(schema);
	writer.WriteString(name);
	writer.WriteField(if_exists);
	SerializeAlterTable(writer);
}

unique_ptr<AlterInfo> AlterTableInfo::Deserialize(FieldReader &reader) {
	auto type = reader.ReadRequired<AlterTableType>();
	auto schema = reader.ReadRequired<string>();
	auto table = reader.ReadRequired<string>();
	auto if_exists = reader.ReadRequired<bool>();

	unique_ptr<AlterTableInfo> info;
	switch (type) {
	case AlterTableType::RENAME_COLUMN:
		return RenameColumnInfo::Deserialize(reader, schema, table, if_exists);
	case AlterTableType::RENAME_TABLE:
		return RenameTableInfo::Deserialize(reader, schema, table, if_exists);
	case AlterTableType::ADD_COLUMN:
		return AddColumnInfo::Deserialize(reader, schema, table, if_exists);
	case AlterTableType::REMOVE_COLUMN:
		return RemoveColumnInfo::Deserialize(reader, schema, table, if_exists);
	case AlterTableType::ALTER_COLUMN_TYPE:
		return ChangeColumnTypeInfo::Deserialize(reader, schema, table, if_exists);
	case AlterTableType::SET_DEFAULT:
		return SetDefaultInfo::Deserialize(reader, schema, table, if_exists);
	case AlterTableType::FOREIGN_KEY_CONSTRAINT:
		return AlterForeignKeyInfo::Deserialize(reader, schema, table, if_exists);
	case AlterTableType::SET_NOT_NULL:
		return SetNotNullInfo::Deserialize(reader, schema, table, if_exists);
	case AlterTableType::DROP_NOT_NULL:
		return DropNotNullInfo::Deserialize(reader, schema, table, if_exists);
	default:
		throw SerializationException("Unknown alter table type for deserialization!");
	}
}

//===--------------------------------------------------------------------===//
// RenameColumnInfo
//===--------------------------------------------------------------------===//
RenameColumnInfo::RenameColumnInfo(string schema_p, string table_p, bool if_exists_p, string old_name_p,
                                   string new_name_p)
    : AlterTableInfo(AlterTableType::RENAME_COLUMN, move(schema_p), move(table_p), if_exists_p),
      old_name(move(old_name_p)), new_name(move(new_name_p)) {
}
RenameColumnInfo::~RenameColumnInfo() {
}

unique_ptr<AlterInfo> RenameColumnInfo::Copy() const {
	return make_unique_base<AlterInfo, RenameColumnInfo>(schema, name, if_exists, old_name, new_name);
}

void RenameColumnInfo::SerializeAlterTable(FieldWriter &writer) const {
	writer.WriteString(old_name);
	writer.WriteString(new_name);
}

unique_ptr<AlterInfo> RenameColumnInfo::Deserialize(FieldReader &reader, string schema, string table, bool if_exists) {
	auto old_name = reader.ReadRequired<string>();
	auto new_name = reader.ReadRequired<string>();
	return make_unique<RenameColumnInfo>(move(schema), move(table), if_exists, old_name, new_name);
}

//===--------------------------------------------------------------------===//
// RenameTableInfo
//===--------------------------------------------------------------------===//
RenameTableInfo::RenameTableInfo(string schema_p, string table_p, bool if_exists, string new_name_p)
    : AlterTableInfo(AlterTableType::RENAME_TABLE, move(schema_p), move(table_p), if_exists),
      new_table_name(move(new_name_p)) {
}
RenameTableInfo::~RenameTableInfo() {
}

unique_ptr<AlterInfo> RenameTableInfo::Copy() const {
	return make_unique_base<AlterInfo, RenameTableInfo>(schema, name, if_exists, new_table_name);
}

void RenameTableInfo::SerializeAlterTable(FieldWriter &writer) const {
	writer.WriteString(new_table_name);
}

unique_ptr<AlterInfo> RenameTableInfo::Deserialize(FieldReader &reader, string schema, string table, bool if_exists) {
	auto new_name = reader.ReadRequired<string>();
	return make_unique<RenameTableInfo>(move(schema), move(table), if_exists, new_name);
}

//===--------------------------------------------------------------------===//
// AddColumnInfo
//===--------------------------------------------------------------------===//
AddColumnInfo::AddColumnInfo(string schema_p, string table_p, bool if_exists_p, ColumnDefinition new_column,
                             bool if_column_not_exists)
    : AlterTableInfo(AlterTableType::ADD_COLUMN, move(schema_p), move(table_p), if_exists_p),
      new_column(move(new_column)), if_column_not_exists(if_column_not_exists) {
}

AddColumnInfo::~AddColumnInfo() {
}

unique_ptr<AlterInfo> AddColumnInfo::Copy() const {
	return make_unique_base<AlterInfo, AddColumnInfo>(schema, name, if_exists, new_column.Copy(), if_column_not_exists);
}

void AddColumnInfo::SerializeAlterTable(FieldWriter &writer) const {
	writer.WriteSerializable(new_column);
	writer.WriteField<bool>(if_column_not_exists);
}

unique_ptr<AlterInfo> AddColumnInfo::Deserialize(FieldReader &reader, string schema, string table, bool if_exists) {
	auto new_column = reader.ReadRequiredSerializable<ColumnDefinition, ColumnDefinition>();
	auto if_column_not_exists = reader.ReadRequired<bool>();
	return make_unique<AddColumnInfo>(move(schema), move(table), if_exists, move(new_column), if_column_not_exists);
}

//===--------------------------------------------------------------------===//
// RemoveColumnInfo
//===--------------------------------------------------------------------===//
RemoveColumnInfo::RemoveColumnInfo(string schema, string table, bool if_exists, string removed_column,
                                   bool if_column_exists, bool cascade)
    : AlterTableInfo(AlterTableType::REMOVE_COLUMN, move(schema), move(table), if_exists),
      removed_column(move(removed_column)), if_column_exists(if_column_exists), cascade(cascade) {
}
RemoveColumnInfo::~RemoveColumnInfo() {
}

unique_ptr<AlterInfo> RemoveColumnInfo::Copy() const {
	return make_unique_base<AlterInfo, RemoveColumnInfo>(schema, name, if_exists, removed_column, if_column_exists,
	                                                     cascade);
}

void RemoveColumnInfo::SerializeAlterTable(FieldWriter &writer) const {
	writer.WriteString(removed_column);
	writer.WriteField<bool>(if_column_exists);
	writer.WriteField<bool>(cascade);
}

unique_ptr<AlterInfo> RemoveColumnInfo::Deserialize(FieldReader &reader, string schema, string table, bool if_exists) {
	auto new_name = reader.ReadRequired<string>();
	auto if_column_exists = reader.ReadRequired<bool>();
	auto cascade = reader.ReadRequired<bool>();
	return make_unique<RemoveColumnInfo>(move(schema), move(table), if_exists, new_name, if_column_exists, cascade);
}

//===--------------------------------------------------------------------===//
// ChangeColumnTypeInfo
//===--------------------------------------------------------------------===//
ChangeColumnTypeInfo::ChangeColumnTypeInfo(string schema_p, string table_p, bool if_exists_p, string column_name,
                                           LogicalType target_type, unique_ptr<ParsedExpression> expression)
    : AlterTableInfo(AlterTableType::ALTER_COLUMN_TYPE, move(schema_p), move(table_p), if_exists_p),
      column_name(move(column_name)), target_type(move(target_type)), expression(move(expression)) {
}
ChangeColumnTypeInfo::~ChangeColumnTypeInfo() {
}

unique_ptr<AlterInfo> ChangeColumnTypeInfo::Copy() const {
	return make_unique_base<AlterInfo, ChangeColumnTypeInfo>(schema, name, if_exists, column_name, target_type,
	                                                         expression->Copy());
}

void ChangeColumnTypeInfo::SerializeAlterTable(FieldWriter &writer) const {
	writer.WriteString(column_name);
	writer.WriteSerializable(target_type);
	writer.WriteOptional(expression);
}

unique_ptr<AlterInfo> ChangeColumnTypeInfo::Deserialize(FieldReader &reader, string schema, string table,
                                                        bool if_exists) {
	auto column_name = reader.ReadRequired<string>();
	auto target_type = reader.ReadRequiredSerializable<LogicalType, LogicalType>();
	auto expression = reader.ReadOptional<ParsedExpression>(nullptr);
	return make_unique<ChangeColumnTypeInfo>(move(schema), move(table), if_exists, move(column_name), move(target_type),
	                                         move(expression));
}

//===--------------------------------------------------------------------===//
// SetDefaultInfo
//===--------------------------------------------------------------------===//
SetDefaultInfo::SetDefaultInfo(string schema_p, string table_p, bool if_exists_p, string column_name_p,
                               unique_ptr<ParsedExpression> new_default)
    : AlterTableInfo(AlterTableType::SET_DEFAULT, move(schema_p), move(table_p), if_exists_p),
      column_name(move(column_name_p)), expression(move(new_default)) {
}
SetDefaultInfo::~SetDefaultInfo() {
}

unique_ptr<AlterInfo> SetDefaultInfo::Copy() const {
	return make_unique_base<AlterInfo, SetDefaultInfo>(schema, name, if_exists, column_name,
	                                                   expression ? expression->Copy() : nullptr);
}

void SetDefaultInfo::SerializeAlterTable(FieldWriter &writer) const {
	writer.WriteString(column_name);
	writer.WriteOptional(expression);
}

unique_ptr<AlterInfo> SetDefaultInfo::Deserialize(FieldReader &reader, string schema, string table, bool if_exists) {
	auto column_name = reader.ReadRequired<string>();
	auto new_default = reader.ReadOptional<ParsedExpression>(nullptr);
	return make_unique<SetDefaultInfo>(move(schema), move(table), if_exists, move(column_name), move(new_default));
}

//===--------------------------------------------------------------------===//
// SetNotNullInfo
//===--------------------------------------------------------------------===//
SetNotNullInfo::SetNotNullInfo(string schema_p, string table_p, bool if_exists_p, string column_name_p)
    : AlterTableInfo(AlterTableType::SET_NOT_NULL, move(schema_p), move(table_p), if_exists_p),
      column_name(move(column_name_p)) {
}
SetNotNullInfo::~SetNotNullInfo() {
}

unique_ptr<AlterInfo> SetNotNullInfo::Copy() const {
	return make_unique_base<AlterInfo, SetNotNullInfo>(schema, name, if_exists, column_name);
}

void SetNotNullInfo::SerializeAlterTable(FieldWriter &writer) const {
	writer.WriteString(column_name);
}

unique_ptr<AlterInfo> SetNotNullInfo::Deserialize(FieldReader &reader, string schema, string table, bool if_exists) {
	auto column_name = reader.ReadRequired<string>();
	return make_unique<SetNotNullInfo>(move(schema), move(table), if_exists, move(column_name));
}

//===--------------------------------------------------------------------===//
// DropNotNullInfo
//===--------------------------------------------------------------------===//
DropNotNullInfo::DropNotNullInfo(string schema_p, string table_p, bool if_exists_p, string column_name_p)
    : AlterTableInfo(AlterTableType::DROP_NOT_NULL, move(schema_p), move(table_p), if_exists_p),
      column_name(move(column_name_p)) {
}
DropNotNullInfo::~DropNotNullInfo() {
}

unique_ptr<AlterInfo> DropNotNullInfo::Copy() const {
	return make_unique_base<AlterInfo, DropNotNullInfo>(schema, name, if_exists, column_name);
}

void DropNotNullInfo::SerializeAlterTable(FieldWriter &writer) const {
	writer.WriteString(column_name);
}

unique_ptr<AlterInfo> DropNotNullInfo::Deserialize(FieldReader &reader, string schema, string table, bool if_exists) {
	auto column_name = reader.ReadRequired<string>();
	return make_unique<DropNotNullInfo>(move(schema), move(table), if_exists, move(column_name));
}

//===--------------------------------------------------------------------===//
// AlterForeignKeyInfo
//===--------------------------------------------------------------------===//
AlterForeignKeyInfo::AlterForeignKeyInfo(string schema_p, string table_p, bool if_exists_p, string fk_table,
                                         vector<string> pk_columns, vector<string> fk_columns,
                                         vector<PhysicalIndex> pk_keys, vector<PhysicalIndex> fk_keys,
                                         AlterForeignKeyType type_p)
    : AlterTableInfo(AlterTableType::FOREIGN_KEY_CONSTRAINT, move(schema_p), move(table_p), if_exists_p),
      fk_table(move(fk_table)), pk_columns(move(pk_columns)), fk_columns(move(fk_columns)), pk_keys(move(pk_keys)),
      fk_keys(move(fk_keys)), type(type_p) {
}
AlterForeignKeyInfo::~AlterForeignKeyInfo() {
}

unique_ptr<AlterInfo> AlterForeignKeyInfo::Copy() const {
	return make_unique_base<AlterInfo, AlterForeignKeyInfo>(schema, name, if_exists, fk_table, pk_columns, fk_columns,
	                                                        pk_keys, fk_keys, type);
}

void AlterForeignKeyInfo::SerializeAlterTable(FieldWriter &writer) const {
	writer.WriteString(fk_table);
	writer.WriteList<string>(pk_columns);
	writer.WriteList<string>(fk_columns);
	writer.WriteIndexList<PhysicalIndex>(pk_keys);
	writer.WriteIndexList<PhysicalIndex>(fk_keys);
	writer.WriteField<AlterForeignKeyType>(type);
}

unique_ptr<AlterInfo> AlterForeignKeyInfo::Deserialize(FieldReader &reader, string schema, string table,
                                                       bool if_exists) {
	auto fk_table = reader.ReadRequired<string>();
	auto pk_columns = reader.ReadRequiredList<string>();
	auto fk_columns = reader.ReadRequiredList<string>();
	auto pk_keys = reader.ReadRequiredIndexList<PhysicalIndex>();
	auto fk_keys = reader.ReadRequiredIndexList<PhysicalIndex>();
	auto type = reader.ReadRequired<AlterForeignKeyType>();
	return make_unique<AlterForeignKeyInfo>(move(schema), move(table), if_exists, move(fk_table), move(pk_columns),
	                                        move(fk_columns), move(pk_keys), move(fk_keys), type);
}

//===--------------------------------------------------------------------===//
// Alter View
//===--------------------------------------------------------------------===//
AlterViewInfo::AlterViewInfo(AlterViewType type, string schema_p, string view_p, bool if_exists_p)
    : AlterInfo(AlterType::ALTER_VIEW, move(schema_p), move(view_p), if_exists_p), alter_view_type(type) {
}
AlterViewInfo::~AlterViewInfo() {
}

CatalogType AlterViewInfo::GetCatalogType() const {
	return CatalogType::VIEW_ENTRY;
}

void AlterViewInfo::Serialize(FieldWriter &writer) const {
	writer.WriteField<AlterViewType>(alter_view_type);
	writer.WriteString(schema);
	writer.WriteString(name);
	writer.WriteField<bool>(if_exists);
	SerializeAlterView(writer);
}

unique_ptr<AlterInfo> AlterViewInfo::Deserialize(FieldReader &reader) {
	auto type = reader.ReadRequired<AlterViewType>();
	auto schema = reader.ReadRequired<string>();
	auto view = reader.ReadRequired<string>();
	auto if_exists = reader.ReadRequired<bool>();
	unique_ptr<AlterViewInfo> info;
	switch (type) {
	case AlterViewType::RENAME_VIEW:
		return RenameViewInfo::Deserialize(reader, schema, view, if_exists);
	default:
		throw SerializationException("Unknown alter view type for deserialization!");
	}
}

//===--------------------------------------------------------------------===//
// RenameViewInfo
//===--------------------------------------------------------------------===//
RenameViewInfo::RenameViewInfo(string schema_p, string view_p, bool if_exists_p, string new_name_p)
    : AlterViewInfo(AlterViewType::RENAME_VIEW, move(schema_p), move(view_p), if_exists_p),
      new_view_name(move(new_name_p)) {
}
RenameViewInfo::~RenameViewInfo() {
}

unique_ptr<AlterInfo> RenameViewInfo::Copy() const {
	return make_unique_base<AlterInfo, RenameViewInfo>(schema, name, if_exists, new_view_name);
}

void RenameViewInfo::SerializeAlterView(FieldWriter &writer) const {
	writer.WriteString(new_view_name);
}

unique_ptr<AlterInfo> RenameViewInfo::Deserialize(FieldReader &reader, string schema, string view, bool if_exists) {
	auto new_name = reader.ReadRequired<string>();
	return make_unique<RenameViewInfo>(move(schema), move(view), if_exists, new_name);
}
} // namespace duckdb




namespace duckdb {

unique_ptr<CreateInfo> CreateIndexInfo::Copy() const {

	auto result = make_unique<CreateIndexInfo>();
	CopyProperties(*result);

	result->index_type = index_type;
	result->index_name = index_name;
	result->constraint_type = constraint_type;
	result->table = unique_ptr_cast<TableRef, BaseTableRef>(table->Copy());
	for (auto &expr : expressions) {
		result->expressions.push_back(expr->Copy());
	}

	result->scan_types = scan_types;
	result->names = names;
	result->column_ids = column_ids;
	return move(result);
}

void CreateIndexInfo::SerializeInternal(Serializer &serializer) const {

	FieldWriter writer(serializer);
	writer.WriteField(index_type);
	writer.WriteString(index_name);
	writer.WriteField(constraint_type);

	writer.WriteSerializableList<ParsedExpression>(expressions);
	writer.WriteSerializableList<ParsedExpression>(parsed_expressions);

	writer.WriteRegularSerializableList(scan_types);
	writer.WriteList<string>(names);
	writer.WriteList<column_t>(column_ids);

	writer.Finalize();
}

unique_ptr<CreateIndexInfo> CreateIndexInfo::Deserialize(Deserializer &deserializer) {

	auto result = make_unique<CreateIndexInfo>();
	result->DeserializeBase(deserializer);

	FieldReader reader(deserializer);
	result->index_type = reader.ReadRequired<IndexType>();
	result->index_name = reader.ReadRequired<string>();
	result->constraint_type = reader.ReadRequired<IndexConstraintType>();

	result->expressions = reader.ReadRequiredSerializableList<ParsedExpression>();
	result->parsed_expressions = reader.ReadRequiredSerializableList<ParsedExpression>();

	result->scan_types = reader.ReadRequiredSerializableList<LogicalType, LogicalType>();
	result->names = reader.ReadRequiredList<string>();
	result->column_ids = reader.ReadRequiredList<column_t>();

	reader.Finalize();
	return result;
}
} // namespace duckdb








namespace duckdb {
void CreateInfo::DeserializeBase(Deserializer &deserializer) {
	this->schema = deserializer.Read<string>();
	this->on_conflict = deserializer.Read<OnCreateConflict>();
	this->temporary = deserializer.Read<bool>();
	this->internal = deserializer.Read<bool>();
	this->sql = deserializer.Read<string>();
}

void CreateInfo::Serialize(Serializer &serializer) const {
	serializer.Write(type);
	serializer.WriteString(schema);
	serializer.Write(on_conflict);
	serializer.Write(temporary);
	serializer.Write(internal);
	serializer.WriteString(sql);
	SerializeInternal(serializer);
}

unique_ptr<CreateInfo> CreateInfo::Deserialize(Deserializer &deserializer) {
	auto type = deserializer.Read<CatalogType>();
	switch (type) {
	case CatalogType::INDEX_ENTRY:
		return CreateIndexInfo::Deserialize(deserializer);
	case CatalogType::TABLE_ENTRY:
		return CreateTableInfo::Deserialize(deserializer);
	case CatalogType::SCHEMA_ENTRY:
		return CreateSchemaInfo::Deserialize(deserializer);
	case CatalogType::VIEW_ENTRY:
		return CreateViewInfo::Deserialize(deserializer);
	default:
		throw NotImplementedException("Cannot deserialize '%s'", CatalogTypeToString(type));
	}
}

unique_ptr<CreateInfo> CreateInfo::Deserialize(Deserializer &source, PlanDeserializationState &state) {
	return Deserialize(source);
}

void CreateInfo::CopyProperties(CreateInfo &other) const {
	other.type = type;
	other.schema = schema;
	other.on_conflict = on_conflict;
	other.temporary = temporary;
	other.internal = internal;
	other.sql = sql;
}

unique_ptr<AlterInfo> CreateInfo::GetAlterInfo() const {
	throw NotImplementedException("GetAlterInfo not implemented for this type");
}

} // namespace duckdb



namespace duckdb {

CreateScalarFunctionInfo::CreateScalarFunctionInfo(ScalarFunction function)
    : CreateFunctionInfo(CatalogType::SCALAR_FUNCTION_ENTRY), functions(function.name) {
	name = function.name;
	functions.AddFunction(move(function));
}
CreateScalarFunctionInfo::CreateScalarFunctionInfo(ScalarFunctionSet set)
    : CreateFunctionInfo(CatalogType::SCALAR_FUNCTION_ENTRY), functions(move(set)) {
	name = functions.name;
	for (auto &func : functions.functions) {
		func.name = functions.name;
	}
}

unique_ptr<CreateInfo> CreateScalarFunctionInfo::Copy() const {
	ScalarFunctionSet set(name);
	set.functions = functions.functions;
	auto result = make_unique<CreateScalarFunctionInfo>(move(set));
	CopyProperties(*result);
	return move(result);
}

unique_ptr<AlterInfo> CreateScalarFunctionInfo::GetAlterInfo() const {
	return make_unique_base<AlterInfo, AddFunctionOverloadInfo>(schema, name, true, functions);
}

} // namespace duckdb


namespace duckdb {

CreateTableInfo::CreateTableInfo() : CreateInfo(CatalogType::TABLE_ENTRY, INVALID_SCHEMA) {
}

CreateTableInfo::CreateTableInfo(string schema_p, string name_p)
    : CreateInfo(CatalogType::TABLE_ENTRY, move(schema_p)), table(move(name_p)) {
}

void CreateTableInfo::SerializeInternal(Serializer &serializer) const {
	FieldWriter writer(serializer);
	writer.WriteString(table);
	columns.Serialize(writer);
	writer.WriteSerializableList(constraints);
	writer.WriteOptional(query);
	writer.Finalize();
}

unique_ptr<CreateTableInfo> CreateTableInfo::Deserialize(Deserializer &deserializer) {
	auto result = make_unique<CreateTableInfo>();
	result->DeserializeBase(deserializer);

	FieldReader reader(deserializer);
	result->table = reader.ReadRequired<string>();
	result->columns = ColumnList::Deserialize(reader);
	result->constraints = reader.ReadRequiredSerializableList<Constraint>();
	result->query = reader.ReadOptional<SelectStatement>(nullptr);
	reader.Finalize();

	return result;
}

unique_ptr<CreateInfo> CreateTableInfo::Copy() const {
	auto result = make_unique<CreateTableInfo>(schema, table);
	CopyProperties(*result);
	result->columns = columns.Copy();
	for (auto &constraint : constraints) {
		result->constraints.push_back(constraint->Copy());
	}
	if (query) {
		result->query = unique_ptr_cast<SQLStatement, SelectStatement>(query->Copy());
	}
	return move(result);
}

} // namespace duckdb



namespace duckdb {

string SampleMethodToString(SampleMethod method) {
	switch (method) {
	case SampleMethod::SYSTEM_SAMPLE:
		return "System";
	case SampleMethod::BERNOULLI_SAMPLE:
		return "Bernoulli";
	case SampleMethod::RESERVOIR_SAMPLE:
		return "Reservoir";
	default:
		return "Unknown";
	}
}

void SampleOptions::Serialize(Serializer &serializer) {
	FieldWriter writer(serializer);
	writer.WriteSerializable(sample_size);
	writer.WriteField<bool>(is_percentage);
	writer.WriteField<SampleMethod>(method);
	writer.WriteField<int64_t>(seed);
	writer.Finalize();
}

unique_ptr<SampleOptions> SampleOptions::Deserialize(Deserializer &source) {
	auto result = make_unique<SampleOptions>();

	FieldReader reader(source);
	result->sample_size = reader.ReadRequiredSerializable<Value, Value>();
	result->is_percentage = reader.ReadRequired<bool>();
	result->method = reader.ReadRequired<SampleMethod>();
	result->seed = reader.ReadRequired<int64_t>();
	reader.Finalize();

	return result;
}

unique_ptr<SampleOptions> SampleOptions::Copy() {
	auto result = make_unique<SampleOptions>();
	result->sample_size = sample_size;
	result->is_percentage = is_percentage;
	result->method = method;
	result->seed = seed;
	return result;
}

bool SampleOptions::Equals(SampleOptions *a, SampleOptions *b) {
	if (a == b) {
		return true;
	}
	if (!a || !b) {
		return false;
	}
	if (a->sample_size != b->sample_size || a->is_percentage != b->is_percentage || a->method != b->method ||
	    a->seed != b->seed) {
		return false;
	}
	return true;
}

} // namespace duckdb








namespace duckdb {

bool ParsedExpression::IsAggregate() const {
	bool is_aggregate = false;
	ParsedExpressionIterator::EnumerateChildren(
	    *this, [&](const ParsedExpression &child) { is_aggregate |= child.IsAggregate(); });
	return is_aggregate;
}

bool ParsedExpression::IsWindow() const {
	bool is_window = false;
	ParsedExpressionIterator::EnumerateChildren(*this,
	                                            [&](const ParsedExpression &child) { is_window |= child.IsWindow(); });
	return is_window;
}

bool ParsedExpression::IsScalar() const {
	bool is_scalar = true;
	ParsedExpressionIterator::EnumerateChildren(*this, [&](const ParsedExpression &child) {
		if (!child.IsScalar()) {
			is_scalar = false;
		}
	});
	return is_scalar;
}

bool ParsedExpression::HasParameter() const {
	bool has_parameter = false;
	ParsedExpressionIterator::EnumerateChildren(
	    *this, [&](const ParsedExpression &child) { has_parameter |= child.HasParameter(); });
	return has_parameter;
}

bool ParsedExpression::HasSubquery() const {
	bool has_subquery = false;
	ParsedExpressionIterator::EnumerateChildren(
	    *this, [&](const ParsedExpression &child) { has_subquery |= child.HasSubquery(); });
	return has_subquery;
}

bool ParsedExpression::Equals(const BaseExpression *other) const {
	if (!BaseExpression::Equals(other)) {
		return false;
	}
	switch (expression_class) {
	case ExpressionClass::BETWEEN:
		return BetweenExpression::Equals((BetweenExpression *)this, (BetweenExpression *)other);
	case ExpressionClass::CASE:
		return CaseExpression::Equals((CaseExpression *)this, (CaseExpression *)other);
	case ExpressionClass::CAST:
		return CastExpression::Equals((CastExpression *)this, (CastExpression *)other);
	case ExpressionClass::COLLATE:
		return CollateExpression::Equals((CollateExpression *)this, (CollateExpression *)other);
	case ExpressionClass::COLUMN_REF:
		return ColumnRefExpression::Equals((ColumnRefExpression *)this, (ColumnRefExpression *)other);
	case ExpressionClass::COMPARISON:
		return ComparisonExpression::Equals((ComparisonExpression *)this, (ComparisonExpression *)other);
	case ExpressionClass::CONJUNCTION:
		return ConjunctionExpression::Equals((ConjunctionExpression *)this, (ConjunctionExpression *)other);
	case ExpressionClass::CONSTANT:
		return ConstantExpression::Equals((ConstantExpression *)this, (ConstantExpression *)other);
	case ExpressionClass::DEFAULT:
		return true;
	case ExpressionClass::FUNCTION:
		return FunctionExpression::Equals((FunctionExpression *)this, (FunctionExpression *)other);
	case ExpressionClass::LAMBDA:
		return LambdaExpression::Equals((LambdaExpression *)this, (LambdaExpression *)other);
	case ExpressionClass::OPERATOR:
		return OperatorExpression::Equals((OperatorExpression *)this, (OperatorExpression *)other);
	case ExpressionClass::PARAMETER:
		return ParameterExpression::Equals((ParameterExpression *)this, (ParameterExpression *)other);
	case ExpressionClass::POSITIONAL_REFERENCE:
		return PositionalReferenceExpression::Equals((PositionalReferenceExpression *)this,
		                                             (PositionalReferenceExpression *)other);
	case ExpressionClass::STAR:
		return StarExpression::Equals((StarExpression *)this, (StarExpression *)other);
	case ExpressionClass::SUBQUERY:
		return SubqueryExpression::Equals((SubqueryExpression *)this, (SubqueryExpression *)other);
	case ExpressionClass::WINDOW:
		return WindowExpression::Equals((WindowExpression *)this, (WindowExpression *)other);
	default:
		throw SerializationException("Unsupported type for expression comparison!");
	}
}

hash_t ParsedExpression::Hash() const {
	hash_t hash = duckdb::Hash<uint32_t>((uint32_t)type);
	ParsedExpressionIterator::EnumerateChildren(
	    *this, [&](const ParsedExpression &child) { hash = CombineHash(child.Hash(), hash); });
	return hash;
}

void ParsedExpression::Serialize(Serializer &serializer) const {
	FieldWriter writer(serializer);
	writer.WriteField<ExpressionClass>(GetExpressionClass());
	writer.WriteField<ExpressionType>(type);
	writer.WriteString(alias);
	Serialize(writer);
	writer.Finalize();
}

unique_ptr<ParsedExpression> ParsedExpression::Deserialize(Deserializer &source) {
	FieldReader reader(source);
	auto expression_class = reader.ReadRequired<ExpressionClass>();
	auto type = reader.ReadRequired<ExpressionType>();
	auto alias = reader.ReadRequired<string>();
	unique_ptr<ParsedExpression> result;
	switch (expression_class) {
	case ExpressionClass::BETWEEN:
		result = BetweenExpression::Deserialize(type, reader);
		break;
	case ExpressionClass::CASE:
		result = CaseExpression::Deserialize(type, reader);
		break;
	case ExpressionClass::CAST:
		result = CastExpression::Deserialize(type, reader);
		break;
	case ExpressionClass::COLLATE:
		result = CollateExpression::Deserialize(type, reader);
		break;
	case ExpressionClass::COLUMN_REF:
		result = ColumnRefExpression::Deserialize(type, reader);
		break;
	case ExpressionClass::COMPARISON:
		result = ComparisonExpression::Deserialize(type, reader);
		break;
	case ExpressionClass::CONJUNCTION:
		result = ConjunctionExpression::Deserialize(type, reader);
		break;
	case ExpressionClass::CONSTANT:
		result = ConstantExpression::Deserialize(type, reader);
		break;
	case ExpressionClass::DEFAULT:
		result = DefaultExpression::Deserialize(type, reader);
		break;
	case ExpressionClass::FUNCTION:
		result = FunctionExpression::Deserialize(type, reader);
		break;
	case ExpressionClass::LAMBDA:
		result = LambdaExpression::Deserialize(type, reader);
		break;
	case ExpressionClass::OPERATOR:
		result = OperatorExpression::Deserialize(type, reader);
		break;
	case ExpressionClass::PARAMETER:
		result = ParameterExpression::Deserialize(type, reader);
		break;
	case ExpressionClass::POSITIONAL_REFERENCE:
		result = PositionalReferenceExpression::Deserialize(type, reader);
		break;
	case ExpressionClass::STAR:
		result = StarExpression::Deserialize(type, reader);
		break;
	case ExpressionClass::SUBQUERY:
		result = SubqueryExpression::Deserialize(type, reader);
		break;
	case ExpressionClass::WINDOW:
		result = WindowExpression::Deserialize(type, reader);
		break;
	default:
		throw SerializationException("Unsupported type for expression deserialization: '%s'!",
		                             ExpressionClassToString(expression_class));
	}
	result->alias = alias;
	reader.Finalize();
	return result;
}

} // namespace duckdb









namespace duckdb {

void ParsedExpressionIterator::EnumerateChildren(const ParsedExpression &expression,
                                                 const std::function<void(const ParsedExpression &child)> &callback) {
	EnumerateChildren((ParsedExpression &)expression, [&](unique_ptr<ParsedExpression> &child) {
		D_ASSERT(child);
		callback(*child);
	});
}

void ParsedExpressionIterator::EnumerateChildren(ParsedExpression &expr,
                                                 const std::function<void(ParsedExpression &child)> &callback) {
	EnumerateChildren(expr, [&](unique_ptr<ParsedExpression> &child) {
		D_ASSERT(child);
		callback(*child);
	});
}

void ParsedExpressionIterator::EnumerateChildren(
    ParsedExpression &expr, const std::function<void(unique_ptr<ParsedExpression> &child)> &callback) {
	switch (expr.expression_class) {
	case ExpressionClass::BETWEEN: {
		auto &cast_expr = (BetweenExpression &)expr;
		callback(cast_expr.input);
		callback(cast_expr.lower);
		callback(cast_expr.upper);
		break;
	}
	case ExpressionClass::CASE: {
		auto &case_expr = (CaseExpression &)expr;
		for (auto &check : case_expr.case_checks) {
			callback(check.when_expr);
			callback(check.then_expr);
		}
		callback(case_expr.else_expr);
		break;
	}
	case ExpressionClass::CAST: {
		auto &cast_expr = (CastExpression &)expr;
		callback(cast_expr.child);
		break;
	}
	case ExpressionClass::COLLATE: {
		auto &cast_expr = (CollateExpression &)expr;
		callback(cast_expr.child);
		break;
	}
	case ExpressionClass::COMPARISON: {
		auto &comp_expr = (ComparisonExpression &)expr;
		callback(comp_expr.left);
		callback(comp_expr.right);
		break;
	}
	case ExpressionClass::CONJUNCTION: {
		auto &conj_expr = (ConjunctionExpression &)expr;
		for (auto &child : conj_expr.children) {
			callback(child);
		}
		break;
	}

	case ExpressionClass::FUNCTION: {
		auto &func_expr = (FunctionExpression &)expr;
		for (auto &child : func_expr.children) {
			callback(child);
		}
		if (func_expr.filter) {
			callback(func_expr.filter);
		}
		if (func_expr.order_bys) {
			for (auto &order : func_expr.order_bys->orders) {
				callback(order.expression);
			}
		}
		break;
	}
	case ExpressionClass::LAMBDA: {
		auto &lambda_expr = (LambdaExpression &)expr;
		callback(lambda_expr.lhs);
		callback(lambda_expr.expr);
		break;
	}
	case ExpressionClass::OPERATOR: {
		auto &op_expr = (OperatorExpression &)expr;
		for (auto &child : op_expr.children) {
			callback(child);
		}
		break;
	}
	case ExpressionClass::SUBQUERY: {
		auto &subquery_expr = (SubqueryExpression &)expr;
		if (subquery_expr.child) {
			callback(subquery_expr.child);
		}
		break;
	}
	case ExpressionClass::WINDOW: {
		auto &window_expr = (WindowExpression &)expr;
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
	case ExpressionClass::BOUND_EXPRESSION:
	case ExpressionClass::COLUMN_REF:
	case ExpressionClass::CONSTANT:
	case ExpressionClass::DEFAULT:
	case ExpressionClass::STAR:
	case ExpressionClass::PARAMETER:
	case ExpressionClass::POSITIONAL_REFERENCE:
		// these node types have no children
		break;
	default:
		// called on non ParsedExpression type!
		throw NotImplementedException("Unimplemented expression class");
	}
}

void ParsedExpressionIterator::EnumerateQueryNodeModifiers(
    QueryNode &node, const std::function<void(unique_ptr<ParsedExpression> &child)> &callback) {

	for (auto &modifier : node.modifiers) {
		switch (modifier->type) {
		case ResultModifierType::LIMIT_MODIFIER: {
			auto &limit_modifier = (LimitModifier &)*modifier;
			if (limit_modifier.limit) {
				callback(limit_modifier.limit);
			}
			if (limit_modifier.offset) {
				callback(limit_modifier.offset);
			}
		} break;

		case ResultModifierType::LIMIT_PERCENT_MODIFIER: {
			auto &limit_modifier = (LimitPercentModifier &)*modifier;
			if (limit_modifier.limit) {
				callback(limit_modifier.limit);
			}
			if (limit_modifier.offset) {
				callback(limit_modifier.offset);
			}
		} break;

		case ResultModifierType::ORDER_MODIFIER: {
			auto &order_modifier = (OrderModifier &)*modifier;
			for (auto &order : order_modifier.orders) {
				callback(order.expression);
			}
		} break;

		case ResultModifierType::DISTINCT_MODIFIER: {
			auto &distinct_modifier = (DistinctModifier &)*modifier;
			for (auto &target : distinct_modifier.distinct_on_targets) {
				callback(target);
			}
		} break;

		// do nothing
		default:
			break;
		}
	}
}

void ParsedExpressionIterator::EnumerateTableRefChildren(
    TableRef &ref, const std::function<void(unique_ptr<ParsedExpression> &child)> &callback) {
	switch (ref.type) {
	case TableReferenceType::CROSS_PRODUCT: {
		auto &cp_ref = (CrossProductRef &)ref;
		EnumerateTableRefChildren(*cp_ref.left, callback);
		EnumerateTableRefChildren(*cp_ref.right, callback);
		break;
	}
	case TableReferenceType::EXPRESSION_LIST: {
		auto &el_ref = (ExpressionListRef &)ref;
		for (idx_t i = 0; i < el_ref.values.size(); i++) {
			for (idx_t j = 0; j < el_ref.values[i].size(); j++) {
				callback(el_ref.values[i][j]);
			}
		}
		break;
	}
	case TableReferenceType::JOIN: {
		auto &j_ref = (JoinRef &)ref;
		EnumerateTableRefChildren(*j_ref.left, callback);
		EnumerateTableRefChildren(*j_ref.right, callback);
		if (j_ref.condition) {
			callback(j_ref.condition);
		}
		break;
	}
	case TableReferenceType::SUBQUERY: {
		auto &sq_ref = (SubqueryRef &)ref;
		EnumerateQueryNodeChildren(*sq_ref.subquery->node, callback);
		break;
	}
	case TableReferenceType::TABLE_FUNCTION: {
		auto &tf_ref = (TableFunctionRef &)ref;
		callback(tf_ref.function);
		break;
	}
	case TableReferenceType::BASE_TABLE:
	case TableReferenceType::EMPTY:
		// these TableRefs do not need to be unfolded
		break;
	default:
		throw NotImplementedException("TableRef type not implemented for traversal");
	}
}

void ParsedExpressionIterator::EnumerateQueryNodeChildren(
    QueryNode &node, const std::function<void(unique_ptr<ParsedExpression> &child)> &callback) {
	switch (node.type) {
	case QueryNodeType::RECURSIVE_CTE_NODE: {
		auto &rcte_node = (RecursiveCTENode &)node;
		EnumerateQueryNodeChildren(*rcte_node.left, callback);
		EnumerateQueryNodeChildren(*rcte_node.right, callback);
		break;
	}
	case QueryNodeType::SELECT_NODE: {
		auto &sel_node = (SelectNode &)node;
		for (idx_t i = 0; i < sel_node.select_list.size(); i++) {
			callback(sel_node.select_list[i]);
		}
		for (idx_t i = 0; i < sel_node.groups.group_expressions.size(); i++) {
			callback(sel_node.groups.group_expressions[i]);
		}
		if (sel_node.where_clause) {
			callback(sel_node.where_clause);
		}
		if (sel_node.having) {
			callback(sel_node.having);
		}
		if (sel_node.qualify) {
			callback(sel_node.qualify);
		}

		EnumerateTableRefChildren(*sel_node.from_table.get(), callback);
		break;
	}
	case QueryNodeType::SET_OPERATION_NODE: {
		auto &setop_node = (SetOperationNode &)node;
		EnumerateQueryNodeChildren(*setop_node.left, callback);
		EnumerateQueryNodeChildren(*setop_node.right, callback);
		break;
	}
	default:
		throw NotImplementedException("QueryNode type not implemented for traversal");
	}

	if (!node.modifiers.empty()) {
		EnumerateQueryNodeModifiers(node, callback);
	}

	for (auto &kv : node.cte_map.map) {
		EnumerateQueryNodeChildren(*kv.second->query->node, callback);
	}
}

} // namespace duckdb
















namespace duckdb {

Parser::Parser(ParserOptions options_p) : options(options_p) {
}

void Parser::ParseQuery(const string &query) {
	Transformer transformer(options.max_expression_depth);
	{
		PostgresParser::SetPreserveIdentifierCase(options.preserve_identifier_case);
		PostgresParser parser;
		parser.Parse(query);

		if (!parser.success) {
			if (options.extensions) {
				for (auto &ext : *options.extensions) {
					D_ASSERT(ext.parse_function);
					auto result = ext.parse_function(ext.parser_info.get(), query);
					if (result.type == ParserExtensionResultType::PARSE_SUCCESSFUL) {
						auto statement = make_unique<ExtensionStatement>(ext, move(result.parse_data));
						statement->stmt_length = query.size();
						statement->stmt_location = 0;
						statements.push_back(move(statement));
						return;
					}
					if (result.type == ParserExtensionResultType::DISPLAY_EXTENSION_ERROR) {
						throw ParserException(result.error);
					}
				}
			}
			throw ParserException(QueryErrorContext::Format(query, parser.error_message, parser.error_location - 1));
		}

		if (!parser.parse_tree) {
			// empty statement
			return;
		}

		// if it succeeded, we transform the Postgres parse tree into a list of
		// SQLStatements
		transformer.TransformParseTree(parser.parse_tree, statements);
	}
	if (!statements.empty()) {
		auto &last_statement = statements.back();
		last_statement->stmt_length = query.size() - last_statement->stmt_location;
		for (auto &statement : statements) {
			statement->query = query;
			if (statement->type == StatementType::CREATE_STATEMENT) {
				auto &create = (CreateStatement &)*statement;
				create.info->sql = query.substr(statement->stmt_location, statement->stmt_length);
			}
		}
	}
}

vector<SimplifiedToken> Parser::Tokenize(const string &query) {
	auto pg_tokens = PostgresParser::Tokenize(query);
	vector<SimplifiedToken> result;
	result.reserve(pg_tokens.size());
	for (auto &pg_token : pg_tokens) {
		SimplifiedToken token;
		switch (pg_token.type) {
		case duckdb_libpgquery::PGSimplifiedTokenType::PG_SIMPLIFIED_TOKEN_IDENTIFIER:
			token.type = SimplifiedTokenType::SIMPLIFIED_TOKEN_IDENTIFIER;
			break;
		case duckdb_libpgquery::PGSimplifiedTokenType::PG_SIMPLIFIED_TOKEN_NUMERIC_CONSTANT:
			token.type = SimplifiedTokenType::SIMPLIFIED_TOKEN_NUMERIC_CONSTANT;
			break;
		case duckdb_libpgquery::PGSimplifiedTokenType::PG_SIMPLIFIED_TOKEN_STRING_CONSTANT:
			token.type = SimplifiedTokenType::SIMPLIFIED_TOKEN_STRING_CONSTANT;
			break;
		case duckdb_libpgquery::PGSimplifiedTokenType::PG_SIMPLIFIED_TOKEN_OPERATOR:
			token.type = SimplifiedTokenType::SIMPLIFIED_TOKEN_OPERATOR;
			break;
		case duckdb_libpgquery::PGSimplifiedTokenType::PG_SIMPLIFIED_TOKEN_KEYWORD:
			token.type = SimplifiedTokenType::SIMPLIFIED_TOKEN_KEYWORD;
			break;
		// comments are not supported by our tokenizer right now
		case duckdb_libpgquery::PGSimplifiedTokenType::PG_SIMPLIFIED_TOKEN_COMMENT: // LCOV_EXCL_START
			token.type = SimplifiedTokenType::SIMPLIFIED_TOKEN_COMMENT;
			break;
		default:
			throw InternalException("Unrecognized token category");
		} // LCOV_EXCL_STOP
		token.start = pg_token.start;
		result.push_back(token);
	}
	return result;
}

bool Parser::IsKeyword(const string &text) {
	return PostgresParser::IsKeyword(text);
}

vector<ParserKeyword> Parser::KeywordList() {
	auto keywords = PostgresParser::KeywordList();
	vector<ParserKeyword> result;
	for (auto &kw : keywords) {
		ParserKeyword res;
		res.name = kw.text;
		switch (kw.category) {
		case duckdb_libpgquery::PGKeywordCategory::PG_KEYWORD_RESERVED:
			res.category = KeywordCategory::KEYWORD_RESERVED;
			break;
		case duckdb_libpgquery::PGKeywordCategory::PG_KEYWORD_UNRESERVED:
			res.category = KeywordCategory::KEYWORD_UNRESERVED;
			break;
		case duckdb_libpgquery::PGKeywordCategory::PG_KEYWORD_TYPE_FUNC:
			res.category = KeywordCategory::KEYWORD_TYPE_FUNC;
			break;
		case duckdb_libpgquery::PGKeywordCategory::PG_KEYWORD_COL_NAME:
			res.category = KeywordCategory::KEYWORD_COL_NAME;
			break;
		default:
			throw InternalException("Unrecognized keyword category");
		}
		result.push_back(res);
	}
	return result;
}

vector<unique_ptr<ParsedExpression>> Parser::ParseExpressionList(const string &select_list, ParserOptions options) {
	// construct a mock query prefixed with SELECT
	string mock_query = "SELECT " + select_list;
	// parse the query
	Parser parser(options);
	parser.ParseQuery(mock_query);
	// check the statements
	if (parser.statements.size() != 1 || parser.statements[0]->type != StatementType::SELECT_STATEMENT) {
		throw ParserException("Expected a single SELECT statement");
	}
	auto &select = (SelectStatement &)*parser.statements[0];
	if (select.node->type != QueryNodeType::SELECT_NODE) {
		throw ParserException("Expected a single SELECT node");
	}
	auto &select_node = (SelectNode &)*select.node;
	return move(select_node.select_list);
}

vector<OrderByNode> Parser::ParseOrderList(const string &select_list, ParserOptions options) {
	// construct a mock query
	string mock_query = "SELECT * FROM tbl ORDER BY " + select_list;
	// parse the query
	Parser parser(options);
	parser.ParseQuery(mock_query);
	// check the statements
	if (parser.statements.size() != 1 || parser.statements[0]->type != StatementType::SELECT_STATEMENT) {
		throw ParserException("Expected a single SELECT statement");
	}
	auto &select = (SelectStatement &)*parser.statements[0];
	if (select.node->type != QueryNodeType::SELECT_NODE) {
		throw ParserException("Expected a single SELECT node");
	}
	auto &select_node = (SelectNode &)*select.node;
	if (select_node.modifiers.empty() || select_node.modifiers[0]->type != ResultModifierType::ORDER_MODIFIER ||
	    select_node.modifiers.size() != 1) {
		throw ParserException("Expected a single ORDER clause");
	}
	auto &order = (OrderModifier &)*select_node.modifiers[0];
	return move(order.orders);
}

void Parser::ParseUpdateList(const string &update_list, vector<string> &update_columns,
                             vector<unique_ptr<ParsedExpression>> &expressions, ParserOptions options) {
	// construct a mock query
	string mock_query = "UPDATE tbl SET " + update_list;
	// parse the query
	Parser parser(options);
	parser.ParseQuery(mock_query);
	// check the statements
	if (parser.statements.size() != 1 || parser.statements[0]->type != StatementType::UPDATE_STATEMENT) {
		throw ParserException("Expected a single UPDATE statement");
	}
	auto &update = (UpdateStatement &)*parser.statements[0];
	update_columns = move(update.columns);
	expressions = move(update.expressions);
}

vector<vector<unique_ptr<ParsedExpression>>> Parser::ParseValuesList(const string &value_list, ParserOptions options) {
	// construct a mock query
	string mock_query = "VALUES " + value_list;
	// parse the query
	Parser parser(options);
	parser.ParseQuery(mock_query);
	// check the statements
	if (parser.statements.size() != 1 || parser.statements[0]->type != StatementType::SELECT_STATEMENT) {
		throw ParserException("Expected a single SELECT statement");
	}
	auto &select = (SelectStatement &)*parser.statements[0];
	if (select.node->type != QueryNodeType::SELECT_NODE) {
		throw ParserException("Expected a single SELECT node");
	}
	auto &select_node = (SelectNode &)*select.node;
	if (!select_node.from_table || select_node.from_table->type != TableReferenceType::EXPRESSION_LIST) {
		throw ParserException("Expected a single VALUES statement");
	}
	auto &values_list = (ExpressionListRef &)*select_node.from_table;
	return move(values_list.values);
}

ColumnList Parser::ParseColumnList(const string &column_list, ParserOptions options) {
	string mock_query = "CREATE TABLE blabla (" + column_list + ")";
	Parser parser(options);
	parser.ParseQuery(mock_query);
	if (parser.statements.size() != 1 || parser.statements[0]->type != StatementType::CREATE_STATEMENT) {
		throw ParserException("Expected a single CREATE statement");
	}
	auto &create = (CreateStatement &)*parser.statements[0];
	if (create.info->type != CatalogType::TABLE_ENTRY) {
		throw InternalException("Expected a single CREATE TABLE statement");
	}
	auto &info = ((CreateTableInfo &)*create.info);
	return move(info.columns);
}

} // namespace duckdb







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
	auto other = (RecursiveCTENode *)other_p;

	if (other->union_all != union_all) {
		return false;
	}
	if (!left->Equals(other->left.get())) {
		return false;
	}
	if (!right->Equals(other->right.get())) {
		return false;
	}
	return true;
}

unique_ptr<QueryNode> RecursiveCTENode::Copy() const {
	auto result = make_unique<RecursiveCTENode>();
	result->ctename = ctename;
	result->union_all = union_all;
	result->left = left->Copy();
	result->right = right->Copy();
	result->aliases = aliases;
	this->CopyProperties(*result);
	return move(result);
}

void RecursiveCTENode::Serialize(FieldWriter &writer) const {
	writer.WriteString(ctename);
	writer.WriteField<bool>(union_all);
	writer.WriteSerializable(*left);
	writer.WriteSerializable(*right);
	writer.WriteList<string>(aliases);
}

unique_ptr<QueryNode> RecursiveCTENode::Deserialize(FieldReader &reader) {
	auto result = make_unique<RecursiveCTENode>();
	result->ctename = reader.ReadRequired<string>();
	result->union_all = reader.ReadRequired<bool>();
	result->left = reader.ReadRequiredSerializable<QueryNode>();
	result->right = reader.ReadRequiredSerializable<QueryNode>();
	result->aliases = reader.ReadRequiredList<string>();
	return move(result);
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
			result += " AS " + KeywordHelper::WriteOptionallyQuoted(select_list[i]->alias);
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
		result += " (" + SampleMethodToString(sample->method);
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
	auto other = (SelectNode *)other_p;

	// SELECT
	if (!ExpressionUtil::ListEquals(select_list, other->select_list)) {
		return false;
	}
	// FROM
	if (from_table) {
		// we have a FROM clause, compare to the other one
		if (!from_table->Equals(other->from_table.get())) {
			return false;
		}
	} else if (other->from_table) {
		// we don't have a FROM clause, if the other statement has one they are
		// not equal
		return false;
	}
	// WHERE
	if (!BaseExpression::Equals(where_clause.get(), other->where_clause.get())) {
		return false;
	}
	// GROUP BY
	if (!ExpressionUtil::ListEquals(groups.group_expressions, other->groups.group_expressions)) {
		return false;
	}
	if (groups.grouping_sets != other->groups.grouping_sets) {
		return false;
	}
	if (!SampleOptions::Equals(sample.get(), other->sample.get())) {
		return false;
	}
	// HAVING
	if (!BaseExpression::Equals(having.get(), other->having.get())) {
		return false;
	}
	// QUALIFY
	if (!BaseExpression::Equals(qualify.get(), other->qualify.get())) {
		return false;
	}
	return true;
}

unique_ptr<QueryNode> SelectNode::Copy() const {
	auto result = make_unique<SelectNode>();
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
	return move(result);
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

unique_ptr<QueryNode> SelectNode::Deserialize(FieldReader &reader) {
	auto result = make_unique<SelectNode>();
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
	return move(result);
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
	auto other = (SetOperationNode *)other_p;
	if (setop_type != other->setop_type) {
		return false;
	}
	if (!left->Equals(other->left.get())) {
		return false;
	}
	if (!right->Equals(other->right.get())) {
		return false;
	}
	return true;
}

unique_ptr<QueryNode> SetOperationNode::Copy() const {
	auto result = make_unique<SetOperationNode>();
	result->setop_type = setop_type;
	result->left = left->Copy();
	result->right = right->Copy();
	this->CopyProperties(*result);
	return move(result);
}

void SetOperationNode::Serialize(FieldWriter &writer) const {
	writer.WriteField<SetOperationType>(setop_type);
	writer.WriteSerializable(*left);
	writer.WriteSerializable(*right);
}

unique_ptr<QueryNode> SetOperationNode::Deserialize(FieldReader &reader) {
	auto result = make_unique<SetOperationNode>();
	result->setop_type = reader.ReadRequired<SetOperationType>();
	result->left = reader.ReadRequiredSerializable<QueryNode>();
	result->right = reader.ReadRequiredSerializable<QueryNode>();
	return move(result);
}

} // namespace duckdb








namespace duckdb {

CommonTableExpressionMap::CommonTableExpressionMap() {
}

CommonTableExpressionMap CommonTableExpressionMap::Copy() const {
	CommonTableExpressionMap res;
	for (auto &kv : this->map) {
		auto kv_info = make_unique<CommonTableExpressionInfo>();
		for (auto &al : kv.second->aliases) {
			kv_info->aliases.push_back(al);
		}
		kv_info->query = unique_ptr_cast<SQLStatement, SelectStatement>(kv.second->query->Copy());
		res.map[kv.first] = move(kv_info);
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
		auto kv_info = make_unique<CommonTableExpressionInfo>();
		for (auto &al : kv.second->aliases) {
			kv_info->aliases.push_back(al);
		}
		kv_info->query = unique_ptr_cast<SQLStatement, SelectStatement>(kv.second->query->Copy());
		other.cte_map.map[kv.first] = move(kv_info);
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

unique_ptr<QueryNode> QueryNode::Deserialize(Deserializer &main_source) {
	FieldReader reader(main_source);

	auto type = reader.ReadRequired<QueryNodeType>();
	auto modifiers = reader.ReadRequiredSerializableList<ResultModifier>();
	// cte_map
	auto cte_count = reader.ReadRequired<uint32_t>();
	auto &source = reader.GetSource();
	unordered_map<string, unique_ptr<CommonTableExpressionInfo>> new_map;
	for (idx_t i = 0; i < cte_count; i++) {
		auto name = source.Read<string>();
		auto info = make_unique<CommonTableExpressionInfo>();
		source.ReadStringVector(info->aliases);
		info->query = SelectStatement::Deserialize(source);
		new_map[name] = move(info);
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
	result->modifiers = move(modifiers);
	result->cte_map.map = move(new_map);
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
	modifiers.push_back(make_unique<DistinctModifier>());
}

} // namespace duckdb
