// See https://raw.githubusercontent.com/duckdb/duckdb/master/LICENSE for licensing information

#include "duckdb.hpp"
#include "duckdb-internal.hpp"
#ifndef DUCKDB_AMALGAMATION
#error header mismatch
#endif



































#include <algorithm>

namespace duckdb {

Catalog::Catalog(AttachedDatabase &db) : db(db) {
}

Catalog::~Catalog() {
}

DatabaseInstance &Catalog::GetDatabase() {
	return db.GetDatabase();
}

AttachedDatabase &Catalog::GetAttached() {
	return db;
}

const string &Catalog::GetName() {
	return GetAttached().GetName();
}

idx_t Catalog::GetOid() {
	return GetAttached().oid;
}

Catalog &Catalog::GetSystemCatalog(ClientContext &context) {
	return Catalog::GetSystemCatalog(*context.db);
}

optional_ptr<Catalog> Catalog::GetCatalogEntry(ClientContext &context, const string &catalog_name) {
	auto &db_manager = DatabaseManager::Get(context);
	if (catalog_name == TEMP_CATALOG) {
		return &ClientData::Get(context).temporary_objects->GetCatalog();
	}
	if (catalog_name == SYSTEM_CATALOG) {
		return &GetSystemCatalog(context);
	}
	auto entry = db_manager.GetDatabase(
	    context, IsInvalidCatalog(catalog_name) ? DatabaseManager::GetDefaultDatabase(context) : catalog_name);
	if (!entry) {
		return nullptr;
	}
	return &entry->GetCatalog();
}

Catalog &Catalog::GetCatalog(ClientContext &context, const string &catalog_name) {
	auto catalog = Catalog::GetCatalogEntry(context, catalog_name);
	if (!catalog) {
		throw BinderException("Catalog \"%s\" does not exist!", catalog_name);
	}
	return *catalog;
}

//===--------------------------------------------------------------------===//
// Schema
//===--------------------------------------------------------------------===//
optional_ptr<CatalogEntry> Catalog::CreateSchema(ClientContext &context, CreateSchemaInfo &info) {
	return CreateSchema(GetCatalogTransaction(context), info);
}

CatalogTransaction Catalog::GetCatalogTransaction(ClientContext &context) {
	return CatalogTransaction(*this, context);
}

//===--------------------------------------------------------------------===//
// Table
//===--------------------------------------------------------------------===//
optional_ptr<CatalogEntry> Catalog::CreateTable(ClientContext &context, BoundCreateTableInfo &info) {
	return CreateTable(GetCatalogTransaction(context), info);
}

optional_ptr<CatalogEntry> Catalog::CreateTable(ClientContext &context, unique_ptr<CreateTableInfo> info) {
	auto binder = Binder::CreateBinder(context);
	auto bound_info = binder->BindCreateTableInfo(std::move(info));
	return CreateTable(context, *bound_info);
}

optional_ptr<CatalogEntry> Catalog::CreateTable(CatalogTransaction transaction, SchemaCatalogEntry &schema,
                                                BoundCreateTableInfo &info) {
	return schema.CreateTable(transaction, info);
}

optional_ptr<CatalogEntry> Catalog::CreateTable(CatalogTransaction transaction, BoundCreateTableInfo &info) {
	auto &schema = GetSchema(transaction, info.base->schema);
	return CreateTable(transaction, schema, info);
}

//===--------------------------------------------------------------------===//
// View
//===--------------------------------------------------------------------===//
optional_ptr<CatalogEntry> Catalog::CreateView(CatalogTransaction transaction, CreateViewInfo &info) {
	auto &schema = GetSchema(transaction, info.schema);
	return CreateView(transaction, schema, info);
}

optional_ptr<CatalogEntry> Catalog::CreateView(ClientContext &context, CreateViewInfo &info) {
	return CreateView(GetCatalogTransaction(context), info);
}

optional_ptr<CatalogEntry> Catalog::CreateView(CatalogTransaction transaction, SchemaCatalogEntry &schema,
                                               CreateViewInfo &info) {
	return schema.CreateView(transaction, info);
}

//===--------------------------------------------------------------------===//
// Sequence
//===--------------------------------------------------------------------===//
optional_ptr<CatalogEntry> Catalog::CreateSequence(CatalogTransaction transaction, CreateSequenceInfo &info) {
	auto &schema = GetSchema(transaction, info.schema);
	return CreateSequence(transaction, schema, info);
}

optional_ptr<CatalogEntry> Catalog::CreateSequence(ClientContext &context, CreateSequenceInfo &info) {
	return CreateSequence(GetCatalogTransaction(context), info);
}

optional_ptr<CatalogEntry> Catalog::CreateSequence(CatalogTransaction transaction, SchemaCatalogEntry &schema,
                                                   CreateSequenceInfo &info) {
	return schema.CreateSequence(transaction, info);
}

//===--------------------------------------------------------------------===//
// Type
//===--------------------------------------------------------------------===//
optional_ptr<CatalogEntry> Catalog::CreateType(CatalogTransaction transaction, CreateTypeInfo &info) {
	auto &schema = GetSchema(transaction, info.schema);
	return CreateType(transaction, schema, info);
}

optional_ptr<CatalogEntry> Catalog::CreateType(ClientContext &context, CreateTypeInfo &info) {
	return CreateType(GetCatalogTransaction(context), info);
}

optional_ptr<CatalogEntry> Catalog::CreateType(CatalogTransaction transaction, SchemaCatalogEntry &schema,
                                               CreateTypeInfo &info) {
	return schema.CreateType(transaction, info);
}

//===--------------------------------------------------------------------===//
// Table Function
//===--------------------------------------------------------------------===//
optional_ptr<CatalogEntry> Catalog::CreateTableFunction(CatalogTransaction transaction, CreateTableFunctionInfo &info) {
	auto &schema = GetSchema(transaction, info.schema);
	return CreateTableFunction(transaction, schema, info);
}

optional_ptr<CatalogEntry> Catalog::CreateTableFunction(ClientContext &context, CreateTableFunctionInfo &info) {
	return CreateTableFunction(GetCatalogTransaction(context), info);
}

optional_ptr<CatalogEntry> Catalog::CreateTableFunction(CatalogTransaction transaction, SchemaCatalogEntry &schema,
                                                        CreateTableFunctionInfo &info) {
	return schema.CreateTableFunction(transaction, info);
}

optional_ptr<CatalogEntry> Catalog::CreateTableFunction(ClientContext &context,
                                                        optional_ptr<CreateTableFunctionInfo> info) {
	return CreateTableFunction(context, *info);
}

//===--------------------------------------------------------------------===//
// Copy Function
//===--------------------------------------------------------------------===//
optional_ptr<CatalogEntry> Catalog::CreateCopyFunction(CatalogTransaction transaction, CreateCopyFunctionInfo &info) {
	auto &schema = GetSchema(transaction, info.schema);
	return CreateCopyFunction(transaction, schema, info);
}

optional_ptr<CatalogEntry> Catalog::CreateCopyFunction(ClientContext &context, CreateCopyFunctionInfo &info) {
	return CreateCopyFunction(GetCatalogTransaction(context), info);
}

optional_ptr<CatalogEntry> Catalog::CreateCopyFunction(CatalogTransaction transaction, SchemaCatalogEntry &schema,
                                                       CreateCopyFunctionInfo &info) {
	return schema.CreateCopyFunction(transaction, info);
}

//===--------------------------------------------------------------------===//
// Pragma Function
//===--------------------------------------------------------------------===//
optional_ptr<CatalogEntry> Catalog::CreatePragmaFunction(CatalogTransaction transaction,
                                                         CreatePragmaFunctionInfo &info) {
	auto &schema = GetSchema(transaction, info.schema);
	return CreatePragmaFunction(transaction, schema, info);
}

optional_ptr<CatalogEntry> Catalog::CreatePragmaFunction(ClientContext &context, CreatePragmaFunctionInfo &info) {
	return CreatePragmaFunction(GetCatalogTransaction(context), info);
}

optional_ptr<CatalogEntry> Catalog::CreatePragmaFunction(CatalogTransaction transaction, SchemaCatalogEntry &schema,
                                                         CreatePragmaFunctionInfo &info) {
	return schema.CreatePragmaFunction(transaction, info);
}

//===--------------------------------------------------------------------===//
// Function
//===--------------------------------------------------------------------===//
optional_ptr<CatalogEntry> Catalog::CreateFunction(CatalogTransaction transaction, CreateFunctionInfo &info) {
	auto &schema = GetSchema(transaction, info.schema);
	return CreateFunction(transaction, schema, info);
}

optional_ptr<CatalogEntry> Catalog::CreateFunction(ClientContext &context, CreateFunctionInfo &info) {
	return CreateFunction(GetCatalogTransaction(context), info);
}

optional_ptr<CatalogEntry> Catalog::CreateFunction(CatalogTransaction transaction, SchemaCatalogEntry &schema,
                                                   CreateFunctionInfo &info) {
	return schema.CreateFunction(transaction, info);
}

optional_ptr<CatalogEntry> Catalog::AddFunction(ClientContext &context, CreateFunctionInfo &info) {
	info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	return CreateFunction(context, info);
}

//===--------------------------------------------------------------------===//
// Collation
//===--------------------------------------------------------------------===//
optional_ptr<CatalogEntry> Catalog::CreateCollation(CatalogTransaction transaction, CreateCollationInfo &info) {
	auto &schema = GetSchema(transaction, info.schema);
	return CreateCollation(transaction, schema, info);
}

optional_ptr<CatalogEntry> Catalog::CreateCollation(ClientContext &context, CreateCollationInfo &info) {
	return CreateCollation(GetCatalogTransaction(context), info);
}

optional_ptr<CatalogEntry> Catalog::CreateCollation(CatalogTransaction transaction, SchemaCatalogEntry &schema,
                                                    CreateCollationInfo &info) {
	return schema.CreateCollation(transaction, info);
}

//===--------------------------------------------------------------------===//
// Index
//===--------------------------------------------------------------------===//
optional_ptr<CatalogEntry> Catalog::CreateIndex(CatalogTransaction transaction, CreateIndexInfo &info) {
	auto &context = transaction.GetContext();
	return CreateIndex(context, info);
}

optional_ptr<CatalogEntry> Catalog::CreateIndex(ClientContext &context, CreateIndexInfo &info) {
	auto &schema = GetSchema(context, info.schema);
	auto &table = GetEntry<TableCatalogEntry>(context, schema.name, info.table->table_name);
	return schema.CreateIndex(context, info, table);
}

//===--------------------------------------------------------------------===//
// Lookup Structures
//===--------------------------------------------------------------------===//
struct CatalogLookup {
	CatalogLookup(Catalog &catalog, string schema_p) : catalog(catalog), schema(std::move(schema_p)) {
	}

	Catalog &catalog;
	string schema;
};

//! Return value of Catalog::LookupEntry
struct CatalogEntryLookup {
	optional_ptr<SchemaCatalogEntry> schema;
	optional_ptr<CatalogEntry> entry;

	DUCKDB_API bool Found() const {
		return entry;
	}
};

//===--------------------------------------------------------------------===//
// Generic
//===--------------------------------------------------------------------===//
void Catalog::DropEntry(ClientContext &context, DropInfo &info) {
	ModifyCatalog();
	if (info.type == CatalogType::SCHEMA_ENTRY) {
		// DROP SCHEMA
		DropSchema(context, info);
		return;
	}

	auto lookup = LookupEntry(context, info.type, info.schema, info.name, info.if_not_found);
	if (!lookup.Found()) {
		return;
	}

	lookup.schema->DropEntry(context, info);
}

SchemaCatalogEntry &Catalog::GetSchema(ClientContext &context, const string &name, QueryErrorContext error_context) {
	return *Catalog::GetSchema(context, name, OnEntryNotFound::THROW_EXCEPTION, error_context);
}

optional_ptr<SchemaCatalogEntry> Catalog::GetSchema(ClientContext &context, const string &schema_name,
                                                    OnEntryNotFound if_not_found, QueryErrorContext error_context) {
	return GetSchema(GetCatalogTransaction(context), schema_name, if_not_found, error_context);
}

SchemaCatalogEntry &Catalog::GetSchema(ClientContext &context, const string &catalog_name, const string &schema_name,
                                       QueryErrorContext error_context) {
	return *Catalog::GetSchema(context, catalog_name, schema_name, OnEntryNotFound::THROW_EXCEPTION, error_context);
}

SchemaCatalogEntry &Catalog::GetSchema(CatalogTransaction transaction, const string &name,
                                       QueryErrorContext error_context) {
	return *GetSchema(transaction, name, OnEntryNotFound::THROW_EXCEPTION, error_context);
}

//===--------------------------------------------------------------------===//
// Lookup
//===--------------------------------------------------------------------===//
SimilarCatalogEntry Catalog::SimilarEntryInSchemas(ClientContext &context, const string &entry_name, CatalogType type,
                                                   const reference_set_t<SchemaCatalogEntry> &schemas) {
	SimilarCatalogEntry result;
	for (auto schema_ref : schemas) {
		auto &schema = schema_ref.get();
		auto transaction = schema.catalog.GetCatalogTransaction(context);
		auto entry = schema.GetSimilarEntry(transaction, type, entry_name);
		if (!entry.Found()) {
			// no similar entry found
			continue;
		}
		if (!result.Found() || result.distance > entry.distance) {
			result = entry;
			result.schema = &schema;
		}
	}
	return result;
}

string FindExtensionGeneric(const string &name, const ExtensionEntry entries[], idx_t size) {
	auto lcase = StringUtil::Lower(name);
	auto it = std::lower_bound(entries, entries + size, lcase,
	                           [](const ExtensionEntry &element, const string &value) { return element.name < value; });
	if (it != entries + size && it->name == lcase) {
		return it->extension;
	}
	return "";
}

string FindExtensionForFunction(const string &name) {
	idx_t size = sizeof(EXTENSION_FUNCTIONS) / sizeof(ExtensionEntry);
	return FindExtensionGeneric(name, EXTENSION_FUNCTIONS, size);
}

string FindExtensionForSetting(const string &name) {
	idx_t size = sizeof(EXTENSION_SETTINGS) / sizeof(ExtensionEntry);
	return FindExtensionGeneric(name, EXTENSION_SETTINGS, size);
}

vector<CatalogSearchEntry> GetCatalogEntries(ClientContext &context, const string &catalog, const string &schema) {
	vector<CatalogSearchEntry> entries;
	auto &search_path = *context.client_data->catalog_search_path;
	if (IsInvalidCatalog(catalog) && IsInvalidSchema(schema)) {
		// no catalog or schema provided - scan the entire search path
		entries = search_path.Get();
	} else if (IsInvalidCatalog(catalog)) {
		auto catalogs = search_path.GetCatalogsForSchema(schema);
		for (auto &catalog_name : catalogs) {
			entries.emplace_back(catalog_name, schema);
		}
		if (entries.empty()) {
			entries.emplace_back(DatabaseManager::GetDefaultDatabase(context), schema);
		}
	} else if (IsInvalidSchema(schema)) {
		auto schemas = search_path.GetSchemasForCatalog(catalog);
		for (auto &schema_name : schemas) {
			entries.emplace_back(catalog, schema_name);
		}
		if (entries.empty()) {
			entries.emplace_back(catalog, DEFAULT_SCHEMA);
		}
	} else {
		// specific catalog and schema provided
		entries.emplace_back(catalog, schema);
	}
	return entries;
}

void FindMinimalQualification(ClientContext &context, const string &catalog_name, const string &schema_name,
                              bool &qualify_database, bool &qualify_schema) {
	// check if we can we qualify ONLY the schema
	bool found = false;
	auto entries = GetCatalogEntries(context, INVALID_CATALOG, schema_name);
	for (auto &entry : entries) {
		if (entry.catalog == catalog_name && entry.schema == schema_name) {
			found = true;
			break;
		}
	}
	if (found) {
		qualify_database = false;
		qualify_schema = true;
		return;
	}
	// check if we can qualify ONLY the catalog
	found = false;
	entries = GetCatalogEntries(context, catalog_name, INVALID_SCHEMA);
	for (auto &entry : entries) {
		if (entry.catalog == catalog_name && entry.schema == schema_name) {
			found = true;
			break;
		}
	}
	if (found) {
		qualify_database = true;
		qualify_schema = false;
		return;
	}
	// need to qualify both catalog and schema
	qualify_database = true;
	qualify_schema = true;
}

CatalogException Catalog::UnrecognizedConfigurationError(ClientContext &context, const string &name) {
	// check if the setting exists in any extensions
	auto extension_name = FindExtensionForSetting(name);
	if (!extension_name.empty()) {
		return CatalogException(
		    "Setting with name \"%s\" is not in the catalog, but it exists in the %s extension.\n\nTo "
		    "install and load the extension, run:\nINSTALL %s;\nLOAD %s;",
		    name, extension_name, extension_name, extension_name);
	}
	// the setting is not in an extension
	// get a list of all options
	vector<string> potential_names = DBConfig::GetOptionNames();
	for (auto &entry : DBConfig::GetConfig(context).extension_parameters) {
		potential_names.push_back(entry.first);
	}

	throw CatalogException("unrecognized configuration parameter \"%s\"\n%s", name,
	                       StringUtil::CandidatesErrorMessage(potential_names, name, "Did you mean"));
}

CatalogException Catalog::CreateMissingEntryException(ClientContext &context, const string &entry_name,
                                                      CatalogType type,
                                                      const reference_set_t<SchemaCatalogEntry> &schemas,
                                                      QueryErrorContext error_context) {
	auto entry = SimilarEntryInSchemas(context, entry_name, type, schemas);

	reference_set_t<SchemaCatalogEntry> unseen_schemas;
	auto &db_manager = DatabaseManager::Get(context);
	auto databases = db_manager.GetDatabases(context);
	for (auto database : databases) {
		auto &catalog = database.get().GetCatalog();
		auto current_schemas = catalog.GetAllSchemas(context);
		for (auto &current_schema : current_schemas) {
			unseen_schemas.insert(current_schema.get());
		}
	}
	// check if the entry exists in any extension
	if (type == CatalogType::TABLE_FUNCTION_ENTRY || type == CatalogType::SCALAR_FUNCTION_ENTRY ||
	    type == CatalogType::AGGREGATE_FUNCTION_ENTRY) {
		auto extension_name = FindExtensionForFunction(entry_name);
		if (!extension_name.empty()) {
			return CatalogException(
			    "Function with name \"%s\" is not in the catalog, but it exists in the %s extension.\n\nTo "
			    "install and load the extension, run:\nINSTALL %s;\nLOAD %s;",
			    entry_name, extension_name, extension_name, extension_name);
		}
	}
	auto unseen_entry = SimilarEntryInSchemas(context, entry_name, type, unseen_schemas);
	string did_you_mean;
	if (unseen_entry.Found() && unseen_entry.distance < entry.distance) {
		// the closest matching entry requires qualification as it is not in the default search path
		// check how to minimally qualify this entry
		auto catalog_name = unseen_entry.schema->catalog.GetName();
		auto schema_name = unseen_entry.schema->name;
		bool qualify_database;
		bool qualify_schema;
		FindMinimalQualification(context, catalog_name, schema_name, qualify_database, qualify_schema);
		did_you_mean = "\nDid you mean \"" + unseen_entry.GetQualifiedName(qualify_database, qualify_schema) + "\"?";
	} else if (entry.Found()) {
		did_you_mean = "\nDid you mean \"" + entry.name + "\"?";
	}

	return CatalogException(error_context.FormatError("%s with name %s does not exist!%s", CatalogTypeToString(type),
	                                                  entry_name, did_you_mean));
}

CatalogEntryLookup Catalog::LookupEntryInternal(CatalogTransaction transaction, CatalogType type, const string &schema,
                                                const string &name) {
	auto schema_entry = GetSchema(transaction, schema, OnEntryNotFound::RETURN_NULL);
	if (!schema_entry) {
		return {nullptr, nullptr};
	}
	auto entry = schema_entry->GetEntry(transaction, type, name);
	if (!entry) {
		return {schema_entry, nullptr};
	}
	return {schema_entry, entry};
}

CatalogEntryLookup Catalog::LookupEntry(ClientContext &context, CatalogType type, const string &schema,
                                        const string &name, OnEntryNotFound if_not_found,
                                        QueryErrorContext error_context) {
	reference_set_t<SchemaCatalogEntry> schemas;
	if (IsInvalidSchema(schema)) {
		// try all schemas for this catalog
		auto catalog_name = GetName();
		if (catalog_name == DatabaseManager::GetDefaultDatabase(context)) {
			catalog_name = INVALID_CATALOG;
		}
		auto entries = GetCatalogEntries(context, GetName(), INVALID_SCHEMA);
		for (auto &entry : entries) {
			auto &candidate_schema = entry.schema;
			auto transaction = GetCatalogTransaction(context);
			auto result = LookupEntryInternal(transaction, type, candidate_schema, name);
			if (result.Found()) {
				return result;
			}
			if (result.schema) {
				schemas.insert(*result.schema);
			}
		}
	} else {
		auto transaction = GetCatalogTransaction(context);
		auto result = LookupEntryInternal(transaction, type, schema, name);
		if (result.Found()) {
			return result;
		}
		if (result.schema) {
			schemas.insert(*result.schema);
		}
	}
	if (if_not_found == OnEntryNotFound::RETURN_NULL) {
		return {nullptr, nullptr};
	}
	throw CreateMissingEntryException(context, name, type, schemas, error_context);
}

CatalogEntryLookup Catalog::LookupEntry(ClientContext &context, vector<CatalogLookup> &lookups, CatalogType type,
                                        const string &name, OnEntryNotFound if_not_found,
                                        QueryErrorContext error_context) {
	reference_set_t<SchemaCatalogEntry> schemas;
	for (auto &lookup : lookups) {
		auto transaction = lookup.catalog.GetCatalogTransaction(context);
		auto result = lookup.catalog.LookupEntryInternal(transaction, type, lookup.schema, name);
		if (result.Found()) {
			return result;
		}
		if (result.schema) {
			schemas.insert(*result.schema);
		}
	}
	if (if_not_found == OnEntryNotFound::RETURN_NULL) {
		return {nullptr, nullptr};
	}
	throw CreateMissingEntryException(context, name, type, schemas, error_context);
}

CatalogEntry &Catalog::GetEntry(ClientContext &context, const string &schema, const string &name) {
	vector<CatalogType> entry_types {CatalogType::TABLE_ENTRY, CatalogType::SEQUENCE_ENTRY};

	for (auto entry_type : entry_types) {
		auto result = GetEntry(context, entry_type, schema, name, OnEntryNotFound::RETURN_NULL);
		if (result) {
			return *result;
		}
	}

	throw CatalogException("CatalogElement \"%s.%s\" does not exist!", schema, name);
}

optional_ptr<CatalogEntry> Catalog::GetEntry(ClientContext &context, CatalogType type, const string &schema_name,
                                             const string &name, OnEntryNotFound if_not_found,
                                             QueryErrorContext error_context) {
	return LookupEntry(context, type, schema_name, name, if_not_found, error_context).entry.get();
}

CatalogEntry &Catalog::GetEntry(ClientContext &context, CatalogType type, const string &schema, const string &name,
                                QueryErrorContext error_context) {
	return *Catalog::GetEntry(context, type, schema, name, OnEntryNotFound::THROW_EXCEPTION, error_context);
}

optional_ptr<CatalogEntry> Catalog::GetEntry(ClientContext &context, CatalogType type, const string &catalog,
                                             const string &schema, const string &name, OnEntryNotFound if_not_found,
                                             QueryErrorContext error_context) {
	auto entries = GetCatalogEntries(context, catalog, schema);
	vector<CatalogLookup> lookups;
	lookups.reserve(entries.size());
	for (auto &entry : entries) {
		if (if_not_found == OnEntryNotFound::RETURN_NULL) {
			auto catalog_entry = Catalog::GetCatalogEntry(context, entry.catalog);
			if (!catalog_entry) {
				return nullptr;
			}
			lookups.emplace_back(*catalog_entry, entry.schema);
		} else {
			lookups.emplace_back(Catalog::GetCatalog(context, entry.catalog), entry.schema);
		}
	}
	auto result = LookupEntry(context, lookups, type, name, if_not_found, error_context);
	if (!result.Found()) {
		D_ASSERT(if_not_found == OnEntryNotFound::RETURN_NULL);
		return nullptr;
	}
	return result.entry.get();
}

CatalogEntry &Catalog::GetEntry(ClientContext &context, CatalogType type, const string &catalog, const string &schema,
                                const string &name, QueryErrorContext error_context) {
	return *Catalog::GetEntry(context, type, catalog, schema, name, OnEntryNotFound::THROW_EXCEPTION, error_context);
}

optional_ptr<SchemaCatalogEntry> Catalog::GetSchema(ClientContext &context, const string &catalog_name,
                                                    const string &schema_name, OnEntryNotFound if_not_found,
                                                    QueryErrorContext error_context) {
	auto entries = GetCatalogEntries(context, catalog_name, schema_name);
	for (idx_t i = 0; i < entries.size(); i++) {
		auto on_not_found = i + 1 == entries.size() ? if_not_found : OnEntryNotFound::RETURN_NULL;
		auto &catalog = Catalog::GetCatalog(context, entries[i].catalog);
		auto result = catalog.GetSchema(context, schema_name, on_not_found, error_context);
		if (result) {
			return result;
		}
	}
	return nullptr;
}

LogicalType Catalog::GetType(ClientContext &context, const string &schema, const string &name,
                             OnEntryNotFound if_not_found) {
	auto type_entry = GetEntry<TypeCatalogEntry>(context, schema, name, if_not_found);
	if (!type_entry) {
		return LogicalType::INVALID;
	}
	auto result_type = type_entry->user_type;
	EnumType::SetCatalog(result_type, type_entry.get());
	return result_type;
}

LogicalType Catalog::GetType(ClientContext &context, const string &catalog_name, const string &schema,
                             const string &name) {
	auto &type_entry = Catalog::GetEntry<TypeCatalogEntry>(context, catalog_name, schema, name);
	auto result_type = type_entry.user_type;
	EnumType::SetCatalog(result_type, &type_entry);
	return result_type;
}

vector<reference<SchemaCatalogEntry>> Catalog::GetSchemas(ClientContext &context) {
	vector<reference<SchemaCatalogEntry>> schemas;
	ScanSchemas(context, [&](SchemaCatalogEntry &entry) { schemas.push_back(entry); });
	return schemas;
}

bool Catalog::TypeExists(ClientContext &context, const string &catalog_name, const string &schema, const string &name) {
	optional_ptr<CatalogEntry> entry;
	entry = GetEntry(context, CatalogType::TYPE_ENTRY, catalog_name, schema, name, OnEntryNotFound::RETURN_NULL);
	if (!entry) {
		// look in the system catalog
		entry = GetEntry(context, CatalogType::TYPE_ENTRY, SYSTEM_CATALOG, schema, name, OnEntryNotFound::RETURN_NULL);
		if (!entry) {
			return false;
		}
	}
	return true;
}

vector<reference<SchemaCatalogEntry>> Catalog::GetSchemas(ClientContext &context, const string &catalog_name) {
	vector<reference<Catalog>> catalogs;
	if (IsInvalidCatalog(catalog_name)) {
		unordered_set<string> name;

		auto &search_path = *context.client_data->catalog_search_path;
		for (auto &entry : search_path.Get()) {
			if (name.find(entry.catalog) != name.end()) {
				continue;
			}
			name.insert(entry.catalog);
			catalogs.push_back(Catalog::GetCatalog(context, entry.catalog));
		}
	} else {
		catalogs.push_back(Catalog::GetCatalog(context, catalog_name));
	}
	vector<reference<SchemaCatalogEntry>> result;
	for (auto catalog : catalogs) {
		auto schemas = catalog.get().GetSchemas(context);
		result.insert(result.end(), schemas.begin(), schemas.end());
	}
	return result;
}

vector<reference<SchemaCatalogEntry>> Catalog::GetAllSchemas(ClientContext &context) {
	vector<reference<SchemaCatalogEntry>> result;

	auto &db_manager = DatabaseManager::Get(context);
	auto databases = db_manager.GetDatabases(context);
	for (auto database : databases) {
		auto &catalog = database.get().GetCatalog();
		auto new_schemas = catalog.GetSchemas(context);
		result.insert(result.end(), new_schemas.begin(), new_schemas.end());
	}
	sort(result.begin(), result.end(),
	     [&](reference<SchemaCatalogEntry> left_p, reference<SchemaCatalogEntry> right_p) {
		     auto &left = left_p.get();
		     auto &right = right_p.get();
		     if (left.catalog.GetName() < right.catalog.GetName()) {
			     return true;
		     }
		     if (left.catalog.GetName() == right.catalog.GetName()) {
			     return left.name < right.name;
		     }
		     return false;
	     });

	return result;
}

void Catalog::Alter(ClientContext &context, AlterInfo &info) {
	ModifyCatalog();
	auto lookup = LookupEntry(context, info.GetCatalogType(), info.schema, info.name, info.if_not_found);
	if (!lookup.Found()) {
		return;
	}
	return lookup.schema->Alter(context, info);
}

void Catalog::Verify() {
}

//===--------------------------------------------------------------------===//
// Catalog Version
//===--------------------------------------------------------------------===//
idx_t Catalog::GetCatalogVersion() {
	return GetDatabase().GetDatabaseManager().catalog_version;
}

idx_t Catalog::ModifyCatalog() {
	return GetDatabase().GetDatabaseManager().ModifyCatalog();
}

bool Catalog::IsSystemCatalog() const {
	return db.IsSystem();
}

bool Catalog::IsTemporaryCatalog() const {
	return db.IsTemporary();
}

} // namespace duckdb





namespace duckdb {

ColumnDependencyManager::ColumnDependencyManager() {
}

ColumnDependencyManager::~ColumnDependencyManager() {
}

void ColumnDependencyManager::AddGeneratedColumn(const ColumnDefinition &column, const ColumnList &list) {
	D_ASSERT(column.Generated());
	vector<string> referenced_columns;
	column.GetListOfDependencies(referenced_columns);
	vector<LogicalIndex> indices;
	for (auto &col : referenced_columns) {
		if (!list.ColumnExists(col)) {
			throw BinderException("Column \"%s\" referenced by generated column does not exist", col);
		}
		auto &entry = list.GetColumn(col);
		indices.push_back(entry.Logical());
	}
	return AddGeneratedColumn(column.Logical(), indices);
}

void ColumnDependencyManager::AddGeneratedColumn(LogicalIndex index, const vector<LogicalIndex> &indices, bool root) {
	if (indices.empty()) {
		return;
	}
	auto &list = dependents_map[index];
	// Create a link between the dependencies
	for (auto &dep : indices) {
		// Add this column as a dependency of the new column
		list.insert(dep);
		// Add the new column as a dependent of the column
		dependencies_map[dep].insert(index);
		// Inherit the dependencies
		if (HasDependencies(dep)) {
			auto &inherited_deps = dependents_map[dep];
			D_ASSERT(!inherited_deps.empty());
			for (auto &inherited_dep : inherited_deps) {
				list.insert(inherited_dep);
				dependencies_map[inherited_dep].insert(index);
			}
		}
		if (!root) {
			continue;
		}
		direct_dependencies[index].insert(dep);
	}
	if (!HasDependents(index)) {
		return;
	}
	auto &dependents = dependencies_map[index];
	if (dependents.count(index)) {
		throw InvalidInputException("Circular dependency encountered when resolving generated column expressions");
	}
	// Also let the dependents of this generated column inherit the dependencies
	for (auto &dependent : dependents) {
		AddGeneratedColumn(dependent, indices, false);
	}
}

vector<LogicalIndex> ColumnDependencyManager::RemoveColumn(LogicalIndex index, idx_t column_amount) {
	// Always add the initial column
	deleted_columns.insert(index);

	RemoveGeneratedColumn(index);
	RemoveStandardColumn(index);

	// Clean up the internal list
	vector<LogicalIndex> new_indices = CleanupInternals(column_amount);
	D_ASSERT(deleted_columns.empty());
	return new_indices;
}

bool ColumnDependencyManager::IsDependencyOf(LogicalIndex gcol, LogicalIndex col) const {
	auto entry = dependents_map.find(gcol);
	if (entry == dependents_map.end()) {
		return false;
	}
	auto &list = entry->second;
	return list.count(col);
}

bool ColumnDependencyManager::HasDependencies(LogicalIndex index) const {
	auto entry = dependents_map.find(index);
	if (entry == dependents_map.end()) {
		return false;
	}
	return true;
}

const logical_index_set_t &ColumnDependencyManager::GetDependencies(LogicalIndex index) const {
	auto entry = dependents_map.find(index);
	D_ASSERT(entry != dependents_map.end());
	return entry->second;
}

bool ColumnDependencyManager::HasDependents(LogicalIndex index) const {
	auto entry = dependencies_map.find(index);
	if (entry == dependencies_map.end()) {
		return false;
	}
	return true;
}

const logical_index_set_t &ColumnDependencyManager::GetDependents(LogicalIndex index) const {
	auto entry = dependencies_map.find(index);
	D_ASSERT(entry != dependencies_map.end());
	return entry->second;
}

void ColumnDependencyManager::RemoveStandardColumn(LogicalIndex index) {
	if (!HasDependents(index)) {
		return;
	}
	auto dependents = dependencies_map[index];
	for (auto &gcol : dependents) {
		// If index is a direct dependency of gcol, remove it from the list
		if (direct_dependencies.find(gcol) != direct_dependencies.end()) {
			direct_dependencies[gcol].erase(index);
		}
		RemoveGeneratedColumn(gcol);
	}
	// Remove this column from the dependencies map
	dependencies_map.erase(index);
}

void ColumnDependencyManager::RemoveGeneratedColumn(LogicalIndex index) {
	deleted_columns.insert(index);
	if (!HasDependencies(index)) {
		return;
	}
	auto &dependencies = dependents_map[index];
	for (auto &col : dependencies) {
		// Remove this generated column from the list of this column
		auto &col_dependents = dependencies_map[col];
		D_ASSERT(col_dependents.count(index));
		col_dependents.erase(index);
		// If the resulting list is empty, remove the column from the dependencies map altogether
		if (col_dependents.empty()) {
			dependencies_map.erase(col);
		}
	}
	// Remove this column from the dependents_map map
	dependents_map.erase(index);
}

void ColumnDependencyManager::AdjustSingle(LogicalIndex idx, idx_t offset) {
	D_ASSERT(idx.index >= offset);
	LogicalIndex new_idx = LogicalIndex(idx.index - offset);
	// Adjust this index in the dependents of this column
	bool has_dependents = HasDependents(idx);
	bool has_dependencies = HasDependencies(idx);

	if (has_dependents) {
		auto &dependents = GetDependents(idx);
		for (auto &dep : dependents) {
			auto &dep_dependencies = dependents_map[dep];
			dep_dependencies.erase(idx);
			D_ASSERT(!dep_dependencies.count(new_idx));
			dep_dependencies.insert(new_idx);
		}
	}
	if (has_dependencies) {
		auto &dependencies = GetDependencies(idx);
		for (auto &dep : dependencies) {
			auto &dep_dependents = dependencies_map[dep];
			dep_dependents.erase(idx);
			D_ASSERT(!dep_dependents.count(new_idx));
			dep_dependents.insert(new_idx);
		}
	}
	if (has_dependents) {
		D_ASSERT(!dependencies_map.count(new_idx));
		dependencies_map[new_idx] = std::move(dependencies_map[idx]);
		dependencies_map.erase(idx);
	}
	if (has_dependencies) {
		D_ASSERT(!dependents_map.count(new_idx));
		dependents_map[new_idx] = std::move(dependents_map[idx]);
		dependents_map.erase(idx);
	}
}

vector<LogicalIndex> ColumnDependencyManager::CleanupInternals(idx_t column_amount) {
	vector<LogicalIndex> to_adjust;
	D_ASSERT(!deleted_columns.empty());
	// Get the lowest index that was deleted
	vector<LogicalIndex> new_indices(column_amount, LogicalIndex(DConstants::INVALID_INDEX));
	idx_t threshold = deleted_columns.begin()->index;

	idx_t offset = 0;
	for (idx_t i = 0; i < column_amount; i++) {
		auto current_index = LogicalIndex(i);
		auto new_index = LogicalIndex(i - offset);
		new_indices[i] = new_index;
		if (deleted_columns.count(current_index)) {
			offset++;
			continue;
		}
		if (i > threshold && (HasDependencies(current_index) || HasDependents(current_index))) {
			to_adjust.push_back(current_index);
		}
	}

	// Adjust all indices inside the dependency managers internal mappings
	for (auto &col : to_adjust) {
		auto offset = col.index - new_indices[col.index].index;
		AdjustSingle(col, offset);
	}
	deleted_columns.clear();
	return new_indices;
}

stack<LogicalIndex> ColumnDependencyManager::GetBindOrder(const ColumnList &columns) {
	stack<LogicalIndex> bind_order;
	queue<LogicalIndex> to_visit;
	logical_index_set_t visited;

	for (auto &entry : direct_dependencies) {
		auto dependent = entry.first;
		//! Skip the dependents that are also dependencies
		if (dependencies_map.find(dependent) != dependencies_map.end()) {
			continue;
		}
		bind_order.push(dependent);
		visited.insert(dependent);
		for (auto &dependency : direct_dependencies[dependent]) {
			to_visit.push(dependency);
		}
	}

	while (!to_visit.empty()) {
		auto column = to_visit.front();
		to_visit.pop();

		//! If this column does not have dependencies, the queue stops getting filled
		if (direct_dependencies.find(column) == direct_dependencies.end()) {
			continue;
		}
		bind_order.push(column);
		visited.insert(column);

		for (auto &dependency : direct_dependencies[column]) {
			to_visit.push(dependency);
		}
	}

	// Add generated columns that have no dependencies, but still might need to have their type resolved
	for (auto &col : columns.Logical()) {
		// Not a generated column
		if (!col.Generated()) {
			continue;
		}
		// Already added to the bind_order stack
		if (visited.count(col.Logical())) {
			continue;
		}
		bind_order.push(col.Logical());
	}

	return bind_order;
}

} // namespace duckdb



namespace duckdb {

CopyFunctionCatalogEntry::CopyFunctionCatalogEntry(Catalog &catalog, SchemaCatalogEntry &schema,
                                                   CreateCopyFunctionInfo &info)
    : StandardEntry(CatalogType::COPY_FUNCTION_ENTRY, schema, catalog, info.name), function(info.function) {
}

} // namespace duckdb




namespace duckdb {

DuckIndexEntry::DuckIndexEntry(Catalog &catalog, SchemaCatalogEntry &schema, CreateIndexInfo &info)
    : IndexCatalogEntry(catalog, schema, info) {
}

DuckIndexEntry::~DuckIndexEntry() {
	// remove the associated index from the info
	if (!info || !index) {
		return;
	}
	info->indexes.RemoveIndex(*index);
}

string DuckIndexEntry::GetSchemaName() const {
	return info->schema;
}

string DuckIndexEntry::GetTableName() const {
	return info->table;
}

} // namespace duckdb




































namespace duckdb {

void FindForeignKeyInformation(CatalogEntry &entry, AlterForeignKeyType alter_fk_type,
                               vector<unique_ptr<AlterForeignKeyInfo>> &fk_arrays) {
	if (entry.type != CatalogType::TABLE_ENTRY) {
		return;
	}
	auto &table_entry = entry.Cast<TableCatalogEntry>();
	auto &constraints = table_entry.GetConstraints();
	for (idx_t i = 0; i < constraints.size(); i++) {
		auto &cond = constraints[i];
		if (cond->type != ConstraintType::FOREIGN_KEY) {
			continue;
		}
		auto &fk = cond->Cast<ForeignKeyConstraint>();
		if (fk.info.type == ForeignKeyType::FK_TYPE_FOREIGN_KEY_TABLE) {
			AlterEntryData alter_data(entry.ParentCatalog().GetName(), fk.info.schema, fk.info.table,
			                          OnEntryNotFound::THROW_EXCEPTION);
			fk_arrays.push_back(make_uniq<AlterForeignKeyInfo>(std::move(alter_data), entry.name, fk.pk_columns,
			                                                   fk.fk_columns, fk.info.pk_keys, fk.info.fk_keys,
			                                                   alter_fk_type));
		} else if (fk.info.type == ForeignKeyType::FK_TYPE_PRIMARY_KEY_TABLE &&
		           alter_fk_type == AlterForeignKeyType::AFT_DELETE) {
			throw CatalogException("Could not drop the table because this table is main key table of the table \"%s\"",
			                       fk.info.table);
		}
	}
}

DuckSchemaEntry::DuckSchemaEntry(Catalog &catalog, string name_p, bool is_internal)
    : SchemaCatalogEntry(catalog, std::move(name_p), is_internal),
      tables(catalog, make_uniq<DefaultViewGenerator>(catalog, *this)), indexes(catalog), table_functions(catalog),
      copy_functions(catalog), pragma_functions(catalog),
      functions(catalog, make_uniq<DefaultFunctionGenerator>(catalog, *this)), sequences(catalog), collations(catalog),
      types(catalog, make_uniq<DefaultTypeGenerator>(catalog, *this)) {
}

optional_ptr<CatalogEntry> DuckSchemaEntry::AddEntryInternal(CatalogTransaction transaction,
                                                             unique_ptr<StandardEntry> entry,
                                                             OnCreateConflict on_conflict,
                                                             DependencyList dependencies) {
	auto entry_name = entry->name;
	auto entry_type = entry->type;
	auto result = entry.get();

	// first find the set for this entry
	auto &set = GetCatalogSet(entry_type);
	dependencies.AddDependency(*this);
	if (on_conflict == OnCreateConflict::REPLACE_ON_CONFLICT) {
		// CREATE OR REPLACE: first try to drop the entry
		auto old_entry = set.GetEntry(transaction, entry_name);
		if (old_entry) {
			if (old_entry->type != entry_type) {
				throw CatalogException("Existing object %s is of type %s, trying to replace with type %s", entry_name,
				                       CatalogTypeToString(old_entry->type), CatalogTypeToString(entry_type));
			}
			(void)set.DropEntry(transaction, entry_name, false, entry->internal);
		}
	}
	// now try to add the entry
	if (!set.CreateEntry(transaction, entry_name, std::move(entry), dependencies)) {
		// entry already exists!
		if (on_conflict == OnCreateConflict::ERROR_ON_CONFLICT) {
			throw CatalogException("%s with name \"%s\" already exists!", CatalogTypeToString(entry_type), entry_name);
		} else {
			return nullptr;
		}
	}
	return result;
}

optional_ptr<CatalogEntry> DuckSchemaEntry::CreateTable(CatalogTransaction transaction, BoundCreateTableInfo &info) {
	auto table = make_uniq<DuckTableEntry>(catalog, *this, info);
	auto &storage = table->GetStorage();
	storage.info->cardinality = storage.GetTotalRows();

	auto entry = AddEntryInternal(transaction, std::move(table), info.Base().on_conflict, info.dependencies);
	if (!entry) {
		return nullptr;
	}

	// add a foreign key constraint in main key table if there is a foreign key constraint
	vector<unique_ptr<AlterForeignKeyInfo>> fk_arrays;
	FindForeignKeyInformation(*entry, AlterForeignKeyType::AFT_ADD, fk_arrays);
	for (idx_t i = 0; i < fk_arrays.size(); i++) {
		// alter primary key table
		auto &fk_info = *fk_arrays[i];
		catalog.Alter(transaction.GetContext(), fk_info);

		// make a dependency between this table and referenced table
		auto &set = GetCatalogSet(CatalogType::TABLE_ENTRY);
		info.dependencies.AddDependency(*set.GetEntry(transaction, fk_info.name));
	}
	return entry;
}

optional_ptr<CatalogEntry> DuckSchemaEntry::CreateFunction(CatalogTransaction transaction, CreateFunctionInfo &info) {
	if (info.on_conflict == OnCreateConflict::ALTER_ON_CONFLICT) {
		// check if the original entry exists
		auto &catalog_set = GetCatalogSet(info.type);
		auto current_entry = catalog_set.GetEntry(transaction, info.name);
		if (current_entry) {
			// the current entry exists - alter it instead
			auto alter_info = info.GetAlterInfo();
			Alter(transaction.GetContext(), *alter_info);
			return nullptr;
		}
	}
	unique_ptr<StandardEntry> function;
	switch (info.type) {
	case CatalogType::SCALAR_FUNCTION_ENTRY:
		function = make_uniq_base<StandardEntry, ScalarFunctionCatalogEntry>(catalog, *this,
		                                                                     info.Cast<CreateScalarFunctionInfo>());
		break;
	case CatalogType::TABLE_FUNCTION_ENTRY:
		function = make_uniq_base<StandardEntry, TableFunctionCatalogEntry>(catalog, *this,
		                                                                    info.Cast<CreateTableFunctionInfo>());
		break;
	case CatalogType::MACRO_ENTRY:
		// create a macro function
		function = make_uniq_base<StandardEntry, ScalarMacroCatalogEntry>(catalog, *this, info.Cast<CreateMacroInfo>());
		break;

	case CatalogType::TABLE_MACRO_ENTRY:
		// create a macro table function
		function = make_uniq_base<StandardEntry, TableMacroCatalogEntry>(catalog, *this, info.Cast<CreateMacroInfo>());
		break;
	case CatalogType::AGGREGATE_FUNCTION_ENTRY:
		D_ASSERT(info.type == CatalogType::AGGREGATE_FUNCTION_ENTRY);
		// create an aggregate function
		function = make_uniq_base<StandardEntry, AggregateFunctionCatalogEntry>(
		    catalog, *this, info.Cast<CreateAggregateFunctionInfo>());
		break;
	default:
		throw InternalException("Unknown function type \"%s\"", CatalogTypeToString(info.type));
	}
	function->internal = info.internal;
	return AddEntry(transaction, std::move(function), info.on_conflict);
}

optional_ptr<CatalogEntry> DuckSchemaEntry::AddEntry(CatalogTransaction transaction, unique_ptr<StandardEntry> entry,
                                                     OnCreateConflict on_conflict) {
	DependencyList dependencies;
	return AddEntryInternal(transaction, std::move(entry), on_conflict, dependencies);
}

optional_ptr<CatalogEntry> DuckSchemaEntry::CreateSequence(CatalogTransaction transaction, CreateSequenceInfo &info) {
	auto sequence = make_uniq<SequenceCatalogEntry>(catalog, *this, info);
	return AddEntry(transaction, std::move(sequence), info.on_conflict);
}

optional_ptr<CatalogEntry> DuckSchemaEntry::CreateType(CatalogTransaction transaction, CreateTypeInfo &info) {
	auto type_entry = make_uniq<TypeCatalogEntry>(catalog, *this, info);
	return AddEntry(transaction, std::move(type_entry), info.on_conflict);
}

optional_ptr<CatalogEntry> DuckSchemaEntry::CreateView(CatalogTransaction transaction, CreateViewInfo &info) {
	auto view = make_uniq<ViewCatalogEntry>(catalog, *this, info);
	return AddEntry(transaction, std::move(view), info.on_conflict);
}

optional_ptr<CatalogEntry> DuckSchemaEntry::CreateIndex(ClientContext &context, CreateIndexInfo &info,
                                                        TableCatalogEntry &table) {
	DependencyList dependencies;
	dependencies.AddDependency(table);
	auto index = make_uniq<DuckIndexEntry>(catalog, *this, info);
	return AddEntryInternal(GetCatalogTransaction(context), std::move(index), info.on_conflict, dependencies);
}

optional_ptr<CatalogEntry> DuckSchemaEntry::CreateCollation(CatalogTransaction transaction, CreateCollationInfo &info) {
	auto collation = make_uniq<CollateCatalogEntry>(catalog, *this, info);
	collation->internal = info.internal;
	return AddEntry(transaction, std::move(collation), info.on_conflict);
}

optional_ptr<CatalogEntry> DuckSchemaEntry::CreateTableFunction(CatalogTransaction transaction,
                                                                CreateTableFunctionInfo &info) {
	auto table_function = make_uniq<TableFunctionCatalogEntry>(catalog, *this, info);
	table_function->internal = info.internal;
	return AddEntry(transaction, std::move(table_function), info.on_conflict);
}

optional_ptr<CatalogEntry> DuckSchemaEntry::CreateCopyFunction(CatalogTransaction transaction,
                                                               CreateCopyFunctionInfo &info) {
	auto copy_function = make_uniq<CopyFunctionCatalogEntry>(catalog, *this, info);
	copy_function->internal = info.internal;
	return AddEntry(transaction, std::move(copy_function), info.on_conflict);
}

optional_ptr<CatalogEntry> DuckSchemaEntry::CreatePragmaFunction(CatalogTransaction transaction,
                                                                 CreatePragmaFunctionInfo &info) {
	auto pragma_function = make_uniq<PragmaFunctionCatalogEntry>(catalog, *this, info);
	pragma_function->internal = info.internal;
	return AddEntry(transaction, std::move(pragma_function), info.on_conflict);
}

void DuckSchemaEntry::Alter(ClientContext &context, AlterInfo &info) {
	CatalogType type = info.GetCatalogType();
	auto &set = GetCatalogSet(type);
	auto transaction = GetCatalogTransaction(context);
	if (info.type == AlterType::CHANGE_OWNERSHIP) {
		if (!set.AlterOwnership(transaction, info.Cast<ChangeOwnershipInfo>())) {
			throw CatalogException("Couldn't change ownership!");
		}
	} else {
		string name = info.name;
		if (!set.AlterEntry(transaction, name, info)) {
			throw CatalogException("Entry with name \"%s\" does not exist!", name);
		}
	}
}

void DuckSchemaEntry::Scan(ClientContext &context, CatalogType type,
                           const std::function<void(CatalogEntry &)> &callback) {
	auto &set = GetCatalogSet(type);
	set.Scan(GetCatalogTransaction(context), callback);
}

void DuckSchemaEntry::Scan(CatalogType type, const std::function<void(CatalogEntry &)> &callback) {
	auto &set = GetCatalogSet(type);
	set.Scan(callback);
}

void DuckSchemaEntry::DropEntry(ClientContext &context, DropInfo &info) {
	auto &set = GetCatalogSet(info.type);

	// first find the entry
	auto transaction = GetCatalogTransaction(context);
	auto existing_entry = set.GetEntry(transaction, info.name);
	if (!existing_entry) {
		throw InternalException("Failed to drop entry \"%s\" - entry could not be found", info.name);
	}
	if (existing_entry->type != info.type) {
		throw CatalogException("Existing object %s is of type %s, trying to replace with type %s", info.name,
		                       CatalogTypeToString(existing_entry->type), CatalogTypeToString(info.type));
	}

	// if there is a foreign key constraint, get that information
	vector<unique_ptr<AlterForeignKeyInfo>> fk_arrays;
	FindForeignKeyInformation(*existing_entry, AlterForeignKeyType::AFT_DELETE, fk_arrays);

	if (!set.DropEntry(transaction, info.name, info.cascade, info.allow_drop_internal)) {
		throw InternalException("Could not drop element because of an internal error");
	}

	// remove the foreign key constraint in main key table if main key table's name is valid
	for (idx_t i = 0; i < fk_arrays.size(); i++) {
		// alter primary key table
		catalog.Alter(context, *fk_arrays[i]);
	}
}

optional_ptr<CatalogEntry> DuckSchemaEntry::GetEntry(CatalogTransaction transaction, CatalogType type,
                                                     const string &name) {
	return GetCatalogSet(type).GetEntry(transaction, name);
}

SimilarCatalogEntry DuckSchemaEntry::GetSimilarEntry(CatalogTransaction transaction, CatalogType type,
                                                     const string &name) {
	return GetCatalogSet(type).SimilarEntry(transaction, name);
}

CatalogSet &DuckSchemaEntry::GetCatalogSet(CatalogType type) {
	switch (type) {
	case CatalogType::VIEW_ENTRY:
	case CatalogType::TABLE_ENTRY:
		return tables;
	case CatalogType::INDEX_ENTRY:
		return indexes;
	case CatalogType::TABLE_FUNCTION_ENTRY:
	case CatalogType::TABLE_MACRO_ENTRY:
		return table_functions;
	case CatalogType::COPY_FUNCTION_ENTRY:
		return copy_functions;
	case CatalogType::PRAGMA_FUNCTION_ENTRY:
		return pragma_functions;
	case CatalogType::AGGREGATE_FUNCTION_ENTRY:
	case CatalogType::SCALAR_FUNCTION_ENTRY:
	case CatalogType::MACRO_ENTRY:
		return functions;
	case CatalogType::SEQUENCE_ENTRY:
		return sequences;
	case CatalogType::COLLATION_ENTRY:
		return collations;
	case CatalogType::TYPE_ENTRY:
		return types;
	default:
		throw InternalException("Unsupported catalog type in schema");
	}
}

void DuckSchemaEntry::Verify(Catalog &catalog) {
	InCatalogEntry::Verify(catalog);

	tables.Verify(catalog);
	indexes.Verify(catalog);
	table_functions.Verify(catalog);
	copy_functions.Verify(catalog);
	pragma_functions.Verify(catalog);
	functions.Verify(catalog);
	sequences.Verify(catalog);
	collations.Verify(catalog);
	types.Verify(catalog);
}

} // namespace duckdb



















namespace duckdb {

void AddDataTableIndex(DataTable &storage, const ColumnList &columns, const vector<PhysicalIndex> &keys,
                       IndexConstraintType constraint_type, BlockPointer *index_block = nullptr) {
	// fetch types and create expressions for the index from the columns
	vector<column_t> column_ids;
	vector<unique_ptr<Expression>> unbound_expressions;
	vector<unique_ptr<Expression>> bound_expressions;
	idx_t key_nr = 0;
	column_ids.reserve(keys.size());
	for (auto &physical_key : keys) {
		auto &column = columns.GetColumn(physical_key);
		D_ASSERT(!column.Generated());
		unbound_expressions.push_back(
		    make_uniq<BoundColumnRefExpression>(column.Name(), column.Type(), ColumnBinding(0, column_ids.size())));

		bound_expressions.push_back(make_uniq<BoundReferenceExpression>(column.Type(), key_nr++));
		column_ids.push_back(column.StorageOid());
	}
	unique_ptr<ART> art;
	// create an adaptive radix tree around the expressions
	if (index_block) {
		art = make_uniq<ART>(column_ids, TableIOManager::Get(storage), std::move(unbound_expressions), constraint_type,
		                     storage.db, index_block->block_id, index_block->offset);
	} else {
		art = make_uniq<ART>(column_ids, TableIOManager::Get(storage), std::move(unbound_expressions), constraint_type,
		                     storage.db);
		if (!storage.IsRoot()) {
			throw TransactionException("Transaction conflict: cannot add an index to a table that has been altered!");
		}
	}
	storage.info->indexes.AddIndex(std::move(art));
}

void AddDataTableIndex(DataTable &storage, const ColumnList &columns, vector<LogicalIndex> &keys,
                       IndexConstraintType constraint_type, BlockPointer *index_block = nullptr) {
	vector<PhysicalIndex> new_keys;
	new_keys.reserve(keys.size());
	for (auto &logical_key : keys) {
		new_keys.push_back(columns.LogicalToPhysical(logical_key));
	}
	AddDataTableIndex(storage, columns, new_keys, constraint_type, index_block);
}

DuckTableEntry::DuckTableEntry(Catalog &catalog, SchemaCatalogEntry &schema, BoundCreateTableInfo &info,
                               std::shared_ptr<DataTable> inherited_storage)
    : TableCatalogEntry(catalog, schema, info.Base()), storage(std::move(inherited_storage)),
      bound_constraints(std::move(info.bound_constraints)),
      column_dependency_manager(std::move(info.column_dependency_manager)) {
	if (!storage) {
		// create the physical storage
		vector<ColumnDefinition> storage_columns;
		for (auto &col_def : columns.Physical()) {
			storage_columns.push_back(col_def.Copy());
		}
		storage = make_shared<DataTable>(catalog.GetAttached(), StorageManager::Get(catalog).GetTableIOManager(&info),
		                                 schema.name, name, std::move(storage_columns), std::move(info.data));

		// create the unique indexes for the UNIQUE and PRIMARY KEY and FOREIGN KEY constraints
		idx_t indexes_idx = 0;
		for (idx_t i = 0; i < bound_constraints.size(); i++) {
			auto &constraint = bound_constraints[i];
			if (constraint->type == ConstraintType::UNIQUE) {
				// unique constraint: create a unique index
				auto &unique = constraint->Cast<BoundUniqueConstraint>();
				IndexConstraintType constraint_type = IndexConstraintType::UNIQUE;
				if (unique.is_primary_key) {
					constraint_type = IndexConstraintType::PRIMARY;
				}
				if (info.indexes.empty()) {
					AddDataTableIndex(*storage, columns, unique.keys, constraint_type);
				} else {
					AddDataTableIndex(*storage, columns, unique.keys, constraint_type, &info.indexes[indexes_idx++]);
				}
			} else if (constraint->type == ConstraintType::FOREIGN_KEY) {
				// foreign key constraint: create a foreign key index
				auto &bfk = constraint->Cast<BoundForeignKeyConstraint>();
				if (bfk.info.type == ForeignKeyType::FK_TYPE_FOREIGN_KEY_TABLE ||
				    bfk.info.type == ForeignKeyType::FK_TYPE_SELF_REFERENCE_TABLE) {
					if (info.indexes.empty()) {
						AddDataTableIndex(*storage, columns, bfk.info.fk_keys, IndexConstraintType::FOREIGN);
					} else {
						AddDataTableIndex(*storage, columns, bfk.info.fk_keys, IndexConstraintType::FOREIGN,
						                  &info.indexes[indexes_idx++]);
					}
				}
			}
		}
	}
}

unique_ptr<BaseStatistics> DuckTableEntry::GetStatistics(ClientContext &context, column_t column_id) {
	if (column_id == COLUMN_IDENTIFIER_ROW_ID) {
		return nullptr;
	}
	auto &column = columns.GetColumn(LogicalIndex(column_id));
	if (column.Generated()) {
		return nullptr;
	}
	return storage->GetStatistics(context, column.StorageOid());
}

unique_ptr<CatalogEntry> DuckTableEntry::AlterEntry(ClientContext &context, AlterInfo &info) {
	D_ASSERT(!internal);
	if (info.type != AlterType::ALTER_TABLE) {
		throw CatalogException("Can only modify table with ALTER TABLE statement");
	}
	auto &table_info = info.Cast<AlterTableInfo>();
	switch (table_info.alter_table_type) {
	case AlterTableType::RENAME_COLUMN: {
		auto &rename_info = table_info.Cast<RenameColumnInfo>();
		return RenameColumn(context, rename_info);
	}
	case AlterTableType::RENAME_TABLE: {
		auto &rename_info = table_info.Cast<RenameTableInfo>();
		auto copied_table = Copy(context);
		copied_table->name = rename_info.new_table_name;
		storage->info->table = rename_info.new_table_name;
		return copied_table;
	}
	case AlterTableType::ADD_COLUMN: {
		auto &add_info = table_info.Cast<AddColumnInfo>();
		return AddColumn(context, add_info);
	}
	case AlterTableType::REMOVE_COLUMN: {
		auto &remove_info = table_info.Cast<RemoveColumnInfo>();
		return RemoveColumn(context, remove_info);
	}
	case AlterTableType::SET_DEFAULT: {
		auto &set_default_info = table_info.Cast<SetDefaultInfo>();
		return SetDefault(context, set_default_info);
	}
	case AlterTableType::ALTER_COLUMN_TYPE: {
		auto &change_type_info = table_info.Cast<ChangeColumnTypeInfo>();
		return ChangeColumnType(context, change_type_info);
	}
	case AlterTableType::FOREIGN_KEY_CONSTRAINT: {
		auto &foreign_key_constraint_info = table_info.Cast<AlterForeignKeyInfo>();
		if (foreign_key_constraint_info.type == AlterForeignKeyType::AFT_ADD) {
			return AddForeignKeyConstraint(context, foreign_key_constraint_info);
		} else {
			return DropForeignKeyConstraint(context, foreign_key_constraint_info);
		}
	}
	case AlterTableType::SET_NOT_NULL: {
		auto &set_not_null_info = table_info.Cast<SetNotNullInfo>();
		return SetNotNull(context, set_not_null_info);
	}
	case AlterTableType::DROP_NOT_NULL: {
		auto &drop_not_null_info = table_info.Cast<DropNotNullInfo>();
		return DropNotNull(context, drop_not_null_info);
	}
	default:
		throw InternalException("Unrecognized alter table type!");
	}
}

void DuckTableEntry::UndoAlter(ClientContext &context, AlterInfo &info) {
	D_ASSERT(!internal);
	D_ASSERT(info.type == AlterType::ALTER_TABLE);
	auto &table_info = info.Cast<AlterTableInfo>();
	switch (table_info.alter_table_type) {
	case AlterTableType::RENAME_TABLE: {
		storage->info->table = this->name;
		break;
	default:
		break;
	}
	}
}

static void RenameExpression(ParsedExpression &expr, RenameColumnInfo &info) {
	if (expr.type == ExpressionType::COLUMN_REF) {
		auto &colref = expr.Cast<ColumnRefExpression>();
		if (colref.column_names.back() == info.old_name) {
			colref.column_names.back() = info.new_name;
		}
	}
	ParsedExpressionIterator::EnumerateChildren(
	    expr, [&](const ParsedExpression &child) { RenameExpression((ParsedExpression &)child, info); });
}

unique_ptr<CatalogEntry> DuckTableEntry::RenameColumn(ClientContext &context, RenameColumnInfo &info) {
	auto rename_idx = GetColumnIndex(info.old_name);
	if (rename_idx.index == COLUMN_IDENTIFIER_ROW_ID) {
		throw CatalogException("Cannot rename rowid column");
	}
	auto create_info = make_uniq<CreateTableInfo>(schema, name);
	create_info->temporary = temporary;
	for (auto &col : columns.Logical()) {
		auto copy = col.Copy();
		if (rename_idx == col.Logical()) {
			copy.SetName(info.new_name);
		}
		if (col.Generated() && column_dependency_manager.IsDependencyOf(col.Logical(), rename_idx)) {
			RenameExpression(copy.GeneratedExpressionMutable(), info);
		}
		create_info->columns.AddColumn(std::move(copy));
	}
	for (idx_t c_idx = 0; c_idx < constraints.size(); c_idx++) {
		auto copy = constraints[c_idx]->Copy();
		switch (copy->type) {
		case ConstraintType::NOT_NULL:
			// NOT NULL constraint: no adjustments necessary
			break;
		case ConstraintType::CHECK: {
			// CHECK constraint: need to rename column references that refer to the renamed column
			auto &check = copy->Cast<CheckConstraint>();
			RenameExpression(*check.expression, info);
			break;
		}
		case ConstraintType::UNIQUE: {
			// UNIQUE constraint: possibly need to rename columns
			auto &unique = copy->Cast<UniqueConstraint>();
			for (idx_t i = 0; i < unique.columns.size(); i++) {
				if (unique.columns[i] == info.old_name) {
					unique.columns[i] = info.new_name;
				}
			}
			break;
		}
		case ConstraintType::FOREIGN_KEY: {
			// FOREIGN KEY constraint: possibly need to rename columns
			auto &fk = copy->Cast<ForeignKeyConstraint>();
			vector<string> columns = fk.pk_columns;
			if (fk.info.type == ForeignKeyType::FK_TYPE_FOREIGN_KEY_TABLE) {
				columns = fk.fk_columns;
			} else if (fk.info.type == ForeignKeyType::FK_TYPE_SELF_REFERENCE_TABLE) {
				for (idx_t i = 0; i < fk.fk_columns.size(); i++) {
					columns.push_back(fk.fk_columns[i]);
				}
			}
			for (idx_t i = 0; i < columns.size(); i++) {
				if (columns[i] == info.old_name) {
					throw CatalogException(
					    "Cannot rename column \"%s\" because this is involved in the foreign key constraint",
					    info.old_name);
				}
			}
			break;
		}
		default:
			throw InternalException("Unsupported constraint for entry!");
		}
		create_info->constraints.push_back(std::move(copy));
	}
	auto binder = Binder::CreateBinder(context);
	auto bound_create_info = binder->BindCreateTableInfo(std::move(create_info));
	return make_uniq<DuckTableEntry>(catalog, schema, *bound_create_info, storage);
}

unique_ptr<CatalogEntry> DuckTableEntry::AddColumn(ClientContext &context, AddColumnInfo &info) {
	auto col_name = info.new_column.GetName();

	// We're checking for the opposite condition (ADD COLUMN IF _NOT_ EXISTS ...).
	if (info.if_column_not_exists && ColumnExists(col_name)) {
		return nullptr;
	}

	auto create_info = make_uniq<CreateTableInfo>(schema, name);
	create_info->temporary = temporary;

	for (auto &col : columns.Logical()) {
		create_info->columns.AddColumn(col.Copy());
	}
	for (auto &constraint : constraints) {
		create_info->constraints.push_back(constraint->Copy());
	}
	Binder::BindLogicalType(context, info.new_column.TypeMutable(), &catalog, schema.name);
	info.new_column.SetOid(columns.LogicalColumnCount());
	info.new_column.SetStorageOid(columns.PhysicalColumnCount());
	auto col = info.new_column.Copy();

	create_info->columns.AddColumn(std::move(col));

	auto binder = Binder::CreateBinder(context);
	auto bound_create_info = binder->BindCreateTableInfo(std::move(create_info));
	auto new_storage =
	    make_shared<DataTable>(context, *storage, info.new_column, bound_create_info->bound_defaults.back().get());
	return make_uniq<DuckTableEntry>(catalog, schema, *bound_create_info, new_storage);
}

void DuckTableEntry::UpdateConstraintsOnColumnDrop(const LogicalIndex &removed_index,
                                                   const vector<LogicalIndex> &adjusted_indices,
                                                   const RemoveColumnInfo &info, CreateTableInfo &create_info,
                                                   bool is_generated) {
	// handle constraints for the new table
	D_ASSERT(constraints.size() == bound_constraints.size());

	for (idx_t constr_idx = 0; constr_idx < constraints.size(); constr_idx++) {
		auto &constraint = constraints[constr_idx];
		auto &bound_constraint = bound_constraints[constr_idx];
		switch (constraint->type) {
		case ConstraintType::NOT_NULL: {
			auto &not_null_constraint = bound_constraint->Cast<BoundNotNullConstraint>();
			auto not_null_index = columns.PhysicalToLogical(not_null_constraint.index);
			if (not_null_index != removed_index) {
				// the constraint is not about this column: we need to copy it
				// we might need to shift the index back by one though, to account for the removed column
				auto new_index = adjusted_indices[not_null_index.index];
				create_info.constraints.push_back(make_uniq<NotNullConstraint>(new_index));
			}
			break;
		}
		case ConstraintType::CHECK: {
			// Generated columns can not be part of an index
			// CHECK constraint
			auto &bound_check = bound_constraint->Cast<BoundCheckConstraint>();
			// check if the removed column is part of the check constraint
			if (is_generated) {
				// generated columns can not be referenced by constraints, we can just add the constraint back
				create_info.constraints.push_back(constraint->Copy());
				break;
			}
			auto physical_index = columns.LogicalToPhysical(removed_index);
			if (bound_check.bound_columns.find(physical_index) != bound_check.bound_columns.end()) {
				if (bound_check.bound_columns.size() > 1) {
					// CHECK constraint that concerns mult
					throw CatalogException(
					    "Cannot drop column \"%s\" because there is a CHECK constraint that depends on it",
					    info.removed_column);
				} else {
					// CHECK constraint that ONLY concerns this column, strip the constraint
				}
			} else {
				// check constraint does not concern the removed column: simply re-add it
				create_info.constraints.push_back(constraint->Copy());
			}
			break;
		}
		case ConstraintType::UNIQUE: {
			auto copy = constraint->Copy();
			auto &unique = copy->Cast<UniqueConstraint>();
			if (unique.index.index != DConstants::INVALID_INDEX) {
				if (unique.index == removed_index) {
					throw CatalogException(
					    "Cannot drop column \"%s\" because there is a UNIQUE constraint that depends on it",
					    info.removed_column);
				}
				unique.index = adjusted_indices[unique.index.index];
			}
			create_info.constraints.push_back(std::move(copy));
			break;
		}
		case ConstraintType::FOREIGN_KEY: {
			auto copy = constraint->Copy();
			auto &fk = copy->Cast<ForeignKeyConstraint>();
			vector<string> columns = fk.pk_columns;
			if (fk.info.type == ForeignKeyType::FK_TYPE_FOREIGN_KEY_TABLE) {
				columns = fk.fk_columns;
			} else if (fk.info.type == ForeignKeyType::FK_TYPE_SELF_REFERENCE_TABLE) {
				for (idx_t i = 0; i < fk.fk_columns.size(); i++) {
					columns.push_back(fk.fk_columns[i]);
				}
			}
			for (idx_t i = 0; i < columns.size(); i++) {
				if (columns[i] == info.removed_column) {
					throw CatalogException(
					    "Cannot drop column \"%s\" because there is a FOREIGN KEY constraint that depends on it",
					    info.removed_column);
				}
			}
			create_info.constraints.push_back(std::move(copy));
			break;
		}
		default:
			throw InternalException("Unsupported constraint for entry!");
		}
	}
}

unique_ptr<CatalogEntry> DuckTableEntry::RemoveColumn(ClientContext &context, RemoveColumnInfo &info) {
	auto removed_index = GetColumnIndex(info.removed_column, info.if_column_exists);
	if (!removed_index.IsValid()) {
		if (!info.if_column_exists) {
			throw CatalogException("Cannot drop column: rowid column cannot be dropped");
		}
		return nullptr;
	}

	auto create_info = make_uniq<CreateTableInfo>(schema, name);
	create_info->temporary = temporary;

	logical_index_set_t removed_columns;
	if (column_dependency_manager.HasDependents(removed_index)) {
		removed_columns = column_dependency_manager.GetDependents(removed_index);
	}
	if (!removed_columns.empty() && !info.cascade) {
		throw CatalogException("Cannot drop column: column is a dependency of 1 or more generated column(s)");
	}
	bool dropped_column_is_generated = false;
	for (auto &col : columns.Logical()) {
		if (col.Logical() == removed_index || removed_columns.count(col.Logical())) {
			if (col.Generated()) {
				dropped_column_is_generated = true;
			}
			continue;
		}
		create_info->columns.AddColumn(col.Copy());
	}
	if (create_info->columns.empty()) {
		throw CatalogException("Cannot drop column: table only has one column remaining!");
	}
	auto adjusted_indices = column_dependency_manager.RemoveColumn(removed_index, columns.LogicalColumnCount());

	UpdateConstraintsOnColumnDrop(removed_index, adjusted_indices, info, *create_info, dropped_column_is_generated);

	auto binder = Binder::CreateBinder(context);
	auto bound_create_info = binder->BindCreateTableInfo(std::move(create_info));
	if (columns.GetColumn(LogicalIndex(removed_index)).Generated()) {
		return make_uniq<DuckTableEntry>(catalog, schema, *bound_create_info, storage);
	}
	auto new_storage =
	    make_shared<DataTable>(context, *storage, columns.LogicalToPhysical(LogicalIndex(removed_index)).index);
	return make_uniq<DuckTableEntry>(catalog, schema, *bound_create_info, new_storage);
}

unique_ptr<CatalogEntry> DuckTableEntry::SetDefault(ClientContext &context, SetDefaultInfo &info) {
	auto create_info = make_uniq<CreateTableInfo>(schema, name);
	auto default_idx = GetColumnIndex(info.column_name);
	if (default_idx.index == COLUMN_IDENTIFIER_ROW_ID) {
		throw CatalogException("Cannot SET DEFAULT for rowid column");
	}

	// Copy all the columns, changing the value of the one that was specified by 'column_name'
	for (auto &col : columns.Logical()) {
		auto copy = col.Copy();
		if (default_idx == col.Logical()) {
			// set the default value of this column
			if (copy.Generated()) {
				throw BinderException("Cannot SET DEFAULT for generated column \"%s\"", col.Name());
			}
			copy.SetDefaultValue(info.expression ? info.expression->Copy() : nullptr);
		}
		create_info->columns.AddColumn(std::move(copy));
	}
	// Copy all the constraints
	for (idx_t i = 0; i < constraints.size(); i++) {
		auto constraint = constraints[i]->Copy();
		create_info->constraints.push_back(std::move(constraint));
	}

	auto binder = Binder::CreateBinder(context);
	auto bound_create_info = binder->BindCreateTableInfo(std::move(create_info));
	return make_uniq<DuckTableEntry>(catalog, schema, *bound_create_info, storage);
}

unique_ptr<CatalogEntry> DuckTableEntry::SetNotNull(ClientContext &context, SetNotNullInfo &info) {

	auto create_info = make_uniq<CreateTableInfo>(schema, name);
	create_info->columns = columns.Copy();

	auto not_null_idx = GetColumnIndex(info.column_name);
	if (columns.GetColumn(LogicalIndex(not_null_idx)).Generated()) {
		throw BinderException("Unsupported constraint for generated column!");
	}
	bool has_not_null = false;
	for (idx_t i = 0; i < constraints.size(); i++) {
		auto constraint = constraints[i]->Copy();
		if (constraint->type == ConstraintType::NOT_NULL) {
			auto &not_null = constraint->Cast<NotNullConstraint>();
			if (not_null.index == not_null_idx) {
				has_not_null = true;
			}
		}
		create_info->constraints.push_back(std::move(constraint));
	}
	if (!has_not_null) {
		create_info->constraints.push_back(make_uniq<NotNullConstraint>(not_null_idx));
	}
	auto binder = Binder::CreateBinder(context);
	auto bound_create_info = binder->BindCreateTableInfo(std::move(create_info));

	// Early return
	if (has_not_null) {
		return make_uniq<DuckTableEntry>(catalog, schema, *bound_create_info, storage);
	}

	// Return with new storage info. Note that we need the bound column index here.
	auto new_storage = make_shared<DataTable>(
	    context, *storage, make_uniq<BoundNotNullConstraint>(columns.LogicalToPhysical(LogicalIndex(not_null_idx))));
	return make_uniq<DuckTableEntry>(catalog, schema, *bound_create_info, new_storage);
}

unique_ptr<CatalogEntry> DuckTableEntry::DropNotNull(ClientContext &context, DropNotNullInfo &info) {
	auto create_info = make_uniq<CreateTableInfo>(schema, name);
	create_info->columns = columns.Copy();

	auto not_null_idx = GetColumnIndex(info.column_name);
	for (idx_t i = 0; i < constraints.size(); i++) {
		auto constraint = constraints[i]->Copy();
		// Skip/drop not_null
		if (constraint->type == ConstraintType::NOT_NULL) {
			auto &not_null = constraint->Cast<NotNullConstraint>();
			if (not_null.index == not_null_idx) {
				continue;
			}
		}
		create_info->constraints.push_back(std::move(constraint));
	}

	auto binder = Binder::CreateBinder(context);
	auto bound_create_info = binder->BindCreateTableInfo(std::move(create_info));
	return make_uniq<DuckTableEntry>(catalog, schema, *bound_create_info, storage);
}

unique_ptr<CatalogEntry> DuckTableEntry::ChangeColumnType(ClientContext &context, ChangeColumnTypeInfo &info) {
	if (info.target_type.id() == LogicalTypeId::USER) {
		info.target_type =
		    Catalog::GetType(context, catalog.GetName(), schema.name, UserType::GetTypeName(info.target_type));
	}
	auto change_idx = GetColumnIndex(info.column_name);
	auto create_info = make_uniq<CreateTableInfo>(schema, name);
	create_info->temporary = temporary;

	for (auto &col : columns.Logical()) {
		auto copy = col.Copy();
		if (change_idx == col.Logical()) {
			// set the type of this column
			if (copy.Generated()) {
				throw NotImplementedException("Changing types of generated columns is not supported yet");
			}
			copy.SetType(info.target_type);
		}
		// TODO: check if the generated_expression breaks, only delete it if it does
		if (copy.Generated() && column_dependency_manager.IsDependencyOf(col.Logical(), change_idx)) {
			throw BinderException(
			    "This column is referenced by the generated column \"%s\", so its type can not be changed",
			    copy.Name());
		}
		create_info->columns.AddColumn(std::move(copy));
	}

	for (idx_t i = 0; i < constraints.size(); i++) {
		auto constraint = constraints[i]->Copy();
		switch (constraint->type) {
		case ConstraintType::CHECK: {
			auto &bound_check = bound_constraints[i]->Cast<BoundCheckConstraint>();
			auto physical_index = columns.LogicalToPhysical(change_idx);
			if (bound_check.bound_columns.find(physical_index) != bound_check.bound_columns.end()) {
				throw BinderException("Cannot change the type of a column that has a CHECK constraint specified");
			}
			break;
		}
		case ConstraintType::NOT_NULL:
			break;
		case ConstraintType::UNIQUE: {
			auto &bound_unique = bound_constraints[i]->Cast<BoundUniqueConstraint>();
			if (bound_unique.key_set.find(change_idx) != bound_unique.key_set.end()) {
				throw BinderException(
				    "Cannot change the type of a column that has a UNIQUE or PRIMARY KEY constraint specified");
			}
			break;
		}
		case ConstraintType::FOREIGN_KEY: {
			auto &bfk = bound_constraints[i]->Cast<BoundForeignKeyConstraint>();
			auto key_set = bfk.pk_key_set;
			if (bfk.info.type == ForeignKeyType::FK_TYPE_FOREIGN_KEY_TABLE) {
				key_set = bfk.fk_key_set;
			} else if (bfk.info.type == ForeignKeyType::FK_TYPE_SELF_REFERENCE_TABLE) {
				for (idx_t i = 0; i < bfk.info.fk_keys.size(); i++) {
					key_set.insert(bfk.info.fk_keys[i]);
				}
			}
			if (key_set.find(columns.LogicalToPhysical(change_idx)) != key_set.end()) {
				throw BinderException("Cannot change the type of a column that has a FOREIGN KEY constraint specified");
			}
			break;
		}
		default:
			throw InternalException("Unsupported constraint for entry!");
		}
		create_info->constraints.push_back(std::move(constraint));
	}

	auto binder = Binder::CreateBinder(context);
	// bind the specified expression
	vector<LogicalIndex> bound_columns;
	AlterBinder expr_binder(*binder, context, *this, bound_columns, info.target_type);
	auto expression = info.expression->Copy();
	auto bound_expression = expr_binder.Bind(expression);
	auto bound_create_info = binder->BindCreateTableInfo(std::move(create_info));
	vector<column_t> storage_oids;
	for (idx_t i = 0; i < bound_columns.size(); i++) {
		storage_oids.push_back(columns.LogicalToPhysical(bound_columns[i]).index);
	}
	if (storage_oids.empty()) {
		storage_oids.push_back(COLUMN_IDENTIFIER_ROW_ID);
	}

	auto new_storage =
	    make_shared<DataTable>(context, *storage, columns.LogicalToPhysical(LogicalIndex(change_idx)).index,
	                           info.target_type, std::move(storage_oids), *bound_expression);
	auto result = make_uniq<DuckTableEntry>(catalog, schema, *bound_create_info, new_storage);
	return std::move(result);
}

unique_ptr<CatalogEntry> DuckTableEntry::AddForeignKeyConstraint(ClientContext &context, AlterForeignKeyInfo &info) {
	D_ASSERT(info.type == AlterForeignKeyType::AFT_ADD);
	auto create_info = make_uniq<CreateTableInfo>(schema, name);
	create_info->temporary = temporary;

	create_info->columns = columns.Copy();
	for (idx_t i = 0; i < constraints.size(); i++) {
		create_info->constraints.push_back(constraints[i]->Copy());
	}
	ForeignKeyInfo fk_info;
	fk_info.type = ForeignKeyType::FK_TYPE_PRIMARY_KEY_TABLE;
	fk_info.schema = info.schema;
	fk_info.table = info.fk_table;
	fk_info.pk_keys = info.pk_keys;
	fk_info.fk_keys = info.fk_keys;
	create_info->constraints.push_back(
	    make_uniq<ForeignKeyConstraint>(info.pk_columns, info.fk_columns, std::move(fk_info)));

	auto binder = Binder::CreateBinder(context);
	auto bound_create_info = binder->BindCreateTableInfo(std::move(create_info));

	return make_uniq<DuckTableEntry>(catalog, schema, *bound_create_info, storage);
}

unique_ptr<CatalogEntry> DuckTableEntry::DropForeignKeyConstraint(ClientContext &context, AlterForeignKeyInfo &info) {
	D_ASSERT(info.type == AlterForeignKeyType::AFT_DELETE);
	auto create_info = make_uniq<CreateTableInfo>(schema, name);
	create_info->temporary = temporary;

	create_info->columns = columns.Copy();
	for (idx_t i = 0; i < constraints.size(); i++) {
		auto constraint = constraints[i]->Copy();
		if (constraint->type == ConstraintType::FOREIGN_KEY) {
			ForeignKeyConstraint &fk = constraint->Cast<ForeignKeyConstraint>();
			if (fk.info.type == ForeignKeyType::FK_TYPE_PRIMARY_KEY_TABLE && fk.info.table == info.fk_table) {
				continue;
			}
		}
		create_info->constraints.push_back(std::move(constraint));
	}

	auto binder = Binder::CreateBinder(context);
	auto bound_create_info = binder->BindCreateTableInfo(std::move(create_info));

	return make_uniq<DuckTableEntry>(catalog, schema, *bound_create_info, storage);
}

unique_ptr<CatalogEntry> DuckTableEntry::Copy(ClientContext &context) const {
	auto create_info = make_uniq<CreateTableInfo>(schema, name);
	create_info->columns = columns.Copy();

	for (idx_t i = 0; i < constraints.size(); i++) {
		auto constraint = constraints[i]->Copy();
		create_info->constraints.push_back(std::move(constraint));
	}

	auto binder = Binder::CreateBinder(context);
	auto bound_create_info = binder->BindCreateTableInfo(std::move(create_info));
	return make_uniq<DuckTableEntry>(catalog, schema, *bound_create_info, storage);
}

void DuckTableEntry::SetAsRoot() {
	storage->SetAsRoot();
	storage->info->table = name;
}

void DuckTableEntry::CommitAlter(string &column_name) {
	D_ASSERT(!column_name.empty());
	idx_t removed_index = DConstants::INVALID_INDEX;
	for (auto &col : columns.Logical()) {
		if (col.Name() == column_name) {
			// No need to alter storage, removed column is generated column
			if (col.Generated()) {
				return;
			}
			removed_index = col.Oid();
			break;
		}
	}
	D_ASSERT(removed_index != DConstants::INVALID_INDEX);
	storage->CommitDropColumn(columns.LogicalToPhysical(LogicalIndex(removed_index)).index);
}

void DuckTableEntry::CommitDrop() {
	storage->CommitDropTable();
}

DataTable &DuckTableEntry::GetStorage() {
	return *storage;
}

const vector<unique_ptr<BoundConstraint>> &DuckTableEntry::GetBoundConstraints() {
	return bound_constraints;
}

TableFunction DuckTableEntry::GetScanFunction(ClientContext &context, unique_ptr<FunctionData> &bind_data) {
	bind_data = make_uniq<TableScanBindData>(*this);
	return TableScanFunction::GetFunction();
}

TableStorageInfo DuckTableEntry::GetStorageInfo(ClientContext &context) {
	TableStorageInfo result;
	result.cardinality = storage->info->cardinality.load();
	storage->GetStorageInfo(result);
	storage->info->indexes.Scan([&](Index &index) {
		IndexInfo info;
		info.is_primary = index.IsPrimary();
		info.is_unique = index.IsUnique() || info.is_primary;
		info.is_foreign = index.IsForeign();
		info.column_set = index.column_id_set;
		result.index_info.push_back(std::move(info));
		return false;
	});
	return result;
}

} // namespace duckdb




namespace duckdb {

IndexCatalogEntry::IndexCatalogEntry(Catalog &catalog, SchemaCatalogEntry &schema, CreateIndexInfo &info)
    : StandardEntry(CatalogType::INDEX_ENTRY, schema, catalog, info.index_name), index(nullptr), sql(info.sql) {
	this->temporary = info.temporary;
}

string IndexCatalogEntry::ToSQL() const {
	if (sql.empty()) {
		return sql;
	}
	if (sql[sql.size() - 1] != ';') {
		return sql + ";";
	}
	return sql;
}

void IndexCatalogEntry::Serialize(Serializer &serializer) const {
	// here we serialize the index metadata in the following order:
	// schema name, table name, index name, sql, index type, index constraint type, expression list, parsed expressions,
	// column IDs

	FieldWriter writer(serializer);
	writer.WriteString(GetSchemaName());
	writer.WriteString(GetTableName());
	writer.WriteString(name);
	writer.WriteString(sql);
	writer.WriteField(index->type);
	writer.WriteField(index->constraint_type);
	writer.WriteSerializableList(expressions);
	writer.WriteSerializableList(parsed_expressions);
	writer.WriteList<idx_t>(index->column_ids);
	writer.Finalize();
}

unique_ptr<CreateIndexInfo> IndexCatalogEntry::Deserialize(Deserializer &source, ClientContext &context) {
	// here we deserialize the index metadata in the following order:
	// schema name, table schema name, table name, index name, sql, index type, index constraint type, expression list,
	// parsed expression list, column IDs

	auto create_index_info = make_uniq<CreateIndexInfo>();

	FieldReader reader(source);

	create_index_info->schema = reader.ReadRequired<string>();
	create_index_info->table = make_uniq<BaseTableRef>();
	create_index_info->table->schema_name = create_index_info->schema;
	create_index_info->table->table_name = reader.ReadRequired<string>();
	create_index_info->index_name = reader.ReadRequired<string>();
	create_index_info->sql = reader.ReadRequired<string>();
	create_index_info->index_type = IndexType(reader.ReadRequired<uint8_t>());
	create_index_info->constraint_type = IndexConstraintType(reader.ReadRequired<uint8_t>());
	create_index_info->expressions = reader.ReadRequiredSerializableList<ParsedExpression>();
	create_index_info->parsed_expressions = reader.ReadRequiredSerializableList<ParsedExpression>();

	create_index_info->column_ids = reader.ReadRequiredList<idx_t>();
	reader.Finalize();
	return create_index_info;
}

} // namespace duckdb





namespace duckdb {

MacroCatalogEntry::MacroCatalogEntry(Catalog &catalog, SchemaCatalogEntry &schema, CreateMacroInfo &info)
    : FunctionEntry(
          (info.function->type == MacroType::SCALAR_MACRO ? CatalogType::MACRO_ENTRY : CatalogType::TABLE_MACRO_ENTRY),
          catalog, schema, info),
      function(std::move(info.function)) {
	this->temporary = info.temporary;
	this->internal = info.internal;
}

ScalarMacroCatalogEntry::ScalarMacroCatalogEntry(Catalog &catalog, SchemaCatalogEntry &schema, CreateMacroInfo &info)
    : MacroCatalogEntry(catalog, schema, info) {
}

TableMacroCatalogEntry::TableMacroCatalogEntry(Catalog &catalog, SchemaCatalogEntry &schema, CreateMacroInfo &info)
    : MacroCatalogEntry(catalog, schema, info) {
}

unique_ptr<CreateMacroInfo> MacroCatalogEntry::GetInfoForSerialization() const {
	auto info = make_uniq<CreateMacroInfo>(type);
	info->catalog = catalog.GetName();
	info->schema = schema.name;
	info->name = name;
	info->function = function->Copy();
	return info;
}
void MacroCatalogEntry::Serialize(Serializer &serializer) const {
	auto info = GetInfoForSerialization();
	info->Serialize(serializer);
}

unique_ptr<CreateMacroInfo> MacroCatalogEntry::Deserialize(Deserializer &main_source, ClientContext &context) {
	return unique_ptr_cast<CreateInfo, CreateMacroInfo>(CreateInfo::Deserialize(main_source));
}

} // namespace duckdb



namespace duckdb {

PragmaFunctionCatalogEntry::PragmaFunctionCatalogEntry(Catalog &catalog, SchemaCatalogEntry &schema,
                                                       CreatePragmaFunctionInfo &info)
    : FunctionEntry(CatalogType::PRAGMA_FUNCTION_ENTRY, catalog, schema, info), functions(std::move(info.functions)) {
}

} // namespace duckdb



namespace duckdb {

ScalarFunctionCatalogEntry::ScalarFunctionCatalogEntry(Catalog &catalog, SchemaCatalogEntry &schema,
                                                       CreateScalarFunctionInfo &info)
    : FunctionEntry(CatalogType::SCALAR_FUNCTION_ENTRY, catalog, schema, info), functions(info.functions) {
}

unique_ptr<CatalogEntry> ScalarFunctionCatalogEntry::AlterEntry(ClientContext &context, AlterInfo &info) {
	if (info.type != AlterType::ALTER_SCALAR_FUNCTION) {
		throw InternalException("Attempting to alter ScalarFunctionCatalogEntry with unsupported alter type");
	}
	auto &function_info = info.Cast<AlterScalarFunctionInfo>();
	if (function_info.alter_scalar_function_type != AlterScalarFunctionType::ADD_FUNCTION_OVERLOADS) {
		throw InternalException(
		    "Attempting to alter ScalarFunctionCatalogEntry with unsupported alter scalar function type");
	}
	auto &add_overloads = function_info.Cast<AddScalarFunctionOverloadInfo>();

	ScalarFunctionSet new_set = functions;
	if (!new_set.MergeFunctionSet(add_overloads.new_overloads)) {
		throw BinderException("Failed to add new function overloads to function \"%s\": function already exists", name);
	}
	CreateScalarFunctionInfo new_info(std::move(new_set));
	return make_uniq<ScalarFunctionCatalogEntry>(catalog, schema, new_info);
}

} // namespace duckdb









#include <sstream>

namespace duckdb {

SchemaCatalogEntry::SchemaCatalogEntry(Catalog &catalog, string name_p, bool internal)
    : InCatalogEntry(CatalogType::SCHEMA_ENTRY, catalog, std::move(name_p)) {
	this->internal = internal;
}

CatalogTransaction SchemaCatalogEntry::GetCatalogTransaction(ClientContext &context) {
	return CatalogTransaction(catalog, context);
}

SimilarCatalogEntry SchemaCatalogEntry::GetSimilarEntry(CatalogTransaction transaction, CatalogType type,
                                                        const string &name) {
	SimilarCatalogEntry result;
	Scan(transaction.GetContext(), type, [&](CatalogEntry &entry) {
		auto ldist = StringUtil::SimilarityScore(entry.name, name);
		if (ldist < result.distance) {
			result.distance = ldist;
			result.name = entry.name;
		}
	});
	return result;
}

void SchemaCatalogEntry::Serialize(Serializer &serializer) const {
	FieldWriter writer(serializer);
	writer.WriteString(name);
	writer.Finalize();
}

unique_ptr<CreateSchemaInfo> SchemaCatalogEntry::Deserialize(Deserializer &source) {
	auto info = make_uniq<CreateSchemaInfo>();

	FieldReader reader(source);
	info->schema = reader.ReadRequired<string>();
	reader.Finalize();

	return info;
}

string SchemaCatalogEntry::ToSQL() const {
	std::stringstream ss;
	ss << "CREATE SCHEMA " << name << ";";
	return ss.str();
}

} // namespace duckdb








#include <algorithm>
#include <sstream>

namespace duckdb {

SequenceCatalogEntry::SequenceCatalogEntry(Catalog &catalog, SchemaCatalogEntry &schema, CreateSequenceInfo &info)
    : StandardEntry(CatalogType::SEQUENCE_ENTRY, schema, catalog, info.name), usage_count(info.usage_count),
      counter(info.start_value), increment(info.increment), start_value(info.start_value), min_value(info.min_value),
      max_value(info.max_value), cycle(info.cycle) {
	this->temporary = info.temporary;
}

void SequenceCatalogEntry::Serialize(Serializer &serializer) const {
	FieldWriter writer(serializer);
	writer.WriteString(schema.name);
	writer.WriteString(name);
	writer.WriteField<uint64_t>(usage_count);
	writer.WriteField<int64_t>(increment);
	writer.WriteField<int64_t>(min_value);
	writer.WriteField<int64_t>(max_value);
	writer.WriteField<int64_t>(counter);
	writer.WriteField<bool>(cycle);
	writer.Finalize();
}

unique_ptr<CreateSequenceInfo> SequenceCatalogEntry::Deserialize(Deserializer &source) {
	auto info = make_uniq<CreateSequenceInfo>();

	FieldReader reader(source);
	info->schema = reader.ReadRequired<string>();
	info->name = reader.ReadRequired<string>();
	info->usage_count = reader.ReadRequired<uint64_t>();
	info->increment = reader.ReadRequired<int64_t>();
	info->min_value = reader.ReadRequired<int64_t>();
	info->max_value = reader.ReadRequired<int64_t>();
	info->start_value = reader.ReadRequired<int64_t>();
	info->cycle = reader.ReadRequired<bool>();
	reader.Finalize();

	return info;
}

string SequenceCatalogEntry::ToSQL() const {
	std::stringstream ss;
	ss << "CREATE SEQUENCE ";
	ss << name;
	ss << " INCREMENT BY " << increment;
	ss << " MINVALUE " << min_value;
	ss << " MAXVALUE " << max_value;
	ss << " START " << counter;
	ss << " " << (cycle ? "CYCLE" : "NO CYCLE") << ";";
	return ss.str();
}
} // namespace duckdb












#include <sstream>

namespace duckdb {

TableCatalogEntry::TableCatalogEntry(Catalog &catalog, SchemaCatalogEntry &schema, CreateTableInfo &info)
    : StandardEntry(CatalogType::TABLE_ENTRY, schema, catalog, info.table), columns(std::move(info.columns)),
      constraints(std::move(info.constraints)) {
	this->temporary = info.temporary;
}

bool TableCatalogEntry::HasGeneratedColumns() const {
	return columns.LogicalColumnCount() != columns.PhysicalColumnCount();
}

LogicalIndex TableCatalogEntry::GetColumnIndex(string &column_name, bool if_exists) {
	auto entry = columns.GetColumnIndex(column_name);
	if (!entry.IsValid()) {
		if (if_exists) {
			return entry;
		}
		throw BinderException("Table \"%s\" does not have a column with name \"%s\"", name, column_name);
	}
	return entry;
}

bool TableCatalogEntry::ColumnExists(const string &name) {
	return columns.ColumnExists(name);
}

const ColumnDefinition &TableCatalogEntry::GetColumn(const string &name) {
	return columns.GetColumn(name);
}

vector<LogicalType> TableCatalogEntry::GetTypes() {
	vector<LogicalType> types;
	for (auto &col : columns.Physical()) {
		types.push_back(col.Type());
	}
	return types;
}

CreateTableInfo TableCatalogEntry::GetTableInfoForSerialization() const {
	CreateTableInfo result;
	result.catalog = catalog.GetName();
	result.schema = schema.name;
	result.table = name;
	result.columns = columns.Copy();
	result.constraints.reserve(constraints.size());
	std::for_each(constraints.begin(), constraints.end(),
	              [&result](const unique_ptr<Constraint> &c) { result.constraints.emplace_back(c->Copy()); });
	return result;
}

void TableCatalogEntry::Serialize(Serializer &serializer) const {
	D_ASSERT(!internal);
	const auto info = GetTableInfoForSerialization();
	FieldWriter writer(serializer);
	writer.WriteString(info.catalog);
	writer.WriteString(info.schema);
	writer.WriteString(info.table);
	info.columns.Serialize(writer);
	writer.WriteSerializableList(info.constraints);
	writer.Finalize();
}

unique_ptr<CreateTableInfo> TableCatalogEntry::Deserialize(Deserializer &source, ClientContext &context) {
	auto info = make_uniq<CreateTableInfo>();

	FieldReader reader(source);
	info->catalog = reader.ReadRequired<string>();
	info->schema = reader.ReadRequired<string>();
	info->table = reader.ReadRequired<string>();
	info->columns = ColumnList::Deserialize(reader);
	info->constraints = reader.ReadRequiredSerializableList<Constraint>();
	reader.Finalize();

	return info;
}

string TableCatalogEntry::ColumnsToSQL(const ColumnList &columns, const vector<unique_ptr<Constraint>> &constraints) {
	std::stringstream ss;

	ss << "(";

	// find all columns that have NOT NULL specified, but are NOT primary key columns
	logical_index_set_t not_null_columns;
	logical_index_set_t unique_columns;
	logical_index_set_t pk_columns;
	unordered_set<string> multi_key_pks;
	vector<string> extra_constraints;
	for (auto &constraint : constraints) {
		if (constraint->type == ConstraintType::NOT_NULL) {
			auto &not_null = constraint->Cast<NotNullConstraint>();
			not_null_columns.insert(not_null.index);
		} else if (constraint->type == ConstraintType::UNIQUE) {
			auto &pk = constraint->Cast<UniqueConstraint>();
			vector<string> constraint_columns = pk.columns;
			if (pk.index.index != DConstants::INVALID_INDEX) {
				// no columns specified: single column constraint
				if (pk.is_primary_key) {
					pk_columns.insert(pk.index);
				} else {
					unique_columns.insert(pk.index);
				}
			} else {
				// multi-column constraint, this constraint needs to go at the end after all columns
				if (pk.is_primary_key) {
					// multi key pk column: insert set of columns into multi_key_pks
					for (auto &col : pk.columns) {
						multi_key_pks.insert(col);
					}
				}
				extra_constraints.push_back(constraint->ToString());
			}
		} else if (constraint->type == ConstraintType::FOREIGN_KEY) {
			auto &fk = constraint->Cast<ForeignKeyConstraint>();
			if (fk.info.type == ForeignKeyType::FK_TYPE_FOREIGN_KEY_TABLE ||
			    fk.info.type == ForeignKeyType::FK_TYPE_SELF_REFERENCE_TABLE) {
				extra_constraints.push_back(constraint->ToString());
			}
		} else {
			extra_constraints.push_back(constraint->ToString());
		}
	}

	for (auto &column : columns.Logical()) {
		if (column.Oid() > 0) {
			ss << ", ";
		}
		ss << KeywordHelper::WriteOptionallyQuoted(column.Name()) << " ";
		ss << column.Type().ToString();
		bool not_null = not_null_columns.find(column.Logical()) != not_null_columns.end();
		bool is_single_key_pk = pk_columns.find(column.Logical()) != pk_columns.end();
		bool is_multi_key_pk = multi_key_pks.find(column.Name()) != multi_key_pks.end();
		bool is_unique = unique_columns.find(column.Logical()) != unique_columns.end();
		if (not_null && !is_single_key_pk && !is_multi_key_pk) {
			// NOT NULL but not a primary key column
			ss << " NOT NULL";
		}
		if (is_single_key_pk) {
			// single column pk: insert constraint here
			ss << " PRIMARY KEY";
		}
		if (is_unique) {
			// single column unique: insert constraint here
			ss << " UNIQUE";
		}
		if (column.DefaultValue()) {
			ss << " DEFAULT(" << column.DefaultValue()->ToString() << ")";
		}
		if (column.Generated()) {
			ss << " GENERATED ALWAYS AS(" << column.GeneratedExpression().ToString() << ")";
		}
	}
	// print any extra constraints that still need to be printed
	for (auto &extra_constraint : extra_constraints) {
		ss << ", ";
		ss << extra_constraint;
	}

	ss << ")";
	return ss.str();
}

string TableCatalogEntry::ToSQL() const {
	std::stringstream ss;

	ss << "CREATE TABLE ";

	if (schema.name != DEFAULT_SCHEMA) {
		ss << KeywordHelper::WriteOptionallyQuoted(schema.name) << ".";
	}

	ss << KeywordHelper::WriteOptionallyQuoted(name);
	ss << ColumnsToSQL(columns, constraints);
	ss << ";";

	return ss.str();
}

const ColumnList &TableCatalogEntry::GetColumns() const {
	return columns;
}

ColumnList &TableCatalogEntry::GetColumnsMutable() {
	return columns;
}

const ColumnDefinition &TableCatalogEntry::GetColumn(LogicalIndex idx) {
	return columns.GetColumn(idx);
}

const vector<unique_ptr<Constraint>> &TableCatalogEntry::GetConstraints() {
	return constraints;
}

DataTable &TableCatalogEntry::GetStorage() {
	throw InternalException("Calling GetStorage on a TableCatalogEntry that is not a DuckTableEntry");
}

const vector<unique_ptr<BoundConstraint>> &TableCatalogEntry::GetBoundConstraints() {
	throw InternalException("Calling GetBoundConstraints on a TableCatalogEntry that is not a DuckTableEntry");
}
} // namespace duckdb



namespace duckdb {

TableFunctionCatalogEntry::TableFunctionCatalogEntry(Catalog &catalog, SchemaCatalogEntry &schema,
                                                     CreateTableFunctionInfo &info)
    : FunctionEntry(CatalogType::TABLE_FUNCTION_ENTRY, catalog, schema, info), functions(std::move(info.functions)) {
	D_ASSERT(this->functions.Size() > 0);
}

unique_ptr<CatalogEntry> TableFunctionCatalogEntry::AlterEntry(ClientContext &context, AlterInfo &info) {
	if (info.type != AlterType::ALTER_TABLE_FUNCTION) {
		throw InternalException("Attempting to alter TableFunctionCatalogEntry with unsupported alter type");
	}
	auto &function_info = info.Cast<AlterTableFunctionInfo>();
	if (function_info.alter_table_function_type != AlterTableFunctionType::ADD_FUNCTION_OVERLOADS) {
		throw InternalException(
		    "Attempting to alter TableFunctionCatalogEntry with unsupported alter table function type");
	}
	auto &add_overloads = function_info.Cast<AddTableFunctionOverloadInfo>();

	TableFunctionSet new_set = functions;
	if (!new_set.MergeFunctionSet(add_overloads.new_overloads)) {
		throw BinderException("Failed to add new function overloads to function \"%s\": function already exists", name);
	}
	CreateTableFunctionInfo new_info(std::move(new_set));
	return make_uniq<TableFunctionCatalogEntry>(catalog, schema, new_info);
}

} // namespace duckdb









#include <algorithm>
#include <sstream>

namespace duckdb {

TypeCatalogEntry::TypeCatalogEntry(Catalog &catalog, SchemaCatalogEntry &schema, CreateTypeInfo &info)
    : StandardEntry(CatalogType::TYPE_ENTRY, schema, catalog, info.name), user_type(info.type) {
	this->temporary = info.temporary;
	this->internal = info.internal;
}

void TypeCatalogEntry::Serialize(Serializer &serializer) const {
	D_ASSERT(!internal);
	FieldWriter writer(serializer);
	writer.WriteString(schema.name);
	writer.WriteString(name);
	if (user_type.id() == LogicalTypeId::ENUM) {
		// We have to serialize Enum Values
		writer.AddField();
		user_type.SerializeEnumType(writer.GetSerializer());
	} else {
		writer.WriteSerializable(user_type);
	}
	writer.Finalize();
}

unique_ptr<CreateTypeInfo> TypeCatalogEntry::Deserialize(Deserializer &source) {
	auto info = make_uniq<CreateTypeInfo>();

	FieldReader reader(source);
	info->schema = reader.ReadRequired<string>();
	info->name = reader.ReadRequired<string>();
	info->type = reader.ReadRequiredSerializable<LogicalType, LogicalType>();
	reader.Finalize();

	return info;
}

string TypeCatalogEntry::ToSQL() const {
	std::stringstream ss;
	switch (user_type.id()) {
	case (LogicalTypeId::ENUM): {
		auto &values_insert_order = EnumType::GetValuesInsertOrder(user_type);
		idx_t size = EnumType::GetSize(user_type);
		ss << "CREATE TYPE ";
		ss << KeywordHelper::WriteOptionallyQuoted(name);
		ss << " AS ENUM ( ";

		for (idx_t i = 0; i < size; i++) {
			ss << "'" << values_insert_order.GetValue(i).ToString() << "'";
			if (i != size - 1) {
				ss << ", ";
			}
		}
		ss << ");";
		break;
	}
	default:
		throw InternalException("Logical Type can't be used as a User Defined Type");
	}

	return ss.str();
}

} // namespace duckdb









#include <algorithm>

namespace duckdb {

void ViewCatalogEntry::Initialize(CreateViewInfo &info) {
	query = std::move(info.query);
	this->aliases = info.aliases;
	this->types = info.types;
	this->temporary = info.temporary;
	this->sql = info.sql;
	this->internal = info.internal;
}

ViewCatalogEntry::ViewCatalogEntry(Catalog &catalog, SchemaCatalogEntry &schema, CreateViewInfo &info)
    : StandardEntry(CatalogType::VIEW_ENTRY, schema, catalog, info.view_name) {
	Initialize(info);
}

unique_ptr<CatalogEntry> ViewCatalogEntry::AlterEntry(ClientContext &context, AlterInfo &info) {
	D_ASSERT(!internal);
	if (info.type != AlterType::ALTER_VIEW) {
		throw CatalogException("Can only modify view with ALTER VIEW statement");
	}
	auto &view_info = info.Cast<AlterViewInfo>();
	switch (view_info.alter_view_type) {
	case AlterViewType::RENAME_VIEW: {
		auto &rename_info = view_info.Cast<RenameViewInfo>();
		auto copied_view = Copy(context);
		copied_view->name = rename_info.new_view_name;
		return copied_view;
	}
	default:
		throw InternalException("Unrecognized alter view type!");
	}
}

void ViewCatalogEntry::Serialize(Serializer &serializer) const {
	D_ASSERT(!internal);
	FieldWriter writer(serializer);
	writer.WriteString(schema.name);
	writer.WriteString(name);
	writer.WriteString(sql);
	writer.WriteSerializable(*query);
	writer.WriteList<string>(aliases);
	writer.WriteRegularSerializableList<LogicalType>(types);
	writer.Finalize();
}

unique_ptr<CreateViewInfo> ViewCatalogEntry::Deserialize(Deserializer &source, ClientContext &context) {
	auto info = make_uniq<CreateViewInfo>();

	FieldReader reader(source);
	info->schema = reader.ReadRequired<string>();
	info->view_name = reader.ReadRequired<string>();
	info->sql = reader.ReadRequired<string>();
	info->query = reader.ReadRequiredSerializable<SelectStatement>();
	info->aliases = reader.ReadRequiredList<string>();
	info->types = reader.ReadRequiredSerializableList<LogicalType, LogicalType>();
	reader.Finalize();

	return info;
}

string ViewCatalogEntry::ToSQL() const {
	if (sql.empty()) {
		//! Return empty sql with view name so pragma view_tables don't complain
		return sql;
	}
	return sql + "\n;";
}

unique_ptr<CatalogEntry> ViewCatalogEntry::Copy(ClientContext &context) const {
	D_ASSERT(!internal);
	CreateViewInfo create_info(schema, name);
	create_info.query = unique_ptr_cast<SQLStatement, SelectStatement>(query->Copy());
	for (idx_t i = 0; i < aliases.size(); i++) {
		create_info.aliases.push_back(aliases[i]);
	}
	for (idx_t i = 0; i < types.size(); i++) {
		create_info.types.push_back(types[i]);
	}
	create_info.temporary = temporary;
	create_info.sql = sql;

	return make_uniq<ViewCatalogEntry>(catalog, schema, create_info);
}

} // namespace duckdb




namespace duckdb {

CatalogEntry::CatalogEntry(CatalogType type, string name_p, idx_t oid)
    : oid(oid), type(type), set(nullptr), name(std::move(name_p)), deleted(false), temporary(false), internal(false),
      parent(nullptr) {
}

CatalogEntry::CatalogEntry(CatalogType type, Catalog &catalog, string name_p)
    : CatalogEntry(type, std::move(name_p), catalog.ModifyCatalog()) {
}

CatalogEntry::~CatalogEntry() {
}

void CatalogEntry::SetAsRoot() {
}

// LCOV_EXCL_START
unique_ptr<CatalogEntry> CatalogEntry::AlterEntry(ClientContext &context, AlterInfo &info) {
	throw InternalException("Unsupported alter type for catalog entry!");
}

void CatalogEntry::UndoAlter(ClientContext &context, AlterInfo &info) {
}

unique_ptr<CatalogEntry> CatalogEntry::Copy(ClientContext &context) const {
	throw InternalException("Unsupported copy type for catalog entry!");
}

string CatalogEntry::ToSQL() const {
	throw InternalException("Unsupported catalog type for ToSQL()");
}
// LCOV_EXCL_STOP

void CatalogEntry::Verify(Catalog &catalog_p) {
}

Catalog &CatalogEntry::ParentCatalog() {
	throw InternalException("CatalogEntry::ParentCatalog called on catalog entry without catalog");
}

SchemaCatalogEntry &CatalogEntry::ParentSchema() {
	throw InternalException("CatalogEntry::ParentSchema called on catalog entry without schema");
}

InCatalogEntry::InCatalogEntry(CatalogType type, Catalog &catalog, string name)
    : CatalogEntry(type, catalog, std::move(name)), catalog(catalog) {
}

InCatalogEntry::~InCatalogEntry() {
}

void InCatalogEntry::Verify(Catalog &catalog_p) {
	D_ASSERT(&catalog_p == &catalog);
}
} // namespace duckdb








namespace duckdb {

CatalogSearchEntry::CatalogSearchEntry(string catalog_p, string schema_p)
    : catalog(std::move(catalog_p)), schema(std::move(schema_p)) {
}

string CatalogSearchEntry::ToString() const {
	if (catalog.empty()) {
		return WriteOptionallyQuoted(schema);
	} else {
		return WriteOptionallyQuoted(catalog) + "." + WriteOptionallyQuoted(schema);
	}
}

string CatalogSearchEntry::WriteOptionallyQuoted(const string &input) {
	for (idx_t i = 0; i < input.size(); i++) {
		if (input[i] == '.' || input[i] == ',') {
			return "\"" + input + "\"";
		}
	}
	return input;
}

string CatalogSearchEntry::ListToString(const vector<CatalogSearchEntry> &input) {
	string result;
	for (auto &entry : input) {
		if (!result.empty()) {
			result += ",";
		}
		result += entry.ToString();
	}
	return result;
}

CatalogSearchEntry CatalogSearchEntry::ParseInternal(const string &input, idx_t &idx) {
	string catalog;
	string schema;
	string entry;
	bool finished = false;
normal:
	for (; idx < input.size(); idx++) {
		if (input[idx] == '"') {
			idx++;
			goto quoted;
		} else if (input[idx] == '.') {
			goto separator;
		} else if (input[idx] == ',') {
			finished = true;
			goto separator;
		}
		entry += input[idx];
	}
	finished = true;
	goto separator;
quoted:
	//! look for another quote
	for (; idx < input.size(); idx++) {
		if (input[idx] == '"') {
			//! unquote
			idx++;
			goto normal;
		}
		entry += input[idx];
	}
	throw ParserException("Unterminated quote in qualified name!");
separator:
	if (entry.empty()) {
		throw ParserException("Unexpected dot - empty CatalogSearchEntry");
	}
	if (schema.empty()) {
		// if we parse one entry it is the schema
		schema = std::move(entry);
	} else if (catalog.empty()) {
		// if we parse two entries it is [catalog.schema]
		catalog = std::move(schema);
		schema = std::move(entry);
	} else {
		throw ParserException("Too many dots - expected [schema] or [catalog.schema] for CatalogSearchEntry");
	}
	entry = "";
	idx++;
	if (finished) {
		goto final;
	}
	goto normal;
final:
	if (schema.empty()) {
		throw ParserException("Unexpected end of entry - empty CatalogSearchEntry");
	}
	return CatalogSearchEntry(std::move(catalog), std::move(schema));
}

CatalogSearchEntry CatalogSearchEntry::Parse(const string &input) {
	idx_t pos = 0;
	auto result = ParseInternal(input, pos);
	if (pos < input.size()) {
		throw ParserException("Failed to convert entry \"%s\" to CatalogSearchEntry - expected a single entry", input);
	}
	return result;
}

vector<CatalogSearchEntry> CatalogSearchEntry::ParseList(const string &input) {
	idx_t pos = 0;
	vector<CatalogSearchEntry> result;
	while (pos < input.size()) {
		auto entry = ParseInternal(input, pos);
		result.push_back(entry);
	}
	return result;
}

CatalogSearchPath::CatalogSearchPath(ClientContext &context_p) : context(context_p) {
	Reset();
}

void CatalogSearchPath::Reset() {
	vector<CatalogSearchEntry> empty;
	SetPaths(empty);
}

void CatalogSearchPath::Set(vector<CatalogSearchEntry> new_paths, bool is_set_schema) {
	if (is_set_schema && new_paths.size() != 1) {
		throw CatalogException("SET schema can set only 1 schema. This has %d", new_paths.size());
	}
	for (auto &path : new_paths) {
		if (!Catalog::GetSchema(context, path.catalog, path.schema, OnEntryNotFound::RETURN_NULL)) {
			if (path.catalog.empty()) {
				// only schema supplied - check if this is a database instead
				auto schema = Catalog::GetSchema(context, path.schema, DEFAULT_SCHEMA, OnEntryNotFound::RETURN_NULL);
				if (schema) {
					path.catalog = std::move(path.schema);
					path.schema = schema->name;
					continue;
				}
			}
			throw CatalogException("SET %s: No catalog + schema named %s found.",
			                       is_set_schema ? "schema" : "search_path", path.ToString());
		}
	}
	if (is_set_schema) {
		if (new_paths[0].catalog == TEMP_CATALOG || new_paths[0].catalog == SYSTEM_CATALOG) {
			throw CatalogException("SET schema cannot be set to internal schema \"%s\"", new_paths[0].catalog);
		}
	}
	this->set_paths = std::move(new_paths);
	SetPaths(set_paths);
}

void CatalogSearchPath::Set(CatalogSearchEntry new_value, bool is_set_schema) {
	vector<CatalogSearchEntry> new_paths {std::move(new_value)};
	Set(std::move(new_paths), is_set_schema);
}

const vector<CatalogSearchEntry> &CatalogSearchPath::Get() {
	return paths;
}

string CatalogSearchPath::GetDefaultSchema(const string &catalog) {
	for (auto &path : paths) {
		if (path.catalog == TEMP_CATALOG) {
			continue;
		}
		if (StringUtil::CIEquals(path.catalog, catalog)) {
			return path.schema;
		}
	}
	return DEFAULT_SCHEMA;
}

string CatalogSearchPath::GetDefaultCatalog(const string &schema) {
	for (auto &path : paths) {
		if (path.catalog == TEMP_CATALOG) {
			continue;
		}
		if (StringUtil::CIEquals(path.schema, schema)) {
			return path.catalog;
		}
	}
	return INVALID_CATALOG;
}

vector<string> CatalogSearchPath::GetCatalogsForSchema(const string &schema) {
	vector<string> schemas;
	for (auto &path : paths) {
		if (StringUtil::CIEquals(path.schema, schema)) {
			schemas.push_back(path.catalog);
		}
	}
	return schemas;
}

vector<string> CatalogSearchPath::GetSchemasForCatalog(const string &catalog) {
	vector<string> schemas;
	for (auto &path : paths) {
		if (StringUtil::CIEquals(path.catalog, catalog)) {
			schemas.push_back(path.schema);
		}
	}
	return schemas;
}

const CatalogSearchEntry &CatalogSearchPath::GetDefault() {
	const auto &paths = Get();
	D_ASSERT(paths.size() >= 2);
	return paths[1];
}

void CatalogSearchPath::SetPaths(vector<CatalogSearchEntry> new_paths) {
	paths.clear();
	paths.reserve(new_paths.size() + 3);
	paths.emplace_back(TEMP_CATALOG, DEFAULT_SCHEMA);
	for (auto &path : new_paths) {
		paths.push_back(std::move(path));
	}
	paths.emplace_back(INVALID_CATALOG, DEFAULT_SCHEMA);
	paths.emplace_back(SYSTEM_CATALOG, DEFAULT_SCHEMA);
	paths.emplace_back(SYSTEM_CATALOG, "pg_catalog");
}

} // namespace duckdb
















namespace duckdb {

//! Class responsible to keep track of state when removing entries from the catalog.
//! When deleting, many types of errors can be thrown, since we want to avoid try/catch blocks
//! this class makes sure that whatever elements were modified are returned to a correct state
//! when exceptions are thrown.
//! The idea here is to use RAII (Resource acquisition is initialization) to mimic a try/catch/finally block.
//! If any exception is raised when this object exists, then its destructor will be called
//! and the entry will return to its previous state during deconstruction.
class EntryDropper {
public:
	//! Both constructor and destructor are privates because they should only be called by DropEntryDependencies
	explicit EntryDropper(EntryIndex &entry_index_p) : entry_index(entry_index_p) {
		old_deleted = entry_index.GetEntry()->deleted;
	}

	~EntryDropper() {
		entry_index.GetEntry()->deleted = old_deleted;
	}

private:
	//! Keeps track of the state of the entry before starting the delete
	bool old_deleted;
	//! Index of entry to be deleted
	EntryIndex &entry_index;
};

CatalogSet::CatalogSet(Catalog &catalog_p, unique_ptr<DefaultGenerator> defaults)
    : catalog((DuckCatalog &)catalog_p), defaults(std::move(defaults)) {
	D_ASSERT(catalog_p.IsDuckCatalog());
}
CatalogSet::~CatalogSet() {
}

EntryIndex CatalogSet::PutEntry(idx_t entry_index, unique_ptr<CatalogEntry> entry) {
	if (entries.find(entry_index) != entries.end()) {
		throw InternalException("Entry with entry index \"%llu\" already exists", entry_index);
	}
	entries.insert(make_pair(entry_index, EntryValue(std::move(entry))));
	return EntryIndex(*this, entry_index);
}

void CatalogSet::PutEntry(EntryIndex index, unique_ptr<CatalogEntry> catalog_entry) {
	auto entry = entries.find(index.GetIndex());
	if (entry == entries.end()) {
		throw InternalException("Entry with entry index \"%llu\" does not exist", index.GetIndex());
	}
	catalog_entry->child = std::move(entry->second.entry);
	catalog_entry->child->parent = catalog_entry.get();
	entry->second.entry = std::move(catalog_entry);
}

bool CatalogSet::CreateEntry(CatalogTransaction transaction, const string &name, unique_ptr<CatalogEntry> value,
                             DependencyList &dependencies) {
	if (value->internal && !catalog.IsSystemCatalog() && name != DEFAULT_SCHEMA) {
		throw InternalException("Attempting to create internal entry \"%s\" in non-system catalog - internal entries "
		                        "can only be created in the system catalog",
		                        name);
	}
	if (!value->internal) {
		if (!value->temporary && catalog.IsSystemCatalog()) {
			throw InternalException(
			    "Attempting to create non-internal entry \"%s\" in system catalog - the system catalog "
			    "can only contain internal entries",
			    name);
		}
		if (value->temporary && !catalog.IsTemporaryCatalog()) {
			throw InternalException("Attempting to create temporary entry \"%s\" in non-temporary catalog", name);
		}
		if (!value->temporary && catalog.IsTemporaryCatalog() && name != DEFAULT_SCHEMA) {
			throw InvalidInputException("Cannot create non-temporary entry \"%s\" in temporary catalog", name);
		}
	}
	// lock the catalog for writing
	lock_guard<mutex> write_lock(catalog.GetWriteLock());
	// lock this catalog set to disallow reading
	unique_lock<mutex> read_lock(catalog_lock);

	// first check if the entry exists in the unordered set
	idx_t index;
	auto mapping_value = GetMapping(transaction, name);
	if (mapping_value == nullptr || mapping_value->deleted) {
		// if it does not: entry has never been created

		// check if there is a default entry
		auto entry = CreateDefaultEntry(transaction, name, read_lock);
		if (entry) {
			return false;
		}

		// first create a dummy deleted entry for this entry
		// so transactions started before the commit of this transaction don't
		// see it yet
		auto dummy_node = make_uniq<InCatalogEntry>(CatalogType::INVALID, value->ParentCatalog(), name);
		dummy_node->timestamp = 0;
		dummy_node->deleted = true;
		dummy_node->set = this;

		auto entry_index = PutEntry(current_entry++, std::move(dummy_node));
		index = entry_index.GetIndex();
		PutMapping(transaction, name, std::move(entry_index));
	} else {
		index = mapping_value->index.GetIndex();
		auto &current = *mapping_value->index.GetEntry();
		// if it does, we have to check version numbers
		if (HasConflict(transaction, current.timestamp)) {
			// current version has been written to by a currently active
			// transaction
			throw TransactionException("Catalog write-write conflict on create with \"%s\"", current.name);
		}
		// there is a current version that has been committed
		// if it has not been deleted there is a conflict
		if (!current.deleted) {
			return false;
		}
	}
	// create a new entry and replace the currently stored one
	// set the timestamp to the timestamp of the current transaction
	// and point it at the dummy node
	value->timestamp = transaction.transaction_id;
	value->set = this;

	// now add the dependency set of this object to the dependency manager
	catalog.GetDependencyManager().AddObject(transaction, *value, dependencies);

	auto value_ptr = value.get();
	EntryIndex entry_index(*this, index);
	PutEntry(std::move(entry_index), std::move(value));
	// push the old entry in the undo buffer for this transaction
	if (transaction.transaction) {
		auto &dtransaction = transaction.transaction->Cast<DuckTransaction>();
		dtransaction.PushCatalogEntry(*value_ptr->child);
	}
	return true;
}

bool CatalogSet::CreateEntry(ClientContext &context, const string &name, unique_ptr<CatalogEntry> value,
                             DependencyList &dependencies) {
	return CreateEntry(catalog.GetCatalogTransaction(context), name, std::move(value), dependencies);
}

optional_ptr<CatalogEntry> CatalogSet::GetEntryInternal(CatalogTransaction transaction, EntryIndex &entry_index) {
	auto &catalog_entry = *entry_index.GetEntry();
	// if it does: we have to retrieve the entry and to check version numbers
	if (HasConflict(transaction, catalog_entry.timestamp)) {
		// current version has been written to by a currently active
		// transaction
		throw TransactionException("Catalog write-write conflict on alter with \"%s\"", catalog_entry.name);
	}
	// there is a current version that has been committed by this transaction
	if (catalog_entry.deleted) {
		// if the entry was already deleted, it now does not exist anymore
		// so we return that we could not find it
		return nullptr;
	}
	return &catalog_entry;
}

optional_ptr<CatalogEntry> CatalogSet::GetEntryInternal(CatalogTransaction transaction, const string &name,
                                                        EntryIndex *entry_index) {
	auto mapping_value = GetMapping(transaction, name);
	if (mapping_value == nullptr || mapping_value->deleted) {
		// the entry does not exist, check if we can create a default entry
		return nullptr;
	}
	if (entry_index) {
		*entry_index = mapping_value->index.Copy();
	}
	return GetEntryInternal(transaction, mapping_value->index);
}

bool CatalogSet::AlterOwnership(CatalogTransaction transaction, ChangeOwnershipInfo &info) {
	auto entry = GetEntryInternal(transaction, info.name, nullptr);
	if (!entry) {
		return false;
	}

	auto &owner_entry = catalog.GetEntry(transaction.GetContext(), info.owner_schema, info.owner_name);
	catalog.GetDependencyManager().AddOwnership(transaction, owner_entry, *entry);
	return true;
}

bool CatalogSet::AlterEntry(CatalogTransaction transaction, const string &name, AlterInfo &alter_info) {
	// lock the catalog for writing
	lock_guard<mutex> write_lock(catalog.GetWriteLock());

	// first check if the entry exists in the unordered set
	EntryIndex entry_index;
	auto entry = GetEntryInternal(transaction, name, &entry_index);
	if (!entry) {
		return false;
	}
	if (!alter_info.allow_internal && entry->internal) {
		throw CatalogException("Cannot alter entry \"%s\" because it is an internal system entry", entry->name);
	}

	// lock this catalog set to disallow reading
	lock_guard<mutex> read_lock(catalog_lock);

	// create a new entry and replace the currently stored one
	// set the timestamp to the timestamp of the current transaction
	// and point it to the updated table node
	string original_name = entry->name;
	if (!transaction.context) {
		throw InternalException("Cannot AlterEntry without client context");
	}
	auto &context = *transaction.context;
	auto value = entry->AlterEntry(context, alter_info);
	if (!value) {
		// alter failed, but did not result in an error
		return true;
	}

	if (value->name != original_name) {
		auto mapping_value = GetMapping(transaction, value->name);
		if (mapping_value && !mapping_value->deleted) {
			auto &original_entry = GetEntryForTransaction(transaction, *mapping_value->index.GetEntry());
			if (!original_entry.deleted) {
				entry->UndoAlter(context, alter_info);
				string rename_err_msg =
				    "Could not rename \"%s\" to \"%s\": another entry with this name already exists!";
				throw CatalogException(rename_err_msg, original_name, value->name);
			}
		}
	}

	if (value->name != original_name) {
		// Do PutMapping and DeleteMapping after dependency check
		PutMapping(transaction, value->name, entry_index.Copy());
		DeleteMapping(transaction, original_name);
	}

	value->timestamp = transaction.transaction_id;
	value->set = this;
	auto new_entry = value.get();
	PutEntry(std::move(entry_index), std::move(value));

	// serialize the AlterInfo into a temporary buffer
	BufferedSerializer serializer;
	serializer.WriteString(alter_info.GetColumnName());
	alter_info.Serialize(serializer);
	BinaryData serialized_alter = serializer.GetData();

	// push the old entry in the undo buffer for this transaction
	if (transaction.transaction) {
		auto &dtransaction = transaction.transaction->Cast<DuckTransaction>();
		dtransaction.PushCatalogEntry(*new_entry->child, serialized_alter.data.get(), serialized_alter.size);
	}

	// Check the dependency manager to verify that there are no conflicting dependencies with this alter
	// Note that we do this AFTER the new entry has been entirely set up in the catalog set
	// that is because in case the alter fails because of a dependency conflict, we need to be able to cleanly roll back
	// to the old entry.
	catalog.GetDependencyManager().AlterObject(transaction, *entry, *new_entry);

	return true;
}

void CatalogSet::DropEntryDependencies(CatalogTransaction transaction, EntryIndex &entry_index, CatalogEntry &entry,
                                       bool cascade) {
	// Stores the deleted value of the entry before starting the process
	EntryDropper dropper(entry_index);

	// To correctly delete the object and its dependencies, it temporarily is set to deleted.
	entry_index.GetEntry()->deleted = true;

	// check any dependencies of this object
	D_ASSERT(entry.ParentCatalog().IsDuckCatalog());
	auto &duck_catalog = entry.ParentCatalog().Cast<DuckCatalog>();
	duck_catalog.GetDependencyManager().DropObject(transaction, entry, cascade);

	// dropper destructor is called here
	// the destructor makes sure to return the value to the previous state
	// dropper.~EntryDropper()
}

void CatalogSet::DropEntryInternal(CatalogTransaction transaction, EntryIndex entry_index, CatalogEntry &entry,
                                   bool cascade) {
	DropEntryDependencies(transaction, entry_index, entry, cascade);

	// create a new entry and replace the currently stored one
	// set the timestamp to the timestamp of the current transaction
	// and point it at the dummy node
	auto value = make_uniq<InCatalogEntry>(CatalogType::DELETED_ENTRY, entry.ParentCatalog(), entry.name);
	value->timestamp = transaction.transaction_id;
	value->set = this;
	value->deleted = true;
	auto value_ptr = value.get();
	PutEntry(std::move(entry_index), std::move(value));

	// push the old entry in the undo buffer for this transaction
	if (transaction.transaction) {
		auto &dtransaction = transaction.transaction->Cast<DuckTransaction>();
		dtransaction.PushCatalogEntry(*value_ptr->child);
	}
}

bool CatalogSet::DropEntry(CatalogTransaction transaction, const string &name, bool cascade, bool allow_drop_internal) {
	// lock the catalog for writing
	lock_guard<mutex> write_lock(catalog.GetWriteLock());
	// we can only delete an entry that exists
	EntryIndex entry_index;
	auto entry = GetEntryInternal(transaction, name, &entry_index);
	if (!entry) {
		return false;
	}
	if (entry->internal && !allow_drop_internal) {
		throw CatalogException("Cannot drop entry \"%s\" because it is an internal system entry", entry->name);
	}

	lock_guard<mutex> read_lock(catalog_lock);
	DropEntryInternal(transaction, std::move(entry_index), *entry, cascade);
	return true;
}

bool CatalogSet::DropEntry(ClientContext &context, const string &name, bool cascade, bool allow_drop_internal) {
	return DropEntry(catalog.GetCatalogTransaction(context), name, cascade, allow_drop_internal);
}

DuckCatalog &CatalogSet::GetCatalog() {
	return catalog;
}

void CatalogSet::CleanupEntry(CatalogEntry &catalog_entry) {
	// destroy the backed up entry: it is no longer required
	D_ASSERT(catalog_entry.parent);
	if (catalog_entry.parent->type != CatalogType::UPDATED_ENTRY) {
		lock_guard<mutex> write_lock(catalog.GetWriteLock());
		lock_guard<mutex> lock(catalog_lock);
		if (!catalog_entry.deleted) {
			// delete the entry from the dependency manager, if it is not deleted yet
			D_ASSERT(catalog_entry.ParentCatalog().IsDuckCatalog());
			catalog_entry.ParentCatalog().Cast<DuckCatalog>().GetDependencyManager().EraseObject(catalog_entry);
		}
		auto parent = catalog_entry.parent;
		parent->child = std::move(catalog_entry.child);
		if (parent->deleted && !parent->child && !parent->parent) {
			auto mapping_entry = mapping.find(parent->name);
			D_ASSERT(mapping_entry != mapping.end());
			auto &entry = mapping_entry->second->index.GetEntry();
			D_ASSERT(entry);
			if (entry.get() == parent.get()) {
				mapping.erase(mapping_entry);
			}
		}
	}
}

bool CatalogSet::HasConflict(CatalogTransaction transaction, transaction_t timestamp) {
	return (timestamp >= TRANSACTION_ID_START && timestamp != transaction.transaction_id) ||
	       (timestamp < TRANSACTION_ID_START && timestamp > transaction.start_time);
}

optional_ptr<MappingValue> CatalogSet::GetMapping(CatalogTransaction transaction, const string &name, bool get_latest) {
	optional_ptr<MappingValue> mapping_value;
	auto entry = mapping.find(name);
	if (entry != mapping.end()) {
		mapping_value = entry->second.get();
	} else {

		return nullptr;
	}
	if (get_latest) {
		return mapping_value;
	}
	while (mapping_value->child) {
		if (UseTimestamp(transaction, mapping_value->timestamp)) {
			break;
		}
		mapping_value = mapping_value->child.get();
		D_ASSERT(mapping_value);
	}
	return mapping_value;
}

void CatalogSet::PutMapping(CatalogTransaction transaction, const string &name, EntryIndex entry_index) {
	auto entry = mapping.find(name);
	auto new_value = make_uniq<MappingValue>(std::move(entry_index));
	new_value->timestamp = transaction.transaction_id;
	if (entry != mapping.end()) {
		if (HasConflict(transaction, entry->second->timestamp)) {
			throw TransactionException("Catalog write-write conflict on name \"%s\"", name);
		}
		new_value->child = std::move(entry->second);
		new_value->child->parent = new_value.get();
	}
	mapping[name] = std::move(new_value);
}

void CatalogSet::DeleteMapping(CatalogTransaction transaction, const string &name) {
	auto entry = mapping.find(name);
	D_ASSERT(entry != mapping.end());
	auto delete_marker = make_uniq<MappingValue>(entry->second->index.Copy());
	delete_marker->deleted = true;
	delete_marker->timestamp = transaction.transaction_id;
	delete_marker->child = std::move(entry->second);
	delete_marker->child->parent = delete_marker.get();
	mapping[name] = std::move(delete_marker);
}

bool CatalogSet::UseTimestamp(CatalogTransaction transaction, transaction_t timestamp) {
	if (timestamp == transaction.transaction_id) {
		// we created this version
		return true;
	}
	if (timestamp < transaction.start_time) {
		// this version was commited before we started the transaction
		return true;
	}
	return false;
}

CatalogEntry &CatalogSet::GetEntryForTransaction(CatalogTransaction transaction, CatalogEntry &current) {
	reference<CatalogEntry> entry(current);
	while (entry.get().child) {
		if (UseTimestamp(transaction, entry.get().timestamp)) {
			break;
		}
		entry = *entry.get().child;
	}
	return entry.get();
}

CatalogEntry &CatalogSet::GetCommittedEntry(CatalogEntry &current) {
	reference<CatalogEntry> entry(current);
	while (entry.get().child) {
		if (entry.get().timestamp < TRANSACTION_ID_START) {
			// this entry is committed: use it
			break;
		}
		entry = *entry.get().child;
	}
	return entry.get();
}

SimilarCatalogEntry CatalogSet::SimilarEntry(CatalogTransaction transaction, const string &name) {
	unique_lock<mutex> lock(catalog_lock);
	CreateDefaultEntries(transaction, lock);

	SimilarCatalogEntry result;
	for (auto &kv : mapping) {
		auto mapping_value = GetMapping(transaction, kv.first);
		if (mapping_value && !mapping_value->deleted) {
			auto ldist = StringUtil::SimilarityScore(kv.first, name);
			if (ldist < result.distance) {
				result.distance = ldist;
				result.name = kv.first;
			}
		}
	}
	return result;
}

optional_ptr<CatalogEntry> CatalogSet::CreateEntryInternal(CatalogTransaction transaction,
                                                           unique_ptr<CatalogEntry> entry) {
	if (mapping.find(entry->name) != mapping.end()) {
		return nullptr;
	}
	auto &name = entry->name;
	auto catalog_entry = entry.get();

	entry->set = this;
	entry->timestamp = 0;

	auto entry_index = PutEntry(current_entry++, std::move(entry));
	PutMapping(transaction, name, std::move(entry_index));
	mapping[name]->timestamp = 0;
	return catalog_entry;
}

optional_ptr<CatalogEntry> CatalogSet::CreateDefaultEntry(CatalogTransaction transaction, const string &name,
                                                          unique_lock<mutex> &lock) {
	// no entry found with this name, check for defaults
	if (!defaults || defaults->created_all_entries) {
		// no defaults either: return null
		return nullptr;
	}
	// this catalog set has a default map defined
	// check if there is a default entry that we can create with this name
	if (!transaction.context) {
		// no context - cannot create default entry
		return nullptr;
	}
	lock.unlock();
	auto entry = defaults->CreateDefaultEntry(*transaction.context, name);

	lock.lock();
	if (!entry) {
		// no default entry
		return nullptr;
	}
	// there is a default entry! create it
	auto result = CreateEntryInternal(transaction, std::move(entry));
	if (result) {
		return result;
	}
	// we found a default entry, but failed
	// this means somebody else created the entry first
	// just retry?
	lock.unlock();
	return GetEntry(transaction, name);
}

optional_ptr<CatalogEntry> CatalogSet::GetEntry(CatalogTransaction transaction, const string &name) {
	unique_lock<mutex> lock(catalog_lock);
	auto mapping_value = GetMapping(transaction, name);
	if (mapping_value != nullptr && !mapping_value->deleted) {
		// we found an entry for this name
		// check the version numbers

		auto &catalog_entry = *mapping_value->index.GetEntry();
		auto &current = GetEntryForTransaction(transaction, catalog_entry);
		if (current.deleted || (current.name != name && !UseTimestamp(transaction, mapping_value->timestamp))) {
			return nullptr;
		}
		return &current;
	}
	return CreateDefaultEntry(transaction, name, lock);
}

optional_ptr<CatalogEntry> CatalogSet::GetEntry(ClientContext &context, const string &name) {
	return GetEntry(catalog.GetCatalogTransaction(context), name);
}

void CatalogSet::UpdateTimestamp(CatalogEntry &entry, transaction_t timestamp) {
	entry.timestamp = timestamp;
	mapping[entry.name]->timestamp = timestamp;
}

void CatalogSet::AdjustUserDependency(CatalogEntry &entry, ColumnDefinition &column, bool remove) {
	auto user_type_catalog_p = EnumType::GetCatalog(column.Type());
	if (!user_type_catalog_p) {
		return;
	}
	auto &user_type_catalog = user_type_catalog_p->Cast<CatalogEntry>();
	auto &dependency_manager = catalog.GetDependencyManager();
	if (remove) {
		dependency_manager.dependents_map[user_type_catalog].erase(*entry.parent);
		dependency_manager.dependencies_map[*entry.parent].erase(user_type_catalog);
	} else {
		dependency_manager.dependents_map[user_type_catalog].insert(entry);
		dependency_manager.dependencies_map[entry].insert(user_type_catalog);
	}
}

void CatalogSet::AdjustDependency(CatalogEntry &entry, TableCatalogEntry &table, ColumnDefinition &column,
                                  bool remove) {
	bool found = false;
	if (column.Type().id() == LogicalTypeId::ENUM) {
		for (auto &old_column : table.GetColumns().Logical()) {
			if (old_column.Name() == column.Name() && old_column.Type().id() != LogicalTypeId::ENUM) {
				AdjustUserDependency(entry, column, remove);
				found = true;
			}
		}
		if (!found) {
			AdjustUserDependency(entry, column, remove);
		}
	} else if (!(column.Type().GetAlias().empty())) {
		auto alias = column.Type().GetAlias();
		for (auto &old_column : table.GetColumns().Logical()) {
			auto old_alias = old_column.Type().GetAlias();
			if (old_column.Name() == column.Name() && old_alias != alias) {
				AdjustUserDependency(entry, column, remove);
				found = true;
			}
		}
		if (!found) {
			AdjustUserDependency(entry, column, remove);
		}
	}
}

void CatalogSet::AdjustTableDependencies(CatalogEntry &entry) {
	if (entry.type == CatalogType::TABLE_ENTRY && entry.parent->type == CatalogType::TABLE_ENTRY) {
		// If it's a table entry we have to check for possibly removing or adding user type dependencies
		auto &old_table = entry.parent->Cast<TableCatalogEntry>();
		auto &new_table = entry.Cast<TableCatalogEntry>();

		for (idx_t i = 0; i < new_table.GetColumns().LogicalColumnCount(); i++) {
			auto &new_column = new_table.GetColumnsMutable().GetColumnMutable(LogicalIndex(i));
			AdjustDependency(entry, old_table, new_column, false);
		}
		for (idx_t i = 0; i < old_table.GetColumns().LogicalColumnCount(); i++) {
			auto &old_column = old_table.GetColumnsMutable().GetColumnMutable(LogicalIndex(i));
			AdjustDependency(entry, new_table, old_column, true);
		}
	}
}

void CatalogSet::Undo(CatalogEntry &entry) {
	lock_guard<mutex> write_lock(catalog.GetWriteLock());
	lock_guard<mutex> lock(catalog_lock);

	// entry has to be restored
	// and entry->parent has to be removed ("rolled back")

	// i.e. we have to place (entry) as (entry->parent) again
	auto &to_be_removed_node = *entry.parent;

	AdjustTableDependencies(entry);

	if (!to_be_removed_node.deleted) {
		// delete the entry from the dependency manager as well
		auto &dependency_manager = catalog.GetDependencyManager();
		dependency_manager.EraseObject(to_be_removed_node);
	}
	if (!StringUtil::CIEquals(entry.name, to_be_removed_node.name)) {
		// rename: clean up the new name when the rename is rolled back
		auto removed_entry = mapping.find(to_be_removed_node.name);
		if (removed_entry->second->child) {
			removed_entry->second->child->parent = nullptr;
			mapping[to_be_removed_node.name] = std::move(removed_entry->second->child);
		} else {
			mapping.erase(removed_entry);
		}
	}
	if (to_be_removed_node.parent) {
		// if the to be removed node has a parent, set the child pointer to the
		// to be restored node
		to_be_removed_node.parent->child = std::move(to_be_removed_node.child);
		entry.parent = to_be_removed_node.parent;
	} else {
		// otherwise we need to update the base entry tables
		auto &name = entry.name;
		to_be_removed_node.child->SetAsRoot();
		mapping[name]->index.GetEntry() = std::move(to_be_removed_node.child);
		entry.parent = nullptr;
	}

	// restore the name if it was deleted
	auto restored_entry = mapping.find(entry.name);
	if (restored_entry->second->deleted || entry.type == CatalogType::INVALID) {
		if (restored_entry->second->child) {
			restored_entry->second->child->parent = nullptr;
			mapping[entry.name] = std::move(restored_entry->second->child);
		} else {
			mapping.erase(restored_entry);
		}
	}
	// we mark the catalog as being modified, since this action can lead to e.g. tables being dropped
	catalog.ModifyCatalog();
}

void CatalogSet::CreateDefaultEntries(CatalogTransaction transaction, unique_lock<mutex> &lock) {
	if (!defaults || defaults->created_all_entries || !transaction.context) {
		return;
	}
	// this catalog set has a default set defined:
	auto default_entries = defaults->GetDefaultEntries();
	for (auto &default_entry : default_entries) {
		auto map_entry = mapping.find(default_entry);
		if (map_entry == mapping.end()) {
			// we unlock during the CreateEntry, since it might reference other catalog sets...
			// specifically for views this can happen since the view will be bound
			lock.unlock();
			auto entry = defaults->CreateDefaultEntry(*transaction.context, default_entry);
			if (!entry) {
				throw InternalException("Failed to create default entry for %s", default_entry);
			}

			lock.lock();
			CreateEntryInternal(transaction, std::move(entry));
		}
	}
	defaults->created_all_entries = true;
}

void CatalogSet::Scan(CatalogTransaction transaction, const std::function<void(CatalogEntry &)> &callback) {
	// lock the catalog set
	unique_lock<mutex> lock(catalog_lock);
	CreateDefaultEntries(transaction, lock);

	for (auto &kv : entries) {
		auto &entry = *kv.second.entry.get();
		auto &entry_for_transaction = GetEntryForTransaction(transaction, entry);
		if (!entry_for_transaction.deleted) {
			callback(entry_for_transaction);
		}
	}
}

void CatalogSet::Scan(ClientContext &context, const std::function<void(CatalogEntry &)> &callback) {
	Scan(catalog.GetCatalogTransaction(context), callback);
}

void CatalogSet::Scan(const std::function<void(CatalogEntry &)> &callback) {
	// lock the catalog set
	lock_guard<mutex> lock(catalog_lock);
	for (auto &kv : entries) {
		auto entry = kv.second.entry.get();
		auto &commited_entry = GetCommittedEntry(*entry);
		if (!commited_entry.deleted) {
			callback(commited_entry);
		}
	}
}

void CatalogSet::Verify(Catalog &catalog_p) {
	D_ASSERT(&catalog_p == &catalog);
	vector<reference<CatalogEntry>> entries;
	Scan([&](CatalogEntry &entry) { entries.push_back(entry); });
	for (auto &entry : entries) {
		entry.get().Verify(catalog_p);
	}
}

} // namespace duckdb





namespace duckdb {

CatalogTransaction::CatalogTransaction(Catalog &catalog, ClientContext &context) {
	auto &transaction = Transaction::Get(context, catalog);
	this->db = &DatabaseInstance::GetDatabase(context);
	if (!transaction.IsDuckTransaction()) {
		this->transaction_id = transaction_t(-1);
		this->start_time = transaction_t(-1);
	} else {
		auto &dtransaction = transaction.Cast<DuckTransaction>();
		this->transaction_id = dtransaction.transaction_id;
		this->start_time = dtransaction.start_time;
	}
	this->transaction = &transaction;
	this->context = &context;
}

CatalogTransaction::CatalogTransaction(DatabaseInstance &db, transaction_t transaction_id_p, transaction_t start_time_p)
    : db(&db), context(nullptr), transaction(nullptr), transaction_id(transaction_id_p), start_time(start_time_p) {
}

ClientContext &CatalogTransaction::GetContext() {
	if (!context) {
		throw InternalException("Attempting to get a context in a CatalogTransaction without a context");
	}
	return *context;
}

CatalogTransaction CatalogTransaction::GetSystemTransaction(DatabaseInstance &db) {
	return CatalogTransaction(db, 1, 1);
}

} // namespace duckdb









namespace duckdb {

static DefaultMacro internal_macros[] = {
	{DEFAULT_SCHEMA, "current_role", {nullptr}, "'duckdb'"},                       // user name of current execution context
	{DEFAULT_SCHEMA, "current_user", {nullptr}, "'duckdb'"},                       // user name of current execution context
	{DEFAULT_SCHEMA, "current_catalog", {nullptr}, "current_database()"},          // name of current database (called "catalog" in the SQL standard)
	{DEFAULT_SCHEMA, "user", {nullptr}, "current_user"},                           // equivalent to current_user
	{DEFAULT_SCHEMA, "session_user", {nullptr}, "'duckdb'"},                       // session user name
	{"pg_catalog", "inet_client_addr", {nullptr}, "NULL"},                       // address of the remote connection
	{"pg_catalog", "inet_client_port", {nullptr}, "NULL"},                       // port of the remote connection
	{"pg_catalog", "inet_server_addr", {nullptr}, "NULL"},                       // address of the local connection
	{"pg_catalog", "inet_server_port", {nullptr}, "NULL"},                       // port of the local connection
	{"pg_catalog", "pg_my_temp_schema", {nullptr}, "0"},                         // OID of session's temporary schema, or 0 if none
	{"pg_catalog", "pg_is_other_temp_schema", {"schema_id", nullptr}, "false"},  // is schema another session's temporary schema?

	{"pg_catalog", "pg_conf_load_time", {nullptr}, "current_timestamp"},         // configuration load time
	{"pg_catalog", "pg_postmaster_start_time", {nullptr}, "current_timestamp"},  // server start time

	{"pg_catalog", "pg_typeof", {"expression", nullptr}, "lower(typeof(expression))"},  // get the data type of any value

	// privilege functions
	// {"has_any_column_privilege", {"user", "table", "privilege", nullptr}, "true"},  //boolean  //does user have privilege for any column of table
	{"pg_catalog", "has_any_column_privilege", {"table", "privilege", nullptr}, "true"},  //boolean  //does current user have privilege for any column of table
	// {"has_column_privilege", {"user", "table", "column", "privilege", nullptr}, "true"},  //boolean  //does user have privilege for column
	{"pg_catalog", "has_column_privilege", {"table", "column", "privilege", nullptr}, "true"},  //boolean  //does current user have privilege for column
	// {"has_database_privilege", {"user", "database", "privilege", nullptr}, "true"},  //boolean  //does user have privilege for database
	{"pg_catalog", "has_database_privilege", {"database", "privilege", nullptr}, "true"},  //boolean  //does current user have privilege for database
	// {"has_foreign_data_wrapper_privilege", {"user", "fdw", "privilege", nullptr}, "true"},  //boolean  //does user have privilege for foreign-data wrapper
	{"pg_catalog", "has_foreign_data_wrapper_privilege", {"fdw", "privilege", nullptr}, "true"},  //boolean  //does current user have privilege for foreign-data wrapper
	// {"has_function_privilege", {"user", "function", "privilege", nullptr}, "true"},  //boolean  //does user have privilege for function
	{"pg_catalog", "has_function_privilege", {"function", "privilege", nullptr}, "true"},  //boolean  //does current user have privilege for function
	// {"has_language_privilege", {"user", "language", "privilege", nullptr}, "true"},  //boolean  //does user have privilege for language
	{"pg_catalog", "has_language_privilege", {"language", "privilege", nullptr}, "true"},  //boolean  //does current user have privilege for language
	// {"has_schema_privilege", {"user", "schema, privilege", nullptr}, "true"},  //boolean  //does user have privilege for schema
	{"pg_catalog", "has_schema_privilege", {"schema", "privilege", nullptr}, "true"},  //boolean  //does current user have privilege for schema
	// {"has_sequence_privilege", {"user", "sequence", "privilege", nullptr}, "true"},  //boolean  //does user have privilege for sequence
	{"pg_catalog", "has_sequence_privilege", {"sequence", "privilege", nullptr}, "true"},  //boolean  //does current user have privilege for sequence
	// {"has_server_privilege", {"user", "server", "privilege", nullptr}, "true"},  //boolean  //does user have privilege for foreign server
	{"pg_catalog", "has_server_privilege", {"server", "privilege", nullptr}, "true"},  //boolean  //does current user have privilege for foreign server
	// {"has_table_privilege", {"user", "table", "privilege", nullptr}, "true"},  //boolean  //does user have privilege for table
	{"pg_catalog", "has_table_privilege", {"table", "privilege", nullptr}, "true"},  //boolean  //does current user have privilege for table
	// {"has_tablespace_privilege", {"user", "tablespace", "privilege", nullptr}, "true"},  //boolean  //does user have privilege for tablespace
	{"pg_catalog", "has_tablespace_privilege", {"tablespace", "privilege", nullptr}, "true"},  //boolean  //does current user have privilege for tablespace

	// various postgres system functions
	{"pg_catalog", "pg_get_viewdef", {"oid", nullptr}, "(select sql from duckdb_views() v where v.view_oid=oid)"},
	{"pg_catalog", "pg_get_constraintdef", {"constraint_oid", "pretty_bool", nullptr}, "(select constraint_text from duckdb_constraints() d_constraint where d_constraint.table_oid=constraint_oid//1000000 and d_constraint.constraint_index=constraint_oid%1000000)"},
	{"pg_catalog", "pg_get_expr", {"pg_node_tree", "relation_oid", nullptr}, "pg_node_tree"},
	{"pg_catalog", "format_pg_type", {"type_name", nullptr}, "case when logical_type='FLOAT' then 'real' when logical_type='DOUBLE' then 'double precision' when logical_type='DECIMAL' then 'numeric' when logical_type='ENUM' then lower(type_name) when logical_type='VARCHAR' then 'character varying' when logical_type='BLOB' then 'bytea' when logical_type='TIMESTAMP' then 'timestamp without time zone' when logical_type='TIME' then 'time without time zone' else lower(logical_type) end"},
	{"pg_catalog", "format_type", {"type_oid", "typemod", nullptr}, "(select format_pg_type(type_name) from duckdb_types() t where t.type_oid=type_oid) || case when typemod>0 then concat('(', typemod//1000, ',', typemod%1000, ')') else '' end"},

	{"pg_catalog", "pg_has_role", {"user", "role", "privilege", nullptr}, "true"},  //boolean  //does user have privilege for role
	{"pg_catalog", "pg_has_role", {"role", "privilege", nullptr}, "true"},  //boolean  //does current user have privilege for role

	{"pg_catalog", "col_description", {"table_oid", "column_number", nullptr}, "NULL"},   // get comment for a table column
	{"pg_catalog", "obj_description", {"object_oid", "catalog_name", nullptr}, "NULL"},   // get comment for a database object
	{"pg_catalog", "shobj_description", {"object_oid", "catalog_name", nullptr}, "NULL"}, // get comment for a shared database object

	// visibility functions
	{"pg_catalog", "pg_collation_is_visible", {"collation_oid", nullptr}, "true"},
	{"pg_catalog", "pg_conversion_is_visible", {"conversion_oid", nullptr}, "true"},
	{"pg_catalog", "pg_function_is_visible", {"function_oid", nullptr}, "true"},
	{"pg_catalog", "pg_opclass_is_visible", {"opclass_oid", nullptr}, "true"},
	{"pg_catalog", "pg_operator_is_visible", {"operator_oid", nullptr}, "true"},
	{"pg_catalog", "pg_opfamily_is_visible", {"opclass_oid", nullptr}, "true"},
	{"pg_catalog", "pg_table_is_visible", {"table_oid", nullptr}, "true"},
	{"pg_catalog", "pg_ts_config_is_visible", {"config_oid", nullptr}, "true"},
	{"pg_catalog", "pg_ts_dict_is_visible", {"dict_oid", nullptr}, "true"},
	{"pg_catalog", "pg_ts_parser_is_visible", {"parser_oid", nullptr}, "true"},
	{"pg_catalog", "pg_ts_template_is_visible", {"template_oid", nullptr}, "true"},
	{"pg_catalog", "pg_type_is_visible", {"type_oid", nullptr}, "true"},

	{"pg_catalog", "pg_size_pretty", {"bytes", nullptr}, "format_bytes(bytes)"},

	{DEFAULT_SCHEMA, "round_even", {"x", "n", nullptr}, "CASE ((abs(x) * power(10, n+1)) % 10) WHEN 5 THEN round(x/2, n) * 2 ELSE round(x, n) END"},
	{DEFAULT_SCHEMA, "roundbankers", {"x", "n", nullptr}, "round_even(x, n)"},
	{DEFAULT_SCHEMA, "nullif", {"a", "b", nullptr}, "CASE WHEN a=b THEN NULL ELSE a END"},
	{DEFAULT_SCHEMA, "list_append", {"l", "e", nullptr}, "list_concat(l, list_value(e))"},
	{DEFAULT_SCHEMA, "array_append", {"arr", "el", nullptr}, "list_append(arr, el)"},
	{DEFAULT_SCHEMA, "list_prepend", {"e", "l", nullptr}, "list_concat(list_value(e), l)"},
	{DEFAULT_SCHEMA, "array_prepend", {"el", "arr", nullptr}, "list_prepend(el, arr)"},
	{DEFAULT_SCHEMA, "array_pop_back", {"arr", nullptr}, "arr[:LEN(arr)-1]"},
	{DEFAULT_SCHEMA, "array_pop_front", {"arr", nullptr}, "arr[2:]"},
	{DEFAULT_SCHEMA, "array_push_back", {"arr", "e", nullptr}, "list_concat(arr, list_value(e))"},
	{DEFAULT_SCHEMA, "array_push_front", {"arr", "e", nullptr}, "list_concat(list_value(e), arr)"},
	{DEFAULT_SCHEMA, "array_to_string", {"arr", "sep", nullptr}, "list_aggr(arr, 'string_agg', sep)"},
	{DEFAULT_SCHEMA, "generate_subscripts", {"arr", "dim", nullptr}, "unnest(generate_series(1, array_length(arr, dim)))"},
	{DEFAULT_SCHEMA, "fdiv", {"x", "y", nullptr}, "floor(x/y)"},
	{DEFAULT_SCHEMA, "fmod", {"x", "y", nullptr}, "(x-y*floor(x/y))"},
	{DEFAULT_SCHEMA, "count_if", {"l", nullptr}, "sum(if(l, 1, 0))"},
	{DEFAULT_SCHEMA, "split_part", {"string", "delimiter", "position", nullptr}, "coalesce(string_split(string, delimiter)[position],'')"},

	// algebraic list aggregates
	{DEFAULT_SCHEMA, "list_avg", {"l", nullptr}, "list_aggr(l, 'avg')"},
	{DEFAULT_SCHEMA, "list_var_samp", {"l", nullptr}, "list_aggr(l, 'var_samp')"},
	{DEFAULT_SCHEMA, "list_var_pop", {"l", nullptr}, "list_aggr(l, 'var_pop')"},
	{DEFAULT_SCHEMA, "list_stddev_pop", {"l", nullptr}, "list_aggr(l, 'stddev_pop')"},
	{DEFAULT_SCHEMA, "list_stddev_samp", {"l", nullptr}, "list_aggr(l, 'stddev_samp')"},
	{DEFAULT_SCHEMA, "list_sem", {"l", nullptr}, "list_aggr(l, 'sem')"},

	// distributive list aggregates
	{DEFAULT_SCHEMA, "list_approx_count_distinct", {"l", nullptr}, "list_aggr(l, 'approx_count_distinct')"},
	{DEFAULT_SCHEMA, "list_bit_xor", {"l", nullptr}, "list_aggr(l, 'bit_xor')"},
	{DEFAULT_SCHEMA, "list_bit_or", {"l", nullptr}, "list_aggr(l, 'bit_or')"},
	{DEFAULT_SCHEMA, "list_bit_and", {"l", nullptr}, "list_aggr(l, 'bit_and')"},
	{DEFAULT_SCHEMA, "list_bool_and", {"l", nullptr}, "list_aggr(l, 'bool_and')"},
	{DEFAULT_SCHEMA, "list_bool_or", {"l", nullptr}, "list_aggr(l, 'bool_or')"},
	{DEFAULT_SCHEMA, "list_count", {"l", nullptr}, "list_aggr(l, 'count')"},
	{DEFAULT_SCHEMA, "list_entropy", {"l", nullptr}, "list_aggr(l, 'entropy')"},
	{DEFAULT_SCHEMA, "list_last", {"l", nullptr}, "list_aggr(l, 'last')"},
	{DEFAULT_SCHEMA, "list_first", {"l", nullptr}, "list_aggr(l, 'first')"},
	{DEFAULT_SCHEMA, "list_any_value", {"l", nullptr}, "list_aggr(l, 'any_value')"},
	{DEFAULT_SCHEMA, "list_kurtosis", {"l", nullptr}, "list_aggr(l, 'kurtosis')"},
	{DEFAULT_SCHEMA, "list_min", {"l", nullptr}, "list_aggr(l, 'min')"},
	{DEFAULT_SCHEMA, "list_max", {"l", nullptr}, "list_aggr(l, 'max')"},
	{DEFAULT_SCHEMA, "list_product", {"l", nullptr}, "list_aggr(l, 'product')"},
	{DEFAULT_SCHEMA, "list_skewness", {"l", nullptr}, "list_aggr(l, 'skewness')"},
	{DEFAULT_SCHEMA, "list_sum", {"l", nullptr}, "list_aggr(l, 'sum')"},
	{DEFAULT_SCHEMA, "list_string_agg", {"l", nullptr}, "list_aggr(l, 'string_agg')"},

	// holistic list aggregates
	{DEFAULT_SCHEMA, "list_mode", {"l", nullptr}, "list_aggr(l, 'mode')"},
	{DEFAULT_SCHEMA, "list_median", {"l", nullptr}, "list_aggr(l, 'median')"},
	{DEFAULT_SCHEMA, "list_mad", {"l", nullptr}, "list_aggr(l, 'mad')"},

	// nested list aggregates
	{DEFAULT_SCHEMA, "list_histogram", {"l", nullptr}, "list_aggr(l, 'histogram')"},

	// date functions
	{DEFAULT_SCHEMA, "date_add", {"date", "interval", nullptr}, "date + interval"},

	{nullptr, nullptr, {nullptr}, nullptr}
	};

unique_ptr<CreateMacroInfo> DefaultFunctionGenerator::CreateInternalTableMacroInfo(DefaultMacro &default_macro, unique_ptr<MacroFunction> function) {
	for (idx_t param_idx = 0; default_macro.parameters[param_idx] != nullptr; param_idx++) {
		function->parameters.push_back(
		    make_uniq<ColumnRefExpression>(default_macro.parameters[param_idx]));
	}

	auto bind_info = make_uniq<CreateMacroInfo>();
	bind_info->schema = default_macro.schema;
	bind_info->name = default_macro.name;
	bind_info->temporary = true;
	bind_info->internal = true;
	bind_info->type = function->type == MacroType::TABLE_MACRO ? CatalogType::TABLE_MACRO_ENTRY : CatalogType::MACRO_ENTRY;
	bind_info->function = std::move(function);
	return bind_info;

}

unique_ptr<CreateMacroInfo> DefaultFunctionGenerator::CreateInternalMacroInfo(DefaultMacro &default_macro) {
	// parse the expression
	auto expressions = Parser::ParseExpressionList(default_macro.macro);
	D_ASSERT(expressions.size() == 1);

	auto result = make_uniq<ScalarMacroFunction>(std::move(expressions[0]));
	return CreateInternalTableMacroInfo(default_macro, std::move(result));
}

unique_ptr<CreateMacroInfo> DefaultFunctionGenerator::CreateInternalTableMacroInfo(DefaultMacro &default_macro) {
	Parser parser;
	parser.ParseQuery(default_macro.macro);
	D_ASSERT(parser.statements.size() == 1);
	D_ASSERT(parser.statements[0]->type == StatementType::SELECT_STATEMENT);

	auto &select = parser.statements[0]->Cast<SelectStatement>();
	auto result = make_uniq<TableMacroFunction>(std::move(select.node));
	return CreateInternalTableMacroInfo(default_macro, std::move(result));
}

static unique_ptr<CreateFunctionInfo> GetDefaultFunction(const string &input_schema, const string &input_name) {
	auto schema = StringUtil::Lower(input_schema);
	auto name = StringUtil::Lower(input_name);
	for (idx_t index = 0; internal_macros[index].name != nullptr; index++) {
		if (internal_macros[index].schema == schema && internal_macros[index].name == name) {
			return DefaultFunctionGenerator::CreateInternalMacroInfo(internal_macros[index]);
		}
	}
	return nullptr;
}

DefaultFunctionGenerator::DefaultFunctionGenerator(Catalog &catalog, SchemaCatalogEntry &schema)
    : DefaultGenerator(catalog), schema(schema) {
}

unique_ptr<CatalogEntry> DefaultFunctionGenerator::CreateDefaultEntry(ClientContext &context,
                                                                      const string &entry_name) {
	auto info = GetDefaultFunction(schema.name, entry_name);
	if (info) {
		return make_uniq_base<CatalogEntry, ScalarMacroCatalogEntry>(catalog, schema, info->Cast<CreateMacroInfo>());
	}
	return nullptr;
}

vector<string> DefaultFunctionGenerator::GetDefaultEntries() {
	vector<string> result;
	for (idx_t index = 0; internal_macros[index].name != nullptr; index++) {
		if (internal_macros[index].schema == schema.name) {
			result.emplace_back(internal_macros[index].name);
		}
	}
	return result;
}

} // namespace duckdb




namespace duckdb {

struct DefaultSchema {
	const char *name;
};

static DefaultSchema internal_schemas[] = {{"information_schema"}, {"pg_catalog"}, {nullptr}};

static bool GetDefaultSchema(const string &input_schema) {
	auto schema = StringUtil::Lower(input_schema);
	for (idx_t index = 0; internal_schemas[index].name != nullptr; index++) {
		if (internal_schemas[index].name == schema) {
			return true;
		}
	}
	return false;
}

DefaultSchemaGenerator::DefaultSchemaGenerator(Catalog &catalog) : DefaultGenerator(catalog) {
}

unique_ptr<CatalogEntry> DefaultSchemaGenerator::CreateDefaultEntry(ClientContext &context, const string &entry_name) {
	if (GetDefaultSchema(entry_name)) {
		return make_uniq_base<CatalogEntry, DuckSchemaEntry>(catalog, StringUtil::Lower(entry_name), true);
	}
	return nullptr;
}

vector<string> DefaultSchemaGenerator::GetDefaultEntries() {
	vector<string> result;
	for (idx_t index = 0; internal_schemas[index].name != nullptr; index++) {
		result.emplace_back(internal_schemas[index].name);
	}
	return result;
}

} // namespace duckdb







namespace duckdb {

struct DefaultType {
	const char *name;
	LogicalTypeId type;
};

static DefaultType internal_types[] = {{"int", LogicalTypeId::INTEGER},
                                       {"int4", LogicalTypeId::INTEGER},
                                       {"signed", LogicalTypeId::INTEGER},
                                       {"integer", LogicalTypeId::INTEGER},
                                       {"integral", LogicalTypeId::INTEGER},
                                       {"int32", LogicalTypeId::INTEGER},
                                       {"varchar", LogicalTypeId::VARCHAR},
                                       {"bpchar", LogicalTypeId::VARCHAR},
                                       {"text", LogicalTypeId::VARCHAR},
                                       {"string", LogicalTypeId::VARCHAR},
                                       {"char", LogicalTypeId::VARCHAR},
                                       {"nvarchar", LogicalTypeId::VARCHAR},
                                       {"bytea", LogicalTypeId::BLOB},
                                       {"blob", LogicalTypeId::BLOB},
                                       {"varbinary", LogicalTypeId::BLOB},
                                       {"binary", LogicalTypeId::BLOB},
                                       {"bit", LogicalTypeId::BIT},
                                       {"bitstring", LogicalTypeId::BIT},
                                       {"int8", LogicalTypeId::BIGINT},
                                       {"bigint", LogicalTypeId::BIGINT},
                                       {"int64", LogicalTypeId::BIGINT},
                                       {"long", LogicalTypeId::BIGINT},
                                       {"oid", LogicalTypeId::BIGINT},
                                       {"int2", LogicalTypeId::SMALLINT},
                                       {"smallint", LogicalTypeId::SMALLINT},
                                       {"short", LogicalTypeId::SMALLINT},
                                       {"int16", LogicalTypeId::SMALLINT},
                                       {"timestamp", LogicalTypeId::TIMESTAMP},
                                       {"datetime", LogicalTypeId::TIMESTAMP},
                                       {"timestamp_us", LogicalTypeId::TIMESTAMP},
                                       {"timestamp_ms", LogicalTypeId::TIMESTAMP_MS},
                                       {"timestamp_ns", LogicalTypeId::TIMESTAMP_NS},
                                       {"timestamp_s", LogicalTypeId::TIMESTAMP_SEC},
                                       {"bool", LogicalTypeId::BOOLEAN},
                                       {"boolean", LogicalTypeId::BOOLEAN},
                                       {"logical", LogicalTypeId::BOOLEAN},
                                       {"decimal", LogicalTypeId::DECIMAL},
                                       {"dec", LogicalTypeId::DECIMAL},
                                       {"numeric", LogicalTypeId::DECIMAL},
                                       {"real", LogicalTypeId::FLOAT},
                                       {"float4", LogicalTypeId::FLOAT},
                                       {"float", LogicalTypeId::FLOAT},
                                       {"double", LogicalTypeId::DOUBLE},
                                       {"float8", LogicalTypeId::DOUBLE},
                                       {"tinyint", LogicalTypeId::TINYINT},
                                       {"int1", LogicalTypeId::TINYINT},
                                       {"date", LogicalTypeId::DATE},
                                       {"time", LogicalTypeId::TIME},
                                       {"interval", LogicalTypeId::INTERVAL},
                                       {"hugeint", LogicalTypeId::HUGEINT},
                                       {"int128", LogicalTypeId::HUGEINT},
                                       {"uuid", LogicalTypeId::UUID},
                                       {"guid", LogicalTypeId::UUID},
                                       {"struct", LogicalTypeId::STRUCT},
                                       {"row", LogicalTypeId::STRUCT},
                                       {"list", LogicalTypeId::LIST},
                                       {"map", LogicalTypeId::MAP},
                                       {"utinyint", LogicalTypeId::UTINYINT},
                                       {"uint8", LogicalTypeId::UTINYINT},
                                       {"usmallint", LogicalTypeId::USMALLINT},
                                       {"uint16", LogicalTypeId::USMALLINT},
                                       {"uinteger", LogicalTypeId::UINTEGER},
                                       {"uint32", LogicalTypeId::UINTEGER},
                                       {"ubigint", LogicalTypeId::UBIGINT},
                                       {"uint64", LogicalTypeId::UBIGINT},
                                       {"union", LogicalTypeId::UNION},
                                       {"timestamptz", LogicalTypeId::TIMESTAMP_TZ},
                                       {"timetz", LogicalTypeId::TIME_TZ},
                                       {"enum", LogicalTypeId::ENUM},
                                       {"null", LogicalTypeId::SQLNULL},
                                       {nullptr, LogicalTypeId::INVALID}};

LogicalTypeId DefaultTypeGenerator::GetDefaultType(const string &name) {
	auto lower_str = StringUtil::Lower(name);
	for (idx_t index = 0; internal_types[index].name != nullptr; index++) {
		if (internal_types[index].name == lower_str) {
			return internal_types[index].type;
		}
	}
	return LogicalTypeId::INVALID;
}

DefaultTypeGenerator::DefaultTypeGenerator(Catalog &catalog, SchemaCatalogEntry &schema)
    : DefaultGenerator(catalog), schema(schema) {
}

unique_ptr<CatalogEntry> DefaultTypeGenerator::CreateDefaultEntry(ClientContext &context, const string &entry_name) {
	if (schema.name != DEFAULT_SCHEMA) {
		return nullptr;
	}
	auto type_id = GetDefaultType(entry_name);
	if (type_id == LogicalTypeId::INVALID) {
		return nullptr;
	}
	CreateTypeInfo info;
	info.name = entry_name;
	info.type = LogicalType(type_id);
	info.internal = true;
	info.temporary = true;
	return make_uniq_base<CatalogEntry, TypeCatalogEntry>(catalog, schema, info);
}

vector<string> DefaultTypeGenerator::GetDefaultEntries() {
	vector<string> result;
	if (schema.name != DEFAULT_SCHEMA) {
		return result;
	}
	for (idx_t index = 0; internal_types[index].name != nullptr; index++) {
		result.emplace_back(internal_types[index].name);
	}
	return result;
}

} // namespace duckdb






namespace duckdb {

struct DefaultView {
	const char *schema;
	const char *name;
	const char *sql;
};

static DefaultView internal_views[] = {
    {DEFAULT_SCHEMA, "pragma_database_list", "SELECT database_oid AS seq, database_name AS name, path AS file FROM duckdb_databases() WHERE NOT internal ORDER BY 1"},
    {DEFAULT_SCHEMA, "sqlite_master", "select 'table' \"type\", table_name \"name\", table_name \"tbl_name\", 0 rootpage, sql from duckdb_tables union all select 'view' \"type\", view_name \"name\", view_name \"tbl_name\", 0 rootpage, sql from duckdb_views union all select 'index' \"type\", index_name \"name\", table_name \"tbl_name\", 0 rootpage, sql from duckdb_indexes;"},
    {DEFAULT_SCHEMA, "sqlite_schema", "SELECT * FROM sqlite_master"},
    {DEFAULT_SCHEMA, "sqlite_temp_master", "SELECT * FROM sqlite_master"},
    {DEFAULT_SCHEMA, "sqlite_temp_schema", "SELECT * FROM sqlite_master"},
    {DEFAULT_SCHEMA, "duckdb_constraints", "SELECT * FROM duckdb_constraints()"},
    {DEFAULT_SCHEMA, "duckdb_columns", "SELECT * FROM duckdb_columns() WHERE NOT internal"},
    {DEFAULT_SCHEMA, "duckdb_databases", "SELECT * FROM duckdb_databases() WHERE NOT internal"},
    {DEFAULT_SCHEMA, "duckdb_indexes", "SELECT * FROM duckdb_indexes()"},
    {DEFAULT_SCHEMA, "duckdb_schemas", "SELECT * FROM duckdb_schemas() WHERE NOT internal"},
    {DEFAULT_SCHEMA, "duckdb_tables", "SELECT * FROM duckdb_tables() WHERE NOT internal"},
    {DEFAULT_SCHEMA, "duckdb_types", "SELECT * FROM duckdb_types()"},
    {DEFAULT_SCHEMA, "duckdb_views", "SELECT * FROM duckdb_views() WHERE NOT internal"},
    {"pg_catalog", "pg_am", "SELECT 0 oid, 'art' amname, NULL amhandler, 'i' amtype"},
    {"pg_catalog", "pg_attribute", "SELECT table_oid attrelid, column_name attname, data_type_id atttypid, 0 attstattarget, NULL attlen, column_index attnum, 0 attndims, -1 attcacheoff, case when data_type ilike '%decimal%' then numeric_precision*1000+numeric_scale else -1 end atttypmod, false attbyval, NULL attstorage, NULL attalign, NOT is_nullable attnotnull, column_default IS NOT NULL atthasdef, false atthasmissing, '' attidentity, '' attgenerated, false attisdropped, true attislocal, 0 attinhcount, 0 attcollation, NULL attcompression, NULL attacl, NULL attoptions, NULL attfdwoptions, NULL attmissingval FROM duckdb_columns()"},
    {"pg_catalog", "pg_attrdef", "SELECT column_index oid, table_oid adrelid, column_index adnum, column_default adbin from duckdb_columns() where column_default is not null;"},
    {"pg_catalog", "pg_class", "SELECT table_oid oid, table_name relname, schema_oid relnamespace, 0 reltype, 0 reloftype, 0 relowner, 0 relam, 0 relfilenode, 0 reltablespace, 0 relpages, estimated_size::real reltuples, 0 relallvisible, 0 reltoastrelid, 0 reltoastidxid, index_count > 0 relhasindex, false relisshared, case when temporary then 't' else 'p' end relpersistence, 'r' relkind, column_count relnatts, check_constraint_count relchecks, false relhasoids, has_primary_key relhaspkey, false relhasrules, false relhastriggers, false relhassubclass, false relrowsecurity, true relispopulated, NULL relreplident, false relispartition, 0 relrewrite, 0 relfrozenxid, NULL relminmxid, NULL relacl, NULL reloptions, NULL relpartbound FROM duckdb_tables() UNION ALL SELECT view_oid oid, view_name relname, schema_oid relnamespace, 0 reltype, 0 reloftype, 0 relowner, 0 relam, 0 relfilenode, 0 reltablespace, 0 relpages, 0 reltuples, 0 relallvisible, 0 reltoastrelid, 0 reltoastidxid, false relhasindex, false relisshared, case when temporary then 't' else 'p' end relpersistence, 'v' relkind, column_count relnatts, 0 relchecks, false relhasoids, false relhaspkey, false relhasrules, false relhastriggers, false relhassubclass, false relrowsecurity, true relispopulated, NULL relreplident, false relispartition, 0 relrewrite, 0 relfrozenxid, NULL relminmxid, NULL relacl, NULL reloptions, NULL relpartbound FROM duckdb_views() UNION ALL SELECT sequence_oid oid, sequence_name relname, schema_oid relnamespace, 0 reltype, 0 reloftype, 0 relowner, 0 relam, 0 relfilenode, 0 reltablespace, 0 relpages, 0 reltuples, 0 relallvisible, 0 reltoastrelid, 0 reltoastidxid, false relhasindex, false relisshared, case when temporary then 't' else 'p' end relpersistence, 'S' relkind, 0 relnatts, 0 relchecks, false relhasoids, false relhaspkey, false relhasrules, false relhastriggers, false relhassubclass, false relrowsecurity, true relispopulated, NULL relreplident, false relispartition, 0 relrewrite, 0 relfrozenxid, NULL relminmxid, NULL relacl, NULL reloptions, NULL relpartbound FROM duckdb_sequences() UNION ALL SELECT index_oid oid, index_name relname, schema_oid relnamespace, 0 reltype, 0 reloftype, 0 relowner, 0 relam, 0 relfilenode, 0 reltablespace, 0 relpages, 0 reltuples, 0 relallvisible, 0 reltoastrelid, 0 reltoastidxid, false relhasindex, false relisshared, 't' relpersistence, 'i' relkind, NULL relnatts, 0 relchecks, false relhasoids, false relhaspkey, false relhasrules, false relhastriggers, false relhassubclass, false relrowsecurity, true relispopulated, NULL relreplident, false relispartition, 0 relrewrite, 0 relfrozenxid, NULL relminmxid, NULL relacl, NULL reloptions, NULL relpartbound FROM duckdb_indexes()"},
    {"pg_catalog", "pg_constraint", "SELECT table_oid*1000000+constraint_index oid, constraint_text conname, schema_oid connamespace, CASE constraint_type WHEN 'CHECK' then 'c' WHEN 'UNIQUE' then 'u' WHEN 'PRIMARY KEY' THEN 'p' WHEN 'FOREIGN KEY' THEN 'f' ELSE 'x' END contype, false condeferrable, false condeferred, true convalidated, table_oid conrelid, 0 contypid, 0 conindid, 0 conparentid, 0 confrelid, NULL confupdtype, NULL confdeltype, NULL confmatchtype, true conislocal, 0 coninhcount, false connoinherit, constraint_column_indexes conkey, NULL confkey, NULL conpfeqop, NULL conppeqop, NULL conffeqop, NULL conexclop, expression conbin FROM duckdb_constraints()"},
	{"pg_catalog", "pg_database", "SELECT database_oid oid, database_name datname FROM duckdb_databases()"},
    {"pg_catalog", "pg_depend", "SELECT * FROM duckdb_dependencies()"},
	{"pg_catalog", "pg_description", "SELECT NULL objoid, NULL classoid, NULL objsubid, NULL description WHERE 1=0"},
    {"pg_catalog", "pg_enum", "SELECT NULL oid, a.type_oid enumtypid, list_position(b.labels, a.elabel) enumsortorder, a.elabel enumlabel FROM (SELECT UNNEST(labels) elabel, type_oid FROM duckdb_types() WHERE logical_type='ENUM') a JOIN duckdb_types() b ON a.type_oid=b.type_oid;"},
    {"pg_catalog", "pg_index", "SELECT index_oid indexrelid, table_oid indrelid, 0 indnatts, 0 indnkeyatts, is_unique indisunique, is_primary indisprimary, false indisexclusion, true indimmediate, false indisclustered, true indisvalid, false indcheckxmin, true indisready, true indislive, false indisreplident, NULL::INT[] indkey, NULL::OID[] indcollation, NULL::OID[] indclass, NULL::INT[] indoption, expressions indexprs, NULL indpred FROM duckdb_indexes()"},
    {"pg_catalog", "pg_indexes", "SELECT schema_name schemaname, table_name tablename, index_name indexname, NULL \"tablespace\", sql indexdef FROM duckdb_indexes()"},
    {"pg_catalog", "pg_namespace", "SELECT oid, schema_name nspname, 0 nspowner, NULL nspacl FROM duckdb_schemas()"},
	{"pg_catalog", "pg_proc", "SELECT f.function_oid oid, function_name proname, s.oid pronamespace, varargs provariadic, function_type = 'aggregate' proisagg, function_type = 'table' proretset, return_type prorettype, parameter_types proargtypes, parameters proargnames FROM duckdb_functions() f LEFT JOIN duckdb_schemas() s USING (database_name, schema_name)"},
    {"pg_catalog", "pg_sequence", "SELECT sequence_oid seqrelid, 0 seqtypid, start_value seqstart, increment_by seqincrement, max_value seqmax, min_value seqmin, 0 seqcache, cycle seqcycle FROM duckdb_sequences()"},
	{"pg_catalog", "pg_sequences", "SELECT schema_name schemaname, sequence_name sequencename, 'duckdb' sequenceowner, 0 data_type, start_value, min_value, max_value, increment_by, cycle, 0 cache_size, last_value FROM duckdb_sequences()"},
	{"pg_catalog", "pg_settings", "SELECT name, value setting, description short_desc, CASE WHEN input_type = 'VARCHAR' THEN 'string' WHEN input_type = 'BOOLEAN' THEN 'bool' WHEN input_type IN ('BIGINT', 'UBIGINT') THEN 'integer' ELSE input_type END vartype FROM duckdb_settings()"},
    {"pg_catalog", "pg_tables", "SELECT schema_name schemaname, table_name tablename, 'duckdb' tableowner, NULL \"tablespace\", index_count > 0 hasindexes, false hasrules, false hastriggers FROM duckdb_tables()"},
    {"pg_catalog", "pg_tablespace", "SELECT 0 oid, 'pg_default' spcname, 0 spcowner, NULL spcacl, NULL spcoptions"},
    {"pg_catalog", "pg_type", "SELECT type_oid oid, format_pg_type(type_name) typname, schema_oid typnamespace, 0 typowner, type_size typlen, false typbyval, CASE WHEN logical_type='ENUM' THEN 'e' else 'b' end typtype, CASE WHEN type_category='NUMERIC' THEN 'N' WHEN type_category='STRING' THEN 'S' WHEN type_category='DATETIME' THEN 'D' WHEN type_category='BOOLEAN' THEN 'B' WHEN type_category='COMPOSITE' THEN 'C' WHEN type_category='USER' THEN 'U' ELSE 'X' END typcategory, false typispreferred, true typisdefined, NULL typdelim, NULL typrelid, NULL typsubscript, NULL typelem, NULL typarray, NULL typinput, NULL typoutput, NULL typreceive, NULL typsend, NULL typmodin, NULL typmodout, NULL typanalyze, 'd' typalign, 'p' typstorage, NULL typnotnull, NULL typbasetype, NULL typtypmod, NULL typndims, NULL typcollation, NULL typdefaultbin, NULL typdefault, NULL typacl FROM duckdb_types() WHERE type_size IS NOT NULL;"},
    {"pg_catalog", "pg_views", "SELECT schema_name schemaname, view_name viewname, 'duckdb' viewowner, sql definition FROM duckdb_views()"},
    {"information_schema", "columns", "SELECT database_name table_catalog, schema_name table_schema, table_name, column_name, column_index ordinal_position, column_default, CASE WHEN is_nullable THEN 'YES' ELSE 'NO' END is_nullable, data_type, character_maximum_length, NULL character_octet_length, numeric_precision, numeric_precision_radix, numeric_scale, NULL datetime_precision, NULL interval_type, NULL interval_precision, NULL character_set_catalog, NULL character_set_schema, NULL character_set_name, NULL collation_catalog, NULL collation_schema, NULL collation_name, NULL domain_catalog, NULL domain_schema, NULL domain_name, NULL udt_catalog, NULL udt_schema, NULL udt_name, NULL scope_catalog, NULL scope_schema, NULL scope_name, NULL maximum_cardinality, NULL dtd_identifier, NULL is_self_referencing, NULL is_identity, NULL identity_generation, NULL identity_start, NULL identity_increment, NULL identity_maximum, NULL identity_minimum, NULL identity_cycle, NULL is_generated, NULL generation_expression, NULL is_updatable FROM duckdb_columns;"},
    {"information_schema", "schemata", "SELECT database_name catalog_name, schema_name, 'duckdb' schema_owner, NULL default_character_set_catalog, NULL default_character_set_schema, NULL default_character_set_name, sql sql_path FROM duckdb_schemas()"},
    {"information_schema", "tables", "SELECT database_name table_catalog, schema_name table_schema, table_name, CASE WHEN temporary THEN 'LOCAL TEMPORARY' ELSE 'BASE TABLE' END table_type, NULL self_referencing_column_name, NULL reference_generation, NULL user_defined_type_catalog, NULL user_defined_type_schema, NULL user_defined_type_name, 'YES' is_insertable_into, 'NO' is_typed, CASE WHEN temporary THEN 'PRESERVE' ELSE NULL END commit_action FROM duckdb_tables() UNION ALL SELECT database_name table_catalog, schema_name table_schema, view_name table_name, 'VIEW' table_type, NULL self_referencing_column_name, NULL reference_generation, NULL user_defined_type_catalog, NULL user_defined_type_schema, NULL user_defined_type_name, 'NO' is_insertable_into, 'NO' is_typed, NULL commit_action FROM duckdb_views;"},
    {nullptr, nullptr, nullptr}};

static unique_ptr<CreateViewInfo> GetDefaultView(ClientContext &context, const string &input_schema, const string &input_name) {
	auto schema = StringUtil::Lower(input_schema);
	auto name = StringUtil::Lower(input_name);
	for (idx_t index = 0; internal_views[index].name != nullptr; index++) {
		if (internal_views[index].schema == schema && internal_views[index].name == name) {
			auto result = make_uniq<CreateViewInfo>();
			result->schema = schema;
			result->view_name = name;
			result->sql = internal_views[index].sql;
			result->temporary = true;
			result->internal = true;

			return CreateViewInfo::FromSelect(context, std::move(result));
		}
	}
	return nullptr;
}

DefaultViewGenerator::DefaultViewGenerator(Catalog &catalog, SchemaCatalogEntry &schema)
    : DefaultGenerator(catalog), schema(schema) {
}

unique_ptr<CatalogEntry> DefaultViewGenerator::CreateDefaultEntry(ClientContext &context, const string &entry_name) {
	auto info = GetDefaultView(context, schema.name, entry_name);
	if (info) {
		return make_uniq_base<CatalogEntry, ViewCatalogEntry>(catalog, schema, *info);
	}
	return nullptr;
}

vector<string> DefaultViewGenerator::GetDefaultEntries() {
	vector<string> result;
	for (idx_t index = 0; internal_views[index].name != nullptr; index++) {
		if (internal_views[index].schema == schema.name) {
			result.emplace_back(internal_views[index].name);
		}
	}
	return result;
}

} // namespace duckdb




namespace duckdb {

void DependencyList::AddDependency(CatalogEntry &entry) {
	if (entry.internal) {
		return;
	}
	set.insert(entry);
}

void DependencyList::VerifyDependencies(Catalog &catalog, const string &name) {
	for (auto &dep_entry : set) {
		auto &dep = dep_entry.get();
		if (&dep.ParentCatalog() != &catalog) {
			throw DependencyException(
			    "Error adding dependency for object \"%s\" - dependency \"%s\" is in catalog "
			    "\"%s\", which does not match the catalog \"%s\".\nCross catalog dependencies are not supported.",
			    name, dep.name, dep.ParentCatalog().GetName(), catalog.GetName());
		}
	}
}

} // namespace duckdb











namespace duckdb {

DependencyManager::DependencyManager(DuckCatalog &catalog) : catalog(catalog) {
}

void DependencyManager::AddObject(CatalogTransaction transaction, CatalogEntry &object, DependencyList &dependencies) {
	// check for each object in the sources if they were not deleted yet
	for (auto &dep : dependencies.set) {
		auto &dependency = dep.get();
		if (&dependency.ParentCatalog() != &object.ParentCatalog()) {
			throw DependencyException(
			    "Error adding dependency for object \"%s\" - dependency \"%s\" is in catalog "
			    "\"%s\", which does not match the catalog \"%s\".\nCross catalog dependencies are not supported.",
			    object.name, dependency.name, dependency.ParentCatalog().GetName(), object.ParentCatalog().GetName());
		}
		if (!dependency.set) {
			throw InternalException("Dependency has no set");
		}
		auto catalog_entry = dependency.set->GetEntryInternal(transaction, dependency.name, nullptr);
		if (!catalog_entry) {
			throw InternalException("Dependency has already been deleted?");
		}
	}
	// indexes do not require CASCADE to be dropped, they are simply always dropped along with the table
	auto dependency_type = object.type == CatalogType::INDEX_ENTRY ? DependencyType::DEPENDENCY_AUTOMATIC
	                                                               : DependencyType::DEPENDENCY_REGULAR;
	// add the object to the dependents_map of each object that it depends on
	for (auto &dependency : dependencies.set) {
		auto &set = dependents_map[dependency];
		set.insert(Dependency(object, dependency_type));
	}
	// create the dependents map for this object: it starts out empty
	dependents_map[object] = dependency_set_t();
	dependencies_map[object] = dependencies.set;
}

void DependencyManager::DropObject(CatalogTransaction transaction, CatalogEntry &object, bool cascade) {
	D_ASSERT(dependents_map.find(object) != dependents_map.end());

	// first check the objects that depend on this object
	auto &dependent_objects = dependents_map[object];
	for (auto &dep : dependent_objects) {
		// look up the entry in the catalog set
		auto &entry = dep.entry.get();
		auto &catalog_set = *entry.set;
		auto mapping_value = catalog_set.GetMapping(transaction, entry.name, true /* get_latest */);
		if (mapping_value == nullptr) {
			continue;
		}
		auto dependency_entry = catalog_set.GetEntryInternal(transaction, mapping_value->index);
		if (!dependency_entry) {
			// the dependent object was already deleted, no conflict
			continue;
		}
		// conflict: attempting to delete this object but the dependent object still exists
		if (cascade || dep.dependency_type == DependencyType::DEPENDENCY_AUTOMATIC ||
		    dep.dependency_type == DependencyType::DEPENDENCY_OWNS) {
			// cascade: drop the dependent object
			catalog_set.DropEntryInternal(transaction, mapping_value->index.Copy(), *dependency_entry, cascade);
		} else {
			// no cascade and there are objects that depend on this object: throw error
			throw DependencyException("Cannot drop entry \"%s\" because there are entries that "
			                          "depend on it. Use DROP...CASCADE to drop all dependents.",
			                          object.name);
		}
	}
}

void DependencyManager::AlterObject(CatalogTransaction transaction, CatalogEntry &old_obj, CatalogEntry &new_obj) {
	D_ASSERT(dependents_map.find(old_obj) != dependents_map.end());
	D_ASSERT(dependencies_map.find(old_obj) != dependencies_map.end());

	// first check the objects that depend on this object
	catalog_entry_vector_t owned_objects_to_add;
	auto &dependent_objects = dependents_map[old_obj];
	for (auto &dep : dependent_objects) {
		// look up the entry in the catalog set
		auto &entry = dep.entry.get();
		auto &catalog_set = *entry.set;
		auto dependency_entry = catalog_set.GetEntryInternal(transaction, entry.name, nullptr);
		if (!dependency_entry) {
			// the dependent object was already deleted, no conflict
			continue;
		}
		if (dep.dependency_type == DependencyType::DEPENDENCY_OWNS) {
			// the dependent object is owned by the current object
			owned_objects_to_add.push_back(dep.entry);
			continue;
		}
		// conflict: attempting to alter this object but the dependent object still exists
		// no cascade and there are objects that depend on this object: throw error
		throw DependencyException("Cannot alter entry \"%s\" because there are entries that "
		                          "depend on it.",
		                          old_obj.name);
	}
	// add the new object to the dependents_map of each object that it depends on
	auto &old_dependencies = dependencies_map[old_obj];
	catalog_entry_vector_t to_delete;
	for (auto &dep : old_dependencies) {
		auto &dependency = dep.get();
		if (dependency.type == CatalogType::TYPE_ENTRY) {
			auto &user_type = dependency.Cast<TypeCatalogEntry>();
			auto &table = new_obj.Cast<TableCatalogEntry>();
			bool deleted_dependency = true;
			for (auto &column : table.GetColumns().Logical()) {
				if (column.Type() == user_type.user_type) {
					deleted_dependency = false;
					break;
				}
			}
			if (deleted_dependency) {
				to_delete.push_back(dependency);
				continue;
			}
		}
		dependents_map[dependency].insert(new_obj);
	}
	for (auto &dep : to_delete) {
		auto &dependency = dep.get();
		old_dependencies.erase(dependency);
		dependents_map[dependency].erase(old_obj);
	}

	// We might have to add a type dependency
	catalog_entry_vector_t to_add;
	if (new_obj.type == CatalogType::TABLE_ENTRY) {
		auto &table = new_obj.Cast<TableCatalogEntry>();
		for (auto &column : table.GetColumns().Logical()) {
			auto user_type_catalog = EnumType::GetCatalog(column.Type());
			if (user_type_catalog) {
				to_add.push_back(*user_type_catalog);
			}
		}
	}
	// add the new object to the dependency manager
	dependents_map[new_obj] = dependency_set_t();
	dependencies_map[new_obj] = old_dependencies;

	for (auto &dependency : to_add) {
		dependencies_map[new_obj].insert(dependency);
		dependents_map[dependency].insert(new_obj);
	}

	for (auto &dependency : owned_objects_to_add) {
		dependents_map[new_obj].insert(Dependency(dependency, DependencyType::DEPENDENCY_OWNS));
		dependents_map[dependency].insert(Dependency(new_obj, DependencyType::DEPENDENCY_OWNED_BY));
		dependencies_map[new_obj].insert(dependency);
	}
}

void DependencyManager::EraseObject(CatalogEntry &object) {
	// obtain the writing lock
	EraseObjectInternal(object);
}

void DependencyManager::EraseObjectInternal(CatalogEntry &object) {
	if (dependents_map.find(object) == dependents_map.end()) {
		// dependencies already removed
		return;
	}
	D_ASSERT(dependents_map.find(object) != dependents_map.end());
	D_ASSERT(dependencies_map.find(object) != dependencies_map.end());
	// now for each of the dependencies, erase the entries from the dependents_map
	for (auto &dependency : dependencies_map[object]) {
		auto entry = dependents_map.find(dependency);
		if (entry != dependents_map.end()) {
			D_ASSERT(entry->second.find(object) != entry->second.end());
			entry->second.erase(object);
		}
	}
	// erase the dependents and dependencies for this object
	dependents_map.erase(object);
	dependencies_map.erase(object);
}

void DependencyManager::Scan(const std::function<void(CatalogEntry &, CatalogEntry &, DependencyType)> &callback) {
	lock_guard<mutex> write_lock(catalog.GetWriteLock());
	for (auto &entry : dependents_map) {
		for (auto &dependent : entry.second) {
			callback(entry.first, dependent.entry, dependent.dependency_type);
		}
	}
}

void DependencyManager::AddOwnership(CatalogTransaction transaction, CatalogEntry &owner, CatalogEntry &entry) {
	// lock the catalog for writing
	lock_guard<mutex> write_lock(catalog.GetWriteLock());

	// If the owner is already owned by something else, throw an error
	for (auto &dep : dependents_map[owner]) {
		if (dep.dependency_type == DependencyType::DEPENDENCY_OWNED_BY) {
			throw DependencyException(owner.name + " already owned by " + dep.entry.get().name);
		}
	}

	// If the entry is already owned, throw an error
	for (auto &dep : dependents_map[entry]) {
		// if the entry is already owned, throw error
		if (&dep.entry.get() != &owner) {
			throw DependencyException(entry.name + " already depends on " + dep.entry.get().name);
		}
		// if the entry owns the owner, throw error
		if (&dep.entry.get() == &owner && dep.dependency_type == DependencyType::DEPENDENCY_OWNS) {
			throw DependencyException(entry.name + " already owns " + owner.name +
			                          ". Cannot have circular dependencies");
		}
	}

	// Emplace guarantees that the same object cannot be inserted twice in the unordered_set
	// In the case AddOwnership is called twice, because of emplace, the object will not be repeated in the set.
	// We use an automatic dependency because if the Owner gets deleted, then the owned objects are also deleted
	dependents_map[owner].emplace(Dependency(entry, DependencyType::DEPENDENCY_OWNS));
	dependents_map[entry].emplace(Dependency(owner, DependencyType::DEPENDENCY_OWNED_BY));
	dependencies_map[owner].emplace(entry);
}

} // namespace duckdb









#ifndef DISABLE_CORE_FUNCTIONS_EXTENSION

#endif

namespace duckdb {

DuckCatalog::DuckCatalog(AttachedDatabase &db)
    : Catalog(db), dependency_manager(make_uniq<DependencyManager>(*this)),
      schemas(make_uniq<CatalogSet>(*this, make_uniq<DefaultSchemaGenerator>(*this))) {
}

DuckCatalog::~DuckCatalog() {
}

void DuckCatalog::Initialize(bool load_builtin) {
	// first initialize the base system catalogs
	// these are never written to the WAL
	// we start these at 1 because deleted entries default to 0
	auto data = CatalogTransaction::GetSystemTransaction(GetDatabase());

	// create the default schema
	CreateSchemaInfo info;
	info.schema = DEFAULT_SCHEMA;
	info.internal = true;
	CreateSchema(data, info);

	if (load_builtin) {
		// initialize default functions
		BuiltinFunctions builtin(data, *this);
		builtin.Initialize();

#ifndef DISABLE_CORE_FUNCTIONS_EXTENSION
		CoreFunctions::RegisterFunctions(*this, data);
#endif
	}

	Verify();
}

bool DuckCatalog::IsDuckCatalog() {
	return true;
}

//===--------------------------------------------------------------------===//
// Schema
//===--------------------------------------------------------------------===//
optional_ptr<CatalogEntry> DuckCatalog::CreateSchemaInternal(CatalogTransaction transaction, CreateSchemaInfo &info) {
	DependencyList dependencies;
	auto entry = make_uniq<DuckSchemaEntry>(*this, info.schema, info.internal);
	auto result = entry.get();
	if (!schemas->CreateEntry(transaction, info.schema, std::move(entry), dependencies)) {
		return nullptr;
	}
	return (CatalogEntry *)result;
}

optional_ptr<CatalogEntry> DuckCatalog::CreateSchema(CatalogTransaction transaction, CreateSchemaInfo &info) {
	D_ASSERT(!info.schema.empty());
	auto result = CreateSchemaInternal(transaction, info);
	if (!result) {
		switch (info.on_conflict) {
		case OnCreateConflict::ERROR_ON_CONFLICT:
			throw CatalogException("Schema with name %s already exists!", info.schema);
		case OnCreateConflict::REPLACE_ON_CONFLICT: {
			DropInfo drop_info;
			drop_info.type = CatalogType::SCHEMA_ENTRY;
			drop_info.catalog = info.catalog;
			drop_info.name = info.schema;
			DropSchema(transaction, drop_info);
			result = CreateSchemaInternal(transaction, info);
			if (!result) {
				throw InternalException("Failed to create schema entry in CREATE_OR_REPLACE");
			}
			break;
		}
		case OnCreateConflict::IGNORE_ON_CONFLICT:
			break;
		default:
			throw InternalException("Unsupported OnCreateConflict for CreateSchema");
		}
		return nullptr;
	}
	return result;
}

void DuckCatalog::DropSchema(CatalogTransaction transaction, DropInfo &info) {
	D_ASSERT(!info.name.empty());
	ModifyCatalog();
	if (!schemas->DropEntry(transaction, info.name, info.cascade)) {
		if (info.if_not_found == OnEntryNotFound::THROW_EXCEPTION) {
			throw CatalogException("Schema with name \"%s\" does not exist!", info.name);
		}
	}
}

void DuckCatalog::DropSchema(ClientContext &context, DropInfo &info) {
	DropSchema(GetCatalogTransaction(context), info);
}

void DuckCatalog::ScanSchemas(ClientContext &context, std::function<void(SchemaCatalogEntry &)> callback) {
	schemas->Scan(GetCatalogTransaction(context),
	              [&](CatalogEntry &entry) { callback(entry.Cast<SchemaCatalogEntry>()); });
}

void DuckCatalog::ScanSchemas(std::function<void(SchemaCatalogEntry &)> callback) {
	schemas->Scan([&](CatalogEntry &entry) { callback(entry.Cast<SchemaCatalogEntry>()); });
}

optional_ptr<SchemaCatalogEntry> DuckCatalog::GetSchema(CatalogTransaction transaction, const string &schema_name,
                                                        OnEntryNotFound if_not_found, QueryErrorContext error_context) {
	D_ASSERT(!schema_name.empty());
	auto entry = schemas->GetEntry(transaction, schema_name);
	if (!entry) {
		if (if_not_found == OnEntryNotFound::THROW_EXCEPTION) {
			throw CatalogException(error_context.FormatError("Schema with name %s does not exist!", schema_name));
		}
		return nullptr;
	}
	return &entry->Cast<SchemaCatalogEntry>();
}

DatabaseSize DuckCatalog::GetDatabaseSize(ClientContext &context) {
	return db.GetStorageManager().GetDatabaseSize();
}

bool DuckCatalog::InMemory() {
	return db.GetStorageManager().InMemory();
}

string DuckCatalog::GetDBPath() {
	return db.GetStorageManager().GetDBPath();
}

void DuckCatalog::Verify() {
#ifdef DEBUG
	schemas->Verify(*this);
#endif
}

} // namespace duckdb




namespace duckdb {

string SimilarCatalogEntry::GetQualifiedName(bool qualify_catalog, bool qualify_schema) const {
	D_ASSERT(Found());
	string result;
	if (qualify_catalog) {
		result += schema->catalog.GetName();
	}
	if (qualify_schema) {
		if (!result.empty()) {
			result += ".";
		}
		result += schema->name;
	}
	if (!result.empty()) {
		result += ".";
	}
	result += name;
	return result;
}

} // namespace duckdb











#include <string.h>
#include <stdlib.h>

// We gotta leak the symbols of the init function
duckdb_adbc::AdbcStatusCode duckdb_adbc_init(size_t count, struct duckdb_adbc::AdbcDriver *driver,
                                             struct duckdb_adbc::AdbcError *error) {
	if (!driver) {
		return ADBC_STATUS_INVALID_ARGUMENT;
	}

	driver->DatabaseNew = duckdb_adbc::DatabaseNew;
	driver->DatabaseSetOption = duckdb_adbc::DatabaseSetOption;
	driver->DatabaseInit = duckdb_adbc::DatabaseInit;
	driver->DatabaseRelease = duckdb_adbc::DatabaseRelease;
	driver->ConnectionNew = duckdb_adbc::ConnectionNew;
	driver->ConnectionSetOption = duckdb_adbc::ConnectionSetOption;
	driver->ConnectionInit = duckdb_adbc::ConnectionInit;
	driver->ConnectionRelease = duckdb_adbc::ConnectionRelease;
	driver->ConnectionGetTableTypes = duckdb_adbc::ConnectionGetTableTypes;
	driver->StatementNew = duckdb_adbc::StatementNew;
	driver->StatementRelease = duckdb_adbc::StatementRelease;
	//	driver->StatementBind = duckdb::adbc::StatementBind;
	driver->StatementBindStream = duckdb_adbc::StatementBindStream;
	driver->StatementExecuteQuery = duckdb_adbc::StatementExecuteQuery;
	driver->StatementPrepare = duckdb_adbc::StatementPrepare;
	driver->StatementSetOption = duckdb_adbc::StatementSetOption;
	driver->StatementSetSqlQuery = duckdb_adbc::StatementSetSqlQuery;
	driver->ConnectionGetObjects = duckdb_adbc::ConnectionGetObjects;
	return ADBC_STATUS_OK;
}

namespace duckdb_adbc {
#define CHECK_TRUE(p, e, m)                                                                                            \
	if (!(p)) {                                                                                                        \
		if (e) {                                                                                                       \
			e->message = strdup(m); /* TODO Set cleanup callback */                                                    \
		}                                                                                                              \
		return ADBC_STATUS_INVALID_ARGUMENT;                                                                           \
	}

#define CHECK_RES(res, e, m)                                                                                           \
	if (res != DuckDBSuccess) {                                                                                        \
		if (e) {                                                                                                       \
			e->message = strdup(m);                                                                                    \
		}                                                                                                              \
		return ADBC_STATUS_INTERNAL;                                                                                   \
	} else {                                                                                                           \
		return ADBC_STATUS_OK;                                                                                         \
	}

struct DuckDBAdbcDatabaseWrapper {
	//! The DuckDB Database Configuration
	::duckdb_config config;
	//! The DuckDB Database
	::duckdb_database database;
	//! Path of Disk-Based Database or :memory: database
	std::string path;
};

AdbcStatusCode DatabaseNew(struct AdbcDatabase *database, struct AdbcError *error) {
	CHECK_TRUE(database, error, "Missing database object");

	database->private_data = nullptr;
	// you can't malloc a struct with a non-trivial C++ constructor
	// and std::string has a non-trivial constructor. so we need
	// to use new and delete rather than malloc and free.
	auto wrapper = new DuckDBAdbcDatabaseWrapper;
	CHECK_TRUE(wrapper, error, "Allocation error");

	database->private_data = wrapper;
	auto res = duckdb_create_config(&wrapper->config);
	CHECK_RES(res, error, "Failed to allocate");
}

AdbcStatusCode DatabaseSetOption(struct AdbcDatabase *database, const char *key, const char *value,
                                 struct AdbcError *error) {
	CHECK_TRUE(database, error, "Missing database object");
	CHECK_TRUE(key, error, "Missing key");

	auto wrapper = (DuckDBAdbcDatabaseWrapper *)database->private_data;
	if (strcmp(key, "path") == 0) {
		wrapper->path = value;
		return ADBC_STATUS_OK;
	}
	auto res = duckdb_set_config(wrapper->config, key, value);

	CHECK_RES(res, error, "Failed to set configuration option");
}

AdbcStatusCode DatabaseInit(struct AdbcDatabase *database, struct AdbcError *error) {
	char *errormsg;
	// TODO can we set the database path via option, too? Does not look like it...
	auto wrapper = (DuckDBAdbcDatabaseWrapper *)database->private_data;
	auto res = duckdb_open_ext(wrapper->path.c_str(), &wrapper->database, wrapper->config, &errormsg);

	// TODO this leaks memory because errormsg is malloc-ed
	CHECK_RES(res, error, errormsg);
}

AdbcStatusCode DatabaseRelease(struct AdbcDatabase *database, struct AdbcError *error) {

	if (database && database->private_data) {
		auto wrapper = (DuckDBAdbcDatabaseWrapper *)database->private_data;

		duckdb_close(&wrapper->database);
		duckdb_destroy_config(&wrapper->config);
		delete wrapper;
		database->private_data = nullptr;
	}
	return ADBC_STATUS_OK;
}

AdbcStatusCode ConnectionNew(struct AdbcConnection *connection, struct AdbcError *error) {

	CHECK_TRUE(connection, error, "Missing connection object");
	connection->private_data = nullptr;
	return ADBC_STATUS_OK;
}

AdbcStatusCode ConnectionSetOption(struct AdbcConnection *connection, const char *key, const char *value,
                                   struct AdbcError *error) {
	// there are no connection-level options that need to be set before connecting
	return ADBC_STATUS_OK;
}

AdbcStatusCode ConnectionInit(struct AdbcConnection *connection, struct AdbcDatabase *database,
                              struct AdbcError *error) {
	CHECK_TRUE(database, error, "Missing database");
	CHECK_TRUE(database->private_data, error, "Invalid database");
	CHECK_TRUE(connection, error, "Missing connection");
	auto database_wrapper = (DuckDBAdbcDatabaseWrapper *)database->private_data;

	connection->private_data = nullptr;
	auto res = duckdb_connect(database_wrapper->database, (duckdb_connection *)&connection->private_data);
	CHECK_RES(res, error, "Failed to connect to Database");
}

AdbcStatusCode ConnectionRelease(struct AdbcConnection *connection, struct AdbcError *error) {
	if (connection && connection->private_data) {
		duckdb_disconnect((duckdb_connection *)&connection->private_data);
		connection->private_data = nullptr;
	}
	return ADBC_STATUS_OK;
}

// some stream callbacks

static int get_schema(struct ArrowArrayStream *stream, struct ArrowSchema *out) {
	if (!stream || !stream->private_data || !out) {
		return DuckDBError;
	}
	return duckdb_query_arrow_schema((duckdb_arrow)stream->private_data, (duckdb_arrow_schema *)&out);
}

static int get_next(struct ArrowArrayStream *stream, struct ArrowArray *out) {
	if (!stream || !stream->private_data || !out) {
		return DuckDBError;
	}
	out->release = nullptr;

	return duckdb_query_arrow_array((duckdb_arrow)stream->private_data, (duckdb_arrow_array *)&out);
}

void release(struct ArrowArrayStream *stream) {
	if (!stream || !stream->release) {
		return;
	}
	stream->release = nullptr;
	if (stream->private_data) {
		duckdb_destroy_arrow((duckdb_arrow *)&stream->private_data);
		stream->private_data = nullptr;
	}
}

const char *get_last_error(struct ArrowArrayStream *stream) {
	if (!stream) {
		return nullptr;
	}
	return nullptr;
	// return duckdb_query_arrow_error(stream);
}

// this is an evil hack, normally we would need a stream factory here, but its probably much easier if the adbc clients
// just hand over a stream

duckdb::unique_ptr<duckdb::ArrowArrayStreamWrapper>
stream_produce(uintptr_t factory_ptr,
               std::pair<std::unordered_map<idx_t, std::string>, std::vector<std::string>> &project_columns,
               duckdb::TableFilterSet *filters) {

	// TODO this will ignore any projections or filters but since we don't expose the scan it should be sort of fine
	auto res = duckdb::make_uniq<duckdb::ArrowArrayStreamWrapper>();
	res->arrow_array_stream = *(ArrowArrayStream *)factory_ptr;
	return res;
}

void stream_schema(uintptr_t factory_ptr, duckdb::ArrowSchemaWrapper &schema) {
	auto stream = (ArrowArrayStream *)factory_ptr;
	get_schema(stream, &schema.arrow_schema);
}

AdbcStatusCode Ingest(duckdb_connection connection, const char *table_name, struct ArrowArrayStream *input,
                      struct AdbcError *error) {

	CHECK_TRUE(connection, error, "Invalid connection");
	CHECK_TRUE(input, error, "Missing input arrow stream pointer");
	CHECK_TRUE(table_name, error, "Missing database object name");

	try {
		// TODO evil cast, do we need a way to do this from the C api?
		auto cconn = (duckdb::Connection *)connection;
		cconn
		    ->TableFunction("arrow_scan",
		                    {duckdb::Value::POINTER((uintptr_t)input),
		                     duckdb::Value::POINTER((uintptr_t)stream_produce),
		                     duckdb::Value::POINTER((uintptr_t)get_schema)}) // TODO make this a parameter somewhere
		    ->Create(table_name); // TODO this should probably be a temp table
		// After creating a table, the arrow array stream is released. Hence we must set it as released to avoid
		// double-releasing it
		input->release = nullptr;
	} catch (std::exception &ex) {
		if (error) {
			error->message = strdup(ex.what());
		}
		return ADBC_STATUS_INTERNAL;
	} catch (...) {
		return ADBC_STATUS_INTERNAL;
	}
	return ADBC_STATUS_OK;
}

struct DuckDBAdbcStatementWrapper {
	::duckdb_connection connection;
	::duckdb_arrow result;
	::duckdb_prepared_statement statement;
	char *ingestion_table_name;
	ArrowArrayStream *ingestion_stream;
};

AdbcStatusCode StatementNew(struct AdbcConnection *connection, struct AdbcStatement *statement,
                            struct AdbcError *error) {

	CHECK_TRUE(connection, error, "Missing connection object");
	CHECK_TRUE(connection->private_data, error, "Invalid connection object");
	CHECK_TRUE(statement, error, "Missing statement object");

	statement->private_data = nullptr;

	auto statement_wrapper = (DuckDBAdbcStatementWrapper *)malloc(sizeof(DuckDBAdbcStatementWrapper));
	CHECK_TRUE(statement_wrapper, error, "Allocation error");

	statement->private_data = statement_wrapper;
	statement_wrapper->connection = (duckdb_connection)connection->private_data;
	statement_wrapper->statement = nullptr;
	statement_wrapper->result = nullptr;
	statement_wrapper->ingestion_stream = nullptr;
	statement_wrapper->ingestion_table_name = nullptr;
	return ADBC_STATUS_OK;
}

AdbcStatusCode StatementRelease(struct AdbcStatement *statement, struct AdbcError *error) {

	if (statement && statement->private_data) {
		auto wrapper = (DuckDBAdbcStatementWrapper *)statement->private_data;
		if (wrapper->statement) {
			duckdb_destroy_prepare(&wrapper->statement);
			wrapper->statement = nullptr;
		}
		if (wrapper->result) {
			duckdb_destroy_arrow(&wrapper->result);
			wrapper->result = nullptr;
		}
		if (wrapper->ingestion_stream) {
			wrapper->ingestion_stream->release(wrapper->ingestion_stream);
			wrapper->ingestion_stream->release = nullptr;
			wrapper->ingestion_stream = nullptr;
		}
		if (wrapper->ingestion_table_name) {
			free(wrapper->ingestion_table_name);
			wrapper->ingestion_table_name = nullptr;
		}
		free(statement->private_data);
		statement->private_data = nullptr;
	}
	return ADBC_STATUS_OK;
}

AdbcStatusCode StatementExecuteQuery(struct AdbcStatement *statement, struct ArrowArrayStream *out,
                                     int64_t *rows_affected, struct AdbcError *error) {
	CHECK_TRUE(statement, error, "Missing statement object");
	CHECK_TRUE(statement->private_data, error, "Invalid statement object");
	auto wrapper = (DuckDBAdbcStatementWrapper *)statement->private_data;

	// TODO: Set affected rows, careful with early return
	if (rows_affected) {
		*rows_affected = 0;
	}

	if (wrapper->ingestion_stream && wrapper->ingestion_table_name) {
		auto stream = wrapper->ingestion_stream;
		wrapper->ingestion_stream = nullptr;
		return Ingest(wrapper->connection, wrapper->ingestion_table_name, stream, error);
	}

	auto res = duckdb_execute_prepared_arrow(wrapper->statement, &wrapper->result);
	CHECK_TRUE(res == DuckDBSuccess, error, duckdb_query_arrow_error(wrapper->result));

	if (out) {
		out->private_data = wrapper->result;
		out->get_schema = get_schema;
		out->get_next = get_next;
		out->release = release;
		out->get_last_error = get_last_error;

		// because we handed out the stream pointer its no longer our responsibility to destroy it in
		// AdbcStatementRelease, this is now done in release()
		wrapper->result = nullptr;
	}

	return ADBC_STATUS_OK;
}

// this is a nop for us
AdbcStatusCode StatementPrepare(struct AdbcStatement *statement, struct AdbcError *error) {
	CHECK_TRUE(statement, error, "Missing statement object");
	CHECK_TRUE(statement->private_data, error, "Invalid statement object");
	return ADBC_STATUS_OK;
}

AdbcStatusCode StatementSetSqlQuery(struct AdbcStatement *statement, const char *query, struct AdbcError *error) {
	CHECK_TRUE(statement, error, "Missing statement object");
	CHECK_TRUE(query, error, "Missing query");

	auto wrapper = (DuckDBAdbcStatementWrapper *)statement->private_data;
	auto res = duckdb_prepare(wrapper->connection, query, &wrapper->statement);

	CHECK_RES(res, error, duckdb_prepare_error(wrapper->statement));
}

AdbcStatusCode StatementBindStream(struct AdbcStatement *statement, struct ArrowArrayStream *values,
                                   struct AdbcError *error) {
	CHECK_TRUE(statement, error, "Missing statement object");
	CHECK_TRUE(values, error, "Missing stream object");
	auto wrapper = (DuckDBAdbcStatementWrapper *)statement->private_data;
	wrapper->ingestion_stream = values;
	return ADBC_STATUS_OK;
}

AdbcStatusCode StatementSetOption(struct AdbcStatement *statement, const char *key, const char *value,
                                  struct AdbcError *error) {
	CHECK_TRUE(statement, error, "Missing statement object");
	CHECK_TRUE(key, error, "Missing key object");
	auto wrapper = (DuckDBAdbcStatementWrapper *)statement->private_data;

	if (strcmp(key, ADBC_INGEST_OPTION_TARGET_TABLE) == 0) {
		wrapper->ingestion_table_name = strdup(value);
		return ADBC_STATUS_OK;
	}
	return ADBC_STATUS_INVALID_ARGUMENT;
}

static AdbcStatusCode QueryInternal(struct AdbcConnection *connection, struct ArrowArrayStream *out, const char *query,
                                    struct AdbcError *error) {
	AdbcStatusCode res;
	AdbcStatement statement;

	res = StatementNew(connection, &statement, error);
	CHECK_TRUE(!res, error, "unable to initialize statement");

	res = StatementSetSqlQuery(&statement, query, error);
	CHECK_TRUE(!res, error, "unable to initialize statement");

	res = StatementExecuteQuery(&statement, out, NULL, error);
	CHECK_TRUE(!res, error, "unable to execute statement");

	return ADBC_STATUS_OK;
}

AdbcStatusCode ConnectionGetObjects(struct AdbcConnection *connection, int depth, const char *catalog,
                                    const char *db_schema, const char *table_name, const char **table_type,
                                    const char *column_name, struct ArrowArrayStream *out, struct AdbcError *error) {
	CHECK_TRUE(catalog == nullptr || strcmp(catalog, "duckdb") == 0, error, "catalog must be NULL or 'duckdb'");
	CHECK_TRUE(table_type == nullptr, error, "table types parameter not yet supported");

	auto q = duckdb::StringUtil::Format(R"(
SELECT table_schema db_schema_name, LIST(table_schema_list) db_schema_tables FROM (
	SELECT table_schema, { table_name : table_name, table_columns : LIST({column_name : column_name, ordinal_position : ordinal_position + 1, remarks : ''})} table_schema_list FROM information_schema.columns WHERE table_schema LIKE '%s' AND table_name LIKE '%s' AND column_name LIKE '%s' GROUP BY table_schema, table_name
	) GROUP BY table_schema;
)",
	                                    db_schema ? db_schema : "%", table_name ? table_name : "%",
	                                    column_name ? column_name : "%");

	return QueryInternal(connection, out, q.c_str(), error);
}

//
// AdbcStatusCode ConnectionGetCatalogs(struct AdbcConnection *connection, struct AdbcStatement *statement,
//                                         struct AdbcError *error) {
//	const char *q = "SELECT 'duckdb' catalog_name";
//
//	return QueryInternal(connection, statement, q, error);
//}
//
// AdbcStatusCode ConnectionGetDbSchemas(struct AdbcConnection *connection, struct AdbcStatement *statement,
//                                          struct AdbcError *error) {
//	const char *q = "SELECT 'duckdb' catalog_name, schema_name db_schema_name FROM information_schema.schemata ORDER "
//	                "BY schema_name";
//	return QueryInternal(connection, statement, q, error);
//}
AdbcStatusCode ConnectionGetTableTypes(struct AdbcConnection *connection, struct ArrowArrayStream *out,
                                       struct AdbcError *error) {
	const char *q = "SELECT DISTINCT table_type FROM information_schema.tables ORDER BY table_type";
	return QueryInternal(connection, out, q, error);
}
//
// AdbcStatusCode ConnectionGetTables(struct AdbcConnection *connection, const char *catalog, const char *db_schema,
//                                       const char *table_name, const char **table_types,
//                                       struct AdbcStatement *statement, struct AdbcError *error) {
//
//	CHECK_TRUE(catalog == nullptr || strcmp(catalog, "duckdb") == 0, error, "catalog must be NULL or 'duckdb'");
//
//	// let's wait for https://github.com/lidavidm/arrow/issues/6
//	CHECK_TRUE(table_types == nullptr, error, "table types parameter not yet supported");
//	auto q = duckdb::StringUtil::Format(
//	    "SELECT 'duckdb' catalog_name, table_schema db_schema_name, table_name, table_type FROM "
//	    "information_schema.tables WHERE table_schema LIKE '%s' AND table_name LIKE '%s' ORDER BY table_schema, "
//	    "table_name",
//	    db_schema ? db_schema : "%", table_name ? table_name : "%");
//
//	return QueryInternal(connection, statement, q.c_str(), error);
//}

} // namespace duckdb_adbc
// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.





#include <algorithm>
#include <cstring>
#include <string>
#include <unordered_map>
#include <utility>

#if defined(_WIN32)
#include <windows.h> // Must come first

#include <libloaderapi.h>
#include <strsafe.h>
#else
#include <dlfcn.h>
#endif // defined(_WIN32)

namespace duckdb_adbc {

// Platform-specific helpers

#if defined(_WIN32)
/// Append a description of the Windows error to the buffer.
void GetWinError(std::string *buffer) {
	DWORD rc = GetLastError();
	LPVOID message;

	FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
	              /*lpSource=*/nullptr, rc, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
	              reinterpret_cast<LPSTR>(&message), /*nSize=*/0, /*Arguments=*/nullptr);

	(*buffer) += '(';
	(*buffer) += std::to_string(rc);
	(*buffer) += ") ";
	(*buffer) += reinterpret_cast<char *>(message);
	LocalFree(message);
}

#endif // defined(_WIN32)

// Error handling

void ReleaseError(struct AdbcError *error) {
	if (error) {
		if (error->message)
			delete[] error->message;
		error->message = nullptr;
		error->release = nullptr;
	}
}

void SetError(struct AdbcError *error, const std::string &message) {
	if (!error)
		return;
	if (error->message) {
		// Append
		std::string buffer = error->message;
		buffer.reserve(buffer.size() + message.size() + 1);
		buffer += '\n';
		buffer += message;
		error->release(error);

		error->message = new char[buffer.size() + 1];
		buffer.copy(error->message, buffer.size());
		error->message[buffer.size()] = '\0';
	} else {
		error->message = new char[message.size() + 1];
		message.copy(error->message, message.size());
		error->message[message.size()] = '\0';
	}
	error->release = ReleaseError;
}

// Driver state

/// Hold the driver DLL and the driver release callback in the driver struct.
struct ManagerDriverState {
	// The original release callback
	AdbcStatusCode (*driver_release)(struct AdbcDriver *driver, struct AdbcError *error);

#if defined(_WIN32)
	// The loaded DLL
	HMODULE handle;
#endif // defined(_WIN32)
};

/// Unload the driver DLL.
static AdbcStatusCode ReleaseDriver(struct AdbcDriver *driver, struct AdbcError *error) {
	AdbcStatusCode status = ADBC_STATUS_OK;

	if (!driver->private_manager)
		return status;
	ManagerDriverState *state = reinterpret_cast<ManagerDriverState *>(driver->private_manager);

	if (state->driver_release) {
		status = state->driver_release(driver, error);
	}

#if defined(_WIN32)
	// TODO(apache/arrow-adbc#204): causes tests to segfault
	// if (!FreeLibrary(state->handle)) {
	//   std::string message = "FreeLibrary() failed: ";
	//   GetWinError(&message);
	//   SetError(error, message);
	// }
#endif // defined(_WIN32)

	driver->private_manager = nullptr;
	delete state;
	return status;
}

// Default stubs

AdbcStatusCode ConnectionCommit(struct AdbcConnection *, struct AdbcError *error) {
	return ADBC_STATUS_NOT_IMPLEMENTED;
}

AdbcStatusCode ConnectionGetInfo(struct AdbcConnection *connection, uint32_t *info_codes, size_t info_codes_length,
                                 struct ArrowArrayStream *out, struct AdbcError *error) {
	return ADBC_STATUS_NOT_IMPLEMENTED;
}

AdbcStatusCode ConnectionGetTableSchema(struct AdbcConnection *, const char *, const char *, const char *,
                                        struct ArrowSchema *, struct AdbcError *error) {
	return ADBC_STATUS_NOT_IMPLEMENTED;
}

AdbcStatusCode ConnectionReadPartition(struct AdbcConnection *connection, const uint8_t *serialized_partition,
                                       size_t serialized_length, struct ArrowArrayStream *out,
                                       struct AdbcError *error) {
	return ADBC_STATUS_NOT_IMPLEMENTED;
}

AdbcStatusCode ConnectionRollback(struct AdbcConnection *, struct AdbcError *error) {
	return ADBC_STATUS_NOT_IMPLEMENTED;
}

AdbcStatusCode StatementBind(struct AdbcStatement *, struct ArrowArray *, struct ArrowSchema *,
                             struct AdbcError *error) {
	return ADBC_STATUS_NOT_IMPLEMENTED;
}

AdbcStatusCode StatementExecutePartitions(struct AdbcStatement *statement, struct ArrowSchema *schema,
                                          struct AdbcPartitions *partitions, int64_t *rows_affected,
                                          struct AdbcError *error) {
	return ADBC_STATUS_NOT_IMPLEMENTED;
}

AdbcStatusCode StatementGetParameterSchema(struct AdbcStatement *statement, struct ArrowSchema *schema,
                                           struct AdbcError *error) {
	return ADBC_STATUS_NOT_IMPLEMENTED;
}
AdbcStatusCode StatementSetSubstraitPlan(struct AdbcStatement *, const uint8_t *, size_t, struct AdbcError *error) {
	return ADBC_STATUS_NOT_IMPLEMENTED;
}

/// Temporary state while the database is being configured.
struct TempDatabase {
	std::unordered_map<std::string, std::string> options;
	std::string driver;
	// Default name (see adbc.h)
	std::string entrypoint = "AdbcDriverInit";
	AdbcDriverInitFunc init_func = nullptr;
};

/// Temporary state while the database is being configured.
struct TempConnection {
	std::unordered_map<std::string, std::string> options;
};

// Direct implementations of API methods

AdbcStatusCode AdbcDatabaseNew(struct AdbcDatabase *database, struct AdbcError *error) {
	// Allocate a temporary structure to store options pre-Init
	database->private_data = new TempDatabase();
	database->private_driver = nullptr;
	return ADBC_STATUS_OK;
}

AdbcStatusCode AdbcDatabaseSetOption(struct AdbcDatabase *database, const char *key, const char *value,
                                     struct AdbcError *error) {
	if (database->private_driver) {
		return database->private_driver->DatabaseSetOption(database, key, value, error);
	}

	TempDatabase *args = reinterpret_cast<TempDatabase *>(database->private_data);
	if (std::strcmp(key, "driver") == 0) {
		args->driver = value;
	} else if (std::strcmp(key, "entrypoint") == 0) {
		args->entrypoint = value;
	} else {
		args->options[key] = value;
	}
	return ADBC_STATUS_OK;
}

AdbcStatusCode AdbcDriverManagerDatabaseSetInitFunc(struct AdbcDatabase *database, AdbcDriverInitFunc init_func,
                                                    struct AdbcError *error) {
	if (database->private_driver) {
		return ADBC_STATUS_INVALID_STATE;
	}

	TempDatabase *args = reinterpret_cast<TempDatabase *>(database->private_data);
	args->init_func = init_func;
	return ADBC_STATUS_OK;
}

AdbcStatusCode AdbcDatabaseInit(struct AdbcDatabase *database, struct AdbcError *error) {
	if (!database->private_data) {
		SetError(error, "Must call AdbcDatabaseNew first");
		return ADBC_STATUS_INVALID_STATE;
	}
	TempDatabase *args = reinterpret_cast<TempDatabase *>(database->private_data);
	if (args->init_func) {
		// Do nothing
	} else if (args->driver.empty()) {
		SetError(error, "Must provide 'driver' parameter");
		return ADBC_STATUS_INVALID_ARGUMENT;
	}

	database->private_driver = new AdbcDriver;
	std::memset(database->private_driver, 0, sizeof(AdbcDriver));
	AdbcStatusCode status;
	// So we don't confuse a driver into thinking it's initialized already
	database->private_data = nullptr;
	if (args->init_func) {
		status = AdbcLoadDriverFromInitFunc(args->init_func, ADBC_VERSION_1_0_0, database->private_driver, error);
	} else {
		status = AdbcLoadDriver(args->driver.c_str(), args->entrypoint.c_str(), ADBC_VERSION_1_0_0,
		                        database->private_driver, error);
	}
	if (status != ADBC_STATUS_OK) {
		// Restore private_data so it will be released by AdbcDatabaseRelease
		database->private_data = args;
		if (database->private_driver->release) {
			database->private_driver->release(database->private_driver, error);
		}
		delete database->private_driver;
		database->private_driver = nullptr;
		return status;
	}
	status = database->private_driver->DatabaseNew(database, error);
	if (status != ADBC_STATUS_OK) {
		if (database->private_driver->release) {
			database->private_driver->release(database->private_driver, error);
		}
		delete database->private_driver;
		database->private_driver = nullptr;
		return status;
	}
	for (const auto &option : args->options) {
		status =
		    database->private_driver->DatabaseSetOption(database, option.first.c_str(), option.second.c_str(), error);
		if (status != ADBC_STATUS_OK) {
			delete args;
			// Release the database
			std::ignore = database->private_driver->DatabaseRelease(database, error);
			if (database->private_driver->release) {
				database->private_driver->release(database->private_driver, error);
			}
			delete database->private_driver;
			database->private_driver = nullptr;
			// Should be redundant, but ensure that AdbcDatabaseRelease
			// below doesn't think that it contains a TempDatabase
			database->private_data = nullptr;
			return status;
		}
	}
	delete args;
	return database->private_driver->DatabaseInit(database, error);
}

AdbcStatusCode AdbcDatabaseRelease(struct AdbcDatabase *database, struct AdbcError *error) {
	if (!database->private_driver) {
		if (database->private_data) {
			TempDatabase *args = reinterpret_cast<TempDatabase *>(database->private_data);
			delete args;
			database->private_data = nullptr;
			return ADBC_STATUS_OK;
		}
		return ADBC_STATUS_INVALID_STATE;
	}
	auto status = database->private_driver->DatabaseRelease(database, error);
	if (database->private_driver->release) {
		database->private_driver->release(database->private_driver, error);
	}
	delete database->private_driver;
	database->private_data = nullptr;
	database->private_driver = nullptr;
	return status;
}

AdbcStatusCode AdbcConnectionCommit(struct AdbcConnection *connection, struct AdbcError *error) {
	if (!connection->private_driver) {
		return ADBC_STATUS_INVALID_STATE;
	}
	return connection->private_driver->ConnectionCommit(connection, error);
}

AdbcStatusCode AdbcConnectionGetInfo(struct AdbcConnection *connection, uint32_t *info_codes, size_t info_codes_length,
                                     struct ArrowArrayStream *out, struct AdbcError *error) {
	if (!connection->private_driver) {
		return ADBC_STATUS_INVALID_STATE;
	}
	return connection->private_driver->ConnectionGetInfo(connection, info_codes, info_codes_length, out, error);
}

AdbcStatusCode AdbcConnectionGetObjects(struct AdbcConnection *connection, int depth, const char *catalog,
                                        const char *db_schema, const char *table_name, const char **table_types,
                                        const char *column_name, struct ArrowArrayStream *stream,
                                        struct AdbcError *error) {
	if (!connection->private_driver) {
		return ADBC_STATUS_INVALID_STATE;
	}
	return connection->private_driver->ConnectionGetObjects(connection, depth, catalog, db_schema, table_name,
	                                                        table_types, column_name, stream, error);
}

AdbcStatusCode AdbcConnectionGetTableSchema(struct AdbcConnection *connection, const char *catalog,
                                            const char *db_schema, const char *table_name, struct ArrowSchema *schema,
                                            struct AdbcError *error) {
	if (!connection->private_driver) {
		return ADBC_STATUS_INVALID_STATE;
	}
	return connection->private_driver->ConnectionGetTableSchema(connection, catalog, db_schema, table_name, schema,
	                                                            error);
}

AdbcStatusCode AdbcConnectionGetTableTypes(struct AdbcConnection *connection, struct ArrowArrayStream *stream,
                                           struct AdbcError *error) {
	if (!connection->private_driver) {
		return ADBC_STATUS_INVALID_STATE;
	}
	return connection->private_driver->ConnectionGetTableTypes(connection, stream, error);
}

AdbcStatusCode AdbcConnectionInit(struct AdbcConnection *connection, struct AdbcDatabase *database,
                                  struct AdbcError *error) {
	if (!connection->private_data) {
		SetError(error, "Must call AdbcConnectionNew first");
		return ADBC_STATUS_INVALID_STATE;
	} else if (!database->private_driver) {
		SetError(error, "Database is not initialized");
		return ADBC_STATUS_INVALID_ARGUMENT;
	}
	TempConnection *args = reinterpret_cast<TempConnection *>(connection->private_data);
	connection->private_data = nullptr;
	std::unordered_map<std::string, std::string> options = std::move(args->options);
	delete args;

	auto status = database->private_driver->ConnectionNew(connection, error);
	if (status != ADBC_STATUS_OK)
		return status;
	connection->private_driver = database->private_driver;

	for (const auto &option : options) {
		status = database->private_driver->ConnectionSetOption(connection, option.first.c_str(), option.second.c_str(),
		                                                       error);
		if (status != ADBC_STATUS_OK)
			return status;
	}
	return connection->private_driver->ConnectionInit(connection, database, error);
}

AdbcStatusCode AdbcConnectionNew(struct AdbcConnection *connection, struct AdbcError *error) {
	// Allocate a temporary structure to store options pre-Init, because
	// we don't get access to the database (and hence the driver
	// function table) until then
	connection->private_data = new TempConnection;
	connection->private_driver = nullptr;
	return ADBC_STATUS_OK;
}

AdbcStatusCode AdbcConnectionReadPartition(struct AdbcConnection *connection, const uint8_t *serialized_partition,
                                           size_t serialized_length, struct ArrowArrayStream *out,
                                           struct AdbcError *error) {
	if (!connection->private_driver) {
		return ADBC_STATUS_INVALID_STATE;
	}
	return connection->private_driver->ConnectionReadPartition(connection, serialized_partition, serialized_length, out,
	                                                           error);
}

AdbcStatusCode AdbcConnectionRelease(struct AdbcConnection *connection, struct AdbcError *error) {
	if (!connection->private_driver) {
		if (connection->private_data) {
			TempConnection *args = reinterpret_cast<TempConnection *>(connection->private_data);
			delete args;
			connection->private_data = nullptr;
			return ADBC_STATUS_OK;
		}
		return ADBC_STATUS_INVALID_STATE;
	}
	auto status = connection->private_driver->ConnectionRelease(connection, error);
	connection->private_driver = nullptr;
	return status;
}

AdbcStatusCode AdbcConnectionRollback(struct AdbcConnection *connection, struct AdbcError *error) {
	if (!connection->private_driver) {
		return ADBC_STATUS_INVALID_STATE;
	}
	return connection->private_driver->ConnectionRollback(connection, error);
}

AdbcStatusCode AdbcConnectionSetOption(struct AdbcConnection *connection, const char *key, const char *value,
                                       struct AdbcError *error) {
	if (!connection->private_data) {
		SetError(error, "AdbcConnectionSetOption: must AdbcConnectionNew first");
		return ADBC_STATUS_INVALID_STATE;
	}
	if (!connection->private_driver) {
		// Init not yet called, save the option
		TempConnection *args = reinterpret_cast<TempConnection *>(connection->private_data);
		args->options[key] = value;
		return ADBC_STATUS_OK;
	}
	return connection->private_driver->ConnectionSetOption(connection, key, value, error);
}

AdbcStatusCode AdbcStatementBind(struct AdbcStatement *statement, struct ArrowArray *values, struct ArrowSchema *schema,
                                 struct AdbcError *error) {
	if (!statement->private_driver) {
		return ADBC_STATUS_INVALID_STATE;
	}
	return statement->private_driver->StatementBind(statement, values, schema, error);
}

AdbcStatusCode AdbcStatementBindStream(struct AdbcStatement *statement, struct ArrowArrayStream *stream,
                                       struct AdbcError *error) {
	if (!statement->private_driver) {
		return ADBC_STATUS_INVALID_STATE;
	}
	return statement->private_driver->StatementBindStream(statement, stream, error);
}

// XXX: cpplint gets confused here if declared as 'struct ArrowSchema* schema'
AdbcStatusCode AdbcStatementExecutePartitions(struct AdbcStatement *statement, ArrowSchema *schema,
                                              struct AdbcPartitions *partitions, int64_t *rows_affected,
                                              struct AdbcError *error) {
	if (!statement->private_driver) {
		return ADBC_STATUS_INVALID_STATE;
	}
	return statement->private_driver->StatementExecutePartitions(statement, schema, partitions, rows_affected, error);
}

AdbcStatusCode AdbcStatementExecuteQuery(struct AdbcStatement *statement, struct ArrowArrayStream *out,
                                         int64_t *rows_affected, struct AdbcError *error) {
	if (!statement->private_driver) {
		return ADBC_STATUS_INVALID_STATE;
	}
	return statement->private_driver->StatementExecuteQuery(statement, out, rows_affected, error);
}

AdbcStatusCode AdbcStatementGetParameterSchema(struct AdbcStatement *statement, struct ArrowSchema *schema,
                                               struct AdbcError *error) {
	if (!statement->private_driver) {
		return ADBC_STATUS_INVALID_STATE;
	}
	return statement->private_driver->StatementGetParameterSchema(statement, schema, error);
}

AdbcStatusCode AdbcStatementNew(struct AdbcConnection *connection, struct AdbcStatement *statement,
                                struct AdbcError *error) {
	if (!connection->private_driver) {
		return ADBC_STATUS_INVALID_STATE;
	}
	auto status = connection->private_driver->StatementNew(connection, statement, error);
	statement->private_driver = connection->private_driver;
	return status;
}

AdbcStatusCode AdbcStatementPrepare(struct AdbcStatement *statement, struct AdbcError *error) {
	if (!statement->private_driver) {
		return ADBC_STATUS_INVALID_STATE;
	}
	return statement->private_driver->StatementPrepare(statement, error);
}

AdbcStatusCode AdbcStatementRelease(struct AdbcStatement *statement, struct AdbcError *error) {
	if (!statement->private_driver) {
		return ADBC_STATUS_INVALID_STATE;
	}
	auto status = statement->private_driver->StatementRelease(statement, error);
	statement->private_driver = nullptr;
	return status;
}

AdbcStatusCode AdbcStatementSetOption(struct AdbcStatement *statement, const char *key, const char *value,
                                      struct AdbcError *error) {
	if (!statement->private_driver) {
		return ADBC_STATUS_INVALID_STATE;
	}
	return statement->private_driver->StatementSetOption(statement, key, value, error);
}

AdbcStatusCode AdbcStatementSetSqlQuery(struct AdbcStatement *statement, const char *query, struct AdbcError *error) {
	if (!statement->private_driver) {
		return ADBC_STATUS_INVALID_STATE;
	}
	return statement->private_driver->StatementSetSqlQuery(statement, query, error);
}

AdbcStatusCode AdbcStatementSetSubstraitPlan(struct AdbcStatement *statement, const uint8_t *plan, size_t length,
                                             struct AdbcError *error) {
	if (!statement->private_driver) {
		return ADBC_STATUS_INVALID_STATE;
	}
	return statement->private_driver->StatementSetSubstraitPlan(statement, plan, length, error);
}

const char *AdbcStatusCodeMessage(AdbcStatusCode code) {
#define STRINGIFY(s)       #s
#define STRINGIFY_VALUE(s) STRINGIFY(s)
#define CASE(CONSTANT)                                                                                                 \
	case CONSTANT:                                                                                                     \
		return #CONSTANT " (" STRINGIFY_VALUE(CONSTANT) ")";

	switch (code) {
		CASE(ADBC_STATUS_OK);
		CASE(ADBC_STATUS_UNKNOWN);
		CASE(ADBC_STATUS_NOT_IMPLEMENTED);
		CASE(ADBC_STATUS_NOT_FOUND);
		CASE(ADBC_STATUS_ALREADY_EXISTS);
		CASE(ADBC_STATUS_INVALID_ARGUMENT);
		CASE(ADBC_STATUS_INVALID_STATE);
		CASE(ADBC_STATUS_INVALID_DATA);
		CASE(ADBC_STATUS_INTEGRITY);
		CASE(ADBC_STATUS_INTERNAL);
		CASE(ADBC_STATUS_IO);
		CASE(ADBC_STATUS_CANCELLED);
		CASE(ADBC_STATUS_TIMEOUT);
		CASE(ADBC_STATUS_UNAUTHENTICATED);
		CASE(ADBC_STATUS_UNAUTHORIZED);
	default:
		return "(invalid code)";
	}
#undef CASE
#undef STRINGIFY_VALUE
#undef STRINGIFY
}

AdbcStatusCode AdbcLoadDriver(const char *driver_name, const char *entrypoint, int version, void *raw_driver,
                              struct AdbcError *error) {
	AdbcDriverInitFunc init_func;
	std::string error_message;

	if (version != ADBC_VERSION_1_0_0) {
		SetError(error, "Only ADBC 1.0.0 is supported");
		return ADBC_STATUS_NOT_IMPLEMENTED;
	}

	auto *driver = reinterpret_cast<struct AdbcDriver *>(raw_driver);

	if (!entrypoint) {
		// Default entrypoint (see adbc.h)
		entrypoint = "AdbcDriverInit";
	}

#if defined(_WIN32)

	HMODULE handle = LoadLibraryExA(driver_name, NULL, 0);
	if (!handle) {
		error_message += driver_name;
		error_message += ": LoadLibraryExA() failed: ";
		GetWinError(&error_message);

		std::string full_driver_name = driver_name;
		full_driver_name += ".lib";
		handle = LoadLibraryExA(full_driver_name.c_str(), NULL, 0);
		if (!handle) {
			error_message += '\n';
			error_message += full_driver_name;
			error_message += ": LoadLibraryExA() failed: ";
			GetWinError(&error_message);
		}
	}
	if (!handle) {
		SetError(error, error_message);
		return ADBC_STATUS_INTERNAL;
	}

	void *load_handle = reinterpret_cast<void *>(GetProcAddress(handle, entrypoint));
	init_func = reinterpret_cast<AdbcDriverInitFunc>(load_handle);
	if (!init_func) {
		std::string message = "GetProcAddress(";
		message += entrypoint;
		message += ") failed: ";
		GetWinError(&message);
		if (!FreeLibrary(handle)) {
			message += "\nFreeLibrary() failed: ";
			GetWinError(&message);
		}
		SetError(error, message);
		return ADBC_STATUS_INTERNAL;
	}

#else

#if defined(__APPLE__)
	const std::string kPlatformLibraryPrefix = "lib";
	const std::string kPlatformLibrarySuffix = ".dylib";
#else
	const std::string kPlatformLibraryPrefix = "lib";
	const std::string kPlatformLibrarySuffix = ".so";
#endif // defined(__APPLE__)

	void *handle = dlopen(driver_name, RTLD_NOW | RTLD_LOCAL);
	if (!handle) {
		error_message = "dlopen() failed: ";
		error_message += dlerror();

		// If applicable, append the shared library prefix/extension and
		// try again (this way you don't have to hardcode driver names by
		// platform in the application)
		const std::string driver_str = driver_name;

		std::string full_driver_name;
		if (driver_str.size() < kPlatformLibraryPrefix.size() ||
		    driver_str.compare(0, kPlatformLibraryPrefix.size(), kPlatformLibraryPrefix) != 0) {
			full_driver_name += kPlatformLibraryPrefix;
		}
		full_driver_name += driver_name;
		if (driver_str.size() < kPlatformLibrarySuffix.size() ||
		    driver_str.compare(full_driver_name.size() - kPlatformLibrarySuffix.size(), kPlatformLibrarySuffix.size(),
		                       kPlatformLibrarySuffix) != 0) {
			full_driver_name += kPlatformLibrarySuffix;
		}
		handle = dlopen(full_driver_name.c_str(), RTLD_NOW | RTLD_LOCAL);
		if (!handle) {
			error_message += "\ndlopen() failed: ";
			error_message += dlerror();
		}
	}
	if (!handle) {
		SetError(error, error_message);
		// AdbcDatabaseInit tries to call this if set
		driver->release = nullptr;
		return ADBC_STATUS_INTERNAL;
	}

	void *load_handle = dlsym(handle, entrypoint);
	if (!load_handle) {
		std::string message = "dlsym(";
		message += entrypoint;
		message += ") failed: ";
		message += dlerror();
		SetError(error, message);
		return ADBC_STATUS_INTERNAL;
	}
	init_func = reinterpret_cast<AdbcDriverInitFunc>(load_handle);

#endif // defined(_WIN32)

	AdbcStatusCode status = AdbcLoadDriverFromInitFunc(init_func, version, driver, error);
	if (status == ADBC_STATUS_OK) {
		ManagerDriverState *state = new ManagerDriverState;
		state->driver_release = driver->release;
#if defined(_WIN32)
		state->handle = handle;
#endif // defined(_WIN32)
		driver->release = &ReleaseDriver;
		driver->private_manager = state;
	} else {
#if defined(_WIN32)
		if (!FreeLibrary(handle)) {
			std::string message = "FreeLibrary() failed: ";
			GetWinError(&message);
			SetError(error, message);
		}
#endif // defined(_WIN32)
	}
	return status;
}

AdbcStatusCode AdbcLoadDriverFromInitFunc(AdbcDriverInitFunc init_func, int version, void *raw_driver,
                                          struct AdbcError *error) {
#define FILL_DEFAULT(DRIVER, STUB)                                                                                     \
	if (!DRIVER->STUB) {                                                                                               \
		DRIVER->STUB = &STUB;                                                                                          \
	}
#define CHECK_REQUIRED(DRIVER, STUB)                                                                                   \
	if (!DRIVER->STUB) {                                                                                               \
		SetError(error, "Driver does not implement required function Adbc" #STUB);                                     \
		return ADBC_STATUS_INTERNAL;                                                                                   \
	}

	auto result = init_func(version, raw_driver, error);
	if (result != ADBC_STATUS_OK) {
		return result;
	}

	if (version == ADBC_VERSION_1_0_0) {
		auto *driver = reinterpret_cast<struct AdbcDriver *>(raw_driver);
		CHECK_REQUIRED(driver, DatabaseNew);
		CHECK_REQUIRED(driver, DatabaseInit);
		CHECK_REQUIRED(driver, DatabaseRelease);
		FILL_DEFAULT(driver, DatabaseSetOption);

		CHECK_REQUIRED(driver, ConnectionNew);
		CHECK_REQUIRED(driver, ConnectionInit);
		CHECK_REQUIRED(driver, ConnectionRelease);
		FILL_DEFAULT(driver, ConnectionCommit);
		FILL_DEFAULT(driver, ConnectionGetInfo);
		FILL_DEFAULT(driver, ConnectionGetObjects);
		FILL_DEFAULT(driver, ConnectionGetTableSchema);
		FILL_DEFAULT(driver, ConnectionGetTableTypes);
		FILL_DEFAULT(driver, ConnectionReadPartition);
		FILL_DEFAULT(driver, ConnectionRollback);
		FILL_DEFAULT(driver, ConnectionSetOption);

		FILL_DEFAULT(driver, StatementExecutePartitions);
		CHECK_REQUIRED(driver, StatementExecuteQuery);
		CHECK_REQUIRED(driver, StatementNew);
		CHECK_REQUIRED(driver, StatementRelease);
		FILL_DEFAULT(driver, StatementBind);
		FILL_DEFAULT(driver, StatementGetParameterSchema);
		FILL_DEFAULT(driver, StatementPrepare);
		FILL_DEFAULT(driver, StatementSetOption);
		FILL_DEFAULT(driver, StatementSetSqlQuery);
		FILL_DEFAULT(driver, StatementSetSubstraitPlan);
	}

	return ADBC_STATUS_OK;

#undef FILL_DEFAULT
#undef CHECK_REQUIRED
}
} // namespace duckdb_adbc






#include <cstdint>

#ifdef DUCKDB_DEBUG_ALLOCATION




#include <execinfo.h>
#endif

#if defined(BUILD_JEMALLOC_EXTENSION) && !defined(WIN32)
#include "jemalloc-extension.hpp"
#endif

namespace duckdb {

AllocatedData::AllocatedData() : allocator(nullptr), pointer(nullptr), allocated_size(0) {
}

AllocatedData::AllocatedData(Allocator &allocator, data_ptr_t pointer, idx_t allocated_size)
    : allocator(&allocator), pointer(pointer), allocated_size(allocated_size) {
	if (!pointer) {
		throw InternalException("AllocatedData object constructed with nullptr");
	}
}
AllocatedData::~AllocatedData() {
	Reset();
}

AllocatedData::AllocatedData(AllocatedData &&other) noexcept
    : allocator(other.allocator), pointer(nullptr), allocated_size(0) {
	std::swap(pointer, other.pointer);
	std::swap(allocated_size, other.allocated_size);
}

AllocatedData &AllocatedData::operator=(AllocatedData &&other) noexcept {
	std::swap(allocator, other.allocator);
	std::swap(pointer, other.pointer);
	std::swap(allocated_size, other.allocated_size);
	return *this;
}

void AllocatedData::Reset() {
	if (!pointer) {
		return;
	}
	D_ASSERT(allocator);
	allocator->FreeData(pointer, allocated_size);
	allocated_size = 0;
	pointer = nullptr;
}

//===--------------------------------------------------------------------===//
// Debug Info
//===--------------------------------------------------------------------===//
struct AllocatorDebugInfo {
#ifdef DEBUG
	AllocatorDebugInfo();
	~AllocatorDebugInfo();

	void AllocateData(data_ptr_t pointer, idx_t size);
	void FreeData(data_ptr_t pointer, idx_t size);
	void ReallocateData(data_ptr_t pointer, data_ptr_t new_pointer, idx_t old_size, idx_t new_size);

private:
	//! The number of bytes that are outstanding (i.e. that have been allocated - but not freed)
	//! Used for debug purposes
	atomic<idx_t> allocation_count;
#ifdef DUCKDB_DEBUG_ALLOCATION
	mutex pointer_lock;
	//! Set of active outstanding pointers together with stack traces
	unordered_map<data_ptr_t, pair<idx_t, string>> pointers;
#endif
#endif
};

PrivateAllocatorData::PrivateAllocatorData() {
}

PrivateAllocatorData::~PrivateAllocatorData() {
}

//===--------------------------------------------------------------------===//
// Allocator
//===--------------------------------------------------------------------===//
#if defined(BUILD_JEMALLOC_EXTENSION) && !defined(WIN32)
Allocator::Allocator()
    : Allocator(JEMallocExtension::Allocate, JEMallocExtension::Free, JEMallocExtension::Reallocate, nullptr) {
}
#else
Allocator::Allocator()
    : Allocator(Allocator::DefaultAllocate, Allocator::DefaultFree, Allocator::DefaultReallocate, nullptr) {
}
#endif

Allocator::Allocator(allocate_function_ptr_t allocate_function_p, free_function_ptr_t free_function_p,
                     reallocate_function_ptr_t reallocate_function_p, unique_ptr<PrivateAllocatorData> private_data_p)
    : allocate_function(allocate_function_p), free_function(free_function_p),
      reallocate_function(reallocate_function_p), private_data(std::move(private_data_p)) {
	D_ASSERT(allocate_function);
	D_ASSERT(free_function);
	D_ASSERT(reallocate_function);
#ifdef DEBUG
	if (!private_data) {
		private_data = make_uniq<PrivateAllocatorData>();
	}
	private_data->debug_info = make_uniq<AllocatorDebugInfo>();
#endif
}

Allocator::~Allocator() {
}

data_ptr_t Allocator::AllocateData(idx_t size) {
	D_ASSERT(size > 0);
	if (size >= MAXIMUM_ALLOC_SIZE) {
		D_ASSERT(false);
		throw InternalException("Requested allocation size of %llu is out of range - maximum allocation size is %llu",
		                        size, MAXIMUM_ALLOC_SIZE);
	}
	auto result = allocate_function(private_data.get(), size);
#ifdef DEBUG
	D_ASSERT(private_data);
	private_data->debug_info->AllocateData(result, size);
#endif
	if (!result) {
		throw OutOfMemoryException("Failed to allocate block of %llu bytes", size);
	}
	return result;
}

void Allocator::FreeData(data_ptr_t pointer, idx_t size) {
	if (!pointer) {
		return;
	}
	D_ASSERT(size > 0);
#ifdef DEBUG
	D_ASSERT(private_data);
	private_data->debug_info->FreeData(pointer, size);
#endif
	free_function(private_data.get(), pointer, size);
}

data_ptr_t Allocator::ReallocateData(data_ptr_t pointer, idx_t old_size, idx_t size) {
	if (!pointer) {
		return nullptr;
	}
	if (size >= MAXIMUM_ALLOC_SIZE) {
		D_ASSERT(false);
		throw InternalException(
		    "Requested re-allocation size of %llu is out of range - maximum allocation size is %llu", size,
		    MAXIMUM_ALLOC_SIZE);
	}
	auto new_pointer = reallocate_function(private_data.get(), pointer, old_size, size);
#ifdef DEBUG
	D_ASSERT(private_data);
	private_data->debug_info->ReallocateData(pointer, new_pointer, old_size, size);
#endif
	if (!new_pointer) {
		throw OutOfMemoryException("Failed to re-allocate block of %llu bytes", size);
	}
	return new_pointer;
}

shared_ptr<Allocator> &Allocator::DefaultAllocatorReference() {
	static shared_ptr<Allocator> DEFAULT_ALLOCATOR = make_shared<Allocator>();
	return DEFAULT_ALLOCATOR;
}

Allocator &Allocator::DefaultAllocator() {
	return *DefaultAllocatorReference();
}

//===--------------------------------------------------------------------===//
// Debug Info (extended)
//===--------------------------------------------------------------------===//
#ifdef DEBUG
AllocatorDebugInfo::AllocatorDebugInfo() {
	allocation_count = 0;
}
AllocatorDebugInfo::~AllocatorDebugInfo() {
#ifdef DUCKDB_DEBUG_ALLOCATION
	if (allocation_count != 0) {
		printf("Outstanding allocations found for Allocator\n");
		for (auto &entry : pointers) {
			printf("Allocation of size %llu at address %p\n", entry.second.first, (void *)entry.first);
			printf("Stack trace:\n%s\n", entry.second.second.c_str());
			printf("\n");
		}
	}
#endif
	//! Verify that there is no outstanding memory still associated with the batched allocator
	//! Only works for access to the batched allocator through the batched allocator interface
	//! If this assertion triggers, enable DUCKDB_DEBUG_ALLOCATION for more information about the allocations
	D_ASSERT(allocation_count == 0);
}

void AllocatorDebugInfo::AllocateData(data_ptr_t pointer, idx_t size) {
	allocation_count += size;
#ifdef DUCKDB_DEBUG_ALLOCATION
	lock_guard<mutex> l(pointer_lock);
	pointers[pointer] = make_pair(size, Exception::GetStackTrace());
#endif
}

void AllocatorDebugInfo::FreeData(data_ptr_t pointer, idx_t size) {
	D_ASSERT(allocation_count >= size);
	allocation_count -= size;
#ifdef DUCKDB_DEBUG_ALLOCATION
	lock_guard<mutex> l(pointer_lock);
	// verify that the pointer exists
	D_ASSERT(pointers.find(pointer) != pointers.end());
	// verify that the stored size matches the passed in size
	D_ASSERT(pointers[pointer].first == size);
	// erase the pointer
	pointers.erase(pointer);
#endif
}

void AllocatorDebugInfo::ReallocateData(data_ptr_t pointer, data_ptr_t new_pointer, idx_t old_size, idx_t new_size) {
	FreeData(pointer, old_size);
	AllocateData(new_pointer, new_size);
}

#endif

} // namespace duckdb








namespace duckdb {

//===--------------------------------------------------------------------===//
// Arrow append data
//===--------------------------------------------------------------------===//
typedef void (*initialize_t)(ArrowAppendData &result, const LogicalType &type, idx_t capacity);
typedef void (*append_vector_t)(ArrowAppendData &append_data, Vector &input, idx_t from, idx_t to, idx_t input_size);
typedef void (*finalize_t)(ArrowAppendData &append_data, const LogicalType &type, ArrowArray *result);

struct ArrowAppendData {
	explicit ArrowAppendData(ArrowOptions &options_p) : options(options_p) {
	}
	// the buffers of the arrow vector
	ArrowBuffer validity;
	ArrowBuffer main_buffer;
	ArrowBuffer aux_buffer;

	idx_t row_count = 0;
	idx_t null_count = 0;

	// function pointers for construction
	initialize_t initialize = nullptr;
	append_vector_t append_vector = nullptr;
	finalize_t finalize = nullptr;

	// child data (if any)
	vector<unique_ptr<ArrowAppendData>> child_data;

	// the arrow array C API data, only set after Finalize
	unique_ptr<ArrowArray> array;
	duckdb::array<const void *, 3> buffers = {{nullptr, nullptr, nullptr}};
	vector<ArrowArray *> child_pointers;

	ArrowOptions options;
};

//===--------------------------------------------------------------------===//
// ArrowAppender
//===--------------------------------------------------------------------===//
static unique_ptr<ArrowAppendData> InitializeArrowChild(const LogicalType &type, idx_t capacity, ArrowOptions &options);
static ArrowArray *FinalizeArrowChild(const LogicalType &type, ArrowAppendData &append_data);

ArrowAppender::ArrowAppender(vector<LogicalType> types_p, idx_t initial_capacity, ArrowOptions options)
    : types(std::move(types_p)) {
	for (auto &type : types) {
		auto entry = InitializeArrowChild(type, initial_capacity, options);
		root_data.push_back(std::move(entry));
	}
}

ArrowAppender::~ArrowAppender() {
}

//===--------------------------------------------------------------------===//
// Append Helper Functions
//===--------------------------------------------------------------------===//
static void GetBitPosition(idx_t row_idx, idx_t &current_byte, uint8_t &current_bit) {
	current_byte = row_idx / 8;
	current_bit = row_idx % 8;
}

static void UnsetBit(uint8_t *data, idx_t current_byte, uint8_t current_bit) {
	data[current_byte] &= ~((uint64_t)1 << current_bit);
}

static void NextBit(idx_t &current_byte, uint8_t &current_bit) {
	current_bit++;
	if (current_bit == 8) {
		current_byte++;
		current_bit = 0;
	}
}

static void ResizeValidity(ArrowBuffer &buffer, idx_t row_count) {
	auto byte_count = (row_count + 7) / 8;
	buffer.resize(byte_count, 0xFF);
}

static void SetNull(ArrowAppendData &append_data, uint8_t *validity_data, idx_t current_byte, uint8_t current_bit) {
	UnsetBit(validity_data, current_byte, current_bit);
	append_data.null_count++;
}

static void AppendValidity(ArrowAppendData &append_data, UnifiedVectorFormat &format, idx_t from, idx_t to) {
	// resize the buffer, filling the validity buffer with all valid values
	idx_t size = to - from;
	ResizeValidity(append_data.validity, append_data.row_count + size);
	if (format.validity.AllValid()) {
		// if all values are valid we don't need to do anything else
		return;
	}

	// otherwise we iterate through the validity mask
	auto validity_data = (uint8_t *)append_data.validity.data();
	uint8_t current_bit;
	idx_t current_byte;
	GetBitPosition(append_data.row_count, current_byte, current_bit);
	for (idx_t i = from; i < to; i++) {
		auto source_idx = format.sel->get_index(i);
		// append the validity mask
		if (!format.validity.RowIsValid(source_idx)) {
			SetNull(append_data, validity_data, current_byte, current_bit);
		}
		NextBit(current_byte, current_bit);
	}
}

//===--------------------------------------------------------------------===//
// Scalar Types
//===--------------------------------------------------------------------===//
struct ArrowScalarConverter {
	template <class TGT, class SRC>
	static TGT Operation(SRC input) {
		return input;
	}

	static bool SkipNulls() {
		return false;
	}

	template <class TGT>
	static void SetNull(TGT &value) {
	}
};

struct ArrowIntervalConverter {
	template <class TGT, class SRC>
	static TGT Operation(SRC input) {
		ArrowInterval result;
		result.months = input.months;
		result.days = input.days;
		result.nanoseconds = input.micros * Interval::NANOS_PER_MICRO;
		return result;
	}

	static bool SkipNulls() {
		return true;
	}

	template <class TGT>
	static void SetNull(TGT &value) {
	}
};

template <class TGT, class SRC = TGT, class OP = ArrowScalarConverter>
struct ArrowScalarBaseData {
	static void Append(ArrowAppendData &append_data, Vector &input, idx_t from, idx_t to, idx_t input_size) {
		idx_t size = to - from;
		UnifiedVectorFormat format;
		input.ToUnifiedFormat(input_size, format);

		// append the validity mask
		AppendValidity(append_data, format, from, to);

		// append the main data
		append_data.main_buffer.resize(append_data.main_buffer.size() + sizeof(TGT) * size);
		auto data = (SRC *)format.data;
		auto result_data = (TGT *)append_data.main_buffer.data();

		for (idx_t i = from; i < to; i++) {
			auto source_idx = format.sel->get_index(i);
			auto result_idx = append_data.row_count + i - from;

			if (OP::SkipNulls() && !format.validity.RowIsValid(source_idx)) {
				OP::template SetNull<TGT>(result_data[result_idx]);
				continue;
			}
			result_data[result_idx] = OP::template Operation<TGT, SRC>(data[source_idx]);
		}
		append_data.row_count += size;
	}
};

template <class TGT, class SRC = TGT, class OP = ArrowScalarConverter>
struct ArrowScalarData : public ArrowScalarBaseData<TGT, SRC, OP> {
	static void Initialize(ArrowAppendData &result, const LogicalType &type, idx_t capacity) {
		result.main_buffer.reserve(capacity * sizeof(TGT));
	}

	static void Finalize(ArrowAppendData &append_data, const LogicalType &type, ArrowArray *result) {
		result->n_buffers = 2;
		result->buffers[1] = append_data.main_buffer.data();
	}
};

//===--------------------------------------------------------------------===//
// Enums
//===--------------------------------------------------------------------===//
template <class TGT>
struct ArrowEnumData : public ArrowScalarBaseData<TGT> {
	static idx_t GetLength(string_t input) {
		return input.GetSize();
	}
	static void WriteData(data_ptr_t target, string_t input) {
		memcpy(target, input.GetData(), input.GetSize());
	}
	static void EnumAppendVector(ArrowAppendData &append_data, const Vector &input, idx_t size) {
		D_ASSERT(input.GetVectorType() == VectorType::FLAT_VECTOR);

		// resize the validity mask and set up the validity buffer for iteration
		ResizeValidity(append_data.validity, append_data.row_count + size);

		// resize the offset buffer - the offset buffer holds the offsets into the child array
		append_data.main_buffer.resize(append_data.main_buffer.size() + sizeof(uint32_t) * (size + 1));
		auto data = (string_t *)FlatVector::GetData<string_t>(input);
		auto offset_data = (uint32_t *)append_data.main_buffer.data();
		if (append_data.row_count == 0) {
			// first entry
			offset_data[0] = 0;
		}
		// now append the string data to the auxiliary buffer
		// the auxiliary buffer's length depends on the string lengths, so we resize as required
		auto last_offset = offset_data[append_data.row_count];
		for (idx_t i = 0; i < size; i++) {
			auto offset_idx = append_data.row_count + i + 1;

			auto string_length = GetLength(data[i]);

			// append the offset data
			auto current_offset = last_offset + string_length;
			offset_data[offset_idx] = current_offset;

			// resize the string buffer if required, and write the string data
			append_data.aux_buffer.resize(current_offset);
			WriteData(append_data.aux_buffer.data() + last_offset, data[i]);

			last_offset = current_offset;
		}
		append_data.row_count += size;
	}
	static void Initialize(ArrowAppendData &result, const LogicalType &type, idx_t capacity) {
		result.main_buffer.reserve(capacity * sizeof(TGT));
		// construct the enum child data
		auto enum_data = InitializeArrowChild(LogicalType::VARCHAR, EnumType::GetSize(type), result.options);
		EnumAppendVector(*enum_data, EnumType::GetValuesInsertOrder(type), EnumType::GetSize(type));
		result.child_data.push_back(std::move(enum_data));
	}

	static void Finalize(ArrowAppendData &append_data, const LogicalType &type, ArrowArray *result) {
		result->n_buffers = 2;
		result->buffers[1] = append_data.main_buffer.data();
		// finalize the enum child data, and assign it to the dictionary
		result->dictionary = FinalizeArrowChild(LogicalType::VARCHAR, *append_data.child_data[0]);
	}
};

//===--------------------------------------------------------------------===//
// Boolean
//===--------------------------------------------------------------------===//
struct ArrowBoolData {
	static void Initialize(ArrowAppendData &result, const LogicalType &type, idx_t capacity) {
		auto byte_count = (capacity + 7) / 8;
		result.main_buffer.reserve(byte_count);
	}

	static void Append(ArrowAppendData &append_data, Vector &input, idx_t from, idx_t to, idx_t input_size) {
		idx_t size = to - from;
		UnifiedVectorFormat format;
		input.ToUnifiedFormat(input_size, format);

		// we initialize both the validity and the bit set to 1's
		ResizeValidity(append_data.validity, append_data.row_count + size);
		ResizeValidity(append_data.main_buffer, append_data.row_count + size);
		auto data = (bool *)format.data;

		auto result_data = (uint8_t *)append_data.main_buffer.data();
		auto validity_data = (uint8_t *)append_data.validity.data();
		uint8_t current_bit;
		idx_t current_byte;
		GetBitPosition(append_data.row_count, current_byte, current_bit);
		for (idx_t i = from; i < to; i++) {
			auto source_idx = format.sel->get_index(i);
			// append the validity mask
			if (!format.validity.RowIsValid(source_idx)) {
				SetNull(append_data, validity_data, current_byte, current_bit);
			} else if (!data[source_idx]) {
				UnsetBit(result_data, current_byte, current_bit);
			}
			NextBit(current_byte, current_bit);
		}
		append_data.row_count += size;
	}

	static void Finalize(ArrowAppendData &append_data, const LogicalType &type, ArrowArray *result) {
		result->n_buffers = 2;
		result->buffers[1] = append_data.main_buffer.data();
	}
};

//===--------------------------------------------------------------------===//
// Varchar
//===--------------------------------------------------------------------===//
struct ArrowVarcharConverter {
	template <class SRC>
	static idx_t GetLength(SRC input) {
		return input.GetSize();
	}

	template <class SRC>
	static void WriteData(data_ptr_t target, SRC input) {
		memcpy(target, input.GetData(), input.GetSize());
	}
};

struct ArrowUUIDConverter {
	template <class SRC>
	static idx_t GetLength(SRC input) {
		return UUID::STRING_SIZE;
	}

	template <class SRC>
	static void WriteData(data_ptr_t target, SRC input) {
		UUID::ToString(input, (char *)target);
	}
};

template <class SRC = string_t, class OP = ArrowVarcharConverter, class BUFTYPE = uint64_t>
struct ArrowVarcharData {
	static void Initialize(ArrowAppendData &result, const LogicalType &type, idx_t capacity) {
		result.main_buffer.reserve((capacity + 1) * sizeof(BUFTYPE));

		result.aux_buffer.reserve(capacity);
	}

	static void Append(ArrowAppendData &append_data, Vector &input, idx_t from, idx_t to, idx_t input_size) {
		idx_t size = to - from;
		UnifiedVectorFormat format;
		input.ToUnifiedFormat(input_size, format);

		// resize the validity mask and set up the validity buffer for iteration
		ResizeValidity(append_data.validity, append_data.row_count + size);
		auto validity_data = (uint8_t *)append_data.validity.data();

		// resize the offset buffer - the offset buffer holds the offsets into the child array
		append_data.main_buffer.resize(append_data.main_buffer.size() + sizeof(BUFTYPE) * (size + 1));
		auto data = (SRC *)format.data;
		auto offset_data = (BUFTYPE *)append_data.main_buffer.data();
		if (append_data.row_count == 0) {
			// first entry
			offset_data[0] = 0;
		}
		// now append the string data to the auxiliary buffer
		// the auxiliary buffer's length depends on the string lengths, so we resize as required
		auto last_offset = offset_data[append_data.row_count];
		idx_t max_offset = append_data.row_count + to - from;
		if (max_offset > NumericLimits<uint32_t>::Maximum() &&
		    append_data.options.offset_size == ArrowOffsetSize::REGULAR) {
			throw InvalidInputException("Arrow Appender: The maximum total string size for regular string buffers is "
			                            "%u but the offset of %lu exceeds this.",
			                            NumericLimits<uint32_t>::Maximum(), max_offset);
		}
		for (idx_t i = from; i < to; i++) {
			auto source_idx = format.sel->get_index(i);
			auto offset_idx = append_data.row_count + i + 1 - from;

			if (!format.validity.RowIsValid(source_idx)) {
				uint8_t current_bit;
				idx_t current_byte;
				GetBitPosition(append_data.row_count + i - from, current_byte, current_bit);
				SetNull(append_data, validity_data, current_byte, current_bit);
				offset_data[offset_idx] = last_offset;
				continue;
			}

			auto string_length = OP::GetLength(data[source_idx]);

			// append the offset data
			auto current_offset = last_offset + string_length;
			offset_data[offset_idx] = current_offset;

			// resize the string buffer if required, and write the string data
			append_data.aux_buffer.resize(current_offset);
			OP::WriteData(append_data.aux_buffer.data() + last_offset, data[source_idx]);

			last_offset = current_offset;
		}
		append_data.row_count += size;
	}

	static void Finalize(ArrowAppendData &append_data, const LogicalType &type, ArrowArray *result) {
		result->n_buffers = 3;
		result->buffers[1] = append_data.main_buffer.data();
		result->buffers[2] = append_data.aux_buffer.data();
	}
};

//===--------------------------------------------------------------------===//
// Structs
//===--------------------------------------------------------------------===//
struct ArrowStructData {
	static void Initialize(ArrowAppendData &result, const LogicalType &type, idx_t capacity) {
		auto &children = StructType::GetChildTypes(type);
		for (auto &child : children) {
			auto child_buffer = InitializeArrowChild(child.second, capacity, result.options);
			result.child_data.push_back(std::move(child_buffer));
		}
	}

	static void Append(ArrowAppendData &append_data, Vector &input, idx_t from, idx_t to, idx_t input_size) {
		UnifiedVectorFormat format;
		input.ToUnifiedFormat(input_size, format);
		idx_t size = to - from;
		AppendValidity(append_data, format, from, to);
		// append the children of the struct
		auto &children = StructVector::GetEntries(input);
		for (idx_t child_idx = 0; child_idx < children.size(); child_idx++) {
			auto &child = children[child_idx];
			auto &child_data = *append_data.child_data[child_idx];
			child_data.append_vector(child_data, *child, from, to, size);
		}
		append_data.row_count += size;
	}

	static void Finalize(ArrowAppendData &append_data, const LogicalType &type, ArrowArray *result) {
		result->n_buffers = 1;

		auto &child_types = StructType::GetChildTypes(type);
		append_data.child_pointers.resize(child_types.size());
		result->children = append_data.child_pointers.data();
		result->n_children = child_types.size();
		for (idx_t i = 0; i < child_types.size(); i++) {
			auto &child_type = child_types[i].second;
			append_data.child_pointers[i] = FinalizeArrowChild(child_type, *append_data.child_data[i]);
		}
	}
};

//===--------------------------------------------------------------------===//
// Lists
//===--------------------------------------------------------------------===//
void AppendListOffsets(ArrowAppendData &append_data, UnifiedVectorFormat &format, idx_t from, idx_t to,
                       vector<sel_t> &child_sel) {
	// resize the offset buffer - the offset buffer holds the offsets into the child array
	idx_t size = to - from;
	append_data.main_buffer.resize(append_data.main_buffer.size() + sizeof(uint32_t) * (size + 1));
	auto data = (list_entry_t *)format.data;
	auto offset_data = (uint32_t *)append_data.main_buffer.data();
	if (append_data.row_count == 0) {
		// first entry
		offset_data[0] = 0;
	}
	// set up the offsets using the list entries
	auto last_offset = offset_data[append_data.row_count];
	for (idx_t i = from; i < to; i++) {
		auto source_idx = format.sel->get_index(i);
		auto offset_idx = append_data.row_count + i + 1 - from;

		if (!format.validity.RowIsValid(source_idx)) {
			offset_data[offset_idx] = last_offset;
			continue;
		}

		// append the offset data
		auto list_length = data[source_idx].length;
		last_offset += list_length;
		offset_data[offset_idx] = last_offset;

		for (idx_t k = 0; k < list_length; k++) {
			child_sel.push_back(data[source_idx].offset + k);
		}
	}
}

struct ArrowListData {
	static void Initialize(ArrowAppendData &result, const LogicalType &type, idx_t capacity) {
		auto &child_type = ListType::GetChildType(type);
		result.main_buffer.reserve((capacity + 1) * sizeof(uint32_t));
		auto child_buffer = InitializeArrowChild(child_type, capacity, result.options);
		result.child_data.push_back(std::move(child_buffer));
	}

	static void Append(ArrowAppendData &append_data, Vector &input, idx_t from, idx_t to, idx_t input_size) {
		UnifiedVectorFormat format;
		input.ToUnifiedFormat(input_size, format);
		idx_t size = to - from;
		vector<sel_t> child_indices;
		AppendValidity(append_data, format, from, to);
		AppendListOffsets(append_data, format, from, to, child_indices);

		// append the child vector of the list
		SelectionVector child_sel(child_indices.data());
		auto &child = ListVector::GetEntry(input);
		auto child_size = child_indices.size();
		if (size != input_size) {
			// Let's avoid doing this
			Vector child_copy(child.GetType());
			child_copy.Slice(child, child_sel, child_size);
			append_data.child_data[0]->append_vector(*append_data.child_data[0], child_copy, 0, child_size, child_size);
		} else {
			// We don't care about the vector, slice it
			child.Slice(child_sel, child_size);
			append_data.child_data[0]->append_vector(*append_data.child_data[0], child, 0, child_size, child_size);
		}
		append_data.row_count += size;
	}

	static void Finalize(ArrowAppendData &append_data, const LogicalType &type, ArrowArray *result) {
		result->n_buffers = 2;
		result->buffers[1] = append_data.main_buffer.data();

		auto &child_type = ListType::GetChildType(type);
		append_data.child_pointers.resize(1);
		result->children = append_data.child_pointers.data();
		result->n_children = 1;
		append_data.child_pointers[0] = FinalizeArrowChild(child_type, *append_data.child_data[0]);
	}
};

//===--------------------------------------------------------------------===//
// Maps
//===--------------------------------------------------------------------===//
struct ArrowMapData {
	static void Initialize(ArrowAppendData &result, const LogicalType &type, idx_t capacity) {
		// map types are stored in a (too) clever way
		// the main buffer holds the null values and the offsets
		// then we have a single child, which is a struct of the map_type, and the key_type
		result.main_buffer.reserve((capacity + 1) * sizeof(uint32_t));

		auto &key_type = MapType::KeyType(type);
		auto &value_type = MapType::ValueType(type);
		auto internal_struct = make_uniq<ArrowAppendData>(result.options);
		internal_struct->child_data.push_back(InitializeArrowChild(key_type, capacity, result.options));
		internal_struct->child_data.push_back(InitializeArrowChild(value_type, capacity, result.options));

		result.child_data.push_back(std::move(internal_struct));
	}

	static void Append(ArrowAppendData &append_data, Vector &input, idx_t from, idx_t to, idx_t input_size) {
		UnifiedVectorFormat format;
		input.ToUnifiedFormat(input_size, format);
		idx_t size = to - from;
		AppendValidity(append_data, format, from, to);
		vector<sel_t> child_indices;
		AppendListOffsets(append_data, format, from, to, child_indices);

		SelectionVector child_sel(child_indices.data());
		auto &key_vector = MapVector::GetKeys(input);
		auto &value_vector = MapVector::GetValues(input);
		auto list_size = child_indices.size();

		auto &struct_data = *append_data.child_data[0];
		auto &key_data = *struct_data.child_data[0];
		auto &value_data = *struct_data.child_data[1];

		if (size != input_size) {
			// Let's avoid doing this
			Vector key_vector_copy(key_vector.GetType());
			key_vector_copy.Slice(key_vector, child_sel, list_size);
			Vector value_vector_copy(value_vector.GetType());
			value_vector_copy.Slice(value_vector, child_sel, list_size);
			key_data.append_vector(key_data, key_vector_copy, 0, list_size, list_size);
			value_data.append_vector(value_data, value_vector_copy, 0, list_size, list_size);
		} else {
			// We don't care about the vector, slice it
			key_vector.Slice(child_sel, list_size);
			value_vector.Slice(child_sel, list_size);
			key_data.append_vector(key_data, key_vector, 0, list_size, list_size);
			value_data.append_vector(value_data, value_vector, 0, list_size, list_size);
		}

		append_data.row_count += size;
		struct_data.row_count += size;
	}

	static void Finalize(ArrowAppendData &append_data, const LogicalType &type, ArrowArray *result) {
		// set up the main map buffer
		result->n_buffers = 2;
		result->buffers[1] = append_data.main_buffer.data();

		// the main map buffer has a single child: a struct
		append_data.child_pointers.resize(1);
		result->children = append_data.child_pointers.data();
		result->n_children = 1;
		append_data.child_pointers[0] = FinalizeArrowChild(type, *append_data.child_data[0]);

		// now that struct has two children: the key and the value type
		auto &struct_data = *append_data.child_data[0];
		auto &struct_result = append_data.child_pointers[0];
		struct_data.child_pointers.resize(2);
		struct_result->n_buffers = 1;
		struct_result->n_children = 2;
		struct_result->length = struct_data.child_data[0]->row_count;
		struct_result->children = struct_data.child_pointers.data();

		D_ASSERT(struct_data.child_data[0]->row_count == struct_data.child_data[1]->row_count);

		auto &key_type = MapType::KeyType(type);
		auto &value_type = MapType::ValueType(type);
		struct_data.child_pointers[0] = FinalizeArrowChild(key_type, *struct_data.child_data[0]);
		struct_data.child_pointers[1] = FinalizeArrowChild(value_type, *struct_data.child_data[1]);

		// keys cannot have null values
		if (struct_data.child_pointers[0]->null_count > 0) {
			throw std::runtime_error("Arrow doesn't accept NULL keys on Maps");
		}
	}
};

//! Append a data chunk to the underlying arrow array
void ArrowAppender::Append(DataChunk &input, idx_t from, idx_t to, idx_t input_size) {
	D_ASSERT(types == input.GetTypes());
	for (idx_t i = 0; i < input.ColumnCount(); i++) {
		root_data[i]->append_vector(*root_data[i], input.data[i], from, to, input_size);
	}
	row_count += to - from;
}
//===--------------------------------------------------------------------===//
// Initialize Arrow Child
//===--------------------------------------------------------------------===//
template <class OP>
static void InitializeFunctionPointers(ArrowAppendData &append_data) {
	append_data.initialize = OP::Initialize;
	append_data.append_vector = OP::Append;
	append_data.finalize = OP::Finalize;
}

static void InitializeFunctionPointers(ArrowAppendData &append_data, const LogicalType &type) {
	// handle special logical types
	switch (type.id()) {
	case LogicalTypeId::BOOLEAN:
		InitializeFunctionPointers<ArrowBoolData>(append_data);
		break;
	case LogicalTypeId::TINYINT:
		InitializeFunctionPointers<ArrowScalarData<int8_t>>(append_data);
		break;
	case LogicalTypeId::SMALLINT:
		InitializeFunctionPointers<ArrowScalarData<int16_t>>(append_data);
		break;
	case LogicalTypeId::DATE:
	case LogicalTypeId::INTEGER:
		InitializeFunctionPointers<ArrowScalarData<int32_t>>(append_data);
		break;
	case LogicalTypeId::TIME:
	case LogicalTypeId::TIMESTAMP_SEC:
	case LogicalTypeId::TIMESTAMP_MS:
	case LogicalTypeId::TIMESTAMP:
	case LogicalTypeId::TIMESTAMP_NS:
	case LogicalTypeId::TIMESTAMP_TZ:
	case LogicalTypeId::TIME_TZ:
	case LogicalTypeId::BIGINT:
		InitializeFunctionPointers<ArrowScalarData<int64_t>>(append_data);
		break;
	case LogicalTypeId::HUGEINT:
		InitializeFunctionPointers<ArrowScalarData<hugeint_t>>(append_data);
		break;
	case LogicalTypeId::UTINYINT:
		InitializeFunctionPointers<ArrowScalarData<uint8_t>>(append_data);
		break;
	case LogicalTypeId::USMALLINT:
		InitializeFunctionPointers<ArrowScalarData<uint16_t>>(append_data);
		break;
	case LogicalTypeId::UINTEGER:
		InitializeFunctionPointers<ArrowScalarData<uint32_t>>(append_data);
		break;
	case LogicalTypeId::UBIGINT:
		InitializeFunctionPointers<ArrowScalarData<uint64_t>>(append_data);
		break;
	case LogicalTypeId::FLOAT:
		InitializeFunctionPointers<ArrowScalarData<float>>(append_data);
		break;
	case LogicalTypeId::DOUBLE:
		InitializeFunctionPointers<ArrowScalarData<double>>(append_data);
		break;
	case LogicalTypeId::DECIMAL:
		switch (type.InternalType()) {
		case PhysicalType::INT16:
			InitializeFunctionPointers<ArrowScalarData<hugeint_t, int16_t>>(append_data);
			break;
		case PhysicalType::INT32:
			InitializeFunctionPointers<ArrowScalarData<hugeint_t, int32_t>>(append_data);
			break;
		case PhysicalType::INT64:
			InitializeFunctionPointers<ArrowScalarData<hugeint_t, int64_t>>(append_data);
			break;
		case PhysicalType::INT128:
			InitializeFunctionPointers<ArrowScalarData<hugeint_t>>(append_data);
			break;
		default:
			throw InternalException("Unsupported internal decimal type");
		}
		break;
	case LogicalTypeId::VARCHAR:
	case LogicalTypeId::BLOB:
	case LogicalTypeId::BIT:
		if (append_data.options.offset_size == ArrowOffsetSize::LARGE) {
			InitializeFunctionPointers<ArrowVarcharData<string_t>>(append_data);
		} else {
			InitializeFunctionPointers<ArrowVarcharData<string_t, ArrowVarcharConverter, uint32_t>>(append_data);
		}
		break;
	case LogicalTypeId::UUID:
		if (append_data.options.offset_size == ArrowOffsetSize::LARGE) {
			InitializeFunctionPointers<ArrowVarcharData<hugeint_t, ArrowUUIDConverter>>(append_data);
		} else {
			InitializeFunctionPointers<ArrowVarcharData<hugeint_t, ArrowUUIDConverter, uint32_t>>(append_data);
		}
		break;
	case LogicalTypeId::ENUM:
		switch (type.InternalType()) {
		case PhysicalType::UINT8:
			InitializeFunctionPointers<ArrowEnumData<uint8_t>>(append_data);
			break;
		case PhysicalType::UINT16:
			InitializeFunctionPointers<ArrowEnumData<uint16_t>>(append_data);
			break;
		case PhysicalType::UINT32:
			InitializeFunctionPointers<ArrowEnumData<uint32_t>>(append_data);
			break;
		default:
			throw InternalException("Unsupported internal enum type");
		}
		break;
	case LogicalTypeId::INTERVAL:
		InitializeFunctionPointers<ArrowScalarData<ArrowInterval, interval_t, ArrowIntervalConverter>>(append_data);
		break;
	case LogicalTypeId::STRUCT:
		InitializeFunctionPointers<ArrowStructData>(append_data);
		break;
	case LogicalTypeId::LIST:
		InitializeFunctionPointers<ArrowListData>(append_data);
		break;
	case LogicalTypeId::MAP:
		InitializeFunctionPointers<ArrowMapData>(append_data);
		break;
	default:
		throw InternalException("Unsupported type in DuckDB -> Arrow Conversion: %s\n", type.ToString());
	}
}

unique_ptr<ArrowAppendData> InitializeArrowChild(const LogicalType &type, idx_t capacity, ArrowOptions &options) {
	auto result = make_uniq<ArrowAppendData>(options);
	InitializeFunctionPointers(*result, type);

	auto byte_count = (capacity + 7) / 8;
	result->validity.reserve(byte_count);
	result->initialize(*result, type, capacity);
	return result;
}

static void ReleaseDuckDBArrowAppendArray(ArrowArray *array) {
	if (!array || !array->release) {
		return;
	}
	array->release = nullptr;
	auto holder = static_cast<ArrowAppendData *>(array->private_data);
	delete holder;
}

//===--------------------------------------------------------------------===//
// Finalize Arrow Child
//===--------------------------------------------------------------------===//
ArrowArray *FinalizeArrowChild(const LogicalType &type, ArrowAppendData &append_data) {
	auto result = make_uniq<ArrowArray>();

	result->private_data = nullptr;
	result->release = ReleaseDuckDBArrowAppendArray;
	result->n_children = 0;
	result->null_count = 0;
	result->offset = 0;
	result->dictionary = nullptr;
	result->buffers = append_data.buffers.data();
	result->null_count = append_data.null_count;
	result->length = append_data.row_count;
	result->buffers[0] = append_data.validity.data();

	if (append_data.finalize) {
		append_data.finalize(append_data, type, result.get());
	}

	append_data.array = std::move(result);
	return append_data.array.get();
}

//! Returns the underlying arrow array
ArrowArray ArrowAppender::Finalize() {
	D_ASSERT(root_data.size() == types.size());
	auto root_holder = make_uniq<ArrowAppendData>(options);

	ArrowArray result;
	root_holder->child_pointers.resize(types.size());
	result.children = root_holder->child_pointers.data();
	result.n_children = types.size();

	// Configure root array
	result.length = row_count;
	result.n_children = types.size();
	result.n_buffers = 1;
	result.buffers = root_holder->buffers.data(); // there is no actual buffer there since we don't have NULLs
	result.offset = 0;
	result.null_count = 0; // needs to be 0
	result.dictionary = nullptr;
	root_holder->child_data = std::move(root_data);

	for (idx_t i = 0; i < root_holder->child_data.size(); i++) {
		root_holder->child_pointers[i] = FinalizeArrowChild(types[i], *root_holder->child_data[i]);
	}

	// Release ownership to caller
	result.private_data = root_holder.release();
	result.release = ReleaseDuckDBArrowAppendArray;
	return result;
}

} // namespace duckdb












#include <list>


namespace duckdb {

void ArrowConverter::ToArrowArray(DataChunk &input, ArrowArray *out_array, ArrowOptions options) {
	ArrowAppender appender(input.GetTypes(), input.size(), options);
	appender.Append(input, 0, input.size(), input.size());
	*out_array = appender.Finalize();
}

//===--------------------------------------------------------------------===//
// Arrow Schema
//===--------------------------------------------------------------------===//
struct DuckDBArrowSchemaHolder {
	// unused in children
	vector<ArrowSchema> children;
	// unused in children
	vector<ArrowSchema *> children_ptrs;
	//! used for nested structures
	std::list<vector<ArrowSchema>> nested_children;
	std::list<vector<ArrowSchema *>> nested_children_ptr;
	//! This holds strings created to represent decimal types
	vector<unsafe_unique_array<char>> owned_type_names;
};

static void ReleaseDuckDBArrowSchema(ArrowSchema *schema) {
	if (!schema || !schema->release) {
		return;
	}
	schema->release = nullptr;
	auto holder = static_cast<DuckDBArrowSchemaHolder *>(schema->private_data);
	delete holder;
}

void InitializeChild(ArrowSchema &child, const string &name = "") {
	//! Child is cleaned up by parent
	child.private_data = nullptr;
	child.release = ReleaseDuckDBArrowSchema;

	//! Store the child schema
	child.flags = ARROW_FLAG_NULLABLE;
	child.name = name.c_str();
	child.n_children = 0;
	child.children = nullptr;
	child.metadata = nullptr;
	child.dictionary = nullptr;
}
void SetArrowFormat(DuckDBArrowSchemaHolder &root_holder, ArrowSchema &child, const LogicalType &type,
                    const string &config_timezone, ArrowOptions options);

void SetArrowMapFormat(DuckDBArrowSchemaHolder &root_holder, ArrowSchema &child, const LogicalType &type,
                       const string &config_timezone, ArrowOptions options) {
	child.format = "+m";
	//! Map has one child which is a struct
	child.n_children = 1;
	root_holder.nested_children.emplace_back();
	root_holder.nested_children.back().resize(1);
	root_holder.nested_children_ptr.emplace_back();
	root_holder.nested_children_ptr.back().push_back(&root_holder.nested_children.back()[0]);
	InitializeChild(root_holder.nested_children.back()[0]);
	child.children = &root_holder.nested_children_ptr.back()[0];
	child.children[0]->name = "entries";
	SetArrowFormat(root_holder, **child.children, ListType::GetChildType(type), config_timezone, options);
}

void SetArrowFormat(DuckDBArrowSchemaHolder &root_holder, ArrowSchema &child, const LogicalType &type,
                    const string &config_timezone, ArrowOptions options) {
	switch (type.id()) {
	case LogicalTypeId::BOOLEAN:
		child.format = "b";
		break;
	case LogicalTypeId::TINYINT:
		child.format = "c";
		break;
	case LogicalTypeId::SMALLINT:
		child.format = "s";
		break;
	case LogicalTypeId::INTEGER:
		child.format = "i";
		break;
	case LogicalTypeId::BIGINT:
		child.format = "l";
		break;
	case LogicalTypeId::UTINYINT:
		child.format = "C";
		break;
	case LogicalTypeId::USMALLINT:
		child.format = "S";
		break;
	case LogicalTypeId::UINTEGER:
		child.format = "I";
		break;
	case LogicalTypeId::UBIGINT:
		child.format = "L";
		break;
	case LogicalTypeId::FLOAT:
		child.format = "f";
		break;
	case LogicalTypeId::HUGEINT:
		child.format = "d:38,0";
		break;
	case LogicalTypeId::DOUBLE:
		child.format = "g";
		break;
	case LogicalTypeId::UUID:
	case LogicalTypeId::VARCHAR:
		if (options.offset_size == ArrowOffsetSize::LARGE) {
			child.format = "U";
		} else {
			child.format = "u";
		}
		break;
	case LogicalTypeId::DATE:
		child.format = "tdD";
		break;
	case LogicalTypeId::TIME:
	case LogicalTypeId::TIME_TZ:
		child.format = "ttu";
		break;
	case LogicalTypeId::TIMESTAMP:
		child.format = "tsu:";
		break;
	case LogicalTypeId::TIMESTAMP_TZ: {
		string format = "tsu:" + config_timezone;
		auto format_ptr = make_unsafe_uniq_array<char>(format.size() + 1);
		for (size_t i = 0; i < format.size(); i++) {
			format_ptr[i] = format[i];
		}
		format_ptr[format.size()] = '\0';
		root_holder.owned_type_names.push_back(std::move(format_ptr));
		child.format = root_holder.owned_type_names.back().get();
		break;
	}
	case LogicalTypeId::TIMESTAMP_SEC:
		child.format = "tss:";
		break;
	case LogicalTypeId::TIMESTAMP_NS:
		child.format = "tsn:";
		break;
	case LogicalTypeId::TIMESTAMP_MS:
		child.format = "tsm:";
		break;
	case LogicalTypeId::INTERVAL:
		child.format = "tin";
		break;
	case LogicalTypeId::DECIMAL: {
		uint8_t width, scale;
		type.GetDecimalProperties(width, scale);
		string format = "d:" + to_string(width) + "," + to_string(scale);
		auto format_ptr = make_unsafe_uniq_array<char>(format.size() + 1);
		for (size_t i = 0; i < format.size(); i++) {
			format_ptr[i] = format[i];
		}
		format_ptr[format.size()] = '\0';
		root_holder.owned_type_names.push_back(std::move(format_ptr));
		child.format = root_holder.owned_type_names.back().get();
		break;
	}
	case LogicalTypeId::SQLNULL: {
		child.format = "n";
		break;
	}
	case LogicalTypeId::BLOB:
	case LogicalTypeId::BIT: {
		if (options.offset_size == ArrowOffsetSize::LARGE) {
			child.format = "Z";
		} else {
			child.format = "z";
		}
		break;
	}
	case LogicalTypeId::LIST: {
		child.format = "+l";
		child.n_children = 1;
		root_holder.nested_children.emplace_back();
		root_holder.nested_children.back().resize(1);
		root_holder.nested_children_ptr.emplace_back();
		root_holder.nested_children_ptr.back().push_back(&root_holder.nested_children.back()[0]);
		InitializeChild(root_holder.nested_children.back()[0]);
		child.children = &root_holder.nested_children_ptr.back()[0];
		child.children[0]->name = "l";
		SetArrowFormat(root_holder, **child.children, ListType::GetChildType(type), config_timezone, options);
		break;
	}
	case LogicalTypeId::STRUCT: {
		child.format = "+s";
		auto &child_types = StructType::GetChildTypes(type);
		child.n_children = child_types.size();
		root_holder.nested_children.emplace_back();
		root_holder.nested_children.back().resize(child_types.size());
		root_holder.nested_children_ptr.emplace_back();
		root_holder.nested_children_ptr.back().resize(child_types.size());
		for (idx_t type_idx = 0; type_idx < child_types.size(); type_idx++) {
			root_holder.nested_children_ptr.back()[type_idx] = &root_holder.nested_children.back()[type_idx];
		}
		child.children = &root_holder.nested_children_ptr.back()[0];
		for (size_t type_idx = 0; type_idx < child_types.size(); type_idx++) {

			InitializeChild(*child.children[type_idx]);

			auto &struct_col_name = child_types[type_idx].first;
			auto name_ptr = make_unsafe_uniq_array<char>(struct_col_name.size() + 1);
			for (size_t i = 0; i < struct_col_name.size(); i++) {
				name_ptr[i] = struct_col_name[i];
			}
			name_ptr[struct_col_name.size()] = '\0';
			root_holder.owned_type_names.push_back(std::move(name_ptr));

			child.children[type_idx]->name = root_holder.owned_type_names.back().get();
			SetArrowFormat(root_holder, *child.children[type_idx], child_types[type_idx].second, config_timezone,
			               options);
		}
		break;
	}
	case LogicalTypeId::MAP: {
		SetArrowMapFormat(root_holder, child, type, config_timezone, options);
		break;
	}
	case LogicalTypeId::ENUM: {
		// TODO what do we do with pointer enums here?
		switch (EnumType::GetPhysicalType(type)) {
		case PhysicalType::UINT8:
			child.format = "C";
			break;
		case PhysicalType::UINT16:
			child.format = "S";
			break;
		case PhysicalType::UINT32:
			child.format = "I";
			break;
		default:
			throw InternalException("Unsupported Enum Internal Type");
		}
		root_holder.nested_children.emplace_back();
		root_holder.nested_children.back().resize(1);
		root_holder.nested_children_ptr.emplace_back();
		root_holder.nested_children_ptr.back().push_back(&root_holder.nested_children.back()[0]);
		InitializeChild(root_holder.nested_children.back()[0]);
		child.dictionary = root_holder.nested_children_ptr.back()[0];
		child.dictionary->format = "u";
		break;
	}
	default:
		throw InternalException("Unsupported Arrow type " + type.ToString());
	}
}

void ArrowConverter::ToArrowSchema(ArrowSchema *out_schema, const vector<LogicalType> &types,
                                   const vector<string> &names, const string &config_timezone, ArrowOptions options) {
	D_ASSERT(out_schema);
	D_ASSERT(types.size() == names.size());
	idx_t column_count = types.size();
	// Allocate as unique_ptr first to cleanup properly on error
	auto root_holder = make_uniq<DuckDBArrowSchemaHolder>();

	// Allocate the children
	root_holder->children.resize(column_count);
	root_holder->children_ptrs.resize(column_count, nullptr);
	for (size_t i = 0; i < column_count; ++i) {
		root_holder->children_ptrs[i] = &root_holder->children[i];
	}
	out_schema->children = root_holder->children_ptrs.data();
	out_schema->n_children = column_count;

	// Store the schema
	out_schema->format = "+s"; // struct apparently
	out_schema->flags = 0;
	out_schema->metadata = nullptr;
	out_schema->name = "duckdb_query_result";
	out_schema->dictionary = nullptr;

	// Configure all child schemas
	for (idx_t col_idx = 0; col_idx < column_count; col_idx++) {

		auto &child = root_holder->children[col_idx];
		InitializeChild(child, names[col_idx]);
		SetArrowFormat(*root_holder, child, types[col_idx], config_timezone, options);
	}

	// Release ownership to caller
	out_schema->private_data = root_holder.release();
	out_schema->release = ReleaseDuckDBArrowSchema;
}

} // namespace duckdb












namespace duckdb {

ArrowSchemaWrapper::~ArrowSchemaWrapper() {
	if (arrow_schema.release) {
		for (int64_t child_idx = 0; child_idx < arrow_schema.n_children; child_idx++) {
			auto &child = *arrow_schema.children[child_idx];
			if (child.release) {
				child.release(&child);
			}
		}
		arrow_schema.release(&arrow_schema);
		arrow_schema.release = nullptr;
	}
}

ArrowArrayWrapper::~ArrowArrayWrapper() {
	if (arrow_array.release) {
		for (int64_t child_idx = 0; child_idx < arrow_array.n_children; child_idx++) {
			auto &child = *arrow_array.children[child_idx];
			if (child.release) {
				child.release(&child);
			}
		}
		arrow_array.release(&arrow_array);
		arrow_array.release = nullptr;
	}
}

ArrowArrayStreamWrapper::~ArrowArrayStreamWrapper() {
	if (arrow_array_stream.release) {
		arrow_array_stream.release(&arrow_array_stream);
		arrow_array_stream.release = nullptr;
	}
}

void ArrowArrayStreamWrapper::GetSchema(ArrowSchemaWrapper &schema) {
	D_ASSERT(arrow_array_stream.get_schema);
	// LCOV_EXCL_START
	if (arrow_array_stream.get_schema(&arrow_array_stream, &schema.arrow_schema)) {
		throw InvalidInputException("arrow_scan: get_schema failed(): %s", string(GetError()));
	}
	if (!schema.arrow_schema.release) {
		throw InvalidInputException("arrow_scan: released schema passed");
	}
	if (schema.arrow_schema.n_children < 1) {
		throw InvalidInputException("arrow_scan: empty schema passed");
	}
	// LCOV_EXCL_STOP
}

shared_ptr<ArrowArrayWrapper> ArrowArrayStreamWrapper::GetNextChunk() {
	auto current_chunk = make_shared<ArrowArrayWrapper>();
	if (arrow_array_stream.get_next(&arrow_array_stream, &current_chunk->arrow_array)) { // LCOV_EXCL_START
		throw InvalidInputException("arrow_scan: get_next failed(): %s", string(GetError()));
	} // LCOV_EXCL_STOP

	return current_chunk;
}

const char *ArrowArrayStreamWrapper::GetError() { // LCOV_EXCL_START
	return arrow_array_stream.get_last_error(&arrow_array_stream);
} // LCOV_EXCL_STOP

int ResultArrowArrayStreamWrapper::MyStreamGetSchema(struct ArrowArrayStream *stream, struct ArrowSchema *out) {
	if (!stream->release) {
		return -1;
	}
	auto my_stream = (ResultArrowArrayStreamWrapper *)stream->private_data;
	if (!my_stream->column_types.empty()) {
		ArrowConverter::ToArrowSchema(out, my_stream->column_types, my_stream->column_names,
		                              my_stream->timezone_config);
		return 0;
	}

	auto &result = *my_stream->result;
	if (result.HasError()) {
		my_stream->last_error = result.GetErrorObject();
		return -1;
	}
	if (result.type == QueryResultType::STREAM_RESULT) {
		auto &stream_result = (StreamQueryResult &)result;
		if (!stream_result.IsOpen()) {
			my_stream->last_error = PreservedError("Query Stream is closed");
			return -1;
		}
	}
	if (my_stream->column_types.empty()) {
		my_stream->column_types = result.types;
		my_stream->column_names = result.names;
	}
	ArrowConverter::ToArrowSchema(out, my_stream->column_types, my_stream->column_names, my_stream->timezone_config);
	return 0;
}

int ResultArrowArrayStreamWrapper::MyStreamGetNext(struct ArrowArrayStream *stream, struct ArrowArray *out) {
	if (!stream->release) {
		return -1;
	}
	auto my_stream = (ResultArrowArrayStreamWrapper *)stream->private_data;
	auto &result = *my_stream->result;
	if (result.HasError()) {
		my_stream->last_error = result.GetErrorObject();
		return -1;
	}
	if (result.type == QueryResultType::STREAM_RESULT) {
		auto &stream_result = (StreamQueryResult &)result;
		if (!stream_result.IsOpen()) {
			// Nothing to output
			out->release = nullptr;
			return 0;
		}
	}
	if (my_stream->column_types.empty()) {
		my_stream->column_types = result.types;
		my_stream->column_names = result.names;
	}
	idx_t result_count;
	PreservedError error;
	if (!ArrowUtil::TryFetchChunk(&result, my_stream->batch_size, out, result_count, error)) {
		D_ASSERT(error);
		my_stream->last_error = error;
		return -1;
	}
	if (result_count == 0) {
		// Nothing to output
		out->release = nullptr;
	}
	return 0;
}

void ResultArrowArrayStreamWrapper::MyStreamRelease(struct ArrowArrayStream *stream) {
	if (!stream || !stream->release) {
		return;
	}
	stream->release = nullptr;
	delete (ResultArrowArrayStreamWrapper *)stream->private_data;
}

const char *ResultArrowArrayStreamWrapper::MyStreamGetLastError(struct ArrowArrayStream *stream) {
	if (!stream->release) {
		return "stream was released";
	}
	D_ASSERT(stream->private_data);
	auto my_stream = (ResultArrowArrayStreamWrapper *)stream->private_data;
	return my_stream->last_error.Message().c_str();
}

ResultArrowArrayStreamWrapper::ResultArrowArrayStreamWrapper(unique_ptr<QueryResult> result_p, idx_t batch_size_p)
    : result(std::move(result_p)) {
	//! We first initialize the private data of the stream
	stream.private_data = this;
	//! Ceil Approx_Batch_Size/STANDARD_VECTOR_SIZE
	if (batch_size_p == 0) {
		throw std::runtime_error("Approximate Batch Size of Record Batch MUST be higher than 0");
	}
	batch_size = batch_size_p;
	//! We initialize the stream functions
	stream.get_schema = ResultArrowArrayStreamWrapper::MyStreamGetSchema;
	stream.get_next = ResultArrowArrayStreamWrapper::MyStreamGetNext;
	stream.release = ResultArrowArrayStreamWrapper::MyStreamRelease;
	stream.get_last_error = ResultArrowArrayStreamWrapper::MyStreamGetLastError;
}

bool ArrowUtil::TryFetchNext(QueryResult &result, unique_ptr<DataChunk> &chunk, PreservedError &error) {
	if (result.type == QueryResultType::STREAM_RESULT) {
		auto &stream_result = (StreamQueryResult &)result;
		if (!stream_result.IsOpen()) {
			return true;
		}
	}
	return result.TryFetch(chunk, error);
}

bool ArrowUtil::TryFetchChunk(QueryResult *result, idx_t chunk_size, ArrowArray *out, idx_t &count,
                              PreservedError &error) {
	count = 0;
	ArrowAppender appender(result->types, chunk_size);
	auto &current_chunk = result->current_chunk;
	if (current_chunk.Valid()) {
		// We start by scanning the non-finished current chunk
		idx_t cur_consumption = current_chunk.RemainingSize() > chunk_size ? chunk_size : current_chunk.RemainingSize();
		count += cur_consumption;
		appender.Append(*current_chunk.data_chunk, current_chunk.position, current_chunk.position + cur_consumption,
		                current_chunk.data_chunk->size());
		current_chunk.position += cur_consumption;
	}
	while (count < chunk_size) {
		unique_ptr<DataChunk> data_chunk;
		if (!TryFetchNext(*result, data_chunk, error)) {
			if (result->HasError()) {
				error = result->GetErrorObject();
			}
			return false;
		}
		if (!data_chunk || data_chunk->size() == 0) {
			break;
		}
		if (count + data_chunk->size() > chunk_size) {
			// We have to split the chunk between this and the next batch
			idx_t available_space = chunk_size - count;
			appender.Append(*data_chunk, 0, available_space, data_chunk->size());
			count += available_space;
			current_chunk.data_chunk = std::move(data_chunk);
			current_chunk.position = available_space;
		} else {
			count += data_chunk->size();
			appender.Append(*data_chunk, 0, data_chunk->size(), data_chunk->size());
		}
	}
	if (count > 0) {
		*out = appender.Finalize();
	}
	return true;
}

idx_t ArrowUtil::FetchChunk(QueryResult *result, idx_t chunk_size, ArrowArray *out) {
	PreservedError error;
	idx_t result_count;
	if (!TryFetchChunk(result, chunk_size, out, result_count, error)) {
		error.Throw();
	}
	return result_count;
}

} // namespace duckdb



namespace duckdb {

void DuckDBAssertInternal(bool condition, const char *condition_name, const char *file, int linenr) {
	if (condition) {
		return;
	}
	throw InternalException("Assertion triggered in file \"%s\" on line %d: %s%s", file, linenr, condition_name,
	                        Exception::GetStackTrace());
}

} // namespace duckdb






#include <numeric>

namespace duckdb {

Value ConvertVectorToValue(vector<Value> set) {
	if (set.empty()) {
		return Value::EMPTYLIST(LogicalType::BOOLEAN);
	}
	return Value::LIST(std::move(set));
}

vector<bool> ParseColumnList(const vector<Value> &set, vector<string> &names, const string &loption) {
	vector<bool> result;

	if (set.empty()) {
		throw BinderException("\"%s\" expects a column list or * as parameter", loption);
	}
	// list of options: parse the list
	case_insensitive_map_t<bool> option_map;
	for (idx_t i = 0; i < set.size(); i++) {
		option_map[set[i].ToString()] = false;
	}
	result.resize(names.size(), false);
	for (idx_t i = 0; i < names.size(); i++) {
		auto entry = option_map.find(names[i]);
		if (entry != option_map.end()) {
			result[i] = true;
			entry->second = true;
		}
	}
	for (auto &entry : option_map) {
		if (!entry.second) {
			throw BinderException("\"%s\" expected to find %s, but it was not found in the table", loption,
			                      entry.first.c_str());
		}
	}
	return result;
}

vector<bool> ParseColumnList(const Value &value, vector<string> &names, const string &loption) {
	vector<bool> result;

	// Only accept a list of arguments
	if (value.type().id() != LogicalTypeId::LIST) {
		// Support a single argument if it's '*'
		if (value.type().id() == LogicalTypeId::VARCHAR && value.GetValue<string>() == "*") {
			result.resize(names.size(), true);
			return result;
		}
		throw BinderException("\"%s\" expects a column list or * as parameter", loption);
	}
	auto &children = ListValue::GetChildren(value);
	// accept '*' as single argument
	if (children.size() == 1 && children[0].type().id() == LogicalTypeId::VARCHAR &&
	    children[0].GetValue<string>() == "*") {
		result.resize(names.size(), true);
		return result;
	}
	return ParseColumnList(children, names, loption);
}

vector<idx_t> ParseColumnsOrdered(const vector<Value> &set, vector<string> &names, const string &loption) {
	vector<idx_t> result;

	if (set.empty()) {
		throw BinderException("\"%s\" expects a column list or * as parameter", loption);
	}

	// Maps option to bool indicating if its found and the index in the original set
	case_insensitive_map_t<std::pair<bool, idx_t>> option_map;
	for (idx_t i = 0; i < set.size(); i++) {
		option_map[set[i].ToString()] = {false, i};
	}
	result.resize(option_map.size());

	for (idx_t i = 0; i < names.size(); i++) {
		auto entry = option_map.find(names[i]);
		if (entry != option_map.end()) {
			result[entry->second.second] = i;
			entry->second.first = true;
		}
	}
	for (auto &entry : option_map) {
		if (!entry.second.first) {
			throw BinderException("\"%s\" expected to find %s, but it was not found in the table", loption,
			                      entry.first.c_str());
		}
	}
	return result;
}

vector<idx_t> ParseColumnsOrdered(const Value &value, vector<string> &names, const string &loption) {
	vector<idx_t> result;

	// Only accept a list of arguments
	if (value.type().id() != LogicalTypeId::LIST) {
		// Support a single argument if it's '*'
		if (value.type().id() == LogicalTypeId::VARCHAR && value.GetValue<string>() == "*") {
			result.resize(names.size(), 0);
			std::iota(std::begin(result), std::end(result), 0);
			return result;
		}
		throw BinderException("\"%s\" expects a column list or * as parameter", loption);
	}
	auto &children = ListValue::GetChildren(value);
	// accept '*' as single argument
	if (children.size() == 1 && children[0].type().id() == LogicalTypeId::VARCHAR &&
	    children[0].GetValue<string>() == "*") {
		result.resize(names.size(), 0);
		std::iota(std::begin(result), std::end(result), 0);
		return result;
	}
	return ParseColumnsOrdered(children, names, loption);
}

} // namespace duckdb







#include <sstream>

namespace duckdb {

const idx_t BoxRenderer::SPLIT_COLUMN = idx_t(-1);

BoxRenderer::BoxRenderer(BoxRendererConfig config_p) : config(std::move(config_p)) {
}

string BoxRenderer::ToString(ClientContext &context, const vector<string> &names, const ColumnDataCollection &result) {
	std::stringstream ss;
	Render(context, names, result, ss);
	return ss.str();
}

void BoxRenderer::Print(ClientContext &context, const vector<string> &names, const ColumnDataCollection &result) {
	Printer::Print(ToString(context, names, result));
}

void BoxRenderer::RenderValue(std::ostream &ss, const string &value, idx_t column_width,
                              ValueRenderAlignment alignment) {
	auto render_width = Utf8Proc::RenderWidth(value);

	const string *render_value = &value;
	string small_value;
	if (render_width > column_width) {
		// the string is too large to fit in this column!
		// the size of this column must have been reduced
		// figure out how much of this value we can render
		idx_t pos = 0;
		idx_t current_render_width = config.DOTDOTDOT_LENGTH;
		while (pos < value.size()) {
			// check if this character fits...
			auto char_size = Utf8Proc::RenderWidth(value.c_str(), value.size(), pos);
			if (current_render_width + char_size >= column_width) {
				// it doesn't! stop
				break;
			}
			// it does! move to the next character
			current_render_width += char_size;
			pos = Utf8Proc::NextGraphemeCluster(value.c_str(), value.size(), pos);
		}
		small_value = value.substr(0, pos) + config.DOTDOTDOT;
		render_value = &small_value;
		render_width = current_render_width;
	}
	auto padding_count = (column_width - render_width) + 2;
	idx_t lpadding;
	idx_t rpadding;
	switch (alignment) {
	case ValueRenderAlignment::LEFT:
		lpadding = 1;
		rpadding = padding_count - 1;
		break;
	case ValueRenderAlignment::MIDDLE:
		lpadding = padding_count / 2;
		rpadding = padding_count - lpadding;
		break;
	case ValueRenderAlignment::RIGHT:
		lpadding = padding_count - 1;
		rpadding = 1;
		break;
	default:
		throw InternalException("Unrecognized value renderer alignment");
	}
	ss << config.VERTICAL;
	ss << string(lpadding, ' ');
	ss << *render_value;
	ss << string(rpadding, ' ');
}

string BoxRenderer::RenderType(const LogicalType &type) {
	switch (type.id()) {
	case LogicalTypeId::TINYINT:
		return "int8";
	case LogicalTypeId::SMALLINT:
		return "int16";
	case LogicalTypeId::INTEGER:
		return "int32";
	case LogicalTypeId::BIGINT:
		return "int64";
	case LogicalTypeId::HUGEINT:
		return "int128";
	case LogicalTypeId::UTINYINT:
		return "uint8";
	case LogicalTypeId::USMALLINT:
		return "uint16";
	case LogicalTypeId::UINTEGER:
		return "uint32";
	case LogicalTypeId::UBIGINT:
		return "uint64";
	case LogicalTypeId::LIST: {
		auto child = RenderType(ListType::GetChildType(type));
		return child + "[]";
	}
	default:
		return StringUtil::Lower(type.ToString());
	}
}

ValueRenderAlignment BoxRenderer::TypeAlignment(const LogicalType &type) {
	switch (type.id()) {
	case LogicalTypeId::TINYINT:
	case LogicalTypeId::SMALLINT:
	case LogicalTypeId::INTEGER:
	case LogicalTypeId::BIGINT:
	case LogicalTypeId::HUGEINT:
	case LogicalTypeId::UTINYINT:
	case LogicalTypeId::USMALLINT:
	case LogicalTypeId::UINTEGER:
	case LogicalTypeId::UBIGINT:
	case LogicalTypeId::DECIMAL:
	case LogicalTypeId::FLOAT:
	case LogicalTypeId::DOUBLE:
		return ValueRenderAlignment::RIGHT;
	default:
		return ValueRenderAlignment::LEFT;
	}
}

list<ColumnDataCollection> BoxRenderer::FetchRenderCollections(ClientContext &context,
                                                               const ColumnDataCollection &result, idx_t top_rows,
                                                               idx_t bottom_rows) {
	auto column_count = result.ColumnCount();
	vector<LogicalType> varchar_types;
	for (idx_t c = 0; c < column_count; c++) {
		varchar_types.emplace_back(LogicalType::VARCHAR);
	}
	std::list<ColumnDataCollection> collections;
	collections.emplace_back(context, varchar_types);
	collections.emplace_back(context, varchar_types);

	auto &top_collection = collections.front();
	auto &bottom_collection = collections.back();

	DataChunk fetch_result;
	fetch_result.Initialize(context, result.Types());

	DataChunk insert_result;
	insert_result.Initialize(context, varchar_types);

	// fetch the top rows from the ColumnDataCollection
	idx_t chunk_idx = 0;
	idx_t row_idx = 0;
	while (row_idx < top_rows) {
		fetch_result.Reset();
		insert_result.Reset();
		// fetch the next chunk
		result.FetchChunk(chunk_idx, fetch_result);
		idx_t insert_count = MinValue<idx_t>(fetch_result.size(), top_rows - row_idx);

		// cast all columns to varchar
		for (idx_t c = 0; c < column_count; c++) {
			VectorOperations::Cast(context, fetch_result.data[c], insert_result.data[c], insert_count);
		}
		insert_result.SetCardinality(insert_count);

		// construct the render collection
		top_collection.Append(insert_result);

		chunk_idx++;
		row_idx += fetch_result.size();
	}

	// fetch the bottom rows from the ColumnDataCollection
	row_idx = 0;
	chunk_idx = result.ChunkCount() - 1;
	while (row_idx < bottom_rows) {
		fetch_result.Reset();
		insert_result.Reset();
		// fetch the next chunk
		result.FetchChunk(chunk_idx, fetch_result);
		idx_t insert_count = MinValue<idx_t>(fetch_result.size(), bottom_rows - row_idx);

		// invert the rows
		SelectionVector inverted_sel(insert_count);
		for (idx_t r = 0; r < insert_count; r++) {
			inverted_sel.set_index(r, fetch_result.size() - r - 1);
		}

		for (idx_t c = 0; c < column_count; c++) {
			Vector slice(fetch_result.data[c], inverted_sel, insert_count);
			VectorOperations::Cast(context, slice, insert_result.data[c], insert_count);
		}
		insert_result.SetCardinality(insert_count);
		// construct the render collection
		bottom_collection.Append(insert_result);

		chunk_idx--;
		row_idx += fetch_result.size();
	}
	return collections;
}

list<ColumnDataCollection> BoxRenderer::PivotCollections(ClientContext &context, list<ColumnDataCollection> input,
                                                         vector<string> &column_names,
                                                         vector<LogicalType> &result_types, idx_t row_count) {
	auto &top = input.front();
	auto &bottom = input.back();

	vector<LogicalType> varchar_types;
	vector<string> new_names;
	new_names.emplace_back("Column");
	new_names.emplace_back("Type");
	varchar_types.emplace_back(LogicalType::VARCHAR);
	varchar_types.emplace_back(LogicalType::VARCHAR);
	for (idx_t r = 0; r < top.Count(); r++) {
		new_names.emplace_back("Row " + to_string(r + 1));
		varchar_types.emplace_back(LogicalType::VARCHAR);
	}
	for (idx_t r = 0; r < bottom.Count(); r++) {
		auto row_index = row_count - bottom.Count() + r + 1;
		new_names.emplace_back("Row " + to_string(row_index));
		varchar_types.emplace_back(LogicalType::VARCHAR);
	}
	//
	DataChunk row_chunk;
	row_chunk.Initialize(Allocator::DefaultAllocator(), varchar_types);
	std::list<ColumnDataCollection> result;
	result.emplace_back(context, varchar_types);
	result.emplace_back(context, varchar_types);
	auto &res_coll = result.front();
	ColumnDataAppendState append_state;
	res_coll.InitializeAppend(append_state);
	for (idx_t c = 0; c < top.ColumnCount(); c++) {
		vector<column_t> column_ids {c};
		auto row_index = row_chunk.size();
		idx_t current_index = 0;
		row_chunk.SetValue(current_index++, row_index, column_names[c]);
		row_chunk.SetValue(current_index++, row_index, RenderType(result_types[c]));
		for (auto &collection : input) {
			for (auto &chunk : collection.Chunks(column_ids)) {
				for (idx_t r = 0; r < chunk.size(); r++) {
					row_chunk.SetValue(current_index++, row_index, chunk.GetValue(0, r));
				}
			}
		}
		row_chunk.SetCardinality(row_chunk.size() + 1);
		if (row_chunk.size() == STANDARD_VECTOR_SIZE || c + 1 == top.ColumnCount()) {
			res_coll.Append(append_state, row_chunk);
			row_chunk.Reset();
		}
	}
	column_names = std::move(new_names);
	result_types = std::move(varchar_types);
	return result;
}

string ConvertRenderValue(const string &input) {
	return StringUtil::Replace(StringUtil::Replace(input, "\n", "\\n"), string("\0", 1), "\\0");
}

string BoxRenderer::GetRenderValue(ColumnDataRowCollection &rows, idx_t c, idx_t r) {
	try {
		auto row = rows.GetValue(c, r);
		if (row.IsNull()) {
			return config.null_value;
		}
		return ConvertRenderValue(StringValue::Get(row));
	} catch (std::exception &ex) {
		return "????INVALID VALUE - " + string(ex.what()) + "?????";
	}
}

vector<idx_t> BoxRenderer::ComputeRenderWidths(const vector<string> &names, const vector<LogicalType> &result_types,
                                               list<ColumnDataCollection> &collections, idx_t min_width,
                                               idx_t max_width, vector<idx_t> &column_map, idx_t &total_length) {
	auto column_count = result_types.size();

	vector<idx_t> widths;
	widths.reserve(column_count);
	for (idx_t c = 0; c < column_count; c++) {
		auto name_width = Utf8Proc::RenderWidth(ConvertRenderValue(names[c]));
		auto type_width = Utf8Proc::RenderWidth(RenderType(result_types[c]));
		widths.push_back(MaxValue<idx_t>(name_width, type_width));
	}

	// now iterate over the data in the render collection and find out the true max width
	for (auto &collection : collections) {
		for (auto &chunk : collection.Chunks()) {
			for (idx_t c = 0; c < column_count; c++) {
				auto string_data = FlatVector::GetData<string_t>(chunk.data[c]);
				for (idx_t r = 0; r < chunk.size(); r++) {
					string render_value;
					if (FlatVector::IsNull(chunk.data[c], r)) {
						render_value = config.null_value;
					} else {
						render_value = ConvertRenderValue(string_data[r].GetString());
					}
					auto render_width = Utf8Proc::RenderWidth(render_value);
					widths[c] = MaxValue<idx_t>(render_width, widths[c]);
				}
			}
		}
	}

	// figure out the total length
	// we start off with a pipe (|)
	total_length = 1;
	for (idx_t c = 0; c < widths.size(); c++) {
		// each column has a space at the beginning, and a space plus a pipe (|) at the end
		// hence + 3
		total_length += widths[c] + 3;
	}
	if (total_length < min_width) {
		// if there are hidden rows we should always display that
		// stretch up the first column until we have space to show the row count
		widths[0] += min_width - total_length;
		total_length = min_width;
	}
	// now we need to constrain the length
	unordered_set<idx_t> pruned_columns;
	if (total_length > max_width) {
		// before we remove columns, check if we can just reduce the size of columns
		for (auto &w : widths) {
			if (w > config.max_col_width) {
				auto max_diff = w - config.max_col_width;
				if (total_length - max_diff <= max_width) {
					// if we reduce the size of this column we fit within the limits!
					// reduce the width exactly enough so that the box fits
					w -= total_length - max_width;
					total_length = max_width;
					break;
				} else {
					// reducing the width of this column does not make the result fit
					// reduce the column width by the maximum amount anyway
					w = config.max_col_width;
					total_length -= max_diff;
				}
			}
		}

		if (total_length > max_width) {
			// the total length is still too large
			// we need to remove columns!
			// first, we add 6 characters to the total length
			// this is what we need to add the "..." in the middle
			total_length += 3 + config.DOTDOTDOT_LENGTH;
			// now select columns to prune
			// we select columns in zig-zag order starting from the middle
			// e.g. if we have 10 columns, we remove #5, then #4, then #6, then #3, then #7, etc
			int64_t offset = 0;
			while (total_length > max_width) {
				idx_t c = column_count / 2 + offset;
				total_length -= widths[c] + 3;
				pruned_columns.insert(c);
				if (offset >= 0) {
					offset = -offset - 1;
				} else {
					offset = -offset;
				}
			}
		}
	}

	bool added_split_column = false;
	vector<idx_t> new_widths;
	for (idx_t c = 0; c < column_count; c++) {
		if (pruned_columns.find(c) == pruned_columns.end()) {
			column_map.push_back(c);
			new_widths.push_back(widths[c]);
		} else {
			if (!added_split_column) {
				// "..."
				column_map.push_back(SPLIT_COLUMN);
				new_widths.push_back(config.DOTDOTDOT_LENGTH);
				added_split_column = true;
			}
		}
	}
	return new_widths;
}

void BoxRenderer::RenderHeader(const vector<string> &names, const vector<LogicalType> &result_types,
                               const vector<idx_t> &column_map, const vector<idx_t> &widths,
                               const vector<idx_t> &boundaries, idx_t total_length, bool has_results,
                               std::ostream &ss) {
	auto column_count = column_map.size();
	// render the top line
	ss << config.LTCORNER;
	idx_t column_index = 0;
	for (idx_t k = 0; k < total_length - 2; k++) {
		if (column_index + 1 < column_count && k == boundaries[column_index]) {
			ss << config.TMIDDLE;
			column_index++;
		} else {
			ss << config.HORIZONTAL;
		}
	}
	ss << config.RTCORNER;
	ss << std::endl;

	// render the header names
	for (idx_t c = 0; c < column_count; c++) {
		auto column_idx = column_map[c];
		string name;
		if (column_idx == SPLIT_COLUMN) {
			name = config.DOTDOTDOT;
		} else {
			name = ConvertRenderValue(names[column_idx]);
		}
		RenderValue(ss, name, widths[c]);
	}
	ss << config.VERTICAL;
	ss << std::endl;

	// render the types
	if (config.render_mode == RenderMode::ROWS) {
		for (idx_t c = 0; c < column_count; c++) {
			auto column_idx = column_map[c];
			auto type = column_idx == SPLIT_COLUMN ? "" : RenderType(result_types[column_idx]);
			RenderValue(ss, type, widths[c]);
		}
		ss << config.VERTICAL;
		ss << std::endl;
	}

	// render the line under the header
	ss << config.LMIDDLE;
	column_index = 0;
	for (idx_t k = 0; k < total_length - 2; k++) {
		if (has_results && column_index + 1 < column_count && k == boundaries[column_index]) {
			ss << config.MIDDLE;
			column_index++;
		} else {
			ss << config.HORIZONTAL;
		}
	}
	ss << config.RMIDDLE;
	ss << std::endl;
}

void BoxRenderer::RenderValues(const list<ColumnDataCollection> &collections, const vector<idx_t> &column_map,
                               const vector<idx_t> &widths, const vector<LogicalType> &result_types, std::ostream &ss) {
	auto &top_collection = collections.front();
	auto &bottom_collection = collections.back();
	// render the top rows
	auto top_rows = top_collection.Count();
	auto bottom_rows = bottom_collection.Count();
	auto column_count = column_map.size();

	vector<ValueRenderAlignment> alignments;
	if (config.render_mode == RenderMode::ROWS) {
		for (idx_t c = 0; c < column_count; c++) {
			auto column_idx = column_map[c];
			if (column_idx == SPLIT_COLUMN) {
				alignments.push_back(ValueRenderAlignment::MIDDLE);
			} else {
				alignments.push_back(TypeAlignment(result_types[column_idx]));
			}
		}
	}

	auto rows = top_collection.GetRows();
	for (idx_t r = 0; r < top_rows; r++) {
		for (idx_t c = 0; c < column_count; c++) {
			auto column_idx = column_map[c];
			string str;
			if (column_idx == SPLIT_COLUMN) {
				str = config.DOTDOTDOT;
			} else {
				str = GetRenderValue(rows, column_idx, r);
			}
			ValueRenderAlignment alignment;
			if (config.render_mode == RenderMode::ROWS) {
				alignment = alignments[c];
			} else {
				if (c < 2) {
					alignment = ValueRenderAlignment::LEFT;
				} else if (c == SPLIT_COLUMN) {
					alignment = ValueRenderAlignment::MIDDLE;
				} else {
					alignment = ValueRenderAlignment::RIGHT;
				}
			}
			RenderValue(ss, str, widths[c], alignment);
		}
		ss << config.VERTICAL;
		ss << std::endl;
	}

	if (bottom_rows > 0) {
		if (config.render_mode == RenderMode::COLUMNS) {
			throw InternalException("Columns render mode does not support bottom rows");
		}
		// render the bottom rows
		// first render the divider
		auto brows = bottom_collection.GetRows();
		for (idx_t k = 0; k < 3; k++) {
			for (idx_t c = 0; c < column_count; c++) {
				auto column_idx = column_map[c];
				string str;
				auto alignment = alignments[c];
				if (alignment == ValueRenderAlignment::MIDDLE || column_idx == SPLIT_COLUMN) {
					str = config.DOT;
				} else {
					// align the dots in the center of the column
					auto top_value = GetRenderValue(rows, column_idx, top_rows - 1);
					auto bottom_value = GetRenderValue(brows, column_idx, bottom_rows - 1);
					auto top_length = MinValue<idx_t>(widths[c], Utf8Proc::RenderWidth(top_value));
					auto bottom_length = MinValue<idx_t>(widths[c], Utf8Proc::RenderWidth(bottom_value));
					auto dot_length = MinValue<idx_t>(top_length, bottom_length);
					if (top_length == 0) {
						dot_length = bottom_length;
					} else if (bottom_length == 0) {
						dot_length = top_length;
					}
					if (dot_length > 1) {
						auto padding = dot_length - 1;
						idx_t left_padding, right_padding;
						switch (alignment) {
						case ValueRenderAlignment::LEFT:
							left_padding = padding / 2;
							right_padding = padding - left_padding;
							break;
						case ValueRenderAlignment::RIGHT:
							right_padding = padding / 2;
							left_padding = padding - right_padding;
							break;
						default:
							throw InternalException("Unrecognized value renderer alignment");
						}
						str = string(left_padding, ' ') + config.DOT + string(right_padding, ' ');
					} else {
						if (dot_length == 0) {
							// everything is empty
							alignment = ValueRenderAlignment::MIDDLE;
						}
						str = config.DOT;
					}
				}
				RenderValue(ss, str, widths[c], alignment);
			}
			ss << config.VERTICAL;
			ss << std::endl;
		}
		// note that the bottom rows are in reverse order
		for (idx_t r = 0; r < bottom_rows; r++) {
			for (idx_t c = 0; c < column_count; c++) {
				auto column_idx = column_map[c];
				string str;
				if (column_idx == SPLIT_COLUMN) {
					str = config.DOTDOTDOT;
				} else {
					str = GetRenderValue(brows, column_idx, bottom_rows - r - 1);
				}
				RenderValue(ss, str, widths[c], alignments[c]);
			}
			ss << config.VERTICAL;
			ss << std::endl;
		}
	}
}

void BoxRenderer::RenderRowCount(string row_count_str, string shown_str, const string &column_count_str,
                                 const vector<idx_t> &boundaries, bool has_hidden_rows, bool has_hidden_columns,
                                 idx_t total_length, idx_t row_count, idx_t column_count, idx_t minimum_row_length,
                                 std::ostream &ss) {
	// check if we can merge the row_count_str and the shown_str
	bool display_shown_separately = has_hidden_rows;
	if (has_hidden_rows && total_length >= row_count_str.size() + shown_str.size() + 5) {
		// we can!
		row_count_str += " " + shown_str;
		shown_str = string();
		display_shown_separately = false;
		minimum_row_length = row_count_str.size() + 4;
	}
	auto minimum_length = row_count_str.size() + column_count_str.size() + 6;
	bool render_rows_and_columns = total_length >= minimum_length &&
	                               ((has_hidden_columns && row_count > 0) || (row_count >= 10 && column_count > 1));
	bool render_rows = total_length >= minimum_row_length && (row_count == 0 || row_count >= 10);
	bool render_anything = true;
	if (!render_rows && !render_rows_and_columns) {
		render_anything = false;
	}
	// render the bottom of the result values, if there are any
	if (row_count > 0) {
		ss << (render_anything ? config.LMIDDLE : config.LDCORNER);
		idx_t column_index = 0;
		for (idx_t k = 0; k < total_length - 2; k++) {
			if (column_index + 1 < boundaries.size() && k == boundaries[column_index]) {
				ss << config.DMIDDLE;
				column_index++;
			} else {
				ss << config.HORIZONTAL;
			}
		}
		ss << (render_anything ? config.RMIDDLE : config.RDCORNER);
		ss << std::endl;
	}
	if (!render_anything) {
		return;
	}

	if (render_rows_and_columns) {
		ss << config.VERTICAL;
		ss << " ";
		ss << row_count_str;
		ss << string(total_length - row_count_str.size() - column_count_str.size() - 4, ' ');
		ss << column_count_str;
		ss << " ";
		ss << config.VERTICAL;
		ss << std::endl;
	} else if (render_rows) {
		RenderValue(ss, row_count_str, total_length - 4);
		ss << config.VERTICAL;
		ss << std::endl;

		if (display_shown_separately) {
			RenderValue(ss, shown_str, total_length - 4);
			ss << config.VERTICAL;
			ss << std::endl;
		}
	}
	// render the bottom line
	ss << config.LDCORNER;
	for (idx_t k = 0; k < total_length - 2; k++) {
		ss << config.HORIZONTAL;
	}
	ss << config.RDCORNER;
	ss << std::endl;
}

void BoxRenderer::Render(ClientContext &context, const vector<string> &names, const ColumnDataCollection &result,
                         std::ostream &ss) {
	if (result.ColumnCount() != names.size()) {
		throw InternalException("Error in BoxRenderer::Render - unaligned columns and names");
	}
	auto max_width = config.max_width;
	if (max_width == 0) {
		if (Printer::IsTerminal(OutputStream::STREAM_STDOUT)) {
			max_width = Printer::TerminalWidth();
		} else {
			max_width = 120;
		}
	}
	// we do not support max widths under 80
	max_width = MaxValue<idx_t>(80, max_width);

	// figure out how many/which rows to render
	idx_t row_count = result.Count();
	idx_t rows_to_render = MinValue<idx_t>(row_count, config.max_rows);
	if (row_count <= config.max_rows + 3) {
		// hiding rows adds 3 extra rows
		// so hiding rows makes no sense if we are only slightly over the limit
		// if we are 1 row over the limit hiding rows will actually increase the number of lines we display!
		// in this case render all the rows
		rows_to_render = row_count;
	}
	idx_t top_rows;
	idx_t bottom_rows;
	if (rows_to_render == row_count) {
		top_rows = row_count;
		bottom_rows = 0;
	} else {
		top_rows = rows_to_render / 2 + (rows_to_render % 2 != 0 ? 1 : 0);
		bottom_rows = rows_to_render - top_rows;
	}
	auto row_count_str = to_string(row_count) + " rows";
	bool has_limited_rows = config.limit > 0 && row_count == config.limit;
	if (has_limited_rows) {
		row_count_str = "? rows";
	}
	string shown_str;
	bool has_hidden_rows = top_rows < row_count;
	if (has_hidden_rows) {
		shown_str = "(";
		if (has_limited_rows) {
			shown_str += ">" + to_string(config.limit - 1) + " rows, ";
		}
		shown_str += to_string(top_rows + bottom_rows) + " shown)";
	}
	auto minimum_row_length = MaxValue<idx_t>(row_count_str.size(), shown_str.size()) + 4;

	// fetch the top and bottom render collections from the result
	auto collections = FetchRenderCollections(context, result, top_rows, bottom_rows);
	auto column_names = names;
	auto result_types = result.Types();
	if (config.render_mode == RenderMode::COLUMNS) {
		collections = PivotCollections(context, std::move(collections), column_names, result_types, row_count);
	}

	// for each column, figure out the width
	// start off by figuring out the name of the header by looking at the column name and column type
	idx_t min_width = has_hidden_rows || row_count == 0 ? minimum_row_length : 0;
	vector<idx_t> column_map;
	idx_t total_length;
	auto widths =
	    ComputeRenderWidths(column_names, result_types, collections, min_width, max_width, column_map, total_length);

	// render boundaries for the individual columns
	vector<idx_t> boundaries;
	for (idx_t c = 0; c < widths.size(); c++) {
		idx_t render_boundary;
		if (c == 0) {
			render_boundary = widths[c] + 2;
		} else {
			render_boundary = boundaries[c - 1] + widths[c] + 3;
		}
		boundaries.push_back(render_boundary);
	}

	// now begin rendering
	// first render the header
	RenderHeader(column_names, result_types, column_map, widths, boundaries, total_length, row_count > 0, ss);

	// render the values, if there are any
	RenderValues(collections, column_map, widths, result_types, ss);

	// render the row count and column count
	auto column_count_str = to_string(result.ColumnCount()) + " column";
	if (result.ColumnCount() > 1) {
		column_count_str += "s";
	}
	bool has_hidden_columns = false;
	for (auto entry : column_map) {
		if (entry == SPLIT_COLUMN) {
			has_hidden_columns = true;
			break;
		}
	}
	idx_t column_count = column_map.size();
	if (config.render_mode == RenderMode::COLUMNS) {
		if (has_hidden_columns) {
			has_hidden_rows = true;
			shown_str = " (" + to_string(column_count - 3) + " shown)";
		} else {
			shown_str = string();
		}
	} else {
		if (has_hidden_columns) {
			column_count--;
			column_count_str += " (" + to_string(column_count) + " shown)";
		}
	}

	RenderRowCount(std::move(row_count_str), std::move(shown_str), column_count_str, boundaries, has_hidden_rows,
	               has_hidden_columns, total_length, row_count, column_count, minimum_row_length, ss);
}

} // namespace duckdb



namespace duckdb {

hash_t Checksum(uint64_t x) {
	return x * UINT64_C(0xbf58476d1ce4e5b9);
}

uint64_t Checksum(uint8_t *buffer, size_t size) {
	uint64_t result = 5381;
	uint64_t *ptr = (uint64_t *)buffer;
	size_t i;
	// for efficiency, we first checksum uint64_t values
	for (i = 0; i < size / 8; i++) {
		result ^= Checksum(ptr[i]);
	}
	if (size - i * 8 > 0) {
		// the remaining 0-7 bytes we hash using a string hash
		result ^= Hash(buffer + i * 8, size - i * 8);
	}
	return result;
}

} // namespace duckdb


namespace duckdb {

StreamWrapper::~StreamWrapper() {
}

CompressedFile::CompressedFile(CompressedFileSystem &fs, unique_ptr<FileHandle> child_handle_p, const string &path)
    : FileHandle(fs, path), compressed_fs(fs), child_handle(std::move(child_handle_p)) {
}

CompressedFile::~CompressedFile() {
	CompressedFile::Close();
}

void CompressedFile::Initialize(bool write) {
	Close();

	this->write = write;
	stream_data.in_buf_size = compressed_fs.InBufferSize();
	stream_data.out_buf_size = compressed_fs.OutBufferSize();
	stream_data.in_buff = make_unsafe_uniq_array<data_t>(stream_data.in_buf_size);
	stream_data.in_buff_start = stream_data.in_buff.get();
	stream_data.in_buff_end = stream_data.in_buff.get();
	stream_data.out_buff = make_unsafe_uniq_array<data_t>(stream_data.out_buf_size);
	stream_data.out_buff_start = stream_data.out_buff.get();
	stream_data.out_buff_end = stream_data.out_buff.get();

	stream_wrapper = compressed_fs.CreateStream();
	stream_wrapper->Initialize(*this, write);
}

int64_t CompressedFile::ReadData(void *buffer, int64_t remaining) {
	idx_t total_read = 0;
	while (true) {
		// first check if there are input bytes available in the output buffers
		if (stream_data.out_buff_start != stream_data.out_buff_end) {
			// there is! copy it into the output buffer
			idx_t available = MinValue<idx_t>(remaining, stream_data.out_buff_end - stream_data.out_buff_start);
			memcpy(data_ptr_t(buffer) + total_read, stream_data.out_buff_start, available);

			// increment the total read variables as required
			stream_data.out_buff_start += available;
			total_read += available;
			remaining -= available;
			if (remaining == 0) {
				// done! read enough
				return total_read;
			}
		}
		if (!stream_wrapper) {
			return total_read;
		}

		// ran out of buffer: read more data from the child stream
		stream_data.out_buff_start = stream_data.out_buff.get();
		stream_data.out_buff_end = stream_data.out_buff.get();
		D_ASSERT(stream_data.in_buff_start <= stream_data.in_buff_end);
		D_ASSERT(stream_data.in_buff_end <= stream_data.in_buff_start + stream_data.in_buf_size);

		// read more input when requested and still data in the input stream
		if (stream_data.refresh && (stream_data.in_buff_end == stream_data.in_buff.get() + stream_data.in_buf_size)) {
			auto bufrem = stream_data.in_buff_end - stream_data.in_buff_start;
			// buffer not empty, move remaining bytes to the beginning
			memmove(stream_data.in_buff.get(), stream_data.in_buff_start, bufrem);
			stream_data.in_buff_start = stream_data.in_buff.get();
			// refill the rest of input buffer
			auto sz = child_handle->Read(stream_data.in_buff_start + bufrem, stream_data.in_buf_size - bufrem);
			stream_data.in_buff_end = stream_data.in_buff_start + bufrem + sz;
			if (sz <= 0) {
				stream_wrapper.reset();
				break;
			}
		}

		// read more input if none available
		if (stream_data.in_buff_start == stream_data.in_buff_end) {
			// empty input buffer: refill from the start
			stream_data.in_buff_start = stream_data.in_buff.get();
			stream_data.in_buff_end = stream_data.in_buff_start;
			auto sz = child_handle->Read(stream_data.in_buff.get(), stream_data.in_buf_size);
			if (sz <= 0) {
				stream_wrapper.reset();
				break;
			}
			stream_data.in_buff_end = stream_data.in_buff_start + sz;
		}

		auto finished = stream_wrapper->Read(stream_data);
		if (finished) {
			stream_wrapper.reset();
		}
	}
	return total_read;
}

int64_t CompressedFile::WriteData(data_ptr_t buffer, int64_t nr_bytes) {
	stream_wrapper->Write(*this, stream_data, buffer, nr_bytes);
	return nr_bytes;
}

void CompressedFile::Close() {
	if (stream_wrapper) {
		stream_wrapper->Close();
		stream_wrapper.reset();
	}
	stream_data.in_buff.reset();
	stream_data.out_buff.reset();
	stream_data.out_buff_start = nullptr;
	stream_data.out_buff_end = nullptr;
	stream_data.in_buff_start = nullptr;
	stream_data.in_buff_end = nullptr;
	stream_data.in_buf_size = 0;
	stream_data.out_buf_size = 0;
}

int64_t CompressedFileSystem::Read(FileHandle &handle, void *buffer, int64_t nr_bytes) {
	auto &compressed_file = (CompressedFile &)handle;
	return compressed_file.ReadData(buffer, nr_bytes);
}

int64_t CompressedFileSystem::Write(FileHandle &handle, void *buffer, int64_t nr_bytes) {
	auto &compressed_file = (CompressedFile &)handle;
	return compressed_file.WriteData((data_ptr_t)buffer, nr_bytes);
}

void CompressedFileSystem::Reset(FileHandle &handle) {
	auto &compressed_file = (CompressedFile &)handle;
	compressed_file.child_handle->Reset();
	compressed_file.Initialize(compressed_file.write);
}

int64_t CompressedFileSystem::GetFileSize(FileHandle &handle) {
	auto &compressed_file = (CompressedFile &)handle;
	return compressed_file.child_handle->GetFileSize();
}

bool CompressedFileSystem::OnDiskFile(FileHandle &handle) {
	auto &compressed_file = (CompressedFile &)handle;
	return compressed_file.child_handle->OnDiskFile();
}

bool CompressedFileSystem::CanSeek() {
	return false;
}

} // namespace duckdb






namespace duckdb {

constexpr const idx_t DConstants::INVALID_INDEX;
const row_t MAX_ROW_ID = 4611686018427388000ULL; // 2^62
const column_t COLUMN_IDENTIFIER_ROW_ID = (column_t)-1;
const sel_t ZERO_VECTOR[STANDARD_VECTOR_SIZE] = {0};
const double PI = 3.141592653589793;

const transaction_t TRANSACTION_ID_START = 4611686018427388000ULL;                // 2^62
const transaction_t MAX_TRANSACTION_ID = NumericLimits<transaction_t>::Maximum(); // 2^63
const transaction_t NOT_DELETED_ID = NumericLimits<transaction_t>::Maximum() - 1; // 2^64 - 1
const transaction_t MAXIMUM_QUERY_ID = NumericLimits<transaction_t>::Maximum();   // 2^64

bool IsPowerOfTwo(uint64_t v) {
	return (v & (v - 1)) == 0;
}

uint64_t NextPowerOfTwo(uint64_t v) {
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v |= v >> 32;
	v++;
	return v;
}

uint64_t PreviousPowerOfTwo(uint64_t v) {
	return NextPowerOfTwo((v / 2) + 1);
}

bool IsInvalidSchema(const string &str) {
	return str.empty();
}

bool IsInvalidCatalog(const string &str) {
	return str.empty();
}

bool IsRowIdColumnId(column_t column_id) {
	return column_id == COLUMN_IDENTIFIER_ROW_ID;
}

} // namespace duckdb
/*
** This code taken from the SQLite test library.  Originally found on
** the internet.  The original header comment follows this comment.
** The code is largerly unchanged, but there have been some modifications.
*/
/*
 * This code implements the MD5 message-digest algorithm.
 * The algorithm is due to Ron Rivest.  This code was
 * written by Colin Plumb in 1993, no copyright is claimed.
 * This code is in the public domain; do with it what you wish.
 *
 * Equivalent code is available from RSA Data Security, Inc.
 * This code has been tested against that, and is equivalent,
 * except that you don't need to include two pages of legalese
 * with every copy.
 *
 * To compute the message digest of a chunk of bytes, declare an
 * MD5Context structure, pass it to MD5Init, call MD5Update as
 * needed on buffers full of bytes, and then call MD5Final, which
 * will fill a supplied 16-byte array with the digest.
 */


namespace duckdb {

/*
 * Note: this code is harmless on little-endian machines.
 */
static void ByteReverse(unsigned char *buf, unsigned longs) {
	uint32_t t;
	do {
		t = (uint32_t)((unsigned)buf[3] << 8 | buf[2]) << 16 | ((unsigned)buf[1] << 8 | buf[0]);
		*(uint32_t *)buf = t;
		buf += 4;
	} while (--longs);
}
/* The four core functions - F1 is optimized somewhat */

/* #define F1(x, y, z) (x & y | ~x & z) */
#define F1(x, y, z) ((z) ^ ((x) & ((y) ^ (z))))
#define F2(x, y, z) F1(z, x, y)
#define F3(x, y, z) ((x) ^ (y) ^ (z))
#define F4(x, y, z) ((y) ^ ((x) | ~(z)))

/* This is the central step in the MD5 algorithm. */
#define MD5STEP(f, w, x, y, z, data, s) ((w) += f(x, y, z) + (data), (w) = (w) << (s) | (w) >> (32 - (s)), (w) += (x))

/*
 * The core of the MD5 algorithm, this alters an existing MD5 hash to
 * reflect the addition of 16 longwords of new data.  MD5Update blocks
 * the data and converts bytes into longwords for this routine.
 */
static void MD5Transform(uint32_t buf[4], const uint32_t in[16]) {
	uint32_t a, b, c, d;

	a = buf[0];
	b = buf[1];
	c = buf[2];
	d = buf[3];

	MD5STEP(F1, a, b, c, d, in[0] + 0xd76aa478, 7);
	MD5STEP(F1, d, a, b, c, in[1] + 0xe8c7b756, 12);
	MD5STEP(F1, c, d, a, b, in[2] + 0x242070db, 17);
	MD5STEP(F1, b, c, d, a, in[3] + 0xc1bdceee, 22);
	MD5STEP(F1, a, b, c, d, in[4] + 0xf57c0faf, 7);
	MD5STEP(F1, d, a, b, c, in[5] + 0x4787c62a, 12);
	MD5STEP(F1, c, d, a, b, in[6] + 0xa8304613, 17);
	MD5STEP(F1, b, c, d, a, in[7] + 0xfd469501, 22);
	MD5STEP(F1, a, b, c, d, in[8] + 0x698098d8, 7);
	MD5STEP(F1, d, a, b, c, in[9] + 0x8b44f7af, 12);
	MD5STEP(F1, c, d, a, b, in[10] + 0xffff5bb1, 17);
	MD5STEP(F1, b, c, d, a, in[11] + 0x895cd7be, 22);
	MD5STEP(F1, a, b, c, d, in[12] + 0x6b901122, 7);
	MD5STEP(F1, d, a, b, c, in[13] + 0xfd987193, 12);
	MD5STEP(F1, c, d, a, b, in[14] + 0xa679438e, 17);
	MD5STEP(F1, b, c, d, a, in[15] + 0x49b40821, 22);

	MD5STEP(F2, a, b, c, d, in[1] + 0xf61e2562, 5);
	MD5STEP(F2, d, a, b, c, in[6] + 0xc040b340, 9);
	MD5STEP(F2, c, d, a, b, in[11] + 0x265e5a51, 14);
	MD5STEP(F2, b, c, d, a, in[0] + 0xe9b6c7aa, 20);
	MD5STEP(F2, a, b, c, d, in[5] + 0xd62f105d, 5);
	MD5STEP(F2, d, a, b, c, in[10] + 0x02441453, 9);
	MD5STEP(F2, c, d, a, b, in[15] + 0xd8a1e681, 14);
	MD5STEP(F2, b, c, d, a, in[4] + 0xe7d3fbc8, 20);
	MD5STEP(F2, a, b, c, d, in[9] + 0x21e1cde6, 5);
	MD5STEP(F2, d, a, b, c, in[14] + 0xc33707d6, 9);
	MD5STEP(F2, c, d, a, b, in[3] + 0xf4d50d87, 14);
	MD5STEP(F2, b, c, d, a, in[8] + 0x455a14ed, 20);
	MD5STEP(F2, a, b, c, d, in[13] + 0xa9e3e905, 5);
	MD5STEP(F2, d, a, b, c, in[2] + 0xfcefa3f8, 9);
	MD5STEP(F2, c, d, a, b, in[7] + 0x676f02d9, 14);
	MD5STEP(F2, b, c, d, a, in[12] + 0x8d2a4c8a, 20);

	MD5STEP(F3, a, b, c, d, in[5] + 0xfffa3942, 4);
	MD5STEP(F3, d, a, b, c, in[8] + 0x8771f681, 11);
	MD5STEP(F3, c, d, a, b, in[11] + 0x6d9d6122, 16);
	MD5STEP(F3, b, c, d, a, in[14] + 0xfde5380c, 23);
	MD5STEP(F3, a, b, c, d, in[1] + 0xa4beea44, 4);
	MD5STEP(F3, d, a, b, c, in[4] + 0x4bdecfa9, 11);
	MD5STEP(F3, c, d, a, b, in[7] + 0xf6bb4b60, 16);
	MD5STEP(F3, b, c, d, a, in[10] + 0xbebfbc70, 23);
	MD5STEP(F3, a, b, c, d, in[13] + 0x289b7ec6, 4);
	MD5STEP(F3, d, a, b, c, in[0] + 0xeaa127fa, 11);
	MD5STEP(F3, c, d, a, b, in[3] + 0xd4ef3085, 16);
	MD5STEP(F3, b, c, d, a, in[6] + 0x04881d05, 23);
	MD5STEP(F3, a, b, c, d, in[9] + 0xd9d4d039, 4);
	MD5STEP(F3, d, a, b, c, in[12] + 0xe6db99e5, 11);
	MD5STEP(F3, c, d, a, b, in[15] + 0x1fa27cf8, 16);
	MD5STEP(F3, b, c, d, a, in[2] + 0xc4ac5665, 23);

	MD5STEP(F4, a, b, c, d, in[0] + 0xf4292244, 6);
	MD5STEP(F4, d, a, b, c, in[7] + 0x432aff97, 10);
	MD5STEP(F4, c, d, a, b, in[14] + 0xab9423a7, 15);
	MD5STEP(F4, b, c, d, a, in[5] + 0xfc93a039, 21);
	MD5STEP(F4, a, b, c, d, in[12] + 0x655b59c3, 6);
	MD5STEP(F4, d, a, b, c, in[3] + 0x8f0ccc92, 10);
	MD5STEP(F4, c, d, a, b, in[10] + 0xffeff47d, 15);
	MD5STEP(F4, b, c, d, a, in[1] + 0x85845dd1, 21);
	MD5STEP(F4, a, b, c, d, in[8] + 0x6fa87e4f, 6);
	MD5STEP(F4, d, a, b, c, in[15] + 0xfe2ce6e0, 10);
	MD5STEP(F4, c, d, a, b, in[6] + 0xa3014314, 15);
	MD5STEP(F4, b, c, d, a, in[13] + 0x4e0811a1, 21);
	MD5STEP(F4, a, b, c, d, in[4] + 0xf7537e82, 6);
	MD5STEP(F4, d, a, b, c, in[11] + 0xbd3af235, 10);
	MD5STEP(F4, c, d, a, b, in[2] + 0x2ad7d2bb, 15);
	MD5STEP(F4, b, c, d, a, in[9] + 0xeb86d391, 21);

	buf[0] += a;
	buf[1] += b;
	buf[2] += c;
	buf[3] += d;
}

/*
 * Start MD5 accumulation.  Set bit count to 0 and buffer to mysterious
 * initialization constants.
 */
MD5Context::MD5Context() {
	buf[0] = 0x67452301;
	buf[1] = 0xefcdab89;
	buf[2] = 0x98badcfe;
	buf[3] = 0x10325476;
	bits[0] = 0;
	bits[1] = 0;
}

/*
 * Update context to reflect the concatenation of another buffer full
 * of bytes.
 */
void MD5Context::MD5Update(const_data_ptr_t input, idx_t len) {
	uint32_t t;

	/* Update bitcount */

	t = bits[0];
	if ((bits[0] = t + ((uint32_t)len << 3)) < t) {
		bits[1]++; /* Carry from low to high */
	}
	bits[1] += len >> 29;

	t = (t >> 3) & 0x3f; /* Bytes already in shsInfo->data */

	/* Handle any leading odd-sized chunks */

	if (t) {
		unsigned char *p = (unsigned char *)in + t;

		t = 64 - t;
		if (len < t) {
			memcpy(p, input, len);
			return;
		}
		memcpy(p, input, t);
		ByteReverse(in, 16);
		MD5Transform(buf, (uint32_t *)in);
		input += t;
		len -= t;
	}

	/* Process data in 64-byte chunks */

	while (len >= 64) {
		memcpy(in, input, 64);
		ByteReverse(in, 16);
		MD5Transform(buf, (uint32_t *)in);
		input += 64;
		len -= 64;
	}

	/* Handle any remaining bytes of data. */
	memcpy(in, input, len);
}

/*
 * Final wrapup - pad to 64-byte boundary with the bit pattern
 * 1 0* (64-bit count of bits processed, MSB-first)
 */
void MD5Context::Finish(data_ptr_t out_digest) {
	unsigned count;
	unsigned char *p;

	/* Compute number of bytes mod 64 */
	count = (bits[0] >> 3) & 0x3F;

	/* Set the first char of padding to 0x80.  This is safe since there is
	   always at least one byte free */
	p = in + count;
	*p++ = 0x80;

	/* Bytes of padding needed to make 64 bytes */
	count = 64 - 1 - count;

	/* Pad out to 56 mod 64 */
	if (count < 8) {
		/* Two lots of padding:  Pad the first block to 64 bytes */
		memset(p, 0, count);
		ByteReverse(in, 16);
		MD5Transform(buf, (uint32_t *)in);

		/* Now fill the next block with 56 bytes */
		memset(in, 0, 56);
	} else {
		/* Pad block to 56 bytes */
		memset(p, 0, count - 8);
	}
	ByteReverse(in, 14);

	/* Append length in bits and transform */
	((uint32_t *)in)[14] = bits[0];
	((uint32_t *)in)[15] = bits[1];

	MD5Transform(buf, (uint32_t *)in);
	ByteReverse((unsigned char *)buf, 4);
	memcpy(out_digest, buf, 16);
}

void MD5Context::DigestToBase16(const_data_ptr_t digest, char *zbuf) {
	static char const HEX_CODES[] = "0123456789abcdef";
	int i, j;

	for (j = i = 0; i < 16; i++) {
		int a = digest[i];
		zbuf[j++] = HEX_CODES[(a >> 4) & 0xf];
		zbuf[j++] = HEX_CODES[a & 0xf];
	}
}

void MD5Context::FinishHex(char *out_digest) {
	data_t digest[MD5_HASH_LENGTH_BINARY];
	Finish(digest);
	DigestToBase16(digest, out_digest);
}

string MD5Context::FinishHex() {
	char digest[MD5_HASH_LENGTH_TEXT];
	FinishHex(digest);
	return string(digest, MD5_HASH_LENGTH_TEXT);
}

void MD5Context::Add(const char *data) {
	MD5Update((const_data_ptr_t)data, strlen(data));
}

} // namespace duckdb
// This file is licensed under Apache License 2.0
// Source code taken from https://github.com/google/benchmark
// It is highly modified




namespace duckdb {

inline uint64_t ChronoNow() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(
	           std::chrono::time_point_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now())
	               .time_since_epoch())
	    .count();
}

inline uint64_t Now() {
#if defined(RDTSC)
#if defined(__i386__)
	uint64_t ret;
	__asm__ volatile("rdtsc" : "=A"(ret));
	return ret;
#elif defined(__x86_64__) || defined(__amd64__)
	uint64_t low, high;
	__asm__ volatile("rdtsc" : "=a"(low), "=d"(high));
	return (high << 32) | low;
#elif defined(__powerpc__) || defined(__ppc__)
	uint64_t tbl, tbu0, tbu1;
	asm("mftbu %0" : "=r"(tbu0));
	asm("mftb  %0" : "=r"(tbl));
	asm("mftbu %0" : "=r"(tbu1));
	tbl &= -static_cast<int64>(tbu0 == tbu1);
	return (tbu1 << 32) | tbl;
#elif defined(__sparc__)
	uint64_t tick;
	asm(".byte 0x83, 0x41, 0x00, 0x00");
	asm("mov   %%g1, %0" : "=r"(tick));
	return tick;
#elif defined(__ia64__)
	uint64_t itc;
	asm("mov %0 = ar.itc" : "=r"(itc));
	return itc;
#elif defined(COMPILER_MSVC) && defined(_M_IX86)
	_asm rdtsc
#elif defined(COMPILER_MSVC)
	return __rdtsc();
#elif defined(__aarch64__)
	uint64_t virtual_timer_value;
	asm volatile("mrs %0, cntvct_el0" : "=r"(virtual_timer_value));
	return virtual_timer_value;
#elif defined(__ARM_ARCH)
#if (__ARM_ARCH >= 6)
	uint32_t pmccntr;
	uint32_t pmuseren;
	uint32_t pmcntenset;
	asm volatile("mrc p15, 0, %0, c9, c14, 0" : "=r"(pmuseren));
	if (pmuseren & 1) { // Allows reading perfmon counters for user mode code.
		asm volatile("mrc p15, 0, %0, c9, c12, 1" : "=r"(pmcntenset));
		if (pmcntenset & 0x80000000ul) { // Is it counting?
			asm volatile("mrc p15, 0, %0, c9, c13, 0" : "=r"(pmccntr));
			return static_cast<uint64_t>(pmccntr) * 64; // Should optimize to << 6
		}
	}
#endif
	return ChronoNow();
#else
	return ChronoNow();
#endif
#else
	return ChronoNow();
#endif // defined(RDTSC)
}
uint64_t CycleCounter::Tick() const {
	return Now();
}
} // namespace duckdb
//-------------------------------------------------------------------------
// This file is automatically generated by scripts/generate_enum_util.py
// Do not edit this file manually, your changes will be overwritten
// If you want to exclude an enum from serialization, add it to the blacklist in the script
//
// Note: The generated code will only work properly if the enum is a top level item in the duckdb namespace
// If the enum is nested in a class, or in another namespace, the generated code will not compile.
// You should move the enum to the duckdb namespace, manually write a specialization or add it to the blacklist
//-------------------------------------------------------------------------
































































































namespace duckdb {

template <>
const char *EnumUtil::ToChars<TaskExecutionMode>(TaskExecutionMode value) {
	switch (value) {
	case TaskExecutionMode::PROCESS_ALL:
		return "PROCESS_ALL";
	case TaskExecutionMode::PROCESS_PARTIAL:
		return "PROCESS_PARTIAL";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
TaskExecutionMode EnumUtil::FromString<TaskExecutionMode>(const char *value) {
	if (StringUtil::Equals(value, "PROCESS_ALL")) {
		return TaskExecutionMode::PROCESS_ALL;
	}
	if (StringUtil::Equals(value, "PROCESS_PARTIAL")) {
		return TaskExecutionMode::PROCESS_PARTIAL;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<TaskExecutionResult>(TaskExecutionResult value) {
	switch (value) {
	case TaskExecutionResult::TASK_FINISHED:
		return "TASK_FINISHED";
	case TaskExecutionResult::TASK_NOT_FINISHED:
		return "TASK_NOT_FINISHED";
	case TaskExecutionResult::TASK_ERROR:
		return "TASK_ERROR";
	case TaskExecutionResult::TASK_BLOCKED:
		return "TASK_BLOCKED";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
TaskExecutionResult EnumUtil::FromString<TaskExecutionResult>(const char *value) {
	if (StringUtil::Equals(value, "TASK_FINISHED")) {
		return TaskExecutionResult::TASK_FINISHED;
	}
	if (StringUtil::Equals(value, "TASK_NOT_FINISHED")) {
		return TaskExecutionResult::TASK_NOT_FINISHED;
	}
	if (StringUtil::Equals(value, "TASK_ERROR")) {
		return TaskExecutionResult::TASK_ERROR;
	}
	if (StringUtil::Equals(value, "TASK_BLOCKED")) {
		return TaskExecutionResult::TASK_BLOCKED;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<InterruptMode>(InterruptMode value) {
	switch (value) {
	case InterruptMode::NO_INTERRUPTS:
		return "NO_INTERRUPTS";
	case InterruptMode::TASK:
		return "TASK";
	case InterruptMode::BLOCKING:
		return "BLOCKING";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
InterruptMode EnumUtil::FromString<InterruptMode>(const char *value) {
	if (StringUtil::Equals(value, "NO_INTERRUPTS")) {
		return InterruptMode::NO_INTERRUPTS;
	}
	if (StringUtil::Equals(value, "TASK")) {
		return InterruptMode::TASK;
	}
	if (StringUtil::Equals(value, "BLOCKING")) {
		return InterruptMode::BLOCKING;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<DistinctType>(DistinctType value) {
	switch (value) {
	case DistinctType::DISTINCT:
		return "DISTINCT";
	case DistinctType::DISTINCT_ON:
		return "DISTINCT_ON";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
DistinctType EnumUtil::FromString<DistinctType>(const char *value) {
	if (StringUtil::Equals(value, "DISTINCT")) {
		return DistinctType::DISTINCT;
	}
	if (StringUtil::Equals(value, "DISTINCT_ON")) {
		return DistinctType::DISTINCT_ON;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<TableFilterType>(TableFilterType value) {
	switch (value) {
	case TableFilterType::CONSTANT_COMPARISON:
		return "CONSTANT_COMPARISON";
	case TableFilterType::IS_NULL:
		return "IS_NULL";
	case TableFilterType::IS_NOT_NULL:
		return "IS_NOT_NULL";
	case TableFilterType::CONJUNCTION_OR:
		return "CONJUNCTION_OR";
	case TableFilterType::CONJUNCTION_AND:
		return "CONJUNCTION_AND";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
TableFilterType EnumUtil::FromString<TableFilterType>(const char *value) {
	if (StringUtil::Equals(value, "CONSTANT_COMPARISON")) {
		return TableFilterType::CONSTANT_COMPARISON;
	}
	if (StringUtil::Equals(value, "IS_NULL")) {
		return TableFilterType::IS_NULL;
	}
	if (StringUtil::Equals(value, "IS_NOT_NULL")) {
		return TableFilterType::IS_NOT_NULL;
	}
	if (StringUtil::Equals(value, "CONJUNCTION_OR")) {
		return TableFilterType::CONJUNCTION_OR;
	}
	if (StringUtil::Equals(value, "CONJUNCTION_AND")) {
		return TableFilterType::CONJUNCTION_AND;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<BindingMode>(BindingMode value) {
	switch (value) {
	case BindingMode::STANDARD_BINDING:
		return "STANDARD_BINDING";
	case BindingMode::EXTRACT_NAMES:
		return "EXTRACT_NAMES";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
BindingMode EnumUtil::FromString<BindingMode>(const char *value) {
	if (StringUtil::Equals(value, "STANDARD_BINDING")) {
		return BindingMode::STANDARD_BINDING;
	}
	if (StringUtil::Equals(value, "EXTRACT_NAMES")) {
		return BindingMode::EXTRACT_NAMES;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<TableColumnType>(TableColumnType value) {
	switch (value) {
	case TableColumnType::STANDARD:
		return "STANDARD";
	case TableColumnType::GENERATED:
		return "GENERATED";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
TableColumnType EnumUtil::FromString<TableColumnType>(const char *value) {
	if (StringUtil::Equals(value, "STANDARD")) {
		return TableColumnType::STANDARD;
	}
	if (StringUtil::Equals(value, "GENERATED")) {
		return TableColumnType::GENERATED;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<AggregateType>(AggregateType value) {
	switch (value) {
	case AggregateType::NON_DISTINCT:
		return "NON_DISTINCT";
	case AggregateType::DISTINCT:
		return "DISTINCT";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
AggregateType EnumUtil::FromString<AggregateType>(const char *value) {
	if (StringUtil::Equals(value, "NON_DISTINCT")) {
		return AggregateType::NON_DISTINCT;
	}
	if (StringUtil::Equals(value, "DISTINCT")) {
		return AggregateType::DISTINCT;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<AggregateOrderDependent>(AggregateOrderDependent value) {
	switch (value) {
	case AggregateOrderDependent::ORDER_DEPENDENT:
		return "ORDER_DEPENDENT";
	case AggregateOrderDependent::NOT_ORDER_DEPENDENT:
		return "NOT_ORDER_DEPENDENT";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
AggregateOrderDependent EnumUtil::FromString<AggregateOrderDependent>(const char *value) {
	if (StringUtil::Equals(value, "ORDER_DEPENDENT")) {
		return AggregateOrderDependent::ORDER_DEPENDENT;
	}
	if (StringUtil::Equals(value, "NOT_ORDER_DEPENDENT")) {
		return AggregateOrderDependent::NOT_ORDER_DEPENDENT;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<FunctionNullHandling>(FunctionNullHandling value) {
	switch (value) {
	case FunctionNullHandling::DEFAULT_NULL_HANDLING:
		return "DEFAULT_NULL_HANDLING";
	case FunctionNullHandling::SPECIAL_HANDLING:
		return "SPECIAL_HANDLING";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
FunctionNullHandling EnumUtil::FromString<FunctionNullHandling>(const char *value) {
	if (StringUtil::Equals(value, "DEFAULT_NULL_HANDLING")) {
		return FunctionNullHandling::DEFAULT_NULL_HANDLING;
	}
	if (StringUtil::Equals(value, "SPECIAL_HANDLING")) {
		return FunctionNullHandling::SPECIAL_HANDLING;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<FunctionSideEffects>(FunctionSideEffects value) {
	switch (value) {
	case FunctionSideEffects::NO_SIDE_EFFECTS:
		return "NO_SIDE_EFFECTS";
	case FunctionSideEffects::HAS_SIDE_EFFECTS:
		return "HAS_SIDE_EFFECTS";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
FunctionSideEffects EnumUtil::FromString<FunctionSideEffects>(const char *value) {
	if (StringUtil::Equals(value, "NO_SIDE_EFFECTS")) {
		return FunctionSideEffects::NO_SIDE_EFFECTS;
	}
	if (StringUtil::Equals(value, "HAS_SIDE_EFFECTS")) {
		return FunctionSideEffects::HAS_SIDE_EFFECTS;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<MacroType>(MacroType value) {
	switch (value) {
	case MacroType::VOID_MACRO:
		return "VOID_MACRO";
	case MacroType::TABLE_MACRO:
		return "TABLE_MACRO";
	case MacroType::SCALAR_MACRO:
		return "SCALAR_MACRO";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
MacroType EnumUtil::FromString<MacroType>(const char *value) {
	if (StringUtil::Equals(value, "VOID_MACRO")) {
		return MacroType::VOID_MACRO;
	}
	if (StringUtil::Equals(value, "TABLE_MACRO")) {
		return MacroType::TABLE_MACRO;
	}
	if (StringUtil::Equals(value, "SCALAR_MACRO")) {
		return MacroType::SCALAR_MACRO;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<ArrowVariableSizeType>(ArrowVariableSizeType value) {
	switch (value) {
	case ArrowVariableSizeType::FIXED_SIZE:
		return "FIXED_SIZE";
	case ArrowVariableSizeType::NORMAL:
		return "NORMAL";
	case ArrowVariableSizeType::SUPER_SIZE:
		return "SUPER_SIZE";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
ArrowVariableSizeType EnumUtil::FromString<ArrowVariableSizeType>(const char *value) {
	if (StringUtil::Equals(value, "FIXED_SIZE")) {
		return ArrowVariableSizeType::FIXED_SIZE;
	}
	if (StringUtil::Equals(value, "NORMAL")) {
		return ArrowVariableSizeType::NORMAL;
	}
	if (StringUtil::Equals(value, "SUPER_SIZE")) {
		return ArrowVariableSizeType::SUPER_SIZE;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<ArrowDateTimeType>(ArrowDateTimeType value) {
	switch (value) {
	case ArrowDateTimeType::MILLISECONDS:
		return "MILLISECONDS";
	case ArrowDateTimeType::MICROSECONDS:
		return "MICROSECONDS";
	case ArrowDateTimeType::NANOSECONDS:
		return "NANOSECONDS";
	case ArrowDateTimeType::SECONDS:
		return "SECONDS";
	case ArrowDateTimeType::DAYS:
		return "DAYS";
	case ArrowDateTimeType::MONTHS:
		return "MONTHS";
	case ArrowDateTimeType::MONTH_DAY_NANO:
		return "MONTH_DAY_NANO";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
ArrowDateTimeType EnumUtil::FromString<ArrowDateTimeType>(const char *value) {
	if (StringUtil::Equals(value, "MILLISECONDS")) {
		return ArrowDateTimeType::MILLISECONDS;
	}
	if (StringUtil::Equals(value, "MICROSECONDS")) {
		return ArrowDateTimeType::MICROSECONDS;
	}
	if (StringUtil::Equals(value, "NANOSECONDS")) {
		return ArrowDateTimeType::NANOSECONDS;
	}
	if (StringUtil::Equals(value, "SECONDS")) {
		return ArrowDateTimeType::SECONDS;
	}
	if (StringUtil::Equals(value, "DAYS")) {
		return ArrowDateTimeType::DAYS;
	}
	if (StringUtil::Equals(value, "MONTHS")) {
		return ArrowDateTimeType::MONTHS;
	}
	if (StringUtil::Equals(value, "MONTH_DAY_NANO")) {
		return ArrowDateTimeType::MONTH_DAY_NANO;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<StrTimeSpecifier>(StrTimeSpecifier value) {
	switch (value) {
	case StrTimeSpecifier::ABBREVIATED_WEEKDAY_NAME:
		return "ABBREVIATED_WEEKDAY_NAME";
	case StrTimeSpecifier::FULL_WEEKDAY_NAME:
		return "FULL_WEEKDAY_NAME";
	case StrTimeSpecifier::WEEKDAY_DECIMAL:
		return "WEEKDAY_DECIMAL";
	case StrTimeSpecifier::DAY_OF_MONTH_PADDED:
		return "DAY_OF_MONTH_PADDED";
	case StrTimeSpecifier::DAY_OF_MONTH:
		return "DAY_OF_MONTH";
	case StrTimeSpecifier::ABBREVIATED_MONTH_NAME:
		return "ABBREVIATED_MONTH_NAME";
	case StrTimeSpecifier::FULL_MONTH_NAME:
		return "FULL_MONTH_NAME";
	case StrTimeSpecifier::MONTH_DECIMAL_PADDED:
		return "MONTH_DECIMAL_PADDED";
	case StrTimeSpecifier::MONTH_DECIMAL:
		return "MONTH_DECIMAL";
	case StrTimeSpecifier::YEAR_WITHOUT_CENTURY_PADDED:
		return "YEAR_WITHOUT_CENTURY_PADDED";
	case StrTimeSpecifier::YEAR_WITHOUT_CENTURY:
		return "YEAR_WITHOUT_CENTURY";
	case StrTimeSpecifier::YEAR_DECIMAL:
		return "YEAR_DECIMAL";
	case StrTimeSpecifier::HOUR_24_PADDED:
		return "HOUR_24_PADDED";
	case StrTimeSpecifier::HOUR_24_DECIMAL:
		return "HOUR_24_DECIMAL";
	case StrTimeSpecifier::HOUR_12_PADDED:
		return "HOUR_12_PADDED";
	case StrTimeSpecifier::HOUR_12_DECIMAL:
		return "HOUR_12_DECIMAL";
	case StrTimeSpecifier::AM_PM:
		return "AM_PM";
	case StrTimeSpecifier::MINUTE_PADDED:
		return "MINUTE_PADDED";
	case StrTimeSpecifier::MINUTE_DECIMAL:
		return "MINUTE_DECIMAL";
	case StrTimeSpecifier::SECOND_PADDED:
		return "SECOND_PADDED";
	case StrTimeSpecifier::SECOND_DECIMAL:
		return "SECOND_DECIMAL";
	case StrTimeSpecifier::MICROSECOND_PADDED:
		return "MICROSECOND_PADDED";
	case StrTimeSpecifier::MILLISECOND_PADDED:
		return "MILLISECOND_PADDED";
	case StrTimeSpecifier::UTC_OFFSET:
		return "UTC_OFFSET";
	case StrTimeSpecifier::TZ_NAME:
		return "TZ_NAME";
	case StrTimeSpecifier::DAY_OF_YEAR_PADDED:
		return "DAY_OF_YEAR_PADDED";
	case StrTimeSpecifier::DAY_OF_YEAR_DECIMAL:
		return "DAY_OF_YEAR_DECIMAL";
	case StrTimeSpecifier::WEEK_NUMBER_PADDED_SUN_FIRST:
		return "WEEK_NUMBER_PADDED_SUN_FIRST";
	case StrTimeSpecifier::WEEK_NUMBER_PADDED_MON_FIRST:
		return "WEEK_NUMBER_PADDED_MON_FIRST";
	case StrTimeSpecifier::LOCALE_APPROPRIATE_DATE_AND_TIME:
		return "LOCALE_APPROPRIATE_DATE_AND_TIME";
	case StrTimeSpecifier::LOCALE_APPROPRIATE_DATE:
		return "LOCALE_APPROPRIATE_DATE";
	case StrTimeSpecifier::LOCALE_APPROPRIATE_TIME:
		return "LOCALE_APPROPRIATE_TIME";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
StrTimeSpecifier EnumUtil::FromString<StrTimeSpecifier>(const char *value) {
	if (StringUtil::Equals(value, "ABBREVIATED_WEEKDAY_NAME")) {
		return StrTimeSpecifier::ABBREVIATED_WEEKDAY_NAME;
	}
	if (StringUtil::Equals(value, "FULL_WEEKDAY_NAME")) {
		return StrTimeSpecifier::FULL_WEEKDAY_NAME;
	}
	if (StringUtil::Equals(value, "WEEKDAY_DECIMAL")) {
		return StrTimeSpecifier::WEEKDAY_DECIMAL;
	}
	if (StringUtil::Equals(value, "DAY_OF_MONTH_PADDED")) {
		return StrTimeSpecifier::DAY_OF_MONTH_PADDED;
	}
	if (StringUtil::Equals(value, "DAY_OF_MONTH")) {
		return StrTimeSpecifier::DAY_OF_MONTH;
	}
	if (StringUtil::Equals(value, "ABBREVIATED_MONTH_NAME")) {
		return StrTimeSpecifier::ABBREVIATED_MONTH_NAME;
	}
	if (StringUtil::Equals(value, "FULL_MONTH_NAME")) {
		return StrTimeSpecifier::FULL_MONTH_NAME;
	}
	if (StringUtil::Equals(value, "MONTH_DECIMAL_PADDED")) {
		return StrTimeSpecifier::MONTH_DECIMAL_PADDED;
	}
	if (StringUtil::Equals(value, "MONTH_DECIMAL")) {
		return StrTimeSpecifier::MONTH_DECIMAL;
	}
	if (StringUtil::Equals(value, "YEAR_WITHOUT_CENTURY_PADDED")) {
		return StrTimeSpecifier::YEAR_WITHOUT_CENTURY_PADDED;
	}
	if (StringUtil::Equals(value, "YEAR_WITHOUT_CENTURY")) {
		return StrTimeSpecifier::YEAR_WITHOUT_CENTURY;
	}
	if (StringUtil::Equals(value, "YEAR_DECIMAL")) {
		return StrTimeSpecifier::YEAR_DECIMAL;
	}
	if (StringUtil::Equals(value, "HOUR_24_PADDED")) {
		return StrTimeSpecifier::HOUR_24_PADDED;
	}
	if (StringUtil::Equals(value, "HOUR_24_DECIMAL")) {
		return StrTimeSpecifier::HOUR_24_DECIMAL;
	}
	if (StringUtil::Equals(value, "HOUR_12_PADDED")) {
		return StrTimeSpecifier::HOUR_12_PADDED;
	}
	if (StringUtil::Equals(value, "HOUR_12_DECIMAL")) {
		return StrTimeSpecifier::HOUR_12_DECIMAL;
	}
	if (StringUtil::Equals(value, "AM_PM")) {
		return StrTimeSpecifier::AM_PM;
	}
	if (StringUtil::Equals(value, "MINUTE_PADDED")) {
		return StrTimeSpecifier::MINUTE_PADDED;
	}
	if (StringUtil::Equals(value, "MINUTE_DECIMAL")) {
		return StrTimeSpecifier::MINUTE_DECIMAL;
	}
	if (StringUtil::Equals(value, "SECOND_PADDED")) {
		return StrTimeSpecifier::SECOND_PADDED;
	}
	if (StringUtil::Equals(value, "SECOND_DECIMAL")) {
		return StrTimeSpecifier::SECOND_DECIMAL;
	}
	if (StringUtil::Equals(value, "MICROSECOND_PADDED")) {
		return StrTimeSpecifier::MICROSECOND_PADDED;
	}
	if (StringUtil::Equals(value, "MILLISECOND_PADDED")) {
		return StrTimeSpecifier::MILLISECOND_PADDED;
	}
	if (StringUtil::Equals(value, "UTC_OFFSET")) {
		return StrTimeSpecifier::UTC_OFFSET;
	}
	if (StringUtil::Equals(value, "TZ_NAME")) {
		return StrTimeSpecifier::TZ_NAME;
	}
	if (StringUtil::Equals(value, "DAY_OF_YEAR_PADDED")) {
		return StrTimeSpecifier::DAY_OF_YEAR_PADDED;
	}
	if (StringUtil::Equals(value, "DAY_OF_YEAR_DECIMAL")) {
		return StrTimeSpecifier::DAY_OF_YEAR_DECIMAL;
	}
	if (StringUtil::Equals(value, "WEEK_NUMBER_PADDED_SUN_FIRST")) {
		return StrTimeSpecifier::WEEK_NUMBER_PADDED_SUN_FIRST;
	}
	if (StringUtil::Equals(value, "WEEK_NUMBER_PADDED_MON_FIRST")) {
		return StrTimeSpecifier::WEEK_NUMBER_PADDED_MON_FIRST;
	}
	if (StringUtil::Equals(value, "LOCALE_APPROPRIATE_DATE_AND_TIME")) {
		return StrTimeSpecifier::LOCALE_APPROPRIATE_DATE_AND_TIME;
	}
	if (StringUtil::Equals(value, "LOCALE_APPROPRIATE_DATE")) {
		return StrTimeSpecifier::LOCALE_APPROPRIATE_DATE;
	}
	if (StringUtil::Equals(value, "LOCALE_APPROPRIATE_TIME")) {
		return StrTimeSpecifier::LOCALE_APPROPRIATE_TIME;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<SimplifiedTokenType>(SimplifiedTokenType value) {
	switch (value) {
	case SimplifiedTokenType::SIMPLIFIED_TOKEN_IDENTIFIER:
		return "SIMPLIFIED_TOKEN_IDENTIFIER";
	case SimplifiedTokenType::SIMPLIFIED_TOKEN_NUMERIC_CONSTANT:
		return "SIMPLIFIED_TOKEN_NUMERIC_CONSTANT";
	case SimplifiedTokenType::SIMPLIFIED_TOKEN_STRING_CONSTANT:
		return "SIMPLIFIED_TOKEN_STRING_CONSTANT";
	case SimplifiedTokenType::SIMPLIFIED_TOKEN_OPERATOR:
		return "SIMPLIFIED_TOKEN_OPERATOR";
	case SimplifiedTokenType::SIMPLIFIED_TOKEN_KEYWORD:
		return "SIMPLIFIED_TOKEN_KEYWORD";
	case SimplifiedTokenType::SIMPLIFIED_TOKEN_COMMENT:
		return "SIMPLIFIED_TOKEN_COMMENT";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
SimplifiedTokenType EnumUtil::FromString<SimplifiedTokenType>(const char *value) {
	if (StringUtil::Equals(value, "SIMPLIFIED_TOKEN_IDENTIFIER")) {
		return SimplifiedTokenType::SIMPLIFIED_TOKEN_IDENTIFIER;
	}
	if (StringUtil::Equals(value, "SIMPLIFIED_TOKEN_NUMERIC_CONSTANT")) {
		return SimplifiedTokenType::SIMPLIFIED_TOKEN_NUMERIC_CONSTANT;
	}
	if (StringUtil::Equals(value, "SIMPLIFIED_TOKEN_STRING_CONSTANT")) {
		return SimplifiedTokenType::SIMPLIFIED_TOKEN_STRING_CONSTANT;
	}
	if (StringUtil::Equals(value, "SIMPLIFIED_TOKEN_OPERATOR")) {
		return SimplifiedTokenType::SIMPLIFIED_TOKEN_OPERATOR;
	}
	if (StringUtil::Equals(value, "SIMPLIFIED_TOKEN_KEYWORD")) {
		return SimplifiedTokenType::SIMPLIFIED_TOKEN_KEYWORD;
	}
	if (StringUtil::Equals(value, "SIMPLIFIED_TOKEN_COMMENT")) {
		return SimplifiedTokenType::SIMPLIFIED_TOKEN_COMMENT;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<KeywordCategory>(KeywordCategory value) {
	switch (value) {
	case KeywordCategory::KEYWORD_RESERVED:
		return "KEYWORD_RESERVED";
	case KeywordCategory::KEYWORD_UNRESERVED:
		return "KEYWORD_UNRESERVED";
	case KeywordCategory::KEYWORD_TYPE_FUNC:
		return "KEYWORD_TYPE_FUNC";
	case KeywordCategory::KEYWORD_COL_NAME:
		return "KEYWORD_COL_NAME";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
KeywordCategory EnumUtil::FromString<KeywordCategory>(const char *value) {
	if (StringUtil::Equals(value, "KEYWORD_RESERVED")) {
		return KeywordCategory::KEYWORD_RESERVED;
	}
	if (StringUtil::Equals(value, "KEYWORD_UNRESERVED")) {
		return KeywordCategory::KEYWORD_UNRESERVED;
	}
	if (StringUtil::Equals(value, "KEYWORD_TYPE_FUNC")) {
		return KeywordCategory::KEYWORD_TYPE_FUNC;
	}
	if (StringUtil::Equals(value, "KEYWORD_COL_NAME")) {
		return KeywordCategory::KEYWORD_COL_NAME;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<ResultModifierType>(ResultModifierType value) {
	switch (value) {
	case ResultModifierType::LIMIT_MODIFIER:
		return "LIMIT_MODIFIER";
	case ResultModifierType::ORDER_MODIFIER:
		return "ORDER_MODIFIER";
	case ResultModifierType::DISTINCT_MODIFIER:
		return "DISTINCT_MODIFIER";
	case ResultModifierType::LIMIT_PERCENT_MODIFIER:
		return "LIMIT_PERCENT_MODIFIER";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
ResultModifierType EnumUtil::FromString<ResultModifierType>(const char *value) {
	if (StringUtil::Equals(value, "LIMIT_MODIFIER")) {
		return ResultModifierType::LIMIT_MODIFIER;
	}
	if (StringUtil::Equals(value, "ORDER_MODIFIER")) {
		return ResultModifierType::ORDER_MODIFIER;
	}
	if (StringUtil::Equals(value, "DISTINCT_MODIFIER")) {
		return ResultModifierType::DISTINCT_MODIFIER;
	}
	if (StringUtil::Equals(value, "LIMIT_PERCENT_MODIFIER")) {
		return ResultModifierType::LIMIT_PERCENT_MODIFIER;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<ConstraintType>(ConstraintType value) {
	switch (value) {
	case ConstraintType::INVALID:
		return "INVALID";
	case ConstraintType::NOT_NULL:
		return "NOT_NULL";
	case ConstraintType::CHECK:
		return "CHECK";
	case ConstraintType::UNIQUE:
		return "UNIQUE";
	case ConstraintType::FOREIGN_KEY:
		return "FOREIGN_KEY";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
ConstraintType EnumUtil::FromString<ConstraintType>(const char *value) {
	if (StringUtil::Equals(value, "INVALID")) {
		return ConstraintType::INVALID;
	}
	if (StringUtil::Equals(value, "NOT_NULL")) {
		return ConstraintType::NOT_NULL;
	}
	if (StringUtil::Equals(value, "CHECK")) {
		return ConstraintType::CHECK;
	}
	if (StringUtil::Equals(value, "UNIQUE")) {
		return ConstraintType::UNIQUE;
	}
	if (StringUtil::Equals(value, "FOREIGN_KEY")) {
		return ConstraintType::FOREIGN_KEY;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<ForeignKeyType>(ForeignKeyType value) {
	switch (value) {
	case ForeignKeyType::FK_TYPE_PRIMARY_KEY_TABLE:
		return "FK_TYPE_PRIMARY_KEY_TABLE";
	case ForeignKeyType::FK_TYPE_FOREIGN_KEY_TABLE:
		return "FK_TYPE_FOREIGN_KEY_TABLE";
	case ForeignKeyType::FK_TYPE_SELF_REFERENCE_TABLE:
		return "FK_TYPE_SELF_REFERENCE_TABLE";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
ForeignKeyType EnumUtil::FromString<ForeignKeyType>(const char *value) {
	if (StringUtil::Equals(value, "FK_TYPE_PRIMARY_KEY_TABLE")) {
		return ForeignKeyType::FK_TYPE_PRIMARY_KEY_TABLE;
	}
	if (StringUtil::Equals(value, "FK_TYPE_FOREIGN_KEY_TABLE")) {
		return ForeignKeyType::FK_TYPE_FOREIGN_KEY_TABLE;
	}
	if (StringUtil::Equals(value, "FK_TYPE_SELF_REFERENCE_TABLE")) {
		return ForeignKeyType::FK_TYPE_SELF_REFERENCE_TABLE;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<ParserExtensionResultType>(ParserExtensionResultType value) {
	switch (value) {
	case ParserExtensionResultType::PARSE_SUCCESSFUL:
		return "PARSE_SUCCESSFUL";
	case ParserExtensionResultType::DISPLAY_ORIGINAL_ERROR:
		return "DISPLAY_ORIGINAL_ERROR";
	case ParserExtensionResultType::DISPLAY_EXTENSION_ERROR:
		return "DISPLAY_EXTENSION_ERROR";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
ParserExtensionResultType EnumUtil::FromString<ParserExtensionResultType>(const char *value) {
	if (StringUtil::Equals(value, "PARSE_SUCCESSFUL")) {
		return ParserExtensionResultType::PARSE_SUCCESSFUL;
	}
	if (StringUtil::Equals(value, "DISPLAY_ORIGINAL_ERROR")) {
		return ParserExtensionResultType::DISPLAY_ORIGINAL_ERROR;
	}
	if (StringUtil::Equals(value, "DISPLAY_EXTENSION_ERROR")) {
		return ParserExtensionResultType::DISPLAY_EXTENSION_ERROR;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<QueryNodeType>(QueryNodeType value) {
	switch (value) {
	case QueryNodeType::SELECT_NODE:
		return "SELECT_NODE";
	case QueryNodeType::SET_OPERATION_NODE:
		return "SET_OPERATION_NODE";
	case QueryNodeType::BOUND_SUBQUERY_NODE:
		return "BOUND_SUBQUERY_NODE";
	case QueryNodeType::RECURSIVE_CTE_NODE:
		return "RECURSIVE_CTE_NODE";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
QueryNodeType EnumUtil::FromString<QueryNodeType>(const char *value) {
	if (StringUtil::Equals(value, "SELECT_NODE")) {
		return QueryNodeType::SELECT_NODE;
	}
	if (StringUtil::Equals(value, "SET_OPERATION_NODE")) {
		return QueryNodeType::SET_OPERATION_NODE;
	}
	if (StringUtil::Equals(value, "BOUND_SUBQUERY_NODE")) {
		return QueryNodeType::BOUND_SUBQUERY_NODE;
	}
	if (StringUtil::Equals(value, "RECURSIVE_CTE_NODE")) {
		return QueryNodeType::RECURSIVE_CTE_NODE;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<SequenceInfo>(SequenceInfo value) {
	switch (value) {
	case SequenceInfo::SEQ_START:
		return "SEQ_START";
	case SequenceInfo::SEQ_INC:
		return "SEQ_INC";
	case SequenceInfo::SEQ_MIN:
		return "SEQ_MIN";
	case SequenceInfo::SEQ_MAX:
		return "SEQ_MAX";
	case SequenceInfo::SEQ_CYCLE:
		return "SEQ_CYCLE";
	case SequenceInfo::SEQ_OWN:
		return "SEQ_OWN";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
SequenceInfo EnumUtil::FromString<SequenceInfo>(const char *value) {
	if (StringUtil::Equals(value, "SEQ_START")) {
		return SequenceInfo::SEQ_START;
	}
	if (StringUtil::Equals(value, "SEQ_INC")) {
		return SequenceInfo::SEQ_INC;
	}
	if (StringUtil::Equals(value, "SEQ_MIN")) {
		return SequenceInfo::SEQ_MIN;
	}
	if (StringUtil::Equals(value, "SEQ_MAX")) {
		return SequenceInfo::SEQ_MAX;
	}
	if (StringUtil::Equals(value, "SEQ_CYCLE")) {
		return SequenceInfo::SEQ_CYCLE;
	}
	if (StringUtil::Equals(value, "SEQ_OWN")) {
		return SequenceInfo::SEQ_OWN;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<AlterScalarFunctionType>(AlterScalarFunctionType value) {
	switch (value) {
	case AlterScalarFunctionType::INVALID:
		return "INVALID";
	case AlterScalarFunctionType::ADD_FUNCTION_OVERLOADS:
		return "ADD_FUNCTION_OVERLOADS";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
AlterScalarFunctionType EnumUtil::FromString<AlterScalarFunctionType>(const char *value) {
	if (StringUtil::Equals(value, "INVALID")) {
		return AlterScalarFunctionType::INVALID;
	}
	if (StringUtil::Equals(value, "ADD_FUNCTION_OVERLOADS")) {
		return AlterScalarFunctionType::ADD_FUNCTION_OVERLOADS;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<AlterTableType>(AlterTableType value) {
	switch (value) {
	case AlterTableType::INVALID:
		return "INVALID";
	case AlterTableType::RENAME_COLUMN:
		return "RENAME_COLUMN";
	case AlterTableType::RENAME_TABLE:
		return "RENAME_TABLE";
	case AlterTableType::ADD_COLUMN:
		return "ADD_COLUMN";
	case AlterTableType::REMOVE_COLUMN:
		return "REMOVE_COLUMN";
	case AlterTableType::ALTER_COLUMN_TYPE:
		return "ALTER_COLUMN_TYPE";
	case AlterTableType::SET_DEFAULT:
		return "SET_DEFAULT";
	case AlterTableType::FOREIGN_KEY_CONSTRAINT:
		return "FOREIGN_KEY_CONSTRAINT";
	case AlterTableType::SET_NOT_NULL:
		return "SET_NOT_NULL";
	case AlterTableType::DROP_NOT_NULL:
		return "DROP_NOT_NULL";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
AlterTableType EnumUtil::FromString<AlterTableType>(const char *value) {
	if (StringUtil::Equals(value, "INVALID")) {
		return AlterTableType::INVALID;
	}
	if (StringUtil::Equals(value, "RENAME_COLUMN")) {
		return AlterTableType::RENAME_COLUMN;
	}
	if (StringUtil::Equals(value, "RENAME_TABLE")) {
		return AlterTableType::RENAME_TABLE;
	}
	if (StringUtil::Equals(value, "ADD_COLUMN")) {
		return AlterTableType::ADD_COLUMN;
	}
	if (StringUtil::Equals(value, "REMOVE_COLUMN")) {
		return AlterTableType::REMOVE_COLUMN;
	}
	if (StringUtil::Equals(value, "ALTER_COLUMN_TYPE")) {
		return AlterTableType::ALTER_COLUMN_TYPE;
	}
	if (StringUtil::Equals(value, "SET_DEFAULT")) {
		return AlterTableType::SET_DEFAULT;
	}
	if (StringUtil::Equals(value, "FOREIGN_KEY_CONSTRAINT")) {
		return AlterTableType::FOREIGN_KEY_CONSTRAINT;
	}
	if (StringUtil::Equals(value, "SET_NOT_NULL")) {
		return AlterTableType::SET_NOT_NULL;
	}
	if (StringUtil::Equals(value, "DROP_NOT_NULL")) {
		return AlterTableType::DROP_NOT_NULL;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<AlterViewType>(AlterViewType value) {
	switch (value) {
	case AlterViewType::INVALID:
		return "INVALID";
	case AlterViewType::RENAME_VIEW:
		return "RENAME_VIEW";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
AlterViewType EnumUtil::FromString<AlterViewType>(const char *value) {
	if (StringUtil::Equals(value, "INVALID")) {
		return AlterViewType::INVALID;
	}
	if (StringUtil::Equals(value, "RENAME_VIEW")) {
		return AlterViewType::RENAME_VIEW;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<AlterTableFunctionType>(AlterTableFunctionType value) {
	switch (value) {
	case AlterTableFunctionType::INVALID:
		return "INVALID";
	case AlterTableFunctionType::ADD_FUNCTION_OVERLOADS:
		return "ADD_FUNCTION_OVERLOADS";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
AlterTableFunctionType EnumUtil::FromString<AlterTableFunctionType>(const char *value) {
	if (StringUtil::Equals(value, "INVALID")) {
		return AlterTableFunctionType::INVALID;
	}
	if (StringUtil::Equals(value, "ADD_FUNCTION_OVERLOADS")) {
		return AlterTableFunctionType::ADD_FUNCTION_OVERLOADS;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<AlterType>(AlterType value) {
	switch (value) {
	case AlterType::INVALID:
		return "INVALID";
	case AlterType::ALTER_TABLE:
		return "ALTER_TABLE";
	case AlterType::ALTER_VIEW:
		return "ALTER_VIEW";
	case AlterType::ALTER_SEQUENCE:
		return "ALTER_SEQUENCE";
	case AlterType::CHANGE_OWNERSHIP:
		return "CHANGE_OWNERSHIP";
	case AlterType::ALTER_SCALAR_FUNCTION:
		return "ALTER_SCALAR_FUNCTION";
	case AlterType::ALTER_TABLE_FUNCTION:
		return "ALTER_TABLE_FUNCTION";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
AlterType EnumUtil::FromString<AlterType>(const char *value) {
	if (StringUtil::Equals(value, "INVALID")) {
		return AlterType::INVALID;
	}
	if (StringUtil::Equals(value, "ALTER_TABLE")) {
		return AlterType::ALTER_TABLE;
	}
	if (StringUtil::Equals(value, "ALTER_VIEW")) {
		return AlterType::ALTER_VIEW;
	}
	if (StringUtil::Equals(value, "ALTER_SEQUENCE")) {
		return AlterType::ALTER_SEQUENCE;
	}
	if (StringUtil::Equals(value, "CHANGE_OWNERSHIP")) {
		return AlterType::CHANGE_OWNERSHIP;
	}
	if (StringUtil::Equals(value, "ALTER_SCALAR_FUNCTION")) {
		return AlterType::ALTER_SCALAR_FUNCTION;
	}
	if (StringUtil::Equals(value, "ALTER_TABLE_FUNCTION")) {
		return AlterType::ALTER_TABLE_FUNCTION;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<PragmaType>(PragmaType value) {
	switch (value) {
	case PragmaType::PRAGMA_STATEMENT:
		return "PRAGMA_STATEMENT";
	case PragmaType::PRAGMA_CALL:
		return "PRAGMA_CALL";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
PragmaType EnumUtil::FromString<PragmaType>(const char *value) {
	if (StringUtil::Equals(value, "PRAGMA_STATEMENT")) {
		return PragmaType::PRAGMA_STATEMENT;
	}
	if (StringUtil::Equals(value, "PRAGMA_CALL")) {
		return PragmaType::PRAGMA_CALL;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<OnCreateConflict>(OnCreateConflict value) {
	switch (value) {
	case OnCreateConflict::ERROR_ON_CONFLICT:
		return "ERROR_ON_CONFLICT";
	case OnCreateConflict::IGNORE_ON_CONFLICT:
		return "IGNORE_ON_CONFLICT";
	case OnCreateConflict::REPLACE_ON_CONFLICT:
		return "REPLACE_ON_CONFLICT";
	case OnCreateConflict::ALTER_ON_CONFLICT:
		return "ALTER_ON_CONFLICT";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
OnCreateConflict EnumUtil::FromString<OnCreateConflict>(const char *value) {
	if (StringUtil::Equals(value, "ERROR_ON_CONFLICT")) {
		return OnCreateConflict::ERROR_ON_CONFLICT;
	}
	if (StringUtil::Equals(value, "IGNORE_ON_CONFLICT")) {
		return OnCreateConflict::IGNORE_ON_CONFLICT;
	}
	if (StringUtil::Equals(value, "REPLACE_ON_CONFLICT")) {
		return OnCreateConflict::REPLACE_ON_CONFLICT;
	}
	if (StringUtil::Equals(value, "ALTER_ON_CONFLICT")) {
		return OnCreateConflict::ALTER_ON_CONFLICT;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<TransactionType>(TransactionType value) {
	switch (value) {
	case TransactionType::INVALID:
		return "INVALID";
	case TransactionType::BEGIN_TRANSACTION:
		return "BEGIN_TRANSACTION";
	case TransactionType::COMMIT:
		return "COMMIT";
	case TransactionType::ROLLBACK:
		return "ROLLBACK";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
TransactionType EnumUtil::FromString<TransactionType>(const char *value) {
	if (StringUtil::Equals(value, "INVALID")) {
		return TransactionType::INVALID;
	}
	if (StringUtil::Equals(value, "BEGIN_TRANSACTION")) {
		return TransactionType::BEGIN_TRANSACTION;
	}
	if (StringUtil::Equals(value, "COMMIT")) {
		return TransactionType::COMMIT;
	}
	if (StringUtil::Equals(value, "ROLLBACK")) {
		return TransactionType::ROLLBACK;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<SampleMethod>(SampleMethod value) {
	switch (value) {
	case SampleMethod::SYSTEM_SAMPLE:
		return "System";
	case SampleMethod::BERNOULLI_SAMPLE:
		return "Bernoulli";
	case SampleMethod::RESERVOIR_SAMPLE:
		return "Reservoir";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
SampleMethod EnumUtil::FromString<SampleMethod>(const char *value) {
	if (StringUtil::Equals(value, "System")) {
		return SampleMethod::SYSTEM_SAMPLE;
	}
	if (StringUtil::Equals(value, "Bernoulli")) {
		return SampleMethod::BERNOULLI_SAMPLE;
	}
	if (StringUtil::Equals(value, "Reservoir")) {
		return SampleMethod::RESERVOIR_SAMPLE;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<ExplainType>(ExplainType value) {
	switch (value) {
	case ExplainType::EXPLAIN_STANDARD:
		return "EXPLAIN_STANDARD";
	case ExplainType::EXPLAIN_ANALYZE:
		return "EXPLAIN_ANALYZE";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
ExplainType EnumUtil::FromString<ExplainType>(const char *value) {
	if (StringUtil::Equals(value, "EXPLAIN_STANDARD")) {
		return ExplainType::EXPLAIN_STANDARD;
	}
	if (StringUtil::Equals(value, "EXPLAIN_ANALYZE")) {
		return ExplainType::EXPLAIN_ANALYZE;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<OnConflictAction>(OnConflictAction value) {
	switch (value) {
	case OnConflictAction::THROW:
		return "THROW";
	case OnConflictAction::NOTHING:
		return "NOTHING";
	case OnConflictAction::UPDATE:
		return "UPDATE";
	case OnConflictAction::REPLACE:
		return "REPLACE";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
OnConflictAction EnumUtil::FromString<OnConflictAction>(const char *value) {
	if (StringUtil::Equals(value, "THROW")) {
		return OnConflictAction::THROW;
	}
	if (StringUtil::Equals(value, "NOTHING")) {
		return OnConflictAction::NOTHING;
	}
	if (StringUtil::Equals(value, "UPDATE")) {
		return OnConflictAction::UPDATE;
	}
	if (StringUtil::Equals(value, "REPLACE")) {
		return OnConflictAction::REPLACE;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<WindowBoundary>(WindowBoundary value) {
	switch (value) {
	case WindowBoundary::INVALID:
		return "INVALID";
	case WindowBoundary::UNBOUNDED_PRECEDING:
		return "UNBOUNDED_PRECEDING";
	case WindowBoundary::UNBOUNDED_FOLLOWING:
		return "UNBOUNDED_FOLLOWING";
	case WindowBoundary::CURRENT_ROW_RANGE:
		return "CURRENT_ROW_RANGE";
	case WindowBoundary::CURRENT_ROW_ROWS:
		return "CURRENT_ROW_ROWS";
	case WindowBoundary::EXPR_PRECEDING_ROWS:
		return "EXPR_PRECEDING_ROWS";
	case WindowBoundary::EXPR_FOLLOWING_ROWS:
		return "EXPR_FOLLOWING_ROWS";
	case WindowBoundary::EXPR_PRECEDING_RANGE:
		return "EXPR_PRECEDING_RANGE";
	case WindowBoundary::EXPR_FOLLOWING_RANGE:
		return "EXPR_FOLLOWING_RANGE";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
WindowBoundary EnumUtil::FromString<WindowBoundary>(const char *value) {
	if (StringUtil::Equals(value, "INVALID")) {
		return WindowBoundary::INVALID;
	}
	if (StringUtil::Equals(value, "UNBOUNDED_PRECEDING")) {
		return WindowBoundary::UNBOUNDED_PRECEDING;
	}
	if (StringUtil::Equals(value, "UNBOUNDED_FOLLOWING")) {
		return WindowBoundary::UNBOUNDED_FOLLOWING;
	}
	if (StringUtil::Equals(value, "CURRENT_ROW_RANGE")) {
		return WindowBoundary::CURRENT_ROW_RANGE;
	}
	if (StringUtil::Equals(value, "CURRENT_ROW_ROWS")) {
		return WindowBoundary::CURRENT_ROW_ROWS;
	}
	if (StringUtil::Equals(value, "EXPR_PRECEDING_ROWS")) {
		return WindowBoundary::EXPR_PRECEDING_ROWS;
	}
	if (StringUtil::Equals(value, "EXPR_FOLLOWING_ROWS")) {
		return WindowBoundary::EXPR_FOLLOWING_ROWS;
	}
	if (StringUtil::Equals(value, "EXPR_PRECEDING_RANGE")) {
		return WindowBoundary::EXPR_PRECEDING_RANGE;
	}
	if (StringUtil::Equals(value, "EXPR_FOLLOWING_RANGE")) {
		return WindowBoundary::EXPR_FOLLOWING_RANGE;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<DataFileType>(DataFileType value) {
	switch (value) {
	case DataFileType::FILE_DOES_NOT_EXIST:
		return "FILE_DOES_NOT_EXIST";
	case DataFileType::DUCKDB_FILE:
		return "DUCKDB_FILE";
	case DataFileType::SQLITE_FILE:
		return "SQLITE_FILE";
	case DataFileType::PARQUET_FILE:
		return "PARQUET_FILE";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
DataFileType EnumUtil::FromString<DataFileType>(const char *value) {
	if (StringUtil::Equals(value, "FILE_DOES_NOT_EXIST")) {
		return DataFileType::FILE_DOES_NOT_EXIST;
	}
	if (StringUtil::Equals(value, "DUCKDB_FILE")) {
		return DataFileType::DUCKDB_FILE;
	}
	if (StringUtil::Equals(value, "SQLITE_FILE")) {
		return DataFileType::SQLITE_FILE;
	}
	if (StringUtil::Equals(value, "PARQUET_FILE")) {
		return DataFileType::PARQUET_FILE;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<StatsInfo>(StatsInfo value) {
	switch (value) {
	case StatsInfo::CAN_HAVE_NULL_VALUES:
		return "CAN_HAVE_NULL_VALUES";
	case StatsInfo::CANNOT_HAVE_NULL_VALUES:
		return "CANNOT_HAVE_NULL_VALUES";
	case StatsInfo::CAN_HAVE_VALID_VALUES:
		return "CAN_HAVE_VALID_VALUES";
	case StatsInfo::CANNOT_HAVE_VALID_VALUES:
		return "CANNOT_HAVE_VALID_VALUES";
	case StatsInfo::CAN_HAVE_NULL_AND_VALID_VALUES:
		return "CAN_HAVE_NULL_AND_VALID_VALUES";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
StatsInfo EnumUtil::FromString<StatsInfo>(const char *value) {
	if (StringUtil::Equals(value, "CAN_HAVE_NULL_VALUES")) {
		return StatsInfo::CAN_HAVE_NULL_VALUES;
	}
	if (StringUtil::Equals(value, "CANNOT_HAVE_NULL_VALUES")) {
		return StatsInfo::CANNOT_HAVE_NULL_VALUES;
	}
	if (StringUtil::Equals(value, "CAN_HAVE_VALID_VALUES")) {
		return StatsInfo::CAN_HAVE_VALID_VALUES;
	}
	if (StringUtil::Equals(value, "CANNOT_HAVE_VALID_VALUES")) {
		return StatsInfo::CANNOT_HAVE_VALID_VALUES;
	}
	if (StringUtil::Equals(value, "CAN_HAVE_NULL_AND_VALID_VALUES")) {
		return StatsInfo::CAN_HAVE_NULL_AND_VALID_VALUES;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<StatisticsType>(StatisticsType value) {
	switch (value) {
	case StatisticsType::NUMERIC_STATS:
		return "NUMERIC_STATS";
	case StatisticsType::STRING_STATS:
		return "STRING_STATS";
	case StatisticsType::LIST_STATS:
		return "LIST_STATS";
	case StatisticsType::STRUCT_STATS:
		return "STRUCT_STATS";
	case StatisticsType::BASE_STATS:
		return "BASE_STATS";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
StatisticsType EnumUtil::FromString<StatisticsType>(const char *value) {
	if (StringUtil::Equals(value, "NUMERIC_STATS")) {
		return StatisticsType::NUMERIC_STATS;
	}
	if (StringUtil::Equals(value, "STRING_STATS")) {
		return StatisticsType::STRING_STATS;
	}
	if (StringUtil::Equals(value, "LIST_STATS")) {
		return StatisticsType::LIST_STATS;
	}
	if (StringUtil::Equals(value, "STRUCT_STATS")) {
		return StatisticsType::STRUCT_STATS;
	}
	if (StringUtil::Equals(value, "BASE_STATS")) {
		return StatisticsType::BASE_STATS;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<ColumnSegmentType>(ColumnSegmentType value) {
	switch (value) {
	case ColumnSegmentType::TRANSIENT:
		return "TRANSIENT";
	case ColumnSegmentType::PERSISTENT:
		return "PERSISTENT";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
ColumnSegmentType EnumUtil::FromString<ColumnSegmentType>(const char *value) {
	if (StringUtil::Equals(value, "TRANSIENT")) {
		return ColumnSegmentType::TRANSIENT;
	}
	if (StringUtil::Equals(value, "PERSISTENT")) {
		return ColumnSegmentType::PERSISTENT;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<ChunkInfoType>(ChunkInfoType value) {
	switch (value) {
	case ChunkInfoType::CONSTANT_INFO:
		return "CONSTANT_INFO";
	case ChunkInfoType::VECTOR_INFO:
		return "VECTOR_INFO";
	case ChunkInfoType::EMPTY_INFO:
		return "EMPTY_INFO";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
ChunkInfoType EnumUtil::FromString<ChunkInfoType>(const char *value) {
	if (StringUtil::Equals(value, "CONSTANT_INFO")) {
		return ChunkInfoType::CONSTANT_INFO;
	}
	if (StringUtil::Equals(value, "VECTOR_INFO")) {
		return ChunkInfoType::VECTOR_INFO;
	}
	if (StringUtil::Equals(value, "EMPTY_INFO")) {
		return ChunkInfoType::EMPTY_INFO;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<BitpackingMode>(BitpackingMode value) {
	switch (value) {
	case BitpackingMode::AUTO:
		return "AUTO";
	case BitpackingMode::CONSTANT:
		return "CONSTANT";
	case BitpackingMode::CONSTANT_DELTA:
		return "CONSTANT_DELTA";
	case BitpackingMode::DELTA_FOR:
		return "DELTA_FOR";
	case BitpackingMode::FOR:
		return "FOR";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
BitpackingMode EnumUtil::FromString<BitpackingMode>(const char *value) {
	if (StringUtil::Equals(value, "AUTO")) {
		return BitpackingMode::AUTO;
	}
	if (StringUtil::Equals(value, "CONSTANT")) {
		return BitpackingMode::CONSTANT;
	}
	if (StringUtil::Equals(value, "CONSTANT_DELTA")) {
		return BitpackingMode::CONSTANT_DELTA;
	}
	if (StringUtil::Equals(value, "DELTA_FOR")) {
		return BitpackingMode::DELTA_FOR;
	}
	if (StringUtil::Equals(value, "FOR")) {
		return BitpackingMode::FOR;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<BlockState>(BlockState value) {
	switch (value) {
	case BlockState::BLOCK_UNLOADED:
		return "BLOCK_UNLOADED";
	case BlockState::BLOCK_LOADED:
		return "BLOCK_LOADED";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
BlockState EnumUtil::FromString<BlockState>(const char *value) {
	if (StringUtil::Equals(value, "BLOCK_UNLOADED")) {
		return BlockState::BLOCK_UNLOADED;
	}
	if (StringUtil::Equals(value, "BLOCK_LOADED")) {
		return BlockState::BLOCK_LOADED;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<VerificationType>(VerificationType value) {
	switch (value) {
	case VerificationType::ORIGINAL:
		return "ORIGINAL";
	case VerificationType::COPIED:
		return "COPIED";
	case VerificationType::DESERIALIZED:
		return "DESERIALIZED";
	case VerificationType::DESERIALIZED_V2:
		return "DESERIALIZED_V2";
	case VerificationType::PARSED:
		return "PARSED";
	case VerificationType::UNOPTIMIZED:
		return "UNOPTIMIZED";
	case VerificationType::NO_OPERATOR_CACHING:
		return "NO_OPERATOR_CACHING";
	case VerificationType::PREPARED:
		return "PREPARED";
	case VerificationType::EXTERNAL:
		return "EXTERNAL";
	case VerificationType::INVALID:
		return "INVALID";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
VerificationType EnumUtil::FromString<VerificationType>(const char *value) {
	if (StringUtil::Equals(value, "ORIGINAL")) {
		return VerificationType::ORIGINAL;
	}
	if (StringUtil::Equals(value, "COPIED")) {
		return VerificationType::COPIED;
	}
	if (StringUtil::Equals(value, "DESERIALIZED")) {
		return VerificationType::DESERIALIZED;
	}
	if (StringUtil::Equals(value, "DESERIALIZED_V2")) {
		return VerificationType::DESERIALIZED_V2;
	}
	if (StringUtil::Equals(value, "PARSED")) {
		return VerificationType::PARSED;
	}
	if (StringUtil::Equals(value, "UNOPTIMIZED")) {
		return VerificationType::UNOPTIMIZED;
	}
	if (StringUtil::Equals(value, "NO_OPERATOR_CACHING")) {
		return VerificationType::NO_OPERATOR_CACHING;
	}
	if (StringUtil::Equals(value, "PREPARED")) {
		return VerificationType::PREPARED;
	}
	if (StringUtil::Equals(value, "EXTERNAL")) {
		return VerificationType::EXTERNAL;
	}
	if (StringUtil::Equals(value, "INVALID")) {
		return VerificationType::INVALID;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<FileLockType>(FileLockType value) {
	switch (value) {
	case FileLockType::NO_LOCK:
		return "NO_LOCK";
	case FileLockType::READ_LOCK:
		return "READ_LOCK";
	case FileLockType::WRITE_LOCK:
		return "WRITE_LOCK";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
FileLockType EnumUtil::FromString<FileLockType>(const char *value) {
	if (StringUtil::Equals(value, "NO_LOCK")) {
		return FileLockType::NO_LOCK;
	}
	if (StringUtil::Equals(value, "READ_LOCK")) {
		return FileLockType::READ_LOCK;
	}
	if (StringUtil::Equals(value, "WRITE_LOCK")) {
		return FileLockType::WRITE_LOCK;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<FileBufferType>(FileBufferType value) {
	switch (value) {
	case FileBufferType::BLOCK:
		return "BLOCK";
	case FileBufferType::MANAGED_BUFFER:
		return "MANAGED_BUFFER";
	case FileBufferType::TINY_BUFFER:
		return "TINY_BUFFER";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
FileBufferType EnumUtil::FromString<FileBufferType>(const char *value) {
	if (StringUtil::Equals(value, "BLOCK")) {
		return FileBufferType::BLOCK;
	}
	if (StringUtil::Equals(value, "MANAGED_BUFFER")) {
		return FileBufferType::MANAGED_BUFFER;
	}
	if (StringUtil::Equals(value, "TINY_BUFFER")) {
		return FileBufferType::TINY_BUFFER;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<ExceptionFormatValueType>(ExceptionFormatValueType value) {
	switch (value) {
	case ExceptionFormatValueType::FORMAT_VALUE_TYPE_DOUBLE:
		return "FORMAT_VALUE_TYPE_DOUBLE";
	case ExceptionFormatValueType::FORMAT_VALUE_TYPE_INTEGER:
		return "FORMAT_VALUE_TYPE_INTEGER";
	case ExceptionFormatValueType::FORMAT_VALUE_TYPE_STRING:
		return "FORMAT_VALUE_TYPE_STRING";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
ExceptionFormatValueType EnumUtil::FromString<ExceptionFormatValueType>(const char *value) {
	if (StringUtil::Equals(value, "FORMAT_VALUE_TYPE_DOUBLE")) {
		return ExceptionFormatValueType::FORMAT_VALUE_TYPE_DOUBLE;
	}
	if (StringUtil::Equals(value, "FORMAT_VALUE_TYPE_INTEGER")) {
		return ExceptionFormatValueType::FORMAT_VALUE_TYPE_INTEGER;
	}
	if (StringUtil::Equals(value, "FORMAT_VALUE_TYPE_STRING")) {
		return ExceptionFormatValueType::FORMAT_VALUE_TYPE_STRING;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<ExtraTypeInfoType>(ExtraTypeInfoType value) {
	switch (value) {
	case ExtraTypeInfoType::INVALID_TYPE_INFO:
		return "INVALID_TYPE_INFO";
	case ExtraTypeInfoType::GENERIC_TYPE_INFO:
		return "GENERIC_TYPE_INFO";
	case ExtraTypeInfoType::DECIMAL_TYPE_INFO:
		return "DECIMAL_TYPE_INFO";
	case ExtraTypeInfoType::STRING_TYPE_INFO:
		return "STRING_TYPE_INFO";
	case ExtraTypeInfoType::LIST_TYPE_INFO:
		return "LIST_TYPE_INFO";
	case ExtraTypeInfoType::STRUCT_TYPE_INFO:
		return "STRUCT_TYPE_INFO";
	case ExtraTypeInfoType::ENUM_TYPE_INFO:
		return "ENUM_TYPE_INFO";
	case ExtraTypeInfoType::USER_TYPE_INFO:
		return "USER_TYPE_INFO";
	case ExtraTypeInfoType::AGGREGATE_STATE_TYPE_INFO:
		return "AGGREGATE_STATE_TYPE_INFO";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
ExtraTypeInfoType EnumUtil::FromString<ExtraTypeInfoType>(const char *value) {
	if (StringUtil::Equals(value, "INVALID_TYPE_INFO")) {
		return ExtraTypeInfoType::INVALID_TYPE_INFO;
	}
	if (StringUtil::Equals(value, "GENERIC_TYPE_INFO")) {
		return ExtraTypeInfoType::GENERIC_TYPE_INFO;
	}
	if (StringUtil::Equals(value, "DECIMAL_TYPE_INFO")) {
		return ExtraTypeInfoType::DECIMAL_TYPE_INFO;
	}
	if (StringUtil::Equals(value, "STRING_TYPE_INFO")) {
		return ExtraTypeInfoType::STRING_TYPE_INFO;
	}
	if (StringUtil::Equals(value, "LIST_TYPE_INFO")) {
		return ExtraTypeInfoType::LIST_TYPE_INFO;
	}
	if (StringUtil::Equals(value, "STRUCT_TYPE_INFO")) {
		return ExtraTypeInfoType::STRUCT_TYPE_INFO;
	}
	if (StringUtil::Equals(value, "ENUM_TYPE_INFO")) {
		return ExtraTypeInfoType::ENUM_TYPE_INFO;
	}
	if (StringUtil::Equals(value, "USER_TYPE_INFO")) {
		return ExtraTypeInfoType::USER_TYPE_INFO;
	}
	if (StringUtil::Equals(value, "AGGREGATE_STATE_TYPE_INFO")) {
		return ExtraTypeInfoType::AGGREGATE_STATE_TYPE_INFO;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<PhysicalType>(PhysicalType value) {
	switch (value) {
	case PhysicalType::BOOL:
		return "BOOL";
	case PhysicalType::UINT8:
		return "UINT8";
	case PhysicalType::INT8:
		return "INT8";
	case PhysicalType::UINT16:
		return "UINT16";
	case PhysicalType::INT16:
		return "INT16";
	case PhysicalType::UINT32:
		return "UINT32";
	case PhysicalType::INT32:
		return "INT32";
	case PhysicalType::UINT64:
		return "UINT64";
	case PhysicalType::INT64:
		return "INT64";
	case PhysicalType::FLOAT:
		return "FLOAT";
	case PhysicalType::DOUBLE:
		return "DOUBLE";
	case PhysicalType::INTERVAL:
		return "INTERVAL";
	case PhysicalType::LIST:
		return "LIST";
	case PhysicalType::STRUCT:
		return "STRUCT";
	case PhysicalType::VARCHAR:
		return "VARCHAR";
	case PhysicalType::INT128:
		return "INT128";
	case PhysicalType::UNKNOWN:
		return "UNKNOWN";
	case PhysicalType::BIT:
		return "BIT";
	case PhysicalType::INVALID:
		return "INVALID";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
PhysicalType EnumUtil::FromString<PhysicalType>(const char *value) {
	if (StringUtil::Equals(value, "BOOL")) {
		return PhysicalType::BOOL;
	}
	if (StringUtil::Equals(value, "UINT8")) {
		return PhysicalType::UINT8;
	}
	if (StringUtil::Equals(value, "INT8")) {
		return PhysicalType::INT8;
	}
	if (StringUtil::Equals(value, "UINT16")) {
		return PhysicalType::UINT16;
	}
	if (StringUtil::Equals(value, "INT16")) {
		return PhysicalType::INT16;
	}
	if (StringUtil::Equals(value, "UINT32")) {
		return PhysicalType::UINT32;
	}
	if (StringUtil::Equals(value, "INT32")) {
		return PhysicalType::INT32;
	}
	if (StringUtil::Equals(value, "UINT64")) {
		return PhysicalType::UINT64;
	}
	if (StringUtil::Equals(value, "INT64")) {
		return PhysicalType::INT64;
	}
	if (StringUtil::Equals(value, "FLOAT")) {
		return PhysicalType::FLOAT;
	}
	if (StringUtil::Equals(value, "DOUBLE")) {
		return PhysicalType::DOUBLE;
	}
	if (StringUtil::Equals(value, "INTERVAL")) {
		return PhysicalType::INTERVAL;
	}
	if (StringUtil::Equals(value, "LIST")) {
		return PhysicalType::LIST;
	}
	if (StringUtil::Equals(value, "STRUCT")) {
		return PhysicalType::STRUCT;
	}
	if (StringUtil::Equals(value, "VARCHAR")) {
		return PhysicalType::VARCHAR;
	}
	if (StringUtil::Equals(value, "INT128")) {
		return PhysicalType::INT128;
	}
	if (StringUtil::Equals(value, "UNKNOWN")) {
		return PhysicalType::UNKNOWN;
	}
	if (StringUtil::Equals(value, "BIT")) {
		return PhysicalType::BIT;
	}
	if (StringUtil::Equals(value, "INVALID")) {
		return PhysicalType::INVALID;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<LogicalTypeId>(LogicalTypeId value) {
	switch (value) {
	case LogicalTypeId::INVALID:
		return "INVALID";
	case LogicalTypeId::SQLNULL:
		return "NULL";
	case LogicalTypeId::UNKNOWN:
		return "UNKNOWN";
	case LogicalTypeId::ANY:
		return "ANY";
	case LogicalTypeId::USER:
		return "USER";
	case LogicalTypeId::BOOLEAN:
		return "BOOLEAN";
	case LogicalTypeId::TINYINT:
		return "TINYINT";
	case LogicalTypeId::SMALLINT:
		return "SMALLINT";
	case LogicalTypeId::INTEGER:
		return "INTEGER";
	case LogicalTypeId::BIGINT:
		return "BIGINT";
	case LogicalTypeId::DATE:
		return "DATE";
	case LogicalTypeId::TIME:
		return "TIME";
	case LogicalTypeId::TIMESTAMP_SEC:
		return "TIMESTAMP_S";
	case LogicalTypeId::TIMESTAMP_MS:
		return "TIMESTAMP_MS";
	case LogicalTypeId::TIMESTAMP:
		return "TIMESTAMP";
	case LogicalTypeId::TIMESTAMP_NS:
		return "TIMESTAMP_NS";
	case LogicalTypeId::DECIMAL:
		return "DECIMAL";
	case LogicalTypeId::FLOAT:
		return "FLOAT";
	case LogicalTypeId::DOUBLE:
		return "DOUBLE";
	case LogicalTypeId::CHAR:
		return "CHAR";
	case LogicalTypeId::VARCHAR:
		return "VARCHAR";
	case LogicalTypeId::BLOB:
		return "BLOB";
	case LogicalTypeId::INTERVAL:
		return "INTERVAL";
	case LogicalTypeId::UTINYINT:
		return "UTINYINT";
	case LogicalTypeId::USMALLINT:
		return "USMALLINT";
	case LogicalTypeId::UINTEGER:
		return "UINTEGER";
	case LogicalTypeId::UBIGINT:
		return "UBIGINT";
	case LogicalTypeId::TIMESTAMP_TZ:
		return "TIMESTAMP WITH TIME ZONE";
	case LogicalTypeId::TIME_TZ:
		return "TIME WITH TIME ZONE";
	case LogicalTypeId::BIT:
		return "BIT";
	case LogicalTypeId::HUGEINT:
		return "HUGEINT";
	case LogicalTypeId::POINTER:
		return "POINTER";
	case LogicalTypeId::VALIDITY:
		return "VALIDITY";
	case LogicalTypeId::UUID:
		return "UUID";
	case LogicalTypeId::STRUCT:
		return "STRUCT";
	case LogicalTypeId::LIST:
		return "LIST";
	case LogicalTypeId::MAP:
		return "MAP";
	case LogicalTypeId::TABLE:
		return "TABLE";
	case LogicalTypeId::ENUM:
		return "ENUM";
	case LogicalTypeId::AGGREGATE_STATE:
		return "AGGREGATE_STATE";
	case LogicalTypeId::LAMBDA:
		return "LAMBDA";
	case LogicalTypeId::UNION:
		return "UNION";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
LogicalTypeId EnumUtil::FromString<LogicalTypeId>(const char *value) {
	if (StringUtil::Equals(value, "INVALID")) {
		return LogicalTypeId::INVALID;
	}
	if (StringUtil::Equals(value, "NULL")) {
		return LogicalTypeId::SQLNULL;
	}
	if (StringUtil::Equals(value, "UNKNOWN")) {
		return LogicalTypeId::UNKNOWN;
	}
	if (StringUtil::Equals(value, "ANY")) {
		return LogicalTypeId::ANY;
	}
	if (StringUtil::Equals(value, "USER")) {
		return LogicalTypeId::USER;
	}
	if (StringUtil::Equals(value, "BOOLEAN")) {
		return LogicalTypeId::BOOLEAN;
	}
	if (StringUtil::Equals(value, "TINYINT")) {
		return LogicalTypeId::TINYINT;
	}
	if (StringUtil::Equals(value, "SMALLINT")) {
		return LogicalTypeId::SMALLINT;
	}
	if (StringUtil::Equals(value, "INTEGER")) {
		return LogicalTypeId::INTEGER;
	}
	if (StringUtil::Equals(value, "BIGINT")) {
		return LogicalTypeId::BIGINT;
	}
	if (StringUtil::Equals(value, "DATE")) {
		return LogicalTypeId::DATE;
	}
	if (StringUtil::Equals(value, "TIME")) {
		return LogicalTypeId::TIME;
	}
	if (StringUtil::Equals(value, "TIMESTAMP_S")) {
		return LogicalTypeId::TIMESTAMP_SEC;
	}
	if (StringUtil::Equals(value, "TIMESTAMP_MS")) {
		return LogicalTypeId::TIMESTAMP_MS;
	}
	if (StringUtil::Equals(value, "TIMESTAMP")) {
		return LogicalTypeId::TIMESTAMP;
	}
	if (StringUtil::Equals(value, "TIMESTAMP_NS")) {
		return LogicalTypeId::TIMESTAMP_NS;
	}
	if (StringUtil::Equals(value, "DECIMAL")) {
		return LogicalTypeId::DECIMAL;
	}
	if (StringUtil::Equals(value, "FLOAT")) {
		return LogicalTypeId::FLOAT;
	}
	if (StringUtil::Equals(value, "DOUBLE")) {
		return LogicalTypeId::DOUBLE;
	}
	if (StringUtil::Equals(value, "CHAR")) {
		return LogicalTypeId::CHAR;
	}
	if (StringUtil::Equals(value, "VARCHAR")) {
		return LogicalTypeId::VARCHAR;
	}
	if (StringUtil::Equals(value, "BLOB")) {
		return LogicalTypeId::BLOB;
	}
	if (StringUtil::Equals(value, "INTERVAL")) {
		return LogicalTypeId::INTERVAL;
	}
	if (StringUtil::Equals(value, "UTINYINT")) {
		return LogicalTypeId::UTINYINT;
	}
	if (StringUtil::Equals(value, "USMALLINT")) {
		return LogicalTypeId::USMALLINT;
	}
	if (StringUtil::Equals(value, "UINTEGER")) {
		return LogicalTypeId::UINTEGER;
	}
	if (StringUtil::Equals(value, "UBIGINT")) {
		return LogicalTypeId::UBIGINT;
	}
	if (StringUtil::Equals(value, "TIMESTAMP WITH TIME ZONE")) {
		return LogicalTypeId::TIMESTAMP_TZ;
	}
	if (StringUtil::Equals(value, "TIME WITH TIME ZONE")) {
		return LogicalTypeId::TIME_TZ;
	}
	if (StringUtil::Equals(value, "BIT")) {
		return LogicalTypeId::BIT;
	}
	if (StringUtil::Equals(value, "HUGEINT")) {
		return LogicalTypeId::HUGEINT;
	}
	if (StringUtil::Equals(value, "POINTER")) {
		return LogicalTypeId::POINTER;
	}
	if (StringUtil::Equals(value, "VALIDITY")) {
		return LogicalTypeId::VALIDITY;
	}
	if (StringUtil::Equals(value, "UUID")) {
		return LogicalTypeId::UUID;
	}
	if (StringUtil::Equals(value, "STRUCT")) {
		return LogicalTypeId::STRUCT;
	}
	if (StringUtil::Equals(value, "LIST")) {
		return LogicalTypeId::LIST;
	}
	if (StringUtil::Equals(value, "MAP")) {
		return LogicalTypeId::MAP;
	}
	if (StringUtil::Equals(value, "TABLE")) {
		return LogicalTypeId::TABLE;
	}
	if (StringUtil::Equals(value, "ENUM")) {
		return LogicalTypeId::ENUM;
	}
	if (StringUtil::Equals(value, "AGGREGATE_STATE")) {
		return LogicalTypeId::AGGREGATE_STATE;
	}
	if (StringUtil::Equals(value, "LAMBDA")) {
		return LogicalTypeId::LAMBDA;
	}
	if (StringUtil::Equals(value, "UNION")) {
		return LogicalTypeId::UNION;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<OutputStream>(OutputStream value) {
	switch (value) {
	case OutputStream::STREAM_STDOUT:
		return "STREAM_STDOUT";
	case OutputStream::STREAM_STDERR:
		return "STREAM_STDERR";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
OutputStream EnumUtil::FromString<OutputStream>(const char *value) {
	if (StringUtil::Equals(value, "STREAM_STDOUT")) {
		return OutputStream::STREAM_STDOUT;
	}
	if (StringUtil::Equals(value, "STREAM_STDERR")) {
		return OutputStream::STREAM_STDERR;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<TimestampCastResult>(TimestampCastResult value) {
	switch (value) {
	case TimestampCastResult::SUCCESS:
		return "SUCCESS";
	case TimestampCastResult::ERROR_INCORRECT_FORMAT:
		return "ERROR_INCORRECT_FORMAT";
	case TimestampCastResult::ERROR_NON_UTC_TIMEZONE:
		return "ERROR_NON_UTC_TIMEZONE";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
TimestampCastResult EnumUtil::FromString<TimestampCastResult>(const char *value) {
	if (StringUtil::Equals(value, "SUCCESS")) {
		return TimestampCastResult::SUCCESS;
	}
	if (StringUtil::Equals(value, "ERROR_INCORRECT_FORMAT")) {
		return TimestampCastResult::ERROR_INCORRECT_FORMAT;
	}
	if (StringUtil::Equals(value, "ERROR_NON_UTC_TIMEZONE")) {
		return TimestampCastResult::ERROR_NON_UTC_TIMEZONE;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<ConflictManagerMode>(ConflictManagerMode value) {
	switch (value) {
	case ConflictManagerMode::SCAN:
		return "SCAN";
	case ConflictManagerMode::THROW:
		return "THROW";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
ConflictManagerMode EnumUtil::FromString<ConflictManagerMode>(const char *value) {
	if (StringUtil::Equals(value, "SCAN")) {
		return ConflictManagerMode::SCAN;
	}
	if (StringUtil::Equals(value, "THROW")) {
		return ConflictManagerMode::THROW;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<LookupResultType>(LookupResultType value) {
	switch (value) {
	case LookupResultType::LOOKUP_MISS:
		return "LOOKUP_MISS";
	case LookupResultType::LOOKUP_HIT:
		return "LOOKUP_HIT";
	case LookupResultType::LOOKUP_NULL:
		return "LOOKUP_NULL";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
LookupResultType EnumUtil::FromString<LookupResultType>(const char *value) {
	if (StringUtil::Equals(value, "LOOKUP_MISS")) {
		return LookupResultType::LOOKUP_MISS;
	}
	if (StringUtil::Equals(value, "LOOKUP_HIT")) {
		return LookupResultType::LOOKUP_HIT;
	}
	if (StringUtil::Equals(value, "LOOKUP_NULL")) {
		return LookupResultType::LOOKUP_NULL;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<MapInvalidReason>(MapInvalidReason value) {
	switch (value) {
	case MapInvalidReason::VALID:
		return "VALID";
	case MapInvalidReason::NULL_KEY_LIST:
		return "NULL_KEY_LIST";
	case MapInvalidReason::NULL_KEY:
		return "NULL_KEY";
	case MapInvalidReason::DUPLICATE_KEY:
		return "DUPLICATE_KEY";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
MapInvalidReason EnumUtil::FromString<MapInvalidReason>(const char *value) {
	if (StringUtil::Equals(value, "VALID")) {
		return MapInvalidReason::VALID;
	}
	if (StringUtil::Equals(value, "NULL_KEY_LIST")) {
		return MapInvalidReason::NULL_KEY_LIST;
	}
	if (StringUtil::Equals(value, "NULL_KEY")) {
		return MapInvalidReason::NULL_KEY;
	}
	if (StringUtil::Equals(value, "DUPLICATE_KEY")) {
		return MapInvalidReason::DUPLICATE_KEY;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<UnionInvalidReason>(UnionInvalidReason value) {
	switch (value) {
	case UnionInvalidReason::VALID:
		return "VALID";
	case UnionInvalidReason::TAG_OUT_OF_RANGE:
		return "TAG_OUT_OF_RANGE";
	case UnionInvalidReason::NO_MEMBERS:
		return "NO_MEMBERS";
	case UnionInvalidReason::VALIDITY_OVERLAP:
		return "VALIDITY_OVERLAP";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
UnionInvalidReason EnumUtil::FromString<UnionInvalidReason>(const char *value) {
	if (StringUtil::Equals(value, "VALID")) {
		return UnionInvalidReason::VALID;
	}
	if (StringUtil::Equals(value, "TAG_OUT_OF_RANGE")) {
		return UnionInvalidReason::TAG_OUT_OF_RANGE;
	}
	if (StringUtil::Equals(value, "NO_MEMBERS")) {
		return UnionInvalidReason::NO_MEMBERS;
	}
	if (StringUtil::Equals(value, "VALIDITY_OVERLAP")) {
		return UnionInvalidReason::VALIDITY_OVERLAP;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<VectorBufferType>(VectorBufferType value) {
	switch (value) {
	case VectorBufferType::STANDARD_BUFFER:
		return "STANDARD_BUFFER";
	case VectorBufferType::DICTIONARY_BUFFER:
		return "DICTIONARY_BUFFER";
	case VectorBufferType::VECTOR_CHILD_BUFFER:
		return "VECTOR_CHILD_BUFFER";
	case VectorBufferType::STRING_BUFFER:
		return "STRING_BUFFER";
	case VectorBufferType::FSST_BUFFER:
		return "FSST_BUFFER";
	case VectorBufferType::STRUCT_BUFFER:
		return "STRUCT_BUFFER";
	case VectorBufferType::LIST_BUFFER:
		return "LIST_BUFFER";
	case VectorBufferType::MANAGED_BUFFER:
		return "MANAGED_BUFFER";
	case VectorBufferType::OPAQUE_BUFFER:
		return "OPAQUE_BUFFER";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
VectorBufferType EnumUtil::FromString<VectorBufferType>(const char *value) {
	if (StringUtil::Equals(value, "STANDARD_BUFFER")) {
		return VectorBufferType::STANDARD_BUFFER;
	}
	if (StringUtil::Equals(value, "DICTIONARY_BUFFER")) {
		return VectorBufferType::DICTIONARY_BUFFER;
	}
	if (StringUtil::Equals(value, "VECTOR_CHILD_BUFFER")) {
		return VectorBufferType::VECTOR_CHILD_BUFFER;
	}
	if (StringUtil::Equals(value, "STRING_BUFFER")) {
		return VectorBufferType::STRING_BUFFER;
	}
	if (StringUtil::Equals(value, "FSST_BUFFER")) {
		return VectorBufferType::FSST_BUFFER;
	}
	if (StringUtil::Equals(value, "STRUCT_BUFFER")) {
		return VectorBufferType::STRUCT_BUFFER;
	}
	if (StringUtil::Equals(value, "LIST_BUFFER")) {
		return VectorBufferType::LIST_BUFFER;
	}
	if (StringUtil::Equals(value, "MANAGED_BUFFER")) {
		return VectorBufferType::MANAGED_BUFFER;
	}
	if (StringUtil::Equals(value, "OPAQUE_BUFFER")) {
		return VectorBufferType::OPAQUE_BUFFER;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<VectorAuxiliaryDataType>(VectorAuxiliaryDataType value) {
	switch (value) {
	case VectorAuxiliaryDataType::ARROW_AUXILIARY:
		return "ARROW_AUXILIARY";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
VectorAuxiliaryDataType EnumUtil::FromString<VectorAuxiliaryDataType>(const char *value) {
	if (StringUtil::Equals(value, "ARROW_AUXILIARY")) {
		return VectorAuxiliaryDataType::ARROW_AUXILIARY;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<PartitionedColumnDataType>(PartitionedColumnDataType value) {
	switch (value) {
	case PartitionedColumnDataType::INVALID:
		return "INVALID";
	case PartitionedColumnDataType::RADIX:
		return "RADIX";
	case PartitionedColumnDataType::HIVE:
		return "HIVE";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
PartitionedColumnDataType EnumUtil::FromString<PartitionedColumnDataType>(const char *value) {
	if (StringUtil::Equals(value, "INVALID")) {
		return PartitionedColumnDataType::INVALID;
	}
	if (StringUtil::Equals(value, "RADIX")) {
		return PartitionedColumnDataType::RADIX;
	}
	if (StringUtil::Equals(value, "HIVE")) {
		return PartitionedColumnDataType::HIVE;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<ColumnDataAllocatorType>(ColumnDataAllocatorType value) {
	switch (value) {
	case ColumnDataAllocatorType::BUFFER_MANAGER_ALLOCATOR:
		return "BUFFER_MANAGER_ALLOCATOR";
	case ColumnDataAllocatorType::IN_MEMORY_ALLOCATOR:
		return "IN_MEMORY_ALLOCATOR";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
ColumnDataAllocatorType EnumUtil::FromString<ColumnDataAllocatorType>(const char *value) {
	if (StringUtil::Equals(value, "BUFFER_MANAGER_ALLOCATOR")) {
		return ColumnDataAllocatorType::BUFFER_MANAGER_ALLOCATOR;
	}
	if (StringUtil::Equals(value, "IN_MEMORY_ALLOCATOR")) {
		return ColumnDataAllocatorType::IN_MEMORY_ALLOCATOR;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<ColumnDataScanProperties>(ColumnDataScanProperties value) {
	switch (value) {
	case ColumnDataScanProperties::INVALID:
		return "INVALID";
	case ColumnDataScanProperties::ALLOW_ZERO_COPY:
		return "ALLOW_ZERO_COPY";
	case ColumnDataScanProperties::DISALLOW_ZERO_COPY:
		return "DISALLOW_ZERO_COPY";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
ColumnDataScanProperties EnumUtil::FromString<ColumnDataScanProperties>(const char *value) {
	if (StringUtil::Equals(value, "INVALID")) {
		return ColumnDataScanProperties::INVALID;
	}
	if (StringUtil::Equals(value, "ALLOW_ZERO_COPY")) {
		return ColumnDataScanProperties::ALLOW_ZERO_COPY;
	}
	if (StringUtil::Equals(value, "DISALLOW_ZERO_COPY")) {
		return ColumnDataScanProperties::DISALLOW_ZERO_COPY;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<PartitionedTupleDataType>(PartitionedTupleDataType value) {
	switch (value) {
	case PartitionedTupleDataType::INVALID:
		return "INVALID";
	case PartitionedTupleDataType::RADIX:
		return "RADIX";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
PartitionedTupleDataType EnumUtil::FromString<PartitionedTupleDataType>(const char *value) {
	if (StringUtil::Equals(value, "INVALID")) {
		return PartitionedTupleDataType::INVALID;
	}
	if (StringUtil::Equals(value, "RADIX")) {
		return PartitionedTupleDataType::RADIX;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<TupleDataPinProperties>(TupleDataPinProperties value) {
	switch (value) {
	case TupleDataPinProperties::INVALID:
		return "INVALID";
	case TupleDataPinProperties::KEEP_EVERYTHING_PINNED:
		return "KEEP_EVERYTHING_PINNED";
	case TupleDataPinProperties::UNPIN_AFTER_DONE:
		return "UNPIN_AFTER_DONE";
	case TupleDataPinProperties::DESTROY_AFTER_DONE:
		return "DESTROY_AFTER_DONE";
	case TupleDataPinProperties::ALREADY_PINNED:
		return "ALREADY_PINNED";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
TupleDataPinProperties EnumUtil::FromString<TupleDataPinProperties>(const char *value) {
	if (StringUtil::Equals(value, "INVALID")) {
		return TupleDataPinProperties::INVALID;
	}
	if (StringUtil::Equals(value, "KEEP_EVERYTHING_PINNED")) {
		return TupleDataPinProperties::KEEP_EVERYTHING_PINNED;
	}
	if (StringUtil::Equals(value, "UNPIN_AFTER_DONE")) {
		return TupleDataPinProperties::UNPIN_AFTER_DONE;
	}
	if (StringUtil::Equals(value, "DESTROY_AFTER_DONE")) {
		return TupleDataPinProperties::DESTROY_AFTER_DONE;
	}
	if (StringUtil::Equals(value, "ALREADY_PINNED")) {
		return TupleDataPinProperties::ALREADY_PINNED;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<PartitionSortStage>(PartitionSortStage value) {
	switch (value) {
	case PartitionSortStage::INIT:
		return "INIT";
	case PartitionSortStage::PREPARE:
		return "PREPARE";
	case PartitionSortStage::MERGE:
		return "MERGE";
	case PartitionSortStage::SORTED:
		return "SORTED";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
PartitionSortStage EnumUtil::FromString<PartitionSortStage>(const char *value) {
	if (StringUtil::Equals(value, "INIT")) {
		return PartitionSortStage::INIT;
	}
	if (StringUtil::Equals(value, "PREPARE")) {
		return PartitionSortStage::PREPARE;
	}
	if (StringUtil::Equals(value, "MERGE")) {
		return PartitionSortStage::MERGE;
	}
	if (StringUtil::Equals(value, "SORTED")) {
		return PartitionSortStage::SORTED;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<PhysicalOperatorType>(PhysicalOperatorType value) {
	switch (value) {
	case PhysicalOperatorType::INVALID:
		return "INVALID";
	case PhysicalOperatorType::ORDER_BY:
		return "ORDER_BY";
	case PhysicalOperatorType::LIMIT:
		return "LIMIT";
	case PhysicalOperatorType::STREAMING_LIMIT:
		return "STREAMING_LIMIT";
	case PhysicalOperatorType::LIMIT_PERCENT:
		return "LIMIT_PERCENT";
	case PhysicalOperatorType::TOP_N:
		return "TOP_N";
	case PhysicalOperatorType::WINDOW:
		return "WINDOW";
	case PhysicalOperatorType::UNNEST:
		return "UNNEST";
	case PhysicalOperatorType::UNGROUPED_AGGREGATE:
		return "UNGROUPED_AGGREGATE";
	case PhysicalOperatorType::HASH_GROUP_BY:
		return "HASH_GROUP_BY";
	case PhysicalOperatorType::PERFECT_HASH_GROUP_BY:
		return "PERFECT_HASH_GROUP_BY";
	case PhysicalOperatorType::FILTER:
		return "FILTER";
	case PhysicalOperatorType::PROJECTION:
		return "PROJECTION";
	case PhysicalOperatorType::COPY_TO_FILE:
		return "COPY_TO_FILE";
	case PhysicalOperatorType::RESERVOIR_SAMPLE:
		return "RESERVOIR_SAMPLE";
	case PhysicalOperatorType::STREAMING_SAMPLE:
		return "STREAMING_SAMPLE";
	case PhysicalOperatorType::STREAMING_WINDOW:
		return "STREAMING_WINDOW";
	case PhysicalOperatorType::PIVOT:
		return "PIVOT";
	case PhysicalOperatorType::TABLE_SCAN:
		return "TABLE_SCAN";
	case PhysicalOperatorType::DUMMY_SCAN:
		return "DUMMY_SCAN";
	case PhysicalOperatorType::COLUMN_DATA_SCAN:
		return "COLUMN_DATA_SCAN";
	case PhysicalOperatorType::CHUNK_SCAN:
		return "CHUNK_SCAN";
	case PhysicalOperatorType::RECURSIVE_CTE_SCAN:
		return "RECURSIVE_CTE_SCAN";
	case PhysicalOperatorType::DELIM_SCAN:
		return "DELIM_SCAN";
	case PhysicalOperatorType::EXPRESSION_SCAN:
		return "EXPRESSION_SCAN";
	case PhysicalOperatorType::POSITIONAL_SCAN:
		return "POSITIONAL_SCAN";
	case PhysicalOperatorType::BLOCKWISE_NL_JOIN:
		return "BLOCKWISE_NL_JOIN";
	case PhysicalOperatorType::NESTED_LOOP_JOIN:
		return "NESTED_LOOP_JOIN";
	case PhysicalOperatorType::HASH_JOIN:
		return "HASH_JOIN";
	case PhysicalOperatorType::CROSS_PRODUCT:
		return "CROSS_PRODUCT";
	case PhysicalOperatorType::PIECEWISE_MERGE_JOIN:
		return "PIECEWISE_MERGE_JOIN";
	case PhysicalOperatorType::IE_JOIN:
		return "IE_JOIN";
	case PhysicalOperatorType::DELIM_JOIN:
		return "DELIM_JOIN";
	case PhysicalOperatorType::INDEX_JOIN:
		return "INDEX_JOIN";
	case PhysicalOperatorType::POSITIONAL_JOIN:
		return "POSITIONAL_JOIN";
	case PhysicalOperatorType::ASOF_JOIN:
		return "ASOF_JOIN";
	case PhysicalOperatorType::UNION:
		return "UNION";
	case PhysicalOperatorType::RECURSIVE_CTE:
		return "RECURSIVE_CTE";
	case PhysicalOperatorType::INSERT:
		return "INSERT";
	case PhysicalOperatorType::BATCH_INSERT:
		return "BATCH_INSERT";
	case PhysicalOperatorType::DELETE_OPERATOR:
		return "DELETE_OPERATOR";
	case PhysicalOperatorType::UPDATE:
		return "UPDATE";
	case PhysicalOperatorType::CREATE_TABLE:
		return "CREATE_TABLE";
	case PhysicalOperatorType::CREATE_TABLE_AS:
		return "CREATE_TABLE_AS";
	case PhysicalOperatorType::BATCH_CREATE_TABLE_AS:
		return "BATCH_CREATE_TABLE_AS";
	case PhysicalOperatorType::CREATE_INDEX:
		return "CREATE_INDEX";
	case PhysicalOperatorType::ALTER:
		return "ALTER";
	case PhysicalOperatorType::CREATE_SEQUENCE:
		return "CREATE_SEQUENCE";
	case PhysicalOperatorType::CREATE_VIEW:
		return "CREATE_VIEW";
	case PhysicalOperatorType::CREATE_SCHEMA:
		return "CREATE_SCHEMA";
	case PhysicalOperatorType::CREATE_MACRO:
		return "CREATE_MACRO";
	case PhysicalOperatorType::DROP:
		return "DROP";
	case PhysicalOperatorType::PRAGMA:
		return "PRAGMA";
	case PhysicalOperatorType::TRANSACTION:
		return "TRANSACTION";
	case PhysicalOperatorType::CREATE_TYPE:
		return "CREATE_TYPE";
	case PhysicalOperatorType::ATTACH:
		return "ATTACH";
	case PhysicalOperatorType::DETACH:
		return "DETACH";
	case PhysicalOperatorType::EXPLAIN:
		return "EXPLAIN";
	case PhysicalOperatorType::EXPLAIN_ANALYZE:
		return "EXPLAIN_ANALYZE";
	case PhysicalOperatorType::EMPTY_RESULT:
		return "EMPTY_RESULT";
	case PhysicalOperatorType::EXECUTE:
		return "EXECUTE";
	case PhysicalOperatorType::PREPARE:
		return "PREPARE";
	case PhysicalOperatorType::VACUUM:
		return "VACUUM";
	case PhysicalOperatorType::EXPORT:
		return "EXPORT";
	case PhysicalOperatorType::SET:
		return "SET";
	case PhysicalOperatorType::LOAD:
		return "LOAD";
	case PhysicalOperatorType::INOUT_FUNCTION:
		return "INOUT_FUNCTION";
	case PhysicalOperatorType::RESULT_COLLECTOR:
		return "RESULT_COLLECTOR";
	case PhysicalOperatorType::RESET:
		return "RESET";
	case PhysicalOperatorType::EXTENSION:
		return "EXTENSION";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
PhysicalOperatorType EnumUtil::FromString<PhysicalOperatorType>(const char *value) {
	if (StringUtil::Equals(value, "INVALID")) {
		return PhysicalOperatorType::INVALID;
	}
	if (StringUtil::Equals(value, "ORDER_BY")) {
		return PhysicalOperatorType::ORDER_BY;
	}
	if (StringUtil::Equals(value, "LIMIT")) {
		return PhysicalOperatorType::LIMIT;
	}
	if (StringUtil::Equals(value, "STREAMING_LIMIT")) {
		return PhysicalOperatorType::STREAMING_LIMIT;
	}
	if (StringUtil::Equals(value, "LIMIT_PERCENT")) {
		return PhysicalOperatorType::LIMIT_PERCENT;
	}
	if (StringUtil::Equals(value, "TOP_N")) {
		return PhysicalOperatorType::TOP_N;
	}
	if (StringUtil::Equals(value, "WINDOW")) {
		return PhysicalOperatorType::WINDOW;
	}
	if (StringUtil::Equals(value, "UNNEST")) {
		return PhysicalOperatorType::UNNEST;
	}
	if (StringUtil::Equals(value, "UNGROUPED_AGGREGATE")) {
		return PhysicalOperatorType::UNGROUPED_AGGREGATE;
	}
	if (StringUtil::Equals(value, "HASH_GROUP_BY")) {
		return PhysicalOperatorType::HASH_GROUP_BY;
	}
	if (StringUtil::Equals(value, "PERFECT_HASH_GROUP_BY")) {
		return PhysicalOperatorType::PERFECT_HASH_GROUP_BY;
	}
	if (StringUtil::Equals(value, "FILTER")) {
		return PhysicalOperatorType::FILTER;
	}
	if (StringUtil::Equals(value, "PROJECTION")) {
		return PhysicalOperatorType::PROJECTION;
	}
	if (StringUtil::Equals(value, "COPY_TO_FILE")) {
		return PhysicalOperatorType::COPY_TO_FILE;
	}
	if (StringUtil::Equals(value, "RESERVOIR_SAMPLE")) {
		return PhysicalOperatorType::RESERVOIR_SAMPLE;
	}
	if (StringUtil::Equals(value, "STREAMING_SAMPLE")) {
		return PhysicalOperatorType::STREAMING_SAMPLE;
	}
	if (StringUtil::Equals(value, "STREAMING_WINDOW")) {
		return PhysicalOperatorType::STREAMING_WINDOW;
	}
	if (StringUtil::Equals(value, "PIVOT")) {
		return PhysicalOperatorType::PIVOT;
	}
	if (StringUtil::Equals(value, "TABLE_SCAN")) {
		return PhysicalOperatorType::TABLE_SCAN;
	}
	if (StringUtil::Equals(value, "DUMMY_SCAN")) {
		return PhysicalOperatorType::DUMMY_SCAN;
	}
	if (StringUtil::Equals(value, "COLUMN_DATA_SCAN")) {
		return PhysicalOperatorType::COLUMN_DATA_SCAN;
	}
	if (StringUtil::Equals(value, "CHUNK_SCAN")) {
		return PhysicalOperatorType::CHUNK_SCAN;
	}
	if (StringUtil::Equals(value, "RECURSIVE_CTE_SCAN")) {
		return PhysicalOperatorType::RECURSIVE_CTE_SCAN;
	}
	if (StringUtil::Equals(value, "DELIM_SCAN")) {
		return PhysicalOperatorType::DELIM_SCAN;
	}
	if (StringUtil::Equals(value, "EXPRESSION_SCAN")) {
		return PhysicalOperatorType::EXPRESSION_SCAN;
	}
	if (StringUtil::Equals(value, "POSITIONAL_SCAN")) {
		return PhysicalOperatorType::POSITIONAL_SCAN;
	}
	if (StringUtil::Equals(value, "BLOCKWISE_NL_JOIN")) {
		return PhysicalOperatorType::BLOCKWISE_NL_JOIN;
	}
	if (StringUtil::Equals(value, "NESTED_LOOP_JOIN")) {
		return PhysicalOperatorType::NESTED_LOOP_JOIN;
	}
	if (StringUtil::Equals(value, "HASH_JOIN")) {
		return PhysicalOperatorType::HASH_JOIN;
	}
	if (StringUtil::Equals(value, "CROSS_PRODUCT")) {
		return PhysicalOperatorType::CROSS_PRODUCT;
	}
	if (StringUtil::Equals(value, "PIECEWISE_MERGE_JOIN")) {
		return PhysicalOperatorType::PIECEWISE_MERGE_JOIN;
	}
	if (StringUtil::Equals(value, "IE_JOIN")) {
		return PhysicalOperatorType::IE_JOIN;
	}
	if (StringUtil::Equals(value, "DELIM_JOIN")) {
		return PhysicalOperatorType::DELIM_JOIN;
	}
	if (StringUtil::Equals(value, "INDEX_JOIN")) {
		return PhysicalOperatorType::INDEX_JOIN;
	}
	if (StringUtil::Equals(value, "POSITIONAL_JOIN")) {
		return PhysicalOperatorType::POSITIONAL_JOIN;
	}
	if (StringUtil::Equals(value, "ASOF_JOIN")) {
		return PhysicalOperatorType::ASOF_JOIN;
	}
	if (StringUtil::Equals(value, "UNION")) {
		return PhysicalOperatorType::UNION;
	}
	if (StringUtil::Equals(value, "RECURSIVE_CTE")) {
		return PhysicalOperatorType::RECURSIVE_CTE;
	}
	if (StringUtil::Equals(value, "INSERT")) {
		return PhysicalOperatorType::INSERT;
	}
	if (StringUtil::Equals(value, "BATCH_INSERT")) {
		return PhysicalOperatorType::BATCH_INSERT;
	}
	if (StringUtil::Equals(value, "DELETE_OPERATOR")) {
		return PhysicalOperatorType::DELETE_OPERATOR;
	}
	if (StringUtil::Equals(value, "UPDATE")) {
		return PhysicalOperatorType::UPDATE;
	}
	if (StringUtil::Equals(value, "CREATE_TABLE")) {
		return PhysicalOperatorType::CREATE_TABLE;
	}
	if (StringUtil::Equals(value, "CREATE_TABLE_AS")) {
		return PhysicalOperatorType::CREATE_TABLE_AS;
	}
	if (StringUtil::Equals(value, "BATCH_CREATE_TABLE_AS")) {
		return PhysicalOperatorType::BATCH_CREATE_TABLE_AS;
	}
	if (StringUtil::Equals(value, "CREATE_INDEX")) {
		return PhysicalOperatorType::CREATE_INDEX;
	}
	if (StringUtil::Equals(value, "ALTER")) {
		return PhysicalOperatorType::ALTER;
	}
	if (StringUtil::Equals(value, "CREATE_SEQUENCE")) {
		return PhysicalOperatorType::CREATE_SEQUENCE;
	}
	if (StringUtil::Equals(value, "CREATE_VIEW")) {
		return PhysicalOperatorType::CREATE_VIEW;
	}
	if (StringUtil::Equals(value, "CREATE_SCHEMA")) {
		return PhysicalOperatorType::CREATE_SCHEMA;
	}
	if (StringUtil::Equals(value, "CREATE_MACRO")) {
		return PhysicalOperatorType::CREATE_MACRO;
	}
	if (StringUtil::Equals(value, "DROP")) {
		return PhysicalOperatorType::DROP;
	}
	if (StringUtil::Equals(value, "PRAGMA")) {
		return PhysicalOperatorType::PRAGMA;
	}
	if (StringUtil::Equals(value, "TRANSACTION")) {
		return PhysicalOperatorType::TRANSACTION;
	}
	if (StringUtil::Equals(value, "CREATE_TYPE")) {
		return PhysicalOperatorType::CREATE_TYPE;
	}
	if (StringUtil::Equals(value, "ATTACH")) {
		return PhysicalOperatorType::ATTACH;
	}
	if (StringUtil::Equals(value, "DETACH")) {
		return PhysicalOperatorType::DETACH;
	}
	if (StringUtil::Equals(value, "EXPLAIN")) {
		return PhysicalOperatorType::EXPLAIN;
	}
	if (StringUtil::Equals(value, "EXPLAIN_ANALYZE")) {
		return PhysicalOperatorType::EXPLAIN_ANALYZE;
	}
	if (StringUtil::Equals(value, "EMPTY_RESULT")) {
		return PhysicalOperatorType::EMPTY_RESULT;
	}
	if (StringUtil::Equals(value, "EXECUTE")) {
		return PhysicalOperatorType::EXECUTE;
	}
	if (StringUtil::Equals(value, "PREPARE")) {
		return PhysicalOperatorType::PREPARE;
	}
	if (StringUtil::Equals(value, "VACUUM")) {
		return PhysicalOperatorType::VACUUM;
	}
	if (StringUtil::Equals(value, "EXPORT")) {
		return PhysicalOperatorType::EXPORT;
	}
	if (StringUtil::Equals(value, "SET")) {
		return PhysicalOperatorType::SET;
	}
	if (StringUtil::Equals(value, "LOAD")) {
		return PhysicalOperatorType::LOAD;
	}
	if (StringUtil::Equals(value, "INOUT_FUNCTION")) {
		return PhysicalOperatorType::INOUT_FUNCTION;
	}
	if (StringUtil::Equals(value, "RESULT_COLLECTOR")) {
		return PhysicalOperatorType::RESULT_COLLECTOR;
	}
	if (StringUtil::Equals(value, "RESET")) {
		return PhysicalOperatorType::RESET;
	}
	if (StringUtil::Equals(value, "EXTENSION")) {
		return PhysicalOperatorType::EXTENSION;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<VectorType>(VectorType value) {
	switch (value) {
	case VectorType::FLAT_VECTOR:
		return "FLAT_VECTOR";
	case VectorType::FSST_VECTOR:
		return "FSST_VECTOR";
	case VectorType::CONSTANT_VECTOR:
		return "CONSTANT_VECTOR";
	case VectorType::DICTIONARY_VECTOR:
		return "DICTIONARY_VECTOR";
	case VectorType::SEQUENCE_VECTOR:
		return "SEQUENCE_VECTOR";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
VectorType EnumUtil::FromString<VectorType>(const char *value) {
	if (StringUtil::Equals(value, "FLAT_VECTOR")) {
		return VectorType::FLAT_VECTOR;
	}
	if (StringUtil::Equals(value, "FSST_VECTOR")) {
		return VectorType::FSST_VECTOR;
	}
	if (StringUtil::Equals(value, "CONSTANT_VECTOR")) {
		return VectorType::CONSTANT_VECTOR;
	}
	if (StringUtil::Equals(value, "DICTIONARY_VECTOR")) {
		return VectorType::DICTIONARY_VECTOR;
	}
	if (StringUtil::Equals(value, "SEQUENCE_VECTOR")) {
		return VectorType::SEQUENCE_VECTOR;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<AccessMode>(AccessMode value) {
	switch (value) {
	case AccessMode::UNDEFINED:
		return "UNDEFINED";
	case AccessMode::AUTOMATIC:
		return "AUTOMATIC";
	case AccessMode::READ_ONLY:
		return "READ_ONLY";
	case AccessMode::READ_WRITE:
		return "READ_WRITE";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
AccessMode EnumUtil::FromString<AccessMode>(const char *value) {
	if (StringUtil::Equals(value, "UNDEFINED")) {
		return AccessMode::UNDEFINED;
	}
	if (StringUtil::Equals(value, "AUTOMATIC")) {
		return AccessMode::AUTOMATIC;
	}
	if (StringUtil::Equals(value, "READ_ONLY")) {
		return AccessMode::READ_ONLY;
	}
	if (StringUtil::Equals(value, "READ_WRITE")) {
		return AccessMode::READ_WRITE;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<FileGlobOptions>(FileGlobOptions value) {
	switch (value) {
	case FileGlobOptions::DISALLOW_EMPTY:
		return "DISALLOW_EMPTY";
	case FileGlobOptions::ALLOW_EMPTY:
		return "ALLOW_EMPTY";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
FileGlobOptions EnumUtil::FromString<FileGlobOptions>(const char *value) {
	if (StringUtil::Equals(value, "DISALLOW_EMPTY")) {
		return FileGlobOptions::DISALLOW_EMPTY;
	}
	if (StringUtil::Equals(value, "ALLOW_EMPTY")) {
		return FileGlobOptions::ALLOW_EMPTY;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<WALType>(WALType value) {
	switch (value) {
	case WALType::INVALID:
		return "INVALID";
	case WALType::CREATE_TABLE:
		return "CREATE_TABLE";
	case WALType::DROP_TABLE:
		return "DROP_TABLE";
	case WALType::CREATE_SCHEMA:
		return "CREATE_SCHEMA";
	case WALType::DROP_SCHEMA:
		return "DROP_SCHEMA";
	case WALType::CREATE_VIEW:
		return "CREATE_VIEW";
	case WALType::DROP_VIEW:
		return "DROP_VIEW";
	case WALType::CREATE_SEQUENCE:
		return "CREATE_SEQUENCE";
	case WALType::DROP_SEQUENCE:
		return "DROP_SEQUENCE";
	case WALType::SEQUENCE_VALUE:
		return "SEQUENCE_VALUE";
	case WALType::CREATE_MACRO:
		return "CREATE_MACRO";
	case WALType::DROP_MACRO:
		return "DROP_MACRO";
	case WALType::CREATE_TYPE:
		return "CREATE_TYPE";
	case WALType::DROP_TYPE:
		return "DROP_TYPE";
	case WALType::ALTER_INFO:
		return "ALTER_INFO";
	case WALType::CREATE_TABLE_MACRO:
		return "CREATE_TABLE_MACRO";
	case WALType::DROP_TABLE_MACRO:
		return "DROP_TABLE_MACRO";
	case WALType::CREATE_INDEX:
		return "CREATE_INDEX";
	case WALType::DROP_INDEX:
		return "DROP_INDEX";
	case WALType::USE_TABLE:
		return "USE_TABLE";
	case WALType::INSERT_TUPLE:
		return "INSERT_TUPLE";
	case WALType::DELETE_TUPLE:
		return "DELETE_TUPLE";
	case WALType::UPDATE_TUPLE:
		return "UPDATE_TUPLE";
	case WALType::CHECKPOINT:
		return "CHECKPOINT";
	case WALType::WAL_FLUSH:
		return "WAL_FLUSH";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
WALType EnumUtil::FromString<WALType>(const char *value) {
	if (StringUtil::Equals(value, "INVALID")) {
		return WALType::INVALID;
	}
	if (StringUtil::Equals(value, "CREATE_TABLE")) {
		return WALType::CREATE_TABLE;
	}
	if (StringUtil::Equals(value, "DROP_TABLE")) {
		return WALType::DROP_TABLE;
	}
	if (StringUtil::Equals(value, "CREATE_SCHEMA")) {
		return WALType::CREATE_SCHEMA;
	}
	if (StringUtil::Equals(value, "DROP_SCHEMA")) {
		return WALType::DROP_SCHEMA;
	}
	if (StringUtil::Equals(value, "CREATE_VIEW")) {
		return WALType::CREATE_VIEW;
	}
	if (StringUtil::Equals(value, "DROP_VIEW")) {
		return WALType::DROP_VIEW;
	}
	if (StringUtil::Equals(value, "CREATE_SEQUENCE")) {
		return WALType::CREATE_SEQUENCE;
	}
	if (StringUtil::Equals(value, "DROP_SEQUENCE")) {
		return WALType::DROP_SEQUENCE;
	}
	if (StringUtil::Equals(value, "SEQUENCE_VALUE")) {
		return WALType::SEQUENCE_VALUE;
	}
	if (StringUtil::Equals(value, "CREATE_MACRO")) {
		return WALType::CREATE_MACRO;
	}
	if (StringUtil::Equals(value, "DROP_MACRO")) {
		return WALType::DROP_MACRO;
	}
	if (StringUtil::Equals(value, "CREATE_TYPE")) {
		return WALType::CREATE_TYPE;
	}
	if (StringUtil::Equals(value, "DROP_TYPE")) {
		return WALType::DROP_TYPE;
	}
	if (StringUtil::Equals(value, "ALTER_INFO")) {
		return WALType::ALTER_INFO;
	}
	if (StringUtil::Equals(value, "CREATE_TABLE_MACRO")) {
		return WALType::CREATE_TABLE_MACRO;
	}
	if (StringUtil::Equals(value, "DROP_TABLE_MACRO")) {
		return WALType::DROP_TABLE_MACRO;
	}
	if (StringUtil::Equals(value, "CREATE_INDEX")) {
		return WALType::CREATE_INDEX;
	}
	if (StringUtil::Equals(value, "DROP_INDEX")) {
		return WALType::DROP_INDEX;
	}
	if (StringUtil::Equals(value, "USE_TABLE")) {
		return WALType::USE_TABLE;
	}
	if (StringUtil::Equals(value, "INSERT_TUPLE")) {
		return WALType::INSERT_TUPLE;
	}
	if (StringUtil::Equals(value, "DELETE_TUPLE")) {
		return WALType::DELETE_TUPLE;
	}
	if (StringUtil::Equals(value, "UPDATE_TUPLE")) {
		return WALType::UPDATE_TUPLE;
	}
	if (StringUtil::Equals(value, "CHECKPOINT")) {
		return WALType::CHECKPOINT;
	}
	if (StringUtil::Equals(value, "WAL_FLUSH")) {
		return WALType::WAL_FLUSH;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<JoinType>(JoinType value) {
	switch (value) {
	case JoinType::INVALID:
		return "INVALID";
	case JoinType::LEFT:
		return "LEFT";
	case JoinType::RIGHT:
		return "RIGHT";
	case JoinType::INNER:
		return "INNER";
	case JoinType::OUTER:
		return "FULL";
	case JoinType::SEMI:
		return "SEMI";
	case JoinType::ANTI:
		return "ANTI";
	case JoinType::MARK:
		return "MARK";
	case JoinType::SINGLE:
		return "SINGLE";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
JoinType EnumUtil::FromString<JoinType>(const char *value) {
	if (StringUtil::Equals(value, "INVALID")) {
		return JoinType::INVALID;
	}
	if (StringUtil::Equals(value, "LEFT")) {
		return JoinType::LEFT;
	}
	if (StringUtil::Equals(value, "RIGHT")) {
		return JoinType::RIGHT;
	}
	if (StringUtil::Equals(value, "INNER")) {
		return JoinType::INNER;
	}
	if (StringUtil::Equals(value, "FULL")) {
		return JoinType::OUTER;
	}
	if (StringUtil::Equals(value, "SEMI")) {
		return JoinType::SEMI;
	}
	if (StringUtil::Equals(value, "ANTI")) {
		return JoinType::ANTI;
	}
	if (StringUtil::Equals(value, "MARK")) {
		return JoinType::MARK;
	}
	if (StringUtil::Equals(value, "SINGLE")) {
		return JoinType::SINGLE;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<FileCompressionType>(FileCompressionType value) {
	switch (value) {
	case FileCompressionType::AUTO_DETECT:
		return "AUTO_DETECT";
	case FileCompressionType::UNCOMPRESSED:
		return "UNCOMPRESSED";
	case FileCompressionType::GZIP:
		return "GZIP";
	case FileCompressionType::ZSTD:
		return "ZSTD";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
FileCompressionType EnumUtil::FromString<FileCompressionType>(const char *value) {
	if (StringUtil::Equals(value, "AUTO_DETECT")) {
		return FileCompressionType::AUTO_DETECT;
	}
	if (StringUtil::Equals(value, "UNCOMPRESSED")) {
		return FileCompressionType::UNCOMPRESSED;
	}
	if (StringUtil::Equals(value, "GZIP")) {
		return FileCompressionType::GZIP;
	}
	if (StringUtil::Equals(value, "ZSTD")) {
		return FileCompressionType::ZSTD;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<ProfilerPrintFormat>(ProfilerPrintFormat value) {
	switch (value) {
	case ProfilerPrintFormat::QUERY_TREE:
		return "QUERY_TREE";
	case ProfilerPrintFormat::JSON:
		return "JSON";
	case ProfilerPrintFormat::QUERY_TREE_OPTIMIZER:
		return "QUERY_TREE_OPTIMIZER";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
ProfilerPrintFormat EnumUtil::FromString<ProfilerPrintFormat>(const char *value) {
	if (StringUtil::Equals(value, "QUERY_TREE")) {
		return ProfilerPrintFormat::QUERY_TREE;
	}
	if (StringUtil::Equals(value, "JSON")) {
		return ProfilerPrintFormat::JSON;
	}
	if (StringUtil::Equals(value, "QUERY_TREE_OPTIMIZER")) {
		return ProfilerPrintFormat::QUERY_TREE_OPTIMIZER;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<StatementType>(StatementType value) {
	switch (value) {
	case StatementType::INVALID_STATEMENT:
		return "INVALID_STATEMENT";
	case StatementType::SELECT_STATEMENT:
		return "SELECT_STATEMENT";
	case StatementType::INSERT_STATEMENT:
		return "INSERT_STATEMENT";
	case StatementType::UPDATE_STATEMENT:
		return "UPDATE_STATEMENT";
	case StatementType::CREATE_STATEMENT:
		return "CREATE_STATEMENT";
	case StatementType::DELETE_STATEMENT:
		return "DELETE_STATEMENT";
	case StatementType::PREPARE_STATEMENT:
		return "PREPARE_STATEMENT";
	case StatementType::EXECUTE_STATEMENT:
		return "EXECUTE_STATEMENT";
	case StatementType::ALTER_STATEMENT:
		return "ALTER_STATEMENT";
	case StatementType::TRANSACTION_STATEMENT:
		return "TRANSACTION_STATEMENT";
	case StatementType::COPY_STATEMENT:
		return "COPY_STATEMENT";
	case StatementType::ANALYZE_STATEMENT:
		return "ANALYZE_STATEMENT";
	case StatementType::VARIABLE_SET_STATEMENT:
		return "VARIABLE_SET_STATEMENT";
	case StatementType::CREATE_FUNC_STATEMENT:
		return "CREATE_FUNC_STATEMENT";
	case StatementType::EXPLAIN_STATEMENT:
		return "EXPLAIN_STATEMENT";
	case StatementType::DROP_STATEMENT:
		return "DROP_STATEMENT";
	case StatementType::EXPORT_STATEMENT:
		return "EXPORT_STATEMENT";
	case StatementType::PRAGMA_STATEMENT:
		return "PRAGMA_STATEMENT";
	case StatementType::SHOW_STATEMENT:
		return "SHOW_STATEMENT";
	case StatementType::VACUUM_STATEMENT:
		return "VACUUM_STATEMENT";
	case StatementType::CALL_STATEMENT:
		return "CALL_STATEMENT";
	case StatementType::SET_STATEMENT:
		return "SET_STATEMENT";
	case StatementType::LOAD_STATEMENT:
		return "LOAD_STATEMENT";
	case StatementType::RELATION_STATEMENT:
		return "RELATION_STATEMENT";
	case StatementType::EXTENSION_STATEMENT:
		return "EXTENSION_STATEMENT";
	case StatementType::LOGICAL_PLAN_STATEMENT:
		return "LOGICAL_PLAN_STATEMENT";
	case StatementType::ATTACH_STATEMENT:
		return "ATTACH_STATEMENT";
	case StatementType::DETACH_STATEMENT:
		return "DETACH_STATEMENT";
	case StatementType::MULTI_STATEMENT:
		return "MULTI_STATEMENT";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
StatementType EnumUtil::FromString<StatementType>(const char *value) {
	if (StringUtil::Equals(value, "INVALID_STATEMENT")) {
		return StatementType::INVALID_STATEMENT;
	}
	if (StringUtil::Equals(value, "SELECT_STATEMENT")) {
		return StatementType::SELECT_STATEMENT;
	}
	if (StringUtil::Equals(value, "INSERT_STATEMENT")) {
		return StatementType::INSERT_STATEMENT;
	}
	if (StringUtil::Equals(value, "UPDATE_STATEMENT")) {
		return StatementType::UPDATE_STATEMENT;
	}
	if (StringUtil::Equals(value, "CREATE_STATEMENT")) {
		return StatementType::CREATE_STATEMENT;
	}
	if (StringUtil::Equals(value, "DELETE_STATEMENT")) {
		return StatementType::DELETE_STATEMENT;
	}
	if (StringUtil::Equals(value, "PREPARE_STATEMENT")) {
		return StatementType::PREPARE_STATEMENT;
	}
	if (StringUtil::Equals(value, "EXECUTE_STATEMENT")) {
		return StatementType::EXECUTE_STATEMENT;
	}
	if (StringUtil::Equals(value, "ALTER_STATEMENT")) {
		return StatementType::ALTER_STATEMENT;
	}
	if (StringUtil::Equals(value, "TRANSACTION_STATEMENT")) {
		return StatementType::TRANSACTION_STATEMENT;
	}
	if (StringUtil::Equals(value, "COPY_STATEMENT")) {
		return StatementType::COPY_STATEMENT;
	}
	if (StringUtil::Equals(value, "ANALYZE_STATEMENT")) {
		return StatementType::ANALYZE_STATEMENT;
	}
	if (StringUtil::Equals(value, "VARIABLE_SET_STATEMENT")) {
		return StatementType::VARIABLE_SET_STATEMENT;
	}
	if (StringUtil::Equals(value, "CREATE_FUNC_STATEMENT")) {
		return StatementType::CREATE_FUNC_STATEMENT;
	}
	if (StringUtil::Equals(value, "EXPLAIN_STATEMENT")) {
		return StatementType::EXPLAIN_STATEMENT;
	}
	if (StringUtil::Equals(value, "DROP_STATEMENT")) {
		return StatementType::DROP_STATEMENT;
	}
	if (StringUtil::Equals(value, "EXPORT_STATEMENT")) {
		return StatementType::EXPORT_STATEMENT;
	}
	if (StringUtil::Equals(value, "PRAGMA_STATEMENT")) {
		return StatementType::PRAGMA_STATEMENT;
	}
	if (StringUtil::Equals(value, "SHOW_STATEMENT")) {
		return StatementType::SHOW_STATEMENT;
	}
	if (StringUtil::Equals(value, "VACUUM_STATEMENT")) {
		return StatementType::VACUUM_STATEMENT;
	}
	if (StringUtil::Equals(value, "CALL_STATEMENT")) {
		return StatementType::CALL_STATEMENT;
	}
	if (StringUtil::Equals(value, "SET_STATEMENT")) {
		return StatementType::SET_STATEMENT;
	}
	if (StringUtil::Equals(value, "LOAD_STATEMENT")) {
		return StatementType::LOAD_STATEMENT;
	}
	if (StringUtil::Equals(value, "RELATION_STATEMENT")) {
		return StatementType::RELATION_STATEMENT;
	}
	if (StringUtil::Equals(value, "EXTENSION_STATEMENT")) {
		return StatementType::EXTENSION_STATEMENT;
	}
	if (StringUtil::Equals(value, "LOGICAL_PLAN_STATEMENT")) {
		return StatementType::LOGICAL_PLAN_STATEMENT;
	}
	if (StringUtil::Equals(value, "ATTACH_STATEMENT")) {
		return StatementType::ATTACH_STATEMENT;
	}
	if (StringUtil::Equals(value, "DETACH_STATEMENT")) {
		return StatementType::DETACH_STATEMENT;
	}
	if (StringUtil::Equals(value, "MULTI_STATEMENT")) {
		return StatementType::MULTI_STATEMENT;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<StatementReturnType>(StatementReturnType value) {
	switch (value) {
	case StatementReturnType::QUERY_RESULT:
		return "QUERY_RESULT";
	case StatementReturnType::CHANGED_ROWS:
		return "CHANGED_ROWS";
	case StatementReturnType::NOTHING:
		return "NOTHING";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
StatementReturnType EnumUtil::FromString<StatementReturnType>(const char *value) {
	if (StringUtil::Equals(value, "QUERY_RESULT")) {
		return StatementReturnType::QUERY_RESULT;
	}
	if (StringUtil::Equals(value, "CHANGED_ROWS")) {
		return StatementReturnType::CHANGED_ROWS;
	}
	if (StringUtil::Equals(value, "NOTHING")) {
		return StatementReturnType::NOTHING;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<OrderPreservationType>(OrderPreservationType value) {
	switch (value) {
	case OrderPreservationType::NO_ORDER:
		return "NO_ORDER";
	case OrderPreservationType::INSERTION_ORDER:
		return "INSERTION_ORDER";
	case OrderPreservationType::FIXED_ORDER:
		return "FIXED_ORDER";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
OrderPreservationType EnumUtil::FromString<OrderPreservationType>(const char *value) {
	if (StringUtil::Equals(value, "NO_ORDER")) {
		return OrderPreservationType::NO_ORDER;
	}
	if (StringUtil::Equals(value, "INSERTION_ORDER")) {
		return OrderPreservationType::INSERTION_ORDER;
	}
	if (StringUtil::Equals(value, "FIXED_ORDER")) {
		return OrderPreservationType::FIXED_ORDER;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<DebugInitialize>(DebugInitialize value) {
	switch (value) {
	case DebugInitialize::NO_INITIALIZE:
		return "NO_INITIALIZE";
	case DebugInitialize::DEBUG_ZERO_INITIALIZE:
		return "DEBUG_ZERO_INITIALIZE";
	case DebugInitialize::DEBUG_ONE_INITIALIZE:
		return "DEBUG_ONE_INITIALIZE";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
DebugInitialize EnumUtil::FromString<DebugInitialize>(const char *value) {
	if (StringUtil::Equals(value, "NO_INITIALIZE")) {
		return DebugInitialize::NO_INITIALIZE;
	}
	if (StringUtil::Equals(value, "DEBUG_ZERO_INITIALIZE")) {
		return DebugInitialize::DEBUG_ZERO_INITIALIZE;
	}
	if (StringUtil::Equals(value, "DEBUG_ONE_INITIALIZE")) {
		return DebugInitialize::DEBUG_ONE_INITIALIZE;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<CatalogType>(CatalogType value) {
	switch (value) {
	case CatalogType::INVALID:
		return "INVALID";
	case CatalogType::TABLE_ENTRY:
		return "TABLE_ENTRY";
	case CatalogType::SCHEMA_ENTRY:
		return "SCHEMA_ENTRY";
	case CatalogType::VIEW_ENTRY:
		return "VIEW_ENTRY";
	case CatalogType::INDEX_ENTRY:
		return "INDEX_ENTRY";
	case CatalogType::PREPARED_STATEMENT:
		return "PREPARED_STATEMENT";
	case CatalogType::SEQUENCE_ENTRY:
		return "SEQUENCE_ENTRY";
	case CatalogType::COLLATION_ENTRY:
		return "COLLATION_ENTRY";
	case CatalogType::TYPE_ENTRY:
		return "TYPE_ENTRY";
	case CatalogType::DATABASE_ENTRY:
		return "DATABASE_ENTRY";
	case CatalogType::TABLE_FUNCTION_ENTRY:
		return "TABLE_FUNCTION_ENTRY";
	case CatalogType::SCALAR_FUNCTION_ENTRY:
		return "SCALAR_FUNCTION_ENTRY";
	case CatalogType::AGGREGATE_FUNCTION_ENTRY:
		return "AGGREGATE_FUNCTION_ENTRY";
	case CatalogType::PRAGMA_FUNCTION_ENTRY:
		return "PRAGMA_FUNCTION_ENTRY";
	case CatalogType::COPY_FUNCTION_ENTRY:
		return "COPY_FUNCTION_ENTRY";
	case CatalogType::MACRO_ENTRY:
		return "MACRO_ENTRY";
	case CatalogType::TABLE_MACRO_ENTRY:
		return "TABLE_MACRO_ENTRY";
	case CatalogType::UPDATED_ENTRY:
		return "UPDATED_ENTRY";
	case CatalogType::DELETED_ENTRY:
		return "DELETED_ENTRY";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
CatalogType EnumUtil::FromString<CatalogType>(const char *value) {
	if (StringUtil::Equals(value, "INVALID")) {
		return CatalogType::INVALID;
	}
	if (StringUtil::Equals(value, "TABLE_ENTRY")) {
		return CatalogType::TABLE_ENTRY;
	}
	if (StringUtil::Equals(value, "SCHEMA_ENTRY")) {
		return CatalogType::SCHEMA_ENTRY;
	}
	if (StringUtil::Equals(value, "VIEW_ENTRY")) {
		return CatalogType::VIEW_ENTRY;
	}
	if (StringUtil::Equals(value, "INDEX_ENTRY")) {
		return CatalogType::INDEX_ENTRY;
	}
	if (StringUtil::Equals(value, "PREPARED_STATEMENT")) {
		return CatalogType::PREPARED_STATEMENT;
	}
	if (StringUtil::Equals(value, "SEQUENCE_ENTRY")) {
		return CatalogType::SEQUENCE_ENTRY;
	}
	if (StringUtil::Equals(value, "COLLATION_ENTRY")) {
		return CatalogType::COLLATION_ENTRY;
	}
	if (StringUtil::Equals(value, "TYPE_ENTRY")) {
		return CatalogType::TYPE_ENTRY;
	}
	if (StringUtil::Equals(value, "DATABASE_ENTRY")) {
		return CatalogType::DATABASE_ENTRY;
	}
	if (StringUtil::Equals(value, "TABLE_FUNCTION_ENTRY")) {
		return CatalogType::TABLE_FUNCTION_ENTRY;
	}
	if (StringUtil::Equals(value, "SCALAR_FUNCTION_ENTRY")) {
		return CatalogType::SCALAR_FUNCTION_ENTRY;
	}
	if (StringUtil::Equals(value, "AGGREGATE_FUNCTION_ENTRY")) {
		return CatalogType::AGGREGATE_FUNCTION_ENTRY;
	}
	if (StringUtil::Equals(value, "PRAGMA_FUNCTION_ENTRY")) {
		return CatalogType::PRAGMA_FUNCTION_ENTRY;
	}
	if (StringUtil::Equals(value, "COPY_FUNCTION_ENTRY")) {
		return CatalogType::COPY_FUNCTION_ENTRY;
	}
	if (StringUtil::Equals(value, "MACRO_ENTRY")) {
		return CatalogType::MACRO_ENTRY;
	}
	if (StringUtil::Equals(value, "TABLE_MACRO_ENTRY")) {
		return CatalogType::TABLE_MACRO_ENTRY;
	}
	if (StringUtil::Equals(value, "UPDATED_ENTRY")) {
		return CatalogType::UPDATED_ENTRY;
	}
	if (StringUtil::Equals(value, "DELETED_ENTRY")) {
		return CatalogType::DELETED_ENTRY;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<SetScope>(SetScope value) {
	switch (value) {
	case SetScope::AUTOMATIC:
		return "AUTOMATIC";
	case SetScope::LOCAL:
		return "LOCAL";
	case SetScope::SESSION:
		return "SESSION";
	case SetScope::GLOBAL:
		return "GLOBAL";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
SetScope EnumUtil::FromString<SetScope>(const char *value) {
	if (StringUtil::Equals(value, "AUTOMATIC")) {
		return SetScope::AUTOMATIC;
	}
	if (StringUtil::Equals(value, "LOCAL")) {
		return SetScope::LOCAL;
	}
	if (StringUtil::Equals(value, "SESSION")) {
		return SetScope::SESSION;
	}
	if (StringUtil::Equals(value, "GLOBAL")) {
		return SetScope::GLOBAL;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<TableScanType>(TableScanType value) {
	switch (value) {
	case TableScanType::TABLE_SCAN_REGULAR:
		return "TABLE_SCAN_REGULAR";
	case TableScanType::TABLE_SCAN_COMMITTED_ROWS:
		return "TABLE_SCAN_COMMITTED_ROWS";
	case TableScanType::TABLE_SCAN_COMMITTED_ROWS_DISALLOW_UPDATES:
		return "TABLE_SCAN_COMMITTED_ROWS_DISALLOW_UPDATES";
	case TableScanType::TABLE_SCAN_COMMITTED_ROWS_OMIT_PERMANENTLY_DELETED:
		return "TABLE_SCAN_COMMITTED_ROWS_OMIT_PERMANENTLY_DELETED";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
TableScanType EnumUtil::FromString<TableScanType>(const char *value) {
	if (StringUtil::Equals(value, "TABLE_SCAN_REGULAR")) {
		return TableScanType::TABLE_SCAN_REGULAR;
	}
	if (StringUtil::Equals(value, "TABLE_SCAN_COMMITTED_ROWS")) {
		return TableScanType::TABLE_SCAN_COMMITTED_ROWS;
	}
	if (StringUtil::Equals(value, "TABLE_SCAN_COMMITTED_ROWS_DISALLOW_UPDATES")) {
		return TableScanType::TABLE_SCAN_COMMITTED_ROWS_DISALLOW_UPDATES;
	}
	if (StringUtil::Equals(value, "TABLE_SCAN_COMMITTED_ROWS_OMIT_PERMANENTLY_DELETED")) {
		return TableScanType::TABLE_SCAN_COMMITTED_ROWS_OMIT_PERMANENTLY_DELETED;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<SetType>(SetType value) {
	switch (value) {
	case SetType::SET:
		return "SET";
	case SetType::RESET:
		return "RESET";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
SetType EnumUtil::FromString<SetType>(const char *value) {
	if (StringUtil::Equals(value, "SET")) {
		return SetType::SET;
	}
	if (StringUtil::Equals(value, "RESET")) {
		return SetType::RESET;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<ExpressionType>(ExpressionType value) {
	switch (value) {
	case ExpressionType::INVALID:
		return "INVALID";
	case ExpressionType::OPERATOR_CAST:
		return "OPERATOR_CAST";
	case ExpressionType::OPERATOR_NOT:
		return "OPERATOR_NOT";
	case ExpressionType::OPERATOR_IS_NULL:
		return "OPERATOR_IS_NULL";
	case ExpressionType::OPERATOR_IS_NOT_NULL:
		return "OPERATOR_IS_NOT_NULL";
	case ExpressionType::COMPARE_EQUAL:
		return "COMPARE_EQUAL";
	case ExpressionType::COMPARE_NOTEQUAL:
		return "COMPARE_NOTEQUAL";
	case ExpressionType::COMPARE_LESSTHAN:
		return "COMPARE_LESSTHAN";
	case ExpressionType::COMPARE_GREATERTHAN:
		return "COMPARE_GREATERTHAN";
	case ExpressionType::COMPARE_LESSTHANOREQUALTO:
		return "COMPARE_LESSTHANOREQUALTO";
	case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
		return "COMPARE_GREATERTHANOREQUALTO";
	case ExpressionType::COMPARE_IN:
		return "COMPARE_IN";
	case ExpressionType::COMPARE_NOT_IN:
		return "COMPARE_NOT_IN";
	case ExpressionType::COMPARE_DISTINCT_FROM:
		return "COMPARE_DISTINCT_FROM";
	case ExpressionType::COMPARE_BETWEEN:
		return "COMPARE_BETWEEN";
	case ExpressionType::COMPARE_NOT_BETWEEN:
		return "COMPARE_NOT_BETWEEN";
	case ExpressionType::COMPARE_NOT_DISTINCT_FROM:
		return "COMPARE_NOT_DISTINCT_FROM";
	case ExpressionType::CONJUNCTION_AND:
		return "CONJUNCTION_AND";
	case ExpressionType::CONJUNCTION_OR:
		return "CONJUNCTION_OR";
	case ExpressionType::VALUE_CONSTANT:
		return "VALUE_CONSTANT";
	case ExpressionType::VALUE_PARAMETER:
		return "VALUE_PARAMETER";
	case ExpressionType::VALUE_TUPLE:
		return "VALUE_TUPLE";
	case ExpressionType::VALUE_TUPLE_ADDRESS:
		return "VALUE_TUPLE_ADDRESS";
	case ExpressionType::VALUE_NULL:
		return "VALUE_NULL";
	case ExpressionType::VALUE_VECTOR:
		return "VALUE_VECTOR";
	case ExpressionType::VALUE_SCALAR:
		return "VALUE_SCALAR";
	case ExpressionType::VALUE_DEFAULT:
		return "VALUE_DEFAULT";
	case ExpressionType::AGGREGATE:
		return "AGGREGATE";
	case ExpressionType::BOUND_AGGREGATE:
		return "BOUND_AGGREGATE";
	case ExpressionType::GROUPING_FUNCTION:
		return "GROUPING_FUNCTION";
	case ExpressionType::WINDOW_AGGREGATE:
		return "WINDOW_AGGREGATE";
	case ExpressionType::WINDOW_RANK:
		return "WINDOW_RANK";
	case ExpressionType::WINDOW_RANK_DENSE:
		return "WINDOW_RANK_DENSE";
	case ExpressionType::WINDOW_NTILE:
		return "WINDOW_NTILE";
	case ExpressionType::WINDOW_PERCENT_RANK:
		return "WINDOW_PERCENT_RANK";
	case ExpressionType::WINDOW_CUME_DIST:
		return "WINDOW_CUME_DIST";
	case ExpressionType::WINDOW_ROW_NUMBER:
		return "WINDOW_ROW_NUMBER";
	case ExpressionType::WINDOW_FIRST_VALUE:
		return "WINDOW_FIRST_VALUE";
	case ExpressionType::WINDOW_LAST_VALUE:
		return "WINDOW_LAST_VALUE";
	case ExpressionType::WINDOW_LEAD:
		return "WINDOW_LEAD";
	case ExpressionType::WINDOW_LAG:
		return "WINDOW_LAG";
	case ExpressionType::WINDOW_NTH_VALUE:
		return "WINDOW_NTH_VALUE";
	case ExpressionType::FUNCTION:
		return "FUNCTION";
	case ExpressionType::BOUND_FUNCTION:
		return "BOUND_FUNCTION";
	case ExpressionType::CASE_EXPR:
		return "CASE_EXPR";
	case ExpressionType::OPERATOR_NULLIF:
		return "OPERATOR_NULLIF";
	case ExpressionType::OPERATOR_COALESCE:
		return "OPERATOR_COALESCE";
	case ExpressionType::ARRAY_EXTRACT:
		return "ARRAY_EXTRACT";
	case ExpressionType::ARRAY_SLICE:
		return "ARRAY_SLICE";
	case ExpressionType::STRUCT_EXTRACT:
		return "STRUCT_EXTRACT";
	case ExpressionType::ARRAY_CONSTRUCTOR:
		return "ARRAY_CONSTRUCTOR";
	case ExpressionType::ARROW:
		return "ARROW";
	case ExpressionType::SUBQUERY:
		return "SUBQUERY";
	case ExpressionType::STAR:
		return "STAR";
	case ExpressionType::TABLE_STAR:
		return "TABLE_STAR";
	case ExpressionType::PLACEHOLDER:
		return "PLACEHOLDER";
	case ExpressionType::COLUMN_REF:
		return "COLUMN_REF";
	case ExpressionType::FUNCTION_REF:
		return "FUNCTION_REF";
	case ExpressionType::TABLE_REF:
		return "TABLE_REF";
	case ExpressionType::CAST:
		return "CAST";
	case ExpressionType::BOUND_REF:
		return "BOUND_REF";
	case ExpressionType::BOUND_COLUMN_REF:
		return "BOUND_COLUMN_REF";
	case ExpressionType::BOUND_UNNEST:
		return "BOUND_UNNEST";
	case ExpressionType::COLLATE:
		return "COLLATE";
	case ExpressionType::LAMBDA:
		return "LAMBDA";
	case ExpressionType::POSITIONAL_REFERENCE:
		return "POSITIONAL_REFERENCE";
	case ExpressionType::BOUND_LAMBDA_REF:
		return "BOUND_LAMBDA_REF";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
ExpressionType EnumUtil::FromString<ExpressionType>(const char *value) {
	if (StringUtil::Equals(value, "INVALID")) {
		return ExpressionType::INVALID;
	}
	if (StringUtil::Equals(value, "OPERATOR_CAST")) {
		return ExpressionType::OPERATOR_CAST;
	}
	if (StringUtil::Equals(value, "OPERATOR_NOT")) {
		return ExpressionType::OPERATOR_NOT;
	}
	if (StringUtil::Equals(value, "OPERATOR_IS_NULL")) {
		return ExpressionType::OPERATOR_IS_NULL;
	}
	if (StringUtil::Equals(value, "OPERATOR_IS_NOT_NULL")) {
		return ExpressionType::OPERATOR_IS_NOT_NULL;
	}
	if (StringUtil::Equals(value, "COMPARE_EQUAL")) {
		return ExpressionType::COMPARE_EQUAL;
	}
	if (StringUtil::Equals(value, "COMPARE_NOTEQUAL")) {
		return ExpressionType::COMPARE_NOTEQUAL;
	}
	if (StringUtil::Equals(value, "COMPARE_LESSTHAN")) {
		return ExpressionType::COMPARE_LESSTHAN;
	}
	if (StringUtil::Equals(value, "COMPARE_GREATERTHAN")) {
		return ExpressionType::COMPARE_GREATERTHAN;
	}
	if (StringUtil::Equals(value, "COMPARE_LESSTHANOREQUALTO")) {
		return ExpressionType::COMPARE_LESSTHANOREQUALTO;
	}
	if (StringUtil::Equals(value, "COMPARE_GREATERTHANOREQUALTO")) {
		return ExpressionType::COMPARE_GREATERTHANOREQUALTO;
	}
	if (StringUtil::Equals(value, "COMPARE_IN")) {
		return ExpressionType::COMPARE_IN;
	}
	if (StringUtil::Equals(value, "COMPARE_NOT_IN")) {
		return ExpressionType::COMPARE_NOT_IN;
	}
	if (StringUtil::Equals(value, "COMPARE_DISTINCT_FROM")) {
		return ExpressionType::COMPARE_DISTINCT_FROM;
	}
	if (StringUtil::Equals(value, "COMPARE_BETWEEN")) {
		return ExpressionType::COMPARE_BETWEEN;
	}
	if (StringUtil::Equals(value, "COMPARE_NOT_BETWEEN")) {
		return ExpressionType::COMPARE_NOT_BETWEEN;
	}
	if (StringUtil::Equals(value, "COMPARE_NOT_DISTINCT_FROM")) {
		return ExpressionType::COMPARE_NOT_DISTINCT_FROM;
	}
	if (StringUtil::Equals(value, "CONJUNCTION_AND")) {
		return ExpressionType::CONJUNCTION_AND;
	}
	if (StringUtil::Equals(value, "CONJUNCTION_OR")) {
		return ExpressionType::CONJUNCTION_OR;
	}
	if (StringUtil::Equals(value, "VALUE_CONSTANT")) {
		return ExpressionType::VALUE_CONSTANT;
	}
	if (StringUtil::Equals(value, "VALUE_PARAMETER")) {
		return ExpressionType::VALUE_PARAMETER;
	}
	if (StringUtil::Equals(value, "VALUE_TUPLE")) {
		return ExpressionType::VALUE_TUPLE;
	}
	if (StringUtil::Equals(value, "VALUE_TUPLE_ADDRESS")) {
		return ExpressionType::VALUE_TUPLE_ADDRESS;
	}
	if (StringUtil::Equals(value, "VALUE_NULL")) {
		return ExpressionType::VALUE_NULL;
	}
	if (StringUtil::Equals(value, "VALUE_VECTOR")) {
		return ExpressionType::VALUE_VECTOR;
	}
	if (StringUtil::Equals(value, "VALUE_SCALAR")) {
		return ExpressionType::VALUE_SCALAR;
	}
	if (StringUtil::Equals(value, "VALUE_DEFAULT")) {
		return ExpressionType::VALUE_DEFAULT;
	}
	if (StringUtil::Equals(value, "AGGREGATE")) {
		return ExpressionType::AGGREGATE;
	}
	if (StringUtil::Equals(value, "BOUND_AGGREGATE")) {
		return ExpressionType::BOUND_AGGREGATE;
	}
	if (StringUtil::Equals(value, "GROUPING_FUNCTION")) {
		return ExpressionType::GROUPING_FUNCTION;
	}
	if (StringUtil::Equals(value, "WINDOW_AGGREGATE")) {
		return ExpressionType::WINDOW_AGGREGATE;
	}
	if (StringUtil::Equals(value, "WINDOW_RANK")) {
		return ExpressionType::WINDOW_RANK;
	}
	if (StringUtil::Equals(value, "WINDOW_RANK_DENSE")) {
		return ExpressionType::WINDOW_RANK_DENSE;
	}
	if (StringUtil::Equals(value, "WINDOW_NTILE")) {
		return ExpressionType::WINDOW_NTILE;
	}
	if (StringUtil::Equals(value, "WINDOW_PERCENT_RANK")) {
		return ExpressionType::WINDOW_PERCENT_RANK;
	}
	if (StringUtil::Equals(value, "WINDOW_CUME_DIST")) {
		return ExpressionType::WINDOW_CUME_DIST;
	}
	if (StringUtil::Equals(value, "WINDOW_ROW_NUMBER")) {
		return ExpressionType::WINDOW_ROW_NUMBER;
	}
	if (StringUtil::Equals(value, "WINDOW_FIRST_VALUE")) {
		return ExpressionType::WINDOW_FIRST_VALUE;
	}
	if (StringUtil::Equals(value, "WINDOW_LAST_VALUE")) {
		return ExpressionType::WINDOW_LAST_VALUE;
	}
	if (StringUtil::Equals(value, "WINDOW_LEAD")) {
		return ExpressionType::WINDOW_LEAD;
	}
	if (StringUtil::Equals(value, "WINDOW_LAG")) {
		return ExpressionType::WINDOW_LAG;
	}
	if (StringUtil::Equals(value, "WINDOW_NTH_VALUE")) {
		return ExpressionType::WINDOW_NTH_VALUE;
	}
	if (StringUtil::Equals(value, "FUNCTION")) {
		return ExpressionType::FUNCTION;
	}
	if (StringUtil::Equals(value, "BOUND_FUNCTION")) {
		return ExpressionType::BOUND_FUNCTION;
	}
	if (StringUtil::Equals(value, "CASE_EXPR")) {
		return ExpressionType::CASE_EXPR;
	}
	if (StringUtil::Equals(value, "OPERATOR_NULLIF")) {
		return ExpressionType::OPERATOR_NULLIF;
	}
	if (StringUtil::Equals(value, "OPERATOR_COALESCE")) {
		return ExpressionType::OPERATOR_COALESCE;
	}
	if (StringUtil::Equals(value, "ARRAY_EXTRACT")) {
		return ExpressionType::ARRAY_EXTRACT;
	}
	if (StringUtil::Equals(value, "ARRAY_SLICE")) {
		return ExpressionType::ARRAY_SLICE;
	}
	if (StringUtil::Equals(value, "STRUCT_EXTRACT")) {
		return ExpressionType::STRUCT_EXTRACT;
	}
	if (StringUtil::Equals(value, "ARRAY_CONSTRUCTOR")) {
		return ExpressionType::ARRAY_CONSTRUCTOR;
	}
	if (StringUtil::Equals(value, "ARROW")) {
		return ExpressionType::ARROW;
	}
	if (StringUtil::Equals(value, "SUBQUERY")) {
		return ExpressionType::SUBQUERY;
	}
	if (StringUtil::Equals(value, "STAR")) {
		return ExpressionType::STAR;
	}
	if (StringUtil::Equals(value, "TABLE_STAR")) {
		return ExpressionType::TABLE_STAR;
	}
	if (StringUtil::Equals(value, "PLACEHOLDER")) {
		return ExpressionType::PLACEHOLDER;
	}
	if (StringUtil::Equals(value, "COLUMN_REF")) {
		return ExpressionType::COLUMN_REF;
	}
	if (StringUtil::Equals(value, "FUNCTION_REF")) {
		return ExpressionType::FUNCTION_REF;
	}
	if (StringUtil::Equals(value, "TABLE_REF")) {
		return ExpressionType::TABLE_REF;
	}
	if (StringUtil::Equals(value, "CAST")) {
		return ExpressionType::CAST;
	}
	if (StringUtil::Equals(value, "BOUND_REF")) {
		return ExpressionType::BOUND_REF;
	}
	if (StringUtil::Equals(value, "BOUND_COLUMN_REF")) {
		return ExpressionType::BOUND_COLUMN_REF;
	}
	if (StringUtil::Equals(value, "BOUND_UNNEST")) {
		return ExpressionType::BOUND_UNNEST;
	}
	if (StringUtil::Equals(value, "COLLATE")) {
		return ExpressionType::COLLATE;
	}
	if (StringUtil::Equals(value, "LAMBDA")) {
		return ExpressionType::LAMBDA;
	}
	if (StringUtil::Equals(value, "POSITIONAL_REFERENCE")) {
		return ExpressionType::POSITIONAL_REFERENCE;
	}
	if (StringUtil::Equals(value, "BOUND_LAMBDA_REF")) {
		return ExpressionType::BOUND_LAMBDA_REF;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<ExpressionClass>(ExpressionClass value) {
	switch (value) {
	case ExpressionClass::INVALID:
		return "INVALID";
	case ExpressionClass::AGGREGATE:
		return "AGGREGATE";
	case ExpressionClass::CASE:
		return "CASE";
	case ExpressionClass::CAST:
		return "CAST";
	case ExpressionClass::COLUMN_REF:
		return "COLUMN_REF";
	case ExpressionClass::COMPARISON:
		return "COMPARISON";
	case ExpressionClass::CONJUNCTION:
		return "CONJUNCTION";
	case ExpressionClass::CONSTANT:
		return "CONSTANT";
	case ExpressionClass::DEFAULT:
		return "DEFAULT";
	case ExpressionClass::FUNCTION:
		return "FUNCTION";
	case ExpressionClass::OPERATOR:
		return "OPERATOR";
	case ExpressionClass::STAR:
		return "STAR";
	case ExpressionClass::SUBQUERY:
		return "SUBQUERY";
	case ExpressionClass::WINDOW:
		return "WINDOW";
	case ExpressionClass::PARAMETER:
		return "PARAMETER";
	case ExpressionClass::COLLATE:
		return "COLLATE";
	case ExpressionClass::LAMBDA:
		return "LAMBDA";
	case ExpressionClass::POSITIONAL_REFERENCE:
		return "POSITIONAL_REFERENCE";
	case ExpressionClass::BETWEEN:
		return "BETWEEN";
	case ExpressionClass::BOUND_AGGREGATE:
		return "BOUND_AGGREGATE";
	case ExpressionClass::BOUND_CASE:
		return "BOUND_CASE";
	case ExpressionClass::BOUND_CAST:
		return "BOUND_CAST";
	case ExpressionClass::BOUND_COLUMN_REF:
		return "BOUND_COLUMN_REF";
	case ExpressionClass::BOUND_COMPARISON:
		return "BOUND_COMPARISON";
	case ExpressionClass::BOUND_CONJUNCTION:
		return "BOUND_CONJUNCTION";
	case ExpressionClass::BOUND_CONSTANT:
		return "BOUND_CONSTANT";
	case ExpressionClass::BOUND_DEFAULT:
		return "BOUND_DEFAULT";
	case ExpressionClass::BOUND_FUNCTION:
		return "BOUND_FUNCTION";
	case ExpressionClass::BOUND_OPERATOR:
		return "BOUND_OPERATOR";
	case ExpressionClass::BOUND_PARAMETER:
		return "BOUND_PARAMETER";
	case ExpressionClass::BOUND_REF:
		return "BOUND_REF";
	case ExpressionClass::BOUND_SUBQUERY:
		return "BOUND_SUBQUERY";
	case ExpressionClass::BOUND_WINDOW:
		return "BOUND_WINDOW";
	case ExpressionClass::BOUND_BETWEEN:
		return "BOUND_BETWEEN";
	case ExpressionClass::BOUND_UNNEST:
		return "BOUND_UNNEST";
	case ExpressionClass::BOUND_LAMBDA:
		return "BOUND_LAMBDA";
	case ExpressionClass::BOUND_LAMBDA_REF:
		return "BOUND_LAMBDA_REF";
	case ExpressionClass::BOUND_EXPRESSION:
		return "BOUND_EXPRESSION";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
ExpressionClass EnumUtil::FromString<ExpressionClass>(const char *value) {
	if (StringUtil::Equals(value, "INVALID")) {
		return ExpressionClass::INVALID;
	}
	if (StringUtil::Equals(value, "AGGREGATE")) {
		return ExpressionClass::AGGREGATE;
	}
	if (StringUtil::Equals(value, "CASE")) {
		return ExpressionClass::CASE;
	}
	if (StringUtil::Equals(value, "CAST")) {
		return ExpressionClass::CAST;
	}
	if (StringUtil::Equals(value, "COLUMN_REF")) {
		return ExpressionClass::COLUMN_REF;
	}
	if (StringUtil::Equals(value, "COMPARISON")) {
		return ExpressionClass::COMPARISON;
	}
	if (StringUtil::Equals(value, "CONJUNCTION")) {
		return ExpressionClass::CONJUNCTION;
	}
	if (StringUtil::Equals(value, "CONSTANT")) {
		return ExpressionClass::CONSTANT;
	}
	if (StringUtil::Equals(value, "DEFAULT")) {
		return ExpressionClass::DEFAULT;
	}
	if (StringUtil::Equals(value, "FUNCTION")) {
		return ExpressionClass::FUNCTION;
	}
	if (StringUtil::Equals(value, "OPERATOR")) {
		return ExpressionClass::OPERATOR;
	}
	if (StringUtil::Equals(value, "STAR")) {
		return ExpressionClass::STAR;
	}
	if (StringUtil::Equals(value, "SUBQUERY")) {
		return ExpressionClass::SUBQUERY;
	}
	if (StringUtil::Equals(value, "WINDOW")) {
		return ExpressionClass::WINDOW;
	}
	if (StringUtil::Equals(value, "PARAMETER")) {
		return ExpressionClass::PARAMETER;
	}
	if (StringUtil::Equals(value, "COLLATE")) {
		return ExpressionClass::COLLATE;
	}
	if (StringUtil::Equals(value, "LAMBDA")) {
		return ExpressionClass::LAMBDA;
	}
	if (StringUtil::Equals(value, "POSITIONAL_REFERENCE")) {
		return ExpressionClass::POSITIONAL_REFERENCE;
	}
	if (StringUtil::Equals(value, "BETWEEN")) {
		return ExpressionClass::BETWEEN;
	}
	if (StringUtil::Equals(value, "BOUND_AGGREGATE")) {
		return ExpressionClass::BOUND_AGGREGATE;
	}
	if (StringUtil::Equals(value, "BOUND_CASE")) {
		return ExpressionClass::BOUND_CASE;
	}
	if (StringUtil::Equals(value, "BOUND_CAST")) {
		return ExpressionClass::BOUND_CAST;
	}
	if (StringUtil::Equals(value, "BOUND_COLUMN_REF")) {
		return ExpressionClass::BOUND_COLUMN_REF;
	}
	if (StringUtil::Equals(value, "BOUND_COMPARISON")) {
		return ExpressionClass::BOUND_COMPARISON;
	}
	if (StringUtil::Equals(value, "BOUND_CONJUNCTION")) {
		return ExpressionClass::BOUND_CONJUNCTION;
	}
	if (StringUtil::Equals(value, "BOUND_CONSTANT")) {
		return ExpressionClass::BOUND_CONSTANT;
	}
	if (StringUtil::Equals(value, "BOUND_DEFAULT")) {
		return ExpressionClass::BOUND_DEFAULT;
	}
	if (StringUtil::Equals(value, "BOUND_FUNCTION")) {
		return ExpressionClass::BOUND_FUNCTION;
	}
	if (StringUtil::Equals(value, "BOUND_OPERATOR")) {
		return ExpressionClass::BOUND_OPERATOR;
	}
	if (StringUtil::Equals(value, "BOUND_PARAMETER")) {
		return ExpressionClass::BOUND_PARAMETER;
	}
	if (StringUtil::Equals(value, "BOUND_REF")) {
		return ExpressionClass::BOUND_REF;
	}
	if (StringUtil::Equals(value, "BOUND_SUBQUERY")) {
		return ExpressionClass::BOUND_SUBQUERY;
	}
	if (StringUtil::Equals(value, "BOUND_WINDOW")) {
		return ExpressionClass::BOUND_WINDOW;
	}
	if (StringUtil::Equals(value, "BOUND_BETWEEN")) {
		return ExpressionClass::BOUND_BETWEEN;
	}
	if (StringUtil::Equals(value, "BOUND_UNNEST")) {
		return ExpressionClass::BOUND_UNNEST;
	}
	if (StringUtil::Equals(value, "BOUND_LAMBDA")) {
		return ExpressionClass::BOUND_LAMBDA;
	}
	if (StringUtil::Equals(value, "BOUND_LAMBDA_REF")) {
		return ExpressionClass::BOUND_LAMBDA_REF;
	}
	if (StringUtil::Equals(value, "BOUND_EXPRESSION")) {
		return ExpressionClass::BOUND_EXPRESSION;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<PendingExecutionResult>(PendingExecutionResult value) {
	switch (value) {
	case PendingExecutionResult::RESULT_READY:
		return "RESULT_READY";
	case PendingExecutionResult::RESULT_NOT_READY:
		return "RESULT_NOT_READY";
	case PendingExecutionResult::EXECUTION_ERROR:
		return "EXECUTION_ERROR";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
PendingExecutionResult EnumUtil::FromString<PendingExecutionResult>(const char *value) {
	if (StringUtil::Equals(value, "RESULT_READY")) {
		return PendingExecutionResult::RESULT_READY;
	}
	if (StringUtil::Equals(value, "RESULT_NOT_READY")) {
		return PendingExecutionResult::RESULT_NOT_READY;
	}
	if (StringUtil::Equals(value, "EXECUTION_ERROR")) {
		return PendingExecutionResult::EXECUTION_ERROR;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<WindowAggregationMode>(WindowAggregationMode value) {
	switch (value) {
	case WindowAggregationMode::WINDOW:
		return "WINDOW";
	case WindowAggregationMode::COMBINE:
		return "COMBINE";
	case WindowAggregationMode::SEPARATE:
		return "SEPARATE";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
WindowAggregationMode EnumUtil::FromString<WindowAggregationMode>(const char *value) {
	if (StringUtil::Equals(value, "WINDOW")) {
		return WindowAggregationMode::WINDOW;
	}
	if (StringUtil::Equals(value, "COMBINE")) {
		return WindowAggregationMode::COMBINE;
	}
	if (StringUtil::Equals(value, "SEPARATE")) {
		return WindowAggregationMode::SEPARATE;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<SubqueryType>(SubqueryType value) {
	switch (value) {
	case SubqueryType::INVALID:
		return "INVALID";
	case SubqueryType::SCALAR:
		return "SCALAR";
	case SubqueryType::EXISTS:
		return "EXISTS";
	case SubqueryType::NOT_EXISTS:
		return "NOT_EXISTS";
	case SubqueryType::ANY:
		return "ANY";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
SubqueryType EnumUtil::FromString<SubqueryType>(const char *value) {
	if (StringUtil::Equals(value, "INVALID")) {
		return SubqueryType::INVALID;
	}
	if (StringUtil::Equals(value, "SCALAR")) {
		return SubqueryType::SCALAR;
	}
	if (StringUtil::Equals(value, "EXISTS")) {
		return SubqueryType::EXISTS;
	}
	if (StringUtil::Equals(value, "NOT_EXISTS")) {
		return SubqueryType::NOT_EXISTS;
	}
	if (StringUtil::Equals(value, "ANY")) {
		return SubqueryType::ANY;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<OrderType>(OrderType value) {
	switch (value) {
	case OrderType::INVALID:
		return "INVALID";
	case OrderType::ORDER_DEFAULT:
		return "ORDER_DEFAULT";
	case OrderType::ASCENDING:
		return "ASCENDING";
	case OrderType::DESCENDING:
		return "DESCENDING";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
OrderType EnumUtil::FromString<OrderType>(const char *value) {
	if (StringUtil::Equals(value, "INVALID")) {
		return OrderType::INVALID;
	}
	if (StringUtil::Equals(value, "ORDER_DEFAULT") || StringUtil::Equals(value, "DEFAULT")) {
		return OrderType::ORDER_DEFAULT;
	}
	if (StringUtil::Equals(value, "ASCENDING") || StringUtil::Equals(value, "ASC")) {
		return OrderType::ASCENDING;
	}
	if (StringUtil::Equals(value, "DESCENDING") || StringUtil::Equals(value, "DESC")) {
		return OrderType::DESCENDING;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<OrderByNullType>(OrderByNullType value) {
	switch (value) {
	case OrderByNullType::INVALID:
		return "INVALID";
	case OrderByNullType::ORDER_DEFAULT:
		return "ORDER_DEFAULT";
	case OrderByNullType::NULLS_FIRST:
		return "NULLS_FIRST";
	case OrderByNullType::NULLS_LAST:
		return "NULLS_LAST";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
OrderByNullType EnumUtil::FromString<OrderByNullType>(const char *value) {
	if (StringUtil::Equals(value, "INVALID")) {
		return OrderByNullType::INVALID;
	}
	if (StringUtil::Equals(value, "ORDER_DEFAULT") || StringUtil::Equals(value, "DEFAULT")) {
		return OrderByNullType::ORDER_DEFAULT;
	}
	if (StringUtil::Equals(value, "NULLS_FIRST") || StringUtil::Equals(value, "NULLS FIRST")) {
		return OrderByNullType::NULLS_FIRST;
	}
	if (StringUtil::Equals(value, "NULLS_LAST") || StringUtil::Equals(value, "NULLS LAST")) {
		return OrderByNullType::NULLS_LAST;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<DefaultOrderByNullType>(DefaultOrderByNullType value) {
	switch (value) {
	case DefaultOrderByNullType::INVALID:
		return "INVALID";
	case DefaultOrderByNullType::NULLS_FIRST:
		return "NULLS_FIRST";
	case DefaultOrderByNullType::NULLS_LAST:
		return "NULLS_LAST";
	case DefaultOrderByNullType::NULLS_FIRST_ON_ASC_LAST_ON_DESC:
		return "NULLS_FIRST_ON_ASC_LAST_ON_DESC";
	case DefaultOrderByNullType::NULLS_LAST_ON_ASC_FIRST_ON_DESC:
		return "NULLS_LAST_ON_ASC_FIRST_ON_DESC";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
DefaultOrderByNullType EnumUtil::FromString<DefaultOrderByNullType>(const char *value) {
	if (StringUtil::Equals(value, "INVALID")) {
		return DefaultOrderByNullType::INVALID;
	}
	if (StringUtil::Equals(value, "NULLS_FIRST")) {
		return DefaultOrderByNullType::NULLS_FIRST;
	}
	if (StringUtil::Equals(value, "NULLS_LAST")) {
		return DefaultOrderByNullType::NULLS_LAST;
	}
	if (StringUtil::Equals(value, "NULLS_FIRST_ON_ASC_LAST_ON_DESC")) {
		return DefaultOrderByNullType::NULLS_FIRST_ON_ASC_LAST_ON_DESC;
	}
	if (StringUtil::Equals(value, "NULLS_LAST_ON_ASC_FIRST_ON_DESC")) {
		return DefaultOrderByNullType::NULLS_LAST_ON_ASC_FIRST_ON_DESC;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<DatePartSpecifier>(DatePartSpecifier value) {
	switch (value) {
	case DatePartSpecifier::YEAR:
		return "YEAR";
	case DatePartSpecifier::MONTH:
		return "MONTH";
	case DatePartSpecifier::DAY:
		return "DAY";
	case DatePartSpecifier::DECADE:
		return "DECADE";
	case DatePartSpecifier::CENTURY:
		return "CENTURY";
	case DatePartSpecifier::MILLENNIUM:
		return "MILLENNIUM";
	case DatePartSpecifier::MICROSECONDS:
		return "MICROSECONDS";
	case DatePartSpecifier::MILLISECONDS:
		return "MILLISECONDS";
	case DatePartSpecifier::SECOND:
		return "SECOND";
	case DatePartSpecifier::MINUTE:
		return "MINUTE";
	case DatePartSpecifier::HOUR:
		return "HOUR";
	case DatePartSpecifier::EPOCH:
		return "EPOCH";
	case DatePartSpecifier::DOW:
		return "DOW";
	case DatePartSpecifier::ISODOW:
		return "ISODOW";
	case DatePartSpecifier::WEEK:
		return "WEEK";
	case DatePartSpecifier::ISOYEAR:
		return "ISOYEAR";
	case DatePartSpecifier::QUARTER:
		return "QUARTER";
	case DatePartSpecifier::DOY:
		return "DOY";
	case DatePartSpecifier::YEARWEEK:
		return "YEARWEEK";
	case DatePartSpecifier::ERA:
		return "ERA";
	case DatePartSpecifier::TIMEZONE:
		return "TIMEZONE";
	case DatePartSpecifier::TIMEZONE_HOUR:
		return "TIMEZONE_HOUR";
	case DatePartSpecifier::TIMEZONE_MINUTE:
		return "TIMEZONE_MINUTE";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
DatePartSpecifier EnumUtil::FromString<DatePartSpecifier>(const char *value) {
	if (StringUtil::Equals(value, "YEAR")) {
		return DatePartSpecifier::YEAR;
	}
	if (StringUtil::Equals(value, "MONTH")) {
		return DatePartSpecifier::MONTH;
	}
	if (StringUtil::Equals(value, "DAY")) {
		return DatePartSpecifier::DAY;
	}
	if (StringUtil::Equals(value, "DECADE")) {
		return DatePartSpecifier::DECADE;
	}
	if (StringUtil::Equals(value, "CENTURY")) {
		return DatePartSpecifier::CENTURY;
	}
	if (StringUtil::Equals(value, "MILLENNIUM")) {
		return DatePartSpecifier::MILLENNIUM;
	}
	if (StringUtil::Equals(value, "MICROSECONDS")) {
		return DatePartSpecifier::MICROSECONDS;
	}
	if (StringUtil::Equals(value, "MILLISECONDS")) {
		return DatePartSpecifier::MILLISECONDS;
	}
	if (StringUtil::Equals(value, "SECOND")) {
		return DatePartSpecifier::SECOND;
	}
	if (StringUtil::Equals(value, "MINUTE")) {
		return DatePartSpecifier::MINUTE;
	}
	if (StringUtil::Equals(value, "HOUR")) {
		return DatePartSpecifier::HOUR;
	}
	if (StringUtil::Equals(value, "EPOCH")) {
		return DatePartSpecifier::EPOCH;
	}
	if (StringUtil::Equals(value, "DOW")) {
		return DatePartSpecifier::DOW;
	}
	if (StringUtil::Equals(value, "ISODOW")) {
		return DatePartSpecifier::ISODOW;
	}
	if (StringUtil::Equals(value, "WEEK")) {
		return DatePartSpecifier::WEEK;
	}
	if (StringUtil::Equals(value, "ISOYEAR")) {
		return DatePartSpecifier::ISOYEAR;
	}
	if (StringUtil::Equals(value, "QUARTER")) {
		return DatePartSpecifier::QUARTER;
	}
	if (StringUtil::Equals(value, "DOY")) {
		return DatePartSpecifier::DOY;
	}
	if (StringUtil::Equals(value, "YEARWEEK")) {
		return DatePartSpecifier::YEARWEEK;
	}
	if (StringUtil::Equals(value, "ERA")) {
		return DatePartSpecifier::ERA;
	}
	if (StringUtil::Equals(value, "TIMEZONE")) {
		return DatePartSpecifier::TIMEZONE;
	}
	if (StringUtil::Equals(value, "TIMEZONE_HOUR")) {
		return DatePartSpecifier::TIMEZONE_HOUR;
	}
	if (StringUtil::Equals(value, "TIMEZONE_MINUTE")) {
		return DatePartSpecifier::TIMEZONE_MINUTE;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<OnEntryNotFound>(OnEntryNotFound value) {
	switch (value) {
	case OnEntryNotFound::THROW_EXCEPTION:
		return "THROW_EXCEPTION";
	case OnEntryNotFound::RETURN_NULL:
		return "RETURN_NULL";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
OnEntryNotFound EnumUtil::FromString<OnEntryNotFound>(const char *value) {
	if (StringUtil::Equals(value, "THROW_EXCEPTION")) {
		return OnEntryNotFound::THROW_EXCEPTION;
	}
	if (StringUtil::Equals(value, "RETURN_NULL")) {
		return OnEntryNotFound::RETURN_NULL;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<LogicalOperatorType>(LogicalOperatorType value) {
	switch (value) {
	case LogicalOperatorType::LOGICAL_INVALID:
		return "LOGICAL_INVALID";
	case LogicalOperatorType::LOGICAL_PROJECTION:
		return "LOGICAL_PROJECTION";
	case LogicalOperatorType::LOGICAL_FILTER:
		return "LOGICAL_FILTER";
	case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
		return "LOGICAL_AGGREGATE_AND_GROUP_BY";
	case LogicalOperatorType::LOGICAL_WINDOW:
		return "LOGICAL_WINDOW";
	case LogicalOperatorType::LOGICAL_UNNEST:
		return "LOGICAL_UNNEST";
	case LogicalOperatorType::LOGICAL_LIMIT:
		return "LOGICAL_LIMIT";
	case LogicalOperatorType::LOGICAL_ORDER_BY:
		return "LOGICAL_ORDER_BY";
	case LogicalOperatorType::LOGICAL_TOP_N:
		return "LOGICAL_TOP_N";
	case LogicalOperatorType::LOGICAL_COPY_TO_FILE:
		return "LOGICAL_COPY_TO_FILE";
	case LogicalOperatorType::LOGICAL_DISTINCT:
		return "LOGICAL_DISTINCT";
	case LogicalOperatorType::LOGICAL_SAMPLE:
		return "LOGICAL_SAMPLE";
	case LogicalOperatorType::LOGICAL_LIMIT_PERCENT:
		return "LOGICAL_LIMIT_PERCENT";
	case LogicalOperatorType::LOGICAL_PIVOT:
		return "LOGICAL_PIVOT";
	case LogicalOperatorType::LOGICAL_GET:
		return "LOGICAL_GET";
	case LogicalOperatorType::LOGICAL_CHUNK_GET:
		return "LOGICAL_CHUNK_GET";
	case LogicalOperatorType::LOGICAL_DELIM_GET:
		return "LOGICAL_DELIM_GET";
	case LogicalOperatorType::LOGICAL_EXPRESSION_GET:
		return "LOGICAL_EXPRESSION_GET";
	case LogicalOperatorType::LOGICAL_DUMMY_SCAN:
		return "LOGICAL_DUMMY_SCAN";
	case LogicalOperatorType::LOGICAL_EMPTY_RESULT:
		return "LOGICAL_EMPTY_RESULT";
	case LogicalOperatorType::LOGICAL_CTE_REF:
		return "LOGICAL_CTE_REF";
	case LogicalOperatorType::LOGICAL_JOIN:
		return "LOGICAL_JOIN";
	case LogicalOperatorType::LOGICAL_DELIM_JOIN:
		return "LOGICAL_DELIM_JOIN";
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
		return "LOGICAL_COMPARISON_JOIN";
	case LogicalOperatorType::LOGICAL_ANY_JOIN:
		return "LOGICAL_ANY_JOIN";
	case LogicalOperatorType::LOGICAL_CROSS_PRODUCT:
		return "LOGICAL_CROSS_PRODUCT";
	case LogicalOperatorType::LOGICAL_POSITIONAL_JOIN:
		return "LOGICAL_POSITIONAL_JOIN";
	case LogicalOperatorType::LOGICAL_ASOF_JOIN:
		return "LOGICAL_ASOF_JOIN";
	case LogicalOperatorType::LOGICAL_UNION:
		return "LOGICAL_UNION";
	case LogicalOperatorType::LOGICAL_EXCEPT:
		return "LOGICAL_EXCEPT";
	case LogicalOperatorType::LOGICAL_INTERSECT:
		return "LOGICAL_INTERSECT";
	case LogicalOperatorType::LOGICAL_RECURSIVE_CTE:
		return "LOGICAL_RECURSIVE_CTE";
	case LogicalOperatorType::LOGICAL_INSERT:
		return "LOGICAL_INSERT";
	case LogicalOperatorType::LOGICAL_DELETE:
		return "LOGICAL_DELETE";
	case LogicalOperatorType::LOGICAL_UPDATE:
		return "LOGICAL_UPDATE";
	case LogicalOperatorType::LOGICAL_ALTER:
		return "LOGICAL_ALTER";
	case LogicalOperatorType::LOGICAL_CREATE_TABLE:
		return "LOGICAL_CREATE_TABLE";
	case LogicalOperatorType::LOGICAL_CREATE_INDEX:
		return "LOGICAL_CREATE_INDEX";
	case LogicalOperatorType::LOGICAL_CREATE_SEQUENCE:
		return "LOGICAL_CREATE_SEQUENCE";
	case LogicalOperatorType::LOGICAL_CREATE_VIEW:
		return "LOGICAL_CREATE_VIEW";
	case LogicalOperatorType::LOGICAL_CREATE_SCHEMA:
		return "LOGICAL_CREATE_SCHEMA";
	case LogicalOperatorType::LOGICAL_CREATE_MACRO:
		return "LOGICAL_CREATE_MACRO";
	case LogicalOperatorType::LOGICAL_DROP:
		return "LOGICAL_DROP";
	case LogicalOperatorType::LOGICAL_PRAGMA:
		return "LOGICAL_PRAGMA";
	case LogicalOperatorType::LOGICAL_TRANSACTION:
		return "LOGICAL_TRANSACTION";
	case LogicalOperatorType::LOGICAL_CREATE_TYPE:
		return "LOGICAL_CREATE_TYPE";
	case LogicalOperatorType::LOGICAL_ATTACH:
		return "LOGICAL_ATTACH";
	case LogicalOperatorType::LOGICAL_DETACH:
		return "LOGICAL_DETACH";
	case LogicalOperatorType::LOGICAL_EXPLAIN:
		return "LOGICAL_EXPLAIN";
	case LogicalOperatorType::LOGICAL_SHOW:
		return "LOGICAL_SHOW";
	case LogicalOperatorType::LOGICAL_PREPARE:
		return "LOGICAL_PREPARE";
	case LogicalOperatorType::LOGICAL_EXECUTE:
		return "LOGICAL_EXECUTE";
	case LogicalOperatorType::LOGICAL_EXPORT:
		return "LOGICAL_EXPORT";
	case LogicalOperatorType::LOGICAL_VACUUM:
		return "LOGICAL_VACUUM";
	case LogicalOperatorType::LOGICAL_SET:
		return "LOGICAL_SET";
	case LogicalOperatorType::LOGICAL_LOAD:
		return "LOGICAL_LOAD";
	case LogicalOperatorType::LOGICAL_RESET:
		return "LOGICAL_RESET";
	case LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR:
		return "LOGICAL_EXTENSION_OPERATOR";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
LogicalOperatorType EnumUtil::FromString<LogicalOperatorType>(const char *value) {
	if (StringUtil::Equals(value, "LOGICAL_INVALID")) {
		return LogicalOperatorType::LOGICAL_INVALID;
	}
	if (StringUtil::Equals(value, "LOGICAL_PROJECTION")) {
		return LogicalOperatorType::LOGICAL_PROJECTION;
	}
	if (StringUtil::Equals(value, "LOGICAL_FILTER")) {
		return LogicalOperatorType::LOGICAL_FILTER;
	}
	if (StringUtil::Equals(value, "LOGICAL_AGGREGATE_AND_GROUP_BY")) {
		return LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY;
	}
	if (StringUtil::Equals(value, "LOGICAL_WINDOW")) {
		return LogicalOperatorType::LOGICAL_WINDOW;
	}
	if (StringUtil::Equals(value, "LOGICAL_UNNEST")) {
		return LogicalOperatorType::LOGICAL_UNNEST;
	}
	if (StringUtil::Equals(value, "LOGICAL_LIMIT")) {
		return LogicalOperatorType::LOGICAL_LIMIT;
	}
	if (StringUtil::Equals(value, "LOGICAL_ORDER_BY")) {
		return LogicalOperatorType::LOGICAL_ORDER_BY;
	}
	if (StringUtil::Equals(value, "LOGICAL_TOP_N")) {
		return LogicalOperatorType::LOGICAL_TOP_N;
	}
	if (StringUtil::Equals(value, "LOGICAL_COPY_TO_FILE")) {
		return LogicalOperatorType::LOGICAL_COPY_TO_FILE;
	}
	if (StringUtil::Equals(value, "LOGICAL_DISTINCT")) {
		return LogicalOperatorType::LOGICAL_DISTINCT;
	}
	if (StringUtil::Equals(value, "LOGICAL_SAMPLE")) {
		return LogicalOperatorType::LOGICAL_SAMPLE;
	}
	if (StringUtil::Equals(value, "LOGICAL_LIMIT_PERCENT")) {
		return LogicalOperatorType::LOGICAL_LIMIT_PERCENT;
	}
	if (StringUtil::Equals(value, "LOGICAL_PIVOT")) {
		return LogicalOperatorType::LOGICAL_PIVOT;
	}
	if (StringUtil::Equals(value, "LOGICAL_GET")) {
		return LogicalOperatorType::LOGICAL_GET;
	}
	if (StringUtil::Equals(value, "LOGICAL_CHUNK_GET")) {
		return LogicalOperatorType::LOGICAL_CHUNK_GET;
	}
	if (StringUtil::Equals(value, "LOGICAL_DELIM_GET")) {
		return LogicalOperatorType::LOGICAL_DELIM_GET;
	}
	if (StringUtil::Equals(value, "LOGICAL_EXPRESSION_GET")) {
		return LogicalOperatorType::LOGICAL_EXPRESSION_GET;
	}
	if (StringUtil::Equals(value, "LOGICAL_DUMMY_SCAN")) {
		return LogicalOperatorType::LOGICAL_DUMMY_SCAN;
	}
	if (StringUtil::Equals(value, "LOGICAL_EMPTY_RESULT")) {
		return LogicalOperatorType::LOGICAL_EMPTY_RESULT;
	}
	if (StringUtil::Equals(value, "LOGICAL_CTE_REF")) {
		return LogicalOperatorType::LOGICAL_CTE_REF;
	}
	if (StringUtil::Equals(value, "LOGICAL_JOIN")) {
		return LogicalOperatorType::LOGICAL_JOIN;
	}
	if (StringUtil::Equals(value, "LOGICAL_DELIM_JOIN")) {
		return LogicalOperatorType::LOGICAL_DELIM_JOIN;
	}
	if (StringUtil::Equals(value, "LOGICAL_COMPARISON_JOIN")) {
		return LogicalOperatorType::LOGICAL_COMPARISON_JOIN;
	}
	if (StringUtil::Equals(value, "LOGICAL_ANY_JOIN")) {
		return LogicalOperatorType::LOGICAL_ANY_JOIN;
	}
	if (StringUtil::Equals(value, "LOGICAL_CROSS_PRODUCT")) {
		return LogicalOperatorType::LOGICAL_CROSS_PRODUCT;
	}
	if (StringUtil::Equals(value, "LOGICAL_POSITIONAL_JOIN")) {
		return LogicalOperatorType::LOGICAL_POSITIONAL_JOIN;
	}
	if (StringUtil::Equals(value, "LOGICAL_ASOF_JOIN")) {
		return LogicalOperatorType::LOGICAL_ASOF_JOIN;
	}
	if (StringUtil::Equals(value, "LOGICAL_UNION")) {
		return LogicalOperatorType::LOGICAL_UNION;
	}
	if (StringUtil::Equals(value, "LOGICAL_EXCEPT")) {
		return LogicalOperatorType::LOGICAL_EXCEPT;
	}
	if (StringUtil::Equals(value, "LOGICAL_INTERSECT")) {
		return LogicalOperatorType::LOGICAL_INTERSECT;
	}
	if (StringUtil::Equals(value, "LOGICAL_RECURSIVE_CTE")) {
		return LogicalOperatorType::LOGICAL_RECURSIVE_CTE;
	}
	if (StringUtil::Equals(value, "LOGICAL_INSERT")) {
		return LogicalOperatorType::LOGICAL_INSERT;
	}
	if (StringUtil::Equals(value, "LOGICAL_DELETE")) {
		return LogicalOperatorType::LOGICAL_DELETE;
	}
	if (StringUtil::Equals(value, "LOGICAL_UPDATE")) {
		return LogicalOperatorType::LOGICAL_UPDATE;
	}
	if (StringUtil::Equals(value, "LOGICAL_ALTER")) {
		return LogicalOperatorType::LOGICAL_ALTER;
	}
	if (StringUtil::Equals(value, "LOGICAL_CREATE_TABLE")) {
		return LogicalOperatorType::LOGICAL_CREATE_TABLE;
	}
	if (StringUtil::Equals(value, "LOGICAL_CREATE_INDEX")) {
		return LogicalOperatorType::LOGICAL_CREATE_INDEX;
	}
	if (StringUtil::Equals(value, "LOGICAL_CREATE_SEQUENCE")) {
		return LogicalOperatorType::LOGICAL_CREATE_SEQUENCE;
	}
	if (StringUtil::Equals(value, "LOGICAL_CREATE_VIEW")) {
		return LogicalOperatorType::LOGICAL_CREATE_VIEW;
	}
	if (StringUtil::Equals(value, "LOGICAL_CREATE_SCHEMA")) {
		return LogicalOperatorType::LOGICAL_CREATE_SCHEMA;
	}
	if (StringUtil::Equals(value, "LOGICAL_CREATE_MACRO")) {
		return LogicalOperatorType::LOGICAL_CREATE_MACRO;
	}
	if (StringUtil::Equals(value, "LOGICAL_DROP")) {
		return LogicalOperatorType::LOGICAL_DROP;
	}
	if (StringUtil::Equals(value, "LOGICAL_PRAGMA")) {
		return LogicalOperatorType::LOGICAL_PRAGMA;
	}
	if (StringUtil::Equals(value, "LOGICAL_TRANSACTION")) {
		return LogicalOperatorType::LOGICAL_TRANSACTION;
	}
	if (StringUtil::Equals(value, "LOGICAL_CREATE_TYPE")) {
		return LogicalOperatorType::LOGICAL_CREATE_TYPE;
	}
	if (StringUtil::Equals(value, "LOGICAL_ATTACH")) {
		return LogicalOperatorType::LOGICAL_ATTACH;
	}
	if (StringUtil::Equals(value, "LOGICAL_DETACH")) {
		return LogicalOperatorType::LOGICAL_DETACH;
	}
	if (StringUtil::Equals(value, "LOGICAL_EXPLAIN")) {
		return LogicalOperatorType::LOGICAL_EXPLAIN;
	}
	if (StringUtil::Equals(value, "LOGICAL_SHOW")) {
		return LogicalOperatorType::LOGICAL_SHOW;
	}
	if (StringUtil::Equals(value, "LOGICAL_PREPARE")) {
		return LogicalOperatorType::LOGICAL_PREPARE;
	}
	if (StringUtil::Equals(value, "LOGICAL_EXECUTE")) {
		return LogicalOperatorType::LOGICAL_EXECUTE;
	}
	if (StringUtil::Equals(value, "LOGICAL_EXPORT")) {
		return LogicalOperatorType::LOGICAL_EXPORT;
	}
	if (StringUtil::Equals(value, "LOGICAL_VACUUM")) {
		return LogicalOperatorType::LOGICAL_VACUUM;
	}
	if (StringUtil::Equals(value, "LOGICAL_SET")) {
		return LogicalOperatorType::LOGICAL_SET;
	}
	if (StringUtil::Equals(value, "LOGICAL_LOAD")) {
		return LogicalOperatorType::LOGICAL_LOAD;
	}
	if (StringUtil::Equals(value, "LOGICAL_RESET")) {
		return LogicalOperatorType::LOGICAL_RESET;
	}
	if (StringUtil::Equals(value, "LOGICAL_EXTENSION_OPERATOR")) {
		return LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<OperatorResultType>(OperatorResultType value) {
	switch (value) {
	case OperatorResultType::NEED_MORE_INPUT:
		return "NEED_MORE_INPUT";
	case OperatorResultType::HAVE_MORE_OUTPUT:
		return "HAVE_MORE_OUTPUT";
	case OperatorResultType::FINISHED:
		return "FINISHED";
	case OperatorResultType::BLOCKED:
		return "BLOCKED";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
OperatorResultType EnumUtil::FromString<OperatorResultType>(const char *value) {
	if (StringUtil::Equals(value, "NEED_MORE_INPUT")) {
		return OperatorResultType::NEED_MORE_INPUT;
	}
	if (StringUtil::Equals(value, "HAVE_MORE_OUTPUT")) {
		return OperatorResultType::HAVE_MORE_OUTPUT;
	}
	if (StringUtil::Equals(value, "FINISHED")) {
		return OperatorResultType::FINISHED;
	}
	if (StringUtil::Equals(value, "BLOCKED")) {
		return OperatorResultType::BLOCKED;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<OperatorFinalizeResultType>(OperatorFinalizeResultType value) {
	switch (value) {
	case OperatorFinalizeResultType::HAVE_MORE_OUTPUT:
		return "HAVE_MORE_OUTPUT";
	case OperatorFinalizeResultType::FINISHED:
		return "FINISHED";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
OperatorFinalizeResultType EnumUtil::FromString<OperatorFinalizeResultType>(const char *value) {
	if (StringUtil::Equals(value, "HAVE_MORE_OUTPUT")) {
		return OperatorFinalizeResultType::HAVE_MORE_OUTPUT;
	}
	if (StringUtil::Equals(value, "FINISHED")) {
		return OperatorFinalizeResultType::FINISHED;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<SourceResultType>(SourceResultType value) {
	switch (value) {
	case SourceResultType::HAVE_MORE_OUTPUT:
		return "HAVE_MORE_OUTPUT";
	case SourceResultType::FINISHED:
		return "FINISHED";
	case SourceResultType::BLOCKED:
		return "BLOCKED";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
SourceResultType EnumUtil::FromString<SourceResultType>(const char *value) {
	if (StringUtil::Equals(value, "HAVE_MORE_OUTPUT")) {
		return SourceResultType::HAVE_MORE_OUTPUT;
	}
	if (StringUtil::Equals(value, "FINISHED")) {
		return SourceResultType::FINISHED;
	}
	if (StringUtil::Equals(value, "BLOCKED")) {
		return SourceResultType::BLOCKED;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<SinkResultType>(SinkResultType value) {
	switch (value) {
	case SinkResultType::NEED_MORE_INPUT:
		return "NEED_MORE_INPUT";
	case SinkResultType::FINISHED:
		return "FINISHED";
	case SinkResultType::BLOCKED:
		return "BLOCKED";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
SinkResultType EnumUtil::FromString<SinkResultType>(const char *value) {
	if (StringUtil::Equals(value, "NEED_MORE_INPUT")) {
		return SinkResultType::NEED_MORE_INPUT;
	}
	if (StringUtil::Equals(value, "FINISHED")) {
		return SinkResultType::FINISHED;
	}
	if (StringUtil::Equals(value, "BLOCKED")) {
		return SinkResultType::BLOCKED;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<SinkFinalizeType>(SinkFinalizeType value) {
	switch (value) {
	case SinkFinalizeType::READY:
		return "READY";
	case SinkFinalizeType::NO_OUTPUT_POSSIBLE:
		return "NO_OUTPUT_POSSIBLE";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
SinkFinalizeType EnumUtil::FromString<SinkFinalizeType>(const char *value) {
	if (StringUtil::Equals(value, "READY")) {
		return SinkFinalizeType::READY;
	}
	if (StringUtil::Equals(value, "NO_OUTPUT_POSSIBLE")) {
		return SinkFinalizeType::NO_OUTPUT_POSSIBLE;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<JoinRefType>(JoinRefType value) {
	switch (value) {
	case JoinRefType::REGULAR:
		return "REGULAR";
	case JoinRefType::NATURAL:
		return "NATURAL";
	case JoinRefType::CROSS:
		return "CROSS";
	case JoinRefType::POSITIONAL:
		return "POSITIONAL";
	case JoinRefType::ASOF:
		return "ASOF";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
JoinRefType EnumUtil::FromString<JoinRefType>(const char *value) {
	if (StringUtil::Equals(value, "REGULAR")) {
		return JoinRefType::REGULAR;
	}
	if (StringUtil::Equals(value, "NATURAL")) {
		return JoinRefType::NATURAL;
	}
	if (StringUtil::Equals(value, "CROSS")) {
		return JoinRefType::CROSS;
	}
	if (StringUtil::Equals(value, "POSITIONAL")) {
		return JoinRefType::POSITIONAL;
	}
	if (StringUtil::Equals(value, "ASOF")) {
		return JoinRefType::ASOF;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<UndoFlags>(UndoFlags value) {
	switch (value) {
	case UndoFlags::EMPTY_ENTRY:
		return "EMPTY_ENTRY";
	case UndoFlags::CATALOG_ENTRY:
		return "CATALOG_ENTRY";
	case UndoFlags::INSERT_TUPLE:
		return "INSERT_TUPLE";
	case UndoFlags::DELETE_TUPLE:
		return "DELETE_TUPLE";
	case UndoFlags::UPDATE_TUPLE:
		return "UPDATE_TUPLE";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
UndoFlags EnumUtil::FromString<UndoFlags>(const char *value) {
	if (StringUtil::Equals(value, "EMPTY_ENTRY")) {
		return UndoFlags::EMPTY_ENTRY;
	}
	if (StringUtil::Equals(value, "CATALOG_ENTRY")) {
		return UndoFlags::CATALOG_ENTRY;
	}
	if (StringUtil::Equals(value, "INSERT_TUPLE")) {
		return UndoFlags::INSERT_TUPLE;
	}
	if (StringUtil::Equals(value, "DELETE_TUPLE")) {
		return UndoFlags::DELETE_TUPLE;
	}
	if (StringUtil::Equals(value, "UPDATE_TUPLE")) {
		return UndoFlags::UPDATE_TUPLE;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<SetOperationType>(SetOperationType value) {
	switch (value) {
	case SetOperationType::NONE:
		return "NONE";
	case SetOperationType::UNION:
		return "UNION";
	case SetOperationType::EXCEPT:
		return "EXCEPT";
	case SetOperationType::INTERSECT:
		return "INTERSECT";
	case SetOperationType::UNION_BY_NAME:
		return "UNION_BY_NAME";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
SetOperationType EnumUtil::FromString<SetOperationType>(const char *value) {
	if (StringUtil::Equals(value, "NONE")) {
		return SetOperationType::NONE;
	}
	if (StringUtil::Equals(value, "UNION")) {
		return SetOperationType::UNION;
	}
	if (StringUtil::Equals(value, "EXCEPT")) {
		return SetOperationType::EXCEPT;
	}
	if (StringUtil::Equals(value, "INTERSECT")) {
		return SetOperationType::INTERSECT;
	}
	if (StringUtil::Equals(value, "UNION_BY_NAME")) {
		return SetOperationType::UNION_BY_NAME;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<OptimizerType>(OptimizerType value) {
	switch (value) {
	case OptimizerType::INVALID:
		return "INVALID";
	case OptimizerType::EXPRESSION_REWRITER:
		return "EXPRESSION_REWRITER";
	case OptimizerType::FILTER_PULLUP:
		return "FILTER_PULLUP";
	case OptimizerType::FILTER_PUSHDOWN:
		return "FILTER_PUSHDOWN";
	case OptimizerType::REGEX_RANGE:
		return "REGEX_RANGE";
	case OptimizerType::IN_CLAUSE:
		return "IN_CLAUSE";
	case OptimizerType::JOIN_ORDER:
		return "JOIN_ORDER";
	case OptimizerType::DELIMINATOR:
		return "DELIMINATOR";
	case OptimizerType::UNNEST_REWRITER:
		return "UNNEST_REWRITER";
	case OptimizerType::UNUSED_COLUMNS:
		return "UNUSED_COLUMNS";
	case OptimizerType::STATISTICS_PROPAGATION:
		return "STATISTICS_PROPAGATION";
	case OptimizerType::COMMON_SUBEXPRESSIONS:
		return "COMMON_SUBEXPRESSIONS";
	case OptimizerType::COMMON_AGGREGATE:
		return "COMMON_AGGREGATE";
	case OptimizerType::COLUMN_LIFETIME:
		return "COLUMN_LIFETIME";
	case OptimizerType::TOP_N:
		return "TOP_N";
	case OptimizerType::REORDER_FILTER:
		return "REORDER_FILTER";
	case OptimizerType::EXTENSION:
		return "EXTENSION";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
OptimizerType EnumUtil::FromString<OptimizerType>(const char *value) {
	if (StringUtil::Equals(value, "INVALID")) {
		return OptimizerType::INVALID;
	}
	if (StringUtil::Equals(value, "EXPRESSION_REWRITER")) {
		return OptimizerType::EXPRESSION_REWRITER;
	}
	if (StringUtil::Equals(value, "FILTER_PULLUP")) {
		return OptimizerType::FILTER_PULLUP;
	}
	if (StringUtil::Equals(value, "FILTER_PUSHDOWN")) {
		return OptimizerType::FILTER_PUSHDOWN;
	}
	if (StringUtil::Equals(value, "REGEX_RANGE")) {
		return OptimizerType::REGEX_RANGE;
	}
	if (StringUtil::Equals(value, "IN_CLAUSE")) {
		return OptimizerType::IN_CLAUSE;
	}
	if (StringUtil::Equals(value, "JOIN_ORDER")) {
		return OptimizerType::JOIN_ORDER;
	}
	if (StringUtil::Equals(value, "DELIMINATOR")) {
		return OptimizerType::DELIMINATOR;
	}
	if (StringUtil::Equals(value, "UNNEST_REWRITER")) {
		return OptimizerType::UNNEST_REWRITER;
	}
	if (StringUtil::Equals(value, "UNUSED_COLUMNS")) {
		return OptimizerType::UNUSED_COLUMNS;
	}
	if (StringUtil::Equals(value, "STATISTICS_PROPAGATION")) {
		return OptimizerType::STATISTICS_PROPAGATION;
	}
	if (StringUtil::Equals(value, "COMMON_SUBEXPRESSIONS")) {
		return OptimizerType::COMMON_SUBEXPRESSIONS;
	}
	if (StringUtil::Equals(value, "COMMON_AGGREGATE")) {
		return OptimizerType::COMMON_AGGREGATE;
	}
	if (StringUtil::Equals(value, "COLUMN_LIFETIME")) {
		return OptimizerType::COLUMN_LIFETIME;
	}
	if (StringUtil::Equals(value, "TOP_N")) {
		return OptimizerType::TOP_N;
	}
	if (StringUtil::Equals(value, "REORDER_FILTER")) {
		return OptimizerType::REORDER_FILTER;
	}
	if (StringUtil::Equals(value, "EXTENSION")) {
		return OptimizerType::EXTENSION;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<CompressionType>(CompressionType value) {
	switch (value) {
	case CompressionType::COMPRESSION_AUTO:
		return "COMPRESSION_AUTO";
	case CompressionType::COMPRESSION_UNCOMPRESSED:
		return "COMPRESSION_UNCOMPRESSED";
	case CompressionType::COMPRESSION_CONSTANT:
		return "COMPRESSION_CONSTANT";
	case CompressionType::COMPRESSION_RLE:
		return "COMPRESSION_RLE";
	case CompressionType::COMPRESSION_DICTIONARY:
		return "COMPRESSION_DICTIONARY";
	case CompressionType::COMPRESSION_PFOR_DELTA:
		return "COMPRESSION_PFOR_DELTA";
	case CompressionType::COMPRESSION_BITPACKING:
		return "COMPRESSION_BITPACKING";
	case CompressionType::COMPRESSION_FSST:
		return "COMPRESSION_FSST";
	case CompressionType::COMPRESSION_CHIMP:
		return "COMPRESSION_CHIMP";
	case CompressionType::COMPRESSION_PATAS:
		return "COMPRESSION_PATAS";
	case CompressionType::COMPRESSION_COUNT:
		return "COMPRESSION_COUNT";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
CompressionType EnumUtil::FromString<CompressionType>(const char *value) {
	if (StringUtil::Equals(value, "COMPRESSION_AUTO")) {
		return CompressionType::COMPRESSION_AUTO;
	}
	if (StringUtil::Equals(value, "COMPRESSION_UNCOMPRESSED")) {
		return CompressionType::COMPRESSION_UNCOMPRESSED;
	}
	if (StringUtil::Equals(value, "COMPRESSION_CONSTANT")) {
		return CompressionType::COMPRESSION_CONSTANT;
	}
	if (StringUtil::Equals(value, "COMPRESSION_RLE")) {
		return CompressionType::COMPRESSION_RLE;
	}
	if (StringUtil::Equals(value, "COMPRESSION_DICTIONARY")) {
		return CompressionType::COMPRESSION_DICTIONARY;
	}
	if (StringUtil::Equals(value, "COMPRESSION_PFOR_DELTA")) {
		return CompressionType::COMPRESSION_PFOR_DELTA;
	}
	if (StringUtil::Equals(value, "COMPRESSION_BITPACKING")) {
		return CompressionType::COMPRESSION_BITPACKING;
	}
	if (StringUtil::Equals(value, "COMPRESSION_FSST")) {
		return CompressionType::COMPRESSION_FSST;
	}
	if (StringUtil::Equals(value, "COMPRESSION_CHIMP")) {
		return CompressionType::COMPRESSION_CHIMP;
	}
	if (StringUtil::Equals(value, "COMPRESSION_PATAS")) {
		return CompressionType::COMPRESSION_PATAS;
	}
	if (StringUtil::Equals(value, "COMPRESSION_COUNT")) {
		return CompressionType::COMPRESSION_COUNT;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<AggregateHandling>(AggregateHandling value) {
	switch (value) {
	case AggregateHandling::STANDARD_HANDLING:
		return "STANDARD_HANDLING";
	case AggregateHandling::NO_AGGREGATES_ALLOWED:
		return "NO_AGGREGATES_ALLOWED";
	case AggregateHandling::FORCE_AGGREGATES:
		return "FORCE_AGGREGATES";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
AggregateHandling EnumUtil::FromString<AggregateHandling>(const char *value) {
	if (StringUtil::Equals(value, "STANDARD_HANDLING")) {
		return AggregateHandling::STANDARD_HANDLING;
	}
	if (StringUtil::Equals(value, "NO_AGGREGATES_ALLOWED")) {
		return AggregateHandling::NO_AGGREGATES_ALLOWED;
	}
	if (StringUtil::Equals(value, "FORCE_AGGREGATES")) {
		return AggregateHandling::FORCE_AGGREGATES;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<TableReferenceType>(TableReferenceType value) {
	switch (value) {
	case TableReferenceType::INVALID:
		return "INVALID";
	case TableReferenceType::BASE_TABLE:
		return "BASE_TABLE";
	case TableReferenceType::SUBQUERY:
		return "SUBQUERY";
	case TableReferenceType::JOIN:
		return "JOIN";
	case TableReferenceType::TABLE_FUNCTION:
		return "TABLE_FUNCTION";
	case TableReferenceType::EXPRESSION_LIST:
		return "EXPRESSION_LIST";
	case TableReferenceType::CTE:
		return "CTE";
	case TableReferenceType::EMPTY:
		return "EMPTY";
	case TableReferenceType::PIVOT:
		return "PIVOT";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
TableReferenceType EnumUtil::FromString<TableReferenceType>(const char *value) {
	if (StringUtil::Equals(value, "INVALID")) {
		return TableReferenceType::INVALID;
	}
	if (StringUtil::Equals(value, "BASE_TABLE")) {
		return TableReferenceType::BASE_TABLE;
	}
	if (StringUtil::Equals(value, "SUBQUERY")) {
		return TableReferenceType::SUBQUERY;
	}
	if (StringUtil::Equals(value, "JOIN")) {
		return TableReferenceType::JOIN;
	}
	if (StringUtil::Equals(value, "TABLE_FUNCTION")) {
		return TableReferenceType::TABLE_FUNCTION;
	}
	if (StringUtil::Equals(value, "EXPRESSION_LIST")) {
		return TableReferenceType::EXPRESSION_LIST;
	}
	if (StringUtil::Equals(value, "CTE")) {
		return TableReferenceType::CTE;
	}
	if (StringUtil::Equals(value, "EMPTY")) {
		return TableReferenceType::EMPTY;
	}
	if (StringUtil::Equals(value, "PIVOT")) {
		return TableReferenceType::PIVOT;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<RelationType>(RelationType value) {
	switch (value) {
	case RelationType::INVALID_RELATION:
		return "INVALID_RELATION";
	case RelationType::TABLE_RELATION:
		return "TABLE_RELATION";
	case RelationType::PROJECTION_RELATION:
		return "PROJECTION_RELATION";
	case RelationType::FILTER_RELATION:
		return "FILTER_RELATION";
	case RelationType::EXPLAIN_RELATION:
		return "EXPLAIN_RELATION";
	case RelationType::CROSS_PRODUCT_RELATION:
		return "CROSS_PRODUCT_RELATION";
	case RelationType::JOIN_RELATION:
		return "JOIN_RELATION";
	case RelationType::AGGREGATE_RELATION:
		return "AGGREGATE_RELATION";
	case RelationType::SET_OPERATION_RELATION:
		return "SET_OPERATION_RELATION";
	case RelationType::DISTINCT_RELATION:
		return "DISTINCT_RELATION";
	case RelationType::LIMIT_RELATION:
		return "LIMIT_RELATION";
	case RelationType::ORDER_RELATION:
		return "ORDER_RELATION";
	case RelationType::CREATE_VIEW_RELATION:
		return "CREATE_VIEW_RELATION";
	case RelationType::CREATE_TABLE_RELATION:
		return "CREATE_TABLE_RELATION";
	case RelationType::INSERT_RELATION:
		return "INSERT_RELATION";
	case RelationType::VALUE_LIST_RELATION:
		return "VALUE_LIST_RELATION";
	case RelationType::DELETE_RELATION:
		return "DELETE_RELATION";
	case RelationType::UPDATE_RELATION:
		return "UPDATE_RELATION";
	case RelationType::WRITE_CSV_RELATION:
		return "WRITE_CSV_RELATION";
	case RelationType::WRITE_PARQUET_RELATION:
		return "WRITE_PARQUET_RELATION";
	case RelationType::READ_CSV_RELATION:
		return "READ_CSV_RELATION";
	case RelationType::SUBQUERY_RELATION:
		return "SUBQUERY_RELATION";
	case RelationType::TABLE_FUNCTION_RELATION:
		return "TABLE_FUNCTION_RELATION";
	case RelationType::VIEW_RELATION:
		return "VIEW_RELATION";
	case RelationType::QUERY_RELATION:
		return "QUERY_RELATION";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
RelationType EnumUtil::FromString<RelationType>(const char *value) {
	if (StringUtil::Equals(value, "INVALID_RELATION")) {
		return RelationType::INVALID_RELATION;
	}
	if (StringUtil::Equals(value, "TABLE_RELATION")) {
		return RelationType::TABLE_RELATION;
	}
	if (StringUtil::Equals(value, "PROJECTION_RELATION")) {
		return RelationType::PROJECTION_RELATION;
	}
	if (StringUtil::Equals(value, "FILTER_RELATION")) {
		return RelationType::FILTER_RELATION;
	}
	if (StringUtil::Equals(value, "EXPLAIN_RELATION")) {
		return RelationType::EXPLAIN_RELATION;
	}
	if (StringUtil::Equals(value, "CROSS_PRODUCT_RELATION")) {
		return RelationType::CROSS_PRODUCT_RELATION;
	}
	if (StringUtil::Equals(value, "JOIN_RELATION")) {
		return RelationType::JOIN_RELATION;
	}
	if (StringUtil::Equals(value, "AGGREGATE_RELATION")) {
		return RelationType::AGGREGATE_RELATION;
	}
	if (StringUtil::Equals(value, "SET_OPERATION_RELATION")) {
		return RelationType::SET_OPERATION_RELATION;
	}
	if (StringUtil::Equals(value, "DISTINCT_RELATION")) {
		return RelationType::DISTINCT_RELATION;
	}
	if (StringUtil::Equals(value, "LIMIT_RELATION")) {
		return RelationType::LIMIT_RELATION;
	}
	if (StringUtil::Equals(value, "ORDER_RELATION")) {
		return RelationType::ORDER_RELATION;
	}
	if (StringUtil::Equals(value, "CREATE_VIEW_RELATION")) {
		return RelationType::CREATE_VIEW_RELATION;
	}
	if (StringUtil::Equals(value, "CREATE_TABLE_RELATION")) {
		return RelationType::CREATE_TABLE_RELATION;
	}
	if (StringUtil::Equals(value, "INSERT_RELATION")) {
		return RelationType::INSERT_RELATION;
	}
	if (StringUtil::Equals(value, "VALUE_LIST_RELATION")) {
		return RelationType::VALUE_LIST_RELATION;
	}
	if (StringUtil::Equals(value, "DELETE_RELATION")) {
		return RelationType::DELETE_RELATION;
	}
	if (StringUtil::Equals(value, "UPDATE_RELATION")) {
		return RelationType::UPDATE_RELATION;
	}
	if (StringUtil::Equals(value, "WRITE_CSV_RELATION")) {
		return RelationType::WRITE_CSV_RELATION;
	}
	if (StringUtil::Equals(value, "WRITE_PARQUET_RELATION")) {
		return RelationType::WRITE_PARQUET_RELATION;
	}
	if (StringUtil::Equals(value, "READ_CSV_RELATION")) {
		return RelationType::READ_CSV_RELATION;
	}
	if (StringUtil::Equals(value, "SUBQUERY_RELATION")) {
		return RelationType::SUBQUERY_RELATION;
	}
	if (StringUtil::Equals(value, "TABLE_FUNCTION_RELATION")) {
		return RelationType::TABLE_FUNCTION_RELATION;
	}
	if (StringUtil::Equals(value, "VIEW_RELATION")) {
		return RelationType::VIEW_RELATION;
	}
	if (StringUtil::Equals(value, "QUERY_RELATION")) {
		return RelationType::QUERY_RELATION;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<FilterPropagateResult>(FilterPropagateResult value) {
	switch (value) {
	case FilterPropagateResult::NO_PRUNING_POSSIBLE:
		return "NO_PRUNING_POSSIBLE";
	case FilterPropagateResult::FILTER_ALWAYS_TRUE:
		return "FILTER_ALWAYS_TRUE";
	case FilterPropagateResult::FILTER_ALWAYS_FALSE:
		return "FILTER_ALWAYS_FALSE";
	case FilterPropagateResult::FILTER_TRUE_OR_NULL:
		return "FILTER_TRUE_OR_NULL";
	case FilterPropagateResult::FILTER_FALSE_OR_NULL:
		return "FILTER_FALSE_OR_NULL";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
FilterPropagateResult EnumUtil::FromString<FilterPropagateResult>(const char *value) {
	if (StringUtil::Equals(value, "NO_PRUNING_POSSIBLE")) {
		return FilterPropagateResult::NO_PRUNING_POSSIBLE;
	}
	if (StringUtil::Equals(value, "FILTER_ALWAYS_TRUE")) {
		return FilterPropagateResult::FILTER_ALWAYS_TRUE;
	}
	if (StringUtil::Equals(value, "FILTER_ALWAYS_FALSE")) {
		return FilterPropagateResult::FILTER_ALWAYS_FALSE;
	}
	if (StringUtil::Equals(value, "FILTER_TRUE_OR_NULL")) {
		return FilterPropagateResult::FILTER_TRUE_OR_NULL;
	}
	if (StringUtil::Equals(value, "FILTER_FALSE_OR_NULL")) {
		return FilterPropagateResult::FILTER_FALSE_OR_NULL;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<IndexType>(IndexType value) {
	switch (value) {
	case IndexType::INVALID:
		return "INVALID";
	case IndexType::ART:
		return "ART";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
IndexType EnumUtil::FromString<IndexType>(const char *value) {
	if (StringUtil::Equals(value, "INVALID")) {
		return IndexType::INVALID;
	}
	if (StringUtil::Equals(value, "ART")) {
		return IndexType::ART;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<ExplainOutputType>(ExplainOutputType value) {
	switch (value) {
	case ExplainOutputType::ALL:
		return "ALL";
	case ExplainOutputType::OPTIMIZED_ONLY:
		return "OPTIMIZED_ONLY";
	case ExplainOutputType::PHYSICAL_ONLY:
		return "PHYSICAL_ONLY";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
ExplainOutputType EnumUtil::FromString<ExplainOutputType>(const char *value) {
	if (StringUtil::Equals(value, "ALL")) {
		return ExplainOutputType::ALL;
	}
	if (StringUtil::Equals(value, "OPTIMIZED_ONLY")) {
		return ExplainOutputType::OPTIMIZED_ONLY;
	}
	if (StringUtil::Equals(value, "PHYSICAL_ONLY")) {
		return ExplainOutputType::PHYSICAL_ONLY;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<NType>(NType value) {
	switch (value) {
	case NType::PREFIX_SEGMENT:
		return "PREFIX_SEGMENT";
	case NType::LEAF_SEGMENT:
		return "LEAF_SEGMENT";
	case NType::LEAF:
		return "LEAF";
	case NType::NODE_4:
		return "NODE_4";
	case NType::NODE_16:
		return "NODE_16";
	case NType::NODE_48:
		return "NODE_48";
	case NType::NODE_256:
		return "NODE_256";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
NType EnumUtil::FromString<NType>(const char *value) {
	if (StringUtil::Equals(value, "PREFIX_SEGMENT")) {
		return NType::PREFIX_SEGMENT;
	}
	if (StringUtil::Equals(value, "LEAF_SEGMENT")) {
		return NType::LEAF_SEGMENT;
	}
	if (StringUtil::Equals(value, "LEAF")) {
		return NType::LEAF;
	}
	if (StringUtil::Equals(value, "NODE_4")) {
		return NType::NODE_4;
	}
	if (StringUtil::Equals(value, "NODE_16")) {
		return NType::NODE_16;
	}
	if (StringUtil::Equals(value, "NODE_48")) {
		return NType::NODE_48;
	}
	if (StringUtil::Equals(value, "NODE_256")) {
		return NType::NODE_256;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<VerifyExistenceType>(VerifyExistenceType value) {
	switch (value) {
	case VerifyExistenceType::APPEND:
		return "APPEND";
	case VerifyExistenceType::APPEND_FK:
		return "APPEND_FK";
	case VerifyExistenceType::DELETE_FK:
		return "DELETE_FK";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
VerifyExistenceType EnumUtil::FromString<VerifyExistenceType>(const char *value) {
	if (StringUtil::Equals(value, "APPEND")) {
		return VerifyExistenceType::APPEND;
	}
	if (StringUtil::Equals(value, "APPEND_FK")) {
		return VerifyExistenceType::APPEND_FK;
	}
	if (StringUtil::Equals(value, "DELETE_FK")) {
		return VerifyExistenceType::DELETE_FK;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<ParserMode>(ParserMode value) {
	switch (value) {
	case ParserMode::PARSING:
		return "PARSING";
	case ParserMode::SNIFFING_DIALECT:
		return "SNIFFING_DIALECT";
	case ParserMode::SNIFFING_DATATYPES:
		return "SNIFFING_DATATYPES";
	case ParserMode::PARSING_HEADER:
		return "PARSING_HEADER";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
ParserMode EnumUtil::FromString<ParserMode>(const char *value) {
	if (StringUtil::Equals(value, "PARSING")) {
		return ParserMode::PARSING;
	}
	if (StringUtil::Equals(value, "SNIFFING_DIALECT")) {
		return ParserMode::SNIFFING_DIALECT;
	}
	if (StringUtil::Equals(value, "SNIFFING_DATATYPES")) {
		return ParserMode::SNIFFING_DATATYPES;
	}
	if (StringUtil::Equals(value, "PARSING_HEADER")) {
		return ParserMode::PARSING_HEADER;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<ErrorType>(ErrorType value) {
	switch (value) {
	case ErrorType::UNSIGNED_EXTENSION:
		return "UNSIGNED_EXTENSION";
	case ErrorType::INVALIDATED_TRANSACTION:
		return "INVALIDATED_TRANSACTION";
	case ErrorType::INVALIDATED_DATABASE:
		return "INVALIDATED_DATABASE";
	case ErrorType::ERROR_COUNT:
		return "ERROR_COUNT";
	case ErrorType::INVALID:
		return "INVALID";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
ErrorType EnumUtil::FromString<ErrorType>(const char *value) {
	if (StringUtil::Equals(value, "UNSIGNED_EXTENSION")) {
		return ErrorType::UNSIGNED_EXTENSION;
	}
	if (StringUtil::Equals(value, "INVALIDATED_TRANSACTION")) {
		return ErrorType::INVALIDATED_TRANSACTION;
	}
	if (StringUtil::Equals(value, "INVALIDATED_DATABASE")) {
		return ErrorType::INVALIDATED_DATABASE;
	}
	if (StringUtil::Equals(value, "ERROR_COUNT")) {
		return ErrorType::ERROR_COUNT;
	}
	if (StringUtil::Equals(value, "INVALID")) {
		return ErrorType::INVALID;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<AppenderType>(AppenderType value) {
	switch (value) {
	case AppenderType::LOGICAL:
		return "LOGICAL";
	case AppenderType::PHYSICAL:
		return "PHYSICAL";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
AppenderType EnumUtil::FromString<AppenderType>(const char *value) {
	if (StringUtil::Equals(value, "LOGICAL")) {
		return AppenderType::LOGICAL;
	}
	if (StringUtil::Equals(value, "PHYSICAL")) {
		return AppenderType::PHYSICAL;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<CheckpointAbort>(CheckpointAbort value) {
	switch (value) {
	case CheckpointAbort::NO_ABORT:
		return "NO_ABORT";
	case CheckpointAbort::DEBUG_ABORT_BEFORE_TRUNCATE:
		return "DEBUG_ABORT_BEFORE_TRUNCATE";
	case CheckpointAbort::DEBUG_ABORT_BEFORE_HEADER:
		return "DEBUG_ABORT_BEFORE_HEADER";
	case CheckpointAbort::DEBUG_ABORT_AFTER_FREE_LIST_WRITE:
		return "DEBUG_ABORT_AFTER_FREE_LIST_WRITE";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
CheckpointAbort EnumUtil::FromString<CheckpointAbort>(const char *value) {
	if (StringUtil::Equals(value, "NO_ABORT")) {
		return CheckpointAbort::NO_ABORT;
	}
	if (StringUtil::Equals(value, "DEBUG_ABORT_BEFORE_TRUNCATE")) {
		return CheckpointAbort::DEBUG_ABORT_BEFORE_TRUNCATE;
	}
	if (StringUtil::Equals(value, "DEBUG_ABORT_BEFORE_HEADER")) {
		return CheckpointAbort::DEBUG_ABORT_BEFORE_HEADER;
	}
	if (StringUtil::Equals(value, "DEBUG_ABORT_AFTER_FREE_LIST_WRITE")) {
		return CheckpointAbort::DEBUG_ABORT_AFTER_FREE_LIST_WRITE;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<ExtensionLoadResult>(ExtensionLoadResult value) {
	switch (value) {
	case ExtensionLoadResult::LOADED_EXTENSION:
		return "LOADED_EXTENSION";
	case ExtensionLoadResult::EXTENSION_UNKNOWN:
		return "EXTENSION_UNKNOWN";
	case ExtensionLoadResult::NOT_LOADED:
		return "NOT_LOADED";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
ExtensionLoadResult EnumUtil::FromString<ExtensionLoadResult>(const char *value) {
	if (StringUtil::Equals(value, "LOADED_EXTENSION")) {
		return ExtensionLoadResult::LOADED_EXTENSION;
	}
	if (StringUtil::Equals(value, "EXTENSION_UNKNOWN")) {
		return ExtensionLoadResult::EXTENSION_UNKNOWN;
	}
	if (StringUtil::Equals(value, "NOT_LOADED")) {
		return ExtensionLoadResult::NOT_LOADED;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<QueryResultType>(QueryResultType value) {
	switch (value) {
	case QueryResultType::MATERIALIZED_RESULT:
		return "MATERIALIZED_RESULT";
	case QueryResultType::STREAM_RESULT:
		return "STREAM_RESULT";
	case QueryResultType::PENDING_RESULT:
		return "PENDING_RESULT";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
QueryResultType EnumUtil::FromString<QueryResultType>(const char *value) {
	if (StringUtil::Equals(value, "MATERIALIZED_RESULT")) {
		return QueryResultType::MATERIALIZED_RESULT;
	}
	if (StringUtil::Equals(value, "STREAM_RESULT")) {
		return QueryResultType::STREAM_RESULT;
	}
	if (StringUtil::Equals(value, "PENDING_RESULT")) {
		return QueryResultType::PENDING_RESULT;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

template <>
const char *EnumUtil::ToChars<CAPIResultSetType>(CAPIResultSetType value) {
	switch (value) {
	case CAPIResultSetType::CAPI_RESULT_TYPE_NONE:
		return "CAPI_RESULT_TYPE_NONE";
	case CAPIResultSetType::CAPI_RESULT_TYPE_MATERIALIZED:
		return "CAPI_RESULT_TYPE_MATERIALIZED";
	case CAPIResultSetType::CAPI_RESULT_TYPE_STREAMING:
		return "CAPI_RESULT_TYPE_STREAMING";
	case CAPIResultSetType::CAPI_RESULT_TYPE_DEPRECATED:
		return "CAPI_RESULT_TYPE_DEPRECATED";
	default:
		throw NotImplementedException(StringUtil::Format("Enum value: '%d' not implemented", value));
	}
}

template <>
CAPIResultSetType EnumUtil::FromString<CAPIResultSetType>(const char *value) {
	if (StringUtil::Equals(value, "CAPI_RESULT_TYPE_NONE")) {
		return CAPIResultSetType::CAPI_RESULT_TYPE_NONE;
	}
	if (StringUtil::Equals(value, "CAPI_RESULT_TYPE_MATERIALIZED")) {
		return CAPIResultSetType::CAPI_RESULT_TYPE_MATERIALIZED;
	}
	if (StringUtil::Equals(value, "CAPI_RESULT_TYPE_STREAMING")) {
		return CAPIResultSetType::CAPI_RESULT_TYPE_STREAMING;
	}
	if (StringUtil::Equals(value, "CAPI_RESULT_TYPE_DEPRECATED")) {
		return CAPIResultSetType::CAPI_RESULT_TYPE_DEPRECATED;
	}
	throw NotImplementedException(StringUtil::Format("Enum value: '%s' not implemented", value));
}

} // namespace duckdb




namespace duckdb {

// LCOV_EXCL_START
string CatalogTypeToString(CatalogType type) {
	switch (type) {
	case CatalogType::COLLATION_ENTRY:
		return "Collation";
	case CatalogType::TYPE_ENTRY:
		return "Type";
	case CatalogType::TABLE_ENTRY:
		return "Table";
	case CatalogType::SCHEMA_ENTRY:
		return "Schema";
	case CatalogType::DATABASE_ENTRY:
		return "Database";
	case CatalogType::TABLE_FUNCTION_ENTRY:
		return "Table Function";
	case CatalogType::SCALAR_FUNCTION_ENTRY:
		return "Scalar Function";
	case CatalogType::AGGREGATE_FUNCTION_ENTRY:
		return "Aggregate Function";
	case CatalogType::COPY_FUNCTION_ENTRY:
		return "Copy Function";
	case CatalogType::PRAGMA_FUNCTION_ENTRY:
		return "Pragma Function";
	case CatalogType::MACRO_ENTRY:
		return "Macro Function";
	case CatalogType::TABLE_MACRO_ENTRY:
		return "Table Macro Function";
	case CatalogType::VIEW_ENTRY:
		return "View";
	case CatalogType::INDEX_ENTRY:
		return "Index";
	case CatalogType::PREPARED_STATEMENT:
		return "Prepared Statement";
	case CatalogType::SEQUENCE_ENTRY:
		return "Sequence";
	case CatalogType::INVALID:
	case CatalogType::DELETED_ENTRY:
	case CatalogType::UPDATED_ENTRY:
		break;
	}
	return "INVALID";
}
// LCOV_EXCL_STOP

} // namespace duckdb




namespace duckdb {

// LCOV_EXCL_START

vector<string> ListCompressionTypes(void) {
	vector<string> compression_types;
	uint8_t amount_of_compression_options = (uint8_t)CompressionType::COMPRESSION_COUNT;
	compression_types.reserve(amount_of_compression_options);
	for (uint8_t i = 0; i < amount_of_compression_options; i++) {
		compression_types.push_back(CompressionTypeToString((CompressionType)i));
	}
	return compression_types;
}

CompressionType CompressionTypeFromString(const string &str) {
	auto compression = StringUtil::Lower(str);
	if (compression == "uncompressed") {
		return CompressionType::COMPRESSION_UNCOMPRESSED;
	} else if (compression == "rle") {
		return CompressionType::COMPRESSION_RLE;
	} else if (compression == "dictionary") {
		return CompressionType::COMPRESSION_DICTIONARY;
	} else if (compression == "pfor") {
		return CompressionType::COMPRESSION_PFOR_DELTA;
	} else if (compression == "bitpacking") {
		return CompressionType::COMPRESSION_BITPACKING;
	} else if (compression == "fsst") {
		return CompressionType::COMPRESSION_FSST;
	} else if (compression == "chimp") {
		return CompressionType::COMPRESSION_CHIMP;
	} else if (compression == "patas") {
		return CompressionType::COMPRESSION_PATAS;
	} else {
		return CompressionType::COMPRESSION_AUTO;
	}
}

string CompressionTypeToString(CompressionType type) {
	switch (type) {
	case CompressionType::COMPRESSION_AUTO:
		return "Auto";
	case CompressionType::COMPRESSION_UNCOMPRESSED:
		return "Uncompressed";
	case CompressionType::COMPRESSION_CONSTANT:
		return "Constant";
	case CompressionType::COMPRESSION_RLE:
		return "RLE";
	case CompressionType::COMPRESSION_DICTIONARY:
		return "Dictionary";
	case CompressionType::COMPRESSION_PFOR_DELTA:
		return "PFOR";
	case CompressionType::COMPRESSION_BITPACKING:
		return "BitPacking";
	case CompressionType::COMPRESSION_FSST:
		return "FSST";
	case CompressionType::COMPRESSION_CHIMP:
		return "Chimp";
	case CompressionType::COMPRESSION_PATAS:
		return "Patas";
	default:
		throw InternalException("Unrecognized compression type!");
	}
}
// LCOV_EXCL_STOP

} // namespace duckdb



namespace duckdb {

bool TryGetDatePartSpecifier(const string &specifier_p, DatePartSpecifier &result) {
	auto specifier = StringUtil::Lower(specifier_p);
	if (specifier == "year" || specifier == "yr" || specifier == "y" || specifier == "years" || specifier == "yrs") {
		result = DatePartSpecifier::YEAR;
	} else if (specifier == "month" || specifier == "mon" || specifier == "months" || specifier == "mons") {
		result = DatePartSpecifier::MONTH;
	} else if (specifier == "day" || specifier == "days" || specifier == "d" || specifier == "dayofmonth") {
		result = DatePartSpecifier::DAY;
	} else if (specifier == "decade" || specifier == "dec" || specifier == "decades" || specifier == "decs") {
		result = DatePartSpecifier::DECADE;
	} else if (specifier == "century" || specifier == "cent" || specifier == "centuries" || specifier == "c") {
		result = DatePartSpecifier::CENTURY;
	} else if (specifier == "millennium" || specifier == "mil" || specifier == "millenniums" ||
	           specifier == "millennia" || specifier == "mils" || specifier == "millenium") {
		result = DatePartSpecifier::MILLENNIUM;
	} else if (specifier == "microseconds" || specifier == "microsecond" || specifier == "us" || specifier == "usec" ||
	           specifier == "usecs" || specifier == "usecond" || specifier == "useconds") {
		result = DatePartSpecifier::MICROSECONDS;
	} else if (specifier == "milliseconds" || specifier == "millisecond" || specifier == "ms" || specifier == "msec" ||
	           specifier == "msecs" || specifier == "msecond" || specifier == "mseconds") {
		result = DatePartSpecifier::MILLISECONDS;
	} else if (specifier == "second" || specifier == "sec" || specifier == "seconds" || specifier == "secs" ||
	           specifier == "s") {
		result = DatePartSpecifier::SECOND;
	} else if (specifier == "minute" || specifier == "min" || specifier == "minutes" || specifier == "mins" ||
	           specifier == "m") {
		result = DatePartSpecifier::MINUTE;
	} else if (specifier == "hour" || specifier == "hr" || specifier == "hours" || specifier == "hrs" ||
	           specifier == "h") {
		result = DatePartSpecifier::HOUR;
	} else if (specifier == "epoch") {
		// seconds since 1970-01-01
		result = DatePartSpecifier::EPOCH;
	} else if (specifier == "dow" || specifier == "dayofweek" || specifier == "weekday") {
		// day of the week (Sunday = 0, Saturday = 6)
		result = DatePartSpecifier::DOW;
	} else if (specifier == "isodow") {
		// isodow (Monday = 1, Sunday = 7)
		result = DatePartSpecifier::ISODOW;
	} else if (specifier == "week" || specifier == "weeks" || specifier == "w" || specifier == "weekofyear") {
		// ISO week number
		result = DatePartSpecifier::WEEK;
	} else if (specifier == "doy" || specifier == "dayofyear") {
		// day of the year (1-365/366)
		result = DatePartSpecifier::DOY;
	} else if (specifier == "quarter" || specifier == "quarters") {
		// quarter of the year (1-4)
		result = DatePartSpecifier::QUARTER;
	} else if (specifier == "yearweek") {
		// Combined isoyear and isoweek YYYYWW
		result = DatePartSpecifier::YEARWEEK;
	} else if (specifier == "isoyear") {
		// ISO year (first week of the year may be in previous year)
		result = DatePartSpecifier::ISOYEAR;
	} else if (specifier == "era") {
		result = DatePartSpecifier::ERA;
	} else if (specifier == "timezone") {
		result = DatePartSpecifier::TIMEZONE;
	} else if (specifier == "timezone_hour") {
		result = DatePartSpecifier::TIMEZONE_HOUR;
	} else if (specifier == "timezone_minute") {
		result = DatePartSpecifier::TIMEZONE_MINUTE;
	} else {
		return false;
	}
	return true;
}

DatePartSpecifier GetDatePartSpecifier(const string &specifier) {
	DatePartSpecifier result;
	if (!TryGetDatePartSpecifier(specifier, result)) {
		throw ConversionException("extract specifier \"%s\" not recognized", specifier);
	}
	return result;
}

} // namespace duckdb





namespace duckdb {

string ExpressionTypeToString(ExpressionType type) {
	switch (type) {
	case ExpressionType::OPERATOR_CAST:
		return "CAST";
	case ExpressionType::OPERATOR_NOT:
		return "NOT";
	case ExpressionType::OPERATOR_IS_NULL:
		return "IS_NULL";
	case ExpressionType::OPERATOR_IS_NOT_NULL:
		return "IS_NOT_NULL";
	case ExpressionType::COMPARE_EQUAL:
		return "EQUAL";
	case ExpressionType::COMPARE_NOTEQUAL:
		return "NOTEQUAL";
	case ExpressionType::COMPARE_LESSTHAN:
		return "LESSTHAN";
	case ExpressionType::COMPARE_GREATERTHAN:
		return "GREATERTHAN";
	case ExpressionType::COMPARE_LESSTHANOREQUALTO:
		return "LESSTHANOREQUALTO";
	case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
		return "GREATERTHANOREQUALTO";
	case ExpressionType::COMPARE_IN:
		return "IN";
	case ExpressionType::COMPARE_DISTINCT_FROM:
		return "DISTINCT_FROM";
	case ExpressionType::COMPARE_NOT_DISTINCT_FROM:
		return "NOT_DISTINCT_FROM";
	case ExpressionType::CONJUNCTION_AND:
		return "AND";
	case ExpressionType::CONJUNCTION_OR:
		return "OR";
	case ExpressionType::VALUE_CONSTANT:
		return "CONSTANT";
	case ExpressionType::VALUE_PARAMETER:
		return "PARAMETER";
	case ExpressionType::VALUE_TUPLE:
		return "TUPLE";
	case ExpressionType::VALUE_TUPLE_ADDRESS:
		return "TUPLE_ADDRESS";
	case ExpressionType::VALUE_NULL:
		return "NULL";
	case ExpressionType::VALUE_VECTOR:
		return "VECTOR";
	case ExpressionType::VALUE_SCALAR:
		return "SCALAR";
	case ExpressionType::AGGREGATE:
		return "AGGREGATE";
	case ExpressionType::WINDOW_AGGREGATE:
		return "WINDOW_AGGREGATE";
	case ExpressionType::WINDOW_RANK:
		return "RANK";
	case ExpressionType::WINDOW_RANK_DENSE:
		return "RANK_DENSE";
	case ExpressionType::WINDOW_PERCENT_RANK:
		return "PERCENT_RANK";
	case ExpressionType::WINDOW_ROW_NUMBER:
		return "ROW_NUMBER";
	case ExpressionType::WINDOW_FIRST_VALUE:
		return "FIRST_VALUE";
	case ExpressionType::WINDOW_LAST_VALUE:
		return "LAST_VALUE";
	case ExpressionType::WINDOW_NTH_VALUE:
		return "NTH_VALUE";
	case ExpressionType::WINDOW_CUME_DIST:
		return "CUME_DIST";
	case ExpressionType::WINDOW_LEAD:
		return "LEAD";
	case ExpressionType::WINDOW_LAG:
		return "LAG";
	case ExpressionType::WINDOW_NTILE:
		return "NTILE";
	case ExpressionType::FUNCTION:
		return "FUNCTION";
	case ExpressionType::CASE_EXPR:
		return "CASE";
	case ExpressionType::OPERATOR_NULLIF:
		return "NULLIF";
	case ExpressionType::OPERATOR_COALESCE:
		return "COALESCE";
	case ExpressionType::ARRAY_EXTRACT:
		return "ARRAY_EXTRACT";
	case ExpressionType::ARRAY_SLICE:
		return "ARRAY_SLICE";
	case ExpressionType::STRUCT_EXTRACT:
		return "STRUCT_EXTRACT";
	case ExpressionType::SUBQUERY:
		return "SUBQUERY";
	case ExpressionType::STAR:
		return "STAR";
	case ExpressionType::PLACEHOLDER:
		return "PLACEHOLDER";
	case ExpressionType::COLUMN_REF:
		return "COLUMN_REF";
	case ExpressionType::FUNCTION_REF:
		return "FUNCTION_REF";
	case ExpressionType::TABLE_REF:
		return "TABLE_REF";
	case ExpressionType::CAST:
		return "CAST";
	case ExpressionType::COMPARE_NOT_IN:
		return "COMPARE_NOT_IN";
	case ExpressionType::COMPARE_BETWEEN:
		return "COMPARE_BETWEEN";
	case ExpressionType::COMPARE_NOT_BETWEEN:
		return "COMPARE_NOT_BETWEEN";
	case ExpressionType::VALUE_DEFAULT:
		return "VALUE_DEFAULT";
	case ExpressionType::BOUND_REF:
		return "BOUND_REF";
	case ExpressionType::BOUND_COLUMN_REF:
		return "BOUND_COLUMN_REF";
	case ExpressionType::BOUND_FUNCTION:
		return "BOUND_FUNCTION";
	case ExpressionType::BOUND_AGGREGATE:
		return "BOUND_AGGREGATE";
	case ExpressionType::GROUPING_FUNCTION:
		return "GROUPING";
	case ExpressionType::ARRAY_CONSTRUCTOR:
		return "ARRAY_CONSTRUCTOR";
	case ExpressionType::TABLE_STAR:
		return "TABLE_STAR";
	case ExpressionType::BOUND_UNNEST:
		return "BOUND_UNNEST";
	case ExpressionType::COLLATE:
		return "COLLATE";
	case ExpressionType::POSITIONAL_REFERENCE:
		return "POSITIONAL_REFERENCE";
	case ExpressionType::BOUND_LAMBDA_REF:
		return "BOUND_LAMBDA_REF";
	case ExpressionType::LAMBDA:
		return "LAMBDA";
	case ExpressionType::ARROW:
		return "ARROW";
	case ExpressionType::INVALID:
		break;
	}
	return "INVALID";
}
string ExpressionClassToString(ExpressionClass type) {
	switch (type) {
	case ExpressionClass::INVALID:
		return "INVALID";
	case ExpressionClass::AGGREGATE:
		return "AGGREGATE";
	case ExpressionClass::CASE:
		return "CASE";
	case ExpressionClass::CAST:
		return "CAST";
	case ExpressionClass::COLUMN_REF:
		return "COLUMN_REF";
	case ExpressionClass::COMPARISON:
		return "COMPARISON";
	case ExpressionClass::CONJUNCTION:
		return "CONJUNCTION";
	case ExpressionClass::CONSTANT:
		return "CONSTANT";
	case ExpressionClass::DEFAULT:
		return "DEFAULT";
	case ExpressionClass::FUNCTION:
		return "FUNCTION";
	case ExpressionClass::OPERATOR:
		return "OPERATOR";
	case ExpressionClass::STAR:
		return "STAR";
	case ExpressionClass::SUBQUERY:
		return "SUBQUERY";
	case ExpressionClass::WINDOW:
		return "WINDOW";
	case ExpressionClass::PARAMETER:
		return "PARAMETER";
	case ExpressionClass::COLLATE:
		return "COLLATE";
	case ExpressionClass::LAMBDA:
		return "LAMBDA";
	case ExpressionClass::POSITIONAL_REFERENCE:
		return "POSITIONAL_REFERENCE";
	case ExpressionClass::BETWEEN:
		return "BETWEEN";
	case ExpressionClass::BOUND_AGGREGATE:
		return "BOUND_AGGREGATE";
	case ExpressionClass::BOUND_CASE:
		return "BOUND_CASE";
	case ExpressionClass::BOUND_CAST:
		return "BOUND_CAST";
	case ExpressionClass::BOUND_COLUMN_REF:
		return "BOUND_COLUMN_REF";
	case ExpressionClass::BOUND_COMPARISON:
		return "BOUND_COMPARISON";
	case ExpressionClass::BOUND_CONJUNCTION:
		return "BOUND_CONJUNCTION";
	case ExpressionClass::BOUND_CONSTANT:
		return "BOUND_CONSTANT";
	case ExpressionClass::BOUND_DEFAULT:
		return "BOUND_DEFAULT";
	case ExpressionClass::BOUND_FUNCTION:
		return "BOUND_FUNCTION";
	case ExpressionClass::BOUND_OPERATOR:
		return "BOUND_OPERATOR";
	case ExpressionClass::BOUND_PARAMETER:
		return "BOUND_PARAMETER";
	case ExpressionClass::BOUND_REF:
		return "BOUND_REF";
	case ExpressionClass::BOUND_SUBQUERY:
		return "BOUND_SUBQUERY";
	case ExpressionClass::BOUND_WINDOW:
		return "BOUND_WINDOW";
	case ExpressionClass::BOUND_BETWEEN:
		return "BOUND_BETWEEN";
	case ExpressionClass::BOUND_UNNEST:
		return "BOUND_UNNEST";
	case ExpressionClass::BOUND_LAMBDA:
		return "BOUND_LAMBDA";
	case ExpressionClass::BOUND_EXPRESSION:
		return "BOUND_EXPRESSION";
	default:
		return "ExpressionClass::!!UNIMPLEMENTED_CASE!!";
	}
}

string ExpressionTypeToOperator(ExpressionType type) {
	switch (type) {
	case ExpressionType::COMPARE_EQUAL:
		return "=";
	case ExpressionType::COMPARE_NOTEQUAL:
		return "!=";
	case ExpressionType::COMPARE_LESSTHAN:
		return "<";
	case ExpressionType::COMPARE_GREATERTHAN:
		return ">";
	case ExpressionType::COMPARE_LESSTHANOREQUALTO:
		return "<=";
	case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
		return ">=";
	case ExpressionType::COMPARE_DISTINCT_FROM:
		return "IS DISTINCT FROM";
	case ExpressionType::COMPARE_NOT_DISTINCT_FROM:
		return "IS NOT DISTINCT FROM";
	case ExpressionType::CONJUNCTION_AND:
		return "AND";
	case ExpressionType::CONJUNCTION_OR:
		return "OR";
	default:
		return "";
	}
}

ExpressionType NegateComparisonExpression(ExpressionType type) {
	ExpressionType negated_type = ExpressionType::INVALID;
	switch (type) {
	case ExpressionType::COMPARE_EQUAL:
		negated_type = ExpressionType::COMPARE_NOTEQUAL;
		break;
	case ExpressionType::COMPARE_NOTEQUAL:
		negated_type = ExpressionType::COMPARE_EQUAL;
		break;
	case ExpressionType::COMPARE_LESSTHAN:
		negated_type = ExpressionType::COMPARE_GREATERTHANOREQUALTO;
		break;
	case ExpressionType::COMPARE_GREATERTHAN:
		negated_type = ExpressionType::COMPARE_LESSTHANOREQUALTO;
		break;
	case ExpressionType::COMPARE_LESSTHANOREQUALTO:
		negated_type = ExpressionType::COMPARE_GREATERTHAN;
		break;
	case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
		negated_type = ExpressionType::COMPARE_LESSTHAN;
		break;
	default:
		throw InternalException("Unsupported comparison type in negation");
	}
	return negated_type;
}

ExpressionType FlipComparisonExpression(ExpressionType type) {
	ExpressionType flipped_type = ExpressionType::INVALID;
	switch (type) {
	case ExpressionType::COMPARE_NOT_DISTINCT_FROM:
	case ExpressionType::COMPARE_DISTINCT_FROM:
	case ExpressionType::COMPARE_NOTEQUAL:
	case ExpressionType::COMPARE_EQUAL:
		flipped_type = type;
		break;
	case ExpressionType::COMPARE_LESSTHAN:
		flipped_type = ExpressionType::COMPARE_GREATERTHAN;
		break;
	case ExpressionType::COMPARE_GREATERTHAN:
		flipped_type = ExpressionType::COMPARE_LESSTHAN;
		break;
	case ExpressionType::COMPARE_LESSTHANOREQUALTO:
		flipped_type = ExpressionType::COMPARE_GREATERTHANOREQUALTO;
		break;
	case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
		flipped_type = ExpressionType::COMPARE_LESSTHANOREQUALTO;
		break;
	default:
		throw InternalException("Unsupported comparison type in flip");
	}
	return flipped_type;
}

ExpressionType OperatorToExpressionType(const string &op) {
	if (op == "=" || op == "==") {
		return ExpressionType::COMPARE_EQUAL;
	} else if (op == "!=" || op == "<>") {
		return ExpressionType::COMPARE_NOTEQUAL;
	} else if (op == "<") {
		return ExpressionType::COMPARE_LESSTHAN;
	} else if (op == ">") {
		return ExpressionType::COMPARE_GREATERTHAN;
	} else if (op == "<=") {
		return ExpressionType::COMPARE_LESSTHANOREQUALTO;
	} else if (op == ">=") {
		return ExpressionType::COMPARE_GREATERTHANOREQUALTO;
	}
	return ExpressionType::INVALID;
}

} // namespace duckdb



namespace duckdb {

FileCompressionType FileCompressionTypeFromString(const string &input) {
	auto parameter = StringUtil::Lower(input);
	if (parameter == "infer" || parameter == "auto") {
		return FileCompressionType::AUTO_DETECT;
	} else if (parameter == "gzip") {
		return FileCompressionType::GZIP;
	} else if (parameter == "zstd") {
		return FileCompressionType::ZSTD;
	} else if (parameter == "uncompressed" || parameter == "none" || parameter.empty()) {
		return FileCompressionType::UNCOMPRESSED;
	} else {
		throw ParserException("Unrecognized file compression type \"%s\"", input);
	}
}

} // namespace duckdb



namespace duckdb {

bool IsLeftOuterJoin(JoinType type) {
	return type == JoinType::LEFT || type == JoinType::OUTER;
}

bool IsRightOuterJoin(JoinType type) {
	return type == JoinType::OUTER || type == JoinType::RIGHT;
}

// **DEPRECATED**: Use EnumUtil directly instead.
string JoinTypeToString(JoinType type) {
	return EnumUtil::ToString(type);
}

} // namespace duckdb


namespace duckdb {

//===--------------------------------------------------------------------===//
// Value <--> String Utilities
//===--------------------------------------------------------------------===//
// LCOV_EXCL_START
string LogicalOperatorToString(LogicalOperatorType type) {
	switch (type) {
	case LogicalOperatorType::LOGICAL_GET:
		return "GET";
	case LogicalOperatorType::LOGICAL_CHUNK_GET:
		return "CHUNK_GET";
	case LogicalOperatorType::LOGICAL_DELIM_GET:
		return "DELIM_GET";
	case LogicalOperatorType::LOGICAL_EMPTY_RESULT:
		return "EMPTY_RESULT";
	case LogicalOperatorType::LOGICAL_EXPRESSION_GET:
		return "EXPRESSION_GET";
	case LogicalOperatorType::LOGICAL_ANY_JOIN:
		return "ANY_JOIN";
	case LogicalOperatorType::LOGICAL_ASOF_JOIN:
		return "ASOF_JOIN";
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
		return "COMPARISON_JOIN";
	case LogicalOperatorType::LOGICAL_DELIM_JOIN:
		return "DELIM_JOIN";
	case LogicalOperatorType::LOGICAL_PROJECTION:
		return "PROJECTION";
	case LogicalOperatorType::LOGICAL_FILTER:
		return "FILTER";
	case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
		return "AGGREGATE";
	case LogicalOperatorType::LOGICAL_WINDOW:
		return "WINDOW";
	case LogicalOperatorType::LOGICAL_UNNEST:
		return "UNNEST";
	case LogicalOperatorType::LOGICAL_LIMIT:
		return "LIMIT";
	case LogicalOperatorType::LOGICAL_ORDER_BY:
		return "ORDER_BY";
	case LogicalOperatorType::LOGICAL_TOP_N:
		return "TOP_N";
	case LogicalOperatorType::LOGICAL_SAMPLE:
		return "SAMPLE";
	case LogicalOperatorType::LOGICAL_LIMIT_PERCENT:
		return "LIMIT_PERCENT";
	case LogicalOperatorType::LOGICAL_COPY_TO_FILE:
		return "COPY_TO_FILE";
	case LogicalOperatorType::LOGICAL_JOIN:
		return "JOIN";
	case LogicalOperatorType::LOGICAL_CROSS_PRODUCT:
		return "CROSS_PRODUCT";
	case LogicalOperatorType::LOGICAL_POSITIONAL_JOIN:
		return "POSITIONAL_JOIN";
	case LogicalOperatorType::LOGICAL_UNION:
		return "UNION";
	case LogicalOperatorType::LOGICAL_EXCEPT:
		return "EXCEPT";
	case LogicalOperatorType::LOGICAL_INTERSECT:
		return "INTERSECT";
	case LogicalOperatorType::LOGICAL_INSERT:
		return "INSERT";
	case LogicalOperatorType::LOGICAL_DISTINCT:
		return "DISTINCT";
	case LogicalOperatorType::LOGICAL_DELETE:
		return "DELETE";
	case LogicalOperatorType::LOGICAL_UPDATE:
		return "UPDATE";
	case LogicalOperatorType::LOGICAL_PREPARE:
		return "PREPARE";
	case LogicalOperatorType::LOGICAL_DUMMY_SCAN:
		return "DUMMY_SCAN";
	case LogicalOperatorType::LOGICAL_CREATE_INDEX:
		return "CREATE_INDEX";
	case LogicalOperatorType::LOGICAL_CREATE_TABLE:
		return "CREATE_TABLE";
	case LogicalOperatorType::LOGICAL_CREATE_MACRO:
		return "CREATE_MACRO";
	case LogicalOperatorType::LOGICAL_EXPLAIN:
		return "EXPLAIN";
	case LogicalOperatorType::LOGICAL_EXECUTE:
		return "EXECUTE";
	case LogicalOperatorType::LOGICAL_VACUUM:
		return "VACUUM";
	case LogicalOperatorType::LOGICAL_RECURSIVE_CTE:
		return "REC_CTE";
	case LogicalOperatorType::LOGICAL_CTE_REF:
		return "CTE_SCAN";
	case LogicalOperatorType::LOGICAL_SHOW:
		return "SHOW";
	case LogicalOperatorType::LOGICAL_ALTER:
		return "ALTER";
	case LogicalOperatorType::LOGICAL_CREATE_SEQUENCE:
		return "CREATE_SEQUENCE";
	case LogicalOperatorType::LOGICAL_CREATE_TYPE:
		return "CREATE_TYPE";
	case LogicalOperatorType::LOGICAL_CREATE_VIEW:
		return "CREATE_VIEW";
	case LogicalOperatorType::LOGICAL_CREATE_SCHEMA:
		return "CREATE_SCHEMA";
	case LogicalOperatorType::LOGICAL_ATTACH:
		return "ATTACH";
	case LogicalOperatorType::LOGICAL_DETACH:
		return "ATTACH";
	case LogicalOperatorType::LOGICAL_DROP:
		return "DROP";
	case LogicalOperatorType::LOGICAL_PRAGMA:
		return "PRAGMA";
	case LogicalOperatorType::LOGICAL_TRANSACTION:
		return "TRANSACTION";
	case LogicalOperatorType::LOGICAL_EXPORT:
		return "EXPORT";
	case LogicalOperatorType::LOGICAL_SET:
		return "SET";
	case LogicalOperatorType::LOGICAL_RESET:
		return "RESET";
	case LogicalOperatorType::LOGICAL_LOAD:
		return "LOAD";
	case LogicalOperatorType::LOGICAL_INVALID:
		break;
	case LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR:
		return "CUSTOM_OP";
	case LogicalOperatorType::LOGICAL_PIVOT:
		return "PIVOT";
	}
	return "INVALID";
}
// LCOV_EXCL_STOP

} // namespace duckdb





namespace duckdb {

struct DefaultOptimizerType {
	const char *name;
	OptimizerType type;
};

static DefaultOptimizerType internal_optimizer_types[] = {
    {"expression_rewriter", OptimizerType::EXPRESSION_REWRITER},
    {"filter_pullup", OptimizerType::FILTER_PULLUP},
    {"filter_pushdown", OptimizerType::FILTER_PUSHDOWN},
    {"regex_range", OptimizerType::REGEX_RANGE},
    {"in_clause", OptimizerType::IN_CLAUSE},
    {"join_order", OptimizerType::JOIN_ORDER},
    {"deliminator", OptimizerType::DELIMINATOR},
    {"unnest_rewriter", OptimizerType::UNNEST_REWRITER},
    {"unused_columns", OptimizerType::UNUSED_COLUMNS},
    {"statistics_propagation", OptimizerType::STATISTICS_PROPAGATION},
    {"common_subexpressions", OptimizerType::COMMON_SUBEXPRESSIONS},
    {"common_aggregate", OptimizerType::COMMON_AGGREGATE},
    {"column_lifetime", OptimizerType::COLUMN_LIFETIME},
    {"top_n", OptimizerType::TOP_N},
    {"reorder_filter", OptimizerType::REORDER_FILTER},
    {"extension", OptimizerType::EXTENSION},
    {nullptr, OptimizerType::INVALID}};

string OptimizerTypeToString(OptimizerType type) {
	for (idx_t i = 0; internal_optimizer_types[i].name; i++) {
		if (internal_optimizer_types[i].type == type) {
			return internal_optimizer_types[i].name;
		}
	}
	throw InternalException("Invalid optimizer type");
}

OptimizerType OptimizerTypeFromString(const string &str) {
	for (idx_t i = 0; internal_optimizer_types[i].name; i++) {
		if (internal_optimizer_types[i].name == str) {
			return internal_optimizer_types[i].type;
		}
	}
	// optimizer not found, construct candidate list
	vector<string> optimizer_names;
	for (idx_t i = 0; internal_optimizer_types[i].name; i++) {
		optimizer_names.emplace_back(internal_optimizer_types[i].name);
	}
	throw ParserException("Optimizer type \"%s\" not recognized\n%s", str,
	                      StringUtil::CandidatesErrorMessage(optimizer_names, str, "Candidate optimizers"));
}

} // namespace duckdb


namespace duckdb {

// LCOV_EXCL_START
string PhysicalOperatorToString(PhysicalOperatorType type) {
	switch (type) {
	case PhysicalOperatorType::TABLE_SCAN:
		return "TABLE_SCAN";
	case PhysicalOperatorType::DUMMY_SCAN:
		return "DUMMY_SCAN";
	case PhysicalOperatorType::CHUNK_SCAN:
		return "CHUNK_SCAN";
	case PhysicalOperatorType::COLUMN_DATA_SCAN:
		return "COLUMN_DATA_SCAN";
	case PhysicalOperatorType::DELIM_SCAN:
		return "DELIM_SCAN";
	case PhysicalOperatorType::ORDER_BY:
		return "ORDER_BY";
	case PhysicalOperatorType::LIMIT:
		return "LIMIT";
	case PhysicalOperatorType::LIMIT_PERCENT:
		return "LIMIT_PERCENT";
	case PhysicalOperatorType::STREAMING_LIMIT:
		return "STREAMING_LIMIT";
	case PhysicalOperatorType::RESERVOIR_SAMPLE:
		return "RESERVOIR_SAMPLE";
	case PhysicalOperatorType::STREAMING_SAMPLE:
		return "STREAMING_SAMPLE";
	case PhysicalOperatorType::TOP_N:
		return "TOP_N";
	case PhysicalOperatorType::WINDOW:
		return "WINDOW";
	case PhysicalOperatorType::STREAMING_WINDOW:
		return "STREAMING_WINDOW";
	case PhysicalOperatorType::UNNEST:
		return "UNNEST";
	case PhysicalOperatorType::UNGROUPED_AGGREGATE:
		return "UNGROUPED_AGGREGATE";
	case PhysicalOperatorType::HASH_GROUP_BY:
		return "HASH_GROUP_BY";
	case PhysicalOperatorType::PERFECT_HASH_GROUP_BY:
		return "PERFECT_HASH_GROUP_BY";
	case PhysicalOperatorType::FILTER:
		return "FILTER";
	case PhysicalOperatorType::PROJECTION:
		return "PROJECTION";
	case PhysicalOperatorType::COPY_TO_FILE:
		return "COPY_TO_FILE";
	case PhysicalOperatorType::BATCH_COPY_TO_FILE:
		return "BATCH_COPY_TO_FILE";
	case PhysicalOperatorType::FIXED_BATCH_COPY_TO_FILE:
		return "FIXED_BATCH_COPY_TO_FILE";
	case PhysicalOperatorType::DELIM_JOIN:
		return "DELIM_JOIN";
	case PhysicalOperatorType::BLOCKWISE_NL_JOIN:
		return "BLOCKWISE_NL_JOIN";
	case PhysicalOperatorType::NESTED_LOOP_JOIN:
		return "NESTED_LOOP_JOIN";
	case PhysicalOperatorType::HASH_JOIN:
		return "HASH_JOIN";
	case PhysicalOperatorType::INDEX_JOIN:
		return "INDEX_JOIN";
	case PhysicalOperatorType::PIECEWISE_MERGE_JOIN:
		return "PIECEWISE_MERGE_JOIN";
	case PhysicalOperatorType::IE_JOIN:
		return "IE_JOIN";
	case PhysicalOperatorType::ASOF_JOIN:
		return "ASOF_JOIN";
	case PhysicalOperatorType::CROSS_PRODUCT:
		return "CROSS_PRODUCT";
	case PhysicalOperatorType::POSITIONAL_JOIN:
		return "POSITIONAL_JOIN";
	case PhysicalOperatorType::POSITIONAL_SCAN:
		return "POSITIONAL_SCAN";
	case PhysicalOperatorType::UNION:
		return "UNION";
	case PhysicalOperatorType::INSERT:
		return "INSERT";
	case PhysicalOperatorType::BATCH_INSERT:
		return "BATCH_INSERT";
	case PhysicalOperatorType::DELETE_OPERATOR:
		return "DELETE";
	case PhysicalOperatorType::UPDATE:
		return "UPDATE";
	case PhysicalOperatorType::EMPTY_RESULT:
		return "EMPTY_RESULT";
	case PhysicalOperatorType::CREATE_TABLE:
		return "CREATE_TABLE";
	case PhysicalOperatorType::CREATE_TABLE_AS:
		return "CREATE_TABLE_AS";
	case PhysicalOperatorType::BATCH_CREATE_TABLE_AS:
		return "BATCH_CREATE_TABLE_AS";
	case PhysicalOperatorType::CREATE_INDEX:
		return "CREATE_INDEX";
	case PhysicalOperatorType::EXPLAIN:
		return "EXPLAIN";
	case PhysicalOperatorType::EXPLAIN_ANALYZE:
		return "EXPLAIN_ANALYZE";
	case PhysicalOperatorType::EXECUTE:
		return "EXECUTE";
	case PhysicalOperatorType::VACUUM:
		return "VACUUM";
	case PhysicalOperatorType::RECURSIVE_CTE:
		return "REC_CTE";
	case PhysicalOperatorType::RECURSIVE_CTE_SCAN:
		return "REC_CTE_SCAN";
	case PhysicalOperatorType::EXPRESSION_SCAN:
		return "EXPRESSION_SCAN";
	case PhysicalOperatorType::ALTER:
		return "ALTER";
	case PhysicalOperatorType::CREATE_SEQUENCE:
		return "CREATE_SEQUENCE";
	case PhysicalOperatorType::CREATE_VIEW:
		return "CREATE_VIEW";
	case PhysicalOperatorType::CREATE_SCHEMA:
		return "CREATE_SCHEMA";
	case PhysicalOperatorType::CREATE_MACRO:
		return "CREATE_MACRO";
	case PhysicalOperatorType::DROP:
		return "DROP";
	case PhysicalOperatorType::PRAGMA:
		return "PRAGMA";
	case PhysicalOperatorType::TRANSACTION:
		return "TRANSACTION";
	case PhysicalOperatorType::PREPARE:
		return "PREPARE";
	case PhysicalOperatorType::EXPORT:
		return "EXPORT";
	case PhysicalOperatorType::SET:
		return "SET";
	case PhysicalOperatorType::RESET:
		return "RESET";
	case PhysicalOperatorType::LOAD:
		return "LOAD";
	case PhysicalOperatorType::INOUT_FUNCTION:
		return "INOUT_FUNCTION";
	case PhysicalOperatorType::CREATE_TYPE:
		return "CREATE_TYPE";
	case PhysicalOperatorType::ATTACH:
		return "ATTACH";
	case PhysicalOperatorType::DETACH:
		return "DETACH";
	case PhysicalOperatorType::RESULT_COLLECTOR:
		return "RESULT_COLLECTOR";
	case PhysicalOperatorType::EXTENSION:
		return "EXTENSION";
	case PhysicalOperatorType::PIVOT:
		return "PIVOT";
	case PhysicalOperatorType::INVALID:
		break;
	}
	return "INVALID";
}
// LCOV_EXCL_STOP

} // namespace duckdb




namespace duckdb {

// LCOV_EXCL_START
string RelationTypeToString(RelationType type) {
	switch (type) {
	case RelationType::TABLE_RELATION:
		return "TABLE_RELATION";
	case RelationType::PROJECTION_RELATION:
		return "PROJECTION_RELATION";
	case RelationType::FILTER_RELATION:
		return "FILTER_RELATION";
	case RelationType::EXPLAIN_RELATION:
		return "EXPLAIN_RELATION";
	case RelationType::CROSS_PRODUCT_RELATION:
		return "CROSS_PRODUCT_RELATION";
	case RelationType::JOIN_RELATION:
		return "JOIN_RELATION";
	case RelationType::AGGREGATE_RELATION:
		return "AGGREGATE_RELATION";
	case RelationType::SET_OPERATION_RELATION:
		return "SET_OPERATION_RELATION";
	case RelationType::DISTINCT_RELATION:
		return "DISTINCT_RELATION";
	case RelationType::LIMIT_RELATION:
		return "LIMIT_RELATION";
	case RelationType::ORDER_RELATION:
		return "ORDER_RELATION";
	case RelationType::CREATE_VIEW_RELATION:
		return "CREATE_VIEW_RELATION";
	case RelationType::CREATE_TABLE_RELATION:
		return "CREATE_TABLE_RELATION";
	case RelationType::INSERT_RELATION:
		return "INSERT_RELATION";
	case RelationType::VALUE_LIST_RELATION:
		return "VALUE_LIST_RELATION";
	case RelationType::DELETE_RELATION:
		return "DELETE_RELATION";
	case RelationType::UPDATE_RELATION:
		return "UPDATE_RELATION";
	case RelationType::WRITE_CSV_RELATION:
		return "WRITE_CSV_RELATION";
	case RelationType::WRITE_PARQUET_RELATION:
		return "WRITE_PARQUET_RELATION";
	case RelationType::READ_CSV_RELATION:
		return "READ_CSV_RELATION";
	case RelationType::SUBQUERY_RELATION:
		return "SUBQUERY_RELATION";
	case RelationType::TABLE_FUNCTION_RELATION:
		return "TABLE_FUNCTION_RELATION";
	case RelationType::VIEW_RELATION:
		return "VIEW_RELATION";
	case RelationType::QUERY_RELATION:
		return "QUERY_RELATION";
	case RelationType::INVALID_RELATION:
		break;
	}
	return "INVALID_RELATION";
}
// LCOV_EXCL_STOP

} // namespace duckdb


namespace duckdb {

// LCOV_EXCL_START
string StatementTypeToString(StatementType type) {
	switch (type) {
	case StatementType::SELECT_STATEMENT:
		return "SELECT";
	case StatementType::INSERT_STATEMENT:
		return "INSERT";
	case StatementType::UPDATE_STATEMENT:
		return "UPDATE";
	case StatementType::DELETE_STATEMENT:
		return "DELETE";
	case StatementType::PREPARE_STATEMENT:
		return "PREPARE";
	case StatementType::EXECUTE_STATEMENT:
		return "EXECUTE";
	case StatementType::ALTER_STATEMENT:
		return "ALTER";
	case StatementType::TRANSACTION_STATEMENT:
		return "TRANSACTION";
	case StatementType::COPY_STATEMENT:
		return "COPY";
	case StatementType::ANALYZE_STATEMENT:
		return "ANALYZE";
	case StatementType::VARIABLE_SET_STATEMENT:
		return "VARIABLE_SET";
	case StatementType::CREATE_FUNC_STATEMENT:
		return "CREATE_FUNC";
	case StatementType::EXPLAIN_STATEMENT:
		return "EXPLAIN";
	case StatementType::CREATE_STATEMENT:
		return "CREATE";
	case StatementType::DROP_STATEMENT:
		return "DROP";
	case StatementType::PRAGMA_STATEMENT:
		return "PRAGMA";
	case StatementType::SHOW_STATEMENT:
		return "SHOW";
	case StatementType::VACUUM_STATEMENT:
		return "VACUUM";
	case StatementType::RELATION_STATEMENT:
		return "RELATION";
	case StatementType::EXPORT_STATEMENT:
		return "EXPORT";
	case StatementType::CALL_STATEMENT:
		return "CALL";
	case StatementType::SET_STATEMENT:
		return "SET";
	case StatementType::LOAD_STATEMENT:
		return "LOAD";
	case StatementType::EXTENSION_STATEMENT:
		return "EXTENSION";
	case StatementType::LOGICAL_PLAN_STATEMENT:
		return "LOGICAL_PLAN";
	case StatementType::ATTACH_STATEMENT:
		return "ATTACH";
	case StatementType::DETACH_STATEMENT:
		return "DETACH";
	case StatementType::MULTI_STATEMENT:
		return "MULTI";
	case StatementType::INVALID_STATEMENT:
		break;
	}
	return "INVALID";
}

string StatementReturnTypeToString(StatementReturnType type) {
	switch (type) {
	case StatementReturnType::QUERY_RESULT:
		return "QUERY_RESULT";
	case StatementReturnType::CHANGED_ROWS:
		return "CHANGED_ROWS";
	case StatementReturnType::NOTHING:
		return "NOTHING";
	}
	return "INVALID";
}
// LCOV_EXCL_STOP

} // namespace duckdb






#ifdef DUCKDB_CRASH_ON_ASSERT

#include <stdio.h>
#include <stdlib.h>
#endif
#ifdef DUCKDB_DEBUG_STACKTRACE
#include <execinfo.h>
#endif

namespace duckdb {

Exception::Exception(const string &msg) : std::exception(), type(ExceptionType::INVALID), raw_message_(msg) {
	exception_message_ = msg;
}

Exception::Exception(ExceptionType exception_type, const string &message)
    : std::exception(), type(exception_type), raw_message_(message) {
	exception_message_ = ExceptionTypeToString(exception_type) + " Error: " + message;
}

const char *Exception::what() const noexcept {
	return exception_message_.c_str();
}

const string &Exception::RawMessage() const {
	return raw_message_;
}

bool Exception::UncaughtException() {
#if __cplusplus >= 201703L
	return std::uncaught_exceptions() > 0;
#else
	return std::uncaught_exception();
#endif
}

string Exception::GetStackTrace(int max_depth) {
#ifdef DUCKDB_DEBUG_STACKTRACE
	string result;
	auto callstack = unique_ptr<void *[]>(new void *[max_depth]);
	int frames = backtrace(callstack.get(), max_depth);
	char **strs = backtrace_symbols(callstack.get(), frames);
	for (int i = 0; i < frames; i++) {
		result += strs[i];
		result += "\n";
	}
	free(strs);
	return "\n" + result;
#else
	// Stack trace not available. Toggle DUCKDB_DEBUG_STACKTRACE in exception.cpp to enable stack traces.
	return "";
#endif
}

string Exception::ConstructMessageRecursive(const string &msg, std::vector<ExceptionFormatValue> &values) {
#ifdef DEBUG
	// Verify that we have the required amount of values for the message
	idx_t parameter_count = 0;
	for (idx_t i = 0; i < msg.size(); i++) {
		if (msg[i] != '%') {
			continue;
		}
		if (i < msg.size() && msg[i + 1] == '%') {
			i++;
			continue;
		}
		parameter_count++;
	}
	if (parameter_count != values.size()) {
		throw InternalException("Expected %d parameters, received %d", parameter_count, values.size());
	}
#endif
	return ExceptionFormatValue::Format(msg, values);
}

string Exception::ExceptionTypeToString(ExceptionType type) {
	switch (type) {
	case ExceptionType::INVALID:
		return "Invalid";
	case ExceptionType::OUT_OF_RANGE:
		return "Out of Range";
	case ExceptionType::CONVERSION:
		return "Conversion";
	case ExceptionType::UNKNOWN_TYPE:
		return "Unknown Type";
	case ExceptionType::DECIMAL:
		return "Decimal";
	case ExceptionType::MISMATCH_TYPE:
		return "Mismatch Type";
	case ExceptionType::DIVIDE_BY_ZERO:
		return "Divide by Zero";
	case ExceptionType::OBJECT_SIZE:
		return "Object Size";
	case ExceptionType::INVALID_TYPE:
		return "Invalid type";
	case ExceptionType::SERIALIZATION:
		return "Serialization";
	case ExceptionType::TRANSACTION:
		return "TransactionContext";
	case ExceptionType::NOT_IMPLEMENTED:
		return "Not implemented";
	case ExceptionType::EXPRESSION:
		return "Expression";
	case ExceptionType::CATALOG:
		return "Catalog";
	case ExceptionType::PARSER:
		return "Parser";
	case ExceptionType::BINDER:
		return "Binder";
	case ExceptionType::PLANNER:
		return "Planner";
	case ExceptionType::SCHEDULER:
		return "Scheduler";
	case ExceptionType::EXECUTOR:
		return "Executor";
	case ExceptionType::CONSTRAINT:
		return "Constraint";
	case ExceptionType::INDEX:
		return "Index";
	case ExceptionType::STAT:
		return "Stat";
	case ExceptionType::CONNECTION:
		return "Connection";
	case ExceptionType::SYNTAX:
		return "Syntax";
	case ExceptionType::SETTINGS:
		return "Settings";
	case ExceptionType::OPTIMIZER:
		return "Optimizer";
	case ExceptionType::NULL_POINTER:
		return "NullPointer";
	case ExceptionType::IO:
		return "IO";
	case ExceptionType::INTERRUPT:
		return "INTERRUPT";
	case ExceptionType::FATAL:
		return "FATAL";
	case ExceptionType::INTERNAL:
		return "INTERNAL";
	case ExceptionType::INVALID_INPUT:
		return "Invalid Input";
	case ExceptionType::OUT_OF_MEMORY:
		return "Out of Memory";
	case ExceptionType::PERMISSION:
		return "Permission";
	case ExceptionType::PARAMETER_NOT_RESOLVED:
		return "Parameter Not Resolved";
	case ExceptionType::PARAMETER_NOT_ALLOWED:
		return "Parameter Not Allowed";
	case ExceptionType::DEPENDENCY:
		return "Dependency";
	case ExceptionType::HTTP:
		return "HTTP";
	default:
		return "Unknown";
	}
}

const HTTPException &Exception::AsHTTPException() const {
	D_ASSERT(type == ExceptionType::HTTP);
	const auto &e = static_cast<const HTTPException *>(this);
	D_ASSERT(e->GetStatusCode() != 0);
	D_ASSERT(e->GetHeaders().size() > 0);
	return *e;
}

void Exception::ThrowAsTypeWithMessage(ExceptionType type, const string &message,
                                       const std::shared_ptr<Exception> &original) {
	switch (type) {
	case ExceptionType::OUT_OF_RANGE:
		throw OutOfRangeException(message);
	case ExceptionType::CONVERSION:
		throw ConversionException(message); // FIXME: make a separation between Conversion/Cast exception?
	case ExceptionType::INVALID_TYPE:
		throw InvalidTypeException(message);
	case ExceptionType::MISMATCH_TYPE:
		throw TypeMismatchException(message);
	case ExceptionType::TRANSACTION:
		throw TransactionException(message);
	case ExceptionType::NOT_IMPLEMENTED:
		throw NotImplementedException(message);
	case ExceptionType::CATALOG:
		throw CatalogException(message);
	case ExceptionType::CONNECTION:
		throw ConnectionException(message);
	case ExceptionType::PARSER:
		throw ParserException(message);
	case ExceptionType::PERMISSION:
		throw PermissionException(message);
	case ExceptionType::SYNTAX:
		throw SyntaxException(message);
	case ExceptionType::CONSTRAINT:
		throw ConstraintException(message);
	case ExceptionType::BINDER:
		throw BinderException(message);
	case ExceptionType::IO:
		throw IOException(message);
	case ExceptionType::SERIALIZATION:
		throw SerializationException(message);
	case ExceptionType::INTERRUPT:
		throw InterruptException();
	case ExceptionType::INTERNAL:
		throw InternalException(message);
	case ExceptionType::INVALID_INPUT:
		throw InvalidInputException(message);
	case ExceptionType::OUT_OF_MEMORY:
		throw OutOfMemoryException(message);
	case ExceptionType::PARAMETER_NOT_ALLOWED:
		throw ParameterNotAllowedException(message);
	case ExceptionType::PARAMETER_NOT_RESOLVED:
		throw ParameterNotResolvedException();
	case ExceptionType::FATAL:
		throw FatalException(message);
	case ExceptionType::DEPENDENCY:
		throw DependencyException(message);
	case ExceptionType::HTTP: {
		original->AsHTTPException().Throw();
	}
	default:
		throw Exception(type, message);
	}
}

StandardException::StandardException(ExceptionType exception_type, const string &message)
    : Exception(exception_type, message) {
}

CastException::CastException(const PhysicalType orig_type, const PhysicalType new_type)
    : Exception(ExceptionType::CONVERSION,
                "Type " + TypeIdToString(orig_type) + " can't be cast as " + TypeIdToString(new_type)) {
}

CastException::CastException(const LogicalType &orig_type, const LogicalType &new_type)
    : Exception(ExceptionType::CONVERSION,
                "Type " + orig_type.ToString() + " can't be cast as " + new_type.ToString()) {
}

CastException::CastException(const string &msg) : Exception(ExceptionType::CONVERSION, msg) {
}

ValueOutOfRangeException::ValueOutOfRangeException(const int64_t value, const PhysicalType orig_type,
                                                   const PhysicalType new_type)
    : Exception(ExceptionType::CONVERSION, "Type " + TypeIdToString(orig_type) + " with value " +
                                               to_string((intmax_t)value) +
                                               " can't be cast because the value is out of range "
                                               "for the destination type " +
                                               TypeIdToString(new_type)) {
}

ValueOutOfRangeException::ValueOutOfRangeException(const double value, const PhysicalType orig_type,
                                                   const PhysicalType new_type)
    : Exception(ExceptionType::CONVERSION, "Type " + TypeIdToString(orig_type) + " with value " + to_string(value) +
                                               " can't be cast because the value is out of range "
                                               "for the destination type " +
                                               TypeIdToString(new_type)) {
}

ValueOutOfRangeException::ValueOutOfRangeException(const hugeint_t value, const PhysicalType orig_type,
                                                   const PhysicalType new_type)
    : Exception(ExceptionType::CONVERSION, "Type " + TypeIdToString(orig_type) + " with value " + value.ToString() +
                                               " can't be cast because the value is out of range "
                                               "for the destination type " +
                                               TypeIdToString(new_type)) {
}

ValueOutOfRangeException::ValueOutOfRangeException(const PhysicalType var_type, const idx_t length)
    : Exception(ExceptionType::OUT_OF_RANGE,
                "The value is too long to fit into type " + TypeIdToString(var_type) + "(" + to_string(length) + ")") {
}

ValueOutOfRangeException::ValueOutOfRangeException(const string &msg) : Exception(ExceptionType::OUT_OF_RANGE, msg) {
}

ConversionException::ConversionException(const string &msg) : Exception(ExceptionType::CONVERSION, msg) {
}

InvalidTypeException::InvalidTypeException(PhysicalType type, const string &msg)
    : Exception(ExceptionType::INVALID_TYPE, "Invalid Type [" + TypeIdToString(type) + "]: " + msg) {
}

InvalidTypeException::InvalidTypeException(const LogicalType &type, const string &msg)
    : Exception(ExceptionType::INVALID_TYPE, "Invalid Type [" + type.ToString() + "]: " + msg) {
}

InvalidTypeException::InvalidTypeException(const string &msg) : Exception(ExceptionType::INVALID_TYPE, msg) {
}

TypeMismatchException::TypeMismatchException(const PhysicalType type_1, const PhysicalType type_2, const string &msg)
    : Exception(ExceptionType::MISMATCH_TYPE,
                "Type " + TypeIdToString(type_1) + " does not match with " + TypeIdToString(type_2) + ". " + msg) {
}

TypeMismatchException::TypeMismatchException(const LogicalType &type_1, const LogicalType &type_2, const string &msg)
    : Exception(ExceptionType::MISMATCH_TYPE,
                "Type " + type_1.ToString() + " does not match with " + type_2.ToString() + ". " + msg) {
}

TypeMismatchException::TypeMismatchException(const string &msg) : Exception(ExceptionType::MISMATCH_TYPE, msg) {
}

TransactionException::TransactionException(const string &msg) : Exception(ExceptionType::TRANSACTION, msg) {
}

NotImplementedException::NotImplementedException(const string &msg) : Exception(ExceptionType::NOT_IMPLEMENTED, msg) {
}

OutOfRangeException::OutOfRangeException(const string &msg) : Exception(ExceptionType::OUT_OF_RANGE, msg) {
}

CatalogException::CatalogException(const string &msg) : StandardException(ExceptionType::CATALOG, msg) {
}

ConnectionException::ConnectionException(const string &msg) : StandardException(ExceptionType::CONNECTION, msg) {
}

ParserException::ParserException(const string &msg) : StandardException(ExceptionType::PARSER, msg) {
}

PermissionException::PermissionException(const string &msg) : StandardException(ExceptionType::PERMISSION, msg) {
}

SyntaxException::SyntaxException(const string &msg) : Exception(ExceptionType::SYNTAX, msg) {
}

ConstraintException::ConstraintException(const string &msg) : Exception(ExceptionType::CONSTRAINT, msg) {
}

DependencyException::DependencyException(const string &msg) : Exception(ExceptionType::DEPENDENCY, msg) {
}

BinderException::BinderException(const string &msg) : StandardException(ExceptionType::BINDER, msg) {
}

IOException::IOException(const string &msg) : Exception(ExceptionType::IO, msg) {
}

MissingExtensionException::MissingExtensionException(const string &msg)
    : Exception(ExceptionType::MISSING_EXTENSION, msg) {
}

SerializationException::SerializationException(const string &msg) : Exception(ExceptionType::SERIALIZATION, msg) {
}

SequenceException::SequenceException(const string &msg) : Exception(ExceptionType::SERIALIZATION, msg) {
}

InterruptException::InterruptException() : Exception(ExceptionType::INTERRUPT, "Interrupted!") {
}

FatalException::FatalException(ExceptionType type, const string &msg) : Exception(type, msg) {
}

InternalException::InternalException(const string &msg) : FatalException(ExceptionType::INTERNAL, msg) {
#ifdef DUCKDB_CRASH_ON_ASSERT
	Printer::Print("ABORT THROWN BY INTERNAL EXCEPTION: " + msg);
	abort();
#endif
}

InvalidInputException::InvalidInputException(const string &msg) : Exception(ExceptionType::INVALID_INPUT, msg) {
}

OutOfMemoryException::OutOfMemoryException(const string &msg) : Exception(ExceptionType::OUT_OF_MEMORY, msg) {
}

ParameterNotAllowedException::ParameterNotAllowedException(const string &msg)
    : StandardException(ExceptionType::PARAMETER_NOT_ALLOWED, msg) {
}

ParameterNotResolvedException::ParameterNotResolvedException()
    : Exception(ExceptionType::PARAMETER_NOT_RESOLVED, "Parameter types could not be resolved") {
}

} // namespace duckdb







namespace duckdb {

ExceptionFormatValue::ExceptionFormatValue(double dbl_val)
    : type(ExceptionFormatValueType::FORMAT_VALUE_TYPE_DOUBLE), dbl_val(dbl_val) {
}
ExceptionFormatValue::ExceptionFormatValue(int64_t int_val)
    : type(ExceptionFormatValueType::FORMAT_VALUE_TYPE_INTEGER), int_val(int_val) {
}
ExceptionFormatValue::ExceptionFormatValue(hugeint_t huge_val)
    : type(ExceptionFormatValueType::FORMAT_VALUE_TYPE_STRING), str_val(Hugeint::ToString(huge_val)) {
}
ExceptionFormatValue::ExceptionFormatValue(string str_val)
    : type(ExceptionFormatValueType::FORMAT_VALUE_TYPE_STRING), str_val(std::move(str_val)) {
}

template <>
ExceptionFormatValue ExceptionFormatValue::CreateFormatValue(PhysicalType value) {
	return ExceptionFormatValue(TypeIdToString(value));
}
template <>
ExceptionFormatValue
ExceptionFormatValue::CreateFormatValue(LogicalType value) { // NOLINT: templating requires us to copy value here
	return ExceptionFormatValue(value.ToString());
}
template <>
ExceptionFormatValue ExceptionFormatValue::CreateFormatValue(float value) {
	return ExceptionFormatValue(double(value));
}
template <>
ExceptionFormatValue ExceptionFormatValue::CreateFormatValue(double value) {
	return ExceptionFormatValue(double(value));
}
template <>
ExceptionFormatValue ExceptionFormatValue::CreateFormatValue(string value) {
	return ExceptionFormatValue(std::move(value));
}

template <>
ExceptionFormatValue
ExceptionFormatValue::CreateFormatValue(SQLString value) { // NOLINT: templating requires us to copy value here
	return KeywordHelper::WriteQuoted(value.raw_string, '\'');
}

template <>
ExceptionFormatValue
ExceptionFormatValue::CreateFormatValue(SQLIdentifier value) { // NOLINT: templating requires us to copy value here
	return KeywordHelper::WriteOptionallyQuoted(value.raw_string, '"');
}

template <>
ExceptionFormatValue ExceptionFormatValue::CreateFormatValue(const char *value) {
	return ExceptionFormatValue(string(value));
}
template <>
ExceptionFormatValue ExceptionFormatValue::CreateFormatValue(char *value) {
	return ExceptionFormatValue(string(value));
}
template <>
ExceptionFormatValue ExceptionFormatValue::CreateFormatValue(hugeint_t value) {
	return ExceptionFormatValue(value);
}

string ExceptionFormatValue::Format(const string &msg, std::vector<ExceptionFormatValue> &values) {
	std::vector<duckdb_fmt::basic_format_arg<duckdb_fmt::printf_context>> format_args;
	for (auto &val : values) {
		switch (val.type) {
		case ExceptionFormatValueType::FORMAT_VALUE_TYPE_DOUBLE:
			format_args.push_back(duckdb_fmt::internal::make_arg<duckdb_fmt::printf_context>(val.dbl_val));
			break;
		case ExceptionFormatValueType::FORMAT_VALUE_TYPE_INTEGER:
			format_args.push_back(duckdb_fmt::internal::make_arg<duckdb_fmt::printf_context>(val.int_val));
			break;
		case ExceptionFormatValueType::FORMAT_VALUE_TYPE_STRING:
			format_args.push_back(duckdb_fmt::internal::make_arg<duckdb_fmt::printf_context>(val.str_val));
			break;
		}
	}
	return duckdb_fmt::vsprintf(msg, duckdb_fmt::basic_format_args<duckdb_fmt::printf_context>(
	                                     format_args.data(), static_cast<int>(format_args.size())));
}

} // namespace duckdb


namespace duckdb {

//===--------------------------------------------------------------------===//
// Field Writer
//===--------------------------------------------------------------------===//
FieldWriter::FieldWriter(Serializer &serializer_p)
    : serializer(serializer_p), buffer(make_uniq<BufferedSerializer>()), field_count(0), finalized(false) {
	buffer->SetVersion(serializer.GetVersion());
	buffer->is_query_plan = serializer.is_query_plan;
}

FieldWriter::~FieldWriter() {
	if (Exception::UncaughtException()) {
		return;
	}
	D_ASSERT(finalized);
	// finalize should always have been called, unless this is destroyed as part of stack unwinding
	D_ASSERT(!buffer);
}

void FieldWriter::WriteData(const_data_ptr_t buffer_ptr, idx_t write_size) {
	D_ASSERT(buffer);
	buffer->WriteData(buffer_ptr, write_size);
}

template <>
void FieldWriter::Write(const string &val) {
	Write<uint32_t>((uint32_t)val.size());
	if (!val.empty()) {
		WriteData((const_data_ptr_t)val.c_str(), val.size());
	}
}

void FieldWriter::Finalize() {
	D_ASSERT(buffer);
	D_ASSERT(!finalized);
	finalized = true;
	serializer.Write<uint32_t>(field_count);
	serializer.Write<uint64_t>(buffer->blob.size);
	serializer.WriteData(buffer->blob.data.get(), buffer->blob.size);

	buffer.reset();
}

//===--------------------------------------------------------------------===//
// Field Deserializer
//===--------------------------------------------------------------------===//
FieldDeserializer::FieldDeserializer(Deserializer &root) : root(root), remaining_data(idx_t(-1)) {
	SetVersion(root.GetVersion());
}

void FieldDeserializer::ReadData(data_ptr_t buffer, idx_t read_size) {
	D_ASSERT(remaining_data != idx_t(-1));
	D_ASSERT(read_size <= remaining_data);
	root.ReadData(buffer, read_size);
	remaining_data -= read_size;
}

idx_t FieldDeserializer::RemainingData() {
	return remaining_data;
}

void FieldDeserializer::SetRemainingData(idx_t remaining_data) {
	this->remaining_data = remaining_data;
}

//===--------------------------------------------------------------------===//
// Field Reader
//===--------------------------------------------------------------------===//
FieldReader::FieldReader(Deserializer &source_p) : source(source_p), field_count(0), finalized(false) {
	max_field_count = source_p.Read<uint32_t>();
	total_size = source_p.Read<uint64_t>();
	D_ASSERT(max_field_count > 0);
	source.SetRemainingData(total_size);
}

FieldReader::~FieldReader() {
	if (Exception::UncaughtException()) {
		return;
	}
	D_ASSERT(finalized);
}

void FieldReader::Finalize() {
	D_ASSERT(!finalized);
	finalized = true;
	if (field_count < max_field_count) {
		// we can handle this case by calling source.ReadData(buffer, source.RemainingData())
		throw SerializationException("Not all fields were read. This file might have been written with a newer version "
		                             "of DuckDB and is incompatible with this version of DuckDB.");
	}
	D_ASSERT(source.RemainingData() == 0);
}

} // namespace duckdb








#include <cstring>

namespace duckdb {

FileBuffer::FileBuffer(Allocator &allocator, FileBufferType type, uint64_t user_size)
    : allocator(allocator), type(type) {
	Init();
	if (user_size) {
		Resize(user_size);
	}
}

void FileBuffer::Init() {
	buffer = nullptr;
	size = 0;
	internal_buffer = nullptr;
	internal_size = 0;
}

FileBuffer::FileBuffer(FileBuffer &source, FileBufferType type_p) : allocator(source.allocator), type(type_p) {
	// take over the structures of the source buffer
	buffer = source.buffer;
	size = source.size;
	internal_buffer = source.internal_buffer;
	internal_size = source.internal_size;

	source.Init();
}

FileBuffer::~FileBuffer() {
	if (!internal_buffer) {
		return;
	}
	allocator.FreeData(internal_buffer, internal_size);
}

void FileBuffer::ReallocBuffer(size_t new_size) {
	data_ptr_t new_buffer;
	if (internal_buffer) {
		new_buffer = allocator.ReallocateData(internal_buffer, internal_size, new_size);
	} else {
		new_buffer = allocator.AllocateData(new_size);
	}
	if (!new_buffer) {
		throw std::bad_alloc();
	}
	internal_buffer = new_buffer;
	internal_size = new_size;
	// Caller must update these.
	buffer = nullptr;
	size = 0;
}

FileBuffer::MemoryRequirement FileBuffer::CalculateMemory(uint64_t user_size) {
	FileBuffer::MemoryRequirement result;

	if (type == FileBufferType::TINY_BUFFER) {
		// We never do IO on tiny buffers, so there's no need to add a header or sector-align.
		result.header_size = 0;
		result.alloc_size = user_size;
	} else {
		result.header_size = Storage::BLOCK_HEADER_SIZE;
		result.alloc_size = AlignValue<uint32_t, Storage::SECTOR_SIZE>(result.header_size + user_size);
	}
	return result;
}

void FileBuffer::Resize(uint64_t new_size) {
	auto req = CalculateMemory(new_size);
	ReallocBuffer(req.alloc_size);

	if (new_size > 0) {
		buffer = internal_buffer + req.header_size;
		size = internal_size - req.header_size;
	}
}

void FileBuffer::Read(FileHandle &handle, uint64_t location) {
	D_ASSERT(type != FileBufferType::TINY_BUFFER);
	handle.Read(internal_buffer, internal_size, location);
}

void FileBuffer::Write(FileHandle &handle, uint64_t location) {
	D_ASSERT(type != FileBufferType::TINY_BUFFER);
	handle.Write(internal_buffer, internal_size, location);
}

void FileBuffer::Clear() {
	memset(internal_buffer, 0, internal_size);
}

void FileBuffer::Initialize(DebugInitialize initialize) {
	if (initialize == DebugInitialize::NO_INITIALIZE) {
		return;
	}
	uint8_t value = initialize == DebugInitialize::DEBUG_ZERO_INITIALIZE ? 0 : 0xFF;
	memset(internal_buffer, value, internal_size);
}

} // namespace duckdb















#include <cstdint>
#include <cstdio>

#ifndef _WIN32
#include <dirent.h>
#include <fcntl.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#else
#include <string>
#include <sysinfoapi.h>

#ifdef __MINGW32__
// need to manually define this for mingw
extern "C" WINBASEAPI BOOL WINAPI GetPhysicallyInstalledSystemMemory(PULONGLONG);
#endif

#undef FILE_CREATE // woo mingw
#endif

namespace duckdb {

FileSystem::~FileSystem() {
}

FileSystem &FileSystem::GetFileSystem(ClientContext &context) {
	auto &client_data = ClientData::Get(context);
	return *client_data.client_file_system;
}

bool PathMatched(const string &path, const string &sub_path) {
	if (path.rfind(sub_path, 0) == 0) {
		return true;
	}
	return false;
}

#ifndef _WIN32

string FileSystem::GetEnvVariable(const string &name) {
	const char *env = getenv(name.c_str());
	if (!env) {
		return string();
	}
	return env;
}

bool FileSystem::IsPathAbsolute(const string &path) {
	auto path_separator = FileSystem::PathSeparator();
	return PathMatched(path, path_separator);
}

string FileSystem::PathSeparator() {
	return "/";
}

void FileSystem::SetWorkingDirectory(const string &path) {
	if (chdir(path.c_str()) != 0) {
		throw IOException("Could not change working directory!");
	}
}

idx_t FileSystem::GetAvailableMemory() {
	errno = 0;
	idx_t max_memory = MinValue<idx_t>((idx_t)sysconf(_SC_PHYS_PAGES) * (idx_t)sysconf(_SC_PAGESIZE), UINTPTR_MAX);
	if (errno != 0) {
		return DConstants::INVALID_INDEX;
	}
	return max_memory;
}

string FileSystem::GetWorkingDirectory() {
	auto buffer = make_unsafe_uniq_array<char>(PATH_MAX);
	char *ret = getcwd(buffer.get(), PATH_MAX);
	if (!ret) {
		throw IOException("Could not get working directory!");
	}
	return string(buffer.get());
}

string FileSystem::NormalizeAbsolutePath(const string &path) {
	D_ASSERT(IsPathAbsolute(path));
	return path;
}

#else

string FileSystem::GetEnvVariable(const string &env) {
	// first convert the environment variable name to the correct encoding
	auto env_w = WindowsUtil::UTF8ToUnicode(env.c_str());
	// use _wgetenv to get the value
	auto res_w = _wgetenv(env_w.c_str());
	if (!res_w) {
		// no environment variable of this name found
		return string();
	}
	return WindowsUtil::UnicodeToUTF8(res_w);
}

bool FileSystem::IsPathAbsolute(const string &path) {
	// 1) A single backslash or forward-slash
	if (PathMatched(path, "\\") || PathMatched(path, "/")) {
		return true;
	}
	// 2) A disk designator with a backslash (e.g., C:\ or C:/)
	auto path_aux = path;
	path_aux.erase(0, 1);
	if (PathMatched(path_aux, ":\\") || PathMatched(path_aux, ":/")) {
		return true;
	}
	return false;
}

string FileSystem::NormalizeAbsolutePath(const string &path) {
	D_ASSERT(IsPathAbsolute(path));
	auto result = StringUtil::Lower(FileSystem::ConvertSeparators(path));
	if (PathMatched(result, "\\")) {
		// Path starts with a single backslash or forward slash
		// prepend drive letter
		return GetWorkingDirectory().substr(0, 2) + result;
	}
	return result;
}

string FileSystem::PathSeparator() {
	return "\\";
}

void FileSystem::SetWorkingDirectory(const string &path) {
	auto unicode_path = WindowsUtil::UTF8ToUnicode(path.c_str());
	if (!SetCurrentDirectoryW(unicode_path.c_str())) {
		throw IOException("Could not change working directory to \"%s\"", path);
	}
}

idx_t FileSystem::GetAvailableMemory() {
	ULONGLONG available_memory_kb;
	if (GetPhysicallyInstalledSystemMemory(&available_memory_kb)) {
		return MinValue<idx_t>(available_memory_kb * 1000, UINTPTR_MAX);
	}
	// fallback: try GlobalMemoryStatusEx
	MEMORYSTATUSEX mem_state;
	mem_state.dwLength = sizeof(MEMORYSTATUSEX);

	if (GlobalMemoryStatusEx(&mem_state)) {
		return MinValue<idx_t>(mem_state.ullTotalPhys, UINTPTR_MAX);
	}
	return DConstants::INVALID_INDEX;
}

string FileSystem::GetWorkingDirectory() {
	idx_t count = GetCurrentDirectoryW(0, nullptr);
	if (count == 0) {
		throw IOException("Could not get working directory!");
	}
	auto buffer = make_unsafe_uniq_array<wchar_t>(count);
	idx_t ret = GetCurrentDirectoryW(count, buffer.get());
	if (count != ret + 1) {
		throw IOException("Could not get working directory!");
	}
	return WindowsUtil::UnicodeToUTF8(buffer.get());
}

#endif

string FileSystem::JoinPath(const string &a, const string &b) {
	// FIXME: sanitize paths
	return a + PathSeparator() + b;
}

string FileSystem::ConvertSeparators(const string &path) {
	auto separator_str = PathSeparator();
	char separator = separator_str[0];
	if (separator == '/') {
		// on unix-based systems we only accept / as a separator
		return path;
	}
	// on windows-based systems we accept both
	return StringUtil::Replace(path, "/", separator_str);
}

string FileSystem::ExtractName(const string &path) {
	if (path.empty()) {
		return string();
	}
	auto normalized_path = ConvertSeparators(path);
	auto sep = PathSeparator();
	auto splits = StringUtil::Split(normalized_path, sep);
	D_ASSERT(!splits.empty());
	return splits.back();
}

string FileSystem::ExtractBaseName(const string &path) {
	if (path.empty()) {
		return string();
	}
	auto vec = StringUtil::Split(ExtractName(path), ".");
	D_ASSERT(!vec.empty());
	return vec[0];
}

string FileSystem::GetHomeDirectory(optional_ptr<FileOpener> opener) {
	// read the home_directory setting first, if it is set
	if (opener) {
		Value result;
		if (opener->TryGetCurrentSetting("home_directory", result)) {
			if (!result.IsNull() && !result.ToString().empty()) {
				return result.ToString();
			}
		}
	}
	// fallback to the default home directories for the specified system
#ifdef DUCKDB_WINDOWS
	return FileSystem::GetEnvVariable("USERPROFILE");
#else
	return FileSystem::GetEnvVariable("HOME");
#endif
}

string FileSystem::GetHomeDirectory() {
	return GetHomeDirectory(nullptr);
}

string FileSystem::ExpandPath(const string &path, optional_ptr<FileOpener> opener) {
	if (path.empty()) {
		return path;
	}
	if (path[0] == '~') {
		return GetHomeDirectory(opener) + path.substr(1);
	}
	return path;
}

string FileSystem::ExpandPath(const string &path) {
	return FileSystem::ExpandPath(path, nullptr);
}

// LCOV_EXCL_START
unique_ptr<FileHandle> FileSystem::OpenFile(const string &path, uint8_t flags, FileLockType lock,
                                            FileCompressionType compression, FileOpener *opener) {
	throw NotImplementedException("%s: OpenFile is not implemented!", GetName());
}

void FileSystem::Read(FileHandle &handle, void *buffer, int64_t nr_bytes, idx_t location) {
	throw NotImplementedException("%s: Read (with location) is not implemented!", GetName());
}

void FileSystem::Write(FileHandle &handle, void *buffer, int64_t nr_bytes, idx_t location) {
	throw NotImplementedException("%s: Write (with location) is not implemented!", GetName());
}

int64_t FileSystem::Read(FileHandle &handle, void *buffer, int64_t nr_bytes) {
	throw NotImplementedException("%s: Read is not implemented!", GetName());
}

int64_t FileSystem::Write(FileHandle &handle, void *buffer, int64_t nr_bytes) {
	throw NotImplementedException("%s: Write is not implemented!", GetName());
}

int64_t FileSystem::GetFileSize(FileHandle &handle) {
	throw NotImplementedException("%s: GetFileSize is not implemented!", GetName());
}

time_t FileSystem::GetLastModifiedTime(FileHandle &handle) {
	throw NotImplementedException("%s: GetLastModifiedTime is not implemented!", GetName());
}

FileType FileSystem::GetFileType(FileHandle &handle) {
	return FileType::FILE_TYPE_INVALID;
}

void FileSystem::Truncate(FileHandle &handle, int64_t new_size) {
	throw NotImplementedException("%s: Truncate is not implemented!", GetName());
}

bool FileSystem::DirectoryExists(const string &directory) {
	throw NotImplementedException("%s: DirectoryExists is not implemented!", GetName());
}

void FileSystem::CreateDirectory(const string &directory) {
	throw NotImplementedException("%s: CreateDirectory is not implemented!", GetName());
}

void FileSystem::RemoveDirectory(const string &directory) {
	throw NotImplementedException("%s: RemoveDirectory is not implemented!", GetName());
}

bool FileSystem::ListFiles(const string &directory, const std::function<void(const string &, bool)> &callback,
                           FileOpener *opener) {
	throw NotImplementedException("%s: ListFiles is not implemented!", GetName());
}

void FileSystem::MoveFile(const string &source, const string &target) {
	throw NotImplementedException("%s: MoveFile is not implemented!", GetName());
}

bool FileSystem::FileExists(const string &filename) {
	throw NotImplementedException("%s: FileExists is not implemented!", GetName());
}

bool FileSystem::IsPipe(const string &filename) {
	throw NotImplementedException("%s: IsPipe is not implemented!", GetName());
}

void FileSystem::RemoveFile(const string &filename) {
	throw NotImplementedException("%s: RemoveFile is not implemented!", GetName());
}

void FileSystem::FileSync(FileHandle &handle) {
	throw NotImplementedException("%s: FileSync is not implemented!", GetName());
}

bool FileSystem::HasGlob(const string &str) {
	for (idx_t i = 0; i < str.size(); i++) {
		switch (str[i]) {
		case '*':
		case '?':
		case '[':
			return true;
		default:
			break;
		}
	}
	return false;
}

vector<string> FileSystem::Glob(const string &path, FileOpener *opener) {
	throw NotImplementedException("%s: Glob is not implemented!", GetName());
}

void FileSystem::RegisterSubSystem(unique_ptr<FileSystem> sub_fs) {
	throw NotImplementedException("%s: Can't register a sub system on a non-virtual file system", GetName());
}

void FileSystem::RegisterSubSystem(FileCompressionType compression_type, unique_ptr<FileSystem> sub_fs) {
	throw NotImplementedException("%s: Can't register a sub system on a non-virtual file system", GetName());
}

void FileSystem::UnregisterSubSystem(const string &name) {
	throw NotImplementedException("%s: Can't unregister a sub system on a non-virtual file system", GetName());
}

vector<string> FileSystem::ListSubSystems() {
	throw NotImplementedException("%s: Can't list sub systems on a non-virtual file system", GetName());
}

bool FileSystem::CanHandleFile(const string &fpath) {
	throw NotImplementedException("%s: CanHandleFile is not implemented!", GetName());
}

vector<string> FileSystem::GlobFiles(const string &pattern, ClientContext &context, FileGlobOptions options) {
	auto result = Glob(pattern);
	if (result.empty()) {
		string required_extension;
		if (FileSystem::IsRemoteFile(pattern)) {
			required_extension = "httpfs";
		}
		if (!required_extension.empty() && !context.db->ExtensionIsLoaded(required_extension)) {
			// an extension is required to read this file but it is not loaded - try to load it
			ExtensionHelper::LoadExternalExtension(context, required_extension);
			// success! glob again
			// check the extension is loaded just in case to prevent an infinite loop here
			if (!context.db->ExtensionIsLoaded(required_extension)) {
				throw InternalException("Extension load \"%s\" did not throw but somehow the extension was not loaded",
				                        required_extension);
			}
			return GlobFiles(pattern, context, options);
		}
		if (options == FileGlobOptions::DISALLOW_EMPTY) {
			throw IOException("No files found that match the pattern \"%s\"", pattern);
		}
	}
	return result;
}

void FileSystem::Seek(FileHandle &handle, idx_t location) {
	throw NotImplementedException("%s: Seek is not implemented!", GetName());
}

void FileSystem::Reset(FileHandle &handle) {
	handle.Seek(0);
}

idx_t FileSystem::SeekPosition(FileHandle &handle) {
	throw NotImplementedException("%s: SeekPosition is not implemented!", GetName());
}

bool FileSystem::CanSeek() {
	throw NotImplementedException("%s: CanSeek is not implemented!", GetName());
}

unique_ptr<FileHandle> FileSystem::OpenCompressedFile(unique_ptr<FileHandle> handle, bool write) {
	throw NotImplementedException("%s: OpenCompressedFile is not implemented!", GetName());
}

bool FileSystem::OnDiskFile(FileHandle &handle) {
	throw NotImplementedException("%s: OnDiskFile is not implemented!", GetName());
}
// LCOV_EXCL_STOP

FileHandle::FileHandle(FileSystem &file_system, string path_p) : file_system(file_system), path(std::move(path_p)) {
}

FileHandle::~FileHandle() {
}

int64_t FileHandle::Read(void *buffer, idx_t nr_bytes) {
	return file_system.Read(*this, buffer, nr_bytes);
}

int64_t FileHandle::Write(void *buffer, idx_t nr_bytes) {
	return file_system.Write(*this, buffer, nr_bytes);
}

void FileHandle::Read(void *buffer, idx_t nr_bytes, idx_t location) {
	file_system.Read(*this, buffer, nr_bytes, location);
}

void FileHandle::Write(void *buffer, idx_t nr_bytes, idx_t location) {
	file_system.Write(*this, buffer, nr_bytes, location);
}

void FileHandle::Seek(idx_t location) {
	file_system.Seek(*this, location);
}

void FileHandle::Reset() {
	file_system.Reset(*this);
}

idx_t FileHandle::SeekPosition() {
	return file_system.SeekPosition(*this);
}

bool FileHandle::CanSeek() {
	return file_system.CanSeek();
}

string FileHandle::ReadLine() {
	string result;
	char buffer[1];
	while (true) {
		idx_t tuples_read = Read(buffer, 1);
		if (tuples_read == 0 || buffer[0] == '\n') {
			return result;
		}
		if (buffer[0] != '\r') {
			result += buffer[0];
		}
	}
}

bool FileHandle::OnDiskFile() {
	return file_system.OnDiskFile(*this);
}

idx_t FileHandle::GetFileSize() {
	return file_system.GetFileSize(*this);
}

void FileHandle::Sync() {
	file_system.FileSync(*this);
}

void FileHandle::Truncate(int64_t new_size) {
	file_system.Truncate(*this, new_size);
}

FileType FileHandle::GetType() {
	return file_system.GetFileType(*this);
}

bool FileSystem::IsRemoteFile(const string &path) {
	const string prefixes[] = {"http://", "https://", "s3://"};
	for (auto &prefix : prefixes) {
		if (StringUtil::StartsWith(path, prefix)) {
			return true;
		}
	}
	return false;
}

} // namespace duckdb



namespace duckdb {

void FilenamePattern::SetFilenamePattern(const string &pattern) {
	const string id_format {"{i}"};
	const string uuid_format {"{uuid}"};

	_base = pattern;

	_pos = _base.find(id_format);
	if (_pos != string::npos) {
		_base = StringUtil::Replace(_base, id_format, "");
		_uuid = false;
	}

	_pos = _base.find(uuid_format);
	if (_pos != string::npos) {
		_base = StringUtil::Replace(_base, uuid_format, "");
		_uuid = true;
	}

	_pos = std::min(_pos, (idx_t)_base.length());
}

string FilenamePattern::CreateFilename(const FileSystem &fs, const string &path, const string &extension,
                                       idx_t offset) const {
	string result(_base);
	string replacement;

	if (_uuid) {
		replacement = UUID::ToString(UUID::GenerateRandomUUID());
	} else {
		replacement = std::to_string(offset);
	}
	result.insert(_pos, replacement);
	return fs.JoinPath(path, result + "." + extension);
}

} // namespace duckdb





namespace duckdb {
string_t FSSTPrimitives::DecompressValue(void *duckdb_fsst_decoder, Vector &result, unsigned char *compressed_string,
                                         idx_t compressed_string_len) {
	D_ASSERT(result.GetVectorType() == VectorType::FLAT_VECTOR);
	unsigned char decompress_buffer[StringUncompressed::STRING_BLOCK_LIMIT + 1];
	auto decompressed_string_size =
	    duckdb_fsst_decompress((duckdb_fsst_decoder_t *)duckdb_fsst_decoder, compressed_string_len, compressed_string,
	                           StringUncompressed::STRING_BLOCK_LIMIT + 1, &decompress_buffer[0]);
	D_ASSERT(decompressed_string_size <= StringUncompressed::STRING_BLOCK_LIMIT);

	return StringVector::AddStringOrBlob(result, (const char *)decompress_buffer, decompressed_string_size);
}

Value FSSTPrimitives::DecompressValue(void *duckdb_fsst_decoder, unsigned char *compressed_string,
                                      idx_t compressed_string_len) {
	unsigned char decompress_buffer[StringUncompressed::STRING_BLOCK_LIMIT + 1];
	auto decompressed_string_size =
	    duckdb_fsst_decompress((duckdb_fsst_decoder_t *)duckdb_fsst_decoder, compressed_string_len, compressed_string,
	                           StringUncompressed::STRING_BLOCK_LIMIT + 1, &decompress_buffer[0]);
	D_ASSERT(decompressed_string_size <= StringUncompressed::STRING_BLOCK_LIMIT);

	return Value(string((char *)decompress_buffer, decompressed_string_size));
}

} // namespace duckdb









namespace duckdb {

/*

  0      2 bytes  magic header  0x1f, 0x8b (\037 \213)
  2      1 byte   compression method
                     0: store (copied)
                     1: compress
                     2: pack
                     3: lzh
                     4..7: reserved
                     8: deflate
  3      1 byte   flags
                     bit 0 set: file probably ascii text
                     bit 1 set: continuation of multi-part gzip file, part number present
                     bit 2 set: extra field present
                     bit 3 set: original file name present
                     bit 4 set: file comment present
                     bit 5 set: file is encrypted, encryption header present
                     bit 6,7:   reserved
  4      4 bytes  file modification time in Unix format
  8      1 byte   extra flags (depend on compression method)
  9      1 byte   OS type
[
         2 bytes  optional part number (second part=1)
]?
[
         2 bytes  optional extra field length (e)
        (e)bytes  optional extra field
]?
[
           bytes  optional original file name, zero terminated
]?
[
           bytes  optional file comment, zero terminated
]?
[
        12 bytes  optional encryption header
]?
           bytes  compressed data
         4 bytes  crc32
         4 bytes  uncompressed input size modulo 2^32

 */

static idx_t GZipConsumeString(FileHandle &input) {
	idx_t size = 1; // terminator
	char buffer[1];
	while (input.Read(buffer, 1) == 1) {
		if (buffer[0] == '\0') {
			break;
		}
		size++;
	}
	return size;
}

struct MiniZStreamWrapper : public StreamWrapper {
	~MiniZStreamWrapper() override;

	CompressedFile *file = nullptr;
	duckdb_miniz::mz_stream *mz_stream_ptr = nullptr;
	bool writing = false;
	duckdb_miniz::mz_ulong crc;
	idx_t total_size;

public:
	void Initialize(CompressedFile &file, bool write) override;

	bool Read(StreamData &stream_data) override;
	void Write(CompressedFile &file, StreamData &stream_data, data_ptr_t buffer, int64_t nr_bytes) override;

	void Close() override;

	void FlushStream();
};

MiniZStreamWrapper::~MiniZStreamWrapper() {
	// avoid closing if destroyed during stack unwinding
	if (Exception::UncaughtException()) {
		return;
	}
	try {
		MiniZStreamWrapper::Close();
	} catch (...) {
	}
}

void MiniZStreamWrapper::Initialize(CompressedFile &file, bool write) {
	Close();
	this->file = &file;
	mz_stream_ptr = new duckdb_miniz::mz_stream();
	memset(mz_stream_ptr, 0, sizeof(duckdb_miniz::mz_stream));
	this->writing = write;

	// TODO use custom alloc/free methods in miniz to throw exceptions on OOM
	uint8_t gzip_hdr[GZIP_HEADER_MINSIZE];
	if (write) {
		crc = MZ_CRC32_INIT;
		total_size = 0;

		MiniZStream::InitializeGZIPHeader(gzip_hdr);
		file.child_handle->Write(gzip_hdr, GZIP_HEADER_MINSIZE);

		auto ret = mz_deflateInit2((duckdb_miniz::mz_streamp)mz_stream_ptr, duckdb_miniz::MZ_DEFAULT_LEVEL, MZ_DEFLATED,
		                           -MZ_DEFAULT_WINDOW_BITS, 1, 0);
		if (ret != duckdb_miniz::MZ_OK) {
			throw InternalException("Failed to initialize miniz");
		}
	} else {
		idx_t data_start = GZIP_HEADER_MINSIZE;
		auto read_count = file.child_handle->Read(gzip_hdr, GZIP_HEADER_MINSIZE);
		GZipFileSystem::VerifyGZIPHeader(gzip_hdr, read_count);
		// Skip over the extra field if necessary
		if (gzip_hdr[3] & GZIP_FLAG_EXTRA) {
			uint8_t gzip_xlen[2];
			file.child_handle->Seek(data_start);
			file.child_handle->Read(gzip_xlen, 2);
			idx_t xlen = (uint8_t)gzip_xlen[0] | (uint8_t)gzip_xlen[1] << 8;
			data_start += xlen + 2;
		}
		// Skip over the file name if necessary
		if (gzip_hdr[3] & GZIP_FLAG_NAME) {
			file.child_handle->Seek(data_start);
			data_start += GZipConsumeString(*file.child_handle);
		}
		file.child_handle->Seek(data_start);
		// stream is now set to beginning of payload data
		auto ret = duckdb_miniz::mz_inflateInit2((duckdb_miniz::mz_streamp)mz_stream_ptr, -MZ_DEFAULT_WINDOW_BITS);
		if (ret != duckdb_miniz::MZ_OK) {
			throw InternalException("Failed to initialize miniz");
		}
	}
}

bool MiniZStreamWrapper::Read(StreamData &sd) {
	// Handling for the concatenated files
	if (sd.refresh) {
		sd.refresh = false;
		auto body_ptr = sd.in_buff_start + GZIP_FOOTER_SIZE;
		uint8_t gzip_hdr[GZIP_HEADER_MINSIZE];
		memcpy(gzip_hdr, body_ptr, GZIP_HEADER_MINSIZE);
		GZipFileSystem::VerifyGZIPHeader(gzip_hdr, GZIP_HEADER_MINSIZE);
		body_ptr += GZIP_HEADER_MINSIZE;
		if (gzip_hdr[3] & GZIP_FLAG_EXTRA) {
			idx_t xlen = (uint8_t)*body_ptr | (uint8_t) * (body_ptr + 1) << 8;
			body_ptr += xlen + 2;
			if (GZIP_FOOTER_SIZE + GZIP_HEADER_MINSIZE + 2 + xlen >= GZIP_HEADER_MAXSIZE) {
				throw InternalException("Extra field resulting in GZIP header larger than defined maximum (%d)",
				                        GZIP_HEADER_MAXSIZE);
			}
		}
		if (gzip_hdr[3] & GZIP_FLAG_NAME) {
			char c;
			do {
				c = *body_ptr;
				body_ptr++;
			} while (c != '\0' && body_ptr < sd.in_buff_end);
			if ((idx_t)(body_ptr - sd.in_buff_start) >= GZIP_HEADER_MAXSIZE) {
				throw InternalException("Filename resulting in GZIP header larger than defined maximum (%d)",
				                        GZIP_HEADER_MAXSIZE);
			}
		}
		sd.in_buff_start = body_ptr;
		if (sd.in_buff_end - sd.in_buff_start < 1) {
			Close();
			return true;
		}
		duckdb_miniz::mz_inflateEnd(mz_stream_ptr);
		auto sta = duckdb_miniz::mz_inflateInit2((duckdb_miniz::mz_streamp)mz_stream_ptr, -MZ_DEFAULT_WINDOW_BITS);
		if (sta != duckdb_miniz::MZ_OK) {
			throw InternalException("Failed to initialize miniz");
		}
	}

	// actually decompress
	mz_stream_ptr->next_in = (data_ptr_t)sd.in_buff_start;
	D_ASSERT(sd.in_buff_end - sd.in_buff_start < NumericLimits<int32_t>::Maximum());
	mz_stream_ptr->avail_in = (uint32_t)(sd.in_buff_end - sd.in_buff_start);
	mz_stream_ptr->next_out = (data_ptr_t)sd.out_buff_end;
	mz_stream_ptr->avail_out = (uint32_t)((sd.out_buff.get() + sd.out_buf_size) - sd.out_buff_end);
	auto ret = duckdb_miniz::mz_inflate(mz_stream_ptr, duckdb_miniz::MZ_NO_FLUSH);
	if (ret != duckdb_miniz::MZ_OK && ret != duckdb_miniz::MZ_STREAM_END) {
		throw IOException("Failed to decode gzip stream: %s", duckdb_miniz::mz_error(ret));
	}
	// update pointers following inflate()
	sd.in_buff_start = (data_ptr_t)mz_stream_ptr->next_in;
	sd.in_buff_end = sd.in_buff_start + mz_stream_ptr->avail_in;
	sd.out_buff_end = (data_ptr_t)mz_stream_ptr->next_out;
	D_ASSERT(sd.out_buff_end + mz_stream_ptr->avail_out == sd.out_buff.get() + sd.out_buf_size);

	// if stream ended, deallocate inflator
	if (ret == duckdb_miniz::MZ_STREAM_END) {
		// Last read from file done and remaining bytes only for footer or less
		if ((sd.in_buff_end < sd.in_buff.get() + sd.in_buf_size) && mz_stream_ptr->avail_in <= GZIP_FOOTER_SIZE) {
			Close();
			return true;
		}
		if (mz_stream_ptr->avail_in > GZIP_FOOTER_SIZE) {
			// Definitely not concatenated gzip
			if (*(sd.in_buff_start + GZIP_FOOTER_SIZE) != 0x1F) {
				Close();
				return true;
			}
		}
		// Concatenated GZIP potentially coming up - refresh input buffer
		sd.refresh = true;
	}
	return false;
}

void MiniZStreamWrapper::Write(CompressedFile &file, StreamData &sd, data_ptr_t uncompressed_data,
                               int64_t uncompressed_size) {
	// update the src and the total size
	crc = duckdb_miniz::mz_crc32(crc, (const unsigned char *)uncompressed_data, uncompressed_size);
	total_size += uncompressed_size;

	auto remaining = uncompressed_size;
	while (remaining > 0) {
		idx_t output_remaining = (sd.out_buff.get() + sd.out_buf_size) - sd.out_buff_start;

		mz_stream_ptr->next_in = (const unsigned char *)uncompressed_data;
		mz_stream_ptr->avail_in = remaining;
		mz_stream_ptr->next_out = sd.out_buff_start;
		mz_stream_ptr->avail_out = output_remaining;

		auto res = mz_deflate(mz_stream_ptr, duckdb_miniz::MZ_NO_FLUSH);
		if (res != duckdb_miniz::MZ_OK) {
			D_ASSERT(res != duckdb_miniz::MZ_STREAM_END);
			throw InternalException("Failed to compress GZIP block");
		}
		sd.out_buff_start += output_remaining - mz_stream_ptr->avail_out;
		if (mz_stream_ptr->avail_out == 0) {
			// no more output buffer available: flush
			file.child_handle->Write(sd.out_buff.get(), sd.out_buff_start - sd.out_buff.get());
			sd.out_buff_start = sd.out_buff.get();
		}
		idx_t written = remaining - mz_stream_ptr->avail_in;
		uncompressed_data += written;
		remaining = mz_stream_ptr->avail_in;
	}
}

void MiniZStreamWrapper::FlushStream() {
	auto &sd = file->stream_data;
	mz_stream_ptr->next_in = nullptr;
	mz_stream_ptr->avail_in = 0;
	while (true) {
		auto output_remaining = (sd.out_buff.get() + sd.out_buf_size) - sd.out_buff_start;
		mz_stream_ptr->next_out = sd.out_buff_start;
		mz_stream_ptr->avail_out = output_remaining;

		auto res = mz_deflate(mz_stream_ptr, duckdb_miniz::MZ_FINISH);
		sd.out_buff_start += (output_remaining - mz_stream_ptr->avail_out);
		if (sd.out_buff_start > sd.out_buff.get()) {
			file->child_handle->Write(sd.out_buff.get(), sd.out_buff_start - sd.out_buff.get());
			sd.out_buff_start = sd.out_buff.get();
		}
		if (res == duckdb_miniz::MZ_STREAM_END) {
			break;
		}
		if (res != duckdb_miniz::MZ_OK) {
			throw InternalException("Failed to compress GZIP block");
		}
	}
}

void MiniZStreamWrapper::Close() {
	if (!mz_stream_ptr) {
		return;
	}
	if (writing) {
		// flush anything remaining in the stream
		FlushStream();

		// write the footer
		unsigned char gzip_footer[MiniZStream::GZIP_FOOTER_SIZE];
		MiniZStream::InitializeGZIPFooter(gzip_footer, crc, total_size);
		file->child_handle->Write(gzip_footer, MiniZStream::GZIP_FOOTER_SIZE);

		duckdb_miniz::mz_deflateEnd(mz_stream_ptr);
	} else {
		duckdb_miniz::mz_inflateEnd(mz_stream_ptr);
	}
	delete mz_stream_ptr;
	mz_stream_ptr = nullptr;
	file = nullptr;
}

class GZipFile : public CompressedFile {
public:
	GZipFile(unique_ptr<FileHandle> child_handle_p, const string &path, bool write)
	    : CompressedFile(gzip_fs, std::move(child_handle_p), path) {
		Initialize(write);
	}

	GZipFileSystem gzip_fs;
};

void GZipFileSystem::VerifyGZIPHeader(uint8_t gzip_hdr[], idx_t read_count) {
	// check for incorrectly formatted files
	if (read_count != GZIP_HEADER_MINSIZE) {
		throw IOException("Input is not a GZIP stream");
	}
	if (gzip_hdr[0] != 0x1F || gzip_hdr[1] != 0x8B) { // magic header
		throw IOException("Input is not a GZIP stream");
	}
	if (gzip_hdr[2] != GZIP_COMPRESSION_DEFLATE) { // compression method
		throw IOException("Unsupported GZIP compression method");
	}
	if (gzip_hdr[3] & GZIP_FLAG_UNSUPPORTED) {
		throw IOException("Unsupported GZIP archive");
	}
}

string GZipFileSystem::UncompressGZIPString(const string &in) {
	// decompress file
	auto body_ptr = in.data();

	auto mz_stream_ptr = new duckdb_miniz::mz_stream();
	memset(mz_stream_ptr, 0, sizeof(duckdb_miniz::mz_stream));

	uint8_t gzip_hdr[GZIP_HEADER_MINSIZE];

	// check for incorrectly formatted files

	// TODO this is mostly the same as gzip_file_system.cpp
	if (in.size() < GZIP_HEADER_MINSIZE) {
		throw IOException("Input is not a GZIP stream");
	}
	memcpy(gzip_hdr, body_ptr, GZIP_HEADER_MINSIZE);
	body_ptr += GZIP_HEADER_MINSIZE;
	GZipFileSystem::VerifyGZIPHeader(gzip_hdr, GZIP_HEADER_MINSIZE);

	if (gzip_hdr[3] & GZIP_FLAG_EXTRA) {
		throw IOException("Extra field in a GZIP stream unsupported");
	}

	if (gzip_hdr[3] & GZIP_FLAG_NAME) {
		char c;
		do {
			c = *body_ptr;
			body_ptr++;
		} while (c != '\0' && (idx_t)(body_ptr - in.data()) < in.size());
	}

	// stream is now set to beginning of payload data
	auto status = duckdb_miniz::mz_inflateInit2(mz_stream_ptr, -MZ_DEFAULT_WINDOW_BITS);
	if (status != duckdb_miniz::MZ_OK) {
		throw InternalException("Failed to initialize miniz");
	}

	auto bytes_remaining = in.size() - (body_ptr - in.data());
	mz_stream_ptr->next_in = (unsigned char *)body_ptr;
	mz_stream_ptr->avail_in = bytes_remaining;

	unsigned char decompress_buffer[BUFSIZ];
	string decompressed;

	while (status == duckdb_miniz::MZ_OK) {
		mz_stream_ptr->next_out = decompress_buffer;
		mz_stream_ptr->avail_out = sizeof(decompress_buffer);
		status = mz_inflate(mz_stream_ptr, duckdb_miniz::MZ_NO_FLUSH);
		if (status != duckdb_miniz::MZ_STREAM_END && status != duckdb_miniz::MZ_OK) {
			throw IOException("Failed to uncompress");
		}
		decompressed.append((char *)decompress_buffer, mz_stream_ptr->total_out - decompressed.size());
	}
	duckdb_miniz::mz_inflateEnd(mz_stream_ptr);
	if (decompressed.empty()) {
		throw IOException("Failed to uncompress");
	}
	return decompressed;
}

unique_ptr<FileHandle> GZipFileSystem::OpenCompressedFile(unique_ptr<FileHandle> handle, bool write) {
	auto path = handle->path;
	return make_uniq<GZipFile>(std::move(handle), path, write);
}

unique_ptr<StreamWrapper> GZipFileSystem::CreateStream() {
	return make_uniq<MiniZStreamWrapper>();
}

idx_t GZipFileSystem::InBufferSize() {
	return BUFFER_SIZE;
}

idx_t GZipFileSystem::OutBufferSize() {
	return BUFFER_SIZE;
}

} // namespace duckdb











namespace duckdb {

static unordered_map<column_t, string> GetKnownColumnValues(string &filename,
                                                            unordered_map<string, column_t> &column_map,
                                                            duckdb_re2::RE2 &compiled_regex, bool filename_col,
                                                            bool hive_partition_cols) {
	unordered_map<column_t, string> result;

	if (filename_col) {
		auto lookup_column_id = column_map.find("filename");
		if (lookup_column_id != column_map.end()) {
			result[lookup_column_id->second] = filename;
		}
	}

	if (hive_partition_cols) {
		auto partitions = HivePartitioning::Parse(filename, compiled_regex);
		for (auto &partition : partitions) {
			auto lookup_column_id = column_map.find(partition.first);
			if (lookup_column_id != column_map.end()) {
				result[lookup_column_id->second] = partition.second;
			}
		}
	}

	return result;
}

// Takes an expression and converts a list of known column_refs to constants
static void ConvertKnownColRefToConstants(unique_ptr<Expression> &expr,
                                          unordered_map<column_t, string> &known_column_values, idx_t table_index) {
	if (expr->type == ExpressionType::BOUND_COLUMN_REF) {
		auto &bound_colref = expr->Cast<BoundColumnRefExpression>();

		// This bound column ref is for another table
		if (table_index != bound_colref.binding.table_index) {
			return;
		}

		auto lookup = known_column_values.find(bound_colref.binding.column_index);
		if (lookup != known_column_values.end()) {
			expr = make_uniq<BoundConstantExpression>(Value(lookup->second).DefaultCastAs(bound_colref.return_type));
		}
	} else {
		ExpressionIterator::EnumerateChildren(*expr, [&](unique_ptr<Expression> &child) {
			ConvertKnownColRefToConstants(child, known_column_values, table_index);
		});
	}
}

// matches hive partitions in file name. For example:
// 	- s3://bucket/var1=value1/bla/bla/var2=value2
//  - http(s)://domain(:port)/lala/kasdl/var1=value1/?not-a-var=not-a-value
//  - folder/folder/folder/../var1=value1/etc/.//var2=value2
const string HivePartitioning::REGEX_STRING = "[\\/\\\\]([^\\/\\?\\\\]+)=([^\\/\\n\\?\\\\]+)";

std::map<string, string> HivePartitioning::Parse(const string &filename, duckdb_re2::RE2 &regex) {
	std::map<string, string> result;
	duckdb_re2::StringPiece input(filename); // Wrap a StringPiece around it

	string var;
	string value;
	while (RE2::FindAndConsume(&input, regex, &var, &value)) {
		result.insert(std::pair<string, string>(var, value));
	}
	return result;
}

std::map<string, string> HivePartitioning::Parse(const string &filename) {
	duckdb_re2::RE2 regex(REGEX_STRING);
	return Parse(filename, regex);
}

// TODO: this can still be improved by removing the parts of filter expressions that are true for all remaining files.
//		 currently, only expressions that cannot be evaluated during pushdown are removed.
void HivePartitioning::ApplyFiltersToFileList(ClientContext &context, vector<string> &files,
                                              vector<unique_ptr<Expression>> &filters,
                                              unordered_map<string, column_t> &column_map, idx_t table_index,
                                              bool hive_enabled, bool filename_enabled) {
	vector<string> pruned_files;
	vector<bool> have_preserved_filter(filters.size(), false);
	vector<unique_ptr<Expression>> pruned_filters;
	duckdb_re2::RE2 regex(REGEX_STRING);

	if ((!filename_enabled && !hive_enabled) || filters.empty()) {
		return;
	}

	for (idx_t i = 0; i < files.size(); i++) {
		auto &file = files[i];
		bool should_prune_file = false;
		auto known_values = GetKnownColumnValues(file, column_map, regex, filename_enabled, hive_enabled);

		FilterCombiner combiner(context);

		for (idx_t j = 0; j < filters.size(); j++) {
			auto &filter = filters[j];
			unique_ptr<Expression> filter_copy = filter->Copy();
			ConvertKnownColRefToConstants(filter_copy, known_values, table_index);
			// Evaluate the filter, if it can be evaluated here, we can not prune this filter
			Value result_value;

			if (!filter_copy->IsScalar() || !filter_copy->IsFoldable() ||
			    !ExpressionExecutor::TryEvaluateScalar(context, *filter_copy, result_value)) {
				// can not be evaluated only with the filename/hive columns added, we can not prune this filter
				if (!have_preserved_filter[j]) {
					pruned_filters.emplace_back(filter->Copy());
					have_preserved_filter[j] = true;
				}
			} else if (!result_value.GetValue<bool>()) {
				// filter evaluates to false
				should_prune_file = true;
			}

			// Use filter combiner to determine that this filter makes
			if (!should_prune_file && combiner.AddFilter(std::move(filter_copy)) == FilterResult::UNSATISFIABLE) {
				should_prune_file = true;
			}
		}

		if (!should_prune_file) {
			pruned_files.push_back(file);
		}
	}

	D_ASSERT(filters.size() >= pruned_filters.size());

	filters = std::move(pruned_filters);
	files = std::move(pruned_files);
}

HivePartitionedColumnData::HivePartitionedColumnData(const HivePartitionedColumnData &other)
    : PartitionedColumnData(other), hashes_v(LogicalType::HASH) {
	// Synchronize to ensure consistency of shared partition map
	if (other.global_state) {
		global_state = other.global_state;
		unique_lock<mutex> lck(global_state->lock);
		SynchronizeLocalMap();
	}
	InitializeKeys();
}

void HivePartitionedColumnData::InitializeKeys() {
	keys.resize(STANDARD_VECTOR_SIZE);
	for (idx_t i = 0; i < STANDARD_VECTOR_SIZE; i++) {
		keys[i].values.resize(group_by_columns.size());
	}
}

template <class T>
static inline Value GetHiveKeyValue(const T &val) {
	return Value::CreateValue<T>(val);
}

template <class T>
static inline Value GetHiveKeyValue(const T &val, const LogicalType &type) {
	auto result = GetHiveKeyValue(val);
	result.Reinterpret(type);
	return result;
}

static inline Value GetHiveKeyNullValue(const LogicalType &type) {
	Value result;
	result.Reinterpret(type);
	return result;
}

template <class T>
static void TemplatedGetHivePartitionValues(Vector &input, vector<HivePartitionKey> &keys, const idx_t col_idx,
                                            const idx_t count) {
	UnifiedVectorFormat format;
	input.ToUnifiedFormat(count, format);

	const auto &sel = *format.sel;
	const auto data = (T *)format.data;
	const auto &validity = format.validity;

	const auto &type = input.GetType();

	const auto reinterpret = Value::CreateValue<T>(data[0]).GetTypeMutable() != type;
	if (reinterpret) {
		for (idx_t i = 0; i < count; i++) {
			auto &key = keys[i];
			const auto idx = sel.get_index(i);
			if (validity.RowIsValid(idx)) {
				key.values[col_idx] = GetHiveKeyValue(data[idx], type);
			} else {
				key.values[col_idx] = GetHiveKeyNullValue(type);
			}
		}
	} else {
		for (idx_t i = 0; i < count; i++) {
			auto &key = keys[i];
			const auto idx = sel.get_index(i);
			if (validity.RowIsValid(idx)) {
				key.values[col_idx] = GetHiveKeyValue(data[idx]);
			} else {
				key.values[col_idx] = GetHiveKeyNullValue(type);
			}
		}
	}
}

static void GetNestedHivePartitionValues(Vector &input, vector<HivePartitionKey> &keys, const idx_t col_idx,
                                         const idx_t count) {
	for (idx_t i = 0; i < count; i++) {
		auto &key = keys[i];
		key.values[col_idx] = input.GetValue(i);
	}
}

static void GetHivePartitionValuesTypeSwitch(Vector &input, vector<HivePartitionKey> &keys, const idx_t col_idx,
                                             const idx_t count) {
	const auto &type = input.GetType();
	switch (type.InternalType()) {
	case PhysicalType::BOOL:
		TemplatedGetHivePartitionValues<bool>(input, keys, col_idx, count);
		break;
	case PhysicalType::INT8:
		TemplatedGetHivePartitionValues<int8_t>(input, keys, col_idx, count);
		break;
	case PhysicalType::INT16:
		TemplatedGetHivePartitionValues<int16_t>(input, keys, col_idx, count);
		break;
	case PhysicalType::INT32:
		TemplatedGetHivePartitionValues<int32_t>(input, keys, col_idx, count);
		break;
	case PhysicalType::INT64:
		TemplatedGetHivePartitionValues<int64_t>(input, keys, col_idx, count);
		break;
	case PhysicalType::INT128:
		TemplatedGetHivePartitionValues<hugeint_t>(input, keys, col_idx, count);
		break;
	case PhysicalType::UINT8:
		TemplatedGetHivePartitionValues<uint8_t>(input, keys, col_idx, count);
		break;
	case PhysicalType::UINT16:
		TemplatedGetHivePartitionValues<uint16_t>(input, keys, col_idx, count);
		break;
	case PhysicalType::UINT32:
		TemplatedGetHivePartitionValues<uint32_t>(input, keys, col_idx, count);
		break;
	case PhysicalType::UINT64:
		TemplatedGetHivePartitionValues<uint64_t>(input, keys, col_idx, count);
		break;
	case PhysicalType::FLOAT:
		TemplatedGetHivePartitionValues<float>(input, keys, col_idx, count);
		break;
	case PhysicalType::DOUBLE:
		TemplatedGetHivePartitionValues<double>(input, keys, col_idx, count);
		break;
	case PhysicalType::INTERVAL:
		TemplatedGetHivePartitionValues<interval_t>(input, keys, col_idx, count);
		break;
	case PhysicalType::VARCHAR:
		TemplatedGetHivePartitionValues<string_t>(input, keys, col_idx, count);
		break;
	case PhysicalType::STRUCT:
	case PhysicalType::LIST:
		GetNestedHivePartitionValues(input, keys, col_idx, count);
		break;
	default:
		throw InternalException("Unsupported type for HivePartitionedColumnData::ComputePartitionIndices");
	}
}

void HivePartitionedColumnData::ComputePartitionIndices(PartitionedColumnDataAppendState &state, DataChunk &input) {
	const auto count = input.size();

	input.Hash(group_by_columns, hashes_v);
	hashes_v.Flatten(count);

	for (idx_t col_idx = 0; col_idx < group_by_columns.size(); col_idx++) {
		auto &group_by_col = input.data[group_by_columns[col_idx]];
		GetHivePartitionValuesTypeSwitch(group_by_col, keys, col_idx, count);
	}

	const auto hashes = FlatVector::GetData<hash_t>(hashes_v);
	const auto partition_indices = FlatVector::GetData<idx_t>(state.partition_indices);
	for (idx_t i = 0; i < count; i++) {
		auto &key = keys[i];
		key.hash = hashes[i];
		auto lookup = local_partition_map.find(key);
		if (lookup == local_partition_map.end()) {
			idx_t new_partition_id = RegisterNewPartition(key, state);
			partition_indices[i] = new_partition_id;
		} else {
			partition_indices[i] = lookup->second;
		}
	}
}

std::map<idx_t, const HivePartitionKey *> HivePartitionedColumnData::GetReverseMap() {
	std::map<idx_t, const HivePartitionKey *> ret;
	for (const auto &pair : local_partition_map) {
		ret[pair.second] = &(pair.first);
	}
	return ret;
}

void HivePartitionedColumnData::GrowAllocators() {
	unique_lock<mutex> lck_gstate(allocators->lock);

	idx_t current_allocator_size = allocators->allocators.size();
	idx_t required_allocators = local_partition_map.size();

	allocators->allocators.reserve(current_allocator_size);
	for (idx_t i = current_allocator_size; i < required_allocators; i++) {
		CreateAllocator();
	}

	D_ASSERT(allocators->allocators.size() == local_partition_map.size());
}

void HivePartitionedColumnData::GrowAppendState(PartitionedColumnDataAppendState &state) {
	idx_t current_append_state_size = state.partition_append_states.size();
	idx_t required_append_state_size = local_partition_map.size();

	for (idx_t i = current_append_state_size; i < required_append_state_size; i++) {
		state.partition_append_states.emplace_back(make_uniq<ColumnDataAppendState>());
		state.partition_buffers.emplace_back(CreatePartitionBuffer());
	}
}

void HivePartitionedColumnData::GrowPartitions(PartitionedColumnDataAppendState &state) {
	idx_t current_partitions = partitions.size();
	idx_t required_partitions = local_partition_map.size();

	D_ASSERT(allocators->allocators.size() == required_partitions);

	for (idx_t i = current_partitions; i < required_partitions; i++) {
		partitions.emplace_back(CreatePartitionCollection(i));
		partitions[i]->InitializeAppend(*state.partition_append_states[i]);
	}
	D_ASSERT(partitions.size() == local_partition_map.size());
}

void HivePartitionedColumnData::SynchronizeLocalMap() {
	// Synchronise global map into local, may contain changes from other threads too
	for (auto it = global_state->partitions.begin() + local_partition_map.size(); it < global_state->partitions.end();
	     it++) {
		local_partition_map[(*it)->first] = (*it)->second;
	}
}

idx_t HivePartitionedColumnData::RegisterNewPartition(HivePartitionKey key, PartitionedColumnDataAppendState &state) {
	if (global_state) {
		idx_t partition_id;

		// Synchronize Global state with our local state with the newly discoveren partition
		{
			unique_lock<mutex> lck_gstate(global_state->lock);

			// Insert into global map, or return partition if already present
			auto res =
			    global_state->partition_map.emplace(std::make_pair(std::move(key), global_state->partition_map.size()));
			auto it = res.first;
			partition_id = it->second;

			// Add iterator to vector to allow incrementally updating local states from global state
			global_state->partitions.emplace_back(it);
			SynchronizeLocalMap();
		}

		// After synchronizing with the global state, we need to grow the shared allocators to support
		// the number of partitions, which guarantees that there's always enough allocators available to each thread
		GrowAllocators();

		// Grow local partition data
		GrowAppendState(state);
		GrowPartitions(state);

		return partition_id;
	} else {
		return local_partition_map.emplace(std::make_pair(std::move(key), local_partition_map.size())).first->second;
	}
}

} // namespace duckdb












#include <cstdint>
#include <cstdio>
#include <sys/stat.h>

#ifndef _WIN32
#include <dirent.h>
#include <fcntl.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#else


#include <io.h>
#include <string>

#ifdef __MINGW32__
// need to manually define this for mingw
extern "C" WINBASEAPI BOOL WINAPI GetPhysicallyInstalledSystemMemory(PULONGLONG);
#endif

#undef FILE_CREATE // woo mingw
#endif

namespace duckdb {

static void AssertValidFileFlags(uint8_t flags) {
#ifdef DEBUG
	bool is_read = flags & FileFlags::FILE_FLAGS_READ;
	bool is_write = flags & FileFlags::FILE_FLAGS_WRITE;
	// require either READ or WRITE (or both)
	D_ASSERT(is_read || is_write);
	// CREATE/Append flags require writing
	D_ASSERT(is_write || !(flags & FileFlags::FILE_FLAGS_APPEND));
	D_ASSERT(is_write || !(flags & FileFlags::FILE_FLAGS_FILE_CREATE));
	D_ASSERT(is_write || !(flags & FileFlags::FILE_FLAGS_FILE_CREATE_NEW));
	// cannot combine CREATE and CREATE_NEW flags
	D_ASSERT(!(flags & FileFlags::FILE_FLAGS_FILE_CREATE && flags & FileFlags::FILE_FLAGS_FILE_CREATE_NEW));
#endif
}

#ifndef _WIN32
bool LocalFileSystem::FileExists(const string &filename) {
	if (!filename.empty()) {
		if (access(filename.c_str(), 0) == 0) {
			struct stat status;
			stat(filename.c_str(), &status);
			if (S_ISREG(status.st_mode)) {
				return true;
			}
		}
	}
	// if any condition fails
	return false;
}

bool LocalFileSystem::IsPipe(const string &filename) {
	if (!filename.empty()) {
		if (access(filename.c_str(), 0) == 0) {
			struct stat status;
			stat(filename.c_str(), &status);
			if (S_ISFIFO(status.st_mode)) {
				return true;
			}
		}
	}
	// if any condition fails
	return false;
}

#else
bool LocalFileSystem::FileExists(const string &filename) {
	auto unicode_path = WindowsUtil::UTF8ToUnicode(filename.c_str());
	const wchar_t *wpath = unicode_path.c_str();
	if (_waccess(wpath, 0) == 0) {
		struct _stati64 status;
		_wstati64(wpath, &status);
		if (status.st_mode & S_IFREG) {
			return true;
		}
	}
	return false;
}
bool LocalFileSystem::IsPipe(const string &filename) {
	auto unicode_path = WindowsUtil::UTF8ToUnicode(filename.c_str());
	const wchar_t *wpath = unicode_path.c_str();
	if (_waccess(wpath, 0) == 0) {
		struct _stati64 status;
		_wstati64(wpath, &status);
		if (status.st_mode & _S_IFCHR) {
			return true;
		}
	}
	return false;
}
#endif

#ifndef _WIN32
// somehow sometimes this is missing
#ifndef O_CLOEXEC
#define O_CLOEXEC 0
#endif

// Solaris
#ifndef O_DIRECT
#define O_DIRECT 0
#endif

struct UnixFileHandle : public FileHandle {
public:
	UnixFileHandle(FileSystem &file_system, string path, int fd) : FileHandle(file_system, std::move(path)), fd(fd) {
	}
	~UnixFileHandle() override {
		UnixFileHandle::Close();
	}

	int fd;

public:
	void Close() override {
		if (fd != -1) {
			close(fd);
			fd = -1;
		}
	};
};

static FileType GetFileTypeInternal(int fd) { // LCOV_EXCL_START
	struct stat s;
	if (fstat(fd, &s) == -1) {
		return FileType::FILE_TYPE_INVALID;
	}
	switch (s.st_mode & S_IFMT) {
	case S_IFBLK:
		return FileType::FILE_TYPE_BLOCKDEV;
	case S_IFCHR:
		return FileType::FILE_TYPE_CHARDEV;
	case S_IFIFO:
		return FileType::FILE_TYPE_FIFO;
	case S_IFDIR:
		return FileType::FILE_TYPE_DIR;
	case S_IFLNK:
		return FileType::FILE_TYPE_LINK;
	case S_IFREG:
		return FileType::FILE_TYPE_REGULAR;
	case S_IFSOCK:
		return FileType::FILE_TYPE_SOCKET;
	default:
		return FileType::FILE_TYPE_INVALID;
	}
} // LCOV_EXCL_STOP

unique_ptr<FileHandle> LocalFileSystem::OpenFile(const string &path_p, uint8_t flags, FileLockType lock_type,
                                                 FileCompressionType compression, FileOpener *opener) {
	auto path = FileSystem::ExpandPath(path_p, opener);
	if (compression != FileCompressionType::UNCOMPRESSED) {
		throw NotImplementedException("Unsupported compression type for default file system");
	}

	AssertValidFileFlags(flags);

	int open_flags = 0;
	int rc;
	bool open_read = flags & FileFlags::FILE_FLAGS_READ;
	bool open_write = flags & FileFlags::FILE_FLAGS_WRITE;
	if (open_read && open_write) {
		open_flags = O_RDWR;
	} else if (open_read) {
		open_flags = O_RDONLY;
	} else if (open_write) {
		open_flags = O_WRONLY;
	} else {
		throw InternalException("READ, WRITE or both should be specified when opening a file");
	}
	if (open_write) {
		// need Read or Write
		D_ASSERT(flags & FileFlags::FILE_FLAGS_WRITE);
		open_flags |= O_CLOEXEC;
		if (flags & FileFlags::FILE_FLAGS_FILE_CREATE) {
			open_flags |= O_CREAT;
		} else if (flags & FileFlags::FILE_FLAGS_FILE_CREATE_NEW) {
			open_flags |= O_CREAT | O_TRUNC;
		}
		if (flags & FileFlags::FILE_FLAGS_APPEND) {
			open_flags |= O_APPEND;
		}
	}
	if (flags & FileFlags::FILE_FLAGS_DIRECT_IO) {
#if defined(__sun) && defined(__SVR4)
		throw Exception("DIRECT_IO not supported on Solaris");
#endif
#if defined(__DARWIN__) || defined(__APPLE__) || defined(__OpenBSD__)
		// OSX does not have O_DIRECT, instead we need to use fcntl afterwards to support direct IO
		open_flags |= O_SYNC;
#else
		open_flags |= O_DIRECT | O_SYNC;
#endif
	}
	int fd = open(path.c_str(), open_flags, 0666);
	if (fd == -1) {
		throw IOException("Cannot open file \"%s\": %s", path, strerror(errno));
	}
	// #if defined(__DARWIN__) || defined(__APPLE__)
	// 	if (flags & FileFlags::FILE_FLAGS_DIRECT_IO) {
	// 		// OSX requires fcntl for Direct IO
	// 		rc = fcntl(fd, F_NOCACHE, 1);
	// 		if (fd == -1) {
	// 			throw IOException("Could not enable direct IO for file \"%s\": %s", path, strerror(errno));
	// 		}
	// 	}
	// #endif
	if (lock_type != FileLockType::NO_LOCK) {
		// set lock on file
		// but only if it is not an input/output stream
		auto file_type = GetFileTypeInternal(fd);
		if (file_type != FileType::FILE_TYPE_FIFO && file_type != FileType::FILE_TYPE_SOCKET) {
			struct flock fl;
			memset(&fl, 0, sizeof fl);
			fl.l_type = lock_type == FileLockType::READ_LOCK ? F_RDLCK : F_WRLCK;
			fl.l_whence = SEEK_SET;
			fl.l_start = 0;
			fl.l_len = 0;
			rc = fcntl(fd, F_SETLK, &fl);
			if (rc == -1) {
				throw IOException("Could not set lock on file \"%s\": %s", path, strerror(errno));
			}
		}
	}
	return make_uniq<UnixFileHandle>(*this, path, fd);
}

void LocalFileSystem::SetFilePointer(FileHandle &handle, idx_t location) {
	int fd = ((UnixFileHandle &)handle).fd;
	off_t offset = lseek(fd, location, SEEK_SET);
	if (offset == (off_t)-1) {
		throw IOException("Could not seek to location %lld for file \"%s\": %s", location, handle.path,
		                  strerror(errno));
	}
}

idx_t LocalFileSystem::GetFilePointer(FileHandle &handle) {
	int fd = ((UnixFileHandle &)handle).fd;
	off_t position = lseek(fd, 0, SEEK_CUR);
	if (position == (off_t)-1) {
		throw IOException("Could not get file position file \"%s\": %s", handle.path, strerror(errno));
	}
	return position;
}

void LocalFileSystem::Read(FileHandle &handle, void *buffer, int64_t nr_bytes, idx_t location) {
	int fd = ((UnixFileHandle &)handle).fd;
	int64_t bytes_read = pread(fd, buffer, nr_bytes, location);
	if (bytes_read == -1) {
		throw IOException("Could not read from file \"%s\": %s", handle.path, strerror(errno));
	}
	if (bytes_read != nr_bytes) {
		throw IOException("Could not read all bytes from file \"%s\": wanted=%lld read=%lld", handle.path, nr_bytes,
		                  bytes_read);
	}
}

int64_t LocalFileSystem::Read(FileHandle &handle, void *buffer, int64_t nr_bytes) {
	int fd = ((UnixFileHandle &)handle).fd;
	int64_t bytes_read = read(fd, buffer, nr_bytes);
	if (bytes_read == -1) {
		throw IOException("Could not read from file \"%s\": %s", handle.path, strerror(errno));
	}
	return bytes_read;
}

void LocalFileSystem::Write(FileHandle &handle, void *buffer, int64_t nr_bytes, idx_t location) {
	int fd = ((UnixFileHandle &)handle).fd;
	int64_t bytes_written = pwrite(fd, buffer, nr_bytes, location);
	if (bytes_written == -1) {
		throw IOException("Could not write file \"%s\": %s", handle.path, strerror(errno));
	}
	if (bytes_written != nr_bytes) {
		throw IOException("Could not write all bytes to file \"%s\": wanted=%lld wrote=%lld", handle.path, nr_bytes,
		                  bytes_written);
	}
}

int64_t LocalFileSystem::Write(FileHandle &handle, void *buffer, int64_t nr_bytes) {
	int fd = ((UnixFileHandle &)handle).fd;
	int64_t bytes_written = write(fd, buffer, nr_bytes);
	if (bytes_written == -1) {
		throw IOException("Could not write file \"%s\": %s", handle.path, strerror(errno));
	}
	return bytes_written;
}

int64_t LocalFileSystem::GetFileSize(FileHandle &handle) {
	int fd = ((UnixFileHandle &)handle).fd;
	struct stat s;
	if (fstat(fd, &s) == -1) {
		return -1;
	}
	return s.st_size;
}

time_t LocalFileSystem::GetLastModifiedTime(FileHandle &handle) {
	int fd = ((UnixFileHandle &)handle).fd;
	struct stat s;
	if (fstat(fd, &s) == -1) {
		return -1;
	}
	return s.st_mtime;
}

FileType LocalFileSystem::GetFileType(FileHandle &handle) {
	int fd = ((UnixFileHandle &)handle).fd;
	return GetFileTypeInternal(fd);
}

void LocalFileSystem::Truncate(FileHandle &handle, int64_t new_size) {
	int fd = ((UnixFileHandle &)handle).fd;
	if (ftruncate(fd, new_size) != 0) {
		throw IOException("Could not truncate file \"%s\": %s", handle.path, strerror(errno));
	}
}

bool LocalFileSystem::DirectoryExists(const string &directory) {
	if (!directory.empty()) {
		if (access(directory.c_str(), 0) == 0) {
			struct stat status;
			stat(directory.c_str(), &status);
			if (status.st_mode & S_IFDIR) {
				return true;
			}
		}
	}
	// if any condition fails
	return false;
}

void LocalFileSystem::CreateDirectory(const string &directory) {
	struct stat st;

	if (stat(directory.c_str(), &st) != 0) {
		/* Directory does not exist. EEXIST for race condition */
		if (mkdir(directory.c_str(), 0755) != 0 && errno != EEXIST) {
			throw IOException("Failed to create directory \"%s\"!", directory);
		}
	} else if (!S_ISDIR(st.st_mode)) {
		throw IOException("Failed to create directory \"%s\": path exists but is not a directory!", directory);
	}
}

int RemoveDirectoryRecursive(const char *path) {
	DIR *d = opendir(path);
	idx_t path_len = (idx_t)strlen(path);
	int r = -1;

	if (d) {
		struct dirent *p;
		r = 0;
		while (!r && (p = readdir(d))) {
			int r2 = -1;
			char *buf;
			idx_t len;
			/* Skip the names "." and ".." as we don't want to recurse on them. */
			if (!strcmp(p->d_name, ".") || !strcmp(p->d_name, "..")) {
				continue;
			}
			len = path_len + (idx_t)strlen(p->d_name) + 2;
			buf = new char[len];
			if (buf) {
				struct stat statbuf;
				snprintf(buf, len, "%s/%s", path, p->d_name);
				if (!stat(buf, &statbuf)) {
					if (S_ISDIR(statbuf.st_mode)) {
						r2 = RemoveDirectoryRecursive(buf);
					} else {
						r2 = unlink(buf);
					}
				}
				delete[] buf;
			}
			r = r2;
		}
		closedir(d);
	}
	if (!r) {
		r = rmdir(path);
	}
	return r;
}

void LocalFileSystem::RemoveDirectory(const string &directory) {
	RemoveDirectoryRecursive(directory.c_str());
}

void LocalFileSystem::RemoveFile(const string &filename) {
	if (std::remove(filename.c_str()) != 0) {
		throw IOException("Could not remove file \"%s\": %s", filename, strerror(errno));
	}
}

bool LocalFileSystem::ListFiles(const string &directory, const std::function<void(const string &, bool)> &callback,
                                FileOpener *opener) {
	if (!DirectoryExists(directory)) {
		return false;
	}
	DIR *dir = opendir(directory.c_str());
	if (!dir) {
		return false;
	}
	struct dirent *ent;
	// loop over all files in the directory
	while ((ent = readdir(dir)) != nullptr) {
		string name = string(ent->d_name);
		// skip . .. and empty files
		if (name.empty() || name == "." || name == "..") {
			continue;
		}
		// now stat the file to figure out if it is a regular file or directory
		string full_path = JoinPath(directory, name);
		if (access(full_path.c_str(), 0) != 0) {
			continue;
		}
		struct stat status;
		stat(full_path.c_str(), &status);
		if (!(status.st_mode & S_IFREG) && !(status.st_mode & S_IFDIR)) {
			// not a file or directory: skip
			continue;
		}
		// invoke callback
		callback(name, status.st_mode & S_IFDIR);
	}
	closedir(dir);
	return true;
}

void LocalFileSystem::FileSync(FileHandle &handle) {
	int fd = ((UnixFileHandle &)handle).fd;
	if (fsync(fd) != 0) {
		throw FatalException("fsync failed!");
	}
}

void LocalFileSystem::MoveFile(const string &source, const string &target) {
	//! FIXME: rename does not guarantee atomicity or overwriting target file if it exists
	if (rename(source.c_str(), target.c_str()) != 0) {
		throw IOException("Could not rename file!");
	}
}

std::string LocalFileSystem::GetLastErrorAsString() {
	return string();
}

#else

constexpr char PIPE_PREFIX[] = "\\\\.\\pipe\\";

// Returns the last Win32 error, in string format. Returns an empty string if there is no error.
std::string LocalFileSystem::GetLastErrorAsString() {
	// Get the error message, if any.
	DWORD errorMessageID = GetLastError();
	if (errorMessageID == 0)
		return std::string(); // No error message has been recorded

	LPSTR messageBuffer = nullptr;
	idx_t size =
	    FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
	                   NULL, errorMessageID, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&messageBuffer, 0, NULL);

	std::string message(messageBuffer, size);

	// Free the buffer.
	LocalFree(messageBuffer);

	return message;
}

struct WindowsFileHandle : public FileHandle {
public:
	WindowsFileHandle(FileSystem &file_system, string path, HANDLE fd)
	    : FileHandle(file_system, path), position(0), fd(fd) {
	}
	~WindowsFileHandle() override {
		Close();
	}

	idx_t position;
	HANDLE fd;

public:
	void Close() override {
		if (!fd) {
			return;
		}
		CloseHandle(fd);
		fd = nullptr;
	};
};

unique_ptr<FileHandle> LocalFileSystem::OpenFile(const string &path_p, uint8_t flags, FileLockType lock_type,
                                                 FileCompressionType compression, FileOpener *opener) {
	auto path = FileSystem::ExpandPath(path_p, opener);
	if (compression != FileCompressionType::UNCOMPRESSED) {
		throw NotImplementedException("Unsupported compression type for default file system");
	}
	AssertValidFileFlags(flags);

	DWORD desired_access;
	DWORD share_mode;
	DWORD creation_disposition = OPEN_EXISTING;
	DWORD flags_and_attributes = FILE_ATTRIBUTE_NORMAL;
	bool open_read = flags & FileFlags::FILE_FLAGS_READ;
	bool open_write = flags & FileFlags::FILE_FLAGS_WRITE;
	if (open_read && open_write) {
		desired_access = GENERIC_READ | GENERIC_WRITE;
		share_mode = 0;
	} else if (open_read) {
		desired_access = GENERIC_READ;
		share_mode = FILE_SHARE_READ;
	} else if (open_write) {
		desired_access = GENERIC_WRITE;
		share_mode = 0;
	} else {
		throw InternalException("READ, WRITE or both should be specified when opening a file");
	}
	if (open_write) {
		if (flags & FileFlags::FILE_FLAGS_FILE_CREATE) {
			creation_disposition = OPEN_ALWAYS;
		} else if (flags & FileFlags::FILE_FLAGS_FILE_CREATE_NEW) {
			creation_disposition = CREATE_ALWAYS;
		}
	}
	if (flags & FileFlags::FILE_FLAGS_DIRECT_IO) {
		flags_and_attributes |= FILE_FLAG_NO_BUFFERING;
	}
	auto unicode_path = WindowsUtil::UTF8ToUnicode(path.c_str());
	HANDLE hFile = CreateFileW(unicode_path.c_str(), desired_access, share_mode, NULL, creation_disposition,
	                           flags_and_attributes, NULL);
	if (hFile == INVALID_HANDLE_VALUE) {
		auto error = LocalFileSystem::GetLastErrorAsString();
		throw IOException("Cannot open file \"%s\": %s", path.c_str(), error);
	}
	auto handle = make_uniq<WindowsFileHandle>(*this, path.c_str(), hFile);
	if (flags & FileFlags::FILE_FLAGS_APPEND) {
		auto file_size = GetFileSize(*handle);
		SetFilePointer(*handle, file_size);
	}
	return std::move(handle);
}

void LocalFileSystem::SetFilePointer(FileHandle &handle, idx_t location) {
	auto &whandle = (WindowsFileHandle &)handle;
	whandle.position = location;
	LARGE_INTEGER wlocation;
	wlocation.QuadPart = location;
	SetFilePointerEx(whandle.fd, wlocation, NULL, FILE_BEGIN);
}

idx_t LocalFileSystem::GetFilePointer(FileHandle &handle) {
	return ((WindowsFileHandle &)handle).position;
}

static DWORD FSInternalRead(FileHandle &handle, HANDLE hFile, void *buffer, int64_t nr_bytes, idx_t location) {
	DWORD bytes_read = 0;
	OVERLAPPED ov = {};
	ov.Internal = 0;
	ov.InternalHigh = 0;
	ov.Offset = location & 0xFFFFFFFF;
	ov.OffsetHigh = location >> 32;
	ov.hEvent = 0;
	auto rc = ReadFile(hFile, buffer, (DWORD)nr_bytes, &bytes_read, &ov);
	if (!rc) {
		auto error = LocalFileSystem::GetLastErrorAsString();
		throw IOException("Could not read file \"%s\" (error in ReadFile(location: %llu, nr_bytes: %lld)): %s",
		                  handle.path, location, nr_bytes, error);
	}
	return bytes_read;
}

void LocalFileSystem::Read(FileHandle &handle, void *buffer, int64_t nr_bytes, idx_t location) {
	HANDLE hFile = ((WindowsFileHandle &)handle).fd;
	auto bytes_read = FSInternalRead(handle, hFile, buffer, nr_bytes, location);
	if (bytes_read != nr_bytes) {
		throw IOException("Could not read all bytes from file \"%s\": wanted=%lld read=%lld", handle.path, nr_bytes,
		                  bytes_read);
	}
}

int64_t LocalFileSystem::Read(FileHandle &handle, void *buffer, int64_t nr_bytes) {
	HANDLE hFile = ((WindowsFileHandle &)handle).fd;
	auto &pos = ((WindowsFileHandle &)handle).position;
	auto n = std::min<idx_t>(std::max<idx_t>(GetFileSize(handle), pos) - pos, nr_bytes);
	auto bytes_read = FSInternalRead(handle, hFile, buffer, n, pos);
	pos += bytes_read;
	return bytes_read;
}

static DWORD FSInternalWrite(FileHandle &handle, HANDLE hFile, void *buffer, int64_t nr_bytes, idx_t location) {
	DWORD bytes_written = 0;
	OVERLAPPED ov = {};
	ov.Internal = 0;
	ov.InternalHigh = 0;
	ov.Offset = location & 0xFFFFFFFF;
	ov.OffsetHigh = location >> 32;
	ov.hEvent = 0;
	auto rc = WriteFile(hFile, buffer, (DWORD)nr_bytes, &bytes_written, &ov);
	if (!rc) {
		auto error = LocalFileSystem::GetLastErrorAsString();
		throw IOException("Could not write file \"%s\" (error in WriteFile): %s", handle.path, error);
	}
	return bytes_written;
}

void LocalFileSystem::Write(FileHandle &handle, void *buffer, int64_t nr_bytes, idx_t location) {
	HANDLE hFile = ((WindowsFileHandle &)handle).fd;
	auto bytes_written = FSInternalWrite(handle, hFile, buffer, nr_bytes, location);
	if (bytes_written != nr_bytes) {
		throw IOException("Could not write all bytes from file \"%s\": wanted=%lld wrote=%lld", handle.path, nr_bytes,
		                  bytes_written);
	}
}

int64_t LocalFileSystem::Write(FileHandle &handle, void *buffer, int64_t nr_bytes) {
	HANDLE hFile = ((WindowsFileHandle &)handle).fd;
	auto &pos = ((WindowsFileHandle &)handle).position;
	auto bytes_written = FSInternalWrite(handle, hFile, buffer, nr_bytes, pos);
	pos += bytes_written;
	return bytes_written;
}

int64_t LocalFileSystem::GetFileSize(FileHandle &handle) {
	HANDLE hFile = ((WindowsFileHandle &)handle).fd;
	LARGE_INTEGER result;
	if (!GetFileSizeEx(hFile, &result)) {
		return -1;
	}
	return result.QuadPart;
}

time_t LocalFileSystem::GetLastModifiedTime(FileHandle &handle) {
	HANDLE hFile = ((WindowsFileHandle &)handle).fd;

	// https://docs.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-getfiletime
	FILETIME last_write;
	if (GetFileTime(hFile, nullptr, nullptr, &last_write) == 0) {
		return -1;
	}

	// https://stackoverflow.com/questions/29266743/what-is-dwlowdatetime-and-dwhighdatetime
	ULARGE_INTEGER ul;
	ul.LowPart = last_write.dwLowDateTime;
	ul.HighPart = last_write.dwHighDateTime;
	int64_t fileTime64 = ul.QuadPart;

	// fileTime64 contains a 64-bit value representing the number of
	// 100-nanosecond intervals since January 1, 1601 (UTC).
	// https://docs.microsoft.com/en-us/windows/win32/api/minwinbase/ns-minwinbase-filetime

	// Adapted from: https://stackoverflow.com/questions/6161776/convert-windows-filetime-to-second-in-unix-linux
	const auto WINDOWS_TICK = 10000000;
	const auto SEC_TO_UNIX_EPOCH = 11644473600LL;
	time_t result = (fileTime64 / WINDOWS_TICK - SEC_TO_UNIX_EPOCH);
	return result;
}

void LocalFileSystem::Truncate(FileHandle &handle, int64_t new_size) {
	HANDLE hFile = ((WindowsFileHandle &)handle).fd;
	// seek to the location
	SetFilePointer(handle, new_size);
	// now set the end of file position
	if (!SetEndOfFile(hFile)) {
		auto error = LocalFileSystem::GetLastErrorAsString();
		throw IOException("Failure in SetEndOfFile call on file \"%s\": %s", handle.path, error);
	}
}

static DWORD WindowsGetFileAttributes(const string &filename) {
	auto unicode_path = WindowsUtil::UTF8ToUnicode(filename.c_str());
	return GetFileAttributesW(unicode_path.c_str());
}

bool LocalFileSystem::DirectoryExists(const string &directory) {
	DWORD attrs = WindowsGetFileAttributes(directory);
	return (attrs != INVALID_FILE_ATTRIBUTES && (attrs & FILE_ATTRIBUTE_DIRECTORY));
}

void LocalFileSystem::CreateDirectory(const string &directory) {
	if (DirectoryExists(directory)) {
		return;
	}
	auto unicode_path = WindowsUtil::UTF8ToUnicode(directory.c_str());
	if (directory.empty() || !CreateDirectoryW(unicode_path.c_str(), NULL) || !DirectoryExists(directory)) {
		throw IOException("Could not create directory!");
	}
}

static void DeleteDirectoryRecursive(FileSystem &fs, string directory) {
	fs.ListFiles(directory, [&](const string &fname, bool is_directory) {
		if (is_directory) {
			DeleteDirectoryRecursive(fs, fs.JoinPath(directory, fname));
		} else {
			fs.RemoveFile(fs.JoinPath(directory, fname));
		}
	});
	auto unicode_path = WindowsUtil::UTF8ToUnicode(directory.c_str());
	if (!RemoveDirectoryW(unicode_path.c_str())) {
		auto error = LocalFileSystem::GetLastErrorAsString();
		throw IOException("Failed to delete directory \"%s\": %s", directory, error);
	}
}

void LocalFileSystem::RemoveDirectory(const string &directory) {
	if (FileExists(directory)) {
		throw IOException("Attempting to delete directory \"%s\", but it is a file and not a directory!", directory);
	}
	if (!DirectoryExists(directory)) {
		return;
	}
	DeleteDirectoryRecursive(*this, directory.c_str());
}

void LocalFileSystem::RemoveFile(const string &filename) {
	auto unicode_path = WindowsUtil::UTF8ToUnicode(filename.c_str());
	if (!DeleteFileW(unicode_path.c_str())) {
		auto error = LocalFileSystem::GetLastErrorAsString();
		throw IOException("Failed to delete file \"%s\": %s", filename, error);
	}
}

bool LocalFileSystem::ListFiles(const string &directory, const std::function<void(const string &, bool)> &callback,
                                FileOpener *opener) {
	string search_dir = JoinPath(directory, "*");

	auto unicode_path = WindowsUtil::UTF8ToUnicode(search_dir.c_str());

	WIN32_FIND_DATAW ffd;
	HANDLE hFind = FindFirstFileW(unicode_path.c_str(), &ffd);
	if (hFind == INVALID_HANDLE_VALUE) {
		return false;
	}
	do {
		string cFileName = WindowsUtil::UnicodeToUTF8(ffd.cFileName);
		if (cFileName == "." || cFileName == "..") {
			continue;
		}
		callback(cFileName, ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY);
	} while (FindNextFileW(hFind, &ffd) != 0);

	DWORD dwError = GetLastError();
	if (dwError != ERROR_NO_MORE_FILES) {
		FindClose(hFind);
		return false;
	}

	FindClose(hFind);
	return true;
}

void LocalFileSystem::FileSync(FileHandle &handle) {
	HANDLE hFile = ((WindowsFileHandle &)handle).fd;
	if (FlushFileBuffers(hFile) == 0) {
		throw IOException("Could not flush file handle to disk!");
	}
}

void LocalFileSystem::MoveFile(const string &source, const string &target) {
	auto source_unicode = WindowsUtil::UTF8ToUnicode(source.c_str());
	auto target_unicode = WindowsUtil::UTF8ToUnicode(target.c_str());
	if (!MoveFileW(source_unicode.c_str(), target_unicode.c_str())) {
		throw IOException("Could not move file: %s", GetLastErrorAsString());
	}
}

FileType LocalFileSystem::GetFileType(FileHandle &handle) {
	auto path = ((WindowsFileHandle &)handle).path;
	// pipes in windows are just files in '\\.\pipe\' folder
	if (strncmp(path.c_str(), PIPE_PREFIX, strlen(PIPE_PREFIX)) == 0) {
		return FileType::FILE_TYPE_FIFO;
	}
	DWORD attrs = WindowsGetFileAttributes(path.c_str());
	if (attrs != INVALID_FILE_ATTRIBUTES) {
		if (attrs & FILE_ATTRIBUTE_DIRECTORY) {
			return FileType::FILE_TYPE_DIR;
		} else {
			return FileType::FILE_TYPE_REGULAR;
		}
	}
	return FileType::FILE_TYPE_INVALID;
}
#endif

bool LocalFileSystem::CanSeek() {
	return true;
}

bool LocalFileSystem::OnDiskFile(FileHandle &handle) {
	return true;
}

void LocalFileSystem::Seek(FileHandle &handle, idx_t location) {
	if (!CanSeek()) {
		throw IOException("Cannot seek in files of this type");
	}
	SetFilePointer(handle, location);
}

idx_t LocalFileSystem::SeekPosition(FileHandle &handle) {
	if (!CanSeek()) {
		throw IOException("Cannot seek in files of this type");
	}
	return GetFilePointer(handle);
}

static bool IsCrawl(const string &glob) {
	// glob must match exactly
	return glob == "**";
}
static bool HasMultipleCrawl(const vector<string> &splits) {
	return std::count(splits.begin(), splits.end(), "**") > 1;
}
static bool IsSymbolicLink(const string &path) {
#ifndef _WIN32
	struct stat status;
	return (lstat(path.c_str(), &status) != -1 && S_ISLNK(status.st_mode));
#else
	auto attributes = WindowsGetFileAttributes(path);
	if (attributes == INVALID_FILE_ATTRIBUTES)
		return false;
	return attributes & FILE_ATTRIBUTE_REPARSE_POINT;
#endif
}

static void RecursiveGlobDirectories(FileSystem &fs, const string &path, vector<string> &result, bool match_directory,
                                     bool join_path) {

	fs.ListFiles(path, [&](const string &fname, bool is_directory) {
		string concat;
		if (join_path) {
			concat = fs.JoinPath(path, fname);
		} else {
			concat = fname;
		}
		if (IsSymbolicLink(concat)) {
			return;
		}
		if (is_directory == match_directory) {
			result.push_back(concat);
		}
		if (is_directory) {
			RecursiveGlobDirectories(fs, concat, result, match_directory, true);
		}
	});
}

static void GlobFilesInternal(FileSystem &fs, const string &path, const string &glob, bool match_directory,
                              vector<string> &result, bool join_path) {
	fs.ListFiles(path, [&](const string &fname, bool is_directory) {
		if (is_directory != match_directory) {
			return;
		}
		if (LikeFun::Glob(fname.c_str(), fname.size(), glob.c_str(), glob.size())) {
			if (join_path) {
				result.push_back(fs.JoinPath(path, fname));
			} else {
				result.push_back(fname);
			}
		}
	});
}

vector<string> LocalFileSystem::FetchFileWithoutGlob(const string &path, FileOpener *opener, bool absolute_path) {
	vector<string> result;
	if (FileExists(path) || IsPipe(path)) {
		result.push_back(path);
	} else if (!absolute_path) {
		Value value;
		if (opener && opener->TryGetCurrentSetting("file_search_path", value)) {
			auto search_paths_str = value.ToString();
			vector<std::string> search_paths = StringUtil::Split(search_paths_str, ',');
			for (const auto &search_path : search_paths) {
				auto joined_path = JoinPath(search_path, path);
				if (FileExists(joined_path) || IsPipe(joined_path)) {
					result.push_back(joined_path);
				}
			}
		}
	}
	return result;
}

vector<string> LocalFileSystem::Glob(const string &path, FileOpener *opener) {
	if (path.empty()) {
		return vector<string>();
	}
	// split up the path into separate chunks
	vector<string> splits;
	idx_t last_pos = 0;
	for (idx_t i = 0; i < path.size(); i++) {
		if (path[i] == '\\' || path[i] == '/') {
			if (i == last_pos) {
				// empty: skip this position
				last_pos = i + 1;
				continue;
			}
			if (splits.empty()) {
				splits.push_back(path.substr(0, i));
			} else {
				splits.push_back(path.substr(last_pos, i - last_pos));
			}
			last_pos = i + 1;
		}
	}
	splits.push_back(path.substr(last_pos, path.size() - last_pos));
	// handle absolute paths
	bool absolute_path = false;
	if (path[0] == '/') {
		// first character is a slash -  unix absolute path
		absolute_path = true;
	} else if (StringUtil::Contains(splits[0], ":")) {
		// first split has a colon -  windows absolute path
		absolute_path = true;
	} else if (splits[0] == "~") {
		// starts with home directory
		auto home_directory = GetHomeDirectory(opener);
		if (!home_directory.empty()) {
			absolute_path = true;
			splits[0] = home_directory;
			D_ASSERT(path[0] == '~');
			if (!HasGlob(path)) {
				return Glob(home_directory + path.substr(1));
			}
		}
	}
	// Check if the path has a glob at all
	if (!HasGlob(path)) {
		// no glob: return only the file (if it exists or is a pipe)
		return FetchFileWithoutGlob(path, opener, absolute_path);
	}
	vector<string> previous_directories;
	if (absolute_path) {
		// for absolute paths, we don't start by scanning the current directory
		previous_directories.push_back(splits[0]);
	} else {
		// If file_search_path is set, use those paths as the first glob elements
		Value value;
		if (opener && opener->TryGetCurrentSetting("file_search_path", value)) {
			auto search_paths_str = value.ToString();
			vector<std::string> search_paths = StringUtil::Split(search_paths_str, ',');
			for (const auto &search_path : search_paths) {
				previous_directories.push_back(search_path);
			}
		}
	}

	if (HasMultipleCrawl(splits)) {
		throw IOException("Cannot use multiple \'**\' in one path");
	}

	for (idx_t i = absolute_path ? 1 : 0; i < splits.size(); i++) {
		bool is_last_chunk = i + 1 == splits.size();
		bool has_glob = HasGlob(splits[i]);
		// if it's the last chunk we need to find files, otherwise we find directories
		// not the last chunk: gather a list of all directories that match the glob pattern
		vector<string> result;
		if (!has_glob) {
			// no glob, just append as-is
			if (previous_directories.empty()) {
				result.push_back(splits[i]);
			} else {
				if (is_last_chunk) {
					for (auto &prev_directory : previous_directories) {
						const string filename = JoinPath(prev_directory, splits[i]);
						if (FileExists(filename) || DirectoryExists(filename)) {
							result.push_back(filename);
						}
					}
				} else {
					for (auto &prev_directory : previous_directories) {
						result.push_back(JoinPath(prev_directory, splits[i]));
					}
				}
			}
		} else {
			if (IsCrawl(splits[i])) {
				if (!is_last_chunk) {
					result = previous_directories;
				}
				if (previous_directories.empty()) {
					RecursiveGlobDirectories(*this, ".", result, !is_last_chunk, false);
				} else {
					for (auto &prev_dir : previous_directories) {
						RecursiveGlobDirectories(*this, prev_dir, result, !is_last_chunk, true);
					}
				}
			} else {
				if (previous_directories.empty()) {
					// no previous directories: list in the current path
					GlobFilesInternal(*this, ".", splits[i], !is_last_chunk, result, false);
				} else {
					// previous directories
					// we iterate over each of the previous directories, and apply the glob of the current directory
					for (auto &prev_directory : previous_directories) {
						GlobFilesInternal(*this, prev_directory, splits[i], !is_last_chunk, result, true);
					}
				}
			}
		}
		if (result.empty()) {
			// no result found that matches the glob
			// last ditch effort: search the path as a string literal
			return FetchFileWithoutGlob(path, opener, absolute_path);
		}
		if (is_last_chunk) {
			return result;
		}
		previous_directories = std::move(result);
	}
	return vector<string>();
}

unique_ptr<FileSystem> FileSystem::CreateLocal() {
	return make_uniq<LocalFileSystem>();
}

} // namespace duckdb









namespace duckdb {

void MultiFileReader::AddParameters(TableFunction &table_function) {
	table_function.named_parameters["filename"] = LogicalType::BOOLEAN;
	table_function.named_parameters["hive_partitioning"] = LogicalType::BOOLEAN;
	table_function.named_parameters["union_by_name"] = LogicalType::BOOLEAN;
}

vector<string> MultiFileReader::GetFileList(ClientContext &context, const Value &input, const string &name,
                                            FileGlobOptions options) {
	auto &config = DBConfig::GetConfig(context);
	if (!config.options.enable_external_access) {
		throw PermissionException("Scanning %s files is disabled through configuration", name);
	}
	if (input.IsNull()) {
		throw ParserException("%s reader cannot take NULL list as parameter", name);
	}
	FileSystem &fs = FileSystem::GetFileSystem(context);
	vector<string> files;
	if (input.type().id() == LogicalTypeId::VARCHAR) {
		auto file_name = StringValue::Get(input);
		files = fs.GlobFiles(file_name, context, options);
	} else if (input.type().id() == LogicalTypeId::LIST) {
		for (auto &val : ListValue::GetChildren(input)) {
			if (val.IsNull()) {
				throw ParserException("%s reader cannot take NULL input as parameter", name);
			}
			auto glob_files = fs.GlobFiles(StringValue::Get(val), context, options);
			files.insert(files.end(), glob_files.begin(), glob_files.end());
		}
	} else {
		throw InternalException("Unsupported type for MultiFileReader::GetFileList");
	}
	if (files.empty() && options == FileGlobOptions::DISALLOW_EMPTY) {
		throw IOException("%s reader needs at least one file to read", name);
	}
	return files;
}

bool MultiFileReader::ParseOption(const string &key, const Value &val, MultiFileReaderOptions &options) {
	auto loption = StringUtil::Lower(key);
	if (loption == "filename") {
		options.filename = BooleanValue::Get(val);
	} else if (loption == "hive_partitioning") {
		options.hive_partitioning = BooleanValue::Get(val);
		options.auto_detect_hive_partitioning = false;
	} else if (loption == "union_by_name") {
		options.union_by_name = BooleanValue::Get(val);
	} else {
		return false;
	}
	return true;
}

bool MultiFileReader::ComplexFilterPushdown(ClientContext &context, vector<string> &files,
                                            const MultiFileReaderOptions &options, LogicalGet &get,
                                            vector<unique_ptr<Expression>> &filters) {
	if (files.empty()) {
		return false;
	}
	if (!options.hive_partitioning && !options.filename) {
		return false;
	}

	unordered_map<string, column_t> column_map;
	for (idx_t i = 0; i < get.column_ids.size(); i++) {
		column_map.insert({get.names[get.column_ids[i]], i});
	}

	auto start_files = files.size();
	HivePartitioning::ApplyFiltersToFileList(context, files, filters, column_map, get.table_index,
	                                         options.hive_partitioning, options.filename);
	if (files.size() != start_files) {
		// we have pruned files
		return true;
	}
	return false;
}

MultiFileReaderBindData MultiFileReader::BindOptions(MultiFileReaderOptions &options, const vector<string> &files,
                                                     vector<LogicalType> &return_types, vector<string> &names) {
	MultiFileReaderBindData bind_data;
	// Add generated constant column for filename
	if (options.filename) {
		if (std::find(names.begin(), names.end(), "filename") != names.end()) {
			throw BinderException("Using filename option on file with column named filename is not supported");
		}
		bind_data.filename_idx = names.size();
		return_types.emplace_back(LogicalType::VARCHAR);
		names.emplace_back("filename");
	}

	// Add generated constant columns from hive partitioning scheme
	if (options.hive_partitioning) {
		D_ASSERT(!files.empty());
		auto partitions = HivePartitioning::Parse(files[0]);
		// verify that all files have the same hive partitioning scheme
		for (auto &f : files) {
			auto file_partitions = HivePartitioning::Parse(f);
			for (auto &part_info : partitions) {
				if (file_partitions.find(part_info.first) == file_partitions.end()) {
					if (options.auto_detect_hive_partitioning == true) {
						throw BinderException(
						    "Hive partitioning was enabled automatically, but an error was encountered: Hive partition "
						    "mismatch between file \"%s\" and \"%s\": key \"%s\" not found\n\nTo switch off hive "
						    "partition, set: HIVE_PARTITIONING=0",
						    files[0], f, part_info.first);
					}
					throw BinderException(
					    "Hive partition mismatch between file \"%s\" and \"%s\": key \"%s\" not found", files[0], f,
					    part_info.first);
				}
			}
			if (partitions.size() != file_partitions.size()) {
				if (options.auto_detect_hive_partitioning == true) {
					throw BinderException("Hive partitioning was enabled automatically, but an error was encountered: "
					                      "Hive partition mismatch between file \"%s\" and \"%s\"\n\nTo switch off "
					                      "hive partition, set: HIVE_PARTITIONING=0",
					                      files[0], f);
				}
				throw BinderException("Hive partition mismatch between file \"%s\" and \"%s\"", files[0], f);
			}
		}
		for (auto &part : partitions) {
			idx_t hive_partitioning_index = DConstants::INVALID_INDEX;
			auto lookup = std::find(names.begin(), names.end(), part.first);
			if (lookup != names.end()) {
				// hive partitioning column also exists in file - override
				auto idx = lookup - names.begin();
				hive_partitioning_index = idx;
				return_types[idx] = LogicalType::VARCHAR;
			} else {
				// hive partitioning column does not exist in file - add a new column containing the key
				hive_partitioning_index = names.size();
				return_types.emplace_back(LogicalType::VARCHAR);
				names.emplace_back(part.first);
			}
			bind_data.hive_partitioning_indexes.emplace_back(part.first, hive_partitioning_index);
		}
	}
	return bind_data;
}

void MultiFileReader::FinalizeBind(const MultiFileReaderOptions &file_options, const MultiFileReaderBindData &options,
                                   const string &filename, const vector<string> &local_names,
                                   const vector<LogicalType> &global_types, const vector<string> &global_names,
                                   const vector<column_t> &global_column_ids, MultiFileReaderData &reader_data) {
	// create a map of name -> column index
	case_insensitive_map_t<idx_t> name_map;
	if (file_options.union_by_name) {
		for (idx_t col_idx = 0; col_idx < local_names.size(); col_idx++) {
			name_map[local_names[col_idx]] = col_idx;
		}
	}
	for (idx_t i = 0; i < global_column_ids.size(); i++) {
		auto column_id = global_column_ids[i];
		if (IsRowIdColumnId(column_id)) {
			// row-id
			reader_data.constant_map.emplace_back(i, Value::BIGINT(42));
			continue;
		}
		if (column_id == options.filename_idx) {
			// filename
			reader_data.constant_map.emplace_back(i, Value(filename));
			continue;
		}
		if (!options.hive_partitioning_indexes.empty()) {
			// hive partition constants
			auto partitions = HivePartitioning::Parse(filename);
			D_ASSERT(partitions.size() == options.hive_partitioning_indexes.size());
			bool found_partition = false;
			for (auto &entry : options.hive_partitioning_indexes) {
				if (column_id == entry.index) {
					reader_data.constant_map.emplace_back(i, Value(partitions[entry.value]));
					found_partition = true;
					break;
				}
			}
			if (found_partition) {
				continue;
			}
		}
		if (file_options.union_by_name) {
			auto &global_name = global_names[column_id];
			auto entry = name_map.find(global_name);
			bool not_present_in_file = entry == name_map.end();
			if (not_present_in_file) {
				// we need to project a column with name \"global_name\" - but it does not exist in the current file
				// push a NULL value of the specified type
				reader_data.constant_map.emplace_back(i, Value(global_types[column_id]));
				continue;
			}
		}
	}
}

void MultiFileReader::CreateNameMapping(const string &file_name, const vector<LogicalType> &local_types,
                                        const vector<string> &local_names, const vector<LogicalType> &global_types,
                                        const vector<string> &global_names, const vector<column_t> &global_column_ids,
                                        MultiFileReaderData &reader_data, const string &initial_file) {
	D_ASSERT(global_types.size() == global_names.size());
	D_ASSERT(local_types.size() == local_names.size());
	// we have expected types: create a map of name -> column index
	case_insensitive_map_t<idx_t> name_map;
	for (idx_t col_idx = 0; col_idx < local_names.size(); col_idx++) {
		name_map[local_names[col_idx]] = col_idx;
	}
	for (idx_t i = 0; i < global_column_ids.size(); i++) {
		// check if this is a constant column
		bool constant = false;
		for (auto &entry : reader_data.constant_map) {
			if (entry.column_id == i) {
				constant = true;
				break;
			}
		}
		if (constant) {
			// this column is constant for this file
			continue;
		}
		// not constant - look up the column in the name map
		auto global_id = global_column_ids[i];
		if (global_id >= global_types.size()) {
			throw InternalException(
			    "MultiFileReader::CreatePositionalMapping - global_id is out of range in global_types for this file");
		}
		auto &global_name = global_names[global_id];
		auto entry = name_map.find(global_name);
		if (entry == name_map.end()) {
			string candidate_names;
			for (auto &local_name : local_names) {
				if (!candidate_names.empty()) {
					candidate_names += ", ";
				}
				candidate_names += local_name;
			}
			throw IOException(
			    StringUtil::Format("Failed to read file \"%s\": schema mismatch in glob: column \"%s\" was read from "
			                       "the original file \"%s\", but could not be found in file \"%s\".\nCandidate names: "
			                       "%s\nIf you are trying to "
			                       "read files with different schemas, try setting union_by_name=True",
			                       file_name, global_name, initial_file, file_name, candidate_names));
		}
		// we found the column in the local file - check if the types are the same
		auto local_id = entry->second;
		D_ASSERT(global_id < global_types.size());
		D_ASSERT(local_id < local_types.size());
		auto &global_type = global_types[global_id];
		auto &local_type = local_types[local_id];
		if (global_type != local_type) {
			reader_data.cast_map[local_id] = global_type;
		}
		// the types are the same - create the mapping
		reader_data.column_mapping.push_back(i);
		reader_data.column_ids.push_back(local_id);
	}
	reader_data.empty_columns = reader_data.column_ids.empty();
}

void MultiFileReader::CreateMapping(const string &file_name, const vector<LogicalType> &local_types,
                                    const vector<string> &local_names, const vector<LogicalType> &global_types,
                                    const vector<string> &global_names, const vector<column_t> &global_column_ids,
                                    optional_ptr<TableFilterSet> filters, MultiFileReaderData &reader_data,
                                    const string &initial_file) {
	CreateNameMapping(file_name, local_types, local_names, global_types, global_names, global_column_ids, reader_data,
	                  initial_file);
	if (filters) {
		reader_data.filter_map.resize(global_types.size());
		for (idx_t c = 0; c < reader_data.column_mapping.size(); c++) {
			auto map_index = reader_data.column_mapping[c];
			reader_data.filter_map[map_index].index = c;
			reader_data.filter_map[map_index].is_constant = false;
		}
		for (idx_t c = 0; c < reader_data.constant_map.size(); c++) {
			auto constant_index = reader_data.constant_map[c].column_id;
			reader_data.filter_map[constant_index].index = c;
			reader_data.filter_map[constant_index].is_constant = true;
		}
	}
}

void MultiFileReader::FinalizeChunk(const MultiFileReaderBindData &bind_data, const MultiFileReaderData &reader_data,
                                    DataChunk &chunk) {
	// reference all the constants set up in MultiFileReader::FinalizeBind
	for (auto &entry : reader_data.constant_map) {
		chunk.data[entry.column_id].Reference(entry.value);
	}
	chunk.Verify();
}

TableFunctionSet MultiFileReader::CreateFunctionSet(TableFunction table_function) {
	TableFunctionSet function_set(table_function.name);
	function_set.AddFunction(table_function);
	D_ASSERT(table_function.arguments.size() == 1 && table_function.arguments[0] == LogicalType::VARCHAR);
	table_function.arguments[0] = LogicalType::LIST(LogicalType::VARCHAR);
	function_set.AddFunction(std::move(table_function));
	return function_set;
}

void MultiFileReaderOptions::Serialize(Serializer &serializer) const {
	FieldWriter writer(serializer);
	writer.WriteField<bool>(filename);
	writer.WriteField<bool>(hive_partitioning);
	writer.WriteField<bool>(union_by_name);
	writer.Finalize();
}

MultiFileReaderOptions MultiFileReaderOptions::Deserialize(Deserializer &source) {
	MultiFileReaderOptions result;
	FieldReader reader(source);
	result.filename = reader.ReadRequired<bool>();
	result.hive_partitioning = reader.ReadRequired<bool>();
	result.union_by_name = reader.ReadRequired<bool>();
	reader.Finalize();
	return result;
}

void MultiFileReaderBindData::Serialize(Serializer &serializer) const {
	FieldWriter writer(serializer);
	writer.WriteField(filename_idx);
	writer.WriteRegularSerializableList<HivePartitioningIndex>(hive_partitioning_indexes);
	writer.Finalize();
}

MultiFileReaderBindData MultiFileReaderBindData::Deserialize(Deserializer &source) {
	MultiFileReaderBindData result;
	FieldReader reader(source);
	result.filename_idx = reader.ReadRequired<idx_t>();
	result.hive_partitioning_indexes =
	    reader.ReadRequiredSerializableList<HivePartitioningIndex, HivePartitioningIndex>();
	reader.Finalize();
	return result;
}

HivePartitioningIndex::HivePartitioningIndex(string value_p, idx_t index) : value(std::move(value_p)), index(index) {
}

void HivePartitioningIndex::Serialize(Serializer &serializer) const {
	FieldWriter writer(serializer);
	writer.WriteString(value);
	writer.WriteField<idx_t>(index);
	writer.Finalize();
}

HivePartitioningIndex HivePartitioningIndex::Deserialize(Deserializer &source) {
	FieldReader reader(source);
	auto value = reader.ReadRequired<string>();
	auto index = reader.ReadRequired<idx_t>();
	reader.Finalize();
	return HivePartitioningIndex(std::move(value), index);
}

void MultiFileReaderOptions::AddBatchInfo(BindInfo &bind_info) const {
	bind_info.InsertOption("filename", Value::BOOLEAN(filename));
	bind_info.InsertOption("hive_partitioning", Value::BOOLEAN(hive_partitioning));
	bind_info.InsertOption("union_by_name", Value::BOOLEAN(union_by_name));
}

void UnionByName::CombineUnionTypes(const vector<string> &col_names, const vector<LogicalType> &sql_types,
                                    vector<LogicalType> &union_col_types, vector<string> &union_col_names,
                                    case_insensitive_map_t<idx_t> &union_names_map) {
	D_ASSERT(col_names.size() == sql_types.size());

	for (idx_t col = 0; col < col_names.size(); ++col) {
		auto union_find = union_names_map.find(col_names[col]);

		if (union_find != union_names_map.end()) {
			// given same name , union_col's type must compatible with col's type
			auto &current_type = union_col_types[union_find->second];
			LogicalType compatible_type;
			compatible_type = LogicalType::MaxLogicalType(current_type, sql_types[col]);
			union_col_types[union_find->second] = compatible_type;
		} else {
			union_names_map[col_names[col]] = union_col_names.size();
			union_col_names.emplace_back(col_names[col]);
			union_col_types.emplace_back(sql_types[col]);
		}
	}
}

} // namespace duckdb
























#include <cctype>
#include <cmath>
#include <cstdlib>

namespace duckdb {

//===--------------------------------------------------------------------===//
// Cast bool -> Numeric
//===--------------------------------------------------------------------===//
template <>
bool TryCast::Operation(bool input, bool &result, bool strict) {
	return NumericTryCast::Operation<bool, bool>(input, result, strict);
}

template <>
bool TryCast::Operation(bool input, int8_t &result, bool strict) {
	return NumericTryCast::Operation<bool, int8_t>(input, result, strict);
}

template <>
bool TryCast::Operation(bool input, int16_t &result, bool strict) {
	return NumericTryCast::Operation<bool, int16_t>(input, result, strict);
}

template <>
bool TryCast::Operation(bool input, int32_t &result, bool strict) {
	return NumericTryCast::Operation<bool, int32_t>(input, result, strict);
}

template <>
bool TryCast::Operation(bool input, int64_t &result, bool strict) {
	return NumericTryCast::Operation<bool, int64_t>(input, result, strict);
}

template <>
bool TryCast::Operation(bool input, hugeint_t &result, bool strict) {
	return NumericTryCast::Operation<bool, hugeint_t>(input, result, strict);
}

template <>
bool TryCast::Operation(bool input, uint8_t &result, bool strict) {
	return NumericTryCast::Operation<bool, uint8_t>(input, result, strict);
}

template <>
bool TryCast::Operation(bool input, uint16_t &result, bool strict) {
	return NumericTryCast::Operation<bool, uint16_t>(input, result, strict);
}

template <>
bool TryCast::Operation(bool input, uint32_t &result, bool strict) {
	return NumericTryCast::Operation<bool, uint32_t>(input, result, strict);
}

template <>
bool TryCast::Operation(bool input, uint64_t &result, bool strict) {
	return NumericTryCast::Operation<bool, uint64_t>(input, result, strict);
}

template <>
bool TryCast::Operation(bool input, float &result, bool strict) {
	return NumericTryCast::Operation<bool, float>(input, result, strict);
}

template <>
bool TryCast::Operation(bool input, double &result, bool strict) {
	return NumericTryCast::Operation<bool, double>(input, result, strict);
}

//===--------------------------------------------------------------------===//
// Cast int8_t -> Numeric
//===--------------------------------------------------------------------===//
template <>
bool TryCast::Operation(int8_t input, bool &result, bool strict) {
	return NumericTryCast::Operation<int8_t, bool>(input, result, strict);
}

template <>
bool TryCast::Operation(int8_t input, int8_t &result, bool strict) {
	return NumericTryCast::Operation<int8_t, int8_t>(input, result, strict);
}

template <>
bool TryCast::Operation(int8_t input, int16_t &result, bool strict) {
	return NumericTryCast::Operation<int8_t, int16_t>(input, result, strict);
}

template <>
bool TryCast::Operation(int8_t input, int32_t &result, bool strict) {
	return NumericTryCast::Operation<int8_t, int32_t>(input, result, strict);
}

template <>
bool TryCast::Operation(int8_t input, int64_t &result, bool strict) {
	return NumericTryCast::Operation<int8_t, int64_t>(input, result, strict);
}

template <>
bool TryCast::Operation(int8_t input, hugeint_t &result, bool strict) {
	return NumericTryCast::Operation<int8_t, hugeint_t>(input, result, strict);
}

template <>
bool TryCast::Operation(int8_t input, uint8_t &result, bool strict) {
	return NumericTryCast::Operation<int8_t, uint8_t>(input, result, strict);
}

template <>
bool TryCast::Operation(int8_t input, uint16_t &result, bool strict) {
	return NumericTryCast::Operation<int8_t, uint16_t>(input, result, strict);
}

template <>
bool TryCast::Operation(int8_t input, uint32_t &result, bool strict) {
	return NumericTryCast::Operation<int8_t, uint32_t>(input, result, strict);
}

template <>
bool TryCast::Operation(int8_t input, uint64_t &result, bool strict) {
	return NumericTryCast::Operation<int8_t, uint64_t>(input, result, strict);
}

template <>
bool TryCast::Operation(int8_t input, float &result, bool strict) {
	return NumericTryCast::Operation<int8_t, float>(input, result, strict);
}

template <>
bool TryCast::Operation(int8_t input, double &result, bool strict) {
	return NumericTryCast::Operation<int8_t, double>(input, result, strict);
}

//===--------------------------------------------------------------------===//
// Cast int16_t -> Numeric
//===--------------------------------------------------------------------===//
template <>
bool TryCast::Operation(int16_t input, bool &result, bool strict) {
	return NumericTryCast::Operation<int16_t, bool>(input, result, strict);
}

template <>
bool TryCast::Operation(int16_t input, int8_t &result, bool strict) {
	return NumericTryCast::Operation<int16_t, int8_t>(input, result, strict);
}

template <>
bool TryCast::Operation(int16_t input, int16_t &result, bool strict) {
	return NumericTryCast::Operation<int16_t, int16_t>(input, result, strict);
}

template <>
bool TryCast::Operation(int16_t input, int32_t &result, bool strict) {
	return NumericTryCast::Operation<int16_t, int32_t>(input, result, strict);
}

template <>
bool TryCast::Operation(int16_t input, int64_t &result, bool strict) {
	return NumericTryCast::Operation<int16_t, int64_t>(input, result, strict);
}

template <>
bool TryCast::Operation(int16_t input, hugeint_t &result, bool strict) {
	return NumericTryCast::Operation<int16_t, hugeint_t>(input, result, strict);
}

template <>
bool TryCast::Operation(int16_t input, uint8_t &result, bool strict) {
	return NumericTryCast::Operation<int16_t, uint8_t>(input, result, strict);
}

template <>
bool TryCast::Operation(int16_t input, uint16_t &result, bool strict) {
	return NumericTryCast::Operation<int16_t, uint16_t>(input, result, strict);
}

template <>
bool TryCast::Operation(int16_t input, uint32_t &result, bool strict) {
	return NumericTryCast::Operation<int16_t, uint32_t>(input, result, strict);
}

template <>
bool TryCast::Operation(int16_t input, uint64_t &result, bool strict) {
	return NumericTryCast::Operation<int16_t, uint64_t>(input, result, strict);
}

template <>
bool TryCast::Operation(int16_t input, float &result, bool strict) {
	return NumericTryCast::Operation<int16_t, float>(input, result, strict);
}

template <>
bool TryCast::Operation(int16_t input, double &result, bool strict) {
	return NumericTryCast::Operation<int16_t, double>(input, result, strict);
}

//===--------------------------------------------------------------------===//
// Cast int32_t -> Numeric
//===--------------------------------------------------------------------===//
template <>
bool TryCast::Operation(int32_t input, bool &result, bool strict) {
	return NumericTryCast::Operation<int32_t, bool>(input, result, strict);
}

template <>
bool TryCast::Operation(int32_t input, int8_t &result, bool strict) {
	return NumericTryCast::Operation<int32_t, int8_t>(input, result, strict);
}

template <>
bool TryCast::Operation(int32_t input, int16_t &result, bool strict) {
	return NumericTryCast::Operation<int32_t, int16_t>(input, result, strict);
}

template <>
bool TryCast::Operation(int32_t input, int32_t &result, bool strict) {
	return NumericTryCast::Operation<int32_t, int32_t>(input, result, strict);
}

template <>
bool TryCast::Operation(int32_t input, int64_t &result, bool strict) {
	return NumericTryCast::Operation<int32_t, int64_t>(input, result, strict);
}

template <>
bool TryCast::Operation(int32_t input, hugeint_t &result, bool strict) {
	return NumericTryCast::Operation<int32_t, hugeint_t>(input, result, strict);
}

template <>
bool TryCast::Operation(int32_t input, uint8_t &result, bool strict) {
	return NumericTryCast::Operation<int32_t, uint8_t>(input, result, strict);
}

template <>
bool TryCast::Operation(int32_t input, uint16_t &result, bool strict) {
	return NumericTryCast::Operation<int32_t, uint16_t>(input, result, strict);
}

template <>
bool TryCast::Operation(int32_t input, uint32_t &result, bool strict) {
	return NumericTryCast::Operation<int32_t, uint32_t>(input, result, strict);
}

template <>
bool TryCast::Operation(int32_t input, uint64_t &result, bool strict) {
	return NumericTryCast::Operation<int32_t, uint64_t>(input, result, strict);
}

template <>
bool TryCast::Operation(int32_t input, float &result, bool strict) {
	return NumericTryCast::Operation<int32_t, float>(input, result, strict);
}

template <>
bool TryCast::Operation(int32_t input, double &result, bool strict) {
	return NumericTryCast::Operation<int32_t, double>(input, result, strict);
}

//===--------------------------------------------------------------------===//
// Cast int64_t -> Numeric
//===--------------------------------------------------------------------===//
template <>
bool TryCast::Operation(int64_t input, bool &result, bool strict) {
	return NumericTryCast::Operation<int64_t, bool>(input, result, strict);
}

template <>
bool TryCast::Operation(int64_t input, int8_t &result, bool strict) {
	return NumericTryCast::Operation<int64_t, int8_t>(input, result, strict);
}

template <>
bool TryCast::Operation(int64_t input, int16_t &result, bool strict) {
	return NumericTryCast::Operation<int64_t, int16_t>(input, result, strict);
}

template <>
bool TryCast::Operation(int64_t input, int32_t &result, bool strict) {
	return NumericTryCast::Operation<int64_t, int32_t>(input, result, strict);
}

template <>
bool TryCast::Operation(int64_t input, int64_t &result, bool strict) {
	return NumericTryCast::Operation<int64_t, int64_t>(input, result, strict);
}

template <>
bool TryCast::Operation(int64_t input, hugeint_t &result, bool strict) {
	return NumericTryCast::Operation<int64_t, hugeint_t>(input, result, strict);
}

template <>
bool TryCast::Operation(int64_t input, uint8_t &result, bool strict) {
	return NumericTryCast::Operation<int64_t, uint8_t>(input, result, strict);
}

template <>
bool TryCast::Operation(int64_t input, uint16_t &result, bool strict) {
	return NumericTryCast::Operation<int64_t, uint16_t>(input, result, strict);
}

template <>
bool TryCast::Operation(int64_t input, uint32_t &result, bool strict) {
	return NumericTryCast::Operation<int64_t, uint32_t>(input, result, strict);
}

template <>
bool TryCast::Operation(int64_t input, uint64_t &result, bool strict) {
	return NumericTryCast::Operation<int64_t, uint64_t>(input, result, strict);
}

template <>
bool TryCast::Operation(int64_t input, float &result, bool strict) {
	return NumericTryCast::Operation<int64_t, float>(input, result, strict);
}

template <>
bool TryCast::Operation(int64_t input, double &result, bool strict) {
	return NumericTryCast::Operation<int64_t, double>(input, result, strict);
}

//===--------------------------------------------------------------------===//
// Cast hugeint_t -> Numeric
//===--------------------------------------------------------------------===//
template <>
bool TryCast::Operation(hugeint_t input, bool &result, bool strict) {
	return NumericTryCast::Operation<hugeint_t, bool>(input, result, strict);
}

template <>
bool TryCast::Operation(hugeint_t input, int8_t &result, bool strict) {
	return NumericTryCast::Operation<hugeint_t, int8_t>(input, result, strict);
}

template <>
bool TryCast::Operation(hugeint_t input, int16_t &result, bool strict) {
	return NumericTryCast::Operation<hugeint_t, int16_t>(input, result, strict);
}

template <>
bool TryCast::Operation(hugeint_t input, int32_t &result, bool strict) {
	return NumericTryCast::Operation<hugeint_t, int32_t>(input, result, strict);
}

template <>
bool TryCast::Operation(hugeint_t input, int64_t &result, bool strict) {
	return NumericTryCast::Operation<hugeint_t, int64_t>(input, result, strict);
}

template <>
bool TryCast::Operation(hugeint_t input, hugeint_t &result, bool strict) {
	return NumericTryCast::Operation<hugeint_t, hugeint_t>(input, result, strict);
}

template <>
bool TryCast::Operation(hugeint_t input, uint8_t &result, bool strict) {
	return NumericTryCast::Operation<hugeint_t, uint8_t>(input, result, strict);
}

template <>
bool TryCast::Operation(hugeint_t input, uint16_t &result, bool strict) {
	return NumericTryCast::Operation<hugeint_t, uint16_t>(input, result, strict);
}

template <>
bool TryCast::Operation(hugeint_t input, uint32_t &result, bool strict) {
	return NumericTryCast::Operation<hugeint_t, uint32_t>(input, result, strict);
}

template <>
bool TryCast::Operation(hugeint_t input, uint64_t &result, bool strict) {
	return NumericTryCast::Operation<hugeint_t, uint64_t>(input, result, strict);
}

template <>
bool TryCast::Operation(hugeint_t input, float &result, bool strict) {
	return NumericTryCast::Operation<hugeint_t, float>(input, result, strict);
}

template <>
bool TryCast::Operation(hugeint_t input, double &result, bool strict) {
	return NumericTryCast::Operation<hugeint_t, double>(input, result, strict);
}

//===--------------------------------------------------------------------===//
// Cast uint8_t -> Numeric
//===--------------------------------------------------------------------===//
template <>
bool TryCast::Operation(uint8_t input, bool &result, bool strict) {
	return NumericTryCast::Operation<uint8_t, bool>(input, result, strict);
}

template <>
bool TryCast::Operation(uint8_t input, int8_t &result, bool strict) {
	return NumericTryCast::Operation<uint8_t, int8_t>(input, result, strict);
}

template <>
bool TryCast::Operation(uint8_t input, int16_t &result, bool strict) {
	return NumericTryCast::Operation<uint8_t, int16_t>(input, result, strict);
}

template <>
bool TryCast::Operation(uint8_t input, int32_t &result, bool strict) {
	return NumericTryCast::Operation<uint8_t, int32_t>(input, result, strict);
}

template <>
bool TryCast::Operation(uint8_t input, int64_t &result, bool strict) {
	return NumericTryCast::Operation<uint8_t, int64_t>(input, result, strict);
}

template <>
bool TryCast::Operation(uint8_t input, hugeint_t &result, bool strict) {
	return NumericTryCast::Operation<uint8_t, hugeint_t>(input, result, strict);
}

template <>
bool TryCast::Operation(uint8_t input, uint8_t &result, bool strict) {
	return NumericTryCast::Operation<uint8_t, uint8_t>(input, result, strict);
}

template <>
bool TryCast::Operation(uint8_t input, uint16_t &result, bool strict) {
	return NumericTryCast::Operation<uint8_t, uint16_t>(input, result, strict);
}

template <>
bool TryCast::Operation(uint8_t input, uint32_t &result, bool strict) {
	return NumericTryCast::Operation<uint8_t, uint32_t>(input, result, strict);
}

template <>
bool TryCast::Operation(uint8_t input, uint64_t &result, bool strict) {
	return NumericTryCast::Operation<uint8_t, uint64_t>(input, result, strict);
}

template <>
bool TryCast::Operation(uint8_t input, float &result, bool strict) {
	return NumericTryCast::Operation<uint8_t, float>(input, result, strict);
}

template <>
bool TryCast::Operation(uint8_t input, double &result, bool strict) {
	return NumericTryCast::Operation<uint8_t, double>(input, result, strict);
}

//===--------------------------------------------------------------------===//
// Cast uint16_t -> Numeric
//===--------------------------------------------------------------------===//
template <>
bool TryCast::Operation(uint16_t input, bool &result, bool strict) {
	return NumericTryCast::Operation<uint16_t, bool>(input, result, strict);
}

template <>
bool TryCast::Operation(uint16_t input, int8_t &result, bool strict) {
	return NumericTryCast::Operation<uint16_t, int8_t>(input, result, strict);
}

template <>
bool TryCast::Operation(uint16_t input, int16_t &result, bool strict) {
	return NumericTryCast::Operation<uint16_t, int16_t>(input, result, strict);
}

template <>
bool TryCast::Operation(uint16_t input, int32_t &result, bool strict) {
	return NumericTryCast::Operation<uint16_t, int32_t>(input, result, strict);
}

template <>
bool TryCast::Operation(uint16_t input, int64_t &result, bool strict) {
	return NumericTryCast::Operation<uint16_t, int64_t>(input, result, strict);
}

template <>
bool TryCast::Operation(uint16_t input, hugeint_t &result, bool strict) {
	return NumericTryCast::Operation<uint16_t, hugeint_t>(input, result, strict);
}

template <>
bool TryCast::Operation(uint16_t input, uint8_t &result, bool strict) {
	return NumericTryCast::Operation<uint16_t, uint8_t>(input, result, strict);
}

template <>
bool TryCast::Operation(uint16_t input, uint16_t &result, bool strict) {
	return NumericTryCast::Operation<uint16_t, uint16_t>(input, result, strict);
}

template <>
bool TryCast::Operation(uint16_t input, uint32_t &result, bool strict) {
	return NumericTryCast::Operation<uint16_t, uint32_t>(input, result, strict);
}

template <>
bool TryCast::Operation(uint16_t input, uint64_t &result, bool strict) {
	return NumericTryCast::Operation<uint16_t, uint64_t>(input, result, strict);
}

template <>
bool TryCast::Operation(uint16_t input, float &result, bool strict) {
	return NumericTryCast::Operation<uint16_t, float>(input, result, strict);
}

template <>
bool TryCast::Operation(uint16_t input, double &result, bool strict) {
	return NumericTryCast::Operation<uint16_t, double>(input, result, strict);
}

//===--------------------------------------------------------------------===//
// Cast uint32_t -> Numeric
//===--------------------------------------------------------------------===//
template <>
bool TryCast::Operation(uint32_t input, bool &result, bool strict) {
	return NumericTryCast::Operation<uint32_t, bool>(input, result, strict);
}

template <>
bool TryCast::Operation(uint32_t input, int8_t &result, bool strict) {
	return NumericTryCast::Operation<uint32_t, int8_t>(input, result, strict);
}

template <>
bool TryCast::Operation(uint32_t input, int16_t &result, bool strict) {
	return NumericTryCast::Operation<uint32_t, int16_t>(input, result, strict);
}

template <>
bool TryCast::Operation(uint32_t input, int32_t &result, bool strict) {
	return NumericTryCast::Operation<uint32_t, int32_t>(input, result, strict);
}

template <>
bool TryCast::Operation(uint32_t input, int64_t &result, bool strict) {
	return NumericTryCast::Operation<uint32_t, int64_t>(input, result, strict);
}

template <>
bool TryCast::Operation(uint32_t input, hugeint_t &result, bool strict) {
	return NumericTryCast::Operation<uint32_t, hugeint_t>(input, result, strict);
}

template <>
bool TryCast::Operation(uint32_t input, uint8_t &result, bool strict) {
	return NumericTryCast::Operation<uint32_t, uint8_t>(input, result, strict);
}

template <>
bool TryCast::Operation(uint32_t input, uint16_t &result, bool strict) {
	return NumericTryCast::Operation<uint32_t, uint16_t>(input, result, strict);
}

template <>
bool TryCast::Operation(uint32_t input, uint32_t &result, bool strict) {
	return NumericTryCast::Operation<uint32_t, uint32_t>(input, result, strict);
}

template <>
bool TryCast::Operation(uint32_t input, uint64_t &result, bool strict) {
	return NumericTryCast::Operation<uint32_t, uint64_t>(input, result, strict);
}

template <>
bool TryCast::Operation(uint32_t input, float &result, bool strict) {
	return NumericTryCast::Operation<uint32_t, float>(input, result, strict);
}

template <>
bool TryCast::Operation(uint32_t input, double &result, bool strict) {
	return NumericTryCast::Operation<uint32_t, double>(input, result, strict);
}

//===--------------------------------------------------------------------===//
// Cast uint64_t -> Numeric
//===--------------------------------------------------------------------===//
template <>
bool TryCast::Operation(uint64_t input, bool &result, bool strict) {
	return NumericTryCast::Operation<uint64_t, bool>(input, result, strict);
}

template <>
bool TryCast::Operation(uint64_t input, int8_t &result, bool strict) {
	return NumericTryCast::Operation<uint64_t, int8_t>(input, result, strict);
}

template <>
bool TryCast::Operation(uint64_t input, int16_t &result, bool strict) {
	return NumericTryCast::Operation<uint64_t, int16_t>(input, result, strict);
}

template <>
bool TryCast::Operation(uint64_t input, int32_t &result, bool strict) {
	return NumericTryCast::Operation<uint64_t, int32_t>(input, result, strict);
}

template <>
bool TryCast::Operation(uint64_t input, int64_t &result, bool strict) {
	return NumericTryCast::Operation<uint64_t, int64_t>(input, result, strict);
}

template <>
bool TryCast::Operation(uint64_t input, hugeint_t &result, bool strict) {
	return NumericTryCast::Operation<uint64_t, hugeint_t>(input, result, strict);
}

template <>
bool TryCast::Operation(uint64_t input, uint8_t &result, bool strict) {
	return NumericTryCast::Operation<uint64_t, uint8_t>(input, result, strict);
}

template <>
bool TryCast::Operation(uint64_t input, uint16_t &result, bool strict) {
	return NumericTryCast::Operation<uint64_t, uint16_t>(input, result, strict);
}

template <>
bool TryCast::Operation(uint64_t input, uint32_t &result, bool strict) {
	return NumericTryCast::Operation<uint64_t, uint32_t>(input, result, strict);
}

template <>
bool TryCast::Operation(uint64_t input, uint64_t &result, bool strict) {
	return NumericTryCast::Operation<uint64_t, uint64_t>(input, result, strict);
}

template <>
bool TryCast::Operation(uint64_t input, float &result, bool strict) {
	return NumericTryCast::Operation<uint64_t, float>(input, result, strict);
}

template <>
bool TryCast::Operation(uint64_t input, double &result, bool strict) {
	return NumericTryCast::Operation<uint64_t, double>(input, result, strict);
}

//===--------------------------------------------------------------------===//
// Cast float -> Numeric
//===--------------------------------------------------------------------===//
template <>
bool TryCast::Operation(float input, bool &result, bool strict) {
	return NumericTryCast::Operation<float, bool>(input, result, strict);
}

template <>
bool TryCast::Operation(float input, int8_t &result, bool strict) {
	return NumericTryCast::Operation<float, int8_t>(input, result, strict);
}

template <>
bool TryCast::Operation(float input, int16_t &result, bool strict) {
	return NumericTryCast::Operation<float, int16_t>(input, result, strict);
}

template <>
bool TryCast::Operation(float input, int32_t &result, bool strict) {
	return NumericTryCast::Operation<float, int32_t>(input, result, strict);
}

template <>
bool TryCast::Operation(float input, int64_t &result, bool strict) {
	return NumericTryCast::Operation<float, int64_t>(input, result, strict);
}

template <>
bool TryCast::Operation(float input, hugeint_t &result, bool strict) {
	return NumericTryCast::Operation<float, hugeint_t>(input, result, strict);
}

template <>
bool TryCast::Operation(float input, uint8_t &result, bool strict) {
	return NumericTryCast::Operation<float, uint8_t>(input, result, strict);
}

template <>
bool TryCast::Operation(float input, uint16_t &result, bool strict) {
	return NumericTryCast::Operation<float, uint16_t>(input, result, strict);
}

template <>
bool TryCast::Operation(float input, uint32_t &result, bool strict) {
	return NumericTryCast::Operation<float, uint32_t>(input, result, strict);
}

template <>
bool TryCast::Operation(float input, uint64_t &result, bool strict) {
	return NumericTryCast::Operation<float, uint64_t>(input, result, strict);
}

template <>
bool TryCast::Operation(float input, float &result, bool strict) {
	return NumericTryCast::Operation<float, float>(input, result, strict);
}

template <>
bool TryCast::Operation(float input, double &result, bool strict) {
	return NumericTryCast::Operation<float, double>(input, result, strict);
}

//===--------------------------------------------------------------------===//
// Cast double -> Numeric
//===--------------------------------------------------------------------===//
template <>
bool TryCast::Operation(double input, bool &result, bool strict) {
	return NumericTryCast::Operation<double, bool>(input, result, strict);
}

template <>
bool TryCast::Operation(double input, int8_t &result, bool strict) {
	return NumericTryCast::Operation<double, int8_t>(input, result, strict);
}

template <>
bool TryCast::Operation(double input, int16_t &result, bool strict) {
	return NumericTryCast::Operation<double, int16_t>(input, result, strict);
}

template <>
bool TryCast::Operation(double input, int32_t &result, bool strict) {
	return NumericTryCast::Operation<double, int32_t>(input, result, strict);
}

template <>
bool TryCast::Operation(double input, int64_t &result, bool strict) {
	return NumericTryCast::Operation<double, int64_t>(input, result, strict);
}

template <>
bool TryCast::Operation(double input, hugeint_t &result, bool strict) {
	return NumericTryCast::Operation<double, hugeint_t>(input, result, strict);
}

template <>
bool TryCast::Operation(double input, uint8_t &result, bool strict) {
	return NumericTryCast::Operation<double, uint8_t>(input, result, strict);
}

template <>
bool TryCast::Operation(double input, uint16_t &result, bool strict) {
	return NumericTryCast::Operation<double, uint16_t>(input, result, strict);
}

template <>
bool TryCast::Operation(double input, uint32_t &result, bool strict) {
	return NumericTryCast::Operation<double, uint32_t>(input, result, strict);
}

template <>
bool TryCast::Operation(double input, uint64_t &result, bool strict) {
	return NumericTryCast::Operation<double, uint64_t>(input, result, strict);
}

template <>
bool TryCast::Operation(double input, float &result, bool strict) {
	return NumericTryCast::Operation<double, float>(input, result, strict);
}

template <>
bool TryCast::Operation(double input, double &result, bool strict) {
	return NumericTryCast::Operation<double, double>(input, result, strict);
}

//===--------------------------------------------------------------------===//
// Cast String -> Numeric
//===--------------------------------------------------------------------===//
template <typename T>
struct IntegerCastData {
	using Result = T;
	Result result;
	bool seen_decimal;
};

struct IntegerCastOperation {
	template <class T, bool NEGATIVE>
	static bool HandleDigit(T &state, uint8_t digit) {
		using result_t = typename T::Result;
		if (NEGATIVE) {
			if (state.result < (NumericLimits<result_t>::Minimum() + digit) / 10) {
				return false;
			}
			state.result = state.result * 10 - digit;
		} else {
			if (state.result > (NumericLimits<result_t>::Maximum() - digit) / 10) {
				return false;
			}
			state.result = state.result * 10 + digit;
		}
		return true;
	}

	template <class T, bool NEGATIVE>
	static bool HandleHexDigit(T &state, uint8_t digit) {
		using result_t = typename T::Result;
		if (state.result > (NumericLimits<result_t>::Maximum() - digit) / 16) {
			return false;
		}
		state.result = state.result * 16 + digit;
		return true;
	}

	template <class T, bool NEGATIVE>
	static bool HandleBinaryDigit(T &state, uint8_t digit) {
		using result_t = typename T::Result;
		if (state.result > (NumericLimits<result_t>::Maximum() - digit) / 2) {
			return false;
		}
		state.result = state.result * 2 + digit;
		return true;
	}

	template <class T, bool NEGATIVE>
	static bool HandleExponent(T &state, int32_t exponent) {
		using result_t = typename T::Result;
		double dbl_res = state.result * std::pow(10.0L, exponent);
		if (dbl_res < (double)NumericLimits<result_t>::Minimum() ||
		    dbl_res > (double)NumericLimits<result_t>::Maximum()) {
			return false;
		}
		state.result = (result_t)std::nearbyint(dbl_res);
		return true;
	}

	template <class T, bool NEGATIVE, bool ALLOW_EXPONENT>
	static bool HandleDecimal(T &state, uint8_t digit) {
		if (state.seen_decimal) {
			return true;
		}
		state.seen_decimal = true;
		// round the integer based on what is after the decimal point
		// if digit >= 5, then we round up (or down in case of negative numbers)
		auto increment = digit >= 5;
		if (!increment) {
			return true;
		}
		if (NEGATIVE) {
			if (state.result == NumericLimits<typename T::Result>::Minimum()) {
				return false;
			}
			state.result--;
		} else {
			if (state.result == NumericLimits<typename T::Result>::Maximum()) {
				return false;
			}
			state.result++;
		}
		return true;
	}

	template <class T, bool NEGATIVE>
	static bool Finalize(T &state) {
		return true;
	}
};

template <class T, bool NEGATIVE, bool ALLOW_EXPONENT, class OP = IntegerCastOperation, char decimal_separator = '.'>
static bool IntegerCastLoop(const char *buf, idx_t len, T &result, bool strict) {
	idx_t start_pos;
	if (NEGATIVE) {
		start_pos = 1;
	} else {
		if (*buf == '+') {
			if (strict) {
				// leading plus is not allowed in strict mode
				return false;
			}
			start_pos = 1;
		} else {
			start_pos = 0;
		}
	}
	idx_t pos = start_pos;
	while (pos < len) {
		if (!StringUtil::CharacterIsDigit(buf[pos])) {
			// not a digit!
			if (buf[pos] == decimal_separator) {
				if (strict) {
					return false;
				}
				bool number_before_period = pos > start_pos;
				// decimal point: we accept decimal values for integers as well
				// we just truncate them
				// make sure everything after the period is a number
				pos++;
				idx_t start_digit = pos;
				while (pos < len) {
					if (!StringUtil::CharacterIsDigit(buf[pos])) {
						break;
					}
					if (!OP::template HandleDecimal<T, NEGATIVE, ALLOW_EXPONENT>(result, buf[pos] - '0')) {
						return false;
					}
					pos++;
				}
				// make sure there is either (1) one number after the period, or (2) one number before the period
				// i.e. we accept "1." and ".1" as valid numbers, but not "."
				if (!(number_before_period || pos > start_digit)) {
					return false;
				}
				if (pos >= len) {
					break;
				}
			}
			if (StringUtil::CharacterIsSpace(buf[pos])) {
				// skip any trailing spaces
				while (++pos < len) {
					if (!StringUtil::CharacterIsSpace(buf[pos])) {
						return false;
					}
				}
				break;
			}
			if (ALLOW_EXPONENT) {
				if (buf[pos] == 'e' || buf[pos] == 'E') {
					if (pos == start_pos) {
						return false;
					}
					pos++;
					if (pos >= len) {
						return false;
					}
					using ExponentData = IntegerCastData<int32_t>;
					ExponentData exponent {0, false};
					int negative = buf[pos] == '-';
					if (negative) {
						if (!IntegerCastLoop<ExponentData, true, false, IntegerCastOperation, decimal_separator>(
						        buf + pos, len - pos, exponent, strict)) {
							return false;
						}
					} else {
						if (!IntegerCastLoop<ExponentData, false, false, IntegerCastOperation, decimal_separator>(
						        buf + pos, len - pos, exponent, strict)) {
							return false;
						}
					}
					return OP::template HandleExponent<T, NEGATIVE>(result, exponent.result);
				}
			}
			return false;
		}
		uint8_t digit = buf[pos++] - '0';
		if (!OP::template HandleDigit<T, NEGATIVE>(result, digit)) {
			return false;
		}
	}
	if (!OP::template Finalize<T, NEGATIVE>(result)) {
		return false;
	}
	return pos > start_pos;
}

template <class T, bool NEGATIVE, bool ALLOW_EXPONENT, class OP = IntegerCastOperation>
static bool IntegerHexCastLoop(const char *buf, idx_t len, T &result, bool strict) {
	if (ALLOW_EXPONENT || NEGATIVE) {
		return false;
	}
	idx_t start_pos = 1;
	idx_t pos = start_pos;
	char current_char;
	while (pos < len) {
		current_char = StringUtil::CharacterToLower(buf[pos]);
		if (!StringUtil::CharacterIsHex(current_char)) {
			return false;
		}
		uint8_t digit;
		if (current_char >= 'a') {
			digit = current_char - 'a' + 10;
		} else {
			digit = current_char - '0';
		}
		pos++;
		if (!OP::template HandleHexDigit<T, NEGATIVE>(result, digit)) {
			return false;
		}
	}
	if (!OP::template Finalize<T, NEGATIVE>(result)) {
		return false;
	}
	return pos > start_pos;
}

template <class T, bool NEGATIVE, bool ALLOW_EXPONENT, class OP = IntegerCastOperation>
static bool IntegerBinaryCastLoop(const char *buf, idx_t len, T &result, bool strict) {
	if (ALLOW_EXPONENT || NEGATIVE) {
		return false;
	}
	idx_t start_pos = 1;
	idx_t pos = start_pos;
	uint8_t digit;
	char current_char;
	while (pos < len) {
		current_char = buf[pos];
		if (current_char == '_' && pos > start_pos) {
			// skip underscore, if it is not the first character
			pos++;
			if (pos == len) {
				// we cant end on an underscore either
				return false;
			}
			continue;
		} else if (current_char == '0') {
			digit = 0;
		} else if (current_char == '1') {
			digit = 1;
		} else {
			return false;
		}
		pos++;
		if (!OP::template HandleBinaryDigit<T, NEGATIVE>(result, digit)) {
			return false;
		}
	}
	if (!OP::template Finalize<T, NEGATIVE>(result)) {
		return false;
	}
	return pos > start_pos;
}

template <class T, bool IS_SIGNED = true, bool ALLOW_EXPONENT = true, class OP = IntegerCastOperation,
          bool ZERO_INITIALIZE = true, char decimal_separator = '.'>
static bool TryIntegerCast(const char *buf, idx_t len, T &result, bool strict) {
	// skip any spaces at the start
	while (len > 0 && StringUtil::CharacterIsSpace(*buf)) {
		buf++;
		len--;
	}
	if (len == 0) {
		return false;
	}
	if (ZERO_INITIALIZE) {
		memset(&result, 0, sizeof(T));
	}
	// if the number is negative, we set the negative flag and skip the negative sign
	if (*buf == '-') {
		if (!IS_SIGNED) {
			// Need to check if its not -0
			idx_t pos = 1;
			while (pos < len) {
				if (buf[pos++] != '0') {
					return false;
				}
			}
		}
		return IntegerCastLoop<T, true, ALLOW_EXPONENT, OP, decimal_separator>(buf, len, result, strict);
	}
	if (len > 1 && *buf == '0') {
		if (buf[1] == 'x' || buf[1] == 'X') {
			// If it starts with 0x or 0X, we parse it as a hex value
			buf++;
			len--;
			return IntegerHexCastLoop<T, false, false, OP>(buf, len, result, strict);
		} else if (buf[1] == 'b' || buf[1] == 'B') {
			// If it starts with 0b or 0B, we parse it as a binary value
			buf++;
			len--;
			return IntegerBinaryCastLoop<T, false, false, OP>(buf, len, result, strict);
		} else if (strict && StringUtil::CharacterIsDigit(buf[1])) {
			// leading zeros are not allowed in strict mode
			return false;
		}
	}
	return IntegerCastLoop<T, false, ALLOW_EXPONENT, OP, decimal_separator>(buf, len, result, strict);
}

template <typename T, bool IS_SIGNED = true>
static inline bool TrySimpleIntegerCast(const char *buf, idx_t len, T &result, bool strict) {
	IntegerCastData<T> data;
	if (TryIntegerCast<IntegerCastData<T>, IS_SIGNED>(buf, len, data, strict)) {
		result = data.result;
		return true;
	}
	return false;
}

template <>
bool TryCast::Operation(string_t input, bool &result, bool strict) {
	auto input_data = input.GetData();
	auto input_size = input.GetSize();

	switch (input_size) {
	case 1: {
		char c = std::tolower(*input_data);
		if (c == 't' || (!strict && c == '1')) {
			result = true;
			return true;
		} else if (c == 'f' || (!strict && c == '0')) {
			result = false;
			return true;
		}
		return false;
	}
	case 4: {
		char t = std::tolower(input_data[0]);
		char r = std::tolower(input_data[1]);
		char u = std::tolower(input_data[2]);
		char e = std::tolower(input_data[3]);
		if (t == 't' && r == 'r' && u == 'u' && e == 'e') {
			result = true;
			return true;
		}
		return false;
	}
	case 5: {
		char f = std::tolower(input_data[0]);
		char a = std::tolower(input_data[1]);
		char l = std::tolower(input_data[2]);
		char s = std::tolower(input_data[3]);
		char e = std::tolower(input_data[4]);
		if (f == 'f' && a == 'a' && l == 'l' && s == 's' && e == 'e') {
			result = false;
			return true;
		}
		return false;
	}
	default:
		return false;
	}
}
template <>
bool TryCast::Operation(string_t input, int8_t &result, bool strict) {
	return TrySimpleIntegerCast<int8_t>(input.GetData(), input.GetSize(), result, strict);
}
template <>
bool TryCast::Operation(string_t input, int16_t &result, bool strict) {
	return TrySimpleIntegerCast<int16_t>(input.GetData(), input.GetSize(), result, strict);
}
template <>
bool TryCast::Operation(string_t input, int32_t &result, bool strict) {
	return TrySimpleIntegerCast<int32_t>(input.GetData(), input.GetSize(), result, strict);
}
template <>
bool TryCast::Operation(string_t input, int64_t &result, bool strict) {
	return TrySimpleIntegerCast<int64_t>(input.GetData(), input.GetSize(), result, strict);
}

template <>
bool TryCast::Operation(string_t input, uint8_t &result, bool strict) {
	return TrySimpleIntegerCast<uint8_t, false>(input.GetData(), input.GetSize(), result, strict);
}
template <>
bool TryCast::Operation(string_t input, uint16_t &result, bool strict) {
	return TrySimpleIntegerCast<uint16_t, false>(input.GetData(), input.GetSize(), result, strict);
}
template <>
bool TryCast::Operation(string_t input, uint32_t &result, bool strict) {
	return TrySimpleIntegerCast<uint32_t, false>(input.GetData(), input.GetSize(), result, strict);
}
template <>
bool TryCast::Operation(string_t input, uint64_t &result, bool strict) {
	return TrySimpleIntegerCast<uint64_t, false>(input.GetData(), input.GetSize(), result, strict);
}

template <class T, char decimal_separator = '.'>
static bool TryDoubleCast(const char *buf, idx_t len, T &result, bool strict) {
	// skip any spaces at the start
	while (len > 0 && StringUtil::CharacterIsSpace(*buf)) {
		buf++;
		len--;
	}
	if (len == 0) {
		return false;
	}
	if (*buf == '+') {
		if (strict) {
			// plus is not allowed in strict mode
			return false;
		}
		buf++;
		len--;
	}
	if (strict && len >= 2) {
		if (buf[0] == '0' && StringUtil::CharacterIsDigit(buf[1])) {
			// leading zeros are not allowed in strict mode
			return false;
		}
	}
	auto endptr = buf + len;
	auto parse_result = duckdb_fast_float::from_chars(buf, buf + len, result, decimal_separator);
	if (parse_result.ec != std::errc()) {
		return false;
	}
	auto current_end = parse_result.ptr;
	if (!strict) {
		while (current_end < endptr && StringUtil::CharacterIsSpace(*current_end)) {
			current_end++;
		}
	}
	return current_end == endptr;
}

template <>
bool TryCast::Operation(string_t input, float &result, bool strict) {
	return TryDoubleCast<float>(input.GetData(), input.GetSize(), result, strict);
}

template <>
bool TryCast::Operation(string_t input, double &result, bool strict) {
	return TryDoubleCast<double>(input.GetData(), input.GetSize(), result, strict);
}

template <>
bool TryCastErrorMessageCommaSeparated::Operation(string_t input, float &result, string *error_message, bool strict) {
	if (!TryDoubleCast<float, ','>(input.GetData(), input.GetSize(), result, strict)) {
		HandleCastError::AssignError(StringUtil::Format("Could not cast string to float: \"%s\"", input.GetString()),
		                             error_message);
		return false;
	}
	return true;
}

template <>
bool TryCastErrorMessageCommaSeparated::Operation(string_t input, double &result, string *error_message, bool strict) {
	if (!TryDoubleCast<double, ','>(input.GetData(), input.GetSize(), result, strict)) {
		HandleCastError::AssignError(StringUtil::Format("Could not cast string to double: \"%s\"", input.GetString()),
		                             error_message);
		return false;
	}
	return true;
}

//===--------------------------------------------------------------------===//
// Cast From Date
//===--------------------------------------------------------------------===//
template <>
bool TryCast::Operation(date_t input, date_t &result, bool strict) {
	result = input;
	return true;
}

template <>
bool TryCast::Operation(date_t input, timestamp_t &result, bool strict) {
	if (input == date_t::infinity()) {
		result = timestamp_t::infinity();
		return true;
	} else if (input == date_t::ninfinity()) {
		result = timestamp_t::ninfinity();
		return true;
	}
	return Timestamp::TryFromDatetime(input, Time::FromTime(0, 0, 0), result);
}

//===--------------------------------------------------------------------===//
// Cast From Time
//===--------------------------------------------------------------------===//
template <>
bool TryCast::Operation(dtime_t input, dtime_t &result, bool strict) {
	result = input;
	return true;
}

//===--------------------------------------------------------------------===//
// Cast From Timestamps
//===--------------------------------------------------------------------===//
template <>
bool TryCast::Operation(timestamp_t input, date_t &result, bool strict) {
	result = Timestamp::GetDate(input);
	return true;
}

template <>
bool TryCast::Operation(timestamp_t input, dtime_t &result, bool strict) {
	if (!Timestamp::IsFinite(input)) {
		return false;
	}
	result = Timestamp::GetTime(input);
	return true;
}

template <>
bool TryCast::Operation(timestamp_t input, timestamp_t &result, bool strict) {
	result = input;
	return true;
}

//===--------------------------------------------------------------------===//
// Cast from Interval
//===--------------------------------------------------------------------===//
template <>
bool TryCast::Operation(interval_t input, interval_t &result, bool strict) {
	result = input;
	return true;
}

//===--------------------------------------------------------------------===//
// Non-Standard Timestamps
//===--------------------------------------------------------------------===//
template <>
duckdb::string_t CastFromTimestampNS::Operation(duckdb::timestamp_t input, Vector &result) {
	return StringCast::Operation<timestamp_t>(Timestamp::FromEpochNanoSeconds(input.value), result);
}
template <>
duckdb::string_t CastFromTimestampMS::Operation(duckdb::timestamp_t input, Vector &result) {
	return StringCast::Operation<timestamp_t>(Timestamp::FromEpochMs(input.value), result);
}
template <>
duckdb::string_t CastFromTimestampSec::Operation(duckdb::timestamp_t input, Vector &result) {
	return StringCast::Operation<timestamp_t>(Timestamp::FromEpochSeconds(input.value), result);
}

template <>
timestamp_t CastTimestampUsToMs::Operation(timestamp_t input) {
	timestamp_t cast_timestamp(Timestamp::GetEpochMs(input));
	return cast_timestamp;
}

template <>
timestamp_t CastTimestampUsToNs::Operation(timestamp_t input) {
	timestamp_t cast_timestamp(Timestamp::GetEpochNanoSeconds(input));
	return cast_timestamp;
}

template <>
timestamp_t CastTimestampUsToSec::Operation(timestamp_t input) {
	timestamp_t cast_timestamp(Timestamp::GetEpochSeconds(input));
	return cast_timestamp;
}
template <>
timestamp_t CastTimestampMsToUs::Operation(timestamp_t input) {
	return Timestamp::FromEpochMs(input.value);
}

template <>
timestamp_t CastTimestampNsToUs::Operation(timestamp_t input) {
	return Timestamp::FromEpochNanoSeconds(input.value);
}

template <>
timestamp_t CastTimestampSecToUs::Operation(timestamp_t input) {
	return Timestamp::FromEpochSeconds(input.value);
}

//===--------------------------------------------------------------------===//
// Cast To Timestamp
//===--------------------------------------------------------------------===//
template <>
bool TryCastToTimestampNS::Operation(string_t input, timestamp_t &result, bool strict) {
	if (!TryCast::Operation<string_t, timestamp_t>(input, result, strict)) {
		return false;
	}
	result = Timestamp::GetEpochNanoSeconds(result);
	return true;
}

template <>
bool TryCastToTimestampMS::Operation(string_t input, timestamp_t &result, bool strict) {
	if (!TryCast::Operation<string_t, timestamp_t>(input, result, strict)) {
		return false;
	}
	result = Timestamp::GetEpochMs(result);
	return true;
}

template <>
bool TryCastToTimestampSec::Operation(string_t input, timestamp_t &result, bool strict) {
	if (!TryCast::Operation<string_t, timestamp_t>(input, result, strict)) {
		return false;
	}
	result = Timestamp::GetEpochSeconds(result);
	return true;
}

template <>
bool TryCastToTimestampNS::Operation(date_t input, timestamp_t &result, bool strict) {
	if (!TryCast::Operation<date_t, timestamp_t>(input, result, strict)) {
		return false;
	}
	if (!TryMultiplyOperator::Operation(result.value, Interval::NANOS_PER_MICRO, result.value)) {
		return false;
	}
	return true;
}

template <>
bool TryCastToTimestampMS::Operation(date_t input, timestamp_t &result, bool strict) {
	if (!TryCast::Operation<date_t, timestamp_t>(input, result, strict)) {
		return false;
	}
	result.value /= Interval::MICROS_PER_MSEC;
	return true;
}

template <>
bool TryCastToTimestampSec::Operation(date_t input, timestamp_t &result, bool strict) {
	if (!TryCast::Operation<date_t, timestamp_t>(input, result, strict)) {
		return false;
	}
	result.value /= Interval::MICROS_PER_MSEC * Interval::MSECS_PER_SEC;
	return true;
}

//===--------------------------------------------------------------------===//
// Cast From Blob
//===--------------------------------------------------------------------===//
template <>
string_t CastFromBlob::Operation(string_t input, Vector &vector) {
	idx_t result_size = Blob::GetStringSize(input);

	string_t result = StringVector::EmptyString(vector, result_size);
	Blob::ToString(input, result.GetDataWriteable());
	result.Finalize();

	return result;
}

//===--------------------------------------------------------------------===//
// Cast From Bit
//===--------------------------------------------------------------------===//
template <>
string_t CastFromBit::Operation(string_t input, Vector &vector) {

	idx_t result_size = Bit::BitLength(input);
	string_t result = StringVector::EmptyString(vector, result_size);
	Bit::ToString(input, result.GetDataWriteable());
	result.Finalize();

	return result;
}

//===--------------------------------------------------------------------===//
// Cast From Pointer
//===--------------------------------------------------------------------===//
template <>
string_t CastFromPointer::Operation(uintptr_t input, Vector &vector) {
	std::string s = duckdb_fmt::format("0x{:x}", input);
	return StringVector::AddString(vector, s);
}

//===--------------------------------------------------------------------===//
// Cast To Blob
//===--------------------------------------------------------------------===//
template <>
bool TryCastToBlob::Operation(string_t input, string_t &result, Vector &result_vector, string *error_message,
                              bool strict) {
	idx_t result_size;
	if (!Blob::TryGetBlobSize(input, result_size, error_message)) {
		return false;
	}

	result = StringVector::EmptyString(result_vector, result_size);
	Blob::ToBlob(input, (data_ptr_t)result.GetDataWriteable());
	result.Finalize();
	return true;
}

//===--------------------------------------------------------------------===//
// Cast To Bit
//===--------------------------------------------------------------------===//
template <>
bool TryCastToBit::Operation(string_t input, string_t &result, Vector &result_vector, string *error_message,
                             bool strict) {
	idx_t result_size;
	if (!Bit::TryGetBitStringSize(input, result_size, error_message)) {
		return false;
	}

	result = StringVector::EmptyString(result_vector, result_size);
	Bit::ToBit(input, result);
	result.Finalize();
	return true;
}

//===--------------------------------------------------------------------===//
// Cast From UUID
//===--------------------------------------------------------------------===//
template <>
string_t CastFromUUID::Operation(hugeint_t input, Vector &vector) {
	string_t result = StringVector::EmptyString(vector, 36);
	UUID::ToString(input, result.GetDataWriteable());
	result.Finalize();
	return result;
}

//===--------------------------------------------------------------------===//
// Cast To UUID
//===--------------------------------------------------------------------===//
template <>
bool TryCastToUUID::Operation(string_t input, hugeint_t &result, Vector &result_vector, string *error_message,
                              bool strict) {
	return UUID::FromString(input.GetString(), result);
}

//===--------------------------------------------------------------------===//
// Cast To Date
//===--------------------------------------------------------------------===//
template <>
bool TryCastErrorMessage::Operation(string_t input, date_t &result, string *error_message, bool strict) {
	if (!TryCast::Operation<string_t, date_t>(input, result, strict)) {
		HandleCastError::AssignError(Date::ConversionError(input), error_message);
		return false;
	}
	return true;
}

template <>
bool TryCast::Operation(string_t input, date_t &result, bool strict) {
	idx_t pos;
	bool special = false;
	return Date::TryConvertDate(input.GetData(), input.GetSize(), pos, result, special, strict);
}

template <>
date_t Cast::Operation(string_t input) {
	return Date::FromCString(input.GetData(), input.GetSize());
}

//===--------------------------------------------------------------------===//
// Cast To Time
//===--------------------------------------------------------------------===//
template <>
bool TryCastErrorMessage::Operation(string_t input, dtime_t &result, string *error_message, bool strict) {
	if (!TryCast::Operation<string_t, dtime_t>(input, result, strict)) {
		HandleCastError::AssignError(Time::ConversionError(input), error_message);
		return false;
	}
	return true;
}

template <>
bool TryCast::Operation(string_t input, dtime_t &result, bool strict) {
	idx_t pos;
	return Time::TryConvertTime(input.GetData(), input.GetSize(), pos, result, strict);
}

template <>
dtime_t Cast::Operation(string_t input) {
	return Time::FromCString(input.GetData(), input.GetSize());
}

//===--------------------------------------------------------------------===//
// Cast To Timestamp
//===--------------------------------------------------------------------===//
template <>
bool TryCastErrorMessage::Operation(string_t input, timestamp_t &result, string *error_message, bool strict) {
	auto cast_result = Timestamp::TryConvertTimestamp(input.GetData(), input.GetSize(), result);
	if (cast_result == TimestampCastResult::SUCCESS) {
		return true;
	}
	if (cast_result == TimestampCastResult::ERROR_INCORRECT_FORMAT) {
		HandleCastError::AssignError(Timestamp::ConversionError(input), error_message);
	} else {
		HandleCastError::AssignError(Timestamp::UnsupportedTimezoneError(input), error_message);
	}
	return false;
}

template <>
bool TryCast::Operation(string_t input, timestamp_t &result, bool strict) {
	return Timestamp::TryConvertTimestamp(input.GetData(), input.GetSize(), result) == TimestampCastResult::SUCCESS;
}

template <>
timestamp_t Cast::Operation(string_t input) {
	return Timestamp::FromCString(input.GetData(), input.GetSize());
}

//===--------------------------------------------------------------------===//
// Cast From Interval
//===--------------------------------------------------------------------===//
template <>
bool TryCastErrorMessage::Operation(string_t input, interval_t &result, string *error_message, bool strict) {
	return Interval::FromCString(input.GetData(), input.GetSize(), result, error_message, strict);
}

//===--------------------------------------------------------------------===//
// Cast From Hugeint
//===--------------------------------------------------------------------===//
// parsing hugeint from string is done a bit differently for performance reasons
// for other integer types we keep track of a single value
// and multiply that value by 10 for every digit we read
// however, for hugeints, multiplication is very expensive (>20X as expensive as for int64)
// for that reason, we parse numbers first into an int64 value
// when that value is full, we perform a HUGEINT multiplication to flush it into the hugeint
// this takes the number of HUGEINT multiplications down from [0-38] to [0-2]
struct HugeIntCastData {
	hugeint_t hugeint;
	int64_t intermediate;
	uint8_t digits;
	bool decimal;

	bool Flush() {
		if (digits == 0 && intermediate == 0) {
			return true;
		}
		if (hugeint.lower != 0 || hugeint.upper != 0) {
			if (digits > 38) {
				return false;
			}
			if (!Hugeint::TryMultiply(hugeint, Hugeint::POWERS_OF_TEN[digits], hugeint)) {
				return false;
			}
		}
		if (!Hugeint::AddInPlace(hugeint, hugeint_t(intermediate))) {
			return false;
		}
		digits = 0;
		intermediate = 0;
		return true;
	}
};

struct HugeIntegerCastOperation {
	template <class T, bool NEGATIVE>
	static bool HandleDigit(T &result, uint8_t digit) {
		if (NEGATIVE) {
			if (result.intermediate < (NumericLimits<int64_t>::Minimum() + digit) / 10) {
				// intermediate is full: need to flush it
				if (!result.Flush()) {
					return false;
				}
			}
			result.intermediate = result.intermediate * 10 - digit;
		} else {
			if (result.intermediate > (NumericLimits<int64_t>::Maximum() - digit) / 10) {
				if (!result.Flush()) {
					return false;
				}
			}
			result.intermediate = result.intermediate * 10 + digit;
		}
		result.digits++;
		return true;
	}

	template <class T, bool NEGATIVE>
	static bool HandleHexDigit(T &result, uint8_t digit) {
		return false;
	}

	template <class T, bool NEGATIVE>
	static bool HandleBinaryDigit(T &result, uint8_t digit) {
		if (result.intermediate > (NumericLimits<int64_t>::Maximum() - digit) / 2) {
			// intermediate is full: need to flush it
			if (!result.Flush()) {
				return false;
			}
		}
		result.intermediate = result.intermediate * 2 + digit;
		result.digits++;
		return true;
	}

	template <class T, bool NEGATIVE>
	static bool HandleExponent(T &result, int32_t exponent) {
		if (!result.Flush()) {
			return false;
		}
		if (exponent < -38 || exponent > 38) {
			// out of range for exact exponent: use double and convert
			double dbl_res = Hugeint::Cast<double>(result.hugeint) * std::pow(10.0L, exponent);
			if (dbl_res < Hugeint::Cast<double>(NumericLimits<hugeint_t>::Minimum()) ||
			    dbl_res > Hugeint::Cast<double>(NumericLimits<hugeint_t>::Maximum())) {
				return false;
			}
			result.hugeint = Hugeint::Convert(dbl_res);
			return true;
		}
		if (exponent < 0) {
			// negative exponent: divide by power of 10
			result.hugeint = Hugeint::Divide(result.hugeint, Hugeint::POWERS_OF_TEN[-exponent]);
			return true;
		} else {
			// positive exponent: multiply by power of 10
			return Hugeint::TryMultiply(result.hugeint, Hugeint::POWERS_OF_TEN[exponent], result.hugeint);
		}
	}

	template <class T, bool NEGATIVE, bool ALLOW_EXPONENT>
	static bool HandleDecimal(T &result, uint8_t digit) {
		// Integer casts round
		if (!result.decimal) {
			if (!result.Flush()) {
				return false;
			}
			if (NEGATIVE) {
				result.intermediate = -(digit >= 5);
			} else {
				result.intermediate = (digit >= 5);
			}
		}
		result.decimal = true;

		return true;
	}

	template <class T, bool NEGATIVE>
	static bool Finalize(T &result) {
		return result.Flush();
	}
};

template <>
bool TryCast::Operation(string_t input, hugeint_t &result, bool strict) {
	HugeIntCastData data;
	if (!TryIntegerCast<HugeIntCastData, true, true, HugeIntegerCastOperation>(input.GetData(), input.GetSize(), data,
	                                                                           strict)) {
		return false;
	}
	result = data.hugeint;
	return true;
}

//===--------------------------------------------------------------------===//
// Decimal String Cast
//===--------------------------------------------------------------------===//

template <class TYPE>
struct DecimalCastData {
	typedef TYPE type_t;
	TYPE result;
	uint8_t width;
	uint8_t scale;
	uint8_t digit_count;
	uint8_t decimal_count;
	//! Whether we have determined if the result should be rounded
	bool round_set;
	//! If the result should be rounded
	bool should_round;
	//! Only set when ALLOW_EXPONENT is enabled
	enum class ExponentType : uint8_t { NONE, POSITIVE, NEGATIVE };
	uint8_t excessive_decimals;
	ExponentType exponent_type;
};

struct DecimalCastOperation {
	template <class T, bool NEGATIVE>
	static bool HandleDigit(T &state, uint8_t digit) {
		if (state.result == 0 && digit == 0) {
			// leading zero's don't count towards the digit count
			return true;
		}
		if (state.digit_count == state.width - state.scale) {
			// width of decimal type is exceeded!
			return false;
		}
		state.digit_count++;
		if (NEGATIVE) {
			if (state.result < (NumericLimits<typename T::type_t>::Minimum() / 10)) {
				return false;
			}
			state.result = state.result * 10 - digit;
		} else {
			if (state.result > (NumericLimits<typename T::type_t>::Maximum() / 10)) {
				return false;
			}
			state.result = state.result * 10 + digit;
		}
		return true;
	}

	template <class T, bool NEGATIVE>
	static bool HandleHexDigit(T &state, uint8_t digit) {
		return false;
	}

	template <class T, bool NEGATIVE>
	static bool HandleBinaryDigit(T &state, uint8_t digit) {
		return false;
	}

	template <class T, bool NEGATIVE>
	static void RoundUpResult(T &state) {
		if (NEGATIVE) {
			state.result -= 1;
		} else {
			state.result += 1;
		}
	}

	template <class T, bool NEGATIVE>
	static bool HandleExponent(T &state, int32_t exponent) {
		auto decimal_excess = (state.decimal_count > state.scale) ? state.decimal_count - state.scale : 0;
		if (exponent > 0) {
			state.exponent_type = T::ExponentType::POSITIVE;
			// Positive exponents need up to 'exponent' amount of digits
			// Everything beyond that amount needs to be truncated
			if (decimal_excess > exponent) {
				// We've allowed too many decimals
				state.excessive_decimals = decimal_excess - exponent;
				exponent = 0;
			} else {
				exponent -= decimal_excess;
			}
			D_ASSERT(exponent >= 0);
		} else if (exponent < 0) {
			state.exponent_type = T::ExponentType::NEGATIVE;
		}
		if (!Finalize<T, NEGATIVE>(state)) {
			return false;
		}
		if (exponent < 0) {
			bool round_up = false;
			for (idx_t i = 0; i < idx_t(-int64_t(exponent)); i++) {
				auto mod = state.result % 10;
				round_up = NEGATIVE ? mod <= -5 : mod >= 5;
				state.result /= 10;
				if (state.result == 0) {
					break;
				}
			}
			if (round_up) {
				RoundUpResult<T, NEGATIVE>(state);
			}
			return true;
		} else {
			// positive exponent: append 0's
			for (idx_t i = 0; i < idx_t(exponent); i++) {
				if (!HandleDigit<T, NEGATIVE>(state, 0)) {
					return false;
				}
			}
			return true;
		}
	}

	template <class T, bool NEGATIVE, bool ALLOW_EXPONENT>
	static bool HandleDecimal(T &state, uint8_t digit) {
		if (state.decimal_count == state.scale && !state.round_set) {
			// Determine whether the last registered decimal should be rounded or not
			state.round_set = true;
			state.should_round = digit >= 5;
		}
		if (!ALLOW_EXPONENT && state.decimal_count == state.scale) {
			// we exceeded the amount of supported decimals
			// however, we don't throw an error here
			// we just truncate the decimal
			return true;
		}
		//! If we expect an exponent, we need to preserve the decimals
		//! But we don't want to overflow, so we prevent overflowing the result with this check
		if (state.digit_count + state.decimal_count >= DecimalWidth<decltype(state.result)>::max) {
			return true;
		}
		state.decimal_count++;
		if (NEGATIVE) {
			state.result = state.result * 10 - digit;
		} else {
			state.result = state.result * 10 + digit;
		}
		return true;
	}

	template <class T, bool NEGATIVE>
	static bool TruncateExcessiveDecimals(T &state) {
		D_ASSERT(state.excessive_decimals);
		bool round_up = false;
		for (idx_t i = 0; i < state.excessive_decimals; i++) {
			auto mod = state.result % 10;
			round_up = NEGATIVE ? mod <= -5 : mod >= 5;
			state.result /= 10.0;
		}
		//! Only round up when exponents are involved
		if (state.exponent_type == T::ExponentType::POSITIVE && round_up) {
			RoundUpResult<T, NEGATIVE>(state);
		}
		D_ASSERT(state.decimal_count > state.scale);
		state.decimal_count = state.scale;
		return true;
	}

	template <class T, bool NEGATIVE>
	static bool Finalize(T &state) {
		if (state.exponent_type != T::ExponentType::POSITIVE && state.decimal_count > state.scale) {
			//! Did not encounter an exponent, but ALLOW_EXPONENT was on
			state.excessive_decimals = state.decimal_count - state.scale;
		}
		if (state.excessive_decimals && !TruncateExcessiveDecimals<T, NEGATIVE>(state)) {
			return false;
		}
		if (state.exponent_type == T::ExponentType::NONE && state.round_set && state.should_round) {
			RoundUpResult<T, NEGATIVE>(state);
		}
		//  if we have not gotten exactly "scale" decimals, we need to multiply the result
		//  e.g. if we have a string "1.0" that is cast to a DECIMAL(9,3), the value needs to be 1000
		//  but we have only gotten the value "10" so far, so we multiply by 1000
		for (uint8_t i = state.decimal_count; i < state.scale; i++) {
			state.result *= 10;
		}
		return true;
	}
};

template <class T, char decimal_separator = '.'>
bool TryDecimalStringCast(string_t input, T &result, string *error_message, uint8_t width, uint8_t scale) {
	DecimalCastData<T> state;
	state.result = 0;
	state.width = width;
	state.scale = scale;
	state.digit_count = 0;
	state.decimal_count = 0;
	state.excessive_decimals = 0;
	state.exponent_type = DecimalCastData<T>::ExponentType::NONE;
	state.round_set = false;
	state.should_round = false;
	if (!TryIntegerCast<DecimalCastData<T>, true, true, DecimalCastOperation, false, decimal_separator>(
	        input.GetData(), input.GetSize(), state, false)) {
		string error = StringUtil::Format("Could not convert string \"%s\" to DECIMAL(%d,%d)", input.GetString(),
		                                  (int)width, (int)scale);
		HandleCastError::AssignError(error, error_message);
		return false;
	}
	result = state.result;
	return true;
}

template <>
bool TryCastToDecimal::Operation(string_t input, int16_t &result, string *error_message, uint8_t width, uint8_t scale) {
	return TryDecimalStringCast<int16_t>(input, result, error_message, width, scale);
}

template <>
bool TryCastToDecimal::Operation(string_t input, int32_t &result, string *error_message, uint8_t width, uint8_t scale) {
	return TryDecimalStringCast<int32_t>(input, result, error_message, width, scale);
}

template <>
bool TryCastToDecimal::Operation(string_t input, int64_t &result, string *error_message, uint8_t width, uint8_t scale) {
	return TryDecimalStringCast<int64_t>(input, result, error_message, width, scale);
}

template <>
bool TryCastToDecimal::Operation(string_t input, hugeint_t &result, string *error_message, uint8_t width,
                                 uint8_t scale) {
	return TryDecimalStringCast<hugeint_t>(input, result, error_message, width, scale);
}

template <>
bool TryCastToDecimalCommaSeparated::Operation(string_t input, int16_t &result, string *error_message, uint8_t width,
                                               uint8_t scale) {
	return TryDecimalStringCast<int16_t, ','>(input, result, error_message, width, scale);
}

template <>
bool TryCastToDecimalCommaSeparated::Operation(string_t input, int32_t &result, string *error_message, uint8_t width,
                                               uint8_t scale) {
	return TryDecimalStringCast<int32_t, ','>(input, result, error_message, width, scale);
}

template <>
bool TryCastToDecimalCommaSeparated::Operation(string_t input, int64_t &result, string *error_message, uint8_t width,
                                               uint8_t scale) {
	return TryDecimalStringCast<int64_t, ','>(input, result, error_message, width, scale);
}

template <>
bool TryCastToDecimalCommaSeparated::Operation(string_t input, hugeint_t &result, string *error_message, uint8_t width,
                                               uint8_t scale) {
	return TryDecimalStringCast<hugeint_t, ','>(input, result, error_message, width, scale);
}

template <>
string_t StringCastFromDecimal::Operation(int16_t input, uint8_t width, uint8_t scale, Vector &result) {
	return DecimalToString::Format<int16_t, uint16_t>(input, width, scale, result);
}

template <>
string_t StringCastFromDecimal::Operation(int32_t input, uint8_t width, uint8_t scale, Vector &result) {
	return DecimalToString::Format<int32_t, uint32_t>(input, width, scale, result);
}

template <>
string_t StringCastFromDecimal::Operation(int64_t input, uint8_t width, uint8_t scale, Vector &result) {
	return DecimalToString::Format<int64_t, uint64_t>(input, width, scale, result);
}

template <>
string_t StringCastFromDecimal::Operation(hugeint_t input, uint8_t width, uint8_t scale, Vector &result) {
	return HugeintToStringCast::FormatDecimal(input, width, scale, result);
}

//===--------------------------------------------------------------------===//
// Decimal Casts
//===--------------------------------------------------------------------===//
// Decimal <-> Bool
//===--------------------------------------------------------------------===//
template <class T, class OP = NumericHelper>
bool TryCastBoolToDecimal(bool input, T &result, string *error_message, uint8_t width, uint8_t scale) {
	if (width > scale) {
		result = input ? OP::POWERS_OF_TEN[scale] : 0;
		return true;
	} else {
		return TryCast::Operation<bool, T>(input, result);
	}
}

template <>
bool TryCastToDecimal::Operation(bool input, int16_t &result, string *error_message, uint8_t width, uint8_t scale) {
	return TryCastBoolToDecimal<int16_t>(input, result, error_message, width, scale);
}

template <>
bool TryCastToDecimal::Operation(bool input, int32_t &result, string *error_message, uint8_t width, uint8_t scale) {
	return TryCastBoolToDecimal<int32_t>(input, result, error_message, width, scale);
}

template <>
bool TryCastToDecimal::Operation(bool input, int64_t &result, string *error_message, uint8_t width, uint8_t scale) {
	return TryCastBoolToDecimal<int64_t>(input, result, error_message, width, scale);
}

template <>
bool TryCastToDecimal::Operation(bool input, hugeint_t &result, string *error_message, uint8_t width, uint8_t scale) {
	return TryCastBoolToDecimal<hugeint_t, Hugeint>(input, result, error_message, width, scale);
}

template <>
bool TryCastFromDecimal::Operation(int16_t input, bool &result, string *error_message, uint8_t width, uint8_t scale) {
	return TryCast::Operation<int16_t, bool>(input, result);
}

template <>
bool TryCastFromDecimal::Operation(int32_t input, bool &result, string *error_message, uint8_t width, uint8_t scale) {
	return TryCast::Operation<int32_t, bool>(input, result);
}

template <>
bool TryCastFromDecimal::Operation(int64_t input, bool &result, string *error_message, uint8_t width, uint8_t scale) {
	return TryCast::Operation<int64_t, bool>(input, result);
}

template <>
bool TryCastFromDecimal::Operation(hugeint_t input, bool &result, string *error_message, uint8_t width, uint8_t scale) {
	return TryCast::Operation<hugeint_t, bool>(input, result);
}

//===--------------------------------------------------------------------===//
// Numeric -> Decimal Cast
//===--------------------------------------------------------------------===//
struct SignedToDecimalOperator {
	template <class SRC, class DST>
	static bool Operation(SRC input, DST max_width) {
		return int64_t(input) >= int64_t(max_width) || int64_t(input) <= int64_t(-max_width);
	}
};

struct UnsignedToDecimalOperator {
	template <class SRC, class DST>
	static bool Operation(SRC input, DST max_width) {
		return uint64_t(input) >= uint64_t(max_width);
	}
};

template <class SRC, class DST, class OP = SignedToDecimalOperator>
bool StandardNumericToDecimalCast(SRC input, DST &result, string *error_message, uint8_t width, uint8_t scale) {
	// check for overflow
	DST max_width = NumericHelper::POWERS_OF_TEN[width - scale];
	if (OP::template Operation<SRC, DST>(input, max_width)) {
		string error = StringUtil::Format("Could not cast value %d to DECIMAL(%d,%d)", input, width, scale);
		HandleCastError::AssignError(error, error_message);
		return false;
	}
	result = DST(input) * NumericHelper::POWERS_OF_TEN[scale];
	return true;
}

template <class SRC>
bool NumericToHugeDecimalCast(SRC input, hugeint_t &result, string *error_message, uint8_t width, uint8_t scale) {
	// check for overflow
	hugeint_t max_width = Hugeint::POWERS_OF_TEN[width - scale];
	hugeint_t hinput = Hugeint::Convert(input);
	if (hinput >= max_width || hinput <= -max_width) {
		string error = StringUtil::Format("Could not cast value %s to DECIMAL(%d,%d)", hinput.ToString(), width, scale);
		HandleCastError::AssignError(error, error_message);
		return false;
	}
	result = hinput * Hugeint::POWERS_OF_TEN[scale];
	return true;
}

//===--------------------------------------------------------------------===//
// Cast int8_t -> Decimal
//===--------------------------------------------------------------------===//
template <>
bool TryCastToDecimal::Operation(int8_t input, int16_t &result, string *error_message, uint8_t width, uint8_t scale) {
	return StandardNumericToDecimalCast<int8_t, int16_t>(input, result, error_message, width, scale);
}
template <>
bool TryCastToDecimal::Operation(int8_t input, int32_t &result, string *error_message, uint8_t width, uint8_t scale) {
	return StandardNumericToDecimalCast<int8_t, int32_t>(input, result, error_message, width, scale);
}
template <>
bool TryCastToDecimal::Operation(int8_t input, int64_t &result, string *error_message, uint8_t width, uint8_t scale) {
	return StandardNumericToDecimalCast<int8_t, int64_t>(input, result, error_message, width, scale);
}
template <>
bool TryCastToDecimal::Operation(int8_t input, hugeint_t &result, string *error_message, uint8_t width, uint8_t scale) {
	return NumericToHugeDecimalCast<int8_t>(input, result, error_message, width, scale);
}

//===--------------------------------------------------------------------===//
// Cast int16_t -> Decimal
//===--------------------------------------------------------------------===//
template <>
bool TryCastToDecimal::Operation(int16_t input, int16_t &result, string *error_message, uint8_t width, uint8_t scale) {
	return StandardNumericToDecimalCast<int16_t, int16_t>(input, result, error_message, width, scale);
}
template <>
bool TryCastToDecimal::Operation(int16_t input, int32_t &result, string *error_message, uint8_t width, uint8_t scale) {
	return StandardNumericToDecimalCast<int16_t, int32_t>(input, result, error_message, width, scale);
}
template <>
bool TryCastToDecimal::Operation(int16_t input, int64_t &result, string *error_message, uint8_t width, uint8_t scale) {
	return StandardNumericToDecimalCast<int16_t, int64_t>(input, result, error_message, width, scale);
}
template <>
bool TryCastToDecimal::Operation(int16_t input, hugeint_t &result, string *error_message, uint8_t width,
                                 uint8_t scale) {
	return NumericToHugeDecimalCast<int16_t>(input, result, error_message, width, scale);
}

//===--------------------------------------------------------------------===//
// Cast int32_t -> Decimal
//===--------------------------------------------------------------------===//
template <>
bool TryCastToDecimal::Operation(int32_t input, int16_t &result, string *error_message, uint8_t width, uint8_t scale) {
	return StandardNumericToDecimalCast<int32_t, int16_t>(input, result, error_message, width, scale);
}
template <>
bool TryCastToDecimal::Operation(int32_t input, int32_t &result, string *error_message, uint8_t width, uint8_t scale) {
	return StandardNumericToDecimalCast<int32_t, int32_t>(input, result, error_message, width, scale);
}
template <>
bool TryCastToDecimal::Operation(int32_t input, int64_t &result, string *error_message, uint8_t width, uint8_t scale) {
	return StandardNumericToDecimalCast<int32_t, int64_t>(input, result, error_message, width, scale);
}
template <>
bool TryCastToDecimal::Operation(int32_t input, hugeint_t &result, string *error_message, uint8_t width,
                                 uint8_t scale) {
	return NumericToHugeDecimalCast<int32_t>(input, result, error_message, width, scale);
}

//===--------------------------------------------------------------------===//
// Cast int64_t -> Decimal
//===--------------------------------------------------------------------===//
template <>
bool TryCastToDecimal::Operation(int64_t input, int16_t &result, string *error_message, uint8_t width, uint8_t scale) {
	return StandardNumericToDecimalCast<int64_t, int16_t>(input, result, error_message, width, scale);
}
template <>
bool TryCastToDecimal::Operation(int64_t input, int32_t &result, string *error_message, uint8_t width, uint8_t scale) {
	return StandardNumericToDecimalCast<int64_t, int32_t>(input, result, error_message, width, scale);
}
template <>
bool TryCastToDecimal::Operation(int64_t input, int64_t &result, string *error_message, uint8_t width, uint8_t scale) {
	return StandardNumericToDecimalCast<int64_t, int64_t>(input, result, error_message, width, scale);
}
template <>
bool TryCastToDecimal::Operation(int64_t input, hugeint_t &result, string *error_message, uint8_t width,
                                 uint8_t scale) {
	return NumericToHugeDecimalCast<int64_t>(input, result, error_message, width, scale);
}

//===--------------------------------------------------------------------===//
// Cast uint8_t -> Decimal
//===--------------------------------------------------------------------===//
template <>
bool TryCastToDecimal::Operation(uint8_t input, int16_t &result, string *error_message, uint8_t width, uint8_t scale) {
	return StandardNumericToDecimalCast<uint8_t, int16_t, UnsignedToDecimalOperator>(input, result, error_message,
	                                                                                 width, scale);
}
template <>
bool TryCastToDecimal::Operation(uint8_t input, int32_t &result, string *error_message, uint8_t width, uint8_t scale) {
	return StandardNumericToDecimalCast<uint8_t, int32_t, UnsignedToDecimalOperator>(input, result, error_message,
	                                                                                 width, scale);
}
template <>
bool TryCastToDecimal::Operation(uint8_t input, int64_t &result, string *error_message, uint8_t width, uint8_t scale) {
	return StandardNumericToDecimalCast<uint8_t, int64_t, UnsignedToDecimalOperator>(input, result, error_message,
	                                                                                 width, scale);
}
template <>
bool TryCastToDecimal::Operation(uint8_t input, hugeint_t &result, string *error_message, uint8_t width,
                                 uint8_t scale) {
	return NumericToHugeDecimalCast<uint8_t>(input, result, error_message, width, scale);
}

//===--------------------------------------------------------------------===//
// Cast uint16_t -> Decimal
//===--------------------------------------------------------------------===//
template <>
bool TryCastToDecimal::Operation(uint16_t input, int16_t &result, string *error_message, uint8_t width, uint8_t scale) {
	return StandardNumericToDecimalCast<uint16_t, int16_t, UnsignedToDecimalOperator>(input, result, error_message,
	                                                                                  width, scale);
}
template <>
bool TryCastToDecimal::Operation(uint16_t input, int32_t &result, string *error_message, uint8_t width, uint8_t scale) {
	return StandardNumericToDecimalCast<uint16_t, int32_t, UnsignedToDecimalOperator>(input, result, error_message,
	                                                                                  width, scale);
}
template <>
bool TryCastToDecimal::Operation(uint16_t input, int64_t &result, string *error_message, uint8_t width, uint8_t scale) {
	return StandardNumericToDecimalCast<uint16_t, int64_t, UnsignedToDecimalOperator>(input, result, error_message,
	                                                                                  width, scale);
}
template <>
bool TryCastToDecimal::Operation(uint16_t input, hugeint_t &result, string *error_message, uint8_t width,
                                 uint8_t scale) {
	return NumericToHugeDecimalCast<uint16_t>(input, result, error_message, width, scale);
}

//===--------------------------------------------------------------------===//
// Cast uint32_t -> Decimal
//===--------------------------------------------------------------------===//
template <>
bool TryCastToDecimal::Operation(uint32_t input, int16_t &result, string *error_message, uint8_t width, uint8_t scale) {
	return StandardNumericToDecimalCast<uint32_t, int16_t, UnsignedToDecimalOperator>(input, result, error_message,
	                                                                                  width, scale);
}
template <>
bool TryCastToDecimal::Operation(uint32_t input, int32_t &result, string *error_message, uint8_t width, uint8_t scale) {
	return StandardNumericToDecimalCast<uint32_t, int32_t, UnsignedToDecimalOperator>(input, result, error_message,
	                                                                                  width, scale);
}
template <>
bool TryCastToDecimal::Operation(uint32_t input, int64_t &result, string *error_message, uint8_t width, uint8_t scale) {
	return StandardNumericToDecimalCast<uint32_t, int64_t, UnsignedToDecimalOperator>(input, result, error_message,
	                                                                                  width, scale);
}
template <>
bool TryCastToDecimal::Operation(uint32_t input, hugeint_t &result, string *error_message, uint8_t width,
                                 uint8_t scale) {
	return NumericToHugeDecimalCast<uint32_t>(input, result, error_message, width, scale);
}

//===--------------------------------------------------------------------===//
// Cast uint64_t -> Decimal
//===--------------------------------------------------------------------===//
template <>
bool TryCastToDecimal::Operation(uint64_t input, int16_t &result, string *error_message, uint8_t width, uint8_t scale) {
	return StandardNumericToDecimalCast<uint64_t, int16_t, UnsignedToDecimalOperator>(input, result, error_message,
	                                                                                  width, scale);
}
template <>
bool TryCastToDecimal::Operation(uint64_t input, int32_t &result, string *error_message, uint8_t width, uint8_t scale) {
	return StandardNumericToDecimalCast<uint64_t, int32_t, UnsignedToDecimalOperator>(input, result, error_message,
	                                                                                  width, scale);
}
template <>
bool TryCastToDecimal::Operation(uint64_t input, int64_t &result, string *error_message, uint8_t width, uint8_t scale) {
	return StandardNumericToDecimalCast<uint64_t, int64_t, UnsignedToDecimalOperator>(input, result, error_message,
	                                                                                  width, scale);
}
template <>
bool TryCastToDecimal::Operation(uint64_t input, hugeint_t &result, string *error_message, uint8_t width,
                                 uint8_t scale) {
	return NumericToHugeDecimalCast<uint64_t>(input, result, error_message, width, scale);
}

//===--------------------------------------------------------------------===//
// Hugeint -> Decimal Cast
//===--------------------------------------------------------------------===//
template <class DST>
bool HugeintToDecimalCast(hugeint_t input, DST &result, string *error_message, uint8_t width, uint8_t scale) {
	// check for overflow
	hugeint_t max_width = Hugeint::POWERS_OF_TEN[width - scale];
	if (input >= max_width || input <= -max_width) {
		string error = StringUtil::Format("Could not cast value %s to DECIMAL(%d,%d)", input.ToString(), width, scale);
		HandleCastError::AssignError(error, error_message);
		return false;
	}
	result = Hugeint::Cast<DST>(input * Hugeint::POWERS_OF_TEN[scale]);
	return true;
}

template <>
bool TryCastToDecimal::Operation(hugeint_t input, int16_t &result, string *error_message, uint8_t width,
                                 uint8_t scale) {
	return HugeintToDecimalCast<int16_t>(input, result, error_message, width, scale);
}

template <>
bool TryCastToDecimal::Operation(hugeint_t input, int32_t &result, string *error_message, uint8_t width,
                                 uint8_t scale) {
	return HugeintToDecimalCast<int32_t>(input, result, error_message, width, scale);
}

template <>
bool TryCastToDecimal::Operation(hugeint_t input, int64_t &result, string *error_message, uint8_t width,
                                 uint8_t scale) {
	return HugeintToDecimalCast<int64_t>(input, result, error_message, width, scale);
}

template <>
bool TryCastToDecimal::Operation(hugeint_t input, hugeint_t &result, string *error_message, uint8_t width,
                                 uint8_t scale) {
	return HugeintToDecimalCast<hugeint_t>(input, result, error_message, width, scale);
}

//===--------------------------------------------------------------------===//
// Float/Double -> Decimal Cast
//===--------------------------------------------------------------------===//
template <class SRC, class DST>
bool DoubleToDecimalCast(SRC input, DST &result, string *error_message, uint8_t width, uint8_t scale) {
	double value = input * NumericHelper::DOUBLE_POWERS_OF_TEN[scale];
	// Add the sign (-1, 0, 1) times a tiny value to fix floating point issues (issue 3091)
	double sign = (double(0) < value) - (value < double(0));
	value += 1e-9 * sign;
	if (value <= -NumericHelper::DOUBLE_POWERS_OF_TEN[width] || value >= NumericHelper::DOUBLE_POWERS_OF_TEN[width]) {
		string error = StringUtil::Format("Could not cast value %f to DECIMAL(%d,%d)", value, width, scale);
		HandleCastError::AssignError(error, error_message);
		return false;
	}
	result = Cast::Operation<SRC, DST>(value);
	return true;
}

template <>
bool TryCastToDecimal::Operation(float input, int16_t &result, string *error_message, uint8_t width, uint8_t scale) {
	return DoubleToDecimalCast<float, int16_t>(input, result, error_message, width, scale);
}

template <>
bool TryCastToDecimal::Operation(float input, int32_t &result, string *error_message, uint8_t width, uint8_t scale) {
	return DoubleToDecimalCast<float, int32_t>(input, result, error_message, width, scale);
}

template <>
bool TryCastToDecimal::Operation(float input, int64_t &result, string *error_message, uint8_t width, uint8_t scale) {
	return DoubleToDecimalCast<float, int64_t>(input, result, error_message, width, scale);
}

template <>
bool TryCastToDecimal::Operation(float input, hugeint_t &result, string *error_message, uint8_t width, uint8_t scale) {
	return DoubleToDecimalCast<float, hugeint_t>(input, result, error_message, width, scale);
}

template <>
bool TryCastToDecimal::Operation(double input, int16_t &result, string *error_message, uint8_t width, uint8_t scale) {
	return DoubleToDecimalCast<double, int16_t>(input, result, error_message, width, scale);
}

template <>
bool TryCastToDecimal::Operation(double input, int32_t &result, string *error_message, uint8_t width, uint8_t scale) {
	return DoubleToDecimalCast<double, int32_t>(input, result, error_message, width, scale);
}

template <>
bool TryCastToDecimal::Operation(double input, int64_t &result, string *error_message, uint8_t width, uint8_t scale) {
	return DoubleToDecimalCast<double, int64_t>(input, result, error_message, width, scale);
}

template <>
bool TryCastToDecimal::Operation(double input, hugeint_t &result, string *error_message, uint8_t width, uint8_t scale) {
	return DoubleToDecimalCast<double, hugeint_t>(input, result, error_message, width, scale);
}

//===--------------------------------------------------------------------===//
// Decimal -> Numeric Cast
//===--------------------------------------------------------------------===//
template <class SRC, class DST>
bool TryCastDecimalToNumeric(SRC input, DST &result, string *error_message, uint8_t scale) {
	// Round away from 0.
	const auto power = NumericHelper::POWERS_OF_TEN[scale];
	// https://graphics.stanford.edu/~seander/bithacks.html#ConditionalNegate
	const auto fNegate = int64_t(input < 0);
	const auto rounding = ((power ^ -fNegate) + fNegate) / 2;
	const auto scaled_value = (input + rounding) / power;
	if (!TryCast::Operation<SRC, DST>(scaled_value, result)) {
		string error = StringUtil::Format("Failed to cast decimal value %d to type %s", scaled_value, GetTypeId<DST>());
		HandleCastError::AssignError(error, error_message);
		return false;
	}
	return true;
}

template <class DST>
bool TryCastHugeDecimalToNumeric(hugeint_t input, DST &result, string *error_message, uint8_t scale) {
	const auto power = Hugeint::POWERS_OF_TEN[scale];
	const auto rounding = ((input < 0) ? -power : power) / 2;
	auto scaled_value = (input + rounding) / power;
	if (!TryCast::Operation<hugeint_t, DST>(scaled_value, result)) {
		string error = StringUtil::Format("Failed to cast decimal value %s to type %s",
		                                  ConvertToString::Operation(scaled_value), GetTypeId<DST>());
		HandleCastError::AssignError(error, error_message);
		return false;
	}
	return true;
}

//===--------------------------------------------------------------------===//
// Cast Decimal -> int8_t
//===--------------------------------------------------------------------===//
template <>
bool TryCastFromDecimal::Operation(int16_t input, int8_t &result, string *error_message, uint8_t width, uint8_t scale) {
	return TryCastDecimalToNumeric<int16_t, int8_t>(input, result, error_message, scale);
}
template <>
bool TryCastFromDecimal::Operation(int32_t input, int8_t &result, string *error_message, uint8_t width, uint8_t scale) {
	return TryCastDecimalToNumeric<int32_t, int8_t>(input, result, error_message, scale);
}
template <>
bool TryCastFromDecimal::Operation(int64_t input, int8_t &result, string *error_message, uint8_t width, uint8_t scale) {
	return TryCastDecimalToNumeric<int64_t, int8_t>(input, result, error_message, scale);
}
template <>
bool TryCastFromDecimal::Operation(hugeint_t input, int8_t &result, string *error_message, uint8_t width,
                                   uint8_t scale) {
	return TryCastHugeDecimalToNumeric<int8_t>(input, result, error_message, scale);
}

//===--------------------------------------------------------------------===//
// Cast Decimal -> int16_t
//===--------------------------------------------------------------------===//
template <>
bool TryCastFromDecimal::Operation(int16_t input, int16_t &result, string *error_message, uint8_t width,
                                   uint8_t scale) {
	return TryCastDecimalToNumeric<int16_t, int16_t>(input, result, error_message, scale);
}
template <>
bool TryCastFromDecimal::Operation(int32_t input, int16_t &result, string *error_message, uint8_t width,
                                   uint8_t scale) {
	return TryCastDecimalToNumeric<int32_t, int16_t>(input, result, error_message, scale);
}
template <>
bool TryCastFromDecimal::Operation(int64_t input, int16_t &result, string *error_message, uint8_t width,
                                   uint8_t scale) {
	return TryCastDecimalToNumeric<int64_t, int16_t>(input, result, error_message, scale);
}
template <>
bool TryCastFromDecimal::Operation(hugeint_t input, int16_t &result, string *error_message, uint8_t width,
                                   uint8_t scale) {
	return TryCastHugeDecimalToNumeric<int16_t>(input, result, error_message, scale);
}

//===--------------------------------------------------------------------===//
// Cast Decimal -> int32_t
//===--------------------------------------------------------------------===//
template <>
bool TryCastFromDecimal::Operation(int16_t input, int32_t &result, string *error_message, uint8_t width,
                                   uint8_t scale) {
	return TryCastDecimalToNumeric<int16_t, int32_t>(input, result, error_message, scale);
}
template <>
bool TryCastFromDecimal::Operation(int32_t input, int32_t &result, string *error_message, uint8_t width,
                                   uint8_t scale) {
	return TryCastDecimalToNumeric<int32_t, int32_t>(input, result, error_message, scale);
}
template <>
bool TryCastFromDecimal::Operation(int64_t input, int32_t &result, string *error_message, uint8_t width,
                                   uint8_t scale) {
	return TryCastDecimalToNumeric<int64_t, int32_t>(input, result, error_message, scale);
}
template <>
bool TryCastFromDecimal::Operation(hugeint_t input, int32_t &result, string *error_message, uint8_t width,
                                   uint8_t scale) {
	return TryCastHugeDecimalToNumeric<int32_t>(input, result, error_message, scale);
}

//===--------------------------------------------------------------------===//
// Cast Decimal -> int64_t
//===--------------------------------------------------------------------===//
template <>
bool TryCastFromDecimal::Operation(int16_t input, int64_t &result, string *error_message, uint8_t width,
                                   uint8_t scale) {
	return TryCastDecimalToNumeric<int16_t, int64_t>(input, result, error_message, scale);
}
template <>
bool TryCastFromDecimal::Operation(int32_t input, int64_t &result, string *error_message, uint8_t width,
                                   uint8_t scale) {
	return TryCastDecimalToNumeric<int32_t, int64_t>(input, result, error_message, scale);
}
template <>
bool TryCastFromDecimal::Operation(int64_t input, int64_t &result, string *error_message, uint8_t width,
                                   uint8_t scale) {
	return TryCastDecimalToNumeric<int64_t, int64_t>(input, result, error_message, scale);
}
template <>
bool TryCastFromDecimal::Operation(hugeint_t input, int64_t &result, string *error_message, uint8_t width,
                                   uint8_t scale) {
	return TryCastHugeDecimalToNumeric<int64_t>(input, result, error_message, scale);
}

//===--------------------------------------------------------------------===//
// Cast Decimal -> uint8_t
//===--------------------------------------------------------------------===//
template <>
bool TryCastFromDecimal::Operation(int16_t input, uint8_t &result, string *error_message, uint8_t width,
                                   uint8_t scale) {
	return TryCastDecimalToNumeric<int16_t, uint8_t>(input, result, error_message, scale);
}
template <>
bool TryCastFromDecimal::Operation(int32_t input, uint8_t &result, string *error_message, uint8_t width,
                                   uint8_t scale) {
	return TryCastDecimalToNumeric<int32_t, uint8_t>(input, result, error_message, scale);
}
template <>
bool TryCastFromDecimal::Operation(int64_t input, uint8_t &result, string *error_message, uint8_t width,
                                   uint8_t scale) {
	return TryCastDecimalToNumeric<int64_t, uint8_t>(input, result, error_message, scale);
}
template <>
bool TryCastFromDecimal::Operation(hugeint_t input, uint8_t &result, string *error_message, uint8_t width,
                                   uint8_t scale) {
	return TryCastHugeDecimalToNumeric<uint8_t>(input, result, error_message, scale);
}

//===--------------------------------------------------------------------===//
// Cast Decimal -> uint16_t
//===--------------------------------------------------------------------===//
template <>
bool TryCastFromDecimal::Operation(int16_t input, uint16_t &result, string *error_message, uint8_t width,
                                   uint8_t scale) {
	return TryCastDecimalToNumeric<int16_t, uint16_t>(input, result, error_message, scale);
}
template <>
bool TryCastFromDecimal::Operation(int32_t input, uint16_t &result, string *error_message, uint8_t width,
                                   uint8_t scale) {
	return TryCastDecimalToNumeric<int32_t, uint16_t>(input, result, error_message, scale);
}
template <>
bool TryCastFromDecimal::Operation(int64_t input, uint16_t &result, string *error_message, uint8_t width,
                                   uint8_t scale) {
	return TryCastDecimalToNumeric<int64_t, uint16_t>(input, result, error_message, scale);
}
template <>
bool TryCastFromDecimal::Operation(hugeint_t input, uint16_t &result, string *error_message, uint8_t width,
                                   uint8_t scale) {
	return TryCastHugeDecimalToNumeric<uint16_t>(input, result, error_message, scale);
}

//===--------------------------------------------------------------------===//
// Cast Decimal -> uint32_t
//===--------------------------------------------------------------------===//
template <>
bool TryCastFromDecimal::Operation(int16_t input, uint32_t &result, string *error_message, uint8_t width,
                                   uint8_t scale) {
	return TryCastDecimalToNumeric<int16_t, uint32_t>(input, result, error_message, scale);
}
template <>
bool TryCastFromDecimal::Operation(int32_t input, uint32_t &result, string *error_message, uint8_t width,
                                   uint8_t scale) {
	return TryCastDecimalToNumeric<int32_t, uint32_t>(input, result, error_message, scale);
}
template <>
bool TryCastFromDecimal::Operation(int64_t input, uint32_t &result, string *error_message, uint8_t width,
                                   uint8_t scale) {
	return TryCastDecimalToNumeric<int64_t, uint32_t>(input, result, error_message, scale);
}
template <>
bool TryCastFromDecimal::Operation(hugeint_t input, uint32_t &result, string *error_message, uint8_t width,
                                   uint8_t scale) {
	return TryCastHugeDecimalToNumeric<uint32_t>(input, result, error_message, scale);
}

//===--------------------------------------------------------------------===//
// Cast Decimal -> uint64_t
//===--------------------------------------------------------------------===//
template <>
bool TryCastFromDecimal::Operation(int16_t input, uint64_t &result, string *error_message, uint8_t width,
                                   uint8_t scale) {
	return TryCastDecimalToNumeric<int16_t, uint64_t>(input, result, error_message, scale);
}
template <>
bool TryCastFromDecimal::Operation(int32_t input, uint64_t &result, string *error_message, uint8_t width,
                                   uint8_t scale) {
	return TryCastDecimalToNumeric<int32_t, uint64_t>(input, result, error_message, scale);
}
template <>
bool TryCastFromDecimal::Operation(int64_t input, uint64_t &result, string *error_message, uint8_t width,
                                   uint8_t scale) {
	return TryCastDecimalToNumeric<int64_t, uint64_t>(input, result, error_message, scale);
}
template <>
bool TryCastFromDecimal::Operation(hugeint_t input, uint64_t &result, string *error_message, uint8_t width,
                                   uint8_t scale) {
	return TryCastHugeDecimalToNumeric<uint64_t>(input, result, error_message, scale);
}

//===--------------------------------------------------------------------===//
// Cast Decimal -> hugeint_t
//===--------------------------------------------------------------------===//
template <>
bool TryCastFromDecimal::Operation(int16_t input, hugeint_t &result, string *error_message, uint8_t width,
                                   uint8_t scale) {
	return TryCastDecimalToNumeric<int16_t, hugeint_t>(input, result, error_message, scale);
}
template <>
bool TryCastFromDecimal::Operation(int32_t input, hugeint_t &result, string *error_message, uint8_t width,
                                   uint8_t scale) {
	return TryCastDecimalToNumeric<int32_t, hugeint_t>(input, result, error_message, scale);
}
template <>
bool TryCastFromDecimal::Operation(int64_t input, hugeint_t &result, string *error_message, uint8_t width,
                                   uint8_t scale) {
	return TryCastDecimalToNumeric<int64_t, hugeint_t>(input, result, error_message, scale);
}
template <>
bool TryCastFromDecimal::Operation(hugeint_t input, hugeint_t &result, string *error_message, uint8_t width,
                                   uint8_t scale) {
	return TryCastHugeDecimalToNumeric<hugeint_t>(input, result, error_message, scale);
}

//===--------------------------------------------------------------------===//
// Decimal -> Float/Double Cast
//===--------------------------------------------------------------------===//
template <class SRC, class DST>
bool TryCastDecimalToFloatingPoint(SRC input, DST &result, uint8_t scale) {
	result = Cast::Operation<SRC, DST>(input) / DST(NumericHelper::DOUBLE_POWERS_OF_TEN[scale]);
	return true;
}

// DECIMAL -> FLOAT
template <>
bool TryCastFromDecimal::Operation(int16_t input, float &result, string *error_message, uint8_t width, uint8_t scale) {
	return TryCastDecimalToFloatingPoint<int16_t, float>(input, result, scale);
}

template <>
bool TryCastFromDecimal::Operation(int32_t input, float &result, string *error_message, uint8_t width, uint8_t scale) {
	return TryCastDecimalToFloatingPoint<int32_t, float>(input, result, scale);
}

template <>
bool TryCastFromDecimal::Operation(int64_t input, float &result, string *error_message, uint8_t width, uint8_t scale) {
	return TryCastDecimalToFloatingPoint<int64_t, float>(input, result, scale);
}

template <>
bool TryCastFromDecimal::Operation(hugeint_t input, float &result, string *error_message, uint8_t width,
                                   uint8_t scale) {
	return TryCastDecimalToFloatingPoint<hugeint_t, float>(input, result, scale);
}

// DECIMAL -> DOUBLE
template <>
bool TryCastFromDecimal::Operation(int16_t input, double &result, string *error_message, uint8_t width, uint8_t scale) {
	return TryCastDecimalToFloatingPoint<int16_t, double>(input, result, scale);
}

template <>
bool TryCastFromDecimal::Operation(int32_t input, double &result, string *error_message, uint8_t width, uint8_t scale) {
	return TryCastDecimalToFloatingPoint<int32_t, double>(input, result, scale);
}

template <>
bool TryCastFromDecimal::Operation(int64_t input, double &result, string *error_message, uint8_t width, uint8_t scale) {
	return TryCastDecimalToFloatingPoint<int64_t, double>(input, result, scale);
}

template <>
bool TryCastFromDecimal::Operation(hugeint_t input, double &result, string *error_message, uint8_t width,
                                   uint8_t scale) {
	return TryCastDecimalToFloatingPoint<hugeint_t, double>(input, result, scale);
}

} // namespace duckdb




namespace duckdb {

template <class T>
string StandardStringCast(T input) {
	Vector v(LogicalType::VARCHAR);
	return StringCast::Operation(input, v).GetString();
}

template <>
string ConvertToString::Operation(bool input) {
	return StandardStringCast(input);
}
template <>
string ConvertToString::Operation(int8_t input) {
	return StandardStringCast(input);
}
template <>
string ConvertToString::Operation(int16_t input) {
	return StandardStringCast(input);
}
template <>
string ConvertToString::Operation(int32_t input) {
	return StandardStringCast(input);
}
template <>
string ConvertToString::Operation(int64_t input) {
	return StandardStringCast(input);
}
template <>
string ConvertToString::Operation(uint8_t input) {
	return StandardStringCast(input);
}
template <>
string ConvertToString::Operation(uint16_t input) {
	return StandardStringCast(input);
}
template <>
string ConvertToString::Operation(uint32_t input) {
	return StandardStringCast(input);
}
template <>
string ConvertToString::Operation(uint64_t input) {
	return StandardStringCast(input);
}
template <>
string ConvertToString::Operation(hugeint_t input) {
	return StandardStringCast(input);
}
template <>
string ConvertToString::Operation(float input) {
	return StandardStringCast(input);
}
template <>
string ConvertToString::Operation(double input) {
	return StandardStringCast(input);
}
template <>
string ConvertToString::Operation(interval_t input) {
	return StandardStringCast(input);
}
template <>
string ConvertToString::Operation(date_t input) {
	return StandardStringCast(input);
}
template <>
string ConvertToString::Operation(dtime_t input) {
	return StandardStringCast(input);
}
template <>
string ConvertToString::Operation(timestamp_t input) {
	return StandardStringCast(input);
}
template <>
string ConvertToString::Operation(string_t input) {
	return input.GetString();
}

} // namespace duckdb










namespace duckdb {

//===--------------------------------------------------------------------===//
// Cast Numeric -> String
//===--------------------------------------------------------------------===//
template <>
string_t StringCast::Operation(bool input, Vector &vector) {
	if (input) {
		return StringVector::AddString(vector, "true", 4);
	} else {
		return StringVector::AddString(vector, "false", 5);
	}
}

template <>
string_t StringCast::Operation(int8_t input, Vector &vector) {
	return NumericHelper::FormatSigned<int8_t, uint8_t>(input, vector);
}

template <>
string_t StringCast::Operation(int16_t input, Vector &vector) {
	return NumericHelper::FormatSigned<int16_t, uint16_t>(input, vector);
}
template <>
string_t StringCast::Operation(int32_t input, Vector &vector) {
	return NumericHelper::FormatSigned<int32_t, uint32_t>(input, vector);
}

template <>
string_t StringCast::Operation(int64_t input, Vector &vector) {
	return NumericHelper::FormatSigned<int64_t, uint64_t>(input, vector);
}
template <>
duckdb::string_t StringCast::Operation(uint8_t input, Vector &vector) {
	return NumericHelper::FormatSigned<uint8_t, uint64_t>(input, vector);
}
template <>
duckdb::string_t StringCast::Operation(uint16_t input, Vector &vector) {
	return NumericHelper::FormatSigned<uint16_t, uint64_t>(input, vector);
}
template <>
duckdb::string_t StringCast::Operation(uint32_t input, Vector &vector) {
	return NumericHelper::FormatSigned<uint32_t, uint64_t>(input, vector);
}
template <>
duckdb::string_t StringCast::Operation(uint64_t input, Vector &vector) {
	return NumericHelper::FormatSigned<uint64_t, uint64_t>(input, vector);
}

template <>
string_t StringCast::Operation(float input, Vector &vector) {
	std::string s = duckdb_fmt::format("{}", input);
	return StringVector::AddString(vector, s);
}

template <>
string_t StringCast::Operation(double input, Vector &vector) {
	std::string s = duckdb_fmt::format("{}", input);
	return StringVector::AddString(vector, s);
}

template <>
string_t StringCast::Operation(interval_t input, Vector &vector) {
	char buffer[70];
	idx_t length = IntervalToStringCast::Format(input, buffer);
	return StringVector::AddString(vector, buffer, length);
}

template <>
duckdb::string_t StringCast::Operation(hugeint_t input, Vector &vector) {
	return HugeintToStringCast::FormatSigned(input, vector);
}

template <>
duckdb::string_t StringCast::Operation(date_t input, Vector &vector) {
	if (input == date_t::infinity()) {
		return StringVector::AddString(vector, Date::PINF);
	} else if (input == date_t::ninfinity()) {
		return StringVector::AddString(vector, Date::NINF);
	}
	int32_t date[3];
	Date::Convert(input, date[0], date[1], date[2]);

	idx_t year_length;
	bool add_bc;
	idx_t length = DateToStringCast::Length(date, year_length, add_bc);

	string_t result = StringVector::EmptyString(vector, length);
	auto data = result.GetDataWriteable();

	DateToStringCast::Format(data, date, year_length, add_bc);

	result.Finalize();
	return result;
}

template <>
duckdb::string_t StringCast::Operation(dtime_t input, Vector &vector) {
	int32_t time[4];
	Time::Convert(input, time[0], time[1], time[2], time[3]);

	char micro_buffer[10];
	idx_t length = TimeToStringCast::Length(time, micro_buffer);

	string_t result = StringVector::EmptyString(vector, length);
	auto data = result.GetDataWriteable();

	TimeToStringCast::Format(data, length, time, micro_buffer);

	result.Finalize();
	return result;
}

template <>
duckdb::string_t StringCast::Operation(timestamp_t input, Vector &vector) {
	if (input == timestamp_t::infinity()) {
		return StringVector::AddString(vector, Date::PINF);
	} else if (input == timestamp_t::ninfinity()) {
		return StringVector::AddString(vector, Date::NINF);
	}
	date_t date_entry;
	dtime_t time_entry;
	Timestamp::Convert(input, date_entry, time_entry);

	int32_t date[3], time[4];
	Date::Convert(date_entry, date[0], date[1], date[2]);
	Time::Convert(time_entry, time[0], time[1], time[2], time[3]);

	// format for timestamp is DATE TIME (separated by space)
	idx_t year_length;
	bool add_bc;
	char micro_buffer[6];
	idx_t date_length = DateToStringCast::Length(date, year_length, add_bc);
	idx_t time_length = TimeToStringCast::Length(time, micro_buffer);
	idx_t length = date_length + time_length + 1;

	string_t result = StringVector::EmptyString(vector, length);
	auto data = result.GetDataWriteable();

	DateToStringCast::Format(data, date, year_length, add_bc);
	data[date_length] = ' ';
	TimeToStringCast::Format(data + date_length + 1, time_length, time, micro_buffer);

	result.Finalize();
	return result;
}

template <>
duckdb::string_t StringCast::Operation(duckdb::string_t input, Vector &result) {
	return StringVector::AddStringOrBlob(result, input);
}

template <>
string_t StringCastTZ::Operation(dtime_t input, Vector &vector) {
	int32_t time[4];
	Time::Convert(input, time[0], time[1], time[2], time[3]);

	// format for timetz is TIME+00
	char micro_buffer[10];
	const auto time_length = TimeToStringCast::Length(time, micro_buffer);
	const idx_t length = time_length + 3;

	string_t result = StringVector::EmptyString(vector, length);
	auto data = result.GetDataWriteable();

	idx_t pos = 0;
	TimeToStringCast::Format(data + pos, length, time, micro_buffer);
	pos += time_length;
	data[pos++] = '+';
	data[pos++] = '0';
	data[pos++] = '0';

	result.Finalize();
	return result;
}

template <>
string_t StringCastTZ::Operation(timestamp_t input, Vector &vector) {
	if (input == timestamp_t::infinity()) {
		return StringVector::AddString(vector, Date::PINF);
	} else if (input == timestamp_t::ninfinity()) {
		return StringVector::AddString(vector, Date::NINF);
	}
	date_t date_entry;
	dtime_t time_entry;
	Timestamp::Convert(input, date_entry, time_entry);

	int32_t date[3], time[4];
	Date::Convert(date_entry, date[0], date[1], date[2]);
	Time::Convert(time_entry, time[0], time[1], time[2], time[3]);

	// format for timestamptz is DATE TIME+00 (separated by space)
	idx_t year_length;
	bool add_bc;
	char micro_buffer[6];
	const idx_t date_length = DateToStringCast::Length(date, year_length, add_bc);
	const idx_t time_length = TimeToStringCast::Length(time, micro_buffer);
	const idx_t length = date_length + 1 + time_length + 3;

	string_t result = StringVector::EmptyString(vector, length);
	auto data = result.GetDataWriteable();

	idx_t pos = 0;
	DateToStringCast::Format(data + pos, date, year_length, add_bc);
	pos += date_length;
	data[pos++] = ' ';
	TimeToStringCast::Format(data + pos, time_length, time, micro_buffer);
	pos += time_length;
	data[pos++] = '+';
	data[pos++] = '0';
	data[pos++] = '0';

	result.Finalize();
	return result;
}

} // namespace duckdb





namespace duckdb {
class PipeFile : public FileHandle {
public:
	PipeFile(unique_ptr<FileHandle> child_handle_p, const string &path)
	    : FileHandle(pipe_fs, path), child_handle(std::move(child_handle_p)) {
	}

	PipeFileSystem pipe_fs;
	unique_ptr<FileHandle> child_handle;

public:
	int64_t ReadChunk(void *buffer, int64_t nr_bytes);
	int64_t WriteChunk(void *buffer, int64_t nr_bytes);

	void Close() override {
	}
};

int64_t PipeFile::ReadChunk(void *buffer, int64_t nr_bytes) {
	return child_handle->Read(buffer, nr_bytes);
}
int64_t PipeFile::WriteChunk(void *buffer, int64_t nr_bytes) {
	return child_handle->Write(buffer, nr_bytes);
}

void PipeFileSystem::Reset(FileHandle &handle) {
	throw InternalException("Cannot reset pipe file system");
}

int64_t PipeFileSystem::Read(FileHandle &handle, void *buffer, int64_t nr_bytes) {
	auto &pipe = (PipeFile &)handle;
	return pipe.ReadChunk(buffer, nr_bytes);
}

int64_t PipeFileSystem::Write(FileHandle &handle, void *buffer, int64_t nr_bytes) {
	auto &pipe = (PipeFile &)handle;
	return pipe.WriteChunk(buffer, nr_bytes);
}

int64_t PipeFileSystem::GetFileSize(FileHandle &handle) {
	return 0;
}

void PipeFileSystem::FileSync(FileHandle &handle) {
}

unique_ptr<FileHandle> PipeFileSystem::OpenPipe(unique_ptr<FileHandle> handle) {
	auto path = handle->path;
	return make_uniq<PipeFile>(std::move(handle), path);
}

} // namespace duckdb







namespace duckdb {

PreservedError::PreservedError() : initialized(false), exception_instance(nullptr) {
}

PreservedError::PreservedError(const Exception &exception)
    : initialized(true), type(exception.type), raw_message(SanitizeErrorMessage(exception.RawMessage())),
      exception_instance(exception.Copy()) {
}

PreservedError::PreservedError(const string &message)
    : initialized(true), type(ExceptionType::INVALID), raw_message(SanitizeErrorMessage(message)),
      exception_instance(nullptr) {
}

const string &PreservedError::Message() {
	if (final_message.empty()) {
		final_message = Exception::ExceptionTypeToString(type) + " Error: " + raw_message;
	}
	return final_message;
}

string PreservedError::SanitizeErrorMessage(string error) {
	return StringUtil::Replace(std::move(error), string("\0", 1), "\\0");
}

void PreservedError::Throw(const string &prepended_message) const {
	D_ASSERT(initialized);
	if (!prepended_message.empty()) {
		string new_message = prepended_message + raw_message;
		Exception::ThrowAsTypeWithMessage(type, new_message, exception_instance);
	}
	Exception::ThrowAsTypeWithMessage(type, raw_message, exception_instance);
}

const ExceptionType &PreservedError::Type() const {
	D_ASSERT(initialized);
	return this->type;
}

PreservedError &PreservedError::AddToMessage(const string &prepended_message) {
	raw_message = prepended_message + raw_message;
	return *this;
}

PreservedError::operator bool() const {
	return initialized;
}

bool PreservedError::operator==(const PreservedError &other) const {
	if (initialized != other.initialized) {
		return false;
	}
	if (type != other.type) {
		return false;
	}
	return raw_message == other.raw_message;
}

} // namespace duckdb




#include <stdio.h>

#ifndef DUCKDB_DISABLE_PRINT
#ifdef DUCKDB_WINDOWS
#include <io.h>
#else
#include <sys/ioctl.h>
#include <stdio.h>
#include <unistd.h>
#endif
#endif

namespace duckdb {

void Printer::RawPrint(OutputStream stream, const string &str) {
#ifndef DUCKDB_DISABLE_PRINT
#ifdef DUCKDB_WINDOWS
	if (IsTerminal(stream)) {
		// print utf8 to terminal
		auto unicode = WindowsUtil::UTF8ToMBCS(str.c_str());
		fprintf(stream == OutputStream::STREAM_STDERR ? stderr : stdout, "%s", unicode.c_str());
		return;
	}
#endif
	fprintf(stream == OutputStream::STREAM_STDERR ? stderr : stdout, "%s", str.c_str());
#endif
}

// LCOV_EXCL_START
void Printer::Print(OutputStream stream, const string &str) {
	Printer::RawPrint(stream, str);
	Printer::RawPrint(stream, "\n");
}
void Printer::Flush(OutputStream stream) {
#ifndef DUCKDB_DISABLE_PRINT
	fflush(stream == OutputStream::STREAM_STDERR ? stderr : stdout);
#endif
}

void Printer::Print(const string &str) {
	Printer::Print(OutputStream::STREAM_STDERR, str);
}

bool Printer::IsTerminal(OutputStream stream) {
#ifndef DUCKDB_DISABLE_PRINT
#ifdef DUCKDB_WINDOWS
	auto stream_handle = stream == OutputStream::STREAM_STDERR ? STD_ERROR_HANDLE : STD_OUTPUT_HANDLE;
	return GetFileType(GetStdHandle(stream_handle)) == FILE_TYPE_CHAR;
#else
	return isatty(stream == OutputStream::STREAM_STDERR ? 2 : 1);
#endif
#else
	throw InternalException("IsTerminal called while printing is disabled");
#endif
}

idx_t Printer::TerminalWidth() {
#ifndef DUCKDB_DISABLE_PRINT
#ifdef DUCKDB_WINDOWS
	CONSOLE_SCREEN_BUFFER_INFO csbi;
	int columns, rows;

	GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
	rows = csbi.srWindow.Right - csbi.srWindow.Left + 1;
	return rows;
#else
	struct winsize w;
	ioctl(0, TIOCGWINSZ, &w);
	return w.ws_col;
#endif
#else
	throw InternalException("TerminalWidth called while printing is disabled");
#endif
}
// LCOV_EXCL_STOP

} // namespace duckdb




namespace duckdb {

void ProgressBar::SystemOverrideCheck(ClientConfig &config) {
	if (config.system_progress_bar_disable_reason != nullptr) {
		throw InvalidInputException("Could not change the progress bar setting because: '%s'",
		                            config.system_progress_bar_disable_reason);
	}
}

unique_ptr<ProgressBarDisplay> ProgressBar::DefaultProgressBarDisplay() {
	return make_uniq<TerminalProgressBarDisplay>();
}

ProgressBar::ProgressBar(Executor &executor, idx_t show_progress_after,
                         progress_bar_display_create_func_t create_display_func)
    : executor(executor), show_progress_after(show_progress_after), current_percentage(-1) {
	if (create_display_func) {
		display = create_display_func();
	}
}

double ProgressBar::GetCurrentPercentage() {
	return current_percentage;
}

void ProgressBar::Start() {
	profiler.Start();
	current_percentage = 0;
	supported = true;
}

bool ProgressBar::PrintEnabled() const {
	return display != nullptr;
}

bool ProgressBar::ShouldPrint(bool final) const {
	if (!PrintEnabled()) {
		// Don't print progress at all
		return false;
	}
	// FIXME - do we need to check supported before running `profiler.Elapsed()` ?
	auto sufficient_time_elapsed = profiler.Elapsed() > show_progress_after / 1000.0;
	if (!sufficient_time_elapsed) {
		// Don't print yet
		return false;
	}
	if (final) {
		// Print the last completed bar
		return true;
	}
	if (!supported) {
		return false;
	}
	return current_percentage > -1;
}

void ProgressBar::Update(bool final) {
	if (!final && !supported) {
		return;
	}
	double new_percentage;
	supported = executor.GetPipelinesProgress(new_percentage);
	if (!final && !supported) {
		return;
	}
	if (new_percentage > current_percentage) {
		current_percentage = new_percentage;
	}
	if (ShouldPrint(final)) {
#ifndef DUCKDB_DISABLE_PRINT
		if (final) {
			FinishProgressBarPrint();
		} else {
			PrintProgress(current_percentage);
		}
#endif
	}
}

void ProgressBar::PrintProgress(int current_percentage) {
	D_ASSERT(display);
	display->Update(current_percentage);
}

void ProgressBar::FinishProgressBarPrint() {
	if (finished) {
		return;
	}
	D_ASSERT(display);
	display->Finish();
	finished = true;
}

} // namespace duckdb




namespace duckdb {

void TerminalProgressBarDisplay::PrintProgressInternal(int percentage) {
	if (percentage > 100) {
		percentage = 100;
	}
	if (percentage < 0) {
		percentage = 0;
	}
	string result;
	// we divide the number of blocks by the percentage
	// 0%   = 0
	// 100% = PROGRESS_BAR_WIDTH
	// the percentage determines how many blocks we need to draw
	double blocks_to_draw = PROGRESS_BAR_WIDTH * (percentage / 100.0);
	// because of the power of unicode, we can also draw partial blocks

	// render the percentage with some padding to ensure everything stays nicely aligned
	result = "\r";
	if (percentage < 100) {
		result += " ";
	}
	if (percentage < 10) {
		result += " ";
	}
	result += to_string(percentage) + "%";
	result += " ";
	result += PROGRESS_START;
	idx_t i;
	for (i = 0; i < idx_t(blocks_to_draw); i++) {
		result += PROGRESS_BLOCK;
	}
	if (i < PROGRESS_BAR_WIDTH) {
		// print a partial block based on the percentage of the progress bar remaining
		idx_t index = idx_t((blocks_to_draw - idx_t(blocks_to_draw)) * PARTIAL_BLOCK_COUNT);
		if (index >= PARTIAL_BLOCK_COUNT) {
			index = PARTIAL_BLOCK_COUNT - 1;
		}
		result += PROGRESS_PARTIAL[index];
		i++;
	}
	for (; i < PROGRESS_BAR_WIDTH; i++) {
		result += PROGRESS_EMPTY;
	}
	result += PROGRESS_END;
	result += " ";

	Printer::RawPrint(OutputStream::STREAM_STDOUT, result);
}

void TerminalProgressBarDisplay::Update(double percentage) {
	PrintProgressInternal(percentage);
	Printer::Flush(OutputStream::STREAM_STDOUT);
}

void TerminalProgressBarDisplay::Finish() {
	PrintProgressInternal(100);
	Printer::RawPrint(OutputStream::STREAM_STDOUT, "\n");
	Printer::Flush(OutputStream::STREAM_STDOUT);
}

} // namespace duckdb








namespace duckdb {

template <class OP, class RETURN_TYPE, typename... ARGS>
RETURN_TYPE RadixBitsSwitch(idx_t radix_bits, ARGS &&... args) {
	D_ASSERT(radix_bits <= sizeof(hash_t) * 8);
	switch (radix_bits) {
	case 1:
		return OP::template Operation<1>(std::forward<ARGS>(args)...);
	case 2:
		return OP::template Operation<2>(std::forward<ARGS>(args)...);
	case 3:
		return OP::template Operation<3>(std::forward<ARGS>(args)...);
	case 4:
		return OP::template Operation<4>(std::forward<ARGS>(args)...);
	case 5:
		return OP::template Operation<5>(std::forward<ARGS>(args)...);
	case 6:
		return OP::template Operation<6>(std::forward<ARGS>(args)...);
	case 7:
		return OP::template Operation<7>(std::forward<ARGS>(args)...);
	case 8:
		return OP::template Operation<8>(std::forward<ARGS>(args)...);
	case 9:
		return OP::template Operation<9>(std::forward<ARGS>(args)...);
	case 10:
		return OP::template Operation<10>(std::forward<ARGS>(args)...);
	default:
		throw InternalException("TODO");
	}
}

template <idx_t radix_bits>
struct RadixLessThan {
	static inline bool Operation(hash_t hash, hash_t cutoff) {
		using CONSTANTS = RadixPartitioningConstants<radix_bits>;
		return CONSTANTS::ApplyMask(hash) < cutoff;
	}
};

struct SelectFunctor {
	template <idx_t radix_bits>
	static idx_t Operation(Vector &hashes, const SelectionVector *sel, idx_t count, idx_t cutoff,
	                       SelectionVector *true_sel, SelectionVector *false_sel) {
		Vector cutoff_vector(Value::HASH(cutoff));
		return BinaryExecutor::Select<hash_t, hash_t, RadixLessThan<radix_bits>>(hashes, cutoff_vector, sel, count,
		                                                                         true_sel, false_sel);
	}
};

idx_t RadixPartitioning::Select(Vector &hashes, const SelectionVector *sel, idx_t count, idx_t radix_bits, idx_t cutoff,
                                SelectionVector *true_sel, SelectionVector *false_sel) {
	return RadixBitsSwitch<SelectFunctor, idx_t>(radix_bits, hashes, sel, count, cutoff, true_sel, false_sel);
}

struct HashsToBinsFunctor {
	template <idx_t radix_bits>
	static void Operation(Vector &hashes, Vector &bins, idx_t count) {
		using CONSTANTS = RadixPartitioningConstants<radix_bits>;
		UnaryExecutor::Execute<hash_t, hash_t>(hashes, bins, count,
		                                       [&](hash_t hash) { return CONSTANTS::ApplyMask(hash); });
	}
};

void RadixPartitioning::HashesToBins(Vector &hashes, idx_t radix_bits, Vector &bins, idx_t count) {
	return RadixBitsSwitch<HashsToBinsFunctor, void>(radix_bits, hashes, bins, count);
}

//===--------------------------------------------------------------------===//
// Row Data Partitioning
//===--------------------------------------------------------------------===//
template <idx_t radix_bits>
static void InitPartitions(BufferManager &buffer_manager, vector<unique_ptr<RowDataCollection>> &partition_collections,
                           RowDataBlock *partition_blocks[], vector<BufferHandle> &partition_handles,
                           data_ptr_t partition_ptrs[], idx_t block_capacity, idx_t row_width) {
	using CONSTANTS = RadixPartitioningConstants<radix_bits>;

	partition_collections.reserve(CONSTANTS::NUM_PARTITIONS);
	partition_handles.reserve(CONSTANTS::NUM_PARTITIONS);
	for (idx_t i = 0; i < CONSTANTS::NUM_PARTITIONS; i++) {
		partition_collections.push_back(make_uniq<RowDataCollection>(buffer_manager, block_capacity, row_width));
		partition_blocks[i] = &partition_collections[i]->CreateBlock();
		partition_handles.push_back(buffer_manager.Pin(partition_blocks[i]->block));
		if (partition_ptrs) {
			partition_ptrs[i] = partition_handles[i].Ptr();
		}
	}
}

struct ComputePartitionIndicesFunctor {
	template <idx_t radix_bits>
	static void Operation(Vector &hashes, Vector &partition_indices, idx_t count) {
		UnaryExecutor::Execute<hash_t, hash_t>(hashes, partition_indices, count, [&](hash_t hash) {
			using CONSTANTS = RadixPartitioningConstants<radix_bits>;
			return CONSTANTS::ApplyMask(hash);
		});
	}
};

//===--------------------------------------------------------------------===//
// Column Data Partitioning
//===--------------------------------------------------------------------===//
RadixPartitionedColumnData::RadixPartitionedColumnData(ClientContext &context_p, vector<LogicalType> types_p,
                                                       idx_t radix_bits_p, idx_t hash_col_idx_p)
    : PartitionedColumnData(PartitionedColumnDataType::RADIX, context_p, std::move(types_p)), radix_bits(radix_bits_p),
      hash_col_idx(hash_col_idx_p) {
	D_ASSERT(hash_col_idx < types.size());
	const auto num_partitions = RadixPartitioning::NumberOfPartitions(radix_bits);
	allocators->allocators.reserve(num_partitions);
	for (idx_t i = 0; i < num_partitions; i++) {
		CreateAllocator();
	}
	D_ASSERT(allocators->allocators.size() == num_partitions);
}

RadixPartitionedColumnData::RadixPartitionedColumnData(const RadixPartitionedColumnData &other)
    : PartitionedColumnData(other), radix_bits(other.radix_bits), hash_col_idx(other.hash_col_idx) {
	for (idx_t i = 0; i < RadixPartitioning::NumberOfPartitions(radix_bits); i++) {
		partitions.emplace_back(CreatePartitionCollection(i));
	}
}

RadixPartitionedColumnData::~RadixPartitionedColumnData() {
}

void RadixPartitionedColumnData::InitializeAppendStateInternal(PartitionedColumnDataAppendState &state) const {
	const auto num_partitions = RadixPartitioning::NumberOfPartitions(radix_bits);
	state.partition_append_states.reserve(num_partitions);
	state.partition_buffers.reserve(num_partitions);
	for (idx_t i = 0; i < num_partitions; i++) {
		state.partition_append_states.emplace_back(make_uniq<ColumnDataAppendState>());
		partitions[i]->InitializeAppend(*state.partition_append_states[i]);
		state.partition_buffers.emplace_back(CreatePartitionBuffer());
	}
}

void RadixPartitionedColumnData::ComputePartitionIndices(PartitionedColumnDataAppendState &state, DataChunk &input) {
	D_ASSERT(partitions.size() == RadixPartitioning::NumberOfPartitions(radix_bits));
	D_ASSERT(state.partition_buffers.size() == RadixPartitioning::NumberOfPartitions(radix_bits));
	RadixBitsSwitch<ComputePartitionIndicesFunctor, void>(radix_bits, input.data[hash_col_idx], state.partition_indices,
	                                                      input.size());
}

//===--------------------------------------------------------------------===//
// Tuple Data Partitioning
//===--------------------------------------------------------------------===//
RadixPartitionedTupleData::RadixPartitionedTupleData(BufferManager &buffer_manager, const TupleDataLayout &layout_p,
                                                     idx_t radix_bits_p, idx_t hash_col_idx_p)
    : PartitionedTupleData(PartitionedTupleDataType::RADIX, buffer_manager, layout_p.Copy()), radix_bits(radix_bits_p),
      hash_col_idx(hash_col_idx_p) {
	D_ASSERT(hash_col_idx < layout.GetTypes().size());
	const auto num_partitions = RadixPartitioning::NumberOfPartitions(radix_bits);
	allocators->allocators.reserve(num_partitions);
	for (idx_t i = 0; i < num_partitions; i++) {
		CreateAllocator();
	}
	D_ASSERT(allocators->allocators.size() == num_partitions);
	Initialize();
}

RadixPartitionedTupleData::RadixPartitionedTupleData(const RadixPartitionedTupleData &other)
    : PartitionedTupleData(other), radix_bits(other.radix_bits), hash_col_idx(other.hash_col_idx) {
	Initialize();
}

RadixPartitionedTupleData::~RadixPartitionedTupleData() {
}

void RadixPartitionedTupleData::Initialize() {
	for (idx_t i = 0; i < RadixPartitioning::NumberOfPartitions(radix_bits); i++) {
		partitions.emplace_back(CreatePartitionCollection(i));
	}
}

void RadixPartitionedTupleData::InitializeAppendStateInternal(PartitionedTupleDataAppendState &state,
                                                              TupleDataPinProperties properties) const {
	// Init pin state per partition
	const auto num_partitions = RadixPartitioning::NumberOfPartitions(radix_bits);
	state.partition_pin_states.reserve(num_partitions);
	for (idx_t i = 0; i < num_partitions; i++) {
		state.partition_pin_states.emplace_back(make_uniq<TupleDataPinState>());
		partitions[i]->InitializeAppend(*state.partition_pin_states[i], properties);
	}

	// Init single chunk state
	auto column_count = layout.ColumnCount();
	vector<column_t> column_ids;
	column_ids.reserve(column_count);
	for (idx_t col_idx = 0; col_idx < column_count; col_idx++) {
		column_ids.emplace_back(col_idx);
	}
	partitions[0]->InitializeAppend(state.chunk_state, std::move(column_ids));
}

void RadixPartitionedTupleData::ComputePartitionIndices(PartitionedTupleDataAppendState &state, DataChunk &input) {
	D_ASSERT(partitions.size() == RadixPartitioning::NumberOfPartitions(radix_bits));
	RadixBitsSwitch<ComputePartitionIndicesFunctor, void>(radix_bits, input.data[hash_col_idx], state.partition_indices,
	                                                      input.size());
}

void RadixPartitionedTupleData::ComputePartitionIndices(Vector &row_locations, idx_t count,
                                                        Vector &partition_indices) const {
	Vector intermediate(LogicalType::HASH);
	partitions[0]->Gather(row_locations, *FlatVector::IncrementalSelectionVector(), count, hash_col_idx, intermediate,
	                      *FlatVector::IncrementalSelectionVector());
	RadixBitsSwitch<ComputePartitionIndicesFunctor, void>(radix_bits, intermediate, partition_indices, count);
}

void RadixPartitionedTupleData::RepartitionFinalizeStates(PartitionedTupleData &old_partitioned_data,
                                                          PartitionedTupleData &new_partitioned_data,
                                                          PartitionedTupleDataAppendState &state,
                                                          idx_t finished_partition_idx) const {
	D_ASSERT(old_partitioned_data.GetType() == PartitionedTupleDataType::RADIX &&
	         new_partitioned_data.GetType() == PartitionedTupleDataType::RADIX);
	const auto &old_radix_partitions = (RadixPartitionedTupleData &)old_partitioned_data;
	const auto &new_radix_partitions = (RadixPartitionedTupleData &)new_partitioned_data;
	const auto old_radix_bits = old_radix_partitions.GetRadixBits();
	const auto new_radix_bits = new_radix_partitions.GetRadixBits();
	D_ASSERT(new_radix_bits > old_radix_bits);

	// We take the most significant digits as the partition index
	// When repartitioning, e.g., partition 0 from "old" goes into the first N partitions in "new"
	// When partition 0 is done, we can already finalize the append states, unpinning blocks
	const auto multiplier = RadixPartitioning::NumberOfPartitions(new_radix_bits - old_radix_bits);
	const auto from_idx = finished_partition_idx * multiplier;
	const auto to_idx = from_idx + multiplier;
	auto &partitions = new_partitioned_data.GetPartitions();
	for (idx_t partition_index = from_idx; partition_index < to_idx; partition_index++) {
		auto &partition = *partitions[partition_index];
		auto &partition_pin_state = *state.partition_pin_states[partition_index];
		partition.FinalizePinState(partition_pin_state);
	}
}

} // namespace duckdb


#include <random>

namespace duckdb {

struct RandomState {
	RandomState() {
	}

	pcg32 pcg;
};

RandomEngine::RandomEngine(int64_t seed) : random_state(make_uniq<RandomState>()) {
	if (seed < 0) {
		random_state->pcg.seed(pcg_extras::seed_seq_from<std::random_device>());
	} else {
		random_state->pcg.seed(seed);
	}
}

RandomEngine::~RandomEngine() {
}

double RandomEngine::NextRandom(double min, double max) {
	D_ASSERT(max >= min);
	return min + (NextRandom() * (max - min));
}

double RandomEngine::NextRandom() {
	return random_state->pcg() / double(std::numeric_limits<uint32_t>::max());
}
uint32_t RandomEngine::NextRandomInteger() {
	return random_state->pcg();
}

void RandomEngine::SetSeed(uint32_t seed) {
	random_state->pcg.seed(seed);
}

} // namespace duckdb

#include <memory>




namespace duckdb_re2 {

Regex::Regex(const std::string &pattern, RegexOptions options) {
	RE2::Options o;
	o.set_case_sensitive(options == RegexOptions::CASE_INSENSITIVE);
	regex = std::make_shared<duckdb_re2::RE2>(StringPiece(pattern), o);
}

bool RegexSearchInternal(const char *input, Match &match, const Regex &r, RE2::Anchor anchor, size_t start,
                         size_t end) {
	auto &regex = r.GetRegex();
	duckdb::vector<StringPiece> target_groups;
	auto group_count = regex.NumberOfCapturingGroups() + 1;
	target_groups.resize(group_count);
	match.groups.clear();
	if (!regex.Match(StringPiece(input), start, end, anchor, target_groups.data(), group_count)) {
		return false;
	}
	for (auto &group : target_groups) {
		GroupMatch group_match;
		group_match.text = group.ToString();
		group_match.position = group.data() - input;
		match.groups.emplace_back(group_match);
	}
	return true;
}

bool RegexSearch(const std::string &input, Match &match, const Regex &regex) {
	return RegexSearchInternal(input.c_str(), match, regex, RE2::UNANCHORED, 0, input.size());
}

bool RegexMatch(const std::string &input, Match &match, const Regex &regex) {
	return RegexSearchInternal(input.c_str(), match, regex, RE2::ANCHOR_BOTH, 0, input.size());
}

bool RegexMatch(const char *start, const char *end, Match &match, const Regex &regex) {
	return RegexSearchInternal(start, match, regex, RE2::ANCHOR_BOTH, 0, end - start);
}

bool RegexMatch(const std::string &input, const Regex &regex) {
	Match nop_match;
	return RegexSearchInternal(input.c_str(), nop_match, regex, RE2::ANCHOR_BOTH, 0, input.size());
}

duckdb::vector<Match> RegexFindAll(const std::string &input, const Regex &regex) {
	duckdb::vector<Match> matches;
	size_t position = 0;
	Match match;
	while (RegexSearchInternal(input.c_str(), match, regex, RE2::UNANCHORED, position, input.size())) {
		position += match.position(0) + match.length(0);
		matches.emplace_back(std::move(match));
	}
	return matches;
}

} // namespace duckdb_re2
//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/common/types/row_operations/row_aggregate.cpp
//
//
//===----------------------------------------------------------------------===//





namespace duckdb {

void RowOperations::InitializeStates(TupleDataLayout &layout, Vector &addresses, const SelectionVector &sel,
                                     idx_t count) {
	if (count == 0) {
		return;
	}
	auto pointers = FlatVector::GetData<data_ptr_t>(addresses);
	auto &offsets = layout.GetOffsets();
	auto aggr_idx = layout.ColumnCount();

	for (const auto &aggr : layout.GetAggregates()) {
		for (idx_t i = 0; i < count; ++i) {
			auto row_idx = sel.get_index(i);
			auto row = pointers[row_idx];
			aggr.function.initialize(row + offsets[aggr_idx]);
		}
		++aggr_idx;
	}
}

void RowOperations::DestroyStates(RowOperationsState &state, TupleDataLayout &layout, Vector &addresses, idx_t count) {
	if (count == 0) {
		return;
	}
	//	Move to the first aggregate state
	VectorOperations::AddInPlace(addresses, layout.GetAggrOffset(), count);
	for (const auto &aggr : layout.GetAggregates()) {
		if (aggr.function.destructor) {
			AggregateInputData aggr_input_data(aggr.GetFunctionData(), state.allocator);
			aggr.function.destructor(addresses, aggr_input_data, count);
		}
		// Move to the next aggregate state
		VectorOperations::AddInPlace(addresses, aggr.payload_size, count);
	}
}

void RowOperations::UpdateStates(RowOperationsState &state, AggregateObject &aggr, Vector &addresses,
                                 DataChunk &payload, idx_t arg_idx, idx_t count) {
	AggregateInputData aggr_input_data(aggr.GetFunctionData(), state.allocator);
	aggr.function.update(aggr.child_count == 0 ? nullptr : &payload.data[arg_idx], aggr_input_data, aggr.child_count,
	                     addresses, count);
}

void RowOperations::UpdateFilteredStates(RowOperationsState &state, AggregateFilterData &filter_data,
                                         AggregateObject &aggr, Vector &addresses, DataChunk &payload, idx_t arg_idx) {
	idx_t count = filter_data.ApplyFilter(payload);
	if (count == 0) {
		return;
	}

	Vector filtered_addresses(addresses, filter_data.true_sel, count);
	filtered_addresses.Flatten(count);

	UpdateStates(state, aggr, filtered_addresses, filter_data.filtered_payload, arg_idx, count);
}

void RowOperations::CombineStates(RowOperationsState &state, TupleDataLayout &layout, Vector &sources, Vector &targets,
                                  idx_t count) {
	if (count == 0) {
		return;
	}

	//	Move to the first aggregate states
	VectorOperations::AddInPlace(sources, layout.GetAggrOffset(), count);
	VectorOperations::AddInPlace(targets, layout.GetAggrOffset(), count);
	for (auto &aggr : layout.GetAggregates()) {
		D_ASSERT(aggr.function.combine);
		AggregateInputData aggr_input_data(aggr.GetFunctionData(), state.allocator);
		aggr.function.combine(sources, targets, aggr_input_data, count);

		// Move to the next aggregate states
		VectorOperations::AddInPlace(sources, aggr.payload_size, count);
		VectorOperations::AddInPlace(targets, aggr.payload_size, count);
	}
}

void RowOperations::FinalizeStates(RowOperationsState &state, TupleDataLayout &layout, Vector &addresses,
                                   DataChunk &result, idx_t aggr_idx) {
	//	Move to the first aggregate state
	VectorOperations::AddInPlace(addresses, layout.GetAggrOffset(), result.size());

	auto &aggregates = layout.GetAggregates();
	for (idx_t i = 0; i < aggregates.size(); i++) {
		auto &target = result.data[aggr_idx + i];
		auto &aggr = aggregates[i];
		AggregateInputData aggr_input_data(aggr.GetFunctionData(), state.allocator);
		aggr.function.finalize(addresses, aggr_input_data, target, result.size(), 0);

		// Move to the next aggregate state
		VectorOperations::AddInPlace(addresses, aggr.payload_size, result.size());
	}
}

} // namespace duckdb
//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/common/types/row_operations/row_external.cpp
//
//
//===----------------------------------------------------------------------===//



namespace duckdb {

using ValidityBytes = RowLayout::ValidityBytes;

void RowOperations::SwizzleColumns(const RowLayout &layout, const data_ptr_t base_row_ptr, const idx_t count) {
	const idx_t row_width = layout.GetRowWidth();
	data_ptr_t heap_row_ptrs[STANDARD_VECTOR_SIZE];
	idx_t done = 0;
	while (done != count) {
		const idx_t next = MinValue<idx_t>(count - done, STANDARD_VECTOR_SIZE);
		const data_ptr_t row_ptr = base_row_ptr + done * row_width;
		// Load heap row pointers
		data_ptr_t heap_ptr_ptr = row_ptr + layout.GetHeapOffset();
		for (idx_t i = 0; i < next; i++) {
			heap_row_ptrs[i] = Load<data_ptr_t>(heap_ptr_ptr);
			heap_ptr_ptr += row_width;
		}
		// Loop through the blob columns
		for (idx_t col_idx = 0; col_idx < layout.ColumnCount(); col_idx++) {
			auto physical_type = layout.GetTypes()[col_idx].InternalType();
			if (TypeIsConstantSize(physical_type)) {
				continue;
			}
			data_ptr_t col_ptr = row_ptr + layout.GetOffsets()[col_idx];
			if (physical_type == PhysicalType::VARCHAR) {
				data_ptr_t string_ptr = col_ptr + string_t::HEADER_SIZE;
				for (idx_t i = 0; i < next; i++) {
					if (Load<uint32_t>(col_ptr) > string_t::INLINE_LENGTH) {
						// Overwrite the string pointer with the within-row offset (if not inlined)
						Store<idx_t>(Load<data_ptr_t>(string_ptr) - heap_row_ptrs[i], string_ptr);
					}
					col_ptr += row_width;
					string_ptr += row_width;
				}
			} else {
				// Non-varchar blob columns
				for (idx_t i = 0; i < next; i++) {
					// Overwrite the column data pointer with the within-row offset
					Store<idx_t>(Load<data_ptr_t>(col_ptr) - heap_row_ptrs[i], col_ptr);
					col_ptr += row_width;
				}
			}
		}
		done += next;
	}
}

void RowOperations::SwizzleHeapPointer(const RowLayout &layout, data_ptr_t row_ptr, const data_ptr_t heap_base_ptr,
                                       const idx_t count, const idx_t base_offset) {
	const idx_t row_width = layout.GetRowWidth();
	row_ptr += layout.GetHeapOffset();
	idx_t cumulative_offset = 0;
	for (idx_t i = 0; i < count; i++) {
		Store<idx_t>(base_offset + cumulative_offset, row_ptr);
		cumulative_offset += Load<uint32_t>(heap_base_ptr + cumulative_offset);
		row_ptr += row_width;
	}
}

void RowOperations::CopyHeapAndSwizzle(const RowLayout &layout, data_ptr_t row_ptr, const data_ptr_t heap_base_ptr,
                                       data_ptr_t heap_ptr, const idx_t count) {
	const auto row_width = layout.GetRowWidth();
	const auto heap_offset = layout.GetHeapOffset();
	for (idx_t i = 0; i < count; i++) {
		// Figure out source and size
		const auto source_heap_ptr = Load<data_ptr_t>(row_ptr + heap_offset);
		const auto size = Load<uint32_t>(source_heap_ptr);
		D_ASSERT(size >= sizeof(uint32_t));

		// Copy and swizzle
		memcpy(heap_ptr, source_heap_ptr, size);
		Store<idx_t>(heap_ptr - heap_base_ptr, row_ptr + heap_offset);

		// Increment for next iteration
		row_ptr += row_width;
		heap_ptr += size;
	}
}

void RowOperations::UnswizzleHeapPointer(const RowLayout &layout, const data_ptr_t base_row_ptr,
                                         const data_ptr_t base_heap_ptr, const idx_t count) {
	const auto row_width = layout.GetRowWidth();
	data_ptr_t heap_ptr_ptr = base_row_ptr + layout.GetHeapOffset();
	for (idx_t i = 0; i < count; i++) {
		Store<data_ptr_t>(base_heap_ptr + Load<idx_t>(heap_ptr_ptr), heap_ptr_ptr);
		heap_ptr_ptr += row_width;
	}
}

static inline void VerifyUnswizzledString(const RowLayout &layout, const idx_t &col_idx, const data_ptr_t &row_ptr) {
#ifdef DEBUG
	if (layout.GetTypes()[col_idx] == LogicalTypeId::BLOB) {
		return;
	}
	idx_t entry_idx;
	idx_t idx_in_entry;
	ValidityBytes::GetEntryIndex(col_idx, entry_idx, idx_in_entry);

	ValidityBytes row_mask(row_ptr);
	if (row_mask.RowIsValid(row_mask.GetValidityEntry(entry_idx), idx_in_entry)) {
		auto str = Load<string_t>(row_ptr + layout.GetOffsets()[col_idx]);
		str.Verify();
	}
#endif
}

void RowOperations::UnswizzlePointers(const RowLayout &layout, const data_ptr_t base_row_ptr,
                                      const data_ptr_t base_heap_ptr, const idx_t count) {
	const idx_t row_width = layout.GetRowWidth();
	data_ptr_t heap_row_ptrs[STANDARD_VECTOR_SIZE];
	idx_t done = 0;
	while (done != count) {
		const idx_t next = MinValue<idx_t>(count - done, STANDARD_VECTOR_SIZE);
		const data_ptr_t row_ptr = base_row_ptr + done * row_width;
		// Restore heap row pointers
		data_ptr_t heap_ptr_ptr = row_ptr + layout.GetHeapOffset();
		for (idx_t i = 0; i < next; i++) {
			heap_row_ptrs[i] = base_heap_ptr + Load<idx_t>(heap_ptr_ptr);
			Store<data_ptr_t>(heap_row_ptrs[i], heap_ptr_ptr);
			heap_ptr_ptr += row_width;
		}
		// Loop through the blob columns
		for (idx_t col_idx = 0; col_idx < layout.ColumnCount(); col_idx++) {
			auto physical_type = layout.GetTypes()[col_idx].InternalType();
			if (TypeIsConstantSize(physical_type)) {
				continue;
			}
			data_ptr_t col_ptr = row_ptr + layout.GetOffsets()[col_idx];
			if (physical_type == PhysicalType::VARCHAR) {
				data_ptr_t string_ptr = col_ptr + string_t::HEADER_SIZE;
				for (idx_t i = 0; i < next; i++) {
					if (Load<uint32_t>(col_ptr) > string_t::INLINE_LENGTH) {
						// Overwrite the string offset with the pointer (if not inlined)
						Store<data_ptr_t>(heap_row_ptrs[i] + Load<idx_t>(string_ptr), string_ptr);
						VerifyUnswizzledString(layout, col_idx, row_ptr + i * row_width);
					}
					col_ptr += row_width;
					string_ptr += row_width;
				}
			} else {
				// Non-varchar blob columns
				for (idx_t i = 0; i < next; i++) {
					// Overwrite the column data offset with the pointer
					Store<data_ptr_t>(heap_row_ptrs[i] + Load<idx_t>(col_ptr), col_ptr);
					col_ptr += row_width;
				}
			}
		}
		done += next;
	}
}

} // namespace duckdb
//===--------------------------------------------------------------------===//
// row_gather.cpp
// Description: This file contains the implementation of the gather operators
//===--------------------------------------------------------------------===//








namespace duckdb {

using ValidityBytes = RowLayout::ValidityBytes;

template <class T>
static void TemplatedGatherLoop(Vector &rows, const SelectionVector &row_sel, Vector &col,
                                const SelectionVector &col_sel, idx_t count, const RowLayout &layout, idx_t col_no,
                                idx_t build_size) {
	// Precompute mask indexes
	const auto &offsets = layout.GetOffsets();
	const auto col_offset = offsets[col_no];
	idx_t entry_idx;
	idx_t idx_in_entry;
	ValidityBytes::GetEntryIndex(col_no, entry_idx, idx_in_entry);

	auto ptrs = FlatVector::GetData<data_ptr_t>(rows);
	auto data = FlatVector::GetData<T>(col);
	auto &col_mask = FlatVector::Validity(col);

	for (idx_t i = 0; i < count; i++) {
		auto row_idx = row_sel.get_index(i);
		auto row = ptrs[row_idx];
		auto col_idx = col_sel.get_index(i);
		data[col_idx] = Load<T>(row + col_offset);
		ValidityBytes row_mask(row);
		if (!row_mask.RowIsValid(row_mask.GetValidityEntry(entry_idx), idx_in_entry)) {
			if (build_size > STANDARD_VECTOR_SIZE && col_mask.AllValid()) {
				//! We need to initialize the mask with the vector size.
				col_mask.Initialize(build_size);
			}
			col_mask.SetInvalid(col_idx);
		}
	}
}

static void GatherVarchar(Vector &rows, const SelectionVector &row_sel, Vector &col, const SelectionVector &col_sel,
                          idx_t count, const RowLayout &layout, idx_t col_no, idx_t build_size,
                          data_ptr_t base_heap_ptr) {
	// Precompute mask indexes
	const auto &offsets = layout.GetOffsets();
	const auto col_offset = offsets[col_no];
	const auto heap_offset = layout.GetHeapOffset();
	idx_t entry_idx;
	idx_t idx_in_entry;
	ValidityBytes::GetEntryIndex(col_no, entry_idx, idx_in_entry);

	auto ptrs = FlatVector::GetData<data_ptr_t>(rows);
	auto data = FlatVector::GetData<string_t>(col);
	auto &col_mask = FlatVector::Validity(col);

	for (idx_t i = 0; i < count; i++) {
		auto row_idx = row_sel.get_index(i);
		auto row = ptrs[row_idx];
		auto col_idx = col_sel.get_index(i);
		auto col_ptr = row + col_offset;
		data[col_idx] = Load<string_t>(col_ptr);
		ValidityBytes row_mask(row);
		if (!row_mask.RowIsValid(row_mask.GetValidityEntry(entry_idx), idx_in_entry)) {
			if (build_size > STANDARD_VECTOR_SIZE && col_mask.AllValid()) {
				//! We need to initialize the mask with the vector size.
				col_mask.Initialize(build_size);
			}
			col_mask.SetInvalid(col_idx);
		} else if (base_heap_ptr && Load<uint32_t>(col_ptr) > string_t::INLINE_LENGTH) {
			//	Not inline, so unswizzle the copied pointer the pointer
			auto heap_ptr_ptr = row + heap_offset;
			auto heap_row_ptr = base_heap_ptr + Load<idx_t>(heap_ptr_ptr);
			auto string_ptr = data_ptr_t(data + col_idx) + string_t::HEADER_SIZE;
			Store<data_ptr_t>(heap_row_ptr + Load<idx_t>(string_ptr), string_ptr);
#ifdef DEBUG
			data[col_idx].Verify();
#endif
		}
	}
}

static void GatherNestedVector(Vector &rows, const SelectionVector &row_sel, Vector &col,
                               const SelectionVector &col_sel, idx_t count, const RowLayout &layout, idx_t col_no,
                               data_ptr_t base_heap_ptr) {
	const auto &offsets = layout.GetOffsets();
	const auto col_offset = offsets[col_no];
	const auto heap_offset = layout.GetHeapOffset();
	auto ptrs = FlatVector::GetData<data_ptr_t>(rows);

	// Build the gather locations
	auto data_locations = make_unsafe_uniq_array<data_ptr_t>(count);
	auto mask_locations = make_unsafe_uniq_array<data_ptr_t>(count);
	for (idx_t i = 0; i < count; i++) {
		auto row_idx = row_sel.get_index(i);
		auto row = ptrs[row_idx];
		mask_locations[i] = row;
		auto col_ptr = ptrs[row_idx] + col_offset;
		if (base_heap_ptr) {
			auto heap_ptr_ptr = row + heap_offset;
			auto heap_row_ptr = base_heap_ptr + Load<idx_t>(heap_ptr_ptr);
			data_locations[i] = heap_row_ptr + Load<idx_t>(col_ptr);
		} else {
			data_locations[i] = Load<data_ptr_t>(col_ptr);
		}
	}

	// Deserialise into the selected locations
	RowOperations::HeapGather(col, count, col_sel, col_no, data_locations.get(), mask_locations.get());
}

void RowOperations::Gather(Vector &rows, const SelectionVector &row_sel, Vector &col, const SelectionVector &col_sel,
                           const idx_t count, const RowLayout &layout, const idx_t col_no, const idx_t build_size,
                           data_ptr_t heap_ptr) {
	D_ASSERT(rows.GetVectorType() == VectorType::FLAT_VECTOR);
	D_ASSERT(rows.GetType().id() == LogicalTypeId::POINTER); // "Cannot gather from non-pointer type!"

	col.SetVectorType(VectorType::FLAT_VECTOR);
	switch (col.GetType().InternalType()) {
	case PhysicalType::UINT8:
		TemplatedGatherLoop<uint8_t>(rows, row_sel, col, col_sel, count, layout, col_no, build_size);
		break;
	case PhysicalType::UINT16:
		TemplatedGatherLoop<uint16_t>(rows, row_sel, col, col_sel, count, layout, col_no, build_size);
		break;
	case PhysicalType::UINT32:
		TemplatedGatherLoop<uint32_t>(rows, row_sel, col, col_sel, count, layout, col_no, build_size);
		break;
	case PhysicalType::UINT64:
		TemplatedGatherLoop<uint64_t>(rows, row_sel, col, col_sel, count, layout, col_no, build_size);
		break;
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
		TemplatedGatherLoop<int8_t>(rows, row_sel, col, col_sel, count, layout, col_no, build_size);
		break;
	case PhysicalType::INT16:
		TemplatedGatherLoop<int16_t>(rows, row_sel, col, col_sel, count, layout, col_no, build_size);
		break;
	case PhysicalType::INT32:
		TemplatedGatherLoop<int32_t>(rows, row_sel, col, col_sel, count, layout, col_no, build_size);
		break;
	case PhysicalType::INT64:
		TemplatedGatherLoop<int64_t>(rows, row_sel, col, col_sel, count, layout, col_no, build_size);
		break;
	case PhysicalType::INT128:
		TemplatedGatherLoop<hugeint_t>(rows, row_sel, col, col_sel, count, layout, col_no, build_size);
		break;
	case PhysicalType::FLOAT:
		TemplatedGatherLoop<float>(rows, row_sel, col, col_sel, count, layout, col_no, build_size);
		break;
	case PhysicalType::DOUBLE:
		TemplatedGatherLoop<double>(rows, row_sel, col, col_sel, count, layout, col_no, build_size);
		break;
	case PhysicalType::INTERVAL:
		TemplatedGatherLoop<interval_t>(rows, row_sel, col, col_sel, count, layout, col_no, build_size);
		break;
	case PhysicalType::VARCHAR:
		GatherVarchar(rows, row_sel, col, col_sel, count, layout, col_no, build_size, heap_ptr);
		break;
	case PhysicalType::LIST:
	case PhysicalType::STRUCT:
		GatherNestedVector(rows, row_sel, col, col_sel, count, layout, col_no, heap_ptr);
		break;
	default:
		throw InternalException("Unimplemented type for RowOperations::Gather");
	}
}

template <class T>
static void TemplatedFullScanLoop(Vector &rows, Vector &col, idx_t count, idx_t col_offset, idx_t col_no) {
	// Precompute mask indexes
	idx_t entry_idx;
	idx_t idx_in_entry;
	ValidityBytes::GetEntryIndex(col_no, entry_idx, idx_in_entry);

	auto ptrs = FlatVector::GetData<data_ptr_t>(rows);
	auto data = FlatVector::GetData<T>(col);
	//	auto &col_mask = FlatVector::Validity(col);

	for (idx_t i = 0; i < count; i++) {
		auto row = ptrs[i];
		data[i] = Load<T>(row + col_offset);
		ValidityBytes row_mask(row);
		if (!row_mask.RowIsValid(row_mask.GetValidityEntry(entry_idx), idx_in_entry)) {
			throw InternalException("Null value comparisons not implemented for perfect hash table yet");
			//			col_mask.SetInvalid(i);
		}
	}
}

void RowOperations::FullScanColumn(const TupleDataLayout &layout, Vector &rows, Vector &col, idx_t count,
                                   idx_t col_no) {
	const auto col_offset = layout.GetOffsets()[col_no];
	col.SetVectorType(VectorType::FLAT_VECTOR);
	switch (col.GetType().InternalType()) {
	case PhysicalType::UINT8:
		TemplatedFullScanLoop<uint8_t>(rows, col, count, col_offset, col_no);
		break;
	case PhysicalType::UINT16:
		TemplatedFullScanLoop<uint16_t>(rows, col, count, col_offset, col_no);
		break;
	case PhysicalType::UINT32:
		TemplatedFullScanLoop<uint32_t>(rows, col, count, col_offset, col_no);
		break;
	case PhysicalType::UINT64:
		TemplatedFullScanLoop<uint64_t>(rows, col, count, col_offset, col_no);
		break;
	case PhysicalType::INT8:
		TemplatedFullScanLoop<int8_t>(rows, col, count, col_offset, col_no);
		break;
	case PhysicalType::INT16:
		TemplatedFullScanLoop<int16_t>(rows, col, count, col_offset, col_no);
		break;
	case PhysicalType::INT32:
		TemplatedFullScanLoop<int32_t>(rows, col, count, col_offset, col_no);
		break;
	case PhysicalType::INT64:
		TemplatedFullScanLoop<int64_t>(rows, col, count, col_offset, col_no);
		break;
	default:
		throw NotImplementedException("Unimplemented type for RowOperations::FullScanColumn");
	}
}

} // namespace duckdb




namespace duckdb {

using ValidityBytes = TemplatedValidityMask<uint8_t>;

template <class T>
static void TemplatedHeapGather(Vector &v, const idx_t count, const SelectionVector &sel, data_ptr_t *key_locations) {
	auto target = FlatVector::GetData<T>(v);

	for (idx_t i = 0; i < count; ++i) {
		const auto col_idx = sel.get_index(i);
		target[col_idx] = Load<T>(key_locations[i]);
		key_locations[i] += sizeof(T);
	}
}

static void HeapGatherStringVector(Vector &v, const idx_t vcount, const SelectionVector &sel,
                                   data_ptr_t *key_locations) {
	const auto &validity = FlatVector::Validity(v);
	auto target = FlatVector::GetData<string_t>(v);

	for (idx_t i = 0; i < vcount; i++) {
		const auto col_idx = sel.get_index(i);
		if (!validity.RowIsValid(col_idx)) {
			continue;
		}
		auto len = Load<uint32_t>(key_locations[i]);
		key_locations[i] += sizeof(uint32_t);
		target[col_idx] = StringVector::AddStringOrBlob(v, string_t((const char *)key_locations[i], len));
		key_locations[i] += len;
	}
}

static void HeapGatherStructVector(Vector &v, const idx_t vcount, const SelectionVector &sel,
                                   data_ptr_t *key_locations) {
	// struct must have a validitymask for its fields
	auto &child_types = StructType::GetChildTypes(v.GetType());
	const idx_t struct_validitymask_size = (child_types.size() + 7) / 8;
	data_ptr_t struct_validitymask_locations[STANDARD_VECTOR_SIZE];
	for (idx_t i = 0; i < vcount; i++) {
		// use key_locations as the validitymask, and create struct_key_locations
		struct_validitymask_locations[i] = key_locations[i];
		key_locations[i] += struct_validitymask_size;
	}

	// now deserialize into the struct vectors
	auto &children = StructVector::GetEntries(v);
	for (idx_t i = 0; i < child_types.size(); i++) {
		RowOperations::HeapGather(*children[i], vcount, sel, i, key_locations, struct_validitymask_locations);
	}
}

static void HeapGatherListVector(Vector &v, const idx_t vcount, const SelectionVector &sel, data_ptr_t *key_locations) {
	const auto &validity = FlatVector::Validity(v);

	auto child_type = ListType::GetChildType(v.GetType());
	auto list_data = ListVector::GetData(v);
	data_ptr_t list_entry_locations[STANDARD_VECTOR_SIZE];

	uint64_t entry_offset = ListVector::GetListSize(v);
	for (idx_t i = 0; i < vcount; i++) {
		const auto col_idx = sel.get_index(i);
		if (!validity.RowIsValid(col_idx)) {
			continue;
		}
		// read list length
		auto entry_remaining = Load<uint64_t>(key_locations[i]);
		key_locations[i] += sizeof(uint64_t);
		// set list entry attributes
		list_data[col_idx].length = entry_remaining;
		list_data[col_idx].offset = entry_offset;
		// skip over the validity mask
		data_ptr_t validitymask_location = key_locations[i];
		idx_t offset_in_byte = 0;
		key_locations[i] += (entry_remaining + 7) / 8;
		// entry sizes
		data_ptr_t var_entry_size_ptr = nullptr;
		if (!TypeIsConstantSize(child_type.InternalType())) {
			var_entry_size_ptr = key_locations[i];
			key_locations[i] += entry_remaining * sizeof(idx_t);
		}

		// now read the list data
		while (entry_remaining > 0) {
			auto next = MinValue(entry_remaining, (idx_t)STANDARD_VECTOR_SIZE);

			// initialize a new vector to append
			Vector append_vector(v.GetType());
			append_vector.SetVectorType(v.GetVectorType());

			auto &list_vec_to_append = ListVector::GetEntry(append_vector);

			// set validity
			//! Since we are constructing the vector, this will always be a flat vector.
			auto &append_validity = FlatVector::Validity(list_vec_to_append);
			for (idx_t entry_idx = 0; entry_idx < next; entry_idx++) {
				append_validity.Set(entry_idx, *(validitymask_location) & (1 << offset_in_byte));
				if (++offset_in_byte == 8) {
					validitymask_location++;
					offset_in_byte = 0;
				}
			}

			// compute entry sizes and set locations where the list entries are
			if (TypeIsConstantSize(child_type.InternalType())) {
				// constant size list entries
				const idx_t type_size = GetTypeIdSize(child_type.InternalType());
				for (idx_t entry_idx = 0; entry_idx < next; entry_idx++) {
					list_entry_locations[entry_idx] = key_locations[i];
					key_locations[i] += type_size;
				}
			} else {
				// variable size list entries
				for (idx_t entry_idx = 0; entry_idx < next; entry_idx++) {
					list_entry_locations[entry_idx] = key_locations[i];
					key_locations[i] += Load<idx_t>(var_entry_size_ptr);
					var_entry_size_ptr += sizeof(idx_t);
				}
			}

			// now deserialize and add to listvector
			RowOperations::HeapGather(list_vec_to_append, next, *FlatVector::IncrementalSelectionVector(), 0,
			                          list_entry_locations, nullptr);
			ListVector::Append(v, list_vec_to_append, next);

			// update for next iteration
			entry_remaining -= next;
			entry_offset += next;
		}
	}
}

void RowOperations::HeapGather(Vector &v, const idx_t &vcount, const SelectionVector &sel, const idx_t &col_no,
                               data_ptr_t *key_locations, data_ptr_t *validitymask_locations) {
	v.SetVectorType(VectorType::FLAT_VECTOR);

	auto &validity = FlatVector::Validity(v);
	if (validitymask_locations) {
		// Precompute mask indexes
		idx_t entry_idx;
		idx_t idx_in_entry;
		ValidityBytes::GetEntryIndex(col_no, entry_idx, idx_in_entry);

		for (idx_t i = 0; i < vcount; i++) {
			ValidityBytes row_mask(validitymask_locations[i]);
			const auto valid = row_mask.RowIsValid(row_mask.GetValidityEntry(entry_idx), idx_in_entry);
			const auto col_idx = sel.get_index(i);
			validity.Set(col_idx, valid);
		}
	}

	auto type = v.GetType().InternalType();
	switch (type) {
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
		TemplatedHeapGather<int8_t>(v, vcount, sel, key_locations);
		break;
	case PhysicalType::INT16:
		TemplatedHeapGather<int16_t>(v, vcount, sel, key_locations);
		break;
	case PhysicalType::INT32:
		TemplatedHeapGather<int32_t>(v, vcount, sel, key_locations);
		break;
	case PhysicalType::INT64:
		TemplatedHeapGather<int64_t>(v, vcount, sel, key_locations);
		break;
	case PhysicalType::UINT8:
		TemplatedHeapGather<uint8_t>(v, vcount, sel, key_locations);
		break;
	case PhysicalType::UINT16:
		TemplatedHeapGather<uint16_t>(v, vcount, sel, key_locations);
		break;
	case PhysicalType::UINT32:
		TemplatedHeapGather<uint32_t>(v, vcount, sel, key_locations);
		break;
	case PhysicalType::UINT64:
		TemplatedHeapGather<uint64_t>(v, vcount, sel, key_locations);
		break;
	case PhysicalType::INT128:
		TemplatedHeapGather<hugeint_t>(v, vcount, sel, key_locations);
		break;
	case PhysicalType::FLOAT:
		TemplatedHeapGather<float>(v, vcount, sel, key_locations);
		break;
	case PhysicalType::DOUBLE:
		TemplatedHeapGather<double>(v, vcount, sel, key_locations);
		break;
	case PhysicalType::INTERVAL:
		TemplatedHeapGather<interval_t>(v, vcount, sel, key_locations);
		break;
	case PhysicalType::VARCHAR:
		HeapGatherStringVector(v, vcount, sel, key_locations);
		break;
	case PhysicalType::STRUCT:
		HeapGatherStructVector(v, vcount, sel, key_locations);
		break;
	case PhysicalType::LIST:
		HeapGatherListVector(v, vcount, sel, key_locations);
		break;
	default:
		throw NotImplementedException("Unimplemented deserialize from row-format");
	}
}

} // namespace duckdb




namespace duckdb {

using ValidityBytes = TemplatedValidityMask<uint8_t>;

static void ComputeStringEntrySizes(UnifiedVectorFormat &vdata, idx_t entry_sizes[], const idx_t ser_count,
                                    const SelectionVector &sel, const idx_t offset) {
	auto strings = (string_t *)vdata.data;
	for (idx_t i = 0; i < ser_count; i++) {
		auto idx = sel.get_index(i);
		auto str_idx = vdata.sel->get_index(idx + offset);
		if (vdata.validity.RowIsValid(str_idx)) {
			entry_sizes[i] += sizeof(uint32_t) + strings[str_idx].GetSize();
		}
	}
}

static void ComputeStructEntrySizes(Vector &v, idx_t entry_sizes[], idx_t vcount, idx_t ser_count,
                                    const SelectionVector &sel, idx_t offset) {
	// obtain child vectors
	idx_t num_children;
	auto &children = StructVector::GetEntries(v);
	num_children = children.size();
	// add struct validitymask size
	const idx_t struct_validitymask_size = (num_children + 7) / 8;
	for (idx_t i = 0; i < ser_count; i++) {
		entry_sizes[i] += struct_validitymask_size;
	}
	// compute size of child vectors
	for (auto &struct_vector : children) {
		RowOperations::ComputeEntrySizes(*struct_vector, entry_sizes, vcount, ser_count, sel, offset);
	}
}

static void ComputeListEntrySizes(Vector &v, UnifiedVectorFormat &vdata, idx_t entry_sizes[], idx_t ser_count,
                                  const SelectionVector &sel, idx_t offset) {
	auto list_data = ListVector::GetData(v);
	auto &child_vector = ListVector::GetEntry(v);
	idx_t list_entry_sizes[STANDARD_VECTOR_SIZE];
	for (idx_t i = 0; i < ser_count; i++) {
		auto idx = sel.get_index(i);
		auto source_idx = vdata.sel->get_index(idx + offset);
		if (vdata.validity.RowIsValid(source_idx)) {
			auto list_entry = list_data[source_idx];

			// make room for list length, list validitymask
			entry_sizes[i] += sizeof(list_entry.length);
			entry_sizes[i] += (list_entry.length + 7) / 8;

			// serialize size of each entry (if non-constant size)
			if (!TypeIsConstantSize(ListType::GetChildType(v.GetType()).InternalType())) {
				entry_sizes[i] += list_entry.length * sizeof(list_entry.length);
			}

			// compute size of each the elements in list_entry and sum them
			auto entry_remaining = list_entry.length;
			auto entry_offset = list_entry.offset;
			while (entry_remaining > 0) {
				// the list entry can span multiple vectors
				auto next = MinValue((idx_t)STANDARD_VECTOR_SIZE, entry_remaining);

				// compute and add to the total
				std::fill_n(list_entry_sizes, next, 0);
				RowOperations::ComputeEntrySizes(child_vector, list_entry_sizes, next, next,
				                                 *FlatVector::IncrementalSelectionVector(), entry_offset);
				for (idx_t list_idx = 0; list_idx < next; list_idx++) {
					entry_sizes[i] += list_entry_sizes[list_idx];
				}

				// update for next iteration
				entry_remaining -= next;
				entry_offset += next;
			}
		}
	}
}

void RowOperations::ComputeEntrySizes(Vector &v, UnifiedVectorFormat &vdata, idx_t entry_sizes[], idx_t vcount,
                                      idx_t ser_count, const SelectionVector &sel, idx_t offset) {
	const auto physical_type = v.GetType().InternalType();
	if (TypeIsConstantSize(physical_type)) {
		const auto type_size = GetTypeIdSize(physical_type);
		for (idx_t i = 0; i < ser_count; i++) {
			entry_sizes[i] += type_size;
		}
	} else {
		switch (physical_type) {
		case PhysicalType::VARCHAR:
			ComputeStringEntrySizes(vdata, entry_sizes, ser_count, sel, offset);
			break;
		case PhysicalType::STRUCT:
			ComputeStructEntrySizes(v, entry_sizes, vcount, ser_count, sel, offset);
			break;
		case PhysicalType::LIST:
			ComputeListEntrySizes(v, vdata, entry_sizes, ser_count, sel, offset);
			break;
		default:
			// LCOV_EXCL_START
			throw NotImplementedException("Column with variable size type %s cannot be serialized to row-format",
			                              v.GetType().ToString());
			// LCOV_EXCL_STOP
		}
	}
}

void RowOperations::ComputeEntrySizes(Vector &v, idx_t entry_sizes[], idx_t vcount, idx_t ser_count,
                                      const SelectionVector &sel, idx_t offset) {
	UnifiedVectorFormat vdata;
	v.ToUnifiedFormat(vcount, vdata);
	ComputeEntrySizes(v, vdata, entry_sizes, vcount, ser_count, sel, offset);
}

template <class T>
static void TemplatedHeapScatter(UnifiedVectorFormat &vdata, const SelectionVector &sel, idx_t count, idx_t col_idx,
                                 data_ptr_t *key_locations, data_ptr_t *validitymask_locations, idx_t offset) {
	auto source = (T *)vdata.data;
	if (!validitymask_locations) {
		for (idx_t i = 0; i < count; i++) {
			auto idx = sel.get_index(i);
			auto source_idx = vdata.sel->get_index(idx + offset);

			auto target = (T *)key_locations[i];
			Store<T>(source[source_idx], (data_ptr_t)target);
			key_locations[i] += sizeof(T);
		}
	} else {
		idx_t entry_idx;
		idx_t idx_in_entry;
		ValidityBytes::GetEntryIndex(col_idx, entry_idx, idx_in_entry);
		const auto bit = ~(1UL << idx_in_entry);
		for (idx_t i = 0; i < count; i++) {
			auto idx = sel.get_index(i);
			auto source_idx = vdata.sel->get_index(idx + offset);

			auto target = (T *)key_locations[i];
			Store<T>(source[source_idx], (data_ptr_t)target);
			key_locations[i] += sizeof(T);

			// set the validitymask
			if (!vdata.validity.RowIsValid(source_idx)) {
				*(validitymask_locations[i] + entry_idx) &= bit;
			}
		}
	}
}

static void HeapScatterStringVector(Vector &v, idx_t vcount, const SelectionVector &sel, idx_t ser_count, idx_t col_idx,
                                    data_ptr_t *key_locations, data_ptr_t *validitymask_locations, idx_t offset) {
	UnifiedVectorFormat vdata;
	v.ToUnifiedFormat(vcount, vdata);

	auto strings = (string_t *)vdata.data;
	if (!validitymask_locations) {
		for (idx_t i = 0; i < ser_count; i++) {
			auto idx = sel.get_index(i);
			auto source_idx = vdata.sel->get_index(idx + offset);
			if (vdata.validity.RowIsValid(source_idx)) {
				auto &string_entry = strings[source_idx];
				// store string size
				Store<uint32_t>(string_entry.GetSize(), key_locations[i]);
				key_locations[i] += sizeof(uint32_t);
				// store the string
				memcpy(key_locations[i], string_entry.GetData(), string_entry.GetSize());
				key_locations[i] += string_entry.GetSize();
			}
		}
	} else {
		idx_t entry_idx;
		idx_t idx_in_entry;
		ValidityBytes::GetEntryIndex(col_idx, entry_idx, idx_in_entry);
		const auto bit = ~(1UL << idx_in_entry);
		for (idx_t i = 0; i < ser_count; i++) {
			auto idx = sel.get_index(i);
			auto source_idx = vdata.sel->get_index(idx + offset);
			if (vdata.validity.RowIsValid(source_idx)) {
				auto &string_entry = strings[source_idx];
				// store string size
				Store<uint32_t>(string_entry.GetSize(), key_locations[i]);
				key_locations[i] += sizeof(uint32_t);
				// store the string
				memcpy(key_locations[i], string_entry.GetData(), string_entry.GetSize());
				key_locations[i] += string_entry.GetSize();
			} else {
				// set the validitymask
				*(validitymask_locations[i] + entry_idx) &= bit;
			}
		}
	}
}

static void HeapScatterStructVector(Vector &v, idx_t vcount, const SelectionVector &sel, idx_t ser_count, idx_t col_idx,
                                    data_ptr_t *key_locations, data_ptr_t *validitymask_locations, idx_t offset) {
	UnifiedVectorFormat vdata;
	v.ToUnifiedFormat(vcount, vdata);

	auto &children = StructVector::GetEntries(v);
	idx_t num_children = children.size();

	// the whole struct itself can be NULL
	idx_t entry_idx;
	idx_t idx_in_entry;
	ValidityBytes::GetEntryIndex(col_idx, entry_idx, idx_in_entry);
	const auto bit = ~(1UL << idx_in_entry);

	// struct must have a validitymask for its fields
	const idx_t struct_validitymask_size = (num_children + 7) / 8;
	data_ptr_t struct_validitymask_locations[STANDARD_VECTOR_SIZE];
	for (idx_t i = 0; i < ser_count; i++) {
		// initialize the struct validity mask
		struct_validitymask_locations[i] = key_locations[i];
		memset(struct_validitymask_locations[i], -1, struct_validitymask_size);
		key_locations[i] += struct_validitymask_size;

		// set whether the whole struct is null
		auto idx = sel.get_index(i);
		auto source_idx = vdata.sel->get_index(idx) + offset;
		if (validitymask_locations && !vdata.validity.RowIsValid(source_idx)) {
			*(validitymask_locations[i] + entry_idx) &= bit;
		}
	}

	// now serialize the struct vectors
	for (idx_t i = 0; i < children.size(); i++) {
		auto &struct_vector = *children[i];
		RowOperations::HeapScatter(struct_vector, vcount, sel, ser_count, i, key_locations,
		                           struct_validitymask_locations, offset);
	}
}

static void HeapScatterListVector(Vector &v, idx_t vcount, const SelectionVector &sel, idx_t ser_count, idx_t col_no,
                                  data_ptr_t *key_locations, data_ptr_t *validitymask_locations, idx_t offset) {
	UnifiedVectorFormat vdata;
	v.ToUnifiedFormat(vcount, vdata);

	idx_t entry_idx;
	idx_t idx_in_entry;
	ValidityBytes::GetEntryIndex(col_no, entry_idx, idx_in_entry);

	auto list_data = ListVector::GetData(v);

	auto &child_vector = ListVector::GetEntry(v);

	UnifiedVectorFormat list_vdata;
	child_vector.ToUnifiedFormat(ListVector::GetListSize(v), list_vdata);
	auto child_type = ListType::GetChildType(v.GetType()).InternalType();

	idx_t list_entry_sizes[STANDARD_VECTOR_SIZE];
	data_ptr_t list_entry_locations[STANDARD_VECTOR_SIZE];

	for (idx_t i = 0; i < ser_count; i++) {
		auto idx = sel.get_index(i);
		auto source_idx = vdata.sel->get_index(idx + offset);
		if (!vdata.validity.RowIsValid(source_idx)) {
			if (validitymask_locations) {
				// set the row validitymask for this column to invalid
				ValidityBytes row_mask(validitymask_locations[i]);
				row_mask.SetInvalidUnsafe(entry_idx, idx_in_entry);
			}
			continue;
		}
		auto list_entry = list_data[source_idx];

		// store list length
		Store<uint64_t>(list_entry.length, key_locations[i]);
		key_locations[i] += sizeof(list_entry.length);

		// make room for the validitymask
		data_ptr_t list_validitymask_location = key_locations[i];
		idx_t entry_offset_in_byte = 0;
		idx_t validitymask_size = (list_entry.length + 7) / 8;
		memset(list_validitymask_location, -1, validitymask_size);
		key_locations[i] += validitymask_size;

		// serialize size of each entry (if non-constant size)
		data_ptr_t var_entry_size_ptr = nullptr;
		if (!TypeIsConstantSize(child_type)) {
			var_entry_size_ptr = key_locations[i];
			key_locations[i] += list_entry.length * sizeof(idx_t);
		}

		auto entry_remaining = list_entry.length;
		auto entry_offset = list_entry.offset;
		while (entry_remaining > 0) {
			// the list entry can span multiple vectors
			auto next = MinValue((idx_t)STANDARD_VECTOR_SIZE, entry_remaining);

			// serialize list validity
			for (idx_t entry_idx = 0; entry_idx < next; entry_idx++) {
				auto list_idx = list_vdata.sel->get_index(entry_idx + entry_offset);
				if (!list_vdata.validity.RowIsValid(list_idx)) {
					*(list_validitymask_location) &= ~(1UL << entry_offset_in_byte);
				}
				if (++entry_offset_in_byte == 8) {
					list_validitymask_location++;
					entry_offset_in_byte = 0;
				}
			}

			if (TypeIsConstantSize(child_type)) {
				// constant size list entries: set list entry locations
				const idx_t type_size = GetTypeIdSize(child_type);
				for (idx_t entry_idx = 0; entry_idx < next; entry_idx++) {
					list_entry_locations[entry_idx] = key_locations[i];
					key_locations[i] += type_size;
				}
			} else {
				// variable size list entries: compute entry sizes and set list entry locations
				std::fill_n(list_entry_sizes, next, 0);
				RowOperations::ComputeEntrySizes(child_vector, list_entry_sizes, next, next,
				                                 *FlatVector::IncrementalSelectionVector(), entry_offset);
				for (idx_t entry_idx = 0; entry_idx < next; entry_idx++) {
					list_entry_locations[entry_idx] = key_locations[i];
					key_locations[i] += list_entry_sizes[entry_idx];
					Store<idx_t>(list_entry_sizes[entry_idx], var_entry_size_ptr);
					var_entry_size_ptr += sizeof(idx_t);
				}
			}

			// now serialize to the locations
			RowOperations::HeapScatter(child_vector, ListVector::GetListSize(v),
			                           *FlatVector::IncrementalSelectionVector(), next, 0, list_entry_locations,
			                           nullptr, entry_offset);

			// update for next iteration
			entry_remaining -= next;
			entry_offset += next;
		}
	}
}

void RowOperations::HeapScatter(Vector &v, idx_t vcount, const SelectionVector &sel, idx_t ser_count, idx_t col_idx,
                                data_ptr_t *key_locations, data_ptr_t *validitymask_locations, idx_t offset) {
	if (TypeIsConstantSize(v.GetType().InternalType())) {
		UnifiedVectorFormat vdata;
		v.ToUnifiedFormat(vcount, vdata);
		RowOperations::HeapScatterVData(vdata, v.GetType().InternalType(), sel, ser_count, col_idx, key_locations,
		                                validitymask_locations, offset);
	} else {
		switch (v.GetType().InternalType()) {
		case PhysicalType::VARCHAR:
			HeapScatterStringVector(v, vcount, sel, ser_count, col_idx, key_locations, validitymask_locations, offset);
			break;
		case PhysicalType::STRUCT:
			HeapScatterStructVector(v, vcount, sel, ser_count, col_idx, key_locations, validitymask_locations, offset);
			break;
		case PhysicalType::LIST:
			HeapScatterListVector(v, vcount, sel, ser_count, col_idx, key_locations, validitymask_locations, offset);
			break;
		default:
			// LCOV_EXCL_START
			throw NotImplementedException("Serialization of variable length vector with type %s",
			                              v.GetType().ToString());
			// LCOV_EXCL_STOP
		}
	}
}

void RowOperations::HeapScatterVData(UnifiedVectorFormat &vdata, PhysicalType type, const SelectionVector &sel,
                                     idx_t ser_count, idx_t col_idx, data_ptr_t *key_locations,
                                     data_ptr_t *validitymask_locations, idx_t offset) {
	switch (type) {
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
		TemplatedHeapScatter<int8_t>(vdata, sel, ser_count, col_idx, key_locations, validitymask_locations, offset);
		break;
	case PhysicalType::INT16:
		TemplatedHeapScatter<int16_t>(vdata, sel, ser_count, col_idx, key_locations, validitymask_locations, offset);
		break;
	case PhysicalType::INT32:
		TemplatedHeapScatter<int32_t>(vdata, sel, ser_count, col_idx, key_locations, validitymask_locations, offset);
		break;
	case PhysicalType::INT64:
		TemplatedHeapScatter<int64_t>(vdata, sel, ser_count, col_idx, key_locations, validitymask_locations, offset);
		break;
	case PhysicalType::UINT8:
		TemplatedHeapScatter<uint8_t>(vdata, sel, ser_count, col_idx, key_locations, validitymask_locations, offset);
		break;
	case PhysicalType::UINT16:
		TemplatedHeapScatter<uint16_t>(vdata, sel, ser_count, col_idx, key_locations, validitymask_locations, offset);
		break;
	case PhysicalType::UINT32:
		TemplatedHeapScatter<uint32_t>(vdata, sel, ser_count, col_idx, key_locations, validitymask_locations, offset);
		break;
	case PhysicalType::UINT64:
		TemplatedHeapScatter<uint64_t>(vdata, sel, ser_count, col_idx, key_locations, validitymask_locations, offset);
		break;
	case PhysicalType::INT128:
		TemplatedHeapScatter<hugeint_t>(vdata, sel, ser_count, col_idx, key_locations, validitymask_locations, offset);
		break;
	case PhysicalType::FLOAT:
		TemplatedHeapScatter<float>(vdata, sel, ser_count, col_idx, key_locations, validitymask_locations, offset);
		break;
	case PhysicalType::DOUBLE:
		TemplatedHeapScatter<double>(vdata, sel, ser_count, col_idx, key_locations, validitymask_locations, offset);
		break;
	case PhysicalType::INTERVAL:
		TemplatedHeapScatter<interval_t>(vdata, sel, ser_count, col_idx, key_locations, validitymask_locations, offset);
		break;
	default:
		throw NotImplementedException("FIXME: Serialize to of constant type column to row-format");
	}
}

} // namespace duckdb
//===--------------------------------------------------------------------===//
// row_match.cpp
// Description: This file contains the implementation of the match operators
//===--------------------------------------------------------------------===//







namespace duckdb {

using ValidityBytes = RowLayout::ValidityBytes;
using Predicates = RowOperations::Predicates;

template <typename OP>
static idx_t SelectComparison(Vector &left, Vector &right, const SelectionVector &sel, idx_t count,
                              SelectionVector *true_sel, SelectionVector *false_sel) {
	throw NotImplementedException("Unsupported nested comparison operand for RowOperations::Match");
}

template <>
idx_t SelectComparison<Equals>(Vector &left, Vector &right, const SelectionVector &sel, idx_t count,
                               SelectionVector *true_sel, SelectionVector *false_sel) {
	return VectorOperations::NestedEquals(left, right, sel, count, true_sel, false_sel);
}

template <>
idx_t SelectComparison<NotEquals>(Vector &left, Vector &right, const SelectionVector &sel, idx_t count,
                                  SelectionVector *true_sel, SelectionVector *false_sel) {
	return VectorOperations::NestedNotEquals(left, right, sel, count, true_sel, false_sel);
}

template <>
idx_t SelectComparison<GreaterThan>(Vector &left, Vector &right, const SelectionVector &sel, idx_t count,
                                    SelectionVector *true_sel, SelectionVector *false_sel) {
	return VectorOperations::DistinctGreaterThan(left, right, &sel, count, true_sel, false_sel);
}

template <>
idx_t SelectComparison<GreaterThanEquals>(Vector &left, Vector &right, const SelectionVector &sel, idx_t count,
                                          SelectionVector *true_sel, SelectionVector *false_sel) {
	return VectorOperations::DistinctGreaterThanEquals(left, right, &sel, count, true_sel, false_sel);
}

template <>
idx_t SelectComparison<LessThan>(Vector &left, Vector &right, const SelectionVector &sel, idx_t count,
                                 SelectionVector *true_sel, SelectionVector *false_sel) {
	return VectorOperations::DistinctLessThan(left, right, &sel, count, true_sel, false_sel);
}

template <>
idx_t SelectComparison<LessThanEquals>(Vector &left, Vector &right, const SelectionVector &sel, idx_t count,
                                       SelectionVector *true_sel, SelectionVector *false_sel) {
	return VectorOperations::DistinctLessThanEquals(left, right, &sel, count, true_sel, false_sel);
}

template <class T, class OP, bool NO_MATCH_SEL>
static void TemplatedMatchType(UnifiedVectorFormat &col, Vector &rows, SelectionVector &sel, idx_t &count,
                               idx_t col_offset, idx_t col_no, SelectionVector *no_match, idx_t &no_match_count) {
	// Precompute row_mask indexes
	idx_t entry_idx;
	idx_t idx_in_entry;
	ValidityBytes::GetEntryIndex(col_no, entry_idx, idx_in_entry);

	auto data = (T *)col.data;
	auto ptrs = FlatVector::GetData<data_ptr_t>(rows);
	idx_t match_count = 0;
	if (!col.validity.AllValid()) {
		for (idx_t i = 0; i < count; i++) {
			auto idx = sel.get_index(i);

			auto row = ptrs[idx];
			ValidityBytes row_mask(row);
			auto isnull = !row_mask.RowIsValid(row_mask.GetValidityEntry(entry_idx), idx_in_entry);

			auto col_idx = col.sel->get_index(idx);
			if (!col.validity.RowIsValid(col_idx)) {
				if (isnull) {
					// match: move to next value to compare
					sel.set_index(match_count++, idx);
				} else {
					if (NO_MATCH_SEL) {
						no_match->set_index(no_match_count++, idx);
					}
				}
			} else {
				auto value = Load<T>(row + col_offset);
				if (!isnull && OP::template Operation<T>(data[col_idx], value)) {
					sel.set_index(match_count++, idx);
				} else {
					if (NO_MATCH_SEL) {
						no_match->set_index(no_match_count++, idx);
					}
				}
			}
		}
	} else {
		for (idx_t i = 0; i < count; i++) {
			auto idx = sel.get_index(i);

			auto row = ptrs[idx];
			ValidityBytes row_mask(row);
			auto isnull = !row_mask.RowIsValid(row_mask.GetValidityEntry(entry_idx), idx_in_entry);

			auto col_idx = col.sel->get_index(idx);
			auto value = Load<T>(row + col_offset);
			if (!isnull && OP::template Operation<T>(data[col_idx], value)) {
				sel.set_index(match_count++, idx);
			} else {
				if (NO_MATCH_SEL) {
					no_match->set_index(no_match_count++, idx);
				}
			}
		}
	}
	count = match_count;
}

//! Forward declaration for recursion
template <class OP, bool NO_MATCH_SEL>
static void TemplatedMatchOp(Vector &vec, UnifiedVectorFormat &col, const TupleDataLayout &layout, Vector &rows,
                             SelectionVector &sel, idx_t &count, idx_t col_no, SelectionVector *no_match,
                             idx_t &no_match_count, const idx_t original_count);

template <class OP, bool NO_MATCH_SEL>
static void TemplatedMatchStruct(Vector &vec, UnifiedVectorFormat &col, const TupleDataLayout &layout, Vector &rows,
                                 SelectionVector &sel, idx_t &count, const idx_t col_no, SelectionVector *no_match,
                                 idx_t &no_match_count, const idx_t original_count) {
	// Precompute row_mask indexes
	idx_t entry_idx;
	idx_t idx_in_entry;
	ValidityBytes::GetEntryIndex(col_no, entry_idx, idx_in_entry);

	// Work our way through the validity of the whole struct
	auto ptrs = FlatVector::GetData<data_ptr_t>(rows);
	idx_t match_count = 0;
	if (!col.validity.AllValid()) {
		for (idx_t i = 0; i < count; i++) {
			auto idx = sel.get_index(i);

			auto row = ptrs[idx];
			ValidityBytes row_mask(row);
			auto isnull = !row_mask.RowIsValid(row_mask.GetValidityEntry(entry_idx), idx_in_entry);

			auto col_idx = col.sel->get_index(idx);
			if (!col.validity.RowIsValid(col_idx)) {
				if (isnull) {
					// match: move to next value to compare
					sel.set_index(match_count++, idx);
				} else {
					if (NO_MATCH_SEL) {
						no_match->set_index(no_match_count++, idx);
					}
				}
			} else {
				if (!isnull) {
					sel.set_index(match_count++, idx);
				} else {
					if (NO_MATCH_SEL) {
						no_match->set_index(no_match_count++, idx);
					}
				}
			}
		}
	} else {
		for (idx_t i = 0; i < count; i++) {
			auto idx = sel.get_index(i);

			auto row = ptrs[idx];
			ValidityBytes row_mask(row);
			auto isnull = !row_mask.RowIsValid(row_mask.GetValidityEntry(entry_idx), idx_in_entry);

			if (!isnull) {
				sel.set_index(match_count++, idx);
			} else {
				if (NO_MATCH_SEL) {
					no_match->set_index(no_match_count++, idx);
				}
			}
		}
	}
	count = match_count;

	// Now we construct row pointers to the structs
	Vector struct_rows(LogicalTypeId::POINTER);
	auto struct_ptrs = FlatVector::GetData<data_ptr_t>(struct_rows);

	const auto col_offset = layout.GetOffsets()[col_no];
	for (idx_t i = 0; i < count; i++) {
		auto idx = sel.get_index(i);
		auto row = ptrs[idx];
		struct_ptrs[idx] = row + col_offset;
	}

	// Get the struct layout, child columns, then recurse
	const auto &struct_layout = layout.GetStructLayout(col_no);
	auto &struct_entries = StructVector::GetEntries(vec);
	D_ASSERT(struct_layout.ColumnCount() == struct_entries.size());
	for (idx_t struct_col_no = 0; struct_col_no < struct_layout.ColumnCount(); struct_col_no++) {
		auto &struct_vec = *struct_entries[struct_col_no];
		UnifiedVectorFormat struct_col;
		struct_vec.ToUnifiedFormat(original_count, struct_col);
		TemplatedMatchOp<OP, NO_MATCH_SEL>(struct_vec, struct_col, struct_layout, struct_rows, sel, count,
		                                   struct_col_no, no_match, no_match_count, original_count);
	}
}

template <class OP, bool NO_MATCH_SEL>
static void TemplatedMatchList(Vector &col, Vector &rows, SelectionVector &sel, idx_t &count,
                               const TupleDataLayout &layout, const idx_t col_no, SelectionVector *no_match,
                               idx_t &no_match_count) {
	// Gather a dense Vector containing the column values being matched
	Vector key(col.GetType());
	const auto gather_function = TupleDataCollection::GetGatherFunction(col.GetType());
	gather_function.function(layout, rows, col_no, sel, count, key, *FlatVector::IncrementalSelectionVector(), key,
	                         gather_function.child_functions);

	// Densify the input column
	Vector sliced(col, sel, count);

	if (NO_MATCH_SEL) {
		SelectionVector no_match_sel_offset(no_match->data() + no_match_count);
		auto match_count = SelectComparison<OP>(sliced, key, sel, count, &sel, &no_match_sel_offset);
		no_match_count += count - match_count;
		count = match_count;
	} else {
		count = SelectComparison<OP>(sliced, key, sel, count, &sel, nullptr);
	}
}

template <class OP, bool NO_MATCH_SEL>
static void TemplatedMatchOp(Vector &vec, UnifiedVectorFormat &col, const TupleDataLayout &layout, Vector &rows,
                             SelectionVector &sel, idx_t &count, idx_t col_no, SelectionVector *no_match,
                             idx_t &no_match_count, const idx_t original_count) {
	if (count == 0) {
		return;
	}
	auto col_offset = layout.GetOffsets()[col_no];
	switch (layout.GetTypes()[col_no].InternalType()) {
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
		TemplatedMatchType<int8_t, OP, NO_MATCH_SEL>(col, rows, sel, count, col_offset, col_no, no_match,
		                                             no_match_count);
		break;
	case PhysicalType::INT16:
		TemplatedMatchType<int16_t, OP, NO_MATCH_SEL>(col, rows, sel, count, col_offset, col_no, no_match,
		                                              no_match_count);
		break;
	case PhysicalType::INT32:
		TemplatedMatchType<int32_t, OP, NO_MATCH_SEL>(col, rows, sel, count, col_offset, col_no, no_match,
		                                              no_match_count);
		break;
	case PhysicalType::INT64:
		TemplatedMatchType<int64_t, OP, NO_MATCH_SEL>(col, rows, sel, count, col_offset, col_no, no_match,
		                                              no_match_count);
		break;
	case PhysicalType::UINT8:
		TemplatedMatchType<uint8_t, OP, NO_MATCH_SEL>(col, rows, sel, count, col_offset, col_no, no_match,
		                                              no_match_count);
		break;
	case PhysicalType::UINT16:
		TemplatedMatchType<uint16_t, OP, NO_MATCH_SEL>(col, rows, sel, count, col_offset, col_no, no_match,
		                                               no_match_count);
		break;
	case PhysicalType::UINT32:
		TemplatedMatchType<uint32_t, OP, NO_MATCH_SEL>(col, rows, sel, count, col_offset, col_no, no_match,
		                                               no_match_count);
		break;
	case PhysicalType::UINT64:
		TemplatedMatchType<uint64_t, OP, NO_MATCH_SEL>(col, rows, sel, count, col_offset, col_no, no_match,
		                                               no_match_count);
		break;
	case PhysicalType::INT128:
		TemplatedMatchType<hugeint_t, OP, NO_MATCH_SEL>(col, rows, sel, count, col_offset, col_no, no_match,
		                                                no_match_count);
		break;
	case PhysicalType::FLOAT:
		TemplatedMatchType<float, OP, NO_MATCH_SEL>(col, rows, sel, count, col_offset, col_no, no_match,
		                                            no_match_count);
		break;
	case PhysicalType::DOUBLE:
		TemplatedMatchType<double, OP, NO_MATCH_SEL>(col, rows, sel, count, col_offset, col_no, no_match,
		                                             no_match_count);
		break;
	case PhysicalType::INTERVAL:
		TemplatedMatchType<interval_t, OP, NO_MATCH_SEL>(col, rows, sel, count, col_offset, col_no, no_match,
		                                                 no_match_count);
		break;
	case PhysicalType::VARCHAR:
		TemplatedMatchType<string_t, OP, NO_MATCH_SEL>(col, rows, sel, count, col_offset, col_no, no_match,
		                                               no_match_count);
		break;
	case PhysicalType::STRUCT:
		TemplatedMatchStruct<OP, NO_MATCH_SEL>(vec, col, layout, rows, sel, count, col_no, no_match, no_match_count,
		                                       original_count);
		break;
	case PhysicalType::LIST:
		TemplatedMatchList<OP, NO_MATCH_SEL>(vec, rows, sel, count, layout, col_no, no_match, no_match_count);
		break;
	default:
		throw InternalException("Unsupported column type for RowOperations::Match");
	}
}

template <bool NO_MATCH_SEL>
static void TemplatedMatch(DataChunk &columns, UnifiedVectorFormat col_data[], const TupleDataLayout &layout,
                           Vector &rows, const Predicates &predicates, SelectionVector &sel, idx_t &count,
                           SelectionVector *no_match, idx_t &no_match_count) {
	for (idx_t col_no = 0; col_no < predicates.size(); ++col_no) {
		auto &vec = columns.data[col_no];
		auto &col = col_data[col_no];
		switch (predicates[col_no]) {
		case ExpressionType::COMPARE_EQUAL:
		case ExpressionType::COMPARE_NOT_DISTINCT_FROM:
		case ExpressionType::COMPARE_DISTINCT_FROM:
			TemplatedMatchOp<Equals, NO_MATCH_SEL>(vec, col, layout, rows, sel, count, col_no, no_match, no_match_count,
			                                       count);
			break;
		case ExpressionType::COMPARE_NOTEQUAL:
			TemplatedMatchOp<NotEquals, NO_MATCH_SEL>(vec, col, layout, rows, sel, count, col_no, no_match,
			                                          no_match_count, count);
			break;
		case ExpressionType::COMPARE_GREATERTHAN:
			TemplatedMatchOp<GreaterThan, NO_MATCH_SEL>(vec, col, layout, rows, sel, count, col_no, no_match,
			                                            no_match_count, count);
			break;
		case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
			TemplatedMatchOp<GreaterThanEquals, NO_MATCH_SEL>(vec, col, layout, rows, sel, count, col_no, no_match,
			                                                  no_match_count, count);
			break;
		case ExpressionType::COMPARE_LESSTHAN:
			TemplatedMatchOp<LessThan, NO_MATCH_SEL>(vec, col, layout, rows, sel, count, col_no, no_match,
			                                         no_match_count, count);
			break;
		case ExpressionType::COMPARE_LESSTHANOREQUALTO:
			TemplatedMatchOp<LessThanEquals, NO_MATCH_SEL>(vec, col, layout, rows, sel, count, col_no, no_match,
			                                               no_match_count, count);
			break;
		default:
			throw InternalException("Unsupported comparison type for RowOperations::Match");
		}
	}
}

idx_t RowOperations::Match(DataChunk &columns, UnifiedVectorFormat col_data[], const TupleDataLayout &layout,
                           Vector &rows, const Predicates &predicates, SelectionVector &sel, idx_t count,
                           SelectionVector *no_match, idx_t &no_match_count) {
	if (no_match) {
		TemplatedMatch<true>(columns, col_data, layout, rows, predicates, sel, count, no_match, no_match_count);
	} else {
		TemplatedMatch<false>(columns, col_data, layout, rows, predicates, sel, count, no_match, no_match_count);
	}

	return count;
}

} // namespace duckdb
