// See https://raw.githubusercontent.com/duckdb/duckdb/master/LICENSE for licensing information

#include "duckdb.hpp"
#include "duckdb-internal.hpp"
#ifndef DUCKDB_AMALGAMATION
#error header mismatch
#endif





namespace duckdb {

ColumnDataAllocator::ColumnDataAllocator(Allocator &allocator) : type(ColumnDataAllocatorType::IN_MEMORY_ALLOCATOR) {
	alloc.allocator = &allocator;
}

ColumnDataAllocator::ColumnDataAllocator(BufferManager &buffer_manager)
    : type(ColumnDataAllocatorType::BUFFER_MANAGER_ALLOCATOR) {
	alloc.buffer_manager = &buffer_manager;
}

ColumnDataAllocator::ColumnDataAllocator(ClientContext &context, ColumnDataAllocatorType allocator_type)
    : type(allocator_type) {
	switch (type) {
	case ColumnDataAllocatorType::BUFFER_MANAGER_ALLOCATOR:
		alloc.buffer_manager = &BufferManager::GetBufferManager(context);
		break;
	case ColumnDataAllocatorType::IN_MEMORY_ALLOCATOR:
		alloc.allocator = &Allocator::Get(context);
		break;
	default:
		throw InternalException("Unrecognized column data allocator type");
	}
}

BufferHandle ColumnDataAllocator::Pin(uint32_t block_id) {
	D_ASSERT(type == ColumnDataAllocatorType::BUFFER_MANAGER_ALLOCATOR);
	if (shared) {
		lock_guard<mutex> guard(lock);
		return PinInternal(block_id);
	} else {
		return PinInternal(block_id);
	}
}

BufferHandle ColumnDataAllocator::PinInternal(uint32_t block_id) {
	return alloc.buffer_manager->Pin(blocks[block_id].handle);
}

BufferHandle ColumnDataAllocator::AllocateBlock() {
	D_ASSERT(type == ColumnDataAllocatorType::BUFFER_MANAGER_ALLOCATOR);
	BlockMetaData data;
	data.size = 0;
	data.capacity = Storage::BLOCK_SIZE;
	auto pin = alloc.buffer_manager->Allocate(Storage::BLOCK_SIZE, false, &data.handle);
	blocks.push_back(move(data));
	return pin;
}

void ColumnDataAllocator::AllocateEmptyBlock(idx_t size) {
	auto allocation_amount = MaxValue<idx_t>(NextPowerOfTwo(size), 4096);
	if (!blocks.empty()) {
		auto last_capacity = blocks.back().capacity;
		auto next_capacity = MinValue<idx_t>(last_capacity * 2, last_capacity + Storage::BLOCK_SIZE);
		allocation_amount = MaxValue<idx_t>(next_capacity, allocation_amount);
	}
	D_ASSERT(type == ColumnDataAllocatorType::IN_MEMORY_ALLOCATOR);
	BlockMetaData data;
	data.size = 0;
	data.capacity = allocation_amount;
	data.handle = nullptr;
	blocks.push_back(move(data));
}

void ColumnDataAllocator::AssignPointer(uint32_t &block_id, uint32_t &offset, data_ptr_t pointer) {
	auto pointer_value = uintptr_t(pointer);
	if (sizeof(uintptr_t) == sizeof(uint32_t)) {
		block_id = uint32_t(pointer_value);
	} else if (sizeof(uintptr_t) == sizeof(uint64_t)) {
		block_id = uint32_t(pointer_value & 0xFFFFFFFF);
		offset = uint32_t(pointer_value >> 32);
	} else {
		throw InternalException("ColumnDataCollection: Architecture not supported!?");
	}
}

void ColumnDataAllocator::AllocateBuffer(idx_t size, uint32_t &block_id, uint32_t &offset,
                                         ChunkManagementState *chunk_state) {
	D_ASSERT(allocated_data.empty());
	if (blocks.empty() || blocks.back().Capacity() < size) {
		auto pinned_block = AllocateBlock();
		if (chunk_state) {
			D_ASSERT(!blocks.empty());
			auto new_block_id = blocks.size() - 1;
			chunk_state->handles[new_block_id] = move(pinned_block);
		}
	}
	auto &block = blocks.back();
	D_ASSERT(size <= block.capacity - block.size);
	block_id = blocks.size() - 1;
	offset = block.size;
	block.size += size;
}

void ColumnDataAllocator::AllocateMemory(idx_t size, uint32_t &block_id, uint32_t &offset,
                                         ChunkManagementState *chunk_state) {
	D_ASSERT(blocks.size() == allocated_data.size());
	if (blocks.empty() || blocks.back().Capacity() < size) {
		AllocateEmptyBlock(size);
		auto &last_block = blocks.back();
		auto allocated = alloc.allocator->Allocate(last_block.capacity);
		allocated_data.push_back(move(allocated));
	}
	auto &block = blocks.back();
	D_ASSERT(size <= block.capacity - block.size);
	AssignPointer(block_id, offset, allocated_data.back().get() + block.size);
	block.size += size;
}

void ColumnDataAllocator::AllocateData(idx_t size, uint32_t &block_id, uint32_t &offset,
                                       ChunkManagementState *chunk_state) {
	switch (type) {
	case ColumnDataAllocatorType::BUFFER_MANAGER_ALLOCATOR:
		if (shared) {
			lock_guard<mutex> guard(lock);
			AllocateBuffer(size, block_id, offset, chunk_state);
		} else {
			AllocateBuffer(size, block_id, offset, chunk_state);
		}
		break;
	case ColumnDataAllocatorType::IN_MEMORY_ALLOCATOR:
		D_ASSERT(!shared);
		AllocateMemory(size, block_id, offset, chunk_state);
		break;
	default:
		throw InternalException("Unrecognized allocator type");
	}
}

void ColumnDataAllocator::Initialize(ColumnDataAllocator &other) {
	D_ASSERT(other.HasBlocks());
	blocks.push_back(other.blocks.back());
}

data_ptr_t ColumnDataAllocator::GetDataPointer(ChunkManagementState &state, uint32_t block_id, uint32_t offset) {
	if (type == ColumnDataAllocatorType::IN_MEMORY_ALLOCATOR) {
		// in-memory allocator: construct pointer from block_id and offset
		if (sizeof(uintptr_t) == sizeof(uint32_t)) {
			uintptr_t pointer_value = uintptr_t(block_id);
			return (data_ptr_t)pointer_value;
		} else if (sizeof(uintptr_t) == sizeof(uint64_t)) {
			uintptr_t pointer_value = (uintptr_t(offset) << 32) | uintptr_t(block_id);
			return (data_ptr_t)pointer_value;
		} else {
			throw InternalException("ColumnDataCollection: Architecture not supported!?");
		}
	}
	D_ASSERT(state.handles.find(block_id) != state.handles.end());
	return state.handles[block_id].Ptr() + offset;
}

void ColumnDataAllocator::DeleteBlock(uint32_t block_id) {
	blocks[block_id].handle->SetCanDestroy(true);
}

Allocator &ColumnDataAllocator::GetAllocator() {
	return type == ColumnDataAllocatorType::IN_MEMORY_ALLOCATOR ? *alloc.allocator
	                                                            : alloc.buffer_manager->GetBufferAllocator();
}

void ColumnDataAllocator::InitializeChunkState(ChunkManagementState &state, ChunkMetaData &chunk) {
	if (type != ColumnDataAllocatorType::BUFFER_MANAGER_ALLOCATOR) {
		// nothing to pin
		return;
	}
	// release any handles that are no longer required
	bool found_handle;
	do {
		found_handle = false;
		for (auto it = state.handles.begin(); it != state.handles.end(); it++) {
			if (chunk.block_ids.find(it->first) != chunk.block_ids.end()) {
				// still required: do not release
				continue;
			}
			state.handles.erase(it);
			found_handle = true;
			break;
		}
	} while (found_handle);

	// grab any handles that are now required
	for (auto &block_id : chunk.block_ids) {
		if (state.handles.find(block_id) != state.handles.end()) {
			// already pinned: don't need to do anything
			continue;
		}
		state.handles[block_id] = Pin(block_id);
	}
}

uint32_t BlockMetaData::Capacity() {
	D_ASSERT(size <= capacity);
	return capacity - size;
}

} // namespace duckdb








namespace duckdb {

struct ColumnDataMetaData;

typedef void (*column_data_copy_function_t)(ColumnDataMetaData &meta_data, const UnifiedVectorFormat &source_data,
                                            Vector &source, idx_t offset, idx_t copy_count);

struct ColumnDataCopyFunction {
	column_data_copy_function_t function;
	vector<ColumnDataCopyFunction> child_functions;
};

struct ColumnDataMetaData {
	ColumnDataMetaData(ColumnDataCopyFunction &copy_function, ColumnDataCollectionSegment &segment,
	                   ColumnDataAppendState &state, ChunkMetaData &chunk_data, VectorDataIndex vector_data_index)
	    : copy_function(copy_function), segment(segment), state(state), chunk_data(chunk_data),
	      vector_data_index(vector_data_index) {
	}
	ColumnDataMetaData(ColumnDataCopyFunction &copy_function, ColumnDataMetaData &parent,
	                   VectorDataIndex vector_data_index)
	    : copy_function(copy_function), segment(parent.segment), state(parent.state), chunk_data(parent.chunk_data),
	      vector_data_index(vector_data_index) {
	}

	ColumnDataCopyFunction &copy_function;
	ColumnDataCollectionSegment &segment;
	ColumnDataAppendState &state;
	ChunkMetaData &chunk_data;
	VectorDataIndex vector_data_index;
	idx_t child_list_size = DConstants::INVALID_INDEX;

	VectorMetaData &GetVectorMetaData() {
		return segment.GetVectorData(vector_data_index);
	}
};

//! Explicitly initialized without types
ColumnDataCollection::ColumnDataCollection(Allocator &allocator_p) {
	types.clear();
	count = 0;
	this->finished_append = false;
	allocator = make_shared<ColumnDataAllocator>(allocator_p);
}

ColumnDataCollection::ColumnDataCollection(Allocator &allocator_p, vector<LogicalType> types_p) {
	Initialize(move(types_p));
	allocator = make_shared<ColumnDataAllocator>(allocator_p);
}

ColumnDataCollection::ColumnDataCollection(BufferManager &buffer_manager, vector<LogicalType> types_p) {
	Initialize(move(types_p));
	allocator = make_shared<ColumnDataAllocator>(buffer_manager);
}

ColumnDataCollection::ColumnDataCollection(shared_ptr<ColumnDataAllocator> allocator_p, vector<LogicalType> types_p) {
	Initialize(move(types_p));
	this->allocator = move(allocator_p);
}

ColumnDataCollection::ColumnDataCollection(ClientContext &context, vector<LogicalType> types_p,
                                           ColumnDataAllocatorType type)
    : ColumnDataCollection(make_shared<ColumnDataAllocator>(context, type), move(types_p)) {
	D_ASSERT(!types.empty());
}

ColumnDataCollection::ColumnDataCollection(ColumnDataCollection &other)
    : ColumnDataCollection(other.allocator, other.types) {
	other.finished_append = true;
	D_ASSERT(!types.empty());
}

ColumnDataCollection::~ColumnDataCollection() {
}

void ColumnDataCollection::Initialize(vector<LogicalType> types_p) {
	this->types = move(types_p);
	this->count = 0;
	this->finished_append = false;
	D_ASSERT(!types.empty());
	copy_functions.reserve(types.size());
	for (auto &type : types) {
		copy_functions.push_back(GetCopyFunction(type));
	}
}

void ColumnDataCollection::CreateSegment() {
	segments.emplace_back(make_unique<ColumnDataCollectionSegment>(allocator, types));
}

//===--------------------------------------------------------------------===//
// ColumnDataRow
//===--------------------------------------------------------------------===//
ColumnDataRow::ColumnDataRow(DataChunk &chunk_p, idx_t row_index, idx_t base_index)
    : chunk(chunk_p), row_index(row_index), base_index(base_index) {
}

Value ColumnDataRow::GetValue(idx_t column_index) const {
	D_ASSERT(column_index < chunk.ColumnCount());
	D_ASSERT(row_index < chunk.size());
	return chunk.data[column_index].GetValue(row_index);
}

idx_t ColumnDataRow::RowIndex() const {
	return base_index + row_index;
}

//===--------------------------------------------------------------------===//
// ColumnDataRowCollection
//===--------------------------------------------------------------------===//
ColumnDataRowCollection::ColumnDataRowCollection(const ColumnDataCollection &collection) {
	if (collection.Count() == 0) {
		return;
	}
	// read all the chunks
	ColumnDataScanState temp_scan_state;
	collection.InitializeScan(temp_scan_state);
	while (true) {
		auto chunk = make_unique<DataChunk>();
		collection.InitializeScanChunk(*chunk);
		if (!collection.Scan(temp_scan_state, *chunk)) {
			break;
		}
		// we keep the BufferHandles that are needed for the materialized collection pinned in the supplied scan_state
		auto &temp_handles = temp_scan_state.current_chunk_state.handles;
		auto &scan_handles = scan_state.current_chunk_state.handles;
		for (auto &temp_handle_pair : temp_handles) {
			auto handle_copy = make_pair<uint32_t, BufferHandle>(scan_handles.size(), move(temp_handle_pair.second));
			scan_state.current_chunk_state.handles.insert(move(handle_copy));
		}
		chunks.push_back(move(chunk));
	}
	// now create all of the column data rows
	rows.reserve(collection.Count());
	idx_t base_row = 0;
	for (auto &chunk : chunks) {
		for (idx_t row_idx = 0; row_idx < chunk->size(); row_idx++) {
			rows.emplace_back(*chunk, row_idx, base_row);
		}
		base_row += chunk->size();
	}
}

ColumnDataRow &ColumnDataRowCollection::operator[](idx_t i) {
	return rows[i];
}

const ColumnDataRow &ColumnDataRowCollection::operator[](idx_t i) const {
	return rows[i];
}

Value ColumnDataRowCollection::GetValue(idx_t column, idx_t index) const {
	return rows[index].GetValue(column);
}

//===--------------------------------------------------------------------===//
// ColumnDataChunkIterator
//===--------------------------------------------------------------------===//
ColumnDataChunkIterationHelper ColumnDataCollection::Chunks() const {
	vector<column_t> column_ids;
	for (idx_t i = 0; i < ColumnCount(); i++) {
		column_ids.push_back(i);
	}
	return Chunks(column_ids);
}

ColumnDataChunkIterationHelper ColumnDataCollection::Chunks(vector<column_t> column_ids) const {
	return ColumnDataChunkIterationHelper(*this, move(column_ids));
}

ColumnDataChunkIterationHelper::ColumnDataChunkIterationHelper(const ColumnDataCollection &collection_p,
                                                               vector<column_t> column_ids_p)
    : collection(collection_p), column_ids(move(column_ids_p)) {
}

ColumnDataChunkIterationHelper::ColumnDataChunkIterator::ColumnDataChunkIterator(
    const ColumnDataCollection *collection_p, vector<column_t> column_ids_p)
    : collection(collection_p), scan_chunk(make_shared<DataChunk>()), row_index(0) {
	if (!collection) {
		return;
	}
	collection->InitializeScan(scan_state, move(column_ids_p));
	collection->InitializeScanChunk(scan_state, *scan_chunk);
	collection->Scan(scan_state, *scan_chunk);
}

void ColumnDataChunkIterationHelper::ColumnDataChunkIterator::Next() {
	if (!collection) {
		return;
	}
	if (!collection->Scan(scan_state, *scan_chunk)) {
		collection = nullptr;
		row_index = 0;
	} else {
		row_index += scan_chunk->size();
	}
}

ColumnDataChunkIterationHelper::ColumnDataChunkIterator &
ColumnDataChunkIterationHelper::ColumnDataChunkIterator::operator++() {
	Next();
	return *this;
}

bool ColumnDataChunkIterationHelper::ColumnDataChunkIterator::operator!=(const ColumnDataChunkIterator &other) const {
	return collection != other.collection || row_index != other.row_index;
}

DataChunk &ColumnDataChunkIterationHelper::ColumnDataChunkIterator::operator*() const {
	return *scan_chunk;
}

//===--------------------------------------------------------------------===//
// ColumnDataRowIterator
//===--------------------------------------------------------------------===//
ColumnDataRowIterationHelper ColumnDataCollection::Rows() const {
	return ColumnDataRowIterationHelper(*this);
}

ColumnDataRowIterationHelper::ColumnDataRowIterationHelper(const ColumnDataCollection &collection_p)
    : collection(collection_p) {
}

ColumnDataRowIterationHelper::ColumnDataRowIterator::ColumnDataRowIterator(const ColumnDataCollection *collection_p)
    : collection(collection_p), scan_chunk(make_shared<DataChunk>()), current_row(*scan_chunk, 0, 0) {
	if (!collection) {
		return;
	}
	collection->InitializeScan(scan_state);
	collection->InitializeScanChunk(*scan_chunk);
	collection->Scan(scan_state, *scan_chunk);
}

void ColumnDataRowIterationHelper::ColumnDataRowIterator::Next() {
	if (!collection) {
		return;
	}
	current_row.row_index++;
	if (current_row.row_index >= scan_chunk->size()) {
		current_row.base_index += scan_chunk->size();
		current_row.row_index = 0;
		if (!collection->Scan(scan_state, *scan_chunk)) {
			// exhausted collection: move iterator to nop state
			current_row.base_index = 0;
			collection = nullptr;
		}
	}
}

ColumnDataRowIterationHelper::ColumnDataRowIterator ColumnDataRowIterationHelper::begin() { // NOLINT
	return ColumnDataRowIterationHelper::ColumnDataRowIterator(collection.Count() == 0 ? nullptr : &collection);
}
ColumnDataRowIterationHelper::ColumnDataRowIterator ColumnDataRowIterationHelper::end() { // NOLINT
	return ColumnDataRowIterationHelper::ColumnDataRowIterator(nullptr);
}

ColumnDataRowIterationHelper::ColumnDataRowIterator &ColumnDataRowIterationHelper::ColumnDataRowIterator::operator++() {
	Next();
	return *this;
}

bool ColumnDataRowIterationHelper::ColumnDataRowIterator::operator!=(const ColumnDataRowIterator &other) const {
	return collection != other.collection || current_row.row_index != other.current_row.row_index ||
	       current_row.base_index != other.current_row.base_index;
}

const ColumnDataRow &ColumnDataRowIterationHelper::ColumnDataRowIterator::operator*() const {
	return current_row;
}

//===--------------------------------------------------------------------===//
// Append
//===--------------------------------------------------------------------===//
void ColumnDataCollection::InitializeAppend(ColumnDataAppendState &state) {
	D_ASSERT(!finished_append);
	state.vector_data.resize(types.size());
	if (segments.empty()) {
		CreateSegment();
	}
	auto &segment = *segments.back();
	if (segment.chunk_data.empty()) {
		segment.AllocateNewChunk();
	}
	segment.InitializeChunkState(segment.chunk_data.size() - 1, state.current_chunk_state);
}

void ColumnDataCopyValidity(const UnifiedVectorFormat &source_data, validity_t *target, idx_t source_offset,
                            idx_t target_offset, idx_t copy_count) {
	ValidityMask validity(target);
	if (target_offset == 0) {
		// first time appending to this vector
		// all data here is still uninitialized
		// initialize the validity mask to set all to valid
		validity.SetAllValid(STANDARD_VECTOR_SIZE);
	}
	// FIXME: we can do something more optimized here using bitshifts & bitwise ors
	if (!source_data.validity.AllValid()) {
		for (idx_t i = 0; i < copy_count; i++) {
			auto idx = source_data.sel->get_index(source_offset + i);
			if (!source_data.validity.RowIsValid(idx)) {
				validity.SetInvalid(target_offset + i);
			}
		}
	}
}

template <class T>
struct BaseValueCopy {
	static idx_t TypeSize() {
		return sizeof(T);
	}

	template <class OP>
	static void Assign(ColumnDataMetaData &meta_data, data_ptr_t target, data_ptr_t source, idx_t target_idx,
	                   idx_t source_idx) {
		auto result_data = (T *)target;
		auto source_data = (T *)source;
		result_data[target_idx] = OP::Operation(meta_data, source_data[source_idx]);
	}
};

template <class T>
struct StandardValueCopy : public BaseValueCopy<T> {
	static T Operation(ColumnDataMetaData &, T input) {
		return input;
	}
};

struct StringValueCopy : public BaseValueCopy<string_t> {
	static string_t Operation(ColumnDataMetaData &meta_data, string_t input) {
		return input.IsInlined() ? input : meta_data.segment.heap.AddBlob(input);
	}
};

struct ListValueCopy : public BaseValueCopy<list_entry_t> {
	using TYPE = list_entry_t;

	static TYPE Operation(ColumnDataMetaData &meta_data, TYPE input) {
		input.offset += meta_data.child_list_size;
		return input;
	}
};

struct StructValueCopy {
	static idx_t TypeSize() {
		return 0;
	}

	template <class OP>
	static void Assign(ColumnDataMetaData &meta_data, data_ptr_t target, data_ptr_t source, idx_t target_idx,
	                   idx_t source_idx) {
	}
};

template <class OP>
static void TemplatedColumnDataCopy(ColumnDataMetaData &meta_data, const UnifiedVectorFormat &source_data,
                                    Vector &source, idx_t offset, idx_t count) {
	auto &segment = meta_data.segment;
	auto &append_state = meta_data.state;

	auto current_index = meta_data.vector_data_index;
	idx_t remaining = count;
	while (remaining > 0) {
		auto &current_segment = segment.GetVectorData(current_index);
		idx_t append_count = MinValue<idx_t>(STANDARD_VECTOR_SIZE - current_segment.count, remaining);

		auto base_ptr = meta_data.segment.allocator->GetDataPointer(append_state.current_chunk_state,
		                                                            current_segment.block_id, current_segment.offset);
		auto validity_data = ColumnDataCollectionSegment::GetValidityPointer(base_ptr, OP::TypeSize());

		ValidityMask result_validity(validity_data);
		if (current_segment.count == 0) {
			// first time appending to this vector
			// all data here is still uninitialized
			// initialize the validity mask to set all to valid
			result_validity.SetAllValid(STANDARD_VECTOR_SIZE);
		}
		for (idx_t i = 0; i < append_count; i++) {
			auto source_idx = source_data.sel->get_index(offset + i);
			if (source_data.validity.RowIsValid(source_idx)) {
				OP::template Assign<OP>(meta_data, base_ptr, source_data.data, current_segment.count + i, source_idx);
			} else {
				result_validity.SetInvalid(current_segment.count + i);
			}
		}
		current_segment.count += append_count;
		offset += append_count;
		remaining -= append_count;
		if (remaining > 0) {
			// need to append more, check if we need to allocate a new vector or not
			if (!current_segment.next_data.IsValid()) {
				segment.AllocateVector(source.GetType(), meta_data.chunk_data, meta_data.state, current_index);
			}
			D_ASSERT(segment.GetVectorData(current_index).next_data.IsValid());
			current_index = segment.GetVectorData(current_index).next_data;
		}
	}
}

template <class T>
static void ColumnDataCopy(ColumnDataMetaData &meta_data, const UnifiedVectorFormat &source_data, Vector &source,
                           idx_t offset, idx_t copy_count) {
	TemplatedColumnDataCopy<StandardValueCopy<T>>(meta_data, source_data, source, offset, copy_count);
}

template <>
void ColumnDataCopy<string_t>(ColumnDataMetaData &meta_data, const UnifiedVectorFormat &source_data, Vector &source,
                              idx_t offset, idx_t copy_count) {
	TemplatedColumnDataCopy<StringValueCopy>(meta_data, source_data, source, offset, copy_count);
}

template <>
void ColumnDataCopy<list_entry_t>(ColumnDataMetaData &meta_data, const UnifiedVectorFormat &source_data, Vector &source,
                                  idx_t offset, idx_t copy_count) {
	auto &segment = meta_data.segment;

	// first append the child entries of the list
	auto &child_vector = ListVector::GetEntry(source);
	idx_t child_list_size = ListVector::GetListSize(source);
	auto &child_type = child_vector.GetType();

	UnifiedVectorFormat child_vector_data;
	child_vector.ToUnifiedFormat(child_list_size, child_vector_data);

	if (!meta_data.GetVectorMetaData().child_index.IsValid()) {
		auto child_index = segment.AllocateVector(child_type, meta_data.chunk_data, meta_data.state);
		meta_data.GetVectorMetaData().child_index = meta_data.segment.AddChildIndex(child_index);
	}
	auto &child_function = meta_data.copy_function.child_functions[0];
	auto child_index = segment.GetChildIndex(meta_data.GetVectorMetaData().child_index);
	// figure out the current list size by traversing the set of child entries
	idx_t current_list_size = 0;
	auto current_child_index = child_index;
	while (current_child_index.IsValid()) {
		auto &child_vdata = segment.GetVectorData(current_child_index);
		current_list_size += child_vdata.count;
		current_child_index = child_vdata.next_data;
	}
	ColumnDataMetaData child_meta_data(child_function, meta_data, child_index);
	// FIXME: appending the entire child list here is not required
	// We can also scan the actual list entries required per the offset/copy_count
	child_function.function(child_meta_data, child_vector_data, child_vector, 0, child_list_size);

	// now copy the list entries
	meta_data.child_list_size = current_list_size;
	TemplatedColumnDataCopy<ListValueCopy>(meta_data, source_data, source, offset, copy_count);
}

void ColumnDataCopyStruct(ColumnDataMetaData &meta_data, const UnifiedVectorFormat &source_data, Vector &source,
                          idx_t offset, idx_t copy_count) {
	auto &segment = meta_data.segment;

	// copy the NULL values for the main struct vector
	TemplatedColumnDataCopy<StructValueCopy>(meta_data, source_data, source, offset, copy_count);

	auto &child_types = StructType::GetChildTypes(source.GetType());
	// now copy all the child vectors
	D_ASSERT(meta_data.GetVectorMetaData().child_index.IsValid());
	auto &child_vectors = StructVector::GetEntries(source);
	for (idx_t child_idx = 0; child_idx < child_types.size(); child_idx++) {
		auto &child_function = meta_data.copy_function.child_functions[child_idx];
		auto child_index = segment.GetChildIndex(meta_data.GetVectorMetaData().child_index, child_idx);
		ColumnDataMetaData child_meta_data(child_function, meta_data, child_index);

		UnifiedVectorFormat child_data;
		child_vectors[child_idx]->ToUnifiedFormat(copy_count, child_data);

		child_function.function(child_meta_data, child_data, *child_vectors[child_idx], offset, copy_count);
	}
}

ColumnDataCopyFunction ColumnDataCollection::GetCopyFunction(const LogicalType &type) {
	ColumnDataCopyFunction result;
	column_data_copy_function_t function;
	switch (type.InternalType()) {
	case PhysicalType::BOOL:
		function = ColumnDataCopy<bool>;
		break;
	case PhysicalType::INT8:
		function = ColumnDataCopy<int8_t>;
		break;
	case PhysicalType::INT16:
		function = ColumnDataCopy<int16_t>;
		break;
	case PhysicalType::INT32:
		function = ColumnDataCopy<int32_t>;
		break;
	case PhysicalType::INT64:
		function = ColumnDataCopy<int64_t>;
		break;
	case PhysicalType::INT128:
		function = ColumnDataCopy<hugeint_t>;
		break;
	case PhysicalType::UINT8:
		function = ColumnDataCopy<uint8_t>;
		break;
	case PhysicalType::UINT16:
		function = ColumnDataCopy<uint16_t>;
		break;
	case PhysicalType::UINT32:
		function = ColumnDataCopy<uint32_t>;
		break;
	case PhysicalType::UINT64:
		function = ColumnDataCopy<uint64_t>;
		break;
	case PhysicalType::FLOAT:
		function = ColumnDataCopy<float>;
		break;
	case PhysicalType::DOUBLE:
		function = ColumnDataCopy<double>;
		break;
	case PhysicalType::INTERVAL:
		function = ColumnDataCopy<interval_t>;
		break;
	case PhysicalType::VARCHAR:
		function = ColumnDataCopy<string_t>;
		break;
	case PhysicalType::STRUCT: {
		function = ColumnDataCopyStruct;
		auto &child_types = StructType::GetChildTypes(type);
		for (auto &kv : child_types) {
			result.child_functions.push_back(GetCopyFunction(kv.second));
		}
		break;
	}
	case PhysicalType::LIST: {
		function = ColumnDataCopy<list_entry_t>;
		auto child_function = GetCopyFunction(ListType::GetChildType(type));
		result.child_functions.push_back(child_function);
		break;
	}
	default:
		throw InternalException("Unsupported type for ColumnDataCollection::GetCopyFunction");
	}
	result.function = function;
	return result;
}

static bool IsComplexType(const LogicalType &type) {
	switch (type.InternalType()) {
	case PhysicalType::STRUCT:
	case PhysicalType::LIST:
		return true;
	default:
		return false;
	};
}

void ColumnDataCollection::Append(ColumnDataAppendState &state, DataChunk &input) {
	D_ASSERT(!finished_append);
	D_ASSERT(types == input.GetTypes());

	auto &segment = *segments.back();
	for (idx_t vector_idx = 0; vector_idx < types.size(); vector_idx++) {
		if (IsComplexType(input.data[vector_idx].GetType())) {
			input.data[vector_idx].Flatten(input.size());
		}
		input.data[vector_idx].ToUnifiedFormat(input.size(), state.vector_data[vector_idx]);
	}

	idx_t remaining = input.size();
	while (remaining > 0) {
		auto &chunk_data = segment.chunk_data.back();
		idx_t append_amount = MinValue<idx_t>(remaining, STANDARD_VECTOR_SIZE - chunk_data.count);
		if (append_amount > 0) {
			idx_t offset = input.size() - remaining;
			for (idx_t vector_idx = 0; vector_idx < types.size(); vector_idx++) {
				ColumnDataMetaData meta_data(copy_functions[vector_idx], segment, state, chunk_data,
				                             chunk_data.vector_data[vector_idx]);
				copy_functions[vector_idx].function(meta_data, state.vector_data[vector_idx], input.data[vector_idx],
				                                    offset, append_amount);
			}
			chunk_data.count += append_amount;
		}
		remaining -= append_amount;
		if (remaining > 0) {
			// more to do
			// allocate a new chunk
			segment.AllocateNewChunk();
			segment.InitializeChunkState(segment.chunk_data.size() - 1, state.current_chunk_state);
		}
	}
	segment.count += input.size();
	count += input.size();
}

void ColumnDataCollection::Append(DataChunk &input) {
	ColumnDataAppendState state;
	InitializeAppend(state);
	Append(state, input);
}

//===--------------------------------------------------------------------===//
// Scan
//===--------------------------------------------------------------------===//
void ColumnDataCollection::InitializeScan(ColumnDataScanState &state, ColumnDataScanProperties properties) const {
	vector<column_t> column_ids;
	column_ids.reserve(types.size());
	for (idx_t i = 0; i < types.size(); i++) {
		column_ids.push_back(i);
	}
	InitializeScan(state, move(column_ids), properties);
}

void ColumnDataCollection::InitializeScan(ColumnDataScanState &state, vector<column_t> column_ids,
                                          ColumnDataScanProperties properties) const {
	state.chunk_index = 0;
	state.segment_index = 0;
	state.current_row_index = 0;
	state.next_row_index = 0;
	state.current_chunk_state.handles.clear();
	state.properties = properties;
	state.column_ids = move(column_ids);
}

void ColumnDataCollection::InitializeScan(ColumnDataParallelScanState &state,
                                          ColumnDataScanProperties properties) const {
	InitializeScan(state.scan_state, properties);
}

void ColumnDataCollection::InitializeScan(ColumnDataParallelScanState &state, vector<column_t> column_ids,
                                          ColumnDataScanProperties properties) const {
	InitializeScan(state.scan_state, move(column_ids), properties);
}

bool ColumnDataCollection::Scan(ColumnDataParallelScanState &state, ColumnDataLocalScanState &lstate,
                                DataChunk &result) const {
	result.Reset();

	idx_t chunk_index;
	idx_t segment_index;
	idx_t row_index;
	{
		lock_guard<mutex> l(state.lock);
		if (!NextScanIndex(state.scan_state, chunk_index, segment_index, row_index)) {
			return false;
		}
	}
	ScanAtIndex(state, lstate, result, chunk_index, segment_index, row_index);
	return true;
}

void ColumnDataCollection::InitializeScanChunk(DataChunk &chunk) const {
	chunk.Initialize(allocator->GetAllocator(), types);
}

void ColumnDataCollection::InitializeScanChunk(ColumnDataScanState &state, DataChunk &chunk) const {
	D_ASSERT(!state.column_ids.empty());
	vector<LogicalType> chunk_types;
	chunk_types.reserve(state.column_ids.size());
	for (idx_t i = 0; i < state.column_ids.size(); i++) {
		auto column_idx = state.column_ids[i];
		D_ASSERT(column_idx < types.size());
		chunk_types.push_back(types[column_idx]);
	}
	chunk.Initialize(allocator->GetAllocator(), chunk_types);
}

bool ColumnDataCollection::NextScanIndex(ColumnDataScanState &state, idx_t &chunk_index, idx_t &segment_index,
                                         idx_t &row_index) const {
	row_index = state.current_row_index = state.next_row_index;
	// check if we still have collections to scan
	if (state.segment_index >= segments.size()) {
		// no more data left in the scan
		return false;
	}
	// check within the current collection if we still have chunks to scan
	while (state.chunk_index >= segments[state.segment_index]->chunk_data.size()) {
		// exhausted all chunks for this internal data structure: move to the next one
		state.chunk_index = 0;
		state.segment_index++;
		state.current_chunk_state.handles.clear();
		if (state.segment_index >= segments.size()) {
			return false;
		}
	}
	state.next_row_index += segments[state.segment_index]->chunk_data[state.chunk_index].count;
	segment_index = state.segment_index;
	chunk_index = state.chunk_index++;
	return true;
}

void ColumnDataCollection::ScanAtIndex(ColumnDataParallelScanState &state, ColumnDataLocalScanState &lstate,
                                       DataChunk &result, idx_t chunk_index, idx_t segment_index,
                                       idx_t row_index) const {
	if (segment_index != lstate.current_segment_index) {
		lstate.current_chunk_state.handles.clear();
		lstate.current_segment_index = segment_index;
	}
	auto &segment = *segments[segment_index];
	lstate.current_chunk_state.properties = state.scan_state.properties;
	segment.ReadChunk(chunk_index, lstate.current_chunk_state, result, state.scan_state.column_ids);
	lstate.current_row_index = row_index;
	result.Verify();
}

bool ColumnDataCollection::Scan(ColumnDataScanState &state, DataChunk &result) const {
	result.Reset();

	idx_t chunk_index;
	idx_t segment_index;
	idx_t row_index;
	if (!NextScanIndex(state, chunk_index, segment_index, row_index)) {
		return false;
	}

	// found a chunk to scan -> scan it
	auto &segment = *segments[segment_index];
	state.current_chunk_state.properties = state.properties;
	segment.ReadChunk(chunk_index, state.current_chunk_state, result, state.column_ids);
	result.Verify();
	return true;
}

ColumnDataRowCollection ColumnDataCollection::GetRows() const {
	return ColumnDataRowCollection(*this);
}

//===--------------------------------------------------------------------===//
// Combine
//===--------------------------------------------------------------------===//
void ColumnDataCollection::Combine(ColumnDataCollection &other) {
	if (other.count == 0) {
		return;
	}
	if (types != other.types) {
		throw InternalException("Attempting to combine ColumnDataCollections with mismatching types");
	}
	this->count += other.count;
	this->segments.reserve(segments.size() + other.segments.size());
	for (auto &other_seg : other.segments) {
		segments.push_back(move(other_seg));
	}
	Verify();
}

//===--------------------------------------------------------------------===//
// Fetch
//===--------------------------------------------------------------------===//
idx_t ColumnDataCollection::ChunkCount() const {
	idx_t chunk_count = 0;
	for (auto &segment : segments) {
		chunk_count += segment->ChunkCount();
	}
	return chunk_count;
}

void ColumnDataCollection::FetchChunk(idx_t chunk_idx, DataChunk &result) const {
	D_ASSERT(chunk_idx < ChunkCount());
	for (auto &segment : segments) {
		if (chunk_idx >= segment->ChunkCount()) {
			chunk_idx -= segment->ChunkCount();
		} else {
			segment->FetchChunk(chunk_idx, result);
			return;
		}
	}
	throw InternalException("Failed to find chunk in ColumnDataCollection");
}

//===--------------------------------------------------------------------===//
// Helpers
//===--------------------------------------------------------------------===//
void ColumnDataCollection::Verify() {
#ifdef DEBUG
	// verify counts
	idx_t total_segment_count = 0;
	for (auto &segment : segments) {
		segment->Verify();
		total_segment_count += segment->count;
	}
	D_ASSERT(total_segment_count == this->count);
#endif
}

string ColumnDataCollection::ToString() const {
	return "Column Data Collection";
}

void ColumnDataCollection::Print() const {
	Printer::Print(ToString());
}

void ColumnDataCollection::Reset() {
	count = 0;
	segments.clear();
}

bool ColumnDataCollection::ResultEquals(const ColumnDataCollection &left, const ColumnDataCollection &right,
                                        string &error_message) {
	if (left.ColumnCount() != right.ColumnCount()) {
		error_message = "Column count mismatch";
		return false;
	}
	if (left.Count() != right.Count()) {
		error_message = "Row count mismatch";
		return false;
	}
	auto left_rows = left.GetRows();
	auto right_rows = right.GetRows();
	for (idx_t r = 0; r < left.Count(); r++) {
		for (idx_t c = 0; c < left.ColumnCount(); c++) {
			auto lvalue = left_rows.GetValue(c, r);
			auto rvalue = left_rows.GetValue(c, r);
			if (!Value::DefaultValuesAreEqual(lvalue, rvalue)) {
				error_message =
				    StringUtil::Format("%s <> %s (row: %lld, col: %lld)\n", lvalue.ToString(), rvalue.ToString(), r, c);
				return false;
			}
		}
	}
	return true;
}

const vector<unique_ptr<ColumnDataCollectionSegment>> &ColumnDataCollection::GetSegments() const {
	return segments;
}

} // namespace duckdb


namespace duckdb {

ColumnDataCollectionSegment::ColumnDataCollectionSegment(shared_ptr<ColumnDataAllocator> allocator_p,
                                                         vector<LogicalType> types_p)
    : allocator(move(allocator_p)), types(move(types_p)), count(0), heap(allocator->GetAllocator()) {
}

idx_t ColumnDataCollectionSegment::GetDataSize(idx_t type_size) {
	return AlignValue(type_size * STANDARD_VECTOR_SIZE);
}

validity_t *ColumnDataCollectionSegment::GetValidityPointer(data_ptr_t base_ptr, idx_t type_size) {
	return (validity_t *)(base_ptr + GetDataSize(type_size));
}

VectorDataIndex ColumnDataCollectionSegment::AllocateVectorInternal(const LogicalType &type, ChunkMetaData &chunk_meta,
                                                                    ChunkManagementState *chunk_state) {
	VectorMetaData meta_data;
	meta_data.count = 0;

	auto internal_type = type.InternalType();
	auto type_size = internal_type == PhysicalType::STRUCT ? 0 : GetTypeIdSize(internal_type);
	allocator->AllocateData(GetDataSize(type_size) + ValidityMask::STANDARD_MASK_SIZE, meta_data.block_id,
	                        meta_data.offset, chunk_state);
	if (allocator->GetType() == ColumnDataAllocatorType::BUFFER_MANAGER_ALLOCATOR) {
		chunk_meta.block_ids.insert(meta_data.block_id);
	}

	auto index = vector_data.size();
	vector_data.push_back(meta_data);
	return VectorDataIndex(index);
}

VectorDataIndex ColumnDataCollectionSegment::AllocateVector(const LogicalType &type, ChunkMetaData &chunk_meta,
                                                            ChunkManagementState *chunk_state,
                                                            VectorDataIndex prev_index) {
	auto index = AllocateVectorInternal(type, chunk_meta, chunk_state);
	if (prev_index.IsValid()) {
		GetVectorData(prev_index).next_data = index;
	}
	if (type.InternalType() == PhysicalType::STRUCT) {
		// initialize the struct children
		auto &child_types = StructType::GetChildTypes(type);
		auto base_child_index = ReserveChildren(child_types.size());
		for (idx_t child_idx = 0; child_idx < child_types.size(); child_idx++) {
			VectorDataIndex prev_child_index;
			if (prev_index.IsValid()) {
				prev_child_index = GetChildIndex(GetVectorData(prev_index).child_index, child_idx);
			}
			auto child_index = AllocateVector(child_types[child_idx].second, chunk_meta, chunk_state, prev_child_index);
			SetChildIndex(base_child_index, child_idx, child_index);
		}
		GetVectorData(index).child_index = base_child_index;
	}
	return index;
}

VectorDataIndex ColumnDataCollectionSegment::AllocateVector(const LogicalType &type, ChunkMetaData &chunk_meta,
                                                            ColumnDataAppendState &append_state,
                                                            VectorDataIndex prev_index) {
	return AllocateVector(type, chunk_meta, &append_state.current_chunk_state, prev_index);
}

void ColumnDataCollectionSegment::AllocateNewChunk() {
	ChunkMetaData meta_data;
	meta_data.count = 0;
	meta_data.vector_data.reserve(types.size());
	for (idx_t i = 0; i < types.size(); i++) {
		auto vector_idx = AllocateVector(types[i], meta_data);
		meta_data.vector_data.push_back(vector_idx);
	}
	chunk_data.push_back(move(meta_data));
}

void ColumnDataCollectionSegment::InitializeChunkState(idx_t chunk_index, ChunkManagementState &state) {
	auto &chunk = chunk_data[chunk_index];
	allocator->InitializeChunkState(state, chunk);
}

VectorDataIndex ColumnDataCollectionSegment::GetChildIndex(VectorChildIndex index, idx_t child_entry) {
	D_ASSERT(index.IsValid());
	D_ASSERT(index.index + child_entry < child_indices.size());
	return VectorDataIndex(child_indices[index.index + child_entry]);
}

VectorChildIndex ColumnDataCollectionSegment::AddChildIndex(VectorDataIndex index) {
	auto result = child_indices.size();
	child_indices.push_back(index);
	return VectorChildIndex(result);
}

VectorChildIndex ColumnDataCollectionSegment::ReserveChildren(idx_t child_count) {
	auto result = child_indices.size();
	for (idx_t i = 0; i < child_count; i++) {
		child_indices.emplace_back();
	}
	return VectorChildIndex(result);
}

void ColumnDataCollectionSegment::SetChildIndex(VectorChildIndex base_idx, idx_t child_number, VectorDataIndex index) {
	D_ASSERT(base_idx.IsValid());
	D_ASSERT(index.IsValid());
	D_ASSERT(base_idx.index + child_number < child_indices.size());
	child_indices[base_idx.index + child_number] = index;
}

idx_t ColumnDataCollectionSegment::ReadVectorInternal(ChunkManagementState &state, VectorDataIndex vector_index,
                                                      Vector &result) {
	auto &vector_type = result.GetType();
	auto internal_type = vector_type.InternalType();
	auto type_size = GetTypeIdSize(internal_type);
	auto &vdata = GetVectorData(vector_index);

	auto base_ptr = allocator->GetDataPointer(state, vdata.block_id, vdata.offset);
	auto validity_data = GetValidityPointer(base_ptr, type_size);
	if (!vdata.next_data.IsValid() && state.properties != ColumnDataScanProperties::DISALLOW_ZERO_COPY) {
		// no next data, we can do a zero-copy read of this vector
		FlatVector::SetData(result, base_ptr);
		FlatVector::Validity(result).Initialize(validity_data);
		return vdata.count;
	}

	// the data for this vector is spread over multiple vector data entries
	// we need to copy over the data for each of the vectors
	// first figure out how many rows we need to copy by looping over all of the child vector indexes
	idx_t vector_count = 0;
	auto next_index = vector_index;
	while (next_index.IsValid()) {
		auto &current_vdata = GetVectorData(next_index);
		vector_count += current_vdata.count;
		next_index = current_vdata.next_data;
	}
	// resize the result vector
	result.Resize(0, vector_count);
	next_index = vector_index;
	// now perform the copy of each of the vectors
	auto target_data = FlatVector::GetData(result);
	auto &target_validity = FlatVector::Validity(result);
	idx_t current_offset = 0;
	while (next_index.IsValid()) {
		auto &current_vdata = GetVectorData(next_index);
		base_ptr = allocator->GetDataPointer(state, current_vdata.block_id, current_vdata.offset);
		validity_data = GetValidityPointer(base_ptr, type_size);
		if (type_size > 0) {
			memcpy(target_data + current_offset * type_size, base_ptr, current_vdata.count * type_size);
		}
		// FIXME: use bitwise operations here
		ValidityMask current_validity(validity_data);
		for (idx_t k = 0; k < current_vdata.count; k++) {
			target_validity.Set(current_offset + k, current_validity.RowIsValid(k));
		}
		current_offset += current_vdata.count;
		next_index = current_vdata.next_data;
	}
	return vector_count;
}

idx_t ColumnDataCollectionSegment::ReadVector(ChunkManagementState &state, VectorDataIndex vector_index,
                                              Vector &result) {
	auto &vector_type = result.GetType();
	auto internal_type = vector_type.InternalType();
	auto &vdata = GetVectorData(vector_index);
	if (vdata.count == 0) {
		return 0;
	}
	auto count = ReadVectorInternal(state, vector_index, result);
	if (internal_type == PhysicalType::LIST) {
		// list: copy child
		auto &child_vector = ListVector::GetEntry(result);
		auto child_count = ReadVector(state, GetChildIndex(vdata.child_index), child_vector);
		ListVector::SetListSize(result, child_count);
	} else if (internal_type == PhysicalType::STRUCT) {
		auto &child_vectors = StructVector::GetEntries(result);
		for (idx_t child_idx = 0; child_idx < child_vectors.size(); child_idx++) {
			auto child_count =
			    ReadVector(state, GetChildIndex(vdata.child_index, child_idx), *child_vectors[child_idx]);
			if (child_count != count) {
				throw InternalException("Column Data Collection: mismatch in struct child sizes");
			}
		}
	}
	return count;
}

void ColumnDataCollectionSegment::ReadChunk(idx_t chunk_index, ChunkManagementState &state, DataChunk &chunk,
                                            const vector<column_t> &column_ids) {
	D_ASSERT(chunk.ColumnCount() == column_ids.size());
	D_ASSERT(state.properties != ColumnDataScanProperties::INVALID);
	InitializeChunkState(chunk_index, state);
	auto &chunk_meta = chunk_data[chunk_index];
	for (idx_t i = 0; i < column_ids.size(); i++) {
		auto vector_idx = column_ids[i];
		D_ASSERT(vector_idx < chunk_meta.vector_data.size());
		ReadVector(state, chunk_meta.vector_data[vector_idx], chunk.data[i]);
	}
	chunk.SetCardinality(chunk_meta.count);
}

idx_t ColumnDataCollectionSegment::ChunkCount() const {
	return chunk_data.size();
}

void ColumnDataCollectionSegment::FetchChunk(idx_t chunk_idx, DataChunk &result) {
	vector<column_t> column_ids;
	column_ids.reserve(types.size());
	for (idx_t i = 0; i < types.size(); i++) {
		column_ids.push_back(i);
	}
	FetchChunk(chunk_idx, result, column_ids);
}

void ColumnDataCollectionSegment::FetchChunk(idx_t chunk_idx, DataChunk &result, const vector<column_t> &column_ids) {
	D_ASSERT(chunk_idx < chunk_data.size());
	ChunkManagementState state;
	state.properties = ColumnDataScanProperties::DISALLOW_ZERO_COPY;
	ReadChunk(chunk_idx, state, result, column_ids);
}

void ColumnDataCollectionSegment::Verify() {
#ifdef DEBUG
	idx_t total_count = 0;
	for (idx_t i = 0; i < chunk_data.size(); i++) {
		total_count += chunk_data[i].count;
	}
	D_ASSERT(total_count == this->count);
#endif
}

} // namespace duckdb


#include <algorithm>

namespace duckdb {

using ChunkReference = ColumnDataConsumer::ChunkReference;

ChunkReference::ChunkReference(ColumnDataCollectionSegment *segment_p, uint32_t chunk_index_p)
    : segment(segment_p), chunk_index_in_segment(chunk_index_p) {
}

uint32_t ChunkReference::GetMinimumBlockID() const {
	const auto &block_ids = segment->chunk_data[chunk_index_in_segment].block_ids;
	return *std::min_element(block_ids.begin(), block_ids.end());
}

ColumnDataConsumer::ColumnDataConsumer(ColumnDataCollection &collection_p, vector<column_t> column_ids)
    : collection(collection_p), column_ids(move(column_ids)) {
}

void ColumnDataConsumer::InitializeScan() {
	chunk_count = collection.ChunkCount();
	current_chunk_index = 0;
	chunk_delete_index = DConstants::INVALID_INDEX;

	// Initialize chunk references and sort them, so we can scan them in a sane order, regardless of how it was created
	chunk_references.reserve(chunk_count);
	for (auto &segment : collection.GetSegments()) {
		for (idx_t chunk_index = 0; chunk_index < segment->chunk_data.size(); chunk_index++) {
			chunk_references.emplace_back(segment.get(), chunk_index);
		}
	}
	std::sort(chunk_references.begin(), chunk_references.end());
}

bool ColumnDataConsumer::AssignChunk(ColumnDataConsumerScanState &state) {
	lock_guard<mutex> guard(lock);
	if (current_chunk_index == chunk_count) {
		// All chunks have been assigned
		state.current_chunk_state.handles.clear();
		state.chunk_index = DConstants::INVALID_INDEX;
		return false;
	}
	// Assign chunk index
	state.chunk_index = current_chunk_index++;
	D_ASSERT(chunks_in_progress.find(state.chunk_index) == chunks_in_progress.end());
	chunks_in_progress.insert(state.chunk_index);
	return true;
}

void ColumnDataConsumer::ScanChunk(ColumnDataConsumerScanState &state, DataChunk &chunk) const {
	D_ASSERT(state.chunk_index < chunk_count);
	auto &chunk_ref = chunk_references[state.chunk_index];
	if (state.allocator != chunk_ref.segment->allocator.get()) {
		// Previously scanned a chunk from a different allocator, reset the handles
		state.allocator = chunk_ref.segment->allocator.get();
		state.current_chunk_state.handles.clear();
	}
	chunk_ref.segment->ReadChunk(chunk_ref.chunk_index_in_segment, state.current_chunk_state, chunk, column_ids);
}

void ColumnDataConsumer::FinishChunk(ColumnDataConsumerScanState &state) {
	D_ASSERT(state.chunk_index < chunk_count);
	idx_t delete_index_start;
	idx_t delete_index_end;
	{
		lock_guard<mutex> guard(lock);
		D_ASSERT(chunks_in_progress.find(state.chunk_index) != chunks_in_progress.end());
		delete_index_start = chunk_delete_index;
		delete_index_end = *std::min_element(chunks_in_progress.begin(), chunks_in_progress.end());
		chunks_in_progress.erase(state.chunk_index);
		chunk_delete_index = delete_index_end;
	}
	ConsumeChunks(delete_index_start, delete_index_end);
}
void ColumnDataConsumer::ConsumeChunks(idx_t delete_index_start, idx_t delete_index_end) {
	for (idx_t chunk_index = delete_index_start; chunk_index < delete_index_end; chunk_index++) {
		if (chunk_index == 0) {
			continue;
		}
		auto &prev_chunk_ref = chunk_references[chunk_index - 1];
		auto &curr_chunk_ref = chunk_references[chunk_index];
		auto prev_allocator = prev_chunk_ref.segment->allocator.get();
		auto curr_allocator = curr_chunk_ref.segment->allocator.get();
		auto prev_min_block_id = prev_chunk_ref.GetMinimumBlockID();
		auto curr_min_block_id = curr_chunk_ref.GetMinimumBlockID();
		if (prev_allocator != curr_allocator) {
			// Moved to the next allocator, delete all remaining blocks in the previous one
			for (uint32_t block_id = prev_min_block_id; block_id < prev_allocator->BlockCount(); block_id++) {
				prev_allocator->DeleteBlock(block_id);
			}
			continue;
		}
		// Same allocator, see if we can delete blocks
		for (uint32_t block_id = prev_min_block_id; block_id < curr_min_block_id; block_id++) {
			prev_allocator->DeleteBlock(block_id);
		}
	}
}

} // namespace duckdb






















namespace duckdb {

DataChunk::DataChunk() : count(0), capacity(STANDARD_VECTOR_SIZE) {
}

DataChunk::~DataChunk() {
}

void DataChunk::InitializeEmpty(const vector<LogicalType> &types) {
	InitializeEmpty(types.begin(), types.end());
}

void DataChunk::Initialize(Allocator &allocator, const vector<LogicalType> &types, idx_t capacity_p) {
	Initialize(allocator, types.begin(), types.end(), capacity_p);
}

void DataChunk::Initialize(ClientContext &context, const vector<LogicalType> &types, idx_t capacity_p) {
	Initialize(Allocator::Get(context), types, capacity_p);
}

void DataChunk::Initialize(Allocator &allocator, vector<LogicalType>::const_iterator begin,
                           vector<LogicalType>::const_iterator end, idx_t capacity_p) {
	D_ASSERT(data.empty());                   // can only be initialized once
	D_ASSERT(std::distance(begin, end) != 0); // empty chunk not allowed
	capacity = capacity_p;
	for (; begin != end; begin++) {
		VectorCache cache(allocator, *begin, capacity);
		data.emplace_back(cache);
		vector_caches.push_back(move(cache));
	}
}

void DataChunk::Initialize(ClientContext &context, vector<LogicalType>::const_iterator begin,
                           vector<LogicalType>::const_iterator end, idx_t capacity_p) {
	Initialize(Allocator::Get(context), begin, end, capacity_p);
}

void DataChunk::InitializeEmpty(vector<LogicalType>::const_iterator begin, vector<LogicalType>::const_iterator end) {
	capacity = STANDARD_VECTOR_SIZE;
	D_ASSERT(data.empty());                   // can only be initialized once
	D_ASSERT(std::distance(begin, end) != 0); // empty chunk not allowed
	for (; begin != end; begin++) {
		data.emplace_back(Vector(*begin, nullptr));
	}
}

void DataChunk::Reset() {
	if (data.empty()) {
		return;
	}
	if (vector_caches.size() != data.size()) {
		throw InternalException("VectorCache and column count mismatch in DataChunk::Reset");
	}
	for (idx_t i = 0; i < ColumnCount(); i++) {
		data[i].ResetFromCache(vector_caches[i]);
	}
	capacity = STANDARD_VECTOR_SIZE;
	SetCardinality(0);
}

void DataChunk::Destroy() {
	data.clear();
	vector_caches.clear();
	capacity = 0;
	SetCardinality(0);
}

Value DataChunk::GetValue(idx_t col_idx, idx_t index) const {
	D_ASSERT(index < size());
	return data[col_idx].GetValue(index);
}

void DataChunk::SetValue(idx_t col_idx, idx_t index, const Value &val) {
	data[col_idx].SetValue(index, val);
}

bool DataChunk::AllConstant() const {
	for (auto &v : data) {
		if (v.GetVectorType() != VectorType::CONSTANT_VECTOR) {
			return false;
		}
	}
	return true;
}

void DataChunk::Reference(DataChunk &chunk) {
	D_ASSERT(chunk.ColumnCount() <= ColumnCount());
	SetCapacity(chunk);
	SetCardinality(chunk);
	for (idx_t i = 0; i < chunk.ColumnCount(); i++) {
		data[i].Reference(chunk.data[i]);
	}
}

void DataChunk::Move(DataChunk &chunk) {
	SetCardinality(chunk);
	SetCapacity(chunk);
	data = move(chunk.data);
	vector_caches = move(chunk.vector_caches);

	chunk.Destroy();
}

void DataChunk::Copy(DataChunk &other, idx_t offset) const {
	D_ASSERT(ColumnCount() == other.ColumnCount());
	D_ASSERT(other.size() == 0);

	for (idx_t i = 0; i < ColumnCount(); i++) {
		D_ASSERT(other.data[i].GetVectorType() == VectorType::FLAT_VECTOR);
		VectorOperations::Copy(data[i], other.data[i], size(), offset, 0);
	}
	other.SetCardinality(size() - offset);
}

void DataChunk::Copy(DataChunk &other, const SelectionVector &sel, const idx_t source_count, const idx_t offset) const {
	D_ASSERT(ColumnCount() == other.ColumnCount());
	D_ASSERT(other.size() == 0);
	D_ASSERT((offset + source_count) <= size());

	for (idx_t i = 0; i < ColumnCount(); i++) {
		D_ASSERT(other.data[i].GetVectorType() == VectorType::FLAT_VECTOR);
		VectorOperations::Copy(data[i], other.data[i], sel, source_count, offset, 0);
	}
	other.SetCardinality(source_count - offset);
}

void DataChunk::Split(DataChunk &other, idx_t split_idx) {
	D_ASSERT(other.size() == 0);
	D_ASSERT(other.data.empty());
	D_ASSERT(split_idx < data.size());
	const idx_t num_cols = data.size();
	for (idx_t col_idx = split_idx; col_idx < num_cols; col_idx++) {
		other.data.push_back(move(data[col_idx]));
		other.vector_caches.push_back(move(vector_caches[col_idx]));
	}
	for (idx_t col_idx = split_idx; col_idx < num_cols; col_idx++) {
		data.pop_back();
		vector_caches.pop_back();
	}
	other.SetCapacity(*this);
	other.SetCardinality(*this);
}

void DataChunk::Fuse(DataChunk &other) {
	D_ASSERT(other.size() == size());
	const idx_t num_cols = other.data.size();
	for (idx_t col_idx = 0; col_idx < num_cols; ++col_idx) {
		data.emplace_back(move(other.data[col_idx]));
		vector_caches.emplace_back(move(other.vector_caches[col_idx]));
	}
	other.Destroy();
}

void DataChunk::ReferenceColumns(DataChunk &other, const vector<column_t> &column_ids) {
	D_ASSERT(ColumnCount() == column_ids.size());
	Reset();
	for (idx_t col_idx = 0; col_idx < ColumnCount(); col_idx++) {
		auto &other_col = other.data[column_ids[col_idx]];
		auto &this_col = data[col_idx];
		D_ASSERT(other_col.GetType() == this_col.GetType());
		this_col.Reference(other_col);
	}
	SetCardinality(other.size());
}

void DataChunk::Append(const DataChunk &other, bool resize, SelectionVector *sel, idx_t sel_count) {
	idx_t new_size = sel ? size() + sel_count : size() + other.size();
	if (other.size() == 0) {
		return;
	}
	if (ColumnCount() != other.ColumnCount()) {
		throw InternalException("Column counts of appending chunk doesn't match!");
	}
	if (new_size > capacity) {
		if (resize) {
			auto new_capacity = NextPowerOfTwo(new_size);
			for (idx_t i = 0; i < ColumnCount(); i++) {
				data[i].Resize(size(), new_capacity);
			}
			capacity = new_capacity;
		} else {
			throw InternalException("Can't append chunk to other chunk without resizing");
		}
	}
	for (idx_t i = 0; i < ColumnCount(); i++) {
		D_ASSERT(data[i].GetVectorType() == VectorType::FLAT_VECTOR);
		if (sel) {
			VectorOperations::Copy(other.data[i], data[i], *sel, sel_count, 0, size());
		} else {
			VectorOperations::Copy(other.data[i], data[i], other.size(), 0, size());
		}
	}
	SetCardinality(new_size);
}

void DataChunk::Flatten() {
	for (idx_t i = 0; i < ColumnCount(); i++) {
		data[i].Flatten(size());
	}
}

vector<LogicalType> DataChunk::GetTypes() {
	vector<LogicalType> types;
	for (idx_t i = 0; i < ColumnCount(); i++) {
		types.push_back(data[i].GetType());
	}
	return types;
}

string DataChunk::ToString() const {
	string retval = "Chunk - [" + to_string(ColumnCount()) + " Columns]\n";
	for (idx_t i = 0; i < ColumnCount(); i++) {
		retval += "- " + data[i].ToString(size()) + "\n";
	}
	return retval;
}

void DataChunk::Serialize(Serializer &serializer) {
	// write the count
	serializer.Write<sel_t>(size());
	serializer.Write<idx_t>(ColumnCount());
	for (idx_t col_idx = 0; col_idx < ColumnCount(); col_idx++) {
		// write the types
		data[col_idx].GetType().Serialize(serializer);
	}
	// write the data
	for (idx_t col_idx = 0; col_idx < ColumnCount(); col_idx++) {
		data[col_idx].Serialize(size(), serializer);
	}
}

void DataChunk::Deserialize(Deserializer &source) {
	auto rows = source.Read<sel_t>();
	idx_t column_count = source.Read<idx_t>();

	vector<LogicalType> types;
	for (idx_t i = 0; i < column_count; i++) {
		types.push_back(LogicalType::Deserialize(source));
	}
	Initialize(Allocator::DefaultAllocator(), types);
	// now load the column data
	SetCardinality(rows);
	for (idx_t i = 0; i < column_count; i++) {
		data[i].Deserialize(rows, source);
	}
	Verify();
}

void DataChunk::Slice(const SelectionVector &sel_vector, idx_t count_p) {
	this->count = count_p;
	SelCache merge_cache;
	for (idx_t c = 0; c < ColumnCount(); c++) {
		data[c].Slice(sel_vector, count_p, merge_cache);
	}
}

void DataChunk::Slice(DataChunk &other, const SelectionVector &sel, idx_t count_p, idx_t col_offset) {
	D_ASSERT(other.ColumnCount() <= col_offset + ColumnCount());
	this->count = count_p;
	SelCache merge_cache;
	for (idx_t c = 0; c < other.ColumnCount(); c++) {
		if (other.data[c].GetVectorType() == VectorType::DICTIONARY_VECTOR) {
			// already a dictionary! merge the dictionaries
			data[col_offset + c].Reference(other.data[c]);
			data[col_offset + c].Slice(sel, count_p, merge_cache);
		} else {
			data[col_offset + c].Slice(other.data[c], sel, count_p);
		}
	}
}

unique_ptr<UnifiedVectorFormat[]> DataChunk::ToUnifiedFormat() {
	auto orrified_data = unique_ptr<UnifiedVectorFormat[]>(new UnifiedVectorFormat[ColumnCount()]);
	for (idx_t col_idx = 0; col_idx < ColumnCount(); col_idx++) {
		data[col_idx].ToUnifiedFormat(size(), orrified_data[col_idx]);
	}
	return orrified_data;
}

void DataChunk::Hash(Vector &result) {
	D_ASSERT(result.GetType().id() == LogicalType::HASH);
	VectorOperations::Hash(data[0], result, size());
	for (idx_t i = 1; i < ColumnCount(); i++) {
		VectorOperations::CombineHash(result, data[i], size());
	}
}

void DataChunk::Verify() {
#ifdef DEBUG
	D_ASSERT(size() <= capacity);
	// verify that all vectors in this chunk have the chunk selection vector
	for (idx_t i = 0; i < ColumnCount(); i++) {
		data[i].Verify(size());
	}
#endif
}

void DataChunk::Print() {
	Printer::Print(ToString());
}

} // namespace duckdb










#include <cstring>
#include <cctype>
#include <algorithm>

namespace duckdb {

static_assert(sizeof(date_t) == sizeof(int32_t), "date_t was padded");

const char *Date::PINF = "infinity";  // NOLINT
const char *Date::NINF = "-infinity"; // NOLINT
const char *Date::EPOCH = "epoch";    // NOLINT

const string_t Date::MONTH_NAMES_ABBREVIATED[] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};
const string_t Date::MONTH_NAMES[] = {"January", "February", "March",     "April",   "May",      "June",
                                      "July",    "August",   "September", "October", "November", "December"};
const string_t Date::DAY_NAMES[] = {"Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"};
const string_t Date::DAY_NAMES_ABBREVIATED[] = {"Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"};

const int32_t Date::NORMAL_DAYS[] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
const int32_t Date::CUMULATIVE_DAYS[] = {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365};
const int32_t Date::LEAP_DAYS[] = {0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
const int32_t Date::CUMULATIVE_LEAP_DAYS[] = {0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366};
const int8_t Date::MONTH_PER_DAY_OF_YEAR[] = {
    1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
    1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
    2,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
    3,  3,  3,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,
    4,  4,  4,  4,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,
    5,  5,  5,  5,  5,  5,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,
    6,  6,  6,  6,  6,  6,  6,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,
    7,  7,  7,  7,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,
    8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,
    9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
    11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
    12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12};
const int8_t Date::LEAP_MONTH_PER_DAY_OF_YEAR[] = {
    1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
    1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
    2,  2,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
    3,  3,  3,  3,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,
    4,  4,  4,  4,  4,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,
    5,  5,  5,  5,  5,  5,  5,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,
    6,  6,  6,  6,  6,  6,  6,  6,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,
    7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,
    8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,
    9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
    11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
    12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12};
const int32_t Date::CUMULATIVE_YEAR_DAYS[] = {
    0,      365,    730,    1096,   1461,   1826,   2191,   2557,   2922,   3287,   3652,   4018,   4383,   4748,
    5113,   5479,   5844,   6209,   6574,   6940,   7305,   7670,   8035,   8401,   8766,   9131,   9496,   9862,
    10227,  10592,  10957,  11323,  11688,  12053,  12418,  12784,  13149,  13514,  13879,  14245,  14610,  14975,
    15340,  15706,  16071,  16436,  16801,  17167,  17532,  17897,  18262,  18628,  18993,  19358,  19723,  20089,
    20454,  20819,  21184,  21550,  21915,  22280,  22645,  23011,  23376,  23741,  24106,  24472,  24837,  25202,
    25567,  25933,  26298,  26663,  27028,  27394,  27759,  28124,  28489,  28855,  29220,  29585,  29950,  30316,
    30681,  31046,  31411,  31777,  32142,  32507,  32872,  33238,  33603,  33968,  34333,  34699,  35064,  35429,
    35794,  36160,  36525,  36890,  37255,  37621,  37986,  38351,  38716,  39082,  39447,  39812,  40177,  40543,
    40908,  41273,  41638,  42004,  42369,  42734,  43099,  43465,  43830,  44195,  44560,  44926,  45291,  45656,
    46021,  46387,  46752,  47117,  47482,  47847,  48212,  48577,  48942,  49308,  49673,  50038,  50403,  50769,
    51134,  51499,  51864,  52230,  52595,  52960,  53325,  53691,  54056,  54421,  54786,  55152,  55517,  55882,
    56247,  56613,  56978,  57343,  57708,  58074,  58439,  58804,  59169,  59535,  59900,  60265,  60630,  60996,
    61361,  61726,  62091,  62457,  62822,  63187,  63552,  63918,  64283,  64648,  65013,  65379,  65744,  66109,
    66474,  66840,  67205,  67570,  67935,  68301,  68666,  69031,  69396,  69762,  70127,  70492,  70857,  71223,
    71588,  71953,  72318,  72684,  73049,  73414,  73779,  74145,  74510,  74875,  75240,  75606,  75971,  76336,
    76701,  77067,  77432,  77797,  78162,  78528,  78893,  79258,  79623,  79989,  80354,  80719,  81084,  81450,
    81815,  82180,  82545,  82911,  83276,  83641,  84006,  84371,  84736,  85101,  85466,  85832,  86197,  86562,
    86927,  87293,  87658,  88023,  88388,  88754,  89119,  89484,  89849,  90215,  90580,  90945,  91310,  91676,
    92041,  92406,  92771,  93137,  93502,  93867,  94232,  94598,  94963,  95328,  95693,  96059,  96424,  96789,
    97154,  97520,  97885,  98250,  98615,  98981,  99346,  99711,  100076, 100442, 100807, 101172, 101537, 101903,
    102268, 102633, 102998, 103364, 103729, 104094, 104459, 104825, 105190, 105555, 105920, 106286, 106651, 107016,
    107381, 107747, 108112, 108477, 108842, 109208, 109573, 109938, 110303, 110669, 111034, 111399, 111764, 112130,
    112495, 112860, 113225, 113591, 113956, 114321, 114686, 115052, 115417, 115782, 116147, 116513, 116878, 117243,
    117608, 117974, 118339, 118704, 119069, 119435, 119800, 120165, 120530, 120895, 121260, 121625, 121990, 122356,
    122721, 123086, 123451, 123817, 124182, 124547, 124912, 125278, 125643, 126008, 126373, 126739, 127104, 127469,
    127834, 128200, 128565, 128930, 129295, 129661, 130026, 130391, 130756, 131122, 131487, 131852, 132217, 132583,
    132948, 133313, 133678, 134044, 134409, 134774, 135139, 135505, 135870, 136235, 136600, 136966, 137331, 137696,
    138061, 138427, 138792, 139157, 139522, 139888, 140253, 140618, 140983, 141349, 141714, 142079, 142444, 142810,
    143175, 143540, 143905, 144271, 144636, 145001, 145366, 145732, 146097};

void Date::ExtractYearOffset(int32_t &n, int32_t &year, int32_t &year_offset) {
	year = Date::EPOCH_YEAR;
	// first we normalize n to be in the year range [1970, 2370]
	// since leap years repeat every 400 years, we can safely normalize just by "shifting" the CumulativeYearDays array
	while (n < 0) {
		n += Date::DAYS_PER_YEAR_INTERVAL;
		year -= Date::YEAR_INTERVAL;
	}
	while (n >= Date::DAYS_PER_YEAR_INTERVAL) {
		n -= Date::DAYS_PER_YEAR_INTERVAL;
		year += Date::YEAR_INTERVAL;
	}
	// interpolation search
	// we can find an upper bound of the year by assuming each year has 365 days
	year_offset = n / 365;
	// because of leap years we might be off by a little bit: compensate by decrementing the year offset until we find
	// our year
	while (n < Date::CUMULATIVE_YEAR_DAYS[year_offset]) {
		year_offset--;
		D_ASSERT(year_offset >= 0);
	}
	year += year_offset;
	D_ASSERT(n >= Date::CUMULATIVE_YEAR_DAYS[year_offset]);
}

void Date::Convert(date_t d, int32_t &year, int32_t &month, int32_t &day) {
	auto n = d.days;
	int32_t year_offset;
	Date::ExtractYearOffset(n, year, year_offset);

	day = n - Date::CUMULATIVE_YEAR_DAYS[year_offset];
	D_ASSERT(day >= 0 && day <= 365);

	bool is_leap_year = (Date::CUMULATIVE_YEAR_DAYS[year_offset + 1] - Date::CUMULATIVE_YEAR_DAYS[year_offset]) == 366;
	if (is_leap_year) {
		month = Date::LEAP_MONTH_PER_DAY_OF_YEAR[day];
		day -= Date::CUMULATIVE_LEAP_DAYS[month - 1];
	} else {
		month = Date::MONTH_PER_DAY_OF_YEAR[day];
		day -= Date::CUMULATIVE_DAYS[month - 1];
	}
	day++;
	D_ASSERT(day > 0 && day <= (is_leap_year ? Date::LEAP_DAYS[month] : Date::NORMAL_DAYS[month]));
	D_ASSERT(month > 0 && month <= 12);
}

bool Date::TryFromDate(int32_t year, int32_t month, int32_t day, date_t &result) {
	int32_t n = 0;
	if (!Date::IsValid(year, month, day)) {
		return false;
	}
	n += Date::IsLeapYear(year) ? Date::CUMULATIVE_LEAP_DAYS[month - 1] : Date::CUMULATIVE_DAYS[month - 1];
	n += day - 1;
	if (year < 1970) {
		int32_t diff_from_base = 1970 - year;
		int32_t year_index = 400 - (diff_from_base % 400);
		int32_t fractions = diff_from_base / 400;
		n += Date::CUMULATIVE_YEAR_DAYS[year_index];
		n -= Date::DAYS_PER_YEAR_INTERVAL;
		n -= fractions * Date::DAYS_PER_YEAR_INTERVAL;
	} else if (year >= 2370) {
		int32_t diff_from_base = year - 2370;
		int32_t year_index = diff_from_base % 400;
		int32_t fractions = diff_from_base / 400;
		n += Date::CUMULATIVE_YEAR_DAYS[year_index];
		n += Date::DAYS_PER_YEAR_INTERVAL;
		n += fractions * Date::DAYS_PER_YEAR_INTERVAL;
	} else {
		n += Date::CUMULATIVE_YEAR_DAYS[year - 1970];
	}
#ifdef DEBUG
	int32_t y, m, d;
	Date::Convert(date_t(n), y, m, d);
	D_ASSERT(year == y);
	D_ASSERT(month == m);
	D_ASSERT(day == d);
#endif
	result = date_t(n);
	return true;
}

date_t Date::FromDate(int32_t year, int32_t month, int32_t day) {
	date_t result;
	if (!Date::TryFromDate(year, month, day, result)) {
		throw ConversionException("Date out of range: %d-%d-%d", year, month, day);
	}
	return result;
}

bool Date::ParseDoubleDigit(const char *buf, idx_t len, idx_t &pos, int32_t &result) {
	if (pos < len && StringUtil::CharacterIsDigit(buf[pos])) {
		result = buf[pos++] - '0';
		if (pos < len && StringUtil::CharacterIsDigit(buf[pos])) {
			result = (buf[pos++] - '0') + result * 10;
		}
		return true;
	}
	return false;
}

static bool TryConvertDateSpecial(const char *buf, idx_t len, idx_t &pos, const char *special) {
	auto p = pos;
	for (; p < len && *special; ++p) {
		const auto s = *special++;
		if (!s || StringUtil::CharacterToLower(buf[p]) != s) {
			return false;
		}
	}
	if (*special) {
		return false;
	}
	pos = p;
	return true;
}

bool Date::TryConvertDate(const char *buf, idx_t len, idx_t &pos, date_t &result, bool &special, bool strict) {
	special = false;
	pos = 0;
	if (len == 0) {
		return false;
	}

	int32_t day = 0;
	int32_t month = -1;
	int32_t year = 0;
	bool yearneg = false;
	int sep;

	// skip leading spaces
	while (pos < len && StringUtil::CharacterIsSpace(buf[pos])) {
		pos++;
	}

	if (pos >= len) {
		return false;
	}
	if (buf[pos] == '-') {
		yearneg = true;
		pos++;
		if (pos >= len) {
			return false;
		}
	}
	if (!StringUtil::CharacterIsDigit(buf[pos])) {
		// Check for special values
		if (TryConvertDateSpecial(buf, len, pos, PINF)) {
			result = yearneg ? date_t::ninfinity() : date_t::infinity();
		} else if (TryConvertDateSpecial(buf, len, pos, EPOCH)) {
			result = date_t::epoch();
		} else {
			return false;
		}
		// skip trailing spaces - parsing must be strict here
		while (pos < len && StringUtil::CharacterIsSpace(buf[pos])) {
			pos++;
		}
		special = true;
		return pos == len;
	}
	// first parse the year
	for (; pos < len && StringUtil::CharacterIsDigit(buf[pos]); pos++) {
		if (year >= 100000000) {
			return false;
		}
		year = (buf[pos] - '0') + year * 10;
	}
	if (yearneg) {
		year = -year;
	}

	if (pos >= len) {
		return false;
	}

	// fetch the separator
	sep = buf[pos++];
	if (sep != ' ' && sep != '-' && sep != '/' && sep != '\\') {
		// invalid separator
		return false;
	}

	// parse the month
	if (!Date::ParseDoubleDigit(buf, len, pos, month)) {
		return false;
	}

	if (pos >= len) {
		return false;
	}

	if (buf[pos++] != sep) {
		return false;
	}

	if (pos >= len) {
		return false;
	}

	// now parse the day
	if (!Date::ParseDoubleDigit(buf, len, pos, day)) {
		return false;
	}

	// check for an optional trailing " (BC)""
	if (len - pos >= 5 && StringUtil::CharacterIsSpace(buf[pos]) && buf[pos + 1] == '(' &&
	    StringUtil::CharacterToLower(buf[pos + 2]) == 'b' && StringUtil::CharacterToLower(buf[pos + 3]) == 'c' &&
	    buf[pos + 4] == ')') {
		if (yearneg || year == 0) {
			return false;
		}
		year = -year + 1;
		pos += 5;
	}

	// in strict mode, check remaining string for non-space characters
	if (strict) {
		// skip trailing spaces
		while (pos < len && StringUtil::CharacterIsSpace((unsigned char)buf[pos])) {
			pos++;
		}
		// check position. if end was not reached, non-space chars remaining
		if (pos < len) {
			return false;
		}
	} else {
		// in non-strict mode, check for any direct trailing digits
		if (pos < len && StringUtil::CharacterIsDigit((unsigned char)buf[pos])) {
			return false;
		}
	}

	return Date::TryFromDate(year, month, day, result);
}

string Date::ConversionError(const string &str) {
	return StringUtil::Format("date field value out of range: \"%s\", "
	                          "expected format is (YYYY-MM-DD)",
	                          str);
}

string Date::ConversionError(string_t str) {
	return ConversionError(str.GetString());
}

date_t Date::FromCString(const char *buf, idx_t len, bool strict) {
	date_t result;
	idx_t pos;
	bool special = false;
	if (!TryConvertDate(buf, len, pos, result, special, strict)) {
		throw ConversionException(ConversionError(string(buf, len)));
	}
	return result;
}

date_t Date::FromString(const string &str, bool strict) {
	return Date::FromCString(str.c_str(), str.size(), strict);
}

string Date::ToString(date_t date) {
	// PG displays temporal infinities in lowercase,
	// but numerics in Titlecase.
	if (date == date_t::infinity()) {
		return PINF;
	} else if (date == date_t::ninfinity()) {
		return NINF;
	}
	int32_t date_units[3];
	idx_t year_length;
	bool add_bc;
	Date::Convert(date, date_units[0], date_units[1], date_units[2]);

	auto length = DateToStringCast::Length(date_units, year_length, add_bc);
	auto buffer = unique_ptr<char[]>(new char[length]);
	DateToStringCast::Format(buffer.get(), date_units, year_length, add_bc);
	return string(buffer.get(), length);
}

string Date::Format(int32_t year, int32_t month, int32_t day) {
	return ToString(Date::FromDate(year, month, day));
}

bool Date::IsLeapYear(int32_t year) {
	return year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
}

bool Date::IsValid(int32_t year, int32_t month, int32_t day) {
	if (month < 1 || month > 12) {
		return false;
	}
	if (day < 1) {
		return false;
	}
	if (year <= DATE_MIN_YEAR) {
		if (year < DATE_MIN_YEAR) {
			return false;
		} else if (year == DATE_MIN_YEAR) {
			if (month < DATE_MIN_MONTH || (month == DATE_MIN_MONTH && day < DATE_MIN_DAY)) {
				return false;
			}
		}
	}
	if (year >= DATE_MAX_YEAR) {
		if (year > DATE_MAX_YEAR) {
			return false;
		} else if (year == DATE_MAX_YEAR) {
			if (month > DATE_MAX_MONTH || (month == DATE_MAX_MONTH && day > DATE_MAX_DAY)) {
				return false;
			}
		}
	}
	return Date::IsLeapYear(year) ? day <= Date::LEAP_DAYS[month] : day <= Date::NORMAL_DAYS[month];
}

int32_t Date::MonthDays(int32_t year, int32_t month) {
	D_ASSERT(month >= 1 && month <= 12);
	return Date::IsLeapYear(year) ? Date::LEAP_DAYS[month] : Date::NORMAL_DAYS[month];
}

date_t Date::EpochDaysToDate(int32_t epoch) {
	return (date_t)epoch;
}

int32_t Date::EpochDays(date_t date) {
	return date.days;
}

date_t Date::EpochToDate(int64_t epoch) {
	return date_t(epoch / Interval::SECS_PER_DAY);
}

int64_t Date::Epoch(date_t date) {
	return ((int64_t)date.days) * Interval::SECS_PER_DAY;
}

int64_t Date::EpochNanoseconds(date_t date) {
	return ((int64_t)date.days) * (Interval::MICROS_PER_DAY * 1000);
}

int64_t Date::EpochMicroseconds(date_t date) {
	int64_t result;
	if (!TryMultiplyOperator::Operation<int64_t, int64_t, int64_t>(date.days, Interval::MICROS_PER_DAY, result)) {
		throw ConversionException("Could not convert DATE to microseconds");
	}
	return result;
}

int32_t Date::ExtractYear(date_t d, int32_t *last_year) {
	auto n = d.days;
	// cached look up: check if year of this date is the same as the last one we looked up
	// note that this only works for years in the range [1970, 2370]
	if (n >= Date::CUMULATIVE_YEAR_DAYS[*last_year] && n < Date::CUMULATIVE_YEAR_DAYS[*last_year + 1]) {
		return Date::EPOCH_YEAR + *last_year;
	}
	int32_t year;
	Date::ExtractYearOffset(n, year, *last_year);
	return year;
}

int32_t Date::ExtractYear(timestamp_t ts, int32_t *last_year) {
	return Date::ExtractYear(Timestamp::GetDate(ts), last_year);
}

int32_t Date::ExtractYear(date_t d) {
	int32_t year, year_offset;
	Date::ExtractYearOffset(d.days, year, year_offset);
	return year;
}

int32_t Date::ExtractMonth(date_t date) {
	int32_t out_year, out_month, out_day;
	Date::Convert(date, out_year, out_month, out_day);
	return out_month;
}

int32_t Date::ExtractDay(date_t date) {
	int32_t out_year, out_month, out_day;
	Date::Convert(date, out_year, out_month, out_day);
	return out_day;
}

int32_t Date::ExtractDayOfTheYear(date_t date) {
	int32_t year, year_offset;
	Date::ExtractYearOffset(date.days, year, year_offset);
	return date.days - Date::CUMULATIVE_YEAR_DAYS[year_offset] + 1;
}

int32_t Date::ExtractISODayOfTheWeek(date_t date) {
	// date of 0 is 1970-01-01, which was a Thursday (4)
	// -7 = 4
	// -6 = 5
	// -5 = 6
	// -4 = 7
	// -3 = 1
	// -2 = 2
	// -1 = 3
	// 0  = 4
	// 1  = 5
	// 2  = 6
	// 3  = 7
	// 4  = 1
	// 5  = 2
	// 6  = 3
	// 7  = 4
	if (date.days < 0) {
		// negative date: start off at 4 and cycle downwards
		return (7 - ((-int64_t(date.days) + 3) % 7));
	} else {
		// positive date: start off at 4 and cycle upwards
		return ((int64_t(date.days) + 3) % 7) + 1;
	}
}

static int32_t GetISOYearWeek(int32_t &year, int32_t month, int32_t day) {
	auto day_of_the_year =
	    (Date::IsLeapYear(year) ? Date::CUMULATIVE_LEAP_DAYS[month] : Date::CUMULATIVE_DAYS[month]) + day;
	// get the first day of the first week of the year
	// the first week is the week that has the 4th of January in it
	const auto weekday_of_the_fourth = Date::ExtractISODayOfTheWeek(Date::FromDate(year, 1, 4));
	// if fourth is monday, then fourth is the first day
	// if fourth is tuesday, third is the first day
	// if fourth is wednesday, second is the first day
	// if fourth is thursday, first is the first day
	// if fourth is friday - sunday, day is in the previous year
	// (day is 0-based, weekday is 1-based)
	const auto first_day_of_the_first_isoweek = 4 - weekday_of_the_fourth;
	if (day_of_the_year < first_day_of_the_first_isoweek) {
		// day is part of last year (13th month)
		--year;
		return GetISOYearWeek(year, 12, day);
	} else {
		return ((day_of_the_year - first_day_of_the_first_isoweek) / 7) + 1;
	}
}

void Date::ExtractISOYearWeek(date_t date, int32_t &year, int32_t &week) {
	int32_t month, day;
	Date::Convert(date, year, month, day);
	week = GetISOYearWeek(year, month - 1, day - 1);
}

int32_t Date::ExtractISOWeekNumber(date_t date) {
	int32_t year, week;
	ExtractISOYearWeek(date, year, week);
	return week;
}

int32_t Date::ExtractISOYearNumber(date_t date) {
	int32_t year, week;
	ExtractISOYearWeek(date, year, week);
	return year;
}

int32_t Date::ExtractWeekNumberRegular(date_t date, bool monday_first) {
	int32_t year, month, day;
	Date::Convert(date, year, month, day);
	month -= 1;
	day -= 1;
	// get the day of the year
	auto day_of_the_year =
	    (Date::IsLeapYear(year) ? Date::CUMULATIVE_LEAP_DAYS[month] : Date::CUMULATIVE_DAYS[month]) + day;
	// now figure out the first monday or sunday of the year
	// what day is January 1st?
	auto day_of_jan_first = Date::ExtractISODayOfTheWeek(Date::FromDate(year, 1, 1));
	// monday = 1, sunday = 7
	int32_t first_week_start;
	if (monday_first) {
		// have to find next "1"
		if (day_of_jan_first == 1) {
			// jan 1 is monday: starts immediately
			first_week_start = 0;
		} else {
			// jan 1 is not monday: count days until next monday
			first_week_start = 8 - day_of_jan_first;
		}
	} else {
		first_week_start = 7 - day_of_jan_first;
	}
	if (day_of_the_year < first_week_start) {
		// day occurs before first week starts: week 0
		return 0;
	}
	return ((day_of_the_year - first_week_start) / 7) + 1;
}

// Returns the date of the monday of the current week.
date_t Date::GetMondayOfCurrentWeek(date_t date) {
	int32_t dotw = Date::ExtractISODayOfTheWeek(date);
	return date - (dotw - 1);
}

} // namespace duckdb



namespace duckdb {

template <class SIGNED, class UNSIGNED>
string TemplatedDecimalToString(SIGNED value, uint8_t width, uint8_t scale) {
	auto len = DecimalToString::DecimalLength<SIGNED, UNSIGNED>(value, width, scale);
	auto data = unique_ptr<char[]>(new char[len + 1]);
	DecimalToString::FormatDecimal<SIGNED, UNSIGNED>(value, width, scale, data.get(), len);
	return string(data.get(), len);
}

string Decimal::ToString(int16_t value, uint8_t width, uint8_t scale) {
	return TemplatedDecimalToString<int16_t, uint16_t>(value, width, scale);
}

string Decimal::ToString(int32_t value, uint8_t width, uint8_t scale) {
	return TemplatedDecimalToString<int32_t, uint32_t>(value, width, scale);
}

string Decimal::ToString(int64_t value, uint8_t width, uint8_t scale) {
	return TemplatedDecimalToString<int64_t, uint64_t>(value, width, scale);
}

string Decimal::ToString(hugeint_t value, uint8_t width, uint8_t scale) {
	auto len = HugeintToStringCast::DecimalLength(value, width, scale);
	auto data = unique_ptr<char[]>(new char[len + 1]);
	HugeintToStringCast::FormatDecimal(value, width, scale, data.get(), len);
	return string(data.get(), len);
}

} // namespace duckdb






#include <functional>
#include <cmath>

namespace duckdb {

template <>
hash_t Hash(uint64_t val) {
	return murmurhash64(val);
}

template <>
hash_t Hash(int64_t val) {
	return murmurhash64((uint64_t)val);
}

template <>
hash_t Hash(hugeint_t val) {
	return murmurhash64(val.lower) ^ murmurhash64(val.upper);
}

template <class T>
struct FloatingPointEqualityTransform {
	static void OP(T &val) {
		if (val == (T)0.0) {
			// Turn negative zero into positive zero
			val = (T)0.0;
		} else if (std::isnan(val)) {
			val = std::numeric_limits<T>::quiet_NaN();
		}
	}
};

template <>
hash_t Hash(float val) {
	static_assert(sizeof(float) == sizeof(uint32_t), "");
	FloatingPointEqualityTransform<float>::OP(val);
	uint32_t uval = *((uint32_t *)&val);
	return murmurhash64(uval);
}

template <>
hash_t Hash(double val) {
	static_assert(sizeof(double) == sizeof(uint64_t), "");
	FloatingPointEqualityTransform<double>::OP(val);
	uint64_t uval = *((uint64_t *)&val);
	return murmurhash64(uval);
}

template <>
hash_t Hash(interval_t val) {
	return Hash(val.days) ^ Hash(val.months) ^ Hash(val.micros);
}

template <>
hash_t Hash(const char *str) {
	return Hash(str, strlen(str));
}

template <>
hash_t Hash(string_t val) {
	return Hash(val.GetDataUnsafe(), val.GetSize());
}

template <>
hash_t Hash(char *val) {
	return Hash<const char *>(val);
}

// MIT License
// Copyright (c) 2018-2021 Martin Ankerl
// https://github.com/martinus/robin-hood-hashing/blob/3.11.5/LICENSE
hash_t HashBytes(void *ptr, size_t len) noexcept {
	static constexpr uint64_t M = UINT64_C(0xc6a4a7935bd1e995);
	static constexpr uint64_t SEED = UINT64_C(0xe17a1465);
	static constexpr unsigned int R = 47;

	auto const *const data64 = static_cast<uint64_t const *>(ptr);
	uint64_t h = SEED ^ (len * M);

	size_t const n_blocks = len / 8;
	for (size_t i = 0; i < n_blocks; ++i) {
		auto k = Load<uint64_t>(reinterpret_cast<const_data_ptr_t>(data64 + i));

		k *= M;
		k ^= k >> R;
		k *= M;

		h ^= k;
		h *= M;
	}

	auto const *const data8 = reinterpret_cast<uint8_t const *>(data64 + n_blocks);
	switch (len & 7U) {
	case 7:
		h ^= static_cast<uint64_t>(data8[6]) << 48U;
		DUCKDB_EXPLICIT_FALLTHROUGH;
	case 6:
		h ^= static_cast<uint64_t>(data8[5]) << 40U;
		DUCKDB_EXPLICIT_FALLTHROUGH;
	case 5:
		h ^= static_cast<uint64_t>(data8[4]) << 32U;
		DUCKDB_EXPLICIT_FALLTHROUGH;
	case 4:
		h ^= static_cast<uint64_t>(data8[3]) << 24U;
		DUCKDB_EXPLICIT_FALLTHROUGH;
	case 3:
		h ^= static_cast<uint64_t>(data8[2]) << 16U;
		DUCKDB_EXPLICIT_FALLTHROUGH;
	case 2:
		h ^= static_cast<uint64_t>(data8[1]) << 8U;
		DUCKDB_EXPLICIT_FALLTHROUGH;
	case 1:
		h ^= static_cast<uint64_t>(data8[0]);
		h *= M;
		DUCKDB_EXPLICIT_FALLTHROUGH;
	default:
		break;
	}
	h ^= h >> R;
	h *= M;
	h ^= h >> R;
	return static_cast<hash_t>(h);
}

hash_t Hash(const char *val, size_t size) {
	return HashBytes((void *)val, size);
}

hash_t Hash(uint8_t *val, size_t size) {
	return HashBytes((void *)val, size);
}

} // namespace duckdb








#include <cmath>
#include <limits>

namespace duckdb {

//===--------------------------------------------------------------------===//
// String Conversion
//===--------------------------------------------------------------------===//
const hugeint_t Hugeint::POWERS_OF_TEN[] {
    hugeint_t(1),
    hugeint_t(10),
    hugeint_t(100),
    hugeint_t(1000),
    hugeint_t(10000),
    hugeint_t(100000),
    hugeint_t(1000000),
    hugeint_t(10000000),
    hugeint_t(100000000),
    hugeint_t(1000000000),
    hugeint_t(10000000000),
    hugeint_t(100000000000),
    hugeint_t(1000000000000),
    hugeint_t(10000000000000),
    hugeint_t(100000000000000),
    hugeint_t(1000000000000000),
    hugeint_t(10000000000000000),
    hugeint_t(100000000000000000),
    hugeint_t(1000000000000000000),
    hugeint_t(1000000000000000000) * hugeint_t(10),
    hugeint_t(1000000000000000000) * hugeint_t(100),
    hugeint_t(1000000000000000000) * hugeint_t(1000),
    hugeint_t(1000000000000000000) * hugeint_t(10000),
    hugeint_t(1000000000000000000) * hugeint_t(100000),
    hugeint_t(1000000000000000000) * hugeint_t(1000000),
    hugeint_t(1000000000000000000) * hugeint_t(10000000),
    hugeint_t(1000000000000000000) * hugeint_t(100000000),
    hugeint_t(1000000000000000000) * hugeint_t(1000000000),
    hugeint_t(1000000000000000000) * hugeint_t(10000000000),
    hugeint_t(1000000000000000000) * hugeint_t(100000000000),
    hugeint_t(1000000000000000000) * hugeint_t(1000000000000),
    hugeint_t(1000000000000000000) * hugeint_t(10000000000000),
    hugeint_t(1000000000000000000) * hugeint_t(100000000000000),
    hugeint_t(1000000000000000000) * hugeint_t(1000000000000000),
    hugeint_t(1000000000000000000) * hugeint_t(10000000000000000),
    hugeint_t(1000000000000000000) * hugeint_t(100000000000000000),
    hugeint_t(1000000000000000000) * hugeint_t(1000000000000000000),
    hugeint_t(1000000000000000000) * hugeint_t(1000000000000000000) * hugeint_t(10),
    hugeint_t(1000000000000000000) * hugeint_t(1000000000000000000) * hugeint_t(100)};

static uint8_t PositiveHugeintHighestBit(hugeint_t bits) {
	uint8_t out = 0;
	if (bits.upper) {
		out = 64;
		uint64_t up = bits.upper;
		while (up) {
			up >>= 1;
			out++;
		}
	} else {
		uint64_t low = bits.lower;
		while (low) {
			low >>= 1;
			out++;
		}
	}
	return out;
}

static bool PositiveHugeintIsBitSet(hugeint_t lhs, uint8_t bit_position) {
	if (bit_position < 64) {
		return lhs.lower & (uint64_t(1) << uint64_t(bit_position));
	} else {
		return lhs.upper & (uint64_t(1) << uint64_t(bit_position - 64));
	}
}

hugeint_t PositiveHugeintLeftShift(hugeint_t lhs, uint32_t amount) {
	D_ASSERT(amount > 0 && amount < 64);
	hugeint_t result;
	result.lower = lhs.lower << amount;
	result.upper = (lhs.upper << amount) + (lhs.lower >> (64 - amount));
	return result;
}

hugeint_t Hugeint::DivModPositive(hugeint_t lhs, uint64_t rhs, uint64_t &remainder) {
	D_ASSERT(lhs.upper >= 0);
	// DivMod code adapted from:
	// https://github.com/calccrypto/uint128_t/blob/master/uint128_t.cpp

	// initialize the result and remainder to 0
	hugeint_t div_result;
	div_result.lower = 0;
	div_result.upper = 0;
	remainder = 0;

	uint8_t highest_bit_set = PositiveHugeintHighestBit(lhs);
	// now iterate over the amount of bits that are set in the LHS
	for (uint8_t x = highest_bit_set; x > 0; x--) {
		// left-shift the current result and remainder by 1
		div_result = PositiveHugeintLeftShift(div_result, 1);
		remainder <<= 1;
		// we get the value of the bit at position X, where position 0 is the least-significant bit
		if (PositiveHugeintIsBitSet(lhs, x - 1)) {
			// increment the remainder
			remainder++;
		}
		if (remainder >= rhs) {
			// the remainder has passed the division multiplier: add one to the divide result
			remainder -= rhs;
			div_result.lower++;
			if (div_result.lower == 0) {
				// overflow
				div_result.upper++;
			}
		}
	}
	return div_result;
}

string Hugeint::ToString(hugeint_t input) {
	uint64_t remainder;
	string result;
	bool negative = input.upper < 0;
	if (negative) {
		NegateInPlace(input);
	}
	while (true) {
		if (!input.lower && !input.upper) {
			break;
		}
		input = Hugeint::DivModPositive(input, 10, remainder);
		result = string(1, '0' + remainder) + result; // NOLINT
	}
	if (result.empty()) {
		// value is zero
		return "0";
	}
	return negative ? "-" + result : result;
}

//===--------------------------------------------------------------------===//
// Multiply
//===--------------------------------------------------------------------===//
bool Hugeint::TryMultiply(hugeint_t lhs, hugeint_t rhs, hugeint_t &result) {
	bool lhs_negative = lhs.upper < 0;
	bool rhs_negative = rhs.upper < 0;
	if (lhs_negative) {
		NegateInPlace(lhs);
	}
	if (rhs_negative) {
		NegateInPlace(rhs);
	}
#if ((__GNUC__ >= 5) || defined(__clang__)) && defined(__SIZEOF_INT128__)
	__uint128_t left = __uint128_t(lhs.lower) + (__uint128_t(lhs.upper) << 64);
	__uint128_t right = __uint128_t(rhs.lower) + (__uint128_t(rhs.upper) << 64);
	__uint128_t result_i128;
	if (__builtin_mul_overflow(left, right, &result_i128)) {
		return false;
	}
	uint64_t upper = uint64_t(result_i128 >> 64);
	if (upper & 0x8000000000000000) {
		return false;
	}
	result.upper = int64_t(upper);
	result.lower = uint64_t(result_i128 & 0xffffffffffffffff);
#else
	// Multiply code adapted from:
	// https://github.com/calccrypto/uint128_t/blob/master/uint128_t.cpp

	// split values into 4 32-bit parts
	uint64_t top[4] = {uint64_t(lhs.upper) >> 32, uint64_t(lhs.upper) & 0xffffffff, lhs.lower >> 32,
	                   lhs.lower & 0xffffffff};
	uint64_t bottom[4] = {uint64_t(rhs.upper) >> 32, uint64_t(rhs.upper) & 0xffffffff, rhs.lower >> 32,
	                      rhs.lower & 0xffffffff};
	uint64_t products[4][4];

	// multiply each component of the values
	for (auto x = 0; x < 4; x++) {
		for (auto y = 0; y < 4; y++) {
			products[x][y] = top[x] * bottom[y];
		}
	}

	// if any of these products are set to a non-zero value, there is always an overflow
	if (products[0][0] || products[0][1] || products[0][2] || products[1][0] || products[2][0] || products[1][1]) {
		return false;
	}
	// if the high bits of any of these are set, there is always an overflow
	if ((products[0][3] & 0xffffffff80000000) || (products[1][2] & 0xffffffff80000000) ||
	    (products[2][1] & 0xffffffff80000000) || (products[3][0] & 0xffffffff80000000)) {
		return false;
	}

	// otherwise we merge the result of the different products together in-order

	// first row
	uint64_t fourth32 = (products[3][3] & 0xffffffff);
	uint64_t third32 = (products[3][2] & 0xffffffff) + (products[3][3] >> 32);
	uint64_t second32 = (products[3][1] & 0xffffffff) + (products[3][2] >> 32);
	uint64_t first32 = (products[3][0] & 0xffffffff) + (products[3][1] >> 32);

	// second row
	third32 += (products[2][3] & 0xffffffff);
	second32 += (products[2][2] & 0xffffffff) + (products[2][3] >> 32);
	first32 += (products[2][1] & 0xffffffff) + (products[2][2] >> 32);

	// third row
	second32 += (products[1][3] & 0xffffffff);
	first32 += (products[1][2] & 0xffffffff) + (products[1][3] >> 32);

	// fourth row
	first32 += (products[0][3] & 0xffffffff);

	// move carry to next digit
	third32 += fourth32 >> 32;
	second32 += third32 >> 32;
	first32 += second32 >> 32;

	// check if the combination of the different products resulted in an overflow
	if (first32 & 0xffffff80000000) {
		return false;
	}

	// remove carry from current digit
	fourth32 &= 0xffffffff;
	third32 &= 0xffffffff;
	second32 &= 0xffffffff;
	first32 &= 0xffffffff;

	// combine components
	result.lower = (third32 << 32) | fourth32;
	result.upper = (first32 << 32) | second32;
#endif
	if (lhs_negative ^ rhs_negative) {
		NegateInPlace(result);
	}
	return true;
}

hugeint_t Hugeint::Multiply(hugeint_t lhs, hugeint_t rhs) {
	hugeint_t result;
	if (!TryMultiply(lhs, rhs, result)) {
		throw OutOfRangeException("Overflow in HUGEINT multiplication!");
	}
	return result;
}

//===--------------------------------------------------------------------===//
// Divide
//===--------------------------------------------------------------------===//
hugeint_t Hugeint::DivMod(hugeint_t lhs, hugeint_t rhs, hugeint_t &remainder) {
	// division by zero not allowed
	D_ASSERT(!(rhs.upper == 0 && rhs.lower == 0));

	bool lhs_negative = lhs.upper < 0;
	bool rhs_negative = rhs.upper < 0;
	if (lhs_negative) {
		Hugeint::NegateInPlace(lhs);
	}
	if (rhs_negative) {
		Hugeint::NegateInPlace(rhs);
	}
	// DivMod code adapted from:
	// https://github.com/calccrypto/uint128_t/blob/master/uint128_t.cpp

	// initialize the result and remainder to 0
	hugeint_t div_result;
	div_result.lower = 0;
	div_result.upper = 0;
	remainder.lower = 0;
	remainder.upper = 0;

	uint8_t highest_bit_set = PositiveHugeintHighestBit(lhs);
	// now iterate over the amount of bits that are set in the LHS
	for (uint8_t x = highest_bit_set; x > 0; x--) {
		// left-shift the current result and remainder by 1
		div_result = PositiveHugeintLeftShift(div_result, 1);
		remainder = PositiveHugeintLeftShift(remainder, 1);

		// we get the value of the bit at position X, where position 0 is the least-significant bit
		if (PositiveHugeintIsBitSet(lhs, x - 1)) {
			// increment the remainder
			Hugeint::AddInPlace(remainder, 1);
		}
		if (Hugeint::GreaterThanEquals(remainder, rhs)) {
			// the remainder has passed the division multiplier: add one to the divide result
			remainder = Hugeint::Subtract(remainder, rhs);
			Hugeint::AddInPlace(div_result, 1);
		}
	}
	if (lhs_negative ^ rhs_negative) {
		Hugeint::NegateInPlace(div_result);
	}
	if (lhs_negative) {
		Hugeint::NegateInPlace(remainder);
	}
	return div_result;
}

hugeint_t Hugeint::Divide(hugeint_t lhs, hugeint_t rhs) {
	hugeint_t remainder;
	return Hugeint::DivMod(lhs, rhs, remainder);
}

hugeint_t Hugeint::Modulo(hugeint_t lhs, hugeint_t rhs) {
	hugeint_t remainder;
	Hugeint::DivMod(lhs, rhs, remainder);
	return remainder;
}

//===--------------------------------------------------------------------===//
// Add/Subtract
//===--------------------------------------------------------------------===//
bool Hugeint::AddInPlace(hugeint_t &lhs, hugeint_t rhs) {
	int overflow = lhs.lower + rhs.lower < lhs.lower;
	if (rhs.upper >= 0) {
		// RHS is positive: check for overflow
		if (lhs.upper > (std::numeric_limits<int64_t>::max() - rhs.upper - overflow)) {
			return false;
		}
		lhs.upper = lhs.upper + overflow + rhs.upper;
	} else {
		// RHS is negative: check for underflow
		if (lhs.upper < std::numeric_limits<int64_t>::min() - rhs.upper - overflow) {
			return false;
		}
		lhs.upper = lhs.upper + (overflow + rhs.upper);
	}
	lhs.lower += rhs.lower;
	if (lhs.upper == std::numeric_limits<int64_t>::min() && lhs.lower == 0) {
		return false;
	}
	return true;
}

bool Hugeint::SubtractInPlace(hugeint_t &lhs, hugeint_t rhs) {
	// underflow
	int underflow = lhs.lower - rhs.lower > lhs.lower;
	if (rhs.upper >= 0) {
		// RHS is positive: check for underflow
		if (lhs.upper < (std::numeric_limits<int64_t>::min() + rhs.upper + underflow)) {
			return false;
		}
		lhs.upper = (lhs.upper - rhs.upper) - underflow;
	} else {
		// RHS is negative: check for overflow
		if (lhs.upper > std::numeric_limits<int64_t>::min() &&
		    lhs.upper - 1 >= (std::numeric_limits<int64_t>::max() + rhs.upper + underflow)) {
			return false;
		}
		lhs.upper = lhs.upper - (rhs.upper + underflow);
	}
	lhs.lower -= rhs.lower;
	if (lhs.upper == std::numeric_limits<int64_t>::min() && lhs.lower == 0) {
		return false;
	}
	return true;
}

hugeint_t Hugeint::Add(hugeint_t lhs, hugeint_t rhs) {
	if (!AddInPlace(lhs, rhs)) {
		throw OutOfRangeException("Overflow in HUGEINT addition");
	}
	return lhs;
}

hugeint_t Hugeint::Subtract(hugeint_t lhs, hugeint_t rhs) {
	if (!SubtractInPlace(lhs, rhs)) {
		throw OutOfRangeException("Underflow in HUGEINT addition");
	}
	return lhs;
}

//===--------------------------------------------------------------------===//
// Hugeint Cast/Conversion
//===--------------------------------------------------------------------===//
template <class DST, bool SIGNED = true>
bool HugeintTryCastInteger(hugeint_t input, DST &result) {
	switch (input.upper) {
	case 0:
		// positive number: check if the positive number is in range
		if (input.lower <= uint64_t(NumericLimits<DST>::Maximum())) {
			result = DST(input.lower);
			return true;
		}
		break;
	case -1:
		if (!SIGNED) {
			return false;
		}
		// negative number: check if the negative number is in range
		if (input.lower >= NumericLimits<uint64_t>::Maximum() - uint64_t(NumericLimits<DST>::Maximum())) {
			result = -DST(NumericLimits<uint64_t>::Maximum() - input.lower) - 1;
			return true;
		}
		break;
	default:
		break;
	}
	return false;
}

template <>
bool Hugeint::TryCast(hugeint_t input, int8_t &result) {
	return HugeintTryCastInteger<int8_t>(input, result);
}

template <>
bool Hugeint::TryCast(hugeint_t input, int16_t &result) {
	return HugeintTryCastInteger<int16_t>(input, result);
}

template <>
bool Hugeint::TryCast(hugeint_t input, int32_t &result) {
	return HugeintTryCastInteger<int32_t>(input, result);
}

template <>
bool Hugeint::TryCast(hugeint_t input, int64_t &result) {
	return HugeintTryCastInteger<int64_t>(input, result);
}

template <>
bool Hugeint::TryCast(hugeint_t input, uint8_t &result) {
	return HugeintTryCastInteger<uint8_t, false>(input, result);
}

template <>
bool Hugeint::TryCast(hugeint_t input, uint16_t &result) {
	return HugeintTryCastInteger<uint16_t, false>(input, result);
}

template <>
bool Hugeint::TryCast(hugeint_t input, uint32_t &result) {
	return HugeintTryCastInteger<uint32_t, false>(input, result);
}

template <>
bool Hugeint::TryCast(hugeint_t input, uint64_t &result) {
	return HugeintTryCastInteger<uint64_t, false>(input, result);
}

template <>
bool Hugeint::TryCast(hugeint_t input, hugeint_t &result) {
	result = input;
	return true;
}

template <>
bool Hugeint::TryCast(hugeint_t input, float &result) {
	double dbl_result;
	Hugeint::TryCast(input, dbl_result);
	result = (float)dbl_result;
	return true;
}

template <class REAL_T>
bool CastBigintToFloating(hugeint_t input, REAL_T &result) {
	switch (input.upper) {
	case -1:
		// special case for upper = -1 to avoid rounding issues in small negative numbers
		result = -REAL_T(NumericLimits<uint64_t>::Maximum() - input.lower) - 1;
		break;
	default:
		result = REAL_T(input.lower) + REAL_T(input.upper) * REAL_T(NumericLimits<uint64_t>::Maximum());
		break;
	}
	return true;
}

template <>
bool Hugeint::TryCast(hugeint_t input, double &result) {
	return CastBigintToFloating<double>(input, result);
}

template <>
bool Hugeint::TryCast(hugeint_t input, long double &result) {
	return CastBigintToFloating<long double>(input, result);
}

template <class DST>
hugeint_t HugeintConvertInteger(DST input) {
	hugeint_t result;
	result.lower = (uint64_t)input;
	result.upper = (input < 0) * -1;
	return result;
}

template <>
bool Hugeint::TryConvert(int8_t value, hugeint_t &result) {
	result = HugeintConvertInteger<int8_t>(value);
	return true;
}

template <>
bool Hugeint::TryConvert(const char *value, hugeint_t &result) {
	auto len = strlen(value);
	string_t string_val(value, len);
	return TryCast::Operation<string_t, hugeint_t>(string_val, result, true);
}

template <>
bool Hugeint::TryConvert(int16_t value, hugeint_t &result) {
	result = HugeintConvertInteger<int16_t>(value);
	return true;
}

template <>
bool Hugeint::TryConvert(int32_t value, hugeint_t &result) {
	result = HugeintConvertInteger<int32_t>(value);
	return true;
}

template <>
bool Hugeint::TryConvert(int64_t value, hugeint_t &result) {
	result = HugeintConvertInteger<int64_t>(value);
	return true;
}
template <>
bool Hugeint::TryConvert(uint8_t value, hugeint_t &result) {
	result = HugeintConvertInteger<uint8_t>(value);
	return true;
}
template <>
bool Hugeint::TryConvert(uint16_t value, hugeint_t &result) {
	result = HugeintConvertInteger<uint16_t>(value);
	return true;
}
template <>
bool Hugeint::TryConvert(uint32_t value, hugeint_t &result) {
	result = HugeintConvertInteger<uint32_t>(value);
	return true;
}
template <>
bool Hugeint::TryConvert(uint64_t value, hugeint_t &result) {
	result = HugeintConvertInteger<uint64_t>(value);
	return true;
}

template <>
bool Hugeint::TryConvert(hugeint_t value, hugeint_t &result) {
	result = value;
	return true;
}

template <>
bool Hugeint::TryConvert(float value, hugeint_t &result) {
	return Hugeint::TryConvert(double(value), result);
}

template <class REAL_T>
bool ConvertFloatingToBigint(REAL_T value, hugeint_t &result) {
	if (!Value::IsFinite<REAL_T>(value)) {
		return false;
	}
	if (value <= -170141183460469231731687303715884105728.0 || value >= 170141183460469231731687303715884105727.0) {
		return false;
	}
	bool negative = value < 0;
	if (negative) {
		value = -value;
	}
	result.lower = (uint64_t)fmod(value, REAL_T(NumericLimits<uint64_t>::Maximum()));
	result.upper = (uint64_t)(value / REAL_T(NumericLimits<uint64_t>::Maximum()));
	if (negative) {
		Hugeint::NegateInPlace(result);
	}
	return true;
}

template <>
bool Hugeint::TryConvert(double value, hugeint_t &result) {
	return ConvertFloatingToBigint<double>(value, result);
}

template <>
bool Hugeint::TryConvert(long double value, hugeint_t &result) {
	return ConvertFloatingToBigint<long double>(value, result);
}

//===--------------------------------------------------------------------===//
// hugeint_t operators
//===--------------------------------------------------------------------===//
hugeint_t::hugeint_t(int64_t value) {
	auto result = Hugeint::Convert(value);
	this->lower = result.lower;
	this->upper = result.upper;
}

bool hugeint_t::operator==(const hugeint_t &rhs) const {
	return Hugeint::Equals(*this, rhs);
}

bool hugeint_t::operator!=(const hugeint_t &rhs) const {
	return Hugeint::NotEquals(*this, rhs);
}

bool hugeint_t::operator<(const hugeint_t &rhs) const {
	return Hugeint::LessThan(*this, rhs);
}

bool hugeint_t::operator<=(const hugeint_t &rhs) const {
	return Hugeint::LessThanEquals(*this, rhs);
}

bool hugeint_t::operator>(const hugeint_t &rhs) const {
	return Hugeint::GreaterThan(*this, rhs);
}

bool hugeint_t::operator>=(const hugeint_t &rhs) const {
	return Hugeint::GreaterThanEquals(*this, rhs);
}

hugeint_t hugeint_t::operator+(const hugeint_t &rhs) const {
	return Hugeint::Add(*this, rhs);
}

hugeint_t hugeint_t::operator-(const hugeint_t &rhs) const {
	return Hugeint::Subtract(*this, rhs);
}

hugeint_t hugeint_t::operator*(const hugeint_t &rhs) const {
	return Hugeint::Multiply(*this, rhs);
}

hugeint_t hugeint_t::operator/(const hugeint_t &rhs) const {
	return Hugeint::Divide(*this, rhs);
}

hugeint_t hugeint_t::operator%(const hugeint_t &rhs) const {
	return Hugeint::Modulo(*this, rhs);
}

hugeint_t hugeint_t::operator-() const {
	return Hugeint::Negate(*this);
}

hugeint_t hugeint_t::operator>>(const hugeint_t &rhs) const {
	hugeint_t result;
	uint64_t shift = rhs.lower;
	if (rhs.upper != 0 || shift >= 128) {
		return hugeint_t(0);
	} else if (shift == 0) {
		return *this;
	} else if (shift == 64) {
		result.upper = (upper < 0) ? -1 : 0;
		result.lower = upper;
	} else if (shift < 64) {
		// perform lower shift in unsigned integer, and mask away the most significant bit
		result.lower = (uint64_t(upper) << (64 - shift)) | (lower >> shift);
		result.upper = upper >> shift;
	} else {
		D_ASSERT(shift < 128);
		result.lower = upper >> (shift - 64);
		result.upper = (upper < 0) ? -1 : 0;
	}
	return result;
}

hugeint_t hugeint_t::operator<<(const hugeint_t &rhs) const {
	if (upper < 0) {
		return hugeint_t(0);
	}
	hugeint_t result;
	uint64_t shift = rhs.lower;
	if (rhs.upper != 0 || shift >= 128) {
		return hugeint_t(0);
	} else if (shift == 64) {
		result.upper = lower;
		result.lower = 0;
	} else if (shift == 0) {
		return *this;
	} else if (shift < 64) {
		// perform upper shift in unsigned integer, and mask away the most significant bit
		uint64_t upper_shift = ((uint64_t(upper) << shift) + (lower >> (64 - shift))) & 0x7FFFFFFFFFFFFFFF;
		result.lower = lower << shift;
		result.upper = upper_shift;
	} else {
		D_ASSERT(shift < 128);
		result.lower = 0;
		result.upper = (lower << (shift - 64)) & 0x7FFFFFFFFFFFFFFF;
	}
	return result;
}

hugeint_t hugeint_t::operator&(const hugeint_t &rhs) const {
	hugeint_t result;
	result.lower = lower & rhs.lower;
	result.upper = upper & rhs.upper;
	return result;
}

hugeint_t hugeint_t::operator|(const hugeint_t &rhs) const {
	hugeint_t result;
	result.lower = lower | rhs.lower;
	result.upper = upper | rhs.upper;
	return result;
}

hugeint_t hugeint_t::operator^(const hugeint_t &rhs) const {
	hugeint_t result;
	result.lower = lower ^ rhs.lower;
	result.upper = upper ^ rhs.upper;
	return result;
}

hugeint_t hugeint_t::operator~() const {
	hugeint_t result;
	result.lower = ~lower;
	result.upper = ~upper;
	return result;
}

hugeint_t &hugeint_t::operator+=(const hugeint_t &rhs) {
	Hugeint::AddInPlace(*this, rhs);
	return *this;
}
hugeint_t &hugeint_t::operator-=(const hugeint_t &rhs) {
	Hugeint::SubtractInPlace(*this, rhs);
	return *this;
}
hugeint_t &hugeint_t::operator*=(const hugeint_t &rhs) {
	*this = Hugeint::Multiply(*this, rhs);
	return *this;
}
hugeint_t &hugeint_t::operator/=(const hugeint_t &rhs) {
	*this = Hugeint::Divide(*this, rhs);
	return *this;
}
hugeint_t &hugeint_t::operator%=(const hugeint_t &rhs) {
	*this = Hugeint::Modulo(*this, rhs);
	return *this;
}
hugeint_t &hugeint_t::operator>>=(const hugeint_t &rhs) {
	*this = *this >> rhs;
	return *this;
}
hugeint_t &hugeint_t::operator<<=(const hugeint_t &rhs) {
	*this = *this << rhs;
	return *this;
}
hugeint_t &hugeint_t::operator&=(const hugeint_t &rhs) {
	lower &= rhs.lower;
	upper &= rhs.upper;
	return *this;
}
hugeint_t &hugeint_t::operator|=(const hugeint_t &rhs) {
	lower |= rhs.lower;
	upper |= rhs.upper;
	return *this;
}
hugeint_t &hugeint_t::operator^=(const hugeint_t &rhs) {
	lower ^= rhs.lower;
	upper ^= rhs.upper;
	return *this;
}

string hugeint_t::ToString() const {
	return Hugeint::ToString(*this);
}

} // namespace duckdb






namespace duckdb {

HyperLogLog::HyperLogLog() : hll(nullptr) {
	hll = duckdb_hll::hll_create();
	// Insert into a dense hll can be vectorized, sparse cannot, so we immediately convert
	duckdb_hll::hllSparseToDense((duckdb_hll::robj *)hll);
}

HyperLogLog::HyperLogLog(void *hll) : hll(hll) {
}

HyperLogLog::~HyperLogLog() {
	duckdb_hll::hll_destroy((duckdb_hll::robj *)hll);
}

void HyperLogLog::Add(data_ptr_t element, idx_t size) {
	if (duckdb_hll::hll_add((duckdb_hll::robj *)hll, element, size) == HLL_C_ERR) {
		throw InternalException("Could not add to HLL?");
	}
}

idx_t HyperLogLog::Count() const {
	// exception from size_t ban
	size_t result;

	if (duckdb_hll::hll_count((duckdb_hll::robj *)hll, &result) != HLL_C_OK) {
		throw InternalException("Could not count HLL?");
	}
	return result;
}

unique_ptr<HyperLogLog> HyperLogLog::Merge(HyperLogLog &other) {
	duckdb_hll::robj *hlls[2];
	hlls[0] = (duckdb_hll::robj *)hll;
	hlls[1] = (duckdb_hll::robj *)other.hll;
	auto new_hll = duckdb_hll::hll_merge(hlls, 2);
	if (!new_hll) {
		throw InternalException("Could not merge HLLs");
	}
	return unique_ptr<HyperLogLog>(new HyperLogLog((void *)new_hll));
}

HyperLogLog *HyperLogLog::MergePointer(HyperLogLog &other) {
	duckdb_hll::robj *hlls[2];
	hlls[0] = (duckdb_hll::robj *)hll;
	hlls[1] = (duckdb_hll::robj *)other.hll;
	auto new_hll = duckdb_hll::hll_merge(hlls, 2);
	if (!new_hll) {
		throw Exception("Could not merge HLLs");
	}
	return new HyperLogLog((void *)new_hll);
}

unique_ptr<HyperLogLog> HyperLogLog::Merge(HyperLogLog logs[], idx_t count) {
	auto hlls_uptr = unique_ptr<duckdb_hll::robj *[]> {
		new duckdb_hll::robj *[count]
	};
	auto hlls = hlls_uptr.get();
	for (idx_t i = 0; i < count; i++) {
		hlls[i] = (duckdb_hll::robj *)logs[i].hll;
	}
	auto new_hll = duckdb_hll::hll_merge(hlls, count);
	if (!new_hll) {
		throw InternalException("Could not merge HLLs");
	}
	return unique_ptr<HyperLogLog>(new HyperLogLog((void *)new_hll));
}

idx_t HyperLogLog::GetSize() {
	return duckdb_hll::get_size();
}

data_ptr_t HyperLogLog::GetPtr() const {
	return (data_ptr_t)((duckdb_hll::robj *)hll)->ptr;
}

unique_ptr<HyperLogLog> HyperLogLog::Copy() {
	auto result = make_unique<HyperLogLog>();
	lock_guard<mutex> guard(lock);
	memcpy(result->GetPtr(), GetPtr(), GetSize());
	D_ASSERT(result->Count() == Count());
	return result;
}

void HyperLogLog::Serialize(FieldWriter &writer) const {
	writer.WriteField<HLLStorageType>(HLLStorageType::UNCOMPRESSED);
	writer.WriteBlob(GetPtr(), GetSize());
}

unique_ptr<HyperLogLog> HyperLogLog::Deserialize(FieldReader &reader) {
	auto result = make_unique<HyperLogLog>();
	auto storage_type = reader.ReadRequired<HLLStorageType>();
	switch (storage_type) {
	case HLLStorageType::UNCOMPRESSED:
		reader.ReadBlob(result->GetPtr(), GetSize());
		break;
	default:
		throw SerializationException("Unknown HyperLogLog storage type!");
	}
	return result;
}

//===--------------------------------------------------------------------===//
// Vectorized HLL implementation
//===--------------------------------------------------------------------===//
//! Taken from https://nullprogram.com/blog/2018/07/31/
template <class T>
inline uint64_t TemplatedHash(const T &elem) {
	uint64_t x = elem;
	x ^= x >> 30;
	x *= UINT64_C(0xbf58476d1ce4e5b9);
	x ^= x >> 27;
	x *= UINT64_C(0x94d049bb133111eb);
	x ^= x >> 31;
	return x;
}

template <>
inline uint64_t TemplatedHash(const hugeint_t &elem) {
	return TemplatedHash<uint64_t>(Load<uint64_t>((data_ptr_t)&elem.upper)) ^ TemplatedHash<uint64_t>(elem.lower);
}

template <idx_t rest>
inline void CreateIntegerRecursive(const data_ptr_t &data, uint64_t &x) {
	x ^= (uint64_t)data[rest - 1] << ((rest - 1) * 8);
	return CreateIntegerRecursive<rest - 1>(data, x);
}

template <>
inline void CreateIntegerRecursive<1>(const data_ptr_t &data, uint64_t &x) {
	x ^= (uint64_t)data[0];
}

inline uint64_t HashOtherSize(const data_ptr_t &data, const idx_t &len) {
	uint64_t x = 0;
	switch (len & 7) {
	case 7:
		CreateIntegerRecursive<7>(data, x);
		break;
	case 6:
		CreateIntegerRecursive<6>(data, x);
		break;
	case 5:
		CreateIntegerRecursive<5>(data, x);
		break;
	case 4:
		CreateIntegerRecursive<4>(data, x);
		break;
	case 3:
		CreateIntegerRecursive<3>(data, x);
		break;
	case 2:
		CreateIntegerRecursive<2>(data, x);
		break;
	case 1:
		CreateIntegerRecursive<1>(data, x);
		break;
	case 0:
		break;
	}
	return TemplatedHash<uint64_t>(x);
}

template <>
inline uint64_t TemplatedHash(const string_t &elem) {
	data_ptr_t data = (data_ptr_t)elem.GetDataUnsafe();
	const auto &len = elem.GetSize();
	uint64_t h = 0;
	for (idx_t i = 0; i + sizeof(uint64_t) <= len; i += sizeof(uint64_t)) {
		h ^= TemplatedHash<uint64_t>(Load<uint64_t>(data));
		data += sizeof(uint64_t);
	}
	switch (len & (sizeof(uint64_t) - 1)) {
	case 4:
		h ^= TemplatedHash<uint32_t>(Load<uint32_t>(data));
		break;
	case 2:
		h ^= TemplatedHash<uint16_t>(Load<uint16_t>(data));
		break;
	case 1:
		h ^= TemplatedHash<uint8_t>(Load<uint8_t>(data));
		break;
	default:
		h ^= HashOtherSize(data, len);
	}
	return h;
}

template <class T>
void TemplatedComputeHashes(UnifiedVectorFormat &vdata, const idx_t &count, uint64_t hashes[]) {
	T *data = (T *)vdata.data;
	for (idx_t i = 0; i < count; i++) {
		auto idx = vdata.sel->get_index(i);
		if (vdata.validity.RowIsValid(idx)) {
			hashes[i] = TemplatedHash<T>(data[idx]);
		} else {
			hashes[i] = 0;
		}
	}
}

static void ComputeHashes(UnifiedVectorFormat &vdata, const LogicalType &type, uint64_t hashes[], idx_t count) {
	switch (type.InternalType()) {
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
	case PhysicalType::UINT8:
		return TemplatedComputeHashes<uint8_t>(vdata, count, hashes);
	case PhysicalType::INT16:
	case PhysicalType::UINT16:
		return TemplatedComputeHashes<uint16_t>(vdata, count, hashes);
	case PhysicalType::INT32:
	case PhysicalType::UINT32:
	case PhysicalType::FLOAT:
		return TemplatedComputeHashes<uint32_t>(vdata, count, hashes);
	case PhysicalType::INT64:
	case PhysicalType::UINT64:
	case PhysicalType::DOUBLE:
		return TemplatedComputeHashes<uint64_t>(vdata, count, hashes);
	case PhysicalType::INT128:
	case PhysicalType::INTERVAL:
		static_assert(sizeof(hugeint_t) == sizeof(interval_t), "ComputeHashes assumes these are the same size!");
		return TemplatedComputeHashes<hugeint_t>(vdata, count, hashes);
	case PhysicalType::VARCHAR:
		return TemplatedComputeHashes<string_t>(vdata, count, hashes);
	default:
		throw InternalException("Unimplemented type for HyperLogLog::ComputeHashes");
	}
}

//! Taken from https://stackoverflow.com/a/72088344
static inline uint8_t CountTrailingZeros(uint64_t &x) {
	static constexpr const uint64_t DEBRUIJN = 0x03f79d71b4cb0a89;
	static constexpr const uint8_t LOOKUP[] = {0,  47, 1,  56, 48, 27, 2,  60, 57, 49, 41, 37, 28, 16, 3,  61,
	                                           54, 58, 35, 52, 50, 42, 21, 44, 38, 32, 29, 23, 17, 11, 4,  62,
	                                           46, 55, 26, 59, 40, 36, 15, 53, 34, 51, 20, 43, 31, 22, 10, 45,
	                                           25, 39, 14, 33, 19, 30, 9,  24, 13, 18, 8,  12, 7,  6,  5,  63};
	return LOOKUP[(DEBRUIJN * (x ^ (x - 1))) >> 58];
}

static inline void ComputeIndexAndCount(uint64_t &hash, uint8_t &prefix) {
	uint64_t index = hash & ((1 << 12) - 1); /* Register index. */
	hash >>= 12;                             /* Remove bits used to address the register. */
	hash |= ((uint64_t)1 << (64 - 12));      /* Make sure the count will be <= Q+1. */

	prefix = CountTrailingZeros(hash) + 1; /* Add 1 since we count the "00000...1" pattern. */
	hash = index;
}

void HyperLogLog::ProcessEntries(UnifiedVectorFormat &vdata, const LogicalType &type, uint64_t hashes[],
                                 uint8_t counts[], idx_t count) {
	ComputeHashes(vdata, type, hashes, count);
	for (idx_t i = 0; i < count; i++) {
		ComputeIndexAndCount(hashes[i], counts[i]);
	}
}

void HyperLogLog::AddToLogs(UnifiedVectorFormat &vdata, idx_t count, uint64_t indices[], uint8_t counts[],
                            HyperLogLog **logs[], const SelectionVector *log_sel) {
	AddToLogsInternal(vdata, count, indices, counts, (void ****)logs, log_sel);
}

void HyperLogLog::AddToLog(UnifiedVectorFormat &vdata, idx_t count, uint64_t indices[], uint8_t counts[]) {
	lock_guard<mutex> guard(lock);
	AddToSingleLogInternal(vdata, count, indices, counts, hll);
}

} // namespace duckdb













namespace duckdb {

bool Interval::FromString(const string &str, interval_t &result) {
	string error_message;
	return Interval::FromCString(str.c_str(), str.size(), result, &error_message, false);
}

template <class T>
void IntervalTryAddition(T &target, int64_t input, int64_t multiplier) {
	int64_t addition;
	if (!TryMultiplyOperator::Operation<int64_t, int64_t, int64_t>(input, multiplier, addition)) {
		throw OutOfRangeException("interval value is out of range");
	}
	T addition_base = Cast::Operation<int64_t, T>(addition);
	if (!TryAddOperator::Operation<T, T, T>(target, addition_base, target)) {
		throw OutOfRangeException("interval value is out of range");
	}
}

bool Interval::FromCString(const char *str, idx_t len, interval_t &result, string *error_message, bool strict) {
	idx_t pos = 0;
	idx_t start_pos;
	bool negative;
	bool found_any = false;
	int64_t number;
	DatePartSpecifier specifier;
	string specifier_str;

	result.days = 0;
	result.micros = 0;
	result.months = 0;

	if (len == 0) {
		return false;
	}

	switch (str[pos]) {
	case '@':
		pos++;
		goto standard_interval;
	case 'P':
	case 'p':
		pos++;
		goto posix_interval;
	default:
		goto standard_interval;
	}
standard_interval:
	// start parsing a standard interval (e.g. 2 years 3 months...)
	for (; pos < len; pos++) {
		char c = str[pos];
		if (c == ' ' || c == '\t' || c == '\n') {
			// skip spaces
			continue;
		} else if (c >= '0' && c <= '9') {
			// start parsing a positive number
			negative = false;
			goto interval_parse_number;
		} else if (c == '-') {
			// negative number
			negative = true;
			pos++;
			goto interval_parse_number;
		} else if (c == 'a' || c == 'A') {
			// parse the word "ago" as the final specifier
			goto interval_parse_ago;
		} else {
			// unrecognized character, expected a number or end of string
			return false;
		}
	}
	goto end_of_string;
interval_parse_number:
	start_pos = pos;
	for (; pos < len; pos++) {
		char c = str[pos];
		if (c >= '0' && c <= '9') {
			// the number continues
			continue;
		} else if (c == ':') {
			// colon: we are parsing a time
			goto interval_parse_time;
		} else {
			if (pos == start_pos) {
				return false;
			}
			// finished the number, parse it from the string
			string_t nr_string(str + start_pos, pos - start_pos);
			number = Cast::Operation<string_t, int64_t>(nr_string);
			if (negative) {
				number = -number;
			}
			goto interval_parse_identifier;
		}
	}
	goto end_of_string;
interval_parse_time : {
	// parse the remainder of the time as a Time type
	dtime_t time;
	idx_t pos;
	if (!Time::TryConvertTime(str + start_pos, len - start_pos, pos, time)) {
		return false;
	}
	result.micros += time.micros;
	found_any = true;
	goto end_of_string;
}
interval_parse_identifier:
	for (; pos < len; pos++) {
		char c = str[pos];
		if (c == ' ' || c == '\t' || c == '\n') {
			// skip spaces at the start
			continue;
		} else {
			break;
		}
	}
	// now parse the identifier
	start_pos = pos;
	for (; pos < len; pos++) {
		char c = str[pos];
		if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
			// keep parsing the string
			continue;
		} else {
			break;
		}
	}
	specifier_str = string(str + start_pos, pos - start_pos);
	if (!TryGetDatePartSpecifier(specifier_str, specifier)) {
		HandleCastError::AssignError(StringUtil::Format("extract specifier \"%s\" not recognized", specifier_str),
		                             error_message);
		return false;
	}
	// add the specifier to the interval
	switch (specifier) {
	case DatePartSpecifier::MILLENNIUM:
		IntervalTryAddition<int32_t>(result.months, number, MONTHS_PER_MILLENIUM);
		break;
	case DatePartSpecifier::CENTURY:
		IntervalTryAddition<int32_t>(result.months, number, MONTHS_PER_CENTURY);
		break;
	case DatePartSpecifier::DECADE:
		IntervalTryAddition<int32_t>(result.months, number, MONTHS_PER_DECADE);
		break;
	case DatePartSpecifier::YEAR:
		IntervalTryAddition<int32_t>(result.months, number, MONTHS_PER_YEAR);
		break;
	case DatePartSpecifier::QUARTER:
		IntervalTryAddition<int32_t>(result.months, number, MONTHS_PER_QUARTER);
		break;
	case DatePartSpecifier::MONTH:
		IntervalTryAddition<int32_t>(result.months, number, 1);
		break;
	case DatePartSpecifier::DAY:
		IntervalTryAddition<int32_t>(result.days, number, 1);
		break;
	case DatePartSpecifier::WEEK:
		IntervalTryAddition<int32_t>(result.days, number, DAYS_PER_WEEK);
		break;
	case DatePartSpecifier::MICROSECONDS:
		IntervalTryAddition<int64_t>(result.micros, number, 1);
		break;
	case DatePartSpecifier::MILLISECONDS:
		IntervalTryAddition<int64_t>(result.micros, number, MICROS_PER_MSEC);
		break;
	case DatePartSpecifier::SECOND:
		IntervalTryAddition<int64_t>(result.micros, number, MICROS_PER_SEC);
		break;
	case DatePartSpecifier::MINUTE:
		IntervalTryAddition<int64_t>(result.micros, number, MICROS_PER_MINUTE);
		break;
	case DatePartSpecifier::HOUR:
		IntervalTryAddition<int64_t>(result.micros, number, MICROS_PER_HOUR);
		break;
	default:
		HandleCastError::AssignError(
		    StringUtil::Format("extract specifier \"%s\" not supported for interval", specifier_str), error_message);
		return false;
	}
	found_any = true;
	goto standard_interval;
interval_parse_ago:
	D_ASSERT(str[pos] == 'a' || str[pos] == 'A');
	// parse the "ago" string at the end of the interval
	if (len - pos < 3) {
		return false;
	}
	pos++;
	if (!(str[pos] == 'g' || str[pos] == 'G')) {
		return false;
	}
	pos++;
	if (!(str[pos] == 'o' || str[pos] == 'O')) {
		return false;
	}
	pos++;
	// parse any trailing whitespace
	for (; pos < len; pos++) {
		char c = str[pos];
		if (c == ' ' || c == '\t' || c == '\n') {
			continue;
		} else {
			return false;
		}
	}
	// invert all the values
	result.months = -result.months;
	result.days = -result.days;
	result.micros = -result.micros;
	goto end_of_string;
end_of_string:
	if (!found_any) {
		// end of string and no identifiers were found: cannot convert empty interval
		return false;
	}
	return true;
posix_interval:
	return false;
}

string Interval::ToString(const interval_t &interval) {
	char buffer[70];
	idx_t length = IntervalToStringCast::Format(interval, buffer);
	return string(buffer, length);
}

int64_t Interval::GetMilli(const interval_t &val) {
	int64_t milli_month, milli_day, milli;
	if (!TryMultiplyOperator::Operation((int64_t)val.months, Interval::MICROS_PER_MONTH / 1000, milli_month)) {
		throw ConversionException("Could not convert Interval to Milliseconds");
	}
	if (!TryMultiplyOperator::Operation((int64_t)val.days, Interval::MICROS_PER_DAY / 1000, milli_day)) {
		throw ConversionException("Could not convert Interval to Milliseconds");
	}
	milli = val.micros / 1000;
	if (!TryAddOperator::Operation<int64_t, int64_t, int64_t>(milli, milli_month, milli)) {
		throw ConversionException("Could not convert Interval to Milliseconds");
	}
	if (!TryAddOperator::Operation<int64_t, int64_t, int64_t>(milli, milli_day, milli)) {
		throw ConversionException("Could not convert Interval to Milliseconds");
	}
	return milli;
}

int64_t Interval::GetMicro(const interval_t &val) {
	int64_t micro_month, micro_day, micro_total;
	micro_total = val.micros;
	if (!TryMultiplyOperator::Operation((int64_t)val.months, MICROS_PER_MONTH, micro_month)) {
		throw ConversionException("Could not convert Month to Microseconds");
	}
	if (!TryMultiplyOperator::Operation((int64_t)val.days, MICROS_PER_DAY, micro_day)) {
		throw ConversionException("Could not convert Day to Microseconds");
	}
	if (!TryAddOperator::Operation<int64_t, int64_t, int64_t>(micro_total, micro_month, micro_total)) {
		throw ConversionException("Could not convert Interval to Microseconds");
	}
	if (!TryAddOperator::Operation<int64_t, int64_t, int64_t>(micro_total, micro_day, micro_total)) {
		throw ConversionException("Could not convert Interval to Microseconds");
	}

	return micro_total;
}

int64_t Interval::GetNanoseconds(const interval_t &val) {
	int64_t nano;
	const auto micro_total = GetMicro(val);
	if (!TryMultiplyOperator::Operation(micro_total, NANOS_PER_MICRO, nano)) {
		throw ConversionException("Could not convert Interval to Nanoseconds");
	}

	return nano;
}

interval_t Interval::GetAge(timestamp_t timestamp_1, timestamp_t timestamp_2) {
	D_ASSERT(Timestamp::IsFinite(timestamp_1) && Timestamp::IsFinite(timestamp_2));
	date_t date1, date2;
	dtime_t time1, time2;

	Timestamp::Convert(timestamp_1, date1, time1);
	Timestamp::Convert(timestamp_2, date2, time2);

	// and from date extract the years, months and days
	int32_t year1, month1, day1;
	int32_t year2, month2, day2;
	Date::Convert(date1, year1, month1, day1);
	Date::Convert(date2, year2, month2, day2);
	// finally perform the differences
	auto year_diff = year1 - year2;
	auto month_diff = month1 - month2;
	auto day_diff = day1 - day2;

	// and from time extract hours, minutes, seconds and milliseconds
	int32_t hour1, min1, sec1, micros1;
	int32_t hour2, min2, sec2, micros2;
	Time::Convert(time1, hour1, min1, sec1, micros1);
	Time::Convert(time2, hour2, min2, sec2, micros2);
	// finally perform the differences
	auto hour_diff = hour1 - hour2;
	auto min_diff = min1 - min2;
	auto sec_diff = sec1 - sec2;
	auto micros_diff = micros1 - micros2;

	// flip sign if necessary
	bool sign_flipped = false;
	if (timestamp_1 < timestamp_2) {
		year_diff = -year_diff;
		month_diff = -month_diff;
		day_diff = -day_diff;
		hour_diff = -hour_diff;
		min_diff = -min_diff;
		sec_diff = -sec_diff;
		micros_diff = -micros_diff;
		sign_flipped = true;
	}
	// now propagate any negative field into the next higher field
	while (micros_diff < 0) {
		micros_diff += MICROS_PER_SEC;
		sec_diff--;
	}
	while (sec_diff < 0) {
		sec_diff += SECS_PER_MINUTE;
		min_diff--;
	}
	while (min_diff < 0) {
		min_diff += MINS_PER_HOUR;
		hour_diff--;
	}
	while (hour_diff < 0) {
		hour_diff += HOURS_PER_DAY;
		day_diff--;
	}
	while (day_diff < 0) {
		if (timestamp_1 < timestamp_2) {
			day_diff += Date::IsLeapYear(year1) ? Date::LEAP_DAYS[month1] : Date::NORMAL_DAYS[month1];
			month_diff--;
		} else {
			day_diff += Date::IsLeapYear(year2) ? Date::LEAP_DAYS[month2] : Date::NORMAL_DAYS[month2];
			month_diff--;
		}
	}
	while (month_diff < 0) {
		month_diff += MONTHS_PER_YEAR;
		year_diff--;
	}

	// recover sign if necessary
	if (sign_flipped) {
		year_diff = -year_diff;
		month_diff = -month_diff;
		day_diff = -day_diff;
		hour_diff = -hour_diff;
		min_diff = -min_diff;
		sec_diff = -sec_diff;
		micros_diff = -micros_diff;
	}
	interval_t interval;
	interval.months = year_diff * MONTHS_PER_YEAR + month_diff;
	interval.days = day_diff;
	interval.micros = Time::FromTime(hour_diff, min_diff, sec_diff, micros_diff).micros;

	return interval;
}

interval_t Interval::GetDifference(timestamp_t timestamp_1, timestamp_t timestamp_2) {
	if (!Timestamp::IsFinite(timestamp_1) || !Timestamp::IsFinite(timestamp_2)) {
		throw InvalidInputException("Cannot subtract infinite timestamps");
	}
	const auto us_1 = Timestamp::GetEpochMicroSeconds(timestamp_1);
	const auto us_2 = Timestamp::GetEpochMicroSeconds(timestamp_2);
	int64_t delta_us;
	if (!TrySubtractOperator::Operation(us_1, us_2, delta_us)) {
		throw ConversionException("Timestamp difference is out of bounds");
	}
	return FromMicro(delta_us);
}

interval_t Interval::FromMicro(int64_t delta_us) {
	interval_t result;
	result.months = 0;
	result.days = delta_us / Interval::MICROS_PER_DAY;
	result.micros = delta_us % Interval::MICROS_PER_DAY;

	return result;
}

static void NormalizeIntervalEntries(interval_t input, int64_t &months, int64_t &days, int64_t &micros) {
	int64_t extra_months_d = input.days / Interval::DAYS_PER_MONTH;
	int64_t extra_months_micros = input.micros / Interval::MICROS_PER_MONTH;
	input.days -= extra_months_d * Interval::DAYS_PER_MONTH;
	input.micros -= extra_months_micros * Interval::MICROS_PER_MONTH;

	int64_t extra_days_micros = input.micros / Interval::MICROS_PER_DAY;
	input.micros -= extra_days_micros * Interval::MICROS_PER_DAY;

	months = input.months + extra_months_d + extra_months_micros;
	days = input.days + extra_days_micros;
	micros = input.micros;
}

bool Interval::Equals(interval_t left, interval_t right) {
	return left.months == right.months && left.days == right.days && left.micros == right.micros;
}

bool Interval::GreaterThan(interval_t left, interval_t right) {
	int64_t lmonths, ldays, lmicros;
	int64_t rmonths, rdays, rmicros;
	NormalizeIntervalEntries(left, lmonths, ldays, lmicros);
	NormalizeIntervalEntries(right, rmonths, rdays, rmicros);

	if (lmonths > rmonths) {
		return true;
	} else if (lmonths < rmonths) {
		return false;
	}
	if (ldays > rdays) {
		return true;
	} else if (ldays < rdays) {
		return false;
	}
	return lmicros > rmicros;
}

bool Interval::GreaterThanEquals(interval_t left, interval_t right) {
	return GreaterThan(left, right) || Equals(left, right);
}

interval_t Interval::Invert(interval_t interval) {
	interval.days = -interval.days;
	interval.micros = -interval.micros;
	interval.months = -interval.months;
	return interval;
}

date_t Interval::Add(date_t left, interval_t right) {
	if (!Date::IsFinite(left)) {
		return left;
	}
	date_t result;
	if (right.months != 0) {
		int32_t year, month, day;
		Date::Convert(left, year, month, day);
		int32_t year_diff = right.months / Interval::MONTHS_PER_YEAR;
		year += year_diff;
		month += right.months - year_diff * Interval::MONTHS_PER_YEAR;
		if (month > Interval::MONTHS_PER_YEAR) {
			year++;
			month -= Interval::MONTHS_PER_YEAR;
		} else if (month <= 0) {
			year--;
			month += Interval::MONTHS_PER_YEAR;
		}
		day = MinValue<int32_t>(day, Date::MonthDays(year, month));
		result = Date::FromDate(year, month, day);
	} else {
		result = left;
	}
	if (right.days != 0) {
		if (!TryAddOperator::Operation(result.days, right.days, result.days)) {
			throw OutOfRangeException("Date out of range");
		}
	}
	if (right.micros != 0) {
		if (!TryAddOperator::Operation(result.days, int32_t(right.micros / Interval::MICROS_PER_DAY), result.days)) {
			throw OutOfRangeException("Date out of range");
		}
	}
	if (!Date::IsFinite(result)) {
		throw OutOfRangeException("Date out of range");
	}
	return result;
}

dtime_t Interval::Add(dtime_t left, interval_t right, date_t &date) {
	int64_t diff = right.micros - ((right.micros / Interval::MICROS_PER_DAY) * Interval::MICROS_PER_DAY);
	left += diff;
	if (left.micros >= Interval::MICROS_PER_DAY) {
		left.micros -= Interval::MICROS_PER_DAY;
		date.days++;
	} else if (left.micros < 0) {
		left.micros += Interval::MICROS_PER_DAY;
		date.days--;
	}
	return left;
}

timestamp_t Interval::Add(timestamp_t left, interval_t right) {
	if (!Timestamp::IsFinite(left)) {
		return left;
	}
	date_t date;
	dtime_t time;
	Timestamp::Convert(left, date, time);
	auto new_date = Interval::Add(date, right);
	auto new_time = Interval::Add(time, right, new_date);
	return Timestamp::FromDatetime(new_date, new_time);
}

} // namespace duckdb





namespace duckdb {

PartitionedColumnData::PartitionedColumnData(PartitionedColumnDataType type_p, ClientContext &context_p,
                                             vector<LogicalType> types_p)
    : type(type_p), context(context_p), types(move(types_p)), allocators(make_shared<PartitionAllocators>()) {
}

PartitionedColumnData::PartitionedColumnData(const PartitionedColumnData &other)
    : type(other.type), context(other.context), types(other.types), allocators(other.allocators) {
}

unique_ptr<PartitionedColumnData> PartitionedColumnData::CreateShared() {
	switch (type) {
	case PartitionedColumnDataType::RADIX:
		return make_unique<RadixPartitionedColumnData>((RadixPartitionedColumnData &)*this);
	default:
		throw NotImplementedException("CreateShared for this type of PartitionedColumnData");
	}
}

PartitionedColumnData::~PartitionedColumnData() {
}

void PartitionedColumnData::InitializeAppendState(PartitionedColumnDataAppendState &state) const {
	state.partition_sel.Initialize();
	state.slice_chunk.Initialize(context, types);
	InitializeAppendStateInternal(state);
}

unique_ptr<DataChunk> PartitionedColumnData::CreatePartitionBuffer() const {
	auto result = make_unique<DataChunk>();
	result->Initialize(BufferManager::GetBufferManager(context).GetBufferAllocator(), types, BufferSize());
	return result;
}

void PartitionedColumnData::Append(PartitionedColumnDataAppendState &state, DataChunk &input) {
	// Compute partition indices and store them in state.partition_indices
	ComputePartitionIndices(state, input);

	// Compute the counts per partition
	const auto count = input.size();
	unordered_map<idx_t, list_entry_t> partition_entries;
	const auto partition_indices = FlatVector::GetData<idx_t>(state.partition_indices);
	switch (state.partition_indices.GetVectorType()) {
	case VectorType::FLAT_VECTOR:
		for (idx_t i = 0; i < count; i++) {
			const auto &partition_index = partition_indices[i];
			auto partition_entry = partition_entries.find(partition_index);
			if (partition_entry == partition_entries.end()) {
				partition_entries[partition_index] = list_entry_t(0, 1);
			} else {
				partition_entry->second.length++;
			}
		}
		break;
	case VectorType::CONSTANT_VECTOR:
		partition_entries[partition_indices[0]] = list_entry_t(0, count);
		break;
	default:
		throw InternalException("Unexpected VectorType in PartitionedColumnData::Append");
	}

	// Early out: check if everything belongs to a single partition
	if (partition_entries.size() == 1) {
		const auto &partition_index = partition_entries.begin()->first;
		auto &partition = *partitions[partition_index];
		auto &partition_append_state = state.partition_append_states[partition_index];
		partition.Append(*partition_append_state, input);
		return;
	}

	// Compute offsets from the counts
	idx_t offset = 0;
	for (auto &pc : partition_entries) {
		auto &partition_entry = pc.second;
		partition_entry.offset = offset;
		offset += partition_entry.length;
	}

	// Now initialize a single selection vector that acts as a selection vector for every partition
	auto &all_partitions_sel = state.partition_sel;
	for (idx_t i = 0; i < count; i++) {
		const auto &partition_index = partition_indices[i];
		auto &partition_offset = partition_entries[partition_index].offset;
		all_partitions_sel[partition_offset++] = i;
	}

	// Loop through the partitions to append the new data to the partition buffers, and flush the buffers if necessary
	SelectionVector partition_sel;
	for (auto &pc : partition_entries) {
		const auto &partition_index = pc.first;

		// Partition, buffer, and append state for this partition index
		auto &partition = *partitions[partition_index];
		auto &partition_buffer = *state.partition_buffers[partition_index];
		auto &partition_append_state = state.partition_append_states[partition_index];

		// Length and offset into the selection vector for this chunk, for this partition
		const auto &partition_entry = pc.second;
		const auto &partition_length = partition_entry.length;
		const auto partition_offset = partition_entry.offset - partition_length;

		// Create a selection vector for this partition using the offset into the single selection vector
		partition_sel.Initialize(all_partitions_sel.data() + partition_offset);

		if (partition_length >= HalfBufferSize()) {
			// Slice the input chunk using the selection vector
			state.slice_chunk.Reset();
			state.slice_chunk.Slice(input, partition_sel, partition_length);

			// Append it to the partition directly
			partition.Append(*partition_append_state, state.slice_chunk);
		} else {
			// Append the input chunk to the partition buffer using the selection vector
			partition_buffer.Append(input, false, &partition_sel, partition_length);

			if (partition_buffer.size() >= HalfBufferSize()) {
				// Next batch won't fit in the buffer, flush it to the partition
				partition.Append(*partition_append_state, partition_buffer);
				partition_buffer.Reset();
				partition_buffer.SetCapacity(BufferSize());
			}
		}
	}
}

void PartitionedColumnData::FlushAppendState(PartitionedColumnDataAppendState &state) {
	for (idx_t i = 0; i < state.partition_buffers.size(); i++) {
		auto &partition_buffer = *state.partition_buffers[i];
		if (partition_buffer.size() > 0) {
			partitions[i]->Append(partition_buffer);
		}
	}
}

void PartitionedColumnData::Combine(PartitionedColumnData &other) {
	// Now combine the state's partitions into this
	lock_guard<mutex> guard(lock);
	if (partitions.empty()) {
		// This is the first merge, we just copy them over
		partitions = move(other.partitions);
	} else {
		// Combine the append state's partitions into this PartitionedColumnData
		for (idx_t i = 0; i < other.partitions.size(); i++) {
			partitions[i]->Combine(*other.partitions[i]);
		}
	}
}

vector<unique_ptr<ColumnDataCollection>> &PartitionedColumnData::GetPartitions() {
	return partitions;
}

void PartitionedColumnData::CreateAllocator() {
	allocators->allocators.emplace_back(make_shared<ColumnDataAllocator>(BufferManager::GetBufferManager(context)));
	allocators->allocators.back()->MakeShared();
}

} // namespace duckdb


namespace duckdb {

RowDataCollection::RowDataCollection(BufferManager &buffer_manager, idx_t block_capacity, idx_t entry_size,
                                     bool keep_pinned)
    : buffer_manager(buffer_manager), count(0), block_capacity(block_capacity), entry_size(entry_size),
      keep_pinned(keep_pinned) {
	D_ASSERT(block_capacity * entry_size + entry_size > Storage::BLOCK_SIZE);
}

idx_t RowDataCollection::AppendToBlock(RowDataBlock &block, BufferHandle &handle,
                                       vector<BlockAppendEntry> &append_entries, idx_t remaining, idx_t entry_sizes[]) {
	idx_t append_count = 0;
	data_ptr_t dataptr;
	if (entry_sizes) {
		D_ASSERT(entry_size == 1);
		// compute how many entries fit if entry size is variable
		dataptr = handle.Ptr() + block.byte_offset;
		for (idx_t i = 0; i < remaining; i++) {
			if (block.byte_offset + entry_sizes[i] > block.capacity) {
				if (block.count == 0 && append_count == 0 && entry_sizes[i] > block.capacity) {
					// special case: single entry is bigger than block capacity
					// resize current block to fit the entry, append it, and move to the next block
					block.capacity = entry_sizes[i];
					buffer_manager.ReAllocate(block.block, block.capacity);
					dataptr = handle.Ptr();
					append_count++;
					block.byte_offset += entry_sizes[i];
				}
				break;
			}
			append_count++;
			block.byte_offset += entry_sizes[i];
		}
	} else {
		append_count = MinValue<idx_t>(remaining, block.capacity - block.count);
		dataptr = handle.Ptr() + block.count * entry_size;
	}
	append_entries.emplace_back(dataptr, append_count);
	block.count += append_count;
	return append_count;
}

RowDataBlock &RowDataCollection::CreateBlock() {
	blocks.push_back(make_unique<RowDataBlock>(buffer_manager, block_capacity, entry_size));
	return *blocks.back();
}

vector<BufferHandle> RowDataCollection::Build(idx_t added_count, data_ptr_t key_locations[], idx_t entry_sizes[],
                                              const SelectionVector *sel) {
	vector<BufferHandle> handles;
	vector<BlockAppendEntry> append_entries;

	// first allocate space of where to serialize the keys and payload columns
	idx_t remaining = added_count;
	{
		// first append to the last block (if any)
		lock_guard<mutex> append_lock(rdc_lock);
		count += added_count;

		if (!blocks.empty()) {
			auto &last_block = *blocks.back();
			if (last_block.count < last_block.capacity) {
				// last block has space: pin the buffer of this block
				auto handle = buffer_manager.Pin(last_block.block);
				// now append to the block
				idx_t append_count = AppendToBlock(last_block, handle, append_entries, remaining, entry_sizes);
				remaining -= append_count;
				handles.push_back(move(handle));
			}
		}
		while (remaining > 0) {
			// now for the remaining data, allocate new buffers to store the data and append there
			auto &new_block = CreateBlock();
			auto handle = buffer_manager.Pin(new_block.block);

			// offset the entry sizes array if we have added entries already
			idx_t *offset_entry_sizes = entry_sizes ? entry_sizes + added_count - remaining : nullptr;

			idx_t append_count = AppendToBlock(new_block, handle, append_entries, remaining, offset_entry_sizes);
			D_ASSERT(new_block.count > 0);
			remaining -= append_count;

			if (keep_pinned) {
				pinned_blocks.push_back(move(handle));
			} else {
				handles.push_back(move(handle));
			}
		}
	}
	// now set up the key_locations based on the append entries
	idx_t append_idx = 0;
	for (auto &append_entry : append_entries) {
		idx_t next = append_idx + append_entry.count;
		if (entry_sizes) {
			for (; append_idx < next; append_idx++) {
				key_locations[append_idx] = append_entry.baseptr;
				append_entry.baseptr += entry_sizes[append_idx];
			}
		} else {
			for (; append_idx < next; append_idx++) {
				auto idx = sel->get_index(append_idx);
				key_locations[idx] = append_entry.baseptr;
				append_entry.baseptr += entry_size;
			}
		}
	}
	// return the unique pointers to the handles because they must stay pinned
	return handles;
}

void RowDataCollection::Merge(RowDataCollection &other) {
	if (other.count == 0) {
		return;
	}
	RowDataCollection temp(buffer_manager, Storage::BLOCK_SIZE, 1);
	{
		//	One lock at a time to avoid deadlocks
		lock_guard<mutex> read_lock(other.rdc_lock);
		temp.count = other.count;
		temp.block_capacity = other.block_capacity;
		temp.entry_size = other.entry_size;
		temp.blocks = move(other.blocks);
		temp.pinned_blocks = move(other.pinned_blocks);
	}
	other.Clear();

	lock_guard<mutex> write_lock(rdc_lock);
	count += temp.count;
	block_capacity = MaxValue(block_capacity, temp.block_capacity);
	entry_size = MaxValue(entry_size, temp.entry_size);
	for (auto &block : temp.blocks) {
		blocks.emplace_back(move(block));
	}
	for (auto &handle : temp.pinned_blocks) {
		pinned_blocks.emplace_back(move(handle));
	}
}

} // namespace duckdb






namespace duckdb {

void RowDataCollectionScanner::AlignHeapBlocks(RowDataCollection &swizzled_block_collection,
                                               RowDataCollection &swizzled_string_heap,
                                               RowDataCollection &block_collection, RowDataCollection &string_heap,
                                               const RowLayout &layout) {
	if (block_collection.count == 0) {
		return;
	}

	if (layout.AllConstant()) {
		// No heap blocks! Just merge fixed-size data
		swizzled_block_collection.Merge(block_collection);
		return;
	}

	// We create one heap block per data block and swizzle the pointers
	D_ASSERT(string_heap.keep_pinned == swizzled_string_heap.keep_pinned);
	auto &buffer_manager = block_collection.buffer_manager;
	auto &heap_blocks = string_heap.blocks;
	idx_t heap_block_idx = 0;
	idx_t heap_block_remaining = heap_blocks[heap_block_idx]->count;
	for (auto &data_block : block_collection.blocks) {
		if (heap_block_remaining == 0) {
			heap_block_remaining = heap_blocks[++heap_block_idx]->count;
		}

		// Pin the data block and swizzle the pointers within the rows
		auto data_handle = buffer_manager.Pin(data_block->block);
		auto data_ptr = data_handle.Ptr();
		if (!string_heap.keep_pinned) {
			D_ASSERT(!data_block->block->IsSwizzled());
			RowOperations::SwizzleColumns(layout, data_ptr, data_block->count);
			data_block->block->SetSwizzling(nullptr);
		}
		// At this point the data block is pinned and the heap pointer is valid
		// so we can copy heap data as needed

		// We want to copy as little of the heap data as possible, check how the data and heap blocks line up
		if (heap_block_remaining >= data_block->count) {
			// Easy: current heap block contains all strings for this data block, just copy (reference) the block
			swizzled_string_heap.blocks.emplace_back(heap_blocks[heap_block_idx]->Copy());
			swizzled_string_heap.blocks.back()->count = data_block->count;

			// Swizzle the heap pointer if we are not pinning the heap
			auto &heap_block = swizzled_string_heap.blocks.back()->block;
			auto heap_handle = buffer_manager.Pin(heap_block);
			if (!swizzled_string_heap.keep_pinned) {
				auto heap_ptr = Load<data_ptr_t>(data_ptr + layout.GetHeapOffset());
				auto heap_offset = heap_ptr - heap_handle.Ptr();
				RowOperations::SwizzleHeapPointer(layout, data_ptr, heap_ptr, data_block->count, heap_offset);
			} else {
				swizzled_string_heap.pinned_blocks.emplace_back(move(heap_handle));
			}

			// Update counter
			heap_block_remaining -= data_block->count;
		} else {
			// Strings for this data block are spread over the current heap block and the next (and possibly more)
			if (string_heap.keep_pinned) {
				// The heap is changing underneath the data block,
				// so swizzle the string pointers to make them portable.
				RowOperations::SwizzleColumns(layout, data_ptr, data_block->count);
			}
			idx_t data_block_remaining = data_block->count;
			vector<std::pair<data_ptr_t, idx_t>> ptrs_and_sizes;
			idx_t total_size = 0;
			const auto base_row_ptr = data_ptr;
			while (data_block_remaining > 0) {
				if (heap_block_remaining == 0) {
					heap_block_remaining = heap_blocks[++heap_block_idx]->count;
				}
				auto next = MinValue<idx_t>(data_block_remaining, heap_block_remaining);

				// Figure out where to start copying strings, and how many bytes we need to copy
				auto heap_start_ptr = Load<data_ptr_t>(data_ptr + layout.GetHeapOffset());
				auto heap_end_ptr =
				    Load<data_ptr_t>(data_ptr + layout.GetHeapOffset() + (next - 1) * layout.GetRowWidth());
				idx_t size = heap_end_ptr - heap_start_ptr + Load<uint32_t>(heap_end_ptr);
				ptrs_and_sizes.emplace_back(heap_start_ptr, size);
				D_ASSERT(size <= heap_blocks[heap_block_idx]->byte_offset);

				// Swizzle the heap pointer
				RowOperations::SwizzleHeapPointer(layout, data_ptr, heap_start_ptr, next, total_size);
				total_size += size;

				// Update where we are in the data and heap blocks
				data_ptr += next * layout.GetRowWidth();
				data_block_remaining -= next;
				heap_block_remaining -= next;
			}

			// Finally, we allocate a new heap block and copy data to it
			swizzled_string_heap.blocks.emplace_back(
			    make_unique<RowDataBlock>(buffer_manager, MaxValue<idx_t>(total_size, (idx_t)Storage::BLOCK_SIZE), 1));
			auto new_heap_handle = buffer_manager.Pin(swizzled_string_heap.blocks.back()->block);
			auto new_heap_ptr = new_heap_handle.Ptr();
			if (swizzled_string_heap.keep_pinned) {
				// Since the heap blocks are pinned, we can unswizzle the data again.
				swizzled_string_heap.pinned_blocks.emplace_back(move(new_heap_handle));
				RowOperations::UnswizzlePointers(layout, base_row_ptr, new_heap_ptr, data_block->count);
				RowOperations::UnswizzleHeapPointer(layout, base_row_ptr, new_heap_ptr, data_block->count);
			}
			for (auto &ptr_and_size : ptrs_and_sizes) {
				memcpy(new_heap_ptr, ptr_and_size.first, ptr_and_size.second);
				new_heap_ptr += ptr_and_size.second;
			}
		}
	}

	// We're done with variable-sized data, now just merge the fixed-size data
	swizzled_block_collection.Merge(block_collection);
	D_ASSERT(swizzled_block_collection.blocks.size() == swizzled_string_heap.blocks.size());

	// Update counts and cleanup
	swizzled_string_heap.count = string_heap.count;
	string_heap.Clear();
}

void RowDataCollectionScanner::ScanState::PinData() {
	auto &rows = scanner.rows;
	D_ASSERT(block_idx < rows.blocks.size());
	auto &data_block = rows.blocks[block_idx];
	if (!data_handle.IsValid() || data_handle.GetBlockHandle() != data_block->block) {
		data_handle = rows.buffer_manager.Pin(data_block->block);
	}
	if (scanner.layout.AllConstant() || !scanner.external) {
		return;
	}

	auto &heap = scanner.heap;
	D_ASSERT(block_idx < heap.blocks.size());
	auto &heap_block = heap.blocks[block_idx];
	if (!heap_handle.IsValid() || heap_handle.GetBlockHandle() != heap_block->block) {
		heap_handle = heap.buffer_manager.Pin(heap_block->block);
	}
}

RowDataCollectionScanner::RowDataCollectionScanner(RowDataCollection &rows_p, RowDataCollection &heap_p,
                                                   const RowLayout &layout_p, bool external_p, bool flush_p)
    : rows(rows_p), heap(heap_p), layout(layout_p), read_state(*this), total_count(rows.count), total_scanned(0),
      external(external_p), flush(flush_p), unswizzling(!layout.AllConstant() && external && !heap.keep_pinned) {

	if (unswizzling) {
		D_ASSERT(rows.blocks.size() == heap.blocks.size());
	}

	ValidateUnscannedBlock();
}

void RowDataCollectionScanner::SwizzleBlock(RowDataBlock &data_block, RowDataBlock &heap_block) {
	// Pin the data block and swizzle the pointers within the rows
	D_ASSERT(!data_block.block->IsSwizzled());
	auto data_handle = rows.buffer_manager.Pin(data_block.block);
	auto data_ptr = data_handle.Ptr();
	RowOperations::SwizzleColumns(layout, data_ptr, data_block.count);
	data_block.block->SetSwizzling(nullptr);

	// Swizzle the heap pointers
	auto heap_handle = heap.buffer_manager.Pin(heap_block.block);
	auto heap_ptr = Load<data_ptr_t>(data_ptr + layout.GetHeapOffset());
	auto heap_offset = heap_ptr - heap_handle.Ptr();
	RowOperations::SwizzleHeapPointer(layout, data_ptr, heap_ptr, data_block.count, heap_offset);
}

void RowDataCollectionScanner::ReSwizzle() {
	if (rows.count == 0) {
		return;
	}

	if (!unswizzling) {
		// No swizzled blocks!
		return;
	}

	D_ASSERT(rows.blocks.size() == heap.blocks.size());
	for (idx_t i = 0; i < rows.blocks.size(); ++i) {
		auto &data_block = rows.blocks[i];
		if (data_block->block && !data_block->block->IsSwizzled()) {
			SwizzleBlock(*data_block, *heap.blocks[i]);
		}
	}
}

void RowDataCollectionScanner::ValidateUnscannedBlock() const {
	if (unswizzling && read_state.block_idx < rows.blocks.size()) {
		D_ASSERT(rows.blocks[read_state.block_idx]->block->IsSwizzled());
	}
}

void RowDataCollectionScanner::Scan(DataChunk &chunk) {
	auto count = MinValue((idx_t)STANDARD_VECTOR_SIZE, total_count - total_scanned);
	if (count == 0) {
		chunk.SetCardinality(count);
		return;
	}

	const idx_t &row_width = layout.GetRowWidth();
	// Set up a batch of pointers to scan data from
	idx_t scanned = 0;
	auto data_pointers = FlatVector::GetData<data_ptr_t>(addresses);

	// We must pin ALL blocks we are going to gather from
	vector<BufferHandle> pinned_blocks;
	while (scanned < count) {
		read_state.PinData();
		auto &data_block = rows.blocks[read_state.block_idx];
		idx_t next = MinValue(data_block->count - read_state.entry_idx, count - scanned);
		const data_ptr_t data_ptr = read_state.data_handle.Ptr() + read_state.entry_idx * row_width;
		// Set up the next pointers
		data_ptr_t row_ptr = data_ptr;
		for (idx_t i = 0; i < next; i++) {
			data_pointers[scanned + i] = row_ptr;
			row_ptr += row_width;
		}
		// Unswizzle the offsets back to pointers (if needed)
		if (unswizzling) {
			RowOperations::UnswizzlePointers(layout, data_ptr, read_state.heap_handle.Ptr(), next);
			rows.blocks[read_state.block_idx]->block->SetSwizzling("RowDataCollectionScanner::Scan");
		}
		// Update state indices
		read_state.entry_idx += next;
		if (read_state.entry_idx == data_block->count) {
			// Pin completed blocks so we don't lose them
			pinned_blocks.emplace_back(rows.buffer_manager.Pin(data_block->block));
			if (unswizzling) {
				auto &heap_block = heap.blocks[read_state.block_idx];
				pinned_blocks.emplace_back(heap.buffer_manager.Pin(heap_block->block));
			}
			read_state.block_idx++;
			read_state.entry_idx = 0;
			ValidateUnscannedBlock();
		}
		scanned += next;
	}
	D_ASSERT(scanned == count);
	// Deserialize the payload data
	for (idx_t col_no = 0; col_no < layout.ColumnCount(); col_no++) {
		RowOperations::Gather(addresses, *FlatVector::IncrementalSelectionVector(), chunk.data[col_no],
		                      *FlatVector::IncrementalSelectionVector(), count, layout, col_no);
	}
	chunk.SetCardinality(count);
	chunk.Verify();
	total_scanned += scanned;

	//	Switch to a new set of pinned blocks
	read_state.pinned_blocks.swap(pinned_blocks);

	if (flush) {
		// Release blocks we have passed.
		for (idx_t i = 0; i < read_state.block_idx; ++i) {
			rows.blocks[i]->block = nullptr;
			if (unswizzling) {
				heap.blocks[i]->block = nullptr;
			}
		}
	} else if (unswizzling) {
		// Reswizzle blocks we have passed so they can be flushed safely.
		for (idx_t i = 0; i < read_state.block_idx; ++i) {
			auto &data_block = rows.blocks[i];
			if (data_block->block && !data_block->block->IsSwizzled()) {
				SwizzleBlock(*data_block, *heap.blocks[i]);
			}
		}
	}
}

} // namespace duckdb
//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/common/types/row_layout.cpp
//
//
//===----------------------------------------------------------------------===//





namespace duckdb {

RowLayout::RowLayout()
    : flag_width(0), data_width(0), aggr_width(0), row_width(0), all_constant(true), heap_pointer_offset(0) {
}

void RowLayout::Initialize(vector<LogicalType> types_p, Aggregates aggregates_p, bool align) {
	offsets.clear();
	types = move(types_p);

	// Null mask at the front - 1 bit per value.
	flag_width = ValidityBytes::ValidityMaskSize(types.size());
	row_width = flag_width;

	// Whether all columns are constant size.
	for (const auto &type : types) {
		all_constant = all_constant && TypeIsConstantSize(type.InternalType());
	}

	// This enables pointer swizzling for out-of-core computation.
	if (!all_constant) {
		// When unswizzled the pointer lives here.
		// When swizzled, the pointer is replaced by an offset.
		heap_pointer_offset = row_width;
		// The 8 byte pointer will be replaced with an 8 byte idx_t when swizzled.
		// However, this cannot be sizeof(data_ptr_t), since 32 bit builds use 4 byte pointers.
		row_width += sizeof(idx_t);
	}

	// Data columns. No alignment required.
	for (const auto &type : types) {
		offsets.push_back(row_width);
		const auto internal_type = type.InternalType();
		if (TypeIsConstantSize(internal_type) || internal_type == PhysicalType::VARCHAR) {
			row_width += GetTypeIdSize(type.InternalType());
		} else {
			// Variable size types use pointers to the actual data (can be swizzled).
			// Again, we would use sizeof(data_ptr_t), but this is not guaranteed to be equal to sizeof(idx_t).
			row_width += sizeof(idx_t);
		}
	}

	// Alignment padding for aggregates
#ifndef DUCKDB_ALLOW_UNDEFINED
	if (align) {
		row_width = AlignValue(row_width);
	}
#endif
	data_width = row_width - flag_width;

	// Aggregate fields.
	aggregates = move(aggregates_p);
	for (auto &aggregate : aggregates) {
		offsets.push_back(row_width);
		row_width += aggregate.payload_size;
#ifndef DUCKDB_ALLOW_UNDEFINED
		D_ASSERT(aggregate.payload_size == AlignValue(aggregate.payload_size));
#endif
	}
	aggr_width = row_width - data_width - flag_width;

	// Alignment padding for the next row
#ifndef DUCKDB_ALLOW_UNDEFINED
	if (align) {
		row_width = AlignValue(row_width);
	}
#endif
}

void RowLayout::Initialize(vector<LogicalType> types_p, bool align) {
	Initialize(move(types_p), Aggregates(), align);
}

void RowLayout::Initialize(Aggregates aggregates_p, bool align) {
	Initialize(vector<LogicalType>(), move(aggregates_p), align);
}

} // namespace duckdb




namespace duckdb {

SelectionData::SelectionData(idx_t count) {
	owned_data = unique_ptr<sel_t[]>(new sel_t[count]);
#ifdef DEBUG
	for (idx_t i = 0; i < count; i++) {
		owned_data[i] = std::numeric_limits<sel_t>::max();
	}
#endif
}

// LCOV_EXCL_START
string SelectionVector::ToString(idx_t count) const {
	string result = "Selection Vector (" + to_string(count) + ") [";
	for (idx_t i = 0; i < count; i++) {
		if (i != 0) {
			result += ", ";
		}
		result += to_string(get_index(i));
	}
	result += "]";
	return result;
}

void SelectionVector::Print(idx_t count) const {
	Printer::Print(ToString(count));
}
// LCOV_EXCL_STOP

buffer_ptr<SelectionData> SelectionVector::Slice(const SelectionVector &sel, idx_t count) const {
	auto data = make_buffer<SelectionData>(count);
	auto result_ptr = data->owned_data.get();
	// for every element, we perform result[i] = target[new[i]]
	for (idx_t i = 0; i < count; i++) {
		auto new_idx = sel.get_index(i);
		auto idx = this->get_index(new_idx);
		result_ptr[i] = idx;
	}
	return data;
}

} // namespace duckdb







#include <cstring>

namespace duckdb {

StringHeap::StringHeap(Allocator &allocator) : allocator(allocator) {
}

void StringHeap::Destroy() {
	allocator.Destroy();
}

void StringHeap::Move(StringHeap &other) {
	other.allocator.Move(allocator);
}

string_t StringHeap::AddString(const char *data, idx_t len) {
	D_ASSERT(Utf8Proc::Analyze(data, len) != UnicodeType::INVALID);
	return AddBlob(data, len);
}

string_t StringHeap::AddString(const char *data) {
	return AddString(data, strlen(data));
}

string_t StringHeap::AddString(const string &data) {
	return AddString(data.c_str(), data.size());
}

string_t StringHeap::AddString(const string_t &data) {
	return AddString(data.GetDataUnsafe(), data.GetSize());
}

string_t StringHeap::AddBlob(const char *data, idx_t len) {
	auto insert_string = EmptyString(len);
	auto insert_pos = insert_string.GetDataWriteable();
	memcpy(insert_pos, data, len);
	insert_string.Finalize();
	return insert_string;
}

string_t StringHeap::AddBlob(const string_t &data) {
	return AddBlob(data.GetDataUnsafe(), data.GetSize());
}

string_t StringHeap::EmptyString(idx_t len) {
	D_ASSERT(len >= string_t::INLINE_LENGTH);
	auto insert_pos = (const char *)allocator.Allocate(len);
	return string_t(insert_pos, len);
}

} // namespace duckdb





namespace duckdb {

void string_t::Verify() const {
	auto dataptr = GetDataUnsafe();
	(void)dataptr;
	D_ASSERT(dataptr);

#ifdef DEBUG
	auto utf_type = Utf8Proc::Analyze(dataptr, GetSize());
	D_ASSERT(utf_type != UnicodeType::INVALID);
#endif

	// verify that the prefix contains the first four characters of the string
	for (idx_t i = 0; i < MinValue<uint32_t>(PREFIX_LENGTH, GetSize()); i++) {
		D_ASSERT(GetPrefix()[i] == dataptr[i]);
	}
	// verify that for strings with length < INLINE_LENGTH, the rest of the string is zero
	for (idx_t i = GetSize(); i < INLINE_LENGTH; i++) {
		D_ASSERT(GetDataUnsafe()[i] == '\0');
	}
}

} // namespace duckdb








#include <cstring>
#include <sstream>
#include <cctype>

namespace duckdb {

static_assert(sizeof(dtime_t) == sizeof(int64_t), "dtime_t was padded");

// string format is hh:mm:ss.microsecondsZ
// microseconds and Z are optional
// ISO 8601

bool Time::TryConvertInternal(const char *buf, idx_t len, idx_t &pos, dtime_t &result, bool strict) {
	int32_t hour = -1, min = -1, sec = -1, micros = -1;
	pos = 0;

	if (len == 0) {
		return false;
	}

	int sep;

	// skip leading spaces
	while (pos < len && StringUtil::CharacterIsSpace(buf[pos])) {
		pos++;
	}

	if (pos >= len) {
		return false;
	}

	if (!StringUtil::CharacterIsDigit(buf[pos])) {
		return false;
	}

	if (!Date::ParseDoubleDigit(buf, len, pos, hour)) {
		return false;
	}
	if (hour < 0 || hour >= 24) {
		return false;
	}

	if (pos >= len) {
		return false;
	}

	// fetch the separator
	sep = buf[pos++];
	if (sep != ':') {
		// invalid separator
		return false;
	}

	if (!Date::ParseDoubleDigit(buf, len, pos, min)) {
		return false;
	}
	if (min < 0 || min >= 60) {
		return false;
	}

	if (pos >= len) {
		return false;
	}

	if (buf[pos++] != sep) {
		return false;
	}

	if (!Date::ParseDoubleDigit(buf, len, pos, sec)) {
		return false;
	}
	if (sec < 0 || sec >= 60) {
		return false;
	}

	micros = 0;
	if (pos < len && buf[pos] == '.') {
		pos++;
		// we expect some microseconds
		int32_t mult = 100000;
		for (; pos < len && StringUtil::CharacterIsDigit(buf[pos]); pos++, mult /= 10) {
			if (mult > 0) {
				micros += (buf[pos] - '0') * mult;
			}
		}
	}

	// in strict mode, check remaining string for non-space characters
	if (strict) {
		// skip trailing spaces
		while (pos < len && StringUtil::CharacterIsSpace(buf[pos])) {
			pos++;
		}
		// check position. if end was not reached, non-space chars remaining
		if (pos < len) {
			return false;
		}
	}

	result = Time::FromTime(hour, min, sec, micros);
	return true;
}

bool Time::TryConvertTime(const char *buf, idx_t len, idx_t &pos, dtime_t &result, bool strict) {
	if (!Time::TryConvertInternal(buf, len, pos, result, strict)) {
		if (!strict) {
			// last chance, check if we can parse as timestamp
			timestamp_t timestamp;
			if (Timestamp::TryConvertTimestamp(buf, len, timestamp)) {
				result = Timestamp::GetTime(timestamp);
				return true;
			}
		}
		return false;
	}
	return true;
}

string Time::ConversionError(const string &str) {
	return StringUtil::Format("time field value out of range: \"%s\", "
	                          "expected format is ([YYYY-MM-DD ]HH:MM:SS[.MS])",
	                          str);
}

string Time::ConversionError(string_t str) {
	return Time::ConversionError(str.GetString());
}

dtime_t Time::FromCString(const char *buf, idx_t len, bool strict) {
	dtime_t result;
	idx_t pos;
	if (!Time::TryConvertTime(buf, len, pos, result, strict)) {
		throw ConversionException(ConversionError(string(buf, len)));
	}
	return result;
}

dtime_t Time::FromString(const string &str, bool strict) {
	return Time::FromCString(str.c_str(), str.size(), strict);
}

string Time::ToString(dtime_t time) {
	int32_t time_units[4];
	Time::Convert(time, time_units[0], time_units[1], time_units[2], time_units[3]);

	char micro_buffer[6];
	auto length = TimeToStringCast::Length(time_units, micro_buffer);
	auto buffer = unique_ptr<char[]>(new char[length]);
	TimeToStringCast::Format(buffer.get(), length, time_units, micro_buffer);
	return string(buffer.get(), length);
}

string Time::ToUTCOffset(int hour_offset, int minute_offset) {
	dtime_t time((hour_offset * Interval::MINS_PER_HOUR + minute_offset) * Interval::MICROS_PER_MINUTE);

	char buffer[1 + 2 + 1 + 2];
	idx_t length = 0;
	buffer[length++] = (time.micros < 0 ? '-' : '+');
	time.micros = std::abs(time.micros);

	int32_t time_units[4];
	Time::Convert(time, time_units[0], time_units[1], time_units[2], time_units[3]);

	TimeToStringCast::FormatTwoDigits(buffer + length, time_units[0]);
	length += 2;
	if (time_units[1]) {
		buffer[length++] = ':';
		TimeToStringCast::FormatTwoDigits(buffer + length, time_units[1]);
		length += 2;
	}

	return string(buffer, length);
}

dtime_t Time::FromTime(int32_t hour, int32_t minute, int32_t second, int32_t microseconds) {
	int64_t result;
	result = hour;                                             // hours
	result = result * Interval::MINS_PER_HOUR + minute;        // hours -> minutes
	result = result * Interval::SECS_PER_MINUTE + second;      // minutes -> seconds
	result = result * Interval::MICROS_PER_SEC + microseconds; // seconds -> microseconds
	return dtime_t(result);
}

// LCOV_EXCL_START
#ifdef DEBUG
static bool AssertValidTime(int32_t hour, int32_t minute, int32_t second, int32_t microseconds) {
	if (hour < 0 || hour >= 24) {
		return false;
	}
	if (minute < 0 || minute >= 60) {
		return false;
	}
	if (second < 0 || second > 60) {
		return false;
	}
	if (microseconds < 0 || microseconds > 1000000) {
		return false;
	}
	return true;
}
#endif
// LCOV_EXCL_STOP

void Time::Convert(dtime_t dtime, int32_t &hour, int32_t &min, int32_t &sec, int32_t &micros) {
	int64_t time = dtime.micros;
	hour = int32_t(time / Interval::MICROS_PER_HOUR);
	time -= int64_t(hour) * Interval::MICROS_PER_HOUR;
	min = int32_t(time / Interval::MICROS_PER_MINUTE);
	time -= int64_t(min) * Interval::MICROS_PER_MINUTE;
	sec = int32_t(time / Interval::MICROS_PER_SEC);
	time -= int64_t(sec) * Interval::MICROS_PER_SEC;
	micros = int32_t(time);
#ifdef DEBUG
	D_ASSERT(AssertValidTime(hour, min, sec, micros));
#endif
}

} // namespace duckdb











#include <ctime>

namespace duckdb {

static_assert(sizeof(timestamp_t) == sizeof(int64_t), "timestamp_t was padded");

// timestamp/datetime uses 64 bits, high 32 bits for date and low 32 bits for time
// string format is YYYY-MM-DDThh:mm:ssZ
// T may be a space
// Z is optional
// ISO 8601
static inline bool CharacterIsTimeZone(char c) {
	return StringUtil::CharacterIsAlpha(c) || StringUtil::CharacterIsDigit(c) || c == '_' || c == '/';
}

bool Timestamp::TryConvertTimestampTZ(const char *str, idx_t len, timestamp_t &result, bool &has_offset, string_t &tz) {
	idx_t pos;
	date_t date;
	dtime_t time;
	has_offset = false;
	if (!Date::TryConvertDate(str, len, pos, date, has_offset)) {
		return false;
	}
	if (pos == len) {
		// no time: only a date or special
		if (date == date_t::infinity()) {
			result = timestamp_t::infinity();
			return true;
		} else if (date == date_t::ninfinity()) {
			result = timestamp_t::ninfinity();
			return true;
		}
		return Timestamp::TryFromDatetime(date, dtime_t(0), result);
	}
	// try to parse a time field
	if (str[pos] == ' ' || str[pos] == 'T') {
		pos++;
	}
	idx_t time_pos = 0;
	if (!Time::TryConvertTime(str + pos, len - pos, time_pos, time)) {
		return false;
	}
	pos += time_pos;
	if (!Timestamp::TryFromDatetime(date, time, result)) {
		return false;
	}
	if (pos < len) {
		// skip a "Z" at the end (as per the ISO8601 specs)
		int hour_offset, minute_offset;
		if (str[pos] == 'Z') {
			pos++;
			has_offset = true;
		} else if (Timestamp::TryParseUTCOffset(str, pos, len, hour_offset, minute_offset)) {
			result -= hour_offset * Interval::MICROS_PER_HOUR + minute_offset * Interval::MICROS_PER_MINUTE;
			has_offset = true;
		} else {
			// Parse a time zone: / [A-Za-z0-9/_]+/
			if (str[pos++] != ' ') {
				return false;
			}
			auto tz_name = str + pos;
			for (; pos < len && CharacterIsTimeZone(str[pos]); ++pos) {
				continue;
			}
			auto tz_len = str + pos - tz_name;
			if (tz_len) {
				tz = string_t(tz_name, tz_len);
			}
			// Note that the caller must reinterpret the instant we return to the given time zone
		}

		// skip any spaces at the end
		while (pos < len && StringUtil::CharacterIsSpace(str[pos])) {
			pos++;
		}
		if (pos < len) {
			return false;
		}
	}
	return true;
}

bool Timestamp::TryConvertTimestamp(const char *str, idx_t len, timestamp_t &result) {
	string_t tz(nullptr, 0);
	bool has_offset = false;
	// We don't understand TZ without an extension, so fail if one was provided.
	return TryConvertTimestampTZ(str, len, result, has_offset, tz) && !tz.GetSize();
}

string Timestamp::ConversionError(const string &str) {
	return StringUtil::Format("timestamp field value out of range: \"%s\", "
	                          "expected format is (YYYY-MM-DD HH:MM:SS[.US][±HH:MM| ZONE])",
	                          str);
}

string Timestamp::ConversionError(string_t str) {
	return Timestamp::ConversionError(str.GetString());
}

timestamp_t Timestamp::FromCString(const char *str, idx_t len) {
	timestamp_t result;
	if (!Timestamp::TryConvertTimestamp(str, len, result)) {
		throw ConversionException(Timestamp::ConversionError(string(str, len)));
	}
	return result;
}

bool Timestamp::TryParseUTCOffset(const char *str, idx_t &pos, idx_t len, int &hour_offset, int &minute_offset) {
	minute_offset = 0;
	idx_t curpos = pos;
	// parse the next 3 characters
	if (curpos + 3 > len) {
		// no characters left to parse
		return false;
	}
	char sign_char = str[curpos];
	if (sign_char != '+' && sign_char != '-') {
		// expected either + or -
		return false;
	}
	curpos++;
	if (!StringUtil::CharacterIsDigit(str[curpos]) || !StringUtil::CharacterIsDigit(str[curpos + 1])) {
		// expected +HH or -HH
		return false;
	}
	hour_offset = (str[curpos] - '0') * 10 + (str[curpos + 1] - '0');
	if (sign_char == '-') {
		hour_offset = -hour_offset;
	}
	curpos += 2;

	// optional minute specifier: expected either "MM" or ":MM"
	if (curpos >= len) {
		// done, nothing left
		pos = curpos;
		return true;
	}
	if (str[curpos] == ':') {
		curpos++;
	}
	if (curpos + 2 > len || !StringUtil::CharacterIsDigit(str[curpos]) ||
	    !StringUtil::CharacterIsDigit(str[curpos + 1])) {
		// no MM specifier
		pos = curpos;
		return true;
	}
	// we have an MM specifier: parse it
	minute_offset = (str[curpos] - '0') * 10 + (str[curpos + 1] - '0');
	if (sign_char == '-') {
		minute_offset = -minute_offset;
	}
	pos = curpos + 2;
	return true;
}

timestamp_t Timestamp::FromString(const string &str) {
	return Timestamp::FromCString(str.c_str(), str.size());
}

string Timestamp::ToString(timestamp_t timestamp) {
	if (timestamp == timestamp_t::infinity()) {
		return Date::PINF;
	} else if (timestamp == timestamp_t::ninfinity()) {
		return Date::NINF;
	}
	date_t date;
	dtime_t time;
	Timestamp::Convert(timestamp, date, time);
	return Date::ToString(date) + " " + Time::ToString(time);
}

date_t Timestamp::GetDate(timestamp_t timestamp) {
	if (timestamp == timestamp_t::infinity()) {
		return date_t::infinity();
	} else if (timestamp == timestamp_t::ninfinity()) {
		return date_t::ninfinity();
	}
	return date_t((timestamp.value + (timestamp.value < 0)) / Interval::MICROS_PER_DAY - (timestamp.value < 0));
}

dtime_t Timestamp::GetTime(timestamp_t timestamp) {
	if (!IsFinite(timestamp)) {
		throw ConversionException("Can't get TIME of infinite TIMESTAMP");
	}
	date_t date = Timestamp::GetDate(timestamp);
	return dtime_t(timestamp.value - (int64_t(date.days) * int64_t(Interval::MICROS_PER_DAY)));
}

bool Timestamp::TryFromDatetime(date_t date, dtime_t time, timestamp_t &result) {
	if (!TryMultiplyOperator::Operation<int64_t, int64_t, int64_t>(date.days, Interval::MICROS_PER_DAY, result.value)) {
		return false;
	}
	if (!TryAddOperator::Operation<int64_t, int64_t, int64_t>(result.value, time.micros, result.value)) {
		return false;
	}
	return Timestamp::IsFinite(result);
}

timestamp_t Timestamp::FromDatetime(date_t date, dtime_t time) {
	timestamp_t result;
	if (!TryFromDatetime(date, time, result)) {
		throw Exception("Overflow exception in date/time -> timestamp conversion");
	}
	return result;
}

void Timestamp::Convert(timestamp_t timestamp, date_t &out_date, dtime_t &out_time) {
	out_date = GetDate(timestamp);
	int64_t days_micros;
	if (!TryMultiplyOperator::Operation<int64_t, int64_t, int64_t>(out_date.days, Interval::MICROS_PER_DAY,
	                                                               days_micros)) {
		throw ConversionException("Date out of range in timestamp conversion");
	}
	out_time = dtime_t(timestamp.value - days_micros);
	D_ASSERT(timestamp == Timestamp::FromDatetime(out_date, out_time));
}

timestamp_t Timestamp::GetCurrentTimestamp() {
	auto now = system_clock::now();
	auto epoch_ms = duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
	return Timestamp::FromEpochMs(epoch_ms);
}

timestamp_t Timestamp::FromEpochSeconds(int64_t sec) {
	int64_t result;
	if (!TryMultiplyOperator::Operation(sec, Interval::MICROS_PER_SEC, result)) {
		throw ConversionException("Could not convert Timestamp(S) to Timestamp(US)");
	}
	return timestamp_t(result);
}

timestamp_t Timestamp::FromEpochMs(int64_t ms) {
	int64_t result;
	if (!TryMultiplyOperator::Operation(ms, Interval::MICROS_PER_MSEC, result)) {
		throw ConversionException("Could not convert Timestamp(MS) to Timestamp(US)");
	}
	return timestamp_t(result);
}

timestamp_t Timestamp::FromEpochMicroSeconds(int64_t micros) {
	return timestamp_t(micros);
}

timestamp_t Timestamp::FromEpochNanoSeconds(int64_t ns) {
	return timestamp_t(ns / 1000);
}

int64_t Timestamp::GetEpochSeconds(timestamp_t timestamp) {
	return timestamp.value / Interval::MICROS_PER_SEC;
}

int64_t Timestamp::GetEpochMs(timestamp_t timestamp) {
	return timestamp.value / Interval::MICROS_PER_MSEC;
}

int64_t Timestamp::GetEpochMicroSeconds(timestamp_t timestamp) {
	return timestamp.value;
}

int64_t Timestamp::GetEpochNanoSeconds(timestamp_t timestamp) {
	int64_t result;
	int64_t ns_in_us = 1000;
	if (!TryMultiplyOperator::Operation(timestamp.value, ns_in_us, result)) {
		throw ConversionException("Could not convert Timestamp(US) to Timestamp(NS)");
	}
	return result;
}

} // namespace duckdb



namespace duckdb {

bool UUID::FromString(string str, hugeint_t &result) {
	auto hex2char = [](char ch) -> unsigned char {
		if (ch >= '0' && ch <= '9') {
			return ch - '0';
		}
		if (ch >= 'a' && ch <= 'f') {
			return 10 + ch - 'a';
		}
		if (ch >= 'A' && ch <= 'F') {
			return 10 + ch - 'A';
		}
		return 0;
	};
	auto is_hex = [](char ch) -> bool {
		return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f') || (ch >= 'A' && ch <= 'F');
	};

	if (str.empty()) {
		return false;
	}
	int has_braces = 0;
	if (str.front() == '{') {
		has_braces = 1;
	}
	if (has_braces && str.back() != '}') {
		return false;
	}

	result.lower = 0;
	result.upper = 0;
	size_t count = 0;
	for (size_t i = has_braces; i < str.size() - has_braces; ++i) {
		if (str[i] == '-') {
			continue;
		}
		if (count >= 32 || !is_hex(str[i])) {
			return false;
		}
		if (count >= 16) {
			result.lower = (result.lower << 4) | hex2char(str[i]);
		} else {
			result.upper = (result.upper << 4) | hex2char(str[i]);
		}
		count++;
	}
	// Flip the first bit to make `order by uuid` same as `order by uuid::varchar`
	result.upper ^= (int64_t(1) << 63);
	return count == 32;
}

void UUID::ToString(hugeint_t input, char *buf) {
	auto byte_to_hex = [](char byte_val, char *buf, idx_t &pos) {
		static char const HEX_DIGITS[] = "0123456789abcdef";
		buf[pos++] = HEX_DIGITS[(byte_val >> 4) & 0xf];
		buf[pos++] = HEX_DIGITS[byte_val & 0xf];
	};

	// Flip back before convert to string
	int64_t upper = input.upper ^ (int64_t(1) << 63);
	idx_t pos = 0;
	byte_to_hex(upper >> 56 & 0xFF, buf, pos);
	byte_to_hex(upper >> 48 & 0xFF, buf, pos);
	byte_to_hex(upper >> 40 & 0xFF, buf, pos);
	byte_to_hex(upper >> 32 & 0xFF, buf, pos);
	buf[pos++] = '-';
	byte_to_hex(upper >> 24 & 0xFF, buf, pos);
	byte_to_hex(upper >> 16 & 0xFF, buf, pos);
	buf[pos++] = '-';
	byte_to_hex(upper >> 8 & 0xFF, buf, pos);
	byte_to_hex(upper & 0xFF, buf, pos);
	buf[pos++] = '-';
	byte_to_hex(input.lower >> 56 & 0xFF, buf, pos);
	byte_to_hex(input.lower >> 48 & 0xFF, buf, pos);
	buf[pos++] = '-';
	byte_to_hex(input.lower >> 40 & 0xFF, buf, pos);
	byte_to_hex(input.lower >> 32 & 0xFF, buf, pos);
	byte_to_hex(input.lower >> 24 & 0xFF, buf, pos);
	byte_to_hex(input.lower >> 16 & 0xFF, buf, pos);
	byte_to_hex(input.lower >> 8 & 0xFF, buf, pos);
	byte_to_hex(input.lower & 0xFF, buf, pos);
}

hugeint_t UUID::GenerateRandomUUID(RandomEngine &engine) {
	uint8_t bytes[16];
	for (int i = 0; i < 16; i += 4) {
		*reinterpret_cast<uint32_t *>(bytes + i) = engine.NextRandomInteger();
	}
	// variant must be 10xxxxxx
	bytes[8] &= 0xBF;
	bytes[8] |= 0x80;
	// version must be 0100xxxx
	bytes[6] &= 0x4F;
	bytes[6] |= 0x40;

	hugeint_t result;
	result.upper = 0;
	result.upper |= ((int64_t)bytes[0] << 56);
	result.upper |= ((int64_t)bytes[1] << 48);
	result.upper |= ((int64_t)bytes[2] << 40);
	result.upper |= ((int64_t)bytes[3] << 32);
	result.upper |= ((int64_t)bytes[4] << 24);
	result.upper |= ((int64_t)bytes[5] << 16);
	result.upper |= ((int64_t)bytes[6] << 8);
	result.upper |= bytes[7];
	result.lower = 0;
	result.lower |= ((uint64_t)bytes[8] << 56);
	result.lower |= ((uint64_t)bytes[9] << 48);
	result.lower |= ((uint64_t)bytes[10] << 40);
	result.lower |= ((uint64_t)bytes[11] << 32);
	result.lower |= ((uint64_t)bytes[12] << 24);
	result.lower |= ((uint64_t)bytes[13] << 16);
	result.lower |= ((uint64_t)bytes[14] << 8);
	result.lower |= bytes[15];
	return result;
}

hugeint_t UUID::GenerateRandomUUID() {
	RandomEngine engine;
	return GenerateRandomUUID(engine);
}

} // namespace duckdb


namespace duckdb {

ValidityData::ValidityData(idx_t count) : TemplatedValidityData(count) {
}
ValidityData::ValidityData(const ValidityMask &original, idx_t count)
    : TemplatedValidityData(original.GetData(), count) {
}

void ValidityMask::Combine(const ValidityMask &other, idx_t count) {
	if (other.AllValid()) {
		// X & 1 = X
		return;
	}
	if (AllValid()) {
		// 1 & Y = Y
		Initialize(other);
		return;
	}
	if (validity_mask == other.validity_mask) {
		// X & X == X
		return;
	}
	// have to merge
	// create a new validity mask that contains the combined mask
	auto owned_data = move(validity_data);
	auto data = GetData();
	auto other_data = other.GetData();

	Initialize(count);
	auto result_data = GetData();

	auto entry_count = ValidityData::EntryCount(count);
	for (idx_t entry_idx = 0; entry_idx < entry_count; entry_idx++) {
		result_data[entry_idx] = data[entry_idx] & other_data[entry_idx];
	}
}

// LCOV_EXCL_START
string ValidityMask::ToString(idx_t count) const {
	string result = "Validity Mask (" + to_string(count) + ") [";
	for (idx_t i = 0; i < count; i++) {
		result += RowIsValid(i) ? "." : "X";
	}
	result += "]";
	return result;
}
// LCOV_EXCL_STOP

void ValidityMask::Resize(idx_t old_size, idx_t new_size) {
	D_ASSERT(new_size >= old_size);
	if (validity_mask) {
		auto new_size_count = EntryCount(new_size);
		auto old_size_count = EntryCount(old_size);
		auto new_validity_data = make_buffer<ValidityBuffer>(new_size);
		auto new_owned_data = new_validity_data->owned_data.get();
		for (idx_t entry_idx = 0; entry_idx < old_size_count; entry_idx++) {
			new_owned_data[entry_idx] = validity_mask[entry_idx];
		}
		for (idx_t entry_idx = old_size_count; entry_idx < new_size_count; entry_idx++) {
			new_owned_data[entry_idx] = ValidityData::MAX_ENTRY;
		}
		validity_data = move(new_validity_data);
		validity_mask = validity_data->owned_data.get();
	} else {
		Initialize(new_size);
	}
}

void ValidityMask::Slice(const ValidityMask &other, idx_t offset, idx_t end) {
	if (other.AllValid()) {
		validity_mask = nullptr;
		validity_data.reset();
		return;
	}
	if (offset == 0) {
		Initialize(other);
		return;
	}
	ValidityMask new_mask(end - offset);

// FIXME THIS NEEDS FIXING!
#if 1
	for (idx_t i = offset; i < end; i++) {
		new_mask.Set(i - offset, other.RowIsValid(i));
	}
	Initialize(new_mask);
#else
	// first shift the "whole" units
	idx_t entire_units = offset / BITS_PER_VALUE;
	idx_t sub_units = offset - entire_units * BITS_PER_VALUE;
	if (entire_units > 0) {
		idx_t validity_idx;
		for (validity_idx = 0; validity_idx + entire_units < STANDARD_ENTRY_COUNT; validity_idx++) {
			new_mask.validity_mask[validity_idx] = other.validity_mask[validity_idx + entire_units];
		}
	}
	// now we shift the remaining sub units
	// this gets a bit more complicated because we have to shift over the borders of the entries
	// e.g. suppose we have 2 entries of length 4 and we left-shift by two
	// 0101|1010
	// a regular left-shift of both gets us:
	// 0100|1000
	// we then OR the overflow (right-shifted by BITS_PER_VALUE - offset) together to get the correct result
	// 0100|1000 ->
	// 0110|1000
	if (sub_units > 0) {
		idx_t validity_idx;
		for (validity_idx = 0; validity_idx + 1 < STANDARD_ENTRY_COUNT; validity_idx++) {
			new_mask.validity_mask[validity_idx] =
			    (other.validity_mask[validity_idx] >> sub_units) |
			    (other.validity_mask[validity_idx + 1] << (BITS_PER_VALUE - sub_units));
		}
		new_mask.validity_mask[validity_idx] >>= sub_units;
	}
#ifdef DEBUG
	for (idx_t i = offset; i < STANDARD_VECTOR_SIZE; i++) {
		D_ASSERT(new_mask.RowIsValid(i - offset) == other.RowIsValid(i));
	}
	Initialize(new_mask);
#endif
#endif
}

} // namespace duckdb































#include <utility>
#include <cmath>

namespace duckdb {

Value::Value(LogicalType type) : type_(move(type)), is_null(true) {
}

Value::Value(int32_t val) : type_(LogicalType::INTEGER), is_null(false) {
	value_.integer = val;
}

Value::Value(int64_t val) : type_(LogicalType::BIGINT), is_null(false) {
	value_.bigint = val;
}

Value::Value(float val) : type_(LogicalType::FLOAT), is_null(false) {
	value_.float_ = val;
}

Value::Value(double val) : type_(LogicalType::DOUBLE), is_null(false) {
	value_.double_ = val;
}

Value::Value(const char *val) : Value(val ? string(val) : string()) {
}

Value::Value(std::nullptr_t val) : Value(LogicalType::VARCHAR) {
}

Value::Value(string_t val) : Value(string(val.GetDataUnsafe(), val.GetSize())) {
}

Value::Value(string val) : type_(LogicalType::VARCHAR), is_null(false), str_value(move(val)) {
	if (!Value::StringIsValid(str_value.c_str(), str_value.size())) {
		throw Exception(ErrorManager::InvalidUnicodeError(str_value, "value construction"));
	}
}

Value::~Value() {
}

Value::Value(const Value &other)
    : type_(other.type_), is_null(other.is_null), value_(other.value_), str_value(other.str_value),
      struct_value(other.struct_value), list_value(other.list_value) {
}

Value::Value(Value &&other) noexcept
    : type_(move(other.type_)), is_null(other.is_null), value_(other.value_), str_value(move(other.str_value)),
      struct_value(move(other.struct_value)), list_value(move(other.list_value)) {
}

Value &Value::operator=(const Value &other) {
	type_ = other.type_;
	is_null = other.is_null;
	value_ = other.value_;
	str_value = other.str_value;
	struct_value = other.struct_value;
	list_value = other.list_value;
	return *this;
}

Value &Value::operator=(Value &&other) noexcept {
	type_ = move(other.type_);
	is_null = other.is_null;
	value_ = other.value_;
	str_value = move(other.str_value);
	struct_value = move(other.struct_value);
	list_value = move(other.list_value);
	return *this;
}

Value Value::MinimumValue(const LogicalType &type) {
	switch (type.id()) {
	case LogicalTypeId::BOOLEAN:
		return Value::BOOLEAN(false);
	case LogicalTypeId::TINYINT:
		return Value::TINYINT(NumericLimits<int8_t>::Minimum());
	case LogicalTypeId::SMALLINT:
		return Value::SMALLINT(NumericLimits<int16_t>::Minimum());
	case LogicalTypeId::INTEGER:
	case LogicalTypeId::SQLNULL:
		return Value::INTEGER(NumericLimits<int32_t>::Minimum());
	case LogicalTypeId::BIGINT:
		return Value::BIGINT(NumericLimits<int64_t>::Minimum());
	case LogicalTypeId::HUGEINT:
		return Value::HUGEINT(NumericLimits<hugeint_t>::Minimum());
	case LogicalTypeId::UUID:
		return Value::UUID(NumericLimits<hugeint_t>::Minimum());
	case LogicalTypeId::UTINYINT:
		return Value::UTINYINT(NumericLimits<uint8_t>::Minimum());
	case LogicalTypeId::USMALLINT:
		return Value::USMALLINT(NumericLimits<uint16_t>::Minimum());
	case LogicalTypeId::UINTEGER:
		return Value::UINTEGER(NumericLimits<uint32_t>::Minimum());
	case LogicalTypeId::UBIGINT:
		return Value::UBIGINT(NumericLimits<uint64_t>::Minimum());
	case LogicalTypeId::DATE:
		return Value::DATE(Date::FromDate(Date::DATE_MIN_YEAR, Date::DATE_MIN_MONTH, Date::DATE_MIN_DAY));
	case LogicalTypeId::TIME:
		return Value::TIME(dtime_t(0));
	case LogicalTypeId::TIMESTAMP:
		return Value::TIMESTAMP(Date::FromDate(Timestamp::MIN_YEAR, Timestamp::MIN_MONTH, Timestamp::MIN_DAY),
		                        dtime_t(0));
	case LogicalTypeId::TIMESTAMP_SEC:
		return MinimumValue(LogicalType::TIMESTAMP).DefaultCastAs(LogicalType::TIMESTAMP_S);
	case LogicalTypeId::TIMESTAMP_MS:
		return MinimumValue(LogicalType::TIMESTAMP).DefaultCastAs(LogicalType::TIMESTAMP_MS);
	case LogicalTypeId::TIMESTAMP_NS:
		return Value::TIMESTAMPNS(timestamp_t(NumericLimits<int64_t>::Minimum()));
	case LogicalTypeId::TIME_TZ:
		return Value::TIMETZ(dtime_t(0));
	case LogicalTypeId::TIMESTAMP_TZ:
		return Value::TIMESTAMPTZ(Timestamp::FromDatetime(
		    Date::FromDate(Timestamp::MIN_YEAR, Timestamp::MIN_MONTH, Timestamp::MIN_DAY), dtime_t(0)));
	case LogicalTypeId::FLOAT:
		return Value::FLOAT(NumericLimits<float>::Minimum());
	case LogicalTypeId::DOUBLE:
		return Value::DOUBLE(NumericLimits<double>::Minimum());
	case LogicalTypeId::DECIMAL: {
		auto width = DecimalType::GetWidth(type);
		auto scale = DecimalType::GetScale(type);
		switch (type.InternalType()) {
		case PhysicalType::INT16:
			return Value::DECIMAL(int16_t(-NumericHelper::POWERS_OF_TEN[width] + 1), width, scale);
		case PhysicalType::INT32:
			return Value::DECIMAL(int32_t(-NumericHelper::POWERS_OF_TEN[width] + 1), width, scale);
		case PhysicalType::INT64:
			return Value::DECIMAL(int64_t(-NumericHelper::POWERS_OF_TEN[width] + 1), width, scale);
		case PhysicalType::INT128:
			return Value::DECIMAL(-Hugeint::POWERS_OF_TEN[width] + 1, width, scale);
		default:
			throw InternalException("Unknown decimal type");
		}
	}
	case LogicalTypeId::ENUM:
		return Value::ENUM(0, type);
	default:
		throw InvalidTypeException(type, "MinimumValue requires numeric type");
	}
}

Value Value::MaximumValue(const LogicalType &type) {
	switch (type.id()) {
	case LogicalTypeId::BOOLEAN:
		return Value::BOOLEAN(true);
	case LogicalTypeId::TINYINT:
		return Value::TINYINT(NumericLimits<int8_t>::Maximum());
	case LogicalTypeId::SMALLINT:
		return Value::SMALLINT(NumericLimits<int16_t>::Maximum());
	case LogicalTypeId::INTEGER:
	case LogicalTypeId::SQLNULL:
		return Value::INTEGER(NumericLimits<int32_t>::Maximum());
	case LogicalTypeId::BIGINT:
		return Value::BIGINT(NumericLimits<int64_t>::Maximum());
	case LogicalTypeId::HUGEINT:
		return Value::HUGEINT(NumericLimits<hugeint_t>::Maximum());
	case LogicalTypeId::UUID:
		return Value::UUID(NumericLimits<hugeint_t>::Maximum());
	case LogicalTypeId::UTINYINT:
		return Value::UTINYINT(NumericLimits<uint8_t>::Maximum());
	case LogicalTypeId::USMALLINT:
		return Value::USMALLINT(NumericLimits<uint16_t>::Maximum());
	case LogicalTypeId::UINTEGER:
		return Value::UINTEGER(NumericLimits<uint32_t>::Maximum());
	case LogicalTypeId::UBIGINT:
		return Value::UBIGINT(NumericLimits<uint64_t>::Maximum());
	case LogicalTypeId::DATE:
		return Value::DATE(Date::FromDate(Date::DATE_MAX_YEAR, Date::DATE_MAX_MONTH, Date::DATE_MAX_DAY));
	case LogicalTypeId::TIME:
		return Value::TIME(dtime_t(Interval::SECS_PER_DAY * Interval::MICROS_PER_SEC - 1));
	case LogicalTypeId::TIMESTAMP:
		return Value::TIMESTAMP(timestamp_t(NumericLimits<int64_t>::Maximum() - 1));
	case LogicalTypeId::TIMESTAMP_MS:
		return MaximumValue(LogicalType::TIMESTAMP).DefaultCastAs(LogicalType::TIMESTAMP_MS);
	case LogicalTypeId::TIMESTAMP_NS:
		return Value::TIMESTAMPNS(timestamp_t(NumericLimits<int64_t>::Maximum() - 1));
	case LogicalTypeId::TIMESTAMP_SEC:
		return MaximumValue(LogicalType::TIMESTAMP).DefaultCastAs(LogicalType::TIMESTAMP_S);
	case LogicalTypeId::TIME_TZ:
		return Value::TIMETZ(dtime_t(Interval::SECS_PER_DAY * Interval::MICROS_PER_SEC - 1));
	case LogicalTypeId::TIMESTAMP_TZ:
		return MaximumValue(LogicalType::TIMESTAMP);
	case LogicalTypeId::FLOAT:
		return Value::FLOAT(NumericLimits<float>::Maximum());
	case LogicalTypeId::DOUBLE:
		return Value::DOUBLE(NumericLimits<double>::Maximum());
	case LogicalTypeId::DECIMAL: {
		auto width = DecimalType::GetWidth(type);
		auto scale = DecimalType::GetScale(type);
		switch (type.InternalType()) {
		case PhysicalType::INT16:
			return Value::DECIMAL(int16_t(NumericHelper::POWERS_OF_TEN[width] - 1), width, scale);
		case PhysicalType::INT32:
			return Value::DECIMAL(int32_t(NumericHelper::POWERS_OF_TEN[width] - 1), width, scale);
		case PhysicalType::INT64:
			return Value::DECIMAL(int64_t(NumericHelper::POWERS_OF_TEN[width] - 1), width, scale);
		case PhysicalType::INT128:
			return Value::DECIMAL(Hugeint::POWERS_OF_TEN[width] - 1, width, scale);
		default:
			throw InternalException("Unknown decimal type");
		}
	}
	case LogicalTypeId::ENUM:
		return Value::ENUM(EnumType::GetSize(type) - 1, type);
	default:
		throw InvalidTypeException(type, "MaximumValue requires numeric type");
	}
}

Value Value::BOOLEAN(int8_t value) {
	Value result(LogicalType::BOOLEAN);
	result.value_.boolean = value ? true : false;
	result.is_null = false;
	return result;
}

Value Value::TINYINT(int8_t value) {
	Value result(LogicalType::TINYINT);
	result.value_.tinyint = value;
	result.is_null = false;
	return result;
}

Value Value::SMALLINT(int16_t value) {
	Value result(LogicalType::SMALLINT);
	result.value_.smallint = value;
	result.is_null = false;
	return result;
}

Value Value::INTEGER(int32_t value) {
	Value result(LogicalType::INTEGER);
	result.value_.integer = value;
	result.is_null = false;
	return result;
}

Value Value::BIGINT(int64_t value) {
	Value result(LogicalType::BIGINT);
	result.value_.bigint = value;
	result.is_null = false;
	return result;
}

Value Value::HUGEINT(hugeint_t value) {
	Value result(LogicalType::HUGEINT);
	result.value_.hugeint = value;
	result.is_null = false;
	return result;
}

Value Value::UUID(hugeint_t value) {
	Value result(LogicalType::UUID);
	result.value_.hugeint = value;
	result.is_null = false;
	return result;
}

Value Value::UUID(const string &value) {
	Value result(LogicalType::UUID);
	result.value_.hugeint = UUID::FromString(value);
	result.is_null = false;
	return result;
}

Value Value::UTINYINT(uint8_t value) {
	Value result(LogicalType::UTINYINT);
	result.value_.utinyint = value;
	result.is_null = false;
	return result;
}

Value Value::USMALLINT(uint16_t value) {
	Value result(LogicalType::USMALLINT);
	result.value_.usmallint = value;
	result.is_null = false;
	return result;
}

Value Value::UINTEGER(uint32_t value) {
	Value result(LogicalType::UINTEGER);
	result.value_.uinteger = value;
	result.is_null = false;
	return result;
}

Value Value::UBIGINT(uint64_t value) {
	Value result(LogicalType::UBIGINT);
	result.value_.ubigint = value;
	result.is_null = false;
	return result;
}

bool Value::FloatIsFinite(float value) {
	return !(std::isnan(value) || std::isinf(value));
}

bool Value::DoubleIsFinite(double value) {
	return !(std::isnan(value) || std::isinf(value));
}

template <>
bool Value::IsNan(float input) {
	return std::isnan(input);
}

template <>
bool Value::IsNan(double input) {
	return std::isnan(input);
}

template <>
bool Value::IsFinite(float input) {
	return Value::FloatIsFinite(input);
}

template <>
bool Value::IsFinite(double input) {
	return Value::DoubleIsFinite(input);
}

template <>
bool Value::IsFinite(date_t input) {
	return Date::IsFinite(input);
}

template <>
bool Value::IsFinite(timestamp_t input) {
	return Timestamp::IsFinite(input);
}

bool Value::StringIsValid(const char *str, idx_t length) {
	auto utf_type = Utf8Proc::Analyze(str, length);
	return utf_type != UnicodeType::INVALID;
}

Value Value::DECIMAL(int16_t value, uint8_t width, uint8_t scale) {
	D_ASSERT(width <= Decimal::MAX_WIDTH_INT16);
	Value result(LogicalType::DECIMAL(width, scale));
	result.value_.smallint = value;
	result.is_null = false;
	return result;
}

Value Value::DECIMAL(int32_t value, uint8_t width, uint8_t scale) {
	D_ASSERT(width >= Decimal::MAX_WIDTH_INT16 && width <= Decimal::MAX_WIDTH_INT32);
	Value result(LogicalType::DECIMAL(width, scale));
	result.value_.integer = value;
	result.is_null = false;
	return result;
}

Value Value::DECIMAL(int64_t value, uint8_t width, uint8_t scale) {
	auto decimal_type = LogicalType::DECIMAL(width, scale);
	Value result(decimal_type);
	switch (decimal_type.InternalType()) {
	case PhysicalType::INT16:
		result.value_.smallint = value;
		break;
	case PhysicalType::INT32:
		result.value_.integer = value;
		break;
	case PhysicalType::INT64:
		result.value_.bigint = value;
		break;
	default:
		result.value_.hugeint = value;
		break;
	}
	result.type_.Verify();
	result.is_null = false;
	return result;
}

Value Value::DECIMAL(hugeint_t value, uint8_t width, uint8_t scale) {
	D_ASSERT(width >= Decimal::MAX_WIDTH_INT64 && width <= Decimal::MAX_WIDTH_INT128);
	Value result(LogicalType::DECIMAL(width, scale));
	result.value_.hugeint = value;
	result.is_null = false;
	return result;
}

Value Value::FLOAT(float value) {
	Value result(LogicalType::FLOAT);
	result.value_.float_ = value;
	result.is_null = false;
	return result;
}

Value Value::DOUBLE(double value) {
	Value result(LogicalType::DOUBLE);
	result.value_.double_ = value;
	result.is_null = false;
	return result;
}

Value Value::HASH(hash_t value) {
	Value result(LogicalType::HASH);
	result.value_.hash = value;
	result.is_null = false;
	return result;
}

Value Value::POINTER(uintptr_t value) {
	Value result(LogicalType::POINTER);
	result.value_.pointer = value;
	result.is_null = false;
	return result;
}

Value Value::DATE(date_t value) {
	Value result(LogicalType::DATE);
	result.value_.date = value;
	result.is_null = false;
	return result;
}

Value Value::DATE(int32_t year, int32_t month, int32_t day) {
	return Value::DATE(Date::FromDate(year, month, day));
}

Value Value::TIME(dtime_t value) {
	Value result(LogicalType::TIME);
	result.value_.time = value;
	result.is_null = false;
	return result;
}

Value Value::TIMETZ(dtime_t value) {
	Value result(LogicalType::TIME_TZ);
	result.value_.time = value;
	result.is_null = false;
	return result;
}

Value Value::TIME(int32_t hour, int32_t min, int32_t sec, int32_t micros) {
	return Value::TIME(Time::FromTime(hour, min, sec, micros));
}

Value Value::TIMESTAMP(timestamp_t value) {
	Value result(LogicalType::TIMESTAMP);
	result.value_.timestamp = value;
	result.is_null = false;
	return result;
}

Value Value::TIMESTAMPTZ(timestamp_t value) {
	Value result(LogicalType::TIMESTAMP_TZ);
	result.value_.timestamp = value;
	result.is_null = false;
	return result;
}

Value Value::TIMESTAMPNS(timestamp_t timestamp) {
	Value result(LogicalType::TIMESTAMP_NS);
	result.value_.timestamp = timestamp;
	result.is_null = false;
	return result;
}

Value Value::TIMESTAMPMS(timestamp_t timestamp) {
	Value result(LogicalType::TIMESTAMP_MS);
	result.value_.timestamp = timestamp;
	result.is_null = false;
	return result;
}

Value Value::TIMESTAMPSEC(timestamp_t timestamp) {
	Value result(LogicalType::TIMESTAMP_S);
	result.value_.timestamp = timestamp;
	result.is_null = false;
	return result;
}

Value Value::TIMESTAMP(date_t date, dtime_t time) {
	return Value::TIMESTAMP(Timestamp::FromDatetime(date, time));
}

Value Value::TIMESTAMP(int32_t year, int32_t month, int32_t day, int32_t hour, int32_t min, int32_t sec,
                       int32_t micros) {
	auto val = Value::TIMESTAMP(Date::FromDate(year, month, day), Time::FromTime(hour, min, sec, micros));
	val.type_ = LogicalType::TIMESTAMP;
	return val;
}

Value Value::STRUCT(child_list_t<Value> values) {
	Value result;
	child_list_t<LogicalType> child_types;
	for (auto &child : values) {
		child_types.push_back(make_pair(move(child.first), child.second.type()));
		result.struct_value.push_back(move(child.second));
	}
	result.type_ = LogicalType::STRUCT(move(child_types));

	result.is_null = false;
	return result;
}

Value Value::MAP(Value key, Value value) {
	Value result;
	child_list_t<LogicalType> child_types;
	child_types.push_back({"key", key.type()});
	child_types.push_back({"value", value.type()});

	result.type_ = LogicalType::MAP(move(child_types));

	result.struct_value.push_back(move(key));
	result.struct_value.push_back(move(value));
	result.is_null = false;
	return result;
}

Value Value::UNION(child_list_t<LogicalType> members, uint8_t tag, Value value) {
	D_ASSERT(members.size() > 0);
	D_ASSERT(members.size() <= UnionType::MAX_UNION_MEMBERS);
	D_ASSERT(members.size() > tag);

	D_ASSERT(value.type() == members[tag].second);

	Value result;
	result.is_null = false;
	// add the tag to the front of the struct
	result.struct_value.emplace_back(Value::TINYINT(tag));
	for (idx_t i = 0; i < members.size(); i++) {
		if (i != tag) {
			result.struct_value.emplace_back(members[i].second);
		} else {
			result.struct_value.emplace_back(nullptr);
		}
	}
	result.struct_value[tag + 1] = move(value);

	result.type_ = LogicalType::UNION(move(members));
	return result;
}

Value Value::LIST(vector<Value> values) {
	if (values.empty()) {
		throw InternalException("Value::LIST without providing a child-type requires a non-empty list of values. Use "
		                        "Value::LIST(child_type, list) instead.");
	}
#ifdef DEBUG
	for (idx_t i = 1; i < values.size(); i++) {
		D_ASSERT(values[i].type() == values[0].type());
	}
#endif
	Value result;
	result.type_ = LogicalType::LIST(values[0].type());
	result.list_value = move(values);
	result.is_null = false;
	return result;
}

Value Value::LIST(LogicalType child_type, vector<Value> values) {
	if (values.empty()) {
		return Value::EMPTYLIST(move(child_type));
	}
	for (auto &val : values) {
		val = val.DefaultCastAs(child_type);
	}
	return Value::LIST(move(values));
}

Value Value::EMPTYLIST(LogicalType child_type) {
	Value result;
	result.type_ = LogicalType::LIST(move(child_type));
	result.is_null = false;
	return result;
}

Value Value::BLOB(const_data_ptr_t data, idx_t len) {
	Value result(LogicalType::BLOB);
	result.is_null = false;
	result.str_value = string((const char *)data, len);
	return result;
}

Value Value::BLOB(const string &data) {
	Value result(LogicalType::BLOB);
	result.is_null = false;
	result.str_value = Blob::ToBlob(string_t(data));
	return result;
}

Value Value::JSON(const char *val) {
	auto result = Value(val);
	result.type_ = LogicalTypeId::JSON;
	return result;
}

Value Value::JSON(string_t val) {
	auto result = Value(val);
	result.type_ = LogicalTypeId::JSON;
	return result;
}

Value Value::JSON(string val) {
	auto result = Value(move(val));
	result.type_ = LogicalTypeId::JSON;
	return result;
}

Value Value::ENUM(uint64_t value, const LogicalType &original_type) {
	D_ASSERT(original_type.id() == LogicalTypeId::ENUM);
	Value result(original_type);
	switch (original_type.InternalType()) {
	case PhysicalType::UINT8:
		result.value_.utinyint = value;
		break;
	case PhysicalType::UINT16:
		result.value_.usmallint = value;
		break;
	case PhysicalType::UINT32:
		result.value_.uinteger = value;
		break;
	case PhysicalType::UINT64: //  DEDUP_POINTER_ENUM
		result.value_.ubigint = value;
		break;
	default:
		throw InternalException("Incorrect Physical Type for ENUM");
	}
	result.is_null = false;
	return result;
}

Value Value::INTERVAL(int32_t months, int32_t days, int64_t micros) {
	Value result(LogicalType::INTERVAL);
	result.is_null = false;
	result.value_.interval.months = months;
	result.value_.interval.days = days;
	result.value_.interval.micros = micros;
	return result;
}

Value Value::INTERVAL(interval_t interval) {
	return Value::INTERVAL(interval.months, interval.days, interval.micros);
}

//===--------------------------------------------------------------------===//
// CreateValue
//===--------------------------------------------------------------------===//
template <>
Value Value::CreateValue(bool value) {
	return Value::BOOLEAN(value);
}

template <>
Value Value::CreateValue(int8_t value) {
	return Value::TINYINT(value);
}

template <>
Value Value::CreateValue(int16_t value) {
	return Value::SMALLINT(value);
}

template <>
Value Value::CreateValue(int32_t value) {
	return Value::INTEGER(value);
}

template <>
Value Value::CreateValue(int64_t value) {
	return Value::BIGINT(value);
}

template <>
Value Value::CreateValue(uint8_t value) {
	return Value::UTINYINT(value);
}

template <>
Value Value::CreateValue(uint16_t value) {
	return Value::USMALLINT(value);
}

template <>
Value Value::CreateValue(uint32_t value) {
	return Value::UINTEGER(value);
}

template <>
Value Value::CreateValue(uint64_t value) {
	return Value::UBIGINT(value);
}

template <>
Value Value::CreateValue(hugeint_t value) {
	return Value::HUGEINT(value);
}

template <>
Value Value::CreateValue(date_t value) {
	return Value::DATE(value);
}

template <>
Value Value::CreateValue(dtime_t value) {
	return Value::TIME(value);
}

template <>
Value Value::CreateValue(dtime_tz_t value) {
	return Value::TIMETZ(value);
}

template <>
Value Value::CreateValue(timestamp_t value) {
	return Value::TIMESTAMP(value);
}

template <>
Value Value::CreateValue(timestamp_sec_t value) {
	return Value::TIMESTAMPSEC(value);
}

template <>
Value Value::CreateValue(timestamp_ms_t value) {
	return Value::TIMESTAMPMS(value);
}

template <>
Value Value::CreateValue(timestamp_ns_t value) {
	return Value::TIMESTAMPNS(value);
}

template <>
Value Value::CreateValue(timestamp_tz_t value) {
	return Value::TIMESTAMPTZ(value);
}

template <>
Value Value::CreateValue(const char *value) {
	return Value(string(value));
}

template <>
Value Value::CreateValue(string value) { // NOLINT: required for templating
	return Value::BLOB(value);
}

template <>
Value Value::CreateValue(string_t value) {
	return Value(value);
}

template <>
Value Value::CreateValue(float value) {
	return Value::FLOAT(value);
}

template <>
Value Value::CreateValue(double value) {
	return Value::DOUBLE(value);
}

template <>
Value Value::CreateValue(interval_t value) {
	return Value::INTERVAL(value);
}

template <>
Value Value::CreateValue(Value value) {
	return value;
}

//===--------------------------------------------------------------------===//
// GetValue
//===--------------------------------------------------------------------===//
template <class T>
T Value::GetValueInternal() const {
	if (IsNull()) {
		throw InternalException("Calling GetValueInternal on a value that is NULL");
	}
	switch (type_.id()) {
	case LogicalTypeId::BOOLEAN:
		return Cast::Operation<bool, T>(value_.boolean);
	case LogicalTypeId::TINYINT:
		return Cast::Operation<int8_t, T>(value_.tinyint);
	case LogicalTypeId::SMALLINT:
		return Cast::Operation<int16_t, T>(value_.smallint);
	case LogicalTypeId::INTEGER:
		return Cast::Operation<int32_t, T>(value_.integer);
	case LogicalTypeId::BIGINT:
		return Cast::Operation<int64_t, T>(value_.bigint);
	case LogicalTypeId::HUGEINT:
	case LogicalTypeId::UUID:
		return Cast::Operation<hugeint_t, T>(value_.hugeint);
	case LogicalTypeId::DATE:
		return Cast::Operation<date_t, T>(value_.date);
	case LogicalTypeId::TIME:
	case LogicalTypeId::TIME_TZ:
		return Cast::Operation<dtime_t, T>(value_.time);
	case LogicalTypeId::TIMESTAMP:
	case LogicalTypeId::TIMESTAMP_TZ:
		return Cast::Operation<timestamp_t, T>(value_.timestamp);
	case LogicalTypeId::UTINYINT:
		return Cast::Operation<uint8_t, T>(value_.utinyint);
	case LogicalTypeId::USMALLINT:
		return Cast::Operation<uint16_t, T>(value_.usmallint);
	case LogicalTypeId::UINTEGER:
		return Cast::Operation<uint32_t, T>(value_.uinteger);
	case LogicalTypeId::TIMESTAMP_MS:
	case LogicalTypeId::TIMESTAMP_NS:
	case LogicalTypeId::TIMESTAMP_SEC:
	case LogicalTypeId::UBIGINT:
		return Cast::Operation<uint64_t, T>(value_.ubigint);
	case LogicalTypeId::FLOAT:
		return Cast::Operation<float, T>(value_.float_);
	case LogicalTypeId::DOUBLE:
		return Cast::Operation<double, T>(value_.double_);
	case LogicalTypeId::VARCHAR:
		return Cast::Operation<string_t, T>(str_value.c_str());
	case LogicalTypeId::INTERVAL:
		return Cast::Operation<interval_t, T>(value_.interval);
	case LogicalTypeId::DECIMAL:
		return DefaultCastAs(LogicalType::DOUBLE).GetValueInternal<T>();
	case LogicalTypeId::ENUM: {
		switch (type_.InternalType()) {
		case PhysicalType::UINT8:
			return Cast::Operation<uint8_t, T>(value_.utinyint);
		case PhysicalType::UINT16:
			return Cast::Operation<uint16_t, T>(value_.usmallint);
		case PhysicalType::UINT32:
			return Cast::Operation<uint32_t, T>(value_.uinteger);
		default:
			throw InternalException("Invalid Internal Type for ENUMs");
		}
	}
	default:
		throw NotImplementedException("Unimplemented type \"%s\" for GetValue()", type_.ToString());
	}
}

template <>
bool Value::GetValue() const {
	return GetValueInternal<int8_t>();
}
template <>
int8_t Value::GetValue() const {
	return GetValueInternal<int8_t>();
}
template <>
int16_t Value::GetValue() const {
	return GetValueInternal<int16_t>();
}
template <>
int32_t Value::GetValue() const {
	if (type_.id() == LogicalTypeId::DATE) {
		return value_.integer;
	}
	return GetValueInternal<int32_t>();
}
template <>
int64_t Value::GetValue() const {
	if (IsNull()) {
		throw InternalException("Calling GetValue on a value that is NULL");
	}
	switch (type_.id()) {
	case LogicalTypeId::TIMESTAMP:
	case LogicalTypeId::TIMESTAMP_SEC:
	case LogicalTypeId::TIMESTAMP_NS:
	case LogicalTypeId::TIMESTAMP_MS:
	case LogicalTypeId::TIME:
	case LogicalTypeId::TIME_TZ:
	case LogicalTypeId::TIMESTAMP_TZ:
		return value_.bigint;
	default:
		return GetValueInternal<int64_t>();
	}
}
template <>
hugeint_t Value::GetValue() const {
	return GetValueInternal<hugeint_t>();
}
template <>
uint8_t Value::GetValue() const {
	return GetValueInternal<uint8_t>();
}
template <>
uint16_t Value::GetValue() const {
	return GetValueInternal<uint16_t>();
}
template <>
uint32_t Value::GetValue() const {
	return GetValueInternal<uint32_t>();
}
template <>
uint64_t Value::GetValue() const {
	return GetValueInternal<uint64_t>();
}
template <>
string Value::GetValue() const {
	return ToString();
}
template <>
float Value::GetValue() const {
	return GetValueInternal<float>();
}
template <>
double Value::GetValue() const {
	return GetValueInternal<double>();
}
template <>
date_t Value::GetValue() const {
	return GetValueInternal<date_t>();
}
template <>
dtime_t Value::GetValue() const {
	return GetValueInternal<dtime_t>();
}
template <>
timestamp_t Value::GetValue() const {
	return GetValueInternal<timestamp_t>();
}

template <>
DUCKDB_API interval_t Value::GetValue() const {
	return GetValueInternal<interval_t>();
}

template <>
DUCKDB_API Value Value::GetValue() const {
	return Value(*this);
}

uintptr_t Value::GetPointer() const {
	D_ASSERT(type() == LogicalType::POINTER);
	return value_.pointer;
}

Value Value::Numeric(const LogicalType &type, int64_t value) {
	switch (type.id()) {
	case LogicalTypeId::BOOLEAN:
		D_ASSERT(value == 0 || value == 1);
		return Value::BOOLEAN(value ? 1 : 0);
	case LogicalTypeId::TINYINT:
		D_ASSERT(value >= NumericLimits<int8_t>::Minimum() && value <= NumericLimits<int8_t>::Maximum());
		return Value::TINYINT((int8_t)value);
	case LogicalTypeId::SMALLINT:
		D_ASSERT(value >= NumericLimits<int16_t>::Minimum() && value <= NumericLimits<int16_t>::Maximum());
		return Value::SMALLINT((int16_t)value);
	case LogicalTypeId::INTEGER:
		D_ASSERT(value >= NumericLimits<int32_t>::Minimum() && value <= NumericLimits<int32_t>::Maximum());
		return Value::INTEGER((int32_t)value);
	case LogicalTypeId::BIGINT:
		return Value::BIGINT(value);
	case LogicalTypeId::UTINYINT:
		D_ASSERT(value >= NumericLimits<uint8_t>::Minimum() && value <= NumericLimits<uint8_t>::Maximum());
		return Value::UTINYINT((uint8_t)value);
	case LogicalTypeId::USMALLINT:
		D_ASSERT(value >= NumericLimits<uint16_t>::Minimum() && value <= NumericLimits<uint16_t>::Maximum());
		return Value::USMALLINT((uint16_t)value);
	case LogicalTypeId::UINTEGER:
		D_ASSERT(value >= NumericLimits<uint32_t>::Minimum() && value <= NumericLimits<uint32_t>::Maximum());
		return Value::UINTEGER((uint32_t)value);
	case LogicalTypeId::UBIGINT:
		D_ASSERT(value >= 0);
		return Value::UBIGINT(value);
	case LogicalTypeId::HUGEINT:
		return Value::HUGEINT(value);
	case LogicalTypeId::DECIMAL:
		return Value::DECIMAL(value, DecimalType::GetWidth(type), DecimalType::GetScale(type));
	case LogicalTypeId::FLOAT:
		return Value((float)value);
	case LogicalTypeId::DOUBLE:
		return Value((double)value);
	case LogicalTypeId::POINTER:
		return Value::POINTER(value);
	case LogicalTypeId::DATE:
		D_ASSERT(value >= NumericLimits<int32_t>::Minimum() && value <= NumericLimits<int32_t>::Maximum());
		return Value::DATE(date_t(value));
	case LogicalTypeId::TIME:
		return Value::TIME(dtime_t(value));
	case LogicalTypeId::TIMESTAMP:
		return Value::TIMESTAMP(timestamp_t(value));
	case LogicalTypeId::TIMESTAMP_NS:
		return Value::TIMESTAMPNS(timestamp_t(value));
	case LogicalTypeId::TIMESTAMP_MS:
		return Value::TIMESTAMPMS(timestamp_t(value));
	case LogicalTypeId::TIMESTAMP_SEC:
		return Value::TIMESTAMPSEC(timestamp_t(value));
	case LogicalTypeId::TIME_TZ:
		return Value::TIMETZ(dtime_t(value));
	case LogicalTypeId::TIMESTAMP_TZ:
		return Value::TIMESTAMPTZ(timestamp_t(value));
	case LogicalTypeId::ENUM:
		switch (type.InternalType()) {
		case PhysicalType::UINT8:
			D_ASSERT(value >= NumericLimits<uint8_t>::Minimum() && value <= NumericLimits<uint8_t>::Maximum());
			return Value::UTINYINT((uint8_t)value);
		case PhysicalType::UINT16:
			D_ASSERT(value >= NumericLimits<uint16_t>::Minimum() && value <= NumericLimits<uint16_t>::Maximum());
			return Value::USMALLINT((uint16_t)value);
		case PhysicalType::UINT32:
			D_ASSERT(value >= NumericLimits<uint32_t>::Minimum() && value <= NumericLimits<uint32_t>::Maximum());
			return Value::UINTEGER((uint32_t)value);
		default:
			throw InternalException("Enum doesn't accept this physical type");
		}
	default:
		throw InvalidTypeException(type, "Numeric requires numeric type");
	}
}

Value Value::Numeric(const LogicalType &type, hugeint_t value) {
#ifdef DEBUG
	// perform a throwing cast to verify that the type fits
	Value::HUGEINT(value).DefaultCastAs(type);
#endif
	switch (type.id()) {
	case LogicalTypeId::HUGEINT:
		return Value::HUGEINT(value);
	case LogicalTypeId::UBIGINT:
		return Value::UBIGINT(Hugeint::Cast<uint64_t>(value));
	default:
		return Value::Numeric(type, Hugeint::Cast<int64_t>(value));
	}
}

//===--------------------------------------------------------------------===//
// GetValueUnsafe
//===--------------------------------------------------------------------===//
template <>
DUCKDB_API bool Value::GetValueUnsafe() const {
	D_ASSERT(type_.InternalType() == PhysicalType::BOOL);
	return value_.boolean;
}

template <>
int8_t Value::GetValueUnsafe() const {
	D_ASSERT(type_.InternalType() == PhysicalType::INT8 || type_.InternalType() == PhysicalType::BOOL);
	return value_.tinyint;
}

template <>
int16_t Value::GetValueUnsafe() const {
	D_ASSERT(type_.InternalType() == PhysicalType::INT16);
	return value_.smallint;
}

template <>
int32_t Value::GetValueUnsafe() const {
	D_ASSERT(type_.InternalType() == PhysicalType::INT32);
	return value_.integer;
}

template <>
int64_t Value::GetValueUnsafe() const {
	D_ASSERT(type_.InternalType() == PhysicalType::INT64);
	return value_.bigint;
}

template <>
hugeint_t Value::GetValueUnsafe() const {
	D_ASSERT(type_.InternalType() == PhysicalType::INT128);
	return value_.hugeint;
}

template <>
uint8_t Value::GetValueUnsafe() const {
	D_ASSERT(type_.InternalType() == PhysicalType::UINT8);
	return value_.utinyint;
}

template <>
uint16_t Value::GetValueUnsafe() const {
	D_ASSERT(type_.InternalType() == PhysicalType::UINT16);
	return value_.usmallint;
}

template <>
uint32_t Value::GetValueUnsafe() const {
	D_ASSERT(type_.InternalType() == PhysicalType::UINT32);
	return value_.uinteger;
}

template <>
uint64_t Value::GetValueUnsafe() const {
	D_ASSERT(type_.InternalType() == PhysicalType::UINT64);
	return value_.ubigint;
}

template <>
string Value::GetValueUnsafe() const {
	D_ASSERT(type_.InternalType() == PhysicalType::VARCHAR);
	return str_value;
}

template <>
DUCKDB_API string_t Value::GetValueUnsafe() const {
	D_ASSERT(type_.InternalType() == PhysicalType::VARCHAR);
	return string_t(str_value);
}

template <>
float Value::GetValueUnsafe() const {
	D_ASSERT(type_.InternalType() == PhysicalType::FLOAT);
	return value_.float_;
}

template <>
double Value::GetValueUnsafe() const {
	D_ASSERT(type_.InternalType() == PhysicalType::DOUBLE);
	return value_.double_;
}

template <>
date_t Value::GetValueUnsafe() const {
	D_ASSERT(type_.InternalType() == PhysicalType::INT32);
	return value_.date;
}

template <>
dtime_t Value::GetValueUnsafe() const {
	D_ASSERT(type_.InternalType() == PhysicalType::INT64);
	return value_.time;
}

template <>
timestamp_t Value::GetValueUnsafe() const {
	D_ASSERT(type_.InternalType() == PhysicalType::INT64);
	return value_.timestamp;
}

template <>
interval_t Value::GetValueUnsafe() const {
	D_ASSERT(type_.InternalType() == PhysicalType::INTERVAL);
	return value_.interval;
}

//===--------------------------------------------------------------------===//
// GetReferenceUnsafe
//===--------------------------------------------------------------------===//
template <>
int8_t &Value::GetReferenceUnsafe() {
	D_ASSERT(type_.InternalType() == PhysicalType::INT8 || type_.InternalType() == PhysicalType::BOOL);
	return value_.tinyint;
}

template <>
int16_t &Value::GetReferenceUnsafe() {
	D_ASSERT(type_.InternalType() == PhysicalType::INT16);
	return value_.smallint;
}

template <>
int32_t &Value::GetReferenceUnsafe() {
	D_ASSERT(type_.InternalType() == PhysicalType::INT32);
	return value_.integer;
}

template <>
int64_t &Value::GetReferenceUnsafe() {
	D_ASSERT(type_.InternalType() == PhysicalType::INT64);
	return value_.bigint;
}

template <>
hugeint_t &Value::GetReferenceUnsafe() {
	D_ASSERT(type_.InternalType() == PhysicalType::INT128);
	return value_.hugeint;
}

template <>
uint8_t &Value::GetReferenceUnsafe() {
	D_ASSERT(type_.InternalType() == PhysicalType::UINT8);
	return value_.utinyint;
}

template <>
uint16_t &Value::GetReferenceUnsafe() {
	D_ASSERT(type_.InternalType() == PhysicalType::UINT16);
	return value_.usmallint;
}

template <>
uint32_t &Value::GetReferenceUnsafe() {
	D_ASSERT(type_.InternalType() == PhysicalType::UINT32);
	return value_.uinteger;
}

template <>
uint64_t &Value::GetReferenceUnsafe() {
	D_ASSERT(type_.InternalType() == PhysicalType::UINT64);
	return value_.ubigint;
}

template <>
float &Value::GetReferenceUnsafe() {
	D_ASSERT(type_.InternalType() == PhysicalType::FLOAT);
	return value_.float_;
}

template <>
double &Value::GetReferenceUnsafe() {
	D_ASSERT(type_.InternalType() == PhysicalType::DOUBLE);
	return value_.double_;
}

template <>
date_t &Value::GetReferenceUnsafe() {
	D_ASSERT(type_.InternalType() == PhysicalType::INT32);
	return value_.date;
}

template <>
dtime_t &Value::GetReferenceUnsafe() {
	D_ASSERT(type_.InternalType() == PhysicalType::INT64);
	return value_.time;
}

template <>
timestamp_t &Value::GetReferenceUnsafe() {
	D_ASSERT(type_.InternalType() == PhysicalType::INT64);
	return value_.timestamp;
}

template <>
interval_t &Value::GetReferenceUnsafe() {
	D_ASSERT(type_.InternalType() == PhysicalType::INTERVAL);
	return value_.interval;
}

//===--------------------------------------------------------------------===//
// Hash
//===--------------------------------------------------------------------===//
hash_t Value::Hash() const {
	if (IsNull()) {
		return 0;
	}
	Vector input(*this);
	Vector result(LogicalType::HASH);
	VectorOperations::Hash(input, result, 1);

	auto data = FlatVector::GetData<hash_t>(result);
	return data[0];
}

string Value::ToString() const {
	if (IsNull()) {
		return "NULL";
	}
	return DefaultCastAs(LogicalType::VARCHAR).str_value;
}

string Value::ToSQLString() const {
	if (IsNull()) {
		return ToString();
	}
	switch (type_.id()) {
	case LogicalTypeId::UUID:
	case LogicalTypeId::DATE:
	case LogicalTypeId::TIME:
	case LogicalTypeId::TIMESTAMP:
	case LogicalTypeId::TIME_TZ:
	case LogicalTypeId::TIMESTAMP_TZ:
	case LogicalTypeId::TIMESTAMP_SEC:
	case LogicalTypeId::TIMESTAMP_MS:
	case LogicalTypeId::TIMESTAMP_NS:
	case LogicalTypeId::INTERVAL:
	case LogicalTypeId::BLOB:
		return "'" + ToString() + "'::" + type_.ToString();
	case LogicalTypeId::VARCHAR:
		return "'" + StringUtil::Replace(ToString(), "'", "''") + "'";
	case LogicalTypeId::STRUCT: {
		string ret = "{";
		auto &child_types = StructType::GetChildTypes(type_);
		for (size_t i = 0; i < struct_value.size(); i++) {
			auto &name = child_types[i].first;
			auto &child = struct_value[i];
			ret += "'" + name + "': " + child.ToSQLString();
			if (i < struct_value.size() - 1) {
				ret += ", ";
			}
		}
		ret += "}";
		return ret;
	}
	case LogicalTypeId::FLOAT:
		if (!FloatIsFinite(FloatValue::Get(*this))) {
			return "'" + ToString() + "'::" + type_.ToString();
		}
		return ToString();
	case LogicalTypeId::DOUBLE: {
		double val = DoubleValue::Get(*this);
		if (!DoubleIsFinite(val)) {
			if (!Value::IsNan(val)) {
				// to infinity and beyond
				return val < 0 ? "-1e1000" : "1e1000";
			}
			return "'" + ToString() + "'::" + type_.ToString();
		}
		return ToString();
	}
	case LogicalTypeId::LIST: {
		string ret = "[";
		for (size_t i = 0; i < list_value.size(); i++) {
			auto &child = list_value[i];
			ret += child.ToSQLString();
			if (i < list_value.size() - 1) {
				ret += ", ";
			}
		}
		ret += "]";
		return ret;
	}
	default:
		return ToString();
	}
}

//===--------------------------------------------------------------------===//
// Type-specific getters
//===--------------------------------------------------------------------===//
bool BooleanValue::Get(const Value &value) {
	return value.GetValueUnsafe<bool>();
}

int8_t TinyIntValue::Get(const Value &value) {
	return value.GetValueUnsafe<int8_t>();
}

int16_t SmallIntValue::Get(const Value &value) {
	return value.GetValueUnsafe<int16_t>();
}

int32_t IntegerValue::Get(const Value &value) {
	return value.GetValueUnsafe<int32_t>();
}

int64_t BigIntValue::Get(const Value &value) {
	return value.GetValueUnsafe<int64_t>();
}

hugeint_t HugeIntValue::Get(const Value &value) {
	return value.GetValueUnsafe<hugeint_t>();
}

uint8_t UTinyIntValue::Get(const Value &value) {
	return value.GetValueUnsafe<uint8_t>();
}

uint16_t USmallIntValue::Get(const Value &value) {
	return value.GetValueUnsafe<uint16_t>();
}

uint32_t UIntegerValue::Get(const Value &value) {
	return value.GetValueUnsafe<uint32_t>();
}

uint64_t UBigIntValue::Get(const Value &value) {
	return value.GetValueUnsafe<uint64_t>();
}

float FloatValue::Get(const Value &value) {
	return value.GetValueUnsafe<float>();
}

double DoubleValue::Get(const Value &value) {
	return value.GetValueUnsafe<double>();
}

const string &StringValue::Get(const Value &value) {
	D_ASSERT(value.type().InternalType() == PhysicalType::VARCHAR);
	return value.str_value;
}

date_t DateValue::Get(const Value &value) {
	return value.GetValueUnsafe<date_t>();
}

dtime_t TimeValue::Get(const Value &value) {
	return value.GetValueUnsafe<dtime_t>();
}

timestamp_t TimestampValue::Get(const Value &value) {
	return value.GetValueUnsafe<timestamp_t>();
}

interval_t IntervalValue::Get(const Value &value) {
	return value.GetValueUnsafe<interval_t>();
}

const vector<Value> &StructValue::GetChildren(const Value &value) {
	D_ASSERT(value.type().InternalType() == PhysicalType::STRUCT);
	return value.struct_value;
}

const vector<Value> &ListValue::GetChildren(const Value &value) {
	D_ASSERT(value.type().InternalType() == PhysicalType::LIST);
	return value.list_value;
}

const Value &UnionValue::GetValue(const Value &value) {
	D_ASSERT(value.type() == LogicalTypeId::UNION);
	auto &children = StructValue::GetChildren(value);
	auto tag = children[0].GetValueUnsafe<uint8_t>();
	D_ASSERT(tag < children.size() - 1);
	return children[tag + 1];
}

uint8_t UnionValue::GetTag(const Value &value) {
	D_ASSERT(value.type() == LogicalTypeId::UNION);
	auto children = StructValue::GetChildren(value);
	auto tag = children[0].GetValueUnsafe<uint8_t>();
	return tag;
}

hugeint_t IntegralValue::Get(const Value &value) {
	switch (value.type().InternalType()) {
	case PhysicalType::INT8:
		return TinyIntValue::Get(value);
	case PhysicalType::INT16:
		return SmallIntValue::Get(value);
	case PhysicalType::INT32:
		return IntegerValue::Get(value);
	case PhysicalType::INT64:
		return BigIntValue::Get(value);
	case PhysicalType::INT128:
		return HugeIntValue::Get(value);
	case PhysicalType::UINT8:
		return UTinyIntValue::Get(value);
	case PhysicalType::UINT16:
		return USmallIntValue::Get(value);
	case PhysicalType::UINT32:
		return UIntegerValue::Get(value);
	case PhysicalType::UINT64:
		return UBigIntValue::Get(value);
	default:
		throw InternalException("Invalid internal type \"%s\" for IntegralValue::Get", value.type().ToString());
	}
}

//===--------------------------------------------------------------------===//
// Comparison Operators
//===--------------------------------------------------------------------===//
bool Value::operator==(const Value &rhs) const {
	return ValueOperations::Equals(*this, rhs);
}

bool Value::operator!=(const Value &rhs) const {
	return ValueOperations::NotEquals(*this, rhs);
}

bool Value::operator<(const Value &rhs) const {
	return ValueOperations::LessThan(*this, rhs);
}

bool Value::operator>(const Value &rhs) const {
	return ValueOperations::GreaterThan(*this, rhs);
}

bool Value::operator<=(const Value &rhs) const {
	return ValueOperations::LessThanEquals(*this, rhs);
}

bool Value::operator>=(const Value &rhs) const {
	return ValueOperations::GreaterThanEquals(*this, rhs);
}

bool Value::operator==(const int64_t &rhs) const {
	return *this == Value::Numeric(type_, rhs);
}

bool Value::operator!=(const int64_t &rhs) const {
	return *this != Value::Numeric(type_, rhs);
}

bool Value::operator<(const int64_t &rhs) const {
	return *this < Value::Numeric(type_, rhs);
}

bool Value::operator>(const int64_t &rhs) const {
	return *this > Value::Numeric(type_, rhs);
}

bool Value::operator<=(const int64_t &rhs) const {
	return *this <= Value::Numeric(type_, rhs);
}

bool Value::operator>=(const int64_t &rhs) const {
	return *this >= Value::Numeric(type_, rhs);
}

bool Value::TryCastAs(CastFunctionSet &set, GetCastFunctionInput &get_input, const LogicalType &target_type,
                      Value &new_value, string *error_message, bool strict) const {
	if (type_ == target_type) {
		new_value = Copy();
		return true;
	}
	Vector input(*this);
	Vector result(target_type);
	if (!VectorOperations::TryCast(set, get_input, input, result, 1, error_message, strict)) {
		return false;
	}
	new_value = result.GetValue(0);
	return true;
}

bool Value::TryCastAs(ClientContext &context, const LogicalType &target_type, Value &new_value, string *error_message,
                      bool strict) const {
	GetCastFunctionInput get_input(context);
	return TryCastAs(CastFunctionSet::Get(context), get_input, target_type, new_value, error_message, strict);
}

bool Value::DefaultTryCastAs(const LogicalType &target_type, Value &new_value, string *error_message,
                             bool strict) const {
	CastFunctionSet set;
	GetCastFunctionInput get_input;
	return TryCastAs(set, get_input, target_type, new_value, error_message, strict);
}

Value Value::CastAs(CastFunctionSet &set, GetCastFunctionInput &get_input, const LogicalType &target_type,
                    bool strict) const {
	Value new_value;
	string error_message;
	if (!TryCastAs(set, get_input, target_type, new_value, &error_message, strict)) {
		throw InvalidInputException("Failed to cast value: %s", error_message);
	}
	return new_value;
}

Value Value::CastAs(ClientContext &context, const LogicalType &target_type, bool strict) const {
	GetCastFunctionInput get_input(context);
	return CastAs(CastFunctionSet::Get(context), get_input, target_type, strict);
}

Value Value::DefaultCastAs(const LogicalType &target_type, bool strict) const {
	CastFunctionSet set;
	GetCastFunctionInput get_input;
	return CastAs(set, get_input, target_type, strict);
}

bool Value::TryCastAs(CastFunctionSet &set, GetCastFunctionInput &get_input, const LogicalType &target_type,
                      bool strict) {
	Value new_value;
	string error_message;
	if (!TryCastAs(set, get_input, target_type, new_value, &error_message, strict)) {
		return false;
	}
	type_ = target_type;
	is_null = new_value.is_null;
	value_ = new_value.value_;
	str_value = new_value.str_value;
	struct_value = new_value.struct_value;
	list_value = new_value.list_value;
	return true;
}

bool Value::TryCastAs(ClientContext &context, const LogicalType &target_type, bool strict) {
	GetCastFunctionInput get_input(context);
	return TryCastAs(CastFunctionSet::Get(context), get_input, target_type, strict);
}

bool Value::DefaultTryCastAs(const LogicalType &target_type, bool strict) {
	CastFunctionSet set;
	GetCastFunctionInput get_input;
	return TryCastAs(set, get_input, target_type, strict);
}

void Value::Serialize(Serializer &main_serializer) const {
	FieldWriter writer(main_serializer);
	writer.WriteSerializable(type_);
	writer.WriteField<bool>(IsNull());
	if (!IsNull()) {
		auto &serializer = writer.GetSerializer();
		switch (type_.InternalType()) {
		case PhysicalType::BOOL:
			serializer.Write<int8_t>(value_.boolean);
			break;
		case PhysicalType::INT8:
			serializer.Write<int8_t>(value_.tinyint);
			break;
		case PhysicalType::INT16:
			serializer.Write<int16_t>(value_.smallint);
			break;
		case PhysicalType::INT32:
			serializer.Write<int32_t>(value_.integer);
			break;
		case PhysicalType::INT64:
			serializer.Write<int64_t>(value_.bigint);
			break;
		case PhysicalType::UINT8:
			serializer.Write<uint8_t>(value_.utinyint);
			break;
		case PhysicalType::UINT16:
			serializer.Write<uint16_t>(value_.usmallint);
			break;
		case PhysicalType::UINT32:
			serializer.Write<uint32_t>(value_.uinteger);
			break;
		case PhysicalType::UINT64:
			serializer.Write<uint64_t>(value_.ubigint);
			break;
		case PhysicalType::INT128:
			serializer.Write<hugeint_t>(value_.hugeint);
			break;
		case PhysicalType::FLOAT:
			serializer.Write<float>(value_.float_);
			break;
		case PhysicalType::DOUBLE:
			serializer.Write<double>(value_.double_);
			break;
		case PhysicalType::INTERVAL:
			serializer.Write<interval_t>(value_.interval);
			break;
		case PhysicalType::VARCHAR:
			serializer.WriteString(str_value);
			break;
		default: {
			Vector v(*this);
			v.Serialize(1, serializer);
			break;
		}
		}
	}
	writer.Finalize();
}

Value Value::Deserialize(Deserializer &main_source) {
	FieldReader reader(main_source);
	auto type = reader.ReadRequiredSerializable<LogicalType, LogicalType>();
	auto is_null = reader.ReadRequired<bool>();
	Value new_value = Value(type);
	if (is_null) {
		reader.Finalize();
		return new_value;
	}
	new_value.is_null = false;
	auto &source = reader.GetSource();
	switch (type.InternalType()) {
	case PhysicalType::BOOL:
		new_value.value_.boolean = source.Read<int8_t>();
		break;
	case PhysicalType::INT8:
		new_value.value_.tinyint = source.Read<int8_t>();
		break;
	case PhysicalType::INT16:
		new_value.value_.smallint = source.Read<int16_t>();
		break;
	case PhysicalType::INT32:
		new_value.value_.integer = source.Read<int32_t>();
		break;
	case PhysicalType::INT64:
		new_value.value_.bigint = source.Read<int64_t>();
		break;
	case PhysicalType::UINT8:
		new_value.value_.utinyint = source.Read<uint8_t>();
		break;
	case PhysicalType::UINT16:
		new_value.value_.usmallint = source.Read<uint16_t>();
		break;
	case PhysicalType::UINT32:
		new_value.value_.uinteger = source.Read<uint32_t>();
		break;
	case PhysicalType::UINT64:
		new_value.value_.ubigint = source.Read<uint64_t>();
		break;
	case PhysicalType::INT128:
		new_value.value_.hugeint = source.Read<hugeint_t>();
		break;
	case PhysicalType::FLOAT:
		new_value.value_.float_ = source.Read<float>();
		break;
	case PhysicalType::DOUBLE:
		new_value.value_.double_ = source.Read<double>();
		break;
	case PhysicalType::INTERVAL:
		new_value.value_.interval = source.Read<interval_t>();
		break;
	case PhysicalType::VARCHAR:
		new_value.str_value = source.Read<string>();
		break;
	default: {
		Vector v(type);
		v.Deserialize(1, source);
		new_value = v.GetValue(0);
		break;
	}
	}
	reader.Finalize();
	return new_value;
}

void Value::Print() const {
	Printer::Print(ToString());
}

bool Value::NotDistinctFrom(const Value &lvalue, const Value &rvalue) {
	return ValueOperations::NotDistinctFrom(lvalue, rvalue);
}

static string SanitizeValue(string input) {
	// some results might contain padding spaces, e.g. when rendering
	// VARCHAR(10) and the string only has 6 characters, they will be padded
	// with spaces to 10 in the rendering. We don't do that here yet as we
	// are looking at internal structures. So just ignore any extra spaces
	// on the right
	StringUtil::RTrim(input);
	// for result checking code, replace null bytes with their escaped value (\0)
	return StringUtil::Replace(input, string("\0", 1), "\\0");
}

bool Value::ValuesAreEqual(CastFunctionSet &set, GetCastFunctionInput &get_input, const Value &result_value,
                           const Value &value) {
	if (result_value.IsNull() != value.IsNull()) {
		return false;
	}
	if (result_value.IsNull() && value.IsNull()) {
		// NULL = NULL in checking code
		return true;
	}
	switch (value.type_.id()) {
	case LogicalTypeId::FLOAT: {
		auto other = result_value.CastAs(set, get_input, LogicalType::FLOAT);
		float ldecimal = value.value_.float_;
		float rdecimal = other.value_.float_;
		return ApproxEqual(ldecimal, rdecimal);
	}
	case LogicalTypeId::DOUBLE: {
		auto other = result_value.CastAs(set, get_input, LogicalType::DOUBLE);
		double ldecimal = value.value_.double_;
		double rdecimal = other.value_.double_;
		return ApproxEqual(ldecimal, rdecimal);
	}
	case LogicalTypeId::VARCHAR: {
		auto other = result_value.CastAs(set, get_input, LogicalType::VARCHAR);
		string left = SanitizeValue(other.str_value);
		string right = SanitizeValue(value.str_value);
		return left == right;
	}
	default:
		if (result_value.type_.id() == LogicalTypeId::FLOAT || result_value.type_.id() == LogicalTypeId::DOUBLE) {
			return Value::ValuesAreEqual(set, get_input, value, result_value);
		}
		return value == result_value;
	}
}

bool Value::ValuesAreEqual(ClientContext &context, const Value &result_value, const Value &value) {
	GetCastFunctionInput get_input(context);
	return Value::ValuesAreEqual(CastFunctionSet::Get(context), get_input, result_value, value);
}
bool Value::DefaultValuesAreEqual(const Value &result_value, const Value &value) {
	CastFunctionSet set;
	GetCastFunctionInput get_input;
	return Value::ValuesAreEqual(set, get_input, result_value, value);
}

} // namespace duckdb




















#include <cstring> // strlen() on Solaris

namespace duckdb {

Vector::Vector(LogicalType type_p, bool create_data, bool zero_data, idx_t capacity)
    : vector_type(VectorType::FLAT_VECTOR), type(move(type_p)), data(nullptr) {
	if (create_data) {
		Initialize(zero_data, capacity);
	}
}

Vector::Vector(LogicalType type_p, idx_t capacity) : Vector(move(type_p), true, false, capacity) {
}

Vector::Vector(LogicalType type_p, data_ptr_t dataptr)
    : vector_type(VectorType::FLAT_VECTOR), type(move(type_p)), data(dataptr) {
	if (dataptr && !type.IsValid()) {
		throw InternalException("Cannot create a vector of type INVALID!");
	}
}

Vector::Vector(const VectorCache &cache) : type(cache.GetType()) {
	ResetFromCache(cache);
}

Vector::Vector(Vector &other) : type(other.type) {
	Reference(other);
}

Vector::Vector(Vector &other, const SelectionVector &sel, idx_t count) : type(other.type) {
	Slice(other, sel, count);
}

Vector::Vector(Vector &other, idx_t offset, idx_t end) : type(other.type) {
	Slice(other, offset, end);
}

Vector::Vector(const Value &value) : type(value.type()) {
	Reference(value);
}

Vector::Vector(Vector &&other) noexcept
    : vector_type(other.vector_type), type(move(other.type)), data(other.data), validity(move(other.validity)),
      buffer(move(other.buffer)), auxiliary(move(other.auxiliary)) {
}

void Vector::Reference(const Value &value) {
	D_ASSERT(GetType().id() == value.type().id());
	this->vector_type = VectorType::CONSTANT_VECTOR;
	buffer = VectorBuffer::CreateConstantVector(value.type());
	auto internal_type = value.type().InternalType();
	if (internal_type == PhysicalType::STRUCT) {
		auto struct_buffer = make_unique<VectorStructBuffer>();
		auto &child_types = StructType::GetChildTypes(value.type());
		auto &child_vectors = struct_buffer->GetChildren();
		auto &value_children = StructValue::GetChildren(value);
		for (idx_t i = 0; i < child_types.size(); i++) {
			auto vector = make_unique<Vector>(value.IsNull() ? Value(child_types[i].second) : value_children[i]);
			child_vectors.push_back(move(vector));
		}
		auxiliary = move(struct_buffer);
		if (value.IsNull()) {
			SetValue(0, value);
		}
	} else if (internal_type == PhysicalType::LIST) {
		auto list_buffer = make_unique<VectorListBuffer>(value.type());
		auxiliary = move(list_buffer);
		data = buffer->GetData();
		SetValue(0, value);
	} else {
		auxiliary.reset();
		data = buffer->GetData();
		SetValue(0, value);
	}
}

void Vector::Reference(Vector &other) {
	D_ASSERT(other.GetType() == GetType());
	Reinterpret(other);
}

void Vector::ReferenceAndSetType(Vector &other) {
	type = other.GetType();
	Reference(other);
}

void Vector::Reinterpret(Vector &other) {
	vector_type = other.vector_type;
	AssignSharedPointer(buffer, other.buffer);
	AssignSharedPointer(auxiliary, other.auxiliary);
	data = other.data;
	validity = other.validity;
}

void Vector::ResetFromCache(const VectorCache &cache) {
	cache.ResetFromCache(*this);
}

void Vector::Slice(Vector &other, idx_t offset, idx_t end) {
	if (other.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		Reference(other);
		return;
	}
	D_ASSERT(other.GetVectorType() == VectorType::FLAT_VECTOR);

	auto internal_type = GetType().InternalType();
	if (internal_type == PhysicalType::STRUCT) {
		Vector new_vector(GetType());
		auto &entries = StructVector::GetEntries(new_vector);
		auto &other_entries = StructVector::GetEntries(other);
		D_ASSERT(entries.size() == other_entries.size());
		for (idx_t i = 0; i < entries.size(); i++) {
			entries[i]->Slice(*other_entries[i], offset, end);
		}
		if (offset > 0) {
			new_vector.validity.Slice(other.validity, offset, end);
		} else {
			new_vector.validity = other.validity;
		}
		Reference(new_vector);
	} else {
		Reference(other);
		if (offset > 0) {
			data = data + GetTypeIdSize(internal_type) * offset;
			validity.Slice(other.validity, offset, end);
		}
	}
}

void Vector::Slice(Vector &other, const SelectionVector &sel, idx_t count) {
	Reference(other);
	Slice(sel, count);
}

void Vector::Slice(const SelectionVector &sel, idx_t count) {
	if (GetVectorType() == VectorType::CONSTANT_VECTOR) {
		// dictionary on a constant is just a constant
		return;
	}
	if (GetVectorType() == VectorType::DICTIONARY_VECTOR) {
		// already a dictionary, slice the current dictionary
		auto &current_sel = DictionaryVector::SelVector(*this);
		auto sliced_dictionary = current_sel.Slice(sel, count);
		buffer = make_buffer<DictionaryBuffer>(move(sliced_dictionary));
		if (GetType().InternalType() == PhysicalType::STRUCT) {
			auto &child_vector = DictionaryVector::Child(*this);

			Vector new_child(child_vector);
			new_child.auxiliary = make_buffer<VectorStructBuffer>(new_child, sel, count);
			auxiliary = make_buffer<VectorChildBuffer>(move(new_child));
		}
		return;
	}

	if (GetVectorType() == VectorType::FSST_VECTOR) {
		Flatten(sel, count);
		return;
	}

	Vector child_vector(*this);
	auto internal_type = GetType().InternalType();
	if (internal_type == PhysicalType::STRUCT) {
		child_vector.auxiliary = make_buffer<VectorStructBuffer>(*this, sel, count);
	}
	auto child_ref = make_buffer<VectorChildBuffer>(move(child_vector));
	auto dict_buffer = make_buffer<DictionaryBuffer>(sel);
	vector_type = VectorType::DICTIONARY_VECTOR;
	buffer = move(dict_buffer);
	auxiliary = move(child_ref);
}

void Vector::Slice(const SelectionVector &sel, idx_t count, SelCache &cache) {
	if (GetVectorType() == VectorType::DICTIONARY_VECTOR && GetType().InternalType() != PhysicalType::STRUCT) {
		// dictionary vector: need to merge dictionaries
		// check if we have a cached entry
		auto &current_sel = DictionaryVector::SelVector(*this);
		auto target_data = current_sel.data();
		auto entry = cache.cache.find(target_data);
		if (entry != cache.cache.end()) {
			// cached entry exists: use that
			this->buffer = make_buffer<DictionaryBuffer>(((DictionaryBuffer &)*entry->second).GetSelVector());
			vector_type = VectorType::DICTIONARY_VECTOR;
		} else {
			Slice(sel, count);
			cache.cache[target_data] = this->buffer;
		}
	} else {
		Slice(sel, count);
	}
}

void Vector::Initialize(bool zero_data, idx_t capacity) {
	auxiliary.reset();
	validity.Reset();
	auto &type = GetType();
	auto internal_type = type.InternalType();
	if (internal_type == PhysicalType::STRUCT) {
		auto struct_buffer = make_unique<VectorStructBuffer>(type, capacity);
		auxiliary = move(struct_buffer);
	} else if (internal_type == PhysicalType::LIST) {
		auto list_buffer = make_unique<VectorListBuffer>(type, capacity);
		auxiliary = move(list_buffer);
	}
	auto type_size = GetTypeIdSize(internal_type);
	if (type_size > 0) {
		buffer = VectorBuffer::CreateStandardVector(type, capacity);
		data = buffer->GetData();
		if (zero_data) {
			memset(data, 0, capacity * type_size);
		}
	}
	if (capacity > STANDARD_VECTOR_SIZE) {
		validity.Resize(STANDARD_VECTOR_SIZE, capacity);
	}
}

struct DataArrays {
	Vector &vec;
	data_ptr_t data;
	VectorBuffer *buffer;
	idx_t type_size;
	bool is_nested;
	DataArrays(Vector &vec, data_ptr_t data, VectorBuffer *buffer, idx_t type_size, bool is_nested)
	    : vec(vec), data(data), buffer(buffer), type_size(type_size), is_nested(is_nested) {};
};

void FindChildren(std::vector<DataArrays> &to_resize, VectorBuffer &auxiliary) {
	if (auxiliary.GetBufferType() == VectorBufferType::LIST_BUFFER) {
		auto &buffer = (VectorListBuffer &)auxiliary;
		auto &child = buffer.GetChild();
		auto data = child.GetData();
		if (!data) {
			//! Nested type
			DataArrays arrays(child, data, child.GetBuffer().get(), GetTypeIdSize(child.GetType().InternalType()),
			                  true);
			to_resize.emplace_back(arrays);
			FindChildren(to_resize, *child.GetAuxiliary());
		} else {
			DataArrays arrays(child, data, child.GetBuffer().get(), GetTypeIdSize(child.GetType().InternalType()),
			                  false);
			to_resize.emplace_back(arrays);
		}
	} else if (auxiliary.GetBufferType() == VectorBufferType::STRUCT_BUFFER) {
		auto &buffer = (VectorStructBuffer &)auxiliary;
		auto &children = buffer.GetChildren();
		for (auto &child : children) {
			auto data = child->GetData();
			if (!data) {
				//! Nested type
				DataArrays arrays(*child, data, child->GetBuffer().get(),
				                  GetTypeIdSize(child->GetType().InternalType()), true);
				to_resize.emplace_back(arrays);
				FindChildren(to_resize, *child->GetAuxiliary());
			} else {
				DataArrays arrays(*child, data, child->GetBuffer().get(),
				                  GetTypeIdSize(child->GetType().InternalType()), false);
				to_resize.emplace_back(arrays);
			}
		}
	}
}
void Vector::Resize(idx_t cur_size, idx_t new_size) {
	std::vector<DataArrays> to_resize;
	if (!buffer) {
		buffer = make_unique<VectorBuffer>(0);
	}
	if (!data) {
		//! this is a nested structure
		DataArrays arrays(*this, data, buffer.get(), GetTypeIdSize(GetType().InternalType()), true);
		to_resize.emplace_back(arrays);
		FindChildren(to_resize, *auxiliary);
	} else {
		DataArrays arrays(*this, data, buffer.get(), GetTypeIdSize(GetType().InternalType()), false);
		to_resize.emplace_back(arrays);
	}
	for (auto &data_to_resize : to_resize) {
		if (!data_to_resize.is_nested) {
			auto new_data = unique_ptr<data_t[]>(new data_t[new_size * data_to_resize.type_size]);
			memcpy(new_data.get(), data_to_resize.data, cur_size * data_to_resize.type_size * sizeof(data_t));
			data_to_resize.buffer->SetData(move(new_data));
			data_to_resize.vec.data = data_to_resize.buffer->GetData();
		}
		data_to_resize.vec.validity.Resize(cur_size, new_size);
	}
}

void Vector::SetValue(idx_t index, const Value &val) {
	if (GetVectorType() == VectorType::DICTIONARY_VECTOR) {
		// dictionary: apply dictionary and forward to child
		auto &sel_vector = DictionaryVector::SelVector(*this);
		auto &child = DictionaryVector::Child(*this);
		return child.SetValue(sel_vector.get_index(index), val);
	}
	if (val.type() != GetType()) {
		SetValue(index, val.DefaultCastAs(GetType()));
		return;
	}
	D_ASSERT(val.type().InternalType() == GetType().InternalType());

	validity.EnsureWritable();
	validity.Set(index, !val.IsNull());
	if (val.IsNull() && GetType().InternalType() != PhysicalType::STRUCT) {
		// for structs we still need to set the child-entries to NULL
		// so we do not bail out yet
		return;
	}

	switch (GetType().InternalType()) {
	case PhysicalType::BOOL:
		((bool *)data)[index] = val.GetValueUnsafe<bool>();
		break;
	case PhysicalType::INT8:
		((int8_t *)data)[index] = val.GetValueUnsafe<int8_t>();
		break;
	case PhysicalType::INT16:
		((int16_t *)data)[index] = val.GetValueUnsafe<int16_t>();
		break;
	case PhysicalType::INT32:
		((int32_t *)data)[index] = val.GetValueUnsafe<int32_t>();
		break;
	case PhysicalType::INT64:
		((int64_t *)data)[index] = val.GetValueUnsafe<int64_t>();
		break;
	case PhysicalType::INT128:
		((hugeint_t *)data)[index] = val.GetValueUnsafe<hugeint_t>();
		break;
	case PhysicalType::UINT8:
		((uint8_t *)data)[index] = val.GetValueUnsafe<uint8_t>();
		break;
	case PhysicalType::UINT16:
		((uint16_t *)data)[index] = val.GetValueUnsafe<uint16_t>();
		break;
	case PhysicalType::UINT32:
		((uint32_t *)data)[index] = val.GetValueUnsafe<uint32_t>();
		break;
	case PhysicalType::UINT64:
		((uint64_t *)data)[index] = val.GetValueUnsafe<uint64_t>();
		break;
	case PhysicalType::FLOAT:
		((float *)data)[index] = val.GetValueUnsafe<float>();
		break;
	case PhysicalType::DOUBLE:
		((double *)data)[index] = val.GetValueUnsafe<double>();
		break;
	case PhysicalType::INTERVAL:
		((interval_t *)data)[index] = val.GetValueUnsafe<interval_t>();
		break;
	case PhysicalType::VARCHAR:
		((string_t *)data)[index] = StringVector::AddStringOrBlob(*this, StringValue::Get(val));
		break;
	case PhysicalType::STRUCT: {
		D_ASSERT(GetVectorType() == VectorType::CONSTANT_VECTOR || GetVectorType() == VectorType::FLAT_VECTOR);

		auto &children = StructVector::GetEntries(*this);
		auto &val_children = StructValue::GetChildren(val);
		D_ASSERT(val.IsNull() || children.size() == val_children.size());
		for (size_t i = 0; i < children.size(); i++) {
			auto &vec_child = children[i];
			if (!val.IsNull()) {
				auto &struct_child = val_children[i];
				vec_child->SetValue(index, struct_child);
			} else {
				vec_child->SetValue(index, Value());
			}
		}
		break;
	}
	case PhysicalType::LIST: {
		auto offset = ListVector::GetListSize(*this);
		auto &val_children = ListValue::GetChildren(val);
		if (!val_children.empty()) {
			for (idx_t i = 0; i < val_children.size(); i++) {
				ListVector::PushBack(*this, val_children[i]);
			}
		}
		//! now set the pointer
		auto &entry = ((list_entry_t *)data)[index];
		entry.length = val_children.size();
		entry.offset = offset;
		break;
	}
	default:
		throw InternalException("Unimplemented type for Vector::SetValue");
	}
}

Value Vector::GetValueInternal(const Vector &v_p, idx_t index_p) {
	const Vector *vector = &v_p;
	idx_t index = index_p;
	bool finished = false;
	while (!finished) {
		switch (vector->GetVectorType()) {
		case VectorType::CONSTANT_VECTOR:
			index = 0;
			finished = true;
			break;
		case VectorType::FLAT_VECTOR:
			finished = true;
			break;
		case VectorType::FSST_VECTOR:
			finished = true;
			break;
		// dictionary: apply dictionary and forward to child
		case VectorType::DICTIONARY_VECTOR: {
			auto &sel_vector = DictionaryVector::SelVector(*vector);
			auto &child = DictionaryVector::Child(*vector);
			vector = &child;
			index = sel_vector.get_index(index);
			break;
		}
		case VectorType::SEQUENCE_VECTOR: {
			int64_t start, increment;
			SequenceVector::GetSequence(*vector, start, increment);
			return Value::Numeric(vector->GetType(), start + increment * index);
		}
		default:
			throw InternalException("Unimplemented vector type for Vector::GetValue");
		}
	}
	auto data = vector->data;
	auto &validity = vector->validity;
	auto &type = vector->GetType();

	if (!validity.RowIsValid(index)) {
		return Value(vector->GetType());
	}

	if (vector->GetVectorType() == VectorType::FSST_VECTOR) {
		if (vector->GetType().InternalType() != PhysicalType::VARCHAR) {
			throw InternalException("FSST Vector with non-string datatype found!");
		}
		auto str_compressed = ((string_t *)data)[index];
		Value result =
		    FSSTPrimitives::DecompressValue(FSSTVector::GetDecoder(const_cast<Vector &>(*vector)),
		                                    (unsigned char *)str_compressed.GetDataUnsafe(), str_compressed.GetSize());
		return result;
	}

	switch (vector->GetType().id()) {
	case LogicalTypeId::BOOLEAN:
		return Value::BOOLEAN(((bool *)data)[index]);
	case LogicalTypeId::TINYINT:
		return Value::TINYINT(((int8_t *)data)[index]);
	case LogicalTypeId::SMALLINT:
		return Value::SMALLINT(((int16_t *)data)[index]);
	case LogicalTypeId::INTEGER:
		return Value::INTEGER(((int32_t *)data)[index]);
	case LogicalTypeId::DATE:
		return Value::DATE(((date_t *)data)[index]);
	case LogicalTypeId::TIME:
		return Value::TIME(((dtime_t *)data)[index]);
	case LogicalTypeId::TIME_TZ:
		return Value::TIMETZ(((dtime_t *)data)[index]);
	case LogicalTypeId::BIGINT:
		return Value::BIGINT(((int64_t *)data)[index]);
	case LogicalTypeId::UTINYINT:
		return Value::UTINYINT(((uint8_t *)data)[index]);
	case LogicalTypeId::USMALLINT:
		return Value::USMALLINT(((uint16_t *)data)[index]);
	case LogicalTypeId::UINTEGER:
		return Value::UINTEGER(((uint32_t *)data)[index]);
	case LogicalTypeId::UBIGINT:
		return Value::UBIGINT(((uint64_t *)data)[index]);
	case LogicalTypeId::TIMESTAMP:
		return Value::TIMESTAMP(((timestamp_t *)data)[index]);
	case LogicalTypeId::TIMESTAMP_NS:
		return Value::TIMESTAMPNS(((timestamp_t *)data)[index]);
	case LogicalTypeId::TIMESTAMP_MS:
		return Value::TIMESTAMPMS(((timestamp_t *)data)[index]);
	case LogicalTypeId::TIMESTAMP_SEC:
		return Value::TIMESTAMPSEC(((timestamp_t *)data)[index]);
	case LogicalTypeId::TIMESTAMP_TZ:
		return Value::TIMESTAMPTZ(((timestamp_t *)data)[index]);
	case LogicalTypeId::HUGEINT:
		return Value::HUGEINT(((hugeint_t *)data)[index]);
	case LogicalTypeId::UUID:
		return Value::UUID(((hugeint_t *)data)[index]);
	case LogicalTypeId::DECIMAL: {
		auto width = DecimalType::GetWidth(type);
		auto scale = DecimalType::GetScale(type);
		switch (type.InternalType()) {
		case PhysicalType::INT16:
			return Value::DECIMAL(((int16_t *)data)[index], width, scale);
		case PhysicalType::INT32:
			return Value::DECIMAL(((int32_t *)data)[index], width, scale);
		case PhysicalType::INT64:
			return Value::DECIMAL(((int64_t *)data)[index], width, scale);
		case PhysicalType::INT128:
			return Value::DECIMAL(((hugeint_t *)data)[index], width, scale);
		default:
			throw InternalException("Physical type '%s' has a width bigger than 38, which is not supported",
			                        TypeIdToString(type.InternalType()));
		}
	}
	case LogicalTypeId::ENUM: {
		switch (type.InternalType()) {
		case PhysicalType::UINT8:
			return Value::ENUM(((uint8_t *)data)[index], type);
		case PhysicalType::UINT16:
			return Value::ENUM(((uint16_t *)data)[index], type);
		case PhysicalType::UINT32:
			return Value::ENUM(((uint32_t *)data)[index], type);
		case PhysicalType::UINT64: //  DEDUP_POINTER_ENUM
			return Value::ENUM(((uint64_t *)data)[index], type);
		default:
			throw InternalException("ENUM can only have unsigned integers as physical types");
		}
	}
	case LogicalTypeId::POINTER:
		return Value::POINTER(((uintptr_t *)data)[index]);
	case LogicalTypeId::FLOAT:
		return Value::FLOAT(((float *)data)[index]);
	case LogicalTypeId::DOUBLE:
		return Value::DOUBLE(((double *)data)[index]);
	case LogicalTypeId::INTERVAL:
		return Value::INTERVAL(((interval_t *)data)[index]);
	case LogicalTypeId::VARCHAR: {
		auto str = ((string_t *)data)[index];
		return Value(str.GetString());
	}
	case LogicalTypeId::JSON: {
		auto str = ((string_t *)data)[index];
		return Value::JSON(str.GetString());
	}
	case LogicalTypeId::AGGREGATE_STATE:
	case LogicalTypeId::BLOB: {
		auto str = ((string_t *)data)[index];
		return Value::BLOB((const_data_ptr_t)str.GetDataUnsafe(), str.GetSize());
	}
	case LogicalTypeId::MAP: {
		auto &child_entries = StructVector::GetEntries(*vector);
		Value key = child_entries[0]->GetValue(index);
		Value value = child_entries[1]->GetValue(index);
		return Value::MAP(move(key), move(value));
	}
	case LogicalTypeId::UNION: {
		auto tag = UnionVector::GetTag(*vector, index);
		auto value = UnionVector::GetMember(*vector, tag).GetValue(index);
		auto members = UnionType::CopyMemberTypes(type);
		return Value::UNION(members, tag, move(value));
	}
	case LogicalTypeId::STRUCT: {
		// we can derive the value schema from the vector schema
		auto &child_entries = StructVector::GetEntries(*vector);
		child_list_t<Value> children;
		for (idx_t child_idx = 0; child_idx < child_entries.size(); child_idx++) {
			auto &struct_child = child_entries[child_idx];
			children.push_back(make_pair(StructType::GetChildName(type, child_idx), struct_child->GetValue(index_p)));
		}
		return Value::STRUCT(move(children));
	}
	case LogicalTypeId::LIST: {
		auto offlen = ((list_entry_t *)data)[index];
		auto &child_vec = ListVector::GetEntry(*vector);
		std::vector<Value> children;
		for (idx_t i = offlen.offset; i < offlen.offset + offlen.length; i++) {
			children.push_back(child_vec.GetValue(i));
		}
		return Value::LIST(ListType::GetChildType(type), move(children));
	}
	default:
		throw InternalException("Unimplemented type for value access");
	}
}

Value Vector::GetValue(const Vector &v_p, idx_t index_p) {
	auto value = GetValueInternal(v_p, index_p);
	// set the alias of the type to the correct value, if there is a type alias
	if (v_p.GetType().HasAlias()) {
		value.type().CopyAuxInfo(v_p.GetType());
	}
	if (v_p.GetType().id() != LogicalTypeId::AGGREGATE_STATE && value.type().id() != LogicalTypeId::AGGREGATE_STATE) {
		D_ASSERT(v_p.GetType() == value.type());
	}
	return value;
}

Value Vector::GetValue(idx_t index) const {
	return GetValue(*this, index);
}

// LCOV_EXCL_START
string VectorTypeToString(VectorType type) {
	switch (type) {
	case VectorType::FLAT_VECTOR:
		return "FLAT";
	case VectorType::FSST_VECTOR:
		return "FSST";
	case VectorType::SEQUENCE_VECTOR:
		return "SEQUENCE";
	case VectorType::DICTIONARY_VECTOR:
		return "DICTIONARY";
	case VectorType::CONSTANT_VECTOR:
		return "CONSTANT";
	default:
		return "UNKNOWN";
	}
}

string Vector::ToString(idx_t count) const {
	string retval =
	    VectorTypeToString(GetVectorType()) + " " + GetType().ToString() + ": " + to_string(count) + " = [ ";
	switch (GetVectorType()) {
	case VectorType::FLAT_VECTOR:
	case VectorType::DICTIONARY_VECTOR:
		for (idx_t i = 0; i < count; i++) {
			retval += GetValue(i).ToString() + (i == count - 1 ? "" : ", ");
		}
		break;
	case VectorType::FSST_VECTOR: {
		for (idx_t i = 0; i < count; i++) {
			string_t compressed_string = ((string_t *)data)[i];
			Value val = FSSTPrimitives::DecompressValue(FSSTVector::GetDecoder(const_cast<Vector &>(*this)),
			                                            (unsigned char *)compressed_string.GetDataUnsafe(),
			                                            compressed_string.GetSize());
			retval += GetValue(i).ToString() + (i == count - 1 ? "" : ", ");
		}
	} break;
	case VectorType::CONSTANT_VECTOR:
		retval += GetValue(0).ToString();
		break;
	case VectorType::SEQUENCE_VECTOR: {
		int64_t start, increment;
		SequenceVector::GetSequence(*this, start, increment);
		for (idx_t i = 0; i < count; i++) {
			retval += to_string(start + increment * i) + (i == count - 1 ? "" : ", ");
		}
		break;
	}
	default:
		retval += "UNKNOWN VECTOR TYPE";
		break;
	}
	retval += "]";
	return retval;
}

void Vector::Print(idx_t count) {
	Printer::Print(ToString(count));
}

string Vector::ToString() const {
	string retval = VectorTypeToString(GetVectorType()) + " " + GetType().ToString() + ": (UNKNOWN COUNT) [ ";
	switch (GetVectorType()) {
	case VectorType::FLAT_VECTOR:
	case VectorType::DICTIONARY_VECTOR:
		break;
	case VectorType::CONSTANT_VECTOR:
		retval += GetValue(0).ToString();
		break;
	case VectorType::SEQUENCE_VECTOR: {
		break;
	}
	default:
		retval += "UNKNOWN VECTOR TYPE";
		break;
	}
	retval += "]";
	return retval;
}

void Vector::Print() {
	Printer::Print(ToString());
}
// LCOV_EXCL_STOP

template <class T>
static void TemplatedFlattenConstantVector(data_ptr_t data, data_ptr_t old_data, idx_t count) {
	auto constant = Load<T>(old_data);
	auto output = (T *)data;
	for (idx_t i = 0; i < count; i++) {
		output[i] = constant;
	}
}

void Vector::Flatten(idx_t count) {
	switch (GetVectorType()) {
	case VectorType::FLAT_VECTOR:
		// already a flat vector
		break;
	case VectorType::FSST_VECTOR: {
		// Even though count may only be a part of the vector, we need to flatten the whole thing due to the way
		// ToUnifiedFormat uses flatten
		idx_t total_count = FSSTVector::GetCount(*this);
		// create vector to decompress into
		Vector other(GetType(), total_count);
		// now copy the data of this vector to the other vector, decompressing the strings in the process
		VectorOperations::Copy(*this, other, total_count, 0, 0);
		// create a reference to the data in the other vector
		this->Reference(other);
		break;
	}
	case VectorType::DICTIONARY_VECTOR: {
		// create a new flat vector of this type
		Vector other(GetType(), count);
		// now copy the data of this vector to the other vector, removing the selection vector in the process
		VectorOperations::Copy(*this, other, count, 0, 0);
		// create a reference to the data in the other vector
		this->Reference(other);
		break;
	}
	case VectorType::CONSTANT_VECTOR: {
		bool is_null = ConstantVector::IsNull(*this);
		// allocate a new buffer for the vector
		auto old_buffer = move(buffer);
		auto old_data = data;
		buffer = VectorBuffer::CreateStandardVector(type, MaxValue<idx_t>(STANDARD_VECTOR_SIZE, count));
		data = buffer->GetData();
		vector_type = VectorType::FLAT_VECTOR;
		if (is_null) {
			// constant NULL, set nullmask
			validity.EnsureWritable();
			validity.SetAllInvalid(count);
			return;
		}
		// non-null constant: have to repeat the constant
		switch (GetType().InternalType()) {
		case PhysicalType::BOOL:
			TemplatedFlattenConstantVector<bool>(data, old_data, count);
			break;
		case PhysicalType::INT8:
			TemplatedFlattenConstantVector<int8_t>(data, old_data, count);
			break;
		case PhysicalType::INT16:
			TemplatedFlattenConstantVector<int16_t>(data, old_data, count);
			break;
		case PhysicalType::INT32:
			TemplatedFlattenConstantVector<int32_t>(data, old_data, count);
			break;
		case PhysicalType::INT64:
			TemplatedFlattenConstantVector<int64_t>(data, old_data, count);
			break;
		case PhysicalType::UINT8:
			TemplatedFlattenConstantVector<uint8_t>(data, old_data, count);
			break;
		case PhysicalType::UINT16:
			TemplatedFlattenConstantVector<uint16_t>(data, old_data, count);
			break;
		case PhysicalType::UINT32:
			TemplatedFlattenConstantVector<uint32_t>(data, old_data, count);
			break;
		case PhysicalType::UINT64:
			TemplatedFlattenConstantVector<uint64_t>(data, old_data, count);
			break;
		case PhysicalType::INT128:
			TemplatedFlattenConstantVector<hugeint_t>(data, old_data, count);
			break;
		case PhysicalType::FLOAT:
			TemplatedFlattenConstantVector<float>(data, old_data, count);
			break;
		case PhysicalType::DOUBLE:
			TemplatedFlattenConstantVector<double>(data, old_data, count);
			break;
		case PhysicalType::INTERVAL:
			TemplatedFlattenConstantVector<interval_t>(data, old_data, count);
			break;
		case PhysicalType::VARCHAR:
			TemplatedFlattenConstantVector<string_t>(data, old_data, count);
			break;
		case PhysicalType::LIST: {
			TemplatedFlattenConstantVector<list_entry_t>(data, old_data, count);
			break;
		}
		case PhysicalType::STRUCT: {
			auto normalified_buffer = make_unique<VectorStructBuffer>();

			auto &new_children = normalified_buffer->GetChildren();

			auto &child_entries = StructVector::GetEntries(*this);
			for (auto &child : child_entries) {
				D_ASSERT(child->GetVectorType() == VectorType::CONSTANT_VECTOR);
				auto vector = make_unique<Vector>(*child);
				vector->Flatten(count);
				new_children.push_back(move(vector));
			}
			auxiliary = move(normalified_buffer);
		} break;
		default:
			throw InternalException("Unimplemented type for VectorOperations::Flatten");
		}
		break;
	}
	case VectorType::SEQUENCE_VECTOR: {
		int64_t start, increment, sequence_count;
		SequenceVector::GetSequence(*this, start, increment, sequence_count);

		buffer = VectorBuffer::CreateStandardVector(GetType());
		data = buffer->GetData();
		VectorOperations::GenerateSequence(*this, sequence_count, start, increment);
		break;
	}
	default:
		throw InternalException("Unimplemented type for normalify");
	}
}

void Vector::Flatten(const SelectionVector &sel, idx_t count) {
	switch (GetVectorType()) {
	case VectorType::FLAT_VECTOR:
		// already a flat vector
		break;
	case VectorType::FSST_VECTOR: {
		// create a new flat vector of this type
		Vector other(GetType());
		// copy the data of this vector to the other vector, removing compression and selection vector in the process
		VectorOperations::Copy(*this, other, sel, count, 0, 0);
		// create a reference to the data in the other vector
		this->Reference(other);
		break;
	}
	case VectorType::SEQUENCE_VECTOR: {
		int64_t start, increment;
		SequenceVector::GetSequence(*this, start, increment);

		buffer = VectorBuffer::CreateStandardVector(GetType());
		data = buffer->GetData();
		VectorOperations::GenerateSequence(*this, count, sel, start, increment);
		break;
	}
	default:
		throw InternalException("Unimplemented type for normalify with selection vector");
	}
}

void Vector::ToUnifiedFormat(idx_t count, UnifiedVectorFormat &data) {
	switch (GetVectorType()) {
	case VectorType::DICTIONARY_VECTOR: {
		auto &sel = DictionaryVector::SelVector(*this);
		auto &child = DictionaryVector::Child(*this);
		if (child.GetVectorType() == VectorType::FLAT_VECTOR) {
			data.sel = &sel;
			data.data = FlatVector::GetData(child);
			data.validity = FlatVector::Validity(child);
		} else {
			// dictionary with non-flat child: create a new reference to the child and normalify it
			Vector child_vector(child);
			child_vector.Flatten(sel, count);
			auto new_aux = make_buffer<VectorChildBuffer>(move(child_vector));

			data.sel = &sel;
			data.data = FlatVector::GetData(new_aux->data);
			data.validity = FlatVector::Validity(new_aux->data);
			this->auxiliary = move(new_aux);
		}
		break;
	}
	case VectorType::CONSTANT_VECTOR:
		data.sel = ConstantVector::ZeroSelectionVector(count, data.owned_sel);
		data.data = ConstantVector::GetData(*this);
		data.validity = ConstantVector::Validity(*this);
		break;
	default:
		Flatten(count);
		data.sel = FlatVector::IncrementalSelectionVector();
		data.data = FlatVector::GetData(*this);
		data.validity = FlatVector::Validity(*this);
		break;
	}
}

void Vector::Sequence(int64_t start, int64_t increment, idx_t count) {
	this->vector_type = VectorType::SEQUENCE_VECTOR;
	this->buffer = make_buffer<VectorBuffer>(sizeof(int64_t) * 3);
	auto data = (int64_t *)buffer->GetData();
	data[0] = start;
	data[1] = increment;
	data[2] = int64_t(count);
	validity.Reset();
	auxiliary.reset();
}

void Vector::Serialize(idx_t count, Serializer &serializer) {
	auto &type = GetType();

	UnifiedVectorFormat vdata;
	ToUnifiedFormat(count, vdata);

	const auto write_validity = (count > 0) && !vdata.validity.AllValid();
	serializer.Write<bool>(write_validity);
	if (write_validity) {
		ValidityMask flat_mask(count);
		for (idx_t i = 0; i < count; ++i) {
			auto row_idx = vdata.sel->get_index(i);
			flat_mask.Set(i, vdata.validity.RowIsValid(row_idx));
		}
		serializer.WriteData((const_data_ptr_t)flat_mask.GetData(), flat_mask.ValidityMaskSize(count));
	}
	if (TypeIsConstantSize(type.InternalType())) {
		// constant size type: simple copy
		idx_t write_size = GetTypeIdSize(type.InternalType()) * count;
		auto ptr = unique_ptr<data_t[]>(new data_t[write_size]);
		VectorOperations::WriteToStorage(*this, count, ptr.get());
		serializer.WriteData(ptr.get(), write_size);
	} else {
		switch (type.InternalType()) {
		case PhysicalType::VARCHAR: {
			auto strings = (string_t *)vdata.data;
			for (idx_t i = 0; i < count; i++) {
				auto idx = vdata.sel->get_index(i);
				auto source = !vdata.validity.RowIsValid(idx) ? NullValue<string_t>() : strings[idx];
				serializer.WriteStringLen((const_data_ptr_t)source.GetDataUnsafe(), source.GetSize());
			}
			break;
		}
		case PhysicalType::STRUCT: {
			Flatten(count);
			auto &entries = StructVector::GetEntries(*this);
			for (auto &entry : entries) {
				entry->Serialize(count, serializer);
			}
			break;
		}
		case PhysicalType::LIST: {
			auto &child = ListVector::GetEntry(*this);
			auto list_size = ListVector::GetListSize(*this);

			// serialize the list entries in a flat array
			auto data = unique_ptr<list_entry_t[]>(new list_entry_t[count]);
			auto source_array = (list_entry_t *)vdata.data;
			for (idx_t i = 0; i < count; i++) {
				auto idx = vdata.sel->get_index(i);
				auto source = source_array[idx];
				data[i].offset = source.offset;
				data[i].length = source.length;
			}

			// write the list size
			serializer.Write<idx_t>(list_size);
			serializer.WriteData((data_ptr_t)data.get(), count * sizeof(list_entry_t));

			child.Serialize(list_size, serializer);
			break;
		}
		default:
			throw InternalException("Unimplemented variable width type for Vector::Serialize!");
		}
	}
}

void Vector::Deserialize(idx_t count, Deserializer &source) {
	auto &type = GetType();

	auto &validity = FlatVector::Validity(*this);
	validity.Reset();
	const auto has_validity = source.Read<bool>();
	if (has_validity) {
		validity.Initialize(count);
		source.ReadData((data_ptr_t)validity.GetData(), validity.ValidityMaskSize(count));
	}

	if (TypeIsConstantSize(type.InternalType())) {
		// constant size type: read fixed amount of data from
		auto column_size = GetTypeIdSize(type.InternalType()) * count;
		auto ptr = unique_ptr<data_t[]>(new data_t[column_size]);
		source.ReadData(ptr.get(), column_size);

		VectorOperations::ReadFromStorage(ptr.get(), count, *this);
	} else {
		switch (type.InternalType()) {
		case PhysicalType::VARCHAR: {
			auto strings = FlatVector::GetData<string_t>(*this);
			for (idx_t i = 0; i < count; i++) {
				// read the strings
				auto str = source.Read<string>();
				// now add the string to the StringHeap of the vector
				// and write the pointer into the vector
				if (validity.RowIsValid(i)) {
					strings[i] = StringVector::AddStringOrBlob(*this, str);
				}
			}
			break;
		}
		case PhysicalType::STRUCT: {
			auto &entries = StructVector::GetEntries(*this);
			for (auto &entry : entries) {
				entry->Deserialize(count, source);
			}
			break;
		}
		case PhysicalType::LIST: {
			// read the list size
			auto list_size = source.Read<idx_t>();
			ListVector::Reserve(*this, list_size);
			ListVector::SetListSize(*this, list_size);

			// read the list entry
			auto list_entries = FlatVector::GetData(*this);
			source.ReadData(list_entries, count * sizeof(list_entry_t));

			// deserialize the child vector
			auto &child = ListVector::GetEntry(*this);
			child.Deserialize(list_size, source);

			break;
		}
		default:
			throw InternalException("Unimplemented variable width type for Vector::Deserialize!");
		}
	}
}

void Vector::SetVectorType(VectorType vector_type_p) {
	this->vector_type = vector_type_p;
	if (TypeIsConstantSize(GetType().InternalType()) &&
	    (GetVectorType() == VectorType::CONSTANT_VECTOR || GetVectorType() == VectorType::FLAT_VECTOR)) {
		auxiliary.reset();
	}
	if (vector_type == VectorType::CONSTANT_VECTOR && GetType().InternalType() == PhysicalType::STRUCT) {
		auto &entries = StructVector::GetEntries(*this);
		for (auto &entry : entries) {
			entry->SetVectorType(vector_type);
		}
	}
}

void Vector::UTFVerify(const SelectionVector &sel, idx_t count) {
#ifdef DEBUG
	if (count == 0) {
		return;
	}
	if (GetType().InternalType() == PhysicalType::VARCHAR) {
		// we just touch all the strings and let the sanitizer figure out if any
		// of them are deallocated/corrupt
		switch (GetVectorType()) {
		case VectorType::CONSTANT_VECTOR: {
			auto string = ConstantVector::GetData<string_t>(*this);
			if (!ConstantVector::IsNull(*this)) {
				string->Verify();
			}
			break;
		}
		case VectorType::FLAT_VECTOR: {
			auto strings = FlatVector::GetData<string_t>(*this);
			for (idx_t i = 0; i < count; i++) {
				auto oidx = sel.get_index(i);
				if (validity.RowIsValid(oidx)) {
					strings[oidx].Verify();
				}
			}
			break;
		}
		default:
			break;
		}
	}
#endif
}

void Vector::UTFVerify(idx_t count) {
	auto flat_sel = FlatVector::IncrementalSelectionVector();

	UTFVerify(*flat_sel, count);
}

void Vector::VerifyMap(Vector &vector_p, const SelectionVector &sel_p, idx_t count) {
#ifdef DEBUG
	D_ASSERT(vector_p.GetType().id() == LogicalTypeId::MAP);
	auto valid_check = CheckMapValidity(vector_p, count, sel_p);
	D_ASSERT(valid_check == MapInvalidReason::VALID);
#endif // DEBUG
}

void Vector::VerifyUnion(Vector &vector_p, const SelectionVector &sel_p, idx_t count) {
#ifdef DEBUG
	D_ASSERT(vector_p.GetType().id() == LogicalTypeId::UNION);
	auto valid_check = CheckUnionValidity(vector_p, count, sel_p);
	D_ASSERT(valid_check == UnionInvalidReason::VALID);
#endif // DEBUG
}

void Vector::Verify(Vector &vector_p, const SelectionVector &sel_p, idx_t count) {
#ifdef DEBUG
	if (count == 0) {
		return;
	}
	Vector *vector = &vector_p;
	const SelectionVector *sel = &sel_p;
	SelectionVector owned_sel;
	auto &type = vector->GetType();
	auto vtype = vector->GetVectorType();
	if (vector->GetVectorType() == VectorType::DICTIONARY_VECTOR) {
		auto &child = DictionaryVector::Child(*vector);
		D_ASSERT(child.GetVectorType() != VectorType::DICTIONARY_VECTOR);
		auto &dict_sel = DictionaryVector::SelVector(*vector);
		// merge the selection vectors and verify the child
		auto new_buffer = dict_sel.Slice(*sel, count);
		owned_sel.Initialize(new_buffer);
		sel = &owned_sel;
		vector = &child;
		vtype = vector->GetVectorType();
	}
	if (TypeIsConstantSize(type.InternalType()) &&
	    (vtype == VectorType::CONSTANT_VECTOR || vtype == VectorType::FLAT_VECTOR)) {
		D_ASSERT(!vector->auxiliary);
	}
	if (type.id() == LogicalTypeId::VARCHAR || type.id() == LogicalTypeId::JSON) {
		// verify that the string is correct unicode
		switch (vtype) {
		case VectorType::FLAT_VECTOR: {
			auto &validity = FlatVector::Validity(*vector);
			auto strings = FlatVector::GetData<string_t>(*vector);
			for (idx_t i = 0; i < count; i++) {
				auto oidx = sel->get_index(i);
				if (validity.RowIsValid(oidx)) {
					strings[oidx].Verify();
				}
			}
			break;
		}
		default:
			break;
		}
	}

	if (type.InternalType() == PhysicalType::STRUCT) {
		auto &child_types = StructType::GetChildTypes(type);
		D_ASSERT(!child_types.empty());
		// create a selection vector of the non-null entries of the struct vector
		auto &children = StructVector::GetEntries(*vector);
		D_ASSERT(child_types.size() == children.size());
		for (idx_t child_idx = 0; child_idx < children.size(); child_idx++) {
			D_ASSERT(children[child_idx]->GetType() == child_types[child_idx].second);
			Vector::Verify(*children[child_idx], sel_p, count);
			if (vtype == VectorType::CONSTANT_VECTOR) {
				D_ASSERT(children[child_idx]->GetVectorType() == VectorType::CONSTANT_VECTOR);
				if (ConstantVector::IsNull(*vector)) {
					D_ASSERT(ConstantVector::IsNull(*children[child_idx]));
				}
			}
			if (vtype != VectorType::FLAT_VECTOR) {
				continue;
			}
			ValidityMask *child_validity;
			SelectionVector owned_child_sel;
			const SelectionVector *child_sel = &owned_child_sel;
			if (children[child_idx]->GetVectorType() == VectorType::FLAT_VECTOR) {
				child_sel = FlatVector::IncrementalSelectionVector();
				child_validity = &FlatVector::Validity(*children[child_idx]);
			} else if (children[child_idx]->GetVectorType() == VectorType::DICTIONARY_VECTOR) {
				auto &child = DictionaryVector::Child(*children[child_idx]);
				if (child.GetVectorType() != VectorType::FLAT_VECTOR) {
					continue;
				}
				child_validity = &FlatVector::Validity(child);
				child_sel = &DictionaryVector::SelVector(*children[child_idx]);
			} else if (children[child_idx]->GetVectorType() == VectorType::CONSTANT_VECTOR) {
				child_sel = ConstantVector::ZeroSelectionVector(count, owned_child_sel);
				child_validity = &ConstantVector::Validity(*children[child_idx]);
			} else {
				continue;
			}
			// for any NULL entry in the struct, the child should be NULL as well
			auto &validity = FlatVector::Validity(*vector);
			for (idx_t i = 0; i < count; i++) {
				auto index = sel->get_index(i);
				if (!validity.RowIsValid(index)) {
					auto child_index = child_sel->get_index(sel_p.get_index(i));
					D_ASSERT(!child_validity->RowIsValid(child_index));
				}
			}
		}
		if (vector->GetType().id() == LogicalTypeId::MAP) {
			VerifyMap(*vector, *sel, count);
		}

		if (vector->GetType().id() == LogicalTypeId::UNION) {
			VerifyUnion(*vector, *sel, count);
		}
	}

	if (type.InternalType() == PhysicalType::LIST) {
		if (vtype == VectorType::CONSTANT_VECTOR) {
			if (!ConstantVector::IsNull(*vector)) {
				auto &child = ListVector::GetEntry(*vector);
				SelectionVector child_sel(ListVector::GetListSize(*vector));
				idx_t child_count = 0;
				auto le = ConstantVector::GetData<list_entry_t>(*vector);
				D_ASSERT(le->offset + le->length <= ListVector::GetListSize(*vector));
				for (idx_t k = 0; k < le->length; k++) {
					child_sel.set_index(child_count++, le->offset + k);
				}
				Vector::Verify(child, child_sel, child_count);
			}
		} else if (vtype == VectorType::FLAT_VECTOR) {
			auto &validity = FlatVector::Validity(*vector);
			auto &child = ListVector::GetEntry(*vector);
			auto child_size = ListVector::GetListSize(*vector);
			auto list_data = FlatVector::GetData<list_entry_t>(*vector);
			idx_t total_size = 0;
			for (idx_t i = 0; i < count; i++) {
				auto idx = sel->get_index(i);
				auto &le = list_data[idx];
				if (validity.RowIsValid(idx)) {
					D_ASSERT(le.offset + le.length <= child_size);
					total_size += le.length;
				}
			}
			SelectionVector child_sel(total_size);
			idx_t child_count = 0;
			for (idx_t i = 0; i < count; i++) {
				auto idx = sel->get_index(i);
				auto &le = list_data[idx];
				if (validity.RowIsValid(idx)) {
					D_ASSERT(le.offset + le.length <= child_size);
					for (idx_t k = 0; k < le.length; k++) {
						child_sel.set_index(child_count++, le.offset + k);
					}
				}
			}
			Vector::Verify(child, child_sel, child_count);
		}
	}
#endif
}

void Vector::Verify(idx_t count) {
	auto flat_sel = FlatVector::IncrementalSelectionVector();
	Verify(*this, *flat_sel, count);
}

void FlatVector::SetNull(Vector &vector, idx_t idx, bool is_null) {
	D_ASSERT(vector.GetVectorType() == VectorType::FLAT_VECTOR);
	vector.validity.Set(idx, !is_null);
	if (is_null && vector.GetType().InternalType() == PhysicalType::STRUCT) {
		// set all child entries to null as well
		auto &entries = StructVector::GetEntries(vector);
		for (auto &entry : entries) {
			FlatVector::SetNull(*entry, idx, is_null);
		}
	}
}

void ConstantVector::SetNull(Vector &vector, bool is_null) {
	D_ASSERT(vector.GetVectorType() == VectorType::CONSTANT_VECTOR);
	vector.validity.Set(0, !is_null);
	if (is_null && vector.GetType().InternalType() == PhysicalType::STRUCT) {
		// set all child entries to null as well
		auto &entries = StructVector::GetEntries(vector);
		for (auto &entry : entries) {
			entry->SetVectorType(VectorType::CONSTANT_VECTOR);
			ConstantVector::SetNull(*entry, is_null);
		}
	}
}

const SelectionVector *ConstantVector::ZeroSelectionVector(idx_t count, SelectionVector &owned_sel) {
	if (count <= STANDARD_VECTOR_SIZE) {
		return ConstantVector::ZeroSelectionVector();
	}
	owned_sel.Initialize(count);
	for (idx_t i = 0; i < count; i++) {
		owned_sel.set_index(i, 0);
	}
	return &owned_sel;
}

void ConstantVector::Reference(Vector &vector, Vector &source, idx_t position, idx_t count) {
	auto &source_type = source.GetType();
	switch (source_type.InternalType()) {
	case PhysicalType::LIST: {
		// retrieve the list entry from the source vector
		UnifiedVectorFormat vdata;
		source.ToUnifiedFormat(count, vdata);

		auto list_index = vdata.sel->get_index(position);
		if (!vdata.validity.RowIsValid(list_index)) {
			// list is null: create null value
			Value null_value(source_type);
			vector.Reference(null_value);
			break;
		}

		auto list_data = (list_entry_t *)vdata.data;
		auto list_entry = list_data[list_index];

		// add the list entry as the first element of "vector"
		// FIXME: we only need to allocate space for 1 tuple here
		auto target_data = FlatVector::GetData<list_entry_t>(vector);
		target_data[0] = list_entry;

		// create a reference to the child list of the source vector
		auto &child = ListVector::GetEntry(vector);
		child.Reference(ListVector::GetEntry(source));

		ListVector::SetListSize(vector, ListVector::GetListSize(source));
		vector.SetVectorType(VectorType::CONSTANT_VECTOR);
		break;
	}
	case PhysicalType::STRUCT: {
		UnifiedVectorFormat vdata;
		source.ToUnifiedFormat(count, vdata);

		auto struct_index = vdata.sel->get_index(position);
		if (!vdata.validity.RowIsValid(struct_index)) {
			// null struct: create null value
			Value null_value(source_type);
			vector.Reference(null_value);
			break;
		}

		// struct: pass constant reference into child entries
		auto &source_entries = StructVector::GetEntries(source);
		auto &target_entries = StructVector::GetEntries(vector);
		for (idx_t i = 0; i < source_entries.size(); i++) {
			ConstantVector::Reference(*target_entries[i], *source_entries[i], position, count);
		}
		vector.SetVectorType(VectorType::CONSTANT_VECTOR);
		break;
	}
	default:
		// default behavior: get a value from the vector and reference it
		// this is not that expensive for scalar types
		auto value = source.GetValue(position);
		vector.Reference(value);
		D_ASSERT(vector.GetVectorType() == VectorType::CONSTANT_VECTOR);
		break;
	}
}

string_t StringVector::AddString(Vector &vector, const char *data, idx_t len) {
	return StringVector::AddString(vector, string_t(data, len));
}

string_t StringVector::AddStringOrBlob(Vector &vector, const char *data, idx_t len) {
	return StringVector::AddStringOrBlob(vector, string_t(data, len));
}

string_t StringVector::AddString(Vector &vector, const char *data) {
	return StringVector::AddString(vector, string_t(data, strlen(data)));
}

string_t StringVector::AddString(Vector &vector, const string &data) {
	return StringVector::AddString(vector, string_t(data.c_str(), data.size()));
}

string_t StringVector::AddString(Vector &vector, string_t data) {
	D_ASSERT(vector.GetType().id() == LogicalTypeId::VARCHAR || vector.GetType().id() == LogicalTypeId::JSON);
	if (data.IsInlined()) {
		// string will be inlined: no need to store in string heap
		return data;
	}
	if (!vector.auxiliary) {
		vector.auxiliary = make_buffer<VectorStringBuffer>();
	}
	D_ASSERT(vector.auxiliary->GetBufferType() == VectorBufferType::STRING_BUFFER);
	auto &string_buffer = (VectorStringBuffer &)*vector.auxiliary;
	return string_buffer.AddString(data);
}

string_t StringVector::AddStringOrBlob(Vector &vector, string_t data) {
	D_ASSERT(vector.GetType().InternalType() == PhysicalType::VARCHAR);
	if (data.IsInlined()) {
		// string will be inlined: no need to store in string heap
		return data;
	}
	if (!vector.auxiliary) {
		vector.auxiliary = make_buffer<VectorStringBuffer>();
	}
	D_ASSERT(vector.auxiliary->GetBufferType() == VectorBufferType::STRING_BUFFER);
	auto &string_buffer = (VectorStringBuffer &)*vector.auxiliary;
	return string_buffer.AddBlob(data);
}

string_t StringVector::EmptyString(Vector &vector, idx_t len) {
	D_ASSERT(vector.GetType().InternalType() == PhysicalType::VARCHAR);
	if (len < string_t::INLINE_LENGTH) {
		return string_t(len);
	}
	if (!vector.auxiliary) {
		vector.auxiliary = make_buffer<VectorStringBuffer>();
	}
	D_ASSERT(vector.auxiliary->GetBufferType() == VectorBufferType::STRING_BUFFER);
	auto &string_buffer = (VectorStringBuffer &)*vector.auxiliary;
	return string_buffer.EmptyString(len);
}

void StringVector::AddHandle(Vector &vector, BufferHandle handle) {
	D_ASSERT(vector.GetType().InternalType() == PhysicalType::VARCHAR);
	if (!vector.auxiliary) {
		vector.auxiliary = make_buffer<VectorStringBuffer>();
	}
	auto &string_buffer = (VectorStringBuffer &)*vector.auxiliary;
	string_buffer.AddHeapReference(make_buffer<ManagedVectorBuffer>(move(handle)));
}

void StringVector::AddBuffer(Vector &vector, buffer_ptr<VectorBuffer> buffer) {
	D_ASSERT(vector.GetType().InternalType() == PhysicalType::VARCHAR);
	D_ASSERT(buffer.get() != vector.auxiliary.get());
	if (!vector.auxiliary) {
		vector.auxiliary = make_buffer<VectorStringBuffer>();
	}
	auto &string_buffer = (VectorStringBuffer &)*vector.auxiliary;
	string_buffer.AddHeapReference(move(buffer));
}

void StringVector::AddHeapReference(Vector &vector, Vector &other) {
	D_ASSERT(vector.GetType().InternalType() == PhysicalType::VARCHAR);
	D_ASSERT(other.GetType().InternalType() == PhysicalType::VARCHAR);

	if (other.GetVectorType() == VectorType::DICTIONARY_VECTOR) {
		StringVector::AddHeapReference(vector, DictionaryVector::Child(other));
		return;
	}
	if (!other.auxiliary) {
		return;
	}
	StringVector::AddBuffer(vector, other.auxiliary);
}

string_t FSSTVector::AddCompressedString(Vector &vector, const char *data, idx_t len) {
	return FSSTVector::AddCompressedString(vector, string_t(data, len));
}

string_t FSSTVector::AddCompressedString(Vector &vector, string_t data) {
	D_ASSERT(vector.GetType().InternalType() == PhysicalType::VARCHAR);
	if (data.IsInlined()) {
		// string will be inlined: no need to store in string heap
		return data;
	}
	if (!vector.auxiliary) {
		vector.auxiliary = make_buffer<VectorFSSTStringBuffer>();
	}
	D_ASSERT(vector.auxiliary->GetBufferType() == VectorBufferType::FSST_BUFFER);
	auto &fsst_string_buffer = (VectorFSSTStringBuffer &)*vector.auxiliary;
	return fsst_string_buffer.AddBlob(data);
}

void *FSSTVector::GetDecoder(const Vector &vector) {
	D_ASSERT(vector.GetType().InternalType() == PhysicalType::VARCHAR);
	if (!vector.auxiliary) {
		throw InternalException("GetDecoder called on FSST Vector without registered buffer");
	}
	D_ASSERT(vector.auxiliary->GetBufferType() == VectorBufferType::FSST_BUFFER);
	auto &fsst_string_buffer = (VectorFSSTStringBuffer &)*vector.auxiliary;
	return (duckdb_fsst_decoder_t *)fsst_string_buffer.GetDecoder();
}

void FSSTVector::RegisterDecoder(Vector &vector, buffer_ptr<void> &duckdb_fsst_decoder) {
	D_ASSERT(vector.GetType().InternalType() == PhysicalType::VARCHAR);

	if (!vector.auxiliary) {
		vector.auxiliary = make_buffer<VectorFSSTStringBuffer>();
	}
	D_ASSERT(vector.auxiliary->GetBufferType() == VectorBufferType::FSST_BUFFER);

	auto &fsst_string_buffer = (VectorFSSTStringBuffer &)*vector.auxiliary;
	fsst_string_buffer.AddDecoder(duckdb_fsst_decoder);
}

void FSSTVector::SetCount(Vector &vector, idx_t count) {
	D_ASSERT(vector.GetType().InternalType() == PhysicalType::VARCHAR);

	if (!vector.auxiliary) {
		vector.auxiliary = make_buffer<VectorFSSTStringBuffer>();
	}
	D_ASSERT(vector.auxiliary->GetBufferType() == VectorBufferType::FSST_BUFFER);

	auto &fsst_string_buffer = (VectorFSSTStringBuffer &)*vector.auxiliary;
	fsst_string_buffer.SetCount(count);
}

idx_t FSSTVector::GetCount(Vector &vector) {
	D_ASSERT(vector.GetType().InternalType() == PhysicalType::VARCHAR);

	if (!vector.auxiliary) {
		vector.auxiliary = make_buffer<VectorFSSTStringBuffer>();
	}
	D_ASSERT(vector.auxiliary->GetBufferType() == VectorBufferType::FSST_BUFFER);

	auto &fsst_string_buffer = (VectorFSSTStringBuffer &)*vector.auxiliary;
	return fsst_string_buffer.GetCount();
}

void FSSTVector::DecompressVector(const Vector &src, Vector &dst, idx_t src_offset, idx_t dst_offset, idx_t copy_count,
                                  const SelectionVector *sel) {
	D_ASSERT(src.GetVectorType() == VectorType::FSST_VECTOR);
	D_ASSERT(dst.GetVectorType() == VectorType::FLAT_VECTOR);
	auto dst_mask = FlatVector::Validity(dst);
	auto ldata = FSSTVector::GetCompressedData<string_t>(src);
	auto tdata = FlatVector::GetData<string_t>(dst);
	for (idx_t i = 0; i < copy_count; i++) {
		auto source_idx = sel->get_index(src_offset + i);
		auto target_idx = dst_offset + i;
		string_t compressed_string = ldata[source_idx];
		if (dst_mask.RowIsValid(target_idx) && compressed_string.GetSize() > 0) {
			tdata[target_idx] = FSSTPrimitives::DecompressValue(FSSTVector::GetDecoder(src), dst,
			                                                    (unsigned char *)compressed_string.GetDataUnsafe(),
			                                                    compressed_string.GetSize());
		} else {
			tdata[target_idx] = string_t(nullptr, 0);
		}
	}
}

Vector &MapVector::GetKeys(Vector &vector) {
	auto &entries = StructVector::GetEntries(vector);
	D_ASSERT(entries.size() == 2);
	return *entries[0];
}
Vector &MapVector::GetValues(Vector &vector) {
	auto &entries = StructVector::GetEntries(vector);
	D_ASSERT(entries.size() == 2);
	return *entries[1];
}

const Vector &MapVector::GetKeys(const Vector &vector) {
	return GetKeys((Vector &)vector);
}
const Vector &MapVector::GetValues(const Vector &vector) {
	return GetValues((Vector &)vector);
}

vector<unique_ptr<Vector>> &StructVector::GetEntries(Vector &vector) {
	D_ASSERT(vector.GetType().id() == LogicalTypeId::STRUCT || vector.GetType().id() == LogicalTypeId::MAP ||
	         vector.GetType().id() == LogicalTypeId::UNION);

	if (vector.GetVectorType() == VectorType::DICTIONARY_VECTOR) {
		auto &child = DictionaryVector::Child(vector);
		return StructVector::GetEntries(child);
	}
	D_ASSERT(vector.GetVectorType() == VectorType::FLAT_VECTOR ||
	         vector.GetVectorType() == VectorType::CONSTANT_VECTOR);
	D_ASSERT(vector.auxiliary);
	D_ASSERT(vector.auxiliary->GetBufferType() == VectorBufferType::STRUCT_BUFFER);
	return ((VectorStructBuffer *)vector.auxiliary.get())->GetChildren();
}

const vector<unique_ptr<Vector>> &StructVector::GetEntries(const Vector &vector) {
	return GetEntries((Vector &)vector);
}

const Vector &ListVector::GetEntry(const Vector &vector) {
	D_ASSERT(vector.GetType().id() == LogicalTypeId::LIST);
	if (vector.GetVectorType() == VectorType::DICTIONARY_VECTOR) {
		auto &child = DictionaryVector::Child(vector);
		return ListVector::GetEntry(child);
	}
	D_ASSERT(vector.GetVectorType() == VectorType::FLAT_VECTOR ||
	         vector.GetVectorType() == VectorType::CONSTANT_VECTOR);
	D_ASSERT(vector.auxiliary);
	D_ASSERT(vector.auxiliary->GetBufferType() == VectorBufferType::LIST_BUFFER);
	return ((VectorListBuffer *)vector.auxiliary.get())->GetChild();
}

Vector &ListVector::GetEntry(Vector &vector) {
	const Vector &cvector = vector;
	return const_cast<Vector &>(ListVector::GetEntry(cvector));
}

void ListVector::Reserve(Vector &vector, idx_t required_capacity) {
	D_ASSERT(vector.GetType().id() == LogicalTypeId::LIST);
	D_ASSERT(vector.GetVectorType() == VectorType::FLAT_VECTOR ||
	         vector.GetVectorType() == VectorType::CONSTANT_VECTOR);
	D_ASSERT(vector.auxiliary);
	D_ASSERT(vector.auxiliary->GetBufferType() == VectorBufferType::LIST_BUFFER);
	auto &child_buffer = *((VectorListBuffer *)vector.auxiliary.get());
	child_buffer.Reserve(required_capacity);
}

template <class T>
void TemplatedSearchInMap(Vector &list, T key, vector<idx_t> &offsets, bool is_key_null, idx_t offset, idx_t length) {
	auto &list_vector = ListVector::GetEntry(list);
	UnifiedVectorFormat vector_data;
	list_vector.ToUnifiedFormat(ListVector::GetListSize(list), vector_data);
	auto data = (T *)vector_data.data;
	auto validity_mask = vector_data.validity;

	if (is_key_null) {
		for (idx_t i = offset; i < offset + length; i++) {
			if (!validity_mask.RowIsValid(i)) {
				offsets.push_back(i);
			}
		}
	} else {
		for (idx_t i = offset; i < offset + length; i++) {
			if (!validity_mask.RowIsValid(i)) {
				continue;
			}
			if (key == data[i]) {
				offsets.push_back(i);
			}
		}
	}
}

template <class T>
void TemplatedSearchInMap(Vector &list, const Value &key, vector<idx_t> &offsets, bool is_key_null, idx_t offset,
                          idx_t length) {
	TemplatedSearchInMap<T>(list, key.template GetValueUnsafe<T>(), offsets, is_key_null, offset, length);
}

void SearchStringInMap(Vector &list, const string &key, vector<idx_t> &offsets, bool is_key_null, idx_t offset,
                       idx_t length) {
	auto &list_vector = ListVector::GetEntry(list);
	UnifiedVectorFormat vector_data;
	list_vector.ToUnifiedFormat(ListVector::GetListSize(list), vector_data);
	auto data = (string_t *)vector_data.data;
	auto validity_mask = vector_data.validity;
	if (is_key_null) {
		for (idx_t i = offset; i < offset + length; i++) {
			if (!validity_mask.RowIsValid(i)) {
				offsets.push_back(i);
			}
		}
	} else {
		string_t key_str_t(key);
		for (idx_t i = offset; i < offset + length; i++) {
			if (!validity_mask.RowIsValid(i)) {
				continue;
			}
			if (Equals::Operation<string_t>(data[i], key_str_t)) {
				offsets.push_back(i);
			}
		}
	}
}

vector<idx_t> ListVector::Search(Vector &list, const Value &key, idx_t row) {
	vector<idx_t> offsets;

	auto &list_vector = ListVector::GetEntry(list);
	auto &entry = ListVector::GetData(list)[row];

	switch (list_vector.GetType().InternalType()) {
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
		TemplatedSearchInMap<int8_t>(list, key, offsets, key.IsNull(), entry.offset, entry.length);
		break;
	case PhysicalType::INT16:
		TemplatedSearchInMap<int16_t>(list, key, offsets, key.IsNull(), entry.offset, entry.length);
		break;
	case PhysicalType::INT32:
		TemplatedSearchInMap<int32_t>(list, key, offsets, key.IsNull(), entry.offset, entry.length);
		break;
	case PhysicalType::INT64:
		TemplatedSearchInMap<int64_t>(list, key, offsets, key.IsNull(), entry.offset, entry.length);
		break;
	case PhysicalType::INT128:
		TemplatedSearchInMap<hugeint_t>(list, key, offsets, key.IsNull(), entry.offset, entry.length);
		break;
	case PhysicalType::UINT8:
		TemplatedSearchInMap<uint8_t>(list, key, offsets, key.IsNull(), entry.offset, entry.length);
		break;
	case PhysicalType::UINT16:
		TemplatedSearchInMap<uint16_t>(list, key, offsets, key.IsNull(), entry.offset, entry.length);
		break;
	case PhysicalType::UINT32:
		TemplatedSearchInMap<uint32_t>(list, key, offsets, key.IsNull(), entry.offset, entry.length);
		break;
	case PhysicalType::UINT64:
		TemplatedSearchInMap<uint64_t>(list, key, offsets, key.IsNull(), entry.offset, entry.length);
		break;
	case PhysicalType::FLOAT:
		TemplatedSearchInMap<float>(list, key, offsets, key.IsNull(), entry.offset, entry.length);
		break;
	case PhysicalType::DOUBLE:
		TemplatedSearchInMap<double>(list, key, offsets, key.IsNull(), entry.offset, entry.length);
		break;
	case PhysicalType::VARCHAR:
		SearchStringInMap(list, StringValue::Get(key), offsets, key.IsNull(), entry.offset, entry.length);
		break;
	default:
		throw InvalidTypeException(list.GetType().id(), "Invalid type for List Vector Search");
	}
	return offsets;
}

Value ListVector::GetValuesFromOffsets(Vector &list, vector<idx_t> &offsets) {
	auto &child_vec = ListVector::GetEntry(list);
	vector<Value> list_values;
	list_values.reserve(offsets.size());
	for (auto &offset : offsets) {
		list_values.push_back(child_vec.GetValue(offset));
	}
	return Value::LIST(ListType::GetChildType(list.GetType()), move(list_values));
}

idx_t ListVector::GetListSize(const Vector &vec) {
	if (vec.GetVectorType() == VectorType::DICTIONARY_VECTOR) {
		auto &child = DictionaryVector::Child(vec);
		return ListVector::GetListSize(child);
	}
	D_ASSERT(vec.auxiliary);
	return ((VectorListBuffer &)*vec.auxiliary).size;
}

idx_t ListVector::GetListCapacity(const Vector &vec) {
	if (vec.GetVectorType() == VectorType::DICTIONARY_VECTOR) {
		auto &child = DictionaryVector::Child(vec);
		return ListVector::GetListSize(child);
	}
	D_ASSERT(vec.auxiliary);
	return ((VectorListBuffer &)*vec.auxiliary).capacity;
}

void ListVector::ReferenceEntry(Vector &vector, Vector &other) {
	D_ASSERT(vector.GetType().id() == LogicalTypeId::LIST);
	D_ASSERT(vector.GetVectorType() == VectorType::FLAT_VECTOR ||
	         vector.GetVectorType() == VectorType::CONSTANT_VECTOR);
	D_ASSERT(other.GetType().id() == LogicalTypeId::LIST);
	D_ASSERT(other.GetVectorType() == VectorType::FLAT_VECTOR || other.GetVectorType() == VectorType::CONSTANT_VECTOR);
	vector.auxiliary = other.auxiliary;
}

void ListVector::SetListSize(Vector &vec, idx_t size) {
	if (vec.GetVectorType() == VectorType::DICTIONARY_VECTOR) {
		auto &child = DictionaryVector::Child(vec);
		ListVector::SetListSize(child, size);
	}
	((VectorListBuffer &)*vec.auxiliary).size = size;
}

void ListVector::Append(Vector &target, const Vector &source, idx_t source_size, idx_t source_offset) {
	if (source_size - source_offset == 0) {
		//! Nothing to add
		return;
	}
	auto &target_buffer = (VectorListBuffer &)*target.auxiliary;
	target_buffer.Append(source, source_size, source_offset);
}

void ListVector::Append(Vector &target, const Vector &source, const SelectionVector &sel, idx_t source_size,
                        idx_t source_offset) {
	if (source_size - source_offset == 0) {
		//! Nothing to add
		return;
	}
	auto &target_buffer = (VectorListBuffer &)*target.auxiliary;
	target_buffer.Append(source, sel, source_size, source_offset);
}

void ListVector::PushBack(Vector &target, const Value &insert) {
	auto &target_buffer = (VectorListBuffer &)*target.auxiliary;
	target_buffer.PushBack(insert);
}

// Union vector
const Vector &UnionVector::GetMember(const Vector &vector, idx_t member_index) {
	D_ASSERT(member_index < UnionType::GetMemberCount(vector.GetType()));
	auto &entries = StructVector::GetEntries(vector);
	return *entries[member_index + 1]; // skip the "tag" entry
}

Vector &UnionVector::GetMember(Vector &vector, idx_t member_index) {
	D_ASSERT(member_index < UnionType::GetMemberCount(vector.GetType()));
	auto &entries = StructVector::GetEntries(vector);
	return *entries[member_index + 1]; // skip the "tag" entry
}

const Vector &UnionVector::GetTags(const Vector &vector) {
	// the tag vector is always the first struct child.
	return *StructVector::GetEntries(vector)[0];
}

Vector &UnionVector::GetTags(Vector &vector) {
	// the tag vector is always the first struct child.
	return *StructVector::GetEntries(vector)[0];
}

void UnionVector::SetToMember(Vector &union_vector, union_tag_t tag, Vector &member_vector, idx_t count,
                              bool keep_tags_for_null) {
	D_ASSERT(union_vector.GetType().id() == LogicalTypeId::UNION);
	D_ASSERT(tag < UnionType::GetMemberCount(union_vector.GetType()));

	// Set the union member to the specified vector
	UnionVector::GetMember(union_vector, tag).Reference(member_vector);
	auto &tag_vector = UnionVector::GetTags(union_vector);

	if (member_vector.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		// if the member vector is constant, we can set the union to constant as well
		union_vector.SetVectorType(VectorType::CONSTANT_VECTOR);
		ConstantVector::GetData<union_tag_t>(tag_vector)[0] = tag;
		ConstantVector::SetNull(union_vector, ConstantVector::IsNull(member_vector));

	} else {
		// otherwise flatten and set to flatvector
		member_vector.Flatten(count);
		union_vector.SetVectorType(VectorType::FLAT_VECTOR);

		if (member_vector.validity.AllValid()) {
			// if the member vector is all valid, we can set the tag to constant
			tag_vector.SetVectorType(VectorType::CONSTANT_VECTOR);
			auto tag_data = ConstantVector::GetData<union_tag_t>(tag_vector);
			*tag_data = tag;
		} else {
			tag_vector.SetVectorType(VectorType::FLAT_VECTOR);
			if (keep_tags_for_null) {
				FlatVector::Validity(tag_vector).SetAllValid(count);
				FlatVector::Validity(union_vector).SetAllValid(count);
			} else {
				// ensure the tags have the same validity as the member
				FlatVector::Validity(union_vector) = FlatVector::Validity(member_vector);
				FlatVector::Validity(tag_vector) = FlatVector::Validity(member_vector);
			}

			auto tag_data = FlatVector::GetData<union_tag_t>(tag_vector);
			memset(tag_data, tag, count);
		}
	}

	// Set the non-selected members to constant null vectors
	for (idx_t i = 0; i < UnionType::GetMemberCount(union_vector.GetType()); i++) {
		if (i != tag) {
			auto &member = UnionVector::GetMember(union_vector, i);
			member.SetVectorType(VectorType::CONSTANT_VECTOR);
			ConstantVector::SetNull(member, true);
		}
	}
}

union_tag_t UnionVector::GetTag(const Vector &vector, idx_t index) {
	// the tag vector is always the first struct child.
	auto &tag_vector = *StructVector::GetEntries(vector)[0];
	if (tag_vector.GetVectorType() == VectorType::DICTIONARY_VECTOR) {
		auto &child = DictionaryVector::Child(tag_vector);
		return FlatVector::GetData<union_tag_t>(child)[index];
	}
	if (tag_vector.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		return ConstantVector::GetData<union_tag_t>(tag_vector)[0];
	}
	return FlatVector::GetData<union_tag_t>(tag_vector)[index];
}

} // namespace duckdb








namespace duckdb {

buffer_ptr<VectorBuffer> VectorBuffer::CreateStandardVector(PhysicalType type, idx_t capacity) {
	return make_buffer<VectorBuffer>(capacity * GetTypeIdSize(type));
}

buffer_ptr<VectorBuffer> VectorBuffer::CreateConstantVector(PhysicalType type) {
	return make_buffer<VectorBuffer>(GetTypeIdSize(type));
}

buffer_ptr<VectorBuffer> VectorBuffer::CreateConstantVector(const LogicalType &type) {
	return VectorBuffer::CreateConstantVector(type.InternalType());
}

buffer_ptr<VectorBuffer> VectorBuffer::CreateStandardVector(const LogicalType &type, idx_t capacity) {
	return VectorBuffer::CreateStandardVector(type.InternalType(), capacity);
}

VectorStringBuffer::VectorStringBuffer() : VectorBuffer(VectorBufferType::STRING_BUFFER) {
}

VectorStringBuffer::VectorStringBuffer(VectorBufferType type) : VectorBuffer(type) {
}

VectorFSSTStringBuffer::VectorFSSTStringBuffer() : VectorStringBuffer(VectorBufferType::FSST_BUFFER) {
}

VectorStructBuffer::VectorStructBuffer() : VectorBuffer(VectorBufferType::STRUCT_BUFFER) {
}

VectorStructBuffer::VectorStructBuffer(const LogicalType &type, idx_t capacity)
    : VectorBuffer(VectorBufferType::STRUCT_BUFFER) {
	auto &child_types = StructType::GetChildTypes(type);
	for (auto &child_type : child_types) {
		auto vector = make_unique<Vector>(child_type.second, capacity);
		children.push_back(move(vector));
	}
}

VectorStructBuffer::VectorStructBuffer(Vector &other, const SelectionVector &sel, idx_t count)
    : VectorBuffer(VectorBufferType::STRUCT_BUFFER) {
	auto &other_vector = StructVector::GetEntries(other);
	for (auto &child_vector : other_vector) {
		auto vector = make_unique<Vector>(*child_vector, sel, count);
		children.push_back(move(vector));
	}
}

VectorStructBuffer::~VectorStructBuffer() {
}

VectorListBuffer::VectorListBuffer(unique_ptr<Vector> vector, idx_t initial_capacity)
    : VectorBuffer(VectorBufferType::LIST_BUFFER), capacity(initial_capacity), child(move(vector)) {
}

VectorListBuffer::VectorListBuffer(const LogicalType &list_type, idx_t initial_capacity)
    : VectorBuffer(VectorBufferType::LIST_BUFFER), capacity(initial_capacity),
      child(make_unique<Vector>(ListType::GetChildType(list_type), initial_capacity)) {
}

void VectorListBuffer::Reserve(idx_t to_reserve) {
	if (to_reserve > capacity) {
		idx_t new_capacity = NextPowerOfTwo(to_reserve);
		D_ASSERT(new_capacity >= to_reserve);
		child->Resize(capacity, new_capacity);
		capacity = new_capacity;
	}
}

void VectorListBuffer::Append(const Vector &to_append, idx_t to_append_size, idx_t source_offset) {
	Reserve(size + to_append_size - source_offset);
	VectorOperations::Copy(to_append, *child, to_append_size, source_offset, size);
	size += to_append_size - source_offset;
}

void VectorListBuffer::Append(const Vector &to_append, const SelectionVector &sel, idx_t to_append_size,
                              idx_t source_offset) {
	Reserve(size + to_append_size - source_offset);
	VectorOperations::Copy(to_append, *child, sel, to_append_size, source_offset, size);
	size += to_append_size - source_offset;
}

void VectorListBuffer::PushBack(const Value &insert) {
	if (size + 1 > capacity) {
		child->Resize(capacity, capacity * 2);
		capacity *= 2;
	}
	child->SetValue(size++, insert);
}

VectorListBuffer::~VectorListBuffer() {
}

ManagedVectorBuffer::ManagedVectorBuffer(BufferHandle handle)
    : VectorBuffer(VectorBufferType::MANAGED_BUFFER), handle(move(handle)) {
}

ManagedVectorBuffer::~ManagedVectorBuffer() {
}

} // namespace duckdb





namespace duckdb {

class VectorCacheBuffer : public VectorBuffer {
public:
	explicit VectorCacheBuffer(Allocator &allocator, const LogicalType &type_p, idx_t capacity_p = STANDARD_VECTOR_SIZE)
	    : VectorBuffer(VectorBufferType::OPAQUE_BUFFER), type(type_p), capacity(capacity_p) {
		auto internal_type = type.InternalType();
		switch (internal_type) {
		case PhysicalType::LIST: {
			// memory for the list offsets
			owned_data = allocator.Allocate(capacity * GetTypeIdSize(internal_type));
			// child data of the list
			auto &child_type = ListType::GetChildType(type);
			child_caches.push_back(make_buffer<VectorCacheBuffer>(allocator, child_type));
			auto child_vector = make_unique<Vector>(child_type, false, false);
			auxiliary = make_unique<VectorListBuffer>(move(child_vector));
			break;
		}
		case PhysicalType::STRUCT: {
			auto &child_types = StructType::GetChildTypes(type);
			for (auto &child_type : child_types) {
				child_caches.push_back(make_buffer<VectorCacheBuffer>(allocator, child_type.second));
			}
			auto struct_buffer = make_unique<VectorStructBuffer>(type);
			auxiliary = move(struct_buffer);
			break;
		}
		default:
			owned_data = allocator.Allocate(capacity * GetTypeIdSize(internal_type));
			break;
		}
	}

	void ResetFromCache(Vector &result, const buffer_ptr<VectorBuffer> &buffer) {
		D_ASSERT(type == result.GetType());
		auto internal_type = type.InternalType();
		result.vector_type = VectorType::FLAT_VECTOR;
		AssignSharedPointer(result.buffer, buffer);
		result.validity.Reset();
		switch (internal_type) {
		case PhysicalType::LIST: {
			result.data = owned_data.get();
			// reinitialize the VectorListBuffer
			AssignSharedPointer(result.auxiliary, auxiliary);
			// propagate through child
			auto &list_buffer = (VectorListBuffer &)*result.auxiliary;
			list_buffer.capacity = capacity;
			list_buffer.size = 0;
			list_buffer.SetAuxiliaryData(nullptr);

			auto &list_child = list_buffer.GetChild();
			auto &child_cache = (VectorCacheBuffer &)*child_caches[0];
			child_cache.ResetFromCache(list_child, child_caches[0]);
			break;
		}
		case PhysicalType::STRUCT: {
			// struct does not have data
			result.data = nullptr;
			// reinitialize the VectorStructBuffer
			auxiliary->SetAuxiliaryData(nullptr);
			AssignSharedPointer(result.auxiliary, auxiliary);
			// propagate through children
			auto &children = ((VectorStructBuffer &)*result.auxiliary).GetChildren();
			for (idx_t i = 0; i < children.size(); i++) {
				auto &child_cache = (VectorCacheBuffer &)*child_caches[i];
				child_cache.ResetFromCache(*children[i], child_caches[i]);
			}
			break;
		}
		default:
			// regular type: no aux data and reset data to cached data
			result.data = owned_data.get();
			result.auxiliary.reset();
			break;
		}
	}

	const LogicalType &GetType() {
		return type;
	}

private:
	//! The type of the vector cache
	LogicalType type;
	//! Owned data
	AllocatedData owned_data;
	//! Child caches (if any). Used for nested types.
	vector<buffer_ptr<VectorBuffer>> child_caches;
	//! Aux data for the vector (if any)
	buffer_ptr<VectorBuffer> auxiliary;
	//! Capacity of the vector
	idx_t capacity;
};

VectorCache::VectorCache(Allocator &allocator, const LogicalType &type_p, idx_t capacity_p) {
	buffer = make_unique<VectorCacheBuffer>(allocator, type_p, capacity_p);
}

void VectorCache::ResetFromCache(Vector &result) const {
	D_ASSERT(buffer);
	auto &vcache = (VectorCacheBuffer &)*buffer;
	vcache.ResetFromCache(result, buffer);
}

const LogicalType &VectorCache::GetType() const {
	auto &vcache = (VectorCacheBuffer &)*buffer;
	return vcache.GetType();
}

} // namespace duckdb


namespace duckdb {

const SelectionVector *ConstantVector::ZeroSelectionVector() {
	static const SelectionVector ZERO_SELECTION_VECTOR = SelectionVector((sel_t *)ConstantVector::ZERO_VECTOR);
	return &ZERO_SELECTION_VECTOR;
}

const SelectionVector *FlatVector::IncrementalSelectionVector() {
	static const SelectionVector INCREMENTAL_SELECTION_VECTOR;
	return &INCREMENTAL_SELECTION_VECTOR;
}

const sel_t ConstantVector::ZERO_VECTOR[STANDARD_VECTOR_SIZE] = {0};

} // namespace duckdb





















#include <cmath>

namespace duckdb {

LogicalType::LogicalType() : LogicalType(LogicalTypeId::INVALID) {
}

LogicalType::LogicalType(LogicalTypeId id) : id_(id) {
	physical_type_ = GetInternalType();
}
LogicalType::LogicalType(LogicalTypeId id, shared_ptr<ExtraTypeInfo> type_info_p)
    : id_(id), type_info_(move(type_info_p)) {
	physical_type_ = GetInternalType();
}

LogicalType::LogicalType(const LogicalType &other)
    : id_(other.id_), physical_type_(other.physical_type_), type_info_(other.type_info_) {
}

LogicalType::LogicalType(LogicalType &&other) noexcept
    : id_(other.id_), physical_type_(other.physical_type_), type_info_(move(other.type_info_)) {
}

hash_t LogicalType::Hash() const {
	return duckdb::Hash<uint8_t>((uint8_t)id_);
}

PhysicalType LogicalType::GetInternalType() {
	switch (id_) {
	case LogicalTypeId::BOOLEAN:
		return PhysicalType::BOOL;
	case LogicalTypeId::TINYINT:
		return PhysicalType::INT8;
	case LogicalTypeId::UTINYINT:
		return PhysicalType::UINT8;
	case LogicalTypeId::SMALLINT:
		return PhysicalType::INT16;
	case LogicalTypeId::USMALLINT:
		return PhysicalType::UINT16;
	case LogicalTypeId::SQLNULL:
	case LogicalTypeId::DATE:
	case LogicalTypeId::INTEGER:
		return PhysicalType::INT32;
	case LogicalTypeId::UINTEGER:
		return PhysicalType::UINT32;
	case LogicalTypeId::BIGINT:
	case LogicalTypeId::TIME:
	case LogicalTypeId::TIMESTAMP:
	case LogicalTypeId::TIMESTAMP_SEC:
	case LogicalTypeId::TIMESTAMP_NS:
	case LogicalTypeId::TIMESTAMP_MS:
	case LogicalTypeId::TIME_TZ:
	case LogicalTypeId::TIMESTAMP_TZ:
		return PhysicalType::INT64;
	case LogicalTypeId::UBIGINT:
		return PhysicalType::UINT64;
	case LogicalTypeId::HUGEINT:
	case LogicalTypeId::UUID:
		return PhysicalType::INT128;
	case LogicalTypeId::FLOAT:
		return PhysicalType::FLOAT;
	case LogicalTypeId::DOUBLE:
		return PhysicalType::DOUBLE;
	case LogicalTypeId::DECIMAL: {
		if (!type_info_) {
			return PhysicalType::INVALID;
		}
		auto width = DecimalType::GetWidth(*this);
		if (width <= Decimal::MAX_WIDTH_INT16) {
			return PhysicalType::INT16;
		} else if (width <= Decimal::MAX_WIDTH_INT32) {
			return PhysicalType::INT32;
		} else if (width <= Decimal::MAX_WIDTH_INT64) {
			return PhysicalType::INT64;
		} else if (width <= Decimal::MAX_WIDTH_INT128) {
			return PhysicalType::INT128;
		} else {
			throw InternalException("Decimal has a width of %d which is bigger than the maximum supported width of %d",
			                        width, DecimalType::MaxWidth());
		}
	}
	case LogicalTypeId::VARCHAR:
	case LogicalTypeId::CHAR:
	case LogicalTypeId::BLOB:
	case LogicalTypeId::JSON:
		return PhysicalType::VARCHAR;
	case LogicalTypeId::INTERVAL:
		return PhysicalType::INTERVAL;
	case LogicalTypeId::MAP:
	case LogicalTypeId::UNION:
	case LogicalTypeId::STRUCT:
		return PhysicalType::STRUCT;
	case LogicalTypeId::LIST:
		return PhysicalType::LIST;
	case LogicalTypeId::POINTER:
		// LCOV_EXCL_START
		if (sizeof(uintptr_t) == sizeof(uint32_t)) {
			return PhysicalType::UINT32;
		} else if (sizeof(uintptr_t) == sizeof(uint64_t)) {
			return PhysicalType::UINT64;
		} else {
			throw InternalException("Unsupported pointer size");
		}
		// LCOV_EXCL_STOP
	case LogicalTypeId::VALIDITY:
		return PhysicalType::BIT;
	case LogicalTypeId::ENUM: {
		D_ASSERT(type_info_);
		return EnumType::GetPhysicalType(*this);
	}
	case LogicalTypeId::TABLE:
	case LogicalTypeId::LAMBDA:
	case LogicalTypeId::ANY:
	case LogicalTypeId::INVALID:
	case LogicalTypeId::UNKNOWN:
		return PhysicalType::INVALID;
	case LogicalTypeId::USER:
		return PhysicalType::UNKNOWN;
	case LogicalTypeId::AGGREGATE_STATE:
		return PhysicalType::VARCHAR;
	default:
		throw InternalException("Invalid LogicalType %s", ToString());
	}
}

constexpr const LogicalTypeId LogicalType::INVALID;
constexpr const LogicalTypeId LogicalType::SQLNULL;
constexpr const LogicalTypeId LogicalType::BOOLEAN;
constexpr const LogicalTypeId LogicalType::TINYINT;
constexpr const LogicalTypeId LogicalType::UTINYINT;
constexpr const LogicalTypeId LogicalType::SMALLINT;
constexpr const LogicalTypeId LogicalType::USMALLINT;
constexpr const LogicalTypeId LogicalType::INTEGER;
constexpr const LogicalTypeId LogicalType::UINTEGER;
constexpr const LogicalTypeId LogicalType::BIGINT;
constexpr const LogicalTypeId LogicalType::UBIGINT;
constexpr const LogicalTypeId LogicalType::HUGEINT;
constexpr const LogicalTypeId LogicalType::UUID;
constexpr const LogicalTypeId LogicalType::FLOAT;
constexpr const LogicalTypeId LogicalType::DOUBLE;
constexpr const LogicalTypeId LogicalType::DATE;

constexpr const LogicalTypeId LogicalType::TIMESTAMP;
constexpr const LogicalTypeId LogicalType::TIMESTAMP_MS;
constexpr const LogicalTypeId LogicalType::TIMESTAMP_NS;
constexpr const LogicalTypeId LogicalType::TIMESTAMP_S;

constexpr const LogicalTypeId LogicalType::TIME;

constexpr const LogicalTypeId LogicalType::TIME_TZ;
constexpr const LogicalTypeId LogicalType::TIMESTAMP_TZ;

constexpr const LogicalTypeId LogicalType::HASH;
constexpr const LogicalTypeId LogicalType::POINTER;

constexpr const LogicalTypeId LogicalType::VARCHAR;
constexpr const LogicalTypeId LogicalType::JSON;

constexpr const LogicalTypeId LogicalType::BLOB;
constexpr const LogicalTypeId LogicalType::INTERVAL;
constexpr const LogicalTypeId LogicalType::ROW_TYPE;

// TODO these are incomplete and should maybe not exist as such
constexpr const LogicalTypeId LogicalType::TABLE;
constexpr const LogicalTypeId LogicalType::LAMBDA;

constexpr const LogicalTypeId LogicalType::ANY;

const vector<LogicalType> LogicalType::Numeric() {
	vector<LogicalType> types = {LogicalType::TINYINT,   LogicalType::SMALLINT,  LogicalType::INTEGER,
	                             LogicalType::BIGINT,    LogicalType::HUGEINT,   LogicalType::FLOAT,
	                             LogicalType::DOUBLE,    LogicalTypeId::DECIMAL, LogicalType::UTINYINT,
	                             LogicalType::USMALLINT, LogicalType::UINTEGER,  LogicalType::UBIGINT};
	return types;
}

const vector<LogicalType> LogicalType::Integral() {
	vector<LogicalType> types = {LogicalType::TINYINT,   LogicalType::SMALLINT, LogicalType::INTEGER,
	                             LogicalType::BIGINT,    LogicalType::HUGEINT,  LogicalType::UTINYINT,
	                             LogicalType::USMALLINT, LogicalType::UINTEGER, LogicalType::UBIGINT};
	return types;
}

const vector<LogicalType> LogicalType::AllTypes() {
	vector<LogicalType> types = {
	    LogicalType::BOOLEAN,  LogicalType::TINYINT,   LogicalType::SMALLINT,     LogicalType::INTEGER,
	    LogicalType::BIGINT,   LogicalType::DATE,      LogicalType::TIMESTAMP,    LogicalType::DOUBLE,
	    LogicalType::FLOAT,    LogicalType::VARCHAR,   LogicalType::BLOB,         LogicalType::INTERVAL,
	    LogicalType::HUGEINT,  LogicalTypeId::DECIMAL, LogicalType::UTINYINT,     LogicalType::USMALLINT,
	    LogicalType::UINTEGER, LogicalType::UBIGINT,   LogicalType::TIME,         LogicalTypeId::LIST,
	    LogicalTypeId::STRUCT, LogicalType::TIME_TZ,   LogicalType::TIMESTAMP_TZ, LogicalTypeId::MAP,
	    LogicalTypeId::UNION,  LogicalType::UUID,      LogicalType::JSON};
	return types;
}

const PhysicalType ROW_TYPE = PhysicalType::INT64;

// LCOV_EXCL_START
string TypeIdToString(PhysicalType type) {
	switch (type) {
	case PhysicalType::BOOL:
		return "BOOL";
	case PhysicalType::INT8:
		return "INT8";
	case PhysicalType::INT16:
		return "INT16";
	case PhysicalType::INT32:
		return "INT32";
	case PhysicalType::INT64:
		return "INT64";
	case PhysicalType::UINT8:
		return "UINT8";
	case PhysicalType::UINT16:
		return "UINT16";
	case PhysicalType::UINT32:
		return "UINT32";
	case PhysicalType::UINT64:
		return "UINT64";
	case PhysicalType::INT128:
		return "INT128";
	case PhysicalType::FLOAT:
		return "FLOAT";
	case PhysicalType::DOUBLE:
		return "DOUBLE";
	case PhysicalType::VARCHAR:
		return "VARCHAR";
	case PhysicalType::INTERVAL:
		return "INTERVAL";
	case PhysicalType::STRUCT:
		return "STRUCT";
	case PhysicalType::LIST:
		return "LIST";
	case PhysicalType::INVALID:
		return "INVALID";
	case PhysicalType::BIT:
		return "BIT";
	case PhysicalType::MAP:
		return "MAP";
	case PhysicalType::UNKNOWN:
		return "UNKNOWN";
	}
	return "INVALID";
}
// LCOV_EXCL_STOP

idx_t GetTypeIdSize(PhysicalType type) {
	switch (type) {
	case PhysicalType::BIT:
	case PhysicalType::BOOL:
		return sizeof(bool);
	case PhysicalType::INT8:
		return sizeof(int8_t);
	case PhysicalType::INT16:
		return sizeof(int16_t);
	case PhysicalType::INT32:
		return sizeof(int32_t);
	case PhysicalType::INT64:
		return sizeof(int64_t);
	case PhysicalType::UINT8:
		return sizeof(uint8_t);
	case PhysicalType::UINT16:
		return sizeof(uint16_t);
	case PhysicalType::UINT32:
		return sizeof(uint32_t);
	case PhysicalType::UINT64:
		return sizeof(uint64_t);
	case PhysicalType::INT128:
		return sizeof(hugeint_t);
	case PhysicalType::FLOAT:
		return sizeof(float);
	case PhysicalType::DOUBLE:
		return sizeof(double);
	case PhysicalType::VARCHAR:
		return sizeof(string_t);
	case PhysicalType::INTERVAL:
		return sizeof(interval_t);
	case PhysicalType::STRUCT:
	case PhysicalType::UNKNOWN:
		return 0; // no own payload
	case PhysicalType::LIST:
		return sizeof(list_entry_t); // offset + len
	default:
		throw InternalException("Invalid PhysicalType for GetTypeIdSize");
	}
}

bool TypeIsConstantSize(PhysicalType type) {
	return (type >= PhysicalType::BOOL && type <= PhysicalType::DOUBLE) || type == PhysicalType::INTERVAL ||
	       type == PhysicalType::INT128;
}
bool TypeIsIntegral(PhysicalType type) {
	return (type >= PhysicalType::UINT8 && type <= PhysicalType::INT64) || type == PhysicalType::INT128;
}
bool TypeIsNumeric(PhysicalType type) {
	return (type >= PhysicalType::UINT8 && type <= PhysicalType::DOUBLE) || type == PhysicalType::INT128;
}
bool TypeIsInteger(PhysicalType type) {
	return (type >= PhysicalType::UINT8 && type <= PhysicalType::INT64) || type == PhysicalType::INT128;
}

// LCOV_EXCL_START
string LogicalTypeIdToString(LogicalTypeId id) {
	switch (id) {
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
	case LogicalTypeId::HUGEINT:
		return "HUGEINT";
	case LogicalTypeId::UUID:
		return "UUID";
	case LogicalTypeId::UTINYINT:
		return "UTINYINT";
	case LogicalTypeId::USMALLINT:
		return "USMALLINT";
	case LogicalTypeId::UINTEGER:
		return "UINTEGER";
	case LogicalTypeId::UBIGINT:
		return "UBIGINT";
	case LogicalTypeId::DATE:
		return "DATE";
	case LogicalTypeId::TIME:
		return "TIME";
	case LogicalTypeId::TIMESTAMP:
		return "TIMESTAMP";
	case LogicalTypeId::TIMESTAMP_MS:
		return "TIMESTAMP_MS";
	case LogicalTypeId::TIMESTAMP_NS:
		return "TIMESTAMP_NS";
	case LogicalTypeId::TIMESTAMP_SEC:
		return "TIMESTAMP_S";
	case LogicalTypeId::TIMESTAMP_TZ:
		return "TIMESTAMP WITH TIME ZONE";
	case LogicalTypeId::TIME_TZ:
		return "TIME WITH TIME ZONE";
	case LogicalTypeId::FLOAT:
		return "FLOAT";
	case LogicalTypeId::DOUBLE:
		return "DOUBLE";
	case LogicalTypeId::DECIMAL:
		return "DECIMAL";
	case LogicalTypeId::VARCHAR:
		return "VARCHAR";
	case LogicalTypeId::BLOB:
		return "BLOB";
	case LogicalTypeId::CHAR:
		return "CHAR";
	case LogicalTypeId::INTERVAL:
		return "INTERVAL";
	case LogicalTypeId::SQLNULL:
		return "NULL";
	case LogicalTypeId::ANY:
		return "ANY";
	case LogicalTypeId::VALIDITY:
		return "VALIDITY";
	case LogicalTypeId::STRUCT:
		return "STRUCT";
	case LogicalTypeId::LIST:
		return "LIST";
	case LogicalTypeId::MAP:
		return "MAP";
	case LogicalTypeId::POINTER:
		return "POINTER";
	case LogicalTypeId::TABLE:
		return "TABLE";
	case LogicalTypeId::LAMBDA:
		return "LAMBDA";
	case LogicalTypeId::INVALID:
		return "INVALID";
	case LogicalTypeId::UNION:
		return "UNION";
	case LogicalTypeId::UNKNOWN:
		return "UNKNOWN";
	case LogicalTypeId::ENUM:
		return "ENUM";
	case LogicalTypeId::AGGREGATE_STATE:
		return "AGGREGATE_STATE";
	case LogicalTypeId::USER:
		return "USER";
	case LogicalTypeId::JSON:
		return "JSON";
	}
	return "UNDEFINED";
}

string LogicalType::ToString() const {
	auto alias = GetAlias();
	if (!alias.empty()) {
		return alias;
	}
	switch (id_) {
	case LogicalTypeId::STRUCT: {
		if (!type_info_) {
			return "STRUCT";
		}
		auto &child_types = StructType::GetChildTypes(*this);
		string ret = "STRUCT(";
		for (size_t i = 0; i < child_types.size(); i++) {
			ret += child_types[i].first + " " + child_types[i].second.ToString();
			if (i < child_types.size() - 1) {
				ret += ", ";
			}
		}
		ret += ")";
		return ret;
	}
	case LogicalTypeId::LIST: {
		if (!type_info_) {
			return "LIST";
		}
		return ListType::GetChildType(*this).ToString() + "[]";
	}
	case LogicalTypeId::MAP: {
		if (!type_info_) {
			return "MAP";
		}
		auto &child_types = StructType::GetChildTypes(*this);
		if (child_types.empty()) {
			return "MAP(?)";
		}
		if (child_types.size() != 2) {
			throw InternalException("Map needs exactly two child elements");
		}
		return "MAP(" + ListType::GetChildType(child_types[0].second).ToString() + ", " +
		       ListType::GetChildType(child_types[1].second).ToString() + ")";
	}
	case LogicalTypeId::UNION: {
		if (!type_info_) {
			return "UNION";
		}
		string ret = "UNION(";
		size_t count = UnionType::GetMemberCount(*this);
		for (size_t i = 0; i < count; i++) {
			ret += UnionType::GetMemberType(*this, i).ToString();
			if (i < count - 1) {
				ret += ", ";
			}
		}
		ret += ")";
		return ret;
	}
	case LogicalTypeId::DECIMAL: {
		if (!type_info_) {
			return "DECIMAL";
		}
		auto width = DecimalType::GetWidth(*this);
		auto scale = DecimalType::GetScale(*this);
		if (width == 0) {
			return "DECIMAL";
		}
		return StringUtil::Format("DECIMAL(%d,%d)", width, scale);
	}
	case LogicalTypeId::ENUM: {
		return KeywordHelper::WriteOptionallyQuoted(EnumType::GetTypeName(*this));
	}
	case LogicalTypeId::USER: {
		return KeywordHelper::WriteOptionallyQuoted(UserType::GetTypeName(*this));
	}
	case LogicalTypeId::AGGREGATE_STATE: {
		return AggregateStateType::GetTypeName(*this);
	}
	default:
		return LogicalTypeIdToString(id_);
	}
}
// LCOV_EXCL_STOP

LogicalTypeId TransformStringToLogicalTypeId(const string &str) {
	auto type = DefaultTypeGenerator::GetDefaultType(str);
	if (type == LogicalTypeId::INVALID) {
		// This is a User Type, at this point we don't know if its one of the User Defined Types or an error
		// It is checked in the binder
		type = LogicalTypeId::USER;
	}
	return type;
}

LogicalType TransformStringToLogicalType(const string &str) {
	if (StringUtil::Lower(str) == "null") {
		return LogicalType::SQLNULL;
	}
	return Parser::ParseColumnList("dummy " + str).GetColumn(LogicalIndex(0)).Type();
}

bool LogicalType::IsIntegral() const {
	switch (id_) {
	case LogicalTypeId::TINYINT:
	case LogicalTypeId::SMALLINT:
	case LogicalTypeId::INTEGER:
	case LogicalTypeId::BIGINT:
	case LogicalTypeId::UTINYINT:
	case LogicalTypeId::USMALLINT:
	case LogicalTypeId::UINTEGER:
	case LogicalTypeId::UBIGINT:
	case LogicalTypeId::HUGEINT:
		return true;
	default:
		return false;
	}
}

bool LogicalType::IsNumeric() const {
	switch (id_) {
	case LogicalTypeId::TINYINT:
	case LogicalTypeId::SMALLINT:
	case LogicalTypeId::INTEGER:
	case LogicalTypeId::BIGINT:
	case LogicalTypeId::HUGEINT:
	case LogicalTypeId::FLOAT:
	case LogicalTypeId::DOUBLE:
	case LogicalTypeId::DECIMAL:
	case LogicalTypeId::UTINYINT:
	case LogicalTypeId::USMALLINT:
	case LogicalTypeId::UINTEGER:
	case LogicalTypeId::UBIGINT:
		return true;
	default:
		return false;
	}
}

bool LogicalType::IsValid() const {
	return id() != LogicalTypeId::INVALID && id() != LogicalTypeId::UNKNOWN;
}

bool LogicalType::GetDecimalProperties(uint8_t &width, uint8_t &scale) const {
	switch (id_) {
	case LogicalTypeId::SQLNULL:
		width = 0;
		scale = 0;
		break;
	case LogicalTypeId::BOOLEAN:
		width = 1;
		scale = 0;
		break;
	case LogicalTypeId::TINYINT:
		// tinyint: [-127, 127] = DECIMAL(3,0)
		width = 3;
		scale = 0;
		break;
	case LogicalTypeId::SMALLINT:
		// smallint: [-32767, 32767] = DECIMAL(5,0)
		width = 5;
		scale = 0;
		break;
	case LogicalTypeId::INTEGER:
		// integer: [-2147483647, 2147483647] = DECIMAL(10,0)
		width = 10;
		scale = 0;
		break;
	case LogicalTypeId::BIGINT:
		// bigint: [-9223372036854775807, 9223372036854775807] = DECIMAL(19,0)
		width = 19;
		scale = 0;
		break;
	case LogicalTypeId::UTINYINT:
		// UInt8 — [0 : 255]
		width = 3;
		scale = 0;
		break;
	case LogicalTypeId::USMALLINT:
		// UInt16 — [0 : 65535]
		width = 5;
		scale = 0;
		break;
	case LogicalTypeId::UINTEGER:
		// UInt32 — [0 : 4294967295]
		width = 10;
		scale = 0;
		break;
	case LogicalTypeId::UBIGINT:
		// UInt64 — [0 : 18446744073709551615]
		width = 20;
		scale = 0;
		break;
	case LogicalTypeId::HUGEINT:
		// hugeint: max size decimal (38, 0)
		// note that a hugeint is not guaranteed to fit in this
		width = 38;
		scale = 0;
		break;
	case LogicalTypeId::DECIMAL:
		width = DecimalType::GetWidth(*this);
		scale = DecimalType::GetScale(*this);
		break;
	default:
		return false;
	}
	return true;
}

//! Grows Decimal width/scale when appropriate
static LogicalType DecimalSizeCheck(const LogicalType &left, const LogicalType &right) {
	D_ASSERT(left.id() == LogicalTypeId::DECIMAL || right.id() == LogicalTypeId::DECIMAL);
	D_ASSERT(left.id() != right.id());

	//! Make sure the 'right' is the DECIMAL type
	if (left.id() == LogicalTypeId::DECIMAL) {
		return DecimalSizeCheck(right, left);
	}
	auto width = DecimalType::GetWidth(right);
	auto scale = DecimalType::GetScale(right);

	uint8_t other_width;
	uint8_t other_scale;
	bool success = left.GetDecimalProperties(other_width, other_scale);
	if (!success) {
		throw InternalException("Type provided to DecimalSizeCheck was not a numeric type");
	}
	D_ASSERT(other_scale == 0);
	const auto effective_width = width - scale;
	if (other_width > effective_width) {
		auto new_width = other_width + scale;
		//! Cap the width at max, if an actual value exceeds this, an exception will be thrown later
		if (new_width > DecimalType::MaxWidth()) {
			new_width = DecimalType::MaxWidth();
		}
		return LogicalType::DECIMAL(new_width, scale);
	}
	return right;
}

static LogicalType CombineNumericTypes(const LogicalType &left, const LogicalType &right) {
	D_ASSERT(left.id() != right.id());
	if (left.id() > right.id()) {
		// this method is symmetric
		// arrange it so the left type is smaller to limit the number of options we need to check
		return CombineNumericTypes(right, left);
	}
	if (CastRules::ImplicitCast(left, right) >= 0) {
		// we can implicitly cast left to right, return right
		//! Depending on the type, we might need to grow the `width` of the DECIMAL type
		if (right.id() == LogicalTypeId::DECIMAL) {
			return DecimalSizeCheck(left, right);
		}
		return right;
	}
	if (CastRules::ImplicitCast(right, left) >= 0) {
		// we can implicitly cast right to left, return left
		//! Depending on the type, we might need to grow the `width` of the DECIMAL type
		if (left.id() == LogicalTypeId::DECIMAL) {
			return DecimalSizeCheck(right, left);
		}
		return left;
	}
	// we can't cast implicitly either way and types are not equal
	// this happens when left is signed and right is unsigned
	// e.g. INTEGER and UINTEGER
	// in this case we need to upcast to make sure the types fit

	if (left.id() == LogicalTypeId::BIGINT || right.id() == LogicalTypeId::UBIGINT) {
		return LogicalType::HUGEINT;
	}
	if (left.id() == LogicalTypeId::INTEGER || right.id() == LogicalTypeId::UINTEGER) {
		return LogicalType::BIGINT;
	}
	if (left.id() == LogicalTypeId::SMALLINT || right.id() == LogicalTypeId::USMALLINT) {
		return LogicalType::INTEGER;
	}
	if (left.id() == LogicalTypeId::TINYINT || right.id() == LogicalTypeId::UTINYINT) {
		return LogicalType::SMALLINT;
	}
	throw InternalException("Cannot combine these numeric types!?");
}

LogicalType LogicalType::MaxLogicalType(const LogicalType &left, const LogicalType &right) {
	// we always prefer aliased types
	if (!left.GetAlias().empty()) {
		return left;
	}
	if (!right.GetAlias().empty()) {
		return right;
	}
	if (left.id() != right.id() && left.IsNumeric() && right.IsNumeric()) {
		return CombineNumericTypes(left, right);
	} else if (left.id() == LogicalTypeId::UNKNOWN) {
		return right;
	} else if (right.id() == LogicalTypeId::UNKNOWN) {
		return left;
	} else if (left.id() < right.id()) {
		return right;
	}
	if (right.id() < left.id()) {
		return left;
	}
	// Since both left and right are equal we get the left type as our type_id for checks
	auto type_id = left.id();
	if (type_id == LogicalTypeId::ENUM) {
		// If both types are different ENUMs we do a string comparison.
		return left == right ? left : LogicalType::VARCHAR;
	}
	if (type_id == LogicalTypeId::VARCHAR) {
		// varchar: use type that has collation (if any)
		if (StringType::GetCollation(right).empty()) {
			return left;
		}
		return right;
	}
	if (type_id == LogicalTypeId::DECIMAL) {
		// unify the width/scale so that the resulting decimal always fits
		// "width - scale" gives us the number of digits on the left side of the decimal point
		// "scale" gives us the number of digits allowed on the right of the decimal point
		// using the max of these of the two types gives us the new decimal size
		auto extra_width_left = DecimalType::GetWidth(left) - DecimalType::GetScale(left);
		auto extra_width_right = DecimalType::GetWidth(right) - DecimalType::GetScale(right);
		auto extra_width = MaxValue<uint8_t>(extra_width_left, extra_width_right);
		auto scale = MaxValue<uint8_t>(DecimalType::GetScale(left), DecimalType::GetScale(right));
		auto width = extra_width + scale;
		if (width > DecimalType::MaxWidth()) {
			// if the resulting decimal does not fit, we truncate the scale
			width = DecimalType::MaxWidth();
			scale = width - extra_width;
		}
		return LogicalType::DECIMAL(width, scale);
	}
	if (type_id == LogicalTypeId::LIST) {
		// list: perform max recursively on child type
		auto new_child = MaxLogicalType(ListType::GetChildType(left), ListType::GetChildType(right));
		return LogicalType::LIST(move(new_child));
	}
	if (type_id == LogicalTypeId::STRUCT || type_id == LogicalTypeId::MAP) {
		// struct: perform recursively
		auto &left_child_types = StructType::GetChildTypes(left);
		auto &right_child_types = StructType::GetChildTypes(right);
		if (left_child_types.size() != right_child_types.size()) {
			// child types are not of equal size, we can't cast anyway
			// just return the left child
			return left;
		}
		child_list_t<LogicalType> child_types;
		for (idx_t i = 0; i < left_child_types.size(); i++) {
			auto child_type = MaxLogicalType(left_child_types[i].second, right_child_types[i].second);
			child_types.push_back(make_pair(left_child_types[i].first, move(child_type)));
		}

		return type_id == LogicalTypeId::STRUCT ? LogicalType::STRUCT(move(child_types))
		                                        : LogicalType::MAP(move(child_types));
	}
	if (type_id == LogicalTypeId::UNION) {
		auto left_member_count = UnionType::GetMemberCount(left);
		auto right_member_count = UnionType::GetMemberCount(right);
		if (left_member_count != right_member_count) {
			// return the "larger" type, with the most members
			return left_member_count > right_member_count ? left : right;
		}
		// otherwise, keep left, dont try to meld the two together.
		return left;
	}
	// types are equal but no extra specifier: just return the type
	return left;
}

void LogicalType::Verify() const {
#ifdef DEBUG
	if (id_ == LogicalTypeId::DECIMAL) {
		D_ASSERT(DecimalType::GetWidth(*this) >= 1 && DecimalType::GetWidth(*this) <= Decimal::MAX_WIDTH_DECIMAL);
		D_ASSERT(DecimalType::GetScale(*this) >= 0 && DecimalType::GetScale(*this) <= DecimalType::GetWidth(*this));
	}
#endif
}

bool ApproxEqual(float ldecimal, float rdecimal) {
	if (Value::IsNan(ldecimal) && Value::IsNan(rdecimal)) {
		return true;
	}
	if (!Value::FloatIsFinite(ldecimal) || !Value::FloatIsFinite(rdecimal)) {
		return ldecimal == rdecimal;
	}
	float epsilon = std::fabs(rdecimal) * 0.01 + 0.00000001;
	return std::fabs(ldecimal - rdecimal) <= epsilon;
}

bool ApproxEqual(double ldecimal, double rdecimal) {
	if (Value::IsNan(ldecimal) && Value::IsNan(rdecimal)) {
		return true;
	}
	if (!Value::DoubleIsFinite(ldecimal) || !Value::DoubleIsFinite(rdecimal)) {
		return ldecimal == rdecimal;
	}
	double epsilon = std::fabs(rdecimal) * 0.01 + 0.00000001;
	return std::fabs(ldecimal - rdecimal) <= epsilon;
}

//===--------------------------------------------------------------------===//
// Extra Type Info
//===--------------------------------------------------------------------===//
enum class ExtraTypeInfoType : uint8_t {
	INVALID_TYPE_INFO = 0,
	GENERIC_TYPE_INFO = 1,
	DECIMAL_TYPE_INFO = 2,
	STRING_TYPE_INFO = 3,
	LIST_TYPE_INFO = 4,
	STRUCT_TYPE_INFO = 5,
	ENUM_TYPE_INFO = 6,
	USER_TYPE_INFO = 7,
	AGGREGATE_STATE_TYPE_INFO = 8
};

struct ExtraTypeInfo {
	explicit ExtraTypeInfo(ExtraTypeInfoType type) : type(type) {
	}
	explicit ExtraTypeInfo(ExtraTypeInfoType type, string alias) : type(type), alias(move(alias)) {
	}
	virtual ~ExtraTypeInfo() {
	}

	ExtraTypeInfoType type;
	string alias;
	TypeCatalogEntry *catalog_entry = nullptr;

public:
	bool Equals(ExtraTypeInfo *other_p) const {
		if (type == ExtraTypeInfoType::INVALID_TYPE_INFO || type == ExtraTypeInfoType::STRING_TYPE_INFO ||
		    type == ExtraTypeInfoType::GENERIC_TYPE_INFO) {
			if (!other_p) {
				if (!alias.empty()) {
					return false;
				}
				//! We only need to compare aliases when both types have them in this case
				return true;
			}
			if (alias != other_p->alias) {
				return false;
			}
			return true;
		}
		if (!other_p) {
			return false;
		}
		if (type != other_p->type) {
			return false;
		}
		return alias == other_p->alias && EqualsInternal(other_p);
	}
	//! Serializes a ExtraTypeInfo to a stand-alone binary blob
	virtual void Serialize(FieldWriter &writer) const {};
	//! Serializes a ExtraTypeInfo to a stand-alone binary blob
	static void Serialize(ExtraTypeInfo *info, FieldWriter &writer);
	//! Deserializes a blob back into an ExtraTypeInfo
	static shared_ptr<ExtraTypeInfo> Deserialize(FieldReader &reader);

protected:
	virtual bool EqualsInternal(ExtraTypeInfo *other_p) const {
		// Do nothing
		return true;
	}
};

void LogicalType::SetAlias(string alias) {
	if (!type_info_) {
		type_info_ = make_shared<ExtraTypeInfo>(ExtraTypeInfoType::GENERIC_TYPE_INFO, move(alias));
	} else {
		type_info_->alias = move(alias);
	}
}

string LogicalType::GetAlias() const {
	if (!type_info_) {
		return string();
	} else {
		return type_info_->alias;
	}
}

bool LogicalType::HasAlias() const {
	if (!type_info_) {
		return false;
	}
	return !type_info_->alias.empty();
}

void LogicalType::SetCatalog(LogicalType &type, TypeCatalogEntry *catalog_entry) {
	auto info = type.AuxInfo();
	D_ASSERT(info);
	((ExtraTypeInfo &)*info).catalog_entry = catalog_entry;
}
TypeCatalogEntry *LogicalType::GetCatalog(const LogicalType &type) {
	auto info = type.AuxInfo();
	if (!info) {
		return nullptr;
	}
	return ((ExtraTypeInfo &)*info).catalog_entry;
}

//===--------------------------------------------------------------------===//
// Decimal Type
//===--------------------------------------------------------------------===//
struct DecimalTypeInfo : public ExtraTypeInfo {
	DecimalTypeInfo(uint8_t width_p, uint8_t scale_p)
	    : ExtraTypeInfo(ExtraTypeInfoType::DECIMAL_TYPE_INFO), width(width_p), scale(scale_p) {
		D_ASSERT(width_p >= scale_p);
	}

	uint8_t width;
	uint8_t scale;

public:
	void Serialize(FieldWriter &writer) const override {
		writer.WriteField<uint8_t>(width);
		writer.WriteField<uint8_t>(scale);
	}

	static shared_ptr<ExtraTypeInfo> Deserialize(FieldReader &reader) {
		auto width = reader.ReadRequired<uint8_t>();
		auto scale = reader.ReadRequired<uint8_t>();
		return make_shared<DecimalTypeInfo>(width, scale);
	}

protected:
	bool EqualsInternal(ExtraTypeInfo *other_p) const override {
		auto &other = (DecimalTypeInfo &)*other_p;
		return width == other.width && scale == other.scale;
	}
};

uint8_t DecimalType::GetWidth(const LogicalType &type) {
	D_ASSERT(type.id() == LogicalTypeId::DECIMAL);
	auto info = type.AuxInfo();
	D_ASSERT(info);
	return ((DecimalTypeInfo &)*info).width;
}

uint8_t DecimalType::GetScale(const LogicalType &type) {
	D_ASSERT(type.id() == LogicalTypeId::DECIMAL);
	auto info = type.AuxInfo();
	D_ASSERT(info);
	return ((DecimalTypeInfo &)*info).scale;
}

uint8_t DecimalType::MaxWidth() {
	return DecimalWidth<hugeint_t>::max;
}

LogicalType LogicalType::DECIMAL(int width, int scale) {
	D_ASSERT(width >= scale);
	auto type_info = make_shared<DecimalTypeInfo>(width, scale);
	return LogicalType(LogicalTypeId::DECIMAL, move(type_info));
}

//===--------------------------------------------------------------------===//
// String Type
//===--------------------------------------------------------------------===//
struct StringTypeInfo : public ExtraTypeInfo {
	explicit StringTypeInfo(string collation_p)
	    : ExtraTypeInfo(ExtraTypeInfoType::STRING_TYPE_INFO), collation(move(collation_p)) {
	}

	string collation;

public:
	void Serialize(FieldWriter &writer) const override {
		writer.WriteString(collation);
	}

	static shared_ptr<ExtraTypeInfo> Deserialize(FieldReader &reader) {
		auto collation = reader.ReadRequired<string>();
		return make_shared<StringTypeInfo>(move(collation));
	}

protected:
	bool EqualsInternal(ExtraTypeInfo *other_p) const override {
		// collation info has no impact on equality
		return true;
	}
};

string StringType::GetCollation(const LogicalType &type) {
	if (type.id() != LogicalTypeId::VARCHAR) {
		return string();
	}
	auto info = type.AuxInfo();
	if (!info) {
		return string();
	}
	if (info->type == ExtraTypeInfoType::GENERIC_TYPE_INFO) {
		return string();
	}
	return ((StringTypeInfo &)*info).collation;
}

LogicalType LogicalType::VARCHAR_COLLATION(string collation) { // NOLINT
	auto string_info = make_shared<StringTypeInfo>(move(collation));
	return LogicalType(LogicalTypeId::VARCHAR, move(string_info));
}

//===--------------------------------------------------------------------===//
// List Type
//===--------------------------------------------------------------------===//
struct ListTypeInfo : public ExtraTypeInfo {
	explicit ListTypeInfo(LogicalType child_type_p)
	    : ExtraTypeInfo(ExtraTypeInfoType::LIST_TYPE_INFO), child_type(move(child_type_p)) {
	}

	LogicalType child_type;

public:
	void Serialize(FieldWriter &writer) const override {
		writer.WriteSerializable(child_type);
	}

	static shared_ptr<ExtraTypeInfo> Deserialize(FieldReader &reader) {
		auto child_type = reader.ReadRequiredSerializable<LogicalType, LogicalType>();
		return make_shared<ListTypeInfo>(move(child_type));
	}

protected:
	bool EqualsInternal(ExtraTypeInfo *other_p) const override {
		auto &other = (ListTypeInfo &)*other_p;
		return child_type == other.child_type;
	}
};

const LogicalType &ListType::GetChildType(const LogicalType &type) {
	D_ASSERT(type.id() == LogicalTypeId::LIST);
	auto info = type.AuxInfo();
	D_ASSERT(info);
	return ((ListTypeInfo &)*info).child_type;
}

LogicalType LogicalType::LIST(LogicalType child) {
	auto info = make_shared<ListTypeInfo>(move(child));
	return LogicalType(LogicalTypeId::LIST, move(info));
}

//===--------------------------------------------------------------------===//
// Struct Type
//===--------------------------------------------------------------------===//
struct StructTypeInfo : public ExtraTypeInfo {
	explicit StructTypeInfo(child_list_t<LogicalType> child_types_p)
	    : ExtraTypeInfo(ExtraTypeInfoType::STRUCT_TYPE_INFO), child_types(move(child_types_p)) {
	}

	child_list_t<LogicalType> child_types;

public:
	void Serialize(FieldWriter &writer) const override {
		writer.WriteField<uint32_t>(child_types.size());
		auto &serializer = writer.GetSerializer();
		for (idx_t i = 0; i < child_types.size(); i++) {
			serializer.WriteString(child_types[i].first);
			child_types[i].second.Serialize(serializer);
		}
	}

	static shared_ptr<ExtraTypeInfo> Deserialize(FieldReader &reader) {
		child_list_t<LogicalType> child_list;
		auto child_types_size = reader.ReadRequired<uint32_t>();
		auto &source = reader.GetSource();
		for (uint32_t i = 0; i < child_types_size; i++) {
			auto name = source.Read<string>();
			auto type = LogicalType::Deserialize(source);
			child_list.push_back(make_pair(move(name), move(type)));
		}
		return make_shared<StructTypeInfo>(move(child_list));
	}

protected:
	bool EqualsInternal(ExtraTypeInfo *other_p) const override {
		auto &other = (StructTypeInfo &)*other_p;
		return child_types == other.child_types;
	}
};

struct AggregateStateTypeInfo : public ExtraTypeInfo {
	explicit AggregateStateTypeInfo(aggregate_state_t state_type_p)
	    : ExtraTypeInfo(ExtraTypeInfoType::AGGREGATE_STATE_TYPE_INFO), state_type(move(state_type_p)) {
	}

	aggregate_state_t state_type;

public:
	void Serialize(FieldWriter &writer) const override {
		auto &serializer = writer.GetSerializer();
		writer.WriteString(state_type.function_name);
		state_type.return_type.Serialize(serializer);
		writer.WriteField<uint32_t>(state_type.bound_argument_types.size());
		for (idx_t i = 0; i < state_type.bound_argument_types.size(); i++) {
			state_type.bound_argument_types[i].Serialize(serializer);
		}
	}

	static shared_ptr<ExtraTypeInfo> Deserialize(FieldReader &reader) {
		auto &source = reader.GetSource();

		auto function_name = reader.ReadRequired<string>();
		auto return_type = LogicalType::Deserialize(source);
		auto bound_argument_types_size = reader.ReadRequired<uint32_t>();
		vector<LogicalType> bound_argument_types;

		for (uint32_t i = 0; i < bound_argument_types_size; i++) {
			auto type = LogicalType::Deserialize(source);
			bound_argument_types.push_back(move(type));
		}
		return make_shared<AggregateStateTypeInfo>(
		    aggregate_state_t(move(function_name), move(return_type), move(bound_argument_types)));
	}

protected:
	bool EqualsInternal(ExtraTypeInfo *other_p) const override {
		auto &other = (AggregateStateTypeInfo &)*other_p;
		return state_type.function_name == other.state_type.function_name &&
		       state_type.return_type == other.state_type.return_type &&
		       state_type.bound_argument_types == other.state_type.bound_argument_types;
	}
};

const aggregate_state_t &AggregateStateType::GetStateType(const LogicalType &type) {
	D_ASSERT(type.id() == LogicalTypeId::AGGREGATE_STATE);
	auto info = type.AuxInfo();
	D_ASSERT(info);
	return ((AggregateStateTypeInfo &)*info).state_type;
}

const string AggregateStateType::GetTypeName(const LogicalType &type) {
	D_ASSERT(type.id() == LogicalTypeId::AGGREGATE_STATE);
	auto info = type.AuxInfo();
	if (!info) {
		return "AGGREGATE_STATE<?>";
	}
	auto aggr_state = ((AggregateStateTypeInfo &)*info).state_type;
	return "AGGREGATE_STATE<" + aggr_state.function_name + "(" +
	       StringUtil::Join(aggr_state.bound_argument_types, aggr_state.bound_argument_types.size(), ", ",
	                        [](const LogicalType &arg_type) { return arg_type.ToString(); }) +
	       ")" + "::" + aggr_state.return_type.ToString() + ">";
}

const child_list_t<LogicalType> &StructType::GetChildTypes(const LogicalType &type) {
	D_ASSERT(type.id() == LogicalTypeId::STRUCT || type.id() == LogicalTypeId::MAP ||
	         type.id() == LogicalTypeId::UNION);

	auto info = type.AuxInfo();
	D_ASSERT(info);
	return ((StructTypeInfo &)*info).child_types;
}

const LogicalType &StructType::GetChildType(const LogicalType &type, idx_t index) {
	auto &child_types = StructType::GetChildTypes(type);
	D_ASSERT(index < child_types.size());
	return child_types[index].second;
}

const string &StructType::GetChildName(const LogicalType &type, idx_t index) {
	auto &child_types = StructType::GetChildTypes(type);
	D_ASSERT(index < child_types.size());
	return child_types[index].first;
}

idx_t StructType::GetChildCount(const LogicalType &type) {
	return StructType::GetChildTypes(type).size();
}

LogicalType LogicalType::STRUCT(child_list_t<LogicalType> children) {
	auto info = make_shared<StructTypeInfo>(move(children));
	return LogicalType(LogicalTypeId::STRUCT, move(info));
}

LogicalType LogicalType::AGGREGATE_STATE(aggregate_state_t state_type) { // NOLINT
	auto info = make_shared<AggregateStateTypeInfo>(move(state_type));
	return LogicalType(LogicalTypeId::AGGREGATE_STATE, move(info));
}

//===--------------------------------------------------------------------===//
// Map Type
//===--------------------------------------------------------------------===//
LogicalType LogicalType::MAP(child_list_t<LogicalType> children) {
	auto info = make_shared<StructTypeInfo>(move(children));
	return LogicalType(LogicalTypeId::MAP, move(info));
}

LogicalType LogicalType::MAP(LogicalType key, LogicalType value) {
	child_list_t<LogicalType> child_types;
	child_types.push_back({"key", LogicalType::LIST(move(key))});
	child_types.push_back({"value", LogicalType::LIST(move(value))});
	return LogicalType::MAP(move(child_types));
}

const LogicalType &MapType::KeyType(const LogicalType &type) {
	D_ASSERT(type.id() == LogicalTypeId::MAP);
	return ListType::GetChildType(StructType::GetChildTypes(type)[0].second);
}

const LogicalType &MapType::ValueType(const LogicalType &type) {
	D_ASSERT(type.id() == LogicalTypeId::MAP);
	return ListType::GetChildType(StructType::GetChildTypes(type)[1].second);
}

//===--------------------------------------------------------------------===//
// Union Type
//===--------------------------------------------------------------------===//

LogicalType LogicalType::UNION(child_list_t<LogicalType> members) {
	D_ASSERT(members.size() > 0);
	D_ASSERT(members.size() <= UnionType::MAX_UNION_MEMBERS);
	// union types always have a hidden "tag" field in front
	members.insert(members.begin(), {"", LogicalType::TINYINT});
	auto info = make_shared<StructTypeInfo>(move(members));
	return LogicalType(LogicalTypeId::UNION, move(info));
}

const LogicalType &UnionType::GetMemberType(const LogicalType &type, idx_t index) {
	auto &child_types = StructType::GetChildTypes(type);
	D_ASSERT(index < child_types.size());
	// skip the "tag" field
	return child_types[index + 1].second;
}

const string &UnionType::GetMemberName(const LogicalType &type, idx_t index) {
	auto &child_types = StructType::GetChildTypes(type);
	D_ASSERT(index < child_types.size());
	// skip the "tag" field
	return child_types[index + 1].first;
}

idx_t UnionType::GetMemberCount(const LogicalType &type) {
	// dont count the "tag" field
	return StructType::GetChildTypes(type).size() - 1;
}
const child_list_t<LogicalType> UnionType::CopyMemberTypes(const LogicalType &type) {
	auto child_types = StructType::GetChildTypes(type);
	child_types.erase(child_types.begin());
	return child_types;
}

//===--------------------------------------------------------------------===//
// User Type
//===--------------------------------------------------------------------===//
struct UserTypeInfo : public ExtraTypeInfo {
	explicit UserTypeInfo(string name_p)
	    : ExtraTypeInfo(ExtraTypeInfoType::USER_TYPE_INFO), user_type_name(move(name_p)) {
	}

	string user_type_name;

public:
	void Serialize(FieldWriter &writer) const override {
		writer.WriteString(user_type_name);
	}

	static shared_ptr<ExtraTypeInfo> Deserialize(FieldReader &reader) {
		auto enum_name = reader.ReadRequired<string>();
		return make_shared<UserTypeInfo>(move(enum_name));
	}

protected:
	bool EqualsInternal(ExtraTypeInfo *other_p) const override {
		auto &other = (UserTypeInfo &)*other_p;
		return other.user_type_name == user_type_name;
	}
};

const string &UserType::GetTypeName(const LogicalType &type) {
	D_ASSERT(type.id() == LogicalTypeId::USER);
	auto info = type.AuxInfo();
	D_ASSERT(info);
	return ((UserTypeInfo &)*info).user_type_name;
}

LogicalType LogicalType::USER(const string &user_type_name) {
	auto info = make_shared<UserTypeInfo>(user_type_name);
	return LogicalType(LogicalTypeId::USER, move(info));
}

//===--------------------------------------------------------------------===//
// Enum Type
//===--------------------------------------------------------------------===//

enum EnumDictType : uint8_t { INVALID = 0, VECTOR_DICT = 1, DEDUP_POINTER = 2 };

struct EnumTypeInfo : public ExtraTypeInfo {
	explicit EnumTypeInfo(string enum_name_p, Vector &values_insert_order_p, idx_t dict_size_p)
	    : ExtraTypeInfo(ExtraTypeInfoType::ENUM_TYPE_INFO), dict_type(EnumDictType::VECTOR_DICT),
	      enum_name(move(enum_name_p)), values_insert_order(values_insert_order_p), dict_size(dict_size_p) {
	}
	explicit EnumTypeInfo()
	    : ExtraTypeInfo(ExtraTypeInfoType::ENUM_TYPE_INFO), dict_type(EnumDictType::DEDUP_POINTER),
	      enum_name("dedup_pointer"), values_insert_order(Vector(LogicalType::VARCHAR)), dict_size(0) {
	}
	EnumDictType dict_type;
	string enum_name;
	Vector values_insert_order;
	idx_t dict_size;

protected:
	// Equalities are only used in enums with different catalog entries
	bool EqualsInternal(ExtraTypeInfo *other_p) const override {
		auto &other = (EnumTypeInfo &)*other_p;
		if (dict_type != other.dict_type) {
			return false;
		}
		if (dict_type == EnumDictType::DEDUP_POINTER) {
			return true;
		}
		D_ASSERT(dict_type == EnumDictType::VECTOR_DICT);
		// We must check if both enums have the same size
		if (other.dict_size != dict_size) {
			return false;
		}
		auto other_vector_ptr = FlatVector::GetData<string_t>(other.values_insert_order);
		auto this_vector_ptr = FlatVector::GetData<string_t>(values_insert_order);

		// Now we must check if all strings are the same
		for (idx_t i = 0; i < dict_size; i++) {
			if (!Equals::Operation(other_vector_ptr[i], this_vector_ptr[i])) {
				return false;
			}
		}
		return true;
	}

	void Serialize(FieldWriter &writer) const override {
		if (dict_type != EnumDictType::VECTOR_DICT) {
			throw InternalException("Cannot serialize non-vector dictionary ENUM types");
		}
		writer.WriteField<uint32_t>(dict_size);
		writer.WriteString(enum_name);
		((Vector &)values_insert_order).Serialize(dict_size, writer.GetSerializer());
	}
};

template <class T>
struct EnumTypeInfoTemplated : public EnumTypeInfo {
	explicit EnumTypeInfoTemplated(const string &enum_name_p, Vector &values_insert_order_p, idx_t size_p)
	    : EnumTypeInfo(enum_name_p, values_insert_order_p, size_p) {
		D_ASSERT(values_insert_order_p.GetType().InternalType() == PhysicalType::VARCHAR);

		UnifiedVectorFormat vdata;
		values_insert_order.ToUnifiedFormat(size_p, vdata);

		auto data = (string_t *)vdata.data;
		for (idx_t i = 0; i < size_p; i++) {
			auto idx = vdata.sel->get_index(i);
			if (!vdata.validity.RowIsValid(idx)) {
				throw InternalException("Attempted to create ENUM type with NULL value");
			}
			if (values.count(data[idx]) > 0) {
				throw InvalidInputException("Attempted to create ENUM type with duplicate value %s",
				                            data[idx].GetString());
			}
			values[data[idx]] = i;
		}
	}

	static shared_ptr<EnumTypeInfoTemplated> Deserialize(FieldReader &reader, uint32_t size) {
		auto enum_name = reader.ReadRequired<string>();
		Vector values_insert_order(LogicalType::VARCHAR, size);
		values_insert_order.Deserialize(size, reader.GetSource());
		return make_shared<EnumTypeInfoTemplated>(move(enum_name), values_insert_order, size);
	}

	string_map_t<T> values;
};

const string &EnumType::GetTypeName(const LogicalType &type) {
	D_ASSERT(type.id() == LogicalTypeId::ENUM);
	auto info = type.AuxInfo();
	D_ASSERT(info);
	return ((EnumTypeInfo &)*info).enum_name;
}

static PhysicalType EnumVectorDictType(idx_t size) {
	if (size <= NumericLimits<uint8_t>::Maximum()) {
		return PhysicalType::UINT8;
	} else if (size <= NumericLimits<uint16_t>::Maximum()) {
		return PhysicalType::UINT16;
	} else if (size <= NumericLimits<uint32_t>::Maximum()) {
		return PhysicalType::UINT32;
	} else {
		throw InternalException("Enum size must be lower than " + std::to_string(NumericLimits<uint32_t>::Maximum()));
	}
}

LogicalType LogicalType::ENUM(const string &enum_name, Vector &ordered_data, idx_t size) {
	// Generate EnumTypeInfo
	shared_ptr<ExtraTypeInfo> info;
	auto enum_internal_type = EnumVectorDictType(size);
	switch (enum_internal_type) {
	case PhysicalType::UINT8:
		info = make_shared<EnumTypeInfoTemplated<uint8_t>>(enum_name, ordered_data, size);
		break;
	case PhysicalType::UINT16:
		info = make_shared<EnumTypeInfoTemplated<uint16_t>>(enum_name, ordered_data, size);
		break;
	case PhysicalType::UINT32:
		info = make_shared<EnumTypeInfoTemplated<uint32_t>>(enum_name, ordered_data, size);
		break;
	default:
		throw InternalException("Invalid Physical Type for ENUMs");
	}
	// Generate Actual Enum Type
	return LogicalType(LogicalTypeId::ENUM, info);
}

LogicalType LogicalType::DEDUP_POINTER_ENUM() { // NOLINT
	auto info = make_shared<EnumTypeInfo>();
	D_ASSERT(info->dict_type == EnumDictType::DEDUP_POINTER);
	return LogicalType(LogicalTypeId::ENUM, info);
}

template <class T>
int64_t TemplatedGetPos(string_map_t<T> &map, const string_t &key) {
	auto it = map.find(key);
	if (it == map.end()) {
		return -1;
	}
	return it->second;
}

int64_t EnumType::GetPos(const LogicalType &type, const string_t &key) {
	auto info = type.AuxInfo();
	switch (type.InternalType()) {
	case PhysicalType::UINT8:
		return TemplatedGetPos(((EnumTypeInfoTemplated<uint8_t> &)*info).values, key);
	case PhysicalType::UINT16:
		return TemplatedGetPos(((EnumTypeInfoTemplated<uint16_t> &)*info).values, key);
	case PhysicalType::UINT32:
		return TemplatedGetPos(((EnumTypeInfoTemplated<uint32_t> &)*info).values, key);
	default:
		throw InternalException("ENUM can only have unsigned integers (except UINT64) as physical types");
	}
}

const string EnumType::GetValue(const Value &val) {
	auto info = val.type().AuxInfo();
	auto &enum_info = ((EnumTypeInfo &)*info);
	if (enum_info.dict_type == EnumDictType::DEDUP_POINTER) {
		return (const char *)val.GetValue<uint64_t>();
	}
	auto &values_insert_order = ((EnumTypeInfo &)*info).values_insert_order;
	return StringValue::Get(values_insert_order.GetValue(val.GetValue<uint32_t>()));
}

Vector &EnumType::GetValuesInsertOrder(const LogicalType &type) {
	D_ASSERT(type.id() == LogicalTypeId::ENUM);
	auto info = type.AuxInfo();
	D_ASSERT(info);
	return ((EnumTypeInfo &)*info).values_insert_order;
}

idx_t EnumType::GetSize(const LogicalType &type) {
	D_ASSERT(type.id() == LogicalTypeId::ENUM);
	auto info = type.AuxInfo();
	D_ASSERT(info);
	return ((EnumTypeInfo &)*info).dict_size;
}

void EnumType::SetCatalog(LogicalType &type, TypeCatalogEntry *catalog_entry) {
	D_ASSERT(type.id() == LogicalTypeId::ENUM);
	auto info = type.AuxInfo();
	D_ASSERT(info);
	((EnumTypeInfo &)*info).catalog_entry = catalog_entry;
}
TypeCatalogEntry *EnumType::GetCatalog(const LogicalType &type) {
	D_ASSERT(type.id() == LogicalTypeId::ENUM);
	auto info = type.AuxInfo();
	D_ASSERT(info);
	return ((EnumTypeInfo &)*info).catalog_entry;
}

PhysicalType EnumType::GetPhysicalType(const LogicalType &type) {
	D_ASSERT(type.id() == LogicalTypeId::ENUM);
	auto aux_info = type.AuxInfo();
	D_ASSERT(aux_info);
	auto &info = (EnumTypeInfo &)*aux_info;

	if (info.dict_type == EnumDictType::DEDUP_POINTER) {
		return PhysicalType::UINT64; // for pointer enum types
	}
	D_ASSERT(info.dict_type == EnumDictType::VECTOR_DICT);
	return EnumVectorDictType(info.dict_size);
}

//===--------------------------------------------------------------------===//
// Extra Type Info
//===--------------------------------------------------------------------===//
void ExtraTypeInfo::Serialize(ExtraTypeInfo *info, FieldWriter &writer) {
	if (!info) {
		writer.WriteField<ExtraTypeInfoType>(ExtraTypeInfoType::INVALID_TYPE_INFO);
		writer.WriteString(string());
	} else {
		writer.WriteField<ExtraTypeInfoType>(info->type);
		info->Serialize(writer);
		writer.WriteString(info->alias);
	}
}
shared_ptr<ExtraTypeInfo> ExtraTypeInfo::Deserialize(FieldReader &reader) {
	auto type = reader.ReadRequired<ExtraTypeInfoType>();
	shared_ptr<ExtraTypeInfo> extra_info;
	switch (type) {
	case ExtraTypeInfoType::INVALID_TYPE_INFO: {
		auto alias = reader.ReadField<string>(string());
		if (!alias.empty()) {
			return make_shared<ExtraTypeInfo>(type, alias);
		}
		return nullptr;
	}
	case ExtraTypeInfoType::GENERIC_TYPE_INFO: {
		extra_info = make_shared<ExtraTypeInfo>(type);
	} break;
	case ExtraTypeInfoType::DECIMAL_TYPE_INFO:
		extra_info = DecimalTypeInfo::Deserialize(reader);
		break;
	case ExtraTypeInfoType::STRING_TYPE_INFO:
		extra_info = StringTypeInfo::Deserialize(reader);
		break;
	case ExtraTypeInfoType::LIST_TYPE_INFO:
		extra_info = ListTypeInfo::Deserialize(reader);
		break;
	case ExtraTypeInfoType::STRUCT_TYPE_INFO:
		extra_info = StructTypeInfo::Deserialize(reader);
		break;
	case ExtraTypeInfoType::USER_TYPE_INFO:
		extra_info = UserTypeInfo::Deserialize(reader);
		break;
	case ExtraTypeInfoType::ENUM_TYPE_INFO: {
		auto enum_size = reader.ReadRequired<uint32_t>();
		auto enum_internal_type = EnumVectorDictType(enum_size);
		switch (enum_internal_type) {
		case PhysicalType::UINT8:
			extra_info = EnumTypeInfoTemplated<uint8_t>::Deserialize(reader, enum_size);
			break;
		case PhysicalType::UINT16:
			extra_info = EnumTypeInfoTemplated<uint16_t>::Deserialize(reader, enum_size);
			break;
		case PhysicalType::UINT32:
			extra_info = EnumTypeInfoTemplated<uint32_t>::Deserialize(reader, enum_size);
			break;
		default:
			throw InternalException("Invalid Physical Type for ENUMs");
		}
	} break;
	case ExtraTypeInfoType::AGGREGATE_STATE_TYPE_INFO:
		extra_info = AggregateStateTypeInfo::Deserialize(reader);
		break;

	default:
		throw InternalException("Unimplemented type info in ExtraTypeInfo::Deserialize");
	}
	auto alias = reader.ReadField<string>(string());
	extra_info->alias = alias;
	return extra_info;
}

//===--------------------------------------------------------------------===//
// Logical Type
//===--------------------------------------------------------------------===//

// the destructor needs to know about the extra type info
LogicalType::~LogicalType() {
}

void LogicalType::Serialize(Serializer &serializer) const {
	FieldWriter writer(serializer);
	writer.WriteField<LogicalTypeId>(id_);
	ExtraTypeInfo::Serialize(type_info_.get(), writer);
	writer.Finalize();
}

LogicalType LogicalType::Deserialize(Deserializer &source) {
	FieldReader reader(source);
	auto id = reader.ReadRequired<LogicalTypeId>();
	auto info = ExtraTypeInfo::Deserialize(reader);
	reader.Finalize();

	return LogicalType(id, move(info));
}

bool LogicalType::EqualTypeInfo(const LogicalType &rhs) const {
	if (type_info_.get() == rhs.type_info_.get()) {
		return true;
	}
	if (type_info_) {
		return type_info_->Equals(rhs.type_info_.get());
	} else {
		D_ASSERT(rhs.type_info_);
		return rhs.type_info_->Equals(type_info_.get());
	}
}

bool LogicalType::operator==(const LogicalType &rhs) const {
	if (id_ != rhs.id_) {
		return false;
	}
	return EqualTypeInfo(rhs);
}

} // namespace duckdb





namespace duckdb {

//===--------------------------------------------------------------------===//
// Comparison Operations
//===--------------------------------------------------------------------===//

struct ValuePositionComparator {
	// Return true if the positional Values definitely match.
	// Default to the same as the final value
	template <typename OP>
	static inline bool Definite(const Value &lhs, const Value &rhs) {
		return Final<OP>(lhs, rhs);
	}

	// Select the positional Values that need further testing.
	// Usually this means Is Not Distinct, as those are the semantics used by Postges
	template <typename OP>
	static inline bool Possible(const Value &lhs, const Value &rhs) {
		return ValueOperations::NotDistinctFrom(lhs, rhs);
	}

	// Return true if the positional Values definitely match in the final position
	// This needs to be specialised.
	template <typename OP>
	static inline bool Final(const Value &lhs, const Value &rhs) {
		return false;
	}

	// Tie-break based on length when one of the sides has been exhausted, returning true if the LHS matches.
	// This essentially means that the existing positions compare equal.
	// Default to the same semantics as the OP for idx_t. This works in most cases.
	template <typename OP>
	static inline bool TieBreak(const idx_t lpos, const idx_t rpos) {
		return OP::Operation(lpos, rpos);
	}
};

// Equals must always check every column
template <>
inline bool ValuePositionComparator::Definite<duckdb::Equals>(const Value &lhs, const Value &rhs) {
	return false;
}

template <>
inline bool ValuePositionComparator::Final<duckdb::Equals>(const Value &lhs, const Value &rhs) {
	return ValueOperations::NotDistinctFrom(lhs, rhs);
}

// NotEquals must check everything that matched
template <>
inline bool ValuePositionComparator::Possible<duckdb::NotEquals>(const Value &lhs, const Value &rhs) {
	return true;
}

template <>
inline bool ValuePositionComparator::Final<duckdb::NotEquals>(const Value &lhs, const Value &rhs) {
	return ValueOperations::NotDistinctFrom(lhs, rhs);
}

// Non-strict inequalities must use strict comparisons for Definite
template <>
bool ValuePositionComparator::Definite<duckdb::LessThanEquals>(const Value &lhs, const Value &rhs) {
	return ValueOperations::DistinctLessThan(lhs, rhs);
}

template <>
bool ValuePositionComparator::Final<duckdb::LessThanEquals>(const Value &lhs, const Value &rhs) {
	return ValueOperations::DistinctLessThanEquals(lhs, rhs);
}

template <>
bool ValuePositionComparator::Definite<duckdb::GreaterThanEquals>(const Value &lhs, const Value &rhs) {
	return ValueOperations::DistinctGreaterThan(lhs, rhs);
}

template <>
bool ValuePositionComparator::Final<duckdb::GreaterThanEquals>(const Value &lhs, const Value &rhs) {
	return ValueOperations::DistinctGreaterThanEquals(lhs, rhs);
}

// Strict inequalities just use strict for both Definite and Final
template <>
bool ValuePositionComparator::Final<duckdb::LessThan>(const Value &lhs, const Value &rhs) {
	return ValueOperations::DistinctLessThan(lhs, rhs);
}

template <>
bool ValuePositionComparator::Final<duckdb::GreaterThan>(const Value &lhs, const Value &rhs) {
	return ValueOperations::DistinctGreaterThan(lhs, rhs);
}

template <class OP>
static bool TemplatedBooleanOperation(const Value &left, const Value &right) {
	const auto &left_type = left.type();
	const auto &right_type = right.type();
	if (left_type != right_type) {
		Value left_copy = left;
		Value right_copy = right;

		LogicalType comparison_type = BoundComparisonExpression::BindComparison(left_type, right_type);
		if (!left_copy.DefaultTryCastAs(comparison_type) || !right_copy.DefaultTryCastAs(comparison_type)) {
			return false;
		}
		D_ASSERT(left_copy.type() == right_copy.type());
		return TemplatedBooleanOperation<OP>(left_copy, right_copy);
	}
	switch (left_type.InternalType()) {
	case PhysicalType::BOOL:
		return OP::Operation(left.GetValueUnsafe<bool>(), right.GetValueUnsafe<bool>());
	case PhysicalType::INT8:
		return OP::Operation(left.GetValueUnsafe<int8_t>(), right.GetValueUnsafe<int8_t>());
	case PhysicalType::INT16:
		return OP::Operation(left.GetValueUnsafe<int16_t>(), right.GetValueUnsafe<int16_t>());
	case PhysicalType::INT32:
		return OP::Operation(left.GetValueUnsafe<int32_t>(), right.GetValueUnsafe<int32_t>());
	case PhysicalType::INT64:
		return OP::Operation(left.GetValueUnsafe<int64_t>(), right.GetValueUnsafe<int64_t>());
	case PhysicalType::UINT8:
		return OP::Operation(left.GetValueUnsafe<uint8_t>(), right.GetValueUnsafe<uint8_t>());
	case PhysicalType::UINT16:
		return OP::Operation(left.GetValueUnsafe<uint16_t>(), right.GetValueUnsafe<uint16_t>());
	case PhysicalType::UINT32:
		return OP::Operation(left.GetValueUnsafe<uint32_t>(), right.GetValueUnsafe<uint32_t>());
	case PhysicalType::UINT64:
		return OP::Operation(left.GetValueUnsafe<uint64_t>(), right.GetValueUnsafe<uint64_t>());
	case PhysicalType::INT128:
		return OP::Operation(left.GetValueUnsafe<hugeint_t>(), right.GetValueUnsafe<hugeint_t>());
	case PhysicalType::FLOAT:
		return OP::Operation(left.GetValueUnsafe<float>(), right.GetValueUnsafe<float>());
	case PhysicalType::DOUBLE:
		return OP::Operation(left.GetValueUnsafe<double>(), right.GetValueUnsafe<double>());
	case PhysicalType::INTERVAL:
		return OP::Operation(left.GetValueUnsafe<interval_t>(), right.GetValueUnsafe<interval_t>());
	case PhysicalType::VARCHAR:
		return OP::Operation(StringValue::Get(left), StringValue::Get(right));
	case PhysicalType::STRUCT: {
		auto &left_children = StructValue::GetChildren(left);
		auto &right_children = StructValue::GetChildren(right);
		// this should be enforced by the type
		D_ASSERT(left_children.size() == right_children.size());
		idx_t i = 0;
		for (; i < left_children.size() - 1; ++i) {
			if (ValuePositionComparator::Definite<OP>(left_children[i], right_children[i])) {
				return true;
			}
			if (!ValuePositionComparator::Possible<OP>(left_children[i], right_children[i])) {
				return false;
			}
		}
		return ValuePositionComparator::Final<OP>(left_children[i], right_children[i]);
	}
	case PhysicalType::LIST: {
		auto &left_children = ListValue::GetChildren(left);
		auto &right_children = ListValue::GetChildren(right);
		for (idx_t pos = 0;; ++pos) {
			if (pos == left_children.size() || pos == right_children.size()) {
				return ValuePositionComparator::TieBreak<OP>(left_children.size(), right_children.size());
			}
			if (ValuePositionComparator::Definite<OP>(left_children[pos], right_children[pos])) {
				return true;
			}
			if (!ValuePositionComparator::Possible<OP>(left_children[pos], right_children[pos])) {
				return false;
			}
		}
		return false;
	}
	default:
		throw InternalException("Unimplemented type for value comparison");
	}
}

bool ValueOperations::Equals(const Value &left, const Value &right) {
	if (left.IsNull() || right.IsNull()) {
		throw InternalException("Comparison on NULL values");
	}
	return TemplatedBooleanOperation<duckdb::Equals>(left, right);
}

bool ValueOperations::NotEquals(const Value &left, const Value &right) {
	return !ValueOperations::Equals(left, right);
}

bool ValueOperations::GreaterThan(const Value &left, const Value &right) {
	if (left.IsNull() || right.IsNull()) {
		throw InternalException("Comparison on NULL values");
	}
	return TemplatedBooleanOperation<duckdb::GreaterThan>(left, right);
}

bool ValueOperations::GreaterThanEquals(const Value &left, const Value &right) {
	if (left.IsNull() || right.IsNull()) {
		throw InternalException("Comparison on NULL values");
	}
	return TemplatedBooleanOperation<duckdb::GreaterThanEquals>(left, right);
}

bool ValueOperations::LessThan(const Value &left, const Value &right) {
	return ValueOperations::GreaterThan(right, left);
}

bool ValueOperations::LessThanEquals(const Value &left, const Value &right) {
	return ValueOperations::GreaterThanEquals(right, left);
}

bool ValueOperations::NotDistinctFrom(const Value &left, const Value &right) {
	if (left.IsNull() && right.IsNull()) {
		return true;
	}
	if (left.IsNull() != right.IsNull()) {
		return false;
	}
	return TemplatedBooleanOperation<duckdb::Equals>(left, right);
}

bool ValueOperations::DistinctFrom(const Value &left, const Value &right) {
	return !ValueOperations::NotDistinctFrom(left, right);
}

bool ValueOperations::DistinctGreaterThan(const Value &left, const Value &right) {
	if (left.IsNull() && right.IsNull()) {
		return false;
	} else if (right.IsNull()) {
		return false;
	} else if (left.IsNull()) {
		return true;
	}
	return TemplatedBooleanOperation<duckdb::GreaterThan>(left, right);
}

bool ValueOperations::DistinctGreaterThanEquals(const Value &left, const Value &right) {
	if (left.IsNull()) {
		return true;
	} else if (right.IsNull()) {
		return false;
	}
	return TemplatedBooleanOperation<duckdb::GreaterThanEquals>(left, right);
}

bool ValueOperations::DistinctLessThan(const Value &left, const Value &right) {
	return ValueOperations::DistinctGreaterThan(right, left);
}

bool ValueOperations::DistinctLessThanEquals(const Value &left, const Value &right) {
	return ValueOperations::DistinctGreaterThanEquals(right, left);
}

} // namespace duckdb
//===--------------------------------------------------------------------===//
// boolean_operators.cpp
// Description: This file contains the implementation of the boolean
// operations AND OR !
//===--------------------------------------------------------------------===//





namespace duckdb {

//===--------------------------------------------------------------------===//
// AND/OR
//===--------------------------------------------------------------------===//
template <class OP>
static void TemplatedBooleanNullmask(Vector &left, Vector &right, Vector &result, idx_t count) {
	D_ASSERT(left.GetType().id() == LogicalTypeId::BOOLEAN && right.GetType().id() == LogicalTypeId::BOOLEAN &&
	         result.GetType().id() == LogicalTypeId::BOOLEAN);

	if (left.GetVectorType() == VectorType::CONSTANT_VECTOR && right.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		// operation on two constants, result is constant vector
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
		auto ldata = ConstantVector::GetData<uint8_t>(left);
		auto rdata = ConstantVector::GetData<uint8_t>(right);
		auto result_data = ConstantVector::GetData<bool>(result);

		bool is_null = OP::Operation(*ldata > 0, *rdata > 0, ConstantVector::IsNull(left),
		                             ConstantVector::IsNull(right), *result_data);
		ConstantVector::SetNull(result, is_null);
	} else {
		// perform generic loop
		UnifiedVectorFormat ldata, rdata;
		left.ToUnifiedFormat(count, ldata);
		right.ToUnifiedFormat(count, rdata);

		result.SetVectorType(VectorType::FLAT_VECTOR);
		auto left_data = (uint8_t *)ldata.data; // we use uint8 to avoid load of gunk bools
		auto right_data = (uint8_t *)rdata.data;
		auto result_data = FlatVector::GetData<bool>(result);
		auto &result_mask = FlatVector::Validity(result);
		if (!ldata.validity.AllValid() || !rdata.validity.AllValid()) {
			for (idx_t i = 0; i < count; i++) {
				auto lidx = ldata.sel->get_index(i);
				auto ridx = rdata.sel->get_index(i);
				bool is_null =
				    OP::Operation(left_data[lidx] > 0, right_data[ridx] > 0, !ldata.validity.RowIsValid(lidx),
				                  !rdata.validity.RowIsValid(ridx), result_data[i]);
				result_mask.Set(i, !is_null);
			}
		} else {
			for (idx_t i = 0; i < count; i++) {
				auto lidx = ldata.sel->get_index(i);
				auto ridx = rdata.sel->get_index(i);
				result_data[i] = OP::SimpleOperation(left_data[lidx], right_data[ridx]);
			}
		}
	}
}

/*
SQL AND Rules:

TRUE  AND TRUE   = TRUE
TRUE  AND FALSE  = FALSE
TRUE  AND NULL   = NULL
FALSE AND TRUE   = FALSE
FALSE AND FALSE  = FALSE
FALSE AND NULL   = FALSE
NULL  AND TRUE   = NULL
NULL  AND FALSE  = FALSE
NULL  AND NULL   = NULL

Basically:
- Only true if both are true
- False if either is false (regardless of NULLs)
- NULL otherwise
*/
struct TernaryAnd {
	static bool SimpleOperation(bool left, bool right) {
		return left && right;
	}
	static bool Operation(bool left, bool right, bool left_null, bool right_null, bool &result) {
		if (left_null && right_null) {
			// both NULL:
			// result is NULL
			return true;
		} else if (left_null) {
			// left is NULL:
			// result is FALSE if right is false
			// result is NULL if right is true
			result = right;
			return right;
		} else if (right_null) {
			// right is NULL:
			// result is FALSE if left is false
			// result is NULL if left is true
			result = left;
			return left;
		} else {
			// no NULL: perform the AND
			result = left && right;
			return false;
		}
	}
};

void VectorOperations::And(Vector &left, Vector &right, Vector &result, idx_t count) {
	TemplatedBooleanNullmask<TernaryAnd>(left, right, result, count);
}

/*
SQL OR Rules:

OR
TRUE  OR TRUE  = TRUE
TRUE  OR FALSE = TRUE
TRUE  OR NULL  = TRUE
FALSE OR TRUE  = TRUE
FALSE OR FALSE = FALSE
FALSE OR NULL  = NULL
NULL  OR TRUE  = TRUE
NULL  OR FALSE = NULL
NULL  OR NULL  = NULL

Basically:
- Only false if both are false
- True if either is true (regardless of NULLs)
- NULL otherwise
*/

struct TernaryOr {
	static bool SimpleOperation(bool left, bool right) {
		return left || right;
	}
	static bool Operation(bool left, bool right, bool left_null, bool right_null, bool &result) {
		if (left_null && right_null) {
			// both NULL:
			// result is NULL
			return true;
		} else if (left_null) {
			// left is NULL:
			// result is TRUE if right is true
			// result is NULL if right is false
			result = right;
			return !right;
		} else if (right_null) {
			// right is NULL:
			// result is TRUE if left is true
			// result is NULL if left is false
			result = left;
			return !left;
		} else {
			// no NULL: perform the OR
			result = left || right;
			return false;
		}
	}
};

void VectorOperations::Or(Vector &left, Vector &right, Vector &result, idx_t count) {
	TemplatedBooleanNullmask<TernaryOr>(left, right, result, count);
}

struct NotOperator {
	template <class TA, class TR>
	static inline TR Operation(TA left) {
		return !left;
	}
};

void VectorOperations::Not(Vector &input, Vector &result, idx_t count) {
	D_ASSERT(input.GetType() == LogicalType::BOOLEAN && result.GetType() == LogicalType::BOOLEAN);
	UnaryExecutor::Execute<bool, bool, NotOperator>(input, result, count);
}

} // namespace duckdb
//===--------------------------------------------------------------------===//
// comparison_operators.cpp
// Description: This file contains the implementation of the comparison
// operations == != >= <= > <
//===--------------------------------------------------------------------===//








namespace duckdb {

template <class T>
bool EqualsFloat(T left, T right) {
	if (DUCKDB_UNLIKELY(Value::IsNan(left) && Value::IsNan(right))) {
		return true;
	}
	return left == right;
}

template <>
bool Equals::Operation(float left, float right) {
	return EqualsFloat<float>(left, right);
}

template <>
bool Equals::Operation(double left, double right) {
	return EqualsFloat<double>(left, right);
}

template <class T>
bool GreaterThanFloat(T left, T right) {
	// handle nans
	// nan is always bigger than everything else
	bool left_is_nan = Value::IsNan(left);
	bool right_is_nan = Value::IsNan(right);
	// if right is nan, there is no number that is bigger than right
	if (DUCKDB_UNLIKELY(right_is_nan)) {
		return false;
	}
	// if left is nan, but right is not, left is always bigger
	if (DUCKDB_UNLIKELY(left_is_nan)) {
		return true;
	}
	return left > right;
}

template <>
bool GreaterThan::Operation(float left, float right) {
	return GreaterThanFloat<float>(left, right);
}

template <>
bool GreaterThan::Operation(double left, double right) {
	return GreaterThanFloat<double>(left, right);
}

template <class T>
bool GreaterThanEqualsFloat(T left, T right) {
	// handle nans
	// nan is always bigger than everything else
	bool left_is_nan = Value::IsNan(left);
	bool right_is_nan = Value::IsNan(right);
	// if right is nan, there is no bigger number
	// we only return true if left is also nan (in which case the numbers are equal)
	if (DUCKDB_UNLIKELY(right_is_nan)) {
		return left_is_nan;
	}
	// if left is nan, but right is not, left is always bigger
	if (DUCKDB_UNLIKELY(left_is_nan)) {
		return true;
	}
	return left >= right;
}

template <>
bool GreaterThanEquals::Operation(float left, float right) {
	return GreaterThanEqualsFloat<float>(left, right);
}

template <>
bool GreaterThanEquals::Operation(double left, double right) {
	return GreaterThanEqualsFloat<double>(left, right);
}

struct ComparisonSelector {
	template <typename OP>
	static idx_t Select(Vector &left, Vector &right, const SelectionVector *sel, idx_t count, SelectionVector *true_sel,
	                    SelectionVector *false_sel) {
		throw NotImplementedException("Unknown comparison operation!");
	}
};

template <>
inline idx_t ComparisonSelector::Select<duckdb::Equals>(Vector &left, Vector &right, const SelectionVector *sel,
                                                        idx_t count, SelectionVector *true_sel,
                                                        SelectionVector *false_sel) {
	return VectorOperations::Equals(left, right, sel, count, true_sel, false_sel);
}

template <>
inline idx_t ComparisonSelector::Select<duckdb::NotEquals>(Vector &left, Vector &right, const SelectionVector *sel,
                                                           idx_t count, SelectionVector *true_sel,
                                                           SelectionVector *false_sel) {
	return VectorOperations::NotEquals(left, right, sel, count, true_sel, false_sel);
}

template <>
inline idx_t ComparisonSelector::Select<duckdb::GreaterThan>(Vector &left, Vector &right, const SelectionVector *sel,
                                                             idx_t count, SelectionVector *true_sel,
                                                             SelectionVector *false_sel) {
	return VectorOperations::GreaterThan(left, right, sel, count, true_sel, false_sel);
}

template <>
inline idx_t ComparisonSelector::Select<duckdb::GreaterThanEquals>(Vector &left, Vector &right,
                                                                   const SelectionVector *sel, idx_t count,
                                                                   SelectionVector *true_sel,
                                                                   SelectionVector *false_sel) {
	return VectorOperations::GreaterThanEquals(left, right, sel, count, true_sel, false_sel);
}

template <>
inline idx_t ComparisonSelector::Select<duckdb::LessThan>(Vector &left, Vector &right, const SelectionVector *sel,
                                                          idx_t count, SelectionVector *true_sel,
                                                          SelectionVector *false_sel) {
	return VectorOperations::LessThan(left, right, sel, count, true_sel, false_sel);
}

template <>
inline idx_t ComparisonSelector::Select<duckdb::LessThanEquals>(Vector &left, Vector &right, const SelectionVector *sel,
                                                                idx_t count, SelectionVector *true_sel,
                                                                SelectionVector *false_sel) {
	return VectorOperations::LessThanEquals(left, right, sel, count, true_sel, false_sel);
}

static void ComparesNotNull(UnifiedVectorFormat &ldata, UnifiedVectorFormat &rdata, ValidityMask &vresult,
                            idx_t count) {
	for (idx_t i = 0; i < count; ++i) {
		auto lidx = ldata.sel->get_index(i);
		auto ridx = rdata.sel->get_index(i);
		if (!ldata.validity.RowIsValid(lidx) || !rdata.validity.RowIsValid(ridx)) {
			vresult.SetInvalid(i);
		}
	}
}

template <typename OP>
static void NestedComparisonExecutor(Vector &left, Vector &right, Vector &result, idx_t count) {
	const auto left_constant = left.GetVectorType() == VectorType::CONSTANT_VECTOR;
	const auto right_constant = right.GetVectorType() == VectorType::CONSTANT_VECTOR;

	if ((left_constant && ConstantVector::IsNull(left)) || (right_constant && ConstantVector::IsNull(right))) {
		// either left or right is constant NULL: result is constant NULL
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
		ConstantVector::SetNull(result, true);
		return;
	}

	if (left_constant && right_constant) {
		// both sides are constant, and neither is NULL so just compare one element.
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
		SelectionVector true_sel(1);
		auto match_count = ComparisonSelector::Select<OP>(left, right, nullptr, 1, &true_sel, nullptr);
		auto result_data = ConstantVector::GetData<bool>(result);
		result_data[0] = match_count > 0;
		return;
	}

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_data = FlatVector::GetData<bool>(result);
	auto &result_validity = FlatVector::Validity(result);

	UnifiedVectorFormat leftv, rightv;
	left.ToUnifiedFormat(count, leftv);
	right.ToUnifiedFormat(count, rightv);
	if (!leftv.validity.AllValid() || !rightv.validity.AllValid()) {
		ComparesNotNull(leftv, rightv, result_validity, count);
	}
	SelectionVector true_sel(count);
	SelectionVector false_sel(count);
	idx_t match_count = ComparisonSelector::Select<OP>(left, right, nullptr, count, &true_sel, &false_sel);

	for (idx_t i = 0; i < match_count; ++i) {
		const auto idx = true_sel.get_index(i);
		result_data[idx] = true;
	}

	const idx_t no_match_count = count - match_count;
	for (idx_t i = 0; i < no_match_count; ++i) {
		const auto idx = false_sel.get_index(i);
		result_data[idx] = false;
	}
}

struct ComparisonExecutor {
private:
	template <class T, class OP>
	static inline void TemplatedExecute(Vector &left, Vector &right, Vector &result, idx_t count) {
		BinaryExecutor::Execute<T, T, bool, OP>(left, right, result, count);
	}

public:
	template <class OP>
	static inline void Execute(Vector &left, Vector &right, Vector &result, idx_t count) {
		D_ASSERT(left.GetType() == right.GetType() && result.GetType() == LogicalType::BOOLEAN);
		// the inplace loops take the result as the last parameter
		switch (left.GetType().InternalType()) {
		case PhysicalType::BOOL:
		case PhysicalType::INT8:
			TemplatedExecute<int8_t, OP>(left, right, result, count);
			break;
		case PhysicalType::INT16:
			TemplatedExecute<int16_t, OP>(left, right, result, count);
			break;
		case PhysicalType::INT32:
			TemplatedExecute<int32_t, OP>(left, right, result, count);
			break;
		case PhysicalType::INT64:
			TemplatedExecute<int64_t, OP>(left, right, result, count);
			break;
		case PhysicalType::UINT8:
			TemplatedExecute<uint8_t, OP>(left, right, result, count);
			break;
		case PhysicalType::UINT16:
			TemplatedExecute<uint16_t, OP>(left, right, result, count);
			break;
		case PhysicalType::UINT32:
			TemplatedExecute<uint32_t, OP>(left, right, result, count);
			break;
		case PhysicalType::UINT64:
			TemplatedExecute<uint64_t, OP>(left, right, result, count);
			break;
		case PhysicalType::INT128:
			TemplatedExecute<hugeint_t, OP>(left, right, result, count);
			break;
		case PhysicalType::FLOAT:
			TemplatedExecute<float, OP>(left, right, result, count);
			break;
		case PhysicalType::DOUBLE:
			TemplatedExecute<double, OP>(left, right, result, count);
			break;
		case PhysicalType::INTERVAL:
			TemplatedExecute<interval_t, OP>(left, right, result, count);
			break;
		case PhysicalType::VARCHAR:
			TemplatedExecute<string_t, OP>(left, right, result, count);
			break;
		case PhysicalType::LIST:
		case PhysicalType::MAP:
		case PhysicalType::STRUCT:
			NestedComparisonExecutor<OP>(left, right, result, count);
			break;
		default:
			throw InternalException("Invalid type for comparison");
		}
	}
};

void VectorOperations::Equals(Vector &left, Vector &right, Vector &result, idx_t count) {
	ComparisonExecutor::Execute<duckdb::Equals>(left, right, result, count);
}

void VectorOperations::NotEquals(Vector &left, Vector &right, Vector &result, idx_t count) {
	ComparisonExecutor::Execute<duckdb::NotEquals>(left, right, result, count);
}

void VectorOperations::GreaterThanEquals(Vector &left, Vector &right, Vector &result, idx_t count) {
	ComparisonExecutor::Execute<duckdb::GreaterThanEquals>(left, right, result, count);
}

void VectorOperations::LessThanEquals(Vector &left, Vector &right, Vector &result, idx_t count) {
	ComparisonExecutor::Execute<duckdb::LessThanEquals>(left, right, result, count);
}

void VectorOperations::GreaterThan(Vector &left, Vector &right, Vector &result, idx_t count) {
	ComparisonExecutor::Execute<duckdb::GreaterThan>(left, right, result, count);
}

void VectorOperations::LessThan(Vector &left, Vector &right, Vector &result, idx_t count) {
	ComparisonExecutor::Execute<duckdb::LessThan>(left, right, result, count);
}

} // namespace duckdb
//===--------------------------------------------------------------------===//
// generators.cpp
// Description: This file contains the implementation of different generators
//===--------------------------------------------------------------------===//





namespace duckdb {

template <class T>
void TemplatedGenerateSequence(Vector &result, idx_t count, int64_t start, int64_t increment) {
	D_ASSERT(result.GetType().IsNumeric());
	if (start > NumericLimits<T>::Maximum() || increment > NumericLimits<T>::Maximum()) {
		throw Exception("Sequence start or increment out of type range");
	}
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_data = FlatVector::GetData<T>(result);
	auto value = (T)start;
	for (idx_t i = 0; i < count; i++) {
		if (i > 0) {
			value += increment;
		}
		result_data[i] = value;
	}
}

void VectorOperations::GenerateSequence(Vector &result, idx_t count, int64_t start, int64_t increment) {
	if (!result.GetType().IsNumeric()) {
		throw InvalidTypeException(result.GetType(), "Can only generate sequences for numeric values!");
	}
	switch (result.GetType().InternalType()) {
	case PhysicalType::INT8:
		TemplatedGenerateSequence<int8_t>(result, count, start, increment);
		break;
	case PhysicalType::INT16:
		TemplatedGenerateSequence<int16_t>(result, count, start, increment);
		break;
	case PhysicalType::INT32:
		TemplatedGenerateSequence<int32_t>(result, count, start, increment);
		break;
	case PhysicalType::INT64:
		TemplatedGenerateSequence<int64_t>(result, count, start, increment);
		break;
	case PhysicalType::FLOAT:
		TemplatedGenerateSequence<float>(result, count, start, increment);
		break;
	case PhysicalType::DOUBLE:
		TemplatedGenerateSequence<double>(result, count, start, increment);
		break;
	default:
		throw NotImplementedException("Unimplemented type for generate sequence");
	}
}

template <class T>
void TemplatedGenerateSequence(Vector &result, idx_t count, const SelectionVector &sel, int64_t start,
                               int64_t increment) {
	D_ASSERT(result.GetType().IsNumeric());
	if (start > NumericLimits<T>::Maximum() || increment > NumericLimits<T>::Maximum()) {
		throw Exception("Sequence start or increment out of type range");
	}
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_data = FlatVector::GetData<T>(result);
	auto value = (T)start;
	for (idx_t i = 0; i < count; i++) {
		auto idx = sel.get_index(i);
		result_data[idx] = value + increment * idx;
	}
}

void VectorOperations::GenerateSequence(Vector &result, idx_t count, const SelectionVector &sel, int64_t start,
                                        int64_t increment) {
	if (!result.GetType().IsNumeric()) {
		throw InvalidTypeException(result.GetType(), "Can only generate sequences for numeric values!");
	}
	switch (result.GetType().InternalType()) {
	case PhysicalType::INT8:
		TemplatedGenerateSequence<int8_t>(result, count, sel, start, increment);
		break;
	case PhysicalType::INT16:
		TemplatedGenerateSequence<int16_t>(result, count, sel, start, increment);
		break;
	case PhysicalType::INT32:
		TemplatedGenerateSequence<int32_t>(result, count, sel, start, increment);
		break;
	case PhysicalType::INT64:
		TemplatedGenerateSequence<int64_t>(result, count, sel, start, increment);
		break;
	case PhysicalType::FLOAT:
		TemplatedGenerateSequence<float>(result, count, sel, start, increment);
		break;
	case PhysicalType::DOUBLE:
		TemplatedGenerateSequence<double>(result, count, sel, start, increment);
		break;
	default:
		throw NotImplementedException("Unimplemented type for generate sequence");
	}
}

} // namespace duckdb



namespace duckdb {

struct DistinctBinaryLambdaWrapper {
	template <class OP, class LEFT_TYPE, class RIGHT_TYPE, class RESULT_TYPE>
	static inline RESULT_TYPE Operation(LEFT_TYPE left, RIGHT_TYPE right, bool is_left_null, bool is_right_null) {
		return OP::template Operation<LEFT_TYPE>(left, right, is_left_null, is_right_null);
	}
};

template <class LEFT_TYPE, class RIGHT_TYPE, class RESULT_TYPE, class OP>
static void DistinctExecuteGenericLoop(LEFT_TYPE *__restrict ldata, RIGHT_TYPE *__restrict rdata,
                                       RESULT_TYPE *__restrict result_data, const SelectionVector *__restrict lsel,
                                       const SelectionVector *__restrict rsel, idx_t count, ValidityMask &lmask,
                                       ValidityMask &rmask, ValidityMask &result_mask) {
	for (idx_t i = 0; i < count; i++) {
		auto lindex = lsel->get_index(i);
		auto rindex = rsel->get_index(i);
		auto lentry = ldata[lindex];
		auto rentry = rdata[rindex];
		result_data[i] =
		    OP::template Operation<LEFT_TYPE>(lentry, rentry, !lmask.RowIsValid(lindex), !rmask.RowIsValid(rindex));
	}
}

template <class LEFT_TYPE, class RIGHT_TYPE, class RESULT_TYPE, class OP>
static void DistinctExecuteConstant(Vector &left, Vector &right, Vector &result) {
	result.SetVectorType(VectorType::CONSTANT_VECTOR);

	auto ldata = ConstantVector::GetData<LEFT_TYPE>(left);
	auto rdata = ConstantVector::GetData<RIGHT_TYPE>(right);
	auto result_data = ConstantVector::GetData<RESULT_TYPE>(result);
	*result_data =
	    OP::template Operation<LEFT_TYPE>(*ldata, *rdata, ConstantVector::IsNull(left), ConstantVector::IsNull(right));
}

template <class LEFT_TYPE, class RIGHT_TYPE, class RESULT_TYPE, class OP>
static void DistinctExecuteGeneric(Vector &left, Vector &right, Vector &result, idx_t count) {
	if (left.GetVectorType() == VectorType::CONSTANT_VECTOR && right.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		DistinctExecuteConstant<LEFT_TYPE, RIGHT_TYPE, RESULT_TYPE, OP>(left, right, result);
	} else {
		UnifiedVectorFormat ldata, rdata;

		left.ToUnifiedFormat(count, ldata);
		right.ToUnifiedFormat(count, rdata);

		result.SetVectorType(VectorType::FLAT_VECTOR);
		auto result_data = FlatVector::GetData<RESULT_TYPE>(result);
		DistinctExecuteGenericLoop<LEFT_TYPE, RIGHT_TYPE, RESULT_TYPE, OP>(
		    (LEFT_TYPE *)ldata.data, (RIGHT_TYPE *)rdata.data, result_data, ldata.sel, rdata.sel, count, ldata.validity,
		    rdata.validity, FlatVector::Validity(result));
	}
}

template <class LEFT_TYPE, class RIGHT_TYPE, class RESULT_TYPE, class OP>
static void DistinctExecuteSwitch(Vector &left, Vector &right, Vector &result, idx_t count) {
	DistinctExecuteGeneric<LEFT_TYPE, RIGHT_TYPE, RESULT_TYPE, OP>(left, right, result, count);
}

template <class LEFT_TYPE, class RIGHT_TYPE, class RESULT_TYPE, class OP>
static void DistinctExecute(Vector &left, Vector &right, Vector &result, idx_t count) {
	DistinctExecuteSwitch<LEFT_TYPE, RIGHT_TYPE, RESULT_TYPE, OP>(left, right, result, count);
}

template <class LEFT_TYPE, class RIGHT_TYPE, class OP, bool NO_NULL, bool HAS_TRUE_SEL, bool HAS_FALSE_SEL>
static inline idx_t
DistinctSelectGenericLoop(LEFT_TYPE *__restrict ldata, RIGHT_TYPE *__restrict rdata,
                          const SelectionVector *__restrict lsel, const SelectionVector *__restrict rsel,
                          const SelectionVector *__restrict result_sel, idx_t count, ValidityMask &lmask,
                          ValidityMask &rmask, SelectionVector *true_sel, SelectionVector *false_sel) {
	idx_t true_count = 0, false_count = 0;
	for (idx_t i = 0; i < count; i++) {
		auto result_idx = result_sel->get_index(i);
		auto lindex = lsel->get_index(i);
		auto rindex = rsel->get_index(i);
		if (NO_NULL) {
			if (OP::Operation(ldata[lindex], rdata[rindex], false, false)) {
				if (HAS_TRUE_SEL) {
					true_sel->set_index(true_count++, result_idx);
				}
			} else {
				if (HAS_FALSE_SEL) {
					false_sel->set_index(false_count++, result_idx);
				}
			}
		} else {
			if (OP::Operation(ldata[lindex], rdata[rindex], !lmask.RowIsValid(lindex), !rmask.RowIsValid(rindex))) {
				if (HAS_TRUE_SEL) {
					true_sel->set_index(true_count++, result_idx);
				}
			} else {
				if (HAS_FALSE_SEL) {
					false_sel->set_index(false_count++, result_idx);
				}
			}
		}
	}
	if (HAS_TRUE_SEL) {
		return true_count;
	} else {
		return count - false_count;
	}
}
template <class LEFT_TYPE, class RIGHT_TYPE, class OP, bool NO_NULL>
static inline idx_t
DistinctSelectGenericLoopSelSwitch(LEFT_TYPE *__restrict ldata, RIGHT_TYPE *__restrict rdata,
                                   const SelectionVector *__restrict lsel, const SelectionVector *__restrict rsel,
                                   const SelectionVector *__restrict result_sel, idx_t count, ValidityMask &lmask,
                                   ValidityMask &rmask, SelectionVector *true_sel, SelectionVector *false_sel) {
	if (true_sel && false_sel) {
		return DistinctSelectGenericLoop<LEFT_TYPE, RIGHT_TYPE, OP, NO_NULL, true, true>(
		    ldata, rdata, lsel, rsel, result_sel, count, lmask, rmask, true_sel, false_sel);
	} else if (true_sel) {
		return DistinctSelectGenericLoop<LEFT_TYPE, RIGHT_TYPE, OP, NO_NULL, true, false>(
		    ldata, rdata, lsel, rsel, result_sel, count, lmask, rmask, true_sel, false_sel);
	} else {
		D_ASSERT(false_sel);
		return DistinctSelectGenericLoop<LEFT_TYPE, RIGHT_TYPE, OP, NO_NULL, false, true>(
		    ldata, rdata, lsel, rsel, result_sel, count, lmask, rmask, true_sel, false_sel);
	}
}

template <class LEFT_TYPE, class RIGHT_TYPE, class OP>
static inline idx_t
DistinctSelectGenericLoopSwitch(LEFT_TYPE *__restrict ldata, RIGHT_TYPE *__restrict rdata,
                                const SelectionVector *__restrict lsel, const SelectionVector *__restrict rsel,
                                const SelectionVector *__restrict result_sel, idx_t count, ValidityMask &lmask,
                                ValidityMask &rmask, SelectionVector *true_sel, SelectionVector *false_sel) {
	if (!lmask.AllValid() || !rmask.AllValid()) {
		return DistinctSelectGenericLoopSelSwitch<LEFT_TYPE, RIGHT_TYPE, OP, false>(
		    ldata, rdata, lsel, rsel, result_sel, count, lmask, rmask, true_sel, false_sel);
	} else {
		return DistinctSelectGenericLoopSelSwitch<LEFT_TYPE, RIGHT_TYPE, OP, true>(
		    ldata, rdata, lsel, rsel, result_sel, count, lmask, rmask, true_sel, false_sel);
	}
}

template <class LEFT_TYPE, class RIGHT_TYPE, class OP>
static idx_t DistinctSelectGeneric(Vector &left, Vector &right, const SelectionVector *sel, idx_t count,
                                   SelectionVector *true_sel, SelectionVector *false_sel) {
	UnifiedVectorFormat ldata, rdata;

	left.ToUnifiedFormat(count, ldata);
	right.ToUnifiedFormat(count, rdata);

	return DistinctSelectGenericLoopSwitch<LEFT_TYPE, RIGHT_TYPE, OP>((LEFT_TYPE *)ldata.data, (RIGHT_TYPE *)rdata.data,
	                                                                  ldata.sel, rdata.sel, sel, count, ldata.validity,
	                                                                  rdata.validity, true_sel, false_sel);
}
template <class LEFT_TYPE, class RIGHT_TYPE, class OP, bool LEFT_CONSTANT, bool RIGHT_CONSTANT, bool NO_NULL,
          bool HAS_TRUE_SEL, bool HAS_FALSE_SEL>
static inline idx_t DistinctSelectFlatLoop(LEFT_TYPE *__restrict ldata, RIGHT_TYPE *__restrict rdata,
                                           const SelectionVector *sel, idx_t count, ValidityMask &lmask,
                                           ValidityMask &rmask, SelectionVector *true_sel, SelectionVector *false_sel) {
	idx_t true_count = 0, false_count = 0;
	for (idx_t i = 0; i < count; i++) {
		idx_t result_idx = sel->get_index(i);
		idx_t lidx = LEFT_CONSTANT ? 0 : i;
		idx_t ridx = RIGHT_CONSTANT ? 0 : i;
		const bool lnull = !lmask.RowIsValid(lidx);
		const bool rnull = !rmask.RowIsValid(ridx);
		bool comparison_result = OP::Operation(ldata[lidx], rdata[ridx], lnull, rnull);
		if (HAS_TRUE_SEL) {
			true_sel->set_index(true_count, result_idx);
			true_count += comparison_result;
		}
		if (HAS_FALSE_SEL) {
			false_sel->set_index(false_count, result_idx);
			false_count += !comparison_result;
		}
	}
	if (HAS_TRUE_SEL) {
		return true_count;
	} else {
		return count - false_count;
	}
}

template <class LEFT_TYPE, class RIGHT_TYPE, class OP, bool LEFT_CONSTANT, bool RIGHT_CONSTANT, bool NO_NULL>
static inline idx_t DistinctSelectFlatLoopSelSwitch(LEFT_TYPE *__restrict ldata, RIGHT_TYPE *__restrict rdata,
                                                    const SelectionVector *sel, idx_t count, ValidityMask &lmask,
                                                    ValidityMask &rmask, SelectionVector *true_sel,
                                                    SelectionVector *false_sel) {
	if (true_sel && false_sel) {
		return DistinctSelectFlatLoop<LEFT_TYPE, RIGHT_TYPE, OP, LEFT_CONSTANT, RIGHT_CONSTANT, NO_NULL, true, true>(
		    ldata, rdata, sel, count, lmask, rmask, true_sel, false_sel);
	} else if (true_sel) {
		return DistinctSelectFlatLoop<LEFT_TYPE, RIGHT_TYPE, OP, LEFT_CONSTANT, RIGHT_CONSTANT, NO_NULL, true, false>(
		    ldata, rdata, sel, count, lmask, rmask, true_sel, false_sel);
	} else {
		D_ASSERT(false_sel);
		return DistinctSelectFlatLoop<LEFT_TYPE, RIGHT_TYPE, OP, LEFT_CONSTANT, RIGHT_CONSTANT, NO_NULL, false, true>(
		    ldata, rdata, sel, count, lmask, rmask, true_sel, false_sel);
	}
}

template <class LEFT_TYPE, class RIGHT_TYPE, class OP, bool LEFT_CONSTANT, bool RIGHT_CONSTANT>
static inline idx_t DistinctSelectFlatLoopSwitch(LEFT_TYPE *__restrict ldata, RIGHT_TYPE *__restrict rdata,
                                                 const SelectionVector *sel, idx_t count, ValidityMask &lmask,
                                                 ValidityMask &rmask, SelectionVector *true_sel,
                                                 SelectionVector *false_sel) {
	return DistinctSelectFlatLoopSelSwitch<LEFT_TYPE, RIGHT_TYPE, OP, LEFT_CONSTANT, RIGHT_CONSTANT, true>(
	    ldata, rdata, sel, count, lmask, rmask, true_sel, false_sel);
}
template <class LEFT_TYPE, class RIGHT_TYPE, class OP, bool LEFT_CONSTANT, bool RIGHT_CONSTANT>
static idx_t DistinctSelectFlat(Vector &left, Vector &right, const SelectionVector *sel, idx_t count,
                                SelectionVector *true_sel, SelectionVector *false_sel) {
	auto ldata = FlatVector::GetData<LEFT_TYPE>(left);
	auto rdata = FlatVector::GetData<RIGHT_TYPE>(right);
	if (LEFT_CONSTANT) {
		ValidityMask validity;
		if (ConstantVector::IsNull(left)) {
			validity.SetAllInvalid(1);
		}
		return DistinctSelectFlatLoopSwitch<LEFT_TYPE, RIGHT_TYPE, OP, LEFT_CONSTANT, RIGHT_CONSTANT>(
		    ldata, rdata, sel, count, validity, FlatVector::Validity(right), true_sel, false_sel);
	} else if (RIGHT_CONSTANT) {
		ValidityMask validity;
		if (ConstantVector::IsNull(right)) {
			validity.SetAllInvalid(1);
		}
		return DistinctSelectFlatLoopSwitch<LEFT_TYPE, RIGHT_TYPE, OP, LEFT_CONSTANT, RIGHT_CONSTANT>(
		    ldata, rdata, sel, count, FlatVector::Validity(left), validity, true_sel, false_sel);
	} else {
		return DistinctSelectFlatLoopSwitch<LEFT_TYPE, RIGHT_TYPE, OP, LEFT_CONSTANT, RIGHT_CONSTANT>(
		    ldata, rdata, sel, count, FlatVector::Validity(left), FlatVector::Validity(right), true_sel, false_sel);
	}
}
template <class LEFT_TYPE, class RIGHT_TYPE, class OP>
static idx_t DistinctSelectConstant(Vector &left, Vector &right, const SelectionVector *sel, idx_t count,
                                    SelectionVector *true_sel, SelectionVector *false_sel) {
	auto ldata = ConstantVector::GetData<LEFT_TYPE>(left);
	auto rdata = ConstantVector::GetData<RIGHT_TYPE>(right);

	// both sides are constant, return either 0 or the count
	// in this case we do not fill in the result selection vector at all
	if (!OP::Operation(*ldata, *rdata, ConstantVector::IsNull(left), ConstantVector::IsNull(right))) {
		if (false_sel) {
			for (idx_t i = 0; i < count; i++) {
				false_sel->set_index(i, sel->get_index(i));
			}
		}
		return 0;
	} else {
		if (true_sel) {
			for (idx_t i = 0; i < count; i++) {
				true_sel->set_index(i, sel->get_index(i));
			}
		}
		return count;
	}
}

template <class LEFT_TYPE, class RIGHT_TYPE, class OP>
static idx_t DistinctSelect(Vector &left, Vector &right, const SelectionVector *sel, idx_t count,
                            SelectionVector *true_sel, SelectionVector *false_sel) {
	if (!sel) {
		sel = FlatVector::IncrementalSelectionVector();
	}
	if (left.GetVectorType() == VectorType::CONSTANT_VECTOR && right.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		return DistinctSelectConstant<LEFT_TYPE, RIGHT_TYPE, OP>(left, right, sel, count, true_sel, false_sel);
	} else if (left.GetVectorType() == VectorType::CONSTANT_VECTOR &&
	           right.GetVectorType() == VectorType::FLAT_VECTOR) {
		return DistinctSelectFlat<LEFT_TYPE, RIGHT_TYPE, OP, true, false>(left, right, sel, count, true_sel, false_sel);
	} else if (left.GetVectorType() == VectorType::FLAT_VECTOR &&
	           right.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		return DistinctSelectFlat<LEFT_TYPE, RIGHT_TYPE, OP, false, true>(left, right, sel, count, true_sel, false_sel);
	} else if (left.GetVectorType() == VectorType::FLAT_VECTOR && right.GetVectorType() == VectorType::FLAT_VECTOR) {
		return DistinctSelectFlat<LEFT_TYPE, RIGHT_TYPE, OP, false, false>(left, right, sel, count, true_sel,
		                                                                   false_sel);
	} else {
		return DistinctSelectGeneric<LEFT_TYPE, RIGHT_TYPE, OP>(left, right, sel, count, true_sel, false_sel);
	}
}

template <class OP>
static idx_t DistinctSelectNotNull(Vector &left, Vector &right, const idx_t count, idx_t &true_count,
                                   const SelectionVector &sel, SelectionVector &maybe_vec, OptionalSelection &true_opt,
                                   OptionalSelection &false_opt) {
	UnifiedVectorFormat lvdata, rvdata;
	left.ToUnifiedFormat(count, lvdata);
	right.ToUnifiedFormat(count, rvdata);

	auto &lmask = lvdata.validity;
	auto &rmask = rvdata.validity;

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
	true_count = 0;
	idx_t false_count = 0;
	for (idx_t i = 0; i < count; ++i) {
		const auto result_idx = sel.get_index(i);
		const auto lidx = lvdata.sel->get_index(i);
		const auto ridx = rvdata.sel->get_index(i);
		const auto lnull = !lmask.RowIsValid(lidx);
		const auto rnull = !rmask.RowIsValid(ridx);
		if (lnull || rnull) {
			// If either is NULL then we can major distinguish them
			if (!OP::Operation(false, false, lnull, rnull)) {
				false_opt.Append(false_count, result_idx);
			} else {
				true_opt.Append(true_count, result_idx);
			}
		} else {
			//	Neither is NULL, distinguish values.
			slicer.set_index(remaining, i);
			maybe_vec.set_index(remaining++, result_idx);
		}
	}

	true_opt.Advance(true_count);
	false_opt.Advance(false_count);

	if (remaining && remaining < count) {
		left.Slice(slicer, remaining);
		right.Slice(slicer, remaining);
	}

	return remaining;
}

struct PositionComparator {
	// Select the rows that definitely match.
	// Default to the same as the final row
	template <typename OP>
	static idx_t Definite(Vector &left, Vector &right, const SelectionVector &sel, idx_t count,
	                      SelectionVector *true_sel, SelectionVector &false_sel) {
		return Final<OP>(left, right, sel, count, true_sel, &false_sel);
	}

	// Select the possible rows that need further testing.
	// Usually this means Is Not Distinct, as those are the semantics used by Postges
	template <typename OP>
	static idx_t Possible(Vector &left, Vector &right, const SelectionVector &sel, idx_t count,
	                      SelectionVector &true_sel, SelectionVector *false_sel) {
		return VectorOperations::NestedEquals(left, right, sel, count, &true_sel, false_sel);
	}

	// Select the matching rows for the final position.
	// This needs to be specialised.
	template <typename OP>
	static idx_t Final(Vector &left, Vector &right, const SelectionVector &sel, idx_t count, SelectionVector *true_sel,
	                   SelectionVector *false_sel) {
		return 0;
	}

	// Tie-break based on length when one of the sides has been exhausted, returning true if the LHS matches.
	// This essentially means that the existing positions compare equal.
	// Default to the same semantics as the OP for idx_t. This works in most cases.
	template <typename OP>
	static bool TieBreak(const idx_t lpos, const idx_t rpos) {
		return OP::Operation(lpos, rpos, false, false);
	}
};

// NotDistinctFrom must always check every column
template <>
idx_t PositionComparator::Definite<duckdb::NotDistinctFrom>(Vector &left, Vector &right, const SelectionVector &sel,
                                                            idx_t count, SelectionVector *true_sel,
                                                            SelectionVector &false_sel) {
	return 0;
}

template <>
idx_t PositionComparator::Final<duckdb::NotDistinctFrom>(Vector &left, Vector &right, const SelectionVector &sel,
                                                         idx_t count, SelectionVector *true_sel,
                                                         SelectionVector *false_sel) {
	return VectorOperations::NestedEquals(left, right, sel, count, true_sel, false_sel);
}

// DistinctFrom must check everything that matched
template <>
idx_t PositionComparator::Possible<duckdb::DistinctFrom>(Vector &left, Vector &right, const SelectionVector &sel,
                                                         idx_t count, SelectionVector &true_sel,
                                                         SelectionVector *false_sel) {
	return count;
}

template <>
idx_t PositionComparator::Final<duckdb::DistinctFrom>(Vector &left, Vector &right, const SelectionVector &sel,
                                                      idx_t count, SelectionVector *true_sel,
                                                      SelectionVector *false_sel) {
	return VectorOperations::NestedNotEquals(left, right, sel, count, true_sel, false_sel);
}

// Non-strict inequalities must use strict comparisons for Definite
template <>
idx_t PositionComparator::Definite<duckdb::DistinctLessThanEquals>(Vector &left, Vector &right,
                                                                   const SelectionVector &sel, idx_t count,
                                                                   SelectionVector *true_sel,
                                                                   SelectionVector &false_sel) {
	return VectorOperations::DistinctLessThan(left, right, &sel, count, true_sel, &false_sel);
}

template <>
idx_t PositionComparator::Final<duckdb::DistinctLessThanEquals>(Vector &left, Vector &right, const SelectionVector &sel,
                                                                idx_t count, SelectionVector *true_sel,
                                                                SelectionVector *false_sel) {
	return VectorOperations::DistinctLessThanEquals(left, right, &sel, count, true_sel, false_sel);
}

template <>
idx_t PositionComparator::Definite<duckdb::DistinctGreaterThanEquals>(Vector &left, Vector &right,
                                                                      const SelectionVector &sel, idx_t count,
                                                                      SelectionVector *true_sel,
                                                                      SelectionVector &false_sel) {
	return VectorOperations::DistinctGreaterThan(left, right, &sel, count, true_sel, &false_sel);
}

template <>
idx_t PositionComparator::Final<duckdb::DistinctGreaterThanEquals>(Vector &left, Vector &right,
                                                                   const SelectionVector &sel, idx_t count,
                                                                   SelectionVector *true_sel,
                                                                   SelectionVector *false_sel) {
	return VectorOperations::DistinctGreaterThanEquals(left, right, &sel, count, true_sel, false_sel);
}

// Strict inequalities just use strict for both Definite and Final
template <>
idx_t PositionComparator::Final<duckdb::DistinctLessThan>(Vector &left, Vector &right, const SelectionVector &sel,
                                                          idx_t count, SelectionVector *true_sel,
                                                          SelectionVector *false_sel) {
	return VectorOperations::DistinctLessThan(left, right, &sel, count, true_sel, false_sel);
}

template <>
idx_t PositionComparator::Final<duckdb::DistinctGreaterThan>(Vector &left, Vector &right, const SelectionVector &sel,
                                                             idx_t count, SelectionVector *true_sel,
                                                             SelectionVector *false_sel) {
	return VectorOperations::DistinctGreaterThan(left, right, &sel, count, true_sel, false_sel);
}

using StructEntries = vector<unique_ptr<Vector>>;

static void ExtractNestedSelection(const SelectionVector &slice_sel, const idx_t count, const SelectionVector &sel,
                                   OptionalSelection &opt) {

	for (idx_t i = 0; i < count;) {
		const auto slice_idx = slice_sel.get_index(i);
		const auto result_idx = sel.get_index(slice_idx);
		opt.Append(i, result_idx);
	}
	opt.Advance(count);
}

static void DensifyNestedSelection(const SelectionVector &dense_sel, const idx_t count, SelectionVector &slice_sel) {
	for (idx_t i = 0; i < count; ++i) {
		slice_sel.set_index(i, dense_sel.get_index(i));
	}
}

template <class OP>
static idx_t DistinctSelectStruct(Vector &left, Vector &right, idx_t count, const SelectionVector &sel,
                                  OptionalSelection &true_opt, OptionalSelection &false_opt) {
	if (count == 0) {
		return 0;
	}

	// Avoid allocating in the 99% of the cases where we don't need to.
	StructEntries lsliced, rsliced;
	auto &lchildren = StructVector::GetEntries(left);
	auto &rchildren = StructVector::GetEntries(right);
	D_ASSERT(lchildren.size() == rchildren.size());

	// In order to reuse the comparators, we have to track what passed and failed internally.
	// To do that, we need local SVs that we then merge back into the real ones after every pass.
	const auto vcount = count;
	SelectionVector slice_sel(count);
	for (idx_t i = 0; i < count; ++i) {
		slice_sel.set_index(i, i);
	}

	SelectionVector true_sel(count);
	SelectionVector false_sel(count);

	idx_t match_count = 0;
	for (idx_t col_no = 0; col_no < lchildren.size(); ++col_no) {
		// Slice the children to maintain density
		Vector lchild(*lchildren[col_no]);
		lchild.Flatten(vcount);
		lchild.Slice(slice_sel, count);

		Vector rchild(*rchildren[col_no]);
		rchild.Flatten(vcount);
		rchild.Slice(slice_sel, count);

		// Find everything that definitely matches
		auto true_count = PositionComparator::Definite<OP>(lchild, rchild, slice_sel, count, &true_sel, false_sel);
		if (true_count > 0) {
			auto false_count = count - true_count;

			// Extract the definite matches into the true result
			ExtractNestedSelection(false_count ? true_sel : slice_sel, true_count, sel, true_opt);

			// Remove the definite matches from the slicing vector
			DensifyNestedSelection(false_sel, false_count, slice_sel);

			match_count += true_count;
			count -= true_count;
		}

		if (col_no != lchildren.size() - 1) {
			// Find what might match on the next position
			true_count = PositionComparator::Possible<OP>(lchild, rchild, slice_sel, count, true_sel, &false_sel);
			auto false_count = count - true_count;

			// Extract the definite failures into the false result
			ExtractNestedSelection(true_count ? false_sel : slice_sel, false_count, sel, false_opt);

			// Remove any definite failures from the slicing vector
			if (false_count) {
				DensifyNestedSelection(true_sel, true_count, slice_sel);
			}

			count = true_count;
		} else {
			true_count = PositionComparator::Final<OP>(lchild, rchild, slice_sel, count, &true_sel, &false_sel);
			auto false_count = count - true_count;

			// Extract the definite matches into the true result
			ExtractNestedSelection(false_count ? true_sel : slice_sel, true_count, sel, true_opt);

			// Extract the definite failures into the false result
			ExtractNestedSelection(true_count ? false_sel : slice_sel, false_count, sel, false_opt);

			match_count += true_count;
		}
	}
	return match_count;
}

static void PositionListCursor(SelectionVector &cursor, UnifiedVectorFormat &vdata, const idx_t pos,
                               const SelectionVector &slice_sel, const idx_t count) {
	const auto data = (const list_entry_t *)vdata.data;
	for (idx_t i = 0; i < count; ++i) {
		const auto slice_idx = slice_sel.get_index(i);

		const auto lidx = vdata.sel->get_index(slice_idx);
		const auto &entry = data[lidx];
		cursor.set_index(i, entry.offset + pos);
	}
}

template <class OP>
static idx_t DistinctSelectList(Vector &left, Vector &right, idx_t count, const SelectionVector &sel,
                                OptionalSelection &true_opt, OptionalSelection &false_opt) {
	if (count == 0) {
		return count;
	}

	// Create dictionary views of the children so we can vectorise the positional comparisons.
	SelectionVector lcursor(count);
	SelectionVector rcursor(count);

	ListVector::GetEntry(left).Flatten(ListVector::GetListSize(left));
	ListVector::GetEntry(right).Flatten(ListVector::GetListSize(right));
	Vector lchild(ListVector::GetEntry(left), lcursor, count);
	Vector rchild(ListVector::GetEntry(right), rcursor, count);

	// To perform the positional comparison, we use a vectorisation of the following algorithm:
	// bool CompareLists(T *left, idx_t nleft, T *right, nright) {
	// 	for (idx_t pos = 0; ; ++pos) {
	// 		if (nleft == pos || nright == pos)
	// 			return OP::TieBreak(nleft, nright);
	// 		if (OP::Definite(*left, *right))
	// 			return true;
	// 		if (!OP::Maybe(*left, *right))
	// 			return false;
	// 		}
	//	 	++left, ++right;
	// 	}
	// }

	// Get pointers to the list entries
	UnifiedVectorFormat lvdata;
	left.ToUnifiedFormat(count, lvdata);
	const auto ldata = (const list_entry_t *)lvdata.data;

	UnifiedVectorFormat rvdata;
	right.ToUnifiedFormat(count, rvdata);
	const auto rdata = (const list_entry_t *)rvdata.data;

	// In order to reuse the comparators, we have to track what passed and failed internally.
	// To do that, we need local SVs that we then merge back into the real ones after every pass.
	SelectionVector slice_sel(count);
	for (idx_t i = 0; i < count; ++i) {
		slice_sel.set_index(i, i);
	}

	SelectionVector true_sel(count);
	SelectionVector false_sel(count);

	idx_t match_count = 0;
	for (idx_t pos = 0; count > 0; ++pos) {
		// Set up the cursors for the current position
		PositionListCursor(lcursor, lvdata, pos, slice_sel, count);
		PositionListCursor(rcursor, rvdata, pos, slice_sel, count);

		// Tie-break the pairs where one of the LISTs is exhausted.
		idx_t true_count = 0;
		idx_t false_count = 0;
		idx_t maybe_count = 0;
		for (idx_t i = 0; i < count; ++i) {
			const auto slice_idx = slice_sel.get_index(i);
			const auto lidx = lvdata.sel->get_index(slice_idx);
			const auto &lentry = ldata[lidx];
			const auto ridx = rvdata.sel->get_index(slice_idx);
			const auto &rentry = rdata[ridx];
			if (lentry.length == pos || rentry.length == pos) {
				const auto idx = sel.get_index(slice_idx);
				if (PositionComparator::TieBreak<OP>(lentry.length, rentry.length)) {
					true_opt.Append(true_count, idx);
				} else {
					false_opt.Append(false_count, idx);
				}
			} else {
				true_sel.set_index(maybe_count++, slice_idx);
			}
		}
		true_opt.Advance(true_count);
		false_opt.Advance(false_count);
		match_count += true_count;

		// Redensify the list cursors
		if (maybe_count < count) {
			count = maybe_count;
			DensifyNestedSelection(true_sel, count, slice_sel);
			PositionListCursor(lcursor, lvdata, pos, slice_sel, count);
			PositionListCursor(rcursor, rvdata, pos, slice_sel, count);
		}

		// Find everything that definitely matches
		true_count = PositionComparator::Definite<OP>(lchild, rchild, slice_sel, count, &true_sel, false_sel);
		if (true_count) {
			false_count = count - true_count;
			ExtractNestedSelection(false_count ? true_sel : slice_sel, true_count, sel, true_opt);
			match_count += true_count;

			// Redensify the list cursors
			count -= true_count;
			DensifyNestedSelection(false_sel, count, slice_sel);
			PositionListCursor(lcursor, lvdata, pos, slice_sel, count);
			PositionListCursor(rcursor, rvdata, pos, slice_sel, count);
		}

		// Find what might match on the next position
		true_count = PositionComparator::Possible<OP>(lchild, rchild, slice_sel, count, true_sel, &false_sel);
		false_count = count - true_count;
		ExtractNestedSelection(true_count ? false_sel : slice_sel, false_count, sel, false_opt);

		if (false_count) {
			DensifyNestedSelection(true_sel, true_count, slice_sel);
		}
		count = true_count;
	}

	return match_count;
}

template <class OP, class OPNESTED>
static idx_t DistinctSelectNested(Vector &left, Vector &right, const SelectionVector *sel, const idx_t count,
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

	idx_t match_count = 0;
	auto unknown =
	    DistinctSelectNotNull<OP>(l_not_null, r_not_null, count, match_count, *sel, maybe_vec, true_opt, false_opt);

	if (PhysicalType::LIST == left.GetType().InternalType()) {
		match_count += DistinctSelectList<OPNESTED>(l_not_null, r_not_null, unknown, maybe_vec, true_opt, false_opt);
	} else {
		match_count += DistinctSelectStruct<OPNESTED>(l_not_null, r_not_null, unknown, maybe_vec, true_opt, false_opt);
	}

	// Copy the buffered selections to the output selections
	if (true_sel) {
		DensifyNestedSelection(true_vec, match_count, *true_sel);
	}

	if (false_sel) {
		DensifyNestedSelection(false_vec, count - match_count, *false_sel);
	}

	return match_count;
}

template <typename OP>
static void NestedDistinctExecute(Vector &left, Vector &right, Vector &result, idx_t count);

template <class T, class OP>
static inline void TemplatedDistinctExecute(Vector &left, Vector &right, Vector &result, idx_t count) {
	DistinctExecute<T, T, bool, OP>(left, right, result, count);
}
template <class OP>
static void ExecuteDistinct(Vector &left, Vector &right, Vector &result, idx_t count) {
	D_ASSERT(left.GetType() == right.GetType() && result.GetType() == LogicalType::BOOLEAN);
	// the inplace loops take the result as the last parameter
	switch (left.GetType().InternalType()) {
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
		TemplatedDistinctExecute<int8_t, OP>(left, right, result, count);
		break;
	case PhysicalType::INT16:
		TemplatedDistinctExecute<int16_t, OP>(left, right, result, count);
		break;
	case PhysicalType::INT32:
		TemplatedDistinctExecute<int32_t, OP>(left, right, result, count);
		break;
	case PhysicalType::INT64:
		TemplatedDistinctExecute<int64_t, OP>(left, right, result, count);
		break;
	case PhysicalType::UINT8:
		TemplatedDistinctExecute<uint8_t, OP>(left, right, result, count);
		break;
	case PhysicalType::UINT16:
		TemplatedDistinctExecute<uint16_t, OP>(left, right, result, count);
		break;
	case PhysicalType::UINT32:
		TemplatedDistinctExecute<uint32_t, OP>(left, right, result, count);
		break;
	case PhysicalType::UINT64:
		TemplatedDistinctExecute<uint64_t, OP>(left, right, result, count);
		break;
	case PhysicalType::INT128:
		TemplatedDistinctExecute<hugeint_t, OP>(left, right, result, count);
		break;
	case PhysicalType::FLOAT:
		TemplatedDistinctExecute<float, OP>(left, right, result, count);
		break;
	case PhysicalType::DOUBLE:
		TemplatedDistinctExecute<double, OP>(left, right, result, count);
		break;
	case PhysicalType::INTERVAL:
		TemplatedDistinctExecute<interval_t, OP>(left, right, result, count);
		break;
	case PhysicalType::VARCHAR:
		TemplatedDistinctExecute<string_t, OP>(left, right, result, count);
		break;
	case PhysicalType::LIST:
	case PhysicalType::MAP:
	case PhysicalType::STRUCT:
		NestedDistinctExecute<OP>(left, right, result, count);
		break;
	default:
		throw InternalException("Invalid type for distinct comparison");
	}
}

template <class OP, class OPNESTED = OP>
static idx_t TemplatedDistinctSelectOperation(Vector &left, Vector &right, const SelectionVector *sel, idx_t count,
                                              SelectionVector *true_sel, SelectionVector *false_sel) {
	// the inplace loops take the result as the last parameter
	switch (left.GetType().InternalType()) {
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
		return DistinctSelect<int8_t, int8_t, OP>(left, right, sel, count, true_sel, false_sel);
	case PhysicalType::INT16:
		return DistinctSelect<int16_t, int16_t, OP>(left, right, sel, count, true_sel, false_sel);
	case PhysicalType::INT32:
		return DistinctSelect<int32_t, int32_t, OP>(left, right, sel, count, true_sel, false_sel);
	case PhysicalType::INT64:
		return DistinctSelect<int64_t, int64_t, OP>(left, right, sel, count, true_sel, false_sel);
	case PhysicalType::UINT8:
		return DistinctSelect<uint8_t, uint8_t, OP>(left, right, sel, count, true_sel, false_sel);
	case PhysicalType::UINT16:
		return DistinctSelect<uint16_t, uint16_t, OP>(left, right, sel, count, true_sel, false_sel);
	case PhysicalType::UINT32:
		return DistinctSelect<uint32_t, uint32_t, OP>(left, right, sel, count, true_sel, false_sel);
	case PhysicalType::UINT64:
		return DistinctSelect<uint64_t, uint64_t, OP>(left, right, sel, count, true_sel, false_sel);
	case PhysicalType::INT128:
		return DistinctSelect<hugeint_t, hugeint_t, OP>(left, right, sel, count, true_sel, false_sel);
	case PhysicalType::FLOAT:
		return DistinctSelect<float, float, OP>(left, right, sel, count, true_sel, false_sel);
	case PhysicalType::DOUBLE:
		return DistinctSelect<double, double, OP>(left, right, sel, count, true_sel, false_sel);
	case PhysicalType::INTERVAL:
		return DistinctSelect<interval_t, interval_t, OP>(left, right, sel, count, true_sel, false_sel);
	case PhysicalType::VARCHAR:
		return DistinctSelect<string_t, string_t, OP>(left, right, sel, count, true_sel, false_sel);
	case PhysicalType::MAP:
	case PhysicalType::STRUCT:
	case PhysicalType::LIST:
		return DistinctSelectNested<OP, OPNESTED>(left, right, sel, count, true_sel, false_sel);
	default:
		throw InternalException("Invalid type for distinct selection");
	}
}

template <typename OP>
static void NestedDistinctExecute(Vector &left, Vector &right, Vector &result, idx_t count) {
	const auto left_constant = left.GetVectorType() == VectorType::CONSTANT_VECTOR;
	const auto right_constant = right.GetVectorType() == VectorType::CONSTANT_VECTOR;

	if (left_constant && right_constant) {
		// both sides are constant, so just compare one element.
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
		auto result_data = ConstantVector::GetData<bool>(result);
		SelectionVector true_sel(1);
		auto match_count = TemplatedDistinctSelectOperation<OP>(left, right, nullptr, 1, &true_sel, nullptr);
		result_data[0] = match_count > 0;
		return;
	}

	SelectionVector true_sel(count);
	SelectionVector false_sel(count);

	// DISTINCT is either true or false
	idx_t match_count = TemplatedDistinctSelectOperation<OP>(left, right, nullptr, count, &true_sel, &false_sel);

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_data = FlatVector::GetData<bool>(result);

	for (idx_t i = 0; i < match_count; ++i) {
		const auto idx = true_sel.get_index(i);
		result_data[idx] = true;
	}

	const idx_t no_match_count = count - match_count;
	for (idx_t i = 0; i < no_match_count; ++i) {
		const auto idx = false_sel.get_index(i);
		result_data[idx] = false;
	}
}

void VectorOperations::DistinctFrom(Vector &left, Vector &right, Vector &result, idx_t count) {
	ExecuteDistinct<duckdb::DistinctFrom>(left, right, result, count);
}

void VectorOperations::NotDistinctFrom(Vector &left, Vector &right, Vector &result, idx_t count) {
	ExecuteDistinct<duckdb::NotDistinctFrom>(left, right, result, count);
}

// true := A != B with nulls being equal
idx_t VectorOperations::DistinctFrom(Vector &left, Vector &right, const SelectionVector *sel, idx_t count,
                                     SelectionVector *true_sel, SelectionVector *false_sel) {
	return TemplatedDistinctSelectOperation<duckdb::DistinctFrom>(left, right, sel, count, true_sel, false_sel);
}
// true := A == B with nulls being equal
idx_t VectorOperations::NotDistinctFrom(Vector &left, Vector &right, const SelectionVector *sel, idx_t count,
                                        SelectionVector *true_sel, SelectionVector *false_sel) {
	return TemplatedDistinctSelectOperation<duckdb::NotDistinctFrom>(left, right, sel, count, true_sel, false_sel);
}

// true := A > B with nulls being maximal
idx_t VectorOperations::DistinctGreaterThan(Vector &left, Vector &right, const SelectionVector *sel, idx_t count,
                                            SelectionVector *true_sel, SelectionVector *false_sel) {
	return TemplatedDistinctSelectOperation<duckdb::DistinctGreaterThan>(left, right, sel, count, true_sel, false_sel);
}

// true := A > B with nulls being minimal
idx_t VectorOperations::DistinctGreaterThanNullsFirst(Vector &left, Vector &right, const SelectionVector *sel,
                                                      idx_t count, SelectionVector *true_sel,
                                                      SelectionVector *false_sel) {
	return TemplatedDistinctSelectOperation<duckdb::DistinctGreaterThanNullsFirst, duckdb::DistinctGreaterThan>(
	    left, right, sel, count, true_sel, false_sel);
}
// true := A >= B with nulls being maximal
idx_t VectorOperations::DistinctGreaterThanEquals(Vector &left, Vector &right, const SelectionVector *sel, idx_t count,
                                                  SelectionVector *true_sel, SelectionVector *false_sel) {
	return TemplatedDistinctSelectOperation<duckdb::DistinctGreaterThanEquals>(left, right, sel, count, true_sel,
	                                                                           false_sel);
}
// true := A < B with nulls being maximal
idx_t VectorOperations::DistinctLessThan(Vector &left, Vector &right, const SelectionVector *sel, idx_t count,
                                         SelectionVector *true_sel, SelectionVector *false_sel) {
	return TemplatedDistinctSelectOperation<duckdb::DistinctLessThan>(left, right, sel, count, true_sel, false_sel);
}

// true := A < B with nulls being minimal
idx_t VectorOperations::DistinctLessThanNullsFirst(Vector &left, Vector &right, const SelectionVector *sel, idx_t count,
                                                   SelectionVector *true_sel, SelectionVector *false_sel) {
	return TemplatedDistinctSelectOperation<duckdb::DistinctLessThanNullsFirst, duckdb::DistinctLessThan>(
	    left, right, sel, count, true_sel, false_sel);
}

// true := A <= B with nulls being maximal
idx_t VectorOperations::DistinctLessThanEquals(Vector &left, Vector &right, const SelectionVector *sel, idx_t count,
                                               SelectionVector *true_sel, SelectionVector *false_sel) {
	return TemplatedDistinctSelectOperation<duckdb::DistinctLessThanEquals>(left, right, sel, count, true_sel,
	                                                                        false_sel);
}

// true := A != B with nulls being equal, inputs selected
idx_t VectorOperations::NestedNotEquals(Vector &left, Vector &right, const SelectionVector &sel, idx_t count,
                                        SelectionVector *true_sel, SelectionVector *false_sel) {
	return TemplatedDistinctSelectOperation<duckdb::DistinctFrom>(left, right, &sel, count, true_sel, false_sel);
}
// true := A == B with nulls being equal, inputs selected
idx_t VectorOperations::NestedEquals(Vector &left, Vector &right, const SelectionVector &sel, idx_t count,
                                     SelectionVector *true_sel, SelectionVector *false_sel) {
	return TemplatedDistinctSelectOperation<duckdb::NotDistinctFrom>(left, right, &sel, count, true_sel, false_sel);
}

} // namespace duckdb
//===--------------------------------------------------------------------===//
// null_operators.cpp
// Description: This file contains the implementation of the
// IS NULL/NOT IS NULL operators
//===--------------------------------------------------------------------===//




namespace duckdb {

template <bool INVERSE>
void IsNullLoop(Vector &input, Vector &result, idx_t count) {
	D_ASSERT(result.GetType() == LogicalType::BOOLEAN);

	if (input.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
		auto result_data = ConstantVector::GetData<bool>(result);
		*result_data = INVERSE ? !ConstantVector::IsNull(input) : ConstantVector::IsNull(input);
	} else {
		UnifiedVectorFormat data;
		input.ToUnifiedFormat(count, data);

		result.SetVectorType(VectorType::FLAT_VECTOR);
		auto result_data = FlatVector::GetData<bool>(result);
		for (idx_t i = 0; i < count; i++) {
			auto idx = data.sel->get_index(i);
			result_data[i] = INVERSE ? data.validity.RowIsValid(idx) : !data.validity.RowIsValid(idx);
		}
	}
}

void VectorOperations::IsNotNull(Vector &input, Vector &result, idx_t count) {
	IsNullLoop<true>(input, result, count);
}

void VectorOperations::IsNull(Vector &input, Vector &result, idx_t count) {
	IsNullLoop<false>(input, result, count);
}

bool VectorOperations::HasNotNull(Vector &input, idx_t count) {
	if (count == 0) {
		return false;
	}
	if (input.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		return !ConstantVector::IsNull(input);
	} else {
		UnifiedVectorFormat data;
		input.ToUnifiedFormat(count, data);

		if (data.validity.AllValid()) {
			return true;
		}
		for (idx_t i = 0; i < count; i++) {
			auto idx = data.sel->get_index(i);
			if (data.validity.RowIsValid(idx)) {
				return true;
			}
		}
		return false;
	}
}

bool VectorOperations::HasNull(Vector &input, idx_t count) {
	if (count == 0) {
		return false;
	}
	if (input.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		return ConstantVector::IsNull(input);
	} else {
		UnifiedVectorFormat data;
		input.ToUnifiedFormat(count, data);

		if (data.validity.AllValid()) {
			return false;
		}
		for (idx_t i = 0; i < count; i++) {
			auto idx = data.sel->get_index(i);
			if (!data.validity.RowIsValid(idx)) {
				return true;
			}
		}
		return false;
	}
}

idx_t VectorOperations::CountNotNull(Vector &input, const idx_t count) {
	idx_t valid = 0;

	UnifiedVectorFormat vdata;
	input.ToUnifiedFormat(count, vdata);
	if (vdata.validity.AllValid()) {
		return count;
	}
	switch (input.GetVectorType()) {
	case VectorType::FLAT_VECTOR:
		valid += vdata.validity.CountValid(count);
		break;
	case VectorType::CONSTANT_VECTOR:
		valid += vdata.validity.CountValid(1) * count;
		break;
	default:
		for (idx_t i = 0; i < count; ++i) {
			const auto row_idx = vdata.sel->get_index(i);
			valid += int(vdata.validity.RowIsValid(row_idx));
		}
		break;
	}

	return valid;
}

} // namespace duckdb
//===--------------------------------------------------------------------===//
// numeric_inplace_operators.cpp
// Description: This file contains the implementation of numeric inplace ops
// += *= /= -= %=
//===--------------------------------------------------------------------===//



#include <algorithm>

namespace duckdb {

//===--------------------------------------------------------------------===//
// In-Place Addition
//===--------------------------------------------------------------------===//

void VectorOperations::AddInPlace(Vector &input, int64_t right, idx_t count) {
	D_ASSERT(input.GetType().id() == LogicalTypeId::POINTER);
	if (right == 0) {
		return;
	}
	switch (input.GetVectorType()) {
	case VectorType::CONSTANT_VECTOR: {
		D_ASSERT(!ConstantVector::IsNull(input));
		auto data = ConstantVector::GetData<uintptr_t>(input);
		*data += right;
		break;
	}
	default: {
		D_ASSERT(input.GetVectorType() == VectorType::FLAT_VECTOR);
		auto data = FlatVector::GetData<uintptr_t>(input);
		for (idx_t i = 0; i < count; i++) {
			data[i] += right;
		}
		break;
	}
	}
}

} // namespace duckdb





namespace duckdb {

bool VectorOperations::TryCast(CastFunctionSet &set, GetCastFunctionInput &input, Vector &source, Vector &result,
                               idx_t count, string *error_message, bool strict) {
	auto cast_function = set.GetCastFunction(source.GetType(), result.GetType(), input);
	CastParameters parameters(cast_function.cast_data.get(), strict, error_message);
	return cast_function.function(source, result, count, parameters);
}

bool VectorOperations::DefaultTryCast(Vector &source, Vector &result, idx_t count, string *error_message, bool strict) {
	CastFunctionSet set;
	GetCastFunctionInput input;
	return VectorOperations::TryCast(set, input, source, result, count, error_message, strict);
}

void VectorOperations::DefaultCast(Vector &source, Vector &result, idx_t count, bool strict) {
	VectorOperations::DefaultTryCast(source, result, count, nullptr, strict);
}

bool VectorOperations::TryCast(ClientContext &context, Vector &source, Vector &result, idx_t count,
                               string *error_message, bool strict) {
	auto &config = DBConfig::GetConfig(context);
	auto &set = config.GetCastFunctions();
	GetCastFunctionInput get_input(context);
	return VectorOperations::TryCast(set, get_input, source, result, count, error_message, strict);
}

void VectorOperations::Cast(ClientContext &context, Vector &source, Vector &result, idx_t count, bool strict) {
	VectorOperations::TryCast(context, source, result, count, nullptr, strict);
}

} // namespace duckdb
//===--------------------------------------------------------------------===//
// copy.cpp
// Description: This file contains the implementation of the different copy
// functions
//===--------------------------------------------------------------------===//







namespace duckdb {

template <class T>
static void TemplatedCopy(const Vector &source, const SelectionVector &sel, Vector &target, idx_t source_offset,
                          idx_t target_offset, idx_t copy_count) {
	auto ldata = FlatVector::GetData<T>(source);
	auto tdata = FlatVector::GetData<T>(target);
	for (idx_t i = 0; i < copy_count; i++) {
		auto source_idx = sel.get_index(source_offset + i);
		tdata[target_offset + i] = ldata[source_idx];
	}
}

void VectorOperations::Copy(const Vector &source_p, Vector &target, const SelectionVector &sel_p, idx_t source_count,
                            idx_t source_offset, idx_t target_offset) {
	D_ASSERT(source_offset <= source_count);
	D_ASSERT(source_p.GetType() == target.GetType());
	idx_t copy_count = source_count - source_offset;

	SelectionVector owned_sel;
	const SelectionVector *sel = &sel_p;

	const Vector *source = &source_p;
	bool finished = false;
	while (!finished) {
		switch (source->GetVectorType()) {
		case VectorType::DICTIONARY_VECTOR: {
			// dictionary vector: merge selection vectors
			auto &child = DictionaryVector::Child(*source);
			auto &dict_sel = DictionaryVector::SelVector(*source);
			// merge the selection vectors and verify the child
			auto new_buffer = dict_sel.Slice(*sel, source_count);
			owned_sel.Initialize(new_buffer);
			sel = &owned_sel;
			source = &child;
			break;
		}
		case VectorType::SEQUENCE_VECTOR: {
			int64_t start, increment;
			Vector seq(source->GetType());
			SequenceVector::GetSequence(*source, start, increment);
			VectorOperations::GenerateSequence(seq, source_count, *sel, start, increment);
			VectorOperations::Copy(seq, target, *sel, source_count, source_offset, target_offset);
			return;
		}
		case VectorType::CONSTANT_VECTOR:
			sel = ConstantVector::ZeroSelectionVector(copy_count, owned_sel);
			finished = true;
			break;
		case VectorType::FSST_VECTOR:
			finished = true;
			break;
		case VectorType::FLAT_VECTOR:
			finished = true;
			break;
		default:
			throw NotImplementedException("FIXME unimplemented vector type for VectorOperations::Copy");
		}
	}

	if (copy_count == 0) {
		return;
	}

	// Allow copying of a single value to constant vectors
	const auto target_vector_type = target.GetVectorType();
	if (copy_count == 1 && target_vector_type == VectorType::CONSTANT_VECTOR) {
		target_offset = 0;
		target.SetVectorType(VectorType::FLAT_VECTOR);
	}
	D_ASSERT(target.GetVectorType() == VectorType::FLAT_VECTOR);

	// first copy the nullmask
	auto &tmask = FlatVector::Validity(target);
	if (source->GetVectorType() == VectorType::CONSTANT_VECTOR) {
		const bool valid = !ConstantVector::IsNull(*source);
		for (idx_t i = 0; i < copy_count; i++) {
			tmask.Set(target_offset + i, valid);
		}
	} else {
		const ValidityMask *smask;
		if (source->GetVectorType() == VectorType::FLAT_VECTOR) {
			smask = &(FlatVector::Validity(*source));
		} else if (source->GetVectorType() == VectorType::FSST_VECTOR) {
			smask = &(FSSTVector::Validity(*source));
		} else {
			throw InternalException("Unsupported vector type in vector copy");
		}

		if (smask->IsMaskSet()) {
			for (idx_t i = 0; i < copy_count; i++) {
				auto idx = sel->get_index(source_offset + i);

				if (smask->RowIsValid(idx)) {
					// set valid
					if (!tmask.AllValid()) {
						tmask.SetValidUnsafe(target_offset + i);
					}
				} else {
					// set invalid
					if (tmask.AllValid()) {
						auto init_size = MaxValue<idx_t>(STANDARD_VECTOR_SIZE, target_offset + copy_count);
						tmask.Initialize(init_size);
					}
					tmask.SetInvalidUnsafe(target_offset + i);
				}
			}
		}
	}

	D_ASSERT(sel);

	// For FSST Vectors we decompress instead of copying.
	if (source->GetVectorType() == VectorType::FSST_VECTOR) {
		FSSTVector::DecompressVector(*source, target, source_offset, target_offset, copy_count, sel);
		return;
	}

	// now copy over the data
	switch (source->GetType().InternalType()) {
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
		TemplatedCopy<int8_t>(*source, *sel, target, source_offset, target_offset, copy_count);
		break;
	case PhysicalType::INT16:
		TemplatedCopy<int16_t>(*source, *sel, target, source_offset, target_offset, copy_count);
		break;
	case PhysicalType::INT32:
		TemplatedCopy<int32_t>(*source, *sel, target, source_offset, target_offset, copy_count);
		break;
	case PhysicalType::INT64:
		TemplatedCopy<int64_t>(*source, *sel, target, source_offset, target_offset, copy_count);
		break;
	case PhysicalType::UINT8:
		TemplatedCopy<uint8_t>(*source, *sel, target, source_offset, target_offset, copy_count);
		break;
	case PhysicalType::UINT16:
		TemplatedCopy<uint16_t>(*source, *sel, target, source_offset, target_offset, copy_count);
		break;
	case PhysicalType::UINT32:
		TemplatedCopy<uint32_t>(*source, *sel, target, source_offset, target_offset, copy_count);
		break;
	case PhysicalType::UINT64:
		TemplatedCopy<uint64_t>(*source, *sel, target, source_offset, target_offset, copy_count);
		break;
	case PhysicalType::INT128:
		TemplatedCopy<hugeint_t>(*source, *sel, target, source_offset, target_offset, copy_count);
		break;
	case PhysicalType::FLOAT:
		TemplatedCopy<float>(*source, *sel, target, source_offset, target_offset, copy_count);
		break;
	case PhysicalType::DOUBLE:
		TemplatedCopy<double>(*source, *sel, target, source_offset, target_offset, copy_count);
		break;
	case PhysicalType::INTERVAL:
		TemplatedCopy<interval_t>(*source, *sel, target, source_offset, target_offset, copy_count);
		break;
	case PhysicalType::VARCHAR: {
		auto ldata = FlatVector::GetData<string_t>(*source);
		auto tdata = FlatVector::GetData<string_t>(target);
		for (idx_t i = 0; i < copy_count; i++) {
			auto source_idx = sel->get_index(source_offset + i);
			auto target_idx = target_offset + i;
			if (tmask.RowIsValid(target_idx)) {
				tdata[target_idx] = StringVector::AddStringOrBlob(target, ldata[source_idx]);
			}
		}
		break;
	}
	case PhysicalType::STRUCT: {
		auto &source_children = StructVector::GetEntries(*source);
		auto &target_children = StructVector::GetEntries(target);
		D_ASSERT(source_children.size() == target_children.size());
		for (idx_t i = 0; i < source_children.size(); i++) {
			VectorOperations::Copy(*source_children[i], *target_children[i], sel_p, source_count, source_offset,
			                       target_offset);
		}
		break;
	}
	case PhysicalType::LIST: {
		D_ASSERT(target.GetType().InternalType() == PhysicalType::LIST);

		auto &source_child = ListVector::GetEntry(*source);
		auto sdata = FlatVector::GetData<list_entry_t>(*source);
		auto tdata = FlatVector::GetData<list_entry_t>(target);

		if (target_vector_type == VectorType::CONSTANT_VECTOR) {
			// If we are only writing one value, then the copied values (if any) are contiguous
			// and we can just Append from the offset position
			if (!tmask.RowIsValid(target_offset)) {
				break;
			}
			auto source_idx = sel->get_index(source_offset);
			auto &source_entry = sdata[source_idx];
			const idx_t source_child_size = source_entry.length + source_entry.offset;

			//! overwrite constant target vectors.
			ListVector::SetListSize(target, 0);
			ListVector::Append(target, source_child, source_child_size, source_entry.offset);

			auto &target_entry = tdata[target_offset];
			target_entry.length = source_entry.length;
			target_entry.offset = 0;
		} else {
			//! if the source has list offsets, we need to append them to the target
			//! build a selection vector for the copied child elements
			vector<sel_t> child_rows;
			for (idx_t i = 0; i < copy_count; ++i) {
				if (tmask.RowIsValid(target_offset + i)) {
					auto source_idx = sel->get_index(source_offset + i);
					auto &source_entry = sdata[source_idx];
					for (idx_t j = 0; j < source_entry.length; ++j) {
						child_rows.emplace_back(source_entry.offset + j);
					}
				}
			}
			idx_t source_child_size = child_rows.size();
			SelectionVector child_sel(child_rows.data());

			idx_t old_target_child_len = ListVector::GetListSize(target);

			//! append to list itself
			ListVector::Append(target, source_child, child_sel, source_child_size);

			//! now write the list offsets
			for (idx_t i = 0; i < copy_count; i++) {
				auto source_idx = sel->get_index(source_offset + i);
				auto &source_entry = sdata[source_idx];
				auto &target_entry = tdata[target_offset + i];

				target_entry.length = source_entry.length;
				target_entry.offset = old_target_child_len;
				if (tmask.RowIsValid(target_offset + i)) {
					old_target_child_len += target_entry.length;
				}
			}
		}
		break;
	}
	default:
		throw NotImplementedException("Unimplemented type '%s' for copy!",
		                              TypeIdToString(source->GetType().InternalType()));
	}

	if (target_vector_type != VectorType::FLAT_VECTOR) {
		target.SetVectorType(target_vector_type);
	}
}

void VectorOperations::Copy(const Vector &source, Vector &target, idx_t source_count, idx_t source_offset,
                            idx_t target_offset) {
	VectorOperations::Copy(source, target, *FlatVector::IncrementalSelectionVector(), source_count, source_offset,
	                       target_offset);
}

} // namespace duckdb
//===--------------------------------------------------------------------===//
// hash.cpp
// Description: This file contains the vectorized hash implementations
//===--------------------------------------------------------------------===//







namespace duckdb {

struct HashOp {
	static const hash_t NULL_HASH = 0xbf58476d1ce4e5b9;

	template <class T>
	static inline hash_t Operation(T input, bool is_null) {
		return is_null ? NULL_HASH : duckdb::Hash<T>(input);
	}
};

static inline hash_t CombineHashScalar(hash_t a, hash_t b) {
	return (a * UINT64_C(0xbf58476d1ce4e5b9)) ^ b;
}

template <bool HAS_RSEL, class T>
static inline void TightLoopHash(T *__restrict ldata, hash_t *__restrict result_data, const SelectionVector *rsel,
                                 idx_t count, const SelectionVector *__restrict sel_vector, ValidityMask &mask) {
	if (!mask.AllValid()) {
		for (idx_t i = 0; i < count; i++) {
			auto ridx = HAS_RSEL ? rsel->get_index(i) : i;
			auto idx = sel_vector->get_index(ridx);
			result_data[ridx] = HashOp::Operation(ldata[idx], !mask.RowIsValid(idx));
		}
	} else {
		for (idx_t i = 0; i < count; i++) {
			auto ridx = HAS_RSEL ? rsel->get_index(i) : i;
			auto idx = sel_vector->get_index(ridx);
			result_data[ridx] = duckdb::Hash<T>(ldata[idx]);
		}
	}
}

template <bool HAS_RSEL, class T>
static inline void TemplatedLoopHash(Vector &input, Vector &result, const SelectionVector *rsel, idx_t count) {
	if (input.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);

		auto ldata = ConstantVector::GetData<T>(input);
		auto result_data = ConstantVector::GetData<hash_t>(result);
		*result_data = HashOp::Operation(*ldata, ConstantVector::IsNull(input));
	} else {
		result.SetVectorType(VectorType::FLAT_VECTOR);

		UnifiedVectorFormat idata;
		input.ToUnifiedFormat(count, idata);

		TightLoopHash<HAS_RSEL, T>((T *)idata.data, FlatVector::GetData<hash_t>(result), rsel, count, idata.sel,
		                           idata.validity);
	}
}

template <bool HAS_RSEL, bool FIRST_HASH>
static inline void StructLoopHash(Vector &input, Vector &hashes, const SelectionVector *rsel, idx_t count) {
	auto &children = StructVector::GetEntries(input);

	D_ASSERT(!children.empty());
	idx_t col_no = 0;
	if (HAS_RSEL) {
		if (FIRST_HASH) {
			VectorOperations::Hash(*children[col_no++], hashes, *rsel, count);
		} else {
			VectorOperations::CombineHash(hashes, *children[col_no++], *rsel, count);
		}
		while (col_no < children.size()) {
			VectorOperations::CombineHash(hashes, *children[col_no++], *rsel, count);
		}
	} else {
		if (FIRST_HASH) {
			VectorOperations::Hash(*children[col_no++], hashes, count);
		} else {
			VectorOperations::CombineHash(hashes, *children[col_no++], count);
		}
		while (col_no < children.size()) {
			VectorOperations::CombineHash(hashes, *children[col_no++], count);
		}
	}
}

template <bool HAS_RSEL, bool FIRST_HASH>
static inline void ListLoopHash(Vector &input, Vector &hashes, const SelectionVector *rsel, idx_t count) {
	auto hdata = FlatVector::GetData<hash_t>(hashes);

	UnifiedVectorFormat idata;
	input.ToUnifiedFormat(count, idata);
	const auto ldata = (const list_entry_t *)idata.data;

	// Hash the children into a temporary
	auto &child = ListVector::GetEntry(input);
	const auto child_count = ListVector::GetListSize(input);

	Vector child_hashes(LogicalType::HASH, child_count);
	if (child_count > 0) {
		VectorOperations::Hash(child, child_hashes, child_count);
	}
	auto chdata = FlatVector::GetData<hash_t>(child_hashes);

	// Reduce the number of entries to check to the non-empty ones
	SelectionVector unprocessed(count);
	SelectionVector cursor(HAS_RSEL ? STANDARD_VECTOR_SIZE : count);
	idx_t remaining = 0;
	for (idx_t i = 0; i < count; ++i) {
		const idx_t ridx = HAS_RSEL ? rsel->get_index(i) : i;
		const auto lidx = idata.sel->get_index(ridx);
		const auto &entry = ldata[lidx];
		if (idata.validity.RowIsValid(lidx) && entry.length > 0) {
			unprocessed.set_index(remaining++, ridx);
			cursor.set_index(ridx, entry.offset);
		} else if (FIRST_HASH) {
			hdata[ridx] = HashOp::NULL_HASH;
		}
		// Empty or NULL non-first elements have no effect.
	}

	count = remaining;
	if (count == 0) {
		return;
	}

	// Merge the first position hash into the main hash
	idx_t position = 1;
	if (FIRST_HASH) {
		remaining = 0;
		for (idx_t i = 0; i < count; ++i) {
			const auto ridx = unprocessed.get_index(i);
			const auto cidx = cursor.get_index(ridx);
			hdata[ridx] = chdata[cidx];

			const auto lidx = idata.sel->get_index(ridx);
			const auto &entry = ldata[lidx];
			if (entry.length > position) {
				// Entry still has values to hash
				unprocessed.set_index(remaining++, ridx);
				cursor.set_index(ridx, cidx + 1);
			}
		}
		count = remaining;
		if (count == 0) {
			return;
		}
		++position;
	}

	// Combine the hashes for the remaining positions until there are none left
	for (;; ++position) {
		remaining = 0;
		for (idx_t i = 0; i < count; ++i) {
			const auto ridx = unprocessed.get_index(i);
			const auto cidx = cursor.get_index(ridx);
			hdata[ridx] = CombineHashScalar(hdata[ridx], chdata[cidx]);

			const auto lidx = idata.sel->get_index(ridx);
			const auto &entry = ldata[lidx];
			if (entry.length > position) {
				// Entry still has values to hash
				unprocessed.set_index(remaining++, ridx);
				cursor.set_index(ridx, cidx + 1);
			}
		}

		count = remaining;
		if (count == 0) {
			break;
		}
	}
}

template <bool HAS_RSEL>
static inline void HashTypeSwitch(Vector &input, Vector &result, const SelectionVector *rsel, idx_t count) {
	D_ASSERT(result.GetType().id() == LogicalType::HASH);
	switch (input.GetType().InternalType()) {
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
		TemplatedLoopHash<HAS_RSEL, int8_t>(input, result, rsel, count);
		break;
	case PhysicalType::INT16:
		TemplatedLoopHash<HAS_RSEL, int16_t>(input, result, rsel, count);
		break;
	case PhysicalType::INT32:
		TemplatedLoopHash<HAS_RSEL, int32_t>(input, result, rsel, count);
		break;
	case PhysicalType::INT64:
		TemplatedLoopHash<HAS_RSEL, int64_t>(input, result, rsel, count);
		break;
	case PhysicalType::UINT8:
		TemplatedLoopHash<HAS_RSEL, uint8_t>(input, result, rsel, count);
		break;
	case PhysicalType::UINT16:
		TemplatedLoopHash<HAS_RSEL, uint16_t>(input, result, rsel, count);
		break;
	case PhysicalType::UINT32:
		TemplatedLoopHash<HAS_RSEL, uint32_t>(input, result, rsel, count);
		break;
	case PhysicalType::UINT64:
		TemplatedLoopHash<HAS_RSEL, uint64_t>(input, result, rsel, count);
		break;
	case PhysicalType::INT128:
		TemplatedLoopHash<HAS_RSEL, hugeint_t>(input, result, rsel, count);
		break;
	case PhysicalType::FLOAT:
		TemplatedLoopHash<HAS_RSEL, float>(input, result, rsel, count);
		break;
	case PhysicalType::DOUBLE:
		TemplatedLoopHash<HAS_RSEL, double>(input, result, rsel, count);
		break;
	case PhysicalType::INTERVAL:
		TemplatedLoopHash<HAS_RSEL, interval_t>(input, result, rsel, count);
		break;
	case PhysicalType::VARCHAR:
		TemplatedLoopHash<HAS_RSEL, string_t>(input, result, rsel, count);
		break;
	case PhysicalType::MAP:
	case PhysicalType::STRUCT:
		StructLoopHash<HAS_RSEL, true>(input, result, rsel, count);
		break;
	case PhysicalType::LIST:
		ListLoopHash<HAS_RSEL, true>(input, result, rsel, count);
		break;
	default:
		throw InvalidTypeException(input.GetType(), "Invalid type for hash");
	}
}

void VectorOperations::Hash(Vector &input, Vector &result, idx_t count) {
	HashTypeSwitch<false>(input, result, nullptr, count);
}

void VectorOperations::Hash(Vector &input, Vector &result, const SelectionVector &sel, idx_t count) {
	HashTypeSwitch<true>(input, result, &sel, count);
}

template <bool HAS_RSEL, class T>
static inline void TightLoopCombineHashConstant(T *__restrict ldata, hash_t constant_hash, hash_t *__restrict hash_data,
                                                const SelectionVector *rsel, idx_t count,
                                                const SelectionVector *__restrict sel_vector, ValidityMask &mask) {
	if (!mask.AllValid()) {
		for (idx_t i = 0; i < count; i++) {
			auto ridx = HAS_RSEL ? rsel->get_index(i) : i;
			auto idx = sel_vector->get_index(ridx);
			auto other_hash = HashOp::Operation(ldata[idx], !mask.RowIsValid(idx));
			hash_data[ridx] = CombineHashScalar(constant_hash, other_hash);
		}
	} else {
		for (idx_t i = 0; i < count; i++) {
			auto ridx = HAS_RSEL ? rsel->get_index(i) : i;
			auto idx = sel_vector->get_index(ridx);
			auto other_hash = duckdb::Hash<T>(ldata[idx]);
			hash_data[ridx] = CombineHashScalar(constant_hash, other_hash);
		}
	}
}

template <bool HAS_RSEL, class T>
static inline void TightLoopCombineHash(T *__restrict ldata, hash_t *__restrict hash_data, const SelectionVector *rsel,
                                        idx_t count, const SelectionVector *__restrict sel_vector, ValidityMask &mask) {
	if (!mask.AllValid()) {
		for (idx_t i = 0; i < count; i++) {
			auto ridx = HAS_RSEL ? rsel->get_index(i) : i;
			auto idx = sel_vector->get_index(ridx);
			auto other_hash = HashOp::Operation(ldata[idx], !mask.RowIsValid(idx));
			hash_data[ridx] = CombineHashScalar(hash_data[ridx], other_hash);
		}
	} else {
		for (idx_t i = 0; i < count; i++) {
			auto ridx = HAS_RSEL ? rsel->get_index(i) : i;
			auto idx = sel_vector->get_index(ridx);
			auto other_hash = duckdb::Hash<T>(ldata[idx]);
			hash_data[ridx] = CombineHashScalar(hash_data[ridx], other_hash);
		}
	}
}

template <bool HAS_RSEL, class T>
void TemplatedLoopCombineHash(Vector &input, Vector &hashes, const SelectionVector *rsel, idx_t count) {
	if (input.GetVectorType() == VectorType::CONSTANT_VECTOR && hashes.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		auto ldata = ConstantVector::GetData<T>(input);
		auto hash_data = ConstantVector::GetData<hash_t>(hashes);

		auto other_hash = HashOp::Operation(*ldata, ConstantVector::IsNull(input));
		*hash_data = CombineHashScalar(*hash_data, other_hash);
	} else {
		UnifiedVectorFormat idata;
		input.ToUnifiedFormat(count, idata);
		if (hashes.GetVectorType() == VectorType::CONSTANT_VECTOR) {
			// mix constant with non-constant, first get the constant value
			auto constant_hash = *ConstantVector::GetData<hash_t>(hashes);
			// now re-initialize the hashes vector to an empty flat vector
			hashes.SetVectorType(VectorType::FLAT_VECTOR);
			TightLoopCombineHashConstant<HAS_RSEL, T>((T *)idata.data, constant_hash,
			                                          FlatVector::GetData<hash_t>(hashes), rsel, count, idata.sel,
			                                          idata.validity);
		} else {
			D_ASSERT(hashes.GetVectorType() == VectorType::FLAT_VECTOR);
			TightLoopCombineHash<HAS_RSEL, T>((T *)idata.data, FlatVector::GetData<hash_t>(hashes), rsel, count,
			                                  idata.sel, idata.validity);
		}
	}
}

template <bool HAS_RSEL>
static inline void CombineHashTypeSwitch(Vector &hashes, Vector &input, const SelectionVector *rsel, idx_t count) {
	D_ASSERT(hashes.GetType().id() == LogicalType::HASH);
	switch (input.GetType().InternalType()) {
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
		TemplatedLoopCombineHash<HAS_RSEL, int8_t>(input, hashes, rsel, count);
		break;
	case PhysicalType::INT16:
		TemplatedLoopCombineHash<HAS_RSEL, int16_t>(input, hashes, rsel, count);
		break;
	case PhysicalType::INT32:
		TemplatedLoopCombineHash<HAS_RSEL, int32_t>(input, hashes, rsel, count);
		break;
	case PhysicalType::INT64:
		TemplatedLoopCombineHash<HAS_RSEL, int64_t>(input, hashes, rsel, count);
		break;
	case PhysicalType::UINT8:
		TemplatedLoopCombineHash<HAS_RSEL, uint8_t>(input, hashes, rsel, count);
		break;
	case PhysicalType::UINT16:
		TemplatedLoopCombineHash<HAS_RSEL, uint16_t>(input, hashes, rsel, count);
		break;
	case PhysicalType::UINT32:
		TemplatedLoopCombineHash<HAS_RSEL, uint32_t>(input, hashes, rsel, count);
		break;
	case PhysicalType::UINT64:
		TemplatedLoopCombineHash<HAS_RSEL, uint64_t>(input, hashes, rsel, count);
		break;
	case PhysicalType::INT128:
		TemplatedLoopCombineHash<HAS_RSEL, hugeint_t>(input, hashes, rsel, count);
		break;
	case PhysicalType::FLOAT:
		TemplatedLoopCombineHash<HAS_RSEL, float>(input, hashes, rsel, count);
		break;
	case PhysicalType::DOUBLE:
		TemplatedLoopCombineHash<HAS_RSEL, double>(input, hashes, rsel, count);
		break;
	case PhysicalType::INTERVAL:
		TemplatedLoopCombineHash<HAS_RSEL, interval_t>(input, hashes, rsel, count);
		break;
	case PhysicalType::VARCHAR:
		TemplatedLoopCombineHash<HAS_RSEL, string_t>(input, hashes, rsel, count);
		break;
	case PhysicalType::MAP:
	case PhysicalType::STRUCT:
		StructLoopHash<HAS_RSEL, false>(input, hashes, rsel, count);
		break;
	case PhysicalType::LIST:
		ListLoopHash<HAS_RSEL, false>(input, hashes, rsel, count);
		break;
	default:
		throw InvalidTypeException(input.GetType(), "Invalid type for hash");
	}
}

void VectorOperations::CombineHash(Vector &hashes, Vector &input, idx_t count) {
	CombineHashTypeSwitch<false>(hashes, input, nullptr, count);
}

void VectorOperations::CombineHash(Vector &hashes, Vector &input, const SelectionVector &rsel, idx_t count) {
	CombineHashTypeSwitch<true>(hashes, input, &rsel, count);
}

} // namespace duckdb




namespace duckdb {

template <class T>
static void CopyToStorageLoop(UnifiedVectorFormat &vdata, idx_t count, data_ptr_t target) {
	auto ldata = (T *)vdata.data;
	auto result_data = (T *)target;
	for (idx_t i = 0; i < count; i++) {
		auto idx = vdata.sel->get_index(i);
		if (!vdata.validity.RowIsValid(idx)) {
			result_data[i] = NullValue<T>();
		} else {
			result_data[i] = ldata[idx];
		}
	}
}

void VectorOperations::WriteToStorage(Vector &source, idx_t count, data_ptr_t target) {
	if (count == 0) {
		return;
	}
	UnifiedVectorFormat vdata;
	source.ToUnifiedFormat(count, vdata);

	switch (source.GetType().InternalType()) {
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
		CopyToStorageLoop<int8_t>(vdata, count, target);
		break;
	case PhysicalType::INT16:
		CopyToStorageLoop<int16_t>(vdata, count, target);
		break;
	case PhysicalType::INT32:
		CopyToStorageLoop<int32_t>(vdata, count, target);
		break;
	case PhysicalType::INT64:
		CopyToStorageLoop<int64_t>(vdata, count, target);
		break;
	case PhysicalType::UINT8:
		CopyToStorageLoop<uint8_t>(vdata, count, target);
		break;
	case PhysicalType::UINT16:
		CopyToStorageLoop<uint16_t>(vdata, count, target);
		break;
	case PhysicalType::UINT32:
		CopyToStorageLoop<uint32_t>(vdata, count, target);
		break;
	case PhysicalType::UINT64:
		CopyToStorageLoop<uint64_t>(vdata, count, target);
		break;
	case PhysicalType::INT128:
		CopyToStorageLoop<hugeint_t>(vdata, count, target);
		break;
	case PhysicalType::FLOAT:
		CopyToStorageLoop<float>(vdata, count, target);
		break;
	case PhysicalType::DOUBLE:
		CopyToStorageLoop<double>(vdata, count, target);
		break;
	case PhysicalType::INTERVAL:
		CopyToStorageLoop<interval_t>(vdata, count, target);
		break;
	default:
		throw NotImplementedException("Unimplemented type for WriteToStorage");
	}
}

template <class T>
static void ReadFromStorageLoop(data_ptr_t source, idx_t count, Vector &result) {
	auto ldata = (T *)source;
	auto result_data = FlatVector::GetData<T>(result);
	for (idx_t i = 0; i < count; i++) {
		result_data[i] = ldata[i];
	}
}

void VectorOperations::ReadFromStorage(data_ptr_t source, idx_t count, Vector &result) {
	result.SetVectorType(VectorType::FLAT_VECTOR);
	switch (result.GetType().InternalType()) {
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
		ReadFromStorageLoop<int8_t>(source, count, result);
		break;
	case PhysicalType::INT16:
		ReadFromStorageLoop<int16_t>(source, count, result);
		break;
	case PhysicalType::INT32:
		ReadFromStorageLoop<int32_t>(source, count, result);
		break;
	case PhysicalType::INT64:
		ReadFromStorageLoop<int64_t>(source, count, result);
		break;
	case PhysicalType::UINT8:
		ReadFromStorageLoop<uint8_t>(source, count, result);
		break;
	case PhysicalType::UINT16:
		ReadFromStorageLoop<uint16_t>(source, count, result);
		break;
	case PhysicalType::UINT32:
		ReadFromStorageLoop<uint32_t>(source, count, result);
		break;
	case PhysicalType::UINT64:
		ReadFromStorageLoop<uint64_t>(source, count, result);
		break;
	case PhysicalType::INT128:
		ReadFromStorageLoop<hugeint_t>(source, count, result);
		break;
	case PhysicalType::FLOAT:
		ReadFromStorageLoop<float>(source, count, result);
		break;
	case PhysicalType::DOUBLE:
		ReadFromStorageLoop<double>(source, count, result);
		break;
	case PhysicalType::INTERVAL:
		ReadFromStorageLoop<interval_t>(source, count, result);
		break;
	default:
		throw NotImplementedException("Unimplemented type for ReadFromStorage");
	}
}

} // namespace duckdb






namespace duckdb {

VirtualFileSystem::VirtualFileSystem() : default_fs(FileSystem::CreateLocal()) {
	RegisterSubSystem(FileCompressionType::GZIP, make_unique<GZipFileSystem>());
}

unique_ptr<FileHandle> VirtualFileSystem::OpenFile(const string &path, uint8_t flags, FileLockType lock,
                                                   FileCompressionType compression, FileOpener *opener) {
	if (compression == FileCompressionType::AUTO_DETECT) {
		// auto detect compression settings based on file name
		auto lower_path = StringUtil::Lower(path);
		if (StringUtil::EndsWith(lower_path, ".gz")) {
			compression = FileCompressionType::GZIP;
		} else if (StringUtil::EndsWith(lower_path, ".zst")) {
			compression = FileCompressionType::ZSTD;
		} else {
			compression = FileCompressionType::UNCOMPRESSED;
		}
	}
	// open the base file handle
	auto file_handle = FindFileSystem(path)->OpenFile(path, flags, lock, FileCompressionType::UNCOMPRESSED, opener);
	if (file_handle->GetType() == FileType::FILE_TYPE_FIFO) {
		file_handle = PipeFileSystem::OpenPipe(move(file_handle));
	} else if (compression != FileCompressionType::UNCOMPRESSED) {
		auto entry = compressed_fs.find(compression);
		if (entry == compressed_fs.end()) {
			throw NotImplementedException(
			    "Attempting to open a compressed file, but the compression type is not supported");
		}
		file_handle = entry->second->OpenCompressedFile(move(file_handle), flags & FileFlags::FILE_FLAGS_WRITE);
	}
	return file_handle;
}

} // namespace duckdb


namespace duckdb {

#ifdef DUCKDB_WINDOWS

std::wstring WindowsUtil::UTF8ToUnicode(const char *input) {
	idx_t result_size;

	result_size = MultiByteToWideChar(CP_UTF8, 0, input, -1, nullptr, 0);
	if (result_size == 0) {
		throw IOException("Failure in MultiByteToWideChar");
	}
	auto buffer = unique_ptr<wchar_t[]>(new wchar_t[result_size]);
	result_size = MultiByteToWideChar(CP_UTF8, 0, input, -1, buffer.get(), result_size);
	if (result_size == 0) {
		throw IOException("Failure in MultiByteToWideChar");
	}
	return std::wstring(buffer.get(), result_size);
}

static string WideCharToMultiByteWrapper(LPCWSTR input, uint32_t code_page) {
	idx_t result_size;

	result_size = WideCharToMultiByte(code_page, 0, input, -1, 0, 0, 0, 0);
	if (result_size == 0) {
		throw IOException("Failure in WideCharToMultiByte");
	}
	auto buffer = unique_ptr<char[]>(new char[result_size]);
	result_size = WideCharToMultiByte(code_page, 0, input, -1, buffer.get(), result_size, 0, 0);
	if (result_size == 0) {
		throw IOException("Failure in WideCharToMultiByte");
	}
	return string(buffer.get(), result_size - 1);
}

string WindowsUtil::UnicodeToUTF8(LPCWSTR input) {
	return WideCharToMultiByteWrapper(input, CP_UTF8);
}

static string WindowsUnicodeToMBCS(LPCWSTR unicode_text, int use_ansi) {
	uint32_t code_page = use_ansi ? CP_ACP : CP_OEMCP;
	return WideCharToMultiByteWrapper(unicode_text, code_page);
}

string WindowsUtil::UTF8ToMBCS(const char *input, bool use_ansi) {
	auto unicode = WindowsUtil::UTF8ToUnicode(input);
	return WindowsUnicodeToMBCS(unicode.c_str(), use_ansi);
}

#endif

} // namespace duckdb



#include <vector>

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

using ValidityBytes = RowLayout::ValidityBytes;

GroupedAggregateHashTable::GroupedAggregateHashTable(ClientContext &context, Allocator &allocator,
                                                     vector<LogicalType> group_types, vector<LogicalType> payload_types,
                                                     const vector<BoundAggregateExpression *> &bindings,
                                                     HtEntryType entry_type)
    : GroupedAggregateHashTable(context, allocator, move(group_types), move(payload_types),
                                AggregateObject::CreateAggregateObjects(bindings), entry_type) {
}

GroupedAggregateHashTable::GroupedAggregateHashTable(ClientContext &context, Allocator &allocator,
                                                     vector<LogicalType> group_types)
    : GroupedAggregateHashTable(context, allocator, move(group_types), {}, vector<AggregateObject>()) {
}

GroupedAggregateHashTable::GroupedAggregateHashTable(ClientContext &context, Allocator &allocator,
                                                     vector<LogicalType> group_types_p,
                                                     vector<LogicalType> payload_types_p,
                                                     vector<AggregateObject> aggregate_objects_p,
                                                     HtEntryType entry_type)
    : BaseAggregateHashTable(context, allocator, aggregate_objects_p, move(payload_types_p)), entry_type(entry_type),
      capacity(0), entries(0), payload_page_offset(0), is_finalized(false), ht_offsets(LogicalTypeId::BIGINT),
      hash_salts(LogicalTypeId::SMALLINT), group_compare_vector(STANDARD_VECTOR_SIZE),
      no_match_vector(STANDARD_VECTOR_SIZE), empty_vector(STANDARD_VECTOR_SIZE) {

	// Append hash column to the end and initialise the row layout
	group_types_p.emplace_back(LogicalType::HASH);
	layout.Initialize(move(group_types_p), move(aggregate_objects_p));

	// HT layout
	hash_offset = layout.GetOffsets()[layout.ColumnCount() - 1];

	tuple_size = layout.GetRowWidth();

	D_ASSERT(tuple_size <= Storage::BLOCK_SIZE);
	tuples_per_block = Storage::BLOCK_SIZE / tuple_size;
	hashes_hdl = buffer_manager.Allocate(Storage::BLOCK_SIZE);
	hashes_hdl_ptr = hashes_hdl.Ptr();

	switch (entry_type) {
	case HtEntryType::HT_WIDTH_64: {
		hash_prefix_shift = (HASH_WIDTH - sizeof(aggr_ht_entry_64::salt)) * 8;
		Resize<aggr_ht_entry_64>(STANDARD_VECTOR_SIZE * 2L);
		break;
	}
	case HtEntryType::HT_WIDTH_32: {
		hash_prefix_shift = (HASH_WIDTH - sizeof(aggr_ht_entry_32::salt)) * 8;
		Resize<aggr_ht_entry_32>(STANDARD_VECTOR_SIZE * 2L);
		break;
	}
	default:
		throw InternalException("Unknown HT entry width");
	}

	predicates.resize(layout.ColumnCount() - 1, ExpressionType::COMPARE_EQUAL);
	string_heap = make_unique<RowDataCollection>(buffer_manager, (idx_t)Storage::BLOCK_SIZE, 1, true);
}

GroupedAggregateHashTable::~GroupedAggregateHashTable() {
	Destroy();
}

template <class FUNC>
void GroupedAggregateHashTable::PayloadApply(FUNC fun) {
	if (entries == 0) {
		return;
	}
	idx_t apply_entries = entries;
	idx_t page_nr = 0;
	idx_t page_offset = 0;

	for (auto &payload_chunk_ptr : payload_hds_ptrs) {
		auto this_entries = MinValue(tuples_per_block, apply_entries);
		page_offset = 0;
		for (data_ptr_t ptr = payload_chunk_ptr, end = payload_chunk_ptr + this_entries * tuple_size; ptr < end;
		     ptr += tuple_size) {
			fun(page_nr, page_offset++, ptr);
		}
		apply_entries -= this_entries;
		page_nr++;
	}
	D_ASSERT(apply_entries == 0);
}

void GroupedAggregateHashTable::NewBlock() {
	auto pin = buffer_manager.Allocate(Storage::BLOCK_SIZE);
	payload_hds.push_back(move(pin));
	payload_hds_ptrs.push_back(payload_hds.back().Ptr());
	payload_page_offset = 0;
}

void GroupedAggregateHashTable::Destroy() {
	// check if there is a destructor
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
	data_ptr_t data_pointers[STANDARD_VECTOR_SIZE];
	Vector state_vector(LogicalType::POINTER, (data_ptr_t)data_pointers);
	idx_t count = 0;

	PayloadApply([&](idx_t page_nr, idx_t page_offset, data_ptr_t ptr) {
		data_pointers[count++] = ptr;
		if (count == STANDARD_VECTOR_SIZE) {
			RowOperations::DestroyStates(layout, state_vector, count);
			count = 0;
		}
	});
	RowOperations::DestroyStates(layout, state_vector, count);
}

template <class ENTRY>
void GroupedAggregateHashTable::VerifyInternal() {
	auto hashes_ptr = (ENTRY *)hashes_hdl_ptr;
	D_ASSERT(payload_hds.size() == payload_hds_ptrs.size());
	idx_t count = 0;
	for (idx_t i = 0; i < capacity; i++) {
		if (hashes_ptr[i].page_nr > 0) {
			D_ASSERT(hashes_ptr[i].page_offset < tuples_per_block);
			D_ASSERT(hashes_ptr[i].page_nr <= payload_hds.size());
			auto ptr = payload_hds_ptrs[hashes_ptr[i].page_nr - 1] + ((hashes_ptr[i].page_offset) * tuple_size);
			auto hash = Load<hash_t>(ptr + hash_offset);
			D_ASSERT((hashes_ptr[i].salt) == (hash >> hash_prefix_shift));

			count++;
		}
	}
	D_ASSERT(count == entries);
}

idx_t GroupedAggregateHashTable::MaxCapacity() {
	idx_t max_pages = 0;
	idx_t max_tuples = 0;

	switch (entry_type) {
	case HtEntryType::HT_WIDTH_32:
		max_pages = NumericLimits<uint8_t>::Maximum();
		max_tuples = NumericLimits<uint16_t>::Maximum();
		break;
	default:
		D_ASSERT(entry_type == HtEntryType::HT_WIDTH_64);
		max_pages = NumericLimits<uint32_t>::Maximum();
		max_tuples = NumericLimits<uint16_t>::Maximum();
		break;
	}

	return max_pages * MinValue(max_tuples, (idx_t)Storage::BLOCK_SIZE / tuple_size);
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
	Verify();

	D_ASSERT(!is_finalized);

	if (size <= capacity) {
		throw InternalException("Cannot downsize a hash table!");
	}
	D_ASSERT(size >= STANDARD_VECTOR_SIZE);

	// size needs to be a power of 2
	D_ASSERT((size & (size - 1)) == 0);
	bitmask = size - 1;

	auto byte_size = size * sizeof(ENTRY);
	if (byte_size > (idx_t)Storage::BLOCK_SIZE) {
		hashes_hdl = buffer_manager.Allocate(byte_size);
		hashes_hdl_ptr = hashes_hdl.Ptr();
	}
	memset(hashes_hdl_ptr, 0, byte_size);
	hashes_end_ptr = hashes_hdl_ptr + byte_size;
	capacity = size;

	auto hashes_arr = (ENTRY *)hashes_hdl_ptr;

	PayloadApply([&](idx_t page_nr, idx_t page_offset, data_ptr_t ptr) {
		auto hash = Load<hash_t>(ptr + hash_offset);
		D_ASSERT((hash & bitmask) == (hash % capacity));
		auto entry_idx = (idx_t)hash & bitmask;
		while (hashes_arr[entry_idx].page_nr > 0) {
			entry_idx++;
			if (entry_idx >= capacity) {
				entry_idx = 0;
			}
		}

		D_ASSERT(!hashes_arr[entry_idx].page_nr);
		D_ASSERT(hash >> hash_prefix_shift <= NumericLimits<uint16_t>::Maximum());

		hashes_arr[entry_idx].salt = hash >> hash_prefix_shift;
		hashes_arr[entry_idx].page_nr = page_nr + 1;
		hashes_arr[entry_idx].page_offset = page_offset;
	});

	Verify();
}

idx_t GroupedAggregateHashTable::AddChunk(DataChunk &groups, DataChunk &payload, AggregateType filter) {
	vector<idx_t> aggregate_filter;

	auto &aggregates = layout.GetAggregates();
	for (idx_t i = 0; i < aggregates.size(); i++) {
		auto &aggregate = aggregates[i];
		if (aggregate.aggr_type == filter) {
			aggregate_filter.push_back(i);
		}
	}
	return AddChunk(groups, payload, aggregate_filter);
}

idx_t GroupedAggregateHashTable::AddChunk(DataChunk &groups, DataChunk &payload, const vector<idx_t> &filter) {
	Vector hashes(LogicalType::HASH);
	groups.Hash(hashes);

	return AddChunk(groups, hashes, payload, filter);
}

idx_t GroupedAggregateHashTable::AddChunk(DataChunk &groups, Vector &group_hashes, DataChunk &payload,
                                          const vector<idx_t> &filter) {
	D_ASSERT(!is_finalized);

	if (groups.size() == 0) {
		return 0;
	}
	// dummy
	SelectionVector new_groups(STANDARD_VECTOR_SIZE);

	D_ASSERT(groups.ColumnCount() + 1 == layout.ColumnCount());
	for (idx_t i = 0; i < groups.ColumnCount(); i++) {
		D_ASSERT(groups.GetTypes()[i] == layout.GetTypes()[i]);
	}

	Vector addresses(LogicalType::POINTER);
	auto new_group_count = FindOrCreateGroups(groups, group_hashes, addresses, new_groups);
	VectorOperations::AddInPlace(addresses, layout.GetAggrOffset(), payload.size());

	// now every cell has an entry
	// update the aggregates
	idx_t payload_idx = 0;

	auto &aggregates = layout.GetAggregates();
	idx_t filter_idx = 0;
	for (idx_t i = 0; i < aggregates.size(); i++) {
		auto &aggr = aggregates[i];
		if (filter_idx >= filter.size() || i < filter[filter_idx]) {
			// Skip all the aggregates that are not in the filter
			payload_idx += aggr.child_count;
			VectorOperations::AddInPlace(addresses, aggr.payload_size, payload.size());
			continue;
		}
		D_ASSERT(i == filter[filter_idx]);

		if (aggr.aggr_type != AggregateType::DISTINCT && aggr.filter) {
			RowOperations::UpdateFilteredStates(filter_set.GetFilterData(i), aggr, addresses, payload, payload_idx);
		} else {
			RowOperations::UpdateStates(aggr, addresses, payload, payload_idx, payload.size());
		}

		// move to the next aggregate
		payload_idx += aggr.child_count;
		VectorOperations::AddInPlace(addresses, aggr.payload_size, payload.size());
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
	Vector addresses(LogicalType::POINTER);
	FindOrCreateGroups(groups, addresses);
	// now fetch the aggregates
	RowOperations::FinalizeStates(layout, addresses, result, 0);
}

template <class ENTRY>
idx_t GroupedAggregateHashTable::FindOrCreateGroupsInternal(DataChunk &groups, Vector &group_hashes, Vector &addresses,
                                                            SelectionVector &new_groups_out) {
	D_ASSERT(!is_finalized);

	if (entries + groups.size() > MaxCapacity()) {
		throw InternalException("Hash table capacity reached");
	}

	// resize at 50% capacity, also need to fit the entire vector
	if (capacity - entries <= groups.size() || entries > capacity / LOAD_FACTOR) {
		Resize<ENTRY>(capacity * 2);
	}

	D_ASSERT(capacity - entries >= groups.size());
	D_ASSERT(groups.ColumnCount() + 1 == layout.ColumnCount());
	// we need to be able to fit at least one vector of data
	D_ASSERT(capacity - entries >= groups.size());
	D_ASSERT(group_hashes.GetType() == LogicalType::HASH);

	group_hashes.Flatten(groups.size());
	auto group_hashes_ptr = FlatVector::GetData<hash_t>(group_hashes);

	D_ASSERT(ht_offsets.GetVectorType() == VectorType::FLAT_VECTOR);
	D_ASSERT(ht_offsets.GetType() == LogicalType::BIGINT);

	D_ASSERT(addresses.GetType() == LogicalType::POINTER);
	addresses.Flatten(groups.size());
	auto addresses_ptr = FlatVector::GetData<data_ptr_t>(addresses);

	// now compute the entry in the table based on the hash using a modulo
	UnaryExecutor::Execute<hash_t, uint64_t>(group_hashes, ht_offsets, groups.size(), [&](hash_t element) {
		D_ASSERT((element & bitmask) == (element % capacity));
		return (element & bitmask);
	});
	auto ht_offsets_ptr = FlatVector::GetData<uint64_t>(ht_offsets);

	// precompute the hash salts for faster comparison below
	D_ASSERT(hash_salts.GetType() == LogicalType::SMALLINT);
	UnaryExecutor::Execute<hash_t, uint16_t>(group_hashes, hash_salts, groups.size(),
	                                         [&](hash_t element) { return (element >> hash_prefix_shift); });
	auto hash_salts_ptr = FlatVector::GetData<uint16_t>(hash_salts);

	// we start out with all entries [0, 1, 2, ..., groups.size()]
	const SelectionVector *sel_vector = FlatVector::IncrementalSelectionVector();

	idx_t remaining_entries = groups.size();

	// make a chunk that references the groups and the hashes
	DataChunk group_chunk;
	group_chunk.InitializeEmpty(layout.GetTypes());
	for (idx_t grp_idx = 0; grp_idx < groups.ColumnCount(); grp_idx++) {
		group_chunk.data[grp_idx].Reference(groups.data[grp_idx]);
	}
	group_chunk.data[groups.ColumnCount()].Reference(group_hashes);
	group_chunk.SetCardinality(groups);

	// convert all vectors to unified format
	auto group_data = group_chunk.ToUnifiedFormat();

	idx_t new_group_count = 0;
	while (remaining_entries > 0) {
		idx_t new_entry_count = 0;
		idx_t need_compare_count = 0;
		idx_t no_match_count = 0;

		// first figure out for each remaining whether or not it belongs to a full or empty group
		for (idx_t i = 0; i < remaining_entries; i++) {
			const idx_t index = sel_vector->get_index(i);
			const auto ht_entry_ptr = ((ENTRY *)this->hashes_hdl_ptr) + ht_offsets_ptr[index];
			if (ht_entry_ptr->page_nr == 0) { // we use page number 0 as a "unused marker"
				// cell is empty; setup the new entry
				if (payload_page_offset == tuples_per_block || payload_hds.empty()) {
					NewBlock();
				}

				auto entry_payload_ptr = payload_hds_ptrs.back() + (payload_page_offset * tuple_size);

				D_ASSERT(group_hashes_ptr[index] >> hash_prefix_shift <= NumericLimits<uint16_t>::Maximum());
				D_ASSERT(payload_page_offset < tuples_per_block);
				D_ASSERT(payload_hds.size() < NumericLimits<uint32_t>::Maximum());
				D_ASSERT(payload_page_offset + 1 < NumericLimits<uint16_t>::Maximum());

				ht_entry_ptr->salt = group_hashes_ptr[index] >> hash_prefix_shift;

				// page numbers start at one so we can use 0 as empty flag
				// GetPtr undoes this
				ht_entry_ptr->page_nr = payload_hds.size();
				ht_entry_ptr->page_offset = payload_page_offset++;

				// update selection lists for outer loops
				empty_vector.set_index(new_entry_count++, index);
				new_groups_out.set_index(new_group_count++, index);
				entries++;

				addresses_ptr[index] = entry_payload_ptr;

			} else {
				// cell is occupied: add to check list
				// only need to check if hash salt in ptr == prefix of hash in payload
				if (ht_entry_ptr->salt == hash_salts_ptr[index]) {
					group_compare_vector.set_index(need_compare_count++, index);

					auto page_ptr = payload_hds_ptrs[ht_entry_ptr->page_nr - 1];
					auto page_offset = ht_entry_ptr->page_offset * tuple_size;
					addresses_ptr[index] = page_ptr + page_offset;

				} else {
					no_match_vector.set_index(no_match_count++, index);
				}
			}
		}

		// for each of the locations that are empty, serialize the group columns to the locations
		RowOperations::Scatter(group_chunk, group_data.get(), layout, addresses, *string_heap, empty_vector,
		                       new_entry_count);
		RowOperations::InitializeStates(layout, addresses, empty_vector, new_entry_count);

		// now we have only the tuples remaining that might match to an existing group
		// start performing comparisons with each of the groups
		RowOperations::Match(group_chunk, group_data.get(), layout, addresses, predicates, group_compare_vector,
		                     need_compare_count, &no_match_vector, no_match_count);

		// each of the entries that do not match we move them to the next entry in the HT
		for (idx_t i = 0; i < no_match_count; i++) {
			idx_t index = no_match_vector.get_index(i);
			ht_offsets_ptr[index]++;
			if (ht_offsets_ptr[index] >= capacity) {
				ht_offsets_ptr[index] = 0;
			}
		}
		sel_vector = &no_match_vector;
		remaining_entries = no_match_count;
	}

	return new_group_count;
}

// this is to support distinct aggregations where we need to record whether we
// have already seen a value for a group
idx_t GroupedAggregateHashTable::FindOrCreateGroups(DataChunk &groups, Vector &group_hashes, Vector &addresses_out,
                                                    SelectionVector &new_groups_out) {
	switch (entry_type) {
	case HtEntryType::HT_WIDTH_64:
		return FindOrCreateGroupsInternal<aggr_ht_entry_64>(groups, group_hashes, addresses_out, new_groups_out);
	case HtEntryType::HT_WIDTH_32:
		return FindOrCreateGroupsInternal<aggr_ht_entry_32>(groups, group_hashes, addresses_out, new_groups_out);
	default:
		throw InternalException("Unknown HT entry width");
	}
}

void GroupedAggregateHashTable::FindOrCreateGroups(DataChunk &groups, Vector &addresses) {
	// create a dummy new_groups sel vector
	SelectionVector new_groups(STANDARD_VECTOR_SIZE);
	FindOrCreateGroups(groups, addresses, new_groups);
}

idx_t GroupedAggregateHashTable::FindOrCreateGroups(DataChunk &groups, Vector &addresses_out,
                                                    SelectionVector &new_groups_out) {
	Vector hashes(LogicalType::HASH);
	groups.Hash(hashes);
	return FindOrCreateGroups(groups, hashes, addresses_out, new_groups_out);
}

struct FlushMoveState {
	FlushMoveState(Allocator &allocator, RowLayout &layout)
	    : new_groups(STANDARD_VECTOR_SIZE), group_addresses(LogicalType::POINTER),
	      new_groups_sel(STANDARD_VECTOR_SIZE) {
		vector<LogicalType> group_types(layout.GetTypes().begin(), layout.GetTypes().end() - 1);
		groups.Initialize(allocator, group_types);
	}

	DataChunk groups;
	SelectionVector new_groups;
	Vector group_addresses;
	SelectionVector new_groups_sel;
};

void GroupedAggregateHashTable::FlushMove(FlushMoveState &state, Vector &source_addresses, Vector &source_hashes,
                                          idx_t count) {
	D_ASSERT(source_addresses.GetType() == LogicalType::POINTER);
	D_ASSERT(source_hashes.GetType() == LogicalType::HASH);

	state.groups.Reset();
	state.groups.SetCardinality(count);
	for (idx_t col_no = 0; col_no < state.groups.ColumnCount(); col_no++) {
		auto &column = state.groups.data[col_no];
		RowOperations::Gather(source_addresses, *FlatVector::IncrementalSelectionVector(), column,
		                      *FlatVector::IncrementalSelectionVector(), count, layout, col_no);
	}

	FindOrCreateGroups(state.groups, source_hashes, state.group_addresses, state.new_groups_sel);

	RowOperations::CombineStates(layout, source_addresses, state.group_addresses, count);
}

void GroupedAggregateHashTable::Combine(GroupedAggregateHashTable &other) {
	D_ASSERT(!is_finalized);

	D_ASSERT(other.layout.GetAggrWidth() == layout.GetAggrWidth());
	D_ASSERT(other.layout.GetDataWidth() == layout.GetDataWidth());
	D_ASSERT(other.layout.GetRowWidth() == layout.GetRowWidth());
	D_ASSERT(other.tuples_per_block == tuples_per_block);

	if (other.entries == 0) {
		return;
	}

	Vector addresses(LogicalType::POINTER);
	auto addresses_ptr = FlatVector::GetData<data_ptr_t>(addresses);

	Vector hashes(LogicalType::HASH);
	auto hashes_ptr = FlatVector::GetData<hash_t>(hashes);

	idx_t group_idx = 0;

	FlushMoveState state(allocator, layout);
	other.PayloadApply([&](idx_t page_nr, idx_t page_offset, data_ptr_t ptr) {
		auto hash = Load<hash_t>(ptr + hash_offset);

		hashes_ptr[group_idx] = hash;
		addresses_ptr[group_idx] = ptr;
		group_idx++;
		if (group_idx == STANDARD_VECTOR_SIZE) {
			FlushMove(state, addresses, hashes, group_idx);
			group_idx = 0;
		}
	});
	FlushMove(state, addresses, hashes, group_idx);
	string_heap->Merge(*other.string_heap);
	Verify();
}

struct PartitionInfo {
	PartitionInfo() : addresses(LogicalType::POINTER), hashes(LogicalType::HASH), group_count(0) {
		addresses_ptr = FlatVector::GetData<data_ptr_t>(addresses);
		hashes_ptr = FlatVector::GetData<hash_t>(hashes);
	};
	Vector addresses;
	Vector hashes;
	idx_t group_count;
	data_ptr_t *addresses_ptr;
	hash_t *hashes_ptr;
};

void GroupedAggregateHashTable::Partition(vector<GroupedAggregateHashTable *> &partition_hts, hash_t mask,
                                          idx_t shift) {
	D_ASSERT(partition_hts.size() > 1);
	vector<PartitionInfo> partition_info(partition_hts.size());

	FlushMoveState state(allocator, layout);
	PayloadApply([&](idx_t page_nr, idx_t page_offset, data_ptr_t ptr) {
		auto hash = Load<hash_t>(ptr + hash_offset);

		idx_t partition = (hash & mask) >> shift;
		D_ASSERT(partition < partition_hts.size());

		auto &info = partition_info[partition];

		info.hashes_ptr[info.group_count] = hash;
		info.addresses_ptr[info.group_count] = ptr;
		info.group_count++;
		if (info.group_count == STANDARD_VECTOR_SIZE) {
			D_ASSERT(partition_hts[partition]);
			partition_hts[partition]->FlushMove(state, info.addresses, info.hashes, info.group_count);
			info.group_count = 0;
		}
	});

	idx_t info_idx = 0;
	idx_t total_count = 0;
	for (auto &partition_entry : partition_hts) {
		auto &info = partition_info[info_idx++];
		partition_entry->FlushMove(state, info.addresses, info.hashes, info.group_count);

		partition_entry->string_heap->Merge(*string_heap);
		partition_entry->Verify();
		total_count += partition_entry->Size();
	}
	(void)total_count;
	D_ASSERT(total_count == entries);
}

idx_t GroupedAggregateHashTable::Scan(AggregateHTScanState &scan_state, DataChunk &result) {
	idx_t this_n;
	Vector addresses(LogicalType::POINTER);
	auto data_pointers = FlatVector::GetData<data_ptr_t>(addresses);
	{
		lock_guard<mutex> l(scan_state.lock);
		if (scan_state.scan_position >= entries) {
			return 0;
		}
		auto remaining = entries - scan_state.scan_position;
		this_n = MinValue((idx_t)STANDARD_VECTOR_SIZE, remaining);

		auto chunk_idx = scan_state.scan_position / tuples_per_block;
		auto chunk_offset = (scan_state.scan_position % tuples_per_block) * tuple_size;
		D_ASSERT(chunk_offset + tuple_size <= Storage::BLOCK_SIZE);

		auto read_ptr = payload_hds_ptrs[chunk_idx++];
		for (idx_t i = 0; i < this_n; i++) {
			data_pointers[i] = read_ptr + chunk_offset;
			chunk_offset += tuple_size;
			if (chunk_offset >= tuples_per_block * tuple_size) {
				read_ptr = payload_hds_ptrs[chunk_idx++];
				chunk_offset = 0;
			}
		}
		scan_state.scan_position += this_n;
	}

	result.SetCardinality(this_n);
	// fetch the group columns (ignoring the final hash column
	const auto group_cols = layout.ColumnCount() - 1;
	for (idx_t col_no = 0; col_no < group_cols; col_no++) {
		auto &column = result.data[col_no];
		RowOperations::Gather(addresses, *FlatVector::IncrementalSelectionVector(), column,
		                      *FlatVector::IncrementalSelectionVector(), result.size(), layout, col_no);
	}

	RowOperations::FinalizeStates(layout, addresses, result, group_cols);
	return this_n;
}

void GroupedAggregateHashTable::Finalize() {
	if (is_finalized) {
		return;
	}

	// early release hashes, not needed for partition/scan
	hashes_hdl.Destroy();
	is_finalized = true;
}

} // namespace duckdb




namespace duckdb {

BaseAggregateHashTable::BaseAggregateHashTable(ClientContext &context, Allocator &allocator,
                                               const vector<AggregateObject> &aggregates,
                                               vector<LogicalType> payload_types_p)
    : allocator(allocator), buffer_manager(BufferManager::GetBufferManager(context)),
      payload_types(move(payload_types_p)) {
	filter_set.Initialize(context, aggregates, payload_types);
}

} // namespace duckdb












namespace duckdb {

ColumnBindingResolver::ColumnBindingResolver() {
}

void ColumnBindingResolver::VisitOperator(LogicalOperator &op) {
	if (op.type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN || op.type == LogicalOperatorType::LOGICAL_DELIM_JOIN) {
		// special case: comparison join
		auto &comp_join = (LogicalComparisonJoin &)op;
		// first get the bindings of the LHS and resolve the LHS expressions
		VisitOperator(*comp_join.children[0]);
		for (auto &cond : comp_join.conditions) {
			VisitExpression(&cond.left);
		}
		if (op.type == LogicalOperatorType::LOGICAL_DELIM_JOIN) {
			// visit the duplicate eliminated columns on the LHS, if any
			auto &delim_join = (LogicalDelimJoin &)op;
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
	} else if (op.type == LogicalOperatorType::LOGICAL_ANY_JOIN) {
		// ANY join, this join is different because we evaluate the expression on the bindings of BOTH join sides at
		// once i.e. we set the bindings first to the bindings of the entire join, and then resolve the expressions of
		// this operator
		VisitOperatorChildren(op);
		bindings = op.GetColumnBindings();
		VisitOperatorExpressions(op);
		return;
	} else if (op.type == LogicalOperatorType::LOGICAL_CREATE_INDEX) {
		// CREATE INDEX statement, add the columns of the table with table index 0 to the binding set
		// afterwards bind the expressions of the CREATE INDEX statement
		auto &create_index = (LogicalCreateIndex &)op;
		bindings = LogicalOperator::GenerateColumnBindings(0, create_index.table.columns.LogicalColumnCount());
		VisitOperatorExpressions(op);
		return;
	} else if (op.type == LogicalOperatorType::LOGICAL_GET) {
		//! We first need to update the current set of bindings and then visit operator expressions
		bindings = op.GetColumnBindings();
		VisitOperatorExpressions(op);
		return;
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
			return make_unique<BoundReferenceExpression>(expr.alias, expr.return_type, i);
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
	default:
		throw InvalidTypeException(input.GetType(), "Invalid type for BETWEEN");
	}
}

unique_ptr<ExpressionState> ExpressionExecutor::InitializeState(const BoundBetweenExpression &expr,
                                                                ExpressionExecutorState &root) {
	auto result = make_unique<ExpressionState>(expr, root);
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
	auto result = make_unique<CaseExpressionState>(expr, root);
	for (auto &case_check : expr.case_checks) {
		result->AddChild(case_check.when_expr.get());
		result->AddChild(case_check.then_expr.get());
	}
	result->AddChild(expr.else_expr.get());
	result->Finalize();
	return move(result);
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
	auto result = make_unique<ExpressionState>(expr, root);
	result->AddChild(expr.child.get());
	result->Finalize();
	return result;
}

void ExpressionExecutor::Execute(const BoundCastExpression &expr, ExpressionState *state, const SelectionVector *sel,
                                 idx_t count, Vector &result) {
	// resolve the child
	state->intermediate_chunk.Reset();

	auto &child = state->intermediate_chunk.data[0];
	auto child_state = state->child_states[0].get();

	Execute(*expr.child, child_state, sel, count, child);
	if (expr.try_cast) {
		string error_message;
		CastParameters parameters(expr.bound_cast.cast_data.get(), false, &error_message);
		expr.bound_cast.function(child, result, count, parameters);
	} else {
		// cast it to the type specified by the cast expression
		D_ASSERT(result.GetType() == expr.return_type);
		CastParameters parameters(expr.bound_cast.cast_data.get(), false, nullptr);
		expr.bound_cast.function(child, result, count, parameters);
	}
}

} // namespace duckdb






#include <algorithm>

namespace duckdb {

unique_ptr<ExpressionState> ExpressionExecutor::InitializeState(const BoundComparisonExpression &expr,
                                                                ExpressionExecutorState &root) {
	auto result = make_unique<ExpressionState>(expr, root);
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
	case PhysicalType::MAP:
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
	return TemplatedSelectOperation<duckdb::LessThan>(left, right, sel, count, true_sel, false_sel);
}

idx_t VectorOperations::LessThanEquals(Vector &left, Vector &right, const SelectionVector *sel, idx_t count,
                                       SelectionVector *true_sel, SelectionVector *false_sel) {
	return TemplatedSelectOperation<duckdb::LessThanEquals>(left, right, sel, count, true_sel, false_sel);
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
		adaptive_filter = make_unique<AdaptiveFilter>(expr);
	}
	unique_ptr<AdaptiveFilter> adaptive_filter;
};

unique_ptr<ExpressionState> ExpressionExecutor::InitializeState(const BoundConjunctionExpression &expr,
                                                                ExpressionExecutorState &root) {
	auto result = make_unique<ConjunctionState>(expr, root);
	for (auto &child : expr.children) {
		result->AddChild(child.get());
	}
	result->Finalize();
	return move(result);
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
			temp_false = make_unique<SelectionVector>(STANDARD_VECTOR_SIZE);
		}
		if (!true_sel) {
			temp_true = make_unique<SelectionVector>(STANDARD_VECTOR_SIZE);
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
			temp_true = make_unique<SelectionVector>(STANDARD_VECTOR_SIZE);
		}
		if (!false_sel) {
			temp_false = make_unique<SelectionVector>(STANDARD_VECTOR_SIZE);
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
	auto result = make_unique<ExpressionState>(expr, root);
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
	auto result = make_unique<ExecuteFunctionState>(expr, root);
	for (auto &child : expr.children) {
		result->AddChild(child.get());
	}
	result->Finalize();
	if (expr.function.init_local_state) {
		result->local_state = expr.function.init_local_state(*result, expr, expr.bind_info.get());
	}
	return move(result);
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
	auto result = make_unique<ExpressionState>(expr, root);
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
			throw Exception("IN needs at least two children");
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
	auto result = make_unique<ExpressionState>(expr, root);
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
	auto result = make_unique<ExpressionState>(expr, root);
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
	auto state = make_unique<ExpressionExecutorState>(expr.ToString());
	Initialize(expr, *state);
	state->Verify();
	states.push_back(move(state));
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
		return Select((BoundBetweenExpression &)expr, state, sel, count, true_sel, false_sel);
	case ExpressionClass::BOUND_COMPARISON:
		return Select((BoundComparisonExpression &)expr, state, sel, count, true_sel, false_sel);
	case ExpressionClass::BOUND_CONJUNCTION:
		return Select((BoundConjunctionExpression &)expr, state, sel, count, true_sel, false_sel);
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
		throw BinderException("Cannot use %s in this context", ((BoundFunctionExpression &)expr).function.name);
	}
	return root.executor->GetContext();
}

ExpressionState::ExpressionState(const Expression &expr, ExpressionExecutorState &root)
    : expr(expr), root(root), name(expr.ToString()) {
}

ExpressionExecutorState::ExpressionExecutorState(const string &name) : profiler(), name(name) {
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
#include <ctgmath>

namespace duckdb {

ART::ART(const vector<column_t> &column_ids, TableIOManager &table_io_manager,
         const vector<unique_ptr<Expression>> &unbound_expressions, IndexConstraintType constraint_type,
         DatabaseInstance &db, idx_t block_id, idx_t block_offset)
    : Index(IndexType::ART, table_io_manager, column_ids, unbound_expressions, constraint_type), db(db),
      estimated_art_size(0), estimated_key_size(16) {
	if (!Radix::IsLittleEndian()) {
		throw NotImplementedException("ART indexes are not supported on big endian architectures");
	}
	if (block_id != DConstants::INVALID_INDEX) {
		tree = Node::Deserialize(*this, block_id, block_offset);
	} else {
		tree = nullptr;
	}
	serialized_data_pointer = BlockPointer(block_id, block_offset);
	for (idx_t i = 0; i < types.size(); i++) {
		switch (types[i]) {
		case PhysicalType::BOOL:
		case PhysicalType::INT8:
		case PhysicalType::UINT8:
			estimated_key_size += sizeof(int8_t);
			break;
		case PhysicalType::INT16:
		case PhysicalType::UINT16:
			estimated_key_size += sizeof(int16_t);
			break;
		case PhysicalType::INT32:
		case PhysicalType::UINT32:
		case PhysicalType::FLOAT:
			estimated_key_size += sizeof(int32_t);
			break;
		case PhysicalType::INT64:
		case PhysicalType::UINT64:
		case PhysicalType::DOUBLE:
			estimated_key_size += sizeof(int64_t);
			break;
		case PhysicalType::INT128:
			estimated_key_size += sizeof(hugeint_t);
			break;
		case PhysicalType::VARCHAR:
			estimated_key_size += 16; // oh well
			break;
		default:
			throw InvalidTypeException(logical_types[i], "Invalid type for index");
		}
	}
}

ART::~ART() {
	if (estimated_art_size > 0) {
		BufferManager::GetBufferManager(db).FreeReservedMemory(estimated_art_size);
		estimated_art_size = 0;
	}
	if (tree) {
		Node::Delete(tree);
		tree = nullptr;
	}
}

unique_ptr<IndexScanState> ART::InitializeScanSinglePredicate(Transaction &transaction, Value value,
                                                              ExpressionType expression_type) {
	auto result = make_unique<ARTIndexScanState>();
	result->values[0] = value;
	result->expressions[0] = expression_type;
	return move(result);
}

unique_ptr<IndexScanState> ART::InitializeScanTwoPredicates(Transaction &transaction, Value low_value,
                                                            ExpressionType low_expression_type, Value high_value,
                                                            ExpressionType high_expression_type) {
	auto result = make_unique<ARTIndexScanState>();
	result->values[0] = low_value;
	result->expressions[0] = low_expression_type;
	result->values[1] = high_value;
	result->expressions[1] = high_expression_type;
	return move(result);
}

//===--------------------------------------------------------------------===//
// Keys
//===--------------------------------------------------------------------===//

template <class T>
static void TemplatedGenerateKeys(ArenaAllocator &allocator, Vector &input, idx_t count, vector<Key> &keys) {
	UnifiedVectorFormat idata;
	input.ToUnifiedFormat(count, idata);

	D_ASSERT(keys.size() >= count);
	auto input_data = (T *)idata.data;
	for (idx_t i = 0; i < count; i++) {
		auto idx = idata.sel->get_index(i);
		if (idata.validity.RowIsValid(idx)) {
			Key::CreateKey<T>(allocator, keys[i], input_data[idx]);
		}
	}
}

template <class T>
static void ConcatenateKeys(ArenaAllocator &allocator, Vector &input, idx_t count, vector<Key> &keys) {
	UnifiedVectorFormat idata;
	input.ToUnifiedFormat(count, idata);

	auto input_data = (T *)idata.data;
	for (idx_t i = 0; i < count; i++) {
		auto idx = idata.sel->get_index(i);

		// key is not NULL (no previous column entry was NULL)
		if (!keys[i].Empty()) {
			if (!idata.validity.RowIsValid(idx)) {
				// this column entry is NULL, set whole key to NULL
				keys[i] = Key();
			} else {
				auto other_key = Key::CreateKey<T>(allocator, input_data[idx]);
				keys[i].ConcatenateKey(allocator, other_key);
			}
		}
	}
}

void ART::GenerateKeys(ArenaAllocator &allocator, DataChunk &input, vector<Key> &keys) {
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
// Insert
//===--------------------------------------------------------------------===//

struct KeySection {
	KeySection(idx_t start_p, idx_t end_p, idx_t depth_p, data_t key_byte_p)
	    : start(start_p), end(end_p), depth(depth_p), key_byte(key_byte_p) {};
	KeySection(idx_t start_p, idx_t end_p, vector<Key> &keys, KeySection &key_section)
	    : start(start_p), end(end_p), depth(key_section.depth + 1), key_byte(keys[end_p].data[key_section.depth]) {};
	idx_t start;
	idx_t end;
	idx_t depth;
	data_t key_byte;
};

void GetChildSections(vector<KeySection> &child_sections, vector<Key> &keys, KeySection &key_section) {

	idx_t child_start_idx = key_section.start;
	for (idx_t i = key_section.start + 1; i <= key_section.end; i++) {
		if (keys[i - 1].data[key_section.depth] != keys[i].data[key_section.depth]) {
			child_sections.emplace_back(child_start_idx, i - 1, keys, key_section);
			child_start_idx = i;
		}
	}
	child_sections.emplace_back(child_start_idx, key_section.end, keys, key_section);
}

void Construct(vector<Key> &keys, row_t *row_ids, Node *&node, KeySection &key_section, bool &has_constraint) {

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
		if (has_constraint && num_row_ids != 1) {
			throw ConstraintException("New data contains duplicates on indexed column(s)");
		}

		node = Leaf::New(start_key, prefix_start, row_ids + key_section.start, num_row_ids);
	} else { // create a new node and recurse

		// we will find at least two child entries of this node, otherwise we'd have reached a leaf
		vector<KeySection> child_sections;
		GetChildSections(child_sections, keys, key_section);

		auto node_type = Node::GetTypeBySize(child_sections.size());
		Node::New(node_type, node);

		auto prefix_length = key_section.depth - prefix_start;
		node->prefix = Prefix(start_key, prefix_start, prefix_length);

		// recurse on each child section
		for (auto &child_section : child_sections) {
			Node *new_child = nullptr;
			Construct(keys, row_ids, new_child, child_section, has_constraint);
			Node::InsertChild(node, child_section.key_byte, new_child);
		}
	}
}

void ART::ConstructAndMerge(IndexLock &lock, PayloadScanner &scanner, Allocator &allocator) {

	auto payload_types = logical_types;
	payload_types.emplace_back(LogicalType::ROW_TYPE);

	ArenaAllocator arena_allocator(BufferAllocator::Get(db));
	vector<Key> keys(STANDARD_VECTOR_SIZE);

	auto temp_art = make_unique<ART>(this->column_ids, this->table_io_manager, this->unbound_expressions,
	                                 this->constraint_type, this->db);

	for (;;) {
		DataChunk ordered_chunk;
		ordered_chunk.Initialize(allocator, payload_types);
		ordered_chunk.SetCardinality(0);
		scanner.Scan(ordered_chunk);
		if (ordered_chunk.size() == 0) {
			break;
		}

		// get the key chunk and the row_identifiers vector
		DataChunk row_id_chunk;
		ordered_chunk.Split(row_id_chunk, ordered_chunk.ColumnCount() - 1);
		auto &row_identifiers = row_id_chunk.data[0];

		D_ASSERT(row_identifiers.GetType().InternalType() == ROW_TYPE);
		D_ASSERT(logical_types[0] == ordered_chunk.data[0].GetType());

		// generate the keys for the given input
		arena_allocator.Reset();
		GenerateKeys(arena_allocator, ordered_chunk, keys);

		// prepare the row_identifiers
		row_identifiers.Flatten(ordered_chunk.size());
		auto row_ids = FlatVector::GetData<row_t>(row_identifiers);

		// construct the ART of this chunk
		auto art = make_unique<ART>(this->column_ids, this->table_io_manager, this->unbound_expressions,
		                            this->constraint_type, this->db);
		auto key_section = KeySection(0, ordered_chunk.size() - 1, 0, 0);
		auto has_constraint = IsUnique();
		Construct(keys, row_ids, art->tree, key_section, has_constraint);

		// merge art into temp_art
		if (!temp_art->MergeIndexes(lock, art.get())) {
			throw ConstraintException("Data contains duplicates on indexed column(s)");
		}
	}

	// NOTE: currently this code is only used for index creation, so we can assume that there are no
	// duplicate violations between the existing index and the new data,
	// so we do not need to revert any changes
	if (!this->MergeIndexes(lock, temp_art.get())) {
		throw ConstraintException("Data contains duplicates on indexed column(s)");
	}
}

bool ART::Insert(IndexLock &lock, DataChunk &input, Vector &row_ids) {
	D_ASSERT(row_ids.GetType().InternalType() == ROW_TYPE);
	D_ASSERT(logical_types[0] == input.data[0].GetType());

	// generate the keys for the given input
	ArenaAllocator arena_allocator(BufferAllocator::Get(db));
	vector<Key> keys(input.size());
	GenerateKeys(arena_allocator, input, keys);

	idx_t extra_memory = estimated_key_size * input.size();
	BufferManager::GetBufferManager(db).ReserveMemory(extra_memory);
	estimated_art_size += extra_memory;

	// now insert the elements into the index
	row_ids.Flatten(input.size());
	auto row_identifiers = FlatVector::GetData<row_t>(row_ids);
	idx_t failed_index = DConstants::INVALID_INDEX;
	for (idx_t i = 0; i < input.size(); i++) {
		if (keys[i].Empty()) {
			continue;
		}

		row_t row_id = row_identifiers[i];
		if (!Insert(tree, keys[i], 0, row_id)) {
			// failed to insert because of constraint violation
			failed_index = i;
			break;
		}
	}
	if (failed_index != DConstants::INVALID_INDEX) {

		// failed to insert because of constraint violation: remove previously inserted entries
		for (idx_t i = 0; i < failed_index; i++) {
			if (keys[i].Empty()) {
				continue;
			}
			row_t row_id = row_identifiers[i];
			Erase(tree, keys[i], 0, row_id);
		}
		return false;
	}
	return true;
}

bool ART::Append(IndexLock &lock, DataChunk &appended_data, Vector &row_identifiers) {
	DataChunk expression_result;
	expression_result.Initialize(Allocator::DefaultAllocator(), logical_types);

	// first resolve the expressions for the index
	ExecuteExpressions(appended_data, expression_result);

	// now insert into the index
	return Insert(lock, expression_result, row_identifiers);
}

void ART::VerifyAppend(DataChunk &chunk) {
	VerifyExistence(chunk, VerifyExistenceType::APPEND);
}

void ART::VerifyAppendForeignKey(DataChunk &chunk, string *err_msg_ptr) {
	VerifyExistence(chunk, VerifyExistenceType::APPEND_FK, err_msg_ptr);
}

void ART::VerifyDeleteForeignKey(DataChunk &chunk, string *err_msg_ptr) {
	VerifyExistence(chunk, VerifyExistenceType::DELETE_FK, err_msg_ptr);
}

bool ART::InsertToLeaf(Leaf &leaf, row_t row_id) {
#ifdef DEBUG
	for (idx_t k = 0; k < leaf.count; k++) {
		D_ASSERT(leaf.GetRowId(k) != row_id);
	}
#endif
	if (IsUnique() && leaf.count != 0) {
		return false;
	}
	leaf.Insert(row_id);
	return true;
}

bool ART::Insert(Node *&node, Key &key, idx_t depth, row_t row_id) {

	if (!node) {
		// node is currently empty, create a leaf here with the key
		node = Leaf::New(key, depth, row_id);
		return true;
	}

	if (node->type == NodeType::NLeaf) {
		// Replace leaf with Node4 and store both leaves in it
		auto leaf = (Leaf *)node;

		auto &leaf_prefix = leaf->prefix;
		uint32_t new_prefix_length = 0;

		// Leaf node is already there (its key matches the current key), update row_id vector
		if (new_prefix_length == leaf->prefix.Size() && depth + leaf->prefix.Size() == key.len) {
			return InsertToLeaf(*leaf, row_id);
		}
		while (leaf_prefix[new_prefix_length] == key[depth + new_prefix_length]) {
			new_prefix_length++;
			// Leaf node is already there (its key matches the current key), update row_id vector
			if (new_prefix_length == leaf->prefix.Size() && depth + leaf->prefix.Size() == key.len) {
				return InsertToLeaf(*leaf, row_id);
			}
		}

		Node *new_node = Node4::New();
		new_node->prefix = Prefix(key, depth, new_prefix_length);
		auto key_byte = node->prefix.Reduce(new_prefix_length);
		Node4::InsertChild(new_node, key_byte, node);
		Node *leaf_node = Leaf::New(key, depth + new_prefix_length + 1, row_id);
		Node4::InsertChild(new_node, key[depth + new_prefix_length], leaf_node);
		node = new_node;
		return true;
	}

	// Handle prefix of inner node
	if (node->prefix.Size()) {
		uint32_t mismatch_pos = node->prefix.KeyMismatchPosition(key, depth);
		if (mismatch_pos != node->prefix.Size()) {
			// Prefix differs, create new node
			Node *new_node = Node4::New();
			new_node->prefix = Prefix(key, depth, mismatch_pos);
			// Break up prefix
			auto key_byte = node->prefix.Reduce(mismatch_pos);
			Node4::InsertChild(new_node, key_byte, node);

			Node *leaf_node = Leaf::New(key, depth + mismatch_pos + 1, row_id);
			Node4::InsertChild(new_node, key[depth + mismatch_pos], leaf_node);
			node = new_node;
			return true;
		}
		depth += node->prefix.Size();
	}

	// Recurse
	D_ASSERT(depth < key.len);
	idx_t pos = node->GetChildPos(key[depth]);
	if (pos != DConstants::INVALID_INDEX) {
		auto child = node->GetChild(*this, pos);
		bool insertion_result = Insert(child, key, depth + 1, row_id);
		node->ReplaceChildPointer(pos, child);
		return insertion_result;
	}
	Node *new_node = Leaf::New(key, depth + 1, row_id);
	Node::InsertChild(node, key[depth], new_node);
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

	idx_t released_memory = MinValue<idx_t>(estimated_art_size, estimated_key_size * input.size());
	BufferManager::GetBufferManager(db).FreeReservedMemory(released_memory);
	estimated_art_size -= released_memory;

	// then generate the keys for the given input
	ArenaAllocator arena_allocator(BufferAllocator::Get(db));
	vector<Key> keys(expression.size());
	GenerateKeys(arena_allocator, expression, keys);

	// now erase the elements from the database
	row_ids.Flatten(input.size());
	auto row_identifiers = FlatVector::GetData<row_t>(row_ids);

	for (idx_t i = 0; i < input.size(); i++) {
		if (keys[i].Empty()) {
			continue;
		}
		Erase(tree, keys[i], 0, row_identifiers[i]);
#ifdef DEBUG
		auto node = Lookup(tree, keys[i], 0);
		if (node) {
			auto leaf = static_cast<Leaf *>(node);
			for (idx_t k = 0; k < leaf->count; k++) {
				D_ASSERT(leaf->GetRowId(k) != row_identifiers[i]);
			}
		}
#endif
	}
}

void ART::Erase(Node *&node, Key &key, idx_t depth, row_t row_id) {
	if (!node) {
		return;
	}
	// Delete a leaf from a tree
	if (node->type == NodeType::NLeaf) {
		// Make sure we have the right leaf
		auto leaf = static_cast<Leaf *>(node);
		leaf->Remove(row_id);
		if (leaf->count == 0) {
			Node::Delete(node);
			node = nullptr;
		}

		return;
	}

	// Handle prefix
	if (node->prefix.Size()) {
		if (node->prefix.KeyMismatchPosition(key, depth) != node->prefix.Size()) {
			return;
		}
		depth += node->prefix.Size();
	}
	idx_t pos = node->GetChildPos(key[depth]);
	if (pos != DConstants::INVALID_INDEX) {
		auto child = node->GetChild(*this, pos);
		D_ASSERT(child);

		if (child->type == NodeType::NLeaf) {
			// Leaf found, remove entry
			auto leaf = (Leaf *)child;
			leaf->Remove(row_id);
			if (leaf->count == 0) {
				// Leaf is empty, delete leaf, decrement node counter and maybe shrink node
				Node::EraseChild(node, pos, *this);
			}
		} else {
			// Recurse
			Erase(child, key, depth + 1, row_id);
			node->ReplaceChildPointer(pos, child);
		}
	}
}

//===--------------------------------------------------------------------===//
// Point Query
//===--------------------------------------------------------------------===//
static Key CreateKey(ArenaAllocator &allocator, PhysicalType type, Value &value) {
	D_ASSERT(type == value.type().InternalType());
	switch (type) {
	case PhysicalType::BOOL:
		return Key::CreateKey<bool>(allocator, value);
	case PhysicalType::INT8:
		return Key::CreateKey<int8_t>(allocator, value);
	case PhysicalType::INT16:
		return Key::CreateKey<int16_t>(allocator, value);
	case PhysicalType::INT32:
		return Key::CreateKey<int32_t>(allocator, value);
	case PhysicalType::INT64:
		return Key::CreateKey<int64_t>(allocator, value);
	case PhysicalType::UINT8:
		return Key::CreateKey<uint8_t>(allocator, value);
	case PhysicalType::UINT16:
		return Key::CreateKey<uint16_t>(allocator, value);
	case PhysicalType::UINT32:
		return Key::CreateKey<uint32_t>(allocator, value);
	case PhysicalType::UINT64:
		return Key::CreateKey<uint64_t>(allocator, value);
	case PhysicalType::INT128:
		return Key::CreateKey<hugeint_t>(allocator, value);
	case PhysicalType::FLOAT:
		return Key::CreateKey<float>(allocator, value);
	case PhysicalType::DOUBLE:
		return Key::CreateKey<double>(allocator, value);
	case PhysicalType::VARCHAR:
		return Key::CreateKey<string_t>(allocator, value);
	default:
		throw InternalException("Invalid type for index");
	}
}

bool ART::SearchEqual(Key &key, idx_t max_count, vector<row_t> &result_ids) {

	auto leaf = static_cast<Leaf *>(Lookup(tree, key, 0));
	if (!leaf) {
		return true;
	}
	if (leaf->count > max_count) {
		return false;
	}
	for (idx_t i = 0; i < leaf->count; i++) {
		row_t row_id = leaf->GetRowId(i);
		result_ids.push_back(row_id);
	}
	return true;
}

void ART::SearchEqualJoinNoFetch(Key &key, idx_t &result_size) {

	// we need to look for a leaf
	auto leaf = Lookup(tree, key, 0);
	if (!leaf) {
		return;
	}
	result_size = leaf->count;
}

Leaf *ART::Lookup(Node *node, Key &key, idx_t depth) {
	while (node) {
		if (node->type == NodeType::NLeaf) {
			auto leaf = (Leaf *)node;
			auto &leaf_prefix = leaf->prefix;
			//! Check leaf
			for (idx_t i = 0; i < leaf->prefix.Size(); i++) {
				if (leaf_prefix[i] != key[i + depth]) {
					return nullptr;
				}
			}
			return (Leaf *)node;
		}
		if (node->prefix.Size()) {
			for (idx_t pos = 0; pos < node->prefix.Size(); pos++) {
				if (key[depth + pos] != node->prefix[pos]) {
					return nullptr;
				}
			}
			depth += node->prefix.Size();
		}
		idx_t pos = node->GetChildPos(key[depth]);
		if (pos == DConstants::INVALID_INDEX) {
			return nullptr;
		}
		node = node->GetChild(*this, pos);
		D_ASSERT(node);
		depth++;
	}
	return nullptr;
}

//===--------------------------------------------------------------------===//
// Greater Than
// Returns: True (If found leaf >= key)
//          False (Otherwise)
//===--------------------------------------------------------------------===//
bool ART::SearchGreater(ARTIndexScanState *state, Key &key, bool inclusive, idx_t max_count,
                        vector<row_t> &result_ids) {

	Iterator *it = &state->iterator;

	// greater than scan: first set the iterator to the node at which we will start our scan by finding the lowest node
	// that satisfies our requirement
	if (!it->art) {
		it->art = this;
		bool found = it->LowerBound(tree, key, inclusive);
		if (!found) {
			return true;
		}
	}
	// after that we continue the scan; we don't need to check the bounds as any value following this value is
	// automatically bigger and hence satisfies our predicate
	Key empty_key = Key();
	return it->Scan(empty_key, max_count, result_ids, false);
}

//===--------------------------------------------------------------------===//
// Less Than
//===--------------------------------------------------------------------===//
bool ART::SearchLess(ARTIndexScanState *state, Key &upper_bound, bool inclusive, idx_t max_count,
                     vector<row_t> &result_ids) {

	if (!tree) {
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
bool ART::SearchCloseRange(ARTIndexScanState *state, Key &lower_bound, Key &upper_bound, bool left_inclusive,
                           bool right_inclusive, idx_t max_count, vector<row_t> &result_ids) {

	Iterator *it = &state->iterator;

	// first find the first node that satisfies the left predicate
	if (!it->art) {
		it->art = this;
		bool found = it->LowerBound(tree, lower_bound, left_inclusive);
		if (!found) {
			return true;
		}
	}
	// now continue the scan until we reach the upper bound
	return it->Scan(upper_bound, max_count, result_ids, right_inclusive);
}

bool ART::Scan(Transaction &transaction, DataTable &table, IndexScanState &table_state, idx_t max_count,
               vector<row_t> &result_ids) {

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

void ART::VerifyExistence(DataChunk &chunk, VerifyExistenceType verify_type, string *err_msg_ptr) {
	if (verify_type != VerifyExistenceType::DELETE_FK && !IsUnique()) {
		return;
	}

	DataChunk expression_chunk;
	expression_chunk.Initialize(Allocator::DefaultAllocator(), logical_types);

	// unique index, check
	lock_guard<mutex> l(lock);
	// first resolve the expressions for the index
	ExecuteExpressions(chunk, expression_chunk);

	// generate the keys for the given input
	ArenaAllocator arena_allocator(BufferAllocator::Get(db));
	vector<Key> keys(expression_chunk.size());
	GenerateKeys(arena_allocator, expression_chunk, keys);

	for (idx_t i = 0; i < chunk.size(); i++) {
		if (keys[i].Empty()) {
			continue;
		}
		Node *node_ptr = Lookup(tree, keys[i], 0);
		bool throw_exception =
		    verify_type == VerifyExistenceType::APPEND_FK ? node_ptr == nullptr : node_ptr != nullptr;
		if (!throw_exception) {
			continue;
		}
		string key_name;
		for (idx_t k = 0; k < expression_chunk.ColumnCount(); k++) {
			if (k > 0) {
				key_name += ", ";
			}
			key_name += unbound_expressions[k]->GetName() + ": " + expression_chunk.data[k].GetValue(i).ToString();
		}
		string exception_msg;
		switch (verify_type) {
		case VerifyExistenceType::APPEND: {
			// node already exists in tree
			string type = IsPrimary() ? "primary key" : "unique";
			exception_msg = "duplicate key \"" + key_name + "\" violates ";
			exception_msg += type + " constraint";
			break;
		}
		case VerifyExistenceType::APPEND_FK: {
			// found node no exists in tree
			exception_msg =
			    "violates foreign key constraint because key \"" + key_name + "\" does not exist in referenced table";
			break;
		}
		case VerifyExistenceType::DELETE_FK: {
			// found node exists in tree
			exception_msg =
			    "violates foreign key constraint because key \"" + key_name + "\" exist in table has foreign key";
			break;
		}
		}
		if (err_msg_ptr) {
			err_msg_ptr[i] = exception_msg;
		} else {
			throw ConstraintException(exception_msg);
		}
	}
}

//===--------------------------------------------------------------------===//
// Serialization
//===--------------------------------------------------------------------===//
BlockPointer ART::Serialize(duckdb::MetaBlockWriter &writer) {
	lock_guard<mutex> l(lock);
	if (tree) {
		serialized_data_pointer = tree->Serialize(*this, writer);
	} else {
		serialized_data_pointer = {(block_id_t)DConstants::INVALID_INDEX, (uint32_t)DConstants::INVALID_INDEX};
	}
	return serialized_data_pointer;
}

//===--------------------------------------------------------------------===//
// Merge ARTs
//===--------------------------------------------------------------------===//
bool ART::MergeIndexes(IndexLock &state, Index *other_index) {
	auto other_art = (ART *)other_index;
	estimated_art_size += other_art->estimated_art_size;
	other_art->estimated_art_size = 0;
	if (!this->tree) {
		this->tree = other_art->tree;
		other_art->tree = nullptr;
		return true;
	}

	return Node::MergeARTs(this, other_art);
}

string ART::ToString() {
	if (tree) {
		return tree->ToString(*this);
	}
	return "[empty]";
}

} // namespace duckdb




namespace duckdb {

Key::Key() : len(0) {
}

Key::Key(data_ptr_t data, idx_t len) : len(len), data(data) {
}

Key::Key(ArenaAllocator &allocator, idx_t len) : len(len) {
	data = allocator.Allocate(len);
}

template <>
Key Key::CreateKey(ArenaAllocator &allocator, string_t value) {
	idx_t len = value.GetSize() + 1;
	auto data = allocator.Allocate(len);
	memcpy(data, value.GetDataUnsafe(), len - 1);
	data[len - 1] = '\0';
	return Key(data, len);
}

template <>
Key Key::CreateKey(ArenaAllocator &allocator, const char *value) {
	return Key::CreateKey(allocator, string_t(value, strlen(value)));
}

template <>
void Key::CreateKey(ArenaAllocator &allocator, Key &key, string_t value) {
	key.len = value.GetSize() + 1;
	key.data = allocator.Allocate(key.len);
	memcpy(key.data, value.GetDataUnsafe(), key.len - 1);
	key.data[key.len - 1] = '\0';
}

template <>
void Key::CreateKey(ArenaAllocator &allocator, Key &key, const char *value) {
	Key::CreateKey(allocator, key, string_t(value, strlen(value)));
}

bool Key::operator>(const Key &k) const {
	for (idx_t i = 0; i < MinValue<idx_t>(len, k.len); i++) {
		if (data[i] > k.data[i]) {
			return true;
		} else if (data[i] < k.data[i]) {
			return false;
		}
	}
	return len > k.len;
}

bool Key::operator<(const Key &k) const {
	for (idx_t i = 0; i < MinValue<idx_t>(len, k.len); i++) {
		if (data[i] < k.data[i]) {
			return true;
		} else if (data[i] > k.data[i]) {
			return false;
		}
	}
	return len < k.len;
}

bool Key::operator>=(const Key &k) const {
	for (idx_t i = 0; i < MinValue<idx_t>(len, k.len); i++) {
		if (data[i] > k.data[i]) {
			return true;
		} else if (data[i] < k.data[i]) {
			return false;
		}
	}
	return len >= k.len;
}

bool Key::operator==(const Key &k) const {
	if (len != k.len) {
		return false;
	}
	for (idx_t i = 0; i < len; i++) {
		if (data[i] != k.data[i]) {
			return false;
		}
	}
	return true;
}

bool Key::ByteMatches(Key &other, idx_t &depth) {
	return data[depth] == other[depth];
}

bool Key::Empty() {
	return len == 0;
}

void Key::ConcatenateKey(ArenaAllocator &allocator, Key &other_key) {

	auto compound_data = allocator.Allocate(len + other_key.len);
	memcpy(compound_data, data, len);
	memcpy(compound_data + len, other_key.data, other_key.len);
	len += other_key.len;
	data = compound_data;
}
} // namespace duckdb




namespace duckdb {
uint8_t &IteratorCurrentKey::operator[](idx_t idx) {
	if (idx >= key.size()) {
		key.push_back(0);
	}
	D_ASSERT(idx < key.size());
	return key[idx];
}

//! Push Byte
void IteratorCurrentKey::Push(uint8_t byte) {
	if (cur_key_pos == key.size()) {
		key.push_back(byte);
	}
	D_ASSERT(cur_key_pos < key.size());
	key[cur_key_pos++] = byte;
}
//! Pops n elements
void IteratorCurrentKey::Pop(idx_t n) {
	cur_key_pos -= n;
	D_ASSERT(cur_key_pos <= key.size());
}

bool IteratorCurrentKey::operator>(const Key &k) const {
	for (idx_t i = 0; i < MinValue<idx_t>(cur_key_pos, k.len); i++) {
		if (key[i] > k.data[i]) {
			return true;
		} else if (key[i] < k.data[i]) {
			return false;
		}
	}
	return cur_key_pos > k.len;
}

bool IteratorCurrentKey::operator>=(const Key &k) const {
	for (idx_t i = 0; i < MinValue<idx_t>(cur_key_pos, k.len); i++) {
		if (key[i] > k.data[i]) {
			return true;
		} else if (key[i] < k.data[i]) {
			return false;
		}
	}
	return cur_key_pos >= k.len;
}

bool IteratorCurrentKey::operator==(const Key &k) const {
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
	Node *next = nullptr;
	idx_t pos = 0;
	// reconstruct the prefix
	for (idx_t i = 0; i < node.prefix.Size(); i++) {
		cur_key.Push(node.prefix[i]);
	}
	switch (node.type) {
	case NodeType::NLeaf:
		last_leaf = (Leaf *)&node;
		return;
	case NodeType::N4: {
		next = ((Node4 &)node).children[0].Unswizzle(*art);
		cur_key.Push(((Node4 &)node).key[0]);
		break;
	}
	case NodeType::N16: {
		next = ((Node16 &)node).children[0].Unswizzle(*art);
		cur_key.Push(((Node16 &)node).key[0]);
		break;
	}
	case NodeType::N48: {
		auto &n48 = (Node48 &)node;
		while (n48.child_index[pos] == Node::EMPTY_MARKER) {
			pos++;
		}
		cur_key.Push(pos);
		next = n48.children[n48.child_index[pos]].Unswizzle(*art);
		break;
	}
	case NodeType::N256: {
		auto &n256 = (Node256 &)node;
		while (!n256.children[pos]) {
			pos++;
		}
		cur_key.Push(pos);
		next = (Node *)n256.children[pos].Unswizzle(*art);
		break;
	}
	}
	nodes.push(IteratorEntry(&node, pos));
	FindMinimum(*next);
}

void Iterator::PushKey(Node *cur_node, uint16_t pos) {
	switch (cur_node->type) {
	case NodeType::N4:
		cur_key.Push(((Node4 *)cur_node)->key[pos]);
		break;
	case NodeType::N16:
		cur_key.Push(((Node16 *)cur_node)->key[pos]);
		break;
	case NodeType::N48:
	case NodeType::N256:
		cur_key.Push(pos);
		break;
	case NodeType::NLeaf:
		break;
	}
}

bool Iterator::Scan(Key &bound, idx_t max_count, vector<row_t> &result_ids, bool is_inclusive) {
	bool has_next;
	do {
		if (!bound.Empty()) {
			if (is_inclusive) {
				if (cur_key > bound) {
					break;
				}
			} else {
				if (cur_key >= bound) {
					break;
				}
			}
		}
		if (result_ids.size() + last_leaf->count > max_count) {
			// adding these elements would exceed the max count
			return false;
		}
		for (idx_t i = 0; i < last_leaf->count; i++) {
			row_t row_id = last_leaf->GetRowId(i);
			result_ids.push_back(row_id);
		}
		has_next = Next();
	} while (has_next);
	return true;
}

void Iterator::PopNode() {
	auto cur_node = nodes.top();
	idx_t elements_to_pop = cur_node.node->prefix.Size() + (nodes.size() != 1);
	cur_key.Pop(elements_to_pop);
	nodes.pop();
}

bool Iterator::Next() {
	if (!nodes.empty()) {
		auto cur_node = nodes.top().node;
		if (cur_node->type == NodeType::NLeaf) {
			// Pop Leaf (We must pop the prefix size + the key to the node (unless we are popping the root)
			PopNode();
		}
	}

	// Look for the next leaf
	while (!nodes.empty()) {
		// cur_node
		auto &top = nodes.top();
		Node *node = top.node;
		if (node->type == NodeType::NLeaf) {
			// found a leaf: move to next node
			last_leaf = (Leaf *)node;
			return true;
		}
		// Find next node
		top.pos = node->GetNextPos(top.pos);
		if (top.pos != DConstants::INVALID_INDEX) {
			// add key-byte of the new node
			PushKey(node, top.pos);
			auto next_node = node->GetChild(*art, top.pos);
			// add prefix of new node
			for (idx_t i = 0; i < next_node->prefix.Size(); i++) {
				cur_key.Push(next_node->prefix[i]);
			}
			// next node found: push it
			nodes.push(IteratorEntry(next_node, DConstants::INVALID_INDEX));
		} else {
			// no node found: move up the tree and Pop prefix and key of current node
			PopNode();
		}
	}
	return false;
}

bool Iterator::LowerBound(Node *node, Key &key, bool inclusive) {
	bool equal = true;
	if (!node) {
		return false;
	}
	idx_t depth = 0;
	while (true) {
		nodes.push(IteratorEntry(node, 0));
		auto &top = nodes.top();
		// reconstruct the prefix
		for (idx_t i = 0; i < top.node->prefix.Size(); i++) {
			cur_key.Push(top.node->prefix[i]);
		}
		// greater case: find leftmost leaf node directly
		if (!equal) {
			while (node->type != NodeType::NLeaf) {
				auto min_pos = node->GetMin();
				PushKey(node, min_pos);
				nodes.push(IteratorEntry(node, min_pos));
				node = node->GetChild(*art, min_pos);
				// reconstruct the prefix
				for (idx_t i = 0; i < node->prefix.Size(); i++) {
					cur_key.Push(node->prefix[i]);
				}
				auto &c_top = nodes.top();
				c_top.node = node;
			}
		}
		if (node->type == NodeType::NLeaf) {
			// found a leaf node: check if it is bigger or equal than the current key
			auto leaf = static_cast<Leaf *>(node);
			last_leaf = leaf;
			// if the search is not inclusive the leaf node could still be equal to the current value
			// check if leaf is equal to the current key
			if (cur_key == key) {
				// if it's not inclusive check if there is a next leaf
				if (!inclusive && !Next()) {
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
		uint32_t mismatch_pos = node->prefix.KeyMismatchPosition(key, depth);
		if (mismatch_pos != node->prefix.Size()) {
			if (node->prefix[mismatch_pos] < key[depth + mismatch_pos]) {
				// Less
				PopNode();
				return Next();
			} else {
				// Greater
				top.pos = DConstants::INVALID_INDEX;
				return Next();
			}
		}

		// prefix matches, search inside the child for the key
		depth += node->prefix.Size();

		top.pos = node->GetChildGreaterEqual(key[depth], equal);
		// The maximum key byte of the current node is less than the key
		// So fall back to the previous node
		if (top.pos == DConstants::INVALID_INDEX) {
			PopNode();
			return Next();
		}
		PushKey(node, top.pos);
		node = node->GetChild(*art, top.pos);
		// This means all children of this node qualify as geq
		depth++;
	}
}

} // namespace duckdb





#include <cstring>

namespace duckdb {
idx_t Leaf::GetCapacity() const {
	return IsInlined() ? 1 : rowids.ptr[0];
}

bool Leaf::IsInlined() const {
	return count <= 1;
}

row_t Leaf::GetRowId(idx_t index) {
	D_ASSERT(index < count);
	if (IsInlined()) {
		return rowids.inlined;
	} else {
		D_ASSERT(rowids.ptr[0] >= count);
		return rowids.ptr[index + 1];
	}
}

row_t *Leaf::GetRowIds() {
	if (IsInlined()) {
		return &rowids.inlined;
	} else {
		return rowids.ptr + 1;
	}
}

Leaf::Leaf(Key &value, uint32_t depth, row_t row_id) : Node(NodeType::NLeaf) {
	count = 1;
	rowids.inlined = row_id;
	D_ASSERT(value.len >= depth);
	prefix = Prefix(value, depth, value.len - depth);
}

Leaf::Leaf(Key &value, uint32_t depth, row_t *row_ids_p, idx_t num_elements_p) : Node(NodeType::NLeaf) {
	D_ASSERT(num_elements_p >= 1);
	if (num_elements_p == 1) {
		// we can inline the row ids
		rowids.inlined = row_ids_p[0];
	} else {
		// new row ids of this leaf
		count = 0;
		Resize(row_ids_p, num_elements_p, num_elements_p);
	}
	count = num_elements_p;
	D_ASSERT(value.len >= depth);
	prefix = Prefix(value, depth, value.len - depth);
}

Leaf::Leaf(row_t *row_ids_p, idx_t num_elements_p, Prefix &prefix_p) : Node(NodeType::NLeaf) {
	D_ASSERT(num_elements_p > 1);
	D_ASSERT(row_ids_p[0] == row_t(num_elements_p)); // first element should contain capacity
	rowids.ptr = row_ids_p;
	count = num_elements_p;
	prefix = prefix_p;
}

Leaf::Leaf(row_t row_id, Prefix &prefix_p) : Node(NodeType::NLeaf) {
	rowids.inlined = row_id;
	count = 1;
	prefix = prefix_p;
}

Leaf::~Leaf() {
	if (!IsInlined()) {
		DeleteArray<row_t>(rowids.ptr, rowids.ptr[0] + 1);
		count = 0;
	}
}

row_t *Leaf::Resize(row_t *current_row_ids, uint32_t current_count, idx_t new_capacity) {
	D_ASSERT(new_capacity >= current_count);
	auto new_allocation = AllocateArray<row_t>(new_capacity + 1);
	new_allocation[0] = new_capacity;
	auto new_row_ids = new_allocation + 1;
	memcpy(new_row_ids, current_row_ids, current_count * sizeof(row_t));
	if (!IsInlined()) {
		// delete the old data
		DeleteArray<row_t>(rowids.ptr, rowids.ptr[0] + 1);
	}
	// set up the new pointers
	rowids.ptr = new_allocation;
	return new_row_ids;
}

void Leaf::Insert(row_t row_id) {
	auto capacity = GetCapacity();
	row_t *row_ids = GetRowIds();
	D_ASSERT(count <= capacity);
	if (count == capacity) {
		// Grow array
		row_ids = Resize(row_ids, count, capacity * 2);
	}
	row_ids[count++] = row_id;
}

void Leaf::Remove(row_t row_id) {
	idx_t entry_offset = DConstants::INVALID_INDEX;
	row_t *row_ids = GetRowIds();
	for (idx_t i = 0; i < count; i++) {
		if (row_ids[i] == row_id) {
			entry_offset = i;
			break;
		}
	}
	if (entry_offset == DConstants::INVALID_INDEX) {
		return;
	}
	if (IsInlined()) {
		D_ASSERT(count == 1);
		count--;
		return;
	}
	count--;
	if (count == 1) {
		// after erasing we can now inline the leaf
		// delete the pointer and inline the remaining rowid
		auto remaining_row_id = row_ids[0] == row_id ? row_ids[1] : row_ids[0];
		DeleteArray<row_t>(rowids.ptr, rowids.ptr[0] + 1);
		rowids.inlined = remaining_row_id;
		return;
	}
	auto capacity = GetCapacity();
	if (capacity > 2 && count < capacity / 2) {
		// Shrink array, if less than half full
		auto new_capacity = capacity / 2;
		auto new_allocation = AllocateArray<row_t>(new_capacity + 1);
		new_allocation[0] = new_capacity;
		auto new_row_ids = new_allocation + 1;
		memcpy(new_row_ids, row_ids, entry_offset * sizeof(row_t));
		memcpy(new_row_ids + entry_offset, row_ids + entry_offset + 1, (count - entry_offset) * sizeof(row_t));
		DeleteArray<row_t>(rowids.ptr, rowids.ptr[0] + 1);
		rowids.ptr = new_allocation;
	} else {
		// Copy the rest
		memmove(row_ids + entry_offset, row_ids + entry_offset + 1, (count - entry_offset) * sizeof(row_t));
	}
}

string Leaf::ToString(Node *node) {
	Leaf *leaf = (Leaf *)node;
	string str = "Leaf: [";
	auto row_ids = leaf->GetRowIds();
	for (idx_t i = 0; i < leaf->count; i++) {
		str += i == 0 ? to_string(row_ids[i]) : ", " + to_string(row_ids[i]);
	}
	return str + "]";
}

void Leaf::Merge(Node *&l_node, Node *&r_node) {
	Leaf *l_n = (Leaf *)l_node;
	Leaf *r_n = (Leaf *)r_node;

	// append non-duplicate row_ids to l_n
	for (idx_t i = 0; i < r_n->count; i++) {
		l_n->Insert(r_n->GetRowId(i));
	}
}

BlockPointer Leaf::Serialize(duckdb::MetaBlockWriter &writer) {
	auto ptr = writer.GetBlockPointer();
	// Write Node Type
	writer.Write(type);
	// Write compression Info
	prefix.Serialize(writer);
	// Write Row Ids
	// Length
	writer.Write<uint16_t>(count);
	// Actual Row Ids
	auto row_ids = GetRowIds();
	for (idx_t i = 0; i < count; i++) {
		writer.Write(row_ids[i]);
	}
	return ptr;
}

Leaf *Leaf::Deserialize(MetaBlockReader &reader) {
	Prefix prefix;
	prefix.Deserialize(reader);
	auto num_elements = reader.Read<uint16_t>();
	if (num_elements == 1) {
		// inlined
		auto element = reader.Read<row_t>();
		return Leaf::New(element, prefix);
	} else {
		// non-inlined
		auto elements = AllocateArray<row_t>(num_elements + 1);
		elements[0] = num_elements;
		for (idx_t i = 0; i < num_elements; i++) {
			elements[i + 1] = reader.Read<row_t>();
		}
		return Leaf::New(elements, num_elements, prefix);
	}
}

} // namespace duckdb








namespace duckdb {

InternalType::InternalType(Node *n) {
	switch (n->type) {
	case NodeType::N4: {
		auto n4 = (Node4 *)n;
		Set(n4->key, 4, n4->children, 4);
		break;
	}
	case NodeType::N16: {
		auto n16 = (Node16 *)n;
		Set(n16->key, 16, n16->children, 16);
		break;
	}
	case NodeType::N48: {
		auto n48 = (Node48 *)n;
		Set(n48->child_index, 256, n48->children, 48);
		break;
	}
	case NodeType::N256: {
		auto n256 = (Node256 *)n;
		Set(nullptr, 0, n256->children, 256);
		break;
	}
	default:
		throw InternalException("This is not an Internal ART Node Type");
	}
}

void InternalType::Set(uint8_t *key_p, uint16_t key_size_p, ARTPointer *children_p, uint16_t children_size_p) {
	key = key_p;
	key_size = key_size_p;
	children = children_p;
	children_size = children_size_p;
}

Node::Node(NodeType type) : count(0), type(type) {
}

// LCOV_EXCL_START
idx_t Node::GetMin() {
	throw InternalException("GetMin not implemented for the specific node type.");
}

Node *Node::GetChild(ART &art, idx_t pos) {
	throw InternalException("GetChild not implemented for the specific node type.");
}

void Node::ReplaceChildPointer(idx_t pos, Node *node) {
	throw InternalException("ReplaceChildPointer not implemented for the specific node type.");
}
// LCOV_EXCL_STOP

void Node::InsertChild(Node *&node, uint8_t key_byte, Node *new_child) {
	switch (node->type) {
	case NodeType::N4:
		Node4::InsertChild(node, key_byte, new_child);
		break;
	case NodeType::N16:
		Node16::InsertChild(node, key_byte, new_child);
		break;
	case NodeType::N48:
		Node48::InsertChild(node, key_byte, new_child);
		break;
	case NodeType::N256:
		Node256::InsertChild(node, key_byte, new_child);
		break;
	default:
		throw InternalException("Unrecognized leaf type for insert");
	}
}

void Node::EraseChild(Node *&node, idx_t pos, ART &art) {
	switch (node->type) {
	case NodeType::N4: {
		Node4::EraseChild(node, pos, art);
		break;
	}
	case NodeType::N16: {
		Node16::EraseChild(node, pos, art);
		break;
	}
	case NodeType::N48: {
		Node48::EraseChild(node, pos, art);
		break;
	}
	case NodeType::N256:
		Node256::EraseChild(node, pos, art);
		break;
	default:
		throw InternalException("Unrecognized leaf type for erase");
	}
}

NodeType Node::GetTypeBySize(idx_t size) {

	if (size <= Node4::GetSize()) {
		return NodeType::N4;
	} else if (size <= Node16::GetSize()) {
		return NodeType::N16;
	} else if (size <= Node48::GetSize()) {
		return NodeType::N48;
	}
	D_ASSERT(size <= Node256::GetSize());
	return NodeType::N256;
}

void Node::New(NodeType &type, Node *&node) {
	switch (type) {
	case NodeType::N4:
		node = (Node *)Node4::New();
		return;
	case NodeType::N16:
		node = (Node *)Node16::New();
		return;
	case NodeType::N48:
		node = (Node *)Node48::New();
		return;
	case NodeType::N256:
		node = (Node *)Node256::New();
		return;
	default:
		throw InternalException("Unrecognized type for new node creation!");
	}
}

Node4 *Node4::New() {
	return AllocateObject<Node4>();
}

Node16 *Node16::New() {
	return AllocateObject<Node16>();
}

Node48 *Node48::New() {
	return AllocateObject<Node48>();
}

Node256 *Node256::New() {
	return AllocateObject<Node256>();
}

Leaf *Leaf::New(Key &value, uint32_t depth, row_t row_id) {
	return AllocateObject<Leaf>(value, depth, row_id);
}

Leaf *Leaf::New(Key &value, uint32_t depth, row_t *row_ids, idx_t num_elements) {
	return AllocateObject<Leaf>(value, depth, row_ids, num_elements);
}

Leaf *Leaf::New(row_t *row_ids, idx_t num_elements, Prefix &prefix) {
	return AllocateObject<Leaf>(row_ids, num_elements, prefix);
}

Leaf *Leaf::New(row_t row_id, Prefix &prefix) {
	return AllocateObject<Leaf>(row_id, prefix);
}

void Node::Delete(Node *ptr) {
	switch (ptr->type) {
	case NodeType::NLeaf:
		DestroyObject((Leaf *)ptr);
		break;
	case NodeType::N4:
		DestroyObject((Node4 *)ptr);
		break;
	case NodeType::N16:
		DestroyObject((Node16 *)ptr);
		break;
	case NodeType::N48:
		DestroyObject((Node48 *)ptr);
		break;
	case NodeType::N256:
		DestroyObject((Node256 *)ptr);
		break;
	default:
		throw InternalException("eek");
	}
}

string Node::ToString(ART &art) {

	string str = "Node";
	switch (this->type) {
	case NodeType::NLeaf:
		return Leaf::ToString(this);
	case NodeType::N4:
		str += to_string(Node4::GetSize());
		break;
	case NodeType::N16:
		str += to_string(Node16::GetSize());
		break;
	case NodeType::N48:
		str += to_string(Node48::GetSize());
		break;
	case NodeType::N256:
		str += to_string(Node256::GetSize());
		break;
	}

	str += ": [";
	auto next_pos = GetNextPos(DConstants::INVALID_INDEX);
	while (next_pos != DConstants::INVALID_INDEX) {
		auto child = GetChild(art, next_pos);
		str += "(" + to_string(next_pos) + ", " + child->ToString(art) + ")";
		next_pos = GetNextPos(next_pos);
	}
	return str + "]";
}

BlockPointer Node::SerializeInternal(ART &art, duckdb::MetaBlockWriter &writer, InternalType &internal_type) {
	// Iterate through children and annotate their offsets
	vector<BlockPointer> child_offsets;
	for (idx_t i = 0; i < internal_type.children_size; i++) {
		child_offsets.emplace_back(internal_type.children[i].Serialize(art, writer));
	}
	auto ptr = writer.GetBlockPointer();
	// Write Node Type
	writer.Write(type);
	// Write count
	writer.Write<uint16_t>(count);
	// Write Prefix
	prefix.Serialize(writer);
	// Write Key values
	for (idx_t i = 0; i < internal_type.key_size; i++) {
		writer.Write(internal_type.key[i]);
	}
	// Write child offsets
	for (auto &offsets : child_offsets) {
		writer.Write(offsets.block_id);
		writer.Write(offsets.offset);
	}
	return ptr;
}

BlockPointer Node::Serialize(ART &art, duckdb::MetaBlockWriter &writer) {
	switch (type) {
	case NodeType::N4:
	case NodeType::N16:
	case NodeType::N48:
	case NodeType::N256: {
		InternalType internal_type(this);
		return SerializeInternal(art, writer, internal_type);
	}
	case NodeType::NLeaf: {
		auto leaf = (Leaf *)this;
		return leaf->Serialize(writer);
	}
	default:
		throw InternalException("Invalid ART Node");
	}
}

void Node::DeserializeInternal(duckdb::MetaBlockReader &reader) {
	InternalType internal_type(this);
	count = reader.Read<uint16_t>();
	prefix.Deserialize(reader);
	// Get Key values
	for (idx_t i = 0; i < internal_type.key_size; i++) {
		internal_type.key[i] = reader.Read<uint8_t>();
	}
	// Get Child offsets
	for (idx_t i = 0; i < internal_type.children_size; i++) {
		internal_type.children[i] = ARTPointer(reader);
	}
}

Node *Node::Deserialize(ART &art, idx_t block_id, idx_t offset) {
	MetaBlockReader reader(art.table_io_manager.GetIndexBlockManager(), block_id);
	reader.offset = offset;
	auto n = reader.Read<uint8_t>();
	NodeType node_type(static_cast<NodeType>(n));
	Node *deserialized_node;
	switch (node_type) {
	case NodeType::NLeaf:
		return Leaf::Deserialize(reader);
	case NodeType::N4: {
		deserialized_node = (Node *)Node4::New();
		break;
	}
	case NodeType::N16: {
		deserialized_node = (Node *)Node16::New();
		break;
	}
	case NodeType::N48: {
		deserialized_node = (Node *)Node48::New();
		break;
	}
	case NodeType::N256: {
		deserialized_node = (Node *)Node256::New();
		break;
	}
	}
	deserialized_node->DeserializeInternal(reader);
	return deserialized_node;
}

void UpdateParentsOfNodes(Node *&l_node, Node *&r_node, ParentsOfNodes &parents) {
	if (parents.l_parent) {
		parents.l_parent->ReplaceChildPointer(parents.l_pos, l_node);
	}
	if (parents.r_parent) {
		parents.r_parent->ReplaceChildPointer(parents.r_pos, r_node);
	}
}

bool Merge(MergeInfo &info, idx_t depth, ParentsOfNodes &parents) {

	// always try to merge the smaller node into the bigger node
	// because maybe there is enough free space in the bigger node to fit the smaller one
	// without too much recursion

	if (info.l_node->type < info.r_node->type) {
		// swap subtrees to ensure that l_node has the bigger node type
		swap(info.l_art, info.r_art);
		swap(info.l_node, info.r_node);
		UpdateParentsOfNodes(info.l_node, info.r_node, parents);
	}

	switch (info.r_node->type) {
	case NodeType::N256:
		return Node256::Merge(info, depth, parents.l_parent, parents.l_pos);
	case NodeType::N48:
		return Node48::Merge(info, depth, parents.l_parent, parents.l_pos);
	case NodeType::N16:
		return Node16::Merge(info, depth, parents.l_parent, parents.l_pos);
	case NodeType::N4:
		return Node4::Merge(info, depth, parents.l_parent, parents.l_pos);
	case NodeType::NLeaf:
		D_ASSERT(info.l_node->type == NodeType::NLeaf);
		D_ASSERT(info.r_node->type == NodeType::NLeaf);
		if (info.l_art->IsUnique()) {
			return false;
		}
		Leaf::Merge(info.l_node, info.r_node);
		return true;
	}
	throw InternalException("Invalid node type for right node in merge.");
}

bool ResolvePrefixesAndMerge(MergeInfo &info, idx_t depth, ParentsOfNodes &parents) {
	auto &l_node = info.l_node;
	auto &r_node = info.r_node;
	Node *null_parent = nullptr;

	// NOTE: we always merge into the left ART
	D_ASSERT(l_node);

	// make sure that r_node has the longer (or equally long) prefix
	if (l_node->prefix.Size() > r_node->prefix.Size()) {
		swap(info.l_art, info.r_art);
		swap(l_node, r_node);
		UpdateParentsOfNodes(l_node, r_node, parents);
	}

	auto mismatch_pos = l_node->prefix.MismatchPosition(r_node->prefix);

	// both nodes have no prefix or the same prefix
	if (mismatch_pos == l_node->prefix.Size() && l_node->prefix.Size() == r_node->prefix.Size()) {
		return Merge(info, depth + mismatch_pos, parents);
	}

	if (mismatch_pos == l_node->prefix.Size()) {
		// r_node's prefix contains l_node's prefix
		// l_node cannot be a leaf, otherwise the key represented by l_node would be a subset of another key
		// which is not possible by our construction
		D_ASSERT(l_node->type != NodeType::NLeaf);

		// test if the next byte (mismatch_pos) in r_node (longer prefix) exists in l_node
		auto mismatch_byte = r_node->prefix[mismatch_pos];
		auto child_pos = l_node->GetChildPos(mismatch_byte);

		// update the prefix of r_node to only consist of the bytes after mismatch_pos
		r_node->prefix.Reduce(mismatch_pos);

		// insert r_node as a child of l_node at empty position
		if (child_pos == DConstants::INVALID_INDEX) {
			Node::InsertChild(l_node, mismatch_byte, r_node);
			UpdateParentsOfNodes(l_node, null_parent, parents);
			r_node = nullptr;
			return true;
		}

		// recurse
		auto child_node = l_node->GetChild(*info.l_art, child_pos);
		MergeInfo child_info(info.l_art, info.r_art, child_node, r_node);
		ParentsOfNodes child_parents(l_node, child_pos, parents.r_parent, parents.r_pos);
		return ResolvePrefixesAndMerge(child_info, depth + mismatch_pos, child_parents);
	}

	// prefixes differ, create new node and insert both nodes as children

	// create new node
	Node *new_node = Node4::New();
	new_node->prefix = Prefix(l_node->prefix, mismatch_pos);

	// insert l_node, break up prefix of l_node
	auto key_byte = l_node->prefix.Reduce(mismatch_pos);
	Node4::InsertChild(new_node, key_byte, l_node);

	// insert r_node, break up prefix of r_node
	key_byte = r_node->prefix.Reduce(mismatch_pos);
	Node4::InsertChild(new_node, key_byte, r_node);

	l_node = new_node;
	UpdateParentsOfNodes(l_node, null_parent, parents);
	r_node = nullptr;
	return true;
}

bool Node::MergeAtByte(MergeInfo &info, idx_t depth, idx_t &l_child_pos, idx_t &r_pos, uint8_t &key_byte,
                       Node *&l_parent, idx_t l_pos) {

	auto r_child = info.r_node->GetChild(*info.r_art, r_pos);

	// insert child at empty position
	if (l_child_pos == DConstants::INVALID_INDEX) {
		Node::InsertChild(info.l_node, key_byte, r_child);
		if (l_parent) {
			l_parent->ReplaceChildPointer(l_pos, info.l_node);
		}
		info.r_node->ReplaceChildPointer(r_pos, nullptr);
		return true;
	}

	// recurse
	auto l_child = info.l_node->GetChild(*info.l_art, l_child_pos);
	MergeInfo child_info(info.l_art, info.r_art, l_child, r_child);
	ParentsOfNodes child_parents(info.l_node, l_child_pos, info.r_node, r_pos);
	return ResolvePrefixesAndMerge(child_info, depth + 1, child_parents);
}

bool Node::MergeARTs(ART *l_art, ART *r_art) {

	Node *null_parent = nullptr;
	MergeInfo info(l_art, r_art, l_art->tree, r_art->tree);
	ParentsOfNodes parents(null_parent, 0, null_parent, 0);
	return ResolvePrefixesAndMerge(info, 0, parents);
}

} // namespace duckdb





#include <cstring>

namespace duckdb {

Node16::Node16() : Node(NodeType::N16) {
	memset(key, 16, sizeof(key));
}

idx_t Node16::GetChildPos(uint8_t k) {
	for (idx_t pos = 0; pos < count; pos++) {
		if (key[pos] == k) {
			return pos;
		}
	}
	return Node::GetChildPos(k);
}

idx_t Node16::GetChildGreaterEqual(uint8_t k, bool &equal) {
	for (idx_t pos = 0; pos < count; pos++) {
		if (key[pos] >= k) {
			if (key[pos] == k) {
				equal = true;
			} else {
				equal = false;
			}

			return pos;
		}
	}
	return DConstants::INVALID_INDEX;
}

idx_t Node16::GetMin() {
	return 0;
}

idx_t Node16::GetNextPos(idx_t pos) {
	if (pos == DConstants::INVALID_INDEX) {
		return 0;
	}
	pos++;
	return pos < count ? pos : DConstants::INVALID_INDEX;
}

Node *Node16::GetChild(ART &art, idx_t pos) {
	D_ASSERT(pos < count);
	return children[pos].Unswizzle(art);
}

void Node16::ReplaceChildPointer(idx_t pos, Node *node) {
	children[pos] = node;
}

void Node16::InsertChild(Node *&node, uint8_t key_byte, Node *new_child) {
	Node16 *n = (Node16 *)node;

	// Insert new child node into node
	if (n->count < 16) {
		// Insert element
		idx_t pos = 0;
		while (pos < node->count && n->key[pos] < key_byte) {
			pos++;
		}
		if (n->children[pos]) {
			for (idx_t i = n->count; i > pos; i--) {
				n->key[i] = n->key[i - 1];
				n->children[i] = n->children[i - 1];
			}
		}
		n->key[pos] = key_byte;
		n->children[pos] = new_child;
		n->count++;
	} else {
		// Grow to Node48
		auto new_node = Node48::New();
		for (idx_t i = 0; i < node->count; i++) {
			new_node->child_index[n->key[i]] = i;
			new_node->children[i] = n->children[i];
			n->children[i] = nullptr;
		}
		new_node->prefix = move(n->prefix);
		new_node->count = node->count;
		Node::Delete(node);
		node = new_node;

		Node48::InsertChild(node, key_byte, new_child);
	}
}

void Node16::EraseChild(Node *&node, int pos, ART &art) {
	auto n = (Node16 *)node;
	// erase the child and decrease the count
	n->children[pos].Reset();
	n->count--;
	// potentially move any children backwards
	for (; pos < n->count; pos++) {
		n->key[pos] = n->key[pos + 1];
		n->children[pos] = n->children[pos + 1];
	}
	// set any remaining nodes as nullptr
	for (; pos < 16; pos++) {
		if (!n->children[pos]) {
			break;
		}
		n->children[pos] = nullptr;
	}

	if (node->count <= 3) {
		// Shrink node
		auto new_node = Node4::New();
		for (unsigned i = 0; i < n->count; i++) {
			new_node->key[new_node->count] = n->key[i];
			new_node->children[new_node->count++] = n->children[i];
			n->children[i] = nullptr;
		}
		new_node->prefix = move(n->prefix);
		Node::Delete(node);
		node = new_node;
	}
}

bool Node16::Merge(MergeInfo &info, idx_t depth, Node *&l_parent, idx_t l_pos) {

	Node16 *r_n = (Node16 *)info.r_node;

	for (idx_t i = 0; i < info.r_node->count; i++) {

		auto l_child_pos = info.l_node->GetChildPos(r_n->key[i]);
		if (!Node::MergeAtByte(info, depth, l_child_pos, i, r_n->key[i], l_parent, l_pos)) {
			return false;
		}
	}
	return true;
}

idx_t Node16::GetSize() {
	return 16;
}

} // namespace duckdb




namespace duckdb {

Node256::Node256() : Node(NodeType::N256) {
}

idx_t Node256::GetChildPos(uint8_t k) {
	if (children[k]) {
		return k;
	} else {
		return DConstants::INVALID_INDEX;
	}
}

idx_t Node256::GetChildGreaterEqual(uint8_t k, bool &equal) {
	for (idx_t pos = k; pos < 256; pos++) {
		if (children[pos]) {
			if (pos == k) {
				equal = true;
			} else {
				equal = false;
			}
			return pos;
		}
	}
	return DConstants::INVALID_INDEX;
}

idx_t Node256::GetMin() {
	for (idx_t i = 0; i < 256; i++) {
		if (children[i]) {
			return i;
		}
	}
	return DConstants::INVALID_INDEX;
}

idx_t Node256::GetNextPos(idx_t pos) {
	for (pos == DConstants::INVALID_INDEX ? pos = 0 : pos++; pos < 256; pos++) {
		if (children[pos]) {
			return pos;
		}
	}
	return Node::GetNextPos(pos);
}

Node *Node256::GetChild(ART &art, idx_t pos) {
	return children[pos].Unswizzle(art);
}

void Node256::ReplaceChildPointer(idx_t pos, Node *node) {
	children[pos] = node;
}

void Node256::InsertChild(Node *&node, uint8_t key_byte, Node *new_child) {
	auto n = (Node256 *)(node);

	n->count++;
	n->children[key_byte] = new_child;
}

void Node256::EraseChild(Node *&node, int pos, ART &art) {
	auto n = (Node256 *)(node);
	n->children[pos].Reset();
	n->count--;
	if (node->count <= 36) {
		auto new_node = Node48::New();
		new_node->prefix = move(n->prefix);
		for (idx_t i = 0; i < 256; i++) {
			if (n->children[i]) {
				new_node->child_index[i] = new_node->count;
				new_node->children[new_node->count] = n->children[i];
				n->children[i] = nullptr;
				new_node->count++;
			}
		}
		Node::Delete(node);
		node = new_node;
	}
}

bool Node256::Merge(MergeInfo &info, idx_t depth, Node *&l_parent, idx_t l_pos) {

	for (idx_t i = 0; i < 256; i++) {
		if (info.r_node->GetChildPos(i) != DConstants::INVALID_INDEX) {

			auto l_child_pos = info.l_node->GetChildPos(i);
			auto key_byte = (uint8_t)i;
			if (!Node::MergeAtByte(info, depth, l_child_pos, i, key_byte, l_parent, l_pos)) {
				return false;
			}
		}
	}
	return true;
}

idx_t Node256::GetSize() {
	return 256;
}

} // namespace duckdb






namespace duckdb {

Node4::Node4() : Node(NodeType::N4) {
	memset(key, 0, sizeof(key));
}

idx_t Node4::GetChildPos(uint8_t k) {
	for (idx_t pos = 0; pos < count; pos++) {
		if (key[pos] == k) {
			return pos;
		}
	}
	return Node::GetChildPos(k);
}

idx_t Node4::GetChildGreaterEqual(uint8_t k, bool &equal) {
	for (idx_t pos = 0; pos < count; pos++) {
		if (key[pos] >= k) {
			if (key[pos] == k) {
				equal = true;
			} else {
				equal = false;
			}
			return pos;
		}
	}
	return DConstants::INVALID_INDEX;
}

idx_t Node4::GetMin() {
	return 0;
}

idx_t Node4::GetNextPos(idx_t pos) {
	if (pos == DConstants::INVALID_INDEX) {
		return 0;
	}
	pos++;
	return pos < count ? pos : DConstants::INVALID_INDEX;
}

Node *Node4::GetChild(ART &art, idx_t pos) {
	D_ASSERT(pos < count);
	return children[pos].Unswizzle(art);
}

void Node4::ReplaceChildPointer(idx_t pos, Node *node) {
	children[pos] = node;
}

void Node4::InsertChild(Node *&node, uint8_t key_byte, Node *new_child) {
	Node4 *n = (Node4 *)node;

	// Insert new child node into node
	if (node->count < 4) {
		// Insert element
		idx_t pos = 0;
		while ((pos < node->count) && (n->key[pos] < key_byte)) {
			pos++;
		}
		if (n->children[pos]) {
			for (idx_t i = n->count; i > pos; i--) {
				n->key[i] = n->key[i - 1];
				n->children[i] = n->children[i - 1];
			}
		}
		n->key[pos] = key_byte;
		n->children[pos] = new_child;
		n->count++;
	} else {
		// Grow to Node16
		auto new_node = Node16::New();
		new_node->count = 4;
		new_node->prefix = move(node->prefix);
		for (idx_t i = 0; i < 4; i++) {
			new_node->key[i] = n->key[i];
			new_node->children[i] = n->children[i];
			n->children[i] = nullptr;
		}
		// Delete old node and replace it with new Node16
		Node::Delete(node);
		node = new_node;
		Node16::InsertChild(node, key_byte, new_child);
	}
}

void Node4::EraseChild(Node *&node, int pos, ART &art) {
	Node4 *n = (Node4 *)node;
	D_ASSERT(pos < n->count);
	// erase the child and decrease the count
	n->children[pos].Reset();
	n->count--;
	// potentially move any children backwards
	for (; pos < n->count; pos++) {
		n->key[pos] = n->key[pos + 1];
		n->children[pos] = n->children[pos + 1];
	}
	// set any remaining nodes as nullptr
	for (; pos < 4; pos++) {
		n->children[pos] = nullptr;
	}

	// This is a one way node
	if (n->count == 1) {
		auto child_ref = n->GetChild(art, 0);
		// concatenate prefixes
		child_ref->prefix.Concatenate(n->key[0], node->prefix);
		n->children[0] = nullptr;
		Node::Delete(node);
		node = child_ref;
	}
}

bool Node4::Merge(MergeInfo &info, idx_t depth, Node *&l_parent, idx_t l_pos) {

	Node4 *r_n = (Node4 *)info.r_node;

	for (idx_t i = 0; i < info.r_node->count; i++) {

		auto l_child_pos = info.l_node->GetChildPos(r_n->key[i]);
		if (!Node::MergeAtByte(info, depth, l_child_pos, i, r_n->key[i], l_parent, l_pos)) {
			return false;
		}
	}
	return true;
}

idx_t Node4::GetSize() {
	return 4;
}

} // namespace duckdb





namespace duckdb {

Node48::Node48() : Node(NodeType::N48) {
	for (idx_t i = 0; i < 256; i++) {
		child_index[i] = Node::EMPTY_MARKER;
	}
}

idx_t Node48::GetChildPos(uint8_t k) {
	if (child_index[k] == Node::EMPTY_MARKER) {
		return DConstants::INVALID_INDEX;
	} else {
		return k;
	}
}

idx_t Node48::GetChildGreaterEqual(uint8_t k, bool &equal) {
	for (idx_t pos = k; pos < 256; pos++) {
		if (child_index[pos] != Node::EMPTY_MARKER) {
			if (pos == k) {
				equal = true;
			} else {
				equal = false;
			}
			return pos;
		}
	}
	return DConstants::INVALID_INDEX;
}

idx_t Node48::GetMin() {
	for (idx_t i = 0; i < 256; i++) {
		if (child_index[i] != Node::EMPTY_MARKER) {
			return i;
		}
	}
	return DConstants::INVALID_INDEX;
}

idx_t Node48::GetNextPos(idx_t pos) {
	for (pos == DConstants::INVALID_INDEX ? pos = 0 : pos++; pos < 256; pos++) {
		if (child_index[pos] != Node::EMPTY_MARKER) {
			return pos;
		}
	}
	return Node::GetNextPos(pos);
}

Node *Node48::GetChild(ART &art, idx_t pos) {
	D_ASSERT(child_index[pos] != Node::EMPTY_MARKER);
	return children[child_index[pos]].Unswizzle(art);
}

void Node48::ReplaceChildPointer(idx_t pos, Node *node) {
	children[child_index[pos]] = node;
}

void Node48::InsertChild(Node *&node, uint8_t key_byte, Node *new_child) {
	auto n = (Node48 *)node;

	// Insert new child node into node
	if (node->count < 48) {
		// Insert element
		idx_t pos = n->count;
		if (n->children[pos]) {
			// find an empty position in the node list if the current position is occupied
			pos = 0;
			while (n->children[pos]) {
				pos++;
			}
		}
		n->children[pos] = new_child;
		n->child_index[key_byte] = pos;
		n->count++;
	} else {
		// Grow to Node256
		auto new_node = Node256::New();
		for (idx_t i = 0; i < 256; i++) {
			if (n->child_index[i] != Node::EMPTY_MARKER) {
				new_node->children[i] = n->children[n->child_index[i]];
				n->children[n->child_index[i]] = nullptr;
			}
		}
		new_node->count = n->count;
		new_node->prefix = move(n->prefix);
		Node::Delete(node);
		node = new_node;
		Node256::InsertChild(node, key_byte, new_child);
	}
}

void Node48::EraseChild(Node *&node, int pos, ART &art) {
	auto n = (Node48 *)(node);
	n->children[n->child_index[pos]].Reset();
	n->child_index[pos] = Node::EMPTY_MARKER;
	n->count--;
	if (node->count <= 12) {
		auto new_node = Node16::New();
		new_node->prefix = move(n->prefix);
		for (idx_t i = 0; i < 256; i++) {
			if (n->child_index[i] != Node::EMPTY_MARKER) {
				new_node->key[new_node->count] = i;
				new_node->children[new_node->count++] = n->children[n->child_index[i]];
				n->children[n->child_index[i]] = nullptr;
			}
		}
		Node::Delete(node);
		node = new_node;
	}
}

bool Node48::Merge(MergeInfo &info, idx_t depth, Node *&l_parent, idx_t l_pos) {

	Node48 *r_n = (Node48 *)info.r_node;

	for (idx_t i = 0; i < 256; i++) {
		if (r_n->child_index[i] != Node::EMPTY_MARKER) {

			auto l_child_pos = info.l_node->GetChildPos(i);
			auto key_byte = (uint8_t)i;
			if (!Node::MergeAtByte(info, depth, l_child_pos, i, key_byte, l_parent, l_pos)) {
				return false;
			}
		}
	}
	return true;
}

idx_t Node48::GetSize() {
	return 48;
}

} // namespace duckdb


namespace duckdb {

uint32_t Prefix::Size() const {
	return size;
}

bool Prefix::IsInlined() const {
	return size <= PREFIX_INLINE_BYTES;
}

uint8_t *Prefix::GetPrefixData() {
	return IsInlined() ? &value.inlined[0] : value.ptr;
}

const uint8_t *Prefix::GetPrefixData() const {
	return IsInlined() ? &value.inlined[0] : value.ptr;
}

uint8_t *Prefix::AllocatePrefix(uint32_t size) {
	Destroy();

	this->size = size;
	uint8_t *prefix;
	if (IsInlined()) {
		prefix = &value.inlined[0];
	} else {
		// allocate new prefix
		value.ptr = AllocateArray<uint8_t>(size);
		prefix = value.ptr;
	}
	return prefix;
}

Prefix::Prefix() : size(0) {
}

Prefix::Prefix(Key &key, uint32_t depth, uint32_t size) : size(0) {
	auto prefix = AllocatePrefix(size);

	// copy key to prefix
	idx_t prefix_idx = 0;
	for (idx_t i = depth; i < size + depth; i++) {
		prefix[prefix_idx++] = key.data[i];
	}
}

Prefix::Prefix(Prefix &other_prefix, uint32_t size) : size(0) {
	auto prefix = AllocatePrefix(size);

	// copy key to Prefix
	auto other_data = other_prefix.GetPrefixData();
	for (idx_t i = 0; i < size; i++) {
		prefix[i] = other_data[i];
	}
}

Prefix::~Prefix() {
	Destroy();
}

void Prefix::Destroy() {
	if (!IsInlined()) {
		DeleteArray<uint8_t>(value.ptr, size);
		size = 0;
	}
}

uint8_t &Prefix::operator[](idx_t idx) {
	D_ASSERT(idx < Size());
	return GetPrefixData()[idx];
}

Prefix &Prefix::operator=(const Prefix &src) {
	auto prefix = AllocatePrefix(src.size);

	// copy prefix
	auto src_prefix = src.GetPrefixData();
	for (idx_t i = 0; i < src.size; i++) {
		prefix[i] = src_prefix[i];
	}
	size = src.size;
	return *this;
}

Prefix &Prefix::operator=(Prefix &&other) noexcept {
	std::swap(size, other.size);
	std::swap(value, other.value);
	return *this;
}

void Prefix::Overwrite(uint32_t new_size, uint8_t *data) {
	if (new_size <= PREFIX_INLINE_BYTES) {
		// new entry would be inlined
		// inline the data and destroy the pointer
		auto prefix = AllocatePrefix(new_size);
		for (idx_t i = 0; i < new_size; i++) {
			prefix[i] = data[i];
		}
		DeleteArray<uint8_t>(data, new_size);
	} else {
		// new entry would not be inlined
		// take over the data directly
		Destroy();
		size = new_size;
		value.ptr = data;
	}
}

void Prefix::Concatenate(uint8_t key, Prefix &other) {
	auto new_length = size + 1 + other.size;
	// have to allocate space in our prefix array
	auto new_prefix = AllocateArray<uint8_t>(new_length);
	idx_t new_prefix_idx = 0;
	// 1) add the to-be deleted node's prefix
	for (uint32_t i = 0; i < other.size; i++) {
		new_prefix[new_prefix_idx++] = other[i];
	}
	// 2) now move the current key as part of the prefix
	new_prefix[new_prefix_idx++] = key;
	// 3) move the existing prefix (if any)
	auto prefix = GetPrefixData();
	for (uint32_t i = 0; i < size; i++) {
		new_prefix[new_prefix_idx++] = prefix[i];
	}
	Overwrite(new_length, new_prefix);
}

uint8_t Prefix::Reduce(uint32_t n) {
	auto new_size = size - n - 1;
	auto prefix = GetPrefixData();
	auto key = prefix[n];
	if (new_size == 0) {
		Destroy();
		size = 0;
		return key;
	}
	auto new_prefix = AllocateArray<uint8_t>(new_size);
	for (idx_t i = 0; i < new_size; i++) {
		new_prefix[i] = prefix[i + n + 1];
	}
	Overwrite(new_size, new_prefix);
	return key;
}

void Prefix::Serialize(duckdb::MetaBlockWriter &writer) {
	writer.Write(size);
	auto prefix = GetPrefixData();
	writer.WriteData(prefix, size);
}

void Prefix::Deserialize(duckdb::MetaBlockReader &reader) {
	auto prefix_size = reader.Read<uint32_t>();
	auto prefix = AllocatePrefix(prefix_size);
	this->size = prefix_size;
	reader.ReadData(prefix, size);
}

uint32_t Prefix::KeyMismatchPosition(Key &key, uint64_t depth) {
	uint64_t pos;
	auto prefix = GetPrefixData();
	for (pos = 0; pos < size; pos++) {
		if (key[depth + pos] != prefix[pos]) {
			return pos;
		}
	}
	return pos;
}

uint32_t Prefix::MismatchPosition(Prefix &other) {
	auto prefix = GetPrefixData();
	auto other_data = other.GetPrefixData();
	for (idx_t i = 0; i < size; i++) {
		if (prefix[i] != other_data[i]) {
			return i;
		}
	}
	return size;
}

} // namespace duckdb


namespace duckdb {
SwizzleablePointer::~SwizzleablePointer() {
	if (pointer) {
		if (!IsSwizzled()) {
			Node::Delete((Node *)pointer);
		}
	}
}

SwizzleablePointer::SwizzleablePointer(duckdb::MetaBlockReader &reader) {
	idx_t block_id = reader.Read<block_id_t>();
	idx_t offset = reader.Read<uint32_t>();
	if (block_id == DConstants::INVALID_INDEX || offset == DConstants::INVALID_INDEX) {
		pointer = 0;
		return;
	}
	idx_t pointer_size = sizeof(pointer) * 8;
	pointer = block_id;
	// This assumes high 32 bits of pointer are zero.
	pointer = pointer << (pointer_size / 2);
	D_ASSERT((pointer >> (pointer_size / 2)) == block_id);
	pointer += offset;
	// Set the left most bit to indicate this is a swizzled pointer and send it back to the mother-ship
	uint64_t mask = 1;
	mask = mask << (pointer_size - 1);
	// This assumes the 33rd most significant bit of the block_id is zero.
	pointer |= mask;
}

SwizzleablePointer &SwizzleablePointer::operator=(const Node *ptr) {
	// If the object already has a non-swizzled pointer, this will leak memory.
	//
	// TODO: If enabled, this assert will fire, indicating a possible leak. If an exception
	// is thrown here, it will cause a double-free. There is some work to do to make all this safer.
	// D_ASSERT(empty() || IsSwizzled());
	if (sizeof(ptr) == 4) {
		pointer = (uint32_t)(size_t)ptr;
	} else {
		pointer = (uint64_t)ptr;
	}
	return *this;
}

bool operator!=(const SwizzleablePointer &s_ptr, const uint64_t &ptr) {
	return (s_ptr.pointer != ptr);
}

BlockPointer SwizzleablePointer::GetSwizzledBlockInfo() {
	D_ASSERT(IsSwizzled());
	idx_t pointer_size = sizeof(pointer) * 8;
	// This is destructive. Pointer will be invalid after this operation.
	// That's okay because this is only ever called from Unswizzle.
	pointer = pointer & ~(1ULL << (pointer_size - 1));
	uint32_t block_id = pointer >> (pointer_size / 2);
	uint32_t offset = pointer & 0xffffffff;
	return {block_id, offset};
}

bool SwizzleablePointer::IsSwizzled() {
	idx_t pointer_size = sizeof(pointer) * 8;
	return (pointer >> (pointer_size - 1)) & 1;
}

void SwizzleablePointer::Reset() {
	if (pointer) {
		if (!IsSwizzled()) {
			Node::Delete((Node *)pointer);
		}
	}
	*this = nullptr;
}

Node *SwizzleablePointer::Unswizzle(ART &art) {
	if (IsSwizzled()) {
		// This means our pointer is not yet in memory, gotta deserialize this
		// first we unset the bae
		auto block_info = GetSwizzledBlockInfo();
		*this = Node::Deserialize(art, block_info.block_id, block_info.offset);
	}
	return (Node *)pointer;
}

BlockPointer SwizzleablePointer::Serialize(ART &art, duckdb::MetaBlockWriter &writer) {
	if (pointer) {
		Unswizzle(art);
		return ((Node *)pointer)->Serialize(art, writer);
	} else {
		return {(block_id_t)DConstants::INVALID_INDEX, (uint32_t)DConstants::INVALID_INDEX};
	}
}

} // namespace duckdb











namespace duckdb {

using ValidityBytes = JoinHashTable::ValidityBytes;
using ScanStructure = JoinHashTable::ScanStructure;
using ProbeSpill = JoinHashTable::ProbeSpill;
using ProbeSpillLocalState = JoinHashTable::ProbeSpillLocalAppendState;

JoinHashTable::JoinHashTable(BufferManager &buffer_manager, const vector<JoinCondition> &conditions,
                             vector<LogicalType> btypes, JoinType type)
    : buffer_manager(buffer_manager), conditions(conditions), build_types(move(btypes)), entry_size(0), tuple_size(0),
      vfound(Value::BOOLEAN(false)), join_type(type), finalized(false), has_null(false), external(false), radix_bits(4),
      tuples_per_round(0), partition_start(0), partition_end(0) {
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

	// compute the per-block capacity of this HT
	idx_t block_capacity = Storage::BLOCK_SIZE / entry_size;
	block_collection = make_unique<RowDataCollection>(buffer_manager, block_capacity, entry_size);
	string_heap = make_unique<RowDataCollection>(buffer_manager, (idx_t)Storage::BLOCK_SIZE, 1, true);
	swizzled_block_collection = block_collection->CloneEmpty();
	swizzled_string_heap = string_heap->CloneEmpty();
}

JoinHashTable::~JoinHashTable() {
}

void JoinHashTable::Merge(JoinHashTable &other) {
	block_collection->Merge(*other.block_collection);
	swizzled_block_collection->Merge(*other.swizzled_block_collection);
	if (!layout.AllConstant()) {
		string_heap->Merge(*other.string_heap);
		swizzled_string_heap->Merge(*other.swizzled_string_heap);
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

	lock_guard<mutex> lock(partitioned_data_lock);
	if (partition_block_collections.empty()) {
		D_ASSERT(partition_string_heaps.empty());
		// Move partitions to this HT
		for (idx_t p = 0; p < other.partition_block_collections.size(); p++) {
			partition_block_collections.push_back(move(other.partition_block_collections[p]));
			if (!layout.AllConstant()) {
				partition_string_heaps.push_back(move(other.partition_string_heaps[p]));
			}
		}
		return;
	}

	// Should have same number of partitions
	D_ASSERT(partition_block_collections.size() == other.partition_block_collections.size());
	D_ASSERT(partition_string_heaps.size() == other.partition_string_heaps.size());
	for (idx_t idx = 0; idx < other.partition_block_collections.size(); idx++) {
		partition_block_collections[idx]->Merge(*other.partition_block_collections[idx]);
		if (!layout.AllConstant()) {
			partition_string_heaps[idx]->Merge(*other.partition_string_heaps[idx]);
		}
	}
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

idx_t JoinHashTable::PrepareKeys(DataChunk &keys, unique_ptr<UnifiedVectorFormat[]> &key_data,
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

void JoinHashTable::Build(DataChunk &keys, DataChunk &payload) {
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
		info.correlated_counts->AddChunk(info.group_chunk, info.correlated_payload, AggregateType::NON_DISTINCT);
	}

	// prepare the keys for processing
	unique_ptr<UnifiedVectorFormat[]> key_data;
	const SelectionVector *current_sel;
	SelectionVector sel(STANDARD_VECTOR_SIZE);
	idx_t added_count = PrepareKeys(keys, key_data, current_sel, sel, true);
	if (added_count < keys.size()) {
		has_null = true;
	}
	if (added_count == 0) {
		return;
	}

	// build out the buffer space
	Vector addresses(LogicalType::POINTER);
	auto key_locations = FlatVector::GetData<data_ptr_t>(addresses);
	auto handles = block_collection->Build(added_count, key_locations, nullptr, current_sel);

	// hash the keys and obtain an entry in the list
	// note that we only hash the keys used in the equality comparison
	Vector hash_values(LogicalType::HASH);
	Hash(keys, *current_sel, added_count, hash_values);

	// build a chunk so we can handle nested types that need more than Orrification
	DataChunk source_chunk;
	source_chunk.InitializeEmpty(layout.GetTypes());

	vector<UnifiedVectorFormat> source_data;
	source_data.reserve(layout.ColumnCount());

	// serialize the keys to the key locations
	for (idx_t i = 0; i < keys.ColumnCount(); i++) {
		source_chunk.data[i].Reference(keys.data[i]);
		source_data.emplace_back(move(key_data[i]));
	}
	// now serialize the payload
	D_ASSERT(build_types.size() == payload.ColumnCount());
	for (idx_t i = 0; i < payload.ColumnCount(); i++) {
		source_chunk.data[source_data.size()].Reference(payload.data[i]);
		UnifiedVectorFormat pdata;
		payload.data[i].ToUnifiedFormat(payload.size(), pdata);
		source_data.emplace_back(move(pdata));
	}
	if (IsRightOuterJoin(join_type)) {
		// for FULL/RIGHT OUTER joins initialize the "found" boolean to false
		source_chunk.data[source_data.size()].Reference(vfound);
		UnifiedVectorFormat fdata;
		vfound.ToUnifiedFormat(keys.size(), fdata);
		source_data.emplace_back(move(fdata));
	}

	// serialise the hashes at the end
	source_chunk.data[source_data.size()].Reference(hash_values);
	UnifiedVectorFormat hdata;
	hash_values.ToUnifiedFormat(keys.size(), hdata);
	source_data.emplace_back(move(hdata));

	source_chunk.SetCardinality(keys);

	RowOperations::Scatter(source_chunk, source_data.data(), layout, addresses, *string_heap, *current_sel,
	                       added_count);
}

template <bool PARALLEL>
static inline void InsertHashesLoop(atomic<data_ptr_t> pointers[], const hash_t indices[], const idx_t count,
                                    const data_ptr_t key_locations[], const idx_t pointer_offset) {
	for (idx_t i = 0; i < count; i++) {
		auto index = indices[i];
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
	idx_t count = external ? MaxValue<idx_t>(tuples_per_round, Count()) : Count();
	idx_t capacity = PointerTableCapacity(count);
	// size needs to be a power of 2
	D_ASSERT((capacity & (capacity - 1)) == 0);
	bitmask = capacity - 1;

	if (!hash_map.get()) {
		// allocate the HT if not yet done
		hash_map = buffer_manager.GetBufferAllocator().Allocate(capacity * sizeof(data_ptr_t));
	}
	D_ASSERT(hash_map.GetSize() == capacity * sizeof(data_ptr_t));

	// initialize HT with all-zero entries
	memset(hash_map.get(), 0, capacity * sizeof(data_ptr_t));
}

void JoinHashTable::Finalize(idx_t block_idx_start, idx_t block_idx_end, bool parallel) {
	// Pointer table should be allocated
	D_ASSERT(hash_map.get());

	vector<BufferHandle> local_pinned_handles;

	Vector hashes(LogicalType::HASH);
	auto hash_data = FlatVector::GetData<hash_t>(hashes);
	data_ptr_t key_locations[STANDARD_VECTOR_SIZE];
	// now construct the actual hash table; scan the nodes
	// as we can the nodes we pin all the blocks of the HT and keep them pinned until the HT is destroyed
	// this is so that we can keep pointers around to the blocks
	for (idx_t block_idx = block_idx_start; block_idx < block_idx_end; block_idx++) {
		auto &block = block_collection->blocks[block_idx];
		auto handle = buffer_manager.Pin(block->block);
		data_ptr_t dataptr = handle.Ptr();
		idx_t entry = 0;
		while (entry < block->count) {
			// fetch the next vector of entries from the blocks
			idx_t next = MinValue<idx_t>(STANDARD_VECTOR_SIZE, block->count - entry);
			for (idx_t i = 0; i < next; i++) {
				hash_data[i] = Load<hash_t>((data_ptr_t)(dataptr + pointer_offset));
				key_locations[i] = dataptr;
				dataptr += entry_size;
			}
			// now insert into the hash table
			InsertHashes(hashes, next, key_locations, parallel);

			entry += next;
		}
		local_pinned_handles.push_back(move(handle));
	}

	lock_guard<mutex> lock(pinned_handles_lock);
	for (auto &local_pinned_handle : local_pinned_handles) {
		pinned_handles.push_back(move(local_pinned_handle));
	}
}

unique_ptr<ScanStructure> JoinHashTable::InitializeScanStructure(DataChunk &keys, const SelectionVector *&current_sel) {
	D_ASSERT(Count() > 0); // should be handled before
	D_ASSERT(finalized);

	// set up the scan structure
	auto ss = make_unique<ScanStructure>(*this);

	if (join_type != JoinType::INNER) {
		ss->found_match = unique_ptr<bool[]>(new bool[STANDARD_VECTOR_SIZE]);
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
	RowOperations::Gather(pointers, sel_vector, result, result_vector, count, ht.layout, col_no);
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
		auto &mask = FlatVector::Validity(vector);
		mask.SetAllInvalid(input.size());
		for (idx_t j = 0; j < result_count; j++) {
			mask.SetValid(result_sel.get_index(j));
		}
		// for the remaining values we fetch the values
		GatherResult(vector, result_sel, result_sel, result_count, i + ht.condition_types.size());
	}
	result.SetCardinality(input.size());

	// like the SEMI, ANTI and MARK join types, the SINGLE join only ever does one pass over the HT per input chunk
	finished = true;
}

idx_t JoinHashTable::ScanFullOuter(JoinHTScanState &state, Vector &addresses) {
	// scan the HT starting from the current position and check which rows from the build side did not find a match
	auto key_locations = FlatVector::GetData<data_ptr_t>(addresses);
	idx_t found_entries = 0;
	for (; state.block_position < block_collection->blocks.size(); state.block_position++, state.position = 0) {
		auto &block = block_collection->blocks[state.block_position];
		auto handle = buffer_manager.Pin(block->block);
		auto baseptr = handle.Ptr();
		for (; state.position < block->count; state.position++, state.scan_index++) {
			auto tuple_base = baseptr + state.position * entry_size;
			auto found_match = Load<bool>(tuple_base + tuple_size);
			if (!found_match) {
				key_locations[found_entries++] = tuple_base;
				if (found_entries == STANDARD_VECTOR_SIZE) {
					state.position++;
					state.scan_index++;
					break;
				}
			}
		}
		if (found_entries == STANDARD_VECTOR_SIZE) {
			break;
		}
	}
	return found_entries;
}

void JoinHashTable::GatherFullOuter(DataChunk &result, Vector &addresses, idx_t found_entries) {
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
		RowOperations::Gather(addresses, sel_vector, vector, sel_vector, found_entries, layout, col_no);
	}
}

idx_t JoinHashTable::FillWithHTOffsets(data_ptr_t *key_locations, JoinHTScanState &state) {
	// iterate over blocks
	idx_t key_count = 0;
	while (state.block_position < block_collection->blocks.size()) {
		auto &block = block_collection->blocks[state.block_position];
		auto handle = buffer_manager.Pin(block->block);
		auto base_ptr = handle.Ptr();
		// go through all the tuples within this block
		while (state.position < block->count) {
			auto tuple_base = base_ptr + state.position * entry_size;
			// store its locations
			key_locations[key_count++] = tuple_base;
			state.position++;
		}
		state.block_position++;
		state.position = 0;
	}
	return key_count;
}

void JoinHashTable::PinAllBlocks() {
	for (auto &block : block_collection->blocks) {
		pinned_handles.push_back(buffer_manager.Pin(block->block));
	}
}

void JoinHashTable::SwizzleBlocks() {
	if (block_collection->count == 0) {
		return;
	}

	if (layout.AllConstant()) {
		// No heap blocks! Just merge fixed-size data
		swizzled_block_collection->Merge(*block_collection);
		return;
	}

	// We create one heap block per data block and swizzle the pointers
	auto &heap_blocks = string_heap->blocks;
	idx_t heap_block_idx = 0;
	idx_t heap_block_remaining = heap_blocks[heap_block_idx]->count;
	for (auto &data_block : block_collection->blocks) {
		if (heap_block_remaining == 0) {
			heap_block_remaining = heap_blocks[++heap_block_idx]->count;
		}

		// Pin the data block and swizzle the pointers within the rows
		auto data_handle = buffer_manager.Pin(data_block->block);
		auto data_ptr = data_handle.Ptr();
		RowOperations::SwizzleColumns(layout, data_ptr, data_block->count);

		// We want to copy as little of the heap data as possible, check how the data and heap blocks line up
		if (heap_block_remaining >= data_block->count) {
			// Easy: current heap block contains all strings for this data block, just copy (reference) the block
			swizzled_string_heap->blocks.emplace_back(heap_blocks[heap_block_idx]->Copy());
			swizzled_string_heap->blocks.back()->count = data_block->count;

			// Swizzle the heap pointer
			auto heap_handle = buffer_manager.Pin(swizzled_string_heap->blocks.back()->block);
			auto heap_ptr = Load<data_ptr_t>(data_ptr + layout.GetHeapOffset());
			auto heap_offset = heap_ptr - heap_handle.Ptr();
			RowOperations::SwizzleHeapPointer(layout, data_ptr, heap_ptr, data_block->count, heap_offset);

			// Update counter
			heap_block_remaining -= data_block->count;
		} else {
			// Strings for this data block are spread over the current heap block and the next (and possibly more)
			idx_t data_block_remaining = data_block->count;
			vector<std::pair<data_ptr_t, idx_t>> ptrs_and_sizes;
			idx_t total_size = 0;
			while (data_block_remaining > 0) {
				if (heap_block_remaining == 0) {
					heap_block_remaining = heap_blocks[++heap_block_idx]->count;
				}
				auto next = MinValue<idx_t>(data_block_remaining, heap_block_remaining);

				// Figure out where to start copying strings, and how many bytes we need to copy
				auto heap_start_ptr = Load<data_ptr_t>(data_ptr + layout.GetHeapOffset());
				auto heap_end_ptr =
				    Load<data_ptr_t>(data_ptr + layout.GetHeapOffset() + (next - 1) * layout.GetRowWidth());
				idx_t size = heap_end_ptr - heap_start_ptr + Load<uint32_t>(heap_end_ptr);
				ptrs_and_sizes.emplace_back(heap_start_ptr, size);
				D_ASSERT(size <= heap_blocks[heap_block_idx]->byte_offset);

				// Swizzle the heap pointer
				RowOperations::SwizzleHeapPointer(layout, data_ptr, heap_start_ptr, next, total_size);
				total_size += size;

				// Update where we are in the data and heap blocks
				data_ptr += next * layout.GetRowWidth();
				data_block_remaining -= next;
				heap_block_remaining -= next;
			}

			// Finally, we allocate a new heap block and copy data to it
			swizzled_string_heap->blocks.emplace_back(
			    make_unique<RowDataBlock>(buffer_manager, MaxValue<idx_t>(total_size, (idx_t)Storage::BLOCK_SIZE), 1));
			auto new_heap_handle = buffer_manager.Pin(swizzled_string_heap->blocks.back()->block);
			auto new_heap_ptr = new_heap_handle.Ptr();
			for (auto &ptr_and_size : ptrs_and_sizes) {
				memcpy(new_heap_ptr, ptr_and_size.first, ptr_and_size.second);
				new_heap_ptr += ptr_and_size.second;
			}
		}
	}

	// We're done with variable-sized data, now just merge the fixed-size data
	swizzled_block_collection->Merge(*block_collection);
	D_ASSERT(swizzled_block_collection->blocks.size() == swizzled_string_heap->blocks.size());

	// Update counts and cleanup
	swizzled_string_heap->count = string_heap->count;
	string_heap->Clear();
}

void JoinHashTable::UnswizzleBlocks() {
	auto &blocks = swizzled_block_collection->blocks;
	auto &heap_blocks = swizzled_string_heap->blocks;
#ifdef DEBUG
	if (!layout.AllConstant()) {
		D_ASSERT(blocks.size() == heap_blocks.size());
		D_ASSERT(swizzled_block_collection->count == swizzled_string_heap->count);
	}
#endif

	for (idx_t block_idx = 0; block_idx < blocks.size(); block_idx++) {
		auto &data_block = blocks[block_idx];

		if (!layout.AllConstant()) {
			auto block_handle = buffer_manager.Pin(data_block->block);

			auto &heap_block = heap_blocks[block_idx];
			D_ASSERT(data_block->count == heap_block->count);
			auto heap_handle = buffer_manager.Pin(heap_block->block);

			// Unswizzle and move
			RowOperations::UnswizzlePointers(layout, block_handle.Ptr(), heap_handle.Ptr(), data_block->count);
			string_heap->blocks.push_back(move(heap_block));
			string_heap->pinned_blocks.push_back(move(heap_handle));
		}

		// Fixed size stuff can just be moved
		block_collection->blocks.push_back(move(data_block));
	}

	// Update counts and clean up
	block_collection->count = swizzled_block_collection->count;
	string_heap->count = swizzled_string_heap->count;
	swizzled_block_collection->Clear();
	swizzled_string_heap->Clear();

	D_ASSERT(SwizzledCount() == 0);
}

void JoinHashTable::ComputePartitionSizes(ClientConfig &config, vector<unique_ptr<JoinHashTable>> &local_hts,
                                          idx_t max_ht_size) {
	external = true;

	// First set the number of tuples in the HT per partitioned round
	total_count = 0;
	idx_t total_size = 0;
	for (auto &ht : local_hts) {
		// TODO: SizeInBytes / SwizzledSize overestimates size by a lot because we make extra references of heap blocks
		//  Need to compute this more accurately
		total_count += ht->Count() + ht->SwizzledCount();
		total_size += ht->SizeInBytes() + ht->SwizzledSize();
	}

	if (total_count == 0) {
		return;
	}

	total_size += PointerTableCapacity(total_count) * sizeof(data_ptr_t);
	idx_t avg_tuple_size = total_size / total_count;
	tuples_per_round = max_ht_size / avg_tuple_size;

	if (config.force_external) {
		// For force_external we do three rounds to test all code paths
		tuples_per_round = (total_count + 2) / 3;
	}

	// Set the number of radix bits (minimum 4, maximum 8)
	for (; radix_bits < 8; radix_bits++) {
		auto num_partitions = RadixPartitioning::NumberOfPartitions(radix_bits);
		auto avg_partition_size = total_size / num_partitions;

		// We aim for at least 8 partitions per probe round (tweaked experimentally)
		if (avg_partition_size * 8 < max_ht_size) {
			break;
		}
	}
}

void JoinHashTable::Partition(JoinHashTable &global_ht) {
#ifdef DEBUG
	D_ASSERT(layout.ColumnCount() == global_ht.layout.ColumnCount());
	for (idx_t col_idx = 0; col_idx < layout.ColumnCount(); col_idx++) {
		D_ASSERT(layout.GetTypes()[col_idx] == global_ht.layout.GetTypes()[col_idx]);
	}
#endif

	// Swizzle and Partition
	SwizzleBlocks();
	RadixPartitioning::PartitionRowData(global_ht.buffer_manager, global_ht.layout, global_ht.pointer_offset,
	                                    *swizzled_block_collection, *swizzled_string_heap, partition_block_collections,
	                                    partition_string_heaps, global_ht.radix_bits);

	// Add to global HT
	global_ht.Merge(*this);
}

void JoinHashTable::Reset() {
	pinned_handles.clear();
	block_collection->Clear();
	string_heap->Clear();
	finalized = false;
}

bool JoinHashTable::PrepareExternalFinalize() {
	idx_t num_partitions = RadixPartitioning::NumberOfPartitions(radix_bits);
	if (partition_block_collections.empty() || partition_end == num_partitions) {
		return false;
	}

	if (finalized) {
		Reset();
	}

	// Determine how many partitions we can do next (at least one)
	idx_t next = 0;
	idx_t count = 0;
	partition_start = partition_end;
	for (idx_t p = partition_start; p < num_partitions; p++) {
		auto partition_count = partition_block_collections[p]->count;
		if (partition_count != 0 && count != 0 && count + partition_count > tuples_per_round) {
			// We skip over empty partitions (partition_count != 0),
			// and need to have at least one partition (count != 0)
			break;
		}
		next++;
		count += partition_count;
	}
	partition_end += next;

	// Move specific partitions to the swizzled_... collections so they can be unswizzled
	D_ASSERT(SwizzledCount() == 0);
	for (idx_t p = partition_start; p < partition_end; p++) {
		auto &p_block_collection = *partition_block_collections[p];
		if (!layout.AllConstant()) {
			auto &p_string_heap = *partition_string_heaps[p];
			D_ASSERT(p_block_collection.count == p_string_heap.count);
			swizzled_string_heap->Merge(p_string_heap);
			// Remove after merging
			partition_string_heaps[p] = nullptr;
		}
		swizzled_block_collection->Merge(p_block_collection);
		// Remove after merging
		partition_block_collections[p] = nullptr;
	}
	D_ASSERT(count == SwizzledCount());

	// Unswizzle them
	D_ASSERT(Count() == 0);
	UnswizzleBlocks();
	D_ASSERT(count == Count());

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
	if (ht.total_count - ht.Count() <= ht.tuples_per_round) {
		// No need to partition as we will only have one more probe round
		partitioned = false;
	} else {
		// More than one probe round to go, so we need to partition
		partitioned = true;
		global_partitions =
		    make_unique<RadixPartitionedColumnData>(context, probe_types, ht.radix_bits, probe_types.size() - 1);
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
		local_partition_append_states.emplace_back(make_unique<PartitionedColumnDataAppendState>());
		local_partitions.back()->InitializeAppendState(*local_partition_append_states.back());

		result.local_partition = local_partitions.back().get();
		result.local_partition_append_state = local_partition_append_states.back().get();
	} else {
		local_spill_collections.emplace_back(
		    make_unique<ColumnDataCollection>(BufferManager::GetBufferManager(context), probe_types));
		local_spill_append_states.emplace_back(make_unique<ColumnDataAppendState>());
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
			    make_unique<ColumnDataCollection>(BufferManager::GetBufferManager(context), probe_types);
		} else {
			global_spill_collection = move(local_spill_collections[0]);
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
			    make_unique<ColumnDataCollection>(BufferManager::GetBufferManager(context), probe_types);
		} else {
			// Move specific partitions to the global spill collection
			global_spill_collection = move(partitions[ht.partition_start]);
			for (idx_t i = ht.partition_start + 1; i < ht.partition_end; i++) {
				global_spill_collection->Combine(*partitions[i]);
			}
		}
	}
	consumer = make_unique<ColumnDataConsumer>(*global_spill_collection, column_ids);
	consumer->InitializeScan();
}

} // namespace duckdb



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
    : function(move(function)), bind_data(bind_data), child_count(child_count), payload_size(payload_size),
      aggr_type(aggr_type), return_type(return_type), filter(filter) {
}

AggregateObject::AggregateObject(BoundAggregateExpression *aggr)
    : AggregateObject(aggr->function, aggr->bind_info.get(), aggr->children.size(),
                      AlignValue(aggr->function.state_size()), aggr->aggr_type, aggr->return_type.InternalType(),
                      aggr->filter.get()) {
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
			filter_data[aggr_idx] = make_unique<AggregateFilterData>(context, *aggr.filter, payload_types);
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
    : indices(move(indices)), aggregates(aggregates) {
	table_count = CreateTableIndexMap();

	const idx_t aggregate_count = aggregates.size();

	total_child_count = 0;
	for (idx_t i = 0; i < aggregate_count; i++) {
		auto &aggregate = (BoundAggregateExpression &)*aggregates[i];

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
		auto &aggregate = (BoundAggregateExpression &)*data.info.aggregates[i];

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
		distinct_output_chunks[table_idx] = make_unique<DataChunk>();
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
		auto &aggregate = (BoundAggregateExpression &)*info.aggregates[i];

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
		grouped_aggregate_data[table_idx] = make_unique<GroupedAggregateData>();
		grouped_aggregate_data[table_idx]->InitializeDistinct(info.aggregates[i], group_expressions);
		radix_tables[table_idx] =
		    make_unique<RadixPartitionedHashTable>(grouping_set, *grouped_aggregate_data[table_idx]);

		// Fill the chunk_types (only contains the payload of the distinct aggregates)
		vector<LogicalType> chunk_types;
		for (auto &child_p : aggregate.children) {
			chunk_types.push_back(child_p->return_type);
		}
	}
}

using aggr_ref_t = std::reference_wrapper<BoundAggregateExpression>;

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
			auto &other_child = (BoundReferenceExpression &)*other.children[i];
			auto &aggr_child = (BoundReferenceExpression &)*aggr.children[i];
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
		auto &aggregate = (BoundAggregateExpression &)*aggregates[agg_idx];

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

const vector<idx_t> &DistinctAggregateCollectionInfo::Indices() const {
	return this->indices;
}

static vector<idx_t> GetDistinctIndices(vector<unique_ptr<Expression>> &aggregates) {
	vector<idx_t> distinct_indices;
	for (idx_t i = 0; i < aggregates.size(); i++) {
		auto &aggregate = aggregates[i];
		auto &aggr = (BoundAggregateExpression &)*aggregate;
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
	return make_unique<DistinctAggregateCollectionInfo>(aggregates, move(indices));
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
