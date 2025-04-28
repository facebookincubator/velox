#include "Arrow.h"
#include <velox/vector/arrow/Bridge.h>

namespace velox4j {
using namespace facebook::velox;

namespace {

void slice(VectorPtr& in) {
  auto* rowBase = in->as<RowVector>();
  if (!rowBase) {
    return;
  }
  for (auto& child : rowBase->children()) {
    if (child->size() > rowBase->size()) {
      child = child->slice(0, rowBase->size());
    }
  }
}

void flatten(VectorPtr& in) {
  facebook::velox::BaseVector::flattenVector(in);
}

ArrowOptions makeOptions() {
  ArrowOptions options;
  options.timestampUnit = static_cast<TimestampUnit>(6);
  return options;
}
} // namespace

void fromBaseVectorToArrow(
    VectorPtr vector,
    ArrowSchema* cSchema,
    ArrowArray* cArray) {
  flatten(vector);
  slice(vector);
  auto options = makeOptions();
  exportToArrow(vector, *cSchema, options);
  exportToArrow(vector, *cArray, vector->pool(), options);
}

VectorPtr fromArrowToBaseVector(
    memory::MemoryPool* pool,
    ArrowSchema* cSchema,
    ArrowArray* cArray) {
  auto options = makeOptions();
  return importFromArrowAsOwner(*cSchema, *cArray, pool);
}
} // namespace velox4j
