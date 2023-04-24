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

#include "conversion.h" 
#include <velox/vector/arrow/Abi.h>
#include <velox/vector/arrow/Bridge.h>

namespace facebook::velox::py {

namespace py = pybind11;

void addConversionBindings(py::module& m, bool asModuleLocalDefinitions) {
   m.def("export_to_arrow", [](VectorPtr& inputVector) {
    auto arrowArray = new ArrowArray();
    std::shared_ptr<facebook::velox::memory::MemoryPool> pool_{
        facebook::velox::memory::addDefaultLeafMemoryPool()};
    facebook::velox::exportToArrow(inputVector, *arrowArray, pool_.get());
    return reinterpret_cast<uintptr_t>(arrowArray);
  });

  m.def(
      "import_from_arrow",
      [](uintptr_t arrowArrayPtr, uintptr_t arrowSchemaPtr) {
        auto arrowArray = reinterpret_cast<ArrowArray*>(arrowArrayPtr);
        auto arrowSchema = reinterpret_cast<ArrowSchema*>(arrowSchemaPtr);
        std::shared_ptr<facebook::velox::memory::MemoryPool> pool_{
            facebook::velox::memory::addDefaultLeafMemoryPool()};
        return importFromArrowAsOwner(*arrowSchema, *arrowArray, pool_.get());
      });
}
} // namespace facebook::velox::py
