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

#include "serde.h" // @manual
#include "context.h"

#include <velox/vector/VectorSaver.h>


namespace facebook::velox::py {

namespace py = pybind11;

namespace {
    static VectorPtr pyRestoreVectorFromFileHelper(
        const char* FOLLY_NONNULL filePath) {
        using namespace facebook::velox;
        memory::MemoryPool* pool = PyVeloxContext::getInstance().pool();
        return restoreVectorFromFile(filePath, pool);
    }
}


void addSerdeBindings(py::module& m, bool asModuleLocalDefinitions) {
    using namespace facebook::velox;

  m.def(
      "save_vector",
      &saveVectorToFile,
      "Serializes the vector into binary format and writes it to a new file.");
  m.def(
      "load_vector",
      &pyRestoreVectorFromFileHelper,
      "Reads and deserializes a vector from a file stored by save_vector.");
}

}