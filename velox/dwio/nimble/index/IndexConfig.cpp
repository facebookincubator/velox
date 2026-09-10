/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
#include "velox/dwio/nimble/index/IndexConfig.h"

#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/index/IndexSerialization.h"

namespace facebook::nimble::index {

std::string_view toString(IndexFamily family) {
  switch (family) {
    case IndexFamily::Cluster:
      return "Cluster";
    case IndexFamily::Dense:
      return "Dense";
  }
  NIMBLE_UNREACHABLE("Unsupported index family: {}", static_cast<int>(family));
}

IndexFamily toIndexFamily(serialization::IndexFamily family) {
  switch (family) {
    case serialization::IndexFamily_Cluster:
      return IndexFamily::Cluster;
    case serialization::IndexFamily_Dense:
      return IndexFamily::Dense;
  }
  NIMBLE_UNREACHABLE(
      "Unsupported serialized index family: {}", static_cast<int>(family));
}

serialization::IndexFamily toIndexFamily(IndexFamily family) {
  switch (family) {
    case IndexFamily::Cluster:
      return serialization::IndexFamily_Cluster;
    case IndexFamily::Dense:
      return serialization::IndexFamily_Dense;
  }
  NIMBLE_UNREACHABLE("Unsupported index family: {}", static_cast<int>(family));
}

} // namespace facebook::nimble::index
