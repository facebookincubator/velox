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

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <caffe2/serialize/inline_container.h> // @manual=//caffe2/caffe2/serialize:inline_container
#include "velox/experimental/torchwave/Execute.h"

namespace torch::wave {

std::vector<std::string> getModelNames(
    caffe2::serialize::PyTorchStreamReader& reader);

LoadedModel loadPt2Model(
    std::shared_ptr<caffe2::serialize::PyTorchStreamReader> reader,
    const std::string& modelName);

} // namespace torch::wave
