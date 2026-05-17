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

#include "velox/experimental/torchwave/KernelHandlerRegistration.h"

#include <c10/util/CallOnce.h>
#include <c10/util/string_view.h>
#include <torch/nativert/kernels/ETCallDelegateKernel.h>
#include <torch/nativert/kernels/KernelFactory.h>
#include "velox/experimental/torchwave/NativertDelegateExecutor.h"

namespace torch::wave {

void registerTorchwaveHandler() {
  static c10::once_flag flag;
  c10::call_once(flag, []() {
    using OpKernelPtr = nativert::KernelFactoryHandler::OpKernelPtr;
    using DelegateExecutorPtr =
        nativert::KernelFactoryHandler::DelegateExecutorPtr;

    nativert::KernelFactory::registerHandler(
        "torchwave_delegate",
        nativert::KernelFactoryHandler(
            [](const nativert::Node& node,
               const nativert::ExecutorConfig& /*executorConfig*/) {
              if (!c10::starts_with(
                      node.target(),
                      "torch.ops.higher_order.executorch_call_delegate")) {
                return false;
              }
              if (node.attributes().empty()) {
                return false;
              }
              const std::string* path =
                  std::get_if<std::string>(&node.attributes()[0].value);
              return path && c10::ends_with(*path, "-torchwave");
            },
            [](const nativert::Node& node,
               // NOLINTNEXTLINE(performance-unnecessary-value-param)
               std::shared_ptr<nativert::Weights> weights,
               const nativert::ExecutorConfig& executorConfig,
               caffe2::serialize::PyTorchStreamReader* packageReader)
                -> std::pair<OpKernelPtr, DelegateExecutorPtr> {
              auto delegateExecutor =
                  std::make_unique<NativertDelegateExecutor>(
                      node, weights, executorConfig, packageReader);

              return {
                  std::make_unique<nativert::ETCallDelegateKernel>(
                      &node, *delegateExecutor),
                  std::move(delegateExecutor)};
            }));
  });
}

} // namespace torch::wave
