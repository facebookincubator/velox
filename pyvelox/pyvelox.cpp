/*
* Copyright (c) Meta Platforms, Inc. and affiliates.
* All rights reserved.
*
* This source code is licensed under the BSD-style license found in the
* LICENSE file in the root directory of this source tree.
*/

#include "pyvelox.h" // @manual

namespace facebook::pyvelox {
using namespace velox;
namespace py = pybind11;

std::string serializeType(const std::shared_ptr<const velox::Type>& type) {
 const auto& obj = type->serialize();
 return folly::json::serialize(obj, velox::getSerializationOptions());
}

#ifdef CREATE_PYVELOX_MODULE
PYBIND11_MODULE(_pyvelox, m) {
 m.doc() = R"pbdoc(
        PyVelox native code module
        -----------------------
       )pbdoc";

 addVeloxBindings(m);

 m.attr("__version__") = "dev";
}
#endif
} // namespace facebook::pyvelox