/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include "SubstraitFunctionCollector.h"
#include "velox/substrait/SubstraitFunction.h"

namespace facebook::velox::substrait {

const void SubstraitFunctionCollector::addFunctionToPlan(
    ::substrait::Plan& substraitPlan) const {
  // TODO    iterate reverseMap and establish the extension function
  auto extensionFunction =
      substraitPlan.add_extensions()->mutable_extension_function();

  extensionFunction->set_extension_uri_reference(0);
  extensionFunction->set_function_anchor(0);
  extensionFunction->set_name("add:opt_i32_i32");

  extensionFunction =
      substraitPlan.add_extensions()->mutable_extension_function();
  extensionFunction->set_extension_uri_reference(0);
  extensionFunction->set_function_anchor(1);
  extensionFunction->set_name("multiply:opt_i32_i32");

  extensionFunction =
      substraitPlan.add_extensions()->mutable_extension_function();
  extensionFunction->set_extension_uri_reference(1);
  extensionFunction->set_function_anchor(2);
  extensionFunction->set_name("lt:i32_i32");

  extensionFunction =
      substraitPlan.add_extensions()->mutable_extension_function();
  extensionFunction->set_extension_uri_reference(0);
  extensionFunction->set_function_anchor(3);
  extensionFunction->set_name("divide:i32_i32");

  extensionFunction =
      substraitPlan.add_extensions()->mutable_extension_function();
  extensionFunction->set_extension_uri_reference(0);
  extensionFunction->set_function_anchor(4);
  extensionFunction->set_name("count:opt_i32");

  extensionFunction =
      substraitPlan.add_extensions()->mutable_extension_function();
  extensionFunction->set_extension_uri_reference(0);
  extensionFunction->set_function_anchor(5);
  extensionFunction->set_name("sum:opt_i32");

  extensionFunction =
      substraitPlan.add_extensions()->mutable_extension_function();
  extensionFunction->set_extension_uri_reference(0);
  extensionFunction->set_function_anchor(6);
  extensionFunction->set_name("modulus:i32_i32");

  extensionFunction =
      substraitPlan.add_extensions()->mutable_extension_function();
  extensionFunction->set_extension_uri_reference(0);
  extensionFunction->set_function_anchor(7);
  extensionFunction->set_name("equal:i64_i64");
}

const int SubstraitFunctionCollector::getFunctionReference(
    const SubstraitFunctionPtr& callTypedExpr) {
  return 0;
}

} // namespace facebook::velox::substrait