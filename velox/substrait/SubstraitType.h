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

#pragma once

#include "SubstraitFunctionLookup.h"
#include "iostream"
#include "velox/substrait/TypeUtils.h"
#include "velox/substrait/proto/substrait/algebra.pb.h"

namespace facebook::velox::substrait {

class SubstraitType {
 private:
  static const std::string BOOL;
  static const std::string I8;
  static const std::string I16;
  static const std::string I32;
  static const std::string I64;
  static const std::string FP32;
  static const std::string FP64;
  static const std::string STRING;
  static const std::string BINARY;
  static const std::string TIMESTAMP;
  static const std::string TIMESTAMP_TZ;
  static const std::string DATE;
  static const std::string TIME;
  static const std::string INTERVAL_YEAR;
  static const std::string INTERVAL_DAY;
  static const std::string UUID;
  static const std::string FIXED_CHAR;
  static const std::string VARCHAR;
  static const std::string FIXED_BINARY;
  static const std::string DECIMAL;
  static const std::string STRUCT;
  static const std::string LIST;
  static const std::string MAP;
  static const std::string ANY;
  static const std::string USER_DEFINED;
  static const std::unordered_map<std::string, std::string> ARGUMENT_TO_SIGNATURE ;
  static const std::unordered_map<::substrait::Type::KindCase, std::string>
      TYPE_KIND_TO_SIGNATURE;

 public:
  /// Return the Substrait signature according to the substrait type
  static const std::string& typeToSignature(const ::substrait::Type& type) ;

  /// Return the Substrait signature according to the  substrait argument
  /// type
  static const std::string& argumentToSignature(const std::string& argumentType) ;

  /// Return true if argument is start with "any" or "ANY"
  static bool isWildcard(const std::string& argumentType);

  static std::string signature(
      const std::string& name,
      const std::vector<::substrait::Type>& types);
};

} // namespace facebook::velox::substrait
