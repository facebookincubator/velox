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

#include "velox/functions/prestosql/types/UuidType.h"
#include <boost/lexical_cast.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_io.hpp>

namespace facebook::velox {

namespace {

class UuidCastOperator : public exec::CastOperator {
 public:
  bool isSupportedFromType(const TypePtr& other) const override {
    return VARCHAR()->equivalent(*other);
  }

  bool isSupportedToType(const TypePtr& other) const override {
    return VARCHAR()->equivalent(*other);
  }

  void castTo(
      const BaseVector& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      const TypePtr& resultType,
      VectorPtr& result) const override {
    context.ensureWritable(rows, resultType, result);

    if (input.typeKind() == TypeKind::VARCHAR) {
      castFromString(input, context, rows, *result);
    } else {
      VELOX_UNSUPPORTED(
          "Cast from {} to UUID not yet supported", input.type()->toString());
    }
  }

  void castFrom(
      const BaseVector& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      const TypePtr& resultType,
      VectorPtr& result) const override {
    context.ensureWritable(rows, resultType, result);

    if (resultType->kind() == TypeKind::VARCHAR) {
      castToString(input, context, rows, *result);
    } else {
      VELOX_UNSUPPORTED(
          "Cast from UUID to {} not yet supported", resultType->toString());
    }
  }

 private:
  static void castToString(
      const BaseVector& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      BaseVector& result) {
    auto* flatResult = result.as<FlatVector<StringView>>();
    const auto* uuids = input.as<SimpleVector<int128_t>>();

    context.applyToSelectedNoThrow(rows, [&](auto row) {
      const auto uuid = uuids->valueAt(row);

      const uint8_t* uuidBytes = reinterpret_cast<const uint8_t*>(&uuid);

      // Do not use boost::lexical_cast. It is very slow.

      // 2 hex digits per each value in [0, 127] range (1 byte).
      static const char* const kHexTable =
          "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"
          "202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f"
          "404142434445464748494a4b4c4d4e4f505152535455565758595a5b5c5d5e5f"
          "606162636465666768696a6b6c6d6e6f707172737475767778797a7b7c7d7e7f"
          "808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9f"
          "a0a1a2a3a4a5a6a7a8a9aaabacadaeafb0b1b2b3b4b5b6b7b8b9babbbcbdbebf"
          "c0c1c2c3c4c5c6c7c8c9cacbcccdcecfd0d1d2d3d4d5d6d7d8d9dadbdcdddedf"
          "e0e1e2e3e4e5e6e7e8e9eaebecedeeeff0f1f2f3f4f5f6f7f8f9fafbfcfdfeff";

      exec::StringWriter<false> result(flatResult, row);
      result.resize(36);

      size_t offset = 0;
      for (auto i = 0; i < 16; ++i) {
        result.data()[offset] = kHexTable[uuidBytes[i] * 2];
        result.data()[offset + 1] = kHexTable[uuidBytes[i] * 2 + 1];

        offset += 2;
        if (i == 3 || i == 5 || i == 7 || i == 9) {
          result.data()[offset] = '-';
          offset++;
        }
      }

      result.finalize();
    });
  }

  static void castFromString(
      const BaseVector& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      BaseVector& result) {
    auto* flatResult = result.as<FlatVector<int128_t>>();
    const auto* uuidStrings = input.as<SimpleVector<StringView>>();

    context.applyToSelectedNoThrow(rows, [&](auto row) {
      const auto uuidString = uuidStrings->valueAt(row);

      auto uuid = boost::lexical_cast<boost::uuids::uuid>(uuidString);

      int128_t u;
      memcpy(&u, &uuid, 16);

      flatResult->set(row, u);
    });
  }
};

class UuidTypeFactories : public CustomTypeFactories {
 public:
  UuidTypeFactories() = default;

  TypePtr getType() const override {
    return UUID();
  }

  exec::CastOperatorPtr getCastOperator() const override {
    return std::make_shared<UuidCastOperator>();
  }
};

} // namespace

void registerUuidType() {
  registerCustomType("uuid", std::make_unique<const UuidTypeFactories>());
}

} // namespace facebook::velox
