/*
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
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

#include "velox/exec/tests/utils/FeatureGen.h"
#include "velox/vector/tests/utils/VectorMaker.h"

namespace facebook::velox::exec::test {

using namespace facebook::velox;

RowTypePtr makeRowType(
    const std::vector<RowVectorPtr>& vectors,
    int32_t column) {
  folly::F14FastSet<int32_t> keys;
  TypePtr valueType;
  for (auto& row : vectors) {
    auto map = row->childAt(column)->as<MapVector>();
    auto keyVector = map->mapKeys()->as<FlatVector<int32_t>>();
    if (!valueType) {
      valueType = map->type()->childAt(1);
    }
    for (auto i = 0; i < keyVector->size(); ++i) {
      keys.insert(keyVector->valueAt(i));
    }
  }
  std::vector<std::string> names;
  std::vector<TypePtr> types;
  for (auto key : keys) {
    names.push_back(fmt::format("{}", key));
    types.push_back(valueType);
  }
  return ROW(std::move(names), std::move(types));
}

BufferPtr evenOffsets(int32_t numRows, int32_t step, memory::MemoryPool* pool) {
  auto buffer = AlignedBuffer::allocate<int32_t>(numRows, pool);
  for (auto i = 0; i < numRows; ++i) {
    buffer->asMutable<int32_t>()[i] = i * step;
  }
  return buffer;
}

BufferPtr evenSizes(int32_t numRows, int32_t step, memory::MemoryPool* pool) {
  auto buffer = AlignedBuffer::allocate<int32_t>(numRows, pool);
  for (auto i = 0; i < numRows; ++i) {
    buffer->asMutable<int32_t>()[i] = step;
  }
  return buffer;
}

std::vector<RowVectorPtr> makeFeatures(
    int32_t numBatches,
    int32_t batchSize,
    FeatureOptions& opts,
    memory::MemoryPool* pool) {
  std::vector<RowVectorPtr> result;
  velox::test::VectorMaker vectorMaker(pool);

  for (auto batchIdx = 0; batchIdx < numBatches; ++batchIdx) {
    auto uids = vectorMaker.flatVector<int64_t>(
        batchSize, [&](auto row) { return row + batchIdx * batchSize; });
    auto tss = vectorMaker.flatVector<int64_t>(batchSize, [&](auto row) {
      return 100 * (row + batchIdx * batchSize);
    });
    auto floatKeys = vectorMaker.flatVector<int32_t>(
        batchSize * opts.numFloat,
        [&](int32_t row) { return (row % opts.numFloat) * 100 + 10000; });
    auto floats = vectorMaker.flatVector<float>(
        batchSize * opts.numFloat,
        [&](int32_t row) { return (row % 123) / 100.0; });

    auto floatFeatures = std::make_shared<MapVector>(
        pool,
        MAP(INTEGER(), REAL()),
        nullptr,
        batchSize,
        evenOffsets(batchSize, opts.numFloat, pool),
        evenSizes(batchSize, opts.numFloat, pool),
        floatKeys,
        floats);

    std::vector<int32_t> idListSize(opts.numIdList);
    for (auto i = 0; i < opts.numIdList; ++i) {
      idListSize[i] = opts.idListMinCard +
          i * ((opts.idListMaxCard - opts.idListMinCard) / opts.numIdList);
    }

    auto idLists = vectorMaker.arrayVector<int64_t>(
        batchSize * opts.numIdList,
        [&](auto row) { return idListSize[row % idListSize.size()]; },
        [&](auto row) { return row * 100 + 1; });
    auto idListKeys = vectorMaker.flatVector<int32_t>(
        batchSize * opts.numIdList,
        [&](auto row) { return (row % opts.numIdList) * 200 + 200000; });
    auto idListFeatures = std::make_shared<MapVector>(
        pool,
        MAP(INTEGER(), ARRAY(BIGINT())),
        nullptr,
        batchSize,
        evenOffsets(batchSize, opts.numIdList, pool),
        evenSizes(batchSize, opts.numIdList, pool),
        idListKeys,
        idLists);
    auto scoreKeys = vectorMaker.flatVector<int32_t>(
        batchSize * opts.numIdScoreList,
        [&](auto row) { return (row % opts.numIdScoreList) * 200 + 200000; });

    auto scores = vectorMaker.mapVector<int64_t, float>(
        batchSize * opts.numIdScoreList,
        [&](auto row) { return idListSize[row % opts.numIdScoreList]; },
        [&](int32_t row, int32_t idx) {
          auto nthArray = (row / opts.numIdScoreList) * opts.numIdList;
          return idLists->elements()->as<SimpleVector<int64_t>>()->valueAt(
              idLists->offsetAt(nthArray) + idx);
        },
        [&](int32_t row, int32_t idx) { return 1.2 * row / idx; });
    auto scoreListFeatures = std::make_shared<MapVector>(
        pool,
        MAP(INTEGER(), MAP(BIGINT(), REAL())),
        nullptr,
        batchSize,
        evenOffsets(batchSize, opts.numIdScoreList, pool),
        evenSizes(batchSize, opts.numIdScoreList, pool),
        scoreKeys,
        scores);
    auto row = vectorMaker.rowVector(
        {"uid",
         "ts",
         "float_features",
         "id_list_features",
         "id_score_list_features"},
        {uids, tss, floatFeatures, idListFeatures, scoreListFeatures});
    result.push_back(std::move(row));
  }
  opts.floatStruct = makeRowType(result, 2);
  opts.idListStruct = makeRowType(result, 3);
  opts.idScoreListStruct = makeRowType(result, 4);
  return result;
}

core::TypedExprPtr floatFeatures() {
  return std::make_shared<core::FieldAccessTypedExpr>(
      MAP(INTEGER(), REAL()), "float_features");
}

core::TypedExprPtr intLiteral(int32_t i) {
  return std::make_shared<core::ConstantTypedExpr>(INTEGER(), variant(i));
}

core::TypedExprPtr floatLiteral(float i) {
  return std::make_shared<core::ConstantTypedExpr>(REAL(), variant(i));
}

core::TypedExprPtr rand() {
  core::TypedExprPtr r = std::make_shared<core::CallTypedExpr>(
      INTEGER(), std::vector<core::TypedExprPtr>{intLiteral(5)}, "rand");
  return std::make_shared<core::CastTypedExpr>(REAL(), r, false);
}

core::TypedExprPtr plus(core::TypedExprPtr x, core::TypedExprPtr y) {
  return std::make_shared<core::CallTypedExpr>(
      REAL(), std::vector<core::TypedExprPtr>{x, y}, "plus");
}

core::TypedExprPtr floatFeature(const FeatureOptions& opts) {
  int32_t id = atoi(
      opts.floatStruct
          ->nameOf(folly::Random::rand32(opts.rng) % opts.floatStruct->size())
          .c_str());
  std::vector<core::TypedExprPtr> args{floatFeatures(), intLiteral(id)};

  return std::make_shared<core::CallTypedExpr>(
      REAL(), std::move(args), "subscript");
}

core::TypedExprPtr plusOne(core::TypedExprPtr expr) {
  std::vector<core::TypedExprPtr> args{expr, floatLiteral(1)};
  return std::make_shared<core::CallTypedExpr>(REAL(), std::move(args), "plus");
}
core::TypedExprPtr uid() {
  return std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "uid");
}

core::TypedExprPtr bigintMod(core::TypedExprPtr x, int64_t y) {
  core::TypedExprPtr lit =
      std::make_shared<core::ConstantTypedExpr>(BIGINT(), variant(y));
  return std::make_shared<core::CallTypedExpr>(
      BIGINT(), std::vector<core::TypedExprPtr>{x, lit}, "mod");
}

core::TypedExprPtr bigintEq(core::TypedExprPtr x, int64_t y) {
  core::TypedExprPtr lit =
      std::make_shared<core::ConstantTypedExpr>(BIGINT(), variant(y));
  return std::make_shared<core::CallTypedExpr>(
      BOOLEAN(), std::vector<core::TypedExprPtr>{x, lit}, "eq");
}

core::TypedExprPtr uidCond(core::TypedExprPtr f) {
  auto cond = bigintEq(bigintMod(uid(), 10), 0);
  return std::make_shared<core::CallTypedExpr>(
      REAL(), std::vector<core::TypedExprPtr>{cond, f, plusOne(f)}, "if");
}

core::TypedExprPtr makeFloatExpr(const FeatureOptions& opts) {
  auto f = floatFeature(opts);
  if (opts.coinToss(opts.plusOnePct)) {
    f = plusOne(f);
  }
  if (opts.coinToss(opts.randomPct)) {
    f = plus(f, rand());
  }
  if (opts.coinToss(opts.multiColumnPct)) {
    auto g = floatFeature(opts);
    if (opts.coinToss(opts.plusOnePct)) {
      g = plusOne(g);
    }
    f = plus(f, g);
  }
  if (opts.coinToss(opts.uidPct)) {
    f = uidCond(f);
  }
  return f;
}

void makeExprs(
    const FeatureOptions& opts,
    std::vector<std::string>& names,
    std::vector<core::TypedExprPtr>& exprs) {
  names = {"uid"};
  exprs = {uid()};
  auto numFloatExprs = (opts.floatStruct->size() * opts.floatExprsPct) / 100.0;
  std::vector<core::TypedExprPtr> floatExprs;
  std::vector<TypePtr> floatTypes;
  for (auto cnt = 0; cnt < numFloatExprs; ++cnt) {
    floatExprs.push_back(makeFloatExpr(opts));
    floatTypes.push_back(REAL());
  }
  if (!floatExprs.empty()) {
    names.push_back("floats");
    exprs.push_back(std::make_shared<core::CallTypedExpr>(
        ROW(std::move(floatTypes)), std::move(floatExprs), "row_constructor"));
  }
}



} // namespace facebook::axiom::optimizer::test
