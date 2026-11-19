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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "velox/experimental/torchwave/SymShape.h"

namespace torch::wave {
namespace {

TEST(SymShapeTest, parsesExporterForms) {
  auto expr = parseSympy(
      "Mul(Integer(1000), Max(Integer(1), Symbol('u1', integer=True)))");
  ASSERT_TRUE(expr.has_value());
  EXPECT_EQ(expr->toString(), "1000*(Max(1, u1))");
  std::vector<std::string> symbols;
  expr->collectSymbols(symbols);
  EXPECT_THAT(symbols, testing::ElementsAre("u1"));
}

TEST(SymShapeTest, evaluatesOverSymbolValues) {
  auto expr = parseSympy(
      "Add(Mul(Integer(799), Symbol('u0', integer=True)), "
      "Pow(Symbol('s3', integer=True, positive=True), Integer(2)))");
  ASSERT_TRUE(expr.has_value());
  EXPECT_EQ(expr->evaluate({{"u0", 2}, {"s3", 5}}), 799 * 2 + 25);
  EXPECT_EQ(expr->evaluate({{"u0", 2}}), std::nullopt);
}

TEST(SymShapeTest, rejectsUnknownForms) {
  EXPECT_EQ(parseSympy("Piecewise((Integer(1), True))"), std::nullopt);
  EXPECT_EQ(parseSympy("Integer(3) trailing"), std::nullopt);
}

TEST(SymShapeTest, recordsShapesAndScalars) {
  SymShapes shapes;
  addSymShape(
      shapes,
      "x",
      {int64_t{1000}, std::string("Symbol('u1', integer=True)")},
      {std::string("Symbol('u1', integer=True)"), int64_t{1}});
  addSymShape(shapes, "y", {int64_t{4}}, {int64_t{1}});
  addSymScalar(shapes, "n", std::string("Symbol('u1', integer=True)"));
  addSymShape(shapes, "bad", {std::string("Unknown(1)")}, {int64_t{1}});

  ASSERT_EQ(shapes.byName.count("x"), 1);
  EXPECT_TRUE(shapes.byName.at("x").dependsOn('u'));
  EXPECT_FALSE(shapes.byName.at("y").dependsOn('u'));
  EXPECT_EQ(shapes.scalarByName.at("n").evaluate({{"u1", 7}}), 7);
  EXPECT_THAT(shapes.unparsed, testing::ElementsAre("bad"));
}

TEST(SymShapeTest, hostDimMatchesTheDimItWasTracedWith) {
  const auto host = hostDimSymbol("getitem_7", 0, 1024);
  EXPECT_TRUE(isHostDimSymbol(host.symbol));
  EXPECT_FALSE(isHostDimSymbol("d1024"));
  EXPECT_EQ(tracedValue(host), 1024);
  const SymExpr input{.kind = SymExpr::Kind::kSymbol, .symbol = "d1024"};
  EXPECT_TRUE(sameDim(host, input));
  EXPECT_TRUE(sameDim(input, host));
  EXPECT_TRUE(sameDim(host, SymExpr::constant(1024)));
  EXPECT_FALSE(sameDim(host, hostDimSymbol("getitem_8", 0, 512)));
}

} // namespace
} // namespace torch::wave
