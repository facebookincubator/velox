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
#include "velox/expression/ConstantExpr.h"

namespace facebook::velox::exec {

void ConstantExpr::evalSpecialForm(
    const SelectivityVector& rows,
    EvalCtx& context,
    VectorPtr& result) {
  if (needToSetIsAscii_) {
    auto* vector =
        sharedConstantValue_->asUnchecked<SimpleVector<StringView>>();
    LocalSingleRow singleRow(context, 0);
    bool isAscii = vector->computeAndSetIsAscii(*singleRow);
    vector->setAllIsAscii(isAscii);
    needToSetIsAscii_ = false;
  }

  if (sharedConstantValue_.unique()) {
    sharedConstantValue_->resize(rows.end());
  } else {
    // By reassigning sharedConstantValue_ we increase the chances that it will
    // be unique the next time this expression is evaluated.
    sharedConstantValue_ =
        BaseVector::wrapInConstant(rows.end(), 0, sharedConstantValue_);
  }

  context.moveOrCopyResult(sharedConstantValue_, rows, result);
}

void ConstantExpr::evalSpecialFormSimplified(
    const SelectivityVector& rows,
    EvalCtx& context,
    VectorPtr& result) {
  ExceptionContextSetter exceptionContext(
      {[](VeloxException::Type /*exceptionType*/, auto* expr) {
         return static_cast<Expr*>(expr)->toString();
       },
       this});

  // Simplified path should never ask us to write to a vector that was already
  // pre-allocated.
  VELOX_CHECK_NULL(result);

  result = BaseVector::wrapInConstant(rows.end(), 0, sharedConstantValue_);
}

std::string ConstantExpr::toString(bool /*recursive*/) const {
  return fmt::format(
      "{}:{}", sharedConstantValue_->toString(0), type()->toString());
}

namespace {
void appendSqlLiteral(
    const BaseVector& vector,
    vector_size_t row,
    std::ostream& out);

void appendSqlLiteralList(
    const BaseVector& vector,
    vector_size_t offset,
    vector_size_t size,
    std::ostream& out) {
  for (auto i = offset; i < offset + size; ++i) {
    if (i > offset) {
      out << ", ";
    }
    appendSqlLiteral(vector, i, out);
  }
}

void appendSqlString(const std::string& value, std::ostream& out) {
  // Escape single quotes: ' -> ''.
  static constexpr char kSingleQuote = '\'';

  out << kSingleQuote;
  auto prevPos = 0;
  auto pos = value.find(kSingleQuote);
  while (pos != std::string::npos) {
    out << value.substr(prevPos, pos - prevPos) << kSingleQuote;
    prevPos = pos;
    pos = value.find(kSingleQuote, prevPos + 1);
  }
  out << value.substr(prevPos, pos) << kSingleQuote;
  return;
}

std::string toSqlType(const TypePtr& type) {
  switch (type->kind()) {
    case TypeKind::ARRAY:
      return fmt::format("{}[]", toSqlType(type->childAt(0)));
    case TypeKind::MAP:
      return fmt::format(
          "MAP({}, {})",
          toSqlType(type->childAt(0)),
          toSqlType(type->childAt(1)));
    case TypeKind::ROW: {
      std::ostringstream out;
      out << "STRUCT(";
      for (auto i = 0; i < type->size(); ++i) {
        if (i > 0) {
          out << ", ";
        }
        out << type->asRow().nameOf(i) << " " << type->childAt(i)->toString();
      }
      out << ")";
      return out.str();
    }
    default:
      return type->toString();
  }
}

void appendSqlLiteral(
    const BaseVector& vector,
    vector_size_t row,
    std::ostream& out) {
  if (vector.isNullAt(row)) {
    if (vector.type()->containsUnknown()) {
      out << "NULL";
    } else {
      out << "NULL::" << toSqlType(vector.type());
    }
    return;
  }

  switch (vector.typeKind()) {
    case TypeKind::BOOLEAN: {
      auto value = vector.as<SimpleVector<bool>>()->valueAt(row);
      out << (value ? "TRUE" : "FALSE");
      break;
    }
    case TypeKind::TINYINT:
    case TypeKind::SMALLINT:
    case TypeKind::INTEGER:
    case TypeKind::BIGINT:
    case TypeKind::DATE:
    case TypeKind::TIMESTAMP:
    case TypeKind::REAL:
    case TypeKind::DOUBLE:
      out << "'" << vector.wrappedVector()->toString(vector.wrappedIndex(row))
          << "'::" << vector.type()->toString();
      break;
    case TypeKind::VARCHAR:
      appendSqlString(
          vector.wrappedVector()->toString(vector.wrappedIndex(row)), out);
      break;
    case TypeKind::ARRAY: {
      out << "ARRAY[";
      auto arrayVector = vector.wrappedVector()->as<ArrayVector>();
      auto arrayRow = vector.wrappedIndex(row);
      auto offset = arrayVector->offsetAt(arrayRow);
      auto size = arrayVector->sizeAt(arrayRow);
      appendSqlLiteralList(*arrayVector->elements(), offset, size, out);
      out << "]";
      break;
    }
    case TypeKind::MAP: {
      out << "map(ARRAY[";
      auto mapVector = vector.wrappedVector()->as<MapVector>();
      auto mapRow = vector.wrappedIndex(row);
      auto offset = mapVector->offsetAt(mapRow);
      auto size = mapVector->sizeAt(mapRow);
      appendSqlLiteralList(*mapVector->mapKeys(), offset, size, out);
      out << "], ARRAY[";
      appendSqlLiteralList(*mapVector->mapValues(), offset, size, out);
      out << "])";
      break;
    }
    case TypeKind::ROW: {
      out << "row_constructor(";
      auto rowVector = vector.wrappedVector()->as<RowVector>();
      auto baseRow = vector.wrappedIndex(row);
      for (auto i = 0; i < rowVector->childrenSize(); ++i) {
        if (i > 0) {
          out << ", ";
        }
        appendSqlLiteral(*rowVector->childAt(i), baseRow, out);
      }
      out << ")";
      break;
    }
    default:
      // TODO: update ExprStatsTest.exceptionPreparingStatsForListener once
      // support for VARBINARY is added.
      VELOX_UNSUPPORTED(
          "Type not supported yet: {}", vector.type()->toString());
  }
}
} // namespace

std::string ConstantExpr::toSql(
    std::vector<VectorPtr>* complexConstants) const {
  VELOX_CHECK_NOT_NULL(sharedConstantValue_);
  std::ostringstream out;
  if (complexConstants && !sharedConstantValue_->type()->isPrimitiveType()) {
    int idx = complexConstants->size();
    out << "__complex_constant(c" << idx << ")";
    complexConstants->push_back(sharedConstantValue_);
  } else {
    appendSqlLiteral(*sharedConstantValue_, 0, out);
  }
  return out.str();
}
} // namespace facebook::velox::exec
