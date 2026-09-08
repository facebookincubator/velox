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

#include <folly/container/F14Map.h>

#include "velox/type/Type.h"

namespace facebook::velox::core {

enum class ExprKind : int32_t {
  kInput = 0,
  kFieldAccess = 1,
  kDereference = 2,
  kCall = 3,
  kCast = 5,
  kConstant = 6,
  kConcat = 7,
  kLambda = 8,
  kNullIf = 9,
};

VELOX_DECLARE_ENUM_NAME(ExprKind);

class ITypedExpr;
class ITypedExprVisitor;
class ITypedExprVisitorContext;

using TypedExprPtr = std::shared_ptr<const ITypedExpr>;

struct ITypedExprHasher {
  size_t operator()(const ITypedExpr* expr) const;
};

struct ITypedExprComparer {
  bool operator()(const ITypedExpr* lhs, const ITypedExpr* rhs) const;
};

/// State threaded through sharing-aware expression printing. Expressions form a
/// DAG: a shared subexpression is one node reached from several parents.
/// Printing such a DAG as a tree takes time and space exponential in its depth.
/// With this context each shared subexpression is written out once, followed by
/// an "as #N" label, and every later occurrence prints just "#N".
struct ExprToStringContext {
  /// Number of times each node is reached in the DAG rooted at the expression
  /// being printed. A node reached more than once is shared.
  folly::F14FastMap<const ITypedExpr*, uint32_t> refCounts;

  /// Labels assigned to shared nodes that have already been printed in full.
  folly::F14FastMap<const ITypedExpr*, uint32_t> labels;

  uint32_t nextLabel{1};
};

/// Strongly-typed expression, e.g. literal, function call, etc.
class ITypedExpr : public ISerializable {
 public:
  ITypedExpr(ExprKind kind, TypePtr type)
      : kind_{kind}, type_{std::move(type)}, inputs_{} {}

  ITypedExpr(ExprKind kind, TypePtr type, std::vector<TypedExprPtr> inputs)
      : kind_{kind}, type_{std::move(type)}, inputs_{std::move(inputs)} {}

  virtual ~ITypedExpr() = default;

  ExprKind kind() const {
    return kind_;
  }

  const TypePtr& type() const {
    return type_;
  }

  const std::vector<TypedExprPtr>& inputs() const {
    return inputs_;
  }

  bool isInputKind() const {
    return kind_ == ExprKind::kInput;
  }

  bool isFieldAccessKind() const {
    return kind_ == ExprKind::kFieldAccess;
  }

  bool isDereferenceKind() const {
    return kind_ == ExprKind::kDereference;
  }

  bool isCallKind() const {
    return kind_ == ExprKind::kCall;
  }

  bool isCastKind() const {
    return kind_ == ExprKind::kCast;
  }

  bool isConstantKind() const {
    return kind_ == ExprKind::kConstant;
  }

  bool isConcatKind() const {
    return kind_ == ExprKind::kConcat;
  }

  bool isLambdaKind() const {
    return kind_ == ExprKind::kLambda;
  }

  bool isNullIfKind() const {
    return kind_ == ExprKind::kNullIf;
  }

  template <typename T>
  const T* asUnchecked() const {
    return dynamic_cast<const T*>(this);
  }

  /// Returns a copy of this expression with input fields replaced according
  /// to specified 'mapping'. Fields specified in the 'mapping' are replaced
  /// by the corresponding expression in 'mapping'.
  /// Fields not present in 'mapping' are left unmodified.
  ///
  /// Used to bind inputs to lambda functions.
  virtual TypedExprPtr rewriteInputNames(
      const std::unordered_map<std::string, TypedExprPtr>& mapping) const = 0;

  /// Part of the visitor pattern. Calls visitor.vist(*this, context) with the
  /// "right" type of the first argument.
  virtual void accept(
      const ITypedExprVisitor& visitor,
      ITypedExprVisitorContext& context) const = 0;

  virtual std::string toString() const = 0;

  /// Appends this expression to 'out', reusing 'ctx' so that a shared
  /// subexpression is printed once with an "as #N" label and every later
  /// occurrence is printed as just "#N". This prints an expression DAG in
  /// linear size, unlike toString() which expands it as a tree.
  void appendWithSharing(std::string& out, ExprToStringContext& ctx) const;

  /// Returns a hash value for this expression node only, not including inputs.
  /// Implementations must use a stable hash like folly::hasher to ensure
  /// stable hashing across processes and builds.
  virtual size_t localHash() const = 0;

  /// Returns a hash value for the entire expression tree rooted at this node.
  /// The hash is computed by combining localHash() with the type's hash and
  /// the hashes of all input expressions.
  ///
  /// STABILITY GUARANTEE: This hash is stable across different processes,
  /// builds, and machines.
  size_t hash() const {
    size_t hash = bits::hashMix(type_->hashKind(), localHash());
    for (size_t i = 0; i < inputs_.size(); ++i) {
      hash = bits::hashMix(hash, inputs_[i]->hash());
    }
    return hash;
  }

  virtual bool operator==(const ITypedExpr& other) const = 0;

  static void registerSerDe();

 protected:
  // Appends this node's own textual form to 'out', recursing into inputs via
  // appendWithSharing so that sharing is preserved. The default prints the node
  // via toString() and is used by leaves and by subclasses that do not opt into
  // sharing-aware printing.
  //
  // Invariant: a subclass that recurses into inputs must EITHER leave both this
  // and toString() at their defaults, OR override this AND make toString()
  // return toStringWithSharing() - the two go together. Overriding only
  // toString() to call toStringWithSharing() without also overriding this loops
  // forever (the default appendToString() calls toString() calls
  // toStringWithSharing() calls the default appendToString()).
  virtual void appendToString(std::string& out, ExprToStringContext& ctx) const;

  // Builds a fresh context and prints this expression sharing-aware. Recursive
  // subclasses call this from toString(); see the invariant on
  // appendToString().
  std::string toStringWithSharing() const;

  folly::dynamic serializeBase(std::string_view name) const;

  std::vector<TypedExprPtr> rewriteInputsRecursive(
      const std::unordered_map<std::string, TypedExprPtr>& mapping) const {
    std::vector<TypedExprPtr> newInputs;
    newInputs.reserve(inputs().size());
    for (const auto& input : inputs()) {
      newInputs.emplace_back(input->rewriteInputNames(mapping));
    }
    return newInputs;
  }

 private:
  const ExprKind kind_;
  const TypePtr type_;
  const std::vector<TypedExprPtr> inputs_;
};

} // namespace facebook::velox::core
