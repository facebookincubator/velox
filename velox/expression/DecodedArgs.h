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
#include <iterator>
#include "velox/expression/EvalCtx.h"

namespace facebook::velox::exec {

/// A helper class to decode VectorFunction arguments.
/// Example:
///    DecodedArgs decodedArgs(rows, args, context);
///    auto base = decodedArgs.at(0);
///    auto exp = decodedArgs.at(1);
///
///    rows.applyToSelected([&](int row) {
///      rawResults[row] =
///        std::pow(base->valueAt<double>(row), exp->valueAt<double>(row));
///    });
///
/// Also iterable over the decoded arguments:
///    for (auto* decoded : decodedArgs) {
///      ...
///    }
///
class DecodedArgs {
 public:
  /// Yields 'DecodedVector*', matching at(), by unwrapping the holders in
  /// place.
  class Iterator {
   public:
    using iterator_category = std::input_iterator_tag;
    using value_type = DecodedVector*;
    using difference_type = std::ptrdiff_t;
    using pointer = DecodedVector**;
    using reference = DecodedVector*;

    Iterator() = default;

    explicit Iterator(exec::LocalDecodedVector* holder) : holder_{holder} {}

    DecodedVector* operator*() const {
      return holder_->get();
    }

    Iterator& operator++() {
      ++holder_;
      return *this;
    }

    Iterator operator++(int) {
      auto previous = *this;
      ++holder_;
      return previous;
    }

    bool operator==(const Iterator& other) const = default;

   private:
    exec::LocalDecodedVector* holder_{nullptr};
  };

  DecodedArgs(
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      exec::EvalCtx& context) {
    holders_.reserve(args.size());
    for (auto& arg : args) {
      holders_.emplace_back(context, *arg, rows);
    }
  }

  DecodedVector* at(int i) const {
    return const_cast<exec::LocalDecodedVector*>(&holders_[i])->get();
  }

  size_t size() const {
    return holders_.size();
  }

  Iterator begin() const {
    return Iterator{holders_.data()};
  }

  Iterator end() const {
    return Iterator{holders_.data() + holders_.size()};
  }

 private:
  std::vector<exec::LocalDecodedVector> holders_;
};
} // namespace facebook::velox::exec
