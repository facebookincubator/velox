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

#include <functional>
#include <string>
#include <string_view>

#include <folly/FBString.h>
#include <folly/Format.h>
#include <folly/Range.h>
#include <folly/dynamic.h>

#include <fmt/format.h>

#include "velox/common/base/BitUtil.h"
#include "velox/common/base/Exceptions.h"

namespace facebook::velox {

/// C++ container to store a variable length string (or binary type) optimized
/// for use in Vectors.
///
/// StringView provides a lightweight, non-owning view of string data with
/// semantics similar to std::string or std::string_view. It implements the
/// following space optimization:
///
/// - Strings <= 12 bytes: Fully inlined, no heap allocation or pointer
///     indirection.
/// - Strings > 12 bytes: Stores pointer + caches first 4 bytes as prefix.
///
/// The prefix caching enables early comparison failures and reduces CPU cache
/// pressure when working with large sets of strings.
///
/// Memory Layout (16 bytes total):
/// - 4 bytes: size.
/// - 4 bytes: prefix (first 4 chars, or part of inline data).
/// - 8 bytes: either inline data continuation OR pointer to external data.
///
/// Key Characteristics:
/// - Non-owning: Does not manage lifetime of referenced data.
/// - Immutable: Provides const access to underlying data.
/// - Efficient comparisons: Uses prefix and size for fast inequality checks.
/// - Zero-overhead for small strings: Fully inlined with no allocations.
///
/// Conversions:
/// - Implicit: from char*, std::string_view (const& only).
/// - Explicit: from std::string, folly::StringPiece.
///
/// Safety Notes:
/// - StringView does NOT own the data it references.
/// - Callers must ensure referenced data outlives the StringView.
/// - Conversion to views from temporaries is explicitly deleted to prevent
///   dangling references to inlined data.
struct StringView {
 public:
  using value_type = char;

  static constexpr size_t kPrefixSize = 4 * sizeof(char);
  static constexpr size_t kInlineSize = 12;

  StringView() {
    static_assert(sizeof(StringView) == 16);
    memset(this, 0, sizeof(StringView));
  }

  StringView(const char* data, int32_t len) : size_(len) {
    VELOX_CHECK_GE(len, 0);
    VELOX_DCHECK(data || len == 0);
    if (isInline()) {
      // Zero the inline part.
      // this makes sure that inline strings can be compared for equality with 2
      // int64 compares.
      memset(prefix_, 0, kPrefixSize);
      if (size_ == 0) {
        return;
      }
      // small string: inlined. Zero the last 8 bytes first to allow for whole
      // word comparison.
      value_.data = nullptr;
#ifdef __GNUC__
      // Use asm volatile to prevent GCC's interprocedural analysis from
      // incorrectly concluding that size could exceed kInlineSize. This is a
      // workaround for a false positive -Wstringop-overflow warning when GCC
      // cannot prove the size constraint through deep template inlining.
      auto sz = static_cast<size_t>(size_);
      asm volatile("" : "+r"(sz));
      memcpy(prefix_, data, sz);
#else
      memcpy(prefix_, data, size_);
#endif
    } else {
      // large string: store pointer
      memcpy(prefix_, data, kPrefixSize);
      value_.data = data;
    }
  }

  /// Enabling StringView to be implicitly constructed from char* and
  /// string literals, in order to allow for a more flexible API and optional
  /// interoperability. E.g:
  ///
  /// >  StringView sv = "literal";
  /// >  std::optional<StringView> osv = "literal";
  ///
  /* implicit */ StringView(const char* FOLLY_NONNULL data)
      : StringView(data, strlen(data)) {}
  /* implicit */ StringView(std::nullptr_t) = delete;

  explicit StringView(const folly::fbstring& value)
      : StringView(value.data(), value.size()) {}
  explicit StringView(folly::fbstring&& value) = delete;

  explicit StringView(const std::string& value)
      : StringView(value.data(), value.size()) {}
  explicit StringView(std::string&& value) = delete;

  explicit StringView(std::string_view value)
      : StringView(value.data(), value.size()) {}

  bool isInline() const {
    return isInline(size_);
  }

  FOLLY_ALWAYS_INLINE static constexpr bool isInline(uint32_t size) {
    return size <= kInlineSize;
  }

  /// Convenience method to create an inline StringView. The API client is
  /// reponsible for providing a string that is small enough to fit inline
  /// (i.e <= kInlineSize).
  static StringView makeInline(std::string_view input) {
    VELOX_DCHECK(
        isInline(input.size()),
        "StringView::makeInline() requires an input string that fits "
        "inline (got string size of {}).",
        input.size());
    return StringView{input};
  }

  const char* data() const&& = delete;
  const char* data() const& {
    return isInline() ? prefix_ : value_.data;
  }

  size_t size() const {
    return size_;
  }

  size_t capacity() const {
    return size_;
  }

  friend std::ostream& operator<<(
      std::ostream& os,
      const StringView& stringView) {
    os.write(stringView.data(), stringView.size());
    return os;
  }

  bool operator==(const StringView& other) const {
    // Compare lengths and first 4 characters.
    if (sizeAndPrefixAsInt64() != other.sizeAndPrefixAsInt64()) {
      return false;
    }
    if (isInline()) {
      // The inline part is zeroed at construction, so we can compare
      // a word at a time if data extends past 'prefix_'.
      return size_ <= kPrefixSize || inlinedAsInt64() == other.inlinedAsInt64();
    }
    // Sizes are equal and this is not inline, therefore both are out
    // of line and have kPrefixSize first in common.
    return memcmp(
               value_.data + kPrefixSize,
               other.value_.data + kPrefixSize,
               size_ - kPrefixSize) == 0;
  }

  /// Returns 0, if this == other
  ///       < 0, if this < other
  ///       > 0, if this > other
  int32_t compare(const StringView& other) const {
    if (prefixAsInt() != other.prefixAsInt()) {
      // The result is decided on prefix. The shorter will be less because the
      // prefix is padded with zeros.
      return memcmp(prefix_, other.prefix_, kPrefixSize);
    }
    int32_t size = std::min(size_, other.size_) - kPrefixSize;
    if (size <= 0) {
      // One ends within the prefix.
      return size_ - other.size_;
    }
    if (isInline() && other.isInline()) {
      int32_t result = memcmp(value_.inlined, other.value_.inlined, size);
      return (result != 0) ? result : size_ - other.size_;
    }
    int32_t result =
        memcmp(data() + kPrefixSize, other.data() + kPrefixSize, size);
    return (result != 0) ? result : size_ - other.size_;
  }

  auto operator<=>(const StringView& other) const {
    const auto cmp = compare(other);
    return cmp < 0 ? std::strong_ordering::less
        : cmp > 0  ? std::strong_ordering::greater
                   : std::strong_ordering::equal;
  }

  /// Conversion to folly::StringPiece can only be done explicitly. For example,
  /// this is ok:
  ///
  /// > StringView sv();
  /// > folly::StringPiece sp(sv);
  ///
  /// but no:
  ///
  /// > StringView sv();
  /// > folly::StringPiece sp = sv;
  ///
  /// Note that we also explicitly disable conversion from a temporary (rvalue)
  /// because this could have resulted in a folly::StringPiece wrapping around a
  /// temporary buffer, if the StringView was inlined:
  ///
  /// > folly::StringPiece sp(StringView()); // unsafe, won't compile.
  ///
  explicit operator folly::StringPiece() const&& = delete;
  explicit operator folly::StringPiece() const& {
    return folly::StringPiece(data(), size());
  }

  /// Similarly, conversion to std::string can only be done explicitly, since
  /// this may result in an allocation and string copy.
  ///
  /// Note that in this case it is ok to enable conversion from a temporary
  /// (rvalue) StringView since its contents will be copied into the
  /// std::string.
  ///
  /// > std::string str(StringView()); // ok
  explicit operator std::string() const {
    return std::string(data(), size());
  }

  /// Conversion to std::string_view are allowed to be implicit, as long as they
  /// are not from a temporary (rvalue) StringView, for the same reasons
  /// described above.
  /* implicit */ operator std::string_view() const&& = delete;
  /* implicit */ operator std::string_view() const& {
    return std::string_view(data(), size());
  }

  /// TODO: Should make this explicit-only.
  ///
  /// Note that in this case it is ok to enable conversion from a temporary
  /// (rvalue) StringView since its contents will be copied into the
  /// folly::dynamic.
  ///
  /// > folly::dynamic str(StringView()); // ok
  /* implicit */ operator folly::dynamic() const {
    return folly::dynamic(folly::StringPiece(data(), size()));
  }

  const char* begin() const&& = delete;
  const char* begin() const& {
    return data();
  }

  const char* end() const&& = delete;
  const char* end() const& {
    return data() + size();
  }

  bool empty() const {
    return size() == 0;
  }

  /// Convenience common std::string conversion aliases.
  std::string str() const {
    return std::string(*this);
  }

  std::string getString() const {
    return std::string(*this);
  }

  std::string materialize() const {
    return std::string(*this);
  }

  /// Searches for 'key == strings[i]' for i >= 0 < numStrings.
  ///
  /// If 'indices' is given, searches for 'key == strings[indices[i]]'. Returns
  /// the first i for which the strings match or -1 if no match is found. Uses
  /// SIMD to accelerate the search.
  ///
  /// Accesses StringView bodies in 32 byte vectors, thus expects up to 31 bytes
  /// of addressable padding after out of line strings. This is the case for
  /// velox Buffers.
  static int32_t linearSearch(
      StringView key,
      const StringView* strings,
      const int32_t* indices,
      int32_t numStrings);

 private:
  int64_t sizeAndPrefixAsInt64() const {
    return reinterpret_cast<const int64_t*>(this)[0];
  }

  int64_t inlinedAsInt64() const {
    return reinterpret_cast<const int64_t*>(this)[1];
  }

  int32_t prefixAsInt() const {
    return *reinterpret_cast<const int32_t*>(&prefix_);
  }

  // We rely on all members being laid out top to bottom . C++
  // guarantees this.
  uint32_t size_;
  char prefix_[4];
  union {
    char inlined[8];
    const char* data;
  } value_;
};

// This creates a user-defined literal for StringView. You can use it as:
//
//   auto myStringView = "my string"_sv;
//   auto vec = {"str1"_sv, "str2"_sv};
inline StringView operator""_sv(const char* str, size_t len) {
  return StringView(str, len);
}

// Specializations needed for string conversion in folly/Conv.h.
template <class TString>
inline void toAppend(const StringView& value, TString* result) {
  result->append(value);
}

inline size_t estimateSpaceNeeded(const StringView& value) {
  return value.size();
}

} // namespace facebook::velox

namespace std {
template <>
struct hash<::facebook::velox::StringView> {
  size_t operator()(const ::facebook::velox::StringView view) const {
    return facebook::velox::bits::hashBytes(1, view.data(), view.size());
  }
};
} // namespace std

namespace folly {

template <>
struct hasher<::facebook::velox::StringView> {
  size_t operator()(const ::facebook::velox::StringView view) const {
    return facebook::velox::bits::hashBytes(1, view.data(), view.size());
  }
};

} // namespace folly

namespace fmt {
template <>
struct formatter<facebook::velox::StringView> : private formatter<string_view> {
  using formatter<string_view>::parse;

  template <typename Context>
  typename Context::iterator format(facebook::velox::StringView s, Context& ctx)
      const {
    return formatter<string_view>::format(string_view{s.data(), s.size()}, ctx);
  }
};
} // namespace fmt
