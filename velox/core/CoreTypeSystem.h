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

#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "velox/common/base/Exceptions.h"
#include "velox/core/Metaprogramming.h"
#include "velox/functions/UDFOutputString.h"
#include "velox/type/Type.h"
#include "velox/type/Variant.h"

namespace facebook {
namespace velox {
namespace core {

// A simple & efficient container/target for user strings
struct StringWriter : public UDFOutputString {
  StringWriter() noexcept : storage_{} {
    setData(storage_.data());
  }

  /* implicit */ StringWriter(StringView /*value*/) {
    VELOX_NYI();
  }

  void setEmpty() {
    VELOX_FAIL("setEmpty is not implemented");
  }

  void setNoCopy(StringView /*value*/) {
    VELOX_FAIL("setNoCopy is not implemented");
  }

  StringWriter(const StringWriter& rh) : storage_{rh.storage_} {
    setData(storage_.data());
    setSize(rh.size());
    setCapacity(rh.capacity());
  }

  StringWriter(StringWriter&& rh) noexcept : storage_{std::move(rh.storage_)} {
    setData(storage_.data());
    setSize(rh.size());
    setCapacity(rh.capacity());
  }

  StringWriter& operator=(const StringWriter& rh) {
    storage_ = rh.storage_;
    reserve(rh.capacity());
    resize(rh.size());
    return *this;
  }

  StringWriter& operator=(StringWriter&& rh) noexcept {
    storage_ = std::move(rh.storage_);
    setData(storage_.data());
    setSize(rh.size());
    setCapacity(rh.capacity());
    return *this;
  }

  void reserve(size_t size) override {
    // Resizing the storage not StringWriter size.
    // This allow us to write directly write into storage_.data() and assuring
    // what we wrote wont be overwritten on future resize calls.
    storage_.resize(size);
    setData(storage_.data());
    setCapacity(size);
  }

  /// Not called by the UDF but should be called internally at the end of the
  /// UDF call
  void finalize() {
    storage_.resize(size());
  }

  operator StringView() const {
    return StringView(data(), size());
  }

 private:
  folly::fbstring storage_;
};

namespace detail {

template <typename T, typename = int32_t>
struct has_velox_type : std::false_type {};

template <typename T>
struct has_velox_type<T, decltype((void)T::veloxType, 0)> : std::true_type {};
} // namespace detail

template <typename T>
struct UdfToType {
  template <
      typename Tx = T,
      typename std::enable_if_t<detail::has_velox_type<Tx>::value, int32_t> = 0>
  static std::shared_ptr<const Type> veloxType() {
    return T::veloxType();
  }

  template <
      typename Tx = T,
      typename std::enable_if_t<!detail::has_velox_type<Tx>::value, int32_t> =
          0>
  static std::shared_ptr<const Type> veloxType() {
    return CppToType<T>::create();
  }
};

template <typename VAL>
class ArrayValWriter {
 public:
  using container_t = typename std::vector<std::optional<VAL>>;
  using iterator = typename container_t::iterator;
  using reference = typename container_t::reference;
  using const_iterator = typename container_t::const_iterator;
  using const_reference = typename container_t::const_reference;
  using val_type = VAL;

  static std::shared_ptr<const Type> veloxType() {
    return ARRAY(UdfToType<val_type>::veloxType());
  }

  ArrayValWriter() = default;
  ArrayValWriter(const_iterator start, const_iterator end)
      : values_(start, end) {}

  void reserve(size_t size) {
    values_.reserve(size);
  }
  iterator begin() {
    return values_.begin();
  }
  iterator end() {
    return values_.end();
  }
  void append(val_type val) {
    values_.push_back(std::optional<val_type>{std::move(val)});
  }
  void append(std::optional<val_type> val) {
    values_.push_back(std::move(val));
  }
  void appendNullable() {
    append(std::optional<val_type>{});
  }
  void clear() {
    values_.clear();
  }

  const_iterator begin() const {
    return values_.begin();
  }
  const_iterator end() const {
    return values_.end();
  }

  const_reference at(size_t index) const {
    return values_.at(index);
  }
  size_t size() const {
    return values_.size();
  }

 private:
  container_t values_;
};

template <typename VAL>
class ArrayValReader : public ArrayValWriter<VAL> {
 public:
  ArrayValReader() = default;
  explicit ArrayValReader(std::vector<VAL> vals) {
    reserve(vals.size());
    for (auto& val : vals) {
      append(std::move(val));
    }
  }

  bool mayHaveNulls() const {
    return false;
  }

  std::optional<VAL> operator[](size_t index) const {
    return {ArrayValWriter<VAL>::at(index)};
  }

  std::optional<VAL> at(size_t index) const {
    return {ArrayValWriter<VAL>::at(index)};
  }
};

template <typename... T>
struct RowWriter {
  template <std::size_t... Is>
  static std::tuple<std::optional<T>...> addOptional(
      const std::tuple<T...>& val,
      std::index_sequence<Is...>) {
    return std::tuple<std::optional<T>...>{std::get<Is>(val)...};
  }

 public:
  static std::shared_ptr<const Type> veloxType() {
    return ROW({UdfToType<T>::veloxType()...});
  }

  RowWriter() {}

  /* implicit */ RowWriter(const std::tuple<std::optional<T>...>& val)
      : values_(val) {}
  /* implicit */ RowWriter(const std::tuple<T...>& val)
      : values_(addOptional(val, std::index_sequence_for<T...>{})) {}

  template <size_t N>
  auto& at() {
    return std::get<N>(values_);
  }

  template <size_t N>
  const auto& at() const {
    return std::get<N>(values_);
  }

  void clear() {
    values_ = std::tuple<std::optional<T>...>();
  }

 private:
  std::tuple<std::optional<T>...> values_;
};

template <typename... T>
struct RowReader : public RowWriter<T...> {};

template <typename KEY, typename VAL>
struct IMapVal {
  static std::shared_ptr<const Type> veloxType() {
    return MAP(UdfToType<KEY>::veloxType(), UdfToType<VAL>::veloxType());
  }
  using container_t = typename folly::F14FastMap<KEY, std::optional<VAL>>;
  using iterator = typename container_t::iterator;
  using reference = typename container_t::reference;
  using const_iterator = typename container_t::const_iterator;
  using mapped_type = typename container_t::mapped_type;
  using key_type = KEY;
  using val_type = VAL;

  iterator begin() {
    return data_.begin();
  }
  iterator end() {
    return data_.end();
  }
  iterator find(const key_type& key) {
    return data_.find(key);
  }
  mapped_type& at(const key_type& key) {
    return data_.at(key);
  }
  val_type& append(const key_type& key) {
    auto& opt = (data_[key] = std::optional(val_type{}));
    return *opt; // todo(youknowjack): avoid presence check here
  }
  std::optional<val_type>& appendNullable(const key_type& key) {
    return data_[key] = {};
  }
  std::pair<iterator, bool> emplace(
      const key_type& key,
      std::optional<val_type> value) {
    return data_.emplace(key, std::move(value));
  }
  void clear() {
    data_.clear();
  }
  void reserve(typename container_t::size_type n) {
    data_.reserve(n);
  }

  const_iterator begin() const {
    return data_.begin();
  }
  const_iterator end() const {
    return data_.end();
  }
  const_iterator find(const key_type& key) const {
    return data_.find(key);
  }
  bool contains(const key_type& key) const {
    return data_.find(key) != end();
  }
  const mapped_type& at(const key_type& key) const {
    return data_.at(key);
  }
  size_t size() const {
    return data_.size();
  }

 private:
  container_t data_;
};

template <typename KEY, typename VAL>
class SlowMapVal : public IMapVal<KEY, VAL> {};

template <typename KEY, typename VAL>
class SlowMapWriter : public IMapVal<KEY, VAL> {
 public:
  // Allow map-like object to be assigned to a map writer. This should change
  // once we implement writer proxies.
  template <template <typename, typename> typename T>
  SlowMapWriter& operator=(const T<KEY, VAL>& rh) {
    assignFrom(rh);

    return *this;
  }

  // Allow Velox's MapView to be assigned to a map writer.  This should also
  // change once we implement writer proxies.
  template <bool nullFree, template <bool, typename, typename> typename T>
  SlowMapWriter& operator=(const T<nullFree, KEY, VAL>& rh) {
    assignFrom(rh);

    return *this;
  }

 private:
  // This allows us to share code between assignment operators.
  // Ensure rh supports a map-like iterator interface before calling this
  // function.
  template <typename T>
  inline void assignFrom(const T& rh) {
    IMapVal<KEY, VAL>::clear();
    for (const auto& it : rh) {
      IMapVal<KEY, VAL>::emplace(it.first, it.second);
    }
  }
};

} // namespace core
} // namespace velox
} // namespace facebook
