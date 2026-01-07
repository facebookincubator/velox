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

#include <map>

#include <fmt/format.h>
#include <folly/Conv.h>

#include "folly/dynamic.h"
#include "velox/common/base/Exceptions.h"
#include "velox/type/Conversions.h"
#include "velox/type/CppToType.h"
#include "velox/type/Type.h"

namespace facebook::velox {

/// Variant is a dynamically-typed container that can hold values of any Velox
/// type, providing a type-safe way to work with heterogeneous data at runtime.
///
/// A Variant encapsulates a value along with its TypeKind, supporting all Velox
/// types including scalars (INTEGER, VARCHAR, DOUBLE, etc.), complex types
/// (ARRAY, MAP, ROW), and special types (TIMESTAMP, OPAQUE). Variants can also
/// represent null values of any type.
///
/// Note that Variants only store the physical type of a variable (the
/// TypeKind), and not its logical type (TypePtr). Variants also do not store
/// the field names of a row/struct.
///
/// **IMPORTANT**: Variants are NOT intended for performance-critical code
/// paths. For production workloads processing large amounts of data, use
/// Vectors instead:
///
/// ## Supported Types
///
/// Scalar types:
/// - Numeric: BOOLEAN, TINYINT, SMALLINT, INTEGER, BIGINT, HUGEINT, REAL,
///   DOUBLE
/// - String: VARCHAR, VARBINARY
/// - Temporal: TIMESTAMP, DATE (via INTEGER)
/// - Special: UNKNOWN (null variant)
///
/// Complex types:
/// - ARRAY: Ordered collection of Variants (all elements must have same
///   TypeKind)
/// - MAP: Key-value mapping of Variants
/// - ROW: Heterogeneous tuple of Variants
///
/// Special types:
/// - OPAQUE: Type-erased wrapper for custom C++ objects
///
/// ## Construction and Factory Methods
///
/// Variants can be created via:
/// - Implicit constructors from native types: `Variant(42)`, `Variant("hello")`
/// - Explicit factory methods: `Variant::create<KIND>(value)`
/// - Type-specific factories: `Variant::array()`, `Variant::map()`,
///   `Variant::row()`
/// - Null values: `Variant::null(TypeKind)` or `Variant(TypeKind)`
/// - Deserialization: `Variant::create(folly::dynamic)`
///
/// ## Type Safety and Value Access
///
/// Variants provide runtime type checking:
/// - `kind()` returns the TypeKind
/// - `value<KIND>()` extracts the value (throws if wrong type)
/// - `isNull()` checks for null values
/// - `inferType()` returns the full Type descriptor
///
/// ## Example Usage
///
/// ```cpp
/// // Creating variants
/// Variant intVar = 42;                    // INTEGER
/// Variant strVar = "hello";               // VARCHAR
/// Variant dblVar = 3.14;                  // DOUBLE
/// Variant nullVar = Variant::null(TypeKind::INTEGER);
///
/// // Complex types
/// Variant arr = Variant::array({Variant(1), Variant(2), Variant(3)});
/// Variant map = Variant::map({{Variant("key"), Variant("value")}});
/// Variant row = Variant::row({Variant(1), Variant("test"), Variant(3.14)});
///
/// // Accessing values
/// int32_t i = intVar.value<TypeKind::INTEGER>();
/// std::string s = strVar.value<TypeKind::VARCHAR>();
/// const auto& elements = arr.value<TypeKind::ARRAY>();
///
/// // Serialization
/// auto serialized = arr.serialize();              // to folly::dynamic
/// auto deserialized = Variant::create(serialized); // from folly::dynamic
/// assert(arr == deserialized);
/// ```
///
/// ## Memory and Lifetime
///
/// - Variants own their data and perform deep copies on assignment.
/// - Values are heap-allocated (except for primitive types).
/// - Variants are movable and copyable with expected semantics
/// - For OPAQUE types, Variants hold shared_ptr, allowing shared ownership.
class Variant;

namespace detail {
template <typename T, TypeKind KIND, bool usesCustomComparison>
struct TypeStorage {
  T storedValue;
};

template <typename T, TypeKind KIND>
struct TypeStorage<T, KIND, true> {
  T storedValue;
  std::shared_ptr<const CanProvideCustomComparisonType<KIND>>
      typeWithCustomComparison;
};

template <TypeKind KIND>
using scalar_stored_type = typename TypeTraits<KIND>::DeepCopiedType;

template <TypeKind KIND, bool usesCustomComparison = false, typename = void>
struct VariantTypeTraits {};

template <TypeKind KIND, bool usesCustomComparison>
struct VariantTypeTraits<
    KIND,
    usesCustomComparison,
    std::enable_if_t<
        TypeTraits<KIND>::isPrimitiveType && !is_string_kind(KIND),
        void>> {
  using native_type = typename TypeTraits<KIND>::NativeType;
  using stored_type =
      TypeStorage<scalar_stored_type<KIND>, KIND, usesCustomComparison>;
  using value_type = scalar_stored_type<KIND>;
};

template <TypeKind KIND, bool usesCustomComparison>
struct VariantTypeTraits<
    KIND,
    usesCustomComparison,
    std::enable_if_t<is_string_kind(KIND), void>> {
  using native_type = std::string_view;
  using stored_type =
      TypeStorage<scalar_stored_type<KIND>, KIND, usesCustomComparison>;
  using value_type = scalar_stored_type<KIND>;
};

template <bool usesCustomComparison>
struct VariantTypeTraits<TypeKind::ROW, usesCustomComparison> {
  using stored_type =
      TypeStorage<std::vector<Variant>, TypeKind::ROW, usesCustomComparison>;
  using value_type = std::vector<Variant>;
};

template <bool usesCustomComparison>
struct VariantTypeTraits<TypeKind::MAP, usesCustomComparison> {
  using stored_type = TypeStorage<
      std::map<Variant, Variant>,
      TypeKind::MAP,
      usesCustomComparison>;
  using value_type = std::map<Variant, Variant>;
};

template <bool usesCustomComparison>
struct VariantTypeTraits<TypeKind::ARRAY, usesCustomComparison> {
  using stored_type =
      TypeStorage<std::vector<Variant>, TypeKind::ARRAY, usesCustomComparison>;
  using value_type = std::vector<Variant>;
};

struct OpaqueCapsule {
  std::shared_ptr<const OpaqueType> type;
  std::shared_ptr<void> obj;

  bool operator<(const OpaqueCapsule& other) const {
    if (type->typeIndex() == other.type->typeIndex()) {
      return obj < other.obj;
    }
    return type->typeIndex() < other.type->typeIndex();
  }

  bool operator==(const OpaqueCapsule& other) const {
    return type->typeIndex() == other.type->typeIndex() && obj == other.obj;
  }
};

template <bool usesCustomComparison>
struct VariantTypeTraits<TypeKind::OPAQUE, usesCustomComparison> {
  using stored_type =
      TypeStorage<OpaqueCapsule, TypeKind::OPAQUE, usesCustomComparison>;
  using value_type = OpaqueCapsule;
};
} // namespace detail

class Variant {
 public:
  /// Constructs a NULL variant with type set to UNKNOWN.
  Variant()
      : ptr_{nullptr}, kind_{TypeKind::UNKNOWN}, usesCustomComparison_(false) {}

  /* implicit */ Variant(TypeKind kind)
      : ptr_{nullptr}, kind_{kind}, usesCustomComparison_(false) {}

  Variant(const Variant& other)
      : ptr_{nullptr},
        kind_{other.kind_},
        usesCustomComparison_(other.usesCustomComparison_) {
    auto op = other.ptr_;
    if (op != nullptr) {
      dynamicCopy(other.ptr_, other.kind_);
    }
  }

  /// String construction methods.
  // Support construction from velox::StringView as well as std::string_view.
  /* implicit */ Variant(StringView view) : Variant{std::string_view{view}} {}

  // Break ties between implicit conversions to StringView/std::string_view.
  /* implicit */ Variant(std::string str)
      : ptr_{new std::string{std::move(str)}},
        kind_{TypeKind::VARCHAR},
        usesCustomComparison_(false) {}

  /* implicit */ Variant(const char* str)
      : ptr_{new std::string{str}},
        kind_{TypeKind::VARCHAR},
        usesCustomComparison_(false) {}

#define VELOX_VARIANT_SCALAR_CONSTRUCTOR(KIND)                            \
  /* implicit */ Variant(                                                 \
      typename detail::VariantTypeTraits<KIND, false>::native_type v)     \
      : ptr_{new detail::VariantTypeTraits<KIND, false>::stored_type{v}}, \
        kind_{KIND},                                                      \
        usesCustomComparison_{false} {}

  VELOX_VARIANT_SCALAR_CONSTRUCTOR(TypeKind::BOOLEAN)
  VELOX_VARIANT_SCALAR_CONSTRUCTOR(TypeKind::TINYINT)
  VELOX_VARIANT_SCALAR_CONSTRUCTOR(TypeKind::SMALLINT)
  VELOX_VARIANT_SCALAR_CONSTRUCTOR(TypeKind::INTEGER)
  VELOX_VARIANT_SCALAR_CONSTRUCTOR(TypeKind::BIGINT)
  VELOX_VARIANT_SCALAR_CONSTRUCTOR(TypeKind::HUGEINT)
  VELOX_VARIANT_SCALAR_CONSTRUCTOR(TypeKind::REAL)
  VELOX_VARIANT_SCALAR_CONSTRUCTOR(TypeKind::DOUBLE)
  // VARBINARY conflicts with VARCHAR, so we don't gen these methods
  // VELOX_VARIANT_SCALAR_CONSTRUCTOR(TypeKind::VARBINARY);
  VELOX_VARIANT_SCALAR_CONSTRUCTOR(TypeKind::TIMESTAMP)
  VELOX_VARIANT_SCALAR_CONSTRUCTOR(TypeKind::UNKNOWN)
#undef VELOX_VARIANT_SCALAR_CONSTRUCTOR

  /* implicit */ Variant(
      typename detail::VariantTypeTraits<TypeKind::VARCHAR, false>::native_type
          v)
      : ptr_{new detail::VariantTypeTraits<TypeKind::VARCHAR, false>::
                 stored_type{std::string(v)}},
        kind_{TypeKind::VARCHAR},
        usesCustomComparison_{false} {}

  // On 64-bit platforms `int64_t` is declared as `long int`, not `long long
  // int`, thus adding an extra overload to make literals like 1LL resolve
  // correctly. Note that one has to use template T because otherwise SFINAE
  // doesn't work, but in this case T = long long
  template <
      typename T = long long,
      std::enable_if_t<
          std::is_same_v<T, long long> && !std::is_same_v<long long, int64_t>,
          bool> = true>
  /* implicit */ Variant(const T& v) : Variant(static_cast<int64_t>(v)) {}

  // Move constructor.
  Variant(Variant&& other) noexcept
      : ptr_{other.ptr_},
        kind_{other.kind_},
        usesCustomComparison_(other.usesCustomComparison_) {
    other.ptr_ = nullptr;
  }

  ~Variant() {
    if (ptr_ != nullptr) {
      dynamicFree();
    }
  }

  /// Creates a non-null Variant of the specified TypeKind with the provided
  /// value.
  ///
  /// This factory method provides explicit control over the Variant's TypeKind
  /// and is particularly useful for:
  ///
  /// - Creating variants for complex types (ARRAY, MAP, ROW, VARCHAR,
  ///   VARBINARY) that require std::vector or std::map containers
  /// - Disambiguating between similar types (e.g., VARCHAR vs VARBINARY)
  /// - Creating variants when the TypeKind is determined at runtime or from
  ///   template parameters
  ///
  /// @tparam KIND The TypeKind of the variant to create (e.g., TypeKind::ARRAY,
  ///              TypeKind::VARCHAR, TypeKind::INTEGER)
  /// @param v The value to store in the variant. The type must match the
  ///          expected value_type for the given KIND:
  ///          - Scalar types: Native C++ type (int8_t, int32_t, float, etc.)
  ///          - VARCHAR/VARBINARY: std::string
  ///          - ARRAY/ROW: std::vector<Variant>
  ///          - MAP: std::map<Variant, Variant>
  ///          - TIMESTAMP: Timestamp
  ///          - OPAQUE: OpaqueCapsule
  ///          The parameter is passed by value and moved into the Variant.
  ///
  /// @return A new non-null Variant containing the provided value
  ///
  /// Example usage:
  ///   // Create an array variant
  ///   auto arr = Variant::create<TypeKind::ARRAY>(
  ///       std::vector<Variant>{Variant(1), Variant(2), Variant(3)});
  ///
  ///   // Create a string variant (disambiguating VARCHAR)
  ///   auto str = Variant::create<TypeKind::VARCHAR>(std::string("hello"));
  ///
  ///   // Create a map variant
  ///   std::map<Variant, Variant> m = {{Variant(1), Variant("one")}};
  ///   auto mapVar = Variant::create<TypeKind::MAP>(std::move(m));
  template <TypeKind KIND>
  static Variant create(
      typename detail::VariantTypeTraits<KIND, false>::value_type v) {
    return Variant{
        KIND,
        new typename detail::VariantTypeTraits<KIND, false>::stored_type{
            std::move(v)},
    };
  }

  // Explicit specializations for other non-deep copied string types.
  template <TypeKind KIND>
  static std::enable_if_t<is_string_kind(KIND), Variant> create(StringView v) {
    return create<KIND>(std::string(v));
  }

  template <TypeKind KIND>
  static std::enable_if_t<is_string_kind(KIND), Variant> create(
      std::string_view v) {
    return create<KIND>(std::string(v));
  }

  template <TypeKind KIND>
  static std::enable_if_t<is_string_kind(KIND), Variant> create(const char* v) {
    return create<KIND>(std::string(v));
  }

  /// Creates a non-null Variant by deducing the TypeKind from the C++ template
  /// type parameter.
  ///
  /// This overload automatically maps C++ types to their corresponding TypeKind
  /// using CppToType traits. Note that while this function provides a more
  /// convenient way to map directly from a C++ type to TypeKind, it is not
  /// guaranteed that every TypeKind will have a direct C++ type mapping. So
  /// this is mostly useful for test code.
  ///
  /// @tparam T The C++ type that determines the TypeKind (e.g., int32_t maps to
  ///           TypeKind::INTEGER, std::string maps to TypeKind::VARCHAR)
  /// @param v The value to store in the variant. Must be of type value_type for
  ///          the deduced TypeKind. The parameter is passed by const reference
  ///          and copied into the Variant.
  ///
  /// @return A new non-null Variant containing the provided value
  ///
  /// Example usage:
  ///   // Deduce TypeKind::INTEGER from int32_t
  ///   auto intVar = Variant::create<int32_t>(42);
  ///
  ///   // Deduce TypeKind::DOUBLE from double
  ///   auto doubleVar = Variant::create<double>(3.14);
  ///
  /// Note: For most scalar types, the implicit Variant constructors are more
  /// convenient (e.g., Variant(42) instead of Variant::create<int32_t>(42)).
  /// This method is primarily useful when type deduction from template
  /// parameters is needed or for consistency with the TypeKind-based overload.
  template <typename T>
  static Variant create(
      const typename detail::VariantTypeTraits<CppToType<T>::typeKind, false>::
          value_type& v) {
    return create<CppToType<T>::typeKind>(v);
  }

  /// Deserializes a Variant from a folly::dynamic object.
  ///
  /// This method reconstructs a Variant from its serialized representation,
  /// serving as the inverse operation of serialize(). It parses a
  /// folly::dynamic object that contains type metadata and value data, and
  /// creates the corresponding Variant.
  ///
  /// @param obj A folly::dynamic object containing the serialized Variant data.
  ///            The object must have the following structure:
  ///            - "type": A string field specifying the TypeKind name (e.g.,
  ///              "INTEGER", "VARCHAR", "ARRAY", "MAP", "ROW", "TIMESTAMP")
  ///            - "value": The actual value, whose format depends on the type:
  ///
  ///              Scalar types (BOOLEAN, TINYINT, SMALLINT, INTEGER, BIGINT,
  ///              HUGEINT, REAL, DOUBLE, VARCHAR):
  ///                The primitive value directly (int, double, string, bool)
  ///
  ///              VARBINARY:
  ///                Base64-encoded string representation of binary data
  ///
  ///              ARRAY or ROW:
  ///                folly::dynamic array where each element is a serialized
  ///                Variant object (recursive structure)
  ///
  ///              MAP:
  ///                folly::dynamic object with two fields:
  ///                - "keys": Array of serialized Variant objects
  ///                - "values": Array of serialized Variant objects
  ///                Both arrays must have the same length
  ///
  ///              TIMESTAMP:
  ///                folly::dynamic object with two fields:
  ///                - "seconds": Integer seconds since epoch
  ///                - "nanos": Integer nanoseconds component
  ///
  ///              Null values:
  ///                null (nullptr in folly::dynamic)
  ///
  /// @return A Variant containing the deserialized value
  ///
  /// @throws VeloxUserError if the object structure is invalid (e.g., unknown
  ///         type name, malformed value data, mismatched array sizes for maps)
  ///
  /// Example usage:
  ///   // Deserialize a scalar value
  ///   folly::dynamic obj = folly::dynamic::object("type", "INTEGER")
  ///                                               ("value", 42);
  ///   auto var = Variant::create(obj);
  ///
  ///   // Deserialize an array
  ///   folly::dynamic arrObj = folly::dynamic::object
  ///     ("type", "ARRAY")
  ///     ("value", folly::dynamic::array(
  ///       folly::dynamic::object("type", "INTEGER")("value", 1),
  ///       folly::dynamic::object("type", "INTEGER")("value", 2)
  ///     ));
  ///   auto arrVar = Variant::create(arrObj);
  ///
  ///   // Round-trip serialization/deserialization
  ///   Variant original = Variant::array({Variant(1), Variant(2)});
  ///   auto serialized = original.serialize();
  ///   auto deserialized = Variant::create(serialized);
  ///   // original == deserialized
  ///
  /// Note: This method is commonly used for persistence, network transmission,
  /// or interfacing with JSON-based systems. For creating Variants from native
  /// C++ values, use the template-based create() methods or constructors
  /// instead.
  static Variant create(const folly::dynamic& obj);

  static Variant row(const std::vector<Variant>& inputs) {
    return {
        TypeKind::ROW,
        new
        typename detail::VariantTypeTraits<TypeKind::ROW, false>::stored_type{
            inputs}};
  }

  static Variant row(std::vector<Variant>&& inputs) {
    return {
        TypeKind::ROW,
        new
        typename detail::VariantTypeTraits<TypeKind::ROW, false>::stored_type{
            std::move(inputs)}};
  }

  static Variant map(const std::map<Variant, Variant>& inputs) {
    return {
        TypeKind::MAP,
        new
        typename detail::VariantTypeTraits<TypeKind::MAP, false>::stored_type{
            inputs}};
  }

  static Variant map(std::map<Variant, Variant>&& inputs) {
    return {
        TypeKind::MAP,
        new
        typename detail::VariantTypeTraits<TypeKind::MAP, false>::stored_type{
            std::move(inputs)}};
  }

  static Variant timestamp(const Timestamp& input) {
    return {
        TypeKind::TIMESTAMP,
        new typename detail::VariantTypeTraits<TypeKind::TIMESTAMP, false>::
            stored_type{input}};
  }

  template <TypeKind KIND>
  static Variant typeWithCustomComparison(
      typename TypeTraits<KIND>::NativeType input,
      const TypePtr& type) {
    using variant_traits = detail::VariantTypeTraits<KIND, true>;
    return {
        KIND,
        new typename variant_traits::stored_type{
            typename variant_traits::value_type{std::move(input)},
            std::dynamic_pointer_cast<
                const CanProvideCustomComparisonType<KIND>>(type)},
        true};
  }

  template <class T>
  static Variant opaque(const std::shared_ptr<T>& input) {
    VELOX_CHECK(input.get(), "Can't create a Variant of nullptr opaque type");
    return {
        TypeKind::OPAQUE,
        new detail::OpaqueCapsule{OpaqueType::create<T>(), input}};
  }

  static Variant opaque(
      const std::shared_ptr<void>& input,
      const std::shared_ptr<const OpaqueType>& type) {
    VELOX_CHECK(input.get(), "Can't create a Variant of nullptr opaque type");
    return {TypeKind::OPAQUE, new detail::OpaqueCapsule{type, input}};
  }

  static Variant array(const std::vector<Variant>& inputs) {
    verifyArrayElements(inputs);
    return {
        TypeKind::ARRAY,
        new
        typename detail::VariantTypeTraits<TypeKind::ARRAY, false>::stored_type{
            inputs}};
  }

  static Variant array(std::vector<Variant>&& inputs) {
    verifyArrayElements(inputs);
    return {
        TypeKind::ARRAY,
        new
        typename detail::VariantTypeTraits<TypeKind::ARRAY, false>::stored_type{
            std::move(inputs)}};
  }

  static void verifyArrayElements(const std::vector<Variant>& inputs);

  static Variant null(TypeKind kind) {
    return Variant{kind};
  }

  static Variant binary(std::string val) {
    return Variant{TypeKind::VARBINARY, new std::string{std::move(val)}};
  }

 private:
  Variant(TypeKind kind, void* ptr, bool usesCustomComparison = false)
      : ptr_{ptr}, kind_{kind}, usesCustomComparison_(usesCustomComparison) {}

  template <TypeKind KIND>
  bool lessThan(const Variant& other) const;

  template <TypeKind KIND>
  bool equals(const Variant& other) const;

  template <TypeKind KIND>
  uint64_t hash() const;

  template <bool usesCustomComparison, TypeKind KIND>
  void typedDestroy() {
    delete static_cast<const typename detail::VariantTypeTraits<
        KIND,
        usesCustomComparison>::stored_type*>(ptr_);
    ptr_ = nullptr;
  }

  template <bool usesCustomComparison, TypeKind KIND>
  void typedCopy(const void* other) {
    using stored_type = typename detail::
        VariantTypeTraits<KIND, usesCustomComparison>::stored_type;
    ptr_ = new stored_type{*static_cast<const stored_type*>(other)};
  }

  void dynamicCopy(const void* p, const TypeKind kind) {
    if (usesCustomComparison_) {
      VELOX_DYNAMIC_TEMPLATE_TYPE_DISPATCH_ALL(typedCopy, true, kind, p);
    } else {
      VELOX_DYNAMIC_TEMPLATE_TYPE_DISPATCH_ALL(typedCopy, false, kind, p);
    }
  }

  void dynamicFree() {
    if (usesCustomComparison_) {
      VELOX_DYNAMIC_TEMPLATE_TYPE_DISPATCH_ALL(typedDestroy, true, kind_);
    } else {
      VELOX_DYNAMIC_TEMPLATE_TYPE_DISPATCH_ALL(typedDestroy, false, kind_);
    }
  }

  [[noreturn]] void throwCheckIsKindError(TypeKind kind) const;

  [[noreturn]] void throwCheckPtrError() const;

 public:
  // Constant used in comparison of REAL and DOUBLE values.
  static constexpr double kEpsilon{0.00001};

  struct Hasher {
    size_t operator()(Variant const& input) const noexcept {
      return input.hash();
    }
  };

  struct NullEqualsNullsComparator {
    bool operator()(const Variant& a, const Variant& b) const {
      return a.equalsWithNullEqualsNull(b);
    }
  };

  Variant& operator=(const Variant& other) {
    if (ptr_ != nullptr) {
      dynamicFree();
    }
    kind_ = other.kind_;
    usesCustomComparison_ = other.usesCustomComparison_;
    if (other.ptr_ != nullptr) {
      dynamicCopy(other.ptr_, other.kind_);
    }
    return *this;
  }

  Variant& operator=(Variant&& other) noexcept {
    if (ptr_ != nullptr) {
      dynamicFree();
    }
    kind_ = other.kind_;
    usesCustomComparison_ = other.usesCustomComparison_;
    if (other.ptr_ != nullptr) {
      ptr_ = other.ptr_;
      other.ptr_ = nullptr;
    }
    return *this;
  }

  bool operator<(const Variant& other) const;

  bool equals(const Variant& other) const;

  bool equalsWithNullEqualsNull(const Variant& other) const;

  std::string toJson(const TypePtr& type) const;

  std::string toJson(const Type& type) const;

  std::string toJsonUnsafe(const TypePtr& type = nullptr) const;

  /// Used by python binding, do not change signature.
  std::string pyToJson() const {
    return toJsonUnsafe();
  }

  /// Returns a string of the Variant value.
  std::string toString(const TypePtr& type) const;

  /// Returns a string representation identical to
  /// BaseVector::createConstant(type, *this)->toString(0).
  std::string toStringAsVector(const TypePtr& type) const;

  folly::dynamic serialize() const;

  bool hasValue() const {
    return !isNull();
  }

  /// Similar to hasValue(). Legacy.
  bool isSet() const {
    return hasValue();
  }

  void checkPtr() const {
    if (ptr_ == nullptr) {
      // Error path outlined to encourage inlining of the branch.
      throwCheckPtrError();
    }
  }

  void checkIsKind(TypeKind kind) const {
    if (kind_ != kind) {
      // Error path outlined to encourage inlining of the branch.
      throwCheckIsKindError(kind);
    }
  }

  TypeKind kind() const {
    return kind_;
  }

  template <TypeKind KIND>
  const auto& value() const {
    checkIsKind(KIND);
    checkPtr();

    if (usesCustomComparison_) {
      return static_cast<const typename detail::VariantTypeTraits<KIND, true>::
                             stored_type*>(ptr_)
          ->storedValue;
    } else {
      return static_cast<const typename detail::VariantTypeTraits<KIND, false>::
                             stored_type*>(ptr_)
          ->storedValue;
    }
  }

  template <typename T>
  const auto& value() const {
    return value<CppToType<T>::typeKind>();
  }

  bool isNull() const {
    return ptr_ == nullptr;
  }

  uint64_t hash() const;

  template <TypeKind KIND>
  const auto* valuePointer() const {
    checkIsKind(KIND);

    if (usesCustomComparison_) {
      return static_cast<
          const typename detail::VariantTypeTraits<KIND, true>::stored_type*>(
          ptr_);
    } else {
      return static_cast<
          const typename detail::VariantTypeTraits<KIND, false>::stored_type*>(
          ptr_);
    }
  }

  template <typename T>
  const auto* valuePointer() const {
    return valuePointer<CppToType<T>::typeKind>();
  }

  const std::vector<Variant>& row() const {
    return value<TypeKind::ROW>();
  }

  /// Returns a map of Variants.
  const std::map<Variant, Variant>& map() const {
    return value<TypeKind::MAP>();
  }

  /// Returns a map of primitive values. Assumes all keys and values are not
  /// null.
  template <typename K, typename V>
  std::map<K, V> map() const {
    const auto& variants = value<TypeKind::MAP>();

    std::map<K, V> values;
    for (const auto& [k, v] : variants) {
      values.emplace(k.template value<K>(), v.template value<V>());
    }

    return values;
  }

  /// Returns a map of optional primitive values. All keys are assumed to be not
  /// null. Null values are returned as unset optional.
  template <typename K, typename V>
  std::map<K, std::optional<V>> nullableMap() const {
    const auto& variants = value<TypeKind::MAP>();

    std::map<K, std::optional<V>> values;
    for (const auto& [k, v] : variants) {
      if (v.isNull()) {
        values.emplace(k.template value<K>(), std::nullopt);
      } else {
        values.emplace(k.template value<K>(), v.template value<V>());
      }
    }

    return values;
  }

  /// Returns a std::map of primitive keys to std::vector of primitive values.
  /// Assumes all keys and values are not null.
  template <typename K, typename V>
  std::map<K, std::vector<V>> mapOfArrays() const {
    const auto& variants = value<TypeKind::MAP>();

    std::map<K, std::vector<V>> values;
    for (const auto& [k, v] : variants) {
      values.emplace(k.template value<K>(), v.template array<V>());
    }

    return values;
  }

  /// Returns a std::map of primitive keys to optional std::vector of optional
  /// primitive values. Null values are returned as unset optional.
  template <typename K, typename V>
  std::map<K, std::optional<std::vector<std::optional<V>>>>
  nullableMapOfArrays() const {
    const auto& variants = value<TypeKind::MAP>();

    std::map<K, std::optional<std::vector<std::optional<V>>>> values;
    for (const auto& [k, v] : variants) {
      if (v.isNull()) {
        values.emplace(k.template value<K>(), std::nullopt);
      } else {
        values.emplace(k.template value<K>(), v.template nullableArray<V>());
      }
    }

    return values;
  }

  /// Returns a std::vector of Variants.
  const std::vector<Variant>& array() const {
    return value<TypeKind::ARRAY>();
  }

  /// Returns a std::vector of primitive values. Assumes that all values are not
  /// null.
  template <typename T>
  std::vector<T> array() const {
    const auto& variants = value<TypeKind::ARRAY>();

    std::vector<T> values;
    values.reserve(variants.size());

    for (const auto& v : variants) {
      values.emplace_back(v.template value<T>());
    }

    return values;
  }

  /// Returns a std::vector of optional primitive values. Null values are
  /// returned as unset optional.
  template <typename T>
  std::vector<std::optional<T>> nullableArray() const {
    const auto& variants = value<TypeKind::ARRAY>();

    std::vector<std::optional<T>> values;
    values.reserve(variants.size());

    for (const auto& v : variants) {
      if (v.isNull()) {
        values.emplace_back(std::nullopt);
      } else {
        values.emplace_back(v.template value<T>());
      }
    }

    return values;
  }

  /// Returns a std::vector of std::vector of primitive values. Assumes that all
  /// values are not null.
  template <typename T>
  std::vector<std::vector<T>> arrayOfArrays() const {
    const auto& variants = value<TypeKind::ARRAY>();

    std::vector<std::vector<T>> values;
    values.reserve(variants.size());

    for (const auto& v : variants) {
      values.emplace_back(v.template array<T>());
    }

    return values;
  }

  /// Returns a std::vector of std::vector of primitive values.Null values are
  /// returned as unset optional.
  template <typename T>
  std::vector<std::optional<std::vector<std::optional<T>>>>
  nullableArrayOfArrays() const {
    const auto& variants = value<TypeKind::ARRAY>();

    std::vector<std::optional<std::vector<std::optional<T>>>> values;
    values.reserve(variants.size());

    for (const auto& v : variants) {
      if (v.isNull()) {
        values.emplace_back(std::nullopt);
      } else {
        values.emplace_back(v.template nullableArray<T>());
      }
    }

    return values;
  }

  template <class T>
  std::shared_ptr<T> opaque() const {
    const auto& capsule = value<TypeKind::OPAQUE>();
    VELOX_CHECK(
        capsule.type->typeIndex() == std::type_index(typeid(T)),
        "Requested {} but contains {}",
        OpaqueType::create<T>()->toString(),
        capsule.type->toString());
    return std::static_pointer_cast<T>(capsule.obj);
  }

  /// Try to cast to the target custom type
  /// Throw if the Variant is not an opaque type
  /// Return nullptr if it's opaque type but the underlying custom type doesn't
  /// match the target. Otherwise return the data in custom type.
  template <class T>
  std::shared_ptr<T> tryOpaque() const {
    const auto& capsule = value<TypeKind::OPAQUE>();
    if (capsule.type->typeIndex() == std::type_index(typeid(T))) {
      return std::static_pointer_cast<T>(capsule.obj);
    }
    return nullptr;
  }

  TypePtr inferType() const;

  /// Returns true if the type of this Variant is compatible with the specified
  /// type. Similar to inferType()->kindEquals(type), but treats
  /// TypeKind::UNKNOWN equal to any other TypeKind.
  bool isTypeCompatible(const TypePtr& type) const;

  friend std::ostream& operator<<(std::ostream& stream, const Variant& k) {
    const auto type = k.inferType();
    stream << k.toJson(type);
    return stream;
  }

  // Uses kEpsilon to compare floating point types (REAL and DOUBLE).
  // For testing purposes.
  bool lessThanWithEpsilon(const Variant& other) const;

  // Uses kEpsilon to compare floating point types (REAL and DOUBLE).
  // For testing purposes.
  bool equalsWithEpsilon(const Variant& other) const;

 private:
  template <TypeKind KIND>
  std::shared_ptr<const CanProvideCustomComparisonType<KIND>>
  customComparisonType() const {
    return static_cast<const typename detail::VariantTypeTraits<KIND, true>::
                           stored_type*>(ptr_)
        ->typeWithCustomComparison;
  }

  // TODO: it'd be more efficient to put union here if it ever becomes a
  // problem
  const void* ptr_;
  TypeKind kind_;

  // If the Variant represents the value of a type that provides custom
  // comparisons.
  bool usesCustomComparison_;
};

inline bool operator==(const Variant& a, const Variant& b) {
  return a.equals(b);
}

struct VariantConverter {
  template <TypeKind FromKind, TypeKind ToKind>
  static Variant convert(const Variant& value) {
    if (value.isNull()) {
      return Variant{value.kind()};
    }

    const auto converted =
        util::Converter<ToKind>::tryCast(value.value<FromKind>())
            .thenOrThrow(folly::identity, [&](const Status& status) {
              VELOX_USER_FAIL("{}", status.message());
            });
    return {converted};
  }

  template <TypeKind ToKind>
  static Variant convert(const Variant& value) {
    switch (value.kind()) {
      case TypeKind::BOOLEAN:
        return convert<TypeKind::BOOLEAN, ToKind>(value);
      case TypeKind::TINYINT:
        return convert<TypeKind::TINYINT, ToKind>(value);
      case TypeKind::SMALLINT:
        return convert<TypeKind::SMALLINT, ToKind>(value);
      case TypeKind::INTEGER:
        return convert<TypeKind::INTEGER, ToKind>(value);
      case TypeKind::BIGINT:
        return convert<TypeKind::BIGINT, ToKind>(value);
      case TypeKind::REAL:
        return convert<TypeKind::REAL, ToKind>(value);
      case TypeKind::DOUBLE:
        return convert<TypeKind::DOUBLE, ToKind>(value);
      case TypeKind::VARCHAR:
        return convert<TypeKind::VARCHAR, ToKind>(value);
      case TypeKind::VARBINARY:
        return convert<TypeKind::VARBINARY, ToKind>(value);
      case TypeKind::TIMESTAMP:
      case TypeKind::HUGEINT:
        // Default date/timestamp conversion is prone to errors and implicit
        // assumptions. Block converting timestamp to integer, double and
        // std::string types. The callers should implement their own
        // conversion
        //  from value.
        VELOX_NYI();
      default:
        VELOX_NYI();
    }
  }

  static Variant convert(const Variant& value, TypeKind toKind) {
    return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(convert, toKind, value);
  }
};

// For backward compatibility.
using variant = Variant;

} // namespace facebook::velox
