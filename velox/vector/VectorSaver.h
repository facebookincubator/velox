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

#include "velox/vector/BaseVector.h"

namespace facebook::velox {

// Serialization format used by the following function is documented at
// https://facebookincubator.github.io/velox/develop/debugging/vector-saver.html

/// Serializes the type into binary format and writes it to the provided
/// output stream. Used for testing.
void saveType(const TypePtr& type, std::ostream& out);

/// Deserializes a type serialized by 'saveType' from the provided input stream.
/// Used for testing.
TypePtr restoreType(std::istream& in);

/// Serializes the vector into binary format and writes it to the provided
/// output stream. The serialization format preserves the encoding.
void saveVector(const BaseVector& vector, std::ostream& out);

/// Serializes the vector into binary format and writes it to a new file in
/// 'filePath'. The serialization preserved encoding. Exceptions will be thrown
/// if any error occurs while writing.
void saveVectorToFile(
    const BaseVector* FOLLY_NONNULL vector,
    const char* FOLLY_NONNULL filePath);

/// Writes 'content' to a new file in 'filePath'. Exceptions will be thrown if
/// any error occurs while writing.
void saveStringToFile(
    const std::string& content,
    const char* FOLLY_NONNULL filePath);

/// Deserializes a vector serialized by 'save' from the provided input stream.
VectorPtr restoreVector(
    std::istream& in,
    memory::MemoryPool* FOLLY_NONNULL pool);

/// Reads and deserializes a vector from a file stored by saveVectorToFile()
/// method call
VectorPtr restoreVectorFromFile(
    const char* FOLLY_NONNULL filePath,
    memory::MemoryPool* FOLLY_NONNULL pool);

/// Reads a string from a file stored by saveStringToFile() method
std::string restoreStringFromFile(const char* FOLLY_NONNULL filePath);

// Write the vector to a file. Contents would include the size of the list
// followed by all the values.
template <typename T>
void saveStdVectorToFile(
    const std::vector<T>& list,
    const char* FOLLY_NONNULL filePath);

// Reads a std::vector from a file stored by saveStdVectorToFile() method.
template <typename T>
std::vector<T> restoreStdVectorFromFile(const char* FOLLY_NONNULL filePath);

/// Serializes a SelectivityVector into binary format and writes it to the
/// provided output stream.
void saveSelectivityVector(const SelectivityVector& rows, std::ostream& out);

/// Deserializes a SelectivityVector serialized by 'saveSelectivityVector' from
/// the provided input stream.
SelectivityVector restoreSelectivityVector(std::istream& in);

} // namespace facebook::velox
