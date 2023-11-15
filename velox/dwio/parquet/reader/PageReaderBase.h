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

namespace facebook::velox::parquet {

enum PageReaderType {
  Common = 0,
  IAA = 1,
};

class PageReaderBase {
 public:
  //   explicit PageReaderBase(){};

  virtual ~PageReaderBase(){};

  virtual PageReaderType getType() = 0;

  /**
   * skips 'numValues' top level rows, touching null flags only.
   * Non-null values are not prepared for reading.
   * @param numValues
   * @return void
   */
  virtual void skipNullsOnly(int64_t numValues) = 0;

  /**
   * Reads 'numValues' null flags into 'nulls' and advances the
   * decoders by as much. The read may span several pages. If there
   * are no nulls, buffer may be set to nullptr.
   * @param numValues
   * @param buffer
   * @return void
   */
  virtual void readNullsOnly(int64_t numValues, BufferPtr& buffer) = 0;

  /**
   * Advances 'numRows' top level rows.
   * @param numRows
   * @return void
   */
  virtual void skip(int64_t numRows) = 0;

  /* Returns the current string dictionary as a FlatVector<StringView>.
   * @param type
   * @return VectorPtr
   */
  virtual const VectorPtr& dictionaryValues(const TypePtr& type) = 0;

  virtual bool isDictionary() const = 0;

  virtual void clearDictionary() = 0;

  virtual bool isDeltaBinaryPacked() const = 0;
};

} // namespace facebook::velox::parquet