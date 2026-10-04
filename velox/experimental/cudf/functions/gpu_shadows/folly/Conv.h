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

// GPU shadow for <folly/Conv.h>, which nvcc's front end rejects. Declares
// tryTo, which a host-only body in DateTimeImpl.h names, so that it parses.
// Not defined: a call would fail to link.
#pragma once

#include <folly/Expected.h>

namespace folly {

enum class ConversionCode : unsigned char {
  SUCCESS,
};

template <class Tgt, class Src>
Expected<Tgt, ConversionCode> tryTo(const Src& value);

} // namespace folly
