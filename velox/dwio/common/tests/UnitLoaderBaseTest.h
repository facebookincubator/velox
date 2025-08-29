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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/dwio/common/UnitLoaderTools.h"
#include "velox/dwio/common/tests/utils/UnitLoaderTestTools.h"

/// Base test class template that provides common test functionality for
/// different UnitLoader implementations. This template class can be inherited
/// by specific test classes to get access to common test methods. Each derived
/// class should provide a createFactory() method that returns the appropriate
/// factory instance.
template <typename UnitLoaderFactoryType>
class UnitLoaderBaseTest : public ::testing::Test {
 protected:
  /// Factory method to create the appropriate UnitLoaderFactory instance.
  /// This method should be implemented by derived classes.
  virtual UnitLoaderFactoryType createFactory() = 0;

  /// Test that UnitLoader handles the case where no units exist but skip is
  /// requested
  void testNoUnitButSkip();

  /// Test that UnitLoader handles initial skip correctly for various skip
  /// values
  void testInitialSkip();

  /// Test that the same unit can be requested multiple times without issues
  void testCanRequestUnitMultipleTimes();

  /// Test that requesting a unit index out of range throws an exception
  void testUnitOutOfRange();

  /// Test that seeking out of range throws an exception
  void testSeekOutOfRange();

  /// Test that seeking out of range in ReaderMock throws appropriate exception
  void testSeekOutOfRangeReaderError();
};
