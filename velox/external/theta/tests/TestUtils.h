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

// Adapted from Apache DataSketches

#pragma once

#include <string>
#include <type_traits>
#include "velox/common/base/Exceptions.h"

namespace {
class Approx {
 private:
  bool marginComparison(double lhs, double rhs, double margin) const {
    return (lhs + margin >= rhs) && (rhs + margin >= lhs);
  }

  bool equalityComparisonImpl(const double other) const {
    // First try with fixed margin, then compute margin based on epsilon, scale
    // and Approx's value Thanks to Richard Harris for his help refining the
    // scaled margin value
    return marginComparison(m_value, other, m_margin) ||
        marginComparison(
               m_value,
               other,
               m_epsilon *
                   (m_scale + std::fabs(std::isinf(m_value) ? 0 : m_value)));
  }
  // Validates the new margin (margin >= 0)
  // out-of-line to avoid including stdexcept in the header
  void setMargin(double newMargin) {
    VELOX_CHECK(
        newMargin >= 0,
        "Invalid Approx::margin: {} Approx::Margin has to be non-negative.",
        newMargin);
    m_margin = newMargin;
  }

  // Validates the new epsilon (0 < epsilon < 1)
  // out-of-line to avoid including stdexcept in the header
  void setEpsilon(double newEpsilon) {
    VELOX_CHECK(
        newEpsilon >= 0 && newEpsilon <= 1.0,
        "Invalid Approx::epsilon: {} Approx::epsilon has to be in [0, 1]",
        newEpsilon);
    m_epsilon = newEpsilon;
  }

 public:
  explicit Approx(double value)
      : m_epsilon(std::numeric_limits<float>::epsilon() * 100),
        m_margin(0.0),
        m_scale(0.0),
        m_value(value) {}

  static Approx custom() {
    return Approx(0);
  }

  Approx operator-() const {
    auto temp(*this);
    temp.m_value = -temp.m_value;
    return temp;
  }

  template <
      typename T,
      typename = typename std::enable_if<
          std::is_constructible<double, T>::value>::type>
  Approx operator()(T const& value) const {
    Approx approx(static_cast<double>(value));
    approx.m_epsilon = m_epsilon;
    approx.m_margin = m_margin;
    approx.m_scale = m_scale;
    return approx;
  }

  template <
      typename T,
      typename = typename std::enable_if<
          std::is_constructible<double, T>::value>::type>
  explicit Approx(T const& value) : Approx(static_cast<double>(value)) {}

  template <
      typename T,
      typename = typename std::enable_if<
          std::is_constructible<double, T>::value>::type>
  friend bool operator==(const T& lhs, Approx const& rhs) {
    auto lhs_v = static_cast<double>(lhs);
    return rhs.equalityComparisonImpl(lhs_v);
  }

  template <
      typename T,
      typename = typename std::enable_if<
          std::is_constructible<double, T>::value>::type>
  friend bool operator==(Approx const& lhs, const T& rhs) {
    return operator==(rhs, lhs);
  }

  template <
      typename T,
      typename = typename std::enable_if<
          std::is_constructible<double, T>::value>::type>
  friend bool operator!=(T const& lhs, Approx const& rhs) {
    return !operator==(lhs, rhs);
  }

  template <
      typename T,
      typename = typename std::enable_if<
          std::is_constructible<double, T>::value>::type>
  friend bool operator!=(Approx const& lhs, T const& rhs) {
    return !operator==(rhs, lhs);
  }

  template <
      typename T,
      typename = typename std::enable_if<
          std::is_constructible<double, T>::value>::type>
  friend bool operator<=(T const& lhs, Approx const& rhs) {
    return static_cast<double>(lhs) < rhs.m_value || lhs == rhs;
  }

  template <
      typename T,
      typename = typename std::enable_if<
          std::is_constructible<double, T>::value>::type>
  friend bool operator<=(Approx const& lhs, T const& rhs) {
    return lhs.m_value < static_cast<double>(rhs) || lhs == rhs;
  }

  template <
      typename T,
      typename = typename std::enable_if<
          std::is_constructible<double, T>::value>::type>
  friend bool operator>=(T const& lhs, Approx const& rhs) {
    return static_cast<double>(lhs) > rhs.m_value || lhs == rhs;
  }

  template <
      typename T,
      typename = typename std::enable_if<
          std::is_constructible<double, T>::value>::type>
  friend bool operator>=(Approx const& lhs, T const& rhs) {
    return lhs.m_value > static_cast<double>(rhs) || lhs == rhs;
  }

  template <
      typename T,
      typename = typename std::enable_if<
          std::is_constructible<double, T>::value>::type>
  Approx& epsilon(T const& newEpsilon) {
    double epsilonAsDouble = static_cast<double>(newEpsilon);
    setEpsilon(epsilonAsDouble);
    return *this;
  }

  template <
      typename T,
      typename = typename std::enable_if<
          std::is_constructible<double, T>::value>::type>
  Approx& margin(T const& newMargin) {
    double marginAsDouble = static_cast<double>(newMargin);
    setMargin(marginAsDouble);
    return *this;
  }

  template <
      typename T,
      typename = typename std::enable_if<
          std::is_constructible<double, T>::value>::type>
  Approx& scale(T const& newScale) {
    m_scale = static_cast<double>(newScale);
    return *this;
  }

  std::string toString() const {
    std::stringstream rss;
    rss << "Approx( " << m_value << " )";
    return rss.str();
  }

 private:
  double m_epsilon;
  double m_margin;
  double m_scale;
  double m_value;
};
} // namespace
