//
// Created by Karteek Murthy Samba Murthy on 9/5/22.
//

#include "velox/type/UnscaledShortDecimal.h"
#include "velox/type/DecimalUtil.h"

namespace std {
facebook::velox::UnscaledShortDecimal
numeric_limits<facebook::velox::UnscaledShortDecimal>::max() {
  // Returning 10^18 - 1.
  return facebook::velox::UnscaledShortDecimal(
      facebook::velox::DecimalUtil::kPowersOfTen[18] - 1);
}

facebook::velox::UnscaledShortDecimal
numeric_limits<facebook::velox::UnscaledShortDecimal>::min() {
  return facebook::velox::UnscaledShortDecimal(
      -facebook::velox::DecimalUtil::kPowersOfTen[18] - 1);
}
}; // namespace std.