#include <memory>
#include <stdexcept>

#include <gtest/gtest.h>
#include "velox/type/custom_type/Int128.h"


namespace facebook::velox::type{


TEST(Int128, objectCreation) {
  int128 a = int128(0);

  ASSERT_EQ(a.hi(), 0);
  ASSERT_EQ(a.lo(), 0);

  int128 b = int128(23,45);

  ASSERT_EQ(a.hi(), 23);
  ASSERT_EQ(a.lo(), 45);

}

TEST(Int128, failTest) {
  ASSERT_EQ(0, 1);
}

} // namespace facebook::velox::type