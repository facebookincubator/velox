#include "velox/type/custom_type/Int128.h"

using namespace facebook::velox::type;

uint128::operator int128() const { return int128(this->hi_, this->lo_); }