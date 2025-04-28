#include "Config.h"

namespace velox4j {
using namespace facebook::velox;
config::ConfigBase::Entry<Preset> VELOX4J_INIT_PRESET(
    "velox4j.init.preset",
    Preset::SPARK);
} // namespace velox4j
