#pragma once

#include <string>

namespace facebook::velox::parquet {

struct KeyMetadata {
  std::string name;
  std::string iv;
  int version;
  std::string eek;

  KeyMetadata(const std::string& name,
              const std::string& iv,
              const int version,
              const std::string& eek): name(name), iv(iv), version(version), eek(eek) {}
};

class KeyMetadataAssembler {
 public:
  static KeyMetadata unAssembly(const std::string& keyMetadata);
};

}
