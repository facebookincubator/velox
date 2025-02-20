#pragma once

#include <string>

namespace facebook::velox::parquet {

class DecryptionKeyRetriever {
 public:
  virtual std::string getKey(const std::string& keyMetadata, const std::string& doAs) = 0;
  virtual ~DecryptionKeyRetriever() = default;
};

}
