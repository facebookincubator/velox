#pragma once

#include <map>
#include <string>
#include "velox/dwio/parquet/crypto/KeyRetriever.h"

namespace facebook::velox::parquet {

class InMemoryKMSClient : public DecryptionKeyRetriever {
 public:
  InMemoryKMSClient() = default;

  std::string getKey(const std::string& keyMetadata, const std::string& doAs) override;
  void putKey(const std::string& keyMetadata, const std::string& key);

 private:
  std::map<std::string, std::string> keyMap_;
};

}
