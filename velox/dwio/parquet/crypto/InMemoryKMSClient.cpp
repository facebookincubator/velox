#include "InMemoryKMSClient.h"
#include "Exception.h"

namespace facebook::velox::parquet {

std::string InMemoryKMSClient::getKey(const std::string& keyMetadata, const std::string& doAs) {
  auto it = keyMap_.find(keyMetadata);
  if (it != keyMap_.end()) {
    return it->second;
  }
  throw CryptoException("[CLAC] http status code 403");
}

void InMemoryKMSClient::putKey(const std::string& keyMetadata, const std::string& key) {
  keyMap_[keyMetadata] = key;
}

}
