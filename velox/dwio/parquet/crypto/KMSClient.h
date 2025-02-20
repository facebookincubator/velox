#pragma once

#include <memory>
#include <string>
#include "velox/dwio/parquet/crypto/Cache.h"
#include "velox/dwio/parquet/crypto/KeyRetriever.h"
#include "EncryptionKey.h"

namespace facebook::velox::parquet {

constexpr int kCacheCleanupInternalSeconds = 60;

// This KMS client is a reference implementation. You may inherit it and
// implement the decryptKey function to call your KMS to decrypt the key or
// create a new KMS Client by implementation the interface DecryptionKeyRetriever directly.
class KMSClient : public DecryptionKeyRetriever {
public:
  std::string getKey(const std::string& keyMetadata, const std::string& doAs) override;

protected:
  virtual std::shared_ptr<EncryptedKeyVersion> parseKeyMetadata(const std::string& keyMetadata);
  virtual KeyVersion decryptKey(std::shared_ptr<EncryptedKeyVersion>& encryptedKeyVersion, const std::string& doAs) = 0;

private:
  Cache cache_{kCacheCleanupInternalSeconds};
  Cache exceptionCache_{kCacheCleanupInternalSeconds};
};

}
