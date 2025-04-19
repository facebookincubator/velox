#pragma once
#include <string>
#include <memory>
#include <utility>
#include <boost/functional/hash.hpp>

namespace facebook::velox::parquet {

struct KeyVersion {
  std::string name;
  std::string versionName;
  std::string material;

  KeyVersion() = default;
  KeyVersion(std::string name,
             std::string versionName,
             std::string material)
      : name(std::move(name)), versionName(std::move(versionName)), material(std::move(material)) {}

  // Equality operator
  bool operator==(const KeyVersion& other) const {
    if (name != other.name) {
      return false;
    }
    if (versionName != other.versionName) {
      return false;
    }
    if (material != other.material) {
      return false;
    }
    return true;
  }
};

struct EncryptedKeyVersion {
  std::string encryptionKeyName;
  std::string encryptionKeyVersionName;
  std::string encryptedKeyIv;
  std::shared_ptr<KeyVersion> encryptedKeyVersion;

  EncryptedKeyVersion() = default;
  EncryptedKeyVersion(std::string& encryptionKeyName,
                      std::string& encryptionKeyVersionName,
                      std::string& encryptedKeyIv,
                      std::shared_ptr<KeyVersion>& encryptedKeyVersion) :
        encryptionKeyName(encryptionKeyName),
        encryptionKeyVersionName(encryptionKeyVersionName),
        encryptedKeyIv(encryptedKeyIv),
        encryptedKeyVersion(encryptedKeyVersion) {}

  // Equality operator
  bool operator==(const EncryptedKeyVersion& other) const {
    if (encryptionKeyName != other.encryptionKeyName) {
      return false;
    }
    if (encryptionKeyVersionName != other.encryptionKeyVersionName) {
      return false;
    }
    if (encryptedKeyIv != other.encryptedKeyIv) {
      return false;
    }
    if (!encryptedKeyVersion && !other.encryptedKeyVersion) {
      return true;
    }
    if ((encryptedKeyVersion && !other.encryptedKeyVersion) || (!encryptedKeyVersion && other.encryptedKeyVersion)) {
      return false;
    }
    return *encryptedKeyVersion == *other.encryptedKeyVersion;
  }
};

struct CacheableEncryptedKeyVersion {
  std::string userName;
  std::shared_ptr<EncryptedKeyVersion> encryptedKeyVersion;

  CacheableEncryptedKeyVersion(std::string userName,
                               std::shared_ptr<EncryptedKeyVersion>& encryptedKeyVersion) :
        userName(std::move(userName)),
        encryptedKeyVersion(encryptedKeyVersion) {}

  // Equality operator
  bool operator==(const CacheableEncryptedKeyVersion& other) const {
    if (userName != other.userName) {
      return false;
    }
    if (!encryptedKeyVersion && !other.encryptedKeyVersion) {
      return true;
    }
    if ((encryptedKeyVersion && !other.encryptedKeyVersion) || (!encryptedKeyVersion && other.encryptedKeyVersion)) {
      return false;
    }
    return *encryptedKeyVersion == *other.encryptedKeyVersion;
  }

  std::size_t hash() const {
    std::size_t seed = 0;
    boost::hash_combine(seed, userName);
    if (!encryptedKeyVersion) {
      return seed;
    }
    boost::hash_combine(seed, encryptedKeyVersion->encryptionKeyName);
    boost::hash_combine(seed, encryptedKeyVersion->encryptionKeyVersionName);
    boost::hash_combine(seed, encryptedKeyVersion->encryptedKeyIv);
    if (!encryptedKeyVersion->encryptedKeyVersion) {
      return seed;
    }
    boost::hash_combine(seed, encryptedKeyVersion->encryptedKeyVersion->name);
    boost::hash_combine(seed, encryptedKeyVersion->encryptedKeyVersion->material);
    boost::hash_combine(seed, encryptedKeyVersion->encryptedKeyVersion->versionName);
    return seed;
  }
};

}
