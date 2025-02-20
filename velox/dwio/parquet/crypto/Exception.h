#pragma once
#include <string>
#include <sstream>

namespace facebook::velox::parquet {

class CryptoException : public std::exception {
 public:
  template <typename... Args>
  explicit CryptoException(Args&&... args) {
    std::ostringstream oss;
    (oss << ... << std::forward<Args>(args));
    msg_ = oss.str();
  }

  explicit CryptoException(std::string msg) : msg_(std::move(msg)) {}

  explicit CryptoException(const char* msg, const std::exception&) : msg_(msg) {}

  CryptoException(const CryptoException&) = default;
  CryptoException& operator=(const CryptoException&) = default;
  CryptoException(CryptoException&&) = default;
  CryptoException& operator=(CryptoException&&) = default;

  const char* what() const noexcept override { return msg_.c_str(); }

 private:
  std::string msg_;
};

}
