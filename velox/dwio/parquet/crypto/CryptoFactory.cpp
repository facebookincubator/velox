#include "velox/dwio/parquet/crypto/CryptoFactory.h"

namespace facebook::velox::parquet {

std::unique_ptr<CryptoFactory> CryptoFactory::instance_ = nullptr;

}
