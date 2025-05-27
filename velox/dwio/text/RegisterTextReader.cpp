/// EQUIVALENT TO fb_velox/text/reader/TextReader.cpp

#include "RegisterTextReader.h"

using namespace facebook::velox::dwio::common;
using namespace facebook::velox::common;

namespace facebook::velox::text {

/// TODO: dummy implementation
std::unique_ptr<Reader> TextReaderFactory::createReader(
    std::unique_ptr<BufferedInput> input,
    const velox::dwio::common::ReaderOptions& options) {
  return nullptr;
}

void registerTextReaderFactory() {
  velox::dwio::common::registerReaderFactory(
      std::make_shared<TextReaderFactory>());
}

void unregisterTextReaderFactory() {
  velox::dwio::common::unregisterReaderFactory(
      velox::dwio::common::FileFormat::TEXT);
}

} // namespace facebook::velox::text
