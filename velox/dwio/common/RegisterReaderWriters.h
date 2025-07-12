//
// Created by Ying Su on 7/4/25.
//
#pragma once

namespace facebook::velox::dwio::common {

void registerReaderFactories();
void registerWriterFactories();

void unregisterReaderFactories();
void unregisterWriterFactories();
}
