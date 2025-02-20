#pragma once
#include <iostream>
#include <functional>
#include <chrono>
#include <thread>

namespace facebook::velox::parquet {

template <typename Func>
void retry(int attempts, std::chrono::milliseconds delay, Func func) {
  for (int i = 0; i < attempts; ++i) {
    try {
      func();
      return; // Success
    } catch (const std::exception& e) {
      if (i < attempts - 1) {
        std::this_thread::sleep_for(delay); // Wait before retrying
      } else {
        throw e;
      }
    }
  }
}

}
