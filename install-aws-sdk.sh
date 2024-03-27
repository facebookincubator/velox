#!/bin/bash
if [ ! -d "aws-sdk-cpp" ]; then
    git clone https://github.com/aws/aws-sdk-cpp --recurse-submodules
fi
cd aws-sdk-cpp

mkdir -p build
cd build

cmake ../ \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_PREFIX_PATH=/usr/local \
  -DCMAKE_INSTALL_PREFIX=/usr/local \
  -DBUILD_ONLY="s3;sts;cognito-identity;identity-management" \
  -DENABLE_TESTING=OFF
cmake --build . --config=Debug
cmake --install . --config=Debug

#cmake ../ -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH=/usr/local -DCMAKE_INSTALL_PREFIX=/usr/local
#make
#sudo make install
