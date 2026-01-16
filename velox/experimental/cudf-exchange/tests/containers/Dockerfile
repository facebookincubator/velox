# Use a Linux base image

FROM 471112500371.dkr.ecr.us-west-2.amazonaws.com/presto/prestissimo-dependency-centos9:ucx1.19.0-latest-exchange

RUN dnf -y install zsh \
    emacs-nox

# Install CMake 31.31.9 from Kitware
WORKDIR /usr/local/share
RUN wget https://github.com/Kitware/CMake/releases/download/v3.31.9/cmake-3.31.9-linux-x86_64.tar.gz && \
    tar fvx cmake-3.31.9-linux-x86_64.tar.gz

RUN ln -fs /usr/local/share/cmake-3.31.9-linux-x86_64/bin/cmake /usr/local/bin && \
    ln -fs /usr/local/share/cmake-3.31.9-linux-x86_64/bin/ctest /usr/local/bin && \
    ln -fs /usr/local/share/cmake-3.31.9-linux-x86_64/bin/cmake-gui /usr/local/bin && \
    ln -fs /usr/local/share/cmake-3.31.9-linux-x86_64/bin/cpack /usr/local/bin && \
    ln -fs /usr/local/share/cmake-3.31.9-linux-x86_64/bin/ccmake /usr/local/bin

WORKDIR /workspace

RUN git clone https://github.com/dan13bauer/velox.git

WORKDIR /workspace/velox

RUN git switch comms_unit_test

RUN EXTRA_CMAKE_FLAGS='-DTHRUST_IGNORE_CUB_VERSION_CHECK="ON" -DVELOX_BUILD_PYTHON_PACKAGE="OFF" -DVELOX_BUILD_TESTING="ON" -DVELOX_BUILD_TEST_UTILS="ON"' \
    CUDA_ARCHITECTURES=80 CUDA_COMPILER=/usr/local/cuda/bin/nvcc make cmake-cudf

RUN cmake --build _build/release -j --target=exchange_tests

# Default command
CMD ["/bin/bash"]
