# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG OS_VERSION=latest
FROM ubuntu:${OS_VERSION}

# Set non-interactive mode for apt.
ENV DEBIAN_FRONTEND=noninteractive

# Install clang-format-18.
RUN apt-get update && \
    apt-get install -y clang-format-18 cmake-format

# Set the default command.
CMD ["echo", "This is the docker image of Velox4J's code formatter."]
