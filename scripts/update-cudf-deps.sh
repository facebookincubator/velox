#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 --branch <branch>
       $0 --pr <pr-number>
       $0 --commit <sha>

Options:
  --branch <branch>    Update all cudf dependencies to latest from branch
  --pr <pr-number>     Update only cudf from a specific PR
  --commit <sha>       Update all dependencies using cudf commit and compatible versions

Examples:
  $0 --branch main
  $0 --branch release/26.02
  $0 --pr 12345
  $0 --commit abc123def456
EOF
}

[[ $# -eq 0 ]] && usage && exit 1

MODE="$1"
ARG="${2:-}"
[[ -z $ARG ]] && echo "Error: $MODE requires an argument" && usage && exit 1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CMAKE_FILE="$SCRIPT_DIR/../CMake/resolve_dependency_modules/cudf.cmake"

get_commit_info() {
  local repo=$1 branch=$2
  curl -sf "https://api.github.com/repos/rapidsai/${repo}/commits/${branch}" |
    jq -r '[.sha, .commit.committer.date[:10]] | join(" ")'
}

get_commit_before_date() {
  local repo=$1 until_date=$2
  curl -sf "https://api.github.com/repos/rapidsai/${repo}/commits?sha=main&until=${until_date}&per_page=1" |
    jq -r '.[0] | [.sha, .commit.committer.date[:10]] | join(" ")'
}

get_sha256() {
  curl -sL "https://github.com/rapidsai/$1/archive/$2.tar.gz" | sha256sum | cut -d' ' -f1
}

get_version() {
  local branch=$1
  if [[ $branch =~ ^release/([0-9]+\.[0-9]+)$ ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    curl -sf "https://raw.githubusercontent.com/rapidsai/cudf/${branch}/VERSION" |
      grep -oP '^[0-9]+\.[0-9]+'
  fi
}

update_dependency() {
  local var=$1 commit=$2 date=$3 checksum=$4
  sed -i "s/# ${var} commit [a-f0-9]* from [0-9-]*/# ${var} commit ${commit:0:7} from ${date}/" "$CMAKE_FILE"
  sed -i "s/set(VELOX_${var}_COMMIT [a-f0-9]*)/set(VELOX_${var}_COMMIT ${commit})/" "$CMAKE_FILE"

  if [[ $var == "cudf" ]]; then
    sed -i "s/set(VELOX_${var}_VERSION [0-9.]* CACHE/set(VELOX_${var}_VERSION ${VERSION} CACHE/" "$CMAKE_FILE"
  else
    sed -i "s/set(VELOX_${var}_VERSION [0-9.]*)/set(VELOX_${var}_VERSION ${VERSION})/" "$CMAKE_FILE"
  fi

  awk -v var="VELOX_${var}_BUILD_SHA256_CHECKSUM" -v sum="$checksum" '
    $0 ~ var { found=1 }
    found && /^[[:space:]]*[a-f0-9]{64}[[:space:]]*$/ { sub(/[a-f0-9]{64}/, sum); found=0 }
    { print }
  ' "$CMAKE_FILE" >"${CMAKE_FILE}.tmp" && mv "${CMAKE_FILE}.tmp" "$CMAKE_FILE"
}

if [[ $MODE == "--pr" ]]; then
  echo "Fetching cuDF PR #${ARG}..."
  PR_INFO=$(curl -sf "https://api.github.com/repos/rapidsai/cudf/pulls/${ARG}")
  SHA=$(echo "$PR_INFO" | jq -r '.head.sha')
  BASE=$(echo "$PR_INFO" | jq -r '.base.ref')
  VERSION=$(get_version "$BASE")
  DATE=$(curl -sf "https://api.github.com/repos/rapidsai/cudf/commits/${SHA}" | jq -r '.commit.committer.date[:10]')

  echo "  Base: $BASE (version $VERSION)"
  echo "  Commit: ${SHA:0:7} from $DATE"
  echo "  Computing SHA256..."
  CHECKSUM=$(get_sha256 "cudf" "$SHA")
  echo "  SHA256: $CHECKSUM"
  echo

  update_dependency "cudf" "$SHA" "$DATE" "$CHECKSUM"
  echo "Done! Updated cudf to PR #${ARG}: ${SHA:0:7} ($DATE)"

elif [[ $MODE == "--commit" ]]; then
  echo "Fetching cuDF commit ${ARG:0:7}..."
  COMMIT_INFO=$(curl -sf "https://api.github.com/repos/rapidsai/cudf/commits/${ARG}")
  SHA=$(echo "$COMMIT_INFO" | jq -r '.sha')
  DATE=$(echo "$COMMIT_INFO" | jq -r '.commit.committer.date[:10]')
  TIMESTAMP=$(echo "$COMMIT_INFO" | jq -r '.commit.committer.date')
  VERSION=$(curl -sf "https://raw.githubusercontent.com/rapidsai/cudf/${SHA}/VERSION" | grep -oP '^[0-9]+\.[0-9]+')

  echo "  Commit: ${SHA:0:7} from $DATE"
  echo "  Version: $VERSION"
  echo

  declare -A COMMITS DATES CHECKSUMS
  COMMITS[cudf]=$SHA
  DATES[cudf]=$DATE

  echo "Finding compatible dependency versions (main branch commits before $TIMESTAMP)..."
  echo

  for dep in rapids_cmake rmm kvikio; do
    repo=${dep//_/-}
    echo "Fetching $repo..."
    read -r commit date < <(get_commit_before_date "$repo" "$TIMESTAMP")
    echo "  Commit: ${commit:0:7} from $date"
    echo "  Computing SHA256..."
    checksum=$(get_sha256 "$repo" "$commit")
    echo "  SHA256: $checksum"

    COMMITS[$dep]=$commit
    DATES[$dep]=$date
    CHECKSUMS[$dep]=$checksum
    echo
  done

  echo "Computing SHA256 for cudf..."
  CHECKSUMS[cudf]=$(get_sha256 "cudf" "$SHA")
  echo "  SHA256: ${CHECKSUMS[cudf]}"
  echo

  echo "Updating $CMAKE_FILE..."
  for dep in rapids_cmake rmm kvikio cudf; do
    update_dependency "$dep" "${COMMITS[$dep]}" "${DATES[$dep]}" "${CHECKSUMS[$dep]}"
  done

  echo "Done! Updated dependencies:"
  for dep in rapids_cmake rmm kvikio cudf; do
    echo "  $dep: ${COMMITS[$dep]:0:7} (${DATES[$dep]})"
  done

elif [[ $MODE == "--branch" ]]; then
  VERSION=$(get_version "$ARG")
  echo "Updating cuDF dependencies from branch $ARG (version $VERSION)"
  echo

  declare -A COMMITS DATES CHECKSUMS

  for dep in rapids_cmake rmm kvikio cudf; do
    repo=${dep//_/-}
    echo "Fetching $repo..."

    read -r commit date < <(get_commit_info "$repo" "$ARG")
    echo "  Commit: ${commit:0:7} from $date"
    echo "  Computing SHA256..."
    checksum=$(get_sha256 "$repo" "$commit")
    echo "  SHA256: $checksum"

    COMMITS[$dep]=$commit
    DATES[$dep]=$date
    CHECKSUMS[$dep]=$checksum
    echo
  done

  echo "Updating $CMAKE_FILE..."
  for dep in rapids_cmake rmm kvikio cudf; do
    update_dependency "$dep" "${COMMITS[$dep]}" "${DATES[$dep]}" "${CHECKSUMS[$dep]}"
  done

  echo "Done! Updated dependencies:"
  for dep in rapids_cmake rmm kvikio cudf; do
    echo "  $dep: ${COMMITS[$dep]:0:7} (${DATES[$dep]})"
  done
else
  usage
  exit 1
fi
