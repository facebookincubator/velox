#!/usr/bin/env bash
# Copyright (c) Facebook, Inc. and its affiliates.
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

Environment Variables:
  GH_TOKEN         GitHub personal access token for higher API rate limits (optional)

Examples:
  $0 --branch main
  $0 --branch release/26.02
  $0 --pr 12345
  $0 --commit abc123def456
  GH_TOKEN=ghp_xxx $0 --branch main
EOF
}

[[ $# -eq 0 ]] && usage && exit 1

MODE="$1"
ARG="${2:-}"
[[ -z $ARG ]] && echo "Error: $MODE requires an argument" && usage && exit 1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CMAKE_FILE="$SCRIPT_DIR/../CMake/resolve_dependency_modules/cudf.cmake"

# Support GitHub token for higher rate limits
GH_TOKEN="${GH_TOKEN:-}"
CURL_AUTH_OPTS=""
if [[ -n $GH_TOKEN ]]; then
  CURL_AUTH_OPTS="-H \"Authorization: token $GH_TOKEN\""
fi

get_commit_info() {
  local repo=$1 branch=$2
  local response
  local curl_opts="--max-time 10 -w %{http_code}"
  
  if [[ -n $GH_TOKEN ]]; then
    curl_opts="$curl_opts -H \"Authorization: token $GH_TOKEN\""
  fi
  
  response=$(timeout 15 curl -s $curl_opts "https://api.github.com/repos/rapidsai/${repo}/commits/${branch}" 2>&1)
  local status=$?
  
  if [[ $status -eq 124 ]]; then
    echo "Error: Request timed out fetching $repo from branch $branch" >&2
    return 1
  fi
  
  if [[ $status -ne 0 ]]; then
    echo "Error: Failed to fetch commit info for $repo from branch $branch (exit code: $status)" >&2
    return 1
  fi
  
  # Extract HTTP code from end of response
  local http_code="${response: -3}"
  local json_body="${response%???}"
  
  if [[ $http_code != "200" ]]; then
    echo "Error: GitHub API returned HTTP $http_code for $repo/$branch" >&2
    if [[ $http_code == "403" ]]; then
      echo "" >&2
      echo "You've exceeded GitHub's API rate limit (60 requests/hour without authentication)." >&2
      echo "" >&2
      echo "SOLUTION: Authenticate with a GitHub token to get 5,000 requests/hour:" >&2
      echo "" >&2
      echo "  export GH_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" >&2
      echo "  ./scripts/update-cudf-deps.sh --branch main" >&2
      echo "" >&2
      echo "Get a token at: https://github.com/settings/tokens/new" >&2
      echo "(Select scope: public_repo)" >&2
    fi
    return 1
  fi
  
  echo "$json_body" | jq -r '[.sha, .commit.committer.date[:10]] | join(" ")' 2>/dev/null || {
    echo "Error: Failed to parse JSON response from $repo" >&2
    return 1
  }
}

get_commit_before_date() {
  local repo=$1 until_date=$2
  local response
  local curl_opts="-sf --max-time 30"
  
  if [[ -n $GH_TOKEN ]]; then
    curl_opts="$curl_opts -H \"Authorization: token $GH_TOKEN\""
  fi
  
  response=$(curl $curl_opts "https://api.github.com/repos/rapidsai/${repo}/commits?sha=main&until=${until_date}&per_page=1" 2>&1)
  local status=$?
  
  if [[ $status -ne 0 ]]; then
    echo "Error: Failed to fetch commits for $repo before $until_date" >&2
    echo "  Curl exit code: $status" >&2
    echo "  Response: $response" >&2
    return 1
  fi
  
  echo "$response" | jq -r '.[0] | [.sha, .commit.committer.date[:10]] | join(" ")' 2>/dev/null || {
    echo "Error: Failed to parse response for $repo before $until_date" >&2
    echo "  Response was: $response" >&2
    return 1
  }
}

get_sha256() {
  curl -sL --max-time 30 "https://github.com/rapidsai/$1/archive/$2.tar.gz" | sha256sum | cut -d' ' -f1 || {
    echo "Error: Failed to compute SHA256 for $1:$2" >&2
    return 1
  }
}

get_version() {
  local branch=$1
  if [[ $branch =~ ^release/([0-9]+\.[0-9]+)$ ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    local response
    local curl_opts="-sf --max-time 30"
    
    if [[ -n $GH_TOKEN ]]; then
      curl_opts="$curl_opts -H \"Authorization: token $GH_TOKEN\""
    fi
    
    response=$(curl $curl_opts "https://raw.githubusercontent.com/rapidsai/cudf/${branch}/VERSION" 2>&1)
    local status=$?
    
    if [[ $status -ne 0 ]]; then
      echo "Error: Failed to fetch version from branch $branch" >&2
      echo "  Curl exit code: $status" >&2
      echo "  Response: $response" >&2
      return 1
    fi
    
    echo "$response" | grep -oP '^[0-9]+\.[0-9]+' || {
      echo "Error: Failed to parse VERSION file from branch $branch" >&2
      echo "  Response was: $response" >&2
      return 1
    }
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
  PR_INFO=$(curl -sf --max-time 30 "https://api.github.com/repos/rapidsai/cudf/pulls/${ARG}") || {
    echo "Error: Failed to fetch PR #${ARG}" >&2
    exit 1
  }
  SHA=$(echo "$PR_INFO" | jq -r '.head.sha')
  BASE=$(echo "$PR_INFO" | jq -r '.base.ref')
  VERSION=$(get_version "$BASE") || exit 1
  DATE=$(curl -sf --max-time 30 "https://api.github.com/repos/rapidsai/cudf/commits/${SHA}" | jq -r '.commit.committer.date[:10]') || {
    echo "Error: Failed to fetch commit date for $SHA" >&2
    exit 1
  }

  echo "  Base: $BASE (version $VERSION)"
  echo "  Commit: ${SHA:0:7} from $DATE"
  echo "  Computing SHA256..."
  CHECKSUM=$(get_sha256 "cudf" "$SHA") || exit 1
  echo "  SHA256: $CHECKSUM"
  echo

  update_dependency "cudf" "$SHA" "$DATE" "$CHECKSUM"
  echo "Done! Updated cudf to PR #${ARG}: ${SHA:0:7} ($DATE)"

elif [[ $MODE == "--commit" ]]; then
  echo "Fetching cuDF commit ${ARG:0:7}..."
  COMMIT_INFO=$(curl -sf --max-time 30 "https://api.github.com/repos/rapidsai/cudf/commits/${ARG}") || {
    echo "Error: Failed to fetch commit $ARG" >&2
    exit 1
  }
  SHA=$(echo "$COMMIT_INFO" | jq -r '.sha')
  DATE=$(echo "$COMMIT_INFO" | jq -r '.commit.committer.date[:10]')
  TIMESTAMP=$(echo "$COMMIT_INFO" | jq -r '.commit.committer.date')
  VERSION=$(curl -sf --max-time 30 "https://raw.githubusercontent.com/rapidsai/cudf/${SHA}/VERSION" | grep -oP '^[0-9]+\.[0-9]+') || {
    echo "Error: Failed to fetch VERSION file for commit $SHA" >&2
    exit 1
  }

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
    read -r commit date < <(get_commit_before_date "$repo" "$TIMESTAMP") || exit 1
    echo "  Commit: ${commit:0:7} from $date"
    echo "  Computing SHA256..."
    checksum=$(get_sha256 "$repo" "$commit") || exit 1
    echo "  SHA256: $checksum"

    COMMITS[$dep]=$commit
    DATES[$dep]=$date
    CHECKSUMS[$dep]=$checksum
    echo
  done

  echo "Computing SHA256 for cudf..."
  CHECKSUMS[cudf]=$(get_sha256 "cudf" "$SHA") || exit 1
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
  VERSION=$(get_version "$ARG") || exit 1
  echo "Updating cuDF dependencies from branch $ARG (version $VERSION)"
  echo

  declare -A COMMITS DATES CHECKSUMS

  for dep in rapids_cmake rmm kvikio cudf; do
    repo=${dep//_/-}
    echo "Fetching $repo..."

    read -r commit date < <(get_commit_info "$repo" "$ARG") || exit 1
    echo "  Commit: ${commit:0:7} from $date"
    echo "  Computing SHA256..."
    checksum=$(get_sha256 "$repo" "$commit") || exit 1
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
