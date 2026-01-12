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

PR_NUMBER=""
BRANCH=""

usage() {
  echo "Usage: $0 --branch <branch>"
  echo "       $0 --pr <pr-number>"
  echo
  echo "Options:"
  echo "  --branch <branch>    Update all cudf dependencies (rapids-cmake, rmm, kvikio, cudf)"
  echo "                       to the latest commits from the specified branch."
  echo "  --pr <pr-number>     Update only cudf to the head commit of the specified PR."
  echo "                       Other dependencies (rapids-cmake, rmm, kvikio) are unchanged."
  echo
  echo "Examples:"
  echo "  $0 --branch main"
  echo "  $0 --branch release/26.02"
  echo "  $0 --pr 12345"
}

if [[ ${1:-} == "--pr" ]]; then
  PR_NUMBER="${2:-}"
  if [[ -z $PR_NUMBER ]]; then
    echo "Error: --pr requires a PR number"
    usage
    exit 1
  fi
elif [[ ${1:-} == "--branch" ]]; then
  BRANCH="${2:-}"
  if [[ -z $BRANCH ]]; then
    echo "Error: --branch requires a branch name"
    usage
    exit 1
  fi
else
  usage
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CMAKE_FILE="$SCRIPT_DIR/../CMake/resolve_dependency_modules/cudf.cmake"

if [[ ! -f $CMAKE_FILE ]]; then
  echo "Error: Cannot find $CMAKE_FILE"
  exit 1
fi

get_commit_info() {
  local repo="$1"
  local branch="$2"
  local response
  response=$(curl -sf "https://api.github.com/repos/rapidsai/${repo}/commits/${branch}")
  if [[ $? -ne 0 ]]; then
    echo "Error: Failed to fetch commit info for $repo branch $branch" >&2
    return 1
  fi
  local sha date
  sha=$(echo "$response" | jq -r '.sha')
  date=$(echo "$response" | jq -r '.commit.committer.date[:10]')
  echo "$sha $date"
}

get_sha256() {
  local repo="$1"
  local commit="$2"
  curl -sL "https://github.com/rapidsai/${repo}/archive/${commit}.tar.gz" | sha256sum | cut -d' ' -f1
}

get_pr_info() {
  local pr_number="$1"
  local response
  response=$(curl -sf "https://api.github.com/repos/rapidsai/cudf/pulls/${pr_number}")
  if [[ $? -ne 0 ]]; then
    echo "Error: Failed to fetch PR info for cudf PR #${pr_number}" >&2
    return 1
  fi
  local sha base_ref
  sha=$(echo "$response" | jq -r '.head.sha')
  base_ref=$(echo "$response" | jq -r '.base.ref')
  echo "$sha $base_ref"
}

# Handle PR mode vs branch mode
if [[ -n $PR_NUMBER ]]; then
  echo "Fetching cuDF PR #${PR_NUMBER}..."
  read -r PR_SHA PR_BASE_REF <<<"$(get_pr_info "$PR_NUMBER")"
  if [[ -z $PR_SHA || $PR_SHA == "null" ]]; then
    echo "Error: Failed to get commit for cudf PR #${PR_NUMBER}"
    exit 1
  fi

  # Get version from the PR's base branch
  if [[ $PR_BASE_REF =~ ^release/([0-9]+\.[0-9]+)$ ]]; then
    VERSION="${BASH_REMATCH[1]}"
  else
    VERSION=$(curl -sf "https://raw.githubusercontent.com/rapidsai/cudf/${PR_BASE_REF}/VERSION" |
      grep -oP '^[0-9]+\.[0-9]+' | head -1)
    if [[ -z $VERSION ]]; then
      echo "Error: Could not determine version from base branch $PR_BASE_REF"
      exit 1
    fi
  fi

  # Get commit date
  PR_DATE=$(curl -sf "https://api.github.com/repos/rapidsai/cudf/commits/${PR_SHA}" |
    jq -r '.commit.committer.date[:10]')

  echo "  PR base: $PR_BASE_REF (version $VERSION)"
  echo "  Commit: ${PR_SHA:0:7} from $PR_DATE"
  echo "  Computing SHA256..."
  PR_CHECKSUM=$(get_sha256 "cudf" "$PR_SHA")
  echo "  SHA256: $PR_CHECKSUM"
  echo
  echo "Updating cuDF..."
else
  # Derive version from branch name (e.g., "release/26.02" -> "26.02") or VERSION file
  if [[ $BRANCH =~ ^release/([0-9]+\.[0-9]+)$ ]]; then
    VERSION="${BASH_REMATCH[1]}"
  else
    # For main or other branches, extract version from cudf's VERSION file
    VERSION=$(curl -sf "https://raw.githubusercontent.com/rapidsai/cudf/${BRANCH}/VERSION" |
      grep -oP '^[0-9]+\.[0-9]+' | head -1)
    if [[ -z $VERSION ]]; then
      echo "Error: Could not determine version from branch $BRANCH"
      exit 1
    fi
  fi

  echo "Updating cuDF dependencies to branch: $BRANCH (version $VERSION)"
  echo
fi

declare -A COMMITS
declare -A DATES
declare -A CHECKSUMS

update_dependency() {
  local var_name="$1"
  local commit="$2"
  local date="$3"
  local checksum="$4"
  local short_commit="${commit:0:7}"

  # Update comment
  sed -i "s/# ${var_name} commit [a-f0-9]* from [0-9-]*/# ${var_name} commit ${short_commit} from ${date}/" "$CMAKE_FILE"

  # Update version
  if [[ $var_name == "cudf" ]]; then
    sed -i "s/set(VELOX_${var_name}_VERSION [0-9.]* CACHE/set(VELOX_${var_name}_VERSION ${VERSION} CACHE/" "$CMAKE_FILE"
  else
    sed -i "s/set(VELOX_${var_name}_VERSION [0-9.]*)/set(VELOX_${var_name}_VERSION ${VERSION})/" "$CMAKE_FILE"
  fi

  # Update commit
  sed -i "s/set(VELOX_${var_name}_COMMIT [a-f0-9]*)/set(VELOX_${var_name}_COMMIT ${commit})/" "$CMAKE_FILE"

  # Update checksum (match 64 hex chars on a line by themselves, after the checksum variable)
  # Use awk for multiline matching
  awk -v var="VELOX_${var_name}_BUILD_SHA256_CHECKSUM" -v new_checksum="$checksum" '
        $0 ~ var { found=1 }
        found && /^[[:space:]]*[a-f0-9]{64}[[:space:]]*$/ {
            sub(/[a-f0-9]{64}/, new_checksum)
            found=0
        }
        { print }
    ' "$CMAKE_FILE" >"${CMAKE_FILE}.tmp" && mv "${CMAKE_FILE}.tmp" "$CMAKE_FILE"
}

if [[ -n $PR_NUMBER ]]; then
  # PR mode: only update cudf
  echo "Updating $CMAKE_FILE..."
  update_dependency "cudf" "$PR_SHA" "$PR_DATE" "$PR_CHECKSUM"
  echo "Done! Updated cudf to PR #${PR_NUMBER}:"
  echo "  cudf: ${PR_SHA:0:7} (${PR_DATE})"
else
  # Branch mode: update all dependencies
  declare -a VAR_NAMES=(rapids_cmake rmm kvikio cudf)
  declare -A REPOS=(
    ["rapids_cmake"]="rapids-cmake"
    ["rmm"]="rmm"
    ["kvikio"]="kvikio"
    ["cudf"]="cudf"
  )

  for var_name in "${VAR_NAMES[@]}"; do
    repo="${REPOS[$var_name]}"
    echo "Fetching $repo..."

    read -r commit date <<<"$(get_commit_info "$repo" "$BRANCH")"
    if [[ -z $commit || $commit == "null" ]]; then
      echo "Error: Failed to get commit for $repo"
      exit 1
    fi

    echo "  Commit: ${commit:0:7} from $date"
    echo "  Computing SHA256..."
    checksum=$(get_sha256 "$repo" "$commit")
    echo "  SHA256: $checksum"

    COMMITS[$var_name]="$commit"
    DATES[$var_name]="$date"
    CHECKSUMS[$var_name]="$checksum"
    echo
  done

  echo "Updating $CMAKE_FILE..."

  for var_name in "${VAR_NAMES[@]}"; do
    update_dependency "$var_name" "${COMMITS[$var_name]}" "${DATES[$var_name]}" "${CHECKSUMS[$var_name]}"
  done

  echo "Done! Updated dependencies to:"
  for var_name in "${VAR_NAMES[@]}"; do
    echo "  $var_name: ${COMMITS[$var_name]:0:7} (${DATES[$var_name]})"
  done
fi
