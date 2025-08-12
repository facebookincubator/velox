#!/bin/bash

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
#

# Warm Storage Location Generator
#
# This script generates warm storage URLs for trace directories based on region names
# and user-defined directory names. It validates region names against a predefined
# list of supported regions and constructs URLs following the pattern:
# ws://ws.dw.{region}0dw0/{directory_name}/
#
# Usage: get_ws_location.sh <region_name> <directory_name> [--session-cmd]
#
# Arguments:
#   region_name      Valid region identifier (case-insensitive)
#   directory_name   User-defined directory name for the trace path
#   --session-cmd    Optional flag to output as Presto session command
#
# Examples:
#   ./get_ws_location.sh pnb ericjiatest
#   ./get_ws_location.sh VLL mydirectory --session-cmd

set -e

# Valid regions across different geographic areas
VALID_REGIONS=(
    # US Regions
    "ASH" "ATN" "EAG" "FRC" "FTW" "LDC" "PRN" "PNB" "RVA" "VLL"
    # EU Regions
    "CLN" "LLA" "ODN"
    # Other/Additional Regions
    "NAO" "NCG" "NHA" "PCI" "GTN" "ZCH" "KCM" "DKL" "ZGD" "ZHU" "CCO" "ZAZ" "MAZ" "HIL" "SNB" "VCN"
)

function show_usage() {
    echo "Usage: $0 <region_name> <directory_name>"
    echo ""
    echo "Get warm storage location for a given region"
    echo ""
    echo "Arguments:"
    echo "  region_name      Region identifier (e.g., pnb, vll, ash)"
    echo "  directory_name   User defined directory name for the warm storage path"
    echo ""
    echo "Examples:"
    echo "  $0 pnb ericjiatest"
    echo "  $0 vll mydirectory"
    echo ""
    echo "Output formats:"
    echo "  --url-only     : Just the WS URL (default)"
    echo "  --session-cmd  : Complete session command"
}

function get_ws_location() {
    local region="$1"
    local directory_name="$2"
    local format="${3:-url-only}"

    # Check if both region and directory name are provided
    if [[ -z "$region" || -z "$directory_name" ]]; then
        echo "Error: Both region name and directory name are required" >&2
        show_usage
        return 1
    fi

    # Convert region to uppercase for validation
    local region_upper
    region_upper=$(echo "$region" | tr '[:lower:]' '[:upper:]')

    # Validate region
    local valid_region=false
    for valid in "${VALID_REGIONS[@]}"; do
        if [[ "$region_upper" == "$valid" ]]; then
            valid_region=true
            break
        fi
    done

    if [[ "$valid_region" == false ]]; then
        echo "Error: Invalid region '$region'" >&2
        echo "Valid regions are:" >&2
        echo "  US Regions: ASH, ATN, EAG, FRC, FTW, LDC, PRN, PNB, RVA, VLL" >&2
        echo "  EU Regions: CLN, LLA, ODN" >&2
        echo "  Other/Additional Regions: NAO, NCG, NHA, PCI, GTN, ZCH, KCM, DKL, ZGD, ZHU, CCO, ZAZ, MAZ, HIL, SNB, VCN" >&2
        return 1
    fi

    # Convert region to lowercase for URL
    local region_lower
    region_lower=$(echo "$region" | tr '[:upper:]' '[:lower:]')

    # Build the warm storage URL using pattern: ws://ws.dw.{region}0dw0/{directory_name}/
    local ws_url="ws://ws.dw.${region_lower}0dw0/${directory_name}/"

    # Output based on format
    case "$format" in
        "url-only")
            echo "$ws_url"
            ;;
        "session-cmd")
            echo "set session native_query_trace_dir='$ws_url';"
            ;;
        *)
            echo "Error: Invalid format '$format'" >&2
            return 1
            ;;
    esac
}

# Parse command line arguments
case "${1:-}" in
    "-h"|"--help"|"help"|"")
        show_usage
        exit 0
        ;;
    "--url-only")
        if [[ $# -lt 3 ]]; then
            echo "Error: Both region name and directory name are required" >&2
            show_usage
            exit 1
        fi
        get_ws_location "$2" "$3" "url-only"
        ;;
    "--session-cmd")
        if [[ $# -lt 3 ]]; then
            echo "Error: Both region name and directory name are required" >&2
            show_usage
            exit 1
        fi
        get_ws_location "$2" "$3" "session-cmd"
        ;;
    *)
        # Default behavior: treat first arg as region name, second as directory name
        if [[ $# -lt 2 ]]; then
            echo "Error: Both region name and directory name are required" >&2
            show_usage
            exit 1
        fi
        get_ws_location "$1" "$2" "url-only"
        ;;
esac
