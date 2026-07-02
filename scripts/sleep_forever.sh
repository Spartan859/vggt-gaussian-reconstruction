#!/usr/bin/env bash
set -euo pipefail

SLEEP_SECONDS="${SLEEP_SECONDS:-3600}"

echo "Sleeping forever. Set SLEEP_SECONDS to change the interval; press Ctrl+C to stop."
while true; do
    sleep "${SLEEP_SECONDS}"
done
