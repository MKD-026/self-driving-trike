#!/usr/bin/env bash
# Download the Piper voice model used by tts.py into ./voices/.
# The model (~60 MB) is intentionally NOT committed to git.
#
# Usage:   ./download_voice.sh            # default voice (en_US-lessac-medium)
#          ./download_voice.sh en_US-amy-medium
set -euo pipefail

VOICE="${1:-en_US-lessac-medium}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$HERE/voices"
cd "$HERE/voices"

echo "Downloading Piper voice: $VOICE -> $HERE/voices"
python -m piper.download_voices "$VOICE"
echo "Done. Files:"
ls -lh "$HERE/voices"
