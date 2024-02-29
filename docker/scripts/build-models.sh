#!/bin/bash -e

test -f /etc/shinit_v2 && source /etc/shinit_v2

if [ ! -d "/root/scratch-space/models/whisper_small_en" ] || [ -z "$(ls -A /root/scratch-space/models/whisper_small_en)" ]; then
    echo "whisper_small_en directory does not exist or is empty. Running build-whisper.sh..."
    ./build-whisper.sh
else
    echo "whisper_small_en directory exists and is not empty. Skipping build-whisper.sh..."
fi
# ./build-mistral.sh
if [ ! -d "/root/scratch-space/models/dolphin-2_6-phi-2" ] || [ -z "$(ls -A /root/scratch-space/models/dolphin-2_6-phi-2)" ]; then
    echo "dolphin-2_6-phi-2 directory does not exist or is empty. Running build-dolphin-2_6-phi-2.sh..."
    ./build-dolphin-2_6-phi-2.sh
else
    echo "dolphin-2_6-phi-2 directory exists and is not empty. Skipping build-dolphin-2_6-phi-2.sh..."
fi
