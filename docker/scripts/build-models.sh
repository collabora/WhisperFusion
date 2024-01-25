#!/bin/bash -e

test -f /etc/shinit_v2 && source /etc/shinit_v2

./build-whisper.sh
# ./build-mistral.sh
./build-dolphin-2_6-phi-2.sh
