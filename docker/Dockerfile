FROM ghcr.io/collabora/whisperbot-base:latest as base

WORKDIR /root
COPY scripts/setup-whisperbot.sh scripts/run-whisperbot.sh scratch-space/models /root/
RUN ./setup-whisperbot.sh

CMD ./run-whisperbot.sh

