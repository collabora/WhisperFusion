FROM ghcr.io/collabora/whisperfusion-base:latest as base

WORKDIR /root
COPY scripts/setup-whisperfusion.sh scripts/run-whisperfusion.sh scratch-space/models /root/
RUN ./setup-whisperfusion.sh

CMD ./run-whisperfusion.sh

