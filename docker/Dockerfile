FROM whisperfusion-base:latest as base

RUN apt-get update && apt-get install -y xz-utils

WORKDIR /root/
COPY scripts/setup-whisperfusion.sh /root/
RUN ./setup-whisperfusion.sh

ENTRYPOINT ["/root/scratch-space/run-whisperfusion.sh"]

