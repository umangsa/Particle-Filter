FROM ubuntu:16.04
COPY install-ubuntu.sh /
RUN /install-ubuntu.sh
