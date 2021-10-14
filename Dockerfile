FROM debian:bullseye-slim AS base

# this weird command is to make installing the jre work correctly
RUN mkdir -p /usr/share/man/man1

# now install our dependencies for the build container and the final container
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -q update && apt-get -y upgrade && \
    apt-get install -y --no-install-recommends python3 python3-venv python3-pip python3-wheel && \
    apt-get install -y --no-install-recommends openjdk-11-jre && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Add a new user, uwcip, who does not have admin privileges in this container
RUN useradd -u 1000 -g 100 -d /home/uwcip --create-home uwcip
RUN mkdir -p /home/uwcip/app && chown 1000:100 /home/uwcip/app
RUN mkdir -p /temp_data && chown 1000:100 /temp_data
ENV PATH="/home/uwcip/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

FROM base AS builder

# install jdk (as root!) so that we can build java applications
RUN apt-get -q  update && \
    apt-get install -y --no-install-recommends openjdk-11-jdk && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

USER uwcip

# build java program
RUN mkdir -p /home/uwcip/src
WORKDIR /home/uwcip/src
COPY gephi-toolkit-0.9.2-all.jar /home/uwcip/src/
COPY src /home/uwcip/src
RUN cd /home/uwcip/src && \
	javac -classpath .:gephi-toolkit-0.9.2-all.jar GephiInterface.java ClassCount.java CliParams.java  && \
    cd /

# install python dependencies
WORKDIR /home/uwcip/app
COPY requirements.txt /home/uwcip/app/requirements.txt
RUN python3 -m venv /home/uwcip/venv && \
    . /home/uwcip/venv/bin/activate && \
    pip install -r /home/uwcip/app/requirements.txt
COPY seg_graph /home/uwcip/app/seg_graph

FROM base AS final

USER root
COPY --from=builder /home/uwcip/venv /home/uwcip/venv
COPY --from=builder /home/uwcip/app /home/uwcip/app
COPY --from=builder /home/uwcip/src/ /home/uwcip/src/

COPY /entrypoint /entrypoint
RUN chmod +x /entrypoint

USER uwcip
ENTRYPOINT ["/entrypoint"]