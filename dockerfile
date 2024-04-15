FROM jenkins/ssh-agent:jdk17
USER root
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y python3 python3-pip gcc && \
    mkdir -p /home/jenkins && \
    chown -R jenkins:jenkins /home/jenkins && \
    rm -rf /var/lib/apt/lists/*
USER jenkins