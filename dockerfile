FROM jenkins/ssh-agent:jdk17
USER root
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y python3 python3-pip python3-venv gcc && \
    mkdir -p /home/jenkins && \
    chown -R jenkins:jenkins /home/jenkins && \
    rm -rf /var/lib/apt/lists/* && \
    python3 -m venv /home/jenkins/venv && \
    /home/jenkins/venv/bin/pip install --upgrade pip && \
    /home/jenkins/venv/bin/pip install pillow==10.2.0 numpy==1.26.2 scipy==1.11.4 scikit-image scikit-learn==1.4.1.post1 opencv-python==4.8.1.78 ultralytics