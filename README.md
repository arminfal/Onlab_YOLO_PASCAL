# YOLO Training with PASCAL VOC2012 Dataset

drive másolat, ha a githubos mappában valamelyik fájl nem található:  
In case some files are not available on GitHub, you can find them in this [Google Drive folder](https://drive.google.com/file/d/1CwBUp6yPpxWQpt_OEg-J85EE9bFEqPp3/view?usp=sharing).

## Dataset

[PASCAL VOC2012](https://www.kaggle.com/datasets/bardiaardakanian/voc0712).

## Structure//felépítés

- `JPEGImages`: Contains JPEG data for training.
- `SegmentationObject`: Contains PNG masks to create text annotations for training.
- `Annotations`: Contains the XML data.
- `IMGAnnotations`: created by the python code, contains the segmentation text files created from the xml and png masks.
- `runs`: Contains the training models.
- `model-name`: Contains the model results on a few hand-selected pictures.
- `onlab.ipynb`: Contains the code in a Jupyter notebook.
- `datproc.py`: Contains the data processing part of the notebook, needed for the jenkins pipeline.
- `training.py`: Contains the training part of the notebook, needed for the jenkins pipeline.
- `dockerfile`: Contains the docker image creation for python+pytorch, installs majority of requirements.
- `jenkinsfile`: Contains the jenkins pipeline.
- `requirements.txt`: Contains the python prerequisites.
## Training Results//Tanitás eredmények

The following images show the results of the training with different configurations:

- AdamW optimisation with custom learning rate and momentum:
  ![AdamW optimisation](runs/segment/trainAdamWopt/results.png)
- Continuing previous AdamW optimisation training with patience/early stopping=30:
  ![AdamW optimisation with early stopping](runs/segment/trainAdeamWopt+patience30/results.png)
- Auto Optimisation, default learning rate and momentum+ custom hyperparameters from tuning:
  ![Auto optimisation](runs/segment/trainautoopt/results.png)
- Default hyperparameters:
  ![Default hyperparameters](runs/segment/traindefhyppar/results.png)

The following images show the model in use (on training data, so not the best test data):

![image](https://github.com/arminfal/Onlab_YOLO_PASCAL/assets/26046723/964e2c0a-810e-4113-8c47-62285d26dca0)
![image](https://github.com/arminfal/Onlab_YOLO_PASCAL/assets/26046723/7e5f05ed-050b-49e0-9ef4-eec5a8650c92)
![image](https://github.com/arminfal/Onlab_YOLO_PASCAL/assets/26046723/e7128fc1-477a-4115-8e00-cd7bec237574)
![image](https://github.com/arminfal/Onlab_YOLO_PASCAL/assets/26046723/015a3abe-c44c-46d1-8721-3f1bfb69fb9a)
![image](https://github.com/arminfal/Onlab_YOLO_PASCAL/assets/26046723/b72e787b-71d7-4dee-88e3-53d2a9bcc91e)
![image](https://github.com/arminfal/Onlab_YOLO_PASCAL/assets/26046723/8cc5a672-e728-4ee1-a499-4a5fd7792901)
![image](https://github.com/arminfal/Onlab_YOLO_PASCAL/assets/26046723/43d2050e-5be5-405a-a97d-a11901fe8dd6)
![image](https://github.com/arminfal/Onlab_YOLO_PASCAL/assets/26046723/e0be5889-39ab-4ff2-88cd-3fa8acb321aa)
![image](https://github.com/arminfal/Onlab_YOLO_PASCAL/assets/26046723/309c9b0f-ff89-47dd-8ead-498a3be4a1e7)
![image](https://github.com/arminfal/Onlab_YOLO_PASCAL/assets/26046723/bc1266de-bf39-4926-98be-7aa0f7342d1f)
![image](https://github.com/arminfal/Onlab_YOLO_PASCAL/assets/26046723/007ec257-864a-45bf-ba46-90ec2102eb1f)
![image](https://github.com/arminfal/Onlab_YOLO_PASCAL/assets/26046723/ee6d84b0-c44b-4860-b375-550e605367a4)
![image](https://github.com/arminfal/Onlab_YOLO_PASCAL/assets/26046723/21959a88-8ef5-4b32-b967-b63d4b972667)
![image](https://github.com/arminfal/Onlab_YOLO_PASCAL/assets/26046723/9267ad65-e484-4cf6-b45e-8cbde8064512)
![image](https://github.com/arminfal/Onlab_YOLO_PASCAL/assets/26046723/d5bd7d1e-efd6-4fc5-97ba-ae3fc67b4ac4)
![image](https://github.com/arminfal/Onlab_YOLO_PASCAL/assets/26046723/36fbccd7-0681-4ca3-977e-10de9eb810b6)
![image](https://github.com/arminfal/Onlab_YOLO_PASCAL/assets/26046723/c6338639-7983-44f4-b0df-971035f20a30)
![image](https://github.com/arminfal/Onlab_YOLO_PASCAL/assets/26046723/3ab1d28d-9192-4560-b2ba-8edc75914536)
![image](https://github.com/arminfal/Onlab_YOLO_PASCAL/assets/26046723/f44cdff2-fdb9-46cd-9f93-b8e84a0b65e7)
![image](https://github.com/arminfal/Onlab_YOLO_PASCAL/assets/26046723/0b02af00-01a3-4fc4-9673-4457184bb0bc)
![image](https://github.com/arminfal/Onlab_YOLO_PASCAL/assets/26046723/043820a1-fbad-4e37-bd16-96a91a5a14ad)
![image](https://github.com/arminfal/Onlab_YOLO_PASCAL/assets/26046723/de0c04ef-ff28-49af-8356-849354e3f9bb)

---
## Jenkins  
Jenkins in docker with docker job agents  
Used this repo/tutorial as base:  
[devopsjourney1/jenkins-101](https://github.com/devopsjourney1/jenkins-101)  
[Jenkins Docker Installation Guide](https://www.jenkins.io/doc/book/installing/docker/)

**Jenkins setup:**  

Jenkins BlueOcean docker image:  

```bash
docker build -t myjenkins-blueocean:2.414.2 .
```

Network creation:  

```bash
docker network create jenkins
```

Container start:  

```bash
docker run --name jenkins-blueocean --restart=on-failure --detach \
  --network jenkins --env DOCKER_HOST=tcp://docker:2376 \
  --env DOCKER_CERT_PATH=/certs/client --env DOCKER_TLS_VERIFY=1 \
  --volume jenkins-data:/var/jenkins_home \
  --volume jenkins-docker-certs:/certs/client:ro \
  --publish 8080:8080 --publish 50000:50000 myjenkins-blueocean:2.414.2
```

Jenkins local site:  
[https://localhost:8080/](https://localhost:8080/)

Docker proxy to host machine docker:  

```bash
docker run -d --restart=always -p 127.0.0.1:2376:2375 --network jenkins -v /var/run/docker.sock:/var/run/docker.sock alpine/socat tcp-listen:2375,fork,reuseaddr unix-connect:/var/run/docker.sock
```

Jenkins Python Agent from dockerfile:  

```bash
docker pull arminfal/onlab:latest
```  
# Jenkins Settings

- **Home directory:** `/var/jenkins_home`
- **Docker Host URI (From docker image):** `tcp://172.18.0.2:2375`

## Docker Agent

- **Name:** Docker-yolo
- **Docker image:** arminfal/onlab:latest
- **Instance Capacity:** 5

## Job Settings

- Discard old builds
- Log Rotation
- **Max # of builds to keep:** 5
- **Github project:** [https://github.com/arminfal/Onlab_YOLO_PASCAL/](https://github.com/arminfal/Onlab_YOLO_PASCAL/)
- **Poll SCM:** H * * * *
  - This means that it looks at the github repo for any commit once an hour, and if there is any change then it runs a new build.

## Pipeline

- Pipeline script from SCM
- **SCM:** Git
- **Repo URL:** [https://github.com/arminfal/Onlab_YOLO_PASCAL](https://github.com/arminfal/Onlab_YOLO_PASCAL)
- **Branches to build:** */main
- **Script path:** jenkinsfile  

# Jenkinsfile Breakdown

- **Install Dependencies**: Installs necessary packages and Python dependencies from `requirements.txt`.
- **Data Processing**: Executes `datproc.py` script for data processing.
- **Training**: Executes `training.py` script for model training.
- **Build Docker Image**: Builds a docker image from the pipeline results.
- **Deploy Docker Deploy**: Tags and pushes the docker image to dockerhub(curently disabled to speed up developing).
- **Start Docker Container**: Starts the docker container, which will run on the docker server even after the job itself stops.  
The prediction uses a pretrained model, because the jenkins pipeline only runs a very short training, which has no real detection, it would need to run at least several hours on a GPU to give any results.  
After execution, it archives all artifacts in the 'runs' directory.  
# CI/CD pipeline diagram  
![image](Jenkinsdiagram.png)
# Pipeline  
![image](JenkinsPipeline.png)
# Created Website  
![image](predictexample.png)
