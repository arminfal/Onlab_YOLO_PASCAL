YOLO training with PASCAL VOC2012 dataset.  
the folder in drive in case some files are not available on github:  
drive másolat, ha a githubos mappában valamelyik fájl nem található:  
https://drive.google.com/file/d/1CwBUp6yPpxWQpt_OEg-J85EE9bFEqPp3/view?usp=sharing  
  
Dataset:  
https://www.kaggle.com/datasets/bardiaardakanian/voc0712  

structure//felépítés:  

dataset -> yaml file for training + place for the training files generated by code  

runs -> training models  

model-name -> model results on a few handselected pictures  

onlab.ipynb -> Code in jupyternotebook  

Training results:  
AdamW optimisaton with custom learning rate  and momentum:  
![trainAdamWopt](runs/segment/trainAdamWopt/results.png)  
Continuing previous AdamW optimisation training with patience/earlystopping=30:  
![trainAdamWopt+earlyStopping](runs/segment/trainAdeamWopt+patience30/results.png)  
Auto Optimisation, default learning rate and momentum+ custom hyperparameters from tuning:  
![trainautoopt](runs/segment/trainautoopt/results.png)  
Default hyperparameters:
![traindefhyppar](runs/segment/traindefhyppar/results.png)
Model in use(on training data, so not the best test data):
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
