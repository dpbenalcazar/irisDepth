# irisDepth
This repository corresponds to the journal article:
**A 3D Iris Scanner from a Single Image using Convolutional Neural Networks.**\n
The paper is currently at revision stage at IEEE Access journal.

Authors: Daniel Benalcazar, Jorge Zambrano, Diego Bastias, Claudio Perez and Kevin Bowyer

The code in this repository is heavily based on T2Net and DenseDepth
  - T2Net by Chuanxia Zheng, Tat-Jen Cham and Jianfei Cai is available at:https://github.com/lyndonzheng/Synthetic2Realistic
  - DenseDepth by Ibraheem Alhashim and Peter Wonka is avalible at: https://github.com/ialhashim/DenseDepth

### Installing the repository
We suggest that you create two separate environments, one for pytorch, and one for tensorflow. The main architecture for training with synthetic and real iris images is based on T2Net. This network is in pytorch. In order to improve the results, we trained DenseDepth over translated images by T2Net. We call this network irisDepth, and it is in tensorflow at the moment. If you are only interested in irisDepth, you only need tensorflow.

**Dependencies for the pytorch envirnoment:**\n
You will need to install: pytorch, cuda 10, visdom, pillow, imageio, and numpy  

**Dependencies for the tensorflow envirnoment:**\n
You will need to install: tensorflow 1.13, keras, cuda 10, pillow, and numpy  

**Networks and weights:**\n
Then, you can clone this repository to your convenience. The networks and weigths, with the same folder structure as this repository, are available at:
https://drive.google.com/drive/folders/1W6yxefISGz-kx6jJcNSFgD3-Ea9pbEZU?usp=sharing

**Real and Synthetic Iris Datasets**\n
This repository contains a micro version of the datasets, with 15 real iris images and 16 synthetic images. For mini version with 60 real and 100 synthetic images, as well as the full synthetic image dataset with 72,000 images please go to:
https://drive.google.com/drive/folders/1W6yxefISGz-kx6jJcNSFgD3-Ea9pbEZU?usp=sharing

Unfortunately, we cannot publish the real iris dataset with 120 subjects and 26,520 images here. That is because we only have written consent from the volunteers to use the dataset in our paper, but we don't have consent to publish their images.  

**Virtual Iris Dataset in Blender**\n
The virtual iris dataset of 100 3D models sculpted in Blender is available at:
https://drive.google.com/drive/folders/1teohEBFo03j3kErZn1DBspypa32sOhve?usp=sharing


### Using T2Net
Once you have set up this repository with network weights and images, it is time to test it. In this research we used T2Net to predict the depth of the human iris using synthetic and real images. You can use and train our networks with the following commands. They assume you are in irisDepth-main directory.

First, you need to run visdom. Otherwise, you will be prompted to warning and error messages.
```
python -m visdom.server
```
Then, in a separate terminal window, enter the pytorch environment:
```
source activate pytorch
```

**Predicting depth from iris images:**\n
You can predict the depthmaps of all the images in a folder using:  
```
 python test_real.py --name irisT2Net --model test --batchSize 4 --img_target_file ../datasets/micro_test/Real-256x256 --results_dir micro_test1
```
You can do the same using a .txt file with the list of the images:  
```
 python test_real.py --name irisT2Net --model test --batchSize 4 --dataset_root ../datasets/ --img_target_file data/mini_test/Real-256.txt --results_dir mini_test1

```

**Translating synthetic images:**\n
You can make all the synthetic images in a folder look more photorealistic using:  
```
 python test_synthetic.py --name irisT2Net --model test --batchSize 4 --img_target_file ../datasets/micro_test/SYN-256x256 --results_dir micro_test2
```

**Training T2Net**\n
You can monitor the training using visdom by opening the browser of your choice and following the URL: http://localhost:8097.

To train T2Net for 5 iterations you can use:
```
python train.py --name t2n_micro --niter 2 --niter_decay 3 --model wsupervised --dataset_root ../datasets/  --img_source_file data/micro_test/SYN-256.txt --img_target_file data/micro_test/Real-256.txt --lab_source_file data/micro_test/DEP-256.txt --lab_target_file data/micro_test/Real-256.txt --display_freq 10 --batchSize 4
```

This is the command we used to train irisT2Net:
```
python train.py --name irisT2Net --niter 6 --niter_decay 6 --model wsupervised --dataset_root ../datasets/  --img_source_file data/iris_256x256/SYN-256_tra.txt --img_target_file data/iris_256x256/DD3-256_tra.txt --lab_source_file data/iris_256x256/DEP-256_tra.txt --lab_target_file data/iris_256x256/DD3-256_tra.txt --display_freq 100 --batchSize 4
```
