# irisDepth
This repository corresponds to the journal article:
**A 3D Iris Scanner from a Single Image using Convolutional Neural Networks.**
The paper has been published by the IEEE Access journal.

Authors: Daniel Benalcazar, Jorge Zambrano, Diego Bastias, Claudio Perez and Kevin Bowyer

Video Summary on Youtube: [Graphical Abstract](https://youtu.be/etUgDOl-U_w), or [Extended Version](https://youtu.be/K4b2Vw8vk64).

The code in this repository is heavily based on T2Net and DenseDepth. Please visit the original repos as well.
  - T2Net by Chuanxia Zheng, Tat-Jen Cham and Jianfei Cai is [available here](https://github.com/lyndonzheng/Synthetic2Realistic)
  - DenseDepth by Ibraheem Alhashim and Peter Wonka is [available here](https://github.com/ialhashim/DenseDepth)

### New Updates
 - Links to weights and datasets are back online.

### Installing the repository
We suggest that you create two separate environments, one for pytorch, and one for tensorflow. The main architecture for training with synthetic and real iris images is based on T2Net. This network is in pytorch. In order to improve the results, we trained DenseDepth over translated images by T2Net. We call this network irisDepth, and it is in tensorflow at the moment. If you are only interested in irisDepth, you only need tensorflow.

##### Dependencies for the pytorch envirnoment:
You will need to install: pytorch, cuda, pillow, ,numpy, visdom, jsonpatch, and dominate

Using anaconda in ubuntu:
```
conda create --name pytorch python=3.6 numpy=1.17 pillow=6
source activate pytorch
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch
conda install -c conda-forge imageio visdom jsonpatch dominate
```

##### Dependencies for the tensorflow envirnoment:
You will need to install: tensorflow, keras, cuda, numpy, opencv,  pydot, scikit-image, and imageio

Using anaconda in ubuntu:
```
conda create --name tensorflow python=3.6 numpy=1.16 imageio matplotlib
source activate tensorflow
conda install -c conda-forge opencv=3.4.2 scikit-learn scikit-image pydot
conda install -c conda-forge tensorflow-gpu=1.13 keras=2.2.4 cudatoolkit=10.0
```

##### Networks and weights:
Then, you can clone this repository to your convenience. The networks and weigths, with the same folder structure as this repository, are [available here](https://drive.google.com/file/d/1Vu-gKwu2uzMjILLnpGjFs-4T7laBjfUn/view?usp=sharing):

##### Real and synthetic iris datasets
This repository contains a micro version of the datasets, with 15 real iris images and 16 synthetic images. For mini version with 60 real and 100 synthetic images, please download it from [here](https://drive.google.com/file/d/1UPPx_jhRhpMXX7JvoM5wr31x0Et-voBW/view?usp=sharing).

For the full synthetic image dataset with 72,000 images please go [here](https://drive.google.com/drive/folders/1W3KphosklcCah34RVCw_cSpaNoI4VQbE?usp=sharing).

Unfortunately, we cannot publish the real iris dataset with 120 subjects and 26,520 images here. That is because we only have written consent from the volunteers to use the dataset in our paper, but we don't have consent to publish their images.  

##### Virtual iris dataset in blender
The virtual iris dataset of 100 3D models sculpted in Blender is available [here](https://drive.google.com/drive/folders/17xtr_ciUgWLOB5dJyo8TRvZwBaWC7IZ6?usp=sharing).


## Using T2Net
Once you have set up this repository with network weights and images, it is time to test it. In this research we used T2Net to predict the depth of the human iris using synthetic and real images. You can use and train our networks with the following commands. They assume you are in irisDepth-main directory.

First, you need to run visdom. Otherwise, you will be prompted to warning and error messages.
```
python -m visdom.server
```
Then, in a separate terminal window, enter the pytorch environment:
```
source activate pytorch
cd T2Net
```

##### Predicting depth from iris images:
You can predict the depthmaps of all the images in a folder using:  
```
 python test_real.py --name irisT2Net --model test --img_target_file ../datasets/micro_test/Real-256x256 --results_dir results/micro_test1
```
You can do the same using a .txt file with the list of the images:  
```
 python test_real.py --name irisT2Net --model test --dataset_root ../datasets/ --img_target_file data/micro_test/Real-256.txt --results_dir results/micro_test2
```

##### Translating synthetic images:
You can make all the synthetic images in a folder look more photorealistic using:  
```
 python test_synthetic.py --name irisT2Net --model test --batchSize 4 --img_target_file ../datasets/micro_test/SYN-256x256 --results_dir micro_test2
```

##### Training T2Net:
You can monitor the training using visdom by opening the browser of your choice and following the URL: http://localhost:8097.

To train T2Net for 3 iterations you can use:
```
python train.py --name t2n_micro --niter 2 --niter_decay 1 --model wsupervised --display_freq 10 --batchSize 1
```

This is the command we used to train irisT2Net:
```
python train.py --name irisT2Net --niter 6 --niter_decay 6 --model wsupervised --dataset_root ../datasets/  --img_source_file data/iris_256x256/SYN-256_tra.txt --img_target_file data/iris_256x256/DD3-256_tra.txt --lab_source_file data/iris_256x256/DEP-256_tra.txt --lab_target_file data/iris_256x256/DD3-256_tra.txt --display_freq 100 --batchSize 4
```


## Using DenseDepth and irisDepth
We obtained the best results by merging the GAN of T2Net with the depth prediction architecture of DenseDepth.

First, activate the tensorflow environment
```
source activate tensorflow
cd DenseDepth
```

##### Predicting depth from iris images:
You can predict the depthmaps of all the images in a folder using:  
```
python evalFolder.py --model models/irisDepth.h5 --inputs ../datasets/micro_test/Real-256x256  --result_dir results/irisDepth/micro_Real
```

Alternatively, you can load the image list from a .txt file:
```
python evalFolder.py --model models/irisDepth.h5 --inputs data/micro_S2R.txt --root_dir ../datasets/ --result_dir results/irisDepth/micro_S2R
```

##### Training irisDepth:
irisDepth is trained with synthetic images translated by T2Net as the input, and the ground truth depthmpas of the synthetic images as the target. To train irisDepth you need to specify a .txt with the path of those images. In each line of the file must contain the input path and the target path separated by a semicolon and a space: "; ". At the moment, the file location is hard coded in data.py.

Also, at the moment, you have to continue training from any checkpoint so that the network can save the json after finishing training. For example, here we are training irisDepth from the checkpoints of irisDepth:  
```
python train.py --data iris --gpus 1 --bs 2 --epochs 1 --checkpoint models/irisDepth.h5 --name irisTest
```


## Obtaining iris 3D models:
The iris 3D models are obtained by extruding the depth information from the RGB image. We have used Matlab to produce and analyze the iris 3D models.


The function **rgbd2mesh.m** produces the 3D model. You can use it to obtain both the point cloud and the mesh models. The mesh model is computationally expensive tough. It uses the square connectivity of the neighboring pixels in the original image.

The point cloud model is obtained using:
```
[pts, colors, normals] = rgbd2mesh(image, depthmap)
pc = pointCloud(pts, 'Color', colors, 'Normal', normals);
pcwrite(pc, 'file_name.ply');
```

The mesh model is obtained using:
```
[verts, colors, normals, faces] = rgbd2mesh(image, depthmap)
plywrite2('file_name.ply', faces, verts, colors, normals);
```

The script in **Iris_3D_model_from_depthmap.m** illustrates how to obtain the iris 3D model from one input image and a depthmap.

The script in **Iris_3D_model_from_folder.m** helps producing the iris 3D models for all the images in a folder. The corresponding depthmaps should be in a different folder with the same order. See the script for more instructions.


### Citation:
You can cite our work in IEEE format as:

D. P. Benalcazar, J. Zambrano, D. Bastias, C. A. Perez, and K. W. Bowyer, “A 3D Iris Scanner from a Single Image using Convolutional Neural Networks,” in IEEE Access, IEEE Access, vol. 8, no. 1, pp. 98584–98599, 2020, doi: 10.1109/ACCESS.2020.2996563.
