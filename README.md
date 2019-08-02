# On modifying..
- Multi GPU
- N batch
- Bugfix

## GVCNN (Group-View Convolutional Neural Networks for 3D Shape Recognition)
![](assets/gvcnn_framework.png)

## Data
- Download [modelnet10-Class Orientation-aligned Subset](http://modelnet.cs.princeton.edu/)
  - make .png images in order below
    - data_utils/make_views_dir.py
    - data_utils/off2obj.py (after 'sudo apt install openctm-tools')
    - data_utils/obj2png.py
  - Or You can create 2D dataset from 3D objects (.obj, .stl, and .off), using [BlenderPhong](https://github.com/WeiTang114/BlenderPhong).
- Directly you get dataset https://drive.google.com/file/d/0B4v2jR3WsindMUE3N2xiLVpyLW8/view from https://github.com/RBirkeland/MVCNN-PyTorch/blob/master/README.md
- Downsized modelnet40(https://drive.google.com/file/d/0B4v2jR3WsindMUE3N2xiLVpyLW8/view) to modelnet12/6-view. 

## Quick Start
- make group-view image tfrecord file
  - dataset_tools/create_modelnet_tf_record.py
- execute train.py

## Retrieval
- For training efficiency, it was implemented at the [other repository.](https://github.com/ace19-dev/mvcnn-tf) 

## Notice
- It had better use lighter model or decrease image size because it needs a big resources.
- Only 1 batch size is available now.  
- There is some bug(NAN) during training.

## References from
- http://openaccess.thecvf.com/content_cvpr_2018/papers/Feng_GVCNN_Group-View_Convolutional_CVPR_2018_paper.pdf
- https://github.com/WeiTang114/MVCNN-TensorFlow
- https://github.com/pclausen/obj2png
- https://github.com/ildoonet/tf-mobilenet-v2
- http://openresearch.ai/t/nccl-efficient-tensorflow-multigpu-training/159

