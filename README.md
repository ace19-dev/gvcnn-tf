## GVCNN (Group-View Convolutional Neural Networks for 3D Shape Recognition)
![](assets/gvcnn_framework.png)

## Data
- download 40-Class Subset from http://modelnet.cs.princeton.edu/
- make .png images in order below
  - data_utils/make_views_dir.py
  - data_utils/off2obj.py
  - data_utils/obj2png.py

## Quick Start
- make group-view image tfrecord file
  - dataset_tools/create_modelnet_tf_record.py
- execute train.py

## Retrieval
- For training efficiency, it was implemented at the [other repository.](https://github.com/ace19-dev/mvcnn-tf) 

## Notice
- It had better use lighter model or decrease image size because it needs a big resources.
- Currently only batch size 1 is available.

## References from
- http://openaccess.thecvf.com/content_cvpr_2018/papers/Feng_GVCNN_Group-View_Convolutional_CVPR_2018_paper.pdf
- https://github.com/WeiTang114/MVCNN-TensorFlow
- https://github.com/pclausen/obj2png

