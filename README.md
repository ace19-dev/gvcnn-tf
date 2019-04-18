## GVCNN (Group-View Convolutional Neural Networks for 3D Shape Recognition)
![](assets/gvcnn_framework.png)

## Data
- download 40-Class Subset from http://modelnet.cs.princeton.edu/
- make images with .py files in data_utils

## Quick Start
- prepare group-view image
- execute train.py

## Retrieval
- For training efficiency, it was implemented at the link below.
- [Here is repo for retrieval module.](https://github.com/ace19-dev/mvcnn-tf) 

## Notice
- It had better use lighter model or decrease image size because it needs a big resources.
- Currently only one batch is available.

## References from
- http://openaccess.thecvf.com/content_cvpr_2018/papers/Feng_GVCNN_Group-View_Convolutional_CVPR_2018_paper.pdf
- https://github.com/WeiTang114/MVCNN-TensorFlow
- https://github.com/pclausen/obj2png

