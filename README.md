## GVCNN (Group-View Convolutional Neural Networks for 3D Shape Recognition)
- It's under development.
![](assets/gvcnn_framework.png)

## Data
- download 40-Class Subset from http://modelnet.cs.princeton.edu/
- make images with .py files in data_utils

## Quick Start
- prepare group-view image
- execute train.py

## Evaluation
-

## TODO
- refine Grouping Module (Intra-Group view_pooling, Group Fusion, ...)
- validate result and fix modules.
- re-check data input module about memory usage
- test various base architecture.
- check whether pre-trained file is using well or not.
- and ?

## References from
- http://openaccess.thecvf.com/content_cvpr_2018/papers/Feng_GVCNN_Group-View_Convolutional_CVPR_2018_paper.pdf
- https://github.com/WeiTang114/MVCNN-TensorFlow
- https://github.com/pclausen/obj2png
