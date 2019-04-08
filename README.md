
## In progress.

## GVCNN (Group-View Convolutional Neural Networks for 3D Shape Recognition)
![](assets/gvcnn_framework.png)

## Data
- download 40-Class Subset from http://modelnet.cs.princeton.edu/
- make images with .py files in data_utils

## Quick Start
- prepare group-view image
- execute train.py

## Evaluation
- classification
- retrieval
    - Euclidean distance
    - low-rank Mahalanobis metric
    - and ...
    
## TODO
- validate training/test
- create retrieval modules
- apply n-batch later

## Notice
- it had better use lighter model because it needs a big resources.

## References from
- http://openaccess.thecvf.com/content_cvpr_2018/papers/Feng_GVCNN_Group-View_Convolutional_CVPR_2018_paper.pdf
- https://github.com/WeiTang114/MVCNN-TensorFlow
- https://github.com/pclausen/obj2png

