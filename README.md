<center> 

# Exploit CAM by itself: Complementary Learning System for Weakly Supervised Semantic Segmentation 

</center>

<center>

[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/pdf/2303.02449) [![Static Badge](https://img.shields.io/badge/Pub-TMLR'24-blue)]() [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</center>

This repository contains the source codes for TMLR'24 paper:

 [**Exploit CAM by itself: Complementary Learning System for Weakly Supervised Semantic Segmentation**](https://arxiv.org/pdf/2303.02449). 

**Author List**: Wankou Yang, Jiren Mai, Fei Zhang, Tongliang Liu, Bo Han.


## Introduction

Weakly Supervised Semantic Segmentation (WSSS) with image-level labels has long been suffering from fragmentary object regions led by Class Activation Map (CAM), which is incapable of generating fine-grained masks for semantic segmentation. To guide CAM to find more non-discriminating object patterns, this paper turns to an interesting working mechanism in agent learning named Complementary Learning System (CLS). CLS holds that the neocortex builds a sensation of general knowledge, while the hippocampus specially learns specific details, completing the learned patterns. Motivated by this simple but effective learning pattern, we propose a General-Specific Learning Mechanism (GSLM) to explicitly drive a coarse-grained CAM to a fine-grained pseudo mask. Specifically, GSLM develops a General Learning Module (GLM) and a Specific Learning Module (SLM). The GLM is trained with image-level supervision to extract coarse and general localization representations from CAM. Based on the general knowledge in the GLM, the SLM progressively exploits the specific spatial knowledge from the localization representations, expanding the CAM in an explicit way. To this end, we propose the Seed Reactivation to help SLM reactivate non-discriminating regions by setting a boundary for activation values, which successively identifies more regions of CAM. Without extra refinement processes, our method is able to achieve improvements for CAM of over 20.0\% mIoU on PASCAL VOC 2012 and 10.0\% mIoU on MS COCO 2014 datasets, representing a new state-of-the-art among existing WSSS methods.



## Acknowledgement
 
 The repository is built mainly upon these repositories:
 
- [jiwoon-ahn/irn [1]](https://github.com/jiwoon-ahn/irn);

[1] Jiwoon Ahn et al. [Weakly supervised learning of instance segmentation with inter-pixel relations](https://openaccess.thecvf.com/content_CVPR_2019/papers/Ahn_Weakly_Supervised_Learning_of_Instance_Segmentation_With_Inter-Pixel_Relations_CVPR_2019_paper.pdf), CVPR 2019.

## Citation
```
@article{
yang2024exploit,
title={Exploit {CAM} by itself: Complementary Learning System for Weakly Supervised Semantic Segmentation},
author={Wankou Yang and Jiren Mai and Fei Zhang and Tongliang Liu and Bo Han},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2024},
url={https://openreview.net/forum?id=KutEe24Yai},
note={}
}
```
