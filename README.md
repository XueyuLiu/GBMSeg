# Feature-prompting GBMSeg: One Shot Reference Guided Training-Free Feature Matching for Glomerular Basement Membrane Segmentation and Quantification
<br>**[Xueyu Liu](https://scholar.google.com.hk/citations?user=jeatLqIAAAAJ&hl=zh-CN), Guangze Shi, Rui Wang, Yexin Lai, Jianan Zhang, Lele Sun, Quan Yang, Yongfei Wu*, Weixia Han, Ming Li, and Wen Zheng**<br>
<sup>1</sup>[Taiyuan University of Technology](https://www.tyut.edu.cn/), &nbsp;
<sup>2</sup>[The Second Affiliated Hospital of Shanxi Medical University](https://www.sydey.com/)，&nbsp;
<sup>3</sup>[Shanxi Provincial People's Hospital](https://www.sxsrmyy.com/)


### This work has been accepted by MICCAI2024!


We present GBMSeg, a training-free framework that automates the segmentation and measurement of the glomerular basement membrane (GBM) in TEM using only one-shot reference images. GBMSeg leverages the robust feature matching capabilities of pretrained foundation models (PFMs) to generate initial prompts, designs novel prompting engineering for optimized prompting methods, and utilizes a class-agnostic segmentation model to obtain the final segmentation result. 

<p align="center">
<img width="800" alt="ablation" src="img/ablation.png">
</p>

## Usage 
### Setup 

- Cuda 12.0
- Python 3.9.18
- PyTorch 2.0.0


### Datasets
    ../                          # parent directory
    ├── ./data                   # data path
    │   ├── reference_image      # the one-shot reference image
    │   ├── reference_mask       # the one-shot reference mask
    │   ├── target_image         # testing images

### Generate prompt
```
cd GBMSeg/feature-matching
python generate_prompt.py
```

### Automatic prompt engineering
```
cd GBMSeg/tools
python automatic_prompt_engineering.py
```

### Segmentation 
```
mkdir GBMSeg/results
cd GBMSeg/segmenting-anything
python segment.py
```

## Acknowledgement
Thanks [DINOv2](https://github.com/facebookresearch/dinov2), [SAM](https://github.com/facebookresearch/segment-anything). for serving as building blocks of GBMSeg.
