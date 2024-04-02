# Feature-prompting GBMSeg: One Shot Reference Guided Training-Free Feature Matching for Glomerular Basement Membrane Segmentation and Quantification
<br>**[Xueyu Liu](https://scholar.google.com.hk/citations?user=jeatLqIAAAAJ&hl=zh-CN), Guangze Shi, Rui Wang, Yexin Lai, Jianan Zhang, Lele Sun, Quan Yang, Yongfei Wu*, Weixia Han, Ming Li, and Wen Zheng**<br>


<p align="center">
<img width="800" alt="structure" src="img/structure.png">
</p>

We present GBMSeg, a training-free framework that automates the segmentation and measurement of the glomerular basement membrane (GBM) in TEM using only one-shot reference images. GBMSeg leverages the robust feature matching capabilities of pretrained foundation models (PFMs) to generate initial prompts, designs novel prompting engineering for optimized prompting methods, and utilizes a class-agnostic segmentation model to obtain the final segmentation result. Additionally, we propose a thickness measurement method based on the segmentation mask to quantify pathological indicators and analyze and predict results by integrating clinical indicators with diagnostic outcomes. 

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
