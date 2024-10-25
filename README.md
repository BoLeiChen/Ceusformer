# Ceusformer
This repository contains the codes for our paper, which is titled as "Joint Segmentation and Differential Diagnosis of Breast Lesions in Multi-modal Ultrasound Images via Causal Learning."

# Abstract

Computer vision-aided ultrasound imaging has become a popular choice for the differential diagnosis of breast lesions due to its radiation-free and minimal cost advantages. Recently, researchers have increasingly recognized the potential for boosting the diagnosis performance using multiple modalities, especially B-mode UltraSound (BUS) and Contrast-Enhanced UltraSound (CEUS). While existing works try to mimic how sonographers extract multi-model patterns and contrastively exploit them, they often perform unsatisfactorily. In particular, some efforts blindly model the sonographer’s diagnostic experience, which we argue hardly produces a diagnoser that outperforms human experts in such a biased learning paradigm. To tackle these issues, we propose a novel Contrast-enhanced ultrasound Transformer (Ceusformer) and a pioneering diagnostic technique rooted in the causal inference paradigm. Ceusformer exploits both modalities’ properties to fully mine complementary features and fuse them interactively to improve the segmentation and diagnosis of breast lesions. By delving into both observable and unobservable confounders within the BUS and CEUS, the back-door and front-door adjustment causal learning modules are proposed to promote unbiased learning by mitigating potential spurious correlations. Sufficient comparative and ablative studies underscore the superiority of our method over previous state-of-the-art approaches.

# Framework and Visualization

(a) An illustration of Ceusformer that integrates causal diagnostic techniques, including BACL and FACL. (b) An illustration of interactive cross-modal feature extraction and fusion, including the CNN (BUS) branch and the Transformer (CEUS) branch. (c) and (d) illustrate the temporal attention module and the contextual attention module in the Transformer (CEUS) branch, respectively. (e) illustrates examples of observable confounders handled by BACL.
<div align="center">
	<img src="./Fig1.png" alt="Editor" width="800">
</div>

Visualization of the class activation maps generated by different methods. The circles and arrows in (a) indicate Ceusformer's attention to the microvascular perfusion details surrounding the lesion. The heat maps in (b) reflect Ceusformer's insights into the segmentation of lesion regions in BUS images.
<div align="center">
	<img src="./Fig2.png" alt="Editor" width="800">
</div>

# Setup
We use an NVIDIA GeForce RTX 3090 GPU. The Python version and important dependent libraries are listed below:
```
conda create -n ceusformer python=3.8
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```
Installing Cupy according to [Cupy](https://docs.cupy.dev/en/stable/install.html).


# Dateset
See the ceus_data directory for data examples and formats. 

```
/ceus_data
├── imagenet_vid_train_5frames.json
├── imagenet_vid_val.json
├── rawframes
│   ├── benign
│   └── malignant
```

# Getting Started
(1) Training
```
python main.py --output_dir ./run/self_attn_final_exp1 --data_mode 5frames --batch_size 2 --lr 5e-4 --masks --cache_mode
```

(2) Testing
```
python main.py --output_dir ./run/self_attn_final_exp1 --data_mode '5frames' --batch_size 4 --lr 5e-5 --masks --cache_mode --eval --resume your_mode_path
```

## Acknowledge
This work is based on [TimeSformer](https://github.com/facebookresearch/TimeSformer). The authors thank Gedas Bertasius, Heng Wang, and Lorenzo Torresani for their works.
