# Ceusformer
This repository contains the codes for our paper, which is titled as "Joint Segmentation and Differential Diagnosis of Breast Lesions in Multi-modal Ultrasound Images via Causal Learning."

# Abstract

Recently, researchers have increasingly recognized the potential for boosting the differential diagnosis of breast lesions using multimodal ultrasound, especially B-mode UltraSound (BUS) and Contrast-Enhanced UltraSound (CEUS). Although existing methods try to mimic how sonographers extract and contrastively exploit multimodel patterns, they often perform unsatisfactorily. In particular, some efforts blindly model the sonographer’s diagnostic experience, which we argue hardly produces a diagnoser that outperforms human experts in such a biased learning paradigm. To tackle these issues, we propose a dual-branch Contrast-enhanced ultrasound Transformer (Ceusformer) that integrates novel diagnostic techniques rooted in the causal inference paradigm. Ceusformer uses a bridging design to mine complementary features fully, which are interactively fused to improve the segmentation and diagnosis of breast lesions. By delving into both observable and unobservable confounders within the BUS and CEUS, the back-door and front-door adjustment causal learning modules are proposed to promote unbiased learning by mitigating potential spurious correlations. Sufficient comparative and ablative studies emphasize that our method is significantly superior to previous methods and human experts.

# Framework and Visualization



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
