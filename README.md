# Ceusformer
Code for "Joint Segmentation and Differential Diagnosis of Breast Lesions in Multi-modal Ultrasound Images via Causal Learning."

# Abstract

Computer vision-aided ultrasound imaging has become a popular choice for the differential diagnosis of breast lesions due to its radiation-free and minimal cost advantages. Recently, researchers have increasingly recognized the potential for boosting the diagnosis performance using multiple modalities, especially B-mode UltraSound (BUS) and Contrast-Enhanced UltraSound (CEUS). While existing works try to mimic how sonographers extract multi-model patterns and contrastively exploit them, they often perform unsatisfactorily. In particular, some efforts blindly model the sonographer’s diagnostic experience, which we argue hardly produces a diagnoser that outperforms human experts in such a biased learning paradigm. To tackle these issues, we propose a novel Contrast-enhanced ultrasound Transformer (Ceusformer) and a pioneering diagnostic technique rooted in the causal inference paradigm. Ceusformer exploits both modalities’ properties to fully mine complementary features and fuse them interactively to improve the segmentation and diagnosis of breast lesions. By delving into both observable and unobservable confounders within the BUS and CEUS, the back-door and front-door adjustment causal learning modules are proposed to promote unbiased learning by mitigating potential spurious correlations. Sufficient comparative and ablative studies underscore the superiority of our method over previous state-of-the-art approaches.

# Setup
Coming Soon ...

# Dateset
Coming soon...

# Getting Started
Coming soon...

# Examples and Demos

Examples of point navigation and target-driven navigation.
<div align="center">
	<img src="./Fig1.png" alt="Editor" width="800">
</div>

Visualization of the navigation process.
<div align="center">
	<img src="./Fig2.png" alt="Editor" width="800">
</div>


## Acknowledge
This work is based on [TimeSformer](https://github.com/facebookresearch/TimeSformer). The authors thank Gedas Bertasius, Heng Wang, and Lorenzo Torresani for their works.
