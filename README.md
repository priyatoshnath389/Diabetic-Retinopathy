#  Diabetic Retinopathy Detection using Deep Learning

This project implements a deep learning pipeline for detecting **Diabetic Retinopathy (DR)** from retina fundus images. The images have been preprocessed using **Gaussian filtering** and resized to **224x224**, making them suitable for use with transfer learning models like EfficientNet.

##  Project Highlights

-  Classification of retina images into 5 stages of DR.
-  Uses **EfficientNet-B3** with transfer learning.
-  Stratified train/validation/test splits to handle class imbalance.
-  Includes model evaluation and visualization tools.
-  GPU-accelerated training (CUDA compatible).

---

##  Dataset

The dataset is based on the **APTOS 2019 Blindness Detection** challenge and consists of Gaussian-filtered retina images stored in folders by class:

```bash
gaussian_filtered_images/
├── No_DR/
├── Mild/
├── Moderate/
├── Severe/
├── Proliferate_DR/
├── train.csv
├── export.pkl


##  Why EfficientNet-B3?

EfficientNet is a family of models developed by Google AI that scales up networks **efficiently** using a compound coefficient to balance **depth**, **width**, and **resolution**. EfficientNet-B3 specifically offers a strong trade-off between computational cost and classification accuracy — ideal for medical image analysis like diabetic retinopathy.

###  Key Features of EfficientNet-B3:

####  1. **Stochastic Depth** (a type of regularization)

- During training, some layers are randomly skipped (dropped) in each forward pass.
- This helps prevent overfitting and encourages the model to become more robust.
- It’s especially helpful in deep networks to avoid vanishing gradients.

####  2. **Inverted Bottleneck**

- Traditional bottlenecks: reduce → process → expand.
- Inverted bottleneck: **expand → process → reduce**.
- This structure allows the model to capture more complex features with fewer parameters and better gradient flow.

####  3. **Squeeze-and-Excitation (SE) Optimization**

- SE blocks help the model **focus on the most informative channels**.
- It works by:
  1. **Squeezing** (global average pooling): summarizing feature maps.
  2. **Exciting** (fully connected layers): reweighting each channel.
- In DR detection, this helps the model highlight small but critical regions like microaneurysms and hemorrhages.

---

##  Dataset

The dataset consists of preprocessed **Gaussian-filtered** retina images organized by DR severity levels:

