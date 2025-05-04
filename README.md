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
