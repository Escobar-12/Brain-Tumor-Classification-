# Brain Tumor Classification Using MobileNet and TensorFlow

## Overview
This project implements a **deep learning pipeline** for classifying brain MRI images into four categories:
- Glioma
- Meningioma
- Pituitary tumor
- No tumor

The model leverages **MobileNet** as a pre-trained feature extractor with a custom classifier on top.  
It uses TensorFlow 2.10.1, runs on GPU if available, and includes data augmentation, early stopping, and model checkpointing.

---

## Features
- Preprocessing and loading images from directory using `tf.keras.utils.image_dataset_from_directory`
- Data augmentation: Random brightness and contrast adjustments
- Transfer learning with MobileNet (pre-trained on ImageNet)
- Custom fully connected classifier layers with BatchNormalization and Dropout
- Early stopping and best model checkpoint saving
- GPU acceleration support via CUDA/cuDNN (if available)
- Evaluation with accuracy, confusion matrix, and classification report
- Visualization of training/validation accuracy and loss
- Visualization of predictions on test images

---

## Requirements
- Python 3.10
- TensorFlow 2.10.1
- NumPy 1.26.4
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- CUDA 11.2 and cuDNN 8.1 (for GPU support)
- Conda (recommended for environment management)

---

## Directory Structure

project_root/  
│  
├─ data/  
│ ├─ train/     
│ │ ├─ glioma/      
│ │ ├─ meningioma/      
│ │ ├─ notumor/ 
│ │ └─ pituitary/   
│ └─ test/  
│   ├─ glioma/    
│   ├─ meningioma/    
│   ├─ notumor/   
│   └─ pituitary/ 
│   
├─ BrainTumourDetection.ipynb # Jupyter notebook with model training & evaluation  
├─ best_model_mobileNet.keras # Saved best model weights  
└─ README.txt
