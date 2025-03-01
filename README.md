# Brain Tumor Detection and Classification

## Overview
This project implements a brain tumor detection and classification system using image processing techniques and a Convolutional Neural Network (CNN). It takes MRI images as input, segments the tumor area, and classifies the tumor as either malignant or benign.

## Features
- **Image Preprocessing**: Resizing and grayscale conversion of MRI images.
- **Tumor Segmentation**: Uses region-based methods to identify and segment tumors.
- **Visualization**: Displays the original image, segmented tumor image, and detected tumors.
- **Tumor Classification**: Classifies tumors as benign or malignant based on area and density.

## Requirements
- MATLAB (version 2018a or later)
- Image Processing Toolbox
- Deep Learning Toolbox

## Installation
1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/yourusername/brain-tumor-detection.git
2. Navigate to project directory:
   cd brain-tumor-detection
   
## Usage
Open MATLAB and run the script tumor_detection.m.
When prompted, select an MRI image from your local storage.
The system will display the input image, resized image, grayscale image, tumor area, and the detected tumor on the input image.
The classification result (benign or malignant) will be displayed in a message box.

## Code Explanation
**Image Reading and Preprocessing:** The script reads the input MRI image and preprocesses it by resizing and converting it to grayscale.
**Segmentation:** The tumor is segmented using region-based methods, and its properties (area, density) are analyzed.
**Neural Network Training**: A Convolutional Neural Network is trained using a dataset of MRI images to classify tumors.
**Output Classification**: The output is classified as benign or malignant based on the properties of the segmented tumor.
