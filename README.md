# Image Classification of Cats and Dogs using Fine-Tuned VGG19 CNN Model

## Problem Statement

This project aims to develop a deep learning-based image classification system to identify cats and dogs using a fine-tuned VGG19 model. By applying transfer learning on a labeled dataset, the model achieves high accuracy with minimal training time.

---

## Dataset Overview

- Classes: 2 (Cat, Dog)  
- Image Format: RGB  
- Preprocessed Image Size: 224×224 pixels  

### Preprocessing Steps

- Resize all images to 224×224 (VGG19 input requirement)
- Normalize images using ImageNet mean and standard deviation
- (Optional) Applied data augmentation: horizontal flips, rotations, etc.

---

## VGG19 Architecture and Working

VGG19 is a 19-layer deep Convolutional Neural Network developed by the Visual Geometry Group (VGG) at Oxford. It is widely used for transfer learning tasks due to its consistent structure and effectiveness.

### Architecture Highlights

- Layers: 16 convolutional + 3 fully connected (FC)
- Convolution: 3×3 filters, stride=1, padding=1, ReLU activation
- Pooling: MaxPooling (2×2) after each conv block
- Fully Connected: Two FC layers with 4096 units; final layer customized for 2-class output

### Working Flow

1. Input: RGB image (224×224)
2. Feature Extraction: Convolution and pooling layers capture patterns (edges, textures, shapes)
3. Flattening: Feature maps are flattened before entering FC layers
4. Classification: Final layer outputs softmax probabilities
5. Prediction: Class with highest probability is selected

---

## Transfer Learning

- Loaded pretrained VGG19 weights from ImageNet
- Replaced final FC layer to suit 2-class problem
- Only final layers were trained; all others frozen
- Achieved fast convergence and high accuracy

---

## Training Configuration

- Epochs: 3  
- Batch Size: 2400 (train), 600 (test)  
- Loss Function: CrossEntropyLoss  
- Optimizers: Adam / SGD  
- Training Time: Measured using `time.time()`  

During training:
- Batches passed through the network
- Predictions compared with labels
- Loss computed, backpropagated, and weights updated

---

## Evaluation and Results

### Accuracy Progress

| Epoch | Start Accuracy | End Accuracy |
|-------|----------------|--------------|
| 1     | 66.67%         | 94.43%       |
| 2     | 90.48%         | 97.17%       |
| 3     | 94.05%         | 97.75%       |

### Observations

- Epoch 1: Rapid learning, high jump in accuracy
- Epoch 2: Minor loss spikes but robust generalization
- Epoch 3: Stable training with accuracy peaking near 98%

---

## Libraries and Tools

- torch, torchvision – Deep learning framework
- PIL, os, numpy – Image and file handling
- matplotlib – Visualization

---

## Confusion Matrix

<img width="706" height="530" alt="image" src="https://github.com/user-attachments/assets/a13c7d39-2168-434d-9584-1d19663f25b0" />

## Predicted Output

<img width="968" height="832" alt="image" src="https://github.com/user-attachments/assets/d3a618e1-ffad-49aa-abe2-8eaf466eb8c4" />

## Future Improvements

- Add real-time prediction with Gradio or Streamlit
- Introduce learning rate schedulers (e.g., StepLR, ReduceLROnPlateau)
- Unfreeze deeper conv layers for more nuanced fine-tuning

---


