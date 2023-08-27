# EigenFaces-Visualizer

**EigenFaces** is a Python program that demonstrates the concept of Eigenfaces, which are images that can be combined with a mean face to generate new facial images using Principal Component Analysis (PCA). This program is designed for face recognition and facial image generation.

## Getting Started

1. Clone this repository to your local machine.
2. Place your facial images in the `images/` folder.
3. Make sure you have Python and required libraries installed.

## Usage

Run the following command to launch the EigenFaces application:

```bash
python3 eigenfaces.py
``````

This will open windows displaying the average face, the generated face using sliders, and individual eigenfaces. Adjust the sliders to modify eigenface weights and observe their effects on the generated face.

## Features

- Calculate eigenfaces using PCA.
- Display average face and generated face with adjustable eigenface weights.
- Display individual eigenfaces to understand their contributions.
- Rank eigenfaces based on their importance using eigenvalues.
