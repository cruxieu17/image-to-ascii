# Image to ASCII Art using Neural Networks

This repository contains a PyTorch implementation of a neural network that converts images to ASCII art. The model uses a pretrained ResNet encoder and an LSTM decoder to generate ASCII art from images.

## Directory Structure

```
image-to-ascii
│
├── data
│   ├── images
│   └── ascii
│
├── src
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
│
├── README.md
└── requirements.txt
```

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- PIL
- numpy

You can install the required packages using:

```bash
pip install -r requirements.txt
```

## Usage

1. Place your images in the `data/images` directory.
2. Place your corresponding ASCII text files in the `data/ascii` directory.
3. Run the training script:

```bash
python src/train.py
```

## Files Description

### `src/dataset.py`

Contains the `ImageToASCIIDataset` class which handles loading and preprocessing of the image and ASCII data.

### `src/model.py`

Contains the `ImageToASCIIModel` class which defines the neural network architecture.

### `src/train.py`

Contains the training loop and logic to train the model.

### `src/utils.py`

Contains utility functions such as the `collate_fn` for handling variable-length sequences.

## Author

Kartik T. Shinde