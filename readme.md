# ASL-Alphabet Recognizer

## Source of Dataset:

The Dataset is GPLv2 licensed and available under following link:

https://www.kaggle.com/datasets/grassknoted/asl-alphabet?resource=download

## How to run main.py file (efficiently with CUDA)

### Windows:

#### Use WSL2 for Cuda acceleration:

Setup a new venv inside of wsl:

``` python3 -m venv tf-gpu ```

Activate the venv:

``` source tf-gpu/bin/activate ```

Update pip inside of venv:

``` pip install --upgrade pip ```

Install with pip:

``` pip install tensorflow[and-cuda] ```

### Linux: 

You know what you`re doing...

### MacOS:

I dont care.