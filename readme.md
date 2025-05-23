# ASL-Alphabet Recognizer

## Source of Dataset:

The Dataset is GPLv2 licensed and available under following link:

https://www.kaggle.com/datasets/grassknoted/asl-alphabet?resource=download

## Preparing the Dataset for training

1. Download the dataset from kaggle.com over the link above.
2. Rename the folder ```.../archive/asl_alphabet_train/asl_alphabet_train``` to ```asl_alphabet_original```
3. Copy the renamed folder to the ```/data``` directory
4. Run the ```split_data.py``` script

### Explaination on Splitting and Methodology

The dataset consists of 2 directories: ```asl_alphabet_train``` which holds approximately 3000 images per letter
and ```asl_alphabet_test``` which holds only one image per letter. 
To simplify things we take only the ```asl_alphabet_train``` directory and split it into training, validation 
and test data.
The ```asl_alphabet_original``` directory stays untouched when you run the ```split_data.py``` script. The train, val and 
test directories will be overwriten if you re-run the script. To ensure correct methodology we split into 50% test, 
20% validation and 30% test data. We are tuning our neuronal net on the validation set and merge the train and validation
dataset for the final training.
Feel free to modify this script to experiment with this dataset.


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
