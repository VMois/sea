# Speech Emotion Analysis

A neural network model for determining human speech emotions from audio recordings.

Based on https://github.com/MITESHPUTHRANNEU/Speech-Emotion-Analyzer

## Prerequisites

* A 64-bit Windows, Linux, or Mac OS machine (certain libraries such as pyarrow don't work on 32-bit machines)
* The Conda package and environment manager https://conda.io/projects/conda/en/latest/user-guide/install/index.html

## Datasets

* The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) https://zenodo.org/record/1188976
* more in future... 

## How to start

### Prepare environment
1. Install Conda through Miniconda/Anaconda
2. Clone the repo
3. Create Conda environment using `conda env create -f env.yaml`
4. Activate the enviroment with `source activate sea`
5. Install Tensorflow for CPU using `conda install tensorflow=1.12.0` (or `tensorflow-gpu` for GPU support)
6. Install keras using `conda install keras=2.2.4`

### Prepare data

#### RAVDESS dataset
1. Download `Audio_Speech_Actors_01-24.zip` and `Audio_Song_Actors_01-24.zip` from https://zenodo.org/record/1188976
2. Place these zip files in a folder called `raw-data` in the main directory

#### more datasets...

#### Extract features
1. Run `python features_extraction.py` to analyze and extract audio features (this may take a few minutes)
2. The results are stored in the Apache Parquet format 
in a directories `data/train/[dataset_name].parquet` and `data/test/[dataset_name].parquet`.

## Parquet data structure (in progress...)

Features from all datasets are combined to unified format. 
All features are stored in parquet files with following structure: 

- **filename**: name of the original audio file
- **gender**: 'male' or 'female' voice
- **emotion**: basic emotions ('neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprise')
- **features**: MFCC features (detailed description will be added soon)
