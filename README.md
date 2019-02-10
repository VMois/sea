# Speech Emotion Analysis

A neural network model for determining human speech emotions from audio recordings.

Based on https://github.com/MITESHPUTHRANNEU/Speech-Emotion-Analyzer

## Prerequisites

* A 64-bit Windows, Linux, or Mac OS machine (certain libraries such as pyarrow don't work on 32-bit machines)
* The Conda package and environment manager https://conda.io/projects/conda/en/latest/user-guide/install/index.html
* The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) https://zenodo.org/record/1188976

## How to start

1. Install Conda through Miniconda/Anaconda
2. Clone the repo
3. Create Conda environment using `conda env create -f env.yml`
4. Activate the enviroment with `source activate sea`
5. Install Tensorflow for CPU using `conda install tensorflow=1.12.0` (or `tensorflow-gpu` for GPU support)
6. Download `Audio_Speech_Actors_01-24.zip` and `Audio_Song_Actors_01-24.zip` from https://zenodo.org/record/1188976
7. Place these zip files in a folder called `raw-data` in the main directory
8. Run `python feature_extraction.py` to analyze and extract audio features (this may take a few minutes)
9. The results (filename, gender, emotion, and features) are stored with the Apache Parquet format in a file called `audio-features.parquet` located in the main directory

## Libraries

* LibROSA (https://librosa.github.io/librosa/)
* TensorFlow (https://www.tensorflow.org/)
* Scikit-learn (https://scikit-learn.org/)
* Numpy (http://www.numpy.org/)
* Matplotlib (https://matplotlib.org/)
* Pandas (https://pandas.pydata.org/)
* Scipy (https://www.scipy.org/)
* Apache Arrow (https://arrow.apache.org/docs/python/)
* Apache Parquet (https://parquet.apache.org/)
