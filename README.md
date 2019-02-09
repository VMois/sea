# Speech Emotion Analysis
A neural network model for determining human speech emotions from audio recordings.

Based on https://github.com/MITESHPUTHRANNEU/Speech-Emotion-Analyzer

## Prerequisites
* A 64-bit Windows, Linux, or Mac OS machine (certain libraries such as pyarrow don't work on 32-bit machines)
* The Conda package and environment manager https://conda.io/projects/conda/en/latest/user-guide/install/index.html
* The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) https://zenodo.org/record/1188976

## Instructions
1. Run `conda env create -f env.yml` in the main directory
2. Activate the project enviroment with `conda activate sea`
3. Download `Audio_Speech_Actors_01-24.zip` and `Audio_Song_Actors_01-24.zip` from https://zenodo.org/record/1188976
4. Place these zip files in a folder called `raw-data` in the main directory
5. Run `python feature_extraction.py` to analyze and extract audio features (this may take a few minutes)
6. The results (filename, gender, emotion, and features) are stored in the Apache Parquet format with gzip compression in a file called `audio-features.parquet.gzip` located in the main directory

## Tests
* To view the contents of `audio-features.parquet.gzip`, run `python tests/view_parquet.py` in the main directory

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
