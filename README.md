# Speech Emotion Analysis

A neural network model for determining human speech emotions from audio recordings.

## Datasets

| id        | name          |
| ------------- |:-------------:|
| 1 | The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) |
| 2 | SAVEE |
| 3 | CREMA-D (Crowd-sourced Emotional Mutimodal Actors Dataset) |

## How to start

### Prerequisites

* A 64-bit Windows, Linux, or Mac OS machine (certain libraries such as pyarrow don't work on 32-bit machines)
* The Conda package and environment manager https://conda.io/projects/conda/en/latest/user-guide/install/index.html

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

## Related articles

- [Speech Emotion Recognition with Convolutional Neural Network](https://towardsdatascience.com/speech-emotion-recognition-with-convolution-neural-network-1e6bb7130ce3)

## Credits

- https://github.com/MITESHPUTHRANNEU/Speech-Emotion-Analyzer
