import os
from features_scripts.ravdess import ravdess_extract


if not os.path.exists('raw-data'):
    print('Please, create raw-data/ folder and put input files according to the repo README')
    exit(1)

# Prepare output folder structure for the data
if not os.path.exists('data'):
    os.makedirs('data')

if not os.path.exists('data/train'):
    os.makedirs('data/train')

if not os.path.exists('data/test'):
    os.makedirs('data/test')

ravdess_extract()
