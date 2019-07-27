import os
from features_scripts.ravdess import ravdess_extract
from features_scripts.savee import savee_extract

if not os.path.exists('raw-data'):
    print('Please, create raw-data/ folder and put input files according to the repo README')
    exit(1)

ravdess_extract(1)
savee_extract(2)
