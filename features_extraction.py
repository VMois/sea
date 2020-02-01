import os
import logging
from features_scripts.images.crema import crema_extract
from features_scripts.images.ravdess import ravdess_extract
from features_scripts.images.savee import savee_extract

logging.basicConfig(level=logging.INFO)

if not os.path.exists('raw-data'):
    logging.info('Please, create raw-data/ folder and put input files according to the repo README')
    exit(1)

logging.info('Features extraction starts...')

ravdess_extract(1)
savee_extract(2)
crema_extract(3)
