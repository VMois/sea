'''

BME AI - Speech Emotion Analysis - Parquet File Viewer
------------------------------------------------------
This script reads audio-features.parquet.gzip and allows you to view its contents.

'''

import pyarrow.parquet as pq
import pandas as pd
import time
import os

print(' ___ __  __ ___     _   ___ ')
print('| _ )  \/  | __|   /_\ |_ _|')
print('| _ \ |\/| | _|   / _ \ | | ')
print('|___/_|  |_|___| /_/ \_\___|')
print('                            ')

# Time script execution for performance evaluation.
start = time.time()

if os.path.exists('audio-features.parquet.gzip'):
  t = pq.read_table('audio-features.parquet.gzip')
  p = t.to_pandas()
  pd.set_option('display.expand_frame_repr', False)
  print(p)
else:
  print('Error: audio-features.parquet.gzip was not found.')
  print('Please run feature_extraction.py first to extract the audio features.')
end = time.time()
print('This script took ', str(round(end - start, 2)), 'seconds to execute.')
