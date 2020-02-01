import os
from fastai.vision import *

labels = ['happy_male', 'happy_female', 'sad_male', 'sad_female']
fn_paths = [f'img/{x}' for x in os.listdir('images')]
pat = r"(\w+_\w+)_\d+\.png$"
data = ImageDataBunch.from_name_re('img/', fn_paths, pat=pat, size=(40, 259))

learn = cnn_learner(data, models.resnet18, metrics=accuracy)
learn.fit(1)