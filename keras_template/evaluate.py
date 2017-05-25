from keras.models import load_model
from preprocessing_utils import *
from config import config

model = load_model(config['save_model'])

def predict(model):
    pass
