from keras.callbacks import ModelCheckpoint
from preprocessing_utils import *
from preprocessing import *
from model import *
from config import config

X_train, X_test, y_train, y_test = config['preprocessing'](config['get_data']())

model = build_model(input_shape=X_train.shape[1:])

checkpointer = ModelCheckpoint(filepath=config['save_model'], verbose=1, save_best_only=True)
model.fit(X_train, y_train,
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        validation_data=(X_test, y_test),
        callbacks=[checkpointer]
        )
