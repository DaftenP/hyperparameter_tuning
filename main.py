from tensorflow import keras
import data_reader
import MLPC

EPOCHS = 200

dr = data_reader.DataReader()

mlp = MLPC.MLPC().model

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
mlp.fit(dr.train_X, dr.train_Y, epochs=EPOCHS, batch_size=512,
        validation_data=(dr.test_X, dr.test_Y),
        callbacks=[early_stop])
