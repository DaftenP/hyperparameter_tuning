from tensorflow import keras

import data_reader
import MLPC
import matplotlib.pyplot as plt
import pandas as pd

EPOCHS = 200

dr = data_reader.DataReader()

mlp = MLPC.MLPC().model

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = mlp.fit(dr.train_X, dr.train_Y, epochs=EPOCHS, batch_size=512,
        validation_data=(dr.test_X, dr.test_Y),
        callbacks=[early_stop])

print()

history_df = pd.DataFrame(history.history)
history_df['loss'].plot()
history_df['val_loss'].plot()
plt.legend()
plt.show()


history_df = pd.DataFrame(history.history)
history_df['accuracy'].plot()
history_df['val_accuracy'].plot()
plt.legend()
plt.show()