from tensorflow import keras
import data_reader
import pandas as pd
import matplotlib.pyplot as plt

dr = data_reader.DataReader()
model = keras.Sequential()

model.add(keras.layers.Dense(11))
model.add(keras.layers.Dense(32, activation="relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(32, activation="relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10, activation="softmax"))

model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(dr.train_X, dr.train_Y, validation_split=0.1, batch_size=256, epochs=100)

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
