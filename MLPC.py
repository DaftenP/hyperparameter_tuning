import tensorflow as tf
from tensorflow import keras
import numpy as np
import kerastuner as kt


class MLPC(kt.HyperModel):
    def build(self, hp):
        model = keras.Sequential()
        model.add(keras.layers.Dense(11))
        for i in range(hp.Int('num_layers', 2, 5)):
            model.add(keras.layers.Dense(units=hp.Int('units_' + str(i),
                                                      min_value=16,
                                                      max_value=180),
                                         activation='relu'))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Dropout(hp.Float('dropout_' + str(i), min_value=0.0, max_value=0.5, step=0.05)))
        model.add(keras.layers.Dense(10, activation='softmax'))

        hp_learning_rate = hp.Float('learning_rate', min_value=1e-6, max_value=1e-3, sampling='LOG')
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Int('batch_size', min_value=16, max_value=512, step=16),
            **kwargs,
        )
