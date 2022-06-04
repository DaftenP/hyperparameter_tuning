from tensorflow import keras
import data_reader
import MLPC
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import kerastuner as kt
from sklearn.metrics import accuracy_score

EPOCHS = 200

dr = data_reader.DataReader()

tuner = kt.Hyperband(MLPC.MLPC(),
                     objective='val_accuracy',
                     max_epochs=100,
                     executions_per_trial=3,
                     overwrite=True,
                     factor=3)

tuner.search(dr.train_X, dr.train_Y, epochs=100, validation_split=0.1)
best_hps = tuner.get_best_hyperparameters()[0]


# 최적 하이퍼파라미터를 출력합니다.

print(f"""
units_0 : {best_hps.get('units_0')}
units_1 : {best_hps.get('units_1')}
units_2 : {best_hps.get('units_2')}
units_3 : {best_hps.get('units_3')}
units_4 : {best_hps.get('units_4')}
dropout_0 : {best_hps.get('dropout_0')}
dropout_1 : {best_hps.get('dropout_1')}
dropout_2 : {best_hps.get('dropout_2')}
dropout_3 : {best_hps.get('dropout_3')}
dropout_4 : {best_hps.get('dropout_4')}
learning_rate : {best_hps.get('learning_rate')}
batch_size : {best_hps.get('batch_size')}
""")


# 배치 크기는 따로 저장했다가 fit 메소드에서 적용합니다.
batch_size = best_hps.get('batch_size')


# 최적값으로 모델을 생성합니다.
model = tuner.hypermodel.build(best_hps)


# 학습을 진행합니다.
# validation_split 아규먼트로 Training 데이터셋의 10%를 Validation 데이터셋으로 사용하도록합니다.
# 예를 들어 배치 크기가 256이라는 것은 전체 데이터셋을 샘플 256개씩으로 나누어서 학습에 사용한다는 의미입니다.
# 예를 들어 에포크(epochs)가 10이라는 것은 전체 train 데이터셋을 10번 본다는 의미입니다.
history = model.fit(
  dr.train_X, dr.train_Y,
  validation_split = 0.1,
  batch_size=batch_size,
  epochs=100,
)


# 학습중 손실 변화를 그래프로 그립니다.
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


pred_Y = model.predict(dr.test_X)

pred_Y = np.argmax(pred_Y, axis=1)
dr.test_Y = np.argmax(dr.test_Y, axis=1)

print(accuracy_score(dr.test_Y, pred_Y))
