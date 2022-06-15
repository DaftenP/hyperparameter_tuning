import data_reader
import MLPC
import matplotlib.pyplot as plt
import pandas as pd
import kerastuner as kt
from sklearn.metrics import accuracy_score, classification_report


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


# 최적 하이퍼파라미터 출력

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


# 배치 크기는 따로 저장했다가 fit 메소드에서 적용
batch_size = best_hps.get('batch_size')


# 최적값으로 모델 생성
model = tuner.hypermodel.build(best_hps)


# 학습 진행
history = model.fit(
  dr.train_X, dr.train_Y,
  validation_split=0.1,
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

print(accuracy_score(pred_Y, dr.test_Y))
print(classification_report(pred_Y, dr.test_Y))
