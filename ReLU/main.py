import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import time


# 1. 커스텀 ReLU 레이어 정의
class ReLU(layers.Layer):
    def __init__(self, **kwargs):
        super(ReLU, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.math.maximum(0.0, inputs)


# 2. 비교 실험 함수
def run_experiment(activation_name):
    # 가상 데이터 생성 (2000개 샘플, 32개 특징)
    X = np.random.standard_normal((2000, 32)).astype('float32')
    y = np.random.standard_normal((2000, 1)).astype('float32')

    # 모델 구성
    model = models.Sequential()
    model.add(layers.Dense(64, input_shape=(32,)))

    # 활성화 함수 적용
    if activation_name == 'relu':
        model.add(ReLU())
    else:
        model.add(layers.Activation(activation_name))

    model.add(layers.Dense(1))

    # 컴파일 및 학습
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss='mse')

    start = time.time()
    history = model.fit(X, y, epochs=10, batch_size=32, verbose=0)
    end = time.time()

    return history.history['loss'][-1], end - start


# 3. 실행 및 결과 출력
for act in ['relu', 'sigmoid', 'tanh']:
    loss, duration = run_experiment(act)
    print(f"[{act:8}] 최종 Loss: {loss:.4f} | 소요 시간: {duration:.2f}s")