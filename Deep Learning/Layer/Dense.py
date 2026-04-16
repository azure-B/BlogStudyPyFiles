import tensorflow as tf


class CustomDense(tf.keras.layers.Layer):
    # 하이퍼파라미터를 저장하는 초기화 메서드임
    def __init__(self, units, activation=None, **kwargs):
        super(CustomDense, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    # 입력 형태가 확정될 때 가중치를 생성함
    def build(self, input_shape):
        # 가중치 행렬 W를 생성함
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True,
            name='kernel'
        )
        # 편향 벡터 b를 생성함
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='bias'
        )
        super(CustomDense, self).build(input_shape)

    # 실제 행렬 연산과 활성화 함수를 적용함
    def call(self, inputs):
        # Y = XW + b 연산 수행함
        linear_output = tf.matmul(inputs, self.w) + self.b

        if self.activation is not None:
            return self.activation(linear_output)
        return linear_output


# 사용 예시임
# 임의의 데이터 텐서 생성함
x_data = tf.random.normal((32, 128))
custom_layer = CustomDense(units=64, activation='relu')

# call 메서드가 실행되어 연산 결과를 반환함
output_data = custom_layer(x_data)
print(output_data.shape)  # 출력 형태: (32, 64)