import tensorflow as tf

# =================================================================
# 1. 하이퍼파라미터 설정
# =================================================================
z_dim = 100
x_dim = 28 * 28


# =================================================================
# 2. 모델 구조 정의
# =================================================================
def build_generator():
    model = tf.keras.Sequential([
        # 입력 z를 받아 128개 노드의 은닉층으로 연결 (ReLU 활성화)
        tf.keras.layers.Dense(128, activation='relu', input_shape=(z_dim,)),
        # 784 픽셀의 이미지로 출력 (Sigmoid 활성화)
        tf.keras.layers.Dense(x_dim, activation='sigmoid')
    ])
    return model


def build_discriminator():
    model = tf.keras.Sequential([
        # 784 픽셀 이미지를 받아 128개 노드의 은닉층으로 연결
        tf.keras.layers.Dense(128, activation='relu', input_shape=(x_dim,)),
        # 0~1 사이의 진짜일 확률 1개 출력 (Sigmoid 활성화)
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model


# 모델 객체 생성
generator = build_generator()
discriminator = build_discriminator()

# =================================================================
# 3. 훈련 도구 설정
# =================================================================
optimizer_G = tf.optimizers.Adam(learning_rate=0.0002)
optimizer_D = tf.optimizers.Adam(learning_rate=0.0002)
eps = 1e-7


# =================================================================
# 4. 훈련 스텝
# =================================================================
@tf.function
def train_step(x_real, batch_size):
    z = tf.random.normal([batch_size, z_dim])

    with tf.GradientTape() as tape_G, tf.GradientTape() as tape_D:
        # 가짜 데이터 생성
        x_fake = generator(z, training=True)

        # 판별자 평가
        D_real = discriminator(x_real, training=True)
        D_fake = discriminator(x_fake, training=True)

        # ---------------------------------------------------------
        # [판별자 손실 계산]
        # 수식: -[log(D(x)) + log(1 - D(G(z)))]
        # ---------------------------------------------------------
        loss_D = -tf.reduce_mean(tf.math.log(D_real + eps) + tf.math.log(1.0 - D_fake + eps))

    # 역전파 계산 (trainable_variables를 통해 모델 내부의 모든 가중치를 자동으로 가져옴)
    grad_D = tape_D.gradient(loss_D, discriminator.trainable_variables)

    # 가중치 업데이트
    optimizer_D.apply_gradients(zip(grad_D, discriminator.trainable_variables))

    return loss_D