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
def train_step_original(x_real, batch_size):
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

        # ---------------------------------------------------------
        # 수식: log(1 - D(G(z))) 최소화
        # 여기서 생성자의 기울기 소실(Vanishing Gradient)이 발생함
        # ---------------------------------------------------------
        loss_G = tf.reduce_mean(tf.math.log(1.0 - D_fake + eps))

    # 역전파 계산 (trainable_variables를 통해 가중치 리스트를 자동으로 가져옴)
    grad_G = tape_G.gradient(loss_G, generator.trainable_variables)
    grad_D = tape_D.gradient(loss_D, discriminator.trainable_variables)

    # 가중치 업데이트
    optimizer_G.apply_gradients(zip(grad_G, generator.trainable_variables))
    optimizer_D.apply_gradients(zip(grad_D, discriminator.trainable_variables))

    # Keras Sequential 내부의 첫 번째 Dense 레이어 가중치(W)의 기울기 절대값 평균 추출
    # grad_G[0]은 모델의 가장 첫 번째 레이어(W_G1)의 기울기를 의미
    grad_G_mean = tf.reduce_mean(tf.abs(grad_G[0]))

    return loss_D, loss_G, grad_G_mean

# =================================================================
# 5. 훈련 루프 (Epoch) 및 결과 출력
# =================================================================
epochs = 1000
batch_size = 64

print("학습 시작 (Original Minimax Loss - 기울기 소실 테스트)")
print("-" * 65)
print(f"{'Epoch':<8} | {'D Loss':<12} | {'G Loss':<12} | {'G Gradient (W_G1)':<15}")
print("-" * 65)

for e in range(1, epochs + 1):
    # 테스트를 위한 임의의 더미 실제 데이터 생성함
    x_real_dummy = tf.random.normal([batch_size, x_dim])

    # 훈련 스텝 실행함
    loss_D, loss_G, grad_G_mean = train_step_original(x_real_dummy, batch_size)

    # 100 에포크 단위로 결과 출력함
    if e % 100 == 0:
        print(f"Epoch {e:<3} | D Loss: {loss_D:.4f} | G Loss: {loss_G:.4f} | G Grad: {grad_G_mean:.8f}")