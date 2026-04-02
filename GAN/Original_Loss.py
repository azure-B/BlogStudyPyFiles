import tensorflow as tf

# =================================================================
# 1. 하이퍼파라미터 및 가중치/편향 직접 초기화
# =================================================================
z_dim = 100
h_dim = 128
x_dim = 28 * 28


def weight_init(shape):
    return tf.Variable(tf.random.normal(shape, stddev=0.1))


def bias_init(shape):
    return tf.Variable(tf.zeros(shape))


W_G1 = weight_init([z_dim, h_dim])
b_G1 = bias_init([h_dim])
W_G2 = weight_init([h_dim, x_dim])
b_G2 = bias_init([x_dim])

W_D1 = weight_init([x_dim, h_dim])
b_D1 = bias_init([h_dim])
W_D2 = weight_init([h_dim, 1])
b_D2 = bias_init([1])


# =================================================================
# 2. 모델 연산 정의
# =================================================================
def generator(z):
    hidden = tf.nn.relu(tf.matmul(z, W_G1) + b_G1)
    out = tf.nn.sigmoid(tf.matmul(hidden, W_G2) + b_G2)
    return out


def discriminator(x):
    hidden = tf.nn.relu(tf.matmul(x, W_D1) + b_D1)
    out = tf.nn.sigmoid(tf.matmul(hidden, W_D2) + b_D2)
    return out


# =================================================================
# 3. 오리지널 목적 함수가 적용된 훈련 스텝 (기울기 소실 발생)
# =================================================================
learning_rate = 0.0002
optimizer_G = tf.optimizers.Adam(learning_rate)
optimizer_D = tf.optimizers.Adam(learning_rate)

eps = 1e-7


@tf.function
def train_step_original(x_real, batch_size):
    z = tf.random.normal([batch_size, z_dim])

    with tf.GradientTape() as tape_G, tf.GradientTape() as tape_D:
        x_fake = generator(z)

        D_real = discriminator(x_real)
        D_fake = discriminator(x_fake)

        # [판별자 손실 계산]
        loss_D = -tf.reduce_mean(tf.math.log(D_real + eps) + tf.math.log(1.0 - D_fake + eps))

        # [생성자 손실 계산 - Original Minimax Loss]
        loss_G = tf.reduce_mean(tf.math.log(1.0 - D_fake + eps))

    grad_G = tape_G.gradient(loss_G, [W_G1, b_G1, W_G2, b_G2])
    grad_D = tape_D.gradient(loss_D, [W_D1, b_D1, W_D2, b_D2])

    optimizer_G.apply_gradients(zip(grad_G, [W_G1, b_G1, W_G2, b_G2]))
    optimizer_D.apply_gradients(zip(grad_D, [W_D1, b_D1, W_D2, b_D2]))

    # W_G1(생성자 첫 번째 가중치)에 전달되는 기울기의 절대값 평균 계산
    grad_G_mean = tf.reduce_mean(tf.abs(grad_G[0]))

    return loss_D, loss_G, grad_G_mean


# =================================================================
# 4. 훈련 루프 (Epoch) 및 결과 출력
# =================================================================
epochs = 1000
batch_size = 64

print("학습 시작 (Original Minimax Loss - 기울기 소실 테스트)")
print("-" * 65)
print(f"{'Epoch':<8} | {'D Loss':<12} | {'G Loss':<12} | {'G Gradient (W_G1)':<15}")
print("-" * 65)

for e in range(1, epochs + 1):
    # 테스트를 위한 임의의 더미 실제 데이터 생성
    x_real_dummy = tf.random.normal([batch_size, x_dim])

    # 훈련 스텝 실행
    loss_D, loss_G, grad_G_mean = train_step_original(x_real_dummy, batch_size)

    # 100 에포크 단위로 결과 출력
    if e % 100 == 0:
        print(f"Epoch {e:<3} | D Loss: {loss_D:.4f} | G Loss: {loss_G:.4f} | G Grad: {grad_G_mean:.8f}")