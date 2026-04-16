import tensorflow as tf

# 1. 최대 10개의 정수(int32)를 담을 수 있는 선입선출(FIFO) 큐 생성
fifo_queue = tf.queue.FIFOQueue(capacity=10, dtypes=tf.int32)

# 2. 생산자 (Producer): 큐의 뒤(Rear)에 데이터를 밀어 넣습니다. (Enqueue)
fifo_queue.enqueue(1)
fifo_queue.enqueue(2)
fifo_queue.enqueue_many([3, 4, 5]) # 여러 개를 한 번에 넣을 수도 있습니다.

print(f"현재 큐에 쌓인 데이터 개수: {fifo_queue.size().numpy()}개")

# 3. 소비자 (Consumer): 큐의 앞(Front)에서 데이터를 꺼냅니다. (Dequeue)
# 가장 먼저 넣었던 '1'이 가장 먼저 나옵니다.
print(f"첫 번째 Dequeue: {fifo_queue.dequeue().numpy()}")
print(f"두 번째 Dequeue: {fifo_queue.dequeue().numpy()}")