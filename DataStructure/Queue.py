from collections import deque
import time

N = 100000

# 1. 성능이 매우 떨어지는 List 기반 큐 (사용 금지)
list_queue = list(range(N))
start = time.time()
while list_queue:
    list_queue.pop(0)  # O(n) 연산이 n번 반복되어 총 O(n^2) 소요
print(f"List pop(0) 소요 시간: {time.time() - start:.4f}초")

# 2. O(1) 성능을 보장하는 Deque (표준 접근법)
deque_queue = deque(range(N))
start = time.time()
while deque_queue:
    deque_queue.popleft()  # O(1) 연산이 n번 반복되어 총 O(n) 소요
print(f"Deque popleft() 소요 시간: {time.time() - start:.4f}초")