class Stack:
    def __init__(self):
        # 데이터를 담을 빈 리스트 초기화
        self._elements = []

    def push(self, item):
        """스택의 맨 위(끝)에 데이터를 추가합니다. O(1)"""
        self._elements.append(item)

    def pop(self):
        """스택의 맨 위(끝)에서 데이터를 꺼내고 반환합니다. O(1)"""
        if self.is_empty():
            raise IndexError("빈 스택에서 pop을 수행할 수 없습니다.")
        return self._elements.pop()

    def peek(self):
        """데이터를 꺼내지 않고 맨 위의 값을 확인만 합니다. O(1)"""
        if self.is_empty():
            raise IndexError("빈 스택에서 peek을 수행할 수 없습니다.")
        return self._elements[-1]

    def is_empty(self):
        """스택이 비어있는지 확인합니다."""
        return len(self._elements) == 0

    def __str__(self):
        return f"Stack(bottom -> top): {self._elements}"

# 사용 예시
my_stack = Stack()
my_stack.push(10)
my_stack.push(20)
my_stack.push(30)
print(my_stack)         # Stack(bottom -> top): [10, 20, 30]
print(my_stack.pop())   # 30 반환