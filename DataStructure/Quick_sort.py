def quick_sort(arr):
    # [유한성을 보장하는 핵심: 종료 조건(Base Case)]
    # 배열의 길이가 1 이하이면, 더 이상 쪼갤 필요 없이 정렬이 끝난 것이므로 그대로 반환합니다.
    if len(arr) <= 1:
        return arr

    # 기준점(pivot) 설정
    pivot = arr[len(arr) // 2]

    # 기준점보다 작은 수, 같은 수, 큰 수로 배열을 분할 (문제의 크기가 점점 작아짐)
    lesser_arr = [x for x in arr if x < pivot]
    equal_arr = [x for x in arr if x == pivot]
    greater_arr = [x for x in arr if x > pivot]

    # 분할된 작은 배열들에 대해 다시 재귀 호출
    return quick_sort(lesser_arr) + equal_arr + quick_sort(greater_arr)


# 실행 테스트
numbers = [3, 6, 8, 10, 1, 2, 1]
print(quick_sort(numbers))
# 출력: [1, 1, 2, 3, 6, 8, 10]