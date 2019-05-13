from math import sin
from functools import wraps
from concurrent.futures.thread import ThreadPoolExecutor
import jacobi
import gauss
import gauss_seidel

_DEFAULT_POOL = ThreadPoolExecutor()


def mt_wrapper(f, executor=None):
    """
    Funkcja zaczerpnięta ze StackOverflow dla
    uproszczenia kodu przy przyspieszeniu obliczeń.
    """
    @wraps(f)
    def wrap(*args, **kwargs):
        return (executor or _DEFAULT_POOL).submit(f, *args, **kwargs)

    return wrap


def prepare():
    n = 998
    a1 = 11.0
    a2 = -1.0
    a3 = -1.0

    matrix = [[0.0]*n]*n

    # matrix = []
    # for i in range(n):
    #     row = [0.0]*n
    #     matrix.append(row)

    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = a1
            elif i == j - 1 or i == j + 1 or i == j - 2 or i == j + 2:  # a2 == a3
                matrix[i][j] = a2

    b = [0.0]*n
    for i in range(n):
        b[i] = sin(i)

    return matrix, b


def main():
    matrix, b = prepare()
    cutoff = pow(10.0, -9)


if __name__ == "__main__":
    main()
