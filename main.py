from math import sin
from functools import wraps
from concurrent.futures.thread import ThreadPoolExecutor
from jacobi import jacobi
from gauss_seidel import gauss_seidel

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


def prepare(n=998, a1=11.0, a2=-1.0, a3=-1.0):
    matrix = [[0.0]*n]*n

    # Mogłoby być bardziej zoptymalizowane...
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = a1
            elif i == j - 1 or i == j + 1:
                matrix[i][j] = a2
            elif i == j - 2 or i == j + 2:
                matrix[i][j] = a3

    b = [0.0]*n
    for i in range(n):
        b[i] = sin(i)

    return matrix, b


def main():
    n = 998
    matrix, b = prepare(n)
    cutoff = 10.0**-9
    jr, ji = jacobi(matrix, b, n, cutoff)
    gsr, gsi = gauss_seidel(matrix, b, n, cutoff)


if __name__ == "__main__":
    main()
