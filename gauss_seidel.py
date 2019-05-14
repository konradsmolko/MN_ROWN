from matrix_calc import norm_res
from main import mt_wrapper


@mt_wrapper
def gauss_seidel(matrix, b, n, cutoff) -> [list, int]:
    iterations = 0
    r = [1.0] * n

    while norm_res(matrix, n, r, b) > cutoff:
        iterations += 1
        if iterations > 10000:
            return None, None

        for i in range(n):
            o = 0
            for j in range(n):
                if i != j:
                    o += matrix[i][j] * r[j]

            r[i] = (b[i] - o) / matrix[i][i]

    return r, iterations
