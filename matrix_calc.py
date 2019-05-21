from math import sqrt, sin


def mul_mat_by_mat(a, b):
    # ma, na = len(a), len(a[0])
    # mb, nb = len(b), len(b[0])
    #
    # if na != mb:
    #     raise ValueError("Macierze nie mogą być przemnożone!")
    #
    # ret = [[0.0 for _ in range(nb)] for _ in range(ma)]
    # # ret = []
    # # for i in range(ma):
    # #     row = [0.0] * nb
    # #     ret.append(row)
    #
    # for i in range(ma):
    #     for j in range(nb):
    #         for k in range(mb):
    #             ret[i][j] += a[i][j] * b[k][j]
    #
    # return ret
    # # podejrzewam że powyższa metoda źle działała
    return [[sum(el_m * el_n for el_m, el_n in zip(row_m, col_n)) for col_n in zip(*b)] for row_m in a]


def pivot_mat(matrix):
    #  Metoda zaczerpnięta z internetu.
    #  https://www.quantstart.com/articles/LU-Decomposition-in-Python-and-NumPy
    m = len(matrix)
    id_mat = [[float(i == j) for i in range(m)] for j in range(m)]
    for j in range(m):
        row = max(range(j, m), key=lambda i: abs(matrix[i][j]))
        if j != row:
            id_mat[j], id_mat[row] = id_mat[row], id_mat[j]

    return id_mat


def lu_decomp(matrix):
    #  Metoda zaczerpnięta z internetu.
    #  https://www.quantstart.com/articles/LU-Decomposition-in-Python-and-NumPy
    n = len(matrix)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]
    P = pivot_mat(matrix)
    PA = mul_mat_by_mat(P, matrix)
    for j in range(n):
        L[j][j] = 1.0
        for i in range(j + 1):
            s1 = sum(U[k][j] * L[i][k] for k in range(i))
            U[i][j] = PA[i][j] - s1

        for i in range(j, n):
            s2 = sum(U[k][j] * L[i][k] for k in range(j))
            L[i][j] = (PA[i][j] - s2) / U[j][j]

    return P, L, U


def dot_product(v1, v2) -> [float]:
    return sum([i * j for (i, j) in zip(v1, v2)])


def back_sub(upper, b) -> [float]:
    n = len(upper)
    r = [0.0 for _ in b]
    for i in range(n-1, 0, -1):
        r[i] = (b[i] - dot_product(upper[i], r)) / float(upper[i][i])

    return r


def forward_sub(lower, b) -> [float]:
    n = len(lower)
    r = [0.0 for _ in b]
    for i in range(n):
        r[i] = (b[i] - dot_product(lower[i], r))/lower[i][i]

    return r


def mul_mat_by_vec(a, b) -> [[float]]:
    result = [0.0 for _ in range(len(a))]

    for i in range(len(a)):
        for k in range(len(b)):
            result[i] += a[i][k] * b[k]

    return result


def sub_vectors(a, b) -> [float]:
    return [a[i] - b[i] for i in range(len(a))]


def norm_res(matrix, r, b) -> float:
    # norm(M*r - b)
    # norm = sqrt(sum(|v|[k]^2)
    # matrix nxn
    # r nx1
    # b nx1
    # v = m*r - b; nx1
    res = sub_vectors(mul_mat_by_vec(matrix, r), b)
    ret = sqrt(sum([e ** 2 for e in res]))  # norm

    return ret


def prepare(n, a1, a2, a3) -> ([[float]], [float]):
    matrix = [[0.0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = a1
            elif i == j - 1 or i == j + 1:
                matrix[i][j] = a2
            elif i == j - 2 or i == j + 2:
                matrix[i][j] = a3

    b = [sin(i) for i in range(n)]

    return matrix, b


def jacobi(matrix, b, n, cutoff) -> [list, int]:
    iterations = 0
    r = [1.0 for _ in range(n)]
    x = [1.0 for _ in range(n)]
    try:
        for iterations in range(100):
            accuracy = norm_res(matrix, r, b)
            if accuracy <= cutoff:
                return r, iterations

            r = [x[i] for i in range(n)]

            for i in range(n):
                o = 0
                for j in range(n):
                    if i != j:
                        o += matrix[i][j] * r[j]

                x[i] = (b[i] - o)/matrix[i][i]

    except OverflowError:
        print("ERROR: OverflowError in Jacobi method, terminating.")
        return None, iterations

    print("ERROR: too many iterations in Jacobi method, terminating.")
    print("Current r:", r)
    return None, iterations


def gauss_seidel(matrix, b, n, cutoff) -> [list, int]:
    iterations = 0
    r = [1.0 for _ in range(n)]

    try:
        for iterations in range(100):
            accuracy = norm_res(matrix, r, b)
            if accuracy <= cutoff:
                return r, iterations

            for i in range(n):
                o = 0
                for j in range(n):
                    if i != j:
                        o += matrix[i][j] * r[j]

                r[i] = (b[i] - o) / matrix[i][i]

    except OverflowError:
        print("ERROR: OverflowError in Jacobi method, terminating.")
        return None, iterations

    print("ERROR: too many iterations in Gauss-Seidel method, terminating.")
    print("Current r:", r)
    return None, iterations


def gauss(matrix, b) -> ([float], float):
    #  Ax = b
    #  PA = LU
    #  LUx = Pb
    #  Ly = Pb
    #  Ux = y
    pivot, lower, upper = lu_decomp(matrix)
    y = back_sub(upper, b)
    x = forward_sub(lower, y)
    return x, norm_res(matrix, b, x)
