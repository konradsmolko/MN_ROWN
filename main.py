from timeit import default_timer as timer
from matrix_calc import prepare, jacobi, gauss_seidel, gauss, mul_mat_by_vec
import simplejson

_debug = False
_test = False


def policz(n, a1, a2, a3, zad) -> (float, float):
    matrix, b = prepare(n, a1, a2, a3)
    cutoff = 10.0 ** -9

    if zad is not None:
        print("Zaczynam zadanie", zad)
    print("Start Jacobi, n =", n)
    start = timer()
    jr, jiters = jacobi(matrix, b, n, cutoff)
    end = timer()
    jtime = end - start

    print("Start Gauss-Seidel, n =", n)
    start = timer()
    gsr, gsiters = gauss_seidel(matrix, b, n, cutoff)
    end = timer()
    gstime = end - start

    if _debug:
        print("Czas dla metody Jacobiego:", jtime)
        print("Iteracje Jacobi:", jiters)
        print("jr:", jr)
    if _test and jr is not None:
        jtest = mul_mat_by_vec(matrix, jr)
        print("test jacobi", jtest)

    if _debug:
        print("Czas dla metody Gaussa-Seidla:", gstime)
        print("Iteracje Gauss-Seidel:", gsiters)
        print("gsr:", gsr)
    if _test and gsr is not None:
        gstest = mul_mat_by_vec(matrix, gsr)
        print("test jacobi", gstest)

    return jtime, gstime


def zadanie_d(n, a1, a2, a3) -> float:
    matrix, b = prepare(n, a1, a2, a3)

    if _debug:
        print("Zaczynam zadanie D")
    print("Start LU, n =", n)
    start = timer()
    lur, error = gauss(matrix, b)
    end = timer()
    lutime = end - start
    if _debug:
        print("Czas dla metody faktoryzacji LU:", lutime)
        print("norm(res):", error)
        print("r:", lur)
    if _test:
        lutest = mul_mat_by_vec(matrix, lur)
        print("test LU", lutest)

    return lutime


def zadanie_e(a1, a2, a3):
    ns = [100, 500, 1000, 2000, 3000]
    times_jacobi = []
    times_gs = []
    times_lu = []
    for n in ns:
        print("Zaczynam iterację pętli. n =", n)
        jtime, gstime = policz(n, a1, a2, a3, None)
        lutime = zadanie_d(n, a1, a2, a3)
        print("Czasy:", jtime, gstime, lutime)
        times_jacobi.append(jtime)
        times_gs.append(gstime)
        times_lu.append(lutime)

    return times_jacobi, times_gs, times_lu


def main():
    n = 998
    a1 = 11.0
    a2 = -1.0
    a3 = -1.0
    # policz(n, a1, a2, a3, "A i B")
    # policz(n, 3.0, a2, a3, "C")
    # zadanie_d(n, 3.0, a2, a3)
    times_jacobi, times_gs, times_lu = zadanie_e(a1, a2, a3)
    print(times_jacobi)
    print(times_gs)
    print(times_lu)
    with open("jacobi.txt", 'w') as file:
        simplejson.dump(times_jacobi, file)
    with open("gs.txt", 'w') as file:
        simplejson.dump(times_gs, file)
    with open("lu.txt", 'w') as file:
        simplejson.dump(times_lu, file)


if __name__ == "__main__":
    main()
