# _*_coding:utf-8_*_

import math


def log_sum(log_a, log_b):
    if log_a < log_b:
        v = log_b + math.log(1 + math.exp(log_a - log_b))
    else:
        v = log_a + math.log(1 + math.exp(log_b - log_a))
    return v


def trigamma(x):
    x = x + 6
    p = 1./(x*x)
    p = (((((0.075757575757576*p-0.033333333333333)*p+0.0238095238095238) *
           p-0.033333333333333)*p+0.166666666666667)*p+1)/x+0.5*p
    for i in range(6):
        x -= 1
        p = 1./(x*x) + p
    return p


def digamma(x):
    x = x + 6
    p = 1 / (x * x)
    p = (((0.004166666666667 * p - 0.003968253986254) * p +
          0.008333333333333) * p - 0.083333333333333) * p
    p = p + math.log(x) - 0.5 / x - 1 / (x - 1) - 1 / (x - 2) - 1 / (x - 3) - 1 / (x - 4) - 1 / (x - 5) - 1 / (x - 6);
    return p


def log_gamma(x):
    z = 1 / (x * x)

    x = x + 6
    z = (((-0.000595238095238 * z + 0.000793650793651)
          * z - 0.002777777777778) * z + 0.083333333333333) / x
    z = (x - 0.5) * math.log(x) - x + 0.918938533204673 + z - math.log(x - 1) -\
        math.log(x - 2) - math.log(x - 3) - math.log(x - 4) - math.log(x - 5) - math.log(x - 6)
    return z


def alhood(a, ss, D, K):
    return D * (log_gamma(K * a) - K * log_gamma(a)) + (a - 1) * ss


def d_alhood(a, ss, D, K):
    return D * (K * digamma(K * a) - K * digamma(a)) + ss


def d2_alhood(a, D, K):
    return D * (K * K * trigamma(K * a) - K * trigamma(a))


def opt_alpha(ss, D, K):
    """
    newtons method
    """
    init_a = 100
    max_iter = 100
    log_a = math.log(init_a)
    for i in range(max_iter):
        a = math.exp(log_a)
        df1 = d_alhood(a, ss, D, K)
        df2 = d2_alhood(a, D, K)
        log_a = log_a - df1 / (df2 * a +df1)

    return math.exp(log_a)
