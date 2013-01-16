from __future__ import division
import numpy as np


def MMD2u(K, m, n):
    """The MMD^2_u unbiased statistic.
    """
    Kx = K[:m, :m]
    Ky = K[m:, m:]
    Kxy = K[:m, m:]
    return 1.0 / (m * (m - 1.0)) * (Kx.sum() - Kx.diagonal().sum()) + \
           1.0 / (n * (n - 1.0)) * (Ky.sum() - Ky.diagonal().sum()) - \
           2.0 / (m * n) * (Kxy.sum() - Kxy.diagonal().sum())


def compute_null_distribution(K, m, n, iterations, verbose=False):
    """Compute the bootstrap null-distribution of MMD2u.
    """
    mmd2u_null = np.zeros(iterations)
    for i in range(iterations):
        if verbose and (i % 1000) == 0:
            print i,
            stdout.flush()
        idx = np.random.permutation(m+n)
        K_i = K[idx, idx[:,None]]
        mmd2u_null[i] = MMD2u(K_i, m, n)
        
    if verbose: print
    return mmd2u_null


def compute_upper_B_10(p):
    """Compute a upper bound on the Bayes factor of H_1 versus H_0
    given a p-value 'p'.

    See 'Calibration of p Values for Testing Precise Null Hypotheses'
    by Sellke et al. 2001.
    """
    return -1.0 / (np.e * p * np.log(p))


def compute_lower_alpha(p):
    """
    See 'Calibration of p Values for Testing Precise Null Hypotheses'
    by Sellke et al. 2001.
    """
    return 1.0 / (1.0 + 1.0 / ( -np.e * p * np.log(p)))


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from sys import stdout

    np.random.seed(0)

    m = 20
    n = 20
    d = 2

    sigma2X = np.eye(d)
    muX = np.zeros(d)

    sigma2Y = np.eye(d)
    muY = np.ones(d)
    # muY = np.zeros(d)

    iterations = 10000

    X = np.random.multivariate_normal(mean=muX, cov=sigma2X, size=m)
    Y = np.random.multivariate_normal(mean=muY, cov=sigma2Y, size=n)

    if d == 2:
        plt.figure()
        plt.plot(X[:,0], X[:,1], 'bo')
        plt.plot(Y[:,0], Y[:,1], 'rx')

    XY = np.vstack([X, Y])
    K = np.dot(XY, XY.T)

    mmd2u = MMD2u(K, m, n)

    print "MMD^2_u =", mmd2u

    print "Computing the null distribution."

    mmd2u_null = compute_null_distribution(K, m, n, iterations)

    plt.figure()
    plt.hist(mmd2u_null, bins=50, normed=True)

    p_value = max(1.0/iterations, (mmd2u_null > mmd2u).sum() / float(iterations))

    print "p-value ~=", p_value, "\t (resolution :", 1.0/iterations, ")"
