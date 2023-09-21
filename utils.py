import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity

def JSV_Gaussian_u(Fea, len_s, n_components):
    """compute one v-div with GMM value"""
    X = Fea[0:len_s, :] # fetch the sample 1
    Y = Fea[len_s:, :] # fetch the sample 2
    ind = np.random.choice(Fea.shape[0], len_s, replace=False)
    XY = Fea[ind]

    jsv = L_Gaussian(0,Fea,X,Y,n_components)

    return jsv

def JSV_Gaussian(Fea, N_per, N1, alpha, n_components):
    """run permutation test with v-div with GMM"""
    jsv_vector = np.zeros(N_per)
    jsv_value = JSV_Gaussian_u(Fea, N1, n_components)
    count = 0
    nxy = Fea.shape[0]
    nx = N1
    for r in range(N_per):
        # print r
        ind = np.random.choice(nxy, nxy, replace=False)
        # divide into new X, Y
        indx = ind[:nx]
        # print(indx)
        indy = ind[nx:]
        Kx = Fea[indx]
        # print(Kx)
        Ky = Fea[indy]
        indxy = np.concatenate((indx[:int(nx/2)], indy[:int(nx/2)]))
        Kxy = Fea[indxy]

        jsv_r = L_Gaussian(r + 1,Fea,Kx,Ky,n_components)

        jsv_vector[r] = jsv_r

        if jsv_vector[r] > jsv_value:
            count = count + 1
        if count > np.ceil(N_per * alpha):
            h = 0
            threshold = "NaN"
            break
        else:
            h = 1
    if h == 1:
        S_jsv_vector = np.sort(jsv_vector)
        threshold = S_jsv_vector[int(np.ceil(N_per * (1 - alpha)))]
    return h, threshold, jsv_value

def JSV_KDE_u(Fea, len_s, bandwidth):
    """compute value of deep-kernel MMD and std of deep-kernel MMD using merged data."""
    X = Fea[0:len_s] # fetch the sample 1
    Y = Fea[len_s:] # fetch the sample 2
    ind = np.random.choice(Fea.shape[0], len_s, replace=False)
    XY = Fea[ind]
    model_x = KernelDensity(bandwidth=bandwidth[0])
    model_y = KernelDensity(bandwidth=bandwidth[1])
    model_xy = KernelDensity(bandwidth=bandwidth[2])
    model_x.fit(X)
    model_y.fit(Y)
    model_xy.fit(XY)


    jsv = L_KDE(0,Fea,X,Y,bandwidth)
    return jsv

def JSV_KDE(Fea, N_per, N1, alpha, bandwidth):
    """run two-sample test (TST) using ordinary Gaussian kernel."""
    jsv_vector = np.zeros(N_per)
    jsv_value = JSV_KDE_u(Fea, N1, bandwidth)
    count = 0
    nxy = Fea.shape[0]
    nx = N1
    for r in range(N_per):
        # print r
        ind = np.random.choice(nxy, nxy, replace=False)
        # divide into new X, Y
        indx = ind[:nx]
        # print(indx)
        indy = ind[nx:]
        Kx = Fea[indx]
        # print(Kx)
        Ky = Fea[indy]
        indxy = np.concatenate((indx[:int(nx/2)], indy[:int(nx/2)]))
        Kxy = Fea[indxy]

        model_x = KernelDensity(bandwidth=bandwidth[0])
        model_y = KernelDensity(bandwidth=bandwidth[1])
        model_xy = KernelDensity(bandwidth=bandwidth[2])
        model_x.fit(Kx)
        model_y.fit(Ky)
        model_xy.fit(Kxy)

        jsv_r = L_KDE(r + 1,Fea,Kx,Ky,bandwidth)

        jsv_vector[r] = jsv_r

    S_jsv_vector = np.sort(jsv_vector)
    threshold = S_jsv_vector[int(np.ceil(N_per * (1 - alpha)))]
    h = 0
    if jsv_value > threshold:
        h = 1
    return h, threshold, jsv_value

def L_KDE(r,Fea,x,y,bandwidth):

    model_x = KernelDensity(bandwidth=bandwidth[0])
    model_y = KernelDensity(bandwidth=bandwidth[1])
    model_xy = KernelDensity(bandwidth=bandwidth[2])

    model_x.fit(x)
    model_y.fit(y)
    model_xy.fit(Fea)

    mixed = 1/2*np.mean(-model_xy.score_samples(x))+1/2*np.mean(-model_xy.score_samples(y))
    x_prob = np.mean(-model_y.score_samples(x))
    y_prob = np.mean(-model_x.score_samples(y))
    gap = abs(x_prob - mixed) + abs(y_prob - mixed)
    
    return abs(gap)


def L_Gaussian(r,Fea,x,y, n_components):

    model_x = GaussianMixture(n_components=n_components)
    model_y = GaussianMixture(n_components=n_components)
    model_xy = GaussianMixture(n_components=n_components)

    model_x.fit(x)
    model_y.fit(y)
    model_xy.fit(Fea)

    mixed = 1/2*np.mean(-model_xy.score_samples(x))+1/2*np.mean(-model_xy.score_samples(y))
    x_prob = np.mean(-model_y.score_samples(x))
    y_prob = np.mean(-model_x.score_samples(y))
    gap = abs(x_prob - mixed) + abs(y_prob - mixed)
    
    return abs(gap)
