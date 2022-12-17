import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def compute_inpca(mat):
    """
    mat: Matrix to compute the in_pca on
    Args:
        mat:  np.array of shape (num_models x samples x classes)
    Return:
        (singular_val, imaginary): Singular values and indicatory if singular
                                   value is imaginary
        projection               : The projections onto each eigen-vector scaled
                                   by singular values. projection[:, i] gives the
                                   ith vector.
    """
    mat = np.sqrt(mat) + 1e-9
    mat1 = np.transpose(mat, axes=[1, 0, 2])
    mat2 = np.transpose(mat, axes=[1, 2, 0])

    Lmat = 0.0
    dim = len(mat1)
    batch = 500

    for i in range(0, dim, batch):
        print(i)
        Lmat += (np.log(mat1[i:i+batch] @ mat2[i:i+batch])).sum(0)


    # Normalization
    Lmat = Lmat / dim

    ldim = Lmat.shape[0]
    Pmat = np.eye(ldim) - 1.0/ ldim
    Wmat = Pmat @ Lmat @ Pmat

    # Factor of 2 adjustment
    Wmat = Wmat / 2

    eigenval, eigenvec = np.linalg.eig(Wmat)

    #Sort eigen-values by magnitude
    sort_ind = np.argsort(-np.abs(eigenval))
    eigenval = eigenval[sort_ind]
    eigenvec = eigenvec[:, sort_ind]

    # Find projections
    singular_val = np.sqrt(np.abs(eigenval))
    imaginary = np.array(eigenval < 0.0)
    projection = eigenvec * singular_val.reshape(1, -1)

    return (singular_val, imaginary, eigenvec), projection

def reshape_comps(rep, rshapes, dynamic):
    """
    * rshapes: array of [(seed, epochs, samples, classes)]
    * rep: numpy array with shape (models x 5)
    """

    if not dynamic:
        return rep.reshape(*rshapes[:-2], 5)

    all_reps = []
    start = 0

    for shape in rshapes:

        nmods = shape[0] * shape[1]
        rslice = rep[start:start+nmods]
        start += nmods

        all_reps.append(rslice.reshape(*shape[0:2], -1))

    return all_reps


def reduce_rep(rep, inpca=True, dynamic_shape=False):
    """
    Use this function if all models have same number of seeds and epochs
    """
    # array of [seeds x epochs x samples x classes]
    if dynamic_shape:
        rshapes = [l.shape for l in rep]
        rep = [l.reshape(-1, *l.shape[2:]) for l in rep]
        rep = np.concatenate(rep, axis=0)
    else:
        rep = np.stack(rep, axis=0)
        rshapes = rep.shape
        rep = rep.reshape(-1, *rep.shape[-2:])

    if inpca:
        singval, prin_comps = compute_inpca(rep)
        prin_comps = prin_comps[:, 0:5]
        prin_comps = reshape_comps(prin_comps, rshapes, dynamic_shape)

        obj = {
            "singular_values": singval[0],
            "imaginary": singval[1],
            "evecs": singval[2],
            "inpca": prin_comps
        }

    else:
        rep = rep.reshape(rep.shape[0], -1)
        pca = PCA()
        pca.fit(rep)
        prin_comps = pca.transform(rep)[:, :5]
        prin_comps = reshape_comps(prin_comps, rshapes, dynamic_shape)
        eig = pca.singular_values_

        obj = {
            "eigen_values": eig,
            "pca": prin_comps
        }

    return obj



def test_inpca():
    test_mat = np.random.uniform(low=0.0, high=1.0, size=(10, 100, 3))
    compute_inpca(test_mat)


if __name__ == '__main__':
    test_inpca()

