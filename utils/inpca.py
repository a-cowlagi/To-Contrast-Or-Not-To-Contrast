import numpy as np


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

    return (singular_val, imaginary), projection


def test_inpca():
    test_mat = np.random.uniform(low=0.0, high=1.0, size=(10, 100, 3))
    compute_inpca(test_mat)


if __name__ == '__main__':
    test_inpca()

