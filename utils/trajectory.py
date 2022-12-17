import numpy as np
from scipy import optimize
from functools import partial


def project(traj, start, end):
    """
    Compute the λ (progress) for each model in the trajectory
    Args
        start: Starting point on the circle
        end:   Final point on the circle
        traj:  Matrix of size (epochs x samples x classes) depicting trajectory taken
               during training
    Return
        Vector of λ (progress) for each point on the trajectory.
        It lies in [0, 1] indicating if point is closer/further from reference
        geodesic.
    """
    def dB(t, n=0):
        vec = interpolate(start, end, t)
        dist = (-np.log((vec * traj[n]).sum(-1))).sum()
        return dist

    lam = []
    for ep in range(len(traj)):
        dn = partial(dB, n=ep)
        l = optimize.minimize_scalar(dn, bounds=(0, 1), method='bounded').x
        lam.append(float(l))

    return lam


def interpolate(start, end, t):
    """
    Interpolate between "start" and "end" using geodesic equation
    """
    cospq = (start * end).sum(-1, keepdims=True)
    dg = np.arccos(np.clip(cospq, 0, 1))

    # Use masks, incase, start and end are identical
    mask = (dg <= 1e-6).reshape(-1)
    gamma = np.array(start)
    gamma[~mask] = np.sin((1-t)* dg[~mask]) * start[~mask] + \
                   np.sin(t    * dg[~mask]) * end[~mask]
    gamma[~mask] = gamma[~mask] / np.sin(dg[~mask])

    return gamma


def sample_trajectory(traj, λ_true, λ_sample):
    """
    Sample from the parameterized trajectory
    """
    ind = 0
    max_ind = len(λ_true)

    samples = []

    for λ in λ_sample:

        # Find first λ_true (from left) bigger than λ
        while ind < max_ind:
            if λ_true[ind] > λ:
                break;
            ind += 1

        # Draw sample
        if ind == max_ind:
            sample = traj[-1]
        elif ind == 0:
            sample = traj[0]
        else:
            seg_start = traj[ind - 1]
            seg_end = traj[ind]

            λ_start = λ_true[ind-1]
            λ_end = λ_true[ind]
            assert(λ_start <= λ and λ_end >= λ)

            λ_segment = (λ - λ_start) / (λ_end - λ_start)
            λ_segment = np.clip(λ_segment, 0, 1)
            sample = interpolate(seg_start, seg_end, λ_segment)

        samples.append(sample)

    samples = np.array(samples)
    return samples


def uniform_sample(traj, start, end, num=50):
    λ_true = project(traj, start, end)

    λmin, λmax = min(λ_true), max(λ_true)
    λ_sample = np.arange(λmin, λmax, (λmax - λmin)/num)
    sample_traj = sample_trajectory(traj, λ_true, λ_sample)
    return sample_traj


def test_sphere_example():
    # 1 point / 3-way problem
    start = np.sqrt(np.array([[1/3., 1/3., 1/3.]]))
    end = np.sqrt(np.array([[1.0, 0.0, 0.0]]))

    # Training trajectory
    xt = np.sqrt(np.array([0.33, 0.35, 0.4, 0.50, 0.70, 0.90, 1]))
    yt = np.sqrt(np.array([0.33, 0.40, 0.5, 0.45, 0.25, 0.07, 0]))
    zt = np.sqrt(np.array([0.33, 0.25, 0.1, 0.05, 0.05, 0.03, 0]))

    train_traj = (np.array([xt, yt, zt]).T).reshape(-1, 1, 3)

    # Training trajectory
    l_true = project(train_traj, start, end)
    lmin = min(l_true)
    lmax = max(l_true)
    print(l_true)
    l_sample = np.arange(lmin, lmax, (lmax - lmin)/50)
    sampled_traj = sample_trajectory(train_traj, l_true, l_sample, start, end)
    proj_back = project(sampled_traj, start, end)

