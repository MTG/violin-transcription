import numpy as np
from typing import List
from numba import jit
import numpy as np
from scipy import signal
from typing import Tuple


from .core import compute_warping_path
from .cost import *



def compute_optimal_chroma_shift(f_chroma1: np.ndarray,
                                 f_chroma2: np.ndarray,
                                 chroma_transpositions: np.ndarray = np.arange(0, 12),
                                 step_sizes: np.ndarray = np.array([[1, 0], [0, 1], [1, 1]], int),
                                 step_weights: np.ndarray = np.array([1.0, 1.0, 1.0], np.float64)) -> int:
    """Computes the optimal chroma shift which minimizes the DTW cost.

    Parameters
    ----------
    f_chroma1 : np.ndarray [shape=(d_chroma, N_chroma)]
        First chroma vector

    f_chroma2 : np.ndarray [shape=(d_chroma, N_chroma)]
        Second chroma vector

    step_sizes : np.ndarray
        DTW step sizes (default: np.array([[1, 0], [0, 1], [1, 1]]))

    step_weights : np.ndarray
        DTW step weights (default: np.array([1.0, 1.0, 1.0]))

    chroma_transpositions : np.ndarray
        Array of chroma shifts (default: np.arange(0, 11))

    Returns
    -------
    opt_chroma_shift : int
        Optimal chroma shift which minimizes the DTW cost.
    """
    if f_chroma2.shape[1] >= 9000 or f_chroma1.shape[1] >= 9000:
        print("Warning: You are attempting to find the optimal chroma shift on sequences of length >= 9000. "
              "This involves full DTW computation. You'll probably want to smooth and downsample your sequences to a"
              " lower feature resolution before doing this.")
    opt_chroma_shift = 0
    dtw_cost = np.inf
    for chroma_shift in chroma_transpositions:
        cost_matrix_tmp = cosine_distance(f_chroma1, shift_chroma_vectors(f_chroma2, chroma_shift))
        D, _, _ = compute_warping_path(cost_matrix_tmp, step_sizes=step_sizes, step_weights=step_weights)
        if D[-1, -1] < dtw_cost:
            dtw_cost = D[-1, -1]
            opt_chroma_shift = chroma_shift

    return opt_chroma_shift


def compute_warping_paths_from_cost_matrices(cost_matrices: List,
                                             step_sizes: np.array = np.array([[1, 0], [0, 1], [1, 1]], int),
                                             step_weights: np.array = np.array([1.0, 1.0, 1.0], np.float64),
                                             implementation: str = 'synctoolbox') -> List:
    """Computes a path via DTW on each matrix in cost_matrices

    Parameters
    ----------
    cost_matrices : list
        List of cost matrices

    step_sizes : np.ndarray
        DTW step sizes (default: np.array([[1, 0], [0, 1], [1, 1]]))

    step_weights : np.ndarray
        DTW step weights (default: np.array([1.0, 1.0, 1.0]))

    implementation : str
        Choose among 'synctoolbox' and 'librosa' (default: 'synctoolbox')

    Returns
    -------
    wp_list : list
        List of warping paths
    """
    return [compute_warping_path(C=C,
                                 step_sizes=step_sizes,
                                 step_weights=step_weights,
                                 implementation=implementation)[2] for C in cost_matrices]


def compute_cost_matrices_between_anchors(f_chroma1: np.ndarray,
                                          f_chroma2: np.ndarray,
                                          anchors: np.ndarray,
                                          f_onset1: np.ndarray = None,
                                          f_onset2: np.ndarray = None,
                                          alpha: float = 0.5) -> List:
    """Computes cost matrices for the given features between subsequent
    pairs of anchors points.

    Parameters
    ----------
    f_chroma1 : np.ndarray [shape=(12, N)]
        Chroma feature matrix of the first sequence

    f_chroma2 : np.ndarray [shape=(12, M)]
        Chroma feature matrix of the second sequence

    anchors : np.ndarray [shape=(2, R)]
        Anchor sequence

    f_onset1 : np.ndarray [shape=(L, N)]
        Onset feature matrix of the first sequence

    f_onset2 : np.ndarray [shape=(L, M)]
        Onset feature matrix of the second sequence

    alpha: float
        Alpha parameter to weight the cost functions.

    Returns
    -------
    cost_matrices: list
        List containing cost matrices
    """
    high_res = False
    if f_onset1 is not None and f_onset2 is not None:
        high_res = True

    cost_matrices = list()
    for k in range(anchors.shape[1] - 1):
        a1 = np.array(anchors[:, k].astype(int), copy=True)
        a2 = np.array(anchors[:, k + 1].astype(int), copy=True)

        if high_res:
            cost_matrices.append(compute_high_res_cost_matrix(f_chroma1[:, a1[0]: a2[0] + 1],
                                                              f_chroma2[:, a1[1]: a2[1] + 1],
                                                              f_onset1[:, a1[0]: a2[0] + 1],
                                                              f_onset2[:, a1[1]: a2[1] + 1],
                                                              weights=np.array([alpha, 1-alpha])))
        else:
            cost_matrices.append(cosine_distance(f_chroma1[:, a1[0]: a2[0] + 1],
                                                 f_chroma2[:, a1[1]: a2[1] + 1]))
    return cost_matrices


def build_path_from_warping_paths(warping_paths: List,
                                  anchors: np.ndarray = None) -> np.ndarray:
    """The function builds a path from a given list of warping paths
    and the anchors used to obtain these paths. The indices of the original
    warping paths are adapted such that they cross the anchors.

    Parameters
    ----------
    warping_paths : list
        List of warping paths

    anchors : np.ndarray [shape=(2, N)]
        Anchor sequence

    Returns
    -------
    path : np.ndarray [shape=(2, M)]
        Merged path
    """

    if anchors is None:
        # When no anchor points are given, we can construct them from the
        # subpaths in the wp_list

        # To do this, we assume that the first path's element is the starting
        # anchor
        anchors = warping_paths[0][:, 0]

        # Retrieve the last element of each path
        anchors_tmp = np.zeros(len(warping_paths), np.float32)
        for idx, x in enumerate(warping_paths):
            anchors_tmp[idx] = x[:, -1]

        # Correct indices, such that the indices of the anchors are given on a
        # common path. Each anchor a_l = [Nnew_[l+1];Mnew_[l+1]]
        #    Nnew_[l+1] = N_l + N_[l+1] -1
        #    Mnew_[l+1] = M_l + M_[l+1] -1

        anchors_tmp = np.cumsum(anchors_tmp, axis=1)
        anchors_tmp[:, 1:] = anchors_tmp[:, 1:] - [np.arange(1, anchors_tmp.shape[1]),
                                                   np.arange(1, anchors_tmp.shape[1])]

        anchors = np.concatenate([anchors, anchors_tmp], axis=1)

    L = len(warping_paths) + 1
    path = None
    wp = None

    for anchor_idx in range(1, L):
        anchor1 = anchors[:, anchor_idx - 1]
        anchor2 = anchors[:, anchor_idx]

        wp = np.array(warping_paths[anchor_idx - 1], copy=True)

        # correct indices in warpingPath
        wp += np.repeat(anchor1.reshape(-1, 1), wp.shape[1], axis=1).astype(wp.dtype)

        # consistency checks
        assert np.array_equal(wp[:, 0], anchor1), 'First entry of warping path does not coincide with anchor point'
        assert np.array_equal(wp[:, -1], anchor2), 'Last entry of warping path does not coincide with anchor point'

        if path is None:
            path = np.array(wp[:, :-1], copy=True)
        else:
            path = np.concatenate([path, wp[:, :-1]], axis=1)

    # append last index of warping path
    path = np.concatenate([path, wp[:, -1].reshape(-1, 1)], axis=1)

    return path


def find_anchor_indices_in_warping_path(warping_path: np.ndarray,
                                        anchors: np.ndarray) -> np.ndarray:
    """Compute the indices in the warping path that corresponds
    to the elements in 'anchors'

    Parameters
    ----------
    warping_path : np.ndarray [shape=(2, N)]
        Warping path

    anchors : np.ndarray [shape=(2, M)]
        Anchor sequence

    Returns
    -------
    indices : np.ndarray [shape=(2, M)]
        Anchor indices in the ``warping_path``
    """
    indices = np.zeros(anchors.shape[1])

    for k in range(anchors.shape[1]):
        a = anchors[:, k]
        indices[k] = np.where((a[0] == warping_path[0, :]) & (a[1] == warping_path[1, :]))[0]

    return indices


def make_path_strictly_monotonic(P: np.ndarray) -> np.ndarray:
    """Compute strict alignment path from a warping path

    Wrapper around "compute_strict_alignment_path_mask" from libfmp.

    Parameters
    ----------
    P: np.ndarray [shape=(2, N)]
        Warping path

    Returns
    -------
    P_mod: np.ndarray [shape=(2, M)]
        Strict alignment path, M <= N
    """
    P_mod = compute_strict_alignment_path_mask(P.T)

    return P_mod.T

def compute_strict_alignment_path_mask(P):
    """Compute strict alignment path from a warping path

    Notebook: C3/C3S3_MusicAppTempoCurve.ipynb

    Args:
        P (list or np.ndarray): Wapring path

    Returns:
        P_mod (list or np.ndarray): Strict alignment path
    """
    P = np.array(P, copy=True)
    N, M = P[-1]
    # Get indices for strict monotonicity
    keep_mask = (P[1:, 0] > P[:-1, 0]) & (P[1:, 1] > P[:-1, 1])
    # Add first index to enforce start boundary condition
    keep_mask = np.concatenate(([True], keep_mask))
    # Remove all indices for of last row or column
    keep_mask[(P[:, 0] == N) | (P[:, 1] == M)] = False
    # Add last index to enforce end boundary condition
    keep_mask[-1] = True
    P_mod = P[keep_mask, :]

    return P_mod


def evaluate_synchronized_positions(ground_truth_positions: np.ndarray,
                                    synchronized_positions: np.ndarray,
                                    tolerances: List = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 250]):
    """Compute standard evaluation measures for evaluating the quality of synchronized (musical) positions.

    When synchronizing two versions of a piece of music, one can evaluate the quality of the resulting alignment
    by comparing errors at musical positions (e.g. beats or measures) that appear in both versions.
    This function implements two measures: mean absolute error at positions and the percentage of correctly transferred
    measures given a threshold.

    Parameters
    ----------
    ground_truth_positions: np.ndarray [shape=N]
        Positions (e.g. beat or measure positions) annotated in the target version of a piece of music, in milliseconds.

    synchronized_positions: np.ndarray [shape=N]
        The same musical positions as in 'ground_truth_positions' obtained by transfer using music synchronization,
        in milliseconds.

    tolerances: list of integers
        Tolerances (in miliseconds) used for comparing annotated and synchronized positions.

    Returns
    -------
    mean_absolute_error: float
        Mean absolute error for synchronized positions, in miliseconds.

    accuracy_at_tolerances: list of floats
        Percentages of correctly transferred measures, for each entry in 'tolerances'.

    """
    absolute_errors_at_positions = np.abs(synchronized_positions - ground_truth_positions)

    print('Measure transfer from recording 1 to 2 yielded:')
    mean_absolute_error = np.mean(absolute_errors_at_positions)
    print('\nMean absolute error (MAE): %.2fms (standard deviation: %.2fms)' % (mean_absolute_error,
                                                                                np.std(absolute_errors_at_positions)))
    print('\nAccuracy of transferred positions at different tolerances:')
    print('\t\t\tAccuracy')
    print('################################')
    accuracy_at_tolerances = []
    for tolerance in tolerances:
        accuracy = np.mean((absolute_errors_at_positions < tolerance)) * 100.0
        accuracy_at_tolerances.append(accuracy)
        print('Tolerance: {} ms \t{:.2f} %'.format(tolerance, accuracy))

    return mean_absolute_error, accuracy_at_tolerances
    
    
def smooth_downsample_feature(f_feature: np.ndarray,
                              input_feature_rate: float,
                              win_len_smooth: int = 0,
                              downsamp_smooth: int = 1) -> Tuple[np.ndarray, float]:
    """Temporal smoothing and downsampling of a feature sequence

    Parameters
    ----------
    f_feature : np.ndarray
        Input feature sequence, size dxN

    input_feature_rate : float
        Input feature rate in Hz

    win_len_smooth : int
        Smoothing window length. For 0, no smoothing is applied.

    downsamp_smooth : int
        Downsampling factor. For 1, no downsampling is applied.

    Returns
    -------
    f_feature_stat : np.ndarray
        Downsampled & smoothed feature.

    new_feature_rate : float
        New feature rate after downsampling
    """
    if win_len_smooth != 0 or downsamp_smooth != 1:
        # hack to get the same results as on MATLAB
        stat_window = np.hanning(win_len_smooth+2)[1:-1]
        stat_window /= np.sum(stat_window)

        # upfirdn filters and downsamples each column of f_stat_help
        f_feature_stat = signal.upfirdn(h=stat_window, x=f_feature, up=1, down=downsamp_smooth)
        seg_num = f_feature.shape[1]
        stat_num = int(np.ceil(seg_num / downsamp_smooth))
        cut = int(np.floor((win_len_smooth - 1) / (2 * downsamp_smooth)))
        f_feature_stat = f_feature_stat[:, cut: stat_num + cut]
    else:
        f_feature_stat = f_feature

    new_feature_rate = input_feature_rate / downsamp_smooth

    return f_feature_stat, new_feature_rate


@jit(nopython=True)
def normalize_feature(feature: np.ndarray,
                      norm_ord: int,
                      threshold: float) -> np.ndarray:
    """Normalizes a feature sequence according to the l^norm_ord norm.

    Parameters
    ----------
    feature : np.ndarray
        Input feature sequence of size d x N
            d: dimensionality of feature vectors
            N: number of feature vectors (time in frames)

    norm_ord : int
        Norm degree

    threshold : float
        If the norm falls below threshold for a feature vector, then the
        normalized feature vector is set to be the normalized unit vector.

    Returns
    -------
    f_normalized : np.ndarray
        Normalized feature sequence
    """
    # TODO rewrite in vectorized fashion
    d, N = feature.shape
    f_normalized = np.zeros((d, N))

    # normalize the vectors according to the l^norm_ord norm
    unit_vec = np.ones(d)
    unit_vec = unit_vec / np.linalg.norm(unit_vec, norm_ord)

    for k in range(N):
        cur_norm = np.linalg.norm(feature[:, k], norm_ord)

        if cur_norm < threshold:
            f_normalized[:, k] = unit_vec
        else:
            f_normalized[:, k] = feature[:, k] / cur_norm

    return f_normalized
