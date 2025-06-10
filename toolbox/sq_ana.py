import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import groupby
from scipy.interpolate import interp1d
from scipy.stats import ttest_ind
from pyinform import transfer_entropy

from toolbox import sq_ana as sa
import importlib
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed










import numpy as np, matplotlib.pyplot as plt, pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
from pyinform import transfer_entropy

# ------------------------------------------------------------
# short helper: build 0/1 sequence from σ-multiplier
# ------------------------------------------------------------
def _state(sig, s_mult, sigma_ref=None):
    thr = s_mult * (sigma_ref if sigma_ref is not None else np.std(sig))
    st  = np.full_like(sig, np.nan, float)
    st[sig <= -thr] = 0
    st[sig >=  thr] = 1
    for i in range(1, len(st)):          # carry forward
        if np.isnan(st[i]): st[i] = st[i-1]
    first = np.where(~np.isnan(st))[0][0]
    st[:first] = st[first]
    return st.astype(int)

# ------------------------------------------------------------
#  main analysis
# ------------------------------------------------------------
def thre_data_ana(
    forcing, sq,
    ages = None,  # optional time axis for plots
    *,
    bins_pre       = 6,
    sigma_mult     = 1.5,
    n_surr         = 200,
    nbins_phase    = 8,
    random_seed    = 0
):
    """
    Transfer-entropy workflow with threshold-defined MCV states.

    Parameters
    ----------
    forcing, sq : 1-D arrays (oldest→youngest or youngest→oldest — any order)
    bins_pre    : number of amplitude bins for the forcing
    sigma_mult  : σ-multiplier that defines ±thr
    n_surr      : # permutation surrogates for global TE test
    nbins_phase : # bins when discretising wavelet phase (unused here but
                  kept for interface homogeneity)
    """
    # ---------- 1. chronological order (oldest → youngest) --------------
    
    f = forcing[::-1]
    s = sq      [::-1]

    if ages is None:
        ages = np.arange(len(f))            # dummy axis for plots

    # ---------- 2. build threshold state -------------------------------
    state = _state(s, sigma_mult)       # 0 = cold, 1 = warm

    # ---------- 3. discretise forcing (equal-width amplitude bins) -----
    bins = np.histogram_bin_edges(f, bins=bins_pre)
    x_disc = np.clip(np.digitize(f, bins) - 1, 0, bins_pre-1)

    # ---------- 4. global TE + surrogates ------------------------------
    te_xy = transfer_entropy(x_disc[:-1], state[1:], k=1)
    te_yx = transfer_entropy(state[:-1],  x_disc[1:], k=1)

    rng = np.random.default_rng(random_seed)
    null_xy = np.empty(n_surr)
    null_yx = np.empty(n_surr)
    for i in range(n_surr):
        null_xy[i] = transfer_entropy(rng.permutation(x_disc)[:-1], state[1:], k=1)
        null_yx[i] = transfer_entropy(rng.permutation(state)[:-1],  x_disc[1:], k=1)

    # two-tailed p
    p_xy = (2*min((null_xy>=te_xy).sum(), (null_xy<=te_xy).sum()) + 1)/(n_surr+1)
    p_yx = (2*min((null_yx>=te_yx).sum(), (null_yx<=te_yx).sum()) + 1)/(n_surr+1)

    # ---------- 5. local TE, P(flip), P(stay) ---------------------------
    local_te = transfer_entropy(x_disc, state, k=1, local=True).flatten()[1:]
    x_prev, y_prev, y_next = x_disc[:-1], state[:-1], state[1:]

    counts = np.zeros((bins_pre, 2, 2), int)
    for xi, yi, zi in zip(x_prev, y_prev, y_next):
        counts[xi, yi, zi] += 1
    totals = counts.sum(axis=2, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        cond_p = counts / totals

    p_flip = np.array([cond_p[x,y,1-y] for x,y in zip(x_prev,y_prev)])
    p_stay = np.array([cond_p[x,y,  y] for x,y in zip(x_prev,y_prev)])

    N = min(len(local_te), len(p_flip))
    t_plot = ages[1:1+N]

    # ---------- 6. figures ---------------------------------------------
    # 6a. TE null distributions  (colours swapped vs. your previous draft)
    plt.figure(figsize=(7,3.3))
    plt.hist(null_xy, bins=25, color='#fd8d3c', alpha=.7, label='null  forcing→MCV')
    plt.hist(null_yx, bins=25, color='#6baed6', alpha=.7, label='null  MCV→forcing')
    plt.axvline(te_xy, color='#a63603', lw=2, label=f'obs forcing→MCV (p={p_xy:.3f})')
    plt.axvline(te_yx, color='#08519c', lw=2, label=f'obs MCV→forcing (p={p_yx:.3f})')
    plt.xlabel('transfer entropy (bits)'); plt.ylabel('count')
    plt.legend(frameon=True); plt.tight_layout()

    # 6b. Stacked time-series plot
    fig = plt.figure(figsize=(10,6))
    gs  = GridSpec(3,1, height_ratios=[0.7,0.9,1.1], hspace=0.07)

    axA = fig.add_subplot(gs[0])
    axA.plot(ages, f, lw=0.8, color='k'); axA.set_ylabel('forcing')
    axA.invert_xaxis(); axA.tick_params(axis='x', labelbottom=False)

    axB = fig.add_subplot(gs[1], sharex=axA)
    axB.plot(t_plot, local_te[:N], lw=0.8, color='C0')
    axB.set_ylabel('local TE'); axB.tick_params(axis='x', labelbottom=False)

    axC = fig.add_subplot(gs[2], sharex=axA)
    lnF, = axC.plot(t_plot, p_flip[:N],  color='C1', label='P(flip)')
    axD   = axC.twinx()
    lnS, = axD.plot(t_plot, p_stay[:N],  color='C2', label='P(stay)')
    axC.set_ylabel('P(flip)', color='C1'); axD.set_ylabel('P(stay)', color='C2')
    axC.set_xlabel('Age index'); axC.legend([lnF,lnS], ['P(flip)','P(stay)'])
    plt.show()

    return {'TE_xy': te_xy, 'p_xy': p_xy,
            'TE_yx': te_yx, 'p_yx': p_yx,
            'local_TE': local_te, 'p_flip': p_flip, 'p_stay': p_stay}
















































# import numpy as np
# import pywt
# import matplotlib.pyplot as plt
# from pyinform import transfer_entropy


# def freq_resolved_te(
#     x, y,                       # 1-D arrays, same length, chronological
#     *,
#     wavelet='cmor1.5-1.0',
#     max_level=None,             # None → use all scales up to Nyquist/2
#     k=1,
#     plot=True,
#     cmap='viridis'
# ):
#     """
#     Frequency-(scale-)resolved transfer entropy matrix:
#         rows   = source scale index
#         cols   = target scale index
#     """
#     if len(x) != len(y):
#         raise ValueError("x and y must have equal length")

#     # --- 1. choose scales automatically --------------------------
#     # use dyadic (1,2,3,... ) until "max_level" or N/2
#     N = len(x)
#     max_level = max_level or (N // 2)
#     scales = np.arange(1, max_level + 1)
#     x=x[::-1]
#     y=y[::-1]   

#     # --- 2. CWT (complex) ---------------------------------------
#     coeffs_x, _ = pywt.cwt(x, scales, wavelet)
#     coeffs_y, _ = pywt.cwt(y, scales, wavelet)

#     # --- 3. discretise phase into 8 bins ------------------------
#     phase_bins = np.linspace(-np.pi, np.pi, 9)
#     disc_x = np.digitize(np.angle(coeffs_x), phase_bins) - 1
#     disc_y = np.digitize(np.angle(coeffs_y), phase_bins) - 1

#     n_sc   = len(scales)
#     te_mat = np.empty((n_sc, n_sc))

#     for i in range(n_sc):              # source scale
#         for j in range(n_sc):          # target scale
#             te_mat[i, j] = transfer_entropy(
#                 disc_x[i, :-1], disc_y[j, 1:], k=k
#             )

#     # --- 4. optional plot ---------------------------------------
#     if plot:
#         plt.figure(figsize=(6, 5))
#         plt.imshow(te_mat, origin='lower', cmap=cmap,
#                    extent=[0, n_sc, 0, n_sc], aspect='auto')
#         plt.colorbar(label='TE (bits)')
#         plt.xlabel('target scale index  (low → high freq)')
#         plt.ylabel('source scale index')
#         plt.title('Frequency-resolved Transfer Entropy')
#         plt.tight_layout()
#         plt.show()

#     return te_mat, scales






# import numpy as np
# import pywt
# import matplotlib.pyplot as plt
# from pyinform import transfer_entropy


# def freq_resolved_te(
#     x, y,                       # 1-D arrays (chronological)
#     *,
#     wavelet='cmor1.5-1.0',
#     max_level=None,
#     k=1,
#     plot=True,
#     cmap='viridis'
# ):
#     """
#     Continuous-wavelet scalogram  +  scale×scale TE matrix.

#     Returns
#     -------
#     te_mat  : 2-D ndarray  [src scale , trg scale]
#     scales  : list of scale numbers (PyWavelets convention)
#     """

#     # ------------------------------------------------------------------
#     # 1. choose dyadic scales  (1 … max_level)
#     # ------------------------------------------------------------------
#     N = len(x)
#     max_level = max_level or (N // 4)          # ≤ N/4 keeps runtime sane
#     scales = np.arange(1, max_level + 1)       # 1 = highest frequency

#     # ------------------------------------------------------------------
#     # 2. complex CWT on both series
#     # ------------------------------------------------------------------
#     coeffs_x, _ = pywt.cwt(x, scales, wavelet)
#     coeffs_y, _ = pywt.cwt(y, scales, wavelet)

#     # absolute power for plotting
#     pow_x = np.abs(coeffs_x) ** 2
#     pow_y = np.abs(coeffs_y) ** 2

#     # ------------------------------------------------------------------
#     # 3. phase → 8-bin symbol sequences
#     # ------------------------------------------------------------------
#     phase_bins = np.linspace(-np.pi, np.pi, 9)
#     disc_x = np.digitize(np.angle(coeffs_x), phase_bins) - 1
#     disc_y = np.digitize(np.angle(coeffs_y), phase_bins) - 1

#     # ------------------------------------------------------------------
#     # 4. TE per (src scale , trg scale)
#     # ------------------------------------------------------------------
#     n_sc = len(scales)
#     te_mat = np.empty((n_sc, n_sc))

#     for i in range(n_sc):
#         for j in range(n_sc):
#             te_mat[i, j] = transfer_entropy(
#                 disc_x[i, :-1],     # X_t  at scale i
#                 disc_y[j, 1:],      # Y_{t+1} at scale j
#                 k=k
#             )

#     # ------------------------------------------------------------------
#     # 5. plots: two scalograms + TE matrix
#     # ------------------------------------------------------------------
#     if plot:
#         fig, ax = plt.subplots(1, 3, figsize=(14, 5),
#                                gridspec_kw={'width_ratios':[1,1,1.2]})

#         im0 = ax[0].imshow(pow_x, origin='lower', aspect='auto',
#                            cmap='hot', extent=[0,N,0,n_sc])
#         ax[0].set_title('Source (pre) scalogram')
#         ax[0].set_ylabel('scale index')
#         plt.colorbar(im0, ax=ax[0], fraction=0.046)

#         im1 = ax[1].imshow(pow_y, origin='lower', aspect='auto',
#                            cmap='hot', extent=[0,N,0,n_sc])
#         ax[1].set_title('Target (sq) scalogram')
#         plt.colorbar(im1, ax=ax[1], fraction=0.046)

#         im2 = ax[2].imshow(te_mat, origin='lower', aspect='auto',
#                            cmap=cmap, extent=[0,n_sc,0,n_sc])
#         ax[2].set_title('Scale×Scale TE (bits)')
#         ax[2].set_xlabel('target scale'); ax[2].set_ylabel('source scale')
#         plt.colorbar(im2, ax=ax[2], fraction=0.046)
#         plt.tight_layout(); plt.show()

#     return te_mat, scales




import numpy as np, pywt, matplotlib.pyplot as plt
from pyinform import transfer_entropy


def freq_resolved_te(
    x, y,
    *,
    wavelet='cmor1.5-1.0',
    sampling_period=10,          # yr per sample
    # ---- source (pre) band -----------------------------------
    src_min_period=20_000,       # yr
    src_max_period=20_000,
    n_src_scales=3,              # if min=max, use 3 scales around it
    # ---- target (sq) band ------------------------------------
    trg_min_period=100,          # yr
    trg_max_period=3_000,
    n_trg_scales=64,
    k=1,
    plot=True,
    cmap='viridis',
    source_vname='Precession',       # e.g. 'pre'
    target_vname='CH₄ MCV'        # e.g. 'sq'
):
    """
    Wavelet-scale × wavelet-scale TE with *separate* period bands
    for source (x) and target (y).

    Returns
    -------
    te_mat     : (n_src × n_trg) array
    periods_x  : list (len n_src)  [yr]
    periods_y  : list (len n_trg)  [yr]
    """
    x= x[::-1]  # reverse chronological order
    y= y[::-1]  # reverse chronological order

    if len(x) != len(y):
        raise ValueError("x and y must have equal length")

    # ---------- helper: build scale list -----------------------
    fc = pywt.central_frequency(wavelet)

    def build_scales(min_p, max_p, n_sc):
        if np.isclose(min_p, max_p):
            ctr = min_p
            periods = ctr * np.geomspace(0.8, 1.2, n_sc)
        else:
            periods = np.geomspace(min_p, max_p, n_sc)
        return periods, periods * fc / sampling_period

    periods_x, scales_x = build_scales(src_min_period, src_max_period,
                                       n_src_scales)
    periods_y, scales_y = build_scales(trg_min_period, trg_max_period,
                                       n_trg_scales)

    # ---------- 1. CWT -----------------------------------------
    coeffs_x, _ = pywt.cwt(x, scales_x, wavelet,
                           sampling_period=sampling_period)
    coeffs_y, _ = pywt.cwt(y, scales_y, wavelet,
                           sampling_period=sampling_period)

    pow_x = np.abs(coeffs_x) ** 2
    pow_y = np.abs(coeffs_y) ** 2

    # Immediately after pow_x is computed:
    row_max = pow_x.mean(axis=1).argmax()
    print("max-power row = %d   →  period ≈ %.1f ka"
        % (row_max, periods_x[row_max]/1000))

    # ---------- 2. discretise phase ----------------------------
    bins = np.linspace(-np.pi, np.pi, 9)
    disc_x = np.digitize(np.angle(coeffs_x), bins) - 1
    disc_y = np.digitize(np.angle(coeffs_y), bins) - 1

    # ---------- 3. TE matrix -----------------------------------
    te_mat = np.empty((len(scales_x), len(scales_y)))
    for i in range(len(scales_x)):        # src scale
        for j in range(len(scales_y)):    # trg scale
            te_mat[i, j] = transfer_entropy(
                disc_x[i, :-1], disc_y[j, 1:], k=k
            )

    # ---------- 4. plots ---------------------------------------
    if plot:
        # t_ka = np.arange(len(x)) * sampling_period / 1000  # time axis in ka
        # # extent_x = [t_ka[0], t_ka[-1],
        # #             periods_x[-1]/1000, periods_x[0]/1000]
        # # extent_y = [t_ka[0], t_ka[-1],
        # #             periods_y[-1]/1000, periods_y[0]/1000]

        # extent_x = [t_ka[0], t_ka[-1],
        #             periods_x[-1]/1000, periods_x[0]/1000]
        # extent_y = [t_ka[0], t_ka[-1],
        #             periods_y[-1]/1000, periods_y[0]/1000]

        # fig, ax = plt.subplots(1, 3, figsize=(15, 5),
        #                        gridspec_kw={'width_ratios':[1,1,1.3]})

        # im0 = ax[0].imshow(pow_x, origin='upper', aspect='auto',
        #                    cmap='hot', extent=extent_x)
        # ax[0].set_title('source scalogram (pre)')
        # ax[0].set_xlabel('time (ka BP)')
        # ax[0].set_ylabel('period (ka)')
        # plt.colorbar(im0, ax=ax[0], fraction=.046)

        # im1 = ax[1].imshow(pow_y, origin='upper', aspect='auto',
        #                    cmap='hot', extent=extent_y)
        # ax[1].set_title('target scalogram (sq)')
        # ax[1].set_xlabel('time (ka BP)')
        # plt.colorbar(im1, ax=ax[1], fraction=.046)

        # # TE matrix with period axes
        # im2 = ax[2].imshow(te_mat, origin='lower', aspect='auto',
        #                    cmap=cmap,
        #                    extent=[periods_y[0]/1000, periods_y[-1]/1000,
        #                            periods_x[0]/1000, periods_x[-1]/1000])
        # ax[2].set_title('TE  (source period → target period)')
        # ax[2].set_xlabel('target period (ka)')
        # ax[2].set_ylabel('source period (ka)')

        # # for a in ax[:2]:
        # #     a.set_yscale('log')
        # #     a.invert_yaxis()       # smaller periods at bottom, like a spectrogram
        # # ax[2].set_yscale('log')
        # # ax[2].set_xscale('log')

        # plt.colorbar(im2, ax=ax[2], fraction=.046)

        # plt.tight_layout(); plt.show()
        # --- build extents -------------------------------------------------
        t_ka = np.arange(len(x)) * sampling_period / 1000   # time axis in ka

        extent_src = [t_ka[0], t_ka[-1],
                    periods_x[0]/1000, periods_x[-1]/1000]   # low → high period
        extent_trg = [t_ka[0], t_ka[-1],
                    periods_y[0]/1000, periods_y[-1]/1000]

        extent_te  = [periods_y[0]/1000, periods_y[-1]/1000,   # x-axis  (target)
                    periods_x[0]/1000, periods_x[-1]/1000]   # y-axis  (source)

        # --- plots ---------------------------------------------------------
        fig, ax = plt.subplots(1, 3, figsize=(15, 3.5),
                            gridspec_kw={'width_ratios':[1,1,1.3]})

        im0 = ax[0].imshow(pow_x, origin='upper', aspect='auto',
                        cmap='hot', extent=extent_src)
        ax[0].set_title('source scalogram (pre)')
        ax[0].set_xlabel('time (ka BP)')
        ax[0].set_ylabel('period (ka)')
        plt.colorbar(im0, ax=ax[0], fraction=.046)

        im1 = ax[1].imshow(pow_y, origin='upper', aspect='auto',
                        cmap='hot', extent=extent_trg)
        ax[1].set_title('target scalogram (sq)')
        ax[1].set_xlabel('time (ka BP)')
        plt.colorbar(im1, ax=ax[1], fraction=.046)

        im2 = ax[2].imshow(te_mat, origin='upper', aspect='auto',
                        cmap=cmap, extent=extent_te, vmin=np.quantile(te_mat,0.5), vmax=te_mat.max())
        ax[2].set_title('TE  (source phases → target phases)')
        ax[2].set_xlabel(f'{target_vname} period (kyr)')
        ax[2].set_ylabel(f'{source_vname} period (kyr)')
        plt.colorbar(im2, ax=ax[2], fraction=.046)

        plt.tight_layout(); plt.show()


    return te_mat, periods_x, periods_y





# def te_matrix_zscore(x, y, n_perm=100, **kwargs):
#     te_obs, px, py = freq_resolved_te(x, y, plot=False, **kwargs)
#     null_stack = np.empty((n_perm, *te_obs.shape))
#     rng = np.random.default_rng(0)

#     for k in range(n_perm):
#         x_perm = rng.permutation(x)
#         null_stack[k], _, _ = freq_resolved_te(x_perm, y, plot=False, **kwargs)

#     mu  = null_stack.mean(axis=0)
#     sig = null_stack.std (axis=0)
#     z   = (te_obs - mu) / sig
#     vmax = np.nanpercentile(abs(z), 99)

#     # plot
#     plt.imshow(z, origin='lower', cmap='bwr', vmin=-vmax, vmax=vmax,
#                extent=[py[0]/1e3, py[-1]/1e3, px[0]/1e3, px[-1]/1e3])
#     plt.colorbar(label='z-score')
#     plt.xlabel('target period (ka)'); plt.ylabel('source period (ka)')
#     plt.title('Scale×scale TE  (z-score vs permutations)')
#     plt.tight_layout(); plt.show()
#     return z


# from joblib import Parallel, delayed
# import numpy as np, matplotlib.pyplot as plt

# def _perm_te(x, y, rng, kwargs):
#     """worker: compute TE matrix for one permutation of x"""
#     x_perm = rng.permutation(x)
#     te_mat, _, _ = freq_resolved_te(x_perm, y, plot=False, **kwargs)
#     return te_mat

# def te_matrix_zscore(
#     x, y,
#     n_perm      = 100,
#     n_jobs      = -1,        # -1 → use all available cores
#     random_seed = 0,
#     **kwargs                  # passed straight to freq_resolved_te
# ):
#     """
#     Permutation test for the scale×scale TE map.
#     Returns z-score matrix; also draws it.
#     """
#     x = x[::-1]  # reverse chronological order
#     y = y[::-1]  # reverse chronological order
#     # 1) observed matrix
#     te_obs, p_src, p_trg = freq_resolved_te(x, y, plot=False, **kwargs)

#     # 2) parallel surrogates
#     rng_master = np.random.default_rng(random_seed)
#     seeds = rng_master.integers(0, 2**32-1, size=n_perm)

#     null_list = Parallel(n_jobs=n_jobs, prefer='processes')(
#         delayed(_perm_te)(x, y,
#                           np.random.default_rng(s),  # private RNG per worker
#                           kwargs)
#         for s in seeds
#     )
#     null_stack = np.stack(null_list, axis=0)

#     # 3) z-score
#     mu  = null_stack.mean(axis=0)
#     sig = null_stack.std (axis=0)
#     z   = (te_obs - mu) / sig

#     # 4) plot
#     vmax = np.nanpercentile(np.abs(z), 99)
#     plt.imshow(z, origin='lower', cmap='bwr',
#                vmin=-vmax, vmax=vmax,
#                extent=[p_trg[0]/1e3, p_trg[-1]/1e3,
#                        p_src[0]/1e3, p_src[-1]/1e3])
#     plt.colorbar(label='z-score')
#     plt.xlabel('target period (ka)')
#     plt.ylabel('source period (ka)')
#     plt.title(f'Scale×scale TE  (z, {n_perm} perms, two-tailed)')
#     plt.tight_layout(); plt.show()

#     return z





















# from dit import Distribution
# from dit.other import pid
# from itertools import product


# def pid_unique_redundant(
#     x, y, z,          # 1-D int arrays (|alphabet| ≤ 16 recommended)
#     k=1,              # use x_t , y_t to predict z_{t+1}
#     pid_method='wb2018'
# ):
#     """
#     PID of {X,Y} → Z_{t+1}.  Returns dict with
#         'unique_x' , 'unique_y' , 'redundant' , 'synergy'
#     in bits.
#     """
#     # build joint symbols for a first-order Markov set-up
#     xs, ys, zs = x[:-1], y[:-1], z[1:]

#     # alphabet tuples (x,y,z)
#     joint = list(zip(xs, ys, zs))
#     outcomes, counts = np.unique(joint, return_counts=True, axis=0)
#     probs = counts / counts.sum()
#     outcomes = [tuple(o) for o in outcomes]

#     d = Distribution(outcomes, probs)
#     d.set_rv_names(('X','Y','Z'))

#     P = pid.pid(d, pid_method, ((0,), (1,)), (2,))
#     return {k: P[k] for k in ('UIX', 'UIY', 'RXY', 'SIXY')}
















































# reload module in case of updates
def mc_TE_heatmap_inter(
    df_sq,
    target_column='sq',
    intervals=None,
    niter=100,
    n_surr=100,
    alpha=0.05,
    k=1,
    gbins=None,
    p_thresh=0.9,
    n_jobs=-1,
    forcing_var='pre',  # 'pre' or 'obl'
    if_plot=False,
    plot=True
):
    """
    Perform Monte Carlo tests of transfer entropy between 'sq' and a chosen forcing variable
    over a range of interpolation intervals and multiple forcing_bins, then plot a heatmap.

    Parameters
    ----------
    df_sq : pandas.DataFrame
        Input data frame containing 'sq' and forcing variables.
    intervals : list of int, optional
        Interpolation intervals to test (default=10,20,...,100).
    niter : int, optional
        Number of Monte Carlo repetitions (default=100).
    n_surr : int, optional
        Number of surrogates per TE test (default=100).
    alpha : float, optional
        Significance level for surrogate test (default=0.05).
    k : int, optional
        History length for transfer entropy (default=1).
    gbins : list of int, optional
        Forcing bins to test (default=range(2,11)).
    p_thresh : float, optional
        Threshold for highlighting cells in heatmap (default=0.9).
    n_jobs : int, optional
        Number of parallel jobs (default=-1 for all cores).
    forcing_var : str, optional
        Which forcing variable to test: 'pre' or 'obl' (default='pre').
    if_plot : bool, optional
        Whether to plot intermediate TE results (default=False).
    plot : bool, optional
        Whether to display the heatmap (default=True).

    Returns
    -------
    fractions : numpy.ndarray
        Matrix of fraction significant values (len(intervals) x len(gbins)).
    fig, ax : matplotlib objects (if plot=True)
        Heatmap Figure and Axes.
    """
    # defaults
    if intervals is None:
        intervals = list(range(10, 101, 10))
    if gbins is None:
        gbins = list(range(2, 11))
    if forcing_var not in ('pre', 'obl'):
        raise ValueError("forcing_var must be 'pre' or 'obl'.")

    # reload toolbox
    importlib.reload(sa)

    # 1) Prepare data for each interval
    data_list = []  # list of (forcing, sq) pairs
    for interval in intervals:
        df_sq_i, df_pre_i, df_obl_i = sa.interpolate_data_forcing(
            df_sq.copy(), interval=interval, if_plot=if_plot
        )
        if forcing_var == 'pre':
            forcing = df_pre_i['pre'].values
        else:
            forcing = df_obl_i['obl'].values
        sq_vals = df_sq_i[target_column].values
        data_list.append((forcing, sq_vals))

    # 2) Worker for one MC iteration
    def one_mc_iter(data_list, gbins, n_surr, alpha, k):
        counts = np.zeros((len(data_list), len(gbins)), dtype=int)
        for i, (forcing, sq_vals) in enumerate(data_list):
            for j, bins in enumerate(gbins):
                sig, _ = sa.transfer_entropy_surrogate_test(
                    forcing, sq_vals,
                    k=k,
                    forcing_bins=bins,
                    n_surr=n_surr,
                    p=alpha,
                    if_plot=False
                )
                counts[i, j] = int(sig)
        return counts

    # 3) Parallel Monte Carlo
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(one_mc_iter)(data_list, gbins, n_surr, alpha, k)
        for _ in range(niter)
    )
    total = np.stack(results).sum(axis=0)
    fractions = total / niter

    # 4) Plot
    if plot:
        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(fractions, origin='lower', aspect='auto', cmap='viridis', vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(gbins)))
        ax.set_xticklabels(gbins)
        ax.set_xlabel('forcing_bins')
        ax.set_yticks(np.arange(len(intervals)))
        ax.set_yticklabels(intervals)
        ax.set_ylabel('interpolation interval')
        ax.set_title(f'TE(signif) for sq vs {forcing_var} across intervals')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Fraction significant')
        # highlight
        for i, j in zip(*np.where(fractions > p_thresh)):
            rect = plt.Rectangle((j-0.5, i-0.5),1,1, fill=False, edgecolor='white', linewidth=2)
            ax.add_patch(rect)
        # annotate
        for i in range(len(intervals)):
            for j in range(len(gbins)):
                pct = fractions[i, j] * 100
                color = 'white' if fractions[i, j] > 0.5 else 'black'
                ax.text(j, i, f"{pct:.0f}%", ha='center', va='center', color=color, fontsize=8)
        plt.tight_layout()
        plt.show()
        return fractions, fig, ax

    return fractions








import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from  scipy.signal import butter, filtfilt


def bandpass_ch4_d18o(
    df_ch4_interp,                 # cols: age , ch4   (common grid)
    df_d18o_interp,                # cols: age , d18O (same grid)
    *,
    low_period = 300,              # [yr]  remove > low_period  (low-cut)
    high_period = 3_000,           # [yr]  remove < high_period (high-cut)
    butter_order = 4,
    flip_sign   = False,           # flip δ18O so “warm = high”
    plot        = True
):
    """
    Butterworth **band-pass** the pre-interpolated CH4 and δ18O series.

    Pass-band = [ 1/high_period  …  1/low_period ]   (Hz)

    Returns
    -------
    df_filt_ch4   : DataFrame['age','filt_ch4']
    df_filt_d18O  : DataFrame['age','filt_d18O']
    """
    # ---------- 1. ensure same age axis ----------------
    ages = df_ch4_interp['age'].values
    if not np.allclose(ages, df_d18o_interp['age'].values):
        raise ValueError("df_ch4_interp and df_d18o_interp must share the same age vector")

    ch4_vals  = df_ch4_interp['ch4'].values
    d18o_vals = df_d18o_interp['d18O'].values
    if flip_sign:
        d18o_vals = -d18o_vals

    # ---------- 2. design Butterworth band-pass --------
    dt = np.median(np.diff(ages))        # yr
    fs = 1.0 / dt                        # samples per yr
    # convert periods [yr] → frequencies [Hz]
    f_low  = 1.0 / high_period           # lower edge (remove slower than this)
    f_high = 1.0 / low_period            # upper edge (remove faster than this)
    Wn = np.array([f_low, f_high]) / (fs * 0.5)   # normalised to Nyquist

    if np.any(Wn <= 0) or np.any(Wn >= 1) or Wn[0] >= Wn[1]:
        raise ValueError("Band edges out of range; check low_period / high_period.")

    b, a = butter(butter_order, Wn, btype='bandpass')
    filt_ch4  = filtfilt(b, a, ch4_vals)
    filt_d18o = filtfilt(b, a, d18o_vals)

    # ---------- 3. wrap to DataFrames ------------------
    df_filt_ch4  = pd.DataFrame({'age': ages, 'filt_ch4' : filt_ch4})
    df_filt_d18O = pd.DataFrame({'age': ages, 'filt_d18O': filt_d18o})

    # ---------- 4. optional quick plots ----------------
    if plot:
        band_lab = f'{low_period//1_000:.1f}–{high_period//1_000:.1f} ka'
        for name, orig, filt in [
            ('CH4',  ch4_vals,  filt_ch4),
            ('δ18O', d18o_vals, filt_d18o)
        ]:
            plt.figure(figsize=(10,3))
            plt.plot(ages, orig, alpha=0.35, label=f'{name} (raw)')
            plt.plot(ages, filt, lw=1.6,    label=f'{name} ({band_lab} band-pass)')
            plt.gca().invert_xaxis()
            plt.xlabel('Age (yr BP)'); plt.ylabel(name)
            plt.legend(); plt.tight_layout(); plt.show()

    return df_filt_ch4, df_filt_d18O






















import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

def highpass_ch4_d18O(
    df_ch4_interp,                 # ← already on common grid; cols: age , ch4
    df_d18O_interp,                # ← same grid;        cols: age , d18O
    *,
    cutoff_period=10_000,          # remove variability slower than this [yr]
    butter_order=4,
    flip_sign=False,                # flip δ18O so “warm = high”
    plot=True
):
    """
    High–pass the pre-interpolated CH4 and δ18O series (> cutoff_period).

    Returns
    -------
    df_filt_ch4   : DataFrame[age , filt_ch4]
    df_filt_d18O  : DataFrame[age , filt_d18O]
    """

    # -------- 1. sanity: same age grid -----------------
    ages = df_ch4_interp['age'].values
    if not np.allclose(ages, df_d18O_interp['age'].values):
        raise ValueError('df_ch4_interp and df_d18o_interp must share the same age vector')

    ch4_vals  = df_ch4_interp['ch4'].values
    d18o_vals = df_d18O_interp['d18O'].values
    if flip_sign:
        d18o_vals = -d18o_vals

    # -------- 2. Butterworth high-pass -----------------
    # dt [yr] from median spacing
    dt = np.median(np.diff(ages))
    fs = 1.0 / dt                       # samples per year
    fc = 1.0 / cutoff_period            # Hz
    Wn = fc / (fs * 0.5)                # normalised to Nyquist

    b, a = butter(butter_order, Wn, btype='highpass')
    filt_ch4  = filtfilt(b, a, ch4_vals)
    filt_d18o = filtfilt(b, a, d18o_vals)

    # -------- 3. wrap to DataFrames --------------------
    df_filt_ch4  = pd.DataFrame({'age': ages, 'filt_ch4' : filt_ch4})
    df_filt_d18O = pd.DataFrame({'age': ages, 'filt_d18O': filt_d18o})

    # -------- 4. quick visual check -------------------
    if plot:
        for name, orig, filt in [
            ('CH4',  ch4_vals,  filt_ch4),
            ('δ18O', d18o_vals, filt_d18o)
        ]:
            plt.figure(figsize=(10,3))
            plt.plot(ages, orig,  alpha=0.4, label=f'{name} (raw)')
            plt.plot(ages, filt,  lw=1.8,   label=f'{name} (> 1/{cutoff_period/1_000:.0f} ka)')
            plt.gca().invert_xaxis()
            plt.xlabel('Age (yr BP)'); plt.ylabel(name)
            plt.legend(); plt.tight_layout(); plt.show()

    return df_filt_ch4, df_filt_d18O











# reload module in case of updates
def mc_TE_heatmap(
    pre,
    sq,
    niter=100,
    n_surr=100,
    alpha=0.05,
    ks=None,
    gbins=None,
    p_thresh=0.9,
    n_jobs=-1,
    if_plot=False,
    plot=True
):

    # default parameter lists
    if ks is None:
        ks = [1, 2, 3, 4, 5, 6]
    if gbins is None:
        gbins = list(range(2, 11))


    # 2) Worker for one MC iteration
    def one_mc_iter(pre, sq, ks, gbins, n_surr, alpha):
        local_counts = np.zeros((len(ks), len(gbins)), dtype=int)
        for i, k in enumerate(ks):
            for j, b in enumerate(gbins):
                sig, _ = transfer_entropy_surrogate_test(
                    pre, sq,
                    k=k,
                    forcing_bins=b,
                    n_surr=n_surr,
                    p=alpha,
                    if_plot=False
                )
                local_counts[i, j] = int(sig)
        return local_counts

    # 3) Parallel Monte Carlo
    results = Parallel(n_jobs=n_jobs, backend="loky")(  
        delayed(one_mc_iter)(pre, sq, ks, gbins, n_surr, alpha)
        for _ in range(niter)
    )
    total_counts = np.stack(results, axis=0).sum(axis=0)
    fractions = total_counts / niter

    # 4) Plot heatmap
    if plot:
        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(
            fractions,
            origin='lower',
            aspect='auto',
            cmap='viridis',
            vmin=0, vmax=1
        )
        ax.set_xticks(np.arange(len(gbins)))
        ax.set_xticklabels(gbins)
        ax.set_xlabel('forcing_bins')
        ax.set_yticks(np.arange(len(ks)))
        ax.set_yticklabels(ks)
        ax.set_ylabel('history length k')
        ax.set_title(f'Fraction significant over {niter} Monte Carlo runs')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Fraction significant')
        # white boxes where above threshold
        for i, j in zip(*np.where(fractions > p_thresh)):
            rect = plt.Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                fill=False, edgecolor='white', linewidth=2
            )
            ax.add_patch(rect)
        # annotate
        for i in range(len(ks)):
            for j in range(len(gbins)):
                pct = fractions[i, j] * 100
                color = 'white' if fractions[i, j] > 0.5 else 'black'
                ax.text(j, i, f"{pct:.0f}%", ha='center', va='center',
                        color=color, fontsize=8)
        plt.tight_layout()
        plt.show()
        return fractions, fig, ax

    return fractions




import numpy as np
import matplotlib.pyplot as plt

def moving_TE(
    df_pre,
    df_sq,
    forcing_column='pre',
    target_column='sq',
    time_column='age',
    nbins_pre=4,
    nbins_target=2,
    window_length=500,
):
    # --- 1) Extract raw numpy arrays and reverse if needed
    pre_raw = df_pre[forcing_column].values[::-1]
    sq_raw  = df_sq[target_column].values[::-1]
    t        = df_pre[time_column].values[::-1]
    N = len(pre_raw)

    # --- 1.1) Find “low-pre” periods (below median here)
    q25   = np.quantile(pre_raw, 0.5)
    low   = pre_raw < q25
    edges = np.diff(low.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    ends   = list(np.where(edges == -1)[0] + 1)
    if low[0]:   starts.insert(0, 0)
    if low[-1]:  ends.append(N)
    low_periods = list(zip(starts, ends))

    # --- 2) Discretize once, globally
    bins_pre = np.histogram_bin_edges(pre_raw, bins=nbins_pre)
    pre_disc = np.digitize(pre_raw, bins_pre) - 1
    sq_disc  = (sq_raw > 0).astype(int)

    # --- 3) Compute local TE in a moving window
    local_te = np.full(N, np.nan)
    for i in range(1, N):
        start = max(0, i - window_length + 1)
        # counts for this window
        c_xyz = {}
        c_xy  = {}
        c_zy  = {}
        c_z   = {}

        # collect counts from j=start+1 .. i
        for j in range(start+1, i+1):
            x_t   = sq_disc[j]
            x_tm1 = sq_disc[j-1]
            y_tm1 = pre_disc[j-1]

            c_xyz[(x_t, x_tm1, y_tm1)] = c_xyz.get((x_t, x_tm1, y_tm1), 0) + 1
            c_xy[(x_t, x_tm1)]         = c_xy.get((x_t, x_tm1), 0) + 1
            c_zy[(x_tm1, y_tm1)]       = c_zy.get((x_tm1, y_tm1), 0) + 1
            c_z[x_tm1]                 = c_z.get(x_tm1, 0) + 1

        x_t   = sq_disc[i]
        x_tm1 = sq_disc[i-1]
        y_tm1 = pre_disc[i-1]

        # only compute if we've seen that history in this window
        if c_zy.get((x_tm1, y_tm1), 0) > 0 and c_z.get(x_tm1, 0) > 0:
            num = c_xyz[(x_t, x_tm1, y_tm1)] / c_zy[(x_tm1, y_tm1)]
            den = c_xy[(x_t, x_tm1)]         / c_z[x_tm1]
            local_te[i] = np.log2(num / den)

    # --- 4) Plot setup with two panels
    fig = plt.figure(figsize=(12, 4))
    gs  = fig.add_gridspec(2, 1, height_ratios=[0.8, 1.6], hspace=0)

    # Top panel: pre & sq
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(t, pre_raw, label=f'{forcing_column} (raw)')
    ax0.set_ylabel(forcing_column)
    ax0.tick_params(axis='x', which='both', labelbottom=False)
    ax0.set_xlim(t[1], t[-1])

    ax0b = ax0.twinx()
    ax0b.plot(t, sq_raw, color='C1', linestyle='-', marker='', label=f'{target_column} (raw)')
    ax0b.set_ylabel(target_column)

    # Shade low-pre
    for start, end in low_periods:
        ax0.axvspan(t[start], t[end-1], color='gray', alpha=0.3)
        ax0b.axvspan(t[start], t[end-1], color='gray', alpha=0.3)

    # Draw bin boundaries as horizontal lines
    for y in bins_pre:
        ax0.axhline(y=y, color='k', linestyle='--', alpha=0.5)

    # combine legends
    # lns = ax0.get_lines() + ax0b.get_lines()
    # labs = [l.get_label() for l in lns]
    # ax0.legend(lns, labs, loc='upper right')

    all_lines = ax0.get_lines() + ax0b.get_lines()
    good_lines = [line for line in all_lines if not line.get_label().startswith('_')]
    good_labels = [line.get_label() for line in good_lines]
    ax0.legend(good_lines, good_labels, loc='upper right')

    # Bottom panel: local TE
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax1.plot(t, np.concatenate([[np.nan], local_te[1:]]), color='C2', label='Local TE')
    ax1.set_xlabel(time_column)
    ax1.set_ylabel('Local TE (bits)')
    ax1.legend(loc='upper right')

    # Shade low-pre on bottom panel
    for start, end in low_periods:
        ax1.axvspan(t[start], t[end-1], color='gray', alpha=0.3)

    plt.show()

    return local_te











import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from matplotlib import cm, colors


def count_sq_pre_contexts_3d(
    sq, pre, nbins_pre: int = 6, cmap_name: str = "jet"
):
    """
    Count the occurrences of each (sq_t, pre_t-bin) pair and visualise
    them in a colour-coded 3-D bar chart.

    The colour of every bar is taken from a warm–cold colormap so that
    taller bars appear brighter/warmer; bars of equal height share the
    same colour.

    Parameters
    ----------
    sq, pre : 1-D array-like (same length)
        Time-series for the square wave and the forcing (precession, etc.).
    nbins_pre : int, default 6
        Number of equal-width bins for the forcing amplitude.
    cmap_name : str, default "coolwarm"
        Any Matplotlib colormap name.

    Returns
    -------
    counts : ndarray shape (2, nbins_pre)
    """
    # --- 0. prep & sanity check -----------------------------------------
    sq = np.asarray(sq).ravel()
    pre = np.asarray(pre).ravel()
    if sq.shape != pre.shape:
        raise ValueError("sq and pre must have the same length")

    # --- 1. discretise ---------------------------------------------------
    sq_disc = (sq > 0).astype(int)                     # 0 / 1
    bins_pre = np.histogram_bin_edges(pre, bins=nbins_pre)
    pre_disc = np.digitize(pre, bins_pre) - 1          # 0 … nbins_pre
    pre_disc = np.clip(pre_disc, 0, nbins_pre - 1)

    # --- 2. tally counts[i, j] ------------------------------------------
    counts = np.zeros((2, nbins_pre), dtype=int)
    for i, j in zip(sq_disc, pre_disc):
        counts[i, j] += 1

    # --- 3. pretty-print table ------------------------------------------
    df = pd.DataFrame(
        counts,
        index=[r"sq=0", r"sq=1"],
        columns=[f"pre={k}" for k in range(nbins_pre)],
    )
    print(df.to_string())

    # --- 4. 3-D bar plot -------------------------------------------------
    fig = plt.figure(figsize=(7.2, 4.8))
    ax = fig.add_subplot(111, projection="3d")

    # bar positions & sizes
    _xs, _ys = np.meshgrid(np.arange(nbins_pre), [0, 1])
    xs = _xs.ravel()
    ys = _ys.ravel()
    zs = np.zeros_like(xs)
    dx = dy = 0.75
    dz = counts.T.ravel().astype(float)

    # colour mapping (same colour ↔ same height)
    cmap = cm.get_cmap(cmap_name)
    norm = colors.Normalize(vmin=dz.min(), vmax=dz.max())
    bar_colors = cmap(norm(dz))

    ax.bar3d(xs, ys, zs, dx, dy, dz, color=bar_colors, shade=True)

    # axis labelling
    ax.set_xlabel("pre bin")
    ax.set_ylabel("sq bin")
    ax.set_zlabel("counts")
    ax.set_xticks(np.arange(nbins_pre) + dx / 2)
    ax.set_xticklabels([f"{k}" for k in range(nbins_pre)])
    ax.set_yticks([0.4, 1.4])
    ax.set_yticklabels(["0", "1"])

    # --- 5. add colour-bar without blocking z-tick labels ---------------
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    fig.colorbar(
        mappable,
        ax=ax,
        shrink=0.6,
        pad=0.22,        # push further right so it doesn't overlap z-ticks
        label="counts",
    )

    # ax.set_title("3-D context counts: (sq_t, pre_t-bin)")
    # plt.tight_layout()
    plt.show()

    return counts















import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def _count_tensor(pre_disc, sq_disc, nbins_pre):
    """helper: counts[x_bin, y_prev, z_next]"""
    x_idx, y_idx, z_idx = pre_disc[:-1], sq_disc[:-1], sq_disc[1:]
    counts = np.zeros((nbins_pre, 2, 2), dtype=int)
    for xi, yi, zi in zip(x_idx, y_idx, z_idx):
        counts[xi, yi, zi] += 1
    return counts

def _delta_from_counts(counts):
    N_flip = counts[:, 0, 1] + counts[:, 1, 0]
    N_stay = counts[:, 0, 0] + counts[:, 1, 1]
    N_tot  = N_flip + N_stay
    with np.errstate(divide='ignore', invalid='ignore'):
        p_flip = np.divide(N_flip, N_tot, where=N_tot > 0)
        p_stay = np.divide(N_stay, N_tot, where=N_tot > 0)
    return p_stay - p_flip           # Δ_j for all bins

def prob_prebins_diffbar_surr(
    df_pre, df_sq,
    forcing_column='pre',
    target_column='sq',
    time_column='age',
    nbins_pre=6,
    n_surr=100,
    alpha=0.05,
    random_state=None,
    y_min=0.6,
    y_max=1.0
):
    """
    Bar plot of Δ = P(stay)-P(flip) per pre-bin
    + surrogate mean ±1σ and permutation p-value.
    Surrogates are generated by permuting the pre-bin sequence.
    """
    rng = np.random.default_rng(random_state)

    # ---------- 1) prep time series ----------
    pre_raw = df_pre[forcing_column].values[::-1]
    sq_raw  = df_sq[target_column].values[::-1]

    bins_pre = np.histogram_bin_edges(pre_raw, bins=nbins_pre)
    pre_disc = np.clip(np.digitize(pre_raw, bins_pre) - 1, 0, nbins_pre-1)
    sq_disc  = (sq_raw > 0).astype(int)

    # ---------- 2) observed Δ ----------
    counts_obs = _count_tensor(pre_disc, sq_disc, nbins_pre)
    delta_obs  = _delta_from_counts(counts_obs)        # shape (nbins_pre,)

    # ---------- 3) surrogates ----------
    delta_surr = np.zeros((n_surr, nbins_pre))
    for s in range(n_surr):
        pre_perm = rng.permutation(pre_disc)
        cnt_s    = _count_tensor(pre_perm, sq_disc, nbins_pre)
        delta_surr[s] = _delta_from_counts(cnt_s)

    mu_surr  = np.nanmean(delta_surr, axis=0)
    sd_surr  = np.nanstd (delta_surr, axis=0)

    # # p-value: two-sided tail
    # pvals = np.array([
    #     (np.sum(np.abs(delta_surr[:,j]) >= abs(delta_obs[j])) + 1)
    #     / (n_surr + 1)
    #     for j in range(nbins_pre)
    # ])

    pvals = np.empty(nbins_pre)
    for j in range(nbins_pre):
        if delta_obs[j] >= mu_surr[j]:
            tail = np.sum(delta_surr[:, j] >= delta_obs[j])
        else:
            tail = np.sum(delta_surr[:, j] <= delta_obs[j])
        pvals[j] = (tail + 1) / (n_surr + 1)



    # ---------- 4) plot ----------
    x = np.arange(nbins_pre)
    colors = np.where(delta_obs >= 0, 'C3', 'C2')   # green / red
    edgecols = np.where(pvals < alpha, 'k', 'grey')
    hatches  = ['' if p<alpha else '////' for p in pvals]



    # Prepare a red→blue colormap
    cmap = plt.get_cmap('coolwarm')  # reversed so low→red, high→blue
    cmap_arrows = plt.get_cmap('coolwarm_r')  # for arrows
    norm = plt.Normalize(0, nbins_pre - 1)
    colors = cmap(norm(np.arange(nbins_pre)))
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    
    # 1) Draw bars with the gradient colors
    for xi, d, c, ec, ht in zip(x, delta_obs, colors, edgecols, hatches):
        ax.bar(xi, d, color=c, edgecolor=ec, hatch=ht, linewidth=1.4)
    
    ax.errorbar(x, mu_surr, yerr=sd_surr, fmt='o', color='k',
                capsize=4, label='surrogate mean ±1σ')
    # ax.axhline(0, color='k', lw=4)

    # annotate Δ and p
    for xi, d, pv in zip(x, delta_obs, pvals):
        ax.text(xi, d + 0.02*np.sign(d),
                f'{d:+.2f}\n(p={pv:.3f})',
                ha='center',
                va='bottom' if d>=0 else 'top',
                fontsize=8)
    # … (labels, title, annotations of Δ and p remain the same) …
    ax.set_ylim(y_min, y_max)

    ax.set_xticks(x)
    ax.set_xlabel(f'{forcing_column} bin (0 … {nbins_pre-1})')
    ax.set_ylabel('Δ  =  P(stay) − P(flip)')
    # 2) Replace the two annotate-arrows with coloured LineCollections
    y0, y1 = ax.get_ylim()
    y_arrow = y0 + 0.03*(y1 - y0)

    # set the linewidth of spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    # helper to draw a gradient arrow from x_start→x_end
    def gradient_arrow(x_start, x_end, y, cmap, norm, ax, head_size=100, dir='left'):
        # make (say) 100 small segments
        xs = np.linspace(x_start, x_end, 200)
        ys = np.full_like(xs, y)
        pts = np.array([xs, ys]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap=cmap, norm=norm, linewidth=5)

        # color each segment by its mid-bin index
        mid_bins = (xs*(nbins_pre-1)/(nbins_pre-1)).clip(0, nbins_pre-1)
        lc.set_array(mid_bins)
        ax.add_collection(lc)
        c0 = cmap(norm(x_start))
        c1 = cmap(norm(x_end))

        # 3) draw small triangle‐markers at each end
        if dir == 'left':
            ax.scatter([x_end], [y],
                    marker='<',
                    s=head_size,     # head_size is area in pts^2
                    color=c1,
                    zorder=lc.get_zorder()+1)
        else:
            ax.scatter([x_end],   [y],
                    marker='>',
                    s=head_size,
                    color=c1,
                    zorder=lc.get_zorder()+1)

    
    # left arrow: from bin mid → bin 0 (decreasing pre)
    gradient_arrow(nbins_pre/2, 0, y_arrow, cmap_arrows, norm, ax, dir='left')
    ax.text(nbins_pre/6, y_arrow + 0.06*(y1-y0),
            'Precession ↓, Insolation ↑',
            ha='center', va='top', fontsize=10)
    
    # right arrow: from bin mid → last bin (increasing pre)
    gradient_arrow(nbins_pre/2, nbins_pre-1, y_arrow, cmap_arrows, norm, ax, dir='right')
    ax.text(nbins_pre*4/6, y_arrow + 0.06*(y1-y0),
            'Precession ↑, Insolation ↓',
            ha='center', va='top', fontsize=10)
    
    # extend the y-axis so your arrows are fully visible
    # ax.set_ylim(y0 - 0.12*(y1 - y0), y1)
    
    plt.tight_layout()
    plt.show()
    
    return delta_obs, mu_surr, sd_surr, pvals









import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection



# ---------- NEW helper: Δ = P(warm-stay) – P(cold-stay) ----------
def _staydiff_from_counts(counts):
    """return Δ_j for all pre-bins"""
    # warm-stay   P(1→1 | pre)
    warm_tot = counts[:, 1, 1] + counts[:, 1, 0]
    cold_tot = counts[:, 0, 0] + counts[:, 0, 1]

    with np.errstate(divide='ignore', invalid='ignore'):
        p_warm = np.divide(counts[:, 1, 1], warm_tot, where=warm_tot > 0)
        p_cold = np.divide(counts[:, 0, 0], cold_tot, where=cold_tot > 0)

    return p_warm - p_cold, p_warm, p_cold


# =================================================================
#  MAIN FUNCTION
# =================================================================
def prob_prebins_staydiff_surr(
    df_pre, df_sq,
    forcing_column='pre',
    target_column='sq',
    nbins_pre=6,
    n_surr=100,
    alpha=0.05,
    random_state=None
):
    """
    Δ = P(warm→warm) − P(cold→cold) for each pre-bin
    compared to permutation surrogates.
    """
    rng = np.random.default_rng(random_state)

    # ---------- 1) prepare series ----------
    pre_raw = df_pre[forcing_column].values[::-1]
    sq_raw  = df_sq [target_column].values[::-1]

    bins_pre = np.histogram_bin_edges(pre_raw, bins=nbins_pre)
    pre_disc = np.clip(np.digitize(pre_raw, bins=bins_pre) - 1, 0, nbins_pre-1)
    sq_disc  = (sq_raw > 0).astype(int)

    # ---------- 2) observed Δ ----------
    counts_obs = _count_tensor(pre_disc, sq_disc, nbins_pre)
    delta_obs, p_warm, p_cold = _staydiff_from_counts(counts_obs)

    # ---------- 3) surrogates ----------
    delta_surr = np.zeros((n_surr, nbins_pre))
    for k in range(n_surr):
        counts_k = _count_tensor(rng.permutation(pre_disc), sq_disc, nbins_pre)
        delta_surr[k] = _staydiff_from_counts(counts_k)[0]

    mu_surr = np.nanmean(delta_surr, axis=0)
    sd_surr = np.nanstd (delta_surr, axis=0)

    # p-values (one-sided toward observed sign)
    pvals = np.empty(nbins_pre)
    for j in range(nbins_pre):
        if delta_obs[j] >= mu_surr[j]:
            tail = np.sum(delta_surr[:, j]>= delta_obs[j])
        else:
            tail = np.sum(delta_surr[:, j] <= delta_obs[j])
        pvals[j] = (tail + 1)/(n_surr + 1)

    # ---------- 4) plot ----------
    x = np.arange(nbins_pre)
    cmap  = plt.get_cmap('coolwarm')
    cmapA = plt.get_cmap('coolwarm_r')
    colors = cmap(np.linspace(0, 1, nbins_pre))
    edgecols = np.where(pvals < alpha, 'k', 'grey')
    hatches  = ['' if p < alpha else '////' for p in pvals]

    fig, ax = plt.subplots(figsize=(6, 4))

    for xi, d, c, ec, ht in zip(x, delta_obs, colors, edgecols, hatches):
        ax.bar(xi, d, color=c, edgecolor=ec, hatch=ht, linewidth=1.4)

    ax.errorbar(x, mu_surr, yerr=sd_surr, fmt='o', color='k',
                capsize=4, label='surrogate mean ±1σ')

    for xi, d, pv in zip(x, delta_obs, pvals):
        ax.text(xi, d + 0.01*np.sign(d),
                f'{d:+.2f}\n(p={pv:.3f})',
                ha='center', va='bottom' if d>=0 else 'top', fontsize=8)

    ax.set_xticks(x)
    ax.set_xlabel(f'{forcing_column} bin (0 … {nbins_pre-1})')
    ax.set_ylabel('Δ  =  P(warm stay) − P(cold stay)')
    ax.set_ylim(-0.04, 0.04)

    # gradient arrows as in your previous version ------------------
    y0, y1 = ax.get_ylim();   y_arrow = y0 - 0.1*(y1 - y0)

    def gradient_arrow(x0, x1, y, cmap, ax, head='left'):
        xs = np.linspace(x0, x1, 200)
        ys = np.full_like(xs, y)
        segs = np.stack([xs, ys], axis=-1).reshape(-1, 1, 2)
        segs = np.concatenate([segs[:-1], segs[1:]], axis=1)
        lc = LineCollection(segs, cmap=cmap, norm=plt.Normalize(0, nbins_pre-1),
                            linewidth=5)
        lc.set_array(xs)
        ax.add_collection(lc)
        ax.scatter([x1], [y], marker='<' if head=='left' else '>',
                   s=90, color=cmap(xs[-1]/(nbins_pre-1)))

    gradient_arrow(nbins_pre/2, 0, y_arrow, cmapA, ax, 'left')
    gradient_arrow(nbins_pre/2, nbins_pre-1, y_arrow, cmapA, ax, 'right')
    ax.text(nbins_pre/6,  y_arrow+0.01, 'Precession ↓', ha='center', va='top')
    ax.text(nbins_pre*5/6,y_arrow+0.01, 'Precession ↑', ha='center', va='top')
    ax.set_ylim(y_arrow-0.01, 0.05)

    plt.legend(loc='upper right'); plt.tight_layout(); plt.show()

    return delta_obs, mu_surr, sd_surr, pvals






import numpy as np
import matplotlib.pyplot as plt

def local_stay_split(
    df_pre, df_sq,
    forcing_column='pre',
    target_column='sq',
    time_column='age',
    nbins_pre=4,
    smooth_win=200
):
    # ---------- 1) reverse to chronological order ----------
    pre_raw = df_pre[forcing_column].values[::-1]
    sq_raw  = df_sq[target_column].values[::-1]
    t       = df_pre[time_column].values[::-1]

    # ---------- 2) low-pre shading (optional) ----------
    q50   = np.quantile(pre_raw, 0.5)
    low   = pre_raw < q50
    edges = np.diff(low.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    ends   = list(np.where(edges == -1)[0] + 1)
    if low[0]:  starts.insert(0, 0)
    if low[-1]: ends.append(len(low))
    low_periods = list(zip(starts, ends))

    # ---------- 3) discretise ----------
    bins_pre = np.histogram_bin_edges(pre_raw, bins=nbins_pre)
    pre_disc = np.clip(np.digitize(pre_raw, bins_pre) - 1, 0, nbins_pre-1)
    sq_disc  = (sq_raw > 0).astype(int)          # 0 = cold, 1 = warm

    x_idx, y_idx, z_idx = pre_disc[:-1], sq_disc[:-1], sq_disc[1:]

    # ---------- 4) counts tensor ----------
    counts = np.zeros((nbins_pre, 2, 2), dtype=int)
    for xi, yi, zi in zip(x_idx, y_idx, z_idx):
        counts[xi, yi, zi] += 1

    # ---------- 5) conditional-prob tensor ----------
    totals = counts.sum(axis=2, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        cond_prob = counts / totals               # P(z | x, y)

    # ---------- helper: centred running mean that ignores NaNs ----------
    def smooth_nan(arr, win):
        arr      = np.asarray(arr, float)
        mask     = ~np.isnan(arr)
        arr_f    = np.where(mask, arr, 0.0)
        kern     = np.ones(win)
        num      = np.convolve(arr_f, kern, mode='same')
        denom    = np.convolve(mask.astype(float), kern, mode='same')
        out      = np.divide(num, denom, where=denom>0)
        out[denom == 0] = np.nan
        return out

    # ---------- 6) build the four local probability series ----------
    cold_stay_raw = np.array([
        cond_prob[x, 0, 0] if y == 0 else np.nan
        for x, y in zip(x_idx, y_idx)
    ])
    warm_stay_raw = np.array([
        cond_prob[x, 1, 1] if y == 1 else np.nan
        for x, y in zip(x_idx, y_idx)
    ])
    cold_stay = smooth_nan(cold_stay_raw, smooth_win)
    warm_stay = smooth_nan(warm_stay_raw, smooth_win)

    # ---------- 7) plotting ----------
    fig = plt.figure(figsize=(12, 4))
    gs  = fig.add_gridspec(2, 1, height_ratios=[0.8, 1.6], hspace=0)

    # (a) raw signals
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(t, pre_raw, label=forcing_column)
    ax0b = ax0.twinx()
    ax0b.plot(t, sq_raw, color='C1', label=target_column)
    for s, e in low_periods:
        ax0 .axvspan(t[s], t[e-1], color='grey', alpha=0.3)
        ax0b.axvspan(t[s], t[e-1], color='grey', alpha=0.3)
    ax0.set_ylabel(forcing_column)
    ax0b.set_ylabel(target_column)
    ax0.tick_params(axis='x', labelbottom=False)
    lns = ax0.get_lines() + ax0b.get_lines()
    ax0.legend(lns, [l.get_label() for l in lns], loc='upper right')
    ax0.set_xlim(t[1], t[-1])

    # (b) persistence probabilities
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ln1, = ax1.plot(t[1:], cold_stay, color='C0', label='P(cold→cold)')
    ax1.set_ylabel('P(cold stay)', color='C0')
    ax1.tick_params(axis='y', colors='C0')


    ax1b = ax1.twinx()
    ln2, = ax1b.plot(t[1:], warm_stay, color='C3', label='P(warm→warm)')
    ax1b.set_ylabel('P(warm stay)', color='C3')
    ax1b.tick_params(axis='y', colors='C3')

    for s, e in low_periods:
        ax1 .axvspan(t[s], t[e-1], color='grey', alpha=0.3)
        ax1b.axvspan(t[s], t[e-1], color='grey', alpha=0.3)

    ax1.set_xlabel(time_column)
    ax1.legend([ln1, ln2], ['cold persistence', 'warm persistence'],
               loc='upper right')

    plt.show()

    return cold_stay, warm_stay



























import numpy as np
import matplotlib.pyplot as plt

def local_prob(
    df_pre,
    df_sq,
    forcing_column='pre',
    target_column='sq',
    time_column='age',
    nbins_pre=4,
    smooth_win=200
):
    # 1) raw series (reverse → old→young if needed)
    pre_raw = df_pre[forcing_column].values[::-1]
    sq_raw  = df_sq[target_column].values[::-1]
    t       = df_pre[time_column].values[::-1]

    # 2) mark "low-pre" periods (here: below 50 % quantile as in your code)
    q50   = np.quantile(pre_raw, 0.5)
    low   = pre_raw < q50
    edges = np.diff(low.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    ends   = list(np.where(edges == -1)[0] + 1)
    if low[0]:  starts.insert(0, 0)
    if low[-1]: ends.append(len(low))
    low_periods = list(zip(starts, ends))

    # 3) discretise
    bins_pre = np.histogram_bin_edges(pre_raw, bins=nbins_pre)
    pre_disc = np.digitize(pre_raw, bins_pre) - 1        # 0…nbins_pre-1
    pre_disc = np.clip(pre_disc, 0, nbins_pre-1)
    sq_disc  = (sq_raw > 0).astype(int)                  # 0/1

    # past indices (length N-1) and future state
    x_idx = pre_disc[:-1]
    y_idx = sq_disc[:-1]
    z_idx = sq_disc[1:]

    # 4) build 3-D count tensor  (pre_bin, sq_prev, sq_next)
    counts = np.zeros((nbins_pre, 2, 2), dtype=int)
    for xi, yi, zi in zip(x_idx, y_idx, z_idx):
        counts[xi, yi, zi] += 1

    # 5) convert to conditional-prob tensor  P(z | x,y)
    cond_prob = np.zeros_like(counts, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        totals = counts.sum(axis=2, keepdims=True)
        cond_prob = np.divide(counts, totals,
                              where=totals>0)          # avoid /0

    # def smooth(arr, win=10):
    #     """Simple centred moving-average, same length as input."""
    #     kern = np.ones(win) / win
    #     return np.convolve(arr, kern, mode='same')

    def smooth(arr, win=10, mode='edge'):
        """
        Centre-aligned moving average with correct length for
        both even and odd window sizes.  Edge values are extended
        ('edge') or mirrored ('reflect').
        """
        arr  = np.asarray(arr)
        L    = len(arr)
        left = win // 2               # floor
        right = win - left - 1        # ensures left+right = win-1
        arr_p = np.pad(arr, (left, right), mode=mode)
        kern  = np.ones(win) / win
        sm    = np.convolve(arr_p, kern, mode='valid')   # length == L
        return sm




    # 6) time-series of flip / stay probabilities
    # flip_prob = np.array([cond_prob[x, y, 1-y] for x, y in zip(x_idx, y_idx)])
    # stay_prob = np.array([cond_prob[x, y,   y ] for x, y in zip(x_idx, y_idx)])

    flip_prob_raw = np.array([cond_prob[x, y, 1-y] for x, y in zip(x_idx, y_idx)])
    stay_prob_raw = np.array([cond_prob[x, y,   y ] for x, y in zip(x_idx, y_idx)])
    flip_prob = smooth(flip_prob_raw, win=200)
    stay_prob = smooth(stay_prob_raw, win=200)


    # -------- PLOT --------
    fig = plt.figure(figsize=(12, 4))
    gs  = fig.add_gridspec(2, 1, height_ratios=[0.8, 1.6], hspace=0)

    # top: original signals
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(t, pre_raw, label=f'{forcing_column} (raw)')
    ax0.set_ylabel(forcing_column)
    ax0b = ax0.twinx()
    ax0b.plot(t, sq_raw, color='C1', label=f'{target_column} (raw)')
    ax0b.set_ylabel(target_column)
    # shade low-pre epochs
    for s, e in low_periods:
        ax0.axvspan(t[s], t[e-1], color='gray', alpha=0.3)
        ax0b.axvspan(t[s], t[e-1], color='gray', alpha=0.3)
    lns = ax0.get_lines() + ax0b.get_lines()
    ax0.legend(lns, [l.get_label() for l in lns], loc='upper right')
    ax0.tick_params(axis='x', which='both', labelbottom=False)
    ax0.set_xlim(t[1], t[-1])


    ax1  = fig.add_subplot(gs[1], sharex=ax0)

    # left axis = flip probability
    ln1, = ax1.plot(t[1:], flip_prob, color='C2', label='P(flip | pre,sq)')
    ax1.set_ylabel('P(flip)', color='C2')
    flip_min, flip_max = flip_prob.min(), flip_prob.max()
    pad = 0.05 * (flip_max - flip_min + 1e-9)
    ax1.set_ylim(flip_min - pad, flip_max + pad)
    ax1.tick_params(axis='y', colors='C2')

    # right axis = stay probability
    ax1b = ax1.twinx()
    ln2, = ax1b.plot(t[1:], stay_prob, color='C3', label='P(stay | pre,sq)')
    ax1b.set_ylabel('P(stay)', color='C3')
    stay_min, stay_max = stay_prob.min(), stay_prob.max()
    pad = 0.05 * (stay_max - stay_min + 1e-9)
    ax1b.set_ylim(stay_min - pad, stay_max + pad)
    ax1b.tick_params(axis='y', colors='C3')

    ax1.set_xlabel(time_column)

    # shade low-pre epochs
    for s, e in low_periods:
        ax1 .axvspan(t[s], t[e-1], color='gray', alpha=0.3)
        ax1b.axvspan(t[s], t[e-1], color='gray', alpha=0.3)

    # combined legend
    ax1.legend([ln1, ln2], ['P(flip)', 'P(stay)'], loc='upper right')

    return flip_prob, stay_prob









def heatmap_binwise_local_te(
        pre, sq,
        pre_bins, sq_bins,
        *, k=1, cmap='viridis', vmin=None, vmax=None,
        te_func=transfer_entropy):
    """
    Compute and plot a bin-wise mean local TE(pre → sq) heat-map.

    Parameters
    ----------
    pre, sq : 1-D array-like, same length
        Time series (index 0 = oldest sample).  Both are REVERSED inside.
    pre_bins, sq_bins : int or 1-D array-like
        If int  -> number of equal-width histogram bins.
        If array -> explicit bin edges as for np.histogram.
    k : int
        History length for TE (default 1).
    cmap, vmin, vmax : passed to imshow.
    te_func : callable
        e.g. pyinform.transfer_entropy.
    """
    # -------- 0) prep -------------------------------------------------------
    pre = np.asarray(pre)[::-1]          # reverse to match your workflow
    sq  = np.asarray(sq )[::-1]
    if pre.shape != sq.shape:
        raise ValueError("pre and sq must have the same length")

    # -------- 1) build bin edges -------------------------------------------
    def _edges(data, spec):
        if np.isscalar(spec):                       # integer → histogram edges
            nb = int(spec)
            if nb < 1:
                raise ValueError("bins integer must be ≥1")
            return np.histogram_bin_edges(data, bins=nb)
        else:                                       # assume iterable of edges
            edges = np.asarray(spec, dtype=float)
            if edges.ndim != 1 or edges.size < 2:
                raise ValueError("bin edges must be 1-D with ≥2 values")
            # ensure monotone
            if not (np.all(np.diff(edges) > 0) or np.all(np.diff(edges) < 0)):
                raise ValueError("bin edges must be strictly monotonic")
            return edges

    pre_edges = _edges(pre, pre_bins)
    sq_edges  = _edges(sq , sq_bins)

    n_pre = len(pre_edges) - 1
    n_sq  = len(sq_edges)  - 1

    # -------- 2) digitise ---------------------------------------------------
    pre_bin = np.digitize(pre, pre_edges) - 1          # 0 … n_pre-1
    sq_bin  = np.digitize(sq , sq_edges ) - 1          # 0 … n_sq-1

    # -------- 3) local TE ---------------------------------------------------
    arr = te_func(pre_bin, sq_bin, k=k, local=True).flatten()
    if   arr.size == pre.size:        # pyinform returned N   → drop the first k
        local_te = arr[k:]
    elif arr.size == pre.size - k:    # pyinform returned N-k → already aligned
        local_te = arr
    else:
        raise ValueError(f"Unexpected local_te length {arr.size}")

    # past-state indices that correspond to each local TE
    idx_pre = pre_bin[:-k]
    idx_sq  = sq_bin [:-k]

    # -------- 4) accumulate by bin -----------------------------------------
    te_sum   = np.zeros((n_pre, n_sq))
    te_count = np.zeros_like(te_sum, dtype=int)

    for i, j, te in zip(idx_pre, idx_sq, local_te):
        if 0 <= i < n_pre and 0 <= j < n_sq:
            te_sum[i, j]   += te
            te_count[i, j] += 1

    te_grid = np.full_like(te_sum, np.nan, dtype=float)
    mask    = te_count > 0
    te_grid[mask] = te_sum[mask] / te_count[mask]

    # -------- 5) plot -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 2))
    im = ax.imshow(
        te_grid.T, origin='lower', aspect='auto',
        cmap=cmap, vmin=vmin, vmax=vmax,
        extent=[pre_edges[0], pre_edges[-1], sq_edges[0], sq_edges[-1]]
    )
    cbar = fig.colorbar(im, ax=ax, label='Mean local TE (bits)')
    ax.set_xlabel('pre bin')
    ax.set_ylabel('sq bin')
    ax.set_title('Bin-wise mean local transfer entropy')
    plt.tight_layout()
    plt.show()

    return te_grid







def local_TE(
    df_pre,
    df_sq,
    forcing_column='pre',
    target_column='sq',
    time_column='age',
    nbins_pre=4,
    nbins_target=2
):

    # --- 1) Extract raw numpy arrays and reverse if needed
    pre_raw = df_pre[forcing_column].values[::-1]
    sq_raw  = df_sq[target_column].values[::-1]
    t        = df_pre[time_column].values[::-1]

    # pre_raw = df_pre[forcing_column].values
    # sq_raw  = df_sq[target_column].values
    # t        = df_pre[time_column].values

    # --- 1.1) Find “low-pre” periods (below 25% quantile)
    q25   = np.quantile(pre_raw, 0.5)
    low   = pre_raw < q25
    # find rising/falling edges of the boolean mask
    edges = np.diff(low.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    ends   = list(np.where(edges == -1)[0] + 1)
    # if it starts low right away
    if low[0]:
        starts.insert(0, 0)
    # if it ends low
    if low[-1]:
        ends.append(len(low))
    # pair them
    low_periods = list(zip(starts, ends))

    # --- 2) Discretize
    bins_pre = np.histogram_bin_edges(pre_raw, bins=nbins_pre)
    pre_disc = np.digitize(pre_raw, bins_pre) - 1
    sq_disc  = (sq_raw > 0).astype(int)
    
    # --- 3) Compute local transfer entropy (unchanged)
    local_te = transfer_entropy(pre_disc, sq_disc, k=1, local=True)

    # --- 4) Plot setup with two panels
    fig = plt.figure(figsize=(12, 4))
    gs  = fig.add_gridspec(2, 1, height_ratios=[0.8, 1.6], hspace=0)

    # Top panel: pre & sq
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(t, pre_raw, label=f'{forcing_column} (raw)')
    ax0.set_ylabel(forcing_column)
    ax0.tick_params(axis='x', which='both', labelbottom=False)
    ax0.set_xlim(t[1], t[-1])
    
    ax0b = ax0.twinx()
    ax0b.plot(t, sq_raw, color='C1', label=f'{target_column} (raw)')
    ax0b.set_ylabel(target_column)

    # Shade low-pre on top panel
    for start, end in low_periods:
        ax0.axvspan(t[start], t[end-1], color='gray', alpha=0.3)
        ax0b.axvspan(t[start], t[end-1], color='gray', alpha=0.3)

    # combine legends
    lns = ax0.get_lines() + ax0b.get_lines()
    labs = [l.get_label() for l in lns]
    ax0.legend(lns, labs, loc='upper right')
    
    # Bottom panel: local TE
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax1.plot(t[1:], local_te.T, color='C2', label='Local TE')
    ax1.set_xlabel(time_column)
    ax1.set_ylabel('Local TE (bits)')
    ax1.legend(loc='upper right')

    # Shade low-pre on bottom panel
    for start, end in low_periods:
        ax1.axvspan(t[start], t[end-1], color='gray', alpha=0.3)

    plt.show()

    return local_te











import numpy as np
import matplotlib.pyplot as plt
from pyinform import transfer_entropy

def transfer_entropy_surrogate_test_v2(
    forcing, sq, k=1,
    forcing_bins=4, sq_bins=2,
    n_surr=100, p=0.05, use_quantile=False,
    if_plot=True, dpi=100
):
    """
    Test for unidirectional causality using transfer entropy and phase-randomized surrogates of `sq`.
    """
    # Reverse series
    x = np.asarray(forcing)[::-1]
    y = np.asarray(sq)[::-1]

    # Discretize data
    if use_quantile:
        xbins = np.quantile(x, np.linspace(0, 1, forcing_bins + 1))
        ybins = np.quantile(y, np.linspace(0, 1, sq_bins + 1))
    else:
        xbins = np.histogram_bin_edges(x, bins=forcing_bins)
        ybins = np.histogram_bin_edges(y, bins=sq_bins)

    x_disc = np.digitize(x, xbins) - 1
    y_disc = np.digitize(y, ybins) - 1

    # Empirical TE
    te_xy = transfer_entropy(x_disc[:-1], y_disc[1:], k=k)
    te_yx = transfer_entropy(y_disc[:-1], x_disc[1:], k=k)

    # Surrogate nulls via phase randomization of `y`
    null_xy = np.zeros(n_surr)
    null_yx = np.zeros(n_surr)
    N = len(y)
    # compute FFT once for original magnitude
    y_fft_orig = np.fft.rfft(y)
    mag = np.abs(y_fft_orig)
    for i in range(n_surr):
        # randomize phases
        phases = np.angle(y_fft_orig)
        random_phases = np.random.uniform(0, 2*np.pi, size=phases.shape)
        # preserve DC and Nyquist (if present)
        random_phases[0] = phases[0]
        if N % 2 == 0:
            random_phases[-1] = phases[-1]
        # build surrogate frequency domain
        y_surr_fft = mag * np.exp(1j * random_phases)
        # inverse FFT
        y_surr = np.fft.irfft(y_surr_fft, n=N)
        # discretize surrogate
        y_surr_disc = np.digitize(y_surr, ybins) - 1
        # compute TE for surrogate
        null_xy[i] = transfer_entropy(x_disc[:-1], y_surr_disc[1:], k=k)
        null_yx[i] = transfer_entropy(y_surr_disc[:-1], x_disc[1:], k=k)

    # p-values
    p_xy = (np.sum(null_xy >= te_xy) + 1) / (n_surr + 1)
    p_yx = (np.sum(null_yx >= te_yx) + 1) / (n_surr + 1)

    fig = None
    if if_plot:
        fig = plt.figure(figsize=(5, 3.5), dpi=dpi)
        plt.hist(null_xy, bins=25, alpha=0.7, label='Null TE (forcing→sq)', edgecolor='white', color = 'darkred')
        plt.axvline(te_xy, lw=2, label=f'TE (forcing→sq), p={p_xy:.3f}', color='darkred')
        plt.hist(null_yx, bins=25, alpha=0.7, label='Null TE (sq→forcing)', edgecolor='white', color = 'skyblue')
        plt.axvline(te_yx, lw=2, label=f'TE (sq→forcing), p={p_yx:.3f}', color='skyblue')
        plt.xlabel('Transfer Entropy (bits)')
        plt.ylabel('Count')
        plt.legend(loc='upper right', frameon=True)
        plt.tight_layout()
        plt.show()

    # significance
    sig_xy = p_xy < p
    sig_yx = p_yx < p
    return sig_xy and not sig_yx, fig











import numpy as np
import matplotlib.pyplot as plt
from pyinform import transfer_entropy
from sklearn.cluster import KMeans

def transfer_entropy_surrogate_test(
    forcing, sq, k=1,
    forcing_bins=4, sq_bins=2,
    n_surr=100, p=0.05, 
    sq_method='hist',  # options: 'hist', 'quantile', 'kmeans'
    binary=False,
    if_plot=True, dpi=100
):
    """
    Test for unidirectional causality using transfer entropy and surrogates.
    """
    # Reverse series
    x = np.asarray(forcing)[::-1]
    y = np.asarray(sq)[::-1]


    # Discretize sq series
    if sq_method == 'quantile':
        # xbins = np.quantile(x, np.linspace(0, 1, forcing_bins + 1))
        ybins = np.quantile(y, np.linspace(0, 1, sq_bins + 1))
        y_disc = np.digitize(y, ybins) - 1
    elif sq_method == 'hist':
        # xbins = np.histogram_bin_edges(x, bins=forcing_bins)
        ybins = np.histogram_bin_edges(y, bins=sq_bins)
        y_disc = np.digitize(y, ybins) - 1

    elif sq_method == 'kmeans':
        # xbins = np.histogram_bin_edges(x, bins=forcing_bins)
        # Fit k-means on the values
        km = KMeans(n_clusters=sq_bins, n_init=10, random_state=0)
        y_disc = km.fit_predict(y.reshape(-1, 1))
    else:
        raise ValueError(f"Unknown sq_method '{sq_method}'. Use 'hist', 'quantile', or 'kmeans'.")

    if binary:
        y_disc = y

    xbins = np.histogram_bin_edges(x, bins=forcing_bins)
    # xbins = np.histogram_bin_edges(x, bins=forcing_bins)
    x_disc = np.digitize(x, xbins) - 1

    
    # Empirical TE (one-step, history k)
    te_xy = transfer_entropy(x_disc[:-1], y_disc[1:], k=k)
    te_yx = transfer_entropy(y_disc[:-1], x_disc[1:], k=k)
    
    # Surrogate nulls
    null_xy = np.zeros(n_surr)
    null_yx = np.zeros(n_surr)
    for i in range(n_surr):
        xs = np.random.permutation(x_disc)
        null_xy[i] = transfer_entropy(xs[:-1], y_disc[1:], k=k)
        ys = np.random.permutation(y_disc)
        null_yx[i] = transfer_entropy(ys[:-1], x_disc[1:], k=k)
    
    # p-values
    p_xy = (np.sum(null_xy >= te_xy) + 1) / (n_surr + 1)
    p_yx = (np.sum(null_yx >= te_yx) + 1) / (n_surr + 1)
    
    fig=[]
    # Plot if requested
    if if_plot:
        fig = plt.figure(figsize=(5, 3.5), dpi=dpi)

        # forcing → sq histogram
        plt.hist(
            null_xy, bins=25,
            color='#CC6677',       # light blue
            alpha=0.7,
            label='Null TE (forcing→sq)',
            edgecolor='white'
        )
        # forcing → sq empirical line
        plt.axvline(
            te_xy,
            color='#882255',       # teal
            lw=2,
            label=f'TE (forcing→sq), p={p_xy:.3f}'
        )
        # sq → forcing histogram
        plt.hist(
            null_yx, bins=25,
            color='#88CCEE',       # pink-red
            alpha=0.7,
            label='Null TE (sq→forcing)',
            edgecolor='white'
        )
        # sq → forcing empirical line
        plt.axvline(
            te_yx,
            color='#44AA99',       # dark magenta
            lw=2,
            label=f'TE (sq→forcing), p={p_yx:.3f}'
        )
        # set the line width of spines
        for spine in plt.gca().spines.values():
            spine.set_linewidth(1.5)

        plt.xlabel('Transfer Entropy (bits)')
        plt.ylabel('Count')
        plt.legend(loc='upper right', frameon=True)
        plt.tight_layout()
        plt.show()
    
    # Determine unidirectional significance
    sig_xy = p_xy < p
    sig_yx = p_yx < p
    return sig_xy and not sig_yx, fig
















def create_shift_forcing(df_sq, interval, if_plot=False,shift=-10000):
    """
    Resample the square‐wave and load & resample
    the precession & obliquity forcing to a common age grid.

    Parameters
    ----------
    df_sq : pandas.DataFrame
        Must have columns ['age', 'sq'].
    interval : float
        Desired age‐step for the resampled grid.
    if_plot : bool, default False
        If True, plots the three resampled series.

    Returns
    -------
    df_sq_resampled, df_pre_resampled, df_obl_resampled : DataFrames
        Each has columns ['age', <variable>] on the same age grid.
    """
    # 1) load raw precession & obliquity
    pre_path = r"D:\VScode\bipolar_seesaw_CCM\inso_data\pre_800_inter100.txt"
    obl_path = r"D:\VScode\bipolar_seesaw_CCM\inso_data\obl_800_inter100.txt"
    df_pre_raw = pd.read_csv(pre_path, sep=r'\s+', header=None, engine='python')
    df_obl_raw = pd.read_csv(obl_path, sep=r'\s+', header=None, engine='python')

    # convert to years & ensure age increasing
    df_pre_raw.iloc[:,0] = df_pre_raw.iloc[:,0].abs() * 1000
    df_obl_raw.iloc[:,0] = df_obl_raw.iloc[:,0].abs() * 1000
    df_pre_raw = df_pre_raw.iloc[::-1].reset_index(drop=True)
    df_obl_raw = df_obl_raw.iloc[::-1].reset_index(drop=True)
    df_pre_raw.columns = ['age','pre']
    df_obl_raw.columns = ['age','obl']

    df_pre_raw['age'] = df_pre_raw['age'].values + shift
    df_obl_raw['age'] = df_obl_raw['age'].values + shift


    # 2) compute overlapping age bounds
    a_min = max(df_sq['age'].min(),
                df_pre_raw['age'].min(),
                df_obl_raw['age'].min())
    a_max = min(df_sq['age'].max(),
                df_pre_raw['age'].max(),
                df_obl_raw['age'].max())

    # 3) create unified age vector
    new_age = np.arange(a_min, a_max + 1, interval)

    # 4) interpolate each series onto new_age
    def interp(df, col):
        f = interp1d(df['age'], df[col],
                     kind='nearest',
                     bounds_error=False,
                     fill_value="extrapolate")
        return f(new_age)
    
    column_names = df_sq.columns[1]

    df_sq_rs  = pd.DataFrame({'age': new_age,
                              column_names: interp(df_sq, column_names)})
    df_pre_rs = pd.DataFrame({'age': new_age,
                              'pre': interp(df_pre_raw, 'pre')})
    df_obl_rs = pd.DataFrame({'age': new_age,
                              'obl': interp(df_obl_raw, 'obl')})

    # 5) optional plotting
    if if_plot:
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, tight_layout=True)
        axes[0].plot(df_sq_rs['age'],  df_sq_rs[column_names],  color='black', label='sq')
        axes[1].plot(df_pre_rs['age'], df_pre_rs['pre'], color='blue',  label='pre')
        axes[2].plot(df_obl_rs['age'], df_obl_rs['obl'], color='green', label='obl')
        for ax in axes:
            ax.legend()
            ax.set_ylabel(ax.get_label())
        axes[-1].set_xlabel('Age (years)')
        plt.show()

    return df_sq_rs, df_pre_rs, df_obl_rs















import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def interpolate_data_lr04(
    df_sq,
    lr04_path=r"D:\VScode\bipolar_seesaw_CCM\other_data\lr04.xlsx",
    sheet_name='Sheet1',
    if_plot=False
):
    """
    Interpolate the LR04 global benthic δ18O stack onto the
    age grid already contained in *df_sq* (e.g., the filtered CH4 grid).

    Parameters
    ----------
    df_sq : pandas.DataFrame
        Must contain an 'age' column (yr BP).  Only the age axis is used.
    lr04_path : str
        Full path to lr04.xlsx (two columns: age [kyr], d18O).
    sheet_name : str or int
        Worksheet index / name inside the Excel file.
    if_plot : bool
        When True, show the raw LR04 series and the interpolated version
        on the df_sq grid.

    Returns
    -------
    df_lr04_interp : pandas.DataFrame
        Columns ['age', 'd18O'] on exactly the same age grid as df_sq.
    """
    # ---------- 1. read & tidy LR04 --------------------
    df_lr04 = pd.read_excel(lr04_path, sheet_name=sheet_name)
    df_lr04.columns = ['age', 'd18O']           # enforce names
    df_lr04['age'] *= 1_000                     # kyr → yr BP

    # ---------- 2. interpolate onto df_sq ages --------
    ages_target = df_sq['age'].values
    d18o_interp = np.interp(ages_target,
                            df_lr04['age'].values,
                            df_lr04['d18O'].values)

    df_lr04_interp = pd.DataFrame({
        'age': ages_target,
        'd18O': d18o_interp
    })

    # ---------- 3. optional plots ---------------------
    if if_plot:
        # (a) raw LR04
        plt.figure(figsize=(10, 3))
        plt.plot(df_lr04['age'], df_lr04['d18O'], lw=0.8, label='LR04 raw')
        plt.gca().invert_xaxis(); plt.gca().invert_yaxis()
        plt.xlabel('Age (yr BP)'); plt.ylabel('δ18O')
        plt.title('LR04 stack (raw)')
        plt.tight_layout(); plt.show()

        # (b) interpolated
        plt.figure(figsize=(10, 3))
        plt.plot(df_lr04_interp['age'], df_lr04_interp['d18O'],
                 lw=1.0, label='LR04 interpolated')
        plt.gca().invert_xaxis(); plt.gca().invert_yaxis()
        plt.xlabel('Age (yr BP)'); plt.ylabel('δ18O')
        plt.title('LR04 interpolated to CH4 grid')
        plt.tight_layout(); plt.show()

    return df_lr04_interp












import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
plt.rcParams.update({
    "figure.dpi": 100,
    "axes.linewidth": 0.8,
    "grid.linestyle": "--",
    "grid.alpha": 0.4,
    "font.size": 9,          # main text
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
})





def age_gap_ana(
        df_ch4: pd.DataFrame,
        age_min: int = 0,
        age_max: int = 640_000,
        pre_path = "pre_800_inter100.txt",
        n_bins: int = 6,
        tolerance: int = 200):
    """
    Plots CH₄ sampling resolution (Δage) against orbital precession.
    Raises ValueError if either record does not cover [age_min, age_max].
    Returns (fig_pair, fig_bar) – two ready-to-use Matplotlib Figure objects.
    """

    # -------------------------------------------------
    # 0. Validate coverage of the requested time range
    # -------------------------------------------------
    df_ch4 = df_ch4.sort_values('age').reset_index(drop=True)
    if not (df_ch4['age'].min() <= age_min and df_ch4['age'].max() >= age_max):
        raise ValueError(
            f"CH₄ data span {df_ch4['age'].min():g}-{df_ch4['age'].max():g} yr BP; "
            f"requested window {age_min}-{age_max} yr BP is outside that range."
        )

    # -------------------------------------------------
    # 1. Crop first, *then* compute Δage            ▼
    # -------------------------------------------------
    df_ch4 = df_ch4.query('age >= @age_min and age <= @age_max').reset_index(drop=True)
    if len(df_ch4) < 2:                               #        ▼
        raise ValueError("Not enough CH₄ points in the selected age window.")  # ▼
    df_ch4['diff_age'] = df_ch4['age'].diff().abs()
    df_ch4 = df_ch4.dropna(subset=['diff_age'])

    # -------------------------------------------------
    # 2. Load & crop the raw precession record
    # -------------------------------------------------
    df_pre = (pd.read_csv(pre_path, sep=r'\s+', header=None, engine='python')
                .rename(columns={0: 'age', 1: 'pre'}))
    df_pre['age'] = df_pre['age'].abs() * 1000
    df_pre = df_pre.iloc[::-1].reset_index(drop=True)

    if not (df_pre['age'].min() <= age_min and df_pre['age'].max() >= age_max):
        raise ValueError(
            f"Precession file span {df_pre['age'].min():g}-{df_pre['age'].max():g} yr BP; "
            f"requested window {age_min}-{age_max} yr BP is outside that range."
        )

    df_pre = df_pre.query('age >= @age_min and age <= @age_max').reset_index(drop=True)

    # -------------------------------------------------
    # 3. Nearest-age merge
    # -------------------------------------------------
    df_merged = (pd.merge_asof(
                    df_ch4, df_pre,
                    on='age', direction='nearest', tolerance=tolerance)
                   .dropna(subset=['pre']))

        
    # 4a. Pair of time-series panels
    fig_pair, ax = plt.subplots(
        2, 1, figsize=(4.3, 4.0), sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1]}
    )

    # -- Precession
    ax[0].plot(df_merged["age"], df_merged["pre"],
            lw=1.1, color="#0072B2")                 # CB-friendly blue
    ax[0].set_ylabel("Precession (arb. u.)")
    ax[0].set_title("(a) Orbital precession", loc="left", fontsize=9, fontweight="bold")
    ax[0].grid(True)

    # -- Δage
    ax[1].plot(df_merged["age"], df_merged["diff_age"],
            lw=1.1, color="#D55E00")                 # CB-friendly orange
    ax[1].set_ylabel("Δage (yr)")
    ax[1].set_xlabel("Age (kyr BP)")
    ax[1].set_title("(b) Methane sampling step", loc="left", fontsize=9, fontweight="bold")
    ax[1].grid(True)

    # -- x-axis: invert and label in kyr

    ax[1].invert_xaxis()
    ax[1].xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, pos: f"{int(x/1_000)}")
    )

    fig_pair.align_ylabels()
    fig_pair.tight_layout()

    # 4b. Bar chart: mean Δage per precession bin
    df_merged["pre_bin"], bin_edges = pd.cut(
        df_merged["pre"], n_bins, labels=False, retbins=True
    )
    avg_by_bin = (df_merged.groupby("pre_bin", observed=True)["diff_age"]
                            .mean()
                            .rename("mean_diff_age")
                            .reset_index())

    # Bar centres (bin mid-points) and widths (bin widths)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths  = np.diff(bin_edges)

    fig_bar, axb = plt.subplots(figsize=(3.5, 2.3))
    axb.bar(bin_centres, avg_by_bin["mean_diff_age"],
            width=bin_widths, align="center", edgecolor="k",
            color="#009E73", lw=0.5)
    axb.set_xlabel("Precession amplitude")
    axb.set_ylabel("Mean Δage (yr)")
    axb.set_title("Sampling resolution vs precession", fontsize=9, pad=4)
    axb.grid(axis="y")

    # X-tick labels = bin mid-points in prec units (rounded)
    xb_labels = [f"{v:.0f}" for v in bin_centres]
    axb.set_xticks(bin_centres)
    axb.set_xticklabels(xb_labels, rotation=0)

    fig_bar.tight_layout()
    return fig_pair, fig_bar

















import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def age_gap_interactive(
    df_ch4: pd.DataFrame,
    age_min: int = 0,
    age_max: int = 640_000,
    H_line: int = 100
) -> go.Figure:
    """
    Computes and plots the methane sampling age step (Δage) over time interactively using Plotly,
    and prints the maximum Δage within the specified age window.

    Parameters
    ----------
    df_ch4 : pd.DataFrame
        DataFrame containing at least an 'age' column (years BP).
    age_min : int, optional
        Minimum age (yr BP) to include in the analysis. Default is 0.
    age_max : int, optional
        Maximum age (yr BP) to include in the analysis. Default is 640,000.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive Plotly figure of Δage vs. age.

    Raises
    ------
    ValueError
        If the data does not span the requested age range or has fewer than 2 points.
    """
    # Ensure sorted
    df = df_ch4.sort_values('age').reset_index(drop=True)

    # Validate coverage
    if df['age'].min() > age_min or df['age'].max() < age_max:
        raise ValueError(
            f"Data span {df['age'].min():g}-{df['age'].max():g} yr BP; "
            f"requested window {age_min}-{age_max} yr BP is outside that range."
        )

    # Crop to window
    df = df.query('age >= @age_min and age <= @age_max').reset_index(drop=True)
    if len(df) < 2:
        raise ValueError("Not enough data points in the selected age window.")

    # Compute Δage
    df['diff_age'] = df['age'].diff().abs()
    df = df.dropna(subset=['diff_age'])

    # Find and print maximum Δage
    max_gap = df['diff_age'].max()
    print(f"Maximum Δage within {age_min}-{age_max} yr BP: {max_gap:.0f} years")
    # print 95% quantile
    quantile_95 = df['diff_age'].quantile(0.95)
    print(f"95% quantile of Δage: {quantile_95:.0f} years")

    # Interactive plot
    fig = px.line(
        df,
        x='age',
        y='diff_age',
        labels={
            'age': 'Age (yr BP)',
            'diff_age': 'Δage (yr)'
        },
        title=''
    )

    # plot a vertical line at H_line
    fig.add_shape(
        type='line',
        x0=age_min, x1=age_max,
        y0=H_line, y1=H_line,
        line=dict(color='red', width=1, dash='dash'),
        name='H Line'
    )

    # Invert x-axis to show decreasing age
    fig.update_xaxes(autorange='reversed')

    fig.update_layout(
        xaxis_title='Age (kyr BP)',
        yaxis_title='Δage (yr)',
        title=dict(x=0.5)
    )

    return fig



































import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyinform import transfer_entropy
import matplotlib.ticker as mticker

# ──────────────────────────────────────────────────────────────────────────────
def te_vs_dt_scan(
        df_sq_raw,
        dt_grid,                 # iterable of resampling steps (yr)
        *,                       # keyword-only after here
        forcing='pre',           # 'pre' or 'obl'
        k=1,                     # TE history length
        forcing_bins=6, sq_bins=2,
        n_surr=100, alpha=0.05,
        sq_method='hist',        # 'hist', 'quantile', 'kmeans'
        dpi=150, figsize=(4.2, 3.2),
        return_fig=True):
    """
    Compute & plot TE(forcing → sq) versus resampling interval Δt.

    Parameters
    ----------
    forcing : {'pre', 'obl'}
        Which astronomical forcing to treat as the driver time-series.

    Returns
    -------
    results : pandas.DataFrame
        Columns ['dt','te_xy','p_xy','te_yx','p_yx','sig_uni']
    (fig, ax) : Matplotlib Figure and Axes (only if return_fig=True)
    """
    if forcing not in {'pre', 'obl'}:
        raise ValueError("forcing must be 'pre' or 'obl'")

    results = []

    # ------------------------------------------------------------------ loop over Δt
    for dt in dt_grid:
        df_sq_i, df_pre_i, df_obl_i = sa.interpolate_data_forcing(
            df_sq_raw.copy(), interval=dt, if_plot=False)

        # choose the driver series -------------------------------------
        if forcing == 'pre':
            x = df_pre_i['pre'].to_numpy()[::-1]      # chronological order
            y_label = 'TE (pre → sq)'
            y_rev   = 'TE (sq → pre)'
        else:  # 'obl'
            x = df_obl_i['obl'].to_numpy()[::-1]
            y_label = 'TE (obl → sq)'
            y_rev   = 'TE (sq → obl)'

        y = df_sq_i.iloc[:, 1].to_numpy()[::-1]       # filt_ch4 target

        # --------------- discretise -----------------------------------
        x_disc = np.digitize(x, np.histogram_bin_edges(x, forcing_bins)) - 1

        if sq_method == 'quantile':
            ybins = np.quantile(y, np.linspace(0, 1, sq_bins + 1))
            y_disc = np.digitize(y, ybins) - 1
        elif sq_method == 'kmeans':
            from sklearn.cluster import KMeans
            y_disc = KMeans(n_clusters=sq_bins, n_init=10,
                            random_state=0).fit_predict(y.reshape(-1, 1))
        else:  # 'hist'
            y_disc = (np.digitize(
                y, np.histogram_bin_edges(y, sq_bins)) - 1)

        # --------------- empirical TE ---------------------------------
        te_xy = transfer_entropy(x_disc[:-1], y_disc[1:], k=k)
        te_yx = transfer_entropy(y_disc[:-1], x_disc[1:], k=k)

        # --------------- surrogate test -------------------------------
        null_xy = np.empty(n_surr)
        null_yx = np.empty(n_surr)
        for i in range(n_surr):
            null_xy[i] = transfer_entropy(
                np.random.permutation(x_disc)[:-1], y_disc[1:], k=k)
            null_yx[i] = transfer_entropy(
                np.random.permutation(y_disc)[:-1], x_disc[1:], k=k)

        p_xy = (null_xy >= te_xy).sum() / n_surr
        p_yx = (null_yx >= te_yx).sum() / n_surr

        results.append(dict(dt=dt, te_xy=te_xy, p_xy=p_xy,
                            te_yx=te_yx, p_yx=p_yx,
                            sig_uni=(p_xy < alpha and p_yx >= alpha)))

    results = pd.DataFrame(results).sort_values('dt').reset_index(drop=True)

    # ------------------------------------------------------------------ plotting
    if not return_fig:
        return results

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(results['dt'], results['te_xy'],
            color='0.3', lw=1.2, marker='o',
            mfc='none', label=y_label)

    # highlight significant points
    sig = results['sig_uni']
    ax.scatter(results.loc[sig, 'dt'],
               results.loc[sig, 'te_xy'],
               s=60, color='#D55E00', zorder=5,
               label=f'Significant (α={alpha})')

    # show reverse TE
    ax.plot(results['dt'], results['te_yx'],
            ls='--', lw=1, color='0.6', label=y_rev)

    ax.set_xscale('log')
    ax.set_xlabel('Resampling interval Δt (yr)')
    ax.set_ylabel('Transfer Entropy (bits)')
    ax.set_title(f'Scale-dependence of TE  ({forcing} → sq)')
    ax.grid(True, which='both', alpha=0.3)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%g'))
    ax.legend(frameon=True, fontsize=8)
    fig.tight_layout()

    return results, (fig, ax)















def interpolate_data_forcing(df_sq, interval, if_plot=False):
    """
    Resample the square‐wave and load & resample
    the precession & obliquity forcing to a common age grid.

    Parameters
    ----------
    df_sq : pandas.DataFrame
        Must have columns ['age', 'sq'].
    interval : float
        Desired age‐step for the resampled grid.
    if_plot : bool, default False
        If True, plots the three resampled series.

    Returns
    -------
    df_sq_resampled, df_pre_resampled, df_obl_resampled : DataFrames
        Each has columns ['age', <variable>] on the same age grid.
    """
    # 1) load raw precession & obliquity
    pre_path = r"D:\VScode\bipolar_seesaw_CCM\inso_data\pre_800_inter100.txt"
    obl_path = r"D:\VScode\bipolar_seesaw_CCM\inso_data\obl_800_inter100.txt"
    df_pre_raw = pd.read_csv(pre_path, sep=r'\s+', header=None, engine='python')
    df_obl_raw = pd.read_csv(obl_path, sep=r'\s+', header=None, engine='python')

    # convert to years & ensure age increasing
    df_pre_raw.iloc[:,0] = df_pre_raw.iloc[:,0].abs() * 1000
    df_obl_raw.iloc[:,0] = df_obl_raw.iloc[:,0].abs() * 1000
    df_pre_raw = df_pre_raw.iloc[::-1].reset_index(drop=True)
    df_obl_raw = df_obl_raw.iloc[::-1].reset_index(drop=True)
    df_pre_raw.columns = ['age','pre']
    df_obl_raw.columns = ['age','obl']

    # 2) compute overlapping age bounds
    a_min = max(df_sq['age'].min(),
                df_pre_raw['age'].min(),
                df_obl_raw['age'].min())
    a_max = min(df_sq['age'].max(),
                df_pre_raw['age'].max(),
                df_obl_raw['age'].max())

    # 3) create unified age vector
    new_age = np.arange(a_min, a_max + 1, interval)

    # 4) interpolate each series onto new_age
    def interp(df, col):
        f = interp1d(df['age'], df[col],
                     kind='nearest',
                     bounds_error=False,
                     fill_value="extrapolate")
        return f(new_age)
    
    column_names = df_sq.columns[1]

    df_sq_rs  = pd.DataFrame({'age': new_age,
                              column_names: interp(df_sq, column_names)})
    df_pre_rs = pd.DataFrame({'age': new_age,
                              'pre': interp(df_pre_raw, 'pre')})
    df_obl_rs = pd.DataFrame({'age': new_age,
                              'obl': interp(df_obl_raw, 'obl')})

    # 5) optional plotting
    if if_plot:
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, tight_layout=True)
        axes[0].plot(df_sq_rs['age'],  df_sq_rs[column_names],  color='black', label='sq')
        axes[1].plot(df_pre_rs['age'], df_pre_rs['pre'], color='blue',  label='pre')
        axes[2].plot(df_obl_rs['age'], df_obl_rs['obl'], color='green', label='obl')
        for ax in axes:
            ax.legend()
            ax.set_ylabel(ax.get_label())
        axes[-1].set_xlabel('Age (years)')
        plt.show()

    return df_sq_rs, df_pre_rs, df_obl_rs



# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import find_peaks
# from scipy.stats import ttest_ind

# def plot_phase_durations(df_pre, df_sq, lags=range(0, 901, 100)):
#     """
#     For each lag in `lags`, shifts df_pre manually (introducing NaNs),
#     truncates df_sq accordingly, computes rising/falling phase durations,
#     runs t-tests, and produces the same two-panel figure per lag.
    
#     Parameters
#     ----------
#     df_pre : pandas.DataFrame
#         Must contain a 'pre' column (precession) and matching index.
#     df_sq : pandas.DataFrame
#         Must contain a 'sq' column (square wave ±1) and matching index.
#     lags : iterable of ints, optional
#         List of integer shifts (in data points) to test.
#     """
#     pre_vals_orig = df_pre['pre'].values
#     sq_orig       = np.where(df_sq['sq'].values > 0, 1, -1)
#     N             = len(pre_vals_orig)

#     def count_transitions(intervals, sq_sign):
#         lt, hl = [], []
#         for i0, i1 in intervals:
#             seg = sq_sign[i0:i1+1]
#             lt.append(np.sum((seg[:-1]==-1) & (seg[1:]==+1)))
#             hl.append(np.sum((seg[:-1]==+1) & (seg[1:]==-1)))
#         return lt, hl

#     for lag in lags:
#         # 1) manual shift with NaNs
#         pre_shifted = np.full(N, np.nan)
#         if lag > 0:
#             pre_shifted[0:N-lag] = pre_vals_orig[lag:N]
#         elif lag < 0:
#             k = -lag
#             pre_shifted[k:N] = pre_vals_orig[0:N-k]
#         else:
#             pre_shifted[:] = pre_vals_orig

#         # 2) truncate to non-NaNs
#         valid    = ~np.isnan(pre_shifted)
#         pre_vals = pre_shifted[valid]
#         sq_sign  = sq_orig[valid]

#         # 3) peak/trough detection
#         peaks,   _ = find_peaks(pre_vals)
#         troughs, _ = find_peaks(-pre_vals)

#         # 4) rising & falling intervals
#         rising, falling = [], []
#         for t in troughs:
#             nxt = peaks[peaks > t]
#             if nxt.size: rising.append((t, nxt[0]))
#         for p in peaks:
#             nxt = troughs[troughs > p]
#             if nxt.size: falling.append((p, nxt[0]))

#         # 5) compute durations for each phase
#         r_cold, r_warm = [], []
#         for i0, i1 in rising:
#             seg = sq_sign[i0:i1+1]
#             r_cold.append(np.sum(seg == -1))
#             r_warm.append(np.sum(seg == +1))

#         d_cold, d_warm = [], []
#         for i0, i1 in falling:
#             seg = sq_sign[i0:i1+1]
#             d_cold.append(np.sum(seg == -1))
#             d_warm.append(np.sum(seg == +1))

#         # 6) t-tests on durations
#         t_r, p_r = ttest_ind(r_cold, r_warm, equal_var=False)
#         sig_r = 'Yes' if p_r < 0.05 else 'No'
#         t_d, p_d = ttest_ind(d_cold, d_warm, equal_var=False)
#         sig_d = 'Yes' if p_d < 0.05 else 'No'

#         # 7) plotting (identical to original)
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4), tight_layout=True)
#         fig.suptitle(f"Lag = {lag} indices", fontsize=14)

#         ax1.hist(r_cold, bins='auto', alpha=0.5, color='green', label='cold length')
#         ax1.axvline(np.mean(r_cold), color='green', linestyle='dashed', linewidth=1)
#         ax1.hist(r_warm, bins='auto', alpha=0.5, color='red',   label='warm length')
#         ax1.axvline(np.mean(r_warm), color='red', linestyle='dashed', linewidth=1)
#         ax1.set_title(f"Rising phases\n t={t_r:.2f}, p={p_r:.2e}\nSig? {sig_r}")
#         ax1.set_xlabel('Duration (data points)')
#         ax1.set_ylabel('Frequency')
#         ax1.legend()

#         ax2.hist(d_cold, bins='auto', alpha=0.5, color='green', label='cold length')
#         ax2.axvline(np.mean(d_cold), color='green', linestyle='dashed', linewidth=1)
#         ax2.hist(d_warm, bins='auto', alpha=0.5, color='red',   label='warm length')
#         ax2.axvline(np.mean(d_warm), color='red', linestyle='dashed', linewidth=1)
#         ax2.set_title(f"Decreasing phases\n t={t_d:.2f}, p={p_d:.2e}\nSig? {sig_d}")
#         ax2.set_xlabel('Duration (data points)')
#         ax2.legend()

#         plt.show()



# import numpy as np
# from scipy.signal import find_peaks
# from scipy.stats import ttest_ind
# import matplotlib.pyplot as plt

# def plot_transition_distribution(df_pre, df_sq, lags=None):
#     """
#     For each lag in `lags`, shifts df_pre manually (introducing NaNs),
#     truncates df_sq accordingly, computes rising/falling phase transitions,
#     runs t-tests, and produces the same two-panel figure per lag.

#     Parameters
#     ----------
#     df_pre : pandas.DataFrame
#         Must contain a 'pre' column.
#     df_sq : pandas.DataFrame
#         Must contain a 'sq' column with values ±1.
#     lags : iterable of ints, optional
#         List of integer shifts (in data points) to test.
#         Default is range(0, 901, 100).
#     """
#     if lags is None:
#         lags = range(0, 901, 100)

#     pre_vals_orig = df_pre['pre'].values
#     sq_orig       = np.where(df_sq['sq'].values > 0, 1, -1)
#     N             = len(pre_vals_orig)

#     def count_transitions(intervals, sq_sign):
#         lt, hl = [], []
#         for i0, i1 in intervals:
#             seg = sq_sign[i0:i1+1]
#             lt.append(np.sum((seg[:-1]==-1) & (seg[1:]==+1)))
#             hl.append(np.sum((seg[:-1]==+1) & (seg[1:]==-1)))
#         return lt, hl

#     for lag in lags:
#         # 1) manual shift with NaNs instead of roll()
#         pre_shifted = np.full(N, np.nan)
#         if lag > 0:
#             pre_shifted[0:N-lag] = pre_vals_orig[lag:N]
#         elif lag < 0:
#             k = -lag
#             pre_shifted[k:N] = pre_vals_orig[0:N-k]
#         else:
#             pre_shifted[:] = pre_vals_orig

#         # 2) mask out NaNs & truncate sq to match
#         valid    = ~np.isnan(pre_shifted)
#         pre_vals = pre_shifted[valid]
#         sq_sign  = sq_orig[valid]

#         # 3) detect peaks/troughs
#         peaks,   _ = find_peaks(pre_vals)
#         troughs, _ = find_peaks(-pre_vals)

#         # 4) rising & falling intervals
#         rising, falling = [], []
#         for t in troughs:
#             nxt = peaks[peaks > t]
#             if nxt.size: rising.append((t, nxt[0]))
#         for p in peaks:
#             nxt = troughs[troughs > p]
#             if nxt.size: falling.append((p, nxt[0]))

#         # 5) count transitions
#         r_lt, r_hl = count_transitions(rising,   sq_sign)
#         f_lt, f_hl = count_transitions(falling,  sq_sign)

#         # 6) t-tests
#         t_r, p_r = ttest_ind(r_lt, r_hl, equal_var=False)
#         sig_r = 'Yes' if p_r < 0.05 else 'No'
#         t_f, p_f = ttest_ind(f_lt, f_hl, equal_var=False)
#         sig_f = 'Yes' if p_f < 0.05 else 'No'

#         # 7) plotting (unchanged)
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), tight_layout=True)
#         fig.suptitle(f"Lag = {lag} indices", fontsize=14)

#         ax1.hist(r_lt, bins='auto', alpha=0.5, color='blue',   label='–1→+1')
#         ax1.axvline(np.mean(r_lt), color='blue', linestyle='--')
#         ax1.hist(r_hl, bins='auto', alpha=0.5, color='orange', label='+1→–1')
#         ax1.axvline(np.mean(r_hl), color='orange', linestyle='--')
#         ax1.set_title(f"Rising phases\n t={t_r:.2f}, p={p_r:.2e}\nSig? {sig_r}")
#         ax1.set_xlabel('Transitions per rising phase')
#         ax1.set_ylabel('Count of phases')
#         ax1.legend()

#         ax2.hist(f_lt, bins='auto', alpha=0.5, color='blue',   label='–1→+1')
#         ax2.axvline(np.mean(f_lt), color='blue', linestyle='--')
#         ax2.hist(f_hl, bins='auto', alpha=0.5, color='orange', label='+1→–1')
#         ax2.axvline(np.mean(f_hl), color='orange', linestyle='--')
#         ax2.set_title(f"Decreasing phases\n t={t_f:.2f}, p={p_f:.2e}\nSig? {sig_f}")
#         ax2.set_xlabel('Transitions per decreasing phase')
#         ax2.legend()

#         plt.show()




