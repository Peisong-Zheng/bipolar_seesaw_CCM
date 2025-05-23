import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import groupby
from scipy.interpolate import interp1d
from scipy.stats import ttest_ind
from pyinform import transfer_entropy




# import numpy as np
# import matplotlib.pyplot as plt
# from pyinform import transfer_entropy

# def local_TE(
#     df_pre,
#     df_sq,
#     forcing_column='pre',
#     target_column='sq',
#     time_column='age',
#     nbins_pre=4,
#     nbins_target=2
# ):

#     # --- 1) Extract raw numpy arrays and reverse if needed
#     pre_raw = df_pre[forcing_column].values
#     sq_raw  = df_sq[target_column].values
#     # if you want to reverse time order, uncomment:
#     pre_raw = pre_raw[::-1]
#     sq_raw  = sq_raw[::-1]
    
#     # --- 2) Discretize
#     # 2a) forcing: histogram bins
#     bins_pre = np.histogram_bin_edges(pre_raw, bins=nbins_pre)
#     pre_disc = np.digitize(pre_raw, bins_pre) - 1

#     # 2b) target: binary threshold at zero
#     sq_disc = (sq_raw > 0).astype(int)
    
#     # --- 3) Compute local transfer entropy
#     local_te = transfer_entropy(pre_disc, sq_disc, k=1, local=True)


#     # --- 4) Set up the figure with two panels
#     fig = plt.figure(figsize=(12, 4))
#     gs  = fig.add_gridspec(2, 1, height_ratios=[0.8, 1.6], hspace=0)

#     # Top panel: pre & sq
#     ax0 = fig.add_subplot(gs[0])
#     t   = df_pre[time_column].values
#     t = t[::-1]  # reverse time order if needed
#     ln1 = ax0.plot(t, pre_raw, label=f'{forcing_column} (raw)')
#     ax0.set_ylabel(forcing_column)
#     ax0.tick_params(axis='x', which='both', labelbottom=False)
#     # reverse x axis
#     ax0.set_xlim(t[1], t[-1])

#     ax0b = ax0.twinx()
#     ln2 = ax0b.plot(t, sq_raw, color='C1', label=f'{target_column} (raw)')
#     ax0b.set_ylabel(target_column)
#     # reverse x 
    

#     # combine legends
#     lns = ln1 + ln2
#     labs = [l.get_label() for l in lns]
#     ax0.legend(lns, labs, loc='upper right')
    
#     # Bottom panel: local TE
#     ax1 = fig.add_subplot(gs[1], sharex=ax0)
#     # TE[0] is undefined (needs prior sample), so plot from index=1
#     ax1.plot(t[1:], local_te.T, color='C2', label='Local TE')
#     ax1.set_xlabel(time_column)
#     ax1.set_ylabel('Local TE (bits)')
#     ax1.legend(loc='upper right')

#     # set the 

#     # plt.tight_layout()
#     plt.show()

#     return local_te


# def analyze_state_space_TE(
#     df_pre,
#     df_sq,
#     forcing_column='pre',
#     target_column='sq',
#     nbins_pre=4,
#     nbins_sq=2,
#     use_quantile=False
# ):

#     # 1) Raw + reverse
#     pre = df_pre[forcing_column].values[::-1]
#     sq  = df_sq[target_column].values[::-1]

#     # 2a) Pre into nbins_pre: use linspace so max goes into last bin
#     x_min, x_max = pre.min(), pre.max()
#     if use_quantile:
#         bins_pre = np.quantile(pre, np.linspace(0, 1, nbins_pre+1))
#     else:
#         bins_pre = np.linspace(x_min, x_max+1e-10, nbins_pre+1)
#     # pre_disc = np.digitize(pre, bins_pre, right=True) - 1
#     pre_disc = np.digitize(pre, bins_pre, right=True) - 1
#     pre_disc = np.clip(pre_disc, 0, nbins_pre-1)

#     # 2b) Sq into nbins_sq
#     y_min, y_max = sq.min(), sq.max()
#     if use_quantile:
#         bins_sq  = np.quantile(sq, np.linspace(0, 1, nbins_sq+1))
#     else:
#         bins_sq = np.linspace(y_min, y_max+1e-10, nbins_sq+1)

#     sq_disc  = np.digitize(sq,  bins_sq,  right=True) - 1
#     sq_disc  = np.clip(sq_disc,  0, nbins_sq -1)

#     # 3) Local TE
#     local_te = transfer_entropy(pre_disc, sq_disc, k=1, local=True)

#     # 4) Align
#     te_arr = local_te.flatten()
#     N      = pre_disc.size
#     if te_arr.size == N:
#         te = te_arr[1:]
#     elif te_arr.size == N-1:
#         te = te_arr
#     else:
#         raise ValueError(f"Unexpected TE length {te_arr.shape}")

#     x_idx = pre_disc[:-1]
#     y_idx = sq_disc[:-1]

#     # 5) Clip any stray out-of-bounds indices
#     x_idx = np.clip(x_idx, 0, nbins_pre-1)
#     y_idx = np.clip(y_idx, 0, nbins_sq -1)

#     # 6) Build heatmap
#     heatmap = np.full((nbins_pre, nbins_sq), np.nan)
#     for i in range(nbins_pre):
#         for j in range(nbins_sq):
#             m = (x_idx == i) & (y_idx == j)
#             if m.any():
#                 heatmap[i, j] = te[m].mean()

#     # 7) Plot
#     plt.figure(figsize=(5, 3))
#     plt.imshow(heatmap.T, origin='lower', aspect='auto',
#                extent=[0, nbins_pre, 0, nbins_sq])
#     plt.colorbar(label='Average Local TE (bits)')
#     plt.xlabel(f'{forcing_column} bin (0…{nbins_pre-1})')
#     plt.ylabel(f'{target_column} bin (0…{nbins_sq-1})')
#     plt.title(f'State‐Space TE ({nbins_pre}×{nbins_sq} bins)')
#     plt.xticks(np.arange(0.5, nbins_pre+0.5), np.arange(nbins_pre))
#     plt.yticks(np.arange(0.5, nbins_sq+0.5), np.arange(nbins_sq))
#     plt.tight_layout()
#     plt.show()

#     # --- 6) Joint & marginal counts
#     # Clip indices just in case
#     x_clip = np.clip(x_idx, 0, nbins_pre-1)
#     y_clip = np.clip(y_idx, 0, nbins_sq-1)

#     joint_counts = np.zeros((nbins_pre, nbins_sq), int)
#     for xi, yi in zip(x_clip, y_clip):
#         joint_counts[xi, yi] += 1

#     pre_counts = np.bincount(x_clip, minlength=nbins_pre)
#     sq_counts  = np.bincount(y_clip, minlength=nbins_sq)

#     print("Joint counts (rows=pre_bin, cols=sq_bin):")
#     print(joint_counts)

#     print("\nMarginal counts for pre_bins:")
#     for i, c in enumerate(pre_counts):
#         print(f"  pre_bin {i}: {c}")

#     print("\nMarginal counts for sq_bins:")
#     for j, c in enumerate(sq_counts):
#         label = 'low' if j==0 else 'high'
#         print(f"  sq {label} ({j}): {c}")

#     # --- 7) Transition probabilities helper
#     z_idx = sq_disc[1:]  # next-state of sq

#     def transition_probs(i, j):
#         mask = (x_clip == i) & (y_clip == j)
#         counts = np.bincount(z_idx[mask], minlength=nbins_sq)
#         total = counts.sum()
#         return counts/total if total>0 else np.zeros(nbins_sq), total

#     # lowest-pre (0), lowest-sq (0)
#     probs_00, total_00 = transition_probs(0, 0)
#     print(f"\nCell (pre=low, sq=low): {total_00} samples")
#     print(f"  P(sq_next=0)={probs_00[0]:.3f}, P(sq_next=1)={probs_00[1]:.3f}")

#     # highest-pre (nbins_pre-1), lowest-sq (0)
#     hp = nbins_pre - 1
#     probs_h0, total_h0 = transition_probs(hp, 0)
#     print(f"\nCell (pre=high, sq=low): {total_h0} samples")
#     print(f"  P(sq_next=0)={probs_h0[0]:.3f}, P(sq_next=1)={probs_h0[1]:.3f}")

#     # Package results
#     return {
#         'heatmap': heatmap,
#         'joint_counts': joint_counts,
#         'pre_counts': pre_counts,
#         'sq_counts': sq_counts,
#         'trans_00': (probs_00, total_00),
#         'trans_high0': (probs_h0, total_h0),
#         'te': te,
#         'x_idx': x_idx,
#         'y_idx': y_idx
#     }



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
    q25   = np.quantile(pre_raw, 0.25)
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
























def plot_aic_delta(series):
    """
    Compute AIC for histogram bin counts B from 2 to 20,
    then plot the AIC and delta-AIC as separate figures.
    """
    series = np.asarray(series)
    x_min, x_max = series.min(), series.max()
    N = series.size

    b_values = np.arange(2, 21)
    aic_list = []

    for B in b_values:
        # Build histogram
        counts, _ = np.histogram(series, bins=B, range=(x_min, x_max))
        bin_width = (x_max - x_min) / B

        # Compute log-likelihood for nonzero bins
        positive = counts > 0
        ll = np.sum(counts[positive] * np.log(counts[positive] / (N * bin_width)))

        # Number of nonempty bins
        K_nonzero = np.count_nonzero(counts)

        # AIC: -2 * log-likelihood + 2 * (number of parameters)
        aic = -2 * ll + 2 * (K_nonzero - 1)
        aic_list.append(aic)

    aic_list = np.array(aic_list)
    delta_aic = np.diff(aic_list)

    # Plot AIC vs B
    plt.figure()
    plt.plot(b_values, aic_list)
    plt.xlabel('Number of bins B')
    plt.ylabel('AIC')
    plt.title('AIC vs Number of bins')
    plt.show()

    # Plot ΔAIC vs B
    plt.figure()
    plt.plot(b_values[1:], delta_aic)
    plt.xlabel('Number of bins B')
    plt.ylabel('ΔAIC')
    plt.title('Delta AIC vs Number of bins')
    plt.show()











import numpy as np
import matplotlib.pyplot as plt
from pyinform import transfer_entropy

def transfer_entropy_surrogate_test(
    forcing, sq, k=1,
    forcing_bins=4, sq_bins=2,
    n_surr=100, p=0.05, if_plot=True, dpi=100
):
    """
    Test for unidirectional causality using transfer entropy and surrogates.
    """
    # Reverse series
    x = np.asarray(forcing)[::-1]
    y = np.asarray(sq)[::-1]
    # Discretize
    xbins = np.histogram_bin_edges(x, bins=forcing_bins)
    # ybins = np.histogram_bin_edges(y, bins=sq_bins)
    # xbins = np.quantile(x, np.linspace(0, 1, forcing_bins))
    # xbins = np.quantile(x, np.linspace(0,1,forcing_bins+1))
    ybins = np.histogram_bin_edges(y, bins=sq_bins)

    x_disc = np.digitize(x, xbins) - 1
    y_disc = np.digitize(y, ybins) - 1
    
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







# import numpy as np
# import matplotlib.pyplot as plt
# from pyinform import transfer_entropy

# def transfer_entropy_surrogate_test(
#     forcing, sq, k=1,
#     forcing_bins=4, sq_bins=2,
#     n_surr=100, p=0.05, if_plot=True, dpi=100
# ):
#     """
#     Test for unidirectional causality using transfer entropy and surrogates.
    
#     Parameters
#     ----------
#     forcing : array-like
#         Time series of the driver (e.g., precession).
#     sq : array-like
#         Time series of the response (e.g., DO signal).
#     forcing_bins : int
#         Number of bins for discretizing 'forcing'.
#     sq_bins : int
#         Number of bins for discretizing 'sq'.
#     n_surr : int
#         Number of surrogate permutations.
#     p : float
#         Significance level threshold (e.g., 0.05).
#     if_plot : bool
#         Whether to plot null distributions and empirical TE lines.
    
#     Returns
#     -------
#     bool
#         True if TE(forcing→sq) is significant at level p and TE(sq→forcing) is not; otherwise False.
#     """
#     # Reverse series
#     x = np.asarray(forcing)[::-1]
#     y = np.asarray(sq)[::-1]
#     # Discretize
#     xbins = np.histogram_bin_edges(x, bins=forcing_bins)
#     ybins = np.histogram_bin_edges(y, bins=sq_bins)
#     x_disc = np.digitize(x, xbins) - 1
#     y_disc = np.digitize(y, ybins) - 1
    
#     # Empirical TE (one-step, history k=5)
#     te_xy = transfer_entropy(x_disc[:-1], y_disc[1:], k=k)
#     te_yx = transfer_entropy(y_disc[:-1], x_disc[1:], k=k)
    
#     # Surrogate nulls
#     null_xy = np.zeros(n_surr)
#     null_yx = np.zeros(n_surr)
#     for i in range(n_surr):
#         xs = np.random.permutation(x_disc)
#         null_xy[i] = transfer_entropy(xs[:-1], y_disc[1:], k=k)
#         ys = np.random.permutation(y_disc)
#         null_yx[i] = transfer_entropy(ys[:-1], x_disc[1:], k=k)
    
#     # p-values
#     p_xy = (np.sum(null_xy >= te_xy) + 1) / (n_surr + 1)
#     p_yx = (np.sum(null_yx >= te_yx) + 1) / (n_surr + 1)
    
#     # Plot if requested
#     if if_plot:
#         plt.figure(figsize=(6, 4), dpi=dpi)
#         # forcing -> sq
#         plt.hist(null_xy, bins=25, color='lightcoral', alpha=1, label='Null TE (forcing→sq)')
#         plt.axvline(te_xy, color='red', lw=2, label=f'Empirical TE (forcing→sq), p={p_xy:.3f}')
#         # sq -> forcing
#         plt.hist(null_yx, bins=25, color='lightblue', alpha=1, label='Null TE (sq→forcing)')
#         plt.axvline(te_yx, color='blue', lw=2, label=f'Empirical TE (sq→forcing), p={p_yx:.3f}')
#         plt.xlabel('Transfer Entropy (bits)')
#         plt.ylabel('Count')
#         # plt.title('Surrogate Test for Transfer Entropy')
#         plt.legend(loc='upper right', frameon=False)
#         plt.tight_layout()
#         plt.show()
    
#     # Determine unidirectional significance
#     sig_xy = p_xy < p
#     sig_yx = p_yx < p
#     return sig_xy and not sig_yx


















def build_DO_sq(df, column_name='none', age_start=0, age_end=641260, extra_sm=5,if_plot=False, dir='gt', metrics='F1'):
    """
    1) Interpolate the δ18O series to 10-yr steps
    2) Grid-search low-freq window & threshold to maximize F1
    3) Build ±1 square wave
    4) Reproduce the two-panel plot exactly as in your script

    Parameters
    ----------
    df : DataFrame
        Must have columns ['age', <δ18O column>]
    column_name : str
        Name of the δ18O column; if 'none', uses df.columns[1].
    if_plot : bool
        If True, shows the final two‐panel figure.
    dir : 'gt' | 'lt' | None
        If 'gt' or 'lt', only tests that direction; otherwise both.

    Returns
    -------
    new_df : DataFrame
        Interpolated δ18O with columns ['age', col].
    df_sq : DataFrame
        The square wave DataFrame with ['age','sq'].
    best : dict
        {'window','thr','dir','f1'} of the optimal detection.
    """
    # --- 0) pick δ18O column & sort ---
    col = df.columns[1] if column_name=='none' else column_name
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not in df")
    df = df[['age', col]].sort_values('age').reset_index(drop=True)

    # --- 1) interpolate to 10-yr grid ---
    # new_age = np.arange(0, 641260, 10)
    # if age_start >11703 or age_end <119140: through error 'Age does not cover all NGRIP DO sequences'

    if age_start > 11703 or age_end < 119140:
        raise ValueError(f"Age does not cover all NGRIP DO sequences")



    new_age = np.arange(age_start, age_end, 10)



    f_interp = interp1d(df['age'], df[col],
                        kind='nearest',
                        bounds_error=False,
                        fill_value=1)
    new_df = pd.DataFrame({'age': new_age,
                           col: f_interp(new_age)})

    # --- 1a) GS truth table & mask ---
    gs_data = {
        "start": [11703, 14692, 23020, 23340, 27780, 28900, 30840, 32500,
                  33740, 35480, 38220, 40160, 41460, 43340, 46860, 49280,
                  54220, 55000, 55800, 58040, 58280, 59080, 59440, 64100,
                  69620, 72340, 76440, 84760, 85060, 90040,
                  104040,104520,106750,108280,115370],
        "end":   [12896, 22900, 23220, 27540, 28600, 30600, 32040, 33360,
                  34740, 36580, 39900, 40800, 42240, 44280, 48340, 49600,
                  54900, 55400, 56500, 58160, 58560, 59300, 63840, 69400,
                  70380, 74100, 77760, 84960, 87600, 90140,
                  104380,105440,106900,110640,119140]
    }
    mask_full = np.zeros_like(new_df['age'], dtype=bool)
    for s,e in zip(gs_data['start'], gs_data['end']):
        mask_full |= (new_df['age'] >= s) & (new_df['age'] <= e)
    mask_crop = mask_full & (new_df['age'] < 120_000)

    # F1 helper
    def f1_score(pred, true):
        tp = np.sum(pred & true)
        fp = np.sum(pred & ~true)
        fn = np.sum(~pred & true)
        return 2*tp/(2*tp + fp + fn) if (2*tp + fp + fn)>0 else 0
    

    def precision_score(pred, true):
        tp = np.sum(pred & true)
        fp = np.sum(pred & ~true)
        return tp/(tp + fp) if (tp + fp) > 0 else 0

    # --- 2) grid-search ---
    window_range = np.arange(500, 6001, 100)
    if metrics == 'F1':
        best = {'window':None, 'thr':None, 'dir':None, 'f1':-1}
    elif metrics == 'precision':
        best = {'window':None, 'thr':None, 'dir':None, 'precision':-1}

    for w in window_range:
        smooth_lo   = new_df[col].rolling(window=w, center=True, min_periods=1).mean()
        anomaly     = new_df[col] - smooth_lo


        if extra_sm>0:
            anomaly_sm = anomaly.rolling(window=extra_sm, center=True, min_periods=1).mean()
        # anomaly_sm  = anomaly.rolling(window=50, center=True, min_periods=1).mean()
        if extra_sm == 0:
            anomaly_sm=anomaly

        vals = anomaly_sm[new_df['age']<120_000].values
        true = mask_crop[new_df['age']<120_000]

        thr_min, thr_max = vals.min(), vals.max()
        print(f"Window {w} → min={thr_min:.3f}, max={thr_max:.3f}")
        thr_cands = np.linspace(thr_min, thr_max, 101)
        dirs = ([dir] if dir in ('gt','lt') else ['gt','lt'])
        for direction in dirs:
            for thr in thr_cands:
                pred = (vals>thr) if direction=='gt' else (vals<thr)
                if metrics == 'F1':
                    score = f1_score(pred, true)
                    
                    if score>best['f1']:
                        best.update(window=w, thr=thr, dir=direction, f1=score)

                elif metrics == 'precision':
                    score = precision_score(pred, true)
                    if score > best['precision']:
                        best.update(window=w, thr=thr, dir=direction, precision=score)

    if metrics == 'F1':
        print(f"→ optimal window={best['window']}, "
            f"threshold {best['dir']} {best['thr']:.3f}, "
            f"max F1={best['f1']:.3f}")
        
    elif metrics == 'precision':
        print(f"→ optimal window={best['window']}, "
            f"threshold {best['dir']} {best['thr']:.3f}, "
            f"max precision={best['precision']:.3f}")

    # --- 3) build square wave on full record ---
    lo       = new_df[col].rolling(window=best['window'], 
                                   center=True, min_periods=1).mean()
    anom     = new_df[col] - lo


    if extra_sm>0:
        anom_sm = anom.rolling(window=extra_sm, center=True, min_periods=1).mean()

    if extra_sm == 0:
        anom_sm=anom

    if best['dir']=='gt':
        is_gs = anom_sm > best['thr']
    else:
        is_gs = anom_sm < best['thr']
    square_wave = np.where(is_gs, -1, 1)
    df_sq = pd.DataFrame({'age': new_df['age'], 'sq': square_wave})

    # --- safe segment building ---
    ages = new_df['age'].values
    segments = []
    for val, grp in groupby(enumerate(is_gs), key=lambda x:x[1]):
        if not val:
            continue
        g = list(grp)
        start_i, end_i = g[0][0], g[-1][0]
        segments.append((ages[start_i], ages[end_i]))

    # replace the second column of new_df with anom_sm
    new_df[col] = anom_sm

    # --- 4) final plot (same as your script) ---
    if if_plot:

        fig, ax = plt.subplots(figsize=(14, 4), dpi=300)

        # 1) official NGRIP stadials in light blue, label only once
        for i, (s, e) in enumerate(zip(gs_data['start'], gs_data['end'])):
            ax.axvspan(s, e, color='lightblue', alpha=1, label='NGRIP Stadials' if i==0 else None)

        # 2) your predicted stadials in orange, label only once
        for i, (s, e) in enumerate(segments):
            ax.axvspan(s, e, color='orange', alpha=0.3,
                    label=f"Stadials in {col}" if i==0 else None)

        # 3) plot the anomaly curve
        ax.plot(new_df['age'], anom_sm, color='green', label=f"{col} anomaly")
        if metrics == 'F1':
            ax.set_title(f"Optimal window={best['window']}, thr={best['thr']:.3f}, F1={best['f1']:.3f}")
        elif metrics == 'precision':
            ax.set_title(f"Optimal window={best['window']}, thr={best['thr']:.3f}, precision={best['precision']:.3f}")

        # # 4) threshold line
        # ax.axhline(best['thr'], color='black', linestyle='--',
        #         label=f"{'>' if best['dir']=='gt' else '<'} {best['thr']:.2f}")

        # 5) overlay square wave on twin‐y
        ax2 = ax.twinx()
        ax2.plot(new_df['age'], square_wave, drawstyle='steps-post',
                color='black', linewidth=1, label='GS square wave')
        ax2.set_ylim(-1.5, 1.5)
        ax2.set_ylabel('±1')

        # 6) zoom age axis to first 50 000 yr
        ax.set_xlim(0, 120000)

        # 7) build a combined legend
        lines, labels = [], []
        for a in (ax, ax2):
            l, lab = a.get_legend_handles_labels()
            lines += l
            labels += lab
        ax.legend(lines, labels, loc='upper right', ncol=2, fontsize='small')

        # 8) labels & title
        ax.set_xlabel('Age (years)')
        ax.set_ylabel(f"{col} anomaly")
        # ax.set_title("Zoomed (0–50 ka) with legend for vertical bands")

        plt.tight_layout()
        plt.show()


    return new_df, df_sq, best



















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



import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import ttest_ind

def plot_phase_durations(df_pre, df_sq, lags=range(0, 901, 100)):
    """
    For each lag in `lags`, shifts df_pre manually (introducing NaNs),
    truncates df_sq accordingly, computes rising/falling phase durations,
    runs t-tests, and produces the same two-panel figure per lag.
    
    Parameters
    ----------
    df_pre : pandas.DataFrame
        Must contain a 'pre' column (precession) and matching index.
    df_sq : pandas.DataFrame
        Must contain a 'sq' column (square wave ±1) and matching index.
    lags : iterable of ints, optional
        List of integer shifts (in data points) to test.
    """
    pre_vals_orig = df_pre['pre'].values
    sq_orig       = np.where(df_sq['sq'].values > 0, 1, -1)
    N             = len(pre_vals_orig)

    def count_transitions(intervals, sq_sign):
        lt, hl = [], []
        for i0, i1 in intervals:
            seg = sq_sign[i0:i1+1]
            lt.append(np.sum((seg[:-1]==-1) & (seg[1:]==+1)))
            hl.append(np.sum((seg[:-1]==+1) & (seg[1:]==-1)))
        return lt, hl

    for lag in lags:
        # 1) manual shift with NaNs
        pre_shifted = np.full(N, np.nan)
        if lag > 0:
            pre_shifted[0:N-lag] = pre_vals_orig[lag:N]
        elif lag < 0:
            k = -lag
            pre_shifted[k:N] = pre_vals_orig[0:N-k]
        else:
            pre_shifted[:] = pre_vals_orig

        # 2) truncate to non-NaNs
        valid    = ~np.isnan(pre_shifted)
        pre_vals = pre_shifted[valid]
        sq_sign  = sq_orig[valid]

        # 3) peak/trough detection
        peaks,   _ = find_peaks(pre_vals)
        troughs, _ = find_peaks(-pre_vals)

        # 4) rising & falling intervals
        rising, falling = [], []
        for t in troughs:
            nxt = peaks[peaks > t]
            if nxt.size: rising.append((t, nxt[0]))
        for p in peaks:
            nxt = troughs[troughs > p]
            if nxt.size: falling.append((p, nxt[0]))

        # 5) compute durations for each phase
        r_cold, r_warm = [], []
        for i0, i1 in rising:
            seg = sq_sign[i0:i1+1]
            r_cold.append(np.sum(seg == -1))
            r_warm.append(np.sum(seg == +1))

        d_cold, d_warm = [], []
        for i0, i1 in falling:
            seg = sq_sign[i0:i1+1]
            d_cold.append(np.sum(seg == -1))
            d_warm.append(np.sum(seg == +1))

        # 6) t-tests on durations
        t_r, p_r = ttest_ind(r_cold, r_warm, equal_var=False)
        sig_r = 'Yes' if p_r < 0.05 else 'No'
        t_d, p_d = ttest_ind(d_cold, d_warm, equal_var=False)
        sig_d = 'Yes' if p_d < 0.05 else 'No'

        # 7) plotting (identical to original)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4), tight_layout=True)
        fig.suptitle(f"Lag = {lag} indices", fontsize=14)

        ax1.hist(r_cold, bins='auto', alpha=0.5, color='green', label='cold length')
        ax1.axvline(np.mean(r_cold), color='green', linestyle='dashed', linewidth=1)
        ax1.hist(r_warm, bins='auto', alpha=0.5, color='red',   label='warm length')
        ax1.axvline(np.mean(r_warm), color='red', linestyle='dashed', linewidth=1)
        ax1.set_title(f"Rising phases\n t={t_r:.2f}, p={p_r:.2e}\nSig? {sig_r}")
        ax1.set_xlabel('Duration (data points)')
        ax1.set_ylabel('Frequency')
        ax1.legend()

        ax2.hist(d_cold, bins='auto', alpha=0.5, color='green', label='cold length')
        ax2.axvline(np.mean(d_cold), color='green', linestyle='dashed', linewidth=1)
        ax2.hist(d_warm, bins='auto', alpha=0.5, color='red',   label='warm length')
        ax2.axvline(np.mean(d_warm), color='red', linestyle='dashed', linewidth=1)
        ax2.set_title(f"Decreasing phases\n t={t_d:.2f}, p={p_d:.2e}\nSig? {sig_d}")
        ax2.set_xlabel('Duration (data points)')
        ax2.legend()

        plt.show()



import numpy as np
from scipy.signal import find_peaks
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

def plot_transition_distribution(df_pre, df_sq, lags=None):
    """
    For each lag in `lags`, shifts df_pre manually (introducing NaNs),
    truncates df_sq accordingly, computes rising/falling phase transitions,
    runs t-tests, and produces the same two-panel figure per lag.

    Parameters
    ----------
    df_pre : pandas.DataFrame
        Must contain a 'pre' column.
    df_sq : pandas.DataFrame
        Must contain a 'sq' column with values ±1.
    lags : iterable of ints, optional
        List of integer shifts (in data points) to test.
        Default is range(0, 901, 100).
    """
    if lags is None:
        lags = range(0, 901, 100)

    pre_vals_orig = df_pre['pre'].values
    sq_orig       = np.where(df_sq['sq'].values > 0, 1, -1)
    N             = len(pre_vals_orig)

    def count_transitions(intervals, sq_sign):
        lt, hl = [], []
        for i0, i1 in intervals:
            seg = sq_sign[i0:i1+1]
            lt.append(np.sum((seg[:-1]==-1) & (seg[1:]==+1)))
            hl.append(np.sum((seg[:-1]==+1) & (seg[1:]==-1)))
        return lt, hl

    for lag in lags:
        # 1) manual shift with NaNs instead of roll()
        pre_shifted = np.full(N, np.nan)
        if lag > 0:
            pre_shifted[0:N-lag] = pre_vals_orig[lag:N]
        elif lag < 0:
            k = -lag
            pre_shifted[k:N] = pre_vals_orig[0:N-k]
        else:
            pre_shifted[:] = pre_vals_orig

        # 2) mask out NaNs & truncate sq to match
        valid    = ~np.isnan(pre_shifted)
        pre_vals = pre_shifted[valid]
        sq_sign  = sq_orig[valid]

        # 3) detect peaks/troughs
        peaks,   _ = find_peaks(pre_vals)
        troughs, _ = find_peaks(-pre_vals)

        # 4) rising & falling intervals
        rising, falling = [], []
        for t in troughs:
            nxt = peaks[peaks > t]
            if nxt.size: rising.append((t, nxt[0]))
        for p in peaks:
            nxt = troughs[troughs > p]
            if nxt.size: falling.append((p, nxt[0]))

        # 5) count transitions
        r_lt, r_hl = count_transitions(rising,   sq_sign)
        f_lt, f_hl = count_transitions(falling,  sq_sign)

        # 6) t-tests
        t_r, p_r = ttest_ind(r_lt, r_hl, equal_var=False)
        sig_r = 'Yes' if p_r < 0.05 else 'No'
        t_f, p_f = ttest_ind(f_lt, f_hl, equal_var=False)
        sig_f = 'Yes' if p_f < 0.05 else 'No'

        # 7) plotting (unchanged)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), tight_layout=True)
        fig.suptitle(f"Lag = {lag} indices", fontsize=14)

        ax1.hist(r_lt, bins='auto', alpha=0.5, color='blue',   label='–1→+1')
        ax1.axvline(np.mean(r_lt), color='blue', linestyle='--')
        ax1.hist(r_hl, bins='auto', alpha=0.5, color='orange', label='+1→–1')
        ax1.axvline(np.mean(r_hl), color='orange', linestyle='--')
        ax1.set_title(f"Rising phases\n t={t_r:.2f}, p={p_r:.2e}\nSig? {sig_r}")
        ax1.set_xlabel('Transitions per rising phase')
        ax1.set_ylabel('Count of phases')
        ax1.legend()

        ax2.hist(f_lt, bins='auto', alpha=0.5, color='blue',   label='–1→+1')
        ax2.axvline(np.mean(f_lt), color='blue', linestyle='--')
        ax2.hist(f_hl, bins='auto', alpha=0.5, color='orange', label='+1→–1')
        ax2.axvline(np.mean(f_hl), color='orange', linestyle='--')
        ax2.set_title(f"Decreasing phases\n t={t_f:.2f}, p={p_f:.2e}\nSig? {sig_f}")
        ax2.set_xlabel('Transitions per decreasing phase')
        ax2.legend()

        plt.show()




