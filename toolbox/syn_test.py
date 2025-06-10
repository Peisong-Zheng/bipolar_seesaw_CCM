import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

def perturb_sequence_v2(df, n_events=10, min_len=1, max_len=10, trans_shift=2,
                     random_state=None, plot=True):
    """
    Perturb a binary stadial/interstadial sequence by:
      1) shifting each true transition ±2 points (no crossings)
      2) inserting N random spurious events

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns 'age' and 'sq' (0 or 1).
    n_events : int
        Number of random segments to insert.
    min_len, max_len : int
        Min/max length (in samples) of each inserted segment.
    random_state : int or None
        Seed for reproducibility.
    plot : bool
        If True, overlays original vs perturbed sequences.

    Returns
    -------
    df_pert : pandas.DataFrame
        Copy of `df` with new column 'sq_pert' for the perturbed sequence.
    f1 : float
        Macro-averaged F₁ score comparing original 'sq' to 'sq_pert'.
    """
    rng = np.random.RandomState(random_state)
    seq_orig = df['sq'].values
    L = len(seq_orig)

    # --- 1) SHIFT TRUE TRANSITIONS ---
    # find original transition points (where value changes)
    trans_idx = np.where(np.diff(seq_orig) != 0)[0] + 1  # positions in [1..L-1]
    # build original boundaries: start=0, each transition, end=L
    orig_bounds = np.concatenate(([0], trans_idx, [L]))
    new_bounds = orig_bounds.copy()

    # jiggle each interior boundary by ±trans_shift, constrained so they don't cross
    for i in range(1, len(orig_bounds) - 1):
        lower = new_bounds[i - 1] + 1
        upper = new_bounds[i + 1] - 1
        shift = rng.choice([-trans_shift, trans_shift])
        nb = orig_bounds[i] + shift
        # clamp within [lower, upper]
        new_bounds[i] = np.clip(nb, lower, upper)

    # rebuild a shifted sequence
    seq_shifted = np.empty_like(seq_orig)
    for j in range(len(new_bounds) - 1):
        start, end = new_bounds[j], new_bounds[j + 1]
        # state comes from the original segment j
        state = seq_orig[orig_bounds[j]]
        seq_shifted[start:end] = state

    # --- 2) INSERT N SPURIOUS EVENTS ---
    seq_pert = seq_shifted.copy()
    for _ in range(n_events):
        length = rng.randint(min_len, max_len + 1)
        start = rng.randint(0, L - length)
        new_state = rng.choice([0, 1])
        seq_pert[start:start + length] = new_state

    # compute F1 against the true original
    f1 = f1_score(seq_orig, seq_pert, average='macro')

    # package into DataFrame
    df_pert = df.copy()
    df_pert['sq'] = seq_pert

    # optional plot
    if plot:
        plt.figure(figsize=(12, 3))
        plt.step(df['age'], seq_orig,      where='post',
                 label='Original', linewidth=1)
        # plt.step(df['age'], seq_shifted,   where='post',
        #          label='Shifted transitions', linestyle='--', alpha=0.7)
        plt.step(df['age'], seq_pert,      where='post',
                 label='Perturbed', alpha=0.7, linewidth=1)
        plt.xlabel('Age')
        plt.ylabel('sq')
        plt.title(f'Original vs Shifted vs Perturbed (F₁₍macro₎ = {f1:.3f})')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

    return df_pert, f1



def perturb_sequence_random_erase(df, n_events=10, min_len=1, max_len=10, trans_shift=2, erase_portion=0.1,
                     random_state=None, plot=True):
    """
    Perturb a binary stadial/interstadial sequence by:
      1) shifting each true transition ±2 points (no crossings)
      2) inserting N random spurious events

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns 'age' and 'sq' (0 or 1).
    n_events : int
        Number of random segments to insert.
    min_len, max_len : int
        Min/max length (in samples) of each inserted segment.
    random_state : int or None
        Seed for reproducibility.
    plot : bool
        If True, overlays original vs perturbed sequences.

    Returns
    -------
    df_pert : pandas.DataFrame
        Copy of `df` with new column 'sq_pert' for the perturbed sequence.
    f1 : float
        Macro-averaged F₁ score comparing original 'sq' to 'sq_pert'.
    """
    rng = np.random.RandomState(random_state)
    seq_orig = df['sq'].values
    L = len(seq_orig)

    # --- 1) SHIFT TRUE TRANSITIONS ---
    # find original transition points (where value changes)
    trans_idx = np.where(np.diff(seq_orig) != 0)[0] + 1  # positions in [1..L-1]
    # build original boundaries: start=0, each transition, end=L
    orig_bounds = np.concatenate(([0], trans_idx, [L]))
    new_bounds = orig_bounds.copy()



    # jiggle each interior boundary by ±trans_shift, constrained so they don't cross
    for i in range(1, len(orig_bounds) - 1):
        lower = new_bounds[i - 1] + 1
        upper = new_bounds[i + 1] - 1
        shift = rng.choice([-trans_shift, trans_shift])
        nb = orig_bounds[i] + shift
        # clamp within [lower, upper]
        new_bounds[i] = np.clip(nb, lower, upper)

    # rebuild a shifted sequence
    seq_shifted = np.empty_like(seq_orig)
    for j in range(len(new_bounds) - 1):
        start, end = new_bounds[j], new_bounds[j + 1]
        # state comes from the original segment j
        state = seq_orig[orig_bounds[j]]
        seq_shifted[start:end] = state

    # # --- 2) ERASE A PORTION OF TRUE TRANSITIONS ---
    # # find transition points in the shifted sequence
    # trans_shifted = np.where(np.diff(seq_shifted) != 0)[0] + 1
    # n_trans_orig  = len(trans_shifted)
    
    # # build segment boundaries: [0, t1, t2, ..., L]
    # seg_bounds    = np.concatenate(([0], trans_shifted, [L]))
    # seq_erased    = seq_shifted.copy()
    
    # erased_count = 0
    # # for each transition at index t = trans_shifted[k]:
    # #   with probability erase_portion, set seq_erased[t:next_bound] 
    # #   to the state just before the transition
    # for k, t in enumerate(trans_shifted):
    #     if rng.random_sample() < erase_portion:
    #         next_bound = seg_bounds[k+2]
    #         prev_state = seq_shifted[t-1]
    #         seq_erased[t:next_bound] = prev_state
    #         erased_count += 1
    
    # # count how many transitions remain
    # trans_erased   = np.where(np.diff(seq_erased) != 0)[0] + 1
    # n_trans_remain = len(trans_erased)
    
    # print(f"Erased {erased_count}/{n_trans_orig} transitions "
    #       f"({100*erased_count/n_trans_orig:.1f}%), "
    #       f"{n_trans_remain} remain.")

    # 2) ERASE A PORTION OF TRUE TRANSITIONS (non‐overlapping)
    trans_shifted = np.where(np.diff(seq_shifted) != 0)[0] + 1
    n_trans       = len(trans_shifted)
    seg_bounds    = np.concatenate(([0], trans_shifted, [L]))



    # if erase_portion <= 0 or >=0.5 thow a ValueError
    if erase_portion <= 0 or erase_portion >= 0.5:
        raise ValueError("erase_portion must be in (0, 0.5]")
    


    # how many to remove
    n_remove = int(round(erase_portion * n_trans))
    # pick that many UNIQUE transitions
    erase_idxs = rng.choice(n_trans, size=n_remove, replace=False)

    seq_erased = seq_shifted.copy()
    for k in sorted(erase_idxs):
        t = trans_shifted[k]
        # segment runs from t to seg_bounds[k+2]
        next_bound = seg_bounds[k+2]
        prev_state = seq_erased[t-1]
        seq_erased[t:next_bound] = prev_state

    # now recompute exactly how many remain
    trans_erased   = np.where(np.diff(seq_erased) != 0)[0] + 1
    n_remain       = len(trans_erased)

    print(f"Requested erase_portion={erase_portion:.2f}: "
          f"Removed {n_remove}/{n_trans} transitions, "
          f"{n_remain} remain.")
    
    # now carry on with spurious insertions using seq_erased
    seq_pert = seq_erased.copy()
    for _ in range(n_events):
        length = rng.randint(min_len, max_len + 1)
        start = rng.randint(0, L - length)
        new_state = rng.choice([0, 1])
        seq_pert[start:start + length] = new_state

    # compute F1 against the true original
    f1 = f1_score(seq_orig, seq_pert, average='macro')

    # package into DataFrame
    df_pert = df.copy()
    df_pert['sq'] = seq_pert

    # # optional plot
    # if plot:
    #     plt.figure(figsize=(12, 3))
    #     # plt.step(df['age'], seq_orig,      where='post',
    #     #          label='Original', linewidth=1)
    #     plt.step(df['age'], seq_erased,   where='post',
    #              label='Shifted + erased transitions', linestyle='-', alpha=0.7)
    #     # plt.step(df['age'], seq_pert,      where='post',
    #     #          label='Perturbed', alpha=0.7, linewidth=1)
    #     plt.xlabel('Age')
    #     plt.ylabel('sq')
    #     plt.title(f'Original vs Shifted vs Perturbed (F₁₍macro₎ = {f1:.3f})')
    #     plt.legend(loc='upper right')
    #     plt.tight_layout()
    #     plt.show()

    if plot:
        plt.figure(figsize=(12, 3))
        # plt.step(df['age'], seq_orig,      where='post', label='Original', linewidth=1)

        plt.step(df['age'], seq_erased,    where='post',
                 label=f'Erased {erase_portion*100:.0f}%', linewidth=1)
        # plt.step(df['age'], seq_pert,      where='post',
        #      label='Perturbed', alpha=0.7, linewidth=1)
        plt.xlabel('Age')
        plt.ylabel('sq')
        plt.title(f'Perturbation (F₁₍macro₎ = {f1:.3f})')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

    return df_pert, f1